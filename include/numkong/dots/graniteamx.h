/**
 *  @brief SIMD-accelerated Batched Dot Products for Granite Rapids.
 *  @file include/numkong/dots/graniteamx.h
 *  @author Ash Vardanian
 *  @date April 9, 2026
 *
 *  @sa include/numkong/dots.h
 *
 *  This file contains tiled matrix-multiplication kernels optimized for Intel AMX-FP16 instructions
 *  on Intel Granite Rapids CPUs. Two kernel families are provided:
 *
 *  1. Native FP16 GEMM: FP16×FP16→FP32 via TDPFP16PS, same tile geometry as BF16 (16×32 = 1KB).
 *  2. F32 Ozaki 2-term GEMM: splits F32 inputs into two FP16 halves (high + low), performs
 *     4 TDPFP16PS operations per depth tile, achieving ~35-40 bit effective mantissa precision.
 *     Outputs to F64, solidly between F32 (23-bit) and F64 (52-bit) precision.
 *
 *  FP16 tiles: 16 rows × 32 elements = 512 FP16 values = 1KB per tile (same as BF16).
 *
 *  Tile register allocation (same 2×2 output blocking as sapphireamx):
 *
 *  - TMM0, TMM1: A matrix tiles (row blocks i and i+16, or high/low halves for Ozaki)
 *  - TMM2, TMM3: B matrix tiles (column blocks j and j+16, or high/low halves for Ozaki)
 *  - TMM4-7: C accumulator tiles (2 × 2 output grid)
 *
 *  @section amx_fp16_instructions Intel AMX-FP16 Instructions (Granite Rapids+)
 *
 *  FP16 matrix multiply (AMX-FP16):
 *
 *      Intrinsic                   Instruction                     Operation
 *      _tile_dpfp16ps              TDPFP16PS (TMM, TMM, TMM)       C += A × B (fp16 → f32)
 *
 *  TDPFP16PS: 16 × 16 × 32 = 8192 FP16 MACs per instruction (same throughput as TDPBF16PS).
 */
#ifndef NK_DOTS_GRANITEAMX_H
#define NK_DOTS_GRANITEAMX_H

#if NK_TARGET_X8664_
#if NK_TARGET_GRANITEAMX

#include "numkong/dots/serial.h"
#include "numkong/dots/sapphireamx.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(                                                                                                     \
    __attribute__((target(                                                                                                        \
        "avx2,avx512f,avx512vl,avx512bw,avx512dq,avx512fp16,avx512vbmi,f16c,fma,bmi,bmi2,amx-tile,amx-bf16,amx-int8,amx-fp16"))), \
    apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "avx512bw", "avx512dq", "avx512fp16", "avx512vbmi", "f16c", "fma", \
                   "bmi", "bmi2", "amx-tile", "amx-bf16", "amx-int8", "amx-fp16")
#endif

#pragma region Tile Types

typedef struct {
    NK_ALIGN64 nk_f16_t data[16][32]; // 16 rows × 32 columns = 1KB
} nk_dots_f16_a16x32_graniteamx_t;

typedef struct {
    NK_ALIGN64 nk_f16_t data[16][16][2]; // 16 depth-groups × 16 columns × 2 = 1KB (pair-interleaved)
} nk_dots_f16_b32x16_graniteamx_t;

typedef struct {
    NK_ALIGN64 nk_f32_t data[16][16]; // 16 × 16 = 1KB accumulator
} nk_dots_f16_state_graniteamx_t;

typedef struct {
    nk_dots_f16_state_graniteamx_t c[2][2]; // 4KB total (2×2 output blocking)
} nk_dots_f16_state2x2_graniteamx_t;

// Extended packed header for F32 Ozaki: stores offsets to low-half tiles and per-tile scale factors
typedef struct {
    nk_u32_t full_column_tiles;
    nk_u32_t full_depth_tiles;
    nk_u32_t column_remainder_count;
    nk_u32_t column_edge_offset;
    nk_u32_t norms_byte_offset;
    nk_u32_t b_low_tiles_offset; // byte offset to B_low tile region
    nk_u32_t scales_offset;      // byte offset to per-column-tile inverse scale factors
    nk_u32_t reserved[9];        // padding to 64 bytes
} nk_dots_f32ozaki_packed_header_graniteamx_t;

#pragma endregion Tile Types

#pragma region Helpers

/* Initialize FP16 output state to zero */
NK_INTERNAL void nk_dots_f16_init_graniteamx_(nk_dots_f16_state_graniteamx_t *state) {
    __m512 zero_f32x16 = _mm512_setzero_ps();
    for (nk_size_t row_idx = 0; row_idx < 16; row_idx++) { _mm512_store_ps(state->data[row_idx], zero_f32x16); }
}

/* Load A tile from FP16 row-major source with masking for edge tiles */
NK_INTERNAL void nk_dots_f16_load_a_graniteamx_(        //
    nk_dots_f16_a16x32_graniteamx_t *a_tile,            //
    nk_f16_t const *src, nk_size_t src_stride_elements, //
    nk_size_t valid_rows, nk_size_t valid_cols) {

    __mmask32 column_mask = (valid_cols >= 32) ? 0xFFFFFFFF : ((__mmask32)1 << valid_cols) - 1;
    __m512i zero_i16x32 = _mm512_setzero_si512();

    for (nk_size_t row_idx = 0; row_idx < 16; row_idx++) {
        if (row_idx < valid_rows) {
            __m512i row_i16x32 = _mm512_maskz_loadu_epi16(column_mask, src + row_idx * src_stride_elements);
            _mm512_store_si512((__m512i *)a_tile->data[row_idx], row_i16x32);
        }
        else { _mm512_store_si512((__m512i *)a_tile->data[row_idx], zero_i16x32); }
    }
    nk_compiler_barrier_sapphireamx_();
}

/* Store F32 state to output matrix with masking for edge tiles */
NK_INTERNAL void nk_dots_f16_store_graniteamx_(   //
    nk_dots_f16_state_graniteamx_t const *state,  //
    nk_f32_t *dst, nk_size_t dst_stride_elements, //
    nk_size_t valid_rows, nk_size_t valid_cols) {

    __mmask16 column_mask = (valid_cols >= 16) ? 0xFFFF : ((__mmask16)1 << valid_cols) - 1;

    for (nk_size_t row_idx = 0; row_idx < valid_rows; row_idx++) {
        __m512 row_f32x16 = _mm512_load_ps(state->data[row_idx]);
        _mm512_mask_storeu_ps(dst + row_idx * dst_stride_elements, column_mask, row_f32x16);
    }
}

/* Store F32 state widened to F64 output with masking for edge tiles */
NK_INTERNAL void nk_dots_f32ozaki_store_f64_graniteamx_( //
    nk_dots_f16_state_graniteamx_t const *state,         //
    nk_f64_t *dst, nk_size_t dst_stride_elements,        //
    nk_size_t valid_rows, nk_size_t valid_cols) {

    for (nk_size_t row_idx = 0; row_idx < valid_rows; row_idx++) {
        __m512 accumulator_f32x16 = _mm512_load_ps(state->data[row_idx]);
        __m512d lower_f64x8 = _mm512_cvtps_pd(_mm512_castps512_ps256(accumulator_f32x16));
        __m512d upper_f64x8 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(accumulator_f32x16, 1));

        if (valid_cols >= 16) {
            _mm512_storeu_pd(dst + row_idx * dst_stride_elements, lower_f64x8);
            _mm512_storeu_pd(dst + row_idx * dst_stride_elements + 8, upper_f64x8);
        }
        else if (valid_cols > 8) {
            _mm512_storeu_pd(dst + row_idx * dst_stride_elements, lower_f64x8);
            __mmask8 upper_mask = (__mmask8)((1u << (valid_cols - 8)) - 1);
            _mm512_mask_storeu_pd(dst + row_idx * dst_stride_elements + 8, upper_mask, upper_f64x8);
        }
        else {
            __mmask8 lower_mask = (__mmask8)((1u << valid_cols) - 1);
            _mm512_mask_storeu_pd(dst + row_idx * dst_stride_elements, lower_mask, lower_f64x8);
        }
    }
}

/* Store 2x2 F32 state to output matrix with masking for edge tiles */
NK_INTERNAL void nk_dots_f16_output2x2_graniteamx_( //
    nk_dots_f16_state2x2_graniteamx_t const *state, //
    nk_f32_t *dst, nk_size_t dst_stride_elements,   //
    nk_size_t valid_rows, nk_size_t valid_cols) {

    nk_size_t const rows_high = (valid_rows > 16) ? 16 : valid_rows;
    nk_size_t const cols_left = (valid_cols > 16) ? 16 : valid_cols;
    nk_size_t const cols_right = (valid_cols > 16) ? valid_cols - 16 : 0;

    if (rows_high > 0 && cols_left > 0)
        nk_dots_f16_store_graniteamx_(&state->c[0][0], dst, dst_stride_elements, rows_high, cols_left);
    if (rows_high > 0 && cols_right > 0)
        nk_dots_f16_store_graniteamx_(&state->c[0][1], dst + 16, dst_stride_elements, rows_high, cols_right);

    if (valid_rows > 16) {
        nk_size_t const rows_low = valid_rows - 16;
        nk_f32_t *dst_low = dst + 16 * dst_stride_elements;
        if (cols_left > 0)
            nk_dots_f16_store_graniteamx_(&state->c[1][0], dst_low, dst_stride_elements, rows_low, cols_left);
        if (cols_right > 0)
            nk_dots_f16_store_graniteamx_(&state->c[1][1], dst_low + 16, dst_stride_elements, rows_low, cols_right);
    }
}

/* Store 2x2 F32 state widened to F64 output with masking for edge tiles */
NK_INTERNAL void nk_dots_f32ozaki_output2x2_f64_graniteamx_( //
    nk_dots_f16_state2x2_graniteamx_t const *state,          //
    nk_f64_t *dst, nk_size_t dst_stride_elements,            //
    nk_size_t valid_rows, nk_size_t valid_cols) {

    nk_size_t const rows_high = (valid_rows > 16) ? 16 : valid_rows;
    nk_size_t const cols_left = (valid_cols > 16) ? 16 : valid_cols;
    nk_size_t const cols_right = (valid_cols > 16) ? valid_cols - 16 : 0;

    if (rows_high > 0 && cols_left > 0)
        nk_dots_f32ozaki_store_f64_graniteamx_(&state->c[0][0], dst, dst_stride_elements, rows_high, cols_left);
    if (rows_high > 0 && cols_right > 0)
        nk_dots_f32ozaki_store_f64_graniteamx_(&state->c[0][1], dst + 16, dst_stride_elements, rows_high, cols_right);

    if (valid_rows > 16) {
        nk_size_t const rows_low = valid_rows - 16;
        nk_f64_t *dst_low = dst + 16 * dst_stride_elements;
        if (cols_left > 0)
            nk_dots_f32ozaki_store_f64_graniteamx_(&state->c[1][0], dst_low, dst_stride_elements, rows_low, cols_left);
        if (cols_right > 0)
            nk_dots_f32ozaki_store_f64_graniteamx_(&state->c[1][1], dst_low + 16, dst_stride_elements, rows_low,
                                                   cols_right);
    }
}

/* Accumulate 3 A × B tile pairs into state using AMX TDPFP16PS */
NK_INTERNAL void nk_dots_f16_update_graniteamx_(     //
    nk_dots_f16_state_graniteamx_t *state,           //
    nk_dots_f16_a16x32_graniteamx_t const *a_tile_0, //
    nk_dots_f16_a16x32_graniteamx_t const *a_tile_1, //
    nk_dots_f16_a16x32_graniteamx_t const *a_tile_2, //
    nk_dots_f16_b32x16_graniteamx_t const *b_tile_0, //
    nk_dots_f16_b32x16_graniteamx_t const *b_tile_1, //
    nk_dots_f16_b32x16_graniteamx_t const *b_tile_2) {

    _tile_loadd(0, state->data, 64);
    _tile_loadd(1, a_tile_0->data, 64);
    _tile_loadd(2, a_tile_1->data, 64);
    _tile_loadd(3, a_tile_2->data, 64);
    _tile_loadd(4, b_tile_0->data, 64);
    _tile_loadd(5, b_tile_1->data, 64);
    _tile_loadd(6, b_tile_2->data, 64);

    _tile_dpfp16ps(0, 1, 4);
    _tile_dpfp16ps(0, 2, 5);
    _tile_dpfp16ps(0, 3, 6);

    _tile_stored(0, state->data, 64);
}

/* Accumulate A × B Ozaki terms into state for symmetric kernel (1-deep depth grouping).
 * Performs 4 TDPFP16PS: high×high + high×low + low×high + low×low.
 * Uses TMM0=C, TMM1=A_high, TMM2=A_low, TMM3=B_high, TMM4=B_low. */
NK_INTERNAL void nk_dots_f32ozaki_update_graniteamx_(   //
    nk_dots_f16_state_graniteamx_t *state,              //
    nk_dots_f16_a16x32_graniteamx_t const *a_high_tile, //
    nk_dots_f16_a16x32_graniteamx_t const *a_low_tile,  //
    nk_dots_f16_b32x16_graniteamx_t const *b_high_tile, //
    nk_dots_f16_b32x16_graniteamx_t const *b_low_tile) {

    _tile_loadd(0, state->data, 64);
    _tile_loadd(1, a_high_tile->data, 64);
    _tile_loadd(2, a_low_tile->data, 64);
    _tile_loadd(3, b_high_tile->data, 64);
    _tile_loadd(4, b_low_tile->data, 64);

    _tile_dpfp16ps(0, 1, 3); // C += A_high × B_high
    _tile_dpfp16ps(0, 1, 4); // C += A_high × B_low
    _tile_dpfp16ps(0, 2, 3); // C += A_low  × B_high
    _tile_dpfp16ps(0, 2, 4); // C += A_low  × B_low

    _tile_stored(0, state->data, 64);
}

/* Split a row of 32 F32 values into high and low FP16 halves (vectorized).
 * The high half captures the top ~10 mantissa bits, the low half captures the residual ~13 bits.
 * Both halves are stored as one tile row (32 FP16 = 64 bytes). */
NK_INTERNAL void nk_dots_f32ozaki_split_row_graniteamx_( //
    nk_f32_t const *src, nk_size_t valid_cols,           //
    nk_f16_t *dst_high, nk_f16_t *dst_low) {

    // First 16 F32 elements
    nk_size_t const first_count = (valid_cols > 16) ? 16 : valid_cols;
    __mmask16 first_mask = (first_count >= 16) ? 0xFFFF : ((__mmask16)1 << first_count) - 1;
    __m512 original_first_f32x16 = _mm512_maskz_loadu_ps(first_mask, src);
    __m256i high_first_f16x16 = _mm512_cvtps_ph(original_first_f32x16, _MM_FROUND_TO_NEAREST_INT);
    __m512 high_roundtrip_first_f32x16 = _mm512_cvtph_ps(high_first_f16x16);
    __m512 residual_first_f32x16 = _mm512_sub_ps(original_first_f32x16, high_roundtrip_first_f32x16);
    __m256i low_first_f16x16 = _mm512_cvtps_ph(residual_first_f32x16, _MM_FROUND_TO_NEAREST_INT);

    // Second 16 F32 elements
    __m256i high_second_f16x16, low_second_f16x16;
    if (valid_cols > 16) {
        nk_size_t const second_count = valid_cols - 16;
        __mmask16 second_mask = (second_count >= 16) ? 0xFFFF : ((__mmask16)1 << second_count) - 1;
        __m512 original_second_f32x16 = _mm512_maskz_loadu_ps(second_mask, src + 16);
        high_second_f16x16 = _mm512_cvtps_ph(original_second_f32x16, _MM_FROUND_TO_NEAREST_INT);
        __m512 high_roundtrip_second_f32x16 = _mm512_cvtph_ps(high_second_f16x16);
        __m512 residual_second_f32x16 = _mm512_sub_ps(original_second_f32x16, high_roundtrip_second_f32x16);
        low_second_f16x16 = _mm512_cvtps_ph(residual_second_f32x16, _MM_FROUND_TO_NEAREST_INT);
    }
    else {
        high_second_f16x16 = _mm256_setzero_si256();
        low_second_f16x16 = _mm256_setzero_si256();
    }

    // Combine into 32-wide rows: [first_16 | second_16]
    __m512i high_row_i16x32 = _mm512_inserti64x4(_mm512_castsi256_si512(high_first_f16x16), high_second_f16x16, 1);
    __m512i low_row_i16x32 = _mm512_inserti64x4(_mm512_castsi256_si512(low_first_f16x16), low_second_f16x16, 1);
    _mm512_store_si512((__m512i *)dst_high, high_row_i16x32);
    _mm512_store_si512((__m512i *)dst_low, low_row_i16x32);
}

/* Split a row of 32 F32 values into high and low FP16 halves with per-element scaling.
 * Pre-multiplies each F32 value by scale_f32x16 before splitting. */
NK_INTERNAL void nk_dots_f32ozaki_split_row_scaled_graniteamx_( //
    nk_f32_t const *src, nk_size_t valid_cols,                  //
    __m512 scale_first_f32x16, __m512 scale_second_f32x16,      //
    nk_f16_t *dst_high, nk_f16_t *dst_low) {

    // First 16 F32 elements
    nk_size_t const first_count = (valid_cols > 16) ? 16 : valid_cols;
    __mmask16 first_mask = (first_count >= 16) ? 0xFFFF : ((__mmask16)1 << first_count) - 1;
    __m512 original_first_f32x16 = _mm512_mul_ps(_mm512_maskz_loadu_ps(first_mask, src), scale_first_f32x16);
    __m256i high_first_f16x16 = _mm512_cvtps_ph(original_first_f32x16, _MM_FROUND_TO_NEAREST_INT);
    __m512 high_roundtrip_first_f32x16 = _mm512_cvtph_ps(high_first_f16x16);
    __m512 residual_first_f32x16 = _mm512_sub_ps(original_first_f32x16, high_roundtrip_first_f32x16);
    __m256i low_first_f16x16 = _mm512_cvtps_ph(residual_first_f32x16, _MM_FROUND_TO_NEAREST_INT);

    // Second 16 F32 elements
    __m256i high_second_f16x16, low_second_f16x16;
    if (valid_cols > 16) {
        nk_size_t const second_count = valid_cols - 16;
        __mmask16 second_mask = (second_count >= 16) ? 0xFFFF : ((__mmask16)1 << second_count) - 1;
        __m512 original_second_f32x16 = _mm512_mul_ps(_mm512_maskz_loadu_ps(second_mask, src + 16),
                                                      scale_second_f32x16);
        high_second_f16x16 = _mm512_cvtps_ph(original_second_f32x16, _MM_FROUND_TO_NEAREST_INT);
        __m512 high_roundtrip_second_f32x16 = _mm512_cvtph_ps(high_second_f16x16);
        __m512 residual_second_f32x16 = _mm512_sub_ps(original_second_f32x16, high_roundtrip_second_f32x16);
        low_second_f16x16 = _mm512_cvtps_ph(residual_second_f32x16, _MM_FROUND_TO_NEAREST_INT);
    }
    else {
        high_second_f16x16 = _mm256_setzero_si256();
        low_second_f16x16 = _mm256_setzero_si256();
    }

    __m512i high_row_i16x32 = _mm512_inserti64x4(_mm512_castsi256_si512(high_first_f16x16), high_second_f16x16, 1);
    __m512i low_row_i16x32 = _mm512_inserti64x4(_mm512_castsi256_si512(low_first_f16x16), low_second_f16x16, 1);
    _mm512_store_si512((__m512i *)dst_high, high_row_i16x32);
    _mm512_store_si512((__m512i *)dst_low, low_row_i16x32);
}

/* Split a block of F32 A rows into high/low FP16 tiles, zero-filling unused rows.
 * Used by the Ozaki packed GEMM to prepare A tiles for a given depth offset. */
NK_INTERNAL void nk_dots_f32ozaki_split_block_graniteamx_(                                              //
    nk_f32_t const *a_base, nk_size_t a_stride_elements, nk_size_t depth_offset, nk_size_t valid_depth, //
    nk_size_t valid_rows,                                                                               //
    nk_dots_f16_a16x32_graniteamx_t *high_tile, nk_dots_f16_a16x32_graniteamx_t *low_tile) {

    for (nk_size_t row_idx = 0; row_idx < valid_rows; row_idx++) {
        nk_f32_t const *src = a_base + row_idx * a_stride_elements + depth_offset;
        nk_dots_f32ozaki_split_row_graniteamx_(src, valid_depth, (nk_f16_t *)high_tile->data[row_idx],
                                               (nk_f16_t *)low_tile->data[row_idx]);
    }
    for (nk_size_t row_idx = valid_rows; row_idx < 16; row_idx++) {
        _mm512_store_si512((__m512i *)high_tile->data[row_idx], _mm512_setzero_si512());
        _mm512_store_si512((__m512i *)low_tile->data[row_idx], _mm512_setzero_si512());
    }
    nk_compiler_barrier_sapphireamx_();
}

#pragma endregion Helpers

#pragma region F16 Native

NK_PUBLIC nk_size_t nk_dots_packed_size_f16_graniteamx(nk_size_t column_count, nk_size_t depth) {
    nk_size_t const tmm_rows = 16;
    nk_size_t const tmm_cols = 32;
    nk_size_t const tile_bytes = 512 * sizeof(nk_f16_t); // 16 × 32 × 2 = 1KB

    nk_size_t const full_column_tiles = column_count / tmm_rows;
    nk_size_t const tiles_along_depth = nk_size_divide_round_up_(depth, tmm_cols);
    nk_size_t const column_remainder_count = column_count - full_column_tiles * tmm_rows;

    // Header (64 bytes aligned)
    nk_size_t size = sizeof(nk_dots_amx_packed_header_t);

    // All tiles for full column rows (pair-interleaved, depth remainder zero-padded)
    size += full_column_tiles * tiles_along_depth * tile_bytes;

    // Column edge: remaining rows for ALL depth columns, stored row-major
    if (column_remainder_count > 0) size += column_remainder_count * depth * sizeof(nk_f16_t);

    // Per-column norms for angular/euclidean distance (4 bytes each: f32)
    size += column_count * sizeof(nk_f32_t);

    return size;
}

NK_PUBLIC void nk_dots_pack_f16_graniteamx(                     //
    nk_f16_t const *b, nk_size_t column_count, nk_size_t depth, //
    nk_size_t b_stride_in_bytes, void *b_packed) {

    // AMX FP16 tile dimensions: 16 rows × 32 columns (512 FP16 elements = 1KB)
    nk_size_t const tmm_rows = 16;
    nk_size_t const tmm_cols = 32;
    nk_size_t const tile_elements = 512;
    nk_size_t const tile_bytes = tile_elements * sizeof(nk_f16_t);
    nk_size_t const b_stride_elements = b_stride_in_bytes / sizeof(nk_f16_t);

    // Compute layout dimensions
    nk_size_t const column_tiles_count = column_count / tmm_rows;
    nk_size_t const depth_tiles_count = nk_size_divide_round_up_(depth, tmm_cols);
    nk_size_t const column_remainder_count = column_count - column_tiles_count * tmm_rows;
    nk_size_t const total_tiles = column_tiles_count * depth_tiles_count;

    // Write header with layout metadata
    nk_dots_amx_packed_header_t *header = (nk_dots_amx_packed_header_t *)b_packed;
    header->full_column_tiles = (nk_u32_t)column_tiles_count;
    header->full_depth_tiles = (nk_u32_t)depth_tiles_count;
    header->column_remainder_count = (nk_u32_t)column_remainder_count;

    // Compute memory region offsets
    nk_size_t const tiles_offset = sizeof(nk_dots_amx_packed_header_t);
    nk_size_t const column_edge_offset = tiles_offset + total_tiles * tile_bytes;
    header->column_edge_offset = (nk_u32_t)column_edge_offset;

    // Pointers to packed data regions
    nk_f16_t *tiles_ptr = (nk_f16_t *)((char *)b_packed + tiles_offset);
    nk_f16_t *column_edge_ptr = (nk_f16_t *)((char *)b_packed + column_edge_offset);

    // Pack tiles: gather 16 strided rows into aligned temporary, transpose via SIMD, copy to packed buffer.
    // FP16 has the same 16-bit pair-interleaved layout as BF16, so reuse the BF16 transpose.
    for (nk_size_t column_tile_idx = 0; column_tile_idx < column_tiles_count; column_tile_idx++) {
        for (nk_size_t depth_tile_idx = 0; depth_tile_idx < depth_tiles_count; depth_tile_idx++) {

            nk_size_t const tile_index = column_tile_idx * depth_tiles_count + depth_tile_idx;
            nk_f16_t *tile_output = tiles_ptr + tile_index * tile_elements;

            nk_size_t const src_row_start = column_tile_idx * tmm_rows;
            nk_size_t const src_column_start = depth_tile_idx * tmm_cols;
            nk_size_t const columns_to_pack = (src_column_start + tmm_cols <= depth) ? tmm_cols
                                                                                     : (depth - src_column_start);

            // Gather 16 strided source rows into a contiguous aligned tile
            nk_dots_bf16_a16x32_sapphireamx_t source_tile;
            if (columns_to_pack == tmm_cols) {
                for (nk_size_t row_idx = 0; row_idx < tmm_rows; row_idx++) {
                    nk_f16_t const *source_row = b + (src_row_start + row_idx) * b_stride_elements + src_column_start;
                    _mm512_store_si512(&source_tile.data[row_idx][0], _mm512_loadu_si512(source_row));
                }
            }
            else {
                __mmask32 depth_mask = (__mmask32)((columns_to_pack < 32) ? ((1U << columns_to_pack) - 1) : ~0U);
                for (nk_size_t row_idx = 0; row_idx < tmm_rows; row_idx++) {
                    nk_f16_t const *source_row = b + (src_row_start + row_idx) * b_stride_elements + src_column_start;
                    _mm512_store_si512(&source_tile.data[row_idx][0], _mm512_maskz_loadu_epi16(depth_mask, source_row));
                }
            }

            // Transpose into aligned local, then copy to (potentially unaligned) packed buffer.
            // BF16 and FP16 share identical 16-bit pair-interleaved layout for TDPBF16PS/TDPFP16PS.
            nk_dots_bf16_b32x16_sapphireamx_t transposed_tile;
            nk_dots_pack_bf16_transposed_sapphireamx_(&source_tile, &transposed_tile);
            for (nk_size_t i = 0; i < tile_bytes; i += 64)
                _mm512_storeu_si512((char *)tile_output + i, _mm512_load_si512((char const *)&transposed_tile + i));
        }
    }

    // Pack column-remainder rows using vectorized masked copies
    if (column_remainder_count > 0) {
        nk_size_t const remainder_start_row = column_tiles_count * tmm_rows;
        for (nk_size_t row_idx = 0; row_idx < column_remainder_count; row_idx++) {
            nk_f16_t const *src_row = b + (remainder_start_row + row_idx) * b_stride_elements;
            nk_f16_t *dst_row = column_edge_ptr + row_idx * depth;
            nk_size_t column_idx = 0;
            for (; column_idx + 32 <= depth; column_idx += 32) {
                _mm512_storeu_si512(dst_row + column_idx, _mm512_loadu_si512(src_row + column_idx));
            }
            if (column_idx < depth) {
                __mmask32 tail_mask = (__mmask32)((1U << (depth - column_idx)) - 1);
                _mm512_mask_storeu_epi16(dst_row + column_idx, tail_mask,
                                         _mm512_maskz_loadu_epi16(tail_mask, src_row + column_idx));
            }
        }
    }

    // Compute and store per-column norms for angular/euclidean distance
    nk_size_t norms_offset = column_edge_offset +
                             (column_remainder_count > 0 ? column_remainder_count * depth * sizeof(nk_f16_t) : 0);
    header->norms_byte_offset = (nk_u32_t)norms_offset;
    nk_f32_t *norms = (nk_f32_t *)((char *)b_packed + norms_offset);
    for (nk_size_t col = 0; col < column_count; col++)
        norms[col] = nk_dots_reduce_sumsq_f16_(b + col * b_stride_elements, depth);
}

NK_PUBLIC void nk_dots_packed_f16_graniteamx(             //
    nk_f16_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows_count, nk_size_t cols_count, nk_size_t depth, nk_size_t a_stride_bytes, nk_size_t c_stride_bytes) {
    nk_unused_(cols_count);

    // Parse packed B header
    nk_dots_amx_packed_header_t const *header = (nk_dots_amx_packed_header_t const *)b_packed;
    nk_size_t const column_tiles_count = header->full_column_tiles;
    nk_size_t const depth_tiles_count = header->full_depth_tiles;
    nk_size_t const column_remainder_count = header->column_remainder_count;

    // Packed B data regions
    nk_f16_t const *b_tiles_base = (nk_f16_t const *)((char const *)b_packed + sizeof(nk_dots_amx_packed_header_t));
    nk_f16_t const *col_edge_ptr = (nk_f16_t const *)((char const *)b_packed + header->column_edge_offset);

    // Stride conversions
    nk_size_t const a_stride_elements = a_stride_bytes / sizeof(nk_f16_t);
    nk_size_t const c_stride_elements = c_stride_bytes / sizeof(nk_f32_t);

    // Tile dimensions
    nk_size_t const tile_depth = 32;
    nk_size_t const tile_size = 512;
    nk_size_t const full_cols = column_tiles_count * 16;

    // Block counts (32 × 32 output blocks = 2 × 2 tiles)
    nk_size_t const row_blocks_count = nk_size_divide_round_up_(rows_count, 32);
    nk_size_t const col_blocks_count = column_tiles_count / 2;

    if (depth_tiles_count == 0) return;

    // Tile buffers for A (only used for edge tiles)
    nk_dots_f16_a16x32_graniteamx_t a_tile_top, a_tile_bottom;
    nk_dots_f16_state2x2_graniteamx_t c_accum_buffer;

    // Precompute: number of full depth-tiles (no masking needed)
    nk_size_t const full_depth_tiles_count = depth / tile_depth;
    nk_size_t const depth_remainder = depth % tile_depth;

    nk_amx_tile_configure_sapphireamx_();

    // Loop order: row_blocks outer, col_blocks inner - maximizes A tile L2 cache reuse
    for (nk_size_t row_block_idx = 0; row_block_idx < row_blocks_count; row_block_idx++) {
        nk_size_t const row_block_start = row_block_idx * 32;
        nk_size_t const valid_rows_count = (row_block_start + 32 <= rows_count) ? 32 : (rows_count - row_block_start);
        nk_size_t const is_full_row_block = (valid_rows_count == 32);

        for (nk_size_t column_block_idx = 0; column_block_idx < col_blocks_count; column_block_idx++) {
            nk_size_t const col_block_start = column_block_idx * 32;
            nk_size_t const b_column_left_base = (column_block_idx * 2) * depth_tiles_count;
            nk_size_t const b_column_right_base = (column_block_idx * 2 + 1) * depth_tiles_count;

            // Zero accumulators (TMM4-7 stay resident across entire depth loop)
            _tile_zero(4);
            _tile_zero(5);
            _tile_zero(6);
            _tile_zero(7);

            // Fast path: full row-block with full depth-tiles → direct A load with 2-deep pipelining
            if (is_full_row_block && full_depth_tiles_count > 0) {
                nk_f16_t const *a_top_base = a + row_block_start * a_stride_elements;
                nk_f16_t const *a_bottom_base = a + (row_block_start + 16) * a_stride_elements;

                nk_dots_f16_b32x16_graniteamx_t const *b_tile_left =
                    (nk_dots_f16_b32x16_graniteamx_t const *)(b_tiles_base + b_column_left_base * tile_size);
                nk_dots_f16_b32x16_graniteamx_t const *b_tile_right =
                    (nk_dots_f16_b32x16_graniteamx_t const *)(b_tiles_base + b_column_right_base * tile_size);

                // Prologue: load first depth tile
                _tile_loadd(0, a_top_base, a_stride_bytes);
                _tile_loadd(1, a_bottom_base, a_stride_bytes);
                _tile_loadd(2, b_tile_left->data, 64);
                _tile_loadd(3, b_tile_right->data, 64);

                // Main loop: 2-deep software pipelining
                for (nk_size_t depth_tile_idx = 0; depth_tile_idx < full_depth_tiles_count - 1; depth_tile_idx++) {
                    nk_size_t const next_depth_offset = (depth_tile_idx + 1) * tile_depth;

                    _tile_dpfp16ps(4, 0, 2);
                    _tile_dpfp16ps(5, 0, 3);
                    _tile_dpfp16ps(6, 1, 2);
                    _tile_dpfp16ps(7, 1, 3);

                    _tile_loadd(0, a_top_base + next_depth_offset, a_stride_bytes);
                    _tile_loadd(1, a_bottom_base + next_depth_offset, a_stride_bytes);
                    b_tile_left = (nk_dots_f16_b32x16_graniteamx_t const *)(b_tiles_base +
                                                                            (b_column_left_base + depth_tile_idx + 1) *
                                                                                tile_size);
                    b_tile_right = (nk_dots_f16_b32x16_graniteamx_t const *)(b_tiles_base + (b_column_right_base +
                                                                                             depth_tile_idx + 1) *
                                                                                                tile_size);
                    _tile_loadd(2, b_tile_left->data, 64);
                    _tile_loadd(3, b_tile_right->data, 64);
                }

                // Epilogue: final depth tile
                _tile_dpfp16ps(4, 0, 2);
                _tile_dpfp16ps(5, 0, 3);
                _tile_dpfp16ps(6, 1, 2);
                _tile_dpfp16ps(7, 1, 3);

                // Handle partial depth-tile (if any)
                if (depth_remainder > 0) {
                    nk_size_t const depth_offset = full_depth_tiles_count * tile_depth;

                    nk_dots_f16_load_a_graniteamx_(&a_tile_top, a_top_base + depth_offset, a_stride_elements, 16,
                                                   depth_remainder);
                    nk_dots_f16_load_a_graniteamx_(&a_tile_bottom, a_bottom_base + depth_offset, a_stride_elements, 16,
                                                   depth_remainder);

                    b_tile_left = (nk_dots_f16_b32x16_graniteamx_t const *)(b_tiles_base + (b_column_left_base +
                                                                                            full_depth_tiles_count) *
                                                                                               tile_size);
                    b_tile_right = (nk_dots_f16_b32x16_graniteamx_t const *)(b_tiles_base + (b_column_right_base +
                                                                                             full_depth_tiles_count) *
                                                                                                tile_size);

                    _tile_loadd(0, a_tile_top.data, 64);
                    _tile_loadd(1, a_tile_bottom.data, 64);
                    _tile_loadd(2, b_tile_left->data, 64);
                    _tile_loadd(3, b_tile_right->data, 64);

                    _tile_dpfp16ps(4, 0, 2);
                    _tile_dpfp16ps(5, 0, 3);
                    _tile_dpfp16ps(6, 1, 2);
                    _tile_dpfp16ps(7, 1, 3);
                }
            }
            // Full row-block but only partial depth tile (depth < tile_depth)
            else if (is_full_row_block) {
                nk_f16_t const *a_top_base = a + row_block_start * a_stride_elements;
                nk_f16_t const *a_bottom_base = a + (row_block_start + 16) * a_stride_elements;

                nk_dots_f16_load_a_graniteamx_(&a_tile_top, a_top_base, a_stride_elements, 16, depth_remainder);
                nk_dots_f16_load_a_graniteamx_(&a_tile_bottom, a_bottom_base, a_stride_elements, 16, depth_remainder);

                nk_dots_f16_b32x16_graniteamx_t const *b_tile_left =
                    (nk_dots_f16_b32x16_graniteamx_t const *)(b_tiles_base + b_column_left_base * tile_size);
                nk_dots_f16_b32x16_graniteamx_t const *b_tile_right =
                    (nk_dots_f16_b32x16_graniteamx_t const *)(b_tiles_base + b_column_right_base * tile_size);

                _tile_loadd(0, a_tile_top.data, 64);
                _tile_loadd(1, a_tile_bottom.data, 64);
                _tile_loadd(2, b_tile_left->data, 64);
                _tile_loadd(3, b_tile_right->data, 64);

                _tile_dpfp16ps(4, 0, 2);
                _tile_dpfp16ps(5, 0, 3);
                _tile_dpfp16ps(6, 1, 2);
                _tile_dpfp16ps(7, 1, 3);
            }
            // Slow path: edge row-block → buffered load with masking
            else {
                nk_size_t const rows_in_high_tile = (valid_rows_count > 16) ? 16 : valid_rows_count;
                nk_size_t const rows_in_low_tile = (valid_rows_count > 16) ? valid_rows_count - 16 : 0;

                for (nk_size_t depth_tile_idx = 0; depth_tile_idx < depth_tiles_count; depth_tile_idx++) {
                    nk_size_t const depth_offset = depth_tile_idx * tile_depth;
                    nk_size_t const valid_depth = (depth_tile_idx < full_depth_tiles_count) ? tile_depth
                                                                                            : depth_remainder;

                    nk_dots_f16_load_a_graniteamx_(&a_tile_top, a + row_block_start * a_stride_elements + depth_offset,
                                                   a_stride_elements, rows_in_high_tile, valid_depth);
                    if (rows_in_low_tile > 0) {
                        nk_dots_f16_load_a_graniteamx_(&a_tile_bottom,
                                                       a + (row_block_start + 16) * a_stride_elements + depth_offset,
                                                       a_stride_elements, rows_in_low_tile, valid_depth);
                    }

                    nk_dots_f16_b32x16_graniteamx_t const *b_tile_left =
                        (nk_dots_f16_b32x16_graniteamx_t const *)(b_tiles_base +
                                                                  (b_column_left_base + depth_tile_idx) * tile_size);
                    nk_dots_f16_b32x16_graniteamx_t const *b_tile_right =
                        (nk_dots_f16_b32x16_graniteamx_t const *)(b_tiles_base +
                                                                  (b_column_right_base + depth_tile_idx) * tile_size);

                    _tile_loadd(0, a_tile_top.data, 64);
                    _tile_loadd(1, a_tile_bottom.data, 64);
                    _tile_loadd(2, b_tile_left->data, 64);
                    _tile_loadd(3, b_tile_right->data, 64);

                    _tile_dpfp16ps(4, 0, 2);
                    _tile_dpfp16ps(5, 0, 3);
                    _tile_dpfp16ps(6, 1, 2);
                    _tile_dpfp16ps(7, 1, 3);
                }
            }

            // Store accumulators to output (once per output block)
            if (is_full_row_block) {
                nk_f32_t *c_block = c + row_block_start * c_stride_elements + col_block_start;
                _tile_stored(4, c_block, c_stride_bytes);
                _tile_stored(5, c_block + 16, c_stride_bytes);
                _tile_stored(6, (nk_f32_t *)((char *)c_block + 16 * c_stride_bytes), c_stride_bytes);
                _tile_stored(7, (nk_f32_t *)((char *)c_block + 16 * c_stride_bytes) + 16, c_stride_bytes);
            }
            else {
                _tile_stored(4, c_accum_buffer.c[0][0].data, 64);
                _tile_stored(5, c_accum_buffer.c[0][1].data, 64);
                _tile_stored(6, c_accum_buffer.c[1][0].data, 64);
                _tile_stored(7, c_accum_buffer.c[1][1].data, 64);
                nk_dots_f16_output2x2_graniteamx_(&c_accum_buffer,
                                                  c + row_block_start * c_stride_elements + col_block_start,
                                                  c_stride_elements, valid_rows_count, 32);
            }
        }
    }

    // Handle odd column-tile (single 16-column tile if column_tiles_count is odd)
    if (column_tiles_count % 2 == 1) {
        nk_size_t const column_tile_idx = column_tiles_count - 1;
        nk_size_t const col_start = column_tile_idx * 16;
        nk_size_t const b_column_base = column_tile_idx * depth_tiles_count;

        for (nk_size_t row_block_idx = 0; row_block_idx < row_blocks_count; row_block_idx++) {
            nk_size_t const row_block_start = row_block_idx * 32;
            nk_size_t const valid_rows_count = (row_block_start + 32 <= rows_count) ? 32
                                                                                    : (rows_count - row_block_start);
            nk_size_t const rows_in_high_tile = (valid_rows_count > 16) ? 16 : valid_rows_count;
            nk_size_t const rows_in_low_tile = (valid_rows_count > 16) ? valid_rows_count - 16 : 0;

            nk_dots_f16_state_graniteamx_t c_high_state, c_low_state;

            _tile_zero(4);
            _tile_zero(6);

            for (nk_size_t depth_tile_idx = 0; depth_tile_idx < depth_tiles_count; depth_tile_idx++) {
                nk_size_t const depth_offset = depth_tile_idx * tile_depth;
                nk_size_t const valid_depth = (depth_tile_idx < full_depth_tiles_count) ? tile_depth : depth_remainder;

                nk_dots_f16_load_a_graniteamx_(&a_tile_top, a + row_block_start * a_stride_elements + depth_offset,
                                               a_stride_elements, rows_in_high_tile, valid_depth);
                if (rows_in_low_tile > 0) {
                    nk_dots_f16_load_a_graniteamx_(&a_tile_bottom,
                                                   a + (row_block_start + 16) * a_stride_elements + depth_offset,
                                                   a_stride_elements, rows_in_low_tile, valid_depth);
                }

                nk_dots_f16_b32x16_graniteamx_t const *b_tile =
                    (nk_dots_f16_b32x16_graniteamx_t const *)(b_tiles_base +
                                                              (b_column_base + depth_tile_idx) * tile_size);

                _tile_loadd(0, a_tile_top.data, 64);
                _tile_loadd(1, a_tile_bottom.data, 64);
                _tile_loadd(2, b_tile->data, 64);

                _tile_dpfp16ps(4, 0, 2);
                _tile_dpfp16ps(6, 1, 2);
            }

            _tile_stored(4, c_high_state.data, 64);
            _tile_stored(6, c_low_state.data, 64);

            nk_dots_f16_store_graniteamx_(&c_high_state, c + row_block_start * c_stride_elements + col_start,
                                          c_stride_elements, rows_in_high_tile, 16);
            if (rows_in_low_tile > 0) {
                nk_dots_f16_store_graniteamx_(&c_low_state, c + (row_block_start + 16) * c_stride_elements + col_start,
                                              c_stride_elements, rows_in_low_tile, 16);
            }
        }
    }

    // Handle column-edge (remaining columns < 16) using AMX with partial tiles
    if (column_remainder_count > 0) {
        for (nk_size_t row_block_idx = 0; row_block_idx < row_blocks_count; row_block_idx++) {
            nk_size_t const row_block_start = row_block_idx * 32;
            nk_size_t const valid_rows_count = (row_block_start + 32 <= rows_count) ? 32
                                                                                    : (rows_count - row_block_start);
            nk_size_t const rows_in_high_tile = (valid_rows_count > 16) ? 16 : valid_rows_count;
            nk_size_t const rows_in_low_tile = (valid_rows_count > 16) ? valid_rows_count - 16 : 0;

            nk_dots_f16_state_graniteamx_t c_high_state, c_low_state;
            nk_dots_bf16_a16x32_sapphireamx_t b_as_a;
            nk_dots_bf16_b32x16_sapphireamx_t b_tile;

            _tile_zero(4);
            _tile_zero(6);

            for (nk_size_t depth_tile_idx = 0; depth_tile_idx < depth_tiles_count; depth_tile_idx++) {
                nk_size_t const depth_offset = depth_tile_idx * tile_depth;
                nk_size_t const valid_depth = (depth_tile_idx < full_depth_tiles_count) ? tile_depth : depth_remainder;

                nk_dots_f16_load_a_graniteamx_(&a_tile_top, a + row_block_start * a_stride_elements + depth_offset,
                                               a_stride_elements, rows_in_high_tile, valid_depth);
                if (rows_in_low_tile > 0) {
                    nk_dots_f16_load_a_graniteamx_(&a_tile_bottom,
                                                   a + (row_block_start + 16) * a_stride_elements + depth_offset,
                                                   a_stride_elements, rows_in_low_tile, valid_depth);
                }

                // Load edge columns as BF16-shaped tile (same 16-bit layout) and transpose on-the-fly
                nk_dots_bf16_load_a_sapphireamx_(&b_as_a, (nk_bf16_t const *)(col_edge_ptr + depth_offset), depth,
                                                 column_remainder_count, valid_depth);
                nk_dots_pack_bf16_transposed_sapphireamx_(&b_as_a, &b_tile);

                _tile_loadd(0, a_tile_top.data, 64);
                _tile_loadd(1, a_tile_bottom.data, 64);
                _tile_loadd(2, b_tile.data, 64);

                _tile_dpfp16ps(4, 0, 2);
                _tile_dpfp16ps(6, 1, 2);
            }

            _tile_stored(4, c_high_state.data, 64);
            _tile_stored(6, c_low_state.data, 64);

            nk_dots_f16_store_graniteamx_(&c_high_state, c + row_block_start * c_stride_elements + full_cols,
                                          c_stride_elements, rows_in_high_tile, column_remainder_count);
            if (rows_in_low_tile > 0) {
                nk_dots_f16_store_graniteamx_(&c_low_state, c + (row_block_start + 16) * c_stride_elements + full_cols,
                                              c_stride_elements, rows_in_low_tile, column_remainder_count);
            }
        }
    }

    _tile_release();
}

NK_PUBLIC void nk_dots_symmetric_f16_graniteamx(                                   //
    nk_f16_t const *vectors, nk_size_t vectors_count, nk_size_t depth,             //
    nk_size_t stride_in_bytes, nk_f32_t *result, nk_size_t result_stride_in_bytes, //
    nk_size_t row_start, nk_size_t row_count) {

    nk_size_t const stride_elements = stride_in_bytes / sizeof(nk_f16_t);
    nk_size_t const result_stride_elements = result_stride_in_bytes / sizeof(nk_f32_t);

    nk_size_t const row_end = (row_count == 0)
                                  ? vectors_count
                                  : (row_start + row_count < vectors_count ? row_start + row_count : vectors_count);

    // Round depth up to multiple of 96 (3 tiles × 32 elements)
    nk_size_t const depth_tiles = nk_size_divide_round_up_(depth, 32);
    nk_size_t const depth_tile_groups = nk_size_divide_round_up_(depth_tiles, 3);

    nk_dots_f16_a16x32_graniteamx_t a_tiles[3];
    nk_dots_f16_a16x32_graniteamx_t b_src_tiles[3];
    nk_dots_f16_b32x16_graniteamx_t b_tiles[3];
    nk_dots_f16_state_graniteamx_t state;

    nk_amx_tile_configure_sapphireamx_();

    for (nk_size_t row_tile = row_start; row_tile < row_end; row_tile += 16) {
        nk_size_t const valid_rows = (row_tile + 16 <= row_end) ? 16 : (row_end - row_tile);

        for (nk_size_t col_tile = 0; col_tile < vectors_count; col_tile += 16) {
            nk_size_t const valid_cols = (col_tile + 16 <= vectors_count) ? 16 : (vectors_count - col_tile);

            nk_dots_f16_init_graniteamx_(&state);

            for (nk_size_t depth_group_idx = 0; depth_group_idx < depth_tile_groups; depth_group_idx++) {
                nk_size_t const depth_base = depth_group_idx * 96;

                for (int tile_idx = 0; tile_idx < 3; tile_idx++) {
                    nk_size_t const depth_start = depth_base + tile_idx * 32;
                    nk_size_t const valid_depth = (depth_start + 32 <= depth)
                                                      ? 32
                                                      : (depth > depth_start ? depth - depth_start : 0);

                    nk_dots_f16_load_a_graniteamx_(                         //
                        &a_tiles[tile_idx],                                 //
                        vectors + row_tile * stride_elements + depth_start, //
                        stride_elements, valid_rows, valid_depth);

                    if (row_tile == col_tile) {
                        // Reuse A data as B (self-correlation on diagonal)
                        nk_dots_pack_bf16_transposed_sapphireamx_(
                            (nk_dots_bf16_a16x32_sapphireamx_t const *)&a_tiles[tile_idx],
                            (nk_dots_bf16_b32x16_sapphireamx_t *)&b_tiles[tile_idx]);
                    }
                    else {
                        nk_dots_f16_load_a_graniteamx_(                         //
                            &b_src_tiles[tile_idx],                             //
                            vectors + col_tile * stride_elements + depth_start, //
                            stride_elements, valid_cols, valid_depth);
                        nk_dots_pack_bf16_transposed_sapphireamx_(
                            (nk_dots_bf16_a16x32_sapphireamx_t const *)&b_src_tiles[tile_idx],
                            (nk_dots_bf16_b32x16_sapphireamx_t *)&b_tiles[tile_idx]);
                    }
                }

                nk_dots_f16_update_graniteamx_(                    //
                    &state, &a_tiles[0], &a_tiles[1], &a_tiles[2], //
                    &b_tiles[0], &b_tiles[1], &b_tiles[2]);
            }

            nk_dots_f16_store_graniteamx_(                                     //
                &state, result + row_tile * result_stride_elements + col_tile, //
                result_stride_elements, valid_rows, valid_cols);
        }
    }
}

#pragma endregion F16 Native

#pragma region F32 Ozaki

NK_PUBLIC nk_size_t nk_dots_packed_size_f32_graniteamx(nk_size_t column_count, nk_size_t depth) {
    nk_size_t const tmm_rows = 16;
    nk_size_t const tmm_cols = 32;
    nk_size_t const tile_bytes = 512 * sizeof(nk_f16_t); // 16 × 32 × 2 = 1KB

    nk_size_t const full_column_tiles = column_count / tmm_rows;
    nk_size_t const tiles_along_depth = nk_size_divide_round_up_(depth, tmm_cols);
    nk_size_t const column_remainder_count = column_count - full_column_tiles * tmm_rows;

    // Header (64 bytes aligned)
    nk_size_t size = sizeof(nk_dots_f32ozaki_packed_header_graniteamx_t);

    // B_high tiles + B_low tiles (2× the normal FP16 tiles)
    size += 2 * full_column_tiles * tiles_along_depth * tile_bytes;

    // Column edge: hi + lo remainder rows for ALL depth, row-major
    if (column_remainder_count > 0) size += 2 * column_remainder_count * depth * sizeof(nk_f16_t);

    // Per-column norms (f64 for F32 inputs)
    size += column_count * sizeof(nk_f64_t);

    // Per-column-tile inverse scale factors
    size += full_column_tiles * sizeof(nk_f32_t);

    return size;
}

NK_PUBLIC void nk_dots_pack_f32_graniteamx(                     //
    nk_f32_t const *b, nk_size_t column_count, nk_size_t depth, //
    nk_size_t b_stride_in_bytes, void *b_packed) {

    nk_size_t const tmm_rows = 16;
    nk_size_t const tmm_cols = 32;
    nk_size_t const tile_elements = 512;
    nk_size_t const tile_bytes = tile_elements * sizeof(nk_f16_t);
    nk_size_t const b_stride_elements = b_stride_in_bytes / sizeof(nk_f32_t);

    nk_size_t const column_tiles_count = column_count / tmm_rows;
    nk_size_t const depth_tiles_count = nk_size_divide_round_up_(depth, tmm_cols);
    nk_size_t const column_remainder_count = column_count - column_tiles_count * tmm_rows;
    nk_size_t const total_tiles = column_tiles_count * depth_tiles_count;

    // Write header
    nk_dots_f32ozaki_packed_header_graniteamx_t *header = (nk_dots_f32ozaki_packed_header_graniteamx_t *)b_packed;
    header->full_column_tiles = (nk_u32_t)column_tiles_count;
    header->full_depth_tiles = (nk_u32_t)depth_tiles_count;
    header->column_remainder_count = (nk_u32_t)column_remainder_count;

    // Compute offsets: [header][B_high tiles][B_low tiles][edge_high][edge_low][norms][scales]
    nk_size_t const high_tiles_offset = sizeof(nk_dots_f32ozaki_packed_header_graniteamx_t);
    nk_size_t const low_tiles_offset = high_tiles_offset + total_tiles * tile_bytes;
    header->b_low_tiles_offset = (nk_u32_t)low_tiles_offset;

    nk_size_t const column_edge_offset = low_tiles_offset + total_tiles * tile_bytes;
    header->column_edge_offset = (nk_u32_t)column_edge_offset;

    nk_size_t const edge_size = column_remainder_count > 0 ? 2 * column_remainder_count * depth * sizeof(nk_f16_t) : 0;
    nk_size_t const norms_offset = column_edge_offset + edge_size;
    header->norms_byte_offset = (nk_u32_t)norms_offset;

    nk_size_t const scales_offset = norms_offset + column_count * sizeof(nk_f64_t);
    header->scales_offset = (nk_u32_t)scales_offset;

    nk_f16_t *high_tiles_ptr = (nk_f16_t *)((char *)b_packed + high_tiles_offset);
    nk_f16_t *low_tiles_ptr = (nk_f16_t *)((char *)b_packed + low_tiles_offset);
    nk_f16_t *edge_high_ptr = (nk_f16_t *)((char *)b_packed + column_edge_offset);
    nk_f16_t *edge_low_ptr = edge_high_ptr + (column_remainder_count > 0 ? column_remainder_count * depth : 0);
    nk_f32_t *inv_scales = (nk_f32_t *)((char *)b_packed + scales_offset);

    // Pack full column tiles with per-tile scaling
    for (nk_size_t column_tile_idx = 0; column_tile_idx < column_tiles_count; column_tile_idx++) {
        nk_size_t const src_row_start = column_tile_idx * tmm_rows;

        // Scan max absolute value across all 16 columns × all depth elements
        __m512 column_max_f32x16 = _mm512_setzero_ps();
        for (nk_size_t depth_idx = 0; depth_idx < depth; depth_idx++) {
            __m512 values_f32x16 = _mm512_set_ps(b[(src_row_start + 15) * b_stride_elements + depth_idx],
                                                 b[(src_row_start + 14) * b_stride_elements + depth_idx],
                                                 b[(src_row_start + 13) * b_stride_elements + depth_idx],
                                                 b[(src_row_start + 12) * b_stride_elements + depth_idx],
                                                 b[(src_row_start + 11) * b_stride_elements + depth_idx],
                                                 b[(src_row_start + 10) * b_stride_elements + depth_idx],
                                                 b[(src_row_start + 9) * b_stride_elements + depth_idx],
                                                 b[(src_row_start + 8) * b_stride_elements + depth_idx],
                                                 b[(src_row_start + 7) * b_stride_elements + depth_idx],
                                                 b[(src_row_start + 6) * b_stride_elements + depth_idx],
                                                 b[(src_row_start + 5) * b_stride_elements + depth_idx],
                                                 b[(src_row_start + 4) * b_stride_elements + depth_idx],
                                                 b[(src_row_start + 3) * b_stride_elements + depth_idx],
                                                 b[(src_row_start + 2) * b_stride_elements + depth_idx],
                                                 b[(src_row_start + 1) * b_stride_elements + depth_idx],
                                                 b[(src_row_start + 0) * b_stride_elements + depth_idx]);
            column_max_f32x16 = _mm512_max_ps(column_max_f32x16, _mm512_abs_ps(values_f32x16));
        }
        nk_f32_t tile_max_abs = _mm512_reduce_max_ps(column_max_f32x16);

        // Compute scale factor: if max exceeds FP16 range, scale down; otherwise use 1.0
        nk_f32_t scale = 1.0f;
        nk_f32_t inv_scale = 1.0f;
        if (tile_max_abs > 65504.0f) {
            scale = 65504.0f / tile_max_abs;
            inv_scale = tile_max_abs / 65504.0f;
        }
        inv_scales[column_tile_idx] = inv_scale;

        // Pack each depth tile: split F32→FP16 high/low with scaling, transpose both halves
        for (nk_size_t depth_tile_idx = 0; depth_tile_idx < depth_tiles_count; depth_tile_idx++) {
            nk_size_t const tile_index = column_tile_idx * depth_tiles_count + depth_tile_idx;
            nk_size_t const src_column_start = depth_tile_idx * tmm_cols;
            nk_size_t const columns_to_pack = (src_column_start + tmm_cols <= depth) ? tmm_cols
                                                                                     : (depth - src_column_start);

            // Split each of the 16 source rows into high/low FP16 tiles
            nk_dots_bf16_a16x32_sapphireamx_t source_high, source_low;
            __m512 scale_first_f32x16 = _mm512_set1_ps(scale);
            __m512 scale_second_f32x16 = _mm512_set1_ps(scale);

            for (nk_size_t row_idx = 0; row_idx < tmm_rows; row_idx++) {
                nk_f32_t const *src_row = b + (src_row_start + row_idx) * b_stride_elements + src_column_start;
                nk_dots_f32ozaki_split_row_scaled_graniteamx_(
                    src_row, columns_to_pack, scale_first_f32x16, scale_second_f32x16,
                    (nk_f16_t *)source_high.data[row_idx], (nk_f16_t *)source_low.data[row_idx]);
            }

            // Transpose both halves using the BF16 transposer (identical 16-bit pair-interleaved layout)
            nk_dots_bf16_b32x16_sapphireamx_t transposed_high, transposed_low;
            nk_dots_pack_bf16_transposed_sapphireamx_(&source_high, &transposed_high);
            nk_dots_pack_bf16_transposed_sapphireamx_(&source_low, &transposed_low);

            // Copy to packed buffer
            nk_f16_t *high_out = high_tiles_ptr + tile_index * tile_elements;
            nk_f16_t *low_out = low_tiles_ptr + tile_index * tile_elements;
            for (nk_size_t i = 0; i < tile_bytes; i += 64) {
                _mm512_storeu_si512((char *)high_out + i, _mm512_load_si512((char const *)&transposed_high + i));
                _mm512_storeu_si512((char *)low_out + i, _mm512_load_si512((char const *)&transposed_low + i));
            }
        }
    }

    // Pack column-remainder using vectorized splitting
    if (column_remainder_count > 0) {
        nk_size_t const remainder_start_row = column_tiles_count * tmm_rows;
        for (nk_size_t row_idx = 0; row_idx < column_remainder_count; row_idx++) {
            nk_f32_t const *src_row = b + (remainder_start_row + row_idx) * b_stride_elements;
            nk_f16_t *dst_high_row = edge_high_ptr + row_idx * depth;
            nk_f16_t *dst_low_row = edge_low_ptr + row_idx * depth;
            nk_size_t col_idx = 0;

            // Process 16 F32 elements at a time
            for (; col_idx + 16 <= depth; col_idx += 16) {
                __m512 original_f32x16 = _mm512_loadu_ps(src_row + col_idx);
                __m256i high_f16x16 = _mm512_cvtps_ph(original_f32x16, _MM_FROUND_TO_NEAREST_INT);
                __m512 roundtrip_f32x16 = _mm512_cvtph_ps(high_f16x16);
                __m512 residual_f32x16 = _mm512_sub_ps(original_f32x16, roundtrip_f32x16);
                __m256i low_f16x16 = _mm512_cvtps_ph(residual_f32x16, _MM_FROUND_TO_NEAREST_INT);
                _mm256_storeu_si256((__m256i *)(dst_high_row + col_idx), high_f16x16);
                _mm256_storeu_si256((__m256i *)(dst_low_row + col_idx), low_f16x16);
            }
            if (col_idx < depth) {
                __mmask16 tail_mask = (__mmask16)((1u << (depth - col_idx)) - 1);
                __m512 original_f32x16 = _mm512_maskz_loadu_ps(tail_mask, src_row + col_idx);
                __m256i high_f16x16 = _mm512_cvtps_ph(original_f32x16, _MM_FROUND_TO_NEAREST_INT);
                __m512 roundtrip_f32x16 = _mm512_cvtph_ps(high_f16x16);
                __m512 residual_f32x16 = _mm512_sub_ps(original_f32x16, roundtrip_f32x16);
                __m256i low_f16x16 = _mm512_cvtps_ph(residual_f32x16, _MM_FROUND_TO_NEAREST_INT);
                _mm256_mask_storeu_epi16(dst_high_row + col_idx, tail_mask, high_f16x16);
                _mm256_mask_storeu_epi16(dst_low_row + col_idx, tail_mask, low_f16x16);
            }
        }
    }

    // Compute and store per-column norms (f64 for F32 inputs)
    nk_f64_t *norms = (nk_f64_t *)((char *)b_packed + norms_offset);
    for (nk_size_t col = 0; col < column_count; col++)
        norms[col] = nk_dots_reduce_sumsq_f32_(b + col * b_stride_elements, depth);
}

NK_PUBLIC void nk_dots_packed_f32_graniteamx(             //
    nk_f32_t const *a, void const *b_packed, nk_f64_t *c, //
    nk_size_t rows_count, nk_size_t cols_count, nk_size_t depth, nk_size_t a_stride_bytes, nk_size_t c_stride_bytes) {
    nk_unused_(cols_count);

    // Parse packed B header
    nk_dots_f32ozaki_packed_header_graniteamx_t const *header =
        (nk_dots_f32ozaki_packed_header_graniteamx_t const *)b_packed;
    nk_size_t const column_tiles_count = header->full_column_tiles;
    nk_size_t const depth_tiles_count = header->full_depth_tiles;
    nk_size_t const column_remainder_count = header->column_remainder_count;

    // Packed B data regions
    nk_size_t const high_tiles_offset = sizeof(nk_dots_f32ozaki_packed_header_graniteamx_t);
    nk_f16_t const *b_high_base = (nk_f16_t const *)((char const *)b_packed + high_tiles_offset);
    nk_f16_t const *b_low_base = (nk_f16_t const *)((char const *)b_packed + header->b_low_tiles_offset);
    nk_f16_t const *col_edge_ptr = (nk_f16_t const *)((char const *)b_packed + header->column_edge_offset);
    nk_f32_t const *inv_scales = (nk_f32_t const *)((char const *)b_packed + header->scales_offset);

    // Stride conversions
    nk_size_t const a_stride_elements = a_stride_bytes / sizeof(nk_f32_t);
    nk_size_t const c_stride_elements = c_stride_bytes / sizeof(nk_f64_t);

    // Tile dimensions
    nk_size_t const tile_depth = 32;
    nk_size_t const tile_size = 512;
    nk_size_t const full_cols = column_tiles_count * 16;

    nk_size_t const row_blocks_count = nk_size_divide_round_up_(rows_count, 32);
    nk_size_t const col_blocks_count = column_tiles_count / 2;

    if (depth_tiles_count == 0) return;

    nk_size_t const full_depth_tiles_count = depth / tile_depth;
    nk_size_t const depth_remainder = depth % tile_depth;

    // Tile buffers for Ozaki A splitting
    nk_dots_f16_a16x32_graniteamx_t a_top_high_tile, a_top_low_tile;
    nk_dots_f16_a16x32_graniteamx_t a_bottom_high_tile, a_bottom_low_tile;
    nk_dots_f16_state2x2_graniteamx_t c_accum_buffer;

    nk_amx_tile_configure_sapphireamx_();

    for (nk_size_t row_block_idx = 0; row_block_idx < row_blocks_count; row_block_idx++) {
        nk_size_t const row_block_start = row_block_idx * 32;
        nk_size_t const valid_rows_count = (row_block_start + 32 <= rows_count) ? 32 : (rows_count - row_block_start);
        nk_size_t const rows_in_high_tile = (valid_rows_count > 16) ? 16 : valid_rows_count;
        nk_size_t const rows_in_low_tile = (valid_rows_count > 16) ? valid_rows_count - 16 : 0;

        for (nk_size_t column_block_idx = 0; column_block_idx < col_blocks_count; column_block_idx++) {
            nk_size_t const col_block_start = column_block_idx * 32;
            nk_size_t const left_col_tile = column_block_idx * 2;
            nk_size_t const right_col_tile = column_block_idx * 2 + 1;
            nk_size_t const b_column_left_base = left_col_tile * depth_tiles_count;
            nk_size_t const b_column_right_base = right_col_tile * depth_tiles_count;

            _tile_zero(4);
            _tile_zero(5);
            _tile_zero(6);
            _tile_zero(7);

            for (nk_size_t depth_tile_idx = 0; depth_tile_idx < depth_tiles_count; depth_tile_idx++) {
                nk_size_t const depth_offset = depth_tile_idx * tile_depth;
                nk_size_t const valid_depth = (depth_tile_idx < full_depth_tiles_count) ? tile_depth : depth_remainder;

                // Split top and bottom A rows into high/low FP16 tiles
                nk_dots_f32ozaki_split_block_graniteamx_(a + row_block_start * a_stride_elements, a_stride_elements,
                                                         depth_offset, valid_depth, rows_in_high_tile, &a_top_high_tile,
                                                         &a_top_low_tile);
                nk_dots_f32ozaki_split_block_graniteamx_(a + (row_block_start + 16) * a_stride_elements,
                                                         a_stride_elements, depth_offset, valid_depth, rows_in_low_tile,
                                                         &a_bottom_high_tile, &a_bottom_low_tile);

                // Load B_high and B_low for left column tile
                nk_dots_f16_b32x16_graniteamx_t const *b_high_left =
                    (nk_dots_f16_b32x16_graniteamx_t const *)(b_high_base +
                                                              (b_column_left_base + depth_tile_idx) * tile_size);
                nk_dots_f16_b32x16_graniteamx_t const *b_low_left =
                    (nk_dots_f16_b32x16_graniteamx_t const *)(b_low_base +
                                                              (b_column_left_base + depth_tile_idx) * tile_size);

                // Top rows × left column: 4 TDPFP16PS into TMM4
                _tile_loadd(2, b_high_left->data, 64);
                _tile_loadd(3, b_low_left->data, 64);

                _tile_loadd(0, a_top_high_tile.data, 64);
                _tile_dpfp16ps(4, 0, 2);
                _tile_dpfp16ps(4, 0, 3);
                _tile_loadd(0, a_top_low_tile.data, 64);
                _tile_dpfp16ps(4, 0, 2);
                _tile_dpfp16ps(4, 0, 3);

                // Bottom rows × left column: 4 TDPFP16PS into TMM6
                _tile_loadd(1, a_bottom_high_tile.data, 64);
                _tile_dpfp16ps(6, 1, 2);
                _tile_dpfp16ps(6, 1, 3);
                _tile_loadd(1, a_bottom_low_tile.data, 64);
                _tile_dpfp16ps(6, 1, 2);
                _tile_dpfp16ps(6, 1, 3);

                // Load B_high and B_low for right column tile
                nk_dots_f16_b32x16_graniteamx_t const *b_high_right =
                    (nk_dots_f16_b32x16_graniteamx_t const *)(b_high_base +
                                                              (b_column_right_base + depth_tile_idx) * tile_size);
                nk_dots_f16_b32x16_graniteamx_t const *b_low_right =
                    (nk_dots_f16_b32x16_graniteamx_t const *)(b_low_base +
                                                              (b_column_right_base + depth_tile_idx) * tile_size);

                // Top rows × right column: 4 TDPFP16PS into TMM5
                _tile_loadd(2, b_high_right->data, 64);
                _tile_loadd(3, b_low_right->data, 64);

                _tile_loadd(0, a_top_high_tile.data, 64);
                _tile_dpfp16ps(5, 0, 2);
                _tile_dpfp16ps(5, 0, 3);
                _tile_loadd(0, a_top_low_tile.data, 64);
                _tile_dpfp16ps(5, 0, 2);
                _tile_dpfp16ps(5, 0, 3);

                // Bottom rows × right column: 4 TDPFP16PS into TMM7
                _tile_loadd(1, a_bottom_high_tile.data, 64);
                _tile_dpfp16ps(7, 1, 2);
                _tile_dpfp16ps(7, 1, 3);
                _tile_loadd(1, a_bottom_low_tile.data, 64);
                _tile_dpfp16ps(7, 1, 2);
                _tile_dpfp16ps(7, 1, 3);
            }

            // Store accumulators, applying inverse scale factors and widening to f64
            _tile_stored(4, c_accum_buffer.c[0][0].data, 64);
            _tile_stored(5, c_accum_buffer.c[0][1].data, 64);
            _tile_stored(6, c_accum_buffer.c[1][0].data, 64);
            _tile_stored(7, c_accum_buffer.c[1][1].data, 64);

            // Apply inverse scale for left and right column tiles
            nk_f32_t left_inv_scale = inv_scales[left_col_tile];
            nk_f32_t right_inv_scale = inv_scales[right_col_tile];
            if (left_inv_scale != 1.0f) {
                __m512 inv_scale_f32x16 = _mm512_set1_ps(left_inv_scale);
                for (nk_size_t r = 0; r < 16; r++) {
                    _mm512_store_ps(c_accum_buffer.c[0][0].data[r],
                                    _mm512_mul_ps(_mm512_load_ps(c_accum_buffer.c[0][0].data[r]), inv_scale_f32x16));
                    _mm512_store_ps(c_accum_buffer.c[1][0].data[r],
                                    _mm512_mul_ps(_mm512_load_ps(c_accum_buffer.c[1][0].data[r]), inv_scale_f32x16));
                }
            }
            if (right_inv_scale != 1.0f) {
                __m512 inv_scale_f32x16 = _mm512_set1_ps(right_inv_scale);
                for (nk_size_t r = 0; r < 16; r++) {
                    _mm512_store_ps(c_accum_buffer.c[0][1].data[r],
                                    _mm512_mul_ps(_mm512_load_ps(c_accum_buffer.c[0][1].data[r]), inv_scale_f32x16));
                    _mm512_store_ps(c_accum_buffer.c[1][1].data[r],
                                    _mm512_mul_ps(_mm512_load_ps(c_accum_buffer.c[1][1].data[r]), inv_scale_f32x16));
                }
            }

            nk_dots_f32ozaki_output2x2_f64_graniteamx_(&c_accum_buffer,
                                                       c + row_block_start * c_stride_elements + col_block_start,
                                                       c_stride_elements, valid_rows_count, 32);
        }
    }

    // Handle odd column-tile
    if (column_tiles_count % 2 == 1) {
        nk_size_t const column_tile_idx = column_tiles_count - 1;
        nk_size_t const col_start = column_tile_idx * 16;
        nk_size_t const b_column_base = column_tile_idx * depth_tiles_count;

        for (nk_size_t row_block_idx = 0; row_block_idx < row_blocks_count; row_block_idx++) {
            nk_size_t const row_block_start = row_block_idx * 32;
            nk_size_t const valid_rows_count = (row_block_start + 32 <= rows_count) ? 32
                                                                                    : (rows_count - row_block_start);
            nk_size_t const rows_in_high_tile = (valid_rows_count > 16) ? 16 : valid_rows_count;
            nk_size_t const rows_in_low_tile = (valid_rows_count > 16) ? valid_rows_count - 16 : 0;

            nk_dots_f16_state_graniteamx_t c_high_state, c_low_state;

            _tile_zero(4);
            _tile_zero(6);

            for (nk_size_t depth_tile_idx = 0; depth_tile_idx < depth_tiles_count; depth_tile_idx++) {
                nk_size_t const depth_offset = depth_tile_idx * tile_depth;
                nk_size_t const valid_depth = (depth_tile_idx < full_depth_tiles_count) ? tile_depth : depth_remainder;

                nk_dots_f32ozaki_split_block_graniteamx_(a + row_block_start * a_stride_elements, a_stride_elements,
                                                         depth_offset, valid_depth, rows_in_high_tile, &a_top_high_tile,
                                                         &a_top_low_tile);
                nk_dots_f32ozaki_split_block_graniteamx_(a + (row_block_start + 16) * a_stride_elements,
                                                         a_stride_elements, depth_offset, valid_depth, rows_in_low_tile,
                                                         &a_bottom_high_tile, &a_bottom_low_tile);

                nk_dots_f16_b32x16_graniteamx_t const *b_high_tile =
                    (nk_dots_f16_b32x16_graniteamx_t const *)(b_high_base +
                                                              (b_column_base + depth_tile_idx) * tile_size);
                nk_dots_f16_b32x16_graniteamx_t const *b_low_tile =
                    (nk_dots_f16_b32x16_graniteamx_t const *)(b_low_base +
                                                              (b_column_base + depth_tile_idx) * tile_size);

                _tile_loadd(2, b_high_tile->data, 64);
                _tile_loadd(3, b_low_tile->data, 64);

                _tile_loadd(0, a_top_high_tile.data, 64);
                _tile_dpfp16ps(4, 0, 2);
                _tile_dpfp16ps(4, 0, 3);
                _tile_loadd(0, a_top_low_tile.data, 64);
                _tile_dpfp16ps(4, 0, 2);
                _tile_dpfp16ps(4, 0, 3);

                _tile_loadd(1, a_bottom_high_tile.data, 64);
                _tile_dpfp16ps(6, 1, 2);
                _tile_dpfp16ps(6, 1, 3);
                _tile_loadd(1, a_bottom_low_tile.data, 64);
                _tile_dpfp16ps(6, 1, 2);
                _tile_dpfp16ps(6, 1, 3);
            }

            _tile_stored(4, c_high_state.data, 64);
            _tile_stored(6, c_low_state.data, 64);

            // Apply inverse scale
            nk_f32_t col_inv_scale = inv_scales[column_tile_idx];
            if (col_inv_scale != 1.0f) {
                __m512 inv_scale_f32x16 = _mm512_set1_ps(col_inv_scale);
                for (nk_size_t r = 0; r < 16; r++) {
                    _mm512_store_ps(c_high_state.data[r],
                                    _mm512_mul_ps(_mm512_load_ps(c_high_state.data[r]), inv_scale_f32x16));
                    _mm512_store_ps(c_low_state.data[r],
                                    _mm512_mul_ps(_mm512_load_ps(c_low_state.data[r]), inv_scale_f32x16));
                }
            }

            nk_dots_f32ozaki_store_f64_graniteamx_(&c_high_state, c + row_block_start * c_stride_elements + col_start,
                                                   c_stride_elements, rows_in_high_tile, 16);
            if (rows_in_low_tile > 0) {
                nk_dots_f32ozaki_store_f64_graniteamx_(&c_low_state,
                                                       c + (row_block_start + 16) * c_stride_elements + col_start,
                                                       c_stride_elements, rows_in_low_tile, 16);
            }
        }
    }

    // Handle column-edge (remaining columns < 16) — no scaling needed for edge (small tile)
    if (column_remainder_count > 0) {
        nk_f16_t const *edge_high = col_edge_ptr;
        nk_f16_t const *edge_low = col_edge_ptr + column_remainder_count * depth;

        for (nk_size_t row_block_idx = 0; row_block_idx < row_blocks_count; row_block_idx++) {
            nk_size_t const row_block_start = row_block_idx * 32;
            nk_size_t const valid_rows_count = (row_block_start + 32 <= rows_count) ? 32
                                                                                    : (rows_count - row_block_start);
            nk_size_t const rows_in_high_tile = (valid_rows_count > 16) ? 16 : valid_rows_count;
            nk_size_t const rows_in_low_tile = (valid_rows_count > 16) ? valid_rows_count - 16 : 0;

            nk_dots_f16_state_graniteamx_t c_high_state, c_low_state;
            nk_dots_bf16_a16x32_sapphireamx_t b_as_a_high, b_as_a_low;
            nk_dots_bf16_b32x16_sapphireamx_t b_tile_high, b_tile_low;

            _tile_zero(4);
            _tile_zero(6);

            for (nk_size_t depth_tile_idx = 0; depth_tile_idx < depth_tiles_count; depth_tile_idx++) {
                nk_size_t const depth_offset = depth_tile_idx * tile_depth;
                nk_size_t const valid_depth = (depth_tile_idx < full_depth_tiles_count) ? tile_depth : depth_remainder;

                // Split top A rows
                for (nk_size_t row_idx = 0; row_idx < rows_in_high_tile; row_idx++) {
                    nk_f32_t const *src = a + (row_block_start + row_idx) * a_stride_elements + depth_offset;
                    nk_dots_f32ozaki_split_row_graniteamx_(src, valid_depth, (nk_f16_t *)a_top_high_tile.data[row_idx],
                                                           (nk_f16_t *)a_top_low_tile.data[row_idx]);
                }
                for (nk_size_t row_idx = rows_in_high_tile; row_idx < 16; row_idx++) {
                    _mm512_store_si512((__m512i *)a_top_high_tile.data[row_idx], _mm512_setzero_si512());
                    _mm512_store_si512((__m512i *)a_top_low_tile.data[row_idx], _mm512_setzero_si512());
                }

                // Split bottom A rows
                for (nk_size_t row_idx = 0; row_idx < rows_in_low_tile; row_idx++) {
                    nk_f32_t const *src = a + (row_block_start + 16 + row_idx) * a_stride_elements + depth_offset;
                    nk_dots_f32ozaki_split_row_graniteamx_(src, valid_depth,
                                                           (nk_f16_t *)a_bottom_high_tile.data[row_idx],
                                                           (nk_f16_t *)a_bottom_low_tile.data[row_idx]);
                }
                for (nk_size_t row_idx = rows_in_low_tile; row_idx < 16; row_idx++) {
                    _mm512_store_si512((__m512i *)a_bottom_high_tile.data[row_idx], _mm512_setzero_si512());
                    _mm512_store_si512((__m512i *)a_bottom_low_tile.data[row_idx], _mm512_setzero_si512());
                }
                nk_compiler_barrier_sapphireamx_();

                // Load edge B high/low and transpose
                nk_dots_bf16_load_a_sapphireamx_(&b_as_a_high, (nk_bf16_t const *)(edge_high + depth_offset), depth,
                                                 column_remainder_count, valid_depth);
                nk_dots_bf16_load_a_sapphireamx_(&b_as_a_low, (nk_bf16_t const *)(edge_low + depth_offset), depth,
                                                 column_remainder_count, valid_depth);
                nk_dots_pack_bf16_transposed_sapphireamx_(&b_as_a_high, &b_tile_high);
                nk_dots_pack_bf16_transposed_sapphireamx_(&b_as_a_low, &b_tile_low);

                // 4 Ozaki terms for top rows
                _tile_loadd(2, b_tile_high.data, 64);
                _tile_loadd(3, b_tile_low.data, 64);

                _tile_loadd(0, a_top_high_tile.data, 64);
                _tile_dpfp16ps(4, 0, 2);
                _tile_dpfp16ps(4, 0, 3);
                _tile_loadd(0, a_top_low_tile.data, 64);
                _tile_dpfp16ps(4, 0, 2);
                _tile_dpfp16ps(4, 0, 3);

                // 4 Ozaki terms for bottom rows
                _tile_loadd(1, a_bottom_high_tile.data, 64);
                _tile_dpfp16ps(6, 1, 2);
                _tile_dpfp16ps(6, 1, 3);
                _tile_loadd(1, a_bottom_low_tile.data, 64);
                _tile_dpfp16ps(6, 1, 2);
                _tile_dpfp16ps(6, 1, 3);
            }

            _tile_stored(4, c_high_state.data, 64);
            _tile_stored(6, c_low_state.data, 64);

            nk_dots_f32ozaki_store_f64_graniteamx_(&c_high_state, c + row_block_start * c_stride_elements + full_cols,
                                                   c_stride_elements, rows_in_high_tile, column_remainder_count);
            if (rows_in_low_tile > 0) {
                nk_dots_f32ozaki_store_f64_graniteamx_(&c_low_state,
                                                       c + (row_block_start + 16) * c_stride_elements + full_cols,
                                                       c_stride_elements, rows_in_low_tile, column_remainder_count);
            }
        }
    }

    _tile_release();
}

NK_PUBLIC void nk_dots_symmetric_f32_graniteamx(                                   //
    nk_f32_t const *vectors, nk_size_t vectors_count, nk_size_t depth,             //
    nk_size_t stride_in_bytes, nk_f64_t *result, nk_size_t result_stride_in_bytes, //
    nk_size_t row_start, nk_size_t row_count) {

    nk_size_t const stride_elements = stride_in_bytes / sizeof(nk_f32_t);
    nk_size_t const result_stride_elements = result_stride_in_bytes / sizeof(nk_f64_t);
    nk_size_t const row_end = (row_count == 0)
                                  ? vectors_count
                                  : (row_start + row_count < vectors_count ? row_start + row_count : vectors_count);

    // 1-deep depth grouping for Ozaki: split A and B from F32, accumulate per depth tile
    nk_size_t const depth_tiles = nk_size_divide_round_up_(depth, 32);

    nk_dots_f16_a16x32_graniteamx_t a_high_tile, a_low_tile;
    nk_dots_f16_a16x32_graniteamx_t b_src_high_tile, b_src_low_tile;
    nk_dots_f16_b32x16_graniteamx_t b_high_tile, b_low_tile;
    nk_dots_f16_state_graniteamx_t state;

    nk_amx_tile_configure_sapphireamx_();

    for (nk_size_t row_tile = row_start; row_tile < row_end; row_tile += 16) {
        nk_size_t const valid_rows = (row_tile + 16 <= row_end) ? 16 : (row_end - row_tile);

        for (nk_size_t col_tile = 0; col_tile < vectors_count; col_tile += 16) {
            nk_size_t const valid_cols = (col_tile + 16 <= vectors_count) ? 16 : (vectors_count - col_tile);

            nk_dots_f16_init_graniteamx_(&state);

            for (nk_size_t depth_tile_idx = 0; depth_tile_idx < depth_tiles; depth_tile_idx++) {
                nk_size_t const depth_start = depth_tile_idx * 32;
                nk_size_t const valid_depth = (depth_start + 32 <= depth)
                                                  ? 32
                                                  : (depth > depth_start ? depth - depth_start : 0);
                if (valid_depth == 0) continue;

                // Split A rows: F32 → FP16 high/low
                for (nk_size_t row_idx = 0; row_idx < valid_rows; row_idx++) {
                    nk_f32_t const *src = vectors + (row_tile + row_idx) * stride_elements + depth_start;
                    nk_dots_f32ozaki_split_row_graniteamx_(src, valid_depth, (nk_f16_t *)a_high_tile.data[row_idx],
                                                           (nk_f16_t *)a_low_tile.data[row_idx]);
                }
                for (nk_size_t row_idx = valid_rows; row_idx < 16; row_idx++) {
                    _mm512_store_si512((__m512i *)a_high_tile.data[row_idx], _mm512_setzero_si512());
                    _mm512_store_si512((__m512i *)a_low_tile.data[row_idx], _mm512_setzero_si512());
                }

                // Split B columns: F32 → FP16 high/low, then transpose
                if (row_tile == col_tile) {
                    // Self-correlation on diagonal: reuse A data as B
                    nk_dots_pack_bf16_transposed_sapphireamx_((nk_dots_bf16_a16x32_sapphireamx_t const *)&a_high_tile,
                                                              (nk_dots_bf16_b32x16_sapphireamx_t *)&b_high_tile);
                    nk_dots_pack_bf16_transposed_sapphireamx_((nk_dots_bf16_a16x32_sapphireamx_t const *)&a_low_tile,
                                                              (nk_dots_bf16_b32x16_sapphireamx_t *)&b_low_tile);
                }
                else {
                    for (nk_size_t col_idx = 0; col_idx < valid_cols; col_idx++) {
                        nk_f32_t const *src = vectors + (col_tile + col_idx) * stride_elements + depth_start;
                        nk_dots_f32ozaki_split_row_graniteamx_(src, valid_depth,
                                                               (nk_f16_t *)b_src_high_tile.data[col_idx],
                                                               (nk_f16_t *)b_src_low_tile.data[col_idx]);
                    }
                    for (nk_size_t col_idx = valid_cols; col_idx < 16; col_idx++) {
                        _mm512_store_si512((__m512i *)b_src_high_tile.data[col_idx], _mm512_setzero_si512());
                        _mm512_store_si512((__m512i *)b_src_low_tile.data[col_idx], _mm512_setzero_si512());
                    }
                    nk_dots_pack_bf16_transposed_sapphireamx_(
                        (nk_dots_bf16_a16x32_sapphireamx_t const *)&b_src_high_tile,
                        (nk_dots_bf16_b32x16_sapphireamx_t *)&b_high_tile);
                    nk_dots_pack_bf16_transposed_sapphireamx_(
                        (nk_dots_bf16_a16x32_sapphireamx_t const *)&b_src_low_tile,
                        (nk_dots_bf16_b32x16_sapphireamx_t *)&b_low_tile);
                }

                nk_compiler_barrier_sapphireamx_();
                nk_dots_f32ozaki_update_graniteamx_(&state, &a_high_tile, &a_low_tile, &b_high_tile, &b_low_tile);
            }

            // Widen F32 state to F64 and store
            nk_dots_f32ozaki_store_f64_graniteamx_(&state, result + row_tile * result_stride_elements + col_tile,
                                                   result_stride_elements, valid_rows, valid_cols);
        }
    }
}

#pragma endregion F32 Ozaki

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_GRANITEAMX
#endif // NK_TARGET_X8664_
#endif // NK_DOTS_GRANITEAMX_H

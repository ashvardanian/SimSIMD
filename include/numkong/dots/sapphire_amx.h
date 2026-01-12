/**
 *  @brief SIMD-accelerated Dot Products for Real and Complex Numbers optimized for Intel Sapphire Rapids CPUs.
 *  @file include/numkong/dots/sapphire.h
 *  @sa include/numkong/dots.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 *
 *  This file contains tiled matrix-multiplication kernels optimized for Intel AMX instructions,
 *  leveraging the new TMM registers on Intel Sapphire Rapids CPUs. Those are much larger than ZMM:
 *
 *  - BF16 tiles: 16 rows × 32 elements = 512 BF16 values = 1KB per tile
 *  - INT8 tiles: 16 rows × 64 elements = 1024 INT8 values = 1KB per tile
 *
 *  We typically use 4 registers for the 2 × 2 tile output for the matrix C accumulators, leaving
 *  4 other registers for parts of A and B matrices:
 *
 *  - TMM0, TMM1: A matrix tiles (row blocks i and i+16)
 *  - TMM2, TMM3: B matrix tiles (column blocks j and j+16)
 *  - TMM4-7: C accumulator tiles (2 × 2 output grid)
 *
 *  In most synthetic benchmarks there seems to be no mahor difference between aggregating into 1 or 4
 *  output tiles, implying the CPU's ability to internally pipeline the accumulation; so using 2 × 2 for
 *  ouputs is more of memory-bandwidth saving measure.
 *
 *  Lacking High Bandwidth Mememory, the performance in GEMM-like BLAS workloads is dominated by memory
 *  bandwidth. Latency hiding is also extremely hard, heavily affecting performance numbers. For reference,
 *  Intel MKL SGEMM for FP32 inputs yeilds arounf 250 GigaOPS per core on Intel Sapphire Rapids, leveraging
 *  AVX-512. At the same time, for AMX:
 *
 *  - BF16 peak: ≈ 3 TeraOPS per core in theory, ≈ 500 GigaOPS per core in practice
 *  - INT8 peak: ≈ 6 TeraOPS per core in theory, ≈ 1000 GigaOPS per core in practice
 *
 *  Several optimizations are used across file:
 *
 *  - Pre-pack B matrix once for repeated inference (avoids runtime reordering)
 *  - Morton Z-curve tile ordering improves L2 cache hit rate by 5-25%
 *  - Use streaming stores for large C matrices to avoid cache pollution
 *
 *  @section amx_instructions Intel AMX Instructions (Sapphire Rapids+)
 *
 *  Tile configuration and data movement:
 *
 *      Intrinsic                   Instruction                     Notes
 *      _tile_loadconfig            LDTILECFG (mem64)               Configure tile palette
 *      _tile_loadd                 TILELOADD (TMM, mem, stride)    Load tile from memory
 *      _tile_stored                TILESTORED (mem, TMM, stride)   Store tile to memory
 *      _tile_zero                  TILEZERO (TMM)                  Zero a tile register
 *
 *  BF16 matrix multiply (AMX-BF16):
 *
 *      Intrinsic                   Instruction                     Operation
 *      _tile_dpbf16ps              TDPBF16PS (TMM, TMM, TMM)       C += A × B (bf16 → f32)
 *
 *  INT8 matrix multiply (AMX-INT8):
 *
 *      Intrinsic                   Instruction                     Operation
 *      _tile_dpbssd                TDPBSSD (TMM, TMM, TMM)         C += A × B (i8 × i8 → i32)
 *      _tile_dpbsud                TDPBSUD (TMM, TMM, TMM)         C += A × B (i8 × u8 → i32)
 *      _tile_dpbusd                TDPBUSD (TMM, TMM, TMM)         C += A × B (u8 × i8 → i32)
 *      _tile_dpbuud                TDPBUUD (TMM, TMM, TMM)         C += A × B (u8 × u8 → u32)
 *
 *  AMX performance characteristics:
 *  - TDPBF16PS: 16 × 16 × 32 = 8192 BF16 MACs per instruction
 *  - TDPBSSD: 16 × 16 × 64 = 16384 INT8 MACs per instruction
 *  - Tile load latency is ~20-30 cycles; software pipelining essential
 *  - PDEP/PEXT used for Morton Z-curve encoding (BMI2): 2-3cy @ p1
 */
#ifndef NK_DOTS_SAPPHIRE_AMX_H
#define NK_DOTS_SAPPHIRE_AMX_H

#if NK_TARGET_X86_
#if NK_TARGET_SAPPHIRE_AMX
#if defined(__clang__)
#pragma clang attribute push(                                                                                        \
    __attribute__((                                                                                                  \
        target("avx2,avx512f,avx512vl,avx512bw,avx512dq,avx512fp16,f16c,fma,bmi,bmi2,amx-tile,amx-bf16,amx-int8"))), \
    apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "avx512bw", "avx512dq", "avx512fp16", "f16c", "fma", "bmi", "bmi2", \
                   "amx-tile", "amx-bf16", "amx-int8")
#endif

#include "numkong/types.h"
#include "numkong/cast/ice.h" // For FP8 ↔ BF16 conversions

#if defined(__cplusplus)
extern "C" {
#endif

/*  AMX-specific packed buffer header (64-byte aligned).
 *  Different from nk_dots_amx_packed_header_t as AMX uses tile-based layout.
 */
typedef struct {
    nk_u32_t full_column_tiles;  // Number of full column tiles (16 rows each)
    nk_u32_t full_depth_tiles;   // Number of depth tiles (32 cols for BF16, 64 for I8)
    nk_u32_t column_edge_rows;   // Remaining rows after full tiles (0-15)
    nk_u32_t column_edge_offset; // Byte offset to edge data region
    nk_u32_t reserved[12];       // Padding to 64 bytes
} nk_dots_amx_packed_header_t;

/*  Composable tile structures for AMX operations.
 *  These enable reusable primitives and cross-correlation (A × Aᵀ) use cases.
 */

/*  BF16 A tile: 16 rows × 32 depth-elements, row-major layout.
 *  Loaded from source matrix, used as left operand in AMX multiply.
 */
typedef struct {
    NK_ALIGN64 nk_bf16_t data[16][32]; // 16 rows × 32 cols = 1KB
} nk_dots_bf16_a16x32_sapphire_amx_t;

/*  BF16 B tile: 32 depth × 16 columns, pair-interleaved for TDPBF16PS.
 *  Access pattern: data[depth/2][column][depth%2] for logical B[depth, column].
 *  Pre-packed from column-major or transposed source.
 */
typedef struct {
    NK_ALIGN64 nk_bf16_t data[16][16][2]; // 16 depth-groups × 16 columns × 2 = 1KB
} nk_dots_bf16_b32x16_sapphire_amx_t;

/*  BF16 output state: 16 × 16 F32 accumulator tile.
 *  Holds partial sums during depth-dimension accumulation.
 */
typedef struct {
    NK_ALIGN64 nk_f32_t data[16][16]; // 16 × 16 = 1KB
} nk_dots_bf16_state_sapphire_amx_t;

/*  INT8 A tile: 16 rows × 64 depth-elements, row-major layout.
 */
typedef struct {
    NK_ALIGN64 nk_i8_t data[16][64]; // 16 rows × 64 cols = 1KB
} nk_dots_i8_a16x64_sapphire_amx_t;

/*  INT8 B tile: 64 depth × 16 columns, quad-interleaved for TDPBSSD.
 *  Access pattern: data[depth/4][column][depth%4] for logical B[depth, column].
 */
typedef struct {
    NK_ALIGN64 nk_i8_t data[16][16][4]; // 16 depth-groups × 16 columns × 4 = 1KB
} nk_dots_i8_b64x16_sapphire_amx_t;

/*  INT8 output state: 16 × 16 I32 accumulator tile.
 */
typedef struct {
    NK_ALIGN64 nk_i32_t data[16][16]; // 16 × 16 = 1KB
} nk_dots_i8_state_sapphire_amx_t;

/*  BF16 2 × 2 output state: 32 × 32 F32 output (4 accumulator tiles).
 *  Used for GEMM's 2 × 2 output blocking pattern.
 */
typedef struct {
    nk_dots_bf16_state_sapphire_amx_t c[2][2]; // 4KB total
} nk_dots_bf16_state2x2_sapphire_amx_t;

/*  INT8 2 × 2 output state: 32 × 32 I32 output (4 accumulator tiles).
 */
typedef struct {
    nk_dots_i8_state_sapphire_amx_t c[2][2]; // 4KB total
} nk_dots_i8_state2x2_sapphire_amx_t;

/*  UINT8 A tile: 16 rows × 64 depth-elements, row-major layout.
 *  Same layout as I8, different interpretation of signed vs unsigned.
 */
typedef struct {
    NK_ALIGN64 nk_u8_t data[16][64]; // 16 rows × 64 cols = 1KB
} nk_dots_u8_a16x64_sapphire_amx_t;

/*  UINT8 B tile: 64 depth × 16 columns, quad-interleaved for TDPBUUD.
 */
typedef struct {
    NK_ALIGN64 nk_u8_t data[16][16][4]; // 16 depth-groups × 16 columns × 4 = 1KB
} nk_dots_u8_b64x16_sapphire_amx_t;

/*  UINT8 output state: 16 × 16 U32 accumulator tile.
 */
typedef struct {
    NK_ALIGN64 nk_u32_t data[16][16]; // 16 × 16 = 1KB
} nk_dots_u8_state_sapphire_amx_t;

/*  UINT8 2 × 2 output state: 32 × 32 U32 output (4 accumulator tiles).
 */
typedef struct {
    nk_dots_u8_state_sapphire_amx_t c[2][2]; // 4KB total
} nk_dots_u8_state2x2_sapphire_amx_t;

/*  Morton Z-curve encoding for cache-friendly tile traversal.
 *  Uses BMI2 PDEP instruction for fast (2-3 cycle) bit interleaving.
 *  Interleaves bits of (tile_row, tile_col) to produce Z-curve index.
 */
NK_INTERNAL nk_u64_t nk_morton_encode_sapphire_amx_(nk_u32_t tile_row, nk_u32_t tile_col) {
    return _pdep_u64(tile_row, 0x5555555555555555ULL) | _pdep_u64(tile_col, 0xAAAAAAAAAAAAAAAAULL);
}

/*  Configure AMX tile registers.
 *  Called once per kernel invocation (idempotent within a thread).
 *  Sets all 8 tiles to standard 16 rows × 64 bytes layout.
 *
 *  Note: OS permission for AMX must be requested before using AMX instructions.
 *  Call `nk_configure_thread(nk_capabilities())` once per thread
 *  before using any Sapphire matmul functions.
 */
NK_INTERNAL void nk_amx_tile_configure_sapphire_amx_(void) {
    NK_ALIGN64 nk_u8_t tile_config[64] = {0};
    tile_config[0] = 1; // palette 1 (standard tile configuration)

    nk_u16_t *bytes_per_row = (nk_u16_t *)&tile_config[16];
    nk_u8_t *rows_per_tile = &tile_config[48];

    for (int tile_idx = 0; tile_idx < 8; tile_idx++) {
        rows_per_tile[tile_idx] = 16; // 16 rows per tile
        bytes_per_row[tile_idx] = 64; // 64 bytes per row (1KB total)
    }
    _tile_loadconfig(tile_config);
}

/*  Compiler memory barrier to ensure stores complete before AMX tile loads.
 *  AMX _tile_loadd reads from memory written by AVX-512 stores. Without this barrier,
 *  the compiler may reorder or optimize away the stores, causing _tile_loadd to read stale data.
 *  This is a compiler-only fence (no CPU fence needed - same core, same memory).
 */
NK_INTERNAL void nk_compiler_barrier_sapphire_amx_(void) { __asm__ volatile("" ::: "memory"); }

/*  Initialize BF16 output state to zero.
 */
NK_INTERNAL void nk_dots_bf16_init_sapphire_amx(nk_dots_bf16_state_sapphire_amx_t *state) {
    __m512 zero = _mm512_setzero_ps();
    for (nk_size_t row_idx = 0; row_idx < 16; row_idx++) { _mm512_store_ps(state->data[row_idx], zero); }
}

/*  Load A tile from row-major source with masking for edge tiles.
 */
NK_INTERNAL void nk_dots_bf16_load_a_sapphire_amx(       //
    nk_dots_bf16_a16x32_sapphire_amx_t *a_tile,          //
    nk_bf16_t const *src, nk_size_t src_stride_elements, //
    nk_size_t valid_rows, nk_size_t valid_cols) {

    __mmask32 col_mask = (valid_cols >= 32) ? 0xFFFFFFFF : ((__mmask32)1 << valid_cols) - 1;
    __m512i zero = _mm512_setzero_si512();

    for (nk_size_t row_idx = 0; row_idx < 16; row_idx++) {
        if (row_idx < valid_rows) {
            __m512i row = _mm512_maskz_loadu_epi16(col_mask, src + row_idx * src_stride_elements);
            _mm512_store_si512((__m512i *)a_tile->data[row_idx], row);
        }
        else { _mm512_store_si512((__m512i *)a_tile->data[row_idx], zero); }
    }
    nk_compiler_barrier_sapphire_amx_();
}

/*  Store state to output matrix with masking for edge tiles.
 */
NK_INTERNAL void nk_dots_bf16_store_sapphire_amx(   //
    nk_dots_bf16_state_sapphire_amx_t const *state, //
    nk_f32_t *dst, nk_size_t dst_stride_elements,   //
    nk_size_t valid_rows, nk_size_t valid_cols) {

    __mmask16 col_mask = (valid_cols >= 16) ? 0xFFFF : ((__mmask16)1 << valid_cols) - 1;

    for (nk_size_t row_idx = 0; row_idx < valid_rows; row_idx++) {
        __m512 row = _mm512_load_ps(state->data[row_idx]);
        _mm512_mask_storeu_ps(dst + row_idx * dst_stride_elements, col_mask, row);
    }
}

/*  Accumulate 3 A × B tile pairs into state using AMX TDPBF16PS.
 *  Processes depth = 96 per call (3 × 32 BF16 elements).
 *  For indivisible depth, pad unused tiles with zeros.
 *
 *  Register allocation (uses 7 of 8 TMM registers):
 *    TMM0: accumulator (C)
 *    TMM1-3: A tiles (a_tile_0, a_tile_1, a_tile_2)
 *    TMM4-6: B tiles (b_tile_0, b_tile_1, b_tile_2)
 *    TMM7: spare
 *
 *  Based on empirical testing, single-accumulator design achieves ~100% of
 *  multi-accumulator throughput due to AMX internal pipelining.
 */
NK_INTERNAL void nk_dots_bf16_update_sapphire_amx(      //
    nk_dots_bf16_state_sapphire_amx_t *state,           //
    nk_dots_bf16_a16x32_sapphire_amx_t const *a_tile_0, //
    nk_dots_bf16_a16x32_sapphire_amx_t const *a_tile_1, //
    nk_dots_bf16_a16x32_sapphire_amx_t const *a_tile_2, //
    nk_dots_bf16_b32x16_sapphire_amx_t const *b_tile_0, //
    nk_dots_bf16_b32x16_sapphire_amx_t const *b_tile_1, //
    nk_dots_bf16_b32x16_sapphire_amx_t const *b_tile_2) {

    // Load all tiles into registers
    _tile_loadd(0, state->data, 64);    // C accumulator
    _tile_loadd(1, a_tile_0->data, 64); // A0
    _tile_loadd(2, a_tile_1->data, 64); // A1
    _tile_loadd(3, a_tile_2->data, 64); // A2
    _tile_loadd(4, b_tile_0->data, 64); // B0
    _tile_loadd(5, b_tile_1->data, 64); // B1
    _tile_loadd(6, b_tile_2->data, 64); // B2

    // Accumulate: C += A0 × B0 + A1 × B1 + A2 × B2
    _tile_dpbf16ps(0, 1, 4); // C += A0 × B0
    _tile_dpbf16ps(0, 2, 5); // C += A1 × B1
    _tile_dpbf16ps(0, 3, 6); // C += A2 × B2

    // Store result
    _tile_stored(0, state->data, 64);
}

/*  Initialize INT8 output state to zero.
 */
NK_INTERNAL void nk_dots_i8_init_sapphire_amx(nk_dots_i8_state_sapphire_amx_t *state) {
    __m512i zero = _mm512_setzero_si512();
    for (nk_size_t row_idx = 0; row_idx < 16; row_idx++) { _mm512_store_si512((__m512i *)state->data[row_idx], zero); }
}

/*  Load A tile from row-major source with masking for edge tiles.
 */
NK_INTERNAL void nk_dots_i8_load_a_sapphire_amx( //
    nk_dots_i8_a16x64_sapphire_amx_t *a_tile,    //
    nk_i8_t const *src, nk_size_t src_stride,    //
    nk_size_t valid_rows, nk_size_t valid_cols) {

    __mmask64 col_mask = (valid_cols >= 64) ? 0xFFFFFFFFFFFFFFFFULL : ((__mmask64)1 << valid_cols) - 1;
    __m512i zero = _mm512_setzero_si512();

    for (nk_size_t row_idx = 0; row_idx < 16; row_idx++) {
        if (row_idx < valid_rows) {
            __m512i row = _mm512_maskz_loadu_epi8(col_mask, src + row_idx * src_stride);
            _mm512_store_si512((__m512i *)a_tile->data[row_idx], row);
        }
        else { _mm512_store_si512((__m512i *)a_tile->data[row_idx], zero); }
    }
    nk_compiler_barrier_sapphire_amx_();
}

/*  Store state to output matrix with masking for edge tiles.
 */
NK_INTERNAL void nk_dots_i8_store_sapphire_amx(   //
    nk_dots_i8_state_sapphire_amx_t const *state, //
    nk_i32_t *dst, nk_size_t dst_stride_elements, //
    nk_size_t valid_rows, nk_size_t valid_cols) {

    __mmask16 col_mask = (valid_cols >= 16) ? 0xFFFF : ((__mmask16)1 << valid_cols) - 1;

    for (nk_size_t row_idx = 0; row_idx < valid_rows; row_idx++) {
        __m512i row = _mm512_load_si512((__m512i const *)state->data[row_idx]);
        _mm512_mask_storeu_epi32(dst + row_idx * dst_stride_elements, col_mask, row);
    }
}

/*  Accumulate 3 A × B tile pairs into state using AMX TDPBSSD.
 *  Processes depth = 192 per call (3 × 64 INT8 elements).
 *  For indivisible depth, pad unused tiles with zeros.
 *
 *  Register allocation (uses 7 of 8 TMM registers):
 *    TMM0: accumulator (C)
 *    TMM1-3: A tiles (a_tile_0, a_tile_1, a_tile_2)
 *    TMM4-6: B tiles (b_tile_0, b_tile_1, b_tile_2)
 *    TMM7: spare
 */
NK_INTERNAL void nk_dots_i8_update_sapphire_amx(      //
    nk_dots_i8_state_sapphire_amx_t *state,           //
    nk_dots_i8_a16x64_sapphire_amx_t const *a_tile_0, //
    nk_dots_i8_a16x64_sapphire_amx_t const *a_tile_1, //
    nk_dots_i8_a16x64_sapphire_amx_t const *a_tile_2, //
    nk_dots_i8_b64x16_sapphire_amx_t const *b_tile_0, //
    nk_dots_i8_b64x16_sapphire_amx_t const *b_tile_1, //
    nk_dots_i8_b64x16_sapphire_amx_t const *b_tile_2) {

    // Load all tiles into registers
    _tile_loadd(0, state->data, 64);    // C accumulator
    _tile_loadd(1, a_tile_0->data, 64); // A0
    _tile_loadd(2, a_tile_1->data, 64); // A1
    _tile_loadd(3, a_tile_2->data, 64); // A2
    _tile_loadd(4, b_tile_0->data, 64); // B0
    _tile_loadd(5, b_tile_1->data, 64); // B1
    _tile_loadd(6, b_tile_2->data, 64); // B2

    // Accumulate: C += A0 × B0 + A1 × B1 + A2 × B2
    _tile_dpbssd(0, 1, 4); // C += A0 × B0
    _tile_dpbssd(0, 2, 5); // C += A1 × B1
    _tile_dpbssd(0, 3, 6); // C += A2 × B2

    // Store result
    _tile_stored(0, state->data, 64);
}

/*  Store BF16 2 × 2 state to output matrix with masking for edge tiles.
 *  Handles any combination of valid_rows (0-32) and valid_cols (0-32).
 */
NK_INTERNAL void nk_dots_bf16_output2x2_sapphire_amx(  //
    nk_dots_bf16_state2x2_sapphire_amx_t const *state, //
    nk_f32_t *dst, nk_size_t dst_stride_elements,      //
    nk_size_t valid_rows, nk_size_t valid_cols) {

    // Rows 0-15
    nk_size_t const rows_upper = (valid_rows > 16) ? 16 : valid_rows;
    nk_size_t const cols_left = (valid_cols > 16) ? 16 : valid_cols;
    nk_size_t const cols_right = (valid_cols > 16) ? valid_cols - 16 : 0;

    if (rows_upper > 0 && cols_left > 0)
        nk_dots_bf16_store_sapphire_amx(&state->c[0][0], dst, dst_stride_elements, rows_upper, cols_left);
    if (rows_upper > 0 && cols_right > 0)
        nk_dots_bf16_store_sapphire_amx(&state->c[0][1], dst + 16, dst_stride_elements, rows_upper, cols_right);

    // Rows 16-31
    if (valid_rows > 16) {
        nk_size_t const rows_lower = valid_rows - 16;
        nk_f32_t *dst_lower = dst + 16 * dst_stride_elements;
        if (cols_left > 0)
            nk_dots_bf16_store_sapphire_amx(&state->c[1][0], dst_lower, dst_stride_elements, rows_lower, cols_left);
        if (cols_right > 0)
            nk_dots_bf16_store_sapphire_amx(&state->c[1][1], dst_lower + 16, dst_stride_elements, rows_lower,
                                            cols_right);
    }
}

/*  Store INT8 2 × 2 state to output matrix with masking for edge tiles.
 */
NK_INTERNAL void nk_dots_i8_output2x2_sapphire_amx(  //
    nk_dots_i8_state2x2_sapphire_amx_t const *state, //
    nk_i32_t *dst, nk_size_t dst_stride_elements,    //
    nk_size_t valid_rows, nk_size_t valid_cols) {

    nk_size_t const rows_upper = (valid_rows > 16) ? 16 : valid_rows;
    nk_size_t const cols_left = (valid_cols > 16) ? 16 : valid_cols;
    nk_size_t const cols_right = (valid_cols > 16) ? valid_cols - 16 : 0;

    if (rows_upper > 0 && cols_left > 0)
        nk_dots_i8_store_sapphire_amx(&state->c[0][0], dst, dst_stride_elements, rows_upper, cols_left);
    if (rows_upper > 0 && cols_right > 0)
        nk_dots_i8_store_sapphire_amx(&state->c[0][1], dst + 16, dst_stride_elements, rows_upper, cols_right);

    if (valid_rows > 16) {
        nk_size_t const rows_lower = valid_rows - 16;
        nk_i32_t *dst_lower = dst + 16 * dst_stride_elements;
        if (cols_left > 0)
            nk_dots_i8_store_sapphire_amx(&state->c[1][0], dst_lower, dst_stride_elements, rows_lower, cols_left);
        if (cols_right > 0)
            nk_dots_i8_store_sapphire_amx(&state->c[1][1], dst_lower + 16, dst_stride_elements, rows_lower, cols_right);
    }
}

/*  Initialize UINT8 output state to zero.
 */
NK_INTERNAL void nk_dots_u8_init_sapphire_amx(nk_dots_u8_state_sapphire_amx_t *state) {
    nk_dots_i8_init_sapphire_amx((nk_dots_i8_state_sapphire_amx_t *)state);
}

/*  Load U8 A tile from row-major source with masking for edge tiles.
 */
NK_INTERNAL void nk_dots_u8_load_a_sapphire_amx( //
    nk_dots_u8_a16x64_sapphire_amx_t *a_tile,    //
    nk_u8_t const *src, nk_size_t src_stride,    //
    nk_size_t valid_rows, nk_size_t valid_cols) {
    nk_dots_i8_load_a_sapphire_amx(                   //
        (nk_dots_i8_a16x64_sapphire_amx_t *)a_tile,   //
        (nk_i8_t const *)src, src_stride, valid_rows, valid_cols);
}

/*  Store U8 state to output matrix with masking for edge tiles.
 */
NK_INTERNAL void nk_dots_u8_store_sapphire_amx(   //
    nk_dots_u8_state_sapphire_amx_t const *state, //
    nk_u32_t *dst, nk_size_t dst_stride_elements, //
    nk_size_t valid_rows, nk_size_t valid_cols) {
    nk_dots_i8_store_sapphire_amx(                    //
        (nk_dots_i8_state_sapphire_amx_t const *)state, //
        (nk_i32_t *)dst, dst_stride_elements, valid_rows, valid_cols);
}

/*  Store UINT8 2 × 2 state to output matrix with masking for edge tiles.
 */
NK_INTERNAL void nk_dots_u8_output2x2_sapphire_amx(  //
    nk_dots_u8_state2x2_sapphire_amx_t const *state, //
    nk_u32_t *dst, nk_size_t dst_stride_elements,    //
    nk_size_t valid_rows, nk_size_t valid_cols) {
    nk_dots_i8_output2x2_sapphire_amx(                    //
        (nk_dots_i8_state2x2_sapphire_amx_t const *)state, //
        (nk_i32_t *)dst, dst_stride_elements, valid_rows, valid_cols);
}

/*  Pack U8 A transposed into B format.
 *  Converts A[row][depth] → B[depth_group][col][quad] with quad-interleaving for TDPBUUD.
 *
 *  Shuffle-only implementation (no gather instructions):
 *  Since quads (d0,d1,d2,d3) are already adjacent in each row as 32-bit elements,
 *  this is effectively a 16×16 transpose of 32-bit elements.
 */
NK_INTERNAL void nk_dots_pack_u8_transposed_sapphire_amx( //
    nk_dots_u8_a16x64_sapphire_amx_t const *a_tile,       //
    nk_dots_u8_b64x16_sapphire_amx_t *b_tile) {

    // Load all 16 rows - each row is 64 UINT8 = 64 bytes = 1 ZMM
    // Treat as 16 × 32-bit elements per row (each 32-bit = quad of UINT8)
    __m512i row00 = _mm512_load_si512(&a_tile->data[0][0]);
    __m512i row01 = _mm512_load_si512(&a_tile->data[1][0]);
    __m512i row02 = _mm512_load_si512(&a_tile->data[2][0]);
    __m512i row03 = _mm512_load_si512(&a_tile->data[3][0]);
    __m512i row04 = _mm512_load_si512(&a_tile->data[4][0]);
    __m512i row05 = _mm512_load_si512(&a_tile->data[5][0]);
    __m512i row06 = _mm512_load_si512(&a_tile->data[6][0]);
    __m512i row07 = _mm512_load_si512(&a_tile->data[7][0]);
    __m512i row08 = _mm512_load_si512(&a_tile->data[8][0]);
    __m512i row09 = _mm512_load_si512(&a_tile->data[9][0]);
    __m512i row10 = _mm512_load_si512(&a_tile->data[10][0]);
    __m512i row11 = _mm512_load_si512(&a_tile->data[11][0]);
    __m512i row12 = _mm512_load_si512(&a_tile->data[12][0]);
    __m512i row13 = _mm512_load_si512(&a_tile->data[13][0]);
    __m512i row14 = _mm512_load_si512(&a_tile->data[14][0]);
    __m512i row15 = _mm512_load_si512(&a_tile->data[15][0]);

    // 16×16 transpose of 32-bit elements using hierarchical unpacks
    // Stage 1: Unpack adjacent row pairs at 32-bit granularity
    __m512i t01_lo = _mm512_unpacklo_epi32(row00, row01);
    __m512i t01_hi = _mm512_unpackhi_epi32(row00, row01);
    __m512i t23_lo = _mm512_unpacklo_epi32(row02, row03);
    __m512i t23_hi = _mm512_unpackhi_epi32(row02, row03);
    __m512i t45_lo = _mm512_unpacklo_epi32(row04, row05);
    __m512i t45_hi = _mm512_unpackhi_epi32(row04, row05);
    __m512i t67_lo = _mm512_unpacklo_epi32(row06, row07);
    __m512i t67_hi = _mm512_unpackhi_epi32(row06, row07);
    __m512i t89_lo = _mm512_unpacklo_epi32(row08, row09);
    __m512i t89_hi = _mm512_unpackhi_epi32(row08, row09);
    __m512i tab_lo = _mm512_unpacklo_epi32(row10, row11);
    __m512i tab_hi = _mm512_unpackhi_epi32(row10, row11);
    __m512i tcd_lo = _mm512_unpacklo_epi32(row12, row13);
    __m512i tcd_hi = _mm512_unpackhi_epi32(row12, row13);
    __m512i tef_lo = _mm512_unpacklo_epi32(row14, row15);
    __m512i tef_hi = _mm512_unpackhi_epi32(row14, row15);

    // Stage 2: Unpack at 64-bit granularity
    __m512i u0123_ll = _mm512_unpacklo_epi64(t01_lo, t23_lo);
    __m512i u0123_lh = _mm512_unpackhi_epi64(t01_lo, t23_lo);
    __m512i u0123_hl = _mm512_unpacklo_epi64(t01_hi, t23_hi);
    __m512i u0123_hh = _mm512_unpackhi_epi64(t01_hi, t23_hi);
    __m512i u4567_ll = _mm512_unpacklo_epi64(t45_lo, t67_lo);
    __m512i u4567_lh = _mm512_unpackhi_epi64(t45_lo, t67_lo);
    __m512i u4567_hl = _mm512_unpacklo_epi64(t45_hi, t67_hi);
    __m512i u4567_hh = _mm512_unpackhi_epi64(t45_hi, t67_hi);
    __m512i u89ab_ll = _mm512_unpacklo_epi64(t89_lo, tab_lo);
    __m512i u89ab_lh = _mm512_unpackhi_epi64(t89_lo, tab_lo);
    __m512i u89ab_hl = _mm512_unpacklo_epi64(t89_hi, tab_hi);
    __m512i u89ab_hh = _mm512_unpackhi_epi64(t89_hi, tab_hi);
    __m512i ucdef_ll = _mm512_unpacklo_epi64(tcd_lo, tef_lo);
    __m512i ucdef_lh = _mm512_unpackhi_epi64(tcd_lo, tef_lo);
    __m512i ucdef_hl = _mm512_unpacklo_epi64(tcd_hi, tef_hi);
    __m512i ucdef_hh = _mm512_unpackhi_epi64(tcd_hi, tef_hi);

    // Stage 3: Shuffle 128-bit lanes
    __m512i v0_a = _mm512_shuffle_i32x4(u0123_ll, u4567_ll, 0x88);
    __m512i v0_b = _mm512_shuffle_i32x4(u0123_ll, u4567_ll, 0xDD);
    __m512i v1_a = _mm512_shuffle_i32x4(u0123_lh, u4567_lh, 0x88);
    __m512i v1_b = _mm512_shuffle_i32x4(u0123_lh, u4567_lh, 0xDD);
    __m512i v2_a = _mm512_shuffle_i32x4(u0123_hl, u4567_hl, 0x88);
    __m512i v2_b = _mm512_shuffle_i32x4(u0123_hl, u4567_hl, 0xDD);
    __m512i v3_a = _mm512_shuffle_i32x4(u0123_hh, u4567_hh, 0x88);
    __m512i v3_b = _mm512_shuffle_i32x4(u0123_hh, u4567_hh, 0xDD);
    __m512i v4_a = _mm512_shuffle_i32x4(u89ab_ll, ucdef_ll, 0x88);
    __m512i v4_b = _mm512_shuffle_i32x4(u89ab_ll, ucdef_ll, 0xDD);
    __m512i v5_a = _mm512_shuffle_i32x4(u89ab_lh, ucdef_lh, 0x88);
    __m512i v5_b = _mm512_shuffle_i32x4(u89ab_lh, ucdef_lh, 0xDD);
    __m512i v6_a = _mm512_shuffle_i32x4(u89ab_hl, ucdef_hl, 0x88);
    __m512i v6_b = _mm512_shuffle_i32x4(u89ab_hl, ucdef_hl, 0xDD);
    __m512i v7_a = _mm512_shuffle_i32x4(u89ab_hh, ucdef_hh, 0x88);
    __m512i v7_b = _mm512_shuffle_i32x4(u89ab_hh, ucdef_hh, 0xDD);

    // Stage 4: Final 256-bit shuffle to complete transpose
    __m512i out00 = _mm512_shuffle_i32x4(v0_a, v4_a, 0x88);
    __m512i out01 = _mm512_shuffle_i32x4(v1_a, v5_a, 0x88);
    __m512i out02 = _mm512_shuffle_i32x4(v2_a, v6_a, 0x88);
    __m512i out03 = _mm512_shuffle_i32x4(v3_a, v7_a, 0x88);
    __m512i out04 = _mm512_shuffle_i32x4(v0_a, v4_a, 0xDD);
    __m512i out05 = _mm512_shuffle_i32x4(v1_a, v5_a, 0xDD);
    __m512i out06 = _mm512_shuffle_i32x4(v2_a, v6_a, 0xDD);
    __m512i out07 = _mm512_shuffle_i32x4(v3_a, v7_a, 0xDD);
    __m512i out08 = _mm512_shuffle_i32x4(v0_b, v4_b, 0x88);
    __m512i out09 = _mm512_shuffle_i32x4(v1_b, v5_b, 0x88);
    __m512i out10 = _mm512_shuffle_i32x4(v2_b, v6_b, 0x88);
    __m512i out11 = _mm512_shuffle_i32x4(v3_b, v7_b, 0x88);
    __m512i out12 = _mm512_shuffle_i32x4(v0_b, v4_b, 0xDD);
    __m512i out13 = _mm512_shuffle_i32x4(v1_b, v5_b, 0xDD);
    __m512i out14 = _mm512_shuffle_i32x4(v2_b, v6_b, 0xDD);
    __m512i out15 = _mm512_shuffle_i32x4(v3_b, v7_b, 0xDD);

    // Store transposed results - each output row is one depth_group
    // Output layout: B.data[depth_group][col][quad] = 16 cols × 4 UINT8 = 64 bytes
    _mm512_store_si512(&b_tile->data[0][0][0], out00);
    _mm512_store_si512(&b_tile->data[1][0][0], out01);
    _mm512_store_si512(&b_tile->data[2][0][0], out02);
    _mm512_store_si512(&b_tile->data[3][0][0], out03);
    _mm512_store_si512(&b_tile->data[4][0][0], out04);
    _mm512_store_si512(&b_tile->data[5][0][0], out05);
    _mm512_store_si512(&b_tile->data[6][0][0], out06);
    _mm512_store_si512(&b_tile->data[7][0][0], out07);
    _mm512_store_si512(&b_tile->data[8][0][0], out08);
    _mm512_store_si512(&b_tile->data[9][0][0], out09);
    _mm512_store_si512(&b_tile->data[10][0][0], out10);
    _mm512_store_si512(&b_tile->data[11][0][0], out11);
    _mm512_store_si512(&b_tile->data[12][0][0], out12);
    _mm512_store_si512(&b_tile->data[13][0][0], out13);
    _mm512_store_si512(&b_tile->data[14][0][0], out14);
    _mm512_store_si512(&b_tile->data[15][0][0], out15);

    nk_compiler_barrier_sapphire_amx_();
}

/*  Load E4M3 A tile with FP8 → BF16 conversion.
 *  E4M3 and BF16 both use 32 elements per tile row (same depth granularity).
 *  Converts each row of 32 E4M3 bytes to 32 BF16 elements.
 */
NK_INTERNAL void nk_dots_e4m3_load_a_sapphire_amx( //
    nk_dots_bf16_a16x32_sapphire_amx_t *a_tile,    //
    nk_e4m3_t const *src, nk_size_t src_stride,    //
    nk_size_t valid_rows, nk_size_t valid_cols) {

    __mmask32 col_mask = (valid_cols >= 32) ? 0xFFFFFFFF : ((__mmask32)1 << valid_cols) - 1;
    __m512i zero = _mm512_setzero_si512();

    for (nk_size_t row_idx = 0; row_idx < 16; row_idx++) {
        if (row_idx < valid_rows) {
            // Load 32 E4M3 bytes with masking
            __m256i e4m3_row = _mm256_maskz_loadu_epi8(col_mask, src + row_idx * src_stride);
            // Convert to 32 BF16 values
            __m512i bf16_row = nk_e4m3x32_to_bf16x32_ice_(e4m3_row);
            _mm512_store_si512((__m512i *)a_tile->data[row_idx], bf16_row);
        }
        else { _mm512_store_si512((__m512i *)a_tile->data[row_idx], zero); }
    }
    nk_compiler_barrier_sapphire_amx_();
}

/*  Load E5M2 A tile with FP8 → BF16 conversion.
 */
NK_INTERNAL void nk_dots_e5m2_load_a_sapphire_amx( //
    nk_dots_bf16_a16x32_sapphire_amx_t *a_tile,    //
    nk_e5m2_t const *src, nk_size_t src_stride,    //
    nk_size_t valid_rows, nk_size_t valid_cols) {

    __mmask32 col_mask = (valid_cols >= 32) ? 0xFFFFFFFF : ((__mmask32)1 << valid_cols) - 1;
    __m512i zero = _mm512_setzero_si512();

    for (nk_size_t row_idx = 0; row_idx < 16; row_idx++) {
        if (row_idx < valid_rows) {
            __m256i e5m2_row = _mm256_maskz_loadu_epi8(col_mask, src + row_idx * src_stride);
            __m512i bf16_row = nk_e5m2x32_to_bf16x32_ice_(e5m2_row);
            _mm512_store_si512((__m512i *)a_tile->data[row_idx], bf16_row);
        }
        else { _mm512_store_si512((__m512i *)a_tile->data[row_idx], zero); }
    }
    nk_compiler_barrier_sapphire_amx_();
}

/*  Pack A transposed into B format for BF16.
 *  Converts A[row][depth] → B[depth_group][col][pair] with pair-interleaving for TDPBF16PS.
 *  Used for cross-correlation: load vectors into A, pack to B, compute A × B.
 *
 *  Shuffle-only implementation (no gather instructions):
 *  Since pairs (d0,d1), (d2,d3), ... are already adjacent in each row,
 *  this is effectively a 16×16 transpose of 32-bit elements.
 */
NK_INTERNAL void nk_dots_pack_bf16_transposed_sapphire_amx( //
    nk_dots_bf16_a16x32_sapphire_amx_t const *a_tile,       //
    nk_dots_bf16_b32x16_sapphire_amx_t *b_tile) {

    // Load all 16 rows - each row is 32 BF16 = 64 bytes = 1 ZMM
    // Treat as 16 × 32-bit elements per row (each 32-bit = pair of BF16)
    __m512i row00 = _mm512_load_si512(&a_tile->data[0][0]);
    __m512i row01 = _mm512_load_si512(&a_tile->data[1][0]);
    __m512i row02 = _mm512_load_si512(&a_tile->data[2][0]);
    __m512i row03 = _mm512_load_si512(&a_tile->data[3][0]);
    __m512i row04 = _mm512_load_si512(&a_tile->data[4][0]);
    __m512i row05 = _mm512_load_si512(&a_tile->data[5][0]);
    __m512i row06 = _mm512_load_si512(&a_tile->data[6][0]);
    __m512i row07 = _mm512_load_si512(&a_tile->data[7][0]);
    __m512i row08 = _mm512_load_si512(&a_tile->data[8][0]);
    __m512i row09 = _mm512_load_si512(&a_tile->data[9][0]);
    __m512i row10 = _mm512_load_si512(&a_tile->data[10][0]);
    __m512i row11 = _mm512_load_si512(&a_tile->data[11][0]);
    __m512i row12 = _mm512_load_si512(&a_tile->data[12][0]);
    __m512i row13 = _mm512_load_si512(&a_tile->data[13][0]);
    __m512i row14 = _mm512_load_si512(&a_tile->data[14][0]);
    __m512i row15 = _mm512_load_si512(&a_tile->data[15][0]);

    // 16×16 transpose of 32-bit elements using hierarchical unpacks
    // Stage 1: Unpack adjacent row pairs at 32-bit granularity
    __m512i t01_lo = _mm512_unpacklo_epi32(row00, row01);
    __m512i t01_hi = _mm512_unpackhi_epi32(row00, row01);
    __m512i t23_lo = _mm512_unpacklo_epi32(row02, row03);
    __m512i t23_hi = _mm512_unpackhi_epi32(row02, row03);
    __m512i t45_lo = _mm512_unpacklo_epi32(row04, row05);
    __m512i t45_hi = _mm512_unpackhi_epi32(row04, row05);
    __m512i t67_lo = _mm512_unpacklo_epi32(row06, row07);
    __m512i t67_hi = _mm512_unpackhi_epi32(row06, row07);
    __m512i t89_lo = _mm512_unpacklo_epi32(row08, row09);
    __m512i t89_hi = _mm512_unpackhi_epi32(row08, row09);
    __m512i tab_lo = _mm512_unpacklo_epi32(row10, row11);
    __m512i tab_hi = _mm512_unpackhi_epi32(row10, row11);
    __m512i tcd_lo = _mm512_unpacklo_epi32(row12, row13);
    __m512i tcd_hi = _mm512_unpackhi_epi32(row12, row13);
    __m512i tef_lo = _mm512_unpacklo_epi32(row14, row15);
    __m512i tef_hi = _mm512_unpackhi_epi32(row14, row15);

    // Stage 2: Unpack at 64-bit granularity
    __m512i u0123_ll = _mm512_unpacklo_epi64(t01_lo, t23_lo);
    __m512i u0123_lh = _mm512_unpackhi_epi64(t01_lo, t23_lo);
    __m512i u0123_hl = _mm512_unpacklo_epi64(t01_hi, t23_hi);
    __m512i u0123_hh = _mm512_unpackhi_epi64(t01_hi, t23_hi);
    __m512i u4567_ll = _mm512_unpacklo_epi64(t45_lo, t67_lo);
    __m512i u4567_lh = _mm512_unpackhi_epi64(t45_lo, t67_lo);
    __m512i u4567_hl = _mm512_unpacklo_epi64(t45_hi, t67_hi);
    __m512i u4567_hh = _mm512_unpackhi_epi64(t45_hi, t67_hi);
    __m512i u89ab_ll = _mm512_unpacklo_epi64(t89_lo, tab_lo);
    __m512i u89ab_lh = _mm512_unpackhi_epi64(t89_lo, tab_lo);
    __m512i u89ab_hl = _mm512_unpacklo_epi64(t89_hi, tab_hi);
    __m512i u89ab_hh = _mm512_unpackhi_epi64(t89_hi, tab_hi);
    __m512i ucdef_ll = _mm512_unpacklo_epi64(tcd_lo, tef_lo);
    __m512i ucdef_lh = _mm512_unpackhi_epi64(tcd_lo, tef_lo);
    __m512i ucdef_hl = _mm512_unpacklo_epi64(tcd_hi, tef_hi);
    __m512i ucdef_hh = _mm512_unpackhi_epi64(tcd_hi, tef_hi);

    // Stage 3: Shuffle 128-bit lanes using permute2x128 equivalent for 512-bit
    // Use shuffle_i32x4 to move 128-bit chunks
    __m512i v0_a = _mm512_shuffle_i32x4(u0123_ll, u4567_ll, 0x88); // lanes 0,2 from each
    __m512i v0_b = _mm512_shuffle_i32x4(u0123_ll, u4567_ll, 0xDD); // lanes 1,3 from each
    __m512i v1_a = _mm512_shuffle_i32x4(u0123_lh, u4567_lh, 0x88);
    __m512i v1_b = _mm512_shuffle_i32x4(u0123_lh, u4567_lh, 0xDD);
    __m512i v2_a = _mm512_shuffle_i32x4(u0123_hl, u4567_hl, 0x88);
    __m512i v2_b = _mm512_shuffle_i32x4(u0123_hl, u4567_hl, 0xDD);
    __m512i v3_a = _mm512_shuffle_i32x4(u0123_hh, u4567_hh, 0x88);
    __m512i v3_b = _mm512_shuffle_i32x4(u0123_hh, u4567_hh, 0xDD);
    __m512i v4_a = _mm512_shuffle_i32x4(u89ab_ll, ucdef_ll, 0x88);
    __m512i v4_b = _mm512_shuffle_i32x4(u89ab_ll, ucdef_ll, 0xDD);
    __m512i v5_a = _mm512_shuffle_i32x4(u89ab_lh, ucdef_lh, 0x88);
    __m512i v5_b = _mm512_shuffle_i32x4(u89ab_lh, ucdef_lh, 0xDD);
    __m512i v6_a = _mm512_shuffle_i32x4(u89ab_hl, ucdef_hl, 0x88);
    __m512i v6_b = _mm512_shuffle_i32x4(u89ab_hl, ucdef_hl, 0xDD);
    __m512i v7_a = _mm512_shuffle_i32x4(u89ab_hh, ucdef_hh, 0x88);
    __m512i v7_b = _mm512_shuffle_i32x4(u89ab_hh, ucdef_hh, 0xDD);

    // Stage 4: Final 256-bit shuffle to complete transpose
    __m512i out00 = _mm512_shuffle_i32x4(v0_a, v4_a, 0x88);
    __m512i out01 = _mm512_shuffle_i32x4(v1_a, v5_a, 0x88);
    __m512i out02 = _mm512_shuffle_i32x4(v2_a, v6_a, 0x88);
    __m512i out03 = _mm512_shuffle_i32x4(v3_a, v7_a, 0x88);
    __m512i out04 = _mm512_shuffle_i32x4(v0_a, v4_a, 0xDD);
    __m512i out05 = _mm512_shuffle_i32x4(v1_a, v5_a, 0xDD);
    __m512i out06 = _mm512_shuffle_i32x4(v2_a, v6_a, 0xDD);
    __m512i out07 = _mm512_shuffle_i32x4(v3_a, v7_a, 0xDD);
    __m512i out08 = _mm512_shuffle_i32x4(v0_b, v4_b, 0x88);
    __m512i out09 = _mm512_shuffle_i32x4(v1_b, v5_b, 0x88);
    __m512i out10 = _mm512_shuffle_i32x4(v2_b, v6_b, 0x88);
    __m512i out11 = _mm512_shuffle_i32x4(v3_b, v7_b, 0x88);
    __m512i out12 = _mm512_shuffle_i32x4(v0_b, v4_b, 0xDD);
    __m512i out13 = _mm512_shuffle_i32x4(v1_b, v5_b, 0xDD);
    __m512i out14 = _mm512_shuffle_i32x4(v2_b, v6_b, 0xDD);
    __m512i out15 = _mm512_shuffle_i32x4(v3_b, v7_b, 0xDD);

    // Store transposed results - each output row is one depth_group
    // Output layout: B.data[depth_group][col][pair] = 16 cols × 2 BF16 = 64 bytes
    _mm512_store_si512(&b_tile->data[0][0][0], out00);
    _mm512_store_si512(&b_tile->data[1][0][0], out01);
    _mm512_store_si512(&b_tile->data[2][0][0], out02);
    _mm512_store_si512(&b_tile->data[3][0][0], out03);
    _mm512_store_si512(&b_tile->data[4][0][0], out04);
    _mm512_store_si512(&b_tile->data[5][0][0], out05);
    _mm512_store_si512(&b_tile->data[6][0][0], out06);
    _mm512_store_si512(&b_tile->data[7][0][0], out07);
    _mm512_store_si512(&b_tile->data[8][0][0], out08);
    _mm512_store_si512(&b_tile->data[9][0][0], out09);
    _mm512_store_si512(&b_tile->data[10][0][0], out10);
    _mm512_store_si512(&b_tile->data[11][0][0], out11);
    _mm512_store_si512(&b_tile->data[12][0][0], out12);
    _mm512_store_si512(&b_tile->data[13][0][0], out13);
    _mm512_store_si512(&b_tile->data[14][0][0], out14);
    _mm512_store_si512(&b_tile->data[15][0][0], out15);

    nk_compiler_barrier_sapphire_amx_();
}

/*  Pack A transposed into B format for INT8.
 *  Converts A[row][depth] → B[depth_group][col][quad] with quad-interleaving for TDPBSSD.
 *
 *  Shuffle-only implementation (no gather instructions):
 *  Since quads (d0,d1,d2,d3) are already adjacent in each row as 32-bit elements,
 *  this is effectively a 16×16 transpose of 32-bit elements.
 */
NK_INTERNAL void nk_dots_pack_i8_transposed_sapphire_amx( //
    nk_dots_i8_a16x64_sapphire_amx_t const *a_tile,       //
    nk_dots_i8_b64x16_sapphire_amx_t *b_tile) {

    // Load all 16 rows - each row is 64 INT8 = 64 bytes = 1 ZMM
    // Treat as 16 × 32-bit elements per row (each 32-bit = quad of INT8)
    __m512i row00 = _mm512_load_si512(&a_tile->data[0][0]);
    __m512i row01 = _mm512_load_si512(&a_tile->data[1][0]);
    __m512i row02 = _mm512_load_si512(&a_tile->data[2][0]);
    __m512i row03 = _mm512_load_si512(&a_tile->data[3][0]);
    __m512i row04 = _mm512_load_si512(&a_tile->data[4][0]);
    __m512i row05 = _mm512_load_si512(&a_tile->data[5][0]);
    __m512i row06 = _mm512_load_si512(&a_tile->data[6][0]);
    __m512i row07 = _mm512_load_si512(&a_tile->data[7][0]);
    __m512i row08 = _mm512_load_si512(&a_tile->data[8][0]);
    __m512i row09 = _mm512_load_si512(&a_tile->data[9][0]);
    __m512i row10 = _mm512_load_si512(&a_tile->data[10][0]);
    __m512i row11 = _mm512_load_si512(&a_tile->data[11][0]);
    __m512i row12 = _mm512_load_si512(&a_tile->data[12][0]);
    __m512i row13 = _mm512_load_si512(&a_tile->data[13][0]);
    __m512i row14 = _mm512_load_si512(&a_tile->data[14][0]);
    __m512i row15 = _mm512_load_si512(&a_tile->data[15][0]);

    // 16×16 transpose of 32-bit elements using hierarchical unpacks
    // Stage 1: Unpack adjacent row pairs at 32-bit granularity
    __m512i t01_lo = _mm512_unpacklo_epi32(row00, row01);
    __m512i t01_hi = _mm512_unpackhi_epi32(row00, row01);
    __m512i t23_lo = _mm512_unpacklo_epi32(row02, row03);
    __m512i t23_hi = _mm512_unpackhi_epi32(row02, row03);
    __m512i t45_lo = _mm512_unpacklo_epi32(row04, row05);
    __m512i t45_hi = _mm512_unpackhi_epi32(row04, row05);
    __m512i t67_lo = _mm512_unpacklo_epi32(row06, row07);
    __m512i t67_hi = _mm512_unpackhi_epi32(row06, row07);
    __m512i t89_lo = _mm512_unpacklo_epi32(row08, row09);
    __m512i t89_hi = _mm512_unpackhi_epi32(row08, row09);
    __m512i tab_lo = _mm512_unpacklo_epi32(row10, row11);
    __m512i tab_hi = _mm512_unpackhi_epi32(row10, row11);
    __m512i tcd_lo = _mm512_unpacklo_epi32(row12, row13);
    __m512i tcd_hi = _mm512_unpackhi_epi32(row12, row13);
    __m512i tef_lo = _mm512_unpacklo_epi32(row14, row15);
    __m512i tef_hi = _mm512_unpackhi_epi32(row14, row15);

    // Stage 2: Unpack at 64-bit granularity
    __m512i u0123_ll = _mm512_unpacklo_epi64(t01_lo, t23_lo);
    __m512i u0123_lh = _mm512_unpackhi_epi64(t01_lo, t23_lo);
    __m512i u0123_hl = _mm512_unpacklo_epi64(t01_hi, t23_hi);
    __m512i u0123_hh = _mm512_unpackhi_epi64(t01_hi, t23_hi);
    __m512i u4567_ll = _mm512_unpacklo_epi64(t45_lo, t67_lo);
    __m512i u4567_lh = _mm512_unpackhi_epi64(t45_lo, t67_lo);
    __m512i u4567_hl = _mm512_unpacklo_epi64(t45_hi, t67_hi);
    __m512i u4567_hh = _mm512_unpackhi_epi64(t45_hi, t67_hi);
    __m512i u89ab_ll = _mm512_unpacklo_epi64(t89_lo, tab_lo);
    __m512i u89ab_lh = _mm512_unpackhi_epi64(t89_lo, tab_lo);
    __m512i u89ab_hl = _mm512_unpacklo_epi64(t89_hi, tab_hi);
    __m512i u89ab_hh = _mm512_unpackhi_epi64(t89_hi, tab_hi);
    __m512i ucdef_ll = _mm512_unpacklo_epi64(tcd_lo, tef_lo);
    __m512i ucdef_lh = _mm512_unpackhi_epi64(tcd_lo, tef_lo);
    __m512i ucdef_hl = _mm512_unpacklo_epi64(tcd_hi, tef_hi);
    __m512i ucdef_hh = _mm512_unpackhi_epi64(tcd_hi, tef_hi);

    // Stage 3: Shuffle 128-bit lanes
    __m512i v0_a = _mm512_shuffle_i32x4(u0123_ll, u4567_ll, 0x88);
    __m512i v0_b = _mm512_shuffle_i32x4(u0123_ll, u4567_ll, 0xDD);
    __m512i v1_a = _mm512_shuffle_i32x4(u0123_lh, u4567_lh, 0x88);
    __m512i v1_b = _mm512_shuffle_i32x4(u0123_lh, u4567_lh, 0xDD);
    __m512i v2_a = _mm512_shuffle_i32x4(u0123_hl, u4567_hl, 0x88);
    __m512i v2_b = _mm512_shuffle_i32x4(u0123_hl, u4567_hl, 0xDD);
    __m512i v3_a = _mm512_shuffle_i32x4(u0123_hh, u4567_hh, 0x88);
    __m512i v3_b = _mm512_shuffle_i32x4(u0123_hh, u4567_hh, 0xDD);
    __m512i v4_a = _mm512_shuffle_i32x4(u89ab_ll, ucdef_ll, 0x88);
    __m512i v4_b = _mm512_shuffle_i32x4(u89ab_ll, ucdef_ll, 0xDD);
    __m512i v5_a = _mm512_shuffle_i32x4(u89ab_lh, ucdef_lh, 0x88);
    __m512i v5_b = _mm512_shuffle_i32x4(u89ab_lh, ucdef_lh, 0xDD);
    __m512i v6_a = _mm512_shuffle_i32x4(u89ab_hl, ucdef_hl, 0x88);
    __m512i v6_b = _mm512_shuffle_i32x4(u89ab_hl, ucdef_hl, 0xDD);
    __m512i v7_a = _mm512_shuffle_i32x4(u89ab_hh, ucdef_hh, 0x88);
    __m512i v7_b = _mm512_shuffle_i32x4(u89ab_hh, ucdef_hh, 0xDD);

    // Stage 4: Final 256-bit shuffle to complete transpose
    __m512i out00 = _mm512_shuffle_i32x4(v0_a, v4_a, 0x88);
    __m512i out01 = _mm512_shuffle_i32x4(v1_a, v5_a, 0x88);
    __m512i out02 = _mm512_shuffle_i32x4(v2_a, v6_a, 0x88);
    __m512i out03 = _mm512_shuffle_i32x4(v3_a, v7_a, 0x88);
    __m512i out04 = _mm512_shuffle_i32x4(v0_a, v4_a, 0xDD);
    __m512i out05 = _mm512_shuffle_i32x4(v1_a, v5_a, 0xDD);
    __m512i out06 = _mm512_shuffle_i32x4(v2_a, v6_a, 0xDD);
    __m512i out07 = _mm512_shuffle_i32x4(v3_a, v7_a, 0xDD);
    __m512i out08 = _mm512_shuffle_i32x4(v0_b, v4_b, 0x88);
    __m512i out09 = _mm512_shuffle_i32x4(v1_b, v5_b, 0x88);
    __m512i out10 = _mm512_shuffle_i32x4(v2_b, v6_b, 0x88);
    __m512i out11 = _mm512_shuffle_i32x4(v3_b, v7_b, 0x88);
    __m512i out12 = _mm512_shuffle_i32x4(v0_b, v4_b, 0xDD);
    __m512i out13 = _mm512_shuffle_i32x4(v1_b, v5_b, 0xDD);
    __m512i out14 = _mm512_shuffle_i32x4(v2_b, v6_b, 0xDD);
    __m512i out15 = _mm512_shuffle_i32x4(v3_b, v7_b, 0xDD);

    // Store transposed results - each output row is one depth_group
    // Output layout: B.data[depth_group][col][quad] = 16 cols × 4 INT8 = 64 bytes
    _mm512_store_si512(&b_tile->data[0][0][0], out00);
    _mm512_store_si512(&b_tile->data[1][0][0], out01);
    _mm512_store_si512(&b_tile->data[2][0][0], out02);
    _mm512_store_si512(&b_tile->data[3][0][0], out03);
    _mm512_store_si512(&b_tile->data[4][0][0], out04);
    _mm512_store_si512(&b_tile->data[5][0][0], out05);
    _mm512_store_si512(&b_tile->data[6][0][0], out06);
    _mm512_store_si512(&b_tile->data[7][0][0], out07);
    _mm512_store_si512(&b_tile->data[8][0][0], out08);
    _mm512_store_si512(&b_tile->data[9][0][0], out09);
    _mm512_store_si512(&b_tile->data[10][0][0], out10);
    _mm512_store_si512(&b_tile->data[11][0][0], out11);
    _mm512_store_si512(&b_tile->data[12][0][0], out12);
    _mm512_store_si512(&b_tile->data[13][0][0], out13);
    _mm512_store_si512(&b_tile->data[14][0][0], out14);
    _mm512_store_si512(&b_tile->data[15][0][0], out15);

    nk_compiler_barrier_sapphire_amx_();
}

/*  Compute self cross-correlation for up to 16 BF16 vectors.
 *  Result[row_idx, column_idx] = dot(vectors[row_idx], vectors[column_idx])
 *
 *  @param vectors Row-major array of n_vectors, each of dimension depth
 *  @param n_vectors Number of vectors (1-16, padded internally if < 16)
 *  @param depth Vector dimension (padded to multiple of 96 internally)
 *  @param stride Byte stride between vectors
 *  @param result Output n_vectors × n_vectors matrix
 *  @param result_stride Byte stride between result rows
 */
NK_PUBLIC void nk_dots_symmetric_bf16_sapphire_amx(                  //
    nk_bf16_t const *vectors, nk_size_t n_vectors, nk_size_t depth, //
    nk_size_t stride, nk_f32_t *result, nk_size_t result_stride) {

    nk_size_t const stride_elements = stride / sizeof(nk_bf16_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);

    // Round depth up to multiple of 96 (3 tiles × 32 elements)
    nk_size_t const depth_tiles = (depth + 31) / 32;
    nk_size_t const depth_tile_groups = (depth_tiles + 2) / 3; // Groups of 3 tiles

    // Allocate tile buffers (3 A tiles + 3 B tiles per group)
    nk_dots_bf16_a16x32_sapphire_amx_t a_tiles[3];
    nk_dots_bf16_b32x16_sapphire_amx_t b_tiles[3];
    nk_dots_bf16_state_sapphire_amx_t state;

    // Configure AMX tiles
    nk_amx_tile_configure_sapphire_amx_();

    // Initialize output state
    nk_dots_bf16_init_sapphire_amx(&state);

    // Process depth dimension in groups of 96 (3 tiles)
    for (nk_size_t depth_group_idx = 0; depth_group_idx < depth_tile_groups; depth_group_idx++) {
        nk_size_t depth_base = depth_group_idx * 96;

        // Load 3 A tiles from vectors
        for (int tile_idx = 0; tile_idx < 3; tile_idx++) {
            nk_size_t depth_start = depth_base + tile_idx * 32;
            nk_size_t valid_cols = (depth_start + 32 <= depth) ? 32 : (depth > depth_start ? depth - depth_start : 0);
            nk_bf16_t const *src = vectors + depth_start;
            nk_dots_bf16_load_a_sapphire_amx(&a_tiles[tile_idx], src, stride_elements, n_vectors, valid_cols);
        }

        // Pack A transposed into B tiles
        for (int tile_idx = 0; tile_idx < 3; tile_idx++) {
            nk_dots_pack_bf16_transposed_sapphire_amx(&a_tiles[tile_idx], &b_tiles[tile_idx]);
        }

        // Accumulate: state += A × B
        nk_dots_bf16_update_sapphire_amx(&state, &a_tiles[0], &a_tiles[1], &a_tiles[2], &b_tiles[0], &b_tiles[1],
                                         &b_tiles[2]);
    }

    // Store result
    nk_dots_bf16_store_sapphire_amx(&state, result, result_stride_elements, n_vectors, n_vectors);
}

/*  Compute self cross-correlation for up to 16 INT8 vectors.
 *  Result[row_idx, column_idx] = dot(vectors[row_idx], vectors[column_idx])
 */
NK_PUBLIC void nk_dots_symmetric_i8_sapphire_amx(                  //
    nk_i8_t const *vectors, nk_size_t n_vectors, nk_size_t depth, //
    nk_size_t stride, nk_i32_t *result, nk_size_t result_stride) {

    nk_size_t const result_stride_elements = result_stride / sizeof(nk_i32_t);

    // Round depth up to multiple of 192 (3 tiles × 64 elements)
    nk_size_t const depth_tiles = (depth + 63) / 64;
    nk_size_t const depth_tile_groups = (depth_tiles + 2) / 3;

    // Allocate tile buffers
    nk_dots_i8_a16x64_sapphire_amx_t a_tiles[3];
    nk_dots_i8_b64x16_sapphire_amx_t b_tiles[3];
    nk_dots_i8_state_sapphire_amx_t state;

    // Configure AMX tiles
    nk_amx_tile_configure_sapphire_amx_();

    // Initialize output state
    nk_dots_i8_init_sapphire_amx(&state);

    // Process depth dimension in groups of 192 (3 tiles)
    for (nk_size_t depth_group_idx = 0; depth_group_idx < depth_tile_groups; depth_group_idx++) {
        nk_size_t depth_base = depth_group_idx * 192;

        // Load 3 A tiles from vectors
        for (int tile_idx = 0; tile_idx < 3; tile_idx++) {
            nk_size_t depth_start = depth_base + tile_idx * 64;
            nk_size_t valid_cols = (depth_start + 64 <= depth) ? 64 : (depth > depth_start ? depth - depth_start : 0);
            nk_i8_t const *src = vectors + depth_start;
            nk_dots_i8_load_a_sapphire_amx(&a_tiles[tile_idx], src, stride, n_vectors, valid_cols);
        }

        // Pack A transposed into B tiles
        for (int tile_idx = 0; tile_idx < 3; tile_idx++) {
            nk_dots_pack_i8_transposed_sapphire_amx(&a_tiles[tile_idx], &b_tiles[tile_idx]);
        }

        // Accumulate: state += A × B
        nk_dots_i8_update_sapphire_amx(&state, &a_tiles[0], &a_tiles[1], &a_tiles[2], &b_tiles[0], &b_tiles[1],
                                       &b_tiles[2]);
    }

    // Store result
    nk_dots_i8_store_sapphire_amx(&state, result, result_stride_elements, n_vectors, n_vectors);
}

/*  BF16 packed buffer size: header + all tiles for full column rows + column edge.
 *  Hybrid layout:
 *
 *  - Tiles include depth remainder (zero-padded) for AMX to handle full dot products
 *  - Column edge (remaining rows) stored row-major for simple AVX-512 edge kernel
 */
NK_PUBLIC nk_size_t nk_dots_packed_size_bf16_sapphire_amx(nk_size_t column_count, nk_size_t depth) {
    nk_size_t const tmm_rows = 16;
    nk_size_t const tmm_cols = 32;
    nk_size_t const tile_bytes = 512 * sizeof(nk_bf16_t); // 16 × 32 × 2 = 1KB

    nk_size_t const full_column_tiles = column_count / tmm_rows;
    nk_size_t const tiles_along_depth = (depth + tmm_cols - 1) / tmm_cols; // Ceiling division
    nk_size_t const column_edge_rows = column_count - full_column_tiles * tmm_rows;

    // Header (64 bytes aligned)
    nk_size_t size = sizeof(nk_dots_amx_packed_header_t);

    // All tiles for full column rows (Morton-ordered, pair-interleaved, depth remainder zero-padded)
    size += full_column_tiles * tiles_along_depth * tile_bytes;

    // Column edge: remaining rows for ALL depth columns, stored row-major
    if (column_edge_rows > 0) size += column_edge_rows * depth * sizeof(nk_bf16_t);

    return size;
}

/*  I8 packed buffer size: header + all tiles for full column rows + column edge.
 */
NK_PUBLIC nk_size_t nk_dots_packed_size_i8_sapphire_amx(nk_size_t column_count, nk_size_t depth) {
    nk_size_t const tmm_rows = 16;
    nk_size_t const tmm_cols = 64;
    nk_size_t const tile_bytes = 1024 * sizeof(nk_i8_t); // 16 × 64×1 = 1KB

    nk_size_t const full_column_tiles = column_count / tmm_rows;
    nk_size_t const tiles_along_depth = (depth + tmm_cols - 1) / tmm_cols; // Ceiling division
    nk_size_t const column_edge_rows = column_count - full_column_tiles * tmm_rows;

    // Header (64 bytes aligned)
    nk_size_t size = sizeof(nk_dots_amx_packed_header_t);

    // All tiles for full column rows (Morton-ordered, quad-interleaved, depth remainder zero-padded)
    size += full_column_tiles * tiles_along_depth * tile_bytes;

    // Column edge: remaining rows for ALL depth columns, stored row-major
    if (column_edge_rows > 0) size += column_edge_rows * depth * sizeof(nk_i8_t);

    return size;
}

NK_PUBLIC nk_size_t nk_dots_packed_size_u8_sapphire_amx(nk_size_t column_count, nk_size_t depth) {
    // Same layout as I8 - just different type interpretation
    return nk_dots_packed_size_i8_sapphire_amx(column_count, depth);
}

NK_PUBLIC nk_size_t nk_dots_packed_size_e4m3_sapphire_amx(nk_size_t column_count, nk_size_t depth) {
    // FP8 uses BF16 tile layout after conversion (same element count: 32 per row)
    return nk_dots_packed_size_bf16_sapphire_amx(column_count, depth);
}

NK_PUBLIC nk_size_t nk_dots_packed_size_e5m2_sapphire_amx(nk_size_t column_count, nk_size_t depth) {
    return nk_dots_packed_size_bf16_sapphire_amx(column_count, depth);
}

/*  Pack BF16 B matrix with hybrid layout:
 *
 *  - Header with layout metadata
 *  - All tiles for full column rows: Morton Z-curve ordered, pair-interleaved (for AMX)
 *    Including depth remainder tiles (zero-padded) so AMX can compute full dot products
 *  - Column edge rows: row-major (for AVX-512 edge kernel)
 *
 *  AMX BF16 tile format: for TDPBF16PS, B tile should have elements arranged so that
 *  consecutive pairs of columns are interleaved by rows:
 *    [col0_row0, col1_row0, col0_row1, col1_row1, ..., col0_row15, col1_row15,
 *     col2_row0, col3_row0, col2_row1, col3_row1, ...]
 *
 *  Interleaving formula: packed_idx = (col / 2) × 32 + row × 2 + (col % 2)
 */
NK_PUBLIC void nk_dots_pack_bf16_sapphire_amx(                   //
    nk_bf16_t const *b, nk_size_t column_count, nk_size_t depth, //
    nk_size_t b_stride, void *b_packed) {

    // AMX BF16 tile dimensions: 16 rows × 32 columns (512 BF16 elements = 1KB)
    nk_size_t const tmm_rows = 16;
    nk_size_t const tmm_cols = 32;
    nk_size_t const tile_elements = 512;
    nk_size_t const tile_bytes = tile_elements * sizeof(nk_bf16_t);
    nk_size_t const b_stride_elements = b_stride / sizeof(nk_bf16_t);

    // Compute layout dimensions
    nk_size_t const num_column_tiles = column_count / tmm_rows;
    nk_size_t const num_depth_tiles = (depth + tmm_cols - 1) / tmm_cols;
    nk_size_t const column_remainder_rows = column_count - num_column_tiles * tmm_rows;
    nk_size_t const total_tiles = num_column_tiles * num_depth_tiles;

    // Write header with layout metadata
    nk_dots_amx_packed_header_t *header = (nk_dots_amx_packed_header_t *)b_packed;
    header->full_column_tiles = (nk_u32_t)num_column_tiles;
    header->full_depth_tiles = (nk_u32_t)num_depth_tiles;
    header->column_edge_rows = (nk_u32_t)column_remainder_rows;

    // Compute memory region offsets
    nk_size_t const tiles_offset = sizeof(nk_dots_amx_packed_header_t);
    nk_size_t const column_edge_offset = tiles_offset + total_tiles * tile_bytes;
    header->column_edge_offset = (nk_u32_t)column_edge_offset;

    // Pointers to packed data regions
    nk_bf16_t *tiles_ptr = (nk_bf16_t *)((char *)b_packed + tiles_offset);
    nk_bf16_t *column_edge_ptr = (nk_bf16_t *)((char *)b_packed + column_edge_offset);

    // Zero-initialize all tiles (handles depth remainder padding)
    for (nk_size_t idx = 0; idx < total_tiles * tile_elements; idx++) tiles_ptr[idx] = 0;

    // Pack tiles using LINEAR ordering: tile_index = column_tile × num_depth_tiles + depth_tile
    // This provides sequential memory access when streaming along depth dimension,
    // which is critical for cache efficiency in the compute kernel.
    for (nk_size_t column_tile_idx = 0; column_tile_idx < num_column_tiles; column_tile_idx++) {
        for (nk_size_t depth_tile_idx = 0; depth_tile_idx < num_depth_tiles; depth_tile_idx++) {

            // Linear tile index: all depth-tiles for one column-tile are contiguous
            nk_size_t const tile_index = column_tile_idx * num_depth_tiles + depth_tile_idx;
            nk_bf16_t *tile_output = tiles_ptr + tile_index * tile_elements;

            // Source coordinates in original B matrix
            nk_size_t const src_row_start = column_tile_idx * tmm_rows;
            nk_size_t const src_col_start = depth_tile_idx * tmm_cols;
            nk_size_t const columns_to_pack = (src_col_start + tmm_cols <= depth) ? tmm_cols : (depth - src_col_start);

            // Pack with pair-interleaving as required by TDPBF16PS instruction.
            // AMX expects: [col0_row0, col1_row0, col0_row1, col1_row1, col2_row0, col3_row0, ...]
            // Formula: packed_idx = (col / 2) × 32 + row × 2 + (col % 2)
            for (nk_size_t row_idx = 0; row_idx < tmm_rows; row_idx++) {
                for (nk_size_t col_idx = 0; col_idx < columns_to_pack; col_idx++) {
                    nk_size_t const src_idx = (src_row_start + row_idx) * b_stride_elements + src_col_start + col_idx;
                    nk_size_t const dst_idx = (col_idx / 2) * 32 + row_idx * 2 + (col_idx % 2);
                    tile_output[dst_idx] = b[src_idx];
                }
            }
        }
    }

    // Pack column-remainder rows in simple row-major format (for AVX-512 fallback)
    if (column_remainder_rows > 0) {
        nk_size_t const remainder_start_row = num_column_tiles * tmm_rows;
        for (nk_size_t row_idx = 0; row_idx < column_remainder_rows; row_idx++) {
            for (nk_size_t col_idx = 0; col_idx < depth; col_idx++) {
                column_edge_ptr[row_idx * depth + col_idx] =
                    b[(remainder_start_row + row_idx) * b_stride_elements + col_idx];
            }
        }
    }
}

/*  Pack I8 B matrix with hybrid layout:
 *
 *  - Header with layout metadata
 *  - All tiles for full column rows: linearly ordered, quad-interleaved (for AMX)
 *    Including depth remainder tiles (zero-padded) so AMX can compute full dot products
 *  - Column edge rows: row-major (for AVX-512 edge kernel)
 *
 *  AMX INT8 tile format: for TDPBSSD, B tile should have 4 consecutive columns
 *  interleaved by rows:
 *    [col0_row0, col1_row0, col2_row0, col3_row0, col0_row1, col1_row1, ...]
 *
 *  Interleaving formula: packed_idx = (col / 4) × 64 + row × 4 + (col % 4)
 */
NK_PUBLIC void nk_dots_pack_i8_sapphire_amx(                   //
    nk_i8_t const *b, nk_size_t column_count, nk_size_t depth, //
    nk_size_t b_stride, void *b_packed) {

    // AMX I8 tile dimensions: 16 rows × 64 columns (1024 I8 elements = 1KB)
    nk_size_t const tmm_rows = 16;
    nk_size_t const tmm_cols = 64;
    nk_size_t const tile_elements = 1024;
    nk_size_t const tile_bytes = tile_elements * sizeof(nk_i8_t);

    // Compute layout dimensions
    nk_size_t const num_column_tiles = column_count / tmm_rows;
    nk_size_t const num_depth_tiles = (depth + tmm_cols - 1) / tmm_cols;
    nk_size_t const column_remainder_rows = column_count - num_column_tiles * tmm_rows;
    nk_size_t const total_tiles = num_column_tiles * num_depth_tiles;

    // Write header with layout metadata
    nk_dots_amx_packed_header_t *header = (nk_dots_amx_packed_header_t *)b_packed;
    header->full_column_tiles = (nk_u32_t)num_column_tiles;
    header->full_depth_tiles = (nk_u32_t)num_depth_tiles;
    header->column_edge_rows = (nk_u32_t)column_remainder_rows;

    // Compute memory region offsets
    nk_size_t const tiles_offset = sizeof(nk_dots_amx_packed_header_t);
    nk_size_t const column_edge_offset = tiles_offset + total_tiles * tile_bytes;
    header->column_edge_offset = (nk_u32_t)column_edge_offset;

    // Pointers to packed data regions
    nk_i8_t *tiles_ptr = (nk_i8_t *)((char *)b_packed + tiles_offset);
    nk_i8_t *column_edge_ptr = (nk_i8_t *)((char *)b_packed + column_edge_offset);

    // Zero-initialize all tiles (handles depth remainder padding)
    for (nk_size_t idx = 0; idx < total_tiles * tile_elements; idx++) tiles_ptr[idx] = 0;

    // Pack tiles using LINEAR ordering: tile_index = column_tile × num_depth_tiles + depth_tile
    // This provides sequential memory access when streaming along depth dimension.
    for (nk_size_t column_tile_idx = 0; column_tile_idx < num_column_tiles; column_tile_idx++) {
        for (nk_size_t depth_tile_idx = 0; depth_tile_idx < num_depth_tiles; depth_tile_idx++) {

            // Linear tile index: all depth-tiles for one column-tile are contiguous
            nk_size_t const tile_index = column_tile_idx * num_depth_tiles + depth_tile_idx;
            nk_i8_t *tile_output = tiles_ptr + tile_index * tile_elements;

            // Source coordinates in original B matrix
            nk_size_t const src_row_start = column_tile_idx * tmm_rows;
            nk_size_t const src_col_start = depth_tile_idx * tmm_cols;
            nk_size_t const columns_to_pack = (src_col_start + tmm_cols <= depth) ? tmm_cols : (depth - src_col_start);

            // Pack with quad-interleaving as required by TDPBSSD instruction.
            // AMX expects: [col0_row0, col1_row0, col2_row0, col3_row0, col0_row1, ...]
            // Formula: packed_idx = (col / 4) × 64 + row × 4 + (col % 4)
            for (nk_size_t row_idx = 0; row_idx < tmm_rows; row_idx++) {
                for (nk_size_t col_idx = 0; col_idx < columns_to_pack; col_idx++) {
                    nk_size_t const src_idx = (src_row_start + row_idx) * b_stride + src_col_start + col_idx;
                    nk_size_t const dst_idx = (col_idx / 4) * 64 + row_idx * 4 + (col_idx % 4);
                    tile_output[dst_idx] = b[src_idx];
                }
            }
        }
    }

    // Pack column-remainder rows in simple row-major format (for AVX-512 fallback)
    if (column_remainder_rows > 0) {
        nk_size_t const remainder_start_row = num_column_tiles * tmm_rows;
        for (nk_size_t row_idx = 0; row_idx < column_remainder_rows; row_idx++) {
            for (nk_size_t col_idx = 0; col_idx < depth; col_idx++) {
                column_edge_ptr[row_idx * depth + col_idx] = b[(remainder_start_row + row_idx) * b_stride + col_idx];
            }
        }
    }
}

/*  Pack U8 B matrix with hybrid layout.
 *  Same layout as I8, just different type interpretation.
 */
NK_PUBLIC void nk_dots_pack_u8_sapphire_amx(                   //
    nk_u8_t const *b, nk_size_t column_count, nk_size_t depth, //
    nk_size_t b_stride, void *b_packed) {

    nk_size_t const tmm_rows = 16;
    nk_size_t const tmm_cols = 64;
    nk_size_t const tile_elements = 1024;
    nk_size_t const tile_bytes = tile_elements * sizeof(nk_u8_t);

    nk_size_t const num_column_tiles = column_count / tmm_rows;
    nk_size_t const num_depth_tiles = (depth + tmm_cols - 1) / tmm_cols;
    nk_size_t const column_remainder_rows = column_count - num_column_tiles * tmm_rows;
    nk_size_t const total_tiles = num_column_tiles * num_depth_tiles;

    nk_dots_amx_packed_header_t *header = (nk_dots_amx_packed_header_t *)b_packed;
    header->full_column_tiles = (nk_u32_t)num_column_tiles;
    header->full_depth_tiles = (nk_u32_t)num_depth_tiles;
    header->column_edge_rows = (nk_u32_t)column_remainder_rows;

    nk_size_t const tiles_offset = sizeof(nk_dots_amx_packed_header_t);
    nk_size_t const column_edge_offset = tiles_offset + total_tiles * tile_bytes;
    header->column_edge_offset = (nk_u32_t)column_edge_offset;

    nk_u8_t *tiles_ptr = (nk_u8_t *)((char *)b_packed + tiles_offset);
    nk_u8_t *column_edge_ptr = (nk_u8_t *)((char *)b_packed + column_edge_offset);

    for (nk_size_t idx = 0; idx < total_tiles * tile_elements; idx++) tiles_ptr[idx] = 0;

    for (nk_size_t column_tile_idx = 0; column_tile_idx < num_column_tiles; column_tile_idx++) {
        for (nk_size_t depth_tile_idx = 0; depth_tile_idx < num_depth_tiles; depth_tile_idx++) {

            nk_size_t const tile_index = column_tile_idx * num_depth_tiles + depth_tile_idx;
            nk_u8_t *tile_output = tiles_ptr + tile_index * tile_elements;

            nk_size_t const src_row_start = column_tile_idx * tmm_rows;
            nk_size_t const src_col_start = depth_tile_idx * tmm_cols;
            nk_size_t const columns_to_pack = (src_col_start + tmm_cols <= depth) ? tmm_cols : (depth - src_col_start);

            // Pack with quad-interleaving as required by TDPBUUD instruction.
            for (nk_size_t row_idx = 0; row_idx < tmm_rows; row_idx++) {
                for (nk_size_t col_idx = 0; col_idx < columns_to_pack; col_idx++) {
                    nk_size_t const src_idx = (src_row_start + row_idx) * b_stride + src_col_start + col_idx;
                    nk_size_t const dst_idx = (col_idx / 4) * 64 + row_idx * 4 + (col_idx % 4);
                    tile_output[dst_idx] = b[src_idx];
                }
            }
        }
    }

    if (column_remainder_rows > 0) {
        nk_size_t const remainder_start_row = num_column_tiles * tmm_rows;
        for (nk_size_t row_idx = 0; row_idx < column_remainder_rows; row_idx++) {
            for (nk_size_t col_idx = 0; col_idx < depth; col_idx++) {
                column_edge_ptr[row_idx * depth + col_idx] = b[(remainder_start_row + row_idx) * b_stride + col_idx];
            }
        }
    }
}

/*  Pack E4M3 B matrix with FP8 → BF16 conversion.
 *  Converts E4M3 to BF16, then uses BF16 pair-interleaved layout.
 */
NK_PUBLIC void nk_dots_pack_e4m3_sapphire_amx(                   //
    nk_e4m3_t const *b, nk_size_t column_count, nk_size_t depth, //
    nk_size_t b_stride, void *b_packed) {

    nk_size_t const tmm_rows = 16;
    nk_size_t const tmm_cols = 32; // Same depth granularity as BF16
    nk_size_t const tile_elements = 512;
    nk_size_t const tile_bytes = tile_elements * sizeof(nk_bf16_t);

    nk_size_t const num_column_tiles = column_count / tmm_rows;
    nk_size_t const num_depth_tiles = (depth + tmm_cols - 1) / tmm_cols;
    nk_size_t const column_remainder_rows = column_count - num_column_tiles * tmm_rows;
    nk_size_t const total_tiles = num_column_tiles * num_depth_tiles;

    nk_dots_amx_packed_header_t *header = (nk_dots_amx_packed_header_t *)b_packed;
    header->full_column_tiles = (nk_u32_t)num_column_tiles;
    header->full_depth_tiles = (nk_u32_t)num_depth_tiles;
    header->column_edge_rows = (nk_u32_t)column_remainder_rows;

    nk_size_t const tiles_offset = sizeof(nk_dots_amx_packed_header_t);
    nk_size_t const column_edge_offset = tiles_offset + total_tiles * tile_bytes;
    header->column_edge_offset = (nk_u32_t)column_edge_offset;

    nk_bf16_t *tiles_ptr = (nk_bf16_t *)((char *)b_packed + tiles_offset);
    nk_bf16_t *column_edge_ptr = (nk_bf16_t *)((char *)b_packed + column_edge_offset);

    for (nk_size_t idx = 0; idx < total_tiles * tile_elements; idx++) tiles_ptr[idx] = 0;

    for (nk_size_t column_tile_idx = 0; column_tile_idx < num_column_tiles; column_tile_idx++) {
        for (nk_size_t depth_tile_idx = 0; depth_tile_idx < num_depth_tiles; depth_tile_idx++) {
            nk_size_t const tile_index = column_tile_idx * num_depth_tiles + depth_tile_idx;
            nk_bf16_t *tile_output = tiles_ptr + tile_index * tile_elements;

            nk_size_t const src_row_start = column_tile_idx * tmm_rows;
            nk_size_t const src_col_start = depth_tile_idx * tmm_cols;
            nk_size_t const columns_to_pack = (src_col_start + tmm_cols <= depth) ? tmm_cols : (depth - src_col_start);

            // Convert E4M3 to BF16 and pack with pair-interleaving
            for (nk_size_t row_idx = 0; row_idx < tmm_rows; row_idx++) {
                nk_size_t src_row = src_row_start + row_idx;
                // Load 32 E4M3 bytes and convert to BF16
                __mmask32 col_mask = (columns_to_pack >= 32) ? 0xFFFFFFFF : ((__mmask32)1 << columns_to_pack) - 1;
                __m256i e4m3_row = _mm256_maskz_loadu_epi8(col_mask, b + src_row * b_stride + src_col_start);
                __m512i bf16_row = nk_e4m3x32_to_bf16x32_ice_(e4m3_row);
                // Store with pair-interleaving
                nk_bf16_t bf16_buf[32];
                _mm512_storeu_si512((__m512i *)bf16_buf, bf16_row);
                for (nk_size_t col_idx = 0; col_idx < columns_to_pack; col_idx++) {
                    nk_size_t const dst_idx = (col_idx / 2) * 32 + row_idx * 2 + (col_idx % 2);
                    tile_output[dst_idx] = bf16_buf[col_idx];
                }
            }
        }
    }

    // Pack column-remainder rows (convert E4M3 to BF16)
    if (column_remainder_rows > 0) {
        nk_size_t const remainder_start_row = num_column_tiles * tmm_rows;
        for (nk_size_t row_idx = 0; row_idx < column_remainder_rows; row_idx++) {
            for (nk_size_t col_idx = 0; col_idx < depth; col_idx += 32) {
                nk_size_t cols = (col_idx + 32 <= depth) ? 32 : (depth - col_idx);
                __mmask32 col_mask = (cols >= 32) ? 0xFFFFFFFF : ((__mmask32)1 << cols) - 1;
                __m256i e4m3_chunk = _mm256_maskz_loadu_epi8(col_mask,
                                                             b + (remainder_start_row + row_idx) * b_stride + col_idx);
                __m512i bf16_chunk = nk_e4m3x32_to_bf16x32_ice_(e4m3_chunk);
                _mm512_mask_storeu_epi16(column_edge_ptr + row_idx * depth + col_idx, col_mask, bf16_chunk);
            }
        }
    }
}

/*  Pack E5M2 B matrix with FP8 → BF16 conversion.
 */
NK_PUBLIC void nk_dots_pack_e5m2_sapphire_amx(                   //
    nk_e5m2_t const *b, nk_size_t column_count, nk_size_t depth, //
    nk_size_t b_stride, void *b_packed) {

    nk_size_t const tmm_rows = 16;
    nk_size_t const tmm_cols = 32;
    nk_size_t const tile_elements = 512;
    nk_size_t const tile_bytes = tile_elements * sizeof(nk_bf16_t);

    nk_size_t const num_column_tiles = column_count / tmm_rows;
    nk_size_t const num_depth_tiles = (depth + tmm_cols - 1) / tmm_cols;
    nk_size_t const column_remainder_rows = column_count - num_column_tiles * tmm_rows;
    nk_size_t const total_tiles = num_column_tiles * num_depth_tiles;

    nk_dots_amx_packed_header_t *header = (nk_dots_amx_packed_header_t *)b_packed;
    header->full_column_tiles = (nk_u32_t)num_column_tiles;
    header->full_depth_tiles = (nk_u32_t)num_depth_tiles;
    header->column_edge_rows = (nk_u32_t)column_remainder_rows;

    nk_size_t const tiles_offset = sizeof(nk_dots_amx_packed_header_t);
    nk_size_t const column_edge_offset = tiles_offset + total_tiles * tile_bytes;
    header->column_edge_offset = (nk_u32_t)column_edge_offset;

    nk_bf16_t *tiles_ptr = (nk_bf16_t *)((char *)b_packed + tiles_offset);
    nk_bf16_t *column_edge_ptr = (nk_bf16_t *)((char *)b_packed + column_edge_offset);

    for (nk_size_t idx = 0; idx < total_tiles * tile_elements; idx++) tiles_ptr[idx] = 0;

    for (nk_size_t column_tile_idx = 0; column_tile_idx < num_column_tiles; column_tile_idx++) {
        for (nk_size_t depth_tile_idx = 0; depth_tile_idx < num_depth_tiles; depth_tile_idx++) {
            nk_size_t const tile_index = column_tile_idx * num_depth_tiles + depth_tile_idx;
            nk_bf16_t *tile_output = tiles_ptr + tile_index * tile_elements;

            nk_size_t const src_row_start = column_tile_idx * tmm_rows;
            nk_size_t const src_col_start = depth_tile_idx * tmm_cols;
            nk_size_t const columns_to_pack = (src_col_start + tmm_cols <= depth) ? tmm_cols : (depth - src_col_start);

            for (nk_size_t row_idx = 0; row_idx < tmm_rows; row_idx++) {
                nk_size_t src_row = src_row_start + row_idx;
                __mmask32 col_mask = (columns_to_pack >= 32) ? 0xFFFFFFFF : ((__mmask32)1 << columns_to_pack) - 1;
                __m256i e5m2_row = _mm256_maskz_loadu_epi8(col_mask, b + src_row * b_stride + src_col_start);
                __m512i bf16_row = nk_e5m2x32_to_bf16x32_ice_(e5m2_row);
                nk_bf16_t bf16_buf[32];
                _mm512_storeu_si512((__m512i *)bf16_buf, bf16_row);
                for (nk_size_t col_idx = 0; col_idx < columns_to_pack; col_idx++) {
                    nk_size_t const dst_idx = (col_idx / 2) * 32 + row_idx * 2 + (col_idx % 2);
                    tile_output[dst_idx] = bf16_buf[col_idx];
                }
            }
        }
    }

    if (column_remainder_rows > 0) {
        nk_size_t const remainder_start_row = num_column_tiles * tmm_rows;
        for (nk_size_t row_idx = 0; row_idx < column_remainder_rows; row_idx++) {
            for (nk_size_t col_idx = 0; col_idx < depth; col_idx += 32) {
                nk_size_t cols = (col_idx + 32 <= depth) ? 32 : (depth - col_idx);
                __mmask32 col_mask = (cols >= 32) ? 0xFFFFFFFF : ((__mmask32)1 << cols) - 1;
                __m256i e5m2_chunk = _mm256_maskz_loadu_epi8(col_mask,
                                                             b + (remainder_start_row + row_idx) * b_stride + col_idx);
                __m512i bf16_chunk = nk_e5m2x32_to_bf16x32_ice_(e5m2_chunk);
                _mm512_mask_storeu_epi16(column_edge_ptr + row_idx * depth + col_idx, col_mask, bf16_chunk);
            }
        }
    }
}

/*  BF16 → F32 matmul: C[row_count × column_count] = A[row_count × depth] × B[column_count × depth]ᵀ
 *
 *  Unified implementation using composable tile primitives.
 *  All I/O goes through aligned tile buffers for consistent behavior.
 *
 *  Uses register-resident 2 × 2 update pattern:
 *
 *  - TMM4-7 hold C accumulators across entire depth-loop (no redundant load/store)
 *  - 32 × 32 output blocks processed per row × column iteration
 *
 *  Single-threaded. For parallel execution, partition A rows across threads.
 */
NK_PUBLIC void nk_dots_packed_bf16_sapphire_amx(           //
    nk_bf16_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows_count, nk_size_t cols_count, nk_size_t depth, nk_size_t a_stride_bytes, nk_size_t c_stride_bytes) {

    // Parse packed B header
    nk_dots_amx_packed_header_t const *header = (nk_dots_amx_packed_header_t const *)b_packed;
    nk_size_t const col_tiles_count = header->full_column_tiles;
    nk_size_t const depth_tiles_count = header->full_depth_tiles;
    nk_size_t const col_edge_count = header->column_edge_rows;

    // Packed B data regions
    nk_bf16_t const *b_tiles_base = (nk_bf16_t const *)((char const *)b_packed + sizeof(nk_dots_amx_packed_header_t));
    nk_bf16_t const *col_edge_ptr = (nk_bf16_t const *)((char const *)b_packed + header->column_edge_offset);

    // Stride conversions
    nk_size_t const a_stride_elements = a_stride_bytes / sizeof(nk_bf16_t);
    nk_size_t const c_stride_elements = c_stride_bytes / sizeof(nk_f32_t);

    // Tile dimensions
    nk_size_t const tile_depth = 32; // depth elements per BF16 tile
    nk_size_t const tile_size = 512; // elements per packed tile
    nk_size_t const full_cols = col_tiles_count * 16;

    // Block counts (32 × 32 output blocks = 2 × 2 tiles)
    nk_size_t const row_blocks_count = (rows_count + 31) / 32;
    nk_size_t const col_blocks_count = col_tiles_count / 2;

    if (depth_tiles_count == 0) return;

    // Tile buffers for A (only used for edge tiles)
    nk_dots_bf16_a16x32_sapphire_amx_t a_tile_upper, a_tile_lower;
    nk_dots_bf16_state2x2_sapphire_amx_t c_accum_buffer;

    // Precompute: number of full depth-tiles (no masking needed)
    nk_size_t const full_depth_tiles_count = depth / tile_depth;
    nk_size_t const depth_remainder = depth % tile_depth;

    nk_amx_tile_configure_sapphire_amx_();

    // Loop order: row_blocks outer, col_blocks inner - maximizes A tile L2 cache reuse
    // A tiles stay in L2 while we sweep through all col_blocks for a given row_block
    for (nk_size_t row_block_idx = 0; row_block_idx < row_blocks_count; row_block_idx++) {
        nk_size_t const row_block_start = row_block_idx * 32;
        nk_size_t const valid_rows_count = (row_block_start + 32 <= rows_count) ? 32 : (rows_count - row_block_start);
        nk_size_t const is_full_row_block = (valid_rows_count == 32);

        for (nk_size_t col_block_idx = 0; col_block_idx < col_blocks_count; col_block_idx++) {
            nk_size_t const col_block_start = col_block_idx * 32;
            nk_size_t const b_col_left_base = (col_block_idx * 2) * depth_tiles_count;
            nk_size_t const b_col_right_base = (col_block_idx * 2 + 1) * depth_tiles_count;

            // Zero accumulators (TMM4-7 stay resident across entire depth loop)
            _tile_zero(4);
            _tile_zero(5);
            _tile_zero(6);
            _tile_zero(7);

            // Fast path: full row-block with full depth-tiles → direct A load with 2-deep pipelining
            if (is_full_row_block && full_depth_tiles_count > 0) {
                nk_bf16_t const *a_upper_base = a + row_block_start * a_stride_elements;
                nk_bf16_t const *a_lower_base = a + (row_block_start + 16) * a_stride_elements;

                nk_dots_bf16_b32x16_sapphire_amx_t const *b_tile_left =
                    (nk_dots_bf16_b32x16_sapphire_amx_t const *)(b_tiles_base + b_col_left_base * tile_size);
                nk_dots_bf16_b32x16_sapphire_amx_t const *b_tile_right =
                    (nk_dots_bf16_b32x16_sapphire_amx_t const *)(b_tiles_base + b_col_right_base * tile_size);

                // Prologue: load first depth tile
                _tile_loadd(0, a_upper_base, a_stride_bytes);
                _tile_loadd(1, a_lower_base, a_stride_bytes);
                _tile_loadd(2, b_tile_left->data, 64);
                _tile_loadd(3, b_tile_right->data, 64);

                // Main loop: 2-deep software pipelining
                for (nk_size_t depth_tile_idx = 0; depth_tile_idx < full_depth_tiles_count - 1; depth_tile_idx++) {
                    nk_size_t const next_depth_offset = (depth_tile_idx + 1) * tile_depth;

                    _tile_dpbf16ps(4, 0, 2);
                    _tile_dpbf16ps(5, 0, 3);
                    _tile_loadd(0, a_upper_base + next_depth_offset, a_stride_bytes);
                    _tile_loadd(1, a_lower_base + next_depth_offset, a_stride_bytes);

                    _tile_dpbf16ps(6, 1, 2);
                    _tile_dpbf16ps(7, 1, 3);
                    b_tile_left = (nk_dots_bf16_b32x16_sapphire_amx_t const *)(b_tiles_base +
                                                                               (b_col_left_base + depth_tile_idx + 1) *
                                                                                   tile_size);
                    b_tile_right = (nk_dots_bf16_b32x16_sapphire_amx_t const *)(b_tiles_base + (b_col_right_base +
                                                                                                depth_tile_idx + 1) *
                                                                                                   tile_size);
                    _tile_loadd(2, b_tile_left->data, 64);
                    _tile_loadd(3, b_tile_right->data, 64);
                }

                // Epilogue: final depth tile
                _tile_dpbf16ps(4, 0, 2);
                _tile_dpbf16ps(5, 0, 3);
                _tile_dpbf16ps(6, 1, 2);
                _tile_dpbf16ps(7, 1, 3);

                // Handle partial depth-tile (if any)
                if (depth_remainder > 0) {
                    nk_size_t const depth_offset = full_depth_tiles_count * tile_depth;

                    nk_dots_bf16_load_a_sapphire_amx(&a_tile_upper, a_upper_base + depth_offset, a_stride_elements, 16,
                                                     depth_remainder);
                    nk_dots_bf16_load_a_sapphire_amx(&a_tile_lower, a_lower_base + depth_offset, a_stride_elements, 16,
                                                     depth_remainder);

                    b_tile_left = (nk_dots_bf16_b32x16_sapphire_amx_t const *)(b_tiles_base + (b_col_left_base +
                                                                                               full_depth_tiles_count) *
                                                                                                  tile_size);
                    b_tile_right =
                        (nk_dots_bf16_b32x16_sapphire_amx_t const *)(b_tiles_base +
                                                                     (b_col_right_base + full_depth_tiles_count) *
                                                                         tile_size);

                    _tile_loadd(0, a_tile_upper.data, 64);
                    _tile_loadd(1, a_tile_lower.data, 64);
                    _tile_loadd(2, b_tile_left->data, 64);
                    _tile_loadd(3, b_tile_right->data, 64);

                    _tile_dpbf16ps(4, 0, 2);
                    _tile_dpbf16ps(5, 0, 3);
                    _tile_dpbf16ps(6, 1, 2);
                    _tile_dpbf16ps(7, 1, 3);
                }
            }
            // Full row-block but only partial depth tile (depth < tile_depth)
            else if (is_full_row_block) {
                nk_bf16_t const *a_upper_base = a + row_block_start * a_stride_elements;
                nk_bf16_t const *a_lower_base = a + (row_block_start + 16) * a_stride_elements;

                nk_dots_bf16_load_a_sapphire_amx(&a_tile_upper, a_upper_base, a_stride_elements, 16, depth_remainder);
                nk_dots_bf16_load_a_sapphire_amx(&a_tile_lower, a_lower_base, a_stride_elements, 16, depth_remainder);

                nk_dots_bf16_b32x16_sapphire_amx_t const *b_tile_left =
                    (nk_dots_bf16_b32x16_sapphire_amx_t const *)(b_tiles_base + b_col_left_base * tile_size);
                nk_dots_bf16_b32x16_sapphire_amx_t const *b_tile_right =
                    (nk_dots_bf16_b32x16_sapphire_amx_t const *)(b_tiles_base + b_col_right_base * tile_size);

                _tile_loadd(0, a_tile_upper.data, 64);
                _tile_loadd(1, a_tile_lower.data, 64);
                _tile_loadd(2, b_tile_left->data, 64);
                _tile_loadd(3, b_tile_right->data, 64);

                _tile_dpbf16ps(4, 0, 2);
                _tile_dpbf16ps(5, 0, 3);
                _tile_dpbf16ps(6, 1, 2);
                _tile_dpbf16ps(7, 1, 3);
            }
            // Slow path: edge row-block → buffered load with masking
            else {
                nk_size_t const rows_in_upper_tile = (valid_rows_count > 16) ? 16 : valid_rows_count;
                nk_size_t const rows_in_lower_tile = (valid_rows_count > 16) ? valid_rows_count - 16 : 0;

                for (nk_size_t depth_tile_idx = 0; depth_tile_idx < depth_tiles_count; depth_tile_idx++) {
                    nk_size_t const depth_offset = depth_tile_idx * tile_depth;
                    nk_size_t const valid_depth = (depth_tile_idx < full_depth_tiles_count) ? tile_depth
                                                                                            : depth_remainder;

                    nk_dots_bf16_load_a_sapphire_amx(&a_tile_upper,
                                                     a + row_block_start * a_stride_elements + depth_offset,
                                                     a_stride_elements, rows_in_upper_tile, valid_depth);
                    if (rows_in_lower_tile > 0) {
                        nk_dots_bf16_load_a_sapphire_amx(&a_tile_lower,
                                                         a + (row_block_start + 16) * a_stride_elements + depth_offset,
                                                         a_stride_elements, rows_in_lower_tile, valid_depth);
                    }

                    nk_dots_bf16_b32x16_sapphire_amx_t const *b_tile_left =
                        (nk_dots_bf16_b32x16_sapphire_amx_t const *)(b_tiles_base +
                                                                     (b_col_left_base + depth_tile_idx) * tile_size);
                    nk_dots_bf16_b32x16_sapphire_amx_t const *b_tile_right =
                        (nk_dots_bf16_b32x16_sapphire_amx_t const *)(b_tiles_base +
                                                                     (b_col_right_base + depth_tile_idx) * tile_size);

                    _tile_loadd(0, a_tile_upper.data, 64);
                    _tile_loadd(1, a_tile_lower.data, 64);
                    _tile_loadd(2, b_tile_left->data, 64);
                    _tile_loadd(3, b_tile_right->data, 64);

                    _tile_dpbf16ps(4, 0, 2);
                    _tile_dpbf16ps(5, 0, 3);
                    _tile_dpbf16ps(6, 1, 2);
                    _tile_dpbf16ps(7, 1, 3);
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
                nk_dots_bf16_output2x2_sapphire_amx(&c_accum_buffer,
                                                    c + row_block_start * c_stride_elements + col_block_start,
                                                    c_stride_elements, valid_rows_count, 32);
            }
        }
    }

    // Handle odd column-tile (single 16-column tile if col_tiles_count is odd)
    if (col_tiles_count % 2 == 1) {
        nk_size_t const col_tile_idx = col_tiles_count - 1;
        nk_size_t const col_start = col_tile_idx * 16;
        nk_size_t const b_col_base = col_tile_idx * depth_tiles_count;

        for (nk_size_t row_block_idx = 0; row_block_idx < row_blocks_count; row_block_idx++) {
            nk_size_t const row_block_start = row_block_idx * 32;
            nk_size_t const valid_rows_count = (row_block_start + 32 <= rows_count) ? 32
                                                                                    : (rows_count - row_block_start);
            nk_size_t const rows_in_upper_tile = (valid_rows_count > 16) ? 16 : valid_rows_count;
            nk_size_t const rows_in_lower_tile = (valid_rows_count > 16) ? valid_rows_count - 16 : 0;

            nk_dots_bf16_state_sapphire_amx_t c_upper_state, c_lower_state;

            _tile_zero(4);
            _tile_zero(6);

            for (nk_size_t depth_tile_idx = 0; depth_tile_idx < depth_tiles_count; depth_tile_idx++) {
                nk_size_t const depth_offset = depth_tile_idx * tile_depth;
                nk_size_t const valid_depth = (depth_tile_idx < full_depth_tiles_count) ? tile_depth : depth_remainder;

                nk_dots_bf16_load_a_sapphire_amx(&a_tile_upper, a + row_block_start * a_stride_elements + depth_offset,
                                                 a_stride_elements, rows_in_upper_tile, valid_depth);
                if (rows_in_lower_tile > 0) {
                    nk_dots_bf16_load_a_sapphire_amx(&a_tile_lower,
                                                     a + (row_block_start + 16) * a_stride_elements + depth_offset,
                                                     a_stride_elements, rows_in_lower_tile, valid_depth);
                }

                nk_dots_bf16_b32x16_sapphire_amx_t const *b_tile =
                    (nk_dots_bf16_b32x16_sapphire_amx_t const *)(b_tiles_base +
                                                                 (b_col_base + depth_tile_idx) * tile_size);

                _tile_loadd(0, a_tile_upper.data, 64);
                _tile_loadd(1, a_tile_lower.data, 64);
                _tile_loadd(2, b_tile->data, 64);

                _tile_dpbf16ps(4, 0, 2);
                _tile_dpbf16ps(6, 1, 2);
            }

            _tile_stored(4, c_upper_state.data, 64);
            _tile_stored(6, c_lower_state.data, 64);

            nk_dots_bf16_store_sapphire_amx(&c_upper_state, c + row_block_start * c_stride_elements + col_start,
                                            c_stride_elements, rows_in_upper_tile, 16);
            if (rows_in_lower_tile > 0) {
                nk_dots_bf16_store_sapphire_amx(&c_lower_state,
                                                c + (row_block_start + 16) * c_stride_elements + col_start,
                                                c_stride_elements, rows_in_lower_tile, 16);
            }
        }
    }

    // Handle column-edge (remaining columns < 16) using AMX with partial tiles
    if (col_edge_count > 0) {
        for (nk_size_t row_block_idx = 0; row_block_idx < row_blocks_count; row_block_idx++) {
            nk_size_t const row_block_start = row_block_idx * 32;
            nk_size_t const valid_rows_count = (row_block_start + 32 <= rows_count) ? 32
                                                                                    : (rows_count - row_block_start);
            nk_size_t const rows_in_upper_tile = (valid_rows_count > 16) ? 16 : valid_rows_count;
            nk_size_t const rows_in_lower_tile = (valid_rows_count > 16) ? valid_rows_count - 16 : 0;

            nk_dots_bf16_state_sapphire_amx_t c_upper_state, c_lower_state;
            nk_dots_bf16_a16x32_sapphire_amx_t b_as_a;
            nk_dots_bf16_b32x16_sapphire_amx_t b_tile;

            _tile_zero(4);
            _tile_zero(6);

            for (nk_size_t depth_tile_idx = 0; depth_tile_idx < depth_tiles_count; depth_tile_idx++) {
                nk_size_t const depth_offset = depth_tile_idx * tile_depth;
                nk_size_t const valid_depth = (depth_tile_idx < full_depth_tiles_count) ? tile_depth : depth_remainder;

                nk_dots_bf16_load_a_sapphire_amx(&a_tile_upper, a + row_block_start * a_stride_elements + depth_offset,
                                                 a_stride_elements, rows_in_upper_tile, valid_depth);
                if (rows_in_lower_tile > 0) {
                    nk_dots_bf16_load_a_sapphire_amx(&a_tile_lower,
                                                     a + (row_block_start + 16) * a_stride_elements + depth_offset,
                                                     a_stride_elements, rows_in_lower_tile, valid_depth);
                }

                nk_dots_bf16_load_a_sapphire_amx(&b_as_a, col_edge_ptr + depth_offset, depth, col_edge_count,
                                                 valid_depth);
                nk_dots_pack_bf16_transposed_sapphire_amx(&b_as_a, &b_tile);

                _tile_loadd(0, a_tile_upper.data, 64);
                _tile_loadd(1, a_tile_lower.data, 64);
                _tile_loadd(2, b_tile.data, 64);

                _tile_dpbf16ps(4, 0, 2);
                _tile_dpbf16ps(6, 1, 2);
            }

            _tile_stored(4, c_upper_state.data, 64);
            _tile_stored(6, c_lower_state.data, 64);

            nk_dots_bf16_store_sapphire_amx(&c_upper_state, c + row_block_start * c_stride_elements + full_cols,
                                            c_stride_elements, rows_in_upper_tile, col_edge_count);
            if (rows_in_lower_tile > 0) {
                nk_dots_bf16_store_sapphire_amx(&c_lower_state,
                                                c + (row_block_start + 16) * c_stride_elements + full_cols,
                                                c_stride_elements, rows_in_lower_tile, col_edge_count);
            }
        }
    }

    _tile_release();
}

/*  BF16 compact: truncate F32 → BF16 in-place using AVX512.
 *  Reads F32 matrix, writes BF16 to same buffer (safe since F32 is larger).
 *  Uses masked loads/stores to handle all sizes without scalar fallback.
 *  Output is tightly packed with stride = column_count × sizeof(bf16).
 */
NK_PUBLIC void nk_dots_compact_bf16_sapphire_amx( //
    void *c, nk_size_t row_count, nk_size_t column_count, nk_size_t c_stride) {

    nk_size_t const c_stride_f32 = c_stride / sizeof(nk_f32_t);
    nk_f32_t const *c_f32 = (nk_f32_t const *)c;
    nk_bf16_t *c_bf16 = (nk_bf16_t *)c;

    for (nk_size_t row_idx = 0; row_idx < row_count; row_idx++) {
        nk_f32_t const *src_row = c_f32 + row_idx * c_stride_f32;
        nk_bf16_t *dst_row = c_bf16 + row_idx * column_count;
        nk_size_t column_idx = 0;

        // Process 16 floats at a time using AVX512-BF16
        for (; column_idx + 16 <= column_count; column_idx += 16) {
            __m512 f32_vec = _mm512_loadu_ps(src_row + column_idx);
            __m256bh bf16_vec = _mm512_cvtneps_pbh(f32_vec);
            _mm256_storeu_si256((__m256i *)(dst_row + column_idx), (__m256i)bf16_vec);
        }

        // Handle remaining elements with masked operations
        if (column_idx < column_count) {
            __mmask16 tail_mask = (__mmask16)((1u << (column_count - column_idx)) - 1);
            __m512 f32_vec = _mm512_maskz_loadu_ps(tail_mask, src_row + column_idx);
            __m256bh bf16_vec = _mm512_cvtneps_pbh(f32_vec);
            _mm256_mask_storeu_epi16(dst_row + column_idx, tail_mask, (__m256i)bf16_vec);
        }
    }
}

/*  I8 → I32 matmul: C[row_count × column_count] = A[row_count × depth] × B[column_count × depth]ᵀ
 *
 *  Unified implementation using composable tile primitives.
 *  All I/O goes through aligned tile buffers for consistent behavior.
 *
 *  Uses register-resident 2 × 2 update pattern:
 *
 *  - TMM4-7 hold C accumulators across entire depth-loop (no redundant load/store)
 *  - 32 × 32 output blocks processed per row × column iteration
 *
 *  Single-threaded. For parallel execution, partition A rows across threads.
 */
NK_PUBLIC void nk_dots_packed_i8_sapphire_amx(           //
    nk_i8_t const *a, void const *b_packed, nk_i32_t *c, //
    nk_size_t rows_count, nk_size_t cols_count, nk_size_t depth, nk_size_t a_stride_bytes, nk_size_t c_stride_bytes) {

    // Parse packed B header
    nk_dots_amx_packed_header_t const *header = (nk_dots_amx_packed_header_t const *)b_packed;
    nk_size_t const col_tiles_count = header->full_column_tiles;
    nk_size_t const depth_tiles_count = header->full_depth_tiles;
    nk_size_t const col_edge_count = header->column_edge_rows;

    // Packed B data regions
    nk_i8_t const *b_tiles_base = (nk_i8_t const *)((char const *)b_packed + sizeof(nk_dots_amx_packed_header_t));
    nk_i8_t const *col_edge_ptr = (nk_i8_t const *)((char const *)b_packed + header->column_edge_offset);

    // Stride conversions
    nk_size_t const c_stride_elements = c_stride_bytes / sizeof(nk_i32_t);

    // Tile dimensions
    nk_size_t const tile_depth = 64;  // depth elements per INT8 tile
    nk_size_t const tile_size = 1024; // bytes per packed tile
    nk_size_t const full_cols = col_tiles_count * 16;

    // Block counts (32 × 32 output blocks = 2 × 2 tiles)
    nk_size_t const row_blocks_count = (rows_count + 31) / 32;
    nk_size_t const col_blocks_count = col_tiles_count / 2;

    if (depth_tiles_count == 0) return;

    // Tile buffers for A (only used for edge tiles)
    nk_dots_i8_a16x64_sapphire_amx_t a_tile_upper, a_tile_lower;
    nk_dots_i8_state2x2_sapphire_amx_t c_accum_buffer;

    // Precompute: number of full depth-tiles (no masking needed)
    nk_size_t const full_depth_tiles_count = depth / tile_depth;
    nk_size_t const depth_remainder = depth % tile_depth;

    nk_amx_tile_configure_sapphire_amx_();

    // Process all 32 × 32 row × column blocks (including partial edge blocks)
    for (nk_size_t row_block_idx = 0; row_block_idx < row_blocks_count; row_block_idx++) {
        nk_size_t const row_block_start = row_block_idx * 32;
        nk_size_t const valid_rows_count = (row_block_start + 32 <= rows_count) ? 32 : (rows_count - row_block_start);
        nk_size_t const is_full_row_block = (valid_rows_count == 32);

        // Process full column-blocks (pairs of 16-column tiles = 32 columns)
        for (nk_size_t col_block_idx = 0; col_block_idx < col_blocks_count; col_block_idx++) {
            nk_size_t const col_block_start = col_block_idx * 32;

            // B tile base indices (linear layout: col_tile × depth_tiles_count + depth_tile)
            nk_size_t const b_col_left_base = (col_block_idx * 2) * depth_tiles_count;
            nk_size_t const b_col_right_base = (col_block_idx * 2 + 1) * depth_tiles_count;

            // Zero accumulators (TMM4-7 stay resident across entire depth loop)
            _tile_zero(4); // C[upper, left]
            _tile_zero(5); // C[upper, right]
            _tile_zero(6); // C[lower, left]
            _tile_zero(7); // C[lower, right]

            // Fast path: full row-block with full depth-tiles → direct A load with 2-deep pipelining
            if (is_full_row_block && full_depth_tiles_count > 0) {
                // A row pointers for direct load
                nk_i8_t const *a_upper_base = a + row_block_start * a_stride_bytes;
                nk_i8_t const *a_lower_base = a + (row_block_start + 16) * a_stride_bytes;

                // B tile pointers
                nk_dots_i8_b64x16_sapphire_amx_t const *b_tile_left =
                    (nk_dots_i8_b64x16_sapphire_amx_t const *)(b_tiles_base + b_col_left_base * tile_size);
                nk_dots_i8_b64x16_sapphire_amx_t const *b_tile_right =
                    (nk_dots_i8_b64x16_sapphire_amx_t const *)(b_tiles_base + b_col_right_base * tile_size);

                // Prologue: load first depth tile into TMM0-3
                _tile_loadd(0, a_upper_base, a_stride_bytes);
                _tile_loadd(1, a_lower_base, a_stride_bytes);
                _tile_loadd(2, b_tile_left->data, 64);
                _tile_loadd(3, b_tile_right->data, 64);

                // Main loop: 2-deep software pipelining (compute current while loading next)
                for (nk_size_t depth_tile_idx = 0; depth_tile_idx < full_depth_tiles_count - 1; depth_tile_idx++) {
                    nk_size_t const next_depth_offset = (depth_tile_idx + 1) * tile_depth;

                    // Compute upper row products while loading next A
                    _tile_dpbssd(4, 0, 2);
                    _tile_dpbssd(5, 0, 3);
                    _tile_loadd(0, a_upper_base + next_depth_offset, a_stride_bytes);
                    _tile_loadd(1, a_lower_base + next_depth_offset, a_stride_bytes);

                    // Compute lower row products while loading next B
                    _tile_dpbssd(6, 1, 2);
                    _tile_dpbssd(7, 1, 3);
                    b_tile_left = (nk_dots_i8_b64x16_sapphire_amx_t const *)(b_tiles_base +
                                                                             (b_col_left_base + depth_tile_idx + 1) *
                                                                                 tile_size);
                    b_tile_right = (nk_dots_i8_b64x16_sapphire_amx_t const *)(b_tiles_base +
                                                                              (b_col_right_base + depth_tile_idx + 1) *
                                                                                  tile_size);
                    _tile_loadd(2, b_tile_left->data, 64);
                    _tile_loadd(3, b_tile_right->data, 64);
                }

                // Epilogue: final depth tile (no next to load)
                _tile_dpbssd(4, 0, 2);
                _tile_dpbssd(5, 0, 3);
                _tile_dpbssd(6, 1, 2);
                _tile_dpbssd(7, 1, 3);

                // Handle partial depth-tile (if any) with buffered load
                if (depth_remainder > 0) {
                    nk_size_t const depth_offset = full_depth_tiles_count * tile_depth;

                    nk_dots_i8_load_a_sapphire_amx(&a_tile_upper, a_upper_base + depth_offset, a_stride_bytes, 16,
                                                   depth_remainder);
                    nk_dots_i8_load_a_sapphire_amx(&a_tile_lower, a_lower_base + depth_offset, a_stride_bytes, 16,
                                                   depth_remainder);

                    b_tile_left = (nk_dots_i8_b64x16_sapphire_amx_t const *)(b_tiles_base + (b_col_left_base +
                                                                                             full_depth_tiles_count) *
                                                                                                tile_size);
                    b_tile_right = (nk_dots_i8_b64x16_sapphire_amx_t const *)(b_tiles_base + (b_col_right_base +
                                                                                              full_depth_tiles_count) *
                                                                                                 tile_size);

                    _tile_loadd(0, a_tile_upper.data, 64);
                    _tile_loadd(1, a_tile_lower.data, 64);
                    _tile_loadd(2, b_tile_left->data, 64);
                    _tile_loadd(3, b_tile_right->data, 64);

                    _tile_dpbssd(4, 0, 2);
                    _tile_dpbssd(5, 0, 3);
                    _tile_dpbssd(6, 1, 2);
                    _tile_dpbssd(7, 1, 3);
                }
            }
            // Full row-block but only partial depth tile (depth < tile_depth)
            else if (is_full_row_block) {
                nk_i8_t const *a_upper_base = a + row_block_start * a_stride_bytes;
                nk_i8_t const *a_lower_base = a + (row_block_start + 16) * a_stride_bytes;

                nk_dots_i8_load_a_sapphire_amx(&a_tile_upper, a_upper_base, a_stride_bytes, 16, depth_remainder);
                nk_dots_i8_load_a_sapphire_amx(&a_tile_lower, a_lower_base, a_stride_bytes, 16, depth_remainder);

                nk_dots_i8_b64x16_sapphire_amx_t const *b_tile_left =
                    (nk_dots_i8_b64x16_sapphire_amx_t const *)(b_tiles_base + b_col_left_base * tile_size);
                nk_dots_i8_b64x16_sapphire_amx_t const *b_tile_right =
                    (nk_dots_i8_b64x16_sapphire_amx_t const *)(b_tiles_base + b_col_right_base * tile_size);

                _tile_loadd(0, a_tile_upper.data, 64);
                _tile_loadd(1, a_tile_lower.data, 64);
                _tile_loadd(2, b_tile_left->data, 64);
                _tile_loadd(3, b_tile_right->data, 64);

                _tile_dpbssd(4, 0, 2);
                _tile_dpbssd(5, 0, 3);
                _tile_dpbssd(6, 1, 2);
                _tile_dpbssd(7, 1, 3);
            }
            // Slow path: edge row-block → always use buffered load with masking
            else {
                nk_size_t const rows_in_upper_tile = (valid_rows_count > 16) ? 16 : valid_rows_count;
                nk_size_t const rows_in_lower_tile = (valid_rows_count > 16) ? valid_rows_count - 16 : 0;

                for (nk_size_t depth_tile_idx = 0; depth_tile_idx < depth_tiles_count; depth_tile_idx++) {
                    nk_size_t const depth_offset = depth_tile_idx * tile_depth;
                    nk_size_t const valid_depth = (depth_tile_idx < full_depth_tiles_count) ? tile_depth
                                                                                            : depth_remainder;

                    nk_dots_i8_load_a_sapphire_amx(&a_tile_upper, a + row_block_start * a_stride_bytes + depth_offset,
                                                   a_stride_bytes, rows_in_upper_tile, valid_depth);
                    if (rows_in_lower_tile > 0) {
                        nk_dots_i8_load_a_sapphire_amx(&a_tile_lower,
                                                       a + (row_block_start + 16) * a_stride_bytes + depth_offset,
                                                       a_stride_bytes, rows_in_lower_tile, valid_depth);
                    }

                    nk_dots_i8_b64x16_sapphire_amx_t const *b_tile_left =
                        (nk_dots_i8_b64x16_sapphire_amx_t const *)(b_tiles_base +
                                                                   (b_col_left_base + depth_tile_idx) * tile_size);
                    nk_dots_i8_b64x16_sapphire_amx_t const *b_tile_right =
                        (nk_dots_i8_b64x16_sapphire_amx_t const *)(b_tiles_base +
                                                                   (b_col_right_base + depth_tile_idx) * tile_size);

                    _tile_loadd(0, a_tile_upper.data, 64);
                    _tile_loadd(1, a_tile_lower.data, 64);
                    _tile_loadd(2, b_tile_left->data, 64);
                    _tile_loadd(3, b_tile_right->data, 64);

                    _tile_dpbssd(4, 0, 2);
                    _tile_dpbssd(5, 0, 3);
                    _tile_dpbssd(6, 1, 2);
                    _tile_dpbssd(7, 1, 3);
                }
            }

            // Store accumulators to output (once per output block, not per depth tile)
            if (is_full_row_block) {
                nk_i32_t *c_block = c + row_block_start * c_stride_elements + col_block_start;
                _tile_stored(4, c_block, c_stride_bytes);
                _tile_stored(5, c_block + 16, c_stride_bytes);
                _tile_stored(6, (nk_i32_t *)((char *)c_block + 16 * c_stride_bytes), c_stride_bytes);
                _tile_stored(7, (nk_i32_t *)((char *)c_block + 16 * c_stride_bytes) + 16, c_stride_bytes);
            }
            else {
                // Slow path: edge row-block needs masked output
                _tile_stored(4, c_accum_buffer.c[0][0].data, 64);
                _tile_stored(5, c_accum_buffer.c[0][1].data, 64);
                _tile_stored(6, c_accum_buffer.c[1][0].data, 64);
                _tile_stored(7, c_accum_buffer.c[1][1].data, 64);
                nk_dots_i8_output2x2_sapphire_amx(&c_accum_buffer,
                                                  c + row_block_start * c_stride_elements + col_block_start,
                                                  c_stride_elements, valid_rows_count, 32);
            }
        }

        // Handle odd column-tile (single 16-column tile if col_tiles_count is odd)
        if (col_tiles_count % 2 == 1) {
            nk_size_t const col_tile_idx = col_tiles_count - 1;
            nk_size_t const col_start = col_tile_idx * 16;
            nk_size_t const b_col_base = col_tile_idx * depth_tiles_count;
            nk_size_t const rows_in_upper_tile = (valid_rows_count > 16) ? 16 : valid_rows_count;
            nk_size_t const rows_in_lower_tile = (valid_rows_count > 16) ? valid_rows_count - 16 : 0;

            // Use 1 × 2 blocking for single column-tile (2 row-tiles × 1 col-tile)
            nk_dots_i8_state_sapphire_amx_t c_upper_state, c_lower_state;

            _tile_zero(4);
            _tile_zero(6);

            for (nk_size_t depth_tile_idx = 0; depth_tile_idx < depth_tiles_count; depth_tile_idx++) {
                nk_size_t const depth_offset = depth_tile_idx * tile_depth;
                nk_size_t const valid_depth = (depth_tile_idx < full_depth_tiles_count) ? tile_depth : depth_remainder;

                nk_dots_i8_load_a_sapphire_amx(&a_tile_upper, a + row_block_start * a_stride_bytes + depth_offset,
                                               a_stride_bytes, rows_in_upper_tile, valid_depth);
                if (rows_in_lower_tile > 0) {
                    nk_dots_i8_load_a_sapphire_amx(&a_tile_lower,
                                                   a + (row_block_start + 16) * a_stride_bytes + depth_offset,
                                                   a_stride_bytes, rows_in_lower_tile, valid_depth);
                }

                nk_dots_i8_b64x16_sapphire_amx_t const *b_tile =
                    (nk_dots_i8_b64x16_sapphire_amx_t const *)(b_tiles_base +
                                                               (b_col_base + depth_tile_idx) * tile_size);

                _tile_loadd(0, a_tile_upper.data, 64);
                _tile_loadd(1, a_tile_lower.data, 64);
                _tile_loadd(2, b_tile->data, 64);

                _tile_dpbssd(4, 0, 2);
                _tile_dpbssd(6, 1, 2);
            }

            _tile_stored(4, c_upper_state.data, 64);
            _tile_stored(6, c_lower_state.data, 64);

            nk_dots_i8_store_sapphire_amx(&c_upper_state, c + row_block_start * c_stride_elements + col_start,
                                          c_stride_elements, rows_in_upper_tile, 16);
            if (rows_in_lower_tile > 0) {
                nk_dots_i8_store_sapphire_amx(&c_lower_state,
                                              c + (row_block_start + 16) * c_stride_elements + col_start,
                                              c_stride_elements, rows_in_lower_tile, 16);
            }
        }

        // Handle column-edge (remaining columns < 16) using AMX with partial tiles
        if (col_edge_count > 0) {
            nk_size_t const rows_in_upper_tile = (valid_rows_count > 16) ? 16 : valid_rows_count;
            nk_size_t const rows_in_lower_tile = (valid_rows_count > 16) ? valid_rows_count - 16 : 0;

            nk_dots_i8_state_sapphire_amx_t c_upper_state, c_lower_state;
            nk_dots_i8_a16x64_sapphire_amx_t b_as_a;
            nk_dots_i8_b64x16_sapphire_amx_t b_tile;

            _tile_zero(4);
            _tile_zero(6);

            for (nk_size_t depth_tile_idx = 0; depth_tile_idx < depth_tiles_count; depth_tile_idx++) {
                nk_size_t const depth_offset = depth_tile_idx * tile_depth;
                nk_size_t const valid_depth = (depth_tile_idx < full_depth_tiles_count) ? tile_depth : depth_remainder;

                // Load A tiles
                nk_dots_i8_load_a_sapphire_amx(&a_tile_upper, a + row_block_start * a_stride_bytes + depth_offset,
                                               a_stride_bytes, rows_in_upper_tile, valid_depth);
                if (rows_in_lower_tile > 0) {
                    nk_dots_i8_load_a_sapphire_amx(&a_tile_lower,
                                                   a + (row_block_start + 16) * a_stride_bytes + depth_offset,
                                                   a_stride_bytes, rows_in_lower_tile, valid_depth);
                }

                // Load B edge data (row-major: b_edge[row × depth + col]) and pack into B tile
                // Each "row" in edge data corresponds to one output column
                nk_dots_i8_load_a_sapphire_amx(&b_as_a, col_edge_ptr + depth_offset, depth, col_edge_count,
                                               valid_depth);
                nk_dots_pack_i8_transposed_sapphire_amx(&b_as_a, &b_tile);

                _tile_loadd(0, a_tile_upper.data, 64);
                _tile_loadd(1, a_tile_lower.data, 64);
                _tile_loadd(2, b_tile.data, 64);

                _tile_dpbssd(4, 0, 2);
                _tile_dpbssd(6, 1, 2);
            }

            _tile_stored(4, c_upper_state.data, 64);
            _tile_stored(6, c_lower_state.data, 64);

            nk_dots_i8_store_sapphire_amx(&c_upper_state, c + row_block_start * c_stride_elements + full_cols,
                                          c_stride_elements, rows_in_upper_tile, col_edge_count);
            if (rows_in_lower_tile > 0) {
                nk_dots_i8_store_sapphire_amx(&c_lower_state,
                                              c + (row_block_start + 16) * c_stride_elements + full_cols,
                                              c_stride_elements, rows_in_lower_tile, col_edge_count);
            }
        }
    }

    _tile_release();
}

/*  U8 × U8 → U32 matmul: C[rows_count × cols_count] = A[rows_count × depth] × B[cols_count × depth]ᵀ
 *  Uses TDPBUUD instruction for unsigned×unsigned → unsigned accumulation.
 */
NK_PUBLIC void nk_dots_packed_u8_sapphire_amx(           //
    nk_u8_t const *a, void const *b_packed, nk_u32_t *c, //
    nk_size_t rows_count, nk_size_t cols_count, nk_size_t depth, nk_size_t a_stride_bytes, nk_size_t c_stride_bytes) {

    // Parse packed B header
    nk_dots_amx_packed_header_t const *header = (nk_dots_amx_packed_header_t const *)b_packed;
    nk_size_t const col_tiles_count = header->full_column_tiles;
    nk_size_t const depth_tiles_count = header->full_depth_tiles;
    nk_size_t const col_edge_count = header->column_edge_rows;

    // Packed B data regions
    nk_u8_t const *b_tiles_base = (nk_u8_t const *)((char const *)b_packed + sizeof(nk_dots_amx_packed_header_t));
    nk_u8_t const *col_edge_ptr = (nk_u8_t const *)((char const *)b_packed + header->column_edge_offset);

    // Stride conversions
    nk_size_t const c_stride_elements = c_stride_bytes / sizeof(nk_u32_t);

    // Tile dimensions
    nk_size_t const tile_depth = 64;  // depth elements per U8 tile
    nk_size_t const tile_size = 1024; // bytes per packed tile
    nk_size_t const full_cols = col_tiles_count * 16;

    // Block counts (32 × 32 output blocks = 2 × 2 tiles)
    nk_size_t const row_blocks_count = (rows_count + 31) / 32;
    nk_size_t const col_blocks_count = col_tiles_count / 2;

    if (depth_tiles_count == 0) return;

    // Tile buffers for A (only used for edge tiles)
    nk_dots_u8_a16x64_sapphire_amx_t a_tile_upper, a_tile_lower;
    nk_dots_u8_state2x2_sapphire_amx_t c_accum_buffer;

    // Precompute: number of full depth-tiles
    nk_size_t const full_depth_tiles_count = depth / tile_depth;
    nk_size_t const depth_remainder = depth % tile_depth;

    nk_amx_tile_configure_sapphire_amx_();

    // Process all 32 × 32 row × column blocks
    for (nk_size_t row_block_idx = 0; row_block_idx < row_blocks_count; row_block_idx++) {
        nk_size_t const row_block_start = row_block_idx * 32;
        nk_size_t const valid_rows_count = (row_block_start + 32 <= rows_count) ? 32 : (rows_count - row_block_start);
        nk_size_t const is_full_row_block = (valid_rows_count == 32);

        // Process full column-blocks (pairs of 16-column tiles = 32 columns)
        for (nk_size_t col_block_idx = 0; col_block_idx < col_blocks_count; col_block_idx++) {
            nk_size_t const col_block_start = col_block_idx * 32;

            // B tile base indices
            nk_size_t const b_col_left_base = (col_block_idx * 2) * depth_tiles_count;
            nk_size_t const b_col_right_base = (col_block_idx * 2 + 1) * depth_tiles_count;

            // Zero accumulators (TMM4-7 stay resident across entire depth loop)
            _tile_zero(4);
            _tile_zero(5);
            _tile_zero(6);
            _tile_zero(7);

            // Fast path: full row-block with full depth-tiles → direct A load with 2-deep pipelining
            if (is_full_row_block && full_depth_tiles_count > 0) {
                nk_u8_t const *a_upper_base = a + row_block_start * a_stride_bytes;
                nk_u8_t const *a_lower_base = a + (row_block_start + 16) * a_stride_bytes;

                nk_dots_u8_b64x16_sapphire_amx_t const *b_tile_left =
                    (nk_dots_u8_b64x16_sapphire_amx_t const *)(b_tiles_base + b_col_left_base * tile_size);
                nk_dots_u8_b64x16_sapphire_amx_t const *b_tile_right =
                    (nk_dots_u8_b64x16_sapphire_amx_t const *)(b_tiles_base + b_col_right_base * tile_size);

                // Prologue: load first depth tile into TMM0-3
                _tile_loadd(0, a_upper_base, a_stride_bytes);
                _tile_loadd(1, a_lower_base, a_stride_bytes);
                _tile_loadd(2, b_tile_left->data, 64);
                _tile_loadd(3, b_tile_right->data, 64);

                // Main loop: 2-deep software pipelining (compute current while loading next)
                for (nk_size_t depth_tile_idx = 0; depth_tile_idx < full_depth_tiles_count - 1; depth_tile_idx++) {
                    nk_size_t const next_depth_offset = (depth_tile_idx + 1) * tile_depth;

                    // Compute upper row products while loading next A
                    _tile_dpbuud(4, 0, 2);
                    _tile_dpbuud(5, 0, 3);
                    _tile_loadd(0, a_upper_base + next_depth_offset, a_stride_bytes);
                    _tile_loadd(1, a_lower_base + next_depth_offset, a_stride_bytes);

                    // Compute lower row products while loading next B
                    _tile_dpbuud(6, 1, 2);
                    _tile_dpbuud(7, 1, 3);
                    b_tile_left = (nk_dots_u8_b64x16_sapphire_amx_t const *)(b_tiles_base +
                                                                             (b_col_left_base + depth_tile_idx + 1) *
                                                                                 tile_size);
                    b_tile_right = (nk_dots_u8_b64x16_sapphire_amx_t const *)(b_tiles_base +
                                                                              (b_col_right_base + depth_tile_idx + 1) *
                                                                                  tile_size);
                    _tile_loadd(2, b_tile_left->data, 64);
                    _tile_loadd(3, b_tile_right->data, 64);
                }

                // Epilogue: final depth tile (no next to load)
                _tile_dpbuud(4, 0, 2);
                _tile_dpbuud(5, 0, 3);
                _tile_dpbuud(6, 1, 2);
                _tile_dpbuud(7, 1, 3);

                // Handle partial depth-tile (if any) with buffered load
                if (depth_remainder > 0) {
                    nk_size_t const depth_offset = full_depth_tiles_count * tile_depth;

                    nk_dots_u8_load_a_sapphire_amx(&a_tile_upper, a_upper_base + depth_offset, a_stride_bytes, 16,
                                                   depth_remainder);
                    nk_dots_u8_load_a_sapphire_amx(&a_tile_lower, a_lower_base + depth_offset, a_stride_bytes, 16,
                                                   depth_remainder);

                    b_tile_left = (nk_dots_u8_b64x16_sapphire_amx_t const *)(b_tiles_base + (b_col_left_base +
                                                                                             full_depth_tiles_count) *
                                                                                                tile_size);
                    b_tile_right = (nk_dots_u8_b64x16_sapphire_amx_t const *)(b_tiles_base + (b_col_right_base +
                                                                                              full_depth_tiles_count) *
                                                                                                 tile_size);

                    _tile_loadd(0, a_tile_upper.data, 64);
                    _tile_loadd(1, a_tile_lower.data, 64);
                    _tile_loadd(2, b_tile_left->data, 64);
                    _tile_loadd(3, b_tile_right->data, 64);

                    _tile_dpbuud(4, 0, 2);
                    _tile_dpbuud(5, 0, 3);
                    _tile_dpbuud(6, 1, 2);
                    _tile_dpbuud(7, 1, 3);
                }
            }
            // Full row-block but only partial depth tile (depth < tile_depth)
            else if (is_full_row_block) {
                nk_u8_t const *a_upper_base = a + row_block_start * a_stride_bytes;
                nk_u8_t const *a_lower_base = a + (row_block_start + 16) * a_stride_bytes;

                nk_dots_u8_load_a_sapphire_amx(&a_tile_upper, a_upper_base, a_stride_bytes, 16, depth_remainder);
                nk_dots_u8_load_a_sapphire_amx(&a_tile_lower, a_lower_base, a_stride_bytes, 16, depth_remainder);

                nk_dots_u8_b64x16_sapphire_amx_t const *b_tile_left =
                    (nk_dots_u8_b64x16_sapphire_amx_t const *)(b_tiles_base + b_col_left_base * tile_size);
                nk_dots_u8_b64x16_sapphire_amx_t const *b_tile_right =
                    (nk_dots_u8_b64x16_sapphire_amx_t const *)(b_tiles_base + b_col_right_base * tile_size);

                _tile_loadd(0, a_tile_upper.data, 64);
                _tile_loadd(1, a_tile_lower.data, 64);
                _tile_loadd(2, b_tile_left->data, 64);
                _tile_loadd(3, b_tile_right->data, 64);

                _tile_dpbuud(4, 0, 2);
                _tile_dpbuud(5, 0, 3);
                _tile_dpbuud(6, 1, 2);
                _tile_dpbuud(7, 1, 3);
            }
            // Slow path: edge row-block → always use buffered load
            else {
                nk_size_t const rows_in_upper_tile = (valid_rows_count > 16) ? 16 : valid_rows_count;
                nk_size_t const rows_in_lower_tile = (valid_rows_count > 16) ? valid_rows_count - 16 : 0;

                for (nk_size_t depth_tile_idx = 0; depth_tile_idx < depth_tiles_count; depth_tile_idx++) {
                    nk_size_t const depth_offset = depth_tile_idx * tile_depth;
                    nk_size_t const valid_depth = (depth_tile_idx < full_depth_tiles_count) ? tile_depth
                                                                                            : depth_remainder;

                    nk_dots_u8_load_a_sapphire_amx(&a_tile_upper, a + row_block_start * a_stride_bytes + depth_offset,
                                                   a_stride_bytes, rows_in_upper_tile, valid_depth);
                    if (rows_in_lower_tile > 0) {
                        nk_dots_u8_load_a_sapphire_amx(&a_tile_lower,
                                                       a + (row_block_start + 16) * a_stride_bytes + depth_offset,
                                                       a_stride_bytes, rows_in_lower_tile, valid_depth);
                    }

                    nk_dots_u8_b64x16_sapphire_amx_t const *b_tile_left =
                        (nk_dots_u8_b64x16_sapphire_amx_t const *)(b_tiles_base +
                                                                   (b_col_left_base + depth_tile_idx) * tile_size);
                    nk_dots_u8_b64x16_sapphire_amx_t const *b_tile_right =
                        (nk_dots_u8_b64x16_sapphire_amx_t const *)(b_tiles_base +
                                                                   (b_col_right_base + depth_tile_idx) * tile_size);

                    _tile_loadd(0, a_tile_upper.data, 64);
                    _tile_loadd(1, a_tile_lower.data, 64);
                    _tile_loadd(2, b_tile_left->data, 64);
                    _tile_loadd(3, b_tile_right->data, 64);

                    _tile_dpbuud(4, 0, 2);
                    _tile_dpbuud(5, 0, 3);
                    _tile_dpbuud(6, 1, 2);
                    _tile_dpbuud(7, 1, 3);
                }
            }

            // Store accumulators to output (once per output block, not per depth tile)
            if (is_full_row_block) {
                nk_u32_t *c_block = c + row_block_start * c_stride_elements + col_block_start;
                _tile_stored(4, c_block, c_stride_bytes);
                _tile_stored(5, c_block + 16, c_stride_bytes);
                _tile_stored(6, (nk_u32_t *)((char *)c_block + 16 * c_stride_bytes), c_stride_bytes);
                _tile_stored(7, (nk_u32_t *)((char *)c_block + 16 * c_stride_bytes) + 16, c_stride_bytes);
            }
            else {
                _tile_stored(4, c_accum_buffer.c[0][0].data, 64);
                _tile_stored(5, c_accum_buffer.c[0][1].data, 64);
                _tile_stored(6, c_accum_buffer.c[1][0].data, 64);
                _tile_stored(7, c_accum_buffer.c[1][1].data, 64);
                nk_dots_u8_output2x2_sapphire_amx(&c_accum_buffer,
                                                  c + row_block_start * c_stride_elements + col_block_start,
                                                  c_stride_elements, valid_rows_count, 32);
            }
        }

        // Handle odd column-tile (single 16-column tile if col_tiles_count is odd)
        if (col_tiles_count % 2 == 1) {
            nk_size_t const col_tile_idx = col_tiles_count - 1;
            nk_size_t const col_start = col_tile_idx * 16;
            nk_size_t const b_col_base = col_tile_idx * depth_tiles_count;
            nk_size_t const rows_in_upper_tile = (valid_rows_count > 16) ? 16 : valid_rows_count;
            nk_size_t const rows_in_lower_tile = (valid_rows_count > 16) ? valid_rows_count - 16 : 0;

            nk_dots_u8_state_sapphire_amx_t c_upper_state, c_lower_state;

            _tile_zero(4);
            _tile_zero(6);

            for (nk_size_t depth_tile_idx = 0; depth_tile_idx < depth_tiles_count; depth_tile_idx++) {
                nk_size_t const depth_offset = depth_tile_idx * tile_depth;
                nk_size_t const valid_depth = (depth_tile_idx < full_depth_tiles_count) ? tile_depth : depth_remainder;

                nk_dots_u8_load_a_sapphire_amx(&a_tile_upper, a + row_block_start * a_stride_bytes + depth_offset,
                                               a_stride_bytes, rows_in_upper_tile, valid_depth);
                if (rows_in_lower_tile > 0) {
                    nk_dots_u8_load_a_sapphire_amx(&a_tile_lower,
                                                   a + (row_block_start + 16) * a_stride_bytes + depth_offset,
                                                   a_stride_bytes, rows_in_lower_tile, valid_depth);
                }

                nk_dots_u8_b64x16_sapphire_amx_t const *b_tile =
                    (nk_dots_u8_b64x16_sapphire_amx_t const *)(b_tiles_base +
                                                               (b_col_base + depth_tile_idx) * tile_size);

                _tile_loadd(0, a_tile_upper.data, 64);
                _tile_loadd(1, a_tile_lower.data, 64);
                _tile_loadd(2, b_tile->data, 64);

                _tile_dpbuud(4, 0, 2);
                _tile_dpbuud(6, 1, 2);
            }

            _tile_stored(4, c_upper_state.data, 64);
            _tile_stored(6, c_lower_state.data, 64);

            nk_dots_u8_store_sapphire_amx(&c_upper_state, c + row_block_start * c_stride_elements + col_start,
                                          c_stride_elements, rows_in_upper_tile, 16);
            if (rows_in_lower_tile > 0) {
                nk_dots_u8_store_sapphire_amx(&c_lower_state,
                                              c + (row_block_start + 16) * c_stride_elements + col_start,
                                              c_stride_elements, rows_in_lower_tile, 16);
            }
        }

        // Handle column-edge (remaining columns < 16) using AMX with partial tiles
        if (col_edge_count > 0) {
            nk_size_t const rows_in_upper_tile = (valid_rows_count > 16) ? 16 : valid_rows_count;
            nk_size_t const rows_in_lower_tile = (valid_rows_count > 16) ? valid_rows_count - 16 : 0;

            nk_dots_u8_state_sapphire_amx_t c_upper_state, c_lower_state;
            nk_dots_u8_a16x64_sapphire_amx_t b_as_a;
            nk_dots_u8_b64x16_sapphire_amx_t b_tile;

            _tile_zero(4);
            _tile_zero(6);

            for (nk_size_t depth_tile_idx = 0; depth_tile_idx < depth_tiles_count; depth_tile_idx++) {
                nk_size_t const depth_offset = depth_tile_idx * tile_depth;
                nk_size_t const valid_depth = (depth_tile_idx < full_depth_tiles_count) ? tile_depth : depth_remainder;

                nk_dots_u8_load_a_sapphire_amx(&a_tile_upper, a + row_block_start * a_stride_bytes + depth_offset,
                                               a_stride_bytes, rows_in_upper_tile, valid_depth);
                if (rows_in_lower_tile > 0) {
                    nk_dots_u8_load_a_sapphire_amx(&a_tile_lower,
                                                   a + (row_block_start + 16) * a_stride_bytes + depth_offset,
                                                   a_stride_bytes, rows_in_lower_tile, valid_depth);
                }

                nk_dots_u8_load_a_sapphire_amx(&b_as_a, col_edge_ptr + depth_offset, depth, col_edge_count,
                                               valid_depth);
                nk_dots_pack_u8_transposed_sapphire_amx(&b_as_a, &b_tile);

                _tile_loadd(0, a_tile_upper.data, 64);
                _tile_loadd(1, a_tile_lower.data, 64);
                _tile_loadd(2, b_tile.data, 64);

                _tile_dpbuud(4, 0, 2);
                _tile_dpbuud(6, 1, 2);
            }

            _tile_stored(4, c_upper_state.data, 64);
            _tile_stored(6, c_lower_state.data, 64);

            nk_dots_u8_store_sapphire_amx(&c_upper_state, c + row_block_start * c_stride_elements + full_cols,
                                          c_stride_elements, rows_in_upper_tile, col_edge_count);
            if (rows_in_lower_tile > 0) {
                nk_dots_u8_store_sapphire_amx(&c_lower_state,
                                              c + (row_block_start + 16) * c_stride_elements + full_cols,
                                              c_stride_elements, rows_in_lower_tile, col_edge_count);
            }
        }
    }

    _tile_release();
}

/*  E4M3 x E4M3 → F32 matmul using BF16 AMX path with on-the-fly conversion.
 *  B matrix is pre-packed with FP8 → BF16 conversion.
 *  A matrix is converted from E4M3 to BF16 during tile loading.
 *
 *  Uses register-resident 2 × 2 update pattern:
 *  - TMM4-7 hold C accumulators across entire depth-loop (no redundant load/store)
 *  - 32 × 32 output blocks processed per row x column iteration
 */
NK_PUBLIC void nk_dots_packed_e4m3_sapphire_amx(           //
    nk_e4m3_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows_count, nk_size_t cols_count, nk_size_t depth, nk_size_t a_stride_bytes, nk_size_t c_stride_bytes) {

    nk_dots_amx_packed_header_t const *header = (nk_dots_amx_packed_header_t const *)b_packed;
    nk_size_t const col_tiles_count = header->full_column_tiles;
    nk_size_t const depth_tiles_count = header->full_depth_tiles;
    nk_size_t const col_edge_count = header->column_edge_rows;

    // B tiles are already in BF16 format
    nk_bf16_t const *b_tiles_base = (nk_bf16_t const *)((char const *)b_packed + sizeof(nk_dots_amx_packed_header_t));
    nk_bf16_t const *col_edge_ptr = (nk_bf16_t const *)((char const *)b_packed + header->column_edge_offset);

    nk_size_t const c_stride_elements = c_stride_bytes / sizeof(nk_f32_t);
    nk_size_t const tile_depth = 32;
    nk_size_t const tile_size = 512;
    nk_size_t const full_cols = col_tiles_count * 16;

    nk_size_t const row_blocks_count = (rows_count + 31) / 32;
    nk_size_t const col_blocks_count = col_tiles_count / 2;

    if (depth_tiles_count == 0) return;

    nk_dots_bf16_a16x32_sapphire_amx_t a_tile_upper, a_tile_lower;
    nk_dots_bf16_state2x2_sapphire_amx_t c_accum_buffer;

    nk_size_t const full_depth_tiles_count = depth / tile_depth;
    nk_size_t const depth_remainder = depth % tile_depth;

    nk_amx_tile_configure_sapphire_amx_();

    // Loop order: row_blocks outer, col_blocks inner
    for (nk_size_t row_block_idx = 0; row_block_idx < row_blocks_count; row_block_idx++) {
        nk_size_t const row_block_start = row_block_idx * 32;
        nk_size_t const valid_rows_count = (row_block_start + 32 <= rows_count) ? 32 : (rows_count - row_block_start);
        nk_size_t const is_full_row_block = (valid_rows_count == 32);
        nk_size_t const rows_in_upper_tile = (valid_rows_count > 16) ? 16 : valid_rows_count;
        nk_size_t const rows_in_lower_tile = (valid_rows_count > 16) ? valid_rows_count - 16 : 0;

        for (nk_size_t col_block_idx = 0; col_block_idx < col_blocks_count; col_block_idx++) {
            nk_size_t const col_block_start = col_block_idx * 32;
            nk_size_t const b_col_left_base = (col_block_idx * 2) * depth_tiles_count;
            nk_size_t const b_col_right_base = (col_block_idx * 2 + 1) * depth_tiles_count;

            // Zero accumulators (TMM4-7 stay resident across entire depth loop)
            _tile_zero(4);
            _tile_zero(5);
            _tile_zero(6);
            _tile_zero(7);

            // FP8 always uses buffered load for E4M3 → BF16 conversion
            for (nk_size_t depth_tile_idx = 0; depth_tile_idx < depth_tiles_count; depth_tile_idx++) {
                nk_size_t const depth_offset = depth_tile_idx * tile_depth;
                nk_size_t const valid_depth = (depth_tile_idx < full_depth_tiles_count) ? tile_depth : depth_remainder;

                // Load A with FP8 → BF16 conversion
                nk_dots_e4m3_load_a_sapphire_amx(&a_tile_upper, a + row_block_start * a_stride_bytes + depth_offset,
                                                 a_stride_bytes, rows_in_upper_tile, valid_depth);
                if (rows_in_lower_tile > 0) {
                    nk_dots_e4m3_load_a_sapphire_amx(&a_tile_lower,
                                                     a + (row_block_start + 16) * a_stride_bytes + depth_offset,
                                                     a_stride_bytes, rows_in_lower_tile, valid_depth);
                }

                nk_dots_bf16_b32x16_sapphire_amx_t const *b_tile_left =
                    (nk_dots_bf16_b32x16_sapphire_amx_t const *)(b_tiles_base +
                                                                 (b_col_left_base + depth_tile_idx) * tile_size);
                nk_dots_bf16_b32x16_sapphire_amx_t const *b_tile_right =
                    (nk_dots_bf16_b32x16_sapphire_amx_t const *)(b_tiles_base +
                                                                 (b_col_right_base + depth_tile_idx) * tile_size);

                _tile_loadd(0, a_tile_upper.data, 64);
                _tile_loadd(1, a_tile_lower.data, 64);
                _tile_loadd(2, b_tile_left->data, 64);
                _tile_loadd(3, b_tile_right->data, 64);

                _tile_dpbf16ps(4, 0, 2);
                _tile_dpbf16ps(5, 0, 3);
                _tile_dpbf16ps(6, 1, 2);
                _tile_dpbf16ps(7, 1, 3);
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
                nk_dots_bf16_output2x2_sapphire_amx(&c_accum_buffer,
                                                    c + row_block_start * c_stride_elements + col_block_start,
                                                    c_stride_elements, valid_rows_count, 32);
            }
        }

        // Handle odd column-tile (single 16-column tile if col_tiles_count is odd)
        if (col_tiles_count % 2 == 1) {
            nk_size_t const col_tile_idx = col_tiles_count - 1;
            nk_size_t const col_start = col_tile_idx * 16;
            nk_size_t const b_col_base = col_tile_idx * depth_tiles_count;

            nk_dots_bf16_state_sapphire_amx_t c_upper_state, c_lower_state;
            _tile_zero(4);
            _tile_zero(6);

            for (nk_size_t depth_tile_idx = 0; depth_tile_idx < depth_tiles_count; depth_tile_idx++) {
                nk_size_t const depth_offset = depth_tile_idx * tile_depth;
                nk_size_t const valid_depth = (depth_tile_idx < full_depth_tiles_count) ? tile_depth : depth_remainder;

                nk_dots_e4m3_load_a_sapphire_amx(&a_tile_upper, a + row_block_start * a_stride_bytes + depth_offset,
                                                 a_stride_bytes, rows_in_upper_tile, valid_depth);
                if (rows_in_lower_tile > 0) {
                    nk_dots_e4m3_load_a_sapphire_amx(&a_tile_lower,
                                                     a + (row_block_start + 16) * a_stride_bytes + depth_offset,
                                                     a_stride_bytes, rows_in_lower_tile, valid_depth);
                }

                nk_dots_bf16_b32x16_sapphire_amx_t const *b_tile =
                    (nk_dots_bf16_b32x16_sapphire_amx_t const *)(b_tiles_base +
                                                                 (b_col_base + depth_tile_idx) * tile_size);

                _tile_loadd(0, a_tile_upper.data, 64);
                _tile_loadd(1, a_tile_lower.data, 64);
                _tile_loadd(2, b_tile->data, 64);

                _tile_dpbf16ps(4, 0, 2);
                _tile_dpbf16ps(6, 1, 2);
            }

            _tile_stored(4, c_upper_state.data, 64);
            _tile_stored(6, c_lower_state.data, 64);

            nk_dots_bf16_store_sapphire_amx(&c_upper_state, c + row_block_start * c_stride_elements + col_start,
                                            c_stride_elements, rows_in_upper_tile, 16);
            if (rows_in_lower_tile > 0) {
                nk_dots_bf16_store_sapphire_amx(&c_lower_state,
                                                c + (row_block_start + 16) * c_stride_elements + col_start,
                                                c_stride_elements, rows_in_lower_tile, 16);
            }
        }

        // Handle column-edge (remaining columns < 16) using AMX with partial tiles
        if (col_edge_count > 0) {
            nk_dots_bf16_state_sapphire_amx_t c_upper_state, c_lower_state;
            nk_dots_bf16_a16x32_sapphire_amx_t b_as_a;
            nk_dots_bf16_b32x16_sapphire_amx_t b_tile;

            _tile_zero(4);
            _tile_zero(6);

            for (nk_size_t depth_tile_idx = 0; depth_tile_idx < depth_tiles_count; depth_tile_idx++) {
                nk_size_t const depth_offset = depth_tile_idx * tile_depth;
                nk_size_t const valid_depth = (depth_tile_idx < full_depth_tiles_count) ? tile_depth : depth_remainder;

                nk_dots_e4m3_load_a_sapphire_amx(&a_tile_upper, a + row_block_start * a_stride_bytes + depth_offset,
                                                 a_stride_bytes, rows_in_upper_tile, valid_depth);
                if (rows_in_lower_tile > 0) {
                    nk_dots_e4m3_load_a_sapphire_amx(&a_tile_lower,
                                                     a + (row_block_start + 16) * a_stride_bytes + depth_offset,
                                                     a_stride_bytes, rows_in_lower_tile, valid_depth);
                }

                // B edge data is already in BF16 format
                nk_dots_bf16_load_a_sapphire_amx(&b_as_a, col_edge_ptr + depth_offset, depth, col_edge_count,
                                                 valid_depth);
                nk_dots_pack_bf16_transposed_sapphire_amx(&b_as_a, &b_tile);

                _tile_loadd(0, a_tile_upper.data, 64);
                _tile_loadd(1, a_tile_lower.data, 64);
                _tile_loadd(2, b_tile.data, 64);

                _tile_dpbf16ps(4, 0, 2);
                _tile_dpbf16ps(6, 1, 2);
            }

            _tile_stored(4, c_upper_state.data, 64);
            _tile_stored(6, c_lower_state.data, 64);

            nk_dots_bf16_store_sapphire_amx(&c_upper_state, c + row_block_start * c_stride_elements + full_cols,
                                            c_stride_elements, rows_in_upper_tile, col_edge_count);
            if (rows_in_lower_tile > 0) {
                nk_dots_bf16_store_sapphire_amx(&c_lower_state,
                                                c + (row_block_start + 16) * c_stride_elements + full_cols,
                                                c_stride_elements, rows_in_lower_tile, col_edge_count);
            }
        }
    }

    _tile_release();
}

/*  E5M2 × E5M2 → F32 matmul using BF16 AMX path with on-the-fly conversion.
 *
 *  Uses register-resident 2 × 2 update pattern:
 *  - TMM4-7 hold C accumulators across entire depth-loop (no redundant load/store)
 *  - 32 × 32 output blocks processed per row x column iteration
 */
NK_PUBLIC void nk_dots_packed_e5m2_sapphire_amx(           //
    nk_e5m2_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows_count, nk_size_t cols_count, nk_size_t depth, nk_size_t a_stride_bytes, nk_size_t c_stride_bytes) {

    nk_dots_amx_packed_header_t const *header = (nk_dots_amx_packed_header_t const *)b_packed;
    nk_size_t const col_tiles_count = header->full_column_tiles;
    nk_size_t const depth_tiles_count = header->full_depth_tiles;
    nk_size_t const col_edge_count = header->column_edge_rows;

    nk_bf16_t const *b_tiles_base = (nk_bf16_t const *)((char const *)b_packed + sizeof(nk_dots_amx_packed_header_t));
    nk_bf16_t const *col_edge_ptr = (nk_bf16_t const *)((char const *)b_packed + header->column_edge_offset);

    nk_size_t const c_stride_elements = c_stride_bytes / sizeof(nk_f32_t);
    nk_size_t const tile_depth = 32;
    nk_size_t const tile_size = 512;
    nk_size_t const full_cols = col_tiles_count * 16;

    nk_size_t const row_blocks_count = (rows_count + 31) / 32;
    nk_size_t const col_blocks_count = col_tiles_count / 2;

    if (depth_tiles_count == 0) return;

    nk_dots_bf16_a16x32_sapphire_amx_t a_tile_upper, a_tile_lower;
    nk_dots_bf16_state2x2_sapphire_amx_t c_accum_buffer;

    nk_size_t const full_depth_tiles_count = depth / tile_depth;
    nk_size_t const depth_remainder = depth % tile_depth;

    nk_amx_tile_configure_sapphire_amx_();

    // Loop order: row_blocks outer, col_blocks inner
    for (nk_size_t row_block_idx = 0; row_block_idx < row_blocks_count; row_block_idx++) {
        nk_size_t const row_block_start = row_block_idx * 32;
        nk_size_t const valid_rows_count = (row_block_start + 32 <= rows_count) ? 32 : (rows_count - row_block_start);
        nk_size_t const is_full_row_block = (valid_rows_count == 32);
        nk_size_t const rows_in_upper_tile = (valid_rows_count > 16) ? 16 : valid_rows_count;
        nk_size_t const rows_in_lower_tile = (valid_rows_count > 16) ? valid_rows_count - 16 : 0;

        for (nk_size_t col_block_idx = 0; col_block_idx < col_blocks_count; col_block_idx++) {
            nk_size_t const col_block_start = col_block_idx * 32;
            nk_size_t const b_col_left_base = (col_block_idx * 2) * depth_tiles_count;
            nk_size_t const b_col_right_base = (col_block_idx * 2 + 1) * depth_tiles_count;

            // Zero accumulators (TMM4-7 stay resident across entire depth loop)
            _tile_zero(4);
            _tile_zero(5);
            _tile_zero(6);
            _tile_zero(7);

            // FP8 always uses buffered load for E5M2 → BF16 conversion
            for (nk_size_t depth_tile_idx = 0; depth_tile_idx < depth_tiles_count; depth_tile_idx++) {
                nk_size_t const depth_offset = depth_tile_idx * tile_depth;
                nk_size_t const valid_depth = (depth_tile_idx < full_depth_tiles_count) ? tile_depth : depth_remainder;

                // Load A with FP8 → BF16 conversion
                nk_dots_e5m2_load_a_sapphire_amx(&a_tile_upper, a + row_block_start * a_stride_bytes + depth_offset,
                                                 a_stride_bytes, rows_in_upper_tile, valid_depth);
                if (rows_in_lower_tile > 0) {
                    nk_dots_e5m2_load_a_sapphire_amx(&a_tile_lower,
                                                     a + (row_block_start + 16) * a_stride_bytes + depth_offset,
                                                     a_stride_bytes, rows_in_lower_tile, valid_depth);
                }

                nk_dots_bf16_b32x16_sapphire_amx_t const *b_tile_left =
                    (nk_dots_bf16_b32x16_sapphire_amx_t const *)(b_tiles_base +
                                                                 (b_col_left_base + depth_tile_idx) * tile_size);
                nk_dots_bf16_b32x16_sapphire_amx_t const *b_tile_right =
                    (nk_dots_bf16_b32x16_sapphire_amx_t const *)(b_tiles_base +
                                                                 (b_col_right_base + depth_tile_idx) * tile_size);

                _tile_loadd(0, a_tile_upper.data, 64);
                _tile_loadd(1, a_tile_lower.data, 64);
                _tile_loadd(2, b_tile_left->data, 64);
                _tile_loadd(3, b_tile_right->data, 64);

                _tile_dpbf16ps(4, 0, 2);
                _tile_dpbf16ps(5, 0, 3);
                _tile_dpbf16ps(6, 1, 2);
                _tile_dpbf16ps(7, 1, 3);
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
                nk_dots_bf16_output2x2_sapphire_amx(&c_accum_buffer,
                                                    c + row_block_start * c_stride_elements + col_block_start,
                                                    c_stride_elements, valid_rows_count, 32);
            }
        }

        // Handle odd column-tile (single 16-column tile if col_tiles_count is odd)
        if (col_tiles_count % 2 == 1) {
            nk_size_t const col_tile_idx = col_tiles_count - 1;
            nk_size_t const col_start = col_tile_idx * 16;
            nk_size_t const b_col_base = col_tile_idx * depth_tiles_count;

            nk_dots_bf16_state_sapphire_amx_t c_upper_state, c_lower_state;
            _tile_zero(4);
            _tile_zero(6);

            for (nk_size_t depth_tile_idx = 0; depth_tile_idx < depth_tiles_count; depth_tile_idx++) {
                nk_size_t const depth_offset = depth_tile_idx * tile_depth;
                nk_size_t const valid_depth = (depth_tile_idx < full_depth_tiles_count) ? tile_depth : depth_remainder;

                nk_dots_e5m2_load_a_sapphire_amx(&a_tile_upper, a + row_block_start * a_stride_bytes + depth_offset,
                                                 a_stride_bytes, rows_in_upper_tile, valid_depth);
                if (rows_in_lower_tile > 0) {
                    nk_dots_e5m2_load_a_sapphire_amx(&a_tile_lower,
                                                     a + (row_block_start + 16) * a_stride_bytes + depth_offset,
                                                     a_stride_bytes, rows_in_lower_tile, valid_depth);
                }

                nk_dots_bf16_b32x16_sapphire_amx_t const *b_tile =
                    (nk_dots_bf16_b32x16_sapphire_amx_t const *)(b_tiles_base +
                                                                 (b_col_base + depth_tile_idx) * tile_size);

                _tile_loadd(0, a_tile_upper.data, 64);
                _tile_loadd(1, a_tile_lower.data, 64);
                _tile_loadd(2, b_tile->data, 64);

                _tile_dpbf16ps(4, 0, 2);
                _tile_dpbf16ps(6, 1, 2);
            }

            _tile_stored(4, c_upper_state.data, 64);
            _tile_stored(6, c_lower_state.data, 64);

            nk_dots_bf16_store_sapphire_amx(&c_upper_state, c + row_block_start * c_stride_elements + col_start,
                                            c_stride_elements, rows_in_upper_tile, 16);
            if (rows_in_lower_tile > 0) {
                nk_dots_bf16_store_sapphire_amx(&c_lower_state,
                                                c + (row_block_start + 16) * c_stride_elements + col_start,
                                                c_stride_elements, rows_in_lower_tile, 16);
            }
        }

        // Handle column-edge (remaining columns < 16) using AMX with partial tiles
        if (col_edge_count > 0) {
            nk_dots_bf16_state_sapphire_amx_t c_upper_state, c_lower_state;
            nk_dots_bf16_a16x32_sapphire_amx_t b_as_a;
            nk_dots_bf16_b32x16_sapphire_amx_t b_tile;

            _tile_zero(4);
            _tile_zero(6);

            for (nk_size_t depth_tile_idx = 0; depth_tile_idx < depth_tiles_count; depth_tile_idx++) {
                nk_size_t const depth_offset = depth_tile_idx * tile_depth;
                nk_size_t const valid_depth = (depth_tile_idx < full_depth_tiles_count) ? tile_depth : depth_remainder;

                nk_dots_e5m2_load_a_sapphire_amx(&a_tile_upper, a + row_block_start * a_stride_bytes + depth_offset,
                                                 a_stride_bytes, rows_in_upper_tile, valid_depth);
                if (rows_in_lower_tile > 0) {
                    nk_dots_e5m2_load_a_sapphire_amx(&a_tile_lower,
                                                     a + (row_block_start + 16) * a_stride_bytes + depth_offset,
                                                     a_stride_bytes, rows_in_lower_tile, valid_depth);
                }

                nk_dots_bf16_load_a_sapphire_amx(&b_as_a, col_edge_ptr + depth_offset, depth, col_edge_count,
                                                 valid_depth);
                nk_dots_pack_bf16_transposed_sapphire_amx(&b_as_a, &b_tile);

                _tile_loadd(0, a_tile_upper.data, 64);
                _tile_loadd(1, a_tile_lower.data, 64);
                _tile_loadd(2, b_tile.data, 64);

                _tile_dpbf16ps(4, 0, 2);
                _tile_dpbf16ps(6, 1, 2);
            }

            _tile_stored(4, c_upper_state.data, 64);
            _tile_stored(6, c_lower_state.data, 64);

            nk_dots_bf16_store_sapphire_amx(&c_upper_state, c + row_block_start * c_stride_elements + full_cols,
                                            c_stride_elements, rows_in_upper_tile, col_edge_count);
            if (rows_in_lower_tile > 0) {
                nk_dots_bf16_store_sapphire_amx(&c_lower_state,
                                                c + (row_block_start + 16) * c_stride_elements + full_cols,
                                                c_stride_elements, rows_in_lower_tile, col_edge_count);
            }
        }
    }

    _tile_release();
}

/*  I8 compact: re-normalize I32 → I8 using precomputed squared norms.
 *  Formula: c_i8[row][col] = c_i32[row][col] × 127 × rsqrt(a_norm[row] × b_norm[col])
 *  Uses AVX512 rsqrt14 with Newton-Raphson refinement for 16 elements at a time.
 *  Output is tightly packed with stride = column_count × sizeof(i8).
 */
NK_PUBLIC void nk_dots_compact_i8_sapphire_amx( //
    void *c, nk_size_t row_count, nk_size_t column_count, nk_size_t c_stride, nk_i32_t const *a_squared_norms,
    nk_i32_t const *b_squared_norms) {

    nk_size_t const c_stride_i32 = c_stride / sizeof(nk_i32_t);
    nk_i32_t const *c_i32 = (nk_i32_t const *)c;
    nk_i8_t *c_i8 = (nk_i8_t *)c;

    // Use space after I8 output for precomputed b_rsqrt (I8 output is 4x smaller than I32 input)
    nk_f32_t *b_rsqrt = (nk_f32_t *)(c_i8 + row_count * column_count);

    // Precompute rsqrt of all b_norms using AVX512 (16 at a time)
    __m512 half_vec = _mm512_set1_ps(0.5f);
    __m512 three_halves_vec = _mm512_set1_ps(1.5f);
    nk_size_t column_idx = 0;

    for (; column_idx + 16 <= column_count; column_idx += 16) {
        __m512i b_norms_i32 = _mm512_loadu_si512(b_squared_norms + column_idx);
        __m512 b_norms_f32 = _mm512_cvtepi32_ps(b_norms_i32);
        __m512 rsqrt_vec = _mm512_rsqrt14_ps(b_norms_f32);
        // Newton-Raphson refinement
        rsqrt_vec = _mm512_mul_ps(
            rsqrt_vec,
            _mm512_sub_ps(three_halves_vec,
                          _mm512_mul_ps(half_vec, _mm512_mul_ps(b_norms_f32, _mm512_mul_ps(rsqrt_vec, rsqrt_vec)))));
        // Zero out rsqrt where norm was zero
        __mmask16 nonzero_mask = _mm512_cmpneq_epi32_mask(b_norms_i32, _mm512_setzero_si512());
        rsqrt_vec = _mm512_maskz_mov_ps(nonzero_mask, rsqrt_vec);
        _mm512_storeu_ps(b_rsqrt + column_idx, rsqrt_vec);
    }

    // Handle remaining b_norms with masked operations
    if (column_idx < column_count) {
        __mmask16 tail_mask = (__mmask16)((1u << (column_count - column_idx)) - 1);
        __m512i b_norms_i32 = _mm512_maskz_loadu_epi32(tail_mask, b_squared_norms + column_idx);
        __m512 b_norms_f32 = _mm512_cvtepi32_ps(b_norms_i32);
        __m512 rsqrt_vec = _mm512_rsqrt14_ps(b_norms_f32);
        rsqrt_vec = _mm512_mul_ps(
            rsqrt_vec,
            _mm512_sub_ps(three_halves_vec,
                          _mm512_mul_ps(half_vec, _mm512_mul_ps(b_norms_f32, _mm512_mul_ps(rsqrt_vec, rsqrt_vec)))));
        __mmask16 nonzero_mask = _mm512_cmpneq_epi32_mask(b_norms_i32, _mm512_setzero_si512());
        rsqrt_vec = _mm512_maskz_mov_ps(nonzero_mask & tail_mask, rsqrt_vec);
        _mm512_mask_storeu_ps(b_rsqrt + column_idx, tail_mask, rsqrt_vec);
    }

    __m512 scale_vec = _mm512_set1_ps(127.0f);

    for (nk_size_t row_idx = 0; row_idx < row_count; row_idx++) {
        nk_i32_t const *src_row = c_i32 + row_idx * c_stride_i32;
        nk_i8_t *dst_row = c_i8 + row_idx * column_count;

        // Compute rsqrt of a_norm for this row, broadcast to vector
        nk_f32_t a_norm_f32 = (nk_f32_t)a_squared_norms[row_idx];
        nk_f32_t a_rsqrt_val = 0.0f;
        if (a_norm_f32 > 0.0f) {
            __m128 a_vec = _mm_set_ss(a_norm_f32);
            __m128 rsqrt_s = _mm_rsqrt_ss(a_vec);
            rsqrt_s = _mm_mul_ss(
                rsqrt_s, _mm_sub_ss(_mm_set_ss(1.5f),
                                    _mm_mul_ss(_mm_set_ss(0.5f), _mm_mul_ss(a_vec, _mm_mul_ss(rsqrt_s, rsqrt_s)))));
            a_rsqrt_val = _mm_cvtss_f32(rsqrt_s);
        }
        __m512 a_rsqrt_vec = _mm512_set1_ps(a_rsqrt_val);
        __m512 row_scale = _mm512_mul_ps(a_rsqrt_vec, scale_vec);

        column_idx = 0;

        // Process 16 elements at a time
        for (; column_idx + 16 <= column_count; column_idx += 16) {
            __m512i c_vals = _mm512_loadu_si512(src_row + column_idx);
            __m512 c_f32 = _mm512_cvtepi32_ps(c_vals);
            __m512 b_rsqrt_vec = _mm512_loadu_ps(b_rsqrt + column_idx);
            __m512 normalized = _mm512_mul_ps(_mm512_mul_ps(c_f32, row_scale), b_rsqrt_vec);
            __m512i result_i32 = _mm512_cvtps_epi32(normalized);
            // Saturating pack I32 → I8 (16 values → 16 bytes in low 128 bits)
            __m128i result_i8 = _mm512_cvtsepi32_epi8(result_i32);
            _mm_storeu_si128((__m128i *)(dst_row + column_idx), result_i8);
        }

        // Handle remaining elements with masked operations
        if (column_idx < column_count) {
            __mmask16 tail_mask = (__mmask16)((1u << (column_count - column_idx)) - 1);
            __m512i c_vals = _mm512_maskz_loadu_epi32(tail_mask, src_row + column_idx);
            __m512 c_f32 = _mm512_cvtepi32_ps(c_vals);
            __m512 b_rsqrt_vec = _mm512_maskz_loadu_ps(tail_mask, b_rsqrt + column_idx);
            __m512 normalized = _mm512_mul_ps(_mm512_mul_ps(c_f32, row_scale), b_rsqrt_vec);
            __m512i result_i32 = _mm512_cvtps_epi32(normalized);
            __m128i result_i8 = _mm512_cvtsepi32_epi8(result_i32);
            _mm_mask_storeu_epi8(dst_row + column_idx, tail_mask, result_i8);
        }
    }
}

#if defined(__cplusplus)
} // extern "C"
#endif

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // NK_TARGET_SAPPHIRE_AMX
#endif // NK_TARGET_X86_

#endif // NK_DOTS_SAPPHIRE_AMX_H

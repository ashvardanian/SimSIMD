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
 */
#ifndef NK_DOTS_SAPPHIRE_AMX_H
#define NK_DOTS_SAPPHIRE_AMX_H

#if NK_TARGET_X86_
#if NK_TARGET_SAPPHIRE_AMX
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "bmi2", "avx512bw", "avx512fp16")
#pragma clang attribute push(__attribute__((target("avx2,avx512f,avx512vl,bmi2,avx512bw,avx512fp16"))), \
                             apply_to = function)

#include "numkong/types.h"

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
} nk_dots_bf16bf16f32_a16x32_sapphire_amx_t;

/*  BF16 B tile: 32 depth × 16 columns, pair-interleaved for TDPBF16PS.
 *  Access pattern: data[depth/2][column][depth%2] for logical B[depth, column].
 *  Pre-packed from column-major or transposed source.
 */
typedef struct {
    NK_ALIGN64 nk_bf16_t data[16][16][2]; // 16 depth-groups × 16 columns × 2 = 1KB
} nk_dots_bf16bf16f32_b32x16_sapphire_amx_t;

/*  BF16 output state: 16 × 16 F32 accumulator tile.
 *  Holds partial sums during depth-dimension accumulation.
 */
typedef struct {
    NK_ALIGN64 nk_f32_t data[16][16]; // 16 × 16 = 1KB
} nk_dots_bf16bf16f32_state_sapphire_amx_t;

/*  INT8 A tile: 16 rows × 64 depth-elements, row-major layout.
 */
typedef struct {
    NK_ALIGN64 nk_i8_t data[16][64]; // 16 rows × 64 cols = 1KB
} nk_dots_i8i8i32_a16x64_sapphire_amx_t;

/*  INT8 B tile: 64 depth × 16 columns, quad-interleaved for TDPBSSD.
 *  Access pattern: data[depth/4][column][depth%4] for logical B[depth, column].
 */
typedef struct {
    NK_ALIGN64 nk_i8_t data[16][16][4]; // 16 depth-groups × 16 columns × 4 = 1KB
} nk_dots_i8i8i32_b64x16_sapphire_amx_t;

/*  INT8 output state: 16 × 16 I32 accumulator tile.
 */
typedef struct {
    NK_ALIGN64 nk_i32_t data[16][16]; // 16 × 16 = 1KB
} nk_dots_i8i8i32_state_sapphire_amx_t;

/*  BF16 2 × 2 output state: 32 × 32 F32 output (4 accumulator tiles).
 *  Used for GEMM's 2 × 2 output blocking pattern.
 */
typedef struct {
    nk_dots_bf16bf16f32_state_sapphire_amx_t c[2][2]; // 4KB total
} nk_dots_bf16bf16f32_state2x2_sapphire_amx_t;

/*  INT8 2 × 2 output state: 32 × 32 I32 output (4 accumulator tiles).
 */
typedef struct {
    nk_dots_i8i8i32_state_sapphire_amx_t c[2][2]; // 4KB total
} nk_dots_i8i8i32_state2x2_sapphire_amx_t;

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
NK_INTERNAL void nk_dots_bf16bf16f32_init_sapphire_amx(nk_dots_bf16bf16f32_state_sapphire_amx_t *state) {
    __m512 zero = _mm512_setzero_ps();
    for (nk_size_t row_idx = 0; row_idx < 16; row_idx++) { _mm512_store_ps(state->data[row_idx], zero); }
}

/*  Load A tile from row-major source with masking for edge tiles.
 */
NK_INTERNAL void nk_dots_bf16bf16f32_load_a_sapphire_amx( //
    nk_dots_bf16bf16f32_a16x32_sapphire_amx_t *a_tile,    //
    nk_bf16_t const *src, nk_size_t src_stride_elements,  //
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
NK_INTERNAL void nk_dots_bf16bf16f32_store_sapphire_amx(   //
    nk_dots_bf16bf16f32_state_sapphire_amx_t const *state, //
    nk_f32_t *dst, nk_size_t dst_stride_elements,          //
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
NK_INTERNAL void nk_dots_bf16bf16f32_update_sapphire_amx(      //
    nk_dots_bf16bf16f32_state_sapphire_amx_t *state,           //
    nk_dots_bf16bf16f32_a16x32_sapphire_amx_t const *a_tile_0, //
    nk_dots_bf16bf16f32_a16x32_sapphire_amx_t const *a_tile_1, //
    nk_dots_bf16bf16f32_a16x32_sapphire_amx_t const *a_tile_2, //
    nk_dots_bf16bf16f32_b32x16_sapphire_amx_t const *b_tile_0, //
    nk_dots_bf16bf16f32_b32x16_sapphire_amx_t const *b_tile_1, //
    nk_dots_bf16bf16f32_b32x16_sapphire_amx_t const *b_tile_2) {

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
NK_INTERNAL void nk_dots_i8i8i32_init_sapphire_amx(nk_dots_i8i8i32_state_sapphire_amx_t *state) {
    __m512i zero = _mm512_setzero_si512();
    for (nk_size_t row_idx = 0; row_idx < 16; row_idx++) { _mm512_store_si512((__m512i *)state->data[row_idx], zero); }
}

/*  Load A tile from row-major source with masking for edge tiles.
 */
NK_INTERNAL void nk_dots_i8i8i32_load_a_sapphire_amx( //
    nk_dots_i8i8i32_a16x64_sapphire_amx_t *a_tile,    //
    nk_i8_t const *src, nk_size_t src_stride,         //
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
NK_INTERNAL void nk_dots_i8i8i32_store_sapphire_amx(   //
    nk_dots_i8i8i32_state_sapphire_amx_t const *state, //
    nk_i32_t *dst, nk_size_t dst_stride_elements,      //
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
NK_INTERNAL void nk_dots_i8i8i32_update_sapphire_amx(      //
    nk_dots_i8i8i32_state_sapphire_amx_t *state,           //
    nk_dots_i8i8i32_a16x64_sapphire_amx_t const *a_tile_0, //
    nk_dots_i8i8i32_a16x64_sapphire_amx_t const *a_tile_1, //
    nk_dots_i8i8i32_a16x64_sapphire_amx_t const *a_tile_2, //
    nk_dots_i8i8i32_b64x16_sapphire_amx_t const *b_tile_0, //
    nk_dots_i8i8i32_b64x16_sapphire_amx_t const *b_tile_1, //
    nk_dots_i8i8i32_b64x16_sapphire_amx_t const *b_tile_2) {

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

/*  Register-Resident 2 × 2 Update Primitives for GEMM
 *
 *  These keep C accumulators in TMM4-7 across the entire depth-loop, avoiding
 *  4KB of load+store per depth iteration. Usage pattern:
 *
 *    nk_dots_bf16bf16f32_zero2x2_sapphire_amx();
 *    for (depth_tile_idx = 0; depth_tile_idx < num_depth_tiles; depth_tile_idx++) {
 *        // prepare a_tile_0, a_tile_1, b_tile_0, b_tile_1
 *        nk_dots_bf16bf16f32_update2x2_sapphire_amx(&a_tile_0, &a_tile_1, &b_tile_0, &b_tile_1);
 *    }
 *    nk_dots_bf16bf16f32_store2x2_sapphire_amx(&state);
 *
 *  Register allocation:
 *    TMM0-1: A tiles (rows 0-15 and 16-31)
 *    TMM2-3: B tiles (columns 0-15 and 16-31)
 *    TMM4-7: C accumulators (2 × 2 output grid, kept in registers)
 */

/*  Zero BF16 2 × 2 C accumulators in TMM4-7 (call once before depth-loop).
 */
NK_INTERNAL void nk_dots_bf16bf16f32_zero2x2_sapphire_amx(void) {
    _tile_zero(4); // C[0,0]
    _tile_zero(5); // C[0,1]
    _tile_zero(6); // C[1,0]
    _tile_zero(7); // C[1,1]
}

/*  Accumulate BF16 2 × 2 output: C[row_idx, column_idx] += A[row_idx] × B[column_idx] for row_idx, column_idx in {0,1}.
 *  Assumes C accumulators already in TMM4-7 from previous zero2x2 or update2x2.
 *  Processes depth = 32 per call (1 tile pair along depth dimension).
 */
NK_INTERNAL void nk_dots_bf16bf16f32_update2x2_sapphire_amx(   //
    nk_dots_bf16bf16f32_a16x32_sapphire_amx_t const *a_tile_0, //
    nk_dots_bf16bf16f32_a16x32_sapphire_amx_t const *a_tile_1, //
    nk_dots_bf16bf16f32_b32x16_sapphire_amx_t const *b_tile_0, //
    nk_dots_bf16bf16f32_b32x16_sapphire_amx_t const *b_tile_1) {

    // Load A tiles (rows 0-15 and 16-31)
    _tile_loadd(0, a_tile_0->data, 64);
    _tile_loadd(1, a_tile_1->data, 64);

    // Load B tiles (columns 0-15 and 16-31)
    _tile_loadd(2, b_tile_0->data, 64);
    _tile_loadd(3, b_tile_1->data, 64);

    // Compute 2 × 2 outer product, accumulating into C registers
    _tile_dpbf16ps(4, 0, 2); // C[0,0] += A0 × B0
    _tile_dpbf16ps(5, 0, 3); // C[0,1] += A0 × B1
    _tile_dpbf16ps(6, 1, 2); // C[1,0] += A1 × B0
    _tile_dpbf16ps(7, 1, 3); // C[1,1] += A1 × B1
}

/*  Accumulate BF16 2 × 2 with DIRECT A load from source matrix.
 *  Use this for full interior tiles to avoid buffer copy overhead.
 *  a_row_ptr_0/a_row_ptr_1 point directly to source rows, a_stride is in bytes.
 */
NK_INTERNAL void nk_dots_bf16bf16f32_update2x2_direct_sapphire_amx(                 //
    nk_bf16_t const *a_row_ptr_0, nk_bf16_t const *a_row_ptr_1, nk_size_t a_stride, //
    nk_dots_bf16bf16f32_b32x16_sapphire_amx_t const *b_tile_0,                      //
    nk_dots_bf16bf16f32_b32x16_sapphire_amx_t const *b_tile_1) {

    // Load A tiles directly from source (no buffer copy!)
    _tile_loadd(0, a_row_ptr_0, a_stride);
    _tile_loadd(1, a_row_ptr_1, a_stride);

    // Load B tiles (already packed)
    _tile_loadd(2, b_tile_0->data, 64);
    _tile_loadd(3, b_tile_1->data, 64);

    // Compute 2 × 2 outer product
    _tile_dpbf16ps(4, 0, 2);
    _tile_dpbf16ps(5, 0, 3);
    _tile_dpbf16ps(6, 1, 2);
    _tile_dpbf16ps(7, 1, 3);
}

/*  Store BF16 2 × 2 C accumulators from TMM4-7 to state buffer (for masked output).
 */
NK_INTERNAL void nk_dots_bf16bf16f32_store2x2_sapphire_amx( //
    nk_dots_bf16bf16f32_state2x2_sapphire_amx_t *state) {
    _tile_stored(4, state->c[0][0].data, 64);
    _tile_stored(5, state->c[0][1].data, 64);
    _tile_stored(6, state->c[1][0].data, 64);
    _tile_stored(7, state->c[1][1].data, 64);
}

/*  Store BF16 2 × 2 C accumulators DIRECTLY to output matrix (no intermediate buffer).
 *  Use for full interior tiles. c_stride is in bytes.
 */
NK_INTERNAL void nk_dots_bf16bf16f32_store2x2_direct_sapphire_amx( //
    nk_f32_t *c, nk_size_t c_stride) {
    _tile_stored(4, c, c_stride);
    _tile_stored(5, c + 16, c_stride);
    _tile_stored(6, (nk_f32_t *)((char *)c + 16 * c_stride), c_stride);
    _tile_stored(7, (nk_f32_t *)((char *)c + 16 * c_stride) + 16, c_stride);
}

/*  Zero INT8 2 × 2 C accumulators in TMM4-7 (call once before depth-loop).
 */
NK_INTERNAL void nk_dots_i8i8i32_zero2x2_sapphire_amx(void) {
    _tile_zero(4);
    _tile_zero(5);
    _tile_zero(6);
    _tile_zero(7);
}

/*  Accumulate INT8 2 × 2 output: C[row_idx, column_idx] += A[row_idx] × B[column_idx] for row_idx, column_idx in {0,1}.
 *  Assumes C accumulators already in TMM4-7.
 *  Processes depth = 64 per call (1 tile pair along depth dimension).
 */
NK_INTERNAL void nk_dots_i8i8i32_update2x2_sapphire_amx(   //
    nk_dots_i8i8i32_a16x64_sapphire_amx_t const *a_tile_0, //
    nk_dots_i8i8i32_a16x64_sapphire_amx_t const *a_tile_1, //
    nk_dots_i8i8i32_b64x16_sapphire_amx_t const *b_tile_0, //
    nk_dots_i8i8i32_b64x16_sapphire_amx_t const *b_tile_1) {

    _tile_loadd(0, a_tile_0->data, 64);
    _tile_loadd(1, a_tile_1->data, 64);
    _tile_loadd(2, b_tile_0->data, 64);
    _tile_loadd(3, b_tile_1->data, 64);

    _tile_dpbssd(4, 0, 2);
    _tile_dpbssd(5, 0, 3);
    _tile_dpbssd(6, 1, 2);
    _tile_dpbssd(7, 1, 3);
}

/*  Accumulate INT8 2 × 2 with DIRECT A load from source matrix.
 *  Use this for full interior tiles to avoid buffer copy overhead.
 */
NK_INTERNAL void nk_dots_i8i8i32_update2x2_direct_sapphire_amx(                 //
    nk_i8_t const *a_row_ptr_0, nk_i8_t const *a_row_ptr_1, nk_size_t a_stride, //
    nk_dots_i8i8i32_b64x16_sapphire_amx_t const *b_tile_0,                      //
    nk_dots_i8i8i32_b64x16_sapphire_amx_t const *b_tile_1) {

    _tile_loadd(0, a_row_ptr_0, a_stride);
    _tile_loadd(1, a_row_ptr_1, a_stride);
    _tile_loadd(2, b_tile_0->data, 64);
    _tile_loadd(3, b_tile_1->data, 64);

    _tile_dpbssd(4, 0, 2);
    _tile_dpbssd(5, 0, 3);
    _tile_dpbssd(6, 1, 2);
    _tile_dpbssd(7, 1, 3);
}

/*  Store INT8 2 × 2 C accumulators from TMM4-7 to state buffer (for masked output).
 */
NK_INTERNAL void nk_dots_i8i8i32_store2x2_sapphire_amx( //
    nk_dots_i8i8i32_state2x2_sapphire_amx_t *state) {
    _tile_stored(4, state->c[0][0].data, 64);
    _tile_stored(5, state->c[0][1].data, 64);
    _tile_stored(6, state->c[1][0].data, 64);
    _tile_stored(7, state->c[1][1].data, 64);
}

/*  Store INT8 2 × 2 C accumulators DIRECTLY to output matrix (no intermediate buffer).
 *  Use for full interior tiles. c_stride is in bytes.
 */
NK_INTERNAL void nk_dots_i8i8i32_store2x2_direct_sapphire_amx( //
    nk_i32_t *c, nk_size_t c_stride) {
    _tile_stored(4, c, c_stride);
    _tile_stored(5, c + 16, c_stride);
    _tile_stored(6, (nk_i32_t *)((char *)c + 16 * c_stride), c_stride);
    _tile_stored(7, (nk_i32_t *)((char *)c + 16 * c_stride) + 16, c_stride);
}

/*  Store BF16 2 × 2 state to output matrix with masking for edge tiles.
 *  Handles any combination of valid_rows (0-32) and valid_cols (0-32).
 */
NK_INTERNAL void nk_dots_bf16bf16f32_output2x2_sapphire_amx(  //
    nk_dots_bf16bf16f32_state2x2_sapphire_amx_t const *state, //
    nk_f32_t *dst, nk_size_t dst_stride_elements,             //
    nk_size_t valid_rows, nk_size_t valid_cols) {

    // Rows 0-15
    nk_size_t const rows_upper = (valid_rows > 16) ? 16 : valid_rows;
    nk_size_t const cols_left = (valid_cols > 16) ? 16 : valid_cols;
    nk_size_t const cols_right = (valid_cols > 16) ? valid_cols - 16 : 0;

    if (rows_upper > 0 && cols_left > 0)
        nk_dots_bf16bf16f32_store_sapphire_amx(&state->c[0][0], dst, dst_stride_elements, rows_upper, cols_left);
    if (rows_upper > 0 && cols_right > 0)
        nk_dots_bf16bf16f32_store_sapphire_amx(&state->c[0][1], dst + 16, dst_stride_elements, rows_upper, cols_right);

    // Rows 16-31
    if (valid_rows > 16) {
        nk_size_t const rows_lower = valid_rows - 16;
        nk_f32_t *dst_lower = dst + 16 * dst_stride_elements;
        if (cols_left > 0)
            nk_dots_bf16bf16f32_store_sapphire_amx(&state->c[1][0], dst_lower, dst_stride_elements, rows_lower,
                                                   cols_left);
        if (cols_right > 0)
            nk_dots_bf16bf16f32_store_sapphire_amx(&state->c[1][1], dst_lower + 16, dst_stride_elements, rows_lower,
                                                   cols_right);
    }
}

/*  Store INT8 2 × 2 state to output matrix with masking for edge tiles.
 */
NK_INTERNAL void nk_dots_i8i8i32_output2x2_sapphire_amx(  //
    nk_dots_i8i8i32_state2x2_sapphire_amx_t const *state, //
    nk_i32_t *dst, nk_size_t dst_stride_elements,         //
    nk_size_t valid_rows, nk_size_t valid_cols) {

    nk_size_t const rows_upper = (valid_rows > 16) ? 16 : valid_rows;
    nk_size_t const cols_left = (valid_cols > 16) ? 16 : valid_cols;
    nk_size_t const cols_right = (valid_cols > 16) ? valid_cols - 16 : 0;

    if (rows_upper > 0 && cols_left > 0)
        nk_dots_i8i8i32_store_sapphire_amx(&state->c[0][0], dst, dst_stride_elements, rows_upper, cols_left);
    if (rows_upper > 0 && cols_right > 0)
        nk_dots_i8i8i32_store_sapphire_amx(&state->c[0][1], dst + 16, dst_stride_elements, rows_upper, cols_right);

    if (valid_rows > 16) {
        nk_size_t const rows_lower = valid_rows - 16;
        nk_i32_t *dst_lower = dst + 16 * dst_stride_elements;
        if (cols_left > 0)
            nk_dots_i8i8i32_store_sapphire_amx(&state->c[1][0], dst_lower, dst_stride_elements, rows_lower, cols_left);
        if (cols_right > 0)
            nk_dots_i8i8i32_store_sapphire_amx(&state->c[1][1], dst_lower + 16, dst_stride_elements, rows_lower,
                                               cols_right);
    }
}

/*  Pack A transposed into B format for BF16.
 *  Converts A[column][depth] -> B[depth][column] with pair-interleaving for TDPBF16PS.
 *  Used for cross-correlation: load vectors into A, pack to B, compute A × B.
 *
 *  Uses vectorized 16 × 32 transpose with YMM operations:
 *
 *  - Process 2 depth-columns at a time (one depth_group)
 *  - Each column has 16 elements (strided in A)
 *  - Interleave pairs and store contiguously to B
 */
NK_INTERNAL void nk_dots_bf16bf16f32_pack_transposed_sapphire_amx( //
    nk_dots_bf16bf16f32_a16x32_sapphire_amx_t const *a_tile,       //
    nk_dots_bf16bf16f32_b32x16_sapphire_amx_t *b_tile) {

    // Process 2 depth-columns at a time (forming one depth_group in output)
    for (nk_size_t depth_group_idx = 0; depth_group_idx < 16; depth_group_idx++) {
        nk_size_t depth_0 = depth_group_idx * 2;
        nk_size_t depth_1 = depth_group_idx * 2 + 1;

        // Load 2 columns from A (16 elements each) using gather-like pattern
        // A.data[column][depth] has stride 32 elements between rows
        // Use YMM for 16 × 16-bit = 256 bits

        // Extract columns depth_0 and depth_1 from all 16 rows
        __m256i col0, col1;
        {
            // Gather column depth_0: elements at A.data[0][depth_0], A.data[1][depth_0], ..., A.data[15][depth_0]
            // Load 8 rows at a time using 32-bit gather (gets 2 BF16 per element)
            __m256i idx_lo = _mm256_setr_epi32(0, 32, 64, 96, 128, 160, 192, 224);
            __m256i idx_hi = _mm256_setr_epi32(256, 288, 320, 352, 384, 416, 448, 480);

            // Gather 32-bit elements containing pairs of BF16
            nk_bf16_t const *base0 = &a_tile->data[0][depth_0];
            __m256i gather_lo = _mm256_i32gather_epi32((int const *)base0, idx_lo, 2);
            __m256i gather_hi = _mm256_i32gather_epi32((int const *)base0, idx_hi, 2);

            // Extract the low 16-bit from each 32-bit gather result
            // gather_lo has [a[0][depth_0:depth_0+1], a[1][depth_0:depth_0+1], ..., a[7][depth_0:depth_0+1]]
            // We want just the depth_0 values (low 16 bits of each 32-bit word)
            __m256i mask_lo16 = _mm256_set1_epi32(0x0000FFFF);
            __m256i col0_lo_32 = _mm256_and_si256(gather_lo, mask_lo16);
            __m256i col0_hi_32 = _mm256_and_si256(gather_hi, mask_lo16);

            // Pack 8 × 32-bit -> 8 × 16-bit for each half
            col0 = _mm256_packus_epi32(col0_lo_32, col0_hi_32);
            // packus interleaves lanes, need to permute
            col0 = _mm256_permute4x64_epi64(col0, 0xD8); // [0,2,1,3]

            // Extract depth_1 values (high 16 bits of each 32-bit word)
            __m256i col1_lo_32 = _mm256_srli_epi32(gather_lo, 16);
            __m256i col1_hi_32 = _mm256_srli_epi32(gather_hi, 16);
            col1 = _mm256_packus_epi32(col1_lo_32, col1_hi_32);
            col1 = _mm256_permute4x64_epi64(col1, 0xD8);
        }

        // Interleave col0 and col1 to get pair-interleaved output
        // Want: [col0[0], col1[0], col0[1], col1[1], ...]
        __m256i interleaved_lo = _mm256_unpacklo_epi16(col0, col1);
        __m256i interleaved_hi = _mm256_unpackhi_epi16(col0, col1);

        // Fix lane ordering after unpack (operates within 128-bit lanes)
        // interleaved_lo has [0-3, 8-11] in lanes, interleaved_hi has [4-7, 12-15]
        __m256i out0 = _mm256_permute2x128_si256(interleaved_lo, interleaved_hi, 0x20);
        __m256i out1 = _mm256_permute2x128_si256(interleaved_lo, interleaved_hi, 0x31);

        // Store to B.data[depth_group_idx][0..15][0..1]
        // B.data[depth_group_idx] is 16 × 2 = 32 BF16 = 64 bytes
        _mm256_store_si256((__m256i *)&b_tile->data[depth_group_idx][0][0], out0);
        _mm256_store_si256((__m256i *)&b_tile->data[depth_group_idx][8][0], out1);
    }
    nk_compiler_barrier_sapphire_amx_();
}

/*  Pack A transposed into B format for INT8.
 *  Converts A[column][depth] -> B[depth][column] with quad-interleaving for TDPBSSD.
 *
 *  Uses vectorized 16 × 64 transpose with ZMM operations:
 *
 *  - Process 4 depth-columns at a time (one depth_group)
 *  - Each column has 16 elements (strided in A)
 *  - Interleave quads and store contiguously to B
 */
NK_INTERNAL void nk_dots_i8i8i32_pack_transposed_sapphire_amx( //
    nk_dots_i8i8i32_a16x64_sapphire_amx_t const *a_tile,       //
    nk_dots_i8i8i32_b64x16_sapphire_amx_t *b_tile) {

    // Process 4 depth-columns at a time (forming one depth_group in output)
    for (nk_size_t depth_group_idx = 0; depth_group_idx < 16; depth_group_idx++) {
        nk_size_t depth_0 = depth_group_idx * 4;

        // For INT8, A.data[column][depth] has stride 64 bytes between rows
        // Load 4 consecutive columns from all 16 rows

        // Use XMM for column extraction (16 × 8-bit = 128 bits per column)
        __m128i col0, col1, col2, col3;

        // Gather columns depth_0..depth_3 using 32-bit gather (gets 4 I8 per element)
        {
            __m128i idx = _mm_setr_epi32(0, 64, 128, 192);
            nk_i8_t const *base = &a_tile->data[0][depth_0];

            // Gather 4 rows at a time
            __m128i g0 = _mm_i32gather_epi32((int const *)base, idx, 1);
            __m128i g1 = _mm_i32gather_epi32((int const *)(base + 4 * 64), idx, 1);
            __m128i g2 = _mm_i32gather_epi32((int const *)(base + 8 * 64), idx, 1);
            __m128i g3 = _mm_i32gather_epi32((int const *)(base + 12 * 64), idx, 1);

            // g0 contains [a[0][depth_0:depth_0+3], a[1][depth_0:depth_0+3], a[2][depth_0:depth_0+3],
            // a[3][depth_0:depth_0+3]] Each 32-bit element has 4 I8 values for columns depth_0, depth_1, depth_2,
            // depth_3

            // Extract column depth_0 (byte 0 from each 32-bit word)
            __m128i mask_byte0 = _mm_set1_epi32(0x000000FF);
            __m128i c0_0 = _mm_and_si128(g0, mask_byte0);
            __m128i c0_1 = _mm_and_si128(g1, mask_byte0);
            __m128i c0_2 = _mm_and_si128(g2, mask_byte0);
            __m128i c0_3 = _mm_and_si128(g3, mask_byte0);

            // Pack into bytes
            __m128i c0_01 = _mm_packus_epi32(c0_0, c0_1);
            __m128i c0_23 = _mm_packus_epi32(c0_2, c0_3);
            col0 = _mm_packus_epi16(c0_01, c0_23);

            // Extract columns depth_1, depth_2, depth_3 similarly
            __m128i c1_0 = _mm_and_si128(_mm_srli_epi32(g0, 8), mask_byte0);
            __m128i c1_1 = _mm_and_si128(_mm_srli_epi32(g1, 8), mask_byte0);
            __m128i c1_2 = _mm_and_si128(_mm_srli_epi32(g2, 8), mask_byte0);
            __m128i c1_3 = _mm_and_si128(_mm_srli_epi32(g3, 8), mask_byte0);
            __m128i c1_01 = _mm_packus_epi32(c1_0, c1_1);
            __m128i c1_23 = _mm_packus_epi32(c1_2, c1_3);
            col1 = _mm_packus_epi16(c1_01, c1_23);

            __m128i c2_0 = _mm_and_si128(_mm_srli_epi32(g0, 16), mask_byte0);
            __m128i c2_1 = _mm_and_si128(_mm_srli_epi32(g1, 16), mask_byte0);
            __m128i c2_2 = _mm_and_si128(_mm_srli_epi32(g2, 16), mask_byte0);
            __m128i c2_3 = _mm_and_si128(_mm_srli_epi32(g3, 16), mask_byte0);
            __m128i c2_01 = _mm_packus_epi32(c2_0, c2_1);
            __m128i c2_23 = _mm_packus_epi32(c2_2, c2_3);
            col2 = _mm_packus_epi16(c2_01, c2_23);

            __m128i c3_0 = _mm_srli_epi32(g0, 24);
            __m128i c3_1 = _mm_srli_epi32(g1, 24);
            __m128i c3_2 = _mm_srli_epi32(g2, 24);
            __m128i c3_3 = _mm_srli_epi32(g3, 24);
            __m128i c3_01 = _mm_packus_epi32(c3_0, c3_1);
            __m128i c3_23 = _mm_packus_epi32(c3_2, c3_3);
            col3 = _mm_packus_epi16(c3_01, c3_23);
        }

        // Quad-interleave: [col0[0], col1[0], col2[0], col3[0], col0[1], ...]
        // Stage 1: byte interleave pairs
        __m128i p01_lo = _mm_unpacklo_epi8(col0, col1);
        __m128i p01_hi = _mm_unpackhi_epi8(col0, col1);
        __m128i p23_lo = _mm_unpacklo_epi8(col2, col3);
        __m128i p23_hi = _mm_unpackhi_epi8(col2, col3);

        // Stage 2: word interleave to quads
        __m128i q0 = _mm_unpacklo_epi16(p01_lo, p23_lo); // rows 0-3
        __m128i q1 = _mm_unpackhi_epi16(p01_lo, p23_lo); // rows 4-7
        __m128i q2 = _mm_unpacklo_epi16(p01_hi, p23_hi); // rows 8-11
        __m128i q3 = _mm_unpackhi_epi16(p01_hi, p23_hi); // rows 12-15

        // Store to B.data[depth_group_idx][0..15][0..3]
        // B.data[depth_group_idx] is 16 × 4 = 64 I8 = 64 bytes
        _mm_store_si128((__m128i *)&b_tile->data[depth_group_idx][0][0], q0);
        _mm_store_si128((__m128i *)&b_tile->data[depth_group_idx][4][0], q1);
        _mm_store_si128((__m128i *)&b_tile->data[depth_group_idx][8][0], q2);
        _mm_store_si128((__m128i *)&b_tile->data[depth_group_idx][12][0], q3);
    }
    nk_compiler_barrier_sapphire_amx_();
}

/*  Compute self cross-correlation for up to 16 BF16 vectors.
 *  Result[row_idx, column_idx] = dot(vectors[row_idx], vectors[column_idx])
 *
 *  @param vectors       Row-major array of n_vectors, each of dimension depth
 *  @param n_vectors     Number of vectors (1-16, padded internally if < 16)
 *  @param depth         Vector dimension (padded to multiple of 96 internally)
 *  @param stride        Byte stride between vectors
 *  @param result        Output n_vectors × n_vectors matrix
 *  @param result_stride Byte stride between result rows
 */
NK_PUBLIC void nk_cross_bf16bf16f32_sapphire_amx(                   //
    nk_bf16_t const *vectors, nk_size_t n_vectors, nk_size_t depth, //
    nk_size_t stride, nk_f32_t *result, nk_size_t result_stride) {

    nk_size_t const stride_elements = stride / sizeof(nk_bf16_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);

    // Round depth up to multiple of 96 (3 tiles × 32 elements)
    nk_size_t const depth_tiles = (depth + 31) / 32;
    nk_size_t const depth_tile_groups = (depth_tiles + 2) / 3; // Groups of 3 tiles

    // Allocate tile buffers (3 A tiles + 3 B tiles per group)
    nk_dots_bf16bf16f32_a16x32_sapphire_amx_t a_tiles[3];
    nk_dots_bf16bf16f32_b32x16_sapphire_amx_t b_tiles[3];
    nk_dots_bf16bf16f32_state_sapphire_amx_t state;

    // Configure AMX tiles
    nk_amx_tile_configure_sapphire_amx_();

    // Initialize output state
    nk_dots_bf16bf16f32_init_sapphire_amx(&state);

    // Process depth dimension in groups of 96 (3 tiles)
    for (nk_size_t depth_group_idx = 0; depth_group_idx < depth_tile_groups; depth_group_idx++) {
        nk_size_t depth_base = depth_group_idx * 96;

        // Load 3 A tiles from vectors
        for (int tile_idx = 0; tile_idx < 3; tile_idx++) {
            nk_size_t depth_start = depth_base + tile_idx * 32;
            nk_size_t valid_cols = (depth_start + 32 <= depth) ? 32 : (depth > depth_start ? depth - depth_start : 0);
            nk_bf16_t const *src = vectors + depth_start;
            nk_dots_bf16bf16f32_load_a_sapphire_amx(&a_tiles[tile_idx], src, stride_elements, n_vectors, valid_cols);
        }

        // Pack A transposed into B tiles
        for (int tile_idx = 0; tile_idx < 3; tile_idx++) {
            nk_dots_bf16bf16f32_pack_transposed_sapphire_amx(&a_tiles[tile_idx], &b_tiles[tile_idx]);
        }

        // Accumulate: state += A × B
        nk_dots_bf16bf16f32_update_sapphire_amx(&state, &a_tiles[0], &a_tiles[1], &a_tiles[2], &b_tiles[0], &b_tiles[1],
                                                &b_tiles[2]);
    }

    // Store result
    nk_dots_bf16bf16f32_store_sapphire_amx(&state, result, result_stride_elements, n_vectors, n_vectors);
}

/*  Compute self cross-correlation for up to 16 INT8 vectors.
 *  Result[row_idx, column_idx] = dot(vectors[row_idx], vectors[column_idx])
 */
NK_PUBLIC void nk_cross_i8i8i32_sapphire_amx(                     //
    nk_i8_t const *vectors, nk_size_t n_vectors, nk_size_t depth, //
    nk_size_t stride, nk_i32_t *result, nk_size_t result_stride) {

    nk_size_t const result_stride_elements = result_stride / sizeof(nk_i32_t);

    // Round depth up to multiple of 192 (3 tiles × 64 elements)
    nk_size_t const depth_tiles = (depth + 63) / 64;
    nk_size_t const depth_tile_groups = (depth_tiles + 2) / 3;

    // Allocate tile buffers
    nk_dots_i8i8i32_a16x64_sapphire_amx_t a_tiles[3];
    nk_dots_i8i8i32_b64x16_sapphire_amx_t b_tiles[3];
    nk_dots_i8i8i32_state_sapphire_amx_t state;

    // Configure AMX tiles
    nk_amx_tile_configure_sapphire_amx_();

    // Initialize output state
    nk_dots_i8i8i32_init_sapphire_amx(&state);

    // Process depth dimension in groups of 192 (3 tiles)
    for (nk_size_t depth_group_idx = 0; depth_group_idx < depth_tile_groups; depth_group_idx++) {
        nk_size_t depth_base = depth_group_idx * 192;

        // Load 3 A tiles from vectors
        for (int tile_idx = 0; tile_idx < 3; tile_idx++) {
            nk_size_t depth_start = depth_base + tile_idx * 64;
            nk_size_t valid_cols = (depth_start + 64 <= depth) ? 64 : (depth > depth_start ? depth - depth_start : 0);
            nk_i8_t const *src = vectors + depth_start;
            nk_dots_i8i8i32_load_a_sapphire_amx(&a_tiles[tile_idx], src, stride, n_vectors, valid_cols);
        }

        // Pack A transposed into B tiles
        for (int tile_idx = 0; tile_idx < 3; tile_idx++) {
            nk_dots_i8i8i32_pack_transposed_sapphire_amx(&a_tiles[tile_idx], &b_tiles[tile_idx]);
        }

        // Accumulate: state += A × B
        nk_dots_i8i8i32_update_sapphire_amx(&state, &a_tiles[0], &a_tiles[1], &a_tiles[2], &b_tiles[0], &b_tiles[1],
                                            &b_tiles[2]);
    }

    // Store result
    nk_dots_i8i8i32_store_sapphire_amx(&state, result, result_stride_elements, n_vectors, n_vectors);
}

/*  BF16 packed buffer size: header + all tiles for full column rows + column edge.
 *  Hybrid layout:
 *
 *  - Tiles include depth remainder (zero-padded) for AMX to handle full dot products
 *  - Column edge (remaining rows) stored row-major for simple AVX-512 edge kernel
 */
NK_PUBLIC nk_size_t nk_dots_bf16bf16f32_packed_size_sapphire_amx(nk_size_t column_count, nk_size_t depth) {
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
NK_PUBLIC nk_size_t nk_dots_i8i8i32_packed_size_sapphire_amx(nk_size_t column_count, nk_size_t depth) {
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
NK_PUBLIC void nk_dots_bf16bf16f32_pack_sapphire_amx(            //
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
NK_PUBLIC void nk_dots_i8i8i32_pack_sapphire_amx(              //
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
NK_PUBLIC void nk_dots_bf16bf16f32_sapphire_amx(           //
    nk_bf16_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t row_count, nk_size_t column_count, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride) {

    // Parse packed B header
    nk_dots_amx_packed_header_t const *header = (nk_dots_amx_packed_header_t const *)b_packed;
    nk_size_t const num_column_tiles = header->full_column_tiles;
    nk_size_t const num_depth_tiles = header->full_depth_tiles;
    nk_size_t const column_edge_rows = header->column_edge_rows;

    // Packed B data regions
    nk_bf16_t const *b_tiles = (nk_bf16_t const *)((char const *)b_packed + sizeof(nk_dots_amx_packed_header_t));
    nk_bf16_t const *column_edge_ptr = (nk_bf16_t const *)((char const *)b_packed + header->column_edge_offset);

    // Stride conversions
    nk_size_t const a_stride_elements = a_stride / sizeof(nk_bf16_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);

    // Tile dimensions
    nk_size_t const tile_depth = 32; // depth elements per BF16 tile
    nk_size_t const tile_size = 512; // elements per packed tile
    nk_size_t const full_columns = num_column_tiles * 16;

    // Block counts (32 × 32 output blocks = 2 × 2 tiles)
    nk_size_t const num_row_blocks = (row_count + 31) / 32; // ceiling division
    nk_size_t const num_column_blocks = num_column_tiles / 2;

    if (num_depth_tiles == 0) return;

    // Tile buffers for A (only used for edge tiles)
    nk_dots_bf16bf16f32_a16x32_sapphire_amx_t a_tile_0, a_tile_1;
    nk_dots_bf16bf16f32_state2x2_sapphire_amx_t c_state;

    // Precompute: number of full depth-tiles (no masking needed)
    nk_size_t const num_full_depth_tiles = depth / tile_depth;
    nk_size_t const depth_remainder = depth % tile_depth;

    nk_amx_tile_configure_sapphire_amx_();

    // Process all 32 × 32 row × column blocks (including partial edge blocks)
    for (nk_size_t row_block_idx = 0; row_block_idx < num_row_blocks; row_block_idx++) {
        nk_size_t const row_block_start = row_block_idx * 32;
        nk_size_t const valid_row_count = (row_block_start + 32 <= row_count) ? 32 : (row_count - row_block_start);
        nk_size_t const is_full_row_block = (valid_row_count == 32);

        // Process full column-blocks (pairs of 16-column tiles = 32 columns)
        for (nk_size_t column_block_idx = 0; column_block_idx < num_column_blocks; column_block_idx++) {
            nk_size_t const column_block_start = column_block_idx * 32;

            // B tile base indices (linear layout: column_tile × num_depth_tiles + depth_tile)
            nk_size_t const b_column_0_base = (column_block_idx * 2) * num_depth_tiles;
            nk_size_t const b_column_1_base = (column_block_idx * 2 + 1) * num_depth_tiles;

            // Zero accumulators in registers
            nk_dots_bf16bf16f32_zero2x2_sapphire_amx();

            // Fast path: full row-block with full depth-tiles → direct A load
            if (is_full_row_block) {
                // Process full depth-tiles with direct load
                for (nk_size_t depth_tile_idx = 0; depth_tile_idx < num_full_depth_tiles; depth_tile_idx++) {
                    nk_size_t const depth_offset = depth_tile_idx * tile_depth;

                    nk_dots_bf16bf16f32_b32x16_sapphire_amx_t const *b_tile_0 =
                        (nk_dots_bf16bf16f32_b32x16_sapphire_amx_t const *)(b_tiles +
                                                                            (b_column_0_base + depth_tile_idx) *
                                                                                tile_size);
                    nk_dots_bf16bf16f32_b32x16_sapphire_amx_t const *b_tile_1 =
                        (nk_dots_bf16bf16f32_b32x16_sapphire_amx_t const *)(b_tiles +
                                                                            (b_column_1_base + depth_tile_idx) *
                                                                                tile_size);

                    // Direct load from source, no buffer copy
                    nk_dots_bf16bf16f32_update2x2_direct_sapphire_amx(
                        a + row_block_start * a_stride_elements + depth_offset,
                        a + (row_block_start + 16) * a_stride_elements + depth_offset, a_stride, b_tile_0, b_tile_1);
                }

                // Handle partial depth-tile (if any) with buffered load
                if (depth_remainder > 0) {
                    nk_size_t const depth_tile_idx = num_full_depth_tiles;
                    nk_size_t const depth_offset = depth_tile_idx * tile_depth;

                    nk_dots_bf16bf16f32_load_a_sapphire_amx(&a_tile_0,
                                                            a + row_block_start * a_stride_elements + depth_offset,
                                                            a_stride_elements, 16, depth_remainder);
                    nk_dots_bf16bf16f32_load_a_sapphire_amx(
                        &a_tile_1, a + (row_block_start + 16) * a_stride_elements + depth_offset, a_stride_elements, 16,
                        depth_remainder);

                    nk_dots_bf16bf16f32_b32x16_sapphire_amx_t const *b_tile_0 =
                        (nk_dots_bf16bf16f32_b32x16_sapphire_amx_t const *)(b_tiles +
                                                                            (b_column_0_base + depth_tile_idx) *
                                                                                tile_size);
                    nk_dots_bf16bf16f32_b32x16_sapphire_amx_t const *b_tile_1 =
                        (nk_dots_bf16bf16f32_b32x16_sapphire_amx_t const *)(b_tiles +
                                                                            (b_column_1_base + depth_tile_idx) *
                                                                                tile_size);

                    nk_dots_bf16bf16f32_update2x2_sapphire_amx(&a_tile_0, &a_tile_1, b_tile_0, b_tile_1);
                }
            }
            // Slow path: edge row-block → always use buffered load with masking
            else {
                nk_size_t const rows_in_tile_0 = (valid_row_count > 16) ? 16 : valid_row_count;
                nk_size_t const rows_in_tile_1 = (valid_row_count > 16) ? valid_row_count - 16 : 0;

                for (nk_size_t depth_tile_idx = 0; depth_tile_idx < num_depth_tiles; depth_tile_idx++) {
                    nk_size_t const depth_offset = depth_tile_idx * tile_depth;
                    nk_size_t const valid_depth = (depth_tile_idx < num_full_depth_tiles) ? tile_depth
                                                                                          : depth_remainder;

                    nk_dots_bf16bf16f32_load_a_sapphire_amx(&a_tile_0,
                                                            a + row_block_start * a_stride_elements + depth_offset,
                                                            a_stride_elements, rows_in_tile_0, valid_depth);
                    if (rows_in_tile_1 > 0) {
                        nk_dots_bf16bf16f32_load_a_sapphire_amx(
                            &a_tile_1, a + (row_block_start + 16) * a_stride_elements + depth_offset, a_stride_elements,
                            rows_in_tile_1, valid_depth);
                    }

                    nk_dots_bf16bf16f32_b32x16_sapphire_amx_t const *b_tile_0 =
                        (nk_dots_bf16bf16f32_b32x16_sapphire_amx_t const *)(b_tiles +
                                                                            (b_column_0_base + depth_tile_idx) *
                                                                                tile_size);
                    nk_dots_bf16bf16f32_b32x16_sapphire_amx_t const *b_tile_1 =
                        (nk_dots_bf16bf16f32_b32x16_sapphire_amx_t const *)(b_tiles +
                                                                            (b_column_1_base + depth_tile_idx) *
                                                                                tile_size);

                    nk_dots_bf16bf16f32_update2x2_sapphire_amx(&a_tile_0, &a_tile_1, b_tile_0, b_tile_1);
                }
            }

            // Store accumulators to output
            if (is_full_row_block) {
                // Fast path: direct store to C, no intermediate buffer
                nk_dots_bf16bf16f32_store2x2_direct_sapphire_amx(
                    c + row_block_start * c_stride_elements + column_block_start, c_stride);
            }
            else {
                // Slow path: edge row-block needs masked output
                nk_dots_bf16bf16f32_store2x2_sapphire_amx(&c_state);
                nk_dots_bf16bf16f32_output2x2_sapphire_amx(&c_state,
                                                           c + row_block_start * c_stride_elements + column_block_start,
                                                           c_stride_elements, valid_row_count, 32);
            }
        }

        // Handle odd column-tile (single 16-column tile if num_column_tiles is odd)
        if (num_column_tiles % 2 == 1) {
            nk_size_t const column_tile_idx = num_column_tiles - 1;
            nk_size_t const column_start = column_tile_idx * 16;
            nk_size_t const b_column_base = column_tile_idx * num_depth_tiles;
            nk_size_t const rows_in_tile_0 = (valid_row_count > 16) ? 16 : valid_row_count;
            nk_size_t const rows_in_tile_1 = (valid_row_count > 16) ? valid_row_count - 16 : 0;

            // Use 1 × 1 blocking for single column-tile
            nk_dots_bf16bf16f32_state_sapphire_amx_t c0_state, c1_state;
            nk_dots_bf16bf16f32_init_sapphire_amx(&c0_state);
            nk_dots_bf16bf16f32_init_sapphire_amx(&c1_state);

            _tile_zero(4);
            _tile_zero(6);

            for (nk_size_t depth_tile_idx = 0; depth_tile_idx < num_depth_tiles; depth_tile_idx++) {
                nk_size_t const depth_offset = depth_tile_idx * tile_depth;
                nk_size_t const valid_depth = (depth_tile_idx < num_full_depth_tiles) ? tile_depth : depth_remainder;

                nk_dots_bf16bf16f32_load_a_sapphire_amx(&a_tile_0,
                                                        a + row_block_start * a_stride_elements + depth_offset,
                                                        a_stride_elements, rows_in_tile_0, valid_depth);
                if (rows_in_tile_1 > 0) {
                    nk_dots_bf16bf16f32_load_a_sapphire_amx(
                        &a_tile_1, a + (row_block_start + 16) * a_stride_elements + depth_offset, a_stride_elements,
                        rows_in_tile_1, valid_depth);
                }

                nk_dots_bf16bf16f32_b32x16_sapphire_amx_t const *b_tile_0 =
                    (nk_dots_bf16bf16f32_b32x16_sapphire_amx_t const *)(b_tiles +
                                                                        (b_column_base + depth_tile_idx) * tile_size);

                _tile_loadd(0, a_tile_0.data, 64);
                _tile_loadd(1, a_tile_1.data, 64);
                _tile_loadd(2, b_tile_0->data, 64);

                _tile_dpbf16ps(4, 0, 2);
                _tile_dpbf16ps(6, 1, 2);
            }

            _tile_stored(4, c0_state.data, 64);
            _tile_stored(6, c1_state.data, 64);

            nk_dots_bf16bf16f32_store_sapphire_amx(&c0_state, c + row_block_start * c_stride_elements + column_start,
                                                   c_stride_elements, rows_in_tile_0, 16);
            if (rows_in_tile_1 > 0) {
                nk_dots_bf16bf16f32_store_sapphire_amx(&c1_state,
                                                       c + (row_block_start + 16) * c_stride_elements + column_start,
                                                       c_stride_elements, rows_in_tile_1, 16);
            }
        }

        // Handle column-edge (remaining columns < 16) using AMX with partial tiles
        if (column_edge_rows > 0) {
            nk_size_t const rows_in_tile_0 = (valid_row_count > 16) ? 16 : valid_row_count;
            nk_size_t const rows_in_tile_1 = (valid_row_count > 16) ? valid_row_count - 16 : 0;

            nk_dots_bf16bf16f32_state_sapphire_amx_t c0_state, c1_state;
            nk_dots_bf16bf16f32_a16x32_sapphire_amx_t b_as_a;
            nk_dots_bf16bf16f32_b32x16_sapphire_amx_t b_tile;

            _tile_zero(4);
            _tile_zero(6);

            for (nk_size_t depth_tile_idx = 0; depth_tile_idx < num_depth_tiles; depth_tile_idx++) {
                nk_size_t const depth_offset = depth_tile_idx * tile_depth;
                nk_size_t const valid_depth = (depth_tile_idx < num_full_depth_tiles) ? tile_depth : depth_remainder;

                // Load A tiles
                nk_dots_bf16bf16f32_load_a_sapphire_amx(&a_tile_0,
                                                        a + row_block_start * a_stride_elements + depth_offset,
                                                        a_stride_elements, rows_in_tile_0, valid_depth);
                if (rows_in_tile_1 > 0) {
                    nk_dots_bf16bf16f32_load_a_sapphire_amx(
                        &a_tile_1, a + (row_block_start + 16) * a_stride_elements + depth_offset, a_stride_elements,
                        rows_in_tile_1, valid_depth);
                }

                // Load B edge data (row-major: b_edge[row * depth + col]) and pack into B tile
                // Each "row" in edge data corresponds to one output column
                nk_dots_bf16bf16f32_load_a_sapphire_amx(&b_as_a, column_edge_ptr + depth_offset, depth,
                                                        column_edge_rows, valid_depth);
                nk_dots_bf16bf16f32_pack_transposed_sapphire_amx(&b_as_a, &b_tile);

                _tile_loadd(0, a_tile_0.data, 64);
                _tile_loadd(1, a_tile_1.data, 64);
                _tile_loadd(2, b_tile.data, 64);

                _tile_dpbf16ps(4, 0, 2);
                _tile_dpbf16ps(6, 1, 2);
            }

            _tile_stored(4, c0_state.data, 64);
            _tile_stored(6, c1_state.data, 64);

            nk_dots_bf16bf16f32_store_sapphire_amx(&c0_state, c + row_block_start * c_stride_elements + full_columns,
                                                   c_stride_elements, rows_in_tile_0, column_edge_rows);
            if (rows_in_tile_1 > 0) {
                nk_dots_bf16bf16f32_store_sapphire_amx(&c1_state,
                                                       c + (row_block_start + 16) * c_stride_elements + full_columns,
                                                       c_stride_elements, rows_in_tile_1, column_edge_rows);
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
NK_PUBLIC void nk_dots_bf16bf16bf16_sapphire_amx( //
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
NK_PUBLIC void nk_dots_i8i8i32_sapphire_amx(             //
    nk_i8_t const *a, void const *b_packed, nk_i32_t *c, //
    nk_size_t row_count, nk_size_t column_count, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride) {

    // Parse packed B header
    nk_dots_amx_packed_header_t const *header = (nk_dots_amx_packed_header_t const *)b_packed;
    nk_size_t const num_column_tiles = header->full_column_tiles;
    nk_size_t const num_depth_tiles = header->full_depth_tiles;
    nk_size_t const column_edge_rows = header->column_edge_rows;

    // Packed B data regions
    nk_i8_t const *b_tiles = (nk_i8_t const *)((char const *)b_packed + sizeof(nk_dots_amx_packed_header_t));
    nk_i8_t const *column_edge_ptr = (nk_i8_t const *)((char const *)b_packed + header->column_edge_offset);

    // Stride conversions
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_i32_t);

    // Tile dimensions
    nk_size_t const tile_depth = 64;  // depth elements per INT8 tile
    nk_size_t const tile_size = 1024; // bytes per packed tile
    nk_size_t const full_columns = num_column_tiles * 16;

    // Block counts (32 × 32 output blocks = 2 × 2 tiles)
    nk_size_t const num_row_blocks = (row_count + 31) / 32; // ceiling division
    nk_size_t const num_column_blocks = num_column_tiles / 2;

    if (num_depth_tiles == 0) return;

    // Tile buffers for A (only used for edge tiles)
    nk_dots_i8i8i32_a16x64_sapphire_amx_t a_tile_0, a_tile_1;
    nk_dots_i8i8i32_state2x2_sapphire_amx_t c_state;

    // Precompute: number of full depth-tiles (no masking needed)
    nk_size_t const num_full_depth_tiles = depth / tile_depth;
    nk_size_t const depth_remainder = depth % tile_depth;

    nk_amx_tile_configure_sapphire_amx_();

    // Process all 32 × 32 row × column blocks (including partial edge blocks)
    for (nk_size_t row_block_idx = 0; row_block_idx < num_row_blocks; row_block_idx++) {
        nk_size_t const row_block_start = row_block_idx * 32;
        nk_size_t const valid_row_count = (row_block_start + 32 <= row_count) ? 32 : (row_count - row_block_start);
        nk_size_t const is_full_row_block = (valid_row_count == 32);

        // Process full column-blocks (pairs of 16-column tiles = 32 columns)
        for (nk_size_t column_block_idx = 0; column_block_idx < num_column_blocks; column_block_idx++) {
            nk_size_t const column_block_start = column_block_idx * 32;

            // B tile base indices (linear layout: column_tile × num_depth_tiles + depth_tile)
            nk_size_t const b_column_0_base = (column_block_idx * 2) * num_depth_tiles;
            nk_size_t const b_column_1_base = (column_block_idx * 2 + 1) * num_depth_tiles;

            // Zero accumulators in registers
            nk_dots_i8i8i32_zero2x2_sapphire_amx();

            // Fast path: full row-block with full depth-tiles → direct A load
            if (is_full_row_block) {
                // Process full depth-tiles with direct load
                for (nk_size_t depth_tile_idx = 0; depth_tile_idx < num_full_depth_tiles; depth_tile_idx++) {
                    nk_size_t const depth_offset = depth_tile_idx * tile_depth;

                    nk_dots_i8i8i32_b64x16_sapphire_amx_t const *b_tile_0 =
                        (nk_dots_i8i8i32_b64x16_sapphire_amx_t const *)(b_tiles +
                                                                        (b_column_0_base + depth_tile_idx) * tile_size);
                    nk_dots_i8i8i32_b64x16_sapphire_amx_t const *b_tile_1 =
                        (nk_dots_i8i8i32_b64x16_sapphire_amx_t const *)(b_tiles +
                                                                        (b_column_1_base + depth_tile_idx) * tile_size);

                    // Direct load from source, no buffer copy
                    nk_dots_i8i8i32_update2x2_direct_sapphire_amx(a + row_block_start * a_stride + depth_offset,
                                                                  a + (row_block_start + 16) * a_stride + depth_offset,
                                                                  a_stride, b_tile_0, b_tile_1);
                }

                // Handle partial depth-tile (if any) with buffered load
                if (depth_remainder > 0) {
                    nk_size_t const depth_tile_idx = num_full_depth_tiles;
                    nk_size_t const depth_offset = depth_tile_idx * tile_depth;

                    nk_dots_i8i8i32_load_a_sapphire_amx(&a_tile_0, a + row_block_start * a_stride + depth_offset,
                                                        a_stride, 16, depth_remainder);
                    nk_dots_i8i8i32_load_a_sapphire_amx(&a_tile_1, a + (row_block_start + 16) * a_stride + depth_offset,
                                                        a_stride, 16, depth_remainder);

                    nk_dots_i8i8i32_b64x16_sapphire_amx_t const *b_tile_0 =
                        (nk_dots_i8i8i32_b64x16_sapphire_amx_t const *)(b_tiles +
                                                                        (b_column_0_base + depth_tile_idx) * tile_size);
                    nk_dots_i8i8i32_b64x16_sapphire_amx_t const *b_tile_1 =
                        (nk_dots_i8i8i32_b64x16_sapphire_amx_t const *)(b_tiles +
                                                                        (b_column_1_base + depth_tile_idx) * tile_size);

                    nk_dots_i8i8i32_update2x2_sapphire_amx(&a_tile_0, &a_tile_1, b_tile_0, b_tile_1);
                }
            }
            // Slow path: edge row-block → always use buffered load with masking
            else {
                nk_size_t const rows_in_tile_0 = (valid_row_count > 16) ? 16 : valid_row_count;
                nk_size_t const rows_in_tile_1 = (valid_row_count > 16) ? valid_row_count - 16 : 0;

                for (nk_size_t depth_tile_idx = 0; depth_tile_idx < num_depth_tiles; depth_tile_idx++) {
                    nk_size_t const depth_offset = depth_tile_idx * tile_depth;
                    nk_size_t const valid_depth = (depth_tile_idx < num_full_depth_tiles) ? tile_depth
                                                                                          : depth_remainder;

                    nk_dots_i8i8i32_load_a_sapphire_amx(&a_tile_0, a + row_block_start * a_stride + depth_offset,
                                                        a_stride, rows_in_tile_0, valid_depth);
                    if (rows_in_tile_1 > 0) {
                        nk_dots_i8i8i32_load_a_sapphire_amx(&a_tile_1,
                                                            a + (row_block_start + 16) * a_stride + depth_offset,
                                                            a_stride, rows_in_tile_1, valid_depth);
                    }

                    nk_dots_i8i8i32_b64x16_sapphire_amx_t const *b_tile_0 =
                        (nk_dots_i8i8i32_b64x16_sapphire_amx_t const *)(b_tiles +
                                                                        (b_column_0_base + depth_tile_idx) * tile_size);
                    nk_dots_i8i8i32_b64x16_sapphire_amx_t const *b_tile_1 =
                        (nk_dots_i8i8i32_b64x16_sapphire_amx_t const *)(b_tiles +
                                                                        (b_column_1_base + depth_tile_idx) * tile_size);

                    nk_dots_i8i8i32_update2x2_sapphire_amx(&a_tile_0, &a_tile_1, b_tile_0, b_tile_1);
                }
            }

            // Store accumulators to output
            if (is_full_row_block) {
                // Fast path: direct store to C, no intermediate buffer
                nk_dots_i8i8i32_store2x2_direct_sapphire_amx(
                    c + row_block_start * c_stride_elements + column_block_start, c_stride);
            }
            else {
                // Slow path: edge row-block needs masked output
                nk_dots_i8i8i32_store2x2_sapphire_amx(&c_state);
                nk_dots_i8i8i32_output2x2_sapphire_amx(&c_state,
                                                       c + row_block_start * c_stride_elements + column_block_start,
                                                       c_stride_elements, valid_row_count, 32);
            }
        }

        // Handle odd column-tile (single 16-column tile if num_column_tiles is odd)
        if (num_column_tiles % 2 == 1) {
            nk_size_t const column_tile_idx = num_column_tiles - 1;
            nk_size_t const column_start = column_tile_idx * 16;
            nk_size_t const b_column_base = column_tile_idx * num_depth_tiles;
            nk_size_t const rows_in_tile_0 = (valid_row_count > 16) ? 16 : valid_row_count;
            nk_size_t const rows_in_tile_1 = (valid_row_count > 16) ? valid_row_count - 16 : 0;

            // Use 1 × 1 blocking for single column-tile
            nk_dots_i8i8i32_state_sapphire_amx_t c0_state, c1_state;
            nk_dots_i8i8i32_init_sapphire_amx(&c0_state);
            nk_dots_i8i8i32_init_sapphire_amx(&c1_state);

            _tile_zero(4);
            _tile_zero(6);

            for (nk_size_t depth_tile_idx = 0; depth_tile_idx < num_depth_tiles; depth_tile_idx++) {
                nk_size_t const depth_offset = depth_tile_idx * tile_depth;
                nk_size_t const valid_depth = (depth_tile_idx < num_full_depth_tiles) ? tile_depth : depth_remainder;

                nk_dots_i8i8i32_load_a_sapphire_amx(&a_tile_0, a + row_block_start * a_stride + depth_offset, a_stride,
                                                    rows_in_tile_0, valid_depth);
                if (rows_in_tile_1 > 0) {
                    nk_dots_i8i8i32_load_a_sapphire_amx(&a_tile_1, a + (row_block_start + 16) * a_stride + depth_offset,
                                                        a_stride, rows_in_tile_1, valid_depth);
                }

                nk_dots_i8i8i32_b64x16_sapphire_amx_t const *b_tile_0 =
                    (nk_dots_i8i8i32_b64x16_sapphire_amx_t const *)(b_tiles +
                                                                    (b_column_base + depth_tile_idx) * tile_size);

                _tile_loadd(0, a_tile_0.data, 64);
                _tile_loadd(1, a_tile_1.data, 64);
                _tile_loadd(2, b_tile_0->data, 64);

                _tile_dpbssd(4, 0, 2);
                _tile_dpbssd(6, 1, 2);
            }

            _tile_stored(4, c0_state.data, 64);
            _tile_stored(6, c1_state.data, 64);

            nk_dots_i8i8i32_store_sapphire_amx(&c0_state, c + row_block_start * c_stride_elements + column_start,
                                               c_stride_elements, rows_in_tile_0, 16);
            if (rows_in_tile_1 > 0) {
                nk_dots_i8i8i32_store_sapphire_amx(&c1_state,
                                                   c + (row_block_start + 16) * c_stride_elements + column_start,
                                                   c_stride_elements, rows_in_tile_1, 16);
            }
        }

        // Handle column-edge (remaining columns < 16) using AMX with partial tiles
        if (column_edge_rows > 0) {
            nk_size_t const rows_in_tile_0 = (valid_row_count > 16) ? 16 : valid_row_count;
            nk_size_t const rows_in_tile_1 = (valid_row_count > 16) ? valid_row_count - 16 : 0;

            nk_dots_i8i8i32_state_sapphire_amx_t c0_state, c1_state;
            nk_dots_i8i8i32_a16x64_sapphire_amx_t b_as_a;
            nk_dots_i8i8i32_b64x16_sapphire_amx_t b_tile;

            _tile_zero(4);
            _tile_zero(6);

            for (nk_size_t depth_tile_idx = 0; depth_tile_idx < num_depth_tiles; depth_tile_idx++) {
                nk_size_t const depth_offset = depth_tile_idx * tile_depth;
                nk_size_t const valid_depth = (depth_tile_idx < num_full_depth_tiles) ? tile_depth : depth_remainder;

                // Load A tiles
                nk_dots_i8i8i32_load_a_sapphire_amx(&a_tile_0, a + row_block_start * a_stride + depth_offset, a_stride,
                                                    rows_in_tile_0, valid_depth);
                if (rows_in_tile_1 > 0) {
                    nk_dots_i8i8i32_load_a_sapphire_amx(&a_tile_1, a + (row_block_start + 16) * a_stride + depth_offset,
                                                        a_stride, rows_in_tile_1, valid_depth);
                }

                // Load B edge data (row-major: b_edge[row * depth + col]) and pack into B tile
                // Each "row" in edge data corresponds to one output column
                nk_dots_i8i8i32_load_a_sapphire_amx(&b_as_a, column_edge_ptr + depth_offset, depth, column_edge_rows,
                                                    valid_depth);
                nk_dots_i8i8i32_pack_transposed_sapphire_amx(&b_as_a, &b_tile);

                _tile_loadd(0, a_tile_0.data, 64);
                _tile_loadd(1, a_tile_1.data, 64);
                _tile_loadd(2, b_tile.data, 64);

                _tile_dpbssd(4, 0, 2);
                _tile_dpbssd(6, 1, 2);
            }

            _tile_stored(4, c0_state.data, 64);
            _tile_stored(6, c1_state.data, 64);

            nk_dots_i8i8i32_store_sapphire_amx(&c0_state, c + row_block_start * c_stride_elements + full_columns,
                                               c_stride_elements, rows_in_tile_0, column_edge_rows);
            if (rows_in_tile_1 > 0) {
                nk_dots_i8i8i32_store_sapphire_amx(&c1_state,
                                                   c + (row_block_start + 16) * c_stride_elements + full_columns,
                                                   c_stride_elements, rows_in_tile_1, column_edge_rows);
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
NK_PUBLIC void nk_dots_i8i8i8_sapphire_amx( //
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

#pragma clang attribute pop
#pragma GCC pop_options
#endif // NK_TARGET_SAPPHIRE_AMX
#endif // NK_TARGET_X86_

#endif // NK_DOTS_SAPPHIRE_AMX_H

/**
 *  @brief SIMD-accelerated Dot Products for Real and Complex Numbers optimized for Intel Sapphire Rapids CPUs.
 *  @file include/numkong/dots/sapphire.h
 *  @sa include/numkong/dots.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 *
 *  AMX tile dimensions:
 *
 *    BF16 tiles: 16 rows × 32 elements = 512 BF16 values = 1KB per tile
 *    INT8 tiles: 16 rows × 64 elements = 1024 INT8 values = 1KB per tile
 *
 *  Output pattern: 2×2 tile layout produces 32×32 output blocks.
 *  Register allocation:
 *    TMM0, TMM1: A matrix tiles (row blocks i and i+16)
 *    TMM2, TMM3: B matrix tiles (column blocks j and j+16)
 *    TMM4-7: C accumulator tiles (2×2 output grid)
 *
 *  Performance characteristics (single-threaded):
 *    - BF16 peak: ~500 GFLOPS per core (2× FP32 throughput)
 *    - INT8 peak: ~1000 GOPS per core (4× FP32 throughput)
 *    - Memory bandwidth: ~80 GB/s DDR5 per core
 *    - Optimal K dimension: multiples of 32 (BF16) or 64 (INT8)
 *
 *  Acceleration opportunities:
 *    - Pre-pack B matrix once for repeated inference (avoids runtime reordering)
 *    - Morton Z-curve tile ordering improves L2 cache hit rate by 5-25%
 *    - Partition A rows across threads for parallel execution
 *    - Use streaming stores for large C matrices to avoid cache pollution
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
    nk_u32_t full_n_tiles;  // Number of full N tiles (16 rows each)
    nk_u32_t full_k_tiles;  // Number of K tiles (32 cols for BF16, 64 for I8)
    nk_u32_t n_edge_rows;   // Remaining rows after full tiles (0-15)
    nk_u32_t n_edge_offset; // Byte offset to edge data region
    nk_u32_t reserved[12];  // Padding to 64 bytes
} nk_dots_amx_packed_header_t;

/*  Composable tile structures for AMX operations.
 *  These enable reusable primitives and cross-correlation (A×Aᵀ) use cases.
 */

/*  BF16 A tile: 16 rows × 32 K-elements, row-major layout.
 *  Loaded from source matrix, used as left operand in AMX multiply.
 */
typedef struct {
    NK_ALIGN64 nk_bf16_t data[16][32]; // 16 rows × 32 cols = 1KB
} nk_dots_bf16bf16f32_a16x32_sapphire_amx_t;

/*  BF16 B tile: 32 K × 16 N, pair-interleaved for TDPBF16PS.
 *  Access pattern: data[k/2][n][k%2] for logical B[k,n].
 *  Pre-packed from column-major or transposed source.
 */
typedef struct {
    NK_ALIGN64 nk_bf16_t data[16][16][2]; // 16 K-groups × 16 N × 2 = 1KB
} nk_dots_bf16bf16f32_b32x16_sapphire_amx_t;

/*  BF16 output state: 16×16 F32 accumulator tile.
 *  Holds partial sums during K-dimension accumulation.
 */
typedef struct {
    NK_ALIGN64 nk_f32_t data[16][16]; // 16×16 = 1KB
} nk_dots_bf16bf16f32_state_sapphire_amx_t;

/*  INT8 A tile: 16 rows × 64 K-elements, row-major layout.
 */
typedef struct {
    NK_ALIGN64 nk_i8_t data[16][64]; // 16 rows × 64 cols = 1KB
} nk_dots_i8i8i32_a16x64_sapphire_amx_t;

/*  INT8 B tile: 64 K × 16 N, quad-interleaved for TDPBSSD.
 *  Access pattern: data[k/4][n][k%4] for logical B[k,n].
 */
typedef struct {
    NK_ALIGN64 nk_i8_t data[16][16][4]; // 16 K-groups × 16 N × 4 = 1KB
} nk_dots_i8i8i32_b64x16_sapphire_amx_t;

/*  INT8 output state: 16×16 I32 accumulator tile.
 */
typedef struct {
    NK_ALIGN64 nk_i32_t data[16][16]; // 16×16 = 1KB
} nk_dots_i8i8i32_state_sapphire_amx_t;

/*  BF16 2×2 output state: 32×32 F32 output (4 accumulator tiles).
 *  Used for GEMM's 2×2 output blocking pattern.
 */
typedef struct {
    nk_dots_bf16bf16f32_state_sapphire_amx_t c[2][2]; // 4KB total
} nk_dots_bf16bf16f32_state2x2_sapphire_amx_t;

/*  INT8 2×2 output state: 32×32 I32 output (4 accumulator tiles).
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

    for (int tile_id = 0; tile_id < 8; tile_id++) {
        rows_per_tile[tile_id] = 16; // 16 rows per tile
        bytes_per_row[tile_id] = 64; // 64 bytes per row (1KB total)
    }
    _tile_loadconfig(tile_config);
}

/*  Compiler memory barrier to ensure stores complete before AMX tile loads.
 *  AMX _tile_loadd reads from memory written by AVX-512 stores. Without this barrier,
 *  the compiler may reorder or optimize away the stores, causing _tile_loadd to read stale data.
 *  This is a compiler-only fence (no CPU fence needed - same core, same memory).
 */
NK_INTERNAL void nk_compiler_barrier_sapphire_amx_(void) { __asm__ volatile("" ::: "memory"); }

/* ============================================================================
 *  Composable Tile Primitives - BF16
 * ============================================================================ */

/*  Initialize BF16 output state to zero.
 */
NK_INTERNAL void nk_dots_bf16bf16f32_init_sapphire_amx(nk_dots_bf16bf16f32_state_sapphire_amx_t *state) {
    __m512 zero = _mm512_setzero_ps();
    for (nk_size_t r = 0; r < 16; r++) { _mm512_store_ps(state->data[r], zero); }
}

/*  Load A tile from row-major source with masking for edge tiles.
 */
NK_INTERNAL void nk_dots_bf16bf16f32_load_a_sapphire_amx( //
    nk_dots_bf16bf16f32_a16x32_sapphire_amx_t *a_tile,    //
    nk_bf16_t const *src, nk_size_t src_stride_elements,  //
    nk_size_t valid_rows, nk_size_t valid_cols) {

    __mmask32 col_mask = (valid_cols >= 32) ? 0xFFFFFFFF : ((__mmask32)1 << valid_cols) - 1;
    __m512i zero = _mm512_setzero_si512();

    for (nk_size_t r = 0; r < 16; r++) {
        if (r < valid_rows) {
            __m512i row = _mm512_maskz_loadu_epi16(col_mask, src + r * src_stride_elements);
            _mm512_store_si512((__m512i *)a_tile->data[r], row);
        }
        else { _mm512_store_si512((__m512i *)a_tile->data[r], zero); }
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

    for (nk_size_t r = 0; r < valid_rows; r++) {
        __m512 row = _mm512_load_ps(state->data[r]);
        _mm512_mask_storeu_ps(dst + r * dst_stride_elements, col_mask, row);
    }
}

/*  Accumulate 3 A×B tile pairs into state using AMX TDPBF16PS.
 *  Processes K=96 per call (3 × 32 BF16 elements).
 *  For indivisible K, pad unused tiles with zeros.
 *
 *  Register allocation (uses 7 of 8 TMM registers):
 *    TMM0: accumulator (C)
 *    TMM1-3: A tiles (a0, a1, a2)
 *    TMM4-6: B tiles (b0, b1, b2)
 *    TMM7: spare
 *
 *  Based on empirical testing, single-accumulator design achieves ~100% of
 *  multi-accumulator throughput due to AMX internal pipelining.
 */
NK_INTERNAL void nk_dots_bf16bf16f32_update_sapphire_amx( //
    nk_dots_bf16bf16f32_state_sapphire_amx_t *state,      //
    nk_dots_bf16bf16f32_a16x32_sapphire_amx_t const *a0,  //
    nk_dots_bf16bf16f32_a16x32_sapphire_amx_t const *a1,  //
    nk_dots_bf16bf16f32_a16x32_sapphire_amx_t const *a2,  //
    nk_dots_bf16bf16f32_b32x16_sapphire_amx_t const *b0,  //
    nk_dots_bf16bf16f32_b32x16_sapphire_amx_t const *b1,  //
    nk_dots_bf16bf16f32_b32x16_sapphire_amx_t const *b2) {

    // Load all tiles into registers
    _tile_loadd(0, state->data, 64); // C accumulator
    _tile_loadd(1, a0->data, 64);    // A0
    _tile_loadd(2, a1->data, 64);    // A1
    _tile_loadd(3, a2->data, 64);    // A2
    _tile_loadd(4, b0->data, 64);    // B0
    _tile_loadd(5, b1->data, 64);    // B1
    _tile_loadd(6, b2->data, 64);    // B2

    // Accumulate: C += A0×B0 + A1×B1 + A2×B2
    _tile_dpbf16ps(0, 1, 4); // C += A0 × B0
    _tile_dpbf16ps(0, 2, 5); // C += A1 × B1
    _tile_dpbf16ps(0, 3, 6); // C += A2 × B2

    // Store result
    _tile_stored(0, state->data, 64);
}

/* ============================================================================
 *  Composable Tile Primitives - INT8
 * ============================================================================ */

/*  Initialize INT8 output state to zero.
 */
NK_INTERNAL void nk_dots_i8i8i32_init_sapphire_amx(nk_dots_i8i8i32_state_sapphire_amx_t *state) {
    __m512i zero = _mm512_setzero_si512();
    for (nk_size_t r = 0; r < 16; r++) { _mm512_store_si512((__m512i *)state->data[r], zero); }
}

/*  Load A tile from row-major source with masking for edge tiles.
 */
NK_INTERNAL void nk_dots_i8i8i32_load_a_sapphire_amx( //
    nk_dots_i8i8i32_a16x64_sapphire_amx_t *a_tile,    //
    nk_i8_t const *src, nk_size_t src_stride,         //
    nk_size_t valid_rows, nk_size_t valid_cols) {

    __mmask64 col_mask = (valid_cols >= 64) ? 0xFFFFFFFFFFFFFFFFULL : ((__mmask64)1 << valid_cols) - 1;
    __m512i zero = _mm512_setzero_si512();

    for (nk_size_t r = 0; r < 16; r++) {
        if (r < valid_rows) {
            __m512i row = _mm512_maskz_loadu_epi8(col_mask, src + r * src_stride);
            _mm512_store_si512((__m512i *)a_tile->data[r], row);
        }
        else { _mm512_store_si512((__m512i *)a_tile->data[r], zero); }
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

    for (nk_size_t r = 0; r < valid_rows; r++) {
        __m512i row = _mm512_load_si512((__m512i const *)state->data[r]);
        _mm512_mask_storeu_epi32(dst + r * dst_stride_elements, col_mask, row);
    }
}

/*  Accumulate 3 A×B tile pairs into state using AMX TDPBSSD.
 *  Processes K=192 per call (3 × 64 INT8 elements).
 *  For indivisible K, pad unused tiles with zeros.
 *
 *  Register allocation (uses 7 of 8 TMM registers):
 *    TMM0: accumulator (C)
 *    TMM1-3: A tiles (a0, a1, a2)
 *    TMM4-6: B tiles (b0, b1, b2)
 *    TMM7: spare
 */
NK_INTERNAL void nk_dots_i8i8i32_update_sapphire_amx( //
    nk_dots_i8i8i32_state_sapphire_amx_t *state,      //
    nk_dots_i8i8i32_a16x64_sapphire_amx_t const *a0,  //
    nk_dots_i8i8i32_a16x64_sapphire_amx_t const *a1,  //
    nk_dots_i8i8i32_a16x64_sapphire_amx_t const *a2,  //
    nk_dots_i8i8i32_b64x16_sapphire_amx_t const *b0,  //
    nk_dots_i8i8i32_b64x16_sapphire_amx_t const *b1,  //
    nk_dots_i8i8i32_b64x16_sapphire_amx_t const *b2) {

    // Load all tiles into registers
    _tile_loadd(0, state->data, 64); // C accumulator
    _tile_loadd(1, a0->data, 64);    // A0
    _tile_loadd(2, a1->data, 64);    // A1
    _tile_loadd(3, a2->data, 64);    // A2
    _tile_loadd(4, b0->data, 64);    // B0
    _tile_loadd(5, b1->data, 64);    // B1
    _tile_loadd(6, b2->data, 64);    // B2

    // Accumulate: C += A0×B0 + A1×B1 + A2×B2
    _tile_dpbssd(0, 1, 4); // C += A0 × B0
    _tile_dpbssd(0, 2, 5); // C += A1 × B1
    _tile_dpbssd(0, 3, 6); // C += A2 × B2

    // Store result
    _tile_stored(0, state->data, 64);
}

/* ============================================================================
 *  Register-Resident 2×2 Update Primitives for GEMM
 *
 *  These keep C accumulators in TMM4-7 across the entire K-loop, avoiding
 *  4KB of load+store per K iteration. Usage pattern:
 *
 *    nk_dots_bf16bf16f32_zero2x2_sapphire_amx();
 *    for (k_tile = 0; k_tile < num_k_tiles; k_tile++) {
 *        // prepare a0, a1, b0, b1
 *        nk_dots_bf16bf16f32_update2x2_sapphire_amx(&a0, &a1, &b0, &b1);
 *    }
 *    nk_dots_bf16bf16f32_store2x2_sapphire_amx(&state);
 *
 *  Register allocation:
 *    TMM0-1: A tiles (rows 0-15 and 16-31)
 *    TMM2-3: B tiles (cols 0-15 and 16-31)
 *    TMM4-7: C accumulators (2×2 output grid, kept in registers)
 * ============================================================================ */

/*  Zero BF16 2×2 C accumulators in TMM4-7 (call once before K-loop).
 */
NK_INTERNAL void nk_dots_bf16bf16f32_zero2x2_sapphire_amx(void) {
    _tile_zero(4); // C[0,0]
    _tile_zero(5); // C[0,1]
    _tile_zero(6); // C[1,0]
    _tile_zero(7); // C[1,1]
}

/*  Accumulate BF16 2×2 output: C[i,j] += A[i] × B[j] for i,j in {0,1}.
 *  Assumes C accumulators already in TMM4-7 from previous zero2x2 or update2x2.
 *  Processes K=32 per call (1 tile pair along K dimension).
 */
NK_INTERNAL void nk_dots_bf16bf16f32_update2x2_sapphire_amx( //
    nk_dots_bf16bf16f32_a16x32_sapphire_amx_t const *a0,     //
    nk_dots_bf16bf16f32_a16x32_sapphire_amx_t const *a1,     //
    nk_dots_bf16bf16f32_b32x16_sapphire_amx_t const *b0,     //
    nk_dots_bf16bf16f32_b32x16_sapphire_amx_t const *b1) {

    // Load A tiles (rows 0-15 and 16-31)
    _tile_loadd(0, a0->data, 64);
    _tile_loadd(1, a1->data, 64);

    // Load B tiles (cols 0-15 and 16-31)
    _tile_loadd(2, b0->data, 64);
    _tile_loadd(3, b1->data, 64);

    // Compute 2×2 outer product, accumulating into C registers
    _tile_dpbf16ps(4, 0, 2); // C[0,0] += A0 × B0
    _tile_dpbf16ps(5, 0, 3); // C[0,1] += A0 × B1
    _tile_dpbf16ps(6, 1, 2); // C[1,0] += A1 × B0
    _tile_dpbf16ps(7, 1, 3); // C[1,1] += A1 × B1
}

/*  Accumulate BF16 2×2 with DIRECT A load from source matrix.
 *  Use this for full interior tiles to avoid buffer copy overhead.
 *  a0_ptr/a1_ptr point directly to source rows, a_stride is in bytes.
 */
NK_INTERNAL void nk_dots_bf16bf16f32_update2x2_direct_sapphire_amx(       //
    nk_bf16_t const *a0_ptr, nk_bf16_t const *a1_ptr, nk_size_t a_stride, //
    nk_dots_bf16bf16f32_b32x16_sapphire_amx_t const *b0,                  //
    nk_dots_bf16bf16f32_b32x16_sapphire_amx_t const *b1) {

    // Load A tiles directly from source (no buffer copy!)
    _tile_loadd(0, a0_ptr, a_stride);
    _tile_loadd(1, a1_ptr, a_stride);

    // Load B tiles (already packed)
    _tile_loadd(2, b0->data, 64);
    _tile_loadd(3, b1->data, 64);

    // Compute 2×2 outer product
    _tile_dpbf16ps(4, 0, 2);
    _tile_dpbf16ps(5, 0, 3);
    _tile_dpbf16ps(6, 1, 2);
    _tile_dpbf16ps(7, 1, 3);
}

/*  Store BF16 2×2 C accumulators from TMM4-7 to state buffer (for masked output).
 */
NK_INTERNAL void nk_dots_bf16bf16f32_store2x2_sapphire_amx( //
    nk_dots_bf16bf16f32_state2x2_sapphire_amx_t *state) {
    _tile_stored(4, state->c[0][0].data, 64);
    _tile_stored(5, state->c[0][1].data, 64);
    _tile_stored(6, state->c[1][0].data, 64);
    _tile_stored(7, state->c[1][1].data, 64);
}

/*  Store BF16 2×2 C accumulators DIRECTLY to output matrix (no intermediate buffer).
 *  Use for full interior tiles. c_stride is in bytes.
 */
NK_INTERNAL void nk_dots_bf16bf16f32_store2x2_direct_sapphire_amx( //
    nk_f32_t *c, nk_size_t c_stride) {
    _tile_stored(4, c, c_stride);
    _tile_stored(5, c + 16, c_stride);
    _tile_stored(6, (nk_f32_t *)((char *)c + 16 * c_stride), c_stride);
    _tile_stored(7, (nk_f32_t *)((char *)c + 16 * c_stride) + 16, c_stride);
}

/*  Zero INT8 2×2 C accumulators in TMM4-7 (call once before K-loop).
 */
NK_INTERNAL void nk_dots_i8i8i32_zero2x2_sapphire_amx(void) {
    _tile_zero(4);
    _tile_zero(5);
    _tile_zero(6);
    _tile_zero(7);
}

/*  Accumulate INT8 2×2 output: C[i,j] += A[i] × B[j] for i,j in {0,1}.
 *  Assumes C accumulators already in TMM4-7.
 *  Processes K=64 per call (1 tile pair along K dimension).
 */
NK_INTERNAL void nk_dots_i8i8i32_update2x2_sapphire_amx( //
    nk_dots_i8i8i32_a16x64_sapphire_amx_t const *a0,     //
    nk_dots_i8i8i32_a16x64_sapphire_amx_t const *a1,     //
    nk_dots_i8i8i32_b64x16_sapphire_amx_t const *b0,     //
    nk_dots_i8i8i32_b64x16_sapphire_amx_t const *b1) {

    _tile_loadd(0, a0->data, 64);
    _tile_loadd(1, a1->data, 64);
    _tile_loadd(2, b0->data, 64);
    _tile_loadd(3, b1->data, 64);

    _tile_dpbssd(4, 0, 2);
    _tile_dpbssd(5, 0, 3);
    _tile_dpbssd(6, 1, 2);
    _tile_dpbssd(7, 1, 3);
}

/*  Accumulate INT8 2×2 with DIRECT A load from source matrix.
 *  Use this for full interior tiles to avoid buffer copy overhead.
 */
NK_INTERNAL void nk_dots_i8i8i32_update2x2_direct_sapphire_amx(       //
    nk_i8_t const *a0_ptr, nk_i8_t const *a1_ptr, nk_size_t a_stride, //
    nk_dots_i8i8i32_b64x16_sapphire_amx_t const *b0,                  //
    nk_dots_i8i8i32_b64x16_sapphire_amx_t const *b1) {

    _tile_loadd(0, a0_ptr, a_stride);
    _tile_loadd(1, a1_ptr, a_stride);
    _tile_loadd(2, b0->data, 64);
    _tile_loadd(3, b1->data, 64);

    _tile_dpbssd(4, 0, 2);
    _tile_dpbssd(5, 0, 3);
    _tile_dpbssd(6, 1, 2);
    _tile_dpbssd(7, 1, 3);
}

/*  Store INT8 2×2 C accumulators from TMM4-7 to state buffer (for masked output).
 */
NK_INTERNAL void nk_dots_i8i8i32_store2x2_sapphire_amx( //
    nk_dots_i8i8i32_state2x2_sapphire_amx_t *state) {
    _tile_stored(4, state->c[0][0].data, 64);
    _tile_stored(5, state->c[0][1].data, 64);
    _tile_stored(6, state->c[1][0].data, 64);
    _tile_stored(7, state->c[1][1].data, 64);
}

/*  Store INT8 2×2 C accumulators DIRECTLY to output matrix (no intermediate buffer).
 *  Use for full interior tiles. c_stride is in bytes.
 */
NK_INTERNAL void nk_dots_i8i8i32_store2x2_direct_sapphire_amx( //
    nk_i32_t *c, nk_size_t c_stride) {
    _tile_stored(4, c, c_stride);
    _tile_stored(5, c + 16, c_stride);
    _tile_stored(6, (nk_i32_t *)((char *)c + 16 * c_stride), c_stride);
    _tile_stored(7, (nk_i32_t *)((char *)c + 16 * c_stride) + 16, c_stride);
}

/* ============================================================================
 *  Store 2×2 State to Output Matrix with Masking
 * ============================================================================ */

/*  Store BF16 2×2 state to output matrix with masking for edge tiles.
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

/*  Store INT8 2×2 state to output matrix with masking for edge tiles.
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

/* ============================================================================
 *  Pack Transposed - for cross-correlation (A × Aᵀ)
 * ============================================================================ */

/*  Pack A transposed into B format for BF16.
 *  Converts A[n][k] -> B[k][n] with pair-interleaving for TDPBF16PS.
 *  Used for cross-correlation: load vectors into A, pack to B, compute A×B.
 *
 *  Uses vectorized 16×32 transpose with YMM operations:
 *  - Process 2 K-columns at a time (one k_group)
 *  - Each column has 16 elements (strided in A)
 *  - Interleave pairs and store contiguously to B
 */
NK_INTERNAL void nk_dots_bf16bf16f32_pack_transposed_sapphire_amx( //
    nk_dots_bf16bf16f32_a16x32_sapphire_amx_t const *a_tile,       //
    nk_dots_bf16bf16f32_b32x16_sapphire_amx_t *b_tile) {

    // Process 2 K-columns at a time (forming one k_group in output)
    for (nk_size_t k_group = 0; k_group < 16; k_group++) {
        nk_size_t k0 = k_group * 2;
        nk_size_t k1 = k_group * 2 + 1;

        // Load 2 columns from A (16 elements each) using gather-like pattern
        // A.data[n][k] has stride 32 elements between rows
        // Use YMM for 16 × 16-bit = 256 bits

        // Extract columns k0 and k1 from all 16 rows
        __m256i col0, col1;
        {
            // Gather column k0: elements at A.data[0][k0], A.data[1][k0], ..., A.data[15][k0]
            // Load 8 rows at a time using 32-bit gather (gets 2 BF16 per element)
            __m256i idx_lo = _mm256_setr_epi32(0, 32, 64, 96, 128, 160, 192, 224);
            __m256i idx_hi = _mm256_setr_epi32(256, 288, 320, 352, 384, 416, 448, 480);

            // Gather 32-bit elements containing pairs of BF16
            nk_bf16_t const *base0 = &a_tile->data[0][k0];
            __m256i gather_lo = _mm256_i32gather_epi32((int const *)base0, idx_lo, 2);
            __m256i gather_hi = _mm256_i32gather_epi32((int const *)base0, idx_hi, 2);

            // Extract the low 16-bit from each 32-bit gather result
            // gather_lo has [a[0][k0:k0+1], a[1][k0:k0+1], ..., a[7][k0:k0+1]]
            // We want just the k0 values (low 16 bits of each 32-bit word)
            __m256i mask_lo16 = _mm256_set1_epi32(0x0000FFFF);
            __m256i col0_lo_32 = _mm256_and_si256(gather_lo, mask_lo16);
            __m256i col0_hi_32 = _mm256_and_si256(gather_hi, mask_lo16);

            // Pack 8×32-bit -> 8×16-bit for each half
            col0 = _mm256_packus_epi32(col0_lo_32, col0_hi_32);
            // packus interleaves lanes, need to permute
            col0 = _mm256_permute4x64_epi64(col0, 0xD8); // [0,2,1,3]

            // Extract k1 values (high 16 bits of each 32-bit word)
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

        // Store to B.data[k_group][0..15][0..1]
        // B.data[k_group] is 16×2 = 32 BF16 = 64 bytes
        _mm256_store_si256((__m256i *)&b_tile->data[k_group][0][0], out0);
        _mm256_store_si256((__m256i *)&b_tile->data[k_group][8][0], out1);
    }
    nk_compiler_barrier_sapphire_amx_();
}

/*  Pack A transposed into B format for INT8.
 *  Converts A[n][k] -> B[k][n] with quad-interleaving for TDPBSSD.
 *
 *  Uses vectorized 16×64 transpose with ZMM operations:
 *  - Process 4 K-columns at a time (one k_group)
 *  - Each column has 16 elements (strided in A)
 *  - Interleave quads and store contiguously to B
 */
NK_INTERNAL void nk_dots_i8i8i32_pack_transposed_sapphire_amx( //
    nk_dots_i8i8i32_a16x64_sapphire_amx_t const *a_tile,       //
    nk_dots_i8i8i32_b64x16_sapphire_amx_t *b_tile) {

    // Process 4 K-columns at a time (forming one k_group in output)
    for (nk_size_t k_group = 0; k_group < 16; k_group++) {
        nk_size_t k0 = k_group * 4;

        // For INT8, A.data[n][k] has stride 64 bytes between rows
        // Load 4 consecutive columns from all 16 rows

        // Use XMM for column extraction (16 × 8-bit = 128 bits per column)
        __m128i col0, col1, col2, col3;

        // Gather columns k0..k3 using 32-bit gather (gets 4 I8 per element)
        {
            __m128i idx = _mm_setr_epi32(0, 64, 128, 192);
            nk_i8_t const *base = &a_tile->data[0][k0];

            // Gather 4 rows at a time
            __m128i g0 = _mm_i32gather_epi32((int const *)base, idx, 1);
            __m128i g1 = _mm_i32gather_epi32((int const *)(base + 4 * 64), idx, 1);
            __m128i g2 = _mm_i32gather_epi32((int const *)(base + 8 * 64), idx, 1);
            __m128i g3 = _mm_i32gather_epi32((int const *)(base + 12 * 64), idx, 1);

            // g0 contains [a[0][k0:k0+3], a[1][k0:k0+3], a[2][k0:k0+3], a[3][k0:k0+3]]
            // Each 32-bit element has 4 I8 values for columns k0,k1,k2,k3

            // Extract column k0 (byte 0 from each 32-bit word)
            __m128i mask_byte0 = _mm_set1_epi32(0x000000FF);
            __m128i c0_0 = _mm_and_si128(g0, mask_byte0);
            __m128i c0_1 = _mm_and_si128(g1, mask_byte0);
            __m128i c0_2 = _mm_and_si128(g2, mask_byte0);
            __m128i c0_3 = _mm_and_si128(g3, mask_byte0);

            // Pack into bytes
            __m128i c0_01 = _mm_packus_epi32(c0_0, c0_1);
            __m128i c0_23 = _mm_packus_epi32(c0_2, c0_3);
            col0 = _mm_packus_epi16(c0_01, c0_23);

            // Extract columns k1, k2, k3 similarly
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

        // Store to B.data[k_group][0..15][0..3]
        // B.data[k_group] is 16×4 = 64 I8 = 64 bytes
        _mm_store_si128((__m128i *)&b_tile->data[k_group][0][0], q0);
        _mm_store_si128((__m128i *)&b_tile->data[k_group][4][0], q1);
        _mm_store_si128((__m128i *)&b_tile->data[k_group][8][0], q2);
        _mm_store_si128((__m128i *)&b_tile->data[k_group][12][0], q3);
    }
    nk_compiler_barrier_sapphire_amx_();
}

/* ============================================================================
 *  Cross-Correlation API (A × Aᵀ)
 * ============================================================================ */

/*  Compute self cross-correlation for up to 16 BF16 vectors.
 *  Result[i,j] = dot(vectors[i], vectors[j])
 *
 *  @param vectors   Row-major array of n_vectors, each of dimension k
 *  @param n_vectors Number of vectors (1-16, padded internally if < 16)
 *  @param k         Vector dimension (padded to multiple of 96 internally)
 *  @param stride    Byte stride between vectors
 *  @param result    Output n_vectors × n_vectors matrix
 *  @param result_stride Byte stride between result rows
 */
NK_PUBLIC void nk_cross_bf16bf16f32_sapphire_amx(               //
    nk_bf16_t const *vectors, nk_size_t n_vectors, nk_size_t k, //
    nk_size_t stride, nk_f32_t *result, nk_size_t result_stride) {

    nk_size_t const stride_elements = stride / sizeof(nk_bf16_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);

    // Round K up to multiple of 96 (3 tiles × 32 elements)
    nk_size_t const k_tiles = (k + 31) / 32;
    nk_size_t const k_tile_groups = (k_tiles + 2) / 3; // Groups of 3 tiles

    // Allocate tile buffers (3 A tiles + 3 B tiles per group)
    nk_dots_bf16bf16f32_a16x32_sapphire_amx_t a_tiles[3];
    nk_dots_bf16bf16f32_b32x16_sapphire_amx_t b_tiles[3];
    nk_dots_bf16bf16f32_state_sapphire_amx_t state;

    // Configure AMX tiles
    nk_amx_tile_configure_sapphire_amx_();

    // Initialize output state
    nk_dots_bf16bf16f32_init_sapphire_amx(&state);

    // Process K dimension in groups of 96 (3 tiles)
    for (nk_size_t kg = 0; kg < k_tile_groups; kg++) {
        nk_size_t k_base = kg * 96;

        // Load 3 A tiles from vectors
        for (int t = 0; t < 3; t++) {
            nk_size_t k_start = k_base + t * 32;
            nk_size_t valid_cols = (k_start + 32 <= k) ? 32 : (k > k_start ? k - k_start : 0);
            nk_bf16_t const *src = vectors + k_start;
            nk_dots_bf16bf16f32_load_a_sapphire_amx(&a_tiles[t], src, stride_elements, n_vectors, valid_cols);
        }

        // Pack A transposed into B tiles
        for (int t = 0; t < 3; t++) { nk_dots_bf16bf16f32_pack_transposed_sapphire_amx(&a_tiles[t], &b_tiles[t]); }

        // Accumulate: state += A × B
        nk_dots_bf16bf16f32_update_sapphire_amx(&state, &a_tiles[0], &a_tiles[1], &a_tiles[2], &b_tiles[0], &b_tiles[1],
                                                &b_tiles[2]);
    }

    // Store result
    nk_dots_bf16bf16f32_store_sapphire_amx(&state, result, result_stride_elements, n_vectors, n_vectors);
}

/*  Compute self cross-correlation for up to 16 INT8 vectors.
 *  Result[i,j] = dot(vectors[i], vectors[j])
 */
NK_PUBLIC void nk_cross_i8i8i32_sapphire_amx(                 //
    nk_i8_t const *vectors, nk_size_t n_vectors, nk_size_t k, //
    nk_size_t stride, nk_i32_t *result, nk_size_t result_stride) {

    nk_size_t const result_stride_elements = result_stride / sizeof(nk_i32_t);

    // Round K up to multiple of 192 (3 tiles × 64 elements)
    nk_size_t const k_tiles = (k + 63) / 64;
    nk_size_t const k_tile_groups = (k_tiles + 2) / 3;

    // Allocate tile buffers
    nk_dots_i8i8i32_a16x64_sapphire_amx_t a_tiles[3];
    nk_dots_i8i8i32_b64x16_sapphire_amx_t b_tiles[3];
    nk_dots_i8i8i32_state_sapphire_amx_t state;

    // Configure AMX tiles
    nk_amx_tile_configure_sapphire_amx_();

    // Initialize output state
    nk_dots_i8i8i32_init_sapphire_amx(&state);

    // Process K dimension in groups of 192 (3 tiles)
    for (nk_size_t kg = 0; kg < k_tile_groups; kg++) {
        nk_size_t k_base = kg * 192;

        // Load 3 A tiles from vectors
        for (int t = 0; t < 3; t++) {
            nk_size_t k_start = k_base + t * 64;
            nk_size_t valid_cols = (k_start + 64 <= k) ? 64 : (k > k_start ? k - k_start : 0);
            nk_i8_t const *src = vectors + k_start;
            nk_dots_i8i8i32_load_a_sapphire_amx(&a_tiles[t], src, stride, n_vectors, valid_cols);
        }

        // Pack A transposed into B tiles
        for (int t = 0; t < 3; t++) { nk_dots_i8i8i32_pack_transposed_sapphire_amx(&a_tiles[t], &b_tiles[t]); }

        // Accumulate: state += A × B
        nk_dots_i8i8i32_update_sapphire_amx(&state, &a_tiles[0], &a_tiles[1], &a_tiles[2], &b_tiles[0], &b_tiles[1],
                                            &b_tiles[2]);
    }

    // Store result
    nk_dots_i8i8i32_store_sapphire_amx(&state, result, result_stride_elements, n_vectors, n_vectors);
}

/*  AVX-512 edge matmul for BF16 → F32.
 *  Computes C[m_start:m_end, n_start:n_end] using row-major B edge data.
 *  Used for boundary regions where AMX is overkill.
 *
 *  B is stored in row-major as b_edge[row * k + col] where row is the N index.
 *  This computes: C[i,j] = sum_over_k(A[i,k] * B[j,k])
 */
NK_INTERNAL void nk_dots_bf16bf16f32_avx512_edge_(            //
    nk_bf16_t const *a, nk_bf16_t const *b_edge, nk_f32_t *c, //
    nk_size_t m, nk_size_t n, nk_size_t k,                    //
    nk_size_t a_stride_elements, nk_size_t b_stride_k, nk_size_t c_stride_elements) {

    // Process each output row
    for (nk_size_t i = 0; i < m; i++) {
        nk_bf16_t const *a_row = a + i * a_stride_elements;

        // Process output columns in chunks of 16 (AVX-512 width for F32)
        for (nk_size_t j = 0; j < n; j += 16) {
            nk_size_t const j_count = (j + 16 <= n) ? 16 : n - j;
            __m512 acc = _mm512_setzero_ps();

            // Dot product over K dimension - process 2 at a time for BF16 pairs
            for (nk_size_t kk = 0; kk < k; kk++) {
                // Broadcast A[i,k] to F32
                nk_f32_t a_val = (nk_f32_t)a_row[kk];
                __m512 a_bc = _mm512_set1_ps(a_val);

                // Gather B[j:j+16, k] - each B row is stored with stride b_stride_k
                // b_edge[row * b_stride_k + col] where row in [0,n), col in [0,k)
                NK_ALIGN64 nk_f32_t b_vals[16];
                for (nk_size_t jj = 0; jj < j_count; jj++) {
                    b_vals[jj] = (nk_f32_t)b_edge[(j + jj) * b_stride_k + kk];
                }
                for (nk_size_t jj = j_count; jj < 16; jj++) b_vals[jj] = 0.0f;

                __m512 b_vec = _mm512_load_ps(b_vals);
                acc = _mm512_fmadd_ps(a_bc, b_vec, acc);
            }

            // Store with mask
            __mmask16 mask = (j_count >= 16) ? 0xFFFF : ((__mmask16)1 << j_count) - 1;
            _mm512_mask_storeu_ps(c + i * c_stride_elements + j, mask, acc);
        }
    }
}

/*  AVX-512 edge matmul for I8 → I32.
 *  Computes C[m_start:m_end, n_start:n_end] using row-major B edge data.
 *  Used for boundary regions where AMX is overkill.
 */
NK_INTERNAL void nk_dots_i8i8i32_avx512_edge_(            //
    nk_i8_t const *a, nk_i8_t const *b_edge, nk_i32_t *c, //
    nk_size_t m, nk_size_t n, nk_size_t k,                //
    nk_size_t a_stride, nk_size_t b_stride_k, nk_size_t c_stride_elements) {

    // Process each output row
    for (nk_size_t i = 0; i < m; i++) {
        nk_i8_t const *a_row = a + i * a_stride;

        // Process output columns in chunks of 16 (AVX-512 width for I32)
        for (nk_size_t j = 0; j < n; j += 16) {
            nk_size_t const j_count = (j + 16 <= n) ? 16 : n - j;
            __m512i acc = _mm512_setzero_si512();

            // Dot product over K dimension
            for (nk_size_t kk = 0; kk < k; kk++) {
                // Broadcast A[i,k] to I32
                nk_i32_t a_val = (nk_i32_t)a_row[kk];
                __m512i a_bc = _mm512_set1_epi32(a_val);

                // Gather B[j:j+16, k]
                NK_ALIGN64 nk_i32_t b_vals[16];
                for (nk_size_t jj = 0; jj < j_count; jj++) {
                    b_vals[jj] = (nk_i32_t)b_edge[(j + jj) * b_stride_k + kk];
                }
                for (nk_size_t jj = j_count; jj < 16; jj++) b_vals[jj] = 0;

                __m512i b_vec = _mm512_load_si512((__m512i const *)b_vals);
                acc = _mm512_add_epi32(acc, _mm512_mullo_epi32(a_bc, b_vec));
            }

            // Store with mask
            __mmask16 mask = (j_count >= 16) ? 0xFFFF : ((__mmask16)1 << j_count) - 1;
            _mm512_mask_storeu_epi32(c + i * c_stride_elements + j, mask, acc);
        }
    }
}

/*  BF16 packed buffer size: header + all tiles for full N rows + N edge.
 *  Hybrid layout:
 *    - Tiles include K remainder (zero-padded) for AMX to handle full dot products
 *    - N edge (remaining rows) stored row-major for simple AVX-512 edge kernel
 */
NK_PUBLIC nk_size_t nk_dots_bf16bf16f32_packed_size_sapphire_amx(nk_size_t n, nk_size_t k) {
    nk_size_t const tmm_rows = 16;
    nk_size_t const tmm_cols = 32;
    nk_size_t const tile_bytes = 512 * sizeof(nk_bf16_t); // 16×32×2 = 1KB

    nk_size_t const full_n_tiles = n / tmm_rows;
    nk_size_t const tiles_along_k = (k + tmm_cols - 1) / tmm_cols; // Ceiling division
    nk_size_t const n_edge_rows = n - full_n_tiles * tmm_rows;

    // Header (64 bytes aligned)
    nk_size_t size = sizeof(nk_dots_amx_packed_header_t);

    // All tiles for full N rows (Morton-ordered, pair-interleaved, K remainder zero-padded)
    size += full_n_tiles * tiles_along_k * tile_bytes;

    // N edge: remaining rows for ALL K columns, stored row-major
    if (n_edge_rows > 0) size += n_edge_rows * k * sizeof(nk_bf16_t);

    return size;
}

/*  I8 packed buffer size: header + all tiles for full N rows + N edge.
 */
NK_PUBLIC nk_size_t nk_dots_i8i8i32_packed_size_sapphire_amx(nk_size_t n, nk_size_t k) {
    nk_size_t const tmm_rows = 16;
    nk_size_t const tmm_cols = 64;
    nk_size_t const tile_bytes = 1024 * sizeof(nk_i8_t); // 16×64×1 = 1KB

    nk_size_t const full_n_tiles = n / tmm_rows;
    nk_size_t const tiles_along_k = (k + tmm_cols - 1) / tmm_cols; // Ceiling division
    nk_size_t const n_edge_rows = n - full_n_tiles * tmm_rows;

    // Header (64 bytes aligned)
    nk_size_t size = sizeof(nk_dots_amx_packed_header_t);

    // All tiles for full N rows (Morton-ordered, quad-interleaved, K remainder zero-padded)
    size += full_n_tiles * tiles_along_k * tile_bytes;

    // N edge: remaining rows for ALL K columns, stored row-major
    if (n_edge_rows > 0) size += n_edge_rows * k * sizeof(nk_i8_t);

    return size;
}

/*  Pack BF16 B matrix with hybrid layout:
 *    - Header with layout metadata
 *    - All tiles for full N rows: Morton Z-curve ordered, pair-interleaved (for AMX)
 *      Including K remainder tiles (zero-padded) so AMX can compute full dot products
 *    - N edge rows: row-major (for AVX-512 edge kernel)
 *
 *  AMX BF16 tile format: for TDPBF16PS, B tile should have elements arranged so that
 *  consecutive pairs of columns are interleaved by rows:
 *    [col0_row0, col1_row0, col0_row1, col1_row1, ..., col0_row15, col1_row15,
 *     col2_row0, col3_row0, col2_row1, col3_row1, ...]
 *
 *  Interleaving formula: packed_idx = (col / 2) * 32 + row * 2 + (col % 2)
 */
NK_PUBLIC void nk_dots_bf16bf16f32_pack_sapphire_amx( //
    nk_bf16_t const *b, nk_size_t n, nk_size_t k,     //
    nk_size_t b_stride, void *b_packed) {

    // AMX BF16 tile dimensions: 16 rows × 32 columns (512 BF16 elements = 1KB)
    nk_size_t const tmm_rows = 16;
    nk_size_t const tmm_cols = 32;
    nk_size_t const tile_elements = 512;
    nk_size_t const tile_bytes = tile_elements * sizeof(nk_bf16_t);
    nk_size_t const b_stride_elements = b_stride / sizeof(nk_bf16_t);

    // Compute layout dimensions
    nk_size_t const num_n_tiles = n / tmm_rows;
    nk_size_t const num_k_tiles = (k + tmm_cols - 1) / tmm_cols;
    nk_size_t const n_remainder_rows = n - num_n_tiles * tmm_rows;
    nk_size_t const total_tiles = num_n_tiles * num_k_tiles;

    // Write header with layout metadata
    nk_dots_amx_packed_header_t *header = (nk_dots_amx_packed_header_t *)b_packed;
    header->full_n_tiles = (nk_u32_t)num_n_tiles;
    header->full_k_tiles = (nk_u32_t)num_k_tiles;
    header->n_edge_rows = (nk_u32_t)n_remainder_rows;

    // Compute memory region offsets
    nk_size_t const tiles_offset = sizeof(nk_dots_amx_packed_header_t);
    nk_size_t const n_edge_offset = tiles_offset + total_tiles * tile_bytes;
    header->n_edge_offset = (nk_u32_t)n_edge_offset;

    // Pointers to packed data regions
    nk_bf16_t *tiles_ptr = (nk_bf16_t *)((char *)b_packed + tiles_offset);
    nk_bf16_t *n_edge_ptr = (nk_bf16_t *)((char *)b_packed + n_edge_offset);

    // Zero-initialize all tiles (handles K remainder padding)
    for (nk_size_t i = 0; i < total_tiles * tile_elements; i++) tiles_ptr[i] = 0;

    // Pack tiles using LINEAR ordering: tile_index = n_tile * num_k_tiles + k_tile
    // This provides sequential memory access when streaming along K dimension,
    // which is critical for cache efficiency in the compute kernel.
    for (nk_size_t n_tile = 0; n_tile < num_n_tiles; n_tile++) {
        for (nk_size_t k_tile = 0; k_tile < num_k_tiles; k_tile++) {

            // Linear tile index: all K-tiles for one N-tile are contiguous
            nk_size_t const tile_index = n_tile * num_k_tiles + k_tile;
            nk_bf16_t *tile_output = tiles_ptr + tile_index * tile_elements;

            // Source coordinates in original B matrix
            nk_size_t const src_row_start = n_tile * tmm_rows;
            nk_size_t const src_col_start = k_tile * tmm_cols;
            nk_size_t const cols_to_pack = (src_col_start + tmm_cols <= k) ? tmm_cols : (k - src_col_start);

            // Pack with pair-interleaving as required by TDPBF16PS instruction.
            // AMX expects: [col0_row0, col1_row0, col0_row1, col1_row1, col2_row0, col3_row0, ...]
            // Formula: packed_idx = (col / 2) * 32 + row * 2 + (col % 2)
            for (nk_size_t row = 0; row < tmm_rows; row++) {
                for (nk_size_t col = 0; col < cols_to_pack; col++) {
                    nk_size_t const src_idx = (src_row_start + row) * b_stride_elements + src_col_start + col;
                    nk_size_t const dst_idx = (col / 2) * 32 + row * 2 + (col % 2);
                    tile_output[dst_idx] = b[src_idx];
                }
            }
        }
    }

    // Pack N-remainder rows in simple row-major format (for AVX-512 fallback)
    if (n_remainder_rows > 0) {
        nk_size_t const remainder_start_row = num_n_tiles * tmm_rows;
        for (nk_size_t row = 0; row < n_remainder_rows; row++) {
            for (nk_size_t col = 0; col < k; col++) {
                n_edge_ptr[row * k + col] = b[(remainder_start_row + row) * b_stride_elements + col];
            }
        }
    }
}

/*  Pack I8 B matrix with hybrid layout:
 *    - Header with layout metadata
 *    - All tiles for full N rows: linearly ordered, quad-interleaved (for AMX)
 *      Including K remainder tiles (zero-padded) so AMX can compute full dot products
 *    - N edge rows: row-major (for AVX-512 edge kernel)
 *
 *  AMX INT8 tile format: for TDPBSSD, B tile should have 4 consecutive columns
 *  interleaved by rows:
 *    [col0_row0, col1_row0, col2_row0, col3_row0, col0_row1, col1_row1, ...]
 *
 *  Interleaving formula: packed_idx = (col / 4) * 64 + row * 4 + (col % 4)
 */
NK_PUBLIC void nk_dots_i8i8i32_pack_sapphire_amx( //
    nk_i8_t const *b, nk_size_t n, nk_size_t k,   //
    nk_size_t b_stride, void *b_packed) {

    // AMX I8 tile dimensions: 16 rows × 64 columns (1024 I8 elements = 1KB)
    nk_size_t const tmm_rows = 16;
    nk_size_t const tmm_cols = 64;
    nk_size_t const tile_elements = 1024;
    nk_size_t const tile_bytes = tile_elements * sizeof(nk_i8_t);

    // Compute layout dimensions
    nk_size_t const num_n_tiles = n / tmm_rows;
    nk_size_t const num_k_tiles = (k + tmm_cols - 1) / tmm_cols;
    nk_size_t const n_remainder_rows = n - num_n_tiles * tmm_rows;
    nk_size_t const total_tiles = num_n_tiles * num_k_tiles;

    // Write header with layout metadata
    nk_dots_amx_packed_header_t *header = (nk_dots_amx_packed_header_t *)b_packed;
    header->full_n_tiles = (nk_u32_t)num_n_tiles;
    header->full_k_tiles = (nk_u32_t)num_k_tiles;
    header->n_edge_rows = (nk_u32_t)n_remainder_rows;

    // Compute memory region offsets
    nk_size_t const tiles_offset = sizeof(nk_dots_amx_packed_header_t);
    nk_size_t const n_edge_offset = tiles_offset + total_tiles * tile_bytes;
    header->n_edge_offset = (nk_u32_t)n_edge_offset;

    // Pointers to packed data regions
    nk_i8_t *tiles_ptr = (nk_i8_t *)((char *)b_packed + tiles_offset);
    nk_i8_t *n_edge_ptr = (nk_i8_t *)((char *)b_packed + n_edge_offset);

    // Zero-initialize all tiles (handles K remainder padding)
    for (nk_size_t i = 0; i < total_tiles * tile_elements; i++) tiles_ptr[i] = 0;

    // Pack tiles using LINEAR ordering: tile_index = n_tile * num_k_tiles + k_tile
    // This provides sequential memory access when streaming along K dimension.
    for (nk_size_t n_tile = 0; n_tile < num_n_tiles; n_tile++) {
        for (nk_size_t k_tile = 0; k_tile < num_k_tiles; k_tile++) {

            // Linear tile index: all K-tiles for one N-tile are contiguous
            nk_size_t const tile_index = n_tile * num_k_tiles + k_tile;
            nk_i8_t *tile_output = tiles_ptr + tile_index * tile_elements;

            // Source coordinates in original B matrix
            nk_size_t const src_row_start = n_tile * tmm_rows;
            nk_size_t const src_col_start = k_tile * tmm_cols;
            nk_size_t const cols_to_pack = (src_col_start + tmm_cols <= k) ? tmm_cols : (k - src_col_start);

            // Pack with quad-interleaving as required by TDPBSSD instruction.
            // AMX expects: [col0_row0, col1_row0, col2_row0, col3_row0, col0_row1, ...]
            // Formula: packed_idx = (col / 4) * 64 + row * 4 + (col % 4)
            for (nk_size_t row = 0; row < tmm_rows; row++) {
                for (nk_size_t col = 0; col < cols_to_pack; col++) {
                    nk_size_t const src_idx = (src_row_start + row) * b_stride + src_col_start + col;
                    nk_size_t const dst_idx = (col / 4) * 64 + row * 4 + (col % 4);
                    tile_output[dst_idx] = b[src_idx];
                }
            }
        }
    }

    // Pack N-remainder rows in simple row-major format (for AVX-512 fallback)
    if (n_remainder_rows > 0) {
        nk_size_t const remainder_start_row = num_n_tiles * tmm_rows;
        for (nk_size_t row = 0; row < n_remainder_rows; row++) {
            for (nk_size_t col = 0; col < k; col++) {
                n_edge_ptr[row * k + col] = b[(remainder_start_row + row) * b_stride + col];
            }
        }
    }
}

/*  BF16 → F32 matmul: C[m×n] = A[m×k] × B[n×k]ᵀ
 *
 *  Unified implementation using composable tile primitives.
 *  All I/O goes through aligned tile buffers for consistent behavior.
 *
 *  Uses register-resident 2×2 update pattern:
 *  - TMM4-7 hold C accumulators across entire K-loop (no redundant load/store)
 *  - 32×32 output blocks processed per M×N iteration
 *
 *  Single-threaded. For parallel execution, partition A rows across threads.
 */
NK_PUBLIC void nk_dots_bf16bf16f32_sapphire_amx(           //
    nk_bf16_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t m, nk_size_t n, nk_size_t k,                 //
    nk_size_t a_stride, nk_size_t c_stride) {

    // Parse packed B header
    nk_dots_amx_packed_header_t const *header = (nk_dots_amx_packed_header_t const *)b_packed;
    nk_size_t const num_n_tiles = header->full_n_tiles;
    nk_size_t const num_k_tiles = header->full_k_tiles;
    nk_size_t const n_edge_rows = header->n_edge_rows;

    // Packed B data regions
    nk_bf16_t const *b_tiles = (nk_bf16_t const *)((char const *)b_packed + sizeof(nk_dots_amx_packed_header_t));
    nk_bf16_t const *n_edge_ptr = (nk_bf16_t const *)((char const *)b_packed + header->n_edge_offset);

    // Stride conversions
    nk_size_t const a_stride_elements = a_stride / sizeof(nk_bf16_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);

    // Tile dimensions
    nk_size_t const tile_k = 32;     // K elements per BF16 tile
    nk_size_t const tile_size = 512; // Elements per packed tile
    nk_size_t const full_n = num_n_tiles * 16;

    // Block counts (32×32 output blocks = 2×2 tiles)
    nk_size_t const num_m_blocks = (m + 31) / 32; // Ceiling division
    nk_size_t const num_n_blocks = num_n_tiles / 2;

    if (num_k_tiles == 0) return;

    // Tile buffers for A (only used for edge tiles)
    nk_dots_bf16bf16f32_a16x32_sapphire_amx_t a0, a1;
    nk_dots_bf16bf16f32_state2x2_sapphire_amx_t c_state;

    // Precompute: number of full K-tiles (no masking needed)
    nk_size_t const num_full_k_tiles = k / tile_k;
    nk_size_t const k_remainder = k % tile_k;

    nk_amx_tile_configure_sapphire_amx_();

    // Process all 32×32 M×N blocks (including partial edge blocks)
    for (nk_size_t m_block = 0; m_block < num_m_blocks; m_block++) {
        nk_size_t const m_row = m_block * 32;
        nk_size_t const rows_valid = (m_row + 32 <= m) ? 32 : (m - m_row);
        nk_size_t const is_full_m_block = (rows_valid == 32);

        // Process full N-blocks (pairs of 16-column tiles = 32 columns)
        for (nk_size_t n_block = 0; n_block < num_n_blocks; n_block++) {
            nk_size_t const n_col = n_block * 32;

            // B tile base indices (linear layout: n_tile * num_k_tiles + k_tile)
            nk_size_t const b_n0_base = (n_block * 2) * num_k_tiles;
            nk_size_t const b_n1_base = (n_block * 2 + 1) * num_k_tiles;

            // Zero accumulators in registers
            nk_dots_bf16bf16f32_zero2x2_sapphire_amx();

            // FAST PATH: Full M-block with full K-tiles → direct A load
            if (is_full_m_block) {
                // Process full K-tiles with direct load
                for (nk_size_t k_tile = 0; k_tile < num_full_k_tiles; k_tile++) {
                    nk_size_t const k_off = k_tile * tile_k;

                    nk_dots_bf16bf16f32_b32x16_sapphire_amx_t const *b0 =
                        (nk_dots_bf16bf16f32_b32x16_sapphire_amx_t const *)(b_tiles + (b_n0_base + k_tile) * tile_size);
                    nk_dots_bf16bf16f32_b32x16_sapphire_amx_t const *b1 =
                        (nk_dots_bf16bf16f32_b32x16_sapphire_amx_t const *)(b_tiles + (b_n1_base + k_tile) * tile_size);

                    // Direct load from source - no buffer copy!
                    nk_dots_bf16bf16f32_update2x2_direct_sapphire_amx(a + m_row * a_stride_elements + k_off,
                                                                      a + (m_row + 16) * a_stride_elements + k_off,
                                                                      a_stride, b0, b1);
                }

                // Handle partial K-tile (if any) with buffered load
                if (k_remainder > 0) {
                    nk_size_t const k_tile = num_full_k_tiles;
                    nk_size_t const k_off = k_tile * tile_k;

                    nk_dots_bf16bf16f32_load_a_sapphire_amx(&a0, a + m_row * a_stride_elements + k_off,
                                                            a_stride_elements, 16, k_remainder);
                    nk_dots_bf16bf16f32_load_a_sapphire_amx(&a1, a + (m_row + 16) * a_stride_elements + k_off,
                                                            a_stride_elements, 16, k_remainder);

                    nk_dots_bf16bf16f32_b32x16_sapphire_amx_t const *b0 =
                        (nk_dots_bf16bf16f32_b32x16_sapphire_amx_t const *)(b_tiles + (b_n0_base + k_tile) * tile_size);
                    nk_dots_bf16bf16f32_b32x16_sapphire_amx_t const *b1 =
                        (nk_dots_bf16bf16f32_b32x16_sapphire_amx_t const *)(b_tiles + (b_n1_base + k_tile) * tile_size);

                    nk_dots_bf16bf16f32_update2x2_sapphire_amx(&a0, &a1, b0, b1);
                }
            }
            // SLOW PATH: Edge M-block → always use buffered load with masking
            else {
                nk_size_t const rows0 = (rows_valid > 16) ? 16 : rows_valid;
                nk_size_t const rows1 = (rows_valid > 16) ? rows_valid - 16 : 0;

                for (nk_size_t k_tile = 0; k_tile < num_k_tiles; k_tile++) {
                    nk_size_t const k_off = k_tile * tile_k;
                    nk_size_t const k_valid = (k_tile < num_full_k_tiles) ? tile_k : k_remainder;

                    nk_dots_bf16bf16f32_load_a_sapphire_amx(&a0, a + m_row * a_stride_elements + k_off,
                                                            a_stride_elements, rows0, k_valid);
                    if (rows1 > 0) {
                        nk_dots_bf16bf16f32_load_a_sapphire_amx(&a1, a + (m_row + 16) * a_stride_elements + k_off,
                                                                a_stride_elements, rows1, k_valid);
                    }

                    nk_dots_bf16bf16f32_b32x16_sapphire_amx_t const *b0 =
                        (nk_dots_bf16bf16f32_b32x16_sapphire_amx_t const *)(b_tiles + (b_n0_base + k_tile) * tile_size);
                    nk_dots_bf16bf16f32_b32x16_sapphire_amx_t const *b1 =
                        (nk_dots_bf16bf16f32_b32x16_sapphire_amx_t const *)(b_tiles + (b_n1_base + k_tile) * tile_size);

                    nk_dots_bf16bf16f32_update2x2_sapphire_amx(&a0, &a1, b0, b1);
                }
            }

            // Store accumulators to output
            if (is_full_m_block) {
                // FAST PATH: Direct store to C - no intermediate buffer!
                nk_dots_bf16bf16f32_store2x2_direct_sapphire_amx(c + m_row * c_stride_elements + n_col, c_stride);
            }
            else {
                // SLOW PATH: Edge M-block needs masked output
                nk_dots_bf16bf16f32_store2x2_sapphire_amx(&c_state);
                nk_dots_bf16bf16f32_output2x2_sapphire_amx(&c_state, c + m_row * c_stride_elements + n_col,
                                                           c_stride_elements, rows_valid, 32);
            }
        }

        // Handle odd N-tile (single 16-column tile if num_n_tiles is odd)
        if (num_n_tiles % 2 == 1) {
            nk_size_t const n_tile = num_n_tiles - 1;
            nk_size_t const n_col = n_tile * 16;
            nk_size_t const b_n_base = n_tile * num_k_tiles;
            nk_size_t const rows0 = (rows_valid > 16) ? 16 : rows_valid;
            nk_size_t const rows1 = (rows_valid > 16) ? rows_valid - 16 : 0;

            // Use 1×1 blocking for single N-tile
            nk_dots_bf16bf16f32_state_sapphire_amx_t c0_state, c1_state;
            nk_dots_bf16bf16f32_init_sapphire_amx(&c0_state);
            nk_dots_bf16bf16f32_init_sapphire_amx(&c1_state);

            _tile_zero(4);
            _tile_zero(6);

            for (nk_size_t k_tile = 0; k_tile < num_k_tiles; k_tile++) {
                nk_size_t const k_off = k_tile * tile_k;
                nk_size_t const k_valid = (k_tile < num_full_k_tiles) ? tile_k : k_remainder;

                nk_dots_bf16bf16f32_load_a_sapphire_amx(&a0, a + m_row * a_stride_elements + k_off, a_stride_elements,
                                                        rows0, k_valid);
                if (rows1 > 0) {
                    nk_dots_bf16bf16f32_load_a_sapphire_amx(&a1, a + (m_row + 16) * a_stride_elements + k_off,
                                                            a_stride_elements, rows1, k_valid);
                }

                nk_dots_bf16bf16f32_b32x16_sapphire_amx_t const *b0 =
                    (nk_dots_bf16bf16f32_b32x16_sapphire_amx_t const *)(b_tiles + (b_n_base + k_tile) * tile_size);

                _tile_loadd(0, a0.data, 64);
                _tile_loadd(1, a1.data, 64);
                _tile_loadd(2, b0->data, 64);

                _tile_dpbf16ps(4, 0, 2);
                _tile_dpbf16ps(6, 1, 2);
            }

            _tile_stored(4, c0_state.data, 64);
            _tile_stored(6, c1_state.data, 64);

            nk_dots_bf16bf16f32_store_sapphire_amx(&c0_state, c + m_row * c_stride_elements + n_col, c_stride_elements,
                                                   rows0, 16);
            if (rows1 > 0) {
                nk_dots_bf16bf16f32_store_sapphire_amx(&c1_state, c + (m_row + 16) * c_stride_elements + n_col,
                                                       c_stride_elements, rows1, 16);
            }
        }
    }

    _tile_release();

    // AVX-512 fallback for N-edge rows (unpacked, row-major in b_packed)
    if (n_edge_rows > 0) {
        nk_dots_bf16bf16f32_avx512_edge_(a, n_edge_ptr, c + full_n, m, n_edge_rows, k, a_stride_elements, k,
                                         c_stride_elements);
    }
}

/*  BF16 compact: truncate F32 → BF16 in-place using AVX512.
 *  Reads F32 matrix, writes BF16 to same buffer (safe since F32 is larger).
 *  Uses masked loads/stores to handle all sizes without scalar fallback.
 *  Output is tightly packed with stride = n * sizeof(bf16).
 */
NK_PUBLIC void nk_dots_bf16bf16bf16_sapphire_amx( //
    void *c, nk_size_t m, nk_size_t n,            //
    nk_size_t c_stride) {

    nk_size_t const c_stride_f32 = c_stride / sizeof(nk_f32_t);
    nk_f32_t const *c_f32 = (nk_f32_t const *)c;
    nk_bf16_t *c_bf16 = (nk_bf16_t *)c;

    for (nk_size_t row = 0; row < m; row++) {
        nk_f32_t const *src_row = c_f32 + row * c_stride_f32;
        nk_bf16_t *dst_row = c_bf16 + row * n;
        nk_size_t col = 0;

        // Process 16 floats at a time using AVX512-BF16
        for (; col + 16 <= n; col += 16) {
            __m512 f32_vec = _mm512_loadu_ps(src_row + col);
            __m256bh bf16_vec = _mm512_cvtneps_pbh(f32_vec);
            _mm256_storeu_si256((__m256i *)(dst_row + col), (__m256i)bf16_vec);
        }

        // Handle remaining elements with masked operations
        if (col < n) {
            __mmask16 tail_mask = (__mmask16)((1u << (n - col)) - 1);
            __m512 f32_vec = _mm512_maskz_loadu_ps(tail_mask, src_row + col);
            __m256bh bf16_vec = _mm512_cvtneps_pbh(f32_vec);
            _mm256_mask_storeu_epi16(dst_row + col, tail_mask, (__m256i)bf16_vec);
        }
    }
}

/*  I8 → I32 matmul: C[m×n] = A[m×k] × B[n×k]ᵀ
 *
 *  Unified implementation using composable tile primitives.
 *  All I/O goes through aligned tile buffers for consistent behavior.
 *
 *  Uses register-resident 2×2 update pattern:
 *  - TMM4-7 hold C accumulators across entire K-loop (no redundant load/store)
 *  - 32×32 output blocks processed per M×N iteration
 *
 *  Single-threaded. For parallel execution, partition A rows across threads.
 */
NK_PUBLIC void nk_dots_i8i8i32_sapphire_amx(             //
    nk_i8_t const *a, void const *b_packed, nk_i32_t *c, //
    nk_size_t m, nk_size_t n, nk_size_t k,               //
    nk_size_t a_stride, nk_size_t c_stride) {

    // Parse packed B header
    nk_dots_amx_packed_header_t const *header = (nk_dots_amx_packed_header_t const *)b_packed;
    nk_size_t const num_n_tiles = header->full_n_tiles;
    nk_size_t const num_k_tiles = header->full_k_tiles;
    nk_size_t const n_edge_rows = header->n_edge_rows;

    // Packed B data regions
    nk_i8_t const *b_tiles = (nk_i8_t const *)((char const *)b_packed + sizeof(nk_dots_amx_packed_header_t));
    nk_i8_t const *n_edge_ptr = (nk_i8_t const *)((char const *)b_packed + header->n_edge_offset);

    // Stride conversions
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_i32_t);

    // Tile dimensions
    nk_size_t const tile_k = 64;      // K elements per INT8 tile
    nk_size_t const tile_size = 1024; // Bytes per packed tile
    nk_size_t const full_n = num_n_tiles * 16;

    // Block counts (32×32 output blocks = 2×2 tiles)
    nk_size_t const num_m_blocks = (m + 31) / 32; // Ceiling division
    nk_size_t const num_n_blocks = num_n_tiles / 2;

    if (num_k_tiles == 0) return;

    // Tile buffers for A (only used for edge tiles)
    nk_dots_i8i8i32_a16x64_sapphire_amx_t a0, a1;
    nk_dots_i8i8i32_state2x2_sapphire_amx_t c_state;

    // Precompute: number of full K-tiles (no masking needed)
    nk_size_t const num_full_k_tiles = k / tile_k;
    nk_size_t const k_remainder = k % tile_k;

    nk_amx_tile_configure_sapphire_amx_();

    // Process all 32×32 M×N blocks (including partial edge blocks)
    for (nk_size_t m_block = 0; m_block < num_m_blocks; m_block++) {
        nk_size_t const m_row = m_block * 32;
        nk_size_t const rows_valid = (m_row + 32 <= m) ? 32 : (m - m_row);
        nk_size_t const is_full_m_block = (rows_valid == 32);

        // Process full N-blocks (pairs of 16-column tiles = 32 columns)
        for (nk_size_t n_block = 0; n_block < num_n_blocks; n_block++) {
            nk_size_t const n_col = n_block * 32;

            // B tile base indices (linear layout: n_tile * num_k_tiles + k_tile)
            nk_size_t const b_n0_base = (n_block * 2) * num_k_tiles;
            nk_size_t const b_n1_base = (n_block * 2 + 1) * num_k_tiles;

            // Zero accumulators in registers
            nk_dots_i8i8i32_zero2x2_sapphire_amx();

            // FAST PATH: Full M-block with full K-tiles → direct A load
            if (is_full_m_block) {
                // Process full K-tiles with direct load
                for (nk_size_t k_tile = 0; k_tile < num_full_k_tiles; k_tile++) {
                    nk_size_t const k_off = k_tile * tile_k;

                    nk_dots_i8i8i32_b64x16_sapphire_amx_t const *b0 =
                        (nk_dots_i8i8i32_b64x16_sapphire_amx_t const *)(b_tiles + (b_n0_base + k_tile) * tile_size);
                    nk_dots_i8i8i32_b64x16_sapphire_amx_t const *b1 =
                        (nk_dots_i8i8i32_b64x16_sapphire_amx_t const *)(b_tiles + (b_n1_base + k_tile) * tile_size);

                    // Direct load from source - no buffer copy!
                    nk_dots_i8i8i32_update2x2_direct_sapphire_amx(
                        a + m_row * a_stride + k_off, a + (m_row + 16) * a_stride + k_off, a_stride, b0, b1);
                }

                // Handle partial K-tile (if any) with buffered load
                if (k_remainder > 0) {
                    nk_size_t const k_tile = num_full_k_tiles;
                    nk_size_t const k_off = k_tile * tile_k;

                    nk_dots_i8i8i32_load_a_sapphire_amx(&a0, a + m_row * a_stride + k_off, a_stride, 16, k_remainder);
                    nk_dots_i8i8i32_load_a_sapphire_amx(&a1, a + (m_row + 16) * a_stride + k_off, a_stride, 16,
                                                        k_remainder);

                    nk_dots_i8i8i32_b64x16_sapphire_amx_t const *b0 =
                        (nk_dots_i8i8i32_b64x16_sapphire_amx_t const *)(b_tiles + (b_n0_base + k_tile) * tile_size);
                    nk_dots_i8i8i32_b64x16_sapphire_amx_t const *b1 =
                        (nk_dots_i8i8i32_b64x16_sapphire_amx_t const *)(b_tiles + (b_n1_base + k_tile) * tile_size);

                    nk_dots_i8i8i32_update2x2_sapphire_amx(&a0, &a1, b0, b1);
                }
            }
            // SLOW PATH: Edge M-block → always use buffered load with masking
            else {
                nk_size_t const rows0 = (rows_valid > 16) ? 16 : rows_valid;
                nk_size_t const rows1 = (rows_valid > 16) ? rows_valid - 16 : 0;

                for (nk_size_t k_tile = 0; k_tile < num_k_tiles; k_tile++) {
                    nk_size_t const k_off = k_tile * tile_k;
                    nk_size_t const k_valid = (k_tile < num_full_k_tiles) ? tile_k : k_remainder;

                    nk_dots_i8i8i32_load_a_sapphire_amx(&a0, a + m_row * a_stride + k_off, a_stride, rows0, k_valid);
                    if (rows1 > 0) {
                        nk_dots_i8i8i32_load_a_sapphire_amx(&a1, a + (m_row + 16) * a_stride + k_off, a_stride, rows1,
                                                            k_valid);
                    }

                    nk_dots_i8i8i32_b64x16_sapphire_amx_t const *b0 =
                        (nk_dots_i8i8i32_b64x16_sapphire_amx_t const *)(b_tiles + (b_n0_base + k_tile) * tile_size);
                    nk_dots_i8i8i32_b64x16_sapphire_amx_t const *b1 =
                        (nk_dots_i8i8i32_b64x16_sapphire_amx_t const *)(b_tiles + (b_n1_base + k_tile) * tile_size);

                    nk_dots_i8i8i32_update2x2_sapphire_amx(&a0, &a1, b0, b1);
                }
            }

            // Store accumulators to output
            if (is_full_m_block) {
                // FAST PATH: Direct store to C - no intermediate buffer!
                nk_dots_i8i8i32_store2x2_direct_sapphire_amx(c + m_row * c_stride_elements + n_col, c_stride);
            }
            else {
                // SLOW PATH: Edge M-block needs masked output
                nk_dots_i8i8i32_store2x2_sapphire_amx(&c_state);
                nk_dots_i8i8i32_output2x2_sapphire_amx(&c_state, c + m_row * c_stride_elements + n_col,
                                                       c_stride_elements, rows_valid, 32);
            }
        }

        // Handle odd N-tile (single 16-column tile if num_n_tiles is odd)
        if (num_n_tiles % 2 == 1) {
            nk_size_t const n_tile = num_n_tiles - 1;
            nk_size_t const n_col = n_tile * 16;
            nk_size_t const b_n_base = n_tile * num_k_tiles;
            nk_size_t const rows0 = (rows_valid > 16) ? 16 : rows_valid;
            nk_size_t const rows1 = (rows_valid > 16) ? rows_valid - 16 : 0;

            // Use 1×1 blocking for single N-tile
            nk_dots_i8i8i32_state_sapphire_amx_t c0_state, c1_state;
            nk_dots_i8i8i32_init_sapphire_amx(&c0_state);
            nk_dots_i8i8i32_init_sapphire_amx(&c1_state);

            _tile_zero(4);
            _tile_zero(6);

            for (nk_size_t k_tile = 0; k_tile < num_k_tiles; k_tile++) {
                nk_size_t const k_off = k_tile * tile_k;
                nk_size_t const k_valid = (k_tile < num_full_k_tiles) ? tile_k : k_remainder;

                nk_dots_i8i8i32_load_a_sapphire_amx(&a0, a + m_row * a_stride + k_off, a_stride, rows0, k_valid);
                if (rows1 > 0) {
                    nk_dots_i8i8i32_load_a_sapphire_amx(&a1, a + (m_row + 16) * a_stride + k_off, a_stride, rows1,
                                                        k_valid);
                }

                nk_dots_i8i8i32_b64x16_sapphire_amx_t const *b0 =
                    (nk_dots_i8i8i32_b64x16_sapphire_amx_t const *)(b_tiles + (b_n_base + k_tile) * tile_size);

                _tile_loadd(0, a0.data, 64);
                _tile_loadd(1, a1.data, 64);
                _tile_loadd(2, b0->data, 64);

                _tile_dpbssd(4, 0, 2);
                _tile_dpbssd(6, 1, 2);
            }

            _tile_stored(4, c0_state.data, 64);
            _tile_stored(6, c1_state.data, 64);

            nk_dots_i8i8i32_store_sapphire_amx(&c0_state, c + m_row * c_stride_elements + n_col, c_stride_elements,
                                               rows0, 16);
            if (rows1 > 0) {
                nk_dots_i8i8i32_store_sapphire_amx(&c1_state, c + (m_row + 16) * c_stride_elements + n_col,
                                                   c_stride_elements, rows1, 16);
            }
        }
    }

    _tile_release();

    // AVX-512 fallback for N-edge rows (unpacked, row-major in b_packed)
    if (n_edge_rows > 0) {
        nk_dots_i8i8i32_avx512_edge_(a, n_edge_ptr, c + full_n, m, n_edge_rows, k, a_stride, k, c_stride_elements);
    }
}

/*  I8 compact: re-normalize I32 → I8 using precomputed squared norms.
 *  Formula: c_i8[i][j] = c_i32[i][j] * 127 * rsqrt(a_norm[i] * b_norm[j])
 *  Uses AVX512 rsqrt14 with Newton-Raphson refinement for 16 elements at a time.
 *  Output is tightly packed with stride = n * sizeof(i8).
 */
NK_PUBLIC void nk_dots_i8i8i8_sapphire_amx( //
    void *c, nk_size_t m, nk_size_t n,      //
    nk_size_t c_stride,                     //
    nk_i32_t const *a_squared_norms, nk_i32_t const *b_squared_norms) {

    nk_size_t const c_stride_i32 = c_stride / sizeof(nk_i32_t);
    nk_i32_t const *c_i32 = (nk_i32_t const *)c;
    nk_i8_t *c_i8 = (nk_i8_t *)c;

    // Use space after I8 output for precomputed b_rsqrt (I8 output is 4x smaller than I32 input)
    nk_f32_t *b_rsqrt = (nk_f32_t *)(c_i8 + m * n);

    // Precompute rsqrt of all b_norms using AVX512 (16 at a time)
    __m512 half_vec = _mm512_set1_ps(0.5f);
    __m512 three_halves_vec = _mm512_set1_ps(1.5f);
    nk_size_t j = 0;

    for (; j + 16 <= n; j += 16) {
        __m512i b_norms_i32 = _mm512_loadu_si512(b_squared_norms + j);
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
        _mm512_storeu_ps(b_rsqrt + j, rsqrt_vec);
    }

    // Handle remaining b_norms with masked operations
    if (j < n) {
        __mmask16 tail_mask = (__mmask16)((1u << (n - j)) - 1);
        __m512i b_norms_i32 = _mm512_maskz_loadu_epi32(tail_mask, b_squared_norms + j);
        __m512 b_norms_f32 = _mm512_cvtepi32_ps(b_norms_i32);
        __m512 rsqrt_vec = _mm512_rsqrt14_ps(b_norms_f32);
        rsqrt_vec = _mm512_mul_ps(
            rsqrt_vec,
            _mm512_sub_ps(three_halves_vec,
                          _mm512_mul_ps(half_vec, _mm512_mul_ps(b_norms_f32, _mm512_mul_ps(rsqrt_vec, rsqrt_vec)))));
        __mmask16 nonzero_mask = _mm512_cmpneq_epi32_mask(b_norms_i32, _mm512_setzero_si512());
        rsqrt_vec = _mm512_maskz_mov_ps(nonzero_mask & tail_mask, rsqrt_vec);
        _mm512_mask_storeu_ps(b_rsqrt + j, tail_mask, rsqrt_vec);
    }

    __m512 scale_vec = _mm512_set1_ps(127.0f);

    for (nk_size_t row = 0; row < m; row++) {
        nk_i32_t const *src_row = c_i32 + row * c_stride_i32;
        nk_i8_t *dst_row = c_i8 + row * n;

        // Compute rsqrt of a_norm for this row, broadcast to vector
        nk_f32_t a_norm_f32 = (nk_f32_t)a_squared_norms[row];
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

        nk_size_t col = 0;

        // Process 16 elements at a time
        for (; col + 16 <= n; col += 16) {
            __m512i c_vals = _mm512_loadu_si512(src_row + col);
            __m512 c_f32 = _mm512_cvtepi32_ps(c_vals);
            __m512 b_rsqrt_vec = _mm512_loadu_ps(b_rsqrt + col);
            __m512 normalized = _mm512_mul_ps(_mm512_mul_ps(c_f32, row_scale), b_rsqrt_vec);
            __m512i result_i32 = _mm512_cvtps_epi32(normalized);
            // Saturating pack I32 → I8 (16 values → 16 bytes in low 128 bits)
            __m128i result_i8 = _mm512_cvtsepi32_epi8(result_i32);
            _mm_storeu_si128((__m128i *)(dst_row + col), result_i8);
        }

        // Handle remaining elements with masked operations
        if (col < n) {
            __mmask16 tail_mask = (__mmask16)((1u << (n - col)) - 1);
            __m512i c_vals = _mm512_maskz_loadu_epi32(tail_mask, src_row + col);
            __m512 c_f32 = _mm512_cvtepi32_ps(c_vals);
            __m512 b_rsqrt_vec = _mm512_maskz_loadu_ps(tail_mask, b_rsqrt + col);
            __m512 normalized = _mm512_mul_ps(_mm512_mul_ps(c_f32, row_scale), b_rsqrt_vec);
            __m512i result_i32 = _mm512_cvtps_epi32(normalized);
            __m128i result_i8 = _mm512_cvtsepi32_epi8(result_i32);
            _mm_mask_storeu_epi8(dst_row + col, tail_mask, result_i8);
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

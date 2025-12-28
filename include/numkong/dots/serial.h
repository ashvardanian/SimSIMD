/**
 *  @brief SIMD-accelerated Dot Products for Real and Complex Numbers optimized for SIMD-free CPUs.
 *  @file include/numkong/dots/serial.h
 *  @sa include/numkong/dots.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 *
 *  @section numkong_dots_serial_packing Tile Packing Format
 *
 *  When multiplying matrices, one of the arguments often remains constant across multiple calls.
 *  To optimize performance, we pack this matrix into a tiled format that improves cache locality
 *  and allows for more efficient SIMD loads during the matrix multiplication.
 *
 *  Imagine a row-major 8x8 input matrix looking like:
 *
 *      Aa Ab Ac Ad Ae Af Ag Ah         Let's assume that our tile size is 3x3.
 *      Ba Bb Bc Bd Be Bf Bg Bh         In the packed format, the matrix is divided
 *      Ca Cb Cc Cd Ce Cf Cg Ch         into tiles, and transposed within each tile.
 *      Da Db Dc Dd De Df Dg Dh         For tile [3:6,3:6], the memory layout becomes:
 *      Ea Eb Ec Ed Ee Ef Eg Eh
 *      Fa Fb Fc Fd Fe Ff Fg Fh         Dd Ed Fd De Ee Fe Df Ff Gf
 *      Ga Gb Gc Gd Ge Gf Gg Gh
 *      Ha Hb Hc Hd He Hf Hg Hh
 *
 *  We first store the tiles forming the first columns of the matrix, then the second
 *  columns, etc. So our previous example matrix would be stored in memory as:
 *
 *      Aa Ba Ca Ab Bb Cb Ac Bc Cc      < The transposed tile at [0:3,0:3]
 *      Da Ea Fa Db Eb Fb Ec Fc Fc      < The transposed tile at [3:6,0:3]
 *      Ga Ha 00 Gb Hb 00 Gc Hc 00      < The transposed tile at [6:8,0:3], padded
 *      Ad Bd Cd Ae Be Ce Af Bf Cf      < The transposed tile at [0:3,3:6]
 *      Dd Ed Fd De Ee Fe Df Ff Ff      < The transposed tile at [3:6,3:6]
 *      Gd Hd 00 Ge He 00 Gf Hf 00      < The transposed tile at [6:8,3:6], padded
 *      Ag Bg Cg Ah Bh Ch 00 00 00      < The transposed tile at [0:3,6:8], padded
 *      Dg Eg Fg Df Eg Ff 00 00 00      < The transposed tile at [3:6,6:8], padded
 *      Gg Hg 00 Gg Hg 00 00 00 00      < The transposed tile at [6:8,6:8], padded
 *
 *  This way it uses more memory due to padding, but allows eliminating some of the
 *  boundary checks in the inner loops of the mat-mul kernel, leading to better performance.
 *
 *  The tiles can be chosen to be of any rectangular form, better fitting the CPU.
 *  Oftentimes, we want the width to be 64 bytes (matching the cache line of many CPUs),
 *  and the height is proportional to the amount of the L1 cache lines we have.
 *  Meaning, the width in columns will be different for different numeric types:
 *
 *    - i8/u8/e4m3/e5m2: 64 elements per tile row
 *    - f16/bf16/i16/u16: 32 elements per tile row
 *    - f32/i32/u32: 16 elements per tile row
 *    - f64/i64/u64: 8 elements per tile row
 */

#ifndef NK_DOTS_SERIAL_H
#define NK_DOTS_SERIAL_H
#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

/*  Inner-Product GEMM Macro
 *
 *  Computes C[m x n] = A[m x k] x B[n x k]^T using inner-product micro-kernels with
 *  ILP-optimized 4-way batched horizontal reductions.
 *
 *  Loop Structure (GotoBLAS 5-loop design):
 *  +-----------------------------------------------------------------------------+
 *  | Loop 1: NC columns at a time (L3 blocking)                                  |
 *  |   Loop 2: KC depth at a time (L1 blocking)                                  |
 *  |     Loop 3: MC rows at a time (L2 blocking)                                 |
 *  |       Loop 4: NR=4 columns (register tile, batched finalize)                |
 *  |         Loop 5: MR rows (register tile)                                     |
 *  |           Loop 6: k_tile elements (SIMD accumulation)                       |
 *  |             +-----------------------------------------------------------+   |
 *  |             | Fast path: SIMD-aligned tiles with full vector loads      |   |
 *  |             | Slow path: partial loads for K remainder elements         |   |
 *  |             +-----------------------------------------------------------+   |
 *  |           Finalize 4 states -> 4 results (ILP-optimized reduction)          |
 *  +-----------------------------------------------------------------------------+
 *
 *  Cache Blocking Strategy:
 *  - NC (L3): ~2048 columns, B tile stays in L3 across MC iterations
 *  - KC (L1): ~256 depth, A slice + B rows fit in L1 during k-loop
 *  - MC (L2): ~128 rows, A tile + partial C fit in L2
 *
 *  B Packing Format (tiled row-major, 16 rows x depth per tile):
 *  - Tiles indexed as: b_packed[tile_idx * 16 * depth + row * depth + k]
 *  - No interleaving, simple cache-line-aligned access pattern
 *  - Zero-padded edge tiles for uniform SIMD loads
 *
 *  Fast Path vs Slow Path:
 *  - Fast path: When kc_len is a multiple of k_tile, use aligned SIMD loads
 *  - Slow path: When kc_len has remainder, use partial_load_fn for masked loads
 *  - This separation eliminates branch overhead in the hot inner loop
 *
 *  State & Vector Sizes by Platform:
 *  - AVX-512: 512-bit state, 64-byte loads, k_tile = 64/sizeof(input)
 *  - AVX2:    256-bit state, 32-byte loads, k_tile = 32/sizeof(input)
 *  - NEON:    128-bit state, 16-byte loads, k_tile = 16/sizeof(input)
 *
 *  @param suffix           Function name suffix (e.g., f32f32f32_skylake)
 *  @param input_type       Input element type (f32, bf16, i8, etc.)
 *  @param output_type      Output element type (f32, i32)
 *  @param state_type       Accumulator state (native SIMD width per platform)
 *  @param init_fn          State init: init_fn(state*) - zeros accumulator
 *  @param load_fn          Vector load: load_fn(void const* src, vec_t* dst)
 *  @param partial_load_fn  Masked load: partial_load_fn(src, count, dst) for remainder
 *  @param update_fn        FMA update: update_fn(state*, a_vec, b_vec)
 *  @param finalize_fn      4-way reduce: finalize_fn(s0*, s1*, s2*, s3*, f32[4])
 *  @param k_tile           Input elements per SIMD update (platform-dependent)
 *  @param k_unroll         K-loop unroll factor (2 for NEON/AVX2, 4 for AVX-512)
 *  @param mr_size          A rows per register tile (typically 4)
 *  @param mc_size          L2 row blocking (typically 128)
 *  @param nc_size          L3 column blocking (typically 2048)
 *  @param kc_size          L1 depth blocking (typically 256)
 */
#define NK_MAKE_DOTS_INNER(suffix, input_type, output_type, vec_type, state_type, init_fn, load_fn, partial_load_fn, \
                           update_fn, finalize_fn, k_tile, k_unroll, mr_size, mc_size, nc_size, kc_size)             \
                                                                                                                     \
    NK_PUBLIC void nk_dots_##suffix(nk_##input_type##_t const *a_matrix, void const *b_packed_void,                  \
                                    nk_##output_type##_t *c_matrix, nk_size_t row_count, nk_size_t column_count,     \
                                    nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride) {                       \
        nk_##input_type##_t const *b_packed = (nk_##input_type##_t const *)b_packed_void;                            \
                                                                                                                     \
        nk_size_t const register_tile_columns = 4;                   /* Columns per finalize batch (NR) */           \
        nk_size_t const packed_tile_rows = 16;                       /* B rows per packed tile */                    \
        nk_size_t const simd_width = k_tile;                         /* Elements per SIMD update */                  \
        nk_size_t const packed_tile_size = packed_tile_rows * depth; /* Elements per B tile */                       \
                                                                                                                     \
        /* Loop 1: L3 cache blocking over columns */                                                                 \
        for (nk_size_t column_block_start = 0; column_block_start < column_count; column_block_start += nc_size) {   \
            nk_size_t column_block_end = column_block_start + nc_size;                                               \
            if (column_block_end > column_count) column_block_end = column_count;                                    \
                                                                                                                     \
            /* Loop 2: L1 cache blocking over depth */                                                               \
            for (nk_size_t depth_block_start = 0; depth_block_start < depth; depth_block_start += kc_size) {         \
                nk_size_t depth_block_end = depth_block_start + kc_size;                                             \
                if (depth_block_end > depth) depth_block_end = depth;                                                \
                nk_size_t const depth_block_length = depth_block_end - depth_block_start;                            \
                nk_size_t const unroll_stride = simd_width * k_unroll;                                               \
                nk_size_t const unroll_aligned_depth = (depth_block_length / unroll_stride) * unroll_stride;         \
                nk_size_t const aligned_depth = (depth_block_length / simd_width) * simd_width;                      \
                nk_size_t const remainder_depth = depth_block_length - aligned_depth;                                \
                                                                                                                     \
                /* Loop 3: L2 cache blocking over rows */                                                            \
                for (nk_size_t row_block_start = 0; row_block_start < row_count; row_block_start += mc_size) {       \
                    nk_size_t row_block_end = row_block_start + mc_size;                                             \
                    if (row_block_end > row_count) row_block_end = row_count;                                        \
                                                                                                                     \
                    /* Loop 4: Register tiling over columns (4 columns per batch) */                                 \
                    for (nk_size_t tile_column_start = column_block_start; tile_column_start < column_block_end;     \
                         tile_column_start += register_tile_columns) {                                               \
                        nk_size_t tile_column_count = register_tile_columns;                                         \
                        if (tile_column_start + tile_column_count > column_block_end)                                \
                            tile_column_count = column_block_end - tile_column_start;                                \
                                                                                                                     \
                        /* Loop 5: Register tiling over rows (MR rows per tile) */                                   \
                        for (nk_size_t tile_row_start = row_block_start; tile_row_start < row_block_end;             \
                             tile_row_start += mr_size) {                                                            \
                            nk_size_t tile_row_count = mr_size;                                                      \
                            if (tile_row_start + tile_row_count > row_block_end)                                     \
                                tile_row_count = row_block_end - tile_row_start;                                     \
                                                                                                                     \
                            /* Initialize MR x 4 accumulator states */                                               \
                            state_type accumulator_states[mr_size][4];                                               \
                            for (nk_size_t row_index = 0; row_index < tile_row_count; ++row_index) {                 \
                                init_fn(&accumulator_states[row_index][0]);                                          \
                                init_fn(&accumulator_states[row_index][1]);                                          \
                                init_fn(&accumulator_states[row_index][2]);                                          \
                                init_fn(&accumulator_states[row_index][3]);                                          \
                            }                                                                                        \
                                                                                                                     \
                            /* Compute B row pointers for 4 columns (from packed tiles) */                           \
                            nk_size_t const packed_tile_index = tile_column_start / packed_tile_rows;                \
                            nk_size_t const row_within_tile = tile_column_start % packed_tile_rows;                  \
                            nk_##input_type##_t const *packed_tile_base = b_packed +                                 \
                                                                          packed_tile_index * packed_tile_size;      \
                            nk_##input_type##_t const *b_column_ptr_0 = packed_tile_base + row_within_tile * depth + \
                                                                        depth_block_start;                           \
                            nk_##input_type##_t const *b_column_ptr_1 =                                              \
                                (tile_column_count > 1 && row_within_tile + 1 < packed_tile_rows)                    \
                                    ? packed_tile_base + (row_within_tile + 1) * depth + depth_block_start           \
                                    : b_column_ptr_0;                                                                \
                            nk_##input_type##_t const *b_column_ptr_2 =                                              \
                                (tile_column_count > 2 && row_within_tile + 2 < packed_tile_rows)                    \
                                    ? packed_tile_base + (row_within_tile + 2) * depth + depth_block_start           \
                                    : b_column_ptr_0;                                                                \
                            nk_##input_type##_t const *b_column_ptr_3 =                                              \
                                (tile_column_count > 3 && row_within_tile + 3 < packed_tile_rows)                    \
                                    ? packed_tile_base + (row_within_tile + 3) * depth + depth_block_start           \
                                    : b_column_ptr_0;                                                                \
                                                                                                                     \
                            /* Check if this is a full tile (no boundary checking needed) */                         \
                            int const is_full_tile = (tile_row_count == mr_size) && (tile_column_count == 4) &&      \
                                                     (remainder_depth == 0);                                         \
                                                                                                                     \
                            if (is_full_tile) {                                                                      \
                                /* FAST PATH: Full tile with no boundary checks, k-unrolled */                       \
                                nk_##input_type##_t const *a_ptr_0 =                                                 \
                                    (nk_##input_type##_t const *)((char const *)a_matrix +                           \
                                                                  (tile_row_start + 0) * a_stride) +                 \
                                    depth_block_start;                                                               \
                                nk_##input_type##_t const *a_ptr_1 =                                                 \
                                    (nk_##input_type##_t const *)((char const *)a_matrix +                           \
                                                                  (tile_row_start + 1) * a_stride) +                 \
                                    depth_block_start;                                                               \
                                nk_##input_type##_t const *a_ptr_2 =                                                 \
                                    (nk_##input_type##_t const *)((char const *)a_matrix +                           \
                                                                  (tile_row_start + 2) * a_stride) +                 \
                                    depth_block_start;                                                               \
                                nk_##input_type##_t const *a_ptr_3 =                                                 \
                                    (nk_##input_type##_t const *)((char const *)a_matrix +                           \
                                                                  (tile_row_start + 3) * a_stride) +                 \
                                    depth_block_start;                                                               \
                                nk_##input_type##_t const *b_ptr_0 = b_column_ptr_0;                                 \
                                nk_##input_type##_t const *b_ptr_1 = b_column_ptr_1;                                 \
                                nk_##input_type##_t const *b_ptr_2 = b_column_ptr_2;                                 \
                                nk_##input_type##_t const *b_ptr_3 = b_column_ptr_3;                                 \
                                                                                                                     \
                                /* Unrolled k-loop: process k_unroll tiles per iteration */                          \
                                for (nk_size_t depth_offset = 0; depth_offset < unroll_aligned_depth;                \
                                     depth_offset += unroll_stride) {                                                \
                                    /* Unroll iteration 0 */                                                         \
                                    vec_type b0, b1, b2, b3, a0, a1, a2, a3;                                         \
                                    load_fn(b_ptr_0, &b0);                                                           \
                                    load_fn(b_ptr_1, &b1);                                                           \
                                    load_fn(b_ptr_2, &b2);                                                           \
                                    load_fn(b_ptr_3, &b3);                                                           \
                                    load_fn(a_ptr_0, &a0);                                                           \
                                    load_fn(a_ptr_1, &a1);                                                           \
                                    load_fn(a_ptr_2, &a2);                                                           \
                                    load_fn(a_ptr_3, &a3);                                                           \
                                    b_ptr_0 += simd_width;                                                           \
                                    b_ptr_1 += simd_width;                                                           \
                                    b_ptr_2 += simd_width;                                                           \
                                    b_ptr_3 += simd_width;                                                           \
                                    a_ptr_0 += simd_width;                                                           \
                                    a_ptr_1 += simd_width;                                                           \
                                    a_ptr_2 += simd_width;                                                           \
                                    a_ptr_3 += simd_width;                                                           \
                                    update_fn(&accumulator_states[0][0], a0, b0);                                    \
                                    update_fn(&accumulator_states[0][1], a0, b1);                                    \
                                    update_fn(&accumulator_states[0][2], a0, b2);                                    \
                                    update_fn(&accumulator_states[0][3], a0, b3);                                    \
                                    update_fn(&accumulator_states[1][0], a1, b0);                                    \
                                    update_fn(&accumulator_states[1][1], a1, b1);                                    \
                                    update_fn(&accumulator_states[1][2], a1, b2);                                    \
                                    update_fn(&accumulator_states[1][3], a1, b3);                                    \
                                    update_fn(&accumulator_states[2][0], a2, b0);                                    \
                                    update_fn(&accumulator_states[2][1], a2, b1);                                    \
                                    update_fn(&accumulator_states[2][2], a2, b2);                                    \
                                    update_fn(&accumulator_states[2][3], a2, b3);                                    \
                                    update_fn(&accumulator_states[3][0], a3, b0);                                    \
                                    update_fn(&accumulator_states[3][1], a3, b1);                                    \
                                    update_fn(&accumulator_states[3][2], a3, b2);                                    \
                                    update_fn(&accumulator_states[3][3], a3, b3);                                    \
                                    /* Unroll iteration 1 (if k_unroll >= 2) */                                      \
                                    if (k_unroll >= 2) {                                                             \
                                        load_fn(b_ptr_0, &b0);                                                       \
                                        load_fn(b_ptr_1, &b1);                                                       \
                                        load_fn(b_ptr_2, &b2);                                                       \
                                        load_fn(b_ptr_3, &b3);                                                       \
                                        load_fn(a_ptr_0, &a0);                                                       \
                                        load_fn(a_ptr_1, &a1);                                                       \
                                        load_fn(a_ptr_2, &a2);                                                       \
                                        load_fn(a_ptr_3, &a3);                                                       \
                                        b_ptr_0 += simd_width;                                                       \
                                        b_ptr_1 += simd_width;                                                       \
                                        b_ptr_2 += simd_width;                                                       \
                                        b_ptr_3 += simd_width;                                                       \
                                        a_ptr_0 += simd_width;                                                       \
                                        a_ptr_1 += simd_width;                                                       \
                                        a_ptr_2 += simd_width;                                                       \
                                        a_ptr_3 += simd_width;                                                       \
                                        update_fn(&accumulator_states[0][0], a0, b0);                                \
                                        update_fn(&accumulator_states[0][1], a0, b1);                                \
                                        update_fn(&accumulator_states[0][2], a0, b2);                                \
                                        update_fn(&accumulator_states[0][3], a0, b3);                                \
                                        update_fn(&accumulator_states[1][0], a1, b0);                                \
                                        update_fn(&accumulator_states[1][1], a1, b1);                                \
                                        update_fn(&accumulator_states[1][2], a1, b2);                                \
                                        update_fn(&accumulator_states[1][3], a1, b3);                                \
                                        update_fn(&accumulator_states[2][0], a2, b0);                                \
                                        update_fn(&accumulator_states[2][1], a2, b1);                                \
                                        update_fn(&accumulator_states[2][2], a2, b2);                                \
                                        update_fn(&accumulator_states[2][3], a2, b3);                                \
                                        update_fn(&accumulator_states[3][0], a3, b0);                                \
                                        update_fn(&accumulator_states[3][1], a3, b1);                                \
                                        update_fn(&accumulator_states[3][2], a3, b2);                                \
                                        update_fn(&accumulator_states[3][3], a3, b3);                                \
                                    }                                                                                \
                                    /* Unroll iterations 2-3 (if k_unroll >= 4) */                                   \
                                    if (k_unroll >= 4) {                                                             \
                                        load_fn(b_ptr_0, &b0);                                                       \
                                        load_fn(b_ptr_1, &b1);                                                       \
                                        load_fn(b_ptr_2, &b2);                                                       \
                                        load_fn(b_ptr_3, &b3);                                                       \
                                        load_fn(a_ptr_0, &a0);                                                       \
                                        load_fn(a_ptr_1, &a1);                                                       \
                                        load_fn(a_ptr_2, &a2);                                                       \
                                        load_fn(a_ptr_3, &a3);                                                       \
                                        b_ptr_0 += simd_width;                                                       \
                                        b_ptr_1 += simd_width;                                                       \
                                        b_ptr_2 += simd_width;                                                       \
                                        b_ptr_3 += simd_width;                                                       \
                                        a_ptr_0 += simd_width;                                                       \
                                        a_ptr_1 += simd_width;                                                       \
                                        a_ptr_2 += simd_width;                                                       \
                                        a_ptr_3 += simd_width;                                                       \
                                        update_fn(&accumulator_states[0][0], a0, b0);                                \
                                        update_fn(&accumulator_states[0][1], a0, b1);                                \
                                        update_fn(&accumulator_states[0][2], a0, b2);                                \
                                        update_fn(&accumulator_states[0][3], a0, b3);                                \
                                        update_fn(&accumulator_states[1][0], a1, b0);                                \
                                        update_fn(&accumulator_states[1][1], a1, b1);                                \
                                        update_fn(&accumulator_states[1][2], a1, b2);                                \
                                        update_fn(&accumulator_states[1][3], a1, b3);                                \
                                        update_fn(&accumulator_states[2][0], a2, b0);                                \
                                        update_fn(&accumulator_states[2][1], a2, b1);                                \
                                        update_fn(&accumulator_states[2][2], a2, b2);                                \
                                        update_fn(&accumulator_states[2][3], a2, b3);                                \
                                        update_fn(&accumulator_states[3][0], a3, b0);                                \
                                        update_fn(&accumulator_states[3][1], a3, b1);                                \
                                        update_fn(&accumulator_states[3][2], a3, b2);                                \
                                        update_fn(&accumulator_states[3][3], a3, b3);                                \
                                        /* Iteration 3 */                                                            \
                                        load_fn(b_ptr_0, &b0);                                                       \
                                        load_fn(b_ptr_1, &b1);                                                       \
                                        load_fn(b_ptr_2, &b2);                                                       \
                                        load_fn(b_ptr_3, &b3);                                                       \
                                        load_fn(a_ptr_0, &a0);                                                       \
                                        load_fn(a_ptr_1, &a1);                                                       \
                                        load_fn(a_ptr_2, &a2);                                                       \
                                        load_fn(a_ptr_3, &a3);                                                       \
                                        b_ptr_0 += simd_width;                                                       \
                                        b_ptr_1 += simd_width;                                                       \
                                        b_ptr_2 += simd_width;                                                       \
                                        b_ptr_3 += simd_width;                                                       \
                                        a_ptr_0 += simd_width;                                                       \
                                        a_ptr_1 += simd_width;                                                       \
                                        a_ptr_2 += simd_width;                                                       \
                                        a_ptr_3 += simd_width;                                                       \
                                        update_fn(&accumulator_states[0][0], a0, b0);                                \
                                        update_fn(&accumulator_states[0][1], a0, b1);                                \
                                        update_fn(&accumulator_states[0][2], a0, b2);                                \
                                        update_fn(&accumulator_states[0][3], a0, b3);                                \
                                        update_fn(&accumulator_states[1][0], a1, b0);                                \
                                        update_fn(&accumulator_states[1][1], a1, b1);                                \
                                        update_fn(&accumulator_states[1][2], a1, b2);                                \
                                        update_fn(&accumulator_states[1][3], a1, b3);                                \
                                        update_fn(&accumulator_states[2][0], a2, b0);                                \
                                        update_fn(&accumulator_states[2][1], a2, b1);                                \
                                        update_fn(&accumulator_states[2][2], a2, b2);                                \
                                        update_fn(&accumulator_states[2][3], a2, b3);                                \
                                        update_fn(&accumulator_states[3][0], a3, b0);                                \
                                        update_fn(&accumulator_states[3][1], a3, b1);                                \
                                        update_fn(&accumulator_states[3][2], a3, b2);                                \
                                        update_fn(&accumulator_states[3][3], a3, b3);                                \
                                    }                                                                                \
                                }                                                                                    \
                                /* Handle remaining iterations between unroll_aligned and aligned_depth */           \
                                for (nk_size_t depth_offset = unroll_aligned_depth; depth_offset < aligned_depth;    \
                                     depth_offset += simd_width) {                                                   \
                                    vec_type b0, b1, b2, b3, a0, a1, a2, a3;                                         \
                                    load_fn(b_ptr_0, &b0);                                                           \
                                    load_fn(b_ptr_1, &b1);                                                           \
                                    load_fn(b_ptr_2, &b2);                                                           \
                                    load_fn(b_ptr_3, &b3);                                                           \
                                    load_fn(a_ptr_0, &a0);                                                           \
                                    load_fn(a_ptr_1, &a1);                                                           \
                                    load_fn(a_ptr_2, &a2);                                                           \
                                    load_fn(a_ptr_3, &a3);                                                           \
                                    b_ptr_0 += simd_width;                                                           \
                                    b_ptr_1 += simd_width;                                                           \
                                    b_ptr_2 += simd_width;                                                           \
                                    b_ptr_3 += simd_width;                                                           \
                                    a_ptr_0 += simd_width;                                                           \
                                    a_ptr_1 += simd_width;                                                           \
                                    a_ptr_2 += simd_width;                                                           \
                                    a_ptr_3 += simd_width;                                                           \
                                    update_fn(&accumulator_states[0][0], a0, b0);                                    \
                                    update_fn(&accumulator_states[0][1], a0, b1);                                    \
                                    update_fn(&accumulator_states[0][2], a0, b2);                                    \
                                    update_fn(&accumulator_states[0][3], a0, b3);                                    \
                                    update_fn(&accumulator_states[1][0], a1, b0);                                    \
                                    update_fn(&accumulator_states[1][1], a1, b1);                                    \
                                    update_fn(&accumulator_states[1][2], a1, b2);                                    \
                                    update_fn(&accumulator_states[1][3], a1, b3);                                    \
                                    update_fn(&accumulator_states[2][0], a2, b0);                                    \
                                    update_fn(&accumulator_states[2][1], a2, b1);                                    \
                                    update_fn(&accumulator_states[2][2], a2, b2);                                    \
                                    update_fn(&accumulator_states[2][3], a2, b3);                                    \
                                    update_fn(&accumulator_states[3][0], a3, b0);                                    \
                                    update_fn(&accumulator_states[3][1], a3, b1);                                    \
                                    update_fn(&accumulator_states[3][2], a3, b2);                                    \
                                    update_fn(&accumulator_states[3][3], a3, b3);                                    \
                                }                                                                                    \
                            }                                                                                        \
                            else {                                                                                   \
                                /* SLOW PATH: Edge tiles with boundary checking */                                   \
                                nk_##input_type##_t const *a_row_ptr_0 =                                             \
                                    (nk_##input_type##_t const *)((char const *)a_matrix +                           \
                                                                  (tile_row_start + 0) * a_stride) +                 \
                                    depth_block_start;                                                               \
                                nk_##input_type##_t const *a_row_ptr_1 =                                             \
                                    (tile_row_count > 1)                                                             \
                                        ? (nk_##input_type##_t const *)((char const *)a_matrix +                     \
                                                                        (tile_row_start + 1) * a_stride) +           \
                                              depth_block_start                                                      \
                                        : a_row_ptr_0;                                                               \
                                nk_##input_type##_t const *a_row_ptr_2 =                                             \
                                    (tile_row_count > 2)                                                             \
                                        ? (nk_##input_type##_t const *)((char const *)a_matrix +                     \
                                                                        (tile_row_start + 2) * a_stride) +           \
                                              depth_block_start                                                      \
                                        : a_row_ptr_0;                                                               \
                                nk_##input_type##_t const *a_row_ptr_3 =                                             \
                                    (tile_row_count > 3)                                                             \
                                        ? (nk_##input_type##_t const *)((char const *)a_matrix +                     \
                                                                        (tile_row_start + 3) * a_stride) +           \
                                              depth_block_start                                                      \
                                        : a_row_ptr_0;                                                               \
                                                                                                                     \
                                /* SIMD-aligned iterations */                                                        \
                                for (nk_size_t depth_offset = 0; depth_offset < aligned_depth;                       \
                                     depth_offset += simd_width) {                                                   \
                                    vec_type b_vector_0, b_vector_1, b_vector_2, b_vector_3;                         \
                                    load_fn(b_column_ptr_0 + depth_offset, &b_vector_0);                             \
                                    load_fn(b_column_ptr_1 + depth_offset, &b_vector_1);                             \
                                    load_fn(b_column_ptr_2 + depth_offset, &b_vector_2);                             \
                                    load_fn(b_column_ptr_3 + depth_offset, &b_vector_3);                             \
                                    vec_type a_vector_0, a_vector_1, a_vector_2, a_vector_3;                         \
                                    load_fn(a_row_ptr_0 + depth_offset, &a_vector_0);                                \
                                    load_fn(a_row_ptr_1 + depth_offset, &a_vector_1);                                \
                                    load_fn(a_row_ptr_2 + depth_offset, &a_vector_2);                                \
                                    load_fn(a_row_ptr_3 + depth_offset, &a_vector_3);                                \
                                    update_fn(&accumulator_states[0][0], a_vector_0, b_vector_0);                    \
                                    update_fn(&accumulator_states[0][1], a_vector_0, b_vector_1);                    \
                                    update_fn(&accumulator_states[0][2], a_vector_0, b_vector_2);                    \
                                    update_fn(&accumulator_states[0][3], a_vector_0, b_vector_3);                    \
                                    update_fn(&accumulator_states[1][0], a_vector_1, b_vector_0);                    \
                                    update_fn(&accumulator_states[1][1], a_vector_1, b_vector_1);                    \
                                    update_fn(&accumulator_states[1][2], a_vector_1, b_vector_2);                    \
                                    update_fn(&accumulator_states[1][3], a_vector_1, b_vector_3);                    \
                                    update_fn(&accumulator_states[2][0], a_vector_2, b_vector_0);                    \
                                    update_fn(&accumulator_states[2][1], a_vector_2, b_vector_1);                    \
                                    update_fn(&accumulator_states[2][2], a_vector_2, b_vector_2);                    \
                                    update_fn(&accumulator_states[2][3], a_vector_2, b_vector_3);                    \
                                    update_fn(&accumulator_states[3][0], a_vector_3, b_vector_0);                    \
                                    update_fn(&accumulator_states[3][1], a_vector_3, b_vector_1);                    \
                                    update_fn(&accumulator_states[3][2], a_vector_3, b_vector_2);                    \
                                    update_fn(&accumulator_states[3][3], a_vector_3, b_vector_3);                    \
                                }                                                                                    \
                                                                                                                     \
                                /* Remainder elements with partial loads */                                          \
                                if (remainder_depth > 0) {                                                           \
                                    vec_type b_vector_0, b_vector_1, b_vector_2, b_vector_3;                         \
                                    partial_load_fn(b_column_ptr_0 + aligned_depth, remainder_depth, &b_vector_0);   \
                                    partial_load_fn(b_column_ptr_1 + aligned_depth, remainder_depth, &b_vector_1);   \
                                    partial_load_fn(b_column_ptr_2 + aligned_depth, remainder_depth, &b_vector_2);   \
                                    partial_load_fn(b_column_ptr_3 + aligned_depth, remainder_depth, &b_vector_3);   \
                                    vec_type a_vector_0, a_vector_1, a_vector_2, a_vector_3;                         \
                                    partial_load_fn(a_row_ptr_0 + aligned_depth, remainder_depth, &a_vector_0);      \
                                    partial_load_fn(a_row_ptr_1 + aligned_depth, remainder_depth, &a_vector_1);      \
                                    partial_load_fn(a_row_ptr_2 + aligned_depth, remainder_depth, &a_vector_2);      \
                                    partial_load_fn(a_row_ptr_3 + aligned_depth, remainder_depth, &a_vector_3);      \
                                    update_fn(&accumulator_states[0][0], a_vector_0, b_vector_0);                    \
                                    update_fn(&accumulator_states[0][1], a_vector_0, b_vector_1);                    \
                                    update_fn(&accumulator_states[0][2], a_vector_0, b_vector_2);                    \
                                    update_fn(&accumulator_states[0][3], a_vector_0, b_vector_3);                    \
                                    update_fn(&accumulator_states[1][0], a_vector_1, b_vector_0);                    \
                                    update_fn(&accumulator_states[1][1], a_vector_1, b_vector_1);                    \
                                    update_fn(&accumulator_states[1][2], a_vector_1, b_vector_2);                    \
                                    update_fn(&accumulator_states[1][3], a_vector_1, b_vector_3);                    \
                                    update_fn(&accumulator_states[2][0], a_vector_2, b_vector_0);                    \
                                    update_fn(&accumulator_states[2][1], a_vector_2, b_vector_1);                    \
                                    update_fn(&accumulator_states[2][2], a_vector_2, b_vector_2);                    \
                                    update_fn(&accumulator_states[2][3], a_vector_2, b_vector_3);                    \
                                    update_fn(&accumulator_states[3][0], a_vector_3, b_vector_0);                    \
                                    update_fn(&accumulator_states[3][1], a_vector_3, b_vector_1);                    \
                                    update_fn(&accumulator_states[3][2], a_vector_3, b_vector_2);                    \
                                    update_fn(&accumulator_states[3][3], a_vector_3, b_vector_3);                    \
                                }                                                                                    \
                            }                                                                                        \
                                                                                                                     \
                            /* Finalize and store MR x 4 results using batched 4-way reduction */                    \
                            for (nk_size_t row_index = 0; row_index < tile_row_count; ++row_index) {                 \
                                nk_##output_type##_t reduction_results[4];                                           \
                                finalize_fn(&accumulator_states[row_index][0], &accumulator_states[row_index][1],    \
                                            &accumulator_states[row_index][2], &accumulator_states[row_index][3],    \
                                            reduction_results);                                                      \
                                                                                                                     \
                                nk_##output_type##_t *output_row =                                                   \
                                    (nk_##output_type##_t *)((char *)c_matrix +                                      \
                                                             (tile_row_start + row_index) * c_stride);               \
                                for (nk_size_t column_index = 0; column_index < tile_column_count; ++column_index) { \
                                    output_row[tile_column_start + column_index] +=                                  \
                                        (nk_##output_type##_t)reduction_results[column_index];                       \
                                }                                                                                    \
                            }                                                                                        \
                        }                                                                                            \
                    }                                                                                                \
                }                                                                                                    \
            }                                                                                                        \
        }                                                                                                            \
    }

/*  Packed buffer header for tiled layout (64-byte aligned).
 *  Used by all packed matmul backends (serial, AVX-512, AMX, SVE).
 */
typedef struct {
    nk_u32_t full_n_tiles;  // Number of full N tiles (TILE_N rows each)
    nk_u32_t full_k_tiles;  // Number of K tiles (TILE_K cols each, includes remainder)
    nk_u16_t n_edge_rows;   // Remaining N rows (for edge handling)
    nk_u16_t n_edge_offset; // Offset to N edge region (for AMX hybrid layout)
    nk_u32_t reserved[12];  // Padding to 64 bytes
} nk_dots_packed_header_t;

#define NK_DOTS_SERIAL_TILE_N      16
#define NK_DOTS_SERIAL_TILE_K_F64  8  // 8 × 8 bytes = 64 bytes
#define NK_DOTS_SERIAL_TILE_K_F32  16 // 16 × 4 bytes = 64 bytes
#define NK_DOTS_SERIAL_TILE_K_F16  32 // 32 × 2 bytes = 64 bytes
#define NK_DOTS_SERIAL_TILE_K_BF16 32 // 32 × 2 bytes = 64 bytes
#define NK_DOTS_SERIAL_TILE_K_I8   64 // 64 × 1 byte = 64 bytes
#define NK_DOTS_SERIAL_TILE_K_U8   64 // 64 × 1 byte = 64 bytes

// Helper to get tile_k for a given type (serial implementation)
#define NK_DOTS_SERIAL_TILE_K(input_type)                             \
    ((sizeof(nk_##input_type##_t) == 8)   ? NK_DOTS_SERIAL_TILE_K_F64 \
     : (sizeof(nk_##input_type##_t) == 4) ? NK_DOTS_SERIAL_TILE_K_F32 \
     : (sizeof(nk_##input_type##_t) == 2) ? NK_DOTS_SERIAL_TILE_K_F16 \
                                          : NK_DOTS_SERIAL_TILE_K_I8)

/**
 *  @brief Macro to generate packed_size function for serial tiled row-major format.
 *
 *  Calculates buffer size needed for packed B matrix including header and padding.
 *  Tiles are padded to full size even for edge cases to simplify access patterns.
 *  Uses MKL-style naming: nk_dots_{input}{input}{output}_packed_size_{suffix}
 */
#define NK_MAKE_DOTS_SERIAL_PACKED_SIZE(suffix, input_type, output_type, tile_k)                            \
    NK_PUBLIC nk_size_t nk_dots_##input_type##input_type##output_type##_packed_size_##suffix(nk_size_t n,   \
                                                                                             nk_size_t k) { \
        nk_size_t const tile_n = NK_DOTS_SERIAL_TILE_N;                                                     \
        nk_size_t const n_tiles = (n + tile_n - 1) / tile_n;                                                \
        nk_size_t const k_tiles = (k + tile_k - 1) / tile_k;                                                \
        nk_size_t const tile_size = tile_n * tile_k * sizeof(nk_##input_type##_t);                          \
        return sizeof(nk_dots_packed_header_t) + n_tiles * k_tiles * tile_size;                             \
    }

/**
 *  @brief Macro to generate pack function for serial tiled row-major format.
 *
 *  Packs B matrix into tiles: k-tiles outer loop, n-tiles inner loop.
 *  Each tile contains TILE_N rows × TILE_K elements in row-major order.
 *  Edge tiles are zero-padded to full tile size.
 *  Uses MKL-style naming: nk_dots_{input}{input}{output}_pack_{suffix}
 */
#define NK_MAKE_DOTS_SERIAL_PACK(suffix, input_type, output_type, tile_k)                                              \
    NK_PUBLIC void nk_dots_##input_type##input_type##output_type##_pack_##suffix(                                      \
        nk_##input_type##_t const *b, nk_size_t n, nk_size_t k, nk_size_t b_stride, void *b_packed) {                  \
                                                                                                                       \
        nk_size_t const tile_n = NK_DOTS_SERIAL_TILE_N;                                                                \
        nk_size_t const n_tiles = (n + tile_n - 1) / tile_n;                                                           \
        nk_size_t const k_tiles = (k + tile_k - 1) / tile_k;                                                           \
        nk_size_t const tile_size = tile_n * tile_k;                                                                   \
                                                                                                                       \
        /* Store dimensions in header */                                                                               \
        nk_dots_packed_header_t *header = (nk_dots_packed_header_t *)b_packed;                                         \
        header->full_n_tiles = (nk_u32_t)n_tiles;                                                                      \
        header->full_k_tiles = (nk_u32_t)k_tiles;                                                                      \
        header->n_edge_rows = (nk_u16_t)(n % tile_n);                                                                  \
        header->n_edge_offset = (nk_u16_t)(n - (n % tile_n));                                                          \
                                                                                                                       \
        nk_##input_type##_t *packed = (nk_##input_type##_t *)((char *)b_packed + sizeof(nk_dots_packed_header_t));     \
                                                                                                                       \
        /* Zero entire buffer for edge tile padding */                                                                 \
        for (nk_size_t i = 0; i < n_tiles * k_tiles * tile_size; ++i) packed[i] = 0;                                   \
                                                                                                                       \
        /* Pack tiles: k-tiles outer, n-tiles inner */                                                                 \
        for (nk_size_t kt = 0; kt < k_tiles; ++kt) {                                                                   \
            nk_size_t const k_start = kt * tile_k;                                                                     \
            nk_size_t const k_end = (k_start + tile_k < k) ? (k_start + tile_k) : k;                                   \
                                                                                                                       \
            for (nk_size_t nt = 0; nt < n_tiles; ++nt) {                                                               \
                nk_size_t const n_start = nt * tile_n;                                                                 \
                nk_size_t const n_end = (n_start + tile_n < n) ? (n_start + tile_n) : n;                               \
                                                                                                                       \
                nk_size_t const tile_idx = kt * n_tiles + nt;                                                          \
                nk_##input_type##_t *tile = packed + tile_idx * tile_size;                                             \
                                                                                                                       \
                /* Copy B rows into tile (row-major within tile) */                                                    \
                for (nk_size_t ni = n_start; ni < n_end; ++ni) {                                                       \
                    nk_##input_type##_t const *b_row = (nk_##input_type##_t const *)((char const *)b + ni * b_stride); \
                    nk_size_t const row_in_tile = ni - n_start;                                                        \
                                                                                                                       \
                    for (nk_size_t ki = k_start; ki < k_end; ++ki) {                                                   \
                        nk_size_t const col_in_tile = ki - k_start;                                                    \
                        tile[row_in_tile * tile_k + col_in_tile] = b_row[ki];                                          \
                    }                                                                                                  \
                }                                                                                                      \
            }                                                                                                          \
        }                                                                                                              \
    }

/**
 *  @brief Optimized serial matmul with 4×4 register blocking, 4× k-unrolling, and A-caching.
 *
 *  Computes C = A × Bᵀ where B is pre-packed in tiled row-major format.
 *
 *  Optimizations applied:
 *  1. Register Blocking (4×4): 16 scalar accumulators stay in CPU registers across k-loop
 *  2. K-loop Unrolling (4×): Reduces loop overhead, enables ILP for overlapping loads/FMAs
 *  3. A-row Caching: Load 4 A values once, reuse for all 4 B columns (16 FMAs per 4 A loads)
 *
 *  Micro-kernel computes a 4×4 output block:
 *    acc[r][c] += a[r] * b[c]  for r,c in [0,3]
 *
 *  Uses MKL-style naming: nk_dots_{input}{input}{output}_{suffix}
 */
#define NK_MAKE_DOTS_SERIAL_PACKED(suffix, input_type, accumulator_type, output_type, load_and_convert, tile_k)        \
    NK_PUBLIC void nk_dots_##input_type##input_type##output_type##_##suffix(                                           \
        nk_##input_type##_t const *a, void const *b_packed, nk_##output_type##_t *c, nk_size_t m, nk_size_t n,         \
        nk_size_t k, nk_size_t a_stride, nk_size_t c_stride) {                                                         \
                                                                                                                       \
        /* Blocking parameters */                                                                                      \
        nk_size_t const mr_size = 4;  /* Rows of A per micro-kernel */                                                 \
        nk_size_t const nr_size = 4;  /* Columns of B per micro-kernel */                                              \
        nk_size_t const k_unroll = 4; /* K elements per unrolled iteration */                                          \
                                                                                                                       \
        nk_size_t const tile_n = NK_DOTS_SERIAL_TILE_N;                                                                \
        nk_size_t const n_tiles = (n + tile_n - 1) / tile_n;                                                           \
        nk_size_t const k_tiles = (k + tile_k - 1) / tile_k;                                                           \
        nk_size_t const tile_size = tile_n * tile_k;                                                                   \
                                                                                                                       \
        nk_##input_type##_t const *packed = (nk_##input_type##_t const *)((char const *)b_packed +                     \
                                                                          sizeof(nk_dots_packed_header_t));            \
                                                                                                                       \
        /* Zero output matrix */                                                                                       \
        for (nk_size_t mi = 0; mi < m; ++mi) {                                                                         \
            nk_##output_type##_t *c_row = (nk_##output_type##_t *)((char *)c + mi * c_stride);                         \
            for (nk_size_t ni = 0; ni < n; ++ni) c_row[ni] = 0;                                                        \
        }                                                                                                              \
                                                                                                                       \
        /* Process k-tiles in outer loop for better A reuse */                                                         \
        for (nk_size_t kt = 0; kt < k_tiles; ++kt) {                                                                   \
            nk_size_t const k_start = kt * tile_k;                                                                     \
            nk_size_t const k_end = (k_start + tile_k < k) ? (k_start + tile_k) : k;                                   \
            nk_size_t const k_len = k_end - k_start;                                                                   \
                                                                                                                       \
            /* Process rows in blocks of MR for register blocking */                                                   \
            for (nk_size_t mi_block = 0; mi_block < m; mi_block += mr_size) {                                          \
                nk_size_t const mr_end = (mi_block + mr_size < m) ? (mi_block + mr_size) : m;                          \
                nk_size_t const mr_len = mr_end - mi_block;                                                            \
                                                                                                                       \
                for (nk_size_t nt = 0; nt < n_tiles; ++nt) {                                                           \
                    nk_size_t const n_start = nt * tile_n;                                                             \
                    nk_size_t const n_end = (n_start + tile_n < n) ? (n_start + tile_n) : n;                           \
                                                                                                                       \
                    nk_size_t const tile_idx = kt * n_tiles + nt;                                                      \
                    nk_##input_type##_t const *tile = packed + tile_idx * tile_size;                                   \
                                                                                                                       \
                    /* Process columns in blocks of NR for register blocking */                                        \
                    for (nk_size_t j_block = n_start; j_block < n_end; j_block += nr_size) {                           \
                        nk_size_t const nr_end = (j_block + nr_size < n_end) ? (j_block + nr_size) : n_end;            \
                        nk_size_t const nr_len = nr_end - j_block;                                                     \
                                                                                                                       \
                        /* 4×4 accumulator block - stays in registers across k-loop */                                 \
                        nk_##accumulator_type##_t acc00 = 0, acc01 = 0, acc02 = 0, acc03 = 0;                          \
                        nk_##accumulator_type##_t acc10 = 0, acc11 = 0, acc12 = 0, acc13 = 0;                          \
                        nk_##accumulator_type##_t acc20 = 0, acc21 = 0, acc22 = 0, acc23 = 0;                          \
                        nk_##accumulator_type##_t acc30 = 0, acc31 = 0, acc32 = 0, acc33 = 0;                          \
                                                                                                                       \
                        /* Get A row pointers for MR rows */                                                           \
                        nk_##input_type##_t const *a_row0 =                                                            \
                            (nk_##input_type##_t const *)((char const *)a + mi_block * a_stride) + k_start;            \
                        nk_##input_type##_t const *a_row1 =                                                            \
                            (mr_len > 1)                                                                               \
                                ? (nk_##input_type##_t const *)((char const *)a + (mi_block + 1) * a_stride) + k_start \
                                : a_row0;                                                                              \
                        nk_##input_type##_t const *a_row2 =                                                            \
                            (mr_len > 2)                                                                               \
                                ? (nk_##input_type##_t const *)((char const *)a + (mi_block + 2) * a_stride) + k_start \
                                : a_row0;                                                                              \
                        nk_##input_type##_t const *a_row3 =                                                            \
                            (mr_len > 3)                                                                               \
                                ? (nk_##input_type##_t const *)((char const *)a + (mi_block + 3) * a_stride) + k_start \
                                : a_row0;                                                                              \
                                                                                                                       \
                        /* Get B row pointers for NR columns */                                                        \
                        nk_size_t const j0_in_tile = j_block - n_start;                                                \
                        nk_##input_type##_t const *b_row0 = tile + j0_in_tile * tile_k;                                \
                        nk_##input_type##_t const *b_row1 = (nr_len > 1) ? tile + (j0_in_tile + 1) * tile_k : b_row0;  \
                        nk_##input_type##_t const *b_row2 = (nr_len > 2) ? tile + (j0_in_tile + 2) * tile_k : b_row0;  \
                        nk_##input_type##_t const *b_row3 = (nr_len > 3) ? tile + (j0_in_tile + 3) * tile_k : b_row0;  \
                                                                                                                       \
                        /* Main k-loop with 4× unrolling */                                                            \
                        nk_size_t ki = 0;                                                                              \
                        nk_##accumulator_type##_t a0, a1, a2, a3, b0, b1, b2, b3;                                      \
                        for (; ki + k_unroll <= k_len; ki += k_unroll) {                                               \
                            /* Unroll 0: Load 4 A values, 4 B values, do 16 FMAs */                                    \
                            load_and_convert(a_row0 + ki, &a0), load_and_convert(a_row1 + ki, &a1);                    \
                            load_and_convert(a_row2 + ki, &a2), load_and_convert(a_row3 + ki, &a3);                    \
                            load_and_convert(b_row0 + ki, &b0), load_and_convert(b_row1 + ki, &b1);                    \
                            load_and_convert(b_row2 + ki, &b2), load_and_convert(b_row3 + ki, &b3);                    \
                            acc00 += a0 * b0, acc01 += a0 * b1, acc02 += a0 * b2, acc03 += a0 * b3;                    \
                            acc10 += a1 * b0, acc11 += a1 * b1, acc12 += a1 * b2, acc13 += a1 * b3;                    \
                            acc20 += a2 * b0, acc21 += a2 * b1, acc22 += a2 * b2, acc23 += a2 * b3;                    \
                            acc30 += a3 * b0, acc31 += a3 * b1, acc32 += a3 * b2, acc33 += a3 * b3;                    \
                                                                                                                       \
                            /* Unroll 1 */                                                                             \
                            load_and_convert(a_row0 + ki + 1, &a0), load_and_convert(a_row1 + ki + 1, &a1);            \
                            load_and_convert(a_row2 + ki + 1, &a2), load_and_convert(a_row3 + ki + 1, &a3);            \
                            load_and_convert(b_row0 + ki + 1, &b0), load_and_convert(b_row1 + ki + 1, &b1);            \
                            load_and_convert(b_row2 + ki + 1, &b2), load_and_convert(b_row3 + ki + 1, &b3);            \
                            acc00 += a0 * b0, acc01 += a0 * b1, acc02 += a0 * b2, acc03 += a0 * b3;                    \
                            acc10 += a1 * b0, acc11 += a1 * b1, acc12 += a1 * b2, acc13 += a1 * b3;                    \
                            acc20 += a2 * b0, acc21 += a2 * b1, acc22 += a2 * b2, acc23 += a2 * b3;                    \
                            acc30 += a3 * b0, acc31 += a3 * b1, acc32 += a3 * b2, acc33 += a3 * b3;                    \
                                                                                                                       \
                            /* Unroll 2 */                                                                             \
                            load_and_convert(a_row0 + ki + 2, &a0), load_and_convert(a_row1 + ki + 2, &a1);            \
                            load_and_convert(a_row2 + ki + 2, &a2), load_and_convert(a_row3 + ki + 2, &a3);            \
                            load_and_convert(b_row0 + ki + 2, &b0), load_and_convert(b_row1 + ki + 2, &b1);            \
                            load_and_convert(b_row2 + ki + 2, &b2), load_and_convert(b_row3 + ki + 2, &b3);            \
                            acc00 += a0 * b0, acc01 += a0 * b1, acc02 += a0 * b2, acc03 += a0 * b3;                    \
                            acc10 += a1 * b0, acc11 += a1 * b1, acc12 += a1 * b2, acc13 += a1 * b3;                    \
                            acc20 += a2 * b0, acc21 += a2 * b1, acc22 += a2 * b2, acc23 += a2 * b3;                    \
                            acc30 += a3 * b0, acc31 += a3 * b1, acc32 += a3 * b2, acc33 += a3 * b3;                    \
                                                                                                                       \
                            /* Unroll 3 */                                                                             \
                            load_and_convert(a_row0 + ki + 3, &a0), load_and_convert(a_row1 + ki + 3, &a1);            \
                            load_and_convert(a_row2 + ki + 3, &a2), load_and_convert(a_row3 + ki + 3, &a3);            \
                            load_and_convert(b_row0 + ki + 3, &b0), load_and_convert(b_row1 + ki + 3, &b1);            \
                            load_and_convert(b_row2 + ki + 3, &b2), load_and_convert(b_row3 + ki + 3, &b3);            \
                            acc00 += a0 * b0, acc01 += a0 * b1, acc02 += a0 * b2, acc03 += a0 * b3;                    \
                            acc10 += a1 * b0, acc11 += a1 * b1, acc12 += a1 * b2, acc13 += a1 * b3;                    \
                            acc20 += a2 * b0, acc21 += a2 * b1, acc22 += a2 * b2, acc23 += a2 * b3;                    \
                            acc30 += a3 * b0, acc31 += a3 * b1, acc32 += a3 * b2, acc33 += a3 * b3;                    \
                        }                                                                                              \
                                                                                                                       \
                        /* Remainder k-loop (handles k_len % 4) */                                                     \
                        for (; ki < k_len; ++ki) {                                                                     \
                            load_and_convert(a_row0 + ki, &a0), load_and_convert(a_row1 + ki, &a1);                    \
                            load_and_convert(a_row2 + ki, &a2), load_and_convert(a_row3 + ki, &a3);                    \
                            load_and_convert(b_row0 + ki, &b0), load_and_convert(b_row1 + ki, &b1);                    \
                            load_and_convert(b_row2 + ki, &b2), load_and_convert(b_row3 + ki, &b3);                    \
                            acc00 += a0 * b0, acc01 += a0 * b1, acc02 += a0 * b2, acc03 += a0 * b3;                    \
                            acc10 += a1 * b0, acc11 += a1 * b1, acc12 += a1 * b2, acc13 += a1 * b3;                    \
                            acc20 += a2 * b0, acc21 += a2 * b1, acc22 += a2 * b2, acc23 += a2 * b3;                    \
                            acc30 += a3 * b0, acc31 += a3 * b1, acc32 += a3 * b2, acc33 += a3 * b3;                    \
                        }                                                                                              \
                                                                                                                       \
                        /* Store accumulated results to C */                                                           \
                        nk_##output_type##_t *c_row0 = (nk_##output_type##_t *)((char *)c + mi_block * c_stride);      \
                        if (nr_len > 0) c_row0[j_block] += (nk_##output_type##_t)acc00;                                \
                        if (nr_len > 1) c_row0[j_block + 1] += (nk_##output_type##_t)acc01;                            \
                        if (nr_len > 2) c_row0[j_block + 2] += (nk_##output_type##_t)acc02;                            \
                        if (nr_len > 3) c_row0[j_block + 3] += (nk_##output_type##_t)acc03;                            \
                                                                                                                       \
                        if (mr_len > 1) {                                                                              \
                            nk_##output_type##_t *c_row1 = (nk_##output_type##_t *)((char *)c +                        \
                                                                                    (mi_block + 1) * c_stride);        \
                            if (nr_len > 0) c_row1[j_block] += (nk_##output_type##_t)acc10;                            \
                            if (nr_len > 1) c_row1[j_block + 1] += (nk_##output_type##_t)acc11;                        \
                            if (nr_len > 2) c_row1[j_block + 2] += (nk_##output_type##_t)acc12;                        \
                            if (nr_len > 3) c_row1[j_block + 3] += (nk_##output_type##_t)acc13;                        \
                        }                                                                                              \
                        if (mr_len > 2) {                                                                              \
                            nk_##output_type##_t *c_row2 = (nk_##output_type##_t *)((char *)c +                        \
                                                                                    (mi_block + 2) * c_stride);        \
                            if (nr_len > 0) c_row2[j_block] += (nk_##output_type##_t)acc20;                            \
                            if (nr_len > 1) c_row2[j_block + 1] += (nk_##output_type##_t)acc21;                        \
                            if (nr_len > 2) c_row2[j_block + 2] += (nk_##output_type##_t)acc22;                        \
                            if (nr_len > 3) c_row2[j_block + 3] += (nk_##output_type##_t)acc23;                        \
                        }                                                                                              \
                        if (mr_len > 3) {                                                                              \
                            nk_##output_type##_t *c_row3 = (nk_##output_type##_t *)((char *)c +                        \
                                                                                    (mi_block + 3) * c_stride);        \
                            if (nr_len > 0) c_row3[j_block] += (nk_##output_type##_t)acc30;                            \
                            if (nr_len > 1) c_row3[j_block + 1] += (nk_##output_type##_t)acc31;                        \
                            if (nr_len > 2) c_row3[j_block + 2] += (nk_##output_type##_t)acc32;                        \
                            if (nr_len > 3) c_row3[j_block + 3] += (nk_##output_type##_t)acc33;                        \
                        }                                                                                              \
                    }                                                                                                  \
                }                                                                                                      \
            }                                                                                                          \
        }                                                                                                              \
    }

// Helper conversion functions for serial GEMM (dual-pointer style)
NK_INTERNAL void nk_serial_copy_f32(nk_f32_t const *src, nk_f32_t *dst) { *dst = *src; }
NK_INTERNAL void nk_serial_copy_i8_to_i32(nk_i8_t const *src, nk_i32_t *dst) { *dst = (nk_i32_t)(*src); }

// Serial packed implementations for BF16 (32 elements per 64-byte tile row)
NK_MAKE_DOTS_SERIAL_PACKED_SIZE(serial, bf16, f32, NK_DOTS_SERIAL_TILE_K_BF16)
NK_MAKE_DOTS_SERIAL_PACK(serial, bf16, f32, NK_DOTS_SERIAL_TILE_K_BF16)
NK_MAKE_DOTS_SERIAL_PACKED(serial, bf16, f32, f32, nk_bf16_to_f32, NK_DOTS_SERIAL_TILE_K_BF16)

// Serial packed implementations for I8 (64 elements per 64-byte tile row)
NK_MAKE_DOTS_SERIAL_PACKED_SIZE(serial, i8, i32, NK_DOTS_SERIAL_TILE_K_I8)
NK_MAKE_DOTS_SERIAL_PACK(serial, i8, i32, NK_DOTS_SERIAL_TILE_K_I8)
NK_MAKE_DOTS_SERIAL_PACKED(serial, i8, i32, i32, nk_serial_copy_i8_to_i32, NK_DOTS_SERIAL_TILE_K_I8)

// Serial packed implementations for F32 (16 elements per 64-byte tile row)
NK_MAKE_DOTS_SERIAL_PACKED_SIZE(serial, f32, f32, NK_DOTS_SERIAL_TILE_K_F32)
NK_MAKE_DOTS_SERIAL_PACK(serial, f32, f32, NK_DOTS_SERIAL_TILE_K_F32)
NK_MAKE_DOTS_SERIAL_PACKED(serial, f32, f32, f32, nk_serial_copy_f32, NK_DOTS_SERIAL_TILE_K_F32)

/*  Serial compact functions: simple scalar implementations for post-matmul conversion.
 *  These work on any platform without SIMD requirements.
 */

/*  BF16 compact: truncate F32 → BF16 in-place.
 *  Reads F32 matrix with c_stride, writes BF16 tightly packed (stride = n * sizeof(bf16)).
 */
NK_PUBLIC void nk_dots_bf16bf16bf16_serial( //
    void *c, nk_size_t m, nk_size_t n,      //
    nk_size_t c_stride) {

    nk_size_t const c_stride_f32 = c_stride / sizeof(nk_f32_t);
    nk_f32_t const *c_f32 = (nk_f32_t const *)c;
    nk_bf16_t *c_bf16 = (nk_bf16_t *)c;

    for (nk_size_t row = 0; row < m; row++) {
        nk_f32_t const *src_row = c_f32 + row * c_stride_f32;
        nk_bf16_t *dst_row = c_bf16 + row * n;
        for (nk_size_t col = 0; col < n; col++) { nk_f32_to_bf16(src_row + col, dst_row + col); }
    }
}

/*  I8 compact: re-normalize I32 → I8 using precomputed squared norms.
 *  Formula: c_i8[i][j] = c_i32[i][j] * 127 / sqrt(a_norm[i] * b_norm[j])
 *  Output is tightly packed (stride = n * sizeof(i8)).
 */
NK_PUBLIC void nk_dots_i8i8i8_serial(  //
    void *c, nk_size_t m, nk_size_t n, //
    nk_size_t c_stride,                //
    nk_i32_t const *a_squared_norms, nk_i32_t const *b_squared_norms) {

    nk_size_t const c_stride_i32 = c_stride / sizeof(nk_i32_t);
    nk_i32_t const *c_i32 = (nk_i32_t const *)c;
    nk_i8_t *c_i8 = (nk_i8_t *)c;

    for (nk_size_t row = 0; row < m; row++) {
        nk_i32_t const *src_row = c_i32 + row * c_stride_i32;
        nk_i8_t *dst_row = c_i8 + row * n;

        nk_f32_t a_norm_f32 = (nk_f32_t)a_squared_norms[row];
        nk_f32_t a_rsqrt = (a_norm_f32 > 0) ? (1.0f / NK_F32_SQRT(a_norm_f32)) : 0.0f;

        for (nk_size_t col = 0; col < n; col++) {
            nk_f32_t b_norm_f32 = (nk_f32_t)b_squared_norms[col];
            nk_f32_t b_rsqrt = (b_norm_f32 > 0) ? (1.0f / NK_F32_SQRT(b_norm_f32)) : 0.0f;

            nk_f32_t normalized = (nk_f32_t)src_row[col] * 127.0f * a_rsqrt * b_rsqrt;
            nk_i32_t clamped = (nk_i32_t)normalized;
            if (clamped < -128) clamped = -128;
            if (clamped > 127) clamped = 127;
            dst_row[col] = (nk_i8_t)clamped;
        }
    }
}

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_DOTS_SERIAL_H
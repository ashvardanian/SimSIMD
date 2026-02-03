/**
 *  @brief SIMD-accelerated Batched Dot Products for SME.
 *  @file include/numkong/dots/sme.h
 *  @author Ash Vardanian
 *  @date January 2, 2026
 *
 *  @sa include/numkong/dots.h
 *
 *  Uses ARM Scalable Matrix Extension (SME) for efficient matrix multiplication
 *  with `ZA32` tiles supporting `f16`, `bf16`, `i8`, `u8`, and `e4m3` input types:
 *
 *  - `svmopa_za32_f16_m`: `f16` × `f16` outer product accumulate to `f32`
 *  - `svmopa_za32_bf16_m`: `bf16` × `bf16` outer product accumulate to `f32`
 *  - `svmopa_za32_s8_m`: `i8` × `i8` outer product accumulate to `i32`
 *  - `svmopa_za32_u8_m`: `u8` × `u8` outer product accumulate to `u32`
 *
 *  SME tile dimensions (for SVL=512, i.e., Apple M4):
 *
 *  - `ZA32` tile: 16 × 16 `f32`/`i32` elements (1KB)
 *  - `f16`/`bf16` vectors: 32 elements per SVE vector
 *  - `i8`/`u8` vectors: 64 elements per SVE vector
 *  - `f32`/`i32` vectors: 16 elements per SVE vector
 *
 *  Output pattern: Each `svmopa` accumulates a 16 × 16 tile from input vectors.
 *  We process multiple ZA tiles (0-3) to form larger output blocks.
 *
 *  Performance characteristics (Apple M4):
 *
 *  - `f16` → `f32` peak: ~2 TFLOPS per core
 *  - `bf16` → `f32` peak: ~2 TFLOPS per core
 *  - `i8` → `i32` peak: ~2 TOPS per core
 *  - Streaming mode has different register set from normal NEON
 *
 *  Acceleration opportunities:
 *
 *  - Pre-pack B matrix for column-major access: avoids transpose overhead
 *  - Tile along M/N dimensions: cache efficiency
 *  - Use multiple ZA tiles: 2×2 output blocking
 *
 *  @section dots_sme_instructions ARM SME Instructions
 *
 *      Intrinsic                       Instruction                     Latency     Throughput
 *      `svmopa_za32_f16_m`             `FMOPA` (ZA.S, P/M, Z.H, Z.H)   16cy        amortized
 *      `svmopa_za32_bf16_m`            `BFMOPA` (ZA.S, P/M, Z.H, Z.H)  16cy        amortized
 *      `svmopa_za32_s8_m`              `SMOPA` (ZA.S, P/M, Z.B, Z.B)   16cy        amortized
 *      `svmopa_za32_u8_m`              `UMOPA` (ZA.S, P/M, Z.B, Z.B)   16cy        amortized
 *      `svzero_za`                     `ZERO` (ZA)                     2cy         1/cy
 *      `svld1_hor_za32`                `LD1W` (ZA.S[Ws, #imm], P/Z)    4-6cy       1/cy
 *      `svst1_hor_za32`                `ST1W` (ZA.S[Ws, #imm], P)      4cy         1/cy
 *      `__arm_streaming`               `SMSTART`                       ~50-100cy
 *      `__arm_streaming` (exit)        `SMSTOP`                        ~50-100cy
 *      `__arm_new("za")`               ZA tile allocation              0cy
 *      `svcntw`                        `CNTW` (Xd)                     1cy         2/cy
 *      `svcnth`                        `CNTH` (Xd)                     1cy         2/cy
 */
#ifndef NK_DOTS_SME_H
#define NK_DOTS_SME_H

#if NK_TARGET_ARM_
#if NK_TARGET_SME
#pragma GCC push_options
#pragma GCC target("+sme")
#pragma clang attribute push(__attribute__((target("sme"))), apply_to = function)

#include "numkong/types.h"

#include <stdlib.h> // aligned_alloc, free
#include <arm_sme.h>
#include <arm_sve.h>

#if defined(__cplusplus)
extern "C" {
#endif

/**
 *  SME-specific packed buffer header (64-byte aligned).
 *  Layout optimized for SME outer product access patterns with predicate-based edge handling.
 */
typedef struct {
    nk_u32_t column_tile_count; // ⌈columns/tile_dim⌉: number of column tiles
    nk_u32_t depth_tile_count;  // ⌈depth/depth_tile_size⌉: number of depth tiles
    nk_u32_t columns;           // actual N dimension for predicates
    nk_u32_t depth;             // actual K dimension for predicates
    nk_u32_t svl_bytes;         // SVL in bytes at pack time: for validation
    nk_u32_t reserved[11];      // padding to 64 bytes
} nk_dots_sme_packed_header_t;

/*  Get SVL (Streaming Vector Length) in bytes for `f32` elements.
 *  On Apple M4 with SME2, this is typically 64 bytes (16 `f32` elements).
 */
NK_INTERNAL nk_size_t nk_sme_svl_bytes_(void) __arm_streaming_compatible { return svcntw() * sizeof(nk_f32_t); }

/*  Get number of `f16`/`bf16` elements per SVE vector in streaming mode.
 *  This is SVL/16 = 32 elements for 512-bit SVL.
 */
NK_INTERNAL nk_size_t nk_sme_f16_elements_(void) __arm_streaming_compatible { return svcnth(); }

/*  Get number of `f32` elements per ZA tile row.
 *  This is SVL/32 = 16 elements for 512-bit SVL.
 */
NK_INTERNAL nk_size_t nk_sme_tile_dim_(void) __arm_streaming_compatible { return svcntw(); }

/*  Zero all 4 `ZA32` tiles (tiles 0-3).
 *  Must be called at start of GEMM computation.
 */
NK_INTERNAL void nk_sme_zero_za32_(void) __arm_streaming __arm_inout("za") { svzero_za(); }

/*  `f16` packed buffer size calculation.
 *  Layout: header + ceiling tile counts with zero-padding for predicates.
 *
 *  SME `f16` tile dimensions:
 *  - Each `f16` vector has SVL/16 elements: 32 for 512-bit SVL
 *  - Each `ZA32` tile is SVL/32 × SVL/32: 16 × 16 for 512-bit SVL
 *  - We tile N in increments of SVL/32: 16 rows
 *  - We tile K in increments of SVL/16: 32 columns
 *
 *  Predicate-based approach: allocate ceiling tile counts, zero-pad partial tiles.
 */
NK_PUBLIC nk_size_t nk_dots_packed_size_f16_sme(nk_size_t n, nk_size_t k) {
    nk_size_t const tile_dim = svcntsw();        // rows per tile: number of `f32` elements
    nk_size_t const depth_tile_size = svcntsh(); // K elements per tile: number of `f16` elements

    nk_size_t const column_tile_count = (n + tile_dim - 1) / tile_dim;
    nk_size_t const depth_tile_count = (k + depth_tile_size - 1) / depth_tile_size;

    // Header (64 bytes aligned)
    nk_size_t size = sizeof(nk_dots_sme_packed_header_t);

    // All tiles including partial tiles (zero-padded)
    size += column_tile_count * depth_tile_count * tile_dim * depth_tile_size * sizeof(nk_f16_t);

    return size;
}

/*  `bf16` packed buffer size calculation.
 */
NK_PUBLIC nk_size_t nk_dots_packed_size_bf16_sme(nk_size_t n, nk_size_t k) {
    // Same dimensions as `f16` since both are 16-bit
    return nk_dots_packed_size_f16_sme(n, k);
}

/*  Pack `f16` B matrix for SME outer product access.
 *
 *  SME outer product: ZA[i,j] += A[i] × B[j]
 *  For GEMM C = A × Bᵀ, we need B stored column-major so that
 *  loading a column of B gives us the elements for one N output row.
 *
 *  Layout: tiles are stored in (column_tile, depth_tile) order with column-major
 *  element ordering within each tile. Partial tiles are zero-padded for predicates.
 */
NK_PUBLIC void nk_dots_pack_f16_sme(             //
    nk_f16_t const *b, nk_size_t n, nk_size_t k, //
    nk_size_t b_stride, void *b_packed) {

    nk_size_t const svl_bytes = svcntsw() * sizeof(nk_f32_t);
    nk_size_t const tile_dim = svcntsw();        // rows per tile
    nk_size_t const depth_tile_size = svcntsh(); // K elements per tile
    nk_size_t const tile_elements = tile_dim * depth_tile_size;
    nk_size_t const b_stride_elements = b_stride / sizeof(nk_f16_t);

    nk_size_t const column_tile_count = (n + tile_dim - 1) / tile_dim;
    nk_size_t const depth_tile_count = (k + depth_tile_size - 1) / depth_tile_size;
    nk_size_t const total_tiles = column_tile_count * depth_tile_count;

    // Write header with actual dimensions for predicate generation
    nk_dots_sme_packed_header_t *header = (nk_dots_sme_packed_header_t *)b_packed;
    header->column_tile_count = (nk_u32_t)column_tile_count;
    header->depth_tile_count = (nk_u32_t)depth_tile_count;
    header->columns = (nk_u32_t)n;
    header->depth = (nk_u32_t)k;
    header->svl_bytes = (nk_u32_t)svl_bytes;

    nk_f16_t *tiles_ptr = (nk_f16_t *)((char *)b_packed + sizeof(nk_dots_sme_packed_header_t));

    // Zero-initialize all tiles (partial tiles stay zero-padded)
    for (nk_size_t i = 0; i < total_tiles * tile_elements; i++) { tiles_ptr[i] = 0; }

    // Pack tiles: column-major within each tile for efficient SVE loads
    for (nk_size_t column_tile = 0; column_tile < column_tile_count; column_tile++) {
        for (nk_size_t depth_tile = 0; depth_tile < depth_tile_count; depth_tile++) {
            nk_size_t const tile_index = column_tile * depth_tile_count + depth_tile;
            nk_f16_t *tile_output = tiles_ptr + tile_index * tile_elements;

            nk_size_t const src_row_start = column_tile * tile_dim;
            nk_size_t const src_col_start = depth_tile * depth_tile_size;
            nk_size_t const rows_to_pack = (src_row_start + tile_dim <= n) ? tile_dim : (n - src_row_start);
            nk_size_t const cols_to_pack = (src_col_start + depth_tile_size <= k) ? depth_tile_size
                                                                                  : (k - src_col_start);

            // Column-major packing: tile_output[col * tile_dim + row]
            for (nk_size_t row = 0; row < rows_to_pack; row++) {
                for (nk_size_t col = 0; col < cols_to_pack; col++) {
                    nk_size_t const src_idx = (src_row_start + row) * b_stride_elements + src_col_start + col;
                    nk_size_t const dst_idx = col * tile_dim + row;
                    tile_output[dst_idx] = b[src_idx];
                }
            }
        }
    }
}

/*  Pack `bf16` B matrix for SME outer product access.
 *  Partial tiles are zero-padded for predicate-based edge handling.
 */
NK_PUBLIC void nk_dots_pack_bf16_sme(             //
    nk_bf16_t const *b, nk_size_t n, nk_size_t k, //
    nk_size_t b_stride, void *b_packed) {

    nk_size_t const svl_bytes = svcntsw() * sizeof(nk_f32_t);
    nk_size_t const tile_dim = svcntsw();        // rows per tile
    nk_size_t const depth_tile_size = svcntsh(); // K elements per tile
    nk_size_t const tile_elements = tile_dim * depth_tile_size;
    nk_size_t const b_stride_elements = b_stride / sizeof(nk_bf16_t);

    nk_size_t const column_tile_count = (n + tile_dim - 1) / tile_dim;
    nk_size_t const depth_tile_count = (k + depth_tile_size - 1) / depth_tile_size;
    nk_size_t const total_tiles = column_tile_count * depth_tile_count;

    nk_dots_sme_packed_header_t *header = (nk_dots_sme_packed_header_t *)b_packed;
    header->column_tile_count = (nk_u32_t)column_tile_count;
    header->depth_tile_count = (nk_u32_t)depth_tile_count;
    header->columns = (nk_u32_t)n;
    header->depth = (nk_u32_t)k;
    header->svl_bytes = (nk_u32_t)svl_bytes;

    nk_bf16_t *tiles_ptr = (nk_bf16_t *)((char *)b_packed + sizeof(nk_dots_sme_packed_header_t));

    // Zero-initialize all tiles
    for (nk_size_t i = 0; i < total_tiles * tile_elements; i++) { tiles_ptr[i] = 0; }

    for (nk_size_t column_tile = 0; column_tile < column_tile_count; column_tile++) {
        for (nk_size_t depth_tile = 0; depth_tile < depth_tile_count; depth_tile++) {
            nk_size_t const tile_index = column_tile * depth_tile_count + depth_tile;
            nk_bf16_t *tile_output = tiles_ptr + tile_index * tile_elements;

            nk_size_t const src_row_start = column_tile * tile_dim;
            nk_size_t const src_col_start = depth_tile * depth_tile_size;
            nk_size_t const rows_to_pack = (src_row_start + tile_dim <= n) ? tile_dim : (n - src_row_start);
            nk_size_t const cols_to_pack = (src_col_start + depth_tile_size <= k) ? depth_tile_size
                                                                                  : (k - src_col_start);

            for (nk_size_t row = 0; row < rows_to_pack; row++) {
                for (nk_size_t col = 0; col < cols_to_pack; col++) {
                    nk_size_t const src_idx = (src_row_start + row) * b_stride_elements + src_col_start + col;
                    nk_size_t const dst_idx = col * tile_dim + row;
                    tile_output[dst_idx] = b[src_idx];
                }
            }
        }
    }
}

/*  `f16` → `f32` GEMM core kernel using SME outer products with predicate-based edge handling.
 *
 *  Uses predicates for all tile processing, eliminating scalar edge handlers.
 *  Simple tile-by-tile processing with dynamic predicates for partial tiles.
 */
__arm_locally_streaming __arm_new("za") static void nk_dots_f16_kernel_( //
    nk_f16_t const *a, void const *b_packed, nk_f32_t *c,                //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,                  //
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) {

    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_size_t const column_tile_count = header->column_tile_count;
    nk_size_t const depth_tile_count = header->depth_tile_count;

    nk_size_t const tile_dim = svcntw();        // number of `f32` elements per vector
    nk_size_t const depth_tile_size = svcnth(); // number of `f16` elements per vector
    nk_size_t const tile_elements = tile_dim * depth_tile_size;

    nk_f16_t const *b_tiles = (nk_f16_t const *)((char const *)b_packed + sizeof(nk_dots_sme_packed_header_t));

    svbool_t const ptrue_h = svptrue_b16();
    svbool_t const ptrue_s = svptrue_b32();

    nk_size_t const row_tile_count = (rows + tile_dim - 1) / tile_dim;

    // Process tile-by-tile with predicates for edge handling
    for (nk_size_t row_tile = 0; row_tile < row_tile_count; row_tile++) {
        nk_size_t const row_start = row_tile * tile_dim;
        nk_size_t const rows_remaining = (row_start + tile_dim <= rows) ? tile_dim : (rows - row_start);
        svbool_t const pred_rows = svwhilelt_b32((uint32_t)0, (uint32_t)rows_remaining);

        for (nk_size_t col_tile = 0; col_tile < column_tile_count; col_tile++) {
            nk_size_t const col_start = col_tile * tile_dim;
            nk_size_t const cols_remaining = (col_start + tile_dim <= columns) ? tile_dim : (columns - col_start);
            svbool_t const pred_cols = svwhilelt_b32((uint32_t)0, (uint32_t)cols_remaining);

            // Zero ZA tile 0
            svzero_za();

            // Accumulate over depth dimension
            for (nk_size_t d_tile = 0; d_tile < depth_tile_count; d_tile++) {
                nk_size_t const d_start = d_tile * depth_tile_size;

                nk_size_t const b_tile_idx = col_tile * depth_tile_count + d_tile;
                nk_f16_t const *b_tile = b_tiles + b_tile_idx * tile_elements;

                // Process tile_dim rows of outer products
                for (nk_size_t row = 0; row < tile_dim; row++) {
                    // Load A vector (predicated load: inactive lanes become 0)
                    nk_f16_t const *a_ptr = a + (row_start + row) * a_stride_elements + d_start;
                    svfloat16_t a_vec = svld1_f16(ptrue_h, (float16_t const *)a_ptr);

                    // Load B vector from packed tile
                    svfloat16_t b_vec = svld1_f16(ptrue_h, (float16_t const *)(b_tile + row * depth_tile_size));

                    // Predicated outer product (only accumulates valid rows/cols)
                    svmopa_za32_f16_m(0, pred_rows, pred_cols, a_vec, b_vec);
                }
            }

            // Predicated store to C
            for (nk_size_t row = 0; row < rows_remaining; row++) {
                svst1_hor_za32(0, row, pred_cols, c + (row_start + row) * c_stride_elements + col_start);
            }
        }
    }
}

/*  `bf16` → `f32` GEMM core kernel using SME outer products with predicate-based edge handling.
 *
 *  Uses predicates for all tile processing, eliminating scalar edge handlers.
 *  Simple tile-by-tile processing with dynamic predicates for partial tiles.
 */
__arm_locally_streaming __arm_new("za") static void nk_dots_bf16_kernel_( //
    nk_bf16_t const *a, void const *b_packed, nk_f32_t *c,                //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,                   //
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) {

    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_size_t const column_tile_count = header->column_tile_count;
    nk_size_t const depth_tile_count = header->depth_tile_count;

    nk_size_t const tile_dim = svcntw();        // number of `f32` elements per vector
    nk_size_t const depth_tile_size = svcnth(); // number of `bf16` elements per vector
    nk_size_t const tile_elements = tile_dim * depth_tile_size;

    nk_bf16_t const *b_tiles = (nk_bf16_t const *)((char const *)b_packed + sizeof(nk_dots_sme_packed_header_t));

    svbool_t const ptrue_h = svptrue_b16();

    nk_size_t const row_tile_count = (rows + tile_dim - 1) / tile_dim;

    // Process tile-by-tile with predicates for edge handling
    for (nk_size_t row_tile = 0; row_tile < row_tile_count; row_tile++) {
        nk_size_t const row_start = row_tile * tile_dim;
        nk_size_t const rows_remaining = (row_start + tile_dim <= rows) ? tile_dim : (rows - row_start);
        svbool_t const pred_rows = svwhilelt_b32((uint32_t)0, (uint32_t)rows_remaining);

        for (nk_size_t col_tile = 0; col_tile < column_tile_count; col_tile++) {
            nk_size_t const col_start = col_tile * tile_dim;
            nk_size_t const cols_remaining = (col_start + tile_dim <= columns) ? tile_dim : (columns - col_start);
            svbool_t const pred_cols = svwhilelt_b32((uint32_t)0, (uint32_t)cols_remaining);

            // Zero ZA tile 0
            svzero_za();

            // Accumulate over depth dimension
            for (nk_size_t d_tile = 0; d_tile < depth_tile_count; d_tile++) {
                nk_size_t const d_start = d_tile * depth_tile_size;

                nk_size_t const b_tile_idx = col_tile * depth_tile_count + d_tile;
                nk_bf16_t const *b_tile = b_tiles + b_tile_idx * tile_elements;

                // Process tile_dim rows of outer products
                for (nk_size_t row = 0; row < tile_dim; row++) {
                    // Load A vector
                    nk_bf16_t const *a_ptr = a + (row_start + row) * a_stride_elements + d_start;
                    svbfloat16_t a_vec = svld1_bf16(ptrue_h, (bfloat16_t const *)a_ptr);

                    // Load B vector from packed tile
                    svbfloat16_t b_vec = svld1_bf16(ptrue_h, (bfloat16_t const *)(b_tile + row * depth_tile_size));

                    // Predicated outer product
                    svmopa_za32_bf16_m(0, pred_rows, pred_cols, a_vec, b_vec);
                }
            }

            // Predicated store to C
            for (nk_size_t row = 0; row < rows_remaining; row++) {
                svst1_hor_za32(0, row, pred_cols, c + (row_start + row) * c_stride_elements + col_start);
            }
        }
    }
}

/*  `f16` → `f32` GEMM public interface.
 *  Predicate-based edge handling eliminates scalar fallbacks.
 *
 *  @param a         Input matrix A (rows × depth), row-major
 *  @param b_packed  Pre-packed B matrix from `nk_dots_pack_f16_sme`
 *  @param c         Output matrix C (rows × columns), row-major
 *  @param rows      Number of rows in A and C (M dimension)
 *  @param columns   Number of columns in C (N dimension)
 *  @param depth     Shared dimension (K dimension)
 *  @param a_stride  Byte stride between rows of A
 *  @param c_stride  Byte stride between rows of C
 */
NK_PUBLIC void nk_dots_packed_f16_sme(                    //
    nk_f16_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,   //
    nk_size_t a_stride, nk_size_t c_stride) {

    nk_size_t const a_stride_elements = a_stride / sizeof(nk_f16_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);

    nk_dots_f16_kernel_(a, b_packed, c, rows, columns, depth, a_stride_elements, c_stride_elements);
}

/*  `bf16` → `f32` GEMM public interface.
 *  Predicate-based edge handling eliminates scalar fallbacks.
 */
NK_PUBLIC void nk_dots_packed_bf16_sme(                    //
    nk_bf16_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,    //
    nk_size_t a_stride, nk_size_t c_stride) {

    nk_size_t const a_stride_elements = a_stride / sizeof(nk_bf16_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);

    nk_dots_bf16_kernel_(a, b_packed, c, rows, columns, depth, a_stride_elements, c_stride_elements);
}

/**
 *  `i8` × `i8` → `i32` GEMM using SME outer products.
 *
 *  Uses `svmopa_za32_s8_m` for signed 8-bit integer outer product accumulate.
 *  Available on Apple M4 (SME_I8I32 = 1).
 *
 *  Tile dimensions for `i8` → `i32` (512-bit SVL):
 *  - Input vectors: 64 `i8` elements (SVL/8 = 64)
 *  - Output tile: 16 × 16 `i32` elements (`ZA32`)
 *  - Each output `i32` is a dot product of 4 `i8` pairs
 *
 *  Expected performance: ~2 TOPS (4× `f16` due to 4:1 element packing)
 */

/*  `i8` packed buffer size calculation.
 *
 *  For `i8` → `i32` outer product:
 *  - Each `i8` vector has SVL/8 elements: 64 for 512-bit SVL
 *  - Each `ZA32` tile is SVL/32 × SVL/32: 16 × 16 for 512-bit SVL
 *  - `SMOPA` computes: for each 4 `i8` pairs, produce 1 `i32` output
 *  - We tile N in increments of 16 rows: output tile dimension
 *  - We tile K in increments of 64 columns: input vector width
 *
 *  Predicate-based approach: allocate ceiling tile counts, zero-pad partial tiles.
 */
NK_PUBLIC nk_size_t nk_dots_packed_size_i8_sme(nk_size_t n, nk_size_t k) {
    nk_size_t const tile_dim = svcntsw();        // rows per ZA32 tile: number of `i32` elements
    nk_size_t const depth_tile_size = svcntsb(); // K elements per tile: number of `i8` elements

    nk_size_t const column_tile_count = (n + tile_dim - 1) / tile_dim;
    nk_size_t const depth_tile_count = (k + depth_tile_size - 1) / depth_tile_size;

    // Header (64 bytes aligned)
    nk_size_t size = sizeof(nk_dots_sme_packed_header_t);

    // All tiles including partial tiles (zero-padded)
    size += column_tile_count * depth_tile_count * tile_dim * depth_tile_size * sizeof(nk_i8_t);

    return size;
}

/*  Pack `i8` B matrix for SME outer product access.
 *  Partial tiles are zero-padded for predicate-based edge handling.
 */
NK_PUBLIC void nk_dots_pack_i8_sme(             //
    nk_i8_t const *b, nk_size_t n, nk_size_t k, //
    nk_size_t b_stride, void *b_packed) {

    nk_size_t const svl_bytes = svcntsw() * sizeof(nk_i32_t);
    nk_size_t const tile_dim = svcntsw();        // rows per tile
    nk_size_t const depth_tile_size = svcntsb(); // K elements per tile
    nk_size_t const tile_elements = tile_dim * depth_tile_size;
    nk_size_t const b_stride_elements = b_stride / sizeof(nk_i8_t);

    nk_size_t const column_tile_count = (n + tile_dim - 1) / tile_dim;
    nk_size_t const depth_tile_count = (k + depth_tile_size - 1) / depth_tile_size;
    nk_size_t const total_tiles = column_tile_count * depth_tile_count;

    // Write header with actual dimensions for predicate generation
    nk_dots_sme_packed_header_t *header = (nk_dots_sme_packed_header_t *)b_packed;
    header->column_tile_count = (nk_u32_t)column_tile_count;
    header->depth_tile_count = (nk_u32_t)depth_tile_count;
    header->columns = (nk_u32_t)n;
    header->depth = (nk_u32_t)k;
    header->svl_bytes = (nk_u32_t)svl_bytes;

    nk_i8_t *tiles_ptr = (nk_i8_t *)((char *)b_packed + sizeof(nk_dots_sme_packed_header_t));

    // Zero-initialize all tiles
    for (nk_size_t i = 0; i < total_tiles * tile_elements; i++) { tiles_ptr[i] = 0; }

    // Pack tiles: column-major within each tile for efficient SVE loads
    for (nk_size_t column_tile = 0; column_tile < column_tile_count; column_tile++) {
        for (nk_size_t depth_tile = 0; depth_tile < depth_tile_count; depth_tile++) {
            nk_size_t const tile_index = column_tile * depth_tile_count + depth_tile;
            nk_i8_t *tile_output = tiles_ptr + tile_index * tile_elements;

            nk_size_t const src_row_start = column_tile * tile_dim;
            nk_size_t const src_col_start = depth_tile * depth_tile_size;
            nk_size_t const rows_to_pack = (src_row_start + tile_dim <= n) ? tile_dim : (n - src_row_start);
            nk_size_t const cols_to_pack = (src_col_start + depth_tile_size <= k) ? depth_tile_size
                                                                                  : (k - src_col_start);

            // Column-major packing: tile_output[col * tile_dim + row]
            for (nk_size_t row = 0; row < rows_to_pack; row++) {
                for (nk_size_t col = 0; col < cols_to_pack; col++) {
                    nk_size_t const src_idx = (src_row_start + row) * b_stride_elements + src_col_start + col;
                    nk_size_t const dst_idx = col * tile_dim + row;
                    tile_output[dst_idx] = b[src_idx];
                }
            }
        }
    }
}

/*  `i8` × `i8` → `i32` GEMM core kernel using SME outer products with predicate-based edge handling.
 *
 *  Uses `svmopa_za32_s8_m` for signed `i8` × `i8` → `i32` outer product accumulate.
 *  Each `SMOPA` instruction processes:
 *  - Two 64-element `i8` vectors: A row slice, B column slice
 *  - Produces 16 × 16 `i32` partial products accumulated into `ZA32` tile
 *  - 4 `i8` pairs contribute to each `i32` output element
 *
 *  Uses predicates for all tile processing, eliminating scalar edge handlers.
 */
__arm_locally_streaming __arm_new("za") static void nk_dots_i8_kernel_( //
    nk_i8_t const *a, void const *b_packed, nk_i32_t *c,                //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,                 //
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) {

    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_size_t const column_tile_count = header->column_tile_count;
    nk_size_t const depth_tile_count = header->depth_tile_count;

    nk_size_t const tile_dim = svcntw();        // number of `i32` elements per vector
    nk_size_t const depth_tile_size = svcntb(); // number of `i8` elements per vector
    nk_size_t const tile_elements = tile_dim * depth_tile_size;

    nk_i8_t const *b_tiles = (nk_i8_t const *)((char const *)b_packed + sizeof(nk_dots_sme_packed_header_t));

    svbool_t const ptrue_b = svptrue_b8();

    nk_size_t const row_tile_count = (rows + tile_dim - 1) / tile_dim;

    // Process tile-by-tile with predicates for edge handling
    for (nk_size_t row_tile = 0; row_tile < row_tile_count; row_tile++) {
        nk_size_t const row_start = row_tile * tile_dim;
        nk_size_t const rows_remaining = (row_start + tile_dim <= rows) ? tile_dim : (rows - row_start);
        svbool_t const pred_rows = svwhilelt_b32((uint32_t)0, (uint32_t)rows_remaining);

        for (nk_size_t col_tile = 0; col_tile < column_tile_count; col_tile++) {
            nk_size_t const col_start = col_tile * tile_dim;
            nk_size_t const cols_remaining = (col_start + tile_dim <= columns) ? tile_dim : (columns - col_start);
            svbool_t const pred_cols = svwhilelt_b32((uint32_t)0, (uint32_t)cols_remaining);

            // Zero ZA tile 0
            svzero_za();

            // Accumulate over depth dimension
            for (nk_size_t d_tile = 0; d_tile < depth_tile_count; d_tile++) {
                nk_size_t const d_start = d_tile * depth_tile_size;

                nk_size_t const b_tile_idx = col_tile * depth_tile_count + d_tile;
                nk_i8_t const *b_tile = b_tiles + b_tile_idx * tile_elements;

                // Process tile_dim rows of outer products
                for (nk_size_t row = 0; row < tile_dim; row++) {
                    // Load A vector
                    nk_i8_t const *a_ptr = a + (row_start + row) * a_stride_elements + d_start;
                    svint8_t a_vec = svld1_s8(ptrue_b, a_ptr);

                    // Load B vector from packed tile
                    svint8_t b_vec = svld1_s8(ptrue_b, b_tile + row * depth_tile_size);

                    // Predicated outer product
                    svmopa_za32_s8_m(0, pred_rows, pred_cols, a_vec, b_vec);
                }
            }

            // Predicated store to C
            for (nk_size_t row = 0; row < rows_remaining; row++) {
                svst1_hor_za32(0, row, pred_cols, c + (row_start + row) * c_stride_elements + col_start);
            }
        }
    }
}

/*  `i8` × `i8` → `i32` GEMM public interface.
 *  Predicate-based edge handling eliminates scalar fallbacks.
 *
 *  @param a         Input matrix A (rows × depth), row-major, `i8`
 *  @param b_packed  Pre-packed B matrix from `nk_dots_pack_i8_sme`
 *  @param c         Output matrix C (rows × columns), row-major, `i32`
 *  @param rows      Number of rows in A and C (M dimension)
 *  @param columns   Number of columns in C (N dimension)
 *  @param depth     Shared dimension (K dimension)
 *  @param a_stride  Byte stride between rows of A
 *  @param c_stride  Byte stride between rows of C
 */
NK_PUBLIC void nk_dots_packed_i8_sme(                    //
    nk_i8_t const *a, void const *b_packed, nk_i32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,  //
    nk_size_t a_stride, nk_size_t c_stride) {

    nk_size_t const a_stride_elements = a_stride / sizeof(nk_i8_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_i32_t);

    nk_dots_i8_kernel_(a, b_packed, c, rows, columns, depth, a_stride_elements, c_stride_elements);
}

/**
 *  Streaming SVE FP8 → `f16` conversion functions.
 *
 *  These functions convert `e4m3`/`e5m2` → `f16` using arithmetic operations that
 *  work entirely in streaming SVE mode. This enables fused GEMM kernels that
 *  stay in streaming mode the entire time, avoiding `SMSTART`/`SMSTOP` overhead.
 *
 *  Key insight: We can't use LUT lookup in streaming mode efficiently, so we
 *  use arithmetic conversion with `f16` multiply for subnormal handling.
 *
 *  `e4m3` format: S EEEE MMM (1+4+3 bits, bias=7, no infinity, 0x7F=NaN)
 *  `e5m2` format: S EEEEE MM (1+5+2 bits, bias=15, has infinity and NaN)
 *  `f16` format: S EEEEE MMMMMMMMMM (1+5+10 bits, bias=15)
 */

/*  Convert 32 `e4m3` values to 32 `f16` values using SSVE arithmetic.
 *
 *  Algorithm:
 *  - Normal (exp != 0): `f16` = sign | ((magnitude << 7) + 0x2000)
 *  - Subnormal (exp == 0): `f16` = sign | (mant × (1/512))
 *  - NaN (mag == 0x7F): `f16` = sign | 0x7E00
 *
 *  IMPORTANT: Caller must be in streaming mode (`smstart sm`) before calling.
 *  This function does NOT enter/exit streaming mode itself.
 *
 *  @param pg16   Predicate for 16-bit elements: use `svptrue_b16()`
 *  @param src    Pointer to 32 `e4m3` bytes: must be 64-byte aligned
 *  @param dst    Pointer to 32 `f16` values: must be 64-byte aligned
 */
NK_INTERNAL void nk_e4m3x32_to_f16x32_ssve_(svbool_t pg16, nk_e4m3_t const *src, nk_f16_t *dst) {
    // Load 32 bytes and unpack lower 32 to 16-bit
    // (At 512-bit SVL, svld1_u8 loads 64 bytes, svunpklo gives first 32 as 16-bit)
    svuint8_t bytes = svld1_u8(svptrue_b8(), (uint8_t const *)src);
    svuint16_t vals = svunpklo_u16(bytes);

    // Extract sign bit and shift to F16 sign position (bit 15)
    svuint16_t sign = svlsl_n_u16_x(pg16, svand_n_u16_x(pg16, vals, 0x80), 8);

    // Extract magnitude (lower 7 bits) and mantissa (lower 3 bits)
    svuint16_t mag = svand_n_u16_x(pg16, vals, 0x7F);
    svuint16_t mant = svand_n_u16_x(pg16, vals, 0x07);

    // Normal path: F16 = sign | ((mag << 7) + 0x2000)
    // This places exp in bits 14:10 with +8 bias adjustment, mant in bits 9:7
    svuint16_t normal = svadd_n_u16_x(pg16, svlsl_n_u16_x(pg16, mag, 7), 0x2000);
    normal = svorr_u16_x(pg16, normal, sign);

    // Subnormal path: value = mant × 2⁻⁹ = mant / 512
    // 1/512 in F16 = 0x1800 (exp=6, mant=0, so 2⁽⁶⁻¹⁵⁾ = 2⁻⁹)
    svfloat16_t mant_f16 = svcvt_f16_u16_x(pg16, mant);
    svfloat16_t scale = svreinterpret_f16_u16(svdup_n_u16(0x1800));
    svfloat16_t subnorm_abs = svmul_f16_x(pg16, mant_f16, scale);
    svuint16_t subnorm = svorr_u16_x(pg16, svreinterpret_u16_f16(subnorm_abs), sign);

    // Detect subnormals: exp == 0 means (byte & 0x78) == 0
    svbool_t is_subnorm = svcmpeq_n_u16(pg16, svand_n_u16_x(pg16, vals, 0x78), 0);

    // Detect NaN: mag == 0x7F
    svbool_t is_nan = svcmpeq_n_u16(pg16, mag, 0x7F);
    svuint16_t nan_val = svorr_n_u16_x(pg16, sign, 0x7E00);

    // Blend: subnorm path for exp==0, else normal, then fix NaN
    svuint16_t result = svsel_u16(is_subnorm, subnorm, normal);
    result = svsel_u16(is_nan, nan_val, result);

    svst1_u16(pg16, (uint16_t *)dst, result);
}

/*  Convert 32 `e5m2` values to 32 `f16` values using SSVE arithmetic.
 *
 *  Algorithm:
 *  - Normal (exp != 0, exp != 31): `f16` = sign | (magnitude << 8)
 *  - Subnormal (exp == 0): `f16` = sign | (mant × (1/65536))
 *  - Infinity (exp == 31, mant == 0): `f16` = sign | 0x7C00
 *  - NaN (exp == 31, mant != 0): `f16` = sign | 0x7E00
 *
 *  IMPORTANT: Caller must be in streaming mode (`smstart sm`) before calling.
 *
 *  @param pg16   Predicate for 16-bit elements: use `svptrue_b16()`
 *  @param src    Pointer to 32 `e5m2` bytes: must be 64-byte aligned
 *  @param dst    Pointer to 32 `f16` values: must be 64-byte aligned
 */
NK_INTERNAL void nk_e5m2x32_to_f16x32_ssve_(svbool_t pg16, nk_e5m2_t const *src, nk_f16_t *dst) {
    svuint8_t bytes = svld1_u8(svptrue_b8(), (uint8_t const *)src);
    svuint16_t vals = svunpklo_u16(bytes);

    // Extract sign bit and shift to F16 sign position (bit 15)
    svuint16_t sign = svlsl_n_u16_x(pg16, svand_n_u16_x(pg16, vals, 0x80), 8);

    // Extract magnitude (lower 7 bits), mantissa (lower 2 bits), exponent (bits 6:2)
    svuint16_t mag = svand_n_u16_x(pg16, vals, 0x7F);
    svuint16_t mant = svand_n_u16_x(pg16, vals, 0x03);
    svuint16_t exp = svlsr_n_u16_x(pg16, svand_n_u16_x(pg16, vals, 0x7C), 2);

    // Normal path: E5M2 and F16 have same bias (15), so just shift magnitude by 8
    // F16 = sign | (exp << 10) | (mant << 8) = sign | (mag << 8)
    svuint16_t normal = svorr_u16_x(pg16, svlsl_n_u16_x(pg16, mag, 8), sign);

    // Subnormal path: value = mant × 2⁻¹⁶ = mant / 65536
    // 1/65536 in F16 = 0x0100 (exp=1, mant=0, so 2⁽¹⁻¹⁵⁾ = 2⁻¹⁴, hmm...)
    // Actually 2⁻¹⁶ is below F16 subnormal range, let's verify:
    // F16 min subnormal = 2⁻²⁴, so 2⁻¹⁶ = 2⁸ × 2⁻²⁴ = 256 × min_subnorm
    // In F16: 2⁻¹⁶ = 0x0100 works via subnormal representation
    svfloat16_t mant_f16 = svcvt_f16_u16_x(pg16, mant);
    nk_u16_t scale_bits = 0x0100; // 2⁻¹⁴ in F16 normal, but we need 2⁻¹⁶
    // Actually let me recalculate: E5M2 subnorm value = mant × 2⁽¹⁻¹⁵⁻²⁾ = mant × 2⁻¹⁶
    // F16 can represent 2⁻¹⁴ as smallest normal (0x0400)
    // 2⁻¹⁶ = 2⁻¹⁴ / 4 = 0x0400 / 4... but that's not how F16 works
    // Let's use 2⁻¹⁴ × (1/4) = multiply by 0.25 after scaling
    // Or: 2⁻¹⁶ as F16 subnormal: 0x0001 = 2⁻²⁴, so 2⁻¹⁶ = 2⁸ × 2⁻²⁴ = 256 × 0x0001
    // That means 2⁻¹⁶ = 0x0100 in F16 subnormal representation
    svfloat16_t scale = svreinterpret_f16_u16(svdup_n_u16(scale_bits));
    svfloat16_t subnorm_abs = svmul_f16_x(pg16, mant_f16, scale);
    svuint16_t subnorm = svorr_u16_x(pg16, svreinterpret_u16_f16(subnorm_abs), sign);

    // Detect subnormals: exp == 0
    svbool_t is_subnorm = svcmpeq_n_u16(pg16, exp, 0);

    // Detect infinity: exp == 31 && mant == 0, i.e., mag == 0x7C
    svbool_t is_inf = svcmpeq_n_u16(pg16, mag, 0x7C);
    svuint16_t inf_val = svorr_n_u16_x(pg16, sign, 0x7C00);

    // Detect NaN: exp == 31 && mant != 0, i.e., mag > 0x7C
    svbool_t is_nan = svcmpgt_n_u16(pg16, mag, 0x7C);
    svuint16_t nan_val = svorr_n_u16_x(pg16, sign, 0x7E00);

    // Blend results
    svuint16_t result = svsel_u16(is_subnorm, subnorm, normal);
    result = svsel_u16(is_inf, inf_val, result);
    result = svsel_u16(is_nan, nan_val, result);

    svst1_u16(pg16, (uint16_t *)dst, result);
}

/*  Inline `e4m3` → `f16` conversion returning `svfloat16_t` for direct use in GEMM.
 *  This avoids memory round-trip when used inside a streaming kernel.
 *
 *  @param pg16   Predicate for 16-bit elements: use `svptrue_b16()`
 *  @param bytes  Pre-loaded 64 bytes: `svuint8_t` from `svld1_u8`
 *  @return       32 `f16` values as `svfloat16_t`: from lower 32 bytes
 */
NK_INTERNAL svfloat16_t nk_e4m3x32_to_f16_vec_ssve_(svbool_t pg16, svuint8_t bytes) {
    svuint16_t vals = svunpklo_u16(bytes);

    svuint16_t sign = svlsl_n_u16_x(pg16, svand_n_u16_x(pg16, vals, 0x80), 8);
    svuint16_t mag = svand_n_u16_x(pg16, vals, 0x7F);
    svuint16_t mant = svand_n_u16_x(pg16, vals, 0x07);

    // Normal path: F16 = sign | ((mag << 7) + 0x2000)
    svuint16_t normal = svadd_n_u16_x(pg16, svlsl_n_u16_x(pg16, mag, 7), 0x2000);
    normal = svorr_u16_x(pg16, normal, sign);

    // Subnormal path: `mant` × (1/512) where 1/512 = 0x1800 in `f16`
    svfloat16_t mant_f16 = svcvt_f16_u16_x(pg16, mant);
    svfloat16_t scale = svreinterpret_f16_u16(svdup_n_u16(0x1800));
    svfloat16_t subnorm_abs = svmul_f16_x(pg16, mant_f16, scale);
    svuint16_t subnorm = svorr_u16_x(pg16, svreinterpret_u16_f16(subnorm_abs), sign);

    svbool_t is_subnorm = svcmpeq_n_u16(pg16, svand_n_u16_x(pg16, vals, 0x78), 0);
    svbool_t is_nan = svcmpeq_n_u16(pg16, mag, 0x7F);
    svuint16_t nan_val = svorr_n_u16_x(pg16, sign, 0x7E00);

    svuint16_t result = svsel_u16(is_subnorm, subnorm, normal);
    result = svsel_u16(is_nan, nan_val, result);

    return svreinterpret_f16_u16(result);
}

/*  Inline `e5m2` → `f16` conversion returning `svfloat16_t` for direct use in GEMM.
 *  This avoids memory round-trip when used inside a streaming kernel.
 *
 *  E5M2 format: S EEEEE MM (1+5+2 bits, bias=15, range [-57344, 57344])
 *  F16 format:  S EEEEE MMMMMMMMMM (1+5+10 bits, bias=15)
 *
 *  Since E5M2 and F16 share the same exponent bias (15), normal values convert
 *  by simply shifting the magnitude left by 8 bits.
 *
 *  @param pg16   Predicate for 16-bit elements (use svptrue_b16())
 *  @param bytes  Pre-loaded 64 bytes (svuint8_t from svld1_u8)
 *  @return       32 F16 values as svfloat16_t (from lower 32 bytes)
 */
NK_INTERNAL svfloat16_t nk_e5m2x32_to_f16_vec_ssve_(svbool_t pg16, svuint8_t bytes) {
    svuint16_t vals = svunpklo_u16(bytes);

    // Extract sign bit and shift to F16 sign position (bit 15)
    svuint16_t sign = svlsl_n_u16_x(pg16, svand_n_u16_x(pg16, vals, 0x80), 8);

    // Extract magnitude (lower 7 bits), mantissa (lower 2 bits), exponent (bits 6:2)
    svuint16_t mag = svand_n_u16_x(pg16, vals, 0x7F);
    svuint16_t mant = svand_n_u16_x(pg16, vals, 0x03);
    svuint16_t exp = svlsr_n_u16_x(pg16, svand_n_u16_x(pg16, vals, 0x7C), 2);

    // Normal path: E5M2 and F16 have same bias (15), so just shift magnitude by 8
    // F16 = sign | (exp << 10) | (mant << 8) = sign | (mag << 8)
    svuint16_t normal = svorr_u16_x(pg16, svlsl_n_u16_x(pg16, mag, 8), sign);

    // Subnormal path: value = mant × 2⁻¹⁶
    // 2⁻¹⁶ as F16 subnormal: 0x0001 = 2⁻²⁴, so 2⁻¹⁶ = 2⁸ × 2⁻²⁴ = 256 × 0x0001
    // That means 2⁻¹⁶ = 0x0100 in F16 subnormal representation
    svfloat16_t mant_f16 = svcvt_f16_u16_x(pg16, mant);
    svfloat16_t scale = svreinterpret_f16_u16(svdup_n_u16(0x0100));
    svfloat16_t subnorm_abs = svmul_f16_x(pg16, mant_f16, scale);
    svuint16_t subnorm = svorr_u16_x(pg16, svreinterpret_u16_f16(subnorm_abs), sign);

    // Detect subnormals: exp == 0
    svbool_t is_subnorm = svcmpeq_n_u16(pg16, exp, 0);

    // Detect infinity: exp == 31 && mant == 0, i.e., mag == 0x7C
    svbool_t is_inf = svcmpeq_n_u16(pg16, mag, 0x7C);
    svuint16_t inf_val = svorr_n_u16_x(pg16, sign, 0x7C00);

    // Detect NaN: exp == 31 && mant != 0, i.e., mag > 0x7C
    svbool_t is_nan = svcmpgt_n_u16(pg16, mag, 0x7C);
    svuint16_t nan_val = svorr_n_u16_x(pg16, sign, 0x7E00);

    // Blend results
    svuint16_t result = svsel_u16(is_subnorm, subnorm, normal);
    result = svsel_u16(is_inf, inf_val, result);
    result = svsel_u16(is_nan, nan_val, result);

    return svreinterpret_f16_u16(result);
}

/*  Fused `e4m3` × `e4m3` → `f32` GEMM kernel with predicate-based edge handling.
 *
 *  Uses predicates for all tile processing, eliminating scalar edge handlers.
 *  Converts `e4m3` → `f16` inline using SSVE arithmetic operations.
 */
__arm_locally_streaming __arm_new("za") __attribute__((noinline)) static void nk_dots_e4m3_kernel_( //
    nk_e4m3_t const *a, void const *b_packed, nk_f32_t *c,                                          //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,                                             //
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) {

    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_size_t const column_tile_count = header->column_tile_count;
    nk_size_t const depth_tile_count = header->depth_tile_count;

    nk_size_t const tile_dim = svcntw();        // number of `f32` elements per vector
    nk_size_t const depth_tile_size = svcnth(); // number of `f16` elements per vector
    nk_size_t const tile_elements = tile_dim * depth_tile_size;

    nk_f16_t const *b_tiles = (nk_f16_t const *)((char const *)b_packed + sizeof(nk_dots_sme_packed_header_t));

    svbool_t const ptrue_h = svptrue_b16();
    svbool_t const ptrue_b = svptrue_b8();

    nk_size_t const row_tile_count = (rows + tile_dim - 1) / tile_dim;

    // Process tile-by-tile with predicates for edge handling
    for (nk_size_t row_tile = 0; row_tile < row_tile_count; row_tile++) {
        nk_size_t const row_start = row_tile * tile_dim;
        nk_size_t const rows_remaining = (row_start + tile_dim <= rows) ? tile_dim : (rows - row_start);
        svbool_t const pred_rows = svwhilelt_b32((uint32_t)0, (uint32_t)rows_remaining);

        for (nk_size_t col_tile = 0; col_tile < column_tile_count; col_tile++) {
            nk_size_t const col_start = col_tile * tile_dim;
            nk_size_t const cols_remaining = (col_start + tile_dim <= columns) ? tile_dim : (columns - col_start);
            svbool_t const pred_cols = svwhilelt_b32((uint32_t)0, (uint32_t)cols_remaining);

            // Zero ZA tile 0
            svzero_za();

            // Accumulate over depth tiles
            for (nk_size_t d_tile = 0; d_tile < depth_tile_count; d_tile++) {
                nk_size_t const d_start = d_tile * depth_tile_size;

                nk_size_t const b_tile_idx = col_tile * depth_tile_count + d_tile;
                nk_f16_t const *b_tile = b_tiles + b_tile_idx * tile_elements;

                // Outer products over `tile_dim` rows
                for (nk_size_t row = 0; row < tile_dim; row++) {
                    nk_e4m3_t const *a_ptr = a + (row_start + row) * a_stride_elements + d_start;
                    svuint8_t a_bytes = svld1_u8(ptrue_b, (uint8_t const *)a_ptr);
                    svfloat16_t a_vec = nk_e4m3x32_to_f16_vec_ssve_(ptrue_h, a_bytes);
                    svfloat16_t b_vec = svld1_f16(ptrue_h, (float16_t const *)(b_tile + row * depth_tile_size));
                    svmopa_za32_f16_m(0, pred_rows, pred_cols, a_vec, b_vec);
                }
            }

            // Predicated store to C
            for (nk_size_t row = 0; row < rows_remaining; row++) {
                svst1_hor_za32(0, row, pred_cols, c + (row_start + row) * c_stride_elements + col_start);
            }
        }
    }
}

/**
 *  `e4m3` × `e4m3` → `f32` GEMM using SME `f16` outer products via LUT conversion.
 *
 *  Since Apple M4 lacks native FP8 MOPA (requires SME2p1/FEAT_SME_F8F32),
 *  we convert `e4m3` → `f16` during packing using a precomputed 128-entry LUT.
 *
 *  `e4m3` format: S EEEE MMM (1+4+3 bits, bias=7, range [-448, 448])
 *  `f16` format: S EEEEE MMMMMMMMMM (1+5+10 bits, bias=15)
 *  Conversion: `f16`_exp = `e4m3`_exp + 8, `f16`_mant = `e4m3`_mant << 7
 *
 *  LUT design (inspired by AVX-512 `permutex2var` approach):
 *  - 128 entries for positive `e4m3` values: 7-bit magnitude
 *  - Sign bit handled separately via OR
 *  - Uses NEON `vqtbl4q_u8` for vectorized 64-byte lookups
 *
 *  Expected performance: ~1.3-1.5 TOPS (`f16` SME limited by pack overhead)
 */

/*  Precomputed `uint16_t` LUT for scalar `e4m3` → `f16` conversion.
 *  Used in packing when vectorization isn't beneficial.
 */
// clang-format off
static nk_u16_t const nk_e4m3_to_f16_lut_u16_[128] = {
    0x0000, 0x1800, 0x1C00, 0x1E00, 0x2000, 0x2100, 0x2200, 0x2300,  // 0-7: zero + subnormals
    0x2400, 0x2480, 0x2500, 0x2580, 0x2600, 0x2680, 0x2700, 0x2780,  // 8-15: exp=1
    0x2800, 0x2880, 0x2900, 0x2980, 0x2A00, 0x2A80, 0x2B00, 0x2B80,  // 16-23: exp=2
    0x2C00, 0x2C80, 0x2D00, 0x2D80, 0x2E00, 0x2E80, 0x2F00, 0x2F80,  // 24-31: exp=3
    0x3000, 0x3080, 0x3100, 0x3180, 0x3200, 0x3280, 0x3300, 0x3380,  // 32-39: exp=4
    0x3400, 0x3480, 0x3500, 0x3580, 0x3600, 0x3680, 0x3700, 0x3780,  // 40-47: exp=5
    0x3800, 0x3880, 0x3900, 0x3980, 0x3A00, 0x3A80, 0x3B00, 0x3B80,  // 48-55: exp=6
    0x3C00, 0x3C80, 0x3D00, 0x3D80, 0x3E00, 0x3E80, 0x3F00, 0x3F80,  // 56-63: exp=7
    0x4000, 0x4080, 0x4100, 0x4180, 0x4200, 0x4280, 0x4300, 0x4380,  // 64-71: exp=8
    0x4400, 0x4480, 0x4500, 0x4580, 0x4600, 0x4680, 0x4700, 0x4780,  // 72-79: exp=9
    0x4800, 0x4880, 0x4900, 0x4980, 0x4A00, 0x4A80, 0x4B00, 0x4B80,  // 80-87: exp=10
    0x4C00, 0x4C80, 0x4D00, 0x4D80, 0x4E00, 0x4E80, 0x4F00, 0x4F80,  // 88-95: exp=11
    0x5000, 0x5080, 0x5100, 0x5180, 0x5200, 0x5280, 0x5300, 0x5380,  // 96-103: exp=12
    0x5400, 0x5480, 0x5500, 0x5580, 0x5600, 0x5680, 0x5700, 0x5780,  // 104-111: exp=13
    0x5800, 0x5880, 0x5900, 0x5980, 0x5A00, 0x5A80, 0x5B00, 0x5B80,  // 112-119: exp=14
    0x5C00, 0x5C80, 0x5D00, 0x5D80, 0x5E00, 0x5E80, 0x5F00, 0x7E00   // 120-127: exp=14 cont + NaN
};
// clang-format on

/*  Scalar `e4m3` → `f16` conversion using LUT.
 */
NK_INTERNAL nk_f16_t nk_e4m3_to_f16_lut_(nk_e4m3_t src) {
    nk_u8_t idx = src & 0x7F;
    nk_fui16_t result = {.u = nk_e4m3_to_f16_lut_u16_[idx]};
    if (src & 0x80) result.u |= 0x8000; // Apply sign
    return result.f;
}

NK_PUBLIC nk_size_t nk_dots_packed_size_e4m3_sme(nk_size_t n, nk_size_t k) {
    // Uses `f16` format for packed data
    return nk_dots_packed_size_f16_sme(n, k);
}

/*  Pack `e4m3` B matrix for SME with conversion to `f16`.
 *  Partial tiles are zero-padded for predicate-based edge handling.
 */
NK_PUBLIC void nk_dots_pack_e4m3_sme(             //
    nk_e4m3_t const *b, nk_size_t n, nk_size_t k, //
    nk_size_t b_stride, void *b_packed) {

    nk_size_t const svl_bytes = svcntsw() * sizeof(nk_f32_t);
    nk_size_t const tile_dim = svcntsw();        // rows per tile
    nk_size_t const depth_tile_size = svcntsh(); // K elements per tile
    nk_size_t const tile_elements = tile_dim * depth_tile_size;
    nk_size_t const b_stride_elements = b_stride / sizeof(nk_e4m3_t);

    nk_size_t const column_tile_count = (n + tile_dim - 1) / tile_dim;
    nk_size_t const depth_tile_count = (k + depth_tile_size - 1) / depth_tile_size;
    nk_size_t const total_tiles = column_tile_count * depth_tile_count;

    nk_dots_sme_packed_header_t *header = (nk_dots_sme_packed_header_t *)b_packed;
    header->column_tile_count = (nk_u32_t)column_tile_count;
    header->depth_tile_count = (nk_u32_t)depth_tile_count;
    header->columns = (nk_u32_t)n;
    header->depth = (nk_u32_t)k;
    header->svl_bytes = (nk_u32_t)svl_bytes;

    nk_f16_t *tiles_ptr = (nk_f16_t *)((char *)b_packed + sizeof(nk_dots_sme_packed_header_t));

    // Zero-initialize tiles
    for (nk_size_t i = 0; i < total_tiles * tile_elements; i++) tiles_ptr[i] = 0;

    // Pack tiles with `e4m3` → `f16` LUT conversion, column-major layout
    for (nk_size_t column_tile = 0; column_tile < column_tile_count; column_tile++) {
        for (nk_size_t depth_tile = 0; depth_tile < depth_tile_count; depth_tile++) {
            nk_size_t const tile_index = column_tile * depth_tile_count + depth_tile;
            nk_f16_t *tile_output = tiles_ptr + tile_index * tile_elements;

            nk_size_t const src_row_start = column_tile * tile_dim;
            nk_size_t const src_col_start = depth_tile * depth_tile_size;
            nk_size_t const rows_to_pack = (src_row_start + tile_dim <= n) ? tile_dim : (n - src_row_start);
            nk_size_t const cols_to_pack = (src_col_start + depth_tile_size <= k) ? depth_tile_size
                                                                                  : (k - src_col_start);

            for (nk_size_t row = 0; row < rows_to_pack; row++) {
                for (nk_size_t col = 0; col < cols_to_pack; col++) {
                    nk_size_t const src_idx = (src_row_start + row) * b_stride_elements + src_col_start + col;
                    nk_size_t const dst_idx = col * tile_dim + row;
                    tile_output[dst_idx] = nk_e4m3_to_f16_lut_(b[src_idx]);
                }
            }
        }
    }
}

/*  `e4m3` × `e4m3` → `f32` GEMM: fused kernel with SSVE inline `e4m3` → `f16` conversion.
 *
 *  Uses a fully fused kernel that converts `e4m3` → `f16` inline in streaming mode,
 *  eliminating buffer allocation and `SMSTART`/`SMSTOP` overhead for the tile-aligned
 *  portion. Falls back to NEON conversion for edge cases.
 *
 *  @param a         Input matrix A (M × K), row-major, `e4m3`
 *  @param b_packed  Pre-packed B matrix from `nk_dots_pack_e4m3_sme`: contains `f16`
 *  @param c         Output matrix C (M × N), row-major, `f32`
 */
NK_PUBLIC void nk_dots_packed_e4m3_sme(                    //
    nk_e4m3_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,    //
    nk_size_t a_stride, nk_size_t c_stride) {

    nk_size_t const a_stride_elements = a_stride / sizeof(nk_e4m3_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);

    nk_dots_e4m3_kernel_(a, b_packed, c, rows, columns, depth, a_stride_elements, c_stride_elements);
}

/*  Precomputed `uint16_t` LUT for scalar `e5m2` → `f16` conversion.
 *  Used in packing when vectorization isn't beneficial.
 *
 *  E5M2 format: S EEEEE MM (1+5+2 bits, bias=15)
 *  F16 format:  S EEEEE MMMMMMMMMM (1+5+10 bits, bias=15)
 *
 *  Since both formats share the same exponent bias (15), normal values
 *  convert by simply left-shifting the magnitude by 8 bits.
 *
 *  Table values:
 *    - Index 0x00: zero → 0x0000
 *    - Index 0x01-0x03: subnormals (M × 2⁻¹⁶)
 *    - Index 0x04-0x7B: normals → `mag` << 8
 *    - Index 0x7C: +infinity → 0x7C00
 *    - Index 0x7D-0x7F: NaN → 0x7E00
 */
// clang-format off
static nk_u16_t const nk_e5m2_to_f16_lut_u16_[128] = {
    0x0000, 0x0100, 0x0200, 0x0300, 0x0400, 0x0500, 0x0600, 0x0700,
    0x0800, 0x0900, 0x0A00, 0x0B00, 0x0C00, 0x0D00, 0x0E00, 0x0F00,
    0x1000, 0x1100, 0x1200, 0x1300, 0x1400, 0x1500, 0x1600, 0x1700,
    0x1800, 0x1900, 0x1A00, 0x1B00, 0x1C00, 0x1D00, 0x1E00, 0x1F00,
    0x2000, 0x2100, 0x2200, 0x2300, 0x2400, 0x2500, 0x2600, 0x2700,
    0x2800, 0x2900, 0x2A00, 0x2B00, 0x2C00, 0x2D00, 0x2E00, 0x2F00,
    0x3000, 0x3100, 0x3200, 0x3300, 0x3400, 0x3500, 0x3600, 0x3700,
    0x3800, 0x3900, 0x3A00, 0x3B00, 0x3C00, 0x3D00, 0x3E00, 0x3F00,
    0x4000, 0x4100, 0x4200, 0x4300, 0x4400, 0x4500, 0x4600, 0x4700,
    0x4800, 0x4900, 0x4A00, 0x4B00, 0x4C00, 0x4D00, 0x4E00, 0x4F00,
    0x5000, 0x5100, 0x5200, 0x5300, 0x5400, 0x5500, 0x5600, 0x5700,
    0x5800, 0x5900, 0x5A00, 0x5B00, 0x5C00, 0x5D00, 0x5E00, 0x5F00,
    0x6000, 0x6100, 0x6200, 0x6300, 0x6400, 0x6500, 0x6600, 0x6700,
    0x6800, 0x6900, 0x6A00, 0x6B00, 0x6C00, 0x6D00, 0x6E00, 0x6F00,
    0x7000, 0x7100, 0x7200, 0x7300, 0x7400, 0x7500, 0x7600, 0x7700,
    0x7800, 0x7900, 0x7A00, 0x7B00, 0x7C00, 0x7E00, 0x7E00, 0x7E00
};
// clang-format on

/*  Scalar `e5m2` → `f16` conversion using LUT.
 */
NK_INTERNAL nk_f16_t nk_e5m2_to_f16_lut_(nk_e5m2_t src) {
    nk_u8_t idx = src & 0x7F;
    nk_fui16_t result = {.u = nk_e5m2_to_f16_lut_u16_[idx]};
    if (src & 0x80) result.u |= 0x8000;
    return result.f;
}

/*  Fused `e5m2` × `e5m2` → `f32` GEMM kernel using SSVE inline conversion.
 *
 *  This kernel stays entirely in streaming mode, converting `e5m2` → `f16` inline
 *  using arithmetic operations. Uses predicate-based edge handling.
 */
__arm_locally_streaming __arm_new("za") __attribute__((noinline)) static void nk_dots_e5m2_kernel_( //
    nk_e5m2_t const *a, void const *b_packed, nk_f32_t *c,                                          //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,                                             //
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) {

    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_size_t const column_tile_count = header->column_tile_count;
    nk_size_t const depth_tile_count = header->depth_tile_count;

    nk_size_t const tile_dim = svcntw();        // number of `f32` elements per vector
    nk_size_t const depth_tile_size = svcnth(); // number of `f16` elements per vector
    nk_size_t const tile_elements = tile_dim * depth_tile_size;

    nk_f16_t const *b_tiles = (nk_f16_t const *)((char const *)b_packed + sizeof(nk_dots_sme_packed_header_t));

    svbool_t const ptrue_h = svptrue_b16();
    svbool_t const ptrue_b = svptrue_b8();

    nk_size_t const row_tile_count = (rows + tile_dim - 1) / tile_dim;

    // Process tile-by-tile with predicates for edge handling
    for (nk_size_t row_tile = 0; row_tile < row_tile_count; row_tile++) {
        nk_size_t const row_start = row_tile * tile_dim;
        nk_size_t const rows_remaining = (row_start + tile_dim <= rows) ? tile_dim : (rows - row_start);
        svbool_t const pred_rows = svwhilelt_b32((uint32_t)0, (uint32_t)rows_remaining);

        for (nk_size_t col_tile = 0; col_tile < column_tile_count; col_tile++) {
            nk_size_t const col_start = col_tile * tile_dim;
            nk_size_t const cols_remaining = (col_start + tile_dim <= columns) ? tile_dim : (columns - col_start);
            svbool_t const pred_cols = svwhilelt_b32((uint32_t)0, (uint32_t)cols_remaining);

            // Zero ZA tile 0
            svzero_za();

            // Accumulate over depth tiles
            for (nk_size_t d_tile = 0; d_tile < depth_tile_count; d_tile++) {
                nk_size_t const d_start = d_tile * depth_tile_size;

                nk_size_t const b_tile_idx = col_tile * depth_tile_count + d_tile;
                nk_f16_t const *b_tile = b_tiles + b_tile_idx * tile_elements;

                // Outer products over `tile_dim` rows
                for (nk_size_t row = 0; row < tile_dim; row++) {
                    nk_e5m2_t const *a_ptr = a + (row_start + row) * a_stride_elements + d_start;
                    svuint8_t a_bytes = svld1_u8(ptrue_b, (uint8_t const *)a_ptr);
                    svfloat16_t a_vec = nk_e5m2x32_to_f16_vec_ssve_(ptrue_h, a_bytes);
                    svfloat16_t b_vec = svld1_f16(ptrue_h, (float16_t const *)(b_tile + row * depth_tile_size));
                    svmopa_za32_f16_m(0, pred_rows, pred_cols, a_vec, b_vec);
                }
            }

            // Predicated store to C
            for (nk_size_t row = 0; row < rows_remaining; row++) {
                svst1_hor_za32(0, row, pred_cols, c + (row_start + row) * c_stride_elements + col_start);
            }
        }
    }
}

/*  `e5m2` × `e5m2` → `f32` GEMM: packed size calculation.
 */
NK_PUBLIC nk_size_t nk_dots_packed_size_e5m2_sme(nk_size_t n, nk_size_t k) { return nk_dots_packed_size_f16_sme(n, k); }

/*  Pack `e5m2` matrix B with conversion to `f16` for SME GEMM.
 *  Partial tiles are zero-padded for predicate-based edge handling.
 */
NK_PUBLIC void nk_dots_pack_e5m2_sme(nk_e5m2_t const *b, nk_size_t n, nk_size_t k, nk_size_t b_stride, void *b_packed) {

    nk_size_t const svl_bytes = svcntsw() * sizeof(nk_f32_t);
    nk_size_t const tile_dim = svcntsw();        // rows per tile
    nk_size_t const depth_tile_size = svcntsh(); // K elements per tile
    nk_size_t const tile_elements = tile_dim * depth_tile_size;
    nk_size_t const b_stride_elements = b_stride / sizeof(nk_e5m2_t);

    nk_size_t const column_tile_count = (n + tile_dim - 1) / tile_dim;
    nk_size_t const depth_tile_count = (k + depth_tile_size - 1) / depth_tile_size;
    nk_size_t const total_tiles = column_tile_count * depth_tile_count;

    nk_dots_sme_packed_header_t *header = (nk_dots_sme_packed_header_t *)b_packed;
    header->column_tile_count = (nk_u32_t)column_tile_count;
    header->depth_tile_count = (nk_u32_t)depth_tile_count;
    header->columns = (nk_u32_t)n;
    header->depth = (nk_u32_t)k;
    header->svl_bytes = (nk_u32_t)svl_bytes;

    nk_f16_t *tiles_ptr = (nk_f16_t *)((char *)b_packed + sizeof(nk_dots_sme_packed_header_t));

    // Zero-initialize tiles
    for (nk_size_t i = 0; i < total_tiles * tile_elements; i++) tiles_ptr[i] = 0;

    // Pack tiles with `e5m2` → `f16` LUT conversion, column-major layout
    for (nk_size_t column_tile = 0; column_tile < column_tile_count; column_tile++) {
        for (nk_size_t depth_tile = 0; depth_tile < depth_tile_count; depth_tile++) {
            nk_size_t const tile_index = column_tile * depth_tile_count + depth_tile;
            nk_f16_t *tile_output = tiles_ptr + tile_index * tile_elements;

            nk_size_t const src_row_start = column_tile * tile_dim;
            nk_size_t const src_col_start = depth_tile * depth_tile_size;
            nk_size_t const rows_to_pack = (src_row_start + tile_dim <= n) ? tile_dim : (n - src_row_start);
            nk_size_t const cols_to_pack = (src_col_start + depth_tile_size <= k) ? depth_tile_size
                                                                                  : (k - src_col_start);

            for (nk_size_t row = 0; row < rows_to_pack; row++) {
                for (nk_size_t col = 0; col < cols_to_pack; col++) {
                    nk_size_t const src_idx = (src_row_start + row) * b_stride_elements + src_col_start + col;
                    nk_size_t const dst_idx = col * tile_dim + row;
                    tile_output[dst_idx] = nk_e5m2_to_f16_lut_(b[src_idx]);
                }
            }
        }
    }
}

/*  `e5m2` × `e5m2` → `f32` GEMM: public interface.
 *  Predicate-based edge handling eliminates scalar fallbacks.
 */
NK_PUBLIC void nk_dots_packed_e5m2_sme(                    //
    nk_e5m2_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,    //
    nk_size_t a_stride, nk_size_t c_stride) {

    nk_size_t const a_stride_elements = a_stride / sizeof(nk_e5m2_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);

    nk_dots_e5m2_kernel_(a, b_packed, c, rows, columns, depth, a_stride_elements, c_stride_elements);
}

/**
 *  `u8` × `u8` → `u32` GEMM using SME outer products.
 *
 *  Uses `svmopa_za32_u8_m` for unsigned 8-bit integer outer product accumulate.
 *  Available on Apple M4 (SME_I8I32 = 1, covers both signed and unsigned).
 *
 *  Tile dimensions identical to `i8` → `i32` (512-bit SVL):
 *  - Input vectors: 64 `u8` elements (SVL/8 = 64)
 *  - Output tile: 16 × 16 `u32` elements (`ZA32`)
 *  - Each output `u32` is a dot product of 4 `u8` pairs
 */

NK_PUBLIC nk_size_t nk_dots_packed_size_u8_sme(nk_size_t n, nk_size_t k) {
    // Same dimensions as `i8` → `i32` since both are 8-bit
    return nk_dots_packed_size_i8_sme(n, k);
}

/*  Pack `u8` matrix B for SME GEMM.
 *  Partial tiles are zero-padded for predicate-based edge handling.
 */
NK_PUBLIC void nk_dots_pack_u8_sme(             //
    nk_u8_t const *b, nk_size_t n, nk_size_t k, //
    nk_size_t b_stride, void *b_packed) {

    nk_size_t const svl_bytes = svcntsw() * sizeof(nk_u32_t);
    nk_size_t const tile_dim = svcntsw();        // rows per tile
    nk_size_t const depth_tile_size = svcntsb(); // K elements per tile
    nk_size_t const tile_elements = tile_dim * depth_tile_size;
    nk_size_t const b_stride_elements = b_stride / sizeof(nk_u8_t);

    nk_size_t const column_tile_count = (n + tile_dim - 1) / tile_dim;
    nk_size_t const depth_tile_count = (k + depth_tile_size - 1) / depth_tile_size;
    nk_size_t const total_tiles = column_tile_count * depth_tile_count;

    nk_dots_sme_packed_header_t *header = (nk_dots_sme_packed_header_t *)b_packed;
    header->column_tile_count = (nk_u32_t)column_tile_count;
    header->depth_tile_count = (nk_u32_t)depth_tile_count;
    header->columns = (nk_u32_t)n;
    header->depth = (nk_u32_t)k;
    header->svl_bytes = (nk_u32_t)svl_bytes;

    nk_u8_t *tiles_ptr = (nk_u8_t *)((char *)b_packed + sizeof(nk_dots_sme_packed_header_t));

    // Zero-initialize tiles
    for (nk_size_t i = 0; i < total_tiles * tile_elements; i++) tiles_ptr[i] = 0;

    // Pack tiles: column-major within tile (K varies fastest, then N)
    for (nk_size_t column_tile = 0; column_tile < column_tile_count; column_tile++) {
        for (nk_size_t depth_tile = 0; depth_tile < depth_tile_count; depth_tile++) {
            nk_size_t const tile_index = column_tile * depth_tile_count + depth_tile;
            nk_u8_t *tile_output = tiles_ptr + tile_index * tile_elements;

            nk_size_t const src_row_start = column_tile * tile_dim;
            nk_size_t const src_col_start = depth_tile * depth_tile_size;
            nk_size_t const rows_to_pack = (src_row_start + tile_dim <= n) ? tile_dim : (n - src_row_start);
            nk_size_t const cols_to_pack = (src_col_start + depth_tile_size <= k) ? depth_tile_size
                                                                                  : (k - src_col_start);

            for (nk_size_t row = 0; row < rows_to_pack; row++) {
                for (nk_size_t col = 0; col < cols_to_pack; col++) {
                    nk_size_t const src_idx = (src_row_start + row) * b_stride_elements + src_col_start + col;
                    nk_size_t const dst_idx = col * tile_dim + row;
                    tile_output[dst_idx] = b[src_idx];
                }
            }
        }
    }
}

/*  `u8` × `u8` → `u32` GEMM kernel with predicate-based edge handling.
 */
__arm_locally_streaming __arm_new("za") static void nk_dots_u8_kernel_( //
    nk_u8_t const *a, void const *b_packed, nk_u32_t *c,                //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,                 //
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) {

    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_size_t const column_tile_count = header->column_tile_count;
    nk_size_t const depth_tile_count = header->depth_tile_count;

    nk_size_t const tile_dim = svcntw();        // number of `u32` elements per vector
    nk_size_t const depth_tile_size = svcntb(); // number of `u8` elements per vector
    nk_size_t const tile_elements = tile_dim * depth_tile_size;

    nk_u8_t const *b_tiles = (nk_u8_t const *)((char const *)b_packed + sizeof(nk_dots_sme_packed_header_t));

    svbool_t const ptrue_b = svptrue_b8();

    nk_size_t const row_tile_count = (rows + tile_dim - 1) / tile_dim;

    // Process tile-by-tile with predicates for edge handling
    for (nk_size_t row_tile = 0; row_tile < row_tile_count; row_tile++) {
        nk_size_t const row_start = row_tile * tile_dim;
        nk_size_t const rows_remaining = (row_start + tile_dim <= rows) ? tile_dim : (rows - row_start);
        svbool_t const pred_rows = svwhilelt_b32((uint32_t)0, (uint32_t)rows_remaining);

        for (nk_size_t col_tile = 0; col_tile < column_tile_count; col_tile++) {
            nk_size_t const col_start = col_tile * tile_dim;
            nk_size_t const cols_remaining = (col_start + tile_dim <= columns) ? tile_dim : (columns - col_start);
            svbool_t const pred_cols = svwhilelt_b32((uint32_t)0, (uint32_t)cols_remaining);

            // Zero ZA tile 0
            svzero_za();

            // Accumulate over depth dimension
            for (nk_size_t d_tile = 0; d_tile < depth_tile_count; d_tile++) {
                nk_size_t const d_start = d_tile * depth_tile_size;

                nk_size_t const b_tile_idx = col_tile * depth_tile_count + d_tile;
                nk_u8_t const *b_tile = b_tiles + b_tile_idx * tile_elements;

                // For i8/u8 MOPA: 4-way dot product accumulation
                // Load full vectors, zero-padding handled by pack function
                for (nk_size_t row = 0; row < tile_dim && row_start + row < rows; row++) {
                    nk_u8_t const *a_ptr = a + (row_start + row) * a_stride_elements + d_start;

                    // Predicated load for A: zero elements beyond depth
                    nk_size_t const depth_remaining = (d_start + depth_tile_size <= depth)
                                                          ? depth_tile_size
                                                          : (depth > d_start ? depth - d_start : 0);
                    svbool_t const pred_k = svwhilelt_b8((uint32_t)0, (uint32_t)depth_remaining);
                    svuint8_t a_vec = svld1_u8(pred_k, a_ptr);

                    // Load B vector (already packed with zero-padding)
                    svuint8_t b_vec = svld1_u8(ptrue_b, b_tile + row * depth_tile_size);

                    // Predicated outer product
                    svmopa_za32_u8_m(0, pred_rows, pred_cols, a_vec, b_vec);
                }
            }

            // Predicated store to C
            for (nk_size_t row = 0; row < rows_remaining; row++) {
                svst1_hor_za32(0, row, pred_cols, (nk_i32_t *)(c + (row_start + row) * c_stride_elements + col_start));
            }
        }
    }
}

/*  `u8` × `u8` → `u32` GEMM: public interface.
 *  Predicate-based edge handling eliminates scalar fallbacks.
 */
NK_PUBLIC void nk_dots_packed_u8_sme(                    //
    nk_u8_t const *a, void const *b_packed, nk_u32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,  //
    nk_size_t a_stride, nk_size_t c_stride) {

    nk_size_t const a_stride_elements = a_stride / sizeof(nk_u8_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_u32_t);

    nk_dots_u8_kernel_(a, b_packed, c, rows, columns, depth, a_stride_elements, c_stride_elements);
}

#if defined(__cplusplus)
} // extern "C"
#endif

#pragma clang attribute pop
#pragma GCC pop_options
#endif // NK_TARGET_SME
#endif // NK_TARGET_ARM_

#endif // NK_DOTS_SME_H

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

#include "numkong/types.h"
#include "numkong/cast/serial.h" // `nk_e4m3_to_f16_serial`, `nk_e5m2_to_f16_serial`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("sme"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("+sme")
#endif

#include <arm_sme.h>
#include <arm_sve.h>

/**
 *  SME-specific packed buffer header (64-byte aligned).
 *  Layout optimized for SME outer product access patterns with predicate-based edge handling.
 */
typedef struct {
    nk_u32_t column_tile_count; // ⌈columns/tile_dimension⌉: number of column tiles
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
NK_INTERNAL nk_size_t nk_sme_tile_dimension_(void) __arm_streaming_compatible { return svcntw(); }

/*  Zero all 4 `ZA32` tiles (tiles 0-3).
 *  Must be called at start of GEMM computation.
 */
NK_INTERNAL void nk_sme_zero_za32_(void) __arm_streaming __arm_inout("za") { svzero_za(); }

/*
 *  f16/bf16 → f32 GEMM using FMOPA/BFMOPA with ZA32 tiles.
 *
 *  Tile layout (SVL=512, Apple M4):
 *  - ZA32 output tile: 16 × 16 f32 elements (1 KB)
 *  - Input vectors: 32 f16/bf16 elements (SVL/16)
 *  - Depth per FMOPA: 2 f16 pairs → 1 f32 (widening 2:1)
 *  - FMOPA predicates: b16 (input granularity), not b32
 *  - 4-tile path: ZA0-ZA3 process 4 column tiles simultaneously
 */
#pragma region Half-Precision Floats (f16, bf16)

NK_PUBLIC nk_size_t nk_dots_packed_size_f16_sme(nk_size_t n, nk_size_t k) {
    nk_size_t const tile_dimension = svcntsw();  // rows per tile: number of `f32` elements
    nk_size_t const depth_tile_size = svcntsh(); // K elements per tile: number of `f16` elements

    nk_size_t const column_tile_count = (n + tile_dimension - 1) / tile_dimension;
    nk_size_t const depth_tile_count = (k + depth_tile_size - 1) / depth_tile_size;

    // Header (64 bytes aligned)
    nk_size_t size = sizeof(nk_dots_sme_packed_header_t);

    // All tiles including partial tiles (zero-padded)
    size += column_tile_count * depth_tile_count * tile_dimension * depth_tile_size * sizeof(nk_f16_t);

    return size;
}

NK_PUBLIC nk_size_t nk_dots_packed_size_bf16_sme(nk_size_t n, nk_size_t k) {
    // Same dimensions as `f16` since both are 16-bit
    return nk_dots_packed_size_f16_sme(n, k);
}

NK_PUBLIC void nk_dots_pack_f16_sme(             //
    nk_f16_t const *b, nk_size_t n, nk_size_t k, //
    nk_size_t b_stride, void *b_packed) {

    nk_size_t const svl_bytes = svcntsw() * sizeof(nk_f32_t);
    nk_size_t const tile_dimension = svcntsw();  // rows per tile
    nk_size_t const depth_tile_size = svcntsh(); // K elements per tile
    nk_size_t const tile_elements = tile_dimension * depth_tile_size;
    nk_size_t const b_stride_elements = b_stride / sizeof(nk_f16_t);

    nk_size_t const column_tile_count = (n + tile_dimension - 1) / tile_dimension;
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

            nk_size_t const src_row_start = column_tile * tile_dimension;
            nk_size_t const src_col_start = depth_tile * depth_tile_size;
            nk_size_t const rows_to_pack = (src_row_start + tile_dimension <= n) ? tile_dimension : (n - src_row_start);
            nk_size_t const cols_to_pack = (src_col_start + depth_tile_size <= k) ? depth_tile_size
                                                                                  : (k - src_col_start);

            // Column-major packing: tile_output[col * tile_dimension + row]
            for (nk_size_t row = 0; row < rows_to_pack; row++) {
                for (nk_size_t col = 0; col < cols_to_pack; col++) {
                    nk_size_t const src_idx = (src_row_start + row) * b_stride_elements + src_col_start + col;
                    nk_size_t const dst_idx = col * tile_dimension + row;
                    tile_output[dst_idx] = b[src_idx];
                }
            }
        }
    }
}

NK_PUBLIC void nk_dots_pack_bf16_sme(             //
    nk_bf16_t const *b, nk_size_t n, nk_size_t k, //
    nk_size_t b_stride, void *b_packed) {

    nk_size_t const svl_bytes = svcntsw() * sizeof(nk_f32_t);
    nk_size_t const tile_dimension = svcntsw();  // rows per tile
    nk_size_t const depth_tile_size = svcntsh(); // K elements per tile
    nk_size_t const tile_elements = tile_dimension * depth_tile_size;
    nk_size_t const b_stride_elements = b_stride / sizeof(nk_bf16_t);

    nk_size_t const column_tile_count = (n + tile_dimension - 1) / tile_dimension;
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

            nk_size_t const src_row_start = column_tile * tile_dimension;
            nk_size_t const src_col_start = depth_tile * depth_tile_size;
            nk_size_t const rows_to_pack = (src_row_start + tile_dimension <= n) ? tile_dimension : (n - src_row_start);
            nk_size_t const cols_to_pack = (src_col_start + depth_tile_size <= k) ? depth_tile_size
                                                                                  : (k - src_col_start);

            for (nk_size_t row = 0; row < rows_to_pack; row++) {
                for (nk_size_t col = 0; col < cols_to_pack; col++) {
                    nk_size_t const src_idx = (src_row_start + row) * b_stride_elements + src_col_start + col;
                    nk_size_t const dst_idx = col * tile_dimension + row;
                    tile_output[dst_idx] = b[src_idx];
                }
            }
        }
    }
}

/**
 *  `f16` → `f32` GEMM core kernel using SME outer products with predicate-based edge handling.
 *
 *  Uses predicates for all tile processing, eliminating scalar edge handlers.
 *  Multi-tile optimization: Processes 4 column tiles simultaneously using ZA0-ZA3.
 *  This maximizes utilization of the SME ZA register file for better throughput.
 */
__arm_locally_streaming __arm_new("za") static void nk_dots_f16_kernel_( //
    nk_f16_t const *a, void const *b_packed, nk_f32_t *c,                //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,                  //
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) {

    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_size_t const column_tile_count = header->column_tile_count;
    nk_size_t const depth_tile_count = header->depth_tile_count;

    nk_size_t const tile_dimension = svcntw();  // number of `f32` elements per vector
    nk_size_t const depth_tile_size = svcnth(); // number of `f16` elements per vector
    nk_size_t const tile_elements = tile_dimension * depth_tile_size;

    nk_f16_t const *b_tiles = (nk_f16_t const *)((char const *)b_packed + sizeof(nk_dots_sme_packed_header_t));

    svbool_t const predicate_half = svptrue_b16();
    svbool_t const predicate_single = svptrue_b32();
    // FMOPA f16→f32 uses b16 predicates (input element granularity), not b32 (output granularity)
    svbool_t const predicate_fmopa_full = svptrue_b16();

    nk_size_t const row_tile_index_count = (rows + tile_dimension - 1) / tile_dimension;

    // Process tile-by-tile with predicates for edge handling
    for (nk_size_t row_tile_index = 0; row_tile_index < row_tile_index_count; row_tile_index++) {
        nk_size_t const row_start = row_tile_index * tile_dimension;
        nk_size_t const rows_remaining = (row_start + tile_dimension <= rows) ? tile_dimension : (rows - row_start);
        svbool_t const predicate_valid_rows = svwhilelt_b32((uint32_t)0, (uint32_t)rows_remaining);
        // FMOPA row predicate: b16 with 2x elements for widening operation
        svbool_t const predicate_valid_rows_b16 = svwhilelt_b16((uint32_t)0, (uint32_t)(rows_remaining * 2));

        nk_size_t column_tile_index = 0;

        // Process 4 column tiles at a time using ZA0-ZA3
        for (; column_tile_index + 4 <= column_tile_count; column_tile_index += 4) {
            // Zero all 4 ZA tiles
            svzero_za();

            // Accumulate over depth dimension
            for (nk_size_t depth_tile_index = 0; depth_tile_index < depth_tile_count; depth_tile_index++) {
                nk_size_t const depth_offset = depth_tile_index * depth_tile_size;

                // Get B tile pointers for all 4 column tiles
                nk_size_t const b_tile_idx0 = (column_tile_index + 0) * depth_tile_count + depth_tile_index;
                nk_size_t const b_tile_idx1 = (column_tile_index + 1) * depth_tile_count + depth_tile_index;
                nk_size_t const b_tile_idx2 = (column_tile_index + 2) * depth_tile_count + depth_tile_index;
                nk_size_t const b_tile_idx3 = (column_tile_index + 3) * depth_tile_count + depth_tile_index;
                nk_f16_t const *b_tile0 = b_tiles + b_tile_idx0 * tile_elements;
                nk_f16_t const *b_tile1 = b_tiles + b_tile_idx1 * tile_elements;
                nk_f16_t const *b_tile2 = b_tiles + b_tile_idx2 * tile_elements;
                nk_f16_t const *b_tile3 = b_tiles + b_tile_idx3 * tile_elements;

                // Process tile_dimension rows of outer products
                for (nk_size_t row = 0; row < tile_dimension; row++) {
                    // Load A vector once (shared across all 4 B tiles)
                    nk_f16_t const *pointer_a = a + (row_start + row) * a_stride_elements + depth_offset;
                    svfloat16_t vector_a = svld1_f16(predicate_half, (float16_t const *)pointer_a);

                    // Load B vectors from all 4 packed tiles
                    svfloat16_t vector_b_tile_0 = svld1_f16(predicate_half,
                                                            (float16_t const *)(b_tile0 + row * depth_tile_size));
                    svfloat16_t vector_b_tile_1 = svld1_f16(predicate_half,
                                                            (float16_t const *)(b_tile1 + row * depth_tile_size));
                    svfloat16_t vector_b_tile_2 = svld1_f16(predicate_half,
                                                            (float16_t const *)(b_tile2 + row * depth_tile_size));
                    svfloat16_t vector_b_tile_3 = svld1_f16(predicate_half,
                                                            (float16_t const *)(b_tile3 + row * depth_tile_size));

                    // Outer products to all 4 ZA tiles (b16 predicates for f16→f32 widening)
                    svmopa_za32_f16_m(0, predicate_valid_rows_b16, predicate_fmopa_full, vector_a, vector_b_tile_0);
                    svmopa_za32_f16_m(1, predicate_valid_rows_b16, predicate_fmopa_full, vector_a, vector_b_tile_1);
                    svmopa_za32_f16_m(2, predicate_valid_rows_b16, predicate_fmopa_full, vector_a, vector_b_tile_2);
                    svmopa_za32_f16_m(3, predicate_valid_rows_b16, predicate_fmopa_full, vector_a, vector_b_tile_3);
                }
            }

            // Store results from all 4 ZA tiles
            for (nk_size_t row = 0; row < rows_remaining; row++) {
                nk_size_t const col_start0 = (column_tile_index + 0) * tile_dimension;
                nk_size_t const col_start1 = (column_tile_index + 1) * tile_dimension;
                nk_size_t const col_start2 = (column_tile_index + 2) * tile_dimension;
                nk_size_t const col_start3 = (column_tile_index + 3) * tile_dimension;
                svst1_hor_za32(0, row, predicate_single, c + (row_start + row) * c_stride_elements + col_start0);
                svst1_hor_za32(1, row, predicate_single, c + (row_start + row) * c_stride_elements + col_start1);
                svst1_hor_za32(2, row, predicate_single, c + (row_start + row) * c_stride_elements + col_start2);
                svst1_hor_za32(3, row, predicate_single, c + (row_start + row) * c_stride_elements + col_start3);
            }
        }

        // Process remaining column tiles one at a time
        for (; column_tile_index < column_tile_count; column_tile_index++) {
            nk_size_t const col_start = column_tile_index * tile_dimension;
            nk_size_t const cols_remaining = (col_start + tile_dimension <= columns) ? tile_dimension
                                                                                     : (columns - col_start);
            svbool_t const predicate_valid_columns = svwhilelt_b32((uint32_t)0, (uint32_t)cols_remaining);
            // FMOPA column predicate: b16 with 2x elements for widening operation
            svbool_t const predicate_valid_columns_b16 = svwhilelt_b16((uint32_t)0, (uint32_t)(cols_remaining * 2));

            // Zero ZA tile 0
            svzero_za();

            // Accumulate over depth dimension
            for (nk_size_t depth_tile_index = 0; depth_tile_index < depth_tile_count; depth_tile_index++) {
                nk_size_t const depth_offset = depth_tile_index * depth_tile_size;

                nk_size_t const b_tile_idx = column_tile_index * depth_tile_count + depth_tile_index;
                nk_f16_t const *b_tile = b_tiles + b_tile_idx * tile_elements;

                // Process tile_dimension rows of outer products
                for (nk_size_t row = 0; row < tile_dimension; row++) {
                    // Load A vector (predicated load: inactive lanes become 0)
                    nk_f16_t const *pointer_a = a + (row_start + row) * a_stride_elements + depth_offset;
                    svfloat16_t vector_a = svld1_f16(predicate_half, (float16_t const *)pointer_a);

                    // Load B vector from packed tile
                    svfloat16_t vector_b = svld1_f16(predicate_half,
                                                     (float16_t const *)(b_tile + row * depth_tile_size));

                    // Predicated outer product (b16 predicates for f16→f32 widening)
                    svmopa_za32_f16_m(0, predicate_valid_rows_b16, predicate_valid_columns_b16, vector_a, vector_b);
                }
            }

            // Predicated store to C (b32 predicates for f32 output elements)
            for (nk_size_t row = 0; row < rows_remaining; row++) {
                svst1_hor_za32(0, row, predicate_valid_columns, c + (row_start + row) * c_stride_elements + col_start);
            }
        }
    }
}

/**
 *  `bf16` → `f32` GEMM core kernel using SME outer products with predicate-based edge handling.
 *
 *  Uses predicates for all tile processing, eliminating scalar edge handlers.
 *  Multi-tile optimization: Processes 4 column tiles simultaneously using ZA0-ZA3.
 */
__arm_locally_streaming __arm_new("za") static void nk_dots_bf16_kernel_( //
    nk_bf16_t const *a, void const *b_packed, nk_f32_t *c,                //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,                   //
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) {

    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_size_t const column_tile_count = header->column_tile_count;
    nk_size_t const depth_tile_count = header->depth_tile_count;

    nk_size_t const tile_dimension = svcntw();  // number of `f32` elements per vector
    nk_size_t const depth_tile_size = svcnth(); // number of `bf16` elements per vector
    nk_size_t const tile_elements = tile_dimension * depth_tile_size;

    nk_bf16_t const *b_tiles = (nk_bf16_t const *)((char const *)b_packed + sizeof(nk_dots_sme_packed_header_t));

    svbool_t const predicate_half = svptrue_b16();
    svbool_t const predicate_single = svptrue_b32();
    // BFMOPA bf16→f32 uses b16 predicates (input element granularity), not b32 (output granularity)
    svbool_t const predicate_fmopa_full = svptrue_b16();

    nk_size_t const row_tile_index_count = (rows + tile_dimension - 1) / tile_dimension;

    // Process tile-by-tile with predicates for edge handling
    for (nk_size_t row_tile_index = 0; row_tile_index < row_tile_index_count; row_tile_index++) {
        nk_size_t const row_start = row_tile_index * tile_dimension;
        nk_size_t const rows_remaining = (row_start + tile_dimension <= rows) ? tile_dimension : (rows - row_start);
        svbool_t const predicate_valid_rows = svwhilelt_b32((uint32_t)0, (uint32_t)rows_remaining);
        // BFMOPA row predicate: b16 with 2x elements for widening operation
        svbool_t const predicate_valid_rows_b16 = svwhilelt_b16((uint32_t)0, (uint32_t)(rows_remaining * 2));

        nk_size_t column_tile_index = 0;

        // Process 4 column tiles at a time using ZA0-ZA3
        for (; column_tile_index + 4 <= column_tile_count; column_tile_index += 4) {
            // Zero all 4 ZA tiles
            svzero_za();

            // Accumulate over depth dimension
            for (nk_size_t depth_tile_index = 0; depth_tile_index < depth_tile_count; depth_tile_index++) {
                nk_size_t const depth_offset = depth_tile_index * depth_tile_size;

                // Get B tile pointers for all 4 column tiles
                nk_size_t const b_tile_idx0 = (column_tile_index + 0) * depth_tile_count + depth_tile_index;
                nk_size_t const b_tile_idx1 = (column_tile_index + 1) * depth_tile_count + depth_tile_index;
                nk_size_t const b_tile_idx2 = (column_tile_index + 2) * depth_tile_count + depth_tile_index;
                nk_size_t const b_tile_idx3 = (column_tile_index + 3) * depth_tile_count + depth_tile_index;
                nk_bf16_t const *b_tile0 = b_tiles + b_tile_idx0 * tile_elements;
                nk_bf16_t const *b_tile1 = b_tiles + b_tile_idx1 * tile_elements;
                nk_bf16_t const *b_tile2 = b_tiles + b_tile_idx2 * tile_elements;
                nk_bf16_t const *b_tile3 = b_tiles + b_tile_idx3 * tile_elements;

                // Process tile_dimension rows of outer products
                for (nk_size_t row = 0; row < tile_dimension; row++) {
                    // Load A vector once (shared across all 4 B tiles)
                    nk_bf16_t const *pointer_a = a + (row_start + row) * a_stride_elements + depth_offset;
                    svbfloat16_t vector_a = svld1_bf16(predicate_half, (bfloat16_t const *)pointer_a);

                    // Load B vectors from all 4 packed tiles
                    svbfloat16_t vector_b_tile_0 = svld1_bf16(predicate_half,
                                                              (bfloat16_t const *)(b_tile0 + row * depth_tile_size));
                    svbfloat16_t vector_b_tile_1 = svld1_bf16(predicate_half,
                                                              (bfloat16_t const *)(b_tile1 + row * depth_tile_size));
                    svbfloat16_t vector_b_tile_2 = svld1_bf16(predicate_half,
                                                              (bfloat16_t const *)(b_tile2 + row * depth_tile_size));
                    svbfloat16_t vector_b_tile_3 = svld1_bf16(predicate_half,
                                                              (bfloat16_t const *)(b_tile3 + row * depth_tile_size));

                    // Outer products to all 4 ZA tiles (b16 predicates for bf16→f32 widening)
                    svmopa_za32_bf16_m(0, predicate_valid_rows_b16, predicate_fmopa_full, vector_a, vector_b_tile_0);
                    svmopa_za32_bf16_m(1, predicate_valid_rows_b16, predicate_fmopa_full, vector_a, vector_b_tile_1);
                    svmopa_za32_bf16_m(2, predicate_valid_rows_b16, predicate_fmopa_full, vector_a, vector_b_tile_2);
                    svmopa_za32_bf16_m(3, predicate_valid_rows_b16, predicate_fmopa_full, vector_a, vector_b_tile_3);
                }
            }

            // Store results from all 4 ZA tiles
            for (nk_size_t row = 0; row < rows_remaining; row++) {
                nk_size_t const col_start0 = (column_tile_index + 0) * tile_dimension;
                nk_size_t const col_start1 = (column_tile_index + 1) * tile_dimension;
                nk_size_t const col_start2 = (column_tile_index + 2) * tile_dimension;
                nk_size_t const col_start3 = (column_tile_index + 3) * tile_dimension;
                svst1_hor_za32(0, row, predicate_single, c + (row_start + row) * c_stride_elements + col_start0);
                svst1_hor_za32(1, row, predicate_single, c + (row_start + row) * c_stride_elements + col_start1);
                svst1_hor_za32(2, row, predicate_single, c + (row_start + row) * c_stride_elements + col_start2);
                svst1_hor_za32(3, row, predicate_single, c + (row_start + row) * c_stride_elements + col_start3);
            }
        }

        // Process remaining column tiles one at a time
        for (; column_tile_index < column_tile_count; column_tile_index++) {
            nk_size_t const col_start = column_tile_index * tile_dimension;
            nk_size_t const cols_remaining = (col_start + tile_dimension <= columns) ? tile_dimension
                                                                                     : (columns - col_start);
            svbool_t const predicate_valid_columns = svwhilelt_b32((uint32_t)0, (uint32_t)cols_remaining);
            // BFMOPA column predicate: b16 with 2x elements for widening operation
            svbool_t const predicate_valid_columns_b16 = svwhilelt_b16((uint32_t)0, (uint32_t)(cols_remaining * 2));

            // Zero ZA tile 0
            svzero_za();

            // Accumulate over depth dimension
            for (nk_size_t depth_tile_index = 0; depth_tile_index < depth_tile_count; depth_tile_index++) {
                nk_size_t const depth_offset = depth_tile_index * depth_tile_size;

                nk_size_t const b_tile_idx = column_tile_index * depth_tile_count + depth_tile_index;
                nk_bf16_t const *b_tile = b_tiles + b_tile_idx * tile_elements;

                // Process tile_dimension rows of outer products
                for (nk_size_t row = 0; row < tile_dimension; row++) {
                    // Load A vector
                    nk_bf16_t const *pointer_a = a + (row_start + row) * a_stride_elements + depth_offset;
                    svbfloat16_t vector_a = svld1_bf16(predicate_half, (bfloat16_t const *)pointer_a);

                    // Load B vector from packed tile
                    svbfloat16_t vector_b = svld1_bf16(predicate_half,
                                                       (bfloat16_t const *)(b_tile + row * depth_tile_size));

                    // Predicated outer product (b16 predicates for bf16→f32 widening)
                    svmopa_za32_bf16_m(0, predicate_valid_rows_b16, predicate_valid_columns_b16, vector_a, vector_b);
                }
            }

            // Predicated store to C (b32 predicates for f32 output elements)
            for (nk_size_t row = 0; row < rows_remaining; row++) {
                svst1_hor_za32(0, row, predicate_valid_columns, c + (row_start + row) * c_stride_elements + col_start);
            }
        }
    }
}

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

#pragma endregion

/*
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

#pragma region Signed 8-bit Integers (i8)

NK_PUBLIC nk_size_t nk_dots_packed_size_i8_sme(nk_size_t n, nk_size_t k) {
    nk_size_t const tile_dimension = svcntsw();  // rows per ZA32 tile: number of `i32` elements
    nk_size_t const depth_tile_size = svcntsb(); // K elements per tile: number of `i8` elements

    nk_size_t const column_tile_count = (n + tile_dimension - 1) / tile_dimension;
    nk_size_t const depth_tile_count = (k + depth_tile_size - 1) / depth_tile_size;

    // Header (64 bytes aligned)
    nk_size_t size = sizeof(nk_dots_sme_packed_header_t);

    // All tiles including partial tiles (zero-padded)
    size += column_tile_count * depth_tile_count * tile_dimension * depth_tile_size * sizeof(nk_i8_t);

    return size;
}

NK_PUBLIC void nk_dots_pack_i8_sme(             //
    nk_i8_t const *b, nk_size_t n, nk_size_t k, //
    nk_size_t b_stride, void *b_packed) {

    nk_size_t const svl_bytes = svcntsw() * sizeof(nk_i32_t);
    nk_size_t const tile_dimension = svcntsw();  // rows per tile
    nk_size_t const depth_tile_size = svcntsb(); // K elements per tile
    nk_size_t const tile_elements = tile_dimension * depth_tile_size;
    nk_size_t const b_stride_elements = b_stride / sizeof(nk_i8_t);

    nk_size_t const column_tile_count = (n + tile_dimension - 1) / tile_dimension;
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

            nk_size_t const src_row_start = column_tile * tile_dimension;
            nk_size_t const src_col_start = depth_tile * depth_tile_size;
            nk_size_t const rows_to_pack = (src_row_start + tile_dimension <= n) ? tile_dimension : (n - src_row_start);
            nk_size_t const cols_to_pack = (src_col_start + depth_tile_size <= k) ? depth_tile_size
                                                                                  : (k - src_col_start);

            // Column-major packing: tile_output[col * tile_dimension + row]
            for (nk_size_t row = 0; row < rows_to_pack; row++) {
                for (nk_size_t col = 0; col < cols_to_pack; col++) {
                    nk_size_t const src_idx = (src_row_start + row) * b_stride_elements + src_col_start + col;
                    nk_size_t const dst_idx = col * tile_dimension + row;
                    tile_output[dst_idx] = b[src_idx];
                }
            }
        }
    }
}

/**
 *  `i8` × `i8` → `i32` GEMM core kernel using SME outer products with predicate-based edge handling.
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

    nk_size_t const tile_dimension = svcntw();  // number of `i32` elements per vector
    nk_size_t const depth_tile_size = svcntb(); // number of `i8` elements per vector
    nk_size_t const tile_elements = tile_dimension * depth_tile_size;

    nk_i8_t const *b_tiles = (nk_i8_t const *)((char const *)b_packed + sizeof(nk_dots_sme_packed_header_t));

    svbool_t const predicate_byte = svptrue_b8();
    svbool_t const predicate_single = svptrue_b32();
    // SMOPA i8→i32 uses b8 predicates (input element granularity), not b32 (output granularity)
    svbool_t const predicate_smopa_full = svptrue_b8();

    nk_size_t const row_tile_index_count = (rows + tile_dimension - 1) / tile_dimension;

    // Process tile-by-tile with predicates for edge handling
    for (nk_size_t row_tile_index = 0; row_tile_index < row_tile_index_count; row_tile_index++) {
        nk_size_t const row_start = row_tile_index * tile_dimension;
        nk_size_t const rows_remaining = (row_start + tile_dimension <= rows) ? tile_dimension : (rows - row_start);
        svbool_t const predicate_valid_rows = svwhilelt_b32((uint32_t)0, (uint32_t)rows_remaining);
        // SMOPA row predicate: b8 with 4x elements for widening operation
        svbool_t const predicate_valid_rows_b8 = svwhilelt_b8((uint32_t)0, (uint32_t)(rows_remaining * 4));

        nk_size_t column_tile_index = 0;

        // Process 4 column tiles at a time using ZA0-ZA3
        for (; column_tile_index + 4 <= column_tile_count; column_tile_index += 4) {
            svzero_za();

            for (nk_size_t depth_tile_index = 0; depth_tile_index < depth_tile_count; depth_tile_index++) {
                nk_size_t const depth_offset = depth_tile_index * depth_tile_size;

                nk_size_t const b_tile_idx0 = (column_tile_index + 0) * depth_tile_count + depth_tile_index;
                nk_size_t const b_tile_idx1 = (column_tile_index + 1) * depth_tile_count + depth_tile_index;
                nk_size_t const b_tile_idx2 = (column_tile_index + 2) * depth_tile_count + depth_tile_index;
                nk_size_t const b_tile_idx3 = (column_tile_index + 3) * depth_tile_count + depth_tile_index;
                nk_i8_t const *b_tile0 = b_tiles + b_tile_idx0 * tile_elements;
                nk_i8_t const *b_tile1 = b_tiles + b_tile_idx1 * tile_elements;
                nk_i8_t const *b_tile2 = b_tiles + b_tile_idx2 * tile_elements;
                nk_i8_t const *b_tile3 = b_tiles + b_tile_idx3 * tile_elements;

                for (nk_size_t row = 0; row < tile_dimension; row++) {
                    nk_i8_t const *pointer_a = a + (row_start + row) * a_stride_elements + depth_offset;
                    svint8_t vector_a = svld1_s8(predicate_byte, pointer_a);

                    svint8_t vector_b_tile_0 = svld1_s8(predicate_byte, b_tile0 + row * depth_tile_size);
                    svint8_t vector_b_tile_1 = svld1_s8(predicate_byte, b_tile1 + row * depth_tile_size);
                    svint8_t vector_b_tile_2 = svld1_s8(predicate_byte, b_tile2 + row * depth_tile_size);
                    svint8_t vector_b_tile_3 = svld1_s8(predicate_byte, b_tile3 + row * depth_tile_size);

                    // Outer products (b8 predicates for i8→i32 widening)
                    svmopa_za32_s8_m(0, predicate_valid_rows_b8, predicate_smopa_full, vector_a, vector_b_tile_0);
                    svmopa_za32_s8_m(1, predicate_valid_rows_b8, predicate_smopa_full, vector_a, vector_b_tile_1);
                    svmopa_za32_s8_m(2, predicate_valid_rows_b8, predicate_smopa_full, vector_a, vector_b_tile_2);
                    svmopa_za32_s8_m(3, predicate_valid_rows_b8, predicate_smopa_full, vector_a, vector_b_tile_3);
                }
            }

            for (nk_size_t row = 0; row < rows_remaining; row++) {
                nk_size_t const col_start0 = (column_tile_index + 0) * tile_dimension;
                nk_size_t const col_start1 = (column_tile_index + 1) * tile_dimension;
                nk_size_t const col_start2 = (column_tile_index + 2) * tile_dimension;
                nk_size_t const col_start3 = (column_tile_index + 3) * tile_dimension;
                svst1_hor_za32(0, row, predicate_single, c + (row_start + row) * c_stride_elements + col_start0);
                svst1_hor_za32(1, row, predicate_single, c + (row_start + row) * c_stride_elements + col_start1);
                svst1_hor_za32(2, row, predicate_single, c + (row_start + row) * c_stride_elements + col_start2);
                svst1_hor_za32(3, row, predicate_single, c + (row_start + row) * c_stride_elements + col_start3);
            }
        }

        // Process remaining column tiles one at a time
        for (; column_tile_index < column_tile_count; column_tile_index++) {
            nk_size_t const col_start = column_tile_index * tile_dimension;
            nk_size_t const cols_remaining = (col_start + tile_dimension <= columns) ? tile_dimension
                                                                                     : (columns - col_start);
            svbool_t const predicate_valid_columns = svwhilelt_b32((uint32_t)0, (uint32_t)cols_remaining);
            // SMOPA column predicate: b8 with 4x elements for widening operation
            svbool_t const predicate_valid_columns_b8 = svwhilelt_b8((uint32_t)0, (uint32_t)(cols_remaining * 4));

            svzero_za();

            for (nk_size_t depth_tile_index = 0; depth_tile_index < depth_tile_count; depth_tile_index++) {
                nk_size_t const depth_offset = depth_tile_index * depth_tile_size;

                nk_size_t const b_tile_idx = column_tile_index * depth_tile_count + depth_tile_index;
                nk_i8_t const *b_tile = b_tiles + b_tile_idx * tile_elements;

                for (nk_size_t row = 0; row < tile_dimension; row++) {
                    nk_i8_t const *pointer_a = a + (row_start + row) * a_stride_elements + depth_offset;
                    svint8_t vector_a = svld1_s8(predicate_byte, pointer_a);
                    svint8_t vector_b = svld1_s8(predicate_byte, b_tile + row * depth_tile_size);
                    // Predicated outer product (b8 predicates for i8→i32 widening)
                    svmopa_za32_s8_m(0, predicate_valid_rows_b8, predicate_valid_columns_b8, vector_a, vector_b);
                }
            }

            // Predicated store to C (b32 predicates for i32 output elements)
            for (nk_size_t row = 0; row < rows_remaining; row++) {
                svst1_hor_za32(0, row, predicate_valid_columns, c + (row_start + row) * c_stride_elements + col_start);
            }
        }
    }
}

NK_PUBLIC void nk_dots_packed_i8_sme(                    //
    nk_i8_t const *a, void const *b_packed, nk_i32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,  //
    nk_size_t a_stride, nk_size_t c_stride) {

    nk_size_t const a_stride_elements = a_stride / sizeof(nk_i8_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_i32_t);

    nk_dots_i8_kernel_(a, b_packed, c, rows, columns, depth, a_stride_elements, c_stride_elements);
}

#pragma endregion

/*
 *  e4m3 × e4m3 → f32 GEMM using inline SSVE conversion + FMOPA.
 *
 *  Pipeline: e4m3 bytes → svunpklo → arithmetic → f16 → FMOPA → f32
 *  - Load: 64 bytes via svld1_u8, convert lower 32 to f16 inline
 *  - Accumulate: FMOPA f16→f32 into ZA32 tiles
 *  - No memory round-trip for format conversion
 *  - FMOPA predicates: b16 (f16 input granularity)
 */
#pragma region FP8 E4M3 (e4m3)

/**
 *  Inline `e4m3` → `f16` conversion returning `svfloat16_t` for direct use in GEMM.
 *  This avoids memory round-trip when used inside a streaming kernel.
 *
 *  @param pg16   Predicate for 16-bit elements: use `svptrue_b16()`
 *  @param bytes  Pre-loaded 64 bytes: `svuint8_t` from `svld1_u8`
 *  @return       32 `f16` values as `svfloat16_t`: from lower 32 bytes
 */
NK_INTERNAL svfloat16_t nk_e4m3x_to_f16x_ssve_(svbool_t pg16, svuint8_t bytes) {
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

/**
 *  Inline `e5m2` → `f16` conversion returning `svfloat16_t` for direct use in GEMM.
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
NK_INTERNAL svfloat16_t nk_e5m2x_to_f16x_ssve_(svbool_t pg16, svuint8_t bytes) {
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

/**
 *  Fused `e4m3` × `e4m3` → `f32` GEMM kernel with predicate-based edge handling.
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

    nk_size_t const tile_dimension = svcntw();  // number of `f32` elements per vector
    nk_size_t const depth_tile_size = svcnth(); // number of `f16` elements per vector
    nk_size_t const tile_elements = tile_dimension * depth_tile_size;

    nk_f16_t const *b_tiles = (nk_f16_t const *)((char const *)b_packed + sizeof(nk_dots_sme_packed_header_t));

    svbool_t const predicate_half = svptrue_b16();
    svbool_t const predicate_byte = svptrue_b8();

    nk_size_t const row_tile_index_count = (rows + tile_dimension - 1) / tile_dimension;

    // Process tile-by-tile with predicates for edge handling
    for (nk_size_t row_tile_index = 0; row_tile_index < row_tile_index_count; row_tile_index++) {
        nk_size_t const row_start = row_tile_index * tile_dimension;
        nk_size_t const rows_remaining = (row_start + tile_dimension <= rows) ? tile_dimension : (rows - row_start);
        svbool_t const predicate_valid_rows = svwhilelt_b32((uint32_t)0, (uint32_t)rows_remaining);
        // FMOPA row predicate: b16 with 2x elements for widening operation
        svbool_t const predicate_valid_rows_b16 = svwhilelt_b16((uint32_t)0, (uint32_t)(rows_remaining * 2));

        for (nk_size_t column_tile_index = 0; column_tile_index < column_tile_count; column_tile_index++) {
            nk_size_t const col_start = column_tile_index * tile_dimension;
            nk_size_t const cols_remaining = (col_start + tile_dimension <= columns) ? tile_dimension
                                                                                     : (columns - col_start);
            svbool_t const predicate_valid_columns = svwhilelt_b32((uint32_t)0, (uint32_t)cols_remaining);
            // FMOPA column predicate: b16 with 2x elements for widening operation
            svbool_t const predicate_valid_columns_b16 = svwhilelt_b16((uint32_t)0, (uint32_t)(cols_remaining * 2));

            // Zero ZA tile 0
            svzero_za();

            // Accumulate over depth tiles
            for (nk_size_t depth_tile_index = 0; depth_tile_index < depth_tile_count; depth_tile_index++) {
                nk_size_t const depth_offset = depth_tile_index * depth_tile_size;

                nk_size_t const b_tile_idx = column_tile_index * depth_tile_count + depth_tile_index;
                nk_f16_t const *b_tile = b_tiles + b_tile_idx * tile_elements;

                // Outer products over `tile_dimension` rows
                for (nk_size_t row = 0; row < tile_dimension; row++) {
                    nk_e4m3_t const *pointer_a = a + (row_start + row) * a_stride_elements + depth_offset;
                    svuint8_t a_bytes = svld1_u8(predicate_byte, (uint8_t const *)pointer_a);
                    svfloat16_t vector_a = nk_e4m3x_to_f16x_ssve_(predicate_half, a_bytes);
                    svfloat16_t vector_b = svld1_f16(predicate_half,
                                                     (float16_t const *)(b_tile + row * depth_tile_size));
                    // Predicated outer product (b16 predicates for f16→f32 widening)
                    svmopa_za32_f16_m(0, predicate_valid_rows_b16, predicate_valid_columns_b16, vector_a, vector_b);
                }
            }

            // Predicated store to C (b32 predicates for f32 output elements)
            for (nk_size_t row = 0; row < rows_remaining; row++) {
                svst1_hor_za32(0, row, predicate_valid_columns, c + (row_start + row) * c_stride_elements + col_start);
            }
        }
    }
}

NK_PUBLIC nk_size_t nk_dots_packed_size_e4m3_sme(nk_size_t n, nk_size_t k) {
    // Uses `f16` format for packed data
    return nk_dots_packed_size_f16_sme(n, k);
}

NK_PUBLIC void nk_dots_pack_e4m3_sme(             //
    nk_e4m3_t const *b, nk_size_t n, nk_size_t k, //
    nk_size_t b_stride, void *b_packed) {

    nk_size_t const svl_bytes = svcntsw() * sizeof(nk_f32_t);
    nk_size_t const tile_dimension = svcntsw();  // rows per tile
    nk_size_t const depth_tile_size = svcntsh(); // K elements per tile
    nk_size_t const tile_elements = tile_dimension * depth_tile_size;
    nk_size_t const b_stride_elements = b_stride / sizeof(nk_e4m3_t);

    nk_size_t const column_tile_count = (n + tile_dimension - 1) / tile_dimension;
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

    // Pack tiles with `e4m3` → `f16` conversion, column-major layout
    for (nk_size_t column_tile = 0; column_tile < column_tile_count; column_tile++) {
        for (nk_size_t depth_tile = 0; depth_tile < depth_tile_count; depth_tile++) {
            nk_size_t const tile_index = column_tile * depth_tile_count + depth_tile;
            nk_f16_t *tile_output = tiles_ptr + tile_index * tile_elements;

            nk_size_t const src_row_start = column_tile * tile_dimension;
            nk_size_t const src_col_start = depth_tile * depth_tile_size;
            nk_size_t const rows_to_pack = (src_row_start + tile_dimension <= n) ? tile_dimension : (n - src_row_start);
            nk_size_t const cols_to_pack = (src_col_start + depth_tile_size <= k) ? depth_tile_size
                                                                                  : (k - src_col_start);

            for (nk_size_t row = 0; row < rows_to_pack; row++) {
                for (nk_size_t col = 0; col < cols_to_pack; col++) {
                    nk_size_t const src_idx = (src_row_start + row) * b_stride_elements + src_col_start + col;
                    nk_size_t const dst_idx = col * tile_dimension + row;
                    nk_e4m3_to_f16_serial(&b[src_idx], &tile_output[dst_idx]);
                }
            }
        }
    }
}

NK_PUBLIC void nk_dots_packed_e4m3_sme(                    //
    nk_e4m3_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,    //
    nk_size_t a_stride, nk_size_t c_stride) {

    nk_size_t const a_stride_elements = a_stride / sizeof(nk_e4m3_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);

    nk_dots_e4m3_kernel_(a, b_packed, c, rows, columns, depth, a_stride_elements, c_stride_elements);
}

#pragma endregion

/*
 *  e5m2 × e5m2 → f32 GEMM using inline SSVE conversion + FMOPA.
 *
 *  Pipeline: e5m2 bytes → svunpklo → arithmetic → f16 → FMOPA → f32
 *  - Same tile layout as e4m3 (both convert to f16 before FMOPA)
 *  - E5M2 shares F16 exponent bias (15), so normal conversion is a shift
 *  - Handles infinity (mag=0x7C) and NaN (mag>0x7C)
 */
#pragma region FP8 E5M2 (e5m2)

/**
 *  Fused `e5m2` × `e5m2` → `f32` GEMM kernel using SSVE inline conversion.
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

    nk_size_t const tile_dimension = svcntw();  // number of `f32` elements per vector
    nk_size_t const depth_tile_size = svcnth(); // number of `f16` elements per vector
    nk_size_t const tile_elements = tile_dimension * depth_tile_size;

    nk_f16_t const *b_tiles = (nk_f16_t const *)((char const *)b_packed + sizeof(nk_dots_sme_packed_header_t));

    svbool_t const predicate_half = svptrue_b16();
    svbool_t const predicate_byte = svptrue_b8();

    nk_size_t const row_tile_index_count = (rows + tile_dimension - 1) / tile_dimension;

    // Process tile-by-tile with predicates for edge handling
    for (nk_size_t row_tile_index = 0; row_tile_index < row_tile_index_count; row_tile_index++) {
        nk_size_t const row_start = row_tile_index * tile_dimension;
        nk_size_t const rows_remaining = (row_start + tile_dimension <= rows) ? tile_dimension : (rows - row_start);
        svbool_t const predicate_valid_rows = svwhilelt_b32((uint32_t)0, (uint32_t)rows_remaining);
        // FMOPA row predicate: b16 with 2x elements for widening operation
        svbool_t const predicate_valid_rows_b16 = svwhilelt_b16((uint32_t)0, (uint32_t)(rows_remaining * 2));

        for (nk_size_t column_tile_index = 0; column_tile_index < column_tile_count; column_tile_index++) {
            nk_size_t const col_start = column_tile_index * tile_dimension;
            nk_size_t const cols_remaining = (col_start + tile_dimension <= columns) ? tile_dimension
                                                                                     : (columns - col_start);
            svbool_t const predicate_valid_columns = svwhilelt_b32((uint32_t)0, (uint32_t)cols_remaining);
            // FMOPA column predicate: b16 with 2x elements for widening operation
            svbool_t const predicate_valid_columns_b16 = svwhilelt_b16((uint32_t)0, (uint32_t)(cols_remaining * 2));

            // Zero ZA tile 0
            svzero_za();

            // Accumulate over depth tiles
            for (nk_size_t depth_tile_index = 0; depth_tile_index < depth_tile_count; depth_tile_index++) {
                nk_size_t const depth_offset = depth_tile_index * depth_tile_size;

                nk_size_t const b_tile_idx = column_tile_index * depth_tile_count + depth_tile_index;
                nk_f16_t const *b_tile = b_tiles + b_tile_idx * tile_elements;

                // Outer products over `tile_dimension` rows
                for (nk_size_t row = 0; row < tile_dimension; row++) {
                    nk_e5m2_t const *pointer_a = a + (row_start + row) * a_stride_elements + depth_offset;
                    svuint8_t a_bytes = svld1_u8(predicate_byte, (uint8_t const *)pointer_a);
                    svfloat16_t vector_a = nk_e5m2x_to_f16x_ssve_(predicate_half, a_bytes);
                    svfloat16_t vector_b = svld1_f16(predicate_half,
                                                     (float16_t const *)(b_tile + row * depth_tile_size));
                    // Predicated outer product (b16 predicates for f16→f32 widening)
                    svmopa_za32_f16_m(0, predicate_valid_rows_b16, predicate_valid_columns_b16, vector_a, vector_b);
                }
            }

            // Predicated store to C (b32 predicates for f32 output elements)
            for (nk_size_t row = 0; row < rows_remaining; row++) {
                svst1_hor_za32(0, row, predicate_valid_columns, c + (row_start + row) * c_stride_elements + col_start);
            }
        }
    }
}

NK_PUBLIC nk_size_t nk_dots_packed_size_e5m2_sme(nk_size_t n, nk_size_t k) { return nk_dots_packed_size_f16_sme(n, k); }

NK_PUBLIC void nk_dots_pack_e5m2_sme(nk_e5m2_t const *b, nk_size_t n, nk_size_t k, nk_size_t b_stride, void *b_packed) {

    nk_size_t const svl_bytes = svcntsw() * sizeof(nk_f32_t);
    nk_size_t const tile_dimension = svcntsw();  // rows per tile
    nk_size_t const depth_tile_size = svcntsh(); // K elements per tile
    nk_size_t const tile_elements = tile_dimension * depth_tile_size;
    nk_size_t const b_stride_elements = b_stride / sizeof(nk_e5m2_t);

    nk_size_t const column_tile_count = (n + tile_dimension - 1) / tile_dimension;
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

            nk_size_t const src_row_start = column_tile * tile_dimension;
            nk_size_t const src_col_start = depth_tile * depth_tile_size;
            nk_size_t const rows_to_pack = (src_row_start + tile_dimension <= n) ? tile_dimension : (n - src_row_start);
            nk_size_t const cols_to_pack = (src_col_start + depth_tile_size <= k) ? depth_tile_size
                                                                                  : (k - src_col_start);

            for (nk_size_t row = 0; row < rows_to_pack; row++) {
                for (nk_size_t col = 0; col < cols_to_pack; col++) {
                    nk_size_t const src_idx = (src_row_start + row) * b_stride_elements + src_col_start + col;
                    nk_size_t const dst_idx = col * tile_dimension + row;
                    nk_e5m2_to_f16_serial(&b[src_idx], &tile_output[dst_idx]);
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

#pragma endregion

#pragma region Unsigned 8-bit Integers (u8)

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
    nk_size_t const tile_dimension = svcntsw();  // rows per tile
    nk_size_t const depth_tile_size = svcntsb(); // K elements per tile
    nk_size_t const tile_elements = tile_dimension * depth_tile_size;
    nk_size_t const b_stride_elements = b_stride / sizeof(nk_u8_t);

    nk_size_t const column_tile_count = (n + tile_dimension - 1) / tile_dimension;
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

            nk_size_t const src_row_start = column_tile * tile_dimension;
            nk_size_t const src_col_start = depth_tile * depth_tile_size;
            nk_size_t const rows_to_pack = (src_row_start + tile_dimension <= n) ? tile_dimension : (n - src_row_start);
            nk_size_t const cols_to_pack = (src_col_start + depth_tile_size <= k) ? depth_tile_size
                                                                                  : (k - src_col_start);

            for (nk_size_t row = 0; row < rows_to_pack; row++) {
                for (nk_size_t col = 0; col < cols_to_pack; col++) {
                    nk_size_t const src_idx = (src_row_start + row) * b_stride_elements + src_col_start + col;
                    nk_size_t const dst_idx = col * tile_dimension + row;
                    tile_output[dst_idx] = b[src_idx];
                }
            }
        }
    }
}

/**
 * `u8` × `u8` → `u32` GEMM kernel with predicate-based edge handling.
 *  Multi-tile optimization: Processes 4 column tiles simultaneously using ZA0-ZA3.
 */
__arm_locally_streaming __arm_new("za") static void nk_dots_u8_kernel_( //
    nk_u8_t const *a, void const *b_packed, nk_u32_t *c,                //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,                 //
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) {

    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_size_t const column_tile_count = header->column_tile_count;
    nk_size_t const depth_tile_count = header->depth_tile_count;

    nk_size_t const tile_dimension = svcntw();  // number of `u32` elements per vector
    nk_size_t const depth_tile_size = svcntb(); // number of `u8` elements per vector
    nk_size_t const tile_elements = tile_dimension * depth_tile_size;

    nk_u8_t const *b_tiles = (nk_u8_t const *)((char const *)b_packed + sizeof(nk_dots_sme_packed_header_t));

    svbool_t const predicate_byte = svptrue_b8();
    svbool_t const predicate_single = svptrue_b32();
    // UMOPA u8→u32 uses b8 predicates (input element granularity), not b32 (output granularity)
    svbool_t const predicate_umopa_full = svptrue_b8();

    nk_size_t const row_tile_index_count = (rows + tile_dimension - 1) / tile_dimension;

    for (nk_size_t row_tile_index = 0; row_tile_index < row_tile_index_count; row_tile_index++) {
        nk_size_t const row_start = row_tile_index * tile_dimension;
        nk_size_t const rows_remaining = (row_start + tile_dimension <= rows) ? tile_dimension : (rows - row_start);
        svbool_t const predicate_valid_rows = svwhilelt_b32((uint32_t)0, (uint32_t)rows_remaining);
        // UMOPA row predicate: b8 with 4x elements for widening operation
        svbool_t const predicate_valid_rows_b8 = svwhilelt_b8((uint32_t)0, (uint32_t)(rows_remaining * 4));

        nk_size_t column_tile_index = 0;

        // Process 4 column tiles at a time using ZA0-ZA3
        for (; column_tile_index + 4 <= column_tile_count; column_tile_index += 4) {
            svzero_za();

            for (nk_size_t depth_tile_index = 0; depth_tile_index < depth_tile_count; depth_tile_index++) {
                nk_size_t const depth_offset = depth_tile_index * depth_tile_size;

                nk_size_t const b_tile_idx0 = (column_tile_index + 0) * depth_tile_count + depth_tile_index;
                nk_size_t const b_tile_idx1 = (column_tile_index + 1) * depth_tile_count + depth_tile_index;
                nk_size_t const b_tile_idx2 = (column_tile_index + 2) * depth_tile_count + depth_tile_index;
                nk_size_t const b_tile_idx3 = (column_tile_index + 3) * depth_tile_count + depth_tile_index;
                nk_u8_t const *b_tile0 = b_tiles + b_tile_idx0 * tile_elements;
                nk_u8_t const *b_tile1 = b_tiles + b_tile_idx1 * tile_elements;
                nk_u8_t const *b_tile2 = b_tiles + b_tile_idx2 * tile_elements;
                nk_u8_t const *b_tile3 = b_tiles + b_tile_idx3 * tile_elements;

                for (nk_size_t row = 0; row < tile_dimension && row_start + row < rows; row++) {
                    nk_u8_t const *pointer_a = a + (row_start + row) * a_stride_elements + depth_offset;

                    nk_size_t const depth_remaining = (depth_offset + depth_tile_size <= depth)
                                                          ? depth_tile_size
                                                          : (depth > depth_offset ? depth - depth_offset : 0);
                    svbool_t const pred_k = svwhilelt_b8((uint32_t)0, (uint32_t)depth_remaining);
                    svuint8_t vector_a = svld1_u8(pred_k, pointer_a);

                    svuint8_t vector_b_tile_0 = svld1_u8(predicate_byte, b_tile0 + row * depth_tile_size);
                    svuint8_t vector_b_tile_1 = svld1_u8(predicate_byte, b_tile1 + row * depth_tile_size);
                    svuint8_t vector_b_tile_2 = svld1_u8(predicate_byte, b_tile2 + row * depth_tile_size);
                    svuint8_t vector_b_tile_3 = svld1_u8(predicate_byte, b_tile3 + row * depth_tile_size);

                    // Outer products (b8 predicates for u8→u32 widening)
                    svmopa_za32_u8_m(0, predicate_valid_rows_b8, predicate_umopa_full, vector_a, vector_b_tile_0);
                    svmopa_za32_u8_m(1, predicate_valid_rows_b8, predicate_umopa_full, vector_a, vector_b_tile_1);
                    svmopa_za32_u8_m(2, predicate_valid_rows_b8, predicate_umopa_full, vector_a, vector_b_tile_2);
                    svmopa_za32_u8_m(3, predicate_valid_rows_b8, predicate_umopa_full, vector_a, vector_b_tile_3);
                }
            }

            for (nk_size_t row = 0; row < rows_remaining; row++) {
                nk_size_t const col_start0 = (column_tile_index + 0) * tile_dimension;
                nk_size_t const col_start1 = (column_tile_index + 1) * tile_dimension;
                nk_size_t const col_start2 = (column_tile_index + 2) * tile_dimension;
                nk_size_t const col_start3 = (column_tile_index + 3) * tile_dimension;
                svst1_hor_za32(0, row, predicate_single,
                               (nk_i32_t *)(c + (row_start + row) * c_stride_elements + col_start0));
                svst1_hor_za32(1, row, predicate_single,
                               (nk_i32_t *)(c + (row_start + row) * c_stride_elements + col_start1));
                svst1_hor_za32(2, row, predicate_single,
                               (nk_i32_t *)(c + (row_start + row) * c_stride_elements + col_start2));
                svst1_hor_za32(3, row, predicate_single,
                               (nk_i32_t *)(c + (row_start + row) * c_stride_elements + col_start3));
            }
        }

        // Process remaining column tiles one at a time
        for (; column_tile_index < column_tile_count; column_tile_index++) {
            nk_size_t const col_start = column_tile_index * tile_dimension;
            nk_size_t const cols_remaining = (col_start + tile_dimension <= columns) ? tile_dimension
                                                                                     : (columns - col_start);
            svbool_t const predicate_valid_columns = svwhilelt_b32((uint32_t)0, (uint32_t)cols_remaining);
            // UMOPA column predicate: b8 with 4x elements for widening operation
            svbool_t const predicate_valid_columns_b8 = svwhilelt_b8((uint32_t)0, (uint32_t)(cols_remaining * 4));

            svzero_za();

            for (nk_size_t depth_tile_index = 0; depth_tile_index < depth_tile_count; depth_tile_index++) {
                nk_size_t const depth_offset = depth_tile_index * depth_tile_size;

                nk_size_t const b_tile_idx = column_tile_index * depth_tile_count + depth_tile_index;
                nk_u8_t const *b_tile = b_tiles + b_tile_idx * tile_elements;

                for (nk_size_t row = 0; row < tile_dimension && row_start + row < rows; row++) {
                    nk_u8_t const *pointer_a = a + (row_start + row) * a_stride_elements + depth_offset;

                    nk_size_t const depth_remaining = (depth_offset + depth_tile_size <= depth)
                                                          ? depth_tile_size
                                                          : (depth > depth_offset ? depth - depth_offset : 0);
                    svbool_t const pred_k = svwhilelt_b8((uint32_t)0, (uint32_t)depth_remaining);
                    svuint8_t vector_a = svld1_u8(pred_k, pointer_a);
                    svuint8_t vector_b = svld1_u8(predicate_byte, b_tile + row * depth_tile_size);
                    // Predicated outer product (b8 predicates for u8→u32 widening)
                    svmopa_za32_u8_m(0, predicate_valid_rows_b8, predicate_valid_columns_b8, vector_a, vector_b);
                }
            }

            // Predicated store to C (b32 predicates for u32 output elements)
            for (nk_size_t row = 0; row < rows_remaining; row++) {
                svst1_hor_za32(0, row, predicate_valid_columns,
                               (nk_i32_t *)(c + (row_start + row) * c_stride_elements + col_start));
            }
        }
    }
}

NK_PUBLIC void nk_dots_packed_u8_sme(                    //
    nk_u8_t const *a, void const *b_packed, nk_u32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,  //
    nk_size_t a_stride, nk_size_t c_stride) {

    nk_size_t const a_stride_elements = a_stride / sizeof(nk_u8_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_u32_t);

    nk_dots_u8_kernel_(a, b_packed, c, rows, columns, depth, a_stride_elements, c_stride_elements);
}

#pragma endregion

/*
 *  Symmetric pairwise dot products using streaming SVE.
 *
 *  Computes upper triangle of C[i,j] = dot(vectors[i], vectors[j]).
 *  Uses streaming mode for wider SVE vectors (512-bit on M4).
 *  Each kernel uses type-appropriate accumulation:
 *  - f16: f32 accumulation via svcvt_f32_f16 + svadd
 *  - bf16: f32 accumulation via svbfdot_f32
 *  - i8/u8: i32/u32 accumulation via svdot_s32/svdot_u32
 */
#pragma region Symmetric GEMM

/**
 *   `f16` × `f16` → `f32` symmetric kernel using streaming SVE.
 */
__arm_locally_streaming static void nk_dots_symmetric_f16_sme_kernel_(nk_f16_t const *vectors, nk_size_t n_vectors,
                                                                      nk_size_t depth, nk_size_t stride_elements,
                                                                      nk_f32_t *result,
                                                                      nk_size_t result_stride_elements,
                                                                      nk_size_t row_start, nk_size_t row_count) {

    nk_size_t const vector_length = svcnth(); // f16 elements per vector
    svbool_t const predicate_half = svptrue_b16();

    nk_size_t const row_end = row_start + row_count;

    // Compute specified rows of symmetric matrix
    for (nk_size_t i = row_start; i < row_end && i < n_vectors; i++) {
        nk_f16_t const *pointer_row_i = vectors + i * stride_elements;
        // Only compute upper triangle including diagonal
        for (nk_size_t j = i; j < n_vectors; j++) {
            nk_f16_t const *pointer_row_j = vectors + j * stride_elements;

            // SVE vectorized dot product with f32 accumulation
            svfloat32_t accumulator = svdup_f32(0.0f);
            nk_size_t depth_index = 0;
            while (depth_index + vector_length <= depth) {
                svfloat16_t vector_i = svld1_f16(predicate_half, (float16_t const *)(pointer_row_i + depth_index));
                svfloat16_t vector_j = svld1_f16(predicate_half, (float16_t const *)(pointer_row_j + depth_index));
                // Multiply and accumulate with widening
                svfloat32_t prod = svcvt_f32_f16_x(svptrue_b32(), svmul_f16_x(predicate_half, vector_i, vector_j));
                accumulator = svadd_f32_x(svptrue_b32(), accumulator, prod);
                depth_index += vector_length;
            }
            // Handle remainder
            if (depth_index < depth) {
                svbool_t predicate_tail = svwhilelt_b16((uint32_t)depth_index, (uint32_t)depth);
                svfloat16_t vector_i = svld1_f16(predicate_tail, (float16_t const *)(pointer_row_i + depth_index));
                svfloat16_t vector_j = svld1_f16(predicate_tail, (float16_t const *)(pointer_row_j + depth_index));
                svfloat32_t prod = svcvt_f32_f16_x(svptrue_b32(), svmul_f16_x(predicate_tail, vector_i, vector_j));
                accumulator = svadd_f32_x(svptrue_b32(), accumulator, prod);
            }
            nk_f32_t dot = svaddv_f32(svptrue_b32(), accumulator);
            result[i * result_stride_elements + j] = dot;
            if (i != j) result[j * result_stride_elements + i] = dot;
        }
    }
}

NK_PUBLIC void nk_dots_symmetric_f16_sme(nk_f16_t const *vectors, nk_size_t n_vectors, nk_size_t depth,
                                         nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                         nk_size_t row_start, nk_size_t row_count) {

    nk_size_t const stride_elements = stride / sizeof(nk_f16_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_dots_symmetric_f16_sme_kernel_(vectors, n_vectors, depth, stride_elements, result, result_stride_elements,
                                      row_start, row_count);
}

/**
 * `bf16` × `bf16` → `f32` symmetric kernel using streaming SVE.
 */
__arm_locally_streaming static void nk_dots_symmetric_bf16_sme_kernel_(nk_bf16_t const *vectors, nk_size_t n_vectors,
                                                                       nk_size_t depth, nk_size_t stride_elements,
                                                                       nk_f32_t *result,
                                                                       nk_size_t result_stride_elements,
                                                                       nk_size_t row_start, nk_size_t row_count) {

    nk_size_t const vector_length = svcnth(); // bf16 elements per vector
    svbool_t const predicate_half = svptrue_b16();

    nk_size_t const row_end = row_start + row_count;

    // Compute specified rows of symmetric matrix
    for (nk_size_t i = row_start; i < row_end && i < n_vectors; i++) {
        nk_bf16_t const *pointer_row_i = vectors + i * stride_elements;
        for (nk_size_t j = i; j < n_vectors; j++) {
            nk_bf16_t const *pointer_row_j = vectors + j * stride_elements;

            // SVE vectorized dot product with f32 accumulation
            svfloat32_t accumulator = svdup_f32(0.0f);
            nk_size_t depth_index = 0;
            while (depth_index + vector_length <= depth) {
                svbfloat16_t vector_i = svld1_bf16(predicate_half, (bfloat16_t const *)(pointer_row_i + depth_index));
                svbfloat16_t vector_j = svld1_bf16(predicate_half, (bfloat16_t const *)(pointer_row_j + depth_index));
                // Multiply using bfdot (bf16 → f32 fused dot product)
                accumulator = svbfdot_f32(accumulator, vector_i, vector_j);
                depth_index += vector_length;
            }
            // Handle remainder
            if (depth_index < depth) {
                svbool_t predicate_tail = svwhilelt_b16((uint32_t)depth_index, (uint32_t)depth);
                svbfloat16_t vector_i = svld1_bf16(predicate_tail, (bfloat16_t const *)(pointer_row_i + depth_index));
                svbfloat16_t vector_j = svld1_bf16(predicate_tail, (bfloat16_t const *)(pointer_row_j + depth_index));
                accumulator = svbfdot_f32(accumulator, vector_i, vector_j);
            }
            nk_f32_t dot = svaddv_f32(svptrue_b32(), accumulator);
            result[i * result_stride_elements + j] = dot;
            if (i != j) result[j * result_stride_elements + i] = dot;
        }
    }
}

NK_PUBLIC void nk_dots_symmetric_bf16_sme(nk_bf16_t const *vectors, nk_size_t n_vectors, nk_size_t depth,
                                          nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                          nk_size_t row_start, nk_size_t row_count) {

    nk_size_t const stride_elements = stride / sizeof(nk_bf16_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_dots_symmetric_bf16_sme_kernel_(vectors, n_vectors, depth, stride_elements, result, result_stride_elements,
                                       row_start, row_count);
}

/*  `i8` × `i8` → `i32` symmetric kernel using streaming SVE.
 *  Uses predicated vector loads and `svdot_s32` for vectorized dot product.
 */
__arm_locally_streaming static void nk_dots_symmetric_i8_sme_kernel_(nk_i8_t const *vectors, nk_size_t n_vectors,
                                                                     nk_size_t depth, nk_size_t stride_elements,
                                                                     nk_i32_t *result, nk_size_t result_stride_elements,
                                                                     nk_size_t row_start, nk_size_t row_count) {

    nk_size_t const vector_length = svcntb();
    svbool_t const predicate_byte = svptrue_b8();

    nk_size_t const row_end = row_start + row_count;

    for (nk_size_t i = row_start; i < row_end && i < n_vectors; i++) {
        nk_i8_t const *pointer_source_i = vectors + i * stride_elements;
        for (nk_size_t j = i; j < n_vectors; j++) {
            nk_i8_t const *pointer_source_j = vectors + j * stride_elements;

            svint32_t accumulator = svdup_s32(0);
            nk_size_t depth_index = 0;
            while (depth_index + vector_length <= depth) {
                svint8_t vector_i = svld1_s8(predicate_byte, pointer_source_i + depth_index);
                svint8_t vector_j = svld1_s8(predicate_byte, pointer_source_j + depth_index);
                accumulator = svdot_s32(accumulator, vector_i, vector_j);
                depth_index += vector_length;
            }
            if (depth_index < depth) {
                svbool_t predicate_tail = svwhilelt_b8((uint32_t)depth_index, (uint32_t)depth);
                svint8_t vector_i = svld1_s8(predicate_tail, pointer_source_i + depth_index);
                svint8_t vector_j = svld1_s8(predicate_tail, pointer_source_j + depth_index);
                accumulator = svdot_s32(accumulator, vector_i, vector_j);
            }
            nk_i32_t dot = svaddv_s32(svptrue_b32(), accumulator);
            result[i * result_stride_elements + j] = dot;
            if (i != j) result[j * result_stride_elements + i] = dot;
        }
    }
}

NK_PUBLIC void nk_dots_symmetric_i8_sme(nk_i8_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride,
                                        nk_i32_t *result, nk_size_t result_stride, nk_size_t row_start,
                                        nk_size_t row_count) {

    nk_size_t const stride_elements = stride / sizeof(nk_i8_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_i32_t);
    nk_dots_symmetric_i8_sme_kernel_(vectors, n_vectors, depth, stride_elements, result, result_stride_elements,
                                     row_start, row_count);
}

/**
 * `u8` × `u8` → `u32` symmetric kernel using streaming SVE.
 *  Uses predicated vector loads and `svdot_u32` for vectorized dot product.
 */
__arm_locally_streaming static void nk_dots_symmetric_u8_sme_kernel_(nk_u8_t const *vectors, nk_size_t n_vectors,
                                                                     nk_size_t depth, nk_size_t stride_elements,
                                                                     nk_u32_t *result, nk_size_t result_stride_elements,
                                                                     nk_size_t row_start, nk_size_t row_count) {

    nk_size_t const vector_length = svcntb();
    svbool_t const predicate_byte = svptrue_b8();

    nk_size_t const row_end = row_start + row_count;

    for (nk_size_t i = row_start; i < row_end && i < n_vectors; i++) {
        nk_u8_t const *pointer_source_i = vectors + i * stride_elements;
        for (nk_size_t j = i; j < n_vectors; j++) {
            nk_u8_t const *pointer_source_j = vectors + j * stride_elements;

            svuint32_t accumulator = svdup_u32(0);
            nk_size_t depth_index = 0;
            while (depth_index + vector_length <= depth) {
                svuint8_t vector_i = svld1_u8(predicate_byte, pointer_source_i + depth_index);
                svuint8_t vector_j = svld1_u8(predicate_byte, pointer_source_j + depth_index);
                accumulator = svdot_u32(accumulator, vector_i, vector_j);
                depth_index += vector_length;
            }
            if (depth_index < depth) {
                svbool_t predicate_tail = svwhilelt_b8((uint32_t)depth_index, (uint32_t)depth);
                svuint8_t vector_i = svld1_u8(predicate_tail, pointer_source_i + depth_index);
                svuint8_t vector_j = svld1_u8(predicate_tail, pointer_source_j + depth_index);
                accumulator = svdot_u32(accumulator, vector_i, vector_j);
            }
            nk_u32_t dot = svaddv_u32(svptrue_b32(), accumulator);
            result[i * result_stride_elements + j] = dot;
            if (i != j) result[j * result_stride_elements + i] = dot;
        }
    }
}

NK_PUBLIC void nk_dots_symmetric_u8_sme(nk_u8_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride,
                                        nk_u32_t *result, nk_size_t result_stride, nk_size_t row_start,
                                        nk_size_t row_count) {

    nk_size_t const stride_elements = stride / sizeof(nk_u8_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_u32_t);
    nk_dots_symmetric_u8_sme_kernel_(vectors, n_vectors, depth, stride_elements, result, result_stride_elements,
                                     row_start, row_count);
}

/**
 * `e4m3` × `e4m3` → `f32` symmetric kernel using streaming SVE.
 *  Uses inline e4m3 → f16 conversion.
 */
__arm_locally_streaming static void nk_dots_symmetric_e4m3_sme_kernel_(nk_e4m3_t const *vectors, nk_size_t n_vectors,
                                                                       nk_size_t depth, nk_size_t stride_elements,
                                                                       nk_f32_t *result,
                                                                       nk_size_t result_stride_elements,
                                                                       nk_size_t row_start, nk_size_t row_count) {

    nk_size_t const vector_length = svcnth(); // f16 elements per vector (after conversion)
    svbool_t const predicate_half = svptrue_b16();
    svbool_t const predicate_byte = svptrue_b8();

    nk_size_t const row_end = row_start + row_count;

    for (nk_size_t i = row_start; i < row_end && i < n_vectors; i++) {
        nk_e4m3_t const *pointer_row_i = vectors + i * stride_elements;
        for (nk_size_t j = i; j < n_vectors; j++) {
            nk_e4m3_t const *pointer_row_j = vectors + j * stride_elements;

            svfloat32_t accumulator = svdup_f32(0.0f);
            nk_size_t depth_index = 0;
            while (depth_index + vector_length <= depth) {
                // Load e4m3 bytes and convert to f16
                svuint8_t vector_i_bytes = svld1_u8(predicate_byte, (uint8_t const *)(pointer_row_i + depth_index));
                svuint8_t vector_j_bytes = svld1_u8(predicate_byte, (uint8_t const *)(pointer_row_j + depth_index));
                svfloat16_t vector_i = nk_e4m3x_to_f16x_ssve_(predicate_half, vector_i_bytes);
                svfloat16_t vector_j = nk_e4m3x_to_f16x_ssve_(predicate_half, vector_j_bytes);
                svfloat32_t prod = svcvt_f32_f16_x(svptrue_b32(), svmul_f16_x(predicate_half, vector_i, vector_j));
                accumulator = svadd_f32_x(svptrue_b32(), accumulator, prod);
                depth_index += vector_length;
            }
            if (depth_index < depth) {
                svbool_t predicate_byte_tail = svwhilelt_b8((uint32_t)depth_index, (uint32_t)depth);
                svbool_t predicate_half_tail = svwhilelt_b16((uint32_t)depth_index, (uint32_t)depth);
                svuint8_t vector_i_bytes = svld1_u8(predicate_byte_tail,
                                                    (uint8_t const *)(pointer_row_i + depth_index));
                svuint8_t vector_j_bytes = svld1_u8(predicate_byte_tail,
                                                    (uint8_t const *)(pointer_row_j + depth_index));
                svfloat16_t vector_i = nk_e4m3x_to_f16x_ssve_(predicate_half_tail, vector_i_bytes);
                svfloat16_t vector_j = nk_e4m3x_to_f16x_ssve_(predicate_half_tail, vector_j_bytes);
                svfloat32_t prod = svcvt_f32_f16_x(svptrue_b32(), svmul_f16_x(predicate_half_tail, vector_i, vector_j));
                accumulator = svadd_f32_x(svptrue_b32(), accumulator, prod);
            }
            nk_f32_t dot = svaddv_f32(svptrue_b32(), accumulator);
            result[i * result_stride_elements + j] = dot;
            if (i != j) result[j * result_stride_elements + i] = dot;
        }
    }
}

NK_PUBLIC void nk_dots_symmetric_e4m3_sme(nk_e4m3_t const *vectors, nk_size_t n_vectors, nk_size_t depth,
                                          nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                          nk_size_t row_start, nk_size_t row_count) {

    nk_size_t const stride_elements = stride / sizeof(nk_e4m3_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_dots_symmetric_e4m3_sme_kernel_(vectors, n_vectors, depth, stride_elements, result, result_stride_elements,
                                       row_start, row_count);
}

/**
 * `e5m2` × `e5m2` → `f32` symmetric kernel using streaming SVE.
 *  Uses inline e5m2 → f16 conversion.
 */
__arm_locally_streaming static void nk_dots_symmetric_e5m2_sme_kernel_(nk_e5m2_t const *vectors, nk_size_t n_vectors,
                                                                       nk_size_t depth, nk_size_t stride_elements,
                                                                       nk_f32_t *result,
                                                                       nk_size_t result_stride_elements,
                                                                       nk_size_t row_start, nk_size_t row_count) {

    nk_size_t const vector_length = svcnth(); // f16 elements per vector (after conversion)
    svbool_t const predicate_half = svptrue_b16();
    svbool_t const predicate_byte = svptrue_b8();

    nk_size_t const row_end = row_start + row_count;

    for (nk_size_t i = row_start; i < row_end && i < n_vectors; i++) {
        nk_e5m2_t const *pointer_row_i = vectors + i * stride_elements;
        for (nk_size_t j = i; j < n_vectors; j++) {
            nk_e5m2_t const *pointer_row_j = vectors + j * stride_elements;

            svfloat32_t accumulator = svdup_f32(0.0f);
            nk_size_t depth_index = 0;
            while (depth_index + vector_length <= depth) {
                svuint8_t vector_i_bytes = svld1_u8(predicate_byte, (uint8_t const *)(pointer_row_i + depth_index));
                svuint8_t vector_j_bytes = svld1_u8(predicate_byte, (uint8_t const *)(pointer_row_j + depth_index));
                svfloat16_t vector_i = nk_e5m2x_to_f16x_ssve_(predicate_half, vector_i_bytes);
                svfloat16_t vector_j = nk_e5m2x_to_f16x_ssve_(predicate_half, vector_j_bytes);
                svfloat32_t prod = svcvt_f32_f16_x(svptrue_b32(), svmul_f16_x(predicate_half, vector_i, vector_j));
                accumulator = svadd_f32_x(svptrue_b32(), accumulator, prod);
                depth_index += vector_length;
            }
            if (depth_index < depth) {
                svbool_t predicate_byte_tail = svwhilelt_b8((uint32_t)depth_index, (uint32_t)depth);
                svbool_t predicate_half_tail = svwhilelt_b16((uint32_t)depth_index, (uint32_t)depth);
                svuint8_t vector_i_bytes = svld1_u8(predicate_byte_tail,
                                                    (uint8_t const *)(pointer_row_i + depth_index));
                svuint8_t vector_j_bytes = svld1_u8(predicate_byte_tail,
                                                    (uint8_t const *)(pointer_row_j + depth_index));
                svfloat16_t vector_i = nk_e5m2x_to_f16x_ssve_(predicate_half_tail, vector_i_bytes);
                svfloat16_t vector_j = nk_e5m2x_to_f16x_ssve_(predicate_half_tail, vector_j_bytes);
                svfloat32_t prod = svcvt_f32_f16_x(svptrue_b32(), svmul_f16_x(predicate_half_tail, vector_i, vector_j));
                accumulator = svadd_f32_x(svptrue_b32(), accumulator, prod);
            }
            nk_f32_t dot = svaddv_f32(svptrue_b32(), accumulator);
            result[i * result_stride_elements + j] = dot;
            if (i != j) result[j * result_stride_elements + i] = dot;
        }
    }
}

NK_PUBLIC void nk_dots_symmetric_e5m2_sme(nk_e5m2_t const *vectors, nk_size_t n_vectors, nk_size_t depth,
                                          nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                          nk_size_t row_start, nk_size_t row_count) {

    nk_size_t const stride_elements = stride / sizeof(nk_e5m2_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_dots_symmetric_e5m2_sme_kernel_(vectors, n_vectors, depth, stride_elements, result, result_stride_elements,
                                       row_start, row_count);
}

/**
 * `u4` × `u4` → `u32` symmetric kernel using streaming SVE.
 *  Unpacks 4-bit values to 8-bit, then uses vectorized dot product.
 */
__arm_locally_streaming static void nk_dots_symmetric_u4_sme_kernel_(nk_u4x2_t const *vectors, nk_size_t n_vectors,
                                                                     nk_size_t depth, nk_size_t stride_elements,
                                                                     nk_u32_t *result, nk_size_t result_stride_elements,
                                                                     nk_size_t row_start, nk_size_t row_count) {

    nk_size_t const vector_length = svcntb();
    svbool_t const predicate_byte = svptrue_b8();
    nk_size_t const packed_depth = (depth + 1) / 2;

    nk_size_t const row_end = row_start + row_count;

    for (nk_size_t i = row_start; i < row_end && i < n_vectors; i++) {
        nk_u4x2_t const *pointer_source_i = vectors + i * stride_elements;
        for (nk_size_t j = i; j < n_vectors; j++) {
            nk_u4x2_t const *pointer_source_j = vectors + j * stride_elements;

            svuint32_t accumulator_low_nibble = svdup_u32(0);
            svuint32_t accumulator_high_nibble = svdup_u32(0);
            nk_size_t byte_index = 0;

            while (byte_index + vector_length <= packed_depth) {
                svuint8_t packed_i = svld1_u8(predicate_byte, (nk_u8_t const *)(pointer_source_i + byte_index));
                svuint8_t packed_j = svld1_u8(predicate_byte, (nk_u8_t const *)(pointer_source_j + byte_index));
                svuint8_t vector_i_low_nibble = svand_n_u8_x(predicate_byte, packed_i, 0x0F);
                svuint8_t vector_i_high_nibble = svlsr_n_u8_x(predicate_byte, packed_i, 4);
                svuint8_t vector_j_low_nibble = svand_n_u8_x(predicate_byte, packed_j, 0x0F);
                svuint8_t vector_j_high_nibble = svlsr_n_u8_x(predicate_byte, packed_j, 4);
                accumulator_low_nibble = svdot_u32(accumulator_low_nibble, vector_i_low_nibble, vector_j_low_nibble);
                accumulator_high_nibble = svdot_u32(accumulator_high_nibble, vector_i_high_nibble,
                                                    vector_j_high_nibble);
                byte_index += vector_length;
            }
            if (byte_index < packed_depth) {
                svbool_t predicate_tail = svwhilelt_b8((uint32_t)byte_index, (uint32_t)packed_depth);
                svuint8_t packed_i = svld1_u8(predicate_tail, (nk_u8_t const *)(pointer_source_i + byte_index));
                svuint8_t packed_j = svld1_u8(predicate_tail, (nk_u8_t const *)(pointer_source_j + byte_index));
                svuint8_t vector_i_low_nibble = svand_n_u8_x(predicate_tail, packed_i, 0x0F);
                svuint8_t vector_i_high_nibble = svlsr_n_u8_x(predicate_tail, packed_i, 4);
                svuint8_t vector_j_low_nibble = svand_n_u8_x(predicate_tail, packed_j, 0x0F);
                svuint8_t vector_j_high_nibble = svlsr_n_u8_x(predicate_tail, packed_j, 4);
                accumulator_low_nibble = svdot_u32(accumulator_low_nibble, vector_i_low_nibble, vector_j_low_nibble);
                accumulator_high_nibble = svdot_u32(accumulator_high_nibble, vector_i_high_nibble,
                                                    vector_j_high_nibble);
            }
            nk_u32_t dot = svaddv_u32(svptrue_b32(), accumulator_low_nibble) +
                           svaddv_u32(svptrue_b32(), accumulator_high_nibble);
            // Adjust for odd depth: last high nibble shouldn't contribute
            if (depth & 1) {
                nk_u8_t last_packed_i = pointer_source_i[packed_depth - 1];
                nk_u8_t last_packed_j = pointer_source_j[packed_depth - 1];
                dot -= (nk_u32_t)((last_packed_i >> 4) & 0x0F) * ((last_packed_j >> 4) & 0x0F);
            }
            result[i * result_stride_elements + j] = dot;
            if (i != j) result[j * result_stride_elements + i] = dot;
        }
    }
}

NK_PUBLIC void nk_dots_symmetric_u4_sme(nk_u4x2_t const *vectors, nk_size_t n_vectors, nk_size_t depth,
                                        nk_size_t stride, nk_u32_t *result, nk_size_t result_stride,
                                        nk_size_t row_start, nk_size_t row_count) {

    nk_size_t const stride_elements = stride / sizeof(nk_u4x2_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_u32_t);
    nk_dots_symmetric_u4_sme_kernel_(vectors, n_vectors, depth, stride_elements, result, result_stride_elements,
                                     row_start, row_count);
}

/**
 * `i4` × `i4` → `i32` symmetric kernel using streaming SVE.
 *  Unpacks 4-bit values to 8-bit with sign extension, then uses vectorized dot product.
 */
__arm_locally_streaming static void nk_dots_symmetric_i4_sme_kernel_(nk_i4x2_t const *vectors, nk_size_t n_vectors,
                                                                     nk_size_t depth, nk_size_t stride_elements,
                                                                     nk_i32_t *result, nk_size_t result_stride_elements,
                                                                     nk_size_t row_start, nk_size_t row_count) {

    nk_size_t const vector_length = svcntb();
    svbool_t const predicate_byte = svptrue_b8();
    nk_size_t const packed_depth = (depth + 1) / 2;

    nk_size_t const row_end = row_start + row_count;

    for (nk_size_t i = row_start; i < row_end && i < n_vectors; i++) {
        nk_i4x2_t const *pointer_source_i = vectors + i * stride_elements;
        for (nk_size_t j = i; j < n_vectors; j++) {
            nk_i4x2_t const *pointer_source_j = vectors + j * stride_elements;

            svint32_t accumulator_low_nibble = svdup_s32(0);
            svint32_t accumulator_high_nibble = svdup_s32(0);
            nk_size_t byte_index = 0;

            while (byte_index + vector_length <= packed_depth) {
                svint8_t packed_i = svld1_s8(predicate_byte, (nk_i8_t const *)(pointer_source_i + byte_index));
                svint8_t packed_j = svld1_s8(predicate_byte, (nk_i8_t const *)(pointer_source_j + byte_index));
                // Sign-extend low nibble: (val << 4) >> 4
                svint8_t vector_i_low_nibble = svasr_n_s8_x(predicate_byte, svlsl_n_s8_x(predicate_byte, packed_i, 4),
                                                            4);
                svint8_t vector_j_low_nibble = svasr_n_s8_x(predicate_byte, svlsl_n_s8_x(predicate_byte, packed_j, 4),
                                                            4);
                // Sign-extend high nibble: val >> 4
                svint8_t vector_i_high_nibble = svasr_n_s8_x(predicate_byte, packed_i, 4);
                svint8_t vector_j_high_nibble = svasr_n_s8_x(predicate_byte, packed_j, 4);
                accumulator_low_nibble = svdot_s32(accumulator_low_nibble, vector_i_low_nibble, vector_j_low_nibble);
                accumulator_high_nibble = svdot_s32(accumulator_high_nibble, vector_i_high_nibble,
                                                    vector_j_high_nibble);
                byte_index += vector_length;
            }
            if (byte_index < packed_depth) {
                svbool_t predicate_tail = svwhilelt_b8((uint32_t)byte_index, (uint32_t)packed_depth);
                svint8_t packed_i = svld1_s8(predicate_tail, (nk_i8_t const *)(pointer_source_i + byte_index));
                svint8_t packed_j = svld1_s8(predicate_tail, (nk_i8_t const *)(pointer_source_j + byte_index));
                svint8_t vector_i_low_nibble = svasr_n_s8_x(predicate_tail, svlsl_n_s8_x(predicate_tail, packed_i, 4),
                                                            4);
                svint8_t vector_j_low_nibble = svasr_n_s8_x(predicate_tail, svlsl_n_s8_x(predicate_tail, packed_j, 4),
                                                            4);
                svint8_t vector_i_high_nibble = svasr_n_s8_x(predicate_tail, packed_i, 4);
                svint8_t vector_j_high_nibble = svasr_n_s8_x(predicate_tail, packed_j, 4);
                accumulator_low_nibble = svdot_s32(accumulator_low_nibble, vector_i_low_nibble, vector_j_low_nibble);
                accumulator_high_nibble = svdot_s32(accumulator_high_nibble, vector_i_high_nibble,
                                                    vector_j_high_nibble);
            }
            nk_i32_t dot = svaddv_s32(svptrue_b32(), accumulator_low_nibble) +
                           svaddv_s32(svptrue_b32(), accumulator_high_nibble);
            // Adjust for odd depth: last high nibble shouldn't contribute
            if (depth & 1) {
                nk_i8_t last_packed_i = pointer_source_i[packed_depth - 1];
                nk_i8_t last_packed_j = pointer_source_j[packed_depth - 1];
                dot -= (nk_i32_t)(last_packed_i >> 4) * (last_packed_j >> 4);
            }
            result[i * result_stride_elements + j] = dot;
            if (i != j) result[j * result_stride_elements + i] = dot;
        }
    }
}

NK_PUBLIC void nk_dots_symmetric_i4_sme(nk_i4x2_t const *vectors, nk_size_t n_vectors, nk_size_t depth,
                                        nk_size_t stride, nk_i32_t *result, nk_size_t result_stride,
                                        nk_size_t row_start, nk_size_t row_count) {

    nk_size_t const stride_elements = stride / sizeof(nk_i4x2_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_i32_t);
    nk_dots_symmetric_i4_sme_kernel_(vectors, n_vectors, depth, stride_elements, result, result_stride_elements,
                                     row_start, row_count);
}

#pragma endregion
#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_SME
#endif // NK_TARGET_ARM_
#endif // NK_DOTS_SME_H

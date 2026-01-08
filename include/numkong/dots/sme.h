/**
 *  @brief SIMD-accelerated GEMM for Half-Precision Datatypes optimized for ARM SME.
 *  @file include/numkong/dots/sme.h
 *  @sa include/numkong/dots.h
 *  @author Ash Vardanian
 *  @date January 2, 2026
 *
 *  Uses ARM Scalable Matrix Extension (SME) for efficient matrix multiplication:
 *  - svmopa_za32_f16_m: F16×F16 → F32 outer product accumulate
 *  - svmopa_za32_bf16_m: BF16×BF16 → F32 outer product accumulate
 *  - svmopa_za64_f64_m: F64×F64 → F64 outer product accumulate (for high-precision F32 GEMM)
 *
 *  SME tile dimensions (for SVL=512, i.e., Apple M4):
 *    - ZA32 tile: 16×16 F32 elements (1KB)
 *    - ZA64 tile: 8×8 F64 elements (512B) - used for high-precision F32 GEMM
 *    - F16/BF16 vectors: 32 elements per SVE vector
 *    - F32 vectors: 16 elements per SVE vector
 *    - F64 vectors: 8 elements per SVE vector
 *    - One outer product: 32×32 → 16×16 F32 tile update (or 8×8 F64)
 *
 *  Output pattern: Each svmopa accumulates a 16×16 F32 tile from 32-element vectors.
 *  We process multiple ZA tiles (0-3) to form larger output blocks.
 *
 *  Performance characteristics (Apple M4):
 *    - F16 → F32 peak: ~2 TFLOPS per core (2x F32 throughput)
 *    - BF16 → F32 peak: ~2 TFLOPS per core
 *    - Streaming mode has different register set from normal NEON
 *
 *  Acceleration opportunities:
 *    - Pre-pack B matrix for column-major access (avoids transpose overhead)
 *    - Tile along M/N dimensions for cache efficiency
 *    - Use multiple ZA tiles for 2×2 output blocking
 *
 *  @section dots_sme_instructions ARM SME Instructions
 *
 *      Intrinsic/Attribute             Instruction                     Latency         Throughput
 *      svmopa_za32_f16_m               FMOPA (ZA.S, P/M, Z.H, Z.H)     16cy (amortized over 16x16 tile)
 *      svmopa_za32_bf16_m              BFMOPA (ZA.S, P/M, Z.H, Z.H)    16cy (amortized over 16x16 tile)
 *      svmopa_za64_f64_m               FMOPA (ZA.D, P/M, Z.D, Z.D)     16cy (amortized over 8x8 tile)
 *      svzero_za                       ZERO (ZA)                       2cy             1/cy
 *      svld1_hor_za32                  LD1W (ZA.S[Ws, #imm], P/Z, [Xn]) 4-6cy          1/cy
 *      svst1_hor_za32                  ST1W (ZA.S[Ws, #imm], P, [Xn])  4cy             1/cy
 *      svld1_f16                       LD1H (Z.H, P/Z, [Xn])           4-6cy           2/cy
 *      svld1_bf16                      LD1H (Z.H, P/Z, [Xn])           4-6cy           2/cy
 *      __arm_streaming                 SMSTART (enter streaming mode)  ~50-100cy       -
 *      __arm_streaming (exit)          SMSTOP (exit streaming mode)    ~50-100cy       -
 *      __arm_new("za")                 ZA tile allocation              0cy (compile-time)
 *      __arm_inout("za")               ZA tile read/write              0cy (compile-time)
 *      svcntw                          CNTW (Xd)                       1cy             2/cy
 *      svcnth                          CNTH (Xd)                       1cy             2/cy
 */
#ifndef NK_DOTS_SME_H
#define NK_DOTS_SME_H

#if NK_TARGET_ARM_
#if NK_TARGET_SME
#pragma GCC push_options
#pragma GCC target("+sme2+sme-f64f64")
#pragma clang attribute push(__attribute__((target("sme2,sme-f64f64"))), apply_to = function)

#include "numkong/types.h"

#include <stdlib.h> // aligned_alloc, free
#include <arm_sme.h>
#include <arm_sve.h>

#if defined(__cplusplus)
extern "C" {
#endif

/*  SME-specific packed buffer header (64-byte aligned).
 *  Layout optimized for SME outer product access patterns.
 */
typedef struct {
    nk_u32_t full_n_tiles;  // Number of full N tiles (SVL/16 rows each for F16)
    nk_u32_t full_k_tiles;  // Number of K tiles (SVL/16 elements for F16)
    nk_u32_t n_edge_rows;   // Remaining rows after full tiles
    nk_u32_t n_edge_offset; // Byte offset to edge data region
    nk_u32_t svl_bytes;     // SVL in bytes at pack time (for validation)
    nk_u32_t reserved[11];  // Padding to 64 bytes
} nk_dots_sme_packed_header_t;

/*  Get SVL (Streaming Vector Length) in bytes for F32 elements.
 *  On Apple M4 with SME2, this is typically 64 bytes (16 F32 elements).
 */
NK_INTERNAL nk_size_t nk_sme_svl_bytes_(void) __arm_streaming_compatible { return svcntw() * sizeof(nk_f32_t); }

/*  Get number of F16/BF16 elements per SVE vector in streaming mode.
 *  This is SVL/16 = 32 elements for 512-bit SVL.
 */
NK_INTERNAL nk_size_t nk_sme_f16_elements_(void) __arm_streaming_compatible { return svcnth(); }

/*  Get number of F32 elements per ZA tile row.
 *  This is SVL/32 = 16 elements for 512-bit SVL.
 */
NK_INTERNAL nk_size_t nk_sme_tile_dim_(void) __arm_streaming_compatible { return svcntw(); }

/*  Zero all 4 ZA32 tiles (tiles 0-3).
 *  Must be called at start of GEMM computation.
 */
NK_INTERNAL void nk_sme_zero_za32_(void) __arm_streaming __arm_inout("za") { svzero_za(); }

/*  F16 packed buffer size calculation.
 *  Layout: header + tiles for full N rows + N edge rows (row-major).
 *
 *  SME F16 tile dimensions:
 *    - Each F16 vector has SVL/16 elements (32 for 512-bit SVL)
 *    - Each ZA32 tile is SVL/32 × SVL/32 (16×16 for 512-bit SVL)
 *    - We tile N in increments of SVL/32 (16 rows)
 *    - We tile K in increments of SVL/16 (32 columns)
 */
NK_PUBLIC nk_size_t nk_dots_f16f16f32_packed_size_sme(nk_size_t n, nk_size_t k) {
    // Use runtime SVL query for accurate sizing
    nk_size_t const svl_bytes = 64;                             // Assume 512-bit SVL (Apple M4)
    nk_size_t const tile_rows = svl_bytes / sizeof(nk_f32_t);   // 16 rows per tile
    nk_size_t const tile_k_cols = svl_bytes / sizeof(nk_f16_t); // 32 K elements per tile

    nk_size_t const full_n_tiles = n / tile_rows;
    nk_size_t const tiles_along_k = (k + tile_k_cols - 1) / tile_k_cols;
    nk_size_t const n_edge_rows = n - full_n_tiles * tile_rows;

    // Header (64 bytes aligned)
    nk_size_t size = sizeof(nk_dots_sme_packed_header_t);

    // All tiles for full N rows (column-major for outer product access)
    // Each tile stores tile_rows × tile_k_cols F16 elements
    size += full_n_tiles * tiles_along_k * tile_rows * tile_k_cols * sizeof(nk_f16_t);

    // N edge: remaining rows for ALL K columns, stored row-major
    if (n_edge_rows > 0) size += n_edge_rows * k * sizeof(nk_f16_t);

    return size;
}

/*  BF16 packed buffer size calculation.
 */
NK_PUBLIC nk_size_t nk_dots_bf16bf16f32_packed_size_sme(nk_size_t n, nk_size_t k) {
    // Same dimensions as F16 since both are 16-bit
    return nk_dots_f16f16f32_packed_size_sme(n, k);
}

/*  Pack F16 B matrix for SME outer product access.
 *
 *  SME outer product: ZA[i,j] += A[i] * B[j]
 *  For GEMM C = A × B^T, we need B stored column-major so that
 *  loading a column of B gives us the elements for one N output row.
 *
 *  Layout: tiles are stored in (n_tile, k_tile) order with column-major
 *  element ordering within each tile.
 */
NK_PUBLIC void nk_dots_f16f16f32_pack_sme(       //
    nk_f16_t const *b, nk_size_t n, nk_size_t k, //
    nk_size_t b_stride, void *b_packed) {

    nk_size_t const svl_bytes = 64;                             // Assume 512-bit SVL
    nk_size_t const tile_rows = svl_bytes / sizeof(nk_f32_t);   // 16
    nk_size_t const tile_k_cols = svl_bytes / sizeof(nk_f16_t); // 32
    nk_size_t const tile_elements = tile_rows * tile_k_cols;    // 512
    nk_size_t const b_stride_elements = b_stride / sizeof(nk_f16_t);

    nk_size_t const num_n_tiles = n / tile_rows;
    nk_size_t const num_k_tiles = (k + tile_k_cols - 1) / tile_k_cols;
    nk_size_t const n_remainder_rows = n - num_n_tiles * tile_rows;
    nk_size_t const total_tiles = num_n_tiles * num_k_tiles;

    // Write header
    nk_dots_sme_packed_header_t *header = (nk_dots_sme_packed_header_t *)b_packed;
    header->full_n_tiles = (nk_u32_t)num_n_tiles;
    header->full_k_tiles = (nk_u32_t)num_k_tiles;
    header->n_edge_rows = (nk_u32_t)n_remainder_rows;
    header->svl_bytes = (nk_u32_t)svl_bytes;

    nk_size_t const tiles_offset = sizeof(nk_dots_sme_packed_header_t);
    nk_size_t const n_edge_offset = tiles_offset + total_tiles * tile_elements * sizeof(nk_f16_t);
    header->n_edge_offset = (nk_u32_t)n_edge_offset;

    nk_f16_t *tiles_ptr = (nk_f16_t *)((char *)b_packed + tiles_offset);
    nk_f16_t *n_edge_ptr = (nk_f16_t *)((char *)b_packed + n_edge_offset);

    // Zero-initialize tiles
    for (nk_size_t i = 0; i < total_tiles * tile_elements; i++) { tiles_ptr[i] = 0; }

    // Pack tiles: column-major within each tile for efficient SVE loads
    // For outer product, we load B as columns (one N index at a time)
    for (nk_size_t n_tile = 0; n_tile < num_n_tiles; n_tile++) {
        for (nk_size_t k_tile = 0; k_tile < num_k_tiles; k_tile++) {
            nk_size_t const tile_index = n_tile * num_k_tiles + k_tile;
            nk_f16_t *tile_output = tiles_ptr + tile_index * tile_elements;

            nk_size_t const src_row_start = n_tile * tile_rows;
            nk_size_t const src_col_start = k_tile * tile_k_cols;
            nk_size_t const cols_to_pack = (src_col_start + tile_k_cols <= k) ? tile_k_cols : (k - src_col_start);

            // Column-major packing: tile_output[col * tile_rows + row]
            for (nk_size_t row = 0; row < tile_rows; row++) {
                for (nk_size_t col = 0; col < cols_to_pack; col++) {
                    nk_size_t const src_idx = (src_row_start + row) * b_stride_elements + src_col_start + col;
                    nk_size_t const dst_idx = col * tile_rows + row;
                    tile_output[dst_idx] = b[src_idx];
                }
            }
        }
    }

    // Pack N-remainder in row-major for NEON fallback
    if (n_remainder_rows > 0) {
        nk_size_t const remainder_start_row = num_n_tiles * tile_rows;
        for (nk_size_t row = 0; row < n_remainder_rows; row++) {
            for (nk_size_t col = 0; col < k; col++) {
                n_edge_ptr[row * k + col] = b[(remainder_start_row + row) * b_stride_elements + col];
            }
        }
    }
}

/*  Pack BF16 B matrix for SME outer product access.
 */
NK_PUBLIC void nk_dots_bf16bf16f32_pack_sme(      //
    nk_bf16_t const *b, nk_size_t n, nk_size_t k, //
    nk_size_t b_stride, void *b_packed) {

    nk_size_t const svl_bytes = 64;
    nk_size_t const tile_rows = svl_bytes / sizeof(nk_f32_t);
    nk_size_t const tile_k_cols = svl_bytes / sizeof(nk_bf16_t);
    nk_size_t const tile_elements = tile_rows * tile_k_cols;
    nk_size_t const b_stride_elements = b_stride / sizeof(nk_bf16_t);

    nk_size_t const num_n_tiles = n / tile_rows;
    nk_size_t const num_k_tiles = (k + tile_k_cols - 1) / tile_k_cols;
    nk_size_t const n_remainder_rows = n - num_n_tiles * tile_rows;
    nk_size_t const total_tiles = num_n_tiles * num_k_tiles;

    nk_dots_sme_packed_header_t *header = (nk_dots_sme_packed_header_t *)b_packed;
    header->full_n_tiles = (nk_u32_t)num_n_tiles;
    header->full_k_tiles = (nk_u32_t)num_k_tiles;
    header->n_edge_rows = (nk_u32_t)n_remainder_rows;
    header->svl_bytes = (nk_u32_t)svl_bytes;

    nk_size_t const tiles_offset = sizeof(nk_dots_sme_packed_header_t);
    nk_size_t const n_edge_offset = tiles_offset + total_tiles * tile_elements * sizeof(nk_bf16_t);
    header->n_edge_offset = (nk_u32_t)n_edge_offset;

    nk_bf16_t *tiles_ptr = (nk_bf16_t *)((char *)b_packed + tiles_offset);
    nk_bf16_t *n_edge_ptr = (nk_bf16_t *)((char *)b_packed + n_edge_offset);

    for (nk_size_t i = 0; i < total_tiles * tile_elements; i++) { tiles_ptr[i] = 0; }

    for (nk_size_t n_tile = 0; n_tile < num_n_tiles; n_tile++) {
        for (nk_size_t k_tile = 0; k_tile < num_k_tiles; k_tile++) {
            nk_size_t const tile_index = n_tile * num_k_tiles + k_tile;
            nk_bf16_t *tile_output = tiles_ptr + tile_index * tile_elements;

            nk_size_t const src_row_start = n_tile * tile_rows;
            nk_size_t const src_col_start = k_tile * tile_k_cols;
            nk_size_t const cols_to_pack = (src_col_start + tile_k_cols <= k) ? tile_k_cols : (k - src_col_start);

            for (nk_size_t row = 0; row < tile_rows; row++) {
                for (nk_size_t col = 0; col < cols_to_pack; col++) {
                    nk_size_t const src_idx = (src_row_start + row) * b_stride_elements + src_col_start + col;
                    nk_size_t const dst_idx = col * tile_rows + row;
                    tile_output[dst_idx] = b[src_idx];
                }
            }
        }
    }

    if (n_remainder_rows > 0) {
        nk_size_t const remainder_start_row = num_n_tiles * tile_rows;
        for (nk_size_t row = 0; row < n_remainder_rows; row++) {
            for (nk_size_t col = 0; col < k; col++) {
                n_edge_ptr[row * k + col] = b[(remainder_start_row + row) * b_stride_elements + col];
            }
        }
    }
}

// SVL (Streaming Vector Length) constants for 512-bit SVL (Apple M4)
// These replace svcnt*() intrinsics to avoid streaming mode attribute issues
#define NK_SME_SVL_BYTES 64 // 512 bits = 64 bytes
#define NK_SME_CNTB      64 // byte elements per vector
#define NK_SME_CNTH      32 // half/F16 elements per vector
#define NK_SME_CNTW      16 // word/F32 elements per vector
#define NK_SME_CNTD      8  // double/F64 elements per vector

/*  F16 → F32 GEMM core kernel using SME outer products with multi-tile blocking.
 *
 *  Uses __arm_locally_streaming __arm_new("za") for streaming mode and ZA access.
 */
__arm_locally_streaming __arm_new("za") static void nk_dots_f16f16f32_sme_kernel_( //
    nk_f16_t const *a, void const *b_packed, nk_f32_t *c,                          //
    nk_size_t m, nk_size_t n, nk_size_t k,                                         //
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) {

    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_size_t const num_n_tiles = header->full_n_tiles;
    nk_size_t const num_k_tiles = header->full_k_tiles;

    nk_size_t const tile_dim = NK_SME_CNTW;    // 16 for 512-bit SVL
    nk_size_t const k_tile_size = NK_SME_CNTH; // 32 for 512-bit SVL
    nk_size_t const tile_elements = tile_dim * k_tile_size;

    nk_f16_t const *b_tiles = (nk_f16_t const *)((char const *)b_packed + sizeof(nk_dots_sme_packed_header_t));

    svbool_t const ptrue_h = svptrue_b16();
    svbool_t const ptrue_s = svptrue_b32();

    nk_size_t const num_m_tiles = m / tile_dim;
    nk_size_t const block_dim = tile_dim * 2; // 32 for 2×2 blocking

    // Process 2×2 tile blocks (32×32 output) for maximum tile utilization
    nk_size_t const num_m_blocks = num_m_tiles / 2;
    nk_size_t const num_n_blocks = num_n_tiles / 2;

    for (nk_size_t m_block = 0; m_block < num_m_blocks; m_block++) {
        nk_size_t const m_row = m_block * block_dim;

        for (nk_size_t n_block = 0; n_block < num_n_blocks; n_block++) {
            nk_size_t const n_col = n_block * block_dim;
            nk_size_t const n_tile_0 = n_block * 2;     // First N-tile in block
            nk_size_t const n_tile_1 = n_block * 2 + 1; // Second N-tile in block

            // Zero all 4 ZA32 tiles
            svzero_za();

            // Accumulate over K dimension
            for (nk_size_t k_tile = 0; k_tile < num_k_tiles; k_tile++) {
                nk_size_t const k_offset = k_tile * k_tile_size;

                // Get B tile pointers for both N-halves
                nk_size_t const b_tile_idx_0 = n_tile_0 * num_k_tiles + k_tile;
                nk_size_t const b_tile_idx_1 = n_tile_1 * num_k_tiles + k_tile;
                nk_f16_t const *b_tile_0 = b_tiles + b_tile_idx_0 * tile_elements;
                nk_f16_t const *b_tile_1 = b_tiles + b_tile_idx_1 * tile_elements;

                // Process all rows with 4-tile outer products
                for (nk_size_t row = 0; row < tile_dim; row++) {
                    // Load A vectors for M rows 0-15 and 16-31
                    nk_f16_t const *a_ptr_0 = a + (m_row + row) * a_stride_elements + k_offset;
                    nk_f16_t const *a_ptr_1 = a + (m_row + tile_dim + row) * a_stride_elements + k_offset;
                    svfloat16_t a_vec_0 = svld1_f16(ptrue_h, (float16_t const *)a_ptr_0);
                    svfloat16_t a_vec_1 = svld1_f16(ptrue_h, (float16_t const *)a_ptr_1);

                    // Load B vectors for N columns 0-15 and 16-31
                    svfloat16_t b_vec_0 = svld1_f16(ptrue_h, (float16_t const *)(b_tile_0 + row * k_tile_size));
                    svfloat16_t b_vec_1 = svld1_f16(ptrue_h, (float16_t const *)(b_tile_1 + row * k_tile_size));

                    // 4 outer products for 2×2 tile arrangement
                    svmopa_za32_f16_m(0, ptrue_s, ptrue_s, a_vec_0, b_vec_0);
                    svmopa_za32_f16_m(1, ptrue_s, ptrue_s, a_vec_0, b_vec_1);
                    svmopa_za32_f16_m(2, ptrue_s, ptrue_s, a_vec_1, b_vec_0);
                    svmopa_za32_f16_m(3, ptrue_s, ptrue_s, a_vec_1, b_vec_1);
                }
            }

            // Store all 4 tiles to C
            for (nk_size_t row = 0; row < tile_dim; row++) {
                // Tile 0: C[m:m+16, n:n+16]
                svst1_hor_za32(0, row, ptrue_s, c + (m_row + row) * c_stride_elements + n_col);
                // Tile 1: C[m:m+16, n+16:n+32]
                svst1_hor_za32(1, row, ptrue_s, c + (m_row + row) * c_stride_elements + n_col + tile_dim);
                // Tile 2: C[m+16:m+32, n:n+16]
                svst1_hor_za32(2, row, ptrue_s, c + (m_row + tile_dim + row) * c_stride_elements + n_col);
                // Tile 3: C[m+16:m+32, n+16:n+32]
                svst1_hor_za32(3, row, ptrue_s, c + (m_row + tile_dim + row) * c_stride_elements + n_col + tile_dim);
            }
        }

        // Handle odd N-tile at end of row (if num_n_tiles is odd)
        if (num_n_tiles % 2 != 0) {
            nk_size_t const n_tile = num_n_tiles - 1;
            nk_size_t const n_col = n_tile * tile_dim;

            // Process with 2 tiles (0 and 2) for the two M-halves
            svzero_za();

            for (nk_size_t k_tile = 0; k_tile < num_k_tiles; k_tile++) {
                nk_size_t const k_offset = k_tile * k_tile_size;
                nk_size_t const b_tile_idx = n_tile * num_k_tiles + k_tile;
                nk_f16_t const *b_tile = b_tiles + b_tile_idx * tile_elements;

                for (nk_size_t row = 0; row < tile_dim; row++) {
                    nk_f16_t const *a_ptr_0 = a + (m_row + row) * a_stride_elements + k_offset;
                    nk_f16_t const *a_ptr_1 = a + (m_row + tile_dim + row) * a_stride_elements + k_offset;
                    svfloat16_t a_vec_0 = svld1_f16(ptrue_h, (float16_t const *)a_ptr_0);
                    svfloat16_t a_vec_1 = svld1_f16(ptrue_h, (float16_t const *)a_ptr_1);
                    svfloat16_t b_vec = svld1_f16(ptrue_h, (float16_t const *)(b_tile + row * k_tile_size));

                    svmopa_za32_f16_m(0, ptrue_s, ptrue_s, a_vec_0, b_vec);
                    svmopa_za32_f16_m(2, ptrue_s, ptrue_s, a_vec_1, b_vec);
                }
            }

            for (nk_size_t row = 0; row < tile_dim; row++) {
                svst1_hor_za32(0, row, ptrue_s, c + (m_row + row) * c_stride_elements + n_col);
                svst1_hor_za32(2, row, ptrue_s, c + (m_row + tile_dim + row) * c_stride_elements + n_col);
            }
        }
    }

    // Handle odd M-tile at end (if num_m_tiles is odd)
    if (num_m_tiles % 2 != 0) {
        nk_size_t const m_tile = num_m_tiles - 1;
        nk_size_t const m_row = m_tile * tile_dim;

        // Process 1×2 tile blocks using tiles 0 and 1
        for (nk_size_t n_block = 0; n_block < num_n_blocks; n_block++) {
            nk_size_t const n_col = n_block * block_dim;
            nk_size_t const n_tile_0 = n_block * 2;
            nk_size_t const n_tile_1 = n_block * 2 + 1;

            svzero_za();

            for (nk_size_t k_tile = 0; k_tile < num_k_tiles; k_tile++) {
                nk_size_t const k_offset = k_tile * k_tile_size;
                nk_size_t const b_tile_idx_0 = n_tile_0 * num_k_tiles + k_tile;
                nk_size_t const b_tile_idx_1 = n_tile_1 * num_k_tiles + k_tile;
                nk_f16_t const *b_tile_0 = b_tiles + b_tile_idx_0 * tile_elements;
                nk_f16_t const *b_tile_1 = b_tiles + b_tile_idx_1 * tile_elements;

                for (nk_size_t row = 0; row < tile_dim; row++) {
                    nk_f16_t const *a_ptr = a + (m_row + row) * a_stride_elements + k_offset;
                    svfloat16_t a_vec = svld1_f16(ptrue_h, (float16_t const *)a_ptr);
                    svfloat16_t b_vec_0 = svld1_f16(ptrue_h, (float16_t const *)(b_tile_0 + row * k_tile_size));
                    svfloat16_t b_vec_1 = svld1_f16(ptrue_h, (float16_t const *)(b_tile_1 + row * k_tile_size));

                    svmopa_za32_f16_m(0, ptrue_s, ptrue_s, a_vec, b_vec_0);
                    svmopa_za32_f16_m(1, ptrue_s, ptrue_s, a_vec, b_vec_1);
                }
            }

            for (nk_size_t row = 0; row < tile_dim; row++) {
                svst1_hor_za32(0, row, ptrue_s, c + (m_row + row) * c_stride_elements + n_col);
                svst1_hor_za32(1, row, ptrue_s, c + (m_row + row) * c_stride_elements + n_col + tile_dim);
            }
        }

        // Handle final single tile (odd M × odd N)
        if (num_n_tiles % 2 != 0) {
            nk_size_t const n_tile = num_n_tiles - 1;
            nk_size_t const n_col = n_tile * tile_dim;

            svzero_za();

            for (nk_size_t k_tile = 0; k_tile < num_k_tiles; k_tile++) {
                nk_size_t const k_offset = k_tile * k_tile_size;
                nk_size_t const b_tile_idx = n_tile * num_k_tiles + k_tile;
                nk_f16_t const *b_tile = b_tiles + b_tile_idx * tile_elements;

                for (nk_size_t row = 0; row < tile_dim; row++) {
                    nk_f16_t const *a_ptr = a + (m_row + row) * a_stride_elements + k_offset;
                    svfloat16_t a_vec = svld1_f16(ptrue_h, (float16_t const *)a_ptr);
                    svfloat16_t b_vec = svld1_f16(ptrue_h, (float16_t const *)(b_tile + row * k_tile_size));

                    svmopa_za32_f16_m(0, ptrue_s, ptrue_s, a_vec, b_vec);
                }
            }

            for (nk_size_t row = 0; row < tile_dim; row++) {
                svst1_hor_za32(0, row, ptrue_s, c + (m_row + row) * c_stride_elements + n_col);
            }
        }
    }
}

/*  BF16 → F32 GEMM core kernel using SME outer products with multi-tile blocking.
 *  Same optimization strategy as F16: uses all 4 ZA32 tiles for 2×2 output blocking.
 *  Note: Cannot use NK_INTERNAL (always_inline) due to streaming mode constraints.
 */
__arm_locally_streaming __arm_new("za") static void nk_dots_bf16bf16f32_sme_kernel_( //
    nk_bf16_t const *a, void const *b_packed, nk_f32_t *c,                           //
    nk_size_t m, nk_size_t n, nk_size_t k,                                           //
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) {

    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_size_t const num_n_tiles = header->full_n_tiles;
    nk_size_t const num_k_tiles = header->full_k_tiles;

    nk_size_t const tile_dim = NK_SME_CNTW;    // 16 for 512-bit SVL
    nk_size_t const k_tile_size = NK_SME_CNTH; // 32 for 512-bit SVL
    nk_size_t const tile_elements = tile_dim * k_tile_size;

    nk_bf16_t const *b_tiles = (nk_bf16_t const *)((char const *)b_packed + sizeof(nk_dots_sme_packed_header_t));

    svbool_t const ptrue_h = svptrue_b16();
    svbool_t const ptrue_s = svptrue_b32();

    nk_size_t const num_m_tiles = m / tile_dim;
    nk_size_t const block_dim = tile_dim * 2; // 32 for 2×2 blocking

    // Process 2×2 tile blocks (32×32 output) for maximum tile utilization
    nk_size_t const num_m_blocks = num_m_tiles / 2;
    nk_size_t const num_n_blocks = num_n_tiles / 2;

    for (nk_size_t m_block = 0; m_block < num_m_blocks; m_block++) {
        nk_size_t const m_row = m_block * block_dim;

        for (nk_size_t n_block = 0; n_block < num_n_blocks; n_block++) {
            nk_size_t const n_col = n_block * block_dim;
            nk_size_t const n_tile_0 = n_block * 2;
            nk_size_t const n_tile_1 = n_block * 2 + 1;

            svzero_za();

            // Accumulate over K dimension
            for (nk_size_t k_tile = 0; k_tile < num_k_tiles; k_tile++) {
                nk_size_t const k_offset = k_tile * k_tile_size;

                nk_size_t const b_tile_idx_0 = n_tile_0 * num_k_tiles + k_tile;
                nk_size_t const b_tile_idx_1 = n_tile_1 * num_k_tiles + k_tile;
                nk_bf16_t const *b_tile_0 = b_tiles + b_tile_idx_0 * tile_elements;
                nk_bf16_t const *b_tile_1 = b_tiles + b_tile_idx_1 * tile_elements;

                for (nk_size_t row = 0; row < tile_dim; row++) {
                    nk_bf16_t const *a_ptr_0 = a + (m_row + row) * a_stride_elements + k_offset;
                    nk_bf16_t const *a_ptr_1 = a + (m_row + tile_dim + row) * a_stride_elements + k_offset;
                    svbfloat16_t a_vec_0 = svld1_bf16(ptrue_h, (bfloat16_t const *)a_ptr_0);
                    svbfloat16_t a_vec_1 = svld1_bf16(ptrue_h, (bfloat16_t const *)a_ptr_1);

                    svbfloat16_t b_vec_0 = svld1_bf16(ptrue_h, (bfloat16_t const *)(b_tile_0 + row * k_tile_size));
                    svbfloat16_t b_vec_1 = svld1_bf16(ptrue_h, (bfloat16_t const *)(b_tile_1 + row * k_tile_size));

                    svmopa_za32_bf16_m(0, ptrue_s, ptrue_s, a_vec_0, b_vec_0);
                    svmopa_za32_bf16_m(1, ptrue_s, ptrue_s, a_vec_0, b_vec_1);
                    svmopa_za32_bf16_m(2, ptrue_s, ptrue_s, a_vec_1, b_vec_0);
                    svmopa_za32_bf16_m(3, ptrue_s, ptrue_s, a_vec_1, b_vec_1);
                }
            }

            for (nk_size_t row = 0; row < tile_dim; row++) {
                svst1_hor_za32(0, row, ptrue_s, c + (m_row + row) * c_stride_elements + n_col);
                svst1_hor_za32(1, row, ptrue_s, c + (m_row + row) * c_stride_elements + n_col + tile_dim);
                svst1_hor_za32(2, row, ptrue_s, c + (m_row + tile_dim + row) * c_stride_elements + n_col);
                svst1_hor_za32(3, row, ptrue_s, c + (m_row + tile_dim + row) * c_stride_elements + n_col + tile_dim);
            }
        }

        // Handle odd N-tile at end of row
        if (num_n_tiles % 2 != 0) {
            nk_size_t const n_tile = num_n_tiles - 1;
            nk_size_t const n_col = n_tile * tile_dim;

            svzero_za();

            for (nk_size_t k_tile = 0; k_tile < num_k_tiles; k_tile++) {
                nk_size_t const k_offset = k_tile * k_tile_size;
                nk_size_t const b_tile_idx = n_tile * num_k_tiles + k_tile;
                nk_bf16_t const *b_tile = b_tiles + b_tile_idx * tile_elements;

                for (nk_size_t row = 0; row < tile_dim; row++) {
                    nk_bf16_t const *a_ptr_0 = a + (m_row + row) * a_stride_elements + k_offset;
                    nk_bf16_t const *a_ptr_1 = a + (m_row + tile_dim + row) * a_stride_elements + k_offset;
                    svbfloat16_t a_vec_0 = svld1_bf16(ptrue_h, (bfloat16_t const *)a_ptr_0);
                    svbfloat16_t a_vec_1 = svld1_bf16(ptrue_h, (bfloat16_t const *)a_ptr_1);
                    svbfloat16_t b_vec = svld1_bf16(ptrue_h, (bfloat16_t const *)(b_tile + row * k_tile_size));

                    svmopa_za32_bf16_m(0, ptrue_s, ptrue_s, a_vec_0, b_vec);
                    svmopa_za32_bf16_m(2, ptrue_s, ptrue_s, a_vec_1, b_vec);
                }
            }

            for (nk_size_t row = 0; row < tile_dim; row++) {
                svst1_hor_za32(0, row, ptrue_s, c + (m_row + row) * c_stride_elements + n_col);
                svst1_hor_za32(2, row, ptrue_s, c + (m_row + tile_dim + row) * c_stride_elements + n_col);
            }
        }
    }

    // Handle odd M-tile at end
    if (num_m_tiles % 2 != 0) {
        nk_size_t const m_tile = num_m_tiles - 1;
        nk_size_t const m_row = m_tile * tile_dim;

        for (nk_size_t n_block = 0; n_block < num_n_blocks; n_block++) {
            nk_size_t const n_col = n_block * block_dim;
            nk_size_t const n_tile_0 = n_block * 2;
            nk_size_t const n_tile_1 = n_block * 2 + 1;

            svzero_za();

            for (nk_size_t k_tile = 0; k_tile < num_k_tiles; k_tile++) {
                nk_size_t const k_offset = k_tile * k_tile_size;
                nk_size_t const b_tile_idx_0 = n_tile_0 * num_k_tiles + k_tile;
                nk_size_t const b_tile_idx_1 = n_tile_1 * num_k_tiles + k_tile;
                nk_bf16_t const *b_tile_0 = b_tiles + b_tile_idx_0 * tile_elements;
                nk_bf16_t const *b_tile_1 = b_tiles + b_tile_idx_1 * tile_elements;

                for (nk_size_t row = 0; row < tile_dim; row++) {
                    nk_bf16_t const *a_ptr = a + (m_row + row) * a_stride_elements + k_offset;
                    svbfloat16_t a_vec = svld1_bf16(ptrue_h, (bfloat16_t const *)a_ptr);
                    svbfloat16_t b_vec_0 = svld1_bf16(ptrue_h, (bfloat16_t const *)(b_tile_0 + row * k_tile_size));
                    svbfloat16_t b_vec_1 = svld1_bf16(ptrue_h, (bfloat16_t const *)(b_tile_1 + row * k_tile_size));

                    svmopa_za32_bf16_m(0, ptrue_s, ptrue_s, a_vec, b_vec_0);
                    svmopa_za32_bf16_m(1, ptrue_s, ptrue_s, a_vec, b_vec_1);
                }
            }

            for (nk_size_t row = 0; row < tile_dim; row++) {
                svst1_hor_za32(0, row, ptrue_s, c + (m_row + row) * c_stride_elements + n_col);
                svst1_hor_za32(1, row, ptrue_s, c + (m_row + row) * c_stride_elements + n_col + tile_dim);
            }
        }

        // Handle final single tile
        if (num_n_tiles % 2 != 0) {
            nk_size_t const n_tile = num_n_tiles - 1;
            nk_size_t const n_col = n_tile * tile_dim;

            svzero_za();

            for (nk_size_t k_tile = 0; k_tile < num_k_tiles; k_tile++) {
                nk_size_t const k_offset = k_tile * k_tile_size;
                nk_size_t const b_tile_idx = n_tile * num_k_tiles + k_tile;
                nk_bf16_t const *b_tile = b_tiles + b_tile_idx * tile_elements;

                for (nk_size_t row = 0; row < tile_dim; row++) {
                    nk_bf16_t const *a_ptr = a + (m_row + row) * a_stride_elements + k_offset;
                    svbfloat16_t a_vec = svld1_bf16(ptrue_h, (bfloat16_t const *)a_ptr);
                    svbfloat16_t b_vec = svld1_bf16(ptrue_h, (bfloat16_t const *)(b_tile + row * k_tile_size));

                    svmopa_za32_bf16_m(0, ptrue_s, ptrue_s, a_vec, b_vec);
                }
            }

            for (nk_size_t row = 0; row < tile_dim; row++) {
                svst1_hor_za32(0, row, ptrue_s, c + (m_row + row) * c_stride_elements + n_col);
            }
        }
    }
}

/*  F16 → F32 GEMM public interface.
 *  Handles streaming mode transitions and edge cases with NEON fallback.
 *
 *  @param a         Input matrix A (M × K), row-major
 *  @param b_packed  Pre-packed B matrix from nk_dots_f16f16f32_pack_sme
 *  @param c         Output matrix C (M × N), row-major
 *  @param m         Number of rows in A and C
 *  @param n         Number of columns in C (rows in original B)
 *  @param k         Shared dimension (columns in A, columns in original B)
 *  @param a_stride  Byte stride between rows of A
 *  @param c_stride  Byte stride between rows of C
 */
NK_PUBLIC void nk_dots_f16f16f32_sme(                     //
    nk_f16_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t m, nk_size_t n, nk_size_t k,                //
    nk_size_t a_stride, nk_size_t c_stride) {

    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_size_t const num_n_tiles = header->full_n_tiles;
    nk_size_t const n_edge_rows = header->n_edge_rows;

    nk_size_t const a_stride_elements = a_stride / sizeof(nk_f16_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);

    // Assume 512-bit SVL (Apple M4)
    nk_size_t const tile_dim = 16;
    nk_size_t const num_m_tiles = m / tile_dim;
    nk_size_t const full_n_cols = num_n_tiles * tile_dim;

    // SME kernel for full tiles
    if (num_m_tiles > 0 && num_n_tiles > 0) {
        nk_dots_f16f16f32_sme_kernel_(a, b_packed, c, num_m_tiles * tile_dim, full_n_cols, k, a_stride_elements,
                                      c_stride_elements);
    }

    // Scalar fallback for N-edge (columns beyond full N-tiles)
    if (n_edge_rows > 0) {
        nk_f16_t const *n_edge_ptr = (nk_f16_t const *)((char const *)b_packed + header->n_edge_offset);
        for (nk_size_t i = 0; i < m; i++) {
            nk_f16_t const *a_row = a + i * a_stride_elements;
            for (nk_size_t j = 0; j < n_edge_rows; j++) {
                nk_f32_t acc = 0.0f;
                for (nk_size_t kk = 0; kk < k; kk++) {
                    nk_f32_t a_val, b_val;
                    nk_f16_to_f32(&a_row[kk], &a_val);
                    nk_f16_to_f32(&n_edge_ptr[j * k + kk], &b_val);
                    acc += a_val * b_val;
                }
                c[i * c_stride_elements + full_n_cols + j] = acc;
            }
        }
    }

    // Scalar fallback for M-edge (rows beyond full M-tiles)
    if (m > num_m_tiles * tile_dim && num_n_tiles > 0) {
        nk_size_t const m_remainder_start = num_m_tiles * tile_dim;
        nk_size_t const m_remainder_count = m - m_remainder_start;

        // Process M-remainder with SME kernel (it handles partial M internally)
        // For simplicity, use scalar fallback here
        nk_f16_t const *b_tiles = (nk_f16_t const *)((char const *)b_packed + sizeof(nk_dots_sme_packed_header_t));
        nk_size_t const num_k_tiles = header->full_k_tiles;
        nk_size_t const k_tile_size = 32; // SVL/16 for F16

        for (nk_size_t i = 0; i < m_remainder_count; i++) {
            nk_f16_t const *a_row = a + (m_remainder_start + i) * a_stride_elements;
            for (nk_size_t n_tile = 0; n_tile < num_n_tiles; n_tile++) {
                nk_f32_t acc[16] = {0};
                for (nk_size_t k_tile = 0; k_tile < num_k_tiles; k_tile++) {
                    nk_size_t const k_offset = k_tile * k_tile_size;
                    nk_size_t const b_tile_idx = n_tile * num_k_tiles + k_tile;
                    nk_f16_t const *b_tile = b_tiles + b_tile_idx * tile_dim * k_tile_size;

                    for (nk_size_t j = 0; j < tile_dim; j++) {
                        for (nk_size_t kk = 0; kk < k_tile_size && k_offset + kk < k; kk++) {
                            nk_f32_t a_val, b_val;
                            nk_f16_to_f32(&a_row[k_offset + kk], &a_val);
                            nk_f16_to_f32(&b_tile[kk * tile_dim + j], &b_val);
                            acc[j] += a_val * b_val;
                        }
                    }
                }
                for (nk_size_t j = 0; j < tile_dim; j++) {
                    c[(m_remainder_start + i) * c_stride_elements + n_tile * tile_dim + j] = acc[j];
                }
            }
        }
    }

    // M-edge × N-edge corner
    if (m > num_m_tiles * tile_dim && n_edge_rows > 0) {
        nk_size_t const m_remainder_start = num_m_tiles * tile_dim;
        nk_size_t const m_remainder_count = m - m_remainder_start;
        nk_f16_t const *n_edge_ptr = (nk_f16_t const *)((char const *)b_packed + header->n_edge_offset);

        for (nk_size_t i = 0; i < m_remainder_count; i++) {
            nk_f16_t const *a_row = a + (m_remainder_start + i) * a_stride_elements;
            for (nk_size_t j = 0; j < n_edge_rows; j++) {
                nk_f32_t acc = 0.0f;
                for (nk_size_t kk = 0; kk < k; kk++) {
                    nk_f32_t a_val, b_val;
                    nk_f16_to_f32(&a_row[kk], &a_val);
                    nk_f16_to_f32(&n_edge_ptr[j * k + kk], &b_val);
                    acc += a_val * b_val;
                }
                c[(m_remainder_start + i) * c_stride_elements + full_n_cols + j] = acc;
            }
        }
    }
}

/*  BF16 → F32 GEMM public interface.
 */
NK_PUBLIC void nk_dots_bf16bf16f32_sme(                    //
    nk_bf16_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t m, nk_size_t n, nk_size_t k,                 //
    nk_size_t a_stride, nk_size_t c_stride) {

    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_size_t const num_n_tiles = header->full_n_tiles;
    nk_size_t const n_edge_rows = header->n_edge_rows;

    nk_size_t const a_stride_elements = a_stride / sizeof(nk_bf16_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);

    nk_size_t const tile_dim = 16;
    nk_size_t const num_m_tiles = m / tile_dim;
    nk_size_t const full_n_cols = num_n_tiles * tile_dim;

    if (num_m_tiles > 0 && num_n_tiles > 0) {
        nk_dots_bf16bf16f32_sme_kernel_(a, b_packed, c, num_m_tiles * tile_dim, full_n_cols, k, a_stride_elements,
                                        c_stride_elements);
    }

    if (n_edge_rows > 0) {
        nk_bf16_t const *n_edge_ptr = (nk_bf16_t const *)((char const *)b_packed + header->n_edge_offset);
        for (nk_size_t i = 0; i < m; i++) {
            nk_bf16_t const *a_row = a + i * a_stride_elements;
            for (nk_size_t j = 0; j < n_edge_rows; j++) {
                nk_f32_t acc = 0.0f;
                for (nk_size_t kk = 0; kk < k; kk++) {
                    nk_f32_t a_val, b_val;
                    nk_bf16_to_f32(&a_row[kk], &a_val);
                    nk_bf16_to_f32(&n_edge_ptr[j * k + kk], &b_val);
                    acc += a_val * b_val;
                }
                c[i * c_stride_elements + full_n_cols + j] = acc;
            }
        }
    }

    if (m > num_m_tiles * tile_dim && num_n_tiles > 0) {
        nk_size_t const m_remainder_start = num_m_tiles * tile_dim;
        nk_size_t const m_remainder_count = m - m_remainder_start;

        nk_bf16_t const *b_tiles = (nk_bf16_t const *)((char const *)b_packed + sizeof(nk_dots_sme_packed_header_t));
        nk_size_t const num_k_tiles = header->full_k_tiles;
        nk_size_t const k_tile_size = 32;

        for (nk_size_t i = 0; i < m_remainder_count; i++) {
            nk_bf16_t const *a_row = a + (m_remainder_start + i) * a_stride_elements;
            for (nk_size_t n_tile = 0; n_tile < num_n_tiles; n_tile++) {
                nk_f32_t acc[16] = {0};
                for (nk_size_t k_tile = 0; k_tile < num_k_tiles; k_tile++) {
                    nk_size_t const k_offset = k_tile * k_tile_size;
                    nk_size_t const b_tile_idx = n_tile * num_k_tiles + k_tile;
                    nk_bf16_t const *b_tile = b_tiles + b_tile_idx * tile_dim * k_tile_size;

                    for (nk_size_t j = 0; j < tile_dim; j++) {
                        for (nk_size_t kk = 0; kk < k_tile_size && k_offset + kk < k; kk++) {
                            nk_f32_t a_val, b_val;
                            nk_bf16_to_f32(&a_row[k_offset + kk], &a_val);
                            nk_bf16_to_f32(&b_tile[kk * tile_dim + j], &b_val);
                            acc[j] += a_val * b_val;
                        }
                    }
                }
                for (nk_size_t j = 0; j < tile_dim; j++) {
                    c[(m_remainder_start + i) * c_stride_elements + n_tile * tile_dim + j] = acc[j];
                }
            }
        }
    }

    if (m > num_m_tiles * tile_dim && n_edge_rows > 0) {
        nk_size_t const m_remainder_start = num_m_tiles * tile_dim;
        nk_size_t const m_remainder_count = m - m_remainder_start;
        nk_bf16_t const *n_edge_ptr = (nk_bf16_t const *)((char const *)b_packed + header->n_edge_offset);

        for (nk_size_t i = 0; i < m_remainder_count; i++) {
            nk_bf16_t const *a_row = a + (m_remainder_start + i) * a_stride_elements;
            for (nk_size_t j = 0; j < n_edge_rows; j++) {
                nk_f32_t acc = 0.0f;
                for (nk_size_t kk = 0; kk < k; kk++) {
                    nk_f32_t a_val, b_val;
                    nk_bf16_to_f32(&a_row[kk], &a_val);
                    nk_bf16_to_f32(&n_edge_ptr[j * k + kk], &b_val);
                    acc += a_val * b_val;
                }
                c[(m_remainder_start + i) * c_stride_elements + full_n_cols + j] = acc;
            }
        }
    }
}

/*  ============================================================================
 *  F32×F32 → F32 GEMM with F64 accumulation for high precision.
 *
 *  Uses svmopa_za64_f64_m for F64×F64 → F64 outer products, which requires
 *  FEAT_SME_F64F64 (available on Apple M4).
 *
 *  Trade-offs vs F32 accumulation:
 *    - Higher precision: ~15 decimal digits vs ~7
 *    - Lower throughput: ZA64 tile is 8×8 vs ZA32's 16×16 (4x fewer elements)
 *    - Conversion overhead: F32 → F64 on input, F64 → F32 on output
 *  ============================================================================
 */

/*  F32 packed buffer size calculation for F64-accumulation GEMM.
 *
 *  Layout uses ZA64 tile dimensions (8×8 for 512-bit SVL):
 *    - Each F64 vector has SVL/64 elements (8 for 512-bit SVL)
 *    - Each ZA64 tile is SVL/64 × SVL/64 (8×8 for 512-bit SVL)
 *    - We tile N in increments of 8 rows
 *    - We tile K in increments of 8 columns
 */
NK_PUBLIC nk_size_t nk_dots_f32f32f32_smef64_packed_size(nk_size_t n, nk_size_t k) {
    nk_size_t const svl_bytes = 64;                             // Assume 512-bit SVL (Apple M4)
    nk_size_t const tile_rows = svl_bytes / sizeof(nk_f64_t);   // 8 rows per ZA64 tile
    nk_size_t const tile_k_cols = svl_bytes / sizeof(nk_f32_t); // 16 F32 K elements per tile

    nk_size_t const full_n_tiles = n / tile_rows;
    nk_size_t const tiles_along_k = (k + tile_k_cols - 1) / tile_k_cols;
    nk_size_t const n_edge_rows = n - full_n_tiles * tile_rows;

    // Header (64 bytes aligned)
    nk_size_t size = sizeof(nk_dots_sme_packed_header_t);

    // All tiles for full N rows (column-major for outer product access)
    // Each tile stores tile_rows × tile_k_cols F32 elements
    size += full_n_tiles * tiles_along_k * tile_rows * tile_k_cols * sizeof(nk_f32_t);

    // N edge: remaining rows for ALL K columns, stored row-major
    if (n_edge_rows > 0) size += n_edge_rows * k * sizeof(nk_f32_t);

    return size;
}

/*  Pack F32 B matrix for SME F64-accumulation outer product access.
 */
NK_PUBLIC void nk_dots_f32f32f32_smef64_pack(    //
    nk_f32_t const *b, nk_size_t n, nk_size_t k, //
    nk_size_t b_stride, void *b_packed) {

    nk_size_t const svl_bytes = 64;
    nk_size_t const tile_rows = svl_bytes / sizeof(nk_f64_t);   // 8
    nk_size_t const tile_k_cols = svl_bytes / sizeof(nk_f32_t); // 16
    nk_size_t const tile_elements = tile_rows * tile_k_cols;    // 128
    nk_size_t const b_stride_elements = b_stride / sizeof(nk_f32_t);

    nk_size_t const num_n_tiles = n / tile_rows;
    nk_size_t const num_k_tiles = (k + tile_k_cols - 1) / tile_k_cols;
    nk_size_t const n_remainder_rows = n - num_n_tiles * tile_rows;
    nk_size_t const total_tiles = num_n_tiles * num_k_tiles;

    // Write header
    nk_dots_sme_packed_header_t *header = (nk_dots_sme_packed_header_t *)b_packed;
    header->full_n_tiles = (nk_u32_t)num_n_tiles;
    header->full_k_tiles = (nk_u32_t)num_k_tiles;
    header->n_edge_rows = (nk_u32_t)n_remainder_rows;
    header->svl_bytes = (nk_u32_t)svl_bytes;

    nk_size_t const tiles_offset = sizeof(nk_dots_sme_packed_header_t);
    nk_size_t const n_edge_offset = tiles_offset + total_tiles * tile_elements * sizeof(nk_f32_t);
    header->n_edge_offset = (nk_u32_t)n_edge_offset;

    nk_f32_t *tiles_ptr = (nk_f32_t *)((char *)b_packed + tiles_offset);
    nk_f32_t *n_edge_ptr = (nk_f32_t *)((char *)b_packed + n_edge_offset);

    // Zero-initialize tiles
    for (nk_size_t i = 0; i < total_tiles * tile_elements; i++) { tiles_ptr[i] = 0.0f; }

    // Pack tiles: column-major within each tile for efficient SVE loads
    for (nk_size_t n_tile = 0; n_tile < num_n_tiles; n_tile++) {
        for (nk_size_t k_tile = 0; k_tile < num_k_tiles; k_tile++) {
            nk_size_t const tile_index = n_tile * num_k_tiles + k_tile;
            nk_f32_t *tile_output = tiles_ptr + tile_index * tile_elements;

            nk_size_t const src_row_start = n_tile * tile_rows;
            nk_size_t const src_col_start = k_tile * tile_k_cols;
            nk_size_t const cols_to_pack = (src_col_start + tile_k_cols <= k) ? tile_k_cols : (k - src_col_start);

            // Column-major packing: tile_output[col * tile_rows + row]
            for (nk_size_t row = 0; row < tile_rows; row++) {
                for (nk_size_t col = 0; col < cols_to_pack; col++) {
                    nk_size_t const src_idx = (src_row_start + row) * b_stride_elements + src_col_start + col;
                    nk_size_t const dst_idx = col * tile_rows + row;
                    tile_output[dst_idx] = b[src_idx];
                }
            }
        }
    }

    // Pack N-remainder in row-major for NEON fallback
    if (n_remainder_rows > 0) {
        nk_size_t const remainder_start_row = num_n_tiles * tile_rows;
        for (nk_size_t row = 0; row < n_remainder_rows; row++) {
            for (nk_size_t col = 0; col < k; col++) {
                n_edge_ptr[row * k + col] = b[(remainder_start_row + row) * b_stride_elements + col];
            }
        }
    }
}

/*  F32 → F64 → F32 GEMM core kernel using SME outer products with F64 accumulation.
 *
 *  Uses 4 ZA64 tiles (8×8 each) for 2×2 blocking (16×16 output blocks):
 *  - Tile 0: C[m:m+8, n:n+8]
 *  - Tile 1: C[m:m+8, n+8:n+16]
 *  - Tile 2: C[m+8:m+16, n:n+8]
 *  - Tile 3: C[m+8:m+16, n+8:n+16]
 *
 *  Algorithm:
 *  1. Load F32 inputs and convert to F64
 *  2. Perform svmopa_za64_f64_m outer product accumulate on 4 tiles
 *  3. Convert F64 results back to F32 when storing
 *
 *  Requires FEAT_SME_F64F64 (available on Apple M4).
 */
__arm_locally_streaming __arm_new("za") static void nk_dots_f32f32f32_smef64_kernel_( //
    nk_f32_t const *a, void const *b_packed, nk_f32_t *c,                             //
    nk_size_t m, nk_size_t n, nk_size_t k,                                            //
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) {

    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_size_t const num_n_tiles = header->full_n_tiles;
    nk_size_t const num_k_tiles = header->full_k_tiles;

    nk_size_t const tile_dim = NK_SME_CNTD;    // 8 for 512-bit SVL (ZA64 tile dimension)
    nk_size_t const k_tile_size = NK_SME_CNTW; // 16 for 512-bit SVL (F32 elements per K-tile)
    nk_size_t const tile_elements = tile_dim * k_tile_size;

    nk_f32_t const *b_tiles = (nk_f32_t const *)((char const *)b_packed + sizeof(nk_dots_sme_packed_header_t));

    svbool_t const ptrue_s = svptrue_b32();
    svbool_t const ptrue_d = svptrue_b64();

    nk_size_t const num_m_tiles = m / tile_dim;
    nk_size_t const block_dim = tile_dim * 2; // 16 for 2×2 blocking

    // Process 2×2 tile blocks (16×16 output) using 4 of 8 ZA64 tiles
    nk_size_t const num_m_blocks = num_m_tiles / 2;
    nk_size_t const num_n_blocks = num_n_tiles / 2;

    for (nk_size_t m_block = 0; m_block < num_m_blocks; m_block++) {
        nk_size_t const m_row = m_block * block_dim;

        for (nk_size_t n_block = 0; n_block < num_n_blocks; n_block++) {
            nk_size_t const n_col = n_block * block_dim;
            nk_size_t const n_tile_0 = n_block * 2;
            nk_size_t const n_tile_1 = n_block * 2 + 1;

            // Zero all 4 ZA64 tiles
            svzero_za();

            // Accumulate over K dimension
            for (nk_size_t k_tile = 0; k_tile < num_k_tiles; k_tile++) {
                nk_size_t const k_offset = k_tile * k_tile_size;

                // Get B tile pointers for both N-halves
                nk_size_t const b_tile_idx_0 = n_tile_0 * num_k_tiles + k_tile;
                nk_size_t const b_tile_idx_1 = n_tile_1 * num_k_tiles + k_tile;
                nk_f32_t const *b_tile_0 = b_tiles + b_tile_idx_0 * tile_elements;
                nk_f32_t const *b_tile_1 = b_tiles + b_tile_idx_1 * tile_elements;

                // Process K in chunks of 8 (F64 vector width)
                for (nk_size_t k_sub = 0; k_sub < k_tile_size; k_sub += tile_dim) {
                    // Process each row within the K-chunk
                    for (nk_size_t row = 0; row < tile_dim; row++) {
                        // Load A vectors for both M-halves
                        nk_f32_t const *a_ptr_0 = a + (m_row + row) * a_stride_elements + k_offset + k_sub;
                        nk_f32_t const *a_ptr_1 = a + (m_row + tile_dim + row) * a_stride_elements + k_offset + k_sub;
                        svfloat64_t a_vec_0 = svcvt_f64_f32_x(ptrue_d, svld1_f32(ptrue_s, a_ptr_0));
                        svfloat64_t a_vec_1 = svcvt_f64_f32_x(ptrue_d, svld1_f32(ptrue_s, a_ptr_1));

                        // Load B vectors for both N-halves
                        nk_f32_t const *b_col_0 = b_tile_0 + (k_sub + row) * tile_dim;
                        nk_f32_t const *b_col_1 = b_tile_1 + (k_sub + row) * tile_dim;
                        svfloat64_t b_vec_0 = svcvt_f64_f32_x(ptrue_d, svld1_f32(ptrue_s, b_col_0));
                        svfloat64_t b_vec_1 = svcvt_f64_f32_x(ptrue_d, svld1_f32(ptrue_s, b_col_1));

                        // 4 outer products for 2×2 tile arrangement
                        svmopa_za64_f64_m(0, ptrue_d, ptrue_d, a_vec_0, b_vec_0); // [0:8, 0:8]
                        svmopa_za64_f64_m(1, ptrue_d, ptrue_d, a_vec_0, b_vec_1); // [0:8, 8:16]
                        svmopa_za64_f64_m(2, ptrue_d, ptrue_d, a_vec_1, b_vec_0); // [8:16, 0:8]
                        svmopa_za64_f64_m(3, ptrue_d, ptrue_d, a_vec_1, b_vec_1); // [8:16, 8:16]
                    }
                }
            }

            // Store all 4 tiles to C (convert F64 back to F32)
            for (nk_size_t row = 0; row < tile_dim; row++) {
                nk_f64_t za_row_0[8], za_row_1[8], za_row_2[8], za_row_3[8];
                svst1_hor_za64(0, row, ptrue_d, za_row_0);
                svst1_hor_za64(1, row, ptrue_d, za_row_1);
                svst1_hor_za64(2, row, ptrue_d, za_row_2);
                svst1_hor_za64(3, row, ptrue_d, za_row_3);

                nk_f32_t *c_row_0 = c + (m_row + row) * c_stride_elements + n_col;
                nk_f32_t *c_row_1 = c + (m_row + tile_dim + row) * c_stride_elements + n_col;
                for (nk_size_t col = 0; col < tile_dim; col++) {
                    c_row_0[col] = (nk_f32_t)za_row_0[col];
                    c_row_0[tile_dim + col] = (nk_f32_t)za_row_1[col];
                    c_row_1[col] = (nk_f32_t)za_row_2[col];
                    c_row_1[tile_dim + col] = (nk_f32_t)za_row_3[col];
                }
            }
        }

        // Handle odd N-tile at end of row
        if (num_n_tiles % 2 != 0) {
            nk_size_t const n_tile = num_n_tiles - 1;
            nk_size_t const n_col = n_tile * tile_dim;

            svzero_za();

            for (nk_size_t k_tile = 0; k_tile < num_k_tiles; k_tile++) {
                nk_size_t const k_offset = k_tile * k_tile_size;
                nk_size_t const b_tile_idx = n_tile * num_k_tiles + k_tile;
                nk_f32_t const *b_tile = b_tiles + b_tile_idx * tile_elements;

                for (nk_size_t k_sub = 0; k_sub < k_tile_size; k_sub += tile_dim) {
                    for (nk_size_t row = 0; row < tile_dim; row++) {
                        nk_f32_t const *a_ptr_0 = a + (m_row + row) * a_stride_elements + k_offset + k_sub;
                        nk_f32_t const *a_ptr_1 = a + (m_row + tile_dim + row) * a_stride_elements + k_offset + k_sub;
                        svfloat64_t a_vec_0 = svcvt_f64_f32_x(ptrue_d, svld1_f32(ptrue_s, a_ptr_0));
                        svfloat64_t a_vec_1 = svcvt_f64_f32_x(ptrue_d, svld1_f32(ptrue_s, a_ptr_1));

                        nk_f32_t const *b_col = b_tile + (k_sub + row) * tile_dim;
                        svfloat64_t b_vec = svcvt_f64_f32_x(ptrue_d, svld1_f32(ptrue_s, b_col));

                        svmopa_za64_f64_m(0, ptrue_d, ptrue_d, a_vec_0, b_vec);
                        svmopa_za64_f64_m(2, ptrue_d, ptrue_d, a_vec_1, b_vec);
                    }
                }
            }

            for (nk_size_t row = 0; row < tile_dim; row++) {
                nk_f64_t za_row_0[8], za_row_2[8];
                svst1_hor_za64(0, row, ptrue_d, za_row_0);
                svst1_hor_za64(2, row, ptrue_d, za_row_2);

                nk_f32_t *c_row_0 = c + (m_row + row) * c_stride_elements + n_col;
                nk_f32_t *c_row_1 = c + (m_row + tile_dim + row) * c_stride_elements + n_col;
                for (nk_size_t col = 0; col < tile_dim; col++) {
                    c_row_0[col] = (nk_f32_t)za_row_0[col];
                    c_row_1[col] = (nk_f32_t)za_row_2[col];
                }
            }
        }
    }

    // Handle odd M-tile at end
    if (num_m_tiles % 2 != 0) {
        nk_size_t const m_tile = num_m_tiles - 1;
        nk_size_t const m_row = m_tile * tile_dim;

        // Process 1×2 tile blocks using tiles 0 and 1
        for (nk_size_t n_block = 0; n_block < num_n_blocks; n_block++) {
            nk_size_t const n_col = n_block * block_dim;
            nk_size_t const n_tile_0 = n_block * 2;
            nk_size_t const n_tile_1 = n_block * 2 + 1;

            svzero_za();

            for (nk_size_t k_tile = 0; k_tile < num_k_tiles; k_tile++) {
                nk_size_t const k_offset = k_tile * k_tile_size;
                nk_size_t const b_tile_idx_0 = n_tile_0 * num_k_tiles + k_tile;
                nk_size_t const b_tile_idx_1 = n_tile_1 * num_k_tiles + k_tile;
                nk_f32_t const *b_tile_0 = b_tiles + b_tile_idx_0 * tile_elements;
                nk_f32_t const *b_tile_1 = b_tiles + b_tile_idx_1 * tile_elements;

                for (nk_size_t k_sub = 0; k_sub < k_tile_size; k_sub += tile_dim) {
                    for (nk_size_t row = 0; row < tile_dim; row++) {
                        nk_f32_t const *a_ptr = a + (m_row + row) * a_stride_elements + k_offset + k_sub;
                        svfloat64_t a_vec = svcvt_f64_f32_x(ptrue_d, svld1_f32(ptrue_s, a_ptr));

                        nk_f32_t const *b_col_0 = b_tile_0 + (k_sub + row) * tile_dim;
                        nk_f32_t const *b_col_1 = b_tile_1 + (k_sub + row) * tile_dim;
                        svfloat64_t b_vec_0 = svcvt_f64_f32_x(ptrue_d, svld1_f32(ptrue_s, b_col_0));
                        svfloat64_t b_vec_1 = svcvt_f64_f32_x(ptrue_d, svld1_f32(ptrue_s, b_col_1));

                        svmopa_za64_f64_m(0, ptrue_d, ptrue_d, a_vec, b_vec_0);
                        svmopa_za64_f64_m(1, ptrue_d, ptrue_d, a_vec, b_vec_1);
                    }
                }
            }

            for (nk_size_t row = 0; row < tile_dim; row++) {
                nk_f64_t za_row_0[8], za_row_1[8];
                svst1_hor_za64(0, row, ptrue_d, za_row_0);
                svst1_hor_za64(1, row, ptrue_d, za_row_1);

                nk_f32_t *c_row = c + (m_row + row) * c_stride_elements + n_col;
                for (nk_size_t col = 0; col < tile_dim; col++) {
                    c_row[col] = (nk_f32_t)za_row_0[col];
                    c_row[tile_dim + col] = (nk_f32_t)za_row_1[col];
                }
            }
        }

        // Handle final single tile (odd M × odd N)
        if (num_n_tiles % 2 != 0) {
            nk_size_t const n_tile = num_n_tiles - 1;
            nk_size_t const n_col = n_tile * tile_dim;

            svzero_za();

            for (nk_size_t k_tile = 0; k_tile < num_k_tiles; k_tile++) {
                nk_size_t const k_offset = k_tile * k_tile_size;
                nk_size_t const b_tile_idx = n_tile * num_k_tiles + k_tile;
                nk_f32_t const *b_tile = b_tiles + b_tile_idx * tile_elements;

                for (nk_size_t k_sub = 0; k_sub < k_tile_size; k_sub += tile_dim) {
                    for (nk_size_t row = 0; row < tile_dim; row++) {
                        nk_f32_t const *a_ptr = a + (m_row + row) * a_stride_elements + k_offset + k_sub;
                        svfloat64_t a_vec = svcvt_f64_f32_x(ptrue_d, svld1_f32(ptrue_s, a_ptr));

                        nk_f32_t const *b_col = b_tile + (k_sub + row) * tile_dim;
                        svfloat64_t b_vec = svcvt_f64_f32_x(ptrue_d, svld1_f32(ptrue_s, b_col));

                        svmopa_za64_f64_m(0, ptrue_d, ptrue_d, a_vec, b_vec);
                    }
                }
            }

            for (nk_size_t row = 0; row < tile_dim; row++) {
                nk_f64_t za_row[8];
                svst1_hor_za64(0, row, ptrue_d, za_row);

                nk_f32_t *c_row = c + (m_row + row) * c_stride_elements + n_col;
                for (nk_size_t col = 0; col < tile_dim; col++) { c_row[col] = (nk_f32_t)za_row[col]; }
            }
        }
    }
}

/*  F32 → F32 GEMM with F64 accumulation public interface.
 *  High-precision matrix multiplication for scientific computing applications.
 *
 *  @param a         Input matrix A (M × K), row-major
 *  @param b_packed  Pre-packed B matrix from nk_dots_f32f32f32_smef64_pack
 *  @param c         Output matrix C (M × N), row-major
 *  @param m         Number of rows in A and C
 *  @param n         Number of columns in C (rows in original B)
 *  @param k         Shared dimension (columns in A, columns in original B)
 *  @param a_stride  Byte stride between rows of A
 *  @param c_stride  Byte stride between rows of C
 */
NK_PUBLIC void nk_dots_f32f32f32_smef64(                  //
    nk_f32_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t m, nk_size_t n, nk_size_t k,                //
    nk_size_t a_stride, nk_size_t c_stride) {

    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_size_t const num_n_tiles = header->full_n_tiles;
    nk_size_t const n_edge_rows = header->n_edge_rows;

    nk_size_t const a_stride_elements = a_stride / sizeof(nk_f32_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);

    // ZA64 tile dimension (8 for 512-bit SVL)
    nk_size_t const tile_dim = 8;
    nk_size_t const num_m_tiles = m / tile_dim;
    nk_size_t const full_n_cols = num_n_tiles * tile_dim;

    // SME kernel for full tiles
    if (num_m_tiles > 0 && num_n_tiles > 0) {
        nk_dots_f32f32f32_smef64_kernel_(a, b_packed, c, num_m_tiles * tile_dim, full_n_cols, k, a_stride_elements,
                                         c_stride_elements);
    }

    // Scalar fallback for N-edge (columns beyond full N-tiles)
    if (n_edge_rows > 0) {
        nk_f32_t const *n_edge_ptr = (nk_f32_t const *)((char const *)b_packed + header->n_edge_offset);
        for (nk_size_t i = 0; i < m; i++) {
            nk_f32_t const *a_row = a + i * a_stride_elements;
            for (nk_size_t j = 0; j < n_edge_rows; j++) {
                nk_f64_t acc = 0.0;
                for (nk_size_t kk = 0; kk < k; kk++) { acc += (nk_f64_t)a_row[kk] * (nk_f64_t)n_edge_ptr[j * k + kk]; }
                c[i * c_stride_elements + full_n_cols + j] = (nk_f32_t)acc;
            }
        }
    }

    // NEON fallback for M-edge (rows beyond full M-tiles)
    if (m > num_m_tiles * tile_dim && num_n_tiles > 0) {
        nk_size_t const m_remainder_start = num_m_tiles * tile_dim;
        nk_size_t const m_remainder_count = m - m_remainder_start;

        nk_f32_t const *b_tiles = (nk_f32_t const *)((char const *)b_packed + sizeof(nk_dots_sme_packed_header_t));
        nk_size_t const num_k_tiles = header->full_k_tiles;
        nk_size_t const k_tile_size = 16; // SVL/32 for F32

        for (nk_size_t i = 0; i < m_remainder_count; i++) {
            nk_f32_t const *a_row = a + (m_remainder_start + i) * a_stride_elements;
            for (nk_size_t n_tile = 0; n_tile < num_n_tiles; n_tile++) {
                nk_f64_t acc[8] = {0}; // F64 accumulation
                for (nk_size_t k_tile = 0; k_tile < num_k_tiles; k_tile++) {
                    nk_size_t const k_offset = k_tile * k_tile_size;
                    nk_size_t const b_tile_idx = n_tile * num_k_tiles + k_tile;
                    nk_f32_t const *b_tile = b_tiles + b_tile_idx * tile_dim * k_tile_size;

                    for (nk_size_t j = 0; j < tile_dim; j++) {
                        for (nk_size_t kk = 0; kk < k_tile_size && k_offset + kk < k; kk++) {
                            acc[j] += (nk_f64_t)a_row[k_offset + kk] * (nk_f64_t)b_tile[kk * tile_dim + j];
                        }
                    }
                }
                for (nk_size_t j = 0; j < tile_dim; j++) {
                    c[(m_remainder_start + i) * c_stride_elements + n_tile * tile_dim + j] = (nk_f32_t)acc[j];
                }
            }
        }
    }

    // M-edge × N-edge corner
    if (m > num_m_tiles * tile_dim && n_edge_rows > 0) {
        nk_size_t const m_remainder_start = num_m_tiles * tile_dim;
        nk_size_t const m_remainder_count = m - m_remainder_start;
        nk_f32_t const *n_edge_ptr = (nk_f32_t const *)((char const *)b_packed + header->n_edge_offset);

        for (nk_size_t i = 0; i < m_remainder_count; i++) {
            nk_f32_t const *a_row = a + (m_remainder_start + i) * a_stride_elements;
            for (nk_size_t j = 0; j < n_edge_rows; j++) {
                nk_f64_t acc = 0.0;
                for (nk_size_t kk = 0; kk < k; kk++) { acc += (nk_f64_t)a_row[kk] * (nk_f64_t)n_edge_ptr[j * k + kk]; }
                c[(m_remainder_start + i) * c_stride_elements + full_n_cols + j] = (nk_f32_t)acc;
            }
        }
    }
}

/*  ============================================================================
 *  I8×I8 → I32 GEMM using SME outer products.
 *
 *  Uses svmopa_za32_s8_m for signed 8-bit integer outer product accumulate.
 *  This is available on Apple M4 (SME_I8I32 = 1).
 *
 *  Tile dimensions for I8 → I32 (512-bit SVL):
 *    - Input vectors: 64 I8 elements (SVL/8 = 64)
 *    - Output tile: 16×16 I32 elements (ZA32)
 *    - Each output I32 is a dot product of 4 I8 pairs
 *
 *  Expected performance: ~2 TOPS (4x F16 due to 4:1 element packing)
 *  ============================================================================
 */

/*  I8 packed buffer size calculation.
 *
 *  For I8 → I32 outer product:
 *    - Each I8 vector has SVL/8 elements (64 for 512-bit SVL)
 *    - Each ZA32 tile is SVL/32 × SVL/32 (16×16 for 512-bit SVL)
 *    - SMOPA computes: for each 4 I8 pairs, produce 1 I32 output
 *    - We tile N in increments of 16 rows (output tile dimension)
 *    - We tile K in increments of 64 columns (input vector width)
 */
NK_PUBLIC nk_size_t nk_dots_i8i8i32_smei32i32_packed_size(nk_size_t n, nk_size_t k) {
    nk_size_t const svl_bytes = 64;                            // Assume 512-bit SVL (Apple M4)
    nk_size_t const tile_rows = svl_bytes / sizeof(nk_i32_t);  // 16 rows per ZA32 tile
    nk_size_t const tile_k_cols = svl_bytes / sizeof(nk_i8_t); // 64 I8 K elements per tile

    nk_size_t const full_n_tiles = n / tile_rows;
    nk_size_t const tiles_along_k = (k + tile_k_cols - 1) / tile_k_cols;
    nk_size_t const n_edge_rows = n - full_n_tiles * tile_rows;

    // Header (64 bytes aligned)
    nk_size_t size = sizeof(nk_dots_sme_packed_header_t);

    // All tiles for full N rows (column-major for outer product access)
    // Each tile stores tile_rows × tile_k_cols I8 elements
    size += full_n_tiles * tiles_along_k * tile_rows * tile_k_cols * sizeof(nk_i8_t);

    // N edge: remaining rows for ALL K columns, stored row-major
    if (n_edge_rows > 0) size += n_edge_rows * k * sizeof(nk_i8_t);

    return size;
}

/*  Pack I8 B matrix for SME outer product access.
 */
NK_PUBLIC void nk_dots_i8i8i32_smei32i32_pack(  //
    nk_i8_t const *b, nk_size_t n, nk_size_t k, //
    nk_size_t b_stride, void *b_packed) {

    nk_size_t const svl_bytes = 64;
    nk_size_t const tile_rows = svl_bytes / sizeof(nk_i32_t);  // 16
    nk_size_t const tile_k_cols = svl_bytes / sizeof(nk_i8_t); // 64
    nk_size_t const tile_elements = tile_rows * tile_k_cols;   // 1024
    nk_size_t const b_stride_elements = b_stride / sizeof(nk_i8_t);

    nk_size_t const num_n_tiles = n / tile_rows;
    nk_size_t const num_k_tiles = (k + tile_k_cols - 1) / tile_k_cols;
    nk_size_t const n_remainder_rows = n - num_n_tiles * tile_rows;
    nk_size_t const total_tiles = num_n_tiles * num_k_tiles;

    // Write header
    nk_dots_sme_packed_header_t *header = (nk_dots_sme_packed_header_t *)b_packed;
    header->full_n_tiles = (nk_u32_t)num_n_tiles;
    header->full_k_tiles = (nk_u32_t)num_k_tiles;
    header->n_edge_rows = (nk_u32_t)n_remainder_rows;
    header->svl_bytes = (nk_u32_t)svl_bytes;

    nk_size_t const tiles_offset = sizeof(nk_dots_sme_packed_header_t);
    nk_size_t const n_edge_offset = tiles_offset + total_tiles * tile_elements * sizeof(nk_i8_t);
    header->n_edge_offset = (nk_u32_t)n_edge_offset;

    nk_i8_t *tiles_ptr = (nk_i8_t *)((char *)b_packed + tiles_offset);
    nk_i8_t *n_edge_ptr = (nk_i8_t *)((char *)b_packed + n_edge_offset);

    // Zero-initialize tiles
    for (nk_size_t i = 0; i < total_tiles * tile_elements; i++) { tiles_ptr[i] = 0; }

    // Pack tiles: column-major within each tile for efficient SVE loads
    for (nk_size_t n_tile = 0; n_tile < num_n_tiles; n_tile++) {
        for (nk_size_t k_tile = 0; k_tile < num_k_tiles; k_tile++) {
            nk_size_t const tile_index = n_tile * num_k_tiles + k_tile;
            nk_i8_t *tile_output = tiles_ptr + tile_index * tile_elements;

            nk_size_t const src_row_start = n_tile * tile_rows;
            nk_size_t const src_col_start = k_tile * tile_k_cols;
            nk_size_t const cols_to_pack = (src_col_start + tile_k_cols <= k) ? tile_k_cols : (k - src_col_start);

            // Column-major packing: tile_output[col * tile_rows + row]
            for (nk_size_t row = 0; row < tile_rows; row++) {
                for (nk_size_t col = 0; col < cols_to_pack; col++) {
                    nk_size_t const src_idx = (src_row_start + row) * b_stride_elements + src_col_start + col;
                    nk_size_t const dst_idx = col * tile_rows + row;
                    tile_output[dst_idx] = b[src_idx];
                }
            }
        }
    }

    // Pack N-remainder in row-major for NEON fallback
    if (n_remainder_rows > 0) {
        nk_size_t const remainder_start_row = num_n_tiles * tile_rows;
        for (nk_size_t row = 0; row < n_remainder_rows; row++) {
            for (nk_size_t col = 0; col < k; col++) {
                n_edge_ptr[row * k + col] = b[(remainder_start_row + row) * b_stride_elements + col];
            }
        }
    }
}

/*  I8×I8 → I32 GEMM core kernel using SME outer products.
 *
 *  Uses svmopa_za32_s8_m for signed I8×I8 → I32 outer product accumulate.
 *  Each SMOPA instruction processes:
 *    - Two 64-element I8 vectors (A row slice, B column slice)
 *    - Produces 16×16 I32 partial products accumulated into ZA32 tile
 *    - 4 I8 pairs contribute to each I32 output element
 */
__arm_locally_streaming __arm_new("za") static void nk_dots_i8i8i32_smei32i32_kernel_( //
    nk_i8_t const *a, void const *b_packed, nk_i32_t *c,                               //
    nk_size_t m, nk_size_t n, nk_size_t k,                                             //
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) {

    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_size_t const num_n_tiles = header->full_n_tiles;
    nk_size_t const num_k_tiles = header->full_k_tiles;

    nk_size_t const tile_dim = NK_SME_CNTW;    // 16 for 512-bit SVL (I32 output dimension)
    nk_size_t const k_tile_size = NK_SME_CNTB; // 64 for 512-bit SVL (I8 input dimension)
    nk_size_t const tile_elements = tile_dim * k_tile_size;

    nk_i8_t const *b_tiles = (nk_i8_t const *)((char const *)b_packed + sizeof(nk_dots_sme_packed_header_t));

    svbool_t const ptrue_b = svptrue_b8();  // 64 elements for I8 loads
    svbool_t const ptrue_s = svptrue_b32(); // 16 elements for I32 stores

    nk_size_t const num_m_tiles = m / tile_dim;
    nk_size_t const block_dim = tile_dim * 2; // 32 for 2×2 blocking (using all 4 ZA32 tiles)

    // Process 2×2 tile blocks (32×32 output) for maximum tile utilization
    nk_size_t const num_m_blocks = num_m_tiles / 2;
    nk_size_t const num_n_blocks = num_n_tiles / 2;

    for (nk_size_t m_block = 0; m_block < num_m_blocks; m_block++) {
        nk_size_t const m_row = m_block * block_dim;

        for (nk_size_t n_block = 0; n_block < num_n_blocks; n_block++) {
            nk_size_t const n_col = n_block * block_dim;
            nk_size_t const n_tile_0 = n_block * 2;     // First N-tile in block
            nk_size_t const n_tile_1 = n_block * 2 + 1; // Second N-tile in block

            // Zero all 4 ZA32 tiles
            svzero_za();

            // Accumulate over K dimension
            for (nk_size_t k_tile = 0; k_tile < num_k_tiles; k_tile++) {
                nk_size_t const k_offset = k_tile * k_tile_size;

                // Get B tile pointers for both N-halves
                nk_size_t const b_tile_idx_0 = n_tile_0 * num_k_tiles + k_tile;
                nk_size_t const b_tile_idx_1 = n_tile_1 * num_k_tiles + k_tile;
                nk_i8_t const *b_tile_0 = b_tiles + b_tile_idx_0 * tile_elements;
                nk_i8_t const *b_tile_1 = b_tiles + b_tile_idx_1 * tile_elements;

                // Process all rows with 4-tile outer products
                for (nk_size_t row = 0; row < tile_dim; row++) {
                    // Load A vectors for M rows 0-15 and 16-31
                    nk_i8_t const *a_ptr_0 = a + (m_row + row) * a_stride_elements + k_offset;
                    nk_i8_t const *a_ptr_1 = a + (m_row + tile_dim + row) * a_stride_elements + k_offset;
                    svint8_t a_vec_0 = svld1_s8(ptrue_b, a_ptr_0);
                    svint8_t a_vec_1 = svld1_s8(ptrue_b, a_ptr_1);

                    // Load B vectors for N columns 0-15 and 16-31
                    svint8_t b_vec_0 = svld1_s8(ptrue_b, b_tile_0 + row * k_tile_size);
                    svint8_t b_vec_1 = svld1_s8(ptrue_b, b_tile_1 + row * k_tile_size);

                    // 4 outer products for 2×2 tile arrangement
                    svmopa_za32_s8_m(0, ptrue_s, ptrue_s, a_vec_0, b_vec_0); // C[0:16, 0:16]
                    svmopa_za32_s8_m(1, ptrue_s, ptrue_s, a_vec_0, b_vec_1); // C[0:16, 16:32]
                    svmopa_za32_s8_m(2, ptrue_s, ptrue_s, a_vec_1, b_vec_0); // C[16:32, 0:16]
                    svmopa_za32_s8_m(3, ptrue_s, ptrue_s, a_vec_1, b_vec_1); // C[16:32, 16:32]
                }
            }

            // Store all 4 tiles to C
            for (nk_size_t row = 0; row < tile_dim; row++) {
                // Tile 0: C[m:m+16, n:n+16]
                svst1_hor_za32(0, row, ptrue_s, c + (m_row + row) * c_stride_elements + n_col);
                // Tile 1: C[m:m+16, n+16:n+32]
                svst1_hor_za32(1, row, ptrue_s, c + (m_row + row) * c_stride_elements + n_col + tile_dim);
                // Tile 2: C[m+16:m+32, n:n+16]
                svst1_hor_za32(2, row, ptrue_s, c + (m_row + tile_dim + row) * c_stride_elements + n_col);
                // Tile 3: C[m+16:m+32, n+16:n+32]
                svst1_hor_za32(3, row, ptrue_s, c + (m_row + tile_dim + row) * c_stride_elements + n_col + tile_dim);
            }
        }

        // Handle odd N-tile at end of row (if num_n_tiles is odd)
        if (num_n_tiles % 2 != 0) {
            nk_size_t const n_tile = num_n_tiles - 1;
            nk_size_t const n_col = n_tile * tile_dim;

            // Process with 2 tiles (0 and 2) for the two M-halves
            svzero_za();

            for (nk_size_t k_tile = 0; k_tile < num_k_tiles; k_tile++) {
                nk_size_t const k_offset = k_tile * k_tile_size;
                nk_size_t const b_tile_idx = n_tile * num_k_tiles + k_tile;
                nk_i8_t const *b_tile = b_tiles + b_tile_idx * tile_elements;

                for (nk_size_t row = 0; row < tile_dim; row++) {
                    nk_i8_t const *a_ptr_0 = a + (m_row + row) * a_stride_elements + k_offset;
                    nk_i8_t const *a_ptr_1 = a + (m_row + tile_dim + row) * a_stride_elements + k_offset;
                    svint8_t a_vec_0 = svld1_s8(ptrue_b, a_ptr_0);
                    svint8_t a_vec_1 = svld1_s8(ptrue_b, a_ptr_1);
                    svint8_t b_vec = svld1_s8(ptrue_b, b_tile + row * k_tile_size);

                    svmopa_za32_s8_m(0, ptrue_s, ptrue_s, a_vec_0, b_vec);
                    svmopa_za32_s8_m(2, ptrue_s, ptrue_s, a_vec_1, b_vec);
                }
            }

            for (nk_size_t row = 0; row < tile_dim; row++) {
                svst1_hor_za32(0, row, ptrue_s, c + (m_row + row) * c_stride_elements + n_col);
                svst1_hor_za32(2, row, ptrue_s, c + (m_row + tile_dim + row) * c_stride_elements + n_col);
            }
        }
    }

    // Handle odd M-tile at end (if num_m_tiles is odd)
    if (num_m_tiles % 2 != 0) {
        nk_size_t const m_tile = num_m_tiles - 1;
        nk_size_t const m_row = m_tile * tile_dim;

        // Process 1×2 tile blocks using tiles 0 and 1
        for (nk_size_t n_block = 0; n_block < num_n_blocks; n_block++) {
            nk_size_t const n_col = n_block * block_dim;
            nk_size_t const n_tile_0 = n_block * 2;
            nk_size_t const n_tile_1 = n_block * 2 + 1;

            svzero_za();

            for (nk_size_t k_tile = 0; k_tile < num_k_tiles; k_tile++) {
                nk_size_t const k_offset = k_tile * k_tile_size;
                nk_size_t const b_tile_idx_0 = n_tile_0 * num_k_tiles + k_tile;
                nk_size_t const b_tile_idx_1 = n_tile_1 * num_k_tiles + k_tile;
                nk_i8_t const *b_tile_0 = b_tiles + b_tile_idx_0 * tile_elements;
                nk_i8_t const *b_tile_1 = b_tiles + b_tile_idx_1 * tile_elements;

                for (nk_size_t row = 0; row < tile_dim; row++) {
                    nk_i8_t const *a_ptr = a + (m_row + row) * a_stride_elements + k_offset;
                    svint8_t a_vec = svld1_s8(ptrue_b, a_ptr);
                    svint8_t b_vec_0 = svld1_s8(ptrue_b, b_tile_0 + row * k_tile_size);
                    svint8_t b_vec_1 = svld1_s8(ptrue_b, b_tile_1 + row * k_tile_size);

                    svmopa_za32_s8_m(0, ptrue_s, ptrue_s, a_vec, b_vec_0);
                    svmopa_za32_s8_m(1, ptrue_s, ptrue_s, a_vec, b_vec_1);
                }
            }

            for (nk_size_t row = 0; row < tile_dim; row++) {
                svst1_hor_za32(0, row, ptrue_s, c + (m_row + row) * c_stride_elements + n_col);
                svst1_hor_za32(1, row, ptrue_s, c + (m_row + row) * c_stride_elements + n_col + tile_dim);
            }
        }

        // Handle final single tile (odd M × odd N)
        if (num_n_tiles % 2 != 0) {
            nk_size_t const n_tile = num_n_tiles - 1;
            nk_size_t const n_col = n_tile * tile_dim;

            svzero_za();

            for (nk_size_t k_tile = 0; k_tile < num_k_tiles; k_tile++) {
                nk_size_t const k_offset = k_tile * k_tile_size;
                nk_size_t const b_tile_idx = n_tile * num_k_tiles + k_tile;
                nk_i8_t const *b_tile = b_tiles + b_tile_idx * tile_elements;

                for (nk_size_t row = 0; row < tile_dim; row++) {
                    nk_i8_t const *a_ptr = a + (m_row + row) * a_stride_elements + k_offset;
                    svint8_t a_vec = svld1_s8(ptrue_b, a_ptr);
                    svint8_t b_vec = svld1_s8(ptrue_b, b_tile + row * k_tile_size);

                    svmopa_za32_s8_m(0, ptrue_s, ptrue_s, a_vec, b_vec);
                }
            }

            for (nk_size_t row = 0; row < tile_dim; row++) {
                svst1_hor_za32(0, row, ptrue_s, c + (m_row + row) * c_stride_elements + n_col);
            }
        }
    }
}

/*  I8×I8 → I32 GEMM public interface.
 *
 *  @param a         Input matrix A (M × K), row-major, I8
 *  @param b_packed  Pre-packed B matrix from nk_dots_i8i8i32_smei32i32_pack
 *  @param c         Output matrix C (M × N), row-major, I32
 *  @param m         Number of rows in A and C
 *  @param n         Number of columns in C (rows in original B)
 *  @param k         Shared dimension (columns in A, columns in original B)
 *  @param a_stride  Byte stride between rows of A
 *  @param c_stride  Byte stride between rows of C
 */
NK_PUBLIC void nk_dots_i8i8i32_smei32(                   //
    nk_i8_t const *a, void const *b_packed, nk_i32_t *c, //
    nk_size_t m, nk_size_t n, nk_size_t k,               //
    nk_size_t a_stride, nk_size_t c_stride) {

    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_size_t const num_n_tiles = header->full_n_tiles;
    nk_size_t const n_edge_rows = header->n_edge_rows;

    nk_size_t const a_stride_elements = a_stride / sizeof(nk_i8_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_i32_t);

    // ZA32 tile dimension (16 for 512-bit SVL)
    nk_size_t const tile_dim = 16;
    nk_size_t const num_m_tiles = m / tile_dim;
    nk_size_t const full_n_cols = num_n_tiles * tile_dim;

    // SME kernel for full tiles
    if (num_m_tiles > 0 && num_n_tiles > 0) {
        nk_dots_i8i8i32_smei32i32_kernel_(a, b_packed, c, num_m_tiles * tile_dim, full_n_cols, k, a_stride_elements,
                                          c_stride_elements);
    }

    // Scalar fallback for N-edge (columns beyond full N-tiles)
    if (n_edge_rows > 0) {
        nk_i8_t const *n_edge_ptr = (nk_i8_t const *)((char const *)b_packed + header->n_edge_offset);
        for (nk_size_t i = 0; i < m; i++) {
            nk_i8_t const *a_row = a + i * a_stride_elements;
            for (nk_size_t j = 0; j < n_edge_rows; j++) {
                nk_i32_t acc = 0;
                for (nk_size_t kk = 0; kk < k; kk++) { acc += (nk_i32_t)a_row[kk] * (nk_i32_t)n_edge_ptr[j * k + kk]; }
                c[i * c_stride_elements + full_n_cols + j] = acc;
            }
        }
    }

    // NEON fallback for M-edge (rows beyond full M-tiles)
    if (m > num_m_tiles * tile_dim && num_n_tiles > 0) {
        nk_size_t const m_remainder_start = num_m_tiles * tile_dim;
        nk_size_t const m_remainder_count = m - m_remainder_start;

        nk_i8_t const *b_tiles = (nk_i8_t const *)((char const *)b_packed + sizeof(nk_dots_sme_packed_header_t));
        nk_size_t const num_k_tiles = header->full_k_tiles;
        nk_size_t const k_tile_size = 64; // SVL/8 for I8

        for (nk_size_t i = 0; i < m_remainder_count; i++) {
            nk_i8_t const *a_row = a + (m_remainder_start + i) * a_stride_elements;
            for (nk_size_t n_tile = 0; n_tile < num_n_tiles; n_tile++) {
                nk_i32_t acc[16] = {0};
                for (nk_size_t k_tile = 0; k_tile < num_k_tiles; k_tile++) {
                    nk_size_t const k_offset = k_tile * k_tile_size;
                    nk_size_t const b_tile_idx = n_tile * num_k_tiles + k_tile;
                    nk_i8_t const *b_tile = b_tiles + b_tile_idx * tile_dim * k_tile_size;

                    for (nk_size_t j = 0; j < tile_dim; j++) {
                        for (nk_size_t kk = 0; kk < k_tile_size && k_offset + kk < k; kk++) {
                            acc[j] += (nk_i32_t)a_row[k_offset + kk] * (nk_i32_t)b_tile[kk * tile_dim + j];
                        }
                    }
                }
                for (nk_size_t j = 0; j < tile_dim; j++) {
                    c[(m_remainder_start + i) * c_stride_elements + n_tile * tile_dim + j] = acc[j];
                }
            }
        }
    }

    // M-edge × N-edge corner (scalar)
    if (m > num_m_tiles * tile_dim && n_edge_rows > 0) {
        nk_size_t const m_remainder_start = num_m_tiles * tile_dim;
        nk_size_t const m_remainder_count = m - m_remainder_start;
        nk_i8_t const *n_edge_ptr = (nk_i8_t const *)((char const *)b_packed + header->n_edge_offset);

        for (nk_size_t i = 0; i < m_remainder_count; i++) {
            nk_i8_t const *a_row = a + (m_remainder_start + i) * a_stride_elements;
            for (nk_size_t j = 0; j < n_edge_rows; j++) {
                nk_i32_t acc = 0;
                for (nk_size_t kk = 0; kk < k; kk++) { acc += (nk_i32_t)a_row[kk] * (nk_i32_t)n_edge_ptr[j * k + kk]; }
                c[(m_remainder_start + i) * c_stride_elements + full_n_cols + j] = acc;
            }
        }
    }
}

/*  ============================================================================
 *  SSVE (Streaming SVE) FP8 to F16 Conversion Functions
 *
 *  These functions convert E4M3/E5M2 to F16 using arithmetic operations that
 *  work entirely in streaming SVE mode. This enables fused GEMM kernels that
 *  stay in streaming mode the entire time, avoiding SMSTART/SMSTOP overhead.
 *
 *  Key insight: We can't use LUT lookup in streaming mode efficiently, so we
 *  use arithmetic conversion with FP16 multiply for subnormal handling.
 *
 *  E4M3 format: S EEEE MMM (1+4+3 bits, bias=7, no infinity, 0x7F=NaN)
 *  E5M2 format: S EEEEE MM (1+5+2 bits, bias=15, has infinity and NaN)
 *  F16 format:  S EEEEE MMMMMMMMMM (1+5+10 bits, bias=15)
 *  ============================================================================
 */

/*  Convert 32 E4M3 values to 32 F16 values using SSVE arithmetic.
 *
 *  Algorithm:
 *    Normal (exp != 0): F16 = sign | ((magnitude << 7) + 0x2000)
 *    Subnormal (exp == 0): F16 = sign | (mant × (1/512))
 *    NaN (mag == 0x7F): F16 = sign | 0x7E00
 *
 *  IMPORTANT: Caller must be in streaming mode (smstart sm) before calling.
 *  This function does NOT enter/exit streaming mode itself.
 *
 *  @param pg16   Predicate for 16-bit elements (use svptrue_b16())
 *  @param src    Pointer to 32 E4M3 bytes (must be 64-byte aligned)
 *  @param dst    Pointer to 32 F16 values (must be 64-byte aligned)
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

/*  Convert 32 E5M2 values to 32 F16 values using SSVE arithmetic.
 *
 *  Algorithm:
 *    Normal (exp != 0, exp != 31): F16 = sign | (magnitude << 8)
 *    Subnormal (exp == 0): F16 = sign | (mant × (1/65536))
 *    Infinity (exp == 31, mant == 0): F16 = sign | 0x7C00
 *    NaN (exp == 31, mant != 0): F16 = sign | 0x7E00
 *
 *  IMPORTANT: Caller must be in streaming mode (smstart sm) before calling.
 *
 *  @param pg16   Predicate for 16-bit elements (use svptrue_b16())
 *  @param src    Pointer to 32 E5M2 bytes (must be 64-byte aligned)
 *  @param dst    Pointer to 32 F16 values (must be 64-byte aligned)
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
    nk_f16_t scale_f16;
    __builtin_memcpy(&scale_f16, &scale_bits, sizeof(scale_f16));
    svfloat16_t scale = svdup_n_f16(scale_f16);
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

/*  Inline E4M3 → F16 conversion returning svfloat16_t for direct use in GEMM.
 *  This avoids memory round-trip when used inside a streaming kernel.
 *
 *  @param pg16   Predicate for 16-bit elements (use svptrue_b16())
 *  @param bytes  Pre-loaded 64 bytes (svuint8_t from svld1_u8)
 *  @return       32 F16 values as svfloat16_t (from lower 32 bytes)
 */
NK_INTERNAL svfloat16_t nk_e4m3x32_to_f16_vec_ssve_(svbool_t pg16, svuint8_t bytes) {
    svuint16_t vals = svunpklo_u16(bytes);

    svuint16_t sign = svlsl_n_u16_x(pg16, svand_n_u16_x(pg16, vals, 0x80), 8);
    svuint16_t mag = svand_n_u16_x(pg16, vals, 0x7F);
    svuint16_t mant = svand_n_u16_x(pg16, vals, 0x07);

    // Normal path: F16 = sign | ((mag << 7) + 0x2000)
    svuint16_t normal = svadd_n_u16_x(pg16, svlsl_n_u16_x(pg16, mag, 7), 0x2000);
    normal = svorr_u16_x(pg16, normal, sign);

    // Subnormal path: mant × (1/512) where 1/512 = 0x1800 in F16
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

/*  Fused E4M3×E4M3 → F32 GEMM kernel using SSVE inline conversion.
 *
 *  This kernel stays entirely in streaming mode, converting E4M3 → F16 inline
 *  using arithmetic operations. Uses inline asm for SME operations to avoid
 *  __arm_locally_streaming codegen issues on Apple M4.
 *
 *  The approach:
 *  - Use inline asm for smstart/smstop and SME tile operations (fmopa, st1w)
 *  - Use regular SVE intrinsics for E4M3 → F16 conversion (work in streaming mode)
 *  - This avoids the rdsvl-before-streaming-mode bug in AppleClang
 */
__arm_locally_streaming __arm_new("za") __attribute__((noinline)) static void nk_dots_e4m3e4m3f32_sme_fused_kernel_( //
    nk_e4m3_t const *a, void const *b_packed, nk_f32_t *c,                                                           //
    nk_size_t m, nk_size_t n, nk_size_t k,                                                                           //
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) {

    // Initialize p0 as all-true predicate for fmopa/st1w operations
    __asm__ volatile("ptrue p0.s" ::: "memory");

    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_size_t const num_n_tiles = header->full_n_tiles;
    nk_size_t const num_k_tiles = header->full_k_tiles;

    nk_size_t const tile_dim = 16;    // SVL/32 for 512-bit SVL
    nk_size_t const k_tile_size = 32; // SVL/16 for 512-bit SVL
    nk_size_t const tile_elements = tile_dim * k_tile_size;

    nk_f16_t const *b_tiles = (nk_f16_t const *)((char const *)b_packed + sizeof(nk_dots_sme_packed_header_t));

    nk_size_t const num_m_tiles = m / tile_dim;
    nk_size_t const block_dim = tile_dim * 2;

    nk_size_t const num_m_blocks = num_m_tiles / 2;
    nk_size_t const num_n_blocks = num_n_tiles / 2;

    // Process 2×2 tile blocks (32×32 output)
    for (nk_size_t m_block = 0; m_block < num_m_blocks; m_block++) {
        nk_size_t const m_row = m_block * block_dim;

        for (nk_size_t n_block = 0; n_block < num_n_blocks; n_block++) {
            nk_size_t const n_col = n_block * block_dim;
            nk_size_t const n_tile_0 = n_block * 2;
            nk_size_t const n_tile_1 = n_block * 2 + 1;

            // Zero all ZA tiles
            __asm__ volatile("zero {za}" ::: "memory");

            for (nk_size_t k_tile = 0; k_tile < num_k_tiles; k_tile++) {
                nk_size_t const k_offset = k_tile * k_tile_size;

                nk_size_t const b_tile_idx_0 = n_tile_0 * num_k_tiles + k_tile;
                nk_size_t const b_tile_idx_1 = n_tile_1 * num_k_tiles + k_tile;
                nk_f16_t const *b_tile_0 = b_tiles + b_tile_idx_0 * tile_elements;
                nk_f16_t const *b_tile_1 = b_tiles + b_tile_idx_1 * tile_elements;

                svbool_t ptrue_h = svptrue_b16();

                svbool_t ptrue_b = svptrue_b8();

                // fmopa with F16 inputs does 2-way outer product:
                // ZA[i,j] += z1[2i]*z2[2j] + z1[2i+1]*z2[2j+1]
                // We iterate K in pairs and gather A values from 16 M rows
                for (nk_size_t k_pair = 0; k_pair < k_tile_size; k_pair += 2) {
                    // Build interleaved A vectors: [A[m0,k], A[m0,k+1], A[m1,k], A[m1,k+1], ...]
                    // Gather 32 E4M3 bytes: 16 M rows × 2 K positions
                    uint8_t a_bytes_buf_0[64] __attribute__((aligned(64)));
                    uint8_t a_bytes_buf_1[64] __attribute__((aligned(64)));

                    for (nk_size_t mi = 0; mi < tile_dim; mi++) {
                        // First M-block (rows m_row to m_row+15)
                        a_bytes_buf_0[mi * 2] = a[(m_row + mi) * a_stride_elements + k_offset + k_pair];
                        a_bytes_buf_0[mi * 2 + 1] = a[(m_row + mi) * a_stride_elements + k_offset + k_pair + 1];

                        // Second M-block (rows m_row+16 to m_row+31)
                        a_bytes_buf_1[mi * 2] = a[(m_row + tile_dim + mi) * a_stride_elements + k_offset + k_pair];
                        a_bytes_buf_1[mi * 2 + 1] =
                            a[(m_row + tile_dim + mi) * a_stride_elements + k_offset + k_pair + 1];
                    }

                    // Convert E4M3 to F16 using SSVE conversion
                    svuint8_t a_e4m3_0 = svld1_u8(ptrue_b, a_bytes_buf_0);
                    svuint8_t a_e4m3_1 = svld1_u8(ptrue_b, a_bytes_buf_1);
                    svfloat16_t a_vec_0 = nk_e4m3x32_to_f16_vec_ssve_(ptrue_h, a_e4m3_0);
                    svfloat16_t a_vec_1 = nk_e4m3x32_to_f16_vec_ssve_(ptrue_h, a_e4m3_1);

                    // Load B vectors (already interleaved in pack: [B[n0,k], B[n0,k+1], ...])
                    // k_pair/2 gives the K-pair index, each K-pair has 32 F16 values
                    svfloat16_t b_vec_0 = svld1_f16(ptrue_h, (float16_t const *)(b_tile_0 + (k_pair / 2) * 32));
                    svfloat16_t b_vec_1 = svld1_f16(ptrue_h, (float16_t const *)(b_tile_1 + (k_pair / 2) * 32));

                    // 4 outer products using inline asm
                    __asm__ volatile("fmopa za0.s, p0/m, p0/m, %[a].h, %[b].h"
                                     :
                                     : [a] "w"(a_vec_0), [b] "w"(b_vec_0)
                                     : "memory");
                    __asm__ volatile("fmopa za1.s, p0/m, p0/m, %[a].h, %[b].h"
                                     :
                                     : [a] "w"(a_vec_0), [b] "w"(b_vec_1)
                                     : "memory");
                    __asm__ volatile("fmopa za2.s, p0/m, p0/m, %[a].h, %[b].h"
                                     :
                                     : [a] "w"(a_vec_1), [b] "w"(b_vec_0)
                                     : "memory");
                    __asm__ volatile("fmopa za3.s, p0/m, p0/m, %[a].h, %[b].h"
                                     :
                                     : [a] "w"(a_vec_1), [b] "w"(b_vec_1)
                                     : "memory");
                }
            }

            // Store all 4 tiles using inline asm
            for (nk_size_t row = 0; row < tile_dim; row++) {
                nk_f32_t *c_ptr_0 = c + (m_row + row) * c_stride_elements + n_col;
                nk_f32_t *c_ptr_1 = c + (m_row + row) * c_stride_elements + n_col + tile_dim;
                nk_f32_t *c_ptr_2 = c + (m_row + tile_dim + row) * c_stride_elements + n_col;
                nk_f32_t *c_ptr_3 = c + (m_row + tile_dim + row) * c_stride_elements + n_col + tile_dim;

                // Use w12 as the tile slice index register
                register uint32_t row_idx __asm__("w12") = (uint32_t)row;
                __asm__ volatile("st1w {za0h.s[%w[idx], 0]}, p0, [%[ptr]]"
                                 :
                                 : [idx] "r"(row_idx), [ptr] "r"(c_ptr_0)
                                 : "memory");
                __asm__ volatile("st1w {za1h.s[%w[idx], 0]}, p0, [%[ptr]]"
                                 :
                                 : [idx] "r"(row_idx), [ptr] "r"(c_ptr_1)
                                 : "memory");
                __asm__ volatile("st1w {za2h.s[%w[idx], 0]}, p0, [%[ptr]]"
                                 :
                                 : [idx] "r"(row_idx), [ptr] "r"(c_ptr_2)
                                 : "memory");
                __asm__ volatile("st1w {za3h.s[%w[idx], 0]}, p0, [%[ptr]]"
                                 :
                                 : [idx] "r"(row_idx), [ptr] "r"(c_ptr_3)
                                 : "memory");
            }
        }

        // Handle odd N-tile at end
        if (num_n_tiles % 2 != 0) {
            nk_size_t const n_tile = num_n_tiles - 1;
            nk_size_t const n_col = n_tile * tile_dim;

            __asm__ volatile("zero {za}" ::: "memory");

            for (nk_size_t k_tile = 0; k_tile < num_k_tiles; k_tile++) {
                nk_size_t const k_offset = k_tile * k_tile_size;
                nk_size_t const b_tile_idx = n_tile * num_k_tiles + k_tile;
                nk_f16_t const *b_tile = b_tiles + b_tile_idx * tile_elements;

                svbool_t ptrue_h = svptrue_b16();
                svbool_t ptrue_b = svptrue_b8();

                for (nk_size_t k_pair = 0; k_pair < k_tile_size; k_pair += 2) {
                    uint8_t a_bytes_buf_0[64] __attribute__((aligned(64)));
                    uint8_t a_bytes_buf_1[64] __attribute__((aligned(64)));

                    for (nk_size_t mi = 0; mi < tile_dim; mi++) {
                        a_bytes_buf_0[mi * 2] = a[(m_row + mi) * a_stride_elements + k_offset + k_pair];
                        a_bytes_buf_0[mi * 2 + 1] = a[(m_row + mi) * a_stride_elements + k_offset + k_pair + 1];

                        a_bytes_buf_1[mi * 2] = a[(m_row + tile_dim + mi) * a_stride_elements + k_offset + k_pair];
                        a_bytes_buf_1[mi * 2 + 1] =
                            a[(m_row + tile_dim + mi) * a_stride_elements + k_offset + k_pair + 1];
                    }

                    svuint8_t a_e4m3_0 = svld1_u8(ptrue_b, a_bytes_buf_0);
                    svuint8_t a_e4m3_1 = svld1_u8(ptrue_b, a_bytes_buf_1);
                    svfloat16_t a_vec_0 = nk_e4m3x32_to_f16_vec_ssve_(ptrue_h, a_e4m3_0);
                    svfloat16_t a_vec_1 = nk_e4m3x32_to_f16_vec_ssve_(ptrue_h, a_e4m3_1);
                    svfloat16_t b_vec = svld1_f16(ptrue_h, (float16_t const *)(b_tile + (k_pair / 2) * 32));

                    __asm__ volatile("fmopa za0.s, p0/m, p0/m, %[a].h, %[b].h"
                                     :
                                     : [a] "w"(a_vec_0), [b] "w"(b_vec)
                                     : "memory");
                    __asm__ volatile("fmopa za2.s, p0/m, p0/m, %[a].h, %[b].h"
                                     :
                                     : [a] "w"(a_vec_1), [b] "w"(b_vec)
                                     : "memory");
                }
            }

            for (nk_size_t row = 0; row < tile_dim; row++) {
                nk_f32_t *c_ptr_0 = c + (m_row + row) * c_stride_elements + n_col;
                nk_f32_t *c_ptr_2 = c + (m_row + tile_dim + row) * c_stride_elements + n_col;
                register uint32_t row_idx __asm__("w12") = (uint32_t)row;
                __asm__ volatile("st1w {za0h.s[%w[idx], 0]}, p0, [%[ptr]]"
                                 :
                                 : [idx] "r"(row_idx), [ptr] "r"(c_ptr_0)
                                 : "memory");
                __asm__ volatile("st1w {za2h.s[%w[idx], 0]}, p0, [%[ptr]]"
                                 :
                                 : [idx] "r"(row_idx), [ptr] "r"(c_ptr_2)
                                 : "memory");
            }
        }
    }

    // Handle odd M-tile at end
    if (num_m_tiles % 2 != 0) {
        nk_size_t const m_tile = num_m_tiles - 1;
        nk_size_t const m_row = m_tile * tile_dim;

        for (nk_size_t n_block = 0; n_block < num_n_blocks; n_block++) {
            nk_size_t const n_col = n_block * block_dim;
            nk_size_t const n_tile_0 = n_block * 2;
            nk_size_t const n_tile_1 = n_block * 2 + 1;

            __asm__ volatile("zero {za}" ::: "memory");

            for (nk_size_t k_tile = 0; k_tile < num_k_tiles; k_tile++) {
                nk_size_t const k_offset = k_tile * k_tile_size;
                nk_size_t const b_tile_idx_0 = n_tile_0 * num_k_tiles + k_tile;
                nk_size_t const b_tile_idx_1 = n_tile_1 * num_k_tiles + k_tile;
                nk_f16_t const *b_tile_0 = b_tiles + b_tile_idx_0 * tile_elements;
                nk_f16_t const *b_tile_1 = b_tiles + b_tile_idx_1 * tile_elements;

                svbool_t ptrue_h = svptrue_b16();
                svbool_t ptrue_b = svptrue_b8();

                for (nk_size_t k_pair = 0; k_pair < k_tile_size; k_pair += 2) {
                    uint8_t a_bytes_buf[64] __attribute__((aligned(64)));

                    for (nk_size_t mi = 0; mi < tile_dim; mi++) {
                        a_bytes_buf[mi * 2] = a[(m_row + mi) * a_stride_elements + k_offset + k_pair];
                        a_bytes_buf[mi * 2 + 1] = a[(m_row + mi) * a_stride_elements + k_offset + k_pair + 1];
                    }

                    svuint8_t a_e4m3 = svld1_u8(ptrue_b, a_bytes_buf);
                    svfloat16_t a_vec = nk_e4m3x32_to_f16_vec_ssve_(ptrue_h, a_e4m3);
                    svfloat16_t b_vec_0 = svld1_f16(ptrue_h, (float16_t const *)(b_tile_0 + (k_pair / 2) * 32));
                    svfloat16_t b_vec_1 = svld1_f16(ptrue_h, (float16_t const *)(b_tile_1 + (k_pair / 2) * 32));

                    __asm__ volatile("fmopa za0.s, p0/m, p0/m, %[a].h, %[b].h"
                                     :
                                     : [a] "w"(a_vec), [b] "w"(b_vec_0)
                                     : "memory");
                    __asm__ volatile("fmopa za1.s, p0/m, p0/m, %[a].h, %[b].h"
                                     :
                                     : [a] "w"(a_vec), [b] "w"(b_vec_1)
                                     : "memory");
                }
            }

            for (nk_size_t row = 0; row < tile_dim; row++) {
                nk_f32_t *c_ptr_0 = c + (m_row + row) * c_stride_elements + n_col;
                nk_f32_t *c_ptr_1 = c + (m_row + row) * c_stride_elements + n_col + tile_dim;
                register uint32_t row_idx __asm__("w12") = (uint32_t)row;
                __asm__ volatile("st1w {za0h.s[%w[idx], 0]}, p0, [%[ptr]]"
                                 :
                                 : [idx] "r"(row_idx), [ptr] "r"(c_ptr_0)
                                 : "memory");
                __asm__ volatile("st1w {za1h.s[%w[idx], 0]}, p0, [%[ptr]]"
                                 :
                                 : [idx] "r"(row_idx), [ptr] "r"(c_ptr_1)
                                 : "memory");
            }
        }

        // Handle final single tile (odd M × odd N)
        if (num_n_tiles % 2 != 0) {
            nk_size_t const n_tile = num_n_tiles - 1;
            nk_size_t const n_col = n_tile * tile_dim;

            __asm__ volatile("zero {za}" ::: "memory");

            for (nk_size_t k_tile = 0; k_tile < num_k_tiles; k_tile++) {
                nk_size_t const k_offset = k_tile * k_tile_size;
                nk_size_t const b_tile_idx = n_tile * num_k_tiles + k_tile;
                nk_f16_t const *b_tile = b_tiles + b_tile_idx * tile_elements;

                svbool_t ptrue_h = svptrue_b16();
                svbool_t ptrue_b = svptrue_b8();

                for (nk_size_t k_pair = 0; k_pair < k_tile_size; k_pair += 2) {
                    uint8_t a_bytes_buf[64] __attribute__((aligned(64)));

                    for (nk_size_t mi = 0; mi < tile_dim; mi++) {
                        a_bytes_buf[mi * 2] = a[(m_row + mi) * a_stride_elements + k_offset + k_pair];
                        a_bytes_buf[mi * 2 + 1] = a[(m_row + mi) * a_stride_elements + k_offset + k_pair + 1];
                    }

                    svuint8_t a_e4m3 = svld1_u8(ptrue_b, a_bytes_buf);
                    svfloat16_t a_vec = nk_e4m3x32_to_f16_vec_ssve_(ptrue_h, a_e4m3);
                    svfloat16_t b_vec = svld1_f16(ptrue_h, (float16_t const *)(b_tile + (k_pair / 2) * 32));

                    __asm__ volatile("fmopa za0.s, p0/m, p0/m, %[a].h, %[b].h"
                                     :
                                     : [a] "w"(a_vec), [b] "w"(b_vec)
                                     : "memory");
                }
            }

            for (nk_size_t row = 0; row < tile_dim; row++) {
                nk_f32_t *c_ptr = c + (m_row + row) * c_stride_elements + n_col;
                register uint32_t row_idx __asm__("w12") = (uint32_t)row;
                __asm__ volatile("st1w {za0h.s[%w[idx], 0]}, p0, [%[ptr]]"
                                 :
                                 : [idx] "r"(row_idx), [ptr] "r"(c_ptr)
                                 : "memory");
            }
        }
    }
}

/*  ============================================================================
 *  E4M3×E4M3 → F32 GEMM using SME F16 outer products via LUT conversion.
 *
 *  Since Apple M4 lacks native FP8 MOPA (requires SME2p1/FEAT_SME_F8F32),
 *  we convert E4M3 to F16 during packing using a precomputed 128-entry LUT.
 *
 *  E4M3 format: S EEEE MMM (1+4+3 bits, bias=7, range [-448, 448])
 *  F16 format:  S EEEEE MMMMMMMMMM (1+5+10 bits, bias=15)
 *  Conversion: F16_exp = E4M3_exp + 8, F16_mant = E4M3_mant << 7
 *
 *  LUT design (inspired by AVX-512 permutex2var approach):
 *  - 128 entries for positive E4M3 values (7-bit magnitude)
 *  - Sign bit handled separately via OR
 *  - Uses NEON vqtbl4q_u8 for vectorized 64-byte lookups
 *
 *  Expected performance: ~1.3-1.5 TOPS (F16 SME limited by pack overhead)
 *  ============================================================================
 */

// clang-format off
/*  E4M3 → F16 lookup table (128 entries, stored as separate low/high bytes for NEON TBL).
 *
 *  Table values computed as:
 *    - Index 0x00: zero → 0x0000
 *    - Index 0x01-0x07: subnormals (M×2⁻⁹) → F16 normals
 *    - Index 0x08-0x7E: normals → F16_exp = E4M3_exp + 8, F16_mant = E4M3_mant << 7
 *    - Index 0x7F: NaN → 0x7E00
 */
static uint8x16_t const nk_e4m3_to_f16_lut_lo_0_ = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                                                         0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80};
static uint8x16_t const nk_e4m3_to_f16_lut_lo_1_ = {0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80,
                                                         0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80};
static uint8x16_t const nk_e4m3_to_f16_lut_lo_2_ = {0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80,
                                                         0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80};
static uint8x16_t const nk_e4m3_to_f16_lut_lo_3_ = {0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80,
                                                         0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80};
static uint8x16_t const nk_e4m3_to_f16_lut_lo_4_ = {0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80,
                                                         0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80};
static uint8x16_t const nk_e4m3_to_f16_lut_lo_5_ = {0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80,
                                                         0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80};
static uint8x16_t const nk_e4m3_to_f16_lut_lo_6_ = {0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80,
                                                         0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80};
static uint8x16_t const nk_e4m3_to_f16_lut_lo_7_ = {0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80,
                                                         0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x00};

static uint8x16_t const nk_e4m3_to_f16_lut_hi_0_ = {0x00, 0x18, 0x1C, 0x1E, 0x20, 0x21, 0x22, 0x23,
                                                         0x24, 0x24, 0x25, 0x25, 0x26, 0x26, 0x27, 0x27};
static uint8x16_t const nk_e4m3_to_f16_lut_hi_1_ = {0x28, 0x28, 0x29, 0x29, 0x2A, 0x2A, 0x2B, 0x2B,
                                                         0x2C, 0x2C, 0x2D, 0x2D, 0x2E, 0x2E, 0x2F, 0x2F};
static uint8x16_t const nk_e4m3_to_f16_lut_hi_2_ = {0x30, 0x30, 0x31, 0x31, 0x32, 0x32, 0x33, 0x33,
                                                         0x34, 0x34, 0x35, 0x35, 0x36, 0x36, 0x37, 0x37};
static uint8x16_t const nk_e4m3_to_f16_lut_hi_3_ = {0x38, 0x38, 0x39, 0x39, 0x3A, 0x3A, 0x3B, 0x3B,
                                                         0x3C, 0x3C, 0x3D, 0x3D, 0x3E, 0x3E, 0x3F, 0x3F};
static uint8x16_t const nk_e4m3_to_f16_lut_hi_4_ = {0x40, 0x40, 0x41, 0x41, 0x42, 0x42, 0x43, 0x43,
                                                         0x44, 0x44, 0x45, 0x45, 0x46, 0x46, 0x47, 0x47};
static uint8x16_t const nk_e4m3_to_f16_lut_hi_5_ = {0x48, 0x48, 0x49, 0x49, 0x4A, 0x4A, 0x4B, 0x4B,
                                                         0x4C, 0x4C, 0x4D, 0x4D, 0x4E, 0x4E, 0x4F, 0x4F};
static uint8x16_t const nk_e4m3_to_f16_lut_hi_6_ = {0x50, 0x50, 0x51, 0x51, 0x52, 0x52, 0x53, 0x53,
                                                         0x54, 0x54, 0x55, 0x55, 0x56, 0x56, 0x57, 0x57};
static uint8x16_t const nk_e4m3_to_f16_lut_hi_7_ = {0x58, 0x58, 0x59, 0x59, 0x5A, 0x5A, 0x5B, 0x5B,
                                                         0x5C, 0x5C, 0x5D, 0x5D, 0x5E, 0x5E, 0x5F, 0x7E};
// clang-format on

/*  Precomputed uint16_t LUT for scalar E4M3 → F16 conversion.
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

/*  Scalar E4M3 → F16 conversion using LUT.
 */
NK_INTERNAL nk_f16_t nk_e4m3_to_f16_lut_(nk_e4m3_t src) {
    nk_u8_t idx = src & 0x7F;
    nk_u16_t result = nk_e4m3_to_f16_lut_u16_[idx];
    if (src & 0x80) result |= 0x8000; // Apply sign
    nk_f16_t f16_result;
    __builtin_memcpy(&f16_result, &result, sizeof(result));
    return f16_result;
}

NK_PUBLIC nk_size_t nk_dots_e4m3e4m3f32_packed_size_sme(nk_size_t n, nk_size_t k) {
    // Uses F16 format for packed data
    return nk_dots_f16f16f32_packed_size_sme(n, k);
}

NK_PUBLIC void nk_dots_e4m3e4m3f32_pack_sme(      //
    nk_e4m3_t const *b, nk_size_t n, nk_size_t k, //
    nk_size_t b_stride, void *b_packed) {

    nk_size_t const svl_bytes = 64;
    nk_size_t const tile_rows = svl_bytes / sizeof(nk_f32_t);   // 16
    nk_size_t const tile_k_cols = svl_bytes / sizeof(nk_f16_t); // 32
    nk_size_t const tile_elements = tile_rows * tile_k_cols;    // 512
    nk_size_t const b_stride_elements = b_stride / sizeof(nk_e4m3_t);

    nk_size_t const num_n_tiles = n / tile_rows;
    nk_size_t const num_k_tiles = (k + tile_k_cols - 1) / tile_k_cols;
    nk_size_t const n_remainder_rows = n - num_n_tiles * tile_rows;
    nk_size_t const total_tiles = num_n_tiles * num_k_tiles;

    nk_dots_sme_packed_header_t *header = (nk_dots_sme_packed_header_t *)b_packed;
    header->full_n_tiles = (nk_u32_t)num_n_tiles;
    header->full_k_tiles = (nk_u32_t)num_k_tiles;
    header->n_edge_rows = (nk_u32_t)n_remainder_rows;
    header->svl_bytes = (nk_u32_t)svl_bytes;

    nk_size_t const tiles_offset = sizeof(nk_dots_sme_packed_header_t);
    nk_size_t const n_edge_offset = tiles_offset + total_tiles * tile_elements * sizeof(nk_f16_t);
    header->n_edge_offset = (nk_u32_t)n_edge_offset;

    nk_f16_t *tiles_ptr = (nk_f16_t *)((char *)b_packed + tiles_offset);
    nk_f16_t *n_edge_ptr = (nk_f16_t *)((char *)b_packed + n_edge_offset);

    // Zero-initialize tiles
    for (nk_size_t i = 0; i < total_tiles * tile_elements; i++) tiles_ptr[i] = 0;

    // Pack tiles with E4M3 → F16 LUT conversion (interleaved K-pairs for fmopa 2-way)
    // fmopa za.s, p0/m, p0/m, z1.h, z2.h computes ZA[i,j] += z1[2i]*z2[2j] + z1[2i+1]*z2[2j+1]
    // So we need b_vec = [B[n0,k], B[n0,k+1], B[n1,k], B[n1,k+1], ..., B[n15,k], B[n15,k+1]]
    // Layout: dst_idx = (k/2) * 32 + n*2 + (k&1)
    for (nk_size_t n_tile = 0; n_tile < num_n_tiles; n_tile++) {
        for (nk_size_t k_tile = 0; k_tile < num_k_tiles; k_tile++) {
            nk_size_t const tile_index = n_tile * num_k_tiles + k_tile;
            nk_f16_t *tile_output = tiles_ptr + tile_index * tile_elements;

            nk_size_t const src_row_start = n_tile * tile_rows;
            nk_size_t const src_col_start = k_tile * tile_k_cols;
            nk_size_t const cols_to_pack = (src_col_start + tile_k_cols <= k) ? tile_k_cols : (k - src_col_start);

            for (nk_size_t row = 0; row < tile_rows; row++) {
                for (nk_size_t col = 0; col < cols_to_pack; col++) {
                    nk_size_t const src_idx = (src_row_start + row) * b_stride_elements + src_col_start + col;
                    // Interleaved layout for 2-way fmopa: K-pairs contiguous, N interleaved within pair
                    nk_size_t const dst_idx = (col / 2) * 32 + row * 2 + (col & 1);
                    tile_output[dst_idx] = nk_e4m3_to_f16_lut_(b[src_idx]);
                }
            }
        }
    }

    // Pack N-remainder with conversion
    if (n_remainder_rows > 0) {
        nk_size_t const remainder_start_row = num_n_tiles * tile_rows;
        for (nk_size_t row = 0; row < n_remainder_rows; row++) {
            for (nk_size_t col = 0; col < k; col++) {
                nk_e4m3_t e4m3_val = b[(remainder_start_row + row) * b_stride_elements + col];
                n_edge_ptr[row * k + col] = nk_e4m3_to_f16_lut_(e4m3_val);
            }
        }
    }
}

/*  E4M3×E4M3 → F32 GEMM: fused kernel with SSVE inline E4M3 → F16 conversion.
 *
 *  Uses a fully fused kernel that converts E4M3 → F16 inline in streaming mode,
 *  eliminating buffer allocation and SMSTART/SMSTOP overhead for the tile-aligned
 *  portion. Falls back to NEON conversion for edge cases.
 *
 *  @param a         Input matrix A (M × K), row-major, E4M3
 *  @param b_packed  Pre-packed B matrix from nk_dots_e4m3e4m3f32_pack_sme (contains F16)
 *  @param c         Output matrix C (M × N), row-major, F32
 */
NK_PUBLIC void nk_dots_e4m3e4m3f32_sme(                    //
    nk_e4m3_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t m, nk_size_t n, nk_size_t k,                 //
    nk_size_t a_stride, nk_size_t c_stride) {

    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_size_t const num_n_tiles = header->full_n_tiles;
    nk_size_t const n_edge_rows = header->n_edge_rows;

    nk_size_t const a_stride_elements = a_stride / sizeof(nk_e4m3_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);

    nk_size_t const tile_dim = 16;
    nk_size_t const num_m_tiles = m / tile_dim;
    nk_size_t const full_n_cols = num_n_tiles * tile_dim;

    // Fused SME kernel for full tiles - converts E4M3 → F16 inline in streaming mode
    if (num_m_tiles > 0 && num_n_tiles > 0) {
        nk_dots_e4m3e4m3f32_sme_fused_kernel_(a, b_packed, c, num_m_tiles * tile_dim, full_n_cols, k, a_stride_elements,
                                              c_stride_elements);
    }

    // Edge handling requires F16 conversion via NEON (outside streaming mode)
    nk_size_t const m_remainder = m - num_m_tiles * tile_dim;

    if (m_remainder == 0 && n_edge_rows == 0) return;

    // Allocate buffer for edge conversions
    nk_f16_t *a_f16_edge = NULL;
    if (n_edge_rows > 0 || m_remainder > 0) {
        nk_size_t edge_buffer_size = (m_remainder > 0 ? m_remainder : 1) * k;
        if (n_edge_rows > 0) edge_buffer_size = m * k; // Need full A for N-edge
        a_f16_edge = (nk_f16_t *)aligned_alloc(64, edge_buffer_size * sizeof(nk_f16_t));
        if (!a_f16_edge) return;
    }

    // N-edge: columns beyond full N-tiles (need all M rows of A converted)
    if (n_edge_rows > 0) {
        // Convert E4M3 → F16 (scalar)
        for (nk_size_t i = 0; i < m; i++) {
            nk_e4m3_t const *a_row = a + i * a_stride_elements;
            nk_f16_t *a_f16_row = a_f16_edge + i * k;
            for (nk_size_t j = 0; j < k; j++) { a_f16_row[j] = nk_e4m3_to_f16_lut_(a_row[j]); }
        }

        // Scalar F16 dot product for N-edge
        nk_f16_t const *n_edge_ptr = (nk_f16_t const *)((char const *)b_packed + header->n_edge_offset);
        for (nk_size_t i = 0; i < m; i++) {
            nk_f16_t const *a_row = a_f16_edge + i * k;
            for (nk_size_t j = 0; j < n_edge_rows; j++) {
                nk_f32_t acc = 0.0f;
                for (nk_size_t kk = 0; kk < k; kk++) {
                    nk_f32_t a_val, b_val;
                    nk_f16_to_f32(&a_row[kk], &a_val);
                    nk_f16_to_f32(&n_edge_ptr[j * k + kk], &b_val);
                    acc += a_val * b_val;
                }
                c[i * c_stride_elements + full_n_cols + j] = acc;
            }
        }
    }

    // M-edge: rows beyond full M-tiles
    if (m_remainder > 0 && num_n_tiles > 0) {
        nk_size_t const m_remainder_start = num_m_tiles * tile_dim;

        nk_f16_t *a_f16_m_edge = a_f16_edge;
        if (n_edge_rows == 0) {
            // Convert E4M3 → F16 (scalar)
            for (nk_size_t i = 0; i < m_remainder; i++) {
                nk_e4m3_t const *a_row = a + (m_remainder_start + i) * a_stride_elements;
                nk_f16_t *a_f16_row = a_f16_edge + i * k;
                for (nk_size_t j = 0; j < k; j++) { a_f16_row[j] = nk_e4m3_to_f16_lut_(a_row[j]); }
            }
        }
        else { a_f16_m_edge = a_f16_edge + m_remainder_start * k; }

        nk_f16_t const *b_tiles = (nk_f16_t const *)((char const *)b_packed + sizeof(nk_dots_sme_packed_header_t));
        nk_size_t const num_k_tiles = header->full_k_tiles;
        nk_size_t const k_tile_size = 32;

        for (nk_size_t i = 0; i < m_remainder; i++) {
            nk_f16_t const *a_row = (n_edge_rows > 0) ? (a_f16_edge + (m_remainder_start + i) * k)
                                                      : (a_f16_m_edge + i * k);
            for (nk_size_t n_tile = 0; n_tile < num_n_tiles; n_tile++) {
                nk_f32_t acc[16] = {0};
                for (nk_size_t k_tile = 0; k_tile < num_k_tiles; k_tile++) {
                    nk_size_t const k_offset = k_tile * k_tile_size;
                    nk_size_t const b_tile_idx = n_tile * num_k_tiles + k_tile;
                    nk_f16_t const *b_tile = b_tiles + b_tile_idx * tile_dim * k_tile_size;

                    for (nk_size_t jj = 0; jj < tile_dim; jj++) {
                        for (nk_size_t kk = 0; kk < k_tile_size && k_offset + kk < k; kk++) {
                            nk_f32_t a_val, b_val;
                            nk_f16_to_f32(&a_row[k_offset + kk], &a_val);
                            // Interleaved layout: b_idx = (k/2)*32 + n*2 + (k&1)
                            nk_size_t const b_idx = (kk / 2) * 32 + jj * 2 + (kk & 1);
                            nk_f16_to_f32(&b_tile[b_idx], &b_val);
                            acc[jj] += a_val * b_val;
                        }
                    }
                }
                for (nk_size_t jj = 0; jj < tile_dim; jj++) {
                    c[(m_remainder_start + i) * c_stride_elements + n_tile * tile_dim + jj] = acc[jj];
                }
            }
        }
    }

    // M-edge × N-edge corner (scalar F16 dot product)
    if (m_remainder > 0 && n_edge_rows > 0) {
        nk_size_t const m_remainder_start = num_m_tiles * tile_dim;
        nk_f16_t const *n_edge_ptr = (nk_f16_t const *)((char const *)b_packed + header->n_edge_offset);
        nk_f16_t const *a_m_edge = a_f16_edge + m_remainder_start * k;

        for (nk_size_t i = 0; i < m_remainder; i++) {
            nk_f16_t const *a_row = a_m_edge + i * k;
            for (nk_size_t j = 0; j < n_edge_rows; j++) {
                nk_f32_t acc = 0.0f;
                for (nk_size_t kk = 0; kk < k; kk++) {
                    nk_f32_t a_val, b_val;
                    nk_f16_to_f32(&a_row[kk], &a_val);
                    nk_f16_to_f32(&n_edge_ptr[j * k + kk], &b_val);
                    acc += a_val * b_val;
                }
                c[(m_remainder_start + i) * c_stride_elements + full_n_cols + j] = acc;
            }
        }
    }

    if (a_f16_edge) free(a_f16_edge);
}

/*  ============================================================================
 *  U8×U8 → U32 GEMM using SME outer products.
 *
 *  Uses svmopa_za32_u8_m for unsigned 8-bit integer outer product accumulate.
 *  This is available on Apple M4 (SME_I8I32 = 1, covers both signed and unsigned).
 *
 *  Tile dimensions identical to I8 → I32 (512-bit SVL):
 *    - Input vectors: 64 U8 elements (SVL/8 = 64)
 *    - Output tile: 16×16 U32 elements (ZA32)
 *    - Each output U32 is a dot product of 4 U8 pairs
 *  ============================================================================
 */

NK_PUBLIC nk_size_t nk_dots_u8u8u32_smei32i32_packed_size(nk_size_t n, nk_size_t k) {
    // Same dimensions as I8 → I32 since both are 8-bit
    return nk_dots_i8i8i32_smei32i32_packed_size(n, k);
}

NK_PUBLIC void nk_dots_u8u8u32_smei32i32_pack(  //
    nk_u8_t const *b, nk_size_t n, nk_size_t k, //
    nk_size_t b_stride, void *b_packed) {

    nk_size_t const svl_bytes = 64;
    nk_size_t const tile_rows = svl_bytes / sizeof(nk_u32_t);  // 16
    nk_size_t const tile_k_cols = svl_bytes / sizeof(nk_u8_t); // 64
    nk_size_t const tile_elements = tile_rows * tile_k_cols;   // 1024
    nk_size_t const b_stride_elements = b_stride / sizeof(nk_u8_t);

    nk_size_t const num_n_tiles = n / tile_rows;
    nk_size_t const num_k_tiles = (k + tile_k_cols - 1) / tile_k_cols;
    nk_size_t const n_remainder_rows = n - num_n_tiles * tile_rows;
    nk_size_t const total_tiles = num_n_tiles * num_k_tiles;

    nk_dots_sme_packed_header_t *header = (nk_dots_sme_packed_header_t *)b_packed;
    header->full_n_tiles = (nk_u32_t)num_n_tiles;
    header->full_k_tiles = (nk_u32_t)num_k_tiles;
    header->n_edge_rows = (nk_u32_t)n_remainder_rows;
    header->svl_bytes = (nk_u32_t)svl_bytes;

    nk_size_t const tiles_offset = sizeof(nk_dots_sme_packed_header_t);
    nk_size_t const n_edge_offset = tiles_offset + total_tiles * tile_elements * sizeof(nk_u8_t);
    header->n_edge_offset = (nk_u32_t)n_edge_offset;

    nk_u8_t *tiles_ptr = (nk_u8_t *)((char *)b_packed + tiles_offset);
    nk_u8_t *n_edge_ptr = (nk_u8_t *)((char *)b_packed + n_edge_offset);

    for (nk_size_t i = 0; i < total_tiles * tile_elements; i++) { tiles_ptr[i] = 0; }

    for (nk_size_t n_tile = 0; n_tile < num_n_tiles; n_tile++) {
        for (nk_size_t k_tile = 0; k_tile < num_k_tiles; k_tile++) {
            nk_size_t const tile_index = n_tile * num_k_tiles + k_tile;
            nk_u8_t *tile_output = tiles_ptr + tile_index * tile_elements;

            nk_size_t const src_row_start = n_tile * tile_rows;
            nk_size_t const src_col_start = k_tile * tile_k_cols;
            nk_size_t const cols_to_pack = (src_col_start + tile_k_cols <= k) ? tile_k_cols : (k - src_col_start);

            for (nk_size_t row = 0; row < tile_rows; row++) {
                for (nk_size_t col = 0; col < cols_to_pack; col++) {
                    nk_size_t const src_idx = (src_row_start + row) * b_stride_elements + src_col_start + col;
                    nk_size_t const dst_idx = col * tile_rows + row;
                    tile_output[dst_idx] = b[src_idx];
                }
            }
        }
    }

    if (n_remainder_rows > 0) {
        nk_size_t const remainder_start_row = num_n_tiles * tile_rows;
        for (nk_size_t row = 0; row < n_remainder_rows; row++) {
            for (nk_size_t col = 0; col < k; col++) {
                n_edge_ptr[row * k + col] = b[(remainder_start_row + row) * b_stride_elements + col];
            }
        }
    }
}

__arm_locally_streaming __arm_new("za") static void nk_dots_u8u8u32_smei32i32_kernel_( //
    nk_u8_t const *a, void const *b_packed, nk_u32_t *c,                               //
    nk_size_t m, nk_size_t n, nk_size_t k,                                             //
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) {

    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_size_t const num_n_tiles = header->full_n_tiles;
    nk_size_t const num_k_tiles = header->full_k_tiles;

    nk_size_t const tile_dim = NK_SME_CNTW;    // 16 for 512-bit SVL
    nk_size_t const k_tile_size = NK_SME_CNTB; // 64 for 512-bit SVL
    nk_size_t const tile_elements = tile_dim * k_tile_size;

    nk_u8_t const *b_tiles = (nk_u8_t const *)((char const *)b_packed + sizeof(nk_dots_sme_packed_header_t));

    svbool_t const ptrue_b = svptrue_b8();
    svbool_t const ptrue_s = svptrue_b32();

    nk_size_t const num_m_tiles = m / tile_dim;
    nk_size_t const block_dim = tile_dim * 2; // 32 for 2×2 blocking (using all 4 ZA32 tiles)

    // Process 2×2 tile blocks (32×32 output) for maximum tile utilization
    nk_size_t const num_m_blocks = num_m_tiles / 2;
    nk_size_t const num_n_blocks = num_n_tiles / 2;

    for (nk_size_t m_block = 0; m_block < num_m_blocks; m_block++) {
        nk_size_t const m_row = m_block * block_dim;

        for (nk_size_t n_block = 0; n_block < num_n_blocks; n_block++) {
            nk_size_t const n_col = n_block * block_dim;
            nk_size_t const n_tile_0 = n_block * 2;
            nk_size_t const n_tile_1 = n_block * 2 + 1;

            svzero_za();

            for (nk_size_t k_tile = 0; k_tile < num_k_tiles; k_tile++) {
                nk_size_t const k_offset = k_tile * k_tile_size;

                nk_size_t const b_tile_idx_0 = n_tile_0 * num_k_tiles + k_tile;
                nk_size_t const b_tile_idx_1 = n_tile_1 * num_k_tiles + k_tile;
                nk_u8_t const *b_tile_0 = b_tiles + b_tile_idx_0 * tile_elements;
                nk_u8_t const *b_tile_1 = b_tiles + b_tile_idx_1 * tile_elements;

                for (nk_size_t row = 0; row < tile_dim; row++) {
                    nk_u8_t const *a_ptr_0 = a + (m_row + row) * a_stride_elements + k_offset;
                    nk_u8_t const *a_ptr_1 = a + (m_row + tile_dim + row) * a_stride_elements + k_offset;
                    svuint8_t a_vec_0 = svld1_u8(ptrue_b, a_ptr_0);
                    svuint8_t a_vec_1 = svld1_u8(ptrue_b, a_ptr_1);

                    svuint8_t b_vec_0 = svld1_u8(ptrue_b, b_tile_0 + row * k_tile_size);
                    svuint8_t b_vec_1 = svld1_u8(ptrue_b, b_tile_1 + row * k_tile_size);

                    svmopa_za32_u8_m(0, ptrue_s, ptrue_s, a_vec_0, b_vec_0);
                    svmopa_za32_u8_m(1, ptrue_s, ptrue_s, a_vec_0, b_vec_1);
                    svmopa_za32_u8_m(2, ptrue_s, ptrue_s, a_vec_1, b_vec_0);
                    svmopa_za32_u8_m(3, ptrue_s, ptrue_s, a_vec_1, b_vec_1);
                }
            }

            for (nk_size_t row = 0; row < tile_dim; row++) {
                svst1_hor_za32(0, row, ptrue_s, (nk_i32_t *)(c + (m_row + row) * c_stride_elements + n_col));
                svst1_hor_za32(1, row, ptrue_s, (nk_i32_t *)(c + (m_row + row) * c_stride_elements + n_col + tile_dim));
                svst1_hor_za32(2, row, ptrue_s, (nk_i32_t *)(c + (m_row + tile_dim + row) * c_stride_elements + n_col));
                svst1_hor_za32(3, row, ptrue_s,
                               (nk_i32_t *)(c + (m_row + tile_dim + row) * c_stride_elements + n_col + tile_dim));
            }
        }

        if (num_n_tiles % 2 != 0) {
            nk_size_t const n_tile = num_n_tiles - 1;
            nk_size_t const n_col = n_tile * tile_dim;

            svzero_za();

            for (nk_size_t k_tile = 0; k_tile < num_k_tiles; k_tile++) {
                nk_size_t const k_offset = k_tile * k_tile_size;
                nk_size_t const b_tile_idx = n_tile * num_k_tiles + k_tile;
                nk_u8_t const *b_tile = b_tiles + b_tile_idx * tile_elements;

                for (nk_size_t row = 0; row < tile_dim; row++) {
                    nk_u8_t const *a_ptr_0 = a + (m_row + row) * a_stride_elements + k_offset;
                    nk_u8_t const *a_ptr_1 = a + (m_row + tile_dim + row) * a_stride_elements + k_offset;
                    svuint8_t a_vec_0 = svld1_u8(ptrue_b, a_ptr_0);
                    svuint8_t a_vec_1 = svld1_u8(ptrue_b, a_ptr_1);
                    svuint8_t b_vec = svld1_u8(ptrue_b, b_tile + row * k_tile_size);

                    svmopa_za32_u8_m(0, ptrue_s, ptrue_s, a_vec_0, b_vec);
                    svmopa_za32_u8_m(2, ptrue_s, ptrue_s, a_vec_1, b_vec);
                }
            }

            for (nk_size_t row = 0; row < tile_dim; row++) {
                svst1_hor_za32(0, row, ptrue_s, (nk_i32_t *)(c + (m_row + row) * c_stride_elements + n_col));
                svst1_hor_za32(2, row, ptrue_s, (nk_i32_t *)(c + (m_row + tile_dim + row) * c_stride_elements + n_col));
            }
        }
    }

    if (num_m_tiles % 2 != 0) {
        nk_size_t const m_tile = num_m_tiles - 1;
        nk_size_t const m_row = m_tile * tile_dim;

        for (nk_size_t n_block = 0; n_block < num_n_blocks; n_block++) {
            nk_size_t const n_col = n_block * block_dim;
            nk_size_t const n_tile_0 = n_block * 2;
            nk_size_t const n_tile_1 = n_block * 2 + 1;

            svzero_za();

            for (nk_size_t k_tile = 0; k_tile < num_k_tiles; k_tile++) {
                nk_size_t const k_offset = k_tile * k_tile_size;
                nk_size_t const b_tile_idx_0 = n_tile_0 * num_k_tiles + k_tile;
                nk_size_t const b_tile_idx_1 = n_tile_1 * num_k_tiles + k_tile;
                nk_u8_t const *b_tile_0 = b_tiles + b_tile_idx_0 * tile_elements;
                nk_u8_t const *b_tile_1 = b_tiles + b_tile_idx_1 * tile_elements;

                for (nk_size_t row = 0; row < tile_dim; row++) {
                    nk_u8_t const *a_ptr = a + (m_row + row) * a_stride_elements + k_offset;
                    svuint8_t a_vec = svld1_u8(ptrue_b, a_ptr);
                    svuint8_t b_vec_0 = svld1_u8(ptrue_b, b_tile_0 + row * k_tile_size);
                    svuint8_t b_vec_1 = svld1_u8(ptrue_b, b_tile_1 + row * k_tile_size);

                    svmopa_za32_u8_m(0, ptrue_s, ptrue_s, a_vec, b_vec_0);
                    svmopa_za32_u8_m(1, ptrue_s, ptrue_s, a_vec, b_vec_1);
                }
            }

            for (nk_size_t row = 0; row < tile_dim; row++) {
                svst1_hor_za32(0, row, ptrue_s, (nk_i32_t *)(c + (m_row + row) * c_stride_elements + n_col));
                svst1_hor_za32(1, row, ptrue_s, (nk_i32_t *)(c + (m_row + row) * c_stride_elements + n_col + tile_dim));
            }
        }

        if (num_n_tiles % 2 != 0) {
            nk_size_t const n_tile = num_n_tiles - 1;
            nk_size_t const n_col = n_tile * tile_dim;

            svzero_za();

            for (nk_size_t k_tile = 0; k_tile < num_k_tiles; k_tile++) {
                nk_size_t const k_offset = k_tile * k_tile_size;
                nk_size_t const b_tile_idx = n_tile * num_k_tiles + k_tile;
                nk_u8_t const *b_tile = b_tiles + b_tile_idx * tile_elements;

                for (nk_size_t row = 0; row < tile_dim; row++) {
                    nk_u8_t const *a_ptr = a + (m_row + row) * a_stride_elements + k_offset;
                    svuint8_t a_vec = svld1_u8(ptrue_b, a_ptr);
                    svuint8_t b_vec = svld1_u8(ptrue_b, b_tile + row * k_tile_size);

                    svmopa_za32_u8_m(0, ptrue_s, ptrue_s, a_vec, b_vec);
                }
            }

            for (nk_size_t row = 0; row < tile_dim; row++) {
                svst1_hor_za32(0, row, ptrue_s, (nk_i32_t *)(c + (m_row + row) * c_stride_elements + n_col));
            }
        }
    }
}

NK_PUBLIC void nk_dots_u8u8u32_smei32(                   //
    nk_u8_t const *a, void const *b_packed, nk_u32_t *c, //
    nk_size_t m, nk_size_t n, nk_size_t k,               //
    nk_size_t a_stride, nk_size_t c_stride) {

    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_size_t const num_n_tiles = header->full_n_tiles;
    nk_size_t const n_edge_rows = header->n_edge_rows;

    nk_size_t const a_stride_elements = a_stride / sizeof(nk_u8_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_u32_t);

    nk_size_t const tile_dim = 16;
    nk_size_t const num_m_tiles = m / tile_dim;
    nk_size_t const full_n_cols = num_n_tiles * tile_dim;

    if (num_m_tiles > 0 && num_n_tiles > 0) {
        nk_dots_u8u8u32_smei32i32_kernel_(a, b_packed, c, num_m_tiles * tile_dim, full_n_cols, k, a_stride_elements,
                                          c_stride_elements);
    }

    // Scalar fallback for N-edge (columns beyond full N-tiles)
    if (n_edge_rows > 0) {
        nk_u8_t const *n_edge_ptr = (nk_u8_t const *)((char const *)b_packed + header->n_edge_offset);
        for (nk_size_t i = 0; i < m; i++) {
            nk_u8_t const *a_row = a + i * a_stride_elements;
            for (nk_size_t j = 0; j < n_edge_rows; j++) {
                nk_u32_t acc = 0;
                for (nk_size_t kk = 0; kk < k; kk++) { acc += (nk_u32_t)a_row[kk] * (nk_u32_t)n_edge_ptr[j * k + kk]; }
                c[i * c_stride_elements + full_n_cols + j] = acc;
            }
        }
    }

    // Scalar fallback for M-edge (rows beyond full M-tiles)
    if (m > num_m_tiles * tile_dim && num_n_tiles > 0) {
        nk_size_t const m_remainder_start = num_m_tiles * tile_dim;
        nk_size_t const m_remainder_count = m - m_remainder_start;

        nk_u8_t const *b_tiles = (nk_u8_t const *)((char const *)b_packed + sizeof(nk_dots_sme_packed_header_t));
        nk_size_t const num_k_tiles = header->full_k_tiles;
        nk_size_t const k_tile_size = 64;

        for (nk_size_t i = 0; i < m_remainder_count; i++) {
            nk_u8_t const *a_row = a + (m_remainder_start + i) * a_stride_elements;
            for (nk_size_t n_tile = 0; n_tile < num_n_tiles; n_tile++) {
                nk_u32_t acc[16] = {0};
                for (nk_size_t k_tile = 0; k_tile < num_k_tiles; k_tile++) {
                    nk_size_t const k_offset = k_tile * k_tile_size;
                    nk_size_t const b_tile_idx = n_tile * num_k_tiles + k_tile;
                    nk_u8_t const *b_tile = b_tiles + b_tile_idx * tile_dim * k_tile_size;

                    for (nk_size_t j = 0; j < tile_dim; j++) {
                        for (nk_size_t kk = 0; kk < k_tile_size && k_offset + kk < k; kk++) {
                            acc[j] += (nk_u32_t)a_row[k_offset + kk] * (nk_u32_t)b_tile[kk * tile_dim + j];
                        }
                    }
                }
                for (nk_size_t j = 0; j < tile_dim; j++) {
                    c[(m_remainder_start + i) * c_stride_elements + n_tile * tile_dim + j] = acc[j];
                }
            }
        }
    }

    // M-edge × N-edge corner (scalar)
    if (m > num_m_tiles * tile_dim && n_edge_rows > 0) {
        nk_size_t const m_remainder_start = num_m_tiles * tile_dim;
        nk_size_t const m_remainder_count = m - m_remainder_start;
        nk_u8_t const *n_edge_ptr = (nk_u8_t const *)((char const *)b_packed + header->n_edge_offset);

        for (nk_size_t i = 0; i < m_remainder_count; i++) {
            nk_u8_t const *a_row = a + (m_remainder_start + i) * a_stride_elements;
            for (nk_size_t j = 0; j < n_edge_rows; j++) {
                nk_u32_t acc = 0;
                for (nk_size_t kk = 0; kk < k; kk++) { acc += (nk_u32_t)a_row[kk] * (nk_u32_t)n_edge_ptr[j * k + kk]; }
                c[(m_remainder_start + i) * c_stride_elements + full_n_cols + j] = acc;
            }
        }
    }
}

/*  ============================================================================
 *  F64×F64 → F64 GEMM using SME outer products with Kahan compensated summation.
 *
 *  Uses svmopa_za64_f64_m for F64×F64 → F64 outer product accumulate.
 *  This is available on Apple M4 (FEAT_SME_F64F64 = 1).
 *
 *  Kahan Summation Algorithm:
 *  Standard FP addition loses low-order bits: sum += x discards error.
 *  Kahan tracks the error and compensates:
 *    y = x - compensation      // Adjust x by accumulated error
 *    t = sum + y               // Standard addition
 *    compensation = (t - sum) - y  // Recovers the lost low-order bits
 *    sum = t
 *
 *  For GEMM, we apply Kahan summation when combining K-tile partial results.
 *  This gives ~16-17 decimal digits of precision vs standard F64's ~15.
 *
 *  Tile dimensions for F64 (512-bit SVL):
 *    - Input vectors: 8 F64 elements (SVL/64 = 8)
 *    - Output tile: 8×8 F64 elements (ZA64)
 *
 *  Expected performance: ~80-100 GFLOPS (overhead from Kahan compensation)
 *  ============================================================================
 */

NK_PUBLIC nk_size_t nk_dots_f64f64f64_smef64f64_packed_size(nk_size_t n, nk_size_t k) {
    nk_size_t const svl_bytes = 64;
    nk_size_t const tile_rows = svl_bytes / sizeof(nk_f64_t);   // 8 rows per ZA64 tile
    nk_size_t const tile_k_cols = svl_bytes / sizeof(nk_f64_t); // 8 F64 K elements per tile

    nk_size_t const full_n_tiles = n / tile_rows;
    nk_size_t const tiles_along_k = (k + tile_k_cols - 1) / tile_k_cols;
    nk_size_t const n_edge_rows = n - full_n_tiles * tile_rows;

    nk_size_t size = sizeof(nk_dots_sme_packed_header_t);
    size += full_n_tiles * tiles_along_k * tile_rows * tile_k_cols * sizeof(nk_f64_t);
    if (n_edge_rows > 0) size += n_edge_rows * k * sizeof(nk_f64_t);

    return size;
}

NK_PUBLIC void nk_dots_f64f64f64_smef64f64_pack( //
    nk_f64_t const *b, nk_size_t n, nk_size_t k, //
    nk_size_t b_stride, void *b_packed) {

    nk_size_t const svl_bytes = 64;
    nk_size_t const tile_rows = svl_bytes / sizeof(nk_f64_t);   // 8
    nk_size_t const tile_k_cols = svl_bytes / sizeof(nk_f64_t); // 8
    nk_size_t const tile_elements = tile_rows * tile_k_cols;    // 64
    nk_size_t const b_stride_elements = b_stride / sizeof(nk_f64_t);

    nk_size_t const num_n_tiles = n / tile_rows;
    nk_size_t const num_k_tiles = (k + tile_k_cols - 1) / tile_k_cols;
    nk_size_t const n_remainder_rows = n - num_n_tiles * tile_rows;
    nk_size_t const total_tiles = num_n_tiles * num_k_tiles;

    nk_dots_sme_packed_header_t *header = (nk_dots_sme_packed_header_t *)b_packed;
    header->full_n_tiles = (nk_u32_t)num_n_tiles;
    header->full_k_tiles = (nk_u32_t)num_k_tiles;
    header->n_edge_rows = (nk_u32_t)n_remainder_rows;
    header->svl_bytes = (nk_u32_t)svl_bytes;

    nk_size_t const tiles_offset = sizeof(nk_dots_sme_packed_header_t);
    nk_size_t const n_edge_offset = tiles_offset + total_tiles * tile_elements * sizeof(nk_f64_t);
    header->n_edge_offset = (nk_u32_t)n_edge_offset;

    nk_f64_t *tiles_ptr = (nk_f64_t *)((char *)b_packed + tiles_offset);
    nk_f64_t *n_edge_ptr = (nk_f64_t *)((char *)b_packed + n_edge_offset);

    for (nk_size_t i = 0; i < total_tiles * tile_elements; i++) { tiles_ptr[i] = 0.0; }

    for (nk_size_t n_tile = 0; n_tile < num_n_tiles; n_tile++) {
        for (nk_size_t k_tile = 0; k_tile < num_k_tiles; k_tile++) {
            nk_size_t const tile_index = n_tile * num_k_tiles + k_tile;
            nk_f64_t *tile_output = tiles_ptr + tile_index * tile_elements;

            nk_size_t const src_row_start = n_tile * tile_rows;
            nk_size_t const src_col_start = k_tile * tile_k_cols;
            nk_size_t const cols_to_pack = (src_col_start + tile_k_cols <= k) ? tile_k_cols : (k - src_col_start);

            for (nk_size_t row = 0; row < tile_rows; row++) {
                for (nk_size_t col = 0; col < cols_to_pack; col++) {
                    nk_size_t const src_idx = (src_row_start + row) * b_stride_elements + src_col_start + col;
                    nk_size_t const dst_idx = col * tile_rows + row;
                    tile_output[dst_idx] = b[src_idx];
                }
            }
        }
    }

    if (n_remainder_rows > 0) {
        nk_size_t const remainder_start_row = num_n_tiles * tile_rows;
        for (nk_size_t row = 0; row < n_remainder_rows; row++) {
            for (nk_size_t col = 0; col < k; col++) {
                n_edge_ptr[row * k + col] = b[(remainder_start_row + row) * b_stride_elements + col];
            }
        }
    }
}

// Batch size for Kahan compensation: accumulate this many K-tiles before extracting
// Tunable: 16-64 works well. 32 provides good balance of precision vs performance
#define NK_KAHAN_BATCH_SIZE 32

__arm_locally_streaming __arm_new("za") static void nk_dots_f64f64f64_smef64f64_kernel_( //
    nk_f64_t const *a, void const *b_packed, nk_f64_t *c,                                //
    nk_size_t m, nk_size_t n, nk_size_t k,                                               //
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) {

    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_size_t const num_n_tiles = header->full_n_tiles;
    nk_size_t const num_k_tiles = header->full_k_tiles;

    nk_size_t const tile_dim = NK_SME_CNTD;    // 8 for 512-bit SVL (ZA64 dimension)
    nk_size_t const k_tile_size = NK_SME_CNTD; // 8 for 512-bit SVL
    nk_size_t const tile_elements = tile_dim * k_tile_size;

    nk_f64_t const *b_tiles = (nk_f64_t const *)((char const *)b_packed + sizeof(nk_dots_sme_packed_header_t));

    svbool_t const ptrue_d = svptrue_b64();

    nk_size_t const num_m_tiles = m / tile_dim;
    nk_size_t const block_dim = tile_dim * 2; // 16 for 2×2 blocking (using 4 of 8 ZA64 tiles)

    // Process 2×2 tile blocks (16×16 output) for better tile utilization
    nk_size_t const num_m_blocks = num_m_tiles / 2;
    nk_size_t const num_n_blocks = num_n_tiles / 2;

    // Temporary buffers for extracting batch partial results (4 tiles × 8×8 = 256 F64 elements)
    nk_f64_t partial_0[64], partial_1[64], partial_2[64], partial_3[64];
    // Kahan compensation terms for each output element in the 2×2 block
    nk_f64_t comp_0[64], comp_1[64], comp_2[64], comp_3[64];
    // Accumulator buffers for running sum (outside of ZA)
    nk_f64_t acc_0[64], acc_1[64], acc_2[64], acc_3[64];

    // Determine batch size (clamp to available K-tiles)
    nk_size_t const batch_size = NK_KAHAN_BATCH_SIZE < num_k_tiles ? NK_KAHAN_BATCH_SIZE : num_k_tiles;

    // Process 2×2 blocks with BATCHED Kahan compensation
    for (nk_size_t m_block = 0; m_block < num_m_blocks; m_block++) {
        nk_size_t const m_row = m_block * block_dim;

        for (nk_size_t n_block = 0; n_block < num_n_blocks; n_block++) {
            nk_size_t const n_col = n_block * block_dim;
            nk_size_t const n_tile_0 = n_block * 2;
            nk_size_t const n_tile_1 = n_block * 2 + 1;

            // Initialize accumulators and compensation terms
            for (nk_size_t i = 0; i < 64; i++) {
                acc_0[i] = acc_1[i] = acc_2[i] = acc_3[i] = 0.0;
                comp_0[i] = comp_1[i] = comp_2[i] = comp_3[i] = 0.0;
            }

            // Process K-tiles in BATCHES for better performance
            for (nk_size_t k_batch_start = 0; k_batch_start < num_k_tiles; k_batch_start += batch_size) {
                nk_size_t const k_batch_end = (k_batch_start + batch_size < num_k_tiles) ? k_batch_start + batch_size
                                                                                         : num_k_tiles;

                // Zero all 4 ZA64 tiles ONCE per batch
                svzero_za();

                // Accumulate entire batch in ZA (no extraction per K-tile!)
                for (nk_size_t k_tile = k_batch_start; k_tile < k_batch_end; k_tile++) {
                    nk_size_t const k_offset = k_tile * k_tile_size;

                    // Get B tile pointers for both N-halves
                    nk_size_t const b_tile_idx_0 = n_tile_0 * num_k_tiles + k_tile;
                    nk_size_t const b_tile_idx_1 = n_tile_1 * num_k_tiles + k_tile;
                    nk_f64_t const *b_tile_0 = b_tiles + b_tile_idx_0 * tile_elements;
                    nk_f64_t const *b_tile_1 = b_tiles + b_tile_idx_1 * tile_elements;

                    // Compute outer products for this K-tile using 4 tiles
                    for (nk_size_t row = 0; row < tile_dim; row++) {
                        nk_f64_t const *a_ptr_0 = a + (m_row + row) * a_stride_elements + k_offset;
                        nk_f64_t const *a_ptr_1 = a + (m_row + tile_dim + row) * a_stride_elements + k_offset;
                        svfloat64_t a_vec_0 = svld1_f64(ptrue_d, a_ptr_0);
                        svfloat64_t a_vec_1 = svld1_f64(ptrue_d, a_ptr_1);

                        svfloat64_t b_vec_0 = svld1_f64(ptrue_d, b_tile_0 + row * k_tile_size);
                        svfloat64_t b_vec_1 = svld1_f64(ptrue_d, b_tile_1 + row * k_tile_size);

                        svmopa_za64_f64_m(0, ptrue_d, ptrue_d, a_vec_0, b_vec_0);
                        svmopa_za64_f64_m(1, ptrue_d, ptrue_d, a_vec_0, b_vec_1);
                        svmopa_za64_f64_m(2, ptrue_d, ptrue_d, a_vec_1, b_vec_0);
                        svmopa_za64_f64_m(3, ptrue_d, ptrue_d, a_vec_1, b_vec_1);
                    }
                }

                // Extract batch results ONCE per batch
                for (nk_size_t row = 0; row < tile_dim; row++) {
                    svst1_hor_za64(0, row, ptrue_d, partial_0 + row * tile_dim);
                    svst1_hor_za64(1, row, ptrue_d, partial_1 + row * tile_dim);
                    svst1_hor_za64(2, row, ptrue_d, partial_2 + row * tile_dim);
                    svst1_hor_za64(3, row, ptrue_d, partial_3 + row * tile_dim);
                }

                // Apply Kahan summation to combine batch with running accumulator
                // Use SVE vectorized operations for each row of 8 F64 elements
                for (nk_size_t row = 0; row < tile_dim; row++) {
                    nk_size_t const base_idx = row * tile_dim;

                    // Load current accumulators and compensation for this row
                    svfloat64_t acc0_v = svld1_f64(ptrue_d, acc_0 + base_idx);
                    svfloat64_t acc1_v = svld1_f64(ptrue_d, acc_1 + base_idx);
                    svfloat64_t acc2_v = svld1_f64(ptrue_d, acc_2 + base_idx);
                    svfloat64_t acc3_v = svld1_f64(ptrue_d, acc_3 + base_idx);

                    svfloat64_t comp0_v = svld1_f64(ptrue_d, comp_0 + base_idx);
                    svfloat64_t comp1_v = svld1_f64(ptrue_d, comp_1 + base_idx);
                    svfloat64_t comp2_v = svld1_f64(ptrue_d, comp_2 + base_idx);
                    svfloat64_t comp3_v = svld1_f64(ptrue_d, comp_3 + base_idx);

                    // Load batch partial results
                    svfloat64_t part0_v = svld1_f64(ptrue_d, partial_0 + base_idx);
                    svfloat64_t part1_v = svld1_f64(ptrue_d, partial_1 + base_idx);
                    svfloat64_t part2_v = svld1_f64(ptrue_d, partial_2 + base_idx);
                    svfloat64_t part3_v = svld1_f64(ptrue_d, partial_3 + base_idx);

                    // Kahan summation: y = partial - comp; t = acc + y; comp = (t - acc) - y; acc = t
                    svfloat64_t y0 = svsub_f64_x(ptrue_d, part0_v, comp0_v);
                    svfloat64_t t0 = svadd_f64_x(ptrue_d, acc0_v, y0);
                    comp0_v = svsub_f64_x(ptrue_d, svsub_f64_x(ptrue_d, t0, acc0_v), y0);
                    acc0_v = t0;

                    svfloat64_t y1 = svsub_f64_x(ptrue_d, part1_v, comp1_v);
                    svfloat64_t t1 = svadd_f64_x(ptrue_d, acc1_v, y1);
                    comp1_v = svsub_f64_x(ptrue_d, svsub_f64_x(ptrue_d, t1, acc1_v), y1);
                    acc1_v = t1;

                    svfloat64_t y2 = svsub_f64_x(ptrue_d, part2_v, comp2_v);
                    svfloat64_t t2 = svadd_f64_x(ptrue_d, acc2_v, y2);
                    comp2_v = svsub_f64_x(ptrue_d, svsub_f64_x(ptrue_d, t2, acc2_v), y2);
                    acc2_v = t2;

                    svfloat64_t y3 = svsub_f64_x(ptrue_d, part3_v, comp3_v);
                    svfloat64_t t3 = svadd_f64_x(ptrue_d, acc3_v, y3);
                    comp3_v = svsub_f64_x(ptrue_d, svsub_f64_x(ptrue_d, t3, acc3_v), y3);
                    acc3_v = t3;

                    // Store updated accumulators and compensation
                    svst1_f64(ptrue_d, acc_0 + base_idx, acc0_v);
                    svst1_f64(ptrue_d, acc_1 + base_idx, acc1_v);
                    svst1_f64(ptrue_d, acc_2 + base_idx, acc2_v);
                    svst1_f64(ptrue_d, acc_3 + base_idx, acc3_v);

                    svst1_f64(ptrue_d, comp_0 + base_idx, comp0_v);
                    svst1_f64(ptrue_d, comp_1 + base_idx, comp1_v);
                    svst1_f64(ptrue_d, comp_2 + base_idx, comp2_v);
                    svst1_f64(ptrue_d, comp_3 + base_idx, comp3_v);
                }
            }

            // Store final accumulated results to C
            for (nk_size_t row = 0; row < tile_dim; row++) {
                nk_f64_t *c_row_0 = c + (m_row + row) * c_stride_elements + n_col;
                nk_f64_t *c_row_1 = c + (m_row + tile_dim + row) * c_stride_elements + n_col;
                nk_size_t const base_idx = row * tile_dim;
                for (nk_size_t col = 0; col < tile_dim; col++) {
                    c_row_0[col] = acc_0[base_idx + col];
                    c_row_0[tile_dim + col] = acc_1[base_idx + col];
                    c_row_1[col] = acc_2[base_idx + col];
                    c_row_1[tile_dim + col] = acc_3[base_idx + col];
                }
            }
        }

        // Handle odd N-tile at end of row (if num_n_tiles is odd)
        if (num_n_tiles % 2 != 0) {
            nk_size_t const n_tile = num_n_tiles - 1;
            nk_size_t const n_col = n_tile * tile_dim;

            for (nk_size_t i = 0; i < 64; i++) {
                acc_0[i] = acc_2[i] = 0.0;
                comp_0[i] = comp_2[i] = 0.0;
            }

            for (nk_size_t k_batch_start = 0; k_batch_start < num_k_tiles; k_batch_start += batch_size) {
                nk_size_t const k_batch_end = (k_batch_start + batch_size < num_k_tiles) ? k_batch_start + batch_size
                                                                                         : num_k_tiles;

                svzero_za();

                for (nk_size_t k_tile = k_batch_start; k_tile < k_batch_end; k_tile++) {
                    nk_size_t const k_offset = k_tile * k_tile_size;
                    nk_size_t const b_tile_idx = n_tile * num_k_tiles + k_tile;
                    nk_f64_t const *b_tile = b_tiles + b_tile_idx * tile_elements;

                    for (nk_size_t row = 0; row < tile_dim; row++) {
                        nk_f64_t const *a_ptr_0 = a + (m_row + row) * a_stride_elements + k_offset;
                        nk_f64_t const *a_ptr_1 = a + (m_row + tile_dim + row) * a_stride_elements + k_offset;
                        svfloat64_t a_vec_0 = svld1_f64(ptrue_d, a_ptr_0);
                        svfloat64_t a_vec_1 = svld1_f64(ptrue_d, a_ptr_1);
                        svfloat64_t b_vec = svld1_f64(ptrue_d, b_tile + row * k_tile_size);

                        svmopa_za64_f64_m(0, ptrue_d, ptrue_d, a_vec_0, b_vec);
                        svmopa_za64_f64_m(2, ptrue_d, ptrue_d, a_vec_1, b_vec);
                    }
                }

                for (nk_size_t row = 0; row < tile_dim; row++) {
                    svst1_hor_za64(0, row, ptrue_d, partial_0 + row * tile_dim);
                    svst1_hor_za64(2, row, ptrue_d, partial_2 + row * tile_dim);
                }

                for (nk_size_t row = 0; row < tile_dim; row++) {
                    nk_size_t const base_idx = row * tile_dim;
                    svfloat64_t acc0_v = svld1_f64(ptrue_d, acc_0 + base_idx);
                    svfloat64_t acc2_v = svld1_f64(ptrue_d, acc_2 + base_idx);
                    svfloat64_t comp0_v = svld1_f64(ptrue_d, comp_0 + base_idx);
                    svfloat64_t comp2_v = svld1_f64(ptrue_d, comp_2 + base_idx);
                    svfloat64_t part0_v = svld1_f64(ptrue_d, partial_0 + base_idx);
                    svfloat64_t part2_v = svld1_f64(ptrue_d, partial_2 + base_idx);

                    svfloat64_t y0 = svsub_f64_x(ptrue_d, part0_v, comp0_v);
                    svfloat64_t t0 = svadd_f64_x(ptrue_d, acc0_v, y0);
                    comp0_v = svsub_f64_x(ptrue_d, svsub_f64_x(ptrue_d, t0, acc0_v), y0);
                    acc0_v = t0;

                    svfloat64_t y2 = svsub_f64_x(ptrue_d, part2_v, comp2_v);
                    svfloat64_t t2 = svadd_f64_x(ptrue_d, acc2_v, y2);
                    comp2_v = svsub_f64_x(ptrue_d, svsub_f64_x(ptrue_d, t2, acc2_v), y2);
                    acc2_v = t2;

                    svst1_f64(ptrue_d, acc_0 + base_idx, acc0_v);
                    svst1_f64(ptrue_d, acc_2 + base_idx, acc2_v);
                    svst1_f64(ptrue_d, comp_0 + base_idx, comp0_v);
                    svst1_f64(ptrue_d, comp_2 + base_idx, comp2_v);
                }
            }

            for (nk_size_t row = 0; row < tile_dim; row++) {
                nk_f64_t *c_row_0 = c + (m_row + row) * c_stride_elements + n_col;
                nk_f64_t *c_row_1 = c + (m_row + tile_dim + row) * c_stride_elements + n_col;
                nk_size_t const base_idx = row * tile_dim;
                for (nk_size_t col = 0; col < tile_dim; col++) {
                    c_row_0[col] = acc_0[base_idx + col];
                    c_row_1[col] = acc_2[base_idx + col];
                }
            }
        }
    }

    // Handle odd M-tile at end (if num_m_tiles is odd)
    if (num_m_tiles % 2 != 0) {
        nk_size_t const m_tile = num_m_tiles - 1;
        nk_size_t const m_row = m_tile * tile_dim;

        // Process 1×2 tile blocks using tiles 0 and 1
        for (nk_size_t n_block = 0; n_block < num_n_blocks; n_block++) {
            nk_size_t const n_col = n_block * block_dim;
            nk_size_t const n_tile_0 = n_block * 2;
            nk_size_t const n_tile_1 = n_block * 2 + 1;

            for (nk_size_t i = 0; i < 64; i++) {
                acc_0[i] = acc_1[i] = 0.0;
                comp_0[i] = comp_1[i] = 0.0;
            }

            for (nk_size_t k_batch_start = 0; k_batch_start < num_k_tiles; k_batch_start += batch_size) {
                nk_size_t const k_batch_end = (k_batch_start + batch_size < num_k_tiles) ? k_batch_start + batch_size
                                                                                         : num_k_tiles;

                svzero_za();

                for (nk_size_t k_tile = k_batch_start; k_tile < k_batch_end; k_tile++) {
                    nk_size_t const k_offset = k_tile * k_tile_size;
                    nk_size_t const b_tile_idx_0 = n_tile_0 * num_k_tiles + k_tile;
                    nk_size_t const b_tile_idx_1 = n_tile_1 * num_k_tiles + k_tile;
                    nk_f64_t const *b_tile_0 = b_tiles + b_tile_idx_0 * tile_elements;
                    nk_f64_t const *b_tile_1 = b_tiles + b_tile_idx_1 * tile_elements;

                    for (nk_size_t row = 0; row < tile_dim; row++) {
                        nk_f64_t const *a_ptr = a + (m_row + row) * a_stride_elements + k_offset;
                        svfloat64_t a_vec = svld1_f64(ptrue_d, a_ptr);
                        svfloat64_t b_vec_0 = svld1_f64(ptrue_d, b_tile_0 + row * k_tile_size);
                        svfloat64_t b_vec_1 = svld1_f64(ptrue_d, b_tile_1 + row * k_tile_size);

                        svmopa_za64_f64_m(0, ptrue_d, ptrue_d, a_vec, b_vec_0);
                        svmopa_za64_f64_m(1, ptrue_d, ptrue_d, a_vec, b_vec_1);
                    }
                }

                for (nk_size_t row = 0; row < tile_dim; row++) {
                    svst1_hor_za64(0, row, ptrue_d, partial_0 + row * tile_dim);
                    svst1_hor_za64(1, row, ptrue_d, partial_1 + row * tile_dim);
                }

                for (nk_size_t row = 0; row < tile_dim; row++) {
                    nk_size_t const base_idx = row * tile_dim;
                    svfloat64_t acc0_v = svld1_f64(ptrue_d, acc_0 + base_idx);
                    svfloat64_t acc1_v = svld1_f64(ptrue_d, acc_1 + base_idx);
                    svfloat64_t comp0_v = svld1_f64(ptrue_d, comp_0 + base_idx);
                    svfloat64_t comp1_v = svld1_f64(ptrue_d, comp_1 + base_idx);
                    svfloat64_t part0_v = svld1_f64(ptrue_d, partial_0 + base_idx);
                    svfloat64_t part1_v = svld1_f64(ptrue_d, partial_1 + base_idx);

                    svfloat64_t y0 = svsub_f64_x(ptrue_d, part0_v, comp0_v);
                    svfloat64_t t0 = svadd_f64_x(ptrue_d, acc0_v, y0);
                    comp0_v = svsub_f64_x(ptrue_d, svsub_f64_x(ptrue_d, t0, acc0_v), y0);
                    acc0_v = t0;

                    svfloat64_t y1 = svsub_f64_x(ptrue_d, part1_v, comp1_v);
                    svfloat64_t t1 = svadd_f64_x(ptrue_d, acc1_v, y1);
                    comp1_v = svsub_f64_x(ptrue_d, svsub_f64_x(ptrue_d, t1, acc1_v), y1);
                    acc1_v = t1;

                    svst1_f64(ptrue_d, acc_0 + base_idx, acc0_v);
                    svst1_f64(ptrue_d, acc_1 + base_idx, acc1_v);
                    svst1_f64(ptrue_d, comp_0 + base_idx, comp0_v);
                    svst1_f64(ptrue_d, comp_1 + base_idx, comp1_v);
                }
            }

            for (nk_size_t row = 0; row < tile_dim; row++) {
                nk_f64_t *c_row = c + (m_row + row) * c_stride_elements + n_col;
                nk_size_t const base_idx = row * tile_dim;
                for (nk_size_t col = 0; col < tile_dim; col++) {
                    c_row[col] = acc_0[base_idx + col];
                    c_row[tile_dim + col] = acc_1[base_idx + col];
                }
            }
        }

        // Handle final single tile (odd M × odd N)
        if (num_n_tiles % 2 != 0) {
            nk_size_t const n_tile = num_n_tiles - 1;
            nk_size_t const n_col = n_tile * tile_dim;

            for (nk_size_t i = 0; i < 64; i++) {
                acc_0[i] = 0.0;
                comp_0[i] = 0.0;
            }

            for (nk_size_t k_batch_start = 0; k_batch_start < num_k_tiles; k_batch_start += batch_size) {
                nk_size_t const k_batch_end = (k_batch_start + batch_size < num_k_tiles) ? k_batch_start + batch_size
                                                                                         : num_k_tiles;

                svzero_za();

                for (nk_size_t k_tile = k_batch_start; k_tile < k_batch_end; k_tile++) {
                    nk_size_t const k_offset = k_tile * k_tile_size;
                    nk_size_t const b_tile_idx = n_tile * num_k_tiles + k_tile;
                    nk_f64_t const *b_tile = b_tiles + b_tile_idx * tile_elements;

                    for (nk_size_t row = 0; row < tile_dim; row++) {
                        nk_f64_t const *a_ptr = a + (m_row + row) * a_stride_elements + k_offset;
                        svfloat64_t a_vec = svld1_f64(ptrue_d, a_ptr);
                        svfloat64_t b_vec = svld1_f64(ptrue_d, b_tile + row * k_tile_size);

                        svmopa_za64_f64_m(0, ptrue_d, ptrue_d, a_vec, b_vec);
                    }
                }

                for (nk_size_t row = 0; row < tile_dim; row++) {
                    svst1_hor_za64(0, row, ptrue_d, partial_0 + row * tile_dim);
                }

                for (nk_size_t row = 0; row < tile_dim; row++) {
                    nk_size_t const base_idx = row * tile_dim;
                    svfloat64_t acc0_v = svld1_f64(ptrue_d, acc_0 + base_idx);
                    svfloat64_t comp0_v = svld1_f64(ptrue_d, comp_0 + base_idx);
                    svfloat64_t part0_v = svld1_f64(ptrue_d, partial_0 + base_idx);

                    svfloat64_t y0 = svsub_f64_x(ptrue_d, part0_v, comp0_v);
                    svfloat64_t t0 = svadd_f64_x(ptrue_d, acc0_v, y0);
                    comp0_v = svsub_f64_x(ptrue_d, svsub_f64_x(ptrue_d, t0, acc0_v), y0);
                    acc0_v = t0;

                    svst1_f64(ptrue_d, acc_0 + base_idx, acc0_v);
                    svst1_f64(ptrue_d, comp_0 + base_idx, comp0_v);
                }
            }

            for (nk_size_t row = 0; row < tile_dim; row++) {
                nk_f64_t *c_row = c + (m_row + row) * c_stride_elements + n_col;
                nk_size_t const base_idx = row * tile_dim;
                for (nk_size_t col = 0; col < tile_dim; col++) { c_row[col] = acc_0[base_idx + col]; }
            }
        }
    }
}

NK_PUBLIC void nk_dots_f64f64f64_smef64(                  //
    nk_f64_t const *a, void const *b_packed, nk_f64_t *c, //
    nk_size_t m, nk_size_t n, nk_size_t k,                //
    nk_size_t a_stride, nk_size_t c_stride) {

    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_size_t const num_n_tiles = header->full_n_tiles;
    nk_size_t const n_edge_rows = header->n_edge_rows;

    nk_size_t const a_stride_elements = a_stride / sizeof(nk_f64_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f64_t);

    nk_size_t const tile_dim = 8; // ZA64 dimension
    nk_size_t const num_m_tiles = m / tile_dim;
    nk_size_t const full_n_cols = num_n_tiles * tile_dim;

    if (num_m_tiles > 0 && num_n_tiles > 0) {
        nk_dots_f64f64f64_smef64f64_kernel_(a, b_packed, c, num_m_tiles * tile_dim, full_n_cols, k, a_stride_elements,
                                            c_stride_elements);
    }

    // Scalar fallback for N-edge (columns beyond full N-tiles)
    if (n_edge_rows > 0) {
        nk_f64_t const *n_edge_ptr = (nk_f64_t const *)((char const *)b_packed + header->n_edge_offset);
        for (nk_size_t i = 0; i < m; i++) {
            nk_f64_t const *a_row = a + i * a_stride_elements;
            for (nk_size_t j = 0; j < n_edge_rows; j++) {
                nk_f64_t acc = 0.0;
                for (nk_size_t kk = 0; kk < k; kk++) { acc += a_row[kk] * n_edge_ptr[j * k + kk]; }
                c[i * c_stride_elements + full_n_cols + j] = acc;
            }
        }
    }

    // Scalar fallback for M-edge (rows beyond full M-tiles)
    if (m > num_m_tiles * tile_dim && num_n_tiles > 0) {
        nk_size_t const m_remainder_start = num_m_tiles * tile_dim;
        nk_size_t const m_remainder_count = m - m_remainder_start;

        nk_f64_t const *b_tiles = (nk_f64_t const *)((char const *)b_packed + sizeof(nk_dots_sme_packed_header_t));
        nk_size_t const num_k_tiles = header->full_k_tiles;
        nk_size_t const k_tile_size = 8;

        for (nk_size_t i = 0; i < m_remainder_count; i++) {
            nk_f64_t const *a_row = a + (m_remainder_start + i) * a_stride_elements;
            for (nk_size_t n_tile = 0; n_tile < num_n_tiles; n_tile++) {
                nk_f64_t acc[8] = {0};
                nk_f64_t comp[8] = {0}; // Kahan compensation terms

                for (nk_size_t k_tile = 0; k_tile < num_k_tiles; k_tile++) {
                    nk_size_t const k_offset = k_tile * k_tile_size;
                    nk_size_t const b_tile_idx = n_tile * num_k_tiles + k_tile;
                    nk_f64_t const *b_tile = b_tiles + b_tile_idx * tile_dim * k_tile_size;

                    for (nk_size_t j = 0; j < tile_dim; j++) {
                        for (nk_size_t kk = 0; kk < k_tile_size && k_offset + kk < k; kk++) {
                            nk_f64_t const product = a_row[k_offset + kk] * b_tile[kk * tile_dim + j];
                            nk_f64_t const y = product - comp[j];
                            nk_f64_t const t = acc[j] + y;
                            comp[j] = (t - acc[j]) - y;
                            acc[j] = t;
                        }
                    }
                }
                for (nk_size_t j = 0; j < tile_dim; j++) {
                    c[(m_remainder_start + i) * c_stride_elements + n_tile * tile_dim + j] = acc[j];
                }
            }
        }
    }

    // M-edge × N-edge corner (scalar)
    if (m > num_m_tiles * tile_dim && n_edge_rows > 0) {
        nk_size_t const m_remainder_start = num_m_tiles * tile_dim;
        nk_size_t const m_remainder_count = m - m_remainder_start;
        nk_f64_t const *n_edge_ptr = (nk_f64_t const *)((char const *)b_packed + header->n_edge_offset);

        for (nk_size_t i = 0; i < m_remainder_count; i++) {
            nk_f64_t const *a_row = a + (m_remainder_start + i) * a_stride_elements;
            for (nk_size_t j = 0; j < n_edge_rows; j++) {
                nk_f64_t acc = 0.0;
                for (nk_size_t kk = 0; kk < k; kk++) { acc += a_row[kk] * n_edge_ptr[j * k + kk]; }
                c[(m_remainder_start + i) * c_stride_elements + full_n_cols + j] = acc;
            }
        }
    }
}

#if defined(__cplusplus)
} // extern "C"
#endif

#pragma clang attribute pop
#pragma GCC pop_options
#endif // NK_TARGET_SME
#endif // NK_TARGET_ARM_

#endif // NK_DOTS_SME_H

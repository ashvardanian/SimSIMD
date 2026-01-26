/**
 *  @brief SIMD-accelerated batched binary set computations for ARM SME.
 *  @file include/numkong/sets/smebi32.h
 *  @sa include/numkong/sets.h
 *  @author Ash Vardanian
 *
 *  Uses ARM Scalable Matrix Extension (SME) for efficient binary set operations.
 *  Leverages streaming mode's wider vectors (512-bit on Apple M4) for fast
 *  XOR+POPCNT operations on binary vectors.
 *
 *  @section smebi32_math Mathematical Foundation
 *
 *  Hamming distance: popcount(a XOR b) = number of differing bits
 *
 *  Jaccard distance using intersection:
 *    intersection = popcount(a AND b)
 *    union = popcount(a) + popcount(b) - intersection
 *    jaccard = 1 - intersection / union
 *
 *  @section smebi32_tiles SME Dimensions (512-bit SVL)
 *
 *  - svcntw(): 16 (number of 32-bit elements per vector)
 *  - svcntb(): 64 (number of bytes per SVE vector)
 *  - Tile blocking: 16x16 output tiles for cache efficiency
 *  - Depth processing: 64 bytes (512 bits) per iteration
 *
 *  @section smebi32_perf Performance Characteristics (Apple M4)
 *
 *  - SVL: 512 bits (64 bytes)
 *  - Streaming mode provides dedicated register file
 *  - Streaming mode overhead: ~50-100 cycles for SMSTART/SMSTOP
 */

#ifndef NK_SETS_SMEBI32_H
#define NK_SETS_SMEBI32_H

#if NK_TARGET_ARM_
#if NK_TARGET_SMEBI32
#pragma GCC push_options
#pragma GCC target("+sme")
#pragma clang attribute push(__attribute__((target("sme"))), apply_to = function)

#include "numkong/types.h"
#include "numkong/sets/serial.h"

#include <arm_sme.h>
#include <arm_sve.h>
#include <stdlib.h> // aligned_alloc, free

// Aligned memory allocation macros for packed buffers
#define NK_ALIGNED_ALLOC_(alignment, size) aligned_alloc((alignment), (size))
#define NK_ALIGNED_FREE_(ptr)              free((ptr))

#if defined(__cplusplus)
extern "C" {
#endif

// Apple M4 SVL constants (512-bit streaming vector length)
// These are used in non-streaming functions to avoid calling streaming intrinsics
#define NK_SME_SVL_BYTES 64  // svcntsb() equivalent
#define NK_SME_SVL_WORDS 16  // svcntsw() equivalent
#define NK_SME_SVL_BITS  512 // SVL in bits

/**
 *  SME-specific packed buffer header for binary set operations (64-byte aligned).
 *  Layout optimized for SME streaming mode access patterns.
 */
typedef struct {
    nk_u32_t row_tile_count;   // ceiling(rows / tile_dim)
    nk_u32_t depth_tile_count; // ceiling(depth_bits / depth_tile_bits)
    nk_u32_t rows;             // actual row count
    nk_u32_t depth_bits;       // actual depth in bits
    nk_u32_t svl_bytes;        // SVL at pack time for validation
    nk_u32_t norms_offset;     // byte offset to norms (0 if none)
    nk_u32_t reserved[10];     // padding to 64 bytes
} nk_sets_smebi32_packed_header_t;

/**
 *  Calculate packed buffer size for Hamming distance with SME using BMOPA.
 *  Layout: header + tiled packed data in column-major u32 format.
 *
 *  For BMOPA, data is packed as u32 values in column-major order within each tile:
 *  - tile_dim = 16 rows per tile
 *  - depth_tile_size = 16 u32 values per depth tile (512 bits)
 *  - tile_elements = 16 × 16 = 256 u32 values per tile
 *  - Layout: tile[depth_u32 * tile_dim + row] for contiguous SVE loads
 */
NK_PUBLIC nk_size_t nk_hammings_packed_size_u1_smebi32(nk_size_t row_count, nk_size_t depth_bits) {
    nk_size_t const tile_dim = NK_SME_SVL_WORDS;        // 16 rows per tile
    nk_size_t const depth_tile_size = NK_SME_SVL_WORDS; // 16 u32 per depth tile = 512 bits

    nk_size_t const depth_u32 = (depth_bits + 31) / 32;
    nk_size_t const row_tile_count = (row_count + tile_dim - 1) / tile_dim;
    nk_size_t const depth_tile_count = (depth_u32 + depth_tile_size - 1) / depth_tile_size;

    nk_size_t const tile_elements = tile_dim * depth_tile_size; // 256 u32 per tile
    nk_size_t size = sizeof(nk_sets_smebi32_packed_header_t);
    size += row_tile_count * depth_tile_count * tile_elements * sizeof(nk_u32_t);

    return size;
}

/**
 *  Calculate packed buffer size for Jaccard distance with SME using BMOPA.
 *  Layout: header + tiled packed data in column-major u32 format + norms array.
 */
NK_PUBLIC nk_size_t nk_jaccards_packed_size_u1_smebi32(nk_size_t row_count, nk_size_t depth_bits) {
    nk_size_t const tile_dim = NK_SME_SVL_WORDS;        // 16 rows per tile
    nk_size_t const depth_tile_size = NK_SME_SVL_WORDS; // 16 u32 per depth tile

    nk_size_t const depth_u32 = (depth_bits + 31) / 32;
    nk_size_t const row_tile_count = (row_count + tile_dim - 1) / tile_dim;
    nk_size_t const depth_tile_count = (depth_u32 + depth_tile_size - 1) / depth_tile_size;

    nk_size_t const tile_elements = tile_dim * depth_tile_size;
    nk_size_t size = sizeof(nk_sets_smebi32_packed_header_t);
    size += row_tile_count * depth_tile_count * tile_elements * sizeof(nk_u32_t);
    size += row_count * sizeof(nk_u32_t); // norms

    return size;
}

/**
 *  Pack binary vectors for Hamming distance with SME BMOPA.
 *  Uses column-major u32 layout within each tile for efficient SVE loads.
 *
 *  For BMOPA outer product: Zn[i] and Zm[j] are u32 values at depth position k.
 *  Packing stores: tile[depth_u32 * tile_dim + row] = source[row][depth_u32]
 *  This enables contiguous svld1_u32 loads for both BMOPA operands.
 */
NK_PUBLIC void nk_hammings_pack_u1_smebi32(nk_u1x8_t const *b, nk_size_t row_count, nk_size_t depth_bits,
                                           nk_size_t b_stride_in_bytes, void *b_packed) {
    nk_size_t const svl_bytes = NK_SME_SVL_BYTES;
    nk_size_t const tile_dim = NK_SME_SVL_WORDS;        // 16 rows per tile
    nk_size_t const depth_tile_size = NK_SME_SVL_WORDS; // 16 u32 per depth tile
    nk_size_t const tile_elements = tile_dim * depth_tile_size;

    nk_size_t const depth_u32_total = (depth_bits + 31) / 32;
    nk_size_t const row_tile_count = (row_count + tile_dim - 1) / tile_dim;
    nk_size_t const depth_tile_count = (depth_u32_total + depth_tile_size - 1) / depth_tile_size;
    nk_size_t const total_tiles = row_tile_count * depth_tile_count;

    nk_sets_smebi32_packed_header_t *header = (nk_sets_smebi32_packed_header_t *)b_packed;
    header->row_tile_count = (nk_u32_t)row_tile_count;
    header->depth_tile_count = (nk_u32_t)depth_tile_count;
    header->rows = (nk_u32_t)row_count;
    header->depth_bits = (nk_u32_t)depth_bits;
    header->svl_bytes = (nk_u32_t)svl_bytes;
    header->norms_offset = 0;

    nk_u32_t *tiles_ptr = (nk_u32_t *)((char *)b_packed + sizeof(nk_sets_smebi32_packed_header_t));

    // Zero-initialize all tiles (partial tiles stay zero-padded for predicated loads)
    for (nk_size_t i = 0; i < total_tiles * tile_elements; i++) tiles_ptr[i] = 0;

    // Pack tiles: column-major u32 within each tile for efficient SVE loads
    for (nk_size_t row_tile = 0; row_tile < row_tile_count; row_tile++) {
        for (nk_size_t depth_tile = 0; depth_tile < depth_tile_count; depth_tile++) {
            nk_size_t const tile_index = row_tile * depth_tile_count + depth_tile;
            nk_u32_t *tile_output = tiles_ptr + tile_index * tile_elements;

            nk_size_t const src_row_start = row_tile * tile_dim;
            nk_size_t const src_u32_start = depth_tile * depth_tile_size;
            nk_size_t const rows_to_pack = (src_row_start + tile_dim <= row_count) ? tile_dim
                                                                                   : (row_count - src_row_start);
            nk_size_t const u32s_to_pack = (src_u32_start + depth_tile_size <= depth_u32_total)
                                               ? depth_tile_size
                                               : (depth_u32_total > src_u32_start ? depth_u32_total - src_u32_start
                                                                                  : 0);

            // Column-major packing: tile_output[col * tile_dim + row]
            for (nk_size_t row = 0; row < rows_to_pack; row++) {
                nk_u32_t const *src_row = (nk_u32_t const *)((char const *)b +
                                                             (src_row_start + row) * b_stride_in_bytes);
                for (nk_size_t col = 0; col < u32s_to_pack; col++) {
                    nk_size_t const dst_idx = col * tile_dim + row; // Column-major!
                    tile_output[dst_idx] = src_row[src_u32_start + col];
                }
            }
        }
    }
}

/**
 *  Pack binary vectors for Jaccard distance with SME BMOPA.
 *  Uses column-major u32 layout and computes population count norms.
 */
NK_PUBLIC void nk_jaccards_pack_u1_smebi32(nk_u1x8_t const *b, nk_size_t row_count, nk_size_t depth_bits,
                                           nk_size_t b_stride_in_bytes, void *b_packed) {
    nk_size_t const svl_bytes = NK_SME_SVL_BYTES;
    nk_size_t const tile_dim = NK_SME_SVL_WORDS;        // 16 rows per tile
    nk_size_t const depth_tile_size = NK_SME_SVL_WORDS; // 16 u32 per depth tile
    nk_size_t const tile_elements = tile_dim * depth_tile_size;
    nk_size_t const depth_in_bytes = (depth_bits + 7) / 8;

    nk_size_t const depth_u32_total = (depth_bits + 31) / 32;
    nk_size_t const row_tile_count = (row_count + tile_dim - 1) / tile_dim;
    nk_size_t const depth_tile_count = (depth_u32_total + depth_tile_size - 1) / depth_tile_size;
    nk_size_t const total_tiles = row_tile_count * depth_tile_count;
    nk_size_t const data_size = total_tiles * tile_elements * sizeof(nk_u32_t);

    nk_sets_smebi32_packed_header_t *header = (nk_sets_smebi32_packed_header_t *)b_packed;
    header->row_tile_count = (nk_u32_t)row_tile_count;
    header->depth_tile_count = (nk_u32_t)depth_tile_count;
    header->rows = (nk_u32_t)row_count;
    header->depth_bits = (nk_u32_t)depth_bits;
    header->svl_bytes = (nk_u32_t)svl_bytes;
    header->norms_offset = (nk_u32_t)(sizeof(nk_sets_smebi32_packed_header_t) + data_size);

    nk_u32_t *tiles_ptr = (nk_u32_t *)((char *)b_packed + sizeof(nk_sets_smebi32_packed_header_t));
    nk_u32_t *norms_ptr = (nk_u32_t *)((char *)b_packed + header->norms_offset);

    // Zero-initialize all tiles
    for (nk_size_t i = 0; i < total_tiles * tile_elements; i++) tiles_ptr[i] = 0;

    // Pack tiles: column-major u32 within each tile
    for (nk_size_t row_tile = 0; row_tile < row_tile_count; row_tile++) {
        for (nk_size_t depth_tile = 0; depth_tile < depth_tile_count; depth_tile++) {
            nk_size_t const tile_index = row_tile * depth_tile_count + depth_tile;
            nk_u32_t *tile_output = tiles_ptr + tile_index * tile_elements;

            nk_size_t const src_row_start = row_tile * tile_dim;
            nk_size_t const src_u32_start = depth_tile * depth_tile_size;
            nk_size_t const rows_to_pack = (src_row_start + tile_dim <= row_count) ? tile_dim
                                                                                   : (row_count - src_row_start);
            nk_size_t const u32s_to_pack = (src_u32_start + depth_tile_size <= depth_u32_total)
                                               ? depth_tile_size
                                               : (depth_u32_total > src_u32_start ? depth_u32_total - src_u32_start
                                                                                  : 0);

            // Column-major packing
            for (nk_size_t row = 0; row < rows_to_pack; row++) {
                nk_u32_t const *src_row = (nk_u32_t const *)((char const *)b +
                                                             (src_row_start + row) * b_stride_in_bytes);
                for (nk_size_t col = 0; col < u32s_to_pack; col++) {
                    nk_size_t const dst_idx = col * tile_dim + row;
                    tile_output[dst_idx] = src_row[src_u32_start + col];
                }
            }
        }
    }

    // Compute norms
    for (nk_size_t row = 0; row < row_count; row++) {
        nk_u1x8_t const *src_row = (nk_u1x8_t const *)((char const *)b + row * b_stride_in_bytes);
        norms_ptr[row] = nk_popcount_vector_serial_(src_row, depth_in_bytes);
    }
}

/**
 *  SME Hamming kernel using BMOPA instruction for efficient pairwise popcount.
 *  Both A and B must be packed in column-major u32 format.
 *
 *  BMOPA computes: ZA[i,j] += popcount(~(Zn[i] ^ Zm[j])) = popcount(XNOR)
 *  This counts matching bits. Hamming distance = depth_bits - ZA[i,j].
 *
 *  For each 16×16 output tile:
 *  - Zero ZA accumulator
 *  - For each depth position (u32): BMOPA with 16 A values × 16 B values
 *  - Store: Hamming[i,j] = depth_bits - ZA[i,j]
 */
__arm_locally_streaming __arm_new("za") static void nk_hammings_u1_smebi32_kernel_packed_(
    void const *a_packed, void const *b_packed, nk_u32_t *c, nk_size_t row_count_a, nk_size_t row_count_b,
    nk_size_t depth_bits, nk_size_t c_stride_in_bytes) {

    nk_sets_smebi32_packed_header_t const *header_a = (nk_sets_smebi32_packed_header_t const *)a_packed;
    nk_sets_smebi32_packed_header_t const *header_b = (nk_sets_smebi32_packed_header_t const *)b_packed;

    nk_size_t const row_tile_count_a = header_a->row_tile_count;
    nk_size_t const row_tile_count_b = header_b->row_tile_count;
    nk_size_t const depth_tile_count = header_a->depth_tile_count;

    nk_size_t const tile_dim = svcntw();        // 16 for 512-bit SVL
    nk_size_t const depth_tile_size = svcntw(); // 16 u32 per depth tile
    nk_size_t const tile_elements = tile_dim * depth_tile_size;
    nk_size_t const depth_u32_total = (depth_bits + 31) / 32;

    nk_u32_t const *a_tiles = (nk_u32_t const *)((char const *)a_packed + sizeof(nk_sets_smebi32_packed_header_t));
    nk_u32_t const *b_tiles = (nk_u32_t const *)((char const *)b_packed + sizeof(nk_sets_smebi32_packed_header_t));

    svbool_t const ptrue_s = svptrue_b32();
    svuint32_t const depth_vec = svdup_u32((nk_u32_t)depth_bits);

    // Process tile-by-tile with predicates for edge handling
    for (nk_size_t row_tile_a = 0; row_tile_a < row_tile_count_a; row_tile_a++) {
        nk_size_t const row_start_a = row_tile_a * tile_dim;
        nk_size_t const rows_a_remaining = (row_start_a + tile_dim <= row_count_a) ? tile_dim
                                                                                   : (row_count_a - row_start_a);
        svbool_t const pred_rows = svwhilelt_b32((nk_u32_t)0, (nk_u32_t)rows_a_remaining);

        // Process 4 B-column tiles in parallel using ZA tiles 0-3
        // This amortizes A loads and extraction overhead across 4× more output
        nk_size_t row_tile_b = 0;
        for (; row_tile_b + 4 <= row_tile_count_b; row_tile_b += 4) {
            svzero_za(); // Zeros all ZA tiles

            // Accumulate into 4 ZA tiles simultaneously
            for (nk_size_t d_tile = 0; d_tile < depth_tile_count; d_tile++) {
                nk_size_t const d_start = d_tile * depth_tile_size;
                nk_size_t const a_tile_idx = row_tile_a * depth_tile_count + d_tile;
                nk_u32_t const *a_tile = a_tiles + a_tile_idx * tile_elements;

                // B tile pointers for 4 column tiles
                nk_u32_t const *b_tile0 = b_tiles + ((row_tile_b + 0) * depth_tile_count + d_tile) * tile_elements;
                nk_u32_t const *b_tile1 = b_tiles + ((row_tile_b + 1) * depth_tile_count + d_tile) * tile_elements;
                nk_u32_t const *b_tile2 = b_tiles + ((row_tile_b + 2) * depth_tile_count + d_tile) * tile_elements;
                nk_u32_t const *b_tile3 = b_tiles + ((row_tile_b + 3) * depth_tile_count + d_tile) * tile_elements;

                for (nk_size_t depth_u32 = 0; depth_u32 < depth_tile_size; depth_u32++) {
                    nk_size_t const abs_depth_u32 = d_start + depth_u32;
                    if (abs_depth_u32 >= depth_u32_total) break;

                    // Load A once, apply to 4 B tiles
                    svuint32_t a_vec = svld1_u32(ptrue_s, a_tile + depth_u32 * tile_dim);
                    svbmopa_za32_u32_m(0, pred_rows, ptrue_s, a_vec,
                                       svld1_u32(ptrue_s, b_tile0 + depth_u32 * tile_dim));
                    svbmopa_za32_u32_m(1, pred_rows, ptrue_s, a_vec,
                                       svld1_u32(ptrue_s, b_tile1 + depth_u32 * tile_dim));
                    svbmopa_za32_u32_m(2, pred_rows, ptrue_s, a_vec,
                                       svld1_u32(ptrue_s, b_tile2 + depth_u32 * tile_dim));
                    svbmopa_za32_u32_m(3, pred_rows, ptrue_s, a_vec,
                                       svld1_u32(ptrue_s, b_tile3 + depth_u32 * tile_dim));
                }
            }

            // Extract all 4 ZA tiles: Hamming = depth_bits - matching_bits
            for (nk_size_t row = 0; row < rows_a_remaining; row++) {
                nk_u32_t *c_row = (nk_u32_t *)((char *)c + (row_start_a + row) * c_stride_in_bytes);

                svuint32_t za0 = svread_hor_za32_u32_m(svdup_u32(0), ptrue_s, 0, row);
                svuint32_t za1 = svread_hor_za32_u32_m(svdup_u32(0), ptrue_s, 1, row);
                svuint32_t za2 = svread_hor_za32_u32_m(svdup_u32(0), ptrue_s, 2, row);
                svuint32_t za3 = svread_hor_za32_u32_m(svdup_u32(0), ptrue_s, 3, row);

                svst1_u32(ptrue_s, c_row + (row_tile_b + 0) * tile_dim, svsub_u32_x(ptrue_s, depth_vec, za0));
                svst1_u32(ptrue_s, c_row + (row_tile_b + 1) * tile_dim, svsub_u32_x(ptrue_s, depth_vec, za1));
                svst1_u32(ptrue_s, c_row + (row_tile_b + 2) * tile_dim, svsub_u32_x(ptrue_s, depth_vec, za2));
                svst1_u32(ptrue_s, c_row + (row_tile_b + 3) * tile_dim, svsub_u32_x(ptrue_s, depth_vec, za3));
            }
        }

        // Handle remaining B-column tiles (0-3 tiles) with single-tile processing
        for (; row_tile_b < row_tile_count_b; row_tile_b++) {
            nk_size_t const row_start_b = row_tile_b * tile_dim;
            nk_size_t const rows_b_remaining = (row_start_b + tile_dim <= row_count_b) ? tile_dim
                                                                                       : (row_count_b - row_start_b);
            svbool_t const pred_cols = svwhilelt_b32((nk_u32_t)0, (nk_u32_t)rows_b_remaining);

            svzero_za();

            for (nk_size_t d_tile = 0; d_tile < depth_tile_count; d_tile++) {
                nk_size_t const d_start = d_tile * depth_tile_size;
                nk_size_t const a_tile_idx = row_tile_a * depth_tile_count + d_tile;
                nk_size_t const b_tile_idx = row_tile_b * depth_tile_count + d_tile;
                nk_u32_t const *a_tile = a_tiles + a_tile_idx * tile_elements;
                nk_u32_t const *b_tile = b_tiles + b_tile_idx * tile_elements;

                for (nk_size_t depth_u32 = 0; depth_u32 < depth_tile_size; depth_u32++) {
                    nk_size_t const abs_depth_u32 = d_start + depth_u32;
                    if (abs_depth_u32 >= depth_u32_total) break;

                    svuint32_t a_vec = svld1_u32(ptrue_s, a_tile + depth_u32 * tile_dim);
                    svuint32_t b_vec = svld1_u32(ptrue_s, b_tile + depth_u32 * tile_dim);
                    svbmopa_za32_u32_m(0, pred_rows, pred_cols, a_vec, b_vec);
                }
            }

            for (nk_size_t row = 0; row < rows_a_remaining; row++) {
                svuint32_t za_row = svread_hor_za32_u32_m(svdup_u32(0), ptrue_s, 0, row);
                svuint32_t hamming = svsub_u32_x(ptrue_s, depth_vec, za_row);
                nk_u32_t *c_row = (nk_u32_t *)((char *)c + (row_start_a + row) * c_stride_in_bytes);
                svst1_u32(pred_cols, c_row + row_start_b, hamming);
            }
        }
    }
}

/**
 *  SME Hamming kernel wrapper that packs A on-the-fly.
 *  This version is for the public API that takes unpacked A.
 *  For performance-critical code, use the symmetric variant or pre-pack A.
 */
__arm_locally_streaming __arm_new("za") static void nk_hammings_u1_smebi32_kernel_(
    nk_u1x8_t const *a, void const *b_packed, nk_u32_t *c, nk_size_t row_count_a, nk_size_t row_count_b,
    nk_size_t depth_bits, nk_size_t a_stride_in_bytes, nk_size_t c_stride_in_bytes) {

    // Fall back to scalar-accumulator approach for unpacked A
    // (BMOPA requires column-major packed format for both operands)
    nk_sets_smebi32_packed_header_t const *header = (nk_sets_smebi32_packed_header_t const *)b_packed;
    nk_size_t const row_tile_count_b = header->row_tile_count;
    nk_size_t const depth_tile_count = header->depth_tile_count;

    nk_size_t const tile_dim = svcntw();
    nk_size_t const depth_tile_size = svcntw();
    nk_size_t const tile_elements = tile_dim * depth_tile_size;
    nk_size_t const depth_u32_total = (depth_bits + 31) / 32;

    nk_u32_t const *b_tiles = (nk_u32_t const *)((char const *)b_packed + sizeof(nk_sets_smebi32_packed_header_t));

    svbool_t const ptrue_b = svptrue_b8();
    nk_size_t const row_tile_count_a = (row_count_a + tile_dim - 1) / tile_dim;
    nk_size_t const depth_tile_bytes = tile_dim * sizeof(nk_u32_t); // 64 bytes per depth slice

    for (nk_size_t row_tile_a = 0; row_tile_a < row_tile_count_a; row_tile_a++) {
        nk_size_t const row_start_a = row_tile_a * tile_dim;
        nk_size_t const rows_a_remaining = (row_start_a + tile_dim <= row_count_a) ? tile_dim
                                                                                   : (row_count_a - row_start_a);

        for (nk_size_t row_tile_b = 0; row_tile_b < row_tile_count_b; row_tile_b++) {
            nk_size_t const row_start_b = row_tile_b * tile_dim;
            nk_size_t const rows_b_remaining = (row_start_b + tile_dim <= row_count_b) ? tile_dim
                                                                                       : (row_count_b - row_start_b);

            nk_u32_t acc[16][16] = {{0}};

            for (nk_size_t d_tile = 0; d_tile < depth_tile_count; d_tile++) {
                nk_size_t const d_start_bytes = d_tile * depth_tile_bytes;
                nk_size_t const d_start_u32 = d_tile * depth_tile_size;

                nk_size_t const b_tile_idx = row_tile_b * depth_tile_count + d_tile;
                nk_u32_t const *b_tile = b_tiles + b_tile_idx * tile_elements;

                for (nk_size_t row_a = 0; row_a < rows_a_remaining; row_a++) {
                    nk_u1x8_t const *a_ptr = (nk_u1x8_t const *)((char const *)a +
                                                                 (row_start_a + row_a) * a_stride_in_bytes) +
                                             d_start_bytes;

                    svuint8_t a_vec = svld1_u8(ptrue_b, a_ptr);

                    // B is now packed column-major as u32, need to access it properly
                    for (nk_size_t row_b = 0; row_b < rows_b_remaining; row_b++) {
                        // Reconstruct B row from column-major packed data
                        nk_u32_t b_row_data[16];
                        for (nk_size_t d = 0; d < depth_tile_size && (d_start_u32 + d) < depth_u32_total; d++) {
                            b_row_data[d] = b_tile[d * tile_dim + row_b];
                        }
                        nk_u1x8_t const *b_ptr = (nk_u1x8_t const *)b_row_data;
                        svuint8_t b_vec = svld1_u8(ptrue_b, b_ptr);

                        svuint8_t xor_vec = sveor_u8_z(ptrue_b, a_vec, b_vec);
                        svuint8_t cnt_vec = svcnt_u8_z(ptrue_b, xor_vec);
                        acc[row_a][row_b] += (nk_u32_t)svaddv_u8(ptrue_b, cnt_vec);
                    }
                }
            }

            for (nk_size_t row_a = 0; row_a < rows_a_remaining; row_a++) {
                nk_u32_t *c_row = (nk_u32_t *)((char *)c + (row_start_a + row_a) * c_stride_in_bytes);
                for (nk_size_t row_b = 0; row_b < rows_b_remaining; row_b++) {
                    c_row[row_start_b + row_b] = acc[row_a][row_b];
                }
            }
        }
    }
}

/**
 *  SME Jaccard kernel using streaming mode SVE operations.
 *  Works with column-major u32 packed B matrix.
 */
__arm_locally_streaming __arm_new("za") static void nk_jaccards_u1_smebi32_kernel_(
    nk_u1x8_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t row_count_a, nk_size_t row_count_b,
    nk_size_t depth_bits, nk_size_t a_stride_in_bytes, nk_size_t c_stride_in_bytes, nk_f32_t const *a_norms) {

    nk_sets_smebi32_packed_header_t const *header = (nk_sets_smebi32_packed_header_t const *)b_packed;
    nk_size_t const row_tile_count_b = header->row_tile_count;
    nk_size_t const depth_tile_count = header->depth_tile_count;

    nk_size_t const tile_dim = svcntw();
    nk_size_t const depth_tile_size = svcntw();
    nk_size_t const tile_elements = tile_dim * depth_tile_size;
    nk_size_t const depth_u32_total = (depth_bits + 31) / 32;
    nk_size_t const depth_tile_bytes = tile_dim * sizeof(nk_u32_t);

    nk_u32_t const *b_tiles = (nk_u32_t const *)((char const *)b_packed + sizeof(nk_sets_smebi32_packed_header_t));
    nk_u32_t const *b_norms = header->norms_offset ? (nk_u32_t const *)((char const *)b_packed + header->norms_offset)
                                                   : (nk_u32_t const *)0;

    svbool_t const ptrue_b = svptrue_b8();
    nk_size_t const row_tile_count_a = (row_count_a + tile_dim - 1) / tile_dim;

    for (nk_size_t row_tile_a = 0; row_tile_a < row_tile_count_a; row_tile_a++) {
        nk_size_t const row_start_a = row_tile_a * tile_dim;
        nk_size_t const rows_a_remaining = (row_start_a + tile_dim <= row_count_a) ? tile_dim
                                                                                   : (row_count_a - row_start_a);

        for (nk_size_t row_tile_b = 0; row_tile_b < row_tile_count_b; row_tile_b++) {
            nk_size_t const row_start_b = row_tile_b * tile_dim;
            nk_size_t const rows_b_remaining = (row_start_b + tile_dim <= row_count_b) ? tile_dim
                                                                                       : (row_count_b - row_start_b);

            nk_u32_t acc[16][16] = {{0}};

            for (nk_size_t d_tile = 0; d_tile < depth_tile_count; d_tile++) {
                nk_size_t const d_start_bytes = d_tile * depth_tile_bytes;
                nk_size_t const d_start_u32 = d_tile * depth_tile_size;

                nk_size_t const b_tile_idx = row_tile_b * depth_tile_count + d_tile;
                nk_u32_t const *b_tile = b_tiles + b_tile_idx * tile_elements;

                for (nk_size_t row_a = 0; row_a < rows_a_remaining; row_a++) {
                    nk_u1x8_t const *a_ptr = (nk_u1x8_t const *)((char const *)a +
                                                                 (row_start_a + row_a) * a_stride_in_bytes) +
                                             d_start_bytes;

                    svuint8_t a_vec = svld1_u8(ptrue_b, a_ptr);

                    // B is packed column-major as u32, reconstruct B rows
                    for (nk_size_t row_b = 0; row_b < rows_b_remaining; row_b++) {
                        nk_u32_t b_row_data[16];
                        for (nk_size_t d = 0; d < depth_tile_size && (d_start_u32 + d) < depth_u32_total; d++) {
                            b_row_data[d] = b_tile[d * tile_dim + row_b];
                        }
                        nk_u1x8_t const *b_ptr = (nk_u1x8_t const *)b_row_data;
                        svuint8_t b_vec = svld1_u8(ptrue_b, b_ptr);

                        svuint8_t and_vec = svand_u8_z(ptrue_b, a_vec, b_vec);
                        svuint8_t cnt_vec = svcnt_u8_z(ptrue_b, and_vec);
                        acc[row_a][row_b] += (nk_u32_t)svaddv_u8(ptrue_b, cnt_vec);
                    }
                }
            }

            for (nk_size_t row_a = 0; row_a < rows_a_remaining; row_a++) {
                nk_f32_t norm_a = a_norms ? a_norms[row_start_a + row_a] : 0;
                nk_f32_t *c_row = (nk_f32_t *)((char *)c + (row_start_a + row_a) * c_stride_in_bytes);

                for (nk_size_t row_b = 0; row_b < rows_b_remaining; row_b++) {
                    nk_f32_t intersection = (nk_f32_t)acc[row_a][row_b];
                    nk_f32_t norm_b = b_norms ? (nk_f32_t)b_norms[row_start_b + row_b] : 0;
                    nk_f32_t union_val = norm_a + norm_b - intersection;
                    c_row[row_start_b + row_b] = (union_val != 0) ? 1.0f - intersection / union_val : 1.0f;
                }
            }
        }
    }
}

NK_PUBLIC void nk_hammings_packed_u1_smebi32(nk_u1x8_t const *a, void const *b_packed, nk_u32_t *c,
                                             nk_size_t row_count_a, nk_size_t row_count_b, nk_size_t depth_bits,
                                             nk_size_t a_stride_in_bytes, nk_size_t c_stride_in_bytes) {
    nk_hammings_u1_smebi32_kernel_(a, b_packed, c, row_count_a, row_count_b, depth_bits, a_stride_in_bytes,
                                   c_stride_in_bytes);
}

NK_PUBLIC void nk_jaccards_packed_u1_smebi32(nk_u1x8_t const *a, void const *b_packed, nk_f32_t *c,
                                             nk_size_t row_count_a, nk_size_t row_count_b, nk_size_t depth_bits,
                                             nk_size_t a_stride_in_bytes, nk_size_t c_stride_in_bytes,
                                             nk_f32_t const *a_norms) {
    nk_jaccards_u1_smebi32_kernel_(a, b_packed, c, row_count_a, row_count_b, depth_bits, a_stride_in_bytes,
                                   c_stride_in_bytes, a_norms);
}

/**
 *  Symmetric Hamming distance using BMOPA.
 *  Packs the input matrix once and uses it for both BMOPA operands.
 *  This is the most efficient path for pairwise distance computation.
 */
NK_PUBLIC void nk_hammings_symmetric_u1_smebi32(nk_u1x8_t const *vectors, nk_size_t n_vectors, nk_size_t depth_bits,
                                                nk_size_t stride, nk_u32_t *result, nk_size_t result_stride,
                                                nk_size_t row_start, nk_size_t row_count) {
    // Allocate packed buffer
    nk_size_t const packed_size = nk_hammings_packed_size_u1_smebi32(n_vectors, depth_bits);
    void *packed = NK_ALIGNED_ALLOC_(64, packed_size);
    if (!packed) {
        nk_hammings_symmetric_u1_serial(vectors, n_vectors, depth_bits, stride, result, result_stride, row_start,
                                        row_count);
        return;
    }

    // Pack the matrix in column-major u32 format
    nk_hammings_pack_u1_smebi32(vectors, n_vectors, depth_bits, stride, packed);

    // Compute pairwise distances using BMOPA (packed A == packed B)
    // Process the specified row range
    nk_u1x8_t const *a_start = (nk_u1x8_t const *)((char const *)vectors + row_start * stride);
    nk_u32_t *c_start = (nk_u32_t *)((char *)result + row_start * result_stride);

    // For the row range, we need to pack just those rows as A
    nk_size_t const a_packed_size = nk_hammings_packed_size_u1_smebi32(row_count, depth_bits);
    void *a_packed = NK_ALIGNED_ALLOC_(64, a_packed_size);
    if (!a_packed) {
        NK_ALIGNED_FREE_(packed);
        nk_hammings_symmetric_u1_serial(vectors, n_vectors, depth_bits, stride, result, result_stride, row_start,
                                        row_count);
        return;
    }

    nk_hammings_pack_u1_smebi32(a_start, row_count, depth_bits, stride, a_packed);

    // Use BMOPA kernel with packed A (row subset) and packed B (all vectors)
    nk_hammings_u1_smebi32_kernel_packed_(a_packed, packed, c_start, row_count, n_vectors, depth_bits, result_stride);

    NK_ALIGNED_FREE_(a_packed);
    NK_ALIGNED_FREE_(packed);
}

NK_PUBLIC void nk_jaccards_symmetric_u1_smebi32(nk_u1x8_t const *vectors, nk_size_t n_vectors, nk_size_t depth_bits,
                                                nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                nk_size_t row_start, nk_size_t row_count) {
    nk_jaccards_symmetric_u1_serial(vectors, n_vectors, depth_bits, stride, result, result_stride, row_start,
                                    row_count);
}

#if defined(__cplusplus)
} // extern "C"
#endif

#pragma clang attribute pop
#pragma GCC pop_options
#endif // NK_TARGET_SMEBI32
#endif // NK_TARGET_ARM_

#endif // NK_SETS_SMEBI32_H

/**
 *  @brief SIMD-accelerated batched binary set computations for ARM SME.
 *  @file include/numkong/sets/smebi32.h
 *  @author Ash Vardanian
 *  @date February 6, 2026
 *  @sa include/numkong/sets.h
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

#include "numkong/types.h"
#include "numkong/set/serial.h" // `nk_u1x8_popcount_`
#include "numkong/sets/serial.h"

#if defined(__cplusplus)
extern "C" {
#endif

/** Count total set bits across a byte vector of given length. */
NK_INTERNAL nk_f32_t nk_popcount_vector_serial_(nk_u1x8_t const *data, nk_size_t n_bytes) {
    nk_u32_t count = 0;
    for (nk_size_t i = 0; i < n_bytes; ++i) count += nk_u1x8_popcount_(data[i]);
    return (nk_f32_t)count;
}

/*
 *  Binary set operations using SME BMOPA instruction.
 *
 *  BMOPA computes: ZA[i,j] += popcount(~(Zn[i] ^ Zm[j])) = popcount(XNOR)
 *  This counts matching bits. Hamming = depth_bits - matching.
 *
 *  Tile layout (SVL=512, Apple M4):
 *  - ZA32 output tile: 16 × 16 u32 elements (1 KB)
 *  - Input vectors: 16 u32 elements (SVL/32)
 *  - Each BMOPA processes 32 bits (one u32) across 16×16 pairs
 *  - BMOPA predicates: b32 (u32 input granularity)
 *  - 4-tile path: ZA0-ZA3 process 4 B-column tiles simultaneously
 *  - Packed format: column-major u32 within each tile
 */

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("sme"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("+sme")
#endif

/*  Get ZA32 tile dimension (number of f32/u32 elements per row). */
NK_INTERNAL nk_size_t nk_smebi32_tile_dim_(void) __arm_streaming_compatible { return svcntw(); }

/*  Get SVL in bytes for validation. */
NK_INTERNAL nk_size_t nk_smebi32_svl_bytes_(void) __arm_streaming_compatible { return svcntw() * sizeof(nk_u32_t); }

typedef struct {
    nk_u32_t row_tile_count;   // ceiling(rows / tile_dim)
    nk_u32_t depth_tile_count; // ceiling(depth_bits / depth_tile_bits)
    nk_u32_t rows;             // actual row count
    nk_u32_t depth_bits;       // actual depth in bits
    nk_u32_t svl_bytes;        // SVL at pack time for validation
    nk_u32_t norms_offset;     // byte offset to norms (0 if none)
    nk_u32_t reserved[10];     // padding to 64 bytes
} nk_sets_smebi32_packed_header_t;

#pragma region Hamming Distance

NK_PUBLIC nk_size_t nk_hammings_packed_size_u1_smebi32(nk_size_t row_count, nk_size_t depth_bits) {
    nk_size_t const tile_dim = nk_smebi32_tile_dim_();        // 16 rows per tile
    nk_size_t const depth_tile_size = nk_smebi32_tile_dim_(); // 16 u32 per depth tile = 512 bits

    nk_size_t const depth_u32 = (depth_bits + 31) / 32;
    nk_size_t const row_tile_count = (row_count + tile_dim - 1) / tile_dim;
    nk_size_t const depth_tile_count = (depth_u32 + depth_tile_size - 1) / depth_tile_size;

    nk_size_t const tile_elements = tile_dim * depth_tile_size; // 256 u32 per tile
    nk_size_t size = sizeof(nk_sets_smebi32_packed_header_t);
    size += row_tile_count * depth_tile_count * tile_elements * sizeof(nk_u32_t);

    return size;
}

NK_PUBLIC void nk_hammings_pack_u1_smebi32(nk_u1x8_t const *b, nk_size_t row_count, nk_size_t depth_bits,
                                           nk_size_t b_stride_in_bytes, void *b_packed) {
    nk_size_t const svl_bytes = nk_smebi32_svl_bytes_();
    nk_size_t const tile_dim = nk_smebi32_tile_dim_();        // 16 rows per tile
    nk_size_t const depth_tile_size = nk_smebi32_tile_dim_(); // 16 u32 per depth tile
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
static
    __attribute__((target("sme2"))) __arm_locally_streaming __arm_new("za") void nk_hammings_u1_smebi32_kernel_packed_(
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

    svbool_t const predicate_single = svptrue_b32();
    svuint32_t const depth_vec = svdup_u32((nk_u32_t)depth_bits);

    // Process tile-by-tile with predicates for edge handling
    for (nk_size_t row_tile_a = 0; row_tile_a < row_tile_count_a; row_tile_a++) {
        nk_size_t const row_start_a = row_tile_a * tile_dim;
        nk_size_t const rows_a_remaining = (row_start_a + tile_dim <= row_count_a) ? tile_dim
                                                                                   : (row_count_a - row_start_a);
        svbool_t const predicate_valid_rows = svwhilelt_b32((nk_u32_t)0, (nk_u32_t)rows_a_remaining);

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
                    svuint32_t a_vec = svld1_u32(predicate_single, a_tile + depth_u32 * tile_dim);
                    svbmopa_za32_u32_m(0, predicate_valid_rows, predicate_single, a_vec,
                                       svld1_u32(predicate_single, b_tile0 + depth_u32 * tile_dim));
                    svbmopa_za32_u32_m(1, predicate_valid_rows, predicate_single, a_vec,
                                       svld1_u32(predicate_single, b_tile1 + depth_u32 * tile_dim));
                    svbmopa_za32_u32_m(2, predicate_valid_rows, predicate_single, a_vec,
                                       svld1_u32(predicate_single, b_tile2 + depth_u32 * tile_dim));
                    svbmopa_za32_u32_m(3, predicate_valid_rows, predicate_single, a_vec,
                                       svld1_u32(predicate_single, b_tile3 + depth_u32 * tile_dim));
                }
            }

            // Extract all 4 ZA tiles: Hamming = depth_bits - matching_bits
            for (nk_size_t row = 0; row < rows_a_remaining; row++) {
                nk_u32_t *c_row = (nk_u32_t *)((char *)c + (row_start_a + row) * c_stride_in_bytes);

                svuint32_t za0 = svread_hor_za32_u32_m(svdup_u32(0), predicate_single, 0, row);
                svuint32_t za1 = svread_hor_za32_u32_m(svdup_u32(0), predicate_single, 1, row);
                svuint32_t za2 = svread_hor_za32_u32_m(svdup_u32(0), predicate_single, 2, row);
                svuint32_t za3 = svread_hor_za32_u32_m(svdup_u32(0), predicate_single, 3, row);

                svst1_u32(predicate_single, c_row + (row_tile_b + 0) * tile_dim,
                          svsub_u32_x(predicate_single, depth_vec, za0));
                svst1_u32(predicate_single, c_row + (row_tile_b + 1) * tile_dim,
                          svsub_u32_x(predicate_single, depth_vec, za1));
                svst1_u32(predicate_single, c_row + (row_tile_b + 2) * tile_dim,
                          svsub_u32_x(predicate_single, depth_vec, za2));
                svst1_u32(predicate_single, c_row + (row_tile_b + 3) * tile_dim,
                          svsub_u32_x(predicate_single, depth_vec, za3));
            }
        }

        // Handle remaining B-column tiles (0-3 tiles) with single-tile processing
        for (; row_tile_b < row_tile_count_b; row_tile_b++) {
            nk_size_t const row_start_b = row_tile_b * tile_dim;
            nk_size_t const rows_b_remaining = (row_start_b + tile_dim <= row_count_b) ? tile_dim
                                                                                       : (row_count_b - row_start_b);
            svbool_t const predicate_valid_columns = svwhilelt_b32((nk_u32_t)0, (nk_u32_t)rows_b_remaining);

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

                    svuint32_t a_vec = svld1_u32(predicate_single, a_tile + depth_u32 * tile_dim);
                    svuint32_t b_vec = svld1_u32(predicate_single, b_tile + depth_u32 * tile_dim);
                    svbmopa_za32_u32_m(0, predicate_valid_rows, predicate_valid_columns, a_vec, b_vec);
                }
            }

            for (nk_size_t row = 0; row < rows_a_remaining; row++) {
                svuint32_t za_row = svread_hor_za32_u32_m(svdup_u32(0), predicate_single, 0, row);
                svuint32_t hamming = svsub_u32_x(predicate_single, depth_vec, za_row);
                nk_u32_t *c_row = (nk_u32_t *)((char *)c + (row_start_a + row) * c_stride_in_bytes);
                svst1_u32(predicate_valid_columns, c_row + row_start_b, hamming);
            }
        }
    }
}

/**
 *  SME Hamming kernel wrapper that packs A on-the-fly.
 *  This version is for the public API that takes unpacked A.
 *  For performance-critical code, use the symmetric variant or pre-pack A.
 */
static __attribute__((target("sme2"))) __arm_locally_streaming __arm_new("za") void nk_hammings_u1_smebi32_kernel_(
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

    svbool_t const predicate_byte = svptrue_b8();
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

                    svuint8_t a_vec = svld1_u8(predicate_byte, a_ptr);

                    // B is now packed column-major as u32, need to access it properly
                    for (nk_size_t row_b = 0; row_b < rows_b_remaining; row_b++) {
                        // Reconstruct B row from column-major packed data
                        nk_u32_t b_row_data[16];
                        for (nk_size_t d = 0; d < depth_tile_size && (d_start_u32 + d) < depth_u32_total; d++) {
                            b_row_data[d] = b_tile[d * tile_dim + row_b];
                        }
                        nk_u1x8_t const *b_ptr = (nk_u1x8_t const *)b_row_data;
                        svuint8_t b_vec = svld1_u8(predicate_byte, b_ptr);

                        svuint8_t xor_vec = sveor_u8_z(predicate_byte, a_vec, b_vec);
                        svuint8_t cnt_vec = svcnt_u8_z(predicate_byte, xor_vec);
                        acc[row_a][row_b] += (nk_u32_t)svaddv_u8(predicate_byte, cnt_vec);
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

NK_PUBLIC void nk_hammings_packed_u1_smebi32(nk_u1x8_t const *a, void const *b_packed, nk_u32_t *c,
                                             nk_size_t row_count_a, nk_size_t row_count_b, nk_size_t depth_bits,
                                             nk_size_t a_stride_in_bytes, nk_size_t c_stride_in_bytes) {
    nk_hammings_u1_smebi32_kernel_(a, b_packed, c, row_count_a, row_count_b, depth_bits, a_stride_in_bytes,
                                   c_stride_in_bytes);
}

/* Symmetric Hamming using stack-allocated tile buffers (no heap allocation). */
NK_PUBLIC __attribute__((target("sme2"))) __arm_locally_streaming __arm_new("za") void nk_hammings_symmetric_u1_smebi32(
    nk_u1x8_t const *vectors, nk_size_t n_vectors, nk_size_t depth_bits, nk_size_t stride, nk_u32_t *result,
    nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count) {

    nk_size_t const tile_dim = svcntw();        // 16 for 512-bit SVL
    nk_size_t const depth_tile_size = svcntw(); // 16 u32 per depth tile
    nk_size_t const depth_u32_total = (depth_bits + 31) / 32;
    nk_size_t const depth_tile_count = (depth_u32_total + depth_tile_size - 1) / depth_tile_size;
    nk_size_t const tile_elements = tile_dim * depth_tile_size; // 256 u32

    // Stack-allocated column-major tile buffers (~1KB each, ~2KB total)
    nk_u32_t a_tile_buf[256]; // tile_dim * depth_tile_size
    nk_u32_t b_tile_buf[256];

    svbool_t const pg = svptrue_b32();
    svuint32_t const depth_vec = svdup_u32((nk_u32_t)depth_bits);

    nk_size_t const row_end = row_start + row_count;
    nk_size_t const row_tile_count_a = (row_count + tile_dim - 1) / tile_dim;
    nk_size_t const col_tile_count = (n_vectors + tile_dim - 1) / tile_dim;

    for (nk_size_t rta = 0; rta < row_tile_count_a; rta++) {
        nk_size_t const abs_row_a = row_start + rta * tile_dim;
        nk_size_t const rows_a = (abs_row_a + tile_dim <= row_end) ? tile_dim : (row_end - abs_row_a);
        svbool_t const pred_rows_a = svwhilelt_b32((nk_u32_t)0, (nk_u32_t)rows_a);

        // Only compute column tiles from abs_row_a onward (symmetric: skip j < i)
        nk_size_t const first_col_tile = abs_row_a / tile_dim;

        for (nk_size_t ctb = first_col_tile; ctb < col_tile_count; ctb++) {
            nk_size_t const abs_col_b = ctb * tile_dim;
            nk_size_t const cols_b = (abs_col_b + tile_dim <= n_vectors) ? tile_dim : (n_vectors - abs_col_b);
            svbool_t const pred_cols_b = svwhilelt_b32((nk_u32_t)0, (nk_u32_t)cols_b);

            svzero_za();

            // Accumulate BMOPA over all depth tiles
            for (nk_size_t dt = 0; dt < depth_tile_count; dt++) {
                nk_size_t const d_start_u32 = dt * depth_tile_size;
                nk_size_t const u32s_this_tile = (d_start_u32 + depth_tile_size <= depth_u32_total)
                                                     ? depth_tile_size
                                                     : (depth_u32_total > d_start_u32 ? depth_u32_total - d_start_u32
                                                                                      : 0);

                // Pack A tile: strided rows → column-major u32
                for (nk_size_t i = 0; i < tile_elements; i++) a_tile_buf[i] = 0;
                for (nk_size_t r = 0; r < rows_a; r++) {
                    nk_u32_t const *src = (nk_u32_t const *)((char const *)vectors + (abs_row_a + r) * stride) +
                                          d_start_u32;
                    for (nk_size_t c = 0; c < u32s_this_tile; c++) a_tile_buf[c * tile_dim + r] = src[c];
                }

                // Pack B tile: strided rows → column-major u32
                for (nk_size_t i = 0; i < tile_elements; i++) b_tile_buf[i] = 0;
                for (nk_size_t r = 0; r < cols_b; r++) {
                    nk_u32_t const *src = (nk_u32_t const *)((char const *)vectors + (abs_col_b + r) * stride) +
                                          d_start_u32;
                    for (nk_size_t c = 0; c < u32s_this_tile; c++) b_tile_buf[c * tile_dim + r] = src[c];
                }

                // BMOPA: ZA[i,j] += popcount(XNOR(a[i], b[j]))
                for (nk_size_t d = 0; d < u32s_this_tile; d++) {
                    svuint32_t a_vec = svld1_u32(pg, a_tile_buf + d * tile_dim);
                    svuint32_t b_vec = svld1_u32(pg, b_tile_buf + d * tile_dim);
                    svbmopa_za32_u32_m(0, pred_rows_a, pred_cols_b, a_vec, b_vec);
                }
            }

            // Extract: hamming = depth_bits - ZA[i,j], write symmetric
            nk_i32_t const is_diagonal = (abs_row_a == abs_col_b);
            for (nk_size_t r = 0; r < rows_a; r++) {
                svuint32_t za_row = svread_hor_za32_u32_m(svdup_u32(0), pg, 0, r);
                svuint32_t hamming = svsub_u32_x(pg, depth_vec, za_row);

                // Write result[abs_row_a + r][abs_col_b .. abs_col_b + cols_b]
                nk_u32_t *c_row = (nk_u32_t *)((char *)result + (abs_row_a + r) * result_stride);
                svst1_u32(pred_cols_b, c_row + abs_col_b, hamming);

                // Write transposed: result[abs_col_b + c][abs_row_a + r] for off-diagonal
                if (!is_diagonal) {
                    NK_ALIGN64 nk_u32_t tmp[16];
                    svst1_u32(pred_cols_b, tmp, hamming);
                    for (nk_size_t c = 0; c < cols_b; c++) {
                        nk_u32_t *c_col = (nk_u32_t *)((char *)result + (abs_col_b + c) * result_stride);
                        c_col[abs_row_a + r] = tmp[c];
                    }
                }
            }

            // For diagonal tile, also write the transposed lower triangle
            if (is_diagonal) {
                for (nk_size_t r = 0; r < rows_a; r++) {
                    nk_u32_t *c_row = (nk_u32_t *)((char *)result + (abs_row_a + r) * result_stride);
                    for (nk_size_t c = r + 1; c < cols_b; c++) {
                        nk_u32_t *c_col = (nk_u32_t *)((char *)result + (abs_col_b + c) * result_stride);
                        c_col[abs_row_a + r] = c_row[abs_col_b + c];
                    }
                }
            }
        }
    }
}

#pragma endregion // Hamming Distance

/*
 *  Jaccard distance via bitwise intersection and union popcount.
 *
 *  intersection = popcount(a AND b)
 *  union = popcount(a) + popcount(b) - intersection
 *  jaccard = 1 - intersection / union
 *
 *  Uses streaming SVE for element-wise operations (no BMOPA).
 *  Packed format shares the Hamming tile layout for B operand.
 */

#pragma region Jaccard Distance

NK_PUBLIC nk_size_t nk_jaccards_packed_size_u1_smebi32(nk_size_t row_count, nk_size_t depth_bits) {
    nk_size_t const tile_dim = nk_smebi32_tile_dim_();        // 16 rows per tile
    nk_size_t const depth_tile_size = nk_smebi32_tile_dim_(); // 16 u32 per depth tile

    nk_size_t const depth_u32 = (depth_bits + 31) / 32;
    nk_size_t const row_tile_count = (row_count + tile_dim - 1) / tile_dim;
    nk_size_t const depth_tile_count = (depth_u32 + depth_tile_size - 1) / depth_tile_size;

    nk_size_t const tile_elements = tile_dim * depth_tile_size;
    nk_size_t size = sizeof(nk_sets_smebi32_packed_header_t);
    size += row_tile_count * depth_tile_count * tile_elements * sizeof(nk_u32_t);
    size += row_count * sizeof(nk_u32_t); // norms

    return size;
}

NK_PUBLIC void nk_jaccards_pack_u1_smebi32(nk_u1x8_t const *b, nk_size_t row_count, nk_size_t depth_bits,
                                           nk_size_t b_stride_in_bytes, void *b_packed) {
    nk_size_t const svl_bytes = nk_smebi32_svl_bytes_();
    nk_size_t const tile_dim = nk_smebi32_tile_dim_();        // 16 rows per tile
    nk_size_t const depth_tile_size = nk_smebi32_tile_dim_(); // 16 u32 per depth tile
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

/* SME Jaccard kernel using streaming mode SVE operations */
__attribute__((target("sme2"))) __arm_locally_streaming __arm_new("za") static void nk_jaccards_u1_smebi32_kernel_(
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

    svbool_t const predicate_byte = svptrue_b8();
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

                    svuint8_t a_vec = svld1_u8(predicate_byte, a_ptr);

                    // B is packed column-major as u32, reconstruct B rows
                    for (nk_size_t row_b = 0; row_b < rows_b_remaining; row_b++) {
                        nk_u32_t b_row_data[16];
                        for (nk_size_t d = 0; d < depth_tile_size && (d_start_u32 + d) < depth_u32_total; d++) {
                            b_row_data[d] = b_tile[d * tile_dim + row_b];
                        }
                        nk_u1x8_t const *b_ptr = (nk_u1x8_t const *)b_row_data;
                        svuint8_t b_vec = svld1_u8(predicate_byte, b_ptr);

                        svuint8_t and_vec = svand_u8_z(predicate_byte, a_vec, b_vec);
                        svuint8_t cnt_vec = svcnt_u8_z(predicate_byte, and_vec);
                        acc[row_a][row_b] += (nk_u32_t)svaddv_u8(predicate_byte, cnt_vec);
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

NK_PUBLIC void nk_jaccards_packed_u1_smebi32(nk_u1x8_t const *a, void const *b_packed, nk_f32_t *c,
                                             nk_size_t row_count_a, nk_size_t row_count_b, nk_size_t depth_bits,
                                             nk_size_t a_stride_in_bytes, nk_size_t c_stride_in_bytes,
                                             nk_f32_t const *a_norms) {
    nk_jaccards_u1_smebi32_kernel_(a, b_packed, c, row_count_a, row_count_b, depth_bits, a_stride_in_bytes,
                                   c_stride_in_bytes, a_norms);
}

NK_PUBLIC void nk_jaccards_symmetric_u1_smebi32(nk_u1x8_t const *vectors, nk_size_t n_vectors, nk_size_t depth_bits,
                                                nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                nk_size_t row_start, nk_size_t row_count) {
    nk_jaccards_symmetric_u1_serial(vectors, n_vectors, depth_bits, stride, result, result_stride, row_start,
                                    row_count);
}

#pragma endregion // Jaccard Distance

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_SMEBI32
#endif // NK_TARGET_ARM_

#endif // NK_SETS_SMEBI32_H

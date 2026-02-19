/**
 *  @brief SIMD-accelerated Batched Set Distances for SME.
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
#include "numkong/dots/sme.h" // `nk_sme_zero_za32_*` constants

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
 *  - Packed kernel: 4-tile path (ZA0-ZA3) for 4 B-column tiles simultaneously
 *  - Unpacked kernel: ZA transpose (ZA0.S=staging, ZA1-3.S=accumulation, 3-tile fast path)
 *  - Packed format: column-major u32 within each tile
 */

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("sme2"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("+sme2")
#endif

/*  Read SVL in bytes from non-streaming context using RDSVL instruction. */
NK_INTERNAL nk_size_t nk_smebi32_svl_bytes_(void) {
    nk_size_t svl_bytes;
    __asm__ volatile("rdsvl %0, #1" : "=r"(svl_bytes));
    return svl_bytes;
}

/*  Get ZA32 tile dimension (number of f32/u32 elements per row). */
NK_INTERNAL nk_size_t nk_smebi32_tile_dim_(void) { return nk_smebi32_svl_bytes_() / sizeof(nk_u32_t); }

typedef struct {
    nk_u32_t row_tile_count;   // ceiling(rows / tile_dim)
    nk_u32_t depth_tile_count; // ceiling(depth_bits / depth_tile_bits)
    nk_u32_t rows;             // actual row count
    nk_u32_t depth_bits;       // actual depth in bits
    nk_u32_t svl_bytes;        // SVL at pack time for validation
    nk_u32_t norms_offset;     // byte offset to norms (0 if none)
    nk_u32_t reserved[10];     // padding to 64 bytes
} nk_sets_smebi32_packed_header_t;

/** Count total set bits across a byte vector using streaming SVE.
 *  Accumulates per-byte popcounts into u32 lanes via svdot; single horizontal reduction at end. */
NK_INTERNAL nk_u32_t nk_popcount_vector_streaming_(nk_u1x8_t const *data, nk_size_t n_bytes) {
    svuint32_t acc = svdup_u32(0);
    svuint8_t const ones = svdup_u8(1);
    for (nk_size_t offset = 0; offset < n_bytes; offset += svcntb()) {
        svbool_t pred = svwhilelt_b8((nk_u32_t)offset, (nk_u32_t)n_bytes);
        acc = svdot_u32(acc, svcnt_u8_z(pred, svld1_u8(pred, data + offset)), ones);
    }
    return (nk_u32_t)svaddv_u32(svptrue_b32(), acc);
}

#pragma region Hamming Distance

NK_PUBLIC nk_size_t nk_hammings_packed_size_u1_smebi32(nk_size_t row_count, nk_size_t depth_bits) {
    nk_size_t const tile_dim = nk_smebi32_tile_dim_();        // 16 rows per tile
    nk_size_t const depth_tile_size = nk_smebi32_tile_dim_(); // 16 u32 per depth tile = 512 bits

    nk_size_t const depth_u32 = nk_size_divide_round_up_(depth_bits, 32);
    nk_size_t const row_tile_count = nk_size_divide_round_up_(row_count, tile_dim);
    nk_size_t const depth_tile_count = nk_size_divide_round_up_(depth_u32, depth_tile_size);

    nk_size_t const tile_elements = tile_dim * depth_tile_size; // 256 u32 per tile
    nk_size_t size = sizeof(nk_sets_smebi32_packed_header_t);
    size += row_tile_count * depth_tile_count * tile_elements * sizeof(nk_u32_t);
    size += row_count * sizeof(nk_u32_t); // per-row population counts

    return size;
}

NK_PUBLIC void nk_hammings_pack_u1_smebi32(nk_u1x8_t const *b, nk_size_t row_count, nk_size_t depth_bits,
                                           nk_size_t b_stride_in_bytes, void *b_packed) {
    nk_size_t const svl_bytes = nk_smebi32_svl_bytes_();
    nk_size_t const tile_dim = nk_smebi32_tile_dim_();        // 16 rows per tile
    nk_size_t const depth_tile_size = nk_smebi32_tile_dim_(); // 16 u32 per depth tile
    nk_size_t const tile_elements = tile_dim * depth_tile_size;
    nk_size_t const depth_in_bytes = nk_size_divide_round_up_(depth_bits, NK_BITS_PER_BYTE);

    nk_size_t const depth_u32_total = nk_size_divide_round_up_(depth_bits, 32);
    nk_size_t const row_tile_count = nk_size_divide_round_up_(row_count, tile_dim);
    nk_size_t const depth_tile_count = nk_size_divide_round_up_(depth_u32_total, depth_tile_size);
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

    // Compute per-row population counts
    for (nk_size_t row = 0; row < row_count; row++) {
        nk_u1x8_t const *src_row = (nk_u1x8_t const *)((char const *)b + row * b_stride_in_bytes);
        norms_ptr[row] = (nk_u32_t)nk_popcount_vector_serial_(src_row, depth_in_bytes);
    }
}

/**
 *  SME Hamming kernel using ZA transpose for unpacked A.
 *  ZA0.S = staging (A rows loaded horizontally, read vertically for BMOPA).
 *  ZA1-3.S = BMOPA accumulation (3 B column tiles in fast path).
 *
 *  Each ZA0.S batch covers 16 depth u32 steps (one full depth tile).
 *  BMOPA expansion=1 for u32: each u32 contributes 32 bits via XNOR+POPCNT.
 */
__arm_locally_streaming __arm_new("za") static void nk_hammings_packed_u1_smebi32_kernel_(
    nk_u1x8_t const *a, void const *b_packed, nk_u32_t *c, nk_size_t row_count_a, nk_size_t row_count_b,
    nk_size_t depth_bits, nk_size_t a_stride_in_bytes, nk_size_t c_stride_in_bytes) {

    nk_sets_smebi32_packed_header_t const *header = (nk_sets_smebi32_packed_header_t const *)b_packed;
    nk_size_t const row_tile_count_b = header->row_tile_count;
    nk_size_t const depth_tile_count = header->depth_tile_count;

    nk_size_t const tile_dim = svcntw();        // 16 for 512-bit SVL
    nk_size_t const depth_tile_size = svcntw(); // 16 u32 per depth tile
    nk_size_t const tile_elements = tile_dim * depth_tile_size;
    nk_size_t const depth_u32_total = nk_size_divide_round_up_(depth_bits, 32);

    nk_u32_t const *b_tiles = (nk_u32_t const *)((char const *)b_packed + sizeof(nk_sets_smebi32_packed_header_t));

    svbool_t const full_predicate_b32 = svptrue_b32();
    svuint32_t const depth_vec = svdup_u32((nk_u32_t)depth_bits);
    nk_size_t const row_tile_count_a = nk_size_divide_round_up_(row_count_a, tile_dim);

    for (nk_size_t row_tile_a = 0; row_tile_a < row_tile_count_a; row_tile_a++) {
        nk_size_t const row_start_a = row_tile_a * tile_dim;
        nk_size_t const rows_a_remaining = (row_start_a + tile_dim <= row_count_a) ? tile_dim
                                                                                   : (row_count_a - row_start_a);
        svbool_t const row_predicate_b32 = svwhilelt_b32((nk_u32_t)0, (nk_u32_t)rows_a_remaining);

        // Fast path: 3 B column tiles using ZA1-ZA3 (ZA0.S = staging)
        nk_size_t row_tile_b = 0;
        for (; row_tile_b + 3 <= row_tile_count_b; row_tile_b += 3) {
            svzero_mask_za(nk_sme_zero_za32_tiles_123_);

            for (nk_size_t d_tile = 0; d_tile < depth_tile_count; d_tile++) {
                nk_size_t const d_start_u32 = d_tile * depth_tile_size;
                nk_size_t const u32s_this_tile = (d_start_u32 + depth_tile_size <= depth_u32_total)
                                                     ? depth_tile_size
                                                     : (depth_u32_total > d_start_u32 ? depth_u32_total - d_start_u32
                                                                                      : 0);
                if (u32s_this_tile == 0) break;

                svzero_mask_za(nk_sme_zero_za32_tile_0_);

                svbool_t const batch_predicate_b32 = svwhilelt_b32((nk_u32_t)0, (nk_u32_t)u32s_this_tile);

                // Load A rows into ZA0.S horizontally as u32 words
                for (nk_size_t row_in_tile = 0; row_in_tile < rows_a_remaining; row_in_tile++) {
                    nk_u32_t const *a_row_u32 = (nk_u32_t const *)((char const *)a +
                                                                   (row_start_a + row_in_tile) * a_stride_in_bytes) +
                                                d_start_u32;
                    svld1_hor_za32(0, row_in_tile, batch_predicate_b32, a_row_u32);
                }

                // B tile pointers for 3 column tiles
                nk_u32_t const *b_tile0 = b_tiles + ((row_tile_b + 0) * depth_tile_count + d_tile) * tile_elements;
                nk_u32_t const *b_tile1 = b_tiles + ((row_tile_b + 1) * depth_tile_count + d_tile) * tile_elements;
                nk_u32_t const *b_tile2 = b_tiles + ((row_tile_b + 2) * depth_tile_count + d_tile) * tile_elements;

                // Vertical read + BMOPA for each depth step
                for (nk_size_t step = 0; step < u32s_this_tile; step++) {
                    svuint32_t a_column_u32 = svread_ver_za32_u32_m(svdup_u32(0), row_predicate_b32, 0, step);

                    svbmopa_za32_u32_m(1, row_predicate_b32, full_predicate_b32, a_column_u32,
                                       svld1_u32(full_predicate_b32, b_tile0 + step * tile_dim));
                    svbmopa_za32_u32_m(2, row_predicate_b32, full_predicate_b32, a_column_u32,
                                       svld1_u32(full_predicate_b32, b_tile1 + step * tile_dim));
                    svbmopa_za32_u32_m(3, row_predicate_b32, full_predicate_b32, a_column_u32,
                                       svld1_u32(full_predicate_b32, b_tile2 + step * tile_dim));
                }
            }

            // Extract from ZA1-3: Hamming = depth_bits - matching_bits
            for (nk_size_t row = 0; row < rows_a_remaining; row++) {
                nk_u32_t *c_row = (nk_u32_t *)((char *)c + (row_start_a + row) * c_stride_in_bytes);

                svuint32_t za1 = svread_hor_za32_u32_m(svdup_u32(0), full_predicate_b32, 1, row);
                svuint32_t za2 = svread_hor_za32_u32_m(svdup_u32(0), full_predicate_b32, 2, row);
                svuint32_t za3 = svread_hor_za32_u32_m(svdup_u32(0), full_predicate_b32, 3, row);

                svst1_u32(full_predicate_b32, c_row + (row_tile_b + 0) * tile_dim,
                          svsub_u32_x(full_predicate_b32, depth_vec, za1));
                svst1_u32(full_predicate_b32, c_row + (row_tile_b + 1) * tile_dim,
                          svsub_u32_x(full_predicate_b32, depth_vec, za2));
                svst1_u32(full_predicate_b32, c_row + (row_tile_b + 2) * tile_dim,
                          svsub_u32_x(full_predicate_b32, depth_vec, za3));
            }
        }

        // Remainder: 1 B column tile at a time using ZA1
        for (; row_tile_b < row_tile_count_b; row_tile_b++) {
            nk_size_t const row_start_b = row_tile_b * tile_dim;
            nk_size_t const rows_b_remaining = (row_start_b + tile_dim <= row_count_b) ? tile_dim
                                                                                       : (row_count_b - row_start_b);
            svbool_t const column_predicate_b32 = svwhilelt_b32((nk_u32_t)0, (nk_u32_t)rows_b_remaining);

            svzero_mask_za(nk_sme_zero_za32_tile_1_);

            for (nk_size_t d_tile = 0; d_tile < depth_tile_count; d_tile++) {
                nk_size_t const d_start_u32 = d_tile * depth_tile_size;
                nk_size_t const u32s_this_tile = (d_start_u32 + depth_tile_size <= depth_u32_total)
                                                     ? depth_tile_size
                                                     : (depth_u32_total > d_start_u32 ? depth_u32_total - d_start_u32
                                                                                      : 0);
                if (u32s_this_tile == 0) break;

                svzero_mask_za(nk_sme_zero_za32_tile_0_);

                svbool_t const batch_predicate_b32 = svwhilelt_b32((nk_u32_t)0, (nk_u32_t)u32s_this_tile);

                // Load A rows into ZA0.S horizontally
                for (nk_size_t row_in_tile = 0; row_in_tile < rows_a_remaining; row_in_tile++) {
                    nk_u32_t const *a_row_u32 = (nk_u32_t const *)((char const *)a +
                                                                   (row_start_a + row_in_tile) * a_stride_in_bytes) +
                                                d_start_u32;
                    svld1_hor_za32(0, row_in_tile, batch_predicate_b32, a_row_u32);
                }

                nk_u32_t const *b_tile = b_tiles + (row_tile_b * depth_tile_count + d_tile) * tile_elements;

                // Vertical read + BMOPA
                for (nk_size_t step = 0; step < u32s_this_tile; step++) {
                    svuint32_t a_column_u32 = svread_ver_za32_u32_m(svdup_u32(0), row_predicate_b32, 0, step);
                    svuint32_t b_vec = svld1_u32(full_predicate_b32, b_tile + step * tile_dim);
                    svbmopa_za32_u32_m(1, row_predicate_b32, column_predicate_b32, a_column_u32, b_vec);
                }
            }

            // Extract from ZA1: Hamming = depth_bits - matching_bits
            for (nk_size_t row = 0; row < rows_a_remaining; row++) {
                svuint32_t za1 = svread_hor_za32_u32_m(svdup_u32(0), full_predicate_b32, 1, row);
                svuint32_t hamming = svsub_u32_x(full_predicate_b32, depth_vec, za1);
                nk_u32_t *c_row = (nk_u32_t *)((char *)c + (row_start_a + row) * c_stride_in_bytes);
                svst1_u32(column_predicate_b32, c_row + row_start_b, hamming);
            }
        }
    }
}

__arm_locally_streaming __arm_new("za")
    NK_PUBLIC void nk_hammings_packed_u1_smebi32(nk_u1x8_t const *a, void const *b_packed, nk_u32_t *c,
                                                 nk_size_t row_count_a, nk_size_t row_count_b, nk_size_t depth_bits,
                                                 nk_size_t a_stride_in_bytes, nk_size_t c_stride_in_bytes) {
    nk_hammings_packed_u1_smebi32_kernel_(a, b_packed, c, row_count_a, row_count_b, depth_bits, a_stride_in_bytes,
                                          c_stride_in_bytes);
}

/**
 *  Symmetric Hamming using ZA0 time-sharing + 3-tile fast path.
 *  ZA0.S = staging (A rows loaded horizontally, read vertically for BMOPA).
 *  ZA1-3.S = BMOPA accumulators (3 B column tiles in fast path).
 *  Mirrors the unpacked kernel nk_hammings_packed_u1_smebi32_kernel_ pattern.
 */
__arm_locally_streaming __arm_new("za")
    NK_PUBLIC void nk_hammings_symmetric_u1_smebi32(nk_u1x8_t const *vectors, nk_size_t n_vectors, nk_size_t depth_bits,
                                                    nk_size_t stride, nk_u32_t *result, nk_size_t result_stride,
                                                    nk_size_t row_start, nk_size_t row_count) {

    nk_size_t const tile_dim = svcntw();        // 16 for 512-bit SVL
    nk_size_t const depth_tile_size = svcntw(); // 16 u32 per depth tile
    nk_size_t const depth_u32_total = nk_size_divide_round_up_(depth_bits, 32);
    nk_size_t const depth_tile_count = nk_size_divide_round_up_(depth_u32_total, depth_tile_size);

    svbool_t const full_predicate_b32 = svptrue_b32();
    svuint32_t const depth_vec = svdup_u32((nk_u32_t)depth_bits);

    NK_ALIGN64 nk_u32_t a_buffer[16][16]; // Stack buffer for A column save

    nk_size_t const row_end = row_start + row_count;
    nk_size_t const column_tile_count = nk_size_divide_round_up_(n_vectors, tile_dim);

    for (nk_size_t row_tile_start = row_start; row_tile_start < row_end && row_tile_start < n_vectors;
         row_tile_start += tile_dim) {
        nk_size_t const rows_remaining = (row_tile_start + tile_dim <= row_end) ? tile_dim : (row_end - row_tile_start);
        nk_size_t const rows_clamped = (row_tile_start + rows_remaining <= n_vectors) ? rows_remaining
                                                                                      : (n_vectors - row_tile_start);
        svbool_t const row_predicate_b32 = svwhilelt_b32((nk_u32_t)0, (nk_u32_t)rows_clamped);

        nk_size_t column_tile_index = 0;

        // Fast path: 3 column tiles using ZA1-ZA3 (ZA0 = staging)
        for (; column_tile_index + 3 <= column_tile_count; column_tile_index += 3) {
            svzero_mask_za(nk_sme_zero_za32_tiles_123_);

            for (nk_size_t d_tile = 0; d_tile < depth_tile_count; d_tile++) {
                nk_size_t const d_start_u32 = d_tile * depth_tile_size;
                nk_size_t const u32s_this_tile = (d_start_u32 + depth_tile_size <= depth_u32_total)
                                                     ? depth_tile_size
                                                     : (depth_u32_total > d_start_u32 ? depth_u32_total - d_start_u32
                                                                                      : 0);
                if (u32s_this_tile == 0) break;

                // Load A rows into ZA0 horizontally
                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                svbool_t const batch_predicate_b32 = svwhilelt_b32((nk_u32_t)0, (nk_u32_t)u32s_this_tile);

                for (nk_size_t row_in_tile = 0; row_in_tile < rows_clamped; row_in_tile++) {
                    nk_u32_t const *a_row_u32 = (nk_u32_t const *)((char const *)vectors +
                                                                   (row_tile_start + row_in_tile) * stride) +
                                                d_start_u32;
                    svld1_hor_za32(0, row_in_tile, batch_predicate_b32, a_row_u32);
                }

                // Save A columns from ZA0 to stack buffer
                for (nk_size_t s = 0; s < u32s_this_tile; s++)
                    svst1_u32(full_predicate_b32, a_buffer[s],
                              svread_ver_za32_u32_m(svdup_u32(0), row_predicate_b32, 0, s));

                // B column tile 0
                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t col = 0; col < tile_dim; col++) {
                    nk_size_t const col_abs = (column_tile_index + 0) * tile_dim + col;
                    if (col_abs < n_vectors) {
                        nk_u32_t const *b_row = (nk_u32_t const *)((char const *)vectors + col_abs * stride) +
                                                d_start_u32;
                        svld1_hor_za32(0, col, batch_predicate_b32, b_row);
                    }
                }
                for (nk_size_t step = 0; step < u32s_this_tile; step++) {
                    svuint32_t a_vec = svld1_u32(full_predicate_b32, a_buffer[step]);
                    svuint32_t b_vec = svread_ver_za32_u32_m(svdup_u32(0), full_predicate_b32, 0, step);
                    svbmopa_za32_u32_m(1, row_predicate_b32, full_predicate_b32, a_vec, b_vec);
                }

                // B column tile 1
                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t col = 0; col < tile_dim; col++) {
                    nk_size_t const col_abs = (column_tile_index + 1) * tile_dim + col;
                    if (col_abs < n_vectors) {
                        nk_u32_t const *b_row = (nk_u32_t const *)((char const *)vectors + col_abs * stride) +
                                                d_start_u32;
                        svld1_hor_za32(0, col, batch_predicate_b32, b_row);
                    }
                }
                for (nk_size_t step = 0; step < u32s_this_tile; step++) {
                    svuint32_t a_vec = svld1_u32(full_predicate_b32, a_buffer[step]);
                    svuint32_t b_vec = svread_ver_za32_u32_m(svdup_u32(0), full_predicate_b32, 0, step);
                    svbmopa_za32_u32_m(2, row_predicate_b32, full_predicate_b32, a_vec, b_vec);
                }

                // B column tile 2
                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t col = 0; col < tile_dim; col++) {
                    nk_size_t const col_abs = (column_tile_index + 2) * tile_dim + col;
                    if (col_abs < n_vectors) {
                        nk_u32_t const *b_row = (nk_u32_t const *)((char const *)vectors + col_abs * stride) +
                                                d_start_u32;
                        svld1_hor_za32(0, col, batch_predicate_b32, b_row);
                    }
                }
                for (nk_size_t step = 0; step < u32s_this_tile; step++) {
                    svuint32_t a_vec = svld1_u32(full_predicate_b32, a_buffer[step]);
                    svuint32_t b_vec = svread_ver_za32_u32_m(svdup_u32(0), full_predicate_b32, 0, step);
                    svbmopa_za32_u32_m(3, row_predicate_b32, full_predicate_b32, a_vec, b_vec);
                }
            }

            // Extract ZA1-3: hamming = depth_bits - ZA[i][j]
            for (nk_size_t row = 0; row < rows_clamped; row++) {
                nk_u32_t *c_row = (nk_u32_t *)((char *)result + (row_tile_start + row) * result_stride);

                svuint32_t za1 = svread_hor_za32_u32_m(svdup_u32(0), full_predicate_b32, 1, row);
                svuint32_t za2 = svread_hor_za32_u32_m(svdup_u32(0), full_predicate_b32, 2, row);
                svuint32_t za3 = svread_hor_za32_u32_m(svdup_u32(0), full_predicate_b32, 3, row);

                svst1_u32(full_predicate_b32, c_row + (column_tile_index + 0) * tile_dim,
                          svsub_u32_x(full_predicate_b32, depth_vec, za1));
                svst1_u32(full_predicate_b32, c_row + (column_tile_index + 1) * tile_dim,
                          svsub_u32_x(full_predicate_b32, depth_vec, za2));
                svst1_u32(full_predicate_b32, c_row + (column_tile_index + 2) * tile_dim,
                          svsub_u32_x(full_predicate_b32, depth_vec, za3));
            }
        }

        // Remainder: 1 column tile at a time using ZA1
        for (; column_tile_index < column_tile_count; column_tile_index++) {
            nk_size_t const col_tile_start = column_tile_index * tile_dim;
            nk_size_t const cols_remaining = (col_tile_start + tile_dim <= n_vectors) ? tile_dim
                                                                                      : (n_vectors - col_tile_start);
            svbool_t const column_predicate_b32 = svwhilelt_b32((nk_u32_t)0, (nk_u32_t)cols_remaining);

            svzero_mask_za(nk_sme_zero_za32_tile_1_);

            for (nk_size_t d_tile = 0; d_tile < depth_tile_count; d_tile++) {
                nk_size_t const d_start_u32 = d_tile * depth_tile_size;
                nk_size_t const u32s_this_tile = (d_start_u32 + depth_tile_size <= depth_u32_total)
                                                     ? depth_tile_size
                                                     : (depth_u32_total > d_start_u32 ? depth_u32_total - d_start_u32
                                                                                      : 0);
                if (u32s_this_tile == 0) break;

                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                svbool_t const batch_predicate_b32 = svwhilelt_b32((nk_u32_t)0, (nk_u32_t)u32s_this_tile);

                // Load A rows into ZA0 horizontally
                for (nk_size_t row_in_tile = 0; row_in_tile < rows_clamped; row_in_tile++) {
                    nk_u32_t const *a_row_u32 = (nk_u32_t const *)((char const *)vectors +
                                                                   (row_tile_start + row_in_tile) * stride) +
                                                d_start_u32;
                    svld1_hor_za32(0, row_in_tile, batch_predicate_b32, a_row_u32);
                }

                // Save A columns from ZA0 to stack buffer
                for (nk_size_t s = 0; s < u32s_this_tile; s++)
                    svst1_u32(full_predicate_b32, a_buffer[s],
                              svread_ver_za32_u32_m(svdup_u32(0), row_predicate_b32, 0, s));

                // Load B column tile into ZA0
                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t col = 0; col < tile_dim; col++) {
                    nk_size_t const col_abs = col_tile_start + col;
                    if (col_abs < n_vectors) {
                        nk_u32_t const *b_row = (nk_u32_t const *)((char const *)vectors + col_abs * stride) +
                                                d_start_u32;
                        svld1_hor_za32(0, col, batch_predicate_b32, b_row);
                    }
                }
                for (nk_size_t step = 0; step < u32s_this_tile; step++) {
                    svuint32_t a_vec = svld1_u32(full_predicate_b32, a_buffer[step]);
                    svuint32_t b_vec = svread_ver_za32_u32_m(svdup_u32(0), column_predicate_b32, 0, step);
                    svbmopa_za32_u32_m(1, row_predicate_b32, column_predicate_b32, a_vec, b_vec);
                }
            }

            for (nk_size_t row = 0; row < rows_clamped; row++) {
                svuint32_t za1 = svread_hor_za32_u32_m(svdup_u32(0), full_predicate_b32, 1, row);
                svuint32_t hamming = svsub_u32_x(full_predicate_b32, depth_vec, za1);
                nk_u32_t *c_row = (nk_u32_t *)((char *)result + (row_tile_start + row) * result_stride);
                svst1_u32(column_predicate_b32, c_row + col_tile_start, hamming);
            }
        }
    }
}

#pragma endregion // Hamming Distance

/*
 *  Jaccard distance via BMOPA matching counts + algebraic normalization.
 *
 *  BMOPA gives: matching = popcount(XNOR(a,b))
 *  Then:
 *    hamming      = depth_bits - matching
 *    intersection = (norm_a + norm_b - hamming) / 2  =  (norm_a + norm_b - depth_bits + matching) / 2
 *    union        = (norm_a + norm_b + hamming) / 2  =  sum_norms - intersection
 *    jaccard      = 1 - intersection / union          (1.0 when union == 0)
 *
 *  Inner BMOPA loop is identical to Hamming; only the extraction phase differs.
 *  Packed format shares the Hamming tile layout for B operand, plus per-row norms.
 */

#pragma region Jaccard Distance

NK_PUBLIC nk_size_t nk_jaccards_packed_size_u1_smebi32(nk_size_t row_count, nk_size_t depth_bits) {
    nk_size_t const tile_dim = nk_smebi32_tile_dim_();        // 16 rows per tile
    nk_size_t const depth_tile_size = nk_smebi32_tile_dim_(); // 16 u32 per depth tile

    nk_size_t const depth_u32 = nk_size_divide_round_up_(depth_bits, 32);
    nk_size_t const row_tile_count = nk_size_divide_round_up_(row_count, tile_dim);
    nk_size_t const depth_tile_count = nk_size_divide_round_up_(depth_u32, depth_tile_size);

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
    nk_size_t const depth_in_bytes = nk_size_divide_round_up_(depth_bits, NK_BITS_PER_BYTE);

    nk_size_t const depth_u32_total = nk_size_divide_round_up_(depth_bits, 32);
    nk_size_t const row_tile_count = nk_size_divide_round_up_(row_count, tile_dim);
    nk_size_t const depth_tile_count = nk_size_divide_round_up_(depth_u32_total, depth_tile_size);
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
 *  SME Jaccard kernel using BMOPA for matching-bit counts.
 *  Mirrors nk_hammings_packed_u1_smebi32_kernel_ exactly in structure,
 *  but derives intersection/union algebraically from the matching counts:
 *    matching      = popcount(XNOR(a,b))          (from BMOPA)
 *    hamming       = depth_bits - matching
 *    intersection  = (norm_a + norm_b - hamming) / 2
 *    union         = (norm_a + norm_b + hamming) / 2
 *    jaccard       = 1 - intersection / union      (1.0 when union == 0)
 */
__arm_locally_streaming __arm_new("za") static void nk_jaccards_packed_u1_smebi32_kernel_(
    nk_u1x8_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t row_count_a, nk_size_t row_count_b,
    nk_size_t depth_bits, nk_size_t a_stride_in_bytes, nk_size_t c_stride_in_bytes, nk_f32_t const *a_norms) {

    nk_sets_smebi32_packed_header_t const *header = (nk_sets_smebi32_packed_header_t const *)b_packed;
    nk_size_t const row_tile_count_b = header->row_tile_count;
    nk_size_t const depth_tile_count = header->depth_tile_count;

    nk_size_t const tile_dim = svcntw();        // 16 for 512-bit SVL
    nk_size_t const depth_tile_size = svcntw(); // 16 u32 per depth tile
    nk_size_t const tile_elements = tile_dim * depth_tile_size;
    nk_size_t const depth_u32_total = nk_size_divide_round_up_(depth_bits, 32);

    nk_u32_t const *b_tiles = (nk_u32_t const *)((char const *)b_packed + sizeof(nk_sets_smebi32_packed_header_t));
    nk_u32_t const *b_norms = header->norms_offset ? (nk_u32_t const *)((char const *)b_packed + header->norms_offset)
                                                   : (nk_u32_t const *)0;

    svbool_t const full_predicate_b32 = svptrue_b32();
    svfloat32_t const depth_f32_vec = svdup_f32((nk_f32_t)depth_bits);
    svfloat32_t const half_f32_vec = svdup_f32(0.5f);
    svfloat32_t const one_f32_vec = svdup_f32(1.0f);
    svfloat32_t const zero_f32_vec = svdup_f32(0.0f);
    nk_size_t const row_tile_count_a = nk_size_divide_round_up_(row_count_a, tile_dim);

    for (nk_size_t row_tile_a = 0; row_tile_a < row_tile_count_a; row_tile_a++) {
        nk_size_t const row_start_a = row_tile_a * tile_dim;
        nk_size_t const rows_a_remaining = (row_start_a + tile_dim <= row_count_a) ? tile_dim
                                                                                   : (row_count_a - row_start_a);
        svbool_t const row_predicate_b32 = svwhilelt_b32((nk_u32_t)0, (nk_u32_t)rows_a_remaining);

        // Fast path: 3 B column tiles using ZA1-ZA3 (ZA0.S = staging)
        nk_size_t row_tile_b = 0;
        for (; row_tile_b + 3 <= row_tile_count_b; row_tile_b += 3) {
            svzero_mask_za(nk_sme_zero_za32_tiles_123_);

            for (nk_size_t d_tile = 0; d_tile < depth_tile_count; d_tile++) {
                nk_size_t const d_start_u32 = d_tile * depth_tile_size;
                nk_size_t const u32s_this_tile = (d_start_u32 + depth_tile_size <= depth_u32_total)
                                                     ? depth_tile_size
                                                     : (depth_u32_total > d_start_u32 ? depth_u32_total - d_start_u32
                                                                                      : 0);
                if (u32s_this_tile == 0) break;

                svzero_mask_za(nk_sme_zero_za32_tile_0_);

                svbool_t const batch_predicate_b32 = svwhilelt_b32((nk_u32_t)0, (nk_u32_t)u32s_this_tile);

                // Load A rows into ZA0.S horizontally as u32 words
                for (nk_size_t row_in_tile = 0; row_in_tile < rows_a_remaining; row_in_tile++) {
                    nk_u32_t const *a_row_u32 = (nk_u32_t const *)((char const *)a +
                                                                   (row_start_a + row_in_tile) * a_stride_in_bytes) +
                                                d_start_u32;
                    svld1_hor_za32(0, row_in_tile, batch_predicate_b32, a_row_u32);
                }

                // B tile pointers for 3 column tiles
                nk_u32_t const *b_tile0 = b_tiles + ((row_tile_b + 0) * depth_tile_count + d_tile) * tile_elements;
                nk_u32_t const *b_tile1 = b_tiles + ((row_tile_b + 1) * depth_tile_count + d_tile) * tile_elements;
                nk_u32_t const *b_tile2 = b_tiles + ((row_tile_b + 2) * depth_tile_count + d_tile) * tile_elements;

                // Vertical read + BMOPA for each depth step
                for (nk_size_t step = 0; step < u32s_this_tile; step++) {
                    svuint32_t a_column_u32 = svread_ver_za32_u32_m(svdup_u32(0), row_predicate_b32, 0, step);

                    svbmopa_za32_u32_m(1, row_predicate_b32, full_predicate_b32, a_column_u32,
                                       svld1_u32(full_predicate_b32, b_tile0 + step * tile_dim));
                    svbmopa_za32_u32_m(2, row_predicate_b32, full_predicate_b32, a_column_u32,
                                       svld1_u32(full_predicate_b32, b_tile1 + step * tile_dim));
                    svbmopa_za32_u32_m(3, row_predicate_b32, full_predicate_b32, a_column_u32,
                                       svld1_u32(full_predicate_b32, b_tile2 + step * tile_dim));
                }
            }

            // Extract from ZA1-3: Jaccard normalization via streaming SVE
            // Hoist B norms outside row loop (same for all A rows in this tile-pair)
            svfloat32_t b_norms_f32_0 = svcvt_f32_u32_x(
                full_predicate_b32, svld1_u32(full_predicate_b32, b_norms + (row_tile_b + 0) * tile_dim));
            svfloat32_t b_norms_f32_1 = svcvt_f32_u32_x(
                full_predicate_b32, svld1_u32(full_predicate_b32, b_norms + (row_tile_b + 1) * tile_dim));
            svfloat32_t b_norms_f32_2 = svcvt_f32_u32_x(
                full_predicate_b32, svld1_u32(full_predicate_b32, b_norms + (row_tile_b + 2) * tile_dim));

            for (nk_size_t row = 0; row < rows_a_remaining; row++) {
                nk_f32_t *c_row = (nk_f32_t *)((char *)c + (row_start_a + row) * c_stride_in_bytes);
                svfloat32_t norm_a_f32 = svdup_f32(a_norms[row_start_a + row]);

                // ZA1
                {
                    svuint32_t za1 = svread_hor_za32_u32_m(svdup_u32(0), full_predicate_b32, 1, row);
                    svfloat32_t matching = svcvt_f32_u32_x(full_predicate_b32, za1);
                    svfloat32_t sum_norms = svadd_f32_x(full_predicate_b32, norm_a_f32, b_norms_f32_0);
                    svfloat32_t intersection = svmul_f32_x(
                        full_predicate_b32,
                        svadd_f32_x(full_predicate_b32, svsub_f32_x(full_predicate_b32, sum_norms, depth_f32_vec),
                                    matching),
                        half_f32_vec);
                    svfloat32_t union_val = svsub_f32_x(full_predicate_b32, sum_norms, intersection);
                    svbool_t nonzero = svcmpne_f32(full_predicate_b32, union_val, zero_f32_vec);
                    svfloat32_t ratio = svdiv_f32_x(full_predicate_b32, intersection, union_val);
                    svfloat32_t jaccard = svsel_f32(nonzero, svsub_f32_x(full_predicate_b32, one_f32_vec, ratio),
                                                    one_f32_vec);
                    svst1_f32(full_predicate_b32, c_row + (row_tile_b + 0) * tile_dim, jaccard);
                }
                // ZA2
                {
                    svuint32_t za2 = svread_hor_za32_u32_m(svdup_u32(0), full_predicate_b32, 2, row);
                    svfloat32_t matching = svcvt_f32_u32_x(full_predicate_b32, za2);
                    svfloat32_t sum_norms = svadd_f32_x(full_predicate_b32, norm_a_f32, b_norms_f32_1);
                    svfloat32_t intersection = svmul_f32_x(
                        full_predicate_b32,
                        svadd_f32_x(full_predicate_b32, svsub_f32_x(full_predicate_b32, sum_norms, depth_f32_vec),
                                    matching),
                        half_f32_vec);
                    svfloat32_t union_val = svsub_f32_x(full_predicate_b32, sum_norms, intersection);
                    svbool_t nonzero = svcmpne_f32(full_predicate_b32, union_val, zero_f32_vec);
                    svfloat32_t ratio = svdiv_f32_x(full_predicate_b32, intersection, union_val);
                    svfloat32_t jaccard = svsel_f32(nonzero, svsub_f32_x(full_predicate_b32, one_f32_vec, ratio),
                                                    one_f32_vec);
                    svst1_f32(full_predicate_b32, c_row + (row_tile_b + 1) * tile_dim, jaccard);
                }
                // ZA3
                {
                    svuint32_t za3 = svread_hor_za32_u32_m(svdup_u32(0), full_predicate_b32, 3, row);
                    svfloat32_t matching = svcvt_f32_u32_x(full_predicate_b32, za3);
                    svfloat32_t sum_norms = svadd_f32_x(full_predicate_b32, norm_a_f32, b_norms_f32_2);
                    svfloat32_t intersection = svmul_f32_x(
                        full_predicate_b32,
                        svadd_f32_x(full_predicate_b32, svsub_f32_x(full_predicate_b32, sum_norms, depth_f32_vec),
                                    matching),
                        half_f32_vec);
                    svfloat32_t union_val = svsub_f32_x(full_predicate_b32, sum_norms, intersection);
                    svbool_t nonzero = svcmpne_f32(full_predicate_b32, union_val, zero_f32_vec);
                    svfloat32_t ratio = svdiv_f32_x(full_predicate_b32, intersection, union_val);
                    svfloat32_t jaccard = svsel_f32(nonzero, svsub_f32_x(full_predicate_b32, one_f32_vec, ratio),
                                                    one_f32_vec);
                    svst1_f32(full_predicate_b32, c_row + (row_tile_b + 2) * tile_dim, jaccard);
                }
            }
        }

        // Remainder: 1 B column tile at a time using ZA1
        for (; row_tile_b < row_tile_count_b; row_tile_b++) {
            nk_size_t const row_start_b = row_tile_b * tile_dim;
            nk_size_t const rows_b_remaining = (row_start_b + tile_dim <= row_count_b) ? tile_dim
                                                                                       : (row_count_b - row_start_b);
            svbool_t const column_predicate_b32 = svwhilelt_b32((nk_u32_t)0, (nk_u32_t)rows_b_remaining);

            svzero_mask_za(nk_sme_zero_za32_tile_1_);

            for (nk_size_t d_tile = 0; d_tile < depth_tile_count; d_tile++) {
                nk_size_t const d_start_u32 = d_tile * depth_tile_size;
                nk_size_t const u32s_this_tile = (d_start_u32 + depth_tile_size <= depth_u32_total)
                                                     ? depth_tile_size
                                                     : (depth_u32_total > d_start_u32 ? depth_u32_total - d_start_u32
                                                                                      : 0);
                if (u32s_this_tile == 0) break;

                svzero_mask_za(nk_sme_zero_za32_tile_0_);

                svbool_t const batch_predicate_b32 = svwhilelt_b32((nk_u32_t)0, (nk_u32_t)u32s_this_tile);

                // Load A rows into ZA0.S horizontally
                for (nk_size_t row_in_tile = 0; row_in_tile < rows_a_remaining; row_in_tile++) {
                    nk_u32_t const *a_row_u32 = (nk_u32_t const *)((char const *)a +
                                                                   (row_start_a + row_in_tile) * a_stride_in_bytes) +
                                                d_start_u32;
                    svld1_hor_za32(0, row_in_tile, batch_predicate_b32, a_row_u32);
                }

                nk_u32_t const *b_tile = b_tiles + (row_tile_b * depth_tile_count + d_tile) * tile_elements;

                // Vertical read + BMOPA
                for (nk_size_t step = 0; step < u32s_this_tile; step++) {
                    svuint32_t a_column_u32 = svread_ver_za32_u32_m(svdup_u32(0), row_predicate_b32, 0, step);
                    svuint32_t b_vec = svld1_u32(full_predicate_b32, b_tile + step * tile_dim);
                    svbmopa_za32_u32_m(1, row_predicate_b32, column_predicate_b32, a_column_u32, b_vec);
                }
            }

            // Extract from ZA1: Jaccard normalization
            svfloat32_t b_norms_f32 = svcvt_f32_u32_x(full_predicate_b32,
                                                      svld1_u32(full_predicate_b32, b_norms + row_start_b));
            for (nk_size_t row = 0; row < rows_a_remaining; row++) {
                svuint32_t za1 = svread_hor_za32_u32_m(svdup_u32(0), full_predicate_b32, 1, row);
                svfloat32_t matching = svcvt_f32_u32_x(full_predicate_b32, za1);
                svfloat32_t norm_a_f32 = svdup_f32(a_norms[row_start_a + row]);
                svfloat32_t sum_norms = svadd_f32_x(full_predicate_b32, norm_a_f32, b_norms_f32);
                svfloat32_t intersection = svmul_f32_x(
                    full_predicate_b32,
                    svadd_f32_x(full_predicate_b32, svsub_f32_x(full_predicate_b32, sum_norms, depth_f32_vec),
                                matching),
                    half_f32_vec);
                svfloat32_t union_val = svsub_f32_x(full_predicate_b32, sum_norms, intersection);
                svbool_t nonzero = svcmpne_f32(full_predicate_b32, union_val, zero_f32_vec);
                svfloat32_t ratio = svdiv_f32_x(full_predicate_b32, intersection, union_val);
                svfloat32_t jaccard = svsel_f32(nonzero, svsub_f32_x(full_predicate_b32, one_f32_vec, ratio),
                                                one_f32_vec);
                nk_f32_t *c_row = (nk_f32_t *)((char *)c + (row_start_a + row) * c_stride_in_bytes);
                svst1_f32(column_predicate_b32, c_row + row_start_b, jaccard);
            }
        }
    }
}

NK_PUBLIC void nk_jaccards_packed_u1_smebi32(nk_u1x8_t const *a, void const *b_packed, nk_f32_t *c,
                                             nk_size_t row_count_a, nk_size_t row_count_b, nk_size_t depth_bits,
                                             nk_size_t a_stride_in_bytes, nk_size_t c_stride_in_bytes,
                                             nk_f32_t const *a_norms) {
    nk_jaccards_packed_u1_smebi32_kernel_(a, b_packed, c, row_count_a, row_count_b, depth_bits, a_stride_in_bytes,
                                          c_stride_in_bytes, a_norms);
}

/**
 *  Symmetric Jaccard kernel using ZA0 time-sharing + 3-tile fast path.
 *  Fills upper triangle only (column_tile >= row_tile); caller sees result[i][j] for j >= i.
 *  Norms computed on-the-fly using streaming SVE popcount.
 */
__arm_locally_streaming __arm_new("za") static void nk_jaccards_symmetric_u1_smebi32_kernel_(
    nk_u1x8_t const *vectors, nk_size_t n_vectors, nk_size_t depth_bits, nk_size_t stride, nk_f32_t *result,
    nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count) {

    nk_size_t const tile_dim = svcntw();        // 16 for 512-bit SVL
    nk_size_t const depth_tile_size = svcntw(); // 16 u32 per depth tile
    nk_size_t const depth_u32_total = nk_size_divide_round_up_(depth_bits, 32);
    nk_size_t const depth_tile_count = nk_size_divide_round_up_(depth_u32_total, depth_tile_size);
    nk_size_t const depth_in_bytes = nk_size_divide_round_up_(depth_bits, NK_BITS_PER_BYTE);

    svbool_t const full_predicate_b32 = svptrue_b32();
    svfloat32_t const depth_f32_vec = svdup_f32((nk_f32_t)depth_bits);
    svfloat32_t const half_f32_vec = svdup_f32(0.5f);
    svfloat32_t const one_f32_vec = svdup_f32(1.0f);
    svfloat32_t const zero_f32_vec = svdup_f32(0.0f);

    NK_ALIGN64 nk_u32_t a_buffer[16][16]; // Stack buffer for A column save

    nk_size_t const row_end = row_start + row_count;
    nk_size_t const column_tile_count = nk_size_divide_round_up_(n_vectors, tile_dim);

    for (nk_size_t row_tile_start = row_start; row_tile_start < row_end && row_tile_start < n_vectors;
         row_tile_start += tile_dim) {
        nk_size_t const rows_remaining = (row_tile_start + tile_dim <= row_end) ? tile_dim : (row_end - row_tile_start);
        nk_size_t const rows_clamped = (row_tile_start + rows_remaining <= n_vectors) ? rows_remaining
                                                                                      : (n_vectors - row_tile_start);
        svbool_t const row_predicate_b32 = svwhilelt_b32((nk_u32_t)0, (nk_u32_t)rows_clamped);

        // Compute A tile norms
        NK_ALIGN64 nk_f32_t a_tile_norms[16];
        for (nk_size_t r = 0; r < rows_clamped; r++) {
            nk_u1x8_t const *a_row = (nk_u1x8_t const *)((char const *)vectors + (row_tile_start + r) * stride);
            a_tile_norms[r] = (nk_f32_t)nk_popcount_vector_streaming_(a_row, depth_in_bytes);
        }
        for (nk_size_t r = rows_clamped; r < tile_dim; r++) a_tile_norms[r] = 0.0f;

        // Upper triangle: start from this row tile's column
        nk_size_t column_tile_index = row_tile_start / tile_dim;

        // Fast path: 3 column tiles using ZA1-ZA3 (ZA0 = staging)
        for (; column_tile_index + 3 <= column_tile_count; column_tile_index += 3) {
            svzero_mask_za(nk_sme_zero_za32_tiles_123_);

            for (nk_size_t d_tile = 0; d_tile < depth_tile_count; d_tile++) {
                nk_size_t const d_start_u32 = d_tile * depth_tile_size;
                nk_size_t const u32s_this_tile = (d_start_u32 + depth_tile_size <= depth_u32_total)
                                                     ? depth_tile_size
                                                     : (depth_u32_total > d_start_u32 ? depth_u32_total - d_start_u32
                                                                                      : 0);
                if (u32s_this_tile == 0) break;

                // Load A rows into ZA0 horizontally
                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                svbool_t const batch_predicate_b32 = svwhilelt_b32((nk_u32_t)0, (nk_u32_t)u32s_this_tile);

                for (nk_size_t row_in_tile = 0; row_in_tile < rows_clamped; row_in_tile++) {
                    nk_u32_t const *a_row_u32 = (nk_u32_t const *)((char const *)vectors +
                                                                   (row_tile_start + row_in_tile) * stride) +
                                                d_start_u32;
                    svld1_hor_za32(0, row_in_tile, batch_predicate_b32, a_row_u32);
                }

                // Save A columns from ZA0 to stack buffer
                for (nk_size_t s = 0; s < u32s_this_tile; s++)
                    svst1_u32(full_predicate_b32, a_buffer[s],
                              svread_ver_za32_u32_m(svdup_u32(0), row_predicate_b32, 0, s));

                // B column tile 0
                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t col = 0; col < tile_dim; col++) {
                    nk_size_t const col_abs = (column_tile_index + 0) * tile_dim + col;
                    if (col_abs < n_vectors) {
                        nk_u32_t const *b_row = (nk_u32_t const *)((char const *)vectors + col_abs * stride) +
                                                d_start_u32;
                        svld1_hor_za32(0, col, batch_predicate_b32, b_row);
                    }
                }
                for (nk_size_t step = 0; step < u32s_this_tile; step++) {
                    svuint32_t a_vec = svld1_u32(full_predicate_b32, a_buffer[step]);
                    svuint32_t b_vec = svread_ver_za32_u32_m(svdup_u32(0), full_predicate_b32, 0, step);
                    svbmopa_za32_u32_m(1, row_predicate_b32, full_predicate_b32, a_vec, b_vec);
                }

                // B column tile 1
                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t col = 0; col < tile_dim; col++) {
                    nk_size_t const col_abs = (column_tile_index + 1) * tile_dim + col;
                    if (col_abs < n_vectors) {
                        nk_u32_t const *b_row = (nk_u32_t const *)((char const *)vectors + col_abs * stride) +
                                                d_start_u32;
                        svld1_hor_za32(0, col, batch_predicate_b32, b_row);
                    }
                }
                for (nk_size_t step = 0; step < u32s_this_tile; step++) {
                    svuint32_t a_vec = svld1_u32(full_predicate_b32, a_buffer[step]);
                    svuint32_t b_vec = svread_ver_za32_u32_m(svdup_u32(0), full_predicate_b32, 0, step);
                    svbmopa_za32_u32_m(2, row_predicate_b32, full_predicate_b32, a_vec, b_vec);
                }

                // B column tile 2
                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t col = 0; col < tile_dim; col++) {
                    nk_size_t const col_abs = (column_tile_index + 2) * tile_dim + col;
                    if (col_abs < n_vectors) {
                        nk_u32_t const *b_row = (nk_u32_t const *)((char const *)vectors + col_abs * stride) +
                                                d_start_u32;
                        svld1_hor_za32(0, col, batch_predicate_b32, b_row);
                    }
                }
                for (nk_size_t step = 0; step < u32s_this_tile; step++) {
                    svuint32_t a_vec = svld1_u32(full_predicate_b32, a_buffer[step]);
                    svuint32_t b_vec = svread_ver_za32_u32_m(svdup_u32(0), full_predicate_b32, 0, step);
                    svbmopa_za32_u32_m(3, row_predicate_b32, full_predicate_b32, a_vec, b_vec);
                }
            }

            // Compute B tile norms for 3 column tiles
            NK_ALIGN64 nk_u32_t b_tile_norms_0[16];
            NK_ALIGN64 nk_u32_t b_tile_norms_1[16];
            NK_ALIGN64 nk_u32_t b_tile_norms_2[16];
            for (nk_size_t col = 0; col < tile_dim; col++) {
                nk_size_t const col_abs_0 = (column_tile_index + 0) * tile_dim + col;
                nk_size_t const col_abs_1 = (column_tile_index + 1) * tile_dim + col;
                nk_size_t const col_abs_2 = (column_tile_index + 2) * tile_dim + col;
                b_tile_norms_0[col] = (col_abs_0 < n_vectors)
                                          ? nk_popcount_vector_streaming_(
                                                (nk_u1x8_t const *)((char const *)vectors + col_abs_0 * stride),
                                                depth_in_bytes)
                                          : 0;
                b_tile_norms_1[col] = (col_abs_1 < n_vectors)
                                          ? nk_popcount_vector_streaming_(
                                                (nk_u1x8_t const *)((char const *)vectors + col_abs_1 * stride),
                                                depth_in_bytes)
                                          : 0;
                b_tile_norms_2[col] = (col_abs_2 < n_vectors)
                                          ? nk_popcount_vector_streaming_(
                                                (nk_u1x8_t const *)((char const *)vectors + col_abs_2 * stride),
                                                depth_in_bytes)
                                          : 0;
            }

            // Extract ZA1-3: Jaccard normalization
            svfloat32_t b_norms_f32_0 = svcvt_f32_u32_x(full_predicate_b32,
                                                        svld1_u32(full_predicate_b32, b_tile_norms_0));
            svfloat32_t b_norms_f32_1 = svcvt_f32_u32_x(full_predicate_b32,
                                                        svld1_u32(full_predicate_b32, b_tile_norms_1));
            svfloat32_t b_norms_f32_2 = svcvt_f32_u32_x(full_predicate_b32,
                                                        svld1_u32(full_predicate_b32, b_tile_norms_2));

            for (nk_size_t row = 0; row < rows_clamped; row++) {
                nk_f32_t *c_row = (nk_f32_t *)((char *)result + (row_tile_start + row) * result_stride);
                svfloat32_t norm_a_f32 = svdup_f32(a_tile_norms[row]);

                // ZA1
                {
                    svuint32_t za1 = svread_hor_za32_u32_m(svdup_u32(0), full_predicate_b32, 1, row);
                    svfloat32_t matching = svcvt_f32_u32_x(full_predicate_b32, za1);
                    svfloat32_t sum_norms = svadd_f32_x(full_predicate_b32, norm_a_f32, b_norms_f32_0);
                    svfloat32_t intersection = svmul_f32_x(
                        full_predicate_b32,
                        svadd_f32_x(full_predicate_b32, svsub_f32_x(full_predicate_b32, sum_norms, depth_f32_vec),
                                    matching),
                        half_f32_vec);
                    svfloat32_t union_val = svsub_f32_x(full_predicate_b32, sum_norms, intersection);
                    svbool_t nonzero = svcmpne_f32(full_predicate_b32, union_val, zero_f32_vec);
                    svfloat32_t ratio = svdiv_f32_x(full_predicate_b32, intersection, union_val);
                    svfloat32_t jaccard = svsel_f32(nonzero, svsub_f32_x(full_predicate_b32, one_f32_vec, ratio),
                                                    one_f32_vec);
                    svst1_f32(full_predicate_b32, c_row + (column_tile_index + 0) * tile_dim, jaccard);
                }
                // ZA2
                {
                    svuint32_t za2 = svread_hor_za32_u32_m(svdup_u32(0), full_predicate_b32, 2, row);
                    svfloat32_t matching = svcvt_f32_u32_x(full_predicate_b32, za2);
                    svfloat32_t sum_norms = svadd_f32_x(full_predicate_b32, norm_a_f32, b_norms_f32_1);
                    svfloat32_t intersection = svmul_f32_x(
                        full_predicate_b32,
                        svadd_f32_x(full_predicate_b32, svsub_f32_x(full_predicate_b32, sum_norms, depth_f32_vec),
                                    matching),
                        half_f32_vec);
                    svfloat32_t union_val = svsub_f32_x(full_predicate_b32, sum_norms, intersection);
                    svbool_t nonzero = svcmpne_f32(full_predicate_b32, union_val, zero_f32_vec);
                    svfloat32_t ratio = svdiv_f32_x(full_predicate_b32, intersection, union_val);
                    svfloat32_t jaccard = svsel_f32(nonzero, svsub_f32_x(full_predicate_b32, one_f32_vec, ratio),
                                                    one_f32_vec);
                    svst1_f32(full_predicate_b32, c_row + (column_tile_index + 1) * tile_dim, jaccard);
                }
                // ZA3
                {
                    svuint32_t za3 = svread_hor_za32_u32_m(svdup_u32(0), full_predicate_b32, 3, row);
                    svfloat32_t matching = svcvt_f32_u32_x(full_predicate_b32, za3);
                    svfloat32_t sum_norms = svadd_f32_x(full_predicate_b32, norm_a_f32, b_norms_f32_2);
                    svfloat32_t intersection = svmul_f32_x(
                        full_predicate_b32,
                        svadd_f32_x(full_predicate_b32, svsub_f32_x(full_predicate_b32, sum_norms, depth_f32_vec),
                                    matching),
                        half_f32_vec);
                    svfloat32_t union_val = svsub_f32_x(full_predicate_b32, sum_norms, intersection);
                    svbool_t nonzero = svcmpne_f32(full_predicate_b32, union_val, zero_f32_vec);
                    svfloat32_t ratio = svdiv_f32_x(full_predicate_b32, intersection, union_val);
                    svfloat32_t jaccard = svsel_f32(nonzero, svsub_f32_x(full_predicate_b32, one_f32_vec, ratio),
                                                    one_f32_vec);
                    svst1_f32(full_predicate_b32, c_row + (column_tile_index + 2) * tile_dim, jaccard);
                }
            }
        }

        // Remainder: 1 column tile at a time using ZA1
        for (; column_tile_index < column_tile_count; column_tile_index++) {
            nk_size_t const col_tile_start = column_tile_index * tile_dim;
            nk_size_t const cols_remaining = (col_tile_start + tile_dim <= n_vectors) ? tile_dim
                                                                                      : (n_vectors - col_tile_start);
            svbool_t const column_predicate_b32 = svwhilelt_b32((nk_u32_t)0, (nk_u32_t)cols_remaining);

            svzero_mask_za(nk_sme_zero_za32_tile_1_);

            for (nk_size_t d_tile = 0; d_tile < depth_tile_count; d_tile++) {
                nk_size_t const d_start_u32 = d_tile * depth_tile_size;
                nk_size_t const u32s_this_tile = (d_start_u32 + depth_tile_size <= depth_u32_total)
                                                     ? depth_tile_size
                                                     : (depth_u32_total > d_start_u32 ? depth_u32_total - d_start_u32
                                                                                      : 0);
                if (u32s_this_tile == 0) break;

                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                svbool_t const batch_predicate_b32 = svwhilelt_b32((nk_u32_t)0, (nk_u32_t)u32s_this_tile);

                // Load A rows into ZA0 horizontally
                for (nk_size_t row_in_tile = 0; row_in_tile < rows_clamped; row_in_tile++) {
                    nk_u32_t const *a_row_u32 = (nk_u32_t const *)((char const *)vectors +
                                                                   (row_tile_start + row_in_tile) * stride) +
                                                d_start_u32;
                    svld1_hor_za32(0, row_in_tile, batch_predicate_b32, a_row_u32);
                }

                // Save A columns from ZA0 to stack buffer
                for (nk_size_t s = 0; s < u32s_this_tile; s++)
                    svst1_u32(full_predicate_b32, a_buffer[s],
                              svread_ver_za32_u32_m(svdup_u32(0), row_predicate_b32, 0, s));

                // Load B column tile into ZA0
                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t col = 0; col < tile_dim; col++) {
                    nk_size_t const col_abs = col_tile_start + col;
                    if (col_abs < n_vectors) {
                        nk_u32_t const *b_row = (nk_u32_t const *)((char const *)vectors + col_abs * stride) +
                                                d_start_u32;
                        svld1_hor_za32(0, col, batch_predicate_b32, b_row);
                    }
                }
                for (nk_size_t step = 0; step < u32s_this_tile; step++) {
                    svuint32_t a_vec = svld1_u32(full_predicate_b32, a_buffer[step]);
                    svuint32_t b_vec = svread_ver_za32_u32_m(svdup_u32(0), column_predicate_b32, 0, step);
                    svbmopa_za32_u32_m(1, row_predicate_b32, column_predicate_b32, a_vec, b_vec);
                }
            }

            // Compute B tile norms for remainder tile
            NK_ALIGN64 nk_u32_t b_tile_norms[16];
            for (nk_size_t col = 0; col < tile_dim; col++) {
                nk_size_t const col_abs = col_tile_start + col;
                b_tile_norms[col] = (col_abs < n_vectors)
                                        ? nk_popcount_vector_streaming_(
                                              (nk_u1x8_t const *)((char const *)vectors + col_abs * stride),
                                              depth_in_bytes)
                                        : 0;
            }

            svfloat32_t b_norms_f32 = svcvt_f32_u32_x(full_predicate_b32, svld1_u32(full_predicate_b32, b_tile_norms));
            for (nk_size_t row = 0; row < rows_clamped; row++) {
                svuint32_t za1 = svread_hor_za32_u32_m(svdup_u32(0), full_predicate_b32, 1, row);
                svfloat32_t matching = svcvt_f32_u32_x(full_predicate_b32, za1);
                svfloat32_t norm_a_f32 = svdup_f32(a_tile_norms[row]);
                svfloat32_t sum_norms = svadd_f32_x(full_predicate_b32, norm_a_f32, b_norms_f32);
                svfloat32_t intersection = svmul_f32_x(
                    full_predicate_b32,
                    svadd_f32_x(full_predicate_b32, svsub_f32_x(full_predicate_b32, sum_norms, depth_f32_vec),
                                matching),
                    half_f32_vec);
                svfloat32_t union_val = svsub_f32_x(full_predicate_b32, sum_norms, intersection);
                svbool_t nonzero = svcmpne_f32(full_predicate_b32, union_val, zero_f32_vec);
                svfloat32_t ratio = svdiv_f32_x(full_predicate_b32, intersection, union_val);
                svfloat32_t jaccard = svsel_f32(nonzero, svsub_f32_x(full_predicate_b32, one_f32_vec, ratio),
                                                one_f32_vec);
                nk_f32_t *c_row = (nk_f32_t *)((char *)result + (row_tile_start + row) * result_stride);
                svst1_f32(column_predicate_b32, c_row + col_tile_start, jaccard);
            }
        }
    }
}

NK_PUBLIC void nk_jaccards_symmetric_u1_smebi32(nk_u1x8_t const *vectors, nk_size_t n_vectors, nk_size_t depth_bits,
                                                nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                nk_size_t row_start, nk_size_t row_count) {
    nk_jaccards_symmetric_u1_smebi32_kernel_(vectors, n_vectors, depth_bits, stride, result, result_stride, row_start,
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

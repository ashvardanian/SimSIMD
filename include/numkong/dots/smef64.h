/**
 *  @brief SIMD-accelerated Batched Dot Products for SME F64.
 *  @file include/numkong/dots/smef64.h
 *  @author Ash Vardanian
 *  @date January 2, 2026
 *
 *  @sa include/numkong/dots.h
 *
 *  Uses ARM SME with `FEAT_SME_F64F64` for high-precision GEMM.
 *  Requires Apple M4 or equivalent with `f64` outer product support.
 *
 *  Provides `f32` and `f64` GEMM using `ZA64` tiles:
 *  - `f32` inputs with `f64` accumulation: higher precision than `ZA32`
 *  - Native `f64` GEMM with Kahan compensated summation
 *
 *  Tile dimensions for SVL=512 (Apple M4):
 *  - `ZA64` tile: 8 × 8 `f64` elements (512B)
 *  - `f64` vectors: 8 elements per SVE vector
 *  - `f32` vectors: 16 elements per SVE vector, converted to `f64`
 *
 *  Key instructions:
 *  - `svmopa_za64_f64_m` / `FMOPA`: `f64` outer product, 16cy amortized
 *  - `svcvt_f64_f32_x` / `FCVT`: `f32` → `f64` conversion
 *  - `svwrite_hor_za64_f64_m` / `MOVA`: direct Z → ZA tile write (no bounce buffer)
 */
#ifndef NK_DOTS_SMEF64_H
#define NK_DOTS_SMEF64_H

#if NK_TARGET_ARM_
#if NK_TARGET_SME

#include "numkong/types.h"
#include "numkong/dots/sme.h" // `nk_dots_sme_packed_header_t`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("sme,sme-f64f64"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("+sme+sme-f64f64")
#endif

/*
 *  f32 → f64 → f32 GEMM using FMOPA with ZA64 tiles (FEAT_SME_F64F64).
 *
 *  Tile layout (SVL=512, Apple M4):
 *  - ZA64 output tile: 8 × 8 f64 elements (512 B)
 *  - f32 input vectors: 16 elements (SVL/32), converted to f64 in chunks of 8
 *  - Depth sub-loop: processes 8 f32 values per iteration (→ 8 f64)
 *  - FMOPA predicates: b64 (f64 output granularity)
 *  - f32 load predicates: b32 (f32 input granularity)
 *  - 4-tile path: ZA0-ZA3 process 4 column tiles simultaneously
 *  - Output: f64 results converted back to f32 via svcvt_f32_f64
 *
 *  Non-widening alternative (FEAT_SME_F32F32, `svmopa_za32_f32_m`): ZA32 tiles are 16×16
 *  (4× area vs ZA64 8×8) with no f32↔f64 conversion, offering ~3-4× raw throughput. However,
 *  ZA32 and ZA64 tiles alias physically (ZA0.S overlaps ZA0.D+ZA1.D), so a periodic flush to
 *  f64 stack accumulators would be needed for precision above f32 — erasing most speedup.
 *  Pure f32 accumulation (no flush) provides only f32 precision, which is already served by
 *  the f16 → f32 GEMM path for reduced-precision workloads. This f64 path exists specifically
 *  for higher-than-f32 accumulation precision; replacing it with f32 FMOPA would be
 *  counterproductive. Apple M4 has `hw.optional.arm.SME_F32F32: 1` but we don't use it here.
 */
#pragma region Single Precision Floats

NK_PUBLIC nk_size_t nk_dots_packed_size_f32_smef64(nk_size_t columns, nk_size_t depth) {
    nk_size_t const tile_dimension = svcntsd();  // rows per `ZA64` tile (8 for SVL=512)
    nk_size_t const depth_tile_size = svcntsw(); // `f32` depth elements per tile (16 for SVL=512)

    nk_size_t const column_tile_count = nk_size_divide_round_up_(columns, tile_dimension);
    nk_size_t const depth_tile_count = nk_size_divide_round_up_(depth, depth_tile_size);

    nk_size_t size = sizeof(nk_dots_sme_packed_header_t);
    size += column_tile_count * depth_tile_count * tile_dimension * depth_tile_size * sizeof(nk_f32_t);

    return size;
}

NK_PUBLIC void nk_dots_pack_f32_smef64(nk_f32_t const *b, nk_size_t columns, nk_size_t depth, nk_size_t b_stride,
                                       void *b_packed) {

    nk_size_t const tile_dimension = svcntsd();                       // rows per `ZA64` tile (8 for SVL=512)
    nk_size_t const depth_tile_size = svcntsw();                      // `f32` depth elements per tile (16 for SVL=512)
    nk_size_t const tile_elements = tile_dimension * depth_tile_size; // 128
    nk_size_t const b_stride_elements = b_stride / sizeof(nk_f32_t);

    nk_size_t const column_tile_count = nk_size_divide_round_up_(columns, tile_dimension);
    nk_size_t const depth_tile_count = nk_size_divide_round_up_(depth, depth_tile_size);
    nk_size_t const total_tiles = column_tile_count * depth_tile_count;

    // Store actual dimensions and tile counts in header
    nk_dots_sme_packed_header_t *header = (nk_dots_sme_packed_header_t *)b_packed;
    header->column_tile_count = (nk_u32_t)column_tile_count;
    header->depth_tile_count = (nk_u32_t)depth_tile_count;
    header->columns = (nk_u32_t)columns;
    header->depth = (nk_u32_t)depth;
    header->svl_bytes = (nk_u32_t)svcntsb(); // streaming vector length in bytes

    nk_f32_t *tiles = (nk_f32_t *)((char *)b_packed + sizeof(nk_dots_sme_packed_header_t));

    // Zero-initialize all tiles (handles partial tile padding)
    nk_size_t const total_elements = total_tiles * tile_elements;
    for (nk_size_t i = 0; i < total_elements; i++) tiles[i] = 0.0f;

    // Pack data into tiles with depth-major layout within each tile:
    // dst_idx = depth_idx * tile_dimension + column_idx
    // This allows loading one B vector per depth step: svld1(b_tile + k * tile_dimension)
    for (nk_size_t column_tile_idx = 0; column_tile_idx < column_tile_count; column_tile_idx++) {
        for (nk_size_t depth_tile_idx = 0; depth_tile_idx < depth_tile_count; depth_tile_idx++) {
            nk_size_t const tile_index = column_tile_idx * depth_tile_count + depth_tile_idx;
            nk_f32_t *tile_output = tiles + tile_index * tile_elements;

            nk_size_t const src_row_start = column_tile_idx * tile_dimension;
            nk_size_t const src_column_start = depth_tile_idx * depth_tile_size;

            // Handle partial tiles at edges
            nk_size_t const rows_to_pack = (src_row_start + tile_dimension <= columns) ? tile_dimension
                                                                                       : (columns - src_row_start);
            nk_size_t const columns_to_pack = (src_column_start + depth_tile_size <= depth)
                                                  ? depth_tile_size
                                                  : (depth - src_column_start);

            for (nk_size_t column_idx = 0; column_idx < rows_to_pack; column_idx++) {
                for (nk_size_t depth_idx = 0; depth_idx < columns_to_pack; depth_idx++) {
                    nk_size_t const src_idx = (src_row_start + column_idx) * b_stride_elements + src_column_start +
                                              depth_idx;
                    nk_size_t const dst_idx = depth_idx * tile_dimension + column_idx;
                    tile_output[dst_idx] = b[src_idx];
                }
            }
        }
    }
}

__arm_locally_streaming __arm_new("za") static void nk_dots_packed_f32_smef64_kernel_(
    nk_f32_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows, nk_size_t columns, nk_size_t depth,
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) {

    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_size_t const column_tile_count = header->column_tile_count;
    nk_size_t const depth_tile_count = header->depth_tile_count;

    nk_size_t const tile_dimension = svcntd();  // 8 for 512-bit SVL
    nk_size_t const depth_tile_size = svcntw(); // 16 for 512-bit SVL
    nk_size_t const tile_elements = tile_dimension * depth_tile_size;
    nk_size_t const depth_steps_per_batch = tile_dimension; // 8 depth steps per ZA0.D load

    nk_f32_t const *b_tiles = (nk_f32_t const *)((char const *)b_packed + sizeof(nk_dots_sme_packed_header_t));

    svbool_t const full_predicate_b64 = svptrue_b64();

    // ZA0.D = staging, ZA1-7.D = accumulation (7-tile fast path)
    for (nk_size_t row_tile_index = 0; row_tile_index < nk_size_divide_round_up_(rows, tile_dimension);
         row_tile_index++) {
        nk_size_t const row_start = row_tile_index * tile_dimension;
        nk_size_t const rows_remaining = (row_start + tile_dimension <= rows) ? tile_dimension : (rows - row_start);
        svbool_t const row_predicate_b64 = svwhilelt_b64((uint64_t)0, rows_remaining);

        nk_size_t column_tile_index = 0;

        // Fast path: 7 column tiles using ZA1-ZA7 (ZA0.D = staging)
        for (; column_tile_index + 7 <= column_tile_count; column_tile_index += 7) {
            svzero_mask_za(nk_sme_zero_za64_tiles_1_7_);

            for (nk_size_t depth_tile_idx = 0; depth_tile_idx < depth_tile_count; depth_tile_idx++) {
                nk_size_t const depth_offset = depth_tile_idx * depth_tile_size;

                // Process depth_tile_size elements in batches of tile_dimension (8)
                for (nk_size_t depth_batch_start = 0; depth_batch_start < depth_tile_size;
                     depth_batch_start += depth_steps_per_batch) {
                    nk_size_t const depth_batch_end = (depth_batch_start + depth_steps_per_batch < depth_tile_size)
                                                          ? depth_batch_start + depth_steps_per_batch
                                                          : depth_tile_size;
                    nk_size_t const batch_size = depth_batch_end - depth_batch_start;

                    // Check if any elements in this batch are valid
                    if (depth_offset + depth_batch_start >= depth) break;

                    svzero_mask_za(nk_sme_zero_za64_tile_0_);

                    // Load A rows into ZA0.D: extending load f32→u64 + convert to f64
                    svbool_t const batch_predicate_b64 = svwhilelt_b64((uint64_t)0, (uint64_t)batch_size);
                    svbool_t const a_depth_predicate_b64 = svwhilelt_b64((uint64_t)(depth_offset + depth_batch_start),
                                                                         (uint64_t)depth);
                    for (nk_size_t row_in_tile = 0; row_in_tile < rows_remaining; row_in_tile++) {
                        nk_size_t const a_row = row_start + row_in_tile;
                        // Extending load: svld1uw_u64 loads f32 bits into lower 32 of each u64 lane
                        svfloat64_t a_row_widened_f64 = svcvt_f64_f32_x(
                            batch_predicate_b64,
                            svreinterpret_f32_u64(svld1uw_u64(
                                a_depth_predicate_b64,
                                (nk_u32_t const *)&a[a_row * a_stride_elements + depth_offset + depth_batch_start])));
                        svwrite_hor_za64_f64_m(0, row_in_tile, batch_predicate_b64, a_row_widened_f64);
                    }

                    // Vertical read + MOPA for each depth step in batch
                    for (nk_size_t step = 0; step < batch_size; step++) {
                        nk_size_t const k_abs = depth_offset + depth_batch_start + step;
                        if (k_abs >= depth) break;

                        svfloat64_t vector_a = svread_ver_za64_f64_m(svdup_f64(0.0), row_predicate_b64, 0, step);

                        nk_size_t const b_k = depth_batch_start + step;

                        // Extending load f32→u64 + convert to f64: svld1uw_u64 replaces svld1_f32 + svunpklo_u64
                        svfloat64_t b_column_tile_1_f64 = svcvt_f64_f32_x(
                            full_predicate_b64,
                            svreinterpret_f32_u64(svld1uw_u64(
                                full_predicate_b64,
                                (nk_u32_t const *)(b_tiles +
                                                   ((column_tile_index + 0) * depth_tile_count + depth_tile_idx) *
                                                       tile_elements +
                                                   b_k * tile_dimension))));
                        svfloat64_t b_column_tile_2_f64 = svcvt_f64_f32_x(
                            full_predicate_b64,
                            svreinterpret_f32_u64(svld1uw_u64(
                                full_predicate_b64,
                                (nk_u32_t const *)(b_tiles +
                                                   ((column_tile_index + 1) * depth_tile_count + depth_tile_idx) *
                                                       tile_elements +
                                                   b_k * tile_dimension))));
                        svfloat64_t b_column_tile_3_f64 = svcvt_f64_f32_x(
                            full_predicate_b64,
                            svreinterpret_f32_u64(svld1uw_u64(
                                full_predicate_b64,
                                (nk_u32_t const *)(b_tiles +
                                                   ((column_tile_index + 2) * depth_tile_count + depth_tile_idx) *
                                                       tile_elements +
                                                   b_k * tile_dimension))));
                        svfloat64_t b_column_tile_4_f64 = svcvt_f64_f32_x(
                            full_predicate_b64,
                            svreinterpret_f32_u64(svld1uw_u64(
                                full_predicate_b64,
                                (nk_u32_t const *)(b_tiles +
                                                   ((column_tile_index + 3) * depth_tile_count + depth_tile_idx) *
                                                       tile_elements +
                                                   b_k * tile_dimension))));
                        svfloat64_t b_column_tile_5_f64 = svcvt_f64_f32_x(
                            full_predicate_b64,
                            svreinterpret_f32_u64(svld1uw_u64(
                                full_predicate_b64,
                                (nk_u32_t const *)(b_tiles +
                                                   ((column_tile_index + 4) * depth_tile_count + depth_tile_idx) *
                                                       tile_elements +
                                                   b_k * tile_dimension))));
                        svfloat64_t b_column_tile_6_f64 = svcvt_f64_f32_x(
                            full_predicate_b64,
                            svreinterpret_f32_u64(svld1uw_u64(
                                full_predicate_b64,
                                (nk_u32_t const *)(b_tiles +
                                                   ((column_tile_index + 5) * depth_tile_count + depth_tile_idx) *
                                                       tile_elements +
                                                   b_k * tile_dimension))));
                        svfloat64_t b_column_tile_7_f64 = svcvt_f64_f32_x(
                            full_predicate_b64,
                            svreinterpret_f32_u64(svld1uw_u64(
                                full_predicate_b64,
                                (nk_u32_t const *)(b_tiles +
                                                   ((column_tile_index + 6) * depth_tile_count + depth_tile_idx) *
                                                       tile_elements +
                                                   b_k * tile_dimension))));

                        svmopa_za64_f64_m(1, row_predicate_b64, full_predicate_b64, vector_a, b_column_tile_1_f64);
                        svmopa_za64_f64_m(2, row_predicate_b64, full_predicate_b64, vector_a, b_column_tile_2_f64);
                        svmopa_za64_f64_m(3, row_predicate_b64, full_predicate_b64, vector_a, b_column_tile_3_f64);
                        svmopa_za64_f64_m(4, row_predicate_b64, full_predicate_b64, vector_a, b_column_tile_4_f64);
                        svmopa_za64_f64_m(5, row_predicate_b64, full_predicate_b64, vector_a, b_column_tile_5_f64);
                        svmopa_za64_f64_m(6, row_predicate_b64, full_predicate_b64, vector_a, b_column_tile_6_f64);
                        svmopa_za64_f64_m(7, row_predicate_b64, full_predicate_b64, vector_a, b_column_tile_7_f64);
                    }
                }
            }

            // Extract from ZA1-7: narrow f64→f32 with svuzp1_u32 packing recipe
            svbool_t const predicate_tile_b32 = svwhilelt_b32((uint32_t)0, (uint32_t)tile_dimension);
            for (nk_size_t row_idx = 0; row_idx < rows_remaining; row_idx++) {
                nk_f32_t *c_row = c + (row_start + row_idx) * c_stride_elements;

                // Narrowing f64→f32 recipe: MOVA ZA→Z (no bounce buffer), svcvt, svuzp1 packs consecutively
                svfloat64_t za_row_f64 = svread_hor_za64_f64_m(svdup_f64(0), full_predicate_b64, 1, row_idx);
                svfloat32_t za_row_narrowed_f32 = svcvt_f32_f64_x(full_predicate_b64, za_row_f64);
                svfloat32_t za_row_packed_f32 = svreinterpret_f32_u32(
                    svuzp1_u32(svreinterpret_u32_f32(za_row_narrowed_f32), svreinterpret_u32_f32(za_row_narrowed_f32)));
                svst1_f32(predicate_tile_b32, c_row + (column_tile_index + 0) * tile_dimension, za_row_packed_f32);

                za_row_f64 = svread_hor_za64_f64_m(svdup_f64(0), full_predicate_b64, 2, row_idx);
                za_row_narrowed_f32 = svcvt_f32_f64_x(full_predicate_b64, za_row_f64);
                za_row_packed_f32 = svreinterpret_f32_u32(
                    svuzp1_u32(svreinterpret_u32_f32(za_row_narrowed_f32), svreinterpret_u32_f32(za_row_narrowed_f32)));
                svst1_f32(predicate_tile_b32, c_row + (column_tile_index + 1) * tile_dimension, za_row_packed_f32);

                za_row_f64 = svread_hor_za64_f64_m(svdup_f64(0), full_predicate_b64, 3, row_idx);
                za_row_narrowed_f32 = svcvt_f32_f64_x(full_predicate_b64, za_row_f64);
                za_row_packed_f32 = svreinterpret_f32_u32(
                    svuzp1_u32(svreinterpret_u32_f32(za_row_narrowed_f32), svreinterpret_u32_f32(za_row_narrowed_f32)));
                svst1_f32(predicate_tile_b32, c_row + (column_tile_index + 2) * tile_dimension, za_row_packed_f32);

                za_row_f64 = svread_hor_za64_f64_m(svdup_f64(0), full_predicate_b64, 4, row_idx);
                za_row_narrowed_f32 = svcvt_f32_f64_x(full_predicate_b64, za_row_f64);
                za_row_packed_f32 = svreinterpret_f32_u32(
                    svuzp1_u32(svreinterpret_u32_f32(za_row_narrowed_f32), svreinterpret_u32_f32(za_row_narrowed_f32)));
                svst1_f32(predicate_tile_b32, c_row + (column_tile_index + 3) * tile_dimension, za_row_packed_f32);

                za_row_f64 = svread_hor_za64_f64_m(svdup_f64(0), full_predicate_b64, 5, row_idx);
                za_row_narrowed_f32 = svcvt_f32_f64_x(full_predicate_b64, za_row_f64);
                za_row_packed_f32 = svreinterpret_f32_u32(
                    svuzp1_u32(svreinterpret_u32_f32(za_row_narrowed_f32), svreinterpret_u32_f32(za_row_narrowed_f32)));
                svst1_f32(predicate_tile_b32, c_row + (column_tile_index + 4) * tile_dimension, za_row_packed_f32);

                za_row_f64 = svread_hor_za64_f64_m(svdup_f64(0), full_predicate_b64, 6, row_idx);
                za_row_narrowed_f32 = svcvt_f32_f64_x(full_predicate_b64, za_row_f64);
                za_row_packed_f32 = svreinterpret_f32_u32(
                    svuzp1_u32(svreinterpret_u32_f32(za_row_narrowed_f32), svreinterpret_u32_f32(za_row_narrowed_f32)));
                svst1_f32(predicate_tile_b32, c_row + (column_tile_index + 5) * tile_dimension, za_row_packed_f32);

                za_row_f64 = svread_hor_za64_f64_m(svdup_f64(0), full_predicate_b64, 7, row_idx);
                za_row_narrowed_f32 = svcvt_f32_f64_x(full_predicate_b64, za_row_f64);
                za_row_packed_f32 = svreinterpret_f32_u32(
                    svuzp1_u32(svreinterpret_u32_f32(za_row_narrowed_f32), svreinterpret_u32_f32(za_row_narrowed_f32)));
                svst1_f32(predicate_tile_b32, c_row + (column_tile_index + 6) * tile_dimension, za_row_packed_f32);
            }
        }

        // Remainder: 1 column tile at a time using ZA1
        for (; column_tile_index < column_tile_count; column_tile_index++) {
            nk_size_t const column_start = column_tile_index * tile_dimension;
            nk_size_t const columns_remaining = (column_start + tile_dimension <= columns) ? tile_dimension
                                                                                           : (columns - column_start);
            svbool_t const column_predicate_b64 = svwhilelt_b64((uint64_t)0, columns_remaining);

            svzero_mask_za(nk_sme_zero_za64_tile_1_);

            for (nk_size_t depth_tile_idx = 0; depth_tile_idx < depth_tile_count; depth_tile_idx++) {
                nk_size_t const depth_offset = depth_tile_idx * depth_tile_size;

                for (nk_size_t depth_batch_start = 0; depth_batch_start < depth_tile_size;
                     depth_batch_start += depth_steps_per_batch) {
                    nk_size_t const depth_batch_end = (depth_batch_start + depth_steps_per_batch < depth_tile_size)
                                                          ? depth_batch_start + depth_steps_per_batch
                                                          : depth_tile_size;
                    nk_size_t const batch_size = depth_batch_end - depth_batch_start;

                    if (depth_offset + depth_batch_start >= depth) break;

                    svzero_mask_za(nk_sme_zero_za64_tile_0_);

                    svbool_t const batch_predicate_b64 = svwhilelt_b64((uint64_t)0, (uint64_t)batch_size);
                    svbool_t const a_depth_pred_b64 = svwhilelt_b64((uint64_t)(depth_offset + depth_batch_start),
                                                                    (uint64_t)depth);
                    for (nk_size_t row_in_tile = 0; row_in_tile < rows_remaining; row_in_tile++) {
                        nk_size_t const a_row = row_start + row_in_tile;
                        svfloat64_t a_row_widened_f64 = svcvt_f64_f32_x(
                            batch_predicate_b64,
                            svreinterpret_f32_u64(svld1uw_u64(
                                a_depth_pred_b64,
                                (nk_u32_t const *)&a[a_row * a_stride_elements + depth_offset + depth_batch_start])));
                        svwrite_hor_za64_f64_m(0, row_in_tile, batch_predicate_b64, a_row_widened_f64);
                    }

                    for (nk_size_t step = 0; step < batch_size; step++) {
                        nk_size_t const k_abs = depth_offset + depth_batch_start + step;
                        if (k_abs >= depth) break;

                        svfloat64_t vector_a = svread_ver_za64_f64_m(svdup_f64(0.0), row_predicate_b64, 0, step);

                        nk_size_t const b_k = depth_batch_start + step;
                        nk_f32_t const *b_tile = b_tiles + (column_tile_index * depth_tile_count + depth_tile_idx) *
                                                               tile_elements;
                        // Extending load f32→u64 + convert to f64
                        svfloat64_t vector_b = svcvt_f64_f32_x(
                            full_predicate_b64,
                            svreinterpret_f32_u64(
                                svld1uw_u64(full_predicate_b64, (nk_u32_t const *)(b_tile + b_k * tile_dimension))));

                        svmopa_za64_f64_m(1, row_predicate_b64, column_predicate_b64, vector_a, vector_b);
                    }
                }
            }

            // Narrowing f64→f32 with svuzp1_u32 packing recipe
            svbool_t const column_predicate_b32 = svwhilelt_b32((uint32_t)0, (uint32_t)columns_remaining);
            for (nk_size_t row_idx = 0; row_idx < rows_remaining; row_idx++) {
                svfloat64_t za_row = svread_hor_za64_f64_m(svdup_f64(0), full_predicate_b64, 1, row_idx);
                svfloat32_t za_row_narrowed = svcvt_f32_f64_x(full_predicate_b64, za_row);
                svfloat32_t za_row_packed = svreinterpret_f32_u32(
                    svuzp1_u32(svreinterpret_u32_f32(za_row_narrowed), svreinterpret_u32_f32(za_row_narrowed)));
                nk_f32_t *c_row = c + (row_start + row_idx) * c_stride_elements + column_start;
                svst1_f32(column_predicate_b32, c_row, za_row_packed);
            }
        }
    }
}

NK_PUBLIC void nk_dots_packed_f32_smef64(nk_f32_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows,
                                         nk_size_t columns, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride) {

    nk_size_t const a_stride_elements = a_stride / sizeof(nk_f32_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);

    nk_dots_packed_f32_smef64_kernel_(a, b_packed, c, rows, columns, depth, a_stride_elements, c_stride_elements);
}

/**
 *  `f32` × `f32` → `f32` symmetric kernel using MOPA self-GEMM with f64 accumulation.
 *  Time-shares ZA0 for both A and B transposition: loads A horizontally,
 *  pre-reads A columns into Z registers, then reloads ZA0 with widened B data
 *  per column tile. Eliminates all scalar B-packing loops.
 */
__arm_locally_streaming __arm_new("za") static void nk_dots_symmetric_f32_smef64_kernel_(
    nk_f32_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride_elements, nk_f32_t *result,
    nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) {

    nk_size_t const tile_dimension = svcntd();              // 8 for SVL=512
    nk_size_t const depth_tile_size = svcntw();             // 16 for SVL=512
    nk_size_t const depth_steps_per_batch = tile_dimension; // 8

    svbool_t const full_predicate_b64 = svptrue_b64();

    NK_ALIGN64 nk_f64_t a_buffer[8][8];

    nk_size_t const row_end = row_start + row_count;
    nk_size_t const column_tile_count = nk_size_divide_round_up_(n_vectors, tile_dimension);
    nk_size_t const depth_tile_count = nk_size_divide_round_up_(depth, depth_tile_size);

    for (nk_size_t row_tile_start = row_start; row_tile_start < row_end && row_tile_start < n_vectors;
         row_tile_start += tile_dimension) {
        nk_size_t const rows_clamped = (row_tile_start + tile_dimension <= row_end) ? tile_dimension
                                                                                    : (row_end - row_tile_start);
        nk_size_t const rows_actual = (row_tile_start + rows_clamped <= n_vectors) ? rows_clamped
                                                                                   : (n_vectors - row_tile_start);
        svbool_t const row_predicate_b64 = svwhilelt_b64((uint64_t)0, (uint64_t)rows_actual);

        nk_size_t column_tile_index = 0;

        // Fast path: 7 column tiles at a time
        for (; column_tile_index + 7 <= column_tile_count; column_tile_index += 7) {
            svzero_mask_za(nk_sme_zero_za64_tiles_1_7_);

            for (nk_size_t depth_tile_idx = 0; depth_tile_idx < depth_tile_count; depth_tile_idx++) {
                nk_size_t const depth_offset = depth_tile_idx * depth_tile_size;

                // Process depth_tile_size in batches of depth_steps_per_batch (8)
                for (nk_size_t depth_batch_start = 0; depth_batch_start < depth_tile_size;
                     depth_batch_start += depth_steps_per_batch) {
                    nk_size_t const depth_batch_end = (depth_batch_start + depth_steps_per_batch < depth_tile_size)
                                                          ? depth_batch_start + depth_steps_per_batch
                                                          : depth_tile_size;
                    nk_size_t const batch_size = depth_batch_end - depth_batch_start;

                    if (depth_offset + depth_batch_start >= depth) break;

                    // ZA transpose for A rows: extending load f32→f64, MOVA directly into ZA0
                    svbool_t const batch_predicate_b64 = svwhilelt_b64((uint64_t)0, (uint64_t)batch_size);
                    svbool_t const a_depth_predicate_b64 = svwhilelt_b64((uint64_t)(depth_offset + depth_batch_start),
                                                                         (uint64_t)depth);
                    svzero_mask_za(nk_sme_zero_za64_tile_0_);
                    for (nk_size_t row_in_tile = 0; row_in_tile < rows_actual; row_in_tile++) {
                        nk_size_t const row_abs = row_tile_start + row_in_tile;
                        svfloat64_t a_row_widened_f64 = svcvt_f64_f32_x(
                            batch_predicate_b64,
                            svreinterpret_f32_u64(svld1uw_u64(
                                a_depth_predicate_b64, (nk_u32_t const *)&vectors[row_abs * stride_elements +
                                                                                  depth_offset + depth_batch_start])));
                        svwrite_hor_za64_f64_m(0, row_in_tile, batch_predicate_b64, a_row_widened_f64);
                    }

                    // Save A columns from ZA0 to stack buffer
                    for (nk_size_t s = 0; s < batch_size; s++)
                        svst1_f64(full_predicate_b64, a_buffer[s],
                                  svread_ver_za64_f64_m(svdup_f64(0), row_predicate_b64, 0, s));

                    // Column tile 0 → ZA1 via MOVA
                    svzero_mask_za(nk_sme_zero_za64_tile_0_);
                    for (nk_size_t column = 0; column < tile_dimension; column++) {
                        nk_size_t const column_abs = (column_tile_index + 0) * tile_dimension + column;
                        if (column_abs < n_vectors) {
                            svfloat64_t widened = svcvt_f64_f32_x(
                                batch_predicate_b64,
                                svreinterpret_f32_u64(svld1uw_u64(
                                    a_depth_predicate_b64,
                                    (nk_u32_t const
                                         *)&vectors[column_abs * stride_elements + depth_offset + depth_batch_start])));
                            svwrite_hor_za64_f64_m(0, column, batch_predicate_b64, widened);
                        }
                    }
                    for (nk_size_t step = 0; step < batch_size; step++) {
                        svfloat64_t a_vec = svld1_f64(full_predicate_b64, a_buffer[step]);
                        svfloat64_t b_vec = svread_ver_za64_f64_m(svdup_f64(0.0), full_predicate_b64, 0, step);
                        svmopa_za64_f64_m(1, row_predicate_b64, full_predicate_b64, a_vec, b_vec);
                    }

                    // Column tile 1 → ZA2 via MOVA
                    svzero_mask_za(nk_sme_zero_za64_tile_0_);
                    for (nk_size_t column = 0; column < tile_dimension; column++) {
                        nk_size_t const column_abs = (column_tile_index + 1) * tile_dimension + column;
                        if (column_abs < n_vectors) {
                            svfloat64_t widened = svcvt_f64_f32_x(
                                batch_predicate_b64,
                                svreinterpret_f32_u64(svld1uw_u64(
                                    a_depth_predicate_b64,
                                    (nk_u32_t const
                                         *)&vectors[column_abs * stride_elements + depth_offset + depth_batch_start])));
                            svwrite_hor_za64_f64_m(0, column, batch_predicate_b64, widened);
                        }
                    }
                    for (nk_size_t step = 0; step < batch_size; step++) {
                        svfloat64_t a_vec = svld1_f64(full_predicate_b64, a_buffer[step]);
                        svfloat64_t b_vec = svread_ver_za64_f64_m(svdup_f64(0.0), full_predicate_b64, 0, step);
                        svmopa_za64_f64_m(2, row_predicate_b64, full_predicate_b64, a_vec, b_vec);
                    }

                    // Column tile 2 → ZA3 via MOVA
                    svzero_mask_za(nk_sme_zero_za64_tile_0_);
                    for (nk_size_t column = 0; column < tile_dimension; column++) {
                        nk_size_t const column_abs = (column_tile_index + 2) * tile_dimension + column;
                        if (column_abs < n_vectors) {
                            svfloat64_t widened = svcvt_f64_f32_x(
                                batch_predicate_b64,
                                svreinterpret_f32_u64(svld1uw_u64(
                                    a_depth_predicate_b64,
                                    (nk_u32_t const
                                         *)&vectors[column_abs * stride_elements + depth_offset + depth_batch_start])));
                            svwrite_hor_za64_f64_m(0, column, batch_predicate_b64, widened);
                        }
                    }
                    for (nk_size_t step = 0; step < batch_size; step++) {
                        svfloat64_t a_vec = svld1_f64(full_predicate_b64, a_buffer[step]);
                        svfloat64_t b_vec = svread_ver_za64_f64_m(svdup_f64(0.0), full_predicate_b64, 0, step);
                        svmopa_za64_f64_m(3, row_predicate_b64, full_predicate_b64, a_vec, b_vec);
                    }

                    // Column tile 3 → ZA4 via MOVA
                    svzero_mask_za(nk_sme_zero_za64_tile_0_);
                    for (nk_size_t column = 0; column < tile_dimension; column++) {
                        nk_size_t const column_abs = (column_tile_index + 3) * tile_dimension + column;
                        if (column_abs < n_vectors) {
                            svfloat64_t widened = svcvt_f64_f32_x(
                                batch_predicate_b64,
                                svreinterpret_f32_u64(svld1uw_u64(
                                    a_depth_predicate_b64,
                                    (nk_u32_t const
                                         *)&vectors[column_abs * stride_elements + depth_offset + depth_batch_start])));
                            svwrite_hor_za64_f64_m(0, column, batch_predicate_b64, widened);
                        }
                    }
                    for (nk_size_t step = 0; step < batch_size; step++) {
                        svfloat64_t a_vec = svld1_f64(full_predicate_b64, a_buffer[step]);
                        svfloat64_t b_vec = svread_ver_za64_f64_m(svdup_f64(0.0), full_predicate_b64, 0, step);
                        svmopa_za64_f64_m(4, row_predicate_b64, full_predicate_b64, a_vec, b_vec);
                    }

                    // Column tile 4 → ZA5 via MOVA
                    svzero_mask_za(nk_sme_zero_za64_tile_0_);
                    for (nk_size_t column = 0; column < tile_dimension; column++) {
                        nk_size_t const column_abs = (column_tile_index + 4) * tile_dimension + column;
                        if (column_abs < n_vectors) {
                            svfloat64_t widened = svcvt_f64_f32_x(
                                batch_predicate_b64,
                                svreinterpret_f32_u64(svld1uw_u64(
                                    a_depth_predicate_b64,
                                    (nk_u32_t const
                                         *)&vectors[column_abs * stride_elements + depth_offset + depth_batch_start])));
                            svwrite_hor_za64_f64_m(0, column, batch_predicate_b64, widened);
                        }
                    }
                    for (nk_size_t step = 0; step < batch_size; step++) {
                        svfloat64_t a_vec = svld1_f64(full_predicate_b64, a_buffer[step]);
                        svfloat64_t b_vec = svread_ver_za64_f64_m(svdup_f64(0.0), full_predicate_b64, 0, step);
                        svmopa_za64_f64_m(5, row_predicate_b64, full_predicate_b64, a_vec, b_vec);
                    }

                    // Column tile 5 → ZA6 via MOVA
                    svzero_mask_za(nk_sme_zero_za64_tile_0_);
                    for (nk_size_t column = 0; column < tile_dimension; column++) {
                        nk_size_t const column_abs = (column_tile_index + 5) * tile_dimension + column;
                        if (column_abs < n_vectors) {
                            svfloat64_t widened = svcvt_f64_f32_x(
                                batch_predicate_b64,
                                svreinterpret_f32_u64(svld1uw_u64(
                                    a_depth_predicate_b64,
                                    (nk_u32_t const
                                         *)&vectors[column_abs * stride_elements + depth_offset + depth_batch_start])));
                            svwrite_hor_za64_f64_m(0, column, batch_predicate_b64, widened);
                        }
                    }
                    for (nk_size_t step = 0; step < batch_size; step++) {
                        svfloat64_t a_vec = svld1_f64(full_predicate_b64, a_buffer[step]);
                        svfloat64_t b_vec = svread_ver_za64_f64_m(svdup_f64(0.0), full_predicate_b64, 0, step);
                        svmopa_za64_f64_m(6, row_predicate_b64, full_predicate_b64, a_vec, b_vec);
                    }

                    // Column tile 6 → ZA7 via MOVA
                    svzero_mask_za(nk_sme_zero_za64_tile_0_);
                    for (nk_size_t column = 0; column < tile_dimension; column++) {
                        nk_size_t const column_abs = (column_tile_index + 6) * tile_dimension + column;
                        if (column_abs < n_vectors) {
                            svfloat64_t widened = svcvt_f64_f32_x(
                                batch_predicate_b64,
                                svreinterpret_f32_u64(svld1uw_u64(
                                    a_depth_predicate_b64,
                                    (nk_u32_t const
                                         *)&vectors[column_abs * stride_elements + depth_offset + depth_batch_start])));
                            svwrite_hor_za64_f64_m(0, column, batch_predicate_b64, widened);
                        }
                    }
                    for (nk_size_t step = 0; step < batch_size; step++) {
                        svfloat64_t a_vec = svld1_f64(full_predicate_b64, a_buffer[step]);
                        svfloat64_t b_vec = svread_ver_za64_f64_m(svdup_f64(0.0), full_predicate_b64, 0, step);
                        svmopa_za64_f64_m(7, row_predicate_b64, full_predicate_b64, a_vec, b_vec);
                    }
                }
            }

            // Extract results: narrow f64→f32 with svuzp1_u32 recipe (unrolled)
            svbool_t const predicate_tile_b32 = svwhilelt_b32((uint32_t)0, (uint32_t)tile_dimension);
            for (nk_size_t row = 0; row < rows_actual; row++) {
                nk_size_t const row_abs = row_tile_start + row;
                nk_f32_t *result_row = result + row_abs * result_stride_elements;

                svfloat64_t za_row_f64 = svread_hor_za64_f64_m(svdup_f64(0), full_predicate_b64, 1, row);
                svfloat32_t za_row_narrowed_f32 = svcvt_f32_f64_x(full_predicate_b64, za_row_f64);
                svfloat32_t za_row_packed_f32 = svreinterpret_f32_u32(
                    svuzp1_u32(svreinterpret_u32_f32(za_row_narrowed_f32), svreinterpret_u32_f32(za_row_narrowed_f32)));
                svst1_f32(predicate_tile_b32, result_row + (column_tile_index + 0) * tile_dimension, za_row_packed_f32);

                za_row_f64 = svread_hor_za64_f64_m(svdup_f64(0), full_predicate_b64, 2, row);
                za_row_narrowed_f32 = svcvt_f32_f64_x(full_predicate_b64, za_row_f64);
                za_row_packed_f32 = svreinterpret_f32_u32(
                    svuzp1_u32(svreinterpret_u32_f32(za_row_narrowed_f32), svreinterpret_u32_f32(za_row_narrowed_f32)));
                svst1_f32(predicate_tile_b32, result_row + (column_tile_index + 1) * tile_dimension, za_row_packed_f32);

                za_row_f64 = svread_hor_za64_f64_m(svdup_f64(0), full_predicate_b64, 3, row);
                za_row_narrowed_f32 = svcvt_f32_f64_x(full_predicate_b64, za_row_f64);
                za_row_packed_f32 = svreinterpret_f32_u32(
                    svuzp1_u32(svreinterpret_u32_f32(za_row_narrowed_f32), svreinterpret_u32_f32(za_row_narrowed_f32)));
                svst1_f32(predicate_tile_b32, result_row + (column_tile_index + 2) * tile_dimension, za_row_packed_f32);

                za_row_f64 = svread_hor_za64_f64_m(svdup_f64(0), full_predicate_b64, 4, row);
                za_row_narrowed_f32 = svcvt_f32_f64_x(full_predicate_b64, za_row_f64);
                za_row_packed_f32 = svreinterpret_f32_u32(
                    svuzp1_u32(svreinterpret_u32_f32(za_row_narrowed_f32), svreinterpret_u32_f32(za_row_narrowed_f32)));
                svst1_f32(predicate_tile_b32, result_row + (column_tile_index + 3) * tile_dimension, za_row_packed_f32);

                za_row_f64 = svread_hor_za64_f64_m(svdup_f64(0), full_predicate_b64, 5, row);
                za_row_narrowed_f32 = svcvt_f32_f64_x(full_predicate_b64, za_row_f64);
                za_row_packed_f32 = svreinterpret_f32_u32(
                    svuzp1_u32(svreinterpret_u32_f32(za_row_narrowed_f32), svreinterpret_u32_f32(za_row_narrowed_f32)));
                svst1_f32(predicate_tile_b32, result_row + (column_tile_index + 4) * tile_dimension, za_row_packed_f32);

                za_row_f64 = svread_hor_za64_f64_m(svdup_f64(0), full_predicate_b64, 6, row);
                za_row_narrowed_f32 = svcvt_f32_f64_x(full_predicate_b64, za_row_f64);
                za_row_packed_f32 = svreinterpret_f32_u32(
                    svuzp1_u32(svreinterpret_u32_f32(za_row_narrowed_f32), svreinterpret_u32_f32(za_row_narrowed_f32)));
                svst1_f32(predicate_tile_b32, result_row + (column_tile_index + 5) * tile_dimension, za_row_packed_f32);

                za_row_f64 = svread_hor_za64_f64_m(svdup_f64(0), full_predicate_b64, 7, row);
                za_row_narrowed_f32 = svcvt_f32_f64_x(full_predicate_b64, za_row_f64);
                za_row_packed_f32 = svreinterpret_f32_u32(
                    svuzp1_u32(svreinterpret_u32_f32(za_row_narrowed_f32), svreinterpret_u32_f32(za_row_narrowed_f32)));
                svst1_f32(predicate_tile_b32, result_row + (column_tile_index + 6) * tile_dimension, za_row_packed_f32);
            }
        }

        // Remainder: 1 column tile at a time
        for (; column_tile_index < column_tile_count; column_tile_index++) {
            nk_size_t const column_tile_start = column_tile_index * tile_dimension;
            nk_size_t const columns_remaining = (column_tile_start + tile_dimension <= n_vectors)
                                                    ? tile_dimension
                                                    : (n_vectors - column_tile_start);
            svbool_t const column_predicate_b64 = svwhilelt_b64((uint64_t)0, (uint64_t)columns_remaining);

            svzero_mask_za(nk_sme_zero_za64_tile_1_);

            for (nk_size_t depth_tile_idx = 0; depth_tile_idx < depth_tile_count; depth_tile_idx++) {
                nk_size_t const depth_offset = depth_tile_idx * depth_tile_size;

                for (nk_size_t depth_batch_start = 0; depth_batch_start < depth_tile_size;
                     depth_batch_start += depth_steps_per_batch) {
                    nk_size_t const depth_batch_end = (depth_batch_start + depth_steps_per_batch < depth_tile_size)
                                                          ? depth_batch_start + depth_steps_per_batch
                                                          : depth_tile_size;
                    nk_size_t const batch_size = depth_batch_end - depth_batch_start;

                    if (depth_offset + depth_batch_start >= depth) break;

                    svbool_t const batch_predicate_b64 = svwhilelt_b64((uint64_t)0, (uint64_t)batch_size);
                    svbool_t const a_depth_pred_b64 = svwhilelt_b64((uint64_t)(depth_offset + depth_batch_start),
                                                                    (uint64_t)depth);
                    svzero_mask_za(nk_sme_zero_za64_tile_0_);
                    for (nk_size_t row_in_tile = 0; row_in_tile < rows_actual; row_in_tile++) {
                        nk_size_t const row_abs = row_tile_start + row_in_tile;
                        svfloat64_t a_row_widened_f64 = svcvt_f64_f32_x(
                            batch_predicate_b64,
                            svreinterpret_f32_u64(svld1uw_u64(
                                a_depth_pred_b64, (nk_u32_t const *)&vectors[row_abs * stride_elements + depth_offset +
                                                                             depth_batch_start])));
                        svwrite_hor_za64_f64_m(0, row_in_tile, batch_predicate_b64, a_row_widened_f64);
                    }

                    // Save A columns from ZA0 to stack buffer
                    for (nk_size_t s = 0; s < batch_size; s++)
                        svst1_f64(full_predicate_b64, a_buffer[s],
                                  svread_ver_za64_f64_m(svdup_f64(0), row_predicate_b64, 0, s));

                    // Load B column tile into ZA0 via MOVA, vertical read + FMOPA into ZA1
                    svzero_mask_za(nk_sme_zero_za64_tile_0_);
                    for (nk_size_t column = 0; column < tile_dimension; column++) {
                        nk_size_t const column_abs = column_tile_start + column;
                        if (column_abs < n_vectors) {
                            svfloat64_t widened = svcvt_f64_f32_x(
                                batch_predicate_b64,
                                svreinterpret_f32_u64(svld1uw_u64(
                                    a_depth_pred_b64, (nk_u32_t const *)&vectors[column_abs * stride_elements +
                                                                                 depth_offset + depth_batch_start])));
                            svwrite_hor_za64_f64_m(0, column, batch_predicate_b64, widened);
                        }
                    }
                    for (nk_size_t step = 0; step < batch_size; step++) {
                        nk_size_t const k_abs = depth_offset + depth_batch_start + step;
                        if (k_abs >= depth) break;
                        svfloat64_t a_vec = svld1_f64(full_predicate_b64, a_buffer[step]);
                        svfloat64_t b_vec = svread_ver_za64_f64_m(svdup_f64(0.0), column_predicate_b64, 0, step);
                        svmopa_za64_f64_m(1, row_predicate_b64, column_predicate_b64, a_vec, b_vec);
                    }
                }
            }

            // Narrow f64→f32 with svuzp1_u32 recipe
            svbool_t const column_predicate_b32 = svwhilelt_b32((uint32_t)0, (uint32_t)columns_remaining);
            for (nk_size_t row = 0; row < rows_actual; row++) {
                nk_size_t const row_abs = row_tile_start + row;
                svfloat64_t za_row = svread_hor_za64_f64_m(svdup_f64(0), full_predicate_b64, 1, row);
                svfloat32_t za_narrowed = svcvt_f32_f64_x(full_predicate_b64, za_row);
                svfloat32_t za_packed = svreinterpret_f32_u32(
                    svuzp1_u32(svreinterpret_u32_f32(za_narrowed), svreinterpret_u32_f32(za_narrowed)));
                svst1_f32(column_predicate_b32, result + row_abs * result_stride_elements + column_tile_start,
                          za_packed);
            }
        }
    }
}

NK_PUBLIC void nk_dots_symmetric_f32_smef64(nk_f32_t const *vectors, nk_size_t n_vectors, nk_size_t depth,
                                            nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                            nk_size_t row_start, nk_size_t row_count) {

    nk_size_t const stride_elements = stride / sizeof(nk_f32_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_dots_symmetric_f32_smef64_kernel_(vectors, n_vectors, depth, stride_elements, result, result_stride_elements,
                                         row_start, row_count);
}

#pragma endregion // Single Precision Floats

/*
 *  Native f64 GEMM using FMOPA with ZA64 tiles and Kahan compensation.
 *  Uses ZA transpose for A-vector construction (expansion=1, no interleaving needed).
 *
 *  Tile layout (SVL=512, Apple M4):
 *  - ZA0.D = staging (horizontal load, vertical read)
 *  - ZA1-7.D = accumulation (7 column tiles in fast path)
 *  - f64 input vectors: 8 elements (SVL/64)
 *  - 8 depth steps per ZA0.D load batch
 *  - Depth tiles batch in groups of 32 for Kahan extraction
 *  - FMOPA predicates: b64 (native f64 granularity)
 *  - Kahan summation: ~16-17 decimal digits precision
 */
#pragma region Double Precision Floats

NK_PUBLIC nk_size_t nk_dots_packed_size_f64_smef64(nk_size_t columns, nk_size_t depth) {
    nk_size_t const tile_dimension = svcntsd();  // rows per `ZA64` tile (8 for SVL=512)
    nk_size_t const depth_tile_size = svcntsd(); // `f64` depth elements per tile (8 for SVL=512)

    nk_size_t const column_tile_count = nk_size_divide_round_up_(columns, tile_dimension);
    nk_size_t const depth_tile_count = nk_size_divide_round_up_(depth, depth_tile_size);

    nk_size_t size = sizeof(nk_dots_sme_packed_header_t);
    size += column_tile_count * depth_tile_count * tile_dimension * depth_tile_size * sizeof(nk_f64_t);

    return size;
}

NK_PUBLIC void nk_dots_pack_f64_smef64(nk_f64_t const *b, nk_size_t columns, nk_size_t depth, nk_size_t b_stride,
                                       void *b_packed) {

    nk_size_t const tile_dimension = svcntsd();  // rows per `ZA64` tile (8 for SVL=512)
    nk_size_t const depth_tile_size = svcntsd(); // `f64` depth elements per tile (8 for SVL=512)
    nk_size_t const tile_elements = tile_dimension * depth_tile_size;
    nk_size_t const b_stride_elements = b_stride / sizeof(nk_f64_t);

    nk_size_t const column_tile_count = nk_size_divide_round_up_(columns, tile_dimension);
    nk_size_t const depth_tile_count = nk_size_divide_round_up_(depth, depth_tile_size);
    nk_size_t const total_tiles = column_tile_count * depth_tile_count;

    // Store actual dimensions and tile counts in header
    nk_dots_sme_packed_header_t *header = (nk_dots_sme_packed_header_t *)b_packed;
    header->column_tile_count = (nk_u32_t)column_tile_count;
    header->depth_tile_count = (nk_u32_t)depth_tile_count;
    header->columns = (nk_u32_t)columns;
    header->depth = (nk_u32_t)depth;
    header->svl_bytes = (nk_u32_t)svcntsb(); // streaming vector length in bytes

    nk_f64_t *tiles = (nk_f64_t *)((char *)b_packed + sizeof(nk_dots_sme_packed_header_t));

    // Zero-initialize all tiles (handles partial tile padding)
    nk_size_t const total_elements = total_tiles * tile_elements;
    for (nk_size_t i = 0; i < total_elements; i++) tiles[i] = 0.0;

    // Pack data into tiles with depth-major layout within each tile:
    // dst_idx = depth_idx * tile_dimension + column_idx
    // This allows loading one B vector per depth step: svld1(b_tile + k * tile_dimension)
    for (nk_size_t column_tile_idx = 0; column_tile_idx < column_tile_count; column_tile_idx++) {
        for (nk_size_t depth_tile_idx = 0; depth_tile_idx < depth_tile_count; depth_tile_idx++) {
            nk_size_t const tile_index = column_tile_idx * depth_tile_count + depth_tile_idx;
            nk_f64_t *tile_output = tiles + tile_index * tile_elements;

            nk_size_t const src_row_start = column_tile_idx * tile_dimension;
            nk_size_t const src_column_start = depth_tile_idx * depth_tile_size;

            // Handle partial tiles at edges
            nk_size_t const rows_to_pack = (src_row_start + tile_dimension <= columns) ? tile_dimension
                                                                                       : (columns - src_row_start);
            nk_size_t const columns_to_pack = (src_column_start + depth_tile_size <= depth)
                                                  ? depth_tile_size
                                                  : (depth - src_column_start);

            for (nk_size_t column_idx = 0; column_idx < rows_to_pack; column_idx++) {
                for (nk_size_t depth_idx = 0; depth_idx < columns_to_pack; depth_idx++) {
                    nk_size_t const src_idx = (src_row_start + column_idx) * b_stride_elements + src_column_start +
                                              depth_idx;
                    nk_size_t const dst_idx = depth_idx * tile_dimension + column_idx;
                    tile_output[dst_idx] = b[src_idx];
                }
            }
        }
    }
}

__arm_locally_streaming __arm_new("za") static void nk_dots_packed_f64_smef64_kernel_(
    nk_f64_t const *a, void const *b_packed, nk_f64_t *c, nk_size_t rows, nk_size_t columns, nk_size_t depth,
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) {

    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_size_t const column_tile_count = header->column_tile_count;
    nk_size_t const depth_tile_count = header->depth_tile_count;

    nk_size_t const tile_dimension = svcntd();  // 8 for 512-bit SVL
    nk_size_t const depth_tile_size = svcntd(); // 8 for 512-bit SVL
    nk_size_t const tile_elements = tile_dimension * depth_tile_size;
    nk_size_t const depth_steps_per_batch = tile_dimension; // 8 depth steps per ZA0.D load

    nk_f64_t const *b_tiles = (nk_f64_t const *)((char const *)b_packed + sizeof(nk_dots_sme_packed_header_t));

    svbool_t const full_predicate_b64 = svptrue_b64();

    // Kahan accumulator and compensation arrays - 7 sets for 7-tile processing
    nk_f64_t partial_sum[7][64];
    nk_f64_t compensation[7][64];
    nk_f64_t kahan_accumulator[7][64];

    nk_size_t const kahan_batch_max = 32; // batch depth-tiles before Kahan extraction
    nk_size_t const kahan_batch_size = kahan_batch_max < depth_tile_count ? kahan_batch_max : depth_tile_count;

    // ZA0.D = staging, ZA1-7.D = accumulation (7-tile fast path)
    for (nk_size_t row_tile_index = 0; row_tile_index < nk_size_divide_round_up_(rows, tile_dimension);
         row_tile_index++) {
        nk_size_t const row_start = row_tile_index * tile_dimension;
        nk_size_t const rows_remaining = (row_start + tile_dimension <= rows) ? tile_dimension : (rows - row_start);
        svbool_t const row_predicate_b64 = svwhilelt_b64((uint64_t)0, rows_remaining);

        nk_size_t column_tile_index = 0;

        // Fast path: 7 column tiles using ZA1-ZA7 (ZA0.D = staging)
        for (; column_tile_index + 7 <= column_tile_count; column_tile_index += 7) {

            // Initialize all 7 Kahan accumulators
            svfloat64_t zero_vec = svdup_f64(0.0);
            for (int t = 0; t < 7; t++) {
                for (nk_size_t row_idx = 0; row_idx < tile_dimension; row_idx++) {
                    svst1_f64(full_predicate_b64, kahan_accumulator[t] + row_idx * tile_dimension, zero_vec);
                    svst1_f64(full_predicate_b64, compensation[t] + row_idx * tile_dimension, zero_vec);
                }
            }

            // Process depth tiles in Kahan batches
            for (nk_size_t kahan_batch_start = 0; kahan_batch_start < depth_tile_count;
                 kahan_batch_start += kahan_batch_size) {
                nk_size_t const kahan_batch_end = (kahan_batch_start + kahan_batch_size < depth_tile_count)
                                                      ? kahan_batch_start + kahan_batch_size
                                                      : depth_tile_count;

                svzero_mask_za(nk_sme_zero_za64_tiles_1_7_);

                for (nk_size_t depth_tile_idx = kahan_batch_start; depth_tile_idx < kahan_batch_end; depth_tile_idx++) {
                    nk_size_t const depth_offset = depth_tile_idx * depth_tile_size;

                    // Each depth tile has `depth_tile_size` (8) elements = one ZA0.D batch
                    if (depth_offset >= depth) break;

                    svzero_mask_za(nk_sme_zero_za64_tile_0_);

                    nk_size_t const batch_size = (depth_offset + depth_tile_size <= depth) ? depth_tile_size
                                                                                           : (depth - depth_offset);
                    svbool_t const batch_predicate_b64 = svwhilelt_b64((uint64_t)0, (uint64_t)batch_size);

                    // Load A rows into ZA0.D horizontally
                    for (nk_size_t row_in_tile = 0; row_in_tile < rows_remaining; row_in_tile++) {
                        nk_size_t const a_row = row_start + row_in_tile;
                        svld1_hor_za64(0, row_in_tile, batch_predicate_b64,
                                       &a[a_row * a_stride_elements + depth_offset]);
                    }

                    // Vertical read + MOPA for each depth step
                    for (nk_size_t step = 0; step < batch_size; step++) {
                        svfloat64_t vector_a = svread_ver_za64_f64_m(svdup_f64(0.0), row_predicate_b64, 0, step);

                        nk_size_t const b_k = step;
                        svfloat64_t b_vector_1 = svld1_f64(
                            full_predicate_b64,
                            b_tiles + ((column_tile_index + 0) * depth_tile_count + depth_tile_idx) * tile_elements +
                                b_k * tile_dimension);
                        svfloat64_t b_vector_2 = svld1_f64(
                            full_predicate_b64,
                            b_tiles + ((column_tile_index + 1) * depth_tile_count + depth_tile_idx) * tile_elements +
                                b_k * tile_dimension);
                        svfloat64_t b_vector_3 = svld1_f64(
                            full_predicate_b64,
                            b_tiles + ((column_tile_index + 2) * depth_tile_count + depth_tile_idx) * tile_elements +
                                b_k * tile_dimension);
                        svfloat64_t b_vector_4 = svld1_f64(
                            full_predicate_b64,
                            b_tiles + ((column_tile_index + 3) * depth_tile_count + depth_tile_idx) * tile_elements +
                                b_k * tile_dimension);
                        svfloat64_t b_vector_5 = svld1_f64(
                            full_predicate_b64,
                            b_tiles + ((column_tile_index + 4) * depth_tile_count + depth_tile_idx) * tile_elements +
                                b_k * tile_dimension);
                        svfloat64_t b_vector_6 = svld1_f64(
                            full_predicate_b64,
                            b_tiles + ((column_tile_index + 5) * depth_tile_count + depth_tile_idx) * tile_elements +
                                b_k * tile_dimension);
                        svfloat64_t b_vector_7 = svld1_f64(
                            full_predicate_b64,
                            b_tiles + ((column_tile_index + 6) * depth_tile_count + depth_tile_idx) * tile_elements +
                                b_k * tile_dimension);

                        svmopa_za64_f64_m(1, row_predicate_b64, full_predicate_b64, vector_a, b_vector_1);
                        svmopa_za64_f64_m(2, row_predicate_b64, full_predicate_b64, vector_a, b_vector_2);
                        svmopa_za64_f64_m(3, row_predicate_b64, full_predicate_b64, vector_a, b_vector_3);
                        svmopa_za64_f64_m(4, row_predicate_b64, full_predicate_b64, vector_a, b_vector_4);
                        svmopa_za64_f64_m(5, row_predicate_b64, full_predicate_b64, vector_a, b_vector_5);
                        svmopa_za64_f64_m(6, row_predicate_b64, full_predicate_b64, vector_a, b_vector_6);
                        svmopa_za64_f64_m(7, row_predicate_b64, full_predicate_b64, vector_a, b_vector_7);
                    }
                }

                // Extract partial sums from ZA1-7
                for (nk_size_t row_idx = 0; row_idx < tile_dimension; row_idx++) {
                    svst1_hor_za64(1, row_idx, full_predicate_b64, partial_sum[0] + row_idx * tile_dimension);
                    svst1_hor_za64(2, row_idx, full_predicate_b64, partial_sum[1] + row_idx * tile_dimension);
                    svst1_hor_za64(3, row_idx, full_predicate_b64, partial_sum[2] + row_idx * tile_dimension);
                    svst1_hor_za64(4, row_idx, full_predicate_b64, partial_sum[3] + row_idx * tile_dimension);
                    svst1_hor_za64(5, row_idx, full_predicate_b64, partial_sum[4] + row_idx * tile_dimension);
                    svst1_hor_za64(6, row_idx, full_predicate_b64, partial_sum[5] + row_idx * tile_dimension);
                    svst1_hor_za64(7, row_idx, full_predicate_b64, partial_sum[6] + row_idx * tile_dimension);
                }

                // Apply Kahan compensation to all 7 tiles
                for (int t = 0; t < 7; t++) {
                    for (nk_size_t row_idx = 0; row_idx < tile_dimension; row_idx++) {
                        nk_size_t const base_idx = row_idx * tile_dimension;

                        svfloat64_t acc_vec = svld1_f64(full_predicate_b64, kahan_accumulator[t] + base_idx);
                        svfloat64_t comp_vec = svld1_f64(full_predicate_b64, compensation[t] + base_idx);
                        svfloat64_t part_vec = svld1_f64(full_predicate_b64, partial_sum[t] + base_idx);

                        // Kahan summation: y = part - comp; s = acc + y; comp = (s - acc) - y; acc = s
                        svfloat64_t y = svsub_f64_x(full_predicate_b64, part_vec, comp_vec);
                        svfloat64_t s = svadd_f64_x(full_predicate_b64, acc_vec, y);
                        comp_vec = svsub_f64_x(full_predicate_b64, svsub_f64_x(full_predicate_b64, s, acc_vec), y);
                        acc_vec = s;

                        svst1_f64(full_predicate_b64, kahan_accumulator[t] + base_idx, acc_vec);
                        svst1_f64(full_predicate_b64, compensation[t] + base_idx, comp_vec);
                    }
                }
            }

            // Store results from all 7 tiles
            for (nk_size_t row_idx = 0; row_idx < rows_remaining; row_idx++) {
                nk_f64_t *c_row = c + (row_start + row_idx) * c_stride_elements;
                nk_size_t const base_idx = row_idx * tile_dimension;
                svst1_f64(full_predicate_b64, c_row + (column_tile_index + 0) * tile_dimension,
                          svld1_f64(full_predicate_b64, kahan_accumulator[0] + base_idx));
                svst1_f64(full_predicate_b64, c_row + (column_tile_index + 1) * tile_dimension,
                          svld1_f64(full_predicate_b64, kahan_accumulator[1] + base_idx));
                svst1_f64(full_predicate_b64, c_row + (column_tile_index + 2) * tile_dimension,
                          svld1_f64(full_predicate_b64, kahan_accumulator[2] + base_idx));
                svst1_f64(full_predicate_b64, c_row + (column_tile_index + 3) * tile_dimension,
                          svld1_f64(full_predicate_b64, kahan_accumulator[3] + base_idx));
                svst1_f64(full_predicate_b64, c_row + (column_tile_index + 4) * tile_dimension,
                          svld1_f64(full_predicate_b64, kahan_accumulator[4] + base_idx));
                svst1_f64(full_predicate_b64, c_row + (column_tile_index + 5) * tile_dimension,
                          svld1_f64(full_predicate_b64, kahan_accumulator[5] + base_idx));
                svst1_f64(full_predicate_b64, c_row + (column_tile_index + 6) * tile_dimension,
                          svld1_f64(full_predicate_b64, kahan_accumulator[6] + base_idx));
            }
        }

        // Remainder: 1 column tile at a time using ZA1 (ZA0.D = staging)
        for (; column_tile_index < column_tile_count; column_tile_index++) {
            nk_size_t const column_start = column_tile_index * tile_dimension;
            nk_size_t const columns_remaining = (column_start + tile_dimension <= columns) ? tile_dimension
                                                                                           : (columns - column_start);
            svbool_t const column_predicate_b64 = svwhilelt_b64((uint64_t)0, columns_remaining);

            // Initialize Kahan accumulators
            svfloat64_t zero_vec = svdup_f64(0.0);
            for (nk_size_t row_idx = 0; row_idx < tile_dimension; row_idx++) {
                svst1_f64(full_predicate_b64, kahan_accumulator[0] + row_idx * tile_dimension, zero_vec);
                svst1_f64(full_predicate_b64, compensation[0] + row_idx * tile_dimension, zero_vec);
            }

            // Process depth tiles in Kahan batches
            for (nk_size_t kahan_batch_start = 0; kahan_batch_start < depth_tile_count;
                 kahan_batch_start += kahan_batch_size) {
                nk_size_t const kahan_batch_end = (kahan_batch_start + kahan_batch_size < depth_tile_count)
                                                      ? kahan_batch_start + kahan_batch_size
                                                      : depth_tile_count;

                svzero_mask_za(nk_sme_zero_za64_tile_1_);

                for (nk_size_t depth_tile_idx = kahan_batch_start; depth_tile_idx < kahan_batch_end; depth_tile_idx++) {
                    nk_size_t const depth_offset = depth_tile_idx * depth_tile_size;

                    if (depth_offset >= depth) break;

                    svzero_mask_za(nk_sme_zero_za64_tile_0_);

                    nk_size_t const batch_size = (depth_offset + depth_tile_size <= depth) ? depth_tile_size
                                                                                           : (depth - depth_offset);
                    svbool_t const batch_predicate_b64 = svwhilelt_b64((uint64_t)0, (uint64_t)batch_size);

                    // Load A rows into ZA0.D horizontally
                    for (nk_size_t row_in_tile = 0; row_in_tile < rows_remaining; row_in_tile++) {
                        nk_size_t const a_row = row_start + row_in_tile;
                        svld1_hor_za64(0, row_in_tile, batch_predicate_b64,
                                       &a[a_row * a_stride_elements + depth_offset]);
                    }

                    // Vertical read + MOPA
                    for (nk_size_t step = 0; step < batch_size; step++) {
                        svfloat64_t vector_a = svread_ver_za64_f64_m(svdup_f64(0.0), row_predicate_b64, 0, step);

                        nk_f64_t const *b_tile = b_tiles + (column_tile_index * depth_tile_count + depth_tile_idx) *
                                                               tile_elements;
                        svfloat64_t vector_b = svld1_f64(full_predicate_b64, b_tile + step * tile_dimension);

                        svmopa_za64_f64_m(1, row_predicate_b64, column_predicate_b64, vector_a, vector_b);
                    }
                }

                // Extract partial sum from ZA1
                for (nk_size_t row_idx = 0; row_idx < tile_dimension; row_idx++) {
                    svst1_hor_za64(1, row_idx, full_predicate_b64, partial_sum[0] + row_idx * tile_dimension);
                }

                // Apply Kahan compensation
                for (nk_size_t row_idx = 0; row_idx < tile_dimension; row_idx++) {
                    nk_size_t const base_idx = row_idx * tile_dimension;

                    svfloat64_t acc_vec = svld1_f64(full_predicate_b64, kahan_accumulator[0] + base_idx);
                    svfloat64_t comp_vec = svld1_f64(full_predicate_b64, compensation[0] + base_idx);
                    svfloat64_t part_vec = svld1_f64(full_predicate_b64, partial_sum[0] + base_idx);

                    // Kahan summation: y = part - comp; s = acc + y; comp = (s - acc) - y; acc = s
                    svfloat64_t y = svsub_f64_x(full_predicate_b64, part_vec, comp_vec);
                    svfloat64_t s = svadd_f64_x(full_predicate_b64, acc_vec, y);
                    comp_vec = svsub_f64_x(full_predicate_b64, svsub_f64_x(full_predicate_b64, s, acc_vec), y);
                    acc_vec = s;

                    svst1_f64(full_predicate_b64, kahan_accumulator[0] + base_idx, acc_vec);
                    svst1_f64(full_predicate_b64, compensation[0] + base_idx, comp_vec);
                }
            }

            // Store results with predication
            for (nk_size_t row_idx = 0; row_idx < rows_remaining; row_idx++) {
                nk_f64_t *c_row = c + (row_start + row_idx) * c_stride_elements + column_start;
                svfloat64_t acc_vec = svld1_f64(full_predicate_b64, kahan_accumulator[0] + row_idx * tile_dimension);
                svst1_f64(column_predicate_b64, c_row, acc_vec);
            }
        }
    }
}

NK_PUBLIC void nk_dots_packed_f64_smef64(nk_f64_t const *a, void const *b_packed, nk_f64_t *c, nk_size_t rows,
                                         nk_size_t columns, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride) {

    nk_size_t const a_stride_elements = a_stride / sizeof(nk_f64_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f64_t);

    nk_dots_packed_f64_smef64_kernel_(a, b_packed, c, rows, columns, depth, a_stride_elements, c_stride_elements);
}

__arm_locally_streaming __arm_new("za") static void nk_dots_symmetric_f64_smef64_kernel_(
    nk_f64_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride_elements, nk_f64_t *result,
    nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) {

    nk_size_t const expansion = 1;
    nk_size_t const tile_dimension = svcntd();
    nk_size_t const depth_step_count = depth;
    nk_size_t const depth_steps_per_batch = tile_dimension;

    svbool_t const full_predicate_b64 = svptrue_b64();

    NK_ALIGN64 nk_f64_t a_buffer[8][8];

    nk_size_t const row_end = row_start + row_count;
    nk_size_t const column_tile_count = nk_size_divide_round_up_(n_vectors, tile_dimension);

    for (nk_size_t row_tile_start = row_start; row_tile_start < row_end && row_tile_start < n_vectors;
         row_tile_start += tile_dimension) {
        nk_size_t const rows_remaining = (row_tile_start + tile_dimension <= row_end) ? tile_dimension
                                                                                      : (row_end - row_tile_start);
        nk_size_t const rows_clamped = (row_tile_start + rows_remaining <= n_vectors) ? rows_remaining
                                                                                      : (n_vectors - row_tile_start);
        svbool_t const row_predicate_b64 = svwhilelt_b64((uint64_t)0, (uint64_t)rows_clamped);

        nk_size_t column_tile_index = 0;

        for (; column_tile_index + 7 <= column_tile_count; column_tile_index += 7) {
            svzero_mask_za(nk_sme_zero_za64_tiles_1_7_);

            for (nk_size_t depth_batch_start = 0; depth_batch_start < depth_step_count;
                 depth_batch_start += depth_steps_per_batch) {
                nk_size_t const depth_batch_end = (depth_batch_start + depth_steps_per_batch < depth_step_count)
                                                      ? depth_batch_start + depth_steps_per_batch
                                                      : depth_step_count;
                nk_size_t const batch_size = depth_batch_end - depth_batch_start;

                svzero_mask_za(nk_sme_zero_za64_tile_0_);
                svbool_t const batch_predicate_b64 = svwhilelt_b64((uint64_t)0, (uint64_t)batch_size);
                for (nk_size_t row_in_tile = 0; row_in_tile < rows_clamped; row_in_tile++) {
                    nk_size_t const row_abs = row_tile_start + row_in_tile;
                    svld1_hor_za64(0, row_in_tile, batch_predicate_b64,
                                   vectors + row_abs * stride_elements + depth_batch_start);
                }

                for (nk_size_t s = 0; s < batch_size; s++)
                    svst1_f64(full_predicate_b64, a_buffer[s],
                              svread_ver_za64_f64_m(svdup_f64(0), row_predicate_b64, 0, s));

                svzero_mask_za(nk_sme_zero_za64_tile_0_);
                for (nk_size_t column = 0; column < tile_dimension; column++) {
                    nk_size_t const column_abs = (column_tile_index + 0) * tile_dimension + column;
                    if (column_abs < n_vectors)
                        svld1_hor_za64(0, column, batch_predicate_b64,
                                       vectors + column_abs * stride_elements + depth_batch_start);
                }
                for (nk_size_t step = 0; step < batch_size; step++) {
                    svfloat64_t a_vec = svld1_f64(full_predicate_b64, a_buffer[step]);
                    svfloat64_t b_vec = svread_ver_za64_f64_m(svdup_f64(0.0), full_predicate_b64, 0, step);
                    svmopa_za64_f64_m(1, row_predicate_b64, full_predicate_b64, a_vec, b_vec);
                }

                svzero_mask_za(nk_sme_zero_za64_tile_0_);
                for (nk_size_t column = 0; column < tile_dimension; column++) {
                    nk_size_t const column_abs = (column_tile_index + 1) * tile_dimension + column;
                    if (column_abs < n_vectors)
                        svld1_hor_za64(0, column, batch_predicate_b64,
                                       vectors + column_abs * stride_elements + depth_batch_start);
                }
                for (nk_size_t step = 0; step < batch_size; step++) {
                    svfloat64_t a_vec = svld1_f64(full_predicate_b64, a_buffer[step]);
                    svfloat64_t b_vec = svread_ver_za64_f64_m(svdup_f64(0.0), full_predicate_b64, 0, step);
                    svmopa_za64_f64_m(2, row_predicate_b64, full_predicate_b64, a_vec, b_vec);
                }

                svzero_mask_za(nk_sme_zero_za64_tile_0_);
                for (nk_size_t column = 0; column < tile_dimension; column++) {
                    nk_size_t const column_abs = (column_tile_index + 2) * tile_dimension + column;
                    if (column_abs < n_vectors)
                        svld1_hor_za64(0, column, batch_predicate_b64,
                                       vectors + column_abs * stride_elements + depth_batch_start);
                }
                for (nk_size_t step = 0; step < batch_size; step++) {
                    svfloat64_t a_vec = svld1_f64(full_predicate_b64, a_buffer[step]);
                    svfloat64_t b_vec = svread_ver_za64_f64_m(svdup_f64(0.0), full_predicate_b64, 0, step);
                    svmopa_za64_f64_m(3, row_predicate_b64, full_predicate_b64, a_vec, b_vec);
                }

                svzero_mask_za(nk_sme_zero_za64_tile_0_);
                for (nk_size_t column = 0; column < tile_dimension; column++) {
                    nk_size_t const column_abs = (column_tile_index + 3) * tile_dimension + column;
                    if (column_abs < n_vectors)
                        svld1_hor_za64(0, column, batch_predicate_b64,
                                       vectors + column_abs * stride_elements + depth_batch_start);
                }
                for (nk_size_t step = 0; step < batch_size; step++) {
                    svfloat64_t a_vec = svld1_f64(full_predicate_b64, a_buffer[step]);
                    svfloat64_t b_vec = svread_ver_za64_f64_m(svdup_f64(0.0), full_predicate_b64, 0, step);
                    svmopa_za64_f64_m(4, row_predicate_b64, full_predicate_b64, a_vec, b_vec);
                }

                svzero_mask_za(nk_sme_zero_za64_tile_0_);
                for (nk_size_t column = 0; column < tile_dimension; column++) {
                    nk_size_t const column_abs = (column_tile_index + 4) * tile_dimension + column;
                    if (column_abs < n_vectors)
                        svld1_hor_za64(0, column, batch_predicate_b64,
                                       vectors + column_abs * stride_elements + depth_batch_start);
                }
                for (nk_size_t step = 0; step < batch_size; step++) {
                    svfloat64_t a_vec = svld1_f64(full_predicate_b64, a_buffer[step]);
                    svfloat64_t b_vec = svread_ver_za64_f64_m(svdup_f64(0.0), full_predicate_b64, 0, step);
                    svmopa_za64_f64_m(5, row_predicate_b64, full_predicate_b64, a_vec, b_vec);
                }

                svzero_mask_za(nk_sme_zero_za64_tile_0_);
                for (nk_size_t column = 0; column < tile_dimension; column++) {
                    nk_size_t const column_abs = (column_tile_index + 5) * tile_dimension + column;
                    if (column_abs < n_vectors)
                        svld1_hor_za64(0, column, batch_predicate_b64,
                                       vectors + column_abs * stride_elements + depth_batch_start);
                }
                for (nk_size_t step = 0; step < batch_size; step++) {
                    svfloat64_t a_vec = svld1_f64(full_predicate_b64, a_buffer[step]);
                    svfloat64_t b_vec = svread_ver_za64_f64_m(svdup_f64(0.0), full_predicate_b64, 0, step);
                    svmopa_za64_f64_m(6, row_predicate_b64, full_predicate_b64, a_vec, b_vec);
                }

                svzero_mask_za(nk_sme_zero_za64_tile_0_);
                for (nk_size_t column = 0; column < tile_dimension; column++) {
                    nk_size_t const column_abs = (column_tile_index + 6) * tile_dimension + column;
                    if (column_abs < n_vectors)
                        svld1_hor_za64(0, column, batch_predicate_b64,
                                       vectors + column_abs * stride_elements + depth_batch_start);
                }
                for (nk_size_t step = 0; step < batch_size; step++) {
                    svfloat64_t a_vec = svld1_f64(full_predicate_b64, a_buffer[step]);
                    svfloat64_t b_vec = svread_ver_za64_f64_m(svdup_f64(0.0), full_predicate_b64, 0, step);
                    svmopa_za64_f64_m(7, row_predicate_b64, full_predicate_b64, a_vec, b_vec);
                }
            }

            for (nk_size_t row = 0; row < rows_clamped; row++) {
                nk_size_t const row_abs = row_tile_start + row;
                nk_f64_t *result_row = result + row_abs * result_stride_elements;
                svst1_hor_za64(1, row, full_predicate_b64, result_row + (column_tile_index + 0) * tile_dimension);
                svst1_hor_za64(2, row, full_predicate_b64, result_row + (column_tile_index + 1) * tile_dimension);
                svst1_hor_za64(3, row, full_predicate_b64, result_row + (column_tile_index + 2) * tile_dimension);
                svst1_hor_za64(4, row, full_predicate_b64, result_row + (column_tile_index + 3) * tile_dimension);
                svst1_hor_za64(5, row, full_predicate_b64, result_row + (column_tile_index + 4) * tile_dimension);
                svst1_hor_za64(6, row, full_predicate_b64, result_row + (column_tile_index + 5) * tile_dimension);
                svst1_hor_za64(7, row, full_predicate_b64, result_row + (column_tile_index + 6) * tile_dimension);
            }
        }

        for (; column_tile_index < column_tile_count; column_tile_index++) {
            nk_size_t const column_tile_start = column_tile_index * tile_dimension;
            nk_size_t const columns_remaining = (column_tile_start + tile_dimension <= n_vectors)
                                                    ? tile_dimension
                                                    : (n_vectors - column_tile_start);
            svbool_t const column_predicate_b64 = svwhilelt_b64((uint64_t)0, (uint64_t)columns_remaining);

            svzero_mask_za(nk_sme_zero_za64_tile_1_);

            for (nk_size_t depth_batch_start = 0; depth_batch_start < depth_step_count;
                 depth_batch_start += depth_steps_per_batch) {
                nk_size_t const depth_batch_end = (depth_batch_start + depth_steps_per_batch < depth_step_count)
                                                      ? depth_batch_start + depth_steps_per_batch
                                                      : depth_step_count;
                nk_size_t const batch_size = depth_batch_end - depth_batch_start;

                svzero_mask_za(nk_sme_zero_za64_tile_0_);
                svbool_t const batch_predicate_b64 = svwhilelt_b64((uint64_t)0, (uint64_t)batch_size);
                for (nk_size_t row_in_tile = 0; row_in_tile < rows_clamped; row_in_tile++) {
                    nk_size_t const row_abs = row_tile_start + row_in_tile;
                    svld1_hor_za64(0, row_in_tile, batch_predicate_b64,
                                   vectors + row_abs * stride_elements + depth_batch_start);
                }

                for (nk_size_t s = 0; s < batch_size; s++)
                    svst1_f64(full_predicate_b64, a_buffer[s],
                              svread_ver_za64_f64_m(svdup_f64(0), row_predicate_b64, 0, s));

                svzero_mask_za(nk_sme_zero_za64_tile_0_);
                for (nk_size_t column = 0; column < tile_dimension; column++) {
                    nk_size_t const column_abs = column_tile_start + column;
                    if (column_abs < n_vectors)
                        svld1_hor_za64(0, column, batch_predicate_b64,
                                       vectors + column_abs * stride_elements + depth_batch_start);
                }
                for (nk_size_t step = 0; step < batch_size; step++) {
                    svfloat64_t a_vec = svld1_f64(full_predicate_b64, a_buffer[step]);
                    svfloat64_t b_vec = svread_ver_za64_f64_m(svdup_f64(0.0), column_predicate_b64, 0, step);
                    svmopa_za64_f64_m(1, row_predicate_b64, column_predicate_b64, a_vec, b_vec);
                }
            }

            for (nk_size_t row = 0; row < rows_clamped; row++) {
                nk_size_t const row_abs = row_tile_start + row;
                svst1_hor_za64(1, row, column_predicate_b64,
                               result + row_abs * result_stride_elements + column_tile_start);
            }
        }
    }
}

NK_PUBLIC void nk_dots_symmetric_f64_smef64(nk_f64_t const *vectors, nk_size_t n_vectors, nk_size_t depth,
                                            nk_size_t stride, nk_f64_t *result, nk_size_t result_stride,
                                            nk_size_t row_start, nk_size_t row_count) {

    nk_size_t const stride_elements = stride / sizeof(nk_f64_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f64_t);
    nk_dots_symmetric_f64_smef64_kernel_(vectors, n_vectors, depth, stride_elements, result, result_stride_elements,
                                         row_start, row_count);
}

#pragma endregion // Double Precision Floats

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
#endif // NK_DOTS_SMEF64_H

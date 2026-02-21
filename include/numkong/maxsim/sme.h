/**
 *  @brief SIMD-accelerated MaxSim (ColBERT late-interaction) for SME.
 *  @file include/numkong/maxsim/sme.h
 *  @author Ash Vardanian
 *  @date February 10, 2026
 *
 *  Computes MaxSim(Q, D) = Σᵢ maxⱼ dot(qᵢ, dⱼ) using ARM SME outer products.
 *
 *  Both Q and D are pre-packed with `nk_dots_pack_bf16_sme` from `dots/sme.h`.
 *  This frees all 4 ZA tiles for accumulation (vs 3 with A-side staging).
 *
 *  Key optimization: vertical column reads for max reduction.
 *  Traditional extraction reads tile rows then calls `svmaxv` (horizontal max, ~8cy).
 *  Our approach reads tile columns with `svread_ver_za32_f32_m`:
 *
 *    - Each column read gives dot products of all query tokens vs one doc token.
 *    - Element-wise `svmax` (~1cy) updates a running max vector across doc tokens.
 *    - Only `svaddv` at the very end: ⌈n_q/16⌉ = 2 horizontal reductions total.
 *
 *  This is ~100x fewer horizontal reductions for typical ColBERT dimensions.
 *
 *  ZA tile layout after BFMOPA accumulation (16x16 f32):
 *
 *  - Row i, Column j = dot(q_{tile_row_start + i}, d_{tile_col_start + j})
 *  - Vertical column read of column j → similarities of all 16 q tokens to doc token j
 *  - Element-wise max across columns → per-query-token max over doc tokens in this tile group
 *
 *  Benchmark results (Apple M4, SVL=512):
 *
 *      Dimensions              dots_packed GEMM    maxsim fused    GEMM speedup    End-to-end speedup
 *      32×128×128 (ColBERT)    840 GFLOPS          1516 GFLOPS     1.81×           5.10×
 *      32×256×128              1037 GFLOPS         1591 GFLOPS     1.53×           5.17×
 *      64×512×128              1016 GFLOPS         1651 GFLOPS     1.62×           5.42×
 *      32×128×256              859 GFLOPS          1725 GFLOPS     2.01×           4.06×
 *      32×1024×768 (BERT)      1124 GFLOPS         1932 GFLOPS     1.72×           2.61×
 *
 *  Speedup sources:
 *
 *  1. Pre-packing both sides → 4 ZA tiles for accumulation (vs 3 with A-staging): +33% MOPA throughput
 *  2. No output matrix materialization → eliminates M×N f32 memory round-trip
 *  3. Vertical column reads → ~128 element-wise svmax (1cy) vs ~256 svmaxv horizontal reductions (8cy)
 */
#ifndef NK_MAXSIM_SME_H
#define NK_MAXSIM_SME_H

#if NK_TARGET_ARM_
#if NK_TARGET_SME

#include "numkong/dots/sme.h" // nk_dots_sme_packed_header_t, nk_dots_pack_{f16,bf16}_sme, nk_dots_packed_size_{f16,bf16}_sme

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("sme"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("+sme")
#endif

/**
 *  MaxSim f16 kernel: both Q and D pre-packed, vertical column read extraction.
 *
 *  4-tile fast path: processes 4 doc column tiles simultaneously using ZA0-ZA3.
 *  Inner loop per depth_step: 1 Q load + 4 D loads + 4 FMOPA = 9 ops.
 *  Extraction per 4-tile group: 4×16 = 64 vertical reads + 64 svmax = ~128 cycles.
 *
 *  1-tile remainder: uses ZA0 only, with predicated loads for partial tiles.
 */
__arm_locally_streaming __arm_new("za") static nk_f32_t nk_maxsim_packed_f16_streaming_( //
    void const *q_packed, nk_size_t n_q,                                                 //
    void const *d_packed, nk_size_t n_d,                                                 //
    nk_size_t depth) {

    nk_dots_sme_packed_header_t const *q_header = (nk_dots_sme_packed_header_t const *)q_packed;
    nk_dots_sme_packed_header_t const *d_header = (nk_dots_sme_packed_header_t const *)d_packed;
    nk_size_t const depth_step_count = q_header->depth_tile_count;
    nk_size_t const q_row_tiles = q_header->column_tile_count;
    nk_size_t const d_col_tiles = d_header->column_tile_count;

    nk_size_t const tile_dimension = svcntw();  // 16: ZA32 tile dimension
    nk_size_t const vector_elements = svcnth(); // 32: f16 elements per SVE vector

    nk_f16_t const *q_vecs = (nk_f16_t const *)((char const *)q_packed + sizeof(nk_dots_sme_packed_header_t));
    nk_f16_t const *d_vecs = (nk_f16_t const *)((char const *)d_packed + sizeof(nk_dots_sme_packed_header_t));

    svbool_t const full_predicate_b16 = svptrue_b16();
    svbool_t const full_predicate_b32 = svptrue_b32();

    nk_f32_t total = 0.0f;

    for (nk_size_t row_tile_index = 0; row_tile_index < q_row_tiles; row_tile_index++) {
        nk_size_t const row_start = row_tile_index * tile_dimension;
        nk_size_t const rows_remaining = (row_start + tile_dimension <= n_q) ? tile_dimension : (n_q - row_start);
        svbool_t const row_predicate_b16 = (rows_remaining == tile_dimension)
                                               ? svptrue_b16()
                                               : svwhilelt_b16((nk_u32_t)0, (nk_u32_t)(rows_remaining * 2));
        svbool_t const row_predicate_b32 = (rows_remaining == tile_dimension)
                                               ? svptrue_b32()
                                               : svwhilelt_b32((nk_u32_t)0, (nk_u32_t)rows_remaining);

        // Running max vector: element i = max_j dot(q_{row_start+i}, d_j) seen so far
        svfloat32_t running_max = svdup_f32(NK_F32_MIN);

        nk_size_t column_tile_index = 0;

        // Fast path: 4 doc column tiles at a time using ZA0-ZA3
        for (; column_tile_index + 4 <= d_col_tiles; column_tile_index += 4) {
            svzero_za(); // Zero all 4 tiles

            // Accumulate: for each depth step, load Q vector and 4 D vectors, issue 4 FMOPAs
            for (nk_size_t depth_step = 0; depth_step < depth_step_count; depth_step++) {
                svfloat16_t a_packed_vector = svld1_f16(
                    row_predicate_b16,
                    (float16_t const *)(q_vecs + (row_tile_index * depth_step_count + depth_step) * vector_elements));
                svfloat16_t b_packed_vector_0 = svld1_f16(
                    full_predicate_b16,
                    (float16_t const *)(d_vecs +
                                        ((column_tile_index + 0) * depth_step_count + depth_step) * vector_elements));
                svfloat16_t b_packed_vector_1 = svld1_f16(
                    full_predicate_b16,
                    (float16_t const *)(d_vecs +
                                        ((column_tile_index + 1) * depth_step_count + depth_step) * vector_elements));
                svfloat16_t b_packed_vector_2 = svld1_f16(
                    full_predicate_b16,
                    (float16_t const *)(d_vecs +
                                        ((column_tile_index + 2) * depth_step_count + depth_step) * vector_elements));
                svfloat16_t b_packed_vector_3 = svld1_f16(
                    full_predicate_b16,
                    (float16_t const *)(d_vecs +
                                        ((column_tile_index + 3) * depth_step_count + depth_step) * vector_elements));
                svmopa_za32_f16_m(0, row_predicate_b16, full_predicate_b16, a_packed_vector, b_packed_vector_0);
                svmopa_za32_f16_m(1, row_predicate_b16, full_predicate_b16, a_packed_vector, b_packed_vector_1);
                svmopa_za32_f16_m(2, row_predicate_b16, full_predicate_b16, a_packed_vector, b_packed_vector_2);
                svmopa_za32_f16_m(3, row_predicate_b16, full_predicate_b16, a_packed_vector, b_packed_vector_3);
            }

            // Vertical column extraction:
            // ZA tile[row i][col j] = dot(q_{row_start+i}, d_{col_tile_start+j})
            // Reading column j gives a vector of dot products for all query tokens vs doc token j.
            // Element-wise max across all 64 columns (4 tiles × 16 columns) gives the per-query-token
            // maximum similarity over these 64 doc tokens.
            for (nk_size_t column_within_tile = 0; column_within_tile < tile_dimension; column_within_tile++) {
                svfloat32_t v0 = svread_ver_za32_f32_m(svdup_f32(NK_F32_MIN), full_predicate_b32, 0,
                                                       column_within_tile);
                svfloat32_t v1 = svread_ver_za32_f32_m(svdup_f32(NK_F32_MIN), full_predicate_b32, 1,
                                                       column_within_tile);
                svfloat32_t v2 = svread_ver_za32_f32_m(svdup_f32(NK_F32_MIN), full_predicate_b32, 2,
                                                       column_within_tile);
                svfloat32_t v3 = svread_ver_za32_f32_m(svdup_f32(NK_F32_MIN), full_predicate_b32, 3,
                                                       column_within_tile);
                svfloat32_t col_max = svmax_f32_x(full_predicate_b32, svmax_f32_x(full_predicate_b32, v0, v1),
                                                  svmax_f32_x(full_predicate_b32, v2, v3));
                running_max = svmax_f32_x(full_predicate_b32, running_max, col_max);
            }
        }

        // Remainder: 1 doc column tile at a time using ZA0 only
        for (; column_tile_index < d_col_tiles; column_tile_index++) {
            nk_size_t const col_start = column_tile_index * tile_dimension;
            nk_size_t const cols_remaining = (col_start + tile_dimension <= n_d) ? tile_dimension : (n_d - col_start);
            svbool_t const column_predicate_b16 = (cols_remaining == tile_dimension)
                                                      ? svptrue_b16()
                                                      : svwhilelt_b16((nk_u32_t)0, (nk_u32_t)(cols_remaining * 2));

            svzero_mask_za(nk_sme_zero_za32_tile_0_); // Zero ZA0 only

            for (nk_size_t depth_step = 0; depth_step < depth_step_count; depth_step++) {
                svfloat16_t a_packed_vector = svld1_f16(
                    row_predicate_b16,
                    (float16_t const *)(q_vecs + (row_tile_index * depth_step_count + depth_step) * vector_elements));
                svfloat16_t b_packed_vector = svld1_f16(
                    column_predicate_b16,
                    (float16_t const *)(d_vecs +
                                        (column_tile_index * depth_step_count + depth_step) * vector_elements));
                svmopa_za32_f16_m(0, row_predicate_b16, column_predicate_b16, a_packed_vector, b_packed_vector);
            }

            // Vertical column extraction from ZA0 — only cols_remaining valid columns
            for (nk_size_t column_within_tile = 0; column_within_tile < cols_remaining; column_within_tile++) {
                svfloat32_t v0 = svread_ver_za32_f32_m(svdup_f32(NK_F32_MIN), full_predicate_b32, 0,
                                                       column_within_tile);
                running_max = svmax_f32_x(full_predicate_b32, running_max, v0);
            }
        }

        // Horizontal sum of the max vector (predicated to rows_remaining active lanes)
        total += svaddv_f32(row_predicate_b32, running_max);
    }

    return total;
}

NK_PUBLIC nk_f32_t nk_maxsim_packed_f16_sme( //
    void const *q_packed, nk_size_t n_q,     //
    void const *d_packed, nk_size_t n_d,     //
    nk_size_t depth) {

    return nk_maxsim_packed_f16_streaming_(q_packed, n_q, d_packed, n_d, depth);
}

/**
 *  MaxSim bf16 kernel: both Q and D pre-packed, vertical column read extraction.
 *
 *  4-tile fast path: processes 4 doc column tiles simultaneously using ZA0-ZA3.
 *  Inner loop per depth_step: 1 Q load + 4 D loads + 4 BFMOPA = 9 ops.
 *  Extraction per 4-tile group: 4×16 = 64 vertical reads + 64 svmax = ~128 cycles.
 *
 *  1-tile remainder: uses ZA0 only, with predicated loads for partial tiles.
 */
__arm_locally_streaming __arm_new("za") static nk_f32_t nk_maxsim_packed_bf16_streaming_( //
    void const *q_packed, nk_size_t n_q,                                                  //
    void const *d_packed, nk_size_t n_d,                                                  //
    nk_size_t depth) {

    nk_dots_sme_packed_header_t const *q_header = (nk_dots_sme_packed_header_t const *)q_packed;
    nk_dots_sme_packed_header_t const *d_header = (nk_dots_sme_packed_header_t const *)d_packed;
    nk_size_t const depth_step_count = q_header->depth_tile_count;
    nk_size_t const q_row_tiles = q_header->column_tile_count;
    nk_size_t const d_col_tiles = d_header->column_tile_count;

    nk_size_t const tile_dimension = svcntw();  // 16: ZA32 tile dimension
    nk_size_t const vector_elements = svcnth(); // 32: bf16 elements per SVE vector

    nk_bf16_t const *q_vecs = (nk_bf16_t const *)((char const *)q_packed + sizeof(nk_dots_sme_packed_header_t));
    nk_bf16_t const *d_vecs = (nk_bf16_t const *)((char const *)d_packed + sizeof(nk_dots_sme_packed_header_t));

    svbool_t const full_predicate_b16 = svptrue_b16();
    svbool_t const full_predicate_b32 = svptrue_b32();

    nk_f32_t total = 0.0f;

    for (nk_size_t row_tile_index = 0; row_tile_index < q_row_tiles; row_tile_index++) {
        nk_size_t const row_start = row_tile_index * tile_dimension;
        nk_size_t const rows_remaining = (row_start + tile_dimension <= n_q) ? tile_dimension : (n_q - row_start);
        svbool_t const row_predicate_b16 = (rows_remaining == tile_dimension)
                                               ? svptrue_b16()
                                               : svwhilelt_b16((nk_u32_t)0, (nk_u32_t)(rows_remaining * 2));
        svbool_t const row_predicate_b32 = (rows_remaining == tile_dimension)
                                               ? svptrue_b32()
                                               : svwhilelt_b32((nk_u32_t)0, (nk_u32_t)rows_remaining);

        // Running max vector: element i = max_j dot(q_{row_start+i}, d_j) seen so far
        svfloat32_t running_max = svdup_f32(NK_F32_MIN);

        nk_size_t column_tile_index = 0;

        // Fast path: 4 doc column tiles at a time using ZA0-ZA3
        for (; column_tile_index + 4 <= d_col_tiles; column_tile_index += 4) {
            svzero_za(); // Zero all 4 tiles

            // Accumulate: for each depth step, load Q vector and 4 D vectors, issue 4 BFMOPAs
            for (nk_size_t depth_step = 0; depth_step < depth_step_count; depth_step++) {
                svbfloat16_t a_packed_vector = svld1_bf16(
                    row_predicate_b16,
                    (bfloat16_t const *)(q_vecs + (row_tile_index * depth_step_count + depth_step) * vector_elements));
                svbfloat16_t b_packed_vector_0 = svld1_bf16(
                    full_predicate_b16,
                    (bfloat16_t const *)(d_vecs +
                                         ((column_tile_index + 0) * depth_step_count + depth_step) * vector_elements));
                svbfloat16_t b_packed_vector_1 = svld1_bf16(
                    full_predicate_b16,
                    (bfloat16_t const *)(d_vecs +
                                         ((column_tile_index + 1) * depth_step_count + depth_step) * vector_elements));
                svbfloat16_t b_packed_vector_2 = svld1_bf16(
                    full_predicate_b16,
                    (bfloat16_t const *)(d_vecs +
                                         ((column_tile_index + 2) * depth_step_count + depth_step) * vector_elements));
                svbfloat16_t b_packed_vector_3 = svld1_bf16(
                    full_predicate_b16,
                    (bfloat16_t const *)(d_vecs +
                                         ((column_tile_index + 3) * depth_step_count + depth_step) * vector_elements));
                svmopa_za32_bf16_m(0, row_predicate_b16, full_predicate_b16, a_packed_vector, b_packed_vector_0);
                svmopa_za32_bf16_m(1, row_predicate_b16, full_predicate_b16, a_packed_vector, b_packed_vector_1);
                svmopa_za32_bf16_m(2, row_predicate_b16, full_predicate_b16, a_packed_vector, b_packed_vector_2);
                svmopa_za32_bf16_m(3, row_predicate_b16, full_predicate_b16, a_packed_vector, b_packed_vector_3);
            }

            // Vertical column extraction:
            // ZA tile[row i][col j] = dot(q_{row_start+i}, d_{col_tile_start+j})
            // Reading column j gives a vector of dot products for all query tokens vs doc token j.
            // Element-wise max across all 64 columns (4 tiles × 16 columns) gives the per-query-token
            // maximum similarity over these 64 doc tokens.
            for (nk_size_t column_within_tile = 0; column_within_tile < tile_dimension; column_within_tile++) {
                svfloat32_t v0 = svread_ver_za32_f32_m(svdup_f32(NK_F32_MIN), full_predicate_b32, 0,
                                                       column_within_tile);
                svfloat32_t v1 = svread_ver_za32_f32_m(svdup_f32(NK_F32_MIN), full_predicate_b32, 1,
                                                       column_within_tile);
                svfloat32_t v2 = svread_ver_za32_f32_m(svdup_f32(NK_F32_MIN), full_predicate_b32, 2,
                                                       column_within_tile);
                svfloat32_t v3 = svread_ver_za32_f32_m(svdup_f32(NK_F32_MIN), full_predicate_b32, 3,
                                                       column_within_tile);
                svfloat32_t col_max = svmax_f32_x(full_predicate_b32, svmax_f32_x(full_predicate_b32, v0, v1),
                                                  svmax_f32_x(full_predicate_b32, v2, v3));
                running_max = svmax_f32_x(full_predicate_b32, running_max, col_max);
            }
        }

        // Remainder: 1 doc column tile at a time using ZA0 only
        for (; column_tile_index < d_col_tiles; column_tile_index++) {
            nk_size_t const col_start = column_tile_index * tile_dimension;
            nk_size_t const cols_remaining = (col_start + tile_dimension <= n_d) ? tile_dimension : (n_d - col_start);
            svbool_t const column_predicate_b16 = (cols_remaining == tile_dimension)
                                                      ? svptrue_b16()
                                                      : svwhilelt_b16((nk_u32_t)0, (nk_u32_t)(cols_remaining * 2));

            svzero_mask_za(nk_sme_zero_za32_tile_0_); // Zero ZA0 only

            for (nk_size_t depth_step = 0; depth_step < depth_step_count; depth_step++) {
                svbfloat16_t a_packed_vector = svld1_bf16(
                    row_predicate_b16,
                    (bfloat16_t const *)(q_vecs + (row_tile_index * depth_step_count + depth_step) * vector_elements));
                svbfloat16_t b_packed_vector = svld1_bf16(
                    column_predicate_b16,
                    (bfloat16_t const *)(d_vecs +
                                         (column_tile_index * depth_step_count + depth_step) * vector_elements));
                svmopa_za32_bf16_m(0, row_predicate_b16, column_predicate_b16, a_packed_vector, b_packed_vector);
            }

            // Vertical column extraction from ZA0 — only cols_remaining valid columns
            for (nk_size_t column_within_tile = 0; column_within_tile < cols_remaining; column_within_tile++) {
                svfloat32_t v0 = svread_ver_za32_f32_m(svdup_f32(NK_F32_MIN), full_predicate_b32, 0,
                                                       column_within_tile);
                running_max = svmax_f32_x(full_predicate_b32, running_max, v0);
            }
        }

        // Horizontal sum of the max vector (predicated to rows_remaining active lanes)
        total += svaddv_f32(row_predicate_b32, running_max);
    }

    return total;
}

NK_PUBLIC nk_f32_t nk_maxsim_packed_bf16_sme( //
    void const *q_packed, nk_size_t n_q,      //
    void const *d_packed, nk_size_t n_d,      //
    nk_size_t depth) {

    return nk_maxsim_packed_bf16_streaming_(q_packed, n_q, d_packed, n_d, depth);
}

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
#endif // NK_MAXSIM_SME_H

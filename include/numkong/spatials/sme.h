/**
 *  @brief Batched Spatial Distances for ARM SME.
 *  @file include/numkong/spatials/sme.h
 *  @author Ash Vardanian
 *  @date February 23, 2026
 *
 *  @sa include/numkong/spatials.h
 */
#ifndef NK_SPATIALS_SME_H
#define NK_SPATIALS_SME_H

#if NK_TARGET_ARM_
#if NK_TARGET_SME

#include "numkong/dots/serial.h"
#include "numkong/dots/sme.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("sme"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("+sme")
#endif

#pragma region Half Precision Floats

__arm_locally_streaming static void nk_angulars_packed_f16_sme_finalize_streaming_( //
    nk_f16_t const *a, void const *b_packed, nk_f32_t *c,                           //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,                             //
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) {
    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_f32_t const *b_norms = (nk_f32_t const *)((char const *)b_packed + header->norms_offset);
    for (nk_size_t row_index = 0; row_index < rows; row_index++) {
        nk_f16_t const *a_row = a + row_index * a_stride_elements;
        nk_f32_t *result_row = c + row_index * c_stride_elements;
        nk_f32_t query_norm_sq_f32 = nk_dots_reduce_sumsq_f16_(a_row, depth);
        svfloat32_t query_norm_sq_f32x = svdup_n_f32(query_norm_sq_f32);
        for (nk_size_t col_index = 0; col_index < columns; col_index += svcntw()) {
            svbool_t predicate_f32x = svwhilelt_b32_u64(col_index, columns);
            svfloat32_t dots_f32x = svld1_f32(predicate_f32x, result_row + col_index);
            svfloat32_t target_norms_sq_f32x = svld1_f32(predicate_f32x, b_norms + col_index);
            svfloat32_t norms_product_f32x = svmul_f32_x(predicate_f32x, query_norm_sq_f32x, target_norms_sq_f32x);
            svfloat32_t rsqrt_f32x = svrsqrte_f32(norms_product_f32x);
            rsqrt_f32x = svmul_f32_x(
                predicate_f32x, rsqrt_f32x,
                svrsqrts_f32(svmul_f32_x(predicate_f32x, norms_product_f32x, rsqrt_f32x), rsqrt_f32x));
            rsqrt_f32x = svmul_f32_x(
                predicate_f32x, rsqrt_f32x,
                svrsqrts_f32(svmul_f32_x(predicate_f32x, norms_product_f32x, rsqrt_f32x), rsqrt_f32x));
            svfloat32_t angular_f32x = svsub_f32_x(predicate_f32x, svdup_n_f32(1.0f),
                                                   svmul_f32_x(predicate_f32x, dots_f32x, rsqrt_f32x));
            angular_f32x = svmax_f32_x(predicate_f32x, angular_f32x, svdup_n_f32(0.0f));
            svst1_f32(predicate_f32x, result_row + col_index, angular_f32x);
        }
    }
}

NK_PUBLIC void nk_angulars_packed_f16_sme(                //
    nk_f16_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,   //
    nk_size_t a_stride_in_bytes, nk_size_t c_stride_in_bytes) {
    nk_size_t const a_stride_elements = a_stride_in_bytes / sizeof(nk_f16_t);
    nk_size_t const c_stride_elements = c_stride_in_bytes / sizeof(nk_f32_t);
    nk_dots_packed_f16_sme_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements, c_stride_elements);
    nk_angulars_packed_f16_sme_finalize_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                   c_stride_elements);
}

__arm_locally_streaming static void nk_euclideans_packed_f16_sme_finalize_streaming_( //
    nk_f16_t const *a, void const *b_packed, nk_f32_t *c,                             //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,                               //
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) {
    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_f32_t const *b_norms = (nk_f32_t const *)((char const *)b_packed + header->norms_offset);
    for (nk_size_t row_index = 0; row_index < rows; row_index++) {
        nk_f16_t const *a_row = a + row_index * a_stride_elements;
        nk_f32_t *result_row = c + row_index * c_stride_elements;
        nk_f32_t query_norm_sq_f32 = nk_dots_reduce_sumsq_f16_(a_row, depth);
        svfloat32_t query_norm_sq_f32x = svdup_n_f32(query_norm_sq_f32);
        for (nk_size_t col_index = 0; col_index < columns; col_index += svcntw()) {
            svbool_t predicate_f32x = svwhilelt_b32_u64(col_index, columns);
            svfloat32_t dots_f32x = svld1_f32(predicate_f32x, result_row + col_index);
            svfloat32_t target_norms_sq_f32x = svld1_f32(predicate_f32x, b_norms + col_index);
            svfloat32_t sum_sq_f32x = svadd_f32_x(predicate_f32x, query_norm_sq_f32x, target_norms_sq_f32x);
            svfloat32_t dist_sq_f32x = svsub_f32_x(predicate_f32x, sum_sq_f32x,
                                                   svmul_f32_x(predicate_f32x, svdup_n_f32(2.0f), dots_f32x));
            dist_sq_f32x = svmax_f32_x(predicate_f32x, dist_sq_f32x, svdup_n_f32(0.0f));
            svst1_f32(predicate_f32x, result_row + col_index, svsqrt_f32_x(predicate_f32x, dist_sq_f32x));
        }
    }
}

NK_PUBLIC void nk_euclideans_packed_f16_sme(              //
    nk_f16_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,   //
    nk_size_t a_stride_in_bytes, nk_size_t c_stride_in_bytes) {
    nk_size_t const a_stride_elements = a_stride_in_bytes / sizeof(nk_f16_t);
    nk_size_t const c_stride_elements = c_stride_in_bytes / sizeof(nk_f32_t);
    nk_dots_packed_f16_sme_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements, c_stride_elements);
    nk_euclideans_packed_f16_sme_finalize_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                     c_stride_elements);
}

__arm_locally_streaming static void nk_angulars_symmetric_f16_sme_finalize_streaming_(        //
    nk_f16_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride_elements, //
    nk_f32_t *result, nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) {
    // Phase 1: cache row norms on diagonal
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_f32_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_f16_(vectors + row_index * stride_elements, depth);
    }
    // Phase 2: column-first post-processing
    nk_f32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < n_vectors; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < n_vectors ? chunk_start + 256 : n_vectors;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_f16_(vectors + col * stride_elements, depth);
        for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
            nk_size_t col_start = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (col_start >= chunk_end) continue;
            nk_f32_t *result_row = result + row_index * result_stride_elements;
            svfloat32_t query_norm_sq_f32x = svdup_n_f32(result_row[row_index]);
            for (nk_size_t col_index = col_start; col_index < chunk_end; col_index += svcntw()) {
                svbool_t predicate_f32x = svwhilelt_b32_u64(col_index, chunk_end);
                svfloat32_t dots_f32x = svld1_f32(predicate_f32x, result_row + col_index);
                svfloat32_t target_norms_sq_f32x = svld1_f32(predicate_f32x, norms_cache + (col_index - chunk_start));
                svfloat32_t norms_product_f32x = svmul_f32_x(predicate_f32x, query_norm_sq_f32x, target_norms_sq_f32x);
                svfloat32_t rsqrt_f32x = svrsqrte_f32(norms_product_f32x);
                rsqrt_f32x = svmul_f32_x(
                    predicate_f32x, rsqrt_f32x,
                    svrsqrts_f32(svmul_f32_x(predicate_f32x, norms_product_f32x, rsqrt_f32x), rsqrt_f32x));
                rsqrt_f32x = svmul_f32_x(
                    predicate_f32x, rsqrt_f32x,
                    svrsqrts_f32(svmul_f32_x(predicate_f32x, norms_product_f32x, rsqrt_f32x), rsqrt_f32x));
                svfloat32_t angular_f32x = svsub_f32_x(predicate_f32x, svdup_n_f32(1.0f),
                                                       svmul_f32_x(predicate_f32x, dots_f32x, rsqrt_f32x));
                angular_f32x = svmax_f32_x(predicate_f32x, angular_f32x, svdup_n_f32(0.0f));
                svst1_f32(predicate_f32x, result_row + col_index, angular_f32x);
            }
        }
    }
    // Phase 3: zero diagonals
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NK_PUBLIC void nk_angulars_symmetric_f16_sme(                                        //
    nk_f16_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride, //
    nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count) {
    nk_size_t const stride_elements = stride / sizeof(nk_f16_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_dots_symmetric_f16_sme_streaming_(vectors, n_vectors, depth, stride_elements, result, result_stride_elements,
                                         row_start, row_count);
    nk_angulars_symmetric_f16_sme_finalize_streaming_(vectors, n_vectors, depth, stride_elements, result,
                                                      result_stride_elements, row_start, row_count);
}

__arm_locally_streaming static void nk_euclideans_symmetric_f16_sme_finalize_streaming_(      //
    nk_f16_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride_elements, //
    nk_f32_t *result, nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) {
    // Phase 1: cache row norms on diagonal
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_f32_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_f16_(vectors + row_index * stride_elements, depth);
    }
    // Phase 2: column-first post-processing
    nk_f32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < n_vectors; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < n_vectors ? chunk_start + 256 : n_vectors;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_f16_(vectors + col * stride_elements, depth);
        for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
            nk_size_t col_start = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (col_start >= chunk_end) continue;
            nk_f32_t *result_row = result + row_index * result_stride_elements;
            svfloat32_t query_norm_sq_f32x = svdup_n_f32(result_row[row_index]);
            for (nk_size_t col_index = col_start; col_index < chunk_end; col_index += svcntw()) {
                svbool_t predicate_f32x = svwhilelt_b32_u64(col_index, chunk_end);
                svfloat32_t dots_f32x = svld1_f32(predicate_f32x, result_row + col_index);
                svfloat32_t target_norms_sq_f32x = svld1_f32(predicate_f32x, norms_cache + (col_index - chunk_start));
                svfloat32_t sum_sq_f32x = svadd_f32_x(predicate_f32x, query_norm_sq_f32x, target_norms_sq_f32x);
                svfloat32_t dist_sq_f32x = svsub_f32_x(predicate_f32x, sum_sq_f32x,
                                                       svmul_f32_x(predicate_f32x, svdup_n_f32(2.0f), dots_f32x));
                dist_sq_f32x = svmax_f32_x(predicate_f32x, dist_sq_f32x, svdup_n_f32(0.0f));
                svst1_f32(predicate_f32x, result_row + col_index, svsqrt_f32_x(predicate_f32x, dist_sq_f32x));
            }
        }
    }
    // Phase 3: zero diagonals
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NK_PUBLIC void nk_euclideans_symmetric_f16_sme(                                      //
    nk_f16_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride, //
    nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count) {
    nk_size_t const stride_elements = stride / sizeof(nk_f16_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_dots_symmetric_f16_sme_streaming_(vectors, n_vectors, depth, stride_elements, result, result_stride_elements,
                                         row_start, row_count);
    nk_euclideans_symmetric_f16_sme_finalize_streaming_(vectors, n_vectors, depth, stride_elements, result,
                                                        result_stride_elements, row_start, row_count);
}

#pragma endregion // Half Precision Floats

#pragma region Brain Float 16

__arm_locally_streaming static void nk_angulars_packed_bf16_sme_finalize_streaming_( //
    nk_bf16_t const *a, void const *b_packed, nk_f32_t *c,                           //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,                              //
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) {
    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_f32_t const *b_norms = (nk_f32_t const *)((char const *)b_packed + header->norms_offset);
    for (nk_size_t row_index = 0; row_index < rows; row_index++) {
        nk_bf16_t const *a_row = a + row_index * a_stride_elements;
        nk_f32_t *result_row = c + row_index * c_stride_elements;
        nk_f32_t query_norm_sq_f32 = nk_dots_reduce_sumsq_bf16_(a_row, depth);
        svfloat32_t query_norm_sq_f32x = svdup_n_f32(query_norm_sq_f32);
        for (nk_size_t col_index = 0; col_index < columns; col_index += svcntw()) {
            svbool_t predicate_f32x = svwhilelt_b32_u64(col_index, columns);
            svfloat32_t dots_f32x = svld1_f32(predicate_f32x, result_row + col_index);
            svfloat32_t target_norms_sq_f32x = svld1_f32(predicate_f32x, b_norms + col_index);
            svfloat32_t norms_product_f32x = svmul_f32_x(predicate_f32x, query_norm_sq_f32x, target_norms_sq_f32x);
            svfloat32_t rsqrt_f32x = svrsqrte_f32(norms_product_f32x);
            rsqrt_f32x = svmul_f32_x(
                predicate_f32x, rsqrt_f32x,
                svrsqrts_f32(svmul_f32_x(predicate_f32x, norms_product_f32x, rsqrt_f32x), rsqrt_f32x));
            rsqrt_f32x = svmul_f32_x(
                predicate_f32x, rsqrt_f32x,
                svrsqrts_f32(svmul_f32_x(predicate_f32x, norms_product_f32x, rsqrt_f32x), rsqrt_f32x));
            svfloat32_t angular_f32x = svsub_f32_x(predicate_f32x, svdup_n_f32(1.0f),
                                                   svmul_f32_x(predicate_f32x, dots_f32x, rsqrt_f32x));
            angular_f32x = svmax_f32_x(predicate_f32x, angular_f32x, svdup_n_f32(0.0f));
            svst1_f32(predicate_f32x, result_row + col_index, angular_f32x);
        }
    }
}

NK_PUBLIC void nk_angulars_packed_bf16_sme(                //
    nk_bf16_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,    //
    nk_size_t a_stride_in_bytes, nk_size_t c_stride_in_bytes) {
    nk_size_t const a_stride_elements = a_stride_in_bytes / sizeof(nk_bf16_t);
    nk_size_t const c_stride_elements = c_stride_in_bytes / sizeof(nk_f32_t);
    nk_dots_packed_bf16_sme_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements, c_stride_elements);
    nk_angulars_packed_bf16_sme_finalize_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                    c_stride_elements);
}

__arm_locally_streaming static void nk_euclideans_packed_bf16_sme_finalize_streaming_( //
    nk_bf16_t const *a, void const *b_packed, nk_f32_t *c,                             //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,                                //
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) {
    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_f32_t const *b_norms = (nk_f32_t const *)((char const *)b_packed + header->norms_offset);
    for (nk_size_t row_index = 0; row_index < rows; row_index++) {
        nk_bf16_t const *a_row = a + row_index * a_stride_elements;
        nk_f32_t *result_row = c + row_index * c_stride_elements;
        nk_f32_t query_norm_sq_f32 = nk_dots_reduce_sumsq_bf16_(a_row, depth);
        svfloat32_t query_norm_sq_f32x = svdup_n_f32(query_norm_sq_f32);
        for (nk_size_t col_index = 0; col_index < columns; col_index += svcntw()) {
            svbool_t predicate_f32x = svwhilelt_b32_u64(col_index, columns);
            svfloat32_t dots_f32x = svld1_f32(predicate_f32x, result_row + col_index);
            svfloat32_t target_norms_sq_f32x = svld1_f32(predicate_f32x, b_norms + col_index);
            svfloat32_t sum_sq_f32x = svadd_f32_x(predicate_f32x, query_norm_sq_f32x, target_norms_sq_f32x);
            svfloat32_t dist_sq_f32x = svsub_f32_x(predicate_f32x, sum_sq_f32x,
                                                   svmul_f32_x(predicate_f32x, svdup_n_f32(2.0f), dots_f32x));
            dist_sq_f32x = svmax_f32_x(predicate_f32x, dist_sq_f32x, svdup_n_f32(0.0f));
            svst1_f32(predicate_f32x, result_row + col_index, svsqrt_f32_x(predicate_f32x, dist_sq_f32x));
        }
    }
}

NK_PUBLIC void nk_euclideans_packed_bf16_sme(              //
    nk_bf16_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,    //
    nk_size_t a_stride_in_bytes, nk_size_t c_stride_in_bytes) {
    nk_size_t const a_stride_elements = a_stride_in_bytes / sizeof(nk_bf16_t);
    nk_size_t const c_stride_elements = c_stride_in_bytes / sizeof(nk_f32_t);
    nk_dots_packed_bf16_sme_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements, c_stride_elements);
    nk_euclideans_packed_bf16_sme_finalize_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                      c_stride_elements);
}

__arm_locally_streaming static void nk_angulars_symmetric_bf16_sme_finalize_streaming_(        //
    nk_bf16_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride_elements, //
    nk_f32_t *result, nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) {
    // Phase 1: cache row norms on diagonal
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_f32_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_bf16_(vectors + row_index * stride_elements, depth);
    }
    // Phase 2: column-first post-processing
    nk_f32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < n_vectors; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < n_vectors ? chunk_start + 256 : n_vectors;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_bf16_(vectors + col * stride_elements, depth);
        for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
            nk_size_t col_start = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (col_start >= chunk_end) continue;
            nk_f32_t *result_row = result + row_index * result_stride_elements;
            svfloat32_t query_norm_sq_f32x = svdup_n_f32(result_row[row_index]);
            for (nk_size_t col_index = col_start; col_index < chunk_end; col_index += svcntw()) {
                svbool_t predicate_f32x = svwhilelt_b32_u64(col_index, chunk_end);
                svfloat32_t dots_f32x = svld1_f32(predicate_f32x, result_row + col_index);
                svfloat32_t target_norms_sq_f32x = svld1_f32(predicate_f32x, norms_cache + (col_index - chunk_start));
                svfloat32_t norms_product_f32x = svmul_f32_x(predicate_f32x, query_norm_sq_f32x, target_norms_sq_f32x);
                svfloat32_t rsqrt_f32x = svrsqrte_f32(norms_product_f32x);
                rsqrt_f32x = svmul_f32_x(
                    predicate_f32x, rsqrt_f32x,
                    svrsqrts_f32(svmul_f32_x(predicate_f32x, norms_product_f32x, rsqrt_f32x), rsqrt_f32x));
                rsqrt_f32x = svmul_f32_x(
                    predicate_f32x, rsqrt_f32x,
                    svrsqrts_f32(svmul_f32_x(predicate_f32x, norms_product_f32x, rsqrt_f32x), rsqrt_f32x));
                svfloat32_t angular_f32x = svsub_f32_x(predicate_f32x, svdup_n_f32(1.0f),
                                                       svmul_f32_x(predicate_f32x, dots_f32x, rsqrt_f32x));
                angular_f32x = svmax_f32_x(predicate_f32x, angular_f32x, svdup_n_f32(0.0f));
                svst1_f32(predicate_f32x, result_row + col_index, angular_f32x);
            }
        }
    }
    // Phase 3: zero diagonals
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NK_PUBLIC void nk_angulars_symmetric_bf16_sme(                                        //
    nk_bf16_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride, //
    nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count) {
    nk_size_t const stride_elements = stride / sizeof(nk_bf16_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_dots_symmetric_bf16_sme_streaming_(vectors, n_vectors, depth, stride_elements, result, result_stride_elements,
                                          row_start, row_count);
    nk_angulars_symmetric_bf16_sme_finalize_streaming_(vectors, n_vectors, depth, stride_elements, result,
                                                       result_stride_elements, row_start, row_count);
}

__arm_locally_streaming static void nk_euclideans_symmetric_bf16_sme_finalize_streaming_(      //
    nk_bf16_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride_elements, //
    nk_f32_t *result, nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) {
    // Phase 1: cache row norms on diagonal
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_f32_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_bf16_(vectors + row_index * stride_elements, depth);
    }
    // Phase 2: column-first post-processing
    nk_f32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < n_vectors; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < n_vectors ? chunk_start + 256 : n_vectors;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_bf16_(vectors + col * stride_elements, depth);
        for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
            nk_size_t col_start = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (col_start >= chunk_end) continue;
            nk_f32_t *result_row = result + row_index * result_stride_elements;
            svfloat32_t query_norm_sq_f32x = svdup_n_f32(result_row[row_index]);
            for (nk_size_t col_index = col_start; col_index < chunk_end; col_index += svcntw()) {
                svbool_t predicate_f32x = svwhilelt_b32_u64(col_index, chunk_end);
                svfloat32_t dots_f32x = svld1_f32(predicate_f32x, result_row + col_index);
                svfloat32_t target_norms_sq_f32x = svld1_f32(predicate_f32x, norms_cache + (col_index - chunk_start));
                svfloat32_t sum_sq_f32x = svadd_f32_x(predicate_f32x, query_norm_sq_f32x, target_norms_sq_f32x);
                svfloat32_t dist_sq_f32x = svsub_f32_x(predicate_f32x, sum_sq_f32x,
                                                       svmul_f32_x(predicate_f32x, svdup_n_f32(2.0f), dots_f32x));
                dist_sq_f32x = svmax_f32_x(predicate_f32x, dist_sq_f32x, svdup_n_f32(0.0f));
                svst1_f32(predicate_f32x, result_row + col_index, svsqrt_f32_x(predicate_f32x, dist_sq_f32x));
            }
        }
    }
    // Phase 3: zero diagonals
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NK_PUBLIC void nk_euclideans_symmetric_bf16_sme(                                      //
    nk_bf16_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride, //
    nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count) {
    nk_size_t const stride_elements = stride / sizeof(nk_bf16_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_dots_symmetric_bf16_sme_streaming_(vectors, n_vectors, depth, stride_elements, result, result_stride_elements,
                                          row_start, row_count);
    nk_euclideans_symmetric_bf16_sme_finalize_streaming_(vectors, n_vectors, depth, stride_elements, result,
                                                         result_stride_elements, row_start, row_count);
}

#pragma endregion // Brain Float 16

#pragma region Quarter Precision E4M3

__arm_locally_streaming static void nk_angulars_packed_e4m3_sme_finalize_streaming_( //
    nk_e4m3_t const *a, void const *b_packed, nk_f32_t *c,                           //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,                              //
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) {
    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_f32_t const *b_norms = (nk_f32_t const *)((char const *)b_packed + header->norms_offset);
    for (nk_size_t row_index = 0; row_index < rows; row_index++) {
        nk_e4m3_t const *a_row = a + row_index * a_stride_elements;
        nk_f32_t *result_row = c + row_index * c_stride_elements;
        nk_f32_t query_norm_sq_f32 = nk_dots_reduce_sumsq_e4m3_(a_row, depth);
        svfloat32_t query_norm_sq_f32x = svdup_n_f32(query_norm_sq_f32);
        for (nk_size_t col_index = 0; col_index < columns; col_index += svcntw()) {
            svbool_t predicate_f32x = svwhilelt_b32_u64(col_index, columns);
            svfloat32_t dots_f32x = svld1_f32(predicate_f32x, result_row + col_index);
            svfloat32_t target_norms_sq_f32x = svld1_f32(predicate_f32x, b_norms + col_index);
            svfloat32_t norms_product_f32x = svmul_f32_x(predicate_f32x, query_norm_sq_f32x, target_norms_sq_f32x);
            svfloat32_t rsqrt_f32x = svrsqrte_f32(norms_product_f32x);
            rsqrt_f32x = svmul_f32_x(
                predicate_f32x, rsqrt_f32x,
                svrsqrts_f32(svmul_f32_x(predicate_f32x, norms_product_f32x, rsqrt_f32x), rsqrt_f32x));
            rsqrt_f32x = svmul_f32_x(
                predicate_f32x, rsqrt_f32x,
                svrsqrts_f32(svmul_f32_x(predicate_f32x, norms_product_f32x, rsqrt_f32x), rsqrt_f32x));
            svfloat32_t angular_f32x = svsub_f32_x(predicate_f32x, svdup_n_f32(1.0f),
                                                   svmul_f32_x(predicate_f32x, dots_f32x, rsqrt_f32x));
            angular_f32x = svmax_f32_x(predicate_f32x, angular_f32x, svdup_n_f32(0.0f));
            svst1_f32(predicate_f32x, result_row + col_index, angular_f32x);
        }
    }
}

NK_PUBLIC void nk_angulars_packed_e4m3_sme(                //
    nk_e4m3_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,    //
    nk_size_t a_stride_in_bytes, nk_size_t c_stride_in_bytes) {
    nk_size_t const a_stride_elements = a_stride_in_bytes / sizeof(nk_e4m3_t);
    nk_size_t const c_stride_elements = c_stride_in_bytes / sizeof(nk_f32_t);
    nk_dots_packed_e4m3_sme_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements, c_stride_elements);
    nk_angulars_packed_e4m3_sme_finalize_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                    c_stride_elements);
}

__arm_locally_streaming static void nk_euclideans_packed_e4m3_sme_finalize_streaming_( //
    nk_e4m3_t const *a, void const *b_packed, nk_f32_t *c,                             //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,                                //
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) {
    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_f32_t const *b_norms = (nk_f32_t const *)((char const *)b_packed + header->norms_offset);
    for (nk_size_t row_index = 0; row_index < rows; row_index++) {
        nk_e4m3_t const *a_row = a + row_index * a_stride_elements;
        nk_f32_t *result_row = c + row_index * c_stride_elements;
        nk_f32_t query_norm_sq_f32 = nk_dots_reduce_sumsq_e4m3_(a_row, depth);
        svfloat32_t query_norm_sq_f32x = svdup_n_f32(query_norm_sq_f32);
        for (nk_size_t col_index = 0; col_index < columns; col_index += svcntw()) {
            svbool_t predicate_f32x = svwhilelt_b32_u64(col_index, columns);
            svfloat32_t dots_f32x = svld1_f32(predicate_f32x, result_row + col_index);
            svfloat32_t target_norms_sq_f32x = svld1_f32(predicate_f32x, b_norms + col_index);
            svfloat32_t sum_sq_f32x = svadd_f32_x(predicate_f32x, query_norm_sq_f32x, target_norms_sq_f32x);
            svfloat32_t dist_sq_f32x = svsub_f32_x(predicate_f32x, sum_sq_f32x,
                                                   svmul_f32_x(predicate_f32x, svdup_n_f32(2.0f), dots_f32x));
            dist_sq_f32x = svmax_f32_x(predicate_f32x, dist_sq_f32x, svdup_n_f32(0.0f));
            svst1_f32(predicate_f32x, result_row + col_index, svsqrt_f32_x(predicate_f32x, dist_sq_f32x));
        }
    }
}

NK_PUBLIC void nk_euclideans_packed_e4m3_sme(              //
    nk_e4m3_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,    //
    nk_size_t a_stride_in_bytes, nk_size_t c_stride_in_bytes) {
    nk_size_t const a_stride_elements = a_stride_in_bytes / sizeof(nk_e4m3_t);
    nk_size_t const c_stride_elements = c_stride_in_bytes / sizeof(nk_f32_t);
    nk_dots_packed_e4m3_sme_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements, c_stride_elements);
    nk_euclideans_packed_e4m3_sme_finalize_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                      c_stride_elements);
}

__arm_locally_streaming static void nk_angulars_symmetric_e4m3_sme_finalize_streaming_(        //
    nk_e4m3_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride_elements, //
    nk_f32_t *result, nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) {
    // Phase 1: cache row norms on diagonal
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_f32_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_e4m3_(vectors + row_index * stride_elements, depth);
    }
    // Phase 2: column-first post-processing
    nk_f32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < n_vectors; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < n_vectors ? chunk_start + 256 : n_vectors;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_e4m3_(vectors + col * stride_elements, depth);
        for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
            nk_size_t col_start = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (col_start >= chunk_end) continue;
            nk_f32_t *result_row = result + row_index * result_stride_elements;
            svfloat32_t query_norm_sq_f32x = svdup_n_f32(result_row[row_index]);
            for (nk_size_t col_index = col_start; col_index < chunk_end; col_index += svcntw()) {
                svbool_t predicate_f32x = svwhilelt_b32_u64(col_index, chunk_end);
                svfloat32_t dots_f32x = svld1_f32(predicate_f32x, result_row + col_index);
                svfloat32_t target_norms_sq_f32x = svld1_f32(predicate_f32x, norms_cache + (col_index - chunk_start));
                svfloat32_t norms_product_f32x = svmul_f32_x(predicate_f32x, query_norm_sq_f32x, target_norms_sq_f32x);
                svfloat32_t rsqrt_f32x = svrsqrte_f32(norms_product_f32x);
                rsqrt_f32x = svmul_f32_x(
                    predicate_f32x, rsqrt_f32x,
                    svrsqrts_f32(svmul_f32_x(predicate_f32x, norms_product_f32x, rsqrt_f32x), rsqrt_f32x));
                rsqrt_f32x = svmul_f32_x(
                    predicate_f32x, rsqrt_f32x,
                    svrsqrts_f32(svmul_f32_x(predicate_f32x, norms_product_f32x, rsqrt_f32x), rsqrt_f32x));
                svfloat32_t angular_f32x = svsub_f32_x(predicate_f32x, svdup_n_f32(1.0f),
                                                       svmul_f32_x(predicate_f32x, dots_f32x, rsqrt_f32x));
                angular_f32x = svmax_f32_x(predicate_f32x, angular_f32x, svdup_n_f32(0.0f));
                svst1_f32(predicate_f32x, result_row + col_index, angular_f32x);
            }
        }
    }
    // Phase 3: zero diagonals
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NK_PUBLIC void nk_angulars_symmetric_e4m3_sme(                                        //
    nk_e4m3_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride, //
    nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count) {
    nk_size_t const stride_elements = stride / sizeof(nk_e4m3_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_dots_symmetric_e4m3_sme_streaming_(vectors, n_vectors, depth, stride_elements, result, result_stride_elements,
                                          row_start, row_count);
    nk_angulars_symmetric_e4m3_sme_finalize_streaming_(vectors, n_vectors, depth, stride_elements, result,
                                                       result_stride_elements, row_start, row_count);
}

__arm_locally_streaming static void nk_euclideans_symmetric_e4m3_sme_finalize_streaming_(      //
    nk_e4m3_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride_elements, //
    nk_f32_t *result, nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) {
    // Phase 1: cache row norms on diagonal
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_f32_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_e4m3_(vectors + row_index * stride_elements, depth);
    }
    // Phase 2: column-first post-processing
    nk_f32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < n_vectors; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < n_vectors ? chunk_start + 256 : n_vectors;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_e4m3_(vectors + col * stride_elements, depth);
        for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
            nk_size_t col_start = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (col_start >= chunk_end) continue;
            nk_f32_t *result_row = result + row_index * result_stride_elements;
            svfloat32_t query_norm_sq_f32x = svdup_n_f32(result_row[row_index]);
            for (nk_size_t col_index = col_start; col_index < chunk_end; col_index += svcntw()) {
                svbool_t predicate_f32x = svwhilelt_b32_u64(col_index, chunk_end);
                svfloat32_t dots_f32x = svld1_f32(predicate_f32x, result_row + col_index);
                svfloat32_t target_norms_sq_f32x = svld1_f32(predicate_f32x, norms_cache + (col_index - chunk_start));
                svfloat32_t sum_sq_f32x = svadd_f32_x(predicate_f32x, query_norm_sq_f32x, target_norms_sq_f32x);
                svfloat32_t dist_sq_f32x = svsub_f32_x(predicate_f32x, sum_sq_f32x,
                                                       svmul_f32_x(predicate_f32x, svdup_n_f32(2.0f), dots_f32x));
                dist_sq_f32x = svmax_f32_x(predicate_f32x, dist_sq_f32x, svdup_n_f32(0.0f));
                svst1_f32(predicate_f32x, result_row + col_index, svsqrt_f32_x(predicate_f32x, dist_sq_f32x));
            }
        }
    }
    // Phase 3: zero diagonals
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NK_PUBLIC void nk_euclideans_symmetric_e4m3_sme(                                      //
    nk_e4m3_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride, //
    nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count) {
    nk_size_t const stride_elements = stride / sizeof(nk_e4m3_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_dots_symmetric_e4m3_sme_streaming_(vectors, n_vectors, depth, stride_elements, result, result_stride_elements,
                                          row_start, row_count);
    nk_euclideans_symmetric_e4m3_sme_finalize_streaming_(vectors, n_vectors, depth, stride_elements, result,
                                                         result_stride_elements, row_start, row_count);
}

#pragma endregion // Quarter Precision E4M3

#pragma region Quarter Precision E5M2

__arm_locally_streaming static void nk_angulars_packed_e5m2_sme_finalize_streaming_( //
    nk_e5m2_t const *a, void const *b_packed, nk_f32_t *c,                           //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,                              //
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) {
    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_f32_t const *b_norms = (nk_f32_t const *)((char const *)b_packed + header->norms_offset);
    for (nk_size_t row_index = 0; row_index < rows; row_index++) {
        nk_e5m2_t const *a_row = a + row_index * a_stride_elements;
        nk_f32_t *result_row = c + row_index * c_stride_elements;
        nk_f32_t query_norm_sq_f32 = nk_dots_reduce_sumsq_e5m2_(a_row, depth);
        svfloat32_t query_norm_sq_f32x = svdup_n_f32(query_norm_sq_f32);
        for (nk_size_t col_index = 0; col_index < columns; col_index += svcntw()) {
            svbool_t predicate_f32x = svwhilelt_b32_u64(col_index, columns);
            svfloat32_t dots_f32x = svld1_f32(predicate_f32x, result_row + col_index);
            svfloat32_t target_norms_sq_f32x = svld1_f32(predicate_f32x, b_norms + col_index);
            svfloat32_t norms_product_f32x = svmul_f32_x(predicate_f32x, query_norm_sq_f32x, target_norms_sq_f32x);
            svfloat32_t rsqrt_f32x = svrsqrte_f32(norms_product_f32x);
            rsqrt_f32x = svmul_f32_x(
                predicate_f32x, rsqrt_f32x,
                svrsqrts_f32(svmul_f32_x(predicate_f32x, norms_product_f32x, rsqrt_f32x), rsqrt_f32x));
            rsqrt_f32x = svmul_f32_x(
                predicate_f32x, rsqrt_f32x,
                svrsqrts_f32(svmul_f32_x(predicate_f32x, norms_product_f32x, rsqrt_f32x), rsqrt_f32x));
            svfloat32_t angular_f32x = svsub_f32_x(predicate_f32x, svdup_n_f32(1.0f),
                                                   svmul_f32_x(predicate_f32x, dots_f32x, rsqrt_f32x));
            angular_f32x = svmax_f32_x(predicate_f32x, angular_f32x, svdup_n_f32(0.0f));
            svst1_f32(predicate_f32x, result_row + col_index, angular_f32x);
        }
    }
}

NK_PUBLIC void nk_angulars_packed_e5m2_sme(                //
    nk_e5m2_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,    //
    nk_size_t a_stride_in_bytes, nk_size_t c_stride_in_bytes) {
    nk_size_t const a_stride_elements = a_stride_in_bytes / sizeof(nk_e5m2_t);
    nk_size_t const c_stride_elements = c_stride_in_bytes / sizeof(nk_f32_t);
    nk_dots_packed_e5m2_sme_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements, c_stride_elements);
    nk_angulars_packed_e5m2_sme_finalize_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                    c_stride_elements);
}

__arm_locally_streaming static void nk_euclideans_packed_e5m2_sme_finalize_streaming_( //
    nk_e5m2_t const *a, void const *b_packed, nk_f32_t *c,                             //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,                                //
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) {
    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_f32_t const *b_norms = (nk_f32_t const *)((char const *)b_packed + header->norms_offset);
    for (nk_size_t row_index = 0; row_index < rows; row_index++) {
        nk_e5m2_t const *a_row = a + row_index * a_stride_elements;
        nk_f32_t *result_row = c + row_index * c_stride_elements;
        nk_f32_t query_norm_sq_f32 = nk_dots_reduce_sumsq_e5m2_(a_row, depth);
        svfloat32_t query_norm_sq_f32x = svdup_n_f32(query_norm_sq_f32);
        for (nk_size_t col_index = 0; col_index < columns; col_index += svcntw()) {
            svbool_t predicate_f32x = svwhilelt_b32_u64(col_index, columns);
            svfloat32_t dots_f32x = svld1_f32(predicate_f32x, result_row + col_index);
            svfloat32_t target_norms_sq_f32x = svld1_f32(predicate_f32x, b_norms + col_index);
            svfloat32_t sum_sq_f32x = svadd_f32_x(predicate_f32x, query_norm_sq_f32x, target_norms_sq_f32x);
            svfloat32_t dist_sq_f32x = svsub_f32_x(predicate_f32x, sum_sq_f32x,
                                                   svmul_f32_x(predicate_f32x, svdup_n_f32(2.0f), dots_f32x));
            dist_sq_f32x = svmax_f32_x(predicate_f32x, dist_sq_f32x, svdup_n_f32(0.0f));
            svst1_f32(predicate_f32x, result_row + col_index, svsqrt_f32_x(predicate_f32x, dist_sq_f32x));
        }
    }
}

NK_PUBLIC void nk_euclideans_packed_e5m2_sme(              //
    nk_e5m2_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,    //
    nk_size_t a_stride_in_bytes, nk_size_t c_stride_in_bytes) {
    nk_size_t const a_stride_elements = a_stride_in_bytes / sizeof(nk_e5m2_t);
    nk_size_t const c_stride_elements = c_stride_in_bytes / sizeof(nk_f32_t);
    nk_dots_packed_e5m2_sme_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements, c_stride_elements);
    nk_euclideans_packed_e5m2_sme_finalize_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                      c_stride_elements);
}

__arm_locally_streaming static void nk_angulars_symmetric_e5m2_sme_finalize_streaming_(        //
    nk_e5m2_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride_elements, //
    nk_f32_t *result, nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) {
    // Phase 1: cache row norms on diagonal
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_f32_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_e5m2_(vectors + row_index * stride_elements, depth);
    }
    // Phase 2: column-first post-processing
    nk_f32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < n_vectors; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < n_vectors ? chunk_start + 256 : n_vectors;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_e5m2_(vectors + col * stride_elements, depth);
        for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
            nk_size_t col_start = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (col_start >= chunk_end) continue;
            nk_f32_t *result_row = result + row_index * result_stride_elements;
            svfloat32_t query_norm_sq_f32x = svdup_n_f32(result_row[row_index]);
            for (nk_size_t col_index = col_start; col_index < chunk_end; col_index += svcntw()) {
                svbool_t predicate_f32x = svwhilelt_b32_u64(col_index, chunk_end);
                svfloat32_t dots_f32x = svld1_f32(predicate_f32x, result_row + col_index);
                svfloat32_t target_norms_sq_f32x = svld1_f32(predicate_f32x, norms_cache + (col_index - chunk_start));
                svfloat32_t norms_product_f32x = svmul_f32_x(predicate_f32x, query_norm_sq_f32x, target_norms_sq_f32x);
                svfloat32_t rsqrt_f32x = svrsqrte_f32(norms_product_f32x);
                rsqrt_f32x = svmul_f32_x(
                    predicate_f32x, rsqrt_f32x,
                    svrsqrts_f32(svmul_f32_x(predicate_f32x, norms_product_f32x, rsqrt_f32x), rsqrt_f32x));
                rsqrt_f32x = svmul_f32_x(
                    predicate_f32x, rsqrt_f32x,
                    svrsqrts_f32(svmul_f32_x(predicate_f32x, norms_product_f32x, rsqrt_f32x), rsqrt_f32x));
                svfloat32_t angular_f32x = svsub_f32_x(predicate_f32x, svdup_n_f32(1.0f),
                                                       svmul_f32_x(predicate_f32x, dots_f32x, rsqrt_f32x));
                angular_f32x = svmax_f32_x(predicate_f32x, angular_f32x, svdup_n_f32(0.0f));
                svst1_f32(predicate_f32x, result_row + col_index, angular_f32x);
            }
        }
    }
    // Phase 3: zero diagonals
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NK_PUBLIC void nk_angulars_symmetric_e5m2_sme(                                        //
    nk_e5m2_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride, //
    nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count) {
    nk_size_t const stride_elements = stride / sizeof(nk_e5m2_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_dots_symmetric_e5m2_sme_streaming_(vectors, n_vectors, depth, stride_elements, result, result_stride_elements,
                                          row_start, row_count);
    nk_angulars_symmetric_e5m2_sme_finalize_streaming_(vectors, n_vectors, depth, stride_elements, result,
                                                       result_stride_elements, row_start, row_count);
}

__arm_locally_streaming static void nk_euclideans_symmetric_e5m2_sme_finalize_streaming_(      //
    nk_e5m2_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride_elements, //
    nk_f32_t *result, nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) {
    // Phase 1: cache row norms on diagonal
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_f32_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_e5m2_(vectors + row_index * stride_elements, depth);
    }
    // Phase 2: column-first post-processing
    nk_f32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < n_vectors; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < n_vectors ? chunk_start + 256 : n_vectors;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_e5m2_(vectors + col * stride_elements, depth);
        for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
            nk_size_t col_start = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (col_start >= chunk_end) continue;
            nk_f32_t *result_row = result + row_index * result_stride_elements;
            svfloat32_t query_norm_sq_f32x = svdup_n_f32(result_row[row_index]);
            for (nk_size_t col_index = col_start; col_index < chunk_end; col_index += svcntw()) {
                svbool_t predicate_f32x = svwhilelt_b32_u64(col_index, chunk_end);
                svfloat32_t dots_f32x = svld1_f32(predicate_f32x, result_row + col_index);
                svfloat32_t target_norms_sq_f32x = svld1_f32(predicate_f32x, norms_cache + (col_index - chunk_start));
                svfloat32_t sum_sq_f32x = svadd_f32_x(predicate_f32x, query_norm_sq_f32x, target_norms_sq_f32x);
                svfloat32_t dist_sq_f32x = svsub_f32_x(predicate_f32x, sum_sq_f32x,
                                                       svmul_f32_x(predicate_f32x, svdup_n_f32(2.0f), dots_f32x));
                dist_sq_f32x = svmax_f32_x(predicate_f32x, dist_sq_f32x, svdup_n_f32(0.0f));
                svst1_f32(predicate_f32x, result_row + col_index, svsqrt_f32_x(predicate_f32x, dist_sq_f32x));
            }
        }
    }
    // Phase 3: zero diagonals
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NK_PUBLIC void nk_euclideans_symmetric_e5m2_sme(                                      //
    nk_e5m2_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride, //
    nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count) {
    nk_size_t const stride_elements = stride / sizeof(nk_e5m2_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_dots_symmetric_e5m2_sme_streaming_(vectors, n_vectors, depth, stride_elements, result, result_stride_elements,
                                          row_start, row_count);
    nk_euclideans_symmetric_e5m2_sme_finalize_streaming_(vectors, n_vectors, depth, stride_elements, result,
                                                         result_stride_elements, row_start, row_count);
}

#pragma endregion // Quarter Precision E5M2

#pragma region Micro Precision E2M3

__arm_locally_streaming static void nk_angulars_packed_e2m3_sme_finalize_streaming_( //
    nk_e2m3_t const *a, void const *b_packed, nk_f32_t *c,                           //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,                              //
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) {
    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_f32_t const *b_norms = (nk_f32_t const *)((char const *)b_packed + header->norms_offset);
    for (nk_size_t row_index = 0; row_index < rows; row_index++) {
        nk_e2m3_t const *a_row = a + row_index * a_stride_elements;
        nk_f32_t *result_row = c + row_index * c_stride_elements;
        nk_f32_t query_norm_sq_f32 = nk_dots_reduce_sumsq_e2m3_(a_row, depth);
        svfloat32_t query_norm_sq_f32x = svdup_n_f32(query_norm_sq_f32);
        for (nk_size_t col_index = 0; col_index < columns; col_index += svcntw()) {
            svbool_t predicate_f32x = svwhilelt_b32_u64(col_index, columns);
            svfloat32_t dots_f32x = svld1_f32(predicate_f32x, result_row + col_index);
            svfloat32_t target_norms_sq_f32x = svld1_f32(predicate_f32x, b_norms + col_index);
            svfloat32_t norms_product_f32x = svmul_f32_x(predicate_f32x, query_norm_sq_f32x, target_norms_sq_f32x);
            svfloat32_t rsqrt_f32x = svrsqrte_f32(norms_product_f32x);
            rsqrt_f32x = svmul_f32_x(
                predicate_f32x, rsqrt_f32x,
                svrsqrts_f32(svmul_f32_x(predicate_f32x, norms_product_f32x, rsqrt_f32x), rsqrt_f32x));
            rsqrt_f32x = svmul_f32_x(
                predicate_f32x, rsqrt_f32x,
                svrsqrts_f32(svmul_f32_x(predicate_f32x, norms_product_f32x, rsqrt_f32x), rsqrt_f32x));
            svfloat32_t angular_f32x = svsub_f32_x(predicate_f32x, svdup_n_f32(1.0f),
                                                   svmul_f32_x(predicate_f32x, dots_f32x, rsqrt_f32x));
            angular_f32x = svmax_f32_x(predicate_f32x, angular_f32x, svdup_n_f32(0.0f));
            svst1_f32(predicate_f32x, result_row + col_index, angular_f32x);
        }
    }
}

NK_PUBLIC void nk_angulars_packed_e2m3_sme(                //
    nk_e2m3_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,    //
    nk_size_t a_stride_in_bytes, nk_size_t c_stride_in_bytes) {
    nk_size_t const a_stride_elements = a_stride_in_bytes / sizeof(nk_e2m3_t);
    nk_size_t const c_stride_elements = c_stride_in_bytes / sizeof(nk_f32_t);
    nk_dots_packed_e2m3_sme_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements, c_stride_elements);
    nk_angulars_packed_e2m3_sme_finalize_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                    c_stride_elements);
}

__arm_locally_streaming static void nk_euclideans_packed_e2m3_sme_finalize_streaming_( //
    nk_e2m3_t const *a, void const *b_packed, nk_f32_t *c,                             //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,                                //
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) {
    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_f32_t const *b_norms = (nk_f32_t const *)((char const *)b_packed + header->norms_offset);
    for (nk_size_t row_index = 0; row_index < rows; row_index++) {
        nk_e2m3_t const *a_row = a + row_index * a_stride_elements;
        nk_f32_t *result_row = c + row_index * c_stride_elements;
        nk_f32_t query_norm_sq_f32 = nk_dots_reduce_sumsq_e2m3_(a_row, depth);
        svfloat32_t query_norm_sq_f32x = svdup_n_f32(query_norm_sq_f32);
        for (nk_size_t col_index = 0; col_index < columns; col_index += svcntw()) {
            svbool_t predicate_f32x = svwhilelt_b32_u64(col_index, columns);
            svfloat32_t dots_f32x = svld1_f32(predicate_f32x, result_row + col_index);
            svfloat32_t target_norms_sq_f32x = svld1_f32(predicate_f32x, b_norms + col_index);
            svfloat32_t sum_sq_f32x = svadd_f32_x(predicate_f32x, query_norm_sq_f32x, target_norms_sq_f32x);
            svfloat32_t dist_sq_f32x = svsub_f32_x(predicate_f32x, sum_sq_f32x,
                                                   svmul_f32_x(predicate_f32x, svdup_n_f32(2.0f), dots_f32x));
            dist_sq_f32x = svmax_f32_x(predicate_f32x, dist_sq_f32x, svdup_n_f32(0.0f));
            svst1_f32(predicate_f32x, result_row + col_index, svsqrt_f32_x(predicate_f32x, dist_sq_f32x));
        }
    }
}

NK_PUBLIC void nk_euclideans_packed_e2m3_sme(              //
    nk_e2m3_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,    //
    nk_size_t a_stride_in_bytes, nk_size_t c_stride_in_bytes) {
    nk_size_t const a_stride_elements = a_stride_in_bytes / sizeof(nk_e2m3_t);
    nk_size_t const c_stride_elements = c_stride_in_bytes / sizeof(nk_f32_t);
    nk_dots_packed_e2m3_sme_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements, c_stride_elements);
    nk_euclideans_packed_e2m3_sme_finalize_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                      c_stride_elements);
}

__arm_locally_streaming static void nk_angulars_symmetric_e2m3_sme_finalize_streaming_(        //
    nk_e2m3_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride_elements, //
    nk_f32_t *result, nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) {
    // Phase 1: cache row norms on diagonal
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_f32_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_e2m3_(vectors + row_index * stride_elements, depth);
    }
    // Phase 2: column-first post-processing
    nk_f32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < n_vectors; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < n_vectors ? chunk_start + 256 : n_vectors;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_e2m3_(vectors + col * stride_elements, depth);
        for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
            nk_size_t col_start = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (col_start >= chunk_end) continue;
            nk_f32_t *result_row = result + row_index * result_stride_elements;
            svfloat32_t query_norm_sq_f32x = svdup_n_f32(result_row[row_index]);
            for (nk_size_t col_index = col_start; col_index < chunk_end; col_index += svcntw()) {
                svbool_t predicate_f32x = svwhilelt_b32_u64(col_index, chunk_end);
                svfloat32_t dots_f32x = svld1_f32(predicate_f32x, result_row + col_index);
                svfloat32_t target_norms_sq_f32x = svld1_f32(predicate_f32x, norms_cache + (col_index - chunk_start));
                svfloat32_t norms_product_f32x = svmul_f32_x(predicate_f32x, query_norm_sq_f32x, target_norms_sq_f32x);
                svfloat32_t rsqrt_f32x = svrsqrte_f32(norms_product_f32x);
                rsqrt_f32x = svmul_f32_x(
                    predicate_f32x, rsqrt_f32x,
                    svrsqrts_f32(svmul_f32_x(predicate_f32x, norms_product_f32x, rsqrt_f32x), rsqrt_f32x));
                rsqrt_f32x = svmul_f32_x(
                    predicate_f32x, rsqrt_f32x,
                    svrsqrts_f32(svmul_f32_x(predicate_f32x, norms_product_f32x, rsqrt_f32x), rsqrt_f32x));
                svfloat32_t angular_f32x = svsub_f32_x(predicate_f32x, svdup_n_f32(1.0f),
                                                       svmul_f32_x(predicate_f32x, dots_f32x, rsqrt_f32x));
                angular_f32x = svmax_f32_x(predicate_f32x, angular_f32x, svdup_n_f32(0.0f));
                svst1_f32(predicate_f32x, result_row + col_index, angular_f32x);
            }
        }
    }
    // Phase 3: zero diagonals
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NK_PUBLIC void nk_angulars_symmetric_e2m3_sme(                                        //
    nk_e2m3_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride, //
    nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count) {
    nk_size_t const stride_elements = stride / sizeof(nk_e2m3_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_dots_symmetric_e2m3_sme_streaming_(vectors, n_vectors, depth, stride_elements, result, result_stride_elements,
                                          row_start, row_count);
    nk_angulars_symmetric_e2m3_sme_finalize_streaming_(vectors, n_vectors, depth, stride_elements, result,
                                                       result_stride_elements, row_start, row_count);
}

__arm_locally_streaming static void nk_euclideans_symmetric_e2m3_sme_finalize_streaming_(      //
    nk_e2m3_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride_elements, //
    nk_f32_t *result, nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) {
    // Phase 1: cache row norms on diagonal
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_f32_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_e2m3_(vectors + row_index * stride_elements, depth);
    }
    // Phase 2: column-first post-processing
    nk_f32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < n_vectors; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < n_vectors ? chunk_start + 256 : n_vectors;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_e2m3_(vectors + col * stride_elements, depth);
        for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
            nk_size_t col_start = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (col_start >= chunk_end) continue;
            nk_f32_t *result_row = result + row_index * result_stride_elements;
            svfloat32_t query_norm_sq_f32x = svdup_n_f32(result_row[row_index]);
            for (nk_size_t col_index = col_start; col_index < chunk_end; col_index += svcntw()) {
                svbool_t predicate_f32x = svwhilelt_b32_u64(col_index, chunk_end);
                svfloat32_t dots_f32x = svld1_f32(predicate_f32x, result_row + col_index);
                svfloat32_t target_norms_sq_f32x = svld1_f32(predicate_f32x, norms_cache + (col_index - chunk_start));
                svfloat32_t sum_sq_f32x = svadd_f32_x(predicate_f32x, query_norm_sq_f32x, target_norms_sq_f32x);
                svfloat32_t dist_sq_f32x = svsub_f32_x(predicate_f32x, sum_sq_f32x,
                                                       svmul_f32_x(predicate_f32x, svdup_n_f32(2.0f), dots_f32x));
                dist_sq_f32x = svmax_f32_x(predicate_f32x, dist_sq_f32x, svdup_n_f32(0.0f));
                svst1_f32(predicate_f32x, result_row + col_index, svsqrt_f32_x(predicate_f32x, dist_sq_f32x));
            }
        }
    }
    // Phase 3: zero diagonals
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NK_PUBLIC void nk_euclideans_symmetric_e2m3_sme(                                      //
    nk_e2m3_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride, //
    nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count) {
    nk_size_t const stride_elements = stride / sizeof(nk_e2m3_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_dots_symmetric_e2m3_sme_streaming_(vectors, n_vectors, depth, stride_elements, result, result_stride_elements,
                                          row_start, row_count);
    nk_euclideans_symmetric_e2m3_sme_finalize_streaming_(vectors, n_vectors, depth, stride_elements, result,
                                                         result_stride_elements, row_start, row_count);
}

#pragma endregion // Micro Precision E2M3

#pragma region Micro Precision E3M2

__arm_locally_streaming static void nk_angulars_packed_e3m2_sme_finalize_streaming_( //
    nk_e3m2_t const *a, void const *b_packed, nk_f32_t *c,                           //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,                              //
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) {
    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_f32_t const *b_norms = (nk_f32_t const *)((char const *)b_packed + header->norms_offset);
    for (nk_size_t row_index = 0; row_index < rows; row_index++) {
        nk_e3m2_t const *a_row = a + row_index * a_stride_elements;
        nk_f32_t *result_row = c + row_index * c_stride_elements;
        nk_f32_t query_norm_sq_f32 = nk_dots_reduce_sumsq_e3m2_(a_row, depth);
        svfloat32_t query_norm_sq_f32x = svdup_n_f32(query_norm_sq_f32);
        for (nk_size_t col_index = 0; col_index < columns; col_index += svcntw()) {
            svbool_t predicate_f32x = svwhilelt_b32_u64(col_index, columns);
            svfloat32_t dots_f32x = svld1_f32(predicate_f32x, result_row + col_index);
            svfloat32_t target_norms_sq_f32x = svld1_f32(predicate_f32x, b_norms + col_index);
            svfloat32_t norms_product_f32x = svmul_f32_x(predicate_f32x, query_norm_sq_f32x, target_norms_sq_f32x);
            svfloat32_t rsqrt_f32x = svrsqrte_f32(norms_product_f32x);
            rsqrt_f32x = svmul_f32_x(
                predicate_f32x, rsqrt_f32x,
                svrsqrts_f32(svmul_f32_x(predicate_f32x, norms_product_f32x, rsqrt_f32x), rsqrt_f32x));
            rsqrt_f32x = svmul_f32_x(
                predicate_f32x, rsqrt_f32x,
                svrsqrts_f32(svmul_f32_x(predicate_f32x, norms_product_f32x, rsqrt_f32x), rsqrt_f32x));
            svfloat32_t angular_f32x = svsub_f32_x(predicate_f32x, svdup_n_f32(1.0f),
                                                   svmul_f32_x(predicate_f32x, dots_f32x, rsqrt_f32x));
            angular_f32x = svmax_f32_x(predicate_f32x, angular_f32x, svdup_n_f32(0.0f));
            svst1_f32(predicate_f32x, result_row + col_index, angular_f32x);
        }
    }
}

NK_PUBLIC void nk_angulars_packed_e3m2_sme(                //
    nk_e3m2_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,    //
    nk_size_t a_stride_in_bytes, nk_size_t c_stride_in_bytes) {
    nk_size_t const a_stride_elements = a_stride_in_bytes / sizeof(nk_e3m2_t);
    nk_size_t const c_stride_elements = c_stride_in_bytes / sizeof(nk_f32_t);
    nk_dots_packed_e3m2_sme_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements, c_stride_elements);
    nk_angulars_packed_e3m2_sme_finalize_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                    c_stride_elements);
}

__arm_locally_streaming static void nk_euclideans_packed_e3m2_sme_finalize_streaming_( //
    nk_e3m2_t const *a, void const *b_packed, nk_f32_t *c,                             //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,                                //
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) {
    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_f32_t const *b_norms = (nk_f32_t const *)((char const *)b_packed + header->norms_offset);
    for (nk_size_t row_index = 0; row_index < rows; row_index++) {
        nk_e3m2_t const *a_row = a + row_index * a_stride_elements;
        nk_f32_t *result_row = c + row_index * c_stride_elements;
        nk_f32_t query_norm_sq_f32 = nk_dots_reduce_sumsq_e3m2_(a_row, depth);
        svfloat32_t query_norm_sq_f32x = svdup_n_f32(query_norm_sq_f32);
        for (nk_size_t col_index = 0; col_index < columns; col_index += svcntw()) {
            svbool_t predicate_f32x = svwhilelt_b32_u64(col_index, columns);
            svfloat32_t dots_f32x = svld1_f32(predicate_f32x, result_row + col_index);
            svfloat32_t target_norms_sq_f32x = svld1_f32(predicate_f32x, b_norms + col_index);
            svfloat32_t sum_sq_f32x = svadd_f32_x(predicate_f32x, query_norm_sq_f32x, target_norms_sq_f32x);
            svfloat32_t dist_sq_f32x = svsub_f32_x(predicate_f32x, sum_sq_f32x,
                                                   svmul_f32_x(predicate_f32x, svdup_n_f32(2.0f), dots_f32x));
            dist_sq_f32x = svmax_f32_x(predicate_f32x, dist_sq_f32x, svdup_n_f32(0.0f));
            svst1_f32(predicate_f32x, result_row + col_index, svsqrt_f32_x(predicate_f32x, dist_sq_f32x));
        }
    }
}

NK_PUBLIC void nk_euclideans_packed_e3m2_sme(              //
    nk_e3m2_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,    //
    nk_size_t a_stride_in_bytes, nk_size_t c_stride_in_bytes) {
    nk_size_t const a_stride_elements = a_stride_in_bytes / sizeof(nk_e3m2_t);
    nk_size_t const c_stride_elements = c_stride_in_bytes / sizeof(nk_f32_t);
    nk_dots_packed_e3m2_sme_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements, c_stride_elements);
    nk_euclideans_packed_e3m2_sme_finalize_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                      c_stride_elements);
}

__arm_locally_streaming static void nk_angulars_symmetric_e3m2_sme_finalize_streaming_(        //
    nk_e3m2_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride_elements, //
    nk_f32_t *result, nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) {
    // Phase 1: cache row norms on diagonal
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_f32_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_e3m2_(vectors + row_index * stride_elements, depth);
    }
    // Phase 2: column-first post-processing
    nk_f32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < n_vectors; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < n_vectors ? chunk_start + 256 : n_vectors;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_e3m2_(vectors + col * stride_elements, depth);
        for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
            nk_size_t col_start = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (col_start >= chunk_end) continue;
            nk_f32_t *result_row = result + row_index * result_stride_elements;
            svfloat32_t query_norm_sq_f32x = svdup_n_f32(result_row[row_index]);
            for (nk_size_t col_index = col_start; col_index < chunk_end; col_index += svcntw()) {
                svbool_t predicate_f32x = svwhilelt_b32_u64(col_index, chunk_end);
                svfloat32_t dots_f32x = svld1_f32(predicate_f32x, result_row + col_index);
                svfloat32_t target_norms_sq_f32x = svld1_f32(predicate_f32x, norms_cache + (col_index - chunk_start));
                svfloat32_t norms_product_f32x = svmul_f32_x(predicate_f32x, query_norm_sq_f32x, target_norms_sq_f32x);
                svfloat32_t rsqrt_f32x = svrsqrte_f32(norms_product_f32x);
                rsqrt_f32x = svmul_f32_x(
                    predicate_f32x, rsqrt_f32x,
                    svrsqrts_f32(svmul_f32_x(predicate_f32x, norms_product_f32x, rsqrt_f32x), rsqrt_f32x));
                rsqrt_f32x = svmul_f32_x(
                    predicate_f32x, rsqrt_f32x,
                    svrsqrts_f32(svmul_f32_x(predicate_f32x, norms_product_f32x, rsqrt_f32x), rsqrt_f32x));
                svfloat32_t angular_f32x = svsub_f32_x(predicate_f32x, svdup_n_f32(1.0f),
                                                       svmul_f32_x(predicate_f32x, dots_f32x, rsqrt_f32x));
                angular_f32x = svmax_f32_x(predicate_f32x, angular_f32x, svdup_n_f32(0.0f));
                svst1_f32(predicate_f32x, result_row + col_index, angular_f32x);
            }
        }
    }
    // Phase 3: zero diagonals
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NK_PUBLIC void nk_angulars_symmetric_e3m2_sme(                                        //
    nk_e3m2_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride, //
    nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count) {
    nk_size_t const stride_elements = stride / sizeof(nk_e3m2_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_dots_symmetric_e3m2_sme_streaming_(vectors, n_vectors, depth, stride_elements, result, result_stride_elements,
                                          row_start, row_count);
    nk_angulars_symmetric_e3m2_sme_finalize_streaming_(vectors, n_vectors, depth, stride_elements, result,
                                                       result_stride_elements, row_start, row_count);
}

__arm_locally_streaming static void nk_euclideans_symmetric_e3m2_sme_finalize_streaming_(      //
    nk_e3m2_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride_elements, //
    nk_f32_t *result, nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) {
    // Phase 1: cache row norms on diagonal
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_f32_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_e3m2_(vectors + row_index * stride_elements, depth);
    }
    // Phase 2: column-first post-processing
    nk_f32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < n_vectors; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < n_vectors ? chunk_start + 256 : n_vectors;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_e3m2_(vectors + col * stride_elements, depth);
        for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
            nk_size_t col_start = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (col_start >= chunk_end) continue;
            nk_f32_t *result_row = result + row_index * result_stride_elements;
            svfloat32_t query_norm_sq_f32x = svdup_n_f32(result_row[row_index]);
            for (nk_size_t col_index = col_start; col_index < chunk_end; col_index += svcntw()) {
                svbool_t predicate_f32x = svwhilelt_b32_u64(col_index, chunk_end);
                svfloat32_t dots_f32x = svld1_f32(predicate_f32x, result_row + col_index);
                svfloat32_t target_norms_sq_f32x = svld1_f32(predicate_f32x, norms_cache + (col_index - chunk_start));
                svfloat32_t sum_sq_f32x = svadd_f32_x(predicate_f32x, query_norm_sq_f32x, target_norms_sq_f32x);
                svfloat32_t dist_sq_f32x = svsub_f32_x(predicate_f32x, sum_sq_f32x,
                                                       svmul_f32_x(predicate_f32x, svdup_n_f32(2.0f), dots_f32x));
                dist_sq_f32x = svmax_f32_x(predicate_f32x, dist_sq_f32x, svdup_n_f32(0.0f));
                svst1_f32(predicate_f32x, result_row + col_index, svsqrt_f32_x(predicate_f32x, dist_sq_f32x));
            }
        }
    }
    // Phase 3: zero diagonals
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NK_PUBLIC void nk_euclideans_symmetric_e3m2_sme(                                      //
    nk_e3m2_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride, //
    nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count) {
    nk_size_t const stride_elements = stride / sizeof(nk_e3m2_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_dots_symmetric_e3m2_sme_streaming_(vectors, n_vectors, depth, stride_elements, result, result_stride_elements,
                                          row_start, row_count);
    nk_euclideans_symmetric_e3m2_sme_finalize_streaming_(vectors, n_vectors, depth, stride_elements, result,
                                                         result_stride_elements, row_start, row_count);
}

#pragma endregion // Micro Precision E3M2
#pragma region Signed 8-bit Integers

__arm_locally_streaming static void nk_angulars_packed_i8_sme_finalize_streaming_( //
    nk_i8_t const *a, void const *b_packed, nk_f32_t *c,                           //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,                            //
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) {
    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_u32_t const *b_norms = (nk_u32_t const *)((char const *)b_packed + header->norms_offset);
    for (nk_size_t row_index = 0; row_index < rows; row_index++) {
        nk_i8_t const *a_row = a + row_index * a_stride_elements;
        nk_f32_t *result_row = c + row_index * c_stride_elements;
        nk_u32_t query_norm_sq_u32 = nk_dots_reduce_sumsq_i8_(a_row, depth);
        svfloat32_t query_norm_sq_f32x = svdup_n_f32((nk_f32_t)query_norm_sq_u32);
        for (nk_size_t col_index = 0; col_index < columns; col_index += svcntw()) {
            svbool_t predicate_f32x = svwhilelt_b32_u64(col_index, columns);
            svfloat32_t dots_f32x = svcvt_f32_s32_x(
                predicate_f32x, svld1_s32(predicate_f32x, (nk_i32_t const *)(result_row + col_index)));
            svfloat32_t target_norms_sq_f32x = svcvt_f32_u32_x(predicate_f32x,
                                                               svld1_u32(predicate_f32x, b_norms + col_index));
            svfloat32_t norms_product_f32x = svmul_f32_x(predicate_f32x, query_norm_sq_f32x, target_norms_sq_f32x);
            svfloat32_t rsqrt_f32x = svrsqrte_f32(norms_product_f32x);
            rsqrt_f32x = svmul_f32_x(
                predicate_f32x, rsqrt_f32x,
                svrsqrts_f32(svmul_f32_x(predicate_f32x, norms_product_f32x, rsqrt_f32x), rsqrt_f32x));
            rsqrt_f32x = svmul_f32_x(
                predicate_f32x, rsqrt_f32x,
                svrsqrts_f32(svmul_f32_x(predicate_f32x, norms_product_f32x, rsqrt_f32x), rsqrt_f32x));
            svfloat32_t angular_f32x = svsub_f32_x(predicate_f32x, svdup_n_f32(1.0f),
                                                   svmul_f32_x(predicate_f32x, dots_f32x, rsqrt_f32x));
            angular_f32x = svmax_f32_x(predicate_f32x, angular_f32x, svdup_n_f32(0.0f));
            svst1_f32(predicate_f32x, result_row + col_index, angular_f32x);
        }
    }
}

NK_PUBLIC void nk_angulars_packed_i8_sme(                //
    nk_i8_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,  //
    nk_size_t a_stride_in_bytes, nk_size_t c_stride_in_bytes) {
    nk_size_t const a_stride_elements = a_stride_in_bytes / sizeof(nk_i8_t);
    nk_size_t const c_stride_elements = c_stride_in_bytes / sizeof(nk_f32_t);
    nk_dots_packed_i8_sme_streaming_(a, b_packed, (nk_i32_t *)c, rows, columns, depth, a_stride_elements,
                                     c_stride_elements);
    nk_angulars_packed_i8_sme_finalize_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                  c_stride_elements);
}

__arm_locally_streaming static void nk_euclideans_packed_i8_sme_finalize_streaming_( //
    nk_i8_t const *a, void const *b_packed, nk_f32_t *c,                             //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,                              //
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) {
    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_u32_t const *b_norms = (nk_u32_t const *)((char const *)b_packed + header->norms_offset);
    for (nk_size_t row_index = 0; row_index < rows; row_index++) {
        nk_i8_t const *a_row = a + row_index * a_stride_elements;
        nk_f32_t *result_row = c + row_index * c_stride_elements;
        nk_u32_t query_norm_sq_u32 = nk_dots_reduce_sumsq_i8_(a_row, depth);
        svfloat32_t query_norm_sq_f32x = svdup_n_f32((nk_f32_t)query_norm_sq_u32);
        for (nk_size_t col_index = 0; col_index < columns; col_index += svcntw()) {
            svbool_t predicate_f32x = svwhilelt_b32_u64(col_index, columns);
            svfloat32_t dots_f32x = svcvt_f32_s32_x(
                predicate_f32x, svld1_s32(predicate_f32x, (nk_i32_t const *)(result_row + col_index)));
            svfloat32_t target_norms_sq_f32x = svcvt_f32_u32_x(predicate_f32x,
                                                               svld1_u32(predicate_f32x, b_norms + col_index));
            svfloat32_t sum_sq_f32x = svadd_f32_x(predicate_f32x, query_norm_sq_f32x, target_norms_sq_f32x);
            svfloat32_t dist_sq_f32x = svsub_f32_x(predicate_f32x, sum_sq_f32x,
                                                   svmul_f32_x(predicate_f32x, svdup_n_f32(2.0f), dots_f32x));
            dist_sq_f32x = svmax_f32_x(predicate_f32x, dist_sq_f32x, svdup_n_f32(0.0f));
            svst1_f32(predicate_f32x, result_row + col_index, svsqrt_f32_x(predicate_f32x, dist_sq_f32x));
        }
    }
}

NK_PUBLIC void nk_euclideans_packed_i8_sme(              //
    nk_i8_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,  //
    nk_size_t a_stride_in_bytes, nk_size_t c_stride_in_bytes) {
    nk_size_t const a_stride_elements = a_stride_in_bytes / sizeof(nk_i8_t);
    nk_size_t const c_stride_elements = c_stride_in_bytes / sizeof(nk_f32_t);
    nk_dots_packed_i8_sme_streaming_(a, b_packed, (nk_i32_t *)c, rows, columns, depth, a_stride_elements,
                                     c_stride_elements);
    nk_euclideans_packed_i8_sme_finalize_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                    c_stride_elements);
}

__arm_locally_streaming static void nk_angulars_symmetric_i8_sme_finalize_streaming_(        //
    nk_i8_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride_elements, //
    nk_f32_t *result, nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) {
    // Phase 1: cache row norms on diagonal (store as u32 in f32 slot)
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_u32_t row_sumsq_u32 = nk_dots_reduce_sumsq_i8_(vectors + row_index * stride_elements, depth);
        ((nk_u32_t *)(result + row_index * result_stride_elements))[row_index] = row_sumsq_u32;
    }
    // Phase 2: column-first post-processing
    nk_u32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < n_vectors; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < n_vectors ? chunk_start + 256 : n_vectors;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_i8_(vectors + col * stride_elements, depth);
        for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
            nk_size_t col_start = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (col_start >= chunk_end) continue;
            nk_f32_t *result_row = result + row_index * result_stride_elements;
            nk_u32_t query_sumsq_u32 = ((nk_u32_t *)result_row)[row_index];
            svfloat32_t query_norm_sq_f32x = svdup_n_f32((nk_f32_t)query_sumsq_u32);
            for (nk_size_t col_index = col_start; col_index < chunk_end; col_index += svcntw()) {
                svbool_t predicate_f32x = svwhilelt_b32_u64(col_index, chunk_end);
                svfloat32_t dots_f32x = svcvt_f32_s32_x(
                    predicate_f32x, svld1_s32(predicate_f32x, (nk_i32_t *)(result_row + col_index)));
                svfloat32_t target_norms_sq_f32x = svcvt_f32_u32_x(
                    predicate_f32x, svld1_u32(predicate_f32x, norms_cache + (col_index - chunk_start)));
                svfloat32_t norms_product_f32x = svmul_f32_x(predicate_f32x, query_norm_sq_f32x, target_norms_sq_f32x);
                svfloat32_t rsqrt_f32x = svrsqrte_f32(norms_product_f32x);
                rsqrt_f32x = svmul_f32_x(
                    predicate_f32x, rsqrt_f32x,
                    svrsqrts_f32(svmul_f32_x(predicate_f32x, norms_product_f32x, rsqrt_f32x), rsqrt_f32x));
                rsqrt_f32x = svmul_f32_x(
                    predicate_f32x, rsqrt_f32x,
                    svrsqrts_f32(svmul_f32_x(predicate_f32x, norms_product_f32x, rsqrt_f32x), rsqrt_f32x));
                svfloat32_t angular_f32x = svsub_f32_x(predicate_f32x, svdup_n_f32(1.0f),
                                                       svmul_f32_x(predicate_f32x, dots_f32x, rsqrt_f32x));
                angular_f32x = svmax_f32_x(predicate_f32x, angular_f32x, svdup_n_f32(0.0f));
                svst1_f32(predicate_f32x, result_row + col_index, angular_f32x);
            }
        }
    }
    // Phase 3: zero diagonals
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NK_PUBLIC void nk_angulars_symmetric_i8_sme(                                        //
    nk_i8_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride, //
    nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count) {
    nk_size_t const stride_elements = stride / sizeof(nk_i8_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_dots_symmetric_i8_sme_streaming_(vectors, n_vectors, depth, stride_elements, (nk_i32_t *)result,
                                        result_stride_elements, row_start, row_count);
    nk_angulars_symmetric_i8_sme_finalize_streaming_(vectors, n_vectors, depth, stride_elements, result,
                                                     result_stride_elements, row_start, row_count);
}

__arm_locally_streaming static void nk_euclideans_symmetric_i8_sme_finalize_streaming_(      //
    nk_i8_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride_elements, //
    nk_f32_t *result, nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) {
    // Phase 1: cache row norms on diagonal (store as u32 in f32 slot)
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_u32_t row_sumsq_u32 = nk_dots_reduce_sumsq_i8_(vectors + row_index * stride_elements, depth);
        ((nk_u32_t *)(result + row_index * result_stride_elements))[row_index] = row_sumsq_u32;
    }
    // Phase 2: column-first post-processing
    nk_u32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < n_vectors; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < n_vectors ? chunk_start + 256 : n_vectors;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_i8_(vectors + col * stride_elements, depth);
        for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
            nk_size_t col_start = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (col_start >= chunk_end) continue;
            nk_f32_t *result_row = result + row_index * result_stride_elements;
            nk_u32_t query_sumsq_u32 = ((nk_u32_t *)result_row)[row_index];
            svfloat32_t query_norm_sq_f32x = svdup_n_f32((nk_f32_t)query_sumsq_u32);
            for (nk_size_t col_index = col_start; col_index < chunk_end; col_index += svcntw()) {
                svbool_t predicate_f32x = svwhilelt_b32_u64(col_index, chunk_end);
                svfloat32_t dots_f32x = svcvt_f32_s32_x(
                    predicate_f32x, svld1_s32(predicate_f32x, (nk_i32_t *)(result_row + col_index)));
                svfloat32_t target_norms_sq_f32x = svcvt_f32_u32_x(
                    predicate_f32x, svld1_u32(predicate_f32x, norms_cache + (col_index - chunk_start)));
                svfloat32_t sum_sq_f32x = svadd_f32_x(predicate_f32x, query_norm_sq_f32x, target_norms_sq_f32x);
                svfloat32_t dist_sq_f32x = svsub_f32_x(predicate_f32x, sum_sq_f32x,
                                                       svmul_f32_x(predicate_f32x, svdup_n_f32(2.0f), dots_f32x));
                dist_sq_f32x = svmax_f32_x(predicate_f32x, dist_sq_f32x, svdup_n_f32(0.0f));
                svst1_f32(predicate_f32x, result_row + col_index, svsqrt_f32_x(predicate_f32x, dist_sq_f32x));
            }
        }
    }
    // Phase 3: zero diagonals
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NK_PUBLIC void nk_euclideans_symmetric_i8_sme(                                      //
    nk_i8_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride, //
    nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count) {
    nk_size_t const stride_elements = stride / sizeof(nk_i8_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_dots_symmetric_i8_sme_streaming_(vectors, n_vectors, depth, stride_elements, (nk_i32_t *)result,
                                        result_stride_elements, row_start, row_count);
    nk_euclideans_symmetric_i8_sme_finalize_streaming_(vectors, n_vectors, depth, stride_elements, result,
                                                       result_stride_elements, row_start, row_count);
}

#pragma endregion // Signed 8-bit Integers

#pragma region Unsigned 8-bit Integers

__arm_locally_streaming static void nk_angulars_packed_u8_sme_finalize_streaming_( //
    nk_u8_t const *a, void const *b_packed, nk_f32_t *c,                           //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,                            //
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) {
    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_u32_t const *b_norms = (nk_u32_t const *)((char const *)b_packed + header->norms_offset);
    for (nk_size_t row_index = 0; row_index < rows; row_index++) {
        nk_u8_t const *a_row = a + row_index * a_stride_elements;
        nk_f32_t *result_row = c + row_index * c_stride_elements;
        nk_u32_t query_norm_sq_u32 = nk_dots_reduce_sumsq_u8_(a_row, depth);
        svfloat32_t query_norm_sq_f32x = svdup_n_f32((nk_f32_t)query_norm_sq_u32);
        for (nk_size_t col_index = 0; col_index < columns; col_index += svcntw()) {
            svbool_t predicate_f32x = svwhilelt_b32_u64(col_index, columns);
            svfloat32_t dots_f32x = svcvt_f32_u32_x(
                predicate_f32x, svld1_u32(predicate_f32x, (nk_u32_t const *)(result_row + col_index)));
            svfloat32_t target_norms_sq_f32x = svcvt_f32_u32_x(predicate_f32x,
                                                               svld1_u32(predicate_f32x, b_norms + col_index));
            svfloat32_t norms_product_f32x = svmul_f32_x(predicate_f32x, query_norm_sq_f32x, target_norms_sq_f32x);
            svfloat32_t rsqrt_f32x = svrsqrte_f32(norms_product_f32x);
            rsqrt_f32x = svmul_f32_x(
                predicate_f32x, rsqrt_f32x,
                svrsqrts_f32(svmul_f32_x(predicate_f32x, norms_product_f32x, rsqrt_f32x), rsqrt_f32x));
            rsqrt_f32x = svmul_f32_x(
                predicate_f32x, rsqrt_f32x,
                svrsqrts_f32(svmul_f32_x(predicate_f32x, norms_product_f32x, rsqrt_f32x), rsqrt_f32x));
            svfloat32_t angular_f32x = svsub_f32_x(predicate_f32x, svdup_n_f32(1.0f),
                                                   svmul_f32_x(predicate_f32x, dots_f32x, rsqrt_f32x));
            angular_f32x = svmax_f32_x(predicate_f32x, angular_f32x, svdup_n_f32(0.0f));
            svst1_f32(predicate_f32x, result_row + col_index, angular_f32x);
        }
    }
}

NK_PUBLIC void nk_angulars_packed_u8_sme(                //
    nk_u8_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,  //
    nk_size_t a_stride_in_bytes, nk_size_t c_stride_in_bytes) {
    nk_size_t const a_stride_elements = a_stride_in_bytes / sizeof(nk_u8_t);
    nk_size_t const c_stride_elements = c_stride_in_bytes / sizeof(nk_f32_t);
    nk_dots_packed_u8_sme_streaming_(a, b_packed, (nk_u32_t *)c, rows, columns, depth, a_stride_elements,
                                     c_stride_elements);
    nk_angulars_packed_u8_sme_finalize_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                  c_stride_elements);
}

__arm_locally_streaming static void nk_euclideans_packed_u8_sme_finalize_streaming_( //
    nk_u8_t const *a, void const *b_packed, nk_f32_t *c,                             //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,                              //
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) {
    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_u32_t const *b_norms = (nk_u32_t const *)((char const *)b_packed + header->norms_offset);
    for (nk_size_t row_index = 0; row_index < rows; row_index++) {
        nk_u8_t const *a_row = a + row_index * a_stride_elements;
        nk_f32_t *result_row = c + row_index * c_stride_elements;
        nk_u32_t query_norm_sq_u32 = nk_dots_reduce_sumsq_u8_(a_row, depth);
        svfloat32_t query_norm_sq_f32x = svdup_n_f32((nk_f32_t)query_norm_sq_u32);
        for (nk_size_t col_index = 0; col_index < columns; col_index += svcntw()) {
            svbool_t predicate_f32x = svwhilelt_b32_u64(col_index, columns);
            svfloat32_t dots_f32x = svcvt_f32_u32_x(
                predicate_f32x, svld1_u32(predicate_f32x, (nk_u32_t const *)(result_row + col_index)));
            svfloat32_t target_norms_sq_f32x = svcvt_f32_u32_x(predicate_f32x,
                                                               svld1_u32(predicate_f32x, b_norms + col_index));
            svfloat32_t sum_sq_f32x = svadd_f32_x(predicate_f32x, query_norm_sq_f32x, target_norms_sq_f32x);
            svfloat32_t dist_sq_f32x = svsub_f32_x(predicate_f32x, sum_sq_f32x,
                                                   svmul_f32_x(predicate_f32x, svdup_n_f32(2.0f), dots_f32x));
            dist_sq_f32x = svmax_f32_x(predicate_f32x, dist_sq_f32x, svdup_n_f32(0.0f));
            svst1_f32(predicate_f32x, result_row + col_index, svsqrt_f32_x(predicate_f32x, dist_sq_f32x));
        }
    }
}

NK_PUBLIC void nk_euclideans_packed_u8_sme(              //
    nk_u8_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,  //
    nk_size_t a_stride_in_bytes, nk_size_t c_stride_in_bytes) {
    nk_size_t const a_stride_elements = a_stride_in_bytes / sizeof(nk_u8_t);
    nk_size_t const c_stride_elements = c_stride_in_bytes / sizeof(nk_f32_t);
    nk_dots_packed_u8_sme_streaming_(a, b_packed, (nk_u32_t *)c, rows, columns, depth, a_stride_elements,
                                     c_stride_elements);
    nk_euclideans_packed_u8_sme_finalize_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                    c_stride_elements);
}

__arm_locally_streaming static void nk_angulars_symmetric_u8_sme_finalize_streaming_(        //
    nk_u8_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride_elements, //
    nk_f32_t *result, nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) {
    // Phase 1: cache row norms on diagonal (store as u32 in f32 slot)
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_u32_t row_sumsq_u32 = nk_dots_reduce_sumsq_u8_(vectors + row_index * stride_elements, depth);
        ((nk_u32_t *)(result + row_index * result_stride_elements))[row_index] = row_sumsq_u32;
    }
    // Phase 2: column-first post-processing
    nk_u32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < n_vectors; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < n_vectors ? chunk_start + 256 : n_vectors;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_u8_(vectors + col * stride_elements, depth);
        for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
            nk_size_t col_start = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (col_start >= chunk_end) continue;
            nk_f32_t *result_row = result + row_index * result_stride_elements;
            nk_u32_t query_sumsq_u32 = ((nk_u32_t *)result_row)[row_index];
            svfloat32_t query_norm_sq_f32x = svdup_n_f32((nk_f32_t)query_sumsq_u32);
            for (nk_size_t col_index = col_start; col_index < chunk_end; col_index += svcntw()) {
                svbool_t predicate_f32x = svwhilelt_b32_u64(col_index, chunk_end);
                svfloat32_t dots_f32x = svcvt_f32_u32_x(
                    predicate_f32x, svld1_u32(predicate_f32x, (nk_u32_t *)(result_row + col_index)));
                svfloat32_t target_norms_sq_f32x = svcvt_f32_u32_x(
                    predicate_f32x, svld1_u32(predicate_f32x, norms_cache + (col_index - chunk_start)));
                svfloat32_t norms_product_f32x = svmul_f32_x(predicate_f32x, query_norm_sq_f32x, target_norms_sq_f32x);
                svfloat32_t rsqrt_f32x = svrsqrte_f32(norms_product_f32x);
                rsqrt_f32x = svmul_f32_x(
                    predicate_f32x, rsqrt_f32x,
                    svrsqrts_f32(svmul_f32_x(predicate_f32x, norms_product_f32x, rsqrt_f32x), rsqrt_f32x));
                rsqrt_f32x = svmul_f32_x(
                    predicate_f32x, rsqrt_f32x,
                    svrsqrts_f32(svmul_f32_x(predicate_f32x, norms_product_f32x, rsqrt_f32x), rsqrt_f32x));
                svfloat32_t angular_f32x = svsub_f32_x(predicate_f32x, svdup_n_f32(1.0f),
                                                       svmul_f32_x(predicate_f32x, dots_f32x, rsqrt_f32x));
                angular_f32x = svmax_f32_x(predicate_f32x, angular_f32x, svdup_n_f32(0.0f));
                svst1_f32(predicate_f32x, result_row + col_index, angular_f32x);
            }
        }
    }
    // Phase 3: zero diagonals
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NK_PUBLIC void nk_angulars_symmetric_u8_sme(                                        //
    nk_u8_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride, //
    nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count) {
    nk_size_t const stride_elements = stride / sizeof(nk_u8_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_dots_symmetric_u8_sme_streaming_(vectors, n_vectors, depth, stride_elements, (nk_u32_t *)result,
                                        result_stride_elements, row_start, row_count);
    nk_angulars_symmetric_u8_sme_finalize_streaming_(vectors, n_vectors, depth, stride_elements, result,
                                                     result_stride_elements, row_start, row_count);
}

__arm_locally_streaming static void nk_euclideans_symmetric_u8_sme_finalize_streaming_(      //
    nk_u8_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride_elements, //
    nk_f32_t *result, nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) {
    // Phase 1: cache row norms on diagonal (store as u32 in f32 slot)
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_u32_t row_sumsq_u32 = nk_dots_reduce_sumsq_u8_(vectors + row_index * stride_elements, depth);
        ((nk_u32_t *)(result + row_index * result_stride_elements))[row_index] = row_sumsq_u32;
    }
    // Phase 2: column-first post-processing
    nk_u32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < n_vectors; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < n_vectors ? chunk_start + 256 : n_vectors;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_u8_(vectors + col * stride_elements, depth);
        for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
            nk_size_t col_start = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (col_start >= chunk_end) continue;
            nk_f32_t *result_row = result + row_index * result_stride_elements;
            nk_u32_t query_sumsq_u32 = ((nk_u32_t *)result_row)[row_index];
            svfloat32_t query_norm_sq_f32x = svdup_n_f32((nk_f32_t)query_sumsq_u32);
            for (nk_size_t col_index = col_start; col_index < chunk_end; col_index += svcntw()) {
                svbool_t predicate_f32x = svwhilelt_b32_u64(col_index, chunk_end);
                svfloat32_t dots_f32x = svcvt_f32_u32_x(
                    predicate_f32x, svld1_u32(predicate_f32x, (nk_u32_t *)(result_row + col_index)));
                svfloat32_t target_norms_sq_f32x = svcvt_f32_u32_x(
                    predicate_f32x, svld1_u32(predicate_f32x, norms_cache + (col_index - chunk_start)));
                svfloat32_t sum_sq_f32x = svadd_f32_x(predicate_f32x, query_norm_sq_f32x, target_norms_sq_f32x);
                svfloat32_t dist_sq_f32x = svsub_f32_x(predicate_f32x, sum_sq_f32x,
                                                       svmul_f32_x(predicate_f32x, svdup_n_f32(2.0f), dots_f32x));
                dist_sq_f32x = svmax_f32_x(predicate_f32x, dist_sq_f32x, svdup_n_f32(0.0f));
                svst1_f32(predicate_f32x, result_row + col_index, svsqrt_f32_x(predicate_f32x, dist_sq_f32x));
            }
        }
    }
    // Phase 3: zero diagonals
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NK_PUBLIC void nk_euclideans_symmetric_u8_sme(                                      //
    nk_u8_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride, //
    nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count) {
    nk_size_t const stride_elements = stride / sizeof(nk_u8_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_dots_symmetric_u8_sme_streaming_(vectors, n_vectors, depth, stride_elements, (nk_u32_t *)result,
                                        result_stride_elements, row_start, row_count);
    nk_euclideans_symmetric_u8_sme_finalize_streaming_(vectors, n_vectors, depth, stride_elements, result,
                                                       result_stride_elements, row_start, row_count);
}

#pragma endregion // Unsigned 8-bit Integers

#pragma region Nibble Signed Integers

__arm_locally_streaming static void nk_angulars_packed_i4_sme_finalize_streaming_( //
    nk_i4x2_t const *a, void const *b_packed, nk_f32_t *c,                         //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,                            //
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) {
    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_u32_t const *b_norms = (nk_u32_t const *)((char const *)b_packed + header->norms_offset);
    for (nk_size_t row_index = 0; row_index < rows; row_index++) {
        nk_i4x2_t const *a_row = a + row_index * a_stride_elements;
        nk_f32_t *result_row = c + row_index * c_stride_elements;
        nk_u32_t query_norm_sq_u32 = nk_dots_reduce_sumsq_i4_(a_row, depth);
        svfloat32_t query_norm_sq_f32x = svdup_n_f32((nk_f32_t)query_norm_sq_u32);
        for (nk_size_t col_index = 0; col_index < columns; col_index += svcntw()) {
            svbool_t predicate_f32x = svwhilelt_b32_u64(col_index, columns);
            svfloat32_t dots_f32x = svcvt_f32_s32_x(
                predicate_f32x, svld1_s32(predicate_f32x, (nk_i32_t const *)(result_row + col_index)));
            svfloat32_t target_norms_sq_f32x = svcvt_f32_u32_x(predicate_f32x,
                                                               svld1_u32(predicate_f32x, b_norms + col_index));
            svfloat32_t norms_product_f32x = svmul_f32_x(predicate_f32x, query_norm_sq_f32x, target_norms_sq_f32x);
            svfloat32_t rsqrt_f32x = svrsqrte_f32(norms_product_f32x);
            rsqrt_f32x = svmul_f32_x(
                predicate_f32x, rsqrt_f32x,
                svrsqrts_f32(svmul_f32_x(predicate_f32x, norms_product_f32x, rsqrt_f32x), rsqrt_f32x));
            rsqrt_f32x = svmul_f32_x(
                predicate_f32x, rsqrt_f32x,
                svrsqrts_f32(svmul_f32_x(predicate_f32x, norms_product_f32x, rsqrt_f32x), rsqrt_f32x));
            svfloat32_t angular_f32x = svsub_f32_x(predicate_f32x, svdup_n_f32(1.0f),
                                                   svmul_f32_x(predicate_f32x, dots_f32x, rsqrt_f32x));
            angular_f32x = svmax_f32_x(predicate_f32x, angular_f32x, svdup_n_f32(0.0f));
            svst1_f32(predicate_f32x, result_row + col_index, angular_f32x);
        }
    }
}

NK_PUBLIC void nk_angulars_packed_i4_sme(                  //
    nk_i4x2_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,    //
    nk_size_t a_stride_in_bytes, nk_size_t c_stride_in_bytes) {
    nk_size_t const a_stride_elements = a_stride_in_bytes / sizeof(nk_i4x2_t);
    nk_size_t const c_stride_elements = c_stride_in_bytes / sizeof(nk_f32_t);
    nk_dots_packed_i4_sme_streaming_(a, b_packed, (nk_i32_t *)c, rows, columns, depth, a_stride_elements,
                                     c_stride_elements);
    nk_angulars_packed_i4_sme_finalize_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                  c_stride_elements);
}

__arm_locally_streaming static void nk_euclideans_packed_i4_sme_finalize_streaming_( //
    nk_i4x2_t const *a, void const *b_packed, nk_f32_t *c,                           //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,                              //
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) {
    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_u32_t const *b_norms = (nk_u32_t const *)((char const *)b_packed + header->norms_offset);
    for (nk_size_t row_index = 0; row_index < rows; row_index++) {
        nk_i4x2_t const *a_row = a + row_index * a_stride_elements;
        nk_f32_t *result_row = c + row_index * c_stride_elements;
        nk_u32_t query_norm_sq_u32 = nk_dots_reduce_sumsq_i4_(a_row, depth);
        svfloat32_t query_norm_sq_f32x = svdup_n_f32((nk_f32_t)query_norm_sq_u32);
        for (nk_size_t col_index = 0; col_index < columns; col_index += svcntw()) {
            svbool_t predicate_f32x = svwhilelt_b32_u64(col_index, columns);
            svfloat32_t dots_f32x = svcvt_f32_s32_x(
                predicate_f32x, svld1_s32(predicate_f32x, (nk_i32_t const *)(result_row + col_index)));
            svfloat32_t target_norms_sq_f32x = svcvt_f32_u32_x(predicate_f32x,
                                                               svld1_u32(predicate_f32x, b_norms + col_index));
            svfloat32_t sum_sq_f32x = svadd_f32_x(predicate_f32x, query_norm_sq_f32x, target_norms_sq_f32x);
            svfloat32_t dist_sq_f32x = svsub_f32_x(predicate_f32x, sum_sq_f32x,
                                                   svmul_f32_x(predicate_f32x, svdup_n_f32(2.0f), dots_f32x));
            dist_sq_f32x = svmax_f32_x(predicate_f32x, dist_sq_f32x, svdup_n_f32(0.0f));
            svst1_f32(predicate_f32x, result_row + col_index, svsqrt_f32_x(predicate_f32x, dist_sq_f32x));
        }
    }
}

NK_PUBLIC void nk_euclideans_packed_i4_sme(                //
    nk_i4x2_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,    //
    nk_size_t a_stride_in_bytes, nk_size_t c_stride_in_bytes) {
    nk_size_t const a_stride_elements = a_stride_in_bytes / sizeof(nk_i4x2_t);
    nk_size_t const c_stride_elements = c_stride_in_bytes / sizeof(nk_f32_t);
    nk_dots_packed_i4_sme_streaming_(a, b_packed, (nk_i32_t *)c, rows, columns, depth, a_stride_elements,
                                     c_stride_elements);
    nk_euclideans_packed_i4_sme_finalize_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                    c_stride_elements);
}

__arm_locally_streaming static void nk_angulars_symmetric_i4_sme_finalize_streaming_(          //
    nk_i4x2_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride_elements, //
    nk_f32_t *result, nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) {
    // Phase 1: cache row norms on diagonal (store as u32 in f32 slot)
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_u32_t row_sumsq_u32 = nk_dots_reduce_sumsq_i4_(vectors + row_index * stride_elements, depth);
        ((nk_u32_t *)(result + row_index * result_stride_elements))[row_index] = row_sumsq_u32;
    }
    // Phase 2: column-first post-processing
    nk_u32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < n_vectors; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < n_vectors ? chunk_start + 256 : n_vectors;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_i4_(vectors + col * stride_elements, depth);
        for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
            nk_size_t col_start = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (col_start >= chunk_end) continue;
            nk_f32_t *result_row = result + row_index * result_stride_elements;
            nk_u32_t query_sumsq_u32 = ((nk_u32_t *)result_row)[row_index];
            svfloat32_t query_norm_sq_f32x = svdup_n_f32((nk_f32_t)query_sumsq_u32);
            for (nk_size_t col_index = col_start; col_index < chunk_end; col_index += svcntw()) {
                svbool_t predicate_f32x = svwhilelt_b32_u64(col_index, chunk_end);
                svfloat32_t dots_f32x = svcvt_f32_s32_x(
                    predicate_f32x, svld1_s32(predicate_f32x, (nk_i32_t *)(result_row + col_index)));
                svfloat32_t target_norms_sq_f32x = svcvt_f32_u32_x(
                    predicate_f32x, svld1_u32(predicate_f32x, norms_cache + (col_index - chunk_start)));
                svfloat32_t norms_product_f32x = svmul_f32_x(predicate_f32x, query_norm_sq_f32x, target_norms_sq_f32x);
                svfloat32_t rsqrt_f32x = svrsqrte_f32(norms_product_f32x);
                rsqrt_f32x = svmul_f32_x(
                    predicate_f32x, rsqrt_f32x,
                    svrsqrts_f32(svmul_f32_x(predicate_f32x, norms_product_f32x, rsqrt_f32x), rsqrt_f32x));
                rsqrt_f32x = svmul_f32_x(
                    predicate_f32x, rsqrt_f32x,
                    svrsqrts_f32(svmul_f32_x(predicate_f32x, norms_product_f32x, rsqrt_f32x), rsqrt_f32x));
                svfloat32_t angular_f32x = svsub_f32_x(predicate_f32x, svdup_n_f32(1.0f),
                                                       svmul_f32_x(predicate_f32x, dots_f32x, rsqrt_f32x));
                angular_f32x = svmax_f32_x(predicate_f32x, angular_f32x, svdup_n_f32(0.0f));
                svst1_f32(predicate_f32x, result_row + col_index, angular_f32x);
            }
        }
    }
    // Phase 3: zero diagonals
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NK_PUBLIC void nk_angulars_symmetric_i4_sme(                                          //
    nk_i4x2_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride, //
    nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count) {
    nk_size_t const stride_elements = stride / sizeof(nk_i4x2_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_dots_symmetric_i4_sme_streaming_(vectors, n_vectors, depth, stride_elements, (nk_i32_t *)result,
                                        result_stride_elements, row_start, row_count);
    nk_angulars_symmetric_i4_sme_finalize_streaming_(vectors, n_vectors, depth, stride_elements, result,
                                                     result_stride_elements, row_start, row_count);
}

__arm_locally_streaming static void nk_euclideans_symmetric_i4_sme_finalize_streaming_(        //
    nk_i4x2_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride_elements, //
    nk_f32_t *result, nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) {
    // Phase 1: cache row norms on diagonal (store as u32 in f32 slot)
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_u32_t row_sumsq_u32 = nk_dots_reduce_sumsq_i4_(vectors + row_index * stride_elements, depth);
        ((nk_u32_t *)(result + row_index * result_stride_elements))[row_index] = row_sumsq_u32;
    }
    // Phase 2: column-first post-processing
    nk_u32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < n_vectors; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < n_vectors ? chunk_start + 256 : n_vectors;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_i4_(vectors + col * stride_elements, depth);
        for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
            nk_size_t col_start = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (col_start >= chunk_end) continue;
            nk_f32_t *result_row = result + row_index * result_stride_elements;
            nk_u32_t query_sumsq_u32 = ((nk_u32_t *)result_row)[row_index];
            svfloat32_t query_norm_sq_f32x = svdup_n_f32((nk_f32_t)query_sumsq_u32);
            for (nk_size_t col_index = col_start; col_index < chunk_end; col_index += svcntw()) {
                svbool_t predicate_f32x = svwhilelt_b32_u64(col_index, chunk_end);
                svfloat32_t dots_f32x = svcvt_f32_s32_x(
                    predicate_f32x, svld1_s32(predicate_f32x, (nk_i32_t *)(result_row + col_index)));
                svfloat32_t target_norms_sq_f32x = svcvt_f32_u32_x(
                    predicate_f32x, svld1_u32(predicate_f32x, norms_cache + (col_index - chunk_start)));
                svfloat32_t sum_sq_f32x = svadd_f32_x(predicate_f32x, query_norm_sq_f32x, target_norms_sq_f32x);
                svfloat32_t dist_sq_f32x = svsub_f32_x(predicate_f32x, sum_sq_f32x,
                                                       svmul_f32_x(predicate_f32x, svdup_n_f32(2.0f), dots_f32x));
                dist_sq_f32x = svmax_f32_x(predicate_f32x, dist_sq_f32x, svdup_n_f32(0.0f));
                svst1_f32(predicate_f32x, result_row + col_index, svsqrt_f32_x(predicate_f32x, dist_sq_f32x));
            }
        }
    }
    // Phase 3: zero diagonals
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NK_PUBLIC void nk_euclideans_symmetric_i4_sme(                                        //
    nk_i4x2_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride, //
    nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count) {
    nk_size_t const stride_elements = stride / sizeof(nk_i4x2_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_dots_symmetric_i4_sme_streaming_(vectors, n_vectors, depth, stride_elements, (nk_i32_t *)result,
                                        result_stride_elements, row_start, row_count);
    nk_euclideans_symmetric_i4_sme_finalize_streaming_(vectors, n_vectors, depth, stride_elements, result,
                                                       result_stride_elements, row_start, row_count);
}

#pragma endregion // Nibble Signed Integers

#pragma region Nibble Unsigned Integers

__arm_locally_streaming static void nk_angulars_packed_u4_sme_finalize_streaming_( //
    nk_u4x2_t const *a, void const *b_packed, nk_f32_t *c,                         //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,                            //
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) {
    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_u32_t const *b_norms = (nk_u32_t const *)((char const *)b_packed + header->norms_offset);
    for (nk_size_t row_index = 0; row_index < rows; row_index++) {
        nk_u4x2_t const *a_row = a + row_index * a_stride_elements;
        nk_f32_t *result_row = c + row_index * c_stride_elements;
        nk_u32_t query_norm_sq_u32 = nk_dots_reduce_sumsq_u4_(a_row, depth);
        svfloat32_t query_norm_sq_f32x = svdup_n_f32((nk_f32_t)query_norm_sq_u32);
        for (nk_size_t col_index = 0; col_index < columns; col_index += svcntw()) {
            svbool_t predicate_f32x = svwhilelt_b32_u64(col_index, columns);
            svfloat32_t dots_f32x = svcvt_f32_u32_x(
                predicate_f32x, svld1_u32(predicate_f32x, (nk_u32_t const *)(result_row + col_index)));
            svfloat32_t target_norms_sq_f32x = svcvt_f32_u32_x(predicate_f32x,
                                                               svld1_u32(predicate_f32x, b_norms + col_index));
            svfloat32_t norms_product_f32x = svmul_f32_x(predicate_f32x, query_norm_sq_f32x, target_norms_sq_f32x);
            svfloat32_t rsqrt_f32x = svrsqrte_f32(norms_product_f32x);
            rsqrt_f32x = svmul_f32_x(
                predicate_f32x, rsqrt_f32x,
                svrsqrts_f32(svmul_f32_x(predicate_f32x, norms_product_f32x, rsqrt_f32x), rsqrt_f32x));
            rsqrt_f32x = svmul_f32_x(
                predicate_f32x, rsqrt_f32x,
                svrsqrts_f32(svmul_f32_x(predicate_f32x, norms_product_f32x, rsqrt_f32x), rsqrt_f32x));
            svfloat32_t angular_f32x = svsub_f32_x(predicate_f32x, svdup_n_f32(1.0f),
                                                   svmul_f32_x(predicate_f32x, dots_f32x, rsqrt_f32x));
            angular_f32x = svmax_f32_x(predicate_f32x, angular_f32x, svdup_n_f32(0.0f));
            svst1_f32(predicate_f32x, result_row + col_index, angular_f32x);
        }
    }
}

NK_PUBLIC void nk_angulars_packed_u4_sme(                  //
    nk_u4x2_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,    //
    nk_size_t a_stride_in_bytes, nk_size_t c_stride_in_bytes) {
    nk_size_t const a_stride_elements = a_stride_in_bytes / sizeof(nk_u4x2_t);
    nk_size_t const c_stride_elements = c_stride_in_bytes / sizeof(nk_f32_t);
    nk_dots_packed_u4_sme_streaming_(a, b_packed, (nk_u32_t *)c, rows, columns, depth, a_stride_elements,
                                     c_stride_elements);
    nk_angulars_packed_u4_sme_finalize_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                  c_stride_elements);
}

__arm_locally_streaming static void nk_euclideans_packed_u4_sme_finalize_streaming_( //
    nk_u4x2_t const *a, void const *b_packed, nk_f32_t *c,                           //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,                              //
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) {
    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_u32_t const *b_norms = (nk_u32_t const *)((char const *)b_packed + header->norms_offset);
    for (nk_size_t row_index = 0; row_index < rows; row_index++) {
        nk_u4x2_t const *a_row = a + row_index * a_stride_elements;
        nk_f32_t *result_row = c + row_index * c_stride_elements;
        nk_u32_t query_norm_sq_u32 = nk_dots_reduce_sumsq_u4_(a_row, depth);
        svfloat32_t query_norm_sq_f32x = svdup_n_f32((nk_f32_t)query_norm_sq_u32);
        for (nk_size_t col_index = 0; col_index < columns; col_index += svcntw()) {
            svbool_t predicate_f32x = svwhilelt_b32_u64(col_index, columns);
            svfloat32_t dots_f32x = svcvt_f32_u32_x(
                predicate_f32x, svld1_u32(predicate_f32x, (nk_u32_t const *)(result_row + col_index)));
            svfloat32_t target_norms_sq_f32x = svcvt_f32_u32_x(predicate_f32x,
                                                               svld1_u32(predicate_f32x, b_norms + col_index));
            svfloat32_t sum_sq_f32x = svadd_f32_x(predicate_f32x, query_norm_sq_f32x, target_norms_sq_f32x);
            svfloat32_t dist_sq_f32x = svsub_f32_x(predicate_f32x, sum_sq_f32x,
                                                   svmul_f32_x(predicate_f32x, svdup_n_f32(2.0f), dots_f32x));
            dist_sq_f32x = svmax_f32_x(predicate_f32x, dist_sq_f32x, svdup_n_f32(0.0f));
            svst1_f32(predicate_f32x, result_row + col_index, svsqrt_f32_x(predicate_f32x, dist_sq_f32x));
        }
    }
}

NK_PUBLIC void nk_euclideans_packed_u4_sme(                //
    nk_u4x2_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,    //
    nk_size_t a_stride_in_bytes, nk_size_t c_stride_in_bytes) {
    nk_size_t const a_stride_elements = a_stride_in_bytes / sizeof(nk_u4x2_t);
    nk_size_t const c_stride_elements = c_stride_in_bytes / sizeof(nk_f32_t);
    nk_dots_packed_u4_sme_streaming_(a, b_packed, (nk_u32_t *)c, rows, columns, depth, a_stride_elements,
                                     c_stride_elements);
    nk_euclideans_packed_u4_sme_finalize_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                    c_stride_elements);
}

__arm_locally_streaming static void nk_angulars_symmetric_u4_sme_finalize_streaming_(          //
    nk_u4x2_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride_elements, //
    nk_f32_t *result, nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) {
    // Phase 1: cache row norms on diagonal (store as u32 in f32 slot)
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_u32_t row_sumsq_u32 = nk_dots_reduce_sumsq_u4_(vectors + row_index * stride_elements, depth);
        ((nk_u32_t *)(result + row_index * result_stride_elements))[row_index] = row_sumsq_u32;
    }
    // Phase 2: column-first post-processing
    nk_u32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < n_vectors; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < n_vectors ? chunk_start + 256 : n_vectors;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_u4_(vectors + col * stride_elements, depth);
        for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
            nk_size_t col_start = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (col_start >= chunk_end) continue;
            nk_f32_t *result_row = result + row_index * result_stride_elements;
            nk_u32_t query_sumsq_u32 = ((nk_u32_t *)result_row)[row_index];
            svfloat32_t query_norm_sq_f32x = svdup_n_f32((nk_f32_t)query_sumsq_u32);
            for (nk_size_t col_index = col_start; col_index < chunk_end; col_index += svcntw()) {
                svbool_t predicate_f32x = svwhilelt_b32_u64(col_index, chunk_end);
                svfloat32_t dots_f32x = svcvt_f32_u32_x(
                    predicate_f32x, svld1_u32(predicate_f32x, (nk_u32_t *)(result_row + col_index)));
                svfloat32_t target_norms_sq_f32x = svcvt_f32_u32_x(
                    predicate_f32x, svld1_u32(predicate_f32x, norms_cache + (col_index - chunk_start)));
                svfloat32_t norms_product_f32x = svmul_f32_x(predicate_f32x, query_norm_sq_f32x, target_norms_sq_f32x);
                svfloat32_t rsqrt_f32x = svrsqrte_f32(norms_product_f32x);
                rsqrt_f32x = svmul_f32_x(
                    predicate_f32x, rsqrt_f32x,
                    svrsqrts_f32(svmul_f32_x(predicate_f32x, norms_product_f32x, rsqrt_f32x), rsqrt_f32x));
                rsqrt_f32x = svmul_f32_x(
                    predicate_f32x, rsqrt_f32x,
                    svrsqrts_f32(svmul_f32_x(predicate_f32x, norms_product_f32x, rsqrt_f32x), rsqrt_f32x));
                svfloat32_t angular_f32x = svsub_f32_x(predicate_f32x, svdup_n_f32(1.0f),
                                                       svmul_f32_x(predicate_f32x, dots_f32x, rsqrt_f32x));
                angular_f32x = svmax_f32_x(predicate_f32x, angular_f32x, svdup_n_f32(0.0f));
                svst1_f32(predicate_f32x, result_row + col_index, angular_f32x);
            }
        }
    }
    // Phase 3: zero diagonals
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NK_PUBLIC void nk_angulars_symmetric_u4_sme(                                          //
    nk_u4x2_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride, //
    nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count) {
    nk_size_t const stride_elements = stride / sizeof(nk_u4x2_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_dots_symmetric_u4_sme_streaming_(vectors, n_vectors, depth, stride_elements, (nk_u32_t *)result,
                                        result_stride_elements, row_start, row_count);
    nk_angulars_symmetric_u4_sme_finalize_streaming_(vectors, n_vectors, depth, stride_elements, result,
                                                     result_stride_elements, row_start, row_count);
}

__arm_locally_streaming static void nk_euclideans_symmetric_u4_sme_finalize_streaming_(        //
    nk_u4x2_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride_elements, //
    nk_f32_t *result, nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) {
    // Phase 1: cache row norms on diagonal (store as u32 in f32 slot)
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_u32_t row_sumsq_u32 = nk_dots_reduce_sumsq_u4_(vectors + row_index * stride_elements, depth);
        ((nk_u32_t *)(result + row_index * result_stride_elements))[row_index] = row_sumsq_u32;
    }
    // Phase 2: column-first post-processing
    nk_u32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < n_vectors; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < n_vectors ? chunk_start + 256 : n_vectors;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_u4_(vectors + col * stride_elements, depth);
        for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
            nk_size_t col_start = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (col_start >= chunk_end) continue;
            nk_f32_t *result_row = result + row_index * result_stride_elements;
            nk_u32_t query_sumsq_u32 = ((nk_u32_t *)result_row)[row_index];
            svfloat32_t query_norm_sq_f32x = svdup_n_f32((nk_f32_t)query_sumsq_u32);
            for (nk_size_t col_index = col_start; col_index < chunk_end; col_index += svcntw()) {
                svbool_t predicate_f32x = svwhilelt_b32_u64(col_index, chunk_end);
                svfloat32_t dots_f32x = svcvt_f32_u32_x(
                    predicate_f32x, svld1_u32(predicate_f32x, (nk_u32_t *)(result_row + col_index)));
                svfloat32_t target_norms_sq_f32x = svcvt_f32_u32_x(
                    predicate_f32x, svld1_u32(predicate_f32x, norms_cache + (col_index - chunk_start)));
                svfloat32_t sum_sq_f32x = svadd_f32_x(predicate_f32x, query_norm_sq_f32x, target_norms_sq_f32x);
                svfloat32_t dist_sq_f32x = svsub_f32_x(predicate_f32x, sum_sq_f32x,
                                                       svmul_f32_x(predicate_f32x, svdup_n_f32(2.0f), dots_f32x));
                dist_sq_f32x = svmax_f32_x(predicate_f32x, dist_sq_f32x, svdup_n_f32(0.0f));
                svst1_f32(predicate_f32x, result_row + col_index, svsqrt_f32_x(predicate_f32x, dist_sq_f32x));
            }
        }
    }
    // Phase 3: zero diagonals
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NK_PUBLIC void nk_euclideans_symmetric_u4_sme(                                        //
    nk_u4x2_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride, //
    nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count) {
    nk_size_t const stride_elements = stride / sizeof(nk_u4x2_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_dots_symmetric_u4_sme_streaming_(vectors, n_vectors, depth, stride_elements, (nk_u32_t *)result,
                                        result_stride_elements, row_start, row_count);
    nk_euclideans_symmetric_u4_sme_finalize_streaming_(vectors, n_vectors, depth, stride_elements, result,
                                                       result_stride_elements, row_start, row_count);
}

#pragma endregion // Nibble Unsigned Integers

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
#endif // NK_SPATIALS_SME_H

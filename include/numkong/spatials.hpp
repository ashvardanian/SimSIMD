/**
 *  @brief C++ wrappers for SIMD-accelerated Batched Spatial Distance Matrices.
 *  @file include/numkong/spatials.hpp
 *  @author Ash Vardanian
 *  @date March 2026
 */
#ifndef NK_SPATIALS_HPP
#define NK_SPATIALS_HPP

#include <cstdint>
#include <type_traits>

#include "numkong/spatials.h"

#include "numkong/types.hpp"

namespace ashvardanian::numkong {

/**
 *  @brief Symmetric angular distance matrix: C[i,j] = angular(A[i], A[j])
 *  @param[in] a Matrix A [n_vectors x depth]
 *  @param[in] n_vectors Number of vectors (n)
 *  @param[in] depth Dimension of each vector (k)
 *  @param[in] a_stride_in_bytes Stride between vectors in A
 *  @param[out] c Output matrix C [n x n]
 *  @param[in] c_stride_in_bytes Stride between rows of C in bytes
 *  @param[in] row_start Starting row index (default 0)
 *  @param[in] row_count Number of rows to compute (default all)
 *
 *  @tparam in_type_ Input element type
 *  @tparam result_type_ Output type, defaults to `in_type_::angular_result_t`
 *  @tparam allow_simd_ Enable SIMD kernel dispatch when `prefer_simd_k`
 */
template <typename in_type_, typename result_type_ = typename in_type_::angular_result_t,
          allow_simd_t allow_simd_ = prefer_simd_k>
void angulars_symmetric(in_type_ const *a, std::size_t n_vectors, std::size_t depth, std::size_t a_stride_in_bytes,
                        result_type_ *c, std::size_t c_stride_in_bytes, std::size_t row_start = 0,
                        std::size_t row_count = std::numeric_limits<std::size_t>::max()) noexcept {
    if (row_count == std::numeric_limits<std::size_t>::max()) row_count = n_vectors;
    constexpr bool dispatch = allow_simd_ == prefer_simd_k &&
                              std::is_same_v<result_type_, typename in_type_::angular_result_t>;

    if constexpr (std::is_same_v<in_type_, f64_t> && dispatch)
        nk_angulars_symmetric_f64(&a->raw_, n_vectors, depth, a_stride_in_bytes, &c->raw_, c_stride_in_bytes, row_start,
                                  row_count);
    else if constexpr (std::is_same_v<in_type_, f32_t> && dispatch)
        nk_angulars_symmetric_f32(&a->raw_, n_vectors, depth, a_stride_in_bytes, &c->raw_, c_stride_in_bytes, row_start,
                                  row_count);
    else if constexpr (std::is_same_v<in_type_, f16_t> && dispatch)
        nk_angulars_symmetric_f16(&a->raw_, n_vectors, depth, a_stride_in_bytes, &c->raw_, c_stride_in_bytes, row_start,
                                  row_count);
    else if constexpr (std::is_same_v<in_type_, bf16_t> && dispatch)
        nk_angulars_symmetric_bf16(&a->raw_, n_vectors, depth, a_stride_in_bytes, &c->raw_, c_stride_in_bytes,
                                   row_start, row_count);
    else if constexpr (std::is_same_v<in_type_, e4m3_t> && dispatch)
        nk_angulars_symmetric_e4m3(&a->raw_, n_vectors, depth, a_stride_in_bytes, &c->raw_, c_stride_in_bytes,
                                   row_start, row_count);
    else if constexpr (std::is_same_v<in_type_, e5m2_t> && dispatch)
        nk_angulars_symmetric_e5m2(&a->raw_, n_vectors, depth, a_stride_in_bytes, &c->raw_, c_stride_in_bytes,
                                   row_start, row_count);
    else if constexpr (std::is_same_v<in_type_, e2m3_t> && dispatch)
        nk_angulars_symmetric_e2m3(&a->raw_, n_vectors, depth, a_stride_in_bytes, &c->raw_, c_stride_in_bytes,
                                   row_start, row_count);
    else if constexpr (std::is_same_v<in_type_, e3m2_t> && dispatch)
        nk_angulars_symmetric_e3m2(&a->raw_, n_vectors, depth, a_stride_in_bytes, &c->raw_, c_stride_in_bytes,
                                   row_start, row_count);
    else if constexpr (std::is_same_v<in_type_, i8_t> && dispatch)
        nk_angulars_symmetric_i8(&a->raw_, n_vectors, depth, a_stride_in_bytes, &c->raw_, c_stride_in_bytes, row_start,
                                 row_count);
    else if constexpr (std::is_same_v<in_type_, u8_t> && dispatch)
        nk_angulars_symmetric_u8(&a->raw_, n_vectors, depth, a_stride_in_bytes, &c->raw_, c_stride_in_bytes, row_start,
                                 row_count);
    else if constexpr (std::is_same_v<in_type_, i4x2_t> && dispatch)
        nk_angulars_symmetric_i4(&a->raw_, n_vectors, depth, a_stride_in_bytes, &c->raw_, c_stride_in_bytes, row_start,
                                 row_count);
    else if constexpr (std::is_same_v<in_type_, u4x2_t> && dispatch)
        nk_angulars_symmetric_u4(&a->raw_, n_vectors, depth, a_stride_in_bytes, &c->raw_, c_stride_in_bytes, row_start,
                                 row_count);
    else {
        std::size_t depth_values = divide_round_up(depth, dimensions_per_value<in_type_>());
        char const *a_bytes = reinterpret_cast<char const *>(a);
        char *c_bytes = reinterpret_cast<char *>(c);
        std::size_t row_end = row_start + row_count < n_vectors ? row_start + row_count : n_vectors;

        for (std::size_t i = row_start; i < row_end; i++) {
            in_type_ const *a_i = reinterpret_cast<in_type_ const *>(a_bytes + i * a_stride_in_bytes);
            result_type_ *c_row = reinterpret_cast<result_type_ *>(c_bytes + i * c_stride_in_bytes);
            for (std::size_t j = 0; j < n_vectors; j++) {
                in_type_ const *a_j = reinterpret_cast<in_type_ const *>(a_bytes + j * a_stride_in_bytes);
                result_type_ ab {}, aa {}, bb {};
                for (std::size_t l = 0; l < depth_values; l++) {
                    ab = fma(a_i[l], a_j[l], ab);
                    aa = fma(a_i[l], a_i[l], aa);
                    bb = fma(a_j[l], a_j[l], bb);
                }
                result_type_ cos_sim = ab / (aa.sqrt() * bb.sqrt());
                result_type_ distance = result_type_(1) - cos_sim;
                c_row[j] = distance > result_type_(0) ? distance : result_type_(0);
            }
        }
    }
}

/**
 *  @brief Symmetric Euclidean distance matrix: C[i,j] = euclidean(A[i], A[j])
 *  @param[in] a Matrix A [n_vectors x depth]
 *  @param[in] n_vectors Number of vectors (n)
 *  @param[in] depth Dimension of each vector (k)
 *  @param[in] a_stride_in_bytes Stride between vectors in A
 *  @param[out] c Output matrix C [n x n]
 *  @param[in] c_stride_in_bytes Stride between rows of C in bytes
 *  @param[in] row_start Starting row index (default 0)
 *  @param[in] row_count Number of rows to compute (default all)
 *
 *  @tparam in_type_ Input element type
 *  @tparam result_type_ Output type, defaults to `in_type_::euclidean_result_t`
 *  @tparam allow_simd_ Enable SIMD kernel dispatch when `prefer_simd_k`
 */
template <typename in_type_, typename result_type_ = typename in_type_::euclidean_result_t,
          allow_simd_t allow_simd_ = prefer_simd_k>
void euclideans_symmetric(in_type_ const *a, std::size_t n_vectors, std::size_t depth, std::size_t a_stride_in_bytes,
                          result_type_ *c, std::size_t c_stride_in_bytes, std::size_t row_start = 0,
                          std::size_t row_count = std::numeric_limits<std::size_t>::max()) noexcept {
    if (row_count == std::numeric_limits<std::size_t>::max()) row_count = n_vectors;
    constexpr bool dispatch = allow_simd_ == prefer_simd_k &&
                              std::is_same_v<result_type_, typename in_type_::euclidean_result_t>;

    if constexpr (std::is_same_v<in_type_, f64_t> && dispatch)
        nk_euclideans_symmetric_f64(&a->raw_, n_vectors, depth, a_stride_in_bytes, &c->raw_, c_stride_in_bytes,
                                    row_start, row_count);
    else if constexpr (std::is_same_v<in_type_, f32_t> && dispatch)
        nk_euclideans_symmetric_f32(&a->raw_, n_vectors, depth, a_stride_in_bytes, &c->raw_, c_stride_in_bytes,
                                    row_start, row_count);
    else if constexpr (std::is_same_v<in_type_, f16_t> && dispatch)
        nk_euclideans_symmetric_f16(&a->raw_, n_vectors, depth, a_stride_in_bytes, &c->raw_, c_stride_in_bytes,
                                    row_start, row_count);
    else if constexpr (std::is_same_v<in_type_, bf16_t> && dispatch)
        nk_euclideans_symmetric_bf16(&a->raw_, n_vectors, depth, a_stride_in_bytes, &c->raw_, c_stride_in_bytes,
                                     row_start, row_count);
    else if constexpr (std::is_same_v<in_type_, e4m3_t> && dispatch)
        nk_euclideans_symmetric_e4m3(&a->raw_, n_vectors, depth, a_stride_in_bytes, &c->raw_, c_stride_in_bytes,
                                     row_start, row_count);
    else if constexpr (std::is_same_v<in_type_, e5m2_t> && dispatch)
        nk_euclideans_symmetric_e5m2(&a->raw_, n_vectors, depth, a_stride_in_bytes, &c->raw_, c_stride_in_bytes,
                                     row_start, row_count);
    else if constexpr (std::is_same_v<in_type_, e2m3_t> && dispatch)
        nk_euclideans_symmetric_e2m3(&a->raw_, n_vectors, depth, a_stride_in_bytes, &c->raw_, c_stride_in_bytes,
                                     row_start, row_count);
    else if constexpr (std::is_same_v<in_type_, e3m2_t> && dispatch)
        nk_euclideans_symmetric_e3m2(&a->raw_, n_vectors, depth, a_stride_in_bytes, &c->raw_, c_stride_in_bytes,
                                     row_start, row_count);
    else if constexpr (std::is_same_v<in_type_, i8_t> && dispatch)
        nk_euclideans_symmetric_i8(&a->raw_, n_vectors, depth, a_stride_in_bytes, &c->raw_, c_stride_in_bytes,
                                   row_start, row_count);
    else if constexpr (std::is_same_v<in_type_, u8_t> && dispatch)
        nk_euclideans_symmetric_u8(&a->raw_, n_vectors, depth, a_stride_in_bytes, &c->raw_, c_stride_in_bytes,
                                   row_start, row_count);
    else if constexpr (std::is_same_v<in_type_, i4x2_t> && dispatch)
        nk_euclideans_symmetric_i4(&a->raw_, n_vectors, depth, a_stride_in_bytes, &c->raw_, c_stride_in_bytes,
                                   row_start, row_count);
    else if constexpr (std::is_same_v<in_type_, u4x2_t> && dispatch)
        nk_euclideans_symmetric_u4(&a->raw_, n_vectors, depth, a_stride_in_bytes, &c->raw_, c_stride_in_bytes,
                                   row_start, row_count);
    else {
        std::size_t depth_values = divide_round_up(depth, dimensions_per_value<in_type_>());
        char const *a_bytes = reinterpret_cast<char const *>(a);
        char *c_bytes = reinterpret_cast<char *>(c);
        std::size_t row_end = row_start + row_count < n_vectors ? row_start + row_count : n_vectors;

        for (std::size_t i = row_start; i < row_end; i++) {
            in_type_ const *a_i = reinterpret_cast<in_type_ const *>(a_bytes + i * a_stride_in_bytes);
            result_type_ *c_row = reinterpret_cast<result_type_ *>(c_bytes + i * c_stride_in_bytes);
            for (std::size_t j = 0; j < n_vectors; j++) {
                in_type_ const *a_j = reinterpret_cast<in_type_ const *>(a_bytes + j * a_stride_in_bytes);
                result_type_ sum {};
                for (std::size_t l = 0; l < depth_values; l++) sum = fdsa(a_i[l], a_j[l], sum);
                c_row[j] = sum.sqrt();
            }
        }
    }
}

} // namespace ashvardanian::numkong

#endif // NK_SPATIALS_HPP

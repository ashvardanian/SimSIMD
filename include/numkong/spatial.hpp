/**
 *  @brief C++ wrappers for SIMD-accelerated Spatial Similarity Measures.
 *  @file include/numkong/spatial.hpp
 *  @author Ash Vardanian
 *  @date February 5, 2026
 */
#ifndef NK_SPATIAL_HPP
#define NK_SPATIAL_HPP

#include <cstdint>
#include <type_traits>

#include "numkong/spatial.h"

#include "numkong/types.hpp"

namespace ashvardanian::numkong {

/**
 *  @brief L₂ (Euclidean) distance: √Σ(aᵢ − bᵢ)²
 *  @param[in] a,b First and second vectors
 *  @param[in] d Number of dimensions in input vectors
 *  @param[out] r Pointer to output distance value
 *
 *  @tparam in_type_ Input vector element type
 *  @tparam result_type_ Accumulator type, defaults to `in_type_::euclidean_result_t`
 *  @tparam allow_simd_ Enable SIMD kernel dispatch when `prefer_simd_k`
 */
template <typename in_type_, typename result_type_ = typename in_type_::euclidean_result_t,
          allow_simd_t allow_simd_ = prefer_simd_k>
void euclidean(in_type_ const *a, in_type_ const *b, std::size_t d, result_type_ *r) noexcept {
    constexpr bool simd = allow_simd_ == prefer_simd_k &&
                          std::is_same_v<result_type_, typename in_type_::euclidean_result_t>;

    if constexpr (std::is_same_v<in_type_, f64_t> && simd) nk_euclidean_f64(&a->raw_, &b->raw_, d, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, f32_t> && simd) nk_euclidean_f32(&a->raw_, &b->raw_, d, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, f16_t> && simd) nk_euclidean_f16(&a->raw_, &b->raw_, d, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, bf16_t> && simd) nk_euclidean_bf16(&a->raw_, &b->raw_, d, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, e4m3_t> && simd) nk_euclidean_e4m3(&a->raw_, &b->raw_, d, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, e5m2_t> && simd) nk_euclidean_e5m2(&a->raw_, &b->raw_, d, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, e2m3_t> && simd) nk_euclidean_e2m3(&a->raw_, &b->raw_, d, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, e3m2_t> && simd) nk_euclidean_e3m2(&a->raw_, &b->raw_, d, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, i8_t> && simd) nk_euclidean_i8(&a->raw_, &b->raw_, d, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, u8_t> && simd) nk_euclidean_u8(&a->raw_, &b->raw_, d, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, i4x2_t> && simd) nk_euclidean_i4(&a->raw_, &b->raw_, d, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, u4x2_t> && simd) nk_euclidean_u4(&a->raw_, &b->raw_, d, &r->raw_);
    // Scalar fallback
    else {
        result_type_ sum {};
        for (std::size_t i = 0; i < divide_round_up(d, dimensions_per_value<in_type_>()); i++)
            sum = fused_difference_squared_add(sum, a[i], b[i]);
        *r = sum.sqrt();
    }
}

/**
 *  @brief Squared L₂ distance: Σ(aᵢ − bᵢ)²
 *  @param[in] a,b First and second vectors
 *  @param[in] d Number of dimensions in input vectors
 *  @param[out] r Pointer to output distance value
 *
 *  @tparam in_type_ Input vector element type
 *  @tparam result_type_ Accumulator type, defaults to `in_type_::sqeuclidean_result_t`
 *  @tparam allow_simd_ Enable SIMD kernel dispatch when `prefer_simd_k`
 */
template <typename in_type_, typename result_type_ = typename in_type_::sqeuclidean_result_t,
          allow_simd_t allow_simd_ = prefer_simd_k>
void sqeuclidean(in_type_ const *a, in_type_ const *b, std::size_t d, result_type_ *r) noexcept {
    constexpr bool simd = allow_simd_ == prefer_simd_k &&
                          std::is_same_v<result_type_, typename in_type_::sqeuclidean_result_t>;

    if constexpr (std::is_same_v<in_type_, f64_t> && simd) nk_sqeuclidean_f64(&a->raw_, &b->raw_, d, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, f32_t> && simd) nk_sqeuclidean_f32(&a->raw_, &b->raw_, d, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, f16_t> && simd) nk_sqeuclidean_f16(&a->raw_, &b->raw_, d, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, bf16_t> && simd) nk_sqeuclidean_bf16(&a->raw_, &b->raw_, d, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, e4m3_t> && simd) nk_sqeuclidean_e4m3(&a->raw_, &b->raw_, d, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, e5m2_t> && simd) nk_sqeuclidean_e5m2(&a->raw_, &b->raw_, d, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, e2m3_t> && simd) nk_sqeuclidean_e2m3(&a->raw_, &b->raw_, d, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, e3m2_t> && simd) nk_sqeuclidean_e3m2(&a->raw_, &b->raw_, d, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, i8_t> && simd) nk_sqeuclidean_i8(&a->raw_, &b->raw_, d, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, u8_t> && simd) nk_sqeuclidean_u8(&a->raw_, &b->raw_, d, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, i4x2_t> && simd) nk_sqeuclidean_i4(&a->raw_, &b->raw_, d, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, u4x2_t> && simd) nk_sqeuclidean_u4(&a->raw_, &b->raw_, d, &r->raw_);
    // Scalar fallback
    else {
        result_type_ sum {};
        for (std::size_t i = 0; i < divide_round_up(d, dimensions_per_value<in_type_>()); i++)
            sum = fused_difference_squared_add(sum, a[i], b[i]);
        *r = sum;
    }
}

/**
 *  @brief Angular similarity (cosine): ⟨a,b⟩ / (‖a‖ × ‖b‖)
 *  @param[in] a,b First and second vectors
 *  @param[in] d Number of dimensions in input vectors
 *  @param[out] r Pointer to output distance value
 *
 *  @tparam in_type_ Input vector element type
 *  @tparam result_type_ Accumulator type, defaults to `in_type_::angular_result_t`
 *  @tparam allow_simd_ Enable SIMD kernel dispatch when `prefer_simd_k`
 */
template <typename in_type_, typename result_type_ = typename in_type_::angular_result_t,
          allow_simd_t allow_simd_ = prefer_simd_k>
void angular(in_type_ const *a, in_type_ const *b, std::size_t d, result_type_ *r) noexcept {
    constexpr bool simd = allow_simd_ == prefer_simd_k &&
                          std::is_same_v<result_type_, typename in_type_::angular_result_t>;

    if constexpr (std::is_same_v<in_type_, f64_t> && simd) nk_angular_f64(&a->raw_, &b->raw_, d, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, f32_t> && simd) nk_angular_f32(&a->raw_, &b->raw_, d, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, f16_t> && simd) nk_angular_f16(&a->raw_, &b->raw_, d, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, bf16_t> && simd) nk_angular_bf16(&a->raw_, &b->raw_, d, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, e4m3_t> && simd) nk_angular_e4m3(&a->raw_, &b->raw_, d, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, e5m2_t> && simd) nk_angular_e5m2(&a->raw_, &b->raw_, d, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, e2m3_t> && simd) nk_angular_e2m3(&a->raw_, &b->raw_, d, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, e3m2_t> && simd) nk_angular_e3m2(&a->raw_, &b->raw_, d, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, i8_t> && simd) nk_angular_i8(&a->raw_, &b->raw_, d, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, u8_t> && simd) nk_angular_u8(&a->raw_, &b->raw_, d, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, i4x2_t> && simd) nk_angular_i4(&a->raw_, &b->raw_, d, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, u4x2_t> && simd) nk_angular_u4(&a->raw_, &b->raw_, d, &r->raw_);
    // Scalar fallback
    else {
        result_type_ ab {}, aa {}, bb {};
        for (std::size_t i = 0; i < divide_round_up(d, dimensions_per_value<in_type_>()); i++) {
            ab = fused_multiply_add(ab, a[i], b[i]);
            aa = fused_multiply_add(aa, a[i], a[i]);
            bb = fused_multiply_add(bb, b[i], b[i]);
        }
        // Angular distance = 1 - cosine_similarity, clamped to [0, 2]
        result_type_ cos_sim = ab / (aa.sqrt() * bb.sqrt());
        result_type_ distance = result_type_(1) - cos_sim;
        *r = distance > result_type_(0) ? distance : result_type_(0);
    }
}

} // namespace ashvardanian::numkong

#endif // NK_SPATIAL_HPP

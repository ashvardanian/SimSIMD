/**
 *  @brief C++ wrappers for SIMD-accelerated Elementwise Arithmetic.
 *  @file include/numkong/each.hpp
 *  @author Ash Vardanian
 *  @date February 5, 2026
 */
#ifndef NK_EACH_HPP
#define NK_EACH_HPP

#include <cstdint>
#include <type_traits>

#include "numkong/each.h"

#include "numkong/types.hpp"

namespace ashvardanian::numkong {

/**
 *  @brief Elementwise sum: cᵢ = aᵢ + bᵢ
 *  @param[in] a,b Input vectors
 *  @param[in] d Number of dimensions in input vectors
 *  @param[out] c Output vector
 *
 *  @tparam in_type_ Element type
 *  @tparam allow_simd_ Enable SIMD kernel dispatch when `prefer_simd_k`
 */
template <typename in_type_, allow_simd_t allow_simd_ = prefer_simd_k>
void sum(in_type_ const *a, in_type_ const *b, std::size_t d, in_type_ *c) noexcept {
    constexpr bool simd = allow_simd_ == prefer_simd_k;

    if constexpr (std::is_same_v<in_type_, f64_t> && simd) nk_sum_f64(&a->raw_, &b->raw_, d, &c->raw_);
    else if constexpr (std::is_same_v<in_type_, f32_t> && simd) nk_sum_f32(&a->raw_, &b->raw_, d, &c->raw_);
    else if constexpr (std::is_same_v<in_type_, f16_t> && simd) nk_sum_f16(&a->raw_, &b->raw_, d, &c->raw_);
    else if constexpr (std::is_same_v<in_type_, bf16_t> && simd) nk_sum_bf16(&a->raw_, &b->raw_, d, &c->raw_);
    else if constexpr (std::is_same_v<in_type_, i8_t> && simd) nk_sum_i8(&a->raw_, &b->raw_, d, &c->raw_);
    else if constexpr (std::is_same_v<in_type_, u8_t> && simd) nk_sum_u8(&a->raw_, &b->raw_, d, &c->raw_);
    else if constexpr (std::is_same_v<in_type_, i16_t> && simd) nk_sum_i16(&a->raw_, &b->raw_, d, &c->raw_);
    else if constexpr (std::is_same_v<in_type_, u16_t> && simd) nk_sum_u16(&a->raw_, &b->raw_, d, &c->raw_);
    else if constexpr (std::is_same_v<in_type_, i32_t> && simd) nk_sum_i32(&a->raw_, &b->raw_, d, &c->raw_);
    else if constexpr (std::is_same_v<in_type_, u32_t> && simd) nk_sum_u32(&a->raw_, &b->raw_, d, &c->raw_);
    else if constexpr (std::is_same_v<in_type_, i64_t> && simd) nk_sum_i64(&a->raw_, &b->raw_, d, &c->raw_);
    else if constexpr (std::is_same_v<in_type_, u64_t> && simd) nk_sum_u64(&a->raw_, &b->raw_, d, &c->raw_);
    // Scalar fallback
    else {
        for (std::size_t i = 0; i < d; i++) c[i] = a[i].saturating_add(b[i]);
    }
}

/**
 *  @brief Elementwise scale: cᵢ = α × aᵢ + β
 *  @param[in] a Input vector
 *  @param[in] d Number of dimensions in input vector
 *  @param[in] alpha,beta Scale and shift coefficients
 *  @param[out] c Output vector
 *
 *  @tparam in_type_ Element type
 *  @tparam precision_type_ Precision type for scalar fallback computations, defaults to `in_type_`
 *  @tparam allow_simd_ Enable SIMD kernel dispatch when `prefer_simd_k`
 */
template <typename in_type_, typename precision_type_ = in_type_, allow_simd_t allow_simd_ = prefer_simd_k>
void scale(in_type_ const *a, std::size_t d, typename in_type_::scale_t const *alpha,
           typename in_type_::scale_t const *beta, in_type_ *c) noexcept {
    constexpr bool simd = allow_simd_ == prefer_simd_k && std::is_same_v<precision_type_, in_type_>;

    if constexpr (std::is_same_v<in_type_, f64_t> && simd) nk_scale_f64(&a->raw_, d, alpha, beta, &c->raw_);
    else if constexpr (std::is_same_v<in_type_, f32_t> && simd) nk_scale_f32(&a->raw_, d, alpha, beta, &c->raw_);
    else if constexpr (std::is_same_v<in_type_, f16_t> && simd) nk_scale_f16(&a->raw_, d, alpha, beta, &c->raw_);
    else if constexpr (std::is_same_v<in_type_, bf16_t> && simd) nk_scale_bf16(&a->raw_, d, alpha, beta, &c->raw_);
    else if constexpr (std::is_same_v<in_type_, i8_t> && simd) nk_scale_i8(&a->raw_, d, alpha, beta, &c->raw_);
    else if constexpr (std::is_same_v<in_type_, u8_t> && simd) nk_scale_u8(&a->raw_, d, alpha, beta, &c->raw_);
    else if constexpr (std::is_same_v<in_type_, i16_t> && simd) nk_scale_i16(&a->raw_, d, alpha, beta, &c->raw_);
    else if constexpr (std::is_same_v<in_type_, u16_t> && simd) nk_scale_u16(&a->raw_, d, alpha, beta, &c->raw_);
    else if constexpr (std::is_same_v<in_type_, i32_t> && simd) nk_scale_i32(&a->raw_, d, alpha, beta, &c->raw_);
    else if constexpr (std::is_same_v<in_type_, u32_t> && simd) nk_scale_u32(&a->raw_, d, alpha, beta, &c->raw_);
    else if constexpr (std::is_same_v<in_type_, i64_t> && simd) nk_scale_i64(&a->raw_, d, alpha, beta, &c->raw_);
    else if constexpr (std::is_same_v<in_type_, u64_t> && simd) nk_scale_u64(&a->raw_, d, alpha, beta, &c->raw_);
    // Scalar fallback with high-precision intermediates
    else {
        for (std::size_t i = 0; i < d; i++)
            c[i] = (precision_type_(a[i]) * precision_type_(*alpha) + precision_type_(*beta)).template to<in_type_>();
    }
}

/**
 *  @brief Weighted sum: cᵢ = α × aᵢ + β × bᵢ
 *  @param[in] a,b Input vectors
 *  @param[in] d Number of dimensions in input vectors
 *  @param[in] alpha,beta Weight coefficients
 *  @param[out] c Output vector
 *
 *  @tparam in_type_ Element type
 *  @tparam precision_type_ Precision type for scalar fallback computations, defaults to `in_type_`
 *  @tparam allow_simd_ Enable SIMD kernel dispatch when `prefer_simd_k`
 */
template <typename in_type_, typename precision_type_ = in_type_, allow_simd_t allow_simd_ = prefer_simd_k>
void wsum(in_type_ const *a, in_type_ const *b, std::size_t d, typename in_type_::scale_t const *alpha,
          typename in_type_::scale_t const *beta, in_type_ *c) noexcept {
    constexpr bool simd = allow_simd_ == prefer_simd_k && std::is_same_v<precision_type_, in_type_>;

    if constexpr (std::is_same_v<in_type_, f64_t> && simd) nk_wsum_f64(&a->raw_, &b->raw_, d, alpha, beta, &c->raw_);
    else if constexpr (std::is_same_v<in_type_, f32_t> && simd)
        nk_wsum_f32(&a->raw_, &b->raw_, d, alpha, beta, &c->raw_);
    else if constexpr (std::is_same_v<in_type_, f16_t> && simd)
        nk_wsum_f16(&a->raw_, &b->raw_, d, alpha, beta, &c->raw_);
    else if constexpr (std::is_same_v<in_type_, bf16_t> && simd)
        nk_wsum_bf16(&a->raw_, &b->raw_, d, alpha, beta, &c->raw_);
    else if constexpr (std::is_same_v<in_type_, i8_t> && simd) nk_wsum_i8(&a->raw_, &b->raw_, d, alpha, beta, &c->raw_);
    else if constexpr (std::is_same_v<in_type_, u8_t> && simd) nk_wsum_u8(&a->raw_, &b->raw_, d, alpha, beta, &c->raw_);
    else if constexpr (std::is_same_v<in_type_, i16_t> && simd)
        nk_wsum_i16(&a->raw_, &b->raw_, d, alpha, beta, &c->raw_);
    else if constexpr (std::is_same_v<in_type_, u16_t> && simd)
        nk_wsum_u16(&a->raw_, &b->raw_, d, alpha, beta, &c->raw_);
    else if constexpr (std::is_same_v<in_type_, i32_t> && simd)
        nk_wsum_i32(&a->raw_, &b->raw_, d, alpha, beta, &c->raw_);
    else if constexpr (std::is_same_v<in_type_, u32_t> && simd)
        nk_wsum_u32(&a->raw_, &b->raw_, d, alpha, beta, &c->raw_);
    else if constexpr (std::is_same_v<in_type_, i64_t> && simd)
        nk_wsum_i64(&a->raw_, &b->raw_, d, alpha, beta, &c->raw_);
    else if constexpr (std::is_same_v<in_type_, u64_t> && simd)
        nk_wsum_u64(&a->raw_, &b->raw_, d, alpha, beta, &c->raw_);
    // Scalar fallback with high-precision intermediates
    else {
        for (std::size_t i = 0; i < d; i++) {
            c[i] = (precision_type_(a[i]) * precision_type_(*alpha) + precision_type_(b[i]) * precision_type_(*beta))
                       .template to<in_type_>();
        }
    }
}

/**
 *  @brief Elementwise FMA: outᵢ = α × aᵢ × bᵢ + β × cᵢ
 *  @param[in] a,b,c Input vectors
 *  @param[in] d Number of dimensions in input vectors
 *  @param[in] alpha,beta Coefficients
 *  @param[out] out Output vector
 *
 *  @tparam in_type_ Element type
 *  @tparam precision_type_ Precision type for scalar fallback computations, defaults to `in_type_`
 *  @tparam allow_simd_ Enable SIMD kernel dispatch when `prefer_simd_k`
 */
template <typename in_type_, typename precision_type_ = in_type_, allow_simd_t allow_simd_ = prefer_simd_k>
void fma(in_type_ const *a, in_type_ const *b, std::size_t d, in_type_ const *c,
         typename in_type_::scale_t const *alpha, typename in_type_::scale_t const *beta, in_type_ *out) noexcept {
    constexpr bool simd = allow_simd_ == prefer_simd_k && std::is_same_v<precision_type_, in_type_>;

    if constexpr (std::is_same_v<in_type_, f64_t> && simd)
        nk_fma_f64(&a->raw_, &b->raw_, &c->raw_, d, alpha, beta, &out->raw_);
    else if constexpr (std::is_same_v<in_type_, f32_t> && simd)
        nk_fma_f32(&a->raw_, &b->raw_, &c->raw_, d, alpha, beta, &out->raw_);
    else if constexpr (std::is_same_v<in_type_, f16_t> && simd)
        nk_fma_f16(&a->raw_, &b->raw_, &c->raw_, d, alpha, beta, &out->raw_);
    else if constexpr (std::is_same_v<in_type_, bf16_t> && simd)
        nk_fma_bf16(&a->raw_, &b->raw_, &c->raw_, d, alpha, beta, &out->raw_);
    else if constexpr (std::is_same_v<in_type_, i8_t> && simd)
        nk_fma_i8(&a->raw_, &b->raw_, &c->raw_, d, alpha, beta, &out->raw_);
    else if constexpr (std::is_same_v<in_type_, u8_t> && simd)
        nk_fma_u8(&a->raw_, &b->raw_, &c->raw_, d, alpha, beta, &out->raw_);
    else if constexpr (std::is_same_v<in_type_, i16_t> && simd)
        nk_fma_i16(&a->raw_, &b->raw_, &c->raw_, d, alpha, beta, &out->raw_);
    else if constexpr (std::is_same_v<in_type_, u16_t> && simd)
        nk_fma_u16(&a->raw_, &b->raw_, &c->raw_, d, alpha, beta, &out->raw_);
    else if constexpr (std::is_same_v<in_type_, i32_t> && simd)
        nk_fma_i32(&a->raw_, &b->raw_, &c->raw_, d, alpha, beta, &out->raw_);
    else if constexpr (std::is_same_v<in_type_, u32_t> && simd)
        nk_fma_u32(&a->raw_, &b->raw_, &c->raw_, d, alpha, beta, &out->raw_);
    else if constexpr (std::is_same_v<in_type_, i64_t> && simd)
        nk_fma_i64(&a->raw_, &b->raw_, &c->raw_, d, alpha, beta, &out->raw_);
    else if constexpr (std::is_same_v<in_type_, u64_t> && simd)
        nk_fma_u64(&a->raw_, &b->raw_, &c->raw_, d, alpha, beta, &out->raw_);
    // Scalar fallback with high-precision intermediates
    else {
        for (std::size_t i = 0; i < d; i++) {
            out[i] = (precision_type_(a[i]) * precision_type_(b[i]) * precision_type_(*alpha) +
                      precision_type_(c[i]) * precision_type_(*beta))
                         .template to<in_type_>();
        }
    }
}

} // namespace ashvardanian::numkong

#endif // NK_EACH_HPP

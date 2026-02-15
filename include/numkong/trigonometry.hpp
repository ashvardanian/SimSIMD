/**
 *  @brief C++ bindings for trigonometric kernels.
 *  @file include/numkong/trigonometry.hpp
 *  @author Ash Vardanian
 *  @date February 5, 2026
 */
#ifndef NK_TRIGONOMETRY_HPP
#define NK_TRIGONOMETRY_HPP

#include <cstdint>
#include <type_traits>

#include "numkong/trigonometry.h"

#include "numkong/types.hpp"

namespace ashvardanian::numkong {

/**
 *  @brief Array sine: outᵢ = sin(inᵢ)
 *  @param[in] in Input array
 *  @param[in] n Number of elements
 *  @param[out] out Output array
 *
 *  @tparam in_type_ Element type (f32_t, f64_t, f16_t)
 *  @tparam precision_type_ Precision type for scalar fallback, defaults to `in_type_`
 *  @tparam allow_simd_ Enable SIMD kernel dispatch when `prefer_simd_k`
 */
template <typename in_type_, typename precision_type_ = in_type_, allow_simd_t allow_simd_ = prefer_simd_k>
void sin(in_type_ const *in, std::size_t n, in_type_ *out) noexcept {
    constexpr bool simd = allow_simd_ == prefer_simd_k && std::is_same_v<in_type_, precision_type_>;

    if constexpr (std::is_same_v<in_type_, f64_t> && simd) nk_each_sin_f64(&in->raw_, n, &out->raw_);
    else if constexpr (std::is_same_v<in_type_, f32_t> && simd) nk_each_sin_f32(&in->raw_, n, &out->raw_);
    else if constexpr (std::is_same_v<in_type_, f16_t> && simd) nk_each_sin_f16(&in->raw_, n, &out->raw_);
    // Scalar fallback
    else {
        for (std::size_t i = 0; i < n; i++) out[i] = precision_type_(in[i]).sin().template to<in_type_>();
    }
}

/**
 *  @brief Array cosine: outᵢ = cos(inᵢ)
 *  @param[in] in Input array
 *  @param[in] n Number of elements
 *  @param[out] out Output array
 *
 *  @tparam in_type_ Element type (f32_t, f64_t, f16_t)
 *  @tparam precision_type_ Precision type for scalar fallback, defaults to `in_type_`
 *  @tparam allow_simd_ Enable SIMD kernel dispatch when `prefer_simd_k`
 */
template <typename in_type_, typename precision_type_ = in_type_, allow_simd_t allow_simd_ = prefer_simd_k>
void cos(in_type_ const *in, std::size_t n, in_type_ *out) noexcept {
    constexpr bool simd = allow_simd_ == prefer_simd_k && std::is_same_v<in_type_, precision_type_>;

    if constexpr (std::is_same_v<in_type_, f64_t> && simd) nk_each_cos_f64(&in->raw_, n, &out->raw_);
    else if constexpr (std::is_same_v<in_type_, f32_t> && simd) nk_each_cos_f32(&in->raw_, n, &out->raw_);
    else if constexpr (std::is_same_v<in_type_, f16_t> && simd) nk_each_cos_f16(&in->raw_, n, &out->raw_);
    // Scalar fallback
    else {
        for (std::size_t i = 0; i < n; i++) out[i] = precision_type_(in[i]).cos().template to<in_type_>();
    }
}

/**
 *  @brief Array arctangent: outᵢ = arctan(inᵢ)
 *  @param[in] in Input array
 *  @param[in] n Number of elements
 *  @param[out] out Output array
 *
 *  @tparam in_type_ Element type (f32_t, f64_t, f16_t)
 *  @tparam precision_type_ Precision type for scalar fallback, defaults to `in_type_`
 *  @tparam allow_simd_ Enable SIMD kernel dispatch when `prefer_simd_k`
 */
template <typename in_type_, typename precision_type_ = in_type_, allow_simd_t allow_simd_ = prefer_simd_k>
void atan(in_type_ const *in, std::size_t n, in_type_ *out) noexcept {
    constexpr bool simd = allow_simd_ == prefer_simd_k && std::is_same_v<in_type_, precision_type_>;

    if constexpr (std::is_same_v<in_type_, f64_t> && simd) nk_each_atan_f64(&in->raw_, n, &out->raw_);
    else if constexpr (std::is_same_v<in_type_, f32_t> && simd) nk_each_atan_f32(&in->raw_, n, &out->raw_);
    else if constexpr (std::is_same_v<in_type_, f16_t> && simd) nk_each_atan_f16(&in->raw_, n, &out->raw_);
    // Scalar fallback
    else {
        for (std::size_t i = 0; i < n; i++) out[i] = precision_type_(in[i]).atan().template to<in_type_>();
    }
}

} // namespace ashvardanian::numkong

#endif // NK_TRIGONOMETRY_HPP

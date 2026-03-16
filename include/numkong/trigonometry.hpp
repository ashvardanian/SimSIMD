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
template <numeric_dtype in_type_, numeric_dtype precision_type_ = in_type_, allow_simd_t allow_simd_ = prefer_simd_k>
void sin(in_type_ const *in, std::size_t n, in_type_ *out) noexcept {
    constexpr bool simd = allow_simd_ == prefer_simd_k && std::is_same_v<in_type_, precision_type_>;

    if constexpr (std::is_same_v<in_type_, f64_t> && simd) nk_each_sin_f64(&in->raw_, n, &out->raw_);
    else if constexpr (std::is_same_v<in_type_, f32_t> && simd) nk_each_sin_f32(&in->raw_, n, &out->raw_);
    else if constexpr (std::is_same_v<in_type_, f16_t> && simd) nk_each_sin_f16(&in->raw_, n, &out->raw_);
    // Scalar fallback
    else {
        for (std::size_t i = 0; i < n; i++) out[i] = in_type_(precision_type_(in[i]).sin());
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
template <numeric_dtype in_type_, numeric_dtype precision_type_ = in_type_, allow_simd_t allow_simd_ = prefer_simd_k>
void cos(in_type_ const *in, std::size_t n, in_type_ *out) noexcept {
    constexpr bool simd = allow_simd_ == prefer_simd_k && std::is_same_v<in_type_, precision_type_>;

    if constexpr (std::is_same_v<in_type_, f64_t> && simd) nk_each_cos_f64(&in->raw_, n, &out->raw_);
    else if constexpr (std::is_same_v<in_type_, f32_t> && simd) nk_each_cos_f32(&in->raw_, n, &out->raw_);
    else if constexpr (std::is_same_v<in_type_, f16_t> && simd) nk_each_cos_f16(&in->raw_, n, &out->raw_);
    // Scalar fallback
    else {
        for (std::size_t i = 0; i < n; i++) out[i] = in_type_(precision_type_(in[i]).cos());
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
template <numeric_dtype in_type_, numeric_dtype precision_type_ = in_type_, allow_simd_t allow_simd_ = prefer_simd_k>
void atan(in_type_ const *in, std::size_t n, in_type_ *out) noexcept {
    constexpr bool simd = allow_simd_ == prefer_simd_k && std::is_same_v<in_type_, precision_type_>;

    if constexpr (std::is_same_v<in_type_, f64_t> && simd) nk_each_atan_f64(&in->raw_, n, &out->raw_);
    else if constexpr (std::is_same_v<in_type_, f32_t> && simd) nk_each_atan_f32(&in->raw_, n, &out->raw_);
    else if constexpr (std::is_same_v<in_type_, f16_t> && simd) nk_each_atan_f16(&in->raw_, n, &out->raw_);
    // Scalar fallback
    else {
        for (std::size_t i = 0; i < n; i++) out[i] = in_type_(precision_type_(in[i]).atan());
    }
}

} // namespace ashvardanian::numkong

#include "numkong/tensor.hpp"

namespace ashvardanian::numkong {

#pragma region - Tensor Trigonometric

/** @brief Elementwise sin into pre-allocated output. */
template <numeric_dtype value_type_, std::size_t max_rank_ = 8>
bool sin(tensor_view<value_type_, max_rank_> input, tensor_span<value_type_, max_rank_> output) noexcept {
    return elementwise_into_<value_type_, max_rank_>(
        input, output, [](tensor_view<value_type_, max_rank_> in, tensor_span<value_type_, max_rank_> out) {
            numkong::sin<value_type_>(in.data(), in.extent(0), out.data());
        });
}

/** @brief Allocating sin. */
template <numeric_dtype value_type_, std::size_t max_rank_ = 8, typename allocator_type_ = aligned_allocator<value_type_>>
tensor<value_type_, allocator_type_, max_rank_> try_sin(tensor_view<value_type_, max_rank_> input) noexcept {
    using out_tensor_t = tensor<value_type_, allocator_type_, max_rank_>;
    if (input.empty()) return out_tensor_t {};
    auto &input_shape = input.shape();
    auto result = out_tensor_t::try_empty(input_shape.extents, input_shape.rank);
    if (result.empty()) return result;
    if (!sin<value_type_, max_rank_>(input, result.span())) return out_tensor_t {};
    return result;
}

/** @brief Elementwise cos into pre-allocated output. */
template <numeric_dtype value_type_, std::size_t max_rank_ = 8>
bool cos(tensor_view<value_type_, max_rank_> input, tensor_span<value_type_, max_rank_> output) noexcept {
    return elementwise_into_<value_type_, max_rank_>(
        input, output, [](tensor_view<value_type_, max_rank_> in, tensor_span<value_type_, max_rank_> out) {
            numkong::cos<value_type_>(in.data(), in.extent(0), out.data());
        });
}

/** @brief Allocating cos. */
template <numeric_dtype value_type_, std::size_t max_rank_ = 8, typename allocator_type_ = aligned_allocator<value_type_>>
tensor<value_type_, allocator_type_, max_rank_> try_cos(tensor_view<value_type_, max_rank_> input) noexcept {
    using out_tensor_t = tensor<value_type_, allocator_type_, max_rank_>;
    if (input.empty()) return out_tensor_t {};
    auto &input_shape = input.shape();
    auto result = out_tensor_t::try_empty(input_shape.extents, input_shape.rank);
    if (result.empty()) return result;
    if (!cos<value_type_, max_rank_>(input, result.span())) return out_tensor_t {};
    return result;
}

/** @brief Elementwise atan into pre-allocated output. */
template <numeric_dtype value_type_, std::size_t max_rank_ = 8>
bool atan(tensor_view<value_type_, max_rank_> input, tensor_span<value_type_, max_rank_> output) noexcept {
    return elementwise_into_<value_type_, max_rank_>(
        input, output, [](tensor_view<value_type_, max_rank_> in, tensor_span<value_type_, max_rank_> out) {
            numkong::atan<value_type_>(in.data(), in.extent(0), out.data());
        });
}

/** @brief Allocating atan. */
template <numeric_dtype value_type_, std::size_t max_rank_ = 8, typename allocator_type_ = aligned_allocator<value_type_>>
tensor<value_type_, allocator_type_, max_rank_> try_atan(tensor_view<value_type_, max_rank_> input) noexcept {
    using out_tensor_t = tensor<value_type_, allocator_type_, max_rank_>;
    if (input.empty()) return out_tensor_t {};
    auto &input_shape = input.shape();
    auto result = out_tensor_t::try_empty(input_shape.extents, input_shape.rank);
    if (result.empty()) return result;
    if (!atan<value_type_, max_rank_>(input, result.span())) return out_tensor_t {};
    return result;
}

#pragma endregion - Tensor Trigonometric

} // namespace ashvardanian::numkong

#endif // NK_TRIGONOMETRY_HPP

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
template <numeric_dtype in_type_, allow_simd_t allow_simd_ = prefer_simd_k>
void sum(in_type_ const *a, in_type_ const *b, std::size_t d, in_type_ *c) noexcept {
    constexpr bool simd = allow_simd_ == prefer_simd_k;

    if constexpr (std::is_same_v<in_type_, f64_t> && simd) nk_each_sum_f64(&a->raw_, &b->raw_, d, &c->raw_);
    else if constexpr (std::is_same_v<in_type_, f32_t> && simd) nk_each_sum_f32(&a->raw_, &b->raw_, d, &c->raw_);
    else if constexpr (std::is_same_v<in_type_, f16_t> && simd) nk_each_sum_f16(&a->raw_, &b->raw_, d, &c->raw_);
    else if constexpr (std::is_same_v<in_type_, bf16_t> && simd) nk_each_sum_bf16(&a->raw_, &b->raw_, d, &c->raw_);
    else if constexpr (std::is_same_v<in_type_, i8_t> && simd) nk_each_sum_i8(&a->raw_, &b->raw_, d, &c->raw_);
    else if constexpr (std::is_same_v<in_type_, u8_t> && simd) nk_each_sum_u8(&a->raw_, &b->raw_, d, &c->raw_);
    else if constexpr (std::is_same_v<in_type_, i16_t> && simd) nk_each_sum_i16(&a->raw_, &b->raw_, d, &c->raw_);
    else if constexpr (std::is_same_v<in_type_, u16_t> && simd) nk_each_sum_u16(&a->raw_, &b->raw_, d, &c->raw_);
    else if constexpr (std::is_same_v<in_type_, i32_t> && simd) nk_each_sum_i32(&a->raw_, &b->raw_, d, &c->raw_);
    else if constexpr (std::is_same_v<in_type_, u32_t> && simd) nk_each_sum_u32(&a->raw_, &b->raw_, d, &c->raw_);
    else if constexpr (std::is_same_v<in_type_, i64_t> && simd) nk_each_sum_i64(&a->raw_, &b->raw_, d, &c->raw_);
    else if constexpr (std::is_same_v<in_type_, u64_t> && simd) nk_each_sum_u64(&a->raw_, &b->raw_, d, &c->raw_);
    else if constexpr (std::is_same_v<in_type_, f32c_t> && simd) nk_each_sum_f32c(&a->raw_, &b->raw_, d, &c->raw_);
    else if constexpr (std::is_same_v<in_type_, f64c_t> && simd) nk_each_sum_f64c(&a->raw_, &b->raw_, d, &c->raw_);
    // Scalar fallback
    else {
        for (std::size_t i = 0; i < d; i++) c[i] = saturating_add(a[i], b[i]);
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
template <numeric_dtype in_type_, numeric_dtype precision_type_ = in_type_, allow_simd_t allow_simd_ = prefer_simd_k>
void scale(in_type_ const *a, std::size_t d, typename in_type_::scale_t const *alpha,
           typename in_type_::scale_t const *beta, in_type_ *c) noexcept {
    constexpr bool simd = allow_simd_ == prefer_simd_k && std::is_same_v<precision_type_, in_type_>;

    if constexpr (std::is_same_v<in_type_, f64_t> && simd) nk_each_scale_f64(&a->raw_, d, alpha, beta, &c->raw_);
    else if constexpr (std::is_same_v<in_type_, f32_t> && simd) nk_each_scale_f32(&a->raw_, d, alpha, beta, &c->raw_);
    else if constexpr (std::is_same_v<in_type_, f16_t> && simd) nk_each_scale_f16(&a->raw_, d, alpha, beta, &c->raw_);
    else if constexpr (std::is_same_v<in_type_, bf16_t> && simd) nk_each_scale_bf16(&a->raw_, d, alpha, beta, &c->raw_);
    else if constexpr (std::is_same_v<in_type_, i8_t> && simd) nk_each_scale_i8(&a->raw_, d, alpha, beta, &c->raw_);
    else if constexpr (std::is_same_v<in_type_, u8_t> && simd) nk_each_scale_u8(&a->raw_, d, alpha, beta, &c->raw_);
    else if constexpr (std::is_same_v<in_type_, i16_t> && simd) nk_each_scale_i16(&a->raw_, d, alpha, beta, &c->raw_);
    else if constexpr (std::is_same_v<in_type_, u16_t> && simd) nk_each_scale_u16(&a->raw_, d, alpha, beta, &c->raw_);
    else if constexpr (std::is_same_v<in_type_, i32_t> && simd) nk_each_scale_i32(&a->raw_, d, alpha, beta, &c->raw_);
    else if constexpr (std::is_same_v<in_type_, u32_t> && simd) nk_each_scale_u32(&a->raw_, d, alpha, beta, &c->raw_);
    else if constexpr (std::is_same_v<in_type_, i64_t> && simd) nk_each_scale_i64(&a->raw_, d, alpha, beta, &c->raw_);
    else if constexpr (std::is_same_v<in_type_, u64_t> && simd) nk_each_scale_u64(&a->raw_, d, alpha, beta, &c->raw_);
    else if constexpr (std::is_same_v<in_type_, f32c_t> && simd) nk_each_scale_f32c(&a->raw_, d, alpha, beta, &c->raw_);
    else if constexpr (std::is_same_v<in_type_, f64c_t> && simd) nk_each_scale_f64c(&a->raw_, d, alpha, beta, &c->raw_);
    // Scalar fallback with high-precision intermediates
    else {
        for (std::size_t i = 0; i < d; i++)
            c[i] = (precision_type_(a[i]) * precision_type_(*alpha) + precision_type_(*beta)).template to<in_type_>();
    }
}

/**
 *  @brief Blend: cᵢ = α × aᵢ + β × bᵢ
 *  @param[in] a,b Input vectors
 *  @param[in] d Number of dimensions in input vectors
 *  @param[in] alpha,beta Weight coefficients
 *  @param[out] c Output vector
 *
 *  @tparam in_type_ Element type
 *  @tparam precision_type_ Precision type for scalar fallback computations, defaults to `in_type_`
 *  @tparam allow_simd_ Enable SIMD kernel dispatch when `prefer_simd_k`
 */
template <numeric_dtype in_type_, numeric_dtype precision_type_ = in_type_, allow_simd_t allow_simd_ = prefer_simd_k>
void blend(in_type_ const *a, in_type_ const *b, std::size_t d, typename in_type_::scale_t const *alpha,
           typename in_type_::scale_t const *beta, in_type_ *c) noexcept {
    constexpr bool simd = allow_simd_ == prefer_simd_k && std::is_same_v<precision_type_, in_type_>;

    if constexpr (std::is_same_v<in_type_, f64_t> && simd)
        nk_each_blend_f64(&a->raw_, &b->raw_, d, alpha, beta, &c->raw_);
    else if constexpr (std::is_same_v<in_type_, f32_t> && simd)
        nk_each_blend_f32(&a->raw_, &b->raw_, d, alpha, beta, &c->raw_);
    else if constexpr (std::is_same_v<in_type_, f16_t> && simd)
        nk_each_blend_f16(&a->raw_, &b->raw_, d, alpha, beta, &c->raw_);
    else if constexpr (std::is_same_v<in_type_, bf16_t> && simd)
        nk_each_blend_bf16(&a->raw_, &b->raw_, d, alpha, beta, &c->raw_);
    else if constexpr (std::is_same_v<in_type_, i8_t> && simd)
        nk_each_blend_i8(&a->raw_, &b->raw_, d, alpha, beta, &c->raw_);
    else if constexpr (std::is_same_v<in_type_, u8_t> && simd)
        nk_each_blend_u8(&a->raw_, &b->raw_, d, alpha, beta, &c->raw_);
    else if constexpr (std::is_same_v<in_type_, i16_t> && simd)
        nk_each_blend_i16(&a->raw_, &b->raw_, d, alpha, beta, &c->raw_);
    else if constexpr (std::is_same_v<in_type_, u16_t> && simd)
        nk_each_blend_u16(&a->raw_, &b->raw_, d, alpha, beta, &c->raw_);
    else if constexpr (std::is_same_v<in_type_, i32_t> && simd)
        nk_each_blend_i32(&a->raw_, &b->raw_, d, alpha, beta, &c->raw_);
    else if constexpr (std::is_same_v<in_type_, u32_t> && simd)
        nk_each_blend_u32(&a->raw_, &b->raw_, d, alpha, beta, &c->raw_);
    else if constexpr (std::is_same_v<in_type_, i64_t> && simd)
        nk_each_blend_i64(&a->raw_, &b->raw_, d, alpha, beta, &c->raw_);
    else if constexpr (std::is_same_v<in_type_, u64_t> && simd)
        nk_each_blend_u64(&a->raw_, &b->raw_, d, alpha, beta, &c->raw_);
    else if constexpr (std::is_same_v<in_type_, f32c_t> && simd)
        nk_each_blend_f32c(&a->raw_, &b->raw_, d, alpha, beta, &c->raw_);
    else if constexpr (std::is_same_v<in_type_, f64c_t> && simd)
        nk_each_blend_f64c(&a->raw_, &b->raw_, d, alpha, beta, &c->raw_);
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
template <numeric_dtype in_type_, numeric_dtype precision_type_ = in_type_, allow_simd_t allow_simd_ = prefer_simd_k>
void fma(in_type_ const *a, in_type_ const *b, std::size_t d, in_type_ const *c,
         typename in_type_::scale_t const *alpha, typename in_type_::scale_t const *beta, in_type_ *out) noexcept {
    constexpr bool simd = allow_simd_ == prefer_simd_k && std::is_same_v<precision_type_, in_type_>;

    if constexpr (std::is_same_v<in_type_, f64_t> && simd)
        nk_each_fma_f64(&a->raw_, &b->raw_, &c->raw_, d, alpha, beta, &out->raw_);
    else if constexpr (std::is_same_v<in_type_, f32_t> && simd)
        nk_each_fma_f32(&a->raw_, &b->raw_, &c->raw_, d, alpha, beta, &out->raw_);
    else if constexpr (std::is_same_v<in_type_, f16_t> && simd)
        nk_each_fma_f16(&a->raw_, &b->raw_, &c->raw_, d, alpha, beta, &out->raw_);
    else if constexpr (std::is_same_v<in_type_, bf16_t> && simd)
        nk_each_fma_bf16(&a->raw_, &b->raw_, &c->raw_, d, alpha, beta, &out->raw_);
    else if constexpr (std::is_same_v<in_type_, i8_t> && simd)
        nk_each_fma_i8(&a->raw_, &b->raw_, &c->raw_, d, alpha, beta, &out->raw_);
    else if constexpr (std::is_same_v<in_type_, u8_t> && simd)
        nk_each_fma_u8(&a->raw_, &b->raw_, &c->raw_, d, alpha, beta, &out->raw_);
    else if constexpr (std::is_same_v<in_type_, i16_t> && simd)
        nk_each_fma_i16(&a->raw_, &b->raw_, &c->raw_, d, alpha, beta, &out->raw_);
    else if constexpr (std::is_same_v<in_type_, u16_t> && simd)
        nk_each_fma_u16(&a->raw_, &b->raw_, &c->raw_, d, alpha, beta, &out->raw_);
    else if constexpr (std::is_same_v<in_type_, i32_t> && simd)
        nk_each_fma_i32(&a->raw_, &b->raw_, &c->raw_, d, alpha, beta, &out->raw_);
    else if constexpr (std::is_same_v<in_type_, u32_t> && simd)
        nk_each_fma_u32(&a->raw_, &b->raw_, &c->raw_, d, alpha, beta, &out->raw_);
    else if constexpr (std::is_same_v<in_type_, i64_t> && simd)
        nk_each_fma_i64(&a->raw_, &b->raw_, &c->raw_, d, alpha, beta, &out->raw_);
    else if constexpr (std::is_same_v<in_type_, u64_t> && simd)
        nk_each_fma_u64(&a->raw_, &b->raw_, &c->raw_, d, alpha, beta, &out->raw_);
    else if constexpr (std::is_same_v<in_type_, f32c_t> && simd)
        nk_each_fma_f32c(&a->raw_, &b->raw_, &c->raw_, d, alpha, beta, &out->raw_);
    else if constexpr (std::is_same_v<in_type_, f64c_t> && simd)
        nk_each_fma_f64c(&a->raw_, &b->raw_, &c->raw_, d, alpha, beta, &out->raw_);
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

#include "numkong/tensor.hpp"

namespace ashvardanian::numkong {

#pragma region Tensor Elementwise

/** @brief Scale: output[i] = α × input[i] + β. */
template <numeric_dtype value_type_, std::size_t max_rank_ = 8>
bool scale(tensor_view<value_type_, max_rank_> input, typename value_type_::scale_t alpha,
           typename value_type_::scale_t beta, tensor_span<value_type_, max_rank_> output) noexcept {
    return elementwise_into_<value_type_, max_rank_>(
        input, output, [&](tensor_view<value_type_, max_rank_> in, tensor_span<value_type_, max_rank_> out) {
            numkong::scale<value_type_>(in.data(), in.extent(0), &alpha, &beta, out.data());
        });
}

/** @brief Allocating scale: result[i] = α × input[i] + β. */
template <numeric_dtype value_type_, std::size_t max_rank_ = 8,
          typename allocator_type_ = aligned_allocator<value_type_>>
tensor<value_type_, allocator_type_, max_rank_> try_scale(tensor_view<value_type_, max_rank_> input,
                                                          typename value_type_::scale_t alpha,
                                                          typename value_type_::scale_t beta) noexcept {
    using out_tensor_t = tensor<value_type_, allocator_type_, max_rank_>;
    if (input.empty()) return out_tensor_t {};
    auto &input_shape = input.shape();
    auto result = out_tensor_t::try_empty(input_shape.extents, input_shape.rank);
    if (result.empty()) return result;
    if (!scale<value_type_, max_rank_>(input, alpha, beta, result.span())) return out_tensor_t {};
    return result;
}

/** @brief Blend: output[i] = α × lhs[i] + β × rhs[i]. */
template <numeric_dtype value_type_, std::size_t max_rank_ = 8>
bool blend(tensor_view<value_type_, max_rank_> lhs, tensor_view<value_type_, max_rank_> rhs,
           typename value_type_::scale_t alpha, typename value_type_::scale_t beta,
           tensor_span<value_type_, max_rank_> output) noexcept {
    return elementwise_into_<value_type_, max_rank_>(
        lhs, rhs, output,
        [&](tensor_view<value_type_, max_rank_> l, tensor_view<value_type_, max_rank_> r,
            tensor_span<value_type_, max_rank_> out) {
            numkong::blend<value_type_>(l.data(), r.data(), l.extent(0), &alpha, &beta, out.data());
        });
}

/** @brief Allocating blend: result[i] = α × lhs[i] + β × rhs[i]. */
template <numeric_dtype value_type_, std::size_t max_rank_ = 8,
          typename allocator_type_ = aligned_allocator<value_type_>>
tensor<value_type_, allocator_type_, max_rank_> try_blend(tensor_view<value_type_, max_rank_> lhs,
                                                          tensor_view<value_type_, max_rank_> rhs,
                                                          typename value_type_::scale_t alpha,
                                                          typename value_type_::scale_t beta) noexcept {
    using out_tensor_t = tensor<value_type_, allocator_type_, max_rank_>;
    if (!shapes_match_(lhs, rhs) || lhs.empty()) return out_tensor_t {};
    auto &input_shape = lhs.shape();
    auto result = out_tensor_t::try_empty(input_shape.extents, input_shape.rank);
    if (result.empty()) return result;
    if (!blend<value_type_, max_rank_>(lhs, rhs, alpha, beta, result.span())) return out_tensor_t {};
    return result;
}

/** @brief FMA: output[i] = α × lhs[i] × rhs[i] + β × addend[i]. */
template <numeric_dtype value_type_, std::size_t max_rank_ = 8>
bool fma(tensor_view<value_type_, max_rank_> lhs, tensor_view<value_type_, max_rank_> rhs,
         tensor_view<value_type_, max_rank_> addend, typename value_type_::scale_t alpha,
         typename value_type_::scale_t beta, tensor_span<value_type_, max_rank_> output) noexcept {
    return elementwise_into_<value_type_, max_rank_>(
        lhs, rhs, addend, output,
        [&](tensor_view<value_type_, max_rank_> a, tensor_view<value_type_, max_rank_> b,
            tensor_view<value_type_, max_rank_> c, tensor_span<value_type_, max_rank_> out) {
            numkong::fma<value_type_>(a.data(), b.data(), a.extent(0), c.data(), &alpha, &beta, out.data());
        });
}

/** @brief Allocating FMA: result[i] = α × lhs[i] × rhs[i] + β × addend[i]. */
template <numeric_dtype value_type_, std::size_t max_rank_ = 8,
          typename allocator_type_ = aligned_allocator<value_type_>>
tensor<value_type_, allocator_type_, max_rank_> try_fma(tensor_view<value_type_, max_rank_> lhs,
                                                        tensor_view<value_type_, max_rank_> rhs,
                                                        tensor_view<value_type_, max_rank_> addend,
                                                        typename value_type_::scale_t alpha,
                                                        typename value_type_::scale_t beta) noexcept {
    using out_tensor_t = tensor<value_type_, allocator_type_, max_rank_>;
    if (!shapes_match_(lhs, rhs) || !shapes_match_(lhs, addend) || lhs.empty()) return out_tensor_t {};
    auto &input_shape = lhs.shape();
    auto result = out_tensor_t::try_empty(input_shape.extents, input_shape.rank);
    if (result.empty()) return result;
    if (!fma<value_type_, max_rank_>(lhs, rhs, addend, alpha, beta, result.span())) return out_tensor_t {};
    return result;
}

/** @brief Elementwise addition: output[i] = lhs[i] + rhs[i]. */
template <numeric_dtype value_type_, std::size_t max_rank_ = 8>
bool add(tensor_view<value_type_, max_rank_> lhs, tensor_view<value_type_, max_rank_> rhs,
         tensor_span<value_type_, max_rank_> output) noexcept {
    return elementwise_into_<value_type_, max_rank_>(
        lhs, rhs, output,
        [](tensor_view<value_type_, max_rank_> l, tensor_view<value_type_, max_rank_> r,
           tensor_span<value_type_, max_rank_> out) {
            numkong::sum<value_type_>(l.data(), r.data(), l.extent(0), out.data());
        });
}

/** @brief Allocating elementwise add: result = lhs + rhs. */
template <numeric_dtype value_type_, std::size_t max_rank_ = 8,
          typename allocator_type_ = aligned_allocator<value_type_>>
tensor<value_type_, allocator_type_, max_rank_> try_add(tensor_view<value_type_, max_rank_> lhs,
                                                        tensor_view<value_type_, max_rank_> rhs) noexcept {
    using out_tensor_t = tensor<value_type_, allocator_type_, max_rank_>;
    if (!shapes_match_(lhs, rhs) || lhs.empty()) return out_tensor_t {};
    auto &input_shape = lhs.shape();
    auto result = out_tensor_t::try_empty(input_shape.extents, input_shape.rank);
    if (result.empty()) return result;
    if (!add<value_type_, max_rank_>(lhs, rhs, result.span())) return out_tensor_t {};
    return result;
}

/** @brief Elementwise add scalar: output[i] = input[i] + scalar. */
template <numeric_dtype value_type_, std::size_t max_rank_ = 8>
bool add(tensor_view<value_type_, max_rank_> input, typename value_type_::scale_t scalar,
         tensor_span<value_type_, max_rank_> output) noexcept {
    typename value_type_::scale_t one {1};
    return scale<value_type_, max_rank_>(input, one, scalar, output);
}

/** @brief Allocating add scalar. */
template <numeric_dtype value_type_, std::size_t max_rank_ = 8,
          typename allocator_type_ = aligned_allocator<value_type_>>
tensor<value_type_, allocator_type_, max_rank_> try_add(tensor_view<value_type_, max_rank_> input,
                                                        typename value_type_::scale_t scalar) noexcept {
    using out_tensor_t = tensor<value_type_, allocator_type_, max_rank_>;
    if (input.empty()) return out_tensor_t {};
    auto &input_shape = input.shape();
    auto result = out_tensor_t::try_empty(input_shape.extents, input_shape.rank);
    if (result.empty()) return result;
    if (!add<value_type_, max_rank_>(input, scalar, result.span())) return out_tensor_t {};
    return result;
}

/** @brief Elementwise subtraction: output[i] = lhs[i] − rhs[i]. */
template <numeric_dtype value_type_, std::size_t max_rank_ = 8>
bool sub(tensor_view<value_type_, max_rank_> lhs, tensor_view<value_type_, max_rank_> rhs,
         tensor_span<value_type_, max_rank_> output) noexcept {
    typename value_type_::scale_t alpha {1}, beta {-1};
    return blend<value_type_, max_rank_>(lhs, rhs, alpha, beta, output);
}

/** @brief Allocating elementwise sub. */
template <numeric_dtype value_type_, std::size_t max_rank_ = 8,
          typename allocator_type_ = aligned_allocator<value_type_>>
tensor<value_type_, allocator_type_, max_rank_> try_sub(tensor_view<value_type_, max_rank_> lhs,
                                                        tensor_view<value_type_, max_rank_> rhs) noexcept {
    using out_tensor_t = tensor<value_type_, allocator_type_, max_rank_>;
    if (!shapes_match_(lhs, rhs) || lhs.empty()) return out_tensor_t {};
    auto &input_shape = lhs.shape();
    auto result = out_tensor_t::try_empty(input_shape.extents, input_shape.rank);
    if (result.empty()) return result;
    if (!sub<value_type_, max_rank_>(lhs, rhs, result.span())) return out_tensor_t {};
    return result;
}

/** @brief Elementwise sub scalar: output[i] = input[i] − scalar. */
template <numeric_dtype value_type_, std::size_t max_rank_ = 8>
bool sub(tensor_view<value_type_, max_rank_> input, typename value_type_::scale_t scalar,
         tensor_span<value_type_, max_rank_> output) noexcept {
    typename value_type_::scale_t one {1};
    typename value_type_::scale_t neg_scalar = -scalar;
    return scale<value_type_, max_rank_>(input, one, neg_scalar, output);
}

/** @brief Allocating sub scalar. */
template <numeric_dtype value_type_, std::size_t max_rank_ = 8,
          typename allocator_type_ = aligned_allocator<value_type_>>
tensor<value_type_, allocator_type_, max_rank_> try_sub(tensor_view<value_type_, max_rank_> input,
                                                        typename value_type_::scale_t scalar) noexcept {
    using out_tensor_t = tensor<value_type_, allocator_type_, max_rank_>;
    if (input.empty()) return out_tensor_t {};
    auto &input_shape = input.shape();
    auto result = out_tensor_t::try_empty(input_shape.extents, input_shape.rank);
    if (result.empty()) return result;
    if (!sub<value_type_, max_rank_>(input, scalar, result.span())) return out_tensor_t {};
    return result;
}

/** @brief Elementwise multiplication: output[i] = lhs[i] × rhs[i]. */
template <numeric_dtype value_type_, std::size_t max_rank_ = 8>
bool mul(tensor_view<value_type_, max_rank_> lhs, tensor_view<value_type_, max_rank_> rhs,
         tensor_span<value_type_, max_rank_> output) noexcept {
    return elementwise_into_<value_type_, max_rank_>(
        lhs, rhs, output,
        [](tensor_view<value_type_, max_rank_> l, tensor_view<value_type_, max_rank_> r,
           tensor_span<value_type_, max_rank_> out) {
            typename value_type_::scale_t alpha {1}, beta {0};
            numkong::fma<value_type_>(l.data(), r.data(), l.extent(0), out.data(), &alpha, &beta, out.data());
        });
}

/** @brief Allocating elementwise multiply. */
template <numeric_dtype value_type_, std::size_t max_rank_ = 8,
          typename allocator_type_ = aligned_allocator<value_type_>>
tensor<value_type_, allocator_type_, max_rank_> try_mul(tensor_view<value_type_, max_rank_> lhs,
                                                        tensor_view<value_type_, max_rank_> rhs) noexcept {
    using out_tensor_t = tensor<value_type_, allocator_type_, max_rank_>;
    if (!shapes_match_(lhs, rhs) || lhs.empty()) return out_tensor_t {};
    auto &input_shape = lhs.shape();
    auto result = out_tensor_t::try_zeros(input_shape.extents, input_shape.rank);
    if (result.empty()) return result;
    if (!mul<value_type_, max_rank_>(lhs, rhs, result.span())) return out_tensor_t {};
    return result;
}

/** @brief Elementwise multiply by scalar: output[i] = input[i] × scalar. */
template <numeric_dtype value_type_, std::size_t max_rank_ = 8>
bool mul(tensor_view<value_type_, max_rank_> input, typename value_type_::scale_t scalar,
         tensor_span<value_type_, max_rank_> output) noexcept {
    typename value_type_::scale_t zero {0};
    return scale<value_type_, max_rank_>(input, scalar, zero, output);
}

/** @brief Allocating multiply by scalar. */
template <numeric_dtype value_type_, std::size_t max_rank_ = 8,
          typename allocator_type_ = aligned_allocator<value_type_>>
tensor<value_type_, allocator_type_, max_rank_> try_mul(tensor_view<value_type_, max_rank_> input,
                                                        typename value_type_::scale_t scalar) noexcept {
    using out_tensor_t = tensor<value_type_, allocator_type_, max_rank_>;
    if (input.empty()) return out_tensor_t {};
    auto &input_shape = input.shape();
    auto result = out_tensor_t::try_empty(input_shape.extents, input_shape.rank);
    if (result.empty()) return result;
    if (!mul<value_type_, max_rank_>(input, scalar, result.span())) return out_tensor_t {};
    return result;
}

#pragma endregion Tensor Elementwise

} // namespace ashvardanian::numkong

#endif // NK_EACH_HPP

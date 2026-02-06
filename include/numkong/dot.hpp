/**
 *  @brief C++ bindings for dot-product kernels: ⟨a,b⟩ = Σ aᵢ × bᵢ
 *  @file include/numkong/dot.hpp
 *  @author Ash Vardanian
 *  @date February 5, 2026
 */
#ifndef NK_DOT_HPP
#define NK_DOT_HPP

#include <cstdint>
#include <type_traits>

#include "numkong/dot.h"

#include "numkong/types.hpp"

namespace ashvardanian::numkong {

template <typename in_type_, typename result_type_ = typename in_type_::dot_result_t,
          allow_simd_t allow_simd_ = prefer_simd_k>
void dot(in_type_ const *a, in_type_ const *b, std::size_t d, result_type_ *r) noexcept {
    constexpr bool simd = allow_simd_ == prefer_simd_k && std::is_same_v<result_type_, typename in_type_::dot_result_t>;

    if constexpr (std::is_same_v<in_type_, f32_t> && simd) nk_dot_f32(&a->raw_, &b->raw_, d, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, f64_t> && simd) nk_dot_f64(&a->raw_, &b->raw_, d, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, f16_t> && simd) nk_dot_f16(&a->raw_, &b->raw_, d, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, bf16_t> && simd) nk_dot_bf16(&a->raw_, &b->raw_, d, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, e4m3_t> && simd) nk_dot_e4m3(&a->raw_, &b->raw_, d, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, e5m2_t> && simd) nk_dot_e5m2(&a->raw_, &b->raw_, d, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, e2m3_t> && simd) nk_dot_e2m3(&a->raw_, &b->raw_, d, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, e3m2_t> && simd) nk_dot_e3m2(&a->raw_, &b->raw_, d, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, i8_t> && simd) nk_dot_i8(&a->raw_, &b->raw_, d, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, u8_t> && simd) nk_dot_u8(&a->raw_, &b->raw_, d, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, f32c_t> && simd) nk_dot_f32c(&a->raw_, &b->raw_, d, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, f64c_t> && simd) nk_dot_f64c(&a->raw_, &b->raw_, d, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, i4x2_t> && simd) nk_dot_i4(&a->raw_, &b->raw_, d, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, u4x2_t> && simd) nk_dot_u4(&a->raw_, &b->raw_, d, &r->raw_);
    else {
        result_type_ sum {};
        std::size_t n = divide_round_up(d, dimensions_per_value<in_type_>());
        for (std::size_t i = 0; i < n; i++) sum = fused_multiply_add(sum, a[i], b[i]);
        *r = sum;
    }
}

template <typename in_type_, typename result_type_ = typename in_type_::dot_result_t,
          allow_simd_t allow_simd_ = prefer_simd_k>
void vdot(in_type_ const *a, in_type_ const *b, std::size_t d, result_type_ *r) noexcept {
    constexpr bool simd = allow_simd_ == prefer_simd_k && std::is_same_v<result_type_, typename in_type_::dot_result_t>;

    if constexpr (std::is_same_v<in_type_, f32c_t> && simd) nk_vdot_f32c(&a->raw_, &b->raw_, d, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, f64c_t> && simd) nk_vdot_f64c(&a->raw_, &b->raw_, d, &r->raw_);
    else {
        result_type_ sum {};
        for (std::size_t i = 0; i < d; i++) sum = fused_conjugate_multiply_add(sum, a[i], b[i]);
        *r = sum;
    }
}

} // namespace ashvardanian::numkong

#endif // NK_DOT_HPP

/**
 *  @brief NumKong SDK for C++23 and newer.
 *  @file include/numkong.hpp
 *  @author Ash Vardanian
 *  @date January 7, 2026
 *
 *  C doesn't have a strong type system or composable infrastructure for complex kernels
 *  and datastructures like the C++ templates and Rust traits. Unlike C++, C also lacks
 *  function overloading, namespaces and templates, thus requiring verbose signatures and
 *  naming conventions, like:
 *
 *  @code{c}
 *  void nk_dot_f64(nk_f64_t const*, nk_f64_t const*, nk_size_t, nk_f64_t *);
 *  void nk_dot_f32(nk_f32_t const*, nk_f32_t const*, nk_size_t, nk_f32_t *);
 *  void nk_dot_f16(nk_f16_t const*, nk_f16_t const*, nk_size_t, nk_f32_t *);
 *  void nk_dot_bf16(nk_bf16_t const*, nk_bf16_t const*, nk_size_t, nk_f32_t *);
 *  void nk_dot_e4m3(nk_e4m3_t const*, nk_e4m3_t const*, nk_size_t, nk_f32_t *);
 *  void nk_dot_e4m2(nk_e4m2_t const*, nk_e4m3_t const*, nk_size_t, nk_f32_t *);
 *  @endcode
 *
 *  As opposed to C++:
 *
 *  @code{cpp}
 *  namespace ashvardanian::numkong {
 *      template <typename input_type_, typename result_type_>
 *      void dot(input_type_ const*, input_type_ const*, size_t, result_type_ *);
 *  }
 *
 *  In HPC implementations, where pretty much every kernel and every datatype uses different
 *  Assembly instructions on different CPU generations/models, those higher-level abstractions
 *  aren't always productive for the primary implementation, but they can still be handy as
 *  a higher-level API for NumKong. They are also used for algorithm verification in no-SIMD
 *  mode, upcasting to much larger number types like `f118_t`.
 */

#ifndef NK_NUMKONG_HPP
#define NK_NUMKONG_HPP

#include "numkong/numkong.h"
#include "numkong/types.hpp"

#include <bit>         // `std::bit_cast`
#include <cmath>       // `std::sqrt`
#include <concepts>    // `std::same_as`
#include <cstdint>     // `std::uint32_t`
#include <limits>      // `std::numeric_limits`
#include <type_traits> // `std::is_same_v`
#include <utility>     // `std::swap`

namespace ashvardanian::numkong {

/** @brief FMA helper template for baseline dot-product implementations. */
template <typename in_type_, typename accumulator_type_>
    requires(in_type_::bits() >= NK_BITS_PER_BYTE)
inline accumulator_type_ fused_multiply_add(accumulator_type_ acc, in_type_ a, in_type_ b) noexcept {
    return acc + static_cast<accumulator_type_>(a) * static_cast<accumulator_type_>(b);
}

/** @brief FMA helper template for baseline conjugate complex dot-product implementations. */
template <typename in_type_, typename accumulator_type_>
    requires(in_type_::bits() >= NK_BITS_PER_BYTE)
inline accumulator_type_ fused_conjugate_multiply_add(accumulator_type_ acc, in_type_ a, in_type_ b) noexcept {
    return acc + static_cast<accumulator_type_>(a.conj()) * static_cast<accumulator_type_>(b);
}

/** @brief Fused addition of squared differences for baseline L2 implementations. */
template <typename in_type_, typename accumulator_type_>
    requires(in_type_::bits() >= NK_BITS_PER_BYTE)
constexpr accumulator_type_ fused_difference_squared_add(accumulator_type_ acc, in_type_ a, in_type_ b) noexcept {
    auto d = static_cast<accumulator_type_>(a) - static_cast<accumulator_type_>(b);
    return acc + d * d;
}

/** @brief FMA specialization for i4x2_t (signed 4-bit packed pairs). */
template <typename accumulator_type_>
inline accumulator_type_ fused_multiply_add(accumulator_type_ acc, i4x2_t a, i4x2_t b) noexcept {
    return acc + accumulator_type_(nk_i32_t(a.low()) * nk_i32_t(b.low()) + nk_i32_t(a.high()) * nk_i32_t(b.high()));
}

/** @brief FMA specialization for u4x2_t (unsigned 4-bit packed pairs). */
template <typename accumulator_type_>
inline accumulator_type_ fused_multiply_add(accumulator_type_ acc, u4x2_t a, u4x2_t b) noexcept {
    return acc + accumulator_type_(nk_u32_t(a.low()) * nk_u32_t(b.low()) + nk_u32_t(a.high()) * nk_u32_t(b.high()));
}

/** @brief FMA specialization for u1x8_t (8 packed bits). Counts matching set bits (popcount of AND). */
template <typename accumulator_type_>
inline accumulator_type_ fused_multiply_add(accumulator_type_ acc, u1x8_t a, u1x8_t b) noexcept {
    return acc + accumulator_type_(std::popcount(static_cast<unsigned>(a.raw() & b.raw())));
}

/** @brief Squared difference specialization for i4x2_t (signed 4-bit packed pairs). */
template <typename accumulator_type_>
constexpr accumulator_type_ fused_difference_squared_add(accumulator_type_ acc, i4x2_t a, i4x2_t b) noexcept {
    nk_i32_t low_difference = nk_i32_t(a.low()) - nk_i32_t(b.low());
    nk_i32_t high_difference = nk_i32_t(a.high()) - nk_i32_t(b.high());
    return acc + accumulator_type_(low_difference * low_difference + high_difference * high_difference);
}

/** @brief Squared difference specialization for u4x2_t (unsigned 4-bit packed pairs). */
template <typename accumulator_type_>
constexpr accumulator_type_ fused_difference_squared_add(accumulator_type_ acc, u4x2_t a, u4x2_t b) noexcept {
    nk_i32_t low_difference = nk_i32_t(a.low()) - nk_i32_t(b.low());
    nk_i32_t high_difference = nk_i32_t(a.high()) - nk_i32_t(b.high());
    return acc + accumulator_type_(low_difference * low_difference + high_difference * high_difference);
}

constexpr f118_t operator+(double a, f118_t b) noexcept { return f118_t(a) + b; }
constexpr f118_t operator-(double a, f118_t b) noexcept { return f118_t(a) - b; }
constexpr f118_t operator*(double a, f118_t b) noexcept { return f118_t(a) * b; }
constexpr f118_t operator/(double a, f118_t b) noexcept { return f118_t(a) / b; }

constexpr bool operator==(double a, f118_t b) noexcept { return f118_t(a) == b; }
constexpr bool operator!=(double a, f118_t b) noexcept { return f118_t(a) != b; }
constexpr bool operator<(double a, f118_t b) noexcept { return f118_t(a) < b; }
constexpr bool operator>(double a, f118_t b) noexcept { return f118_t(a) > b; }
constexpr bool operator<=(double a, f118_t b) noexcept { return f118_t(a) <= b; }
constexpr bool operator>=(double a, f118_t b) noexcept { return f118_t(a) >= b; }

#pragma region Dot Kernels

/**
 *  @brief Dot product with configurable precision.
 *  @tparam in_type_ Input vector element type
 *  @tparam result_type_ Accumulator type (default: in_type_::dot_result_t)
 *  @tparam allow_simd_ Enable SIMD kernel dispatch when true
 */
template <typename in_type_, typename result_type_ = typename in_type_::dot_result_t, bool allow_simd_ = true>
void dot(in_type_ const *a, in_type_ const *b, std::size_t n, result_type_ *r) noexcept {
    constexpr bool simd = allow_simd_ && std::is_same_v<result_type_, typename in_type_::dot_result_t>;

    if constexpr (std::is_same_v<in_type_, f32_t> && simd) nk_dot_f32(&a->raw_, &b->raw_, n, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, f64_t> && simd) nk_dot_f64(&a->raw_, &b->raw_, n, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, f16_t> && simd) nk_dot_f16(&a->raw_, &b->raw_, n, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, bf16_t> && simd) nk_dot_bf16(&a->raw_, &b->raw_, n, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, e4m3_t> && simd) nk_dot_e4m3(&a->raw_, &b->raw_, n, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, e5m2_t> && simd) nk_dot_e5m2(&a->raw_, &b->raw_, n, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, i8_t> && simd) nk_dot_i8(&a->raw_, &b->raw_, n, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, u8_t> && simd) nk_dot_u8(&a->raw_, &b->raw_, n, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, f32c_t> && simd) nk_dot_f32c(&a->raw_, &b->raw_, n, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, f64c_t> && simd) nk_dot_f64c(&a->raw_, &b->raw_, n, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, i4x2_t> && simd) nk_dot_i4(&a->raw_, &b->raw_, n, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, u4x2_t> && simd) nk_dot_u4(&a->raw_, &b->raw_, n, &r->raw_);
    // Scalar fallback
    else {
        result_type_ sum {};
        for (std::size_t i = 0; i < n; i++) sum = fused_multiply_add(sum, a[i], b[i]);
        *r = sum;
    }
}

/**
 *  @brief Conjugate dot product (vdot): sum(conj(a[i]) * b[i])
 *  @tparam in_type_ Complex input type (f32c_t, f64c_t, f16c_t, bf16c_t)
 *  @tparam result_type_ Accumulator type (default: in_type_::dot_result_t)
 *  @tparam allow_simd_ Enable SIMD kernel dispatch when true
 */
template <typename in_type_, typename result_type_ = typename in_type_::dot_result_t, bool allow_simd_ = true>
void vdot(in_type_ const *a, in_type_ const *b, std::size_t n, result_type_ *r) noexcept {
    constexpr bool simd = allow_simd_ && std::is_same_v<result_type_, typename in_type_::dot_result_t>;

    if constexpr (std::is_same_v<in_type_, f32c_t> && simd) nk_vdot_f32c(&a->raw_, &b->raw_, n, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, f64c_t> && simd) nk_vdot_f64c(&a->raw_, &b->raw_, n, &r->raw_);
    else {
        result_type_ sum {};
        for (std::size_t i = 0; i < n; i++) sum = fused_conjugate_multiply_add(sum, a[i], b[i]);
        *r = sum;
    }
}

#pragma endregion // Dot Kernels

#pragma region Spatial Kernels

/**
 *  @brief L2 (Euclidean) distance.
 *  @tparam in_type_ Input vector element type
 *  @tparam result_type_ Accumulator type (default: in_type_::l2sq_result_t)
 *  @tparam allow_simd_ Enable SIMD kernel dispatch when true
 */
template <typename in_type_, typename result_type_ = typename in_type_::l2sq_result_t, bool allow_simd_ = true>
void l2(in_type_ const *a, in_type_ const *b, std::size_t n, result_type_ *r) noexcept {
    constexpr bool simd = allow_simd_ && std::is_same_v<result_type_, typename in_type_::l2sq_result_t>;

    if constexpr (std::is_same_v<in_type_, f64_t> && simd) nk_l2_f64(&a->raw_, &b->raw_, n, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, f32_t> && simd) nk_l2_f32(&a->raw_, &b->raw_, n, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, f16_t> && simd) nk_l2_f16(&a->raw_, &b->raw_, n, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, bf16_t> && simd) nk_l2_bf16(&a->raw_, &b->raw_, n, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, e4m3_t> && simd) nk_l2_e4m3(&a->raw_, &b->raw_, n, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, e5m2_t> && simd) nk_l2_e5m2(&a->raw_, &b->raw_, n, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, i8_t> && simd) nk_l2_i8(&a->raw_, &b->raw_, n, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, u8_t> && simd) nk_l2_u8(&a->raw_, &b->raw_, n, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, i4x2_t> && simd) nk_l2_i4(&a->raw_, &b->raw_, n, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, u4x2_t> && simd) nk_l2_u4(&a->raw_, &b->raw_, n, &r->raw_);
    // Scalar fallback
    else {
        result_type_ sum {};
        for (std::size_t i = 0; i < n; i++) sum = fused_difference_squared_add(sum, a[i], b[i]);
        *r = sum.sqrt();
    }
}

/**
 *  @brief Squared L2 (Euclidean) distance.
 *  @tparam in_type_ Input vector element type
 *  @tparam result_type_ Accumulator type (default: in_type_::l2sq_result_t)
 *  @tparam allow_simd_ Enable SIMD kernel dispatch when true
 */
template <typename in_type_, typename result_type_ = typename in_type_::l2sq_result_t, bool allow_simd_ = true>
void l2sq(in_type_ const *a, in_type_ const *b, std::size_t n, result_type_ *r) noexcept {
    constexpr bool simd = allow_simd_ && std::is_same_v<result_type_, typename in_type_::l2sq_result_t>;

    if constexpr (std::is_same_v<in_type_, f64_t> && simd) nk_l2sq_f64(&a->raw_, &b->raw_, n, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, f32_t> && simd) nk_l2sq_f32(&a->raw_, &b->raw_, n, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, f16_t> && simd) nk_l2sq_f16(&a->raw_, &b->raw_, n, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, bf16_t> && simd) nk_l2sq_bf16(&a->raw_, &b->raw_, n, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, e4m3_t> && simd) nk_l2sq_e4m3(&a->raw_, &b->raw_, n, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, e5m2_t> && simd) nk_l2sq_e5m2(&a->raw_, &b->raw_, n, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, i8_t> && simd) nk_l2sq_i8(&a->raw_, &b->raw_, n, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, u8_t> && simd) nk_l2sq_u8(&a->raw_, &b->raw_, n, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, i4x2_t> && simd) nk_l2sq_i4(&a->raw_, &b->raw_, n, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, u4x2_t> && simd) nk_l2sq_u4(&a->raw_, &b->raw_, n, &r->raw_);
    // Scalar fallback
    else {
        result_type_ sum {};
        for (std::size_t i = 0; i < n; i++) sum = fused_difference_squared_add(sum, a[i], b[i]);
        *r = sum;
    }
}

/**
 *  @brief Angular similarity (cosine): dot(a,b) / (|a| * |b|)
 *  @tparam in_type_ Input vector element type
 *  @tparam result_type_ Accumulator type (default: in_type_::angular_result_t)
 *  @tparam allow_simd_ Enable SIMD kernel dispatch when true
 */
template <typename in_type_, typename result_type_ = typename in_type_::angular_result_t, bool allow_simd_ = true>
void angular(in_type_ const *a, in_type_ const *b, std::size_t n, result_type_ *r) noexcept {
    constexpr bool simd = allow_simd_ && std::is_same_v<result_type_, typename in_type_::angular_result_t>;

    if constexpr (std::is_same_v<in_type_, f64_t> && simd) nk_angular_f64(&a->raw_, &b->raw_, n, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, f32_t> && simd) nk_angular_f32(&a->raw_, &b->raw_, n, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, f16_t> && simd) nk_angular_f16(&a->raw_, &b->raw_, n, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, bf16_t> && simd) nk_angular_bf16(&a->raw_, &b->raw_, n, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, e4m3_t> && simd) nk_angular_e4m3(&a->raw_, &b->raw_, n, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, e5m2_t> && simd) nk_angular_e5m2(&a->raw_, &b->raw_, n, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, i8_t> && simd) nk_angular_i8(&a->raw_, &b->raw_, n, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, u8_t> && simd) nk_angular_u8(&a->raw_, &b->raw_, n, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, i4x2_t> && simd) nk_angular_i4(&a->raw_, &b->raw_, n, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, u4x2_t> && simd) nk_angular_u4(&a->raw_, &b->raw_, n, &r->raw_);
    // Scalar fallback
    else {
        result_type_ ab {}, aa {}, bb {};
        for (std::size_t i = 0; i < n; i++) {
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

#pragma endregion Spatial Kernels

#pragma region Probability Kernels

/**
 *  @brief Kullback-Leibler divergence: sum(p[i] * log(p[i] / q[i]))
 *  @tparam in_type_ Input distribution type (probability vectors)
 *  @tparam result_type_ Accumulator type (default: f118_t for validation)
 *  @tparam allow_simd_ Enable SIMD kernel dispatch when true
 */
template <typename in_type_, typename result_type_ = typename in_type_::kld_result_t, bool allow_simd_ = true>
void kld(in_type_ const *p, in_type_ const *q, std::size_t n, result_type_ *r) noexcept {
    constexpr bool simd = allow_simd_ && std::is_same_v<result_type_, typename in_type_::kld_result_t>;

    if constexpr (std::is_same_v<in_type_, f64_t> && simd) nk_kld_f64(&p->raw_, &q->raw_, n, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, f32_t> && simd) nk_kld_f32(&p->raw_, &q->raw_, n, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, f16_t> && simd) nk_kld_f16(&p->raw_, &q->raw_, n, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, bf16_t> && simd) nk_kld_bf16(&p->raw_, &q->raw_, n, &r->raw_);
    // Scalar fallback
    else {
        result_type_ sum {};
        for (std::size_t i = 0; i < n; i++) {
            result_type_ pi(p[i]), qi(q[i]);
            if (pi > result_type_(0)) sum = sum + pi * (pi / qi).log();
        }
        *r = sum;
    }
}

/**
 *  @brief Jensen-Shannon divergence: 0.5 * (KL(p||m) + KL(q||m)) where m = (p+q)/2
 *  @tparam in_type_ Input distribution type (probability vectors)
 *  @tparam result_type_ Accumulator type (default: f118_t for validation)
 *  @tparam allow_simd_ Enable SIMD kernel dispatch when true
 */
template <typename in_type_, typename result_type_ = typename in_type_::jsd_result_t, bool allow_simd_ = true>
void jsd(in_type_ const *p, in_type_ const *q, std::size_t n, result_type_ *r) noexcept {
    constexpr bool simd = allow_simd_ && std::is_same_v<result_type_, typename in_type_::jsd_result_t>;

    if constexpr (std::is_same_v<in_type_, f64_t> && simd) nk_jsd_f64(&p->raw_, &q->raw_, n, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, f32_t> && simd) nk_jsd_f32(&p->raw_, &q->raw_, n, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, f16_t> && simd) nk_jsd_f16(&p->raw_, &q->raw_, n, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, bf16_t> && simd) nk_jsd_bf16(&p->raw_, &q->raw_, n, &r->raw_);
    // Scalar fallback
    else {
        result_type_ sum {};
        result_type_ half(0.5);
        for (std::size_t i = 0; i < n; i++) {
            result_type_ pi(p[i]), qi(q[i]);
            result_type_ mi = half * (pi + qi);
            if (pi > result_type_(0)) sum = sum + pi * (pi / mi).log();
            if (qi > result_type_(0)) sum = sum + qi * (qi / mi).log();
        }
        // JSD distance = sqrt(divergence / 2), clamped to non-negative
        result_type_ divergence = half * sum;
        *r = divergence > result_type_(0) ? divergence.sqrt() : result_type_(0);
    }
}

#pragma endregion Probability Kernels

#pragma region Elementwise Kernels

/**
 *  @brief Elementwise sum: c[i] = a[i] + b[i]
 *  @tparam in_type_ Element type
 *  @tparam allow_simd_ Enable SIMD dispatch
 */
template <typename in_type_, bool allow_simd_ = true>
void sum(in_type_ const *a, in_type_ const *b, std::size_t n, in_type_ *c) noexcept {
    constexpr bool simd = allow_simd_;

    if constexpr (std::is_same_v<in_type_, f64_t> && simd) nk_sum_f64(&a->raw_, &b->raw_, n, &c->raw_);
    else if constexpr (std::is_same_v<in_type_, f32_t> && simd) nk_sum_f32(&a->raw_, &b->raw_, n, &c->raw_);
    else if constexpr (std::is_same_v<in_type_, f16_t> && simd) nk_sum_f16(&a->raw_, &b->raw_, n, &c->raw_);
    else if constexpr (std::is_same_v<in_type_, bf16_t> && simd) nk_sum_bf16(&a->raw_, &b->raw_, n, &c->raw_);
    else if constexpr (std::is_same_v<in_type_, i8_t> && simd) nk_sum_i8(&a->raw_, &b->raw_, n, &c->raw_);
    else if constexpr (std::is_same_v<in_type_, u8_t> && simd) nk_sum_u8(&a->raw_, &b->raw_, n, &c->raw_);
    else if constexpr (std::is_same_v<in_type_, i16_t> && simd) nk_sum_i16(&a->raw_, &b->raw_, n, &c->raw_);
    else if constexpr (std::is_same_v<in_type_, u16_t> && simd) nk_sum_u16(&a->raw_, &b->raw_, n, &c->raw_);
    else if constexpr (std::is_same_v<in_type_, i32_t> && simd) nk_sum_i32(&a->raw_, &b->raw_, n, &c->raw_);
    else if constexpr (std::is_same_v<in_type_, u32_t> && simd) nk_sum_u32(&a->raw_, &b->raw_, n, &c->raw_);
    else if constexpr (std::is_same_v<in_type_, i64_t> && simd) nk_sum_i64(&a->raw_, &b->raw_, n, &c->raw_);
    else if constexpr (std::is_same_v<in_type_, u64_t> && simd) nk_sum_u64(&a->raw_, &b->raw_, n, &c->raw_);
    // Scalar fallback
    else {
        for (std::size_t i = 0; i < n; i++) c[i] = a[i].saturating_add(b[i]);
    }
}

/**
 *  @brief Elementwise scale: c[i] = alpha * a[i] + beta
 *  @tparam in_type_ Element type
 *  @tparam scale_type_ Scalar type for alpha/beta
 *  @tparam allow_simd_ Enable SIMD dispatch
 */
template <typename in_type_, typename scale_type_ = typename in_type_::scale_t, bool allow_simd_ = true>
void scale(in_type_ const *a, std::size_t n, scale_type_ const *alpha, scale_type_ const *beta, in_type_ *c) noexcept {
    constexpr bool simd = allow_simd_ && std::is_same_v<scale_type_, typename in_type_::scale_t>;

    if constexpr (std::is_same_v<in_type_, f64_t> && simd) nk_scale_f64(&a->raw_, n, alpha, beta, &c->raw_);
    else if constexpr (std::is_same_v<in_type_, f32_t> && simd) nk_scale_f32(&a->raw_, n, alpha, beta, &c->raw_);
    else if constexpr (std::is_same_v<in_type_, f16_t> && simd) nk_scale_f16(&a->raw_, n, alpha, beta, &c->raw_);
    else if constexpr (std::is_same_v<in_type_, bf16_t> && simd) nk_scale_bf16(&a->raw_, n, alpha, beta, &c->raw_);
    else if constexpr (std::is_same_v<in_type_, i8_t> && simd) nk_scale_i8(&a->raw_, n, alpha, beta, &c->raw_);
    else if constexpr (std::is_same_v<in_type_, u8_t> && simd) nk_scale_u8(&a->raw_, n, alpha, beta, &c->raw_);
    else if constexpr (std::is_same_v<in_type_, i16_t> && simd) nk_scale_i16(&a->raw_, n, alpha, beta, &c->raw_);
    else if constexpr (std::is_same_v<in_type_, u16_t> && simd) nk_scale_u16(&a->raw_, n, alpha, beta, &c->raw_);
    else if constexpr (std::is_same_v<in_type_, i32_t> && simd) nk_scale_i32(&a->raw_, n, alpha, beta, &c->raw_);
    else if constexpr (std::is_same_v<in_type_, u32_t> && simd) nk_scale_u32(&a->raw_, n, alpha, beta, &c->raw_);
    else if constexpr (std::is_same_v<in_type_, i64_t> && simd) nk_scale_i64(&a->raw_, n, alpha, beta, &c->raw_);
    else if constexpr (std::is_same_v<in_type_, u64_t> && simd) nk_scale_u64(&a->raw_, n, alpha, beta, &c->raw_);
    // Scalar fallback
    else {
        for (std::size_t i = 0; i < n; i++) c[i] = in_type_(scale_type_(a[i]) * (*alpha) + (*beta));
    }
}

/**
 *  @brief Weighted sum: c[i] = alpha * a[i] + beta * b[i]
 *  @tparam in_type_ Element type
 *  @tparam scale_type_ Scalar type for alpha/beta
 *  @tparam allow_simd_ Enable SIMD dispatch
 */
template <typename in_type_, typename scale_type_ = typename in_type_::scale_t, bool allow_simd_ = true>
void wsum(in_type_ const *a, in_type_ const *b, std::size_t n, scale_type_ const *alpha, scale_type_ const *beta,
          in_type_ *c) noexcept {
    constexpr bool simd = allow_simd_ && std::is_same_v<scale_type_, typename in_type_::scale_t>;

    if constexpr (std::is_same_v<in_type_, f64_t> && simd) nk_wsum_f64(&a->raw_, &b->raw_, n, alpha, beta, &c->raw_);
    else if constexpr (std::is_same_v<in_type_, f32_t> && simd)
        nk_wsum_f32(&a->raw_, &b->raw_, n, alpha, beta, &c->raw_);
    else if constexpr (std::is_same_v<in_type_, f16_t> && simd)
        nk_wsum_f16(&a->raw_, &b->raw_, n, alpha, beta, &c->raw_);
    else if constexpr (std::is_same_v<in_type_, bf16_t> && simd)
        nk_wsum_bf16(&a->raw_, &b->raw_, n, alpha, beta, &c->raw_);
    else if constexpr (std::is_same_v<in_type_, i8_t> && simd) nk_wsum_i8(&a->raw_, &b->raw_, n, alpha, beta, &c->raw_);
    else if constexpr (std::is_same_v<in_type_, u8_t> && simd) nk_wsum_u8(&a->raw_, &b->raw_, n, alpha, beta, &c->raw_);
    else if constexpr (std::is_same_v<in_type_, i16_t> && simd)
        nk_wsum_i16(&a->raw_, &b->raw_, n, alpha, beta, &c->raw_);
    else if constexpr (std::is_same_v<in_type_, u16_t> && simd)
        nk_wsum_u16(&a->raw_, &b->raw_, n, alpha, beta, &c->raw_);
    else if constexpr (std::is_same_v<in_type_, i32_t> && simd)
        nk_wsum_i32(&a->raw_, &b->raw_, n, alpha, beta, &c->raw_);
    else if constexpr (std::is_same_v<in_type_, u32_t> && simd)
        nk_wsum_u32(&a->raw_, &b->raw_, n, alpha, beta, &c->raw_);
    else if constexpr (std::is_same_v<in_type_, i64_t> && simd)
        nk_wsum_i64(&a->raw_, &b->raw_, n, alpha, beta, &c->raw_);
    else if constexpr (std::is_same_v<in_type_, u64_t> && simd)
        nk_wsum_u64(&a->raw_, &b->raw_, n, alpha, beta, &c->raw_);
    // Scalar fallback
    else {
        for (std::size_t i = 0; i < n; i++) {
            c[i] = in_type_(scale_type_(a[i]) * (*alpha) + scale_type_(b[i]) * (*beta));
        }
    }
}

/**
 *  @brief Elementwise fused multiply-add: d[i] = alpha * a[i] * b[i] + beta * c[i]
 *  @tparam in_type_ Element type
 *  @tparam scale_type_ Scalar type for alpha/beta
 *  @tparam allow_simd_ Enable SIMD dispatch
 */
template <typename in_type_, typename scale_type_ = typename in_type_::scale_t, bool allow_simd_ = true>
void fma(in_type_ const *a, in_type_ const *b, std::size_t n, in_type_ const *c, scale_type_ const *alpha,
         scale_type_ const *beta, in_type_ *d) noexcept {
    constexpr bool simd = allow_simd_ && std::is_same_v<scale_type_, typename in_type_::scale_t>;

    if constexpr (std::is_same_v<in_type_, f64_t> && simd)
        nk_fma_f64(&a->raw_, &b->raw_, &c->raw_, n, alpha, beta, &d->raw_);
    else if constexpr (std::is_same_v<in_type_, f32_t> && simd)
        nk_fma_f32(&a->raw_, &b->raw_, &c->raw_, n, alpha, beta, &d->raw_);
    else if constexpr (std::is_same_v<in_type_, f16_t> && simd)
        nk_fma_f16(&a->raw_, &b->raw_, &c->raw_, n, alpha, beta, &d->raw_);
    else if constexpr (std::is_same_v<in_type_, bf16_t> && simd)
        nk_fma_bf16(&a->raw_, &b->raw_, &c->raw_, n, alpha, beta, &d->raw_);
    else if constexpr (std::is_same_v<in_type_, i8_t> && simd)
        nk_fma_i8(&a->raw_, &b->raw_, &c->raw_, n, alpha, beta, &d->raw_);
    else if constexpr (std::is_same_v<in_type_, u8_t> && simd)
        nk_fma_u8(&a->raw_, &b->raw_, &c->raw_, n, alpha, beta, &d->raw_);
    else if constexpr (std::is_same_v<in_type_, i16_t> && simd)
        nk_fma_i16(&a->raw_, &b->raw_, &c->raw_, n, alpha, beta, &d->raw_);
    else if constexpr (std::is_same_v<in_type_, u16_t> && simd)
        nk_fma_u16(&a->raw_, &b->raw_, &c->raw_, n, alpha, beta, &d->raw_);
    else if constexpr (std::is_same_v<in_type_, i32_t> && simd)
        nk_fma_i32(&a->raw_, &b->raw_, &c->raw_, n, alpha, beta, &d->raw_);
    else if constexpr (std::is_same_v<in_type_, u32_t> && simd)
        nk_fma_u32(&a->raw_, &b->raw_, &c->raw_, n, alpha, beta, &d->raw_);
    else if constexpr (std::is_same_v<in_type_, i64_t> && simd)
        nk_fma_i64(&a->raw_, &b->raw_, &c->raw_, n, alpha, beta, &d->raw_);
    else if constexpr (std::is_same_v<in_type_, u64_t> && simd)
        nk_fma_u64(&a->raw_, &b->raw_, &c->raw_, n, alpha, beta, &d->raw_);
    // Scalar fallback
    else {
        for (std::size_t i = 0; i < n; i++) {
            d[i] = in_type_(scale_type_(a[i]) * scale_type_(b[i]) * (*alpha) + scale_type_(c[i]) * (*beta));
        }
    }
}

#pragma endregion Elementwise Kernels

#pragma region Reduction Kernels

/**
 *  @brief Sum all elements: sum(data[i])
 *  @tparam in_type_ Input vector element type
 *  @tparam result_type_ Accumulator type (default: in_type_::reduce_add_result_t, often widened)
 *  @tparam allow_simd_ Enable SIMD kernel dispatch when true
 *
 *  @param data Input array
 *  @param count Number of elements
 *  @param stride_bytes Stride between elements in bytes (use sizeof(in_type_) for contiguous)
 *  @param result Output sum
 */
template <typename in_type_, typename result_type_ = typename in_type_::reduce_add_result_t, bool allow_simd_ = true>
void reduce_add(in_type_ const *data, std::size_t count, std::size_t stride_bytes, result_type_ *result) noexcept {
    constexpr bool simd = allow_simd_ && std::is_same_v<result_type_, typename in_type_::reduce_add_result_t>;

    if constexpr (std::is_same_v<in_type_, f64_t> && simd)
        nk_reduce_add_f64(&data->raw_, count, stride_bytes, &result->raw_);
    else if constexpr (std::is_same_v<in_type_, f32_t> && simd)
        nk_reduce_add_f32(&data->raw_, count, stride_bytes, &result->raw_);
    else if constexpr (std::is_same_v<in_type_, f16_t> && simd)
        nk_reduce_add_f16(&data->raw_, count, stride_bytes, &result->raw_);
    else if constexpr (std::is_same_v<in_type_, bf16_t> && simd)
        nk_reduce_add_bf16(&data->raw_, count, stride_bytes, &result->raw_);
    else if constexpr (std::is_same_v<in_type_, e4m3_t> && simd)
        nk_reduce_add_e4m3(&data->raw_, count, stride_bytes, &result->raw_);
    else if constexpr (std::is_same_v<in_type_, e5m2_t> && simd)
        nk_reduce_add_e5m2(&data->raw_, count, stride_bytes, &result->raw_);
    else if constexpr (std::is_same_v<in_type_, i8_t> && simd)
        nk_reduce_add_i8(&data->raw_, count, stride_bytes, &result->raw_);
    else if constexpr (std::is_same_v<in_type_, u8_t> && simd)
        nk_reduce_add_u8(&data->raw_, count, stride_bytes, &result->raw_);
    else if constexpr (std::is_same_v<in_type_, i16_t> && simd)
        nk_reduce_add_i16(&data->raw_, count, stride_bytes, &result->raw_);
    else if constexpr (std::is_same_v<in_type_, u16_t> && simd)
        nk_reduce_add_u16(&data->raw_, count, stride_bytes, &result->raw_);
    else if constexpr (std::is_same_v<in_type_, i32_t> && simd)
        nk_reduce_add_i32(&data->raw_, count, stride_bytes, &result->raw_);
    else if constexpr (std::is_same_v<in_type_, u32_t> && simd)
        nk_reduce_add_u32(&data->raw_, count, stride_bytes, &result->raw_);
    else if constexpr (std::is_same_v<in_type_, i64_t> && simd)
        nk_reduce_add_i64(&data->raw_, count, stride_bytes, &result->raw_);
    else if constexpr (std::is_same_v<in_type_, u64_t> && simd)
        nk_reduce_add_u64(&data->raw_, count, stride_bytes, &result->raw_);
    // Scalar fallback
    else {
        result_type_ sum {};
        auto const *ptr = reinterpret_cast<std::byte const *>(data);
        for (std::size_t i = 0; i < count; i++, ptr += stride_bytes)
            sum = sum.saturating_add(result_type_(*reinterpret_cast<in_type_ const *>(ptr)));
        *result = sum;
    }
}

/**
 *  @brief Find minimum element and its index: argmin(data[i])
 *  @tparam in_type_ Input vector element type
 *  @tparam result_type_ Result type for the minimum value (f32_t for half-precision inputs)
 *  @tparam allow_simd_ Enable SIMD kernel dispatch when true
 *
 *  @param data Input array
 *  @param count Number of elements
 *  @param stride_bytes Stride between elements in bytes (use sizeof(in_type_) for contiguous)
 *  @param min_value Output minimum value
 *  @param min_index Output index of minimum value
 *
 *  @note For f16_t, bf16_t, e4m3_t, e5m2_t inputs, result_type_ must be f32_t for SIMD dispatch
 */
template <typename in_type_, bool allow_simd_ = true>
void reduce_min(in_type_ const *data, std::size_t count, std::size_t stride_bytes, in_type_ *min_value,
                std::size_t *min_index) noexcept {

    if constexpr (std::is_same_v<in_type_, f64_t> && allow_simd_)
        nk_reduce_min_f64(&data->raw_, count, stride_bytes, &min_value->raw_, min_index);
    else if constexpr (std::is_same_v<in_type_, f32_t> && allow_simd_)
        nk_reduce_min_f32(&data->raw_, count, stride_bytes, &min_value->raw_, min_index);
    else if constexpr (std::is_same_v<in_type_, f16_t> && allow_simd_)
        nk_reduce_min_f16(&data->raw_, count, stride_bytes, &min_value->raw_, min_index);
    else if constexpr (std::is_same_v<in_type_, bf16_t> && allow_simd_)
        nk_reduce_min_bf16(&data->raw_, count, stride_bytes, &min_value->raw_, min_index);
    else if constexpr (std::is_same_v<in_type_, e4m3_t> && allow_simd_)
        nk_reduce_min_e4m3(&data->raw_, count, stride_bytes, &min_value->raw_, min_index);
    else if constexpr (std::is_same_v<in_type_, e5m2_t> && allow_simd_)
        nk_reduce_min_e5m2(&data->raw_, count, stride_bytes, &min_value->raw_, min_index);
    else if constexpr (std::is_same_v<in_type_, i8_t> && allow_simd_)
        nk_reduce_min_i8(&data->raw_, count, stride_bytes, &min_value->raw_, min_index);
    else if constexpr (std::is_same_v<in_type_, u8_t> && allow_simd_)
        nk_reduce_min_u8(&data->raw_, count, stride_bytes, &min_value->raw_, min_index);
    else if constexpr (std::is_same_v<in_type_, i16_t> && allow_simd_)
        nk_reduce_min_i16(&data->raw_, count, stride_bytes, &min_value->raw_, min_index);
    else if constexpr (std::is_same_v<in_type_, u16_t> && allow_simd_)
        nk_reduce_min_u16(&data->raw_, count, stride_bytes, &min_value->raw_, min_index);
    else if constexpr (std::is_same_v<in_type_, i32_t> && allow_simd_)
        nk_reduce_min_i32(&data->raw_, count, stride_bytes, &min_value->raw_, min_index);
    else if constexpr (std::is_same_v<in_type_, u32_t> && allow_simd_)
        nk_reduce_min_u32(&data->raw_, count, stride_bytes, &min_value->raw_, min_index);
    else if constexpr (std::is_same_v<in_type_, i64_t> && allow_simd_)
        nk_reduce_min_i64(&data->raw_, count, stride_bytes, &min_value->raw_, min_index);
    else if constexpr (std::is_same_v<in_type_, u64_t> && allow_simd_)
        nk_reduce_min_u64(&data->raw_, count, stride_bytes, &min_value->raw_, min_index);
    // Scalar fallback
    else {
        in_type_ best_value = in_type_::max();
        std::size_t best_index = 0;
        auto const *ptr = reinterpret_cast<std::byte const *>(data);
        for (std::size_t i = 0; i < count; i++, ptr += stride_bytes) {
            in_type_ v = *reinterpret_cast<in_type_ const *>(ptr);
            if (v < best_value) best_value = v, best_index = i;
        }
        *min_value = best_value;
        *min_index = best_index;
    }
}

/**
 *  @brief Find maximum element and its index: argmax(data[i])
 *  @tparam in_type_ Input vector element type
 *  @tparam result_type_ Result type for the maximum value (f32_t for half-precision inputs)
 *  @tparam allow_simd_ Enable SIMD kernel dispatch when true
 *
 *  @param data Input array
 *  @param count Number of elements
 *  @param stride_bytes Stride between elements in bytes (use sizeof(in_type_) for contiguous)
 *  @param max_value Output maximum value
 *  @param max_index Output index of maximum value
 *
 *  @note For f16_t, bf16_t, e4m3_t, e5m2_t inputs, result_type_ must be f32_t for SIMD dispatch
 */
template <typename in_type_, typename result_type_ = in_type_, bool allow_simd_ = true>
void reduce_max(in_type_ const *data, std::size_t count, std::size_t stride_bytes, result_type_ *max_value,
                std::size_t *max_index) noexcept {

    if constexpr (std::is_same_v<in_type_, f64_t> && allow_simd_)
        nk_reduce_max_f64(&data->raw_, count, stride_bytes, &max_value->raw_, max_index);
    else if constexpr (std::is_same_v<in_type_, f32_t> && allow_simd_)
        nk_reduce_max_f32(&data->raw_, count, stride_bytes, &max_value->raw_, max_index);
    else if constexpr (std::is_same_v<in_type_, f16_t> && allow_simd_)
        nk_reduce_max_f16(&data->raw_, count, stride_bytes, &max_value->raw_, max_index);
    else if constexpr (std::is_same_v<in_type_, bf16_t> && allow_simd_)
        nk_reduce_max_bf16(&data->raw_, count, stride_bytes, &max_value->raw_, max_index);
    else if constexpr (std::is_same_v<in_type_, e4m3_t> && allow_simd_)
        nk_reduce_max_e4m3(&data->raw_, count, stride_bytes, &max_value->raw_, max_index);
    else if constexpr (std::is_same_v<in_type_, e5m2_t> && allow_simd_)
        nk_reduce_max_e5m2(&data->raw_, count, stride_bytes, &max_value->raw_, max_index);
    else if constexpr (std::is_same_v<in_type_, i8_t> && allow_simd_)
        nk_reduce_max_i8(&data->raw_, count, stride_bytes, &max_value->raw_, max_index);
    else if constexpr (std::is_same_v<in_type_, u8_t> && allow_simd_)
        nk_reduce_max_u8(&data->raw_, count, stride_bytes, &max_value->raw_, max_index);
    else if constexpr (std::is_same_v<in_type_, i16_t> && allow_simd_)
        nk_reduce_max_i16(&data->raw_, count, stride_bytes, &max_value->raw_, max_index);
    else if constexpr (std::is_same_v<in_type_, u16_t> && allow_simd_)
        nk_reduce_max_u16(&data->raw_, count, stride_bytes, &max_value->raw_, max_index);
    else if constexpr (std::is_same_v<in_type_, i32_t> && allow_simd_)
        nk_reduce_max_i32(&data->raw_, count, stride_bytes, &max_value->raw_, max_index);
    else if constexpr (std::is_same_v<in_type_, u32_t> && allow_simd_)
        nk_reduce_max_u32(&data->raw_, count, stride_bytes, &max_value->raw_, max_index);
    else if constexpr (std::is_same_v<in_type_, i64_t> && allow_simd_)
        nk_reduce_max_i64(&data->raw_, count, stride_bytes, &max_value->raw_, max_index);
    else if constexpr (std::is_same_v<in_type_, u64_t> && allow_simd_)
        nk_reduce_max_u64(&data->raw_, count, stride_bytes, &max_value->raw_, max_index);
    // Scalar fallback
    else {
        in_type_ best_value = in_type_::min();
        std::size_t best_index = 0;
        auto const *ptr = reinterpret_cast<std::byte const *>(data);
        for (std::size_t i = 0; i < count; i++, ptr += stride_bytes) {
            in_type_ v = *reinterpret_cast<in_type_ const *>(ptr);
            if (v > best_value) best_value = v, best_index = i;
        }
        *max_value = best_value;
        *max_index = best_index;
    }
}

#pragma endregion Reduction Kernels

#pragma region Curved Kernels

/**
 *  @brief Bilinear form: a^T * C * b where C is an n×n matrix (row-major)
 *  @tparam in_type_ Input vector element type (real or complex)
 *  @tparam result_type_ Accumulator type (default: in_type_::bilinear_result_t)
 *  @tparam allow_simd_ Enable SIMD kernel dispatch when true
 *  @note For weighted inner products, Mahalanobis distance, etc.
 */
template <typename in_type_, typename result_type_ = typename in_type_::bilinear_result_t, bool allow_simd_ = true>
void bilinear(in_type_ const *a, in_type_ const *b, in_type_ const *c, std::size_t n, result_type_ *r) noexcept {
    constexpr bool simd = allow_simd_ && std::is_same_v<result_type_, typename in_type_::bilinear_result_t>;

    // Real types
    if constexpr (std::is_same_v<in_type_, f64_t> && simd) nk_bilinear_f64(&a->raw_, &b->raw_, &c->raw_, n, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, f32_t> && simd)
        nk_bilinear_f32(&a->raw_, &b->raw_, &c->raw_, n, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, f16_t> && simd)
        nk_bilinear_f16(&a->raw_, &b->raw_, &c->raw_, n, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, bf16_t> && simd)
        nk_bilinear_bf16(&a->raw_, &b->raw_, &c->raw_, n, &r->raw_);
    // Complex types
    else if constexpr (std::is_same_v<in_type_, f64c_t> && simd)
        nk_bilinear_f64c(&a->raw_, &b->raw_, &c->raw_, n, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, f32c_t> && simd)
        nk_bilinear_f32c(&a->raw_, &b->raw_, &c->raw_, n, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, f16c_t> && simd)
        nk_bilinear_f16c(&a->raw_, &b->raw_, &c->raw_, n, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, bf16c_t> && simd)
        nk_bilinear_bf16c(&a->raw_, &b->raw_, &c->raw_, n, &r->raw_);
    // Scalar fallback
    else {
        result_type_ sum {};
        for (std::size_t i = 0; i < n; i++) {
            for (std::size_t j = 0; j < n; j++) {
                sum = sum + result_type_(a[i]) * result_type_(c[i * n + j]) * result_type_(b[j]);
            }
        }
        *r = sum;
    }
}

/**
 *  @brief Mahalanobis distance: sqrt((a-b)^T * C * (a-b)) where C is an n×n matrix (row-major)
 *  @tparam in_type_ Input vector element type
 *  @tparam result_type_ Accumulator type (default: in_type_::mahalanobis_result_t)
 *  @tparam allow_simd_ Enable SIMD kernel dispatch when true
 */
template <typename in_type_, typename result_type_ = typename in_type_::mahalanobis_result_t, bool allow_simd_ = true>
void mahalanobis(in_type_ const *a, in_type_ const *b, in_type_ const *c, std::size_t n, result_type_ *r) noexcept {
    constexpr bool simd = allow_simd_ && std::is_same_v<result_type_, typename in_type_::mahalanobis_result_t>;

    if constexpr (std::is_same_v<in_type_, f64_t> && simd)
        nk_mahalanobis_f64(&a->raw_, &b->raw_, &c->raw_, n, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, f32_t> && simd)
        nk_mahalanobis_f32(&a->raw_, &b->raw_, &c->raw_, n, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, f16_t> && simd)
        nk_mahalanobis_f16(&a->raw_, &b->raw_, &c->raw_, n, &r->raw_);
    else if constexpr (std::is_same_v<in_type_, bf16_t> && simd)
        nk_mahalanobis_bf16(&a->raw_, &b->raw_, &c->raw_, n, &r->raw_);
    // Scalar fallback
    else {
        result_type_ sum {};
        for (std::size_t i = 0; i < n; i++) {
            result_type_ di = result_type_(a[i]) - result_type_(b[i]);
            for (std::size_t j = 0; j < n; j++) {
                result_type_ dj = result_type_(a[j]) - result_type_(b[j]);
                sum = sum + di * result_type_(c[i * n + j]) * dj;
            }
        }
        *r = sum.sqrt();
    }
}

#pragma endregion Curved Kernels

#pragma region Geospatial Kernels

/**
 *  @brief Batched Haversine distance (great-circle distance on sphere)
 *  @tparam in_type_ Input coordinate type (f32_t, f64_t)
 *  @tparam precision_type_ Precision type for scalar fallback computations (default: double)
 *  @tparam allow_simd_ Enable SIMD kernel dispatch when true
 *
 *  @param a_lats Array of latitudes for first points (radians)
 *  @param a_lons Array of longitudes for first points (radians)
 *  @param b_lats Array of latitudes for second points (radians)
 *  @param b_lons Array of longitudes for second points (radians)
 *  @param n Number of point pairs
 *  @param results Output array of distances (meters)
 *
 *  @note Uses spherical Earth model with mediatorial radius (6335439.0 m)
 *  @note Accuracy: 0.3-0.6% vs WGS-84, suitable for ranking/similarity
 */
template <typename in_type_, typename precision_type_ = in_type_, bool allow_simd_ = true>
void haversine(in_type_ const *a_lats, in_type_ const *a_lons, in_type_ const *b_lats, in_type_ const *b_lons,
               std::size_t n, in_type_ *results) noexcept {
    constexpr bool simd = allow_simd_ && std::is_same_v<in_type_, precision_type_>;

    if constexpr (std::is_same_v<in_type_, f64_t> && simd)
        nk_haversine_f64(&a_lats->raw_, &a_lons->raw_, &b_lats->raw_, &b_lons->raw_, n, &results->raw_);
    else if constexpr (std::is_same_v<in_type_, f32_t> && simd)
        nk_haversine_f32(&a_lats->raw_, &a_lons->raw_, &b_lats->raw_, &b_lons->raw_, n, &results->raw_);
    // Scalar fallback
    else {
        precision_type_ const earth_radius = precision_type_(6335439.0); // mediatorial radius in meters

        for (std::size_t i = 0; i < n; i++) {
            precision_type_ first_latitude = precision_type_(a_lats[i]);
            precision_type_ first_longitude = precision_type_(a_lons[i]);
            precision_type_ second_latitude = precision_type_(b_lats[i]);
            precision_type_ second_longitude = precision_type_(b_lons[i]);

            precision_type_ latitude_delta = second_latitude - first_latitude;
            precision_type_ longitude_delta = second_longitude - first_longitude;

            // Haversine formula: a = sin²(Δlat/2) + cos(lat1)×cos(lat2)×sin²(Δlon/2)
            precision_type_ sin_latitude_delta_half = (latitude_delta * precision_type_(0.5)).sin();
            precision_type_ sin_longitude_delta_half = (longitude_delta * precision_type_(0.5)).sin();
            precision_type_ cos_first_latitude = first_latitude.cos();
            precision_type_ cos_second_latitude = second_latitude.cos();

            precision_type_ haversine_term = sin_latitude_delta_half * sin_latitude_delta_half +
                                             cos_first_latitude * cos_second_latitude * sin_longitude_delta_half *
                                                 sin_longitude_delta_half;

            // Central angle: c = 2 * atan2(sqrt(a), sqrt(1-a))
            precision_type_ sqrt_haversine = haversine_term.sqrt();
            precision_type_ sqrt_complement = (precision_type_(1.0) - haversine_term).sqrt();
            precision_type_ central_angle = precision_type_(2.0) * sqrt_haversine.atan2(sqrt_complement);

            results[i] = in_type_(static_cast<double>(earth_radius * central_angle));
        }
    }
}

/**
 *  @brief Batched Vincenty distance (geodesic distance on ellipsoid)
 *  @tparam in_type_ Input coordinate type (f32_t, f64_t)
 *  @tparam precision_type_ Precision type for scalar fallback computations (default: double)
 *  @tparam allow_simd_ Enable SIMD kernel dispatch when true
 *
 *  @param a_lats Array of latitudes for first points (radians)
 *  @param a_lons Array of longitudes for first points (radians)
 *  @param b_lats Array of latitudes for second points (radians)
 *  @param b_lons Array of longitudes for second points (radians)
 *  @param n Number of point pairs
 *  @param results Output array of distances (meters)
 *
 *  @note Uses WGS-84/IERS-2003 ellipsoid model
 *  @note Accuracy: 0.01-0.2% vs WGS-84, 3-20x more accurate than Haversine
 *  @note Iterative algorithm with max 100 iterations
 */
template <typename in_type_, typename precision_type_ = in_type_, bool allow_simd_ = true>
void vincenty(in_type_ const *a_lats, in_type_ const *a_lons, in_type_ const *b_lats, in_type_ const *b_lons,
              std::size_t n, in_type_ *results) noexcept {
    constexpr bool simd = allow_simd_ && std::is_same_v<in_type_, precision_type_>;

    if constexpr (std::is_same_v<in_type_, f64_t> && simd)
        nk_vincenty_f64(&a_lats->raw_, &a_lons->raw_, &b_lats->raw_, &b_lons->raw_, n, &results->raw_);
    else if constexpr (std::is_same_v<in_type_, f32_t> && simd)
        nk_vincenty_f32(&a_lats->raw_, &a_lons->raw_, &b_lats->raw_, &b_lons->raw_, n, &results->raw_);
    // Scalar fallback
    else {
        precision_type_ const equatorial_radius = precision_type_(6378136.6);
        precision_type_ const polar_radius = precision_type_(6356751.9);
        precision_type_ const flattening = precision_type_(1.0) / precision_type_(298.25642);
        precision_type_ const convergence_threshold = precision_type_(1e-12);
        constexpr int max_iterations = 100;

        for (std::size_t i = 0; i < n; i++) {
            precision_type_ first_latitude = precision_type_(a_lats[i]);
            precision_type_ second_latitude = precision_type_(b_lats[i]);
            precision_type_ longitude_difference = precision_type_(b_lons[i]) - precision_type_(a_lons[i]);

            // Reduced latitudes on the auxiliary sphere
            precision_type_ tan_reduced_first = (precision_type_(1.0) - flattening) * first_latitude.tan();
            precision_type_ tan_reduced_second = (precision_type_(1.0) - flattening) * second_latitude.tan();
            precision_type_ cos_reduced_first = precision_type_(1.0) /
                                                (precision_type_(1.0) + tan_reduced_first * tan_reduced_first).sqrt();
            precision_type_ sin_reduced_first = tan_reduced_first * cos_reduced_first;
            precision_type_ cos_reduced_second =
                precision_type_(1.0) / (precision_type_(1.0) + tan_reduced_second * tan_reduced_second).sqrt();
            precision_type_ sin_reduced_second = tan_reduced_second * cos_reduced_second;

            // Iterative convergence of lambda (difference in longitude on auxiliary sphere)
            precision_type_ lambda = longitude_difference;
            precision_type_ lambda_previous;
            precision_type_ sin_angular_distance, cos_angular_distance, angular_distance;
            precision_type_ sin_azimuth, cos_squared_azimuth, cos_double_angular_midpoint;
            bool coincident = false;

            for (unsigned int iteration = 0; iteration < max_iterations; iteration++) {
                precision_type_ sin_lambda = lambda.sin();
                precision_type_ cos_lambda = lambda.cos();

                precision_type_ cross_term = cos_reduced_second * sin_lambda;
                precision_type_ mixed_term = cos_reduced_first * sin_reduced_second -
                                             sin_reduced_first * cos_reduced_second * cos_lambda;
                sin_angular_distance = (cross_term * cross_term + mixed_term * mixed_term).sqrt();

                if (sin_angular_distance == precision_type_(0.0)) {
                    coincident = true;
                    break;
                }

                cos_angular_distance = sin_reduced_first * sin_reduced_second +
                                       cos_reduced_first * cos_reduced_second * cos_lambda;
                angular_distance = sin_angular_distance.atan2(cos_angular_distance);

                sin_azimuth = cos_reduced_first * cos_reduced_second * sin_lambda / sin_angular_distance;
                cos_squared_azimuth = precision_type_(1.0) - sin_azimuth * sin_azimuth;

                // Handle equatorial geodesic case
                cos_double_angular_midpoint = (cos_squared_azimuth != precision_type_(0.0))
                                                  ? cos_angular_distance - precision_type_(2.0) * sin_reduced_first *
                                                                               sin_reduced_second / cos_squared_azimuth
                                                  : precision_type_(0.0);

                precision_type_ correction_factor =
                    flattening / precision_type_(16.0) * cos_squared_azimuth *
                    (precision_type_(4.0) +
                     flattening * (precision_type_(4.0) - precision_type_(3.0) * cos_squared_azimuth));

                lambda_previous = lambda;
                lambda = longitude_difference +
                         (precision_type_(1.0) - correction_factor) * flattening * sin_azimuth *
                             (angular_distance +
                              correction_factor * sin_angular_distance *
                                  (cos_double_angular_midpoint +
                                   correction_factor * cos_angular_distance *
                                       (precision_type_(-1.0) + precision_type_(2.0) * cos_double_angular_midpoint *
                                                                    cos_double_angular_midpoint)));

                if ((lambda - lambda_previous).abs() < convergence_threshold) break;
            }

            if (coincident) {
                results[i] = in_type_(0.0);
                continue;
            }

            // Final distance calculation
            precision_type_ u_squared = cos_squared_azimuth *
                                        (equatorial_radius * equatorial_radius - polar_radius * polar_radius) /
                                        (polar_radius * polar_radius);
            precision_type_ series_a =
                precision_type_(1.0) +
                u_squared / precision_type_(16384.0) *
                    (precision_type_(4096.0) +
                     u_squared * (precision_type_(-768.0) +
                                  u_squared * (precision_type_(320.0) - precision_type_(175.0) * u_squared)));
            precision_type_ series_b = u_squared / precision_type_(1024.0) *
                                       (precision_type_(256.0) +
                                        u_squared *
                                            (precision_type_(-128.0) +
                                             u_squared * (precision_type_(74.0) - precision_type_(47.0) * u_squared)));

            precision_type_ angular_correction =
                series_b * sin_angular_distance *
                (cos_double_angular_midpoint +
                 series_b / precision_type_(4.0) *
                     (cos_angular_distance *
                          (precision_type_(-1.0) +
                           precision_type_(2.0) * cos_double_angular_midpoint * cos_double_angular_midpoint) -
                      series_b / precision_type_(6.0) * cos_double_angular_midpoint *
                          (precision_type_(-3.0) + precision_type_(4.0) * sin_angular_distance * sin_angular_distance) *
                          (precision_type_(-3.0) +
                           precision_type_(4.0) * cos_double_angular_midpoint * cos_double_angular_midpoint)));

            results[i] = in_type_(
                static_cast<double>(polar_radius * series_a * (angular_distance - angular_correction)));
        }
    }
}

#pragma endregion Geospatial Kernels

#pragma region Sparse Kernels

/**
 *  @brief Count intersection of two sorted index arrays
 *  @tparam index_type_ Index type (u16_t, u32_t)
 *  @tparam allow_simd_ Enable SIMD kernel dispatch when true
 *
 *  @param a First sorted index array (ascending, unique elements)
 *  @param b Second sorted index array (ascending, unique elements)
 *  @param a_length Number of elements in first array
 *  @param b_length Number of elements in second array
 *  @param count Output intersection count
 */
template <typename index_type_, bool allow_simd_ = true>
void intersect(index_type_ const *a, index_type_ const *b, std::size_t a_length, std::size_t b_length,
               std::uint32_t *count) noexcept {
    constexpr bool simd = allow_simd_;

    if constexpr (std::is_same_v<index_type_, u16_t> && simd)
        nk_intersect_u16(&a->raw_, &b->raw_, a_length, b_length, count);
    else if constexpr (std::is_same_v<index_type_, u32_t> && simd)
        nk_intersect_u32(&a->raw_, &b->raw_, a_length, b_length, count);
    // Scalar fallback
    else {
        std::uint32_t c = 0;
        std::size_t i = 0, j = 0;
        while (i < a_length && j < b_length) {
            if (a[i] < b[j]) i++;
            else if (b[j] < a[i]) j++;
            else c++, i++, j++;
        }
        *count = c;
    }
}

/**
 *  @brief Sparse weighted dot product (sorted indices)
 *  @tparam index_type_ Index type (u16_t, u32_t)
 *  @tparam weight_t Weight type (bf16_t for u16 indices, f32_t for u32 indices)
 *  @tparam result_type_ Result type (default: f32_t)
 *  @tparam allow_simd_ Enable SIMD kernel dispatch when true
 *
 *  @param a First sorted index array (ascending, unique elements)
 *  @param b Second sorted index array (ascending, unique elements)
 *  @param a_weights Weights corresponding to first indices
 *  @param b_weights Weights corresponding to second indices
 *  @param a_length Number of elements in first array
 *  @param b_length Number of elements in second array
 *  @param product Output dot product
 *
 *  @note Computes sum of a_weights[i] * b_weights[j] for all i,j where a[i] == b[j]
 */
template <typename index_type_, typename weight_t, typename result_type_ = f32_t, bool allow_simd_ = true>
void sparse_dot(index_type_ const *a, index_type_ const *b, weight_t const *a_weights, weight_t const *b_weights,
                std::size_t a_length, std::size_t b_length, result_type_ *product) noexcept {
    constexpr bool simd = allow_simd_ && std::is_same_v<result_type_, typename weight_t::dot_result_t>;

    // u16 indices + bf16 weights → f32 product
    if constexpr (std::is_same_v<index_type_, u16_t> && std::is_same_v<weight_t, bf16_t> && simd)
        nk_sparse_dot_u16bf16(&a->raw_, &b->raw_, &a_weights->raw_, &b_weights->raw_, a_length, b_length,
                              &product->raw_);
    else if constexpr (std::is_same_v<index_type_, u32_t> && std::is_same_v<weight_t, f32_t> && simd)
        nk_sparse_dot_u32f32(&a->raw_, &b->raw_, &a_weights->raw_, &b_weights->raw_, a_length, b_length,
                             &product->raw_);
    // Scalar fallback
    else {
        result_type_ sum {};
        std::size_t i = 0, j = 0;
        while (i < a_length && j < b_length) {
            if (a[i] < b[j]) i++;
            else if (b[j] < a[i]) j++;
            else sum = fused_multiply_add<weight_t, result_type_>(sum, a_weights[i], b_weights[j]), i++, j++;
        }
        *product = sum;
    }
}

#pragma endregion Sparse Kernels

#pragma region Binary Kernels

/**
 *  @brief Hamming distance between bit vectors
 *  @param a First bit vector [n] (each element is u1x8_t = 8 bits)
 *  @param b Second bit vector [n]
 *  @param n Number of u1x8_t elements (total bits = n * 8)
 *  @return Number of differing bits
 */
inline void hamming(u1x8_t const *a, u1x8_t const *b, std::size_t n, u32_t *r) noexcept {
    std::size_t sum = 0;
    for (std::size_t i = 0; i < n; i++) sum += a[i].hamming(b[i]);
    *r = static_cast<u32_t>(sum);
}

/**
 *  @brief Jaccard similarity for bit vectors: |A & B| / |A | B|
 *  @param a First bit vector [n]
 *  @param b Second bit vector [n]
 *  @param n Number of u1x8_t elements
 *  @return Jaccard similarity in [0, 1]
 */
void jaccard(u1x8_t const *a, u1x8_t const *b, std::size_t n, f32_t *r) noexcept {
    std::uint32_t intersection_count = 0;
    std::uint32_t union_count = 0;
    for (std::size_t i = 0; i < n; i++) {
        intersection_count += a[i].intersection(b[i]);
        union_count += a[i].union_size(b[i]);
    }
    // Jaccard distance = 1 - (intersection / union)
    // Both empty = identical = distance 0; otherwise 1 - similarity
    if (union_count == 0) *r = f32_t(0.0);
    else *r = f32_t(1.0f - static_cast<float>(intersection_count) / static_cast<float>(union_count));
}

#pragma endregion Binary Kernels

#pragma region SVD Helpers for Scalar Fallbacks

/** @brief 3×3 matrix determinant. */
template <typename scalar_type_>
scalar_type_ det3x3_(scalar_type_ const *m) {
    return m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) + m[2] * (m[3] * m[7] - m[4] * m[6]);
}

/** @brief Conditional swap helper. */
template <typename scalar_type_>
void conditional_swap_(bool c, scalar_type_ *x, scalar_type_ *y) {
    scalar_type_ temp = *x;
    *x = c ? *y : *x;
    *y = c ? temp : *y;
}

/** @brief Conditional negating swap helper. */
template <typename scalar_type_>
void conditional_negating_swap_(bool c, scalar_type_ *x, scalar_type_ *y) {
    scalar_type_ neg_x = scalar_type_(0.0) - *x;
    *x = c ? *y : *x;
    *y = c ? neg_x : *y;
}

/** @brief Approximate Givens quaternion for Jacobi eigenanalysis. */
template <typename scalar_type_>
void approximate_givens_quaternion_(scalar_type_ a11, scalar_type_ a12, scalar_type_ a22, scalar_type_ *cos_half,
                                    scalar_type_ *sin_half) {
    constexpr scalar_type_ gamma_k = scalar_type_(5.828427124746190);  // γ = (√8 + 3)² / 4
    constexpr scalar_type_ cstar_k = scalar_type_(0.9238795325112867); // cos(π/8)
    constexpr scalar_type_ sstar_k = scalar_type_(0.3826834323650898); // sin(π/8)

    *cos_half = scalar_type_(2.0) * (a11 - a22);
    *sin_half = a12;
    bool use_givens = gamma_k * (*sin_half) * (*sin_half) < (*cos_half) * (*cos_half);
    scalar_type_ w = ((*cos_half) * (*cos_half) + (*sin_half) * (*sin_half)).rsqrt();
    *cos_half = use_givens ? w * (*cos_half) : cstar_k;
    *sin_half = use_givens ? w * (*sin_half) : sstar_k;
}

/** @brief Jacobi conjugation step for eigenanalysis. */
template <typename scalar_type_>
void jacobi_conjugation_(int idx_x, int idx_y, int idx_z, scalar_type_ *s11, scalar_type_ *s21, scalar_type_ *s22,
                         scalar_type_ *s31, scalar_type_ *s32, scalar_type_ *s33, scalar_type_ *quat) {

    scalar_type_ cos_half, sin_half;
    approximate_givens_quaternion_(*s11, *s21, *s22, &cos_half, &sin_half);
    scalar_type_ scale = cos_half * cos_half + sin_half * sin_half;
    scalar_type_ cos_theta = (cos_half * cos_half - sin_half * sin_half) / scale;
    scalar_type_ sin_theta = (scalar_type_(2.0) * sin_half * cos_half) / scale;
    scalar_type_ s11_old = *s11, s21_old = *s21, s22_old = *s22;
    scalar_type_ s31_old = *s31, s32_old = *s32, s33_old = *s33;

    *s11 = cos_theta * (cos_theta * s11_old + sin_theta * s21_old) +
           sin_theta * (cos_theta * s21_old + sin_theta * s22_old);
    *s21 = cos_theta * ((scalar_type_(0.0) - sin_theta) * s11_old + cos_theta * s21_old) +
           sin_theta * ((scalar_type_(0.0) - sin_theta) * s21_old + cos_theta * s22_old);
    *s22 = (scalar_type_(0.0) - sin_theta) * ((scalar_type_(0.0) - sin_theta) * s11_old + cos_theta * s21_old) +
           cos_theta * ((scalar_type_(0.0) - sin_theta) * s21_old + cos_theta * s22_old);
    *s31 = cos_theta * s31_old + sin_theta * s32_old;
    *s32 = (scalar_type_(0.0) - sin_theta) * s31_old + cos_theta * s32_old;
    *s33 = s33_old;

    // Update quaternion accumulator
    scalar_type_ quat_temp[3];
    quat_temp[0] = quat[0] * sin_half;
    quat_temp[1] = quat[1] * sin_half;
    quat_temp[2] = quat[2] * sin_half;
    sin_half = sin_half * quat[3];
    quat[0] = quat[0] * cos_half;
    quat[1] = quat[1] * cos_half;
    quat[2] = quat[2] * cos_half;
    quat[3] = quat[3] * cos_half;
    quat[idx_z] = quat[idx_z] + sin_half;
    quat[3] = quat[3] - quat_temp[idx_z];
    quat[idx_x] = quat[idx_x] + quat_temp[idx_y];
    quat[idx_y] = quat[idx_y] - quat_temp[idx_x];
    // Cyclic permutation of matrix elements
    s11_old = *s22, s21_old = *s32, s22_old = *s33, s31_old = *s21, s32_old = *s31, s33_old = *s11;
    *s11 = s11_old, *s21 = s21_old, *s22 = s22_old, *s31 = s31_old, *s32 = s32_old, *s33 = s33_old;
}

/** @brief Convert quaternion to 3×3 rotation matrix. */
template <typename scalar_type_>
void quaternion_to_mat3x3_(scalar_type_ const *quat, scalar_type_ *matrix) {
    scalar_type_ w = quat[3], x = quat[0], y = quat[1], z = quat[2];
    scalar_type_ q_xx = x * x, q_yy = y * y, q_zz = z * z;
    scalar_type_ q_xz = x * z, q_xy = x * y, q_yz = y * z;
    scalar_type_ q_wx = w * x, q_wy = w * y, q_wz = w * z;
    matrix[0] = scalar_type_(1.0) - scalar_type_(2.0) * (q_yy + q_zz);
    matrix[1] = scalar_type_(2.0) * (q_xy - q_wz);
    matrix[2] = scalar_type_(2.0) * (q_xz + q_wy);
    matrix[3] = scalar_type_(2.0) * (q_xy + q_wz);
    matrix[4] = scalar_type_(1.0) - scalar_type_(2.0) * (q_xx + q_zz);
    matrix[5] = scalar_type_(2.0) * (q_yz - q_wx);
    matrix[6] = scalar_type_(2.0) * (q_xz - q_wy);
    matrix[7] = scalar_type_(2.0) * (q_yz + q_wx);
    matrix[8] = scalar_type_(1.0) - scalar_type_(2.0) * (q_xx + q_yy);
}

/** @brief Jacobi eigenanalysis for symmetric 3×3 matrix. */
template <typename scalar_type_>
void jacobi_eigenanalysis_(scalar_type_ *s11, scalar_type_ *s21, scalar_type_ *s22, scalar_type_ *s31,
                           scalar_type_ *s32, scalar_type_ *s33, scalar_type_ *quat) {
    quat[0] = scalar_type_(0.0);
    quat[1] = scalar_type_(0.0);
    quat[2] = scalar_type_(0.0);
    quat[3] = scalar_type_(1.0);
    // 16 iterations for better convergence
    for (unsigned int iter = 0; iter < 16; iter++) {
        jacobi_conjugation_(0, 1, 2, s11, s21, s22, s31, s32, s33, quat);
        jacobi_conjugation_(1, 2, 0, s11, s21, s22, s31, s32, s33, quat);
        jacobi_conjugation_(2, 0, 1, s11, s21, s22, s31, s32, s33, quat);
    }
    scalar_type_ norm = (quat[0] * quat[0] + quat[1] * quat[1] + quat[2] * quat[2] + quat[3] * quat[3]).rsqrt();
    quat[0] = quat[0] * norm;
    quat[1] = quat[1] * norm;
    quat[2] = quat[2] * norm;
    quat[3] = quat[3] * norm;
}

/** @brief QR Givens quaternion for QR decomposition. */
template <typename scalar_type_>
void qr_givens_quaternion_(scalar_type_ a1, scalar_type_ a2, scalar_type_ *cos_half, scalar_type_ *sin_half) {
    constexpr scalar_type_ epsilon_k = scalar_type_(1e-12);

    scalar_type_ a1_sq_plus_a2_sq = a1 * a1 + a2 * a2;
    scalar_type_ rho = a1_sq_plus_a2_sq * a1_sq_plus_a2_sq.rsqrt();
    rho = a1_sq_plus_a2_sq > epsilon_k ? rho : scalar_type_(0.0);
    *sin_half = rho > epsilon_k ? a2 : scalar_type_(0.0);
    scalar_type_ abs_a1 = a1 < scalar_type_(0.0) ? (scalar_type_(0.0) - a1) : a1;
    scalar_type_ max_rho = rho > epsilon_k ? rho : epsilon_k;
    *cos_half = abs_a1 + max_rho;
    bool should_swap = a1 < scalar_type_(0.0);
    conditional_swap_(should_swap, sin_half, cos_half);
    scalar_type_ w = ((*cos_half) * (*cos_half) + (*sin_half) * (*sin_half)).rsqrt();
    *cos_half = (*cos_half) * w;
    *sin_half = (*sin_half) * w;
}

/** @brief Sort singular values in descending order. */
template <typename scalar_type_>
void sort_singular_values_(scalar_type_ *b, scalar_type_ *v) {
    scalar_type_ rho1 = b[0] * b[0] + b[3] * b[3] + b[6] * b[6];
    scalar_type_ rho2 = b[1] * b[1] + b[4] * b[4] + b[7] * b[7];
    scalar_type_ rho3 = b[2] * b[2] + b[5] * b[5] + b[8] * b[8];
    bool should_swap;
    // Sort columns by descending singular value magnitude
    should_swap = rho1 < rho2;
    conditional_negating_swap_(should_swap, &b[0], &b[1]);
    conditional_negating_swap_(should_swap, &v[0], &v[1]);
    conditional_negating_swap_(should_swap, &b[3], &b[4]);
    conditional_negating_swap_(should_swap, &v[3], &v[4]);
    conditional_negating_swap_(should_swap, &b[6], &b[7]);
    conditional_negating_swap_(should_swap, &v[6], &v[7]);
    conditional_swap_(should_swap, &rho1, &rho2);
    should_swap = rho1 < rho3;
    conditional_negating_swap_(should_swap, &b[0], &b[2]);
    conditional_negating_swap_(should_swap, &v[0], &v[2]);
    conditional_negating_swap_(should_swap, &b[3], &b[5]);
    conditional_negating_swap_(should_swap, &v[3], &v[5]);
    conditional_negating_swap_(should_swap, &b[6], &b[8]);
    conditional_negating_swap_(should_swap, &v[6], &v[8]);
    conditional_swap_(should_swap, &rho1, &rho3);
    should_swap = rho2 < rho3;
    conditional_negating_swap_(should_swap, &b[1], &b[2]);
    conditional_negating_swap_(should_swap, &v[1], &v[2]);
    conditional_negating_swap_(should_swap, &b[4], &b[5]);
    conditional_negating_swap_(should_swap, &v[4], &v[5]);
    conditional_negating_swap_(should_swap, &b[7], &b[8]);
    conditional_negating_swap_(should_swap, &v[7], &v[8]);
}

/** @brief QR decomposition of 3×3 matrix. */
template <typename scalar_type_>
void qr_decomposition_(scalar_type_ const *input, scalar_type_ *q, scalar_type_ *r) {
    scalar_type_ cos_half_1, sin_half_1;
    scalar_type_ cos_half_2, sin_half_2;
    scalar_type_ cos_half_3, sin_half_3;
    scalar_type_ cos_theta, sin_theta;
    scalar_type_ rotation_temp[9], matrix_temp[9];
    // First Givens rotation (zero input[3])
    qr_givens_quaternion_(input[0], input[3], &cos_half_1, &sin_half_1);
    cos_theta = scalar_type_(1.0) - scalar_type_(2.0) * sin_half_1 * sin_half_1;
    sin_theta = scalar_type_(2.0) * cos_half_1 * sin_half_1;
    rotation_temp[0] = cos_theta * input[0] + sin_theta * input[3];
    rotation_temp[1] = cos_theta * input[1] + sin_theta * input[4];
    rotation_temp[2] = cos_theta * input[2] + sin_theta * input[5];
    rotation_temp[3] = (scalar_type_(0.0) - sin_theta) * input[0] + cos_theta * input[3];
    rotation_temp[4] = (scalar_type_(0.0) - sin_theta) * input[1] + cos_theta * input[4];
    rotation_temp[5] = (scalar_type_(0.0) - sin_theta) * input[2] + cos_theta * input[5];
    rotation_temp[6] = input[6];
    rotation_temp[7] = input[7];
    rotation_temp[8] = input[8];
    // Second Givens rotation (zero rotation_temp[6])
    qr_givens_quaternion_(rotation_temp[0], rotation_temp[6], &cos_half_2, &sin_half_2);
    cos_theta = scalar_type_(1.0) - scalar_type_(2.0) * sin_half_2 * sin_half_2;
    sin_theta = scalar_type_(2.0) * cos_half_2 * sin_half_2;
    matrix_temp[0] = cos_theta * rotation_temp[0] + sin_theta * rotation_temp[6];
    matrix_temp[1] = cos_theta * rotation_temp[1] + sin_theta * rotation_temp[7];
    matrix_temp[2] = cos_theta * rotation_temp[2] + sin_theta * rotation_temp[8];
    matrix_temp[3] = rotation_temp[3];
    matrix_temp[4] = rotation_temp[4];
    matrix_temp[5] = rotation_temp[5];
    matrix_temp[6] = (scalar_type_(0.0) - sin_theta) * rotation_temp[0] + cos_theta * rotation_temp[6];
    matrix_temp[7] = (scalar_type_(0.0) - sin_theta) * rotation_temp[1] + cos_theta * rotation_temp[7];
    matrix_temp[8] = (scalar_type_(0.0) - sin_theta) * rotation_temp[2] + cos_theta * rotation_temp[8];
    // Third Givens rotation (zero matrix_temp[7])
    qr_givens_quaternion_(matrix_temp[4], matrix_temp[7], &cos_half_3, &sin_half_3);
    cos_theta = scalar_type_(1.0) - scalar_type_(2.0) * sin_half_3 * sin_half_3;
    sin_theta = scalar_type_(2.0) * cos_half_3 * sin_half_3;
    r[0] = matrix_temp[0];
    r[1] = matrix_temp[1];
    r[2] = matrix_temp[2];
    r[3] = cos_theta * matrix_temp[3] + sin_theta * matrix_temp[6];
    r[4] = cos_theta * matrix_temp[4] + sin_theta * matrix_temp[7];
    r[5] = cos_theta * matrix_temp[5] + sin_theta * matrix_temp[8];
    r[6] = (scalar_type_(0.0) - sin_theta) * matrix_temp[3] + cos_theta * matrix_temp[6];
    r[7] = (scalar_type_(0.0) - sin_theta) * matrix_temp[4] + cos_theta * matrix_temp[7];
    r[8] = (scalar_type_(0.0) - sin_theta) * matrix_temp[5] + cos_theta * matrix_temp[8];
    // Construct Q = Q1 * Q2 * Q3 (closed-form expressions)
    scalar_type_ sin_half_1_sq = sin_half_1 * sin_half_1;
    scalar_type_ sin_half_2_sq = sin_half_2 * sin_half_2;
    scalar_type_ sin_half_3_sq = sin_half_3 * sin_half_3;
    q[0] = (scalar_type_(-1.0) + scalar_type_(2.0) * sin_half_1_sq) *
           (scalar_type_(-1.0) + scalar_type_(2.0) * sin_half_2_sq);
    q[1] = scalar_type_(4.0) * cos_half_2 * cos_half_3 * (scalar_type_(-1.0) + scalar_type_(2.0) * sin_half_1_sq) *
               sin_half_2 * sin_half_3 +
           scalar_type_(2.0) * cos_half_1 * sin_half_1 * (scalar_type_(-1.0) + scalar_type_(2.0) * sin_half_3_sq);
    q[2] = scalar_type_(4.0) * cos_half_1 * cos_half_3 * sin_half_1 * sin_half_3 -
           scalar_type_(2.0) * cos_half_2 * (scalar_type_(-1.0) + scalar_type_(2.0) * sin_half_1_sq) * sin_half_2 *
               (scalar_type_(-1.0) + scalar_type_(2.0) * sin_half_3_sq);
    q[3] = scalar_type_(2.0) * cos_half_1 * sin_half_1 * (scalar_type_(1.0) - scalar_type_(2.0) * sin_half_2_sq);
    q[4] = scalar_type_(-8.0) * cos_half_1 * cos_half_2 * cos_half_3 * sin_half_1 * sin_half_2 * sin_half_3 +
           (scalar_type_(-1.0) + scalar_type_(2.0) * sin_half_1_sq) *
               (scalar_type_(-1.0) + scalar_type_(2.0) * sin_half_3_sq);
    q[5] = scalar_type_(-2.0) * cos_half_3 * sin_half_3 +
           scalar_type_(4.0) * sin_half_1 *
               (cos_half_3 * sin_half_1 * sin_half_3 +
                cos_half_1 * cos_half_2 * sin_half_2 * (scalar_type_(-1.0) + scalar_type_(2.0) * sin_half_3_sq));
    q[6] = scalar_type_(2.0) * cos_half_2 * sin_half_2;
    q[7] = scalar_type_(2.0) * cos_half_3 * (scalar_type_(1.0) - scalar_type_(2.0) * sin_half_2_sq) * sin_half_3;
    q[8] = (scalar_type_(-1.0) + scalar_type_(2.0) * sin_half_2_sq) *
           (scalar_type_(-1.0) + scalar_type_(2.0) * sin_half_3_sq);
}

/** @brief 3×3 SVD: A = U * S * V^scalar_type_ using McAdams algorithm. */
template <typename scalar_type_>
void svd3x3_(scalar_type_ const *a, scalar_type_ *svd_u, scalar_type_ *svd_s, scalar_type_ *svd_v) {
    // Compute Aᵀ * A (symmetric)
    scalar_type_ ata[9];
    ata[0] = a[0] * a[0] + a[3] * a[3] + a[6] * a[6];
    ata[1] = a[0] * a[1] + a[3] * a[4] + a[6] * a[7];
    ata[2] = a[0] * a[2] + a[3] * a[5] + a[6] * a[8];
    ata[3] = ata[1];
    ata[4] = a[1] * a[1] + a[4] * a[4] + a[7] * a[7];
    ata[5] = a[1] * a[2] + a[4] * a[5] + a[7] * a[8];
    ata[6] = ata[2];
    ata[7] = ata[5];
    ata[8] = a[2] * a[2] + a[5] * a[5] + a[8] * a[8];
    // Jacobi eigenanalysis of Aᵀ * A
    scalar_type_ quat[4];
    jacobi_eigenanalysis_(&ata[0], &ata[1], &ata[4], &ata[2], &ata[5], &ata[8], quat);
    quaternion_to_mat3x3_(quat, svd_v);
    // B = A * V
    scalar_type_ product[9];
    product[0] = a[0] * svd_v[0] + a[1] * svd_v[3] + a[2] * svd_v[6];
    product[1] = a[0] * svd_v[1] + a[1] * svd_v[4] + a[2] * svd_v[7];
    product[2] = a[0] * svd_v[2] + a[1] * svd_v[5] + a[2] * svd_v[8];
    product[3] = a[3] * svd_v[0] + a[4] * svd_v[3] + a[5] * svd_v[6];
    product[4] = a[3] * svd_v[1] + a[4] * svd_v[4] + a[5] * svd_v[7];
    product[5] = a[3] * svd_v[2] + a[4] * svd_v[5] + a[5] * svd_v[8];
    product[6] = a[6] * svd_v[0] + a[7] * svd_v[3] + a[8] * svd_v[6];
    product[7] = a[6] * svd_v[1] + a[7] * svd_v[4] + a[8] * svd_v[7];
    product[8] = a[6] * svd_v[2] + a[7] * svd_v[5] + a[8] * svd_v[8];
    // Sort singular values and update V
    sort_singular_values_(product, svd_v);
    // Compute singular values from column norms of sorted B
    scalar_type_ s1_sq = product[0] * product[0] + product[3] * product[3] + product[6] * product[6];
    scalar_type_ s2_sq = product[1] * product[1] + product[4] * product[4] + product[7] * product[7];
    scalar_type_ s3_sq = product[2] * product[2] + product[5] * product[5] + product[8] * product[8];
    // QR decomposition: B = U * R
    scalar_type_ qr_r[9];
    qr_decomposition_(product, svd_u, qr_r);
    // Store singular values in diagonal of svd_s
    svd_s[0] = s1_sq.sqrt();
    svd_s[1] = scalar_type_(0.0);
    svd_s[2] = scalar_type_(0.0);
    svd_s[3] = scalar_type_(0.0);
    svd_s[4] = s2_sq.sqrt();
    svd_s[5] = scalar_type_(0.0);
    svd_s[6] = scalar_type_(0.0);
    svd_s[7] = scalar_type_(0.0);
    svd_s[8] = s3_sq.sqrt();
}

#pragma endregion SVD Helpers for Scalar Fallbacks

#pragma region Mesh Alignment Kernels

/**
 *  @brief Root Mean Square Deviation between two 3D point clouds (no alignment)
 *  @tparam in_type_ Input point type (f32_t, f64_t, f16_t, bf16_t)
 *  @tparam result_type_ Result type for outputs
 *  @tparam allow_simd_ Enable SIMD kernel dispatch when true
 *
 *  @param a First point cloud [n x 3] interleaved (x0,y0,z0, x1,y1,z1, ...)
 *  @param b Second point cloud [n x 3] interleaved
 *  @param n Number of 3D points
 *  @param a_centroid Output centroid of first cloud (3 values), can be nullptr
 *  @param b_centroid Output centroid of second cloud (3 values), can be nullptr
 *  @param rotation Output 3x3 rotation matrix (9 values), always identity, can be nullptr
 *  @param scale Output scale factor, always 1.0, can be nullptr
 *  @param result Output RMSD value
 */
template <typename in_type_, typename result_type_ = typename in_type_::rmsd_result_t, bool allow_simd_ = true>
void rmsd(                                               //
    in_type_ const *a, in_type_ const *b, std::size_t n, //
    result_type_ *a_centroid, result_type_ *b_centroid,  //
    result_type_ *rotation, result_type_ *scale, result_type_ *result) noexcept {
    constexpr bool simd = allow_simd_ && std::is_same_v<result_type_, typename in_type_::rmsd_result_t>;

    if constexpr (std::is_same_v<in_type_, f64_t> && simd)
        nk_rmsd_f64(&a->raw_, &b->raw_, n, &a_centroid->raw_, &b_centroid->raw_, &rotation->raw_, &scale->raw_,
                    &result->raw_);
    else if constexpr (std::is_same_v<in_type_, f32_t> && simd)
        nk_rmsd_f32(&a->raw_, &b->raw_, n, &a_centroid->raw_, &b_centroid->raw_, &rotation->raw_, &scale->raw_,
                    &result->raw_);
    else if constexpr (std::is_same_v<in_type_, f16_t> && simd)
        nk_rmsd_f16(&a->raw_, &b->raw_, n, &a_centroid->raw_, &b_centroid->raw_, &rotation->raw_, &scale->raw_,
                    &result->raw_);
    else if constexpr (std::is_same_v<in_type_, bf16_t> && simd)
        nk_rmsd_bf16(&a->raw_, &b->raw_, n, &a_centroid->raw_, &b_centroid->raw_, &rotation->raw_,
                     scale ? &scale->raw_ : nullptr, &result->raw_);
    // Scalar fallback
    else {
        // Step 1: Compute centroids
        result_type_ sum_a_x {}, sum_a_y {}, sum_a_z {};
        result_type_ sum_b_x {}, sum_b_y {}, sum_b_z {};
        result_type_ val_a_x, val_a_y, val_a_z, val_b_x, val_b_y, val_b_z;

        for (std::size_t i = 0; i < n; i++) {
            val_a_x = result_type_(a[i * 3 + 0]);
            val_a_y = result_type_(a[i * 3 + 1]);
            val_a_z = result_type_(a[i * 3 + 2]);
            val_b_x = result_type_(b[i * 3 + 0]);
            val_b_y = result_type_(b[i * 3 + 1]);
            val_b_z = result_type_(b[i * 3 + 2]);
            sum_a_x = sum_a_x + val_a_x;
            sum_a_y = sum_a_y + val_a_y;
            sum_a_z = sum_a_z + val_a_z;
            sum_b_x = sum_b_x + val_b_x;
            sum_b_y = sum_b_y + val_b_y;
            sum_b_z = sum_b_z + val_b_z;
        }

        result_type_ inv_n = result_type_(1.0) / result_type_(static_cast<double>(n));
        result_type_ centroid_a_x = sum_a_x * inv_n;
        result_type_ centroid_a_y = sum_a_y * inv_n;
        result_type_ centroid_a_z = sum_a_z * inv_n;
        result_type_ centroid_b_x = sum_b_x * inv_n;
        result_type_ centroid_b_y = sum_b_y * inv_n;
        result_type_ centroid_b_z = sum_b_z * inv_n;

        // Step 2: Store centroids if requested
        if (a_centroid) a_centroid[0] = centroid_a_x, a_centroid[1] = centroid_a_y, a_centroid[2] = centroid_a_z;
        if (b_centroid) b_centroid[0] = centroid_b_x, b_centroid[1] = centroid_b_y, b_centroid[2] = centroid_b_z;

        // Step 3: RMSD uses identity rotation and scale=1.0
        if (rotation) {
            rotation[0] = result_type_(1.0);
            rotation[1] = result_type_(0.0);
            rotation[2] = result_type_(0.0);
            rotation[3] = result_type_(0.0);
            rotation[4] = result_type_(1.0);
            rotation[5] = result_type_(0.0);
            rotation[6] = result_type_(0.0);
            rotation[7] = result_type_(0.0);
            rotation[8] = result_type_(1.0);
        }
        if (scale) *scale = result_type_(1.0);

        // Step 4: Compute RMSD between centered point clouds
        result_type_ sum_squared {};
        for (std::size_t i = 0; i < n; i++) {
            val_a_x = result_type_(a[i * 3 + 0]);
            val_a_y = result_type_(a[i * 3 + 1]);
            val_a_z = result_type_(a[i * 3 + 2]);
            val_b_x = result_type_(b[i * 3 + 0]);
            val_b_y = result_type_(b[i * 3 + 1]);
            val_b_z = result_type_(b[i * 3 + 2]);
            result_type_ dx = (val_a_x - centroid_a_x) - (val_b_x - centroid_b_x);
            result_type_ dy = (val_a_y - centroid_a_y) - (val_b_y - centroid_b_y);
            result_type_ dz = (val_a_z - centroid_a_z) - (val_b_z - centroid_b_z);
            sum_squared = sum_squared + dx * dx + dy * dy + dz * dz;
        }

        *result = (sum_squared * inv_n).sqrt();
    }
}

/**
 *  @brief Kabsch algorithm for optimal rigid body alignment (rotation only)
 *  @tparam in_type_ Input point type (f32_t, f64_t, f16_t, bf16_t)
 *  @tparam result_type_ Result type for outputs
 *  @tparam allow_simd_ Enable SIMD kernel dispatch when true
 *
 *  @param a First point cloud [n x 3] interleaved (source)
 *  @param b Second point cloud [n x 3] interleaved (target)
 *  @param n Number of 3D points
 *  @param a_centroid Output centroid of first cloud (3 values), can be nullptr
 *  @param b_centroid Output centroid of second cloud (3 values), can be nullptr
 *  @param rotation Output 3x3 rotation matrix (9 values, row-major), can be nullptr
 *  @param scale Output scale factor, always 1.0 for Kabsch, can be nullptr
 *  @param result Output RMSD after optimal rotation
 */
template <typename in_type_, typename result_type_ = typename in_type_::rmsd_result_t, bool allow_simd_ = true>
void kabsch(                                             //
    in_type_ const *a, in_type_ const *b, std::size_t n, //
    result_type_ *a_centroid, result_type_ *b_centroid,  //
    result_type_ *rotation, result_type_ *scale, result_type_ *result) noexcept {
    constexpr bool simd = allow_simd_ && std::is_same_v<result_type_, typename in_type_::rmsd_result_t>;

    if constexpr (std::is_same_v<in_type_, f64_t> && simd)
        nk_kabsch_f64(&a->raw_, &b->raw_, n, a_centroid ? &a_centroid->raw_ : nullptr, &b_centroid->raw_,
                      &rotation->raw_, &scale->raw_, &result->raw_);
    else if constexpr (std::is_same_v<in_type_, f32_t> && simd)
        nk_kabsch_f32(&a->raw_, &b->raw_, n, &a_centroid->raw_, &b_centroid->raw_, &rotation->raw_, &scale->raw_,
                      &result->raw_);
    else if constexpr (std::is_same_v<in_type_, f16_t> && simd)
        nk_kabsch_f16(&a->raw_, &b->raw_, n, &a_centroid->raw_, &b_centroid->raw_, &rotation->raw_, &scale->raw_,
                      &result->raw_);
    else if constexpr (std::is_same_v<in_type_, bf16_t> && simd)
        nk_kabsch_bf16(&a->raw_, &b->raw_, n, &a_centroid->raw_, &b_centroid->raw_, &rotation->raw_, &scale->raw_,
                       &result->raw_);
    // Scalar fallback
    else {
        // Step 1: Compute centroids
        result_type_ sum_a_x {}, sum_a_y {}, sum_a_z {};
        result_type_ sum_b_x {}, sum_b_y {}, sum_b_z {};
        result_type_ val_a_x, val_a_y, val_a_z, val_b_x, val_b_y, val_b_z;

        for (std::size_t i = 0; i < n; i++) {
            val_a_x = result_type_(a[i * 3 + 0]);
            val_a_y = result_type_(a[i * 3 + 1]);
            val_a_z = result_type_(a[i * 3 + 2]);
            val_b_x = result_type_(b[i * 3 + 0]);
            val_b_y = result_type_(b[i * 3 + 1]);
            val_b_z = result_type_(b[i * 3 + 2]);
            sum_a_x = sum_a_x + val_a_x;
            sum_a_y = sum_a_y + val_a_y;
            sum_a_z = sum_a_z + val_a_z;
            sum_b_x = sum_b_x + val_b_x;
            sum_b_y = sum_b_y + val_b_y;
            sum_b_z = sum_b_z + val_b_z;
        }

        result_type_ inv_n = result_type_(1.0) / result_type_(static_cast<double>(n));
        result_type_ centroid_a_x = sum_a_x * inv_n;
        result_type_ centroid_a_y = sum_a_y * inv_n;
        result_type_ centroid_a_z = sum_a_z * inv_n;
        result_type_ centroid_b_x = sum_b_x * inv_n;
        result_type_ centroid_b_y = sum_b_y * inv_n;
        result_type_ centroid_b_z = sum_b_z * inv_n;

        if (a_centroid) a_centroid[0] = centroid_a_x, a_centroid[1] = centroid_a_y, a_centroid[2] = centroid_a_z;

        if (b_centroid) b_centroid[0] = centroid_b_x, b_centroid[1] = centroid_b_y, b_centroid[2] = centroid_b_z;

        // Step 2: Build 3×3 covariance matrix H = (A - Ā)ᵀ × (B - B̄)
        result_type_ cross_covariance[9] = {};
        for (std::size_t i = 0; i < n; i++) {
            val_a_x = result_type_(a[i * 3 + 0]) - centroid_a_x;
            val_a_y = result_type_(a[i * 3 + 1]) - centroid_a_y;
            val_a_z = result_type_(a[i * 3 + 2]) - centroid_a_z;
            val_b_x = result_type_(b[i * 3 + 0]) - centroid_b_x;
            val_b_y = result_type_(b[i * 3 + 1]) - centroid_b_y;
            val_b_z = result_type_(b[i * 3 + 2]) - centroid_b_z;
            cross_covariance[0] = cross_covariance[0] + val_a_x * val_b_x;
            cross_covariance[1] = cross_covariance[1] + val_a_x * val_b_y;
            cross_covariance[2] = cross_covariance[2] + val_a_x * val_b_z;
            cross_covariance[3] = cross_covariance[3] + val_a_y * val_b_x;
            cross_covariance[4] = cross_covariance[4] + val_a_y * val_b_y;
            cross_covariance[5] = cross_covariance[5] + val_a_y * val_b_z;
            cross_covariance[6] = cross_covariance[6] + val_a_z * val_b_x;
            cross_covariance[7] = cross_covariance[7] + val_a_z * val_b_y;
            cross_covariance[8] = cross_covariance[8] + val_a_z * val_b_z;
        }

        // Step 3: SVD of H = U * S * Vᵀ
        result_type_ svd_u[9], svd_s[9], svd_v[9];
        svd3x3_(cross_covariance, svd_u, svd_s, svd_v);

        // Step 4: R = V * Uᵀ
        result_type_ rotation_matrix[9];
        rotation_matrix[0] = svd_v[0] * svd_u[0] + svd_v[1] * svd_u[1] + svd_v[2] * svd_u[2];
        rotation_matrix[1] = svd_v[0] * svd_u[3] + svd_v[1] * svd_u[4] + svd_v[2] * svd_u[5];
        rotation_matrix[2] = svd_v[0] * svd_u[6] + svd_v[1] * svd_u[7] + svd_v[2] * svd_u[8];
        rotation_matrix[3] = svd_v[3] * svd_u[0] + svd_v[4] * svd_u[1] + svd_v[5] * svd_u[2];
        rotation_matrix[4] = svd_v[3] * svd_u[3] + svd_v[4] * svd_u[4] + svd_v[5] * svd_u[5];
        rotation_matrix[5] = svd_v[3] * svd_u[6] + svd_v[4] * svd_u[7] + svd_v[5] * svd_u[8];
        rotation_matrix[6] = svd_v[6] * svd_u[0] + svd_v[7] * svd_u[1] + svd_v[8] * svd_u[2];
        rotation_matrix[7] = svd_v[6] * svd_u[3] + svd_v[7] * svd_u[4] + svd_v[8] * svd_u[5];
        rotation_matrix[8] = svd_v[6] * svd_u[6] + svd_v[7] * svd_u[7] + svd_v[8] * svd_u[8];

        // Handle reflection: if det(R) < 0, negate third column of V and recompute R
        result_type_ rotation_det = det3x3_(rotation_matrix);
        if (rotation_det < result_type_(0.0)) {
            svd_v[2] = result_type_(0.0) - svd_v[2];
            svd_v[5] = result_type_(0.0) - svd_v[5];
            svd_v[8] = result_type_(0.0) - svd_v[8];
            rotation_matrix[0] = svd_v[0] * svd_u[0] + svd_v[1] * svd_u[1] + svd_v[2] * svd_u[2];
            rotation_matrix[1] = svd_v[0] * svd_u[3] + svd_v[1] * svd_u[4] + svd_v[2] * svd_u[5];
            rotation_matrix[2] = svd_v[0] * svd_u[6] + svd_v[1] * svd_u[7] + svd_v[2] * svd_u[8];
            rotation_matrix[3] = svd_v[3] * svd_u[0] + svd_v[4] * svd_u[1] + svd_v[5] * svd_u[2];
            rotation_matrix[4] = svd_v[3] * svd_u[3] + svd_v[4] * svd_u[4] + svd_v[5] * svd_u[5];
            rotation_matrix[5] = svd_v[3] * svd_u[6] + svd_v[4] * svd_u[7] + svd_v[5] * svd_u[8];
            rotation_matrix[6] = svd_v[6] * svd_u[0] + svd_v[7] * svd_u[1] + svd_v[8] * svd_u[2];
            rotation_matrix[7] = svd_v[6] * svd_u[3] + svd_v[7] * svd_u[4] + svd_v[8] * svd_u[5];
            rotation_matrix[8] = svd_v[6] * svd_u[6] + svd_v[7] * svd_u[7] + svd_v[8] * svd_u[8];
        }

        // Output rotation matrix and scale=1.0
        if (rotation) {
            for (unsigned int j = 0; j < 9; j++) rotation[j] = rotation_matrix[j];
        }
        if (scale) *scale = result_type_(1.0);

        // Step 5: Compute RMSD after rotation
        result_type_ sum_squared {};
        for (std::size_t i = 0; i < n; i++) {
            result_type_ point_a[3], point_b[3], rotated_point_a[3];
            point_a[0] = result_type_(a[i * 3 + 0]) - centroid_a_x;
            point_a[1] = result_type_(a[i * 3 + 1]) - centroid_a_y;
            point_a[2] = result_type_(a[i * 3 + 2]) - centroid_a_z;
            point_b[0] = result_type_(b[i * 3 + 0]) - centroid_b_x;
            point_b[1] = result_type_(b[i * 3 + 1]) - centroid_b_y;
            point_b[2] = result_type_(b[i * 3 + 2]) - centroid_b_z;
            rotated_point_a[0] = rotation_matrix[0] * point_a[0] + rotation_matrix[1] * point_a[1] +
                                 rotation_matrix[2] * point_a[2];
            rotated_point_a[1] = rotation_matrix[3] * point_a[0] + rotation_matrix[4] * point_a[1] +
                                 rotation_matrix[5] * point_a[2];
            rotated_point_a[2] = rotation_matrix[6] * point_a[0] + rotation_matrix[7] * point_a[1] +
                                 rotation_matrix[8] * point_a[2];
            result_type_ dx = rotated_point_a[0] - point_b[0];
            result_type_ dy = rotated_point_a[1] - point_b[1];
            result_type_ dz = rotated_point_a[2] - point_b[2];
            sum_squared = sum_squared + dx * dx + dy * dy + dz * dz;
        }

        *result = (sum_squared * inv_n).sqrt();
    }
}

/**
 *  @brief Umeyama algorithm for similarity transform (rotation + uniform scale)
 *  @tparam in_type_ Input point type (f32_t, f64_t, f16_t, bf16_t)
 *  @tparam result_type_ Result type for outputs
 *  @tparam allow_simd_ Enable SIMD kernel dispatch when true
 *
 *  @param a First point cloud [n x 3] interleaved (source)
 *  @param b Second point cloud [n x 3] interleaved (target)
 *  @param n Number of 3D points
 *  @param a_centroid Output centroid of first cloud (3 values), can be nullptr
 *  @param b_centroid Output centroid of second cloud (3 values), can be nullptr
 *  @param rotation Output 3x3 rotation matrix (9 values, row-major), can be nullptr
 *  @param scale Output uniform scale factor, can be nullptr
 *  @param result Output RMSD after optimal transformation
 */
template <typename in_type_, typename result_type_ = typename in_type_::dot_result_t, bool allow_simd_ = true>
void umeyama(in_type_ const *a, in_type_ const *b, std::size_t n, result_type_ *a_centroid, result_type_ *b_centroid,
             result_type_ *rotation, result_type_ *scale, result_type_ *result) noexcept {
    constexpr bool simd = allow_simd_ && std::is_same_v<result_type_, typename in_type_::rmsd_result_t>;

    if constexpr (std::is_same_v<in_type_, f64_t> && simd)
        nk_umeyama_f64(&a->raw_, &b->raw_, n, &a_centroid->raw_, &b_centroid->raw_, &rotation->raw_, &scale->raw_,
                       &result->raw_);
    else if constexpr (std::is_same_v<in_type_, f32_t> && simd)
        nk_umeyama_f32(&a->raw_, &b->raw_, n, &a_centroid->raw_, &b_centroid->raw_, &rotation->raw_, &scale->raw_,
                       &result->raw_);
    else if constexpr (std::is_same_v<in_type_, f16_t> && simd)
        nk_umeyama_f16(&a->raw_, &b->raw_, n, &a_centroid->raw_, &b_centroid->raw_, &rotation->raw_, &scale->raw_,
                       &result->raw_);
    else if constexpr (std::is_same_v<in_type_, bf16_t> && simd)
        nk_umeyama_bf16(&a->raw_, &b->raw_, n, &a_centroid->raw_, &b_centroid->raw_, &rotation->raw_, &scale->raw_,
                        &result->raw_);
    // Scalar fallback
    else {
        // Step 1: Compute centroids
        result_type_ sum_a_x {}, sum_a_y {}, sum_a_z {};
        result_type_ sum_b_x {}, sum_b_y {}, sum_b_z {};
        result_type_ val_a_x, val_a_y, val_a_z, val_b_x, val_b_y, val_b_z;

        for (std::size_t i = 0; i < n; i++) {
            val_a_x = result_type_(a[i * 3 + 0]);
            val_a_y = result_type_(a[i * 3 + 1]);
            val_a_z = result_type_(a[i * 3 + 2]);
            val_b_x = result_type_(b[i * 3 + 0]);
            val_b_y = result_type_(b[i * 3 + 1]);
            val_b_z = result_type_(b[i * 3 + 2]);
            sum_a_x = sum_a_x + val_a_x;
            sum_a_y = sum_a_y + val_a_y;
            sum_a_z = sum_a_z + val_a_z;
            sum_b_x = sum_b_x + val_b_x;
            sum_b_y = sum_b_y + val_b_y;
            sum_b_z = sum_b_z + val_b_z;
        }

        result_type_ inv_n = result_type_(1.0) / result_type_(static_cast<double>(n));
        result_type_ centroid_a_x = sum_a_x * inv_n;
        result_type_ centroid_a_y = sum_a_y * inv_n;
        result_type_ centroid_a_z = sum_a_z * inv_n;
        result_type_ centroid_b_x = sum_b_x * inv_n;
        result_type_ centroid_b_y = sum_b_y * inv_n;
        result_type_ centroid_b_z = sum_b_z * inv_n;

        if (a_centroid) {
            a_centroid[0] = centroid_a_x;
            a_centroid[1] = centroid_a_y;
            a_centroid[2] = centroid_a_z;
        }
        if (b_centroid) {
            b_centroid[0] = centroid_b_x;
            b_centroid[1] = centroid_b_y;
            b_centroid[2] = centroid_b_z;
        }

        // Step 2: Build covariance matrix H and compute variance of A
        result_type_ cross_covariance[9] = {};
        result_type_ variance_a {};
        for (std::size_t i = 0; i < n; i++) {
            val_a_x = result_type_(a[i * 3 + 0]) - centroid_a_x;
            val_a_y = result_type_(a[i * 3 + 1]) - centroid_a_y;
            val_a_z = result_type_(a[i * 3 + 2]) - centroid_a_z;
            val_b_x = result_type_(b[i * 3 + 0]) - centroid_b_x;
            val_b_y = result_type_(b[i * 3 + 1]) - centroid_b_y;
            val_b_z = result_type_(b[i * 3 + 2]) - centroid_b_z;
            variance_a = variance_a + val_a_x * val_a_x + val_a_y * val_a_y + val_a_z * val_a_z;
            cross_covariance[0] = cross_covariance[0] + val_a_x * val_b_x;
            cross_covariance[1] = cross_covariance[1] + val_a_x * val_b_y;
            cross_covariance[2] = cross_covariance[2] + val_a_x * val_b_z;
            cross_covariance[3] = cross_covariance[3] + val_a_y * val_b_x;
            cross_covariance[4] = cross_covariance[4] + val_a_y * val_b_y;
            cross_covariance[5] = cross_covariance[5] + val_a_y * val_b_z;
            cross_covariance[6] = cross_covariance[6] + val_a_z * val_b_x;
            cross_covariance[7] = cross_covariance[7] + val_a_z * val_b_y;
            cross_covariance[8] = cross_covariance[8] + val_a_z * val_b_z;
        }
        variance_a = variance_a * inv_n;

        // Step 3: SVD of H = U * S * Vᵀ
        result_type_ svd_u[9], svd_s[9], svd_v[9];
        svd3x3_(cross_covariance, svd_u, svd_s, svd_v);

        // Step 4: R = V * Uᵀ
        result_type_ rotation_matrix[9];
        rotation_matrix[0] = svd_v[0] * svd_u[0] + svd_v[1] * svd_u[1] + svd_v[2] * svd_u[2];
        rotation_matrix[1] = svd_v[0] * svd_u[3] + svd_v[1] * svd_u[4] + svd_v[2] * svd_u[5];
        rotation_matrix[2] = svd_v[0] * svd_u[6] + svd_v[1] * svd_u[7] + svd_v[2] * svd_u[8];
        rotation_matrix[3] = svd_v[3] * svd_u[0] + svd_v[4] * svd_u[1] + svd_v[5] * svd_u[2];
        rotation_matrix[4] = svd_v[3] * svd_u[3] + svd_v[4] * svd_u[4] + svd_v[5] * svd_u[5];
        rotation_matrix[5] = svd_v[3] * svd_u[6] + svd_v[4] * svd_u[7] + svd_v[5] * svd_u[8];
        rotation_matrix[6] = svd_v[6] * svd_u[0] + svd_v[7] * svd_u[1] + svd_v[8] * svd_u[2];
        rotation_matrix[7] = svd_v[6] * svd_u[3] + svd_v[7] * svd_u[4] + svd_v[8] * svd_u[5];
        rotation_matrix[8] = svd_v[6] * svd_u[6] + svd_v[7] * svd_u[7] + svd_v[8] * svd_u[8];

        // Handle reflection and compute scale: c = trace(D*S) / variance_a
        // D = diag(1, 1, det(R)), svd_s contains singular values on diagonal
        result_type_ rotation_det = det3x3_(rotation_matrix);
        result_type_ sign_det = rotation_det < result_type_(0.0) ? result_type_(-1.0) : result_type_(1.0);
        result_type_ trace_scaled_s = svd_s[0] + svd_s[4] + sign_det * svd_s[8];
        result_type_ scale_factor = trace_scaled_s / (result_type_(static_cast<double>(n)) * variance_a);

        if (scale) *scale = scale_factor;

        if (rotation_det < result_type_(0.0)) {
            svd_v[2] = result_type_(0.0) - svd_v[2];
            svd_v[5] = result_type_(0.0) - svd_v[5];
            svd_v[8] = result_type_(0.0) - svd_v[8];
            rotation_matrix[0] = svd_v[0] * svd_u[0] + svd_v[1] * svd_u[1] + svd_v[2] * svd_u[2];
            rotation_matrix[1] = svd_v[0] * svd_u[3] + svd_v[1] * svd_u[4] + svd_v[2] * svd_u[5];
            rotation_matrix[2] = svd_v[0] * svd_u[6] + svd_v[1] * svd_u[7] + svd_v[2] * svd_u[8];
            rotation_matrix[3] = svd_v[3] * svd_u[0] + svd_v[4] * svd_u[1] + svd_v[5] * svd_u[2];
            rotation_matrix[4] = svd_v[3] * svd_u[3] + svd_v[4] * svd_u[4] + svd_v[5] * svd_u[5];
            rotation_matrix[5] = svd_v[3] * svd_u[6] + svd_v[4] * svd_u[7] + svd_v[5] * svd_u[8];
            rotation_matrix[6] = svd_v[6] * svd_u[0] + svd_v[7] * svd_u[1] + svd_v[8] * svd_u[2];
            rotation_matrix[7] = svd_v[6] * svd_u[3] + svd_v[7] * svd_u[4] + svd_v[8] * svd_u[5];
            rotation_matrix[8] = svd_v[6] * svd_u[6] + svd_v[7] * svd_u[7] + svd_v[8] * svd_u[8];
        }

        // Output rotation matrix
        if (rotation) {
            for (unsigned int j = 0; j < 9; j++) rotation[j] = rotation_matrix[j];
        }

        // Step 5: Compute RMSD after similarity transform: ||c*R*a - b||
        result_type_ sum_squared {};
        for (std::size_t i = 0; i < n; i++) {
            result_type_ point_a[3], point_b[3], rotated_point_a[3];
            point_a[0] = result_type_(a[i * 3 + 0]) - centroid_a_x;
            point_a[1] = result_type_(a[i * 3 + 1]) - centroid_a_y;
            point_a[2] = result_type_(a[i * 3 + 2]) - centroid_a_z;
            point_b[0] = result_type_(b[i * 3 + 0]) - centroid_b_x;
            point_b[1] = result_type_(b[i * 3 + 1]) - centroid_b_y;
            point_b[2] = result_type_(b[i * 3 + 2]) - centroid_b_z;
            rotated_point_a[0] = scale_factor * (rotation_matrix[0] * point_a[0] + rotation_matrix[1] * point_a[1] +
                                                 rotation_matrix[2] * point_a[2]);
            rotated_point_a[1] = scale_factor * (rotation_matrix[3] * point_a[0] + rotation_matrix[4] * point_a[1] +
                                                 rotation_matrix[5] * point_a[2]);
            rotated_point_a[2] = scale_factor * (rotation_matrix[6] * point_a[0] + rotation_matrix[7] * point_a[1] +
                                                 rotation_matrix[8] * point_a[2]);
            result_type_ dx = rotated_point_a[0] - point_b[0];
            result_type_ dy = rotated_point_a[1] - point_b[1];
            result_type_ dz = rotated_point_a[2] - point_b[2];
            sum_squared = sum_squared + dx * dx + dy * dy + dz * dz;
        }

        *result = (sum_squared * inv_n).sqrt();
    }
}

#pragma endregion Mesh Alignment Kernels

#pragma region Trigonometry Kernels

/**
 *  @brief Array sine: out[i] = sin(in[i])
 *  @tparam in_type_ Element type (f32_t, f64_t, f16_t)
 *  @tparam allow_simd_ Enable SIMD kernel dispatch
 */
template <typename in_type_, typename precision_type_ = in_type_, bool allow_simd_ = true>
void sin(in_type_ const *in, std::size_t n, in_type_ *out) noexcept {
    constexpr bool simd = allow_simd_ && std::is_same_v<in_type_, precision_type_>;

    if constexpr (std::is_same_v<in_type_, f64_t> && simd) nk_sin_f64(&in->raw_, n, &out->raw_);
    else if constexpr (std::is_same_v<in_type_, f32_t> && simd) nk_sin_f32(&in->raw_, n, &out->raw_);
    else if constexpr (std::is_same_v<in_type_, f16_t> && simd) nk_sin_f16(&in->raw_, n, &out->raw_);
    // Scalar fallback
    else {
        for (std::size_t i = 0; i < n; i++) out[i] = precision_type_(in[i]).sin().template to<in_type_>();
    }
}

/**
 *  @brief Array cosine: out[i] = cos(in[i])
 *  @tparam in_type_ Element type (f32_t, f64_t, f16_t)
 *  @tparam allow_simd_ Enable SIMD kernel dispatch
 */
template <typename in_type_, typename precision_type_ = in_type_, bool allow_simd_ = true>
void cos(in_type_ const *in, std::size_t n, in_type_ *out) noexcept {
    constexpr bool simd = allow_simd_ && std::is_same_v<in_type_, precision_type_>;

    if constexpr (std::is_same_v<in_type_, f64_t> && simd) nk_cos_f64(&in->raw_, n, &out->raw_);
    else if constexpr (std::is_same_v<in_type_, f32_t> && simd) nk_cos_f32(&in->raw_, n, &out->raw_);
    else if constexpr (std::is_same_v<in_type_, f16_t> && simd) nk_cos_f16(&in->raw_, n, &out->raw_);
    // Scalar fallback
    else {
        for (std::size_t i = 0; i < n; i++) out[i] = precision_type_(in[i]).cos().template to<in_type_>();
    }
}

/**
 *  @brief Array atan: out[i] = atan(in[i])
 *  @tparam in_type_ Element type (f32_t, f64_t, f16_t)
 *  @tparam allow_simd_ Enable SIMD kernel dispatch
 */
template <typename in_type_, typename precision_type_ = in_type_, bool allow_simd_ = true>
void atan(in_type_ const *in, std::size_t n, in_type_ *out) noexcept {
    constexpr bool simd = allow_simd_ && std::is_same_v<in_type_, precision_type_>;

    if constexpr (std::is_same_v<in_type_, f64_t> && simd) nk_atan_f64(&in->raw_, n, &out->raw_);
    else if constexpr (std::is_same_v<in_type_, f32_t> && simd) nk_atan_f32(&in->raw_, n, &out->raw_);
    else if constexpr (std::is_same_v<in_type_, f16_t> && simd) nk_atan_f16(&in->raw_, n, &out->raw_);
    // Scalar fallback
    else {
        for (std::size_t i = 0; i < n; i++) out[i] = precision_type_(in[i]).atan().template to<in_type_>();
    }
}

#pragma endregion Trigonometry Kernels

#pragma region Dots Kernels

/**
 *  @brief Estimates the memory requirements for packed B matrix.
 *  @return Size in bytes for column-major B data plus stride metadata
 */
template <typename in_type_, bool allow_simd_ = true>
NK_PUBLIC size_t dots_packed_size(size_t column_count, size_t depth) {
    if constexpr (std::is_same_v<in_type_, f64_t> && allow_simd_) return nk_dots_packed_size_f64(column_count, depth);
    else if constexpr (std::is_same_v<in_type_, f32_t> && allow_simd_)
        return nk_dots_packed_size_f32(column_count, depth);
    else if constexpr (std::is_same_v<in_type_, f16_t> && allow_simd_)
        return nk_dots_packed_size_f16(column_count, depth);
    else if constexpr (std::is_same_v<in_type_, bf16_t> && allow_simd_)
        return nk_dots_packed_size_bf16(column_count, depth);
    else if constexpr (std::is_same_v<in_type_, i8_t> && allow_simd_)
        return nk_dots_packed_size_i8(column_count, depth);
    else if constexpr (std::is_same_v<in_type_, u8_t> && allow_simd_)
        return nk_dots_packed_size_u8(column_count, depth);
    else if constexpr (std::is_same_v<in_type_, e4m3_t> && allow_simd_)
        return nk_dots_packed_size_e4m3(column_count, depth);
    else if constexpr (std::is_same_v<in_type_, e5m2_t> && allow_simd_)
        return nk_dots_packed_size_e5m2(column_count, depth);
    else if constexpr (std::is_same_v<in_type_, u1x8_t> && allow_simd_)
        return nk_dots_packed_size_u1(column_count, depth);
    else if constexpr (std::is_same_v<in_type_, u4x2_t> && allow_simd_)
        return nk_dots_packed_size_u4(column_count, depth);
    else if constexpr (std::is_same_v<in_type_, i4x2_t> && allow_simd_)
        return nk_dots_packed_size_i4(column_count, depth);
    else return column_count * depth * sizeof(typename in_type_::raw_t) + sizeof(size_t);
}

/**
 *  @brief Packs matrix B into column-major form for efficient dots_packed access.
 *  @param b Input matrix B in row-major form [column_count x depth]
 *  @param column_count Number of columns in B (n)
 *  @param depth Number of elements per column (k)
 *  @param b_stride_in_bytes Stride between rows of B in bytes
 *  @param b_packed Output buffer for packed column-major B with metadata
 */
template <typename in_type_, bool allow_simd_ = true>
NK_PUBLIC void dots_pack(in_type_ const *b, size_t column_count, size_t depth, size_t b_stride_in_bytes,
                         void *b_packed) {
    using raw_t = typename in_type_::raw_t;
    if constexpr (std::is_same_v<in_type_, f64_t> && allow_simd_)
        nk_dots_pack_f64(reinterpret_cast<raw_t const *>(b), column_count, depth, b_stride_in_bytes, b_packed);
    else if constexpr (std::is_same_v<in_type_, f32_t> && allow_simd_)
        nk_dots_pack_f32(reinterpret_cast<raw_t const *>(b), column_count, depth, b_stride_in_bytes, b_packed);
    else if constexpr (std::is_same_v<in_type_, f16_t> && allow_simd_)
        nk_dots_pack_f16(reinterpret_cast<raw_t const *>(b), column_count, depth, b_stride_in_bytes, b_packed);
    else if constexpr (std::is_same_v<in_type_, bf16_t> && allow_simd_)
        nk_dots_pack_bf16(reinterpret_cast<raw_t const *>(b), column_count, depth, b_stride_in_bytes, b_packed);
    else if constexpr (std::is_same_v<in_type_, i8_t> && allow_simd_)
        nk_dots_pack_i8(reinterpret_cast<raw_t const *>(b), column_count, depth, b_stride_in_bytes, b_packed);
    else if constexpr (std::is_same_v<in_type_, u8_t> && allow_simd_)
        nk_dots_pack_u8(reinterpret_cast<raw_t const *>(b), column_count, depth, b_stride_in_bytes, b_packed);
    else if constexpr (std::is_same_v<in_type_, e4m3_t> && allow_simd_)
        nk_dots_pack_e4m3(reinterpret_cast<raw_t const *>(b), column_count, depth, b_stride_in_bytes, b_packed);
    else if constexpr (std::is_same_v<in_type_, e5m2_t> && allow_simd_)
        nk_dots_pack_e5m2(reinterpret_cast<raw_t const *>(b), column_count, depth, b_stride_in_bytes, b_packed);
    else if constexpr (std::is_same_v<in_type_, u1x8_t> && allow_simd_)
        nk_dots_pack_u1(reinterpret_cast<raw_t const *>(b), column_count, depth, b_stride_in_bytes, b_packed);
    else if constexpr (std::is_same_v<in_type_, u4x2_t> && allow_simd_)
        nk_dots_pack_u4(reinterpret_cast<raw_t const *>(b), column_count, depth, b_stride_in_bytes, b_packed);
    else if constexpr (std::is_same_v<in_type_, i4x2_t> && allow_simd_)
        nk_dots_pack_i4(reinterpret_cast<raw_t const *>(b), column_count, depth, b_stride_in_bytes, b_packed);
    else {
        // Transpose B from row-major [column_count x depth] to column-major
        char const *b_bytes = reinterpret_cast<char const *>(b);
        in_type_ *out = static_cast<in_type_ *>(b_packed);
        for (size_t j = 0; j < column_count; j++) {
            in_type_ const *b_row = reinterpret_cast<in_type_ const *>(b_bytes + j * b_stride_in_bytes);
            for (size_t l = 0; l < depth; l++) out[j * depth + l] = b_row[l];
        }
        // Append stride metadata
        size_t packed_stride = depth * sizeof(raw_t);
        size_t bytes_for_data = column_count * depth * sizeof(raw_t);
        char *metadata_ptr = static_cast<char *>(b_packed) + bytes_for_data;
        std::memcpy(metadata_ptr, &packed_stride, sizeof(size_t));
    }
}

/**
 *  @brief Packed dot products (batch matrix multiply): C = A * B (row-major)
 *  @param a Matrix A [m x k]
 *  @param b_packed Packed matrix B [k x n] with stride metadata appended
 *  @param c Output matrix C [m x n]
 *  @param row_count Rows of A and C (m)
 *  @param column_count Columns of B and C (n)
 *  @param depth Columns of A, Rows of B (k)
 *  @param a_stride_in_bytes Stride between rows of A in bytes
 *  @param c_stride_in_bytes Stride between rows of C in bytes
 *  @tparam in_type_ Input element type
 *  @tparam result_type_ Accumulator/output type
 */
template <typename in_type_, typename result_type_ = typename in_type_::dot_result_t, bool allow_simd_ = true>
void dots_packed(in_type_ const *a, void const *b_packed, result_type_ *c, std::size_t row_count,
                 std::size_t column_count, std::size_t depth, std::size_t a_stride_in_bytes,
                 std::size_t c_stride_in_bytes) noexcept {
    using raw_t = typename in_type_::raw_t;
    constexpr bool dispatch = allow_simd_ && std::is_same_v<result_type_, typename in_type_::dot_result_t>;
    if constexpr (std::is_same_v<in_type_, f64_t> && dispatch)
        nk_dots_packed_f64(&a->raw_, b_packed, c, row_count, column_count, depth, a_stride_in_bytes, c_stride_in_bytes);
    else if constexpr (std::is_same_v<in_type_, f32_t> && dispatch)
        nk_dots_packed_f32(&a->raw_, b_packed, c, row_count, column_count, depth, a_stride_in_bytes, c_stride_in_bytes);
    else if constexpr (std::is_same_v<in_type_, f16_t> && dispatch)
        nk_dots_packed_f16(&a->raw_, b_packed, c, row_count, column_count, depth, a_stride_in_bytes, c_stride_in_bytes);
    else if constexpr (std::is_same_v<in_type_, bf16_t> && dispatch)
        nk_dots_packed_bf16(&a->raw_, b_packed, c, row_count, column_count, depth, a_stride_in_bytes,
                            c_stride_in_bytes);
    else if constexpr (std::is_same_v<in_type_, i8_t> && dispatch)
        nk_dots_packed_i8(&a->raw_, b_packed, c, row_count, column_count, depth, a_stride_in_bytes, c_stride_in_bytes);
    else if constexpr (std::is_same_v<in_type_, u8_t> && dispatch)
        nk_dots_packed_u8(&a->raw_, b_packed, c, row_count, column_count, depth, a_stride_in_bytes, c_stride_in_bytes);
    else if constexpr (std::is_same_v<in_type_, e4m3_t> && dispatch)
        nk_dots_packed_e4m3(&a->raw_, b_packed, c, row_count, column_count, depth, a_stride_in_bytes,
                            c_stride_in_bytes);
    else if constexpr (std::is_same_v<in_type_, e5m2_t> && dispatch)
        nk_dots_packed_e5m2(&a->raw_, b_packed, c, row_count, column_count, depth, a_stride_in_bytes,
                            c_stride_in_bytes);
    else if constexpr (std::is_same_v<in_type_, u1x8_t> && dispatch)
        nk_dots_packed_u1(&a->raw_, b_packed, c, row_count, column_count, depth, a_stride_in_bytes, c_stride_in_bytes);
    else if constexpr (std::is_same_v<in_type_, u4x2_t> && dispatch)
        nk_dots_packed_u4(&a->raw_, b_packed, c, row_count, column_count, depth, a_stride_in_bytes, c_stride_in_bytes);
    else if constexpr (std::is_same_v<in_type_, i4x2_t> && dispatch)
        nk_dots_packed_i4(&a->raw_, b_packed, c, row_count, column_count, depth, a_stride_in_bytes, c_stride_in_bytes);
    else {
        // Extract b_stride from the end of packed data
        std::size_t bytes_for_b = column_count * depth * sizeof(raw_t);
        char const *metadata_ptr = static_cast<char const *>(b_packed) + bytes_for_b;
        std::size_t b_stride_in_bytes;
        std::memcpy(&b_stride_in_bytes, metadata_ptr, sizeof(std::size_t));

        // Use byte pointers for strided access
        char const *a_bytes = reinterpret_cast<char const *>(a);
        char const *b_bytes = static_cast<char const *>(b_packed);
        char *c_bytes = reinterpret_cast<char *>(c);

        for (std::size_t i = 0; i < row_count; i++) {
            in_type_ const *a_row = reinterpret_cast<in_type_ const *>(a_bytes + i * a_stride_in_bytes);
            result_type_ *c_row = reinterpret_cast<result_type_ *>(c_bytes + i * c_stride_in_bytes);
            for (std::size_t j = 0; j < column_count; j++) {
                in_type_ const *b_col = reinterpret_cast<in_type_ const *>(b_bytes + j * b_stride_in_bytes);
                result_type_ sum {};
                for (std::size_t l = 0; l < depth; l++) sum = fused_multiply_add(sum, a_row[l], b_col[l]);
                c_row[j] = sum;
            }
        }
    }
}

#pragma endregion Dots Kernels

} // namespace ashvardanian::numkong

#endif // NK_NUMKONG_HPP

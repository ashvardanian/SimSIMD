/**
 *  @brief NumKong strong types for C++23 and newer.
 *  @file include/numkong/types.hpp
 *  @author Ash Vardanian
 *  @date January 7, 2026
 *
 *  C doesn't have a strong type system or composable infrastructure for complex kernels
 *  and datastructures like the C++ templates and Rust traits. Moreover, C++ is the dominant
 *  language in High-Performance Computing and now NumKong exposes all of its unusual types
 *  to C++ users with support for all traditional `operator`s, compatible with `std::mdspan`.
 *
 *  This file several light-weight mostly `constexpr` classes:
 *
 *  1. Wrappers for traditional types, like `f32_t`, `i16_t`, `u8_t`, and complex `f64c_t`
 *  2. Low-precision machine-learning types, like `bf16_t`, `f16_t`, `e4m3_t`, and `e5m2_t`
 *  3. Sub-byte types packed into tuples, like `u1x8_t`, `u4x2_t`, `i4x2_t`
 *  4. Extreme-precision types for accuracy tests, like `f118_t` via double-double logic
 *
 *  Each of those contains:
 *
 *  - Traditional arithmetic operator overloads returning the same storage type
 *  - Trigonometric functions like the `sin`, `tanh`, `acos`
 *  - Rust-style `total_cmp`, `round`, `to_radians`, `mul_add` for FMA operations
 *  - Type definitions for NumKong mixed-precision often-widening operations
 *
 *  @section terminology Terminology: Word vs Value vs Dimension
 *
 *  Each type defines three bit-width constants that describe its memory layout:
 *
 *  - @b value: A C++ container element, like the `i4x2_t`, `f32_t`, and `f64c_t`.
 *
 *  - @b word: A raw Assembly-level storage unit, like a single `unsigned char` under
 *    the `i4x2_t` or one of the two `double`s forming the `f64c_t`.
 *
 *  - @b dimension: A logical unit of work, like the a single nibble of `i4x2_t`, or
 *    a single bit of `u1x8_t`. For complex numbers, it would be a pair of floating-point values.
 *
 *  In a tabular form, with more examples, it would look like:
 *
 *      Type       bits/word   bits/dimension   bits/value   dims/value   words/value
 *      f32_t         32             32             32           1             1
 *      f32c_t        32             64             64           1             2
 *      u1x8_t         8              1              8           8             1
 *      i4x2_t         8              4              8           2             1
 *
 *  @sa `dimensions_per_value<T>()` to convert dimension counts to value counts.
 *  @sa `bits_per_value<T>()` to infer the size of each value.
 */

#ifndef NK_TYPES_HPP
#define NK_TYPES_HPP

#include <bit>      // `std::bit_cast`
#include <compare>  // `std::strong_ordering`
#include <concepts> // `std::integral`
#include <cmath>    // `std::sqrt`
#include <complex>  // `std::complex`
#include <cstdint>  // `nk_u32_t`
#include <limits>   // `std::numeric_limits`
#include <utility>  // `std::swap`

#include "numkong/types.h"
#include "numkong/cast.h"

namespace ashvardanian::numkong {

/* Forward declarations for all numeric wrapper types */
struct f32_t;
struct f64_t;
struct f16_t;
struct bf16_t;
struct e4m3_t;
struct e5m2_t;
struct i8_t;
struct u8_t;
struct i16_t;
struct u16_t;
struct i32_t;
struct u32_t;
struct i64_t;
struct u64_t;
struct i4x2_t;
struct u4x2_t;
struct u1x8_t;
struct f32c_t;
struct f64c_t;
struct f118_t;

/**
 *  @brief Single-precision IEEE 754 floating-point wrapper.
 *
 *  Provides strong type identity for `float`, compatible with NumKong kernels
 *  and `std::mdspan`. API inspired by Rust's f32, std::numeric_limits, and
 *  Eigen's NumTraits.
 *
 *  Features:
 *  - Arithmetic operators and Rust-style methods (total_cmp, signum, mul_add)
 *  - Classification: is_nan, is_finite, is_normal, is_subnormal
 *  - Bit manipulation: to_bits, from_bits, copysign (constexpr)
 *  - Type aliases for NumKong kernel signatures (dot_result_t, reduce_add_result_t, etc.)
 *
 *  @note Only bit-manipulation and pure-arithmetic functions are constexpr.
 *        STL cmath functions become constexpr in C++26.
 */
struct f32_t {

    using raw_t = nk_f32_t;
    using uint_t = nk_u32_t;

    using dot_result_t = f32_t;         // `nk_dot_f32` output
    using reduce_add_result_t = f64_t;  // `nk_reduce_add_f32` widened output
    using sqeuclidean_result_t = f32_t; // `nk_sqeuclidean_f32` output
    using angular_result_t = f32_t;     // `nk_angular_f32` output
    using curved_result_t = f32_t;      // bilinear, mahalanobis
    using geospatial_result_t = f32_t;  // haversine, vincenty
    using probability_result_t = f32_t; // kld, jsd
    using mesh_result_t = f32_t;        // `nk_rmsd_f32` output
    using scale_t = nk_f32_t;

    static constexpr nk_dtype_t dtype() noexcept { return nk_f32_k; }
    static constexpr char const *dtype_name() noexcept { return "f32"; }
    static constexpr unsigned bits_per_word() noexcept { return 32; }
    static constexpr unsigned bits_per_dimension() noexcept { return 32; }
    static constexpr unsigned bits_per_value() noexcept { return 32; }
    static constexpr unsigned mantissa_bits() noexcept { return 23; }
    static constexpr unsigned exponent_bits() noexcept { return 8; }
    static constexpr int min_exponent() noexcept { return -125; }
    static constexpr int max_exponent() noexcept { return 128; }
    static constexpr bool is_integer() noexcept { return false; }
    static constexpr bool is_signed() noexcept { return true; }
    static constexpr bool is_complex() noexcept { return false; }
    static constexpr bool is_exact() noexcept { return false; }
    static constexpr bool has_infinity() noexcept { return true; }
    static constexpr bool has_nan() noexcept { return true; }

    static constexpr double rmsd_tolerance() noexcept { return 1e-5; }
    static constexpr double kabsch_tolerance() noexcept { return 1e-5; }
    static constexpr double umeyama_tolerance() noexcept { return 1e-4; }

    using dot_kernel_t = void (*)(raw_t const *, raw_t const *, nk_size_t, nk_f32_t *);
    using sqeuclidean_kernel_t = void (*)(raw_t const *, raw_t const *, nk_size_t, nk_f32_t *);
    using angular_kernel_t = void (*)(raw_t const *, raw_t const *, nk_size_t, nk_f32_t *);
    using kld_kernel_t = void (*)(raw_t const *, raw_t const *, nk_size_t, nk_f32_t *);
    using jsd_kernel_t = void (*)(raw_t const *, raw_t const *, nk_size_t, nk_f32_t *);
    using sum_kernel_t = void (*)(raw_t const *, raw_t const *, nk_size_t, raw_t *);
    using scale_kernel_t = void (*)(raw_t const *, nk_size_t, scale_t const *, scale_t const *, raw_t *);
    using wsum_kernel_t = void (*)(raw_t const *, raw_t const *, nk_size_t, scale_t const *, scale_t const *, raw_t *);
    using fma_kernel_t = void (*)(raw_t const *, raw_t const *, raw_t const *, nk_size_t, scale_t const *,
                                  scale_t const *, raw_t *);
    using reduce_add_kernel_t = void (*)(raw_t const *, nk_size_t, nk_size_t, nk_f64_t *);
    using reduce_extremum_kernel_t = void (*)(raw_t const *, nk_size_t, nk_size_t, raw_t *, nk_size_t *);
    using trig_kernel_t = void (*)(raw_t const *, nk_size_t, raw_t *);
    using mesh_kernel_t = void (*)(raw_t const *, raw_t const *, nk_size_t, raw_t *, raw_t *, raw_t *, raw_t *,
                                   raw_t *);
    using bilinear_kernel_t = void (*)(raw_t const *, raw_t const *, raw_t const *, nk_size_t, raw_t *);
    using haversine_kernel_t = void (*)(raw_t const *, raw_t const *, raw_t const *, raw_t const *, nk_size_t, raw_t *);
    using dots_packed_size_kernel_t = nk_size_t (*)(nk_size_t, nk_size_t);
    using dots_pack_kernel_t = void (*)(raw_t const *, nk_size_t, nk_size_t, nk_size_t, void *);
    using dots_packed_kernel_t = void (*)(raw_t const *, void const *, nk_f32_t *, nk_size_t, nk_size_t, nk_size_t,
                                          nk_size_t, nk_size_t);
    using dots_symmetric_kernel_t = void (*)(raw_t const *, nk_size_t, nk_size_t, nk_size_t, nk_f32_t *, nk_size_t,
                                             nk_size_t, nk_size_t);
    using sparse_dot_index_t = u32_t;
    using sparse_dot_kernel_t = void (*)(nk_u32_t const *, nk_u32_t const *, raw_t const *, raw_t const *, nk_size_t,
                                         nk_size_t, nk_f32_t *);

    raw_t raw_;

    constexpr f32_t() noexcept : raw_(0) {}
    constexpr f32_t(float v) noexcept : raw_(v) {}
    constexpr explicit f32_t(double v) noexcept : raw_(static_cast<float>(v)) {}
    template <std::integral integral_type_>
    constexpr f32_t(integral_type_ v) noexcept : raw_(static_cast<float>(v)) {}
    constexpr operator float() const noexcept { return raw_; }
    constexpr float raw() const noexcept { return raw_; }
    static constexpr f32_t from_raw(raw_t r) noexcept { return f32_t {r}; }
    static constexpr f32_t from_bits(uint_t bits) noexcept { return f32_t {std::bit_cast<raw_t>(bits)}; }
    constexpr uint_t to_bits() const noexcept { return std::bit_cast<uint_t>(raw_); }

    static constexpr f32_t finite_max() noexcept { return f32_t {3.40282347e+38f}; }
    static constexpr f32_t finite_min() noexcept { return f32_t {-3.40282347e+38f}; }
    static constexpr f32_t positive_min() noexcept { return f32_t {1.17549435e-38f}; }
    static constexpr f32_t subnormal_min() noexcept { return f32_t {1.40129846e-45f}; }
    static constexpr f32_t epsilon() noexcept { return f32_t {1.19209290e-07f}; }
    static constexpr f32_t positive_infinity() noexcept { return f32_t {std::bit_cast<raw_t>(0x7F800000u)}; }
    static constexpr f32_t negative_infinity() noexcept { return f32_t {std::bit_cast<raw_t>(0xFF800000u)}; }
    static constexpr f32_t quiet_nan() noexcept { return f32_t {std::bit_cast<raw_t>(0x7FC00000u)}; }
    static constexpr f32_t signaling_nan() noexcept { return f32_t {std::bit_cast<raw_t>(0x7F800001u)}; }

    // Mathematical constants
    static constexpr f32_t pi_k() noexcept { return f32_t {3.14159265358979323846f}; }
    static constexpr f32_t two_pi_k() noexcept { return f32_t {6.28318530717958647692f}; }
    static constexpr f32_t half_pi_k() noexcept { return f32_t {1.57079632679489661923f}; }
    static constexpr f32_t e_k() noexcept { return f32_t {2.71828182845904523536f}; }
    static constexpr f32_t sqrt2_k() noexcept { return f32_t {1.41421356237309504880f}; }
    static constexpr f32_t inv_sqrt2_k() noexcept { return f32_t {0.70710678118654752440f}; }
    static constexpr f32_t ln2_k() noexcept { return f32_t {0.69314718055994530942f}; }

    constexpr bool is_nan() const noexcept { return (to_bits() & 0x7FFFFFFFu) > 0x7F800000u; }
    constexpr bool is_infinite() const noexcept { return (to_bits() & 0x7FFFFFFFu) == 0x7F800000u; }
    constexpr bool is_finite() const noexcept { return (to_bits() & 0x7F800000u) != 0x7F800000u; }
    constexpr bool is_normal() const noexcept {
        uint_t exp = (to_bits() >> 23) & 0xFF;
        return exp != 0 && exp != 0xFF;
    }
    constexpr bool is_subnormal() const noexcept {
        uint_t bits = to_bits();
        return ((bits >> 23) & 0xFF) == 0 && (bits & 0x007FFFFFu) != 0;
    }
    constexpr bool is_sign_positive() const noexcept { return (to_bits() & 0x80000000u) == 0; }
    constexpr bool is_sign_negative() const noexcept { return (to_bits() & 0x80000000u) != 0; }

    constexpr f32_t operator+() const noexcept { return *this; }
    constexpr f32_t operator-() const noexcept { return f32_t {-raw_}; }
    constexpr f32_t operator+(f32_t o) const noexcept { return f32_t {raw_ + o.raw_}; }
    constexpr f32_t operator-(f32_t o) const noexcept { return f32_t {raw_ - o.raw_}; }
    constexpr f32_t operator*(f32_t o) const noexcept { return f32_t {raw_ * o.raw_}; }
    constexpr f32_t operator/(f32_t o) const noexcept { return f32_t {raw_ / o.raw_}; }
    constexpr f32_t &operator+=(f32_t o) noexcept {
        raw_ += o.raw_;
        return *this;
    }
    constexpr f32_t &operator-=(f32_t o) noexcept {
        raw_ -= o.raw_;
        return *this;
    }
    constexpr f32_t &operator*=(f32_t o) noexcept {
        raw_ *= o.raw_;
        return *this;
    }
    constexpr f32_t &operator/=(f32_t o) noexcept {
        raw_ /= o.raw_;
        return *this;
    }

    constexpr bool operator==(f32_t o) const noexcept { return raw_ == o.raw_; }
    constexpr bool operator!=(f32_t o) const noexcept { return raw_ != o.raw_; }
    constexpr bool operator<(f32_t o) const noexcept { return raw_ < o.raw_; }
    constexpr bool operator>(f32_t o) const noexcept { return raw_ > o.raw_; }
    constexpr bool operator<=(f32_t o) const noexcept { return raw_ <= o.raw_; }
    constexpr bool operator>=(f32_t o) const noexcept { return raw_ >= o.raw_; }

    /** @brief Total ordering comparison (Rust-style): -NaN < -Inf < ... < +Inf < +NaN. */
    constexpr int total_cmp(f32_t o) const noexcept {
        std::int32_t a = std::bit_cast<std::int32_t>(raw_);
        std::int32_t b = std::bit_cast<std::int32_t>(o.raw_);
        if (a < 0) a = std::int32_t(0x80000000u) - a;
        if (b < 0) b = std::int32_t(0x80000000u) - b;
        return (a > b) - (a < b);
    }

    constexpr f32_t abs() const noexcept { return from_bits(to_bits() & 0x7FFFFFFFu); }
    constexpr f32_t copysign(f32_t sign) const noexcept {
        return from_bits((to_bits() & 0x7FFFFFFFu) | (sign.to_bits() & 0x80000000u));
    }
    constexpr f32_t signum() const noexcept {
        if (is_nan()) return *this;
        return is_sign_negative() ? f32_t {-1.0f} : f32_t {1.0f};
    }

    inline f32_t floor() const noexcept { return f32_t {std::floor(raw_)}; }
    inline f32_t ceil() const noexcept { return f32_t {std::ceil(raw_)}; }
    inline f32_t round() const noexcept { return f32_t {std::round(raw_)}; }
    inline f32_t trunc() const noexcept { return f32_t {std::trunc(raw_)}; }
    inline f32_t fract() const noexcept { return f32_t {raw_ - std::trunc(raw_)}; }

    inline f32_t sqrt() const noexcept { return f32_t {std::sqrt(raw_)}; }
    inline f32_t cbrt() const noexcept { return f32_t {std::cbrt(raw_)}; }
    inline f32_t rsqrt() const noexcept { return f32_t {1.0f / std::sqrt(raw_)}; }
    constexpr f32_t recip() const noexcept { return f32_t {1.0f / raw_}; }
    inline f32_t mul_add(f32_t a, f32_t b) const noexcept { return f32_t {std::fma(raw_, a.raw_, b.raw_)}; }
    inline f32_t powf(f32_t exp) const noexcept { return f32_t {std::pow(raw_, exp.raw_)}; }
    constexpr f32_t powi(int n) const noexcept {
        float result = 1.0f, base = raw_;
        if (n < 0) {
            base = 1.0f / base;
            n = -n;
        }
        while (n) {
            if (n & 1) result *= base;
            base *= base;
            n >>= 1;
        }
        return f32_t {result};
    }

    inline f32_t exp() const noexcept { return f32_t {std::exp(raw_)}; }
    inline f32_t exp2() const noexcept { return f32_t {std::exp2(raw_)}; }
    inline f32_t exp_m1() const noexcept { return f32_t {std::expm1(raw_)}; }
    inline f32_t ln() const noexcept { return f32_t {std::log(raw_)}; }
    inline f32_t ln_1p() const noexcept { return f32_t {std::log1p(raw_)}; }
    inline f32_t log2() const noexcept { return f32_t {std::log2(raw_)}; }
    inline f32_t log10() const noexcept { return f32_t {std::log10(raw_)}; }
    inline f32_t log(f32_t base) const noexcept { return f32_t {std::log(raw_) / std::log(base.raw_)}; }

    inline f32_t sin() const noexcept { return f32_t {std::sin(raw_)}; }
    inline f32_t cos() const noexcept { return f32_t {std::cos(raw_)}; }
    inline f32_t tan() const noexcept { return f32_t {std::tan(raw_)}; }
    inline f32_t asin() const noexcept { return f32_t {std::asin(raw_)}; }
    inline f32_t acos() const noexcept { return f32_t {std::acos(raw_)}; }
    inline f32_t atan() const noexcept { return f32_t {std::atan(raw_)}; }
    inline f32_t atan2(f32_t x) const noexcept { return f32_t {std::atan2(raw_, x.raw_)}; }
    inline f32_t hypot(f32_t y) const noexcept { return f32_t {std::hypot(raw_, y.raw_)}; }
    constexpr f32_t to_radians() const noexcept { return f32_t {raw_ * 0.017453292519943295f}; }
    constexpr f32_t to_degrees() const noexcept { return f32_t {raw_ * 57.29577951308232f}; }

    inline f32_t sinh() const noexcept { return f32_t {std::sinh(raw_)}; }
    inline f32_t cosh() const noexcept { return f32_t {std::cosh(raw_)}; }
    inline f32_t tanh() const noexcept { return f32_t {std::tanh(raw_)}; }
    inline f32_t asinh() const noexcept { return f32_t {std::asinh(raw_)}; }
    inline f32_t acosh() const noexcept { return f32_t {std::acosh(raw_)}; }
    inline f32_t atanh() const noexcept { return f32_t {std::atanh(raw_)}; }

    inline f32_t min(f32_t o) const noexcept { return f32_t {std::fmin(raw_, o.raw_)}; }
    inline f32_t max(f32_t o) const noexcept { return f32_t {std::fmax(raw_, o.raw_)}; }
    inline f32_t clamp(f32_t lo, f32_t hi) const noexcept { return max(lo).min(hi); }

    /** @brief Saturating addition: clamps to finite range on overflow. */
    constexpr f32_t saturating_add(f32_t o) const noexcept {
        float result = raw_ + o.raw_;
        if (result == std::numeric_limits<float>::infinity()) return finite_max();
        if (result == -std::numeric_limits<float>::infinity()) return finite_min();
        return f32_t {result};
    }

    /** @brief Saturating subtraction: clamps to finite range on overflow. */
    constexpr f32_t saturating_sub(f32_t o) const noexcept {
        float result = raw_ - o.raw_;
        if (result == std::numeric_limits<float>::infinity()) return finite_max();
        if (result == -std::numeric_limits<float>::infinity()) return finite_min();
        return f32_t {result};
    }

    /** @brief Convert to any numeric type. */
    template <typename target_type_>
    constexpr target_type_ to() const noexcept {
        return target_type_(raw_);
    }

  private:
    constexpr f32_t(raw_t v, int) noexcept : raw_(v) {} // Private constructor for from_raw
};

/**
 *  @brief Double-precision IEEE 754 floating-point wrapper.
 *
 *  Provides strong type identity for `double`, compatible with NumKong kernels
 *  and `std::mdspan`. API inspired by Rust's f64, std::numeric_limits, and
 *  Eigen's NumTraits.
 *
 *  @note Only bit-manipulation and pure-arithmetic functions are constexpr.
 *        STL cmath functions become constexpr in C++26.
 */
struct f64_t {

    using raw_t = nk_f64_t;
    using uint_t = nk_u64_t;

    using dot_result_t = f64_t;         // `nk_dot_f64` output
    using reduce_add_result_t = f64_t;  // `nk_reduce_add_f64` output
    using sqeuclidean_result_t = f64_t; // `nk_sqeuclidean_f64` output
    using angular_result_t = f64_t;     // `nk_angular_f64` output
    using curved_result_t = f64_t;      // bilinear, mahalanobis
    using probability_result_t = f64_t; // kld, jsd
    using mesh_result_t = f64_t;        // `nk_rmsd_f64` output
    using scale_t = nk_f64_t;

    static constexpr nk_dtype_t dtype() noexcept { return nk_f64_k; }
    static constexpr char const *dtype_name() noexcept { return "f64"; }
    static constexpr unsigned bits_per_word() noexcept { return 64; }
    static constexpr unsigned bits_per_dimension() noexcept { return 64; }
    static constexpr unsigned bits_per_value() noexcept { return 64; }
    static constexpr unsigned mantissa_bits() noexcept { return 52; }
    static constexpr unsigned exponent_bits() noexcept { return 11; }
    static constexpr int min_exponent() noexcept { return -1021; }
    static constexpr int max_exponent() noexcept { return 1024; }
    static constexpr bool is_integer() noexcept { return false; }
    static constexpr bool is_signed() noexcept { return true; }
    static constexpr bool is_complex() noexcept { return false; }
    static constexpr bool is_exact() noexcept { return false; }
    static constexpr bool has_infinity() noexcept { return true; }
    static constexpr bool has_nan() noexcept { return true; }

    static constexpr double rmsd_tolerance() noexcept { return 1e-10; }
    static constexpr double kabsch_tolerance() noexcept { return 1e-10; }
    static constexpr double umeyama_tolerance() noexcept { return 1e-9; }

    using dot_kernel_t = void (*)(raw_t const *, raw_t const *, nk_size_t, nk_f64_t *);
    using sqeuclidean_kernel_t = void (*)(raw_t const *, raw_t const *, nk_size_t, nk_f64_t *);
    using angular_kernel_t = void (*)(raw_t const *, raw_t const *, nk_size_t, nk_f64_t *);
    using kld_kernel_t = void (*)(raw_t const *, raw_t const *, nk_size_t, nk_f64_t *);
    using jsd_kernel_t = void (*)(raw_t const *, raw_t const *, nk_size_t, nk_f64_t *);
    using sum_kernel_t = void (*)(raw_t const *, raw_t const *, nk_size_t, raw_t *);
    using scale_kernel_t = void (*)(raw_t const *, nk_size_t, scale_t const *, scale_t const *, raw_t *);
    using wsum_kernel_t = void (*)(raw_t const *, raw_t const *, nk_size_t, scale_t const *, scale_t const *, raw_t *);
    using fma_kernel_t = void (*)(raw_t const *, raw_t const *, raw_t const *, nk_size_t, scale_t const *,
                                  scale_t const *, raw_t *);
    using reduce_add_kernel_t = void (*)(raw_t const *, nk_size_t, nk_size_t, nk_f64_t *);
    using reduce_extremum_kernel_t = void (*)(raw_t const *, nk_size_t, nk_size_t, raw_t *, nk_size_t *);
    using trig_kernel_t = void (*)(raw_t const *, nk_size_t, raw_t *);
    using bilinear_kernel_t = void (*)(raw_t const *, raw_t const *, raw_t const *, nk_size_t, raw_t *);
    using haversine_kernel_t = void (*)(raw_t const *, raw_t const *, raw_t const *, raw_t const *, nk_size_t, raw_t *);
    using mesh_kernel_t = void (*)(raw_t const *, raw_t const *, nk_size_t, raw_t *, raw_t *, raw_t *, raw_t *,
                                   raw_t *);
    using dots_packed_size_kernel_t = nk_size_t (*)(nk_size_t, nk_size_t);
    using dots_pack_kernel_t = void (*)(raw_t const *, nk_size_t, nk_size_t, nk_size_t, void *);
    using dots_packed_kernel_t = void (*)(raw_t const *, void const *, nk_f64_t *, nk_size_t, nk_size_t, nk_size_t,
                                          nk_size_t, nk_size_t);
    using dots_symmetric_kernel_t = void (*)(raw_t const *, nk_size_t, nk_size_t, nk_size_t, nk_f64_t *, nk_size_t,
                                             nk_size_t, nk_size_t);

    raw_t raw_;

    constexpr f64_t() noexcept : raw_(0) {}
    constexpr f64_t(double v) noexcept : raw_(v) {}
    constexpr explicit f64_t(float v) noexcept : raw_(static_cast<double>(v)) {}
    template <std::integral integral_type_>
    constexpr f64_t(integral_type_ v) noexcept : raw_(static_cast<double>(v)) {}
    constexpr operator double() const noexcept { return raw_; }
    constexpr double raw() const noexcept { return raw_; }
    static constexpr f64_t from_raw(raw_t r) noexcept { return f64_t {r}; }
    static constexpr f64_t from_bits(uint_t bits) noexcept { return f64_t {std::bit_cast<raw_t>(bits)}; }
    constexpr uint_t to_bits() const noexcept { return std::bit_cast<uint_t>(raw_); }

    static constexpr f64_t finite_max() noexcept { return f64_t {1.7976931348623157e+308}; }
    static constexpr f64_t finite_min() noexcept { return f64_t {-1.7976931348623157e+308}; }
    static constexpr f64_t positive_min() noexcept { return f64_t {2.2250738585072014e-308}; }
    static constexpr f64_t subnormal_min() noexcept { return f64_t {4.9406564584124654e-324}; }
    static constexpr f64_t epsilon() noexcept { return f64_t {2.2204460492503131e-16}; }
    static constexpr f64_t positive_infinity() noexcept { return f64_t {std::bit_cast<raw_t>(0x7FF0000000000000ull)}; }
    static constexpr f64_t negative_infinity() noexcept { return f64_t {std::bit_cast<raw_t>(0xFFF0000000000000ull)}; }
    static constexpr f64_t quiet_nan() noexcept { return f64_t {std::bit_cast<raw_t>(0x7FF8000000000000ull)}; }
    static constexpr f64_t signaling_nan() noexcept { return f64_t {std::bit_cast<raw_t>(0x7FF0000000000001ull)}; }

    // Mathematical constants
    static constexpr f64_t pi_k() noexcept { return f64_t {3.14159265358979323846}; }
    static constexpr f64_t two_pi_k() noexcept { return f64_t {6.28318530717958647692}; }
    static constexpr f64_t half_pi_k() noexcept { return f64_t {1.57079632679489661923}; }
    static constexpr f64_t e_k() noexcept { return f64_t {2.71828182845904523536}; }
    static constexpr f64_t sqrt2_k() noexcept { return f64_t {1.41421356237309504880}; }
    static constexpr f64_t inv_sqrt2_k() noexcept { return f64_t {0.70710678118654752440}; }
    static constexpr f64_t ln2_k() noexcept { return f64_t {0.69314718055994530942}; }

    constexpr bool is_nan() const noexcept { return (to_bits() & 0x7FFFFFFFFFFFFFFFull) > 0x7FF0000000000000ull; }
    constexpr bool is_infinite() const noexcept { return (to_bits() & 0x7FFFFFFFFFFFFFFFull) == 0x7FF0000000000000ull; }
    constexpr bool is_finite() const noexcept { return (to_bits() & 0x7FF0000000000000ull) != 0x7FF0000000000000ull; }
    constexpr bool is_normal() const noexcept {
        uint_t exp = (to_bits() >> 52) & 0x7FF;
        return exp != 0 && exp != 0x7FF;
    }
    constexpr bool is_subnormal() const noexcept {
        uint_t bits = to_bits();
        return ((bits >> 52) & 0x7FF) == 0 && (bits & 0x000FFFFFFFFFFFFFull) != 0;
    }
    constexpr bool is_sign_positive() const noexcept { return (to_bits() & 0x8000000000000000ull) == 0; }
    constexpr bool is_sign_negative() const noexcept { return (to_bits() & 0x8000000000000000ull) != 0; }

    constexpr f64_t operator+() const noexcept { return *this; }
    constexpr f64_t operator-() const noexcept { return f64_t {-raw_}; }
    constexpr f64_t operator+(f64_t o) const noexcept { return f64_t {raw_ + o.raw_}; }
    constexpr f64_t operator-(f64_t o) const noexcept { return f64_t {raw_ - o.raw_}; }
    constexpr f64_t operator*(f64_t o) const noexcept { return f64_t {raw_ * o.raw_}; }
    constexpr f64_t operator/(f64_t o) const noexcept { return f64_t {raw_ / o.raw_}; }
    constexpr f64_t &operator+=(f64_t o) noexcept {
        raw_ += o.raw_;
        return *this;
    }
    constexpr f64_t &operator-=(f64_t o) noexcept {
        raw_ -= o.raw_;
        return *this;
    }
    constexpr f64_t &operator*=(f64_t o) noexcept {
        raw_ *= o.raw_;
        return *this;
    }
    constexpr f64_t &operator/=(f64_t o) noexcept {
        raw_ /= o.raw_;
        return *this;
    }

    constexpr bool operator==(f64_t o) const noexcept { return raw_ == o.raw_; }
    constexpr bool operator!=(f64_t o) const noexcept { return raw_ != o.raw_; }
    constexpr bool operator<(f64_t o) const noexcept { return raw_ < o.raw_; }
    constexpr bool operator>(f64_t o) const noexcept { return raw_ > o.raw_; }
    constexpr bool operator<=(f64_t o) const noexcept { return raw_ <= o.raw_; }
    constexpr bool operator>=(f64_t o) const noexcept { return raw_ >= o.raw_; }

    /** @brief Total ordering comparison (Rust-style): -NaN < -Inf < ... < +Inf < +NaN. */
    constexpr int total_cmp(f64_t o) const noexcept {
        std::int64_t a = std::bit_cast<std::int64_t>(raw_);
        std::int64_t b = std::bit_cast<std::int64_t>(o.raw_);
        if (a < 0) a = std::int64_t(0x8000000000000000ull) - a;
        if (b < 0) b = std::int64_t(0x8000000000000000ull) - b;
        return (a > b) - (a < b);
    }

    constexpr f64_t abs() const noexcept { return from_bits(to_bits() & 0x7FFFFFFFFFFFFFFFull); }
    constexpr f64_t copysign(f64_t sign) const noexcept {
        return from_bits((to_bits() & 0x7FFFFFFFFFFFFFFFull) | (sign.to_bits() & 0x8000000000000000ull));
    }
    constexpr f64_t signum() const noexcept {
        if (is_nan()) return *this;
        return is_sign_negative() ? f64_t {-1.0} : f64_t {1.0};
    }

    inline f64_t floor() const noexcept { return f64_t {std::floor(raw_)}; }
    inline f64_t ceil() const noexcept { return f64_t {std::ceil(raw_)}; }
    inline f64_t round() const noexcept { return f64_t {std::round(raw_)}; }
    inline f64_t trunc() const noexcept { return f64_t {std::trunc(raw_)}; }
    inline f64_t fract() const noexcept { return f64_t {raw_ - std::trunc(raw_)}; }

    inline f64_t sqrt() const noexcept { return f64_t {std::sqrt(raw_)}; }
    inline f64_t cbrt() const noexcept { return f64_t {std::cbrt(raw_)}; }
    inline f64_t rsqrt() const noexcept { return f64_t {1.0 / std::sqrt(raw_)}; }
    constexpr f64_t recip() const noexcept { return f64_t {1.0 / raw_}; }
    inline f64_t mul_add(f64_t a, f64_t b) const noexcept { return f64_t {std::fma(raw_, a.raw_, b.raw_)}; }
    inline f64_t powf(f64_t exp) const noexcept { return f64_t {std::pow(raw_, exp.raw_)}; }
    constexpr f64_t powi(int n) const noexcept {
        double result = 1.0, base = raw_;
        if (n < 0) {
            base = 1.0 / base;
            n = -n;
        }
        while (n) {
            if (n & 1) result *= base;
            base *= base;
            n >>= 1;
        }
        return f64_t {result};
    }

    inline f64_t exp() const noexcept { return f64_t {std::exp(raw_)}; }
    inline f64_t exp2() const noexcept { return f64_t {std::exp2(raw_)}; }
    inline f64_t exp_m1() const noexcept { return f64_t {std::expm1(raw_)}; }
    inline f64_t ln() const noexcept { return f64_t {std::log(raw_)}; }
    inline f64_t ln_1p() const noexcept { return f64_t {std::log1p(raw_)}; }
    inline f64_t log2() const noexcept { return f64_t {std::log2(raw_)}; }
    inline f64_t log10() const noexcept { return f64_t {std::log10(raw_)}; }
    inline f64_t log(f64_t base) const noexcept { return f64_t {std::log(raw_) / std::log(base.raw_)}; }

    inline f64_t sin() const noexcept { return f64_t {std::sin(raw_)}; }
    inline f64_t cos() const noexcept { return f64_t {std::cos(raw_)}; }
    inline f64_t tan() const noexcept { return f64_t {std::tan(raw_)}; }
    inline f64_t asin() const noexcept { return f64_t {std::asin(raw_)}; }
    inline f64_t acos() const noexcept { return f64_t {std::acos(raw_)}; }
    inline f64_t atan() const noexcept { return f64_t {std::atan(raw_)}; }
    inline f64_t atan2(f64_t x) const noexcept { return f64_t {std::atan2(raw_, x.raw_)}; }
    inline f64_t hypot(f64_t y) const noexcept { return f64_t {std::hypot(raw_, y.raw_)}; }
    constexpr f64_t to_radians() const noexcept { return f64_t {raw_ * 0.017453292519943295}; }
    constexpr f64_t to_degrees() const noexcept { return f64_t {raw_ * 57.29577951308232}; }

    inline f64_t sinh() const noexcept { return f64_t {std::sinh(raw_)}; }
    inline f64_t cosh() const noexcept { return f64_t {std::cosh(raw_)}; }
    inline f64_t tanh() const noexcept { return f64_t {std::tanh(raw_)}; }
    inline f64_t asinh() const noexcept { return f64_t {std::asinh(raw_)}; }
    inline f64_t acosh() const noexcept { return f64_t {std::acosh(raw_)}; }
    inline f64_t atanh() const noexcept { return f64_t {std::atanh(raw_)}; }

    inline f64_t min(f64_t o) const noexcept { return f64_t {std::fmin(raw_, o.raw_)}; }
    inline f64_t max(f64_t o) const noexcept { return f64_t {std::fmax(raw_, o.raw_)}; }
    inline f64_t clamp(f64_t lo, f64_t hi) const noexcept { return max(lo).min(hi); }

    /** @brief Saturating addition: clamps to finite range on overflow. */
    constexpr f64_t saturating_add(f64_t o) const noexcept {
        double result = raw_ + o.raw_;
        if (result == std::numeric_limits<double>::infinity()) return finite_max();
        if (result == -std::numeric_limits<double>::infinity()) return finite_min();
        return f64_t {result};
    }

    /** @brief Saturating subtraction: clamps to finite range on overflow. */
    constexpr f64_t saturating_sub(f64_t o) const noexcept {
        double result = raw_ - o.raw_;
        if (result == std::numeric_limits<double>::infinity()) return finite_max();
        if (result == -std::numeric_limits<double>::infinity()) return finite_min();
        return f64_t {result};
    }

    /** @brief Convert to any numeric type. */
    template <typename target_type_>
    constexpr target_type_ to() const noexcept {
        return target_type_(raw_);
    }
};

/**
 *  @brief Single-precision complex number wrapper using composition.
 *
 *  Provides strong type identity for complex float, compatible with NumKong kernels
 *  and `std::mdspan`. Uses composition of two `f32_t` components (not inheritance).
 *
 *  Features:
 *  - Real/imaginary accessors via `real()` and `imag()`
 *  - Complex arithmetic: `+`, `-`, `*`, `/`
 *  - Complex-specific: `conj()`, `norm()`, `abs()`, `arg()`
 *  - Full transcendentals: `exp()`, `log()`, `sqrt()`, `pow()`, trig, hyperbolic
 *
 *  @note Non-constexpr due to reliance on `f32_t` STL-forwarding functions.
 */
struct f32c_t {
    using component_t = f32_t;
    using raw_t = nk_f32c_t;

    using dot_result_t = f32c_t;
    using vdot_result_t = f32c_t;
    using curved_result_t = f32c_t; // bilinear

    using dot_kernel_t = void (*)(raw_t const *, raw_t const *, nk_size_t, nk_f32c_t *);
    using vdot_kernel_t = void (*)(raw_t const *, raw_t const *, nk_size_t, nk_f32c_t *);
    using bilinear_kernel_t = void (*)(raw_t const *, raw_t const *, raw_t const *, nk_size_t, raw_t *);

    raw_t raw_;

    static constexpr nk_dtype_t dtype() noexcept { return nk_f32c_k; }
    static constexpr char const *dtype_name() noexcept { return "f32c"; }
    static constexpr unsigned bits_per_word() noexcept { return 32; }      // One float component
    static constexpr unsigned bits_per_dimension() noexcept { return 64; } // One complex number
    static constexpr unsigned bits_per_value() noexcept { return 64; }
    static constexpr unsigned component_bits() noexcept { return 32; }
    static constexpr bool is_integer() noexcept { return false; }
    static constexpr bool is_signed() noexcept { return true; }
    static constexpr bool is_complex() noexcept { return true; }
    static constexpr bool is_exact() noexcept { return false; }
    static constexpr bool has_infinity() noexcept { return true; }
    static constexpr bool has_nan() noexcept { return true; }

    constexpr f32c_t() noexcept : raw_ {0, 0} {}
    constexpr f32c_t(f32_t r) noexcept : raw_ {r.raw_, 0} {}
    constexpr f32c_t(float r) noexcept : raw_ {r, 0} {}
    constexpr f32c_t(f32_t r, f32_t i) noexcept : raw_ {r.raw_, i.raw_} {}
    constexpr f32c_t(float r, float i) noexcept : raw_ {r, i} {}

    static constexpr f32c_t from_raw(raw_t r) noexcept { return f32c_t {r.real, r.imag}; }

    constexpr f32_t real() const noexcept { return f32_t(raw_.real); }
    constexpr f32_t imag() const noexcept { return f32_t(raw_.imag); }

    static constexpr f32c_t zero() noexcept { return f32c_t {}; }
    static constexpr f32c_t one() noexcept { return f32c_t {1.0f}; }
    static constexpr f32c_t i() noexcept { return f32c_t {0.0f, 1.0f}; }

    static constexpr f32c_t finite_max() noexcept { return f32c_t {3.40282347e+38f, 3.40282347e+38f}; }
    static constexpr f32c_t finite_min() noexcept { return f32c_t {-3.40282347e+38f, -3.40282347e+38f}; }
    static constexpr f32c_t positive_infinity() noexcept {
        return f32c_t {f32_t::positive_infinity(), f32_t::positive_infinity()};
    }
    static constexpr f32c_t negative_infinity() noexcept {
        return f32c_t {f32_t::negative_infinity(), f32_t::negative_infinity()};
    }
    static constexpr f32c_t quiet_nan() noexcept { return f32c_t {f32_t::quiet_nan(), f32_t::quiet_nan()}; }

    constexpr f32c_t operator+() const noexcept { return *this; }
    constexpr f32c_t operator-() const noexcept { return f32c_t {-raw_.real, -raw_.imag}; }

    constexpr f32c_t operator+(f32c_t o) const noexcept {
        return f32c_t {raw_.real + o.raw_.real, raw_.imag + o.raw_.imag};
    }
    constexpr f32c_t operator-(f32c_t o) const noexcept {
        return f32c_t {raw_.real - o.raw_.real, raw_.imag - o.raw_.imag};
    }

    constexpr f32c_t operator*(f32c_t o) const noexcept {
        return f32c_t {raw_.real * o.raw_.real - raw_.imag * o.raw_.imag,
                       raw_.real * o.raw_.imag + raw_.imag * o.raw_.real};
    }

    inline f32c_t operator/(f32c_t o) const noexcept {
        float denom = o.raw_.real * o.raw_.real + o.raw_.imag * o.raw_.imag;
        return f32c_t {(raw_.real * o.raw_.real + raw_.imag * o.raw_.imag) / denom,
                       (raw_.imag * o.raw_.real - raw_.real * o.raw_.imag) / denom};
    }

    constexpr f32c_t &operator+=(f32c_t o) noexcept {
        raw_.real += o.raw_.real;
        raw_.imag += o.raw_.imag;
        return *this;
    }
    constexpr f32c_t &operator-=(f32c_t o) noexcept {
        raw_.real -= o.raw_.real;
        raw_.imag -= o.raw_.imag;
        return *this;
    }
    inline f32c_t &operator*=(f32c_t o) noexcept { return *this = *this * o; }
    inline f32c_t &operator/=(f32c_t o) noexcept { return *this = *this / o; }

    constexpr f32c_t operator*(f32_t s) const noexcept { return f32c_t {raw_.real * s.raw_, raw_.imag * s.raw_}; }
    constexpr f32c_t operator/(f32_t s) const noexcept { return f32c_t {raw_.real / s.raw_, raw_.imag / s.raw_}; }

    constexpr bool operator==(f32c_t o) const noexcept { return raw_.real == o.raw_.real && raw_.imag == o.raw_.imag; }
    constexpr bool operator!=(f32c_t o) const noexcept { return !(*this == o); }

    /** @brief Complex conjugate: `a - bi`. */
    constexpr f32c_t conj() const noexcept { return f32c_t {raw_.real, -raw_.imag}; }

    /** @brief Squared magnitude: `a² + b²`. */
    constexpr f32_t norm() const noexcept { return f32_t(raw_.real * raw_.real + raw_.imag * raw_.imag); }

    /** @brief Magnitude: `√(a² + b²)`. */
    inline f32_t abs() const noexcept { return norm().sqrt(); }

    /** @brief Phase angle: `atan2(b, a)`. */
    inline f32_t arg() const noexcept { return f32_t(raw_.imag).atan2(f32_t(raw_.real)); }

    /** @brief Complex exponential: `e^(a+bi) = e^a(cos(b) + i·sin(b))`. */
    inline f32c_t exp() const noexcept {
        f32_t ea = real().exp();
        return f32c_t {ea * imag().cos(), ea * imag().sin()};
    }

    /** @brief Complex natural logarithm: `ln|z| + i·arg(z)`. */
    inline f32c_t log() const noexcept { return f32c_t {abs().ln(), arg()}; }

    /** @brief Complex base-10 logarithm. */
    inline f32c_t log10() const noexcept {
        f32c_t ln_z = log();
        return ln_z * f32_t {0.4342944819032518f}; // log10(e)
    }

    /** @brief Complex square root (principal value). */
    inline f32c_t sqrt() const noexcept {
        f32_t r = abs();
        f32_t half_arg = arg() * f32_t {0.5f};
        f32_t sqrt_r = r.sqrt();
        return f32c_t {sqrt_r * half_arg.cos(), sqrt_r * half_arg.sin()};
    }

    /** @brief Complex power: `z^w = e^(w·ln(z))`. */
    inline f32c_t pow(f32c_t w) const noexcept { return (w * log()).exp(); }

    /** @brief Real power: `z^x`. */
    inline f32c_t powf(f32_t x) const noexcept {
        f32_t r = abs();
        f32_t theta = arg();
        f32_t r_pow = r.powf(x);
        f32_t new_theta = theta * x;
        return f32c_t {r_pow * new_theta.cos(), r_pow * new_theta.sin()};
    }

    /** @brief Complex sine: `sin(z) = (e^(iz) - e^(-iz)) / (2i)`. */
    inline f32c_t sin() const noexcept {
        // sin(a + bi) = sin(a)cosh(b) + i·cos(a)sinh(b)
        return f32c_t {real().sin() * imag().cosh(), real().cos() * imag().sinh()};
    }

    /** @brief Complex cosine: `cos(z) = (e^(iz) + e^(-iz)) / 2`. */
    inline f32c_t cos() const noexcept {
        // cos(a + bi) = cos(a)cosh(b) - i·sin(a)sinh(b)
        return f32c_t {real().cos() * imag().cosh(), -(real().sin() * imag().sinh())};
    }

    /** @brief Complex tangent: `tan(z) = sin(z) / cos(z)`. */
    inline f32c_t tan() const noexcept { return sin() / cos(); }

    /** @brief Complex arcsine: `asin(z) = -i·ln(iz + √(1 - z²))`. */
    inline f32c_t asin() const noexcept {
        f32c_t iz = f32c_t {-raw_.imag, raw_.real}; // i * z
        f32c_t one_minus_z2 = one() - *this * *this;
        f32c_t sqrt_part = one_minus_z2.sqrt();
        f32c_t ln_part = (iz + sqrt_part).log();
        return f32c_t {ln_part.raw_.imag, -ln_part.raw_.real}; // -i * ln_part
    }

    /** @brief Complex arccosine: `acos(z) = π/2 - asin(z)`. */
    inline f32c_t acos() const noexcept {
        constexpr float half_pi = 1.5707963267948966f;
        f32c_t as = asin();
        return f32c_t {half_pi - as.raw_.real, -as.raw_.imag};
    }

    /** @brief Complex arctangent: `atan(z) = (i/2)·ln((i+z)/(i-z))`. */
    inline f32c_t atan() const noexcept {
        f32c_t i_unit = i();
        f32c_t num = i_unit + *this;
        f32c_t denom = i_unit - *this;
        f32c_t ln_part = (num / denom).log();
        // (i/2) * ln_part = (0 + 0.5i) * (a + bi) = -0.5b + 0.5ai
        return f32c_t {ln_part.raw_.imag * -0.5f, ln_part.raw_.real * 0.5f};
    }

    /** @brief Complex hyperbolic sine: `sinh(z) = (e^z - e^(-z)) / 2`. */
    inline f32c_t sinh() const noexcept {
        // sinh(a + bi) = sinh(a)cos(b) + i·cosh(a)sin(b)
        return f32c_t {real().sinh() * imag().cos(), real().cosh() * imag().sin()};
    }

    /** @brief Complex hyperbolic cosine: `cosh(z) = (e^z + e^(-z)) / 2`. */
    inline f32c_t cosh() const noexcept {
        // cosh(a + bi) = cosh(a)cos(b) + i·sinh(a)sin(b)
        return f32c_t {real().cosh() * imag().cos(), real().sinh() * imag().sin()};
    }

    /** @brief Complex hyperbolic tangent: `tanh(z) = sinh(z) / cosh(z)`. */
    inline f32c_t tanh() const noexcept { return sinh() / cosh(); }

    /** @brief Complex inverse hyperbolic sine: `asinh(z) = ln(z + √(z² + 1))`. */
    inline f32c_t asinh() const noexcept {
        f32c_t z2_plus_1 = *this * *this + one();
        return (*this + z2_plus_1.sqrt()).log();
    }

    /** @brief Complex inverse hyperbolic cosine: `acosh(z) = ln(z + √(z² - 1))`. */
    inline f32c_t acosh() const noexcept {
        f32c_t z2_minus_1 = *this * *this - one();
        return (*this + z2_minus_1.sqrt()).log();
    }

    /** @brief Complex inverse hyperbolic tangent: `atanh(z) = (1/2)·ln((1+z)/(1-z))`. */
    inline f32c_t atanh() const noexcept {
        f32c_t one_plus_z = one() + *this;
        f32c_t one_minus_z = one() - *this;
        f32c_t ln_part = (one_plus_z / one_minus_z).log();
        return f32c_t {ln_part.raw_.real * 0.5f, ln_part.raw_.imag * 0.5f};
    }

    /** @brief Reciprocal: `1 / z`. */
    inline f32c_t recip() const noexcept { return one() / *this; }
};

/**
 *  @brief Double-precision complex number wrapper using composition.
 *
 *  Provides strong type identity for complex double, compatible with NumKong kernels
 *  and `std::mdspan`. Uses composition of two `f64_t` components (not inheritance).
 *
 *  Features:
 *  - Real/imaginary accessors via `real()` and `imag()`
 *  - Complex arithmetic: `+`, `-`, `*`, `/`
 *  - Complex-specific: `conj()`, `norm()`, `abs()`, `arg()`
 *  - Full transcendentals: `exp()`, `log()`, `sqrt()`, `pow()`, trig, hyperbolic
 *
 *  @note Non-constexpr due to reliance on `f64_t` STL-forwarding functions.
 */
struct f64c_t {
    using component_t = f64_t;
    using raw_t = nk_f64c_t;

    using dot_result_t = f64c_t;
    using vdot_result_t = f64c_t;
    using curved_result_t = f64c_t; // bilinear

    using dot_kernel_t = void (*)(raw_t const *, raw_t const *, nk_size_t, nk_f64c_t *);
    using vdot_kernel_t = void (*)(raw_t const *, raw_t const *, nk_size_t, nk_f64c_t *);
    using bilinear_kernel_t = void (*)(raw_t const *, raw_t const *, raw_t const *, nk_size_t, raw_t *);

    raw_t raw_;

    static constexpr nk_dtype_t dtype() noexcept { return nk_f64c_k; }
    static constexpr char const *dtype_name() noexcept { return "f64c"; }
    static constexpr unsigned bits_per_word() noexcept { return 64; }       // One double component
    static constexpr unsigned bits_per_dimension() noexcept { return 128; } // One complex number
    static constexpr unsigned bits_per_value() noexcept { return 128; }
    static constexpr unsigned component_bits() noexcept { return 64; }
    static constexpr bool is_integer() noexcept { return false; }
    static constexpr bool is_signed() noexcept { return true; }
    static constexpr bool is_complex() noexcept { return true; }
    static constexpr bool is_exact() noexcept { return false; }
    static constexpr bool has_infinity() noexcept { return true; }
    static constexpr bool has_nan() noexcept { return true; }

    constexpr f64c_t() noexcept : raw_ {0, 0} {}
    constexpr f64c_t(f64_t r) noexcept : raw_ {r.raw_, 0} {}
    constexpr f64c_t(double r) noexcept : raw_ {r, 0} {}
    constexpr f64c_t(f64_t r, f64_t i) noexcept : raw_ {r.raw_, i.raw_} {}
    constexpr f64c_t(double r, double i) noexcept : raw_ {r, i} {}

    static constexpr f64c_t from_raw(raw_t r) noexcept { return f64c_t {r.real, r.imag}; }

    constexpr f64_t real() const noexcept { return f64_t(raw_.real); }
    constexpr f64_t imag() const noexcept { return f64_t(raw_.imag); }

    static constexpr f64c_t zero() noexcept { return f64c_t {}; }
    static constexpr f64c_t one() noexcept { return f64c_t {1.0}; }
    static constexpr f64c_t i() noexcept { return f64c_t {0.0, 1.0}; }

    static constexpr f64c_t finite_max() noexcept { return f64c_t {1.7976931348623157e+308, 1.7976931348623157e+308}; }
    static constexpr f64c_t finite_min() noexcept {
        return f64c_t {-1.7976931348623157e+308, -1.7976931348623157e+308};
    }
    static constexpr f64c_t positive_infinity() noexcept {
        return f64c_t {f64_t::positive_infinity(), f64_t::positive_infinity()};
    }
    static constexpr f64c_t negative_infinity() noexcept {
        return f64c_t {f64_t::negative_infinity(), f64_t::negative_infinity()};
    }
    static constexpr f64c_t quiet_nan() noexcept { return f64c_t {f64_t::quiet_nan(), f64_t::quiet_nan()}; }

    constexpr f64c_t operator+() const noexcept { return *this; }
    constexpr f64c_t operator-() const noexcept { return f64c_t {-raw_.real, -raw_.imag}; }

    constexpr f64c_t operator+(f64c_t o) const noexcept {
        return f64c_t {raw_.real + o.raw_.real, raw_.imag + o.raw_.imag};
    }
    constexpr f64c_t operator-(f64c_t o) const noexcept {
        return f64c_t {raw_.real - o.raw_.real, raw_.imag - o.raw_.imag};
    }

    constexpr f64c_t operator*(f64c_t o) const noexcept {
        return f64c_t {raw_.real * o.raw_.real - raw_.imag * o.raw_.imag,
                       raw_.real * o.raw_.imag + raw_.imag * o.raw_.real};
    }

    inline f64c_t operator/(f64c_t o) const noexcept {
        double denom = o.raw_.real * o.raw_.real + o.raw_.imag * o.raw_.imag;
        return f64c_t {(raw_.real * o.raw_.real + raw_.imag * o.raw_.imag) / denom,
                       (raw_.imag * o.raw_.real - raw_.real * o.raw_.imag) / denom};
    }

    constexpr f64c_t &operator+=(f64c_t o) noexcept {
        raw_.real += o.raw_.real;
        raw_.imag += o.raw_.imag;
        return *this;
    }
    constexpr f64c_t &operator-=(f64c_t o) noexcept {
        raw_.real -= o.raw_.real;
        raw_.imag -= o.raw_.imag;
        return *this;
    }
    inline f64c_t &operator*=(f64c_t o) noexcept { return *this = *this * o; }
    inline f64c_t &operator/=(f64c_t o) noexcept { return *this = *this / o; }

    constexpr f64c_t operator*(f64_t s) const noexcept { return f64c_t {raw_.real * s.raw_, raw_.imag * s.raw_}; }
    constexpr f64c_t operator/(f64_t s) const noexcept { return f64c_t {raw_.real / s.raw_, raw_.imag / s.raw_}; }

    constexpr bool operator==(f64c_t o) const noexcept { return raw_.real == o.raw_.real && raw_.imag == o.raw_.imag; }
    constexpr bool operator!=(f64c_t o) const noexcept { return !(*this == o); }

    /** @brief Complex conjugate: `a - bi`. */
    constexpr f64c_t conj() const noexcept { return f64c_t {raw_.real, -raw_.imag}; }

    /** @brief Squared magnitude: `a² + b²`. */
    constexpr f64_t norm() const noexcept { return f64_t(raw_.real * raw_.real + raw_.imag * raw_.imag); }

    /** @brief Magnitude: `√(a² + b²)`. */
    inline f64_t abs() const noexcept { return norm().sqrt(); }

    /** @brief Phase angle: `atan2(b, a)`. */
    inline f64_t arg() const noexcept { return f64_t(raw_.imag).atan2(f64_t(raw_.real)); }

    /** @brief Complex exponential: `e^(a+bi) = e^a(cos(b) + i·sin(b))`. */
    inline f64c_t exp() const noexcept {
        f64_t ea = real().exp();
        return f64c_t {ea * imag().cos(), ea * imag().sin()};
    }

    /** @brief Complex natural logarithm: `ln|z| + i·arg(z)`. */
    inline f64c_t log() const noexcept { return f64c_t {abs().ln(), arg()}; }

    /** @brief Complex base-10 logarithm. */
    inline f64c_t log10() const noexcept {
        f64c_t ln_z = log();
        return ln_z * f64_t {0.4342944819032518}; // log10(e)
    }

    /** @brief Complex square root (principal value). */
    inline f64c_t sqrt() const noexcept {
        f64_t r = abs();
        f64_t half_arg = arg() * f64_t {0.5};
        f64_t sqrt_r = r.sqrt();
        return f64c_t {sqrt_r * half_arg.cos(), sqrt_r * half_arg.sin()};
    }

    /** @brief Complex power: `z^w = e^(w·ln(z))`. */
    inline f64c_t pow(f64c_t w) const noexcept { return (w * log()).exp(); }

    /** @brief Real power: `z^x`. */
    inline f64c_t powf(f64_t x) const noexcept {
        f64_t r = abs();
        f64_t theta = arg();
        f64_t r_pow = r.powf(x);
        f64_t new_theta = theta * x;
        return f64c_t {r_pow * new_theta.cos(), r_pow * new_theta.sin()};
    }

    /** @brief Complex sine: `sin(z) = (e^(iz) - e^(-iz)) / (2i)`. */
    inline f64c_t sin() const noexcept {
        // sin(a + bi) = sin(a)cosh(b) + i·cos(a)sinh(b)
        return f64c_t {real().sin() * imag().cosh(), real().cos() * imag().sinh()};
    }

    /** @brief Complex cosine: `cos(z) = (e^(iz) + e^(-iz)) / 2`. */
    inline f64c_t cos() const noexcept {
        // cos(a + bi) = cos(a)cosh(b) - i·sin(a)sinh(b)
        return f64c_t {real().cos() * imag().cosh(), -(real().sin() * imag().sinh())};
    }

    /** @brief Complex tangent: `tan(z) = sin(z) / cos(z)`. */
    inline f64c_t tan() const noexcept { return sin() / cos(); }

    /** @brief Complex arcsine: `asin(z) = -i·ln(iz + √(1 - z²))`. */
    inline f64c_t asin() const noexcept {
        f64c_t iz = f64c_t {-raw_.imag, raw_.real}; // i * z
        f64c_t one_minus_z2 = one() - *this * *this;
        f64c_t sqrt_part = one_minus_z2.sqrt();
        f64c_t ln_part = (iz + sqrt_part).log();
        return f64c_t {ln_part.raw_.imag, -ln_part.raw_.real}; // -i * ln_part
    }

    /** @brief Complex arccosine: `acos(z) = π/2 - asin(z)`. */
    inline f64c_t acos() const noexcept {
        constexpr double half_pi = 1.5707963267948966;
        f64c_t as = asin();
        return f64c_t {half_pi - as.raw_.real, -as.raw_.imag};
    }

    /** @brief Complex arctangent: `atan(z) = (i/2)·ln((i+z)/(i-z))`. */
    inline f64c_t atan() const noexcept {
        f64c_t i_unit = i();
        f64c_t num = i_unit + *this;
        f64c_t denom = i_unit - *this;
        f64c_t ln_part = (num / denom).log();
        // (i/2) * ln_part = (0 + 0.5i) * (a + bi) = -0.5b + 0.5ai
        return f64c_t {ln_part.raw_.imag * -0.5, ln_part.raw_.real * 0.5};
    }

    /** @brief Complex hyperbolic sine: `sinh(z) = (e^z - e^(-z)) / 2`. */
    inline f64c_t sinh() const noexcept {
        // sinh(a + bi) = sinh(a)cos(b) + i·cosh(a)sin(b)
        return f64c_t {real().sinh() * imag().cos(), real().cosh() * imag().sin()};
    }

    /** @brief Complex hyperbolic cosine: `cosh(z) = (e^z + e^(-z)) / 2`. */
    inline f64c_t cosh() const noexcept {
        // cosh(a + bi) = cosh(a)cos(b) + i·sinh(a)sin(b)
        return f64c_t {real().cosh() * imag().cos(), real().sinh() * imag().sin()};
    }

    /** @brief Complex hyperbolic tangent: `tanh(z) = sinh(z) / cosh(z)`. */
    inline f64c_t tanh() const noexcept { return sinh() / cosh(); }

    /** @brief Complex inverse hyperbolic sine: `asinh(z) = ln(z + √(z² + 1))`. */
    inline f64c_t asinh() const noexcept {
        f64c_t z2_plus_1 = *this * *this + one();
        return (*this + z2_plus_1.sqrt()).log();
    }

    /** @brief Complex inverse hyperbolic cosine: `acosh(z) = ln(z + √(z² - 1))`. */
    inline f64c_t acosh() const noexcept {
        f64c_t z2_minus_1 = *this * *this - one();
        return (*this + z2_minus_1.sqrt()).log();
    }

    /** @brief Complex inverse hyperbolic tangent: `atanh(z) = (1/2)·ln((1+z)/(1-z))`. */
    inline f64c_t atanh() const noexcept {
        f64c_t one_plus_z = one() + *this;
        f64c_t one_minus_z = one() - *this;
        f64c_t ln_part = (one_plus_z / one_minus_z).log();
        return f64c_t {ln_part.raw_.real * 0.5, ln_part.raw_.imag * 0.5};
    }

    /** @brief Reciprocal: `1 / z`. */
    inline f64c_t recip() const noexcept { return one() / *this; }
};

/**
 *  @brief Half-precision (16-bit) IEEE 754 floating-point wrapper.
 *
 *  Provides strong type identity for half-precision floats, compatible with NumKong
 *  kernels and `std::mdspan`. All kernel outputs are widened to f32.
 *
 *  Features:
 *  - Arithmetic operators (via f32 upcast/downcast)
 *  - Full math functions (via f32 upcast, compute, downcast)
 *  - Classification and special values
 *
 *  @note Not constexpr due to conversion functions. All math done in f32.
 */
struct f16_t {
    // Core type aliases
    using raw_t = nk_f16_t;
    using uint_t = nk_u16_t;

    using dot_result_t = f32_t;         // `nk_dot_f16` output (widened)
    using reduce_add_result_t = f32_t;  // `nk_reduce_add_f16` output (widened)
    using sqeuclidean_result_t = f32_t; // `nk_sqeuclidean_f16` output (widened)
    using angular_result_t = f32_t;     // `nk_angular_f16` output (widened)
    using mesh_result_t = f32_t;        // `nk_rmsd_f16` etc. output (widened)
    using scale_t = nk_f32_t;

    using dot_kernel_t = void (*)(raw_t const *, raw_t const *, nk_size_t, nk_f32_t *);
    using sqeuclidean_kernel_t = void (*)(raw_t const *, raw_t const *, nk_size_t, nk_f32_t *);
    using angular_kernel_t = void (*)(raw_t const *, raw_t const *, nk_size_t, nk_f32_t *);
    using trig_kernel_t = void (*)(raw_t const *, nk_size_t, raw_t *);
    using dots_packed_size_kernel_t = nk_size_t (*)(nk_size_t, nk_size_t);
    using dots_pack_kernel_t = void (*)(raw_t const *, nk_size_t, nk_size_t, nk_size_t, void *);
    using dots_packed_kernel_t = void (*)(raw_t const *, void const *, nk_f32_t *, nk_size_t, nk_size_t, nk_size_t,
                                          nk_size_t, nk_size_t);
    using dots_symmetric_kernel_t = void (*)(raw_t const *, nk_size_t, nk_size_t, nk_size_t, nk_f32_t *, nk_size_t,
                                             nk_size_t, nk_size_t);
    using mesh_kernel_t = void (*)(raw_t const *, raw_t const *, nk_size_t, nk_f32_t *, nk_f32_t *, nk_f32_t *,
                                   nk_f32_t *, nk_f32_t *);

    static constexpr nk_dtype_t dtype() noexcept { return nk_f16_k; }
    static constexpr char const *dtype_name() noexcept { return "f16"; }
    static constexpr unsigned bits_per_word() noexcept { return 16; }
    static constexpr unsigned bits_per_dimension() noexcept { return 16; }
    static constexpr unsigned bits_per_value() noexcept { return 16; }
    static constexpr unsigned mantissa_bits() noexcept { return 10; }
    static constexpr unsigned exponent_bits() noexcept { return 5; }
    static constexpr int min_exponent() noexcept { return -13; }
    static constexpr int max_exponent() noexcept { return 16; }
    static constexpr bool is_integer() noexcept { return false; }
    static constexpr bool is_signed() noexcept { return true; }
    static constexpr bool is_complex() noexcept { return false; }
    static constexpr bool is_exact() noexcept { return false; }
    static constexpr bool has_infinity() noexcept { return true; }
    static constexpr bool has_nan() noexcept { return true; }

    raw_t raw_;

    inline float to_f32() const noexcept {
        float r;
        nk_f16_to_f32(&raw_, &r);
        return r;
    }
    static inline f16_t from_f32(float v) noexcept {
        f16_t r;
        nk_f32_to_f16(&v, &r.raw_);
        return r;
    }

    constexpr f16_t() noexcept : raw_(0) {}
    f16_t(float v) noexcept { nk_f32_to_f16(&v, &raw_); }
    explicit f16_t(double v) noexcept {
        float f = static_cast<float>(v);
        nk_f32_to_f16(&f, &raw_);
    }
    template <std::integral integral_type_>
    f16_t(integral_type_ v) noexcept {
        float f = static_cast<float>(v);
        nk_f32_to_f16(&f, &raw_);
    }
    operator float() const noexcept { return to_f32(); }
    float raw() const noexcept { return to_f32(); }
    static f16_t from_raw(raw_t r) noexcept {
        f16_t v;
        v.raw_ = r;
        return v;
    }
    static constexpr f16_t from_bits(uint_t bits) noexcept {
        f16_t v;
        v.raw_ = std::bit_cast<raw_t>(bits);
        return v;
    }
    constexpr uint_t to_bits() const noexcept { return std::bit_cast<uint_t>(raw_); }

    static constexpr f16_t finite_max() noexcept { return from_bits(0x7BFF); }    // 65504.0
    static constexpr f16_t finite_min() noexcept { return from_bits(0xFBFF); }    // -65504.0
    static constexpr f16_t positive_min() noexcept { return from_bits(0x0400); }  // Smallest positive normal
    static constexpr f16_t subnormal_min() noexcept { return from_bits(0x0001); } // Smallest positive subnormal
    static constexpr f16_t epsilon() noexcept { return from_bits(0x1400); }       // 2⁻¹⁰ ≈ 0.00097656
    static constexpr f16_t positive_infinity() noexcept { return from_bits(0x7C00); }
    static constexpr f16_t negative_infinity() noexcept { return from_bits(0xFC00); }
    static constexpr f16_t quiet_nan() noexcept { return from_bits(0x7E00); }
    static constexpr f16_t signaling_nan() noexcept { return from_bits(0x7C01); }

    // Mathematical constants
    static inline f16_t pi_k() noexcept { return from_f32(3.14159265358979323846f); }
    static inline f16_t two_pi_k() noexcept { return from_f32(6.28318530717958647692f); }
    static inline f16_t half_pi_k() noexcept { return from_f32(1.57079632679489661923f); }
    static inline f16_t e_k() noexcept { return from_f32(2.71828182845904523536f); }
    static inline f16_t sqrt2_k() noexcept { return from_f32(1.41421356237309504880f); }
    static inline f16_t inv_sqrt2_k() noexcept { return from_f32(0.70710678118654752440f); }
    static inline f16_t ln2_k() noexcept { return from_f32(0.69314718055994530942f); }

    constexpr bool is_nan() const noexcept { return (to_bits() & 0x7FFF) > 0x7C00; }
    constexpr bool is_infinite() const noexcept { return (to_bits() & 0x7FFF) == 0x7C00; }
    constexpr bool is_finite() const noexcept { return (to_bits() & 0x7C00) != 0x7C00; }
    constexpr bool is_normal() const noexcept {
        uint_t exp = (to_bits() >> 10) & 0x1F;
        return exp != 0 && exp != 0x1F;
    }
    constexpr bool is_subnormal() const noexcept {
        return ((to_bits() >> 10) & 0x1F) == 0 && (to_bits() & 0x03FF) != 0;
    }
    constexpr bool is_sign_positive() const noexcept { return (to_bits() & 0x8000) == 0; }
    constexpr bool is_sign_negative() const noexcept { return (to_bits() & 0x8000) != 0; }

    inline f16_t operator+() const noexcept { return *this; }
    inline f16_t operator-() const noexcept { return from_bits(to_bits() ^ 0x8000); }
    inline f16_t operator+(f16_t o) const noexcept { return from_f32(to_f32() + o.to_f32()); }
    inline f16_t operator-(f16_t o) const noexcept { return from_f32(to_f32() - o.to_f32()); }
    inline f16_t operator*(f16_t o) const noexcept { return from_f32(to_f32() * o.to_f32()); }
    inline f16_t operator/(f16_t o) const noexcept { return from_f32(to_f32() / o.to_f32()); }

    inline f16_t &operator+=(f16_t o) noexcept { return *this = *this + o; }
    inline f16_t &operator-=(f16_t o) noexcept { return *this = *this - o; }
    inline f16_t &operator*=(f16_t o) noexcept { return *this = *this * o; }
    inline f16_t &operator/=(f16_t o) noexcept { return *this = *this / o; }

    inline bool operator==(f16_t o) const noexcept { return to_f32() == o.to_f32(); }
    inline bool operator!=(f16_t o) const noexcept { return to_f32() != o.to_f32(); }
    inline bool operator<(f16_t o) const noexcept { return to_f32() < o.to_f32(); }
    inline bool operator>(f16_t o) const noexcept { return to_f32() > o.to_f32(); }
    inline bool operator<=(f16_t o) const noexcept { return to_f32() <= o.to_f32(); }
    inline bool operator>=(f16_t o) const noexcept { return to_f32() >= o.to_f32(); }

    /** @brief Total ordering comparison (Rust-style). */
    inline int total_cmp(f16_t o) const noexcept {
        std::int16_t a = std::bit_cast<std::int16_t>(raw_);
        std::int16_t b = std::bit_cast<std::int16_t>(o.raw_);
        if (a < 0) a = std::int16_t(0x8000) - a;
        if (b < 0) b = std::int16_t(0x8000) - b;
        return (a > b) - (a < b);
    }

    constexpr f16_t abs() const noexcept { return from_bits(to_bits() & 0x7FFF); }
    constexpr f16_t copysign(f16_t sign) const noexcept {
        return from_bits((to_bits() & 0x7FFF) | (sign.to_bits() & 0x8000));
    }
    inline f16_t signum() const noexcept {
        if (is_nan()) return *this;
        return is_sign_negative() ? f16_t {-1.0f} : f16_t {1.0f};
    }

    inline f16_t floor() const noexcept { return from_f32(std::floor(to_f32())); }
    inline f16_t ceil() const noexcept { return from_f32(std::ceil(to_f32())); }
    inline f16_t round() const noexcept { return from_f32(std::round(to_f32())); }
    inline f16_t trunc() const noexcept { return from_f32(std::trunc(to_f32())); }
    inline f16_t fract() const noexcept {
        float f = to_f32();
        return from_f32(f - std::trunc(f));
    }

    inline f16_t sqrt() const noexcept { return from_f32(std::sqrt(to_f32())); }
    inline f16_t cbrt() const noexcept { return from_f32(std::cbrt(to_f32())); }
    inline f16_t rsqrt() const noexcept { return from_f32(1.0f / std::sqrt(to_f32())); }
    inline f16_t recip() const noexcept { return from_f32(1.0f / to_f32()); }
    inline f16_t mul_add(f16_t a, f16_t b) const noexcept {
        return from_f32(std::fma(to_f32(), a.to_f32(), b.to_f32()));
    }
    inline f16_t powf(f16_t exp) const noexcept { return from_f32(std::pow(to_f32(), exp.to_f32())); }

    inline f16_t exp() const noexcept { return from_f32(std::exp(to_f32())); }
    inline f16_t exp2() const noexcept { return from_f32(std::exp2(to_f32())); }
    inline f16_t exp_m1() const noexcept { return from_f32(std::expm1(to_f32())); }
    inline f16_t ln() const noexcept { return from_f32(std::log(to_f32())); }
    inline f16_t ln_1p() const noexcept { return from_f32(std::log1p(to_f32())); }
    inline f16_t log2() const noexcept { return from_f32(std::log2(to_f32())); }
    inline f16_t log10() const noexcept { return from_f32(std::log10(to_f32())); }

    inline f16_t sin() const noexcept { return from_f32(std::sin(to_f32())); }
    inline f16_t cos() const noexcept { return from_f32(std::cos(to_f32())); }
    inline f16_t tan() const noexcept { return from_f32(std::tan(to_f32())); }
    inline f16_t asin() const noexcept { return from_f32(std::asin(to_f32())); }
    inline f16_t acos() const noexcept { return from_f32(std::acos(to_f32())); }
    inline f16_t atan() const noexcept { return from_f32(std::atan(to_f32())); }
    inline f16_t atan2(f16_t x) const noexcept { return from_f32(std::atan2(to_f32(), x.to_f32())); }
    inline f16_t hypot(f16_t y) const noexcept { return from_f32(std::hypot(to_f32(), y.to_f32())); }
    inline f16_t to_radians() const noexcept { return from_f32(to_f32() * 0.017453292519943295f); }
    inline f16_t to_degrees() const noexcept { return from_f32(to_f32() * 57.29577951308232f); }

    inline f16_t sinh() const noexcept { return from_f32(std::sinh(to_f32())); }
    inline f16_t cosh() const noexcept { return from_f32(std::cosh(to_f32())); }
    inline f16_t tanh() const noexcept { return from_f32(std::tanh(to_f32())); }
    inline f16_t asinh() const noexcept { return from_f32(std::asinh(to_f32())); }
    inline f16_t acosh() const noexcept { return from_f32(std::acosh(to_f32())); }
    inline f16_t atanh() const noexcept { return from_f32(std::atanh(to_f32())); }

    inline f16_t min(f16_t o) const noexcept { return from_f32(std::fmin(to_f32(), o.to_f32())); }
    inline f16_t max(f16_t o) const noexcept { return from_f32(std::fmax(to_f32(), o.to_f32())); }
    inline f16_t clamp(f16_t lo, f16_t hi) const noexcept { return max(lo).min(hi); }
};

/**
 *  @brief Brain floating-point (16-bit) wrapper with f32-compatible exponent range.
 *
 *  bf16 has the same exponent range as f32 (8 bits) but only 7 mantissa bits.
 *  Provides strong type identity, compatible with NumKong kernels. All kernel
 *  outputs are widened to f32.
 *
 *  Features:
 *  - Same dynamic range as f32 (~38 orders of magnitude)
 *  - Lower precision than f16 (7 vs 10 mantissa bits)
 *  - Full math functions (via f32 upcast, compute, downcast)
 *
 *  @note Not constexpr due to conversion functions. All math done in f32.
 */
struct bf16_t {
    // Core type aliases
    using raw_t = nk_bf16_t;
    using uint_t = nk_u16_t;

    using dot_result_t = f32_t;         // `nk_dot_bf16` output (widened)
    using reduce_add_result_t = f32_t;  // `nk_reduce_add_bf16` output (widened)
    using sqeuclidean_result_t = f32_t; // `nk_sqeuclidean_bf16` output (widened)
    using angular_result_t = f32_t;     // `nk_angular_bf16` output (widened)
    using mesh_result_t = f32_t;        // `nk_rmsd_bf16` etc. output (widened)
    using scale_t = nk_f32_t;

    using dot_kernel_t = void (*)(raw_t const *, raw_t const *, nk_size_t, nk_f32_t *);
    using sqeuclidean_kernel_t = void (*)(raw_t const *, raw_t const *, nk_size_t, nk_f32_t *);
    using angular_kernel_t = void (*)(raw_t const *, raw_t const *, nk_size_t, nk_f32_t *);
    using dots_packed_size_kernel_t = nk_size_t (*)(nk_size_t, nk_size_t);
    using dots_pack_kernel_t = void (*)(raw_t const *, nk_size_t, nk_size_t, nk_size_t, void *);
    using dots_packed_kernel_t = void (*)(raw_t const *, void const *, nk_f32_t *, nk_size_t, nk_size_t, nk_size_t,
                                          nk_size_t, nk_size_t);
    using dots_symmetric_kernel_t = void (*)(raw_t const *, nk_size_t, nk_size_t, nk_size_t, nk_f32_t *, nk_size_t,
                                             nk_size_t, nk_size_t);
    using mesh_kernel_t = void (*)(raw_t const *, raw_t const *, nk_size_t, nk_f32_t *, nk_f32_t *, nk_f32_t *,
                                   nk_f32_t *, nk_f32_t *);
    using sparse_dot_index_t = u16_t;
    using sparse_dot_kernel_t = void (*)(nk_u16_t const *, nk_u16_t const *, raw_t const *, raw_t const *, nk_size_t,
                                         nk_size_t, nk_f32_t *);

    static constexpr nk_dtype_t dtype() noexcept { return nk_bf16_k; }
    static constexpr char const *dtype_name() noexcept { return "bf16"; }
    static constexpr unsigned bits_per_word() noexcept { return 16; }
    static constexpr unsigned bits_per_dimension() noexcept { return 16; }
    static constexpr unsigned bits_per_value() noexcept { return 16; }
    static constexpr unsigned mantissa_bits() noexcept { return 7; }
    static constexpr unsigned exponent_bits() noexcept { return 8; }
    static constexpr int min_exponent() noexcept { return -125; } // Same as f32
    static constexpr int max_exponent() noexcept { return 128; }  // Same as f32
    static constexpr bool is_integer() noexcept { return false; }
    static constexpr bool is_signed() noexcept { return true; }
    static constexpr bool is_complex() noexcept { return false; }
    static constexpr bool is_exact() noexcept { return false; }
    static constexpr bool has_infinity() noexcept { return true; }
    static constexpr bool has_nan() noexcept { return true; }

    raw_t raw_;

    inline float to_f32() const noexcept {
        float r;
        nk_bf16_to_f32(&raw_, &r);
        return r;
    }
    static inline bf16_t from_f32(float v) noexcept {
        bf16_t r;
        nk_f32_to_bf16(&v, &r.raw_);
        return r;
    }

    constexpr bf16_t() noexcept : raw_(0) {}
    bf16_t(float v) noexcept { nk_f32_to_bf16(&v, &raw_); }
    explicit bf16_t(double v) noexcept {
        float f = static_cast<float>(v);
        nk_f32_to_bf16(&f, &raw_);
    }
    template <std::integral integral_type_>
    bf16_t(integral_type_ v) noexcept {
        float f = static_cast<float>(v);
        nk_f32_to_bf16(&f, &raw_);
    }
    operator float() const noexcept { return to_f32(); }
    float raw() const noexcept { return to_f32(); }
    static bf16_t from_raw(raw_t r) noexcept {
        bf16_t v;
        v.raw_ = r;
        return v;
    }
    static constexpr bf16_t from_bits(uint_t bits) noexcept {
        bf16_t v;
        v.raw_ = std::bit_cast<raw_t>(bits);
        return v;
    }
    constexpr uint_t to_bits() const noexcept { return std::bit_cast<uint_t>(raw_); }

    static constexpr bf16_t finite_max() noexcept { return from_bits(0x7F7F); }    // ~3.39e38
    static constexpr bf16_t finite_min() noexcept { return from_bits(0xFF7F); }    // ~-3.39e38
    static constexpr bf16_t positive_min() noexcept { return from_bits(0x0080); }  // Smallest positive normal
    static constexpr bf16_t subnormal_min() noexcept { return from_bits(0x0001); } // Smallest positive subnormal
    static constexpr bf16_t epsilon() noexcept { return from_bits(0x3C00); }       // 2⁻⁷ ≈ 0.0078125
    static constexpr bf16_t positive_infinity() noexcept { return from_bits(0x7F80); }
    static constexpr bf16_t negative_infinity() noexcept { return from_bits(0xFF80); }
    static constexpr bf16_t quiet_nan() noexcept { return from_bits(0x7FC0); }
    static constexpr bf16_t signaling_nan() noexcept { return from_bits(0x7F81); }

    // Mathematical constants
    static inline bf16_t pi_k() noexcept { return from_f32(3.14159265358979323846f); }
    static inline bf16_t two_pi_k() noexcept { return from_f32(6.28318530717958647692f); }
    static inline bf16_t half_pi_k() noexcept { return from_f32(1.57079632679489661923f); }
    static inline bf16_t e_k() noexcept { return from_f32(2.71828182845904523536f); }
    static inline bf16_t sqrt2_k() noexcept { return from_f32(1.41421356237309504880f); }
    static inline bf16_t inv_sqrt2_k() noexcept { return from_f32(0.70710678118654752440f); }
    static inline bf16_t ln2_k() noexcept { return from_f32(0.69314718055994530942f); }

    constexpr bool is_nan() const noexcept { return (to_bits() & 0x7FFF) > 0x7F80; }
    constexpr bool is_infinite() const noexcept { return (to_bits() & 0x7FFF) == 0x7F80; }
    constexpr bool is_finite() const noexcept { return (to_bits() & 0x7F80) != 0x7F80; }
    constexpr bool is_normal() const noexcept {
        uint_t exp = (to_bits() >> 7) & 0xFF;
        return exp != 0 && exp != 0xFF;
    }
    constexpr bool is_subnormal() const noexcept { return ((to_bits() >> 7) & 0xFF) == 0 && (to_bits() & 0x007F) != 0; }
    constexpr bool is_sign_positive() const noexcept { return (to_bits() & 0x8000) == 0; }
    constexpr bool is_sign_negative() const noexcept { return (to_bits() & 0x8000) != 0; }

    inline bf16_t operator+() const noexcept { return *this; }
    inline bf16_t operator-() const noexcept { return from_bits(to_bits() ^ 0x8000); }
    inline bf16_t operator+(bf16_t o) const noexcept { return from_f32(to_f32() + o.to_f32()); }
    inline bf16_t operator-(bf16_t o) const noexcept { return from_f32(to_f32() - o.to_f32()); }
    inline bf16_t operator*(bf16_t o) const noexcept { return from_f32(to_f32() * o.to_f32()); }
    inline bf16_t operator/(bf16_t o) const noexcept { return from_f32(to_f32() / o.to_f32()); }

    inline bf16_t &operator+=(bf16_t o) noexcept { return *this = *this + o; }
    inline bf16_t &operator-=(bf16_t o) noexcept { return *this = *this - o; }
    inline bf16_t &operator*=(bf16_t o) noexcept { return *this = *this * o; }
    inline bf16_t &operator/=(bf16_t o) noexcept { return *this = *this / o; }

    inline bool operator==(bf16_t o) const noexcept { return to_f32() == o.to_f32(); }
    inline bool operator!=(bf16_t o) const noexcept { return to_f32() != o.to_f32(); }
    inline bool operator<(bf16_t o) const noexcept { return to_f32() < o.to_f32(); }
    inline bool operator>(bf16_t o) const noexcept { return to_f32() > o.to_f32(); }
    inline bool operator<=(bf16_t o) const noexcept { return to_f32() <= o.to_f32(); }
    inline bool operator>=(bf16_t o) const noexcept { return to_f32() >= o.to_f32(); }

    /** @brief Total ordering comparison (Rust-style). */
    inline int total_cmp(bf16_t o) const noexcept {
        std::int16_t a = std::bit_cast<std::int16_t>(raw_);
        std::int16_t b = std::bit_cast<std::int16_t>(o.raw_);
        if (a < 0) a = std::int16_t(0x8000) - a;
        if (b < 0) b = std::int16_t(0x8000) - b;
        return (a > b) - (a < b);
    }

    constexpr bf16_t abs() const noexcept { return from_bits(to_bits() & 0x7FFF); }
    constexpr bf16_t copysign(bf16_t sign) const noexcept {
        return from_bits((to_bits() & 0x7FFF) | (sign.to_bits() & 0x8000));
    }
    inline bf16_t signum() const noexcept {
        if (is_nan()) return *this;
        return is_sign_negative() ? bf16_t {-1.0f} : bf16_t {1.0f};
    }

    inline bf16_t floor() const noexcept { return from_f32(std::floor(to_f32())); }
    inline bf16_t ceil() const noexcept { return from_f32(std::ceil(to_f32())); }
    inline bf16_t round() const noexcept { return from_f32(std::round(to_f32())); }
    inline bf16_t trunc() const noexcept { return from_f32(std::trunc(to_f32())); }
    inline bf16_t fract() const noexcept {
        float f = to_f32();
        return from_f32(f - std::trunc(f));
    }

    inline bf16_t sqrt() const noexcept { return from_f32(std::sqrt(to_f32())); }
    inline bf16_t cbrt() const noexcept { return from_f32(std::cbrt(to_f32())); }
    inline bf16_t rsqrt() const noexcept { return from_f32(1.0f / std::sqrt(to_f32())); }
    inline bf16_t recip() const noexcept { return from_f32(1.0f / to_f32()); }
    inline bf16_t mul_add(bf16_t a, bf16_t b) const noexcept {
        return from_f32(std::fma(to_f32(), a.to_f32(), b.to_f32()));
    }
    inline bf16_t powf(bf16_t exp) const noexcept { return from_f32(std::pow(to_f32(), exp.to_f32())); }

    inline bf16_t exp() const noexcept { return from_f32(std::exp(to_f32())); }
    inline bf16_t exp2() const noexcept { return from_f32(std::exp2(to_f32())); }
    inline bf16_t exp_m1() const noexcept { return from_f32(std::expm1(to_f32())); }
    inline bf16_t ln() const noexcept { return from_f32(std::log(to_f32())); }
    inline bf16_t ln_1p() const noexcept { return from_f32(std::log1p(to_f32())); }
    inline bf16_t log2() const noexcept { return from_f32(std::log2(to_f32())); }
    inline bf16_t log10() const noexcept { return from_f32(std::log10(to_f32())); }

    inline bf16_t sin() const noexcept { return from_f32(std::sin(to_f32())); }
    inline bf16_t cos() const noexcept { return from_f32(std::cos(to_f32())); }
    inline bf16_t tan() const noexcept { return from_f32(std::tan(to_f32())); }
    inline bf16_t asin() const noexcept { return from_f32(std::asin(to_f32())); }
    inline bf16_t acos() const noexcept { return from_f32(std::acos(to_f32())); }
    inline bf16_t atan() const noexcept { return from_f32(std::atan(to_f32())); }
    inline bf16_t atan2(bf16_t x) const noexcept { return from_f32(std::atan2(to_f32(), x.to_f32())); }
    inline bf16_t hypot(bf16_t y) const noexcept { return from_f32(std::hypot(to_f32(), y.to_f32())); }
    inline bf16_t to_radians() const noexcept { return from_f32(to_f32() * 0.017453292519943295f); }
    inline bf16_t to_degrees() const noexcept { return from_f32(to_f32() * 57.29577951308232f); }

    inline bf16_t sinh() const noexcept { return from_f32(std::sinh(to_f32())); }
    inline bf16_t cosh() const noexcept { return from_f32(std::cosh(to_f32())); }
    inline bf16_t tanh() const noexcept { return from_f32(std::tanh(to_f32())); }
    inline bf16_t asinh() const noexcept { return from_f32(std::asinh(to_f32())); }
    inline bf16_t acosh() const noexcept { return from_f32(std::acosh(to_f32())); }
    inline bf16_t atanh() const noexcept { return from_f32(std::atanh(to_f32())); }

    inline bf16_t min(bf16_t o) const noexcept { return from_f32(std::fmin(to_f32(), o.to_f32())); }
    inline bf16_t max(bf16_t o) const noexcept { return from_f32(std::fmax(to_f32(), o.to_f32())); }
    inline bf16_t clamp(bf16_t lo, bf16_t hi) const noexcept { return max(lo).min(hi); }
};

/**
 *  @brief Half-precision complex number wrapper using composition.
 *
 *  Provides strong type identity for complex f16, compatible with NumKong kernels.
 *  Uses composition of two `f16_t` components. Results widened to f32c for precision.
 *
 *  Features:
 *  - Real/imaginary accessors via `real()` and `imag()`
 *  - Complex arithmetic: `+`, `-`, `*`, `/`
 *  - Complex-specific: `conj()`, `norm()`, `abs()`
 *
 *  @note Math computed via f32 upcast for precision.
 */
struct f16c_t {
    using component_t = f16_t;
    using raw_t = nk_f16c_t;

    using dot_result_t = f32c_t;    // widened to f32c
    using vdot_result_t = f32c_t;   // widened to f32c
    using curved_result_t = f32c_t; // widened to f32c

    // Kernel signatures: input f16c, output widened to f32c
    using dot_kernel_t = void (*)(raw_t const *, raw_t const *, nk_size_t, nk_f32c_t *);
    using vdot_kernel_t = void (*)(raw_t const *, raw_t const *, nk_size_t, nk_f32c_t *);
    using bilinear_kernel_t = void (*)(raw_t const *, raw_t const *, raw_t const *, nk_size_t, nk_f32c_t *);

    raw_t raw_;

    static constexpr nk_dtype_t dtype() noexcept { return nk_f16c_k; }
    static constexpr char const *dtype_name() noexcept { return "f16c"; }
    static constexpr unsigned bits_per_word() noexcept { return 16; }      // One f16 component
    static constexpr unsigned bits_per_dimension() noexcept { return 32; } // One complex number
    static constexpr unsigned bits_per_value() noexcept { return 32; }
    static constexpr unsigned component_bits() noexcept { return 16; }
    static constexpr bool is_integer() noexcept { return false; }
    static constexpr bool is_signed() noexcept { return true; }
    static constexpr bool is_complex() noexcept { return true; }
    static constexpr bool is_exact() noexcept { return false; }
    static constexpr bool has_infinity() noexcept { return true; }
    static constexpr bool has_nan() noexcept { return true; }

    constexpr f16c_t() noexcept : raw_ {0, 0} {}
    constexpr f16c_t(f16_t r) noexcept : raw_ {r.raw_, 0} {}
    constexpr f16c_t(f16_t r, f16_t i) noexcept : raw_ {r.raw_, i.raw_} {}

    static constexpr f16c_t from_raw(raw_t r) noexcept {
        return f16c_t {f16_t::from_raw(r.real), f16_t::from_raw(r.imag)};
    }

    inline f16_t real() const noexcept { return f16_t::from_raw(raw_.real); }
    inline f16_t imag() const noexcept { return f16_t::from_raw(raw_.imag); }

    static constexpr f16c_t zero() noexcept { return f16c_t {}; }

    static constexpr f16c_t finite_max() noexcept { return f16c_t {f16_t::finite_max(), f16_t::finite_max()}; }
    static constexpr f16c_t finite_min() noexcept { return f16c_t {f16_t::finite_min(), f16_t::finite_min()}; }
    static constexpr f16c_t positive_infinity() noexcept {
        return f16c_t {f16_t::positive_infinity(), f16_t::positive_infinity()};
    }
    static constexpr f16c_t negative_infinity() noexcept {
        return f16c_t {f16_t::negative_infinity(), f16_t::negative_infinity()};
    }
    static constexpr f16c_t quiet_nan() noexcept { return f16c_t {f16_t::quiet_nan(), f16_t::quiet_nan()}; }

    inline f16c_t operator+() const noexcept { return *this; }
    inline f16c_t operator-() const noexcept { return f16c_t {-real(), -imag()}; }

    inline f16c_t operator+(f16c_t o) const noexcept { return f16c_t {real() + o.real(), imag() + o.imag()}; }
    inline f16c_t operator-(f16c_t o) const noexcept { return f16c_t {real() - o.real(), imag() - o.imag()}; }

    inline f16c_t operator*(f16c_t o) const noexcept {
        return f16c_t {real() * o.real() - imag() * o.imag(), real() * o.imag() + imag() * o.real()};
    }

    inline f16c_t operator/(f16c_t o) const noexcept {
        f16_t denom = o.real() * o.real() + o.imag() * o.imag();
        return f16c_t {(real() * o.real() + imag() * o.imag()) / denom,
                       (imag() * o.real() - real() * o.imag()) / denom};
    }

    inline f16c_t &operator+=(f16c_t o) noexcept {
        raw_.real = (real() + o.real()).raw_;
        raw_.imag = (imag() + o.imag()).raw_;
        return *this;
    }
    inline f16c_t &operator-=(f16c_t o) noexcept {
        raw_.real = (real() - o.real()).raw_;
        raw_.imag = (imag() - o.imag()).raw_;
        return *this;
    }
    inline f16c_t &operator*=(f16c_t o) noexcept { return *this = *this * o; }
    inline f16c_t &operator/=(f16c_t o) noexcept { return *this = *this / o; }

    inline bool operator==(f16c_t o) const noexcept { return raw_.real == o.raw_.real && raw_.imag == o.raw_.imag; }
    inline bool operator!=(f16c_t o) const noexcept { return !(*this == o); }

    inline f16c_t conj() const noexcept { return f16c_t {real(), -imag()}; }
    inline f16_t norm() const noexcept { return real() * real() + imag() * imag(); }
    inline f16_t abs() const noexcept { return norm().sqrt(); }
};

/**
 *  @brief BFloat16 complex number wrapper using composition.
 *
 *  Provides strong type identity for complex bf16, compatible with NumKong kernels.
 *  Uses composition of two `bf16_t` components. Results widened to f32c for precision.
 *
 *  Features:
 *  - Real/imaginary accessors via `real()` and `imag()`
 *  - Complex arithmetic: `+`, `-`, `*`, `/`
 *  - Complex-specific: `conj()`, `norm()`, `abs()`
 *
 *  @note Math computed via f32 upcast for precision.
 */
struct bf16c_t {
    using component_t = bf16_t;
    using raw_t = nk_bf16c_t;

    using dot_result_t = f32c_t;    // widened to f32c
    using vdot_result_t = f32c_t;   // widened to f32c
    using curved_result_t = f32c_t; // widened to f32c

    // Kernel signatures: input bf16c, output widened to f32c
    using dot_kernel_t = void (*)(raw_t const *, raw_t const *, nk_size_t, nk_f32c_t *);
    using vdot_kernel_t = void (*)(raw_t const *, raw_t const *, nk_size_t, nk_f32c_t *);
    using bilinear_kernel_t = void (*)(raw_t const *, raw_t const *, raw_t const *, nk_size_t, nk_f32c_t *);

    raw_t raw_;

    static constexpr nk_dtype_t dtype() noexcept { return nk_bf16c_k; }
    static constexpr char const *dtype_name() noexcept { return "bf16c"; }
    static constexpr unsigned bits_per_word() noexcept { return 16; }      // One bf16 component
    static constexpr unsigned bits_per_dimension() noexcept { return 32; } // One complex number
    static constexpr unsigned bits_per_value() noexcept { return 32; }
    static constexpr unsigned component_bits() noexcept { return 16; }
    static constexpr bool is_integer() noexcept { return false; }
    static constexpr bool is_signed() noexcept { return true; }
    static constexpr bool is_complex() noexcept { return true; }
    static constexpr bool is_exact() noexcept { return false; }
    static constexpr bool has_infinity() noexcept { return true; }
    static constexpr bool has_nan() noexcept { return true; }

    constexpr bf16c_t() noexcept : raw_ {0, 0} {}
    constexpr bf16c_t(bf16_t r) noexcept : raw_ {r.raw_, 0} {}
    constexpr bf16c_t(bf16_t r, bf16_t i) noexcept : raw_ {r.raw_, i.raw_} {}

    static constexpr bf16c_t from_raw(raw_t r) noexcept {
        return bf16c_t {bf16_t::from_raw(r.real), bf16_t::from_raw(r.imag)};
    }

    inline bf16_t real() const noexcept { return bf16_t::from_raw(raw_.real); }
    inline bf16_t imag() const noexcept { return bf16_t::from_raw(raw_.imag); }

    static constexpr bf16c_t zero() noexcept { return bf16c_t {}; }

    static constexpr bf16c_t finite_max() noexcept { return bf16c_t {bf16_t::finite_max(), bf16_t::finite_max()}; }
    static constexpr bf16c_t finite_min() noexcept { return bf16c_t {bf16_t::finite_min(), bf16_t::finite_min()}; }
    static constexpr bf16c_t positive_infinity() noexcept {
        return bf16c_t {bf16_t::positive_infinity(), bf16_t::positive_infinity()};
    }
    static constexpr bf16c_t negative_infinity() noexcept {
        return bf16c_t {bf16_t::negative_infinity(), bf16_t::negative_infinity()};
    }
    static constexpr bf16c_t quiet_nan() noexcept { return bf16c_t {bf16_t::quiet_nan(), bf16_t::quiet_nan()}; }

    inline bf16c_t operator+() const noexcept { return *this; }
    inline bf16c_t operator-() const noexcept { return bf16c_t {-real(), -imag()}; }

    inline bf16c_t operator+(bf16c_t o) const noexcept { return bf16c_t {real() + o.real(), imag() + o.imag()}; }
    inline bf16c_t operator-(bf16c_t o) const noexcept { return bf16c_t {real() - o.real(), imag() - o.imag()}; }

    inline bf16c_t operator*(bf16c_t o) const noexcept {
        return bf16c_t {real() * o.real() - imag() * o.imag(), real() * o.imag() + imag() * o.real()};
    }

    inline bf16c_t operator/(bf16c_t o) const noexcept {
        bf16_t denom = o.real() * o.real() + o.imag() * o.imag();
        return bf16c_t {(real() * o.real() + imag() * o.imag()) / denom,
                        (imag() * o.real() - real() * o.imag()) / denom};
    }

    inline bf16c_t &operator+=(bf16c_t o) noexcept {
        raw_.real = (real() + o.real()).raw_;
        raw_.imag = (imag() + o.imag()).raw_;
        return *this;
    }
    inline bf16c_t &operator-=(bf16c_t o) noexcept {
        raw_.real = (real() - o.real()).raw_;
        raw_.imag = (imag() - o.imag()).raw_;
        return *this;
    }
    inline bf16c_t &operator*=(bf16c_t o) noexcept { return *this = *this * o; }
    inline bf16c_t &operator/=(bf16c_t o) noexcept { return *this = *this / o; }

    inline bool operator==(bf16c_t o) const noexcept { return raw_.real == o.raw_.real && raw_.imag == o.raw_.imag; }
    inline bool operator!=(bf16c_t o) const noexcept { return !(*this == o); }

    inline bf16c_t conj() const noexcept { return bf16c_t {real(), -imag()}; }
    inline bf16_t norm() const noexcept { return real() * real() + imag() * imag(); }
    inline bf16_t abs() const noexcept { return norm().sqrt(); }
};

/**
 *  @brief FP8 E4M3 (8-bit float with 4-bit exponent, 3-bit mantissa) wrapper.
 *
 *  E4M3 is an 8-bit floating-point format optimized for machine learning inference.
 *  Range: ±448, smallest normal: 2⁻⁶. No infinity representation (all exponent bits
 *  set = NaN). Provides strong type identity, compatible with NumKong kernels.
 *
 *  Features:
 *  - Compact 8-bit storage for ML weights and activations
 *  - Full math functions (via f32 upcast, compute, downcast)
 *  - All kernel outputs widened to f32
 *
 *  @note Not constexpr due to conversion functions. All math done in f32.
 */
struct e4m3_t {
    // Core type aliases
    using raw_t = nk_e4m3_t;
    using uint_t = nk_u8_t;

    using dot_result_t = f32_t;         // `nk_dot_e4m3` output (widened)
    using reduce_add_result_t = f32_t;  // `nk_reduce_add_e4m3` output (widened)
    using sqeuclidean_result_t = f32_t; // `nk_sqeuclidean_e4m3` output (widened)
    using scale_t = nk_f32_t;

    using dot_kernel_t = void (*)(raw_t const *, raw_t const *, nk_size_t, nk_f32_t *);
    using reduce_add_kernel_t = void (*)(raw_t const *, nk_size_t, nk_size_t, nk_f32_t *);
    using reduce_extremum_kernel_t = void (*)(raw_t const *, nk_size_t, nk_size_t, raw_t *, nk_size_t *);
    using scale_kernel_t = void (*)(raw_t const *, nk_size_t, scale_t const *, scale_t const *, raw_t *);
    using sum_kernel_t = void (*)(raw_t const *, raw_t const *, nk_size_t, raw_t *);
    using wsum_kernel_t = void (*)(raw_t const *, raw_t const *, nk_size_t, scale_t const *, scale_t const *, raw_t *);
    using fma_kernel_t = void (*)(raw_t const *, raw_t const *, raw_t const *, nk_size_t, scale_t const *,
                                  scale_t const *, raw_t *);
    using dots_packed_size_kernel_t = nk_size_t (*)(nk_size_t, nk_size_t);
    using dots_pack_kernel_t = void (*)(raw_t const *, nk_size_t, nk_size_t, nk_size_t, void *);
    using dots_symmetric_kernel_t = void (*)(raw_t const *, nk_size_t, nk_size_t, nk_size_t, nk_f32_t *, nk_size_t,
                                             nk_size_t, nk_size_t);
    using dots_packed_kernel_t = void (*)(raw_t const *, void const *, nk_f32_t *, nk_size_t, nk_size_t, nk_size_t,
                                          nk_size_t, nk_size_t);

    static constexpr nk_dtype_t dtype() noexcept { return nk_e4m3_k; }
    static constexpr char const *dtype_name() noexcept { return "e4m3"; }
    static constexpr unsigned bits_per_word() noexcept { return 8; }
    static constexpr unsigned bits_per_dimension() noexcept { return 8; }
    static constexpr unsigned bits_per_value() noexcept { return 8; }
    static constexpr unsigned mantissa_bits() noexcept { return 3; }
    static constexpr unsigned exponent_bits() noexcept { return 4; }
    static constexpr bool is_integer() noexcept { return false; }
    static constexpr bool is_signed() noexcept { return true; }
    static constexpr bool is_complex() noexcept { return false; }
    static constexpr bool is_exact() noexcept { return false; }
    static constexpr bool has_infinity() noexcept { return false; } // E4M3 has no infinity
    static constexpr bool has_nan() noexcept { return true; }

    raw_t raw_;

    inline float to_f32() const noexcept {
        float r;
        nk_e4m3_to_f32(&raw_, &r);
        return r;
    }
    static inline e4m3_t from_f32(float v) noexcept {
        e4m3_t r;
        nk_f32_to_e4m3(&v, &r.raw_);
        return r;
    }

    e4m3_t() noexcept : raw_(0) {}
    e4m3_t(float v) noexcept { nk_f32_to_e4m3(&v, &raw_); }
    explicit e4m3_t(double v) noexcept {
        float f = static_cast<float>(v);
        nk_f32_to_e4m3(&f, &raw_);
    }
    template <std::integral integral_type_>
    e4m3_t(integral_type_ v) noexcept {
        float f = static_cast<float>(v);
        nk_f32_to_e4m3(&f, &raw_);
    }
    operator float() const noexcept { return to_f32(); }
    float raw() const noexcept { return to_f32(); }
    static e4m3_t from_raw(raw_t r) noexcept {
        e4m3_t v;
        v.raw_ = r;
        return v;
    }
    static constexpr e4m3_t from_bits(uint_t bits) noexcept {
        e4m3_t v;
        v.raw_ = bits;
        return v;
    }
    constexpr uint_t to_bits() const noexcept { return raw_; }

    // Exp all 1s (0xF) with non-zero mantissa = NaN, no infinity representation
    static constexpr e4m3_t finite_max() noexcept { return from_bits(0x7E); }    // 448.0
    static constexpr e4m3_t finite_min() noexcept { return from_bits(0xFE); }    // -448.0
    static constexpr e4m3_t positive_min() noexcept { return from_bits(0x08); }  // Smallest positive normal (2⁻⁶)
    static constexpr e4m3_t subnormal_min() noexcept { return from_bits(0x01); } // Smallest positive subnormal
    static constexpr e4m3_t quiet_nan() noexcept { return from_bits(0x7F); }     // +NaN

    // Mathematical constants
    static constexpr e4m3_t pi_k() noexcept { return from_bits(0x45); }        // 3.25 (closest to π)
    static constexpr e4m3_t two_pi_k() noexcept { return from_bits(0x4D); }    // 6.5 (closest to 2π)
    static constexpr e4m3_t half_pi_k() noexcept { return from_bits(0x3D); }   // 1.625 (closest to π/2)
    static constexpr e4m3_t e_k() noexcept { return from_bits(0x43); }         // 2.75 (closest to e)
    static constexpr e4m3_t sqrt2_k() noexcept { return from_bits(0x3B); }     // 1.375 (closest to √2)
    static constexpr e4m3_t inv_sqrt2_k() noexcept { return from_bits(0x33); } // 0.6875 (closest to 1/√2)
    static constexpr e4m3_t ln2_k() noexcept { return from_bits(0x33); }       // 0.6875 (closest to ln(2))

    constexpr bool is_nan() const noexcept { return (raw_ & 0x7F) == 0x7F; }
    constexpr bool is_infinite() const noexcept { return false; } // E4M3 has no infinity
    constexpr bool is_finite() const noexcept { return !is_nan(); }
    constexpr bool is_normal() const noexcept {
        uint_t exp = (raw_ >> 3) & 0x0F;
        return exp != 0 && exp != 0x0F;
    }
    constexpr bool is_subnormal() const noexcept { return ((raw_ >> 3) & 0x0F) == 0 && (raw_ & 0x07) != 0; }
    constexpr bool is_sign_positive() const noexcept { return (raw_ & 0x80) == 0; }
    constexpr bool is_sign_negative() const noexcept { return (raw_ & 0x80) != 0; }

    inline e4m3_t operator+() const noexcept { return *this; }
    inline e4m3_t operator-() const noexcept { return from_bits(raw_ ^ 0x80); }
    inline e4m3_t operator+(e4m3_t o) const noexcept { return from_f32(to_f32() + o.to_f32()); }
    inline e4m3_t operator-(e4m3_t o) const noexcept { return from_f32(to_f32() - o.to_f32()); }
    inline e4m3_t operator*(e4m3_t o) const noexcept { return from_f32(to_f32() * o.to_f32()); }
    inline e4m3_t operator/(e4m3_t o) const noexcept { return from_f32(to_f32() / o.to_f32()); }

    inline e4m3_t &operator+=(e4m3_t o) noexcept { return *this = *this + o; }
    inline e4m3_t &operator-=(e4m3_t o) noexcept { return *this = *this - o; }
    inline e4m3_t &operator*=(e4m3_t o) noexcept { return *this = *this * o; }
    inline e4m3_t &operator/=(e4m3_t o) noexcept { return *this = *this / o; }

    inline bool operator==(e4m3_t o) const noexcept { return to_f32() == o.to_f32(); }
    inline bool operator!=(e4m3_t o) const noexcept { return to_f32() != o.to_f32(); }
    inline bool operator<(e4m3_t o) const noexcept { return to_f32() < o.to_f32(); }
    inline bool operator>(e4m3_t o) const noexcept { return to_f32() > o.to_f32(); }
    inline bool operator<=(e4m3_t o) const noexcept { return to_f32() <= o.to_f32(); }
    inline bool operator>=(e4m3_t o) const noexcept { return to_f32() >= o.to_f32(); }

    /** @brief Total ordering comparison (Rust-style). */
    inline int total_cmp(e4m3_t o) const noexcept {
        std::int8_t a = std::bit_cast<std::int8_t>(raw_);
        std::int8_t b = std::bit_cast<std::int8_t>(o.raw_);
        if (a < 0) a = std::int8_t(0x80) - a;
        if (b < 0) b = std::int8_t(0x80) - b;
        return (a > b) - (a < b);
    }

    constexpr e4m3_t abs() const noexcept { return from_bits(raw_ & 0x7F); }
    constexpr e4m3_t copysign(e4m3_t sign) const noexcept { return from_bits((raw_ & 0x7F) | (sign.raw_ & 0x80)); }
    inline e4m3_t signum() const noexcept {
        if (is_nan()) return *this;
        return is_sign_negative() ? e4m3_t {-1.0f} : e4m3_t {1.0f};
    }

    inline e4m3_t floor() const noexcept { return from_f32(std::floor(to_f32())); }
    inline e4m3_t ceil() const noexcept { return from_f32(std::ceil(to_f32())); }
    inline e4m3_t round() const noexcept { return from_f32(std::round(to_f32())); }
    inline e4m3_t trunc() const noexcept { return from_f32(std::trunc(to_f32())); }
    inline e4m3_t fract() const noexcept {
        float f = to_f32();
        return from_f32(f - std::trunc(f));
    }

    inline e4m3_t sqrt() const noexcept { return from_f32(std::sqrt(to_f32())); }
    inline e4m3_t cbrt() const noexcept { return from_f32(std::cbrt(to_f32())); }
    inline e4m3_t rsqrt() const noexcept { return from_f32(1.0f / std::sqrt(to_f32())); }
    inline e4m3_t recip() const noexcept { return from_f32(1.0f / to_f32()); }
    inline e4m3_t mul_add(e4m3_t a, e4m3_t b) const noexcept {
        return from_f32(std::fma(to_f32(), a.to_f32(), b.to_f32()));
    }
    inline e4m3_t powf(e4m3_t exp) const noexcept { return from_f32(std::pow(to_f32(), exp.to_f32())); }

    inline e4m3_t exp() const noexcept { return from_f32(std::exp(to_f32())); }
    inline e4m3_t exp2() const noexcept { return from_f32(std::exp2(to_f32())); }
    inline e4m3_t ln() const noexcept { return from_f32(std::log(to_f32())); }
    inline e4m3_t log2() const noexcept { return from_f32(std::log2(to_f32())); }
    inline e4m3_t log10() const noexcept { return from_f32(std::log10(to_f32())); }

    inline e4m3_t sin() const noexcept { return from_f32(std::sin(to_f32())); }
    inline e4m3_t cos() const noexcept { return from_f32(std::cos(to_f32())); }
    inline e4m3_t tan() const noexcept { return from_f32(std::tan(to_f32())); }
    inline e4m3_t tanh() const noexcept { return from_f32(std::tanh(to_f32())); }

    inline e4m3_t min(e4m3_t o) const noexcept { return from_f32(std::fmin(to_f32(), o.to_f32())); }
    inline e4m3_t max(e4m3_t o) const noexcept { return from_f32(std::fmax(to_f32(), o.to_f32())); }
    inline e4m3_t clamp(e4m3_t lo, e4m3_t hi) const noexcept { return max(lo).min(hi); }

    /** @brief Saturating addition: clamps to finite range on overflow. */
    inline e4m3_t saturating_add(e4m3_t o) const noexcept {
        float result = to_f32() + o.to_f32();
        if (result >= finite_max().to_f32()) return finite_max();
        if (result <= finite_min().to_f32()) return finite_min();
        return from_f32(result);
    }

    /** @brief Saturating subtraction: clamps to finite range on overflow. */
    inline e4m3_t saturating_sub(e4m3_t o) const noexcept {
        float result = to_f32() - o.to_f32();
        if (result >= finite_max().to_f32()) return finite_max();
        if (result <= finite_min().to_f32()) return finite_min();
        return from_f32(result);
    }
};

/**
 *  @brief FP8 E5M2 (8-bit float with 5-bit exponent, 2-bit mantissa) wrapper.
 *
 *  E5M2 is an 8-bit floating-point format with the same exponent range as f16.
 *  Range: ±57344, supports infinity and NaN. Provides strong type identity,
 *  compatible with NumKong kernels.
 *
 *  Features:
 *  - Same dynamic range as f16 (5-bit exponent)
 *  - Lower precision than E4M3 (2 vs 3 mantissa bits)
 *  - Full math functions (via f32 upcast, compute, downcast)
 *  - All kernel outputs widened to f32
 *
 *  @note Not constexpr due to conversion functions. All math done in f32.
 */
struct e5m2_t {
    // Core type aliases
    using raw_t = nk_e5m2_t;
    using uint_t = nk_u8_t;

    using dot_result_t = f32_t;         // `nk_dot_e5m2` output (widened)
    using reduce_add_result_t = f32_t;  // `nk_reduce_add_e5m2` output (widened)
    using sqeuclidean_result_t = f32_t; // `nk_sqeuclidean_e5m2` output (widened)
    using scale_t = nk_f32_t;

    using dot_kernel_t = void (*)(raw_t const *, raw_t const *, nk_size_t, nk_f32_t *);
    using reduce_add_kernel_t = void (*)(raw_t const *, nk_size_t, nk_size_t, nk_f32_t *);
    using reduce_extremum_kernel_t = void (*)(raw_t const *, nk_size_t, nk_size_t, raw_t *, nk_size_t *);
    using scale_kernel_t = void (*)(raw_t const *, nk_size_t, scale_t const *, scale_t const *, raw_t *);
    using sum_kernel_t = void (*)(raw_t const *, raw_t const *, nk_size_t, raw_t *);
    using wsum_kernel_t = void (*)(raw_t const *, raw_t const *, nk_size_t, scale_t const *, scale_t const *, raw_t *);
    using fma_kernel_t = void (*)(raw_t const *, raw_t const *, raw_t const *, nk_size_t, scale_t const *,
                                  scale_t const *, raw_t *);
    using dots_packed_size_kernel_t = nk_size_t (*)(nk_size_t, nk_size_t);
    using dots_symmetric_kernel_t = void (*)(raw_t const *, nk_size_t, nk_size_t, nk_size_t, nk_f32_t *, nk_size_t,
                                             nk_size_t, nk_size_t);
    using dots_pack_kernel_t = void (*)(raw_t const *, nk_size_t, nk_size_t, nk_size_t, void *);
    using dots_packed_kernel_t = void (*)(raw_t const *, void const *, nk_f32_t *, nk_size_t, nk_size_t, nk_size_t,
                                          nk_size_t, nk_size_t);

    static constexpr nk_dtype_t dtype() noexcept { return nk_e5m2_k; }
    static constexpr char const *dtype_name() noexcept { return "e5m2"; }
    static constexpr unsigned bits_per_word() noexcept { return 8; }
    static constexpr unsigned bits_per_dimension() noexcept { return 8; }
    static constexpr unsigned bits_per_value() noexcept { return 8; }
    static constexpr unsigned mantissa_bits() noexcept { return 2; }
    static constexpr unsigned exponent_bits() noexcept { return 5; }
    static constexpr bool is_integer() noexcept { return false; }
    static constexpr bool is_signed() noexcept { return true; }
    static constexpr bool is_complex() noexcept { return false; }
    static constexpr bool is_exact() noexcept { return false; }
    static constexpr bool has_infinity() noexcept { return true; } // E5M2 has infinity
    static constexpr bool has_nan() noexcept { return true; }

    raw_t raw_;

    inline float to_f32() const noexcept {
        float r;
        nk_e5m2_to_f32(&raw_, &r);
        return r;
    }
    static inline e5m2_t from_f32(float v) noexcept {
        e5m2_t r;
        nk_f32_to_e5m2(&v, &r.raw_);
        return r;
    }

    e5m2_t() noexcept : raw_(0) {}
    e5m2_t(float v) noexcept { nk_f32_to_e5m2(&v, &raw_); }
    explicit e5m2_t(double v) noexcept {
        float f = static_cast<float>(v);
        nk_f32_to_e5m2(&f, &raw_);
    }
    template <std::integral integral_type_>
    e5m2_t(integral_type_ v) noexcept {
        float f = static_cast<float>(v);
        nk_f32_to_e5m2(&f, &raw_);
    }
    operator float() const noexcept { return to_f32(); }
    float raw() const noexcept { return to_f32(); }
    static e5m2_t from_raw(raw_t r) noexcept {
        e5m2_t v;
        v.raw_ = r;
        return v;
    }
    static constexpr e5m2_t from_bits(uint_t bits) noexcept {
        e5m2_t v;
        v.raw_ = bits;
        return v;
    }
    constexpr uint_t to_bits() const noexcept { return raw_; }

    static constexpr e5m2_t finite_max() noexcept { return from_bits(0x7B); }    // 57344.0
    static constexpr e5m2_t finite_min() noexcept { return from_bits(0xFB); }    // -57344.0
    static constexpr e5m2_t positive_min() noexcept { return from_bits(0x04); }  // Smallest positive normal
    static constexpr e5m2_t subnormal_min() noexcept { return from_bits(0x01); } // Smallest positive subnormal
    static constexpr e5m2_t positive_infinity() noexcept { return from_bits(0x7C); }
    static constexpr e5m2_t negative_infinity() noexcept { return from_bits(0xFC); }
    static constexpr e5m2_t quiet_nan() noexcept { return from_bits(0x7E); }
    static constexpr e5m2_t signaling_nan() noexcept { return from_bits(0x7D); }

    // Mathematical constants
    static constexpr e5m2_t pi_k() noexcept { return from_bits(0x42); }        // 3.0 (closest to π)
    static constexpr e5m2_t two_pi_k() noexcept { return from_bits(0x46); }    // 6.0 (closest to 2π)
    static constexpr e5m2_t half_pi_k() noexcept { return from_bits(0x3E); }   // 1.5 (closest to π/2)
    static constexpr e5m2_t e_k() noexcept { return from_bits(0x41); }         // 2.5 (closest to e)
    static constexpr e5m2_t sqrt2_k() noexcept { return from_bits(0x3E); }     // 1.5 (closest to √2)
    static constexpr e5m2_t inv_sqrt2_k() noexcept { return from_bits(0x3A); } // 0.75 (closest to 1/√2)
    static constexpr e5m2_t ln2_k() noexcept { return from_bits(0x3A); }       // 0.75 (closest to ln(2))

    constexpr bool is_nan() const noexcept { return (raw_ & 0x7F) > 0x7C; }
    constexpr bool is_infinite() const noexcept { return (raw_ & 0x7F) == 0x7C; }
    constexpr bool is_finite() const noexcept { return (raw_ & 0x7C) != 0x7C; }
    constexpr bool is_normal() const noexcept {
        uint_t exp = (raw_ >> 2) & 0x1F;
        return exp != 0 && exp != 0x1F;
    }
    constexpr bool is_subnormal() const noexcept { return ((raw_ >> 2) & 0x1F) == 0 && (raw_ & 0x03) != 0; }
    constexpr bool is_sign_positive() const noexcept { return (raw_ & 0x80) == 0; }
    constexpr bool is_sign_negative() const noexcept { return (raw_ & 0x80) != 0; }

    inline e5m2_t operator+() const noexcept { return *this; }
    inline e5m2_t operator-() const noexcept { return from_bits(raw_ ^ 0x80); }
    inline e5m2_t operator+(e5m2_t o) const noexcept { return from_f32(to_f32() + o.to_f32()); }
    inline e5m2_t operator-(e5m2_t o) const noexcept { return from_f32(to_f32() - o.to_f32()); }
    inline e5m2_t operator*(e5m2_t o) const noexcept { return from_f32(to_f32() * o.to_f32()); }
    inline e5m2_t operator/(e5m2_t o) const noexcept { return from_f32(to_f32() / o.to_f32()); }

    inline e5m2_t &operator+=(e5m2_t o) noexcept { return *this = *this + o; }
    inline e5m2_t &operator-=(e5m2_t o) noexcept { return *this = *this - o; }
    inline e5m2_t &operator*=(e5m2_t o) noexcept { return *this = *this * o; }
    inline e5m2_t &operator/=(e5m2_t o) noexcept { return *this = *this / o; }

    inline bool operator==(e5m2_t o) const noexcept { return to_f32() == o.to_f32(); }
    inline bool operator!=(e5m2_t o) const noexcept { return to_f32() != o.to_f32(); }
    inline bool operator<(e5m2_t o) const noexcept { return to_f32() < o.to_f32(); }
    inline bool operator>(e5m2_t o) const noexcept { return to_f32() > o.to_f32(); }
    inline bool operator<=(e5m2_t o) const noexcept { return to_f32() <= o.to_f32(); }
    inline bool operator>=(e5m2_t o) const noexcept { return to_f32() >= o.to_f32(); }

    /** @brief Total ordering comparison (Rust-style). */
    inline int total_cmp(e5m2_t o) const noexcept {
        std::int8_t a = std::bit_cast<std::int8_t>(raw_);
        std::int8_t b = std::bit_cast<std::int8_t>(o.raw_);
        if (a < 0) a = std::int8_t(0x80) - a;
        if (b < 0) b = std::int8_t(0x80) - b;
        return (a > b) - (a < b);
    }

    constexpr e5m2_t abs() const noexcept { return from_bits(raw_ & 0x7F); }
    constexpr e5m2_t copysign(e5m2_t sign) const noexcept { return from_bits((raw_ & 0x7F) | (sign.raw_ & 0x80)); }
    inline e5m2_t signum() const noexcept {
        if (is_nan()) return *this;
        return is_sign_negative() ? e5m2_t {-1.0f} : e5m2_t {1.0f};
    }

    inline e5m2_t floor() const noexcept { return from_f32(std::floor(to_f32())); }
    inline e5m2_t ceil() const noexcept { return from_f32(std::ceil(to_f32())); }
    inline e5m2_t round() const noexcept { return from_f32(std::round(to_f32())); }
    inline e5m2_t trunc() const noexcept { return from_f32(std::trunc(to_f32())); }
    inline e5m2_t fract() const noexcept {
        float f = to_f32();
        return from_f32(f - std::trunc(f));
    }

    inline e5m2_t sqrt() const noexcept { return from_f32(std::sqrt(to_f32())); }
    inline e5m2_t cbrt() const noexcept { return from_f32(std::cbrt(to_f32())); }
    inline e5m2_t rsqrt() const noexcept { return from_f32(1.0f / std::sqrt(to_f32())); }
    inline e5m2_t recip() const noexcept { return from_f32(1.0f / to_f32()); }
    inline e5m2_t mul_add(e5m2_t a, e5m2_t b) const noexcept {
        return from_f32(std::fma(to_f32(), a.to_f32(), b.to_f32()));
    }
    inline e5m2_t powf(e5m2_t exp) const noexcept { return from_f32(std::pow(to_f32(), exp.to_f32())); }

    inline e5m2_t exp() const noexcept { return from_f32(std::exp(to_f32())); }
    inline e5m2_t exp2() const noexcept { return from_f32(std::exp2(to_f32())); }
    inline e5m2_t ln() const noexcept { return from_f32(std::log(to_f32())); }
    inline e5m2_t log2() const noexcept { return from_f32(std::log2(to_f32())); }
    inline e5m2_t log10() const noexcept { return from_f32(std::log10(to_f32())); }

    inline e5m2_t sin() const noexcept { return from_f32(std::sin(to_f32())); }
    inline e5m2_t cos() const noexcept { return from_f32(std::cos(to_f32())); }
    inline e5m2_t tan() const noexcept { return from_f32(std::tan(to_f32())); }
    inline e5m2_t tanh() const noexcept { return from_f32(std::tanh(to_f32())); }

    inline e5m2_t min(e5m2_t o) const noexcept { return from_f32(std::fmin(to_f32(), o.to_f32())); }
    inline e5m2_t max(e5m2_t o) const noexcept { return from_f32(std::fmax(to_f32(), o.to_f32())); }
    inline e5m2_t clamp(e5m2_t lo, e5m2_t hi) const noexcept { return max(lo).min(hi); }

    /** @brief Saturating addition: clamps to finite range on overflow. */
    inline e5m2_t saturating_add(e5m2_t o) const noexcept {
        float result = to_f32() + o.to_f32();
        if (result >= finite_max().to_f32()) return finite_max();
        if (result <= finite_min().to_f32()) return finite_min();
        return from_f32(result);
    }

    /** @brief Saturating subtraction: clamps to finite range on overflow. */
    inline e5m2_t saturating_sub(e5m2_t o) const noexcept {
        float result = to_f32() - o.to_f32();
        if (result >= finite_max().to_f32()) return finite_max();
        if (result <= finite_min().to_f32()) return finite_min();
        return from_f32(result);
    }
};

/**
 *  @brief Float6 E2M3FN: 1 sign + 2 exponent (bias=1) + 3 mantissa bits, with 2 bits of padding.
 *
 *  Range: [-7.5, +7.5], stored byte-aligned in 8 bits (0b00SEEMM, upper 2 bits unused/padding).
 *  Format: 6-bit payload + 2-bit padding for efficient byte-aligned SIMD operations.
 *  No Inf/NaN support (OCP Microscaling FN format).
 *  Optimized for ARM NEONFHM using FMLAL (widening FP16→FP32).
 *
 *  References:
 *  - https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
 *  - https://arxiv.org/abs/2401.14112 (FP6-LLM paper)
 */
struct e2m3_t {
    // Core type aliases
    using raw_t = nk_e2m3_t;
    using uint_t = nk_u8_t;

    using dot_result_t = f32_t;
    using reduce_add_result_t = f32_t;
    using sqeuclidean_result_t = f32_t;
    using scale_t = nk_f32_t;

    using dot_kernel_t = void (*)(raw_t const *, raw_t const *, nk_size_t, nk_f32_t *);
    using reduce_add_kernel_t = void (*)(raw_t const *, nk_size_t, nk_size_t, nk_f32_t *);
    using reduce_extremum_kernel_t = void (*)(raw_t const *, nk_size_t, nk_size_t, raw_t *, nk_size_t *);
    using scale_kernel_t = void (*)(raw_t const *, nk_size_t, scale_t const *, scale_t const *, raw_t *);
    using sum_kernel_t = void (*)(raw_t const *, raw_t const *, nk_size_t, raw_t *);
    using wsum_kernel_t = void (*)(raw_t const *, raw_t const *, nk_size_t, scale_t const *, scale_t const *, raw_t *);
    using fma_kernel_t = void (*)(raw_t const *, raw_t const *, raw_t const *, nk_size_t, scale_t const *,
                                  scale_t const *, raw_t *);
    using dots_packed_size_kernel_t = nk_size_t (*)(nk_size_t, nk_size_t);
    using dots_pack_kernel_t = void (*)(raw_t const *, nk_size_t, nk_size_t, nk_size_t, void *);
    using dots_symmetric_kernel_t = void (*)(raw_t const *, nk_size_t, nk_size_t, nk_size_t, nk_f32_t *, nk_size_t,
                                             nk_size_t, nk_size_t);
    using dots_packed_kernel_t = void (*)(raw_t const *, void const *, nk_f32_t *, nk_size_t, nk_size_t, nk_size_t,
                                          nk_size_t, nk_size_t);

    static constexpr nk_dtype_t dtype() noexcept { return nk_e2m3_k; }
    static constexpr char const *dtype_name() noexcept { return "e2m3"; }
    static constexpr unsigned bits_per_word() noexcept { return 8; }
    static constexpr unsigned bits_per_dimension() noexcept { return 8; }
    static constexpr unsigned bits_per_value() noexcept { return 8; }
    static constexpr unsigned mantissa_bits() noexcept { return 3; }
    static constexpr unsigned exponent_bits() noexcept { return 2; }
    static constexpr bool is_integer() noexcept { return false; }
    static constexpr bool is_signed() noexcept { return true; }
    static constexpr bool is_complex() noexcept { return false; }
    static constexpr bool is_exact() noexcept { return false; }
    static constexpr bool has_infinity() noexcept { return false; }
    static constexpr bool has_nan() noexcept { return false; }

    raw_t raw_;

    inline float to_f32() const noexcept {
        float r;
        nk_e2m3_to_f32(&raw_, &r);
        return r;
    }
    static inline e2m3_t from_f32(float v) noexcept {
        e2m3_t r;
        nk_f32_to_e2m3(&v, &r.raw_);
        return r;
    }

    inline e2m3_t() noexcept : raw_(0) {}
    inline e2m3_t(float v) noexcept { nk_f32_to_e2m3(&v, &raw_); }
    explicit e2m3_t(double v) noexcept {
        float f = static_cast<float>(v);
        nk_f32_to_e2m3(&f, &raw_);
    }
    template <std::integral integral_type_>
    e2m3_t(integral_type_ v) noexcept {
        float f = static_cast<float>(v);
        nk_f32_to_e2m3(&f, &raw_);
    }
    inline operator float() const noexcept { return to_f32(); }
    inline float raw() const noexcept { return to_f32(); }
    static constexpr e2m3_t from_raw(raw_t r) noexcept {
        e2m3_t v;
        v.raw_ = r;
        return v;
    }
    static constexpr e2m3_t from_bits(uint_t bits) noexcept {
        e2m3_t v;
        v.raw_ = bits;
        return v;
    }
    constexpr uint_t to_bits() const noexcept { return raw_; }

    // E2M3FN range: [-7.5, +7.5], no Inf/NaN
    static constexpr e2m3_t finite_max() noexcept { return from_bits(0x1F); }   // +7.5
    static constexpr e2m3_t finite_min() noexcept { return from_bits(0x3F); }   // -7.5
    static constexpr e2m3_t positive_min() noexcept { return from_bits(0x08); } // Smallest positive normal (2⁰ = 1.0)
    static constexpr e2m3_t subnormal_min() noexcept { return from_bits(0x01); } // Smallest positive subnormal

    // Mathematical constants
    static constexpr e2m3_t pi_k() noexcept { return from_bits(0x15); }        // 3.25 (closest to π)
    static constexpr e2m3_t two_pi_k() noexcept { return from_bits(0x1D); }    // 6.5 (closest to 2π)
    static constexpr e2m3_t half_pi_k() noexcept { return from_bits(0x0D); }   // 1.625 (closest to π/2)
    static constexpr e2m3_t e_k() noexcept { return from_bits(0x13); }         // 2.75 (closest to e)
    static constexpr e2m3_t sqrt2_k() noexcept { return from_bits(0x0B); }     // 1.375 (closest to √2)
    static constexpr e2m3_t inv_sqrt2_k() noexcept { return from_bits(0x03); } // 0.375 (closest to 1/√2)
    static constexpr e2m3_t ln2_k() noexcept { return from_bits(0x03); }       // 0.375 (closest to ln(2))

    constexpr bool is_nan() const noexcept { return false; }
    constexpr bool is_infinite() const noexcept { return false; }
    constexpr bool is_finite() const noexcept { return true; }
    constexpr bool is_normal() const noexcept {
        uint_t exp = (raw_ >> 3) & 0x03;
        return exp != 0;
    }
    constexpr bool is_subnormal() const noexcept { return ((raw_ >> 3) & 0x03) == 0 && (raw_ & 0x07) != 0; }
    constexpr bool is_sign_positive() const noexcept { return (raw_ & 0x20) == 0; }
    constexpr bool is_sign_negative() const noexcept { return (raw_ & 0x20) != 0; }

    inline e2m3_t operator+() const noexcept { return *this; }
    inline e2m3_t operator-() const noexcept { return from_bits(raw_ ^ 0x20); }
    inline e2m3_t operator+(e2m3_t o) const noexcept { return from_f32(to_f32() + o.to_f32()); }
    inline e2m3_t operator-(e2m3_t o) const noexcept { return from_f32(to_f32() - o.to_f32()); }
    inline e2m3_t operator*(e2m3_t o) const noexcept { return from_f32(to_f32() * o.to_f32()); }
    inline e2m3_t operator/(e2m3_t o) const noexcept { return from_f32(to_f32() / o.to_f32()); }

    inline e2m3_t &operator+=(e2m3_t o) noexcept { return *this = *this + o; }
    inline e2m3_t &operator-=(e2m3_t o) noexcept { return *this = *this - o; }
    inline e2m3_t &operator*=(e2m3_t o) noexcept { return *this = *this * o; }
    inline e2m3_t &operator/=(e2m3_t o) noexcept { return *this = *this / o; }

    inline bool operator==(e2m3_t o) const noexcept { return to_f32() == o.to_f32(); }
    inline bool operator!=(e2m3_t o) const noexcept { return to_f32() != o.to_f32(); }
    inline bool operator<(e2m3_t o) const noexcept { return to_f32() < o.to_f32(); }
    inline bool operator>(e2m3_t o) const noexcept { return to_f32() > o.to_f32(); }
    inline bool operator<=(e2m3_t o) const noexcept { return to_f32() <= o.to_f32(); }
    inline bool operator>=(e2m3_t o) const noexcept { return to_f32() >= o.to_f32(); }

    constexpr e2m3_t abs() const noexcept { return from_bits(raw_ & 0x1F); }
    constexpr e2m3_t copysign(e2m3_t sign) const noexcept { return from_bits((raw_ & 0x1F) | (sign.raw_ & 0x20)); }

    inline e2m3_t sqrt() const noexcept { return from_f32(std::sqrt(to_f32())); }
    inline e2m3_t min(e2m3_t o) const noexcept { return from_f32(std::fmin(to_f32(), o.to_f32())); }
    inline e2m3_t max(e2m3_t o) const noexcept { return from_f32(std::fmax(to_f32(), o.to_f32())); }
    inline e2m3_t clamp(e2m3_t lo, e2m3_t hi) const noexcept { return max(lo).min(hi); }

    inline e2m3_t saturating_add(e2m3_t o) const noexcept {
        float result = to_f32() + o.to_f32();
        if (result >= finite_max().to_f32()) return finite_max();
        if (result <= finite_min().to_f32()) return finite_min();
        return from_f32(result);
    }

    inline e2m3_t saturating_sub(e2m3_t o) const noexcept {
        float result = to_f32() - o.to_f32();
        if (result >= finite_max().to_f32()) return finite_max();
        if (result <= finite_min().to_f32()) return finite_min();
        return from_f32(result);
    }
};

/**
 *  @brief Float6 E3M2FN: 1 sign + 3 exponent (bias=3) + 2 mantissa bits, with 2 bits of padding.
 *
 *  Range: [-28, +28], stored byte-aligned in 8 bits (0b00SEEEMM, upper 2 bits unused/padding).
 *  Format: 6-bit payload + 2-bit padding for efficient byte-aligned SIMD operations.
 *  No Inf/NaN support (OCP Microscaling FN format).
 *  Optimized for ARM NEONFHM using FMLAL (widening FP16→FP32).
 *
 *  References:
 *  - https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
 *  - https://arxiv.org/abs/2401.14112 (FP6-LLM paper)
 */
struct e3m2_t {
    // Core type aliases
    using raw_t = nk_e3m2_t;
    using uint_t = nk_u8_t;

    using dot_result_t = f32_t;
    using reduce_add_result_t = f32_t;
    using sqeuclidean_result_t = f32_t;
    using scale_t = nk_f32_t;

    using dot_kernel_t = void (*)(raw_t const *, raw_t const *, nk_size_t, nk_f32_t *);
    using reduce_add_kernel_t = void (*)(raw_t const *, nk_size_t, nk_size_t, nk_f32_t *);
    using reduce_extremum_kernel_t = void (*)(raw_t const *, nk_size_t, nk_size_t, raw_t *, nk_size_t *);
    using scale_kernel_t = void (*)(raw_t const *, nk_size_t, scale_t const *, scale_t const *, raw_t *);
    using sum_kernel_t = void (*)(raw_t const *, raw_t const *, nk_size_t, raw_t *);
    using wsum_kernel_t = void (*)(raw_t const *, raw_t const *, nk_size_t, scale_t const *, scale_t const *, raw_t *);
    using fma_kernel_t = void (*)(raw_t const *, raw_t const *, raw_t const *, nk_size_t, scale_t const *,
                                  scale_t const *, raw_t *);
    using dots_packed_size_kernel_t = nk_size_t (*)(nk_size_t, nk_size_t);
    using dots_pack_kernel_t = void (*)(raw_t const *, nk_size_t, nk_size_t, nk_size_t, void *);
    using dots_symmetric_kernel_t = void (*)(raw_t const *, nk_size_t, nk_size_t, nk_size_t, nk_f32_t *, nk_size_t,
                                             nk_size_t, nk_size_t);
    using dots_packed_kernel_t = void (*)(raw_t const *, void const *, nk_f32_t *, nk_size_t, nk_size_t, nk_size_t,
                                          nk_size_t, nk_size_t);

    static constexpr nk_dtype_t dtype() noexcept { return nk_e3m2_k; }
    static constexpr char const *dtype_name() noexcept { return "e3m2"; }
    static constexpr unsigned bits_per_word() noexcept { return 8; }
    static constexpr unsigned bits_per_dimension() noexcept { return 8; }
    static constexpr unsigned bits_per_value() noexcept { return 8; }
    static constexpr unsigned mantissa_bits() noexcept { return 2; }
    static constexpr unsigned exponent_bits() noexcept { return 3; }
    static constexpr bool is_integer() noexcept { return false; }
    static constexpr bool is_signed() noexcept { return true; }
    static constexpr bool is_complex() noexcept { return false; }
    static constexpr bool is_exact() noexcept { return false; }
    static constexpr bool has_infinity() noexcept { return false; }
    static constexpr bool has_nan() noexcept { return false; }

    raw_t raw_;

    inline float to_f32() const noexcept {
        float r;
        nk_e3m2_to_f32(&raw_, &r);
        return r;
    }
    static inline e3m2_t from_f32(float v) noexcept {
        e3m2_t r;
        nk_f32_to_e3m2(&v, &r.raw_);
        return r;
    }

    inline e3m2_t() noexcept : raw_(0) {}
    inline e3m2_t(float v) noexcept { nk_f32_to_e3m2(&v, &raw_); }
    explicit e3m2_t(double v) noexcept {
        float f = static_cast<float>(v);
        nk_f32_to_e3m2(&f, &raw_);
    }
    template <std::integral integral_type_>
    e3m2_t(integral_type_ v) noexcept {
        float f = static_cast<float>(v);
        nk_f32_to_e3m2(&f, &raw_);
    }
    inline operator float() const noexcept { return to_f32(); }
    inline float raw() const noexcept { return to_f32(); }
    static constexpr e3m2_t from_raw(raw_t r) noexcept {
        e3m2_t v;
        v.raw_ = r;
        return v;
    }
    static constexpr e3m2_t from_bits(uint_t bits) noexcept {
        e3m2_t v;
        v.raw_ = bits;
        return v;
    }
    constexpr uint_t to_bits() const noexcept { return raw_; }

    // E3M2FN range: [-28, +28], no Inf/NaN
    static constexpr e3m2_t finite_max() noexcept { return from_bits(0x1F); }   // +28.0
    static constexpr e3m2_t finite_min() noexcept { return from_bits(0x3F); }   // -28.0
    static constexpr e3m2_t positive_min() noexcept { return from_bits(0x0C); } // Smallest positive normal (2⁰ = 1.0)
    static constexpr e3m2_t subnormal_min() noexcept { return from_bits(0x01); } // Smallest positive subnormal

    // Mathematical constants
    static constexpr e3m2_t pi_k() noexcept { return from_bits(0x12); }        // 3.0 (closest to π)
    static constexpr e3m2_t two_pi_k() noexcept { return from_bits(0x16); }    // 6.0 (closest to 2π)
    static constexpr e3m2_t half_pi_k() noexcept { return from_bits(0x0E); }   // 1.5 (closest to π/2)
    static constexpr e3m2_t e_k() noexcept { return from_bits(0x11); }         // 2.5 (closest to e)
    static constexpr e3m2_t sqrt2_k() noexcept { return from_bits(0x0E); }     // 1.5 (closest to √2)
    static constexpr e3m2_t inv_sqrt2_k() noexcept { return from_bits(0x0A); } // 0.75 (closest to 1/√2)
    static constexpr e3m2_t ln2_k() noexcept { return from_bits(0x0A); }       // 0.75 (closest to ln(2))

    constexpr bool is_nan() const noexcept { return false; }
    constexpr bool is_infinite() const noexcept { return false; }
    constexpr bool is_finite() const noexcept { return true; }
    constexpr bool is_normal() const noexcept {
        uint_t exp = (raw_ >> 2) & 0x07;
        return exp != 0;
    }
    constexpr bool is_subnormal() const noexcept { return ((raw_ >> 2) & 0x07) == 0 && (raw_ & 0x03) != 0; }
    constexpr bool is_sign_positive() const noexcept { return (raw_ & 0x20) == 0; }
    constexpr bool is_sign_negative() const noexcept { return (raw_ & 0x20) != 0; }

    inline e3m2_t operator+() const noexcept { return *this; }
    inline e3m2_t operator-() const noexcept { return from_bits(raw_ ^ 0x20); }
    inline e3m2_t operator+(e3m2_t o) const noexcept { return from_f32(to_f32() + o.to_f32()); }
    inline e3m2_t operator-(e3m2_t o) const noexcept { return from_f32(to_f32() - o.to_f32()); }
    inline e3m2_t operator*(e3m2_t o) const noexcept { return from_f32(to_f32() * o.to_f32()); }
    inline e3m2_t operator/(e3m2_t o) const noexcept { return from_f32(to_f32() / o.to_f32()); }

    inline e3m2_t &operator+=(e3m2_t o) noexcept { return *this = *this + o; }
    inline e3m2_t &operator-=(e3m2_t o) noexcept { return *this = *this - o; }
    inline e3m2_t &operator*=(e3m2_t o) noexcept { return *this = *this * o; }
    inline e3m2_t &operator/=(e3m2_t o) noexcept { return *this = *this / o; }

    inline bool operator==(e3m2_t o) const noexcept { return to_f32() == o.to_f32(); }
    inline bool operator!=(e3m2_t o) const noexcept { return to_f32() != o.to_f32(); }
    inline bool operator<(e3m2_t o) const noexcept { return to_f32() < o.to_f32(); }
    inline bool operator>(e3m2_t o) const noexcept { return to_f32() > o.to_f32(); }
    inline bool operator<=(e3m2_t o) const noexcept { return to_f32() <= o.to_f32(); }
    inline bool operator>=(e3m2_t o) const noexcept { return to_f32() >= o.to_f32(); }

    constexpr e3m2_t abs() const noexcept { return from_bits(raw_ & 0x1F); }
    constexpr e3m2_t copysign(e3m2_t sign) const noexcept { return from_bits((raw_ & 0x1F) | (sign.raw_ & 0x20)); }

    inline e3m2_t sqrt() const noexcept { return from_f32(std::sqrt(to_f32())); }
    inline e3m2_t min(e3m2_t o) const noexcept { return from_f32(std::fmin(to_f32(), o.to_f32())); }
    inline e3m2_t max(e3m2_t o) const noexcept { return from_f32(std::fmax(to_f32(), o.to_f32())); }
    inline e3m2_t clamp(e3m2_t lo, e3m2_t hi) const noexcept { return max(lo).min(hi); }

    inline e3m2_t saturating_add(e3m2_t o) const noexcept {
        float result = to_f32() + o.to_f32();
        if (result >= finite_max().to_f32()) return finite_max();
        if (result <= finite_min().to_f32()) return finite_min();
        return from_f32(result);
    }

    inline e3m2_t saturating_sub(e3m2_t o) const noexcept {
        float result = to_f32() - o.to_f32();
        if (result >= finite_max().to_f32()) return finite_max();
        if (result <= finite_min().to_f32()) return finite_min();
        return from_f32(result);
    }
};

/**
 *  @brief Hardware-friendly @b "double-double" arithmetic with ~106-bit mantissa.
 *
 *  Uses Knuth two-sum + FMA for error-free transformations. Provides ~106 bits of
 *  mantissa precision using two doubles, at roughly 11x the cost of native `double`
 *  arithmetic, being roughly 8x more efficient than Boost.Multiprecision types.
 *
 *  Speed comparison (relative to double):
 *
 *      Type                        Speed  Mantissa  Notes
 *      double                      1.0x   53-bit    Hardware
 *      long double                 1.5x   64-bit    x87 hardware
 *      f118_t                      11x    ~106-bit  Software, FMA-based
 *      __float128                  88x    113-bit   libquadmath
 *      boost::float128             91x    113-bit   Wrapper around __float128
 *      boost::cpp_bin_float_quad   200x   113-bit   Pure C++ (slowest!)
 *      boost::cpp_bin_float_50     237x   ~166-bit  50 decimal digits
 *
 *  Measured precision vs __float128 (1M random samples, full double-double input):
 *
 *      Operation   Max Rel Err   Precision   Notes
 *      +, -        ~2e-28        ~91 bits    Two-Sum algorithm
 *      *           2.5e-32       ~104 bits   FMA-based Two-Prod
 *      /           2.4e-32       ~105 bits   3-iteration Newton-Raphson
 *      sqrt        6.8e-32       ~103 bits   Newton-Raphson refinement
 *      cbrt        2.4e-31       ~101 bits   Newton-Raphson refinement
 *      exp         2.2e-31       ~101 bits   Argument reduction + Taylor
 *      log         2.3e-32       ~105 bits   Newton-Raphson refinement
 *      sin         6.8e-32       ~103 bits   |x|<10, quad-double π/2 + Taylor
 *      cos         7.8e-32       ~103 bits   |x|<10, quad-double π/2 + Taylor
 *      tan         9.4e-32       ~103 bits   sin/cos ratio
 *      sinh        1.4e-31       ~102 bits   Taylor for |x|<1, optimized exp
 *      cosh        1.5e-31       ~102 bits   Via exp()
 *      tanh        1.0e-31       ~102 bits   sinh/cosh ratio
 *      pow         1.0e-30       ~99 bits    Via exp(y·log(x))
 */
struct f118_t {
    double high_, low_;

    static constexpr nk_dtype_t dtype() noexcept { return nk_dtype_unknown_k; }
    static constexpr char const *dtype_name() noexcept { return "f118"; }
    static constexpr unsigned bits_per_word() noexcept { return 128; }
    static constexpr unsigned bits_per_dimension() noexcept { return 128; }
    static constexpr unsigned bits_per_value() noexcept { return 128; }
    static constexpr unsigned mantissa_bits() noexcept { return 103; } // ~103 bits mantissa precision
    static constexpr bool is_integer() noexcept { return false; }
    static constexpr bool is_signed() noexcept { return true; }
    static constexpr bool is_complex() noexcept { return false; }
    static constexpr bool is_exact() noexcept { return false; }
    static constexpr bool has_infinity() noexcept { return true; }
    static constexpr bool has_nan() noexcept { return true; }

    static constexpr f118_t finite_max() noexcept { return f118_t(1.7976931348623157e+308, 9.9792015476736e+291); }
    static constexpr f118_t finite_min() noexcept { return f118_t(-1.7976931348623157e+308, -9.9792015476736e+291); }
    static constexpr f118_t positive_min() noexcept { return f118_t(2.2250738585072014e-308); } // Smallest positive
    static constexpr f118_t zero() noexcept { return f118_t(); }
    static constexpr f118_t one() noexcept { return f118_t(1.0); }

    /** @brief Default constructor, initializes to zero. */
    constexpr f118_t() noexcept : high_(0), low_(0) {}
    /** @brief Construct from high and low double components. */
    constexpr f118_t(double h, double l) noexcept : high_(h), low_(l) {}
    /** @brief Construct from single double. */
    constexpr f118_t(double v) noexcept : high_(v), low_(0) {}
    /** @brief Construct from int (resolves ambiguity with double,double constructor). */
    constexpr f118_t(int v) noexcept : high_(v), low_(0) {}
    /** @brief Construct from unsigned int. */
    constexpr f118_t(unsigned int v) noexcept : high_(v), low_(0) {}
    /** @brief Construct from int64_t, capturing precision beyond double's 53-bit mantissa. */
    constexpr f118_t(std::int64_t v) noexcept : high_(static_cast<double>(v)), low_(0) {
        if (v != 0 && (v > (1LL << 53) || v < -(1LL << 53)))
            low_ = static_cast<double>(v - static_cast<std::int64_t>(high_));
    }
    /** @brief Construct from uint64_t, capturing precision beyond double's 53-bit mantissa. */
    constexpr f118_t(std::uint64_t v) noexcept : high_(static_cast<double>(v)), low_(0) {
        if (v > (1ULL << 53)) low_ = static_cast<double>(v - static_cast<std::uint64_t>(high_));
    }

#ifdef __SIZEOF_FLOAT128__
    /** @brief Construct from __float128, splitting into high_/low_ components. */
    constexpr explicit f118_t(__float128 v) noexcept {
        high_ = static_cast<double>(v);
        low_ = static_cast<double>(v - __float128(high_));
    }
#endif

    /** @brief Addition with ~91 bits precision (max rel err: 2.3e-28 vs __float128). */
    constexpr f118_t operator+(f118_t const &o) const noexcept {
        f118_t s = two_sum_(high_, o.high_);
        s.low_ += low_ + o.low_;
        return quick_two_sum_(s.high_, s.low_);
    }

    /** @brief In-place addition. */
    constexpr f118_t &operator+=(f118_t const &o) noexcept { return *this = *this + o; }
    /** @brief In-place subtraction. */
    constexpr f118_t &operator-=(f118_t const &o) noexcept { return *this = *this - o; }
    /** @brief In-place multiplication. */
    constexpr f118_t &operator*=(f118_t const &o) noexcept { return *this = *this * o; }
    /** @brief In-place division. */
    constexpr f118_t &operator/=(f118_t const &o) noexcept { return *this = *this / o; }

    /** @brief Subtraction with ~93 bits precision (max rel err: 7.4e-29 vs __float128). */
    constexpr f118_t operator-(f118_t const &o) const noexcept {
        f118_t s = two_sum_(high_, -o.high_);
        s.low_ += low_ - o.low_;
        return quick_two_sum_(s.high_, s.low_);
    }

    /** @brief Multiplication with ~104 bits precision (max rel err: 2.5e-32 vs __float128, ~105 bits vs Boost). */
    constexpr f118_t operator*(f118_t const &o) const noexcept {
        double p1 = high_ * o.high_;
        double p2 = std::fma(high_, o.high_, -p1);
        p2 += high_ * o.low_;
        p2 += low_ * o.high_;
        p2 += low_ * o.low_;
        return quick_two_sum_(p1, p2);
    }

    /** @brief Division with ~105 bits precision (max rel err: 2.4e-32 vs __float128, ~106 bits vs Boost). */
    constexpr f118_t operator/(f118_t const &o) const noexcept {
        double q1 = high_ / o.high_;
        f118_t r = *this - o * f118_t(q1);

        double q2 = r.high_ / o.high_;
        r = r - o * f118_t(q2);

        double q3 = r.high_ / o.high_;
        return two_sum_(q1, q2) + f118_t(q3);
    }

    /** @brief Saturating addition - clamps to max finite value instead of infinity. */
    constexpr f118_t saturating_add(f118_t o) const noexcept {
        f118_t result = *this + o;
        if (result.is_infinite()) return result.high_ > 0 ? finite_max() : finite_min();
        return result;
    }

    /** @brief Saturating subtraction - clamps to max finite value instead of infinity. */
    constexpr f118_t saturating_sub(f118_t o) const noexcept {
        f118_t result = *this - o;
        if (result.is_infinite()) return result.high_ > 0 ? finite_max() : finite_min();
        return result;
    }

    /** @brief Exact equality (both high_ and low_ must match). */
    constexpr bool operator==(f118_t const &o) const noexcept { return high_ == o.high_ && low_ == o.low_; }
    /** @brief Exact inequality. */
    constexpr bool operator!=(f118_t const &o) const noexcept { return !(*this == o); }
    /** @brief Less-than comparison (lexicographic on high_, then low_). */
    constexpr bool operator<(f118_t const &o) const noexcept {
        return high_ < o.high_ || (high_ == o.high_ && low_ < o.low_);
    }
    /** @brief Greater-than comparison. */
    constexpr bool operator>(f118_t const &o) const noexcept { return o < *this; }
    /** @brief Less-than-or-equal comparison. */
    constexpr bool operator<=(f118_t const &o) const noexcept { return !(o < *this); }
    /** @brief Greater-than-or-equal comparison. */
    constexpr bool operator>=(f118_t const &o) const noexcept { return !(*this < o); }

    /** @brief Convert to double (loses ~53 bits of precision). */
    constexpr explicit operator double() const noexcept { return high_ + low_; }

    /** @brief Convert to any numeric type. */
    template <typename target_type_>
    constexpr target_type_ to() const noexcept {
        constexpr bool is_raw_integral = std::is_integral_v<target_type_>;
        constexpr bool is_wrapper_integral = std::is_class_v<target_type_> && target_type_::is_integer();
        // Raw integral: sum both components for precision
        if constexpr (is_raw_integral) { return static_cast<target_type_>(high_) + static_cast<target_type_>(low_); }
        // Wrapper integral: cast through raw_t to avoid ambiguous constructors
        else if constexpr (is_wrapper_integral) {
            using raw_t = typename target_type_::raw_t;
            return target_type_(static_cast<raw_t>(high_) + static_cast<raw_t>(low_));
        }
        else { return target_type_(static_cast<double>(*this)); }
    }

    /** @brief Square root with ~103 bits precision (max rel err: 6.8e-32 vs __float128, ~103 bits vs Boost). */
    constexpr f118_t sqrt() const noexcept {
        if (high_ <= 0) return f118_t(std::sqrt(high_));
        double inv_sqrt_approx = 1.0 / std::sqrt(high_);
        double sqrt_approx = high_ * inv_sqrt_approx;
        f118_t sqrt_approx_dd(sqrt_approx);
        f118_t residual = *this - sqrt_approx_dd * sqrt_approx_dd;
        return sqrt_approx_dd + residual * f118_t(inv_sqrt_approx * 0.5);
    }

    /** @brief Reciprocal square root (1/sqrt). */
    constexpr f118_t rsqrt() const noexcept { return f118_t(1.0) / sqrt(); }

    /** @brief Exponential with ~101 bits precision (max rel err: 2.2e-31 vs __float128, ~102 bits vs Boost). */
    constexpr f118_t exp() const noexcept {
        // High-precision ln(2)
        constexpr double ln2_high = 0.6931471805599453;
        constexpr double ln2_low = 2.3190468138462996e-17;
        f118_t ln2(ln2_high, ln2_low);

        // Argument reduction: x = k·ln(2) + r, |r| < ln(2)/2 (where k = exponent_scale, r = reduced_arg)
        double exponent_scale = std::round(high_ / ln2_high);
        f118_t reduced_arg = *this - ln2 * f118_t(exponent_scale);

        // Taylor series: eʳ = 1 + r + r²/2! + … (where r = reduced_arg)
        f118_t series_sum(1.0), current_term(1.0);
        for (unsigned int term_index = 1; term_index <= 25; ++term_index) {
            current_term = current_term * reduced_arg / f118_t(double(term_index));
            series_sum = series_sum + current_term;
            if (std::abs(current_term.high_) < 1e-32 * std::abs(series_sum.high_)) break;
        }

        // Reconstruct: eˣ = eʳ · 2ᵏ (where r = reduced_arg, k = exponent_scale)
        return series_sum * f118_t(std::ldexp(1.0, int(exponent_scale)));
    }

    /** @brief Natural logarithm with ~105 bits precision (max rel err: 2.2e-32 vs __float128, ~105 bits vs Boost). */
    constexpr f118_t log() const noexcept {
        if (high_ <= 0) return f118_t(std::log(high_)); // NaN or -inf

        // High-precision ln(2)
        constexpr double ln2_high = 0.6931471805599453;
        constexpr double ln2_low = 2.3190468138462996e-17;
        f118_t ln2(ln2_high, ln2_low);

        // Extract exponent: x = mantissa · 2ᵏ, mantissa ∈ [0.5, 1)
        int exponent = 0;
        double mantissa_high = std::frexp(high_, &exponent);
        // Properly scale low_ component
        double exponent_scale = std::ldexp(1.0, -exponent);
        f118_t mantissa_dd = two_sum_(mantissa_high, low_ * exponent_scale);

        // For better convergence, adjust mantissa to be near 1: if mantissa < √0.5, use mantissa×2 and exponent−1
        if (mantissa_high < 0.7071067811865476) {
            mantissa_dd = mantissa_dd + mantissa_dd; // mantissa_dd *= 2
            exponent -= 1;
        }

        // Newton-Raphson: given initial log_estimate = log(mantissa_dd.high_), refine
        double log_estimate = std::log(mantissa_dd.high_);
        f118_t log_result(log_estimate);

        // Two Newton iterations for full precision
        for (unsigned iter = 0; iter < 2; ++iter) {
            f118_t exp_of_log = log_result.exp();
            log_result = log_result + (mantissa_dd - exp_of_log) / exp_of_log;
        }

        return log_result + ln2 * f118_t(double(exponent));
    }

    /**
     *  @brief Sine with ~103 bits precision (max rel err: 6.8e-32 vs __float128) for |x| < 10.
     *  @note Uses quad-double π/2 with staged error-free reduction. Precision degrades for |x| > 1000.
     */
    constexpr f118_t sin() const noexcept {
        if (is_nan() || is_infinite()) return f118_t(std::sin(high_));

        f118_t reduced_angle;
        int quadrant = 0;
        reduce_trig_arg_(reduced_angle, quadrant);

        // Use quadrant to select sin or cos, with appropriate sign
        switch (quadrant) {
        case 0: return sin_taylor_(reduced_angle);
        case 1: return cos_taylor_(reduced_angle);
        case 2: return -sin_taylor_(reduced_angle);
        case 3: return -cos_taylor_(reduced_angle);
        default: return sin_taylor_(reduced_angle);
        }
    }

    /**
     *  @brief Cosine with ~103 bits precision (max rel err: 7.8e-32 vs __float128) for |x| < 10.
     *  @note Uses quad-double π/2 with staged error-free reduction. Precision degrades for |x| > 1000.
     */
    constexpr f118_t cos() const noexcept {
        if (is_nan() || is_infinite()) return f118_t(std::cos(high_));

        f118_t reduced_angle;
        int quadrant = 0;
        reduce_trig_arg_(reduced_angle, quadrant);

        // Use quadrant to select sin or cos, with appropriate sign
        switch (quadrant) {
        case 0: return cos_taylor_(reduced_angle);
        case 1: return -sin_taylor_(reduced_angle);
        case 2: return -cos_taylor_(reduced_angle);
        case 3: return sin_taylor_(reduced_angle);
        default: return cos_taylor_(reduced_angle);
        }
    }

    /** @brief Absolute value. */
    constexpr f118_t abs() const noexcept { return high_ >= 0 ? *this : f118_t(-high_, -low_); }

    /** @brief Unary minus. */
    constexpr f118_t operator-() const noexcept { return f118_t(-high_, -low_); }

    /** @brief Returns true if either component is NaN. */
    constexpr bool is_nan() const noexcept { return high_ != high_ || low_ != low_; }

    /** @brief Returns true if the value is positive or negative infinity. */
    constexpr bool is_infinite() const noexcept {
        return high_ == std::numeric_limits<double>::infinity() || high_ == -std::numeric_limits<double>::infinity();
    }

    /** @brief Returns true if the value is neither infinite nor NaN. */
    constexpr bool is_finite() const noexcept { return std::isfinite(high_); }

    /** @brief Returns true if the sign bit is positive (includes +0). */
    constexpr bool is_sign_positive() const noexcept { return high_ > 0.0 || (high_ == 0.0 && !std::signbit(high_)); }

    /** @brief Returns true if the sign bit is negative (includes -0). */
    constexpr bool is_sign_negative() const noexcept { return high_ < 0.0 || (high_ == 0.0 && std::signbit(high_)); }

    /** @brief Largest integer less than or equal to self. */
    constexpr f118_t floor() const noexcept {
        double floor_high = std::floor(high_);
        if (floor_high != high_) return f118_t(floor_high);
        return quick_two_sum_(floor_high, std::floor(low_));
    }

    /** @brief Smallest integer greater than or equal to self. */
    constexpr f118_t ceil() const noexcept {
        double ceil_high = std::ceil(high_);
        if (ceil_high != high_) return f118_t(ceil_high);
        return quick_two_sum_(ceil_high, std::ceil(low_));
    }

    /** @brief Nearest integer, rounding half away from zero. */
    constexpr f118_t round() const noexcept {
        double round_high = std::round(high_);
        if (round_high != high_) return f118_t(round_high);
        return quick_two_sum_(round_high, std::round(low_));
    }

    /** @brief Integer part (truncate toward zero). */
    constexpr f118_t trunc() const noexcept {
        double trunc_high = std::trunc(high_);
        if (trunc_high != high_) return f118_t(trunc_high);
        return quick_two_sum_(trunc_high, std::trunc(low_));
    }

    /** @brief Fractional part (self - trunc(self)). */
    constexpr f118_t fract() const noexcept { return *this - trunc(); }

    /** @brief Returns the minimum of self and other. */
    constexpr f118_t min(f118_t o) const noexcept { return *this < o ? *this : o; }

    /** @brief Returns the maximum of self and other. */
    constexpr f118_t max(f118_t o) const noexcept { return *this > o ? *this : o; }

    /** @brief Clamps self to the range [lower, upper]. */
    constexpr f118_t clamp(f118_t lower, f118_t upper) const noexcept { return max(lower).min(upper); }

    /** @brief Total ordering: -NaN < -Inf < ... < -0 < +0 < ... < +Inf < +NaN. Returns -1, 0, or 1. */
    constexpr int total_cmp(f118_t o) const noexcept {
        // Handle NaN cases first
        bool this_nan = is_nan(), o_nan = o.is_nan();
        if (this_nan && o_nan) return 0;
        if (this_nan) return is_sign_negative() ? -1 : 1;
        if (o_nan) return o.is_sign_negative() ? 1 : -1;
        // Normal comparison
        if (*this < o) return -1;
        if (*this > o) return 1;
        // Handle -0 vs +0
        if (high_ == 0.0 && o.high_ == 0.0) {
            bool this_neg = std::signbit(high_), o_neg = std::signbit(o.high_);
            if (this_neg && !o_neg) return -1;
            if (!this_neg && o_neg) return 1;
        }
        return 0;
    }

    /** @brief Returns -1, 0, or 1 based on sign. */
    constexpr f118_t signum() const noexcept {
        if (is_nan()) return *this;
        if (high_ > 0.0) return f118_t(1.0);
        if (high_ < 0.0) return f118_t(-1.0);
        return f118_t(0.0);
    }

    /** @brief Returns value with magnitude of self and sign of `sign`. */
    constexpr f118_t copysign(f118_t sign) const noexcept {
        bool this_neg = is_sign_negative();
        bool sign_neg = sign.is_sign_negative();
        if (this_neg == sign_neg) return *this;
        return -(*this);
    }

    /** @brief Returns 1 / self. */
    constexpr f118_t recip() const noexcept { return f118_t(1.0) / *this; }

    /** @brief xʸ with ~99 bits precision (max rel err: 1.0e-30 vs __float128). */
    constexpr f118_t powf(f118_t y) const noexcept {
        if (high_ == 0.0) return f118_t(0.0);
        if (y.high_ == 0.0) return f118_t(1.0);
        return (y * log()).exp();
    }

    /** @brief Integer power via binary exponentiation. */
    constexpr f118_t powi(int exponent) const noexcept {
        if (exponent == 0) return f118_t(1.0);
        f118_t result(1.0), base = *this;
        bool is_negative = exponent < 0;
        if (is_negative) exponent = -exponent;
        while (exponent > 0) {
            if (exponent & 1) result = result * base;
            base = base * base;
            exponent >>= 1;
        }
        return is_negative ? result.recip() : result;
    }

    /** @brief Cube root with ~101 bits precision (max rel err: 2.4e-31 vs __float128). */
    constexpr f118_t cbrt() const noexcept {
        if (high_ == 0.0) return f118_t(0.0);
        double cbrt_approx = std::cbrt(high_);
        // One Newton-Raphson iteration: x′ = x − (x³ − a) / (3x²) = (2x + a/x²) / 3
        f118_t cbrt_approx_dd(cbrt_approx);
        f118_t cbrt_squared = cbrt_approx_dd * cbrt_approx_dd;
        return (cbrt_approx_dd * f118_t(2.0) + *this / cbrt_squared) / f118_t(3.0);
    }

    /** @brief 2ˣ. */
    constexpr f118_t exp2() const noexcept {
        constexpr double ln2_high = 0.6931471805599453;
        constexpr double ln2_low = 2.3190468138462996e-17;
        return (*this * f118_t(ln2_high, ln2_low)).exp();
    }

    /** @brief Base-2 logarithm. */
    constexpr f118_t log2() const noexcept {
        constexpr double log2e_high = 1.4426950408889634;
        constexpr double log2e_low = 2.0355273740931033e-17;
        return log() * f118_t(log2e_high, log2e_low);
    }

    /** @brief Base-10 logarithm. */
    constexpr f118_t log10() const noexcept {
        constexpr double log10e_high = 0.4342944819032518;
        constexpr double log10e_low = 1.098319650216765e-17;
        return log() * f118_t(log10e_high, log10e_low);
    }

    /** @brief eˣ − 1, accurate for small x. */
    constexpr f118_t exp_m1() const noexcept {
        // For small x, use Taylor series directly for accuracy
        if (std::abs(high_) < 0.5) {
            f118_t series_sum(0.0), current_term(1.0);
            for (unsigned int term_index = 1; term_index <= 25; ++term_index) {
                current_term = current_term * *this / f118_t(double(term_index));
                series_sum = series_sum + current_term;
                if (std::abs(current_term.high_) < 1e-32 * std::abs(series_sum.high_)) break;
            }
            return series_sum;
        }
        return exp() - f118_t(1.0);
    }

    /** @brief ln(1 + x), accurate for small x. */
    constexpr f118_t ln_1p() const noexcept {
        // For small x, use series: ln(1+x) = x − x²/2 + x³/3 − …
        if (std::abs(high_) < 0.5) {
            f118_t x_squared = *this * *this;
            f118_t series_sum = *this, current_term = *this;
            for (unsigned int term_index = 2; term_index <= 30; ++term_index) {
                current_term = current_term * (*this) * f118_t(term_index & 1 ? 1.0 : -1.0);
                series_sum = series_sum + current_term / f118_t(double(term_index));
                if (std::abs(current_term.high_) < 1e-32 * std::abs(series_sum.high_)) break;
            }
            return series_sum;
        }
        return (*this + f118_t(1.0)).log();
    }

    /** @brief Tangent with ~103 bits precision (max rel err: 9.4e-32 vs __float128). */
    constexpr f118_t tan() const noexcept { return sin() / cos(); }

    /** @brief Arcsine (inverse sine). */
    constexpr f118_t asin() const noexcept {
        // asin(x) = atan(x / sqrt(1 − x²))
        if (std::abs(high_) >= 1.0) return f118_t(std::asin(high_));
        f118_t x_squared = *this * *this;
        return (*this / (f118_t(1.0) - x_squared).sqrt()).atan();
    }

    /** @brief Arccosine (inverse cosine). */
    constexpr f118_t acos() const noexcept {
        // acos(x) = pi/2 - asin(x)
        constexpr double half_pi_high = 1.5707963267948966;
        constexpr double half_pi_low = 6.123233995736766e-17;
        return f118_t(half_pi_high, half_pi_low) - asin();
    }

    /** @brief Arctangent (inverse tangent). */
    constexpr f118_t atan() const noexcept {
        constexpr double half_pi_high = 1.5707963267948966;
        constexpr double half_pi_low = 6.123233995736766e-17;
        f118_t half_pi(half_pi_high, half_pi_low);

        constexpr double quarter_pi_high = 0.7853981633974483;
        constexpr double quarter_pi_low = 3.061616997868383e-17;
        f118_t quarter_pi(quarter_pi_high, quarter_pi_low);

        // Reduce to |x| <= 1 using atan(x) = sign(x)*π/2 - atan(1/x) for |x| > 1
        if (std::abs(high_) > 1.0) {
            f118_t recip_atan = recip().atan();
            return high_ > 0 ? half_pi - recip_atan : -half_pi - recip_atan;
        }

        // For x near ±1, apply argument reduction to accelerate Taylor convergence.
        // The Taylor series for atan converges slowly when |x| ≈ 1 (needs ~30 terms).
        // Using the identity atan(x) = π/4 + atan((x-1)/(x+1)) transforms the argument
        // to be much smaller: at x=1, the reduced argument becomes 0 (instant convergence).
        //
        // Threshold analysis: For |x-1| < 0.5, the reduced argument satisfies:
        //   |(x-1)/(x+1)| < 0.25, which converges in ~5 terms vs ~30 for the direct series.
        // We use a slightly conservative threshold of 0.4 to ensure the reduction is beneficial
        // while avoiding unnecessary overhead for x far from 1.
        constexpr double reduction_threshold_near_one = 0.4; // Covers x ∈ [0.6, 1.4]

        if (std::abs(high_ - 1.0) < reduction_threshold_near_one) {
            // atan(x) = π/4 + atan((x-1)/(x+1))
            // Reduces argument from |x| ≈ 1 to |(x-1)/(x+1)| < 0.25
            f118_t numerator = *this - f118_t(1.0);
            f118_t denominator = *this + f118_t(1.0);
            f118_t reduced_arg = numerator / denominator;
            return quarter_pi + reduced_arg.atan();
        }

        if (std::abs(high_ + 1.0) < reduction_threshold_near_one) {
            // atan(x) = -π/4 + atan((x+1)/(1-x))
            // Symmetric case for x near -1
            f118_t numerator = *this + f118_t(1.0);
            f118_t denominator = f118_t(1.0) - *this;
            f118_t reduced_arg = numerator / denominator;
            return -quarter_pi + reduced_arg.atan();
        }

        // Taylor: atan(x) = x − x³/3 + x⁵/5 − x⁷/7 + …
        f118_t x_squared = *this * *this;
        f118_t series_sum = *this, current_term = *this;
        for (unsigned int term_index = 1; term_index <= 30; ++term_index) {
            current_term = current_term * x_squared * f118_t(-1.0);
            series_sum = series_sum + current_term / f118_t(2.0 * term_index + 1);
            if (std::abs(current_term.high_) < 1e-32 * std::abs(series_sum.high_)) break;
        }
        return series_sum;
    }

    /** @brief Four-quadrant arctangent: atan2(y, x) where this = y. */
    constexpr f118_t atan2(f118_t x) const noexcept {
        constexpr double pi_high = 3.141592653589793;
        constexpr double pi_low = 1.2246467991473532e-16;
        f118_t pi(pi_high, pi_low);

        if (x.high_ > 0.0) return (*this / x).atan();
        if (x.high_ < 0.0) {
            if (high_ >= 0.0) return (*this / x).atan() + pi;
            return (*this / x).atan() - pi;
        }
        // x == 0
        if (high_ > 0.0) return pi / f118_t(2.0);
        if (high_ < 0.0) return -pi / f118_t(2.0);
        return f118_t(0.0); // Both zero
    }

    /** @brief Computes both sin(x) and cos(x), returning them in an array. */
    constexpr void sin_cos(f118_t &out_sin, f118_t &out_cos) const noexcept {
        out_sin = sin();
        out_cos = cos();
    }

    /** @brief Hyperbolic sine with ~102 bits precision (max rel err: ~1.4e-31 vs __float128). */
    constexpr f118_t sinh() const noexcept {
        // Use Taylor series for |x| < 1 to avoid catastrophic cancellation
        if (std::abs(high_) < 1.0) {
            // Taylor: sinh(x) = x + x³/3! + x⁵/5! + x⁷/7! + …
            f118_t x_squared = *this * *this;
            f118_t series_sum = *this, current_term = *this;
            for (unsigned int term_index = 1; term_index <= 20; ++term_index) {
                current_term = current_term * x_squared / f118_t(double((2 * term_index) * (2 * term_index + 1)));
                series_sum = series_sum + current_term;
                if (std::abs(current_term.high_) < 1e-33 * std::abs(series_sum.high_)) break;
            }
            return series_sum;
        }
        // For |x| >= 1: use sinh(x) = sign(x) · exp(|x|) · (1 - exp(-2|x|)) / 2
        // This avoids subtracting two similar-magnitude values
        f118_t abs_x = abs();
        f118_t exp_abs = abs_x.exp();
        f118_t exp_neg2 = (abs_x * f118_t(-2.0)).exp();
        f118_t result = exp_abs * (f118_t(1.0) - exp_neg2) / f118_t(2.0);
        return high_ < 0 ? -result : result;
    }

    /** @brief Hyperbolic cosine with ~102 bits precision (max rel err: 1.5e-31 vs __float128). */
    constexpr f118_t cosh() const noexcept {
        f118_t exp_x = exp();
        return (exp_x + exp_x.recip()) / f118_t(2.0);
    }

    /** @brief Hyperbolic tangent with ~102 bits precision (max rel err: ~1e-31 vs __float128). */
    constexpr f118_t tanh() const noexcept {
        if (std::abs(high_) > 20.0) return high_ > 0 ? f118_t(1.0) : f118_t(-1.0);
        // Use sinh/cosh which are optimized with Taylor series for small args
        return sinh() / cosh();
    }

    /** @brief Inverse hyperbolic sine: ln(x + √(x² + 1)). */
    constexpr f118_t asinh() const noexcept { return (*this + (*this * *this + f118_t(1.0)).sqrt()).log(); }

    /** @brief Inverse hyperbolic cosine: ln(x + √(x² − 1)). */
    constexpr f118_t acosh() const noexcept {
        if (high_ < 1.0) return f118_t(std::numeric_limits<double>::quiet_NaN());
        return (*this + (*this * *this - f118_t(1.0)).sqrt()).log();
    }

    /** @brief Inverse hyperbolic tangent: ½ · ln((1+x)/(1−x)). */
    constexpr f118_t atanh() const noexcept {
        if (std::abs(high_) >= 1.0) return f118_t(std::atanh(high_));
        return ((f118_t(1.0) + *this) / (f118_t(1.0) - *this)).log() / f118_t(2.0);
    }

    /** @brief √(x² + y²) without overflow. */
    constexpr f118_t hypot(f118_t y) const noexcept {
        f118_t abs_x = abs(), abs_y = y.abs();
        if (abs_x < abs_y) std::swap(abs_x, abs_y);
        if (abs_x.high_ == 0.0) return f118_t(0.0);
        f118_t ratio = abs_y / abs_x;
        return abs_x * (f118_t(1.0) + ratio * ratio).sqrt();
    }

    /** @brief Fused multiply-add: self · a + b. */
    constexpr f118_t mul_add(f118_t a, f118_t b) const noexcept { return *this * a + b; }

    /** @brief Convert degrees to radians. */
    constexpr f118_t to_radians() const noexcept {
        constexpr double deg_to_rad_high = 0.017453292519943295;
        constexpr double deg_to_rad_low = 2.9486522708701687e-19;
        return *this * f118_t(deg_to_rad_high, deg_to_rad_low);
    }

    /** @brief Convert radians to degrees. */
    constexpr f118_t to_degrees() const noexcept {
        constexpr double rad_to_deg_high = 57.29577951308232;
        constexpr double rad_to_deg_low = -1.9878495670576283e-15;
        return *this * f118_t(rad_to_deg_high, rad_to_deg_low);
    }

  private:
    /**
     *  @brief Cody-Waite argument reduction for sin/cos using quad-double π/2.
     *
     *  Reduces angle to [-π/4, π/4] range and determines quadrant for
     *  proper sign and function selection (sin vs cos).
     *
     *  @param[out] reduced_angle Reduced angle in [-π/4, π/4]
     *  @param[out] quadrant Quadrant index (0-3) for sign/function selection
     *  @note Precision: ~106 bits for |x| < 2⁵² using staged error-free subtraction
     */
    constexpr void reduce_trig_arg_(f118_t &reduced_angle, int &quadrant) const noexcept {
        // Quad-double π/2 for ~212 bits of precision (more than we need)
        // π/2 = 1.5707963267948966192313216916397514420985846996875529...
        // Split into 4 parts to ensure exact representation of each chunk
        constexpr double half_pi_0 = 1.5707963267948966;      // ~53 bits
        constexpr double half_pi_1 = 6.123233995736766e-17;   // next ~53 bits
        constexpr double half_pi_2 = -1.4973849048591698e-33; // next ~53 bits
        constexpr double half_pi_3 = -1.0795385423361699e-49; // next ~53 bits

        // For small arguments, no reduction needed
        if (std::abs(high_) < half_pi_0 * 0.5) {
            reduced_angle = *this;
            quadrant = 0;
            return;
        }

        // Compute number of quarter-turns: k = round(x / (π/2))
        double quarter_turns = std::round(high_ / half_pi_0);

        // For very large arguments, pre-reduce with fmod to avoid precision loss
        if (std::abs(quarter_turns) > 4503599627370496.0) { // 2⁵²
            constexpr double two_pi = 6.283185307179586;
            double coarse_reduced = std::fmod(high_, two_pi);
            if (coarse_reduced < 0) coarse_reduced += two_pi;
            f118_t(coarse_reduced, 0).reduce_trig_arg_(reduced_angle, quadrant);
            return;
        }

        quadrant = static_cast<int>(quarter_turns) & 3; // k mod 4

        // Staged Cody-Waite reduction: subtract each π/2 component separately
        // This preserves maximum precision by using error-free transformations
        // r = x - k·π₀ - k·π₁ - k·π₂ - k·π₃

        // Stage 1: Subtract k·π₀ using two_prod for exact k·π₀
        f118_t prod0 = two_prod_(half_pi_0, quarter_turns);
        reduced_angle = *this - prod0;

        // Stage 2: Subtract k·π₁
        f118_t prod1 = two_prod_(half_pi_1, quarter_turns);
        reduced_angle = reduced_angle - prod1;

        // Stage 3: Subtract k·π₂ (scalar, precision beyond double-double)
        reduced_angle = reduced_angle - f118_t(half_pi_2 * quarter_turns);

        // Stage 4: Subtract k·π₃ (extra precision for edge cases)
        reduced_angle = reduced_angle - f118_t(half_pi_3 * quarter_turns);
    }

    /**
     *  @brief Taylor series for sin(x), assumes |x| < π/4.
     *  @param angle Input angle in radians (must be small)
     *  @return sin(angle) with ~106 bits precision
     */
    constexpr f118_t sin_taylor_(f118_t angle) const noexcept {
        f118_t angle_squared = angle * angle;
        f118_t series_sum = angle;
        f118_t current_term = angle;
        for (unsigned term_index = 1; term_index <= 25; ++term_index) {
            // termₙ₊₁ = termₙ · (−x²) / ((2n)(2n+1))
            double divisor = -double((2 * term_index) * (2 * term_index + 1));
            current_term = current_term * angle_squared / f118_t(divisor);
            f118_t updated_sum = series_sum + current_term;
            if (std::abs(current_term.high_) < 1e-35 * std::abs(series_sum.high_)) break;
            series_sum = updated_sum;
        }
        return series_sum;
    }

    /**
     *  @brief Taylor series for cos(x), assumes |x| < π/4.
     *  @param angle Input angle in radians (must be small)
     *  @return cos(angle) with ~106 bits precision
     */
    constexpr f118_t cos_taylor_(f118_t angle) const noexcept {
        f118_t angle_squared = angle * angle;
        f118_t series_sum(1.0);
        f118_t current_term(1.0);
        for (unsigned term_index = 1; term_index <= 25; ++term_index) {
            // termₙ₊₁ = termₙ · (−x²) / ((2n−1)(2n))
            double divisor = -double((2 * term_index - 1) * (2 * term_index));
            current_term = current_term * angle_squared / f118_t(divisor);
            f118_t updated_sum = series_sum + current_term;
            if (std::abs(current_term.high_) < 1e-35 * std::abs(series_sum.high_)) break;
            series_sum = updated_sum;
        }
        return series_sum;
    }

    /**
     *  @brief Error-free addition (Knuth two-sum algorithm).
     *
     *  Computes sum exactly, where `high_ + low_ == a + b` with no rounding error.
     *  Works for any IEEE 754 floats.
     *
     *  @param a First operand
     *  @param b Second operand
     *  @return f118_t where `high_ == a+b` (rounded), `low_ == rounding error`
     *
     *  Reference: Knuth, "The Art of Computer Programming", Vol 2, Section 4.2.2
     */
    static constexpr f118_t two_sum_(double a, double b) noexcept {
        double sum = a + b;
        double b_virtual = sum - a;
        return f118_t(sum, (a - (sum - b_virtual)) + (b - b_virtual));
    }

    /**
     *  @brief Fast error-free addition when |a| ≥ |b|.
     *
     *  Faster variant of `two_sum_` that requires |a| ≥ |b| as a precondition.
     *  Saves 3 floating-point operations vs `two_sum_`.
     *
     *  @param a First operand (must satisfy |a| ≥ |b|)
     *  @param b Second operand
     *  @return f118_t where `high_ == a+b` (rounded), `low_ == rounding error`
     *
     *  Reference: Dekker, "A floating-point technique for extending
     *             the available precision", 1971
     */
    static constexpr f118_t quick_two_sum_(double a, double b) noexcept { return f118_t(a + b, b - ((a + b) - a)); }

    /**
     *  @brief Error-free product using FMA (two_prod_ algorithm).
     *
     *  Computes product exactly as a double-double, where `high_ + low_ == a * b`
     *  with no rounding error. Requires FMA hardware support.
     *
     *  @param a First operand
     *  @param b Second operand
     *  @return f118_t where `high_ == a*b` (rounded), `low_ == rounding error`
     */
    static constexpr f118_t two_prod_(double a, double b) noexcept {
        double product = a * b;
        double error = std::fma(a, b, -product);
        return f118_t(product, error);
    }

    /** @brief Multiply double-double by scalar with extended precision. */
    constexpr f118_t mul_scalar_(double k) const noexcept {
        f118_t p = two_prod_(high_, k);
        p.low_ += low_ * k;
        return quick_two_sum_(p.high_, p.low_);
    }
};

/**
 *  @brief Extended-precision complex number using f118_t components.
 *
 *  Provides ~91 bits of precision for complex arithmetic, useful for:
 *  - High-precision reference computations in tests
 *  - Complex dot products and correlations
 *  - Avoiding Boost.Multiprecision dependency
 *
 *  Format: real + i*imag where real and imag are f118_t (double-double).
 */
struct f118c_t {
    f118_t real_, imag_;

    // Type metadata for consistency with other scalar types
    static constexpr nk_dtype_t dtype() noexcept { return nk_dtype_unknown_k; }
    static constexpr char const *dtype_name() noexcept { return "f118c"; }
    static constexpr unsigned bits_per_word() noexcept { return 128; }      // One double-double component
    static constexpr unsigned bits_per_dimension() noexcept { return 256; } // One complex number
    static constexpr unsigned bits_per_value() noexcept { return 256; }
    static constexpr unsigned mantissa_bits() noexcept { return 103; }
    static constexpr bool is_integer() noexcept { return false; }
    static constexpr bool is_signed() noexcept { return true; }
    static constexpr bool is_complex() noexcept { return true; }
    static constexpr bool is_exact() noexcept { return false; }
    static constexpr bool has_infinity() noexcept { return true; }
    static constexpr bool has_nan() noexcept { return true; }

    // Static constants
    static constexpr f118c_t zero() noexcept { return f118c_t {}; }
    static constexpr f118c_t one() noexcept { return f118c_t {f118_t::one()}; }

    static constexpr f118c_t finite_max() noexcept { return f118c_t {f118_t::finite_max(), f118_t::finite_max()}; }
    static constexpr f118c_t finite_min() noexcept { return f118c_t {f118_t::finite_min(), f118_t::finite_min()}; }

    constexpr f118c_t() noexcept : real_(), imag_() {}
    constexpr f118c_t(f118_t r) noexcept : real_(r), imag_() {}
    constexpr f118c_t(f118_t r, f118_t i) noexcept : real_(r), imag_(i) {}
    constexpr f118c_t(double r, double i = 0.0) noexcept : real_(r), imag_(i) {}
    constexpr f118c_t(int v) noexcept : real_(v), imag_() {}

    constexpr f118c_t(f32c_t c) noexcept : real_(c.real()), imag_(c.imag()) {}
    constexpr f118c_t(f64c_t c) noexcept : real_(c.real()), imag_(c.imag()) {}
    inline f118c_t(f16c_t c) noexcept : real_(static_cast<float>(c.real())), imag_(static_cast<float>(c.imag())) {}
    inline f118c_t(bf16c_t c) noexcept : real_(static_cast<float>(c.real())), imag_(static_cast<float>(c.imag())) {}

    constexpr f118_t real() const noexcept { return real_; }
    constexpr f118_t imag() const noexcept { return imag_; }

    constexpr f118c_t operator+(f118c_t o) const noexcept { return {real_ + o.real_, imag_ + o.imag_}; }
    constexpr f118c_t operator-(f118c_t o) const noexcept { return {real_ - o.real_, imag_ - o.imag_}; }
    constexpr f118c_t operator*(f118c_t o) const noexcept {
        return {real_ * o.real_ - imag_ * o.imag_, real_ * o.imag_ + imag_ * o.real_};
    }
    constexpr f118c_t operator/(f118c_t o) const noexcept {
        f118_t denom = o.real_ * o.real_ + o.imag_ * o.imag_;
        return {(real_ * o.real_ + imag_ * o.imag_) / denom, (imag_ * o.real_ - real_ * o.imag_) / denom};
    }

    constexpr f118c_t &operator+=(f118c_t o) noexcept { return *this = *this + o; }
    constexpr f118c_t &operator-=(f118c_t o) noexcept { return *this = *this - o; }
    constexpr f118c_t &operator*=(f118c_t o) noexcept { return *this = *this * o; }
    constexpr f118c_t &operator/=(f118c_t o) noexcept { return *this = *this / o; }

    constexpr f118c_t operator-() const noexcept { return {-real_, -imag_}; }
    constexpr f118c_t conj() const noexcept { return {real_, -imag_}; }

    /** @brief Squared magnitude: |z|² = real² + imag² */
    constexpr f118_t norm_sq() const noexcept { return real_ * real_ + imag_ * imag_; }

    /** @brief Magnitude: |z| = sqrt(real² + imag²) */
    constexpr f118_t abs() const noexcept { return norm_sq().sqrt(); }

    constexpr bool operator==(f118c_t const &o) const noexcept { return real_ == o.real_ && imag_ == o.imag_; }
    constexpr bool operator!=(f118c_t const &o) const noexcept { return !(*this == o); }

    constexpr f64c_t to_f64c() const noexcept { return f64c_t(real_.high_, imag_.high_); }
    constexpr f32c_t to_f32c() const noexcept {
        return f32c_t(static_cast<float>(real_.high_), static_cast<float>(imag_.high_));
    }
};

/**
 *  @brief Signed 8-bit integer wrapper.
 *
 *  Provides strong type identity for `int8_t`, compatible with NumKong kernels
 *  and `std::mdspan`.
 *
 *  Features:
 *  - All arithmetic operators
 *  - Bitwise operators: `&`, `|`, `^`, `~`, `<<`, `>>`
 *  - Integer-specific: `abs()`, widened accumulation types
 *  - No NaN/infinity concepts (integers are exact)
 */
struct i8_t {
    // Core type aliases
    using raw_t = nk_i8_t;
    using unsigned_t = nk_u8_t;

    using dot_result_t = i32_t;         // `nk_dot_i8` output (widened)
    using reduce_add_result_t = i64_t;  // `nk_reduce_add_i8` widened output
    using sqeuclidean_result_t = u32_t; // `nk_sqeuclidean_i8` output (widened)

    using dot_kernel_t = void (*)(raw_t const *, raw_t const *, nk_size_t, nk_i32_t *);
    using dots_symmetric_kernel_t = void (*)(raw_t const *, nk_size_t, nk_size_t, nk_size_t, nk_i32_t *, nk_size_t,
                                             nk_size_t, nk_size_t);
    using dots_packed_size_kernel_t = nk_size_t (*)(nk_size_t, nk_size_t);
    using dots_pack_kernel_t = void (*)(raw_t const *, nk_size_t, nk_size_t, nk_size_t, void *);
    using dots_packed_kernel_t = void (*)(raw_t const *, void const *, nk_i32_t *, nk_size_t, nk_size_t, nk_size_t,
                                          nk_size_t, nk_size_t);

    static constexpr nk_dtype_t dtype() noexcept { return nk_i8_k; }
    static constexpr char const *dtype_name() noexcept { return "i8"; }
    static constexpr unsigned bits_per_word() noexcept { return 8; }
    static constexpr unsigned bits_per_dimension() noexcept { return 8; }
    static constexpr unsigned bits_per_value() noexcept { return 8; }
    static constexpr bool is_integer() noexcept { return true; }
    static constexpr bool is_signed() noexcept { return true; }
    static constexpr bool is_complex() noexcept { return false; }
    static constexpr bool is_exact() noexcept { return true; }
    static constexpr bool has_infinity() noexcept { return false; }
    static constexpr bool has_nan() noexcept { return false; }

    raw_t raw_;

    constexpr i8_t() noexcept : raw_(0) {}
    constexpr i8_t(std::int8_t v) noexcept : raw_(v) {}
    template <std::integral integral_type_>
    constexpr i8_t(integral_type_ v) noexcept : raw_(static_cast<raw_t>(v)) {}
    constexpr operator std::int8_t() const noexcept { return raw_; }
    constexpr std::int8_t raw() const noexcept { return raw_; }
    static constexpr i8_t from_raw(raw_t r) noexcept { return i8_t {r}; }

    static constexpr i8_t finite_min() noexcept { return i8_t {std::int8_t(-128)}; }
    static constexpr i8_t finite_max() noexcept { return i8_t {std::int8_t(127)}; }
    static constexpr i8_t zero() noexcept { return i8_t {}; }
    static constexpr i8_t one() noexcept { return i8_t {std::int8_t(1)}; }

    constexpr i8_t operator+() const noexcept { return *this; }
    constexpr i8_t operator-() const noexcept { return i8_t {static_cast<raw_t>(-raw_)}; }
    constexpr i8_t operator+(i8_t o) const noexcept { return i8_t {static_cast<raw_t>(raw_ + o.raw_)}; }
    constexpr i8_t operator-(i8_t o) const noexcept { return i8_t {static_cast<raw_t>(raw_ - o.raw_)}; }
    constexpr i8_t operator*(i8_t o) const noexcept { return i8_t {static_cast<raw_t>(raw_ * o.raw_)}; }
    constexpr i8_t operator/(i8_t o) const noexcept { return i8_t {static_cast<raw_t>(raw_ / o.raw_)}; }
    constexpr i8_t operator%(i8_t o) const noexcept { return i8_t {static_cast<raw_t>(raw_ % o.raw_)}; }

    constexpr i8_t &operator+=(i8_t o) noexcept {
        raw_ = static_cast<raw_t>(raw_ + o.raw_);
        return *this;
    }
    constexpr i8_t &operator-=(i8_t o) noexcept {
        raw_ = static_cast<raw_t>(raw_ - o.raw_);
        return *this;
    }
    constexpr i8_t &operator*=(i8_t o) noexcept {
        raw_ = static_cast<raw_t>(raw_ * o.raw_);
        return *this;
    }
    constexpr i8_t &operator/=(i8_t o) noexcept {
        raw_ = static_cast<raw_t>(raw_ / o.raw_);
        return *this;
    }
    constexpr i8_t &operator%=(i8_t o) noexcept {
        raw_ = static_cast<raw_t>(raw_ % o.raw_);
        return *this;
    }

    constexpr i8_t operator~() const noexcept { return i8_t {static_cast<raw_t>(~raw_)}; }
    constexpr i8_t operator&(i8_t o) const noexcept { return i8_t {static_cast<raw_t>(raw_ & o.raw_)}; }
    constexpr i8_t operator|(i8_t o) const noexcept { return i8_t {static_cast<raw_t>(raw_ | o.raw_)}; }
    constexpr i8_t operator^(i8_t o) const noexcept { return i8_t {static_cast<raw_t>(raw_ ^ o.raw_)}; }
    constexpr i8_t operator<<(int n) const noexcept { return i8_t {static_cast<raw_t>(raw_ << n)}; }
    constexpr i8_t operator>>(int n) const noexcept { return i8_t {static_cast<raw_t>(raw_ >> n)}; }

    constexpr i8_t &operator&=(i8_t o) noexcept {
        raw_ = static_cast<raw_t>(raw_ & o.raw_);
        return *this;
    }
    constexpr i8_t &operator|=(i8_t o) noexcept {
        raw_ = static_cast<raw_t>(raw_ | o.raw_);
        return *this;
    }
    constexpr i8_t &operator^=(i8_t o) noexcept {
        raw_ = static_cast<raw_t>(raw_ ^ o.raw_);
        return *this;
    }
    constexpr i8_t &operator<<=(int n) noexcept {
        raw_ = static_cast<raw_t>(raw_ << n);
        return *this;
    }
    constexpr i8_t &operator>>=(int n) noexcept {
        raw_ = static_cast<raw_t>(raw_ >> n);
        return *this;
    }

    constexpr bool operator==(i8_t o) const noexcept { return raw_ == o.raw_; }
    constexpr bool operator!=(i8_t o) const noexcept { return raw_ != o.raw_; }
    constexpr bool operator<(i8_t o) const noexcept { return raw_ < o.raw_; }
    constexpr bool operator>(i8_t o) const noexcept { return raw_ > o.raw_; }
    constexpr bool operator<=(i8_t o) const noexcept { return raw_ <= o.raw_; }
    constexpr bool operator>=(i8_t o) const noexcept { return raw_ >= o.raw_; }

    constexpr i8_t abs() const noexcept { return raw_ < 0 ? i8_t {static_cast<raw_t>(-raw_)} : *this; }
    constexpr i8_t signum() const noexcept {
        if (raw_ > 0) return i8_t {std::int8_t(1)};
        if (raw_ < 0) return i8_t {std::int8_t(-1)};
        return i8_t {};
    }
    constexpr int total_cmp(i8_t o) const noexcept { return (raw_ > o.raw_) - (raw_ < o.raw_); }
    constexpr i8_t min(i8_t o) const noexcept { return raw_ < o.raw_ ? *this : o; }
    constexpr i8_t max(i8_t o) const noexcept { return raw_ > o.raw_ ? *this : o; }
    constexpr i8_t clamp(i8_t lo, i8_t hi) const noexcept { return max(lo).min(hi); }

    constexpr i8_t saturating_add(i8_t o) const noexcept {
        nk_i32_t result = nk_i32_t(raw_) + nk_i32_t(o.raw_);
        if (result > NK_I8_MAX) return i8_t::finite_max();
        if (result < NK_I8_MIN) return i8_t::finite_min();
        return i8_t {static_cast<raw_t>(result)};
    }
    constexpr i8_t saturating_sub(i8_t o) const noexcept {
        nk_i32_t result = nk_i32_t(raw_) - nk_i32_t(o.raw_);
        if (result > NK_I8_MAX) return i8_t::finite_max();
        if (result < NK_I8_MIN) return i8_t::finite_min();
        return i8_t {static_cast<raw_t>(result)};
    }
};

/**
 *  @brief Unsigned 8-bit integer wrapper.
 *
 *  Provides strong type identity for `uint8_t`, compatible with NumKong kernels
 *  and `std::mdspan`.
 *
 *  Features:
 *  - All arithmetic operators
 *  - Bitwise operators: `&`, `|`, `^`, `~`, `<<`, `>>`
 *  - Widened accumulation types for kernel outputs
 *  - No NaN/infinity concepts (integers are exact)
 */
struct u8_t {
    // Core type aliases
    using raw_t = nk_u8_t;
    using signed_t = nk_i8_t;

    using dot_result_t = u32_t;         // `nk_dot_u8` output (widened)
    using reduce_add_result_t = u64_t;  // `nk_reduce_add_u8` widened output
    using sqeuclidean_result_t = u32_t; // `nk_sqeuclidean_u8` output (widened)
    using hamming_result_t = u32_t;     // `nk_hamming_u8` output

    using dots_symmetric_kernel_t = void (*)(raw_t const *, nk_size_t, nk_size_t, nk_size_t, nk_u32_t *, nk_size_t,
                                             nk_size_t, nk_size_t);
    using dot_kernel_t = void (*)(raw_t const *, raw_t const *, nk_size_t, nk_u32_t *);
    using dots_packed_size_kernel_t = nk_size_t (*)(nk_size_t, nk_size_t);
    using dots_pack_kernel_t = void (*)(raw_t const *, nk_size_t, nk_size_t, nk_size_t, void *);
    using dots_packed_kernel_t = void (*)(raw_t const *, void const *, nk_u32_t *, nk_size_t, nk_size_t, nk_size_t,
                                          nk_size_t, nk_size_t);

    static constexpr nk_dtype_t dtype() noexcept { return nk_u8_k; }
    static constexpr char const *dtype_name() noexcept { return "u8"; }
    static constexpr unsigned bits_per_word() noexcept { return 8; }
    static constexpr unsigned bits_per_dimension() noexcept { return 8; }
    static constexpr unsigned bits_per_value() noexcept { return 8; }
    static constexpr bool is_integer() noexcept { return true; }
    static constexpr bool is_signed() noexcept { return false; }
    static constexpr bool is_complex() noexcept { return false; }
    static constexpr bool is_exact() noexcept { return true; }
    static constexpr bool has_infinity() noexcept { return false; }
    static constexpr bool has_nan() noexcept { return false; }

    raw_t raw_;

    constexpr u8_t() noexcept : raw_(0) {}
    constexpr u8_t(std::uint8_t v) noexcept : raw_(v) {}
    template <std::integral integral_type_>
    constexpr u8_t(integral_type_ v) noexcept : raw_(static_cast<raw_t>(v)) {}
    constexpr operator std::uint8_t() const noexcept { return raw_; }
    constexpr std::uint8_t raw() const noexcept { return raw_; }
    static constexpr u8_t from_raw(raw_t r) noexcept { return u8_t {r}; }

    static constexpr u8_t finite_min() noexcept { return u8_t {}; }
    static constexpr u8_t finite_max() noexcept { return u8_t {std::uint8_t(255)}; }
    static constexpr u8_t zero() noexcept { return u8_t {}; }
    static constexpr u8_t one() noexcept { return u8_t {std::uint8_t(1)}; }

    constexpr u8_t operator+() const noexcept { return *this; }
    constexpr u8_t operator+(u8_t o) const noexcept { return u8_t {static_cast<raw_t>(raw_ + o.raw_)}; }
    constexpr u8_t operator-(u8_t o) const noexcept { return u8_t {static_cast<raw_t>(raw_ - o.raw_)}; }
    constexpr u8_t operator*(u8_t o) const noexcept { return u8_t {static_cast<raw_t>(raw_ * o.raw_)}; }
    constexpr u8_t operator/(u8_t o) const noexcept { return u8_t {static_cast<raw_t>(raw_ / o.raw_)}; }
    constexpr u8_t operator%(u8_t o) const noexcept { return u8_t {static_cast<raw_t>(raw_ % o.raw_)}; }

    constexpr u8_t &operator+=(u8_t o) noexcept {
        raw_ = static_cast<raw_t>(raw_ + o.raw_);
        return *this;
    }
    constexpr u8_t &operator-=(u8_t o) noexcept {
        raw_ = static_cast<raw_t>(raw_ - o.raw_);
        return *this;
    }
    constexpr u8_t &operator*=(u8_t o) noexcept {
        raw_ = static_cast<raw_t>(raw_ * o.raw_);
        return *this;
    }
    constexpr u8_t &operator/=(u8_t o) noexcept {
        raw_ = static_cast<raw_t>(raw_ / o.raw_);
        return *this;
    }
    constexpr u8_t &operator%=(u8_t o) noexcept {
        raw_ = static_cast<raw_t>(raw_ % o.raw_);
        return *this;
    }

    constexpr u8_t operator~() const noexcept { return u8_t {static_cast<raw_t>(~raw_)}; }
    constexpr u8_t operator&(u8_t o) const noexcept { return u8_t {static_cast<raw_t>(raw_ & o.raw_)}; }
    constexpr u8_t operator|(u8_t o) const noexcept { return u8_t {static_cast<raw_t>(raw_ | o.raw_)}; }
    constexpr u8_t operator^(u8_t o) const noexcept { return u8_t {static_cast<raw_t>(raw_ ^ o.raw_)}; }
    constexpr u8_t operator<<(int n) const noexcept { return u8_t {static_cast<raw_t>(raw_ << n)}; }
    constexpr u8_t operator>>(int n) const noexcept { return u8_t {static_cast<raw_t>(raw_ >> n)}; }

    constexpr u8_t &operator&=(u8_t o) noexcept {
        raw_ = static_cast<raw_t>(raw_ & o.raw_);
        return *this;
    }
    constexpr u8_t &operator|=(u8_t o) noexcept {
        raw_ = static_cast<raw_t>(raw_ | o.raw_);
        return *this;
    }
    constexpr u8_t &operator^=(u8_t o) noexcept {
        raw_ = static_cast<raw_t>(raw_ ^ o.raw_);
        return *this;
    }
    constexpr u8_t &operator<<=(int n) noexcept {
        raw_ = static_cast<raw_t>(raw_ << n);
        return *this;
    }
    constexpr u8_t &operator>>=(int n) noexcept {
        raw_ = static_cast<raw_t>(raw_ >> n);
        return *this;
    }

    constexpr bool operator==(u8_t o) const noexcept { return raw_ == o.raw_; }
    constexpr bool operator!=(u8_t o) const noexcept { return raw_ != o.raw_; }
    constexpr bool operator<(u8_t o) const noexcept { return raw_ < o.raw_; }
    constexpr bool operator>(u8_t o) const noexcept { return raw_ > o.raw_; }
    constexpr bool operator<=(u8_t o) const noexcept { return raw_ <= o.raw_; }
    constexpr bool operator>=(u8_t o) const noexcept { return raw_ >= o.raw_; }

    constexpr int total_cmp(u8_t o) const noexcept { return (raw_ > o.raw_) - (raw_ < o.raw_); }
    constexpr u8_t min(u8_t o) const noexcept { return raw_ < o.raw_ ? *this : o; }
    constexpr u8_t max(u8_t o) const noexcept { return raw_ > o.raw_ ? *this : o; }
    constexpr u8_t clamp(u8_t lo, u8_t hi) const noexcept { return max(lo).min(hi); }

    constexpr u8_t saturating_add(u8_t o) const noexcept {
        nk_u32_t result = nk_u32_t(raw_) + nk_u32_t(o.raw_);
        return result > NK_U8_MAX ? u8_t::finite_max() : u8_t {static_cast<raw_t>(result)};
    }
    constexpr u8_t saturating_sub(u8_t o) const noexcept {
        return o.raw_ > raw_ ? u8_t::zero() : u8_t {static_cast<raw_t>(raw_ - o.raw_)};
    }
};

/**
 *  @brief Signed 32-bit integer wrapper.
 *
 *  Provides strong type identity for `int32_t`, compatible with NumKong kernels
 *  and `std::mdspan`.
 *
 *  Features:
 *  - All arithmetic operators
 *  - Bitwise operators: `&`, `|`, `^`, `~`, `<<`, `>>`
 *  - Integer-specific: `abs()`, widened accumulation types
 *  - No NaN/infinity concepts (integers are exact)
 */
struct i32_t {
    // Core type aliases
    using raw_t = nk_i32_t;
    using unsigned_t = nk_u32_t;

    using reduce_add_result_t = i64_t; // `nk_reduce_add_i32` widened output
    using reduce_add_kernel_t = void (*)(raw_t const *, nk_size_t, nk_size_t, nk_i64_t *);
    using reduce_extremum_kernel_t = void (*)(raw_t const *, nk_size_t, nk_size_t, raw_t *, nk_size_t *);

    static constexpr nk_dtype_t dtype() noexcept { return nk_i32_k; }
    static constexpr char const *dtype_name() noexcept { return "i32"; }
    static constexpr unsigned bits_per_word() noexcept { return 32; }
    static constexpr unsigned bits_per_dimension() noexcept { return 32; }
    static constexpr unsigned bits_per_value() noexcept { return 32; }
    static constexpr bool is_integer() noexcept { return true; }
    static constexpr bool is_signed() noexcept { return true; }
    static constexpr bool is_complex() noexcept { return false; }
    static constexpr bool is_exact() noexcept { return true; }
    static constexpr bool has_infinity() noexcept { return false; }
    static constexpr bool has_nan() noexcept { return false; }

    raw_t raw_;

    constexpr i32_t() noexcept : raw_(0) {}
    constexpr i32_t(std::int32_t v) noexcept : raw_(v) {}
    template <std::integral integral_type_>
    constexpr i32_t(integral_type_ v) noexcept : raw_(static_cast<raw_t>(v)) {}
    constexpr operator std::int32_t() const noexcept { return raw_; }
    constexpr std::int32_t raw() const noexcept { return raw_; }
    static constexpr i32_t from_raw(raw_t r) noexcept { return i32_t {r}; }

    static constexpr i32_t finite_min() noexcept { return i32_t {std::int32_t(-2147483648)}; }
    static constexpr i32_t finite_max() noexcept { return i32_t {std::int32_t(2147483647)}; }
    static constexpr i32_t zero() noexcept { return i32_t {}; }
    static constexpr i32_t one() noexcept { return i32_t {1}; }

    constexpr i32_t operator+() const noexcept { return *this; }
    constexpr i32_t operator-() const noexcept { return i32_t {-raw_}; }
    constexpr i32_t operator+(i32_t o) const noexcept { return i32_t {raw_ + o.raw_}; }
    constexpr i32_t operator-(i32_t o) const noexcept { return i32_t {raw_ - o.raw_}; }
    constexpr i32_t operator*(i32_t o) const noexcept { return i32_t {raw_ * o.raw_}; }
    constexpr i32_t operator/(i32_t o) const noexcept { return i32_t {raw_ / o.raw_}; }
    constexpr i32_t operator%(i32_t o) const noexcept { return i32_t {raw_ % o.raw_}; }

    constexpr i32_t &operator+=(i32_t o) noexcept {
        raw_ += o.raw_;
        return *this;
    }
    constexpr i32_t &operator-=(i32_t o) noexcept {
        raw_ -= o.raw_;
        return *this;
    }
    constexpr i32_t &operator*=(i32_t o) noexcept {
        raw_ *= o.raw_;
        return *this;
    }
    constexpr i32_t &operator/=(i32_t o) noexcept {
        raw_ /= o.raw_;
        return *this;
    }
    constexpr i32_t &operator%=(i32_t o) noexcept {
        raw_ %= o.raw_;
        return *this;
    }

    constexpr i32_t operator~() const noexcept { return i32_t {~raw_}; }
    constexpr i32_t operator&(i32_t o) const noexcept { return i32_t {raw_ & o.raw_}; }
    constexpr i32_t operator|(i32_t o) const noexcept { return i32_t {raw_ | o.raw_}; }
    constexpr i32_t operator^(i32_t o) const noexcept { return i32_t {raw_ ^ o.raw_}; }
    constexpr i32_t operator<<(int n) const noexcept { return i32_t {raw_ << n}; }
    constexpr i32_t operator>>(int n) const noexcept { return i32_t {raw_ >> n}; }

    constexpr i32_t &operator&=(i32_t o) noexcept {
        raw_ &= o.raw_;
        return *this;
    }
    constexpr i32_t &operator|=(i32_t o) noexcept {
        raw_ |= o.raw_;
        return *this;
    }
    constexpr i32_t &operator^=(i32_t o) noexcept {
        raw_ ^= o.raw_;
        return *this;
    }
    constexpr i32_t &operator<<=(int n) noexcept {
        raw_ <<= n;
        return *this;
    }
    constexpr i32_t &operator>>=(int n) noexcept {
        raw_ >>= n;
        return *this;
    }

    constexpr bool operator==(i32_t o) const noexcept { return raw_ == o.raw_; }
    constexpr bool operator!=(i32_t o) const noexcept { return raw_ != o.raw_; }
    constexpr bool operator<(i32_t o) const noexcept { return raw_ < o.raw_; }
    constexpr bool operator>(i32_t o) const noexcept { return raw_ > o.raw_; }
    constexpr bool operator<=(i32_t o) const noexcept { return raw_ <= o.raw_; }
    constexpr bool operator>=(i32_t o) const noexcept { return raw_ >= o.raw_; }

    constexpr i32_t abs() const noexcept { return raw_ < 0 ? i32_t {-raw_} : *this; }
    constexpr i32_t signum() const noexcept {
        if (raw_ > 0) return i32_t {1};
        if (raw_ < 0) return i32_t {-1};
        return i32_t {};
    }
    constexpr int total_cmp(i32_t o) const noexcept { return (raw_ > o.raw_) - (raw_ < o.raw_); }
    constexpr i32_t min(i32_t o) const noexcept { return raw_ < o.raw_ ? *this : o; }
    constexpr i32_t max(i32_t o) const noexcept { return raw_ > o.raw_ ? *this : o; }
    constexpr i32_t clamp(i32_t lo, i32_t hi) const noexcept { return max(lo).min(hi); }

    constexpr i32_t saturating_add(i32_t o) const noexcept {
        nk_i64_t result = nk_i64_t(raw_) + nk_i64_t(o.raw_);
        if (result > NK_I32_MAX) return i32_t::finite_max();
        if (result < NK_I32_MIN) return i32_t::finite_min();
        return i32_t {static_cast<raw_t>(result)};
    }
    constexpr i32_t saturating_sub(i32_t o) const noexcept {
        nk_i64_t result = nk_i64_t(raw_) - nk_i64_t(o.raw_);
        if (result > NK_I32_MAX) return i32_t::finite_max();
        if (result < NK_I32_MIN) return i32_t::finite_min();
        return i32_t {static_cast<raw_t>(result)};
    }
};

/**
 *  @brief Unsigned 32-bit integer wrapper.
 *
 *  Provides strong type identity for `std::uint32_t`, compatible with NumKong
 *  kernels and `std::mdspan`.
 *
 *  Features:
 *  - Arithmetic operators (wrapping semantics)
 *  - Bitwise operators
 *  - Spaceship operator for comparisons
 *  - Integer-specific: min, max, clamp
 */
struct u32_t {
    // Core type aliases
    using raw_t = nk_u32_t;
    using signed_t = nk_i32_t;

    using reduce_add_result_t = u64_t; // `nk_reduce_add_u32` widened output
    using jaccard_result_t = f32_t;    // `nk_jaccard_u32` output

    using sparse_intersect_kernel_t = void (*)(raw_t const *, raw_t const *, nk_size_t, nk_size_t, raw_t *,
                                               nk_size_t *);

    static constexpr nk_dtype_t dtype() noexcept { return nk_u32_k; }
    static constexpr char const *dtype_name() noexcept { return "u32"; }
    static constexpr unsigned bits_per_word() noexcept { return 32; }
    static constexpr unsigned bits_per_dimension() noexcept { return 32; }
    static constexpr unsigned bits_per_value() noexcept { return 32; }
    static constexpr bool is_integer() noexcept { return true; }
    static constexpr bool is_signed() noexcept { return false; }
    static constexpr bool is_complex() noexcept { return false; }
    static constexpr bool is_exact() noexcept { return true; }
    static constexpr bool has_infinity() noexcept { return false; }
    static constexpr bool has_nan() noexcept { return false; }

    raw_t raw_;

    constexpr u32_t() noexcept : raw_(0) {}
    constexpr u32_t(std::uint32_t v) noexcept : raw_(v) {}
    template <std::integral integral_type_>
    constexpr u32_t(integral_type_ v) noexcept : raw_(static_cast<raw_t>(v)) {}
    constexpr operator std::uint32_t() const noexcept { return raw_; }
    constexpr std::uint32_t raw() const noexcept { return raw_; }
    static constexpr u32_t from_raw(raw_t r) noexcept { return u32_t {r}; }

    static constexpr u32_t finite_min() noexcept { return u32_t {}; }
    static constexpr u32_t finite_max() noexcept { return u32_t {std::uint32_t(4294967295u)}; }
    static constexpr u32_t zero() noexcept { return u32_t {}; }
    static constexpr u32_t one() noexcept { return u32_t {std::uint32_t(1)}; }

    constexpr u32_t operator+() const noexcept { return *this; }
    constexpr u32_t operator+(u32_t o) const noexcept { return u32_t {raw_ + o.raw_}; }
    constexpr u32_t operator-(u32_t o) const noexcept { return u32_t {raw_ - o.raw_}; }
    constexpr u32_t operator*(u32_t o) const noexcept { return u32_t {raw_ * o.raw_}; }
    constexpr u32_t operator/(u32_t o) const noexcept { return u32_t {raw_ / o.raw_}; }
    constexpr u32_t operator%(u32_t o) const noexcept { return u32_t {raw_ % o.raw_}; }

    constexpr u32_t &operator+=(u32_t o) noexcept {
        raw_ += o.raw_;
        return *this;
    }
    constexpr u32_t &operator-=(u32_t o) noexcept {
        raw_ -= o.raw_;
        return *this;
    }
    constexpr u32_t &operator*=(u32_t o) noexcept {
        raw_ *= o.raw_;
        return *this;
    }
    constexpr u32_t &operator/=(u32_t o) noexcept {
        raw_ /= o.raw_;
        return *this;
    }
    constexpr u32_t &operator%=(u32_t o) noexcept {
        raw_ %= o.raw_;
        return *this;
    }

    constexpr u32_t operator~() const noexcept { return u32_t {~raw_}; }
    constexpr u32_t operator&(u32_t o) const noexcept { return u32_t {raw_ & o.raw_}; }
    constexpr u32_t operator|(u32_t o) const noexcept { return u32_t {raw_ | o.raw_}; }
    constexpr u32_t operator^(u32_t o) const noexcept { return u32_t {raw_ ^ o.raw_}; }
    constexpr u32_t operator<<(int n) const noexcept { return u32_t {raw_ << n}; }
    constexpr u32_t operator>>(int n) const noexcept { return u32_t {raw_ >> n}; }

    constexpr u32_t &operator&=(u32_t o) noexcept {
        raw_ &= o.raw_;
        return *this;
    }
    constexpr u32_t &operator|=(u32_t o) noexcept {
        raw_ |= o.raw_;
        return *this;
    }
    constexpr u32_t &operator^=(u32_t o) noexcept {
        raw_ ^= o.raw_;
        return *this;
    }
    constexpr u32_t &operator<<=(int n) noexcept {
        raw_ <<= n;
        return *this;
    }
    constexpr u32_t &operator>>=(int n) noexcept {
        raw_ >>= n;
        return *this;
    }

    constexpr auto operator<=>(u32_t const &o) const noexcept = default;

    constexpr int total_cmp(u32_t o) const noexcept { return (raw_ > o.raw_) - (raw_ < o.raw_); }
    constexpr u32_t min(u32_t o) const noexcept { return raw_ < o.raw_ ? *this : o; }
    constexpr u32_t max(u32_t o) const noexcept { return raw_ > o.raw_ ? *this : o; }
    constexpr u32_t clamp(u32_t lo, u32_t hi) const noexcept { return max(lo).min(hi); }

    constexpr u32_t saturating_add(u32_t o) const noexcept {
        nk_u64_t result = nk_u64_t(raw_) + nk_u64_t(o.raw_);
        return result > NK_U32_MAX ? u32_t::finite_max() : u32_t {static_cast<raw_t>(result)};
    }
    constexpr u32_t saturating_sub(u32_t o) const noexcept {
        return o.raw_ > raw_ ? u32_t::zero() : u32_t {static_cast<raw_t>(raw_ - o.raw_)};
    }
};

/**
 *  @brief Signed 64-bit integer wrapper.
 *
 *  Provides strong type identity for `std::int64_t`, compatible with NumKong
 *  kernels and `std::mdspan`.
 *
 *  Features:
 *  - Arithmetic operators (wrapping semantics)
 *  - Bitwise operators
 *  - Integer-specific: abs, signum, min, max, clamp
 */
struct i64_t {
    // Core type aliases
    using raw_t = nk_i64_t;
    using unsigned_t = nk_u64_t;

    using reduce_add_result_t = i64_t; // `nk_reduce_add_i64` (no widening, already max)

    static constexpr nk_dtype_t dtype() noexcept { return nk_i64_k; }
    static constexpr char const *dtype_name() noexcept { return "i64"; }
    static constexpr unsigned bits_per_word() noexcept { return 64; }
    static constexpr unsigned bits_per_dimension() noexcept { return 64; }
    static constexpr unsigned bits_per_value() noexcept { return 64; }
    static constexpr bool is_integer() noexcept { return true; }
    static constexpr bool is_signed() noexcept { return true; }
    static constexpr bool is_complex() noexcept { return false; }
    static constexpr bool is_exact() noexcept { return true; }
    static constexpr bool has_infinity() noexcept { return false; }
    static constexpr bool has_nan() noexcept { return false; }

    raw_t raw_;

    constexpr i64_t() noexcept : raw_(0) {}
    constexpr i64_t(std::int64_t v) noexcept : raw_(v) {}
    template <std::integral integral_type_>
    constexpr i64_t(integral_type_ v) noexcept : raw_(static_cast<raw_t>(v)) {}
    constexpr operator std::int64_t() const noexcept { return raw_; }
    constexpr std::int64_t raw() const noexcept { return raw_; }
    static constexpr i64_t from_raw(raw_t r) noexcept { return i64_t {r}; }

    static constexpr i64_t finite_min() noexcept { return i64_t {std::int64_t(-9223372036854775807LL - 1)}; }
    static constexpr i64_t finite_max() noexcept { return i64_t {std::int64_t(9223372036854775807LL)}; }
    static constexpr i64_t zero() noexcept { return i64_t {}; }
    static constexpr i64_t one() noexcept { return i64_t {1}; }

    constexpr i64_t operator+() const noexcept { return *this; }
    constexpr i64_t operator-() const noexcept { return i64_t {-raw_}; }
    constexpr i64_t operator+(i64_t o) const noexcept { return i64_t {raw_ + o.raw_}; }
    constexpr i64_t operator-(i64_t o) const noexcept { return i64_t {raw_ - o.raw_}; }
    constexpr i64_t operator*(i64_t o) const noexcept { return i64_t {raw_ * o.raw_}; }
    constexpr i64_t operator/(i64_t o) const noexcept { return i64_t {raw_ / o.raw_}; }
    constexpr i64_t operator%(i64_t o) const noexcept { return i64_t {raw_ % o.raw_}; }

    constexpr i64_t &operator+=(i64_t o) noexcept {
        raw_ += o.raw_;
        return *this;
    }
    constexpr i64_t &operator-=(i64_t o) noexcept {
        raw_ -= o.raw_;
        return *this;
    }
    constexpr i64_t &operator*=(i64_t o) noexcept {
        raw_ *= o.raw_;
        return *this;
    }
    constexpr i64_t &operator/=(i64_t o) noexcept {
        raw_ /= o.raw_;
        return *this;
    }
    constexpr i64_t &operator%=(i64_t o) noexcept {
        raw_ %= o.raw_;
        return *this;
    }

    constexpr i64_t operator~() const noexcept { return i64_t {~raw_}; }
    constexpr i64_t operator&(i64_t o) const noexcept { return i64_t {raw_ & o.raw_}; }
    constexpr i64_t operator|(i64_t o) const noexcept { return i64_t {raw_ | o.raw_}; }
    constexpr i64_t operator^(i64_t o) const noexcept { return i64_t {raw_ ^ o.raw_}; }
    constexpr i64_t operator<<(int n) const noexcept { return i64_t {raw_ << n}; }
    constexpr i64_t operator>>(int n) const noexcept { return i64_t {raw_ >> n}; }

    constexpr i64_t &operator&=(i64_t o) noexcept {
        raw_ &= o.raw_;
        return *this;
    }
    constexpr i64_t &operator|=(i64_t o) noexcept {
        raw_ |= o.raw_;
        return *this;
    }
    constexpr i64_t &operator^=(i64_t o) noexcept {
        raw_ ^= o.raw_;
        return *this;
    }
    constexpr i64_t &operator<<=(int n) noexcept {
        raw_ <<= n;
        return *this;
    }
    constexpr i64_t &operator>>=(int n) noexcept {
        raw_ >>= n;
        return *this;
    }

    constexpr bool operator==(i64_t o) const noexcept { return raw_ == o.raw_; }
    constexpr bool operator!=(i64_t o) const noexcept { return raw_ != o.raw_; }
    constexpr bool operator<(i64_t o) const noexcept { return raw_ < o.raw_; }
    constexpr bool operator>(i64_t o) const noexcept { return raw_ > o.raw_; }
    constexpr bool operator<=(i64_t o) const noexcept { return raw_ <= o.raw_; }
    constexpr bool operator>=(i64_t o) const noexcept { return raw_ >= o.raw_; }

    constexpr i64_t abs() const noexcept { return raw_ < 0 ? i64_t {-raw_} : *this; }
    constexpr i64_t signum() const noexcept {
        if (raw_ > 0) return i64_t {1};
        if (raw_ < 0) return i64_t {-1};
        return i64_t {};
    }
    constexpr int total_cmp(i64_t o) const noexcept { return (raw_ > o.raw_) - (raw_ < o.raw_); }
    constexpr i64_t min(i64_t o) const noexcept { return raw_ < o.raw_ ? *this : o; }
    constexpr i64_t max(i64_t o) const noexcept { return raw_ > o.raw_ ? *this : o; }
    constexpr i64_t clamp(i64_t lo, i64_t hi) const noexcept { return max(lo).min(hi); }

    constexpr i64_t saturating_add(i64_t o) const noexcept {
        // Check for overflow: if signs match and result has different sign
        nk_i64_t result = raw_ + o.raw_;
        if (o.raw_ > 0 && raw_ > NK_I64_MAX - o.raw_) return i64_t::finite_max();
        if (o.raw_ < 0 && raw_ < NK_I64_MIN - o.raw_) return i64_t::finite_min();
        return i64_t {result};
    }
    constexpr i64_t saturating_sub(i64_t o) const noexcept {
        nk_i64_t result = raw_ - o.raw_;
        if (o.raw_ < 0 && raw_ > NK_I64_MAX + o.raw_) return i64_t::finite_max();
        if (o.raw_ > 0 && raw_ < NK_I64_MIN + o.raw_) return i64_t::finite_min();
        return i64_t {result};
    }
};

/**
 *  @brief Unsigned 64-bit integer wrapper.
 *
 *  Provides strong type identity for `std::uint64_t`, compatible with NumKong
 *  kernels and `std::mdspan`.
 *
 *  Features:
 *  - Arithmetic operators (wrapping semantics)
 *  - Bitwise operators
 *  - Spaceship operator for comparisons
 *  - Integer-specific: min, max, clamp
 */
struct u64_t {
    // Core type aliases
    using raw_t = nk_u64_t;
    using signed_t = nk_i64_t;

    using reduce_add_result_t = u64_t; // `nk_reduce_add_u64` (no widening, already max)

    using sparse_intersect_kernel_t = void (*)(raw_t const *, raw_t const *, nk_size_t, nk_size_t, raw_t *,
                                               nk_size_t *);

    static constexpr nk_dtype_t dtype() noexcept { return nk_u64_k; }
    static constexpr char const *dtype_name() noexcept { return "u64"; }
    static constexpr unsigned bits_per_word() noexcept { return 64; }
    static constexpr unsigned bits_per_dimension() noexcept { return 64; }
    static constexpr unsigned bits_per_value() noexcept { return 64; }
    static constexpr bool is_integer() noexcept { return true; }
    static constexpr bool is_signed() noexcept { return false; }
    static constexpr bool is_complex() noexcept { return false; }
    static constexpr bool is_exact() noexcept { return true; }
    static constexpr bool has_infinity() noexcept { return false; }
    static constexpr bool has_nan() noexcept { return false; }

    raw_t raw_;

    constexpr u64_t() noexcept : raw_(0) {}
    constexpr u64_t(std::uint64_t v) noexcept : raw_(v) {}
    template <std::integral integral_type_>
    constexpr u64_t(integral_type_ v) noexcept : raw_(static_cast<raw_t>(v)) {}
    constexpr operator std::uint64_t() const noexcept { return raw_; }
    constexpr std::uint64_t raw() const noexcept { return raw_; }
    static constexpr u64_t from_raw(raw_t r) noexcept { return u64_t {r}; }

    static constexpr u64_t finite_min() noexcept { return u64_t {}; }
    static constexpr u64_t finite_max() noexcept { return u64_t {std::uint64_t(18446744073709551615ULL)}; }
    static constexpr u64_t zero() noexcept { return u64_t {}; }
    static constexpr u64_t one() noexcept { return u64_t {std::uint64_t(1)}; }

    constexpr u64_t operator+() const noexcept { return *this; }
    constexpr u64_t operator+(u64_t o) const noexcept { return u64_t {raw_ + o.raw_}; }
    constexpr u64_t operator-(u64_t o) const noexcept { return u64_t {raw_ - o.raw_}; }
    constexpr u64_t operator*(u64_t o) const noexcept { return u64_t {raw_ * o.raw_}; }
    constexpr u64_t operator/(u64_t o) const noexcept { return u64_t {raw_ / o.raw_}; }
    constexpr u64_t operator%(u64_t o) const noexcept { return u64_t {raw_ % o.raw_}; }

    constexpr u64_t &operator+=(u64_t o) noexcept {
        raw_ += o.raw_;
        return *this;
    }
    constexpr u64_t &operator-=(u64_t o) noexcept {
        raw_ -= o.raw_;
        return *this;
    }
    constexpr u64_t &operator*=(u64_t o) noexcept {
        raw_ *= o.raw_;
        return *this;
    }
    constexpr u64_t &operator/=(u64_t o) noexcept {
        raw_ /= o.raw_;
        return *this;
    }
    constexpr u64_t &operator%=(u64_t o) noexcept {
        raw_ %= o.raw_;
        return *this;
    }

    constexpr u64_t operator~() const noexcept { return u64_t {~raw_}; }
    constexpr u64_t operator&(u64_t o) const noexcept { return u64_t {raw_ & o.raw_}; }
    constexpr u64_t operator|(u64_t o) const noexcept { return u64_t {raw_ | o.raw_}; }
    constexpr u64_t operator^(u64_t o) const noexcept { return u64_t {raw_ ^ o.raw_}; }
    constexpr u64_t operator<<(int n) const noexcept { return u64_t {raw_ << n}; }
    constexpr u64_t operator>>(int n) const noexcept { return u64_t {raw_ >> n}; }

    constexpr u64_t &operator&=(u64_t o) noexcept {
        raw_ &= o.raw_;
        return *this;
    }
    constexpr u64_t &operator|=(u64_t o) noexcept {
        raw_ |= o.raw_;
        return *this;
    }
    constexpr u64_t &operator^=(u64_t o) noexcept {
        raw_ ^= o.raw_;
        return *this;
    }
    constexpr u64_t &operator<<=(int n) noexcept {
        raw_ <<= n;
        return *this;
    }
    constexpr u64_t &operator>>=(int n) noexcept {
        raw_ >>= n;
        return *this;
    }

    constexpr auto operator<=>(u64_t const &o) const noexcept = default;

    constexpr int total_cmp(u64_t o) const noexcept { return (raw_ > o.raw_) - (raw_ < o.raw_); }
    constexpr u64_t min(u64_t o) const noexcept { return raw_ < o.raw_ ? *this : o; }
    constexpr u64_t max(u64_t o) const noexcept { return raw_ > o.raw_ ? *this : o; }
    constexpr u64_t clamp(u64_t lo, u64_t hi) const noexcept { return max(lo).min(hi); }

    constexpr u64_t saturating_add(u64_t o) const noexcept {
        nk_u64_t result = raw_ + o.raw_;
        return result < raw_ ? u64_t::finite_max() : u64_t {result}; // overflow check
    }
    constexpr u64_t saturating_sub(u64_t o) const noexcept {
        return o.raw_ > raw_ ? u64_t::zero() : u64_t {raw_ - o.raw_};
    }
};

/**
 *  @brief Signed 16-bit integer wrapper.
 *
 *  Provides strong type identity for `std::int16_t`, compatible with NumKong
 *  kernels and `std::mdspan`.
 *
 *  Features:
 *  - Arithmetic operators (wrapping semantics)
 *  - Bitwise operators
 *  - Spaceship operator for comparisons
 *  - Integer-specific: abs, signum, min, max, clamp
 */
struct i16_t {
    // Core type aliases
    using raw_t = nk_i16_t;
    using unsigned_t = nk_u16_t;

    using reduce_add_result_t = i64_t; // `nk_reduce_add_i16` widened output

    static constexpr nk_dtype_t dtype() noexcept { return nk_i16_k; }
    static constexpr char const *dtype_name() noexcept { return "i16"; }
    static constexpr unsigned bits_per_word() noexcept { return 16; }
    static constexpr unsigned bits_per_dimension() noexcept { return 16; }
    static constexpr unsigned bits_per_value() noexcept { return 16; }
    static constexpr bool is_integer() noexcept { return true; }
    static constexpr bool is_signed() noexcept { return true; }
    static constexpr bool is_complex() noexcept { return false; }
    static constexpr bool is_exact() noexcept { return true; }
    static constexpr bool has_infinity() noexcept { return false; }
    static constexpr bool has_nan() noexcept { return false; }

    raw_t raw_;

    constexpr i16_t() noexcept : raw_(0) {}
    constexpr i16_t(std::int16_t v) noexcept : raw_(v) {}
    template <std::integral integral_type_>
    constexpr i16_t(integral_type_ v) noexcept : raw_(static_cast<raw_t>(v)) {}
    constexpr operator std::int16_t() const noexcept { return raw_; }
    constexpr std::int16_t raw() const noexcept { return raw_; }
    static constexpr i16_t from_raw(raw_t r) noexcept { return i16_t {r}; }

    static constexpr i16_t finite_min() noexcept { return i16_t {std::int16_t(-32768)}; }
    static constexpr i16_t finite_max() noexcept { return i16_t {std::int16_t(32767)}; }
    static constexpr i16_t zero() noexcept { return i16_t {}; }
    static constexpr i16_t one() noexcept { return i16_t {std::int16_t(1)}; }

    constexpr i16_t operator+() const noexcept { return *this; }
    constexpr i16_t operator-() const noexcept { return i16_t {static_cast<raw_t>(-raw_)}; }
    constexpr i16_t operator+(i16_t o) const noexcept { return i16_t {static_cast<raw_t>(raw_ + o.raw_)}; }
    constexpr i16_t operator-(i16_t o) const noexcept { return i16_t {static_cast<raw_t>(raw_ - o.raw_)}; }
    constexpr i16_t operator*(i16_t o) const noexcept { return i16_t {static_cast<raw_t>(raw_ * o.raw_)}; }
    constexpr i16_t operator/(i16_t o) const noexcept { return i16_t {static_cast<raw_t>(raw_ / o.raw_)}; }
    constexpr i16_t operator%(i16_t o) const noexcept { return i16_t {static_cast<raw_t>(raw_ % o.raw_)}; }

    constexpr i16_t &operator+=(i16_t o) noexcept {
        raw_ = static_cast<raw_t>(raw_ + o.raw_);
        return *this;
    }
    constexpr i16_t &operator-=(i16_t o) noexcept {
        raw_ = static_cast<raw_t>(raw_ - o.raw_);
        return *this;
    }
    constexpr i16_t &operator*=(i16_t o) noexcept {
        raw_ = static_cast<raw_t>(raw_ * o.raw_);
        return *this;
    }
    constexpr i16_t &operator/=(i16_t o) noexcept {
        raw_ = static_cast<raw_t>(raw_ / o.raw_);
        return *this;
    }
    constexpr i16_t &operator%=(i16_t o) noexcept {
        raw_ = static_cast<raw_t>(raw_ % o.raw_);
        return *this;
    }

    constexpr i16_t operator~() const noexcept { return i16_t {static_cast<raw_t>(~raw_)}; }
    constexpr i16_t operator&(i16_t o) const noexcept { return i16_t {static_cast<raw_t>(raw_ & o.raw_)}; }
    constexpr i16_t operator|(i16_t o) const noexcept { return i16_t {static_cast<raw_t>(raw_ | o.raw_)}; }
    constexpr i16_t operator^(i16_t o) const noexcept { return i16_t {static_cast<raw_t>(raw_ ^ o.raw_)}; }
    constexpr i16_t operator<<(int n) const noexcept { return i16_t {static_cast<raw_t>(raw_ << n)}; }
    constexpr i16_t operator>>(int n) const noexcept { return i16_t {static_cast<raw_t>(raw_ >> n)}; }

    constexpr i16_t &operator&=(i16_t o) noexcept {
        raw_ = static_cast<raw_t>(raw_ & o.raw_);
        return *this;
    }
    constexpr i16_t &operator|=(i16_t o) noexcept {
        raw_ = static_cast<raw_t>(raw_ | o.raw_);
        return *this;
    }
    constexpr i16_t &operator^=(i16_t o) noexcept {
        raw_ = static_cast<raw_t>(raw_ ^ o.raw_);
        return *this;
    }
    constexpr i16_t &operator<<=(int n) noexcept {
        raw_ = static_cast<raw_t>(raw_ << n);
        return *this;
    }
    constexpr i16_t &operator>>=(int n) noexcept {
        raw_ = static_cast<raw_t>(raw_ >> n);
        return *this;
    }

    constexpr auto operator<=>(i16_t const &o) const noexcept = default;

    constexpr i16_t abs() const noexcept { return raw_ < 0 ? i16_t {static_cast<raw_t>(-raw_)} : *this; }
    constexpr i16_t signum() const noexcept {
        if (raw_ > 0) return i16_t {std::int16_t(1)};
        if (raw_ < 0) return i16_t {std::int16_t(-1)};
        return i16_t {};
    }
    constexpr int total_cmp(i16_t o) const noexcept { return (raw_ > o.raw_) - (raw_ < o.raw_); }
    constexpr i16_t min(i16_t o) const noexcept { return raw_ < o.raw_ ? *this : o; }
    constexpr i16_t max(i16_t o) const noexcept { return raw_ > o.raw_ ? *this : o; }
    constexpr i16_t clamp(i16_t lo, i16_t hi) const noexcept { return max(lo).min(hi); }

    constexpr i16_t saturating_add(i16_t o) const noexcept {
        nk_i32_t result = nk_i32_t(raw_) + nk_i32_t(o.raw_);
        if (result > NK_I16_MAX) return i16_t::finite_max();
        if (result < NK_I16_MIN) return i16_t::finite_min();
        return i16_t {static_cast<raw_t>(result)};
    }
    constexpr i16_t saturating_sub(i16_t o) const noexcept {
        nk_i32_t result = nk_i32_t(raw_) - nk_i32_t(o.raw_);
        if (result > NK_I16_MAX) return i16_t::finite_max();
        if (result < NK_I16_MIN) return i16_t::finite_min();
        return i16_t {static_cast<raw_t>(result)};
    }
};

/**
 *  @brief Unsigned 16-bit integer wrapper.
 *
 *  Provides strong type identity for `std::uint16_t`, compatible with NumKong
 *  kernels and `std::mdspan`.
 *
 *  Features:
 *  - Arithmetic operators (wrapping semantics)
 *  - Bitwise operators
 *  - Spaceship operator for comparisons
 *  - Integer-specific: min, max, clamp
 */
struct u16_t {
    // Core type aliases
    using raw_t = nk_u16_t;
    using signed_t = nk_i16_t;

    using reduce_add_result_t = u64_t; // `nk_reduce_add_u16` widened output
    using jaccard_result_t = f32_t;    // `nk_jaccard_u16` output

    using sparse_intersect_kernel_t = void (*)(raw_t const *, raw_t const *, nk_size_t, nk_size_t, raw_t *,
                                               nk_size_t *);

    static constexpr nk_dtype_t dtype() noexcept { return nk_u16_k; }
    static constexpr char const *dtype_name() noexcept { return "u16"; }
    static constexpr unsigned bits_per_word() noexcept { return 16; }
    static constexpr unsigned bits_per_dimension() noexcept { return 16; }
    static constexpr unsigned bits_per_value() noexcept { return 16; }
    static constexpr bool is_integer() noexcept { return true; }
    static constexpr bool is_signed() noexcept { return false; }
    static constexpr bool is_complex() noexcept { return false; }
    static constexpr bool is_exact() noexcept { return true; }
    static constexpr bool has_infinity() noexcept { return false; }
    static constexpr bool has_nan() noexcept { return false; }

    raw_t raw_;

    constexpr u16_t() noexcept : raw_(0) {}
    constexpr u16_t(std::uint16_t v) noexcept : raw_(v) {}
    template <std::integral integral_type_>
    constexpr u16_t(integral_type_ v) noexcept : raw_(static_cast<raw_t>(v)) {}
    constexpr operator std::uint16_t() const noexcept { return raw_; }
    constexpr std::uint16_t raw() const noexcept { return raw_; }
    static constexpr u16_t from_raw(raw_t r) noexcept { return u16_t {r}; }

    static constexpr u16_t finite_min() noexcept { return u16_t {}; }
    static constexpr u16_t finite_max() noexcept { return u16_t {std::uint16_t(65535)}; }
    static constexpr u16_t zero() noexcept { return u16_t {}; }
    static constexpr u16_t one() noexcept { return u16_t {std::uint16_t(1)}; }

    constexpr u16_t operator+() const noexcept { return *this; }
    constexpr u16_t operator+(u16_t o) const noexcept { return u16_t {static_cast<raw_t>(raw_ + o.raw_)}; }
    constexpr u16_t operator-(u16_t o) const noexcept { return u16_t {static_cast<raw_t>(raw_ - o.raw_)}; }
    constexpr u16_t operator*(u16_t o) const noexcept { return u16_t {static_cast<raw_t>(raw_ * o.raw_)}; }
    constexpr u16_t operator/(u16_t o) const noexcept { return u16_t {static_cast<raw_t>(raw_ / o.raw_)}; }
    constexpr u16_t operator%(u16_t o) const noexcept { return u16_t {static_cast<raw_t>(raw_ % o.raw_)}; }

    constexpr u16_t &operator+=(u16_t o) noexcept {
        raw_ = static_cast<raw_t>(raw_ + o.raw_);
        return *this;
    }
    constexpr u16_t &operator-=(u16_t o) noexcept {
        raw_ = static_cast<raw_t>(raw_ - o.raw_);
        return *this;
    }
    constexpr u16_t &operator*=(u16_t o) noexcept {
        raw_ = static_cast<raw_t>(raw_ * o.raw_);
        return *this;
    }
    constexpr u16_t &operator/=(u16_t o) noexcept {
        raw_ = static_cast<raw_t>(raw_ / o.raw_);
        return *this;
    }
    constexpr u16_t &operator%=(u16_t o) noexcept {
        raw_ = static_cast<raw_t>(raw_ % o.raw_);
        return *this;
    }

    constexpr u16_t operator~() const noexcept { return u16_t {static_cast<raw_t>(~raw_)}; }
    constexpr u16_t operator&(u16_t o) const noexcept { return u16_t {static_cast<raw_t>(raw_ & o.raw_)}; }
    constexpr u16_t operator|(u16_t o) const noexcept { return u16_t {static_cast<raw_t>(raw_ | o.raw_)}; }
    constexpr u16_t operator^(u16_t o) const noexcept { return u16_t {static_cast<raw_t>(raw_ ^ o.raw_)}; }
    constexpr u16_t operator<<(int n) const noexcept { return u16_t {static_cast<raw_t>(raw_ << n)}; }
    constexpr u16_t operator>>(int n) const noexcept { return u16_t {static_cast<raw_t>(raw_ >> n)}; }

    constexpr u16_t &operator&=(u16_t o) noexcept {
        raw_ = static_cast<raw_t>(raw_ & o.raw_);
        return *this;
    }
    constexpr u16_t &operator|=(u16_t o) noexcept {
        raw_ = static_cast<raw_t>(raw_ | o.raw_);
        return *this;
    }
    constexpr u16_t &operator^=(u16_t o) noexcept {
        raw_ = static_cast<raw_t>(raw_ ^ o.raw_);
        return *this;
    }
    constexpr u16_t &operator<<=(int n) noexcept {
        raw_ = static_cast<raw_t>(raw_ << n);
        return *this;
    }
    constexpr u16_t &operator>>=(int n) noexcept {
        raw_ = static_cast<raw_t>(raw_ >> n);
        return *this;
    }

    constexpr auto operator<=>(u16_t const &o) const noexcept = default;

    constexpr int total_cmp(u16_t o) const noexcept { return (raw_ > o.raw_) - (raw_ < o.raw_); }
    constexpr u16_t min(u16_t o) const noexcept { return raw_ < o.raw_ ? *this : o; }
    constexpr u16_t max(u16_t o) const noexcept { return raw_ > o.raw_ ? *this : o; }
    constexpr u16_t clamp(u16_t lo, u16_t hi) const noexcept { return max(lo).min(hi); }

    constexpr u16_t saturating_add(u16_t o) const noexcept {
        nk_u32_t result = nk_u32_t(raw_) + nk_u32_t(o.raw_);
        return result > NK_U16_MAX ? u16_t::finite_max() : u16_t {static_cast<raw_t>(result)};
    }
    constexpr u16_t saturating_sub(u16_t o) const noexcept {
        return o.raw_ > raw_ ? u16_t::zero() : u16_t {static_cast<raw_t>(raw_ - o.raw_)};
    }
};

/**
 *  @brief Packed 8-bit bit-vector wrapper (8 booleans in one byte).
 *
 *  Storage/transport type for binary data. Provides bitwise operations
 *  and population count for Hamming distance and Jaccard similarity.
 *
 *  Features:
 *  - Bitwise operators (&, |, ^, ~)
 *  - Individual bit access
 *  - Population count (popcount)
 *  - Hamming distance and intersection/union helpers
 */
struct u1x8_t {
    // Core type aliases
    using raw_t = nk_u1x8_t;
    using dot_result_t = u32_t;
    using hamming_result_t = u32_t;
    using jaccard_result_t = f32_t;

    // Kernel function pointer types (note: n is in bits, not bytes)
    using dots_symmetric_kernel_t = void (*)(raw_t const *, nk_size_t, nk_size_t, nk_size_t, nk_u32_t *, nk_size_t,
                                             nk_size_t, nk_size_t);
    using hamming_kernel_t = void (*)(raw_t const *, raw_t const *, nk_size_t, nk_u32_t *);
    using jaccard_kernel_t = void (*)(raw_t const *, raw_t const *, nk_size_t, nk_f32_t *);
    using dots_packed_size_kernel_t = nk_size_t (*)(nk_size_t, nk_size_t);
    using dots_pack_kernel_t = void (*)(raw_t const *, nk_size_t, nk_size_t, nk_size_t, void *);
    using dots_packed_kernel_t = void (*)(raw_t const *, void const *, nk_u32_t *, nk_size_t, nk_size_t, nk_size_t,
                                          nk_size_t, nk_size_t);
    using hammings_packed_size_kernel_t = nk_size_t (*)(nk_size_t, nk_size_t);
    using hammings_pack_kernel_t = void (*)(raw_t const *, nk_size_t, nk_size_t, nk_size_t, void *);
    using hammings_packed_kernel_t = void (*)(raw_t const *, void const *, nk_u32_t *, nk_size_t, nk_size_t, nk_size_t,
                                              nk_size_t, nk_size_t);
    using hammings_symmetric_kernel_t = void (*)(raw_t const *, nk_size_t, nk_size_t, nk_size_t, nk_u32_t *, nk_size_t,
                                                 nk_size_t, nk_size_t);

    static constexpr nk_dtype_t dtype() noexcept { return nk_u1_k; }
    static constexpr char const *dtype_name() noexcept { return "u1x8"; }
    static constexpr unsigned bits_per_word() noexcept { return 8; }
    static constexpr unsigned bits_per_dimension() noexcept { return 1; } // Each bit is one dimension
    static constexpr unsigned bits_per_value() noexcept { return 8; }
    static constexpr unsigned elements() noexcept { return 8; }
    static constexpr bool is_integer() noexcept { return true; }
    static constexpr bool is_signed() noexcept { return false; }
    static constexpr bool is_complex() noexcept { return false; }
    static constexpr bool is_exact() noexcept { return true; }
    static constexpr bool has_infinity() noexcept { return false; }
    static constexpr bool has_nan() noexcept { return false; }

    raw_t raw_;

    constexpr u1x8_t() noexcept : raw_(0) {}
    constexpr explicit u1x8_t(raw_t v) noexcept : raw_(v) {}
    constexpr raw_t raw() const noexcept { return raw_; }
    static constexpr u1x8_t from_raw(raw_t r) noexcept { return u1x8_t {r}; }

    static constexpr u1x8_t zero() noexcept { return u1x8_t {}; }
    static constexpr u1x8_t all_ones() noexcept { return u1x8_t {static_cast<raw_t>(0xFF)}; }

    constexpr bool bit(unsigned i) const noexcept { return (raw_ >> i) & 1; }
    constexpr bool operator[](unsigned i) const noexcept { return bit(i); }

    constexpr u1x8_t set_bit(unsigned i) const noexcept { return u1x8_t {static_cast<raw_t>(raw_ | (1 << i))}; }
    constexpr u1x8_t clear_bit(unsigned i) const noexcept { return u1x8_t {static_cast<raw_t>(raw_ & ~(1 << i))}; }
    constexpr u1x8_t toggle_bit(unsigned i) const noexcept { return u1x8_t {static_cast<raw_t>(raw_ ^ (1 << i))}; }

    constexpr u1x8_t operator~() const noexcept { return u1x8_t {static_cast<raw_t>(~raw_)}; }
    constexpr u1x8_t operator&(u1x8_t o) const noexcept { return u1x8_t {static_cast<raw_t>(raw_ & o.raw_)}; }
    constexpr u1x8_t operator|(u1x8_t o) const noexcept { return u1x8_t {static_cast<raw_t>(raw_ | o.raw_)}; }
    constexpr u1x8_t operator^(u1x8_t o) const noexcept { return u1x8_t {static_cast<raw_t>(raw_ ^ o.raw_)}; }

    constexpr u1x8_t &operator&=(u1x8_t o) noexcept {
        raw_ = static_cast<raw_t>(raw_ & o.raw_);
        return *this;
    }
    constexpr u1x8_t &operator|=(u1x8_t o) noexcept {
        raw_ = static_cast<raw_t>(raw_ | o.raw_);
        return *this;
    }
    constexpr u1x8_t &operator^=(u1x8_t o) noexcept {
        raw_ = static_cast<raw_t>(raw_ ^ o.raw_);
        return *this;
    }

    constexpr auto operator<=>(u1x8_t const &o) const noexcept = default;
    constexpr unsigned popcount() const noexcept { return std::popcount(raw_); }
    constexpr unsigned hamming(u1x8_t o) const noexcept { return (*this ^ o).popcount(); }
    constexpr unsigned intersection(u1x8_t o) const noexcept { return (*this & o).popcount(); }
    constexpr unsigned union_size(u1x8_t o) const noexcept { return (*this | o).popcount(); }

    constexpr bool any() const noexcept { return raw_ != 0; }
    constexpr bool all() const noexcept { return raw_ == 0xFF; }
    constexpr bool none() const noexcept { return raw_ == 0; }
};

/**
 *  @brief Packed 4-bit signed integer pair (2 x i4 in one byte).
 *
 *  Storage/transport type for 4-bit quantized data. Elements are sign-extended
 *  to i8 for arithmetic. Layout: [high:4][low:4].
 *
 *  Features:
 *  - Element access (low/high nibbles, sign-extended to i8)
 *  - Dot product (widened to i32)
 *  - Element-wise operations returning widened types
 */
struct i4x2_t {
    // Core type aliases
    using raw_t = nk_i4x2_t;
    using element_t = nk_i8_t; // Elements widen to i8 for arithmetic

    using dot_result_t = i32_t;
    using sqeuclidean_result_t = u32_t;
    using angular_result_t = f32_t;

    using dots_symmetric_kernel_t = void (*)(raw_t const *, nk_size_t, nk_size_t, nk_size_t, nk_i32_t *, nk_size_t,
                                             nk_size_t, nk_size_t);
    using dot_kernel_t = void (*)(raw_t const *, raw_t const *, nk_size_t, nk_i32_t *);
    using sqeuclidean_kernel_t = void (*)(raw_t const *, raw_t const *, nk_size_t, nk_u32_t *);
    using angular_kernel_t = void (*)(raw_t const *, raw_t const *, nk_size_t, nk_f32_t *);
    using dots_packed_size_kernel_t = nk_size_t (*)(nk_size_t, nk_size_t);
    using dots_pack_kernel_t = void (*)(raw_t const *, nk_size_t, nk_size_t, nk_size_t, void *);
    using dots_packed_kernel_t = void (*)(raw_t const *, void const *, nk_i32_t *, nk_size_t, nk_size_t, nk_size_t,
                                          nk_size_t, nk_size_t);

    static constexpr nk_dtype_t dtype() noexcept { return nk_i4_k; }
    static constexpr char const *dtype_name() noexcept { return "i4x2"; }
    static constexpr unsigned bits_per_word() noexcept { return 8; }
    static constexpr unsigned bits_per_dimension() noexcept { return 4; } // Each nibble is one dimension
    static constexpr unsigned bits_per_value() noexcept { return 8; }
    static constexpr unsigned elements() noexcept { return 2; }
    static constexpr unsigned element_bits() noexcept { return 4; }
    static constexpr bool is_integer() noexcept { return true; }
    static constexpr bool is_signed() noexcept { return true; }
    static constexpr bool is_complex() noexcept { return false; }
    static constexpr bool is_exact() noexcept { return true; }
    static constexpr bool has_infinity() noexcept { return false; }
    static constexpr bool has_nan() noexcept { return false; }

    static constexpr element_t element_min() noexcept { return -8; }
    static constexpr element_t element_max() noexcept { return 7; }

    raw_t raw_;

    static constexpr element_t sign_extend(nk_u8_t nibble) noexcept {
        return static_cast<element_t>(static_cast<nk_i8_t>((nibble ^ 8) - 8));
    }

    constexpr i4x2_t() noexcept : raw_(0) {}
    constexpr explicit i4x2_t(raw_t v) noexcept : raw_(v) {}
    constexpr i4x2_t(element_t low, element_t high) noexcept
        : raw_(static_cast<raw_t>(((high & 0x0F) << 4) | (low & 0x0F))) {}
    constexpr raw_t raw() const noexcept { return raw_; }
    static constexpr i4x2_t from_raw(raw_t r) noexcept { return i4x2_t {r}; }

    static constexpr i4x2_t zero() noexcept { return i4x2_t {}; }

    constexpr element_t low() const noexcept { return sign_extend(raw_ & 0x0F); }
    constexpr element_t high() const noexcept { return sign_extend((raw_ >> 4) & 0x0F); }

    constexpr std::pair<element_t, element_t> widening_add(i4x2_t o) const noexcept {
        return {static_cast<element_t>(low() + o.low()), static_cast<element_t>(high() + o.high())};
    }

    constexpr std::pair<element_t, element_t> widening_sub(i4x2_t o) const noexcept {
        return {static_cast<element_t>(low() - o.low()), static_cast<element_t>(high() - o.high())};
    }

    constexpr std::pair<element_t, element_t> widening_mul(i4x2_t o) const noexcept {
        return {static_cast<element_t>(low() * o.low()), static_cast<element_t>(high() * o.high())};
    }

    constexpr i4x2_t saturating_add(i4x2_t o) const noexcept {
        auto clamp = [](int v) -> element_t {
            if (v < -8) return -8;
            if (v > 7) return 7;
            return static_cast<element_t>(v);
        };
        return i4x2_t {clamp(low() + o.low()), clamp(high() + o.high())};
    }

    constexpr i4x2_t saturating_sub(i4x2_t o) const noexcept {
        auto clamp = [](int v) -> element_t {
            if (v < -8) return -8;
            if (v > 7) return 7;
            return static_cast<element_t>(v);
        };
        return i4x2_t {clamp(low() - o.low()), clamp(high() - o.high())};
    }

    constexpr i4x2_t wrapping_add(i4x2_t o) const noexcept {
        return i4x2_t {static_cast<element_t>((low() + o.low()) & 0x0F),
                       static_cast<element_t>((high() + o.high()) & 0x0F)};
    }

    constexpr auto operator<=>(i4x2_t const &o) const noexcept = default;
};

/**
 *  @brief Packed 4-bit unsigned integer pair (2 x u4 in one byte).
 *
 *  Storage/transport type for 4-bit quantized data. Elements are zero-extended
 *  to u8 for arithmetic. Layout: [high:4][low:4].
 *
 *  Features:
 *  - Element access (low/high nibbles as u8)
 *  - Dot product (widened to u32)
 *  - Element-wise operations returning widened types
 */
struct u4x2_t {
    // Core type aliases
    using raw_t = nk_u4x2_t;
    using element_t = nk_u8_t; // Elements widen to u8 for arithmetic

    using dot_result_t = u32_t;
    using sqeuclidean_result_t = u32_t;
    using angular_result_t = f32_t;
    using dots_symmetric_kernel_t = void (*)(raw_t const *, nk_size_t, nk_size_t, nk_size_t, nk_u32_t *, nk_size_t,
                                             nk_size_t, nk_size_t);

    using dot_kernel_t = void (*)(raw_t const *, raw_t const *, nk_size_t, nk_u32_t *);
    using sqeuclidean_kernel_t = void (*)(raw_t const *, raw_t const *, nk_size_t, nk_u32_t *);
    using angular_kernel_t = void (*)(raw_t const *, raw_t const *, nk_size_t, nk_f32_t *);
    using dots_packed_size_kernel_t = nk_size_t (*)(nk_size_t, nk_size_t);
    using dots_pack_kernel_t = void (*)(raw_t const *, nk_size_t, nk_size_t, nk_size_t, void *);
    using dots_packed_kernel_t = void (*)(raw_t const *, void const *, nk_u32_t *, nk_size_t, nk_size_t, nk_size_t,
                                          nk_size_t, nk_size_t);

    static constexpr nk_dtype_t dtype() noexcept { return nk_u4_k; }
    static constexpr char const *dtype_name() noexcept { return "u4x2"; }
    static constexpr unsigned bits_per_word() noexcept { return 8; }
    static constexpr unsigned bits_per_dimension() noexcept { return 4; } // Each nibble is one dimension
    static constexpr unsigned bits_per_value() noexcept { return 8; }
    static constexpr unsigned elements() noexcept { return 2; }
    static constexpr unsigned element_bits() noexcept { return 4; }
    static constexpr bool is_integer() noexcept { return true; }
    static constexpr bool is_signed() noexcept { return false; }
    static constexpr bool is_complex() noexcept { return false; }
    static constexpr bool is_exact() noexcept { return true; }
    static constexpr bool has_infinity() noexcept { return false; }
    static constexpr bool has_nan() noexcept { return false; }

    static constexpr element_t element_min() noexcept { return 0; }
    static constexpr element_t element_max() noexcept { return 15; }

    raw_t raw_;

    constexpr u4x2_t() noexcept : raw_(0) {}
    constexpr explicit u4x2_t(raw_t v) noexcept : raw_(v) {}
    constexpr u4x2_t(element_t low, element_t high) noexcept
        : raw_(static_cast<raw_t>(((high & 0x0F) << 4) | (low & 0x0F))) {}
    constexpr raw_t raw() const noexcept { return raw_; }
    static constexpr u4x2_t from_raw(raw_t r) noexcept { return u4x2_t {r}; }

    static constexpr u4x2_t zero() noexcept { return u4x2_t {}; }
    static constexpr u4x2_t finite_max() noexcept { return u4x2_t {15, 15}; }
    static constexpr u4x2_t finite_min() noexcept { return u4x2_t {}; }

    constexpr element_t low() const noexcept { return raw_ & 0x0F; }
    constexpr element_t high() const noexcept { return (raw_ >> 4) & 0x0F; }

    constexpr std::pair<element_t, element_t> widening_add(u4x2_t o) const noexcept {
        return {static_cast<element_t>(low() + o.low()), static_cast<element_t>(high() + o.high())};
    }

    constexpr std::pair<nk_i8_t, nk_i8_t> widening_sub(u4x2_t o) const noexcept {
        return {static_cast<nk_i8_t>(low() - o.low()), static_cast<nk_i8_t>(high() - o.high())};
    }

    constexpr std::pair<element_t, element_t> widening_mul(u4x2_t o) const noexcept {
        return {static_cast<element_t>(low() * o.low()), static_cast<element_t>(high() * o.high())};
    }

    constexpr u4x2_t saturating_add(u4x2_t o) const noexcept {
        auto clamp = [](unsigned v) -> element_t { return v > 15 ? 15 : static_cast<element_t>(v); };
        return u4x2_t {clamp(low() + o.low()), clamp(high() + o.high())};
    }

    constexpr u4x2_t saturating_sub(u4x2_t o) const noexcept {
        auto clamp = [](int v) -> element_t { return v < 0 ? 0 : static_cast<element_t>(v); };
        return u4x2_t {clamp(static_cast<int>(low()) - static_cast<int>(o.low())),
                       clamp(static_cast<int>(high()) - static_cast<int>(o.high()))};
    }

    constexpr u4x2_t wrapping_add(u4x2_t o) const noexcept {
        return u4x2_t {static_cast<element_t>((low() + o.low()) & 0x0F),
                       static_cast<element_t>((high() + o.high()) & 0x0F)};
    }

    constexpr auto operator<=>(u4x2_t const &o) const noexcept = default;
};

#pragma region - Enum Conversion

/**
 *  @brief Maps `nk_dtype_t` enum values to their corresponding C++ wrapper types.
 *  @tparam dtype_ The dtype enum value.
 */
template <nk_dtype_t dtype_>
struct type_for;

template <>
struct type_for<nk_f32_k> {
    using type = f32_t;
};
template <>
struct type_for<nk_f64_k> {
    using type = f64_t;
};
template <>
struct type_for<nk_f16_k> {
    using type = f16_t;
};
template <>
struct type_for<nk_bf16_k> {
    using type = bf16_t;
};
template <>
struct type_for<nk_e4m3_k> {
    using type = e4m3_t;
};
template <>
struct type_for<nk_e5m2_k> {
    using type = e5m2_t;
};
template <>
struct type_for<nk_e2m3_k> {
    using type = e2m3_t;
};
template <>
struct type_for<nk_e3m2_k> {
    using type = e3m2_t;
};
template <>
struct type_for<nk_f32c_k> {
    using type = f32c_t;
};
template <>
struct type_for<nk_f64c_k> {
    using type = f64c_t;
};
template <>
struct type_for<nk_f16c_k> {
    using type = f16c_t;
};
template <>
struct type_for<nk_bf16c_k> {
    using type = bf16c_t;
};
template <>
struct type_for<nk_i8_k> {
    using type = i8_t;
};
template <>
struct type_for<nk_i16_k> {
    using type = i16_t;
};
template <>
struct type_for<nk_i32_k> {
    using type = i32_t;
};
template <>
struct type_for<nk_i64_k> {
    using type = i64_t;
};
template <>
struct type_for<nk_u8_k> {
    using type = u8_t;
};
template <>
struct type_for<nk_u16_k> {
    using type = u16_t;
};
template <>
struct type_for<nk_u32_k> {
    using type = u32_t;
};
template <>
struct type_for<nk_u64_k> {
    using type = u64_t;
};
template <>
struct type_for<nk_u1_k> {
    using type = u1x8_t;
};
template <>
struct type_for<nk_i4_k> {
    using type = i4x2_t;
};
template <>
struct type_for<nk_u4_k> {
    using type = u4x2_t;
};

#pragma endregion - Enum Conversion

#pragma region - Numeric Limits

/** @brief Detect NumKong wrapper types with required static members. */
template <typename scalar_type_>
constexpr bool is_numeric_class() noexcept {
    return requires {
        { scalar_type_::dtype() } -> std::same_as<nk_dtype_t>;
    };
}

/** @brief Check if a type is an integer type. */
template <typename scalar_type_>
constexpr bool is_integer() noexcept {
    if constexpr (is_numeric_class<scalar_type_>()) return scalar_type_::is_integer();
    else return std::is_integral_v<scalar_type_>;
}

template <typename scalar_type_>
struct is_std_complex_sfinae_ : std::false_type {};

template <typename scalar_type_>
struct is_std_complex_sfinae_<std::complex<scalar_type_>> : std::true_type {};

template <typename scalar_type_>
constexpr bool is_std_complex_() noexcept {
    return is_std_complex_sfinae_<scalar_type_>::value;
}

/** @brief Check if a type is an complex type - STL or NumKong. */
template <typename scalar_type_>
constexpr bool is_complex() noexcept {
    if constexpr (is_numeric_class<scalar_type_>()) return scalar_type_::is_complex();
    else return is_std_complex_<scalar_type_>();
}

/** @brief Check if a type is an complex type - STL or NumKong. */
template <typename scalar_type_>
constexpr bool is_signed() noexcept {
    if constexpr (is_numeric_class<scalar_type_>()) return scalar_type_::is_signed();
    else if constexpr (is_complex<scalar_type_>()) return is_signed<scalar_type_::value_type>();
    else return std::is_signed<scalar_type_>::value;
}

/** @brief Get the maximum representable value for a type. */
template <typename scalar_type_>
constexpr auto finite_max() noexcept {
    if constexpr (is_numeric_class<scalar_type_>()) return scalar_type_::finite_max();
    else return std::numeric_limits<scalar_type_>::max();
}

/** @brief Get the lowest representable value for a type. */
template <typename scalar_type_>
constexpr auto finite_min() noexcept {
    if constexpr (is_numeric_class<scalar_type_>()) return scalar_type_::finite_min();
    return std::numeric_limits<scalar_type_>::lowest();
}

/** @brief Bits per value. For complex types matches value size. */
template <typename scalar_type_>
constexpr unsigned bits_per_value() noexcept {
    if constexpr (is_numeric_class<scalar_type_>()) return scalar_type_::bits_per_value();
    else return sizeof(scalar_type_) * NK_BITS_PER_BYTE;
}

/** @brief Bits per value. For complex types matches value size. */
template <typename scalar_type_>
constexpr unsigned bits_per_dimension() noexcept {
    if constexpr (is_numeric_class<scalar_type_>()) return scalar_type_::bits_per_dimension();
    else return sizeof(scalar_type_) * NK_BITS_PER_BYTE;
}

/** @brief Dimensions per container value. For normal types = 1, for sub-byte packed types > 1. */
template <typename scalar_type_>
constexpr unsigned dimensions_per_value() noexcept {
    return bits_per_value<scalar_type_>() / bits_per_dimension<scalar_type_>();
}

/**
 *  @brief Extract the word type from a value type.
 *
 *  For complex types (f32c_t, f64c_t, etc.), returns the component type (f32_t, f64_t).
 *  For all other types, returns the type itself.
 */
template <typename value_type_, typename = void>
struct word_type {
    using type = value_type_;
};

template <typename value_type_>
struct word_type<value_type_, std::void_t<typename value_type_::component_t>> {
    using type = typename value_type_::component_t;
};

template <typename scalar_type_>
struct word_type<std::complex<scalar_type_>> {
    using type = scalar_type_;
};

/**
 *  @brief Extract the raw C-level type.
 */
template <typename value_type_, typename = void>
struct raw_pod_type {
    using type = value_type_;
};

template <typename value_type_>
struct raw_pod_type<value_type_, std::void_t<typename value_type_::raw_t>> {
    using type = typename value_type_::raw_t;
};

/**
 *  @brief Ceiling division: (n + divisor - 1) / divisor (compile-time divisor).
 */
template <std::size_t divisor_>
constexpr std::size_t divide_round_up(std::size_t n) noexcept {
    return (n + divisor_ - 1) / divisor_;
}

/**
 *  @brief Ceiling division: (n + divisor - 1) / divisor (runtime divisor).
 */
constexpr std::size_t divide_round_up(std::size_t n, std::size_t divisor) noexcept {
    return (n + divisor - 1) / divisor;
}

/**
 *  @brief Round up to next multiple: ((n + multiple - 1) / multiple) * multiple.
 */
template <std::size_t multiple_>
constexpr std::size_t round_up_to_multiple(std::size_t n) {
    return divide_round_up<multiple_>(n) * multiple_;
}

#pragma endregion - Numeric Limits

#pragma region - SIMD Dispatch Helpers

/** @brief Controls whether template wrappers dispatch to SIMD C kernels. */
enum allow_simd_t {
    prefer_simd_k = 0,
    no_simd_k = 1,
};

/** @brief FMA helper template for baseline dot-product implementations. */
template <typename in_type_, typename accumulator_type_>
    requires(in_type_::bits_per_value() >= NK_BITS_PER_BYTE)
inline accumulator_type_ fused_multiply_add(accumulator_type_ acc, in_type_ a, in_type_ b) noexcept {
    return acc + static_cast<accumulator_type_>(a) * static_cast<accumulator_type_>(b);
}

/** @brief FMA helper template for baseline conjugate complex dot-product implementations. */
template <typename in_type_, typename accumulator_type_>
    requires(in_type_::bits_per_value() >= NK_BITS_PER_BYTE)
inline accumulator_type_ fused_conjugate_multiply_add(accumulator_type_ acc, in_type_ a, in_type_ b) noexcept {
    return acc + static_cast<accumulator_type_>(a.conj()) * static_cast<accumulator_type_>(b);
}

/** @brief Fused addition of squared differences for baseline L2 implementations. */
template <typename in_type_, typename accumulator_type_>
    requires(in_type_::bits_per_value() >= NK_BITS_PER_BYTE)
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

#pragma endregion - SIMD Dispatch Helpers

#pragma region - f118_t Mixed Operators

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

#pragma endregion - f118_t Mixed Operators

} // namespace ashvardanian::numkong

#endif // NK_TYPES_HPP

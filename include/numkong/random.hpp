/**
 *  @brief Random number generation utilities for NumKong types.
 *  @file include/numkong/random.hpp
 *  @author Ash Vardanian
 *  @date February 5, 2026
 *
 *  Lightweight header with random fill functions for testing and benchmarking.
 *  Only depends on types.hpp to minimize compilation overhead.
 */
#ifndef NK_RANDOM_HPP
#define NK_RANDOM_HPP

#include <random>

#include "numkong/types.hpp"

namespace ashvardanian::numkong {

/**
 *  @brief Fill array with uniform random values.
 *
 *  @tparam value_type_ A NumKong wrapper type (e.g., f32_t, f64_t, i32_t).
 *  @tparam generator_type_ A random number generator type (e.g., std::mt19937_64).
 *  @param generator The random number generator.
 *  @param values_ptr Pointer to the array to fill.
 *  @param values_count Number of storage values to fill (not dimensions for sub-byte types).
 */
template <typename value_type_, typename generator_type_>
void fill_uniform(generator_type_ &generator, value_type_ *values_ptr, std::size_t values_count) noexcept {

    // Packed types (u1x8, u4x2, i4x2) need special handling - just fill with random bytes
    if constexpr (std::is_same_v<value_type_, u1x8_t> || std::is_same_v<value_type_, u4x2_t> ||
                  std::is_same_v<value_type_, i4x2_t>) {
        std::uniform_int_distribution<std::uint32_t> distribution(0, 255);
        for (std::size_t i = 0; i < values_count; ++i)
            values_ptr[i].raw_ = static_cast<std::uint8_t>(distribution(generator));
    }
    // Integer distribution types aren't always defined
    else if constexpr (is_integer<value_type_>()) {
        using small_integer_t = std::conditional_t<is_signed<value_type_>(), std::int32_t, std::uint32_t>;
        using large_integer_t = std::conditional_t<is_signed<value_type_>(), std::int64_t, std::uint64_t>;
        using distribution_integer_t = std::conditional_t<sizeof(value_type_) <= 4, small_integer_t, large_integer_t>;
        std::uniform_int_distribution<distribution_integer_t> distribution(finite_min<distribution_integer_t>(),
                                                                           finite_max<distribution_integer_t>());
        for (std::size_t i = 0; i < values_count; ++i)
            values_ptr[i] = static_cast<value_type_>(distribution(generator));
    }
    // Complex types need both real and imaginary parts filled
    else if constexpr (is_complex<value_type_>()) {
        using component_t = typename value_type_::component_t;
        using distribution_float_t = std::conditional_t<sizeof(component_t) <= 4, float, double>;
        std::uniform_real_distribution<distribution_float_t> distribution(-10.0, 10.0);
        for (std::size_t i = 0; i < values_count; ++i) {
            auto real = distribution(generator), imag = distribution(generator);
            values_ptr[i] = value_type_(component_t(static_cast<distribution_float_t>(real)),
                                        component_t(static_cast<distribution_float_t>(imag)));
        }
    }
    // Floats and other types use a fixed range for better numerical stability
    else {
        using distribution_float_t = std::conditional_t<sizeof(value_type_) <= 4, float, double>;
        std::uniform_real_distribution<distribution_float_t> distribution(-10.0, 10.0);
        for (std::size_t i = 0; i < values_count; ++i)
            values_ptr[i] = static_cast<value_type_>(distribution(generator));
    }
}

/**
 *  @brief Fill array with uniform random values within specified range.
 */
template <typename value_type_, typename generator_type_>
void fill_uniform(generator_type_ &generator, value_type_ *values_ptr, std::size_t values_count, value_type_ min_val,
                  value_type_ max_val) noexcept {

    if constexpr (is_integer<value_type_>()) {
        using small_integer_t = std::conditional_t<is_signed<value_type_>(), std::int32_t, std::uint32_t>;
        using large_integer_t = std::conditional_t<is_signed<value_type_>(), std::int64_t, std::uint64_t>;
        using distribution_integer_t = std::conditional_t<sizeof(value_type_) <= 4, small_integer_t, large_integer_t>;
        std::uniform_int_distribution<distribution_integer_t> distribution(
            static_cast<distribution_integer_t>(min_val), static_cast<distribution_integer_t>(max_val));
        for (std::size_t i = 0; i < values_count; ++i)
            values_ptr[i] = static_cast<value_type_>(distribution(generator));
    }
    else {
        using distribution_float_t = std::conditional_t<sizeof(value_type_) <= 4, float, double>;
        std::uniform_real_distribution<distribution_float_t> distribution(static_cast<distribution_float_t>(min_val),
                                                                          static_cast<distribution_float_t>(max_val));
        for (std::size_t i = 0; i < values_count; ++i)
            values_ptr[i] = static_cast<value_type_>(distribution(generator));
    }
}

/**
 *  @brief Fill array with lognormal distribution (good for detecting numerical edge cases).
 */
template <typename value_type_, typename generator_type_>
void fill_lognormal(generator_type_ &generator, value_type_ *values_ptr, std::size_t values_count,
                    value_type_ mean = value_type_(0), value_type_ stddev = value_type_(0.5)) noexcept {

    using distribution_float_t = std::conditional_t<sizeof(value_type_) <= 4, float, double>;
    std::lognormal_distribution<distribution_float_t> distribution(mean, stddev);
    std::bernoulli_distribution sign_distribution(0.5);
    distribution_float_t const clamp_max = finite_max<value_type_>();

    for (std::size_t i = 0; i < values_count; ++i) {
        distribution_float_t val = distribution(generator);
        if (val > clamp_max) val = clamp_max;
        if (sign_distribution(generator)) val = -val;
        values_ptr[i] = static_cast<value_type_>(val);
    }
}

/**
 *  @brief Fill array with Cauchy distribution (heavy tails for stress testing).
 */
template <typename value_type_, typename generator_type_>
void fill_cauchy(generator_type_ &generator, value_type_ *values_ptr, std::size_t values_count,
                 value_type_ location = 0, value_type_ scale = 1) noexcept {

    using distribution_float_t = std::conditional_t<sizeof(value_type_) <= 4, float, double>;
    std::cauchy_distribution<distribution_float_t> distribution(location, scale);
    distribution_float_t const clamp_max = finite_max<value_type_>();
    distribution_float_t const clamp_min = finite_min<value_type_>();

    for (std::size_t i = 0; i < values_count; ++i) {
        distribution_float_t val = distribution(generator);
        if (val > clamp_max) val = clamp_max;
        else if (val < clamp_min) val = clamp_min;
        values_ptr[i] = static_cast<value_type_>(val);
    }
}

/**
 *  @brief Fill arrays with latitude and longitude coordinate values in radians.
 */
template <typename value_type_, typename generator_type_>
void fill_coordinates(generator_type_ &generator, value_type_ *lats_ptr, value_type_ *lons_ptr,
                      std::size_t values_count) noexcept {

    using distribution_float_t = std::conditional_t<sizeof(value_type_) <= 4, float, double>;
    constexpr distribution_float_t pi = distribution_float_t(3.14159265358979323846);
    std::uniform_real_distribution<distribution_float_t> lat_disttribution(-pi / 2, pi / 2);
    std::uniform_real_distribution<distribution_float_t> lon_disttribution(-pi, pi);
    for (std::size_t i = 0; i < values_count; ++i) {
        lats_ptr[i] = static_cast<value_type_>(lat_disttribution(generator));
        lons_ptr[i] = static_cast<value_type_>(lon_disttribution(generator));
    }
}

/**
 *  @brief Fill array as a probability distribution (sums to 1.0).
 */
template <typename value_type_, typename generator_type_>
void fill_probability(generator_type_ &generator, value_type_ *values_ptr, std::size_t values_count) noexcept {

    using distribution_float_t = std::conditional_t<sizeof(value_type_) <= 4, float, double>;
    std::uniform_real_distribution<distribution_float_t> distribution(distribution_float_t(0.01),
                                                                      distribution_float_t(1));

    distribution_float_t sum = 0;
    for (std::size_t i = 0; i < values_count; ++i) {
        distribution_float_t val = distribution(generator);
        values_ptr[i] = static_cast<value_type_>(val);
        sum += val;
    }
    for (std::size_t i = 0; i < values_count; ++i)
        values_ptr[i] = static_cast<value_type_>(static_cast<distribution_float_t>(values_ptr[i]) / sum);
}

} // namespace ashvardanian::numkong

#endif // NK_RANDOM_HPP

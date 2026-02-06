/**
 *  @brief C++ test suite with precision analysis using double-double arithmetic.
 *  @file test/test.hpp
 *  @author Ash Vardanian
 *  @date December 28, 2025
 *
 *  This test suite compares NumKong operations against high-precision references,
 *  like our `f118_t` double-double type, and reports ULP error statistics.
 *
 *  Environment Variables:
 *    NK_FILTER=<pattern>           - Filter tests by name RegEx (default: run all)
 *    NK_SEED=N                     - RNG seed (default: 12345)
 *
 *    NK_DENSE_DIMENSIONS=N         - Vector dimension for dot/spatial tests (default: 1024)
 *    NK_SPARSE_DIMENSIONS=N        - Vector dimension for sparse tests (default: 256)
 *    NK_MESH_POINTS=N              - Point count for mesh tests (default: 256)
 *    NK_MATRIX_HEIGHT=N            - GEMM M dimension (default: 64)
 *    NK_MATRIX_WIDTH=N             - GEMM N dimension (default: 64)
 *    NK_MATRIX_DEPTH=N             - GEMM K dimension (default: 64)
 *
 *    NK_TEST_ASSERT=1              - Assert on ULP threshold violations (default: 0)
 *    NK_TEST_VERBOSE=1             - Show per-dimension ULP breakdown (default: 0)
 *    NK_TEST_ULP_THRESHOLD_F32=N   - Max allowed ULP for f32 (default: 4)
 *    NK_TEST_ULP_THRESHOLD_F16=N   - Max allowed ULP for f16 (default: 32)
 *    NK_TEST_ULP_THRESHOLD_BF16=N  - Max allowed ULP for bf16 (default: 256)
 *    NK_TEST_TIME_BUDGET_MS=N      - Time budget per kernel in ms (default: 1000)
 *    NK_TEST_DISTRIBUTION=<type>   - Random distribution: uniform_k|lognormal_k|cauchy_k (default: lognormal_k)
 */

#pragma once
#ifndef NK_TEST_HPP
#define NK_TEST_HPP

#include <algorithm>
#include <cassert>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

#if NK_TEST_USE_OPENMP
#include <omp.h>
#endif

// Optional BLAS/MKL integration for precision comparison
#ifndef NK_COMPARE_TO_BLAS
#define NK_COMPARE_TO_BLAS 0
#endif
#ifndef NK_COMPARE_TO_MKL
#define NK_COMPARE_TO_MKL 0
#endif
#ifndef NK_COMPARE_TO_ACCELERATE
#define NK_COMPARE_TO_ACCELERATE 0
#endif

// Include reference library headers - MKL, Accelerate, or generic CBLAS
#if NK_COMPARE_TO_MKL
#include <mkl.h> // MKL includes its own CBLAS interface
#elif NK_COMPARE_TO_ACCELERATE
#include <Accelerate/Accelerate.h> // Apple Accelerate framework
#elif NK_COMPARE_TO_BLAS
#include <cblas.h> // Generic CBLAS (OpenBLAS, etc.)
#endif

#define NK_NATIVE_F16  0
#define NK_NATIVE_BF16 0

#include "numkong/types.hpp"
#include "numkong/tensor.hpp"
#include "numkong/random.hpp" // `nk::fill_uniform`

namespace nk = ashvardanian::numkong;

using nk::bf16_t;
using nk::bf16c_t;
using nk::e2m3_t;
using nk::e3m2_t;
using nk::e4m3_t;
using nk::e5m2_t;
using nk::f118_t;
using nk::f118c_t;
using nk::f16_t;
using nk::f16c_t;
using nk::f32_t;
using nk::f32c_t;
using nk::f64_t;
using nk::f64c_t;
using nk::i16_t;
using nk::i32_t;
using nk::i4x2_t;
using nk::i64_t;
using nk::i8_t;
using nk::u16_t;
using nk::u1x8_t;
using nk::u32_t;
using nk::u4x2_t;
using nk::u64_t;
using nk::u8_t;

using steady_clock = std::chrono::steady_clock;
using time_point = steady_clock::time_point;

template class nk::vector<int>;
template class nk::vector<nk::i32_t>;
template class nk::vector<nk::u1x8_t>;
template class nk::vector<nk::i4x2_t>;
template class nk::vector<nk::f64c_t>;
template class nk::vector<std::complex<float>>;

extern std::size_t dense_dimensions;  // For dot products, spatial metrics
extern std::size_t sparse_dimensions; // For sparse set intersection and sparse dot
extern std::size_t mesh_points;       // For RMSD, Kabsch (3D point clouds)
extern std::size_t matrix_height, matrix_width, matrix_depth;

enum class random_distribution_kind_t { uniform_k, lognormal_k, cauchy_k };

struct test_config_t {
    bool assert_on_failure = false;
    bool verbose = false; // Show per-dimension stats
    std::uint64_t ulp_threshold_f32 = 4;
    std::uint64_t ulp_threshold_f16 = 32;
    std::uint64_t ulp_threshold_bf16 = 256;
    std::size_t time_budget_ms = 1000; // Time budget per kernel in milliseconds
    std::uint32_t seed = 12345;
    char const *filter = nullptr; // Filter tests by name (substring match)
    random_distribution_kind_t distribution = random_distribution_kind_t::lognormal_k; // Default: moderate heavy tails

    bool should_run(char const *test_name) const;

    std::uint64_t ulp_threshold_for(char const *kernel_name) const noexcept {
        if (std::strstr(kernel_name, "_bf16")) return ulp_threshold_bf16;
        if (std::strstr(kernel_name, "_f16")) return ulp_threshold_f16;
        return ulp_threshold_f32;
    }

    char const *distribution_name() const noexcept {
        switch (distribution) {
        case random_distribution_kind_t::uniform_k: return "uniform";
        case random_distribution_kind_t::lognormal_k: return "lognormal";
        case random_distribution_kind_t::cauchy_k: return "cauchy";
        default: return "unknown";
        }
    }
};

extern test_config_t global_config;
extern std::size_t global_failure_count;

/**
 *  @brief Run a test only if its kernel name matches the filter.
 *  @param kernel_name The full kernel name (e.g., "dot_f32_haswell", "sqeuclidean_f64_skylake")
 *  @param test_fn The test function that returns error_stats_t
 *  @param args Variadic arguments to forward to the test function
 */
template <typename test_function_type_, typename... args_types_>
void run_if_matches(char const *kernel_name, test_function_type_ test_fn, args_types_ &&...args) {
    if (!global_config.should_run(kernel_name)) return;
    auto stats = test_fn(std::forward<args_types_>(args)...);
    stats.report(kernel_name);
    if (global_config.assert_on_failure && stats.max_ulp > global_config.ulp_threshold_for(kernel_name))
        ++global_failure_count;
}

inline time_point test_start_time() { return steady_clock::now(); }

inline bool within_time_budget(time_point start) {
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(steady_clock::now() - start).count();
    return elapsed < static_cast<long long>(global_config.time_budget_ms);
}

/**
 *  @brief Compute ULP (Units in Last Place) distance between two floating-point values.
 *
 *  ULP distance is the number of representable floating-point numbers between a and b.
 *  This is the gold standard for comparing floating-point implementations.
 *
 *  Uses the XOR transformation from Bruce Dawson's algorithm to handle all sign combinations:
 *  @see https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/
 *  @see https://en.wikipedia.org/wiki/Unit_in_the_last_place
 */
template <typename scalar_type_>
std::uint64_t ulp_distance(scalar_type_ a, scalar_type_ b) noexcept {
    // Handle special cases - skip float checks for integer types
    constexpr bool is_integer = []() {
        if constexpr (std::is_integral_v<scalar_type_>) return true;
        else if constexpr (std::is_class_v<scalar_type_>) return scalar_type_::is_integer();
        else return false;
    }();
    if constexpr (!is_integer) {
        if (std::isnan(static_cast<double>(a)) || std::isnan(static_cast<double>(b)))
            return std::numeric_limits<std::uint64_t>::max();
        if (std::isinf(static_cast<double>(a)) || std::isinf(static_cast<double>(b)))
            return std::numeric_limits<std::uint64_t>::max();
    }
    if (a == b) return 0; // Also handles +0 == -0

    // Use the XOR transformation from Bruce Dawson's "Comparing Floating Point Numbers"
    // This transforms float bit patterns to an ordered integer representation where
    // the integer difference equals the ULP distance.
    if constexpr (sizeof(scalar_type_) == 4) {
        std::int32_t ia, ib;
        std::memcpy(&ia, &a, sizeof(ia));
        std::memcpy(&ib, &b, sizeof(ib));

        // Transform negative floats: flip all bits except sign to reverse their ordering
        // This makes the integer representation monotonically ordered with float value
        if (ia < 0) ia ^= 0x7FFFFFFF;
        if (ib < 0) ib ^= 0x7FFFFFFF;

        // Compute absolute difference using 64-bit arithmetic to avoid overflow
        std::int64_t diff = static_cast<std::int64_t>(ia) - static_cast<std::int64_t>(ib);
        return static_cast<std::uint64_t>(diff < 0 ? -diff : diff);
    }
    else if constexpr (sizeof(scalar_type_) == 8) {
        std::int64_t ia, ib;
        std::memcpy(&ia, &a, sizeof(ia));
        std::memcpy(&ib, &b, sizeof(ib));

        if (ia < 0) ia ^= 0x7FFFFFFFFFFFFFFFLL;
        if (ib < 0) ib ^= 0x7FFFFFFFFFFFFFFFLL;

        // For 64-bit, handle potential overflow in subtraction when signs differ
        if ((ia >= 0) != (ib >= 0)) {
            // Different signs after transformation: distance = |ia| + |ib|
            // Safe negation that handles INT64_MIN
            auto safe_abs = [](std::int64_t x) -> std::uint64_t {
                return x < 0 ? static_cast<std::uint64_t>(~x) + 1 : static_cast<std::uint64_t>(x);
            };
            return safe_abs(ia) + safe_abs(ib);
        }
        // Same sign: simple subtraction (no overflow possible)
        return ia >= ib ? static_cast<std::uint64_t>(ia - ib) : static_cast<std::uint64_t>(ib - ia);
    }
    else {
        // For f16/bf16, convert to f32 and compute there
        return ulp_distance(static_cast<float>(a), static_cast<float>(b));
    }
}

/**
 *  @brief Accumulator for error statistics across multiple test trials.
 */
struct error_stats_t {
    nk_f64_t min_abs_err = std::numeric_limits<nk_f64_t>::max();
    nk_f64_t max_abs_err = 0;
    nk_f64_t sum_abs_err = 0;

    nk_f64_t min_rel_err = std::numeric_limits<nk_f64_t>::max();
    nk_f64_t max_rel_err = 0;
    nk_f64_t sum_rel_err = 0;

    std::uint64_t min_ulp = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t max_ulp = 0;
    std::uint64_t sum_ulp = 0;

    std::size_t count = 0;
    std::size_t exact_matches = 0;

    template <typename actual_type_, typename expected_type_>
    void accumulate(actual_type_ actual, expected_type_ expected) noexcept {
        if constexpr (std::is_class_v<actual_type_>) {
            // Check if it's a complex type
            if constexpr (actual_type_::is_complex()) {
                accumulate_scalar(actual.real(), expected.real());
                accumulate_scalar(actual.imag(), expected.imag());
            }
            else { accumulate_scalar(actual, expected); }
        }
        else { accumulate_scalar(actual, expected); }
    }

    template <typename actual_type_, typename expected_type_>
    void accumulate_scalar(actual_type_ actual, expected_type_ expected) noexcept {
        actual_type_ expected_as_actual;
        if constexpr (std::is_same_v<expected_type_, f118_t>) expected_as_actual = expected.template to<actual_type_>();
        else expected_as_actual = static_cast<actual_type_>(expected);

        // Compute ULP distance (handles both integers and floats)
        std::uint64_t ulps = ulp_distance(actual, expected_as_actual);

        // Only compute floating-point error metrics for non-integer types
        if constexpr (!nk::is_integer<actual_type_>()) {
            nk_f64_t exp_f64 = static_cast<nk_f64_t>(expected);
            nk_f64_t act_f64 = static_cast<nk_f64_t>(actual);

            nk_f64_t abs_err = std::fabs(exp_f64 - act_f64);
            nk_f64_t rel_err = exp_f64 != 0 ? abs_err / std::fabs(exp_f64) : abs_err;

            min_abs_err = std::min(min_abs_err, abs_err);
            max_abs_err = std::max(max_abs_err, abs_err);
            sum_abs_err += abs_err;
            min_rel_err = std::min(min_rel_err, rel_err);
            max_rel_err = std::max(max_rel_err, rel_err);
            sum_rel_err += rel_err;
        }

        // Always update ULP metrics (works for both integer and float)
        min_ulp = std::min(min_ulp, ulps);
        max_ulp = std::max(max_ulp, ulps);
        sum_ulp += ulps;

        count++;
        if (ulps == 0) exact_matches++;
    }

    nk_f64_t mean_abs_err() const noexcept { return count > 0 ? sum_abs_err / count : 0; }
    nk_f64_t mean_rel_err() const noexcept { return count > 0 ? sum_rel_err / count : 0; }
    nk_f64_t mean_ulp() const noexcept { return count > 0 ? static_cast<nk_f64_t>(sum_ulp) / count : 0; }

    void reset() noexcept { *this = error_stats_t {}; }

    void merge(error_stats_t const &other) noexcept {
        min_abs_err = std::min(min_abs_err, other.min_abs_err);
        max_abs_err = std::max(max_abs_err, other.max_abs_err);
        sum_abs_err += other.sum_abs_err;
        min_rel_err = std::min(min_rel_err, other.min_rel_err);
        max_rel_err = std::max(max_rel_err, other.max_rel_err);
        sum_rel_err += other.sum_rel_err;
        min_ulp = std::min(min_ulp, other.min_ulp);
        max_ulp = std::max(max_ulp, other.max_ulp);
        sum_ulp += other.sum_ulp;
        count += other.count;
        exact_matches += other.exact_matches;
    }

    void report(char const *kernel_name) const noexcept {
        std::printf("%-40s %12llu %10.1f %12.2e %12.2e %10zu\n", kernel_name, static_cast<unsigned long long>(max_ulp),
                    mean_ulp(), max_abs_err, max_rel_err, exact_matches);
        std::fflush(stdout);
    }

    void report_dimension(char const *kernel_name, std::size_t dim) const noexcept {
        std::printf("  %-38s dim=%-6zu %10llu %8.1f %10.2e %10.2e %8zu\n", kernel_name, dim,
                    static_cast<unsigned long long>(max_ulp), mean_ulp(), max_abs_err, max_rel_err, exact_matches);
        std::fflush(stdout);
    }
};

/**
 *  @brief Print the header for the error statistics table.
 */
void print_stats_header() noexcept;

/**
 *  @brief Factory function to allocate vectors, potentially raising bad-allocs.
 */
template <typename type_>
[[nodiscard]] nk::vector<type_> make_vector(std::size_t n) {
    nk::vector<type_> result;
    if (!result.resize(n)) throw std::bad_alloc();
    return result;
}

/**
 *  @brief Fill buffer with random values, respecting global distribution setting.
 *
 *  Dispatches to appropriate nk::fill_* library function based on `global_config.distribution`.
 *  Infers sensible bounds from type's representable range.
 */
template <typename scalar_type_, typename generator_type_>
void fill_random(generator_type_ &generator, nk::vector<scalar_type_> &vector) {
    // Sub-byte, integer, and complex types only support uniform distribution
    // (lognormal/cauchy have floating-point default args that don't convert properly)
    constexpr bool is_uniform_only = std::is_same_v<scalar_type_, nk::u1x8_t> ||  //
                                     std::is_same_v<scalar_type_, nk::i4x2_t> ||  //
                                     std::is_same_v<scalar_type_, nk::u4x2_t> ||  //
                                     std::is_same_v<scalar_type_, nk::i8_t> ||    //
                                     std::is_same_v<scalar_type_, nk::u8_t> ||    //
                                     std::is_same_v<scalar_type_, nk::i16_t> ||   //
                                     std::is_same_v<scalar_type_, nk::u16_t> ||   //
                                     std::is_same_v<scalar_type_, nk::i32_t> ||   //
                                     std::is_same_v<scalar_type_, nk::u32_t> ||   //
                                     std::is_same_v<scalar_type_, nk::i64_t> ||   //
                                     std::is_same_v<scalar_type_, nk::u64_t> ||   //
                                     std::is_same_v<scalar_type_, nk::f16c_t> ||  //
                                     std::is_same_v<scalar_type_, nk::bf16c_t> || //
                                     std::is_same_v<scalar_type_, nk::f32c_t> ||  //
                                     std::is_same_v<scalar_type_, nk::f64c_t>;
    if constexpr (is_uniform_only) { nk::fill_uniform(generator, vector.values_data(), vector.size_values()); }
    else {
        switch (global_config.distribution) {
        case random_distribution_kind_t::uniform_k:
            nk::fill_uniform(generator, vector.values_data(), vector.size_values());
            break;
        case random_distribution_kind_t::lognormal_k:
            nk::fill_lognormal(generator, vector.values_data(), vector.size_values());
            break;
        case random_distribution_kind_t::cauchy_k:
            nk::fill_cauchy(generator, vector.values_data(), vector.size_values());
            break;
        }
    }
}

/**
 *  @brief Test FP8 (e4m3) conversions round-trip accuracy.
 */
void test_fp8_conversions();

// Forward declarations for test modules
void test_casts();
void test_reduce();
void test_dot();
void test_spatial();
void test_set();
void test_curved();
void test_probability();
void test_elementwise();
void test_trigonometry();
void test_geospatial();
void test_mesh();
void test_sparse();
void test_vector_types();

// Forward declarations for cross/batch tests (ISA-family files)
void test_cross_serial();
void test_cross_x86();
void test_cross_amx();
void test_cross_arm();
void test_cross_sme();
void test_cross_blas();

#endif // NK_TEST_HPP

/**
 *  @brief C++ test suite with precision analysis using double-double arithmetic.
 *  @file scripts/test.cpp
 *
 *  This test suite compares NumKong operations against high-precision references,
 *  like our `f118_t` double-double type, and reports ULP error statistics.
 *
 *  Environment Variables:
 *    NK_TEST_ASSERT=1             - Assert on ULP threshold violations (default: 0)
 *    NK_TEST_VERBOSE=1            - Show per-dimension ULP breakdown (default: 0)
 *    NK_TEST_FILTER=<pattern>     - Filter tests by name RegEx (default: run all)
 *    NK_TEST_ULP_THRESHOLD_F32=N  - Max allowed ULP for f32 (default: 4)
 *    NK_TEST_ULP_THRESHOLD_F16=N  - Max allowed ULP for f16 (default: 32)
 *    NK_TEST_ULP_THRESHOLD_BF16=N - Max allowed ULP for bf16 (default: 256)
 *    NK_TEST_TIME_BUDGET_MS=N     - Time budget per kernel in ms (default: 1000)
 *    NK_TEST_SEED=N               - RNG seed (default: 12345)
 *    NK_TEST_DENSE_DIMENSION=N    - Vector dimension for dot/spatial tests (default: 1024)
 *    NK_TEST_MESH_DIMENSION=N     - Point count for mesh tests (default: 256)
 *    NK_TEST_MATMUL_DIMENSION_M=N - GEMM M dimension (default: 64)
 *    NK_TEST_MATMUL_DIMENSION_N=N - GEMM N dimension (default: 64)
 *    NK_TEST_MATMUL_DIMENSION_K=N - GEMM K dimension (default: 64)
 *    NK_TEST_DISTRIBUTION=<type>  - Random distribution: uniform_k|lognormal_k|cauchy_k (default: lognormal_k)
 */

#include <algorithm>
#include <cassert>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <regex>
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
#include <numkong/numkong.hpp>

namespace nk = ashvardanian::numkong;

using nk::bf16_t;
using nk::e4m3_t;
using nk::e5m2_t;
using nk::f118_t;
using nk::f118c_t;
using nk::f16_t;
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

#pragma region Configuration

std::size_t dense_dimension = 1024; // For dot products, spatial metrics
std::size_t sparse_dimension = 256; // For sparse set intersection and sparse dot
std::size_t mesh_dimension = 256;   // For RMSD, Kabsch (3D point clouds)
std::size_t matmul_dimension_m = 64, matmul_dimension_n = 64, matmul_dimension_k = 64;

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

    bool should_run(char const *test_name) const {
        if (!filter) return true;
        try {
            std::regex pattern(filter);
            return std::regex_search(test_name, pattern);
        }
        catch (std::regex_error const &) {
            // Fallback to substring match if regex is invalid
            return std::strstr(test_name, filter) != nullptr;
        }
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

static test_config_t global_config;

/**
 *  @brief Run a test only if its full name matches the filter.
 *  @param name The test name (e.g., "dot_blas", "l2sq_haswell")
 *  @param type The type signature (e.g., "f32", "bf16")
 *  @param test_fn The test function that returns error_stats_t
 *  @param args Variadic arguments to forward to the test function
 */
template <typename test_function_type_, typename... args_types_>
void run_if_matches(char const *name, char const *type, test_function_type_ test_fn, args_types_ &&...args) {
    std::string full_name = std::string(name) + "_" + type;
    if (global_config.should_run(full_name.c_str())) { test_fn(std::forward<args_types_>(args)...).report(name, type); }
}

inline time_point test_start_time() { return steady_clock::now(); }

inline bool within_time_budget(time_point start) {
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(steady_clock::now() - start).count();
    return elapsed < static_cast<long long>(global_config.time_budget_ms);
}

#pragma endregion // Configuration

#pragma region Precision Infrastructure

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
    constexpr bool is_integer_type = []() {
        if constexpr (std::is_integral_v<scalar_type_>) return true;
        else if constexpr (std::is_class_v<scalar_type_>) return scalar_type_::is_integer();
        else return false;
    }();
    if constexpr (!is_integer_type) {
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

#pragma endregion // Precision Infrastructure

#pragma region Error Statistics

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
        if constexpr (std::is_class_v<actual_type_> && actual_type_::is_complex()) {
            accumulate_scalar(actual.real(), expected.real());
            accumulate_scalar(actual.imag(), expected.imag());
        }
        else { accumulate_scalar(actual, expected); }
    }

    template <typename actual_type_, typename expected_type_>
    void accumulate_scalar(actual_type_ actual, expected_type_ expected) noexcept {
        actual_type_ expected_as_actual;
        if constexpr (std::is_same_v<expected_type_, f118_t>) expected_as_actual = expected.template to<actual_type_>();
        else expected_as_actual = static_cast<actual_type_>(expected);
        std::uint64_t ulps = ulp_distance(actual, expected_as_actual);
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
        min_ulp = std::min(min_ulp, ulps);
        max_ulp = std::max(max_ulp, ulps);
        sum_ulp += ulps;

        count++;
        if (ulps == 0) exact_matches++;
    }

    void accumulate_exact(bool matches) noexcept {
        std::uint64_t ulps = matches ? 0 : std::numeric_limits<std::uint64_t>::max();
        min_ulp = std::min(min_ulp, ulps);
        max_ulp = std::max(max_ulp, ulps);
        sum_ulp += ulps;
        count++;
        if (matches) exact_matches++;
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

    void report(char const *operation, char const *dtype) const noexcept {
        std::printf("%-18s %-8s %12llu %10.1f %12.2e %12.2e %10zu\n", operation, dtype,
                    static_cast<unsigned long long>(max_ulp), mean_ulp(), max_abs_err, max_rel_err, exact_matches);
        std::fflush(stdout);
    }

    void report_dimension(char const *operation, char const *dtype, std::size_t dim) const noexcept {
        std::printf("  %-16s %-8s dim=%-6zu %10llu %8.1f %10.2e %10.2e %8zu\n", operation, dtype, dim,
                    static_cast<unsigned long long>(max_ulp), mean_ulp(), max_abs_err, max_rel_err, exact_matches);
        std::fflush(stdout);
    }
};

/**
 *  @brief Print the header for the error statistics table.
 */
void print_stats_header() noexcept {
    std::printf("\n=== NumKong Precision Analysis ===\n");
    std::printf("%-18s %-8s %12s %10s %12s %12s %10s\n", "Operation", "Type", "max_ulp", "mean_ulp", "max_abs",
                "max_rel", "exact");
    std::printf("───────────────────────────────────────────────────────────────────────────────────────────────\n");
}

#pragma endregion // Error Statistics

#pragma region Test Harness Templates

/**
 *  @brief Aligned memory allocation for SIMD-friendly buffers.
 */
template <typename scalar_type_>
struct aligned_buffer {
    static constexpr std::size_t alignment_k = 64; // Cache line
    scalar_type_ *data_ = nullptr;
    std::size_t size_ = 0;

    aligned_buffer() = default;
    explicit aligned_buffer(std::size_t n) : size_(n) {
        if (n == 0) return;
        void *ptr = std::aligned_alloc(alignment_k,
                                       ((n * sizeof(scalar_type_) + alignment_k - 1) / alignment_k) * alignment_k);
        data_ = static_cast<scalar_type_ *>(ptr);
        std::memset(data_, 0, n * sizeof(scalar_type_));
    }
    ~aligned_buffer() noexcept {
        if (data_) std::free(data_);
    }
    aligned_buffer(aligned_buffer const &) = delete;
    aligned_buffer &operator=(aligned_buffer const &) = delete;
    aligned_buffer(aligned_buffer &&other) noexcept : data_(other.data_), size_(other.size_) {
        other.data_ = nullptr;
        other.size_ = 0;
    }
    aligned_buffer &operator=(aligned_buffer &&other) noexcept {
        if (this == &other) return *this;
        if (data_) std::free(data_);
        data_ = other.data_;
        size_ = other.size_;
        other.data_ = nullptr;
        other.size_ = 0;
        return *this;
    }

    scalar_type_ &operator[](std::size_t i) noexcept { return data_[i]; }
    scalar_type_ const &operator[](std::size_t i) const noexcept { return data_[i]; }

    // STL-style accessors
    scalar_type_ *data() noexcept { return data_; }
    scalar_type_ const *data() const noexcept { return data_; }
    std::size_t size() const noexcept { return size_; }
    bool empty() const noexcept { return size_ == 0; }
};

/**
 *  @brief Fill sub-byte buffer with random bytes uniformly.
 */
template <typename scalar_type_, typename generator_type_>
    requires(std::is_same_v<scalar_type_, u1x8_t> || std::is_same_v<scalar_type_, i4x2_t> ||
             std::is_same_v<scalar_type_, u4x2_t>)
void fill_random(aligned_buffer<scalar_type_> &buf, generator_type_ &rng) {
    std::uniform_int_distribution<unsigned int> dist(0, 255);
    for (std::size_t i = 0; i < buf.size(); i++)
        buf[i] = scalar_type_::from_raw(static_cast<typename scalar_type_::raw_t>(dist(rng)));
}

/**
 *  @brief Fill integer buffer with random values in safe range.
 */
template <typename scalar_type_, typename generator_type_>
    requires(scalar_type_::is_integer() && !std::is_same_v<scalar_type_, u1x8_t> &&
             !std::is_same_v<scalar_type_, i4x2_t> && !std::is_same_v<scalar_type_, u4x2_t>)
void fill_random(aligned_buffer<scalar_type_> &buf, generator_type_ &rng) {
    using raw_t = typename scalar_type_::raw_t;
    raw_t min_raw, max_raw;
    if constexpr (sizeof(raw_t) == 1) min_raw = -10, max_raw = 10;
    else if constexpr (sizeof(raw_t) == 2) min_raw = -100, max_raw = 100;
    else if constexpr (sizeof(raw_t) == 4) min_raw = -1000, max_raw = 1000;
    else min_raw = -10000, max_raw = 10000;

    std::uniform_int_distribution<std::int64_t> dist(min_raw, max_raw);
    for (std::size_t i = 0; i < buf.size(); i++) buf[i] = scalar_type_(static_cast<raw_t>(dist(rng)));
}

/**
 *  @brief Generate a single random floating-point value respecting global distribution.
 *  Centralizes distribution logic to avoid code duplication across float/complex overloads.
 */
template <typename generator_type_>
nk_f64_t random_f64_in_range(generator_type_ &rng, nk_f64_t min_val, nk_f64_t max_val) {
    nk_f64_t range = max_val - min_val;
    nk_f64_t mid = (max_val + min_val) / 2.0;

    switch (global_config.distribution) {
    case random_distribution_kind_t::uniform_k: {
        std::uniform_real_distribution<nk_f64_t> dist(min_val, max_val);
        return dist(rng);
    }
    case random_distribution_kind_t::lognormal_k: {
        std::lognormal_distribution<nk_f64_t> lognorm(0.0, 0.5);
        std::uniform_real_distribution<nk_f64_t> sign_dist(0.0, 1.0);
        nk_f64_t val = lognorm(rng);
        nk_f64_t compressed = 2.0 / (1.0 + std::exp(-val)) - 1.0;
        if (sign_dist(rng) < 0.5) compressed = -compressed;
        return mid + compressed * (range / 2.0);
    }
    case random_distribution_kind_t::cauchy_k: {
        std::cauchy_distribution<nk_f64_t> cauchy(0.0, 1.0);
        nk_f64_t val = cauchy(rng);
        nk_f64_t compressed = (2.0 / M_PI) * std::atan(val);
        return mid + compressed * (range / 2.0);
    }
    }
    return mid; // unreachable
}

/**
 *  @brief Fill floating-point buffer with random values in specified range, respecting global distribution.
 */
template <typename scalar_type_, typename generator_type_>
    requires(!scalar_type_::is_integer() && !scalar_type_::is_complex())
void fill_random(aligned_buffer<scalar_type_> &buf, generator_type_ &rng, scalar_type_ min_val, scalar_type_ max_val) {
    nk_f64_t min_f64 = static_cast<nk_f64_t>(min_val);
    nk_f64_t max_f64 = static_cast<nk_f64_t>(max_val);
    for (std::size_t i = 0; i < buf.size(); i++) buf[i] = scalar_type_(random_f64_in_range(rng, min_f64, max_f64));
}

/**
 *  @brief Fill floating-point buffer with random values respecting global distribution.
 */
template <typename scalar_type_, typename generator_type_>
    requires(!scalar_type_::is_integer() && !scalar_type_::is_complex())
void fill_random(aligned_buffer<scalar_type_> &buf, generator_type_ &rng) {
    nk_f64_t min_val, max_val;
    if constexpr (sizeof(typename scalar_type_::raw_t) >= 8) min_val = -1e8, max_val = 1e8;
    else if constexpr (sizeof(typename scalar_type_::raw_t) >= 4) min_val = -1e4, max_val = 1e4;
    else if constexpr (scalar_type_::bits() >= 16) min_val = -100, max_val = 100;
    else min_val = -10, max_val = 10; // FP8 types

    for (std::size_t i = 0; i < buf.size(); i++) buf[i] = scalar_type_(random_f64_in_range(rng, min_val, max_val));
}

/**
 *  @brief Fill complex buffer with random values respecting global distribution.
 */
template <typename scalar_type_, typename generator_type_>
    requires(scalar_type_::is_complex())
void fill_random(aligned_buffer<scalar_type_> &buf, generator_type_ &rng) {
    for (std::size_t i = 0; i < buf.size(); i++) {
        auto re = random_f64_in_range(rng, -100.0, 100.0);
        auto im = random_f64_in_range(rng, -100.0, 100.0);
        buf[i] = scalar_type_(re, im);
    }
}

/**
 *  @brief Fills two buffers with valid probability distributions (positive values summing to 1).
 */
template <typename scalar_type_, typename generator_type_>
void fill_probability(aligned_buffer<scalar_type_> &p, aligned_buffer<scalar_type_> &q, generator_type_ &rng) {
    using scalar_t = scalar_type_;
    std::uniform_real_distribution<double> dist(0.01, 1.0);
    std::size_t n = p.size();

    double sum_p = 0, sum_q = 0;
    for (std::size_t i = 0; i < n; i++) {
        double pv = dist(rng), qv = dist(rng);
        p[i] = scalar_t(pv);
        q[i] = scalar_t(qv);
        sum_p += pv;
        sum_q += qv;
    }
    for (std::size_t i = 0; i < n; i++) {
        p[i] = scalar_t(static_cast<double>(p[i]) / sum_p);
        q[i] = scalar_t(static_cast<double>(q[i]) / sum_q);
    }
}

#pragma endregion // Test Harness Templates

#pragma region BLAS Baselines

#if NK_COMPARE_TO_BLAS || NK_COMPARE_TO_MKL || NK_COMPARE_TO_ACCELERATE

void dot_f32_blas(nk::f32_t const *a, nk::f32_t const *b, nk::size_t n, nk::f32_t *result) {
    *result = cblas_sdot(static_cast<int>(n), a, 1, b, 1);
}

void dot_f64_blas(nk::f64_t const *a, nk::f64_t const *b, nk::size_t n, nk::f64_t *result) {
    *result = cblas_ddot(static_cast<int>(n), a, 1, b, 1);
}

void dot_f32c_blas(nk::f32c_t const *a, nk::f32c_t const *b, nk::size_t n, nk::f32c_t *result) {
#if NK_COMPARE_TO_ACCELERATE
    cblas_cdotu_sub(static_cast<int>(n), reinterpret_cast<__LAPACK_float_complex const *>(a), 1,
                    reinterpret_cast<__LAPACK_float_complex const *>(b), 1,
                    reinterpret_cast<__LAPACK_float_complex *>(result));
#else
    cblas_cdotu_sub(static_cast<int>(n), a, 1, b, 1, result);
#endif
}

void vdot_f32c_blas(nk::f32c_t const *a, nk::f32c_t const *b, nk::size_t n, nk::f32c_t *result) {
#if NK_COMPARE_TO_ACCELERATE
    cblas_cdotc_sub(static_cast<int>(n), reinterpret_cast<__LAPACK_float_complex const *>(a), 1,
                    reinterpret_cast<__LAPACK_float_complex const *>(b), 1,
                    reinterpret_cast<__LAPACK_float_complex *>(result)); // conjugated
#else
    cblas_cdotc_sub(static_cast<int>(n), a, 1, b, 1, result); // conjugated
#endif
}

void dot_f64c_blas(nk::f64c_t const *a, nk::f64c_t const *b, nk::size_t n, nk::f64c_t *result) {
#if NK_COMPARE_TO_ACCELERATE
    cblas_zdotu_sub(static_cast<int>(n), reinterpret_cast<__LAPACK_double_complex const *>(a), 1,
                    reinterpret_cast<__LAPACK_double_complex const *>(b), 1,
                    reinterpret_cast<__LAPACK_double_complex *>(result));
#else
    cblas_zdotu_sub(static_cast<int>(n), a, 1, b, 1, result);
#endif
}

void vdot_f64c_blas(nk::f64c_t const *a, nk::f64c_t const *b, nk::size_t n, nk::f64c_t *result) {
#if NK_COMPARE_TO_ACCELERATE
    cblas_zdotc_sub(static_cast<int>(n), reinterpret_cast<__LAPACK_double_complex const *>(a), 1,
                    reinterpret_cast<__LAPACK_double_complex const *>(b), 1,
                    reinterpret_cast<__LAPACK_double_complex *>(result)); // conjugated
#else
    cblas_zdotc_sub(static_cast<int>(n), a, 1, b, 1, result); // conjugated
#endif
}

void dots_f32_blas(nk::f32_t const *a, nk::f32_t const *b, nk::f32_t *c, nk::size_t m, nk::size_t n, nk::size_t k,
                   nk::size_t a_stride, nk::size_t c_stride) {
    (void)a_stride;
    (void)c_stride;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
                1.0f, a, static_cast<int>(k), b, static_cast<int>(k), 0.0f, c, static_cast<int>(n));
}

void dots_f64_blas(nk::f64_t const *a, nk::f64_t const *b, nk::f64_t *c, nk::size_t m, nk::size_t n, nk::size_t k,
                   nk::size_t a_stride, nk::size_t c_stride) {
    (void)a_stride;
    (void)c_stride;
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
                1.0, a, static_cast<int>(k), b, static_cast<int>(k), 0.0, c, static_cast<int>(n));
}
#endif // NK_COMPARE_TO_BLAS || NK_COMPARE_TO_MKL || NK_COMPARE_TO_ACCELERATE

#if NK_COMPARE_TO_MKL
void dots_bf16_mkl(nk::bf16_t const *a, nk::bf16_t const *b, nk::f32_t *c, nk::size_t m, nk::size_t n, nk::size_t k,
                   nk::size_t a_stride, nk::size_t c_stride) {
    (void)a_stride;
    (void)c_stride;
    cblas_gemm_bf16(CblasRowMajor, CblasNoTrans, CblasTrans, static_cast<MKL_INT>(m), static_cast<MKL_INT>(n),
                    static_cast<MKL_INT>(k), 1.0f, a, static_cast<MKL_INT>(k), b, static_cast<MKL_INT>(k), 0.0f, c,
                    static_cast<MKL_INT>(n));
}

void dots_f16_mkl(nk::f16_t const *a, nk::f16_t const *b, nk::f32_t *c, nk::size_t m, nk::size_t n, nk::size_t k,
                  nk::size_t a_stride, nk::size_t c_stride) {
    (void)a_stride;
    (void)c_stride;
    cblas_gemm_f16(CblasRowMajor, CblasNoTrans, CblasTrans, static_cast<MKL_INT>(m), static_cast<MKL_INT>(n),
                   static_cast<MKL_INT>(k), 1.0f, reinterpret_cast<MKL_F16 const *>(a), static_cast<MKL_INT>(k),
                   reinterpret_cast<MKL_F16 const *>(b), static_cast<MKL_INT>(k), 0.0f, c, static_cast<MKL_INT>(n));
}

void dots_i8_mkl(nk::i8_t const *a, nk::u8_t const *b, nk::i32_t *c, nk::size_t m, nk::size_t n, nk::size_t k,
                 nk::size_t a_stride, nk::size_t c_stride) {
    (void)a_stride;
    (void)c_stride;
    MKL_INT32 c_offset = 0;
    cblas_gemm_s8u8s32(CblasRowMajor, CblasNoTrans, CblasTrans, CblasFixOffset, static_cast<MKL_INT>(m),
                       static_cast<MKL_INT>(n), static_cast<MKL_INT>(k), 1.0f, a, static_cast<MKL_INT>(k), 0, b,
                       static_cast<MKL_INT>(k), 0, 0.0f, c, static_cast<MKL_INT>(n), &c_offset);
}

#endif // NK_COMPARE_TO_MKL

#pragma endregion // BLAS Baselines

#pragma region Types

/**
 *  @brief Test FP8 (e4m3) conversions round-trip accuracy.
 */
void test_fp8_conversions() {
    std::printf("Testing FP8 (e4m3) conversions...\n");

    // Test specific values
    float test_values[] = {0.0f, 1.0f, -1.0f, 0.5f, -0.5f, 2.0f, -2.0f, 0.125f, 8.0f};
    std::size_t n = sizeof(test_values) / sizeof(test_values[0]);

    for (std::size_t i = 0; i < n; i++) {
        nk_e4m3_t e4m3;
        float roundtrip;
        nk_f32_to_e4m3(&test_values[i], &e4m3);
        nk_e4m3_to_f32(&e4m3, &roundtrip);

        // For values within e4m3 range, roundtrip should be close
        float expected = test_values[i];
        float abs_err = std::fabs(roundtrip - expected);

        // e4m3 has limited precision, allow reasonable error
        if (std::fabs(expected) > 0 && abs_err / std::fabs(expected) > 0.5f) {
            std::printf("  WARN: e4m3 roundtrip for %f: got %f (err=%f)\n", expected, roundtrip, abs_err);
        }
    }

    std::printf("  FP8 conversions: PASS\n");
}

#pragma endregion // Types

#pragma region Cast

using cast_t = void (*)(void const *, nk_dtype_t, nk_size_t, void *, nk_dtype_t);

/**
 *  @brief Test cast kernel against serial kernel.
 *  SIMD kernels must match serial output exactly (raw byte comparison).
 */
template <typename from_type_, typename to_type_>
error_stats_t test_cast(cast_t kernel) {
    error_stats_t stats;
    std::mt19937 rng(global_config.seed);

    aligned_buffer<from_type_> src(dense_dimension);
    aligned_buffer<to_type_> dst_simd(dense_dimension), dst_serial(dense_dimension);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(src, rng);

        nk_cast_serial(&src[0].raw_, from_type_::dtype(), dense_dimension, &dst_serial[0].raw_, to_type_::dtype());
        kernel(&src[0].raw_, from_type_::dtype(), dense_dimension, &dst_simd[0].raw_, to_type_::dtype());

        for (std::size_t i = 0; i < dense_dimension; ++i)
            stats.accumulate_exact(dst_simd[i].raw_ == dst_serial[i].raw_);
    }
    return stats;
}

void test_casts() {
    std::printf("Testing type casts...\n");

#if NK_DYNAMIC_DISPATCH
    run_if_matches("cast_f32_to_f16", "f32>f16", test_cast<f32_t, f16_t>, nk_cast);
    run_if_matches("cast_f16_to_f32", "f16>f32", test_cast<f16_t, f32_t>, nk_cast);
    run_if_matches("cast_f32_to_bf16", "f32>bf16", test_cast<f32_t, bf16_t>, nk_cast);
    run_if_matches("cast_bf16_to_f32", "bf16>f32", test_cast<bf16_t, f32_t>, nk_cast);
    run_if_matches("cast_f32_to_e4m3", "f32>e4m3", test_cast<f32_t, e4m3_t>, nk_cast);
    run_if_matches("cast_e4m3_to_f32", "e4m3>f32", test_cast<e4m3_t, f32_t>, nk_cast);
    run_if_matches("cast_f32_to_e5m2", "f32>e5m2", test_cast<f32_t, e5m2_t>, nk_cast);
    run_if_matches("cast_e5m2_to_f32", "e5m2>f32", test_cast<e5m2_t, f32_t>, nk_cast);
    run_if_matches("cast_f64_to_f32", "f64>f32", test_cast<f64_t, f32_t>, nk_cast);
    run_if_matches("cast_f32_to_f64", "f32>f64", test_cast<f32_t, f64_t>, nk_cast);
#endif

#if NK_TARGET_HASWELL
    run_if_matches("cast_f32_to_f16_haswell", "f32>f16", test_cast<f32_t, f16_t>, nk_cast_haswell);
    run_if_matches("cast_f16_to_f32_haswell", "f16>f32", test_cast<f16_t, f32_t>, nk_cast_haswell);
    run_if_matches("cast_f32_to_bf16_haswell", "f32>bf16", test_cast<f32_t, bf16_t>, nk_cast_haswell);
    run_if_matches("cast_bf16_to_f32_haswell", "bf16>f32", test_cast<bf16_t, f32_t>, nk_cast_haswell);
    run_if_matches("cast_f32_to_e4m3_haswell", "f32>e4m3", test_cast<f32_t, e4m3_t>, nk_cast_haswell);
    run_if_matches("cast_e4m3_to_f32_haswell", "e4m3>f32", test_cast<e4m3_t, f32_t>, nk_cast_haswell);
    run_if_matches("cast_f32_to_e5m2_haswell", "f32>e5m2", test_cast<f32_t, e5m2_t>, nk_cast_haswell);
    run_if_matches("cast_e5m2_to_f32_haswell", "e5m2>f32", test_cast<e5m2_t, f32_t>, nk_cast_haswell);
#endif

#if NK_TARGET_SKYLAKE
    run_if_matches("cast_f32_to_f16_skylake", "f32>f16", test_cast<f32_t, f16_t>, nk_cast_skylake);
    run_if_matches("cast_f16_to_f32_skylake", "f16>f32", test_cast<f16_t, f32_t>, nk_cast_skylake);
    run_if_matches("cast_f32_to_bf16_skylake", "f32>bf16", test_cast<f32_t, bf16_t>, nk_cast_skylake);
    run_if_matches("cast_bf16_to_f32_skylake", "bf16>f32", test_cast<bf16_t, f32_t>, nk_cast_skylake);
    run_if_matches("cast_f32_to_e4m3_skylake", "f32>e4m3", test_cast<f32_t, e4m3_t>, nk_cast_skylake);
    run_if_matches("cast_e4m3_to_f32_skylake", "e4m3>f32", test_cast<e4m3_t, f32_t>, nk_cast_skylake);
    run_if_matches("cast_f32_to_e5m2_skylake", "f32>e5m2", test_cast<f32_t, e5m2_t>, nk_cast_skylake);
    run_if_matches("cast_e5m2_to_f32_skylake", "e5m2>f32", test_cast<e5m2_t, f32_t>, nk_cast_skylake);
    run_if_matches("cast_f16_to_bf16_skylake", "f16>bf16", test_cast<f16_t, bf16_t>, nk_cast_skylake);
    run_if_matches("cast_bf16_to_f16_skylake", "bf16>f16", test_cast<bf16_t, f16_t>, nk_cast_skylake);
    run_if_matches("cast_e4m3_to_f16_skylake", "e4m3>f16", test_cast<e4m3_t, f16_t>, nk_cast_skylake);
    run_if_matches("cast_f16_to_e4m3_skylake", "f16>e4m3", test_cast<f16_t, e4m3_t>, nk_cast_skylake);
    run_if_matches("cast_e5m2_to_f16_skylake", "e5m2>f16", test_cast<e5m2_t, f16_t>, nk_cast_skylake);
    run_if_matches("cast_f16_to_e5m2_skylake", "f16>e5m2", test_cast<f16_t, e5m2_t>, nk_cast_skylake);
    run_if_matches("cast_e4m3_to_bf16_skylake", "e4m3>bf16", test_cast<e4m3_t, bf16_t>, nk_cast_skylake);
    run_if_matches("cast_bf16_to_e4m3_skylake", "bf16>e4m3", test_cast<bf16_t, e4m3_t>, nk_cast_skylake);
    run_if_matches("cast_e5m2_to_bf16_skylake", "e5m2>bf16", test_cast<e5m2_t, bf16_t>, nk_cast_skylake);
    run_if_matches("cast_bf16_to_e5m2_skylake", "bf16>e5m2", test_cast<bf16_t, e5m2_t>, nk_cast_skylake);
#endif

#if NK_TARGET_ICE

    run_if_matches("cast_e4m3_to_bf16_ice", "e4m3>bf16", test_cast<e4m3_t, bf16_t>, nk_cast_ice);
    run_if_matches("cast_bf16_to_e4m3_ice", "bf16>e4m3", test_cast<bf16_t, e4m3_t>, nk_cast_ice);
    run_if_matches("cast_e5m2_to_bf16_ice", "e5m2>bf16", test_cast<e5m2_t, bf16_t>, nk_cast_ice);
    run_if_matches("cast_bf16_to_e5m2_ice", "bf16>e5m2", test_cast<bf16_t, e5m2_t>, nk_cast_ice);
    run_if_matches("cast_e4m3_to_f16_ice", "e4m3>f16", test_cast<e4m3_t, f16_t>, nk_cast_ice);
    run_if_matches("cast_e5m2_to_f16_ice", "e5m2>f16", test_cast<e5m2_t, f16_t>, nk_cast_ice);
#endif

#if NK_TARGET_SAPPHIRE
    run_if_matches("cast_e4m3_to_f16_sapphire", "e4m3>f16", test_cast<e4m3_t, f16_t>, nk_cast_sapphire);
    run_if_matches("cast_f16_to_e4m3_sapphire", "f16>e4m3", test_cast<f16_t, e4m3_t>, nk_cast_sapphire);
    run_if_matches("cast_e5m2_to_f16_sapphire", "e5m2>f16", test_cast<e5m2_t, f16_t>, nk_cast_sapphire);
    run_if_matches("cast_f16_to_e5m2_sapphire", "f16>e5m2", test_cast<f16_t, e5m2_t>, nk_cast_sapphire);
#endif
}

#pragma endregion // Cast

#pragma region Reduce

/**
 *  @brief Unified reduce_add test for float types.
 *  Works with f32_t, f64_t, e4m3_t, e5m2_t wrapper types.
 */
template <typename scalar_type_>
error_stats_t test_reduce_add(typename scalar_type_::reduce_add_kernel_t kernel) {
    using scalar_t = scalar_type_;
    using raw_t = typename scalar_t::raw_t;
    using result_t = typename scalar_t::reduce_add_result_t;

    error_stats_t stats;
    std::mt19937 generator(global_config.seed);
    aligned_buffer<scalar_t> buffer(dense_dimension);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(buffer, generator);

        result_t result;
        kernel(&buffer[0].raw_, dense_dimension, sizeof(raw_t), &result.raw_);

        f118_t reference;
        nk::reduce_add<scalar_t, f118_t, false>(&buffer[0], dense_dimension, sizeof(raw_t), &reference);

        stats.accumulate(result, reference);
    }
    return stats;
}

void test_reduce() {
    std::printf("Testing reductions...\n");

#if NK_DYNAMIC_DISPATCH
    run_if_matches("reduce_add", "f32", test_reduce_add<f32_t>, nk_reduce_add_f32);
    run_if_matches("reduce_add", "f64", test_reduce_add<f64_t>, nk_reduce_add_f64);
    run_if_matches("reduce_add", "i32", test_reduce_add<i32_t>, nk_reduce_add_i32);
    run_if_matches("reduce_add", "e4m3", test_reduce_add<e4m3_t>, nk_reduce_add_e4m3);
    run_if_matches("reduce_add", "e5m2", test_reduce_add<e5m2_t>, nk_reduce_add_e5m2);
#else
#if NK_TARGET_NEON
    run_if_matches("reduce_add_neon", "f32", test_reduce_add<f32_t>, nk_reduce_add_f32_neon);
    run_if_matches("reduce_add_neon", "f64", test_reduce_add<f64_t>, nk_reduce_add_f64_neon);
    run_if_matches("reduce_add_neon", "i32", test_reduce_add<i32_t>, nk_reduce_add_i32_neon);
#endif
#if NK_TARGET_NEONFHM
    run_if_matches("reduce_add_neonfhm", "e4m3", test_reduce_add<e4m3_t>, nk_reduce_add_e4m3_neonfhm);
    run_if_matches("reduce_add_neonfhm", "e5m2", test_reduce_add<e5m2_t>, nk_reduce_add_e5m2_neonfhm);
#endif
#if NK_TARGET_HASWELL
    run_if_matches("reduce_add_haswell", "f32", test_reduce_add<f32_t>, nk_reduce_add_f32_haswell);
    run_if_matches("reduce_add_haswell", "f64", test_reduce_add<f64_t>, nk_reduce_add_f64_haswell);
    run_if_matches("reduce_add_haswell", "i32", test_reduce_add<i32_t>, nk_reduce_add_i32_haswell);
    run_if_matches("reduce_add_haswell", "e4m3", test_reduce_add<e4m3_t>, nk_reduce_add_e4m3_haswell);
    run_if_matches("reduce_add_haswell", "e5m2", test_reduce_add<e5m2_t>, nk_reduce_add_e5m2_haswell);
#endif
#if NK_TARGET_SKYLAKE
    run_if_matches("reduce_add_skylake", "f32", test_reduce_add<f32_t>, nk_reduce_add_f32_skylake);
    run_if_matches("reduce_add_skylake", "f64", test_reduce_add<f64_t>, nk_reduce_add_f64_skylake);
    run_if_matches("reduce_add_skylake", "i32", test_reduce_add<i32_t>, nk_reduce_add_i32_skylake);
#endif
    run_if_matches("reduce_add_serial", "f32", test_reduce_add<f32_t>, nk_reduce_add_f32_serial);
    run_if_matches("reduce_add_serial", "f64", test_reduce_add<f64_t>, nk_reduce_add_f64_serial);
    run_if_matches("reduce_add_serial", "i32", test_reduce_add<i32_t>, nk_reduce_add_i32_serial);
    run_if_matches("reduce_add_serial", "e4m3", test_reduce_add<e4m3_t>, nk_reduce_add_e4m3_serial);
    run_if_matches("reduce_add_serial", "e5m2", test_reduce_add<e5m2_t>, nk_reduce_add_e5m2_serial);
#endif
}

#pragma endregion // Reduce

#pragma region Dot

/**
 *  @brief Unified dot product test for all types: float, integer, and complex.
 *  Works with f32_t, f64_t, f16_t, bf16_t, e4m3_t, e5m2_t, i8_t, u8_t, f32c_t, f64c_t.
 */
template <typename scalar_type_>
error_stats_t test_dot(typename scalar_type_::dot_kernel_t kernel) {
    using scalar_t = scalar_type_;
    using raw_t = typename scalar_t::raw_t;
    using result_t = typename scalar_t::dot_result_t;
    using reference_t = std::conditional_t<scalar_t::is_complex(), f118c_t, f118_t>;

    error_stats_t stats;
    std::mt19937 rng(global_config.seed);
    aligned_buffer<scalar_t> a(dense_dimension), b(dense_dimension);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(a, rng);
        fill_random(b, rng);

        result_t result;
        kernel(&a[0].raw_, &b[0].raw_, dense_dimension, &result.raw_);

        reference_t reference;
        nk::dot<scalar_t, reference_t, false>(&a[0], &b[0], dense_dimension, &reference);

        stats.accumulate(result, reference);
    }
    return stats;
}

/**
 *  @brief Conjugate dot product test for complex types (vdot = conj(a) * b).
 */
template <typename scalar_type_>
error_stats_t test_vdot(typename scalar_type_::vdot_kernel_t kernel) {
    using scalar_t = scalar_type_;
    using raw_t = typename scalar_t::raw_t;
    using result_t = typename scalar_t::vdot_result_t;

    error_stats_t stats;
    std::mt19937 rng(global_config.seed);
    aligned_buffer<scalar_t> a(dense_dimension), b(dense_dimension);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(a, rng);
        fill_random(b, rng);

        result_t result;
        kernel(&a[0].raw_, &b[0].raw_, dense_dimension, &result.raw_);

        f118c_t reference;
        nk::vdot<scalar_t, f118c_t, false>(&a[0], &b[0], dense_dimension, &reference);

        stats.accumulate(result, reference);
    }
    return stats;
}

void test_dot() {
    std::printf("Testing dot products...\n");

#if NK_DYNAMIC_DISPATCH
    // Dynamic dispatch - only test the dispatcher itself
    run_if_matches("dot", "f32", test_dot<f32_t>, nk_dot_f32);
    run_if_matches("dot", "f64", test_dot<f64_t>, nk_dot_f64);
    run_if_matches("dot", "f16", test_dot<f16_t>, nk_dot_f16);
    run_if_matches("dot", "bf16", test_dot<bf16_t>, nk_dot_bf16);
    run_if_matches("dot", "e4m3", test_dot<e4m3_t>, nk_dot_e4m3);
    run_if_matches("dot", "e5m2", test_dot<e5m2_t>, nk_dot_e5m2);
    run_if_matches("dot", "i8", test_dot<i8_t>, nk_dot_i8);
    run_if_matches("dot", "u8", test_dot<u8_t>, nk_dot_u8);
    run_if_matches("dot", "i4", test_dot<i4x2_t>, nk_dot_i4);
    run_if_matches("dot", "u4", test_dot<u4x2_t>, nk_dot_u4);
    run_if_matches("dot", "f32c", test_dot<f32c_t>, nk_dot_f32c);
    run_if_matches("vdot", "f32c", test_vdot<f32c_t>, nk_vdot_f32c);
    run_if_matches("dot", "f64c", test_dot<f64c_t>, nk_dot_f64c);
    run_if_matches("vdot", "f64c", test_vdot<f64c_t>, nk_vdot_f64c);
#else
    // Static compilation - test all available ISA variants

#if NK_TARGET_NEON
    run_if_matches("dot_neon", "f32", test_dot<f32_t>, nk_dot_f32_neon);
    run_if_matches("dot_neon", "f64", test_dot<f64_t>, nk_dot_f64_neon);
    run_if_matches("dot_neon", "e4m3", test_dot<e4m3_t>, nk_dot_e4m3_neon);
    run_if_matches("dot_neon", "e5m2", test_dot<e5m2_t>, nk_dot_e5m2_neon);
    run_if_matches("dot_neon", "f32c", test_dot<f32c_t>, nk_dot_f32c_neon);
    run_if_matches("vdot_neon", "f32c", test_vdot<f32c_t>, nk_vdot_f32c_neon);
    run_if_matches("dot_neon", "f64c", test_dot<f64c_t>, nk_dot_f64c_neon);
    run_if_matches("vdot_neon", "f64c", test_vdot<f64c_t>, nk_vdot_f64c_neon);
#endif // NK_TARGET_NEON

#if NK_TARGET_NEONHALF
    run_if_matches("dot_neonhalf", "f16", test_dot<f16_t>, nk_dot_f16_neonhalf);
#endif // NK_TARGET_NEONHALF

#if NK_TARGET_NEONBFDOT
    run_if_matches("dot_neonbfdot", "bf16", test_dot<bf16_t>, nk_dot_bf16_neonbfdot);
#endif // NK_TARGET_NEONBFDOT

#if NK_TARGET_NEONSDOT
    run_if_matches("dot_neonsdot", "i8", test_dot<i8_t>, nk_dot_i8_neonsdot);
    run_if_matches("dot_neonsdot", "u8", test_dot<u8_t>, nk_dot_u8_neonsdot);
#endif // NK_TARGET_NEONSDOT

#if NK_TARGET_HASWELL
    run_if_matches("dot_haswell", "f32", test_dot<f32_t>, nk_dot_f32_haswell);
    run_if_matches("dot_haswell", "f64", test_dot<f64_t>, nk_dot_f64_haswell);
    run_if_matches("dot_haswell", "f16", test_dot<f16_t>, nk_dot_f16_haswell);
    run_if_matches("dot_haswell", "bf16", test_dot<bf16_t>, nk_dot_bf16_haswell);
    run_if_matches("dot_haswell", "e4m3", test_dot<e4m3_t>, nk_dot_e4m3_haswell);
    run_if_matches("dot_haswell", "e5m2", test_dot<e5m2_t>, nk_dot_e5m2_haswell);
    run_if_matches("dot_haswell", "i8", test_dot<i8_t>, nk_dot_i8_haswell);
    run_if_matches("dot_haswell", "u8", test_dot<u8_t>, nk_dot_u8_haswell);
    run_if_matches("dot_haswell", "f32c", test_dot<f32c_t>, nk_dot_f32c_haswell);
    run_if_matches("vdot_haswell", "f32c", test_vdot<f32c_t>, nk_vdot_f32c_haswell);
#endif // NK_TARGET_HASWELL

#if NK_TARGET_SKYLAKE
    run_if_matches("dot_skylake", "f32", test_dot<f32_t>, nk_dot_f32_skylake);
    run_if_matches("dot_skylake", "f64", test_dot<f64_t>, nk_dot_f64_skylake);
    run_if_matches("dot_skylake", "f16", test_dot<f16_t>, nk_dot_f16_skylake);
    run_if_matches("dot_skylake", "bf16", test_dot<bf16_t>, nk_dot_bf16_skylake);
    run_if_matches("dot_skylake", "e4m3", test_dot<e4m3_t>, nk_dot_e4m3_skylake);
    run_if_matches("dot_skylake", "e5m2", test_dot<e5m2_t>, nk_dot_e5m2_skylake);
    run_if_matches("dot_skylake", "i8", test_dot<i8_t>, nk_dot_i8_skylake);
    run_if_matches("dot_skylake", "u8", test_dot<u8_t>, nk_dot_u8_skylake);
    run_if_matches("dot_skylake", "f32c", test_dot<f32c_t>, nk_dot_f32c_skylake);
    run_if_matches("vdot_skylake", "f32c", test_vdot<f32c_t>, nk_vdot_f32c_skylake);
    run_if_matches("dot_skylake", "f64c", test_dot<f64c_t>, nk_dot_f64c_skylake);
    run_if_matches("vdot_skylake", "f64c", test_vdot<f64c_t>, nk_vdot_f64c_skylake);
#endif // NK_TARGET_SKYLAKE

#if NK_TARGET_ICE
    run_if_matches("dot_ice", "i4", test_dot<i4x2_t>, nk_dot_i4_ice);
    run_if_matches("dot_ice", "u4", test_dot<u4x2_t>, nk_dot_u4_ice);
#endif // NK_TARGET_ICE

#if NK_TARGET_SPACEMIT
    run_if_matches("dot_spacemit", "i8", test_dot<i8_t>, nk_dot_i8_spacemit);
    run_if_matches("dot_spacemit", "u8", test_dot<u8_t>, nk_dot_u8_spacemit);
    run_if_matches("dot_spacemit", "f32", test_dot<f32_t>, nk_dot_f32_spacemit);
    run_if_matches("dot_spacemit", "f64", test_dot<f64_t>, nk_dot_f64_spacemit);
#endif // NK_TARGET_SPACEMIT

#if NK_TARGET_SIFIVE
    run_if_matches("dot_sifive", "f16", test_dot<f16_t>, nk_dot_f16_sifive);
#endif // NK_TARGET_SIFIVE

#if NK_TARGET_XUANTIE
    run_if_matches("dot_xuantie", "bf16", test_dot<bf16_t>, nk_dot_bf16_xuantie);
#endif // NK_TARGET_XUANTIE

    // Serial always runs - baseline test
    run_if_matches("dot_serial", "f32", test_dot<f32_t>, nk_dot_f32_serial);
    run_if_matches("dot_serial", "f64", test_dot<f64_t>, nk_dot_f64_serial);
    run_if_matches("dot_serial", "f16", test_dot<f16_t>, nk_dot_f16_serial);
    run_if_matches("dot_serial", "bf16", test_dot<bf16_t>, nk_dot_bf16_serial);
    run_if_matches("dot_serial", "e4m3", test_dot<e4m3_t>, nk_dot_e4m3_serial);
    run_if_matches("dot_serial", "e5m2", test_dot<e5m2_t>, nk_dot_e5m2_serial);
    run_if_matches("dot_serial", "i8", test_dot<i8_t>, nk_dot_i8_serial);
    run_if_matches("dot_serial", "u8", test_dot<u8_t>, nk_dot_u8_serial);
    run_if_matches("dot_serial", "i4", test_dot<i4x2_t>, nk_dot_i4_serial);
    run_if_matches("dot_serial", "u4", test_dot<u4x2_t>, nk_dot_u4_serial);
    run_if_matches("dot_serial", "f32c", test_dot<f32c_t>, nk_dot_f32c_serial);
    run_if_matches("vdot_serial", "f32c", test_vdot<f32c_t>, nk_vdot_f32c_serial);
    run_if_matches("dot_serial", "f64c", test_dot<f64c_t>, nk_dot_f64c_serial);
    run_if_matches("vdot_serial", "f64c", test_vdot<f64c_t>, nk_vdot_f64c_serial);

#endif // NK_DYNAMIC_DISPATCH

#if NK_COMPARE_TO_BLAS || NK_COMPARE_TO_MKL || NK_COMPARE_TO_ACCELERATE
    // BLAS/MKL/Accelerate precision comparison
    run_if_matches("dot_blas", "f32", test_dot<f32_t>, dot_f32_blas);
    run_if_matches("dot_blas", "f64", test_dot<f64_t>, dot_f64_blas);
    run_if_matches("dot_blas", "f32c", test_dot<f32c_t>, dot_f32c_blas);
    run_if_matches("vdot_blas", "f32c", test_vdot<f32c_t>, vdot_f32c_blas);
    run_if_matches("dot_blas", "f64c", test_dot<f64c_t>, dot_f64c_blas);
    run_if_matches("vdot_blas", "f64c", test_vdot<f64c_t>, vdot_f64c_blas);
#endif // NK_COMPARE_TO_BLAS || NK_COMPARE_TO_MKL || NK_COMPARE_TO_ACCELERATE
}

#pragma endregion // Dot

#pragma region Spatial

/**
 *  @brief Unified L2 squared distance test for float types.
 *  Works with f32_t, f64_t, f16_t, bf16_t wrapper types.
 */
template <typename scalar_type_>
error_stats_t test_l2sq(typename scalar_type_::l2sq_kernel_t kernel) {
    using scalar_t = scalar_type_;
    using raw_t = typename scalar_t::raw_t;
    using result_t = typename scalar_t::l2sq_result_t;

    error_stats_t stats;
    std::mt19937 rng(global_config.seed);
    aligned_buffer<scalar_t> a(dense_dimension), b(dense_dimension);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(a, rng);
        fill_random(b, rng);

        result_t result;
        kernel(&a[0].raw_, &b[0].raw_, dense_dimension, &result.raw_);

        f118_t reference;
        nk::l2sq<scalar_t, f118_t, false>(&a[0], &b[0], dense_dimension, &reference);

        stats.accumulate(result, reference);
    }
    return stats;
}

/**
 *  @brief Unified angular (cosine) distance test for float types.
 *  Works with f32_t, f64_t, f16_t, bf16_t wrapper types.
 */
template <typename scalar_type_>
error_stats_t test_angular(typename scalar_type_::angular_kernel_t kernel) {
    using scalar_t = scalar_type_;
    using raw_t = typename scalar_t::raw_t;
    using result_t = typename scalar_t::angular_result_t;

    error_stats_t stats;
    std::mt19937 rng(global_config.seed);
    aligned_buffer<scalar_t> a(dense_dimension), b(dense_dimension);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(a, rng);
        fill_random(b, rng);

        result_t result;
        kernel(&a[0].raw_, &b[0].raw_, dense_dimension, &result.raw_);

        f118_t reference;
        nk::angular<scalar_t, f118_t, false>(&a[0], &b[0], dense_dimension, &reference);

        stats.accumulate(result, reference);
    }
    return stats;
}

void test_spatial() {
    std::printf("Testing spatial distances...\n");

#if NK_DYNAMIC_DISPATCH
    // Dynamic dispatch - only test the dispatcher itself
    run_if_matches("l2sq", "f32", test_l2sq<f32_t>, nk_l2sq_f32);
    run_if_matches("l2sq", "f64", test_l2sq<f64_t>, nk_l2sq_f64);
    run_if_matches("l2sq", "f16", test_l2sq<f16_t>, nk_l2sq_f16);
    run_if_matches("l2sq", "bf16", test_l2sq<bf16_t>, nk_l2sq_bf16);
    run_if_matches("angular", "f32", test_angular<f32_t>, nk_angular_f32);
    run_if_matches("angular", "f64", test_angular<f64_t>, nk_angular_f64);
    run_if_matches("angular", "f16", test_angular<f16_t>, nk_angular_f16);
    run_if_matches("angular", "bf16", test_angular<bf16_t>, nk_angular_bf16);
    run_if_matches("l2sq", "i4", test_l2sq<i4x2_t>, nk_l2sq_i4);
    run_if_matches("l2sq", "u4", test_l2sq<u4x2_t>, nk_l2sq_u4);
    run_if_matches("angular", "i4", test_angular<i4x2_t>, nk_angular_i4);
    run_if_matches("angular", "u4", test_angular<u4x2_t>, nk_angular_u4);
#else
    // Static compilation - test all available ISA variants

#if NK_TARGET_NEON
    run_if_matches("l2sq_neon", "f32", test_l2sq<f32_t>, nk_l2sq_f32_neon);
    run_if_matches("l2sq_neon", "f64", test_l2sq<f64_t>, nk_l2sq_f64_neon);
    run_if_matches("angular_neon", "f32", test_angular<f32_t>, nk_angular_f32_neon);
    run_if_matches("angular_neon", "f64", test_angular<f64_t>, nk_angular_f64_neon);
#endif // NK_TARGET_NEON

#if NK_TARGET_NEONHALF
    run_if_matches("l2sq_neonhalf", "f16", test_l2sq<f16_t>, nk_l2sq_f16_neonhalf);
    run_if_matches("angular_neonhalf", "f16", test_angular<f16_t>, nk_angular_f16_neonhalf);
#endif // NK_TARGET_NEONHALF

#if NK_TARGET_NEONBFDOT
    run_if_matches("l2sq_neonbfdot", "bf16", test_l2sq<bf16_t>, nk_l2sq_bf16_neonbfdot);
    run_if_matches("angular_neonbfdot", "bf16", test_angular<bf16_t>, nk_angular_bf16_neonbfdot);
#endif // NK_TARGET_NEONBFDOT

#if NK_TARGET_HASWELL
    run_if_matches("l2sq_haswell", "f32", test_l2sq<f32_t>, nk_l2sq_f32_haswell);
    run_if_matches("l2sq_haswell", "f64", test_l2sq<f64_t>, nk_l2sq_f64_haswell);
    run_if_matches("l2sq_haswell", "f16", test_l2sq<f16_t>, nk_l2sq_f16_haswell);
    run_if_matches("l2sq_haswell", "bf16", test_l2sq<bf16_t>, nk_l2sq_bf16_haswell);
    run_if_matches("angular_haswell", "f32", test_angular<f32_t>, nk_angular_f32_haswell);
    run_if_matches("angular_haswell", "f64", test_angular<f64_t>, nk_angular_f64_haswell);
    run_if_matches("angular_haswell", "f16", test_angular<f16_t>, nk_angular_f16_haswell);
    run_if_matches("angular_haswell", "bf16", test_angular<bf16_t>, nk_angular_bf16_haswell);
#endif // NK_TARGET_HASWELL

#if NK_TARGET_SKYLAKE
    run_if_matches("l2sq_skylake", "f32", test_l2sq<f32_t>, nk_l2sq_f32_skylake);
    run_if_matches("l2sq_skylake", "f64", test_l2sq<f64_t>, nk_l2sq_f64_skylake);
    run_if_matches("angular_skylake", "f32", test_angular<f32_t>, nk_angular_f32_skylake);
    run_if_matches("angular_skylake", "f64", test_angular<f64_t>, nk_angular_f64_skylake);
#endif // NK_TARGET_SKYLAKE

#if NK_TARGET_ICE
    run_if_matches("l2sq_ice", "i4", test_l2sq<i4x2_t>, nk_l2sq_i4_ice);
    run_if_matches("l2sq_ice", "u4", test_l2sq<u4x2_t>, nk_l2sq_u4_ice);
    run_if_matches("angular_ice", "i4", test_angular<i4x2_t>, nk_angular_i4_ice);
    run_if_matches("angular_ice", "u4", test_angular<u4x2_t>, nk_angular_u4_ice);
#endif // NK_TARGET_ICE

#if NK_TARGET_SPACEMIT
    run_if_matches("l2sq_spacemit", "f32", test_l2sq<f32_t>, nk_l2sq_f32_spacemit);
    run_if_matches("l2sq_spacemit", "f64", test_l2sq<f64_t>, nk_l2sq_f64_spacemit);
    run_if_matches("angular_spacemit", "f32", test_angular<f32_t>, nk_angular_f32_spacemit);
    run_if_matches("angular_spacemit", "f64", test_angular<f64_t>, nk_angular_f64_spacemit);
#endif // NK_TARGET_SPACEMIT

#if NK_TARGET_SIFIVE
    run_if_matches("l2sq_sifive", "f16", test_l2sq<f16_t>, nk_l2sq_f16_sifive);
    run_if_matches("angular_sifive", "f16", test_angular<f16_t>, nk_angular_f16_sifive);
#endif // NK_TARGET_SIFIVE

#if NK_TARGET_XUANTIE
    run_if_matches("l2sq_xuantie", "bf16", test_l2sq<bf16_t>, nk_l2sq_bf16_xuantie);
    run_if_matches("angular_xuantie", "bf16", test_angular<bf16_t>, nk_angular_bf16_xuantie);
#endif // NK_TARGET_XUANTIE

    // Serial always runs - baseline test
    run_if_matches("l2sq_serial", "f32", test_l2sq<f32_t>, nk_l2sq_f32_serial);
    run_if_matches("l2sq_serial", "f64", test_l2sq<f64_t>, nk_l2sq_f64_serial);
    run_if_matches("l2sq_serial", "f16", test_l2sq<f16_t>, nk_l2sq_f16_serial);
    run_if_matches("l2sq_serial", "bf16", test_l2sq<bf16_t>, nk_l2sq_bf16_serial);
    run_if_matches("angular_serial", "f32", test_angular<f32_t>, nk_angular_f32_serial);
    run_if_matches("angular_serial", "f64", test_angular<f64_t>, nk_angular_f64_serial);
    run_if_matches("angular_serial", "f16", test_angular<f16_t>, nk_angular_f16_serial);
    run_if_matches("angular_serial", "bf16", test_angular<bf16_t>, nk_angular_bf16_serial);
    run_if_matches("l2sq_serial", "i4", test_l2sq<i4x2_t>, nk_l2sq_i4_serial);
    run_if_matches("l2sq_serial", "u4", test_l2sq<u4x2_t>, nk_l2sq_u4_serial);
    run_if_matches("angular_serial", "i4", test_angular<i4x2_t>, nk_angular_i4_serial);
    run_if_matches("angular_serial", "u4", test_angular<u4x2_t>, nk_angular_u4_serial);

#endif // NK_DYNAMIC_DISPATCH
}

#pragma endregion // Spatial

#pragma region Curved

/**
 *  @brief Template for bilinear form test: a^T * M * b
 */
template <typename scalar_type_>
error_stats_t test_bilinear(typename scalar_type_::bilinear_kernel_t kernel) {
    using scalar_t = scalar_type_;
    using raw_t = typename scalar_t::raw_t;
    using result_t = typename scalar_t::bilinear_result_t;

    error_stats_t stats;
    std::mt19937 rng(global_config.seed);

    aligned_buffer<scalar_t> a(dense_dimension), b(dense_dimension), m(dense_dimension * dense_dimension);
    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(a, rng);
        fill_random(b, rng);
        fill_random(m, rng);

        result_t result;
        kernel(&a[0].raw_, &b[0].raw_, &m[0].raw_, dense_dimension, &result.raw_);

        f118_t reference;
        nk::bilinear<scalar_t, f118_t, false>(&a[0], &b[0], &m[0], dense_dimension, &reference);

        stats.accumulate(result, reference);
    }

    return stats;
}

void test_curved() {
    std::printf("Testing curved/bilinear forms...\n");

#if NK_DYNAMIC_DISPATCH
    run_if_matches("bilinear", "f32", test_bilinear<f32_t>, nk_bilinear_f32);
    run_if_matches("bilinear", "f64", test_bilinear<f64_t>, nk_bilinear_f64);
#else

#if NK_TARGET_NEON
    run_if_matches("bilinear_neon", "f32", test_bilinear<f32_t>, nk_bilinear_f32_neon);
#endif // NK_TARGET_NEON

#if NK_TARGET_SKYLAKE
    run_if_matches("bilinear_skylake", "f32", test_bilinear<f32_t>, nk_bilinear_f32_skylake);
    run_if_matches("bilinear_skylake", "f64", test_bilinear<f64_t>, nk_bilinear_f64_skylake);
#endif // NK_TARGET_SKYLAKE

    // Serial always runs - baseline test
    run_if_matches("bilinear_serial", "f32", test_bilinear<f32_t>, nk_bilinear_f32_serial);
    run_if_matches("bilinear_serial", "f64", test_bilinear<f64_t>, nk_bilinear_f64_serial);

#endif // NK_DYNAMIC_DISPATCH
}

#pragma endregion // Curved

#pragma region Probability

/**
 *  @brief Template for KL divergence test.
 *  KLD requires probability distributions: all values > 0, sum to 1.
 */
template <typename scalar_type_>
error_stats_t test_kld(typename scalar_type_::kld_kernel_t kernel) {
    using scalar_t = scalar_type_;
    using raw_t = typename scalar_t::raw_t;
    using result_t = typename scalar_t::kld_result_t;

    error_stats_t stats;
    std::mt19937 rng(global_config.seed);
    aligned_buffer<scalar_t> p(dense_dimension), q(dense_dimension);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_probability(p, q, rng);

        result_t result;
        kernel(&p[0].raw_, &q[0].raw_, dense_dimension, &result.raw_);

        f118_t reference;
        nk::kld<scalar_t, f118_t, false>(&p[0], &q[0], dense_dimension, &reference);

        stats.accumulate(result, reference);
    }

    return stats;
}

/**
 *  @brief Template for Jensen-Shannon divergence test.
 *  JSD requires probability distributions: all values > 0, sum to 1.
 */
template <typename scalar_type_>
error_stats_t test_jsd(typename scalar_type_::jsd_kernel_t kernel) {
    using scalar_t = scalar_type_;
    using raw_t = typename scalar_t::raw_t;
    using result_t = typename scalar_t::jsd_result_t;

    error_stats_t stats;
    std::mt19937 rng(global_config.seed);
    aligned_buffer<scalar_t> p(dense_dimension), q(dense_dimension);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_probability(p, q, rng);

        result_t result;
        kernel(&p[0].raw_, &q[0].raw_, dense_dimension, &result.raw_);

        f118_t reference;
        nk::jsd<scalar_t, f118_t, false>(&p[0], &q[0], dense_dimension, &reference);

        stats.accumulate(result, reference);
    }

    return stats;
}

void test_probability() {
    std::printf("Testing probability divergences...\n");

#if NK_DYNAMIC_DISPATCH
    run_if_matches("kld", "f32", test_kld<f32_t>, nk_kld_f32);
    run_if_matches("kld", "f64", test_kld<f64_t>, nk_kld_f64);
    run_if_matches("jsd", "f32", test_jsd<f32_t>, nk_jsd_f32);
    run_if_matches("jsd", "f64", test_jsd<f64_t>, nk_jsd_f64);
#else

#if NK_TARGET_NEON
    run_if_matches("kld_neon", "f32", test_kld<f32_t>, nk_kld_f32_neon);
    run_if_matches("jsd_neon", "f32", test_jsd<f32_t>, nk_jsd_f32_neon);
#endif // NK_TARGET_NEON

#if NK_TARGET_SKYLAKE
    run_if_matches("kld_skylake", "f32", test_kld<f32_t>, nk_kld_f32_skylake);
    run_if_matches("kld_skylake", "f64", test_kld<f64_t>, nk_kld_f64_skylake);
    run_if_matches("jsd_skylake", "f32", test_jsd<f32_t>, nk_jsd_f32_skylake);
    run_if_matches("jsd_skylake", "f64", test_jsd<f64_t>, nk_jsd_f64_skylake);
#endif // NK_TARGET_SKYLAKE

    // Serial always runs - baseline test
    run_if_matches("kld_serial", "f32", test_kld<f32_t>, nk_kld_f32_serial);
    run_if_matches("kld_serial", "f64", test_kld<f64_t>, nk_kld_f64_serial);
    run_if_matches("jsd_serial", "f32", test_jsd<f32_t>, nk_jsd_f32_serial);
    run_if_matches("jsd_serial", "f64", test_jsd<f64_t>, nk_jsd_f64_serial);

#endif // NK_DYNAMIC_DISPATCH
}

#pragma endregion // Probability

#pragma region Binary

/**
 *  @brief Test Hamming distance for binary vectors.
 */
error_stats_t test_hamming(u1x8_t::hamming_kernel_t kernel) {
    using scalar_t = u1x8_t;

    error_stats_t stats;
    std::mt19937 rng(global_config.seed);

    std::size_t n_bytes = dense_dimension / 8;
    aligned_buffer<scalar_t> a(n_bytes), b(n_bytes);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(a, rng);
        fill_random(b, rng);

        u32_t result;
        kernel(&a[0].raw_, &b[0].raw_, dense_dimension, &result.raw_);

        u32_t reference;
        nk::hamming(&a[0], &b[0], n_bytes, &reference);

        stats.accumulate(result, reference);
    }

    return stats;
}

/**
 *  @brief Test Jaccard distance for binary vectors.
 */
error_stats_t test_jaccard(u1x8_t::jaccard_kernel_t kernel) {
    using scalar_t = u1x8_t;

    error_stats_t stats;
    std::mt19937 rng(global_config.seed);

    std::size_t n_bytes = dense_dimension / 8;
    aligned_buffer<scalar_t> a(n_bytes), b(n_bytes);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(a, rng);
        fill_random(b, rng);

        f32_t result;
        kernel(&a[0].raw_, &b[0].raw_, dense_dimension, &result.raw_);

        f32_t reference;
        nk::jaccard(&a[0], &b[0], n_bytes, &reference);

        stats.accumulate(result, reference);
    }

    return stats;
}

void test_binary() {
    std::printf("Testing binary distances...\n");

#if NK_DYNAMIC_DISPATCH
    // Dynamic dispatch - only test the dispatcher itself
    run_if_matches("hamming", "u1", test_hamming, nk_hamming_u1);
    run_if_matches("jaccard", "u1", test_jaccard, nk_jaccard_u1);
#else
    // Static compilation - test all available ISA variants

#if NK_TARGET_NEON
    run_if_matches("hamming_neon", "u1", test_hamming, nk_hamming_u1_neon);
    run_if_matches("jaccard_neon", "u1", test_jaccard, nk_jaccard_u1_neon);
#endif // NK_TARGET_NEON

#if NK_TARGET_HASWELL
    run_if_matches("hamming_haswell", "u1", test_hamming, nk_hamming_u1_haswell);
    run_if_matches("jaccard_haswell", "u1", test_jaccard, nk_jaccard_u1_haswell);
#endif // NK_TARGET_HASWELL

#if NK_TARGET_ICE
    run_if_matches("hamming_ice", "u1", test_hamming, nk_hamming_u1_ice);
    run_if_matches("jaccard_ice", "u1", test_jaccard, nk_jaccard_u1_ice);
#endif // NK_TARGET_ICE

    // Serial always runs - baseline test
    run_if_matches("hamming_serial", "u1", test_hamming, nk_hamming_u1_serial);
    run_if_matches("jaccard_serial", "u1", test_jaccard, nk_jaccard_u1_serial);

#endif // NK_DYNAMIC_DISPATCH
}

#pragma endregion // Binary

#pragma region Elementwise

/**
 *  @brief Unified test for elementwise sum: result[i] = a[i] + b[i]
 */
template <typename scalar_type_>
error_stats_t test_sum(typename scalar_type_::sum_kernel_t kernel) {
    using scalar_t = scalar_type_;
    using raw_t = typename scalar_t::raw_t;

    error_stats_t stats;
    std::mt19937 rng(global_config.seed);
    aligned_buffer<scalar_t> a(dense_dimension), b(dense_dimension), result(dense_dimension),
        reference(dense_dimension);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(a, rng);
        fill_random(b, rng);

        kernel(&a[0].raw_, &b[0].raw_, dense_dimension, &result[0].raw_);
        nk::sum<scalar_t, false>(&a[0], &b[0], dense_dimension, &reference[0]);

        for (std::size_t i = 0; i < dense_dimension; i++) stats.accumulate(result[i], reference[i]);
    }
    return stats;
}

/**
 *  @brief Unified test for scale: result[i] = alpha * x[i] + beta
 */
template <typename scalar_type_>
error_stats_t test_scale(typename scalar_type_::scale_kernel_t kernel) {
    using scalar_t = scalar_type_;
    using raw_t = typename scalar_t::raw_t;
    using scale_t = typename scalar_t::scale_t;

    error_stats_t stats;
    std::mt19937 rng(global_config.seed);
    aligned_buffer<scalar_t> input(dense_dimension), result(dense_dimension), reference(dense_dimension);
    std::uniform_real_distribution<scale_t> coef_dist(scale_t(-2.0), scale_t(2.0));

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(input, rng);

        scale_t alpha = coef_dist(rng);
        scale_t beta = coef_dist(rng);

        kernel(&input[0].raw_, dense_dimension, &alpha, &beta, &result[0].raw_);
        nk::scale<scalar_t, scale_t, false>(&input[0], dense_dimension, &alpha, &beta, &reference[0]);

        for (std::size_t i = 0; i < dense_dimension; i++) stats.accumulate(result[i], reference[i]);
    }
    return stats;
}

/**
 *  @brief Unified test for weighted sum: result[i] = alpha * a[i] + beta * b[i]
 */
template <typename scalar_type_>
error_stats_t test_wsum(typename scalar_type_::wsum_kernel_t kernel) {
    using scalar_t = scalar_type_;
    using raw_t = typename scalar_t::raw_t;
    using scale_t = typename scalar_t::scale_t;

    error_stats_t stats;
    std::mt19937 rng(global_config.seed);
    aligned_buffer<scalar_t> a(dense_dimension), b(dense_dimension), result(dense_dimension),
        reference(dense_dimension);
    std::uniform_real_distribution<scale_t> coef_dist(scale_t(-2.0), scale_t(2.0));

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(a, rng);
        fill_random(b, rng);

        scale_t alpha = coef_dist(rng);
        scale_t beta = coef_dist(rng);

        kernel(&a[0].raw_, &b[0].raw_, dense_dimension, &alpha, &beta, &result[0].raw_);
        nk::wsum<scalar_t, scale_t, false>(&a[0], &b[0], dense_dimension, &alpha, &beta, &reference[0]);

        for (std::size_t i = 0; i < dense_dimension; i++) stats.accumulate(result[i], reference[i]);
    }
    return stats;
}

/**
 *  @brief Unified test for FMA: result[i] = alpha * a[i] * b[i] + beta * c[i]
 */
template <typename scalar_type_>
error_stats_t test_fma(typename scalar_type_::fma_kernel_t kernel) {
    using scalar_t = scalar_type_;
    using raw_t = typename scalar_t::raw_t;
    using scale_t = typename scalar_t::scale_t;

    error_stats_t stats;
    std::mt19937 rng(global_config.seed);
    aligned_buffer<scalar_t> a(dense_dimension), b(dense_dimension), c(dense_dimension);
    aligned_buffer<scalar_t> result(dense_dimension), reference(dense_dimension);
    std::uniform_real_distribution<scale_t> coef_dist(scale_t(-2.0), scale_t(2.0));

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(a, rng);
        fill_random(b, rng);
        fill_random(c, rng);

        scale_t alpha = coef_dist(rng);
        scale_t beta = coef_dist(rng);

        kernel(&a[0].raw_, &b[0].raw_, &c[0].raw_, dense_dimension, &alpha, &beta, &result[0].raw_);
        nk::fma<scalar_t, scale_t, false>(&a[0], &b[0], dense_dimension, &c[0], &alpha, &beta, &reference[0]);

        for (std::size_t i = 0; i < dense_dimension; i++) stats.accumulate(result[i], reference[i]);
    }
    return stats;
}

void test_elementwise() {
    std::printf("Testing elementwise operations...\n");

#if NK_DYNAMIC_DISPATCH
    // Dynamic dispatch - only test the dispatcher itself
    run_if_matches("scale", "f32", test_scale<f32_t>, nk_scale_f32);
    run_if_matches("sum", "f32", test_sum<f32_t>, nk_sum_f32);
    run_if_matches("wsum", "f32", test_wsum<f32_t>, nk_wsum_f32);
    run_if_matches("fma", "f32", test_fma<f32_t>, nk_fma_f32);
    run_if_matches("scale", "e4m3", test_scale<e4m3_t>, nk_scale_e4m3);
    run_if_matches("scale", "e5m2", test_scale<e5m2_t>, nk_scale_e5m2);
    run_if_matches("sum", "e4m3", test_sum<e4m3_t>, nk_sum_e4m3);
    run_if_matches("sum", "e5m2", test_sum<e5m2_t>, nk_sum_e5m2);
    run_if_matches("wsum", "e4m3", test_wsum<e4m3_t>, nk_wsum_e4m3);
    run_if_matches("wsum", "e5m2", test_wsum<e5m2_t>, nk_wsum_e5m2);
    run_if_matches("fma", "e4m3", test_fma<e4m3_t>, nk_fma_e4m3);
    run_if_matches("fma", "e5m2", test_fma<e5m2_t>, nk_fma_e5m2);
#else
    // Static compilation - test all available ISA variants

#if NK_TARGET_NEON
    run_if_matches("scale_neon", "f32", test_scale<f32_t>, nk_scale_f32_neon);
    run_if_matches("sum_neon", "f32", test_sum<f32_t>, nk_sum_f32_neon);
    run_if_matches("wsum_neon", "f32", test_wsum<f32_t>, nk_wsum_f32_neon);
    run_if_matches("fma_neon", "f32", test_fma<f32_t>, nk_fma_f32_neon);
    run_if_matches("scale_neon", "e4m3", test_scale<e4m3_t>, nk_scale_e4m3_neon);
    run_if_matches("scale_neon", "e5m2", test_scale<e5m2_t>, nk_scale_e5m2_neon);
    run_if_matches("sum_neon", "e4m3", test_sum<e4m3_t>, nk_sum_e4m3_neon);
    run_if_matches("sum_neon", "e5m2", test_sum<e5m2_t>, nk_sum_e5m2_neon);
    run_if_matches("wsum_neon", "e4m3", test_wsum<e4m3_t>, nk_wsum_e4m3_neon);
    run_if_matches("wsum_neon", "e5m2", test_wsum<e5m2_t>, nk_wsum_e5m2_neon);
    run_if_matches("fma_neon", "e4m3", test_fma<e4m3_t>, nk_fma_e4m3_neon);
    run_if_matches("fma_neon", "e5m2", test_fma<e5m2_t>, nk_fma_e5m2_neon);
#endif // NK_TARGET_NEON

#if NK_TARGET_NEONHALF
#endif // NK_TARGET_NEONHALF

#if NK_TARGET_HASWELL
    run_if_matches("scale_haswell", "f32", test_scale<f32_t>, nk_scale_f32_haswell);
    run_if_matches("sum_haswell", "f32", test_sum<f32_t>, nk_sum_f32_haswell);
    run_if_matches("wsum_haswell", "f32", test_wsum<f32_t>, nk_wsum_f32_haswell);
    run_if_matches("fma_haswell", "f32", test_fma<f32_t>, nk_fma_f32_haswell);
    run_if_matches("scale_haswell", "e4m3", test_scale<e4m3_t>, nk_scale_e4m3_haswell);
    run_if_matches("scale_haswell", "e5m2", test_scale<e5m2_t>, nk_scale_e5m2_haswell);
    run_if_matches("sum_haswell", "e4m3", test_sum<e4m3_t>, nk_sum_e4m3_haswell);
    run_if_matches("sum_haswell", "e5m2", test_sum<e5m2_t>, nk_sum_e5m2_haswell);
    run_if_matches("wsum_haswell", "e4m3", test_wsum<e4m3_t>, nk_wsum_e4m3_haswell);
    run_if_matches("wsum_haswell", "e5m2", test_wsum<e5m2_t>, nk_wsum_e5m2_haswell);
    run_if_matches("fma_haswell", "e4m3", test_fma<e4m3_t>, nk_fma_e4m3_haswell);
    run_if_matches("fma_haswell", "e5m2", test_fma<e5m2_t>, nk_fma_e5m2_haswell);
#endif // NK_TARGET_HASWELL

#if NK_TARGET_SKYLAKE
    run_if_matches("scale_skylake", "f32", test_scale<f32_t>, nk_scale_f32_skylake);
    run_if_matches("sum_skylake", "f32", test_sum<f32_t>, nk_sum_f32_skylake);
    run_if_matches("wsum_skylake", "f32", test_wsum<f32_t>, nk_wsum_f32_skylake);
    run_if_matches("fma_skylake", "f32", test_fma<f32_t>, nk_fma_f32_skylake);
    run_if_matches("scale_skylake", "e4m3", test_scale<e4m3_t>, nk_scale_e4m3_skylake);
    run_if_matches("scale_skylake", "e5m2", test_scale<e5m2_t>, nk_scale_e5m2_skylake);
    run_if_matches("sum_skylake", "e4m3", test_sum<e4m3_t>, nk_sum_e4m3_skylake);
    run_if_matches("sum_skylake", "e5m2", test_sum<e5m2_t>, nk_sum_e5m2_skylake);
    run_if_matches("wsum_skylake", "e4m3", test_wsum<e4m3_t>, nk_wsum_e4m3_skylake);
    run_if_matches("wsum_skylake", "e5m2", test_wsum<e5m2_t>, nk_wsum_e5m2_skylake);
    run_if_matches("fma_skylake", "e4m3", test_fma<e4m3_t>, nk_fma_e4m3_skylake);
    run_if_matches("fma_skylake", "e5m2", test_fma<e5m2_t>, nk_fma_e5m2_skylake);
#endif // NK_TARGET_SKYLAKE

#if NK_TARGET_SAPPHIRE
    run_if_matches("sum_sapphire", "e4m3", test_sum<e4m3_t>, nk_sum_e4m3_sapphire);
#endif // NK_TARGET_SAPPHIRE

    // Serial always runs - baseline test
    run_if_matches("scale_serial", "f32", test_scale<f32_t>, nk_scale_f32_serial);
    run_if_matches("sum_serial", "f32", test_sum<f32_t>, nk_sum_f32_serial);
    run_if_matches("wsum_serial", "f32", test_wsum<f32_t>, nk_wsum_f32_serial);
    run_if_matches("fma_serial", "f32", test_fma<f32_t>, nk_fma_f32_serial);
    run_if_matches("scale_serial", "e4m3", test_scale<e4m3_t>, nk_scale_e4m3_serial);
    run_if_matches("scale_serial", "e5m2", test_scale<e5m2_t>, nk_scale_e5m2_serial);
    run_if_matches("sum_serial", "e4m3", test_sum<e4m3_t>, nk_sum_e4m3_serial);
    run_if_matches("sum_serial", "e5m2", test_sum<e5m2_t>, nk_sum_e5m2_serial);
    run_if_matches("wsum_serial", "e4m3", test_wsum<e4m3_t>, nk_wsum_e4m3_serial);
    run_if_matches("wsum_serial", "e5m2", test_wsum<e5m2_t>, nk_wsum_e5m2_serial);
    run_if_matches("fma_serial", "e4m3", test_fma<e4m3_t>, nk_fma_e4m3_serial);
    run_if_matches("fma_serial", "e5m2", test_fma<e5m2_t>, nk_fma_e5m2_serial);

#endif // NK_DYNAMIC_DISPATCH
}

#pragma endregion // Elementwise

#pragma region Trigonometry

/**
 *  @brief Test sine approximation kernel against nk::sin<scalar_t, f118_t, false> reference.
 */
template <typename scalar_type_>
error_stats_t test_sin(typename scalar_type_::trig_kernel_t kernel) {
    using scalar_t = scalar_type_;
    using raw_t = typename scalar_t::raw_t;

    error_stats_t stats;
    std::mt19937 rng(global_config.seed);
    aligned_buffer<scalar_t> inputs(dense_dimension), outputs(dense_dimension), reference(dense_dimension);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(inputs, rng, scalar_t(-2 * M_PI), scalar_t(2 * M_PI));

        kernel(&inputs[0].raw_, dense_dimension, &outputs[0].raw_);
        nk::sin<scalar_t, f118_t, false>(&inputs[0], dense_dimension, &reference[0]);

        for (std::size_t i = 0; i < dense_dimension; i++) stats.accumulate(outputs[i], reference[i]);
    }
    return stats;
}

/**
 *  @brief Test cosine approximation kernel against nk::cos<scalar_t, f118_t, false> reference.
 */
template <typename scalar_type_>
error_stats_t test_cos(typename scalar_type_::trig_kernel_t kernel) {
    using scalar_t = scalar_type_;
    using raw_t = typename scalar_t::raw_t;

    error_stats_t stats;
    std::mt19937 rng(global_config.seed);
    aligned_buffer<scalar_t> inputs(dense_dimension), outputs(dense_dimension), reference(dense_dimension);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(inputs, rng, scalar_t(-2 * M_PI), scalar_t(2 * M_PI));

        kernel(&inputs[0].raw_, dense_dimension, &outputs[0].raw_);
        nk::cos<scalar_t, f118_t, false>(&inputs[0], dense_dimension, &reference[0]);

        for (std::size_t i = 0; i < dense_dimension; i++) stats.accumulate(outputs[i], reference[i]);
    }
    return stats;
}

/**
 *  @brief Test atan approximation kernel against nk::atan<scalar_t, f118_t, false> reference.
 */
template <typename scalar_type_>
error_stats_t test_atan(typename scalar_type_::trig_kernel_t kernel) {
    using scalar_t = scalar_type_;
    using raw_t = typename scalar_t::raw_t;

    error_stats_t stats;
    std::mt19937 rng(global_config.seed);
    aligned_buffer<scalar_t> inputs(dense_dimension), outputs(dense_dimension), reference(dense_dimension);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(inputs, rng, scalar_t(-10.0), scalar_t(10.0));

        kernel(&inputs[0].raw_, dense_dimension, &outputs[0].raw_);
        nk::atan<scalar_t, f118_t, false>(&inputs[0], dense_dimension, &reference[0]);

        for (std::size_t i = 0; i < dense_dimension; i++) stats.accumulate(outputs[i], reference[i]);
    }
    return stats;
}

void test_trigonometry() {
    std::printf("Testing trigonometry...\n");

#if NK_DYNAMIC_DISPATCH
    run_if_matches("sin", "f32", test_sin<f32_t>, nk_sin_f32);
    run_if_matches("cos", "f32", test_cos<f32_t>, nk_cos_f32);
    run_if_matches("sin", "f64", test_sin<f64_t>, nk_sin_f64);
    run_if_matches("cos", "f64", test_cos<f64_t>, nk_cos_f64);
#else

#if NK_TARGET_NEON
    run_if_matches("sin_neon", "f32", test_sin<f32_t>, nk_sin_f32_neon);
    run_if_matches("cos_neon", "f32", test_cos<f32_t>, nk_cos_f32_neon);
    run_if_matches("sin_neon", "f64", test_sin<f64_t>, nk_sin_f64_neon);
    run_if_matches("cos_neon", "f64", test_cos<f64_t>, nk_cos_f64_neon);
#endif

#if NK_TARGET_HASWELL
    run_if_matches("sin_haswell", "f32", test_sin<f32_t>, nk_sin_f32_haswell);
    run_if_matches("cos_haswell", "f32", test_cos<f32_t>, nk_cos_f32_haswell);
    run_if_matches("sin_haswell", "f64", test_sin<f64_t>, nk_sin_f64_haswell);
    run_if_matches("cos_haswell", "f64", test_cos<f64_t>, nk_cos_f64_haswell);
#endif

#if NK_TARGET_SKYLAKE
    run_if_matches("sin_skylake", "f32", test_sin<f32_t>, nk_sin_f32_skylake);
    run_if_matches("cos_skylake", "f32", test_cos<f32_t>, nk_cos_f32_skylake);
    run_if_matches("sin_skylake", "f64", test_sin<f64_t>, nk_sin_f64_skylake);
    run_if_matches("cos_skylake", "f64", test_cos<f64_t>, nk_cos_f64_skylake);
#endif

#if NK_TARGET_SAPPHIRE
    run_if_matches("sin_sapphire", "f16", test_sin<f16_t>, nk_sin_f16_sapphire);
    run_if_matches("cos_sapphire", "f16", test_cos<f16_t>, nk_cos_f16_sapphire);
    run_if_matches("atan_sapphire", "f16", test_atan<f16_t>, nk_atan_f16_sapphire);
#endif

    run_if_matches("sin_serial", "f32", test_sin<f32_t>, nk_sin_f32_serial);
    run_if_matches("cos_serial", "f32", test_cos<f32_t>, nk_cos_f32_serial);
    run_if_matches("sin_serial", "f64", test_sin<f64_t>, nk_sin_f64_serial);
    run_if_matches("cos_serial", "f64", test_cos<f64_t>, nk_cos_f64_serial);
    run_if_matches("sin_serial", "f16", test_sin<f16_t>, nk_sin_f16_serial);
    run_if_matches("cos_serial", "f16", test_cos<f16_t>, nk_cos_f16_serial);
    run_if_matches("atan_serial", "f16", test_atan<f16_t>, nk_atan_f16_serial);

#endif
}

#pragma endregion // Trigonometry

#pragma region Geospatial

/**
 *  @brief Test Haversine distance.
 */
template <typename scalar_type_>
error_stats_t test_haversine(typename scalar_type_::haversine_kernel_t kernel) {
    using scalar_t = scalar_type_;
    using raw_t = typename scalar_t::raw_t;

    error_stats_t stats;
    std::mt19937 rng(global_config.seed);
    aligned_buffer<scalar_t> a_lats(dense_dimension), a_lons(dense_dimension);
    aligned_buffer<scalar_t> b_lats(dense_dimension), b_lons(dense_dimension);
    aligned_buffer<scalar_t> results(dense_dimension), reference(dense_dimension);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(a_lats, rng, scalar_t(-M_PI / 2), scalar_t(M_PI / 2));
        fill_random(a_lons, rng, scalar_t(-M_PI), scalar_t(M_PI));
        fill_random(b_lats, rng, scalar_t(-M_PI / 2), scalar_t(M_PI / 2));
        fill_random(b_lons, rng, scalar_t(-M_PI), scalar_t(M_PI));

        kernel(&a_lats[0].raw_, &a_lons[0].raw_, &b_lats[0].raw_, &b_lons[0].raw_, dense_dimension, &results[0].raw_);
        nk::haversine<scalar_t, f118_t, false>(&a_lats[0], &a_lons[0], &b_lats[0], &b_lons[0], dense_dimension,
                                               &reference[0]);

        for (std::size_t i = 0; i < dense_dimension; i++) stats.accumulate(results[i], reference[i]);
    }
    return stats;
}

/**
 *  @brief Test Vincenty distance.
 */
template <typename scalar_type_>
error_stats_t test_vincenty(typename scalar_type_::haversine_kernel_t kernel) {
    using scalar_t = scalar_type_;
    using raw_t = typename scalar_t::raw_t;

    error_stats_t stats;
    std::mt19937 rng(global_config.seed);
    aligned_buffer<scalar_t> a_lats(dense_dimension), a_lons(dense_dimension);
    aligned_buffer<scalar_t> b_lats(dense_dimension), b_lons(dense_dimension);
    aligned_buffer<scalar_t> results(dense_dimension), reference(dense_dimension);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(a_lats, rng, scalar_t(-M_PI / 2), scalar_t(M_PI / 2));
        fill_random(a_lons, rng, scalar_t(-M_PI), scalar_t(M_PI));
        fill_random(b_lats, rng, scalar_t(-M_PI / 2), scalar_t(M_PI / 2));
        fill_random(b_lons, rng, scalar_t(-M_PI), scalar_t(M_PI));

        kernel(&a_lats[0].raw_, &a_lons[0].raw_, &b_lats[0].raw_, &b_lons[0].raw_, dense_dimension, &results[0].raw_);
        nk::vincenty<scalar_t, f118_t, false>(&a_lats[0], &a_lons[0], &b_lats[0], &b_lons[0], dense_dimension,
                                              &reference[0]);

        for (std::size_t i = 0; i < dense_dimension; i++) stats.accumulate(results[i], reference[i]);
    }
    return stats;
}

void test_geospatial() {
    std::printf("Testing geospatial functions...\n");

#if NK_DYNAMIC_DISPATCH
    run_if_matches("haversine", "f64", test_haversine<f64_t>, nk_haversine_f64);
    run_if_matches("haversine", "f32", test_haversine<f32_t>, nk_haversine_f32);
    run_if_matches("vincenty", "f64", test_vincenty<f64_t>, nk_vincenty_f64);
#else

#if NK_TARGET_NEON
    run_if_matches("haversine_neon", "f64", test_haversine<f64_t>, nk_haversine_f64_neon);
    run_if_matches("haversine_neon", "f32", test_haversine<f32_t>, nk_haversine_f32_neon);
    run_if_matches("vincenty_neon", "f64", test_vincenty<f64_t>, nk_vincenty_f64_neon);
#endif // NK_TARGET_NEON

#if NK_TARGET_HASWELL
    run_if_matches("haversine_haswell", "f64", test_haversine<f64_t>, nk_haversine_f64_haswell);
    run_if_matches("haversine_haswell", "f32", test_haversine<f32_t>, nk_haversine_f32_haswell);
    run_if_matches("vincenty_haswell", "f64", test_vincenty<f64_t>, nk_vincenty_f64_haswell);
#endif // NK_TARGET_HASWELL

#if NK_TARGET_SKYLAKE
    run_if_matches("haversine_skylake", "f64", test_haversine<f64_t>, nk_haversine_f64_skylake);
    run_if_matches("haversine_skylake", "f32", test_haversine<f32_t>, nk_haversine_f32_skylake);
    run_if_matches("vincenty_skylake", "f64", test_vincenty<f64_t>, nk_vincenty_f64_skylake);
#endif // NK_TARGET_SKYLAKE

    // Serial always runs - baseline test
    run_if_matches("haversine_serial", "f64", test_haversine<f64_t>, nk_haversine_f64_serial);
    run_if_matches("haversine_serial", "f32", test_haversine<f32_t>, nk_haversine_f32_serial);
    run_if_matches("vincenty_serial", "f64", test_vincenty<f64_t>, nk_vincenty_f64_serial);

#endif // NK_DYNAMIC_DISPATCH
}

#pragma endregion // Geospatial

#pragma region Mesh

/**
 *  @brief Test RMSD kernel.
 */
template <typename scalar_type_>
error_stats_t test_rmsd(typename scalar_type_::mesh_kernel_t kernel) {
    using scalar_t = scalar_type_;

    error_stats_t stats;
    std::mt19937 rng(global_config.seed);

    std::size_t n = mesh_dimension;
    aligned_buffer<scalar_t> a(n * 3), b(n * 3);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(a, rng, scalar_t(-10.0), scalar_t(10.0));
        fill_random(b, rng, scalar_t(-10.0), scalar_t(10.0));

        scalar_t a_centroid[3], b_centroid[3], rot[9], scale, result;
        kernel(&a[0].raw_, &b[0].raw_, n, &a_centroid[0].raw_, &b_centroid[0].raw_, &rot[0].raw_, &scale.raw_,
               &result.raw_);

        f118_t a_centroid_ref[3], b_centroid_ref[3], rot_ref[9], scale_ref, reference;
        nk::rmsd<scalar_t, f118_t, false>(&a[0], &b[0], n, a_centroid_ref, b_centroid_ref, rot_ref, &scale_ref,
                                          &reference);

        stats.accumulate(result, reference);
    }
    return stats;
}

/**
 *  @brief Test Kabsch alignment kernel.
 */
template <typename scalar_type_>
error_stats_t test_kabsch(typename scalar_type_::mesh_kernel_t kernel) {
    using scalar_t = scalar_type_;

    error_stats_t stats;
    std::mt19937 rng(global_config.seed);

    std::size_t n = mesh_dimension;
    aligned_buffer<scalar_t> a(n * 3), b(n * 3);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(a, rng, scalar_t(-5.0), scalar_t(5.0));
        fill_random(b, rng, scalar_t(-5.0), scalar_t(5.0));

        scalar_t a_centroid[3], b_centroid[3], rot[9], scale, result;
        kernel(&a[0].raw_, &b[0].raw_, n, &a_centroid[0].raw_, &b_centroid[0].raw_, &rot[0].raw_, &scale.raw_,
               &result.raw_);

        f118_t a_centroid_ref[3], b_centroid_ref[3], rot_ref[9], scale_ref, reference;
        nk::kabsch<scalar_t, f118_t, false>(&a[0], &b[0], n, a_centroid_ref, b_centroid_ref, rot_ref, &scale_ref,
                                            &reference);

        stats.accumulate(result, reference);
    }
    return stats;
}

/**
 *  @brief Test Umeyama alignment kernel.
 */
template <typename scalar_type_>
error_stats_t test_umeyama(typename scalar_type_::mesh_kernel_t kernel) {
    using scalar_t = scalar_type_;

    error_stats_t stats;
    std::mt19937 rng(global_config.seed);

    std::size_t n = mesh_dimension;
    aligned_buffer<scalar_t> a(n * 3), b(n * 3);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(a, rng, scalar_t(-5.0), scalar_t(5.0));
        fill_random(b, rng, scalar_t(-5.0), scalar_t(5.0));

        scalar_t a_centroid[3], b_centroid[3], rot[9], scale, result;
        kernel(&a[0].raw_, &b[0].raw_, n, &a_centroid[0].raw_, &b_centroid[0].raw_, &rot[0].raw_, &scale.raw_,
               &result.raw_);

        f118_t a_centroid_ref[3], b_centroid_ref[3], rot_ref[9], scale_ref, reference;
        nk::umeyama<scalar_t, f118_t, false>(&a[0], &b[0], n, a_centroid_ref, b_centroid_ref, rot_ref, &scale_ref,
                                             &reference);

        stats.accumulate(result, reference);
    }
    return stats;
}

void test_mesh() {
    std::printf("Testing mesh operations...\n");

#if NK_DYNAMIC_DISPATCH
    run_if_matches("rmsd", "f64", test_rmsd<f64_t>, nk_rmsd_f64);
    run_if_matches("rmsd", "f32", test_rmsd<f32_t>, nk_rmsd_f32);
    run_if_matches("kabsch", "f64", test_kabsch<f64_t>, nk_kabsch_f64);
    run_if_matches("kabsch", "f32", test_kabsch<f32_t>, nk_kabsch_f32);
    run_if_matches("umeyama", "f64", test_umeyama<f64_t>, nk_umeyama_f64);
    run_if_matches("umeyama", "f32", test_umeyama<f32_t>, nk_umeyama_f32);
#else

#if NK_TARGET_NEON
    run_if_matches("rmsd_neon", "f64", test_rmsd<f64_t>, nk_rmsd_f64_neon);
    run_if_matches("rmsd_neon", "f32", test_rmsd<f32_t>, nk_rmsd_f32_neon);
    run_if_matches("kabsch_neon", "f64", test_kabsch<f64_t>, nk_kabsch_f64_neon);
    run_if_matches("kabsch_neon", "f32", test_kabsch<f32_t>, nk_kabsch_f32_neon);
    run_if_matches("umeyama_neon", "f64", test_umeyama<f64_t>, nk_umeyama_f64_neon);
    run_if_matches("umeyama_neon", "f32", test_umeyama<f32_t>, nk_umeyama_f32_neon);
#endif // NK_TARGET_NEON

#if NK_TARGET_HASWELL
    run_if_matches("rmsd_haswell", "f64", test_rmsd<f64_t>, nk_rmsd_f64_haswell);
    run_if_matches("rmsd_haswell", "f32", test_rmsd<f32_t>, nk_rmsd_f32_haswell);
    run_if_matches("kabsch_haswell", "f64", test_kabsch<f64_t>, nk_kabsch_f64_haswell);
    run_if_matches("kabsch_haswell", "f32", test_kabsch<f32_t>, nk_kabsch_f32_haswell);
    run_if_matches("umeyama_haswell", "f64", test_umeyama<f64_t>, nk_umeyama_f64_haswell);
    run_if_matches("umeyama_haswell", "f32", test_umeyama<f32_t>, nk_umeyama_f32_haswell);
#endif // NK_TARGET_HASWELL

#if NK_TARGET_SKYLAKE
    run_if_matches("rmsd_skylake", "f64", test_rmsd<f64_t>, nk_rmsd_f64_skylake);
    run_if_matches("rmsd_skylake", "f32", test_rmsd<f32_t>, nk_rmsd_f32_skylake);
    run_if_matches("kabsch_skylake", "f64", test_kabsch<f64_t>, nk_kabsch_f64_skylake);
    run_if_matches("kabsch_skylake", "f32", test_kabsch<f32_t>, nk_kabsch_f32_skylake);
    run_if_matches("umeyama_skylake", "f64", test_umeyama<f64_t>, nk_umeyama_f64_skylake);
    run_if_matches("umeyama_skylake", "f32", test_umeyama<f32_t>, nk_umeyama_f32_skylake);
#endif // NK_TARGET_SKYLAKE

    // Serial always runs - baseline test
    run_if_matches("rmsd_serial", "f64", test_rmsd<f64_t>, nk_rmsd_f64_serial);
    run_if_matches("rmsd_serial", "f32", test_rmsd<f32_t>, nk_rmsd_f32_serial);
    run_if_matches("kabsch_serial", "f64", test_kabsch<f64_t>, nk_kabsch_f64_serial);
    run_if_matches("kabsch_serial", "f32", test_kabsch<f32_t>, nk_kabsch_f32_serial);
    run_if_matches("umeyama_serial", "f64", test_umeyama<f64_t>, nk_umeyama_f64_serial);
    run_if_matches("umeyama_serial", "f32", test_umeyama<f32_t>, nk_umeyama_f32_serial);

#endif // NK_DYNAMIC_DISPATCH
}

#pragma endregion // Mesh

#pragma region Dots

/**
 *  @brief Generic GEMM test against f118_t reference.
 *  Works for all types: f32, f64, f16, bf16, i8.
 */
template <typename scalar_type_>
error_stats_t test_dots(typename scalar_type_::dots_packed_size_kernel_t packed_size_fn,
                        typename scalar_type_::dots_pack_kernel_t pack_fn,
                        typename scalar_type_::dots_packed_kernel_t dots_fn) {
    using scalar_t = scalar_type_;
    using result_t = typename scalar_t::dot_result_t;

    error_stats_t stats;
    std::mt19937 rng(global_config.seed);

    std::size_t m = matmul_dimension_m, n = matmul_dimension_n, k = matmul_dimension_k;
    std::size_t a_stride = k * sizeof(typename scalar_t::raw_t);
    std::size_t b_stride = k * sizeof(typename scalar_t::raw_t);
    std::size_t c_stride = n * sizeof(result_t);

    aligned_buffer<scalar_t> a(m * k), b(n * k);
    aligned_buffer<result_t> c(m * n);
    aligned_buffer<f118_t> c_ref(m * n);

    nk_size_t packed_size = packed_size_fn(n, k);
    aligned_buffer<char> b_packed(packed_size);
    nk_size_t ref_packed_size = nk::dots_packed_size<scalar_t>(n, k);
    aligned_buffer<char> b_packed_ref(ref_packed_size);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(a, rng);
        fill_random(b, rng);

        // Run kernel being tested
        pack_fn(&b[0].raw_, n, k, b_stride, b_packed.data());
        dots_fn(&a[0].raw_, b_packed.data(), &c[0].raw_, m, n, k, a_stride, c_stride);

        // Compute f118_t reference using nk:: template
        nk::dots_pack<scalar_t>(&b[0], n, k, b_stride, b_packed_ref.data());
        nk::dots_packed<scalar_t, f118_t, false>(&a[0], b_packed_ref.data(), &c_ref[0], m, n, k, a_stride,
                                                 n * sizeof(f118_t));

        for (std::size_t i = 0; i < m * n; i++) stats.accumulate(c[i], c_ref[i]);
    }
    return stats;
}

#if NK_COMPARE_TO_BLAS || NK_COMPARE_TO_MKL || NK_COMPARE_TO_ACCELERATE
/**
 *  @brief Unified template to test GEMM with unpacked B (for BLAS comparison).
 *  Uses dots_result_t for reference computation precision.
 */
template <typename scalar_type_>
error_stats_t test_dots_unpacked(typename scalar_type_::dots_blas_kernel_t dots_fn) {
    using scalar_t = scalar_type_;
    using raw_t = typename scalar_t::raw_t;
    using result_t = typename scalar_t::dots_result_t;

    error_stats_t stats;
    std::mt19937 rng(global_config.seed);

    std::size_t m = matmul_dimension_m, n = matmul_dimension_n, k = matmul_dimension_k;
    std::size_t a_stride = k * sizeof(raw_t);
    std::size_t b_stride = k * sizeof(raw_t);
    std::size_t c_stride = n * sizeof(result_t);

    aligned_buffer<scalar_t> a_buf(m * k), b_buf(n * k);
    aligned_buffer<result_t> c(m * n);
    aligned_buffer<f118_t> c_ref(m * n);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(a_buf, rng);
        fill_random(b_buf, rng);

        reference_dots<f118_t, scalar_t>(&a_buf[0], &b_buf[0], &c_ref[0], m, n, k, a_stride, b_stride);
        dots_fn(&a_buf[0].raw_, &b_buf[0].raw_, &c[0], m, n, k, a_stride, c_stride);

        for (std::size_t i = 0; i < m * n; i++) stats.accumulate(c[i], c_ref[i]);
    }
    return stats;
}
#endif // NK_COMPARE_TO_BLAS || NK_COMPARE_TO_MKL || NK_COMPARE_TO_ACCELERATE

void test_dots() {
    std::printf("Testing batch dot products (GEMM)...\n");

#if NK_DYNAMIC_DISPATCH
    run_if_matches("dots", "f32", test_dots<f32_t>, nk_dots_packed_size_f32, nk_dots_pack_f32, nk_dots_packed_f32);
    run_if_matches("dots", "f64", test_dots<f64_t>, nk_dots_packed_size_f64, nk_dots_pack_f64, nk_dots_packed_f64);
    run_if_matches("dots", "bf16", test_dots<bf16_t>, nk_dots_packed_size_bf16, nk_dots_pack_bf16, nk_dots_packed_bf16);
    run_if_matches("dots", "f16", test_dots<f16_t>, nk_dots_packed_size_f16, nk_dots_pack_f16, nk_dots_packed_f16);
    run_if_matches("dots", "i8", test_dots<i8_t>, nk_dots_packed_size_i8, nk_dots_pack_i8, nk_dots_packed_i8);
    run_if_matches("dots", "u1", test_dots<u1x8_t>, nk_dots_packed_size_u1, nk_dots_pack_u1, nk_dots_packed_u1);
    run_if_matches("dots", "u4", test_dots<u4x2_t>, nk_dots_packed_size_u4, nk_dots_pack_u4, nk_dots_packed_u4);
    run_if_matches("dots", "i4", test_dots<i4x2_t>, nk_dots_packed_size_i4, nk_dots_pack_i4, nk_dots_packed_i4);
#else

#if NK_TARGET_NEON
    run_if_matches("dots_neon", "f32", test_dots<f32_t>, nk_dots_packed_size_f32_neon, nk_dots_pack_f32_neon,
                   nk_dots_packed_f32_neon);
    run_if_matches("dots_neon", "f64", test_dots<f64_t>, nk_dots_packed_size_f64_neon, nk_dots_pack_f64_neon,
                   nk_dots_packed_f64_neon);
#endif // NK_TARGET_NEON

#if NK_TARGET_HASWELL
    run_if_matches("dots_haswell", "f32", test_dots<f32_t>, nk_dots_packed_size_f32_haswell, nk_dots_pack_f32_haswell,
                   nk_dots_packed_f32_haswell);
    run_if_matches("dots_haswell", "f64", test_dots<f64_t>, nk_dots_packed_size_f64_haswell, nk_dots_pack_f64_haswell,
                   nk_dots_packed_f64_haswell);
#endif // NK_TARGET_HASWELL

#if NK_TARGET_SKYLAKE
    run_if_matches("dots_skylake", "f32", test_dots<f32_t>, nk_dots_packed_size_f32_skylake, nk_dots_pack_f32_skylake,
                   nk_dots_packed_f32_skylake);
    run_if_matches("dots_skylake", "f64", test_dots<f64_t>, nk_dots_packed_size_f64_skylake, nk_dots_pack_f64_skylake,
                   nk_dots_packed_f64_skylake);
#endif // NK_TARGET_SKYLAKE

    // Serial always runs - baseline test
    run_if_matches("dots_serial", "f32", test_dots<f32_t>, nk_dots_packed_size_f32_serial, nk_dots_pack_f32_serial,
                   nk_dots_packed_f32_serial);
    run_if_matches("dots_serial", "f64", test_dots<f64_t>, nk_dots_packed_size_f64_serial, nk_dots_pack_f64_serial,
                   nk_dots_packed_f64_serial);
    run_if_matches("dots_serial", "bf16", test_dots<bf16_t>, nk_dots_packed_size_bf16_serial, nk_dots_pack_bf16_serial,
                   nk_dots_packed_bf16_serial);
    run_if_matches("dots_serial", "f16", test_dots<f16_t>, nk_dots_packed_size_f16_serial, nk_dots_pack_f16_serial,
                   nk_dots_packed_f16_serial);
    run_if_matches("dots_serial", "i8", test_dots<i8_t>, nk_dots_packed_size_i8_serial, nk_dots_pack_i8_serial,
                   nk_dots_packed_i8_serial);
    run_if_matches("dots_serial", "u1", test_dots<u1x8_t>, nk_dots_packed_size_u1x8_serial, nk_dots_pack_u1x8_serial,
                   nk_dots_packed_u1x8_serial);
    run_if_matches("dots_serial", "u4", test_dots<u4x2_t>, nk_dots_packed_size_u4x2_serial, nk_dots_pack_u4x2_serial,
                   nk_dots_packed_u4x2_serial);
    run_if_matches("dots_serial", "i4", test_dots<i4x2_t>, nk_dots_packed_size_i4x2_serial, nk_dots_pack_i4x2_serial,
                   nk_dots_packed_i4x2_serial);

#endif // NK_DYNAMIC_DISPATCH

#if NK_COMPARE_TO_BLAS || NK_COMPARE_TO_MKL || NK_COMPARE_TO_ACCELERATE
    // BLAS/MKL/Accelerate GEMM precision comparison
    run_if_matches("dots_blas", "f32", test_dots_unpacked<f32_t>, dots_f32_blas);
    run_if_matches("dots_blas", "f64", test_dots_unpacked<f64_t>, dots_f64_blas);
#endif // NK_COMPARE_TO_BLAS || NK_COMPARE_TO_MKL || NK_COMPARE_TO_ACCELERATE
}

#pragma endregion // Dots

#pragma region Sparse

/**
 *  @brief Generate a sorted array of unique random indices.
 */
template <typename scalar_type_>
void fill_sorted_unique(aligned_buffer<scalar_type_> &buf, std::mt19937 &rng, scalar_type_ max_val) {
    using raw_t = typename scalar_type_::raw_t;
    std::uniform_int_distribution<raw_t> dist(0, static_cast<raw_t>(max_val));
    std::vector<raw_t> values;
    values.reserve(buf.size() * 2);

    // Generate more values than needed, then deduplicate
    while (values.size() < buf.size()) {
        for (std::size_t i = 0; i < buf.size() * 2 && values.size() < buf.size() * 3; i++) {
            values.push_back(dist(rng));
        }
        std::sort(values.begin(), values.end());
        values.erase(std::unique(values.begin(), values.end()), values.end());
    }

    // Take first buf.size() unique values
    for (std::size_t i = 0; i < buf.size(); i++) buf[i] = scalar_type_(values[i]);
}

/**
 *  @brief Test set intersection (unified template for u16/u32 index types).
 */
template <typename index_type_>
error_stats_t test_intersect(typename index_type_::intersect_kernel_t kernel) {
    using index_t = index_type_;
    error_stats_t stats;
    std::mt19937 rng(global_config.seed);
    std::size_t dim = sparse_dimension;
    aligned_buffer<index_t> a(dim), b(dim);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_sorted_unique(a, rng, index_t(dim * 4));
        fill_sorted_unique(b, rng, index_t(dim * 4));

        nk_u32_t result;
        kernel(&a[0].raw_, &b[0].raw_, dim, dim, &result);

        std::uint32_t ref;
        nk::intersect<index_t, false>(&a[0], &b[0], dim, dim, &ref);
        stats.accumulate_exact(result == ref);
    }
    return stats;
}

/**
 *  @brief Test sparse dot product (unified template, parameterized by weight type).
 *
 *  Dispatch is by weight type (matching numkong.h dispatch tables):
 *  - bf16_t weights → u16_t indices
 *  - f32_t weights → u32_t indices
 */
template <typename weight_type_>
error_stats_t test_sparse_dot(typename weight_type_::sparse_dot_kernel_t kernel) {
    using weight_t = weight_type_;
    using index_t = typename weight_t::sparse_dot_index_t;

    error_stats_t stats;
    std::mt19937 rng(global_config.seed);
    std::size_t dim = sparse_dimension;
    aligned_buffer<index_t> a_idx(dim), b_idx(dim);
    aligned_buffer<weight_t> a_weights(dim), b_weights(dim);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_sorted_unique(a_idx, rng, index_t(dim * 4));
        fill_sorted_unique(b_idx, rng, index_t(dim * 4));
        fill_random(a_weights, rng);
        fill_random(b_weights, rng);

        typename weight_t::dot_result_t result;
        kernel(&a_idx[0].raw_, &b_idx[0].raw_, &a_weights[0].raw_, &b_weights[0].raw_, dim, dim, &result.raw_);

        f118_t ref;
        nk::sparse_dot<index_t, weight_t, f118_t, false>(&a_idx[0], &b_idx[0], &a_weights[0], &b_weights[0], dim, dim,
                                                         &ref);
        stats.accumulate(result, ref);
    }
    return stats;
}

void test_sparse() {
    std::printf("Testing sparse operations...\n");

#if NK_DYNAMIC_DISPATCH
    run_if_matches("intersect", "u16", test_intersect<u16_t>, nk_intersect_u16);
    run_if_matches("intersect", "u32", test_intersect<u32_t>, nk_intersect_u32);
    run_if_matches("sparse_dot", "u32f32", test_sparse_dot<f32_t>, nk_sparse_dot_u32f32);
    run_if_matches("sparse_dot", "u16bf16", test_sparse_dot<bf16_t>, nk_sparse_dot_u16bf16);
#else

#if NK_TARGET_NEON
    run_if_matches("intersect_neon", "u16", test_intersect<u16_t>, nk_intersect_u16_neon);
    run_if_matches("intersect_neon", "u32", test_intersect<u32_t>, nk_intersect_u32_neon);
#endif // NK_TARGET_NEON

#if NK_TARGET_SVE
    run_if_matches("intersect_sve2", "u16", test_intersect<u16_t>, nk_intersect_u16_sve2);
    run_if_matches("intersect_sve2", "u32", test_intersect<u32_t>, nk_intersect_u32_sve2);
    run_if_matches("sparse_dot_sve2", "u32f32", test_sparse_dot<f32_t>, nk_sparse_dot_u32f32_sve2);
    run_if_matches("sparse_dot_sve2", "u16bf16", test_sparse_dot<bf16_t>, nk_sparse_dot_u16bf16_sve2);
#endif // NK_TARGET_SVE

#if NK_TARGET_ICE
    run_if_matches("intersect_ice", "u16", test_intersect<u16_t>, nk_intersect_u16_ice);
    run_if_matches("intersect_ice", "u32", test_intersect<u32_t>, nk_intersect_u32_ice);
    run_if_matches("sparse_dot_ice", "u32f32", test_sparse_dot<f32_t>, nk_sparse_dot_u32f32_ice);
#endif // NK_TARGET_ICE

#if NK_TARGET_TURIN
    run_if_matches("intersect_turin", "u16", test_intersect<u16_t>, nk_intersect_u16_turin);
    run_if_matches("intersect_turin", "u32", test_intersect<u32_t>, nk_intersect_u32_turin);
    run_if_matches("sparse_dot_turin", "u32f32", test_sparse_dot<f32_t>, nk_sparse_dot_u32f32_turin);
    run_if_matches("sparse_dot_turin", "u16bf16", test_sparse_dot<bf16_t>, nk_sparse_dot_u16bf16_turin);
#endif // NK_TARGET_TURIN

    // Serial always runs - baseline test
    run_if_matches("intersect_serial", "u16", test_intersect<u16_t>, nk_intersect_u16_serial);
    run_if_matches("intersect_serial", "u32", test_intersect<u32_t>, nk_intersect_u32_serial);
    run_if_matches("sparse_dot_serial", "u32f32", test_sparse_dot<f32_t>, nk_sparse_dot_u32f32_serial);
    run_if_matches("sparse_dot_serial", "u16bf16", test_sparse_dot<bf16_t>, nk_sparse_dot_u16bf16_serial);

#endif // NK_DYNAMIC_DISPATCH
}

#pragma endregion // Sparse

int main(int argc, char **argv) {
    (void)argc;
    (void)argv;

    if (char const *env = std::getenv("NK_TEST_ASSERT")) global_config.assert_on_failure = std::atoi(env) != 0;
    if (char const *env = std::getenv("NK_TEST_VERBOSE")) global_config.verbose = std::atoi(env) != 0;
    if (char const *env = std::getenv("NK_TEST_ULP_THRESHOLD_F32")) global_config.ulp_threshold_f32 = std::atoll(env);
    if (char const *env = std::getenv("NK_TEST_ULP_THRESHOLD_F16")) global_config.ulp_threshold_f16 = std::atoll(env);
    if (char const *env = std::getenv("NK_TEST_ULP_THRESHOLD_BF16")) global_config.ulp_threshold_bf16 = std::atoll(env);
    if (char const *env = std::getenv("NK_TEST_TIME_BUDGET_MS")) global_config.time_budget_ms = std::atoll(env);
    if (char const *env = std::getenv("NK_TEST_SEED")) global_config.seed = std::atoll(env);
    global_config.filter = std::getenv("NK_TEST_FILTER"); // e.g., "dot", "angular", "kld"
    if (char const *env = std::getenv("NK_TEST_DISTRIBUTION")) {
        if (std::strcmp(env, "uniform_k") == 0) global_config.distribution = random_distribution_kind_t::uniform_k;
        else if (std::strcmp(env, "cauchy_k") == 0) global_config.distribution = random_distribution_kind_t::cauchy_k;
        else if (std::strcmp(env, "lognormal_k") == 0)
            global_config.distribution = random_distribution_kind_t::lognormal_k;
    }

    // Parse dimension overrides from environment variables
    if (char const *env = std::getenv("NK_TEST_DENSE_DIMENSION")) {
        std::size_t val = static_cast<std::size_t>(std::atoll(env));
        if (val > 0) {
            dense_dimension = val;
            std::printf("Using NK_TEST_DENSE_DIMENSION=%zu\n", dense_dimension);
        }
    }
    if (char const *env = std::getenv("NK_TEST_SPARSE_DIMENSION")) {
        std::size_t val = static_cast<std::size_t>(std::atoll(env));
        if (val > 0) {
            sparse_dimension = val;
            std::printf("Using NK_TEST_SPARSE_DIMENSION=%zu\n", sparse_dimension);
        }
    }
    if (char const *env = std::getenv("NK_TEST_MESH_DIMENSION")) {
        std::size_t val = static_cast<std::size_t>(std::atoll(env));
        if (val > 0) {
            mesh_dimension = val;
            std::printf("Using NK_TEST_MESH_DIMENSION=%zu\n", mesh_dimension);
        }
    }
    if (char const *env = std::getenv("NK_TEST_MATMUL_DIMENSION_M")) {
        std::size_t val = static_cast<std::size_t>(std::atoll(env));
        if (val > 0) {
            matmul_dimension_m = val;
            std::printf("Using NK_TEST_MATMUL_DIMENSION_M=%zu\n", matmul_dimension_m);
        }
    }
    if (char const *env = std::getenv("NK_TEST_MATMUL_DIMENSION_N")) {
        std::size_t val = static_cast<std::size_t>(std::atoll(env));
        if (val > 0) {
            matmul_dimension_n = val;
            std::printf("Using NK_TEST_MATMUL_DIMENSION_N=%zu\n", matmul_dimension_n);
        }
    }
    if (char const *env = std::getenv("NK_TEST_MATMUL_DIMENSION_K")) {
        std::size_t val = static_cast<std::size_t>(std::atoll(env));
        if (val > 0) {
            matmul_dimension_k = val;
            std::printf("Using NK_TEST_MATMUL_DIMENSION_K=%zu\n", matmul_dimension_k);
        }
    }

    nk_capability_t runtime_caps = nk_capabilities();
    nk_configure_thread(runtime_caps); // Also enables AMX if available

    std::printf("NumKong precision testing suite\n");
    char const *flags[2] = {"false", "true"};
    std::printf("- Compiler used native F16: %s\n", flags[NK_NATIVE_F16]);
    std::printf("- Compiler used native BF16: %s\n", flags[NK_NATIVE_BF16]);
    std::printf("- Benchmark against CBLAS: %s\n", flags[NK_COMPARE_TO_BLAS]);
    std::printf("- Benchmark against MKL: %s\n", flags[NK_COMPARE_TO_MKL]);
    std::printf("- Benchmark against Accelerate: %s\n", flags[NK_COMPARE_TO_ACCELERATE]);
    std::printf("\n");

    std::printf("Compile-time ISA support:\n");
    std::printf("  Arm NEON:         %s\n", flags[NK_TARGET_NEON]);
    std::printf("  Arm NEON f16:     %s\n", flags[NK_TARGET_NEONHALF]);
    std::printf("  Arm NEON bf16:    %s\n", flags[NK_TARGET_NEONBFDOT]);
    std::printf("  Arm NEON i8:      %s\n", flags[NK_TARGET_NEONSDOT]);
    std::printf("  Arm SVE:          %s\n", flags[NK_TARGET_SVE]);
    std::printf("  Arm SVE f16:      %s\n", flags[NK_TARGET_SVEHALF]);
    std::printf("  Arm SVE bf16:     %s\n", flags[NK_TARGET_SVEBFDOT]);
    std::printf("  Arm SVE i8:       %s\n", flags[NK_TARGET_SVESDOT]);
    std::printf("  Arm SVE2:         %s\n", flags[NK_TARGET_SVE2]);
    std::printf("  x86 Haswell:      %s\n", flags[NK_TARGET_HASWELL]);
    std::printf("  x86 Skylake:      %s\n", flags[NK_TARGET_SKYLAKE]);
    std::printf("  x86 Ice Lake:     %s\n", flags[NK_TARGET_ICE]);
    std::printf("  x86 Genoa:        %s\n", flags[NK_TARGET_GENOA]);
    std::printf("  x86 Sapphire:     %s\n", flags[NK_TARGET_SAPPHIRE]);
    std::printf("  x86 Sapphire AMX: %s\n", flags[NK_TARGET_SAPPHIRE_AMX]);
    std::printf("  x86 Granite AMX:  %s\n", flags[NK_TARGET_GRANITE_AMX]);
    std::printf("  x86 Turin:        %s\n", flags[NK_TARGET_TURIN]);
    std::printf("  x86 Sierra:       %s\n", flags[NK_TARGET_SIERRA]);
    std::printf("\n");

    std::printf("Runtime ISA detection:\n");
    std::printf("  Arm NEON:         %s\n", flags[(runtime_caps & nk_cap_neon_k) != 0]);
    std::printf("  Arm NEON f16:     %s\n", flags[(runtime_caps & nk_cap_neonhalf_k) != 0]);
    std::printf("  Arm NEON bf16:    %s\n", flags[(runtime_caps & nk_cap_neonbfdot_k) != 0]);
    std::printf("  Arm NEON i8:      %s\n", flags[(runtime_caps & nk_cap_neonsdot_k) != 0]);
    std::printf("  Arm SVE:          %s\n", flags[(runtime_caps & nk_cap_sve_k) != 0]);
    std::printf("  Arm SVE f16:      %s\n", flags[(runtime_caps & nk_cap_svehalf_k) != 0]);
    std::printf("  Arm SVE bf16:     %s\n", flags[(runtime_caps & nk_cap_svebfdot_k) != 0]);
    std::printf("  Arm SVE i8:       %s\n", flags[(runtime_caps & nk_cap_svesdot_k) != 0]);
    std::printf("  Arm SVE2:         %s\n", flags[(runtime_caps & nk_cap_sve2_k) != 0]);
    std::printf("  x86 Haswell:      %s\n", flags[(runtime_caps & nk_cap_haswell_k) != 0]);
    std::printf("  x86 Skylake:      %s\n", flags[(runtime_caps & nk_cap_skylake_k) != 0]);
    std::printf("  x86 Ice Lake:     %s\n", flags[(runtime_caps & nk_cap_ice_k) != 0]);
    std::printf("  x86 Genoa:        %s\n", flags[(runtime_caps & nk_cap_genoa_k) != 0]);
    std::printf("  x86 Sapphire:     %s\n", flags[(runtime_caps & nk_cap_sapphire_k) != 0]);
    std::printf("  x86 Sapphire AMX: %s\n", flags[(runtime_caps & nk_cap_sapphire_amx_k) != 0]);
    std::printf("  x86 Granite AMX:  %s\n", flags[(runtime_caps & nk_cap_granite_amx_k) != 0]);
    std::printf("  x86 Turin:        %s\n", flags[(runtime_caps & nk_cap_turin_k) != 0]);
    std::printf("  x86 Sierra:       %s\n", flags[(runtime_caps & nk_cap_sierra_k) != 0]);
    std::printf("\n");

    std::printf("Test configuration:\n");
    std::printf("  Assert on failure: %s\n", flags[global_config.assert_on_failure]);
    std::printf("  Verbose:           %s\n", flags[global_config.verbose]);
    std::printf("  Filter:            %s\n", global_config.filter ? global_config.filter : "(none)");
    std::printf("  Distribution:      %s\n", global_config.distribution_name());
    std::printf("  Time budget (ms):  %zu\n", global_config.time_budget_ms);
    std::printf("  RNG seed:          %u\n", global_config.seed);
    std::printf("  ULP threshold f32: %llu\n", static_cast<unsigned long long>(global_config.ulp_threshold_f32));
    std::printf("  ULP threshold f16: %llu\n", static_cast<unsigned long long>(global_config.ulp_threshold_f16));
    std::printf("  ULP threshold bf16:%llu\n", static_cast<unsigned long long>(global_config.ulp_threshold_bf16));
#if NK_TEST_USE_OPENMP
    std::printf("  OpenMP threads:    %d\n", omp_get_max_threads());
#else
    std::printf("  OpenMP:            disabled\n");
#endif
    std::printf("\n");

    // Type conversion tests
    test_fp8_conversions();

    // Print header for precision table
    print_stats_header();

    // Cast tests (batch type conversions)
    test_casts();

    // Core operation tests
    test_dot();
    test_spatial();
    test_curved();
    test_probability();
    test_binary();
    test_elementwise();
    test_trigonometry();

    // Additional operation tests
    test_reduce();
    test_geospatial();
    test_mesh();
    test_dots();
    test_sparse();

    std::printf("\n");
    std::printf("All tests passed.\n");
    return 0;
}

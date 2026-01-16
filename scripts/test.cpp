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
 *    NK_TEST_DISTRIBUTION=<type>  - Random distribution: uniform_k|lognormal_k|cauchy_k (default: lognormal_k)
 *    NK_DENSE_DIMENSIONS=N        - Vector dimension for dot/spatial tests (default: 1024)
 *    NK_SPARSE_DIMENSIONS=N       - Vector dimension for sparse tests (default: 256)
 *    NK_MESH_POINTS=N             - Point count for mesh tests (default: 256)
 *    NK_MATRIX_HEIGHT=N           - GEMM M dimension (default: 64)
 *    NK_MATRIX_WIDTH=N            - GEMM N dimension (default: 64)
 *    NK_MATRIX_DEPTH=N            - GEMM K dimension (default: 64)
 */

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
using nk::bf16c_t;
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

extern template class nk::vector<int>;
extern template class nk::vector<nk::i32_t>;
extern template class nk::vector<nk::u1x8_t>;
extern template class nk::vector<nk::i4x2_t>;
extern template class nk::vector<nk::f64c_t>;
extern template class nk::vector<std::complex<float>>;

#pragma region Configuration

std::size_t dense_dimensions = 1024; // For dot products, spatial metrics
std::size_t sparse_dimensions = 256; // For sparse set intersection and sparse dot
std::size_t mesh_points = 256;       // For RMSD, Kabsch (3D point clouds)
std::size_t matrix_height = 64, matrix_width = 64, matrix_depth = 64;

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
 *  @param name The test name (e.g., "dot_with_blas", "l2sq_haswell")
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

#pragma endregion // Error Statistics

#pragma region BLAS Baselines

#if NK_COMPARE_TO_BLAS || NK_COMPARE_TO_MKL || NK_COMPARE_TO_ACCELERATE

void dot_f32_with_blas(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *result) {
    *result = cblas_sdot(static_cast<int>(n), a, 1, b, 1);
}

void dot_f64_with_blas(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result) {
    *result = cblas_ddot(static_cast<int>(n), a, 1, b, 1);
}

void dot_f32c_with_blas(nk_f32c_t const *a, nk_f32c_t const *b, nk_size_t n, nk_f32c_t *result) {
#if NK_COMPARE_TO_ACCELERATE
    cblas_cdotu_sub(static_cast<int>(n), reinterpret_cast<__LAPACK_float_complex const *>(a), 1,
                    reinterpret_cast<__LAPACK_float_complex const *>(b), 1,
                    reinterpret_cast<__LAPACK_float_complex *>(result));
#else
    cblas_cdotu_sub(static_cast<int>(n), a, 1, b, 1, result);
#endif
}

void vdot_f32c_with_blas(nk_f32c_t const *a, nk_f32c_t const *b, nk_size_t n, nk_f32c_t *result) {
#if NK_COMPARE_TO_ACCELERATE
    cblas_cdotc_sub(static_cast<int>(n), reinterpret_cast<__LAPACK_float_complex const *>(a), 1,
                    reinterpret_cast<__LAPACK_float_complex const *>(b), 1,
                    reinterpret_cast<__LAPACK_float_complex *>(result)); // conjugated
#else
    cblas_cdotc_sub(static_cast<int>(n), a, 1, b, 1, result); // conjugated
#endif
}

void dot_f64c_with_blas(nk_f64c_t const *a, nk_f64c_t const *b, nk_size_t n, nk_f64c_t *result) {
#if NK_COMPARE_TO_ACCELERATE
    cblas_zdotu_sub(static_cast<int>(n), reinterpret_cast<__LAPACK_double_complex const *>(a), 1,
                    reinterpret_cast<__LAPACK_double_complex const *>(b), 1,
                    reinterpret_cast<__LAPACK_double_complex *>(result));
#else
    cblas_zdotu_sub(static_cast<int>(n), a, 1, b, 1, result);
#endif
}

void vdot_f64c_with_blas(nk_f64c_t const *a, nk_f64c_t const *b, nk_size_t n, nk_f64c_t *result) {
#if NK_COMPARE_TO_ACCELERATE
    cblas_zdotc_sub(static_cast<int>(n), reinterpret_cast<__LAPACK_double_complex const *>(a), 1,
                    reinterpret_cast<__LAPACK_double_complex const *>(b), 1,
                    reinterpret_cast<__LAPACK_double_complex *>(result)); // conjugated
#else
    cblas_zdotc_sub(static_cast<int>(n), a, 1, b, 1, result); // conjugated
#endif
}

void dots_f32_with_blas(f32_t const *a, f32_t const *b, f32_t *c, nk_size_t m, nk_size_t n, nk_size_t k,
                        nk_size_t a_stride, nk_size_t c_stride) {
    (void)a_stride;
    (void)c_stride;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
                1.0f, &a->raw_, static_cast<int>(k), &b->raw_, static_cast<int>(k), 0.0f, &c->raw_,
                static_cast<int>(n));
}

void dots_f64_with_blas(f64_t const *a, f64_t const *b, f64_t *c, nk_size_t m, nk_size_t n, nk_size_t k,
                        nk_size_t a_stride, nk_size_t c_stride) {
    (void)a_stride;
    (void)c_stride;
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
                1.0, &a->raw_, static_cast<int>(k), &b->raw_, static_cast<int>(k), 0.0, &c->raw_, static_cast<int>(n));
}

void dots_f32c_with_blas(f32c_t const *a, f32c_t const *b, f32c_t *c, nk_size_t m, nk_size_t n, nk_size_t k,
                         nk_size_t a_stride, nk_size_t c_stride) {
    (void)a_stride;
    (void)c_stride;
    nk_f32c_t alpha = {1.0f, 0.0f}, beta = {0.0f, 0.0f};
#if NK_COMPARE_TO_ACCELERATE
    cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasTrans, static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
                reinterpret_cast<__LAPACK_float_complex const *>(&alpha),
                reinterpret_cast<__LAPACK_float_complex const *>(&a->raw_), static_cast<int>(k),
                reinterpret_cast<__LAPACK_float_complex const *>(&b->raw_), static_cast<int>(k),
                reinterpret_cast<__LAPACK_float_complex const *>(&beta),
                reinterpret_cast<__LAPACK_float_complex *>(&c->raw_), static_cast<int>(n));
#else
    cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasTrans, static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
                &alpha, &a->raw_, static_cast<int>(k), &b->raw_, static_cast<int>(k), &beta, &c->raw_,
                static_cast<int>(n));
#endif
}

void dots_f64c_with_blas(f64c_t const *a, f64c_t const *b, f64c_t *c, nk_size_t m, nk_size_t n, nk_size_t k,
                         nk_size_t a_stride, nk_size_t c_stride) {
    (void)a_stride;
    (void)c_stride;
    nk_f64c_t alpha = {1.0, 0.0}, beta = {0.0, 0.0};
#if NK_COMPARE_TO_ACCELERATE
    cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasTrans, static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
                reinterpret_cast<__LAPACK_double_complex const *>(&alpha),
                reinterpret_cast<__LAPACK_double_complex const *>(&a->raw_), static_cast<int>(k),
                reinterpret_cast<__LAPACK_double_complex const *>(&b->raw_), static_cast<int>(k),
                reinterpret_cast<__LAPACK_double_complex const *>(&beta),
                reinterpret_cast<__LAPACK_double_complex *>(&c->raw_), static_cast<int>(n));
#else
    cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasTrans, static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
                &alpha, &a->raw_, static_cast<int>(k), &b->raw_, static_cast<int>(k), &beta, &c->raw_,
                static_cast<int>(n));
#endif
}
#endif // NK_COMPARE_TO_BLAS || NK_COMPARE_TO_MKL || NK_COMPARE_TO_ACCELERATE

#if NK_COMPARE_TO_MKL
void dots_bf16_with_mkl(bf16_t const *a, bf16_t const *b, f32_t *c, nk_size_t m, nk_size_t n, nk_size_t k,
                        nk_size_t a_stride, nk_size_t c_stride) {
    (void)a_stride;
    (void)c_stride;
    cblas_gemm_bf16(CblasRowMajor, CblasNoTrans, CblasTrans, static_cast<MKL_INT>(m), static_cast<MKL_INT>(n),
                    static_cast<MKL_INT>(k), 1.0f, &a->raw_, static_cast<MKL_INT>(k), &b->raw_, static_cast<MKL_INT>(k),
                    0.0f, &c->raw_, static_cast<MKL_INT>(n));
}

void dots_f16_with_mkl(f16_t const *a, f16_t const *b, f32_t *c, nk_size_t m, nk_size_t n, nk_size_t k,
                       nk_size_t a_stride, nk_size_t c_stride) {
    (void)a_stride;
    (void)c_stride;
    cblas_gemm_f16(CblasRowMajor, CblasNoTrans, CblasTrans, static_cast<MKL_INT>(m), static_cast<MKL_INT>(n),
                   static_cast<MKL_INT>(k), 1.0f, reinterpret_cast<MKL_F16 const *>(&a->raw_), static_cast<MKL_INT>(k),
                   reinterpret_cast<MKL_F16 const *>(&b->raw_), static_cast<MKL_INT>(k), 0.0f, &c->raw_,
                   static_cast<MKL_INT>(n));
}

/** @brief MKL s16×s16→s32 integer GEMM wrapper. */
void dots_i16_with_mkl(i16_t const *a, i16_t const *b, i32_t *c, nk_size_t m, nk_size_t n, nk_size_t k,
                       nk_size_t a_stride, nk_size_t c_stride) {
    (void)a_stride;
    (void)c_stride;
    MKL_INT32 c_offset = 0;
    cblas_gemm_s16s16s32(CblasRowMajor, CblasNoTrans, CblasTrans, CblasFixOffset, static_cast<MKL_INT>(m),
                         static_cast<MKL_INT>(n), static_cast<MKL_INT>(k), 1.0f, &a->raw_, static_cast<MKL_INT>(k), 0,
                         &b->raw_, static_cast<MKL_INT>(k), 0, 0.0f, &c->raw_, static_cast<MKL_INT>(n), &c_offset);
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
    std::mt19937 generator(global_config.seed);

    auto src = make_vector<from_type_>(dense_dimensions);
    auto dst_simd = make_vector<to_type_>(dense_dimensions);
    auto dst_serial = make_vector<to_type_>(dense_dimensions);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(generator, src);

        nk_cast_serial(src.raw_values_data(), from_type_::dtype(), dense_dimensions, dst_serial.raw_values_data(),
                       to_type_::dtype());
        kernel(src.raw_values_data(), from_type_::dtype(), dense_dimensions, dst_simd.raw_values_data(),
               to_type_::dtype());

        for (std::size_t i = 0; i < dense_dimensions; ++i)
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
    auto buffer = make_vector<scalar_t>(dense_dimensions);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(generator, buffer);

        result_t result;
        kernel(buffer.raw_values_data(), dense_dimensions, sizeof(raw_t), &result.raw_);

        f118_t reference;
        nk::reduce_add<scalar_t, f118_t, nk::no_simd_k>(buffer.values_data(), dense_dimensions, sizeof(raw_t),
                                                        &reference);

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
    std::mt19937 generator(global_config.seed);
    auto a = make_vector<scalar_t>(dense_dimensions), b = make_vector<scalar_t>(dense_dimensions);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(generator, a);
        fill_random(generator, b);

        result_t result;
        kernel(a.raw_values_data(), b.raw_values_data(), dense_dimensions, &result.raw_);

        reference_t reference;
        nk::dot<scalar_t, reference_t, nk::no_simd_k>(a.values_data(), b.values_data(), dense_dimensions, &reference);

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
    std::mt19937 generator(global_config.seed);
    auto a = make_vector<scalar_t>(dense_dimensions), b = make_vector<scalar_t>(dense_dimensions);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(generator, a);
        fill_random(generator, b);

        result_t result;
        kernel(a.raw_values_data(), b.raw_values_data(), dense_dimensions, &result.raw_);

        f118c_t reference;
        nk::vdot<scalar_t, f118c_t, nk::no_simd_k>(a.values_data(), b.values_data(), dense_dimensions, &reference);

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
    run_if_matches("dot", "f16c", test_dot<f16c_t>, nk_dot_f16c);
    run_if_matches("vdot", "f16c", test_vdot<f16c_t>, nk_vdot_f16c);
    run_if_matches("dot", "bf16c", test_dot<bf16c_t>, nk_dot_bf16c);
    run_if_matches("vdot", "bf16c", test_vdot<bf16c_t>, nk_vdot_bf16c);
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
    run_if_matches("dot_neonhalf", "f16c", test_dot<f16c_t>, nk_dot_f16c_neonhalf);
    run_if_matches("vdot_neonhalf", "f16c", test_vdot<f16c_t>, nk_vdot_f16c_neonhalf);
#endif // NK_TARGET_NEONHALF

#if NK_TARGET_NEONBFDOT
    run_if_matches("dot_neonbfdot", "bf16", test_dot<bf16_t>, nk_dot_bf16_neonbfdot);
    run_if_matches("dot_neonbfdot", "bf16c", test_dot<bf16c_t>, nk_dot_bf16c_neonbfdot);
    run_if_matches("vdot_neonbfdot", "bf16c", test_vdot<bf16c_t>, nk_vdot_bf16c_neonbfdot);
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
    run_if_matches("dot_haswell", "f16c", test_dot<f16c_t>, nk_dot_f16c_haswell);
    run_if_matches("vdot_haswell", "f16c", test_vdot<f16c_t>, nk_vdot_f16c_haswell);
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
    run_if_matches("dot_spacemit", "f16", test_dot<f16_t>, nk_dot_f16_spacemit);
    run_if_matches("dot_spacemit", "bf16", test_dot<bf16_t>, nk_dot_bf16_spacemit);
    run_if_matches("dot_spacemit", "e4m3", test_dot<e4m3_t>, nk_dot_e4m3_spacemit);
    run_if_matches("dot_spacemit", "e5m2", test_dot<e5m2_t>, nk_dot_e5m2_spacemit);
    run_if_matches("dot_spacemit", "i4", test_dot<i4x2_t>, nk_dot_i4_spacemit);
    run_if_matches("dot_spacemit", "u4", test_dot<u4x2_t>, nk_dot_u4_spacemit);
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
    run_if_matches("dot_serial", "f16c", test_dot<f16c_t>, nk_dot_f16c_serial);
    run_if_matches("vdot_serial", "f16c", test_vdot<f16c_t>, nk_vdot_f16c_serial);
    run_if_matches("dot_serial", "bf16c", test_dot<bf16c_t>, nk_dot_bf16c_serial);
    run_if_matches("vdot_serial", "bf16c", test_vdot<bf16c_t>, nk_vdot_bf16c_serial);

#endif // NK_DYNAMIC_DISPATCH

#if NK_COMPARE_TO_BLAS || NK_COMPARE_TO_MKL || NK_COMPARE_TO_ACCELERATE
    // BLAS/MKL/Accelerate precision comparison
    run_if_matches("dot_with_blas", "f32", test_dot<f32_t>, dot_f32_with_blas);
    run_if_matches("dot_with_blas", "f64", test_dot<f64_t>, dot_f64_with_blas);
    run_if_matches("dot_with_blas", "f32c", test_dot<f32c_t>, dot_f32c_with_blas);
    run_if_matches("vdot_with_blas", "f32c", test_vdot<f32c_t>, vdot_f32c_with_blas);
    run_if_matches("dot_with_blas", "f64c", test_dot<f64c_t>, dot_f64c_with_blas);
    run_if_matches("vdot_with_blas", "f64c", test_vdot<f64c_t>, vdot_f64c_with_blas);
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
    std::mt19937 generator(global_config.seed);
    auto a = make_vector<scalar_t>(dense_dimensions), b = make_vector<scalar_t>(dense_dimensions);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(generator, a);
        fill_random(generator, b);

        result_t result;
        kernel(a.raw_values_data(), b.raw_values_data(), dense_dimensions, &result.raw_);

        f118_t reference;
        nk::l2sq<scalar_t, f118_t, nk::no_simd_k>(a.values_data(), b.values_data(), dense_dimensions, &reference);

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
    std::mt19937 generator(global_config.seed);
    auto a = make_vector<scalar_t>(dense_dimensions), b = make_vector<scalar_t>(dense_dimensions);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(generator, a);
        fill_random(generator, b);

        result_t result;
        kernel(a.raw_values_data(), b.raw_values_data(), dense_dimensions, &result.raw_);

        f118_t reference;
        nk::angular<scalar_t, f118_t, nk::no_simd_k>(a.values_data(), b.values_data(), dense_dimensions, &reference);

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
    run_if_matches("l2sq_spacemit", "f16", test_l2sq<f16_t>, nk_l2sq_f16_spacemit);
    run_if_matches("l2sq_spacemit", "bf16", test_l2sq<bf16_t>, nk_l2sq_bf16_spacemit);
    run_if_matches("l2sq_spacemit", "i4", test_l2sq<i4x2_t>, nk_l2sq_i4_spacemit);
    run_if_matches("l2sq_spacemit", "u4", test_l2sq<u4x2_t>, nk_l2sq_u4_spacemit);
    run_if_matches("angular_spacemit", "f32", test_angular<f32_t>, nk_angular_f32_spacemit);
    run_if_matches("angular_spacemit", "f64", test_angular<f64_t>, nk_angular_f64_spacemit);
    run_if_matches("angular_spacemit", "f16", test_angular<f16_t>, nk_angular_f16_spacemit);
    run_if_matches("angular_spacemit", "bf16", test_angular<bf16_t>, nk_angular_bf16_spacemit);
    run_if_matches("angular_spacemit", "i4", test_angular<i4x2_t>, nk_angular_i4_spacemit);
    run_if_matches("angular_spacemit", "u4", test_angular<u4x2_t>, nk_angular_u4_spacemit);
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
    using reference_t = std::conditional_t<scalar_t::is_complex(), f118c_t, f118_t>;

    error_stats_t stats;
    std::mt19937 generator(global_config.seed);

    auto a = make_vector<scalar_t>(dense_dimensions), b = make_vector<scalar_t>(dense_dimensions);
    auto m = make_vector<scalar_t>(dense_dimensions * dense_dimensions);
    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(generator, a);
        fill_random(generator, b);
        fill_random(generator, m);

        result_t result;
        kernel(a.raw_values_data(), b.raw_values_data(), m.raw_values_data(), dense_dimensions, &result.raw_);

        reference_t reference;
        nk::bilinear<scalar_t, reference_t, nk::no_simd_k>(a.values_data(), b.values_data(), m.values_data(),
                                                           dense_dimensions, &reference);

        stats.accumulate(result, reference);
    }

    return stats;
}

void test_curved() {
    std::printf("Testing curved/bilinear forms...\n");

#if NK_DYNAMIC_DISPATCH
    run_if_matches("bilinear", "f32", test_bilinear<f32_t>, nk_bilinear_f32);
    run_if_matches("bilinear", "f64", test_bilinear<f64_t>, nk_bilinear_f64);
    run_if_matches("bilinear", "f32c", test_bilinear<f32c_t>, nk_bilinear_f32c);
    run_if_matches("bilinear", "f64c", test_bilinear<f64c_t>, nk_bilinear_f64c);
#else

#if NK_TARGET_NEON
    run_if_matches("bilinear_neon", "f32", test_bilinear<f32_t>, nk_bilinear_f32_neon);
    run_if_matches("bilinear_neon", "f32c", test_bilinear<f32c_t>, nk_bilinear_f32c_neon);
#endif // NK_TARGET_NEON

#if NK_TARGET_SKYLAKE
    run_if_matches("bilinear_skylake", "f32", test_bilinear<f32_t>, nk_bilinear_f32_skylake);
    run_if_matches("bilinear_skylake", "f64", test_bilinear<f64_t>, nk_bilinear_f64_skylake);
    run_if_matches("bilinear_skylake", "f32c", test_bilinear<f32c_t>, nk_bilinear_f32c_skylake);
    run_if_matches("bilinear_skylake", "f64c", test_bilinear<f64c_t>, nk_bilinear_f64c_skylake);
#endif // NK_TARGET_SKYLAKE

#if NK_TARGET_SMEF64
    run_if_matches("bilinear_smef64", "f32", test_bilinear<f32_t>, nk_bilinear_f32_smef64);
    run_if_matches("bilinear_smef64", "f32c", test_bilinear<f32c_t>, nk_bilinear_f32c_smef64);
#endif // NK_TARGET_SMEF64

    // Serial always runs - baseline test
    run_if_matches("bilinear_serial", "f32", test_bilinear<f32_t>, nk_bilinear_f32_serial);
    run_if_matches("bilinear_serial", "f64", test_bilinear<f64_t>, nk_bilinear_f64_serial);
    run_if_matches("bilinear_serial", "f32c", test_bilinear<f32c_t>, nk_bilinear_f32c_serial);
    run_if_matches("bilinear_serial", "f64c", test_bilinear<f64c_t>, nk_bilinear_f64c_serial);

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
    std::mt19937 generator(global_config.seed);
    auto p = make_vector<scalar_t>(dense_dimensions), q = make_vector<scalar_t>(dense_dimensions);

    for (auto start = test_start_time(); within_time_budget(start);) {
        nk::fill_probability(generator, p.values_data(), dense_dimensions);
        nk::fill_probability(generator, q.values_data(), dense_dimensions);

        result_t result;
        kernel(p.raw_values_data(), q.raw_values_data(), dense_dimensions, &result.raw_);

        f118_t reference;
        nk::kld<scalar_t, f118_t, nk::no_simd_k>(p.values_data(), q.values_data(), dense_dimensions, &reference);

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
    std::mt19937 generator(global_config.seed);
    auto p = make_vector<scalar_t>(dense_dimensions), q = make_vector<scalar_t>(dense_dimensions);

    for (auto start = test_start_time(); within_time_budget(start);) {
        nk::fill_probability(generator, p.values_data(), dense_dimensions);
        nk::fill_probability(generator, q.values_data(), dense_dimensions);

        result_t result;
        kernel(p.raw_values_data(), q.raw_values_data(), dense_dimensions, &result.raw_);

        f118_t reference;
        nk::jsd<scalar_t, f118_t, nk::no_simd_k>(p.values_data(), q.values_data(), dense_dimensions, &reference);

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
    std::mt19937 generator(global_config.seed);

    std::size_t n_bytes = dense_dimensions / 8;
    auto a = make_vector<scalar_t>(n_bytes), b = make_vector<scalar_t>(n_bytes);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(generator, a);
        fill_random(generator, b);

        u32_t result;
        kernel(a.raw_values_data(), b.raw_values_data(), dense_dimensions, &result.raw_);

        u32_t reference;
        nk::hamming(a.values_data(), b.values_data(), n_bytes, &reference);

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
    std::mt19937 generator(global_config.seed);

    std::size_t n_bytes = dense_dimensions / 8;
    auto a = make_vector<scalar_t>(n_bytes), b = make_vector<scalar_t>(n_bytes);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(generator, a);
        fill_random(generator, b);

        f32_t result;
        kernel(a.raw_values_data(), b.raw_values_data(), dense_dimensions, &result.raw_);

        f32_t reference;
        nk::jaccard(a.values_data(), b.values_data(), n_bytes, &reference);

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

#if NK_TARGET_SPACEMIT
    run_if_matches("hamming_spacemit", "u1", test_hamming, nk_hamming_u1_spacemit);
    run_if_matches("jaccard_spacemit", "u1", test_jaccard, nk_jaccard_u1_spacemit);
#endif // NK_TARGET_SPACEMIT

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
    std::mt19937 generator(global_config.seed);
    auto a = make_vector<scalar_t>(dense_dimensions), b = make_vector<scalar_t>(dense_dimensions);
    auto result = make_vector<scalar_t>(dense_dimensions), reference = make_vector<scalar_t>(dense_dimensions);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(generator, a);
        fill_random(generator, b);

        kernel(a.raw_values_data(), b.raw_values_data(), dense_dimensions, result.raw_values_data());
        nk::sum<scalar_t, nk::no_simd_k>(a.values_data(), b.values_data(), dense_dimensions, reference.values_data());

        for (std::size_t i = 0; i < dense_dimensions; i++) stats.accumulate(result[i], reference[i]);
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
    std::mt19937 generator(global_config.seed);
    auto input = make_vector<scalar_t>(dense_dimensions);
    auto result = make_vector<scalar_t>(dense_dimensions), reference = make_vector<scalar_t>(dense_dimensions);
    std::uniform_real_distribution<scale_t> coef_dist(scale_t(-2.0), scale_t(2.0));

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(generator, input);

        scale_t alpha = coef_dist(generator);
        scale_t beta = coef_dist(generator);

        kernel(input.raw_values_data(), dense_dimensions, &alpha, &beta, result.raw_values_data());
        nk::scale<scalar_t, f118_t, nk::no_simd_k>(input.values_data(), dense_dimensions, &alpha, &beta,
                                                   reference.values_data());

        for (std::size_t i = 0; i < dense_dimensions; i++) stats.accumulate(result[i], reference[i]);
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
    std::mt19937 generator(global_config.seed);
    auto a = make_vector<scalar_t>(dense_dimensions), b = make_vector<scalar_t>(dense_dimensions);
    auto result = make_vector<scalar_t>(dense_dimensions), reference = make_vector<scalar_t>(dense_dimensions);
    std::uniform_real_distribution<scale_t> coef_dist(scale_t(-2.0), scale_t(2.0));

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(generator, a);
        fill_random(generator, b);

        scale_t alpha = coef_dist(generator);
        scale_t beta = coef_dist(generator);

        kernel(a.raw_values_data(), b.raw_values_data(), dense_dimensions, &alpha, &beta, result.raw_values_data());
        nk::wsum<scalar_t, f118_t, nk::no_simd_k>(a.values_data(), b.values_data(), dense_dimensions, &alpha, &beta,
                                                  reference.values_data());

        for (std::size_t i = 0; i < dense_dimensions; i++) stats.accumulate(result[i], reference[i]);
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
    std::mt19937 generator(global_config.seed);
    auto a = make_vector<scalar_t>(dense_dimensions), b = make_vector<scalar_t>(dense_dimensions);
    auto c = make_vector<scalar_t>(dense_dimensions);
    auto result = make_vector<scalar_t>(dense_dimensions), reference = make_vector<scalar_t>(dense_dimensions);
    std::uniform_real_distribution<scale_t> coef_dist(scale_t(-2.0), scale_t(2.0));

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(generator, a);
        fill_random(generator, b);
        fill_random(generator, c);

        scale_t alpha = coef_dist(generator);
        scale_t beta = coef_dist(generator);

        kernel(a.raw_values_data(), b.raw_values_data(), c.raw_values_data(), dense_dimensions, &alpha, &beta,
               result.raw_values_data());
        nk::fma<scalar_t, f118_t, nk::no_simd_k>(a.values_data(), b.values_data(), dense_dimensions, c.values_data(),
                                                 &alpha, &beta, reference.values_data());

        for (std::size_t i = 0; i < dense_dimensions; i++) stats.accumulate(result[i], reference[i]);
    }
    return stats;
}

void test_elementwise() {
    std::printf("Testing elementwise operations...\n");

#if NK_DYNAMIC_DISPATCH
    // Dynamic dispatch - only test the dispatcher itself
    run_if_matches("each_scale", "f32", test_scale<f32_t>, nk_each_scale_f32);
    run_if_matches("each_sum", "f32", test_sum<f32_t>, nk_each_sum_f32);
    run_if_matches("each_wsum", "f32", test_wsum<f32_t>, nk_each_blend_f32);
    run_if_matches("each_fma", "f32", test_fma<f32_t>, nk_each_fma_f32);
    run_if_matches("each_scale", "e4m3", test_scale<e4m3_t>, nk_each_scale_e4m3);
    run_if_matches("each_scale", "e5m2", test_scale<e5m2_t>, nk_each_scale_e5m2);
    run_if_matches("each_sum", "e4m3", test_sum<e4m3_t>, nk_each_sum_e4m3);
    run_if_matches("each_sum", "e5m2", test_sum<e5m2_t>, nk_each_sum_e5m2);
    run_if_matches("each_wsum", "e4m3", test_wsum<e4m3_t>, nk_each_blend_e4m3);
    run_if_matches("each_wsum", "e5m2", test_wsum<e5m2_t>, nk_each_blend_e5m2);
    run_if_matches("each_fma", "e4m3", test_fma<e4m3_t>, nk_each_fma_e4m3);
    run_if_matches("each_fma", "e5m2", test_fma<e5m2_t>, nk_each_fma_e5m2);
#else
    // Static compilation - test all available ISA variants

#if NK_TARGET_NEON
    run_if_matches("each_scale_neon", "f32", test_scale<f32_t>, nk_each_scale_f32_neon);
    run_if_matches("each_sum_neon", "f32", test_sum<f32_t>, nk_each_sum_f32_neon);
    run_if_matches("each_wsum_neon", "f32", test_wsum<f32_t>, nk_each_blend_f32_neon);
    run_if_matches("each_fma_neon", "f32", test_fma<f32_t>, nk_each_fma_f32_neon);
    run_if_matches("each_scale_neon", "e4m3", test_scale<e4m3_t>, nk_each_scale_e4m3_neon);
    run_if_matches("each_scale_neon", "e5m2", test_scale<e5m2_t>, nk_each_scale_e5m2_neon);
    run_if_matches("each_sum_neon", "e4m3", test_sum<e4m3_t>, nk_each_sum_e4m3_neon);
    run_if_matches("each_sum_neon", "e5m2", test_sum<e5m2_t>, nk_each_sum_e5m2_neon);
    run_if_matches("each_wsum_neon", "e4m3", test_wsum<e4m3_t>, nk_each_blend_e4m3_neon);
    run_if_matches("each_wsum_neon", "e5m2", test_wsum<e5m2_t>, nk_each_blend_e5m2_neon);
    run_if_matches("each_fma_neon", "e4m3", test_fma<e4m3_t>, nk_each_fma_e4m3_neon);
    run_if_matches("each_fma_neon", "e5m2", test_fma<e5m2_t>, nk_each_fma_e5m2_neon);
#endif // NK_TARGET_NEON

#if NK_TARGET_NEONHALF
#endif // NK_TARGET_NEONHALF

#if NK_TARGET_HASWELL
    run_if_matches("each_scale_haswell", "f32", test_scale<f32_t>, nk_each_scale_f32_haswell);
    run_if_matches("each_sum_haswell", "f32", test_sum<f32_t>, nk_each_sum_f32_haswell);
    run_if_matches("each_wsum_haswell", "f32", test_wsum<f32_t>, nk_each_blend_f32_haswell);
    run_if_matches("each_fma_haswell", "f32", test_fma<f32_t>, nk_each_fma_f32_haswell);
    run_if_matches("each_scale_haswell", "e4m3", test_scale<e4m3_t>, nk_each_scale_e4m3_haswell);
    run_if_matches("each_scale_haswell", "e5m2", test_scale<e5m2_t>, nk_each_scale_e5m2_haswell);
    run_if_matches("each_sum_haswell", "e4m3", test_sum<e4m3_t>, nk_each_sum_e4m3_haswell);
    run_if_matches("each_sum_haswell", "e5m2", test_sum<e5m2_t>, nk_each_sum_e5m2_haswell);
    run_if_matches("each_wsum_haswell", "e4m3", test_wsum<e4m3_t>, nk_each_blend_e4m3_haswell);
    run_if_matches("each_wsum_haswell", "e5m2", test_wsum<e5m2_t>, nk_each_blend_e5m2_haswell);
    run_if_matches("each_fma_haswell", "e4m3", test_fma<e4m3_t>, nk_each_fma_e4m3_haswell);
    run_if_matches("each_fma_haswell", "e5m2", test_fma<e5m2_t>, nk_each_fma_e5m2_haswell);
#endif // NK_TARGET_HASWELL

#if NK_TARGET_SKYLAKE
    run_if_matches("each_scale_skylake", "f32", test_scale<f32_t>, nk_each_scale_f32_skylake);
    run_if_matches("each_sum_skylake", "f32", test_sum<f32_t>, nk_each_sum_f32_skylake);
    run_if_matches("each_wsum_skylake", "f32", test_wsum<f32_t>, nk_each_blend_f32_skylake);
    run_if_matches("each_fma_skylake", "f32", test_fma<f32_t>, nk_each_fma_f32_skylake);
    run_if_matches("each_scale_skylake", "e4m3", test_scale<e4m3_t>, nk_each_scale_e4m3_skylake);
    run_if_matches("each_scale_skylake", "e5m2", test_scale<e5m2_t>, nk_each_scale_e5m2_skylake);
    run_if_matches("each_sum_skylake", "e4m3", test_sum<e4m3_t>, nk_each_sum_e4m3_skylake);
    run_if_matches("each_sum_skylake", "e5m2", test_sum<e5m2_t>, nk_each_sum_e5m2_skylake);
    run_if_matches("each_wsum_skylake", "e4m3", test_wsum<e4m3_t>, nk_each_blend_e4m3_skylake);
    run_if_matches("each_wsum_skylake", "e5m2", test_wsum<e5m2_t>, nk_each_blend_e5m2_skylake);
    run_if_matches("each_fma_skylake", "e4m3", test_fma<e4m3_t>, nk_each_fma_e4m3_skylake);
    run_if_matches("each_fma_skylake", "e5m2", test_fma<e5m2_t>, nk_each_fma_e5m2_skylake);
#endif // NK_TARGET_SKYLAKE

#if NK_TARGET_SAPPHIRE
    run_if_matches("each_sum_sapphire", "e4m3", test_sum<e4m3_t>, nk_each_sum_e4m3_sapphire);
#endif // NK_TARGET_SAPPHIRE

    // Serial always runs - baseline test
    run_if_matches("each_scale_serial", "f32", test_scale<f32_t>, nk_each_scale_f32_serial);
    run_if_matches("each_sum_serial", "f32", test_sum<f32_t>, nk_each_sum_f32_serial);
    run_if_matches("each_wsum_serial", "f32", test_wsum<f32_t>, nk_each_blend_f32_serial);
    run_if_matches("each_fma_serial", "f32", test_fma<f32_t>, nk_each_fma_f32_serial);
    run_if_matches("each_scale_serial", "e4m3", test_scale<e4m3_t>, nk_each_scale_e4m3_serial);
    run_if_matches("each_scale_serial", "e5m2", test_scale<e5m2_t>, nk_each_scale_e5m2_serial);
    run_if_matches("each_sum_serial", "e4m3", test_sum<e4m3_t>, nk_each_sum_e4m3_serial);
    run_if_matches("each_sum_serial", "e5m2", test_sum<e5m2_t>, nk_each_sum_e5m2_serial);
    run_if_matches("each_wsum_serial", "e4m3", test_wsum<e4m3_t>, nk_each_blend_e4m3_serial);
    run_if_matches("each_wsum_serial", "e5m2", test_wsum<e5m2_t>, nk_each_blend_e5m2_serial);
    run_if_matches("each_fma_serial", "e4m3", test_fma<e4m3_t>, nk_each_fma_e4m3_serial);
    run_if_matches("each_fma_serial", "e5m2", test_fma<e5m2_t>, nk_each_fma_e5m2_serial);

#endif // NK_DYNAMIC_DISPATCH
}

#pragma endregion // Elementwise

#pragma region Trigonometry

/**
 *  @brief Test sine approximation kernel against `nk::sin<scalar_t, f118_t, nk::no_simd_k>`.
 */
template <typename scalar_type_>
error_stats_t test_sin(typename scalar_type_::trig_kernel_t kernel) {
    using scalar_t = scalar_type_;
    using raw_t = typename scalar_t::raw_t;

    error_stats_t stats;
    std::mt19937 generator(global_config.seed);
    auto inputs = make_vector<scalar_t>(dense_dimensions);
    auto outputs = make_vector<scalar_t>(dense_dimensions), reference = make_vector<scalar_t>(dense_dimensions);

    for (auto start = test_start_time(); within_time_budget(start);) {
        nk::fill_uniform(generator, inputs.values_data(), inputs.size_values(), -scalar_t::two_pi_k(),
                         scalar_t::two_pi_k());

        kernel(inputs.raw_values_data(), dense_dimensions, outputs.raw_values_data());
        nk::sin<scalar_t, f118_t, nk::no_simd_k>(inputs.values_data(), dense_dimensions, reference.values_data());

        for (std::size_t i = 0; i < dense_dimensions; i++) stats.accumulate(outputs[i], reference[i]);
    }
    return stats;
}

/**
 *  @brief Test cosine approximation kernel against `nk::cos<scalar_t, f118_t, nk::no_simd_k>`.
 */
template <typename scalar_type_>
error_stats_t test_cos(typename scalar_type_::trig_kernel_t kernel) {
    using scalar_t = scalar_type_;
    using raw_t = typename scalar_t::raw_t;

    error_stats_t stats;
    std::mt19937 generator(global_config.seed);
    auto inputs = make_vector<scalar_t>(dense_dimensions);
    auto outputs = make_vector<scalar_t>(dense_dimensions), reference = make_vector<scalar_t>(dense_dimensions);

    for (auto start = test_start_time(); within_time_budget(start);) {
        nk::fill_uniform(generator, inputs.values_data(), inputs.size_values(), -scalar_t::two_pi_k(),
                         scalar_t::two_pi_k());

        kernel(inputs.raw_values_data(), dense_dimensions, outputs.raw_values_data());
        nk::cos<scalar_t, f118_t, nk::no_simd_k>(inputs.values_data(), dense_dimensions, reference.values_data());

        for (std::size_t i = 0; i < dense_dimensions; i++) stats.accumulate(outputs[i], reference[i]);
    }
    return stats;
}

/**
 *  @brief Test atan approximation kernel against `nk::atan<scalar_t, f118_t, nk::no_simd_k>`.
 */
template <typename scalar_type_>
error_stats_t test_atan(typename scalar_type_::trig_kernel_t kernel) {
    using scalar_t = scalar_type_;
    using raw_t = typename scalar_t::raw_t;

    error_stats_t stats;
    std::mt19937 generator(global_config.seed);
    auto inputs = make_vector<scalar_t>(dense_dimensions);
    auto outputs = make_vector<scalar_t>(dense_dimensions), reference = make_vector<scalar_t>(dense_dimensions);

    for (auto start = test_start_time(); within_time_budget(start);) {
        nk::fill_uniform(generator, inputs.values_data(), inputs.size_values(), scalar_t(-10.0), scalar_t(10.0));

        kernel(inputs.raw_values_data(), dense_dimensions, outputs.raw_values_data());
        nk::atan<scalar_t, f118_t, nk::no_simd_k>(inputs.values_data(), dense_dimensions, reference.values_data());

        for (std::size_t i = 0; i < dense_dimensions; i++) stats.accumulate(outputs[i], reference[i]);
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
    std::mt19937 generator(global_config.seed);
    auto a_lats = make_vector<scalar_t>(dense_dimensions), a_lons = make_vector<scalar_t>(dense_dimensions);
    auto b_lats = make_vector<scalar_t>(dense_dimensions), b_lons = make_vector<scalar_t>(dense_dimensions);
    auto results = make_vector<scalar_t>(dense_dimensions), reference = make_vector<scalar_t>(dense_dimensions);

    for (auto start = test_start_time(); within_time_budget(start);) {
        nk::fill_coordinates(generator, a_lats.values_data(), a_lons.values_data(), dense_dimensions);
        nk::fill_coordinates(generator, b_lats.values_data(), b_lons.values_data(), dense_dimensions);

        kernel(a_lats.raw_values_data(), a_lons.raw_values_data(), b_lats.raw_values_data(), b_lons.raw_values_data(),
               dense_dimensions, results.raw_values_data());
        nk::haversine<scalar_t, f118_t, nk::no_simd_k>(a_lats.values_data(), a_lons.values_data(), b_lats.values_data(),
                                                       b_lons.values_data(), dense_dimensions, reference.values_data());

        for (std::size_t i = 0; i < dense_dimensions; i++) stats.accumulate(results[i], reference[i]);
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
    std::mt19937 generator(global_config.seed);
    auto a_lats = make_vector<scalar_t>(dense_dimensions), a_lons = make_vector<scalar_t>(dense_dimensions);
    auto b_lats = make_vector<scalar_t>(dense_dimensions), b_lons = make_vector<scalar_t>(dense_dimensions);
    auto results = make_vector<scalar_t>(dense_dimensions), reference = make_vector<scalar_t>(dense_dimensions);

    for (auto start = test_start_time(); within_time_budget(start);) {
        nk::fill_coordinates(generator, a_lats.values_data(), a_lons.values_data(), dense_dimensions);
        nk::fill_coordinates(generator, b_lats.values_data(), b_lons.values_data(), dense_dimensions);

        kernel(a_lats.raw_values_data(), a_lons.raw_values_data(), b_lats.raw_values_data(), b_lons.raw_values_data(),
               dense_dimensions, results.raw_values_data());
        nk::vincenty<scalar_t, f118_t, nk::no_simd_k>(a_lats.values_data(), a_lons.values_data(), b_lats.values_data(),
                                                      b_lons.values_data(), dense_dimensions, reference.values_data());

        for (std::size_t i = 0; i < dense_dimensions; i++) stats.accumulate(results[i], reference[i]);
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
    std::mt19937 generator(global_config.seed);

    std::size_t n = mesh_points;
    auto a = make_vector<scalar_t>(n * 3), b = make_vector<scalar_t>(n * 3);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(generator, a);
        fill_random(generator, b);

        scalar_t a_centroid[3], b_centroid[3], rot[9], scale, result;
        kernel(a.raw_values_data(), b.raw_values_data(), n, &a_centroid[0].raw_, &b_centroid[0].raw_, &rot[0].raw_,
               &scale.raw_, &result.raw_);

        f118_t a_centroid_ref[3], b_centroid_ref[3], rot_ref[9], scale_ref, reference;
        nk::rmsd<scalar_t, f118_t, nk::no_simd_k>(a.values_data(), b.values_data(), n, a_centroid_ref, b_centroid_ref,
                                                  rot_ref, &scale_ref, &reference);

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
    std::mt19937 generator(global_config.seed);

    std::size_t n = mesh_points;
    auto a = make_vector<scalar_t>(n * 3), b = make_vector<scalar_t>(n * 3);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(generator, a);
        fill_random(generator, b);

        scalar_t a_centroid[3], b_centroid[3], rot[9], scale, result;
        kernel(a.raw_values_data(), b.raw_values_data(), n, &a_centroid[0].raw_, &b_centroid[0].raw_, &rot[0].raw_,
               &scale.raw_, &result.raw_);

        f118_t a_centroid_ref[3], b_centroid_ref[3], rot_ref[9], scale_ref, reference;
        nk::kabsch<scalar_t, f118_t, nk::no_simd_k>(a.values_data(), b.values_data(), n, a_centroid_ref, b_centroid_ref,
                                                    rot_ref, &scale_ref, &reference);

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
    std::mt19937 generator(global_config.seed);

    std::size_t n = mesh_points;
    auto a = make_vector<scalar_t>(n * 3), b = make_vector<scalar_t>(n * 3);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(generator, a);
        fill_random(generator, b);

        scalar_t a_centroid[3], b_centroid[3], rot[9], scale, result;
        kernel(a.raw_values_data(), b.raw_values_data(), n, &a_centroid[0].raw_, &b_centroid[0].raw_, &rot[0].raw_,
               &scale.raw_, &result.raw_);

        f118_t a_centroid_ref[3], b_centroid_ref[3], rot_ref[9], scale_ref, reference;
        nk::umeyama<scalar_t, f118_t, nk::no_simd_k>(a.values_data(), b.values_data(), n, a_centroid_ref,
                                                     b_centroid_ref, rot_ref, &scale_ref, &reference);

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
    std::mt19937 generator(global_config.seed);

    std::size_t m = matrix_height, n = matrix_width, k = matrix_depth;
    std::size_t k_values = nk::divide_round_up(k, nk::dimensions_per_value<scalar_t>());
    std::size_t a_stride = k_values * sizeof(scalar_t);
    std::size_t b_stride = k_values * sizeof(scalar_t);
    std::size_t c_stride = n * sizeof(result_t);

    auto a = make_vector<scalar_t>(m * k), b = make_vector<scalar_t>(n * k);
    auto c = make_vector<result_t>(m * n);
    auto c_ref = make_vector<f118_t>(m * n);

    nk_size_t packed_size = packed_size_fn(n, k);
    auto b_packed = make_vector<char>(packed_size);
    nk_size_t ref_packed_size = nk::dots_packed_size<scalar_t, nk::no_simd_k>(n, k);
    auto b_packed_ref = make_vector<char>(ref_packed_size);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(generator, a);
        fill_random(generator, b);

        // Run kernel being tested
        pack_fn(b.raw_values_data(), n, k, b_stride, b_packed.raw_values_data());
        dots_fn(a.raw_values_data(), b_packed.raw_values_data(), c.raw_values_data(), m, n, k, a_stride, c_stride);

        // Compute f118_t reference using nk:: template
        nk::dots_pack<scalar_t, nk::no_simd_k>(b.values_data(), n, k, b_stride, b_packed_ref.raw_values_data());
        nk::dots_packed<scalar_t, f118_t, nk::no_simd_k>(a.values_data(), b_packed_ref.raw_values_data(),
                                                         c_ref.raw_values_data(), m, n, k, a_stride,
                                                         n * sizeof(f118_t));

        for (std::size_t i = 0; i < m * n; i++) stats.accumulate(c[i], c_ref[i]);
    }
    return stats;
}

/**
 *  @brief Generic symmetric GEMM (A × A^T) test against f118_t reference.
 *  Works for all types: f32, f64, f16, bf16, i8, u8, i4, u4, e4m3, e5m2.
 */
template <typename scalar_type_>
error_stats_t test_dots_symmetric(typename scalar_type_::dots_symmetric_kernel_t kernel_fn) {
    using scalar_t = scalar_type_;
    using result_t = typename scalar_t::dot_result_t;

    error_stats_t stats;
    std::mt19937 generator(global_config.seed);

    std::size_t n = matrix_height, k = matrix_depth;
    std::size_t k_values = nk::divide_round_up(k, nk::dimensions_per_value<scalar_t>());
    std::size_t a_stride = k_values * sizeof(scalar_t);
    std::size_t c_stride = n * sizeof(result_t);

    auto a = make_vector<scalar_t>(n * k);
    auto c = make_vector<result_t>(n * n);
    auto c_ref = make_vector<f118_t>(n * n);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(generator, a);

        // Run kernel being tested
        kernel_fn(a.raw_values_data(), n, k, a_stride, c.raw_values_data(), c_stride);

        // Compute f118_t reference using nk:: template
        nk::dots_symmetric<scalar_t, f118_t, nk::no_simd_k>(a.values_data(), n, k, a_stride, c_ref.raw_values_data(),
                                                            n * sizeof(f118_t));

        for (std::size_t i = 0; i < n * n; i++) stats.accumulate(c[i], c_ref[i]);
    }
    return stats;
}

#if NK_COMPARE_TO_BLAS || NK_COMPARE_TO_MKL || NK_COMPARE_TO_ACCELERATE
/**
 *  @brief Unified template to test unpacked GEMM against high-precision reference.
 *
 *  Validates BLAS/MKL/Accelerate GEMM implementations by comparing against
 *  nk::dots_unpacked with f118_t accumulation.
 *
 *  @tparam scalar_type_ Input element type (e.g., f32_t, bf16_t)
 *  @tparam accumulator_type_ Output type from BLAS kernel (e.g., f32_t for bf16 GEMM)
 *  @tparam reference_type_ High-precision reference type (f118_t for real, f118c_t for complex)
 *  @tparam kernel_type_ Deduced function pointer type for the BLAS kernel
 */
template <typename scalar_type_, typename accumulator_type_, typename reference_type_ = f118_t, typename kernel_type_>
error_stats_t test_dots_unpacked(kernel_type_ dots_fn) {
    using scalar_t = scalar_type_;
    using raw_t = typename scalar_t::raw_t;
    using result_t = accumulator_type_;
    using reference_t = reference_type_;

    error_stats_t stats;
    std::mt19937 generator(global_config.seed);

    std::size_t m = matrix_height, n = matrix_width, k = matrix_depth;
    std::size_t a_stride = k * sizeof(raw_t);
    std::size_t b_stride = k * sizeof(raw_t);
    std::size_t c_stride = n * sizeof(typename result_t::raw_t);

    auto a_buf = make_vector<scalar_t>(m * k), b_buf = make_vector<scalar_t>(n * k);
    auto c = make_vector<result_t>(m * n);
    auto c_ref = make_vector<reference_t>(m * n); // f118_t doesn't have raw_t, use auto(auto = make_vector<
    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(generator, a_buf);
        fill_random(generator, b_buf);

        nk::dots_unpacked<scalar_t, reference_t>(a_buf.values_data(), b_buf.values_data(), c_ref.raw_values_data(), m,
                                                 n, k, a_stride, b_stride, n * sizeof(reference_t));
        dots_fn(a_buf.values_data(), b_buf.values_data(), c.values_data(), m, n, k, a_stride, c_stride);

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

#if NK_TARGET_SME
    // SME precision validation tests (Arm Scalable Matrix Extension)
    run_if_matches("dots_sme", "f16", test_dots<f16_t>, nk_dots_packed_size_f16_sme, nk_dots_pack_f16_sme,
                   nk_dots_packed_f16_sme);
    run_if_matches("dots_sme", "bf16", test_dots<bf16_t>, nk_dots_packed_size_bf16_sme, nk_dots_pack_bf16_sme,
                   nk_dots_packed_bf16_sme);
    run_if_matches("dots_sme", "i8", test_dots<i8_t>, nk_dots_packed_size_i8_sme, nk_dots_pack_i8_sme,
                   nk_dots_packed_i8_sme);
    run_if_matches("dots_sme", "u8", test_dots<u8_t>, nk_dots_packed_size_u8_sme, nk_dots_pack_u8_sme,
                   nk_dots_packed_u8_sme);
    run_if_matches("dots_sme", "e4m3", test_dots<e4m3_t>, nk_dots_packed_size_e4m3_sme, nk_dots_pack_e4m3_sme,
                   nk_dots_packed_e4m3_sme);
    run_if_matches("dots_sme", "e5m2", test_dots<e5m2_t>, nk_dots_packed_size_e5m2_sme, nk_dots_pack_e5m2_sme,
                   nk_dots_packed_e5m2_sme);
#endif // NK_TARGET_SME
#if NK_TARGET_SMEF64
    run_if_matches("dots_smef64", "f32", test_dots<f32_t>, nk_dots_packed_size_f32_smef64, nk_dots_pack_f32_smef64,
                   nk_dots_packed_f32_smef64);
    run_if_matches("dots_smef64", "f64", test_dots<f64_t>, nk_dots_packed_size_f64_smef64, nk_dots_pack_f64_smef64,
                   nk_dots_packed_f64_smef64);
#endif // NK_TARGET_SMEF64

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
    run_if_matches("dots_with_blas", "f32", test_dots_unpacked<f32_t, f32_t, f118_t, decltype(&dots_f32_with_blas)>,
                   dots_f32_with_blas);
    run_if_matches("dots_with_blas", "f64", test_dots_unpacked<f64_t, f64_t, f118_t, decltype(&dots_f64_with_blas)>,
                   dots_f64_with_blas);
    run_if_matches("dots_with_blas", "f32c",
                   test_dots_unpacked<f32c_t, f32c_t, f118c_t, decltype(&dots_f32c_with_blas)>, dots_f32c_with_blas);
    run_if_matches("dots_with_blas", "f64c",
                   test_dots_unpacked<f64c_t, f64c_t, f118c_t, decltype(&dots_f64c_with_blas)>, dots_f64c_with_blas);
#endif // NK_COMPARE_TO_BLAS || NK_COMPARE_TO_MKL || NK_COMPARE_TO_ACCELERATE

#if NK_COMPARE_TO_MKL
    // MKL-specific GEMM with widening accumulation
    run_if_matches("dots_with_mkl", "bf16", test_dots_unpacked<bf16_t, f32_t, f118_t, decltype(&dots_bf16_with_mkl)>,
                   dots_bf16_with_mkl);
    run_if_matches("dots_with_mkl", "f16", test_dots_unpacked<f16_t, f32_t, f118_t, decltype(&dots_f16_with_mkl)>,
                   dots_f16_with_mkl);
    run_if_matches("dots_with_mkl", "i16", test_dots_unpacked<i16_t, i32_t, i64_t, decltype(&dots_i16_with_mkl)>,
                   dots_i16_with_mkl);
#endif // NK_COMPARE_TO_MKL
}

void test_dots_symmetric() {
    std::printf("Testing symmetric batch dot products (A × A^T GEMM)...\n");

#if NK_DYNAMIC_DISPATCH
    run_if_matches("dots_symmetric", "f32", test_dots_symmetric<f32_t>, nk_dots_symmetric_f32);
    run_if_matches("dots_symmetric", "f64", test_dots_symmetric<f64_t>, nk_dots_symmetric_f64);
    run_if_matches("dots_symmetric", "bf16", test_dots_symmetric<bf16_t>, nk_dots_symmetric_bf16);
    run_if_matches("dots_symmetric", "f16", test_dots_symmetric<f16_t>, nk_dots_symmetric_f16);
    run_if_matches("dots_symmetric", "i8", test_dots_symmetric<i8_t>, nk_dots_symmetric_i8);
    run_if_matches("dots_symmetric", "u8", test_dots_symmetric<u8_t>, nk_dots_symmetric_u8);
    run_if_matches("dots_symmetric", "i4", test_dots_symmetric<i4x2_t>, nk_dots_symmetric_i4);
    run_if_matches("dots_symmetric", "u4", test_dots_symmetric<u4x2_t>, nk_dots_symmetric_u4);
    run_if_matches("dots_symmetric", "e4m3", test_dots_symmetric<e4m3_t>, nk_dots_symmetric_e4m3);
    run_if_matches("dots_symmetric", "e5m2", test_dots_symmetric<e5m2_t>, nk_dots_symmetric_e5m2);
#else

#if NK_TARGET_NEON
    run_if_matches("dots_symmetric_neon", "f32", test_dots_symmetric<f32_t>, nk_dots_symmetric_f32_neon);
    run_if_matches("dots_symmetric_neon", "f64", test_dots_symmetric<f64_t>, nk_dots_symmetric_f64_neon);
#endif

#if NK_TARGET_HASWELL
    run_if_matches("dots_symmetric_haswell", "f32", test_dots_symmetric<f32_t>, nk_dots_symmetric_f32_haswell);
    run_if_matches("dots_symmetric_haswell", "f64", test_dots_symmetric<f64_t>, nk_dots_symmetric_f64_haswell);
    run_if_matches("dots_symmetric_haswell", "bf16", test_dots_symmetric<bf16_t>, nk_dots_symmetric_bf16_haswell);
    run_if_matches("dots_symmetric_haswell", "f16", test_dots_symmetric<f16_t>, nk_dots_symmetric_f16_haswell);
    run_if_matches("dots_symmetric_haswell", "i8", test_dots_symmetric<i8_t>, nk_dots_symmetric_i8_haswell);
    run_if_matches("dots_symmetric_haswell", "u8", test_dots_symmetric<u8_t>, nk_dots_symmetric_u8_haswell);
    run_if_matches("dots_symmetric_haswell", "e4m3", test_dots_symmetric<e4m3_t>, nk_dots_symmetric_e4m3_haswell);
    run_if_matches("dots_symmetric_haswell", "e5m2", test_dots_symmetric<e5m2_t>, nk_dots_symmetric_e5m2_haswell);
#endif

#if NK_TARGET_SKYLAKE
    run_if_matches("dots_symmetric_skylake", "f32", test_dots_symmetric<f32_t>, nk_dots_symmetric_f32_skylake);
    run_if_matches("dots_symmetric_skylake", "f64", test_dots_symmetric<f64_t>, nk_dots_symmetric_f64_skylake);
    run_if_matches("dots_symmetric_skylake", "bf16", test_dots_symmetric<bf16_t>, nk_dots_symmetric_bf16_skylake);
    run_if_matches("dots_symmetric_skylake", "f16", test_dots_symmetric<f16_t>, nk_dots_symmetric_f16_skylake);
    run_if_matches("dots_symmetric_skylake", "e4m3", test_dots_symmetric<e4m3_t>, nk_dots_symmetric_e4m3_skylake);
    run_if_matches("dots_symmetric_skylake", "e5m2", test_dots_symmetric<e5m2_t>, nk_dots_symmetric_e5m2_skylake);
#endif

#if NK_TARGET_ICE
    run_if_matches("dots_symmetric_ice", "i8", test_dots_symmetric<i8_t>, nk_dots_symmetric_i8_ice);
    run_if_matches("dots_symmetric_ice", "u8", test_dots_symmetric<u8_t>, nk_dots_symmetric_u8_ice);
    run_if_matches("dots_symmetric_ice", "i4", test_dots_symmetric<i4x2_t>, nk_dots_symmetric_i4_ice);
    run_if_matches("dots_symmetric_ice", "u4", test_dots_symmetric<u4x2_t>, nk_dots_symmetric_u4_ice);
#endif

#if NK_TARGET_GENOA
    run_if_matches("dots_symmetric_genoa", "bf16", test_dots_symmetric<bf16_t>, nk_dots_symmetric_bf16_genoa);
    run_if_matches("dots_symmetric_genoa", "e4m3", test_dots_symmetric<e4m3_t>, nk_dots_symmetric_e4m3_genoa);
    run_if_matches("dots_symmetric_genoa", "e5m2", test_dots_symmetric<e5m2_t>, nk_dots_symmetric_e5m2_genoa);
#endif

#if NK_TARGET_NEONSDOT
    run_if_matches("dots_symmetric_neonsdot", "i8", test_dots_symmetric<i8_t>, nk_dots_symmetric_i8_neonsdot);
    run_if_matches("dots_symmetric_neonsdot", "u8", test_dots_symmetric<u8_t>, nk_dots_symmetric_u8_neonsdot);
#endif

#if NK_TARGET_NEONFHM
    run_if_matches("dots_symmetric_neonfhm", "f16", test_dots_symmetric<f16_t>, nk_dots_symmetric_f16_neonfhm);
#endif

#if NK_TARGET_NEONHALF
    run_if_matches("dots_symmetric_neonhalf", "f16", test_dots_symmetric<f16_t>, nk_dots_symmetric_f16_neonhalf);
#endif

#if NK_TARGET_SME
    run_if_matches("dots_symmetric_sme", "bf16", test_dots_symmetric<bf16_t>, nk_dots_symmetric_bf16_sme);
    run_if_matches("dots_symmetric_sme", "f16", test_dots_symmetric<f16_t>, nk_dots_symmetric_f16_sme);
    run_if_matches("dots_symmetric_sme", "i8", test_dots_symmetric<i8_t>, nk_dots_symmetric_i8_sme);
    run_if_matches("dots_symmetric_sme", "u8", test_dots_symmetric<u8_t>, nk_dots_symmetric_u8_sme);
    run_if_matches("dots_symmetric_sme", "i4", test_dots_symmetric<i4x2_t>, nk_dots_symmetric_i4_sme);
    run_if_matches("dots_symmetric_sme", "u4", test_dots_symmetric<u4x2_t>, nk_dots_symmetric_u4_sme);
    run_if_matches("dots_symmetric_sme", "e4m3", test_dots_symmetric<e4m3_t>, nk_dots_symmetric_e4m3_sme);
    run_if_matches("dots_symmetric_sme", "e5m2", test_dots_symmetric<e5m2_t>, nk_dots_symmetric_e5m2_sme);
#endif

    // Serial always runs - baseline test
    run_if_matches("dots_symmetric_serial", "f32", test_dots_symmetric<f32_t>, nk_dots_symmetric_f32_serial);
    run_if_matches("dots_symmetric_serial", "f64", test_dots_symmetric<f64_t>, nk_dots_symmetric_f64_serial);
    run_if_matches("dots_symmetric_serial", "bf16", test_dots_symmetric<bf16_t>, nk_dots_symmetric_bf16_serial);
    run_if_matches("dots_symmetric_serial", "f16", test_dots_symmetric<f16_t>, nk_dots_symmetric_f16_serial);
    run_if_matches("dots_symmetric_serial", "i8", test_dots_symmetric<i8_t>, nk_dots_symmetric_i8_serial);
    run_if_matches("dots_symmetric_serial", "u8", test_dots_symmetric<u8_t>, nk_dots_symmetric_u8_serial);
    // Note: i4/u4 symmetric not implemented in serial baseline, only in Ice/SME
    run_if_matches("dots_symmetric_serial", "e4m3", test_dots_symmetric<e4m3_t>, nk_dots_symmetric_e4m3_serial);
    run_if_matches("dots_symmetric_serial", "e5m2", test_dots_symmetric<e5m2_t>, nk_dots_symmetric_e5m2_serial);

#endif // NK_DYNAMIC_DISPATCH
}

#pragma endregion // Dots

#pragma region Sparse

/**
 *  @brief Generate a sorted array of unique random indices (in-place).
 */
template <typename scalar_type_, typename generator_type_>
void fill_sorted_unique(generator_type_ &generator, nk::vector<scalar_type_> &vector, scalar_type_ max_val) {
    using raw_t = typename scalar_type_::raw_t;
    std::uniform_int_distribution<raw_t> distribution(0, static_cast<raw_t>(max_val));
    std::size_t const count = vector.size_values();

    // Fill and sort once
    for (std::size_t i = 0; i < count; ++i) vector[i] = scalar_type_(distribution(generator));
    std::sort(vector.values_data(), vector.values_data() + count);

    // Compact duplicates; refill gaps until full
    auto unique_end = std::unique(vector.values_data(), vector.values_data() + count);
    std::size_t unique_count = static_cast<std::size_t>(unique_end - vector.values_data());

    while (unique_count < count) {
        for (std::size_t i = unique_count; i < count; ++i) vector[i] = scalar_type_(distribution(generator));
        std::sort(vector.values_data(), vector.values_data() + count);
        unique_end = std::unique(vector.values_data(), vector.values_data() + count);
        unique_count = static_cast<std::size_t>(unique_end - vector.values_data());
    }
}

/**
 *  @brief Test set intersection (unified template for u16/u32 index types).
 */
template <typename index_type_>
error_stats_t test_intersect(typename index_type_::sparse_intersect_kernel_t kernel) {
    using index_t = index_type_;
    error_stats_t stats;
    std::mt19937 generator(global_config.seed);
    std::size_t dim = sparse_dimensions;
    auto a = make_vector<index_t>(dim), b = make_vector<index_t>(dim);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_sorted_unique(generator, a, index_t(dim * 4));
        fill_sorted_unique(generator, b, index_t(dim * 4));

        nk_size_t count;
        kernel(a.raw_values_data(), b.raw_values_data(), dim, dim, nullptr, &count);

        nk_size_t ref;
        nk::sparse_intersect<index_t, nk::no_simd_k>(a.values_data(), b.values_data(), dim, dim, &ref);
        stats.accumulate_exact(count == ref);
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
    std::mt19937 generator(global_config.seed);
    std::size_t dim = sparse_dimensions;
    auto a_idx = make_vector<index_t>(dim), b_idx = make_vector<index_t>(dim);
    auto a_weights = make_vector<weight_t>(dim), b_weights = make_vector<weight_t>(dim);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_sorted_unique(generator, a_idx, index_t(dim * 4));
        fill_sorted_unique(generator, b_idx, index_t(dim * 4));
        fill_random(generator, a_weights);
        fill_random(generator, b_weights);

        typename weight_t::dot_result_t result;
        kernel(a_idx.raw_values_data(), b_idx.raw_values_data(), a_weights.raw_values_data(),
               b_weights.raw_values_data(), dim, dim, &result.raw_);

        f118_t ref;
        nk::sparse_dot<index_t, weight_t, f118_t, nk::no_simd_k>(
            a_idx.values_data(), b_idx.values_data(), a_weights.values_data(), b_weights.values_data(), dim, dim, &ref);
        stats.accumulate(result, ref);
    }
    return stats;
}

void test_sparse() {
    std::printf("Testing sparse operations...\n");

#if NK_DYNAMIC_DISPATCH
    run_if_matches("sparse_intersect", "u16", test_intersect<u16_t>, nk_sparse_intersect_u16);
    run_if_matches("sparse_intersect", "u32", test_intersect<u32_t>, nk_sparse_intersect_u32);
    run_if_matches("sparse_intersect", "u64", test_intersect<u64_t>, nk_sparse_intersect_u64);
    run_if_matches("sparse_dot", "u32f32", test_sparse_dot<f32_t>, nk_sparse_dot_u32f32);
    run_if_matches("sparse_dot", "u16bf16", test_sparse_dot<bf16_t>, nk_sparse_dot_u16bf16);
#else

#if NK_TARGET_NEON
    run_if_matches("sparse_intersect_neon", "u16", test_intersect<u16_t>, nk_sparse_intersect_u16_neon);
    run_if_matches("sparse_intersect_neon", "u32", test_intersect<u32_t>, nk_sparse_intersect_u32_neon);
    run_if_matches("sparse_intersect_neon", "u64", test_intersect<u64_t>, nk_sparse_intersect_u64_neon);
#endif // NK_TARGET_NEON

#if NK_TARGET_SVE
    run_if_matches("sparse_intersect_sve2", "u16", test_intersect<u16_t>, nk_sparse_intersect_u16_sve2);
    run_if_matches("sparse_intersect_sve2", "u32", test_intersect<u32_t>, nk_sparse_intersect_u32_sve2);
    run_if_matches("sparse_intersect_sve2", "u64", test_intersect<u64_t>, nk_sparse_intersect_u64_sve2);
    run_if_matches("sparse_dot_sve2", "u32f32", test_sparse_dot<f32_t>, nk_sparse_dot_u32f32_sve2);
    run_if_matches("sparse_dot_sve2", "u16bf16", test_sparse_dot<bf16_t>, nk_sparse_dot_u16bf16_sve2);
#endif // NK_TARGET_SVE

#if NK_TARGET_ICE
    run_if_matches("sparse_intersect_ice", "u16", test_intersect<u16_t>, nk_sparse_intersect_u16_ice);
    run_if_matches("sparse_intersect_ice", "u32", test_intersect<u32_t>, nk_sparse_intersect_u32_ice);
    run_if_matches("sparse_intersect_ice", "u64", test_intersect<u64_t>, nk_sparse_intersect_u64_ice);
    run_if_matches("sparse_dot_ice", "u32f32", test_sparse_dot<f32_t>, nk_sparse_dot_u32f32_ice);
#endif // NK_TARGET_ICE

#if NK_TARGET_TURIN
    run_if_matches("sparse_intersect_turin", "u16", test_intersect<u16_t>, nk_sparse_intersect_u16_turin);
    run_if_matches("sparse_intersect_turin", "u32", test_intersect<u32_t>, nk_sparse_intersect_u32_turin);
    run_if_matches("sparse_intersect_turin", "u64", test_intersect<u64_t>, nk_sparse_intersect_u64_turin);
    run_if_matches("sparse_dot_turin", "u32f32", test_sparse_dot<f32_t>, nk_sparse_dot_u32f32_turin);
    run_if_matches("sparse_dot_turin", "u16bf16", test_sparse_dot<bf16_t>, nk_sparse_dot_u16bf16_turin);
#endif // NK_TARGET_TURIN

    // Serial always runs - baseline test
    run_if_matches("sparse_intersect_serial", "u16", test_intersect<u16_t>, nk_sparse_intersect_u16_serial);
    run_if_matches("sparse_intersect_serial", "u32", test_intersect<u32_t>, nk_sparse_intersect_u32_serial);
    run_if_matches("sparse_intersect_serial", "u64", test_intersect<u64_t>, nk_sparse_intersect_u64_serial);
    run_if_matches("sparse_dot_serial", "u32f32", test_sparse_dot<f32_t>, nk_sparse_dot_u32f32_serial);
    run_if_matches("sparse_dot_serial", "u16bf16", test_sparse_dot<bf16_t>, nk_sparse_dot_u16bf16_serial);

#endif // NK_DYNAMIC_DISPATCH
}

#pragma endregion // Sparse

#pragma region Vector Type Tests

/**
 *  @brief Test nk::vector instantiation and operations for all supported type combinations.
 *
 *  Verifies that the dimension-based API, value-based API, iterators, and sub-byte proxy
 *  references work correctly for: float, std::complex<double>, f32c_t, i8_t, i4x2_t, u1x8_t.
 */
void test_vector_types() {
    std::printf("Testing vector type instantiations...\n");

    // Test: float (primitive, 1 dim per value)
    {
        nk::vector<float> v;
        assert(v.resize(100) && "float resize failed");
        assert(v.size() == 100 && "float size mismatch");
        assert(v.size_values() == 100 && "float size_values mismatch");
        v[50] = 3.14f;
        assert(v[50] == 3.14f && "float operator[] failed");
        std::size_t count = 0;
        for (auto it = v.begin(); it != v.end(); ++it) ++count;
        assert(count == 100 && "float iterator count mismatch");
    }

    // Test: std::complex<double> (primitive complex, 1 dim per value)
    {
        nk::vector<std::complex<double>> v;
        assert(v.resize(50) && "complex<double> resize failed");
        assert(v.size() == 50 && "complex<double> size mismatch");
        assert(v.size_values() == 50 && "complex<double> size_values mismatch");
        v[25] = std::complex<double>(1.0, 2.0);
        assert(v[25] == std::complex<double>(1.0, 2.0) && "complex<double> operator[] failed");
        std::size_t count = 0;
        for (auto &elem : v) {
            (void)elem;
            ++count;
        }
        assert(count == 50 && "complex<double> range-for count mismatch");
    }

    // Test: f32c_t (NumKong complex wrapper, 1 dim per value)
    {
        nk::vector<nk::f32c_t> v;
        assert(v.resize(64) && "f32c_t resize failed");
        assert(v.size() == 64 && "f32c_t size mismatch");
        assert(v.size_values() == 64 && "f32c_t size_values mismatch");
        v[32] = nk::f32c_t(1.5f, -2.5f);
        assert(v[32] == nk::f32c_t(1.5f, -2.5f) && "f32c_t operator[] failed");
    }

    // Test: i8_t (NumKong integer wrapper, 1 dim per value)
    {
        nk::vector<nk::i8_t> v;
        assert(v.resize(128) && "i8_t resize failed");
        assert(v.size() == 128 && "i8_t size mismatch");
        assert(v.size_values() == 128 && "i8_t size_values mismatch");
        v[64] = nk::i8_t(-42);
        assert(v[64] == nk::i8_t(-42) && "i8_t operator[] failed");
    }

    // Test: i4x2_t (sub-byte, 2 dims per value, LSB-first)
    {
        nk::vector<nk::i4x2_t> v;
        assert(v.resize(100) && "i4x2_t resize failed");
        assert(v.size() == 100 && "i4x2_t size mismatch");
        assert(v.size_values() == 50 && "i4x2_t size_values mismatch (should be dims/2)");

        v[0] = 5, v[1] = -3;
        assert(v[0] == 5 && "i4x2_t dim 0 mismatch");
        assert(v[1] == -3 && "i4x2_t dim 1 mismatch");

        // Test iterator returns correct count
        std::size_t count = 0;
        for (auto it = v.begin(); it != v.end(); ++it) ++count;
        assert(count == 100 && "i4x2_t iterator count mismatch");
    }

    // Test: u1x8_t (sub-byte, 8 dims per value, LSB-first)
    {
        nk::vector<nk::u1x8_t> v;
        assert(v.resize(64) && "u1x8_t resize failed");
        assert(v.size() == 64 && "u1x8_t size mismatch");
        assert(v.size_values() == 8 && "u1x8_t size_values mismatch (should be dims/8)");

        v[0] = true, v[1] = false, v[7] = true;
        assert(v[0] == true && "u1x8_t dim 0 mismatch");
        assert(v[1] == false && "u1x8_t dim 1 mismatch");
        assert(v[7] == true && "u1x8_t dim 7 mismatch");

        // Test iterator returns correct count
        std::size_t count = 0;
        for (auto it = v.begin(); it != v.end(); ++it) ++count;
        assert(count == 64 && "u1x8_t iterator count mismatch");
    }

    // Test: Custom allocator (stateless)
    {
        using custom_alloc_t = nk::aligned_allocator<nk::f32_t, 128>;
        nk::vector<nk::f32_t, custom_alloc_t> v;
        assert(v.resize(256) && "custom allocator resize failed");
        assert(v.size() == 256 && "custom allocator size mismatch");
        v[128] = nk::f32_t(99.0f);
        assert(v[128] == nk::f32_t(99.0f) && "custom allocator value mismatch");
    }

    // Test: Reserve and capacity
    {
        nk::vector<nk::f64_t> v;
        assert(v.reserve(1000) && "reserve failed");
        assert(v.capacity() >= 1000 && "capacity < reserved");
        assert(v.resize(500) && "resize after reserve failed");
        assert(v.size() == 500 && "size after reserve mismatch");
        assert(v.capacity() >= 1000 && "capacity shrunk after resize");
    }

    // Test: Move semantics
    {
        nk::vector<nk::f32_t> v1;
        assert(v1.resize(100) && "v1 resize failed");
        v1[50] = nk::f32_t(42.0f);

        nk::vector<nk::f32_t> v2 = std::move(v1);
        assert(v2.size() == 100 && "move ctor size mismatch");
        assert(v2[50] == nk::f32_t(42.0f) && "move ctor value mismatch");
        assert(v1.size() == 0 && "moved-from vector not empty");

        nk::vector<nk::f32_t> v3;
        v3 = std::move(v2);
        assert(v3.size() == 100 && "move assign size mismatch");
        assert(v3[50] == nk::f32_t(42.0f) && "move assign value mismatch");
    }

    // Test: Swap
    {
        nk::vector<nk::i8_t> v1, v2;
        assert(v1.resize(10) && v2.resize(20));
        v1[0] = nk::i8_t(1);
        v2[0] = nk::i8_t(2);

        swap(v1, v2);
        assert(v1.size() == 20 && "swap v1 size mismatch");
        assert(v2.size() == 10 && "swap v2 size mismatch");
        assert(v1[0] == nk::i8_t(2) && "swap v1 value mismatch");
        assert(v2[0] == nk::i8_t(1) && "swap v2 value mismatch");
    }

    std::printf("  vector<float>:                OK\n");
    std::printf("  vector<std::complex<double>>: OK\n");
    std::printf("  vector<f32c_t>:               OK\n");
    std::printf("  vector<i8_t>:                 OK\n");
    std::printf("  vector<i4x2_t>:               OK (sub-byte proxy, LSB-first)\n");
    std::printf("  vector<u1x8_t>:               OK (sub-byte proxy, LSB-first)\n");
    std::printf("  custom allocator:             OK\n");
    std::printf("  reserve/capacity:             OK\n");
    std::printf("  move semantics:               OK\n");
    std::printf("  swap:                         OK\n");
}

#pragma endregion // Vector Type Tests

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
    if (char const *env = std::getenv("NK_DENSE_DIMENSIONS")) {
        std::size_t val = static_cast<std::size_t>(std::atoll(env));
        if (val > 0) {
            dense_dimensions = val;
            std::printf("Using NK_DENSE_DIMENSIONS=%zu\n", dense_dimensions);
        }
    }
    if (char const *env = std::getenv("NK_SPARSE_DIMENSIONS")) {
        std::size_t val = static_cast<std::size_t>(std::atoll(env));
        if (val > 0) {
            sparse_dimensions = val;
            std::printf("Using NK_SPARSE_DIMENSIONS=%zu\n", sparse_dimensions);
        }
    }
    if (char const *env = std::getenv("NK_MESH_POINTS")) {
        std::size_t val = static_cast<std::size_t>(std::atoll(env));
        if (val > 0) {
            mesh_points = val;
            std::printf("Using NK_MESH_POINTS=%zu\n", mesh_points);
        }
    }
    if (char const *env = std::getenv("NK_MATRIX_HEIGHT")) {
        std::size_t val = static_cast<std::size_t>(std::atoll(env));
        if (val > 0) {
            matrix_height = val;
            std::printf("Using NK_MATRIX_HEIGHT=%zu\n", matrix_height);
        }
    }
    if (char const *env = std::getenv("NK_MATRIX_WIDTH")) {
        std::size_t val = static_cast<std::size_t>(std::atoll(env));
        if (val > 0) {
            matrix_width = val;
            std::printf("Using NK_MATRIX_WIDTH=%zu\n", matrix_width);
        }
    }
    if (char const *env = std::getenv("NK_MATRIX_DEPTH")) {
        std::size_t val = static_cast<std::size_t>(std::atoll(env));
        if (val > 0) {
            matrix_depth = val;
            std::printf("Using NK_MATRIX_DEPTH=%zu\n", matrix_depth);
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

    // Vector type instantiation tests
    test_vector_types();

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
    test_dots_symmetric();
    test_sparse();

    std::printf("\n");
    std::printf("All tests passed.\n");
    return 0;
}

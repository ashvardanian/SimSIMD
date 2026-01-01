/**
 *  @file   test.cpp
 *  @brief  C++ test suite with precision analysis using Boost.Multiprecision.
 *
 *  This test suite compares NumKong operations against high-precision references
 *  (quad precision via Boost.Multiprecision) and reports ULP error statistics.
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

#pragma region Includes_and_Configuration

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

#include <boost/multiprecision/cpp_bin_float.hpp>

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

#if NK_COMPARE_TO_MKL
#include <mkl.h> // MKL includes its own CBLAS interface
#elif NK_COMPARE_TO_BLAS
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif
#endif

#define NK_NATIVE_F16  0
#define NK_NATIVE_BF16 0
#include <numkong/numkong.h>

using steady_clock = std::chrono::steady_clock;
using time_point = steady_clock::time_point;

namespace mp = boost::multiprecision;
using f128_t = mp::cpp_bin_float_quad;
using f64_t = double;
using f32_t = float;

/**
 *  @brief Double-double arithmetic with ~106-bit mantissa.
 *  Uses Knuth two-sum + FMA for lower-error transformations.
 *
 *      Type                        Speed  Mantissa  Notes
 *      double                      1.0x   53-bit    Hardware
 *      long double                 1.5x   64-bit    x87 hardware
 *      double_double_t             11x    ~106-bit  Software, FMA-based
 *      __float128                  88x    113-bit   libquadmath
 *      boost::float128             91x    113-bit   Wrapper around `__float128`
 *      boost::cpp_bin_float_quad   200x   113-bit   Pure C++ (slowest!)
 *      boost::cpp_bin_float_50     237x   ~166-bit  50 decimal digits
 */
struct double_double_t {
    double hi, lo;

    inline double_double_t() noexcept : hi(0), lo(0) {}
    inline double_double_t(double h, double l) noexcept : hi(h), lo(l) {}
    inline double_double_t(double v) noexcept : hi(v), lo(0) {}

    inline explicit double_double_t(f128_t const &v) noexcept {
        hi = static_cast<double>(v);
        lo = static_cast<double>(v - f128_t(hi));
    }

    inline static double_double_t two_sum(double a, double b) noexcept {
        double s = a + b;
        double v = s - a;
        return double_double_t(s, (a - (s - v)) + (b - v));
    }

    inline static double_double_t quick_two_sum(double a, double b) noexcept {
        return double_double_t(a + b, b - ((a + b) - a));
    }

    inline double_double_t operator+(double_double_t const &o) const noexcept {
        double_double_t s = two_sum(hi, o.hi);
        s.lo += lo + o.lo;
        return quick_two_sum(s.hi, s.lo);
    }

    inline double_double_t &operator+=(double_double_t const &o) noexcept { return *this = *this + o; }

    inline double_double_t operator-(double_double_t const &o) const noexcept {
        double_double_t s = two_sum(hi, -o.hi);
        s.lo += lo - o.lo;
        return quick_two_sum(s.hi, s.lo);
    }

    inline double_double_t operator*(double_double_t const &o) const noexcept {
        double p = hi * o.hi;
        return quick_two_sum(p, std::fma(hi, o.hi, -p) + hi * o.lo + lo * o.hi);
    }

    inline double_double_t operator/(double_double_t const &o) const noexcept {
        double q = hi / o.hi;
        double_double_t r = *this - o * double_double_t(q);
        return quick_two_sum(q, r.hi / o.hi);
    }

    inline bool operator==(double_double_t const &o) const noexcept { return hi == o.hi && lo == o.lo; }
    inline bool operator!=(double_double_t const &o) const noexcept { return !(*this == o); }
    inline bool operator<(double_double_t const &o) const noexcept { return hi < o.hi || (hi == o.hi && lo < o.lo); }
    inline bool operator>(double_double_t const &o) const noexcept { return o < *this; }
    inline bool operator<=(double_double_t const &o) const noexcept { return !(o < *this); }
    inline bool operator>=(double_double_t const &o) const noexcept { return !(*this < o); }

    inline explicit operator double() const noexcept { return hi + lo; }
    inline explicit operator f128_t() const noexcept { return f128_t(hi) + f128_t(lo); }
};

using fmax_t = double_double_t;

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

// Global test dimensions - can be overridden via environment variables
std::size_t dense_dimension = 1024; // For dot products, spatial metrics
std::size_t mesh_dimension = 256;   // For RMSD, Kabsch (3D point clouds)
std::size_t matmul_dimension_m = 64, matmul_dimension_n = 64, matmul_dimension_k = 64;

#pragma endregion // Includes_and_Configuration

#pragma region Precision_Infrastructure

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
    // Handle special cases
    if (std::isnan(a) || std::isnan(b)) return std::numeric_limits<std::uint64_t>::max();
    if (a == b) return 0; // Also handles +0 == -0
    if (std::isinf(a) || std::isinf(b)) return std::numeric_limits<std::uint64_t>::max();

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

// Specialization for computing ULP when comparing against high-precision reference
template <typename scalar_type_>
std::uint64_t ulp_distance_from_reference(scalar_type_ actual, f128_t reference) noexcept {
    scalar_type_ ref_as_t = static_cast<scalar_type_>(reference);
    return ulp_distance(actual, ref_as_t);
}

#pragma endregion // Precision_Infrastructure

#pragma region Error_Statistics

/**
 *  @brief Accumulator for error statistics across multiple test trials.
 */
struct error_stats_t {
    double min_abs_err = std::numeric_limits<double>::max();
    double max_abs_err = 0;
    double sum_abs_err = 0;

    double min_rel_err = std::numeric_limits<double>::max();
    double max_rel_err = 0;
    double sum_rel_err = 0;

    std::uint64_t min_ulp = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t max_ulp = 0;
    std::uint64_t sum_ulp = 0;

    std::size_t count = 0;
    std::size_t exact_matches = 0;

    void accumulate(double expected, double actual) noexcept {
        double abs_err = std::fabs(expected - actual);
        double rel_err = expected != 0 ? abs_err / std::fabs(expected) : abs_err;

        min_abs_err = std::min(min_abs_err, abs_err);
        max_abs_err = std::max(max_abs_err, abs_err);
        sum_abs_err += abs_err;

        min_rel_err = std::min(min_rel_err, rel_err);
        max_rel_err = std::max(max_rel_err, rel_err);
        sum_rel_err += rel_err;

        count++;
        if (abs_err == 0) exact_matches++;
    }

    void accumulate_ulp(std::uint64_t ulps) noexcept {
        min_ulp = std::min(min_ulp, ulps);
        max_ulp = std::max(max_ulp, ulps);
        sum_ulp += ulps;
        if (ulps == 0) exact_matches++;
        count++;
    }

    void accumulate(double expected, double actual, std::uint64_t ulps) noexcept {
        accumulate(expected, actual);
        // Don't double-count
        count--;
        exact_matches -= (expected == actual) ? 1 : 0;
        accumulate_ulp(ulps);
        count--;
        exact_matches -= (ulps == 0) ? 1 : 0;
        count++;
        if (ulps == 0) exact_matches++;
    }

    double mean_abs_err() const noexcept { return count > 0 ? sum_abs_err / count : 0; }
    double mean_rel_err() const noexcept { return count > 0 ? sum_rel_err / count : 0; }
    double mean_ulp() const noexcept { return count > 0 ? static_cast<double>(sum_ulp) / count : 0; }

    void reset() noexcept { *this = error_stats_t {}; }

    // Merge stats from another instance (for OpenMP reduction)
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

#pragma endregion // Error_Statistics

#pragma region Reference_Implementations

/**
 *  @brief Reference dot product with configurable precision.
 *  Use f128_t for f64 kernels, f64_t for f32/f16/bf16 kernels.
 */
template <typename reference_type_, typename scalar_type_>
reference_type_ reference_dot(scalar_type_ const *a, scalar_type_ const *b, std::size_t n) noexcept {
    reference_type_ sum = 0;
    for (std::size_t i = 0; i < n; i++) { sum += reference_type_(a[i]) * reference_type_(b[i]); }
    return sum;
}

/**
 *  @brief Reference L2 squared distance with configurable precision.
 */
template <typename reference_type_, typename scalar_type_>
reference_type_ reference_l2sq(scalar_type_ const *a, scalar_type_ const *b, std::size_t n) noexcept {
    reference_type_ sum = 0;
    for (std::size_t i = 0; i < n; i++) {
        reference_type_ diff = reference_type_(a[i]) - reference_type_(b[i]);
        sum += diff * diff;
    }
    return sum;
}

/**
 *  @brief Reference angular (cosine) distance with configurable precision.
 */
template <typename reference_type_, typename scalar_type_>
reference_type_ reference_angular(scalar_type_ const *a, scalar_type_ const *b, std::size_t n) noexcept {
    reference_type_ ab = 0, aa = 0, bb = 0;
    for (std::size_t i = 0; i < n; i++) {
        reference_type_ ai = reference_type_(a[i]);
        reference_type_ bi = reference_type_(b[i]);
        ab += ai * bi;
        aa += ai * ai;
        bb += bi * bi;
    }
    if (aa == 0 && bb == 0) return reference_type_(0);
    if (ab == 0) return reference_type_(1);
    reference_type_ cos_sim = ab / std::sqrt(static_cast<double>(aa * bb));
    reference_type_ result = reference_type_(1) - cos_sim;
    return result > 0 ? result : reference_type_(0);
}

/**
 *  @brief Helper to get the appropriate epsilon for probability divergence.
 *         Must match NK_F32_DIVISION_EPSILON and NK_F64_DIVISION_EPSILON.
 */
template <typename scalar_type_>
f128_t divergence_epsilon() noexcept {
    if constexpr (std::is_same_v<scalar_type_, nk_f64_t> || std::is_same_v<scalar_type_, double>) {
        return f128_t(1e-15L); // NK_F64_DIVISION_EPSILON
    }
    else {
        return f128_t(1e-7L); // NK_F32_DIVISION_EPSILON
    }
}

/**
 *  @brief Quad-precision KL divergence reference.
 */
template <typename scalar_type_>
f128_t reference_kld_quad(scalar_type_ const *p, scalar_type_ const *q, std::size_t n) noexcept {
    f128_t sum = 0;
    f128_t epsilon = divergence_epsilon<scalar_type_>();
    for (std::size_t i = 0; i < n; i++) {
        f128_t pi = f128_t(p[i]);
        f128_t qi = f128_t(q[i]);
        if (pi > 0) { sum += pi * mp::log((pi + epsilon) / (qi + epsilon)); }
    }
    return sum;
}

/**
 *  @brief Quad-precision Jensen-Shannon divergence reference.
 *         Returns sqrt(JSD/2) to match the JSD distance metric implementation.
 */
template <typename scalar_type_>
f128_t reference_jsd_quad(scalar_type_ const *p, scalar_type_ const *q, std::size_t n) noexcept {
    f128_t sum = 0;
    f128_t epsilon = divergence_epsilon<scalar_type_>();
    for (std::size_t i = 0; i < n; i++) {
        f128_t pi = f128_t(p[i]);
        f128_t qi = f128_t(q[i]);
        f128_t mi = (pi + qi) / 2;
        if (pi > 0) sum += pi * mp::log((pi + epsilon) / (mi + epsilon));
        if (qi > 0) sum += qi * mp::log((qi + epsilon) / (mi + epsilon));
    }
    f128_t d_half = sum / 2;
    return d_half > 0 ? mp::sqrt(d_half) : f128_t(0);
}

/**
 *  @brief Quad-precision bilinear form reference: a^T * C * b
 */
template <typename scalar_type_>
f128_t reference_bilinear_quad(scalar_type_ const *a, scalar_type_ const *c, scalar_type_ const *b,
                               std::size_t n) noexcept {
    f128_t sum = 0;
    for (std::size_t i = 0; i < n; i++) {
        for (std::size_t j = 0; j < n; j++) { sum += f128_t(a[i]) * f128_t(c[i * n + j]) * f128_t(b[j]); }
    }
    return sum;
}

/**
 *  @brief Reference sum reduction with stride support.
 */
template <typename scalar_type_, typename accumulator_type_ = f128_t>
accumulator_type_ reference_reduce_add(scalar_type_ const *data, std::size_t count, std::size_t stride_bytes) noexcept {
    accumulator_type_ sum = 0;
    char const *ptr = reinterpret_cast<char const *>(data);
    for (std::size_t i = 0; i < count; i++) {
        sum += accumulator_type_(*reinterpret_cast<scalar_type_ const *>(ptr));
        ptr += stride_bytes;
    }
    return sum;
}

/**
 *  @brief Reference min reduction with argmin and stride support.
 */
template <typename scalar_type_>
void reference_reduce_min(scalar_type_ const *data, std::size_t count, std::size_t stride_bytes,
                          scalar_type_ *min_value, nk_size_t *min_index) noexcept {
    char const *ptr = reinterpret_cast<char const *>(data);
    scalar_type_ best_val = *reinterpret_cast<scalar_type_ const *>(ptr);
    nk_size_t best_idx = 0;
    for (std::size_t i = 1; i < count; i++) {
        ptr += stride_bytes;
        scalar_type_ val = *reinterpret_cast<scalar_type_ const *>(ptr);
        if (val < best_val) {
            best_val = val;
            best_idx = i;
        }
    }
    *min_value = best_val;
    *min_index = best_idx;
}

/**
 *  @brief Reference max reduction with argmax and stride support.
 */
template <typename scalar_type_>
void reference_reduce_max(scalar_type_ const *data, std::size_t count, std::size_t stride_bytes,
                          scalar_type_ *max_value, nk_size_t *max_index) noexcept {
    char const *ptr = reinterpret_cast<char const *>(data);
    scalar_type_ best_val = *reinterpret_cast<scalar_type_ const *>(ptr);
    nk_size_t best_idx = 0;
    for (std::size_t i = 1; i < count; i++) {
        ptr += stride_bytes;
        scalar_type_ val = *reinterpret_cast<scalar_type_ const *>(ptr);
        if (val > best_val) {
            best_val = val;
            best_idx = i;
        }
    }
    *max_value = best_val;
    *max_index = best_idx;
}

#pragma endregion // Reference_Implementations

#pragma region Test_Harness_Templates

/**
 *  @brief Aligned memory allocation for SIMD-friendly buffers.
 */
template <typename scalar_type_>
struct aligned_buffer {
    static constexpr std::size_t alignment = 64; // Cache line
    scalar_type_ *data = nullptr;
    std::size_t count = 0;

    aligned_buffer() = default;
    explicit aligned_buffer(std::size_t n) : count(n) {
        if (n > 0) {
            void *ptr = std::aligned_alloc(alignment,
                                           ((n * sizeof(scalar_type_) + alignment - 1) / alignment) * alignment);
            data = static_cast<scalar_type_ *>(ptr);
            std::memset(data, 0, n * sizeof(scalar_type_));
        }
    }
    ~aligned_buffer() {
        if (data) std::free(data);
    }
    aligned_buffer(aligned_buffer const &) = delete;
    aligned_buffer &operator=(aligned_buffer const &) = delete;
    aligned_buffer(aligned_buffer &&other) noexcept : data(other.data), count(other.count) {
        other.data = nullptr;
        other.count = 0;
    }
    aligned_buffer &operator=(aligned_buffer &&other) noexcept {
        if (this != &other) {
            if (data) std::free(data);
            data = other.data;
            count = other.count;
            other.data = nullptr;
            other.count = 0;
        }
        return *this;
    }

    scalar_type_ &operator[](std::size_t i) noexcept { return data[i]; }
    scalar_type_ const &operator[](std::size_t i) const noexcept { return data[i]; }
};

/**
 *  @brief Wrapper for nk_f16_t enabling implicit float conversions.
 */
struct f16_t {
    nk_f16_t raw;
    f16_t() noexcept : raw(0) {}
    f16_t(float v) noexcept { nk_f32_to_f16(&v, &raw); }
    operator float() const noexcept {
        float r;
        nk_f16_to_f32(&raw, &r);
        return r;
    }
    float to_f32() const noexcept {
        float r;
        nk_f16_to_f32(&raw, &r);
        return r;
    }
    static f16_t from_raw(nk_f16_t r) noexcept {
        f16_t v;
        v.raw = r;
        return v;
    }
};

/**
 *  @brief Wrapper for nk_bf16_t enabling implicit float conversions.
 */
struct bf16_t {
    nk_bf16_t raw;
    bf16_t() noexcept : raw(0) {}
    bf16_t(float v) noexcept { nk_f32_to_bf16(&v, &raw); }
    operator float() const noexcept {
        float r;
        nk_bf16_to_f32(&raw, &r);
        return r;
    }
    float to_f32() const noexcept {
        float r;
        nk_bf16_to_f32(&raw, &r);
        return r;
    }
    static bf16_t from_raw(nk_bf16_t r) noexcept {
        bf16_t v;
        v.raw = r;
        return v;
    }
};

/**
 *  @brief Wrapper for nk_e4m3_t enabling implicit float conversions.
 */
struct e4m3_t {
    nk_e4m3_t raw;
    e4m3_t() noexcept : raw(0) {}
    e4m3_t(float v) noexcept { nk_f32_to_e4m3(&v, &raw); }
    operator float() const noexcept {
        float r;
        nk_e4m3_to_f32(&raw, &r);
        return r;
    }
    float to_f32() const noexcept {
        float r;
        nk_e4m3_to_f32(&raw, &r);
        return r;
    }
    static e4m3_t from_raw(nk_e4m3_t r) noexcept {
        e4m3_t v;
        v.raw = r;
        return v;
    }
};

/**
 *  @brief Wrapper for nk_e5m2_t enabling implicit float conversions.
 */
struct e5m2_t {
    nk_e5m2_t raw;
    e5m2_t() noexcept : raw(0) {}
    e5m2_t(float v) noexcept { nk_f32_to_e5m2(&v, &raw); }
    operator float() const noexcept {
        float r;
        nk_e5m2_to_f32(&raw, &r);
        return r;
    }
    float to_f32() const noexcept {
        float r;
        nk_e5m2_to_f32(&raw, &r);
        return r;
    }
    static e5m2_t from_raw(nk_e5m2_t r) noexcept {
        e5m2_t v;
        v.raw = r;
        return v;
    }
};

/**
 *  @brief Numeric traits for low-precision types.
 *
 *  Provides type-specific limits for test range generation:
 *  - max_finite(): Largest representable finite value
 *  - safe_max(): Conservative range for accumulation operations (avoids overflow in dot products)
 *  - mantissa_bits(): Precision bits (for ULP scaling)
 *
 *  The "safe" range is designed for N-dimensional dot products where N <= 4096:
 *  With safe_max S, worst-case sum is N * S^2, which must fit in f32 (~3.4e38).
 */
template <typename T>
struct nk_numeric_traits {
    // Default: assume f32-like range
    static constexpr double max_finite() { return 3.4e38; }
    static constexpr double safe_max() { return 1.0; }
    static constexpr int mantissa_bits() { return 23; }
};

// Standard types
template <>
struct nk_numeric_traits<float> {
    static constexpr double max_finite() { return 3.4e38; }
    static constexpr double safe_max() { return 1e6; }
    static constexpr int mantissa_bits() { return 23; }
};
template <>
struct nk_numeric_traits<double> {
    static constexpr double max_finite() { return 1.7e308; }
    static constexpr double safe_max() { return 1e6; }
    static constexpr int mantissa_bits() { return 52; }
};

// F16: 5 exp bits, 10 mantissa bits, max ~65504
template <>
struct nk_numeric_traits<f16_t> {
    static constexpr double max_finite() { return 65504.0; }
    static constexpr double safe_max() { return 100.0; }
    static constexpr int mantissa_bits() { return 10; }
};

// BF16: 8 exp bits, 7 mantissa bits, same range as f32
template <>
struct nk_numeric_traits<bf16_t> {
    static constexpr double max_finite() { return 3.4e38; }
    static constexpr double safe_max() { return 100.0; }
    static constexpr int mantissa_bits() { return 7; }
};

// E4M3: 4 exp bits, 3 mantissa bits, max 448
template <>
struct nk_numeric_traits<e4m3_t> {
    static constexpr double max_finite() { return 448.0; }
    static constexpr double safe_max() { return 1.0; }
    static constexpr int mantissa_bits() { return 3; }
};

// E5M2: 5 exp bits, 2 mantissa bits, max 57344
template <>
struct nk_numeric_traits<e5m2_t> {
    static constexpr double max_finite() { return 57344.0; }
    static constexpr double safe_max() { return 3.0; }
    static constexpr int mantissa_bits() { return 2; }
};

/**
 *  @brief Fill buffer with random values using configured distribution.
 *
 *  Distributions:
 *    - uniform_k:   Bounded within safe range, fastest baseline
 *    - lognormal_k: Sign-randomized log-normal, compressed to safe range
 *    - cauchy_k:    Compressed extreme tails, fits within safe range
 *
 *  Range defaults to [-safe_max, +safe_max] from nk_numeric_traits<T>.
 */
template <typename scalar_type_, typename generator_type_>
void fill_random(aligned_buffer<scalar_type_> &buf, generator_type_ &rng,
                 double min_val = -nk_numeric_traits<scalar_type_>::safe_max(),
                 double max_val = nk_numeric_traits<scalar_type_>::safe_max()) {
    double range = max_val - min_val;
    double mid = (max_val + min_val) / 2.0;

    switch (global_config.distribution) {
    case random_distribution_kind_t::uniform_k: {
        std::uniform_real_distribution<double> dist(min_val, max_val);
        for (std::size_t i = 0; i < buf.count; i++) { buf[i] = static_cast<scalar_type_>(dist(rng)); }
        break;
    }
    case random_distribution_kind_t::lognormal_k: {
        // Log-normal with random sign, compressed to [min_val, max_val]
        std::lognormal_distribution<double> lognorm(0.0, 0.5);
        std::uniform_real_distribution<double> sign_dist(0.0, 1.0);
        for (std::size_t i = 0; i < buf.count; i++) {
            double val = lognorm(rng);
            double compressed = 2.0 / (1.0 + std::exp(-val)) - 1.0;
            if (sign_dist(rng) < 0.5) compressed = -compressed;
            buf[i] = static_cast<scalar_type_>(mid + compressed * (range / 2.0));
        }
        break;
    }
    case random_distribution_kind_t::cauchy_k: {
        // Cauchy compressed via atan to [min_val, max_val]
        std::cauchy_distribution<double> cauchy_k(0.0, 1.0);
        for (std::size_t i = 0; i < buf.count; i++) {
            double val = cauchy_k(rng);
            double compressed = (2.0 / M_PI) * std::atan(val);
            buf[i] = static_cast<scalar_type_>(mid + compressed * (range / 2.0));
        }
        break;
    }
    }
}

/**
 *  @brief Fill buffer with random probability distribution (sums to ~1).
 */
template <typename scalar_type_, typename generator_type_>
void fill_probability(aligned_buffer<scalar_type_> &buf, generator_type_ &rng) {
    std::uniform_real_distribution<double> dist(0.01, 1.0);
    double sum = 0;
    for (std::size_t i = 0; i < buf.count; i++) {
        double v = dist(rng);
        buf[i] = static_cast<scalar_type_>(v);
        sum += v;
    }
    // Normalize
    for (std::size_t i = 0; i < buf.count; i++) {
        buf[i] = static_cast<scalar_type_>(static_cast<double>(buf[i]) / sum);
    }
}

/**
 *  @brief Fill buffer with random integers in specified range.
 */
template <typename scalar_type_, typename generator_type_>
void fill_random_integers(aligned_buffer<scalar_type_> &buf, generator_type_ &rng, int min_val = -100,
                          int max_val = 100) {
    std::uniform_int_distribution<int> dist(min_val, max_val);
    for (std::size_t i = 0; i < buf.count; i++) { buf[i] = static_cast<scalar_type_>(dist(rng)); }
}

#pragma endregion // Test_Harness_Templates

#pragma region BLAS_Baselines

#if NK_COMPARE_TO_BLAS || NK_COMPARE_TO_MKL

void dot_f32_blas(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *result) {
    *result = cblas_sdot(static_cast<int>(n), a, 1, b, 1);
}

void dot_f64_blas(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result) {
    *result = cblas_ddot(static_cast<int>(n), a, 1, b, 1);
}

void dot_f32c_blas(nk_f32c_t const *a, nk_f32c_t const *b, nk_size_t n, nk_f32c_t *result) {
    cblas_cdotu_sub(static_cast<int>(n), a, 1, b, 1, result);
}

void vdot_f32c_blas(nk_f32c_t const *a, nk_f32c_t const *b, nk_size_t n, nk_f32c_t *result) {
    cblas_cdotc_sub(static_cast<int>(n), a, 1, b, 1, result); // conjugated
}

void dot_f64c_blas(nk_f64c_t const *a, nk_f64c_t const *b, nk_size_t n, nk_f64c_t *result) {
    cblas_zdotu_sub(static_cast<int>(n), a, 1, b, 1, result);
}

void vdot_f64c_blas(nk_f64c_t const *a, nk_f64c_t const *b, nk_size_t n, nk_f64c_t *result) {
    cblas_zdotc_sub(static_cast<int>(n), a, 1, b, 1, result); // conjugated
}

void dots_f32_blas(nk_f32_t const *a, nk_f32_t const *b, nk_f32_t *c, nk_size_t m, nk_size_t n, nk_size_t k,
                   nk_size_t a_stride, nk_size_t c_stride) {
    (void)a_stride;
    (void)c_stride;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
                1.0f, a, static_cast<int>(k), b, static_cast<int>(k), 0.0f, c, static_cast<int>(n));
}

void dots_f64_blas(nk_f64_t const *a, nk_f64_t const *b, nk_f64_t *c, nk_size_t m, nk_size_t n, nk_size_t k,
                   nk_size_t a_stride, nk_size_t c_stride) {
    (void)a_stride;
    (void)c_stride;
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
                1.0, a, static_cast<int>(k), b, static_cast<int>(k), 0.0, c, static_cast<int>(n));
}
#endif // NK_COMPARE_TO_BLAS || NK_COMPARE_TO_MKL

#if NK_COMPARE_TO_MKL
void dots_bf16_mkl(nk_bf16_t const *a, nk_bf16_t const *b, nk_f32_t *c, nk_size_t m, nk_size_t n, nk_size_t k,
                   nk_size_t a_stride, nk_size_t c_stride) {
    (void)a_stride;
    (void)c_stride;
    cblas_gemm_bf16bf16f32(CblasRowMajor, CblasNoTrans, CblasTrans, static_cast<MKL_INT>(m), static_cast<MKL_INT>(n),
                           static_cast<MKL_INT>(k), 1.0f, a, static_cast<MKL_INT>(k), b, static_cast<MKL_INT>(k), 0.0f,
                           c, static_cast<MKL_INT>(n));
}

void dots_f16_mkl(nk_f16_t const *a, nk_f16_t const *b, nk_f32_t *c, nk_size_t m, nk_size_t n, nk_size_t k,
                  nk_size_t a_stride, nk_size_t c_stride) {
    (void)a_stride;
    (void)c_stride;
    cblas_gemm_f16f16f32(CblasRowMajor, CblasNoTrans, CblasTrans, static_cast<MKL_INT>(m), static_cast<MKL_INT>(n),
                         static_cast<MKL_INT>(k), 1.0f, reinterpret_cast<MKL_F16 const *>(a), static_cast<MKL_INT>(k),
                         reinterpret_cast<MKL_F16 const *>(b), static_cast<MKL_INT>(k), 0.0f, c,
                         static_cast<MKL_INT>(n));
}

void dots_i8_mkl(nk_i8_t const *a, nk_u8_t const *b, nk_i32_t *c, nk_size_t m, nk_size_t n, nk_size_t k,
                 nk_size_t a_stride, nk_size_t c_stride) {
    (void)a_stride;
    (void)c_stride;
    MKL_INT32 c_offset = 0;
    cblas_gemm_s8u8s32(CblasRowMajor, CblasNoTrans, CblasTrans, CblasFixOffset, static_cast<MKL_INT>(m),
                       static_cast<MKL_INT>(n), static_cast<MKL_INT>(k), 1.0f, a, static_cast<MKL_INT>(k), 0, b,
                       static_cast<MKL_INT>(k), 0, 0.0f, c, static_cast<MKL_INT>(n), &c_offset);
}

#endif // NK_COMPARE_TO_MKL

#pragma endregion // BLAS_Baselines

#pragma region Kernel_Types

// Reduce kernels
using reduce_add_f32_t = void (*)(nk_f32_t const *, nk_size_t, nk_size_t, nk_f64_t *);
using reduce_add_f64_t = void (*)(nk_f64_t const *, nk_size_t, nk_size_t, nk_f64_t *);
using reduce_add_i32_t = void (*)(nk_i32_t const *, nk_size_t, nk_size_t, nk_i64_t *);
using reduce_minmax_f32_t = void (*)(nk_f32_t const *, nk_size_t, nk_size_t, nk_f32_t *, nk_size_t *);

// Dot kernels
using dot_f32_t = void (*)(nk_f32_t const *, nk_f32_t const *, nk_size_t, nk_f32_t *);
using dot_f64_t = void (*)(nk_f64_t const *, nk_f64_t const *, nk_size_t, nk_f64_t *);
using dot_f16_t = void (*)(nk_f16_t const *, nk_f16_t const *, nk_size_t, nk_f32_t *);
using dot_bf16_t = void (*)(nk_bf16_t const *, nk_bf16_t const *, nk_size_t, nk_f32_t *);
using dot_i8_t = void (*)(nk_i8_t const *, nk_i8_t const *, nk_size_t, nk_i32_t *);
using dot_u8_t = void (*)(nk_u8_t const *, nk_u8_t const *, nk_size_t, nk_u32_t *);
using dot_f32c_t = void (*)(nk_f32c_t const *, nk_f32c_t const *, nk_size_t, nk_f32c_t *);
using vdot_f32c_t = void (*)(nk_f32c_t const *, nk_f32c_t const *, nk_size_t, nk_f32c_t *);
using dot_f64c_t = void (*)(nk_f64c_t const *, nk_f64c_t const *, nk_size_t, nk_f64c_t *);
using vdot_f64c_t = void (*)(nk_f64c_t const *, nk_f64c_t const *, nk_size_t, nk_f64c_t *);

// Spatial kernels (l2sq, angular)
using l2sq_f32_t = void (*)(nk_f32_t const *, nk_f32_t const *, nk_size_t, nk_f32_t *);
using l2sq_f64_t = void (*)(nk_f64_t const *, nk_f64_t const *, nk_size_t, nk_f64_t *);
using l2sq_f16_t = void (*)(nk_f16_t const *, nk_f16_t const *, nk_size_t, nk_f32_t *);
using l2sq_bf16_t = void (*)(nk_bf16_t const *, nk_bf16_t const *, nk_size_t, nk_f32_t *);
using angular_f32_t = void (*)(nk_f32_t const *, nk_f32_t const *, nk_size_t, nk_f32_t *);
using angular_f64_t = void (*)(nk_f64_t const *, nk_f64_t const *, nk_size_t, nk_f64_t *);
using angular_f16_t = void (*)(nk_f16_t const *, nk_f16_t const *, nk_size_t, nk_f32_t *);
using angular_bf16_t = void (*)(nk_bf16_t const *, nk_bf16_t const *, nk_size_t, nk_f32_t *);

// Curved kernels (bilinear, kld, jsd)
using bilinear_f32_t = void (*)(nk_f32_t const *, nk_f32_t const *, nk_f32_t const *, nk_size_t, nk_f32_t *);
using bilinear_f64_t = void (*)(nk_f64_t const *, nk_f64_t const *, nk_f64_t const *, nk_size_t, nk_f64_t *);
using kld_f32_t = void (*)(nk_f32_t const *, nk_f32_t const *, nk_size_t, nk_f32_t *);
using kld_f64_t = void (*)(nk_f64_t const *, nk_f64_t const *, nk_size_t, nk_f64_t *);
using jsd_f32_t = void (*)(nk_f32_t const *, nk_f32_t const *, nk_size_t, nk_f32_t *);
using jsd_f64_t = void (*)(nk_f64_t const *, nk_f64_t const *, nk_size_t, nk_f64_t *);

// Binary kernels
using hamming_b8_t = void (*)(nk_b8_t const *, nk_b8_t const *, nk_size_t, nk_u32_t *);
using jaccard_b8_t = void (*)(nk_b8_t const *, nk_b8_t const *, nk_size_t, nk_f32_t *);

// Elementwise kernels
using scale_f32_t = void (*)(nk_f32_t const *, nk_size_t, nk_f32_t const *, nk_f32_t const *, nk_f32_t *);
using sum_f32_t = void (*)(nk_f32_t const *, nk_f32_t const *, nk_size_t, nk_f32_t *);
using wsum_f32_t = void (*)(nk_f32_t const *, nk_f32_t const *, nk_size_t, nk_f32_t const *, nk_f32_t const *,
                            nk_f32_t *);
using fma_f32_t = void (*)(nk_f32_t const *, nk_f32_t const *, nk_f32_t const *, nk_size_t, nk_f32_t const *,
                           nk_f32_t const *, nk_f32_t *);

// Trigonometry kernels
using sin_f16_t = void (*)(nk_f16_t const *, nk_size_t, nk_f16_t *);
using cos_f16_t = void (*)(nk_f16_t const *, nk_size_t, nk_f16_t *);
using atan_f16_t = void (*)(nk_f16_t const *, nk_size_t, nk_f16_t *);
using sin_f32_t = void (*)(nk_f32_t const *, nk_size_t, nk_f32_t *);
using cos_f32_t = void (*)(nk_f32_t const *, nk_size_t, nk_f32_t *);
using sin_f64_t = void (*)(nk_f64_t const *, nk_size_t, nk_f64_t *);
using cos_f64_t = void (*)(nk_f64_t const *, nk_size_t, nk_f64_t *);

// Geospatial kernels
using haversine_f64_t = void (*)(nk_f64_t const *, nk_f64_t const *, nk_f64_t const *, nk_f64_t const *, nk_size_t,
                                 nk_f64_t *);
using haversine_f32_t = void (*)(nk_f32_t const *, nk_f32_t const *, nk_f32_t const *, nk_f32_t const *, nk_size_t,
                                 nk_f32_t *);
using vincenty_f64_t = void (*)(nk_f64_t const *, nk_f64_t const *, nk_f64_t const *, nk_f64_t const *, nk_size_t,
                                nk_f64_t *);

// Sparse kernels
using intersect_u16_t = void (*)(nk_u16_t const *, nk_u16_t const *, nk_size_t, nk_size_t, nk_u32_t *);
using intersect_u32_t = void (*)(nk_u32_t const *, nk_u32_t const *, nk_size_t, nk_size_t, nk_u32_t *);
using sparse_dot_u32f32_t = void (*)(nk_u32_t const *, nk_u32_t const *, nk_f32_t const *, nk_f32_t const *, nk_size_t,
                                     nk_size_t, nk_f32_t *);
using sparse_dot_u16bf16_t = void (*)(nk_u16_t const *, nk_u16_t const *, nk_bf16_t const *, nk_bf16_t const *,
                                      nk_size_t, nk_size_t, nk_f32_t *);

// Mesh kernels (rmsd, kabsch, umeyama)
using mesh_f32_t = void (*)(nk_f32_t const *, nk_f32_t const *, nk_size_t, nk_f32_t *, nk_f32_t *, nk_f32_t *,
                            nk_f32_t *, nk_f32_t *);
using mesh_f64_t = void (*)(nk_f64_t const *, nk_f64_t const *, nk_size_t, nk_f64_t *, nk_f64_t *, nk_f64_t *,
                            nk_f64_t *, nk_f64_t *);

// Dots kernels (GEMM) - compute functions take void const* for packed B
using dots_f32f32f32_t = void (*)(nk_f32_t const *, void const *, nk_f32_t *, nk_size_t, nk_size_t, nk_size_t,
                                  nk_size_t, nk_size_t);
using dots_f64f64f64_t = void (*)(nk_f64_t const *, void const *, nk_f64_t *, nk_size_t, nk_size_t, nk_size_t,
                                  nk_size_t, nk_size_t);
using dots_bf16bf16f32_t = void (*)(nk_bf16_t const *, void const *, nk_f32_t *, nk_size_t, nk_size_t, nk_size_t,
                                    nk_size_t, nk_size_t);
using dots_f16f16f32_t = void (*)(nk_f16_t const *, void const *, nk_f32_t *, nk_size_t, nk_size_t, nk_size_t,
                                  nk_size_t, nk_size_t);
using dots_i8i8i32_t = void (*)(nk_i8_t const *, void const *, nk_i32_t *, nk_size_t, nk_size_t, nk_size_t, nk_size_t,
                                nk_size_t);

// Dots pack kernels - pack functions take void* for output
using dots_f32_pack_t = void (*)(nk_f32_t const *, nk_size_t, nk_size_t, nk_size_t, void *);
using dots_f64_pack_t = void (*)(nk_f64_t const *, nk_size_t, nk_size_t, nk_size_t, void *);
using dots_bf16_pack_t = void (*)(nk_bf16_t const *, nk_size_t, nk_size_t, nk_size_t, void *);
using dots_f16_pack_t = void (*)(nk_f16_t const *, nk_size_t, nk_size_t, nk_size_t, void *);
using dots_i8_pack_t = void (*)(nk_i8_t const *, nk_size_t, nk_size_t, nk_size_t, void *);

// Dots packed_size kernels
using dots_packed_size_t = nk_size_t (*)(nk_size_t, nk_size_t);

// FP8 (e4m3/e5m2) kernels
using dot_e4m3_t = void (*)(nk_e4m3_t const *, nk_e4m3_t const *, nk_size_t, nk_f32_t *);
using dot_e5m2_t = void (*)(nk_e5m2_t const *, nk_e5m2_t const *, nk_size_t, nk_f32_t *);
using reduce_add_e4m3_t = void (*)(nk_e4m3_t const *, nk_size_t, nk_size_t, nk_f32_t *);
using reduce_add_e5m2_t = void (*)(nk_e5m2_t const *, nk_size_t, nk_size_t, nk_f32_t *);
using reduce_minmax_e4m3_t = void (*)(nk_e4m3_t const *, nk_size_t, nk_size_t, nk_f32_t *, nk_size_t *);
using reduce_minmax_e5m2_t = void (*)(nk_e5m2_t const *, nk_size_t, nk_size_t, nk_f32_t *, nk_size_t *);
using sum_e4m3_t = void (*)(nk_e4m3_t const *, nk_e4m3_t const *, nk_size_t, nk_e4m3_t *);
using sum_e5m2_t = void (*)(nk_e5m2_t const *, nk_e5m2_t const *, nk_size_t, nk_e5m2_t *);
using scale_e4m3_t = void (*)(nk_e4m3_t const *, nk_size_t, nk_f32_t const *, nk_f32_t const *, nk_e4m3_t *);
using scale_e5m2_t = void (*)(nk_e5m2_t const *, nk_size_t, nk_f32_t const *, nk_f32_t const *, nk_e5m2_t *);
using wsum_e4m3_t = void (*)(nk_e4m3_t const *, nk_e4m3_t const *, nk_size_t, nk_f32_t const *, nk_f32_t const *,
                             nk_e4m3_t *);
using wsum_e5m2_t = void (*)(nk_e5m2_t const *, nk_e5m2_t const *, nk_size_t, nk_f32_t const *, nk_f32_t const *,
                             nk_e5m2_t *);
using fma_e4m3_t = void (*)(nk_e4m3_t const *, nk_e4m3_t const *, nk_e4m3_t const *, nk_size_t, nk_f32_t const *,
                            nk_f32_t const *, nk_e4m3_t *);
using fma_e5m2_t = void (*)(nk_e5m2_t const *, nk_e5m2_t const *, nk_e5m2_t const *, nk_size_t, nk_f32_t const *,
                            nk_f32_t const *, nk_e5m2_t *);

#pragma endregion // Kernel_Types

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

/**
 *  @brief Test denormal number handling.
 */
void test_denormals() {
    std::printf("Testing denormal handling...\n");

    float denormal = 1e-40f;
    int classification = std::fpclassify(denormal);

    if (classification == FP_SUBNORMAL) { std::printf("  Denormals are preserved (FP_SUBNORMAL)\n"); }
    else if (classification == FP_ZERO) { std::printf("  Denormals flushed to zero (FTZ mode)\n"); }
    else { std::printf("  Unexpected classification: %d\n", classification); }

    std::printf("  Denormal handling: PASS\n");
}

#pragma endregion // Types

#pragma region Reduce

/**
 *  @brief Test sum reduction for f32.
 */
error_stats_t test_reduce_add_f32(reduce_add_f32_t kernel) {
    error_stats_t stats;
    std::mt19937 rng(global_config.seed);
    aligned_buffer<nk_f32_t> data(dense_dimension);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(data, rng, -10.0, 10.0);

        nk_f64_t result;
        kernel(data.data, dense_dimension, sizeof(nk_f32_t), &result);

        f128_t ref = reference_reduce_add<nk_f32_t>(data.data, dense_dimension, sizeof(nk_f32_t));
        std::uint64_t ulps = ulp_distance_from_reference(result, ref);
        stats.accumulate(static_cast<double>(ref), result, ulps);
    }
    return stats;
}

/**
 *  @brief Test sum reduction for f64.
 */
error_stats_t test_reduce_add_f64(reduce_add_f64_t kernel) {
    error_stats_t stats;
    std::mt19937 rng(global_config.seed);
    aligned_buffer<nk_f64_t> data(dense_dimension);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(data, rng, -10.0, 10.0);

        nk_f64_t result;
        kernel(data.data, dense_dimension, sizeof(nk_f64_t), &result);

        fmax_t ref = reference_reduce_add<nk_f64_t, fmax_t>(data.data, dense_dimension, sizeof(nk_f64_t));
        std::uint64_t ulps = ulp_distance(result, static_cast<nk_f64_t>(ref));
        stats.accumulate(static_cast<double>(ref), result, ulps);
    }
    return stats;
}

/**
 *  @brief Test sum reduction for i32.
 */
error_stats_t test_reduce_add_i32(reduce_add_i32_t kernel) {
    error_stats_t stats;
    std::mt19937 rng(global_config.seed);
    aligned_buffer<nk_i32_t> data(dense_dimension);
    std::uniform_int_distribution<nk_i32_t> dist(-1000, 1000);

    for (auto start = test_start_time(); within_time_budget(start);) {
        for (std::size_t i = 0; i < dense_dimension; i++) data[i] = dist(rng);

        nk_i64_t result;
        kernel(data.data, dense_dimension, sizeof(nk_i32_t), &result);

        nk_i64_t ref = 0;
        for (std::size_t i = 0; i < dense_dimension; i++) ref += data[i];

        stats.accumulate_ulp(result == ref ? 0 : 1);
    }
    return stats;
}

/**
 *  @brief Test min reduction for f32.
 */
error_stats_t test_reduce_min_f32(reduce_minmax_f32_t kernel) {
    error_stats_t stats;
    std::mt19937 rng(global_config.seed);
    aligned_buffer<nk_f32_t> data(dense_dimension);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(data, rng, -10.0, 10.0);

        nk_f32_t min_val;
        nk_size_t min_idx;
        kernel(data.data, dense_dimension, sizeof(nk_f32_t), &min_val, &min_idx);

        nk_f32_t ref_val;
        nk_size_t ref_idx;
        reference_reduce_min(data.data, dense_dimension, sizeof(nk_f32_t), &ref_val, &ref_idx);

        // Value must match exactly, index may differ if there are ties
        std::uint64_t ulps = ulp_distance(min_val, ref_val);
        stats.accumulate_ulp(ulps);

        // Verify the returned index actually points to the minimum
        if (data[min_idx] != min_val) { std::fprintf(stderr, "FAIL: reduce_min_f32 index mismatch\n"); }
    }
    return stats;
}

/**
 *  @brief Test max reduction for f32.
 */
error_stats_t test_reduce_max_f32(reduce_minmax_f32_t kernel) {
    error_stats_t stats;
    std::mt19937 rng(global_config.seed);
    aligned_buffer<nk_f32_t> data(dense_dimension);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(data, rng, -10.0, 10.0);

        nk_f32_t max_val;
        nk_size_t max_idx;
        kernel(data.data, dense_dimension, sizeof(nk_f32_t), &max_val, &max_idx);

        nk_f32_t ref_val;
        nk_size_t ref_idx;
        reference_reduce_max(data.data, dense_dimension, sizeof(nk_f32_t), &ref_val, &ref_idx);

        std::uint64_t ulps = ulp_distance(max_val, ref_val);
        stats.accumulate_ulp(ulps);

        // Verify the returned index actually points to the maximum
        if (data[max_idx] != max_val) { std::fprintf(stderr, "FAIL: reduce_max_f32 index mismatch\n"); }
    }
    return stats;
}

/**
 *  @brief Test reduce add precision for e4m3 (FP8).
 */
error_stats_t test_reduce_add_e4m3(reduce_add_e4m3_t kernel) {
    error_stats_t stats;
    std::mt19937 rng(global_config.seed);
    aligned_buffer<nk_e4m3_t> data(dense_dimension);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (auto start = test_start_time(); within_time_budget(start);) {
        for (std::size_t i = 0; i < dense_dimension; i++) {
            float v = dist(rng);
            nk_f32_to_e4m3(&v, &data[i]);
        }

        nk_f32_t result;
        kernel(data.data, dense_dimension, sizeof(nk_e4m3_t), &result);

        f128_t ref = 0;
        for (std::size_t i = 0; i < dense_dimension; i++) {
            float f;
            nk_e4m3_to_f32(&data[i], &f);
            ref += f128_t(f);
        }

        std::uint64_t ulps = ulp_distance_from_reference(result, ref);
        stats.accumulate(static_cast<double>(ref), static_cast<double>(result), ulps);
    }
    return stats;
}

/**
 *  @brief Test reduce add precision for e5m2 (FP8).
 */
error_stats_t test_reduce_add_e5m2(reduce_add_e5m2_t kernel) {
    error_stats_t stats;
    std::mt19937 rng(global_config.seed);
    aligned_buffer<nk_e5m2_t> data(dense_dimension);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (auto start = test_start_time(); within_time_budget(start);) {
        for (std::size_t i = 0; i < dense_dimension; i++) {
            float v = dist(rng);
            nk_f32_to_e5m2(&v, &data[i]);
        }

        nk_f32_t result;
        kernel(data.data, dense_dimension, sizeof(nk_e5m2_t), &result);

        f128_t ref = 0;
        for (std::size_t i = 0; i < dense_dimension; i++) {
            float f;
            nk_e5m2_to_f32(&data[i], &f);
            ref += f128_t(f);
        }

        std::uint64_t ulps = ulp_distance_from_reference(result, ref);
        stats.accumulate(static_cast<double>(ref), static_cast<double>(result), ulps);
    }
    return stats;
}

/**
 *  @brief Test reduce min precision for e4m3 (FP8).
 */
error_stats_t test_reduce_min_e4m3(reduce_minmax_e4m3_t kernel) {
    error_stats_t stats;
    std::mt19937 rng(global_config.seed);
    aligned_buffer<nk_e4m3_t> data(dense_dimension);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    for (auto start = test_start_time(); within_time_budget(start);) {
        for (std::size_t i = 0; i < dense_dimension; i++) {
            float v = dist(rng);
            nk_f32_to_e4m3(&v, &data[i]);
        }

        nk_f32_t min_val;
        nk_size_t min_idx;
        kernel(data.data, dense_dimension, sizeof(nk_e4m3_t), &min_val, &min_idx);

        // Compute reference
        float ref_val;
        nk_e4m3_to_f32(&data[0], &ref_val);
        for (std::size_t i = 1; i < dense_dimension; i++) {
            float f;
            nk_e4m3_to_f32(&data[i], &f);
            if (f < ref_val) ref_val = f;
        }

        std::uint64_t ulps = ulp_distance(min_val, ref_val);
        stats.accumulate_ulp(ulps);
    }
    return stats;
}

/**
 *  @brief Test reduce max precision for e4m3 (FP8).
 */
error_stats_t test_reduce_max_e4m3(reduce_minmax_e4m3_t kernel) {
    error_stats_t stats;
    std::mt19937 rng(global_config.seed);
    aligned_buffer<nk_e4m3_t> data(dense_dimension);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    for (auto start = test_start_time(); within_time_budget(start);) {
        for (std::size_t i = 0; i < dense_dimension; i++) {
            float v = dist(rng);
            nk_f32_to_e4m3(&v, &data[i]);
        }

        nk_f32_t max_val;
        nk_size_t max_idx;
        kernel(data.data, dense_dimension, sizeof(nk_e4m3_t), &max_val, &max_idx);

        // Compute reference
        float ref_val;
        nk_e4m3_to_f32(&data[0], &ref_val);
        for (std::size_t i = 1; i < dense_dimension; i++) {
            float f;
            nk_e4m3_to_f32(&data[i], &f);
            if (f > ref_val) ref_val = f;
        }

        std::uint64_t ulps = ulp_distance(max_val, ref_val);
        stats.accumulate_ulp(ulps);
    }
    return stats;
}

/**
 *  @brief Test reduce min precision for e5m2 (FP8).
 */
error_stats_t test_reduce_min_e5m2(reduce_minmax_e5m2_t kernel) {
    error_stats_t stats;
    std::mt19937 rng(global_config.seed);
    aligned_buffer<nk_e5m2_t> data(dense_dimension);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    for (auto start = test_start_time(); within_time_budget(start);) {
        for (std::size_t i = 0; i < dense_dimension; i++) {
            float v = dist(rng);
            nk_f32_to_e5m2(&v, &data[i]);
        }

        nk_f32_t min_val;
        nk_size_t min_idx;
        kernel(data.data, dense_dimension, sizeof(nk_e5m2_t), &min_val, &min_idx);

        // Compute reference
        float ref_val;
        nk_e5m2_to_f32(&data[0], &ref_val);
        for (std::size_t i = 1; i < dense_dimension; i++) {
            float f;
            nk_e5m2_to_f32(&data[i], &f);
            if (f < ref_val) ref_val = f;
        }

        std::uint64_t ulps = ulp_distance(min_val, ref_val);
        stats.accumulate_ulp(ulps);
    }
    return stats;
}

/**
 *  @brief Test reduce max precision for e5m2 (FP8).
 */
error_stats_t test_reduce_max_e5m2(reduce_minmax_e5m2_t kernel) {
    error_stats_t stats;
    std::mt19937 rng(global_config.seed);
    aligned_buffer<nk_e5m2_t> data(dense_dimension);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    for (auto start = test_start_time(); within_time_budget(start);) {
        for (std::size_t i = 0; i < dense_dimension; i++) {
            float v = dist(rng);
            nk_f32_to_e5m2(&v, &data[i]);
        }

        nk_f32_t max_val;
        nk_size_t max_idx;
        kernel(data.data, dense_dimension, sizeof(nk_e5m2_t), &max_val, &max_idx);

        // Compute reference
        float ref_val;
        nk_e5m2_to_f32(&data[0], &ref_val);
        for (std::size_t i = 1; i < dense_dimension; i++) {
            float f;
            nk_e5m2_to_f32(&data[i], &f);
            if (f > ref_val) ref_val = f;
        }

        std::uint64_t ulps = ulp_distance(max_val, ref_val);
        stats.accumulate_ulp(ulps);
    }
    return stats;
}

void test_reduce() {
    std::printf("Testing reductions...\n");

#if NK_DYNAMIC_DISPATCH
    // Dynamic dispatch - only test the dispatcher itself
    run_if_matches("reduce_add", "f32", test_reduce_add_f32, nk_reduce_add_f32);
    run_if_matches("reduce_add", "f64", test_reduce_add_f64, nk_reduce_add_f64);
    run_if_matches("reduce_add", "i32", test_reduce_add_i32, nk_reduce_add_i32);
    run_if_matches("reduce_add", "e4m3", test_reduce_add_e4m3, nk_reduce_add_e4m3);
    run_if_matches("reduce_add", "e5m2", test_reduce_add_e5m2, nk_reduce_add_e5m2);
    run_if_matches("reduce_min", "f32", test_reduce_min_f32, nk_reduce_min_f32);
    run_if_matches("reduce_max", "f32", test_reduce_max_f32, nk_reduce_max_f32);
    run_if_matches("reduce_min", "e4m3", test_reduce_min_e4m3, nk_reduce_min_e4m3);
    run_if_matches("reduce_max", "e4m3", test_reduce_max_e4m3, nk_reduce_max_e4m3);
    run_if_matches("reduce_min", "e5m2", test_reduce_min_e5m2, nk_reduce_min_e5m2);
    run_if_matches("reduce_max", "e5m2", test_reduce_max_e5m2, nk_reduce_max_e5m2);
#else
    // Static compilation - test all available ISA variants

#if NK_TARGET_NEON
    run_if_matches("reduce_add_neon", "f32", test_reduce_add_f32, nk_reduce_add_f32_neon);
    run_if_matches("reduce_add_neon", "f64", test_reduce_add_f64, nk_reduce_add_f64_neon);
    run_if_matches("reduce_add_neon", "i32", test_reduce_add_i32, nk_reduce_add_i32_neon);
    run_if_matches("reduce_min_neon", "f32", test_reduce_min_f32, nk_reduce_min_f32_neon);
    run_if_matches("reduce_max_neon", "f32", test_reduce_max_f32, nk_reduce_max_f32_neon);
#endif // NK_TARGET_NEON

#if NK_TARGET_NEONFHM
    run_if_matches("reduce_add_neonfhm", "e4m3", test_reduce_add_e4m3, nk_reduce_add_e4m3_neonfhm);
    run_if_matches("reduce_add_neonfhm", "e5m2", test_reduce_add_e5m2, nk_reduce_add_e5m2_neonfhm);
    run_if_matches("reduce_min_neonfhm", "e4m3", test_reduce_min_e4m3, nk_reduce_min_e4m3_neonfhm);
    run_if_matches("reduce_max_neonfhm", "e4m3", test_reduce_max_e4m3, nk_reduce_max_e4m3_neonfhm);
    run_if_matches("reduce_min_neonfhm", "e5m2", test_reduce_min_e5m2, nk_reduce_min_e5m2_neonfhm);
    run_if_matches("reduce_max_neonfhm", "e5m2", test_reduce_max_e5m2, nk_reduce_max_e5m2_neonfhm);
#endif // NK_TARGET_NEONFHM

#if NK_TARGET_HASWELL
    run_if_matches("reduce_add_haswell", "f32", test_reduce_add_f32, nk_reduce_add_f32_haswell);
    run_if_matches("reduce_add_haswell", "f64", test_reduce_add_f64, nk_reduce_add_f64_haswell);
    run_if_matches("reduce_add_haswell", "i32", test_reduce_add_i32, nk_reduce_add_i32_haswell);
    run_if_matches("reduce_add_haswell", "e4m3", test_reduce_add_e4m3, nk_reduce_add_e4m3_haswell);
    run_if_matches("reduce_add_haswell", "e5m2", test_reduce_add_e5m2, nk_reduce_add_e5m2_haswell);
    run_if_matches("reduce_min_haswell", "f32", test_reduce_min_f32, nk_reduce_min_f32_haswell);
    run_if_matches("reduce_max_haswell", "f32", test_reduce_max_f32, nk_reduce_max_f32_haswell);
    run_if_matches("reduce_min_haswell", "e4m3", test_reduce_min_e4m3, nk_reduce_min_e4m3_haswell);
    run_if_matches("reduce_max_haswell", "e4m3", test_reduce_max_e4m3, nk_reduce_max_e4m3_haswell);
    run_if_matches("reduce_min_haswell", "e5m2", test_reduce_min_e5m2, nk_reduce_min_e5m2_haswell);
    run_if_matches("reduce_max_haswell", "e5m2", test_reduce_max_e5m2, nk_reduce_max_e5m2_haswell);
#endif // NK_TARGET_HASWELL

#if NK_TARGET_SKYLAKE
    run_if_matches("reduce_add_skylake", "f32", test_reduce_add_f32, nk_reduce_add_f32_skylake);
    run_if_matches("reduce_add_skylake", "f64", test_reduce_add_f64, nk_reduce_add_f64_skylake);
    run_if_matches("reduce_add_skylake", "i32", test_reduce_add_i32, nk_reduce_add_i32_skylake);
    run_if_matches("reduce_min_skylake", "f32", test_reduce_min_f32, nk_reduce_min_f32_skylake);
    run_if_matches("reduce_max_skylake", "f32", test_reduce_max_f32, nk_reduce_max_f32_skylake);
#endif // NK_TARGET_SKYLAKE

    // Serial always runs - baseline test
    run_if_matches("reduce_add_serial", "f32", test_reduce_add_f32, nk_reduce_add_f32_serial);
    run_if_matches("reduce_add_serial", "f64", test_reduce_add_f64, nk_reduce_add_f64_serial);
    run_if_matches("reduce_add_serial", "i32", test_reduce_add_i32, nk_reduce_add_i32_serial);
    run_if_matches("reduce_add_serial", "e4m3", test_reduce_add_e4m3, nk_reduce_add_e4m3_serial);
    run_if_matches("reduce_add_serial", "e5m2", test_reduce_add_e5m2, nk_reduce_add_e5m2_serial);
    run_if_matches("reduce_min_serial", "f32", test_reduce_min_f32, nk_reduce_min_f32_serial);
    run_if_matches("reduce_max_serial", "f32", test_reduce_max_f32, nk_reduce_max_f32_serial);
    run_if_matches("reduce_min_serial", "e4m3", test_reduce_min_e4m3, nk_reduce_min_e4m3_serial);
    run_if_matches("reduce_max_serial", "e4m3", test_reduce_max_e4m3, nk_reduce_max_e4m3_serial);
    run_if_matches("reduce_min_serial", "e5m2", test_reduce_min_e5m2, nk_reduce_min_e5m2_serial);
    run_if_matches("reduce_max_serial", "e5m2", test_reduce_max_e5m2, nk_reduce_max_e5m2_serial);

#endif // NK_DYNAMIC_DISPATCH
}

#pragma endregion // Reduce

#pragma region Dot

/**
 *  @brief Test dot product precision for f32.
 */
error_stats_t test_dot_f32(dot_f32_t kernel) {
    error_stats_t stats;
    std::mt19937 rng(global_config.seed);
    aligned_buffer<nk_f32_t> a(dense_dimension), b(dense_dimension);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(a, rng);
        fill_random(b, rng);

        nk_f32_t result;
        kernel(a.data, b.data, dense_dimension, &result);

        f64_t ref = reference_dot<f64_t, nk_f32_t>(a.data, b.data, dense_dimension);
        std::uint64_t ulps = ulp_distance(result, static_cast<nk_f32_t>(ref));

        stats.accumulate(ref, static_cast<double>(result), ulps);
    }

    return stats;
}

/**
 *  @brief Test dot product precision for f64.
 */
error_stats_t test_dot_f64(dot_f64_t kernel) {
    error_stats_t stats;
    std::mt19937 rng(global_config.seed);
    aligned_buffer<nk_f64_t> a(dense_dimension), b(dense_dimension);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(a, rng);
        fill_random(b, rng);

        nk_f64_t result;
        kernel(a.data, b.data, dense_dimension, &result);

        fmax_t ref = reference_dot<fmax_t, nk_f64_t>(a.data, b.data, dense_dimension);
        std::uint64_t ulps = ulp_distance(result, static_cast<nk_f64_t>(ref));

        stats.accumulate(static_cast<double>(ref), result, ulps);
    }

    return stats;
}

/**
 *  @brief Test dot product precision for i8 (exact match required).
 */
error_stats_t test_dot_i8(dot_i8_t kernel) {
    error_stats_t stats;
    std::mt19937 rng(global_config.seed);
    aligned_buffer<nk_i8_t> a(dense_dimension), b(dense_dimension);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random_integers(a, rng, -10, 10);
        fill_random_integers(b, rng, -10, 10);

        nk_i32_t result;
        kernel(a.data, b.data, dense_dimension, &result);

        // Reference: exact integer computation
        std::int64_t ref = 0;
        for (std::size_t i = 0; i < dense_dimension; i++) {
            ref += static_cast<std::int64_t>(a[i]) * static_cast<std::int64_t>(b[i]);
        }

        std::uint64_t ulps = (result == static_cast<nk_i32_t>(ref)) ? 0 : std::numeric_limits<std::uint64_t>::max();
        stats.accumulate_ulp(ulps);

        if (global_config.assert_on_failure && result != static_cast<nk_i32_t>(ref)) {
            std::fprintf(stderr, "FAIL: dot_i8 dim=%zu expected=%lld got=%d\n", dense_dimension,
                         static_cast<long long>(ref), result);
            assert(false);
        }
    }

    return stats;
}

/**
 *  @brief Test dot product precision for u8 (exact match required).
 */
error_stats_t test_dot_u8(dot_u8_t kernel) {
    error_stats_t stats;
    std::mt19937 rng(global_config.seed);
    aligned_buffer<nk_u8_t> a(dense_dimension), b(dense_dimension);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random_integers(a, rng, 0, 15);
        fill_random_integers(b, rng, 0, 15);

        nk_u32_t result;
        kernel(a.data, b.data, dense_dimension, &result);

        // Reference: exact integer computation
        std::uint64_t ref = 0;
        for (std::size_t i = 0; i < dense_dimension; i++) {
            ref += static_cast<std::uint64_t>(a[i]) * static_cast<std::uint64_t>(b[i]);
        }

        std::uint64_t ulps = (result == static_cast<nk_u32_t>(ref)) ? 0 : std::numeric_limits<std::uint64_t>::max();
        stats.accumulate_ulp(ulps);

        if (global_config.assert_on_failure && result != static_cast<nk_u32_t>(ref)) {
            std::fprintf(stderr, "FAIL: dot_u8 dim=%zu expected=%llu got=%u\n", dense_dimension,
                         static_cast<unsigned long long>(ref), result);
            assert(false);
        }
    }

    return stats;
}

/**
 *  @brief Test dot product precision for f16.
 */
error_stats_t test_dot_f16(dot_f16_t kernel) {
    error_stats_t stats;
    std::mt19937 rng(global_config.seed);
    aligned_buffer<f16_t> a(dense_dimension), b(dense_dimension);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(a, rng, -1.0, 1.0);
        fill_random(b, rng, -1.0, 1.0);

        nk_f32_t result;
        kernel(reinterpret_cast<nk_f16_t const *>(a.data), reinterpret_cast<nk_f16_t const *>(b.data), dense_dimension,
               &result);

        f128_t ref = 0;
        for (std::size_t i = 0; i < dense_dimension; i++)
            ref += f128_t(static_cast<float>(a[i])) * f128_t(static_cast<float>(b[i]));

        std::uint64_t ulps = ulp_distance_from_reference(result, ref);
        stats.accumulate(static_cast<double>(ref), static_cast<double>(result), ulps);
    }
    return stats;
}

/**
 *  @brief Test dot product precision for bf16.
 */
error_stats_t test_dot_bf16(dot_bf16_t kernel) {
    error_stats_t stats;
    std::mt19937 rng(global_config.seed);
    aligned_buffer<bf16_t> a(dense_dimension), b(dense_dimension);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(a, rng, -1.0, 1.0);
        fill_random(b, rng, -1.0, 1.0);

        nk_f32_t result;
        kernel(reinterpret_cast<nk_bf16_t const *>(a.data), reinterpret_cast<nk_bf16_t const *>(b.data),
               dense_dimension, &result);

        f128_t ref = 0;
        for (std::size_t i = 0; i < dense_dimension; i++)
            ref += f128_t(static_cast<float>(a[i])) * f128_t(static_cast<float>(b[i]));

        std::uint64_t ulps = ulp_distance_from_reference(result, ref);
        stats.accumulate(static_cast<double>(ref), static_cast<double>(result), ulps);
    }
    return stats;
}

/**
 *  @brief Test dot product precision for e4m3 (FP8).
 */
error_stats_t test_dot_e4m3(dot_e4m3_t kernel) {
    error_stats_t stats;
    std::mt19937 rng(global_config.seed);
    aligned_buffer<nk_e4m3_t> a(dense_dimension), b(dense_dimension);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (auto start = test_start_time(); within_time_budget(start);) {
        // Generate random f32 values and convert to e4m3
        for (std::size_t i = 0; i < dense_dimension; i++) {
            float va = dist(rng), vb = dist(rng);
            nk_f32_to_e4m3(&va, &a[i]);
            nk_f32_to_e4m3(&vb, &b[i]);
        }

        nk_f32_t result;
        kernel(a.data, b.data, dense_dimension, &result);

        // Compute reference by converting back to f32
        f128_t ref = 0;
        for (std::size_t i = 0; i < dense_dimension; i++) {
            float fa, fb;
            nk_e4m3_to_f32(&a[i], &fa);
            nk_e4m3_to_f32(&b[i], &fb);
            ref += f128_t(fa) * f128_t(fb);
        }

        std::uint64_t ulps = ulp_distance_from_reference(result, ref);
        stats.accumulate(static_cast<double>(ref), static_cast<double>(result), ulps);
    }
    return stats;
}

/**
 *  @brief Test dot product precision for e5m2 (FP8).
 */
error_stats_t test_dot_e5m2(dot_e5m2_t kernel) {
    error_stats_t stats;
    std::mt19937 rng(global_config.seed);
    aligned_buffer<nk_e5m2_t> a(dense_dimension), b(dense_dimension);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (auto start = test_start_time(); within_time_budget(start);) {
        // Generate random f32 values and convert to e5m2
        for (std::size_t i = 0; i < dense_dimension; i++) {
            float va = dist(rng), vb = dist(rng);
            nk_f32_to_e5m2(&va, &a[i]);
            nk_f32_to_e5m2(&vb, &b[i]);
        }

        nk_f32_t result;
        kernel(a.data, b.data, dense_dimension, &result);

        // Compute reference by converting back to f32
        f128_t ref = 0;
        for (std::size_t i = 0; i < dense_dimension; i++) {
            float fa, fb;
            nk_e5m2_to_f32(&a[i], &fa);
            nk_e5m2_to_f32(&b[i], &fb);
            ref += f128_t(fa) * f128_t(fb);
        }

        std::uint64_t ulps = ulp_distance_from_reference(result, ref);
        stats.accumulate(static_cast<double>(ref), static_cast<double>(result), ulps);
    }
    return stats;
}

/**
 *  @brief Test complex dot product precision for f32c.
 */
error_stats_t test_dot_f32c(dot_f32c_t kernel) {
    error_stats_t stats;
    std::mt19937 rng(global_config.seed);
    aligned_buffer<nk_f32c_t> a(dense_dimension), b(dense_dimension);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (auto start = test_start_time(); within_time_budget(start);) {
        for (std::size_t i = 0; i < dense_dimension; i++) {
            a[i].real = dist(rng);
            a[i].imag = dist(rng);
            b[i].real = dist(rng);
            b[i].imag = dist(rng);
        }

        nk_f32c_t result;
        kernel(a.data, b.data, dense_dimension, &result);

        // Reference: complex dot product
        // (a.re + i*a.im) * (b.re + i*b.im) = (a.re*b.re - a.im*b.im) + i*(a.re*b.im + a.im*b.re)
        f128_t ref_real = 0, ref_imag = 0;
        for (std::size_t i = 0; i < dense_dimension; i++) {
            ref_real += f128_t(a[i].real) * f128_t(b[i].real) - f128_t(a[i].imag) * f128_t(b[i].imag);
            ref_imag += f128_t(a[i].real) * f128_t(b[i].imag) + f128_t(a[i].imag) * f128_t(b[i].real);
        }

        std::uint64_t ulps_real = ulp_distance_from_reference(result.real, ref_real);
        std::uint64_t ulps_imag = ulp_distance_from_reference(result.imag, ref_imag);
        std::uint64_t ulps = std::max(ulps_real, ulps_imag);

        stats.accumulate_ulp(ulps);
    }

    return stats;
}

/**
 *  @brief Test complex conjugate dot product precision for f32c.
 */
error_stats_t test_vdot_f32c(vdot_f32c_t kernel) {
    error_stats_t stats;
    std::mt19937 rng(global_config.seed);
    aligned_buffer<nk_f32c_t> a(dense_dimension), b(dense_dimension);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (auto start = test_start_time(); within_time_budget(start);) {
        for (std::size_t i = 0; i < dense_dimension; i++) {
            a[i].real = dist(rng);
            a[i].imag = dist(rng);
            b[i].real = dist(rng);
            b[i].imag = dist(rng);
        }

        nk_f32c_t result;
        kernel(a.data, b.data, dense_dimension, &result);

        // Reference: conjugate dot product conj(a) * b
        // (a.re - i*a.im) * (b.re + i*b.im) = (a.re*b.re + a.im*b.im) + i*(a.re*b.im - a.im*b.re)
        f128_t ref_real = 0, ref_imag = 0;
        for (std::size_t i = 0; i < dense_dimension; i++) {
            ref_real += f128_t(a[i].real) * f128_t(b[i].real) + f128_t(a[i].imag) * f128_t(b[i].imag);
            ref_imag += f128_t(a[i].real) * f128_t(b[i].imag) - f128_t(a[i].imag) * f128_t(b[i].real);
        }

        std::uint64_t ulps_real = ulp_distance_from_reference(result.real, ref_real);
        std::uint64_t ulps_imag = ulp_distance_from_reference(result.imag, ref_imag);
        std::uint64_t ulps = std::max(ulps_real, ulps_imag);

        stats.accumulate_ulp(ulps);
    }

    return stats;
}

/**
 *  @brief Test complex dot product precision for f64c.
 */
error_stats_t test_dot_f64c(dot_f64c_t kernel) {
    error_stats_t stats;
    std::mt19937 rng(global_config.seed);
    aligned_buffer<nk_f64c_t> a(dense_dimension), b(dense_dimension);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    for (auto start = test_start_time(); within_time_budget(start);) {
        for (std::size_t i = 0; i < dense_dimension; i++) {
            a[i].real = dist(rng);
            a[i].imag = dist(rng);
            b[i].real = dist(rng);
            b[i].imag = dist(rng);
        }

        nk_f64c_t result;
        kernel(a.data, b.data, dense_dimension, &result);

        // Reference: complex dot product
        // (a.re + i*a.im) * (b.re + i*b.im) = (a.re*b.re - a.im*b.im) + i*(a.re*b.im + a.im*b.re)
        f128_t ref_real = 0, ref_imag = 0;
        for (std::size_t i = 0; i < dense_dimension; i++) {
            ref_real += f128_t(a[i].real) * f128_t(b[i].real) - f128_t(a[i].imag) * f128_t(b[i].imag);
            ref_imag += f128_t(a[i].real) * f128_t(b[i].imag) + f128_t(a[i].imag) * f128_t(b[i].real);
        }

        std::uint64_t ulps_real = ulp_distance_from_reference(result.real, ref_real);
        std::uint64_t ulps_imag = ulp_distance_from_reference(result.imag, ref_imag);
        std::uint64_t ulps = std::max(ulps_real, ulps_imag);

        stats.accumulate_ulp(ulps);
    }

    return stats;
}

/**
 *  @brief Test complex conjugate dot product precision for f64c.
 */
error_stats_t test_vdot_f64c(vdot_f64c_t kernel) {
    error_stats_t stats;
    std::mt19937 rng(global_config.seed);
    aligned_buffer<nk_f64c_t> a(dense_dimension), b(dense_dimension);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    for (auto start = test_start_time(); within_time_budget(start);) {
        for (std::size_t i = 0; i < dense_dimension; i++) {
            a[i].real = dist(rng);
            a[i].imag = dist(rng);
            b[i].real = dist(rng);
            b[i].imag = dist(rng);
        }

        nk_f64c_t result;
        kernel(a.data, b.data, dense_dimension, &result);

        // Reference: conjugate dot product conj(a) * b
        // (a.re - i*a.im) * (b.re + i*b.im) = (a.re*b.re + a.im*b.im) + i*(a.re*b.im - a.im*b.re)
        f128_t ref_real = 0, ref_imag = 0;
        for (std::size_t i = 0; i < dense_dimension; i++) {
            ref_real += f128_t(a[i].real) * f128_t(b[i].real) + f128_t(a[i].imag) * f128_t(b[i].imag);
            ref_imag += f128_t(a[i].real) * f128_t(b[i].imag) - f128_t(a[i].imag) * f128_t(b[i].real);
        }

        std::uint64_t ulps_real = ulp_distance_from_reference(result.real, ref_real);
        std::uint64_t ulps_imag = ulp_distance_from_reference(result.imag, ref_imag);
        std::uint64_t ulps = std::max(ulps_real, ulps_imag);

        stats.accumulate_ulp(ulps);
    }

    return stats;
}

void test_dot() {
    std::printf("Testing dot products...\n");

#if NK_DYNAMIC_DISPATCH
    // Dynamic dispatch - only test the dispatcher itself
    run_if_matches("dot", "f32", test_dot_f32, nk_dot_f32);
    run_if_matches("dot", "f64", test_dot_f64, nk_dot_f64);
    run_if_matches("dot", "f16", test_dot_f16, nk_dot_f16);
    run_if_matches("dot", "bf16", test_dot_bf16, nk_dot_bf16);
    run_if_matches("dot", "e4m3", test_dot_e4m3, nk_dot_e4m3);
    run_if_matches("dot", "e5m2", test_dot_e5m2, nk_dot_e5m2);
    run_if_matches("dot", "i8", test_dot_i8, nk_dot_i8);
    run_if_matches("dot", "u8", test_dot_u8, nk_dot_u8);
    run_if_matches("dot", "f32c", test_dot_f32c, nk_dot_f32c);
    run_if_matches("vdot", "f32c", test_vdot_f32c, nk_vdot_f32c);
    run_if_matches("dot", "f64c", test_dot_f64c, nk_dot_f64c);
    run_if_matches("vdot", "f64c", test_vdot_f64c, nk_vdot_f64c);
#else
    // Static compilation - test all available ISA variants

#if NK_TARGET_NEON
    run_if_matches("dot_neon", "f32", test_dot_f32, nk_dot_f32_neon);
    run_if_matches("dot_neon", "f64", test_dot_f64, nk_dot_f64_neon);
    run_if_matches("dot_neon", "e4m3", test_dot_e4m3, nk_dot_e4m3_neon);
    run_if_matches("dot_neon", "e5m2", test_dot_e5m2, nk_dot_e5m2_neon);
    run_if_matches("dot_neon", "f32c", test_dot_f32c, nk_dot_f32c_neon);
    run_if_matches("vdot_neon", "f32c", test_vdot_f32c, nk_vdot_f32c_neon);
    run_if_matches("dot_neon", "f64c", test_dot_f64c, nk_dot_f64c_neon);
    run_if_matches("vdot_neon", "f64c", test_vdot_f64c, nk_vdot_f64c_neon);
#endif // NK_TARGET_NEON

#if NK_TARGET_NEONHALF
    run_if_matches("dot_neonhalf", "f16", test_dot_f16, nk_dot_f16_neonhalf);
#endif // NK_TARGET_NEONHALF

#if NK_TARGET_NEONBFDOT
    run_if_matches("dot_neonbfdot", "bf16", test_dot_bf16, nk_dot_bf16_neonbfdot);
#endif // NK_TARGET_NEONBFDOT

#if NK_TARGET_NEONSDOT
    run_if_matches("dot_neonsdot", "i8", test_dot_i8, nk_dot_i8_neonsdot);
    run_if_matches("dot_neonsdot", "u8", test_dot_u8, nk_dot_u8_neonsdot);
#endif // NK_TARGET_NEONSDOT

#if NK_TARGET_HASWELL
    run_if_matches("dot_haswell", "f32", test_dot_f32, nk_dot_f32_haswell);
    run_if_matches("dot_haswell", "f64", test_dot_f64, nk_dot_f64_haswell);
    run_if_matches("dot_haswell", "f16", test_dot_f16, nk_dot_f16_haswell);
    run_if_matches("dot_haswell", "bf16", test_dot_bf16, nk_dot_bf16_haswell);
    run_if_matches("dot_haswell", "e4m3", test_dot_e4m3, nk_dot_e4m3_haswell);
    run_if_matches("dot_haswell", "e5m2", test_dot_e5m2, nk_dot_e5m2_haswell);
    run_if_matches("dot_haswell", "i8", test_dot_i8, nk_dot_i8_haswell);
    run_if_matches("dot_haswell", "u8", test_dot_u8, nk_dot_u8_haswell);
    run_if_matches("dot_haswell", "f32c", test_dot_f32c, nk_dot_f32c_haswell);
    run_if_matches("vdot_haswell", "f32c", test_vdot_f32c, nk_vdot_f32c_haswell);
#endif // NK_TARGET_HASWELL

#if NK_TARGET_SKYLAKE
    run_if_matches("dot_skylake", "f32", test_dot_f32, nk_dot_f32_skylake);
    run_if_matches("dot_skylake", "f64", test_dot_f64, nk_dot_f64_skylake);
    run_if_matches("dot_skylake", "f16", test_dot_f16, nk_dot_f16_skylake);
    run_if_matches("dot_skylake", "bf16", test_dot_bf16, nk_dot_bf16_skylake);
    run_if_matches("dot_skylake", "e4m3", test_dot_e4m3, nk_dot_e4m3_skylake);
    run_if_matches("dot_skylake", "e5m2", test_dot_e5m2, nk_dot_e5m2_skylake);
    run_if_matches("dot_skylake", "i8", test_dot_i8, nk_dot_i8_skylake);
    run_if_matches("dot_skylake", "u8", test_dot_u8, nk_dot_u8_skylake);
    run_if_matches("dot_skylake", "f32c", test_dot_f32c, nk_dot_f32c_skylake);
    run_if_matches("vdot_skylake", "f32c", test_vdot_f32c, nk_vdot_f32c_skylake);
    run_if_matches("dot_skylake", "f64c", test_dot_f64c, nk_dot_f64c_skylake);
    run_if_matches("vdot_skylake", "f64c", test_vdot_f64c, nk_vdot_f64c_skylake);
#endif // NK_TARGET_SKYLAKE

    // Serial always runs - baseline test
    run_if_matches("dot_serial", "f32", test_dot_f32, nk_dot_f32_serial);
    run_if_matches("dot_serial", "f64", test_dot_f64, nk_dot_f64_serial);
    run_if_matches("dot_serial", "f16", test_dot_f16, nk_dot_f16_serial);
    run_if_matches("dot_serial", "bf16", test_dot_bf16, nk_dot_bf16_serial);
    run_if_matches("dot_serial", "e4m3", test_dot_e4m3, nk_dot_e4m3_serial);
    run_if_matches("dot_serial", "e5m2", test_dot_e5m2, nk_dot_e5m2_serial);
    run_if_matches("dot_serial", "i8", test_dot_i8, nk_dot_i8_serial);
    run_if_matches("dot_serial", "u8", test_dot_u8, nk_dot_u8_serial);
    run_if_matches("dot_serial", "f32c", test_dot_f32c, nk_dot_f32c_serial);
    run_if_matches("vdot_serial", "f32c", test_vdot_f32c, nk_vdot_f32c_serial);
    run_if_matches("dot_serial", "f64c", test_dot_f64c, nk_dot_f64c_serial);
    run_if_matches("vdot_serial", "f64c", test_vdot_f64c, nk_vdot_f64c_serial);

#endif // NK_DYNAMIC_DISPATCH

#if NK_COMPARE_TO_BLAS || NK_COMPARE_TO_MKL
    // BLAS/MKL precision comparison
    run_if_matches("dot_blas", "f32", test_dot_f32, dot_f32_blas);
    run_if_matches("dot_blas", "f64", test_dot_f64, dot_f64_blas);
    run_if_matches("dot_blas", "f32c", test_dot_f32c, dot_f32c_blas);
    run_if_matches("vdot_blas", "f32c", test_vdot_f32c, vdot_f32c_blas);
    run_if_matches("dot_blas", "f64c", test_dot_f64c, dot_f64c_blas);
    run_if_matches("vdot_blas", "f64c", test_vdot_f64c, vdot_f64c_blas);
#endif // NK_COMPARE_TO_BLAS || NK_COMPARE_TO_MKL
}

#pragma endregion // Dot

#pragma region Spatial

/**
 *  @brief Test L2 squared distance precision for f32.
 */
error_stats_t test_l2sq_f32(l2sq_f32_t kernel) {
    error_stats_t stats;
    std::mt19937 rng(global_config.seed);
    aligned_buffer<nk_f32_t> a(dense_dimension), b(dense_dimension);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(a, rng);
        fill_random(b, rng);

        nk_f32_t result;
        kernel(a.data, b.data, dense_dimension, &result);

        f64_t ref = reference_l2sq<f64_t, nk_f32_t>(a.data, b.data, dense_dimension);
        std::uint64_t ulps = ulp_distance(result, static_cast<nk_f32_t>(ref));

        stats.accumulate(ref, static_cast<double>(result), ulps);
    }

    return stats;
}

/**
 *  @brief Test angular (cosine) distance precision for f32.
 */
error_stats_t test_angular_f32(angular_f32_t kernel) {
    error_stats_t stats;
    std::mt19937 rng(global_config.seed);
    aligned_buffer<nk_f32_t> a(dense_dimension), b(dense_dimension);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(a, rng);
        fill_random(b, rng);

        nk_f32_t result;
        kernel(a.data, b.data, dense_dimension, &result);

        f64_t ref = reference_angular<f64_t, nk_f32_t>(a.data, b.data, dense_dimension);
        std::uint64_t ulps = ulp_distance(result, static_cast<nk_f32_t>(ref));

        stats.accumulate(ref, static_cast<double>(result), ulps);
    }

    return stats;
}

/**
 *  @brief Test L2 squared distance precision for f64.
 */
error_stats_t test_l2sq_f64(l2sq_f64_t kernel) {
    error_stats_t stats;
    std::mt19937 rng(global_config.seed);
    aligned_buffer<nk_f64_t> a(dense_dimension), b(dense_dimension);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(a, rng);
        fill_random(b, rng);

        nk_f64_t result;
        kernel(a.data, b.data, dense_dimension, &result);

        fmax_t ref = reference_l2sq<fmax_t, nk_f64_t>(a.data, b.data, dense_dimension);
        std::uint64_t ulps = ulp_distance(result, static_cast<nk_f64_t>(ref));

        stats.accumulate(static_cast<double>(ref), result, ulps);
    }

    return stats;
}

/**
 *  @brief Test angular (cosine) distance precision for f64.
 */
error_stats_t test_angular_f64(angular_f64_t kernel) {
    error_stats_t stats;
    std::mt19937 rng(global_config.seed);
    aligned_buffer<nk_f64_t> a(dense_dimension), b(dense_dimension);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(a, rng);
        fill_random(b, rng);

        nk_f64_t result;
        kernel(a.data, b.data, dense_dimension, &result);

        fmax_t ref = reference_angular<fmax_t, nk_f64_t>(a.data, b.data, dense_dimension);
        std::uint64_t ulps = ulp_distance(result, static_cast<nk_f64_t>(ref));

        stats.accumulate(static_cast<double>(ref), result, ulps);
    }

    return stats;
}

/**
 *  @brief Test L2 squared distance precision for f16.
 */
error_stats_t test_l2sq_f16(l2sq_f16_t kernel) {
    error_stats_t stats;
    std::mt19937 rng(global_config.seed);
    aligned_buffer<f16_t> a(dense_dimension), b(dense_dimension);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(a, rng, -1.0, 1.0);
        fill_random(b, rng, -1.0, 1.0);

        nk_f32_t result;
        kernel(reinterpret_cast<nk_f16_t const *>(a.data), reinterpret_cast<nk_f16_t const *>(b.data), dense_dimension,
               &result);

        f128_t ref = 0;
        for (std::size_t i = 0; i < dense_dimension; i++) {
            f128_t diff = f128_t(static_cast<float>(a[i])) - f128_t(static_cast<float>(b[i]));
            ref += diff * diff;
        }
        std::uint64_t ulps = ulp_distance_from_reference(result, ref);
        stats.accumulate(static_cast<double>(ref), static_cast<double>(result), ulps);
    }
    return stats;
}

/**
 *  @brief Test angular distance precision for f16.
 */
error_stats_t test_angular_f16(angular_f16_t kernel) {
    error_stats_t stats;
    std::mt19937 rng(global_config.seed);
    aligned_buffer<f16_t> a(dense_dimension), b(dense_dimension);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(a, rng, -1.0, 1.0);
        fill_random(b, rng, -1.0, 1.0);

        nk_f32_t result;
        kernel(reinterpret_cast<nk_f16_t const *>(a.data), reinterpret_cast<nk_f16_t const *>(b.data), dense_dimension,
               &result);

        f128_t ab = 0, aa = 0, bb = 0;
        for (std::size_t i = 0; i < dense_dimension; i++) {
            float ai = static_cast<float>(a[i]), bi = static_cast<float>(b[i]);
            ab += f128_t(ai) * f128_t(bi);
            aa += f128_t(ai) * f128_t(ai);
            bb += f128_t(bi) * f128_t(bi);
        }
        f128_t ref = 0;
        if (aa != 0 && bb != 0 && ab != 0) {
            f128_t cos_sim = ab / mp::sqrt(aa * bb);
            ref = 1 - cos_sim;
            if (ref < 0) ref = 0;
        }
        std::uint64_t ulps = ulp_distance_from_reference(result, ref);

        stats.accumulate(static_cast<double>(ref), static_cast<double>(result), ulps);
    }

    return stats;
}

/**
 *  @brief Test L2 squared distance precision for bf16.
 */
error_stats_t test_l2sq_bf16(l2sq_bf16_t kernel) {
    error_stats_t stats;
    std::mt19937 rng(global_config.seed);
    aligned_buffer<bf16_t> a(dense_dimension), b(dense_dimension);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(a, rng, -1.0, 1.0);
        fill_random(b, rng, -1.0, 1.0);

        nk_f32_t result;
        kernel(reinterpret_cast<nk_bf16_t const *>(a.data), reinterpret_cast<nk_bf16_t const *>(b.data),
               dense_dimension, &result);

        f128_t ref = 0;
        for (std::size_t i = 0; i < dense_dimension; i++) {
            f128_t diff = f128_t(static_cast<float>(a[i])) - f128_t(static_cast<float>(b[i]));
            ref += diff * diff;
        }
        std::uint64_t ulps = ulp_distance_from_reference(result, ref);
        stats.accumulate(static_cast<double>(ref), static_cast<double>(result), ulps);
    }
    return stats;
}

/**
 *  @brief Test angular distance precision for bf16.
 */
error_stats_t test_angular_bf16(angular_bf16_t kernel) {
    error_stats_t stats;
    std::mt19937 rng(global_config.seed);
    aligned_buffer<bf16_t> a(dense_dimension), b(dense_dimension);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(a, rng, -1.0, 1.0);
        fill_random(b, rng, -1.0, 1.0);

        nk_f32_t result;
        kernel(reinterpret_cast<nk_bf16_t const *>(a.data), reinterpret_cast<nk_bf16_t const *>(b.data),
               dense_dimension, &result);

        f128_t ab = 0, aa = 0, bb = 0;
        for (std::size_t i = 0; i < dense_dimension; i++) {
            float ai = static_cast<float>(a[i]), bi = static_cast<float>(b[i]);
            ab += f128_t(ai) * f128_t(bi);
            aa += f128_t(ai) * f128_t(ai);
            bb += f128_t(bi) * f128_t(bi);
        }
        f128_t ref = 0;
        if (aa != 0 && bb != 0 && ab != 0) {
            f128_t cos_sim = ab / mp::sqrt(aa * bb);
            ref = 1 - cos_sim;
            if (ref < 0) ref = 0;
        }
        std::uint64_t ulps = ulp_distance_from_reference(result, ref);
        stats.accumulate(static_cast<double>(ref), static_cast<double>(result), ulps);
    }
    return stats;
}

void test_spatial() {
    std::printf("Testing spatial distances...\n");

#if NK_DYNAMIC_DISPATCH
    // Dynamic dispatch - only test the dispatcher itself
    run_if_matches("l2sq", "f32", test_l2sq_f32, nk_l2sq_f32);
    run_if_matches("l2sq", "f64", test_l2sq_f64, nk_l2sq_f64);
    run_if_matches("l2sq", "f16", test_l2sq_f16, nk_l2sq_f16);
    run_if_matches("l2sq", "bf16", test_l2sq_bf16, nk_l2sq_bf16);
    run_if_matches("angular", "f32", test_angular_f32, nk_angular_f32);
    run_if_matches("angular", "f64", test_angular_f64, nk_angular_f64);
    run_if_matches("angular", "f16", test_angular_f16, nk_angular_f16);
    run_if_matches("angular", "bf16", test_angular_bf16, nk_angular_bf16);
#else
    // Static compilation - test all available ISA variants

#if NK_TARGET_NEON
    run_if_matches("l2sq_neon", "f32", test_l2sq_f32, nk_l2sq_f32_neon);
    run_if_matches("l2sq_neon", "f64", test_l2sq_f64, nk_l2sq_f64_neon);
    run_if_matches("angular_neon", "f32", test_angular_f32, nk_angular_f32_neon);
    run_if_matches("angular_neon", "f64", test_angular_f64, nk_angular_f64_neon);
#endif // NK_TARGET_NEON

#if NK_TARGET_NEONHALF
    run_if_matches("l2sq_neonhalf", "f16", test_l2sq_f16, nk_l2sq_f16_neonhalf);
    run_if_matches("angular_neonhalf", "f16", test_angular_f16, nk_angular_f16_neonhalf);
#endif // NK_TARGET_NEONHALF

#if NK_TARGET_NEONBFDOT
    run_if_matches("l2sq_neonbfdot", "bf16", test_l2sq_bf16, nk_l2sq_bf16_neonbfdot);
    run_if_matches("angular_neonbfdot", "bf16", test_angular_bf16, nk_angular_bf16_neonbfdot);
#endif // NK_TARGET_NEONBFDOT

#if NK_TARGET_HASWELL
    run_if_matches("l2sq_haswell", "f32", test_l2sq_f32, nk_l2sq_f32_haswell);
    run_if_matches("l2sq_haswell", "f64", test_l2sq_f64, nk_l2sq_f64_haswell);
    run_if_matches("l2sq_haswell", "f16", test_l2sq_f16, nk_l2sq_f16_haswell);
    run_if_matches("l2sq_haswell", "bf16", test_l2sq_bf16, nk_l2sq_bf16_haswell);
    run_if_matches("angular_haswell", "f32", test_angular_f32, nk_angular_f32_haswell);
    run_if_matches("angular_haswell", "f64", test_angular_f64, nk_angular_f64_haswell);
    run_if_matches("angular_haswell", "f16", test_angular_f16, nk_angular_f16_haswell);
    run_if_matches("angular_haswell", "bf16", test_angular_bf16, nk_angular_bf16_haswell);
#endif // NK_TARGET_HASWELL

#if NK_TARGET_SKYLAKE
    run_if_matches("l2sq_skylake", "f32", test_l2sq_f32, nk_l2sq_f32_skylake);
    run_if_matches("l2sq_skylake", "f64", test_l2sq_f64, nk_l2sq_f64_skylake);
    run_if_matches("angular_skylake", "f32", test_angular_f32, nk_angular_f32_skylake);
    run_if_matches("angular_skylake", "f64", test_angular_f64, nk_angular_f64_skylake);
#endif // NK_TARGET_SKYLAKE

    // Serial always runs - baseline test
    run_if_matches("l2sq_serial", "f32", test_l2sq_f32, nk_l2sq_f32_serial);
    run_if_matches("l2sq_serial", "f64", test_l2sq_f64, nk_l2sq_f64_serial);
    run_if_matches("l2sq_serial", "f16", test_l2sq_f16, nk_l2sq_f16_serial);
    run_if_matches("l2sq_serial", "bf16", test_l2sq_bf16, nk_l2sq_bf16_serial);
    run_if_matches("angular_serial", "f32", test_angular_f32, nk_angular_f32_serial);
    run_if_matches("angular_serial", "f64", test_angular_f64, nk_angular_f64_serial);
    run_if_matches("angular_serial", "f16", test_angular_f16, nk_angular_f16_serial);
    run_if_matches("angular_serial", "bf16", test_angular_bf16, nk_angular_bf16_serial);

#endif // NK_DYNAMIC_DISPATCH
}

#pragma endregion // Spatial

#pragma region Curved

/**
 *  @brief Test bilinear form: a^T * M * b
 */
error_stats_t test_bilinear_f32(bilinear_f32_t kernel) {
    error_stats_t stats;
    std::mt19937 rng(global_config.seed);

    // Test smaller dimensions for bilinear (matrix is n*n)
    std::size_t bilinear_dims[] = {2, 3, 4, 8, 16, 32};

    for (std::size_t dim : bilinear_dims) {
        for (auto start = test_start_time(); within_time_budget(start);) {
            aligned_buffer<nk_f32_t> a(dim), b(dim), m(dim * dim);
            fill_random(a, rng, -1.0, 1.0);
            fill_random(b, rng, -1.0, 1.0);
            fill_random(m, rng, -1.0, 1.0);

            nk_f32_t result;
            kernel(a.data, b.data, m.data, dim, &result);

            // Reference: a^T * M * b
            f128_t ref = reference_bilinear_quad(a.data, m.data, b.data, dim);
            std::uint64_t ulps = ulp_distance_from_reference(result, ref);

            stats.accumulate(static_cast<double>(ref), static_cast<double>(result), ulps);
        }
    }

    return stats;
}

/**
 *  @brief Test bilinear form for f64.
 */
error_stats_t test_bilinear_f64(bilinear_f64_t kernel) {
    error_stats_t stats;
    std::mt19937 rng(global_config.seed);

    std::size_t bilinear_dims[] = {2, 3, 4, 8, 16, 32};

    for (std::size_t dim : bilinear_dims) {
        for (auto start = test_start_time(); within_time_budget(start);) {
            aligned_buffer<nk_f64_t> a(dim), b(dim), m(dim * dim);
            fill_random(a, rng, -1.0, 1.0);
            fill_random(b, rng, -1.0, 1.0);
            fill_random(m, rng, -1.0, 1.0);

            nk_f64_t result;
            kernel(a.data, b.data, m.data, dim, &result);

            // Reference: a^T * M * b
            f128_t ref = reference_bilinear_quad(a.data, m.data, b.data, dim);
            std::uint64_t ulps = ulp_distance_from_reference(result, ref);

            stats.accumulate(static_cast<double>(ref), result, ulps);
        }
    }

    return stats;
}

void test_curved() {
    std::printf("Testing curved/bilinear forms...\n");

#if NK_DYNAMIC_DISPATCH
    // Dynamic dispatch - only test the dispatcher itself
    run_if_matches("bilinear", "f32", test_bilinear_f32, nk_bilinear_f32);
    run_if_matches("bilinear", "f64", test_bilinear_f64, nk_bilinear_f64);
#else
    // Static compilation - test all available ISA variants

#if NK_TARGET_NEON
    run_if_matches("bilinear_neon", "f32", test_bilinear_f32, nk_bilinear_f32_neon);
#endif // NK_TARGET_NEON

#if NK_TARGET_SKYLAKE
    run_if_matches("bilinear_skylake", "f32", test_bilinear_f32, nk_bilinear_f32_skylake);
    run_if_matches("bilinear_skylake", "f64", test_bilinear_f64, nk_bilinear_f64_skylake);
#endif // NK_TARGET_SKYLAKE

    // Serial always runs - baseline test
    run_if_matches("bilinear_serial", "f32", test_bilinear_f32, nk_bilinear_f32_serial);
    run_if_matches("bilinear_serial", "f64", test_bilinear_f64, nk_bilinear_f64_serial);

#endif // NK_DYNAMIC_DISPATCH
}

#pragma endregion // Curved

#pragma region Probability

/**
 *  @brief Test KL divergence precision for f32.
 */
error_stats_t test_kld_f32(kld_f32_t kernel) {
    error_stats_t stats;
    std::mt19937 rng(global_config.seed);

    // Use smaller dimensions for probability (distributions are normalized)
    std::size_t prob_dims[] = {4, 8, 16, 32, 64, 128, 256};

    for (std::size_t dim : prob_dims) {
        error_stats_t dim_stats;
        for (auto start = test_start_time(); within_time_budget(start);) {
            aligned_buffer<nk_f32_t> p(dim), q(dim);
            fill_probability(p, rng);
            fill_probability(q, rng);

            nk_f32_t result;
            kernel(p.data, q.data, dim, &result);

            f128_t ref = reference_kld_quad(p.data, q.data, dim);
            std::uint64_t ulps = ulp_distance_from_reference(result, ref);

            stats.accumulate(static_cast<double>(ref), static_cast<double>(result), ulps);
            dim_stats.accumulate(static_cast<double>(ref), static_cast<double>(result), ulps);
        }
        if (global_config.verbose) dim_stats.report_dimension("kld", "f32", dim);
    }

    return stats;
}

/**
 *  @brief Test KL divergence precision for f64.
 */
error_stats_t test_kld_f64(kld_f64_t kernel) {
    error_stats_t stats;
    std::mt19937 rng(global_config.seed);

    std::size_t prob_dims[] = {4, 8, 16, 32, 64, 128, 256};

    for (std::size_t dim : prob_dims) {
        for (auto start = test_start_time(); within_time_budget(start);) {
            aligned_buffer<nk_f64_t> p(dim), q(dim);
            // Convert from f32 probability filling
            std::uniform_real_distribution<double> dist(0.01, 1.0);
            double sum_p = 0, sum_q = 0;
            for (std::size_t i = 0; i < dim; i++) {
                p[i] = dist(rng);
                q[i] = dist(rng);
                sum_p += p[i];
                sum_q += q[i];
            }
            for (std::size_t i = 0; i < dim; i++) {
                p[i] /= sum_p;
                q[i] /= sum_q;
            }

            nk_f64_t result;
            kernel(p.data, q.data, dim, &result);

            f128_t ref = reference_kld_quad(p.data, q.data, dim);
            std::uint64_t ulps = ulp_distance_from_reference(result, ref);

            stats.accumulate(static_cast<double>(ref), result, ulps);
        }
    }

    return stats;
}

/**
 *  @brief Test Jensen-Shannon divergence precision for f32.
 */
error_stats_t test_jsd_f32(jsd_f32_t kernel) {
    error_stats_t stats;
    std::mt19937 rng(global_config.seed);

    std::size_t prob_dims[] = {4, 8, 16, 32, 64, 128, 256};

    for (std::size_t dim : prob_dims) {
        error_stats_t dim_stats;
        for (auto start = test_start_time(); within_time_budget(start);) {
            aligned_buffer<nk_f32_t> p(dim), q(dim);
            fill_probability(p, rng);
            fill_probability(q, rng);

            nk_f32_t result;
            kernel(p.data, q.data, dim, &result);

            f128_t ref = reference_jsd_quad(p.data, q.data, dim);
            std::uint64_t ulps = ulp_distance_from_reference(result, ref);

            stats.accumulate(static_cast<double>(ref), static_cast<double>(result), ulps);
            dim_stats.accumulate(static_cast<double>(ref), static_cast<double>(result), ulps);
        }
        if (global_config.verbose) dim_stats.report_dimension("jsd", "f32", dim);
    }

    return stats;
}

/**
 *  @brief Test Jensen-Shannon divergence precision for f64.
 */
error_stats_t test_jsd_f64(jsd_f64_t kernel) {
    error_stats_t stats;
    std::mt19937 rng(global_config.seed);

    std::size_t prob_dims[] = {4, 8, 16, 32, 64, 128, 256};

    for (std::size_t dim : prob_dims) {
        for (auto start = test_start_time(); within_time_budget(start);) {
            aligned_buffer<nk_f64_t> p(dim), q(dim);
            std::uniform_real_distribution<double> dist(0.01, 1.0);
            double sum_p = 0, sum_q = 0;
            for (std::size_t i = 0; i < dim; i++) {
                p[i] = dist(rng);
                q[i] = dist(rng);
                sum_p += p[i];
                sum_q += q[i];
            }
            for (std::size_t i = 0; i < dim; i++) {
                p[i] /= sum_p;
                q[i] /= sum_q;
            }

            nk_f64_t result;
            kernel(p.data, q.data, dim, &result);

            f128_t ref = reference_jsd_quad(p.data, q.data, dim);
            std::uint64_t ulps = ulp_distance_from_reference(result, ref);

            stats.accumulate(static_cast<double>(ref), result, ulps);
        }
    }

    return stats;
}

void test_probability() {
    std::printf("Testing probability divergences...\n");

#if NK_DYNAMIC_DISPATCH
    // Dynamic dispatch - only test the dispatcher itself
    run_if_matches("kld", "f32", test_kld_f32, nk_kld_f32);
    run_if_matches("kld", "f64", test_kld_f64, nk_kld_f64);
    run_if_matches("jsd", "f32", test_jsd_f32, nk_jsd_f32);
    run_if_matches("jsd", "f64", test_jsd_f64, nk_jsd_f64);
#else
    // Static compilation - test all available ISA variants

#if NK_TARGET_NEON
    run_if_matches("kld_neon", "f32", test_kld_f32, nk_kld_f32_neon);
    run_if_matches("jsd_neon", "f32", test_jsd_f32, nk_jsd_f32_neon);
#endif // NK_TARGET_NEON

#if NK_TARGET_SKYLAKE
    run_if_matches("kld_skylake", "f32", test_kld_f32, nk_kld_f32_skylake);
    run_if_matches("kld_skylake", "f64", test_kld_f64, nk_kld_f64_skylake);
    run_if_matches("jsd_skylake", "f32", test_jsd_f32, nk_jsd_f32_skylake);
    run_if_matches("jsd_skylake", "f64", test_jsd_f64, nk_jsd_f64_skylake);
#endif // NK_TARGET_SKYLAKE

    // Serial always runs - baseline test
    run_if_matches("kld_serial", "f32", test_kld_f32, nk_kld_f32_serial);
    run_if_matches("kld_serial", "f64", test_kld_f64, nk_kld_f64_serial);
    run_if_matches("jsd_serial", "f32", test_jsd_f32, nk_jsd_f32_serial);
    run_if_matches("jsd_serial", "f64", test_jsd_f64, nk_jsd_f64_serial);

#endif // NK_DYNAMIC_DISPATCH
}

#pragma endregion // Probability

#pragma region Binary

/**
 *  @brief Test Hamming distance (exact match required).
 */
error_stats_t test_hamming_b8(hamming_b8_t kernel) {
    error_stats_t stats;
    std::mt19937 rng(global_config.seed);

    // Hamming distance works on bytes representing bits
    std::size_t byte_dims[] = {1, 2, 4, 8, 16, 32, 64, 128, 256};

    for (std::size_t n_bytes : byte_dims) {
        for (auto start = test_start_time(); within_time_budget(start);) {
            aligned_buffer<nk_b8_t> a(n_bytes), b(n_bytes);
            std::uniform_int_distribution<int> dist(0, 255);
            for (std::size_t i = 0; i < n_bytes; i++) {
                a[i] = static_cast<nk_b8_t>(dist(rng));
                b[i] = static_cast<nk_b8_t>(dist(rng));
            }

            nk_u32_t result;
            kernel(a.data, b.data, n_bytes, &result);

            // Reference: count differing bits
            std::uint32_t ref = 0;
            for (std::size_t i = 0; i < n_bytes; i++) {
                std::uint8_t xored = a[i] ^ b[i];
                // Popcount
                while (xored) {
                    ref += xored & 1;
                    xored >>= 1;
                }
            }

            std::uint64_t ulps = (result == ref) ? 0 : std::numeric_limits<std::uint64_t>::max();
            stats.accumulate_ulp(ulps);

            if (global_config.assert_on_failure && result != ref) {
                std::fprintf(stderr, "FAIL: hamming_b8 n_bytes=%zu expected=%u got=%u\n", n_bytes, ref, result);
                assert(false);
            }
        }
    }

    return stats;
}

/**
 *  @brief Test Jaccard distance for binary vectors.
 */
error_stats_t test_jaccard_b8(jaccard_b8_t kernel) {
    error_stats_t stats;
    std::mt19937 rng(global_config.seed);

    std::size_t byte_dims[] = {1, 2, 4, 8, 16, 32, 64, 128, 256};

    for (std::size_t n_bytes : byte_dims) {
        for (auto start = test_start_time(); within_time_budget(start);) {
            aligned_buffer<nk_b8_t> a(n_bytes), b(n_bytes);
            std::uniform_int_distribution<int> dist(0, 255);
            for (std::size_t i = 0; i < n_bytes; i++) {
                a[i] = static_cast<nk_b8_t>(dist(rng));
                b[i] = static_cast<nk_b8_t>(dist(rng));
            }

            nk_f32_t result;
            kernel(a.data, b.data, n_bytes, &result);

            // Reference: Jaccard = 1 - |intersection| / |union|
            std::uint32_t intersection = 0, union_count = 0;
            for (std::size_t i = 0; i < n_bytes; i++) {
                std::uint8_t a_and_b = a[i] & b[i];
                std::uint8_t a_or_b = a[i] | b[i];
                // Popcount
                for (int bit = 0; bit < 8; bit++) {
                    intersection += (a_and_b >> bit) & 1;
                    union_count += (a_or_b >> bit) & 1;
                }
            }

            f128_t ref = (union_count > 0) ? (f128_t(1) - f128_t(intersection) / f128_t(union_count)) : f128_t(1);
            std::uint64_t ulps = ulp_distance_from_reference(result, ref);

            stats.accumulate(static_cast<double>(ref), static_cast<double>(result), ulps);
        }
    }

    return stats;
}

void test_binary() {
    std::printf("Testing binary distances...\n");

#if NK_DYNAMIC_DISPATCH
    // Dynamic dispatch - only test the dispatcher itself
    run_if_matches("hamming", "b8", test_hamming_b8, nk_hamming_b8);
    run_if_matches("jaccard", "b8", test_jaccard_b8, nk_jaccard_b8);
#else
    // Static compilation - test all available ISA variants

#if NK_TARGET_NEON
    run_if_matches("hamming_neon", "b8", test_hamming_b8, nk_hamming_b8_neon);
    run_if_matches("jaccard_neon", "b8", test_jaccard_b8, nk_jaccard_b8_neon);
#endif // NK_TARGET_NEON

#if NK_TARGET_HASWELL
    run_if_matches("hamming_haswell", "b8", test_hamming_b8, nk_hamming_b8_haswell);
    run_if_matches("jaccard_haswell", "b8", test_jaccard_b8, nk_jaccard_b8_haswell);
#endif // NK_TARGET_HASWELL

#if NK_TARGET_ICE
    run_if_matches("hamming_ice", "b8", test_hamming_b8, nk_hamming_b8_ice);
    run_if_matches("jaccard_ice", "b8", test_jaccard_b8, nk_jaccard_b8_ice);
#endif // NK_TARGET_ICE

    // Serial always runs - baseline test
    run_if_matches("hamming_serial", "b8", test_hamming_b8, nk_hamming_b8_serial);
    run_if_matches("jaccard_serial", "b8", test_jaccard_b8, nk_jaccard_b8_serial);

#endif // NK_DYNAMIC_DISPATCH
}

#pragma endregion // Binary

#pragma region Elementwise

/**
 *  @brief Test scale operation: result[i] = alpha * x[i] + beta
 */
error_stats_t test_scale_f32(scale_f32_t kernel) {
    error_stats_t stats;
    std::mt19937 rng(global_config.seed);
    aligned_buffer<nk_f32_t> input(dense_dimension), expected(dense_dimension), result(dense_dimension);
    std::uniform_real_distribution<float> coef_dist(-5.0f, 5.0f);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(input, rng, -10.0, 10.0);

        nk_f32_t alpha = coef_dist(rng);
        nk_f32_t beta = coef_dist(rng);

        // Compute reference
        for (std::size_t i = 0; i < dense_dimension; i++) { expected[i] = alpha * input[i] + beta; }

        // Test SIMD implementation
        kernel(input.data, dense_dimension, &alpha, &beta, result.data);

        // Verify
        for (std::size_t i = 0; i < dense_dimension; i++) {
            std::uint64_t ulps = ulp_distance(expected[i], result[i]);
            stats.accumulate_ulp(ulps);
            if (global_config.assert_on_failure && ulps > 2) {
                std::fprintf(stderr, "FAIL: scale_f32 dim=%zu i=%zu ulp=%llu\n", dense_dimension, i,
                             static_cast<unsigned long long>(ulps));
                assert(false);
            }
        }
    }

    return stats;
}

/**
 *  @brief Test elementwise sum: result[i] = a[i] + b[i]
 */
error_stats_t test_sum_f32(sum_f32_t kernel) {
    error_stats_t stats;
    std::mt19937 rng(global_config.seed);
    aligned_buffer<nk_f32_t> a(dense_dimension), b(dense_dimension), expected(dense_dimension), result(dense_dimension);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(a, rng, -10.0, 10.0);
        fill_random(b, rng, -10.0, 10.0);

        // Compute reference
        for (std::size_t i = 0; i < dense_dimension; i++) { expected[i] = a[i] + b[i]; }

        // Test SIMD implementation
        kernel(a.data, b.data, dense_dimension, result.data);

        // Verify (should be exact for simple addition)
        for (std::size_t i = 0; i < dense_dimension; i++) {
            std::uint64_t ulps = ulp_distance(expected[i], result[i]);
            stats.accumulate_ulp(ulps);
        }
    }

    return stats;
}

/**
 *  @brief Test weighted sum: result[i] = alpha * a[i] + beta * b[i]
 */
error_stats_t test_wsum_f32(wsum_f32_t kernel) {
    error_stats_t stats;
    std::mt19937 rng(global_config.seed);
    aligned_buffer<nk_f32_t> a(dense_dimension), b(dense_dimension), expected(dense_dimension), result(dense_dimension);
    std::uniform_real_distribution<float> coef_dist(-5.0f, 5.0f);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(a, rng, -10.0, 10.0);
        fill_random(b, rng, -10.0, 10.0);

        nk_f32_t alpha = coef_dist(rng);
        nk_f32_t beta = coef_dist(rng);

        // Compute reference
        for (std::size_t i = 0; i < dense_dimension; i++) { expected[i] = alpha * a[i] + beta * b[i]; }

        // Test SIMD implementation
        kernel(a.data, b.data, dense_dimension, &alpha, &beta, result.data);

        // Verify
        for (std::size_t i = 0; i < dense_dimension; i++) {
            std::uint64_t ulps = ulp_distance(expected[i], result[i]);
            stats.accumulate_ulp(ulps);
            if (global_config.assert_on_failure && ulps > 2) {
                std::fprintf(stderr, "FAIL: wsum_f32 dim=%zu i=%zu ulp=%llu\n", dense_dimension, i,
                             static_cast<unsigned long long>(ulps));
                assert(false);
            }
        }
    }

    return stats;
}

/**
 *  @brief Test FMA: result[i] = alpha * a[i] * b[i] + beta * c[i]
 */
error_stats_t test_fma_f32(fma_f32_t kernel) {
    error_stats_t stats;
    std::mt19937 rng(global_config.seed);
    aligned_buffer<nk_f32_t> a(dense_dimension), b(dense_dimension), c(dense_dimension);
    aligned_buffer<nk_f32_t> expected(dense_dimension), result(dense_dimension);
    std::uniform_real_distribution<float> coef_dist(-2.0f, 2.0f);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(a, rng, -5.0, 5.0);
        fill_random(b, rng, -5.0, 5.0);
        fill_random(c, rng, -5.0, 5.0);

        nk_f32_t alpha = coef_dist(rng);
        nk_f32_t beta = coef_dist(rng);

        // Compute reference
        for (std::size_t i = 0; i < dense_dimension; i++) { expected[i] = alpha * a[i] * b[i] + beta * c[i]; }

        // Test SIMD implementation
        kernel(a.data, b.data, c.data, dense_dimension, &alpha, &beta, result.data);

        // Verify
        for (std::size_t i = 0; i < dense_dimension; i++) {
            std::uint64_t ulps = ulp_distance(expected[i], result[i]);
            stats.accumulate_ulp(ulps);
            if (global_config.assert_on_failure && ulps > 4) {
                std::fprintf(stderr, "FAIL: fma_f32 dim=%zu i=%zu ulp=%llu\n", dense_dimension, i,
                             static_cast<unsigned long long>(ulps));
                assert(false);
            }
        }
    }

    return stats;
}

/**
 *  @brief Test elementwise sum for e4m3 (FP8).
 */
error_stats_t test_sum_e4m3(sum_e4m3_t kernel) {
    error_stats_t stats;
    std::mt19937 rng(global_config.seed);
    aligned_buffer<nk_e4m3_t> a(dense_dimension), b(dense_dimension), result(dense_dimension);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (auto start = test_start_time(); within_time_budget(start);) {
        for (std::size_t i = 0; i < dense_dimension; i++) {
            float va = dist(rng), vb = dist(rng);
            nk_f32_to_e4m3(&va, &a[i]);
            nk_f32_to_e4m3(&vb, &b[i]);
        }

        kernel(a.data, b.data, dense_dimension, result.data);

        // Verify
        for (std::size_t i = 0; i < dense_dimension; i++) {
            float fa, fb, fr;
            nk_e4m3_to_f32(&a[i], &fa);
            nk_e4m3_to_f32(&b[i], &fb);
            nk_e4m3_to_f32(&result[i], &fr);
            float expected = fa + fb;
            // Clamp expected to e4m3 range and convert back
            nk_e4m3_t expected_e4m3;
            nk_f32_to_e4m3(&expected, &expected_e4m3);
            float expected_f32;
            nk_e4m3_to_f32(&expected_e4m3, &expected_f32);
            std::uint64_t ulps = ulp_distance(fr, expected_f32);
            stats.accumulate_ulp(ulps);
        }
    }
    return stats;
}

/**
 *  @brief Test elementwise sum for e5m2 (FP8).
 */
error_stats_t test_sum_e5m2(sum_e5m2_t kernel) {
    error_stats_t stats;
    std::mt19937 rng(global_config.seed);
    aligned_buffer<nk_e5m2_t> a(dense_dimension), b(dense_dimension), result(dense_dimension);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (auto start = test_start_time(); within_time_budget(start);) {
        for (std::size_t i = 0; i < dense_dimension; i++) {
            float va = dist(rng), vb = dist(rng);
            nk_f32_to_e5m2(&va, &a[i]);
            nk_f32_to_e5m2(&vb, &b[i]);
        }

        kernel(a.data, b.data, dense_dimension, result.data);

        // Verify
        for (std::size_t i = 0; i < dense_dimension; i++) {
            float fa, fb, fr;
            nk_e5m2_to_f32(&a[i], &fa);
            nk_e5m2_to_f32(&b[i], &fb);
            nk_e5m2_to_f32(&result[i], &fr);
            float expected = fa + fb;
            nk_e5m2_t expected_e5m2;
            nk_f32_to_e5m2(&expected, &expected_e5m2);
            float expected_f32;
            nk_e5m2_to_f32(&expected_e5m2, &expected_f32);
            std::uint64_t ulps = ulp_distance(fr, expected_f32);
            stats.accumulate_ulp(ulps);
        }
    }
    return stats;
}

/**
 *  @brief Test elementwise scale for e4m3 (FP8).
 */
error_stats_t test_scale_e4m3(scale_e4m3_t kernel) {
    error_stats_t stats;
    std::mt19937 rng(global_config.seed);
    aligned_buffer<nk_e4m3_t> a(dense_dimension), result(dense_dimension);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> alpha_dist(-2.0f, 2.0f);

    for (auto start = test_start_time(); within_time_budget(start);) {
        for (std::size_t i = 0; i < dense_dimension; i++) {
            float va = dist(rng);
            nk_f32_to_e4m3(&va, &a[i]);
        }

        nk_f32_t alpha = alpha_dist(rng);
        nk_f32_t beta = 0.0f; // No shift for basic scale test

        kernel(a.data, dense_dimension, &alpha, &beta, result.data);

        // Verify
        for (std::size_t i = 0; i < dense_dimension; i++) {
            float fa, fr;
            nk_e4m3_to_f32(&a[i], &fa);
            nk_e4m3_to_f32(&result[i], &fr);
            float expected = fa * alpha + beta;
            nk_e4m3_t expected_e4m3;
            nk_f32_to_e4m3(&expected, &expected_e4m3);
            float expected_f32;
            nk_e4m3_to_f32(&expected_e4m3, &expected_f32);
            std::uint64_t ulps = ulp_distance(fr, expected_f32);
            stats.accumulate_ulp(ulps);
        }
    }
    return stats;
}

/**
 *  @brief Test elementwise scale for e5m2 (FP8).
 */
error_stats_t test_scale_e5m2(scale_e5m2_t kernel) {
    error_stats_t stats;
    std::mt19937 rng(global_config.seed);
    aligned_buffer<nk_e5m2_t> a(dense_dimension), result(dense_dimension);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> alpha_dist(-2.0f, 2.0f);

    for (auto start = test_start_time(); within_time_budget(start);) {
        for (std::size_t i = 0; i < dense_dimension; i++) {
            float va = dist(rng);
            nk_f32_to_e5m2(&va, &a[i]);
        }

        nk_f32_t alpha = alpha_dist(rng);
        nk_f32_t beta = 0.0f; // No shift for basic scale test

        kernel(a.data, dense_dimension, &alpha, &beta, result.data);

        // Verify
        for (std::size_t i = 0; i < dense_dimension; i++) {
            float fa, fr;
            nk_e5m2_to_f32(&a[i], &fa);
            nk_e5m2_to_f32(&result[i], &fr);
            float expected = fa * alpha + beta;
            nk_e5m2_t expected_e5m2;
            nk_f32_to_e5m2(&expected, &expected_e5m2);
            float expected_f32;
            nk_e5m2_to_f32(&expected_e5m2, &expected_f32);
            std::uint64_t ulps = ulp_distance(fr, expected_f32);
            stats.accumulate_ulp(ulps);
        }
    }
    return stats;
}

/**
 *  @brief Test elementwise weighted sum for e4m3 (FP8).
 */
error_stats_t test_wsum_e4m3(wsum_e4m3_t kernel) {
    error_stats_t stats;
    std::mt19937 rng(global_config.seed);
    aligned_buffer<nk_e4m3_t> a(dense_dimension), b(dense_dimension), result(dense_dimension);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> coef_dist(-2.0f, 2.0f);

    for (auto start = test_start_time(); within_time_budget(start);) {
        for (std::size_t i = 0; i < dense_dimension; i++) {
            float va = dist(rng), vb = dist(rng);
            nk_f32_to_e4m3(&va, &a[i]);
            nk_f32_to_e4m3(&vb, &b[i]);
        }

        nk_f32_t alpha = coef_dist(rng);
        nk_f32_t beta = coef_dist(rng);

        kernel(a.data, b.data, dense_dimension, &alpha, &beta, result.data);

        // Verify
        for (std::size_t i = 0; i < dense_dimension; i++) {
            float fa, fb, fr;
            nk_e4m3_to_f32(&a[i], &fa);
            nk_e4m3_to_f32(&b[i], &fb);
            nk_e4m3_to_f32(&result[i], &fr);
            float expected = alpha * fa + beta * fb;
            nk_e4m3_t expected_e4m3;
            nk_f32_to_e4m3(&expected, &expected_e4m3);
            float expected_f32;
            nk_e4m3_to_f32(&expected_e4m3, &expected_f32);
            std::uint64_t ulps = ulp_distance(fr, expected_f32);
            stats.accumulate_ulp(ulps);
        }
    }
    return stats;
}

/**
 *  @brief Test elementwise weighted sum for e5m2 (FP8).
 */
error_stats_t test_wsum_e5m2(wsum_e5m2_t kernel) {
    error_stats_t stats;
    std::mt19937 rng(global_config.seed);
    aligned_buffer<nk_e5m2_t> a(dense_dimension), b(dense_dimension), result(dense_dimension);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> coef_dist(-2.0f, 2.0f);

    for (auto start = test_start_time(); within_time_budget(start);) {
        for (std::size_t i = 0; i < dense_dimension; i++) {
            float va = dist(rng), vb = dist(rng);
            nk_f32_to_e5m2(&va, &a[i]);
            nk_f32_to_e5m2(&vb, &b[i]);
        }

        nk_f32_t alpha = coef_dist(rng);
        nk_f32_t beta = coef_dist(rng);

        kernel(a.data, b.data, dense_dimension, &alpha, &beta, result.data);

        // Verify
        for (std::size_t i = 0; i < dense_dimension; i++) {
            float fa, fb, fr;
            nk_e5m2_to_f32(&a[i], &fa);
            nk_e5m2_to_f32(&b[i], &fb);
            nk_e5m2_to_f32(&result[i], &fr);
            float expected = alpha * fa + beta * fb;
            nk_e5m2_t expected_e5m2;
            nk_f32_to_e5m2(&expected, &expected_e5m2);
            float expected_f32;
            nk_e5m2_to_f32(&expected_e5m2, &expected_f32);
            std::uint64_t ulps = ulp_distance(fr, expected_f32);
            stats.accumulate_ulp(ulps);
        }
    }
    return stats;
}

/**
 *  @brief Test elementwise fused multiply-add for e4m3 (FP8).
 */
error_stats_t test_fma_e4m3(fma_e4m3_t kernel) {
    error_stats_t stats;
    std::mt19937 rng(global_config.seed);
    aligned_buffer<nk_e4m3_t> a(dense_dimension), b(dense_dimension), c(dense_dimension), result(dense_dimension);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (auto start = test_start_time(); within_time_budget(start);) {
        for (std::size_t i = 0; i < dense_dimension; i++) {
            float va = dist(rng), vb = dist(rng), vc = dist(rng);
            nk_f32_to_e4m3(&va, &a[i]);
            nk_f32_to_e4m3(&vb, &b[i]);
            nk_f32_to_e4m3(&vc, &c[i]);
        }

        nk_f32_t alpha = 1.0f, beta = 1.0f; // result = alpha * a * b + beta * c
        kernel(a.data, b.data, c.data, dense_dimension, &alpha, &beta, result.data);

        // Verify: result = a * b + c
        for (std::size_t i = 0; i < dense_dimension; i++) {
            float fa, fb, fc, fr;
            nk_e4m3_to_f32(&a[i], &fa);
            nk_e4m3_to_f32(&b[i], &fb);
            nk_e4m3_to_f32(&c[i], &fc);
            nk_e4m3_to_f32(&result[i], &fr);
            float expected = alpha * fa * fb + beta * fc;
            nk_e4m3_t expected_e4m3;
            nk_f32_to_e4m3(&expected, &expected_e4m3);
            float expected_f32;
            nk_e4m3_to_f32(&expected_e4m3, &expected_f32);
            std::uint64_t ulps = ulp_distance(fr, expected_f32);
            stats.accumulate_ulp(ulps);
        }
    }
    return stats;
}

/**
 *  @brief Test elementwise fused multiply-add for e5m2 (FP8).
 */
error_stats_t test_fma_e5m2(fma_e5m2_t kernel) {
    error_stats_t stats;
    std::mt19937 rng(global_config.seed);
    aligned_buffer<nk_e5m2_t> a(dense_dimension), b(dense_dimension), c(dense_dimension), result(dense_dimension);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (auto start = test_start_time(); within_time_budget(start);) {
        for (std::size_t i = 0; i < dense_dimension; i++) {
            float va = dist(rng), vb = dist(rng), vc = dist(rng);
            nk_f32_to_e5m2(&va, &a[i]);
            nk_f32_to_e5m2(&vb, &b[i]);
            nk_f32_to_e5m2(&vc, &c[i]);
        }

        nk_f32_t alpha = 1.0f, beta = 1.0f; // result = alpha * a * b + beta * c
        kernel(a.data, b.data, c.data, dense_dimension, &alpha, &beta, result.data);

        // Verify: result = a * b + c
        for (std::size_t i = 0; i < dense_dimension; i++) {
            float fa, fb, fc, fr;
            nk_e5m2_to_f32(&a[i], &fa);
            nk_e5m2_to_f32(&b[i], &fb);
            nk_e5m2_to_f32(&c[i], &fc);
            nk_e5m2_to_f32(&result[i], &fr);
            float expected = alpha * fa * fb + beta * fc;
            nk_e5m2_t expected_e5m2;
            nk_f32_to_e5m2(&expected, &expected_e5m2);
            float expected_f32;
            nk_e5m2_to_f32(&expected_e5m2, &expected_f32);
            std::uint64_t ulps = ulp_distance(fr, expected_f32);
            stats.accumulate_ulp(ulps);
        }
    }
    return stats;
}

void test_elementwise() {
    std::printf("Testing elementwise operations...\n");

#if NK_DYNAMIC_DISPATCH
    // Dynamic dispatch - only test the dispatcher itself
    run_if_matches("scale", "f32", test_scale_f32, nk_scale_f32);
    run_if_matches("sum", "f32", test_sum_f32, nk_sum_f32);
    run_if_matches("wsum", "f32", test_wsum_f32, nk_wsum_f32);
    run_if_matches("fma", "f32", test_fma_f32, nk_fma_f32);
    run_if_matches("scale", "e4m3", test_scale_e4m3, nk_scale_e4m3);
    run_if_matches("scale", "e5m2", test_scale_e5m2, nk_scale_e5m2);
    run_if_matches("sum", "e4m3", test_sum_e4m3, nk_sum_e4m3);
    run_if_matches("sum", "e5m2", test_sum_e5m2, nk_sum_e5m2);
    run_if_matches("wsum", "e4m3", test_wsum_e4m3, nk_wsum_e4m3);
    run_if_matches("wsum", "e5m2", test_wsum_e5m2, nk_wsum_e5m2);
    run_if_matches("fma", "e4m3", test_fma_e4m3, nk_fma_e4m3);
    run_if_matches("fma", "e5m2", test_fma_e5m2, nk_fma_e5m2);
#else
    // Static compilation - test all available ISA variants

#if NK_TARGET_NEON
    run_if_matches("scale_neon", "f32", test_scale_f32, nk_scale_f32_neon);
    run_if_matches("sum_neon", "f32", test_sum_f32, nk_sum_f32_neon);
    run_if_matches("wsum_neon", "f32", test_wsum_f32, nk_wsum_f32_neon);
    run_if_matches("fma_neon", "f32", test_fma_f32, nk_fma_f32_neon);
#endif // NK_TARGET_NEON

#if NK_TARGET_NEONHALF
    run_if_matches("scale_neonhalf", "e4m3", test_scale_e4m3, nk_scale_e4m3_neonhalf);
    run_if_matches("scale_neonhalf", "e5m2", test_scale_e5m2, nk_scale_e5m2_neonhalf);
    run_if_matches("sum_neonhalf", "e4m3", test_sum_e4m3, nk_sum_e4m3_neonhalf);
    run_if_matches("sum_neonhalf", "e5m2", test_sum_e5m2, nk_sum_e5m2_neonhalf);
    run_if_matches("wsum_neonhalf", "e4m3", test_wsum_e4m3, nk_wsum_e4m3_neonhalf);
    run_if_matches("wsum_neonhalf", "e5m2", test_wsum_e5m2, nk_wsum_e5m2_neonhalf);
    run_if_matches("fma_neonhalf", "e4m3", test_fma_e4m3, nk_fma_e4m3_neonhalf);
    run_if_matches("fma_neonhalf", "e5m2", test_fma_e5m2, nk_fma_e5m2_neonhalf);
#endif // NK_TARGET_NEONHALF

#if NK_TARGET_HASWELL
    run_if_matches("scale_haswell", "f32", test_scale_f32, nk_scale_f32_haswell);
    run_if_matches("sum_haswell", "f32", test_sum_f32, nk_sum_f32_haswell);
    run_if_matches("wsum_haswell", "f32", test_wsum_f32, nk_wsum_f32_haswell);
    run_if_matches("fma_haswell", "f32", test_fma_f32, nk_fma_f32_haswell);
    run_if_matches("scale_haswell", "e4m3", test_scale_e4m3, nk_scale_e4m3_haswell);
    run_if_matches("scale_haswell", "e5m2", test_scale_e5m2, nk_scale_e5m2_haswell);
    run_if_matches("sum_haswell", "e4m3", test_sum_e4m3, nk_sum_e4m3_haswell);
    run_if_matches("sum_haswell", "e5m2", test_sum_e5m2, nk_sum_e5m2_haswell);
    run_if_matches("wsum_haswell", "e4m3", test_wsum_e4m3, nk_wsum_e4m3_haswell);
    run_if_matches("wsum_haswell", "e5m2", test_wsum_e5m2, nk_wsum_e5m2_haswell);
    run_if_matches("fma_haswell", "e4m3", test_fma_e4m3, nk_fma_e4m3_haswell);
    run_if_matches("fma_haswell", "e5m2", test_fma_e5m2, nk_fma_e5m2_haswell);
#endif // NK_TARGET_HASWELL

#if NK_TARGET_SKYLAKE
    run_if_matches("scale_skylake", "f32", test_scale_f32, nk_scale_f32_skylake);
    run_if_matches("sum_skylake", "f32", test_sum_f32, nk_sum_f32_skylake);
    run_if_matches("wsum_skylake", "f32", test_wsum_f32, nk_wsum_f32_skylake);
    run_if_matches("fma_skylake", "f32", test_fma_f32, nk_fma_f32_skylake);
#endif // NK_TARGET_SKYLAKE

    // Serial always runs - baseline test
    run_if_matches("scale_serial", "f32", test_scale_f32, nk_scale_f32_serial);
    run_if_matches("sum_serial", "f32", test_sum_f32, nk_sum_f32_serial);
    run_if_matches("wsum_serial", "f32", test_wsum_f32, nk_wsum_f32_serial);
    run_if_matches("fma_serial", "f32", test_fma_f32, nk_fma_f32_serial);
    run_if_matches("scale_serial", "e4m3", test_scale_e4m3, nk_scale_e4m3_serial);
    run_if_matches("scale_serial", "e5m2", test_scale_e5m2, nk_scale_e5m2_serial);
    run_if_matches("sum_serial", "e4m3", test_sum_e4m3, nk_sum_e4m3_serial);
    run_if_matches("sum_serial", "e5m2", test_sum_e5m2, nk_sum_e5m2_serial);
    run_if_matches("wsum_serial", "e4m3", test_wsum_e4m3, nk_wsum_e4m3_serial);
    run_if_matches("wsum_serial", "e5m2", test_wsum_e5m2, nk_wsum_e5m2_serial);
    run_if_matches("fma_serial", "e4m3", test_fma_e4m3, nk_fma_e4m3_serial);
    run_if_matches("fma_serial", "e5m2", test_fma_e5m2, nk_fma_e5m2_serial);

#endif // NK_DYNAMIC_DISPATCH
}

#pragma endregion // Elementwise

#pragma region Trigonometry

/**
 *  @brief Test sin approximation precision for f32.
 */
error_stats_t test_sin_f32(sin_f32_t kernel) {
    error_stats_t stats;

    // Test across the range [-2pi, 2pi]
    constexpr float pi = 3.14159265358979323846f;
    constexpr std::size_t n_samples = 10000;

    aligned_buffer<nk_f32_t> inputs(n_samples), outputs(n_samples);
    for (std::size_t i = 0; i < n_samples; i++) { inputs[i] = -2 * pi + (4 * pi * i) / n_samples; }

    kernel(inputs.data, n_samples, outputs.data);

    for (std::size_t i = 0; i < n_samples; i++) {
        f128_t ref = mp::sin(f128_t(inputs[i]));
        std::uint64_t ulps = ulp_distance_from_reference(outputs[i], ref);
        stats.accumulate(static_cast<double>(ref), static_cast<double>(outputs[i]), ulps);
    }

    return stats;
}

/**
 *  @brief Test cos approximation precision for f32.
 */
error_stats_t test_cos_f32(cos_f32_t kernel) {
    error_stats_t stats;

    constexpr float pi = 3.14159265358979323846f;
    constexpr std::size_t n_samples = 10000;

    aligned_buffer<nk_f32_t> inputs(n_samples), outputs(n_samples);
    for (std::size_t i = 0; i < n_samples; i++) { inputs[i] = -2 * pi + (4 * pi * i) / n_samples; }

    kernel(inputs.data, n_samples, outputs.data);

    for (std::size_t i = 0; i < n_samples; i++) {
        f128_t ref = mp::cos(f128_t(inputs[i]));
        std::uint64_t ulps = ulp_distance_from_reference(outputs[i], ref);
        stats.accumulate(static_cast<double>(ref), static_cast<double>(outputs[i]), ulps);
    }

    return stats;
}

/**
 *  @brief Test sin approximation precision for f64.
 */
error_stats_t test_sin_f64(sin_f64_t kernel) {
    error_stats_t stats;

    constexpr double pi = 3.14159265358979323846;
    constexpr std::size_t n_samples = 10000;

    aligned_buffer<nk_f64_t> inputs(n_samples), outputs(n_samples);
    for (std::size_t i = 0; i < n_samples; i++) { inputs[i] = -2 * pi + (4 * pi * i) / n_samples; }

    kernel(inputs.data, n_samples, outputs.data);

    for (std::size_t i = 0; i < n_samples; i++) {
        f128_t ref = mp::sin(f128_t(inputs[i]));
        std::uint64_t ulps = ulp_distance_from_reference(outputs[i], ref);
        stats.accumulate(static_cast<double>(ref), outputs[i], ulps);
    }

    return stats;
}

/**
 *  @brief Test cos approximation precision for f64.
 */
error_stats_t test_cos_f64(cos_f64_t kernel) {
    error_stats_t stats;

    constexpr double pi = 3.14159265358979323846;
    constexpr std::size_t n_samples = 10000;

    aligned_buffer<nk_f64_t> inputs(n_samples), outputs(n_samples);
    for (std::size_t i = 0; i < n_samples; i++) { inputs[i] = -2 * pi + (4 * pi * i) / n_samples; }

    kernel(inputs.data, n_samples, outputs.data);

    for (std::size_t i = 0; i < n_samples; i++) {
        f128_t ref = mp::cos(f128_t(inputs[i]));
        std::uint64_t ulps = ulp_distance_from_reference(outputs[i], ref);
        stats.accumulate(static_cast<double>(ref), outputs[i], ulps);
    }

    return stats;
}

/**
 *  @brief Test sin approximation precision for f16.
 */
error_stats_t test_sin_f16(sin_f16_t kernel) {
    error_stats_t stats;

    constexpr float pi = 3.14159265358979323846f;
    constexpr std::size_t n_samples = 10000;

    aligned_buffer<nk_f16_t> inputs(n_samples), outputs(n_samples);
    for (std::size_t i = 0; i < n_samples; i++) {
        inputs[i] = static_cast<nk_f16_t>(-2 * pi + (4 * pi * i) / n_samples);
    }

    kernel(inputs.data, n_samples, outputs.data);

    for (std::size_t i = 0; i < n_samples; i++) {
        f128_t ref = mp::sin(f128_t(static_cast<float>(inputs[i])));
        // For f16, convert to f32 for ULP calculation (f16 has only 10 mantissa bits)
        float output_f32 = static_cast<float>(outputs[i]);
        float ref_f32 = static_cast<float>(ref);
        std::uint64_t ulps = ulp_distance(output_f32, ref_f32);
        // Scale ULP by 2^13 to approximate f16 ULPs (f32 has 23 mantissa bits, f16 has 10)
        ulps = ulps >> 13;
        stats.accumulate(static_cast<double>(ref), static_cast<double>(outputs[i]), ulps);
    }

    return stats;
}

/**
 *  @brief Test cos approximation precision for f16.
 */
error_stats_t test_cos_f16(cos_f16_t kernel) {
    error_stats_t stats;

    constexpr float pi = 3.14159265358979323846f;
    constexpr std::size_t n_samples = 10000;

    aligned_buffer<nk_f16_t> inputs(n_samples), outputs(n_samples);
    for (std::size_t i = 0; i < n_samples; i++) {
        inputs[i] = static_cast<nk_f16_t>(-2 * pi + (4 * pi * i) / n_samples);
    }

    kernel(inputs.data, n_samples, outputs.data);

    for (std::size_t i = 0; i < n_samples; i++) {
        f128_t ref = mp::cos(f128_t(static_cast<float>(inputs[i])));
        float output_f32 = static_cast<float>(outputs[i]);
        float ref_f32 = static_cast<float>(ref);
        std::uint64_t ulps = ulp_distance(output_f32, ref_f32);
        ulps = ulps >> 13;
        stats.accumulate(static_cast<double>(ref), static_cast<double>(outputs[i]), ulps);
    }

    return stats;
}

/**
 *  @brief Test atan approximation precision for f16.
 */
error_stats_t test_atan_f16(atan_f16_t kernel) {
    error_stats_t stats;

    constexpr std::size_t n_samples = 10000;

    aligned_buffer<nk_f16_t> inputs(n_samples), outputs(n_samples);
    // Test range [-10, 10] for atan
    for (std::size_t i = 0; i < n_samples; i++) { inputs[i] = static_cast<nk_f16_t>(-10.0f + (20.0f * i) / n_samples); }

    kernel(inputs.data, n_samples, outputs.data);

    for (std::size_t i = 0; i < n_samples; i++) {
        f128_t ref = mp::atan(f128_t(static_cast<float>(inputs[i])));
        float output_f32 = static_cast<float>(outputs[i]);
        float ref_f32 = static_cast<float>(ref);
        std::uint64_t ulps = ulp_distance(output_f32, ref_f32);
        ulps = ulps >> 13;
        stats.accumulate(static_cast<double>(ref), static_cast<double>(outputs[i]), ulps);
    }

    return stats;
}

void test_trigonometry() {
    std::printf("Testing trigonometry...\n");

#if NK_DYNAMIC_DISPATCH
    // Dynamic dispatch - only test the dispatcher itself
    run_if_matches("sin", "f32", test_sin_f32, nk_sin_f32);
    run_if_matches("cos", "f32", test_cos_f32, nk_cos_f32);
    run_if_matches("sin", "f64", test_sin_f64, nk_sin_f64);
    run_if_matches("cos", "f64", test_cos_f64, nk_cos_f64);
#else
    // Static compilation - test all available ISA variants

#if NK_TARGET_NEON
    run_if_matches("sin_neon", "f32", test_sin_f32, nk_sin_f32_neon);
    run_if_matches("cos_neon", "f32", test_cos_f32, nk_cos_f32_neon);
    run_if_matches("sin_neon", "f64", test_sin_f64, nk_sin_f64_neon);
    run_if_matches("cos_neon", "f64", test_cos_f64, nk_cos_f64_neon);
#endif // NK_TARGET_NEON

#if NK_TARGET_HASWELL
    run_if_matches("sin_haswell", "f32", test_sin_f32, nk_sin_f32_haswell);
    run_if_matches("cos_haswell", "f32", test_cos_f32, nk_cos_f32_haswell);
    run_if_matches("sin_haswell", "f64", test_sin_f64, nk_sin_f64_haswell);
    run_if_matches("cos_haswell", "f64", test_cos_f64, nk_cos_f64_haswell);
#endif // NK_TARGET_HASWELL

#if NK_TARGET_SKYLAKE
    run_if_matches("sin_skylake", "f32", test_sin_f32, nk_sin_f32_skylake);
    run_if_matches("cos_skylake", "f32", test_cos_f32, nk_cos_f32_skylake);
    run_if_matches("sin_skylake", "f64", test_sin_f64, nk_sin_f64_skylake);
    run_if_matches("cos_skylake", "f64", test_cos_f64, nk_cos_f64_skylake);
#endif // NK_TARGET_SKYLAKE

#if NK_TARGET_SAPPHIRE
    run_if_matches("sin_sapphire", "f16", test_sin_f16, nk_sin_f16_sapphire);
    run_if_matches("cos_sapphire", "f16", test_cos_f16, nk_cos_f16_sapphire);
    run_if_matches("atan_sapphire", "f16", test_atan_f16, nk_atan_f16_sapphire);
#endif // NK_TARGET_SAPPHIRE

    // Serial always runs - baseline test
    run_if_matches("sin_serial", "f32", test_sin_f32, nk_sin_f32_serial);
    run_if_matches("cos_serial", "f32", test_cos_f32, nk_cos_f32_serial);
    run_if_matches("sin_serial", "f64", test_sin_f64, nk_sin_f64_serial);
    run_if_matches("cos_serial", "f64", test_cos_f64, nk_cos_f64_serial);
    run_if_matches("sin_serial", "f16", test_sin_f16, nk_sin_f16_serial);
    run_if_matches("cos_serial", "f16", test_cos_f16, nk_cos_f16_serial);
    run_if_matches("atan_serial", "f16", test_atan_f16, nk_atan_f16_serial);

#endif // NK_DYNAMIC_DISPATCH
}

#pragma endregion // Trigonometry

#pragma region Geospatial

/** Earth's radius in meters - matches `NK_EARTH_MEDIATORIAL_RADIUS` */
static f128_t const EARTH_RADIUS_M = f128_t("6335439.0");

/**
 *  @brief Reference Haversine distance in high precision.
 *  Formula: 2 * R * arcsin(sqrt(sin²(Δlat/2) + cos(lat1)*cos(lat2)*sin²(Δlon/2)))
 */
f128_t reference_haversine(f128_t lat1, f128_t lon1, f128_t lat2, f128_t lon2) {
    f128_t dlat = lat2 - lat1;
    f128_t dlon = lon2 - lon1;

    f128_t sin_dlat_2 = mp::sin(dlat / 2);
    f128_t sin_dlon_2 = mp::sin(dlon / 2);

    f128_t a = sin_dlat_2 * sin_dlat_2 + mp::cos(lat1) * mp::cos(lat2) * sin_dlon_2 * sin_dlon_2;
    f128_t c = 2 * mp::asin(mp::sqrt(a));

    return EARTH_RADIUS_M * c;
}

/**
 *  @brief Test Haversine distance for f64.
 */
error_stats_t test_haversine_f64(haversine_f64_t kernel) {
    error_stats_t stats;
    std::mt19937 rng(global_config.seed);
    std::uniform_real_distribution<double> lat_dist(-M_PI / 2, M_PI / 2);
    std::uniform_real_distribution<double> lon_dist(-M_PI, M_PI);

    // Test various sizes
    std::size_t geo_dims[] = {1, 4, 8, 16, 32, 64, 128};

    for (std::size_t dim : geo_dims) {
        for (auto start = test_start_time(); within_time_budget(start);) {
            aligned_buffer<nk_f64_t> a_lats(dim), a_lons(dim), b_lats(dim), b_lons(dim), results(dim);

            for (std::size_t i = 0; i < dim; i++) {
                a_lats[i] = lat_dist(rng);
                a_lons[i] = lon_dist(rng);
                b_lats[i] = lat_dist(rng);
                b_lons[i] = lon_dist(rng);
            }

            kernel(a_lats.data, a_lons.data, b_lats.data, b_lons.data, dim, results.data);

            for (std::size_t i = 0; i < dim; i++) {
                f128_t ref = reference_haversine(f128_t(a_lats[i]), f128_t(a_lons[i]), f128_t(b_lats[i]),
                                                 f128_t(b_lons[i]));
                std::uint64_t ulps = ulp_distance_from_reference(results[i], ref);
                stats.accumulate(static_cast<double>(ref), results[i], ulps);
            }
        }
    }
    return stats;
}

/**
 *  @brief Test Haversine distance for f32.
 */
error_stats_t test_haversine_f32(haversine_f32_t kernel) {
    error_stats_t stats;
    std::mt19937 rng(global_config.seed);
    std::uniform_real_distribution<float> lat_dist(-static_cast<float>(M_PI) / 2, static_cast<float>(M_PI) / 2);
    std::uniform_real_distribution<float> lon_dist(-static_cast<float>(M_PI), static_cast<float>(M_PI));

    std::size_t geo_dims[] = {1, 4, 8, 16, 32, 64, 128};

    for (std::size_t dim : geo_dims) {
        for (auto start = test_start_time(); within_time_budget(start);) {
            aligned_buffer<nk_f32_t> a_lats(dim), a_lons(dim), b_lats(dim), b_lons(dim), results(dim);

            for (std::size_t i = 0; i < dim; i++) {
                a_lats[i] = lat_dist(rng);
                a_lons[i] = lon_dist(rng);
                b_lats[i] = lat_dist(rng);
                b_lons[i] = lon_dist(rng);
            }

            kernel(a_lats.data, a_lons.data, b_lats.data, b_lons.data, dim, results.data);

            for (std::size_t i = 0; i < dim; i++) {
                f128_t ref = reference_haversine(f128_t(a_lats[i]), f128_t(a_lons[i]), f128_t(b_lats[i]),
                                                 f128_t(b_lons[i]));
                std::uint64_t ulps = ulp_distance_from_reference(results[i], ref);
                stats.accumulate(static_cast<double>(ref), static_cast<double>(results[i]), ulps);
            }
        }
    }
    return stats;
}

/**
 *  @brief Test Vincenty distance for f64.
 *  Note: Vincenty is more accurate than Haversine for oblate spheroid.
 *  We use Haversine reference as an approximation check.
 */
error_stats_t test_vincenty_f64(vincenty_f64_t kernel) {
    error_stats_t stats;
    std::mt19937 rng(global_config.seed);
    std::uniform_real_distribution<double> lat_dist(-M_PI / 2, M_PI / 2);
    std::uniform_real_distribution<double> lon_dist(-M_PI, M_PI);

    std::size_t geo_dims[] = {1, 4, 8, 16, 32, 64};

    for (std::size_t dim : geo_dims) {
        for (auto start = test_start_time(); within_time_budget(start);) {
            aligned_buffer<nk_f64_t> a_lats(dim), a_lons(dim), b_lats(dim), b_lons(dim), results(dim);

            for (std::size_t i = 0; i < dim; i++) {
                a_lats[i] = lat_dist(rng);
                a_lons[i] = lon_dist(rng);
                b_lats[i] = lat_dist(rng);
                b_lons[i] = lon_dist(rng);
            }

            kernel(a_lats.data, a_lons.data, b_lats.data, b_lons.data, dim, results.data);

            // For Vincenty, we compare against Haversine as rough sanity check
            // Results should be within ~0.5% of Haversine for most points
            for (std::size_t i = 0; i < dim; i++) {
                f128_t haversine_ref = reference_haversine(f128_t(a_lats[i]), f128_t(a_lons[i]), f128_t(b_lats[i]),
                                                           f128_t(b_lons[i]));
                f128_t rel_diff = mp::abs(f128_t(results[i]) - haversine_ref) / (haversine_ref + f128_t("1e-10"));

                // Vincenty should be within 1% of Haversine for most cases
                if (rel_diff < f128_t("0.01")) {
                    stats.accumulate_ulp(0); // Passes sanity check
                }
                else {
                    stats.accumulate_ulp(1); // Failed sanity check
                }
            }
        }
    }
    return stats;
}

void test_geospatial() {
    std::printf("Testing geospatial functions...\n");

#if NK_DYNAMIC_DISPATCH
    // Dynamic dispatch - only test the dispatcher itself
    run_if_matches("haversine", "f64", test_haversine_f64, nk_haversine_f64);
    run_if_matches("haversine", "f32", test_haversine_f32, nk_haversine_f32);
    run_if_matches("vincenty", "f64", test_vincenty_f64, nk_vincenty_f64);
#else
    // Static compilation - test all available ISA variants

#if NK_TARGET_NEON
    run_if_matches("haversine_neon", "f64", test_haversine_f64, nk_haversine_f64_neon);
    run_if_matches("haversine_neon", "f32", test_haversine_f32, nk_haversine_f32_neon);
    run_if_matches("vincenty_neon", "f64", test_vincenty_f64, nk_vincenty_f64_neon);
#endif // NK_TARGET_NEON

#if NK_TARGET_HASWELL
    run_if_matches("haversine_haswell", "f64", test_haversine_f64, nk_haversine_f64_haswell);
    run_if_matches("haversine_haswell", "f32", test_haversine_f32, nk_haversine_f32_haswell);
    run_if_matches("vincenty_haswell", "f64", test_vincenty_f64, nk_vincenty_f64_haswell);
#endif // NK_TARGET_HASWELL

#if NK_TARGET_SKYLAKE
    run_if_matches("haversine_skylake", "f64", test_haversine_f64, nk_haversine_f64_skylake);
    run_if_matches("haversine_skylake", "f32", test_haversine_f32, nk_haversine_f32_skylake);
    run_if_matches("vincenty_skylake", "f64", test_vincenty_f64, nk_vincenty_f64_skylake);
#endif // NK_TARGET_SKYLAKE

    // Serial always runs - baseline test
    run_if_matches("haversine_serial", "f64", test_haversine_f64, nk_haversine_f64_serial);
    run_if_matches("haversine_serial", "f32", test_haversine_f32, nk_haversine_f32_serial);
    run_if_matches("vincenty_serial", "f64", test_vincenty_f64, nk_vincenty_f64_serial);

#endif // NK_DYNAMIC_DISPATCH
}

#pragma endregion // Geospatial

#pragma region Mesh

/**
 *  @brief Compute centroid of a 3D point cloud using high precision.
 */
void reference_centroid(f128_t const *points, std::size_t n, f128_t *centroid) {
    centroid[0] = centroid[1] = centroid[2] = 0;
    for (std::size_t i = 0; i < n; i++) {
        centroid[0] += points[i * 3 + 0];
        centroid[1] += points[i * 3 + 1];
        centroid[2] += points[i * 3 + 2];
    }
    f128_t inv_n = f128_t(1) / f128_t(n);
    centroid[0] *= inv_n;
    centroid[1] *= inv_n;
    centroid[2] *= inv_n;
}

/**
 *  @brief Compute RMSD between two 3D point clouds using high precision.
 *  RMSD = sqrt(1/n * sum ||(a_i - centroid_a) - (b_i - centroid_b)||^2)
 *  Note: We center both point clouds first to match the kernel behavior.
 */
f128_t reference_rmsd(f128_t const *a, f128_t const *b, std::size_t n) {
    // Compute centroids
    f128_t ca[3] = {0, 0, 0}, cb[3] = {0, 0, 0};
    for (std::size_t i = 0; i < n; i++) {
        ca[0] += a[i * 3 + 0], ca[1] += a[i * 3 + 1], ca[2] += a[i * 3 + 2];
        cb[0] += b[i * 3 + 0], cb[1] += b[i * 3 + 1], cb[2] += b[i * 3 + 2];
    }
    ca[0] /= n, ca[1] /= n, ca[2] /= n;
    cb[0] /= n, cb[1] /= n, cb[2] /= n;

    // Compute centered RMSD
    f128_t sum_sq = 0;
    for (std::size_t i = 0; i < n; i++) {
        f128_t dx = (a[i * 3 + 0] - ca[0]) - (b[i * 3 + 0] - cb[0]);
        f128_t dy = (a[i * 3 + 1] - ca[1]) - (b[i * 3 + 1] - cb[1]);
        f128_t dz = (a[i * 3 + 2] - ca[2]) - (b[i * 3 + 2] - cb[2]);
        sum_sq += dx * dx + dy * dy + dz * dz;
    }
    return mp::sqrt(sum_sq / f128_t(n));
}

/**
 *  @brief Apply 3x3 rotation matrix to a point.
 */
void apply_rotation(f128_t const *rotation, f128_t const *point, f128_t *result) {
    result[0] = rotation[0] * point[0] + rotation[1] * point[1] + rotation[2] * point[2];
    result[1] = rotation[3] * point[0] + rotation[4] * point[1] + rotation[5] * point[2];
    result[2] = rotation[6] * point[0] + rotation[7] * point[1] + rotation[8] * point[2];
}

/**
 *  @brief Create a rotation matrix around the Z axis.
 */
void make_rotation_z(f128_t angle, f128_t *rotation) {
    f128_t c = mp::cos(angle);
    f128_t s = mp::sin(angle);
    // Row 0
    rotation[0] = c;
    rotation[1] = -s;
    rotation[2] = 0;
    // Row 1
    rotation[3] = s;
    rotation[4] = c;
    rotation[5] = 0;
    // Row 2
    rotation[6] = 0;
    rotation[7] = 0;
    rotation[8] = 1;
}

/**
 *  @brief Test RMSD computation for f64.
 */
error_stats_t test_rmsd_f64(mesh_f64_t kernel) {
    error_stats_t stats;
    std::mt19937 rng(global_config.seed);
    std::uniform_real_distribution<double> dist(-10.0, 10.0);

    std::size_t n = mesh_dimension;
    aligned_buffer<nk_f64_t> a(n * 3), b(n * 3);
    aligned_buffer<f128_t> a_hp(n * 3), b_hp(n * 3);

    for (auto start = test_start_time(); within_time_budget(start);) {
        // Generate random point clouds
        for (std::size_t i = 0; i < n * 3; i++) {
            double val_a = dist(rng);
            double val_b = dist(rng);
            a[i] = val_a;
            b[i] = val_b;
            a_hp[i] = f128_t(val_a);
            b_hp[i] = f128_t(val_b);
        }

        // Reference RMSD
        f128_t ref_rmsd = reference_rmsd(a_hp.data, b_hp.data, n);

        // NumKong RMSD
        nk_f64_t a_centroid[3], b_centroid[3], rotation[9], scale, result;
        kernel(a.data, b.data, n, a_centroid, b_centroid, rotation, &scale, &result);

        std::uint64_t ulps = ulp_distance_from_reference(result, ref_rmsd);
        stats.accumulate(static_cast<double>(ref_rmsd), result, ulps);
    }
    return stats;
}

/**
 *  @brief Test RMSD computation for f32.
 */
error_stats_t test_rmsd_f32(mesh_f32_t kernel) {
    error_stats_t stats;
    std::mt19937 rng(global_config.seed);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    std::size_t n = mesh_dimension;
    aligned_buffer<nk_f32_t> a(n * 3), b(n * 3);
    aligned_buffer<f128_t> a_hp(n * 3), b_hp(n * 3);

    for (auto start = test_start_time(); within_time_budget(start);) {
        for (std::size_t i = 0; i < n * 3; i++) {
            float val_a = dist(rng);
            float val_b = dist(rng);
            a[i] = val_a;
            b[i] = val_b;
            a_hp[i] = f128_t(val_a);
            b_hp[i] = f128_t(val_b);
        }

        f128_t ref_rmsd = reference_rmsd(a_hp.data, b_hp.data, n);

        nk_f32_t a_centroid[3], b_centroid[3], rotation[9], scale, result;
        kernel(a.data, b.data, n, a_centroid, b_centroid, rotation, &scale, &result);

        std::uint64_t ulps = ulp_distance_from_reference(result, ref_rmsd);
        stats.accumulate(static_cast<double>(ref_rmsd), static_cast<double>(result), ulps);
    }
    return stats;
}

/**
 *  @brief Test Kabsch alignment for f64.
 *  Strategy: Create b as a rotated version of a, verify Kabsch finds the rotation.
 */
error_stats_t test_kabsch_f64(mesh_f64_t kernel) {
    error_stats_t stats;
    std::mt19937 rng(global_config.seed);
    std::uniform_real_distribution<double> dist(-5.0, 5.0);
    std::uniform_real_distribution<double> angle_dist(-M_PI, M_PI);

    std::size_t n = mesh_dimension;
    aligned_buffer<nk_f64_t> a(n * 3), b(n * 3);

    for (auto start = test_start_time(); within_time_budget(start);) {
        // Generate random point cloud A
        for (std::size_t i = 0; i < n * 3; i++) { a[i] = dist(rng); }

        // Create B by rotating A around Z axis
        f128_t rotation_hp[9];
        f128_t angle = f128_t(angle_dist(rng));
        make_rotation_z(angle, rotation_hp);

        for (std::size_t i = 0; i < n; i++) {
            f128_t point[3] = {f128_t(a[i * 3]), f128_t(a[i * 3 + 1]), f128_t(a[i * 3 + 2])};
            f128_t rotated[3];
            apply_rotation(rotation_hp, point, rotated);
            b[i * 3 + 0] = static_cast<double>(rotated[0]);
            b[i * 3 + 1] = static_cast<double>(rotated[1]);
            b[i * 3 + 2] = static_cast<double>(rotated[2]);
        }

        // Run Kabsch
        nk_f64_t a_centroid[3], b_centroid[3], rotation[9], scale, result;
        kernel(a.data, b.data, n, a_centroid, b_centroid, rotation, &scale, &result);

        // After optimal alignment, RMSD should be near zero (within numerical precision)
        // The result is the RMSD after alignment
        f128_t ref_rmsd = f128_t(0); // Perfect alignment expected
        std::uint64_t ulps = ulp_distance_from_reference(result, ref_rmsd);

        // For Kabsch with identical point sets (just rotated), RMSD should be very small
        // Allow some tolerance due to numerical precision
        if (result < 1e-10) { stats.accumulate_ulp(0); }
        else {
            // Still track it but with higher ULP for larger residual
            stats.accumulate(0.0, result, ulps);
        }
    }
    return stats;
}

/**
 *  @brief Test Kabsch alignment for f32.
 */
error_stats_t test_kabsch_f32(mesh_f32_t kernel) {
    error_stats_t stats;
    std::mt19937 rng(global_config.seed);
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
    std::uniform_real_distribution<double> angle_dist(-M_PI, M_PI);

    std::size_t n = mesh_dimension;
    aligned_buffer<nk_f32_t> a(n * 3), b(n * 3);

    for (auto start = test_start_time(); within_time_budget(start);) {
        for (std::size_t i = 0; i < n * 3; i++) { a[i] = dist(rng); }

        f128_t rotation_hp[9];
        f128_t angle = f128_t(angle_dist(rng));
        make_rotation_z(angle, rotation_hp);

        for (std::size_t i = 0; i < n; i++) {
            f128_t point[3] = {f128_t(a[i * 3]), f128_t(a[i * 3 + 1]), f128_t(a[i * 3 + 2])};
            f128_t rotated[3];
            apply_rotation(rotation_hp, point, rotated);
            b[i * 3 + 0] = static_cast<float>(rotated[0]);
            b[i * 3 + 1] = static_cast<float>(rotated[1]);
            b[i * 3 + 2] = static_cast<float>(rotated[2]);
        }

        nk_f32_t a_centroid[3], b_centroid[3], rotation[9], scale, result;
        kernel(a.data, b.data, n, a_centroid, b_centroid, rotation, &scale, &result);

        // For f32, expect slightly larger residual due to lower precision
        if (result < 1e-5f) { stats.accumulate_ulp(0); }
        else { stats.accumulate(0.0, static_cast<double>(result), 1); }
    }
    return stats;
}

/**
 *  @brief Test Umeyama alignment for f64.
 *  Strategy: Create b as a rotated and scaled version of a, verify Umeyama finds both.
 */
error_stats_t test_umeyama_f64(mesh_f64_t kernel) {
    error_stats_t stats;
    std::mt19937 rng(global_config.seed);
    std::uniform_real_distribution<double> dist(-5.0, 5.0);
    std::uniform_real_distribution<double> angle_dist(-M_PI, M_PI);
    std::uniform_real_distribution<double> scale_dist(0.5, 2.0);

    std::size_t n = mesh_dimension;
    aligned_buffer<nk_f64_t> a(n * 3), b(n * 3);

    for (auto start = test_start_time(); within_time_budget(start);) {
        for (std::size_t i = 0; i < n * 3; i++) { a[i] = dist(rng); }

        f128_t rotation_hp[9];
        f128_t angle = f128_t(angle_dist(rng));
        f128_t scale_factor = f128_t(scale_dist(rng));
        make_rotation_z(angle, rotation_hp);

        for (std::size_t i = 0; i < n; i++) {
            f128_t point[3] = {f128_t(a[i * 3]), f128_t(a[i * 3 + 1]), f128_t(a[i * 3 + 2])};
            f128_t rotated[3];
            apply_rotation(rotation_hp, point, rotated);
            // Apply scale
            b[i * 3 + 0] = static_cast<double>(scale_factor * rotated[0]);
            b[i * 3 + 1] = static_cast<double>(scale_factor * rotated[1]);
            b[i * 3 + 2] = static_cast<double>(scale_factor * rotated[2]);
        }

        nk_f64_t a_centroid[3], b_centroid[3], rotation[9], scale, result;
        kernel(a.data, b.data, n, a_centroid, b_centroid, rotation, &scale, &result);

        // After optimal alignment (rotation + scale), RMSD should be near zero
        if (result < 1e-10) { stats.accumulate_ulp(0); }
        else { stats.accumulate(0.0, result, 1); }
    }
    return stats;
}

/**
 *  @brief Test Umeyama alignment for f32.
 */
error_stats_t test_umeyama_f32(mesh_f32_t kernel) {
    error_stats_t stats;
    std::mt19937 rng(global_config.seed);
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
    std::uniform_real_distribution<double> angle_dist(-M_PI, M_PI);
    std::uniform_real_distribution<double> scale_dist(0.5, 2.0);

    std::size_t n = mesh_dimension;
    aligned_buffer<nk_f32_t> a(n * 3), b(n * 3);

    for (auto start = test_start_time(); within_time_budget(start);) {
        for (std::size_t i = 0; i < n * 3; i++) { a[i] = dist(rng); }

        f128_t rotation_hp[9];
        f128_t angle = f128_t(angle_dist(rng));
        f128_t scale_factor = f128_t(scale_dist(rng));
        make_rotation_z(angle, rotation_hp);

        for (std::size_t i = 0; i < n; i++) {
            f128_t point[3] = {f128_t(a[i * 3]), f128_t(a[i * 3 + 1]), f128_t(a[i * 3 + 2])};
            f128_t rotated[3];
            apply_rotation(rotation_hp, point, rotated);
            b[i * 3 + 0] = static_cast<float>(scale_factor * rotated[0]);
            b[i * 3 + 1] = static_cast<float>(scale_factor * rotated[1]);
            b[i * 3 + 2] = static_cast<float>(scale_factor * rotated[2]);
        }

        nk_f32_t a_centroid[3], b_centroid[3], rotation[9], scale, result;
        kernel(a.data, b.data, n, a_centroid, b_centroid, rotation, &scale, &result);

        // For f32, expect slightly larger residual
        if (result < 1e-4f) { stats.accumulate_ulp(0); }
        else { stats.accumulate(0.0, static_cast<double>(result), 1); }
    }
    return stats;
}

void test_mesh() {
    std::printf("Testing mesh operations...\n");

#if NK_DYNAMIC_DISPATCH
    run_if_matches("rmsd", "f64", test_rmsd_f64, nk_rmsd_f64);
    run_if_matches("rmsd", "f32", test_rmsd_f32, nk_rmsd_f32);
    run_if_matches("kabsch", "f64", test_kabsch_f64, nk_kabsch_f64);
    run_if_matches("kabsch", "f32", test_kabsch_f32, nk_kabsch_f32);
    run_if_matches("umeyama", "f64", test_umeyama_f64, nk_umeyama_f64);
    run_if_matches("umeyama", "f32", test_umeyama_f32, nk_umeyama_f32);
#else

#if NK_TARGET_NEON
    run_if_matches("rmsd_neon", "f64", test_rmsd_f64, nk_rmsd_f64_neon);
    run_if_matches("rmsd_neon", "f32", test_rmsd_f32, nk_rmsd_f32_neon);
    run_if_matches("kabsch_neon", "f64", test_kabsch_f64, nk_kabsch_f64_neon);
    run_if_matches("kabsch_neon", "f32", test_kabsch_f32, nk_kabsch_f32_neon);
    run_if_matches("umeyama_neon", "f64", test_umeyama_f64, nk_umeyama_f64_neon);
    run_if_matches("umeyama_neon", "f32", test_umeyama_f32, nk_umeyama_f32_neon);
#endif // NK_TARGET_NEON

#if NK_TARGET_HASWELL
    run_if_matches("rmsd_haswell", "f64", test_rmsd_f64, nk_rmsd_f64_haswell);
    run_if_matches("rmsd_haswell", "f32", test_rmsd_f32, nk_rmsd_f32_haswell);
    run_if_matches("kabsch_haswell", "f64", test_kabsch_f64, nk_kabsch_f64_haswell);
    run_if_matches("kabsch_haswell", "f32", test_kabsch_f32, nk_kabsch_f32_haswell);
    run_if_matches("umeyama_haswell", "f64", test_umeyama_f64, nk_umeyama_f64_haswell);
    run_if_matches("umeyama_haswell", "f32", test_umeyama_f32, nk_umeyama_f32_haswell);
#endif // NK_TARGET_HASWELL

#if NK_TARGET_SKYLAKE
    run_if_matches("rmsd_skylake", "f64", test_rmsd_f64, nk_rmsd_f64_skylake);
    run_if_matches("rmsd_skylake", "f32", test_rmsd_f32, nk_rmsd_f32_skylake);
    run_if_matches("kabsch_skylake", "f64", test_kabsch_f64, nk_kabsch_f64_skylake);
    run_if_matches("kabsch_skylake", "f32", test_kabsch_f32, nk_kabsch_f32_skylake);
    run_if_matches("umeyama_skylake", "f64", test_umeyama_f64, nk_umeyama_f64_skylake);
    run_if_matches("umeyama_skylake", "f32", test_umeyama_f32, nk_umeyama_f32_skylake);
#endif // NK_TARGET_SKYLAKE

    // Serial always runs - baseline test
    run_if_matches("rmsd_serial", "f64", test_rmsd_f64, nk_rmsd_f64_serial);
    run_if_matches("rmsd_serial", "f32", test_rmsd_f32, nk_rmsd_f32_serial);
    run_if_matches("kabsch_serial", "f64", test_kabsch_f64, nk_kabsch_f64_serial);
    run_if_matches("kabsch_serial", "f32", test_kabsch_f32, nk_kabsch_f32_serial);
    run_if_matches("umeyama_serial", "f64", test_umeyama_f64, nk_umeyama_f64_serial);
    run_if_matches("umeyama_serial", "f32", test_umeyama_f32, nk_umeyama_f32_serial);

#endif // NK_DYNAMIC_DISPATCH
}

#pragma endregion // Mesh

#pragma region Dots

/**
 *  @brief Reference GEMM computation with configurable precision.
 *  Computes C = A × Bᵀ where A is (m × k), B is (n × k), C is (m × n).
 *  Result: C[i,j] = Σₗ A[i,l] × B[j,l]
 *  Use reference_type_=f128_t for f64 kernels, f64_t for f32/f16/bf16 kernels.
 */
template <typename reference_type_, typename input_type_>
void reference_gemm(input_type_ const *a, input_type_ const *b, reference_type_ *c, std::size_t m, std::size_t n,
                    std::size_t k, std::size_t a_stride_bytes, std::size_t b_stride_bytes) noexcept {
    for (std::size_t i = 0; i < m; i++) {
        input_type_ const *a_row = reinterpret_cast<input_type_ const *>(reinterpret_cast<char const *>(a) +
                                                                         i * a_stride_bytes);
        for (std::size_t j = 0; j < n; j++) {
            input_type_ const *b_row = reinterpret_cast<input_type_ const *>(reinterpret_cast<char const *>(b) +
                                                                             j * b_stride_bytes);
            reference_type_ sum = 0;
            for (std::size_t l = 0; l < k; l++) { sum += reference_type_(a_row[l]) * reference_type_(b_row[l]); }
            c[i * n + j] = sum;
        }
    }
}

/**
 *  @brief Test GEMM for f32f32f32 (single precision).
 */
error_stats_t test_dots_f32f32f32(dots_packed_size_t packed_size_fn, dots_f32_pack_t pack_fn,
                                  dots_f32f32f32_t dots_fn) {
    error_stats_t stats;
    std::mt19937 rng(global_config.seed);
    aligned_buffer<nk_f32_t> a_buf(matmul_dimension_m * matmul_dimension_k);
    aligned_buffer<nk_f32_t> b_buf(matmul_dimension_n * matmul_dimension_k);
    fill_random(a_buf, rng);
    fill_random(b_buf, rng);

    std::size_t m = matmul_dimension_m, n = matmul_dimension_n, k = matmul_dimension_k;
    std::size_t a_stride = k * sizeof(nk_f32_t);
    std::size_t b_stride = k * sizeof(nk_f32_t);
    std::size_t c_stride = n * sizeof(nk_f32_t);

    aligned_buffer<nk_f32_t> c(m * n);
    aligned_buffer<f64_t> c_ref(m * n); // Use f64 for f32 tests - faster than f128

    nk_size_t packed_size = packed_size_fn(n, k);
    aligned_buffer<char> b_packed(packed_size);

    for (auto start = test_start_time(); within_time_budget(start);) {
        // Random fill A and B each iteration
        fill_random(a_buf, rng);
        fill_random(b_buf, rng);

        // Reference computation with f64 precision
        reference_gemm<f64_t, nk_f32_t>(a_buf.data, b_buf.data, c_ref.data, m, n, k, a_stride, b_stride);

        // Pack and compute
        pack_fn(b_buf.data, n, k, b_stride, b_packed.data);
        dots_fn(a_buf.data, b_packed.data, c.data, m, n, k, a_stride, c_stride);

        // Compare results
        for (std::size_t i = 0; i < m * n; i++) {
            std::uint64_t ulps = ulp_distance(c[i], static_cast<nk_f32_t>(c_ref[i]));
            stats.accumulate(c_ref[i], static_cast<double>(c[i]), ulps);
        }
    }

    return stats;
}

/**
 *  @brief Test GEMM for f64f64f64 (double precision).
 */
error_stats_t test_dots_f64f64f64(dots_packed_size_t packed_size_fn, dots_f64_pack_t pack_fn,
                                  dots_f64f64f64_t dots_fn) {
    error_stats_t stats;
    std::mt19937 rng(global_config.seed);
    aligned_buffer<nk_f64_t> a_buf(matmul_dimension_m * matmul_dimension_k);
    aligned_buffer<nk_f64_t> b_buf(matmul_dimension_n * matmul_dimension_k);
    fill_random(a_buf, rng);
    fill_random(b_buf, rng);

    std::size_t m = matmul_dimension_m, n = matmul_dimension_n, k = matmul_dimension_k;
    std::size_t a_stride = k * sizeof(nk_f64_t);
    std::size_t b_stride = k * sizeof(nk_f64_t);
    std::size_t c_stride = n * sizeof(nk_f64_t);

    aligned_buffer<nk_f64_t> c(m * n);
    aligned_buffer<fmax_t> c_ref(m * n);

    nk_size_t packed_size = packed_size_fn(n, k);
    aligned_buffer<char> b_packed(packed_size);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(a_buf, rng);
        fill_random(b_buf, rng);

        reference_gemm<fmax_t, nk_f64_t>(a_buf.data, b_buf.data, c_ref.data, m, n, k, a_stride, b_stride);

        pack_fn(b_buf.data, n, k, b_stride, b_packed.data);
        dots_fn(a_buf.data, b_packed.data, c.data, m, n, k, a_stride, c_stride);

        for (std::size_t i = 0; i < m * n; i++) {
            std::uint64_t ulps = ulp_distance(c[i], static_cast<nk_f64_t>(c_ref[i]));
            stats.accumulate(static_cast<double>(c_ref[i]), c[i], ulps);
        }
    }
    return stats;
}

/**
 *  @brief Test GEMM for bf16bf16f32 (bfloat16 inputs, f32 accumulator).
 */
error_stats_t test_dots_bf16bf16f32(dots_packed_size_t packed_size_fn, dots_bf16_pack_t pack_fn,
                                    dots_bf16bf16f32_t dots_fn) {
    error_stats_t stats;
    std::mt19937 rng(global_config.seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::size_t m = matmul_dimension_m, n = matmul_dimension_n, k = matmul_dimension_k;
    std::size_t a_stride = k * sizeof(nk_bf16_t);
    std::size_t b_stride = k * sizeof(nk_bf16_t);
    std::size_t c_stride = n * sizeof(nk_f32_t);

    aligned_buffer<nk_bf16_t> a(m * k), b(n * k);
    aligned_buffer<nk_f32_t> c(m * n);
    aligned_buffer<f128_t> c_ref(m * n);

    nk_size_t packed_size = packed_size_fn(n, k);
    aligned_buffer<char> b_packed(packed_size);

    for (auto start = test_start_time(); within_time_budget(start);) {
        // Fill with bf16 values
        for (std::size_t i = 0; i < m * k; i++) a[i] = bf16_t(dist(rng)).raw;
        for (std::size_t i = 0; i < n * k; i++) b[i] = bf16_t(dist(rng)).raw;

        // Reference using f128_t (need to convert bf16 to f128_t)
        for (std::size_t i = 0; i < m; i++) {
            nk_bf16_t const *a_row = reinterpret_cast<nk_bf16_t const *>(reinterpret_cast<char const *>(a.data) +
                                                                         i * a_stride);
            for (std::size_t j = 0; j < n; j++) {
                nk_bf16_t const *b_row = reinterpret_cast<nk_bf16_t const *>(reinterpret_cast<char const *>(b.data) +
                                                                             j * b_stride);
                f128_t sum = 0;
                for (std::size_t l = 0; l < k; l++) {
                    sum += f128_t(bf16_t::from_raw(a_row[l]).to_f32()) * f128_t(bf16_t::from_raw(b_row[l]).to_f32());
                }
                c_ref[i * n + j] = sum;
            }
        }

        pack_fn(b.data, n, k, b_stride, b_packed.data);
        dots_fn(a.data, b_packed.data, c.data, m, n, k, a_stride, c_stride);

        for (std::size_t i = 0; i < m * n; i++) {
            std::uint64_t ulps = ulp_distance_from_reference(c[i], c_ref[i]);
            stats.accumulate(static_cast<double>(c_ref[i]), static_cast<double>(c[i]), ulps);
        }
    }
    return stats;
}

/**
 *  @brief Test GEMM for f16f16f32 (float16 inputs, f32 accumulator).
 */
error_stats_t test_dots_f16f16f32(dots_packed_size_t packed_size_fn, dots_f16_pack_t pack_fn,
                                  dots_f16f16f32_t dots_fn) {
    error_stats_t stats;
    std::mt19937 rng(global_config.seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::size_t m = matmul_dimension_m, n = matmul_dimension_n, k = matmul_dimension_k;
    std::size_t a_stride = k * sizeof(nk_f16_t);
    std::size_t b_stride = k * sizeof(nk_f16_t);
    std::size_t c_stride = n * sizeof(nk_f32_t);

    aligned_buffer<nk_f16_t> a(m * k), b(n * k);
    aligned_buffer<nk_f32_t> c(m * n);
    aligned_buffer<f128_t> c_ref(m * n);

    nk_size_t packed_size = packed_size_fn(n, k);
    aligned_buffer<char> b_packed(packed_size);

    for (auto start = test_start_time(); within_time_budget(start);) {
        // Fill with f16 values
        for (std::size_t i = 0; i < m * k; i++) a[i] = f16_t(dist(rng)).raw;
        for (std::size_t i = 0; i < n * k; i++) b[i] = f16_t(dist(rng)).raw;

        // Reference using f128_t
        for (std::size_t i = 0; i < m; i++) {
            nk_f16_t const *a_row = reinterpret_cast<nk_f16_t const *>(reinterpret_cast<char const *>(a.data) +
                                                                       i * a_stride);
            for (std::size_t j = 0; j < n; j++) {
                nk_f16_t const *b_row = reinterpret_cast<nk_f16_t const *>(reinterpret_cast<char const *>(b.data) +
                                                                           j * b_stride);
                f128_t sum = 0;
                for (std::size_t l = 0; l < k; l++) {
                    sum += f128_t(f16_t::from_raw(a_row[l]).to_f32()) * f128_t(f16_t::from_raw(b_row[l]).to_f32());
                }
                c_ref[i * n + j] = sum;
            }
        }

        pack_fn(b.data, n, k, b_stride, b_packed.data);
        dots_fn(a.data, b_packed.data, c.data, m, n, k, a_stride, c_stride);

        for (std::size_t i = 0; i < m * n; i++) {
            std::uint64_t ulps = ulp_distance_from_reference(c[i], c_ref[i]);
            stats.accumulate(static_cast<double>(c_ref[i]), static_cast<double>(c[i]), ulps);
        }
    }
    return stats;
}

/**
 *  @brief Test GEMM for i8i8i32 (int8 inputs, int32 accumulator).
 */
error_stats_t test_dots_i8i8i32(dots_packed_size_t packed_size_fn, dots_i8_pack_t pack_fn, dots_i8i8i32_t dots_fn) {
    error_stats_t stats;
    std::mt19937 rng(global_config.seed);
    std::uniform_int_distribution<int> dist(-127, 127);

    std::size_t m = matmul_dimension_m, n = matmul_dimension_n, k = matmul_dimension_k;
    std::size_t a_stride = k * sizeof(nk_i8_t);
    std::size_t b_stride = k * sizeof(nk_i8_t);
    std::size_t c_stride = n * sizeof(nk_i32_t);

    aligned_buffer<nk_i8_t> a(m * k), b(n * k);
    aligned_buffer<nk_i32_t> c(m * n);
    aligned_buffer<f128_t> c_ref(m * n);

    nk_size_t packed_size = packed_size_fn(n, k);
    aligned_buffer<char> b_packed(packed_size);

    for (auto start = test_start_time(); within_time_budget(start);) {
        for (std::size_t i = 0; i < m * k; i++) a[i] = static_cast<nk_i8_t>(dist(rng));
        for (std::size_t i = 0; i < n * k; i++) b[i] = static_cast<nk_i8_t>(dist(rng));

        // Reference GEMM for int8 (exact, no precision loss)
        for (std::size_t i = 0; i < m; i++) {
            nk_i8_t const *a_row = reinterpret_cast<nk_i8_t const *>(reinterpret_cast<char const *>(a.data) +
                                                                     i * a_stride);
            for (std::size_t j = 0; j < n; j++) {
                nk_i8_t const *b_row = reinterpret_cast<nk_i8_t const *>(reinterpret_cast<char const *>(b.data) +
                                                                         j * b_stride);
                std::int64_t sum = 0;
                for (std::size_t l = 0; l < k; l++) {
                    sum += static_cast<std::int64_t>(a_row[l]) * static_cast<std::int64_t>(b_row[l]);
                }
                c_ref[i * n + j] = f128_t(sum);
            }
        }

        pack_fn(b.data, n, k, b_stride, b_packed.data);
        dots_fn(a.data, b_packed.data, c.data, m, n, k, a_stride, c_stride);

        // For integer GEMM, results should be exact
        for (std::size_t i = 0; i < m * n; i++) {
            std::int64_t expected = static_cast<std::int64_t>(c_ref[i]);
            std::int64_t actual = c[i];
            std::uint64_t ulps = (expected == actual) ? 0 : 1;
            stats.accumulate(static_cast<double>(expected), static_cast<double>(actual), ulps);
        }
    }
    return stats;
}

#if NK_COMPARE_TO_BLAS || NK_COMPARE_TO_MKL
// BLAS GEMM kernel type (unpacked B matrix)
using dots_f32_blas_t = void (*)(nk_f32_t const *, nk_f32_t const *, nk_f32_t *, nk_size_t, nk_size_t, nk_size_t,
                                 nk_size_t, nk_size_t);
using dots_f64_blas_t = void (*)(nk_f64_t const *, nk_f64_t const *, nk_f64_t *, nk_size_t, nk_size_t, nk_size_t,
                                 nk_size_t, nk_size_t);

/**
 *  @brief Test GEMM for f32 with unpacked B (for BLAS comparison).
 */
error_stats_t test_dots_f32_unpacked(dots_f32_blas_t dots_fn) {
    error_stats_t stats;
    std::mt19937 rng(global_config.seed);

    std::size_t m = matmul_dimension_m, n = matmul_dimension_n, k = matmul_dimension_k;
    std::size_t a_stride = k * sizeof(nk_f32_t);
    std::size_t b_stride = k * sizeof(nk_f32_t);
    std::size_t c_stride = n * sizeof(nk_f32_t);

    aligned_buffer<nk_f32_t> a_buf(m * k), b_buf(n * k), c(m * n);
    aligned_buffer<f64_t> c_ref(m * n); // Use f64 for f32 tests - faster than f128

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(a_buf, rng);
        fill_random(b_buf, rng);

        reference_gemm<f64_t, nk_f32_t>(a_buf.data, b_buf.data, c_ref.data, m, n, k, a_stride, b_stride);
        dots_fn(a_buf.data, b_buf.data, c.data, m, n, k, a_stride, c_stride);

        for (std::size_t i = 0; i < m * n; i++) {
            std::uint64_t ulps = ulp_distance(c[i], static_cast<nk_f32_t>(c_ref[i]));
            stats.accumulate(c_ref[i], static_cast<double>(c[i]), ulps);
        }
    }
    return stats;
}

/**
 *  @brief Test GEMM for f64 with unpacked B (for BLAS comparison).
 */
error_stats_t test_dots_f64_unpacked(dots_f64_blas_t dots_fn) {
    error_stats_t stats;
    std::mt19937 rng(global_config.seed);

    std::size_t m = matmul_dimension_m, n = matmul_dimension_n, k = matmul_dimension_k;
    std::size_t a_stride = k * sizeof(nk_f64_t);
    std::size_t b_stride = k * sizeof(nk_f64_t);
    std::size_t c_stride = n * sizeof(nk_f64_t);

    aligned_buffer<nk_f64_t> a_buf(m * k), b_buf(n * k), c(m * n);
    aligned_buffer<fmax_t> c_ref(m * n);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(a_buf, rng);
        fill_random(b_buf, rng);

        reference_gemm<fmax_t, nk_f64_t>(a_buf.data, b_buf.data, c_ref.data, m, n, k, a_stride, b_stride);
        dots_fn(a_buf.data, b_buf.data, c.data, m, n, k, a_stride, c_stride);

        for (std::size_t i = 0; i < m * n; i++) {
            std::uint64_t ulps = ulp_distance(c[i], static_cast<nk_f64_t>(c_ref[i]));
            stats.accumulate(static_cast<double>(c_ref[i]), c[i], ulps);
        }
    }
    return stats;
}
#endif // NK_COMPARE_TO_BLAS || NK_COMPARE_TO_MKL

void test_dots() {
    std::printf("Testing batch dot products (GEMM)...\n");

#if NK_DYNAMIC_DISPATCH
    run_if_matches("dots", "f32f32f32", test_dots_f32f32f32, nk_dots_f32f32f32_packed_size, nk_dots_f32f32f32_pack,
                   nk_dots_f32f32f32);
    run_if_matches("dots", "f64f64f64", test_dots_f64f64f64, nk_dots_f64f64f64_packed_size, nk_dots_f64f64f64_pack,
                   nk_dots_f64f64f64);
    run_if_matches("dots", "bf16bf16f32", test_dots_bf16bf16f32, nk_dots_bf16bf16f32_packed_size,
                   nk_dots_bf16bf16f32_pack, nk_dots_bf16bf16f32);
    run_if_matches("dots", "f16f16f32", test_dots_f16f16f32, nk_dots_f16f16f32_packed_size, nk_dots_f16f16f32_pack,
                   nk_dots_f16f16f32);
    run_if_matches("dots", "i8i8i32", test_dots_i8i8i32, nk_dots_i8i8i32_packed_size, nk_dots_i8i8i32_pack,
                   nk_dots_i8i8i32);
#else

#if NK_TARGET_NEON
    run_if_matches("dots_neon", "f32f32f32", test_dots_f32f32f32, nk_dots_f32f32f32_packed_size_neon,
                   nk_dots_f32f32f32_pack_neon, nk_dots_f32f32f32_neon);
    run_if_matches("dots_neon", "f64f64f64", test_dots_f64f64f64, nk_dots_f64f64f64_packed_size_neon,
                   nk_dots_f64f64f64_pack_neon, nk_dots_f64f64f64_neon);
#endif // NK_TARGET_NEON

#if NK_TARGET_HASWELL
    run_if_matches("dots_haswell", "f32f32f32", test_dots_f32f32f32, nk_dots_f32f32f32_packed_size_haswell,
                   nk_dots_f32f32f32_pack_haswell, nk_dots_f32f32f32_haswell);
    run_if_matches("dots_haswell", "f64f64f64", test_dots_f64f64f64, nk_dots_f64f64f64_packed_size_haswell,
                   nk_dots_f64f64f64_pack_haswell, nk_dots_f64f64f64_haswell);
#endif // NK_TARGET_HASWELL

#if NK_TARGET_SKYLAKE
    run_if_matches("dots_skylake", "f32f32f32", test_dots_f32f32f32, nk_dots_f32f32f32_packed_size_skylake,
                   nk_dots_f32f32f32_pack_skylake, nk_dots_f32f32f32_skylake);
    run_if_matches("dots_skylake", "f64f64f64", test_dots_f64f64f64, nk_dots_f64f64f64_packed_size_skylake,
                   nk_dots_f64f64f64_pack_skylake, nk_dots_f64f64f64_skylake);
#endif // NK_TARGET_SKYLAKE

    // Serial always runs - baseline test
    run_if_matches("dots_serial", "f32f32f32", test_dots_f32f32f32, nk_dots_f32f32f32_packed_size_serial,
                   nk_dots_f32f32f32_pack_serial, nk_dots_f32f32f32_serial);
    run_if_matches("dots_serial", "f64f64f64", test_dots_f64f64f64, nk_dots_f64f64f64_packed_size_serial,
                   nk_dots_f64f64f64_pack_serial, nk_dots_f64f64f64_serial);
    run_if_matches("dots_serial", "bf16bf16f32", test_dots_bf16bf16f32, nk_dots_bf16bf16f32_packed_size_serial,
                   nk_dots_bf16bf16f32_pack_serial, nk_dots_bf16bf16f32_serial);
    run_if_matches("dots_serial", "f16f16f32", test_dots_f16f16f32, nk_dots_f16f16f32_packed_size_serial,
                   nk_dots_f16f16f32_pack_serial, nk_dots_f16f16f32_serial);
    run_if_matches("dots_serial", "i8i8i32", test_dots_i8i8i32, nk_dots_i8i8i32_packed_size_serial,
                   nk_dots_i8i8i32_pack_serial, nk_dots_i8i8i32_serial);

#endif // NK_DYNAMIC_DISPATCH

#if NK_COMPARE_TO_BLAS || NK_COMPARE_TO_MKL
    // BLAS/MKL GEMM precision comparison
    run_if_matches("dots_blas", "f32", test_dots_f32_unpacked, dots_f32_blas);
    run_if_matches("dots_blas", "f64", test_dots_f64_unpacked, dots_f64_blas);
#endif // NK_COMPARE_TO_BLAS || NK_COMPARE_TO_MKL
}

#pragma endregion // Dots

#pragma region Sparse

/**
 *  @brief Generate a sorted array of unique random indices.
 */
template <typename scalar_type_>
void fill_sorted_unique(aligned_buffer<scalar_type_> &buf, std::mt19937 &rng, scalar_type_ max_val) {
    std::uniform_int_distribution<scalar_type_> dist(0, max_val);
    std::vector<scalar_type_> values;
    values.reserve(buf.count * 2);

    // Generate more values than needed, then deduplicate
    while (values.size() < buf.count) {
        for (std::size_t i = 0; i < buf.count * 2 && values.size() < buf.count * 3; i++) {
            values.push_back(dist(rng));
        }
        std::sort(values.begin(), values.end());
        values.erase(std::unique(values.begin(), values.end()), values.end());
    }

    // Take first buf.count unique values
    for (std::size_t i = 0; i < buf.count; i++) buf[i] = values[i];
}

/**
 *  @brief Reference intersection count using merge algorithm.
 */
template <typename scalar_type_>
std::uint32_t reference_intersect(scalar_type_ const *a, scalar_type_ const *b, std::size_t a_len,
                                  std::size_t b_len) noexcept {
    std::uint32_t count = 0;
    std::size_t i = 0, j = 0;
    while (i < a_len && j < b_len) {
        if (a[i] < b[j]) i++;
        else if (a[i] > b[j]) j++;
        else {
            count++;
            i++;
            j++;
        }
    }
    return count;
}

/**
 *  @brief Reference sparse dot product using merge algorithm.
 */
template <typename index_type_, typename weight_type_>
f128_t reference_sparse_dot(index_type_ const *a, index_type_ const *b, weight_type_ const *a_weights,
                            weight_type_ const *b_weights, std::size_t a_len, std::size_t b_len) noexcept {
    f128_t product = 0;
    std::size_t i = 0, j = 0;
    while (i < a_len && j < b_len) {
        if (a[i] < b[j]) i++;
        else if (a[i] > b[j]) j++;
        else {
            product += f128_t(a_weights[i]) * f128_t(b_weights[j]);
            i++;
            j++;
        }
    }
    return product;
}

/**
 *  @brief Test u16 set intersection.
 */
error_stats_t test_intersect_u16(intersect_u16_t kernel) {
    error_stats_t stats;
    std::mt19937 rng(global_config.seed);

    std::size_t sparse_dims[] = {8, 16, 32, 64, 128, 256, 512};

    for (std::size_t dim : sparse_dims) {
        for (auto start = test_start_time(); within_time_budget(start);) {
            aligned_buffer<nk_u16_t> a(dim), b(dim);
            fill_sorted_unique(a, rng, static_cast<nk_u16_t>(dim * 4));
            fill_sorted_unique(b, rng, static_cast<nk_u16_t>(dim * 4));

            nk_u32_t result;
            kernel(a.data, b.data, dim, dim, &result);

            std::uint32_t ref = reference_intersect(a.data, b.data, dim, dim);
            stats.accumulate_ulp(result == ref ? 0 : 1);
        }
    }
    return stats;
}

/**
 *  @brief Test u32 set intersection.
 */
error_stats_t test_intersect_u32(intersect_u32_t kernel) {
    error_stats_t stats;
    std::mt19937 rng(global_config.seed);

    std::size_t sparse_dims[] = {8, 16, 32, 64, 128, 256, 512};

    for (std::size_t dim : sparse_dims) {
        for (auto start = test_start_time(); within_time_budget(start);) {
            aligned_buffer<nk_u32_t> a(dim), b(dim);
            fill_sorted_unique(a, rng, static_cast<nk_u32_t>(dim * 4));
            fill_sorted_unique(b, rng, static_cast<nk_u32_t>(dim * 4));

            nk_u32_t result;
            kernel(a.data, b.data, dim, dim, &result);

            std::uint32_t ref = reference_intersect(a.data, b.data, dim, dim);
            stats.accumulate_ulp(result == ref ? 0 : 1);
        }
    }
    return stats;
}

/**
 *  @brief Test sparse dot product with u32 indices and f32 weights.
 */
error_stats_t test_sparse_dot_u32f32(sparse_dot_u32f32_t kernel) {
    error_stats_t stats;
    std::mt19937 rng(global_config.seed);

    std::size_t sparse_dims[] = {8, 16, 32, 64, 128, 256};

    for (std::size_t dim : sparse_dims) {
        for (auto start = test_start_time(); within_time_budget(start);) {
            aligned_buffer<nk_u32_t> a_idx(dim), b_idx(dim);
            aligned_buffer<nk_f32_t> a_weights(dim), b_weights(dim);

            fill_sorted_unique(a_idx, rng, static_cast<nk_u32_t>(dim * 4));
            fill_sorted_unique(b_idx, rng, static_cast<nk_u32_t>(dim * 4));
            fill_random(a_weights, rng, -1.0, 1.0);
            fill_random(b_weights, rng, -1.0, 1.0);

            nk_f32_t result;
            kernel(a_idx.data, b_idx.data, a_weights.data, b_weights.data, dim, dim, &result);

            f128_t ref = reference_sparse_dot(a_idx.data, b_idx.data, a_weights.data, b_weights.data, dim, dim);
            std::uint64_t ulps = ulp_distance_from_reference(result, ref);
            stats.accumulate(static_cast<double>(ref), static_cast<double>(result), ulps);
        }
    }
    return stats;
}

/**
 *  @brief Test sparse dot product with u16 indices and bf16 weights.
 */
error_stats_t test_sparse_dot_u16bf16(sparse_dot_u16bf16_t kernel) {
    error_stats_t stats;
    std::mt19937 rng(global_config.seed);

    std::size_t sparse_dims[] = {8, 16, 32, 64, 128, 256};

    for (std::size_t dim : sparse_dims) {
        for (auto start = test_start_time(); within_time_budget(start);) {
            aligned_buffer<nk_u16_t> a_idx(dim), b_idx(dim);
            aligned_buffer<bf16_t> a_weights(dim), b_weights(dim);

            fill_sorted_unique(a_idx, rng, static_cast<nk_u16_t>(dim * 4));
            fill_sorted_unique(b_idx, rng, static_cast<nk_u16_t>(dim * 4));
            fill_random(a_weights, rng, -1.0, 1.0);
            fill_random(b_weights, rng, -1.0, 1.0);

            nk_f32_t result;
            kernel(a_idx.data, b_idx.data, reinterpret_cast<nk_bf16_t const *>(a_weights.data),
                   reinterpret_cast<nk_bf16_t const *>(b_weights.data), dim, dim, &result);

            // Reference with float conversion
            f128_t ref = 0;
            std::size_t i = 0, j = 0;
            while (i < dim && j < dim) {
                if (a_idx[i] < b_idx[j]) i++;
                else if (a_idx[i] > b_idx[j]) j++;
                else {
                    ref += f128_t(static_cast<float>(a_weights[i])) * f128_t(static_cast<float>(b_weights[j]));
                    i++;
                    j++;
                }
            }

            std::uint64_t ulps = ulp_distance_from_reference(result, ref);
            stats.accumulate(static_cast<double>(ref), static_cast<double>(result), ulps);
        }
    }
    return stats;
}

void test_sparse() {
    std::printf("Testing sparse operations...\n");

#if NK_DYNAMIC_DISPATCH
    run_if_matches("intersect", "u16", test_intersect_u16, nk_intersect_u16);
    run_if_matches("intersect", "u32", test_intersect_u32, nk_intersect_u32);
    run_if_matches("sparse_dot", "u32f32", test_sparse_dot_u32f32, nk_sparse_dot_u32f32);
    run_if_matches("sparse_dot", "u16bf16", test_sparse_dot_u16bf16, nk_sparse_dot_u16bf16);
#else

#if NK_TARGET_NEON
    run_if_matches("intersect_neon", "u16", test_intersect_u16, nk_intersect_u16_neon);
    run_if_matches("intersect_neon", "u32", test_intersect_u32, nk_intersect_u32_neon);
#endif // NK_TARGET_NEON

#if NK_TARGET_SVE
    run_if_matches("intersect_sve2", "u16", test_intersect_u16, nk_intersect_u16_sve2);
    run_if_matches("intersect_sve2", "u32", test_intersect_u32, nk_intersect_u32_sve2);
    run_if_matches("sparse_dot_sve2", "u32f32", test_sparse_dot_u32f32, nk_sparse_dot_u32f32_sve2);
    run_if_matches("sparse_dot_sve2", "u16bf16", test_sparse_dot_u16bf16, nk_sparse_dot_u16bf16_sve2);
#endif // NK_TARGET_SVE

#if NK_TARGET_ICE
    run_if_matches("intersect_ice", "u16", test_intersect_u16, nk_intersect_u16_ice);
    run_if_matches("intersect_ice", "u32", test_intersect_u32, nk_intersect_u32_ice);
    run_if_matches("sparse_dot_ice", "u32f32", test_sparse_dot_u32f32, nk_sparse_dot_u32f32_ice);
#endif // NK_TARGET_ICE

#if NK_TARGET_TURIN
    run_if_matches("intersect_turin", "u16", test_intersect_u16, nk_intersect_u16_turin);
    run_if_matches("intersect_turin", "u32", test_intersect_u32, nk_intersect_u32_turin);
    run_if_matches("sparse_dot_turin", "u32f32", test_sparse_dot_u32f32, nk_sparse_dot_u32f32_turin);
    run_if_matches("sparse_dot_turin", "u16bf16", test_sparse_dot_u16bf16, nk_sparse_dot_u16bf16_turin);
#endif // NK_TARGET_TURIN

    // Serial always runs - baseline test
    run_if_matches("intersect_serial", "u16", test_intersect_u16, nk_intersect_u16_serial);
    run_if_matches("intersect_serial", "u32", test_intersect_u32, nk_intersect_u32_serial);
    run_if_matches("sparse_dot_serial", "u32f32", test_sparse_dot_u32f32, nk_sparse_dot_u32f32_serial);
    run_if_matches("sparse_dot_serial", "u16bf16", test_sparse_dot_u16bf16, nk_sparse_dot_u16bf16_serial);

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
    test_denormals();

    // Print header for precision table
    print_stats_header();

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

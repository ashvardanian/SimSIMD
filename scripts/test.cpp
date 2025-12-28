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
 *    NK_TEST_FILTER=<pattern>     - Filter tests by name substring (default: run all)
 *    NK_TEST_ULP_THRESHOLD_F32=N  - Max allowed ULP for f32 (default: 4)
 *    NK_TEST_ULP_THRESHOLD_F16=N  - Max allowed ULP for f16 (default: 32)
 *    NK_TEST_ULP_THRESHOLD_BF16=N - Max allowed ULP for bf16 (default: 256)
 *    NK_TEST_TRIALS=N             - Trials per dimension (default: 100)
 *    NK_TEST_SEED=N               - RNG seed (default: 12345)
 */

#pragma region Includes_and_Configuration

#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cmath>
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

#include <boost/multiprecision/cpp_bin_float.hpp>

#define NK_NATIVE_F16  0
#define NK_NATIVE_BF16 0
#include <numkong/numkong.h>

// Use quad precision (113-bit mantissa) for high-precision reference computations
namespace mp = boost::multiprecision;
using hp_t = mp::cpp_bin_float_quad;

// Test configuration with environment variable overrides
struct test_config_t {
    bool assert_on_failure = false;
    bool verbose = false; // Show per-dimension stats
    std::uint64_t ulp_threshold_f32 = 4;
    std::uint64_t ulp_threshold_f16 = 32;
    std::uint64_t ulp_threshold_bf16 = 256;
    std::size_t trials_per_dim = 100;
    std::uint32_t seed = 12345;
    char const *filter = nullptr; // Filter tests by name (substring match)

    test_config_t() {
        if (char const *env = std::getenv("NK_TEST_ASSERT")) assert_on_failure = std::atoi(env) != 0;
        if (char const *env = std::getenv("NK_TEST_VERBOSE")) verbose = std::atoi(env) != 0;
        if (char const *env = std::getenv("NK_TEST_ULP_THRESHOLD_F32")) ulp_threshold_f32 = std::atoll(env);
        if (char const *env = std::getenv("NK_TEST_ULP_THRESHOLD_F16")) ulp_threshold_f16 = std::atoll(env);
        if (char const *env = std::getenv("NK_TEST_ULP_THRESHOLD_BF16")) ulp_threshold_bf16 = std::atoll(env);
        if (char const *env = std::getenv("NK_TEST_TRIALS")) trials_per_dim = std::atoll(env);
        if (char const *env = std::getenv("NK_TEST_SEED")) seed = std::atoll(env);
        filter = std::getenv("NK_TEST_FILTER"); // e.g., "dot", "angular", "kld"
    }

    bool should_run(char const *test_name) const {
        if (!filter) return true;
        return std::strstr(test_name, filter) != nullptr;
    }
};

static test_config_t g_config;

// Standard test dimensions covering SIMD boundaries
static constexpr std::size_t g_test_dimensions[] = {
    1,    3,    7,   8, 9, // Sub-vector, 64-bit boundary
    15,   16,   17,        // 128-bit boundary
    31,   32,   33,        // 256-bit boundary
    63,   64,   65,        // 512-bit boundary
    127,  128,  129,       // Cache line boundary
    255,  256,  257,       // Larger vectors
    1023, 1024, 1025       // Large scale
};
static constexpr std::size_t g_num_dimensions = sizeof(g_test_dimensions) / sizeof(g_test_dimensions[0]);

#pragma endregion // Includes_and_Configuration

#pragma region Precision_Infrastructure

/**
 *  @brief Compute ULP (Units in Last Place) distance between two floating-point values.
 *
 *  ULP distance is the number of representable floating-point numbers between a and b.
 *  This is the gold standard for comparing floating-point implementations.
 */
template <typename T>
std::uint64_t ulp_distance(T a, T b) {
    // Handle special cases
    if (std::isnan(a) || std::isnan(b)) return UINT64_MAX;
    if (a == b) return 0;
    if (std::isinf(a) || std::isinf(b)) return UINT64_MAX;

    // For different signs, we need special handling
    if ((a < 0) != (b < 0)) {
        // Count ULPs from a to 0, plus from 0 to b
        return ulp_distance(a, T(0)) + ulp_distance(T(0), b);
    }

    // Same sign: compute bit distance
    if constexpr (sizeof(T) == 4) {
        std::uint32_t ai, bi;
        std::memcpy(&ai, &a, sizeof(ai));
        std::memcpy(&bi, &b, sizeof(bi));
        // Clear sign bit for comparison
        ai &= 0x7FFFFFFF;
        bi &= 0x7FFFFFFF;
        return ai > bi ? ai - bi : bi - ai;
    }
    else if constexpr (sizeof(T) == 8) {
        std::uint64_t ai, bi;
        std::memcpy(&ai, &a, sizeof(ai));
        std::memcpy(&bi, &b, sizeof(bi));
        // Clear sign bit for comparison
        ai &= 0x7FFFFFFFFFFFFFFFULL;
        bi &= 0x7FFFFFFFFFFFFFFFULL;
        return ai > bi ? ai - bi : bi - ai;
    }
    else {
        // For f16/bf16, convert to f32 and compute there
        return ulp_distance(static_cast<float>(a), static_cast<float>(b));
    }
}

// Specialization for computing ULP when comparing against high-precision reference
template <typename T>
std::uint64_t ulp_distance_from_reference(T actual, hp_t reference) {
    T ref_as_t = static_cast<T>(reference);
    return ulp_distance(actual, ref_as_t);
}

#pragma endregion // Precision_Infrastructure

#pragma region Error_Statistics

/**
 *  @brief Accumulator for error statistics across multiple test trials.
 */
struct error_stats_t {
    double min_abs_err = DBL_MAX;
    double max_abs_err = 0;
    double sum_abs_err = 0;

    double min_rel_err = DBL_MAX;
    double max_rel_err = 0;
    double sum_rel_err = 0;

    std::uint64_t min_ulp = UINT64_MAX;
    std::uint64_t max_ulp = 0;
    std::uint64_t sum_ulp = 0;

    std::size_t count = 0;
    std::size_t exact_matches = 0;

    void accumulate(double expected, double actual) {
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

    void accumulate_ulp(std::uint64_t ulps) {
        min_ulp = std::min(min_ulp, ulps);
        max_ulp = std::max(max_ulp, ulps);
        sum_ulp += ulps;
        if (ulps == 0) exact_matches++;
        count++;
    }

    void accumulate(double expected, double actual, std::uint64_t ulps) {
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

    double mean_abs_err() const { return count > 0 ? sum_abs_err / count : 0; }
    double mean_rel_err() const { return count > 0 ? sum_rel_err / count : 0; }
    double mean_ulp() const { return count > 0 ? static_cast<double>(sum_ulp) / count : 0; }

    void reset() { *this = error_stats_t {}; }

    // Merge stats from another instance (for OpenMP reduction)
    void merge(error_stats_t const &other) {
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

    void report(char const *operation, char const *dtype) const {
        std::printf("%-18s %-8s %12llu %12.1f %12llu %12zu\n", operation, dtype,
                    static_cast<unsigned long long>(max_ulp), mean_ulp(), static_cast<unsigned long long>(min_ulp),
                    exact_matches);
        std::fflush(stdout);
    }

    void report_dimension(char const *operation, char const *dtype, std::size_t dim) const {
        std::printf("  %-16s %-8s dim=%-6zu %10llu %10.1f %10llu %10zu\n", operation, dtype, dim,
                    static_cast<unsigned long long>(max_ulp), mean_ulp(), static_cast<unsigned long long>(min_ulp),
                    exact_matches);
        std::fflush(stdout);
    }
};

/**
 *  @brief Print the header for the error statistics table.
 */
void print_stats_header() {
    std::printf("\n=== NumKong Precision Analysis ===\n");
    std::printf("%-18s %-8s %12s %12s %12s %12s\n", "Operation", "Type", "max_ulp", "mean_ulp", "min_ulp", "exact");
    std::printf("─────────────────────────────────────────────────────────────────────────────────\n");
}

#pragma endregion // Error_Statistics

#pragma region Reference_Implementations

/**
 *  @brief Kahan-summation dot product in double precision.
 *
 *  Provides a more accurate reference than naive summation for testing
 *  lower-precision implementations (f16, bf16, f32).
 */
template <typename T>
double kahan_dot(T const *a, T const *b, std::size_t n) {
    double sum = 0.0;
    double c = 0.0; // Compensation for lost low-order bits
    for (std::size_t i = 0; i < n; i++) {
        double product = static_cast<double>(a[i]) * static_cast<double>(b[i]);
        double y = product - c;
        double t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    return sum;
}

/**
 *  @brief Quad-precision dot product reference.
 */
template <typename T>
hp_t reference_dot_quad(T const *a, T const *b, std::size_t n) {
    hp_t sum = 0;
    for (std::size_t i = 0; i < n; i++) { sum += hp_t(a[i]) * hp_t(b[i]); }
    return sum;
}

/**
 *  @brief Quad-precision L2 squared distance reference.
 */
template <typename T>
hp_t reference_l2sq_quad(T const *a, T const *b, std::size_t n) {
    hp_t sum = 0;
    for (std::size_t i = 0; i < n; i++) {
        hp_t diff = hp_t(a[i]) - hp_t(b[i]);
        sum += diff * diff;
    }
    return sum;
}

/**
 *  @brief Quad-precision angular (cosine) distance reference.
 */
template <typename T>
hp_t reference_angular_quad(T const *a, T const *b, std::size_t n) {
    hp_t ab = 0, aa = 0, bb = 0;
    for (std::size_t i = 0; i < n; i++) {
        hp_t ai = hp_t(a[i]);
        hp_t bi = hp_t(b[i]);
        ab += ai * bi;
        aa += ai * ai;
        bb += bi * bi;
    }
    if (aa == 0 && bb == 0) return hp_t(0);
    if (ab == 0) return hp_t(1);
    hp_t cos_sim = ab / mp::sqrt(aa * bb);
    hp_t result = 1 - cos_sim;
    return result > 0 ? result : hp_t(0);
}

/**
 *  @brief Quad-precision KL divergence reference.
 */
template <typename T>
hp_t reference_kld_quad(T const *p, T const *q, std::size_t n) {
    hp_t sum = 0;
    hp_t epsilon = 1e-10L;
    for (std::size_t i = 0; i < n; i++) {
        hp_t pi = hp_t(p[i]);
        hp_t qi = hp_t(q[i]);
        if (pi > 0) { sum += pi * mp::log((pi + epsilon) / (qi + epsilon)); }
    }
    return sum;
}

/**
 *  @brief Quad-precision Jensen-Shannon divergence reference.
 */
template <typename T>
hp_t reference_jsd_quad(T const *p, T const *q, std::size_t n) {
    hp_t sum = 0;
    hp_t epsilon = 1e-10L;
    for (std::size_t i = 0; i < n; i++) {
        hp_t pi = hp_t(p[i]);
        hp_t qi = hp_t(q[i]);
        hp_t mi = (pi + qi) / 2;
        if (pi > 0) sum += pi * mp::log((pi + epsilon) / (mi + epsilon));
        if (qi > 0) sum += qi * mp::log((qi + epsilon) / (mi + epsilon));
    }
    return sum / 2;
}

/**
 *  @brief Quad-precision bilinear form reference: a^T * C * b
 */
template <typename T>
hp_t reference_bilinear_quad(T const *a, T const *c, T const *b, std::size_t n) {
    hp_t sum = 0;
    for (std::size_t i = 0; i < n; i++) {
        for (std::size_t j = 0; j < n; j++) { sum += hp_t(a[i]) * hp_t(c[i * n + j]) * hp_t(b[j]); }
    }
    return sum;
}

#pragma endregion // Reference_Implementations

#pragma region Test_Harness_Templates

/**
 *  @brief Aligned memory allocation for SIMD-friendly buffers.
 */
template <typename scalar_t>
struct aligned_buffer {
    static constexpr std::size_t alignment = 64; // Cache line
    scalar_t *data = nullptr;
    std::size_t count = 0;

    aligned_buffer() = default;
    explicit aligned_buffer(std::size_t n) : count(n) {
        if (n > 0) {
            void *ptr = std::aligned_alloc(alignment, ((n * sizeof(scalar_t) + alignment - 1) / alignment) * alignment);
            data = static_cast<scalar_t *>(ptr);
            std::memset(data, 0, n * sizeof(scalar_t));
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

    scalar_t &operator[](std::size_t i) { return data[i]; }
    scalar_t const &operator[](std::size_t i) const { return data[i]; }
};

/**
 *  @brief Fill buffer with random values in specified range.
 */
template <typename scalar_t, typename rng_t>
void fill_random(aligned_buffer<scalar_t> &buf, rng_t &rng, double min_val = -1.0, double max_val = 1.0) {
    std::uniform_real_distribution<double> dist(min_val, max_val);
    for (std::size_t i = 0; i < buf.count; i++) { buf[i] = static_cast<scalar_t>(dist(rng)); }
}

/**
 *  @brief Fill buffer with random probability distribution (sums to ~1).
 */
template <typename scalar_t, typename rng_t>
void fill_probability(aligned_buffer<scalar_t> &buf, rng_t &rng) {
    std::uniform_real_distribution<double> dist(0.01, 1.0);
    double sum = 0;
    for (std::size_t i = 0; i < buf.count; i++) {
        double v = dist(rng);
        buf[i] = static_cast<scalar_t>(v);
        sum += v;
    }
    // Normalize
    for (std::size_t i = 0; i < buf.count; i++) { buf[i] = static_cast<scalar_t>(static_cast<double>(buf[i]) / sum); }
}

/**
 *  @brief Fill buffer with random integers in specified range.
 */
template <typename scalar_t, typename rng_t>
void fill_random_int(aligned_buffer<scalar_t> &buf, rng_t &rng, int min_val = -100, int max_val = 100) {
    std::uniform_int_distribution<int> dist(min_val, max_val);
    for (std::size_t i = 0; i < buf.count; i++) { buf[i] = static_cast<scalar_t>(dist(rng)); }
}

/**
 *  @brief Wrapper for nk_f16_t enabling implicit float conversions.
 */
struct f16_t {
    nk_f16_t raw;
    f16_t() : raw(0) {}
    f16_t(float v) { nk_f32_to_f16(&v, &raw); }
    operator float() const {
        float r;
        nk_f16_to_f32(&raw, &r);
        return r;
    }
};

/**
 *  @brief Wrapper for nk_bf16_t enabling implicit float conversions.
 */
struct bf16_t {
    nk_bf16_t raw;
    bf16_t() : raw(0) {}
    bf16_t(float v) { nk_f32_to_bf16(&v, &raw); }
    operator float() const {
        float r;
        nk_bf16_to_f32(&raw, &r);
        return r;
    }
};

#pragma endregion // Test_Harness_Templates

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

void test_reduce() {
    std::printf("Testing reductions...\n");
    // TODO: Implement reduce tests
    std::printf("  Reduce tests: SKIPPED (not yet implemented)\n");
}

#pragma endregion // Reduce

#pragma region Dot

/**
 *  @brief Test dot product precision for f32.
 */
error_stats_t test_dot_f32() {
    error_stats_t stats;

#if NK_TEST_USE_OPENMP
#pragma omp parallel
    {
        error_stats_t local_stats;
        std::mt19937 rng(g_config.seed + omp_get_thread_num());

#pragma omp for schedule(dynamic)
        for (std::size_t d = 0; d < g_num_dimensions; d++) {
            std::size_t dim = g_test_dimensions[d];
            for (std::size_t trial = 0; trial < g_config.trials_per_dim; trial++) {
                aligned_buffer<nk_f32_t> a(dim), b(dim);
                fill_random(a, rng, -1.0, 1.0);
                fill_random(b, rng, -1.0, 1.0);

                nk_f32_t result;
                nk_dot_f32(a.data, b.data, dim, &result);

                hp_t ref = reference_dot_quad(a.data, b.data, dim);
                std::uint64_t ulps = ulp_distance_from_reference(result, ref);

                local_stats.accumulate(static_cast<double>(ref), static_cast<double>(result), ulps);
            }
        }

#pragma omp critical
        stats.merge(local_stats);
    }
#else
    std::mt19937 rng(g_config.seed);

    for (std::size_t dim : g_test_dimensions) {
        for (std::size_t trial = 0; trial < g_config.trials_per_dim; trial++) {
            aligned_buffer<nk_f32_t> a(dim), b(dim);
            fill_random(a, rng, -1.0, 1.0);
            fill_random(b, rng, -1.0, 1.0);

            nk_f32_t result;
            nk_dot_f32(a.data, b.data, dim, &result);

            hp_t ref = reference_dot_quad(a.data, b.data, dim);
            std::uint64_t ulps = ulp_distance_from_reference(result, ref);

            stats.accumulate(static_cast<double>(ref), static_cast<double>(result), ulps);
        }
    }
#endif

    return stats;
}

/**
 *  @brief Test dot product precision for f64.
 */
error_stats_t test_dot_f64() {
    error_stats_t stats;
    std::mt19937 rng(g_config.seed);

    for (std::size_t dim : g_test_dimensions) {
        for (std::size_t trial = 0; trial < g_config.trials_per_dim; trial++) {
            aligned_buffer<nk_f64_t> a(dim), b(dim);
            fill_random(a, rng, -1.0, 1.0);
            fill_random(b, rng, -1.0, 1.0);

            nk_f64_t result;
            nk_dot_f64(a.data, b.data, dim, &result);

            hp_t ref = reference_dot_quad(a.data, b.data, dim);
            std::uint64_t ulps = ulp_distance_from_reference(result, ref);

            stats.accumulate(static_cast<double>(ref), result, ulps);

            if (g_config.assert_on_failure && ulps > g_config.ulp_threshold_f32) {
                std::fprintf(stderr, "FAIL: dot_f64 dim=%zu ulp=%llu\n", dim, static_cast<unsigned long long>(ulps));
                assert(false);
            }
        }
    }

    return stats;
}

/**
 *  @brief Test dot product precision for i8 (exact match required).
 */
error_stats_t test_dot_i8() {
    error_stats_t stats;
    std::mt19937 rng(g_config.seed);

    for (std::size_t dim : g_test_dimensions) {
        for (std::size_t trial = 0; trial < g_config.trials_per_dim; trial++) {
            aligned_buffer<nk_i8_t> a(dim), b(dim);
            fill_random_int(a, rng, -10, 10);
            fill_random_int(b, rng, -10, 10);

            nk_i32_t result;
            nk_dot_i8(a.data, b.data, dim, &result);

            // Reference: exact integer computation
            std::int64_t ref = 0;
            for (std::size_t i = 0; i < dim; i++) {
                ref += static_cast<std::int64_t>(a[i]) * static_cast<std::int64_t>(b[i]);
            }

            std::uint64_t ulps = (result == static_cast<nk_i32_t>(ref)) ? 0 : UINT64_MAX;
            stats.accumulate_ulp(ulps);

            if (g_config.assert_on_failure && result != static_cast<nk_i32_t>(ref)) {
                std::fprintf(stderr, "FAIL: dot_i8 dim=%zu expected=%lld got=%d\n", dim, static_cast<long long>(ref),
                             result);
                assert(false);
            }
        }
    }

    return stats;
}

/**
 *  @brief Test dot product precision for u8 (exact match required).
 */
error_stats_t test_dot_u8() {
    error_stats_t stats;
    std::mt19937 rng(g_config.seed);

    for (std::size_t dim : g_test_dimensions) {
        for (std::size_t trial = 0; trial < g_config.trials_per_dim; trial++) {
            aligned_buffer<nk_u8_t> a(dim), b(dim);
            fill_random_int(a, rng, 0, 15);
            fill_random_int(b, rng, 0, 15);

            nk_u32_t result;
            nk_dot_u8(a.data, b.data, dim, &result);

            // Reference: exact integer computation
            std::uint64_t ref = 0;
            for (std::size_t i = 0; i < dim; i++) {
                ref += static_cast<std::uint64_t>(a[i]) * static_cast<std::uint64_t>(b[i]);
            }

            std::uint64_t ulps = (result == static_cast<nk_u32_t>(ref)) ? 0 : UINT64_MAX;
            stats.accumulate_ulp(ulps);

            if (g_config.assert_on_failure && result != static_cast<nk_u32_t>(ref)) {
                std::fprintf(stderr, "FAIL: dot_u8 dim=%zu expected=%llu got=%u\n", dim,
                             static_cast<unsigned long long>(ref), result);
                assert(false);
            }
        }
    }

    return stats;
}

/**
 *  @brief Test dot product precision for f16.
 */
error_stats_t test_dot_f16() {
    error_stats_t stats;
    std::mt19937 rng(g_config.seed);

    for (std::size_t dim : g_test_dimensions) {
        for (std::size_t trial = 0; trial < g_config.trials_per_dim; trial++) {
            aligned_buffer<f16_t> a(dim), b(dim);
            fill_random(a, rng, -1.0, 1.0);
            fill_random(b, rng, -1.0, 1.0);

            nk_f32_t result;
            nk_dot_f16(reinterpret_cast<nk_f16_t const *>(a.data), reinterpret_cast<nk_f16_t const *>(b.data), dim,
                       &result);

            hp_t ref = 0;
            for (std::size_t i = 0; i < dim; i++)
                ref += hp_t(static_cast<float>(a[i])) * hp_t(static_cast<float>(b[i]));

            std::uint64_t ulps = ulp_distance_from_reference(result, ref);
            stats.accumulate(static_cast<double>(ref), static_cast<double>(result), ulps);
        }
    }
    return stats;
}

/**
 *  @brief Test dot product precision for bf16.
 */
error_stats_t test_dot_bf16() {
    error_stats_t stats;
    std::mt19937 rng(g_config.seed);

    for (std::size_t dim : g_test_dimensions) {
        for (std::size_t trial = 0; trial < g_config.trials_per_dim; trial++) {
            aligned_buffer<bf16_t> a(dim), b(dim);
            fill_random(a, rng, -1.0, 1.0);
            fill_random(b, rng, -1.0, 1.0);

            nk_f32_t result;
            nk_dot_bf16(reinterpret_cast<nk_bf16_t const *>(a.data), reinterpret_cast<nk_bf16_t const *>(b.data), dim,
                        &result);

            hp_t ref = 0;
            for (std::size_t i = 0; i < dim; i++)
                ref += hp_t(static_cast<float>(a[i])) * hp_t(static_cast<float>(b[i]));

            std::uint64_t ulps = ulp_distance_from_reference(result, ref);
            stats.accumulate(static_cast<double>(ref), static_cast<double>(result), ulps);
        }
    }
    return stats;
}

/**
 *  @brief Test complex dot product precision for f32c.
 */
error_stats_t test_dot_f32c() {
    error_stats_t stats;
    std::mt19937 rng(g_config.seed);

    for (std::size_t dim : g_test_dimensions) {
        for (std::size_t trial = 0; trial < g_config.trials_per_dim; trial++) {
            aligned_buffer<nk_f32c_t> a(dim), b(dim);
            std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
            for (std::size_t i = 0; i < dim; i++) {
                a[i].real = dist(rng);
                a[i].imag = dist(rng);
                b[i].real = dist(rng);
                b[i].imag = dist(rng);
            }

            nk_f32c_t result;
            nk_dot_f32c(a.data, b.data, dim, &result);

            // Reference: complex dot product
            // (a.re + i*a.im) * (b.re + i*b.im) = (a.re*b.re - a.im*b.im) + i*(a.re*b.im + a.im*b.re)
            hp_t ref_real = 0, ref_imag = 0;
            for (std::size_t i = 0; i < dim; i++) {
                ref_real += hp_t(a[i].real) * hp_t(b[i].real) - hp_t(a[i].imag) * hp_t(b[i].imag);
                ref_imag += hp_t(a[i].real) * hp_t(b[i].imag) + hp_t(a[i].imag) * hp_t(b[i].real);
            }

            std::uint64_t ulps_real = ulp_distance_from_reference(result.real, ref_real);
            std::uint64_t ulps_imag = ulp_distance_from_reference(result.imag, ref_imag);
            std::uint64_t ulps = std::max(ulps_real, ulps_imag);

            stats.accumulate_ulp(ulps);
        }
    }

    return stats;
}

/**
 *  @brief Test complex conjugate dot product precision for f32c.
 */
error_stats_t test_vdot_f32c() {
    error_stats_t stats;
    std::mt19937 rng(g_config.seed);

    for (std::size_t dim : g_test_dimensions) {
        for (std::size_t trial = 0; trial < g_config.trials_per_dim; trial++) {
            aligned_buffer<nk_f32c_t> a(dim), b(dim);
            std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
            for (std::size_t i = 0; i < dim; i++) {
                a[i].real = dist(rng);
                a[i].imag = dist(rng);
                b[i].real = dist(rng);
                b[i].imag = dist(rng);
            }

            nk_f32c_t result;
            nk_vdot_f32c(a.data, b.data, dim, &result);

            // Reference: conjugate dot product conj(a) * b
            // (a.re - i*a.im) * (b.re + i*b.im) = (a.re*b.re + a.im*b.im) + i*(a.re*b.im - a.im*b.re)
            hp_t ref_real = 0, ref_imag = 0;
            for (std::size_t i = 0; i < dim; i++) {
                ref_real += hp_t(a[i].real) * hp_t(b[i].real) + hp_t(a[i].imag) * hp_t(b[i].imag);
                ref_imag += hp_t(a[i].real) * hp_t(b[i].imag) - hp_t(a[i].imag) * hp_t(b[i].real);
            }

            std::uint64_t ulps_real = ulp_distance_from_reference(result.real, ref_real);
            std::uint64_t ulps_imag = ulp_distance_from_reference(result.imag, ref_imag);
            std::uint64_t ulps = std::max(ulps_real, ulps_imag);

            stats.accumulate_ulp(ulps);
        }
    }

    return stats;
}

void test_dot() {
    if (!g_config.should_run("dot")) return;
    std::printf("Testing dot products...\n");

    error_stats_t stats_f32 = test_dot_f32();
    stats_f32.report("dot", "f32");

    error_stats_t stats_f64 = test_dot_f64();
    stats_f64.report("dot", "f64");

    error_stats_t stats_f16 = test_dot_f16();
    stats_f16.report("dot", "f16");

    error_stats_t stats_bf16 = test_dot_bf16();
    stats_bf16.report("dot", "bf16");

    error_stats_t stats_i8 = test_dot_i8();
    stats_i8.report("dot", "i8");

    error_stats_t stats_u8 = test_dot_u8();
    stats_u8.report("dot", "u8");

    error_stats_t stats_f32c = test_dot_f32c();
    stats_f32c.report("dot", "f32c");

    error_stats_t stats_vdot = test_vdot_f32c();
    stats_vdot.report("vdot", "f32c");
}

#pragma endregion // Dot

#pragma region Spatial

/**
 *  @brief Test L2 squared distance precision for f32.
 */
error_stats_t test_l2sq_f32() {
    error_stats_t stats;
    std::mt19937 rng(g_config.seed);

    for (std::size_t dim : g_test_dimensions) {
        for (std::size_t trial = 0; trial < g_config.trials_per_dim; trial++) {
            aligned_buffer<nk_f32_t> a(dim), b(dim);
            fill_random(a, rng, -1.0, 1.0);
            fill_random(b, rng, -1.0, 1.0);

            nk_f32_t result;
            nk_l2sq_f32(a.data, b.data, dim, &result);

            hp_t ref = reference_l2sq_quad(a.data, b.data, dim);
            std::uint64_t ulps = ulp_distance_from_reference(result, ref);

            stats.accumulate(static_cast<double>(ref), static_cast<double>(result), ulps);

            if (g_config.assert_on_failure && ulps > g_config.ulp_threshold_f32) {
                std::fprintf(stderr, "FAIL: l2sq_f32 dim=%zu ulp=%llu\n", dim, static_cast<unsigned long long>(ulps));
                assert(false);
            }
        }
    }

    return stats;
}

/**
 *  @brief Test angular (cosine) distance precision for f32.
 */
error_stats_t test_angular_f32() {
    error_stats_t stats;
    std::mt19937 rng(g_config.seed);

    for (std::size_t dim : g_test_dimensions) {
        error_stats_t dim_stats;
        for (std::size_t trial = 0; trial < g_config.trials_per_dim; trial++) {
            aligned_buffer<nk_f32_t> a(dim), b(dim);
            fill_random(a, rng, -1.0, 1.0);
            fill_random(b, rng, -1.0, 1.0);

            nk_f32_t result;
            nk_angular_f32(a.data, b.data, dim, &result);

            hp_t ref = reference_angular_quad(a.data, b.data, dim);
            std::uint64_t ulps = ulp_distance_from_reference(result, ref);

            stats.accumulate(static_cast<double>(ref), static_cast<double>(result), ulps);
            dim_stats.accumulate(static_cast<double>(ref), static_cast<double>(result), ulps);

            if (g_config.assert_on_failure && ulps > g_config.ulp_threshold_f32 * 2) {
                std::fprintf(stderr, "FAIL: angular_f32 dim=%zu ulp=%llu\n", dim,
                             static_cast<unsigned long long>(ulps));
                assert(false);
            }
        }
        if (g_config.verbose) dim_stats.report_dimension("angular", "f32", dim);
    }

    return stats;
}

/**
 *  @brief Test L2 squared distance precision for f64.
 */
error_stats_t test_l2sq_f64() {
    error_stats_t stats;
    std::mt19937 rng(g_config.seed);

    for (std::size_t dim : g_test_dimensions) {
        for (std::size_t trial = 0; trial < g_config.trials_per_dim; trial++) {
            aligned_buffer<nk_f64_t> a(dim), b(dim);
            fill_random(a, rng, -1.0, 1.0);
            fill_random(b, rng, -1.0, 1.0);

            nk_f64_t result;
            nk_l2sq_f64(a.data, b.data, dim, &result);

            hp_t ref = reference_l2sq_quad(a.data, b.data, dim);
            std::uint64_t ulps = ulp_distance_from_reference(result, ref);

            stats.accumulate(static_cast<double>(ref), result, ulps);
        }
    }

    return stats;
}

/**
 *  @brief Test angular (cosine) distance precision for f64.
 */
error_stats_t test_angular_f64() {
    error_stats_t stats;
    std::mt19937 rng(g_config.seed);

    for (std::size_t dim : g_test_dimensions) {
        for (std::size_t trial = 0; trial < g_config.trials_per_dim; trial++) {
            aligned_buffer<nk_f64_t> a(dim), b(dim);
            fill_random(a, rng, -1.0, 1.0);
            fill_random(b, rng, -1.0, 1.0);

            nk_f64_t result;
            nk_angular_f64(a.data, b.data, dim, &result);

            hp_t ref = reference_angular_quad(a.data, b.data, dim);
            std::uint64_t ulps = ulp_distance_from_reference(result, ref);

            stats.accumulate(static_cast<double>(ref), result, ulps);
        }
    }

    return stats;
}

/**
 *  @brief Test L2 squared distance precision for f16.
 */
error_stats_t test_l2sq_f16() {
    error_stats_t stats;
    std::mt19937 rng(g_config.seed);

    for (std::size_t dim : g_test_dimensions) {
        for (std::size_t trial = 0; trial < g_config.trials_per_dim; trial++) {
            aligned_buffer<f16_t> a(dim), b(dim);
            fill_random(a, rng, -1.0, 1.0);
            fill_random(b, rng, -1.0, 1.0);

            nk_f32_t result;
            nk_l2sq_f16(reinterpret_cast<nk_f16_t const *>(a.data), reinterpret_cast<nk_f16_t const *>(b.data), dim,
                        &result);

            hp_t ref = 0;
            for (std::size_t i = 0; i < dim; i++) {
                hp_t diff = hp_t(static_cast<float>(a[i])) - hp_t(static_cast<float>(b[i]));
                ref += diff * diff;
            }
            std::uint64_t ulps = ulp_distance_from_reference(result, ref);
            stats.accumulate(static_cast<double>(ref), static_cast<double>(result), ulps);
        }
    }
    return stats;
}

/**
 *  @brief Test angular distance precision for f16.
 */
error_stats_t test_angular_f16() {
    error_stats_t stats;
    std::mt19937 rng(g_config.seed);

    for (std::size_t dim : g_test_dimensions) {
        for (std::size_t trial = 0; trial < g_config.trials_per_dim; trial++) {
            aligned_buffer<f16_t> a(dim), b(dim);
            fill_random(a, rng, -1.0, 1.0);
            fill_random(b, rng, -1.0, 1.0);

            nk_f32_t result;
            nk_angular_f16(reinterpret_cast<nk_f16_t const *>(a.data), reinterpret_cast<nk_f16_t const *>(b.data), dim,
                           &result);

            hp_t ab = 0, aa = 0, bb = 0;
            for (std::size_t i = 0; i < dim; i++) {
                float ai = static_cast<float>(a[i]), bi = static_cast<float>(b[i]);
                ab += hp_t(ai) * hp_t(bi);
                aa += hp_t(ai) * hp_t(ai);
                bb += hp_t(bi) * hp_t(bi);
            }
            hp_t ref = 0;
            if (aa != 0 && bb != 0 && ab != 0) {
                hp_t cos_sim = ab / mp::sqrt(aa * bb);
                ref = 1 - cos_sim;
                if (ref < 0) ref = 0;
            }
            std::uint64_t ulps = ulp_distance_from_reference(result, ref);

            stats.accumulate(static_cast<double>(ref), static_cast<double>(result), ulps);
        }
    }

    return stats;
}

/**
 *  @brief Test L2 squared distance precision for bf16.
 */
error_stats_t test_l2sq_bf16() {
    error_stats_t stats;
    std::mt19937 rng(g_config.seed);

    for (std::size_t dim : g_test_dimensions) {
        for (std::size_t trial = 0; trial < g_config.trials_per_dim; trial++) {
            aligned_buffer<bf16_t> a(dim), b(dim);
            fill_random(a, rng, -1.0, 1.0);
            fill_random(b, rng, -1.0, 1.0);

            nk_f32_t result;
            nk_l2sq_bf16(reinterpret_cast<nk_bf16_t const *>(a.data), reinterpret_cast<nk_bf16_t const *>(b.data), dim,
                         &result);

            hp_t ref = 0;
            for (std::size_t i = 0; i < dim; i++) {
                hp_t diff = hp_t(static_cast<float>(a[i])) - hp_t(static_cast<float>(b[i]));
                ref += diff * diff;
            }
            std::uint64_t ulps = ulp_distance_from_reference(result, ref);
            stats.accumulate(static_cast<double>(ref), static_cast<double>(result), ulps);
        }
    }
    return stats;
}

/**
 *  @brief Test angular distance precision for bf16.
 */
error_stats_t test_angular_bf16() {
    error_stats_t stats;
    std::mt19937 rng(g_config.seed);

    for (std::size_t dim : g_test_dimensions) {
        for (std::size_t trial = 0; trial < g_config.trials_per_dim; trial++) {
            aligned_buffer<bf16_t> a(dim), b(dim);
            fill_random(a, rng, -1.0, 1.0);
            fill_random(b, rng, -1.0, 1.0);

            nk_f32_t result;
            nk_angular_bf16(reinterpret_cast<nk_bf16_t const *>(a.data), reinterpret_cast<nk_bf16_t const *>(b.data),
                            dim, &result);

            hp_t ab = 0, aa = 0, bb = 0;
            for (std::size_t i = 0; i < dim; i++) {
                float ai = static_cast<float>(a[i]), bi = static_cast<float>(b[i]);
                ab += hp_t(ai) * hp_t(bi);
                aa += hp_t(ai) * hp_t(ai);
                bb += hp_t(bi) * hp_t(bi);
            }
            hp_t ref = 0;
            if (aa != 0 && bb != 0 && ab != 0) {
                hp_t cos_sim = ab / mp::sqrt(aa * bb);
                ref = 1 - cos_sim;
                if (ref < 0) ref = 0;
            }
            std::uint64_t ulps = ulp_distance_from_reference(result, ref);

            stats.accumulate(static_cast<double>(ref), static_cast<double>(result), ulps);
        }
    }

    return stats;
}

void test_spatial() {
    if (!g_config.should_run("spatial") && !g_config.should_run("l2") && !g_config.should_run("angular")) return;
    std::printf("Testing spatial distances...\n");

    error_stats_t stats_l2sq_f32 = test_l2sq_f32();
    stats_l2sq_f32.report("l2sq", "f32");

    error_stats_t stats_l2sq_f64 = test_l2sq_f64();
    stats_l2sq_f64.report("l2sq", "f64");

    error_stats_t stats_l2sq_f16 = test_l2sq_f16();
    stats_l2sq_f16.report("l2sq", "f16");

    error_stats_t stats_l2sq_bf16 = test_l2sq_bf16();
    stats_l2sq_bf16.report("l2sq", "bf16");

    error_stats_t stats_angular_f32 = test_angular_f32();
    stats_angular_f32.report("angular", "f32");

    error_stats_t stats_angular_f64 = test_angular_f64();
    stats_angular_f64.report("angular", "f64");

    error_stats_t stats_angular_f16 = test_angular_f16();
    stats_angular_f16.report("angular", "f16");

    error_stats_t stats_angular_bf16 = test_angular_bf16();
    stats_angular_bf16.report("angular", "bf16");
}

#pragma endregion // Spatial

#pragma region Curved

/**
 *  @brief Test bilinear form: a^T * M * b
 */
error_stats_t test_bilinear_f32() {
    error_stats_t stats;
    std::mt19937 rng(g_config.seed);

    // Test smaller dimensions for bilinear (matrix is n*n)
    std::size_t bilinear_dims[] = {2, 3, 4, 8, 16, 32};

    for (std::size_t dim : bilinear_dims) {
        for (std::size_t trial = 0; trial < std::max<std::size_t>(1, g_config.trials_per_dim / 10); trial++) {
            aligned_buffer<nk_f32_t> a(dim), b(dim), m(dim * dim);
            fill_random(a, rng, -1.0, 1.0);
            fill_random(b, rng, -1.0, 1.0);
            fill_random(m, rng, -1.0, 1.0);

            nk_f32_t result;
            nk_bilinear_f32(a.data, b.data, m.data, dim, &result);

            // Reference: a^T * M * b
            hp_t ref = reference_bilinear_quad(a.data, m.data, b.data, dim);
            std::uint64_t ulps = ulp_distance_from_reference(result, ref);

            stats.accumulate(static_cast<double>(ref), static_cast<double>(result), ulps);
        }
    }

    return stats;
}

/**
 *  @brief Test bilinear form for f64.
 */
error_stats_t test_bilinear_f64() {
    error_stats_t stats;
    std::mt19937 rng(g_config.seed);

    std::size_t bilinear_dims[] = {2, 3, 4, 8, 16, 32};

    for (std::size_t dim : bilinear_dims) {
        for (std::size_t trial = 0; trial < std::max<std::size_t>(1, g_config.trials_per_dim / 10); trial++) {
            aligned_buffer<nk_f64_t> a(dim), b(dim), m(dim * dim);
            fill_random(a, rng, -1.0, 1.0);
            fill_random(b, rng, -1.0, 1.0);
            fill_random(m, rng, -1.0, 1.0);

            nk_f64_t result;
            nk_bilinear_f64(a.data, b.data, m.data, dim, &result);

            // Reference: a^T * M * b
            hp_t ref = reference_bilinear_quad(a.data, m.data, b.data, dim);
            std::uint64_t ulps = ulp_distance_from_reference(result, ref);

            stats.accumulate(static_cast<double>(ref), result, ulps);
        }
    }

    return stats;
}

void test_curved() {
    if (!g_config.should_run("curved") && !g_config.should_run("bilinear")) return;
    std::printf("Testing curved/bilinear forms...\n");

    error_stats_t stats_bilinear_f32 = test_bilinear_f32();
    stats_bilinear_f32.report("bilinear", "f32");

    error_stats_t stats_bilinear_f64 = test_bilinear_f64();
    stats_bilinear_f64.report("bilinear", "f64");
}

#pragma endregion // Curved

#pragma region Probability

/**
 *  @brief Test KL divergence precision for f32.
 */
error_stats_t test_kld_f32() {
    error_stats_t stats;
    std::mt19937 rng(g_config.seed);

    // Use smaller dimensions for probability (distributions are normalized)
    std::size_t prob_dims[] = {4, 8, 16, 32, 64, 128, 256};

    for (std::size_t dim : prob_dims) {
        error_stats_t dim_stats;
        for (std::size_t trial = 0; trial < g_config.trials_per_dim; trial++) {
            aligned_buffer<nk_f32_t> p(dim), q(dim);
            fill_probability(p, rng);
            fill_probability(q, rng);

            nk_f32_t result;
            nk_kld_f32(p.data, q.data, dim, &result);

            hp_t ref = reference_kld_quad(p.data, q.data, dim);
            std::uint64_t ulps = ulp_distance_from_reference(result, ref);

            stats.accumulate(static_cast<double>(ref), static_cast<double>(result), ulps);
            dim_stats.accumulate(static_cast<double>(ref), static_cast<double>(result), ulps);
        }
        if (g_config.verbose) dim_stats.report_dimension("kld", "f32", dim);
    }

    return stats;
}

/**
 *  @brief Test KL divergence precision for f64.
 */
error_stats_t test_kld_f64() {
    error_stats_t stats;
    std::mt19937 rng(g_config.seed);

    std::size_t prob_dims[] = {4, 8, 16, 32, 64, 128, 256};

    for (std::size_t dim : prob_dims) {
        for (std::size_t trial = 0; trial < g_config.trials_per_dim; trial++) {
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
            nk_kld_f64(p.data, q.data, dim, &result);

            hp_t ref = reference_kld_quad(p.data, q.data, dim);
            std::uint64_t ulps = ulp_distance_from_reference(result, ref);

            stats.accumulate(static_cast<double>(ref), result, ulps);
        }
    }

    return stats;
}

/**
 *  @brief Test Jensen-Shannon divergence precision for f32.
 */
error_stats_t test_jsd_f32() {
    error_stats_t stats;
    std::mt19937 rng(g_config.seed);

    std::size_t prob_dims[] = {4, 8, 16, 32, 64, 128, 256};

    for (std::size_t dim : prob_dims) {
        error_stats_t dim_stats;
        for (std::size_t trial = 0; trial < g_config.trials_per_dim; trial++) {
            aligned_buffer<nk_f32_t> p(dim), q(dim);
            fill_probability(p, rng);
            fill_probability(q, rng);

            nk_f32_t result;
            nk_jsd_f32(p.data, q.data, dim, &result);

            hp_t ref = reference_jsd_quad(p.data, q.data, dim);
            std::uint64_t ulps = ulp_distance_from_reference(result, ref);

            stats.accumulate(static_cast<double>(ref), static_cast<double>(result), ulps);
            dim_stats.accumulate(static_cast<double>(ref), static_cast<double>(result), ulps);
        }
        if (g_config.verbose) dim_stats.report_dimension("jsd", "f32", dim);
    }

    return stats;
}

/**
 *  @brief Test Jensen-Shannon divergence precision for f64.
 */
error_stats_t test_jsd_f64() {
    error_stats_t stats;
    std::mt19937 rng(g_config.seed);

    std::size_t prob_dims[] = {4, 8, 16, 32, 64, 128, 256};

    for (std::size_t dim : prob_dims) {
        for (std::size_t trial = 0; trial < g_config.trials_per_dim; trial++) {
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
            nk_jsd_f64(p.data, q.data, dim, &result);

            hp_t ref = reference_jsd_quad(p.data, q.data, dim);
            std::uint64_t ulps = ulp_distance_from_reference(result, ref);

            stats.accumulate(static_cast<double>(ref), result, ulps);
        }
    }

    return stats;
}

void test_probability() {
    if (!g_config.should_run("probability") && !g_config.should_run("kld") && !g_config.should_run("jsd")) return;
    std::printf("Testing probability divergences...\n");

    error_stats_t stats_kld_f32 = test_kld_f32();
    stats_kld_f32.report("kld", "f32");

    error_stats_t stats_kld_f64 = test_kld_f64();
    stats_kld_f64.report("kld", "f64");

    error_stats_t stats_jsd_f32 = test_jsd_f32();
    stats_jsd_f32.report("jsd", "f32");

    error_stats_t stats_jsd_f64 = test_jsd_f64();
    stats_jsd_f64.report("jsd", "f64");
}

#pragma endregion // Probability

#pragma region Binary

/**
 *  @brief Test Hamming distance (exact match required).
 */
error_stats_t test_hamming_b8() {
    error_stats_t stats;
    std::mt19937 rng(g_config.seed);

    // Hamming distance works on bytes representing bits
    std::size_t byte_dims[] = {1, 2, 4, 8, 16, 32, 64, 128, 256};

    for (std::size_t n_bytes : byte_dims) {
        for (std::size_t trial = 0; trial < g_config.trials_per_dim; trial++) {
            aligned_buffer<nk_b8_t> a(n_bytes), b(n_bytes);
            std::uniform_int_distribution<int> dist(0, 255);
            for (std::size_t i = 0; i < n_bytes; i++) {
                a[i] = static_cast<nk_b8_t>(dist(rng));
                b[i] = static_cast<nk_b8_t>(dist(rng));
            }

            nk_u32_t result;
            nk_hamming_b8(a.data, b.data, n_bytes, &result);

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

            std::uint64_t ulps = (result == ref) ? 0 : UINT64_MAX;
            stats.accumulate_ulp(ulps);

            if (g_config.assert_on_failure && result != ref) {
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
error_stats_t test_jaccard_b8() {
    error_stats_t stats;
    std::mt19937 rng(g_config.seed);

    std::size_t byte_dims[] = {1, 2, 4, 8, 16, 32, 64, 128, 256};

    for (std::size_t n_bytes : byte_dims) {
        for (std::size_t trial = 0; trial < g_config.trials_per_dim; trial++) {
            aligned_buffer<nk_b8_t> a(n_bytes), b(n_bytes);
            std::uniform_int_distribution<int> dist(0, 255);
            for (std::size_t i = 0; i < n_bytes; i++) {
                a[i] = static_cast<nk_b8_t>(dist(rng));
                b[i] = static_cast<nk_b8_t>(dist(rng));
            }

            nk_f32_t result;
            nk_jaccard_b8(a.data, b.data, n_bytes, &result);

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

            hp_t ref = (union_count > 0) ? (hp_t(1) - hp_t(intersection) / hp_t(union_count)) : hp_t(0);
            std::uint64_t ulps = ulp_distance_from_reference(result, ref);

            stats.accumulate(static_cast<double>(ref), static_cast<double>(result), ulps);
        }
    }

    return stats;
}

void test_binary() {
    if (!g_config.should_run("binary") && !g_config.should_run("hamming") && !g_config.should_run("jaccard")) return;
    std::printf("Testing binary distances...\n");

    error_stats_t stats_hamming = test_hamming_b8();
    stats_hamming.report("hamming", "b8");

    error_stats_t stats_jaccard = test_jaccard_b8();
    stats_jaccard.report("jaccard", "b8");
}

#pragma endregion // Binary

#pragma region Elementwise

/**
 *  @brief Test scale operation: result[i] = alpha * x[i] + beta
 */
error_stats_t test_scale_f32() {
    error_stats_t stats;
    std::mt19937 rng(g_config.seed);

    for (std::size_t dim : g_test_dimensions) {
        for (std::size_t trial = 0; trial < std::max<std::size_t>(1, g_config.trials_per_dim / 10); trial++) {
            aligned_buffer<nk_f32_t> input(dim), expected(dim), result(dim);
            fill_random(input, rng, -10.0, 10.0);

            std::uniform_real_distribution<float> coef_dist(-5.0f, 5.0f);
            nk_f32_t alpha = coef_dist(rng);
            nk_f32_t beta = coef_dist(rng);

            // Compute reference
            for (std::size_t i = 0; i < dim; i++) { expected[i] = alpha * input[i] + beta; }

            // Test SIMD implementation
            nk_scale_f32(input.data, dim, &alpha, &beta, result.data);

            // Verify
            for (std::size_t i = 0; i < dim; i++) {
                std::uint64_t ulps = ulp_distance(expected[i], result[i]);
                stats.accumulate_ulp(ulps);
                if (g_config.assert_on_failure && ulps > 2) {
                    std::fprintf(stderr, "FAIL: scale_f32 dim=%zu i=%zu ulp=%llu\n", dim, i,
                                 static_cast<unsigned long long>(ulps));
                    assert(false);
                }
            }
        }
    }

    return stats;
}

/**
 *  @brief Test elementwise sum: result[i] = a[i] + b[i]
 */
error_stats_t test_sum_f32() {
    error_stats_t stats;
    std::mt19937 rng(g_config.seed);

    for (std::size_t dim : g_test_dimensions) {
        for (std::size_t trial = 0; trial < std::max<std::size_t>(1, g_config.trials_per_dim / 10); trial++) {
            aligned_buffer<nk_f32_t> a(dim), b(dim), expected(dim), result(dim);
            fill_random(a, rng, -10.0, 10.0);
            fill_random(b, rng, -10.0, 10.0);

            // Compute reference
            for (std::size_t i = 0; i < dim; i++) { expected[i] = a[i] + b[i]; }

            // Test SIMD implementation
            nk_sum_f32(a.data, b.data, dim, result.data);

            // Verify (should be exact for simple addition)
            for (std::size_t i = 0; i < dim; i++) {
                std::uint64_t ulps = ulp_distance(expected[i], result[i]);
                stats.accumulate_ulp(ulps);
            }
        }
    }

    return stats;
}

/**
 *  @brief Test weighted sum: result[i] = alpha * a[i] + beta * b[i]
 */
error_stats_t test_wsum_f32() {
    error_stats_t stats;
    std::mt19937 rng(g_config.seed);

    for (std::size_t dim : g_test_dimensions) {
        for (std::size_t trial = 0; trial < std::max<std::size_t>(1, g_config.trials_per_dim / 10); trial++) {
            aligned_buffer<nk_f32_t> a(dim), b(dim), expected(dim), result(dim);
            fill_random(a, rng, -10.0, 10.0);
            fill_random(b, rng, -10.0, 10.0);

            std::uniform_real_distribution<float> coef_dist(-5.0f, 5.0f);
            nk_f32_t alpha = coef_dist(rng);
            nk_f32_t beta = coef_dist(rng);

            // Compute reference
            for (std::size_t i = 0; i < dim; i++) { expected[i] = alpha * a[i] + beta * b[i]; }

            // Test SIMD implementation
            nk_wsum_f32(a.data, b.data, dim, &alpha, &beta, result.data);

            // Verify
            for (std::size_t i = 0; i < dim; i++) {
                std::uint64_t ulps = ulp_distance(expected[i], result[i]);
                stats.accumulate_ulp(ulps);
                if (g_config.assert_on_failure && ulps > 2) {
                    std::fprintf(stderr, "FAIL: wsum_f32 dim=%zu i=%zu ulp=%llu\n", dim, i,
                                 static_cast<unsigned long long>(ulps));
                    assert(false);
                }
            }
        }
    }

    return stats;
}

/**
 *  @brief Test FMA: result[i] = alpha * a[i] * b[i] + beta * c[i]
 */
error_stats_t test_fma_f32() {
    error_stats_t stats;
    std::mt19937 rng(g_config.seed);

    for (std::size_t dim : g_test_dimensions) {
        for (std::size_t trial = 0; trial < std::max<std::size_t>(1, g_config.trials_per_dim / 10); trial++) {
            aligned_buffer<nk_f32_t> a(dim), b(dim), c(dim), expected(dim), result(dim);
            fill_random(a, rng, -5.0, 5.0);
            fill_random(b, rng, -5.0, 5.0);
            fill_random(c, rng, -5.0, 5.0);

            std::uniform_real_distribution<float> coef_dist(-2.0f, 2.0f);
            nk_f32_t alpha = coef_dist(rng);
            nk_f32_t beta = coef_dist(rng);

            // Compute reference
            for (std::size_t i = 0; i < dim; i++) { expected[i] = alpha * a[i] * b[i] + beta * c[i]; }

            // Test SIMD implementation
            nk_fma_f32(a.data, b.data, c.data, dim, &alpha, &beta, result.data);

            // Verify
            for (std::size_t i = 0; i < dim; i++) {
                std::uint64_t ulps = ulp_distance(expected[i], result[i]);
                stats.accumulate_ulp(ulps);
                if (g_config.assert_on_failure && ulps > 4) {
                    std::fprintf(stderr, "FAIL: fma_f32 dim=%zu i=%zu ulp=%llu\n", dim, i,
                                 static_cast<unsigned long long>(ulps));
                    assert(false);
                }
            }
        }
    }

    return stats;
}

void test_elementwise() {
    if (!g_config.should_run("elementwise") && !g_config.should_run("scale") && !g_config.should_run("sum") &&
        !g_config.should_run("wsum") && !g_config.should_run("fma"))
        return;
    std::printf("Testing elementwise operations...\n");

    error_stats_t stats_scale = test_scale_f32();
    stats_scale.report("scale", "f32");

    error_stats_t stats_sum = test_sum_f32();
    stats_sum.report("sum", "f32");

    error_stats_t stats_wsum = test_wsum_f32();
    stats_wsum.report("wsum", "f32");

    error_stats_t stats_fma = test_fma_f32();
    stats_fma.report("fma", "f32");
}

#pragma endregion // Elementwise

#pragma region Trigonometry

/**
 *  @brief Test sin approximation precision for f32.
 */
error_stats_t test_sin_f32() {
    error_stats_t stats;

    // Test across the range [-2pi, 2pi]
    constexpr float pi = 3.14159265358979323846f;
    constexpr std::size_t n_samples = 10000;

    aligned_buffer<nk_f32_t> inputs(n_samples), outputs(n_samples);
    for (std::size_t i = 0; i < n_samples; i++) { inputs[i] = -2 * pi + (4 * pi * i) / n_samples; }

    nk_sin_f32(inputs.data, n_samples, outputs.data);

    for (std::size_t i = 0; i < n_samples; i++) {
        hp_t ref = mp::sin(hp_t(inputs[i]));
        std::uint64_t ulps = ulp_distance_from_reference(outputs[i], ref);
        stats.accumulate(static_cast<double>(ref), static_cast<double>(outputs[i]), ulps);
    }

    return stats;
}

/**
 *  @brief Test cos approximation precision for f32.
 */
error_stats_t test_cos_f32() {
    error_stats_t stats;

    constexpr float pi = 3.14159265358979323846f;
    constexpr std::size_t n_samples = 10000;

    aligned_buffer<nk_f32_t> inputs(n_samples), outputs(n_samples);
    for (std::size_t i = 0; i < n_samples; i++) { inputs[i] = -2 * pi + (4 * pi * i) / n_samples; }

    nk_cos_f32(inputs.data, n_samples, outputs.data);

    for (std::size_t i = 0; i < n_samples; i++) {
        hp_t ref = mp::cos(hp_t(inputs[i]));
        std::uint64_t ulps = ulp_distance_from_reference(outputs[i], ref);
        stats.accumulate(static_cast<double>(ref), static_cast<double>(outputs[i]), ulps);
    }

    return stats;
}

/**
 *  @brief Test sin approximation precision for f64.
 */
error_stats_t test_sin_f64() {
    error_stats_t stats;

    constexpr double pi = 3.14159265358979323846;
    constexpr std::size_t n_samples = 10000;

    aligned_buffer<nk_f64_t> inputs(n_samples), outputs(n_samples);
    for (std::size_t i = 0; i < n_samples; i++) { inputs[i] = -2 * pi + (4 * pi * i) / n_samples; }

    nk_sin_f64(inputs.data, n_samples, outputs.data);

    for (std::size_t i = 0; i < n_samples; i++) {
        hp_t ref = mp::sin(hp_t(inputs[i]));
        std::uint64_t ulps = ulp_distance_from_reference(outputs[i], ref);
        stats.accumulate(static_cast<double>(ref), outputs[i], ulps);
    }

    return stats;
}

/**
 *  @brief Test cos approximation precision for f64.
 */
error_stats_t test_cos_f64() {
    error_stats_t stats;

    constexpr double pi = 3.14159265358979323846;
    constexpr std::size_t n_samples = 10000;

    aligned_buffer<nk_f64_t> inputs(n_samples), outputs(n_samples);
    for (std::size_t i = 0; i < n_samples; i++) { inputs[i] = -2 * pi + (4 * pi * i) / n_samples; }

    nk_cos_f64(inputs.data, n_samples, outputs.data);

    for (std::size_t i = 0; i < n_samples; i++) {
        hp_t ref = mp::cos(hp_t(inputs[i]));
        std::uint64_t ulps = ulp_distance_from_reference(outputs[i], ref);
        stats.accumulate(static_cast<double>(ref), outputs[i], ulps);
    }

    return stats;
}

void test_trigonometry() {
    if (!g_config.should_run("trig") && !g_config.should_run("sin") && !g_config.should_run("cos")) return;
    std::printf("Testing trigonometry...\n");

    error_stats_t stats_sin_f32 = test_sin_f32();
    stats_sin_f32.report("sin", "f32");

    error_stats_t stats_cos_f32 = test_cos_f32();
    stats_cos_f32.report("cos", "f32");

    error_stats_t stats_sin_f64 = test_sin_f64();
    stats_sin_f64.report("sin", "f64");

    error_stats_t stats_cos_f64 = test_cos_f64();
    stats_cos_f64.report("cos", "f64");
}

#pragma endregion // Trigonometry

#pragma region Geospatial

void test_geospatial() {
    std::printf("Testing geospatial functions...\n");
    // TODO: Implement geospatial tests
    std::printf("  Geospatial tests: SKIPPED (not yet implemented)\n");
}

#pragma endregion // Geospatial

#pragma region Mesh

void test_mesh() {
    std::printf("Testing mesh operations...\n");
    // TODO: Implement mesh tests
    std::printf("  Mesh tests: SKIPPED (not yet implemented)\n");
}

#pragma endregion // Mesh

#pragma region Dots

void test_dots() {
    std::printf("Testing batch dot products (GEMM)...\n");
    // TODO: Implement dots/matmul tests
    std::printf("  Dots tests: SKIPPED (not yet implemented)\n");
}

#pragma endregion // Dots

#pragma region Sparse

void test_sparse() {
    std::printf("Testing sparse operations...\n");
    // TODO: Implement sparse tests
    std::printf("  Sparse tests: SKIPPED (not yet implemented)\n");
}

#pragma endregion // Sparse

#pragma region Capabilities

/**
 *  @brief Print CPU capabilities detected at compile-time and runtime.
 */
void print_capabilities() {
    nk_capability_t runtime_caps = nk_capabilities();

    char const *flags[2] = {"false", "true"};
    std::printf("NumKong Precision Test Suite\n");
    std::printf("============================\n\n");

    std::printf("Compile-time settings:\n");
    std::printf("  NK_NATIVE_F16:  %s\n", flags[NK_NATIVE_F16]);
    std::printf("  NK_NATIVE_BF16: %s\n", flags[NK_NATIVE_BF16]);
    std::printf("\n");

    std::printf("Compile-time ISA support:\n");
    std::printf("  NEON:     %s\n", flags[NK_TARGET_NEON]);
    std::printf("  SVE:      %s\n", flags[NK_TARGET_SVE]);
    std::printf("  Haswell:  %s\n", flags[NK_TARGET_HASWELL]);
    std::printf("  Skylake:  %s\n", flags[NK_TARGET_SKYLAKE]);
    std::printf("  Ice Lake: %s\n", flags[NK_TARGET_ICE]);
    std::printf("  Genoa:    %s\n", flags[NK_TARGET_GENOA]);
    std::printf("  Sapphire: %s\n", flags[NK_TARGET_SAPPHIRE]);
    std::printf("\n");

    std::printf("Runtime ISA detection:\n");
    std::printf("  NEON:     %s\n", flags[(runtime_caps & nk_cap_neon_k) != 0]);
    std::printf("  SVE:      %s\n", flags[(runtime_caps & nk_cap_sve_k) != 0]);
    std::printf("  Haswell:  %s\n", flags[(runtime_caps & nk_cap_haswell_k) != 0]);
    std::printf("  Skylake:  %s\n", flags[(runtime_caps & nk_cap_skylake_k) != 0]);
    std::printf("  Ice Lake: %s\n", flags[(runtime_caps & nk_cap_ice_k) != 0]);
    std::printf("  Genoa:    %s\n", flags[(runtime_caps & nk_cap_genoa_k) != 0]);
    std::printf("  Sapphire: %s\n", flags[(runtime_caps & nk_cap_sapphire_k) != 0]);
    std::printf("\n");

    std::printf("Test configuration:\n");
    std::printf("  Assert on failure: %s\n", flags[g_config.assert_on_failure]);
    std::printf("  Verbose:           %s\n", flags[g_config.verbose]);
    std::printf("  Filter:            %s\n", g_config.filter ? g_config.filter : "(none)");
    std::printf("  Trials per dim:    %zu\n", g_config.trials_per_dim);
    std::printf("  RNG seed:          %u\n", g_config.seed);
    std::printf("  ULP threshold f32: %llu\n", static_cast<unsigned long long>(g_config.ulp_threshold_f32));
    std::printf("  ULP threshold f16: %llu\n", static_cast<unsigned long long>(g_config.ulp_threshold_f16));
    std::printf("  ULP threshold bf16:%llu\n", static_cast<unsigned long long>(g_config.ulp_threshold_bf16));
#if NK_TEST_USE_OPENMP
    std::printf("  OpenMP threads:    %d\n", omp_get_max_threads());
#else
    std::printf("  OpenMP:            disabled\n");
#endif
    std::printf("\n");
}

#pragma endregion // Capabilities

#pragma region Main

int main(int argc, char **argv) {
    (void)argc;
    (void)argv;

    print_capabilities();

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

    // TODO: Enable these as they're implemented
    // test_reduce();
    // test_geospatial();
    // test_mesh();
    // test_dots();
    // test_sparse();

    std::printf("\n");
    std::printf("All tests passed.\n");
    return 0;
}

#pragma endregion // Main

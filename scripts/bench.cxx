/**
 *  SimSIMD C++ Benchmark Suite
 *
 *  Comprehensive benchmarks comparing SimSIMD SIMD-optimized functions against
 *  baseline implementations using Google Benchmark framework. Run with:
 *
 *  ```bash
 *  cmake -B build_release -D SIMSIMD_BUILD_BENCHMARKS=1
 *  cmake --build build_release
 *  build_release/simsimd_bench
 *  ```
 */

#include <array>         // `std::array`
#include <cmath>         // `std::sqrt`
#include <cstdlib>       // `std::aligned_alloc`
#include <cstring>       // `std::memcpy`
#include <numeric>       // `std::accumulate`
#include <random>        // `std::uniform_int_distribution`
#include <thread>        // `std::thread`
#include <tuple>         // `std::tuple` for callable introspection
#include <type_traits>   // `std::numeric_limits`
#include <unordered_set> // `std::unordered_set`
#include <vector>        // `std::vector`

#include <benchmark/benchmark.h>

#if !defined(SIMSIMD_BUILD_BENCHMARKS_WITH_MKL)
#define SIMSIMD_BUILD_BENCHMARKS_WITH_MKL 0
#endif
#if !defined(SIMSIMD_BUILD_BENCHMARKS_WITH_CBLAS)
#define SIMSIMD_BUILD_BENCHMARKS_WITH_CBLAS 0
#endif

// Include BLAS headers - MKL takes precedence if both are enabled
// (MKL provides a superset of CBLAS functionality)
#if SIMSIMD_BUILD_BENCHMARKS_WITH_MKL
#include <mkl.h>
// MKL provides additional GEMM routines:
// - cblas_gemm_bf16bf16f32: BF16 inputs → F32 output
// - cblas_hgemm: F16 GEMM (if available)
#elif SIMSIMD_BUILD_BENCHMARKS_WITH_CBLAS
#include <cblas.h>
// OpenBLAS thread control (weak symbol to avoid link errors if not present)
extern "C" void openblas_set_num_threads(int) __attribute__((weak));
#endif

// Unified BLAS availability check (MKL provides CBLAS interface)
#if SIMSIMD_BUILD_BENCHMARKS_WITH_MKL || SIMSIMD_BUILD_BENCHMARKS_WITH_CBLAS
#define SIMSIMD_BUILD_BENCHMARKS_WITH_BLAS 1
#else
#define SIMSIMD_BUILD_BENCHMARKS_WITH_BLAS 0
#endif

// It's important to note, that out compression/decompression routines
// are quite inaccurate. They are not meant to be used in production code.
// So when benchmarking, if possible, please use the native types, if those
// are implemented.
#define SIMSIMD_NATIVE_F16  1
#define SIMSIMD_NATIVE_BF16 1
#include <simsimd/simsimd.h>

constexpr std::size_t default_seconds = 10;
constexpr std::size_t default_threads = 1;
constexpr simsimd_distance_t signaling_distance = std::numeric_limits<simsimd_distance_t>::signaling_NaN();

/// For sub-byte data types
/// Can be overridden at runtime via `SIMSIMD_BENCH_DENSE_DIMENSIONS` environment variable
std::size_t dense_dimensions = 1536;
/// Has quadratic impact on the number of operations
/// Can be overridden at runtime via `SIMSIMD_BENCH_CURVED_DIMENSIONS` environment variable
std::size_t curved_dimensions = 8;
/// Number of 3D points for mesh metrics (RMSD, Kabsch)
/// Can be overridden at runtime via `SIMSIMD_BENCH_MESH_DIMENSIONS` environment variable
std::size_t mesh_dimensions = 1000;
/// Matrix multiplication benchmark globals
/// Can be overridden at runtime via `SIMSIMD_BENCH_MATMUL_DIMENSION_M/N/K` environment variables
std::size_t matmul_dimension_m = 128, matmul_dimension_n = 512, matmul_dimension_k = 256;
/// Random seed for reproducible benchmarks
/// Can be overridden at runtime via `SIMSIMD_BENCH_RANDOM_SEED` environment variable
std::uint32_t random_seed = 42;

namespace bm = benchmark;

/// Returns a new random engine seeded with the global random_seed.
inline std::mt19937 make_random_engine() { return std::mt19937(random_seed); }

// clang-format off
template <simsimd_datatype_t> struct datatype_enum_to_type_gt { using value_t = void; using scalar_t = void; static constexpr std::size_t components_k = 1; };
template <> struct datatype_enum_to_type_gt<simsimd_f64_k> { using value_t = simsimd_f64_t; using scalar_t = simsimd_f64_t; static constexpr std::size_t components_k = 1; };
template <> struct datatype_enum_to_type_gt<simsimd_f32_k> { using value_t = simsimd_f32_t; using scalar_t = simsimd_f32_t; static constexpr std::size_t components_k = 1; };
template <> struct datatype_enum_to_type_gt<simsimd_f16_k> { using value_t = simsimd_f16_t; using scalar_t = simsimd_f16_t; static constexpr std::size_t components_k = 1; };
template <> struct datatype_enum_to_type_gt<simsimd_bf16_k> { using value_t = simsimd_bf16_t; using scalar_t = simsimd_bf16_t; static constexpr std::size_t components_k = 1; };
template <> struct datatype_enum_to_type_gt<simsimd_e4m3_k> { using value_t = simsimd_e4m3_t; using scalar_t = simsimd_e4m3_t; static constexpr std::size_t components_k = 1; };
template <> struct datatype_enum_to_type_gt<simsimd_e5m2_k> { using value_t = simsimd_e5m2_t; using scalar_t = simsimd_e5m2_t; static constexpr std::size_t components_k = 1; };
template <> struct datatype_enum_to_type_gt<simsimd_f64c_k> { using value_t = simsimd_f64c_t; using scalar_t = simsimd_f64_t; static constexpr std::size_t components_k = 2; };
template <> struct datatype_enum_to_type_gt<simsimd_f32c_k> { using value_t = simsimd_f32c_t; using scalar_t = simsimd_f32_t; static constexpr std::size_t components_k = 2; };
template <> struct datatype_enum_to_type_gt<simsimd_f16c_k> { using value_t = simsimd_f16c_t; using scalar_t = simsimd_f16_t; static constexpr std::size_t components_k = 2; };
template <> struct datatype_enum_to_type_gt<simsimd_bf16c_k> { using value_t = simsimd_bf16c_t; using scalar_t = simsimd_bf16_t; static constexpr std::size_t components_k = 2; };
template <> struct datatype_enum_to_type_gt<simsimd_b8_k> { using value_t = simsimd_b8_t; using scalar_t = simsimd_b8_t; static constexpr std::size_t components_k = 1; };
template <> struct datatype_enum_to_type_gt<simsimd_i8_k> { using value_t = simsimd_i8_t; using scalar_t = simsimd_i8_t; static constexpr std::size_t components_k = 1; };
template <> struct datatype_enum_to_type_gt<simsimd_u8_k> { using value_t = simsimd_u8_t; using scalar_t = simsimd_u8_t; static constexpr std::size_t components_k = 1; };
template <> struct datatype_enum_to_type_gt<simsimd_i16_k> { using value_t = simsimd_i16_t; using scalar_t = simsimd_i16_t; static constexpr std::size_t components_k = 1; };
template <> struct datatype_enum_to_type_gt<simsimd_u16_k> { using value_t = simsimd_u16_t; using scalar_t = simsimd_u16_t; static constexpr std::size_t components_k = 1; };
template <> struct datatype_enum_to_type_gt<simsimd_i32_k> { using value_t = simsimd_i32_t; using scalar_t = simsimd_i32_t; static constexpr std::size_t components_k = 1; };
template <> struct datatype_enum_to_type_gt<simsimd_u32_k> { using value_t = simsimd_u32_t; using scalar_t = simsimd_u32_t; static constexpr std::size_t components_k = 1; };
template <> struct datatype_enum_to_type_gt<simsimd_i64_k> { using value_t = simsimd_i64_t; using scalar_t = simsimd_i64_t; static constexpr std::size_t components_k = 1; };
template <> struct datatype_enum_to_type_gt<simsimd_u64_k> { using value_t = simsimd_u64_t; using scalar_t = simsimd_u64_t; static constexpr std::size_t components_k = 1; };
// clang-format on

template <std::size_t multiple>
constexpr std::size_t divide_round_up(std::size_t n) {
    return ((n + multiple - 1) / multiple) * multiple;
}

/**
 *  @brief Vector-like fixed capacity buffer, ensuring cache-line alignment.
 *  @tparam datatype_ak The data type of the vector elements, represented as a `simsimd_datatype_t`.
 */
template <simsimd_datatype_t datatype_ak>
struct vector_gt {
    using datatype_reflection_t = datatype_enum_to_type_gt<datatype_ak>;
    using scalar_t = typename datatype_reflection_t::scalar_t;
    using value_t = typename datatype_reflection_t::value_t;
    static constexpr std::size_t components_k = datatype_reflection_t::components_k;

    static constexpr bool is_integral =                                 //
        datatype_ak == simsimd_b8_k ||                                  //
        datatype_ak == simsimd_i8_k || datatype_ak == simsimd_u8_k ||   //
        datatype_ak == simsimd_i16_k || datatype_ak == simsimd_u16_k || //
        datatype_ak == simsimd_i32_k || datatype_ak == simsimd_u32_k || datatype_ak == simsimd_i64_k ||
        datatype_ak == simsimd_u64_k;
    static constexpr std::size_t cacheline_length = 64;

    value_t *values_ptr_ = nullptr;
    std::size_t values_count_ = 0;

    vector_gt() = default;
    vector_gt(std::size_t values_count) noexcept(false)
        : values_count_(values_count),
          values_ptr_(static_cast<value_t *>(std::aligned_alloc(
              cacheline_length, divide_round_up<cacheline_length>(values_count * sizeof(value_t))))) {
        if (!values_ptr_) throw std::bad_alloc();
    }

    ~vector_gt() noexcept { std::free(values_ptr_); }

    vector_gt(vector_gt const &other) : vector_gt(other.size()) {
        std::memcpy(values_ptr_, other.values_ptr_, divide_round_up<cacheline_length>(values_count_ * sizeof(value_t)));
    }
    vector_gt &operator=(vector_gt const &other) {
        if (this != &other) {
            if (values_count_ != other.size()) {
                std::free(values_ptr_);
                values_count_ = other.size();
                values_ptr_ = static_cast<value_t *>(std::aligned_alloc(
                    cacheline_length, divide_round_up<cacheline_length>(values_count_ * sizeof(value_t))));
                if (!values_ptr_) throw std::bad_alloc();
            }
            std::memcpy(values_ptr_, other.values_ptr_,
                        divide_round_up<cacheline_length>(values_count_ * sizeof(value_t)));
        }
        return *this;
    }

    value_t *data() noexcept { return values_ptr_; }
    value_t const *data() const noexcept { return values_ptr_; }
    std::size_t size() const noexcept { return values_count_; }
    std::size_t size_bytes() const noexcept {
        return divide_round_up<cacheline_length>(values_count_ * sizeof(value_t));
    }

    scalar_t *data_scalars() noexcept { return reinterpret_cast<scalar_t *>(data()); }
    scalar_t const *data_scalars() const noexcept { return reinterpret_cast<scalar_t *>(data()); }
    std::size_t size_scalars() const noexcept { return size() * components_k; }

    /**
     *  @brief Broadcast a scalar value to all elements of the vector.
     *  @param v The scalar value to broadcast.
     */
    void set(scalar_t v) noexcept {
        for (std::size_t i = 0; i != size_scalars(); ++i) data_scalars()[i] = v;
    }

    /**
     *  @brief Compresses a double value into the vector's scalar type.
     *  @param from The double value to compress.
     *  @param to The scalar type where the compressed value will be stored.
     */
    static void compress(double const &from, scalar_t &to) noexcept {
        // In a NaN, the sign bit is irrelevant, mantissa describes the kind of NaN,
        // and the exponent is all ones - we can only check the the exponent bits.
        // Brain float is similar: https://en.wikipedia.org/wiki/Bfloat16_floating-point_format
        constexpr unsigned short exponent_mask_f16 = 0b0111110000000000;               // 1 sign, 5 exp, 10 mantissa
        constexpr unsigned short exponent_mask_bf16 = 0b0111111110000000;              // 1 sign, 8 exp, 7 mantissa
        constexpr unsigned int exponent_mask_f32 = 0b01111111100000000000000000000000; // 1 sign, 8 exp, 23 mantissa
        constexpr unsigned long long exponent_mask_f64 =                               // 1 sign, 11 exp, 52 mantissa
            0b011111111110000000000000000000000000000000000000000000000000000;

#if !SIMSIMD_NATIVE_BF16
        if constexpr (datatype_ak == simsimd_bf16_k || datatype_ak == simsimd_bf16c_k) {
            simsimd_f32_t f32 = static_cast<simsimd_f32_t>(from);
            simsimd_f32_to_bf16(&f32, &to);
            if ((to & exponent_mask_bf16) == exponent_mask_bf16) to = 0;
            static_assert(sizeof(scalar_t) == sizeof(simsimd_bf16_t));
            return;
        }
#endif
#if !SIMSIMD_NATIVE_F16
        if constexpr (datatype_ak == simsimd_f16_k || datatype_ak == simsimd_f16c_k) {
            simsimd_f32_t f32 = static_cast<simsimd_f32_t>(from);
            simsimd_f32_to_f16(&f32, &to);
            if ((to & exponent_mask_f16) == exponent_mask_f16) to = 0;
            static_assert(sizeof(scalar_t) == sizeof(simsimd_f16_t));
            return;
        }
#endif
        to = static_cast<scalar_t>(from);
    }

    /**
     *  @brief Decompresses the vector's scalar type into a double value.
     *  @param from The compressed scalar value to decompress.
     *  @return The decompressed double value.
     */
    static double uncompress(scalar_t const &from) noexcept {
#if !SIMSIMD_NATIVE_BF16
        if constexpr (datatype_ak == simsimd_bf16_k || datatype_ak == simsimd_bf16c_k) {
            simsimd_f32_t f32;
            simsimd_bf16_to_f32((simsimd_bf16_t const *)&from, &f32);
            return f32;
        }
#endif
#if !SIMSIMD_NATIVE_F16
        if constexpr (datatype_ak == simsimd_f16_k || datatype_ak == simsimd_f16c_k) {
            simsimd_f32_t f32;
            simsimd_f16_to_f32((simsimd_f16_t const *)&from, &f32);
            return f32;
        }
#endif
        return from;
    }

    /**
     *  @brief Randomizes the vector elements with normalized values.
     *
     *  This method fills the vector with random values. For floating-point types, the vector is normalized
     *  so that the sum of the squares of its elements equals 1. For integral types, the values are generated
     *  within the range of the scalar type.
     */
    void randomize(std::uint32_t seed) noexcept {

        static std::mt19937 generator;
        generator.seed(seed);

        if constexpr (is_integral) {
            std::uniform_int_distribution<scalar_t> distribution(std::numeric_limits<scalar_t>::min(),
                                                                 std::numeric_limits<scalar_t>::max());
            for (std::size_t i = 0; i != size_scalars(); ++i) { data_scalars()[i] = distribution(generator); }
        }
        else {
            // Using non-uniform distribution helps detect tail errors
            std::normal_distribution<double> distribution(0.1, 1.0);
            double squared_sum = 0.0;
            for (std::size_t i = 0; i != size_scalars(); ++i) {
                double a_i = distribution(generator);
                squared_sum += a_i * a_i;
                compress(a_i, data_scalars()[i]);
            }

            // Normalize the vectors:
            squared_sum = std::sqrt(squared_sum);
            for (std::size_t i = 0; i != size_scalars(); ++i) {
                compress(uncompress(data_scalars()[i]) / squared_sum, data_scalars()[i]);
                // Zero out NaNs
                if (std::isnan(uncompress(data_scalars()[i]))) data_scalars()[i] = 0;
            }
        }
    }
};

template <simsimd_datatype_t datatype_ak>
struct vectors_pair_gt {
    using vector_t = vector_gt<datatype_ak>;
    using scalar_t = typename vector_t::scalar_t;
    static constexpr bool is_integral = vector_t::is_integral;

    vector_t a;
    vector_t b;

    vectors_pair_gt() noexcept = default;
    vectors_pair_gt(std::size_t dimensions) noexcept : a(dimensions), b(dimensions) {}
    vectors_pair_gt(std::size_t size_a, std::size_t size_b) noexcept : a(size_a), b(size_b) {}
    vectors_pair_gt(vectors_pair_gt const &other) noexcept(false) : a(other.a), b(other.b) {}
    vectors_pair_gt &operator=(vectors_pair_gt const &other) noexcept(false) {
        if (this != &other) a = other.a, b = other.b;
        return *this;
    }
};

/**
 *  @brief Measures the performance of a @b dense metric function against a baseline using Google Benchmark.
 *  @tparam pair_at The type representing the vector pair used in the measurement.
 *  @tparam metric_at The type of the metric function (default is void).
 *  @param state The benchmark state object provided by Google Benchmark.
 *  @param metric The metric function to benchmark.
 *  @param baseline The baseline function to compare against.
 *  @param dimensions The number of dimensions in the vectors.
 */
template <typename pair_at, typename metric_at = void>
void measure_dense(bm::State &state, metric_at metric, metric_at baseline, std::size_t dimensions) {

    using pair_t = pair_at;
    using vector_t = typename pair_at::vector_t;

    auto call_baseline = [&](pair_t &pair) -> double {
        // Output for real vectors have a single dimensions.
        // Output for complex vectors have two dimensions.
        simsimd_distance_t results[2] = {signaling_distance, signaling_distance};
        baseline(pair.a.data(), pair.b.data(), pair.a.size(), &results[0]);
        return results[0];
    };
    auto call_contender = [&](pair_t &pair) -> double {
        // Output for real vectors have a single dimensions.
        // Output for complex vectors have two dimensions.
        simsimd_distance_t results[2] = {signaling_distance, signaling_distance};
        metric(pair.a.data(), pair.b.data(), pair.a.size(), &results[0]);
        return results[0];
    };

    // Let's average the distance results over many pairs.
    constexpr std::size_t pairs_count = 128;
    std::vector<pair_t> pairs(pairs_count);
    for (std::size_t i = 0; i != pairs.size(); ++i) {
        auto &pair = pairs[i];
        pair.a = pair.b = vector_t(dimensions);
        pair.a.randomize(static_cast<std::uint32_t>(i)), pair.b.randomize(static_cast<std::uint32_t>(i) + 54321u);
    }

    // Initialize the output buffers for distance calculations.
    std::vector<double> results_baseline(pairs.size());
    std::vector<double> results_contender(pairs.size());
    for (std::size_t i = 0; i != pairs.size(); ++i)
        results_baseline[i] = call_baseline(pairs[i]), results_contender[i] = call_contender(pairs[i]);

    // The actual benchmarking loop.
    std::size_t iterations = 0;
    for (auto _ : state)
        bm::DoNotOptimize((
            results_contender[iterations & (pairs_count - 1)] = call_contender(pairs[iterations & (pairs_count - 1)]))),
            iterations++;

    // Measure the mean absolute delta and relative error.
    double mean_delta = 0, mean_relative_error = 0;
    for (std::size_t i = 0; i != pairs.size(); ++i) {
        auto abs_delta = std::abs(results_contender[i] - results_baseline[i]);
        mean_delta += abs_delta;
        double error = abs_delta != 0 && results_baseline[i] != 0 ? abs_delta / std::abs(results_baseline[i]) : 0;
        mean_relative_error += error;
    }
    mean_delta /= pairs.size();
    mean_relative_error /= pairs.size();
    state.counters["abs_delta"] = mean_delta;
    state.counters["relative_error"] = mean_relative_error;
    state.counters["bytes"] = bm::Counter(iterations * pairs[0].a.size_bytes() * 2, bm::Counter::kIsRate);
    state.counters["pairs"] = bm::Counter(iterations, bm::Counter::kIsRate);
}

/**
 *  @brief Measures the performance of a @b curved metric function against a baseline using Google Benchmark.
 *  @tparam pair_at The type representing the vector pair used in the measurement.
 *  @tparam metric_at The type of the metric function (default is void).
 *  @param state The benchmark state object provided by Google Benchmark.
 *  @param metric The metric function to benchmark.
 *  @param baseline The baseline function to compare against.
 *  @param dimensions The number of dimensions in the vectors.
 */
template <typename pair_at, typename metric_at = void>
void measure_curved(bm::State &state, metric_at metric, metric_at baseline, std::size_t dimensions) {

    using pair_t = pair_at;
    using vector_t = typename pair_at::vector_t;

    auto call_baseline = [&](pair_t const &pair, vector_t const &tensor) -> double {
        simsimd_distance_t results[2] = {signaling_distance, 0};
        baseline(pair.a.data(), pair.b.data(), tensor.data(), pair.a.size(), &results[0]);
        return results[0] + results[1];
    };
    auto call_contender = [&](pair_t const &pair, vector_t const &tensor) -> double {
        simsimd_distance_t results[2] = {signaling_distance, 0};
        metric(pair.a.data(), pair.b.data(), tensor.data(), pair.a.size(), &results[0]);
        return results[0] + results[1];
    };

    // Let's average the distance results over many pairs.
    constexpr std::size_t pairs_count = 128;
    std::vector<pair_t> pairs(pairs_count);
    std::vector<vector_t> tensors(pairs_count);
    for (std::size_t i = 0; i != pairs.size(); ++i) {
        pair_t &pair = pairs[i];
        pair.a = pair.b = vector_t(dimensions);
        pair.a.randomize(static_cast<std::uint32_t>(i)), pair.b.randomize(static_cast<std::uint32_t>(i) + 54321u);
        vector_t &tensor = tensors[i];
        tensor = vector_t(dimensions * dimensions);
        tensor.randomize(static_cast<std::uint32_t>(i) + 123456u);
    }

    // Initialize the output buffers for distance calculations.
    std::vector<double> results_baseline(pairs.size());
    std::vector<double> results_contender(pairs.size());
    for (std::size_t i = 0; i != pairs.size(); ++i)
        results_baseline[i] = call_baseline(pairs[i], tensors[i]),
        results_contender[i] = call_contender(pairs[i], tensors[i]);

    // The actual benchmarking loop.
    std::size_t iterations = 0;
    for (auto _ : state)
        bm::DoNotOptimize((results_contender[iterations & (pairs_count - 1)] = call_contender(
                               pairs[iterations & (pairs_count - 1)], tensors[iterations & (pairs_count - 1)]))),
            iterations++;

    // Measure the mean absolute delta and relative error.
    double mean_delta = 0, mean_relative_error = 0;
    for (std::size_t i = 0; i != pairs.size(); ++i) {
        auto abs_delta = std::abs(results_contender[i] - results_baseline[i]);
        mean_delta += abs_delta;
        double error = abs_delta != 0 && results_baseline[i] != 0 ? abs_delta / std::abs(results_baseline[i]) : 0;
        mean_relative_error += error;
    }
    mean_delta /= pairs.size();
    mean_relative_error /= pairs.size();
    state.counters["abs_delta"] = mean_delta;
    state.counters["relative_error"] = mean_relative_error;
    state.counters["bytes"] = bm::Counter(iterations * pairs[0].a.size_bytes() * 2, bm::Counter::kIsRate);
    state.counters["pairs"] = bm::Counter(iterations, bm::Counter::kIsRate);
}

/**
 *  @brief Measures the performance of a @b sparse metric function against a baseline using Google Benchmark.
 *  @tparam pair_at The type representing the vector pair used in the measurement.
 *  @tparam metric_at The type of the metric function (default is void).
 *  @param state The benchmark state object provided by Google Benchmark.
 *  @param metric The metric function to benchmark.
 *  @param baseline The baseline function to compare against.
 *  @param size_a The number of elements in the smaller vector.
 *  @param size_b The number of elements in the larger vector.
 *  @param intersection_size The expected number of common scalars between the vectors.
 */
template <typename pair_at, typename metric_at = void>
void measure_sparse(bm::State &state, metric_at metric, metric_at baseline, std::size_t size_a, std::size_t size_b,
                    std::size_t intersection_size) {

    using pair_t = pair_at;
    using vector_t = typename pair_at::vector_t;
    using scalar_t = typename vector_t::scalar_t;

    auto call_baseline = [&](pair_t &pair) -> double {
        simsimd_distance_t result = std::numeric_limits<simsimd_distance_t>::signaling_NaN();
        baseline(pair.a.data(), pair.b.data(), pair.a.size(), pair.b.size(), &result);
        return result;
    };
    auto call_contender = [&](pair_t &pair) -> double {
        simsimd_distance_t result = std::numeric_limits<simsimd_distance_t>::signaling_NaN();
        metric(pair.a.data(), pair.b.data(), pair.a.size(), pair.b.size(), &result);
        return result;
    };

    // Let's average the distance results over many pairs.
    constexpr std::size_t pairs_count = 128;
    std::vector<pair_t> pairs(pairs_count);
    auto generator = make_random_engine();
    std::uniform_int_distribution<scalar_t> distribution(0, std::numeric_limits<scalar_t>::max());

    // Randomizing the vectors for sparse distances is a bit more complex then:
    //
    //      pair.a.randomize(), pair.b.randomize();
    //
    // We need to ensure that the intersection is of the expected size.
    std::unordered_set<scalar_t> intersection_set, unique_a, unique_b;
    intersection_set.reserve(intersection_size);
    unique_a.reserve(size_a - intersection_size);
    unique_b.reserve(size_b - intersection_size);

    for (auto &pair : pairs) {
        pair.a = vector_t(size_a);
        pair.b = vector_t(size_b);

        // Step 1: Generate intersection set
        intersection_set.clear();
        while (intersection_set.size() < intersection_size) intersection_set.insert(distribution(generator));

        unique_a.clear();
        while (unique_a.size() < size_a - intersection_size) {
            scalar_t element = distribution(generator);
            if (intersection_set.find(element) == intersection_set.end()) unique_a.insert(element);
        }

        unique_b.clear();
        while (unique_b.size() < size_b - intersection_size) {
            scalar_t element = distribution(generator);
            if (intersection_set.find(element) == intersection_set.end() && unique_a.find(element) == unique_a.end())
                unique_b.insert(element);
        }

        // Step 2: Merge and sort
        std::copy(intersection_set.begin(), intersection_set.end(), pair.a.values_ptr_);
        std::copy(intersection_set.begin(), intersection_set.end(), pair.b.values_ptr_);
        std::copy(unique_a.begin(), unique_a.end(), pair.a.values_ptr_ + intersection_size);
        std::copy(unique_b.begin(), unique_b.end(), pair.b.values_ptr_ + intersection_size);
        std::sort(pair.a.values_ptr_, pair.a.values_ptr_ + size_a);
        std::sort(pair.b.values_ptr_, pair.b.values_ptr_ + size_b);
    }

    // Initialize the output buffers for distance calculations.
    std::vector<double> results_baseline(pairs.size());
    std::vector<double> results_contender(pairs.size());
    for (std::size_t i = 0; i != pairs.size(); ++i)
        results_baseline[i] = call_baseline(pairs[i]), results_contender[i] = call_contender(pairs[i]);

    // The actual benchmarking loop.
    std::size_t iterations = 0;
    for (auto _ : state)
        bm::DoNotOptimize((
            results_contender[iterations & (pairs_count - 1)] = call_contender(pairs[iterations & (pairs_count - 1)]))),
            iterations++;

    // Measure the mean absolute delta and relative error.
    double mean_error = 0;
    for (std::size_t i = 0; i != pairs.size(); ++i) {
        auto abs_error = std::abs(results_contender[i] - results_baseline[i]);
        mean_error += abs_error;
    }
    mean_error /= pairs.size();
    state.counters["error"] = mean_error;
    state.counters["bytes"] = bm::Counter(iterations * (pairs[0].a.size_bytes() + pairs[0].b.size_bytes()),
                                          bm::Counter::kIsRate);
    state.counters["pairs"] = bm::Counter(iterations, bm::Counter::kIsRate);
    state.counters["matches"] = std::accumulate(results_contender.begin(), results_contender.end(), 0.0) /
                                results_contender.size();
}

/**
 *  @brief Point cloud pair for mesh metrics (RMSD, Kabsch).
 *  Each point cloud contains n 3D points stored as [x0,y0,z0,x1,y1,z1,...].
 */
template <simsimd_datatype_t datatype_ak>
struct mesh_pair_gt {
    using vector_t = vector_gt<datatype_ak>;
    using scalar_t = typename vector_t::scalar_t;

    vector_t a;
    vector_t b;
    std::size_t num_points;

    mesh_pair_gt() noexcept = default;
    mesh_pair_gt(std::size_t points) noexcept : a(points * 3), b(points * 3), num_points(points) {}
    mesh_pair_gt(mesh_pair_gt const &other) noexcept(false) : a(other.a), b(other.b), num_points(other.num_points) {}
    mesh_pair_gt &operator=(mesh_pair_gt const &other) noexcept(false) {
        if (this != &other) a = other.a, b = other.b, num_points = other.num_points;
        return *this;
    }
};

/**
 *  @brief Measures the performance of a @b mesh metric function (RMSD/Kabsch) against a baseline.
 *  @tparam pair_at The type representing the point cloud pair.
 *  @tparam metric_at The type of the metric function.
 *  @param state The benchmark state object provided by Google Benchmark.
 *  @param metric The metric function to benchmark.
 *  @param baseline The baseline function to compare against.
 *  @param num_points The number of 3D points in each point cloud.
 */
template <typename pair_at, typename metric_at = void>
void measure_mesh(bm::State &state, metric_at metric, metric_at baseline, std::size_t num_points) {

    using pair_t = pair_at;
    using vector_t = typename pair_at::vector_t;
    using scalar_t = typename vector_t::scalar_t;

    auto call_baseline = [&](pair_t &pair) -> double {
        simsimd_distance_t result = signaling_distance, scale = 0;
        scalar_t a_centroid[3], b_centroid[3], rotation[9];
        baseline(pair.a.data(), pair.b.data(), pair.num_points, a_centroid, b_centroid, rotation, &scale, &result);
        return result;
    };
    auto call_contender = [&](pair_t &pair) -> double {
        simsimd_distance_t result = signaling_distance, scale = 0;
        scalar_t a_centroid[3], b_centroid[3], rotation[9];
        metric(pair.a.data(), pair.b.data(), pair.num_points, a_centroid, b_centroid, rotation, &scale, &result);
        return result;
    };

    // Let's average the distance results over many pairs.
    constexpr std::size_t pairs_count = 128;
    std::vector<pair_t> pairs(pairs_count);
    for (std::size_t i = 0; i != pairs.size(); ++i) {
        auto &pair = pairs[i];
        pair = pair_t(num_points);
        pair.a.randomize(static_cast<std::uint32_t>(i)), pair.b.randomize(static_cast<std::uint32_t>(i) + 54321u);
    }

    // Initialize the output buffers for distance calculations.
    std::vector<double> results_baseline(pairs.size());
    std::vector<double> results_contender(pairs.size());
    for (std::size_t i = 0; i != pairs.size(); ++i)
        results_baseline[i] = call_baseline(pairs[i]), results_contender[i] = call_contender(pairs[i]);

    // The actual benchmarking loop.
    std::size_t iterations = 0;
    for (auto _ : state)
        bm::DoNotOptimize((
            results_contender[iterations & (pairs_count - 1)] = call_contender(pairs[iterations & (pairs_count - 1)]))),
            iterations++;

    // Measure the mean absolute delta and relative error.
    double mean_delta = 0, mean_relative_error = 0;
    for (std::size_t i = 0; i != pairs.size(); ++i) {
        auto abs_delta = std::abs(results_contender[i] - results_baseline[i]);
        mean_delta += abs_delta;
        double error = abs_delta != 0 && results_baseline[i] != 0 ? abs_delta / std::abs(results_baseline[i]) : 0;
        mean_relative_error += error;
    }
    mean_delta /= pairs.size();
    mean_relative_error /= pairs.size();
    state.counters["abs_delta"] = mean_delta;
    state.counters["relative_error"] = mean_relative_error;
    state.counters["bytes"] = bm::Counter(iterations * pairs[0].a.size_bytes() * 2, bm::Counter::kIsRate);
    state.counters["pairs"] = bm::Counter(iterations, bm::Counter::kIsRate);
}

/**
 *  @brief Measures the performance of a vector-vector @b FMA function against a baseline using Google Benchmark.
 *  @tparam pair_at The type representing the vector pair used in the measurement.
 *  @tparam kernel_at The type of the kernel function (default is void).
 *  @param state The benchmark state object provided by Google Benchmark.
 *  @param kernel The kernel function to benchmark.
 *  @param baseline The baseline function to compare against.
 *  @param l2_metric The L2 metric function to compute the error
 *  @param dimensions The number of dimensions in the vectors.
 */
template <typename pair_at, simsimd_metric_kind_t kernel_ak, typename kernel_at = void, typename l2_metric_at = void>
void measure_elementwise(bm::State &state, kernel_at kernel, kernel_at baseline, l2_metric_at l2_metric,
                         std::size_t dimensions) {

    using pair_t = pair_at;
    using vector_t = typename pair_at::vector_t;

    constexpr simsimd_distance_t alpha = 0.2;
    constexpr simsimd_distance_t beta = 0.3;
    auto call_baseline = [&](vector_t const &a, vector_t const &b, vector_t const &c, vector_t &d) {
        if constexpr (kernel_ak == simsimd_metric_wsum_k) {
            baseline(a.data(), c.data(), a.size(), alpha, beta, d.data());
        }
        else if constexpr (kernel_ak == simsimd_metric_fma_k) {
            baseline(a.data(), b.data(), c.data(), a.size(), alpha, beta, d.data());
        }
        else if constexpr (kernel_ak == simsimd_metric_sum_k) { baseline(a.data(), c.data(), a.size(), d.data()); }
        else if constexpr (kernel_ak == simsimd_metric_scale_k) { baseline(a.data(), a.size(), alpha, beta, d.data()); }
        else { baseline(a.data(), a.size(), d.data()); }
    };
    auto call_contender = [&](vector_t const &a, vector_t const &b, vector_t const &c, vector_t &d) {
        if constexpr (kernel_ak == simsimd_metric_wsum_k) {
            kernel(a.data(), c.data(), a.size(), alpha, beta, d.data());
        }
        else if constexpr (kernel_ak == simsimd_metric_fma_k) {
            kernel(a.data(), b.data(), c.data(), a.size(), alpha, beta, d.data());
        }
        else if constexpr (kernel_ak == simsimd_metric_sum_k) { kernel(a.data(), c.data(), a.size(), d.data()); }
        else if constexpr (kernel_ak == simsimd_metric_scale_k) { kernel(a.data(), a.size(), alpha, beta, d.data()); }
        else { kernel(a.data(), a.size(), d.data()); }
    };

    // Let's average the distance results over many quads.
    struct quad_t {
        vector_t a, b, c, d;
    };
    constexpr std::size_t quads_count = 128;
    std::vector<quad_t> quads(quads_count);
    for (std::size_t i = 0; i != quads.size(); ++i) {
        auto &quad = quads[i];
        quad.a = quad.b = quad.c = quad.d = vector_t(dimensions);
        quad.a.randomize(static_cast<std::uint32_t>(i));
        quad.b.set(2); // Having a small constant here will help avoid overflows
        quad.c.randomize(static_cast<std::uint32_t>(i) + 54321u);
    }

    // Initialize the output buffers for distance calculations.
    vector_t baseline_d(dimensions), contender_d(dimensions), zeros(dimensions);
    std::vector<simsimd_distance_t> l2_metric_from_baseline(quads.size());
    std::vector<simsimd_distance_t> l2_baseline_result_norm(quads.size());
    std::vector<simsimd_distance_t> l2_contender_result_norm(quads.size());
    zeros.set(0);
    double mean_delta = 0, mean_relative_error = 0;
    for (std::size_t i = 0; i != quads.size(); ++i) {
        quad_t &quad = quads[i];
        call_baseline(quad.a, quad.b, quad.c, baseline_d);
        call_contender(quad.a, quad.b, quad.c, contender_d);
        l2_metric(baseline_d.data(), contender_d.data(), dimensions, &l2_metric_from_baseline[i]);
        l2_metric(baseline_d.data(), zeros.data(), dimensions, &l2_baseline_result_norm[i]);
        l2_metric(contender_d.data(), zeros.data(), dimensions, &l2_contender_result_norm[i]);

        mean_delta += std::abs(l2_metric_from_baseline[i]);
        mean_relative_error += std::abs(l2_metric_from_baseline[i]) /
                               (std::max)(l2_baseline_result_norm[i], l2_contender_result_norm[i]);
    }
    mean_delta /= quads_count;
    mean_relative_error /= quads_count;

    // The actual benchmarking loop.
    std::size_t iterations = 0;
    for (auto _ : state) {
        quad_t &quad = quads[iterations & (quads_count - 1)];
        call_contender(quad.a, quad.b, quad.c, quad.d);
        iterations++;
    }

    std::size_t bytes_per_call = quads[0].a.size_bytes();
    if constexpr (kernel_ak == simsimd_metric_wsum_k) { bytes_per_call *= 2; }
    else if constexpr (kernel_ak == simsimd_metric_fma_k) { bytes_per_call *= 3; }
    else if constexpr (kernel_ak == simsimd_metric_sum_k) { bytes_per_call *= 2; }
    else if constexpr (kernel_ak == simsimd_metric_scale_k) { bytes_per_call *= 1; }

    // Measure the mean absolute delta and relative error.
    state.counters["abs_delta"] = mean_delta;
    state.counters["relative_error"] = mean_relative_error;
    state.counters["bytes"] = bm::Counter(iterations * bytes_per_call, bm::Counter::kIsRate);
    state.counters["pairs"] = bm::Counter(iterations, bm::Counter::kIsRate);
}

/**
 *  @brief Measures the performance of a geospatial operations between 4 arrays: 2 latitudes, 2 longitudes.
 *  @tparam pair_at The type representing the vector pair used in the measurement.
 *  @tparam kernel_at The type of the kernel function (default is void).
 *  @param state The benchmark state object provided by Google Benchmark.
 *  @param kernel The kernel function to benchmark.
 *  @param baseline The baseline function to compare against.
 *  @param l2_metric The L2 metric function to compute the error
 *  @param dimensions The number of dimensions in the vectors.
 */
template <typename pair_at, typename kernel_at = void, typename l2_metric_at = void>
void measure_geospatial(bm::State &state, kernel_at kernel, kernel_at baseline, l2_metric_at l2_metric,
                        std::size_t dimensions) {

    using pair_t = pair_at;
    using vector_t = typename pair_at::vector_t;
    using scalar_t = typename vector_t::scalar_t;
    struct quad_t {
        vector_t lat1, lon1, lat2, lon2;
    };

    using distances_t = vector_gt<simsimd_f64_k>;
    auto call_baseline = [&](quad_t const &quad, distances_t &d) {
        baseline(quad.lat1.data(), quad.lon1.data(), quad.lat2.data(), quad.lon2.data(), quad.lat1.size(), d.data());
    };
    auto call_contender = [&](quad_t const &quad, distances_t &d) {
        kernel(quad.lat1.data(), quad.lon1.data(), quad.lat2.data(), quad.lon2.data(), quad.lat1.size(), d.data());
    };

    // Let's average the distance results over many quads.
    constexpr std::size_t quads_count = 128;
    std::vector<quad_t> quads(quads_count);
    auto random_generator = make_random_engine();
    /// Latitude range (-90 to 90 degrees) in radians
    std::uniform_real_distribution<scalar_t> lat_dist(-M_PI_2, M_PI_2);
    /// Longitude range (-180 to 180 degrees) in radians
    std::uniform_real_distribution<scalar_t> lon_dist(-M_PI, M_PI);
    for (std::size_t i = 0; i != quads.size(); ++i) {
        auto &quad = quads[i];
        quad.lat1 = quad.lat2 = quad.lon1 = quad.lon2 = vector_t(dimensions);
        std::generate(quad.lat1.data(), quad.lat1.data() + dimensions, [&]() { return lat_dist(random_generator); });
        std::generate(quad.lat2.data(), quad.lat2.data() + dimensions, [&]() { return lat_dist(random_generator); });
        std::generate(quad.lon1.data(), quad.lon1.data() + dimensions, [&]() { return lon_dist(random_generator); });
        std::generate(quad.lon2.data(), quad.lon2.data() + dimensions, [&]() { return lon_dist(random_generator); });
    }

    // Initialize the output buffers for distance calculations.
    distances_t baseline_d(dimensions), contender_d(dimensions), zeros(dimensions);
    std::vector<simsimd_distance_t> l2_metric_from_baseline(quads.size());
    std::vector<simsimd_distance_t> l2_baseline_result_norm(quads.size());
    std::vector<simsimd_distance_t> l2_contender_result_norm(quads.size());
    zeros.set(0);
    double mean_delta = 0, mean_relative_error = 0;
    for (std::size_t i = 0; i != quads.size(); ++i) {
        quad_t &quad = quads[i];
        call_baseline(quad, baseline_d);
        call_contender(quad, contender_d);
        l2_metric(baseline_d.data(), contender_d.data(), dimensions, &l2_metric_from_baseline[i]);
        l2_metric(baseline_d.data(), zeros.data(), dimensions, &l2_baseline_result_norm[i]);
        l2_metric(contender_d.data(), zeros.data(), dimensions, &l2_contender_result_norm[i]);

        mean_delta += std::abs(l2_metric_from_baseline[i]);
        mean_relative_error += std::abs(l2_metric_from_baseline[i]) /
                               (std::max)(l2_baseline_result_norm[i], l2_contender_result_norm[i]);
    }
    mean_delta /= quads_count;
    mean_relative_error /= quads_count;

    // The actual benchmarking loop.
    std::size_t iterations = 0;
    for (auto _ : state) {
        quad_t &quad = quads[iterations & (quads_count - 1)];
        call_contender(quad, contender_d);
        iterations++;
    }

    std::size_t const bytes_per_call =                            //
        quads[0].lat1.size_bytes() + quads[0].lat2.size_bytes() + //
        quads[0].lon1.size_bytes() + quads[0].lon2.size_bytes();
    // Measure the mean absolute delta and relative error.
    state.counters["abs_delta"] = mean_delta;
    state.counters["relative_error"] = mean_relative_error;
    state.counters["bytes"] = bm::Counter(iterations * bytes_per_call, bm::Counter::kIsRate);
    state.counters["pairs"] = bm::Counter(iterations * dimensions, bm::Counter::kIsRate);
}

template <simsimd_datatype_t datatype_ak, typename metric_at = void>
void dense_(std::string name, metric_at *distance_func, metric_at *baseline_func) {
    using pair_t = vectors_pair_gt<datatype_ak>;
    std::string bench_name = name + "<" + std::to_string(dense_dimensions) + "d>";
    bm::RegisterBenchmark(bench_name.c_str(), measure_dense<pair_t, metric_at *>, distance_func, baseline_func,
                          dense_dimensions)
        ->MinTime(default_seconds)
        ->Threads(default_threads);
}

template <simsimd_datatype_t datatype_ak, simsimd_metric_kind_t kernel_ak = simsimd_metric_unknown_k,
          typename kernel_at = void, typename l2_metric_at = void>
void elementwise_(std::string name, kernel_at *kernel_func, kernel_at *baseline_func, l2_metric_at *l2_metric_func) {
    using pair_t = vectors_pair_gt<datatype_ak>;
    std::string bench_name = name + "<" + std::to_string(dense_dimensions) + "d>";
    bm::RegisterBenchmark(bench_name.c_str(), measure_elementwise<pair_t, kernel_ak, kernel_at *, l2_metric_at *>,
                          kernel_func, baseline_func, l2_metric_func, dense_dimensions)
        ->MinTime(default_seconds)
        ->Threads(default_threads);
}

template <simsimd_datatype_t datatype_ak, typename kernel_at = void, typename l2_metric_at = void>
void geospatial_(std::string name, kernel_at *kernel_func, kernel_at *baseline_func, l2_metric_at *l2_metric_func) {
    using pair_t = vectors_pair_gt<datatype_ak>;
    std::string bench_name = name + "<" + std::to_string(dense_dimensions) + "d>";
    bm::RegisterBenchmark(bench_name.c_str(), measure_geospatial<pair_t, kernel_at *, l2_metric_at *>, kernel_func,
                          baseline_func, l2_metric_func, dense_dimensions)
        ->MinTime(default_seconds)
        ->Threads(default_threads);
}

template <simsimd_datatype_t datatype_ak, typename metric_at = void>
void sparse_(std::string name, metric_at *distance_func, metric_at *baseline_func) {

    using pair_t = vectors_pair_gt<datatype_ak>;

    // Register different lengths, intersection sizes, and distributions
    // 2 first lengths * 3 second length multipliers * 4 intersection grades = 24 benchmarks for each metric.
    for (std::size_t first_len : {128, 1024}) {                         //< 2 lengths
        for (std::size_t second_len_multiplier : {1, 8, 64}) {          //< 3 length multipliers
            for (double intersection_share : {0.01, 0.05, 0.5, 0.95}) { //< 4 intersection grades
                std::size_t intersection_size = static_cast<std::size_t>(first_len * intersection_share);
                std::size_t second_len = first_len * second_len_multiplier;
                std::string bench_name = name + "<|A|=" + std::to_string(first_len) +
                                         ",|B|=" + std::to_string(second_len) +
                                         ",|A∩B|=" + std::to_string(intersection_size) + ">";
                if (second_len > 8192) continue;
                bm::RegisterBenchmark(bench_name.c_str(), measure_sparse<pair_t, metric_at *>, distance_func,
                                      baseline_func, first_len, second_len, intersection_size)
                    ->MinTime(default_seconds)
                    ->Threads(default_threads);
            }
        }
    }
}

template <simsimd_datatype_t datatype_ak, typename metric_at = void>
void curved_(std::string name, metric_at *distance_func, metric_at *baseline_func) {

    using pair_t = vectors_pair_gt<datatype_ak>;
    std::string bench_name = name + "<" + std::to_string(curved_dimensions) + "d>";
    bm::RegisterBenchmark(bench_name.c_str(), measure_curved<pair_t, metric_at *>, distance_func, baseline_func,
                          curved_dimensions)
        ->MinTime(default_seconds)
        ->Threads(default_threads);
}

template <simsimd_datatype_t datatype_ak, typename metric_at = void>
void mesh_(std::string name, metric_at *distance_func, metric_at *baseline_func) {

    using pair_t = mesh_pair_gt<datatype_ak>;
    std::string bench_name = name + "<" + std::to_string(mesh_dimensions) + "pts>";
    bm::RegisterBenchmark(bench_name.c_str(), measure_mesh<pair_t, metric_at *>, distance_func, baseline_func,
                          mesh_dimensions)
        ->MinTime(default_seconds)
        ->Threads(default_threads);
}

//  Matmul measurement for packed B matrix API
template <typename input_at, typename output_at>
void measure_matmul_packed(bm::State &state,                                                                  //
                           simsimd_size_t (*packed_size_fn)(simsimd_size_t, simsimd_size_t),                  //
                           void (*pack_fn)(input_at const *, simsimd_size_t, simsimd_size_t, simsimd_size_t,  //
                                           void *),                                                           //
                           void (*matmul_fn)(input_at const *, void const *, output_at *, simsimd_size_t,     //
                                             simsimd_size_t, simsimd_size_t, simsimd_size_t, simsimd_size_t), //
                           std::size_t m, std::size_t n, std::size_t k) {

    // Allocate matrices
    std::vector<input_at> a(m * k);
    std::vector<input_at> b(n * k);
    simsimd_size_t packed_bytes = packed_size_fn(n, k);
    std::vector<char> b_packed(packed_bytes, 0);
    std::vector<output_at> c(m * n);

    // Initialize with small random values
    auto gen = make_random_engine();
    if constexpr (std::is_integral<input_at>::value) {
        std::uniform_int_distribution<int> dis(-10, 10);
        for (auto &v : a) v = static_cast<input_at>(dis(gen));
        for (auto &v : b) v = static_cast<input_at>(dis(gen));
    }
    else {
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
        for (std::size_t i = 0; i < a.size(); ++i) a[i] = static_cast<input_at>(dis(gen));
        for (std::size_t i = 0; i < b.size(); ++i) b[i] = static_cast<input_at>(dis(gen));
    }

    // Pack B matrix once (amortized cost for repeated inference)
    pack_fn(b.data(), n, k, k * sizeof(input_at), b_packed.data());

    std::size_t iterations = 0;
    for (auto _ : state) {
        bm::DoNotOptimize(c.data());
        matmul_fn(a.data(), b_packed.data(), c.data(), m, n, k, k * sizeof(input_at), n * sizeof(output_at));
        ++iterations;
    }

    // Report FLOPS: 2*m*n*k operations per matmul (multiply + add)
    double flops_per_call = 2.0 * m * n * k;
    state.counters["flops"] = bm::Counter(iterations * flops_per_call, bm::Counter::kIsRate);
    state.counters["bytes_a"] = bm::Counter(iterations * m * k * sizeof(input_at), bm::Counter::kIsRate);
    state.counters["bytes_c"] = bm::Counter(iterations * m * n * sizeof(output_at), bm::Counter::kIsRate);
}

template <typename input_at, typename output_at>
void matmul_(std::string name,                                                                              //
             simsimd_size_t (*packed_size_fn)(simsimd_size_t, simsimd_size_t),                              //
             void (*pack_fn)(input_at const *, simsimd_size_t, simsimd_size_t, simsimd_size_t, void *),     //
             void (*matmul_fn)(input_at const *, void const *, output_at *, simsimd_size_t, simsimd_size_t, //
                               simsimd_size_t, simsimd_size_t, simsimd_size_t)) {                           //
    std::string bench_name = name + "<" + std::to_string(matmul_dimension_m) + "x" +
                             std::to_string(matmul_dimension_n) + "x" + std::to_string(matmul_dimension_k) + ">";
    bm::RegisterBenchmark(bench_name.c_str(), measure_matmul_packed<input_at, output_at>, packed_size_fn, pack_fn,
                          matmul_fn, matmul_dimension_m, matmul_dimension_n, matmul_dimension_k)
        ->MinTime(default_seconds)
        ->Threads(1); // Single-threaded for packed matmul
}

template <typename scalar_at>
void l2_with_stl(scalar_at const *a, scalar_at const *b, simsimd_size_t n, simsimd_distance_t *result) {
    simsimd_distance_t sum = 0;
    for (simsimd_size_t i = 0; i != n; ++i) {
        simsimd_distance_t delta = (simsimd_distance_t)a[i] - (simsimd_distance_t)b[i];
        sum += delta * delta;
    }
    *result = std::sqrt(sum);
}

template <typename scalar_at, typename accumulator_at = scalar_at>
simsimd_distance_t haversine_one_with_stl(scalar_at lat1, scalar_at lon1, scalar_at lat2, scalar_at lon2) {
    // Convert angle to radians:
    // lat1 *= M_PI / 180, lon1 *= M_PI / 180;
    // lat2 *= M_PI / 180, lon2 *= M_PI / 180;
    accumulator_at dlat = lat2 - lat1;
    accumulator_at dlon = lon2 - lon1;
    accumulator_at a = //
        std::sin(dlat / 2) * std::sin(dlat / 2) +
        std::cos(lat1) * std::cos(lat2) * std::sin(dlon / 2) * std::sin(dlon / 2);
    accumulator_at c = 2 * std::atan2(std::sqrt(a), std::sqrt(1 - a));
    return c;
}

template <typename scalar_at, typename accumulator_at = scalar_at>
void haversine_with_stl(                              //
    scalar_at const *a_lats, scalar_at const *a_lons, //
    scalar_at const *b_lats, scalar_at const *b_lons, //
    simsimd_size_t n, simsimd_distance_t *results) {
    for (simsimd_size_t i = 0; i != n; ++i) {
        scalar_at lat1 = a_lats[i], lon1 = a_lons[i];
        scalar_at lat2 = b_lats[i], lon2 = b_lons[i];
        results[i] = haversine_one_with_stl<scalar_at, accumulator_at>(lat1, lon1, lat2, lon2);
    }
}

template <typename scalar_at, typename accumulator_at = scalar_at>
void vincenty_with_stl(                               //
    scalar_at const *a_lats, scalar_at const *a_lons, //
    scalar_at const *b_lats, scalar_at const *b_lons, //
    simsimd_size_t n, simsimd_distance_t *results) {
    // Simplified Vincenty baseline - uses same iterative algorithm as serial implementation
    constexpr accumulator_at equatorial_radius = 6378136.6;
    constexpr accumulator_at polar_radius = 6356751.9;
    constexpr accumulator_at flattening = 1.0 / 298.25642;
    constexpr accumulator_at convergence_threshold = 1e-12;
    constexpr int max_iterations = 100;

    for (simsimd_size_t i = 0; i != n; ++i) {
        accumulator_at lat1 = a_lats[i], lon1 = a_lons[i];
        accumulator_at lat2 = b_lats[i], lon2 = b_lons[i];
        accumulator_at longitude_diff = lon2 - lon1;

        // Reduced latitudes
        accumulator_at tan_reduced_lat_1 = (1.0 - flattening) * std::tan(lat1);
        accumulator_at tan_reduced_lat_2 = (1.0 - flattening) * std::tan(lat2);
        accumulator_at cos_reduced_lat_1 = 1.0 / std::sqrt(1.0 + tan_reduced_lat_1 * tan_reduced_lat_1);
        accumulator_at sin_reduced_lat_1 = tan_reduced_lat_1 * cos_reduced_lat_1;
        accumulator_at cos_reduced_lat_2 = 1.0 / std::sqrt(1.0 + tan_reduced_lat_2 * tan_reduced_lat_2);
        accumulator_at sin_reduced_lat_2 = tan_reduced_lat_2 * cos_reduced_lat_2;

        accumulator_at lambda_longitude = longitude_diff, lambda_previous;
        accumulator_at sin_sigma, cos_sigma, sigma, sin_alpha, cos_squared_azimuth, cos_twice_sigma_midpoint;
        int iteration_count = 0;
        bool points_are_coincident = false;

        do {
            accumulator_at sin_lambda = std::sin(lambda_longitude);
            accumulator_at cos_lambda = std::cos(lambda_longitude);
            accumulator_at sin_sigma_term_a = cos_reduced_lat_2 * sin_lambda;
            accumulator_at sin_sigma_term_b = cos_reduced_lat_1 * sin_reduced_lat_2 -
                                              sin_reduced_lat_1 * cos_reduced_lat_2 * cos_lambda;
            sin_sigma = std::sqrt(sin_sigma_term_a * sin_sigma_term_a + sin_sigma_term_b * sin_sigma_term_b);

            if (sin_sigma == 0.0) {
                points_are_coincident = true;
                break;
            }

            cos_sigma = sin_reduced_lat_1 * sin_reduced_lat_2 + cos_reduced_lat_1 * cos_reduced_lat_2 * cos_lambda;
            sigma = std::atan2(sin_sigma, cos_sigma);
            sin_alpha = cos_reduced_lat_1 * cos_reduced_lat_2 * sin_lambda / sin_sigma;
            cos_squared_azimuth = 1.0 - sin_alpha * sin_alpha;
            cos_twice_sigma_midpoint = (cos_squared_azimuth != 0.0)
                                           ? cos_sigma -
                                                 2.0 * sin_reduced_lat_1 * sin_reduced_lat_2 / cos_squared_azimuth
                                           : 0.0;
            accumulator_at longitude_correction_coeff = flattening / 16.0 * cos_squared_azimuth *
                                                        (4.0 + flattening * (4.0 - 3.0 * cos_squared_azimuth));

            lambda_previous = lambda_longitude;
            lambda_longitude = longitude_diff +
                               (1.0 - longitude_correction_coeff) * flattening * sin_alpha *
                                   (sigma + longitude_correction_coeff * sin_sigma *
                                                (cos_twice_sigma_midpoint + longitude_correction_coeff * cos_sigma *
                                                                                (-1.0 + 2.0 * cos_twice_sigma_midpoint *
                                                                                            cos_twice_sigma_midpoint)));
            iteration_count++;
        } while (std::abs(lambda_longitude - lambda_previous) > convergence_threshold &&
                 iteration_count < max_iterations);

        if (points_are_coincident) {
            results[i] = 0.0;
            continue;
        }

        accumulator_at u_squared = cos_squared_azimuth *
                                   (equatorial_radius * equatorial_radius - polar_radius * polar_radius) /
                                   (polar_radius * polar_radius);
        accumulator_at geodesic_length_coeff =
            1.0 + u_squared / 16384.0 * (4096.0 + u_squared * (-768.0 + u_squared * (320.0 - 175.0 * u_squared)));
        accumulator_at delta_sigma_coeff = u_squared / 1024.0 *
                                           (256.0 + u_squared * (-128.0 + u_squared * (74.0 - 47.0 * u_squared)));
        accumulator_at delta_sigma =
            delta_sigma_coeff * sin_sigma *
            (cos_twice_sigma_midpoint +
             delta_sigma_coeff / 4.0 *
                 (cos_sigma * (-1.0 + 2.0 * cos_twice_sigma_midpoint * cos_twice_sigma_midpoint) -
                  delta_sigma_coeff / 6.0 * cos_twice_sigma_midpoint * (-3.0 + 4.0 * sin_sigma * sin_sigma) *
                      (-3.0 + 4.0 * cos_twice_sigma_midpoint * cos_twice_sigma_midpoint)));

        results[i] = polar_radius * geodesic_length_coeff * (sigma - delta_sigma);
    }
}

template <typename scalar_at>
struct sin_with_stl {
    scalar_at operator()(scalar_at x) const { return std::sin(x); }
};
template <typename scalar_at>
struct cos_with_stl {
    scalar_at operator()(scalar_at x) const { return std::cos(x); }
};
template <typename scalar_at>
struct atan_with_stl {
    scalar_at operator()(scalar_at x) const { return std::atan(x); }
};

namespace av::simsimd {
struct sin {
    simsimd_f32_t operator()(simsimd_f32_t x) const { return simsimd_f32_sin(x); }
    simsimd_f64_t operator()(simsimd_f64_t x) const { return simsimd_f64_sin(x); }
};
struct cos {
    simsimd_f32_t operator()(simsimd_f32_t x) const { return simsimd_f32_cos(x); }
    simsimd_f64_t operator()(simsimd_f64_t x) const { return simsimd_f64_cos(x); }
};
struct atan {
    simsimd_f32_t operator()(simsimd_f32_t x) const { return simsimd_f32_atan(x); }
    simsimd_f64_t operator()(simsimd_f64_t x) const { return simsimd_f64_atan(x); }
};
} // namespace av::simsimd

template <typename scalar_at, typename kernel_at>
void elementwise_with_stl(scalar_at const *ins, simsimd_size_t n, scalar_at *outs) {
    for (simsimd_size_t i = 0; i != n; ++i) outs[i] = kernel_at {}(ins[i]);
}

#if SIMSIMD_BUILD_BENCHMARKS_WITH_BLAS

void dot_f32_blas(simsimd_f32_t const *a, simsimd_f32_t const *b, simsimd_size_t n, simsimd_distance_t *result) {
    *result = cblas_sdot(static_cast<int>(n), a, 1, b, 1);
}

void dot_f64_blas(simsimd_f64_t const *a, simsimd_f64_t const *b, simsimd_size_t n, simsimd_distance_t *result) {
    *result = cblas_ddot(static_cast<int>(n), a, 1, b, 1);
}

void dot_f32c_blas(simsimd_f32c_t const *a, simsimd_f32c_t const *b, simsimd_size_t n, simsimd_distance_t *result) {
    simsimd_f32_t f32_result[2] = {0, 0};
    cblas_cdotu_sub(static_cast<int>(n), reinterpret_cast<simsimd_f32_t const *>(a), 1,
                    reinterpret_cast<simsimd_f32_t const *>(b), 1, f32_result);
    result[0] = f32_result[0];
    result[1] = f32_result[1];
}

void dot_f64c_blas(simsimd_f64c_t const *a, simsimd_f64c_t const *b, simsimd_size_t n, simsimd_distance_t *result) {
    cblas_zdotu_sub(static_cast<int>(n), reinterpret_cast<simsimd_f64_t const *>(a), 1,
                    reinterpret_cast<simsimd_f64_t const *>(b), 1, result);
}

void vdot_f32c_blas(simsimd_f32c_t const *a, simsimd_f32c_t const *b, simsimd_size_t n, simsimd_distance_t *result) {
    simsimd_f32_t f32_result[2] = {0, 0};
    cblas_cdotc_sub(static_cast<int>(n), reinterpret_cast<simsimd_f32_t const *>(a), 1,
                    reinterpret_cast<simsimd_f32_t const *>(b), 1, f32_result);
    result[0] = f32_result[0];
    result[1] = f32_result[1];
}

void vdot_f64c_blas(simsimd_f64c_t const *a, simsimd_f64c_t const *b, simsimd_size_t n, simsimd_distance_t *result) {
    cblas_zdotc_sub(static_cast<int>(n), reinterpret_cast<simsimd_f64_t const *>(a), 1,
                    reinterpret_cast<simsimd_f64_t const *>(b), 1, result);
}

void bilinear_f32_blas(simsimd_f32_t const *a, simsimd_f32_t const *b, simsimd_f32_t const *c, simsimd_size_t n,
                       simsimd_distance_t *result) {
    static thread_local std::vector<simsimd_f32_t> intermediate;
    if (intermediate.size() < n) intermediate.resize(n);
    int const ni = static_cast<int>(n);
    cblas_sgemv(CblasRowMajor, CblasNoTrans, ni, ni, 1.0f, c, ni, b, 1, 0.0f, intermediate.data(), 1);
    *result = cblas_sdot(ni, a, 1, intermediate.data(), 1);
}

void bilinear_f64_blas(simsimd_f64_t const *a, simsimd_f64_t const *b, simsimd_f64_t const *c, simsimd_size_t n,
                       simsimd_distance_t *result) {
    static thread_local std::vector<simsimd_f64_t> intermediate;
    if (intermediate.size() < n) intermediate.resize(n);
    int const ni = static_cast<int>(n);
    cblas_dgemv(CblasRowMajor, CblasNoTrans, ni, ni, 1.0, c, ni, b, 1, 0.0, intermediate.data(), 1);
    *result = cblas_ddot(ni, a, 1, intermediate.data(), 1);
}

void bilinear_f32c_blas(simsimd_f32c_t const *a, simsimd_f32c_t const *b, simsimd_f32c_t const *c, simsimd_size_t n,
                        simsimd_distance_t *results) {
    static thread_local std::vector<simsimd_f32c_t> intermediate;
    if (intermediate.size() < n) intermediate.resize(n);
    int const ni = static_cast<int>(n);
    simsimd_f32c_t alpha = {1.0f, 0.0f}, beta = {0.0f, 0.0f};
    cblas_cgemv(CblasRowMajor, CblasNoTrans, ni, ni, &alpha, c, ni, b, 1, &beta, intermediate.data(), 1);
    simsimd_f32_t f32_result[2] = {0, 0};
    cblas_cdotu_sub(ni, reinterpret_cast<simsimd_f32_t const *>(a), 1,
                    reinterpret_cast<simsimd_f32_t const *>(intermediate.data()), 1, f32_result);
    results[0] = f32_result[0];
    results[1] = f32_result[1];
}

void bilinear_f64c_blas(simsimd_f64c_t const *a, simsimd_f64c_t const *b, simsimd_f64c_t const *c, simsimd_size_t n,
                        simsimd_distance_t *results) {
    static thread_local std::vector<simsimd_f64c_t> intermediate;
    if (intermediate.size() < n) intermediate.resize(n);
    int const ni = static_cast<int>(n);
    simsimd_f64c_t alpha = {1.0, 0.0}, beta = {0.0, 0.0};
    cblas_zgemv(CblasRowMajor, CblasNoTrans, ni, ni, &alpha, c, ni, b, 1, &beta, intermediate.data(), 1);
    cblas_zdotu_sub(ni, reinterpret_cast<simsimd_f64_t const *>(a), 1,
                    reinterpret_cast<simsimd_f64_t const *>(intermediate.data()), 1, results);
}

void simsimd_sum_f32_blas(simsimd_f32_t const *a, simsimd_f32_t const *b, simsimd_size_t n, simsimd_f32_t *result) {
    int const ni = static_cast<int>(n);
    cblas_scopy(ni, a, 1, result, 1);
    cblas_saxpy(ni, 1.0f, b, 1, result, 1);
}

void simsimd_sum_f64_blas(simsimd_f64_t const *a, simsimd_f64_t const *b, simsimd_size_t n, simsimd_f64_t *result) {
    int const ni = static_cast<int>(n);
    cblas_dcopy(ni, a, 1, result, 1);
    cblas_daxpy(ni, 1.0, b, 1, result, 1);
}

void simsimd_wsum_f32_blas(simsimd_f32_t const *a, simsimd_f32_t const *b, simsimd_size_t n, simsimd_distance_t alpha,
                           simsimd_distance_t beta, simsimd_f32_t *result) {
    int const ni = static_cast<int>(n);
    std::memset(result, 0, n * sizeof(simsimd_f32_t));
    if (alpha != 0) cblas_saxpy(ni, static_cast<float>(alpha), a, 1, result, 1);
    if (beta != 0) cblas_saxpy(ni, static_cast<float>(beta), b, 1, result, 1);
}

void simsimd_wsum_f64_blas(simsimd_f64_t const *a, simsimd_f64_t const *b, simsimd_size_t n, simsimd_distance_t alpha,
                           simsimd_distance_t beta, simsimd_f64_t *result) {
    int const ni = static_cast<int>(n);
    std::memset(result, 0, n * sizeof(simsimd_f64_t));
    if (alpha != 0) cblas_daxpy(ni, alpha, a, 1, result, 1);
    if (beta != 0) cblas_daxpy(ni, beta, b, 1, result, 1);
}

// SGEMM baseline for matmul comparison using OpenBLAS: F32×F32→F32
void measure_sgemm_blas(bm::State &state, std::size_t m, std::size_t n, std::size_t k) {
    std::vector<float> a(m * k), b(n * k), c(m * n);
    auto gen = make_random_engine();
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (auto &v : a) v = dis(gen);
    for (auto &v : b) v = dis(gen);

    std::size_t iterations = 0;
    for (auto _ : state) {
        bm::DoNotOptimize(c.data());
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, static_cast<int>(m), static_cast<int>(n),
                    static_cast<int>(k), 1.0f, a.data(), static_cast<int>(k), b.data(), static_cast<int>(k), 0.0f,
                    c.data(), static_cast<int>(n));
        ++iterations;
    }

    state.counters["flops"] = bm::Counter(iterations * 2.0 * m * n * k, bm::Counter::kIsRate);
}

#endif

#if SIMSIMD_BUILD_BENCHMARKS_WITH_MKL

/// Converts float to MKL_BF16 using SimSIMD's conversion.
inline MKL_BF16 f32_to_bf16(float val) {
    simsimd_bf16_t result;
    simsimd_f32_to_bf16(&val, &result);
    MKL_BF16 mkl_result;
    std::memcpy(&mkl_result, &result, sizeof(mkl_result));
    return mkl_result;
}

/// Converts float to MKL_F16 using SimSIMD's conversion.
inline MKL_F16 f32_to_f16(float val) {
    simsimd_f16_t result;
    simsimd_f32_to_f16(&val, &result);
    MKL_F16 mkl_result;
    std::memcpy(&mkl_result, &result, sizeof(mkl_result));
    return mkl_result;
}

/// Generic MKL GEMM benchmark template - reduces duplication across precision variants.
/// Pattern follows measure_matmul_packed but adds init functors for type-specific conversion.
template <typename input_a_at, typename input_b_at, typename output_at, //
          typename init_a_at, typename init_b_at, typename gemm_at>
void measure_gemm_mkl(bm::State &state, std::size_t m, std::size_t n, std::size_t k, //
                      init_a_at init_a, init_b_at init_b, gemm_at gemm_fn) {
    std::vector<input_a_at> a(m * k);
    std::vector<input_b_at> b(n * k);
    std::vector<output_at> c(m * n);
    auto gen = make_random_engine();
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    for (auto &v : a) v = init_a(dis(gen));
    for (auto &v : b) v = init_b(dis(gen));

    std::size_t iterations = 0;
    for (auto _ : state) {
        bm::DoNotOptimize(c.data());
        gemm_fn(a.data(), b.data(), c.data(), m, n, k);
        ++iterations;
    }
    state.counters["ops"] = bm::Counter(iterations * 2.0 * m * n * k, bm::Counter::kIsRate);
}

/// Overload for integer types - uses int distribution instead of float.
template <typename input_a_at, typename input_b_at, typename output_at, typename gemm_at>
void measure_gemm_mkl_int(bm::State &state, std::size_t m, std::size_t n, std::size_t k, gemm_at gemm_fn) {
    std::vector<input_a_at> a(m * k);
    std::vector<input_b_at> b(n * k);
    std::vector<output_at> c(m * n);
    auto gen = make_random_engine();
    std::uniform_int_distribution<int> dis(-64, 63);

    for (auto &v : a) v = static_cast<input_a_at>(dis(gen));
    for (auto &v : b) v = static_cast<input_b_at>(dis(gen));

    std::size_t iterations = 0;
    for (auto _ : state) {
        bm::DoNotOptimize(c.data());
        gemm_fn(a.data(), b.data(), c.data(), m, n, k);
        ++iterations;
    }
    state.counters["ops"] = bm::Counter(iterations * 2.0 * m * n * k, bm::Counter::kIsRate);
}

void measure_sgemm_mkl(bm::State &state, std::size_t m, std::size_t n, std::size_t k) {
    auto identity = [](float v) { return v; };
    measure_gemm_mkl<float, float, float>(
        state, m, n, k, identity, identity,
        [](float *a, float *b, float *c, std::size_t m, std::size_t n, std::size_t k) {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, (MKL_INT)m, (MKL_INT)n, (MKL_INT)k, 1.0f, a,
                        (MKL_INT)k, b, (MKL_INT)k, 0.0f, c, (MKL_INT)n);
        });
}

void measure_bf16gemm_mkl(bm::State &state, std::size_t m, std::size_t n, std::size_t k) {
    measure_gemm_mkl<MKL_BF16, MKL_BF16, float>(
        state, m, n, k, f32_to_bf16, f32_to_bf16,
        [](MKL_BF16 *a, MKL_BF16 *b, float *c, std::size_t m, std::size_t n, std::size_t k) {
            cblas_gemm_bf16bf16f32(CblasRowMajor, CblasNoTrans, CblasTrans, (MKL_INT)m, (MKL_INT)n, (MKL_INT)k, 1.0f, a,
                                   (MKL_INT)k, b, (MKL_INT)k, 0.0f, c, (MKL_INT)n);
        });
}

void measure_f16gemm_mkl(bm::State &state, std::size_t m, std::size_t n, std::size_t k) {
    measure_gemm_mkl<MKL_F16, MKL_F16, float>(
        state, m, n, k, f32_to_f16, f32_to_f16,
        [](MKL_F16 *a, MKL_F16 *b, float *c, std::size_t m, std::size_t n, std::size_t k) {
            cblas_gemm_f16f16f32(CblasRowMajor, CblasNoTrans, CblasTrans, (MKL_INT)m, (MKL_INT)n, (MKL_INT)k, 1.0f, a,
                                 (MKL_INT)k, b, (MKL_INT)k, 0.0f, c, (MKL_INT)n);
        });
}

void measure_s8u8s32gemm_mkl(bm::State &state, std::size_t m, std::size_t n, std::size_t k) {
    measure_gemm_mkl_int<std::uint8_t, std::int8_t, std::int32_t>(
        state, m, n, k,
        [](std::uint8_t *a, std::int8_t *b, std::int32_t *c, std::size_t m, std::size_t n, std::size_t k) {
            MKL_INT32 c_offset = 0;
            cblas_gemm_s8u8s32(CblasRowMajor, CblasNoTrans, CblasTrans, CblasFixOffset, (MKL_INT)m, (MKL_INT)n,
                               (MKL_INT)k, 1.0f, a, (MKL_INT)k, 0, b, (MKL_INT)k, 0, 0.0f, c, (MKL_INT)n, &c_offset);
        });
}

void measure_s16s16s32gemm_mkl(bm::State &state, std::size_t m, std::size_t n, std::size_t k) {
    measure_gemm_mkl_int<std::int16_t, std::int16_t, std::int32_t>(
        state, m, n, k,
        [](std::int16_t *a, std::int16_t *b, std::int32_t *c, std::size_t m, std::size_t n, std::size_t k) {
            MKL_INT32 c_offset = 0;
            cblas_gemm_s16s16s32(CblasRowMajor, CblasNoTrans, CblasTrans, CblasFixOffset, (MKL_INT)m, (MKL_INT)n,
                                 (MKL_INT)k, 1.0f, a, (MKL_INT)k, 0, b, (MKL_INT)k, 0, 0.0f, c, (MKL_INT)n, &c_offset);
        });
}

#endif

int main(int argc, char **argv) {
    simsimd_capability_t runtime_caps = simsimd_capabilities();
    simsimd_flush_denormals(runtime_caps); // Also enables AMX if available

#if SIMSIMD_BUILD_BENCHMARKS_WITH_MKL
    // Set MKL to single-threaded for fair comparison with SimSIMD (which is single-threaded)
    mkl_set_num_threads(1);
#elif SIMSIMD_BUILD_BENCHMARKS_WITH_CBLAS
    // Set OpenBLAS to single-threaded for fair comparison with SimSIMD (which is single-threaded)
    if (openblas_set_num_threads) openblas_set_num_threads(1);
#endif

    // Log supported functionality
    char const *flags[2] = {"false", "true"};
    std::printf("Benchmarking Similarity Measures\n");
    std::printf("- Compiler used native F16: %s\n", flags[SIMSIMD_NATIVE_F16]);
    std::printf("- Compiler used native BF16: %s\n", flags[SIMSIMD_NATIVE_BF16]);
    std::printf("- Benchmark against CBLAS: %s\n", flags[SIMSIMD_BUILD_BENCHMARKS_WITH_CBLAS]);
    std::printf("- Benchmark against MKL: %s\n", flags[SIMSIMD_BUILD_BENCHMARKS_WITH_MKL]);
    std::printf("\n");
    std::printf("Compile-time settings:\n");
    std::printf("- Arm NEON support enabled: %s\n", flags[SIMSIMD_TARGET_NEON]);
    std::printf("- Arm SVE support enabled: %s\n", flags[SIMSIMD_TARGET_SVE]);
    std::printf("- Arm SVE2 support enabled: %s\n", flags[SIMSIMD_TARGET_SVE2]);
    std::printf("- x86 Haswell support enabled: %s\n", flags[SIMSIMD_TARGET_HASWELL]);
    std::printf("- x86 Skylake support enabled: %s\n", flags[SIMSIMD_TARGET_SKYLAKE]);
    std::printf("- x86 Ice Lake support enabled: %s\n", flags[SIMSIMD_TARGET_ICE]);
    std::printf("- x86 Genoa support enabled: %s\n", flags[SIMSIMD_TARGET_GENOA]);
    std::printf("- x86 Sapphire Rapids support enabled: %s\n", flags[SIMSIMD_TARGET_SAPPHIRE]);
    std::printf("- x86 Turin support enabled: %s\n", flags[SIMSIMD_TARGET_TURIN]);
    std::printf("\n");
    std::printf("Run-time settings:\n");
    std::printf("- Arm NEON support enabled: %s\n", flags[(runtime_caps & simsimd_cap_neon_k) != 0]);
    std::printf("- Arm NEON F16 support enabled: %s\n", flags[(runtime_caps & simsimd_cap_neon_f16_k) != 0]);
    std::printf("- Arm NEON BF16 support enabled: %s\n", flags[(runtime_caps & simsimd_cap_neon_bf16_k) != 0]);
    std::printf("- Arm NEON I8 support enabled: %s\n", flags[(runtime_caps & simsimd_cap_neon_i8_k) != 0]);
    std::printf("- Arm SVE support enabled: %s\n", flags[(runtime_caps & simsimd_cap_sve_k) != 0]);
    std::printf("- Arm SVE F16 support enabled: %s\n", flags[(runtime_caps & simsimd_cap_sve_f16_k) != 0]);
    std::printf("- Arm SVE BF16 support enabled: %s\n", flags[(runtime_caps & simsimd_cap_sve_bf16_k) != 0]);
    std::printf("- Arm SVE I8 support enabled: %s\n", flags[(runtime_caps & simsimd_cap_sve_i8_k) != 0]);
    std::printf("- Arm SVE2 support enabled: %s\n", flags[(runtime_caps & simsimd_cap_sve2_k) != 0]);
    std::printf("- x86 Haswell support enabled: %s\n", flags[(runtime_caps & simsimd_cap_haswell_k) != 0]);
    std::printf("- x86 Skylake support enabled: %s\n", flags[(runtime_caps & simsimd_cap_skylake_k) != 0]);
    std::printf("- x86 Ice Lake support enabled: %s\n", flags[(runtime_caps & simsimd_cap_ice_k) != 0]);
    std::printf("- x86 Genoa support enabled: %s\n", flags[(runtime_caps & simsimd_cap_genoa_k) != 0]);
    std::printf("- x86 Sapphire Rapids support enabled: %s\n", flags[(runtime_caps & simsimd_cap_sapphire_k) != 0]);
    std::printf("- x86 Turin support enabled: %s\n", flags[(runtime_caps & simsimd_cap_turin_k) != 0]);
    std::printf("- x86 Sierra Forest support enabled: %s\n", flags[(runtime_caps & simsimd_cap_sierra_k) != 0]);
    std::printf("\n");

    // Override dimensions from environment variables if provided
    if (char const *env_dense = std::getenv("SIMSIMD_BENCH_DENSE_DIMENSIONS")) {
        std::size_t parsed_dense = std::atoi(env_dense);
        if (parsed_dense > 0) {
            dense_dimensions = parsed_dense;
            std::printf("Overriding `dense_dimensions` to %zu from SIMSIMD_BENCH_DENSE_DIMENSIONS\n", dense_dimensions);
        }
    }
    if (char const *env_curved = std::getenv("SIMSIMD_BENCH_CURVED_DIMENSIONS")) {
        std::size_t parsed_curved = std::atoi(env_curved);
        if (parsed_curved > 0) {
            curved_dimensions = parsed_curved;
            std::printf("Overriding `curved_dimensions` to %zu from SIMSIMD_BENCH_CURVED_DIMENSIONS\n",
                        curved_dimensions);
        }
    }
    if (char const *env_mesh = std::getenv("SIMSIMD_BENCH_MESH_DIMENSIONS")) {
        std::size_t parsed_mesh = std::atoi(env_mesh);
        if (parsed_mesh > 0) {
            mesh_dimensions = parsed_mesh;
            std::printf("Overriding `mesh_dimensions` to %zu from SIMSIMD_BENCH_MESH_DIMENSIONS\n", mesh_dimensions);
        }
    }
    if (char const *env_matmul_dimension_m = std::getenv("SIMSIMD_BENCH_MATMUL_DIMENSION_M")) {
        std::size_t parsed = std::atoi(env_matmul_dimension_m);
        if (parsed > 0) {
            matmul_dimension_m = parsed;
            std::printf("Overriding `matmul_dimension_m` to %zu from SIMSIMD_BENCH_MATMUL_DIMENSION_M\n",
                        matmul_dimension_m);
        }
    }
    if (char const *env_matmul_dimension_n = std::getenv("SIMSIMD_BENCH_MATMUL_DIMENSION_N")) {
        std::size_t parsed = std::atoi(env_matmul_dimension_n);
        if (parsed > 0) {
            matmul_dimension_n = parsed;
            std::printf("Overriding `matmul_dimension_n` to %zu from SIMSIMD_BENCH_MATMUL_DIMENSION_N\n",
                        matmul_dimension_n);
        }
    }
    if (char const *env_matmul_dimension_k = std::getenv("SIMSIMD_BENCH_MATMUL_DIMENSION_K")) {
        std::size_t parsed = std::atoi(env_matmul_dimension_k);
        if (parsed > 0) {
            matmul_dimension_k = parsed;
            std::printf("Overriding `matmul_dimension_k` to %zu from SIMSIMD_BENCH_MATMUL_DIMENSION_K\n",
                        matmul_dimension_k);
        }
    }
    if (char const *env_random_seed = std::getenv("SIMSIMD_BENCH_RANDOM_SEED")) {
        std::uint32_t parsed = static_cast<std::uint32_t>(std::atoi(env_random_seed));
        random_seed = parsed;
        std::printf("Overriding `random_seed` to %u from SIMSIMD_BENCH_RANDOM_SEED\n", random_seed);
    }
    std::printf("\n");

    // Run the benchmarks
    bm::Initialize(&argc, argv);
    if (bm::ReportUnrecognizedArguments(argc, argv)) return 1;

    constexpr simsimd_datatype_t b8_k = simsimd_b8_k;
    constexpr simsimd_datatype_t i4x2_k = simsimd_i4x2_k;
    constexpr simsimd_datatype_t i8_k = simsimd_i8_k;
    constexpr simsimd_datatype_t i16_k = simsimd_i16_k;
    constexpr simsimd_datatype_t i32_k = simsimd_i32_k;
    constexpr simsimd_datatype_t i64_k = simsimd_i64_k;
    constexpr simsimd_datatype_t u8_k = simsimd_u8_k;
    constexpr simsimd_datatype_t u16_k = simsimd_u16_k;
    constexpr simsimd_datatype_t u32_k = simsimd_u32_k;
    constexpr simsimd_datatype_t u64_k = simsimd_u64_k;
    constexpr simsimd_datatype_t f64_k = simsimd_f64_k;
    constexpr simsimd_datatype_t f32_k = simsimd_f32_k;
    constexpr simsimd_datatype_t f16_k = simsimd_f16_k;
    constexpr simsimd_datatype_t bf16_k = simsimd_bf16_k;
    constexpr simsimd_datatype_t e4m3_k = simsimd_e4m3_k;
    constexpr simsimd_datatype_t e5m2_k = simsimd_e5m2_k;
    constexpr simsimd_datatype_t f64c_k = simsimd_f64c_k;
    constexpr simsimd_datatype_t f32c_k = simsimd_f32c_k;
    constexpr simsimd_datatype_t f16c_k = simsimd_f16c_k;
    constexpr simsimd_datatype_t bf16c_k = simsimd_bf16c_k;

#if SIMSIMD_BUILD_BENCHMARKS_WITH_BLAS

    dense_<f32_k>("dot_f32_blas", dot_f32_blas, simsimd_dot_f32_accurate);
    dense_<f64_k>("dot_f64_blas", dot_f64_blas, simsimd_dot_f64_serial);
    dense_<f32c_k>("dot_f32c_blas", dot_f32c_blas, simsimd_dot_f32c_accurate);
    dense_<f64c_k>("dot_f64c_blas", dot_f64c_blas, simsimd_dot_f64c_serial);
    dense_<f32c_k>("vdot_f32c_blas", vdot_f32c_blas, simsimd_vdot_f32c_accurate);
    dense_<f64c_k>("vdot_f64c_blas", vdot_f64c_blas, simsimd_vdot_f64c_serial);

    elementwise_<f32_k, simsimd_metric_sum_k>("sum_f32_blas", simsimd_sum_f32_blas, simsimd_sum_f32_accurate,
                                              simsimd_l2_f32_accurate);
    elementwise_<f32_k, simsimd_metric_wsum_k>("wsum_f32_blas", simsimd_wsum_f32_blas, simsimd_wsum_f32_accurate,
                                               simsimd_l2_f32_accurate);
    elementwise_<f64_k, simsimd_metric_sum_k>("sum_f64_blas", simsimd_sum_f64_blas, simsimd_sum_f64_serial,
                                              simsimd_l2_f64_serial);
    elementwise_<f64_k, simsimd_metric_wsum_k>("wsum_f64_blas", simsimd_wsum_f64_blas, simsimd_wsum_f64_serial,
                                               simsimd_l2_f64_serial);

    curved_<f64_k>("bilinear_f64_blas", bilinear_f64_blas, simsimd_bilinear_f64_serial);
    curved_<f64c_k>("bilinear_f64c_blas", bilinear_f64c_blas, simsimd_bilinear_f64c_serial);
    curved_<f32_k>("bilinear_f32_blas", bilinear_f32_blas, simsimd_bilinear_f32_accurate);
    curved_<f32c_k>("bilinear_f32c_blas", bilinear_f32c_blas, simsimd_bilinear_f32c_accurate);

    // SGEMM baseline for matmul comparison (FP32, same layout as SimSIMD: A×Bᵀ)
    {
        std::string bench_name = "sgemm_blas<" + std::to_string(matmul_dimension_m) + "x" +
                                 std::to_string(matmul_dimension_n) + "x" + std::to_string(matmul_dimension_k) + ">";
        bm::RegisterBenchmark(bench_name.c_str(), measure_sgemm_blas, matmul_dimension_m, matmul_dimension_n,
                              matmul_dimension_k)
            ->MinTime(default_seconds)
            ->Threads(1);
    }

#endif

#if SIMSIMD_BUILD_BENCHMARKS_WITH_MKL
    // MKL mixed-precision GEMM baselines for matmul comparison
    {
        std::string dims = std::to_string(matmul_dimension_m) + "x" + std::to_string(matmul_dimension_n) + "x" +
                           std::to_string(matmul_dimension_k);
        bm::RegisterBenchmark(("sgemm_mkl<" + dims + ">").c_str(), measure_sgemm_mkl, matmul_dimension_m,
                              matmul_dimension_n, matmul_dimension_k)
            ->MinTime(default_seconds)
            ->Threads(1);
        bm::RegisterBenchmark(("bf16gemm_mkl<" + dims + ">").c_str(), measure_bf16gemm_mkl, matmul_dimension_m,
                              matmul_dimension_n, matmul_dimension_k)
            ->MinTime(default_seconds)
            ->Threads(1);
        bm::RegisterBenchmark(("f16gemm_mkl<" + dims + ">").c_str(), measure_f16gemm_mkl, matmul_dimension_m,
                              matmul_dimension_n, matmul_dimension_k)
            ->MinTime(default_seconds)
            ->Threads(1);
        bm::RegisterBenchmark(("s8u8s32gemm_mkl<" + dims + ">").c_str(), measure_s8u8s32gemm_mkl, matmul_dimension_m,
                              matmul_dimension_n, matmul_dimension_k)
            ->MinTime(default_seconds)
            ->Threads(1);
        bm::RegisterBenchmark(("s16s16s32gemm_mkl<" + dims + ">").c_str(), measure_s16s16s32gemm_mkl,
                              matmul_dimension_m, matmul_dimension_n, matmul_dimension_k)
            ->MinTime(default_seconds)
            ->Threads(1);
    }
#endif

#if SIMSIMD_TARGET_NEON
    dense_<f32_k>("dot_f32_neon", simsimd_dot_f32_neon, simsimd_dot_f32_accurate);
    dense_<f32_k>("angular_f32_neon", simsimd_angular_f32_neon, simsimd_angular_f32_accurate);
    dense_<f32_k>("l2sq_f32_neon", simsimd_l2sq_f32_neon, simsimd_l2sq_f32_accurate);
    dense_<f32_k>("l2_f32_neon", simsimd_l2_f32_neon, simsimd_l2_f32_accurate);
    dense_<f32_k>("kld_f32_neon", simsimd_kld_f32_neon, simsimd_kld_f32_accurate);
    dense_<f32_k>("jsd_f32_neon", simsimd_jsd_f32_neon, simsimd_jsd_f32_accurate);

    dense_<f64_k>("angular_f64_neon", simsimd_angular_f64_neon, simsimd_angular_f64_serial);
    dense_<f64_k>("l2sq_f64_neon", simsimd_l2sq_f64_neon, simsimd_l2sq_f64_serial);
    dense_<f64_k>("l2_f64_neon", simsimd_l2_f64_neon, simsimd_l2_f64_serial);

    dense_<i8_k>("angular_i8_neon", simsimd_angular_i8_neon, simsimd_angular_i8_serial);
    dense_<i8_k>("l2sq_i8_neon", simsimd_l2sq_i8_neon, simsimd_l2sq_i8_serial);
    dense_<i8_k>("l2_i8_neon", simsimd_l2_i8_neon, simsimd_l2_i8_serial);
    dense_<i8_k>("dot_i8_neon", simsimd_dot_i8_neon, simsimd_dot_i8_serial);

    dense_<u8_k>("angular_u8_neon", simsimd_angular_u8_neon, simsimd_angular_u8_serial);
    dense_<u8_k>("l2sq_u8_neon", simsimd_l2sq_u8_neon, simsimd_l2sq_u8_serial);
    dense_<u8_k>("l2_u8_neon", simsimd_l2_u8_neon, simsimd_l2_u8_serial);
    dense_<u8_k>("dot_u8_neon", simsimd_dot_u8_neon, simsimd_dot_u8_serial);

    dense_<b8_k>("hamming_b8_neon", simsimd_hamming_b8_neon, simsimd_hamming_b8_serial);
    dense_<b8_k>("jaccard_b8_neon", simsimd_jaccard_b8_neon, simsimd_jaccard_b8_serial);

    dense_<f32c_k>("dot_f32c_neon", simsimd_dot_f32c_neon, simsimd_dot_f32c_accurate);
    dense_<f32c_k>("vdot_f32c_neon", simsimd_vdot_f32c_neon, simsimd_vdot_f32c_accurate);

    curved_<f32_k>("bilinear_f32_neon", simsimd_bilinear_f32_neon, simsimd_bilinear_f32_accurate);
    curved_<f32_k>("mahalanobis_f32_neon", simsimd_mahalanobis_f32_neon, simsimd_mahalanobis_f32_accurate);
    curved_<f32c_k>("bilinear_f32c_neon", simsimd_bilinear_f32c_neon, simsimd_bilinear_f32c_accurate);

    sparse_<u16_k>("intersect_u16_neon", simsimd_intersect_u16_neon, simsimd_intersect_u16_accurate);
    sparse_<u32_k>("intersect_u32_neon", simsimd_intersect_u32_neon, simsimd_intersect_u32_accurate);

    elementwise_<f32_k, simsimd_metric_fma_k>("fma_f32_neon", simsimd_fma_f32_neon, simsimd_fma_f32_accurate,
                                              simsimd_l2_f32_accurate);
    elementwise_<f32_k, simsimd_metric_wsum_k>("wsum_f32_neon", simsimd_wsum_f32_neon, simsimd_wsum_f32_accurate,
                                               simsimd_l2_f32_accurate);
    elementwise_<f32_k, simsimd_metric_fma_k>("fma_f32_serial", simsimd_fma_f32_serial, simsimd_fma_f32_accurate,
                                              simsimd_l2_f32_accurate);
    elementwise_<f32_k, simsimd_metric_wsum_k>("wsum_f32_serial", simsimd_wsum_f32_serial, simsimd_wsum_f32_accurate,
                                               simsimd_l2_f32_accurate);

#endif

#if SIMSIMD_TARGET_NEON_F16
    dense_<f16c_k>("dot_f16c_neon", simsimd_dot_f16c_neon, simsimd_dot_f16c_accurate);
    dense_<f16c_k>("vdot_f16c_neon", simsimd_vdot_f16c_neon, simsimd_vdot_f16c_accurate);

    dense_<f16_k>("dot_f16_neon", simsimd_dot_f16_neon, simsimd_dot_f16_accurate);
    dense_<f16_k>("angular_f16_neon", simsimd_angular_f16_neon, simsimd_angular_f16_accurate);
    dense_<f16_k>("l2sq_f16_neon", simsimd_l2sq_f16_neon, simsimd_l2sq_f16_accurate);
    dense_<f16_k>("l2_f16_neon", simsimd_l2_f16_neon, simsimd_l2sq_f16_accurate);
    dense_<f16_k>("kld_f16_neon", simsimd_kld_f16_neon, simsimd_kld_f16_accurate);
    dense_<f16_k>("jsd_f16_neon", simsimd_jsd_f16_neon, simsimd_jsd_f16_accurate);

    curved_<f16_k>("bilinear_f16_neon", simsimd_bilinear_f16_neon, simsimd_bilinear_f16_accurate);
    curved_<f16_k>("mahalanobis_f16_neon", simsimd_mahalanobis_f16_neon, simsimd_mahalanobis_f16_accurate);
    curved_<f16c_k>("bilinear_f16c_neon", simsimd_bilinear_f16c_neon, simsimd_bilinear_f16c_accurate);

    elementwise_<f16_k, simsimd_metric_fma_k>("fma_f16_neon", simsimd_fma_f16_neon, simsimd_fma_f16_accurate,
                                              simsimd_l2_f16_accurate);
    elementwise_<f16_k, simsimd_metric_wsum_k>("wsum_f16_neon", simsimd_wsum_f16_neon, simsimd_wsum_f16_accurate,
                                               simsimd_l2_f16_accurate);

    // FMA kernels for `u8` on NEON use `f16` arithmetic
    elementwise_<u8_k, simsimd_metric_fma_k>("fma_u8_neon", simsimd_fma_u8_neon, simsimd_fma_u8_accurate,
                                             simsimd_l2_u8_serial);
    elementwise_<u8_k, simsimd_metric_wsum_k>("wsum_u8_neon", simsimd_wsum_u8_neon, simsimd_wsum_u8_accurate,
                                              simsimd_l2_u8_serial);
    elementwise_<i8_k, simsimd_metric_fma_k>("fma_i8_neon", simsimd_fma_i8_neon, simsimd_fma_i8_accurate,
                                             simsimd_l2_i8_serial);
    elementwise_<i8_k, simsimd_metric_wsum_k>("wsum_i8_neon", simsimd_wsum_i8_neon, simsimd_wsum_i8_accurate,
                                              simsimd_l2_i8_serial);
#endif

#if SIMSIMD_TARGET_NEON_BF16
    dense_<bf16c_k>("dot_bf16c_neon", simsimd_dot_bf16c_neon, simsimd_dot_bf16c_accurate);
    dense_<bf16c_k>("vdot_bf16c_neon", simsimd_vdot_bf16c_neon, simsimd_vdot_bf16c_accurate);

    dense_<bf16_k>("dot_bf16_neon", simsimd_dot_bf16_neon, simsimd_dot_bf16_accurate);
    dense_<bf16_k>("angular_bf16_neon", simsimd_angular_bf16_neon, simsimd_angular_bf16_accurate);
    dense_<bf16_k>("l2sq_bf16_neon", simsimd_l2sq_bf16_neon, simsimd_l2sq_bf16_accurate);
    dense_<bf16_k>("l2_bf16_neon", simsimd_l2_bf16_neon, simsimd_l2_bf16_accurate);

    curved_<bf16_k>("bilinear_bf16_neon", simsimd_bilinear_bf16_neon, simsimd_bilinear_bf16_accurate);
    curved_<bf16_k>("mahalanobis_bf16_neon", simsimd_mahalanobis_bf16_neon, simsimd_mahalanobis_bf16_accurate);
    curved_<bf16c_k>("bilinear_bf16c_neon", simsimd_bilinear_bf16c_neon, simsimd_bilinear_bf16c_accurate);

    elementwise_<bf16_k, simsimd_metric_fma_k>("fma_bf16_neon", simsimd_fma_bf16_neon, simsimd_fma_bf16_accurate,
                                               simsimd_l2_bf16_accurate);
    elementwise_<bf16_k, simsimd_metric_wsum_k>("wsum_bf16_neon", simsimd_wsum_bf16_neon, simsimd_wsum_bf16_accurate,
                                                simsimd_l2_bf16_accurate);
#endif

#if SIMSIMD_TARGET_SVE
    dense_<f16_k>("dot_f16_sve", simsimd_dot_f16_sve, simsimd_dot_f16_accurate);
    dense_<f16_k>("angular_f16_sve", simsimd_angular_f16_sve, simsimd_angular_f16_accurate);
    dense_<f16_k>("l2sq_f16_sve", simsimd_l2sq_f16_sve, simsimd_l2sq_f16_accurate);
    dense_<f16_k>("l2_f16_sve", simsimd_l2_f16_sve, simsimd_l2_f16_accurate);

    dense_<f32_k>("dot_f32_sve", simsimd_dot_f32_sve, simsimd_dot_f32_accurate);
    dense_<f32_k>("angular_f32_sve", simsimd_angular_f32_sve, simsimd_angular_f32_accurate);
    dense_<f32_k>("l2sq_f32_sve", simsimd_l2sq_f32_sve, simsimd_l2sq_f32_accurate);
    dense_<f32_k>("l2_f32_sve", simsimd_l2_f32_sve, simsimd_l2_f32_accurate);

    dense_<f64_k>("dot_f64_sve", simsimd_dot_f64_sve, simsimd_dot_f64_serial);
    dense_<f64_k>("angular_f64_sve", simsimd_angular_f64_sve, simsimd_angular_f64_serial);
    dense_<f64_k>("l2sq_f64_sve", simsimd_l2sq_f64_sve, simsimd_l2sq_f64_serial);
    dense_<f64_k>("l2_f64_sve", simsimd_l2_f64_sve, simsimd_l2_f64_serial);

    dense_<b8_k>("hamming_b8_sve", simsimd_hamming_b8_sve, simsimd_hamming_b8_serial);
    dense_<b8_k>("jaccard_b8_sve", simsimd_jaccard_b8_sve, simsimd_jaccard_b8_serial);

    dense_<f32c_k>("dot_f32c_sve", simsimd_dot_f32c_sve, simsimd_dot_f32c_accurate);
    dense_<f32c_k>("vdot_f32c_sve", simsimd_vdot_f32c_sve, simsimd_vdot_f32c_accurate);
    dense_<f64c_k>("dot_f64c_sve", simsimd_dot_f64c_sve, simsimd_dot_f64c_serial);
    dense_<f64c_k>("vdot_f64c_sve", simsimd_vdot_f64c_sve, simsimd_vdot_f64c_serial);
#endif

#if SIMSIMD_TARGET_SVE_F16
    dense_<f16_k>("dot_f16_sve", simsimd_dot_f16_sve, simsimd_dot_f16_accurate);
    dense_<f16_k>("angular_f16_sve", simsimd_angular_f16_sve, simsimd_angular_f16_accurate);
    dense_<f16_k>("l2sq_f16_sve", simsimd_l2sq_f16_sve, simsimd_l2sq_f16_accurate);
    dense_<f16_k>("l2_f16_sve", simsimd_l2_f16_sve, simsimd_l2sq_f16_accurate);
    dense_<f16c_k>("dot_f16c_sve", simsimd_dot_f16c_sve, simsimd_dot_f16c_accurate);
    dense_<f16c_k>("vdot_f16c_sve", simsimd_vdot_f16c_sve, simsimd_vdot_f16c_accurate);
#endif

#if SIMSIMD_TARGET_SVE_BF16
    dense_<bf16_k>("angular_bf16_sve", simsimd_angular_bf16_sve, simsimd_angular_bf16_accurate);
    dense_<bf16_k>("l2sq_bf16_sve", simsimd_l2sq_bf16_sve, simsimd_l2sq_bf16_accurate);
    dense_<bf16_k>("l2_bf16_sve", simsimd_l2_bf16_sve, simsimd_l2_bf16_accurate);
#endif

#if SIMSIMD_TARGET_SVE2
    sparse_<u16_k>("intersect_u16_sve2", simsimd_intersect_u16_sve2, simsimd_intersect_u16_accurate);
    sparse_<u32_k>("intersect_u32_sve2", simsimd_intersect_u32_sve2, simsimd_intersect_u32_accurate);
#endif

#if SIMSIMD_TARGET_HASWELL
    dense_<f16_k>("dot_f16_haswell", simsimd_dot_f16_haswell, simsimd_dot_f16_accurate);
    dense_<f16_k>("angular_f16_haswell", simsimd_angular_f16_haswell, simsimd_angular_f16_accurate);
    dense_<f16_k>("l2sq_f16_haswell", simsimd_l2sq_f16_haswell, simsimd_l2sq_f16_accurate);
    dense_<f16_k>("l2_f16_haswell", simsimd_l2_f16_haswell, simsimd_l2_f16_accurate);
    dense_<f16_k>("kld_f16_haswell", simsimd_kld_f16_haswell, simsimd_kld_f16_accurate);
    dense_<f16_k>("jsd_f16_haswell", simsimd_jsd_f16_haswell, simsimd_jsd_f16_accurate);

    dense_<bf16_k>("dot_bf16_haswell", simsimd_dot_bf16_haswell, simsimd_dot_bf16_accurate);
    dense_<bf16_k>("angular_bf16_haswell", simsimd_angular_bf16_haswell, simsimd_angular_bf16_accurate);
    dense_<bf16_k>("l2sq_bf16_haswell", simsimd_l2sq_bf16_haswell, simsimd_l2sq_bf16_accurate);
    dense_<bf16_k>("l2_bf16_haswell", simsimd_l2_bf16_haswell, simsimd_l2_bf16_accurate);

    dense_<e4m3_k>("dot_e4m3_haswell", simsimd_dot_e4m3_haswell, simsimd_dot_e4m3_serial);
    dense_<e5m2_k>("dot_e5m2_haswell", simsimd_dot_e5m2_haswell, simsimd_dot_e5m2_serial);

    dense_<i8_k>("angular_i8_haswell", simsimd_angular_i8_haswell, simsimd_angular_i8_serial);
    dense_<i8_k>("l2sq_i8_haswell", simsimd_l2sq_i8_haswell, simsimd_l2sq_i8_serial);
    dense_<i8_k>("l2_i8_haswell", simsimd_l2_i8_haswell, simsimd_l2_i8_serial);
    dense_<i8_k>("dot_i8_haswell", simsimd_dot_i8_haswell, simsimd_dot_i8_serial);

    dense_<u8_k>("angular_u8_haswell", simsimd_angular_u8_haswell, simsimd_angular_u8_serial);
    dense_<u8_k>("l2sq_u8_haswell", simsimd_l2sq_u8_haswell, simsimd_l2sq_u8_serial);
    dense_<u8_k>("l2_u8_haswell", simsimd_l2_u8_haswell, simsimd_l2_u8_serial);
    dense_<u8_k>("dot_u8_haswell", simsimd_dot_u8_haswell, simsimd_dot_u8_serial);

    dense_<b8_k>("hamming_b8_haswell", simsimd_hamming_b8_haswell, simsimd_hamming_b8_serial);
    dense_<b8_k>("jaccard_b8_haswell", simsimd_jaccard_b8_haswell, simsimd_jaccard_b8_serial);

    dense_<f16c_k>("dot_f16c_haswell", simsimd_dot_f16c_haswell, simsimd_dot_f16c_accurate);
    dense_<f16c_k>("vdot_f16c_haswell", simsimd_vdot_f16c_haswell, simsimd_vdot_f16c_accurate);
    dense_<f32c_k>("dot_f32c_haswell", simsimd_dot_f32c_haswell, simsimd_dot_f32c_accurate);
    dense_<f32c_k>("vdot_f32c_haswell", simsimd_vdot_f32c_haswell, simsimd_vdot_f32c_accurate);

    curved_<f16_k>("bilinear_f16_haswell", simsimd_bilinear_f16_haswell, simsimd_bilinear_f16_accurate);
    curved_<f16_k>("mahalanobis_f16_haswell", simsimd_mahalanobis_f16_haswell, simsimd_mahalanobis_f16_accurate);
    curved_<bf16_k>("bilinear_bf16_haswell", simsimd_bilinear_bf16_haswell, simsimd_bilinear_bf16_accurate);
    curved_<bf16_k>("mahalanobis_bf16_haswell", simsimd_mahalanobis_bf16_haswell, simsimd_mahalanobis_bf16_accurate);

    elementwise_<f64_k, simsimd_metric_scale_k>("scale_f64_haswell", simsimd_scale_f64_haswell,
                                                simsimd_scale_f64_serial, simsimd_l2_f64_serial);
    elementwise_<f64_k, simsimd_metric_fma_k>("fma_f64_haswell", simsimd_fma_f64_haswell, simsimd_fma_f64_serial,
                                              simsimd_l2_f64_serial);
    elementwise_<f64_k, simsimd_metric_wsum_k>("wsum_f64_haswell", simsimd_wsum_f64_haswell, simsimd_wsum_f64_serial,
                                               simsimd_l2_f64_serial);
    elementwise_<f32_k, simsimd_metric_scale_k>("scale_f32_haswell", simsimd_scale_f32_haswell,
                                                simsimd_scale_f32_serial, simsimd_l2_f32_accurate);
    elementwise_<f32_k, simsimd_metric_fma_k>("fma_f32_haswell", simsimd_fma_f32_haswell, simsimd_fma_f32_serial,
                                              simsimd_l2_f32_accurate);
    elementwise_<f32_k, simsimd_metric_wsum_k>("wsum_f32_haswell", simsimd_wsum_f32_haswell, simsimd_wsum_f32_serial,
                                               simsimd_l2_f32_accurate);
    elementwise_<f16_k, simsimd_metric_scale_k>("scale_f16_haswell", simsimd_scale_f16_haswell,
                                                simsimd_scale_f16_serial, simsimd_l2_f16_accurate);
    elementwise_<f16_k, simsimd_metric_fma_k>("fma_f16_haswell", simsimd_fma_f16_haswell, simsimd_fma_f16_serial,
                                              simsimd_l2_f16_accurate);
    elementwise_<f16_k, simsimd_metric_wsum_k>("wsum_f16_haswell", simsimd_wsum_f16_haswell, simsimd_wsum_f16_serial,
                                               simsimd_l2_f16_accurate);
    elementwise_<bf16_k, simsimd_metric_scale_k>("scale_bf16_haswell", simsimd_scale_bf16_haswell,
                                                 simsimd_scale_bf16_serial, simsimd_l2_bf16_accurate);
    elementwise_<bf16_k, simsimd_metric_fma_k>("fma_bf16_haswell", simsimd_fma_bf16_haswell, simsimd_fma_bf16_serial,
                                               simsimd_l2_bf16_accurate);
    elementwise_<bf16_k, simsimd_metric_wsum_k>("wsum_bf16_haswell", simsimd_wsum_bf16_haswell,
                                                simsimd_wsum_bf16_serial, simsimd_l2_bf16_accurate);
    elementwise_<i8_k, simsimd_metric_scale_k>("scale_i8_haswell", simsimd_scale_i8_haswell, simsimd_scale_i8_serial,
                                               simsimd_l2_i8_serial);
    elementwise_<i8_k, simsimd_metric_fma_k>("fma_i8_haswell", simsimd_fma_i8_haswell, simsimd_fma_i8_serial,
                                             simsimd_l2_i8_serial);
    elementwise_<i8_k, simsimd_metric_wsum_k>("wsum_i8_haswell", simsimd_wsum_i8_haswell, simsimd_wsum_i8_serial,
                                              simsimd_l2_i8_serial);
    elementwise_<u8_k, simsimd_metric_scale_k>("scale_u8_haswell", simsimd_scale_u8_haswell, simsimd_scale_u8_serial,
                                               simsimd_l2_u8_serial);
    elementwise_<u8_k, simsimd_metric_fma_k>("fma_u8_haswell", simsimd_fma_u8_haswell, simsimd_fma_u8_serial,
                                             simsimd_l2_u8_serial);
    elementwise_<u8_k, simsimd_metric_wsum_k>("wsum_u8_haswell", simsimd_wsum_u8_haswell, simsimd_wsum_u8_serial,
                                              simsimd_l2_u8_serial);
    elementwise_<i16_k, simsimd_metric_scale_k>("scale_i16_haswell", simsimd_scale_i16_haswell,
                                                simsimd_scale_i16_serial, l2_with_stl<simsimd_i16_t>);
    elementwise_<i16_k, simsimd_metric_fma_k>("fma_i16_haswell", simsimd_fma_i16_haswell, simsimd_fma_i16_serial,
                                              l2_with_stl<simsimd_i16_t>);
    elementwise_<u16_k, simsimd_metric_scale_k>("scale_u16_haswell", simsimd_scale_u16_haswell,
                                                simsimd_scale_u16_serial, l2_with_stl<simsimd_u16_t>);
    elementwise_<u16_k, simsimd_metric_fma_k>("fma_u16_haswell", simsimd_fma_u16_haswell, simsimd_fma_u16_serial,
                                              l2_with_stl<simsimd_u16_t>);

    geospatial_<f32_k>("haversine_f32_haswell", simsimd_haversine_f32_haswell,
                       haversine_with_stl<simsimd_f32_t, simsimd_f64_t>, l2_with_stl<simsimd_f64_t>);
    geospatial_<f64_k>("haversine_f64_haswell", simsimd_haversine_f64_haswell, haversine_with_stl<simsimd_f64_t>,
                       l2_with_stl<simsimd_f64_t>);
    geospatial_<f32_k>("vincenty_f32_haswell", simsimd_vincenty_f32_haswell,
                       vincenty_with_stl<simsimd_f32_t, simsimd_f64_t>, l2_with_stl<simsimd_f64_t>);
    geospatial_<f64_k>("vincenty_f64_haswell", simsimd_vincenty_f64_haswell, vincenty_with_stl<simsimd_f64_t>,
                       l2_with_stl<simsimd_f64_t>);

    // Trigonometry benchmarks for Haswell
    elementwise_<f32_k>("sin_f32_haswell", simsimd_sin_f32_haswell,
                        elementwise_with_stl<simsimd_f32_t, sin_with_stl<simsimd_f64_t>>, l2_with_stl<simsimd_f32_t>);
    elementwise_<f32_k>("cos_f32_haswell", simsimd_cos_f32_haswell,
                        elementwise_with_stl<simsimd_f32_t, cos_with_stl<simsimd_f64_t>>, l2_with_stl<simsimd_f32_t>);
    elementwise_<f32_k>("atan_f32_haswell", simsimd_atan_f32_haswell,
                        elementwise_with_stl<simsimd_f32_t, atan_with_stl<simsimd_f64_t>>, l2_with_stl<simsimd_f32_t>);
    elementwise_<f64_k>("sin_f64_haswell", simsimd_sin_f64_haswell,
                        elementwise_with_stl<simsimd_f64_t, sin_with_stl<simsimd_f64_t>>, l2_with_stl<simsimd_f64_t>);
    elementwise_<f64_k>("cos_f64_haswell", simsimd_cos_f64_haswell,
                        elementwise_with_stl<simsimd_f64_t, cos_with_stl<simsimd_f64_t>>, l2_with_stl<simsimd_f64_t>);
    elementwise_<f64_k>("atan_f64_haswell", simsimd_atan_f64_haswell,
                        elementwise_with_stl<simsimd_f64_t, atan_with_stl<simsimd_f64_t>>, l2_with_stl<simsimd_f64_t>);

#endif

#if SIMSIMD_TARGET_SKYLAKE
    dense_<f32_k>("dot_f32_skylake", simsimd_dot_f32_skylake, simsimd_dot_f32_accurate);
    dense_<f32_k>("angular_f32_skylake", simsimd_angular_f32_skylake, simsimd_angular_f32_accurate);
    dense_<f32_k>("l2sq_f32_skylake", simsimd_l2sq_f32_skylake, simsimd_l2sq_f32_accurate);
    dense_<f32_k>("l2_f32_skylake", simsimd_l2_f32_skylake, simsimd_l2_f32_accurate);
    dense_<f32_k>("kld_f32_skylake", simsimd_kld_f32_skylake, simsimd_kld_f32_accurate);
    dense_<f32_k>("jsd_f32_skylake", simsimd_jsd_f32_skylake, simsimd_jsd_f32_accurate);

    dense_<f32c_k>("dot_f32c_skylake", simsimd_dot_f32c_skylake, simsimd_dot_f32c_accurate);
    dense_<f32c_k>("vdot_f32c_skylake", simsimd_vdot_f32c_skylake, simsimd_vdot_f32c_accurate);
    dense_<f64c_k>("dot_f64c_skylake", simsimd_dot_f64c_skylake, simsimd_dot_f64c_serial);
    dense_<f64c_k>("vdot_f64c_skylake", simsimd_vdot_f64c_skylake, simsimd_vdot_f64c_serial);

    dense_<e4m3_k>("dot_e4m3_skylake", simsimd_dot_e4m3_skylake, simsimd_dot_e4m3_serial);
    dense_<e5m2_k>("dot_e5m2_skylake", simsimd_dot_e5m2_skylake, simsimd_dot_e5m2_serial);

    elementwise_<f64_k, simsimd_metric_fma_k>("fma_f64_skylake", simsimd_fma_f64_skylake, simsimd_fma_f64_serial,
                                              simsimd_l2_f64_serial);
    elementwise_<f64_k, simsimd_metric_wsum_k>("wsum_f64_skylake", simsimd_wsum_f64_skylake, simsimd_wsum_f64_serial,
                                               simsimd_l2_f64_serial);
    elementwise_<f32_k, simsimd_metric_fma_k>("fma_f32_skylake", simsimd_fma_f32_skylake, simsimd_fma_f32_serial,
                                              simsimd_l2_f32_accurate);
    elementwise_<f32_k, simsimd_metric_wsum_k>("wsum_f32_skylake", simsimd_wsum_f32_skylake, simsimd_wsum_f32_serial,
                                               simsimd_l2_f32_accurate);
    elementwise_<bf16_k, simsimd_metric_fma_k>("fma_bf16_skylake", simsimd_fma_bf16_skylake, simsimd_fma_bf16_serial,
                                               simsimd_l2_bf16_accurate);
    elementwise_<bf16_k, simsimd_metric_wsum_k>("wsum_bf16_skylake", simsimd_wsum_bf16_skylake,
                                                simsimd_wsum_bf16_serial, simsimd_l2_bf16_accurate);

    elementwise_<f32_k>("sin_f32_skylake", simsimd_sin_f32_skylake,
                        elementwise_with_stl<simsimd_f32_t, sin_with_stl<simsimd_f64_t>>, l2_with_stl<simsimd_f32_t>);
    elementwise_<f32_k>("cos_f32_skylake", simsimd_cos_f32_skylake,
                        elementwise_with_stl<simsimd_f32_t, cos_with_stl<simsimd_f64_t>>, l2_with_stl<simsimd_f32_t>);
    elementwise_<f32_k>("atan_f32_skylake", simsimd_atan_f32_skylake,
                        elementwise_with_stl<simsimd_f32_t, atan_with_stl<simsimd_f64_t>>, l2_with_stl<simsimd_f32_t>);
    elementwise_<f64_k>("sin_f64_skylake", simsimd_sin_f64_skylake,
                        elementwise_with_stl<simsimd_f64_t, sin_with_stl<simsimd_f64_t>>, l2_with_stl<simsimd_f64_t>);
    elementwise_<f64_k>("cos_f64_skylake", simsimd_cos_f64_skylake,
                        elementwise_with_stl<simsimd_f64_t, cos_with_stl<simsimd_f64_t>>, l2_with_stl<simsimd_f64_t>);
    elementwise_<f64_k>("atan_f64_skylake", simsimd_atan_f64_skylake,
                        elementwise_with_stl<simsimd_f64_t, atan_with_stl<simsimd_f64_t>>, l2_with_stl<simsimd_f64_t>);

    curved_<f32_k>("bilinear_f32_skylake", simsimd_bilinear_f32_skylake, simsimd_bilinear_f32_serial);
    curved_<f32c_k>("bilinear_f32c_skylake", simsimd_bilinear_f32c_skylake, simsimd_bilinear_f32c_serial);
    curved_<f64_k>("bilinear_f64_skylake", simsimd_bilinear_f64_skylake, simsimd_bilinear_f64_serial);
    curved_<f64c_k>("bilinear_f64c_skylake", simsimd_bilinear_f64c_skylake, simsimd_bilinear_f64c_serial);

    geospatial_<f32_k>("haversine_f32_skylake", simsimd_haversine_f32_skylake,
                       haversine_with_stl<simsimd_f32_t, simsimd_f64_t>, l2_with_stl<simsimd_f64_t>);
    geospatial_<f64_k>("haversine_f64_skylake", simsimd_haversine_f64_skylake, haversine_with_stl<simsimd_f64_t>,
                       l2_with_stl<simsimd_f64_t>);
    geospatial_<f32_k>("vincenty_f32_skylake", simsimd_vincenty_f32_skylake,
                       vincenty_with_stl<simsimd_f32_t, simsimd_f64_t>, l2_with_stl<simsimd_f64_t>);
    geospatial_<f64_k>("vincenty_f64_skylake", simsimd_vincenty_f64_skylake, vincenty_with_stl<simsimd_f64_t>,
                       l2_with_stl<simsimd_f64_t>);

    mesh_<f32_k>("rmsd_f32_skylake", simsimd_rmsd_f32_skylake, simsimd_rmsd_f32_serial);
    mesh_<f32_k>("kabsch_f32_skylake", simsimd_kabsch_f32_skylake, simsimd_kabsch_f32_serial);

    matmul_<simsimd_f32_t, simsimd_f32_t>("dots_f32f32f32_skylake", simsimd_dots_f32f32f32_packed_size_skylake,
                                          simsimd_dots_f32f32f32_pack_skylake, simsimd_dots_f32f32f32_skylake);

#endif

#if SIMSIMD_TARGET_ICE
    matmul_<simsimd_i8_t, simsimd_i32_t>("dots_i8i8i32_ice", simsimd_dots_i8i8i32_packed_size_ice,
                                         simsimd_dots_i8i8i32_pack_ice, simsimd_dots_i8i8i32_ice);

    dense_<i8_k>("angular_i8_ice", simsimd_angular_i8_ice, simsimd_angular_i8_serial);
    dense_<i8_k>("l2sq_i8_ice", simsimd_l2sq_i8_ice, simsimd_l2sq_i8_serial);
    dense_<i8_k>("l2_i8_ice", simsimd_l2_i8_ice, simsimd_l2_i8_serial);
    dense_<i8_k>("dot_i8_ice", simsimd_dot_i8_ice, simsimd_dot_i8_serial);

    dense_<u8_k>("angular_u8_ice", simsimd_angular_u8_ice, simsimd_angular_u8_serial);
    dense_<u8_k>("l2sq_u8_ice", simsimd_l2sq_u8_ice, simsimd_l2sq_u8_serial);
    dense_<u8_k>("l2_u8_ice", simsimd_l2_u8_ice, simsimd_l2_u8_serial);
    dense_<u8_k>("dot_u8_ice", simsimd_dot_u8_ice, simsimd_dot_u8_serial);

    dense_<f64_k>("dot_f64_skylake", simsimd_dot_f64_skylake, simsimd_dot_f64_serial);
    dense_<f64_k>("angular_f64_skylake", simsimd_angular_f64_skylake, simsimd_angular_f64_serial);
    dense_<f64_k>("l2sq_f64_skylake", simsimd_l2sq_f64_skylake, simsimd_l2sq_f64_serial);
    dense_<f64_k>("l2_f64_skylake", simsimd_l2_f64_skylake, simsimd_l2_f64_serial);

    dense_<b8_k>("hamming_b8_ice", simsimd_hamming_b8_ice, simsimd_hamming_b8_serial);
    dense_<b8_k>("jaccard_b8_ice", simsimd_jaccard_b8_ice, simsimd_jaccard_b8_serial);

    sparse_<u16_k>("intersect_u16_ice", simsimd_intersect_u16_ice, simsimd_intersect_u16_accurate);
    sparse_<u32_k>("intersect_u32_ice", simsimd_intersect_u32_ice, simsimd_intersect_u32_accurate);
#endif

#if SIMSIMD_TARGET_GENOA
    dense_<bf16_k>("dot_bf16_genoa", simsimd_dot_bf16_genoa, simsimd_dot_bf16_accurate);
    dense_<bf16_k>("angular_bf16_genoa", simsimd_angular_bf16_genoa, simsimd_angular_bf16_accurate);
    dense_<bf16_k>("l2sq_bf16_genoa", simsimd_l2sq_bf16_genoa, simsimd_l2sq_bf16_accurate);
    dense_<bf16_k>("l2_bf16_genoa", simsimd_l2_bf16_genoa, simsimd_l2_bf16_accurate);
    dense_<bf16c_k>("dot_bf16c_genoa", simsimd_dot_bf16c_genoa, simsimd_dot_bf16c_accurate);
    dense_<bf16c_k>("vdot_bf16c_genoa", simsimd_vdot_bf16c_genoa, simsimd_vdot_bf16c_accurate);

    dense_<e4m3_k>("dot_e4m3_genoa", simsimd_dot_e4m3_genoa, simsimd_dot_e4m3_serial);
    dense_<e5m2_k>("dot_e5m2_genoa", simsimd_dot_e5m2_genoa, simsimd_dot_e5m2_serial);

    curved_<bf16_k>("bilinear_bf16_genoa", simsimd_bilinear_bf16_genoa, simsimd_bilinear_bf16_accurate);
    curved_<bf16_k>("mahalanobis_bf16_genoa", simsimd_mahalanobis_bf16_genoa, simsimd_mahalanobis_bf16_accurate);
    curved_<bf16c_k>("bilinear_bf16c_genoa", simsimd_bilinear_bf16c_genoa, simsimd_bilinear_bf16c_accurate);

    matmul_<simsimd_bf16_t, simsimd_f32_t>("dots_bf16bf16f32_genoa", simsimd_dots_bf16bf16f32_packed_size_genoa,
                                           simsimd_dots_bf16bf16f32_pack_genoa, simsimd_dots_bf16bf16f32_genoa);

#endif

#if SIMSIMD_TARGET_SAPPHIRE
    dense_<f16_k>("dot_f16_sapphire", simsimd_dot_f16_sapphire, simsimd_dot_f16_accurate);
    dense_<f16_k>("angular_f16_sapphire", simsimd_angular_f16_sapphire, simsimd_angular_f16_accurate);
    dense_<f16_k>("l2sq_f16_sapphire", simsimd_l2sq_f16_sapphire, simsimd_l2sq_f16_accurate);
    dense_<f16_k>("l2_f16_sapphire", simsimd_l2_f16_sapphire, simsimd_l2_f16_accurate);
    dense_<f16_k>("kld_f16_sapphire", simsimd_kld_f16_sapphire, simsimd_kld_f16_accurate);
    dense_<f16_k>("jsd_f16_sapphire", simsimd_jsd_f16_sapphire, simsimd_jsd_f16_accurate);

    dense_<f16c_k>("dot_f16c_sapphire", simsimd_dot_f16c_sapphire, simsimd_dot_f16c_accurate);
    dense_<f16c_k>("vdot_f16c_sapphire", simsimd_vdot_f16c_sapphire, simsimd_vdot_f16c_accurate);

    dense_<e4m3_k>("dot_e4m3_sapphire", simsimd_dot_e4m3_sapphire, simsimd_dot_e4m3_serial);
    dense_<e5m2_k>("dot_e5m2_sapphire", simsimd_dot_e5m2_sapphire, simsimd_dot_e5m2_serial);

    elementwise_<u8_k, simsimd_metric_fma_k>("fma_u8_sapphire", simsimd_fma_u8_sapphire, simsimd_fma_u8_serial,
                                             simsimd_l2_u8_serial);
    elementwise_<u8_k, simsimd_metric_wsum_k>("wsum_u8_sapphire", simsimd_wsum_u8_sapphire, simsimd_wsum_u8_serial,
                                              simsimd_l2_u8_serial);
    elementwise_<i8_k, simsimd_metric_fma_k>("fma_i8_sapphire", simsimd_fma_i8_sapphire, simsimd_fma_i8_serial,
                                             simsimd_l2_i8_serial);
    elementwise_<i8_k, simsimd_metric_wsum_k>("wsum_i8_sapphire", simsimd_wsum_i8_sapphire, simsimd_wsum_i8_serial,
                                              simsimd_l2_i8_serial);

    curved_<f16_k>("bilinear_f16_sapphire", simsimd_bilinear_f16_sapphire, simsimd_bilinear_f16_accurate);
    curved_<f16_k>("mahalanobis_f16_sapphire", simsimd_mahalanobis_f16_sapphire, simsimd_mahalanobis_f16_accurate);
    curved_<f16c_k>("bilinear_f16c_sapphire", simsimd_bilinear_f16c_sapphire, simsimd_bilinear_f16c_accurate);

    matmul_<simsimd_bf16_t, simsimd_f32_t>(
        "dots_bf16bf16f32_sapphire_amx", simsimd_dots_bf16bf16f32_packed_size_sapphire_amx,
        simsimd_dots_bf16bf16f32_pack_sapphire_amx, simsimd_dots_bf16bf16f32_sapphire_amx);
    matmul_<simsimd_i8_t, simsimd_i32_t>("dots_i8i8i32_sapphire_amx", simsimd_dots_i8i8i32_packed_size_sapphire_amx,
                                         simsimd_dots_i8i8i32_pack_sapphire_amx, simsimd_dots_i8i8i32_sapphire_amx);

#endif

#if SIMSIMD_TARGET_TURIN
    sparse_<u16_k>("intersect_u16_turin", simsimd_intersect_u16_turin, simsimd_intersect_u16_accurate);
    sparse_<u32_k>("intersect_u32_turin", simsimd_intersect_u32_turin, simsimd_intersect_u32_accurate);
#endif

    sparse_<u16_k>("intersect_u16_serial", simsimd_intersect_u16_serial, simsimd_intersect_u16_accurate);
    sparse_<u16_k>("intersect_u16_accurate", simsimd_intersect_u16_accurate, simsimd_intersect_u16_accurate);
    sparse_<u32_k>("intersect_u32_serial", simsimd_intersect_u32_serial, simsimd_intersect_u32_accurate);
    sparse_<u32_k>("intersect_u32_accurate", simsimd_intersect_u32_accurate, simsimd_intersect_u32_accurate);

    curved_<f64_k>("bilinear_f64_serial", simsimd_bilinear_f64_serial, simsimd_bilinear_f64_serial);
    curved_<f64c_k>("bilinear_f64c_serial", simsimd_bilinear_f64c_serial, simsimd_bilinear_f64c_serial);
    curved_<f64_k>("mahalanobis_f64_serial", simsimd_mahalanobis_f64_serial, simsimd_mahalanobis_f64_serial);
    curved_<f32_k>("bilinear_f32_serial", simsimd_bilinear_f32_serial, simsimd_bilinear_f32_accurate);
    curved_<f32c_k>("bilinear_f32c_serial", simsimd_bilinear_f32c_serial, simsimd_bilinear_f32c_accurate);
    curved_<f32_k>("mahalanobis_f32_serial", simsimd_mahalanobis_f32_serial, simsimd_mahalanobis_f32_accurate);
    curved_<f16_k>("bilinear_f16_serial", simsimd_bilinear_f16_serial, simsimd_bilinear_f16_accurate);
    curved_<f16c_k>("bilinear_f16c_serial", simsimd_bilinear_f16c_serial, simsimd_bilinear_f16c_accurate);
    curved_<f16_k>("mahalanobis_f16_serial", simsimd_mahalanobis_f16_serial, simsimd_mahalanobis_f16_accurate);
    curved_<bf16_k>("bilinear_bf16_serial", simsimd_bilinear_bf16_serial, simsimd_bilinear_bf16_accurate);
    curved_<bf16c_k>("bilinear_bf16c_serial", simsimd_bilinear_bf16c_serial, simsimd_bilinear_bf16c_accurate);
    curved_<bf16_k>("mahalanobis_bf16_serial", simsimd_mahalanobis_bf16_serial, simsimd_mahalanobis_bf16_accurate);

    mesh_<f32_k>("rmsd_f32_serial", simsimd_rmsd_f32_serial, simsimd_rmsd_f32_serial);
    mesh_<f32_k>("kabsch_f32_serial", simsimd_kabsch_f32_serial, simsimd_kabsch_f32_serial);
    mesh_<f64_k>("rmsd_f64_serial", simsimd_rmsd_f64_serial, simsimd_rmsd_f64_serial);
    mesh_<f64_k>("kabsch_f64_serial", simsimd_kabsch_f64_serial, simsimd_kabsch_f64_serial);

    dense_<bf16_k>("dot_bf16_serial", simsimd_dot_bf16_serial, simsimd_dot_bf16_accurate);
    dense_<bf16_k>("angular_bf16_serial", simsimd_angular_bf16_serial, simsimd_angular_bf16_accurate);
    dense_<bf16_k>("l2sq_bf16_serial", simsimd_l2sq_bf16_serial, simsimd_l2sq_bf16_accurate);
    dense_<bf16_k>("l2_bf16_serial", simsimd_l2_bf16_serial, simsimd_l2_bf16_accurate);
    dense_<bf16_k>("kld_bf16_serial", simsimd_kld_bf16_serial, simsimd_kld_bf16_accurate);
    dense_<bf16_k>("jsd_bf16_serial", simsimd_jsd_bf16_serial, simsimd_jsd_bf16_accurate);

    dense_<e4m3_k>("dot_e4m3_serial", simsimd_dot_e4m3_serial, simsimd_dot_e4m3_serial);
    dense_<e5m2_k>("dot_e5m2_serial", simsimd_dot_e5m2_serial, simsimd_dot_e5m2_serial);

    dense_<f16_k>("dot_f16_serial", simsimd_dot_f16_serial, simsimd_dot_f16_accurate);
    dense_<f16_k>("angular_f16_serial", simsimd_angular_f16_serial, simsimd_angular_f16_accurate);
    dense_<f16_k>("l2sq_f16_serial", simsimd_l2sq_f16_serial, simsimd_l2sq_f16_accurate);
    dense_<f16_k>("l2_f16_serial", simsimd_l2_f16_serial, simsimd_l2_f16_accurate);
    dense_<f16_k>("kld_f16_serial", simsimd_kld_f16_serial, simsimd_kld_f16_accurate);
    dense_<f16_k>("jsd_f16_serial", simsimd_jsd_f16_serial, simsimd_jsd_f16_accurate);

    dense_<f32_k>("dot_f32_serial", simsimd_dot_f32_serial, simsimd_dot_f32_accurate);
    dense_<f32_k>("angular_f32_serial", simsimd_angular_f32_serial, simsimd_angular_f32_accurate);
    dense_<f32_k>("l2sq_f32_serial", simsimd_l2sq_f32_serial, simsimd_l2sq_f32_accurate);
    dense_<f32_k>("l2_f32_serial", simsimd_l2_f32_serial, simsimd_l2_f32_accurate);
    dense_<f32_k>("kld_f32_serial", simsimd_kld_f32_serial, simsimd_kld_f32_accurate);
    dense_<f32_k>("jsd_f32_serial", simsimd_jsd_f32_serial, simsimd_jsd_f32_accurate);

    dense_<f64_k>("dot_f64_serial", simsimd_dot_f64_serial, simsimd_dot_f64_serial);
    dense_<f64_k>("angular_f64_serial", simsimd_angular_f64_serial, simsimd_angular_f64_serial);
    dense_<f64_k>("l2sq_f64_serial", simsimd_l2sq_f64_serial, simsimd_l2sq_f64_serial);
    dense_<f64_k>("l2_f64_serial", simsimd_l2_f64_serial, simsimd_l2_f64_serial);

    dense_<i8_k>("angular_i8_serial", simsimd_angular_i8_serial, simsimd_angular_i8_serial);
    dense_<i8_k>("l2sq_i8_serial", simsimd_l2sq_i8_serial, simsimd_l2sq_i8_serial);
    dense_<i8_k>("l2_i8_serial", simsimd_l2_i8_serial, simsimd_l2_i8_serial);
    dense_<i8_k>("dot_i8_serial", simsimd_dot_i8_serial, simsimd_dot_i8_serial);

    dense_<u8_k>("angular_u8_serial", simsimd_angular_u8_serial, simsimd_angular_u8_serial);
    dense_<u8_k>("l2sq_u8_serial", simsimd_l2sq_u8_serial, simsimd_l2sq_u8_serial);
    dense_<u8_k>("l2_u8_serial", simsimd_l2_u8_serial, simsimd_l2_u8_serial);
    dense_<u8_k>("dot_u8_serial", simsimd_dot_u8_serial, simsimd_dot_u8_serial);

    dense_<f64c_k>("dot_f64c_serial", simsimd_dot_f64c_serial, simsimd_dot_f64c_serial);
    dense_<f32c_k>("dot_f32c_serial", simsimd_dot_f32c_serial, simsimd_dot_f32c_accurate);
    dense_<f16c_k>("dot_f16c_serial", simsimd_dot_f16c_serial, simsimd_dot_f16c_accurate);
    dense_<bf16c_k>("dot_bf16c_serial", simsimd_dot_bf16c_serial, simsimd_dot_bf16c_accurate);
    dense_<f64c_k>("vdot_f64c_serial", simsimd_vdot_f64c_serial, simsimd_vdot_f64c_serial);
    dense_<f32c_k>("vdot_f32c_serial", simsimd_vdot_f32c_serial, simsimd_vdot_f32c_accurate);
    dense_<f16c_k>("vdot_f16c_serial", simsimd_vdot_f16c_serial, simsimd_vdot_f16c_accurate);
    dense_<bf16c_k>("vdot_bf16c_serial", simsimd_vdot_bf16c_serial, simsimd_vdot_bf16c_accurate);

    dense_<b8_k>("hamming_b8_serial", simsimd_hamming_b8_serial, simsimd_hamming_b8_serial);
    dense_<b8_k>("jaccard_b8_serial", simsimd_jaccard_b8_serial, simsimd_jaccard_b8_serial);

    elementwise_<f32_k>("sin_f32_stl", elementwise_with_stl<simsimd_f32_t, sin_with_stl<simsimd_f32_t>>,
                        elementwise_with_stl<simsimd_f32_t, sin_with_stl<simsimd_f64_t>>, l2_with_stl<simsimd_f32_t>);
    elementwise_<f32_k>("cos_f32_stl", elementwise_with_stl<simsimd_f32_t, cos_with_stl<simsimd_f32_t>>,
                        elementwise_with_stl<simsimd_f32_t, cos_with_stl<simsimd_f64_t>>, l2_with_stl<simsimd_f32_t>);
    elementwise_<f32_k>("atan_f32_stl", elementwise_with_stl<simsimd_f32_t, atan_with_stl<simsimd_f32_t>>,
                        elementwise_with_stl<simsimd_f32_t, atan_with_stl<simsimd_f64_t>>, l2_with_stl<simsimd_f32_t>);
    elementwise_<f32_k>("sin_f32_serial", simsimd_sin_f32_serial,
                        elementwise_with_stl<simsimd_f32_t, sin_with_stl<simsimd_f64_t>>, l2_with_stl<simsimd_f32_t>);
    elementwise_<f32_k>("cos_f32_serial", simsimd_cos_f32_serial,
                        elementwise_with_stl<simsimd_f32_t, cos_with_stl<simsimd_f64_t>>, l2_with_stl<simsimd_f32_t>);
    elementwise_<f32_k>("atan_f32_serial", simsimd_atan_f32_serial,
                        elementwise_with_stl<simsimd_f32_t, atan_with_stl<simsimd_f64_t>>, l2_with_stl<simsimd_f32_t>);
    elementwise_<f64_k>("sin_f64_stl", elementwise_with_stl<simsimd_f64_t, sin_with_stl<simsimd_f64_t>>,
                        elementwise_with_stl<simsimd_f64_t, sin_with_stl<simsimd_f64_t>>, l2_with_stl<simsimd_f64_t>);
    elementwise_<f64_k>("cos_f64_stl", elementwise_with_stl<simsimd_f64_t, cos_with_stl<simsimd_f64_t>>,
                        elementwise_with_stl<simsimd_f64_t, cos_with_stl<simsimd_f64_t>>, l2_with_stl<simsimd_f64_t>);
    elementwise_<f64_k>("atan_f64_stl", elementwise_with_stl<simsimd_f64_t, atan_with_stl<simsimd_f64_t>>,
                        elementwise_with_stl<simsimd_f64_t, atan_with_stl<simsimd_f64_t>>, l2_with_stl<simsimd_f64_t>);
    elementwise_<f64_k>("sin_f64_serial", simsimd_sin_f64_serial,
                        elementwise_with_stl<simsimd_f64_t, sin_with_stl<simsimd_f64_t>>, l2_with_stl<simsimd_f64_t>);
    elementwise_<f64_k>("cos_f64_serial", simsimd_cos_f64_serial,
                        elementwise_with_stl<simsimd_f64_t, cos_with_stl<simsimd_f64_t>>, l2_with_stl<simsimd_f64_t>);
    elementwise_<f64_k>("atan_f64_serial", simsimd_atan_f64_serial,
                        elementwise_with_stl<simsimd_f64_t, atan_with_stl<simsimd_f64_t>>, l2_with_stl<simsimd_f64_t>);

    elementwise_<f16_k, simsimd_metric_fma_k>("fma_f16_serial", simsimd_fma_f16_serial, simsimd_fma_f16_accurate,
                                              simsimd_l2_f16_accurate);
    elementwise_<f16_k, simsimd_metric_wsum_k>("wsum_f16_serial", simsimd_wsum_f16_serial, simsimd_wsum_f16_accurate,
                                               simsimd_l2_f16_accurate);
    elementwise_<u8_k, simsimd_metric_fma_k>("fma_u8_serial", simsimd_fma_u8_serial, simsimd_fma_u8_accurate,
                                             simsimd_l2_u8_serial);
    elementwise_<u8_k, simsimd_metric_wsum_k>("wsum_u8_serial", simsimd_wsum_u8_serial, simsimd_wsum_u8_accurate,
                                              simsimd_l2_u8_serial);
    elementwise_<i8_k, simsimd_metric_fma_k>("fma_i8_serial", simsimd_fma_i8_serial, simsimd_fma_i8_accurate,
                                             simsimd_l2_i8_serial);
    elementwise_<i8_k, simsimd_metric_wsum_k>("wsum_i8_serial", simsimd_wsum_i8_serial, simsimd_wsum_i8_accurate,
                                              simsimd_l2_i8_serial);

    geospatial_<f32_k>("haversine_f32_serial", simsimd_haversine_f32_serial,
                       haversine_with_stl<simsimd_f32_t, simsimd_f64_t>, l2_with_stl<simsimd_f64_t>);
    geospatial_<f64_k>("haversine_f64_serial", simsimd_haversine_f64_serial, haversine_with_stl<simsimd_f64_t>,
                       l2_with_stl<simsimd_f64_t>);
    geospatial_<f32_k>("vincenty_f32_serial", simsimd_vincenty_f32_serial,
                       vincenty_with_stl<simsimd_f32_t, simsimd_f64_t>, l2_with_stl<simsimd_f64_t>);
    geospatial_<f64_k>("vincenty_f64_serial", simsimd_vincenty_f64_serial, vincenty_with_stl<simsimd_f64_t>,
                       l2_with_stl<simsimd_f64_t>);

    matmul_<simsimd_bf16_t, simsimd_f32_t>("dots_bf16bf16f32_serial", simsimd_dots_bf16bf16f32_packed_size_serial,
                                           simsimd_dots_bf16bf16f32_pack_serial, simsimd_dots_bf16bf16f32_serial);
    matmul_<simsimd_i8_t, simsimd_i32_t>("dots_i8i8i32_serial", simsimd_dots_i8i8i32_packed_size_serial,
                                         simsimd_dots_i8i8i32_pack_serial, simsimd_dots_i8i8i32_serial);
    matmul_<simsimd_f32_t, simsimd_f32_t>("dots_f32f32f32_serial", simsimd_dots_f32f32f32_packed_size_serial,
                                          simsimd_dots_f32f32f32_pack_serial, simsimd_dots_f32f32f32_serial);

    bm::RunSpecifiedBenchmarks();
    bm::Shutdown();
    return 0;
}

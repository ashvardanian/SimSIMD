/**
 *  @brief NumKong C++ Benchmark Suite using Google Benchmark.
 *  @file scripts/bench.cpp
 *
 *  Comprehensive benchmarks comparing NumKong SIMD-optimized functions against
 *  baseline implementations. Run with:
 *
 *  ```bash
 *  cmake -B build_release -D NK_BUILD_BENCHMARKS=1
 *  cmake --build build_release
 *  build_release/nk_bench
 *  ```
 *
 *  Environment Variables:
 *    NK_BENCH_FILTER=<pattern>       - Filter benchmarks by name regex (default: run all)
 *    NK_BENCH_DENSE_DIMENSION=N      - Vector dimension for dot/spatial benchmarks (default: 1536)
 *    NK_BENCH_CURVED_DIMENSION=N     - Vector dimension for curved benchmarks (default: 8)
 *    NK_BENCH_MESH_DIMENSION=N       - Point count for mesh benchmarks (default: 1000)
 *    NK_BENCH_MATMUL_DIMENSION_M=N   - GEMM M dimension (default: 128)
 *    NK_BENCH_MATMUL_DIMENSION_N=N   - GEMM N dimension (default: 512)
 *    NK_BENCH_MATMUL_DIMENSION_K=N   - GEMM K dimension (default: 256)
 *    NK_BENCH_SEED=N                 - RNG seed (default: 42)
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
#include <complex>       // `std::complex`

#include <benchmark/benchmark.h>

#if !defined(NK_COMPARE_TO_MKL)
#define NK_COMPARE_TO_MKL 0
#endif
#if !defined(NK_COMPARE_TO_BLAS)
#define NK_COMPARE_TO_BLAS 0
#endif
#if !defined(NK_COMPARE_TO_ACCELERATE)
#define NK_COMPARE_TO_ACCELERATE 0
#endif

#if NK_COMPARE_TO_MKL
#include <mkl.h>
// MKL provides additional GEMM routines:
// - cblas_gemm_bf16: BF16 inputs → F32 output
// - cblas_hgemm: F16 GEMM (if available)
#elif NK_COMPARE_TO_ACCELERATE
#include <Accelerate/Accelerate.h> // Apple Accelerate framework
#elif NK_COMPARE_TO_BLAS
#include <cblas.h> // Generic CBLAS (OpenBLAS, etc.)
// OpenBLAS thread control (weak symbol to avoid link errors if not present)
extern "C" void openblas_set_num_threads(int) __attribute__((weak));
#endif

// It's important to note, that out compression/decompression routines
// are quite inaccurate. They are not meant to be used in production code.
// So when benchmarking, if possible, please use the native types, if those
// are implemented.
#define NK_NATIVE_F16  1
#define NK_NATIVE_BF16 1
#include <numkong/numkong.h>

constexpr std::size_t default_seconds = 10;
constexpr std::size_t default_threads = 1;
constexpr nk_fmax_t signaling_distance = std::numeric_limits<nk_fmax_t>::signaling_NaN();

/// For sub-byte data types
/// Can be overridden at runtime via `NK_BENCH_DENSE_DIMENSION` environment variable
std::size_t dense_dimension = 1536;
/// Has quadratic impact on the number of operations
/// Can be overridden at runtime via `NK_BENCH_CURVED_DIMENSION` environment variable
std::size_t curved_dimension = 8;
/// Number of 3D points for mesh metrics (RMSD, Kabsch)
/// Can be overridden at runtime via `NK_BENCH_MESH_DIMENSION` environment variable
std::size_t mesh_dimension = 1000;
/// Matrix multiplication benchmark globals
/// Can be overridden at runtime via `NK_BENCH_MATMUL_DIMENSION_M/N/K` environment variables
std::size_t matmul_dimension_m = 128, matmul_dimension_n = 512, matmul_dimension_k = 256;
/// Random seed for reproducible benchmarks
/// Can be overridden at runtime via `NK_BENCH_SEED` environment variable
std::uint32_t random_seed = 42;

namespace bm = benchmark;

/// Returns a new random engine seeded with the global random_seed.
inline std::mt19937 make_random_engine() { return std::mt19937(random_seed); }

template <nk_dtype_t>
struct dtype_enum_to_type {
    using value_t = void;
    using scalar_t = void;
    static constexpr std::size_t components_k = 1;
};
template <>
struct dtype_enum_to_type<nk_f64_k> {
    using value_t = nk_f64_t;
    using scalar_t = nk_f64_t;
    static constexpr std::size_t components_k = 1;
};
template <>
struct dtype_enum_to_type<nk_f32_k> {
    using value_t = nk_f32_t;
    using scalar_t = nk_f32_t;
    static constexpr std::size_t components_k = 1;
};
template <>
struct dtype_enum_to_type<nk_f16_k> {
    using value_t = nk_f16_t;
    using scalar_t = nk_f16_t;
    static constexpr std::size_t components_k = 1;
};
template <>
struct dtype_enum_to_type<nk_bf16_k> {
    using value_t = nk_bf16_t;
    using scalar_t = nk_bf16_t;
    static constexpr std::size_t components_k = 1;
};
template <>
struct dtype_enum_to_type<nk_e4m3_k> {
    using value_t = nk_e4m3_t;
    using scalar_t = nk_e4m3_t;
    static constexpr std::size_t components_k = 1;
};
template <>
struct dtype_enum_to_type<nk_e5m2_k> {
    using value_t = nk_e5m2_t;
    using scalar_t = nk_e5m2_t;
    static constexpr std::size_t components_k = 1;
};
template <>
struct dtype_enum_to_type<nk_f64c_k> {
    using value_t = nk_f64c_t;
    using scalar_t = nk_f64_t;
    static constexpr std::size_t components_k = 2;
};
template <>
struct dtype_enum_to_type<nk_f32c_k> {
    using value_t = nk_f32c_t;
    using scalar_t = nk_f32_t;
    static constexpr std::size_t components_k = 2;
};
template <>
struct dtype_enum_to_type<nk_f16c_k> {
    using value_t = nk_f16c_t;
    using scalar_t = nk_f16_t;
    static constexpr std::size_t components_k = 2;
};
template <>
struct dtype_enum_to_type<nk_bf16c_k> {
    using value_t = nk_bf16c_t;
    using scalar_t = nk_bf16_t;
    static constexpr std::size_t components_k = 2;
};
template <>
struct dtype_enum_to_type<nk_u1_k> {
    using value_t = nk_u1x8_t;
    using scalar_t = nk_u1x8_t;
    static constexpr std::size_t components_k = 1;
};
template <>
struct dtype_enum_to_type<nk_i4_k> {
    using value_t = nk_i4x2_t;
    using scalar_t = nk_u8_t; // Packed byte unit
    static constexpr std::size_t components_k = 1;
};
template <>
struct dtype_enum_to_type<nk_u4_k> {
    using value_t = nk_u4x2_t;
    using scalar_t = nk_u8_t; // Packed byte unit
    static constexpr std::size_t components_k = 1;
};
template <>
struct dtype_enum_to_type<nk_i8_k> {
    using value_t = nk_i8_t;
    using scalar_t = nk_i8_t;
    static constexpr std::size_t components_k = 1;
};
template <>
struct dtype_enum_to_type<nk_u8_k> {
    using value_t = nk_u8_t;
    using scalar_t = nk_u8_t;
    static constexpr std::size_t components_k = 1;
};
template <>
struct dtype_enum_to_type<nk_i16_k> {
    using value_t = nk_i16_t;
    using scalar_t = nk_i16_t;
    static constexpr std::size_t components_k = 1;
};
template <>
struct dtype_enum_to_type<nk_u16_k> {
    using value_t = nk_u16_t;
    using scalar_t = nk_u16_t;
    static constexpr std::size_t components_k = 1;
};
template <>
struct dtype_enum_to_type<nk_i32_k> {
    using value_t = nk_i32_t;
    using scalar_t = nk_i32_t;
    static constexpr std::size_t components_k = 1;
};
template <>
struct dtype_enum_to_type<nk_u32_k> {
    using value_t = nk_u32_t;
    using scalar_t = nk_u32_t;
    static constexpr std::size_t components_k = 1;
};
template <>
struct dtype_enum_to_type<nk_i64_k> {
    using value_t = nk_i64_t;
    using scalar_t = nk_i64_t;
    static constexpr std::size_t components_k = 1;
};
template <>
struct dtype_enum_to_type<nk_u64_k> {
    using value_t = nk_u64_t;
    using scalar_t = nk_u64_t;
    static constexpr std::size_t components_k = 1;
};

template <std::size_t multiple>
constexpr std::size_t divide_round_up(std::size_t n) {
    return ((n + multiple - 1) / multiple) * multiple;
}

/**
 *  @brief Vector-like fixed capacity buffer, ensuring cache-line alignment.
 *  @tparam dtype_ The data type of the vector elements, represented as a `nk_dtype_t`.
 */
template <nk_dtype_t dtype_>
struct vector {
    using dtype_reflection_t = dtype_enum_to_type<dtype_>;
    using scalar_t = typename dtype_reflection_t::scalar_t;
    using value_t = typename dtype_reflection_t::value_t;
    static constexpr std::size_t components_k = dtype_reflection_t::components_k;

    static constexpr bool is_integral =                                //
        dtype_ == nk_u1_k || dtype_ == nk_u4_k || dtype_ == nk_i4_k || //
        dtype_ == nk_i8_k || dtype_ == nk_u8_k ||                      //
        dtype_ == nk_i16_k || dtype_ == nk_u16_k ||                    //
        dtype_ == nk_i32_k || dtype_ == nk_u32_k || dtype_ == nk_i64_k || dtype_ == nk_u64_k;
    static constexpr std::size_t cacheline_length = 64;

    value_t *values_ptr_ = nullptr;
    std::size_t values_count_ = 0;

    vector() = default;
    vector(std::size_t values_count) noexcept(false)
        : values_count_(values_count),
          values_ptr_(static_cast<value_t *>(std::aligned_alloc(
              cacheline_length, divide_round_up<cacheline_length>(values_count * sizeof(value_t))))) {
        if (!values_ptr_) throw std::bad_alloc();
    }

    ~vector() noexcept { std::free(values_ptr_); }

    vector(vector const &other) : vector(other.size()) {
        std::memcpy(values_ptr_, other.values_ptr_, divide_round_up<cacheline_length>(values_count_ * sizeof(value_t)));
    }
    vector &operator=(vector const &other) {
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

#if !NK_NATIVE_BF16
        if constexpr (dtype_ == nk_bf16_k || dtype_ == nk_bf16c_k) {
            nk_f32_t f32 = static_cast<nk_f32_t>(from);
            nk_f32_to_bf16(&f32, &to);
            if ((to & exponent_mask_bf16) == exponent_mask_bf16) to = 0;
            static_assert(sizeof(scalar_t) == sizeof(nk_bf16_t));
            return;
        }
#endif
#if !NK_NATIVE_F16
        if constexpr (dtype_ == nk_f16_k || dtype_ == nk_f16c_k) {
            nk_f32_t f32 = static_cast<nk_f32_t>(from);
            nk_f32_to_f16(&f32, &to);
            if ((to & exponent_mask_f16) == exponent_mask_f16) to = 0;
            static_assert(sizeof(scalar_t) == sizeof(nk_f16_t));
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
#if !NK_NATIVE_BF16
        if constexpr (dtype_ == nk_bf16_k || dtype_ == nk_bf16c_k) {
            nk_f32_t f32;
            nk_bf16_to_f32((nk_bf16_t const *)&from, &f32);
            return f32;
        }
#endif
#if !NK_NATIVE_F16
        if constexpr (dtype_ == nk_f16_k || dtype_ == nk_f16c_k) {
            nk_f32_t f32;
            nk_f16_to_f32((nk_f16_t const *)&from, &f32);
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

template <nk_dtype_t dtype_>
struct vectors_pair {
    using vector_t = vector<dtype_>;
    using scalar_t = typename vector_t::scalar_t;
    static constexpr bool is_integral = vector_t::is_integral;

    vector_t a;
    vector_t b;

    vectors_pair() noexcept = default;
    vectors_pair(std::size_t dimensions) noexcept : a(dimensions), b(dimensions) {}
    vectors_pair(std::size_t size_a, std::size_t size_b) noexcept : a(size_a), b(size_b) {}
    vectors_pair(vectors_pair const &other) noexcept(false) : a(other.a), b(other.b) {}
    vectors_pair &operator=(vectors_pair const &other) noexcept(false) {
        if (this != &other) a = other.a, b = other.b;
        return *this;
    }
};

/**
 *  @brief Measures the performance of a @b dense metric function against a baseline using Google Benchmark.
 *  @tparam pair_type_ The type representing the vector pair used in the measurement.
 *  @tparam output_dtype_ The dtype of the output (e.g., f32_k, f64_k, u32_k).
 *  @tparam test_metric_type_ The type of the test metric function.
 *  @tparam baseline_metric_type_ The type of the baseline metric function.
 *  @param state The benchmark state object provided by Google Benchmark.
 *  @param metric The metric function to benchmark.
 *  @param baseline The baseline function to compare against.
 *  @param dimensions The number of dimensions in the vectors.
 */
template <typename pair_type_, nk_dtype_t test_output_dtype_, nk_dtype_t baseline_output_dtype_,
          typename test_metric_type_ = void, typename baseline_metric_type_ = void>
void measure_dense(bm::State &state, test_metric_type_ metric, baseline_metric_type_ baseline, std::size_t dimensions) {

    using pair_t = pair_type_;
    using vector_t = typename pair_type_::vector_t;
    using value_t = typename vector_t::value_t;

    // Determine result types from explicit dtype parameters
    using test_result_t = typename dtype_enum_to_type<test_output_dtype_>::value_t;
    using baseline_result_t = typename dtype_enum_to_type<baseline_output_dtype_>::value_t;

    // For u1 type, dimensions are bits but size() returns bytes, so multiply by 8
    constexpr bool is_binary = std::is_same_v<value_t, nk_u1x8_t>;
    auto get_dimensions = [](pair_t &pair) -> std::size_t {
        if constexpr (is_binary) return pair.a.size() * 8;
        else return pair.a.size();
    };

    auto call_baseline = [&](pair_t &pair) -> double {
        // Baseline (accurate) always uses f64 output (real or complex)
        baseline_result_t results_f64[2] = {{0}, {0}};
        baseline(pair.a.data(), pair.b.data(), get_dimensions(pair), &results_f64[0]);
        if constexpr (std::is_same_v<baseline_result_t, nk_f64c_t>) { return static_cast<double>(results_f64[0].real); }
        else { return static_cast<double>(results_f64[0]); }
    };
    auto call_contender = [&](pair_t &pair) -> double {
        test_result_t results[2] = {{0}, {0}};
        metric(pair.a.data(), pair.b.data(), get_dimensions(pair), &results[0]);
        if constexpr (std::is_same_v<test_result_t, nk_f32c_t> || std::is_same_v<test_result_t, nk_f64c_t>) {
            return static_cast<double>(results[0].real);
        }
        else { return static_cast<double>(results[0]); }
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
 *  @tparam pair_type_ The type representing the vector pair used in the measurement.
 *  @tparam output_dtype_ The dtype of the output (e.g., f32_k, f64_k).
 *  @tparam test_metric_type_ The type of the test metric function.
 *  @tparam baseline_metric_type_ The type of the baseline metric function.
 *  @param state The benchmark state object provided by Google Benchmark.
 *  @param metric The metric function to benchmark.
 *  @param baseline The baseline function to compare against.
 *  @param dimensions The number of dimensions in the vectors.
 */
template <typename pair_type_, nk_dtype_t test_output_dtype_, nk_dtype_t baseline_output_dtype_,
          typename test_metric_type_ = void, typename baseline_metric_type_ = void>
void measure_curved(bm::State &state, test_metric_type_ metric, baseline_metric_type_ baseline,
                    std::size_t dimensions) {

    using pair_t = pair_type_;
    using vector_t = typename pair_type_::vector_t;
    using value_t = typename vector_t::value_t;

    // Determine result types from explicit dtype parameters
    using test_result_t = typename dtype_enum_to_type<test_output_dtype_>::value_t;
    using baseline_result_t = typename dtype_enum_to_type<baseline_output_dtype_>::value_t;

    auto call_baseline = [&](pair_t const &pair, vector_t const &tensor) -> double {
        // Baseline (accurate) always uses f64 output (real or complex)
        baseline_result_t results[2] = {{0}, {0}};
        baseline(pair.a.data(), pair.b.data(), tensor.data(), pair.a.size(), &results[0]);
        if constexpr (baseline_output_dtype_ == nk_f32c_k || baseline_output_dtype_ == nk_f64c_k ||
                      baseline_output_dtype_ == nk_f16c_k || baseline_output_dtype_ == nk_bf16c_k) {
            return static_cast<double>(results[0].real + results[1].real);
        }
        else { return static_cast<double>(results[0] + results[1]); }
    };
    auto call_contender = [&](pair_t const &pair, vector_t const &tensor) -> double {
        test_result_t results[2] = {{0}, {0}};
        metric(pair.a.data(), pair.b.data(), tensor.data(), pair.a.size(), &results[0]);
        if constexpr (test_output_dtype_ == nk_f32c_k || test_output_dtype_ == nk_f64c_k ||
                      test_output_dtype_ == nk_f16c_k || test_output_dtype_ == nk_bf16c_k) {
            return static_cast<double>(results[0].real + results[1].real);
        }
        else { return static_cast<double>(results[0] + results[1]); }
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
 *  @tparam pair_type_ The type representing the vector pair used in the measurement.
 *  @tparam metric_type_ The type of the metric function (default is void).
 *  @param state The benchmark state object provided by Google Benchmark.
 *  @param metric The metric function to benchmark.
 *  @param baseline The baseline function to compare against.
 *  @param size_a The number of elements in the smaller vector.
 *  @param size_b The number of elements in the larger vector.
 *  @param intersection_size The expected number of common scalars between the vectors.
 */
template <typename pair_type_, nk_dtype_t test_output_dtype_, nk_dtype_t baseline_output_dtype_,
          typename test_metric_type_ = void, typename baseline_metric_type_ = void>
void measure_sparse(bm::State &state, test_metric_type_ metric, baseline_metric_type_ baseline, std::size_t size_a,
                    std::size_t size_b, std::size_t intersection_size) {

    using pair_t = pair_type_;
    using vector_t = typename pair_type_::vector_t;
    using scalar_t = typename vector_t::scalar_t;
    using test_result_t = typename dtype_enum_to_type<test_output_dtype_>::scalar_t;
    using baseline_result_t = typename dtype_enum_to_type<baseline_output_dtype_>::scalar_t;

    auto call_baseline = [&](pair_t &pair) -> double {
        baseline_result_t result = std::numeric_limits<baseline_result_t>::signaling_NaN();
        baseline(pair.a.data(), pair.b.data(), pair.a.size(), pair.b.size(), &result);
        return static_cast<double>(result);
    };
    auto call_contender = [&](pair_t &pair) -> double {
        test_result_t result = std::numeric_limits<test_result_t>::signaling_NaN();
        metric(pair.a.data(), pair.b.data(), pair.a.size(), pair.b.size(), &result);
        return static_cast<double>(result);
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
template <nk_dtype_t dtype_>
struct mesh_pair {
    using vector_t = vector<dtype_>;
    using scalar_t = typename vector_t::scalar_t;

    vector_t a;
    vector_t b;
    std::size_t num_points;

    mesh_pair() noexcept = default;
    mesh_pair(std::size_t points) noexcept : a(points * 3), b(points * 3), num_points(points) {}
    mesh_pair(mesh_pair const &other) noexcept(false) : a(other.a), b(other.b), num_points(other.num_points) {}
    mesh_pair &operator=(mesh_pair const &other) noexcept(false) {
        if (this != &other) a = other.a, b = other.b, num_points = other.num_points;
        return *this;
    }
};

/**
 *  @brief Measures the performance of a @b mesh metric function (RMSD/Kabsch) against a baseline.
 *  @tparam pair_type_ The type representing the point cloud pair.
 *  @tparam metric_type_ The type of the metric function.
 *  @param state The benchmark state object provided by Google Benchmark.
 *  @param metric The metric function to benchmark.
 *  @param baseline The baseline function to compare against.
 *  @param num_points The number of 3D points in each point cloud.
 */
template <typename pair_type_, nk_dtype_t test_output_dtype_, nk_dtype_t baseline_output_dtype_,
          typename test_metric_type_ = void, typename baseline_metric_type_ = void>
void measure_mesh(bm::State &state, test_metric_type_ metric, baseline_metric_type_ baseline, std::size_t num_points) {

    using pair_t = pair_type_;
    using vector_t = typename pair_type_::vector_t;
    using scalar_t = typename vector_t::scalar_t;
    using test_result_t = typename dtype_enum_to_type<test_output_dtype_>::scalar_t;
    using baseline_result_t = typename dtype_enum_to_type<baseline_output_dtype_>::scalar_t;

    auto call_baseline = [&](pair_t &pair) -> double {
        baseline_result_t result = std::numeric_limits<baseline_result_t>::signaling_NaN(), scale = 0;
        scalar_t a_centroid[3], b_centroid[3], rotation[9];
        baseline(pair.a.data(), pair.b.data(), pair.num_points, a_centroid, b_centroid, rotation, &scale, &result);
        return static_cast<double>(result);
    };
    auto call_contender = [&](pair_t &pair) -> double {
        test_result_t result = std::numeric_limits<test_result_t>::signaling_NaN(), scale = 0;
        scalar_t a_centroid[3], b_centroid[3], rotation[9];
        metric(pair.a.data(), pair.b.data(), pair.num_points, a_centroid, b_centroid, rotation, &scale, &result);
        return static_cast<double>(result);
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
 *  @tparam pair_type_ The type representing the vector pair used in the measurement.
 *  @tparam kernel_type_ The type of the kernel function (default is void).
 *  @param state The benchmark state object provided by Google Benchmark.
 *  @param kernel The kernel function to benchmark.
 *  @param baseline The baseline function to compare against.
 *  @param l2_metric The L2 metric function to compute the error
 *  @param dimensions The number of dimensions in the vectors.
 */
template <typename pair_type_, nk_kernel_kind_t kernel_, nk_dtype_t test_output_dtype_,
          nk_dtype_t baseline_output_dtype_, nk_dtype_t test_alpha_dtype_, nk_dtype_t baseline_alpha_dtype_,
          typename test_kernel_type_ = void, typename baseline_kernel_type_ = void, typename l2_metric_type_ = void>
void measure_elementwise(bm::State &state, test_kernel_type_ kernel, baseline_kernel_type_ baseline,
                         l2_metric_type_ l2_metric, std::size_t dimensions) {

    using pair_t = pair_type_;
    using vector_t = typename pair_type_::vector_t;
    using scalar_t = typename pair_type_::scalar_t;

    // Determine alpha/beta types from explicit dtype parameters
    using test_alpha_t = typename dtype_enum_to_type<test_alpha_dtype_>::scalar_t;
    using baseline_alpha_t = typename dtype_enum_to_type<baseline_alpha_dtype_>::scalar_t;

    // Alpha and beta parameters
    test_alpha_t alpha = 0.2;
    test_alpha_t beta = 0.3;
    baseline_alpha_t alpha_baseline = 0.2;
    baseline_alpha_t beta_baseline = 0.3;

    auto call_baseline = [&](vector_t const &a, vector_t const &b, vector_t const &c, vector_t &d) {
        if constexpr (kernel_ == nk_kernel_wsum_k) {
            baseline(a.data(), c.data(), a.size(), &alpha_baseline, &beta_baseline, d.data());
        }
        else if constexpr (kernel_ == nk_kernel_fma_k) {
            baseline(a.data(), b.data(), c.data(), a.size(), &alpha_baseline, &beta_baseline, d.data());
        }
        else if constexpr (kernel_ == nk_kernel_sum_k) { baseline(a.data(), c.data(), a.size(), d.data()); }
        else if constexpr (kernel_ == nk_kernel_scale_k) {
            baseline(a.data(), a.size(), &alpha_baseline, &beta_baseline, d.data());
        }
        else { baseline(a.data(), a.size(), d.data()); }
    };
    auto call_contender = [&](vector_t const &a, vector_t const &b, vector_t const &c, vector_t &d) {
        if constexpr (kernel_ == nk_kernel_wsum_k) { kernel(a.data(), c.data(), a.size(), &alpha, &beta, d.data()); }
        else if constexpr (kernel_ == nk_kernel_fma_k) {
            kernel(a.data(), b.data(), c.data(), a.size(), &alpha, &beta, d.data());
        }
        else if constexpr (kernel_ == nk_kernel_sum_k) { kernel(a.data(), c.data(), a.size(), d.data()); }
        else if constexpr (kernel_ == nk_kernel_scale_k) { kernel(a.data(), a.size(), &alpha, &beta, d.data()); }
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
    // L2 result type is always f64 for accurate verification functions
    using l2_result_t = nk_f64_t;
    std::vector<l2_result_t> l2_metric_from_baseline(quads.size());
    std::vector<l2_result_t> l2_baseline_result_norm(quads.size());
    std::vector<l2_result_t> l2_contender_result_norm(quads.size());
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
    if constexpr (kernel_ == nk_kernel_wsum_k) { bytes_per_call *= 2; }
    else if constexpr (kernel_ == nk_kernel_fma_k) { bytes_per_call *= 3; }
    else if constexpr (kernel_ == nk_kernel_sum_k) { bytes_per_call *= 2; }
    else if constexpr (kernel_ == nk_kernel_scale_k) { bytes_per_call *= 1; }

    // Measure the mean absolute delta and relative error.
    state.counters["abs_delta"] = mean_delta;
    state.counters["relative_error"] = mean_relative_error;
    state.counters["bytes"] = bm::Counter(iterations * bytes_per_call, bm::Counter::kIsRate);
    state.counters["pairs"] = bm::Counter(iterations, bm::Counter::kIsRate);
}

/**
 *  @brief Measures the performance of a geospatial operations between 4 arrays: 2 latitudes, 2 longitudes.
 *  @tparam pair_type_ The type representing the vector pair used in the measurement.
 *  @tparam kernel_type_ The type of the kernel function (default is void).
 *  @param state The benchmark state object provided by Google Benchmark.
 *  @param kernel The kernel function to benchmark.
 *  @param baseline The baseline function to compare against.
 *  @param l2_metric The L2 metric function to compute the error
 *  @param dimensions The number of dimensions in the vectors.
 */
template <typename pair_type_, nk_dtype_t test_output_dtype_, nk_dtype_t baseline_output_dtype_,
          typename test_kernel_type_ = void, typename baseline_kernel_type_ = void, typename l2_metric_type_ = void>
void measure_geospatial(bm::State &state, test_kernel_type_ kernel, baseline_kernel_type_ baseline,
                        l2_metric_type_ l2_metric, std::size_t dimensions) {

    using pair_t = pair_type_;
    using vector_t = typename pair_type_::vector_t;
    using scalar_t = typename vector_t::scalar_t;
    struct quad_t {
        vector_t lat1, lon1, lat2, lon2;
    };

    using test_distances_t = vector<test_output_dtype_>;
    using baseline_distances_t = vector<baseline_output_dtype_>;
    auto call_baseline = [&](quad_t const &quad, baseline_distances_t &d) {
        baseline(quad.lat1.data(), quad.lon1.data(), quad.lat2.data(), quad.lon2.data(), quad.lat1.size(), d.data());
    };
    auto call_contender = [&](quad_t const &quad, test_distances_t &d) {
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
    baseline_distances_t baseline_d(dimensions);
    test_distances_t contender_d(dimensions);
    baseline_distances_t zeros_baseline(dimensions);
    test_distances_t zeros_contender(dimensions);
    // For L2 comparison, convert test output to baseline type (f64)
    baseline_distances_t contender_d_f64(dimensions);
    std::vector<nk_f64_t> l2_metric_from_baseline(quads.size());
    std::vector<nk_f64_t> l2_baseline_result_norm(quads.size());
    std::vector<nk_f64_t> l2_contender_result_norm(quads.size());
    zeros_baseline.set(0);
    zeros_contender.set(0);
    double mean_delta = 0, mean_relative_error = 0;
    for (std::size_t i = 0; i != quads.size(); ++i) {
        quad_t &quad = quads[i];
        call_baseline(quad, baseline_d);
        call_contender(quad, contender_d);
        // Convert test output to f64 for comparison
        for (std::size_t j = 0; j < dimensions; ++j)
            contender_d_f64.data()[j] = static_cast<nk_f64_t>(contender_d.data()[j]);
        l2_metric(baseline_d.data(), contender_d_f64.data(), dimensions, &l2_metric_from_baseline[i]);
        l2_metric(baseline_d.data(), zeros_baseline.data(), dimensions, &l2_baseline_result_norm[i]);
        l2_metric(contender_d_f64.data(), zeros_baseline.data(), dimensions, &l2_contender_result_norm[i]);

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

template <nk_dtype_t dtype_, nk_dtype_t test_output_dtype_, nk_dtype_t baseline_output_dtype_,
          typename test_metric_type_ = void, typename baseline_metric_type_ = void>
void dense_(std::string name, test_metric_type_ *distance_func, baseline_metric_type_ *baseline_func) {
    using pair_t = vectors_pair<dtype_>;
    std::string bench_name = name + "<" + std::to_string(dense_dimension) + "d>";
    bm::RegisterBenchmark(
        bench_name.c_str(),
        measure_dense<pair_t, test_output_dtype_, baseline_output_dtype_, test_metric_type_ *, baseline_metric_type_ *>,
        distance_func, baseline_func, dense_dimension)
        ->MinTime(default_seconds)
        ->Threads(default_threads);
}

template <nk_dtype_t dtype_, nk_kernel_kind_t kernel_ = nk_kernel_unknown_k,
          nk_dtype_t test_output_dtype_ = nk_dtype_unknown_k, nk_dtype_t baseline_output_dtype_ = nk_dtype_unknown_k,
          nk_dtype_t test_alpha_dtype_ = nk_dtype_unknown_k, nk_dtype_t baseline_alpha_dtype_ = nk_dtype_unknown_k,
          typename test_kernel_type_ = void, typename baseline_kernel_type_ = void, typename l2_metric_type_ = void>
void elementwise_(std::string name, test_kernel_type_ *kernel_func, baseline_kernel_type_ *baseline_func,
                  l2_metric_type_ *l2_metric_func) {
    using pair_t = vectors_pair<dtype_>;
    std::string bench_name = name + "<" + std::to_string(dense_dimension) + "d>";
    bm::RegisterBenchmark(
        bench_name.c_str(),
        measure_elementwise<pair_t, kernel_, test_output_dtype_, baseline_output_dtype_, test_alpha_dtype_,
                            baseline_alpha_dtype_, test_kernel_type_ *, baseline_kernel_type_ *, l2_metric_type_ *>,
        kernel_func, baseline_func, l2_metric_func, dense_dimension)
        ->MinTime(default_seconds)
        ->Threads(default_threads);
}

template <nk_dtype_t dtype_, nk_dtype_t test_output_dtype_, nk_dtype_t baseline_output_dtype_,
          typename test_kernel_type_ = void, typename baseline_kernel_type_ = void, typename l2_metric_type_ = void>
void geospatial_(std::string name, test_kernel_type_ *kernel_func, baseline_kernel_type_ *baseline_func,
                 l2_metric_type_ *l2_metric_func) {
    using pair_t = vectors_pair<dtype_>;
    std::string bench_name = name + "<" + std::to_string(dense_dimension) + "d>";
    bm::RegisterBenchmark(bench_name.c_str(),
                          measure_geospatial<pair_t, test_output_dtype_, baseline_output_dtype_, test_kernel_type_ *,
                                             baseline_kernel_type_ *, l2_metric_type_ *>,
                          kernel_func, baseline_func, l2_metric_func, dense_dimension)
        ->MinTime(default_seconds)
        ->Threads(default_threads);
}

template <nk_dtype_t dtype_, nk_dtype_t test_output_dtype_, nk_dtype_t baseline_output_dtype_,
          typename test_metric_type_ = void, typename baseline_metric_type_ = void>
void sparse_(std::string name, test_metric_type_ *distance_func, baseline_metric_type_ *baseline_func) {

    using pair_t = vectors_pair<dtype_>;

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
                bm::RegisterBenchmark(bench_name.c_str(),
                                      measure_sparse<pair_t, test_output_dtype_, baseline_output_dtype_,
                                                     test_metric_type_ *, baseline_metric_type_ *>,
                                      distance_func, baseline_func, first_len, second_len, intersection_size)
                    ->MinTime(default_seconds)
                    ->Threads(default_threads);
            }
        }
    }
}

template <nk_dtype_t dtype_, nk_dtype_t test_output_dtype_, nk_dtype_t baseline_output_dtype_,
          typename test_metric_type_ = void, typename baseline_metric_type_ = void>
void curved_(std::string name, test_metric_type_ *distance_func, baseline_metric_type_ *baseline_func) {

    using pair_t = vectors_pair<dtype_>;
    std::string bench_name = name + "<" + std::to_string(curved_dimension) + "d>";
    bm::RegisterBenchmark(bench_name.c_str(),
                          measure_curved<pair_t, test_output_dtype_, baseline_output_dtype_, test_metric_type_ *,
                                         baseline_metric_type_ *>,
                          distance_func, baseline_func, curved_dimension)
        ->MinTime(default_seconds)
        ->Threads(default_threads);
}

template <nk_dtype_t dtype_, nk_dtype_t test_output_dtype_, nk_dtype_t baseline_output_dtype_,
          typename test_metric_type_ = void, typename baseline_metric_type_ = void>
void mesh_(std::string name, test_metric_type_ *distance_func, baseline_metric_type_ *baseline_func) {

    using pair_t = mesh_pair<dtype_>;
    std::string bench_name = name + "<" + std::to_string(mesh_dimension) + "pts>";
    bm::RegisterBenchmark(
        bench_name.c_str(),
        measure_mesh<pair_t, test_output_dtype_, baseline_output_dtype_, test_metric_type_ *, baseline_metric_type_ *>,
        distance_func, baseline_func, mesh_dimension)
        ->MinTime(default_seconds)
        ->Threads(default_threads);
}

//  Matmul measurement for packed B matrix API
template <typename input_type_, typename output_type_>
void measure_matmul_packed(                                                        //
    bm::State &state,                                                              //
    nk_size_t (*packed_size_fn)(nk_size_t, nk_size_t),                             //
    void (*pack_fn)(input_type_ const *, nk_size_t, nk_size_t, nk_size_t, void *), //
    void (*matmul_fn)(input_type_ const *, void const *, output_type_ *, nk_size_t, nk_size_t, nk_size_t, nk_size_t,
                      nk_size_t), //
    std::size_t m, std::size_t n, std::size_t k) {

    // Allocate matrices
    std::vector<input_type_> a(m * k);
    std::vector<input_type_> b(n * k);
    nk_size_t packed_bytes = packed_size_fn(n, k);
    std::vector<char> b_packed(packed_bytes, 0);
    std::vector<output_type_> c(m * n);

    // Initialize with small random values
    auto gen = make_random_engine();
    if constexpr (std::is_integral<input_type_>::value) {
        std::uniform_int_distribution<int> dis(-10, 10);
        for (auto &v : a) v = static_cast<input_type_>(dis(gen));
        for (auto &v : b) v = static_cast<input_type_>(dis(gen));
    }
    else {
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
        for (std::size_t i = 0; i < a.size(); ++i) a[i] = static_cast<input_type_>(dis(gen));
        for (std::size_t i = 0; i < b.size(); ++i) b[i] = static_cast<input_type_>(dis(gen));
    }

    // Pack B matrix once (amortized cost for repeated inference)
    pack_fn(b.data(), n, k, k * sizeof(input_type_), b_packed.data());

    std::size_t iterations = 0;
    for (auto _ : state) {
        bm::DoNotOptimize(c.data());
        matmul_fn(a.data(), b_packed.data(), c.data(), m, n, k, k * sizeof(input_type_), n * sizeof(output_type_));
        ++iterations;
    }

    state.counters["tops"] = bm::Counter(iterations * 2.0 * m * n * k, bm::Counter::kIsRate);
}

template <typename input_type_, typename output_type_>
void matmul_(std::string name,                                                                          //
             nk_size_t (*packed_size_fn)(nk_size_t, nk_size_t),                                         //
             void (*pack_fn)(input_type_ const *, nk_size_t, nk_size_t, nk_size_t, void *),             //
             void (*matmul_fn)(input_type_ const *, void const *, output_type_ *, nk_size_t, nk_size_t, //
                               nk_size_t, nk_size_t, nk_size_t)) {                                      //
    std::string bench_name = name + "<" + std::to_string(matmul_dimension_m) + "x" +
                             std::to_string(matmul_dimension_n) + "x" + std::to_string(matmul_dimension_k) + ">";
    bm::RegisterBenchmark(bench_name.c_str(), measure_matmul_packed<input_type_, output_type_>, packed_size_fn, pack_fn,
                          matmul_fn, matmul_dimension_m, matmul_dimension_n, matmul_dimension_k)
        ->MinTime(default_seconds)
        ->Threads(1); // Single-threaded for packed matmul
}

template <typename scalar_type_>
void l2_with_stl(scalar_type_ const *a, scalar_type_ const *b, nk_size_t n, nk_fmax_t *result) {
    nk_fmax_t sum = 0;
    for (nk_size_t i = 0; i != n; ++i) {
        nk_fmax_t delta = (nk_fmax_t)a[i] - (nk_fmax_t)b[i];
        sum += delta * delta;
    }
    *result = std::sqrt(sum);
}

template <typename scalar_type_, typename accumulator_type_ = scalar_type_>
nk_fmax_t haversine_one_with_stl(scalar_type_ lat1, scalar_type_ lon1, scalar_type_ lat2, scalar_type_ lon2) {
    // Convert angle to radians:
    // lat1 *= M_PI / 180, lon1 *= M_PI / 180;
    // lat2 *= M_PI / 180, lon2 *= M_PI / 180;
    accumulator_type_ dlat = lat2 - lat1;
    accumulator_type_ dlon = lon2 - lon1;
    accumulator_type_ a = //
        std::sin(dlat / 2) * std::sin(dlat / 2) +
        std::cos(lat1) * std::cos(lat2) * std::sin(dlon / 2) * std::sin(dlon / 2);
    accumulator_type_ c = 2 * std::atan2(std::sqrt(a), std::sqrt(1 - a));
    return c;
}

template <typename scalar_type_, typename accumulator_type_ = scalar_type_>
void haversine_with_stl(                                    //
    scalar_type_ const *a_lats, scalar_type_ const *a_lons, //
    scalar_type_ const *b_lats, scalar_type_ const *b_lons, //
    nk_size_t n, nk_fmax_t *results) {
    for (nk_size_t i = 0; i != n; ++i) {
        scalar_type_ lat1 = a_lats[i], lon1 = a_lons[i];
        scalar_type_ lat2 = b_lats[i], lon2 = b_lons[i];
        results[i] = haversine_one_with_stl<scalar_type_, accumulator_type_>(lat1, lon1, lat2, lon2);
    }
}

template <typename scalar_type_, typename accumulator_type_ = scalar_type_>
void vincenty_with_stl(                                     //
    scalar_type_ const *a_lats, scalar_type_ const *a_lons, //
    scalar_type_ const *b_lats, scalar_type_ const *b_lons, //
    nk_size_t n, nk_fmax_t *results) {
    // Simplified Vincenty baseline - uses same iterative algorithm as serial implementation
    constexpr accumulator_type_ equatorial_radius = 6378136.6;
    constexpr accumulator_type_ polar_radius = 6356751.9;
    constexpr accumulator_type_ flattening = 1.0 / 298.25642;
    constexpr accumulator_type_ convergence_threshold = 1e-12;
    constexpr int max_iterations = 100;

    for (nk_size_t i = 0; i != n; ++i) {
        accumulator_type_ lat1 = a_lats[i], lon1 = a_lons[i];
        accumulator_type_ lat2 = b_lats[i], lon2 = b_lons[i];
        accumulator_type_ longitude_diff = lon2 - lon1;

        // Reduced latitudes
        accumulator_type_ tan_reduced_lat_1 = (1.0 - flattening) * std::tan(lat1);
        accumulator_type_ tan_reduced_lat_2 = (1.0 - flattening) * std::tan(lat2);
        accumulator_type_ cos_reduced_lat_1 = 1.0 / std::sqrt(1.0 + tan_reduced_lat_1 * tan_reduced_lat_1);
        accumulator_type_ sin_reduced_lat_1 = tan_reduced_lat_1 * cos_reduced_lat_1;
        accumulator_type_ cos_reduced_lat_2 = 1.0 / std::sqrt(1.0 + tan_reduced_lat_2 * tan_reduced_lat_2);
        accumulator_type_ sin_reduced_lat_2 = tan_reduced_lat_2 * cos_reduced_lat_2;

        accumulator_type_ lambda_longitude = longitude_diff, lambda_previous;
        accumulator_type_ sin_sigma, cos_sigma, sigma, sin_alpha, cos_squared_azimuth, cos_twice_sigma_midpoint;
        int iteration_count = 0;
        bool points_are_coincident = false;

        do {
            accumulator_type_ sin_lambda = std::sin(lambda_longitude);
            accumulator_type_ cos_lambda = std::cos(lambda_longitude);
            accumulator_type_ sin_sigma_term_a = cos_reduced_lat_2 * sin_lambda;
            accumulator_type_ sin_sigma_term_b = cos_reduced_lat_1 * sin_reduced_lat_2 -
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
            accumulator_type_ longitude_correction_coeff = flattening / 16.0 * cos_squared_azimuth *
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

        accumulator_type_ u_squared = cos_squared_azimuth *
                                      (equatorial_radius * equatorial_radius - polar_radius * polar_radius) /
                                      (polar_radius * polar_radius);
        accumulator_type_ geodesic_length_coeff =
            1.0 + u_squared / 16384.0 * (4096.0 + u_squared * (-768.0 + u_squared * (320.0 - 175.0 * u_squared)));
        accumulator_type_ delta_sigma_coeff = u_squared / 1024.0 *
                                              (256.0 + u_squared * (-128.0 + u_squared * (74.0 - 47.0 * u_squared)));
        accumulator_type_ delta_sigma =
            delta_sigma_coeff * sin_sigma *
            (cos_twice_sigma_midpoint +
             delta_sigma_coeff / 4.0 *
                 (cos_sigma * (-1.0 + 2.0 * cos_twice_sigma_midpoint * cos_twice_sigma_midpoint) -
                  delta_sigma_coeff / 6.0 * cos_twice_sigma_midpoint * (-3.0 + 4.0 * sin_sigma * sin_sigma) *
                      (-3.0 + 4.0 * cos_twice_sigma_midpoint * cos_twice_sigma_midpoint)));

        results[i] = polar_radius * geodesic_length_coeff * (sigma - delta_sigma);
    }
}

template <typename scalar_type_>
struct sin_with_stl {
    scalar_type_ operator()(scalar_type_ x) const { return std::sin(x); }
};
template <typename scalar_type_>
struct cos_with_stl {
    scalar_type_ operator()(scalar_type_ x) const { return std::cos(x); }
};
template <typename scalar_type_>
struct atan_with_stl {
    scalar_type_ operator()(scalar_type_ x) const { return std::atan(x); }
};

namespace av::numkong {
struct sin {
    nk_f32_t operator()(nk_f32_t x) const { return nk_f32_sin(x); }
    nk_f64_t operator()(nk_f64_t x) const { return nk_f64_sin(x); }
};
struct cos {
    nk_f32_t operator()(nk_f32_t x) const { return nk_f32_cos(x); }
    nk_f64_t operator()(nk_f64_t x) const { return nk_f64_cos(x); }
};
struct atan {
    nk_f32_t operator()(nk_f32_t x) const { return nk_f32_atan(x); }
    nk_f64_t operator()(nk_f64_t x) const { return nk_f64_atan(x); }
};
} // namespace av::numkong

template <typename scalar_type_, typename kernel_type_>
void elementwise_with_stl(scalar_type_ const *ins, nk_size_t n, scalar_type_ *outs) {
    for (nk_size_t i = 0; i != n; ++i) outs[i] = kernel_type_ {}(ins[i]);
}

#if NK_COMPARE_TO_BLAS || NK_COMPARE_TO_MKL || NK_COMPARE_TO_ACCELERATE

void dot_f32_blas(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *result) {
    *result = cblas_sdot(static_cast<int>(n), a, 1, b, 1);
}

void dot_f64_blas(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result) {
    *result = cblas_ddot(static_cast<int>(n), a, 1, b, 1);
}

void dot_f32c_blas(nk_f32c_t const *a, nk_f32c_t const *b, nk_size_t n, nk_f32c_t *result) {
#if NK_COMPARE_TO_ACCELERATE
    // Apple Accelerate uses __LAPACK_float_complex which is std::complex<float>
    cblas_cdotu_sub(static_cast<int>(n), reinterpret_cast<std::complex<float> const *>(a), 1,
                    reinterpret_cast<std::complex<float> const *>(b), 1,
                    reinterpret_cast<std::complex<float> *>(result));
#else
    cblas_cdotu_sub(static_cast<int>(n), reinterpret_cast<nk_f32_t const *>(a), 1,
                    reinterpret_cast<nk_f32_t const *>(b), 1, reinterpret_cast<nk_f32_t *>(result));
#endif
}

void dot_f64c_blas(nk_f64c_t const *a, nk_f64c_t const *b, nk_size_t n, nk_f64c_t *result) {
#if NK_COMPARE_TO_ACCELERATE
    cblas_zdotu_sub(static_cast<int>(n), reinterpret_cast<std::complex<double> const *>(a), 1,
                    reinterpret_cast<std::complex<double> const *>(b), 1,
                    reinterpret_cast<std::complex<double> *>(result));
#else
    cblas_zdotu_sub(static_cast<int>(n), reinterpret_cast<nk_f64_t const *>(a), 1,
                    reinterpret_cast<nk_f64_t const *>(b), 1, reinterpret_cast<nk_f64_t *>(result));
#endif
}

void vdot_f32c_blas(nk_f32c_t const *a, nk_f32c_t const *b, nk_size_t n, nk_f32c_t *result) {
#if NK_COMPARE_TO_ACCELERATE
    cblas_cdotc_sub(static_cast<int>(n), reinterpret_cast<std::complex<float> const *>(a), 1,
                    reinterpret_cast<std::complex<float> const *>(b), 1,
                    reinterpret_cast<std::complex<float> *>(result));
#else
    cblas_cdotc_sub(static_cast<int>(n), reinterpret_cast<nk_f32_t const *>(a), 1,
                    reinterpret_cast<nk_f32_t const *>(b), 1, reinterpret_cast<nk_f32_t *>(result));
#endif
}

void vdot_f64c_blas(nk_f64c_t const *a, nk_f64c_t const *b, nk_size_t n, nk_f64c_t *result) {
#if NK_COMPARE_TO_ACCELERATE
    cblas_zdotc_sub(static_cast<int>(n), reinterpret_cast<std::complex<double> const *>(a), 1,
                    reinterpret_cast<std::complex<double> const *>(b), 1,
                    reinterpret_cast<std::complex<double> *>(result));
#else
    cblas_zdotc_sub(static_cast<int>(n), reinterpret_cast<nk_f64_t const *>(a), 1,
                    reinterpret_cast<nk_f64_t const *>(b), 1, reinterpret_cast<nk_f64_t *>(result));
#endif
}

void bilinear_f32_blas(nk_f32_t const *a, nk_f32_t const *b, nk_f32_t const *c, nk_size_t n, nk_f32_t *result) {
    static thread_local std::vector<nk_f32_t> intermediate;
    if (intermediate.size() < n) intermediate.resize(n);
    int const ni = static_cast<int>(n);
    cblas_sgemv(CblasRowMajor, CblasNoTrans, ni, ni, 1.0f, c, ni, b, 1, 0.0f, intermediate.data(), 1);
    *result = cblas_sdot(ni, a, 1, intermediate.data(), 1);
}

void bilinear_f64_blas(nk_f64_t const *a, nk_f64_t const *b, nk_f64_t const *c, nk_size_t n, nk_f64_t *result) {
    static thread_local std::vector<nk_f64_t> intermediate;
    if (intermediate.size() < n) intermediate.resize(n);
    int const ni = static_cast<int>(n);
    cblas_dgemv(CblasRowMajor, CblasNoTrans, ni, ni, 1.0, c, ni, b, 1, 0.0, intermediate.data(), 1);
    *result = cblas_ddot(ni, a, 1, intermediate.data(), 1);
}

void bilinear_f32c_blas(nk_f32c_t const *a, nk_f32c_t const *b, nk_f32c_t const *c, nk_size_t n, nk_f32c_t *results) {
    static thread_local std::vector<nk_f32c_t> intermediate;
    if (intermediate.size() < n) intermediate.resize(n);
    int const ni = static_cast<int>(n);
#if NK_COMPARE_TO_ACCELERATE
    std::complex<float> alpha = {1.0f, 0.0f}, beta = {0.0f, 0.0f};
    cblas_cgemv(CblasRowMajor, CblasNoTrans, ni, ni, &alpha, reinterpret_cast<std::complex<float> const *>(c), ni,
                reinterpret_cast<std::complex<float> const *>(b), 1, &beta,
                reinterpret_cast<std::complex<float> *>(intermediate.data()), 1);
    cblas_cdotu_sub(ni, reinterpret_cast<std::complex<float> const *>(a), 1,
                    reinterpret_cast<std::complex<float> const *>(intermediate.data()), 1,
                    reinterpret_cast<std::complex<float> *>(results));
#else
    nk_f32c_t alpha = {1.0f, 0.0f}, beta = {0.0f, 0.0f};
    cblas_cgemv(CblasRowMajor, CblasNoTrans, ni, ni, &alpha, c, ni, b, 1, &beta, intermediate.data(), 1);
    cblas_cdotu_sub(ni, reinterpret_cast<nk_f32_t const *>(a), 1,
                    reinterpret_cast<nk_f32_t const *>(intermediate.data()), 1, reinterpret_cast<nk_f32_t *>(results));
#endif
}

void bilinear_f64c_blas(nk_f64c_t const *a, nk_f64c_t const *b, nk_f64c_t const *c, nk_size_t n, nk_f64c_t *results) {
    static thread_local std::vector<nk_f64c_t> intermediate;
    if (intermediate.size() < n) intermediate.resize(n);
    int const ni = static_cast<int>(n);
#if NK_COMPARE_TO_ACCELERATE
    std::complex<double> alpha = {1.0, 0.0}, beta = {0.0, 0.0};
    cblas_zgemv(CblasRowMajor, CblasNoTrans, ni, ni, &alpha, reinterpret_cast<std::complex<double> const *>(c), ni,
                reinterpret_cast<std::complex<double> const *>(b), 1, &beta,
                reinterpret_cast<std::complex<double> *>(intermediate.data()), 1);
    cblas_zdotu_sub(ni, reinterpret_cast<std::complex<double> const *>(a), 1,
                    reinterpret_cast<std::complex<double> const *>(intermediate.data()), 1,
                    reinterpret_cast<std::complex<double> *>(results));
#else
    nk_f64c_t alpha = {1.0, 0.0}, beta = {0.0, 0.0};
    cblas_zgemv(CblasRowMajor, CblasNoTrans, ni, ni, &alpha, c, ni, b, 1, &beta, intermediate.data(), 1);
    cblas_zdotu_sub(ni, reinterpret_cast<nk_f64_t const *>(a), 1,
                    reinterpret_cast<nk_f64_t const *>(intermediate.data()), 1, reinterpret_cast<nk_f64_t *>(results));
#endif
}

void sum_f32_blas(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *result) {
    int const ni = static_cast<int>(n);
    cblas_scopy(ni, a, 1, result, 1);
    cblas_saxpy(ni, 1.0f, b, 1, result, 1);
}

void sum_f64_blas(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result) {
    int const ni = static_cast<int>(n);
    cblas_dcopy(ni, a, 1, result, 1);
    cblas_daxpy(ni, 1.0, b, 1, result, 1);
}

void wsum_f32_blas(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                   nk_f32_t *result) {
    int const ni = static_cast<int>(n);
    std::memset(result, 0, n * sizeof(nk_f32_t));
    if (*alpha != 0) cblas_saxpy(ni, *alpha, a, 1, result, 1);
    if (*beta != 0) cblas_saxpy(ni, *beta, b, 1, result, 1);
}

void wsum_f64_blas(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta,
                   nk_f64_t *result) {
    int const ni = static_cast<int>(n);
    std::memset(result, 0, n * sizeof(nk_f64_t));
    if (*alpha != 0) cblas_daxpy(ni, *alpha, a, 1, result, 1);
    if (*beta != 0) cblas_daxpy(ni, *beta, b, 1, result, 1);
}

template <typename scalar_type_, typename gemm_type_>
void measure_gemm_blas(bm::State &state, std::size_t m, std::size_t n, std::size_t k, gemm_type_ gemm_fn) {
    std::vector<scalar_type_> a(m * k), b(n * k), c(m * n);
    auto gen = make_random_engine();
    std::uniform_real_distribution<scalar_type_> dis(-1.0, 1.0);
    for (auto &v : a) v = dis(gen);
    for (auto &v : b) v = dis(gen);

    std::size_t iterations = 0;
    for (auto _ : state) {
        bm::DoNotOptimize(c.data());
        gemm_fn(a.data(), b.data(), c.data(), m, n, k);
        ++iterations;
    }
    state.counters["tops"] = bm::Counter(iterations * 2.0 * m * n * k, bm::Counter::kIsRate);
}

void measure_dots_f32_blas(bm::State &state, std::size_t m, std::size_t n, std::size_t k) {
    measure_gemm_blas<float>(state, m, n, k,
                             [](float *a, float *b, float *c, std::size_t m, std::size_t n, std::size_t k) {
                                 cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, static_cast<int>(m),
                                             static_cast<int>(n), static_cast<int>(k), 1.0f, a, static_cast<int>(k), b,
                                             static_cast<int>(k), 0.0f, c, static_cast<int>(n));
                             });
}

void measure_dots_f64_blas(bm::State &state, std::size_t m, std::size_t n, std::size_t k) {
    measure_gemm_blas<double>(state, m, n, k,
                              [](double *a, double *b, double *c, std::size_t m, std::size_t n, std::size_t k) {
                                  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, static_cast<int>(m),
                                              static_cast<int>(n), static_cast<int>(k), 1.0, a, static_cast<int>(k), b,
                                              static_cast<int>(k), 0.0, c, static_cast<int>(n));
                              });
}

#endif // NK_COMPARE_TO_BLAS || NK_COMPARE_TO_MKL || NK_COMPARE_TO_ACCELERATE

#if NK_COMPARE_TO_MKL

/// Converts float to MKL_BF16 using NumKong's conversion.
inline MKL_BF16 f32_to_bf16(float val) {
    nk_bf16_t result;
    nk_f32_to_bf16(&val, &result);
    MKL_BF16 mkl_result;
    std::memcpy(&mkl_result, &result, sizeof(mkl_result));
    return mkl_result;
}

/// Converts float to MKL_F16 using NumKong's conversion.
inline MKL_F16 f32_to_f16(float val) {
    nk_f16_t result;
    nk_f32_to_f16(&val, &result);
    MKL_F16 mkl_result;
    std::memcpy(&mkl_result, &result, sizeof(mkl_result));
    return mkl_result;
}

/// Generic MKL GEMM benchmark template - reduces duplication across precision variants.
/// Pattern follows measure_matmul_packed but adds init functors for type-specific conversion.
template <typename input_a_type_, typename input_b_type_, typename output_type_, //
          typename init_a_type_, typename init_b_type_, typename gemm_type_>
void measure_gemm_mkl(bm::State &state, std::size_t m, std::size_t n, std::size_t k, //
                      init_a_type_ init_a, init_b_type_ init_b, gemm_type_ gemm_fn) {
    std::vector<input_a_type_> a(m * k);
    std::vector<input_b_type_> b(n * k);
    std::vector<output_type_> c(m * n);
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
    state.counters["tops"] = bm::Counter(iterations * 2.0 * m * n * k, bm::Counter::kIsRate);
}

/// Overload for integer types - uses int distribution instead of float.
template <typename input_a_type_, typename input_b_type_, typename output_type_, typename gemm_type_>
void measure_gemm_mkl_int(bm::State &state, std::size_t m, std::size_t n, std::size_t k, gemm_type_ gemm_fn) {
    std::vector<input_a_type_> a(m * k);
    std::vector<input_b_type_> b(n * k);
    std::vector<output_type_> c(m * n);
    auto gen = make_random_engine();
    std::uniform_int_distribution<int> dis(-64, 63);

    for (auto &v : a) v = static_cast<input_a_type_>(dis(gen));
    for (auto &v : b) v = static_cast<input_b_type_>(dis(gen));

    std::size_t iterations = 0;
    for (auto _ : state) {
        bm::DoNotOptimize(c.data());
        gemm_fn(a.data(), b.data(), c.data(), m, n, k);
        ++iterations;
    }
    state.counters["tops"] = bm::Counter(iterations * 2.0 * m * n * k, bm::Counter::kIsRate);
}

void measure_dots_f32_mkl(bm::State &state, std::size_t m, std::size_t n, std::size_t k) {
    auto identity = [](float v) { return v; };
    measure_gemm_mkl<float, float, float>(
        state, m, n, k, identity, identity,
        [](float *a, float *b, float *c, std::size_t m, std::size_t n, std::size_t k) {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, (MKL_INT)m, (MKL_INT)n, (MKL_INT)k, 1.0f, a,
                        (MKL_INT)k, b, (MKL_INT)k, 0.0f, c, (MKL_INT)n);
        });
}

void measure_dots_bf16_mkl(bm::State &state, std::size_t m, std::size_t n, std::size_t k) {
    measure_gemm_mkl<MKL_BF16, MKL_BF16, float>(
        state, m, n, k, f32_to_bf16, f32_to_bf16,
        [](MKL_BF16 *a, MKL_BF16 *b, float *c, std::size_t m, std::size_t n, std::size_t k) {
            cblas_gemm_bf16(CblasRowMajor, CblasNoTrans, CblasTrans, (MKL_INT)m, (MKL_INT)n, (MKL_INT)k, 1.0f, a,
                            (MKL_INT)k, b, (MKL_INT)k, 0.0f, c, (MKL_INT)n);
        });
}

void measure_dots_f16_mkl(bm::State &state, std::size_t m, std::size_t n, std::size_t k) {
    measure_gemm_mkl<MKL_F16, MKL_F16, float>(
        state, m, n, k, f32_to_f16, f32_to_f16,
        [](MKL_F16 *a, MKL_F16 *b, float *c, std::size_t m, std::size_t n, std::size_t k) {
            cblas_gemm_f16(CblasRowMajor, CblasNoTrans, CblasTrans, (MKL_INT)m, (MKL_INT)n, (MKL_INT)k, 1.0f, a,
                           (MKL_INT)k, b, (MKL_INT)k, 0.0f, c, (MKL_INT)n);
        });
}

void measure_dots_u8i8i32_mkl(bm::State &state, std::size_t m, std::size_t n, std::size_t k) {
    measure_gemm_mkl_int<std::uint8_t, std::int8_t, std::int32_t>(
        state, m, n, k,
        [](std::uint8_t *a, std::int8_t *b, std::int32_t *c, std::size_t m, std::size_t n, std::size_t k) {
            MKL_INT32 c_offset = 0;
            cblas_gemm_s8u8s32(CblasRowMajor, CblasNoTrans, CblasTrans, CblasFixOffset, (MKL_INT)m, (MKL_INT)n,
                               (MKL_INT)k, 1.0f, a, (MKL_INT)k, 0, b, (MKL_INT)k, 0, 0.0f, c, (MKL_INT)n, &c_offset);
        });
}

void measure_dots_i16i16i32_mkl(bm::State &state, std::size_t m, std::size_t n, std::size_t k) {
    measure_gemm_mkl_int<std::int16_t, std::int16_t, std::int32_t>(
        state, m, n, k,
        [](std::int16_t *a, std::int16_t *b, std::int32_t *c, std::size_t m, std::size_t n, std::size_t k) {
            MKL_INT32 c_offset = 0;
            cblas_gemm_s16s16s32(CblasRowMajor, CblasNoTrans, CblasTrans, CblasFixOffset, (MKL_INT)m, (MKL_INT)n,
                                 (MKL_INT)k, 1.0f, a, (MKL_INT)k, 0, b, (MKL_INT)k, 0, 0.0f, c, (MKL_INT)n, &c_offset);
        });
}

#endif

using cast_kernel_t = void (*)(void const *, nk_dtype_t, nk_size_t, void *, nk_dtype_t);

template <nk_dtype_t from_dtype_, nk_dtype_t to_dtype_>
void measure_cast(bm::State &state, cast_kernel_t kernel, std::size_t n) {
    using from_t = typename dtype_enum_to_type<from_dtype_>::scalar_t;
    using to_t = typename dtype_enum_to_type<to_dtype_>::scalar_t;

    constexpr std::size_t alignment = 64;
    std::size_t from_bytes = ((n * sizeof(from_t) + alignment - 1) / alignment) * alignment;
    std::size_t to_bytes = ((n * sizeof(to_t) + alignment - 1) / alignment) * alignment;

    from_t *src = static_cast<from_t *>(std::aligned_alloc(alignment, from_bytes));
    to_t *dst = static_cast<to_t *>(std::aligned_alloc(alignment, to_bytes));

    // Initialize source with small random values
    auto gen = make_random_engine();
    std::uniform_real_distribution<float> dis(-8.0f, 8.0f);
    for (std::size_t i = 0; i < n; ++i) {
        float val = dis(gen);
        if constexpr (std::is_same_v<from_t, nk_f64_t>) { src[i] = val; }
        else if constexpr (std::is_same_v<from_t, nk_f32_t>) { src[i] = val; }
        else if constexpr (std::is_same_v<from_t, nk_f16_t>) { nk_f32_to_f16(&val, &src[i]); }
        else if constexpr (std::is_same_v<from_t, nk_bf16_t>) { nk_f32_to_bf16(&val, &src[i]); }
        else if constexpr (std::is_same_v<from_t, nk_e4m3_t>) { nk_f32_to_e4m3(&val, &src[i]); }
        else if constexpr (std::is_same_v<from_t, nk_e5m2_t>) { nk_f32_to_e5m2(&val, &src[i]); }
    }

    std::size_t bytes_processed = 0;
    for (auto _ : state) {
        kernel(src, from_dtype_, n, dst, to_dtype_);
        bytes_processed += n * sizeof(from_t) + n * sizeof(to_t);
        bm::ClobberMemory();
    }

    state.SetBytesProcessed(static_cast<std::int64_t>(bytes_processed));
    state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations() * n));

    std::free(src);
    std::free(dst);
}

template <nk_dtype_t from_dtype_, nk_dtype_t to_dtype_>
void cast_(std::string name, cast_kernel_t kernel) {
    std::string bench_name = name + "<" + std::to_string(dense_dimension) + "d>";
    bm::RegisterBenchmark(bench_name.c_str(), measure_cast<from_dtype_, to_dtype_>, kernel, dense_dimension)
        ->MinTime(default_seconds)
        ->Threads(default_threads);
}

int main(int argc, char **argv) {
    nk_capability_t runtime_caps = nk_capabilities();
    nk_configure_thread(runtime_caps); // Also enables AMX if available

#if NK_COMPARE_TO_MKL
    // Set MKL to single-threaded for fair comparison with NumKong (which is single-threaded)
    mkl_set_num_threads(1);
#elif NK_COMPARE_TO_BLAS
    // Set OpenBLAS to single-threaded for fair comparison with NumKong (which is single-threaded)
    if (openblas_set_num_threads) openblas_set_num_threads(1);
#endif
    // Note: Apple Accelerate is typically single-threaded by default for vecLib/BLAS routines

    // Log supported functionality
    char const *flags[2] = {"false", "true"};
    std::printf("NumKong benchmarking suite\n");
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

    // Override dimensions from environment variables if provided
    if (char const *env_dense = std::getenv("NK_BENCH_DENSE_DIMENSION")) {
        std::size_t parsed_dense = static_cast<std::size_t>(std::atoll(env_dense));
        if (parsed_dense > 0) {
            dense_dimension = parsed_dense;
            std::printf("Overriding `dense_dimension` to %zu from NK_BENCH_DENSE_DIMENSION\n", dense_dimension);
        }
    }
    if (char const *env_curved = std::getenv("NK_BENCH_CURVED_DIMENSION")) {
        std::size_t parsed_curved = static_cast<std::size_t>(std::atoll(env_curved));
        if (parsed_curved > 0) {
            curved_dimension = parsed_curved;
            std::printf("Overriding `curved_dimension` to %zu from NK_BENCH_CURVED_DIMENSION\n", curved_dimension);
        }
    }
    if (char const *env_mesh = std::getenv("NK_BENCH_MESH_DIMENSION")) {
        std::size_t parsed_mesh = static_cast<std::size_t>(std::atoll(env_mesh));
        if (parsed_mesh > 0) {
            mesh_dimension = parsed_mesh;
            std::printf("Overriding `mesh_dimension` to %zu from NK_BENCH_MESH_DIMENSION\n", mesh_dimension);
        }
    }
    if (char const *env_matmul_dimension_m = std::getenv("NK_BENCH_MATMUL_DIMENSION_M")) {
        std::size_t parsed = static_cast<std::size_t>(std::atoll(env_matmul_dimension_m));
        if (parsed > 0) {
            matmul_dimension_m = parsed;
            std::printf("Overriding `matmul_dimension_m` to %zu from NK_BENCH_MATMUL_DIMENSION_M\n",
                        matmul_dimension_m);
        }
    }
    if (char const *env_matmul_dimension_n = std::getenv("NK_BENCH_MATMUL_DIMENSION_N")) {
        std::size_t parsed = static_cast<std::size_t>(std::atoll(env_matmul_dimension_n));
        if (parsed > 0) {
            matmul_dimension_n = parsed;
            std::printf("Overriding `matmul_dimension_n` to %zu from NK_BENCH_MATMUL_DIMENSION_N\n",
                        matmul_dimension_n);
        }
    }
    if (char const *env_matmul_dimension_k = std::getenv("NK_BENCH_MATMUL_DIMENSION_K")) {
        std::size_t parsed = static_cast<std::size_t>(std::atoll(env_matmul_dimension_k));
        if (parsed > 0) {
            matmul_dimension_k = parsed;
            std::printf("Overriding `matmul_dimension_k` to %zu from NK_BENCH_MATMUL_DIMENSION_K\n",
                        matmul_dimension_k);
        }
    }
    if (char const *env_seed = std::getenv("NK_BENCH_SEED")) {
        std::uint32_t parsed = static_cast<std::uint32_t>(std::atoll(env_seed));
        random_seed = parsed;
        std::printf("Overriding `random_seed` to %u from NK_BENCH_SEED\n", random_seed);
    }
    std::printf("\n");

    // Handle NK_BENCH_FILTER environment variable by injecting --benchmark_filter argument
    std::vector<char *> modified_argv(argv, argv + argc);
    std::string filter_arg;
    if (char const *env_filter = std::getenv("NK_BENCH_FILTER")) {
        filter_arg = std::string("--benchmark_filter=") + env_filter;
        modified_argv.push_back(const_cast<char *>(filter_arg.c_str()));
        std::printf("Applying benchmark filter from NK_BENCH_FILTER: %s\n\n", env_filter);
    }
    int modified_argc = static_cast<int>(modified_argv.size());
    char **modified_argv_ptr = modified_argv.data();

    // Run the benchmarks
    bm::Initialize(&modified_argc, modified_argv_ptr);
    if (bm::ReportUnrecognizedArguments(modified_argc, modified_argv_ptr)) return 1;

    constexpr nk_dtype_t u1_k = nk_u1_k;
    constexpr nk_dtype_t i4_k = nk_i4_k;
    constexpr nk_dtype_t u4_k = nk_u4_k;
    constexpr nk_dtype_t i8_k = nk_i8_k;
    constexpr nk_dtype_t i16_k = nk_i16_k;
    constexpr nk_dtype_t i32_k = nk_i32_k;
    constexpr nk_dtype_t i64_k = nk_i64_k;
    constexpr nk_dtype_t u8_k = nk_u8_k;
    constexpr nk_dtype_t u16_k = nk_u16_k;
    constexpr nk_dtype_t u32_k = nk_u32_k;
    constexpr nk_dtype_t u64_k = nk_u64_k;
    constexpr nk_dtype_t f64_k = nk_f64_k;
    constexpr nk_dtype_t f32_k = nk_f32_k;
    constexpr nk_dtype_t f16_k = nk_f16_k;
    constexpr nk_dtype_t bf16_k = nk_bf16_k;
    constexpr nk_dtype_t e4m3_k = nk_e4m3_k;
    constexpr nk_dtype_t e5m2_k = nk_e5m2_k;
    constexpr nk_dtype_t f64c_k = nk_f64c_k;
    constexpr nk_dtype_t f32c_k = nk_f32c_k;
    constexpr nk_dtype_t f16c_k = nk_f16c_k;
    constexpr nk_dtype_t bf16c_k = nk_bf16c_k;

    // Metric kind aliases for readability
    constexpr nk_kernel_kind_t fma_k = nk_kernel_fma_k;
    constexpr nk_kernel_kind_t wsum_k = nk_kernel_wsum_k;
    constexpr nk_kernel_kind_t sum_k = nk_kernel_sum_k;
    constexpr nk_kernel_kind_t scale_k = nk_kernel_scale_k;
    constexpr nk_kernel_kind_t unknown_k = nk_kernel_unknown_k;

#if NK_COMPARE_TO_BLAS || NK_COMPARE_TO_MKL || NK_COMPARE_TO_ACCELERATE

    dense_<f32_k, f32_k, f64_k>("dot_f32_blas", dot_f32_blas, nk_dot_f32_accurate);
    dense_<f64_k, f64_k, f64_k>("dot_f64_blas", dot_f64_blas, nk_dot_f64_serial);
    dense_<f32c_k, f32c_k, f64c_k>("dot_f32c_blas", dot_f32c_blas, nk_dot_f32c_accurate);
    dense_<f64c_k, f64c_k, f64c_k>("dot_f64c_blas", dot_f64c_blas, nk_dot_f64c_serial);
    dense_<f32c_k, f32c_k, f64c_k>("vdot_f32c_blas", vdot_f32c_blas, nk_vdot_f32c_accurate);
    dense_<f64c_k, f64c_k, f64c_k>("vdot_f64c_blas", vdot_f64c_blas, nk_vdot_f64c_serial);

    elementwise_<f32_k, nk_kernel_sum_k, f32_k, f64_k, f32_k, f64_k>("sum_f32_blas", sum_f32_blas, nk_sum_f32_accurate,
                                                                     nk_l2_f32_accurate);
    elementwise_<f32_k, nk_kernel_wsum_k, f32_k, f64_k, f32_k, f64_k>("wsum_f32_blas", wsum_f32_blas,
                                                                      nk_wsum_f32_accurate, nk_l2_f32_accurate);
    elementwise_<f64_k, nk_kernel_sum_k, f64_k, f64_k, f64_k, f64_k>("sum_f64_blas", sum_f64_blas, nk_sum_f64_serial,
                                                                     nk_l2_f64_serial);
    elementwise_<f64_k, nk_kernel_wsum_k, f64_k, f64_k, f64_k, f64_k>("wsum_f64_blas", wsum_f64_blas,
                                                                      nk_wsum_f64_serial, nk_l2_f64_serial);

    curved_<f64_k, f64_k, f64_k>("bilinear_f64_blas", bilinear_f64_blas, nk_bilinear_f64_serial);
    curved_<f64c_k, f64c_k, f64c_k>("bilinear_f64c_blas", bilinear_f64c_blas, nk_bilinear_f64c_serial);
    curved_<f32_k, f32_k, f64_k>("bilinear_f32_blas", bilinear_f32_blas, nk_bilinear_f32_accurate);
    curved_<f32c_k, f32c_k, f64c_k>("bilinear_f32c_blas", bilinear_f32c_blas, nk_bilinear_f32c_accurate);

    // BLAS GEMM baselines for matmul comparison (same layout as NumKong: A×Bᵀ)
    {
        std::string dims = std::to_string(matmul_dimension_m) + "x" + std::to_string(matmul_dimension_n) + "x" +
                           std::to_string(matmul_dimension_k);
        bm::RegisterBenchmark(("dots_f32_blas<" + dims + ">").c_str(), measure_dots_f32_blas, matmul_dimension_m,
                              matmul_dimension_n, matmul_dimension_k)
            ->MinTime(default_seconds)
            ->Threads(1);
        bm::RegisterBenchmark(("dots_f64_blas<" + dims + ">").c_str(), measure_dots_f64_blas, matmul_dimension_m,
                              matmul_dimension_n, matmul_dimension_k)
            ->MinTime(default_seconds)
            ->Threads(1);
    }

#endif // NK_COMPARE_TO_BLAS || NK_COMPARE_TO_MKL || NK_COMPARE_TO_ACCELERATE

#if NK_COMPARE_TO_MKL
    // MKL GEMM baselines for matmul comparison
    {
        std::string dims = std::to_string(matmul_dimension_m) + "x" + std::to_string(matmul_dimension_n) + "x" +
                           std::to_string(matmul_dimension_k);
        bm::RegisterBenchmark(("dots_f32_mkl<" + dims + ">").c_str(), measure_dots_f32_mkl, matmul_dimension_m,
                              matmul_dimension_n, matmul_dimension_k)
            ->MinTime(default_seconds)
            ->Threads(1);
        bm::RegisterBenchmark(("dots_bf16_mkl<" + dims + ">").c_str(), measure_dots_bf16_mkl, matmul_dimension_m,
                              matmul_dimension_n, matmul_dimension_k)
            ->MinTime(default_seconds)
            ->Threads(1);
        bm::RegisterBenchmark(("dots_f16_mkl<" + dims + ">").c_str(), measure_dots_f16_mkl, matmul_dimension_m,
                              matmul_dimension_n, matmul_dimension_k)
            ->MinTime(default_seconds)
            ->Threads(1);
        bm::RegisterBenchmark(("dots_u8i8i32_mkl<" + dims + ">").c_str(), measure_dots_u8i8i32_mkl, matmul_dimension_m,
                              matmul_dimension_n, matmul_dimension_k)
            ->MinTime(default_seconds)
            ->Threads(1);
        bm::RegisterBenchmark(("dots_i16i16i32_mkl<" + dims + ">").c_str(), measure_dots_i16i16i32_mkl,
                              matmul_dimension_m, matmul_dimension_n, matmul_dimension_k)
            ->MinTime(default_seconds)
            ->Threads(1);
    }
#endif

#if NK_TARGET_NEON
    dense_<f32_k, f32_k, f64_k>("dot_f32_neon", nk_dot_f32_neon, nk_dot_f32_accurate);
    dense_<f32_k, f32_k, f64_k>("angular_f32_neon", nk_angular_f32_neon, nk_angular_f32_accurate);
    dense_<f32_k, f32_k, f64_k>("l2sq_f32_neon", nk_l2sq_f32_neon, nk_l2sq_f32_accurate);
    dense_<f32_k, f32_k, f64_k>("l2_f32_neon", nk_l2_f32_neon, nk_l2_f32_accurate);
    dense_<f32_k, f32_k, f64_k>("kld_f32_neon", nk_kld_f32_neon, nk_kld_f32_accurate);
    dense_<f32_k, f32_k, f64_k>("jsd_f32_neon", nk_jsd_f32_neon, nk_jsd_f32_accurate);

    dense_<f64_k, f64_k, f64_k>("angular_f64_neon", nk_angular_f64_neon, nk_angular_f64_serial);
    dense_<f64_k, f64_k, f64_k>("l2sq_f64_neon", nk_l2sq_f64_neon, nk_l2sq_f64_serial);
    dense_<f64_k, f64_k, f64_k>("l2_f64_neon", nk_l2_f64_neon, nk_l2_f64_serial);

    dense_<u1_k, u32_k, u32_k>("hamming_u1_neon", nk_hamming_u1_neon, nk_hamming_u1_serial);
    dense_<u1_k, f32_k, f32_k>("jaccard_u1_neon", nk_jaccard_u1_neon, nk_jaccard_u1_serial);

    dense_<f32c_k, f32c_k, f64c_k>("dot_f32c_neon", nk_dot_f32c_neon, nk_dot_f32c_accurate);
    dense_<f32c_k, f32c_k, f64c_k>("vdot_f32c_neon", nk_vdot_f32c_neon, nk_vdot_f32c_accurate);

    curved_<f32_k, f32_k, f64_k>("bilinear_f32_neon", nk_bilinear_f32_neon, nk_bilinear_f32_accurate);
    curved_<f32_k, f32_k, f64_k>("mahalanobis_f32_neon", nk_mahalanobis_f32_neon, nk_mahalanobis_f32_accurate);
    curved_<f32c_k, f32c_k, f64c_k>("bilinear_f32c_neon", nk_bilinear_f32c_neon, nk_bilinear_f32c_accurate);

    sparse_<u16_k, u32_k, u32_k>("intersect_u16_neon", nk_intersect_u16_neon, nk_intersect_u16_accurate);
    sparse_<u32_k, u32_k, u32_k>("intersect_u32_neon", nk_intersect_u32_neon, nk_intersect_u32_accurate);

    elementwise_<f32_k, nk_kernel_fma_k, f32_k, f64_k, f32_k, f64_k>("fma_f32_neon", nk_fma_f32_neon,
                                                                     nk_fma_f32_accurate, nk_l2_f32_accurate);
    elementwise_<f32_k, nk_kernel_wsum_k, f32_k, f64_k, f32_k, f64_k>("wsum_f32_neon", nk_wsum_f32_neon,
                                                                      nk_wsum_f32_accurate, nk_l2_f32_accurate);
    elementwise_<f32_k, nk_kernel_fma_k, f32_k, f64_k, f32_k, f64_k>("fma_f32_serial", nk_fma_f32_serial,
                                                                     nk_fma_f32_accurate, nk_l2_f32_accurate);
    elementwise_<f32_k, nk_kernel_wsum_k, f32_k, f64_k, f32_k, f64_k>("wsum_f32_serial", nk_wsum_f32_serial,
                                                                      nk_wsum_f32_accurate, nk_l2_f32_accurate);

    matmul_<nk_f32_t, nk_f32_t>("dots_packed_f32_neon", nk_dots_packed_size_f32_neon, nk_dots_pack_f32_neon,
                                nk_dots_packed_f32_neon);
    matmul_<nk_f64_t, nk_f64_t>("dots_packed_f64_neon", nk_dots_packed_size_f64_neon, nk_dots_pack_f64_neon,
                                nk_dots_packed_f64_neon);

    mesh_<f32_k, f32_k, f32_k>("rmsd_f32_neon", nk_rmsd_f32_neon, nk_rmsd_f32_serial);
    mesh_<f32_k, f32_k, f32_k>("kabsch_f32_neon", nk_kabsch_f32_neon, nk_kabsch_f32_serial);
    mesh_<f32_k, f32_k, f32_k>("umeyama_f32_neon", nk_umeyama_f32_neon, nk_umeyama_f32_serial);
    mesh_<f64_k, f64_k, f64_k>("rmsd_f64_neon", nk_rmsd_f64_neon, nk_rmsd_f64_serial);
    mesh_<f64_k, f64_k, f64_k>("kabsch_f64_neon", nk_kabsch_f64_neon, nk_kabsch_f64_serial);
    mesh_<f64_k, f64_k, f64_k>("umeyama_f64_neon", nk_umeyama_f64_neon, nk_umeyama_f64_serial);

#endif

#if NK_TARGET_NEONSDOT
    dense_<i8_k, f32_k, f32_k>("angular_i8_neonsdot", nk_angular_i8_neonsdot, nk_angular_i8_serial);
    dense_<i8_k, u32_k, u32_k>("l2sq_i8_neonsdot", nk_l2sq_i8_neonsdot, nk_l2sq_i8_serial);
    dense_<i8_k, f32_k, f64_k>("l2_i8_neonsdot", nk_l2_i8_neonsdot, nk_l2_i8_accurate);
    dense_<i8_k, i32_k, i32_k>("dot_i8_neonsdot", nk_dot_i8_neonsdot, nk_dot_i8_serial);

    dense_<u8_k, f32_k, f32_k>("angular_u8_neonsdot", nk_angular_u8_neonsdot, nk_angular_u8_serial);
    dense_<u8_k, u32_k, u32_k>("l2sq_u8_neonsdot", nk_l2sq_u8_neonsdot, nk_l2sq_u8_serial);
    dense_<u8_k, f32_k, f64_k>("l2_u8_neonsdot", nk_l2_u8_neonsdot, nk_l2_u8_accurate);
    dense_<u8_k, u32_k, u32_k>("dot_u8_neonsdot", nk_dot_u8_neonsdot, nk_dot_u8_serial);

    matmul_<nk_i8_t, nk_i32_t>("dots_packed_i8_neonsdot", nk_dots_packed_size_i8_neonsdot, nk_dots_pack_i8_neonsdot,
                               nk_dots_packed_i8_neonsdot);
    matmul_<nk_u8_t, nk_u32_t>("dots_packed_u8_neonsdot", nk_dots_packed_size_u8_neonsdot, nk_dots_pack_u8_neonsdot,
                               nk_dots_packed_u8_neonsdot);
#endif

#if NK_TARGET_NEONHALF
    dense_<f16c_k, f32c_k, f64c_k>("dot_f16c_neonhalf", nk_dot_f16c_neonhalf, nk_dot_f16c_accurate);
    dense_<f16c_k, f32c_k, f64c_k>("vdot_f16c_neonhalf", nk_vdot_f16c_neonhalf, nk_vdot_f16c_accurate);

    dense_<f16_k, f32_k, f64_k>("dot_f16_neonhalf", nk_dot_f16_neonhalf, nk_dot_f16_accurate);
    dense_<f16_k, f32_k, f64_k>("angular_f16_neonhalf", nk_angular_f16_neonhalf, nk_angular_f16_accurate);
    dense_<f16_k, f32_k, f64_k>("l2sq_f16_neonhalf", nk_l2sq_f16_neonhalf, nk_l2sq_f16_accurate);
    dense_<f16_k, f32_k, f64_k>("l2_f16_neonhalf", nk_l2_f16_neonhalf, nk_l2sq_f16_accurate);
    dense_<f16_k, f32_k, f64_k>("kld_f16_neonhalf", nk_kld_f16_neonhalf, nk_kld_f16_accurate);
    dense_<f16_k, f32_k, f64_k>("jsd_f16_neonhalf", nk_jsd_f16_neonhalf, nk_jsd_f16_accurate);

    curved_<f16_k, f32_k, f64_k>("bilinear_f16_neonhalf", nk_bilinear_f16_neonhalf, nk_bilinear_f16_accurate);
    curved_<f16_k, f32_k, f64_k>("mahalanobis_f16_neonhalf", nk_mahalanobis_f16_neonhalf, nk_mahalanobis_f16_accurate);
    curved_<f16c_k, f32c_k, f64c_k>("bilinear_f16c_neonhalf", nk_bilinear_f16c_neonhalf, nk_bilinear_f16c_accurate);

    elementwise_<f16_k, nk_kernel_fma_k, f32_k, f64_k, f32_k, f64_k>("fma_f16_neonhalf", nk_fma_f16_neonhalf,
                                                                     nk_fma_f16_accurate, nk_l2_f16_accurate);
    elementwise_<f16_k, nk_kernel_wsum_k, f32_k, f64_k, f32_k, f64_k>("wsum_f16_neonhalf", nk_wsum_f16_neonhalf,
                                                                      nk_wsum_f16_accurate, nk_l2_f16_accurate);

    // FMA kernels for `u8` on NEON use `f16` arithmetic
    elementwise_<u8_k, nk_kernel_fma_k, f32_k, f64_k, f32_k, f64_k>("fma_u8_neonhalf", nk_fma_u8_neonhalf,
                                                                    nk_fma_u8_accurate, nk_l2_u8_accurate);
    elementwise_<u8_k, nk_kernel_wsum_k, f32_k, f64_k, f32_k, f64_k>("wsum_u8_neonhalf", nk_wsum_u8_neonhalf,
                                                                     nk_wsum_u8_accurate, nk_l2_u8_accurate);
    elementwise_<i8_k, nk_kernel_fma_k, f32_k, f64_k, f32_k, f64_k>("fma_i8_neonhalf", nk_fma_i8_neonhalf,
                                                                    nk_fma_i8_accurate, nk_l2_i8_accurate);
    elementwise_<i8_k, nk_kernel_wsum_k, f32_k, f64_k, f32_k, f64_k>("wsum_i8_neonhalf", nk_wsum_i8_neonhalf,
                                                                     nk_wsum_i8_accurate, nk_l2_i8_accurate);

    matmul_<nk_f16_t, nk_f32_t>("dots_packed_f16_neonhalf", nk_dots_packed_size_f16_neonhalf, nk_dots_pack_f16_neonhalf,
                                nk_dots_packed_f16_neonhalf);
#endif

#if NK_TARGET_NEONFHM
    dense_<f16_k, f32_k, f64_k>("dot_f16_neonfhm", nk_dot_f16_neonfhm, nk_dot_f16_accurate);

    matmul_<nk_f16_t, nk_f32_t>("dots_packed_f16_neonfhm", nk_dots_packed_size_f16_neonfhm, nk_dots_pack_f16_neonfhm,
                                nk_dots_packed_f16_neonfhm);
#endif // NK_TARGET_NEONFHM

#if NK_TARGET_NEONBFDOT
    dense_<bf16c_k, f32c_k, f64c_k>("dot_bf16c_neonbfdot", nk_dot_bf16c_neonbfdot, nk_dot_bf16c_accurate);
    dense_<bf16c_k, f32c_k, f64c_k>("vdot_bf16c_neonbfdot", nk_vdot_bf16c_neonbfdot, nk_vdot_bf16c_accurate);

    dense_<bf16_k, f32_k, f64_k>("dot_bf16_neonbfdot", nk_dot_bf16_neonbfdot, nk_dot_bf16_accurate);
    dense_<bf16_k, f32_k, f64_k>("angular_bf16_neonbfdot", nk_angular_bf16_neonbfdot, nk_angular_bf16_accurate);
    dense_<bf16_k, f32_k, f64_k>("l2sq_bf16_neonbfdot", nk_l2sq_bf16_neonbfdot, nk_l2sq_bf16_accurate);
    dense_<bf16_k, f32_k, f64_k>("l2_bf16_neonbfdot", nk_l2_bf16_neonbfdot, nk_l2_bf16_accurate);

    curved_<bf16_k, f32_k, f64_k>("bilinear_bf16_neonbfdot", nk_bilinear_bf16_neonbfdot, nk_bilinear_bf16_accurate);
    curved_<bf16_k, f32_k, f64_k>("mahalanobis_bf16_neonbfdot", nk_mahalanobis_bf16_neonbfdot,
                                  nk_mahalanobis_bf16_accurate);
    curved_<bf16c_k, f32c_k, f64c_k>("bilinear_bf16c_neonbfdot", nk_bilinear_bf16c_neonbfdot,
                                     nk_bilinear_bf16c_accurate);

    elementwise_<bf16_k, nk_kernel_fma_k, f32_k, f64_k, f32_k, f64_k>("fma_bf16_neonbfdot", nk_fma_bf16_neonbfdot,
                                                                      nk_fma_bf16_accurate, nk_l2_bf16_accurate);
    elementwise_<bf16_k, nk_kernel_wsum_k, f32_k, f64_k, f32_k, f64_k>("wsum_bf16_neonbfdot", nk_wsum_bf16_neonbfdot,
                                                                       nk_wsum_bf16_accurate, nk_l2_bf16_accurate);

    matmul_<nk_bf16_t, nk_f32_t>("dots_packed_bf16_neonbfdot", nk_dots_packed_size_bf16_neonbfdot,
                                 nk_dots_pack_bf16_neonbfdot, nk_dots_packed_bf16_neonbfdot);
#endif

#if NK_TARGET_SVE
    dense_<f32_k, f32_k, f64_k>("dot_f32_sve", nk_dot_f32_sve, nk_dot_f32_accurate);
    dense_<f32_k, f32_k, f64_k>("angular_f32_sve", nk_angular_f32_sve, nk_angular_f32_accurate);
    dense_<f32_k, f32_k, f64_k>("l2sq_f32_sve", nk_l2sq_f32_sve, nk_l2sq_f32_accurate);
    dense_<f32_k, f32_k, f64_k>("l2_f32_sve", nk_l2_f32_sve, nk_l2_i8_accurate);

    dense_<f64_k, f64_k, f64_k>("dot_f64_sve", nk_dot_f64_sve, nk_dot_f64_serial);
    dense_<f64_k, f64_k, f64_k>("angular_f64_sve", nk_angular_f64_sve, nk_angular_f64_serial);
    dense_<f64_k, f64_k, f64_k>("l2sq_f64_sve", nk_l2sq_f64_sve, nk_l2sq_f64_serial);
    dense_<f64_k, f64_k, f64_k>("l2_f64_sve", nk_l2_f64_sve, nk_l2_f32_accurate);

    dense_<u1_k, u32_k, u32_k>("hamming_u1_sve", nk_hamming_u1_sve, nk_hamming_u1_serial);
    dense_<u1_k, f32_k, f32_k>("jaccard_u1_sve", nk_jaccard_u1_sve, nk_jaccard_u1_serial);

    dense_<f32c_k, f32c_k, f64c_k>("dot_f32c_sve", nk_dot_f32c_sve, nk_dot_f32c_accurate);
    dense_<f32c_k, f32c_k, f64c_k>("vdot_f32c_sve", nk_vdot_f32c_sve, nk_vdot_f32c_accurate);
    dense_<f64c_k, f64c_k, f64c_k>("dot_f64c_sve", nk_dot_f64c_sve, nk_dot_f64c_serial);
    dense_<f64c_k, f64c_k, f64c_k>("vdot_f64c_sve", nk_vdot_f64c_sve, nk_vdot_f64c_serial);
#endif

#if NK_TARGET_SVEHALF
    dense_<f16_k, f32_k, f64_k>("dot_f16_svehalf", nk_dot_f16_svehalf, nk_dot_f16_accurate);
    dense_<f16_k, f32_k, f64_k>("angular_f16_svehalf", nk_angular_f16_svehalf, nk_angular_f16_accurate);
    dense_<f16_k, f32_k, f64_k>("l2sq_f16_svehalf", nk_l2sq_f16_svehalf, nk_l2sq_f16_accurate);
    dense_<f16_k, f32_k, f64_k>("l2_f16_svehalf", nk_l2_f16_svehalf, nk_l2sq_f16_accurate);
    dense_<f16c_k, f32c_k, f64c_k>("dot_f16c_svehalf", nk_dot_f16c_svehalf, nk_dot_f16c_accurate);
    dense_<f16c_k, f32c_k, f64c_k>("vdot_f16c_svehalf", nk_vdot_f16c_svehalf, nk_vdot_f16c_accurate);
#endif

#if NK_TARGET_SVEBFDOT
    dense_<bf16_k, f32_k, f64_k>("angular_bf16_svebfdot", nk_angular_bf16_svebfdot, nk_angular_bf16_accurate);
    dense_<bf16_k, f32_k, f64_k>("l2sq_bf16_svebfdot", nk_l2sq_bf16_svebfdot, nk_l2sq_bf16_accurate);
    dense_<bf16_k, f32_k, f64_k>("l2_bf16_svebfdot", nk_l2_bf16_svebfdot, nk_l2_bf16_accurate);
#endif

#if NK_TARGET_SVE2
    sparse_<u16_k, u32_k, u32_k>("intersect_u16_sve2", nk_intersect_u16_sve2, nk_intersect_u16_serial);
    sparse_<u32_k, u32_k, u32_k>("intersect_u32_sve2", nk_intersect_u32_sve2, nk_intersect_u32_serial);
#endif

#if NK_TARGET_SME
    matmul_<nk_f16_t, nk_f32_t>("dots_packed_f16_sme", nk_dots_packed_size_f16_sme, nk_dots_pack_f16_sme,
                                nk_dots_packed_f16_sme);
    matmul_<nk_bf16_t, nk_f32_t>("dots_packed_bf16_sme", nk_dots_packed_size_bf16_sme, nk_dots_pack_bf16_sme,
                                 nk_dots_packed_bf16_sme);
    matmul_<nk_i8_t, nk_i32_t>("dots_packed_i8_sme", nk_dots_packed_size_i8_sme, nk_dots_pack_i8_sme,
                               nk_dots_packed_i8_sme);
    matmul_<nk_u8_t, nk_u32_t>("dots_packed_u8_sme", nk_dots_packed_size_u8_sme, nk_dots_pack_u8_sme,
                               nk_dots_packed_u8_sme);
    matmul_<nk_e4m3_t, nk_f32_t>("dots_packed_e4m3_sme", nk_dots_packed_size_e4m3_sme, nk_dots_pack_e4m3_sme,
                                 nk_dots_packed_e4m3_sme);
    matmul_<nk_e5m2_t, nk_f32_t>("dots_packed_e5m2_sme", nk_dots_packed_size_e5m2_sme, nk_dots_pack_e5m2_sme,
                                 nk_dots_packed_e5m2_sme);
#endif
#if NK_TARGET_SMEF64
    matmul_<nk_f32_t, nk_f32_t>("dots_packed_f32_smef64", nk_dots_packed_size_f32_smef64, nk_dots_pack_f32_smef64,
                                nk_dots_packed_f32_smef64);
    matmul_<nk_f64_t, nk_f64_t>("dots_packed_f64_smef64", nk_dots_packed_size_f64_smef64, nk_dots_pack_f64_smef64,
                                nk_dots_packed_f64_smef64);

    curved_<f32_k, f32_k, f64_k>("bilinear_f32_smef64", nk_bilinear_f32_smef64, nk_bilinear_f32_serial);
    curved_<f32c_k, f32c_k, f64c_k>("bilinear_f32c_smef64", nk_bilinear_f32c_smef64, nk_bilinear_f32c_serial);
    curved_<f32_k, f32_k, f64_k>("mahalanobis_f32_smef64", nk_mahalanobis_f32_smef64, nk_mahalanobis_f32_serial);
#endif

#if NK_TARGET_HASWELL
    dense_<f16_k, f32_k, f64_k>("dot_f16_haswell", nk_dot_f16_haswell, nk_dot_f16_accurate);
    dense_<f16_k, f32_k, f64_k>("angular_f16_haswell", nk_angular_f16_haswell, nk_angular_f16_accurate);
    dense_<f16_k, f32_k, f64_k>("l2sq_f16_haswell", nk_l2sq_f16_haswell, nk_l2sq_f16_accurate);
    dense_<f16_k, f32_k, f64_k>("l2_f16_haswell", nk_l2_f16_haswell, nk_l2_f16_accurate);
    dense_<f16_k, f32_k, f64_k>("kld_f16_haswell", nk_kld_f16_haswell, nk_kld_f16_accurate);
    dense_<f16_k, f32_k, f64_k>("jsd_f16_haswell", nk_jsd_f16_haswell, nk_jsd_f16_accurate);

    dense_<bf16_k, f32_k, f64_k>("dot_bf16_haswell", nk_dot_bf16_haswell, nk_dot_bf16_accurate);
    dense_<bf16_k, f32_k, f64_k>("angular_bf16_haswell", nk_angular_bf16_haswell, nk_angular_bf16_accurate);
    dense_<bf16_k, f32_k, f64_k>("l2sq_bf16_haswell", nk_l2sq_bf16_haswell, nk_l2sq_bf16_accurate);
    dense_<bf16_k, f32_k, f64_k>("l2_bf16_haswell", nk_l2_bf16_haswell, nk_l2_bf16_accurate);

    dense_<e4m3_k, f32_k, f32_k>("dot_e4m3_haswell", nk_dot_e4m3_haswell, nk_dot_e4m3_serial);
    dense_<e5m2_k, f32_k, f32_k>("dot_e5m2_haswell", nk_dot_e5m2_haswell, nk_dot_e5m2_serial);

    dense_<i8_k, f32_k, f32_k>("angular_i8_haswell", nk_angular_i8_haswell, nk_angular_i8_serial);
    dense_<i8_k, u32_k, u32_k>("l2sq_i8_haswell", nk_l2sq_i8_haswell, nk_l2sq_i8_serial);
    dense_<i8_k, f32_k, f64_k>("l2_i8_haswell", nk_l2_i8_haswell, nk_l2_i8_accurate);
    dense_<i8_k, i32_k, i32_k>("dot_i8_haswell", nk_dot_i8_haswell, nk_dot_i8_serial);

    dense_<u8_k, f32_k, f32_k>("angular_u8_haswell", nk_angular_u8_haswell, nk_angular_u8_serial);
    dense_<u8_k, u32_k, u32_k>("l2sq_u8_haswell", nk_l2sq_u8_haswell, nk_l2sq_u8_serial);
    dense_<u8_k, f32_k, f64_k>("l2_u8_haswell", nk_l2_u8_haswell, nk_l2_u8_accurate);
    dense_<u8_k, u32_k, u32_k>("dot_u8_haswell", nk_dot_u8_haswell, nk_dot_u8_serial);

    dense_<u1_k, u32_k, u32_k>("hamming_u1_haswell", nk_hamming_u1_haswell, nk_hamming_u1_serial);
    dense_<u1_k, f32_k, f32_k>("jaccard_u1_haswell", nk_jaccard_u1_haswell, nk_jaccard_u1_serial);

    dense_<f16c_k, f32c_k, f64c_k>("dot_f16c_haswell", nk_dot_f16c_haswell, nk_dot_f16c_accurate);
    dense_<f16c_k, f32c_k, f64c_k>("vdot_f16c_haswell", nk_vdot_f16c_haswell, nk_vdot_f16c_accurate);
    dense_<f32c_k, f32c_k, f64c_k>("dot_f32c_haswell", nk_dot_f32c_haswell, nk_dot_f32c_accurate);
    dense_<f32c_k, f32c_k, f64c_k>("vdot_f32c_haswell", nk_vdot_f32c_haswell, nk_vdot_f32c_accurate);

    curved_<f16_k, f32_k, f64_k>("bilinear_f16_haswell", nk_bilinear_f16_haswell, nk_bilinear_f16_accurate);
    curved_<f16_k, f32_k, f64_k>("mahalanobis_f16_haswell", nk_mahalanobis_f16_haswell, nk_mahalanobis_f16_accurate);
    curved_<bf16_k, f32_k, f64_k>("bilinear_bf16_haswell", nk_bilinear_bf16_haswell, nk_bilinear_bf16_accurate);
    curved_<bf16_k, f32_k, f64_k>("mahalanobis_bf16_haswell", nk_mahalanobis_bf16_haswell,
                                  nk_mahalanobis_bf16_accurate);

    elementwise_<f64_k, nk_kernel_scale_k, f64_k, f64_k, f64_k, f64_k>("scale_f64_haswell", nk_scale_f64_haswell,
                                                                       nk_scale_f64_serial, nk_l2_f64_serial);
    elementwise_<f64_k, nk_kernel_fma_k, f64_k, f64_k, f64_k, f64_k>("fma_f64_haswell", nk_fma_f64_haswell,
                                                                     nk_fma_f64_serial, nk_l2_f64_serial);
    elementwise_<f64_k, nk_kernel_wsum_k, f64_k, f64_k, f64_k, f64_k>("wsum_f64_haswell", nk_wsum_f64_haswell,
                                                                      nk_wsum_f64_serial, nk_l2_f64_serial);
    elementwise_<f32_k, nk_kernel_scale_k, f32_k, f32_k, f32_k, f32_k>("scale_f32_haswell", nk_scale_f32_haswell,
                                                                       nk_scale_f32_serial, nk_l2_f32_accurate);
    elementwise_<f32_k, nk_kernel_fma_k, f32_k, f32_k, f32_k, f32_k>("fma_f32_haswell", nk_fma_f32_haswell,
                                                                     nk_fma_f32_serial, nk_l2_f32_accurate);
    elementwise_<f32_k, nk_kernel_wsum_k, f32_k, f32_k, f32_k, f32_k>("wsum_f32_haswell", nk_wsum_f32_haswell,
                                                                      nk_wsum_f32_serial, nk_l2_f32_accurate);
    elementwise_<f16_k, nk_kernel_scale_k, f32_k, f32_k, f32_k, f32_k>("scale_f16_haswell", nk_scale_f16_haswell,
                                                                       nk_scale_f16_serial, nk_l2_f16_accurate);
    elementwise_<f16_k, nk_kernel_fma_k, f32_k, f32_k, f32_k, f32_k>("fma_f16_haswell", nk_fma_f16_haswell,
                                                                     nk_fma_f16_serial, nk_l2_f16_accurate);
    elementwise_<f16_k, nk_kernel_wsum_k, f32_k, f32_k, f32_k, f32_k>("wsum_f16_haswell", nk_wsum_f16_haswell,
                                                                      nk_wsum_f16_serial, nk_l2_f16_accurate);
    elementwise_<bf16_k, nk_kernel_scale_k, f32_k, f32_k, f32_k, f32_k>("scale_bf16_haswell", nk_scale_bf16_haswell,
                                                                        nk_scale_bf16_serial, nk_l2_bf16_accurate);
    elementwise_<bf16_k, nk_kernel_fma_k, f32_k, f32_k, f32_k, f32_k>("fma_bf16_haswell", nk_fma_bf16_haswell,
                                                                      nk_fma_bf16_serial, nk_l2_bf16_accurate);
    elementwise_<bf16_k, nk_kernel_wsum_k, f32_k, f32_k, f32_k, f32_k>("wsum_bf16_haswell", nk_wsum_bf16_haswell,
                                                                       nk_wsum_bf16_serial, nk_l2_bf16_accurate);
    elementwise_<i8_k, nk_kernel_scale_k, f32_k, f32_k, f32_k, f32_k>("scale_i8_haswell", nk_scale_i8_haswell,
                                                                      nk_scale_i8_serial, nk_l2_i8_accurate);
    elementwise_<i8_k, nk_kernel_fma_k, f32_k, f32_k, f32_k, f32_k>("fma_i8_haswell", nk_fma_i8_haswell,
                                                                    nk_fma_i8_serial, nk_l2_i8_accurate);
    elementwise_<i8_k, nk_kernel_wsum_k, f32_k, f32_k, f32_k, f32_k>("wsum_i8_haswell", nk_wsum_i8_haswell,
                                                                     nk_wsum_i8_serial, nk_l2_i8_accurate);
    elementwise_<u8_k, nk_kernel_scale_k, f32_k, f32_k, f32_k, f32_k>("scale_u8_haswell", nk_scale_u8_haswell,
                                                                      nk_scale_u8_serial, nk_l2_u8_accurate);
    elementwise_<u8_k, nk_kernel_fma_k, f32_k, f32_k, f32_k, f32_k>("fma_u8_haswell", nk_fma_u8_haswell,
                                                                    nk_fma_u8_serial, nk_l2_u8_accurate);
    elementwise_<u8_k, nk_kernel_wsum_k, f32_k, f32_k, f32_k, f32_k>("wsum_u8_haswell", nk_wsum_u8_haswell,
                                                                     nk_wsum_u8_serial, nk_l2_u8_accurate);
    elementwise_<i16_k, nk_kernel_scale_k, f32_k, f32_k, f32_k, f32_k>("scale_i16_haswell", nk_scale_i16_haswell,
                                                                       nk_scale_i16_serial, l2_with_stl<nk_i16_t>);
    elementwise_<i16_k, nk_kernel_fma_k, f32_k, f32_k, f32_k, f32_k>("fma_i16_haswell", nk_fma_i16_haswell,
                                                                     nk_fma_i16_serial, l2_with_stl<nk_i16_t>);
    elementwise_<u16_k, nk_kernel_scale_k, f32_k, f32_k, f32_k, f32_k>("scale_u16_haswell", nk_scale_u16_haswell,
                                                                       nk_scale_u16_serial, l2_with_stl<nk_u16_t>);
    elementwise_<u16_k, nk_kernel_fma_k, f32_k, f32_k, f32_k, f32_k>("fma_u16_haswell", nk_fma_u16_haswell,
                                                                     nk_fma_u16_serial, l2_with_stl<nk_u16_t>);

    geospatial_<f32_k, f32_k, f64_k>("haversine_f32_haswell", nk_haversine_f32_haswell,
                                     haversine_with_stl<nk_f32_t, nk_f64_t>, l2_with_stl<nk_f64_t>);
    geospatial_<f64_k, f64_k, f64_k>("haversine_f64_haswell", nk_haversine_f64_haswell, haversine_with_stl<nk_f64_t>,
                                     l2_with_stl<nk_f64_t>);
    geospatial_<f32_k, f32_k, f64_k>("vincenty_f32_haswell", nk_vincenty_f32_haswell,
                                     vincenty_with_stl<nk_f32_t, nk_f64_t>, l2_with_stl<nk_f64_t>);
    geospatial_<f64_k, f64_k, f64_k>("vincenty_f64_haswell", nk_vincenty_f64_haswell, vincenty_with_stl<nk_f64_t>,
                                     l2_with_stl<nk_f64_t>);

    elementwise_<f32_k, nk_kernel_unknown_k, f32_k, f32_k, f32_k, f32_k>(
        "sin_f32_haswell", nk_sin_f32_haswell, elementwise_with_stl<nk_f32_t, sin_with_stl<nk_f64_t>>,
        l2_with_stl<nk_f32_t>);
    elementwise_<f32_k, nk_kernel_unknown_k, f32_k, f32_k, f32_k, f32_k>(
        "cos_f32_haswell", nk_cos_f32_haswell, elementwise_with_stl<nk_f32_t, cos_with_stl<nk_f64_t>>,
        l2_with_stl<nk_f32_t>);
    elementwise_<f32_k, nk_kernel_unknown_k, f32_k, f32_k, f32_k, f32_k>(
        "atan_f32_haswell", nk_atan_f32_haswell, elementwise_with_stl<nk_f32_t, atan_with_stl<nk_f64_t>>,
        l2_with_stl<nk_f32_t>);
    elementwise_<f64_k, nk_kernel_unknown_k, f64_k, f64_k, f64_k, f64_k>(
        "sin_f64_haswell", nk_sin_f64_haswell, elementwise_with_stl<nk_f64_t, sin_with_stl<nk_f64_t>>,
        l2_with_stl<nk_f64_t>);
    elementwise_<f64_k, nk_kernel_unknown_k, f64_k, f64_k, f64_k, f64_k>(
        "cos_f64_haswell", nk_cos_f64_haswell, elementwise_with_stl<nk_f64_t, cos_with_stl<nk_f64_t>>,
        l2_with_stl<nk_f64_t>);
    elementwise_<f64_k, nk_kernel_unknown_k, f64_k, f64_k, f64_k, f64_k>(
        "atan_f64_haswell", nk_atan_f64_haswell, elementwise_with_stl<nk_f64_t, atan_with_stl<nk_f64_t>>,
        l2_with_stl<nk_f64_t>);

    matmul_<nk_f32_t, nk_f32_t>("dots_packed_f32_haswell", nk_dots_packed_size_f32_haswell, nk_dots_pack_f32_haswell,
                                nk_dots_packed_f32_haswell);
    matmul_<nk_f64_t, nk_f64_t>("dots_packed_f64_haswell", nk_dots_packed_size_f64_haswell, nk_dots_pack_f64_haswell,
                                nk_dots_packed_f64_haswell);

#endif

#if NK_TARGET_SKYLAKE
    dense_<f32_k, f32_k, f64_k>("dot_f32_skylake", nk_dot_f32_skylake, nk_dot_f32_accurate);
    dense_<f32_k, f32_k, f64_k>("angular_f32_skylake", nk_angular_f32_skylake, nk_angular_f32_accurate);
    dense_<f32_k, f32_k, f64_k>("l2sq_f32_skylake", nk_l2sq_f32_skylake, nk_l2sq_f32_accurate);
    dense_<f32_k, f32_k, f64_k>("l2_f32_skylake", nk_l2_f32_skylake, nk_l2_f32_accurate);
    dense_<f32_k, f32_k, f64_k>("kld_f32_skylake", nk_kld_f32_skylake, nk_kld_f32_accurate);
    dense_<f32_k, f32_k, f64_k>("jsd_f32_skylake", nk_jsd_f32_skylake, nk_jsd_f32_accurate);

    dense_<f32c_k, f32c_k, f64c_k>("dot_f32c_skylake", nk_dot_f32c_skylake, nk_dot_f32c_accurate);
    dense_<f32c_k, f32c_k, f64c_k>("vdot_f32c_skylake", nk_vdot_f32c_skylake, nk_vdot_f32c_accurate);
    dense_<f64c_k, f64c_k, f64c_k>("dot_f64c_skylake", nk_dot_f64c_skylake, nk_dot_f64c_serial);
    dense_<f64c_k, f64c_k, f64c_k>("vdot_f64c_skylake", nk_vdot_f64c_skylake, nk_vdot_f64c_serial);

    dense_<e4m3_k, f32_k, f32_k>("dot_e4m3_skylake", nk_dot_e4m3_skylake, nk_dot_e4m3_serial);
    dense_<e5m2_k, f32_k, f32_k>("dot_e5m2_skylake", nk_dot_e5m2_skylake, nk_dot_e5m2_serial);

    elementwise_<f64_k, nk_kernel_fma_k, f64_k, f64_k, f64_k, f64_k>("fma_f64_skylake", nk_fma_f64_skylake,
                                                                     nk_fma_f64_serial, nk_l2_f64_serial);
    elementwise_<f64_k, nk_kernel_wsum_k, f64_k, f64_k, f64_k, f64_k>("wsum_f64_skylake", nk_wsum_f64_skylake,
                                                                      nk_wsum_f64_serial, nk_l2_f64_serial);
    elementwise_<f32_k, nk_kernel_fma_k, f32_k, f32_k, f32_k, f32_k>("fma_f32_skylake", nk_fma_f32_skylake,
                                                                     nk_fma_f32_serial, nk_l2_f32_accurate);
    elementwise_<f32_k, nk_kernel_wsum_k, f32_k, f32_k, f32_k, f32_k>("wsum_f32_skylake", nk_wsum_f32_skylake,
                                                                      nk_wsum_f32_serial, nk_l2_f32_accurate);
    elementwise_<bf16_k, nk_kernel_fma_k, f32_k, f32_k, f32_k, f32_k>("fma_bf16_skylake", nk_fma_bf16_skylake,
                                                                      nk_fma_bf16_serial, nk_l2_bf16_accurate);
    elementwise_<bf16_k, nk_kernel_wsum_k, f32_k, f32_k, f32_k, f32_k>("wsum_bf16_skylake", nk_wsum_bf16_skylake,
                                                                       nk_wsum_bf16_serial, nk_l2_bf16_accurate);

    elementwise_<f32_k, nk_kernel_unknown_k, f32_k, f32_k, f32_k, f32_k>(
        "sin_f32_skylake", nk_sin_f32_skylake, elementwise_with_stl<nk_f32_t, sin_with_stl<nk_f64_t>>,
        l2_with_stl<nk_f32_t>);
    elementwise_<f32_k, nk_kernel_unknown_k, f32_k, f32_k, f32_k, f32_k>(
        "cos_f32_skylake", nk_cos_f32_skylake, elementwise_with_stl<nk_f32_t, cos_with_stl<nk_f64_t>>,
        l2_with_stl<nk_f32_t>);
    elementwise_<f32_k, nk_kernel_unknown_k, f32_k, f32_k, f32_k, f32_k>(
        "atan_f32_skylake", nk_atan_f32_skylake, elementwise_with_stl<nk_f32_t, atan_with_stl<nk_f64_t>>,
        l2_with_stl<nk_f32_t>);
    elementwise_<f64_k, nk_kernel_unknown_k, f64_k, f64_k, f64_k, f64_k>(
        "sin_f64_skylake", nk_sin_f64_skylake, elementwise_with_stl<nk_f64_t, sin_with_stl<nk_f64_t>>,
        l2_with_stl<nk_f64_t>);
    elementwise_<f64_k, nk_kernel_unknown_k, f64_k, f64_k, f64_k, f64_k>(
        "cos_f64_skylake", nk_cos_f64_skylake, elementwise_with_stl<nk_f64_t, cos_with_stl<nk_f64_t>>,
        l2_with_stl<nk_f64_t>);
    elementwise_<f64_k, nk_kernel_unknown_k, f64_k, f64_k, f64_k, f64_k>(
        "atan_f64_skylake", nk_atan_f64_skylake, elementwise_with_stl<nk_f64_t, atan_with_stl<nk_f64_t>>,
        l2_with_stl<nk_f64_t>);

    curved_<f32_k, f32_k, f32_k>("bilinear_f32_skylake", nk_bilinear_f32_skylake, nk_bilinear_f32_serial);
    curved_<f32c_k, f32c_k, f32c_k>("bilinear_f32c_skylake", nk_bilinear_f32c_skylake, nk_bilinear_f32c_serial);
    curved_<f64_k, f64_k, f64_k>("bilinear_f64_skylake", nk_bilinear_f64_skylake, nk_bilinear_f64_serial);
    curved_<f64c_k, f64c_k, f64c_k>("bilinear_f64c_skylake", nk_bilinear_f64c_skylake, nk_bilinear_f64c_serial);

    geospatial_<f32_k, f32_k, f64_k>("haversine_f32_skylake", nk_haversine_f32_skylake,
                                     haversine_with_stl<nk_f32_t, nk_f64_t>, l2_with_stl<nk_f64_t>);
    geospatial_<f64_k, f64_k, f64_k>("haversine_f64_skylake", nk_haversine_f64_skylake, haversine_with_stl<nk_f64_t>,
                                     l2_with_stl<nk_f64_t>);
    geospatial_<f32_k, f32_k, f64_k>("vincenty_f32_skylake", nk_vincenty_f32_skylake,
                                     vincenty_with_stl<nk_f32_t, nk_f64_t>, l2_with_stl<nk_f64_t>);
    geospatial_<f64_k, f64_k, f64_k>("vincenty_f64_skylake", nk_vincenty_f64_skylake, vincenty_with_stl<nk_f64_t>,
                                     l2_with_stl<nk_f64_t>);

    mesh_<f32_k, f32_k, f32_k>("rmsd_f32_skylake", nk_rmsd_f32_skylake, nk_rmsd_f32_serial);
    mesh_<f32_k, f32_k, f32_k>("kabsch_f32_skylake", nk_kabsch_f32_skylake, nk_kabsch_f32_serial);

    matmul_<nk_f32_t, nk_f32_t>("dots_packed_f32_skylake", nk_dots_packed_size_f32_skylake, nk_dots_pack_f32_skylake,
                                nk_dots_packed_f32_skylake);
    matmul_<nk_f64_t, nk_f64_t>("dots_packed_f64_skylake", nk_dots_packed_size_f64_skylake, nk_dots_pack_f64_skylake,
                                nk_dots_packed_f64_skylake);
    matmul_<nk_e4m3_t, nk_f32_t>("dots_packed_e4m3_skylake", nk_dots_packed_size_e4m3_skylake,
                                 nk_dots_pack_e4m3_skylake, nk_dots_packed_e4m3_skylake);
    matmul_<nk_e5m2_t, nk_f32_t>("dots_packed_e5m2_skylake", nk_dots_packed_size_e5m2_skylake,
                                 nk_dots_pack_e5m2_skylake, nk_dots_packed_e5m2_skylake);

#endif

#if NK_TARGET_ICE
    matmul_<nk_i8_t, nk_i32_t>("dots_packed_i8_ice", nk_dots_packed_size_i8_ice, nk_dots_pack_i8_ice,
                               nk_dots_packed_i8_ice);

    dense_<i8_k, f32_k, f32_k>("angular_i8_ice", nk_angular_i8_ice, nk_angular_i8_serial);
    dense_<i8_k, u32_k, u32_k>("l2sq_i8_ice", nk_l2sq_i8_ice, nk_l2sq_i8_serial);
    dense_<i8_k, f32_k, f64_k>("l2_i8_ice", nk_l2_i8_ice, nk_l2_i8_accurate);
    dense_<i8_k, i32_k, i32_k>("dot_i8_ice", nk_dot_i8_ice, nk_dot_i8_serial);

    dense_<u8_k, f32_k, f32_k>("angular_u8_ice", nk_angular_u8_ice, nk_angular_u8_serial);
    dense_<u8_k, u32_k, u32_k>("l2sq_u8_ice", nk_l2sq_u8_ice, nk_l2sq_u8_serial);
    dense_<u8_k, f32_k, f64_k>("l2_u8_ice", nk_l2_u8_ice, nk_l2_u8_accurate);
    dense_<u8_k, u32_k, u32_k>("dot_u8_ice", nk_dot_u8_ice, nk_dot_u8_serial);

    dense_<i4_k, f32_k, f32_k>("angular_i4_ice", nk_angular_i4_ice, nk_angular_i4_serial);
    dense_<i4_k, u32_k, u32_k>("l2sq_i4_ice", nk_l2sq_i4_ice, nk_l2sq_i4_serial);
    dense_<i4_k, f32_k, f32_k>("l2_i4_ice", nk_l2_i4_ice, nk_l2_i4_serial);
    dense_<i4_k, i32_k, i32_k>("dot_i4_ice", nk_dot_i4_ice, nk_dot_i4_serial);

    dense_<u4_k, f32_k, f32_k>("angular_u4_ice", nk_angular_u4_ice, nk_angular_u4_serial);
    dense_<u4_k, u32_k, u32_k>("l2sq_u4_ice", nk_l2sq_u4_ice, nk_l2sq_u4_serial);
    dense_<u4_k, f32_k, f32_k>("l2_u4_ice", nk_l2_u4_ice, nk_l2_u4_serial);
    dense_<u4_k, u32_k, u32_k>("dot_u4_ice", nk_dot_u4_ice, nk_dot_u4_serial);

    dense_<f64_k, f64_k, f64_k>("dot_f64_skylake", nk_dot_f64_skylake, nk_dot_f64_serial);
    dense_<f64_k, f64_k, f64_k>("angular_f64_skylake", nk_angular_f64_skylake, nk_angular_f64_serial);
    dense_<f64_k, f64_k, f64_k>("l2sq_f64_skylake", nk_l2sq_f64_skylake, nk_l2sq_f64_serial);
    dense_<f64_k, f64_k, f64_k>("l2_f64_skylake", nk_l2_f64_skylake, nk_l2_f64_serial);

    dense_<u1_k, u32_k, u32_k>("hamming_u1_ice", nk_hamming_u1_ice, nk_hamming_u1_serial);
    dense_<u1_k, f32_k, f32_k>("jaccard_u1_ice", nk_jaccard_u1_ice, nk_jaccard_u1_serial);

    sparse_<u16_k, u32_k, u32_k>("intersect_u16_ice", nk_intersect_u16_ice, nk_intersect_u16_accurate);
    sparse_<u32_k, u32_k, u32_k>("intersect_u32_ice", nk_intersect_u32_ice, nk_intersect_u32_accurate);
#endif

#if NK_TARGET_GENOA
    dense_<bf16_k, f32_k, f64_k>("dot_bf16_genoa", nk_dot_bf16_genoa, nk_dot_bf16_accurate);
    dense_<bf16_k, f32_k, f64_k>("angular_bf16_genoa", nk_angular_bf16_genoa, nk_angular_bf16_accurate);
    dense_<bf16_k, f32_k, f64_k>("l2sq_bf16_genoa", nk_l2sq_bf16_genoa, nk_l2sq_bf16_accurate);
    dense_<bf16_k, f32_k, f64_k>("l2_bf16_genoa", nk_l2_bf16_genoa, nk_l2_bf16_accurate);
    dense_<bf16c_k, f32c_k, f64c_k>("dot_bf16c_genoa", nk_dot_bf16c_genoa, nk_dot_bf16c_accurate);
    dense_<bf16c_k, f32c_k, f64c_k>("vdot_bf16c_genoa", nk_vdot_bf16c_genoa, nk_vdot_bf16c_accurate);

    dense_<e4m3_k, f32_k, f32_k>("dot_e4m3_genoa", nk_dot_e4m3_genoa, nk_dot_e4m3_serial);
    dense_<e5m2_k, f32_k, f32_k>("dot_e5m2_genoa", nk_dot_e5m2_genoa, nk_dot_e5m2_serial);

    curved_<bf16_k, f32_k, f64_k>("bilinear_bf16_genoa", nk_bilinear_bf16_genoa, nk_bilinear_bf16_accurate);
    curved_<bf16_k, f32_k, f64_k>("mahalanobis_bf16_genoa", nk_mahalanobis_bf16_genoa, nk_mahalanobis_bf16_accurate);
    curved_<bf16c_k, f32c_k, f64c_k>("bilinear_bf16c_genoa", nk_bilinear_bf16c_genoa, nk_bilinear_bf16c_accurate);

    matmul_<nk_bf16_t, nk_f32_t>("dots_packed_bf16_genoa", nk_dots_packed_size_bf16_genoa, nk_dots_pack_bf16_genoa,
                                 nk_dots_packed_bf16_genoa);

    matmul_<nk_e4m3_t, nk_f32_t>("dots_packed_e4m3_genoa", nk_dots_packed_size_e4m3_genoa, nk_dots_pack_e4m3_genoa,
                                 nk_dots_packed_e4m3_genoa);
    matmul_<nk_e5m2_t, nk_f32_t>("dots_packed_e5m2_genoa", nk_dots_packed_size_e5m2_genoa, nk_dots_pack_e5m2_genoa,
                                 nk_dots_packed_e5m2_genoa);

#endif

#if NK_TARGET_SAPPHIRE
    dense_<f16_k, f32_k, f64_k>("kld_f16_sapphire", nk_kld_f16_sapphire, nk_kld_f16_accurate);
    dense_<f16_k, f32_k, f64_k>("jsd_f16_sapphire", nk_jsd_f16_sapphire, nk_jsd_f16_accurate);
    dense_<e4m3_k, f32_k, f32_k>("l2_e4m3_sapphire", nk_l2_e4m3_sapphire, nk_l2_e4m3_serial);
    dense_<e4m3_k, f32_k, f32_k>("l2sq_e4m3_sapphire", nk_l2sq_e4m3_sapphire, nk_l2sq_e4m3_serial);

    elementwise_<u8_k, nk_kernel_fma_k, f32_k, f32_k, f32_k, f32_k>("fma_u8_sapphire", nk_fma_u8_sapphire,
                                                                    nk_fma_u8_serial, nk_l2_u8_accurate);
    elementwise_<u8_k, nk_kernel_wsum_k, f32_k, f32_k, f32_k, f32_k>("wsum_u8_sapphire", nk_wsum_u8_sapphire,
                                                                     nk_wsum_u8_serial, nk_l2_u8_accurate);
    elementwise_<i8_k, nk_kernel_fma_k, f32_k, f32_k, f32_k, f32_k>("fma_i8_sapphire", nk_fma_i8_sapphire,
                                                                    nk_fma_i8_serial, nk_l2_i8_accurate);
    elementwise_<i8_k, nk_kernel_wsum_k, f32_k, f32_k, f32_k, f32_k>("wsum_i8_sapphire", nk_wsum_i8_sapphire,
                                                                     nk_wsum_i8_serial, nk_l2_i8_accurate);

    curved_<f16_k, f32_k, f64_k>("bilinear_f16_sapphire", nk_bilinear_f16_sapphire, nk_bilinear_f16_accurate);
    curved_<f16_k, f32_k, f64_k>("mahalanobis_f16_sapphire", nk_mahalanobis_f16_sapphire, nk_mahalanobis_f16_accurate);
    curved_<f16c_k, f32c_k, f64c_k>("bilinear_f16c_sapphire", nk_bilinear_f16c_sapphire, nk_bilinear_f16c_accurate);

    elementwise_<f16_k, nk_kernel_unknown_k, f32_k, f32_k, f32_k, f32_k>(
        "sin_f16_sapphire", nk_sin_f16_sapphire, elementwise_with_stl<nk_f16_t, sin_with_stl<nk_f64_t>>,
        nk_l2_f16_accurate);
    elementwise_<f16_k, nk_kernel_unknown_k, f32_k, f32_k, f32_k, f32_k>(
        "cos_f16_sapphire", nk_cos_f16_sapphire, elementwise_with_stl<nk_f16_t, cos_with_stl<nk_f64_t>>,
        nk_l2_f16_accurate);
    elementwise_<f16_k, nk_kernel_unknown_k, f32_k, f32_k, f32_k, f32_k>(
        "atan_f16_sapphire", nk_atan_f16_sapphire, elementwise_with_stl<nk_f16_t, atan_with_stl<nk_f64_t>>,
        nk_l2_f16_accurate);

    matmul_<nk_bf16_t, nk_f32_t>("dots_packed_bf16_sapphire_amx", nk_dots_packed_size_bf16_sapphire_amx,
                                 nk_dots_pack_bf16_sapphire_amx, nk_dots_packed_bf16_sapphire_amx);
    matmul_<nk_i8_t, nk_i32_t>("dots_packed_i8_sapphire_amx", nk_dots_packed_size_i8_sapphire_amx,
                               nk_dots_pack_i8_sapphire_amx, nk_dots_packed_i8_sapphire_amx);
    matmul_<nk_e4m3_t, nk_f32_t>("dots_packed_e4m3_sapphire_amx", nk_dots_packed_size_e4m3_sapphire_amx,
                                 nk_dots_pack_e4m3_sapphire_amx, nk_dots_packed_e4m3_sapphire_amx);
    matmul_<nk_e5m2_t, nk_f32_t>("dots_packed_e5m2_sapphire_amx", nk_dots_packed_size_e5m2_sapphire_amx,
                                 nk_dots_pack_e5m2_sapphire_amx, nk_dots_packed_e5m2_sapphire_amx);

#endif

#if NK_TARGET_TURIN
    sparse_<u16_k, u32_k, u32_k>("intersect_u16_turin", nk_intersect_u16_turin, nk_intersect_u16_accurate);
    sparse_<u32_k, u32_k, u32_k>("intersect_u32_turin", nk_intersect_u32_turin, nk_intersect_u32_accurate);
#endif

    sparse_<u16_k, u32_k, u32_k>("intersect_u16_serial", nk_intersect_u16_serial, nk_intersect_u16_accurate);
    sparse_<u16_k, u32_k, u32_k>("intersect_u16_accurate", nk_intersect_u16_accurate, nk_intersect_u16_accurate);
    sparse_<u32_k, u32_k, u32_k>("intersect_u32_serial", nk_intersect_u32_serial, nk_intersect_u32_accurate);
    sparse_<u32_k, u32_k, u32_k>("intersect_u32_accurate", nk_intersect_u32_accurate, nk_intersect_u32_accurate);

    curved_<f64_k, f64_k, f64_k>("bilinear_f64_serial", nk_bilinear_f64_serial, nk_bilinear_f64_serial);
    curved_<f64c_k, f64c_k, f64c_k>("bilinear_f64c_serial", nk_bilinear_f64c_serial, nk_bilinear_f64c_serial);
    curved_<f64_k, f64_k, f64_k>("mahalanobis_f64_serial", nk_mahalanobis_f64_serial, nk_mahalanobis_f64_serial);
    curved_<f32_k, f32_k, f64_k>("bilinear_f32_serial", nk_bilinear_f32_serial, nk_bilinear_f32_accurate);
    curved_<f32c_k, f32c_k, f64c_k>("bilinear_f32c_serial", nk_bilinear_f32c_serial, nk_bilinear_f32c_accurate);
    curved_<f32_k, f32_k, f64_k>("mahalanobis_f32_serial", nk_mahalanobis_f32_serial, nk_mahalanobis_f32_accurate);
    curved_<f16_k, f32_k, f64_k>("bilinear_f16_serial", nk_bilinear_f16_serial, nk_bilinear_f16_accurate);
    curved_<f16c_k, f32c_k, f64c_k>("bilinear_f16c_serial", nk_bilinear_f16c_serial, nk_bilinear_f16c_accurate);
    curved_<f16_k, f32_k, f64_k>("mahalanobis_f16_serial", nk_mahalanobis_f16_serial, nk_mahalanobis_f16_accurate);
    curved_<bf16_k, f32_k, f64_k>("bilinear_bf16_serial", nk_bilinear_bf16_serial, nk_bilinear_bf16_accurate);
    curved_<bf16c_k, f32c_k, f64c_k>("bilinear_bf16c_serial", nk_bilinear_bf16c_serial, nk_bilinear_bf16c_accurate);
    curved_<bf16_k, f32_k, f64_k>("mahalanobis_bf16_serial", nk_mahalanobis_bf16_serial, nk_mahalanobis_bf16_accurate);

    mesh_<f32_k, f32_k, f32_k>("rmsd_f32_serial", nk_rmsd_f32_serial, nk_rmsd_f32_serial);
    mesh_<f32_k, f32_k, f32_k>("kabsch_f32_serial", nk_kabsch_f32_serial, nk_kabsch_f32_serial);
    mesh_<f32_k, f32_k, f32_k>("umeyama_f32_serial", nk_umeyama_f32_serial, nk_umeyama_f32_serial);
    mesh_<f64_k, f64_k, f64_k>("rmsd_f64_serial", nk_rmsd_f64_serial, nk_rmsd_f64_serial);
    mesh_<f64_k, f64_k, f64_k>("kabsch_f64_serial", nk_kabsch_f64_serial, nk_kabsch_f64_serial);
    mesh_<f64_k, f64_k, f64_k>("umeyama_f64_serial", nk_umeyama_f64_serial, nk_umeyama_f64_serial);

    dense_<bf16_k, f32_k, f64_k>("dot_bf16_serial", nk_dot_bf16_serial, nk_dot_bf16_accurate);
    dense_<bf16_k, f32_k, f64_k>("angular_bf16_serial", nk_angular_bf16_serial, nk_angular_bf16_accurate);
    dense_<bf16_k, f32_k, f64_k>("l2sq_bf16_serial", nk_l2sq_bf16_serial, nk_l2sq_bf16_accurate);
    dense_<bf16_k, f32_k, f64_k>("l2_bf16_serial", nk_l2_bf16_serial, nk_l2_bf16_accurate);
    dense_<bf16_k, f32_k, f64_k>("kld_bf16_serial", nk_kld_bf16_serial, nk_kld_bf16_accurate);
    dense_<bf16_k, f32_k, f64_k>("jsd_bf16_serial", nk_jsd_bf16_serial, nk_jsd_bf16_accurate);

    dense_<e4m3_k, f32_k, f32_k>("dot_e4m3_serial", nk_dot_e4m3_serial, nk_dot_e4m3_serial);
    dense_<e5m2_k, f32_k, f32_k>("dot_e5m2_serial", nk_dot_e5m2_serial, nk_dot_e5m2_serial);

    dense_<f16_k, f32_k, f64_k>("dot_f16_serial", nk_dot_f16_serial, nk_dot_f16_accurate);
    dense_<f16_k, f32_k, f64_k>("angular_f16_serial", nk_angular_f16_serial, nk_angular_f16_accurate);
    dense_<f16_k, f32_k, f64_k>("l2sq_f16_serial", nk_l2sq_f16_serial, nk_l2sq_f16_accurate);
    dense_<f16_k, f32_k, f64_k>("l2_f16_serial", nk_l2_f16_serial, nk_l2_f16_accurate);
    dense_<f16_k, f32_k, f64_k>("kld_f16_serial", nk_kld_f16_serial, nk_kld_f16_accurate);
    dense_<f16_k, f32_k, f64_k>("jsd_f16_serial", nk_jsd_f16_serial, nk_jsd_f16_accurate);

    dense_<f32_k, f32_k, f64_k>("dot_f32_serial", nk_dot_f32_serial, nk_dot_f32_accurate);
    dense_<f32_k, f32_k, f64_k>("angular_f32_serial", nk_angular_f32_serial, nk_angular_f32_accurate);
    dense_<f32_k, f32_k, f64_k>("l2sq_f32_serial", nk_l2sq_f32_serial, nk_l2sq_f32_accurate);
    dense_<f32_k, f32_k, f64_k>("l2_f32_serial", nk_l2_f32_serial, nk_l2_f32_accurate);
    dense_<f32_k, f32_k, f64_k>("kld_f32_serial", nk_kld_f32_serial, nk_kld_f32_accurate);
    dense_<f32_k, f32_k, f64_k>("jsd_f32_serial", nk_jsd_f32_serial, nk_jsd_f32_accurate);

    dense_<f64_k, f64_k, f64_k>("dot_f64_serial", nk_dot_f64_serial, nk_dot_f64_serial);
    dense_<f64_k, f64_k, f64_k>("angular_f64_serial", nk_angular_f64_serial, nk_angular_f64_serial);
    dense_<f64_k, f64_k, f64_k>("l2sq_f64_serial", nk_l2sq_f64_serial, nk_l2sq_f64_serial);
    dense_<f64_k, f64_k, f64_k>("l2_f64_serial", nk_l2_f64_serial, nk_l2_f64_serial);

    dense_<i8_k, f32_k, f32_k>("angular_i8_serial", nk_angular_i8_serial, nk_angular_i8_serial);
    dense_<i8_k, u32_k, u32_k>("l2sq_i8_serial", nk_l2sq_i8_serial, nk_l2sq_i8_serial);
    dense_<i8_k, f32_k, f64_k>("l2_i8_serial", nk_l2_i8_serial, nk_l2_i8_accurate);
    dense_<i8_k, i32_k, i32_k>("dot_i8_serial", nk_dot_i8_serial, nk_dot_i8_serial);

    dense_<u8_k, f32_k, f32_k>("angular_u8_serial", nk_angular_u8_serial, nk_angular_u8_serial);
    dense_<u8_k, u32_k, u32_k>("l2sq_u8_serial", nk_l2sq_u8_serial, nk_l2sq_u8_serial);
    dense_<u8_k, f32_k, f64_k>("l2_u8_serial", nk_l2_u8_serial, nk_l2_u8_accurate);
    dense_<u8_k, u32_k, u32_k>("dot_u8_serial", nk_dot_u8_serial, nk_dot_u8_serial);

    dense_<i4_k, f32_k, f32_k>("angular_i4_serial", nk_angular_i4_serial, nk_angular_i4_serial);
    dense_<i4_k, u32_k, u32_k>("l2sq_i4_serial", nk_l2sq_i4_serial, nk_l2sq_i4_serial);
    dense_<i4_k, f32_k, f32_k>("l2_i4_serial", nk_l2_i4_serial, nk_l2_i4_serial);
    dense_<i4_k, i32_k, i32_k>("dot_i4_serial", nk_dot_i4_serial, nk_dot_i4_serial);

    dense_<u4_k, f32_k, f32_k>("angular_u4_serial", nk_angular_u4_serial, nk_angular_u4_serial);
    dense_<u4_k, u32_k, u32_k>("l2sq_u4_serial", nk_l2sq_u4_serial, nk_l2sq_u4_serial);
    dense_<u4_k, f32_k, f32_k>("l2_u4_serial", nk_l2_u4_serial, nk_l2_u4_serial);
    dense_<u4_k, u32_k, u32_k>("dot_u4_serial", nk_dot_u4_serial, nk_dot_u4_serial);

    dense_<f64c_k, f64c_k, f64c_k>("dot_f64c_serial", nk_dot_f64c_serial, nk_dot_f64c_serial);
    dense_<f32c_k, f32c_k, f64c_k>("dot_f32c_serial", nk_dot_f32c_serial, nk_dot_f32c_accurate);
    dense_<f16c_k, f32c_k, f64c_k>("dot_f16c_serial", nk_dot_f16c_serial, nk_dot_f16c_accurate);
    dense_<bf16c_k, f32c_k, f64c_k>("dot_bf16c_serial", nk_dot_bf16c_serial, nk_dot_bf16c_accurate);
    dense_<f64c_k, f64c_k, f64c_k>("vdot_f64c_serial", nk_vdot_f64c_serial, nk_vdot_f64c_serial);
    dense_<f32c_k, f32c_k, f64c_k>("vdot_f32c_serial", nk_vdot_f32c_serial, nk_vdot_f32c_accurate);
    dense_<f16c_k, f32c_k, f64c_k>("vdot_f16c_serial", nk_vdot_f16c_serial, nk_vdot_f16c_accurate);
    dense_<bf16c_k, f32c_k, f64c_k>("vdot_bf16c_serial", nk_vdot_bf16c_serial, nk_vdot_bf16c_accurate);

    dense_<u1_k, u32_k, u32_k>("hamming_u1_serial", nk_hamming_u1_serial, nk_hamming_u1_serial);
    dense_<u1_k, f32_k, f32_k>("jaccard_u1_serial", nk_jaccard_u1_serial, nk_jaccard_u1_serial);

    elementwise_<f32_k, nk_kernel_unknown_k, f32_k, f32_k, f32_k, f32_k>(
        "sin_f32_stl", elementwise_with_stl<nk_f32_t, sin_with_stl<nk_f32_t>>,
        elementwise_with_stl<nk_f32_t, sin_with_stl<nk_f64_t>>, l2_with_stl<nk_f32_t>);
    elementwise_<f32_k, nk_kernel_unknown_k, f32_k, f32_k, f32_k, f32_k>(
        "cos_f32_stl", elementwise_with_stl<nk_f32_t, cos_with_stl<nk_f32_t>>,
        elementwise_with_stl<nk_f32_t, cos_with_stl<nk_f64_t>>, l2_with_stl<nk_f32_t>);
    elementwise_<f32_k, nk_kernel_unknown_k, f32_k, f32_k, f32_k, f32_k>(
        "atan_f32_stl", elementwise_with_stl<nk_f32_t, atan_with_stl<nk_f32_t>>,
        elementwise_with_stl<nk_f32_t, atan_with_stl<nk_f64_t>>, l2_with_stl<nk_f32_t>);
    elementwise_<f32_k, nk_kernel_unknown_k, f32_k, f32_k, f32_k, f32_k>(
        "sin_f32_serial", nk_sin_f32_serial, elementwise_with_stl<nk_f32_t, sin_with_stl<nk_f64_t>>,
        l2_with_stl<nk_f32_t>);
    elementwise_<f32_k, nk_kernel_unknown_k, f32_k, f32_k, f32_k, f32_k>(
        "cos_f32_serial", nk_cos_f32_serial, elementwise_with_stl<nk_f32_t, cos_with_stl<nk_f64_t>>,
        l2_with_stl<nk_f32_t>);
    elementwise_<f32_k, nk_kernel_unknown_k, f32_k, f32_k, f32_k, f32_k>(
        "atan_f32_serial", nk_atan_f32_serial, elementwise_with_stl<nk_f32_t, atan_with_stl<nk_f64_t>>,
        l2_with_stl<nk_f32_t>);
    elementwise_<f64_k, nk_kernel_unknown_k, f64_k, f64_k, f64_k, f64_k>(
        "sin_f64_stl", elementwise_with_stl<nk_f64_t, sin_with_stl<nk_f64_t>>,
        elementwise_with_stl<nk_f64_t, sin_with_stl<nk_f64_t>>, l2_with_stl<nk_f64_t>);
    elementwise_<f64_k, nk_kernel_unknown_k, f64_k, f64_k, f64_k, f64_k>(
        "cos_f64_stl", elementwise_with_stl<nk_f64_t, cos_with_stl<nk_f64_t>>,
        elementwise_with_stl<nk_f64_t, cos_with_stl<nk_f64_t>>, l2_with_stl<nk_f64_t>);
    elementwise_<f64_k, nk_kernel_unknown_k, f64_k, f64_k, f64_k, f64_k>(
        "atan_f64_stl", elementwise_with_stl<nk_f64_t, atan_with_stl<nk_f64_t>>,
        elementwise_with_stl<nk_f64_t, atan_with_stl<nk_f64_t>>, l2_with_stl<nk_f64_t>);
    elementwise_<f64_k, nk_kernel_unknown_k, f64_k, f64_k, f64_k, f64_k>(
        "sin_f64_serial", nk_sin_f64_serial, elementwise_with_stl<nk_f64_t, sin_with_stl<nk_f64_t>>,
        l2_with_stl<nk_f64_t>);
    elementwise_<f64_k, nk_kernel_unknown_k, f64_k, f64_k, f64_k, f64_k>(
        "cos_f64_serial", nk_cos_f64_serial, elementwise_with_stl<nk_f64_t, cos_with_stl<nk_f64_t>>,
        l2_with_stl<nk_f64_t>);
    elementwise_<f64_k, nk_kernel_unknown_k, f64_k, f64_k, f64_k, f64_k>(
        "atan_f64_serial", nk_atan_f64_serial, elementwise_with_stl<nk_f64_t, atan_with_stl<nk_f64_t>>,
        l2_with_stl<nk_f64_t>);

    elementwise_<f16_k, nk_kernel_unknown_k, f32_k, f32_k, f32_k, f32_k>(
        "sin_f16_serial", nk_sin_f16_serial, elementwise_with_stl<nk_f16_t, sin_with_stl<nk_f64_t>>,
        nk_l2_f16_accurate);
    elementwise_<f16_k, nk_kernel_unknown_k, f32_k, f32_k, f32_k, f32_k>(
        "cos_f16_serial", nk_cos_f16_serial, elementwise_with_stl<nk_f16_t, cos_with_stl<nk_f64_t>>,
        nk_l2_f16_accurate);
    elementwise_<f16_k, nk_kernel_unknown_k, f32_k, f32_k, f32_k, f32_k>(
        "atan_f16_serial", nk_atan_f16_serial, elementwise_with_stl<nk_f16_t, atan_with_stl<nk_f64_t>>,
        nk_l2_f16_accurate);

    elementwise_<f16_k, nk_kernel_fma_k, f32_k, f64_k, f32_k, f64_k>("fma_f16_serial", nk_fma_f16_serial,
                                                                     nk_fma_f16_accurate, nk_l2_f16_accurate);
    elementwise_<f16_k, nk_kernel_wsum_k, f32_k, f64_k, f32_k, f64_k>("wsum_f16_serial", nk_wsum_f16_serial,
                                                                      nk_wsum_f16_accurate, nk_l2_f16_accurate);
    elementwise_<u8_k, nk_kernel_fma_k, f32_k, f64_k, f32_k, f64_k>("fma_u8_serial", nk_fma_u8_serial,
                                                                    nk_fma_u8_accurate, nk_l2_u8_accurate);
    elementwise_<u8_k, nk_kernel_wsum_k, f32_k, f64_k, f32_k, f64_k>("wsum_u8_serial", nk_wsum_u8_serial,
                                                                     nk_wsum_u8_accurate, nk_l2_u8_accurate);
    elementwise_<i8_k, nk_kernel_fma_k, f32_k, f64_k, f32_k, f64_k>("fma_i8_serial", nk_fma_i8_serial,
                                                                    nk_fma_i8_accurate, nk_l2_i8_accurate);
    elementwise_<i8_k, nk_kernel_wsum_k, f32_k, f64_k, f32_k, f64_k>("wsum_i8_serial", nk_wsum_i8_serial,
                                                                     nk_wsum_i8_accurate, nk_l2_i8_accurate);

    geospatial_<f32_k, f32_k, f64_k>("haversine_f32_serial", nk_haversine_f32_serial,
                                     haversine_with_stl<nk_f32_t, nk_f64_t>, l2_with_stl<nk_f64_t>);
    geospatial_<f64_k, f64_k, f64_k>("haversine_f64_serial", nk_haversine_f64_serial, haversine_with_stl<nk_f64_t>,
                                     l2_with_stl<nk_f64_t>);
    geospatial_<f32_k, f32_k, f64_k>("vincenty_f32_serial", nk_vincenty_f32_serial,
                                     vincenty_with_stl<nk_f32_t, nk_f64_t>, l2_with_stl<nk_f64_t>);
    geospatial_<f64_k, f64_k, f64_k>("vincenty_f64_serial", nk_vincenty_f64_serial, vincenty_with_stl<nk_f64_t>,
                                     l2_with_stl<nk_f64_t>);

    matmul_<nk_bf16_t, nk_f32_t>("dots_packed_bf16_serial", nk_dots_packed_size_bf16_serial, nk_dots_pack_bf16_serial,
                                 nk_dots_packed_bf16_serial);
    matmul_<nk_i8_t, nk_i32_t>("dots_packed_i8_serial", nk_dots_packed_size_i8_serial, nk_dots_pack_i8_serial,
                               nk_dots_packed_i8_serial);
    matmul_<nk_f32_t, nk_f32_t>("dots_packed_f32_serial", nk_dots_packed_size_f32_serial, nk_dots_pack_f32_serial,
                                nk_dots_packed_f32_serial);
    matmul_<nk_u1x8_t, nk_u32_t>("dots_packed_u1_serial", nk_dots_packed_size_u1x8_serial, nk_dots_pack_u1x8_serial,
                                 nk_dots_packed_u1x8_serial);
    matmul_<nk_u4x2_t, nk_u32_t>("dots_packed_u4_serial", nk_dots_packed_size_u4x2_serial, nk_dots_pack_u4x2_serial,
                                 nk_dots_packed_u4x2_serial);
    matmul_<nk_i4x2_t, nk_i32_t>("dots_packed_i4_serial", nk_dots_packed_size_i4x2_serial, nk_dots_pack_i4x2_serial,
                                 nk_dots_packed_i4x2_serial);

    cast_<nk_f32_k, nk_f16_k>("cast_f32_to_f16_serial", nk_cast_serial);
    cast_<nk_f16_k, nk_f32_k>("cast_f16_to_f32_serial", nk_cast_serial);
    cast_<nk_f32_k, nk_bf16_k>("cast_f32_to_bf16_serial", nk_cast_serial);
    cast_<nk_bf16_k, nk_f32_k>("cast_bf16_to_f32_serial", nk_cast_serial);
    cast_<nk_f32_k, nk_e4m3_k>("cast_f32_to_e4m3_serial", nk_cast_serial);
    cast_<nk_e4m3_k, nk_f32_k>("cast_e4m3_to_f32_serial", nk_cast_serial);
    cast_<nk_f32_k, nk_e5m2_k>("cast_f32_to_e5m2_serial", nk_cast_serial);
    cast_<nk_e5m2_k, nk_f32_k>("cast_e5m2_to_f32_serial", nk_cast_serial);
    cast_<nk_f64_k, nk_f32_k>("cast_f64_to_f32_serial", nk_cast_serial);
    cast_<nk_f32_k, nk_f64_k>("cast_f32_to_f64_serial", nk_cast_serial);

#if NK_TARGET_HASWELL
    cast_<nk_f32_k, nk_f16_k>("cast_f32_to_f16_haswell", nk_cast_haswell);
    cast_<nk_f16_k, nk_f32_k>("cast_f16_to_f32_haswell", nk_cast_haswell);
    cast_<nk_f32_k, nk_bf16_k>("cast_f32_to_bf16_haswell", nk_cast_haswell);
    cast_<nk_bf16_k, nk_f32_k>("cast_bf16_to_f32_haswell", nk_cast_haswell);
    cast_<nk_f32_k, nk_e4m3_k>("cast_f32_to_e4m3_haswell", nk_cast_haswell);
    cast_<nk_e4m3_k, nk_f32_k>("cast_e4m3_to_f32_haswell", nk_cast_haswell);
    cast_<nk_f32_k, nk_e5m2_k>("cast_f32_to_e5m2_haswell", nk_cast_haswell);
    cast_<nk_e5m2_k, nk_f32_k>("cast_e5m2_to_f32_haswell", nk_cast_haswell);
#endif

#if NK_TARGET_SKYLAKE
    cast_<nk_f32_k, nk_f16_k>("cast_f32_to_f16_skylake", nk_cast_skylake);
    cast_<nk_f16_k, nk_f32_k>("cast_f16_to_f32_skylake", nk_cast_skylake);
    cast_<nk_f32_k, nk_bf16_k>("cast_f32_to_bf16_skylake", nk_cast_skylake);
    cast_<nk_bf16_k, nk_f32_k>("cast_bf16_to_f32_skylake", nk_cast_skylake);
    cast_<nk_f32_k, nk_e4m3_k>("cast_f32_to_e4m3_skylake", nk_cast_skylake);
    cast_<nk_e4m3_k, nk_f32_k>("cast_e4m3_to_f32_skylake", nk_cast_skylake);
    cast_<nk_f32_k, nk_e5m2_k>("cast_f32_to_e5m2_skylake", nk_cast_skylake);
    cast_<nk_e5m2_k, nk_f32_k>("cast_e5m2_to_f32_skylake", nk_cast_skylake);
#endif

#if NK_TARGET_ICE
    cast_<nk_f32_k, nk_f16_k>("cast_f32_to_f16_ice", nk_cast_ice);
    cast_<nk_f16_k, nk_f32_k>("cast_f16_to_f32_ice", nk_cast_ice);
    cast_<nk_f32_k, nk_e4m3_k>("cast_f32_to_e4m3_ice", nk_cast_ice);
    cast_<nk_e4m3_k, nk_f32_k>("cast_e4m3_to_f32_ice", nk_cast_ice);
#endif

#if NK_TARGET_SAPPHIRE
    cast_<nk_f32_k, nk_f16_k>("cast_f32_to_f16_sapphire", nk_cast_sapphire);
    cast_<nk_f16_k, nk_f32_k>("cast_f16_to_f32_sapphire", nk_cast_sapphire);
#endif

    bm::RunSpecifiedBenchmarks();
    bm::Shutdown();
    return 0;
}

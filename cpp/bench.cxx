#include <cmath>         // `std::sqrt`
#include <cstdlib>       // `std::aligned_alloc`
#include <cstring>       // `std::memcpy`
#include <random>        // `std::uniform_int_distribution`
#include <thread>        // `std::thread`
#include <unordered_set> // `std::unordered_set`
#include <vector>        // `std::vector`

#include <benchmark/benchmark.h>

#if !defined(SIMSIMD_BUILD_BENCHMARKS_WITH_CBLAS)
#define SIMSIMD_BUILD_BENCHMARKS_WITH_CBLAS 0
#endif
#if SIMSIMD_BUILD_BENCHMARKS_WITH_CBLAS
#include <cblas.h>
#endif

// It's important to note, that out compression/decompression routines
// are quite inaccurate. They are not meant to be used in production code.
// So when benchmarking, if possible, please use the native types, if those
// are implemented.
#define SIMSIMD_NATIVE_F16 1
#define SIMSIMD_NATIVE_BF16 1
#include <simsimd/simsimd.h>

constexpr std::size_t default_seconds = 10;
constexpr std::size_t default_threads = 1;
constexpr simsimd_distance_t signaling_distance = std::numeric_limits<simsimd_distance_t>::signaling_NaN();

/// Matches OpenAI embedding size
constexpr std::size_t dense_dimensions = 128;
/// Has quadratic impact on the number of operations
constexpr std::size_t curved_dimensions = 128;

namespace bm = benchmark;

// clang-format off
template <simsimd_datatype_t> struct datatype_enum_to_type_gt { using value_t = void; };
template <> struct datatype_enum_to_type_gt<simsimd_datatype_f64_k> { using value_t = simsimd_f64_t; };
template <> struct datatype_enum_to_type_gt<simsimd_datatype_f32_k> { using value_t = simsimd_f32_t; };
template <> struct datatype_enum_to_type_gt<simsimd_datatype_f16_k> { using value_t = simsimd_f16_t; };
template <> struct datatype_enum_to_type_gt<simsimd_datatype_bf16_k> { using value_t = simsimd_bf16_t; };
template <> struct datatype_enum_to_type_gt<simsimd_datatype_f64c_k> { using value_t = simsimd_f64_t; };
template <> struct datatype_enum_to_type_gt<simsimd_datatype_f32c_k> { using value_t = simsimd_f32_t; };
template <> struct datatype_enum_to_type_gt<simsimd_datatype_f16c_k> { using value_t = simsimd_f16_t; };
template <> struct datatype_enum_to_type_gt<simsimd_datatype_bf16c_k> { using value_t = simsimd_bf16_t; };
template <> struct datatype_enum_to_type_gt<simsimd_datatype_b8_k> { using value_t = simsimd_b8_t; };
template <> struct datatype_enum_to_type_gt<simsimd_datatype_i8_k> { using value_t = simsimd_i8_t; };
template <> struct datatype_enum_to_type_gt<simsimd_datatype_u8_k> { using value_t = simsimd_u8_t; };
template <> struct datatype_enum_to_type_gt<simsimd_datatype_i16_k> { using value_t = simsimd_i16_t; };
template <> struct datatype_enum_to_type_gt<simsimd_datatype_u16_k> { using value_t = simsimd_u16_t; };
template <> struct datatype_enum_to_type_gt<simsimd_datatype_i32_k> { using value_t = simsimd_i32_t; };
template <> struct datatype_enum_to_type_gt<simsimd_datatype_u32_k> { using value_t = simsimd_u32_t; };
template <> struct datatype_enum_to_type_gt<simsimd_datatype_i64_k> { using value_t = simsimd_i64_t; };
template <> struct datatype_enum_to_type_gt<simsimd_datatype_u64_k> { using value_t = simsimd_u64_t; };
// clang-format on

template <std::size_t multiple> std::size_t divide_round_up(std::size_t n) {
    return ((n + multiple - 1) / multiple) * multiple;
}

/**
 *  @brief Vector-like fixed capacity buffer, ensuring cache-line alignment.
 *  @tparam datatype_ak The data type of the vector elements, represented as a `simsimd_datatype_t`.
 */
template <simsimd_datatype_t datatype_ak> struct vector_gt {
    using scalar_t = typename datatype_enum_to_type_gt<datatype_ak>::value_t;
    using compressed16_t = unsigned short;
    static constexpr bool is_integral =
        datatype_ak == datatype_ak == simsimd_datatype_b8_k ||                            //
        datatype_ak == simsimd_datatype_i8_k || datatype_ak == simsimd_datatype_u8_k ||   //
        datatype_ak == simsimd_datatype_i16_k || datatype_ak == simsimd_datatype_u16_k || //
        datatype_ak == simsimd_datatype_i32_k || datatype_ak == simsimd_datatype_u32_k ||
        datatype_ak == simsimd_datatype_i64_k || datatype_ak == simsimd_datatype_u64_k;
    static constexpr std::size_t cacheline_length = 64;

    scalar_t* buffer_ = nullptr;
    std::size_t dimensions_ = 0;

    vector_gt() = default;
    vector_gt(std::size_t dimensions) noexcept(false)
        : dimensions_(dimensions),
          buffer_(static_cast<scalar_t*>(
              std::aligned_alloc(cacheline_length, divide_round_up<cacheline_length>(dimensions * sizeof(scalar_t))))) {
        if (!buffer_)
            throw std::bad_alloc();
    }

    ~vector_gt() noexcept { std::free(buffer_); }

    vector_gt(vector_gt const& other) : vector_gt(other.dimensions()) {
        std::memcpy(buffer_, other.buffer_, divide_round_up<cacheline_length>(dimensions_ * sizeof(scalar_t)));
    }
    vector_gt& operator=(vector_gt const& other) {
        if (this != &other) {
            if (dimensions_ != other.dimensions()) {
                std::free(buffer_);
                dimensions_ = other.dimensions();
                buffer_ = static_cast<scalar_t*>(std::aligned_alloc(
                    cacheline_length, divide_round_up<cacheline_length>(dimensions_ * sizeof(scalar_t))));
                if (!buffer_)
                    throw std::bad_alloc();
            }
            std::memcpy(buffer_, other.buffer_, divide_round_up<cacheline_length>(dimensions_ * sizeof(scalar_t)));
        }
        return *this;
    }

    std::size_t dimensions() const noexcept { return dimensions_; }
    std::size_t size_bytes() const noexcept {
        return divide_round_up<cacheline_length>(dimensions_ * sizeof(scalar_t));
    }
    scalar_t const* data() const noexcept { return buffer_; }

    /**
     *  @brief Broadcast a scalar value to all elements of the vector.
     *  @param v The scalar value to broadcast.
     */
    void set(scalar_t v) noexcept {
        for (std::size_t i = 0; i != dimensions_; ++i)
            buffer_[i] = v;
    }

    /**
     *  @brief Compresses a double value into the vector's scalar type.
     *  @param from The double value to compress.
     *  @param to The scalar type where the compressed value will be stored.
     */
    static void compress(double const& from, scalar_t& to) noexcept {
        // In a NaN, the sign bit is irrelevant, mantissa describes the kind of NaN,
        // and the exponent is all ones - we can only check the the exponent bits.
        // Brain float is similar: https://en.wikipedia.org/wiki/Bfloat16_floating-point_format
        constexpr unsigned short exponent_mask_f16 = 0b0111110000000000;               // 1 sign, 5 exp, 10 mantissa
        constexpr unsigned short exponent_mask_bf16 = 0b0111111110000000;              // 1 sign, 8 exp, 7 mantissa
        constexpr unsigned int exponent_mask_f32 = 0b01111111100000000000000000000000; // 1 sign, 8 exp, 23 mantissa
        constexpr unsigned long long exponent_mask_f64 =                               // 1 sign, 11 exp, 52 mantissa
            0b011111111110000000000000000000000000000000000000000000000000000;

#if !SIMSIMD_NATIVE_BF16
        if constexpr (datatype_ak == simsimd_datatype_bf16_k || datatype_ak == simsimd_datatype_bf16c_k) {
            simsimd_f32_to_bf16(from, &to);
            if ((to & exponent_mask_bf16) == exponent_mask_bf16)
                to = 0;
            static_assert(sizeof(scalar_t) == sizeof(simsimd_bf16_t));
            return;
        }
#endif
#if !SIMSIMD_NATIVE_F16
        if constexpr (datatype_ak == simsimd_datatype_f16_k || datatype_ak == simsimd_datatype_f16c_k) {
            simsimd_f32_to_f16(from, &to);
            if ((to & exponent_mask_f16) == exponent_mask_f16)
                to = 0;
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
    static double uncompress(scalar_t const& from) noexcept {
#if !SIMSIMD_NATIVE_BF16
        if constexpr (datatype_ak == simsimd_datatype_bf16_k || datatype_ak == simsimd_datatype_bf16c_k) {
            return simsimd_bf16_to_f32((simsimd_bf16_t const*)&from);
        }
#endif
#if !SIMSIMD_NATIVE_F16
        if constexpr (datatype_ak == simsimd_datatype_f16_k || datatype_ak == simsimd_datatype_f16c_k) {
            return simsimd_f16_to_f32((simsimd_f16_t const*)&from);
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
    void randomize() noexcept {

        std::random_device random_device;
        std::mt19937 generator(random_device());

        if constexpr (std::is_integral_v<scalar_t>) {
            std::uniform_int_distribution<scalar_t> distribution(0, std::numeric_limits<scalar_t>::max());
            for (std::size_t i = 0; i != dimensions_; ++i) {
                buffer_[i] = distribution(generator);
            }
        } else {
            // Using non-uniform distribution helps detect tail errors
            std::normal_distribution<double> distribution(0.1, 1.0);
            double squared_sum = 0.0;
            for (std::size_t i = 0; i != dimensions_; ++i) {
                double a_i = distribution(generator);
                squared_sum += a_i * a_i;
                compress(a_i, buffer_[i]);
            }

            // Normalize the vectors:
            squared_sum = std::sqrt(squared_sum);
            for (std::size_t i = 0; i != dimensions_; ++i) {
                compress(uncompress(buffer_[i]) / squared_sum, buffer_[i]);
                // Zero out NaNs
                if (std::isnan(uncompress(buffer_[i])))
                    buffer_[i] = 0;
            }
        }
    }
};

template <simsimd_datatype_t datatype_ak> struct vectors_pair_gt {
    using vector_t = vector_gt<datatype_ak>;
    using scalar_t = typename vector_t::scalar_t;
    static constexpr bool is_integral = vector_t::is_integral;

    vector_t a;
    vector_t b;

    vectors_pair_gt() noexcept = default;
    vectors_pair_gt(std::size_t dimensions) noexcept : a(dimensions), b(dimensions) {}
    vectors_pair_gt(std::size_t dimensions_a, std::size_t dimensions_b) noexcept : a(dimensions_a), b(dimensions_b) {}
    vectors_pair_gt(vectors_pair_gt const& other) noexcept(false) : a(other.a), b(other.b) {}
    vectors_pair_gt& operator=(vectors_pair_gt const& other) noexcept(false) {
        if (this != &other)
            a = other.a, b = other.b;
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
void measure_dense(bm::State& state, metric_at metric, metric_at baseline, std::size_t dimensions) {

    using pair_t = pair_at;
    using vector_t = typename pair_at::vector_t;

    auto call_baseline = [&](pair_t& pair) -> double {
        // Output for real vectors have a single dimensions.
        // Output for complex vectors have two dimensions.
        simsimd_distance_t results[2] = {signaling_distance, signaling_distance};
        baseline(pair.a.data(), pair.b.data(), pair.a.dimensions(), &results[0]);
        return results[0];
    };
    auto call_contender = [&](pair_t& pair) -> double {
        // Output for real vectors have a single dimensions.
        // Output for complex vectors have two dimensions.
        simsimd_distance_t results[2] = {signaling_distance, signaling_distance};
        metric(pair.a.data(), pair.b.data(), pair.a.dimensions(), &results[0]);
        return results[0];
    };

    // Let's average the distance results over many pairs.
    constexpr std::size_t pairs_count = 128;
    std::vector<pair_t> pairs(pairs_count);
    for (auto& pair : pairs) {
        pair.a = pair.b = vector_t(dimensions);
        pair.a.randomize(), pair.b.randomize();
    }

    // Initialize the output buffers for distance calculations.
    std::vector<double> results_baseline(pairs.size());
    std::vector<double> results_contender(pairs.size());
    for (std::size_t i = 0; i != pairs.size(); ++i)
        results_baseline[i] = call_baseline(pairs[i]), results_contender[i] = call_contender(pairs[i]);

    // The actual benchmarking loop.
    std::size_t iterations = 0;
    for (auto _ : state)
        bm::DoNotOptimize((results_contender[iterations & (pairs_count - 1)] =
                               call_contender(pairs[iterations & (pairs_count - 1)]))),
            iterations++;

    // Measure the mean absolute delta and relative error.
    double mean_delta = 0, mean_relative_error = 0;
    for (std::size_t i = 0; i != pairs.size(); ++i) {
        auto abs_delta = std::abs(results_contender[i] - results_baseline[i]);
        mean_delta += abs_delta;
        double error = abs_delta != 0 && results_baseline[i] != 0 ? abs_delta / results_baseline[i] : 0;
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
void measure_curved(bm::State& state, metric_at metric, metric_at baseline, std::size_t dimensions) {

    using pair_t = pair_at;
    using vector_t = typename pair_at::vector_t;

    auto call_baseline = [&](pair_t const& pair, vector_t const& tensor) -> double {
        simsimd_distance_t result = signaling_distance;
        baseline(pair.a.data(), pair.b.data(), tensor.data(), pair.a.dimensions(), &result);
        return result;
    };
    auto call_contender = [&](pair_t const& pair, vector_t const& tensor) -> double {
        simsimd_distance_t result = signaling_distance;
        metric(pair.a.data(), pair.b.data(), tensor.data(), pair.a.dimensions(), &result);
        return result;
    };

    // Let's average the distance results over many pairs.
    constexpr std::size_t pairs_count = 128;
    std::vector<pair_t> pairs(pairs_count);
    std::vector<vector_t> tensors(pairs_count);
    for (std::size_t i = 0; i != pairs.size(); ++i) {
        pair_t& pair = pairs[i];
        pair.a = pair.b = vector_t(dimensions);
        pair.a.randomize(), pair.b.randomize();
        vector_t& tensor = tensors[i];
        tensor = vector_t(dimensions * dimensions);
        tensor.randomize();
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
        double error = abs_delta != 0 && results_baseline[i] != 0 ? abs_delta / results_baseline[i] : 0;
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
 *  @param dimensions_a The number of dimensions in the smaller vector.
 *  @param dimensions_b The number of dimensions in the larger vector.
 *  @param intersection_size The expected number of common scalars between the vectors.
 */
template <typename pair_at, typename metric_at = void>
void measure_sparse(bm::State& state, metric_at metric, metric_at baseline, std::size_t dimensions_a,
                    std::size_t dimensions_b, std::size_t intersection_size) {

    using pair_t = pair_at;
    using vector_t = typename pair_at::vector_t;
    using scalar_t = typename vector_t::scalar_t;

    auto call_baseline = [&](pair_t& pair) -> double {
        simsimd_distance_t result = std::numeric_limits<simsimd_distance_t>::signaling_NaN();
        baseline(pair.a.data(), pair.b.data(), pair.a.dimensions(), pair.b.dimensions(), &result);
        return result;
    };
    auto call_contender = [&](pair_t& pair) -> double {
        simsimd_distance_t result = std::numeric_limits<simsimd_distance_t>::signaling_NaN();
        metric(pair.a.data(), pair.b.data(), pair.a.dimensions(), pair.b.dimensions(), &result);
        return result;
    };

    // Let's average the distance results over many pairs.
    constexpr std::size_t pairs_count = 128;
    std::vector<pair_t> pairs(pairs_count);
    std::random_device seed_source;
    std::mt19937 generator(seed_source());
    std::uniform_int_distribution<scalar_t> distribution(0, std::numeric_limits<scalar_t>::max());

    // Randomizing the vectors for sparse distances is a bit more complex then:
    //
    //      pair.a.randomize(), pair.b.randomize();
    //
    // We need to ensure that the intersection is of the expected size.
    std::unordered_set<scalar_t> intersection_set, unique_a, unique_b;
    intersection_set.reserve(intersection_size);
    unique_a.reserve(dimensions_a - intersection_size);
    unique_b.reserve(dimensions_b - intersection_size);

    for (auto& pair : pairs) {
        pair.a = vector_t(dimensions_a);
        pair.b = vector_t(dimensions_b);

        // Step 1: Generate intersection set
        intersection_set.clear();
        while (intersection_set.size() < intersection_size)
            intersection_set.insert(distribution(generator));

        unique_a.clear();
        while (unique_a.size() < dimensions_a - intersection_size) {
            scalar_t element = distribution(generator);
            if (intersection_set.find(element) == intersection_set.end())
                unique_a.insert(element);
        }

        unique_b.clear();
        while (unique_b.size() < dimensions_b - intersection_size) {
            scalar_t element = distribution(generator);
            if (intersection_set.find(element) == intersection_set.end() && unique_a.find(element) == unique_a.end())
                unique_b.insert(element);
        }

        // Step 2: Merge and sort
        std::copy(intersection_set.begin(), intersection_set.end(), pair.a.buffer_);
        std::copy(intersection_set.begin(), intersection_set.end(), pair.b.buffer_);
        std::copy(unique_a.begin(), unique_a.end(), pair.a.buffer_ + intersection_size);
        std::copy(unique_b.begin(), unique_b.end(), pair.b.buffer_ + intersection_size);
        std::sort(pair.a.buffer_, pair.a.buffer_ + dimensions_a);
        std::sort(pair.b.buffer_, pair.b.buffer_ + dimensions_b);
    }

    // Initialize the output buffers for distance calculations.
    std::vector<double> results_baseline(pairs.size());
    std::vector<double> results_contender(pairs.size());
    for (std::size_t i = 0; i != pairs.size(); ++i)
        results_baseline[i] = call_baseline(pairs[i]), results_contender[i] = call_contender(pairs[i]);

    // The actual benchmarking loop.
    std::size_t iterations = 0;
    for (auto _ : state)
        bm::DoNotOptimize((results_contender[iterations & (pairs_count - 1)] =
                               call_contender(pairs[iterations & (pairs_count - 1)]))),
            iterations++;

    // Measure the mean absolute delta and relative error.
    double mean_error = 0;
    for (std::size_t i = 0; i != pairs.size(); ++i) {
        auto abs_error = std::abs(results_contender[i] - results_baseline[i]);
        mean_error += abs_error;
    }
    mean_error /= pairs.size();
    state.counters["error"] = mean_error;
    state.counters["bytes"] =
        bm::Counter(iterations * (pairs[0].a.size_bytes() + pairs[0].b.size_bytes()), bm::Counter::kIsRate);
    state.counters["pairs"] = bm::Counter(iterations, bm::Counter::kIsRate);
    state.counters["matches"] =
        std::accumulate(results_contender.begin(), results_contender.end(), 0.0) / results_contender.size();
}

template <simsimd_datatype_t datatype_ak, typename metric_at = void>
void dense_(std::string name, metric_at* distance_func, metric_at* baseline_func) {
    using pair_t = vectors_pair_gt<datatype_ak>;
    std::string bench_name = name + "<" + std::to_string(dense_dimensions) + "d>";
    bm::RegisterBenchmark(bench_name.c_str(), measure_dense<pair_t, metric_at*>, distance_func, baseline_func,
                          dense_dimensions)
        ->MinTime(default_seconds)
        ->Threads(default_threads);
}

template <simsimd_datatype_t datatype_ak, typename metric_at = void>
void sparse_(std::string name, metric_at* distance_func, metric_at* baseline_func) {

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
                if (second_len > 8192)
                    continue;
                bm::RegisterBenchmark(bench_name.c_str(), measure_sparse<pair_t, metric_at*>, distance_func,
                                      baseline_func, first_len, second_len, intersection_size)
                    ->MinTime(default_seconds)
                    ->Threads(default_threads);
            }
        }
    }
}

template <simsimd_datatype_t datatype_ak, typename metric_at = void>
void curved_(std::string name, metric_at* distance_func, metric_at* baseline_func) {

    using pair_t = vectors_pair_gt<datatype_ak>;
    std::string bench_name = name + "<" + std::to_string(curved_dimensions) + "d>";
    bm::RegisterBenchmark(bench_name.c_str(), measure_curved<pair_t, metric_at*>, distance_func, baseline_func,
                          curved_dimensions)
        ->MinTime(default_seconds)
        ->Threads(default_threads);
}

#if SIMSIMD_BUILD_BENCHMARKS_WITH_CBLAS

void dot_f32_blas(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t n, simsimd_distance_t* result) {
    *result = cblas_sdot((int)n, a, 1, b, 1);
}

void dot_f64_blas(simsimd_f64_t const* a, simsimd_f64_t const* b, simsimd_size_t n, simsimd_distance_t* result) {
    *result = cblas_ddot((int)n, a, 1, b, 1);
}

void dot_f32c_blas(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t n, simsimd_distance_t* result) {
    simsimd_f32_t f32_result[2] = {0, 0};
    cblas_cdotu_sub((int)n / 2, a, 1, b, 1, f32_result);
    result[0] = f32_result[0];
    result[1] = f32_result[1];
}

void dot_f64c_blas(simsimd_f64_t const* a, simsimd_f64_t const* b, simsimd_size_t n, simsimd_distance_t* result) {
    cblas_zdotu_sub((int)n / 2, a, 1, b, 1, result);
}

void vdot_f32c_blas(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t n, simsimd_distance_t* result) {
    simsimd_f32_t f32_result[2] = {0, 0};
    cblas_cdotc_sub((int)n / 2, a, 1, b, 1, f32_result);
    result[0] = f32_result[0];
    result[1] = f32_result[1];
}

void vdot_f64c_blas(simsimd_f64_t const* a, simsimd_f64_t const* b, simsimd_size_t n, simsimd_distance_t* result) {
    cblas_zdotc_sub((int)n / 2, a, 1, b, 1, result);
}

#endif

int main(int argc, char** argv) {
    simsimd_capability_t runtime_caps = simsimd_capabilities();

    // Log supported functionality
    char const* flags[2] = {"false", "true"};
    std::printf("Benchmarking Similarity Measures\n");
    std::printf("- Compiler used native F16: %s\n", flags[SIMSIMD_NATIVE_F16]);
    std::printf("- Compiler used native BF16: %s\n", flags[SIMSIMD_NATIVE_BF16]);
    std::printf("- Benchmark against CBLAS: %s\n", flags[SIMSIMD_BUILD_BENCHMARKS_WITH_CBLAS]);
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
    std::printf("\n");

    // Run the benchmarks
    bm::Initialize(&argc, argv);
    if (bm::ReportUnrecognizedArguments(argc, argv))
        return 1;

    constexpr simsimd_datatype_t b8_k = simsimd_datatype_b8_k;
    constexpr simsimd_datatype_t i8_k = simsimd_datatype_i8_k;
    constexpr simsimd_datatype_t i16_k = simsimd_datatype_i16_k;
    constexpr simsimd_datatype_t i32_k = simsimd_datatype_i32_k;
    constexpr simsimd_datatype_t i64_k = simsimd_datatype_i64_k;
    constexpr simsimd_datatype_t u8_k = simsimd_datatype_u8_k;
    constexpr simsimd_datatype_t u16_k = simsimd_datatype_u16_k;
    constexpr simsimd_datatype_t u32_k = simsimd_datatype_u32_k;
    constexpr simsimd_datatype_t u64_k = simsimd_datatype_u64_k;
    constexpr simsimd_datatype_t f64_k = simsimd_datatype_f64_k;
    constexpr simsimd_datatype_t f32_k = simsimd_datatype_f32_k;
    constexpr simsimd_datatype_t f16_k = simsimd_datatype_f16_k;
    constexpr simsimd_datatype_t bf16_k = simsimd_datatype_bf16_k;
    constexpr simsimd_datatype_t f64c_k = simsimd_datatype_f64c_k;
    constexpr simsimd_datatype_t f32c_k = simsimd_datatype_f32c_k;
    constexpr simsimd_datatype_t f16c_k = simsimd_datatype_f16c_k;
    constexpr simsimd_datatype_t bf16c_k = simsimd_datatype_bf16c_k;

#if SIMSIMD_BUILD_BENCHMARKS_WITH_CBLAS

    dense_<f32_k>("dot_f32_blas", dot_f32_blas, simsimd_dot_f32_accurate);
    dense_<f64_k>("dot_f64_blas", dot_f64_blas, simsimd_dot_f64_serial);
    dense_<f32c_k>("dot_f32c_blas", dot_f32c_blas, simsimd_dot_f32c_accurate);
    dense_<f64c_k>("dot_f64c_blas", dot_f64c_blas, simsimd_dot_f64c_serial);
    dense_<f32c_k>("vdot_f32c_blas", vdot_f32c_blas, simsimd_vdot_f32c_accurate);
    dense_<f64c_k>("vdot_f64c_blas", vdot_f64c_blas, simsimd_vdot_f64c_serial);

#endif

#if SIMSIMD_TARGET_NEON
    dense_<f16_k>("dot_f16_neon", simsimd_dot_f16_neon, simsimd_dot_f16_accurate);
    dense_<f16_k>("cos_f16_neon", simsimd_cos_f16_neon, simsimd_cos_f16_accurate);
    dense_<f16_k>("l2sq_f16_neon", simsimd_l2sq_f16_neon, simsimd_l2sq_f16_accurate);
    dense_<f16_k>("kl_f16_neon", simsimd_kl_f16_neon, simsimd_kl_f16_accurate);
    dense_<f16_k>("js_f16_neon", simsimd_js_f16_neon, simsimd_js_f16_accurate);

    dense_<bf16_k>("dot_bf16_neon", simsimd_dot_bf16_neon, simsimd_dot_bf16_accurate);
    dense_<bf16_k>("cos_bf16_neon", simsimd_cos_bf16_neon, simsimd_cos_bf16_accurate);
    dense_<bf16_k>("l2sq_bf16_neon", simsimd_l2sq_bf16_neon, simsimd_l2sq_bf16_accurate);

    dense_<f32_k>("dot_f32_neon", simsimd_dot_f32_neon, simsimd_dot_f32_accurate);
    dense_<f32_k>("cos_f32_neon", simsimd_cos_f32_neon, simsimd_cos_f32_accurate);
    dense_<f32_k>("l2sq_f32_neon", simsimd_l2sq_f32_neon, simsimd_l2sq_f32_accurate);
    dense_<f32_k>("kl_f32_neon", simsimd_kl_f32_neon, simsimd_kl_f32_accurate);
    dense_<f32_k>("js_f32_neon", simsimd_js_f32_neon, simsimd_js_f32_accurate);

    dense_<f64_k>("cos_f64_neon", simsimd_cos_f64_neon, simsimd_cos_f64_serial);
    dense_<f64_k>("l2sq_f64_neon", simsimd_l2sq_f64_neon, simsimd_l2sq_f64_serial);

    dense_<i8_k>("cos_i8_neon", simsimd_cos_i8_neon, simsimd_cos_i8_accurate);
    dense_<i8_k>("dot_i8_neon", simsimd_dot_i8_neon, simsimd_dot_i8_serial);
    dense_<i8_k>("l2sq_i8_neon", simsimd_l2sq_i8_neon, simsimd_l2sq_i8_accurate);

    dense_<b8_k>("hamming_b8_neon", simsimd_hamming_b8_neon, simsimd_hamming_b8_serial);
    dense_<b8_k>("jaccard_b8_neon", simsimd_jaccard_b8_neon, simsimd_jaccard_b8_serial);

    dense_<bf16c_k>("dot_bf16c_neon", simsimd_dot_bf16c_neon, simsimd_dot_bf16c_accurate);
    dense_<bf16c_k>("vdot_bf16c_neon", simsimd_vdot_bf16c_neon, simsimd_vdot_bf16c_accurate);
    dense_<f16c_k>("dot_f16c_neon", simsimd_dot_f16c_neon, simsimd_dot_f16c_accurate);
    dense_<f16c_k>("vdot_f16c_neon", simsimd_vdot_f16c_neon, simsimd_vdot_f16c_accurate);
    dense_<f32c_k>("dot_f32c_neon", simsimd_dot_f32c_neon, simsimd_dot_f32c_accurate);
    dense_<f32c_k>("vdot_f32c_neon", simsimd_vdot_f32c_neon, simsimd_vdot_f32c_accurate);

    curved_<f32_k>("bilinear_f32_neon", simsimd_bilinear_f32_neon, simsimd_bilinear_f32_accurate);
    curved_<f32_k>("mahalanobis_f32_neon", simsimd_mahalanobis_f32_neon, simsimd_mahalanobis_f32_accurate);
    curved_<f16_k>("bilinear_f16_neon", simsimd_bilinear_f16_neon, simsimd_bilinear_f16_accurate);
    curved_<f16_k>("mahalanobis_f16_neon", simsimd_mahalanobis_f16_neon, simsimd_mahalanobis_f16_accurate);
    curved_<bf16_k>("bilinear_bf16_neon", simsimd_bilinear_bf16_neon, simsimd_bilinear_bf16_accurate);
    curved_<bf16_k>("mahalanobis_bf16_neon", simsimd_mahalanobis_bf16_neon, simsimd_mahalanobis_bf16_accurate);

    sparse_<u16_k>("intersect_u16_neon", simsimd_intersect_u16_neon, simsimd_intersect_u16_accurate);
    sparse_<u32_k>("intersect_u32_neon", simsimd_intersect_u32_neon, simsimd_intersect_u32_accurate);
#endif

#if SIMSIMD_TARGET_SVE
    dense_<f16_k>("dot_f16_sve", simsimd_dot_f16_sve, simsimd_dot_f16_accurate);
    dense_<f16_k>("cos_f16_sve", simsimd_cos_f16_sve, simsimd_cos_f16_accurate);
    dense_<f16_k>("l2sq_f16_sve", simsimd_l2sq_f16_sve, simsimd_l2sq_f16_accurate);

    dense_<bf16_k>("cos_bf16_sve", simsimd_cos_bf16_sve, simsimd_cos_bf16_accurate);
    dense_<bf16_k>("l2sq_bf16_sve", simsimd_l2sq_bf16_sve, simsimd_l2sq_bf16_accurate);

    dense_<f32_k>("dot_f32_sve", simsimd_dot_f32_sve, simsimd_dot_f32_accurate);
    dense_<f32_k>("cos_f32_sve", simsimd_cos_f32_sve, simsimd_cos_f32_accurate);
    dense_<f32_k>("l2sq_f32_sve", simsimd_l2sq_f32_sve, simsimd_l2sq_f32_accurate);

    dense_<f64_k>("dot_f64_sve", simsimd_dot_f64_sve, simsimd_dot_f64_serial);
    dense_<f64_k>("cos_f64_sve", simsimd_cos_f64_sve, simsimd_cos_f64_serial);
    dense_<f64_k>("l2sq_f64_sve", simsimd_l2sq_f64_sve, simsimd_l2sq_f64_serial);

    dense_<b8_k>("hamming_b8_sve", simsimd_hamming_b8_sve, simsimd_hamming_b8_serial);
    dense_<b8_k>("jaccard_b8_sve", simsimd_jaccard_b8_sve, simsimd_jaccard_b8_serial);

    dense_<f16c_k>("dot_f16c_sve", simsimd_dot_f16c_sve, simsimd_dot_f16c_accurate);
    dense_<f16c_k>("vdot_f16c_sve", simsimd_vdot_f16c_sve, simsimd_vdot_f16c_accurate);
    dense_<f32c_k>("dot_f32c_sve", simsimd_dot_f32c_sve, simsimd_dot_f32c_accurate);
    dense_<f32c_k>("vdot_f32c_sve", simsimd_vdot_f32c_sve, simsimd_vdot_f32c_accurate);
    dense_<f64c_k>("dot_f64c_sve", simsimd_dot_f64c_sve, simsimd_dot_f64c_serial);
    dense_<f64c_k>("vdot_f64c_sve", simsimd_vdot_f64c_sve, simsimd_vdot_f64c_serial);
#endif

#if SIMSIMD_TARGET_SVE2
    sparse_<u16_k>("intersect_u16_sve2", simsimd_intersect_u16_sve2, simsimd_intersect_u16_accurate);
    sparse_<u32_k>("intersect_u32_sve2", simsimd_intersect_u32_sve2, simsimd_intersect_u32_accurate);
#endif

#if SIMSIMD_TARGET_HASWELL
    dense_<f16_k>("dot_f16_haswell", simsimd_dot_f16_haswell, simsimd_dot_f16_accurate);
    dense_<f16_k>("cos_f16_haswell", simsimd_cos_f16_haswell, simsimd_cos_f16_accurate);
    dense_<f16_k>("l2sq_f16_haswell", simsimd_l2sq_f16_haswell, simsimd_l2sq_f16_accurate);
    dense_<f16_k>("kl_f16_haswell", simsimd_kl_f16_haswell, simsimd_kl_f16_accurate);
    dense_<f16_k>("js_f16_haswell", simsimd_js_f16_haswell, simsimd_js_f16_accurate);

    dense_<bf16_k>("dot_bf16_haswell", simsimd_dot_bf16_haswell, simsimd_dot_bf16_accurate);
    dense_<bf16_k>("cos_bf16_haswell", simsimd_cos_bf16_haswell, simsimd_cos_bf16_accurate);
    dense_<bf16_k>("l2sq_bf16_haswell", simsimd_l2sq_bf16_haswell, simsimd_l2sq_bf16_accurate);

    dense_<i8_k>("cos_i8_haswell", simsimd_cos_i8_haswell, simsimd_cos_i8_accurate);
    dense_<i8_k>("dot_i8_haswell", simsimd_dot_i8_haswell, simsimd_dot_i8_serial);
    dense_<i8_k>("l2sq_i8_haswell", simsimd_l2sq_i8_haswell, simsimd_l2sq_i8_accurate);

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

#endif

#if SIMSIMD_TARGET_GENOA
    dense_<bf16_k>("dot_bf16_genoa", simsimd_dot_bf16_genoa, simsimd_dot_bf16_accurate);
    dense_<bf16_k>("cos_bf16_genoa", simsimd_cos_bf16_genoa, simsimd_cos_bf16_accurate);
    dense_<bf16_k>("l2sq_bf16_genoa", simsimd_l2sq_bf16_genoa, simsimd_l2sq_bf16_accurate);

    dense_<bf16_k>("dot_bf16c_genoa", simsimd_dot_bf16c_genoa, simsimd_dot_bf16c_accurate);
    dense_<bf16_k>("vdot_bf16c_genoa", simsimd_vdot_bf16c_genoa, simsimd_vdot_bf16c_accurate);

    curved_<bf16_k>("bilinear_bf16_genoa", simsimd_bilinear_bf16_genoa, simsimd_bilinear_bf16_accurate);
    curved_<bf16_k>("mahalanobis_bf16_genoa", simsimd_mahalanobis_bf16_genoa, simsimd_mahalanobis_bf16_accurate);
#endif

#if SIMSIMD_TARGET_SAPPHIRE
    dense_<f16_k>("dot_f16_sapphire", simsimd_dot_f16_sapphire, simsimd_dot_f16_accurate);
    dense_<f16_k>("cos_f16_sapphire", simsimd_cos_f16_sapphire, simsimd_cos_f16_accurate);
    dense_<f16_k>("l2sq_f16_sapphire", simsimd_l2sq_f16_sapphire, simsimd_l2sq_f16_accurate);
    dense_<f16_k>("kl_f16_sapphire", simsimd_kl_f16_sapphire, simsimd_kl_f16_accurate);
    dense_<f16_k>("js_f16_sapphire", simsimd_js_f16_sapphire, simsimd_js_f16_accurate);

    dense_<f16c_k>("dot_f16c_sapphire", simsimd_dot_f16c_sapphire, simsimd_dot_f16c_accurate);
    dense_<f16c_k>("vdot_f16c_sapphire", simsimd_vdot_f16c_sapphire, simsimd_vdot_f16c_accurate);
#endif

#if SIMSIMD_TARGET_ICE
    dense_<i8_k>("cos_i8_ice", simsimd_cos_i8_ice, simsimd_cos_i8_accurate);
    dense_<i8_k>("dot_i8_ice", simsimd_dot_i8_ice, simsimd_dot_i8_serial);
    dense_<i8_k>("l2sq_i8_ice", simsimd_l2sq_i8_ice, simsimd_l2sq_i8_accurate);

    dense_<f64_k>("dot_f64_skylake", simsimd_dot_f64_skylake, simsimd_dot_f64_serial);
    dense_<f64_k>("cos_f64_skylake", simsimd_cos_f64_skylake, simsimd_cos_f64_serial);
    dense_<f64_k>("l2sq_f64_skylake", simsimd_l2sq_f64_skylake, simsimd_l2sq_f64_serial);

    dense_<b8_k>("hamming_b8_ice", simsimd_hamming_b8_ice, simsimd_hamming_b8_serial);
    dense_<b8_k>("jaccard_b8_ice", simsimd_jaccard_b8_ice, simsimd_jaccard_b8_serial);

    sparse_<u16_k>("intersect_u16_ice", simsimd_intersect_u16_ice, simsimd_intersect_u16_accurate);
    sparse_<u32_k>("intersect_u32_ice", simsimd_intersect_u32_ice, simsimd_intersect_u32_accurate);
#endif

#if SIMSIMD_TARGET_SKYLAKE
    dense_<f32_k>("dot_f32_skylake", simsimd_dot_f32_skylake, simsimd_dot_f32_accurate);
    dense_<f32_k>("cos_f32_skylake", simsimd_cos_f32_skylake, simsimd_cos_f32_accurate);
    dense_<f32_k>("l2sq_f32_skylake", simsimd_l2sq_f32_skylake, simsimd_l2sq_f32_accurate);
    dense_<f32_k>("kl_f32_skylake", simsimd_kl_f32_skylake, simsimd_kl_f32_accurate);
    dense_<f32_k>("js_f32_skylake", simsimd_js_f32_skylake, simsimd_js_f32_accurate);

    dense_<f32c_k>("dot_f32c_skylake", simsimd_dot_f32c_skylake, simsimd_dot_f32c_accurate);
    dense_<f32c_k>("vdot_f32c_skylake", simsimd_vdot_f32c_skylake, simsimd_vdot_f32c_accurate);
    dense_<f64c_k>("dot_f64c_skylake", simsimd_dot_f64c_skylake, simsimd_dot_f64c_serial);
    dense_<f64c_k>("vdot_f64c_skylake", simsimd_vdot_f64c_skylake, simsimd_vdot_f64c_serial);
#endif

    sparse_<u16_k>("intersect_u16_serial", simsimd_intersect_u16_serial, simsimd_intersect_u16_accurate);
    sparse_<u16_k>("intersect_u16_accurate", simsimd_intersect_u16_accurate, simsimd_intersect_u16_accurate);
    sparse_<u32_k>("intersect_u32_serial", simsimd_intersect_u32_serial, simsimd_intersect_u32_accurate);
    sparse_<u32_k>("intersect_u32_accurate", simsimd_intersect_u32_accurate, simsimd_intersect_u32_accurate);

    curved_<f64_k>("bilinear_f64_serial", simsimd_bilinear_f64_serial, simsimd_bilinear_f64_serial);
    curved_<f64_k>("mahalanobis_f64_serial", simsimd_mahalanobis_f64_serial, simsimd_mahalanobis_f64_serial);
    curved_<f32_k>("bilinear_f32_serial", simsimd_bilinear_f32_serial, simsimd_bilinear_f32_accurate);
    curved_<f32_k>("mahalanobis_f32_serial", simsimd_mahalanobis_f32_serial, simsimd_mahalanobis_f32_accurate);
    curved_<f16_k>("bilinear_f16_serial", simsimd_bilinear_f16_serial, simsimd_bilinear_f16_accurate);
    curved_<f16_k>("mahalanobis_f16_serial", simsimd_mahalanobis_f16_serial, simsimd_mahalanobis_f16_accurate);
    curved_<bf16_k>("bilinear_bf16_serial", simsimd_bilinear_bf16_serial, simsimd_bilinear_bf16_accurate);
    curved_<bf16_k>("mahalanobis_bf16_serial", simsimd_mahalanobis_bf16_serial, simsimd_mahalanobis_bf16_accurate);

    dense_<bf16_k>("dot_bf16_serial", simsimd_dot_bf16_serial, simsimd_dot_bf16_accurate);
    dense_<bf16_k>("cos_bf16_serial", simsimd_cos_bf16_serial, simsimd_cos_bf16_accurate);
    dense_<bf16_k>("l2sq_bf16_serial", simsimd_l2sq_bf16_serial, simsimd_l2sq_bf16_accurate);
    dense_<bf16_k>("kl_bf16_serial", simsimd_kl_bf16_serial, simsimd_kl_bf16_accurate);
    dense_<bf16_k>("js_bf16_serial", simsimd_js_bf16_serial, simsimd_js_bf16_accurate);

    dense_<f16_k>("dot_f16_serial", simsimd_dot_f16_serial, simsimd_dot_f16_accurate);
    dense_<f16_k>("cos_f16_serial", simsimd_cos_f16_serial, simsimd_cos_f16_accurate);
    dense_<f16_k>("l2sq_f16_serial", simsimd_l2sq_f16_serial, simsimd_l2sq_f16_accurate);
    dense_<f16_k>("kl_f16_serial", simsimd_kl_f16_serial, simsimd_kl_f16_accurate);
    dense_<f16_k>("js_f16_serial", simsimd_js_f16_serial, simsimd_js_f16_accurate);

    dense_<f32_k>("dot_f32_serial", simsimd_dot_f32_serial, simsimd_dot_f32_accurate);
    dense_<f32_k>("cos_f32_serial", simsimd_cos_f32_serial, simsimd_cos_f32_accurate);
    dense_<f32_k>("l2sq_f32_serial", simsimd_l2sq_f32_serial, simsimd_l2sq_f32_accurate);
    dense_<f32_k>("kl_f32_serial", simsimd_kl_f32_serial, simsimd_kl_f32_accurate);
    dense_<f32_k>("js_f32_serial", simsimd_js_f32_serial, simsimd_js_f32_accurate);

    dense_<f64_k>("dot_f64_serial", simsimd_dot_f64_serial, simsimd_dot_f64_serial);
    dense_<f64_k>("cos_f64_serial", simsimd_cos_f64_serial, simsimd_cos_f64_serial);
    dense_<f64_k>("l2sq_f64_serial", simsimd_l2sq_f64_serial, simsimd_l2sq_f64_serial);

    dense_<i8_k>("cos_i8_serial", simsimd_cos_i8_serial, simsimd_cos_i8_accurate);
    dense_<i8_k>("dot_i8_serial", simsimd_dot_i8_serial, simsimd_dot_i8_serial);
    dense_<i8_k>("l2sq_i8_serial", simsimd_l2sq_i8_serial, simsimd_l2sq_i8_accurate);

    dense_<f64c_k>("dot_f64c_serial", simsimd_dot_f64c_serial, simsimd_dot_f64c_serial);
    dense_<f32c_k>("dot_f32c_serial", simsimd_dot_f32c_serial, simsimd_dot_f32c_accurate);
    dense_<f16c_k>("dot_f16c_serial", simsimd_dot_f16c_serial, simsimd_dot_f16c_accurate);
    dense_<bf16c_k>("dot_bf16c_serial", simsimd_dot_bf16c_serial, simsimd_dot_bf16c_accurate);
    dense_<f64c_k>("vdot_f64c_serial", simsimd_vdot_f64c_serial, simsimd_vdot_f64c_serial);
    dense_<f32c_k>("vdot_f32c_serial", simsimd_vdot_f32c_serial, simsimd_vdot_f32c_accurate);
    dense_<f16c_k>("vdot_f16c_serial", simsimd_vdot_f16c_serial, simsimd_vdot_f16c_accurate);
    dense_<bf16c_k>("vdot_bf16c_serial", simsimd_vdot_bf16c_serial, simsimd_vdot_bf16c_accurate);

    dense_<f16c_k>("vdot_f16c_serial", simsimd_vdot_f16c_serial, simsimd_vdot_f16c_accurate);
    dense_<bf16c_k>("vdot_bf16c_serial", simsimd_vdot_bf16c_serial, simsimd_vdot_bf16c_accurate);

    dense_<b8_k>("hamming_b8_serial", simsimd_hamming_b8_serial, simsimd_hamming_b8_serial);
    dense_<b8_k>("jaccard_b8_serial", simsimd_jaccard_b8_serial, simsimd_jaccard_b8_serial);

    bm::RunSpecifiedBenchmarks();
    bm::Shutdown();
    return 0;
}

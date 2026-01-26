/**
 *  @brief NumKong C++ Benchmark Suite using Google Benchmark.
 *  @file scripts/bench.cpp
 *
 *  Comprehensive benchmarks for NumKong SIMD-optimized functions measuring
 *  throughput performance. Run with:
 *
 *  ```bash
 *  cmake -B build_release -D NK_BUILD_BENCHMARKS=1
 *  cmake --build build_release
 *  build_release/nk_bench
 *  ```
 *
 *  Environment Variables:
 *    NK_FILTER=<pattern>           - Filter benchmarks by name regex (default: run all)
 *    NK_SEED=N                     - RNG seed (default: 42)
 *
 *    NK_DENSE_DIMENSIONS=N         - Vector dimension for dot/spatial benchmarks (default: 1536)
 *    NK_MESH_POINTS=N              - Point count for mesh benchmarks (default: 1000)
 *    NK_MATRIX_HEIGHT=N            - GEMM M dimension (default: 1024), like dataset size for kNN
 *    NK_MATRIX_WIDTH=N             - GEMM N dimension (default: 128), like queries count for kNN
 *    NK_MATRIX_DEPTH=N             - GEMM K dimension (default: 1536), like vector dimensions in KNN
 *
 *    NK_CURVED_DIMENSIONS=N        - Vector dimension for curved benchmarks (default: 64)
 *    NK_SPARSE_FIRST_LENGTH=N      - First set size for sparse benchmarks (default: 1024)
 *    NK_SPARSE_SECOND_LENGTH=N     - Second set size for sparse benchmarks (default: 8192)
 *    NK_SPARSE_INTERSECTION=F      - Intersection share 0.0-1.0 (default: 0.5)
 */

#include <array>         // `std::array`
#include <cmath>         // `std::sqrt`
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
// - cblas_gemm_bf16bf16f32: BF16 inputs → F32 output
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
#include <numkong/numkong.hpp>

namespace bm = benchmark;
namespace nk = ashvardanian::numkong;

constexpr std::size_t default_seconds = 10;
constexpr std::size_t default_threads = 1;

/// Vector dimension for dot products and spatial metrics
/// Can be overridden at runtime via `NK_DENSE_DIMENSIONS` environment variable
std::size_t dense_dimensions = 1536;
/// Has quadratic impact on the number of operations
/// Can be overridden at runtime via `NK_CURVED_DIMENSIONS` environment variable
std::size_t curved_dimensions = 64;
/// Number of 3D points for mesh metrics (RMSD, Kabsch)
/// Can be overridden at runtime via `NK_MESH_POINTS` environment variable
std::size_t mesh_points = 1000;
/// Matrix multiplication benchmark globals
/// Can be overridden at runtime via `NK_MATRIX_HEIGHT/WIDTH/DEPTH` environment variables
std::size_t matrix_height = 1024, matrix_width = 128, matrix_depth = 1536;
/// Random seed for reproducible benchmarks
/// Can be overridden at runtime via `NK_SEED` environment variable
std::uint32_t random_seed = 42;
/// Sparse set intersection benchmark globals
/// Can be overridden at runtime via `NK_SPARSE_*` environment variables
std::size_t sparse_first_length = 1024;
std::size_t sparse_second_length = 8192;
double sparse_intersection_share = 0.5;

inline std::mt19937 make_random_engine() { return std::mt19937(random_seed); }

/** @brief Factory function to allocate vectors, potentially raising bad-allocs. */
template <typename type_>
[[nodiscard]] nk::vector<type_> make_vector(std::size_t count) {
    nk::vector<type_> result;
    if (!result.resize(count)) throw std::bad_alloc();
    return result;
}

/**
 *  @brief Factory function to allocate matrix buffers with correct size for sub-byte types.
 *
 *  For sub-byte types (u4, i4, u1), calculates the actual number of raw_t elements needed
 *  to store a matrix of logical dimensions rows × cols.
 *
 *  @tparam dtype_ The NumKong dtype (e.g., nk_u4_k, nk_i4_k)
 *  @param rows Number of logical rows
 *  @param cols Number of logical columns per row
 *  @return Vector with correct byte capacity for the matrix
 */
template <nk_dtype_t dtype_>
[[nodiscard]] nk::vector<typename nk::type_for<dtype_>::type> make_vector_for_matrix(std::size_t rows,
                                                                                     std::size_t cols) {
    using type_ = typename nk::type_for<dtype_>::type;
    using raw_t = typename type_::raw_t;

    nk_size_t bits_per_element = nk_dtype_bits(dtype_);
    nk_size_t bits_per_row = cols * bits_per_element;
    nk_size_t bytes_per_row = nk::divide_round_up(bits_per_row, NK_BITS_PER_BYTE);
    nk_size_t total_values = rows * bytes_per_row / sizeof(raw_t);

    nk::vector<type_> result;
    if (!result.resize_values(total_values)) throw std::bad_alloc();
    return result;
}

/**
 *  @brief Measures the performance of a @b dense kernel function using Google Benchmark.
 *  @param state The benchmark state object provided by Google Benchmark.
 *  @param kernel The kernel function to benchmark.
 *  @param dimensions The number of dimensions in the vectors.
 */
template <nk_dtype_t input_dtype_, nk_dtype_t output_dtype_, typename kernel_type_ = void>
void measure_dense(bm::State &state, kernel_type_ kernel, std::size_t dimensions) {

    using input_t = typename nk::type_for<input_dtype_>::type;
    using output_t = typename nk::type_for<output_dtype_>::type;
    using input_vector_t = nk::vector<input_t>;

    // Preallocate inputs (1024 vector pairs to avoid cache effects)
    constexpr std::size_t vectors_count = 1024;
    std::vector<input_vector_t> first_vectors(vectors_count), second_vectors(vectors_count);
    auto generator = make_random_engine();
    for (std::size_t index = 0; index != vectors_count; ++index) {
        first_vectors[index] = make_vector<input_t>(dimensions);
        second_vectors[index] = make_vector<input_t>(dimensions);
        nk::fill_uniform(generator, first_vectors[index].values_data(), first_vectors[index].size_values());
        nk::fill_uniform(generator, second_vectors[index].values_data(), second_vectors[index].size_values());
    }

    // Benchmark loop
    std::size_t iterations = 0;
    for (auto _ : state) {
        output_t output;
        std::size_t const index = iterations & (vectors_count - 1);
        kernel(first_vectors[index].raw_values_data(), second_vectors[index].raw_values_data(), dimensions,
               &output.raw_);
        bm::DoNotOptimize(output);
        iterations++;
    }

    state.counters["bytes"] = bm::Counter(iterations * first_vectors[0].size_bytes() * 2, bm::Counter::kIsRate);
    state.counters["calls"] = bm::Counter(iterations, bm::Counter::kIsRate);
}

/**
 *  @brief Measures the performance of a @b curved (bilinear/Mahalanobis) kernel function using Google Benchmark.
 *  @param state The benchmark state object provided by Google Benchmark.
 *  @param kernel The kernel function to benchmark.
 *  @param dimensions The number of dimensions in the vectors.
 */
template <nk_dtype_t input_dtype_, nk_dtype_t output_dtype_, typename kernel_type_ = void>
void measure_curved(bm::State &state, kernel_type_ kernel, std::size_t dimensions) {

    using input_t = typename nk::type_for<input_dtype_>::type;
    using output_t = typename nk::type_for<output_dtype_>::type;
    using input_vector_t = nk::vector<input_t>;

    // Preallocate inputs: pairs of vectors + metric tensors (dimensions × dimensions)
    constexpr std::size_t vectors_count = 1024;
    std::vector<input_vector_t> first_vectors(vectors_count), second_vectors(vectors_count), tensors(vectors_count);
    auto generator = make_random_engine();
    for (std::size_t index = 0; index != vectors_count; ++index) {
        first_vectors[index] = make_vector<input_t>(dimensions);
        second_vectors[index] = make_vector<input_t>(dimensions);
        tensors[index] = make_vector<input_t>(dimensions * dimensions);
        nk::fill_uniform(generator, first_vectors[index].values_data(), first_vectors[index].size_values());
        nk::fill_uniform(generator, second_vectors[index].values_data(), second_vectors[index].size_values());
        nk::fill_uniform(generator, tensors[index].values_data(), tensors[index].size_values());
    }

    // Benchmark loop
    std::size_t iterations = 0;
    for (auto _ : state) {
        output_t output[2] = {};
        std::size_t const index = iterations & (vectors_count - 1);
        kernel(first_vectors[index].raw_values_data(), second_vectors[index].raw_values_data(),
               tensors[index].raw_values_data(), dimensions, &output[0].raw_);
        bm::DoNotOptimize(output);
        iterations++;
    }

    state.counters["bytes"] = bm::Counter(iterations * first_vectors[0].size_bytes() * 2, bm::Counter::kIsRate);
    state.counters["calls"] = bm::Counter(iterations, bm::Counter::kIsRate);
}

/**
 *  @brief Measures the performance of a @b sparse (set intersection) kernel function using Google Benchmark.
 *  @param state The benchmark state object provided by Google Benchmark.
 *  @param kernel The kernel function to benchmark.
 *  @param first_size The number of elements in the first (smaller) set.
 *  @param second_size The number of elements in the second (larger) set.
 *  @param intersection_size The expected number of common elements between the sets.
 */
template <nk_dtype_t input_dtype_, nk_dtype_t output_dtype_, typename kernel_type_ = void>
void measure_sparse(bm::State &state, kernel_type_ kernel, std::size_t first_size, std::size_t second_size,
                    std::size_t intersection_size) {

    using input_t = typename nk::type_for<input_dtype_>::type;
    using input_vector_t = nk::vector<input_t>;
    using scalar_t = typename input_t::raw_t;

    // Preallocate sorted unique set vectors
    constexpr std::size_t vectors_count = 1024;
    std::vector<input_vector_t> first_vectors(vectors_count), second_vectors(vectors_count);
    auto generator = make_random_engine();
    std::uniform_int_distribution<scalar_t> distribution(0, std::numeric_limits<scalar_t>::max());

    // Generating sorted unique sets with controlled intersection is complex:
    // we generate intersection elements, then unique elements for each set.
    std::unordered_set<scalar_t> intersection_elements, unique_first, unique_second;
    intersection_elements.reserve(intersection_size);
    unique_first.reserve(first_size - intersection_size);
    unique_second.reserve(second_size - intersection_size);

    for (std::size_t index = 0; index != vectors_count; ++index) {
        first_vectors[index] = make_vector<input_t>(first_size);
        second_vectors[index] = make_vector<input_t>(second_size);

        // Step 1: Generate intersection elements
        intersection_elements.clear();
        while (intersection_elements.size() < intersection_size) intersection_elements.insert(distribution(generator));

        unique_first.clear();
        while (unique_first.size() < first_size - intersection_size) {
            scalar_t element = distribution(generator);
            if (intersection_elements.find(element) == intersection_elements.end()) unique_first.insert(element);
        }

        unique_second.clear();
        while (unique_second.size() < second_size - intersection_size) {
            scalar_t element = distribution(generator);
            if (intersection_elements.find(element) == intersection_elements.end() &&
                unique_first.find(element) == unique_first.end())
                unique_second.insert(element);
        }

        // Step 2: Merge and sort
        input_t *first_data = first_vectors[index].values_data();
        input_t *second_data = second_vectors[index].values_data();
        std::size_t offset = 0;
        for (scalar_t element : intersection_elements) first_data[offset++].raw_ = element;
        for (scalar_t element : unique_first) first_data[offset++].raw_ = element;
        offset = 0;
        for (scalar_t element : intersection_elements) second_data[offset++].raw_ = element;
        for (scalar_t element : unique_second) second_data[offset++].raw_ = element;
        std::sort(first_data, first_data + first_size, [](input_t a, input_t b) { return a.raw_ < b.raw_; });
        std::sort(second_data, second_data + second_size, [](input_t a, input_t b) { return a.raw_ < b.raw_; });
    }

    // Benchmark loop
    std::size_t iterations = 0;
    for (auto _ : state) {
        nk_size_t count;
        std::size_t const index = iterations & (vectors_count - 1);
        kernel(first_vectors[index].raw_values_data(), second_vectors[index].raw_values_data(), first_size, second_size,
               nullptr, &count);
        bm::DoNotOptimize(count);
        iterations++;
    }

    state.counters["bytes"] = bm::Counter(iterations * (first_vectors[0].size_bytes() + second_vectors[0].size_bytes()),
                                          bm::Counter::kIsRate);
    state.counters["calls"] = bm::Counter(iterations, bm::Counter::kIsRate);
}

/**
 *  @brief Measures the performance of a @b mesh kernel function (RMSD/Kabsch/Umeyama) using Google Benchmark.
 *  @param state The benchmark state object provided by Google Benchmark.
 *  @param kernel The kernel function to benchmark.
 *  @param points_count The number of 3D points in each point cloud.
 */
template <nk_dtype_t input_dtype_, nk_dtype_t output_dtype_, typename kernel_type_ = void>
void measure_mesh(bm::State &state, kernel_type_ kernel, std::size_t points_count) {

    using input_t = typename nk::type_for<input_dtype_>::type;
    using raw_input_t = typename input_t::raw_t;
    using output_t = typename nk::type_for<output_dtype_>::type;
    using input_vector_t = nk::vector<input_t>;

    // Preallocate point clouds: each contains points_count 3D points stored as [x0,y0,z0,x1,y1,z1,...]
    constexpr std::size_t clouds_count = 1024;
    std::vector<input_vector_t> first_clouds(clouds_count), second_clouds(clouds_count);
    auto generator = make_random_engine();
    for (std::size_t index = 0; index != clouds_count; ++index) {
        first_clouds[index] = make_vector<input_t>(points_count * 3);
        second_clouds[index] = make_vector<input_t>(points_count * 3);
        nk::fill_uniform(generator, first_clouds[index].values_data(), first_clouds[index].size_values());
        nk::fill_uniform(generator, second_clouds[index].values_data(), second_clouds[index].size_values());
    }

    // Benchmark loop
    std::size_t iterations = 0;
    for (auto _ : state) {
        output_t result, scale;
        raw_input_t first_centroid[3], second_centroid[3], rotation[9];
        std::size_t const index = iterations & (clouds_count - 1);
        kernel(first_clouds[index].raw_values_data(), second_clouds[index].raw_values_data(), points_count,
               first_centroid, second_centroid, rotation, &scale.raw_, &result.raw_);
        bm::DoNotOptimize(result);
        iterations++;
    }

    state.counters["bytes"] = bm::Counter(iterations * first_clouds[0].size_bytes() * 2, bm::Counter::kIsRate);
    state.counters["calls"] = bm::Counter(iterations, bm::Counter::kIsRate);
}

/**
 *  @brief Measures the performance of elementwise operations (sum, wsum, fma, scale, trig) using Google Benchmark.
 *  @param state The benchmark state object provided by Google Benchmark.
 *  @param kernel The kernel function to benchmark.
 *  @param dimensions The number of dimensions in the vectors.
 */
template <nk_dtype_t input_dtype_, nk_kernel_kind_t kernel_kind_, nk_dtype_t alpha_dtype_, typename kernel_type_ = void>
void measure_elementwise(bm::State &state, kernel_type_ kernel, std::size_t dimensions) {

    using input_t = typename nk::type_for<input_dtype_>::type;
    using alpha_t = typename nk::type_for<alpha_dtype_>::type;
    using input_vector_t = nk::vector<input_t>;

    // Scaling parameters for FMA/wsum/scale kernels
    alpha_t alpha = alpha_t(0.2f);
    alpha_t beta = alpha_t(0.3f);

    // Preallocate vectors for different kernel types:
    // - sum: input_a, input_c -> output
    // - wsum: input_a, input_c + α, β -> output
    // - fma: input_a, input_b, input_c + α, β -> output
    // - scale: input_a + α, β -> output
    // - trig (unknown): input_a -> output
    constexpr std::size_t vectors_count = 1024;
    std::vector<input_vector_t> input_a(vectors_count), input_b(vectors_count);
    std::vector<input_vector_t> input_c(vectors_count), output(vectors_count);
    auto generator = make_random_engine();
    for (std::size_t index = 0; index != vectors_count; ++index) {
        input_a[index] = make_vector<input_t>(dimensions);
        input_b[index] = make_vector<input_t>(dimensions);
        input_c[index] = make_vector<input_t>(dimensions);
        output[index] = make_vector<input_t>(dimensions);
        nk::fill_uniform(generator, input_a[index].values_data(), dimensions);
        std::fill(input_b[index].values_data(), input_b[index].values_data() + dimensions,
                  input_t(2)); // Small constant
        nk::fill_uniform(generator, input_c[index].values_data(), dimensions);
    }

    // Benchmark loop with kernel dispatch
    std::size_t iterations = 0;
    for (auto _ : state) {
        std::size_t const index = iterations & (vectors_count - 1);
        if constexpr (kernel_kind_ == nk_kernel_each_blend_k) {
            kernel(input_a[index].raw_values_data(), input_c[index].raw_values_data(), dimensions, &alpha.raw_,
                   &beta.raw_, output[index].raw_values_data());
        }
        else if constexpr (kernel_kind_ == nk_kernel_each_fma_k) {
            kernel(input_a[index].raw_values_data(), input_b[index].raw_values_data(), input_c[index].raw_values_data(),
                   dimensions, &alpha.raw_, &beta.raw_, output[index].raw_values_data());
        }
        else if constexpr (kernel_kind_ == nk_kernel_each_sum_k) {
            kernel(input_a[index].raw_values_data(), input_c[index].raw_values_data(), dimensions,
                   output[index].raw_values_data());
        }
        else if constexpr (kernel_kind_ == nk_kernel_each_scale_k) {
            kernel(input_a[index].raw_values_data(), dimensions, &alpha.raw_, &beta.raw_,
                   output[index].raw_values_data());
        }
        else {
            // Trigonometric or other unary kernels
            kernel(input_a[index].raw_values_data(), dimensions, output[index].raw_values_data());
        }
        bm::ClobberMemory();
        iterations++;
    }

    std::size_t bytes_per_call = input_a[0].size_bytes();
    if constexpr (kernel_kind_ == nk_kernel_each_blend_k) bytes_per_call *= 2;
    else if constexpr (kernel_kind_ == nk_kernel_each_fma_k) bytes_per_call *= 3;
    else if constexpr (kernel_kind_ == nk_kernel_each_sum_k) bytes_per_call *= 2;
    else if constexpr (kernel_kind_ == nk_kernel_each_scale_k) bytes_per_call *= 1;

    state.counters["bytes"] = bm::Counter(iterations * bytes_per_call, bm::Counter::kIsRate);
    state.counters["calls"] = bm::Counter(iterations, bm::Counter::kIsRate);
}

/**
 *  @brief Measures the performance of geospatial operations (Haversine/Vincenty) using Google Benchmark.
 *  @param state The benchmark state object provided by Google Benchmark.
 *  @param kernel The kernel function to benchmark.
 *  @param coordinates_count The number of coordinate pairs to process.
 */
template <nk_dtype_t input_dtype_, nk_dtype_t output_dtype_, typename kernel_type_ = void>
void measure_geospatial(bm::State &state, kernel_type_ kernel, std::size_t coordinates_count) {

    using input_t = typename nk::type_for<input_dtype_>::type;
    using output_t = typename nk::type_for<output_dtype_>::type;
    using input_vector_t = nk::vector<input_t>;
    using output_vector_t = nk::vector<output_t>;

    // Preallocate coordinate arrays: latitude1, longitude1, latitude2, longitude2
    constexpr std::size_t batches_count = 1024;
    std::vector<input_vector_t> latitudes_first(batches_count), longitudes_first(batches_count);
    std::vector<input_vector_t> latitudes_second(batches_count), longitudes_second(batches_count);
    auto generator = make_random_engine();
    for (std::size_t index = 0; index != batches_count; ++index) {
        latitudes_first[index] = make_vector<input_t>(coordinates_count);
        longitudes_first[index] = make_vector<input_t>(coordinates_count);
        latitudes_second[index] = make_vector<input_t>(coordinates_count);
        longitudes_second[index] = make_vector<input_t>(coordinates_count);
        nk::fill_coordinates(generator, latitudes_first[index].values_data(), longitudes_first[index].values_data(),
                             coordinates_count);
        nk::fill_coordinates(generator, latitudes_second[index].values_data(), longitudes_second[index].values_data(),
                             coordinates_count);
    }

    // Output distances buffer
    output_vector_t distances = make_vector<output_t>(coordinates_count);

    // Benchmark loop
    std::size_t iterations = 0;
    for (auto _ : state) {
        std::size_t const index = iterations & (batches_count - 1);
        kernel(latitudes_first[index].raw_values_data(), longitudes_first[index].raw_values_data(),
               latitudes_second[index].raw_values_data(), longitudes_second[index].raw_values_data(), coordinates_count,
               distances.raw_values_data());
        bm::ClobberMemory();
        iterations++;
    }

    std::size_t const bytes_per_call = latitudes_first[0].size_bytes() * 4; // 4 coordinate arrays
    state.counters["bytes"] = bm::Counter(iterations * bytes_per_call, bm::Counter::kIsRate);
    state.counters["calls"] = bm::Counter(iterations * coordinates_count, bm::Counter::kIsRate);
}

template <nk_dtype_t input_dtype_, nk_dtype_t output_dtype_, typename kernel_type_ = void>
void dense_(std::string name, kernel_type_ *kernel) {
    std::string bench_name = name + "<" + std::to_string(dense_dimensions) + "d>";
    bm::RegisterBenchmark(bench_name.c_str(), measure_dense<input_dtype_, output_dtype_, kernel_type_ *>, kernel,
                          dense_dimensions)
        ->MinTime(default_seconds)
        ->Threads(default_threads);
}

template <nk_dtype_t input_dtype_, nk_kernel_kind_t kernel_kind_ = nk_kernel_unknown_k,
          nk_dtype_t alpha_dtype_ = nk_dtype_unknown_k, typename kernel_type_ = void>
void elementwise_(std::string name, kernel_type_ *kernel) {
    std::string bench_name = name + "<" + std::to_string(dense_dimensions) + "d>";
    bm::RegisterBenchmark(bench_name.c_str(),
                          measure_elementwise<input_dtype_, kernel_kind_, alpha_dtype_, kernel_type_ *>, kernel,
                          dense_dimensions)
        ->MinTime(default_seconds)
        ->Threads(default_threads);
}

template <nk_dtype_t input_dtype_, nk_dtype_t output_dtype_, typename kernel_type_ = void>
void geospatial_(std::string name, kernel_type_ *kernel) {
    std::string bench_name = name + "<" + std::to_string(dense_dimensions) + "d>";
    bm::RegisterBenchmark(bench_name.c_str(), measure_geospatial<input_dtype_, output_dtype_, kernel_type_ *>, kernel,
                          dense_dimensions)
        ->MinTime(default_seconds)
        ->Threads(default_threads);
}

template <nk_dtype_t input_dtype_, nk_dtype_t output_dtype_, typename kernel_type_ = void>
void sparse_(std::string name, kernel_type_ *kernel) {
    std::size_t intersection_size = static_cast<std::size_t>(std::min(sparse_first_length, sparse_second_length) *
                                                             sparse_intersection_share);
    std::string bench_name = name + "<|A|=" + std::to_string(sparse_first_length) +
                             ",|B|=" + std::to_string(sparse_second_length) +
                             ",|A∩B|=" + std::to_string(intersection_size) + ">";
    bm::RegisterBenchmark(bench_name.c_str(), measure_sparse<input_dtype_, output_dtype_, kernel_type_ *>, kernel,
                          sparse_first_length, sparse_second_length, intersection_size)
        ->MinTime(default_seconds)
        ->Threads(default_threads);
}

template <nk_dtype_t input_dtype_, nk_dtype_t output_dtype_, typename kernel_type_ = void>
void curved_(std::string name, kernel_type_ *kernel) {
    std::string bench_name = name + "<" + std::to_string(curved_dimensions) + "d>";
    bm::RegisterBenchmark(bench_name.c_str(), measure_curved<input_dtype_, output_dtype_, kernel_type_ *>, kernel,
                          curved_dimensions)
        ->MinTime(default_seconds)
        ->Threads(default_threads);
}

template <nk_dtype_t input_dtype_, nk_dtype_t output_dtype_, typename kernel_type_ = void>
void mesh_(std::string name, kernel_type_ *kernel) {
    std::string bench_name = name + "<" + std::to_string(mesh_points) + "pts>";
    bm::RegisterBenchmark(bench_name.c_str(), measure_mesh<input_dtype_, output_dtype_, kernel_type_ *>, kernel,
                          mesh_points)
        ->MinTime(default_seconds)
        ->Threads(default_threads);
}

//  Batched dot products measurement for packed B matrix API
template <nk_dtype_t input_dtype_, nk_dtype_t output_dtype_>
void measure_dots_packed(                                                                //
    bm::State &state,                                                                    //
    typename nk::type_for<input_dtype_>::type::dots_packed_size_kernel_t packed_size_fn, //
    typename nk::type_for<input_dtype_>::type::dots_pack_kernel_t pack_fn,               //
    typename nk::type_for<input_dtype_>::type::dots_packed_kernel_t kernel,              //
    std::size_t m, std::size_t n, std::size_t k) {

    using input_t = typename nk::type_for<input_dtype_>::type;
    using output_t = typename nk::type_for<output_dtype_>::type;
    using raw_input_t = typename input_t::raw_t;
    using raw_output_t = typename output_t::raw_t;

    // Calculate correct strides for sub-byte types (u4, i4, u1, etc.)
    nk_size_t values_per_row = nk::divide_round_up(k, nk::dimensions_per_value<input_t>());
    nk_size_t a_stride_bytes = values_per_row * sizeof(typename input_t::raw_t);
    nk_size_t b_stride_bytes = values_per_row * sizeof(typename input_t::raw_t); // B is n×k, so k columns per row

    // Allocate matrices with correct sizes for sub-byte types
    auto matrix_a = make_vector_for_matrix<input_dtype_>(m, k);
    auto matrix_b = make_vector_for_matrix<input_dtype_>(n, k);
    nk_size_t packed_bytes = packed_size_fn(n, k);
    std::vector<char> matrix_b_packed(packed_bytes, 0);
    auto matrix_c = make_vector<output_t>(m * n);

    // Initialize with random values
    auto generator = make_random_engine();
    nk::fill_uniform(generator, matrix_a.values_data(), matrix_a.size_values());
    nk::fill_uniform(generator, matrix_b.values_data(), matrix_b.size_values());

    // Pack B matrix once (amortized cost for repeated inference) with correct stride
    pack_fn(matrix_b.raw_values_data(), n, k, b_stride_bytes, matrix_b_packed.data());

    std::size_t iterations = 0;
    for (auto _ : state) {
        bm::DoNotOptimize(matrix_c.raw_values_data());
        kernel(matrix_a.raw_values_data(), matrix_b_packed.data(), matrix_c.raw_values_data(), //
               m, n, k, a_stride_bytes, n * sizeof(raw_output_t));
        ++iterations;
    }

    state.counters["scalar-ops"] = bm::Counter(iterations * 2.0 * m * n * k, bm::Counter::kIsRate);
}

template <nk_dtype_t input_dtype_, nk_dtype_t output_dtype_>
void dots_(std::string name, //
           typename nk::type_for<input_dtype_>::type::dots_packed_size_kernel_t packed_size_fn,
           typename nk::type_for<input_dtype_>::type::dots_pack_kernel_t pack_fn,
           typename nk::type_for<input_dtype_>::type::dots_packed_kernel_t kernel) {
    std::string bench_name = name + "<" + std::to_string(matrix_height) + "x" + std::to_string(matrix_width) + "x" +
                             std::to_string(matrix_depth) + ">";
    bm::RegisterBenchmark(bench_name.c_str(), measure_dots_packed<input_dtype_, output_dtype_>, packed_size_fn, pack_fn,
                          kernel, matrix_height, matrix_width, matrix_depth)
        ->MinTime(default_seconds)
        ->Threads(1); // Single-threaded for packed matmul
}

template <nk_dtype_t input_dtype_, nk_dtype_t output_dtype_>
void measure_dots_symmetric(                                                   //
    bm::State &state,                                                          //
    typename nk::type_for<input_dtype_>::type::dots_symmetric_kernel_t kernel, //
    std::size_t n, std::size_t k) {

    using input_t = typename nk::type_for<input_dtype_>::type;
    using output_t = typename nk::type_for<output_dtype_>::type;
    using raw_input_t = typename input_t::raw_t;
    using raw_output_t = typename output_t::raw_t;

    // Calculate correct strides for sub-byte types (u4, i4, u1, etc.)
    nk_size_t input_values_per_row = nk::divide_round_up(k, nk::dimensions_per_value<input_t>());
    nk_size_t input_stride_bytes = input_values_per_row * sizeof(typename input_t::raw_t);
    nk_size_t output_stride_bytes = n * sizeof(raw_output_t);

    // Allocate matrix A (n vectors × k dimensions) and result matrix C (n × n)
    auto matrix_a = make_vector_for_matrix<input_dtype_>(n, k);
    auto matrix_c = make_vector<output_t>(n * n);

    // Initialize with random values
    auto generator = make_random_engine();
    nk::fill_uniform(generator, matrix_a.values_data(), matrix_a.size_values());

    std::size_t iterations = 0;
    for (auto _ : state) {
        bm::DoNotOptimize(matrix_c.raw_values_data());
        kernel(matrix_a.raw_values_data(), n, k, input_stride_bytes, //
               matrix_c.raw_values_data(), output_stride_bytes, 0, n);
        ++iterations;
    }

    // Symmetric operations compute upper triangle: N×(N+1)/2 dot products × K multiply-adds × 2 scalar-ops = N×(N+1)×K
    // total scalar-ops
    state.counters["scalar-ops"] = bm::Counter(iterations * n * (n + 1) * k, bm::Counter::kIsRate);
}

template <nk_dtype_t input_dtype_, nk_dtype_t output_dtype_>
void dots_symmetric_(std::string name, //
                     typename nk::type_for<input_dtype_>::type::dots_symmetric_kernel_t kernel) {
    std::string bench_name = name + "<" + std::to_string(matrix_height) + "x" + std::to_string(matrix_depth) + ">";
    bm::RegisterBenchmark(bench_name.c_str(), measure_dots_symmetric<input_dtype_, output_dtype_>, //
                          kernel, matrix_height, matrix_depth)
        ->MinTime(default_seconds)
        ->Threads(1); // Single-threaded for symmetric matmul
}

/**
 *  @brief Measure packed Hamming distance computation.
 */
template <nk_dtype_t input_dtype_, nk_dtype_t output_dtype_>
void measure_hammings_packed(                                                                //
    bm::State &state,                                                                        //
    typename nk::type_for<input_dtype_>::type::hammings_packed_size_kernel_t packed_size_fn, //
    typename nk::type_for<input_dtype_>::type::hammings_pack_kernel_t pack_fn,               //
    typename nk::type_for<input_dtype_>::type::hammings_packed_kernel_t kernel,              //
    std::size_t m, std::size_t n, std::size_t k) {

    using input_t = typename nk::type_for<input_dtype_>::type;
    using output_t = typename nk::type_for<output_dtype_>::type;
    using raw_input_t = typename input_t::raw_t;
    using raw_output_t = typename output_t::raw_t;

    // Calculate correct strides for binary data (k is in bits)
    nk_size_t values_per_row = nk::divide_round_up(k, 8);
    nk_size_t a_stride_bytes = values_per_row * sizeof(typename input_t::raw_t);
    nk_size_t b_stride_bytes = values_per_row * sizeof(typename input_t::raw_t);

    // Allocate matrices
    auto matrix_a = make_vector<input_t>(m * values_per_row);
    auto matrix_b = make_vector<input_t>(n * values_per_row);
    nk_size_t packed_bytes = packed_size_fn(n, k);
    std::vector<char> matrix_b_packed(packed_bytes, 0);
    auto matrix_c = make_vector<output_t>(m * n);

    // Initialize with random values
    auto generator = make_random_engine();
    nk::fill_uniform(generator, matrix_a.values_data(), matrix_a.size_values());
    nk::fill_uniform(generator, matrix_b.values_data(), matrix_b.size_values());

    // Pack B matrix once (amortized cost for repeated inference) with correct stride
    pack_fn(matrix_b.raw_values_data(), n, k, b_stride_bytes, matrix_b_packed.data());

    std::size_t iterations = 0;
    for (auto _ : state) {
        bm::DoNotOptimize(matrix_c.raw_values_data());
        kernel(matrix_a.raw_values_data(), matrix_b_packed.data(), matrix_c.raw_values_data(), //
               m, n, k, a_stride_bytes, n * sizeof(raw_output_t));
        ++iterations;
    }

    state.counters["scalar-ops"] = bm::Counter(iterations * m * n * k, bm::Counter::kIsRate);
}

/**
 *  @brief Measure symmetric Hamming distance matrix computation.
 */
template <nk_dtype_t input_dtype_, nk_dtype_t output_dtype_>
void measure_hammings_symmetric(                                                   //
    bm::State &state,                                                              //
    typename nk::type_for<input_dtype_>::type::hammings_symmetric_kernel_t kernel, //
    std::size_t n, std::size_t k) {

    using input_t = typename nk::type_for<input_dtype_>::type;
    using output_t = typename nk::type_for<output_dtype_>::type;
    using raw_input_t = typename input_t::raw_t;
    using raw_output_t = typename output_t::raw_t;

    // Calculate correct strides for binary data (k is in bits)
    nk_size_t input_values_per_row = nk::divide_round_up(k, 8);
    nk_size_t input_stride_bytes = input_values_per_row * sizeof(typename input_t::raw_t);
    nk_size_t output_stride_bytes = n * sizeof(raw_output_t);

    // Allocate matrix A (n vectors × k bits) and result matrix C (n × n)
    auto matrix_a = make_vector<input_t>(n * input_values_per_row);
    auto matrix_c = make_vector<output_t>(n * n);

    // Initialize with random values
    auto generator = make_random_engine();
    nk::fill_uniform(generator, matrix_a.values_data(), matrix_a.size_values());

    std::size_t iterations = 0;
    for (auto _ : state) {
        bm::DoNotOptimize(matrix_c.raw_values_data());
        kernel(matrix_a.raw_values_data(), n, k, input_stride_bytes, //
               matrix_c.raw_values_data(), output_stride_bytes, 0, n);
        ++iterations;
    }

    // Symmetric operations compute upper triangle: N×(N+1)/2 comparisons × K bits
    state.counters["scalar-ops"] = bm::Counter(iterations * n * (n + 1) * k / 2, bm::Counter::kIsRate);
}

template <nk_dtype_t input_dtype_, nk_dtype_t output_dtype_>
void hammings_(std::string name, //
               typename nk::type_for<input_dtype_>::type::hammings_packed_size_kernel_t packed_size_fn,
               typename nk::type_for<input_dtype_>::type::hammings_pack_kernel_t pack_fn,
               typename nk::type_for<input_dtype_>::type::hammings_packed_kernel_t kernel) {
    std::string bench_name = name + "<" + std::to_string(matrix_height) + "x" + std::to_string(matrix_width) + "x" +
                             std::to_string(matrix_depth) + ">";
    bm::RegisterBenchmark(bench_name.c_str(), measure_hammings_packed<input_dtype_, output_dtype_>, packed_size_fn,
                          pack_fn, kernel, matrix_height, matrix_width, matrix_depth)
        ->MinTime(default_seconds)
        ->Threads(1); // Single-threaded for packed Hamming distances
}

template <nk_dtype_t input_dtype_, nk_dtype_t output_dtype_>
void hammings_symmetric_(std::string name, //
                         typename nk::type_for<input_dtype_>::type::hammings_symmetric_kernel_t kernel) {
    std::string bench_name = name + "<" + std::to_string(matrix_height) + "x" + std::to_string(matrix_depth) + ">";
    bm::RegisterBenchmark(bench_name.c_str(), measure_hammings_symmetric<input_dtype_, output_dtype_>, //
                          kernel, matrix_height, matrix_depth)
        ->MinTime(default_seconds)
        ->Threads(1); // Single-threaded for symmetric Hamming distances
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

template <typename scalar_type_, typename kernel_type_>
void elementwise_with_stl(scalar_type_ const *ins, nk_size_t n, scalar_type_ *outs) {
    for (nk_size_t i = 0; i != n; ++i) outs[i] = kernel_type_ {}(ins[i]);
}

#if NK_COMPARE_TO_BLAS || NK_COMPARE_TO_MKL || NK_COMPARE_TO_ACCELERATE

void dot_f32_with_blas(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *result) {
    *result = cblas_sdot(static_cast<int>(n), a, 1, b, 1);
}

void dot_f64_with_blas(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result) {
    *result = cblas_ddot(static_cast<int>(n), a, 1, b, 1);
}

void dot_f32c_with_blas(nk_f32c_t const *a, nk_f32c_t const *b, nk_size_t n, nk_f32c_t *result) {
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

void dot_f64c_with_blas(nk_f64c_t const *a, nk_f64c_t const *b, nk_size_t n, nk_f64c_t *result) {
#if NK_COMPARE_TO_ACCELERATE
    cblas_zdotu_sub(static_cast<int>(n), reinterpret_cast<std::complex<double> const *>(a), 1,
                    reinterpret_cast<std::complex<double> const *>(b), 1,
                    reinterpret_cast<std::complex<double> *>(result));
#else
    cblas_zdotu_sub(static_cast<int>(n), reinterpret_cast<nk_f64_t const *>(a), 1,
                    reinterpret_cast<nk_f64_t const *>(b), 1, reinterpret_cast<nk_f64_t *>(result));
#endif
}

void vdot_f32c_with_blas(nk_f32c_t const *a, nk_f32c_t const *b, nk_size_t n, nk_f32c_t *result) {
#if NK_COMPARE_TO_ACCELERATE
    cblas_cdotc_sub(static_cast<int>(n), reinterpret_cast<std::complex<float> const *>(a), 1,
                    reinterpret_cast<std::complex<float> const *>(b), 1,
                    reinterpret_cast<std::complex<float> *>(result));
#else
    cblas_cdotc_sub(static_cast<int>(n), reinterpret_cast<nk_f32_t const *>(a), 1,
                    reinterpret_cast<nk_f32_t const *>(b), 1, reinterpret_cast<nk_f32_t *>(result));
#endif
}

void vdot_f64c_with_blas(nk_f64c_t const *a, nk_f64c_t const *b, nk_size_t n, nk_f64c_t *result) {
#if NK_COMPARE_TO_ACCELERATE
    cblas_zdotc_sub(static_cast<int>(n), reinterpret_cast<std::complex<double> const *>(a), 1,
                    reinterpret_cast<std::complex<double> const *>(b), 1,
                    reinterpret_cast<std::complex<double> *>(result));
#else
    cblas_zdotc_sub(static_cast<int>(n), reinterpret_cast<nk_f64_t const *>(a), 1,
                    reinterpret_cast<nk_f64_t const *>(b), 1, reinterpret_cast<nk_f64_t *>(result));
#endif
}

void bilinear_f32_with_blas(nk_f32_t const *a, nk_f32_t const *b, nk_f32_t const *c, nk_size_t n, nk_f32_t *result) {
    static thread_local std::vector<nk_f32_t> intermediate;
    if (intermediate.size() < n) intermediate.resize(n);
    int const ni = static_cast<int>(n);
    cblas_sgemv(CblasRowMajor, CblasNoTrans, ni, ni, 1.0f, c, ni, b, 1, 0.0f, intermediate.data(), 1);
    *result = cblas_sdot(ni, a, 1, intermediate.data(), 1);
}

void bilinear_f64_with_blas(nk_f64_t const *a, nk_f64_t const *b, nk_f64_t const *c, nk_size_t n, nk_f64_t *result) {
    static thread_local std::vector<nk_f64_t> intermediate;
    if (intermediate.size() < n) intermediate.resize(n);
    int const ni = static_cast<int>(n);
    cblas_dgemv(CblasRowMajor, CblasNoTrans, ni, ni, 1.0, c, ni, b, 1, 0.0, intermediate.data(), 1);
    *result = cblas_ddot(ni, a, 1, intermediate.data(), 1);
}

void bilinear_f32c_with_blas(nk_f32c_t const *a, nk_f32c_t const *b, nk_f32c_t const *c, nk_size_t n,
                             nk_f32c_t *results) {
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

void bilinear_f64c_with_blas(nk_f64c_t const *a, nk_f64c_t const *b, nk_f64c_t const *c, nk_size_t n,
                             nk_f64c_t *results) {
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

void sum_f32_with_blas(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *result) {
    int const ni = static_cast<int>(n);
    cblas_scopy(ni, a, 1, result, 1);
    cblas_saxpy(ni, 1.0f, b, 1, result, 1);
}

void sum_f64_with_blas(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result) {
    int const ni = static_cast<int>(n);
    cblas_dcopy(ni, a, 1, result, 1);
    cblas_daxpy(ni, 1.0, b, 1, result, 1);
}

void wsum_f32_with_blas(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                        nk_f32_t *result) {
    int const ni = static_cast<int>(n);
    std::memset(result, 0, n * sizeof(nk_f32_t));
    if (*alpha != 0) cblas_saxpy(ni, *alpha, a, 1, result, 1);
    if (*beta != 0) cblas_saxpy(ni, *beta, b, 1, result, 1);
}

void wsum_f64_with_blas(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta,
                        nk_f64_t *result) {
    int const ni = static_cast<int>(n);
    std::memset(result, 0, n * sizeof(nk_f64_t));
    if (*alpha != 0) cblas_daxpy(ni, *alpha, a, 1, result, 1);
    if (*beta != 0) cblas_daxpy(ni, *beta, b, 1, result, 1);
}

struct identity_init {
    template <typename scalar_type_>
    scalar_type_ operator()(scalar_type_ v) const noexcept {
        return v;
    }
};

/**
 *  @brief Unified batched dots measurement template supporting BLAS and MKL variants.
 *
 *  Consolidates measure_gemm_with_blas, measure_gemm_with_mkl, and measure_gemm_with_mkl_int
 *  by using type traits to select appropriate random distributions and
 *  optional init functors for type conversion (e.g., float → bf16).
 *
 *  @tparam input_type_ Type for matrix A.
 *  @tparam input_b_type_ Type for matrix B (defaults to input_type_).
 *  @tparam output_type_ Type for matrix C (defaults to input_type_).
 */
template <typename input_type_, typename input_b_type_ = input_type_, typename output_type_ = input_type_,
          typename init_first_type_ = identity_init, typename init_second_type_ = identity_init, typename kernel_type_>
void measure_dots_unpacked(bm::State &state, std::size_t m, std::size_t n, std::size_t k, kernel_type_ kernel,
                           init_first_type_ init_first = {}, init_second_type_ init_second = {}) {
    std::vector<input_type_> matrix_a(m * k);
    std::vector<input_b_type_> matrix_b(n * k);
    std::vector<output_type_> matrix_c(m * n);
    auto generator = make_random_engine();
    nk::fill_uniform(generator, matrix_a.data(), matrix_a.size());
    nk::fill_uniform(generator, matrix_b.data(), matrix_b.size());

    std::size_t iterations = 0;
    for (auto _ : state) {
        bm::DoNotOptimize(matrix_c.data());
        kernel(matrix_a.data(), matrix_b.data(), matrix_c.data(), m, n, k);
        ++iterations;
    }
    state.counters["scalar-ops"] = bm::Counter(iterations * 2.0 * m * n * k, bm::Counter::kIsRate);
}

void measure_dots_f32_with_blas(bm::State &state, std::size_t m, std::size_t n, std::size_t k) {
    measure_dots_unpacked<float>(state, m, n, k,
                                 [](float *a, float *b, float *c, std::size_t m, std::size_t n, std::size_t k) {
                                     cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, static_cast<int>(m),
                                                 static_cast<int>(n), static_cast<int>(k), 1.0f, a, static_cast<int>(k),
                                                 b, static_cast<int>(k), 0.0f, c, static_cast<int>(n));
                                 });
}

void measure_dots_f64_with_blas(bm::State &state, std::size_t m, std::size_t n, std::size_t k) {
    measure_dots_unpacked<double>(state, m, n, k,
                                  [](double *a, double *b, double *c, std::size_t m, std::size_t n, std::size_t k) {
                                      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, static_cast<int>(m),
                                                  static_cast<int>(n), static_cast<int>(k), 1.0, a, static_cast<int>(k),
                                                  b, static_cast<int>(k), 0.0, c, static_cast<int>(n));
                                  });
}

/**
 *  @brief Unified symmetric rank-k update (SYRK) benchmark template: C = A × Aᵀ
 *
 *  Follows the same pattern as measure_dots_unpacked but for symmetric operations
 *  where only a single input matrix A is needed to compute C = A × Aᵀ.
 *
 *  @tparam input_type_ Type for matrix A.
 *  @tparam output_type_ Type for matrix C (defaults to input_type_).
 */
template <typename input_type_, typename output_type_ = input_type_, typename init_type_ = identity_init,
          typename kernel_type_>
void measure_dots_symmetric_unpacked(bm::State &state, std::size_t n, std::size_t k, kernel_type_ kernel,
                                     init_type_ init = {}) {
    std::vector<input_type_> matrix_a(n * k);
    std::vector<output_type_> matrix_c(n * n);
    auto generator = make_random_engine();
    nk::fill_uniform(generator, matrix_a.data(), matrix_a.size());

    std::size_t iterations = 0;
    for (auto _ : state) {
        bm::DoNotOptimize(matrix_c.data());
        kernel(matrix_a.data(), matrix_c.data(), n, k);
        ++iterations;
    }
    // Symmetric operations compute upper triangle: N×(N+1)/2 dot products × K multiply-adds × 2 scalar-ops = N×(N+1)×K
    // total scalar-ops
    state.counters["scalar-ops"] = bm::Counter(iterations * n * (n + 1) * k, bm::Counter::kIsRate);
}

void measure_dots_symmetric_f32_with_blas(bm::State &state, std::size_t n, std::size_t k) {
    measure_dots_symmetric_unpacked<float>(state, n, k, [](float *a, float *c, std::size_t n, std::size_t k) {
        // C = α×A×Aᵀ + β×C (CblasUpper: compute upper triangle only)
        cblas_ssyrk(CblasRowMajor, CblasUpper, CblasNoTrans, static_cast<int>(n), static_cast<int>(k), 1.0f, a,
                    static_cast<int>(k), 0.0f, c, static_cast<int>(n));
    });
}

void measure_dots_symmetric_f64_with_blas(bm::State &state, std::size_t n, std::size_t k) {
    measure_dots_symmetric_unpacked<double>(state, n, k, [](double *a, double *c, std::size_t n, std::size_t k) {
        // C = α×A×Aᵀ + β×C (CblasUpper: compute upper triangle only)
        cblas_dsyrk(CblasRowMajor, CblasUpper, CblasNoTrans, static_cast<int>(n), static_cast<int>(k), 1.0, a,
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

void measure_dots_f32_with_mkl(bm::State &state, std::size_t m, std::size_t n, std::size_t k) {
    measure_dots_unpacked<float>(state, m, n, k,
                                 [](float *a, float *b, float *c, std::size_t m, std::size_t n, std::size_t k) {
                                     cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, (MKL_INT)m, (MKL_INT)n,
                                                 (MKL_INT)k, 1.0f, a, (MKL_INT)k, b, (MKL_INT)k, 0.0f, c, (MKL_INT)n);
                                 });
}

void measure_dots_bf16_with_mkl(bm::State &state, std::size_t m, std::size_t n, std::size_t k) {
    measure_dots_unpacked<MKL_BF16, MKL_BF16, float>(
        state, m, n, k,
        [](MKL_BF16 *a, MKL_BF16 *b, float *c, std::size_t m, std::size_t n, std::size_t k) {
            cblas_gemm_bf16bf16f32(CblasRowMajor, CblasNoTrans, CblasTrans, (MKL_INT)m, (MKL_INT)n, (MKL_INT)k, 1.0f, a,
                                   (MKL_INT)k, b, (MKL_INT)k, 0.0f, c, (MKL_INT)n);
        },
        f32_to_bf16, f32_to_bf16);
}

void measure_dots_f16_with_mkl(bm::State &state, std::size_t m, std::size_t n, std::size_t k) {
    measure_dots_unpacked<MKL_F16, MKL_F16, float>(
        state, m, n, k,
        [](MKL_F16 *a, MKL_F16 *b, float *c, std::size_t m, std::size_t n, std::size_t k) {
            cblas_gemm_f16f16f32(CblasRowMajor, CblasNoTrans, CblasTrans, (MKL_INT)m, (MKL_INT)n, (MKL_INT)k, 1.0f, a,
                                 (MKL_INT)k, b, (MKL_INT)k, 0.0f, c, (MKL_INT)n);
        },
        f32_to_f16, f32_to_f16);
}

void measure_dots_f64_with_mkl(bm::State &state, std::size_t m, std::size_t n, std::size_t k) {
    measure_dots_unpacked<double>(state, m, n, k,
                                  [](double *a, double *b, double *c, std::size_t m, std::size_t n, std::size_t k) {
                                      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, (MKL_INT)m, (MKL_INT)n,
                                                  (MKL_INT)k, 1.0, a, (MKL_INT)k, b, (MKL_INT)k, 0.0, c, (MKL_INT)n);
                                  });
}

void measure_dots_u8i8i32_with_mkl(bm::State &state, std::size_t m, std::size_t n, std::size_t k) {
    measure_dots_unpacked<std::uint8_t, std::int8_t, std::int32_t>(
        state, m, n, k,
        [](std::uint8_t *a, std::int8_t *b, std::int32_t *c, std::size_t m, std::size_t n, std::size_t k) {
            MKL_INT32 c_offset = 0;
            cblas_gemm_s8u8s32(CblasRowMajor, CblasNoTrans, CblasTrans, CblasFixOffset, (MKL_INT)m, (MKL_INT)n,
                               (MKL_INT)k, 1.0f, a, (MKL_INT)k, 0, b, (MKL_INT)k, 0, 0.0f, c, (MKL_INT)n, &c_offset);
        });
}

void measure_dots_i16i16i32_with_mkl(bm::State &state, std::size_t m, std::size_t n, std::size_t k) {
    measure_dots_unpacked<std::int16_t, std::int16_t, std::int32_t>(
        state, m, n, k,
        [](std::int16_t *a, std::int16_t *b, std::int32_t *c, std::size_t m, std::size_t n, std::size_t k) {
            MKL_INT32 c_offset = 0;
            cblas_gemm_s16s16s32(CblasRowMajor, CblasNoTrans, CblasTrans, CblasFixOffset, (MKL_INT)m, (MKL_INT)n,
                                 (MKL_INT)k, 1.0f, a, (MKL_INT)k, 0, b, (MKL_INT)k, 0, 0.0f, c, (MKL_INT)n, &c_offset);
        });
}

#endif

using cast_kernel_t = void (*)(void const *, nk_dtype_t, nk_size_t, void *, nk_dtype_t);

/**
 *  @brief Measures the performance of type casting operations using Google Benchmark.
 *  @param state The benchmark state object provided by Google Benchmark.
 *  @param kernel The cast kernel function to benchmark.
 *  @param count The number of elements to cast.
 */
template <nk_dtype_t input_dtype_, nk_dtype_t output_dtype_>
void measure_cast(bm::State &state, cast_kernel_t kernel, std::size_t count) {

    using input_t = typename nk::type_for<input_dtype_>::type;
    using output_t = typename nk::type_for<output_dtype_>::type;

    auto input = make_vector<input_t>(count);
    auto output = make_vector<output_t>(count);

    // Initialize input with random values
    auto generator = make_random_engine();
    nk::fill_uniform(generator, input.values_data(), count);

    // Benchmark loop
    std::size_t iterations = 0;
    for (auto _ : state) {
        kernel(input.values_data(), input_dtype_, count, output.values_data(), output_dtype_);
        bm::ClobberMemory();
        iterations++;
    }

    std::size_t const bytes_per_call = count * (sizeof(input_t) + sizeof(output_t));
    state.counters["bytes"] = bm::Counter(iterations * bytes_per_call, bm::Counter::kIsRate);
    state.counters["calls"] = bm::Counter(iterations, bm::Counter::kIsRate);
}

template <nk_dtype_t input_dtype_, nk_dtype_t output_dtype_>
void cast_(std::string name, cast_kernel_t kernel) {
    std::string bench_name = name + "<" + std::to_string(dense_dimensions) + "d>";
    bm::RegisterBenchmark(bench_name.c_str(), measure_cast<input_dtype_, output_dtype_>, kernel, dense_dimensions)
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
    if (char const *env_dense = std::getenv("NK_DENSE_DIMENSIONS")) {
        std::size_t parsed_dense = static_cast<std::size_t>(std::atoll(env_dense));
        if (parsed_dense > 0) {
            dense_dimensions = parsed_dense;
            std::printf("Using NK_DENSE_DIMENSIONS=%zu\n", dense_dimensions);
        }
    }
    if (char const *env_curved = std::getenv("NK_CURVED_DIMENSIONS")) {
        std::size_t parsed_curved = static_cast<std::size_t>(std::atoll(env_curved));
        if (parsed_curved > 0) {
            curved_dimensions = parsed_curved;
            std::printf("Using NK_CURVED_DIMENSIONS=%zu\n", curved_dimensions);
        }
    }
    if (char const *env_mesh = std::getenv("NK_MESH_POINTS")) {
        std::size_t parsed_mesh = static_cast<std::size_t>(std::atoll(env_mesh));
        if (parsed_mesh > 0) {
            mesh_points = parsed_mesh;
            std::printf("Using NK_MESH_POINTS=%zu\n", mesh_points);
        }
    }
    if (char const *env_matrix_height = std::getenv("NK_MATRIX_HEIGHT")) {
        std::size_t parsed = static_cast<std::size_t>(std::atoll(env_matrix_height));
        if (parsed > 0) {
            matrix_height = parsed;
            std::printf("Using NK_MATRIX_HEIGHT=%zu\n", matrix_height);
        }
    }
    if (char const *env_matrix_width = std::getenv("NK_MATRIX_WIDTH")) {
        std::size_t parsed = static_cast<std::size_t>(std::atoll(env_matrix_width));
        if (parsed > 0) {
            matrix_width = parsed;
            std::printf("Using NK_MATRIX_WIDTH=%zu\n", matrix_width);
        }
    }
    if (char const *env_matrix_depth = std::getenv("NK_MATRIX_DEPTH")) {
        std::size_t parsed = static_cast<std::size_t>(std::atoll(env_matrix_depth));
        if (parsed > 0) {
            matrix_depth = parsed;
            std::printf("Using NK_MATRIX_DEPTH=%zu\n", matrix_depth);
        }
    }
    if (char const *env_seed = std::getenv("NK_SEED")) {
        std::uint32_t parsed = static_cast<std::uint32_t>(std::atoll(env_seed));
        random_seed = parsed;
        std::printf("Overriding `random_seed` to %u from NK_SEED\n", random_seed);
    }
    if (char const *env_sparse_first = std::getenv("NK_SPARSE_FIRST_LENGTH")) {
        std::size_t parsed = static_cast<std::size_t>(std::atoll(env_sparse_first));
        if (parsed > 0) {
            sparse_first_length = parsed;
            std::printf("Overriding `sparse_first_length` to %zu from NK_SPARSE_FIRST_LENGTH\n", sparse_first_length);
        }
    }
    if (char const *env_sparse_second = std::getenv("NK_SPARSE_SECOND_LENGTH")) {
        std::size_t parsed = static_cast<std::size_t>(std::atoll(env_sparse_second));
        if (parsed > 0) {
            sparse_second_length = parsed;
            std::printf("Overriding `sparse_second_length` to %zu from NK_SPARSE_SECOND_LENGTH\n",
                        sparse_second_length);
        }
    }
    if (char const *env_sparse_intersection = std::getenv("NK_SPARSE_INTERSECTION")) {
        double parsed = std::atof(env_sparse_intersection);
        if (parsed >= 0.0 && parsed <= 1.0) {
            sparse_intersection_share = parsed;
            std::printf("Overriding `sparse_intersection_share` to %.2f from NK_SPARSE_INTERSECTION\n",
                        sparse_intersection_share);
        }
    }
    std::printf("\n");

    // Handle NK_FILTER environment variable by injecting --benchmark_filter argument
    std::vector<char *> modified_argv(argv, argv + argc);
    std::string filter_arg;
    if (char const *env_filter = std::getenv("NK_FILTER")) {
        filter_arg = std::string("--benchmark_filter=") + env_filter;
        modified_argv.push_back(const_cast<char *>(filter_arg.c_str()));
        std::printf("Applying benchmark filter from NK_FILTER: %s\n\n", env_filter);
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
    constexpr nk_dtype_t e2m3_k = nk_e2m3_k;
    constexpr nk_dtype_t e3m2_k = nk_e3m2_k;
    constexpr nk_dtype_t f64c_k = nk_f64c_k;
    constexpr nk_dtype_t f32c_k = nk_f32c_k;
    constexpr nk_dtype_t f16c_k = nk_f16c_k;
    constexpr nk_dtype_t bf16c_k = nk_bf16c_k;

    // Kernel kind aliases for readability
    constexpr nk_kernel_kind_t fma_k = nk_kernel_each_fma_k;
    constexpr nk_kernel_kind_t wsum_k = nk_kernel_each_blend_k;
    constexpr nk_kernel_kind_t sum_k = nk_kernel_each_sum_k;
    constexpr nk_kernel_kind_t scale_k = nk_kernel_each_scale_k;
    constexpr nk_kernel_kind_t unknown_k = nk_kernel_unknown_k;

#if NK_COMPARE_TO_BLAS || NK_COMPARE_TO_MKL || NK_COMPARE_TO_ACCELERATE

    dense_<f32_k, f32_k>("dot_f32_with_blas", dot_f32_with_blas);
    dense_<f64_k, f64_k>("dot_f64_with_blas", dot_f64_with_blas);
    dense_<f32c_k, f32c_k>("dot_f32c_with_blas", dot_f32c_with_blas);
    dense_<f64c_k, f64c_k>("dot_f64c_with_blas", dot_f64c_with_blas);
    dense_<f32c_k, f32c_k>("vdot_f32c_with_blas", vdot_f32c_with_blas);
    dense_<f64c_k, f64c_k>("vdot_f64c_with_blas", vdot_f64c_with_blas);

    elementwise_<f32_k, sum_k, f32_k>("sum_f32_with_blas", sum_f32_with_blas);
    elementwise_<f32_k, wsum_k, f32_k>("each_wsum_f32_with_blas", wsum_f32_with_blas);
    elementwise_<f64_k, sum_k, f64_k>("sum_f64_with_blas", sum_f64_with_blas);
    elementwise_<f64_k, wsum_k, f64_k>("each_wsum_f64_with_blas", wsum_f64_with_blas);

    curved_<f64_k, f64_k>("bilinear_f64_with_blas", bilinear_f64_with_blas);
    curved_<f64c_k, f64c_k>("bilinear_f64c_with_blas", bilinear_f64c_with_blas);
    curved_<f32_k, f32_k>("bilinear_f32_with_blas", bilinear_f32_with_blas);
    curved_<f32c_k, f32c_k>("bilinear_f32c_with_blas", bilinear_f32c_with_blas);

    // BLAS GEMM baselines for matmul comparison (same layout as NumKong: A×Bᵀ)
    {
        std::string dims = std::to_string(matrix_height) + "x" + std::to_string(matrix_width) + "x" +
                           std::to_string(matrix_depth);
        bm::RegisterBenchmark(("dots_f32_with_blas<" + dims + ">").c_str(), measure_dots_f32_with_blas, matrix_height,
                              matrix_width, matrix_depth)
            ->MinTime(default_seconds)
            ->Threads(1);
        bm::RegisterBenchmark(("dots_f64_with_blas<" + dims + ">").c_str(), measure_dots_f64_with_blas, matrix_height,
                              matrix_width, matrix_depth)
            ->MinTime(default_seconds)
            ->Threads(1);
    }

    // BLAS SYRK baselines for symmetric operations (correct operation for dots_symmetric: A×Aᵀ)
    {
        std::string dims = std::to_string(matrix_height) + "x" + std::to_string(matrix_depth);
        bm::RegisterBenchmark(("dots_symmetric_f32_with_blas<" + dims + ">").c_str(),
                              measure_dots_symmetric_f32_with_blas, matrix_height, matrix_depth)
            ->MinTime(default_seconds)
            ->Threads(1);
        bm::RegisterBenchmark(("dots_symmetric_f64_with_blas<" + dims + ">").c_str(),
                              measure_dots_symmetric_f64_with_blas, matrix_height, matrix_depth)
            ->MinTime(default_seconds)
            ->Threads(1);
    }

#endif // NK_COMPARE_TO_BLAS || NK_COMPARE_TO_MKL || NK_COMPARE_TO_ACCELERATE

#if NK_COMPARE_TO_MKL
    // MKL GEMM baselines for matmul comparison
    {
        std::string dims = std::to_string(matrix_height) + "x" + std::to_string(matrix_width) + "x" +
                           std::to_string(matrix_depth);
        bm::RegisterBenchmark(("dots_f32_with_mkl<" + dims + ">").c_str(), measure_dots_f32_with_mkl, matrix_height,
                              matrix_width, matrix_depth)
            ->MinTime(default_seconds)
            ->Threads(1);
        bm::RegisterBenchmark(("dots_bf16_with_mkl<" + dims + ">").c_str(), measure_dots_bf16_with_mkl, matrix_height,
                              matrix_width, matrix_depth)
            ->MinTime(default_seconds)
            ->Threads(1);
        bm::RegisterBenchmark(("dots_f16_with_mkl<" + dims + ">").c_str(), measure_dots_f16_with_mkl, matrix_height,
                              matrix_width, matrix_depth)
            ->MinTime(default_seconds)
            ->Threads(1);
        bm::RegisterBenchmark(("dots_f64_with_mkl<" + dims + ">").c_str(), measure_dots_f64_with_mkl, matrix_height,
                              matrix_width, matrix_depth)
            ->MinTime(default_seconds)
            ->Threads(1);
        bm::RegisterBenchmark(("dots_u8i8i32_with_mkl<" + dims + ">").c_str(), measure_dots_u8i8i32_with_mkl,
                              matrix_height, matrix_width, matrix_depth)
            ->MinTime(default_seconds)
            ->Threads(1);
        bm::RegisterBenchmark(("dots_i16i16i32_with_mkl<" + dims + ">").c_str(), measure_dots_i16i16i32_with_mkl,
                              matrix_height, matrix_width, matrix_depth)
            ->MinTime(default_seconds)
            ->Threads(1);
    }
#endif

#if NK_TARGET_NEON
    dense_<f32_k, f32_k>("dot_f32_neon", nk_dot_f32_neon);
    dense_<f32_k, f32_k>("angular_f32_neon", nk_angular_f32_neon);
    dense_<f32_k, f32_k>("sqeuclidean_f32_neon", nk_sqeuclidean_f32_neon);
    dense_<f32_k, f32_k>("euclidean_f32_neon", nk_euclidean_f32_neon);
    dense_<f32_k, f32_k>("kld_f32_neon", nk_kld_f32_neon);
    dense_<f32_k, f32_k>("jsd_f32_neon", nk_jsd_f32_neon);

    dense_<f64_k, f64_k>("angular_f64_neon", nk_angular_f64_neon);
    dense_<f64_k, f64_k>("sqeuclidean_f64_neon", nk_sqeuclidean_f64_neon);
    dense_<f64_k, f64_k>("euclidean_f64_neon", nk_euclidean_f64_neon);

    dense_<u1_k, u32_k>("hamming_u1_neon", nk_hamming_u1_neon);
    dense_<u1_k, f32_k>("jaccard_u1_neon", nk_jaccard_u1_neon);

    dense_<f32c_k, f32c_k>("dot_f32c_neon", nk_dot_f32c_neon);
    dense_<f32c_k, f32c_k>("vdot_f32c_neon", nk_vdot_f32c_neon);

    curved_<f32_k, f32_k>("bilinear_f32_neon", nk_bilinear_f32_neon);
    curved_<f32_k, f32_k>("mahalanobis_f32_neon", nk_mahalanobis_f32_neon);
    curved_<f32c_k, f32c_k>("bilinear_f32c_neon", nk_bilinear_f32c_neon);

    sparse_<u16_k, u64_k>("sparse_intersect_u16_neon", nk_sparse_intersect_u16_neon);
    sparse_<u32_k, u64_k>("sparse_intersect_u32_neon", nk_sparse_intersect_u32_neon);
    sparse_<u64_k, u64_k>("sparse_intersect_u64_neon", nk_sparse_intersect_u64_neon);

    elementwise_<f32_k, fma_k, f32_k>("each_fma_f32_neon", nk_each_fma_f32_neon);
    elementwise_<f32_k, wsum_k, f32_k>("each_wsum_f32_neon", nk_each_blend_f32_neon);
    elementwise_<f32_k, fma_k, f32_k>("each_fma_f32_serial", nk_each_fma_f32_serial);
    elementwise_<f32_k, wsum_k, f32_k>("each_wsum_f32_serial", nk_each_blend_f32_serial);

    mesh_<f32_k, f32_k>("rmsd_f32_neon", nk_rmsd_f32_neon);
    mesh_<f32_k, f32_k>("kabsch_f32_neon", nk_kabsch_f32_neon);
    mesh_<f32_k, f32_k>("umeyama_f32_neon", nk_umeyama_f32_neon);
    mesh_<f64_k, f64_k>("rmsd_f64_neon", nk_rmsd_f64_neon);
    mesh_<f64_k, f64_k>("kabsch_f64_neon", nk_kabsch_f64_neon);
    mesh_<f64_k, f64_k>("umeyama_f64_neon", nk_umeyama_f64_neon);

    dots_<f32_k, f32_k>("dots_packed_f32_neon", nk_dots_packed_size_f32_neon, nk_dots_pack_f32_neon,
                        nk_dots_packed_f32_neon);
    dots_<f64_k, f64_k>("dots_packed_f64_neon", nk_dots_packed_size_f64_neon, nk_dots_pack_f64_neon,
                        nk_dots_packed_f64_neon);

    dots_symmetric_<f32_k, f32_k>("dots_symmetric_f32_neon", nk_dots_symmetric_f32_neon);
    dots_symmetric_<f64_k, f64_k>("dots_symmetric_f64_neon", nk_dots_symmetric_f64_neon);

    hammings_<nk_u1_k, nk_u32_k>("hammings_u1_neon", nk_hammings_packed_size_u1_neon, nk_hammings_pack_u1_neon,
                                 nk_hammings_packed_u1_neon);

    hammings_symmetric_<nk_u1_k, nk_u32_k>("hammings_symmetric_u1_neon", nk_hammings_symmetric_u1_neon);

#endif

#if NK_TARGET_NEONSDOT
    dense_<i8_k, f32_k>("angular_i8_neonsdot", nk_angular_i8_neonsdot);
    dense_<i8_k, u32_k>("sqeuclidean_i8_neonsdot", nk_sqeuclidean_i8_neonsdot);
    dense_<i8_k, f32_k>("euclidean_i8_neonsdot", nk_euclidean_i8_neonsdot);
    dense_<i8_k, i32_k>("dot_i8_neonsdot", nk_dot_i8_neonsdot);

    dense_<u8_k, f32_k>("angular_u8_neonsdot", nk_angular_u8_neonsdot);
    dense_<u8_k, u32_k>("sqeuclidean_u8_neonsdot", nk_sqeuclidean_u8_neonsdot);
    dense_<u8_k, f32_k>("euclidean_u8_neonsdot", nk_euclidean_u8_neonsdot);
    dense_<u8_k, u32_k>("dot_u8_neonsdot", nk_dot_u8_neonsdot);

    dense_<i4_k, i32_k>("dot_i4_neonsdot", nk_dot_i4_neonsdot);
    dense_<u4_k, u32_k>("dot_u4_neonsdot", nk_dot_u4_neonsdot);

    dots_<i8_k, i32_k>("dots_packed_i8_neonsdot", nk_dots_packed_size_i8_neonsdot, nk_dots_pack_i8_neonsdot,
                       nk_dots_packed_i8_neonsdot);
    dots_<u8_k, u32_k>("dots_packed_u8_neonsdot", nk_dots_packed_size_u8_neonsdot, nk_dots_pack_u8_neonsdot,
                       nk_dots_packed_u8_neonsdot);
    dots_<i4_k, i32_k>("dots_packed_i4_neonsdot", nk_dots_packed_size_i4_neonsdot, nk_dots_pack_i4_neonsdot,
                       nk_dots_packed_i4_neonsdot);
    dots_<u4_k, u32_k>("dots_packed_u4_neonsdot", nk_dots_packed_size_u4_neonsdot, nk_dots_pack_u4_neonsdot,
                       nk_dots_packed_u4_neonsdot);

    dots_symmetric_<i8_k, i32_k>("dots_symmetric_i8_neonsdot", nk_dots_symmetric_i8_neonsdot);
    dots_symmetric_<u8_k, u32_k>("dots_symmetric_u8_neonsdot", nk_dots_symmetric_u8_neonsdot);
    dots_symmetric_<i4_k, i32_k>("dots_symmetric_i4_neonsdot", nk_dots_symmetric_i4_neonsdot);
    dots_symmetric_<u4_k, u32_k>("dots_symmetric_u4_neonsdot", nk_dots_symmetric_u4_neonsdot);

#endif

#if NK_TARGET_NEONHALF
    dense_<f16c_k, f32c_k>("dot_f16c_neonhalf", nk_dot_f16c_neonhalf);
    dense_<f16c_k, f32c_k>("vdot_f16c_neonhalf", nk_vdot_f16c_neonhalf);

    dense_<f16_k, f32_k>("dot_f16_neonhalf", nk_dot_f16_neonhalf);
    dense_<f16_k, f32_k>("angular_f16_neonhalf", nk_angular_f16_neonhalf);
    dense_<f16_k, f32_k>("sqeuclidean_f16_neonhalf", nk_sqeuclidean_f16_neonhalf);
    dense_<f16_k, f32_k>("euclidean_f16_neonhalf", nk_euclidean_f16_neonhalf);
    dense_<f16_k, f32_k>("kld_f16_neonhalf", nk_kld_f16_neonhalf);
    dense_<f16_k, f32_k>("jsd_f16_neonhalf", nk_jsd_f16_neonhalf);

    curved_<f16_k, f32_k>("bilinear_f16_neonhalf", nk_bilinear_f16_neonhalf);
    curved_<f16_k, f32_k>("mahalanobis_f16_neonhalf", nk_mahalanobis_f16_neonhalf);
    curved_<f16c_k, f32c_k>("bilinear_f16c_neonhalf", nk_bilinear_f16c_neonhalf);

    elementwise_<f16_k, fma_k, f32_k>("each_fma_f16_neonhalf", nk_each_fma_f16_neonhalf);
    elementwise_<f16_k, wsum_k, f32_k>("each_wsum_f16_neonhalf", nk_each_blend_f16_neonhalf);

    // FMA kernels for `u8` on NEON use `f16` arithmetic
    elementwise_<u8_k, fma_k, f32_k>("each_fma_u8_neonhalf", nk_each_fma_u8_neonhalf);
    elementwise_<u8_k, wsum_k, f32_k>("each_wsum_u8_neonhalf", nk_each_blend_u8_neonhalf);
    elementwise_<i8_k, fma_k, f32_k>("each_fma_i8_neonhalf", nk_each_fma_i8_neonhalf);
    elementwise_<i8_k, wsum_k, f32_k>("each_wsum_i8_neonhalf", nk_each_blend_i8_neonhalf);

    dots_<f16_k, f32_k>("dots_packed_f16_neonhalf", nk_dots_packed_size_f16_neonhalf, nk_dots_pack_f16_neonhalf,
                        nk_dots_packed_f16_neonhalf);

    dots_symmetric_<f16_k, f32_k>("dots_symmetric_f16_neonhalf", nk_dots_symmetric_f16_neonhalf);
#endif

#if NK_TARGET_NEONFHM
    dense_<f16_k, f32_k>("dot_f16_neonfhm", nk_dot_f16_neonfhm);
    dense_<e2m3_k, f32_k>("dot_e2m3_neonfhm", nk_dot_e2m3_neonfhm);
    dense_<e3m2_k, f32_k>("dot_e3m2_neonfhm", nk_dot_e3m2_neonfhm);

    dots_<f16_k, f32_k>("dots_packed_f16_neonfhm", nk_dots_packed_size_f16_neonfhm, nk_dots_pack_f16_neonfhm,
                        nk_dots_packed_f16_neonfhm);

    dots_symmetric_<f16_k, f32_k>("dots_symmetric_f16_neonfhm", nk_dots_symmetric_f16_neonfhm);
    dots_symmetric_<e2m3_k, f32_k>("dots_symmetric_e2m3_neonfhm", nk_dots_symmetric_e2m3_neonfhm);
    dots_symmetric_<e3m2_k, f32_k>("dots_symmetric_e3m2_neonfhm", nk_dots_symmetric_e3m2_neonfhm);
#endif // NK_TARGET_NEONFHM

#if NK_TARGET_NEONBFDOT
    dense_<bf16c_k, f32c_k>("dot_bf16c_neonbfdot", nk_dot_bf16c_neonbfdot);
    dense_<bf16c_k, f32c_k>("vdot_bf16c_neonbfdot", nk_vdot_bf16c_neonbfdot);

    dense_<bf16_k, f32_k>("dot_bf16_neonbfdot", nk_dot_bf16_neonbfdot);
    dense_<bf16_k, f32_k>("angular_bf16_neonbfdot", nk_angular_bf16_neonbfdot);
    dense_<bf16_k, f32_k>("sqeuclidean_bf16_neonbfdot", nk_sqeuclidean_bf16_neonbfdot);
    dense_<bf16_k, f32_k>("euclidean_bf16_neonbfdot", nk_euclidean_bf16_neonbfdot);

    curved_<bf16_k, f32_k>("bilinear_bf16_neonbfdot", nk_bilinear_bf16_neonbfdot);
    curved_<bf16_k, f32_k>("mahalanobis_bf16_neonbfdot", nk_mahalanobis_bf16_neonbfdot);
    curved_<bf16c_k, f32c_k>("bilinear_bf16c_neonbfdot", nk_bilinear_bf16c_neonbfdot);

    elementwise_<bf16_k, fma_k, f32_k>("each_fma_bf16_neonbfdot", nk_each_fma_bf16_neonbfdot);
    elementwise_<bf16_k, wsum_k, f32_k>("each_wsum_bf16_neonbfdot", nk_each_blend_bf16_neonbfdot);

    dots_<bf16_k, f32_k>("dots_packed_bf16_neonbfdot", nk_dots_packed_size_bf16_neonbfdot, nk_dots_pack_bf16_neonbfdot,
                         nk_dots_packed_bf16_neonbfdot);
#endif

#if NK_TARGET_SVE
    dense_<f32_k, f32_k>("dot_f32_sve", nk_dot_f32_sve);
    dense_<f32_k, f32_k>("angular_f32_sve", nk_angular_f32_sve);
    dense_<f32_k, f32_k>("sqeuclidean_f32_sve", nk_sqeuclidean_f32_sve);
    dense_<f32_k, f32_k>("euclidean_f32_sve", nk_euclidean_f32_sve);

    dense_<f64_k, f64_k>("dot_f64_sve", nk_dot_f64_sve);
    dense_<f64_k, f64_k>("angular_f64_sve", nk_angular_f64_sve);
    dense_<f64_k, f64_k>("sqeuclidean_f64_sve", nk_sqeuclidean_f64_sve);
    dense_<f64_k, f64_k>("euclidean_f64_sve", nk_euclidean_f64_sve);

    dense_<u1_k, u32_k>("hamming_u1_sve", nk_hamming_u1_sve);
    dense_<u1_k, f32_k>("jaccard_u1_sve", nk_jaccard_u1_sve);

    dense_<f32c_k, f32c_k>("dot_f32c_sve", nk_dot_f32c_sve);
    dense_<f32c_k, f32c_k>("vdot_f32c_sve", nk_vdot_f32c_sve);
    dense_<f64c_k, f64c_k>("dot_f64c_sve", nk_dot_f64c_sve);
    dense_<f64c_k, f64c_k>("vdot_f64c_sve", nk_vdot_f64c_sve);
#endif

#if NK_TARGET_SVEHALF
    dense_<f16_k, f32_k>("dot_f16_svehalf", nk_dot_f16_svehalf);
    dense_<f16_k, f32_k>("angular_f16_svehalf", nk_angular_f16_svehalf);
    dense_<f16_k, f32_k>("sqeuclidean_f16_svehalf", nk_sqeuclidean_f16_svehalf);
    dense_<f16_k, f32_k>("euclidean_f16_svehalf", nk_euclidean_f16_svehalf);
    dense_<f16c_k, f32c_k>("dot_f16c_svehalf", nk_dot_f16c_svehalf);
    dense_<f16c_k, f32c_k>("vdot_f16c_svehalf", nk_vdot_f16c_svehalf);
#endif

#if NK_TARGET_SVEBFDOT
    dense_<bf16_k, f32_k>("angular_bf16_svebfdot", nk_angular_bf16_svebfdot);
    dense_<bf16_k, f32_k>("sqeuclidean_bf16_svebfdot", nk_sqeuclidean_bf16_svebfdot);
    dense_<bf16_k, f32_k>("euclidean_bf16_svebfdot", nk_euclidean_bf16_svebfdot);
#endif

#if NK_TARGET_SVE2
    sparse_<u16_k, u32_k, u32_k>("sparse_intersect_u16_sve2", nk_sparse_intersect_u16_sve2);
    sparse_<u32_k, u32_k, u32_k>("sparse_intersect_u32_sve2", nk_sparse_intersect_u32_sve2);
    sparse_<u64_k, u64_k>("sparse_intersect_u64_sve2", nk_sparse_intersect_u64_sve2);
#endif

#if NK_TARGET_SME
    dots_<f16_k, f32_k>("dots_packed_f16_sme", nk_dots_packed_size_f16_sme, nk_dots_pack_f16_sme,
                        nk_dots_packed_f16_sme);
    dots_<bf16_k, f32_k>("dots_packed_bf16_sme", nk_dots_packed_size_bf16_sme, nk_dots_pack_bf16_sme,
                         nk_dots_packed_bf16_sme);
    dots_<i8_k, i32_k>("dots_packed_i8_sme", nk_dots_packed_size_i8_sme, nk_dots_pack_i8_sme, nk_dots_packed_i8_sme);
    dots_<u8_k, u32_k>("dots_packed_u8_sme", nk_dots_packed_size_u8_sme, nk_dots_pack_u8_sme, nk_dots_packed_u8_sme);
    dots_<e4m3_k, f32_k>("dots_packed_e4m3_sme", nk_dots_packed_size_e4m3_sme, nk_dots_pack_e4m3_sme,
                         nk_dots_packed_e4m3_sme);
    dots_<e5m2_k, f32_k>("dots_packed_e5m2_sme", nk_dots_packed_size_e5m2_sme, nk_dots_pack_e5m2_sme,
                         nk_dots_packed_e5m2_sme);

    dots_symmetric_<bf16_k, f32_k>("dots_symmetric_bf16_sme", nk_dots_symmetric_bf16_sme);
    dots_symmetric_<f16_k, f32_k>("dots_symmetric_f16_sme", nk_dots_symmetric_f16_sme);
    dots_symmetric_<i8_k, i32_k>("dots_symmetric_i8_sme", nk_dots_symmetric_i8_sme);
    dots_symmetric_<u8_k, u32_k>("dots_symmetric_u8_sme", nk_dots_symmetric_u8_sme);
    dots_symmetric_<e4m3_k, f32_k>("dots_symmetric_e4m3_sme", nk_dots_symmetric_e4m3_sme);
    dots_symmetric_<e5m2_k, f32_k>("dots_symmetric_e5m2_sme", nk_dots_symmetric_e5m2_sme);
    dots_symmetric_<i4_k, i32_k>("dots_symmetric_i4_sme", nk_dots_symmetric_i4_sme);
    dots_symmetric_<u4_k, u32_k>("dots_symmetric_u4_sme", nk_dots_symmetric_u4_sme);
#endif
#if NK_TARGET_SMEF64
    dots_<f32_k, f32_k>("dots_packed_f32_smef64", nk_dots_packed_size_f32_smef64, nk_dots_pack_f32_smef64,
                        nk_dots_packed_f32_smef64);
    dots_<f64_k, f64_k>("dots_packed_f64_smef64", nk_dots_packed_size_f64_smef64, nk_dots_pack_f64_smef64,
                        nk_dots_packed_f64_smef64);

    curved_<f32_k, f32_k, f64_k>("bilinear_f32_smef64", nk_bilinear_f32_smef64);
    curved_<f32c_k, f32c_k, f64c_k>("bilinear_f32c_smef64", nk_bilinear_f32c_smef64);
    curved_<f32_k, f32_k, f64_k>("mahalanobis_f32_smef64", nk_mahalanobis_f32_smef64);
#endif

#if NK_TARGET_HASWELL
    dense_<f16_k, f32_k>("dot_f16_haswell", nk_dot_f16_haswell);
    dense_<f16_k, f32_k>("angular_f16_haswell", nk_angular_f16_haswell);
    dense_<f16_k, f32_k>("sqeuclidean_f16_haswell", nk_sqeuclidean_f16_haswell);
    dense_<f16_k, f32_k>("euclidean_f16_haswell", nk_euclidean_f16_haswell);
    dense_<f16_k, f32_k>("kld_f16_haswell", nk_kld_f16_haswell);
    dense_<f16_k, f32_k>("jsd_f16_haswell", nk_jsd_f16_haswell);

    dense_<bf16_k, f32_k>("dot_bf16_haswell", nk_dot_bf16_haswell);
    dense_<bf16_k, f32_k>("angular_bf16_haswell", nk_angular_bf16_haswell);
    dense_<bf16_k, f32_k>("sqeuclidean_bf16_haswell", nk_sqeuclidean_bf16_haswell);
    dense_<bf16_k, f32_k>("euclidean_bf16_haswell", nk_euclidean_bf16_haswell);

    dense_<e4m3_k, f32_k>("dot_e4m3_haswell", nk_dot_e4m3_haswell);
    dense_<e5m2_k, f32_k>("dot_e5m2_haswell", nk_dot_e5m2_haswell);
    dense_<e2m3_k, f32_k>("dot_e2m3_haswell", nk_dot_e2m3_haswell);
    dense_<e3m2_k, f32_k>("dot_e3m2_haswell", nk_dot_e3m2_haswell);

    dense_<i8_k, f32_k>("angular_i8_haswell", nk_angular_i8_haswell);
    dense_<i8_k, u32_k>("sqeuclidean_i8_haswell", nk_sqeuclidean_i8_haswell);
    dense_<i8_k, f32_k>("euclidean_i8_haswell", nk_euclidean_i8_haswell);
    dense_<i8_k, i32_k>("dot_i8_haswell", nk_dot_i8_haswell);

    dense_<u8_k, f32_k>("angular_u8_haswell", nk_angular_u8_haswell);
    dense_<u8_k, u32_k>("sqeuclidean_u8_haswell", nk_sqeuclidean_u8_haswell);
    dense_<u8_k, f32_k>("euclidean_u8_haswell", nk_euclidean_u8_haswell);
    dense_<u8_k, u32_k>("dot_u8_haswell", nk_dot_u8_haswell);

    dense_<u1_k, u32_k>("hamming_u1_haswell", nk_hamming_u1_haswell);
    dense_<u1_k, f32_k>("jaccard_u1_haswell", nk_jaccard_u1_haswell);

    dense_<f16c_k, f32c_k>("dot_f16c_haswell", nk_dot_f16c_haswell);
    dense_<f16c_k, f32c_k>("vdot_f16c_haswell", nk_vdot_f16c_haswell);
    dense_<f32c_k, f32c_k>("dot_f32c_haswell", nk_dot_f32c_haswell);
    dense_<f32c_k, f32c_k>("vdot_f32c_haswell", nk_vdot_f32c_haswell);
    dense_<bf16c_k, f32c_k>("dot_bf16c_haswell", nk_dot_bf16c_haswell);
    dense_<bf16c_k, f32c_k>("vdot_bf16c_haswell", nk_vdot_bf16c_haswell);
    dense_<i4_k, i32_k>("dot_i4_haswell", nk_dot_i4_haswell);
    dense_<u4_k, u32_k>("dot_u4_haswell", nk_dot_u4_haswell);

    curved_<f16_k, f32_k>("bilinear_f16_haswell", nk_bilinear_f16_haswell);
    curved_<f16_k, f32_k>("mahalanobis_f16_haswell", nk_mahalanobis_f16_haswell);
    curved_<bf16_k, f32_k>("bilinear_bf16_haswell", nk_bilinear_bf16_haswell);
    curved_<bf16_k, f32_k>("mahalanobis_bf16_haswell", nk_mahalanobis_bf16_haswell);

    elementwise_<f64_k, scale_k, f64_k>("each_scale_f64_haswell", nk_each_scale_f64_haswell);
    elementwise_<f64_k, fma_k, f64_k>("each_fma_f64_haswell", nk_each_fma_f64_haswell);
    elementwise_<f64_k, wsum_k, f64_k>("each_wsum_f64_haswell", nk_each_blend_f64_haswell);
    elementwise_<f32_k, scale_k, f32_k>("each_scale_f32_haswell", nk_each_scale_f32_haswell);
    elementwise_<f32_k, fma_k, f32_k>("each_fma_f32_haswell", nk_each_fma_f32_haswell);
    elementwise_<f32_k, wsum_k, f32_k>("each_wsum_f32_haswell", nk_each_blend_f32_haswell);
    elementwise_<f16_k, scale_k, f32_k>("each_scale_f16_haswell", nk_each_scale_f16_haswell);
    elementwise_<f16_k, fma_k, f32_k>("each_fma_f16_haswell", nk_each_fma_f16_haswell);
    elementwise_<f16_k, wsum_k, f32_k>("each_wsum_f16_haswell", nk_each_blend_f16_haswell);
    elementwise_<bf16_k, scale_k, f32_k>("each_scale_bf16_haswell", nk_each_scale_bf16_haswell);
    elementwise_<bf16_k, fma_k, f32_k>("each_fma_bf16_haswell", nk_each_fma_bf16_haswell);
    elementwise_<bf16_k, wsum_k, f32_k>("each_wsum_bf16_haswell", nk_each_blend_bf16_haswell);
    elementwise_<i8_k, scale_k, f32_k>("each_scale_i8_haswell", nk_each_scale_i8_haswell);
    elementwise_<i8_k, fma_k, f32_k>("each_fma_i8_haswell", nk_each_fma_i8_haswell);
    elementwise_<i8_k, wsum_k, f32_k>("each_wsum_i8_haswell", nk_each_blend_i8_haswell);
    elementwise_<u8_k, scale_k, f32_k>("each_scale_u8_haswell", nk_each_scale_u8_haswell);
    elementwise_<u8_k, fma_k, f32_k>("each_fma_u8_haswell", nk_each_fma_u8_haswell);
    elementwise_<u8_k, wsum_k, f32_k>("each_wsum_u8_haswell", nk_each_blend_u8_haswell);
    elementwise_<i16_k, scale_k, f32_k>("each_scale_i16_haswell", nk_each_scale_i16_haswell);
    elementwise_<i16_k, fma_k, f32_k>("each_fma_i16_haswell", nk_each_fma_i16_haswell);
    elementwise_<u16_k, scale_k, f32_k>("each_scale_u16_haswell", nk_each_scale_u16_haswell);
    elementwise_<u16_k, fma_k, f32_k>("each_fma_u16_haswell", nk_each_fma_u16_haswell);

    elementwise_<f32_k, unknown_k, f32_k>("sin_f32_haswell", nk_sin_f32_haswell);
    elementwise_<f32_k, unknown_k, f32_k>("cos_f32_haswell", nk_cos_f32_haswell);
    elementwise_<f32_k, unknown_k, f32_k>("atan_f32_haswell", nk_atan_f32_haswell);
    elementwise_<f64_k, unknown_k, f64_k>("sin_f64_haswell", nk_sin_f64_haswell);
    elementwise_<f64_k, unknown_k, f64_k>("cos_f64_haswell", nk_cos_f64_haswell);
    elementwise_<f64_k, unknown_k, f64_k>("atan_f64_haswell", nk_atan_f64_haswell);

    geospatial_<f32_k, f32_k>("haversine_f32_haswell", nk_haversine_f32_haswell);
    geospatial_<f64_k, f64_k>("haversine_f64_haswell", nk_haversine_f64_haswell);
    geospatial_<f32_k, f32_k>("vincenty_f32_haswell", nk_vincenty_f32_haswell);
    geospatial_<f64_k, f64_k>("vincenty_f64_haswell", nk_vincenty_f64_haswell);

    dots_<f32_k, f32_k>("dots_packed_f32_haswell", nk_dots_packed_size_f32_haswell, nk_dots_pack_f32_haswell,
                        nk_dots_packed_f32_haswell);
    dots_<f64_k, f64_k>("dots_packed_f64_haswell", nk_dots_packed_size_f64_haswell, nk_dots_pack_f64_haswell,
                        nk_dots_packed_f64_haswell);
    dots_<f16_k, f32_k>("dots_packed_f16_haswell", nk_dots_packed_size_f16_haswell, nk_dots_pack_f16_haswell,
                        nk_dots_packed_f16_haswell);
    dots_<bf16_k, f32_k>("dots_packed_bf16_haswell", nk_dots_packed_size_bf16_haswell, nk_dots_pack_bf16_haswell,
                         nk_dots_packed_bf16_haswell);
    dots_<e4m3_k, f32_k>("dots_packed_e4m3_haswell", nk_dots_packed_size_e4m3_haswell, nk_dots_pack_e4m3_haswell,
                         nk_dots_packed_e4m3_haswell);
    dots_<e5m2_k, f32_k>("dots_packed_e5m2_haswell", nk_dots_packed_size_e5m2_haswell, nk_dots_pack_e5m2_haswell,
                         nk_dots_packed_e5m2_haswell);
    dots_<e2m3_k, f32_k>("dots_packed_e2m3_haswell", nk_dots_packed_size_e2m3_haswell, nk_dots_pack_e2m3_haswell,
                         nk_dots_packed_e2m3_haswell);
    dots_<e3m2_k, f32_k>("dots_packed_e3m2_haswell", nk_dots_packed_size_e3m2_haswell, nk_dots_pack_e3m2_haswell,
                         nk_dots_packed_e3m2_haswell);
    dots_<i8_k, i32_k>("dots_packed_i8_haswell", nk_dots_packed_size_i8_haswell, nk_dots_pack_i8_haswell,
                       nk_dots_packed_i8_haswell);
    dots_<u8_k, u32_k>("dots_packed_u8_haswell", nk_dots_packed_size_u8_haswell, nk_dots_pack_u8_haswell,
                       nk_dots_packed_u8_haswell);

    dots_symmetric_<f32_k, f32_k>("dots_symmetric_f32_haswell", nk_dots_symmetric_f32_haswell);
    dots_symmetric_<f64_k, f64_k>("dots_symmetric_f64_haswell", nk_dots_symmetric_f64_haswell);
    dots_symmetric_<bf16_k, f32_k>("dots_symmetric_bf16_haswell", nk_dots_symmetric_bf16_haswell);
    dots_symmetric_<f16_k, f32_k>("dots_symmetric_f16_haswell", nk_dots_symmetric_f16_haswell);
    dots_symmetric_<i8_k, i32_k>("dots_symmetric_i8_haswell", nk_dots_symmetric_i8_haswell);
    dots_symmetric_<u8_k, u32_k>("dots_symmetric_u8_haswell", nk_dots_symmetric_u8_haswell);
    dots_symmetric_<e4m3_k, f32_k>("dots_symmetric_e4m3_haswell", nk_dots_symmetric_e4m3_haswell);
    dots_symmetric_<e5m2_k, f32_k>("dots_symmetric_e5m2_haswell", nk_dots_symmetric_e5m2_haswell);
    dots_symmetric_<e2m3_k, f32_k>("dots_symmetric_e2m3_haswell", nk_dots_symmetric_e2m3_haswell);
    dots_symmetric_<e3m2_k, f32_k>("dots_symmetric_e3m2_haswell", nk_dots_symmetric_e3m2_haswell);

    hammings_<nk_u1_k, nk_u32_k>("hammings_u1_haswell", nk_hammings_packed_size_u1_haswell, nk_hammings_pack_u1_haswell,
                                 nk_hammings_packed_u1_haswell);

    hammings_symmetric_<nk_u1_k, nk_u32_k>("hammings_symmetric_u1_haswell", nk_hammings_symmetric_u1_haswell);

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
    dense_<f32_k, f32_k>("dot_f32_skylake", nk_dot_f32_skylake);
    dense_<f32_k, f32_k>("angular_f32_skylake", nk_angular_f32_skylake);
    dense_<f32_k, f32_k>("sqeuclidean_f32_skylake", nk_sqeuclidean_f32_skylake);
    dense_<f32_k, f32_k>("euclidean_f32_skylake", nk_euclidean_f32_skylake);
    dense_<f32_k, f32_k>("kld_f32_skylake", nk_kld_f32_skylake);
    dense_<f32_k, f32_k>("jsd_f32_skylake", nk_jsd_f32_skylake);

    dense_<f64_k, f64_k>("dot_f64_skylake", nk_dot_f64_skylake);
    dense_<f64_k, f64_k>("angular_f64_skylake", nk_angular_f64_skylake);
    dense_<f64_k, f64_k>("sqeuclidean_f64_skylake", nk_sqeuclidean_f64_skylake);
    dense_<f64_k, f64_k>("euclidean_f64_skylake", nk_euclidean_f64_skylake);

    dense_<bf16_k, f32_k>("dot_bf16_skylake", nk_dot_bf16_skylake);
    dense_<f16_k, f32_k>("dot_f16_skylake", nk_dot_f16_skylake);

    dense_<e4m3_k, f32_k>("dot_e4m3_skylake", nk_dot_e4m3_skylake);
    dense_<e5m2_k, f32_k>("dot_e5m2_skylake", nk_dot_e5m2_skylake);
    dense_<e2m3_k, f32_k>("dot_e2m3_skylake", nk_dot_e2m3_skylake);
    dense_<e3m2_k, f32_k>("dot_e3m2_skylake", nk_dot_e3m2_skylake);

    dense_<e4m3_k, f32_k>("angular_e4m3_skylake", nk_angular_e4m3_skylake);
    dense_<e4m3_k, f32_k>("sqeuclidean_e4m3_skylake", nk_sqeuclidean_e4m3_skylake);
    dense_<e4m3_k, f32_k>("euclidean_e4m3_skylake", nk_euclidean_e4m3_skylake);
    dense_<e5m2_k, f32_k>("angular_e5m2_skylake", nk_angular_e5m2_skylake);
    dense_<e5m2_k, f32_k>("sqeuclidean_e5m2_skylake", nk_sqeuclidean_e5m2_skylake);
    dense_<e5m2_k, f32_k>("euclidean_e5m2_skylake", nk_euclidean_e5m2_skylake);
    dense_<e2m3_k, f32_k>("angular_e2m3_skylake", nk_angular_e2m3_skylake);
    dense_<e2m3_k, f32_k>("sqeuclidean_e2m3_skylake", nk_sqeuclidean_e2m3_skylake);
    dense_<e2m3_k, f32_k>("euclidean_e2m3_skylake", nk_euclidean_e2m3_skylake);
    dense_<e3m2_k, f32_k>("angular_e3m2_skylake", nk_angular_e3m2_skylake);
    dense_<e3m2_k, f32_k>("sqeuclidean_e3m2_skylake", nk_sqeuclidean_e3m2_skylake);
    dense_<e3m2_k, f32_k>("euclidean_e3m2_skylake", nk_euclidean_e3m2_skylake);

    dense_<i8_k, i32_k>("dot_i8_skylake", nk_dot_i8_skylake);
    dense_<u8_k, u32_k>("dot_u8_skylake", nk_dot_u8_skylake);

    dense_<f32c_k, f32c_k>("dot_f32c_skylake", nk_dot_f32c_skylake);
    dense_<f32c_k, f32c_k>("vdot_f32c_skylake", nk_vdot_f32c_skylake);
    dense_<f64c_k, f64c_k>("dot_f64c_skylake", nk_dot_f64c_skylake);
    dense_<f64c_k, f64c_k>("vdot_f64c_skylake", nk_vdot_f64c_skylake);

    curved_<f32_k, f32_k>("bilinear_f32_skylake", nk_bilinear_f32_skylake);
    curved_<f32c_k, f32c_k>("bilinear_f32c_skylake", nk_bilinear_f32c_skylake);
    curved_<f64_k, f64_k>("bilinear_f64_skylake", nk_bilinear_f64_skylake);
    curved_<f64c_k, f64c_k>("bilinear_f64c_skylake", nk_bilinear_f64c_skylake);

    elementwise_<f64_k, fma_k, f64_k>("each_fma_f64_skylake", nk_each_fma_f64_skylake);
    elementwise_<f64_k, wsum_k, f64_k>("each_wsum_f64_skylake", nk_each_blend_f64_skylake);
    elementwise_<f32_k, fma_k, f32_k>("each_fma_f32_skylake", nk_each_fma_f32_skylake);
    elementwise_<f32_k, wsum_k, f32_k>("each_wsum_f32_skylake", nk_each_blend_f32_skylake);
    elementwise_<bf16_k, fma_k, f32_k>("each_fma_bf16_skylake", nk_each_fma_bf16_skylake);
    elementwise_<bf16_k, wsum_k, f32_k>("each_wsum_bf16_skylake", nk_each_blend_bf16_skylake);

    elementwise_<f32_k, unknown_k, f32_k>("sin_f32_skylake", nk_sin_f32_skylake);
    elementwise_<f32_k, unknown_k, f32_k>("cos_f32_skylake", nk_cos_f32_skylake);
    elementwise_<f32_k, unknown_k, f32_k>("atan_f32_skylake", nk_atan_f32_skylake);
    elementwise_<f64_k, unknown_k, f64_k>("sin_f64_skylake", nk_sin_f64_skylake);
    elementwise_<f64_k, unknown_k, f64_k>("cos_f64_skylake", nk_cos_f64_skylake);
    elementwise_<f64_k, unknown_k, f64_k>("atan_f64_skylake", nk_atan_f64_skylake);

    geospatial_<f32_k, f32_k>("haversine_f32_skylake", nk_haversine_f32_skylake);
    geospatial_<f64_k, f64_k>("haversine_f64_skylake", nk_haversine_f64_skylake);
    geospatial_<f32_k, f32_k>("vincenty_f32_skylake", nk_vincenty_f32_skylake);
    geospatial_<f64_k, f64_k>("vincenty_f64_skylake", nk_vincenty_f64_skylake);

    mesh_<f32_k, f32_k>("rmsd_f32_skylake", nk_rmsd_f32_skylake);
    mesh_<f32_k, f32_k>("kabsch_f32_skylake", nk_kabsch_f32_skylake);

    dots_<f32_k, f32_k>("dots_packed_f32_skylake", nk_dots_packed_size_f32_skylake, nk_dots_pack_f32_skylake,
                        nk_dots_packed_f32_skylake);
    dots_<f64_k, f64_k>("dots_packed_f64_skylake", nk_dots_packed_size_f64_skylake, nk_dots_pack_f64_skylake,
                        nk_dots_packed_f64_skylake);
    dots_<bf16_k, f32_k>("dots_packed_bf16_skylake", nk_dots_packed_size_bf16_skylake, nk_dots_pack_bf16_skylake,
                         nk_dots_packed_bf16_skylake);
    dots_<f16_k, f32_k>("dots_packed_f16_skylake", nk_dots_packed_size_f16_skylake, nk_dots_pack_f16_skylake,
                        nk_dots_packed_f16_skylake);
    dots_<e4m3_k, f32_k>("dots_packed_e4m3_skylake", nk_dots_packed_size_e4m3_skylake, nk_dots_pack_e4m3_skylake,
                         nk_dots_packed_e4m3_skylake);
    dots_<e5m2_k, f32_k>("dots_packed_e5m2_skylake", nk_dots_packed_size_e5m2_skylake, nk_dots_pack_e5m2_skylake,
                         nk_dots_packed_e5m2_skylake);
    dots_<e2m3_k, f32_k>("dots_packed_e2m3_skylake", nk_dots_packed_size_e2m3_skylake, nk_dots_pack_e2m3_skylake,
                         nk_dots_packed_e2m3_skylake);
    dots_<e3m2_k, f32_k>("dots_packed_e3m2_skylake", nk_dots_packed_size_e3m2_skylake, nk_dots_pack_e3m2_skylake,
                         nk_dots_packed_e3m2_skylake);

    dots_symmetric_<f32_k, f32_k>("dots_symmetric_f32_skylake", nk_dots_symmetric_f32_skylake);
    dots_symmetric_<f64_k, f64_k>("dots_symmetric_f64_skylake", nk_dots_symmetric_f64_skylake);
    dots_symmetric_<bf16_k, f32_k>("dots_symmetric_bf16_skylake", nk_dots_symmetric_bf16_skylake);
    dots_symmetric_<f16_k, f32_k>("dots_symmetric_f16_skylake", nk_dots_symmetric_f16_skylake);
    dots_symmetric_<e4m3_k, f32_k>("dots_symmetric_e4m3_skylake", nk_dots_symmetric_e4m3_skylake);
    dots_symmetric_<e5m2_k, f32_k>("dots_symmetric_e5m2_skylake", nk_dots_symmetric_e5m2_skylake);
    dots_symmetric_<e2m3_k, f32_k>("dots_symmetric_e2m3_skylake", nk_dots_symmetric_e2m3_skylake);
    dots_symmetric_<e3m2_k, f32_k>("dots_symmetric_e3m2_skylake", nk_dots_symmetric_e3m2_skylake);

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
    dense_<i8_k, f32_k>("angular_i8_ice", nk_angular_i8_ice);
    dense_<i8_k, u32_k>("sqeuclidean_i8_ice", nk_sqeuclidean_i8_ice);
    dense_<i8_k, f32_k>("euclidean_i8_ice", nk_euclidean_i8_ice);
    dense_<i8_k, i32_k>("dot_i8_ice", nk_dot_i8_ice);

    dense_<u8_k, f32_k>("angular_u8_ice", nk_angular_u8_ice);
    dense_<u8_k, u32_k>("sqeuclidean_u8_ice", nk_sqeuclidean_u8_ice);
    dense_<u8_k, f32_k>("euclidean_u8_ice", nk_euclidean_u8_ice);
    dense_<u8_k, u32_k>("dot_u8_ice", nk_dot_u8_ice);

    dense_<i4_k, f32_k>("angular_i4_ice", nk_angular_i4_ice);
    dense_<i4_k, u32_k>("sqeuclidean_i4_ice", nk_sqeuclidean_i4_ice);
    dense_<i4_k, f32_k>("euclidean_i4_ice", nk_euclidean_i4_ice);
    dense_<i4_k, i32_k>("dot_i4_ice", nk_dot_i4_ice);

    dense_<u4_k, f32_k>("angular_u4_ice", nk_angular_u4_ice);
    dense_<u4_k, u32_k>("sqeuclidean_u4_ice", nk_sqeuclidean_u4_ice);
    dense_<u4_k, f32_k>("euclidean_u4_ice", nk_euclidean_u4_ice);
    dense_<u4_k, u32_k>("dot_u4_ice", nk_dot_u4_ice);

    dense_<u1_k, u32_k>("hamming_u1_ice", nk_hamming_u1_ice);
    dense_<u1_k, f32_k>("jaccard_u1_ice", nk_jaccard_u1_ice);

    sparse_<u16_k, u32_k>("sparse_intersect_u16_ice", nk_sparse_intersect_u16_ice);
    sparse_<u32_k, u32_k>("sparse_intersect_u32_ice", nk_sparse_intersect_u32_ice);
    sparse_<u64_k, u64_k>("sparse_intersect_u64_ice", nk_sparse_intersect_u64_ice);

    dots_<i4_k, i32_k>("dots_packed_i4_ice", nk_dots_packed_size_i4_ice, nk_dots_pack_i4_ice, nk_dots_packed_i4_ice);
    dots_<u4_k, u32_k>("dots_packed_u4_ice", nk_dots_packed_size_u4_ice, nk_dots_pack_u4_ice, nk_dots_packed_u4_ice);
    dots_<i8_k, i32_k>("dots_packed_i8_ice", nk_dots_packed_size_i8_ice, nk_dots_pack_i8_ice, nk_dots_packed_i8_ice);
    dots_<u8_k, u32_k>("dots_packed_u8_ice", nk_dots_packed_size_u8_ice, nk_dots_pack_u8_ice, nk_dots_packed_u8_ice);

    dots_symmetric_<i8_k, i32_k>("dots_symmetric_i8_ice", nk_dots_symmetric_i8_ice);
    dots_symmetric_<u8_k, u32_k>("dots_symmetric_u8_ice", nk_dots_symmetric_u8_ice);
    dots_symmetric_<i4_k, i32_k>("dots_symmetric_i4_ice", nk_dots_symmetric_i4_ice);
    dots_symmetric_<u4_k, u32_k>("dots_symmetric_u4_ice", nk_dots_symmetric_u4_ice);

    hammings_<nk_u1_k, nk_u32_k>("hammings_u1_ice", nk_hammings_packed_size_u1_ice, nk_hammings_pack_u1_ice,
                                 nk_hammings_packed_u1_ice);

    hammings_symmetric_<nk_u1_k, nk_u32_k>("hammings_symmetric_u1_ice", nk_hammings_symmetric_u1_ice);

    cast_<nk_f32_k, nk_f16_k>("cast_f32_to_f16_ice", nk_cast_ice);
    cast_<nk_f16_k, nk_f32_k>("cast_f16_to_f32_ice", nk_cast_ice);
    cast_<nk_f32_k, nk_e4m3_k>("cast_f32_to_e4m3_ice", nk_cast_ice);
    cast_<nk_e4m3_k, nk_f32_k>("cast_e4m3_to_f32_ice", nk_cast_ice);
#endif

#if NK_TARGET_GENOA
    dense_<bf16_k, f32_k>("dot_bf16_genoa", nk_dot_bf16_genoa);
    dense_<bf16_k, f32_k>("angular_bf16_genoa", nk_angular_bf16_genoa);
    dense_<bf16_k, f32_k>("sqeuclidean_bf16_genoa", nk_sqeuclidean_bf16_genoa);
    dense_<bf16_k, f32_k>("euclidean_bf16_genoa", nk_euclidean_bf16_genoa);
    dense_<bf16c_k, f32c_k>("dot_bf16c_genoa", nk_dot_bf16c_genoa);
    dense_<bf16c_k, f32c_k>("vdot_bf16c_genoa", nk_vdot_bf16c_genoa);

    dense_<e4m3_k, f32_k>("dot_e4m3_genoa", nk_dot_e4m3_genoa);
    dense_<e5m2_k, f32_k>("dot_e5m2_genoa", nk_dot_e5m2_genoa);
    dense_<e2m3_k, f32_k>("dot_e2m3_genoa", nk_dot_e2m3_genoa);
    dense_<e3m2_k, f32_k>("dot_e3m2_genoa", nk_dot_e3m2_genoa);

    dense_<e4m3_k, f32_k>("angular_e4m3_genoa", nk_angular_e4m3_genoa);
    dense_<e4m3_k, f32_k>("sqeuclidean_e4m3_genoa", nk_sqeuclidean_e4m3_genoa);
    dense_<e4m3_k, f32_k>("euclidean_e4m3_genoa", nk_euclidean_e4m3_genoa);
    dense_<e5m2_k, f32_k>("angular_e5m2_genoa", nk_angular_e5m2_genoa);
    dense_<e5m2_k, f32_k>("sqeuclidean_e5m2_genoa", nk_sqeuclidean_e5m2_genoa);
    dense_<e5m2_k, f32_k>("euclidean_e5m2_genoa", nk_euclidean_e5m2_genoa);
    dense_<e2m3_k, f32_k>("angular_e2m3_genoa", nk_angular_e2m3_genoa);
    dense_<e2m3_k, f32_k>("sqeuclidean_e2m3_genoa", nk_sqeuclidean_e2m3_genoa);
    dense_<e2m3_k, f32_k>("euclidean_e2m3_genoa", nk_euclidean_e2m3_genoa);
    dense_<e3m2_k, f32_k>("angular_e3m2_genoa", nk_angular_e3m2_genoa);
    dense_<e3m2_k, f32_k>("sqeuclidean_e3m2_genoa", nk_sqeuclidean_e3m2_genoa);
    dense_<e3m2_k, f32_k>("euclidean_e3m2_genoa", nk_euclidean_e3m2_genoa);

    curved_<bf16_k, f32_k>("bilinear_bf16_genoa", nk_bilinear_bf16_genoa);
    curved_<bf16_k, f32_k>("mahalanobis_bf16_genoa", nk_mahalanobis_bf16_genoa);
    curved_<bf16c_k, f32c_k>("bilinear_bf16c_genoa", nk_bilinear_bf16c_genoa);

    dots_<bf16_k, f32_k>("dots_packed_bf16_genoa", nk_dots_packed_size_bf16_genoa, nk_dots_pack_bf16_genoa,
                         nk_dots_packed_bf16_genoa);

    dots_<e4m3_k, f32_k>("dots_packed_e4m3_genoa", nk_dots_packed_size_e4m3_genoa, nk_dots_pack_e4m3_genoa,
                         nk_dots_packed_e4m3_genoa);
    dots_<e5m2_k, f32_k>("dots_packed_e5m2_genoa", nk_dots_packed_size_e5m2_genoa, nk_dots_pack_e5m2_genoa,
                         nk_dots_packed_e5m2_genoa);
    dots_<e2m3_k, f32_k>("dots_packed_e2m3_genoa", nk_dots_packed_size_e2m3_genoa, nk_dots_pack_e2m3_genoa,
                         nk_dots_packed_e2m3_genoa);
    dots_<e3m2_k, f32_k>("dots_packed_e3m2_genoa", nk_dots_packed_size_e3m2_genoa, nk_dots_pack_e3m2_genoa,
                         nk_dots_packed_e3m2_genoa);

    dots_symmetric_<bf16_k, f32_k>("dots_symmetric_bf16_genoa", nk_dots_symmetric_bf16_genoa);
    dots_symmetric_<e4m3_k, f32_k>("dots_symmetric_e4m3_genoa", nk_dots_symmetric_e4m3_genoa);
    dots_symmetric_<e5m2_k, f32_k>("dots_symmetric_e5m2_genoa", nk_dots_symmetric_e5m2_genoa);
    dots_symmetric_<e2m3_k, f32_k>("dots_symmetric_e2m3_genoa", nk_dots_symmetric_e2m3_genoa);
    dots_symmetric_<e3m2_k, f32_k>("dots_symmetric_e3m2_genoa", nk_dots_symmetric_e3m2_genoa);

#endif

#if NK_TARGET_SAPPHIRE
    dense_<f16_k, f32_k>("kld_f16_sapphire", nk_kld_f16_sapphire);
    dense_<f16_k, f32_k>("jsd_f16_sapphire", nk_jsd_f16_sapphire);
    dense_<e4m3_k, f32_k>("euclidean_e4m3_sapphire", nk_euclidean_e4m3_sapphire);
    dense_<e4m3_k, f32_k>("sqeuclidean_e4m3_sapphire", nk_sqeuclidean_e4m3_sapphire);

    elementwise_<u8_k, fma_k, f32_k>("each_fma_u8_sapphire", nk_each_fma_u8_sapphire);
    elementwise_<u8_k, wsum_k, f32_k>("each_wsum_u8_sapphire", nk_each_blend_u8_sapphire);
    elementwise_<i8_k, fma_k, f32_k>("each_fma_i8_sapphire", nk_each_fma_i8_sapphire);
    elementwise_<i8_k, wsum_k, f32_k>("each_wsum_i8_sapphire", nk_each_blend_i8_sapphire);

    curved_<f16_k, f32_k>("bilinear_f16_sapphire", nk_bilinear_f16_sapphire);
    curved_<f16_k, f32_k>("mahalanobis_f16_sapphire", nk_mahalanobis_f16_sapphire);
    curved_<f16c_k, f32c_k>("bilinear_f16c_sapphire", nk_bilinear_f16c_sapphire);

    elementwise_<f16_k, unknown_k, f32_k>("sin_f16_sapphire", nk_sin_f16_sapphire);
    elementwise_<f16_k, unknown_k, f32_k>("cos_f16_sapphire", nk_cos_f16_sapphire);
    elementwise_<f16_k, unknown_k, f32_k>("atan_f16_sapphire", nk_atan_f16_sapphire);

    dots_<bf16_k, f32_k>("dots_packed_bf16_sapphire_amx", nk_dots_packed_size_bf16_sapphire_amx,
                         nk_dots_pack_bf16_sapphire_amx, nk_dots_packed_bf16_sapphire_amx);
    dots_<i8_k, i32_k>("dots_packed_i8_sapphire_amx", nk_dots_packed_size_i8_sapphire_amx, nk_dots_pack_i8_sapphire_amx,
                       nk_dots_packed_i8_sapphire_amx);
    dots_<e4m3_k, f32_k>("dots_packed_e4m3_sapphire_amx", nk_dots_packed_size_e4m3_sapphire_amx,
                         nk_dots_pack_e4m3_sapphire_amx, nk_dots_packed_e4m3_sapphire_amx);
    dots_<e5m2_k, f32_k>("dots_packed_e5m2_sapphire_amx", nk_dots_packed_size_e5m2_sapphire_amx,
                         nk_dots_pack_e5m2_sapphire_amx, nk_dots_packed_e5m2_sapphire_amx);

#endif

#if NK_TARGET_TURIN
    sparse_<u16_k, u32_k>("sparse_intersect_u16_turin", nk_sparse_intersect_u16_turin);
    sparse_<u32_k, u32_k>("sparse_intersect_u32_turin", nk_sparse_intersect_u32_turin);
    sparse_<u64_k, u64_k>("sparse_intersect_u64_turin", nk_sparse_intersect_u64_turin);
#endif

#if NK_TARGET_SPACEMIT
    // Binary operations
    dense_<u1_k, u32_k>("hamming_u1_spacemit", nk_hamming_u1_spacemit);
    dense_<u1_k, f32_k>("jaccard_u1_spacemit", nk_jaccard_u1_spacemit);
    dense_<u8_k, u32_k>("hamming_u8_spacemit", nk_hamming_u8_spacemit);
    dense_<u16_k, f32_k>("jaccard_u16_spacemit", nk_jaccard_u16_spacemit);
    dense_<u32_k, f32_k>("jaccard_u32_spacemit", nk_jaccard_u32_spacemit);

    // Dot products
    dense_<i8_k, i32_k>("dot_i8_spacemit", nk_dot_i8_spacemit);
    dense_<u8_k, u32_k>("dot_u8_spacemit", nk_dot_u8_spacemit);
    dense_<f32_k, f32_k>("dot_f32_spacemit", nk_dot_f32_spacemit);
    dense_<f64_k, f64_k>("dot_f64_spacemit", nk_dot_f64_spacemit);

    // Spatial operations
    dense_<f32_k, f32_k>("sqeuclidean_f32_spacemit", nk_sqeuclidean_f32_spacemit);
    dense_<f64_k, f64_k>("sqeuclidean_f64_spacemit", nk_sqeuclidean_f64_spacemit);
    dense_<f32_k, f32_k>("angular_f32_spacemit", nk_angular_f32_spacemit);
    dense_<f64_k, f64_k>("angular_f64_spacemit", nk_angular_f64_spacemit);
#endif

    sparse_<u16_k, u32_k>("sparse_intersect_u16_serial", nk_sparse_intersect_u16_serial);
    sparse_<u32_k, u32_k>("sparse_intersect_u32_serial", nk_sparse_intersect_u32_serial);
    sparse_<u64_k, u64_k>("sparse_intersect_u64_serial", nk_sparse_intersect_u64_serial);

    curved_<f64_k, f64_k>("bilinear_f64_serial", nk_bilinear_f64_serial);
    curved_<f64c_k, f64c_k>("bilinear_f64c_serial", nk_bilinear_f64c_serial);
    curved_<f64_k, f64_k>("mahalanobis_f64_serial", nk_mahalanobis_f64_serial);
    curved_<f32_k, f32_k>("bilinear_f32_serial", nk_bilinear_f32_serial);
    curved_<f32c_k, f32c_k>("bilinear_f32c_serial", nk_bilinear_f32c_serial);
    curved_<f32_k, f32_k>("mahalanobis_f32_serial", nk_mahalanobis_f32_serial);
    curved_<f16_k, f32_k>("bilinear_f16_serial", nk_bilinear_f16_serial);
    curved_<f16c_k, f32c_k>("bilinear_f16c_serial", nk_bilinear_f16c_serial);
    curved_<f16_k, f32_k>("mahalanobis_f16_serial", nk_mahalanobis_f16_serial);
    curved_<bf16_k, f32_k>("bilinear_bf16_serial", nk_bilinear_bf16_serial);
    curved_<bf16c_k, f32c_k>("bilinear_bf16c_serial", nk_bilinear_bf16c_serial);
    curved_<bf16_k, f32_k>("mahalanobis_bf16_serial", nk_mahalanobis_bf16_serial);

    mesh_<f32_k, f32_k>("rmsd_f32_serial", nk_rmsd_f32_serial);
    mesh_<f32_k, f32_k>("kabsch_f32_serial", nk_kabsch_f32_serial);
    mesh_<f32_k, f32_k>("umeyama_f32_serial", nk_umeyama_f32_serial);
    mesh_<f64_k, f64_k>("rmsd_f64_serial", nk_rmsd_f64_serial);
    mesh_<f64_k, f64_k>("kabsch_f64_serial", nk_kabsch_f64_serial);
    mesh_<f64_k, f64_k>("umeyama_f64_serial", nk_umeyama_f64_serial);

    dense_<bf16_k, f32_k>("dot_bf16_serial", nk_dot_bf16_serial);
    dense_<bf16_k, f32_k>("angular_bf16_serial", nk_angular_bf16_serial);
    dense_<bf16_k, f32_k>("sqeuclidean_bf16_serial", nk_sqeuclidean_bf16_serial);
    dense_<bf16_k, f32_k>("euclidean_bf16_serial", nk_euclidean_bf16_serial);
    dense_<bf16_k, f32_k>("kld_bf16_serial", nk_kld_bf16_serial);
    dense_<bf16_k, f32_k>("jsd_bf16_serial", nk_jsd_bf16_serial);

    dense_<e4m3_k, f32_k>("dot_e4m3_serial", nk_dot_e4m3_serial);
    dense_<e5m2_k, f32_k>("dot_e5m2_serial", nk_dot_e5m2_serial);
    dense_<e4m3_k, f32_k>("angular_e4m3_serial", nk_angular_e4m3_serial);
    dense_<e4m3_k, f32_k>("sqeuclidean_e4m3_serial", nk_sqeuclidean_e4m3_serial);
    dense_<e4m3_k, f32_k>("euclidean_e4m3_serial", nk_euclidean_e4m3_serial);
    dense_<e5m2_k, f32_k>("angular_e5m2_serial", nk_angular_e5m2_serial);
    dense_<e5m2_k, f32_k>("sqeuclidean_e5m2_serial", nk_sqeuclidean_e5m2_serial);
    dense_<e5m2_k, f32_k>("euclidean_e5m2_serial", nk_euclidean_e5m2_serial);
    dense_<e2m3_k, f32_k>("dot_e2m3_serial", nk_dot_e2m3_serial);
    dense_<e2m3_k, f32_k>("angular_e2m3_serial", nk_angular_e2m3_serial);
    dense_<e2m3_k, f32_k>("sqeuclidean_e2m3_serial", nk_sqeuclidean_e2m3_serial);
    dense_<e2m3_k, f32_k>("euclidean_e2m3_serial", nk_euclidean_e2m3_serial);
    dense_<e3m2_k, f32_k>("dot_e3m2_serial", nk_dot_e3m2_serial);
    dense_<e3m2_k, f32_k>("angular_e3m2_serial", nk_angular_e3m2_serial);
    dense_<e3m2_k, f32_k>("sqeuclidean_e3m2_serial", nk_sqeuclidean_e3m2_serial);
    dense_<e3m2_k, f32_k>("euclidean_e3m2_serial", nk_euclidean_e3m2_serial);

    dense_<f16_k, f32_k>("dot_f16_serial", nk_dot_f16_serial);
    dense_<f16_k, f32_k>("angular_f16_serial", nk_angular_f16_serial);
    dense_<f16_k, f32_k>("sqeuclidean_f16_serial", nk_sqeuclidean_f16_serial);
    dense_<f16_k, f32_k>("euclidean_f16_serial", nk_euclidean_f16_serial);
    dense_<f16_k, f32_k>("kld_f16_serial", nk_kld_f16_serial);
    dense_<f16_k, f32_k>("jsd_f16_serial", nk_jsd_f16_serial);

    dense_<f32_k, f32_k>("dot_f32_serial", nk_dot_f32_serial);
    dense_<f32_k, f32_k>("angular_f32_serial", nk_angular_f32_serial);
    dense_<f32_k, f32_k>("sqeuclidean_f32_serial", nk_sqeuclidean_f32_serial);
    dense_<f32_k, f32_k>("euclidean_f32_serial", nk_euclidean_f32_serial);
    dense_<f32_k, f32_k>("kld_f32_serial", nk_kld_f32_serial);
    dense_<f32_k, f32_k>("jsd_f32_serial", nk_jsd_f32_serial);

    dense_<f64_k, f64_k>("dot_f64_serial", nk_dot_f64_serial);
    dense_<f64_k, f64_k>("angular_f64_serial", nk_angular_f64_serial);
    dense_<f64_k, f64_k>("sqeuclidean_f64_serial", nk_sqeuclidean_f64_serial);
    dense_<f64_k, f64_k>("euclidean_f64_serial", nk_euclidean_f64_serial);

    dense_<i8_k, f32_k>("angular_i8_serial", nk_angular_i8_serial);
    dense_<i8_k, u32_k>("sqeuclidean_i8_serial", nk_sqeuclidean_i8_serial);
    dense_<i8_k, f32_k>("euclidean_i8_serial", nk_euclidean_i8_serial);
    dense_<i8_k, i32_k>("dot_i8_serial", nk_dot_i8_serial);

    dense_<u8_k, f32_k>("angular_u8_serial", nk_angular_u8_serial);
    dense_<u8_k, u32_k>("sqeuclidean_u8_serial", nk_sqeuclidean_u8_serial);
    dense_<u8_k, f32_k>("euclidean_u8_serial", nk_euclidean_u8_serial);
    dense_<u8_k, u32_k>("dot_u8_serial", nk_dot_u8_serial);

    dense_<i4_k, f32_k>("angular_i4_serial", nk_angular_i4_serial);
    dense_<i4_k, u32_k>("sqeuclidean_i4_serial", nk_sqeuclidean_i4_serial);
    dense_<i4_k, f32_k>("euclidean_i4_serial", nk_euclidean_i4_serial);
    dense_<i4_k, i32_k>("dot_i4_serial", nk_dot_i4_serial);

    dense_<u4_k, f32_k>("angular_u4_serial", nk_angular_u4_serial);
    dense_<u4_k, u32_k>("sqeuclidean_u4_serial", nk_sqeuclidean_u4_serial);
    dense_<u4_k, f32_k>("euclidean_u4_serial", nk_euclidean_u4_serial);
    dense_<u4_k, u32_k>("dot_u4_serial", nk_dot_u4_serial);

    dense_<f64c_k, f64c_k>("dot_f64c_serial", nk_dot_f64c_serial);
    dense_<f32c_k, f32c_k>("dot_f32c_serial", nk_dot_f32c_serial);
    dense_<f16c_k, f32c_k>("dot_f16c_serial", nk_dot_f16c_serial);
    dense_<bf16c_k, f32c_k>("dot_bf16c_serial", nk_dot_bf16c_serial);
    dense_<f64c_k, f64c_k>("vdot_f64c_serial", nk_vdot_f64c_serial);
    dense_<f32c_k, f32c_k>("vdot_f32c_serial", nk_vdot_f32c_serial);
    dense_<f16c_k, f32c_k>("vdot_f16c_serial", nk_vdot_f16c_serial);
    dense_<bf16c_k, f32c_k>("vdot_bf16c_serial", nk_vdot_bf16c_serial);

    dense_<u1_k, u32_k>("hamming_u1_serial", nk_hamming_u1_serial);
    dense_<u1_k, f32_k>("jaccard_u1_serial", nk_jaccard_u1_serial);

    elementwise_<f32_k, unknown_k, f32_k>("sin_f32_stl", elementwise_with_stl<nk_f32_t, sin_with_stl<nk_f32_t>>);
    elementwise_<f32_k, unknown_k, f32_k>("cos_f32_stl", elementwise_with_stl<nk_f32_t, cos_with_stl<nk_f32_t>>);
    elementwise_<f32_k, unknown_k, f32_k>("atan_f32_stl", elementwise_with_stl<nk_f32_t, atan_with_stl<nk_f32_t>>);
    elementwise_<f32_k, unknown_k, f32_k>("sin_f32_serial", nk_sin_f32_serial);
    elementwise_<f32_k, unknown_k, f32_k>("cos_f32_serial", nk_cos_f32_serial);
    elementwise_<f32_k, unknown_k, f32_k>("atan_f32_serial", nk_atan_f32_serial);
    elementwise_<f64_k, unknown_k, f64_k>("sin_f64_stl", elementwise_with_stl<nk_f64_t, sin_with_stl<nk_f64_t>>);
    elementwise_<f64_k, unknown_k, f64_k>("cos_f64_stl", elementwise_with_stl<nk_f64_t, cos_with_stl<nk_f64_t>>);
    elementwise_<f64_k, unknown_k, f64_k>("atan_f64_stl", elementwise_with_stl<nk_f64_t, atan_with_stl<nk_f64_t>>);
    elementwise_<f64_k, unknown_k, f64_k>("sin_f64_serial", nk_sin_f64_serial);
    elementwise_<f64_k, unknown_k, f64_k>("cos_f64_serial", nk_cos_f64_serial);
    elementwise_<f64_k, unknown_k, f64_k>("atan_f64_serial", nk_atan_f64_serial);

    elementwise_<f16_k, unknown_k, f32_k>("sin_f16_serial", nk_sin_f16_serial);
    elementwise_<f16_k, unknown_k, f32_k>("cos_f16_serial", nk_cos_f16_serial);
    elementwise_<f16_k, unknown_k, f32_k>("atan_f16_serial", nk_atan_f16_serial);

    elementwise_<f16_k, fma_k, f32_k>("each_fma_f16_serial", nk_each_fma_f16_serial);
    elementwise_<f16_k, wsum_k, f32_k>("each_wsum_f16_serial", nk_each_blend_f16_serial);
    elementwise_<u8_k, fma_k, f32_k>("each_fma_u8_serial", nk_each_fma_u8_serial);
    elementwise_<u8_k, wsum_k, f32_k>("each_wsum_u8_serial", nk_each_blend_u8_serial);
    elementwise_<i8_k, fma_k, f32_k>("each_fma_i8_serial", nk_each_fma_i8_serial);
    elementwise_<i8_k, wsum_k, f32_k>("each_wsum_i8_serial", nk_each_blend_i8_serial);

    geospatial_<f32_k, f32_k>("haversine_f32_serial", nk_haversine_f32_serial);
    geospatial_<f64_k, f64_k>("haversine_f64_serial", nk_haversine_f64_serial);
    geospatial_<f32_k, f32_k>("vincenty_f32_serial", nk_vincenty_f32_serial);
    geospatial_<f64_k, f64_k>("vincenty_f64_serial", nk_vincenty_f64_serial);

    dots_<bf16_k, f32_k>("dots_packed_bf16_serial", nk_dots_packed_size_bf16_serial, nk_dots_pack_bf16_serial,
                         nk_dots_packed_bf16_serial);
    dots_<i8_k, i32_k>("dots_packed_i8_serial", nk_dots_packed_size_i8_serial, nk_dots_pack_i8_serial,
                       nk_dots_packed_i8_serial);
    dots_<f32_k, f32_k>("dots_packed_f32_serial", nk_dots_packed_size_f32_serial, nk_dots_pack_f32_serial,
                        nk_dots_packed_f32_serial);
    dots_<u4_k, u32_k>("dots_packed_u4_serial", nk_dots_packed_size_u4_serial, nk_dots_pack_u4_serial,
                       nk_dots_packed_u4_serial);
    dots_<i4_k, i32_k>("dots_packed_i4_serial", nk_dots_packed_size_i4_serial, nk_dots_pack_i4_serial,
                       nk_dots_packed_i4_serial);
    dots_<e4m3_k, f32_k>("dots_packed_e4m3_serial", nk_dots_packed_size_e4m3_serial, nk_dots_pack_e4m3_serial,
                         nk_dots_packed_e4m3_serial);
    dots_<e5m2_k, f32_k>("dots_packed_e5m2_serial", nk_dots_packed_size_e5m2_serial, nk_dots_pack_e5m2_serial,
                         nk_dots_packed_e5m2_serial);
    dots_<e2m3_k, f32_k>("dots_packed_e2m3_serial", nk_dots_packed_size_e2m3_serial, nk_dots_pack_e2m3_serial,
                         nk_dots_packed_e2m3_serial);
    dots_<e3m2_k, f32_k>("dots_packed_e3m2_serial", nk_dots_packed_size_e3m2_serial, nk_dots_pack_e3m2_serial,
                         nk_dots_packed_e3m2_serial);

    // Symmetric GEMM benchmarks (A × Aᵀ)
    dots_symmetric_<f32_k, f32_k>("dots_symmetric_f32_serial", nk_dots_symmetric_f32_serial);
    dots_symmetric_<f64_k, f64_k>("dots_symmetric_f64_serial", nk_dots_symmetric_f64_serial);
    dots_symmetric_<bf16_k, f32_k>("dots_symmetric_bf16_serial", nk_dots_symmetric_bf16_serial);
    dots_symmetric_<f16_k, f32_k>("dots_symmetric_f16_serial", nk_dots_symmetric_f16_serial);
    dots_symmetric_<i8_k, i32_k>("dots_symmetric_i8_serial", nk_dots_symmetric_i8_serial);
    dots_symmetric_<u8_k, u32_k>("dots_symmetric_u8_serial", nk_dots_symmetric_u8_serial);
    dots_symmetric_<i4_k, i32_k>("dots_symmetric_i4_serial", nk_dots_symmetric_i4_serial);
    dots_symmetric_<u4_k, u32_k>("dots_symmetric_u4_serial", nk_dots_symmetric_u4_serial);
    dots_symmetric_<e4m3_k, f32_k>("dots_symmetric_e4m3_serial", nk_dots_symmetric_e4m3_serial);
    dots_symmetric_<e5m2_k, f32_k>("dots_symmetric_e5m2_serial", nk_dots_symmetric_e5m2_serial);
    dots_symmetric_<e2m3_k, f32_k>("dots_symmetric_e2m3_serial", nk_dots_symmetric_e2m3_serial);
    dots_symmetric_<e3m2_k, f32_k>("dots_symmetric_e3m2_serial", nk_dots_symmetric_e3m2_serial);

    // Hamming distances - packed
    hammings_<nk_u1_k, nk_u32_k>("hammings_u1_serial", nk_hammings_packed_size_u1_serial, nk_hammings_pack_u1_serial,
                                 nk_hammings_packed_u1_serial);

    // Hamming distances - symmetric matrix
    hammings_symmetric_<nk_u1_k, nk_u32_k>("hammings_symmetric_u1_serial", nk_hammings_symmetric_u1_serial);

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

#if NK_TARGET_SAPPHIRE
    cast_<nk_f32_k, nk_f16_k>("cast_f32_to_f16_sapphire", nk_cast_sapphire);
    cast_<nk_f16_k, nk_f32_k>("cast_f16_to_f32_sapphire", nk_cast_sapphire);
#endif

    bm::RunSpecifiedBenchmarks();
    bm::Shutdown();
    return 0;
}

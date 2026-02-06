/**
 *  @brief NumKong C++ Benchmark Suite using Google Benchmark - Header file.
 *  @file bench/bench.hpp
 *  @author Ash Vardanian
 *  @date March 14, 2023
 *
 *  Comprehensive benchmarks for NumKong SIMD-optimized functions measuring
 *  throughput performance. This header contains measurement infrastructure
 *  and helper templates used across multiple benchmark files.
 */

#pragma once

#include <cmath>       // `std::sqrt`
#include <random>      // `std::uniform_int_distribution`
#include <type_traits> // `std::conditional_t`
#include <vector>      // `std::vector`
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
// - cblas_gemm_bf16bf16f32: BF16 inputs -> F32 output
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

#include "numkong/types.hpp"
#include "numkong/tensor.hpp"
#include "numkong/random.hpp"

namespace bm = benchmark;
namespace nk = ashvardanian::numkong;

constexpr std::size_t default_seconds = 10;
constexpr std::size_t default_threads = 1;

/// Vector dimension for dot products and spatial metrics
/// Can be overridden at runtime via `NK_DENSE_DIMENSIONS` environment variable
extern std::size_t dense_dimensions;
/// Has quadratic impact on the number of operations
/// Can be overridden at runtime via `NK_CURVED_DIMENSIONS` environment variable
extern std::size_t curved_dimensions;
/// Number of 3D points for mesh metrics (RMSD, Kabsch)
/// Can be overridden at runtime via `NK_MESH_POINTS` environment variable
extern std::size_t mesh_points;
/// Matrix multiplication benchmark globals
/// Can be overridden at runtime via `NK_MATRIX_HEIGHT/WIDTH/DEPTH` environment variables
extern std::size_t matrix_height, matrix_width, matrix_depth;
/// Random seed for reproducible benchmarks
/// Can be overridden at runtime via `NK_SEED` environment variable
extern std::uint32_t random_seed;
/// Sparse set intersection benchmark globals
/// Can be overridden at runtime via `NK_SPARSE_*` environment variables
extern std::size_t sparse_first_length;
extern std::size_t sparse_second_length;
extern double sparse_intersection_share;

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
 *  to store a matrix of logical dimensions rows x cols.
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
 *
 *  Used by: bench_dot.cpp, bench_spatial.cpp, bench_set.cpp, bench_probability.cpp
 *
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

template <nk_dtype_t input_dtype_, nk_dtype_t output_dtype_, typename kernel_type_ = void>
void dense_(std::string name, kernel_type_ *kernel) {
    std::string bench_name = name + "<" + std::to_string(dense_dimensions) + "d>";
    bm::RegisterBenchmark(bench_name.c_str(), measure_dense<input_dtype_, output_dtype_, kernel_type_ *>, kernel,
                          dense_dimensions)
        ->MinTime(default_seconds)
        ->Threads(default_threads);
}

//  Batched dot products measurement for packed B matrix API
//  Used by: all bench_cross_*.cpp files
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
    nk_size_t b_stride_bytes = values_per_row * sizeof(typename input_t::raw_t); // B is n x k, so k columns per row

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

    // Allocate matrix A (n vectors x k dimensions) and result matrix C (n x n)
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

    // Symmetric operations compute upper triangle: N x (N+1)/2 dot products x K multiply-adds x 2 scalar-ops = N x
    // (N+1) x K total scalar-ops
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
 *  Used by: all bench_cross_*.cpp files
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

    // Allocate matrix A (n vectors x k bits) and result matrix C (n x n)
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

    // Symmetric operations compute upper triangle: N x (N+1)/2 comparisons x K bits
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

// Forward declarations for benchmark registration functions (defined in separate files)
void bench_dot();
void bench_spatial();
void bench_set();
void bench_curved();
void bench_probability();
void bench_each();
void bench_trigonometry();
void bench_geospatial();
void bench_mesh();
void bench_sparse();
void bench_cast();
void bench_reduce();

// Forward declarations for cross/batch operations (ISA-family files)
void bench_cross_serial();
void bench_cross_x86();
void bench_cross_amx();
void bench_cross_arm();
void bench_cross_sme();
void bench_cross_blas();

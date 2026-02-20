/**
 *  @brief Elementwise operations benchmarks (fma, wsum, sum, scale).
 *  @file bench/bench_each.cpp
 *  @author Ash Vardanian
 *  @date March 14, 2023
 */

#include <cstring> // std::memset

#include "numkong/capabilities.h" // nk_kernel_kind_t
#include "numkong/each.h"

#include "bench.hpp"

#if NK_COMPARE_TO_BLAS || NK_COMPARE_TO_MKL || NK_COMPARE_TO_ACCELERATE

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

#endif // NK_COMPARE_TO_BLAS || NK_COMPARE_TO_MKL || NK_COMPARE_TO_ACCELERATE

/**
 *  @brief Measures the performance of elementwise operations (sum, wsum, fma, scale) using Google Benchmark.
 *  @param state The benchmark state object provided by Google Benchmark.
 *  @param kernel The kernel function to benchmark.
 *  @param dimensions The number of dimensions in the vectors.
 */
template <nk_dtype_t input_dtype_, nk_kernel_kind_t kernel_kind_, nk_dtype_t alpha_dtype_, typename kernel_type_ = void>
void measure_each(bm::State &state, kernel_type_ kernel, std::size_t dimensions) {

    using input_t = typename nk::type_for<input_dtype_>::type;
    using alpha_t = typename nk::type_for<alpha_dtype_>::type;
    using input_vector_t = nk::vector<input_t>;

    // Scaling parameters for FMA/wsum/scale kernels
    alpha_t alpha = alpha_t(0.2f);
    alpha_t beta = alpha_t(0.3f);

    // Preallocate vectors for different kernel types:
    // - sum: input_a, input_c -> output
    // - wsum: input_a, input_c + alpha, beta -> output
    // - fma: input_a, input_b, input_c + alpha, beta -> output
    // - scale: input_a + alpha, beta -> output
    std::size_t bytes_per_set = bench_dtype_bytes(input_dtype_, 4 * dimensions);
    std::size_t const vectors_count = bench_input_count(bytes_per_set);
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

template <nk_dtype_t input_dtype_, nk_kernel_kind_t kernel_kind_ = nk_kernel_unknown_k,
          nk_dtype_t alpha_dtype_ = nk_dtype_unknown_k, typename kernel_type_ = void>
void each_(std::string name, kernel_type_ *kernel) {
    std::string bench_name = name + "<" + std::to_string(dense_dimensions) + "d>";
    bm::RegisterBenchmark(bench_name.c_str(), measure_each<input_dtype_, kernel_kind_, alpha_dtype_, kernel_type_ *>,
                          kernel, dense_dimensions);
}

void bench_each() {
    constexpr nk_dtype_t i8_k = nk_i8_k;
    constexpr nk_dtype_t u8_k = nk_u8_k;
    constexpr nk_dtype_t i16_k = nk_i16_k;
    constexpr nk_dtype_t u16_k = nk_u16_k;
    constexpr nk_dtype_t f64_k = nk_f64_k;
    constexpr nk_dtype_t f32_k = nk_f32_k;
    constexpr nk_dtype_t f16_k = nk_f16_k;
    constexpr nk_dtype_t bf16_k = nk_bf16_k;

    constexpr nk_kernel_kind_t fma_k = nk_kernel_each_fma_k;
    constexpr nk_kernel_kind_t wsum_k = nk_kernel_each_blend_k;
    constexpr nk_kernel_kind_t sum_k = nk_kernel_each_sum_k;
    constexpr nk_kernel_kind_t scale_k = nk_kernel_each_scale_k;

#if NK_COMPARE_TO_BLAS || NK_COMPARE_TO_MKL || NK_COMPARE_TO_ACCELERATE
    each_<f32_k, sum_k, f32_k>("sum_f32_with_blas", sum_f32_with_blas);
    each_<f32_k, wsum_k, f32_k>("each_wsum_f32_with_blas", wsum_f32_with_blas);
    each_<f64_k, sum_k, f64_k>("sum_f64_with_blas", sum_f64_with_blas);
    each_<f64_k, wsum_k, f64_k>("each_wsum_f64_with_blas", wsum_f64_with_blas);
#endif

#if NK_TARGET_NEON
    each_<f32_k, fma_k, f32_k>("each_fma_f32_neon", nk_each_fma_f32_neon);
    each_<f32_k, wsum_k, f32_k>("each_wsum_f32_neon", nk_each_blend_f32_neon);
    each_<f32_k, fma_k, f32_k>("each_fma_f32_serial", nk_each_fma_f32_serial);
    each_<f32_k, wsum_k, f32_k>("each_wsum_f32_serial", nk_each_blend_f32_serial);
#endif

#if NK_TARGET_NEONHALF
    each_<f16_k, fma_k, f32_k>("each_fma_f16_neonhalf", nk_each_fma_f16_neonhalf);
    each_<f16_k, wsum_k, f32_k>("each_wsum_f16_neonhalf", nk_each_blend_f16_neonhalf);
    each_<u8_k, fma_k, f32_k>("each_fma_u8_neonhalf", nk_each_fma_u8_neonhalf);
    each_<u8_k, wsum_k, f32_k>("each_wsum_u8_neonhalf", nk_each_blend_u8_neonhalf);
    each_<i8_k, fma_k, f32_k>("each_fma_i8_neonhalf", nk_each_fma_i8_neonhalf);
    each_<i8_k, wsum_k, f32_k>("each_wsum_i8_neonhalf", nk_each_blend_i8_neonhalf);
#endif

#if NK_TARGET_NEONBFDOT
    each_<bf16_k, fma_k, f32_k>("each_fma_bf16_neonbfdot", nk_each_fma_bf16_neonbfdot);
    each_<bf16_k, wsum_k, f32_k>("each_wsum_bf16_neonbfdot", nk_each_blend_bf16_neonbfdot);
#endif

#if NK_TARGET_HASWELL
    each_<f64_k, scale_k, f64_k>("each_scale_f64_haswell", nk_each_scale_f64_haswell);
    each_<f64_k, fma_k, f64_k>("each_fma_f64_haswell", nk_each_fma_f64_haswell);
    each_<f64_k, wsum_k, f64_k>("each_wsum_f64_haswell", nk_each_blend_f64_haswell);
    each_<f32_k, scale_k, f32_k>("each_scale_f32_haswell", nk_each_scale_f32_haswell);
    each_<f32_k, fma_k, f32_k>("each_fma_f32_haswell", nk_each_fma_f32_haswell);
    each_<f32_k, wsum_k, f32_k>("each_wsum_f32_haswell", nk_each_blend_f32_haswell);
    each_<f16_k, scale_k, f32_k>("each_scale_f16_haswell", nk_each_scale_f16_haswell);
    each_<f16_k, fma_k, f32_k>("each_fma_f16_haswell", nk_each_fma_f16_haswell);
    each_<f16_k, wsum_k, f32_k>("each_wsum_f16_haswell", nk_each_blend_f16_haswell);
    each_<bf16_k, scale_k, f32_k>("each_scale_bf16_haswell", nk_each_scale_bf16_haswell);
    each_<bf16_k, fma_k, f32_k>("each_fma_bf16_haswell", nk_each_fma_bf16_haswell);
    each_<bf16_k, wsum_k, f32_k>("each_wsum_bf16_haswell", nk_each_blend_bf16_haswell);
    each_<i8_k, scale_k, f32_k>("each_scale_i8_haswell", nk_each_scale_i8_haswell);
    each_<i8_k, fma_k, f32_k>("each_fma_i8_haswell", nk_each_fma_i8_haswell);
    each_<i8_k, wsum_k, f32_k>("each_wsum_i8_haswell", nk_each_blend_i8_haswell);
    each_<u8_k, scale_k, f32_k>("each_scale_u8_haswell", nk_each_scale_u8_haswell);
    each_<u8_k, fma_k, f32_k>("each_fma_u8_haswell", nk_each_fma_u8_haswell);
    each_<u8_k, wsum_k, f32_k>("each_wsum_u8_haswell", nk_each_blend_u8_haswell);
    each_<i16_k, scale_k, f32_k>("each_scale_i16_haswell", nk_each_scale_i16_haswell);
    each_<i16_k, fma_k, f32_k>("each_fma_i16_haswell", nk_each_fma_i16_haswell);
    each_<u16_k, scale_k, f32_k>("each_scale_u16_haswell", nk_each_scale_u16_haswell);
    each_<u16_k, fma_k, f32_k>("each_fma_u16_haswell", nk_each_fma_u16_haswell);
#endif

#if NK_TARGET_SKYLAKE
    each_<f64_k, fma_k, f64_k>("each_fma_f64_skylake", nk_each_fma_f64_skylake);
    each_<f64_k, wsum_k, f64_k>("each_wsum_f64_skylake", nk_each_blend_f64_skylake);
    each_<f32_k, fma_k, f32_k>("each_fma_f32_skylake", nk_each_fma_f32_skylake);
    each_<f32_k, wsum_k, f32_k>("each_wsum_f32_skylake", nk_each_blend_f32_skylake);
    each_<bf16_k, fma_k, f32_k>("each_fma_bf16_skylake", nk_each_fma_bf16_skylake);
    each_<bf16_k, wsum_k, f32_k>("each_wsum_bf16_skylake", nk_each_blend_bf16_skylake);
#endif

#if NK_TARGET_SAPPHIRE
    each_<u8_k, fma_k, f32_k>("each_fma_u8_sapphire", nk_each_fma_u8_sapphire);
    each_<u8_k, wsum_k, f32_k>("each_wsum_u8_sapphire", nk_each_blend_u8_sapphire);
    each_<i8_k, fma_k, f32_k>("each_fma_i8_sapphire", nk_each_fma_i8_sapphire);
    each_<i8_k, wsum_k, f32_k>("each_wsum_i8_sapphire", nk_each_blend_i8_sapphire);
#endif

    // Serial fallbacks
    each_<f16_k, fma_k, f32_k>("each_fma_f16_serial", nk_each_fma_f16_serial);
    each_<f16_k, wsum_k, f32_k>("each_wsum_f16_serial", nk_each_blend_f16_serial);
    each_<u8_k, fma_k, f32_k>("each_fma_u8_serial", nk_each_fma_u8_serial);
    each_<u8_k, wsum_k, f32_k>("each_wsum_u8_serial", nk_each_blend_u8_serial);
    each_<i8_k, fma_k, f32_k>("each_fma_i8_serial", nk_each_fma_i8_serial);
    each_<i8_k, wsum_k, f32_k>("each_wsum_i8_serial", nk_each_blend_i8_serial);
}

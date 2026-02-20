/**
 *  @brief Trigonometry benchmarks (sin, cos, atan).
 *  @file bench/bench_trigonometry.cpp
 *  @author Ash Vardanian
 *  @date March 14, 2023
 */

#include "numkong/capabilities.h" // nk_kernel_kind_t
#include "numkong/trigonometry.h"

#include "bench.hpp"

// STL baseline wrappers for trigonometric operations
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

/**
 *  @brief Measures the performance of trigonometric operations (sin, cos, atan) using Google Benchmark.
 *  @param state The benchmark state object provided by Google Benchmark.
 *  @param kernel The kernel function to benchmark.
 *  @param dimensions The number of dimensions in the vectors.
 */
template <nk_dtype_t input_dtype_, nk_kernel_kind_t kernel_kind_, nk_dtype_t alpha_dtype_, typename kernel_type_ = void>
void measure_trigonometry(bm::State &state, kernel_type_ kernel, std::size_t dimensions) {

    using input_t = typename nk::type_for<input_dtype_>::type;
    using input_vector_t = nk::vector<input_t>;

    // Preallocate vectors for trigonometric kernels (unary: input + output)
    std::size_t bytes_per_set = bench_dtype_bytes(input_dtype_, 2 * dimensions);
    std::size_t const vectors_count = bench_input_count(bytes_per_set);
    std::vector<input_vector_t> input_a(vectors_count), output(vectors_count);
    auto generator = make_random_engine();
    for (std::size_t index = 0; index != vectors_count; ++index) {
        input_a[index] = make_vector<input_t>(dimensions);
        output[index] = make_vector<input_t>(dimensions);
        nk::fill_uniform(generator, input_a[index].values_data(), dimensions);
    }

    // Benchmark loop
    std::size_t iterations = 0;
    for (auto _ : state) {
        std::size_t const index = iterations & (vectors_count - 1);
        kernel(input_a[index].raw_values_data(), dimensions, output[index].raw_values_data());
        bm::ClobberMemory();
        iterations++;
    }

    state.counters["bytes"] = bm::Counter(iterations * input_a[0].size_bytes(), bm::Counter::kIsRate);
    state.counters["calls"] = bm::Counter(iterations, bm::Counter::kIsRate);
}

template <nk_dtype_t input_dtype_, nk_kernel_kind_t kernel_kind_ = nk_kernel_unknown_k,
          nk_dtype_t alpha_dtype_ = nk_dtype_unknown_k, typename kernel_type_ = void>
void trigonometry_(std::string name, kernel_type_ *kernel) {
    std::string bench_name = name + "<" + std::to_string(dense_dimensions) + "d>";
    bm::RegisterBenchmark(bench_name.c_str(),
                          measure_trigonometry<input_dtype_, kernel_kind_, alpha_dtype_, kernel_type_ *>, kernel,
                          dense_dimensions);
}

void bench_trigonometry() {
    constexpr nk_dtype_t f64_k = nk_f64_k;
    constexpr nk_dtype_t f32_k = nk_f32_k;
    constexpr nk_dtype_t f16_k = nk_f16_k;

    constexpr nk_kernel_kind_t unknown_k = nk_kernel_unknown_k;

#if NK_TARGET_NEON
    trigonometry_<f32_k, unknown_k, f32_k>("sin_f32_neon", nk_each_sin_f32_neon);
    trigonometry_<f32_k, unknown_k, f32_k>("cos_f32_neon", nk_each_cos_f32_neon);
    trigonometry_<f32_k, unknown_k, f32_k>("atan_f32_neon", nk_each_atan_f32_neon);
    trigonometry_<f64_k, unknown_k, f64_k>("sin_f64_neon", nk_each_sin_f64_neon);
    trigonometry_<f64_k, unknown_k, f64_k>("cos_f64_neon", nk_each_cos_f64_neon);
    trigonometry_<f64_k, unknown_k, f64_k>("atan_f64_neon", nk_each_atan_f64_neon);
#endif

#if NK_TARGET_HASWELL
    trigonometry_<f32_k, unknown_k, f32_k>("each_sin_f32_haswell", nk_each_sin_f32_haswell);
    trigonometry_<f32_k, unknown_k, f32_k>("each_cos_f32_haswell", nk_each_cos_f32_haswell);
    trigonometry_<f32_k, unknown_k, f32_k>("each_atan_f32_haswell", nk_each_atan_f32_haswell);
    trigonometry_<f64_k, unknown_k, f64_k>("each_sin_f64_haswell", nk_each_sin_f64_haswell);
    trigonometry_<f64_k, unknown_k, f64_k>("each_cos_f64_haswell", nk_each_cos_f64_haswell);
    trigonometry_<f64_k, unknown_k, f64_k>("each_atan_f64_haswell", nk_each_atan_f64_haswell);
#endif

#if NK_TARGET_SKYLAKE
    trigonometry_<f32_k, unknown_k, f32_k>("each_sin_f32_skylake", nk_each_sin_f32_skylake);
    trigonometry_<f32_k, unknown_k, f32_k>("each_cos_f32_skylake", nk_each_cos_f32_skylake);
    trigonometry_<f32_k, unknown_k, f32_k>("each_atan_f32_skylake", nk_each_atan_f32_skylake);
    trigonometry_<f64_k, unknown_k, f64_k>("each_sin_f64_skylake", nk_each_sin_f64_skylake);
    trigonometry_<f64_k, unknown_k, f64_k>("each_cos_f64_skylake", nk_each_cos_f64_skylake);
    trigonometry_<f64_k, unknown_k, f64_k>("each_atan_f64_skylake", nk_each_atan_f64_skylake);
#endif

#if NK_TARGET_SAPPHIRE
    trigonometry_<f16_k, unknown_k, f32_k>("each_sin_f16_sapphire", nk_each_sin_f16_sapphire);
    trigonometry_<f16_k, unknown_k, f32_k>("each_cos_f16_sapphire", nk_each_cos_f16_sapphire);
    trigonometry_<f16_k, unknown_k, f32_k>("each_atan_f16_sapphire", nk_each_atan_f16_sapphire);
#endif

    // STL baselines
    trigonometry_<f32_k, unknown_k, f32_k>("each_sin_f32_stl", elementwise_with_stl<nk_f32_t, sin_with_stl<nk_f32_t>>);
    trigonometry_<f32_k, unknown_k, f32_k>("each_cos_f32_stl", elementwise_with_stl<nk_f32_t, cos_with_stl<nk_f32_t>>);
    trigonometry_<f32_k, unknown_k, f32_k>("each_atan_f32_stl",
                                           elementwise_with_stl<nk_f32_t, atan_with_stl<nk_f32_t>>);
    trigonometry_<f64_k, unknown_k, f64_k>("each_sin_f64_stl", elementwise_with_stl<nk_f64_t, sin_with_stl<nk_f64_t>>);
    trigonometry_<f64_k, unknown_k, f64_k>("each_cos_f64_stl", elementwise_with_stl<nk_f64_t, cos_with_stl<nk_f64_t>>);
    trigonometry_<f64_k, unknown_k, f64_k>("each_atan_f64_stl",
                                           elementwise_with_stl<nk_f64_t, atan_with_stl<nk_f64_t>>);

    // Serial fallbacks
    trigonometry_<f32_k, unknown_k, f32_k>("each_sin_f32_serial", nk_each_sin_f32_serial);
    trigonometry_<f32_k, unknown_k, f32_k>("each_cos_f32_serial", nk_each_cos_f32_serial);
    trigonometry_<f32_k, unknown_k, f32_k>("each_atan_f32_serial", nk_each_atan_f32_serial);
    trigonometry_<f64_k, unknown_k, f64_k>("each_sin_f64_serial", nk_each_sin_f64_serial);
    trigonometry_<f64_k, unknown_k, f64_k>("each_cos_f64_serial", nk_each_cos_f64_serial);
    trigonometry_<f64_k, unknown_k, f64_k>("each_atan_f64_serial", nk_each_atan_f64_serial);
    trigonometry_<f16_k, unknown_k, f32_k>("each_sin_f16_serial", nk_each_sin_f16_serial);
    trigonometry_<f16_k, unknown_k, f32_k>("each_cos_f16_serial", nk_each_cos_f16_serial);
    trigonometry_<f16_k, unknown_k, f32_k>("each_atan_f16_serial", nk_each_atan_f16_serial);
}

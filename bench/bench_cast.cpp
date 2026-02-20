/**
 *  @brief Type casting benchmarks.
 *  @file bench/bench_cast.cpp
 *  @author Ash Vardanian
 *  @date March 14, 2023
 */

#include "numkong/cast.h"

#include "bench.hpp"

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
    std::string bench_name = name + "<" + std::to_string(bench_config.dense_dimensions) + "d>";
    bm::RegisterBenchmark(bench_name.c_str(), measure_cast<input_dtype_, output_dtype_>, kernel,
                          bench_config.dense_dimensions);
}

void bench_cast() {

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

#if NK_TARGET_ICELAKE
    cast_<nk_f32_k, nk_f16_k>("cast_f32_to_f16_icelake", nk_cast_icelake);
    cast_<nk_f16_k, nk_f32_k>("cast_f16_to_f32_icelake", nk_cast_icelake);
    cast_<nk_f32_k, nk_e4m3_k>("cast_f32_to_e4m3_icelake", nk_cast_icelake);
    cast_<nk_e4m3_k, nk_f32_k>("cast_e4m3_to_f32_icelake", nk_cast_icelake);
#endif

#if NK_TARGET_SAPPHIRE
    cast_<nk_f32_k, nk_f16_k>("cast_f32_to_f16_sapphire", nk_cast_sapphire);
    cast_<nk_f16_k, nk_f32_k>("cast_f16_to_f32_sapphire", nk_cast_sapphire);
#endif

    // Serial fallbacks
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
}

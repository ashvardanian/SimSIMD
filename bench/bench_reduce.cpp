/**
 *  @brief Reduction benchmarks (reduce_add).
 *  @file bench/bench_reduce.cpp
 *  @author Ash Vardanian
 *  @date February 6, 2026
 */

#include "numkong/reduce.h"

#include "bench.hpp"

/**
 *  @brief Measures the performance of a reduce_add kernel using Google Benchmark.
 *  Reduce takes a single input vector + stride and produces a scalar output.
 */
template <nk_dtype_t input_dtype_, nk_dtype_t output_dtype_, typename kernel_type_ = void>
void measure_reduce(bm::State &state, kernel_type_ kernel, std::size_t dimensions) {

    using input_t = typename nk::type_for<input_dtype_>::type;
    using output_t = typename nk::type_for<output_dtype_>::type;

    constexpr std::size_t vectors_count = 1024;
    std::vector<nk::vector<input_t>> vectors(vectors_count);
    auto generator = make_random_engine();
    for (auto &v : vectors) {
        v = make_vector<input_t>(dimensions);
        nk::fill_uniform(generator, v.values_data(), v.size_values());
    }

    std::size_t iterations = 0;
    for (auto _ : state) {
        output_t output;
        auto const &v = vectors[iterations & (vectors_count - 1)];
        kernel(v.raw_values_data(), dimensions, sizeof(typename input_t::raw_t), &output.raw_);
        bm::DoNotOptimize(output);
        ++iterations;
    }

    state.counters["bytes"] = bm::Counter(iterations * vectors[0].size_bytes(), bm::Counter::kIsRate);
    state.counters["calls"] = bm::Counter(iterations, bm::Counter::kIsRate);
}

template <nk_dtype_t input_dtype_, nk_dtype_t output_dtype_, typename kernel_type_ = void>
void reduce_(std::string name, kernel_type_ *kernel) {
    std::string bench_name = name + "<" + std::to_string(dense_dimensions) + "d>";
    bm::RegisterBenchmark(bench_name.c_str(), measure_reduce<input_dtype_, output_dtype_, kernel_type_ *>, kernel,
                          dense_dimensions)
        ->MinTime(default_seconds)
        ->Threads(default_threads);
}

void bench_reduce() {
    constexpr nk_dtype_t f32_k = nk_f32_k;
    constexpr nk_dtype_t f64_k = nk_f64_k;
    constexpr nk_dtype_t i32_k = nk_i32_k;
    constexpr nk_dtype_t i64_k = nk_i64_k;
    constexpr nk_dtype_t e4m3_k = nk_e4m3_k;
    constexpr nk_dtype_t e5m2_k = nk_e5m2_k;

    // Serial
    reduce_<f32_k, f64_k>("reduce_add_f32_serial", nk_reduce_add_f32_serial);
    reduce_<f64_k, f64_k>("reduce_add_f64_serial", nk_reduce_add_f64_serial);
    reduce_<i32_k, i64_k>("reduce_add_i32_serial", nk_reduce_add_i32_serial);
    reduce_<e4m3_k, f32_k>("reduce_add_e4m3_serial", nk_reduce_add_e4m3_serial);
    reduce_<e5m2_k, f32_k>("reduce_add_e5m2_serial", nk_reduce_add_e5m2_serial);

#if NK_TARGET_NEON
    reduce_<f32_k, f64_k>("reduce_add_f32_neon", nk_reduce_add_f32_neon);
    reduce_<f64_k, f64_k>("reduce_add_f64_neon", nk_reduce_add_f64_neon);
#endif

#if NK_TARGET_NEONHALF
    reduce_<nk_f16_k, f32_k>("reduce_add_f16_neonhalf", nk_reduce_add_f16_neonhalf);
#endif

#if NK_TARGET_NEONBFDOT
    reduce_<nk_bf16_k, f32_k>("reduce_add_bf16_neonbfdot", nk_reduce_add_bf16_neonbfdot);
#endif

#if NK_TARGET_NEONSDOT
    reduce_<nk_i8_k, i64_k>("reduce_add_i8_neonsdot", nk_reduce_add_i8_neonsdot);
    reduce_<nk_u8_k, nk_u64_k>("reduce_add_u8_neonsdot", nk_reduce_add_u8_neonsdot);
#endif

#if NK_TARGET_NEONFHM
    reduce_<e4m3_k, f32_k>("reduce_add_e4m3_neonfhm", nk_reduce_add_e4m3_neonfhm);
    reduce_<e5m2_k, f32_k>("reduce_add_e5m2_neonfhm", nk_reduce_add_e5m2_neonfhm);
#endif

#if NK_TARGET_HASWELL
    reduce_<f32_k, f64_k>("reduce_add_f32_haswell", nk_reduce_add_f32_haswell);
    reduce_<f64_k, f64_k>("reduce_add_f64_haswell", nk_reduce_add_f64_haswell);
    reduce_<i32_k, i64_k>("reduce_add_i32_haswell", nk_reduce_add_i32_haswell);
    reduce_<e4m3_k, f32_k>("reduce_add_e4m3_haswell", nk_reduce_add_e4m3_haswell);
    reduce_<e5m2_k, f32_k>("reduce_add_e5m2_haswell", nk_reduce_add_e5m2_haswell);
#endif

#if NK_TARGET_SKYLAKE
    reduce_<f32_k, f64_k>("reduce_add_f32_skylake", nk_reduce_add_f32_skylake);
    reduce_<f64_k, f64_k>("reduce_add_f64_skylake", nk_reduce_add_f64_skylake);
    reduce_<i32_k, i64_k>("reduce_add_i32_skylake", nk_reduce_add_i32_skylake);
#endif

#if NK_TARGET_SIERRA
    reduce_<nk_i8_k, i64_k>("reduce_add_i8_sierra", nk_reduce_add_i8_sierra);
    reduce_<nk_u8_k, nk_u64_k>("reduce_add_u8_sierra", nk_reduce_add_u8_sierra);
    reduce_<nk_i16_k, i64_k>("reduce_add_i16_sierra", nk_reduce_add_i16_sierra);
    reduce_<nk_u16_k, nk_u64_k>("reduce_add_u16_sierra", nk_reduce_add_u16_sierra);
#endif
}

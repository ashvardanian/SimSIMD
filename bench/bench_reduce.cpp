/**
 *  @brief Reduction benchmarks (reduce_moments, reduce_minmax).
 *  @file bench/bench_reduce.cpp
 *  @author Ash Vardanian
 *  @date February 6, 2026
 */

#include "numkong/reduce.h"
#include "numkong/reduce/serial.h"

#include "bench.hpp"

/**
 *  @brief Measures the performance of a reduce_moments kernel (sum + sumsq).
 */
template <nk_dtype_t input_dtype_, nk_dtype_t output_dtype_, nk_dtype_t sumsq_dtype_ = output_dtype_,
          typename kernel_type_ = void>
void measure_reduce_moments(bm::State &state, kernel_type_ kernel, std::size_t dimensions) {
    using input_t = typename nk::type_for<input_dtype_>::type;
    using output_t = typename nk::type_for<output_dtype_>::type;
    using sumsq_t = typename nk::type_for<sumsq_dtype_>::type;

    std::size_t const vectors_count = bench_input_count(bench_dtype_bytes(input_dtype_, dimensions));
    std::vector<nk::vector<input_t>> vectors(vectors_count);
    auto generator = make_random_engine();
    for (auto &v : vectors) {
        v = make_vector<input_t>(dimensions);
        nk::fill_uniform(generator, v.values_data(), v.size_values());
    }

    std::size_t iterations = 0;
    for (auto _ : state) {
        typename output_t::raw_t sum;
        typename sumsq_t::raw_t sumsq;
        auto const &v = vectors[iterations & (vectors_count - 1)];
        kernel(v.raw_values_data(), dimensions, sizeof(typename input_t::raw_t), &sum, &sumsq);
        bm::DoNotOptimize(sum);
        bm::DoNotOptimize(sumsq);
        ++iterations;
    }

    state.counters["bytes"] = bm::Counter(iterations * vectors[0].size_bytes(), bm::Counter::kIsRate);
    state.counters["calls"] = bm::Counter(iterations, bm::Counter::kIsRate);
}

template <nk_dtype_t input_dtype_, nk_dtype_t output_dtype_, nk_dtype_t sumsq_dtype_ = output_dtype_,
          typename kernel_type_ = void>
void reduce_moments_(std::string name, kernel_type_ *kernel) {
    std::string bench_name = name + "<" + std::to_string(bench_config.dense_dimensions) + "d>";
    bm::RegisterBenchmark(bench_name.c_str(),
                          measure_reduce_moments<input_dtype_, output_dtype_, sumsq_dtype_, kernel_type_ *>, kernel,
                          bench_config.dense_dimensions);
}

/**
 *  @brief Measures the performance of a reduce_minmax kernel (min + max with indices).
 */
template <nk_dtype_t input_dtype_, nk_dtype_t output_dtype_, typename kernel_type_ = void>
void measure_reduce_minmax(bm::State &state, kernel_type_ kernel, std::size_t dimensions) {
    using input_t = typename nk::type_for<input_dtype_>::type;
    using output_t = typename nk::type_for<output_dtype_>::type;

    std::size_t const vectors_count = bench_input_count(bench_dtype_bytes(input_dtype_, dimensions));
    std::vector<nk::vector<input_t>> vectors(vectors_count);
    auto generator = make_random_engine();
    for (auto &v : vectors) {
        v = make_vector<input_t>(dimensions);
        nk::fill_uniform(generator, v.values_data(), v.size_values());
    }

    std::size_t iterations = 0;
    for (auto _ : state) {
        typename output_t::raw_t min_val, max_val;
        nk_size_t min_idx, max_idx;
        auto const &v = vectors[iterations & (vectors_count - 1)];
        kernel(v.raw_values_data(), dimensions, sizeof(typename input_t::raw_t), &min_val, &min_idx, &max_val,
               &max_idx);
        bm::DoNotOptimize(min_val);
        bm::DoNotOptimize(max_val);
        ++iterations;
    }

    state.counters["bytes"] = bm::Counter(iterations * vectors[0].size_bytes(), bm::Counter::kIsRate);
    state.counters["calls"] = bm::Counter(iterations, bm::Counter::kIsRate);
}

template <nk_dtype_t input_dtype_, nk_dtype_t output_dtype_, typename kernel_type_ = void>
void reduce_minmax_(std::string name, kernel_type_ *kernel) {
    std::string bench_name = name + "<" + std::to_string(bench_config.dense_dimensions) + "d>";
    bm::RegisterBenchmark(bench_name.c_str(), measure_reduce_minmax<input_dtype_, output_dtype_, kernel_type_ *>,
                          kernel, bench_config.dense_dimensions);
}

void bench_reduce() {
    constexpr nk_dtype_t f32_k = nk_f32_k;
    constexpr nk_dtype_t f64_k = nk_f64_k;
    constexpr nk_dtype_t i8_k = nk_i8_k;
    constexpr nk_dtype_t u8_k = nk_u8_k;
    constexpr nk_dtype_t i16_k = nk_i16_k;
    constexpr nk_dtype_t u16_k = nk_u16_k;
    constexpr nk_dtype_t i32_k = nk_i32_k;
    constexpr nk_dtype_t u32_k = nk_u32_k;
    constexpr nk_dtype_t i64_k = nk_i64_k;
    constexpr nk_dtype_t u64_k = nk_u64_k;
    constexpr nk_dtype_t e4m3_k = nk_e4m3_k;
    constexpr nk_dtype_t e5m2_k = nk_e5m2_k;
    constexpr nk_dtype_t e2m3_k = nk_e2m3_k;
    constexpr nk_dtype_t e3m2_k = nk_e3m2_k;
    constexpr nk_dtype_t f16_k = nk_f16_k;
    constexpr nk_dtype_t bf16_k = nk_bf16_k;
    constexpr nk_dtype_t i4_k = nk_i4_k;
    constexpr nk_dtype_t u4_k = nk_u4_k;
    constexpr nk_dtype_t u1_k = nk_u1_k;

#if NK_TARGET_NEON
    reduce_moments_<f32_k, f64_k>("reduce_moments_f32_neon", nk_reduce_moments_f32_neon);
    reduce_moments_<f64_k, f64_k>("reduce_moments_f64_neon", nk_reduce_moments_f64_neon);
    reduce_moments_<i8_k, i64_k, u64_k>("reduce_moments_i8_neon", nk_reduce_moments_i8_neon);
    reduce_moments_<u8_k, u64_k>("reduce_moments_u8_neon", nk_reduce_moments_u8_neon);
    reduce_moments_<i16_k, i64_k, u64_k>("reduce_moments_i16_neon", nk_reduce_moments_i16_neon);
    reduce_moments_<u16_k, u64_k>("reduce_moments_u16_neon", nk_reduce_moments_u16_neon);
    reduce_moments_<i32_k, i64_k, u64_k>("reduce_moments_i32_neon", nk_reduce_moments_i32_neon);
    reduce_moments_<u32_k, u64_k>("reduce_moments_u32_neon", nk_reduce_moments_u32_neon);
    reduce_moments_<i64_k, i64_k, u64_k>("reduce_moments_i64_neon", nk_reduce_moments_i64_neon);
    reduce_moments_<u64_k, u64_k>("reduce_moments_u64_neon", nk_reduce_moments_u64_neon);
    reduce_moments_<e4m3_k, f32_k>("reduce_moments_e4m3_neon", nk_reduce_moments_e4m3_neon);
    reduce_moments_<e5m2_k, f32_k>("reduce_moments_e5m2_neon", nk_reduce_moments_e5m2_neon);
    reduce_moments_<e2m3_k, f32_k>("reduce_moments_e2m3_neon", nk_reduce_moments_e2m3_neon);
    reduce_moments_<e3m2_k, f32_k>("reduce_moments_e3m2_neon", nk_reduce_moments_e3m2_neon);
    reduce_minmax_<f32_k, f32_k>("reduce_minmax_f32_neon", nk_reduce_minmax_f32_neon);
    reduce_minmax_<f64_k, f64_k>("reduce_minmax_f64_neon", nk_reduce_minmax_f64_neon);
    reduce_minmax_<i8_k, i8_k>("reduce_minmax_i8_neon", nk_reduce_minmax_i8_neon);
    reduce_minmax_<u8_k, u8_k>("reduce_minmax_u8_neon", nk_reduce_minmax_u8_neon);
    reduce_minmax_<i16_k, i16_k>("reduce_minmax_i16_neon", nk_reduce_minmax_i16_neon);
    reduce_minmax_<u16_k, u16_k>("reduce_minmax_u16_neon", nk_reduce_minmax_u16_neon);
    reduce_minmax_<i32_k, i32_k>("reduce_minmax_i32_neon", nk_reduce_minmax_i32_neon);
    reduce_minmax_<u32_k, u32_k>("reduce_minmax_u32_neon", nk_reduce_minmax_u32_neon);
    reduce_minmax_<i64_k, i64_k>("reduce_minmax_i64_neon", nk_reduce_minmax_i64_neon);
    reduce_minmax_<u64_k, u64_k>("reduce_minmax_u64_neon", nk_reduce_minmax_u64_neon);
    reduce_minmax_<e4m3_k, e4m3_k>("reduce_minmax_e4m3_neon", nk_reduce_minmax_e4m3_neon);
    reduce_minmax_<e5m2_k, e5m2_k>("reduce_minmax_e5m2_neon", nk_reduce_minmax_e5m2_neon);
    reduce_minmax_<e2m3_k, e2m3_k>("reduce_minmax_e2m3_neon", nk_reduce_minmax_e2m3_neon);
    reduce_minmax_<e3m2_k, e3m2_k>("reduce_minmax_e3m2_neon", nk_reduce_minmax_e3m2_neon);
#endif

#if NK_TARGET_NEONHALF
    reduce_moments_<f16_k, f32_k>("reduce_moments_f16_neonhalf", nk_reduce_moments_f16_neonhalf);
    reduce_minmax_<f16_k, f16_k>("reduce_minmax_f16_neonhalf", nk_reduce_minmax_f16_neonhalf);
#endif

#if NK_TARGET_NEONBFDOT
    reduce_moments_<bf16_k, f32_k>("reduce_moments_bf16_neonbfdot", nk_reduce_moments_bf16_neonbfdot);
    reduce_minmax_<bf16_k, bf16_k>("reduce_minmax_bf16_neonbfdot", nk_reduce_minmax_bf16_neonbfdot);
#endif

#if NK_TARGET_NEONSDOT
    reduce_moments_<i8_k, i64_k, u64_k>("reduce_moments_i8_neonsdot", nk_reduce_moments_i8_neonsdot);
    reduce_moments_<u8_k, u64_k>("reduce_moments_u8_neonsdot", nk_reduce_moments_u8_neonsdot);
    reduce_moments_<e2m3_k, f32_k>("reduce_moments_e2m3_neonsdot", nk_reduce_moments_e2m3_neonsdot);
#endif

#if NK_TARGET_NEONFHM
    reduce_moments_<e4m3_k, f32_k>("reduce_moments_e4m3_neonfhm", nk_reduce_moments_e4m3_neonfhm);
    reduce_moments_<e5m2_k, f32_k>("reduce_moments_e5m2_neonfhm", nk_reduce_moments_e5m2_neonfhm);
    reduce_moments_<e3m2_k, f32_k>("reduce_moments_e3m2_neonfhm", nk_reduce_moments_e3m2_neonfhm);
    reduce_minmax_<e4m3_k, e4m3_k>("reduce_minmax_e4m3_neonfhm", nk_reduce_minmax_e4m3_neonfhm);
    reduce_minmax_<e5m2_k, e5m2_k>("reduce_minmax_e5m2_neonfhm", nk_reduce_minmax_e5m2_neonfhm);
#endif

#if NK_TARGET_HASWELL
    reduce_moments_<f32_k, f64_k>("reduce_moments_f32_haswell", nk_reduce_moments_f32_haswell);
    reduce_moments_<f64_k, f64_k>("reduce_moments_f64_haswell", nk_reduce_moments_f64_haswell);
    reduce_moments_<i8_k, i64_k, u64_k>("reduce_moments_i8_haswell", nk_reduce_moments_i8_haswell);
    reduce_moments_<u8_k, u64_k>("reduce_moments_u8_haswell", nk_reduce_moments_u8_haswell);
    reduce_moments_<i16_k, i64_k, u64_k>("reduce_moments_i16_haswell", nk_reduce_moments_i16_haswell);
    reduce_moments_<u16_k, u64_k>("reduce_moments_u16_haswell", nk_reduce_moments_u16_haswell);
    reduce_moments_<i32_k, i64_k, u64_k>("reduce_moments_i32_haswell", nk_reduce_moments_i32_haswell);
    reduce_moments_<u32_k, u64_k>("reduce_moments_u32_haswell", nk_reduce_moments_u32_haswell);
    reduce_moments_<i64_k, i64_k, u64_k>("reduce_moments_i64_haswell", nk_reduce_moments_i64_haswell);
    reduce_moments_<u64_k, u64_k>("reduce_moments_u64_haswell", nk_reduce_moments_u64_haswell);
    reduce_moments_<e4m3_k, f32_k>("reduce_moments_e4m3_haswell", nk_reduce_moments_e4m3_haswell);
    reduce_moments_<e5m2_k, f32_k>("reduce_moments_e5m2_haswell", nk_reduce_moments_e5m2_haswell);
    reduce_moments_<e2m3_k, f32_k>("reduce_moments_e2m3_haswell", nk_reduce_moments_e2m3_haswell);
    reduce_moments_<e3m2_k, f32_k>("reduce_moments_e3m2_haswell", nk_reduce_moments_e3m2_haswell);
    reduce_moments_<i4_k, i64_k, u64_k>("reduce_moments_i4_haswell", nk_reduce_moments_i4_haswell);
    reduce_moments_<u4_k, u64_k>("reduce_moments_u4_haswell", nk_reduce_moments_u4_haswell);
    reduce_moments_<u1_k, u64_k>("reduce_moments_u1_haswell", nk_reduce_moments_u1_haswell);
    reduce_moments_<bf16_k, f32_k>("reduce_moments_bf16_haswell", nk_reduce_moments_bf16_haswell);
    reduce_moments_<f16_k, f32_k>("reduce_moments_f16_haswell", nk_reduce_moments_f16_haswell);
    reduce_minmax_<f32_k, f32_k>("reduce_minmax_f32_haswell", nk_reduce_minmax_f32_haswell);
    reduce_minmax_<f64_k, f64_k>("reduce_minmax_f64_haswell", nk_reduce_minmax_f64_haswell);
    reduce_minmax_<i8_k, i8_k>("reduce_minmax_i8_haswell", nk_reduce_minmax_i8_haswell);
    reduce_minmax_<u8_k, u8_k>("reduce_minmax_u8_haswell", nk_reduce_minmax_u8_haswell);
    reduce_minmax_<i16_k, i16_k>("reduce_minmax_i16_haswell", nk_reduce_minmax_i16_haswell);
    reduce_minmax_<u16_k, u16_k>("reduce_minmax_u16_haswell", nk_reduce_minmax_u16_haswell);
    reduce_minmax_<i32_k, i32_k>("reduce_minmax_i32_haswell", nk_reduce_minmax_i32_haswell);
    reduce_minmax_<u32_k, u32_k>("reduce_minmax_u32_haswell", nk_reduce_minmax_u32_haswell);
    reduce_minmax_<i64_k, i64_k>("reduce_minmax_i64_haswell", nk_reduce_minmax_i64_haswell);
    reduce_minmax_<u64_k, u64_k>("reduce_minmax_u64_haswell", nk_reduce_minmax_u64_haswell);
    reduce_minmax_<e4m3_k, e4m3_k>("reduce_minmax_e4m3_haswell", nk_reduce_minmax_e4m3_haswell);
    reduce_minmax_<e5m2_k, e5m2_k>("reduce_minmax_e5m2_haswell", nk_reduce_minmax_e5m2_haswell);
    reduce_minmax_<e2m3_k, e2m3_k>("reduce_minmax_e2m3_haswell", nk_reduce_minmax_e2m3_haswell);
    reduce_minmax_<e3m2_k, e3m2_k>("reduce_minmax_e3m2_haswell", nk_reduce_minmax_e3m2_haswell);
    reduce_minmax_<bf16_k, bf16_k>("reduce_minmax_bf16_haswell", nk_reduce_minmax_bf16_haswell);
    reduce_minmax_<f16_k, f16_k>("reduce_minmax_f16_haswell", nk_reduce_minmax_f16_haswell);
#endif

#if NK_TARGET_SKYLAKE
    reduce_moments_<f32_k, f64_k>("reduce_moments_f32_skylake", nk_reduce_moments_f32_skylake);
    reduce_moments_<f64_k, f64_k>("reduce_moments_f64_skylake", nk_reduce_moments_f64_skylake);
    reduce_moments_<i8_k, i64_k, u64_k>("reduce_moments_i8_skylake", nk_reduce_moments_i8_skylake);
    reduce_moments_<u8_k, u64_k>("reduce_moments_u8_skylake", nk_reduce_moments_u8_skylake);
    reduce_moments_<i16_k, i64_k, u64_k>("reduce_moments_i16_skylake", nk_reduce_moments_i16_skylake);
    reduce_moments_<u16_k, u64_k>("reduce_moments_u16_skylake", nk_reduce_moments_u16_skylake);
    reduce_moments_<i32_k, i64_k, u64_k>("reduce_moments_i32_skylake", nk_reduce_moments_i32_skylake);
    reduce_moments_<u32_k, u64_k>("reduce_moments_u32_skylake", nk_reduce_moments_u32_skylake);
    reduce_moments_<i64_k, i64_k, u64_k>("reduce_moments_i64_skylake", nk_reduce_moments_i64_skylake);
    reduce_moments_<u64_k, u64_k>("reduce_moments_u64_skylake", nk_reduce_moments_u64_skylake);
    reduce_moments_<e4m3_k, f32_k>("reduce_moments_e4m3_skylake", nk_reduce_moments_e4m3_skylake);
    reduce_moments_<e5m2_k, f32_k>("reduce_moments_e5m2_skylake", nk_reduce_moments_e5m2_skylake);
    reduce_moments_<e2m3_k, f32_k>("reduce_moments_e2m3_skylake", nk_reduce_moments_e2m3_skylake);
    reduce_moments_<e3m2_k, f32_k>("reduce_moments_e3m2_skylake", nk_reduce_moments_e3m2_skylake);
    reduce_moments_<i4_k, i64_k, u64_k>("reduce_moments_i4_skylake", nk_reduce_moments_i4_skylake);
    reduce_moments_<u4_k, u64_k>("reduce_moments_u4_skylake", nk_reduce_moments_u4_skylake);
    reduce_moments_<u1_k, u64_k>("reduce_moments_u1_skylake", nk_reduce_moments_u1_skylake);
    reduce_minmax_<f32_k, f32_k>("reduce_minmax_f32_skylake", nk_reduce_minmax_f32_skylake);
    reduce_minmax_<f64_k, f64_k>("reduce_minmax_f64_skylake", nk_reduce_minmax_f64_skylake);
    reduce_minmax_<i8_k, i8_k>("reduce_minmax_i8_skylake", nk_reduce_minmax_i8_skylake);
    reduce_minmax_<u8_k, u8_k>("reduce_minmax_u8_skylake", nk_reduce_minmax_u8_skylake);
    reduce_minmax_<i16_k, i16_k>("reduce_minmax_i16_skylake", nk_reduce_minmax_i16_skylake);
    reduce_minmax_<u16_k, u16_k>("reduce_minmax_u16_skylake", nk_reduce_minmax_u16_skylake);
    reduce_minmax_<i32_k, i32_k>("reduce_minmax_i32_skylake", nk_reduce_minmax_i32_skylake);
    reduce_minmax_<u32_k, u32_k>("reduce_minmax_u32_skylake", nk_reduce_minmax_u32_skylake);
    reduce_minmax_<i64_k, i64_k>("reduce_minmax_i64_skylake", nk_reduce_minmax_i64_skylake);
    reduce_minmax_<u64_k, u64_k>("reduce_minmax_u64_skylake", nk_reduce_minmax_u64_skylake);
    reduce_minmax_<e4m3_k, e4m3_k>("reduce_minmax_e4m3_skylake", nk_reduce_minmax_e4m3_skylake);
    reduce_minmax_<e5m2_k, e5m2_k>("reduce_minmax_e5m2_skylake", nk_reduce_minmax_e5m2_skylake);
    reduce_minmax_<e2m3_k, e2m3_k>("reduce_minmax_e2m3_skylake", nk_reduce_minmax_e2m3_skylake);
    reduce_minmax_<e3m2_k, e3m2_k>("reduce_minmax_e3m2_skylake", nk_reduce_minmax_e3m2_skylake);
    reduce_moments_<bf16_k, f32_k>("reduce_moments_bf16_skylake", nk_reduce_moments_bf16_skylake);
    reduce_minmax_<bf16_k, bf16_k>("reduce_minmax_bf16_skylake", nk_reduce_minmax_bf16_skylake);
    reduce_moments_<f16_k, f32_k>("reduce_moments_f16_skylake", nk_reduce_moments_f16_skylake);
    reduce_minmax_<f16_k, f16_k>("reduce_minmax_f16_skylake", nk_reduce_minmax_f16_skylake);
#endif
#if NK_TARGET_ICELAKE
    reduce_moments_<i8_k, i64_k, u64_k>("reduce_moments_i8_icelake", nk_reduce_moments_i8_icelake);
    reduce_moments_<u8_k, u64_k>("reduce_moments_u8_icelake", nk_reduce_moments_u8_icelake);
    reduce_moments_<i16_k, i64_k, u64_k>("reduce_moments_i16_icelake", nk_reduce_moments_i16_icelake);
    reduce_moments_<e2m3_k, f32_k>("reduce_moments_e2m3_icelake", nk_reduce_moments_e2m3_icelake);
    reduce_moments_<e3m2_k, f32_k>("reduce_moments_e3m2_icelake", nk_reduce_moments_e3m2_icelake);
#endif
#if NK_TARGET_GENOA
    reduce_moments_<bf16_k, f32_k>("reduce_moments_bf16_genoa", nk_reduce_moments_bf16_genoa);
    reduce_moments_<e4m3_k, f32_k>("reduce_moments_e4m3_genoa", nk_reduce_moments_e4m3_genoa);
    reduce_moments_<e5m2_k, f32_k>("reduce_moments_e5m2_genoa", nk_reduce_moments_e5m2_genoa);
    reduce_moments_<e2m3_k, f32_k>("reduce_moments_e2m3_genoa", nk_reduce_moments_e2m3_genoa);
    reduce_moments_<e3m2_k, f32_k>("reduce_moments_e3m2_genoa", nk_reduce_moments_e3m2_genoa);
#endif

#if NK_TARGET_SIERRA
    reduce_moments_<i8_k, i64_k, u64_k>("reduce_moments_i8_sierra", nk_reduce_moments_i8_sierra);
    reduce_moments_<u8_k, u64_k>("reduce_moments_u8_sierra", nk_reduce_moments_u8_sierra);
    reduce_moments_<i16_k, i64_k, u64_k>("reduce_moments_i16_sierra", nk_reduce_moments_i16_sierra);
    reduce_moments_<u16_k, u64_k>("reduce_moments_u16_sierra", nk_reduce_moments_u16_sierra);
    reduce_moments_<e2m3_k, f32_k>("reduce_moments_e2m3_sierra", nk_reduce_moments_e2m3_sierra);
    reduce_moments_<e3m2_k, f32_k>("reduce_moments_e3m2_sierra", nk_reduce_moments_e3m2_sierra);
#endif

#if NK_TARGET_RVV
    reduce_moments_<f32_k, f64_k>("reduce_moments_f32_rvv", nk_reduce_moments_f32_rvv);
    reduce_moments_<f64_k, f64_k>("reduce_moments_f64_rvv", nk_reduce_moments_f64_rvv);
    reduce_moments_<i8_k, i64_k, u64_k>("reduce_moments_i8_rvv", nk_reduce_moments_i8_rvv);
    reduce_moments_<u8_k, u64_k>("reduce_moments_u8_rvv", nk_reduce_moments_u8_rvv);
    reduce_moments_<i16_k, i64_k, u64_k>("reduce_moments_i16_rvv", nk_reduce_moments_i16_rvv);
    reduce_moments_<u16_k, u64_k>("reduce_moments_u16_rvv", nk_reduce_moments_u16_rvv);
    reduce_moments_<i32_k, i64_k, u64_k>("reduce_moments_i32_rvv", nk_reduce_moments_i32_rvv);
    reduce_moments_<u32_k, u64_k>("reduce_moments_u32_rvv", nk_reduce_moments_u32_rvv);
    reduce_moments_<i64_k, i64_k, u64_k>("reduce_moments_i64_rvv", nk_reduce_moments_i64_rvv);
    reduce_moments_<u64_k, u64_k>("reduce_moments_u64_rvv", nk_reduce_moments_u64_rvv);
    reduce_moments_<bf16_k, f32_k>("reduce_moments_bf16_rvv", nk_reduce_moments_bf16_rvv);
    reduce_moments_<f16_k, f32_k>("reduce_moments_f16_rvv", nk_reduce_moments_f16_rvv);
    reduce_moments_<e4m3_k, f32_k>("reduce_moments_e4m3_rvv", nk_reduce_moments_e4m3_rvv);
    reduce_moments_<e5m2_k, f32_k>("reduce_moments_e5m2_rvv", nk_reduce_moments_e5m2_rvv);
    reduce_moments_<e2m3_k, f32_k>("reduce_moments_e2m3_rvv", nk_reduce_moments_e2m3_rvv);
    reduce_moments_<e3m2_k, f32_k>("reduce_moments_e3m2_rvv", nk_reduce_moments_e3m2_rvv);
    reduce_minmax_<f32_k, f32_k>("reduce_minmax_f32_rvv", nk_reduce_minmax_f32_rvv);
    reduce_minmax_<f64_k, f64_k>("reduce_minmax_f64_rvv", nk_reduce_minmax_f64_rvv);
    reduce_minmax_<i8_k, i8_k>("reduce_minmax_i8_rvv", nk_reduce_minmax_i8_rvv);
    reduce_minmax_<u8_k, u8_k>("reduce_minmax_u8_rvv", nk_reduce_minmax_u8_rvv);
    reduce_minmax_<i16_k, i16_k>("reduce_minmax_i16_rvv", nk_reduce_minmax_i16_rvv);
    reduce_minmax_<u16_k, u16_k>("reduce_minmax_u16_rvv", nk_reduce_minmax_u16_rvv);
    reduce_minmax_<i32_k, i32_k>("reduce_minmax_i32_rvv", nk_reduce_minmax_i32_rvv);
    reduce_minmax_<u32_k, u32_k>("reduce_minmax_u32_rvv", nk_reduce_minmax_u32_rvv);
    reduce_minmax_<i64_k, i64_k>("reduce_minmax_i64_rvv", nk_reduce_minmax_i64_rvv);
    reduce_minmax_<u64_k, u64_k>("reduce_minmax_u64_rvv", nk_reduce_minmax_u64_rvv);
    reduce_minmax_<bf16_k, bf16_k>("reduce_minmax_bf16_rvv", nk_reduce_minmax_bf16_rvv);
    reduce_minmax_<f16_k, f16_k>("reduce_minmax_f16_rvv", nk_reduce_minmax_f16_rvv);
    reduce_minmax_<e4m3_k, e4m3_k>("reduce_minmax_e4m3_rvv", nk_reduce_minmax_e4m3_rvv);
    reduce_minmax_<e5m2_k, e5m2_k>("reduce_minmax_e5m2_rvv", nk_reduce_minmax_e5m2_rvv);
    reduce_minmax_<e2m3_k, e2m3_k>("reduce_minmax_e2m3_rvv", nk_reduce_minmax_e2m3_rvv);
    reduce_minmax_<e3m2_k, e3m2_k>("reduce_minmax_e3m2_rvv", nk_reduce_minmax_e3m2_rvv);
#endif

#if NK_TARGET_V128RELAXED
    reduce_moments_<f32_k, f64_k>("reduce_moments_f32_v128relaxed", nk_reduce_moments_f32_v128relaxed);
    reduce_moments_<f64_k, f64_k>("reduce_moments_f64_v128relaxed", nk_reduce_moments_f64_v128relaxed);
    reduce_moments_<i8_k, i64_k, u64_k>("reduce_moments_i8_v128relaxed", nk_reduce_moments_i8_v128relaxed);
    reduce_moments_<u8_k, u64_k>("reduce_moments_u8_v128relaxed", nk_reduce_moments_u8_v128relaxed);
    reduce_moments_<i16_k, i64_k, u64_k>("reduce_moments_i16_v128relaxed", nk_reduce_moments_i16_v128relaxed);
    reduce_moments_<u16_k, u64_k>("reduce_moments_u16_v128relaxed", nk_reduce_moments_u16_v128relaxed);
    reduce_moments_<i32_k, i64_k, u64_k>("reduce_moments_i32_v128relaxed", nk_reduce_moments_i32_v128relaxed);
    reduce_moments_<u32_k, u64_k>("reduce_moments_u32_v128relaxed", nk_reduce_moments_u32_v128relaxed);
    reduce_moments_<i64_k, i64_k, u64_k>("reduce_moments_i64_v128relaxed", nk_reduce_moments_i64_v128relaxed);
    reduce_moments_<u64_k, u64_k>("reduce_moments_u64_v128relaxed", nk_reduce_moments_u64_v128relaxed);
    reduce_moments_<e4m3_k, f32_k>("reduce_moments_e4m3_v128relaxed", nk_reduce_moments_e4m3_v128relaxed);
    reduce_moments_<e5m2_k, f32_k>("reduce_moments_e5m2_v128relaxed", nk_reduce_moments_e5m2_v128relaxed);
    reduce_moments_<e2m3_k, f32_k>("reduce_moments_e2m3_v128relaxed", nk_reduce_moments_e2m3_v128relaxed);
    reduce_moments_<e3m2_k, f32_k>("reduce_moments_e3m2_v128relaxed", nk_reduce_moments_e3m2_v128relaxed);
    reduce_moments_<bf16_k, f32_k>("reduce_moments_bf16_v128relaxed", nk_reduce_moments_bf16_v128relaxed);
    reduce_moments_<f16_k, f32_k>("reduce_moments_f16_v128relaxed", nk_reduce_moments_f16_v128relaxed);
    reduce_minmax_<f32_k, f32_k>("reduce_minmax_f32_v128relaxed", nk_reduce_minmax_f32_v128relaxed);
    reduce_minmax_<f64_k, f64_k>("reduce_minmax_f64_v128relaxed", nk_reduce_minmax_f64_v128relaxed);
    reduce_minmax_<i8_k, i8_k>("reduce_minmax_i8_v128relaxed", nk_reduce_minmax_i8_v128relaxed);
    reduce_minmax_<u8_k, u8_k>("reduce_minmax_u8_v128relaxed", nk_reduce_minmax_u8_v128relaxed);
    reduce_minmax_<i16_k, i16_k>("reduce_minmax_i16_v128relaxed", nk_reduce_minmax_i16_v128relaxed);
    reduce_minmax_<u16_k, u16_k>("reduce_minmax_u16_v128relaxed", nk_reduce_minmax_u16_v128relaxed);
    reduce_minmax_<i32_k, i32_k>("reduce_minmax_i32_v128relaxed", nk_reduce_minmax_i32_v128relaxed);
    reduce_minmax_<u32_k, u32_k>("reduce_minmax_u32_v128relaxed", nk_reduce_minmax_u32_v128relaxed);
    reduce_minmax_<i64_k, i64_k>("reduce_minmax_i64_v128relaxed", nk_reduce_minmax_i64_v128relaxed);
    reduce_minmax_<u64_k, u64_k>("reduce_minmax_u64_v128relaxed", nk_reduce_minmax_u64_v128relaxed);
    reduce_minmax_<e4m3_k, e4m3_k>("reduce_minmax_e4m3_v128relaxed", nk_reduce_minmax_e4m3_v128relaxed);
    reduce_minmax_<e5m2_k, e5m2_k>("reduce_minmax_e5m2_v128relaxed", nk_reduce_minmax_e5m2_v128relaxed);
    reduce_minmax_<e2m3_k, e2m3_k>("reduce_minmax_e2m3_v128relaxed", nk_reduce_minmax_e2m3_v128relaxed);
    reduce_minmax_<e3m2_k, e3m2_k>("reduce_minmax_e3m2_v128relaxed", nk_reduce_minmax_e3m2_v128relaxed);
    reduce_minmax_<bf16_k, bf16_k>("reduce_minmax_bf16_v128relaxed", nk_reduce_minmax_bf16_v128relaxed);
    reduce_minmax_<f16_k, f16_k>("reduce_minmax_f16_v128relaxed", nk_reduce_minmax_f16_v128relaxed);
#endif

    reduce_moments_<f32_k, f64_k>("reduce_moments_f32_serial", nk_reduce_moments_f32_serial);
    reduce_moments_<f64_k, f64_k>("reduce_moments_f64_serial", nk_reduce_moments_f64_serial);
    reduce_moments_<i8_k, i64_k, u64_k>("reduce_moments_i8_serial", nk_reduce_moments_i8_serial);
    reduce_moments_<u8_k, u64_k>("reduce_moments_u8_serial", nk_reduce_moments_u8_serial);
    reduce_moments_<i16_k, i64_k, u64_k>("reduce_moments_i16_serial", nk_reduce_moments_i16_serial);
    reduce_moments_<u16_k, u64_k>("reduce_moments_u16_serial", nk_reduce_moments_u16_serial);
    reduce_moments_<i32_k, i64_k, u64_k>("reduce_moments_i32_serial", nk_reduce_moments_i32_serial);
    reduce_moments_<u32_k, u64_k>("reduce_moments_u32_serial", nk_reduce_moments_u32_serial);
    reduce_moments_<i64_k, i64_k, u64_k>("reduce_moments_i64_serial", nk_reduce_moments_i64_serial);
    reduce_moments_<u64_k, u64_k>("reduce_moments_u64_serial", nk_reduce_moments_u64_serial);
    reduce_moments_<f16_k, f32_k>("reduce_moments_f16_serial", nk_reduce_moments_f16_serial);
    reduce_moments_<bf16_k, f32_k>("reduce_moments_bf16_serial", nk_reduce_moments_bf16_serial);
    reduce_moments_<e4m3_k, f32_k>("reduce_moments_e4m3_serial", nk_reduce_moments_e4m3_serial);
    reduce_moments_<e5m2_k, f32_k>("reduce_moments_e5m2_serial", nk_reduce_moments_e5m2_serial);
    reduce_moments_<e2m3_k, f32_k>("reduce_moments_e2m3_serial", nk_reduce_moments_e2m3_serial);
    reduce_moments_<e3m2_k, f32_k>("reduce_moments_e3m2_serial", nk_reduce_moments_e3m2_serial);
    reduce_moments_<i4_k, i64_k, u64_k>("reduce_moments_i4_serial", nk_reduce_moments_i4_serial);
    reduce_moments_<u4_k, u64_k>("reduce_moments_u4_serial", nk_reduce_moments_u4_serial);
    reduce_moments_<u1_k, u64_k>("reduce_moments_u1_serial", nk_reduce_moments_u1_serial);
    reduce_minmax_<f32_k, f32_k>("reduce_minmax_f32_serial", nk_reduce_minmax_f32_serial);
    reduce_minmax_<f64_k, f64_k>("reduce_minmax_f64_serial", nk_reduce_minmax_f64_serial);
    reduce_minmax_<i8_k, i8_k>("reduce_minmax_i8_serial", nk_reduce_minmax_i8_serial);
    reduce_minmax_<u8_k, u8_k>("reduce_minmax_u8_serial", nk_reduce_minmax_u8_serial);
    reduce_minmax_<i16_k, i16_k>("reduce_minmax_i16_serial", nk_reduce_minmax_i16_serial);
    reduce_minmax_<u16_k, u16_k>("reduce_minmax_u16_serial", nk_reduce_minmax_u16_serial);
    reduce_minmax_<i32_k, i32_k>("reduce_minmax_i32_serial", nk_reduce_minmax_i32_serial);
    reduce_minmax_<u32_k, u32_k>("reduce_minmax_u32_serial", nk_reduce_minmax_u32_serial);
    reduce_minmax_<i64_k, i64_k>("reduce_minmax_i64_serial", nk_reduce_minmax_i64_serial);
    reduce_minmax_<u64_k, u64_k>("reduce_minmax_u64_serial", nk_reduce_minmax_u64_serial);
    reduce_minmax_<f16_k, f16_k>("reduce_minmax_f16_serial", nk_reduce_minmax_f16_serial);
    reduce_minmax_<bf16_k, bf16_k>("reduce_minmax_bf16_serial", nk_reduce_minmax_bf16_serial);
    reduce_minmax_<e4m3_k, e4m3_k>("reduce_minmax_e4m3_serial", nk_reduce_minmax_e4m3_serial);
    reduce_minmax_<e5m2_k, e5m2_k>("reduce_minmax_e5m2_serial", nk_reduce_minmax_e5m2_serial);
    reduce_minmax_<e2m3_k, e2m3_k>("reduce_minmax_e2m3_serial", nk_reduce_minmax_e2m3_serial);
    reduce_minmax_<e3m2_k, e3m2_k>("reduce_minmax_e3m2_serial", nk_reduce_minmax_e3m2_serial);
    reduce_minmax_<i4_k, i8_k>("reduce_minmax_i4_serial", nk_reduce_minmax_i4_serial);
    reduce_minmax_<u4_k, u8_k>("reduce_minmax_u4_serial", nk_reduce_minmax_u4_serial);
    reduce_minmax_<u1_k, u8_k>("reduce_minmax_u1_serial", nk_reduce_minmax_u1_serial);
}

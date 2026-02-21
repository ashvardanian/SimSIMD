/**
 *  @brief Reduction tests.
 *  @file test/test_reduce.cpp
 *  @author Ash Vardanian
 *  @date February 6, 2026
 */

#include "test.hpp"
#include "numkong/reduce.hpp"
#include "numkong/reduce/serial.h"

template <typename input_type_>
error_stats_t test_reduce_moments(typename input_type_::reduce_moments_kernel_t kernel,
                                  typename input_type_::reduce_moments_kernel_t reference) {
    using sum_type_ = typename input_type_::reduce_moments_sum_t;
    using sumsq_type_ = typename input_type_::reduce_moments_sumsq_t;
    error_stats_t stats;
    std::mt19937 generator(global_config.seed);
    auto buffer = make_vector<input_t_>(global_config.dense_dimensions);
    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(generator, buffer);
        typename sum_type_::raw_t sum;
        typename sumsq_type_::raw_t sumsq;
        kernel(buffer.raw_values_data(), global_config.dense_dimensions, sizeof(typename input_type_::raw_t), &sum,
               &sumsq);
        typename sum_type_::raw_t ref_sum;
        typename sumsq_type_::raw_t ref_sumsq;
        reference(buffer.raw_values_data(), global_config.dense_dimensions, sizeof(typename input_type_::raw_t),
                  &ref_sum, &ref_sumsq);
        stats.accumulate(sum_type_::from_raw(sum), sum_type_::from_raw(ref_sum));
        stats.accumulate(sumsq_type_::from_raw(sumsq), sumsq_type_::from_raw(ref_sumsq));
    }
    return stats;
}

template <typename input_type_>
error_stats_t test_reduce_minmax(typename input_type_::reduce_minmax_kernel_t kernel,
                                 typename input_type_::reduce_minmax_kernel_t reference) {
    using output_type_ = typename input_type_::reduce_minmax_value_t;
    error_stats_t stats;
    std::mt19937 generator(global_config.seed);
    auto buffer = make_vector<input_type_>(global_config.dense_dimensions);
    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(generator, buffer);
        typename output_type_::raw_t min_val, max_val;
        nk_size_t min_idx, max_idx;
        kernel(buffer.raw_values_data(), global_config.dense_dimensions, sizeof(typename input_type_::raw_t), &min_val,
               &min_idx, &max_val, &max_idx);
        typename output_type_::raw_t ref_min, ref_max;
        nk_size_t ref_min_idx, ref_max_idx;
        reference(buffer.raw_values_data(), global_config.dense_dimensions, sizeof(typename input_type_::raw_t),
                  &ref_min, &ref_min_idx, &ref_max, &ref_max_idx);
        stats.accumulate(output_type_::from_raw(min_val), output_type_::from_raw(ref_min));
        stats.accumulate(output_type_::from_raw(max_val), output_type_::from_raw(ref_max));
    }
    return stats;
}

void test_reduce() {
    std::puts("");
    std::printf("Reductions:\n");

#if NK_DYNAMIC_DISPATCH

#else

#if NK_TARGET_NEON

#endif

#if NK_TARGET_HASWELL
    run_if_matches("reduce_moments_f32_haswell", test_reduce_moments<f32_t>, nk_reduce_moments_f32_haswell,
                   nk_reduce_moments_f32_serial);
    run_if_matches("reduce_moments_f64_haswell", test_reduce_moments<f64_t>, nk_reduce_moments_f64_haswell,
                   nk_reduce_moments_f64_serial);
    run_if_matches("reduce_moments_i8_haswell", test_reduce_moments<i8_t>, nk_reduce_moments_i8_haswell,
                   nk_reduce_moments_i8_serial);
    run_if_matches("reduce_moments_u8_haswell", test_reduce_moments<u8_t>, nk_reduce_moments_u8_haswell,
                   nk_reduce_moments_u8_serial);
    run_if_matches("reduce_moments_i16_haswell", test_reduce_moments<i16_t>, nk_reduce_moments_i16_haswell,
                   nk_reduce_moments_i16_serial);
    run_if_matches("reduce_moments_u16_haswell", test_reduce_moments<u16_t>, nk_reduce_moments_u16_haswell,
                   nk_reduce_moments_u16_serial);
    run_if_matches("reduce_moments_i32_haswell", test_reduce_moments<i32_t>, nk_reduce_moments_i32_haswell,
                   nk_reduce_moments_i32_serial);
    run_if_matches("reduce_moments_u32_haswell", test_reduce_moments<u32_t>, nk_reduce_moments_u32_haswell,
                   nk_reduce_moments_u32_serial);
    run_if_matches("reduce_moments_i64_haswell", test_reduce_moments<i64_t>, nk_reduce_moments_i64_haswell,
                   nk_reduce_moments_i64_serial);
    run_if_matches("reduce_moments_u64_haswell", test_reduce_moments<u64_t>, nk_reduce_moments_u64_haswell,
                   nk_reduce_moments_u64_serial);
    run_if_matches("reduce_moments_e4m3_haswell", test_reduce_moments<e4m3_t>, nk_reduce_moments_e4m3_haswell,
                   nk_reduce_moments_e4m3_serial);
    run_if_matches("reduce_moments_e5m2_haswell", test_reduce_moments<e5m2_t>, nk_reduce_moments_e5m2_haswell,
                   nk_reduce_moments_e5m2_serial);
    run_if_matches("reduce_moments_e2m3_haswell", test_reduce_moments<e2m3_t>, nk_reduce_moments_e2m3_haswell,
                   nk_reduce_moments_e2m3_serial);
    run_if_matches("reduce_moments_e3m2_haswell", test_reduce_moments<e3m2_t>, nk_reduce_moments_e3m2_haswell,
                   nk_reduce_moments_e3m2_serial);
    run_if_matches("reduce_moments_i4_haswell", test_reduce_moments<i4x2_t>, nk_reduce_moments_i4_haswell,
                   nk_reduce_moments_i4_serial);
    run_if_matches("reduce_moments_u4_haswell", test_reduce_moments<u4x2_t>, nk_reduce_moments_u4_haswell,
                   nk_reduce_moments_u4_serial);
    run_if_matches("reduce_moments_u1_haswell", test_reduce_moments<u1x8_t>, nk_reduce_moments_u1_haswell,
                   nk_reduce_moments_u1_serial);
    run_if_matches("reduce_moments_bf16_haswell", test_reduce_moments<bf16_t>, nk_reduce_moments_bf16_haswell,
                   nk_reduce_moments_bf16_serial);
    run_if_matches("reduce_moments_f16_haswell", test_reduce_moments<f16_t>, nk_reduce_moments_f16_haswell,
                   nk_reduce_moments_f16_serial);
    run_if_matches("reduce_minmax_f32_haswell", test_reduce_minmax<f32_t>, nk_reduce_minmax_f32_haswell,
                   nk_reduce_minmax_f32_serial);
    run_if_matches("reduce_minmax_f64_haswell", test_reduce_minmax<f64_t>, nk_reduce_minmax_f64_haswell,
                   nk_reduce_minmax_f64_serial);
    run_if_matches("reduce_minmax_i8_haswell", test_reduce_minmax<i8_t>, nk_reduce_minmax_i8_haswell,
                   nk_reduce_minmax_i8_serial);
    run_if_matches("reduce_minmax_u8_haswell", test_reduce_minmax<u8_t>, nk_reduce_minmax_u8_haswell,
                   nk_reduce_minmax_u8_serial);
    run_if_matches("reduce_minmax_i16_haswell", test_reduce_minmax<i16_t>, nk_reduce_minmax_i16_haswell,
                   nk_reduce_minmax_i16_serial);
    run_if_matches("reduce_minmax_u16_haswell", test_reduce_minmax<u16_t>, nk_reduce_minmax_u16_haswell,
                   nk_reduce_minmax_u16_serial);
    run_if_matches("reduce_minmax_i32_haswell", test_reduce_minmax<i32_t>, nk_reduce_minmax_i32_haswell,
                   nk_reduce_minmax_i32_serial);
    run_if_matches("reduce_minmax_u32_haswell", test_reduce_minmax<u32_t>, nk_reduce_minmax_u32_haswell,
                   nk_reduce_minmax_u32_serial);
    run_if_matches("reduce_minmax_i64_haswell", test_reduce_minmax<i64_t>, nk_reduce_minmax_i64_haswell,
                   nk_reduce_minmax_i64_serial);
    run_if_matches("reduce_minmax_u64_haswell", test_reduce_minmax<u64_t>, nk_reduce_minmax_u64_haswell,
                   nk_reduce_minmax_u64_serial);
    run_if_matches("reduce_minmax_e4m3_haswell", test_reduce_minmax<e4m3_t>, nk_reduce_minmax_e4m3_haswell,
                   nk_reduce_minmax_e4m3_serial);
    run_if_matches("reduce_minmax_e5m2_haswell", test_reduce_minmax<e5m2_t>, nk_reduce_minmax_e5m2_haswell,
                   nk_reduce_minmax_e5m2_serial);
    run_if_matches("reduce_minmax_e2m3_haswell", test_reduce_minmax<e2m3_t>, nk_reduce_minmax_e2m3_haswell,
                   nk_reduce_minmax_e2m3_serial);
    run_if_matches("reduce_minmax_e3m2_haswell", test_reduce_minmax<e3m2_t>, nk_reduce_minmax_e3m2_haswell,
                   nk_reduce_minmax_e3m2_serial);
    run_if_matches("reduce_minmax_bf16_haswell", test_reduce_minmax<bf16_t>, nk_reduce_minmax_bf16_haswell,
                   nk_reduce_minmax_bf16_serial);
    run_if_matches("reduce_minmax_f16_haswell", test_reduce_minmax<f16_t>, nk_reduce_minmax_f16_haswell,
                   nk_reduce_minmax_f16_serial);
#endif

#if NK_TARGET_SKYLAKE
    run_if_matches("reduce_moments_f32_skylake", test_reduce_moments<f32_t>, nk_reduce_moments_f32_skylake,
                   nk_reduce_moments_f32_serial);
    run_if_matches("reduce_moments_f64_skylake", test_reduce_moments<f64_t>, nk_reduce_moments_f64_skylake,
                   nk_reduce_moments_f64_serial);
    run_if_matches("reduce_moments_i8_skylake", test_reduce_moments<i8_t>, nk_reduce_moments_i8_skylake,
                   nk_reduce_moments_i8_serial);
    run_if_matches("reduce_moments_u8_skylake", test_reduce_moments<u8_t>, nk_reduce_moments_u8_skylake,
                   nk_reduce_moments_u8_serial);
    run_if_matches("reduce_moments_i16_skylake", test_reduce_moments<i16_t>, nk_reduce_moments_i16_skylake,
                   nk_reduce_moments_i16_serial);
    run_if_matches("reduce_moments_u16_skylake", test_reduce_moments<u16_t>, nk_reduce_moments_u16_skylake,
                   nk_reduce_moments_u16_serial);
    run_if_matches("reduce_moments_i32_skylake", test_reduce_moments<i32_t>, nk_reduce_moments_i32_skylake,
                   nk_reduce_moments_i32_serial);
    run_if_matches("reduce_moments_u32_skylake", test_reduce_moments<u32_t>, nk_reduce_moments_u32_skylake,
                   nk_reduce_moments_u32_serial);
    run_if_matches("reduce_moments_i64_skylake", test_reduce_moments<i64_t>, nk_reduce_moments_i64_skylake,
                   nk_reduce_moments_i64_serial);
    run_if_matches("reduce_moments_u64_skylake", test_reduce_moments<u64_t>, nk_reduce_moments_u64_skylake,
                   nk_reduce_moments_u64_serial);
    run_if_matches("reduce_moments_e4m3_skylake", test_reduce_moments<e4m3_t>, nk_reduce_moments_e4m3_skylake,
                   nk_reduce_moments_e4m3_serial);
    run_if_matches("reduce_moments_e5m2_skylake", test_reduce_moments<e5m2_t>, nk_reduce_moments_e5m2_skylake,
                   nk_reduce_moments_e5m2_serial);
    run_if_matches("reduce_moments_e2m3_skylake", test_reduce_moments<e2m3_t>, nk_reduce_moments_e2m3_skylake,
                   nk_reduce_moments_e2m3_serial);
    run_if_matches("reduce_moments_e3m2_skylake", test_reduce_moments<e3m2_t>, nk_reduce_moments_e3m2_skylake,
                   nk_reduce_moments_e3m2_serial);
    run_if_matches("reduce_moments_i4_skylake", test_reduce_moments<i4x2_t>, nk_reduce_moments_i4_skylake,
                   nk_reduce_moments_i4_serial);
    run_if_matches("reduce_moments_u4_skylake", test_reduce_moments<u4x2_t>, nk_reduce_moments_u4_skylake,
                   nk_reduce_moments_u4_serial);
    run_if_matches("reduce_moments_u1_skylake", test_reduce_moments<u1x8_t>, nk_reduce_moments_u1_skylake,
                   nk_reduce_moments_u1_serial);
    run_if_matches("reduce_minmax_f32_skylake", test_reduce_minmax<f32_t>, nk_reduce_minmax_f32_skylake,
                   nk_reduce_minmax_f32_serial);
    run_if_matches("reduce_minmax_f64_skylake", test_reduce_minmax<f64_t>, nk_reduce_minmax_f64_skylake,
                   nk_reduce_minmax_f64_serial);
    run_if_matches("reduce_minmax_i8_skylake", test_reduce_minmax<i8_t>, nk_reduce_minmax_i8_skylake,
                   nk_reduce_minmax_i8_serial);
    run_if_matches("reduce_minmax_u8_skylake", test_reduce_minmax<u8_t>, nk_reduce_minmax_u8_skylake,
                   nk_reduce_minmax_u8_serial);
    run_if_matches("reduce_minmax_i16_skylake", test_reduce_minmax<i16_t>, nk_reduce_minmax_i16_skylake,
                   nk_reduce_minmax_i16_serial);
    run_if_matches("reduce_minmax_u16_skylake", test_reduce_minmax<u16_t>, nk_reduce_minmax_u16_skylake,
                   nk_reduce_minmax_u16_serial);
    run_if_matches("reduce_minmax_i32_skylake", test_reduce_minmax<i32_t>, nk_reduce_minmax_i32_skylake,
                   nk_reduce_minmax_i32_serial);
    run_if_matches("reduce_minmax_u32_skylake", test_reduce_minmax<u32_t>, nk_reduce_minmax_u32_skylake,
                   nk_reduce_minmax_u32_serial);
    run_if_matches("reduce_minmax_i64_skylake", test_reduce_minmax<i64_t>, nk_reduce_minmax_i64_skylake,
                   nk_reduce_minmax_i64_serial);
    run_if_matches("reduce_minmax_u64_skylake", test_reduce_minmax<u64_t>, nk_reduce_minmax_u64_skylake,
                   nk_reduce_minmax_u64_serial);
    run_if_matches("reduce_minmax_e4m3_skylake", test_reduce_minmax<e4m3_t>, nk_reduce_minmax_e4m3_skylake,
                   nk_reduce_minmax_e4m3_serial);
    run_if_matches("reduce_minmax_e5m2_skylake", test_reduce_minmax<e5m2_t>, nk_reduce_minmax_e5m2_skylake,
                   nk_reduce_minmax_e5m2_serial);
    run_if_matches("reduce_minmax_e2m3_skylake", test_reduce_minmax<e2m3_t>, nk_reduce_minmax_e2m3_skylake,
                   nk_reduce_minmax_e2m3_serial);
    run_if_matches("reduce_minmax_e3m2_skylake", test_reduce_minmax<e3m2_t>, nk_reduce_minmax_e3m2_skylake,
                   nk_reduce_minmax_e3m2_serial);
    run_if_matches("reduce_moments_bf16_skylake", test_reduce_moments<bf16_t>, nk_reduce_moments_bf16_skylake,
                   nk_reduce_moments_bf16_serial);
    run_if_matches("reduce_minmax_bf16_skylake", test_reduce_minmax<bf16_t>, nk_reduce_minmax_bf16_skylake,
                   nk_reduce_minmax_bf16_serial);
    run_if_matches("reduce_moments_f16_skylake", test_reduce_moments<f16_t>, nk_reduce_moments_f16_skylake,
                   nk_reduce_moments_f16_serial);
    run_if_matches("reduce_minmax_f16_skylake", test_reduce_minmax<f16_t>, nk_reduce_minmax_f16_skylake,
                   nk_reduce_minmax_f16_serial);
#endif

#if NK_TARGET_ICELAKE
    run_if_matches("reduce_moments_i8_icelake", test_reduce_moments<i8_t>, nk_reduce_moments_i8_icelake,
                   nk_reduce_moments_i8_serial);
    run_if_matches("reduce_moments_u8_icelake", test_reduce_moments<u8_t>, nk_reduce_moments_u8_icelake,
                   nk_reduce_moments_u8_serial);
    run_if_matches("reduce_moments_i16_icelake", test_reduce_moments<i16_t>, nk_reduce_moments_i16_icelake,
                   nk_reduce_moments_i16_serial);
    run_if_matches("reduce_moments_e2m3_icelake", test_reduce_moments<e2m3_t>, nk_reduce_moments_e2m3_icelake,
                   nk_reduce_moments_e2m3_serial);
    run_if_matches("reduce_moments_e3m2_icelake", test_reduce_moments<e3m2_t>, nk_reduce_moments_e3m2_icelake,
                   nk_reduce_moments_e3m2_serial);
#endif

#if NK_TARGET_GENOA
    run_if_matches("reduce_moments_bf16_genoa", test_reduce_moments<bf16_t>, nk_reduce_moments_bf16_genoa,
                   nk_reduce_moments_bf16_serial);
    run_if_matches("reduce_moments_e4m3_genoa", test_reduce_moments<e4m3_t>, nk_reduce_moments_e4m3_genoa,
                   nk_reduce_moments_e4m3_serial);
    run_if_matches("reduce_moments_e5m2_genoa", test_reduce_moments<e5m2_t>, nk_reduce_moments_e5m2_genoa,
                   nk_reduce_moments_e5m2_serial);
    run_if_matches("reduce_moments_e2m3_genoa", test_reduce_moments<e2m3_t>, nk_reduce_moments_e2m3_genoa,
                   nk_reduce_moments_e2m3_serial);
    run_if_matches("reduce_moments_e3m2_genoa", test_reduce_moments<e3m2_t>, nk_reduce_moments_e3m2_genoa,
                   nk_reduce_moments_e3m2_serial);
#endif

#if NK_TARGET_SIERRA
    run_if_matches("reduce_moments_e2m3_sierra", test_reduce_moments<e2m3_t>, nk_reduce_moments_e2m3_sierra,
                   nk_reduce_moments_e2m3_serial);
    run_if_matches("reduce_moments_e3m2_sierra", test_reduce_moments<e3m2_t>, nk_reduce_moments_e3m2_sierra,
                   nk_reduce_moments_e3m2_serial);
#endif

#if NK_TARGET_RVV
    run_if_matches("reduce_moments_f32_rvv", test_reduce_moments<f32_t>, nk_reduce_moments_f32_rvv,
                   nk_reduce_moments_f32_serial);
    run_if_matches("reduce_moments_f64_rvv", test_reduce_moments<f64_t>, nk_reduce_moments_f64_rvv,
                   nk_reduce_moments_f64_serial);
    run_if_matches("reduce_moments_i8_rvv", test_reduce_moments<i8_t>, nk_reduce_moments_i8_rvv,
                   nk_reduce_moments_i8_serial);
    run_if_matches("reduce_moments_u8_rvv", test_reduce_moments<u8_t>, nk_reduce_moments_u8_rvv,
                   nk_reduce_moments_u8_serial);
    run_if_matches("reduce_moments_i16_rvv", test_reduce_moments<i16_t>, nk_reduce_moments_i16_rvv,
                   nk_reduce_moments_i16_serial);
    run_if_matches("reduce_moments_u16_rvv", test_reduce_moments<u16_t>, nk_reduce_moments_u16_rvv,
                   nk_reduce_moments_u16_serial);
    run_if_matches("reduce_moments_i32_rvv", test_reduce_moments<i32_t>, nk_reduce_moments_i32_rvv,
                   nk_reduce_moments_i32_serial);
    run_if_matches("reduce_moments_u32_rvv", test_reduce_moments<u32_t>, nk_reduce_moments_u32_rvv,
                   nk_reduce_moments_u32_serial);
    run_if_matches("reduce_moments_i64_rvv", test_reduce_moments<i64_t>, nk_reduce_moments_i64_rvv,
                   nk_reduce_moments_i64_serial);
    run_if_matches("reduce_moments_u64_rvv", test_reduce_moments<u64_t>, nk_reduce_moments_u64_rvv,
                   nk_reduce_moments_u64_serial);
    run_if_matches("reduce_moments_bf16_rvv", test_reduce_moments<bf16_t>, nk_reduce_moments_bf16_rvv,
                   nk_reduce_moments_bf16_serial);
    run_if_matches("reduce_moments_f16_rvv", test_reduce_moments<f16_t>, nk_reduce_moments_f16_rvv,
                   nk_reduce_moments_f16_serial);
    run_if_matches("reduce_moments_e4m3_rvv", test_reduce_moments<e4m3_t>, nk_reduce_moments_e4m3_rvv,
                   nk_reduce_moments_e4m3_serial);
    run_if_matches("reduce_moments_e5m2_rvv", test_reduce_moments<e5m2_t>, nk_reduce_moments_e5m2_rvv,
                   nk_reduce_moments_e5m2_serial);
    run_if_matches("reduce_moments_e2m3_rvv", test_reduce_moments<e2m3_t>, nk_reduce_moments_e2m3_rvv,
                   nk_reduce_moments_e2m3_serial);
    run_if_matches("reduce_moments_e3m2_rvv", test_reduce_moments<e3m2_t>, nk_reduce_moments_e3m2_rvv,
                   nk_reduce_moments_e3m2_serial);
    run_if_matches("reduce_minmax_f32_rvv", test_reduce_minmax<f32_t>, nk_reduce_minmax_f32_rvv,
                   nk_reduce_minmax_f32_serial);
    run_if_matches("reduce_minmax_f64_rvv", test_reduce_minmax<f64_t>, nk_reduce_minmax_f64_rvv,
                   nk_reduce_minmax_f64_serial);
    run_if_matches("reduce_minmax_i8_rvv", test_reduce_minmax<i8_t>, nk_reduce_minmax_i8_rvv,
                   nk_reduce_minmax_i8_serial);
    run_if_matches("reduce_minmax_u8_rvv", test_reduce_minmax<u8_t>, nk_reduce_minmax_u8_rvv,
                   nk_reduce_minmax_u8_serial);
    run_if_matches("reduce_minmax_i16_rvv", test_reduce_minmax<i16_t>, nk_reduce_minmax_i16_rvv,
                   nk_reduce_minmax_i16_serial);
    run_if_matches("reduce_minmax_u16_rvv", test_reduce_minmax<u16_t>, nk_reduce_minmax_u16_rvv,
                   nk_reduce_minmax_u16_serial);
    run_if_matches("reduce_minmax_i32_rvv", test_reduce_minmax<i32_t>, nk_reduce_minmax_i32_rvv,
                   nk_reduce_minmax_i32_serial);
    run_if_matches("reduce_minmax_u32_rvv", test_reduce_minmax<u32_t>, nk_reduce_minmax_u32_rvv,
                   nk_reduce_minmax_u32_serial);
    run_if_matches("reduce_minmax_i64_rvv", test_reduce_minmax<i64_t>, nk_reduce_minmax_i64_rvv,
                   nk_reduce_minmax_i64_serial);
    run_if_matches("reduce_minmax_u64_rvv", test_reduce_minmax<u64_t>, nk_reduce_minmax_u64_rvv,
                   nk_reduce_minmax_u64_serial);
    run_if_matches("reduce_minmax_bf16_rvv", test_reduce_minmax<bf16_t>, nk_reduce_minmax_bf16_rvv,
                   nk_reduce_minmax_bf16_serial);
    run_if_matches("reduce_minmax_f16_rvv", test_reduce_minmax<f16_t>, nk_reduce_minmax_f16_rvv,
                   nk_reduce_minmax_f16_serial);
    run_if_matches("reduce_minmax_e4m3_rvv", test_reduce_minmax<e4m3_t>, nk_reduce_minmax_e4m3_rvv,
                   nk_reduce_minmax_e4m3_serial);
    run_if_matches("reduce_minmax_e5m2_rvv", test_reduce_minmax<e5m2_t>, nk_reduce_minmax_e5m2_rvv,
                   nk_reduce_minmax_e5m2_serial);
    run_if_matches("reduce_minmax_e2m3_rvv", test_reduce_minmax<e2m3_t>, nk_reduce_minmax_e2m3_rvv,
                   nk_reduce_minmax_e2m3_serial);
    run_if_matches("reduce_minmax_e3m2_rvv", test_reduce_minmax<e3m2_t>, nk_reduce_minmax_e3m2_rvv,
                   nk_reduce_minmax_e3m2_serial);
#endif
    run_if_matches("reduce_moments_f32_serial", test_reduce_moments<f32_t>, nk_reduce_moments_f32_serial,
                   nk_reduce_moments_f32_serial);
    run_if_matches("reduce_moments_f64_serial", test_reduce_moments<f64_t>, nk_reduce_moments_f64_serial,
                   nk_reduce_moments_f64_serial);
    run_if_matches("reduce_moments_i8_serial", test_reduce_moments<i8_t>, nk_reduce_moments_i8_serial,
                   nk_reduce_moments_i8_serial);
    run_if_matches("reduce_moments_u8_serial", test_reduce_moments<u8_t>, nk_reduce_moments_u8_serial,
                   nk_reduce_moments_u8_serial);
    run_if_matches("reduce_moments_i16_serial", test_reduce_moments<i16_t>, nk_reduce_moments_i16_serial,
                   nk_reduce_moments_i16_serial);
    run_if_matches("reduce_moments_u16_serial", test_reduce_moments<u16_t>, nk_reduce_moments_u16_serial,
                   nk_reduce_moments_u16_serial);
    run_if_matches("reduce_moments_i32_serial", test_reduce_moments<i32_t>, nk_reduce_moments_i32_serial,
                   nk_reduce_moments_i32_serial);
    run_if_matches("reduce_moments_u32_serial", test_reduce_moments<u32_t>, nk_reduce_moments_u32_serial,
                   nk_reduce_moments_u32_serial);
    run_if_matches("reduce_moments_i64_serial", test_reduce_moments<i64_t>, nk_reduce_moments_i64_serial,
                   nk_reduce_moments_i64_serial);
    run_if_matches("reduce_moments_u64_serial", test_reduce_moments<u64_t>, nk_reduce_moments_u64_serial,
                   nk_reduce_moments_u64_serial);
    run_if_matches("reduce_moments_f16_serial", test_reduce_moments<f16_t>, nk_reduce_moments_f16_serial,
                   nk_reduce_moments_f16_serial);
    run_if_matches("reduce_moments_bf16_serial", test_reduce_moments<bf16_t>, nk_reduce_moments_bf16_serial,
                   nk_reduce_moments_bf16_serial);
    run_if_matches("reduce_moments_e4m3_serial", test_reduce_moments<e4m3_t>, nk_reduce_moments_e4m3_serial,
                   nk_reduce_moments_e4m3_serial);
    run_if_matches("reduce_moments_e5m2_serial", test_reduce_moments<e5m2_t>, nk_reduce_moments_e5m2_serial,
                   nk_reduce_moments_e5m2_serial);
    run_if_matches("reduce_moments_e2m3_serial", test_reduce_moments<e2m3_t>, nk_reduce_moments_e2m3_serial,
                   nk_reduce_moments_e2m3_serial);
    run_if_matches("reduce_moments_e3m2_serial", test_reduce_moments<e3m2_t>, nk_reduce_moments_e3m2_serial,
                   nk_reduce_moments_e3m2_serial);
    run_if_matches("reduce_moments_i4_serial", test_reduce_moments<i4x2_t>, nk_reduce_moments_i4_serial,
                   nk_reduce_moments_i4_serial);
    run_if_matches("reduce_moments_u4_serial", test_reduce_moments<u4x2_t>, nk_reduce_moments_u4_serial,
                   nk_reduce_moments_u4_serial);
    run_if_matches("reduce_moments_u1_serial", test_reduce_moments<u1x8_t>, nk_reduce_moments_u1_serial,
                   nk_reduce_moments_u1_serial);
    run_if_matches("reduce_minmax_f32_serial", test_reduce_minmax<f32_t>, nk_reduce_minmax_f32_serial,
                   nk_reduce_minmax_f32_serial);
    run_if_matches("reduce_minmax_f64_serial", test_reduce_minmax<f64_t>, nk_reduce_minmax_f64_serial,
                   nk_reduce_minmax_f64_serial);
    run_if_matches("reduce_minmax_i8_serial", test_reduce_minmax<i8_t>, nk_reduce_minmax_i8_serial,
                   nk_reduce_minmax_i8_serial);
    run_if_matches("reduce_minmax_u8_serial", test_reduce_minmax<u8_t>, nk_reduce_minmax_u8_serial,
                   nk_reduce_minmax_u8_serial);
    run_if_matches("reduce_minmax_i16_serial", test_reduce_minmax<i16_t>, nk_reduce_minmax_i16_serial,
                   nk_reduce_minmax_i16_serial);
    run_if_matches("reduce_minmax_u16_serial", test_reduce_minmax<u16_t>, nk_reduce_minmax_u16_serial,
                   nk_reduce_minmax_u16_serial);
    run_if_matches("reduce_minmax_i32_serial", test_reduce_minmax<i32_t>, nk_reduce_minmax_i32_serial,
                   nk_reduce_minmax_i32_serial);
    run_if_matches("reduce_minmax_u32_serial", test_reduce_minmax<u32_t>, nk_reduce_minmax_u32_serial,
                   nk_reduce_minmax_u32_serial);
    run_if_matches("reduce_minmax_i64_serial", test_reduce_minmax<i64_t>, nk_reduce_minmax_i64_serial,
                   nk_reduce_minmax_i64_serial);
    run_if_matches("reduce_minmax_u64_serial", test_reduce_minmax<u64_t>, nk_reduce_minmax_u64_serial,
                   nk_reduce_minmax_u64_serial);
    run_if_matches("reduce_minmax_f16_serial", test_reduce_minmax<f16_t>, nk_reduce_minmax_f16_serial,
                   nk_reduce_minmax_f16_serial);
    run_if_matches("reduce_minmax_bf16_serial", test_reduce_minmax<bf16_t>, nk_reduce_minmax_bf16_serial,
                   nk_reduce_minmax_bf16_serial);
    run_if_matches("reduce_minmax_e4m3_serial", test_reduce_minmax<e4m3_t>, nk_reduce_minmax_e4m3_serial,
                   nk_reduce_minmax_e4m3_serial);
    run_if_matches("reduce_minmax_e5m2_serial", test_reduce_minmax<e5m2_t>, nk_reduce_minmax_e5m2_serial,
                   nk_reduce_minmax_e5m2_serial);
    run_if_matches("reduce_minmax_e2m3_serial", test_reduce_minmax<e2m3_t>, nk_reduce_minmax_e2m3_serial,
                   nk_reduce_minmax_e2m3_serial);
    run_if_matches("reduce_minmax_e3m2_serial", test_reduce_minmax<e3m2_t>, nk_reduce_minmax_e3m2_serial,
                   nk_reduce_minmax_e3m2_serial);
    run_if_matches("reduce_minmax_i4_serial", test_reduce_minmax<i4x2_t>, nk_reduce_minmax_i4_serial,
                   nk_reduce_minmax_i4_serial);
    run_if_matches("reduce_minmax_u4_serial", test_reduce_minmax<u4x2_t>, nk_reduce_minmax_u4_serial,
                   nk_reduce_minmax_u4_serial);
    run_if_matches("reduce_minmax_u1_serial", test_reduce_minmax<u1x8_t>, nk_reduce_minmax_u1_serial,
                   nk_reduce_minmax_u1_serial);
#endif
}

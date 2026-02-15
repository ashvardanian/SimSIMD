/**
 *  @brief Reduction tests.
 *  @file test/test_reduce.cpp
 *  @author Ash Vardanian
 *  @date February 6, 2026
 */

#include "test.hpp"
#include "numkong/reduce.hpp"
#include "numkong/reduce/serial.h"

template <typename input_t_, typename sum_acc_t_, typename sumsq_acc_t_ = sum_acc_t_>
error_stats_t test_reduce_moments_(void (*kernel)(typename input_t_::raw_t const *, nk_size_t, nk_size_t,
                                                  typename sum_acc_t_::raw_t *, typename sumsq_acc_t_::raw_t *),
                                   void (*reference)(typename input_t_::raw_t const *, nk_size_t, nk_size_t,
                                                     typename sum_acc_t_::raw_t *, typename sumsq_acc_t_::raw_t *)) {
    error_stats_t stats;
    std::mt19937 generator(global_config.seed);
    auto buffer = make_vector<input_t_>(dense_dimensions);
    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(generator, buffer);
        typename sum_acc_t_::raw_t sum;
        typename sumsq_acc_t_::raw_t sumsq;
        kernel(buffer.raw_values_data(), dense_dimensions, sizeof(typename input_t_::raw_t), &sum, &sumsq);
        typename sum_acc_t_::raw_t ref_sum;
        typename sumsq_acc_t_::raw_t ref_sumsq;
        reference(buffer.raw_values_data(), dense_dimensions, sizeof(typename input_t_::raw_t), &ref_sum, &ref_sumsq);
        stats.accumulate(sum_acc_t_::from_raw(sum), sum_acc_t_::from_raw(ref_sum));
        stats.accumulate(sumsq_acc_t_::from_raw(sumsq), sumsq_acc_t_::from_raw(ref_sumsq));
    }
    return stats;
}

template <typename input_t_, typename output_t_>
error_stats_t test_reduce_minmax_(void (*kernel)(typename input_t_::raw_t const *, nk_size_t, nk_size_t,
                                                 typename output_t_::raw_t *, nk_size_t *, typename output_t_::raw_t *,
                                                 nk_size_t *),
                                  void (*reference)(typename input_t_::raw_t const *, nk_size_t, nk_size_t,
                                                    typename output_t_::raw_t *, nk_size_t *,
                                                    typename output_t_::raw_t *, nk_size_t *)) {
    error_stats_t stats;
    std::mt19937 generator(global_config.seed);
    auto buffer = make_vector<input_t_>(dense_dimensions);
    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(generator, buffer);
        typename output_t_::raw_t min_val, max_val;
        nk_size_t min_idx, max_idx;
        kernel(buffer.raw_values_data(), dense_dimensions, sizeof(typename input_t_::raw_t), &min_val, &min_idx,
               &max_val, &max_idx);
        typename output_t_::raw_t ref_min, ref_max;
        nk_size_t ref_min_idx, ref_max_idx;
        reference(buffer.raw_values_data(), dense_dimensions, sizeof(typename input_t_::raw_t), &ref_min, &ref_min_idx,
                  &ref_max, &ref_max_idx);
        stats.accumulate(output_t_::from_raw(min_val), output_t_::from_raw(ref_min));
        stats.accumulate(output_t_::from_raw(max_val), output_t_::from_raw(ref_max));
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
    run_if_matches("reduce_moments_f32_haswell", test_reduce_moments_<f32_t, f64_t>, nk_reduce_moments_f32_haswell,
                   nk_reduce_moments_f32_serial);
    run_if_matches("reduce_moments_f64_haswell", test_reduce_moments_<f64_t, f64_t>, nk_reduce_moments_f64_haswell,
                   nk_reduce_moments_f64_serial);
    run_if_matches("reduce_moments_i8_haswell", test_reduce_moments_<i8_t, i64_t, u64_t>, nk_reduce_moments_i8_haswell,
                   nk_reduce_moments_i8_serial);
    run_if_matches("reduce_moments_u8_haswell", test_reduce_moments_<u8_t, u64_t>, nk_reduce_moments_u8_haswell,
                   nk_reduce_moments_u8_serial);
    run_if_matches("reduce_moments_i16_haswell", test_reduce_moments_<i16_t, i64_t, u64_t>,
                   nk_reduce_moments_i16_haswell, nk_reduce_moments_i16_serial);
    run_if_matches("reduce_moments_u16_haswell", test_reduce_moments_<u16_t, u64_t>, nk_reduce_moments_u16_haswell,
                   nk_reduce_moments_u16_serial);
    run_if_matches("reduce_moments_i32_haswell", test_reduce_moments_<i32_t, i64_t, u64_t>,
                   nk_reduce_moments_i32_haswell, nk_reduce_moments_i32_serial);
    run_if_matches("reduce_moments_u32_haswell", test_reduce_moments_<u32_t, u64_t>, nk_reduce_moments_u32_haswell,
                   nk_reduce_moments_u32_serial);
    run_if_matches("reduce_moments_i64_haswell", test_reduce_moments_<i64_t, i64_t, u64_t>,
                   nk_reduce_moments_i64_haswell, nk_reduce_moments_i64_serial);
    run_if_matches("reduce_moments_u64_haswell", test_reduce_moments_<u64_t, u64_t>, nk_reduce_moments_u64_haswell,
                   nk_reduce_moments_u64_serial);
    run_if_matches("reduce_moments_e4m3_haswell", test_reduce_moments_<e4m3_t, f32_t>, nk_reduce_moments_e4m3_haswell,
                   nk_reduce_moments_e4m3_serial);
    run_if_matches("reduce_moments_e5m2_haswell", test_reduce_moments_<e5m2_t, f32_t>, nk_reduce_moments_e5m2_haswell,
                   nk_reduce_moments_e5m2_serial);
    run_if_matches("reduce_moments_e2m3_haswell", test_reduce_moments_<e2m3_t, f32_t>, nk_reduce_moments_e2m3_haswell,
                   nk_reduce_moments_e2m3_serial);
    run_if_matches("reduce_moments_e3m2_haswell", test_reduce_moments_<e3m2_t, f32_t>, nk_reduce_moments_e3m2_haswell,
                   nk_reduce_moments_e3m2_serial);
    run_if_matches("reduce_moments_i4_haswell", test_reduce_moments_<i4x2_t, i64_t, u64_t>,
                   nk_reduce_moments_i4_haswell, nk_reduce_moments_i4_serial);
    run_if_matches("reduce_moments_u4_haswell", test_reduce_moments_<u4x2_t, u64_t>, nk_reduce_moments_u4_haswell,
                   nk_reduce_moments_u4_serial);
    run_if_matches("reduce_moments_u1_haswell", test_reduce_moments_<u1x8_t, u64_t>, nk_reduce_moments_u1_haswell,
                   nk_reduce_moments_u1_serial);
    run_if_matches("reduce_moments_bf16_haswell", test_reduce_moments_<bf16_t, f32_t>, nk_reduce_moments_bf16_haswell,
                   nk_reduce_moments_bf16_serial);
    run_if_matches("reduce_moments_f16_haswell", test_reduce_moments_<f16_t, f32_t>, nk_reduce_moments_f16_haswell,
                   nk_reduce_moments_f16_serial);
    run_if_matches("reduce_minmax_f32_haswell", test_reduce_minmax_<f32_t, f32_t>, nk_reduce_minmax_f32_haswell,
                   nk_reduce_minmax_f32_serial);
    run_if_matches("reduce_minmax_f64_haswell", test_reduce_minmax_<f64_t, f64_t>, nk_reduce_minmax_f64_haswell,
                   nk_reduce_minmax_f64_serial);
    run_if_matches("reduce_minmax_i8_haswell", test_reduce_minmax_<i8_t, i8_t>, nk_reduce_minmax_i8_haswell,
                   nk_reduce_minmax_i8_serial);
    run_if_matches("reduce_minmax_u8_haswell", test_reduce_minmax_<u8_t, u8_t>, nk_reduce_minmax_u8_haswell,
                   nk_reduce_minmax_u8_serial);
    run_if_matches("reduce_minmax_i16_haswell", test_reduce_minmax_<i16_t, i16_t>, nk_reduce_minmax_i16_haswell,
                   nk_reduce_minmax_i16_serial);
    run_if_matches("reduce_minmax_u16_haswell", test_reduce_minmax_<u16_t, u16_t>, nk_reduce_minmax_u16_haswell,
                   nk_reduce_minmax_u16_serial);
    run_if_matches("reduce_minmax_i32_haswell", test_reduce_minmax_<i32_t, i32_t>, nk_reduce_minmax_i32_haswell,
                   nk_reduce_minmax_i32_serial);
    run_if_matches("reduce_minmax_u32_haswell", test_reduce_minmax_<u32_t, u32_t>, nk_reduce_minmax_u32_haswell,
                   nk_reduce_minmax_u32_serial);
    run_if_matches("reduce_minmax_i64_haswell", test_reduce_minmax_<i64_t, i64_t>, nk_reduce_minmax_i64_haswell,
                   nk_reduce_minmax_i64_serial);
    run_if_matches("reduce_minmax_u64_haswell", test_reduce_minmax_<u64_t, u64_t>, nk_reduce_minmax_u64_haswell,
                   nk_reduce_minmax_u64_serial);
    run_if_matches("reduce_minmax_e4m3_haswell", test_reduce_minmax_<e4m3_t, e4m3_t>, nk_reduce_minmax_e4m3_haswell,
                   nk_reduce_minmax_e4m3_serial);
    run_if_matches("reduce_minmax_e5m2_haswell", test_reduce_minmax_<e5m2_t, e5m2_t>, nk_reduce_minmax_e5m2_haswell,
                   nk_reduce_minmax_e5m2_serial);
    run_if_matches("reduce_minmax_e2m3_haswell", test_reduce_minmax_<e2m3_t, e2m3_t>, nk_reduce_minmax_e2m3_haswell,
                   nk_reduce_minmax_e2m3_serial);
    run_if_matches("reduce_minmax_e3m2_haswell", test_reduce_minmax_<e3m2_t, e3m2_t>, nk_reduce_minmax_e3m2_haswell,
                   nk_reduce_minmax_e3m2_serial);
    run_if_matches("reduce_minmax_bf16_haswell", test_reduce_minmax_<bf16_t, bf16_t>, nk_reduce_minmax_bf16_haswell,
                   nk_reduce_minmax_bf16_serial);
    run_if_matches("reduce_minmax_f16_haswell", test_reduce_minmax_<f16_t, f16_t>, nk_reduce_minmax_f16_haswell,
                   nk_reduce_minmax_f16_serial);
#endif
#if NK_TARGET_SKYLAKE
    run_if_matches("reduce_moments_f32_skylake", test_reduce_moments_<f32_t, f64_t>, nk_reduce_moments_f32_skylake,
                   nk_reduce_moments_f32_serial);
    run_if_matches("reduce_moments_f64_skylake", test_reduce_moments_<f64_t, f64_t>, nk_reduce_moments_f64_skylake,
                   nk_reduce_moments_f64_serial);
    run_if_matches("reduce_moments_i8_skylake", test_reduce_moments_<i8_t, i64_t, u64_t>, nk_reduce_moments_i8_skylake,
                   nk_reduce_moments_i8_serial);
    run_if_matches("reduce_moments_u8_skylake", test_reduce_moments_<u8_t, u64_t>, nk_reduce_moments_u8_skylake,
                   nk_reduce_moments_u8_serial);
    run_if_matches("reduce_moments_i16_skylake", test_reduce_moments_<i16_t, i64_t, u64_t>,
                   nk_reduce_moments_i16_skylake, nk_reduce_moments_i16_serial);
    run_if_matches("reduce_moments_u16_skylake", test_reduce_moments_<u16_t, u64_t>, nk_reduce_moments_u16_skylake,
                   nk_reduce_moments_u16_serial);
    run_if_matches("reduce_moments_i32_skylake", test_reduce_moments_<i32_t, i64_t, u64_t>,
                   nk_reduce_moments_i32_skylake, nk_reduce_moments_i32_serial);
    run_if_matches("reduce_moments_u32_skylake", test_reduce_moments_<u32_t, u64_t>, nk_reduce_moments_u32_skylake,
                   nk_reduce_moments_u32_serial);
    run_if_matches("reduce_moments_i64_skylake", test_reduce_moments_<i64_t, i64_t, u64_t>,
                   nk_reduce_moments_i64_skylake, nk_reduce_moments_i64_serial);
    run_if_matches("reduce_moments_u64_skylake", test_reduce_moments_<u64_t, u64_t>, nk_reduce_moments_u64_skylake,
                   nk_reduce_moments_u64_serial);
    run_if_matches("reduce_moments_e4m3_skylake", test_reduce_moments_<e4m3_t, f32_t>, nk_reduce_moments_e4m3_skylake,
                   nk_reduce_moments_e4m3_serial);
    run_if_matches("reduce_moments_e5m2_skylake", test_reduce_moments_<e5m2_t, f32_t>, nk_reduce_moments_e5m2_skylake,
                   nk_reduce_moments_e5m2_serial);
    run_if_matches("reduce_moments_e2m3_skylake", test_reduce_moments_<e2m3_t, f32_t>, nk_reduce_moments_e2m3_skylake,
                   nk_reduce_moments_e2m3_serial);
    run_if_matches("reduce_moments_e3m2_skylake", test_reduce_moments_<e3m2_t, f32_t>, nk_reduce_moments_e3m2_skylake,
                   nk_reduce_moments_e3m2_serial);
    run_if_matches("reduce_moments_i4_skylake", test_reduce_moments_<i4x2_t, i64_t, u64_t>,
                   nk_reduce_moments_i4_skylake, nk_reduce_moments_i4_serial);
    run_if_matches("reduce_moments_u4_skylake", test_reduce_moments_<u4x2_t, u64_t>, nk_reduce_moments_u4_skylake,
                   nk_reduce_moments_u4_serial);
    run_if_matches("reduce_moments_u1_skylake", test_reduce_moments_<u1x8_t, u64_t>, nk_reduce_moments_u1_skylake,
                   nk_reduce_moments_u1_serial);
    run_if_matches("reduce_minmax_f32_skylake", test_reduce_minmax_<f32_t, f32_t>, nk_reduce_minmax_f32_skylake,
                   nk_reduce_minmax_f32_serial);
    run_if_matches("reduce_minmax_f64_skylake", test_reduce_minmax_<f64_t, f64_t>, nk_reduce_minmax_f64_skylake,
                   nk_reduce_minmax_f64_serial);
    run_if_matches("reduce_minmax_i8_skylake", test_reduce_minmax_<i8_t, i8_t>, nk_reduce_minmax_i8_skylake,
                   nk_reduce_minmax_i8_serial);
    run_if_matches("reduce_minmax_u8_skylake", test_reduce_minmax_<u8_t, u8_t>, nk_reduce_minmax_u8_skylake,
                   nk_reduce_minmax_u8_serial);
    run_if_matches("reduce_minmax_i16_skylake", test_reduce_minmax_<i16_t, i16_t>, nk_reduce_minmax_i16_skylake,
                   nk_reduce_minmax_i16_serial);
    run_if_matches("reduce_minmax_u16_skylake", test_reduce_minmax_<u16_t, u16_t>, nk_reduce_minmax_u16_skylake,
                   nk_reduce_minmax_u16_serial);
    run_if_matches("reduce_minmax_i32_skylake", test_reduce_minmax_<i32_t, i32_t>, nk_reduce_minmax_i32_skylake,
                   nk_reduce_minmax_i32_serial);
    run_if_matches("reduce_minmax_u32_skylake", test_reduce_minmax_<u32_t, u32_t>, nk_reduce_minmax_u32_skylake,
                   nk_reduce_minmax_u32_serial);
    run_if_matches("reduce_minmax_i64_skylake", test_reduce_minmax_<i64_t, i64_t>, nk_reduce_minmax_i64_skylake,
                   nk_reduce_minmax_i64_serial);
    run_if_matches("reduce_minmax_u64_skylake", test_reduce_minmax_<u64_t, u64_t>, nk_reduce_minmax_u64_skylake,
                   nk_reduce_minmax_u64_serial);
    run_if_matches("reduce_minmax_e4m3_skylake", test_reduce_minmax_<e4m3_t, e4m3_t>, nk_reduce_minmax_e4m3_skylake,
                   nk_reduce_minmax_e4m3_serial);
    run_if_matches("reduce_minmax_e5m2_skylake", test_reduce_minmax_<e5m2_t, e5m2_t>, nk_reduce_minmax_e5m2_skylake,
                   nk_reduce_minmax_e5m2_serial);
    run_if_matches("reduce_minmax_e2m3_skylake", test_reduce_minmax_<e2m3_t, e2m3_t>, nk_reduce_minmax_e2m3_skylake,
                   nk_reduce_minmax_e2m3_serial);
    run_if_matches("reduce_minmax_e3m2_skylake", test_reduce_minmax_<e3m2_t, e3m2_t>, nk_reduce_minmax_e3m2_skylake,
                   nk_reduce_minmax_e3m2_serial);
    run_if_matches("reduce_moments_bf16_skylake", test_reduce_moments_<bf16_t, f32_t>, nk_reduce_moments_bf16_skylake,
                   nk_reduce_moments_bf16_serial);
    run_if_matches("reduce_minmax_bf16_skylake", test_reduce_minmax_<bf16_t, bf16_t>, nk_reduce_minmax_bf16_skylake,
                   nk_reduce_minmax_bf16_serial);
    run_if_matches("reduce_moments_f16_skylake", test_reduce_moments_<f16_t, f32_t>, nk_reduce_moments_f16_skylake,
                   nk_reduce_moments_f16_serial);
    run_if_matches("reduce_minmax_f16_skylake", test_reduce_minmax_<f16_t, f16_t>, nk_reduce_minmax_f16_skylake,
                   nk_reduce_minmax_f16_serial);
#endif
#if NK_TARGET_ICELAKE
    run_if_matches("reduce_moments_i8_icelake", test_reduce_moments_<i8_t, i64_t, u64_t>, nk_reduce_moments_i8_icelake,
                   nk_reduce_moments_i8_serial);
    run_if_matches("reduce_moments_u8_icelake", test_reduce_moments_<u8_t, u64_t>, nk_reduce_moments_u8_icelake,
                   nk_reduce_moments_u8_serial);
    run_if_matches("reduce_moments_i16_icelake", test_reduce_moments_<i16_t, i64_t, u64_t>,
                   nk_reduce_moments_i16_icelake, nk_reduce_moments_i16_serial);
    run_if_matches("reduce_moments_e2m3_icelake", test_reduce_moments_<e2m3_t, f32_t>, nk_reduce_moments_e2m3_icelake,
                   nk_reduce_moments_e2m3_serial);
    run_if_matches("reduce_moments_e3m2_icelake", test_reduce_moments_<e3m2_t, f32_t>, nk_reduce_moments_e3m2_icelake,
                   nk_reduce_moments_e3m2_serial);
#endif
#if NK_TARGET_GENOA
    run_if_matches("reduce_moments_bf16_genoa", test_reduce_moments_<bf16_t, f32_t>, nk_reduce_moments_bf16_genoa,
                   nk_reduce_moments_bf16_serial);
    run_if_matches("reduce_moments_e4m3_genoa", test_reduce_moments_<e4m3_t, f32_t>, nk_reduce_moments_e4m3_genoa,
                   nk_reduce_moments_e4m3_serial);
    run_if_matches("reduce_moments_e5m2_genoa", test_reduce_moments_<e5m2_t, f32_t>, nk_reduce_moments_e5m2_genoa,
                   nk_reduce_moments_e5m2_serial);
    run_if_matches("reduce_moments_e2m3_genoa", test_reduce_moments_<e2m3_t, f32_t>, nk_reduce_moments_e2m3_genoa,
                   nk_reduce_moments_e2m3_serial);
    run_if_matches("reduce_moments_e3m2_genoa", test_reduce_moments_<e3m2_t, f32_t>, nk_reduce_moments_e3m2_genoa,
                   nk_reduce_moments_e3m2_serial);
#endif
#if NK_TARGET_SIERRA
    run_if_matches("reduce_moments_e2m3_sierra", test_reduce_moments_<e2m3_t, f32_t>, nk_reduce_moments_e2m3_sierra,
                   nk_reduce_moments_e2m3_serial);
    run_if_matches("reduce_moments_e3m2_sierra", test_reduce_moments_<e3m2_t, f32_t>, nk_reduce_moments_e3m2_sierra,
                   nk_reduce_moments_e3m2_serial);
#endif
#if NK_TARGET_RVV
    run_if_matches("reduce_moments_f32_rvv", test_reduce_moments_<f32_t, f64_t>, nk_reduce_moments_f32_rvv,
                   nk_reduce_moments_f32_serial);
    run_if_matches("reduce_moments_f64_rvv", test_reduce_moments_<f64_t, f64_t>, nk_reduce_moments_f64_rvv,
                   nk_reduce_moments_f64_serial);
    run_if_matches("reduce_moments_i8_rvv", test_reduce_moments_<i8_t, i64_t, u64_t>, nk_reduce_moments_i8_rvv,
                   nk_reduce_moments_i8_serial);
    run_if_matches("reduce_moments_u8_rvv", test_reduce_moments_<u8_t, u64_t>, nk_reduce_moments_u8_rvv,
                   nk_reduce_moments_u8_serial);
    run_if_matches("reduce_moments_i16_rvv", test_reduce_moments_<i16_t, i64_t, u64_t>, nk_reduce_moments_i16_rvv,
                   nk_reduce_moments_i16_serial);
    run_if_matches("reduce_moments_u16_rvv", test_reduce_moments_<u16_t, u64_t>, nk_reduce_moments_u16_rvv,
                   nk_reduce_moments_u16_serial);
    run_if_matches("reduce_moments_i32_rvv", test_reduce_moments_<i32_t, i64_t, u64_t>, nk_reduce_moments_i32_rvv,
                   nk_reduce_moments_i32_serial);
    run_if_matches("reduce_moments_u32_rvv", test_reduce_moments_<u32_t, u64_t>, nk_reduce_moments_u32_rvv,
                   nk_reduce_moments_u32_serial);
    run_if_matches("reduce_moments_i64_rvv", test_reduce_moments_<i64_t, i64_t, u64_t>, nk_reduce_moments_i64_rvv,
                   nk_reduce_moments_i64_serial);
    run_if_matches("reduce_moments_u64_rvv", test_reduce_moments_<u64_t, u64_t>, nk_reduce_moments_u64_rvv,
                   nk_reduce_moments_u64_serial);
    run_if_matches("reduce_moments_bf16_rvv", test_reduce_moments_<bf16_t, f32_t>, nk_reduce_moments_bf16_rvv,
                   nk_reduce_moments_bf16_serial);
    run_if_matches("reduce_moments_f16_rvv", test_reduce_moments_<f16_t, f32_t>, nk_reduce_moments_f16_rvv,
                   nk_reduce_moments_f16_serial);
    run_if_matches("reduce_moments_e4m3_rvv", test_reduce_moments_<e4m3_t, f32_t>, nk_reduce_moments_e4m3_rvv,
                   nk_reduce_moments_e4m3_serial);
    run_if_matches("reduce_moments_e5m2_rvv", test_reduce_moments_<e5m2_t, f32_t>, nk_reduce_moments_e5m2_rvv,
                   nk_reduce_moments_e5m2_serial);
    run_if_matches("reduce_moments_e2m3_rvv", test_reduce_moments_<e2m3_t, f32_t>, nk_reduce_moments_e2m3_rvv,
                   nk_reduce_moments_e2m3_serial);
    run_if_matches("reduce_moments_e3m2_rvv", test_reduce_moments_<e3m2_t, f32_t>, nk_reduce_moments_e3m2_rvv,
                   nk_reduce_moments_e3m2_serial);
    run_if_matches("reduce_minmax_f32_rvv", test_reduce_minmax_<f32_t, f32_t>, nk_reduce_minmax_f32_rvv,
                   nk_reduce_minmax_f32_serial);
    run_if_matches("reduce_minmax_f64_rvv", test_reduce_minmax_<f64_t, f64_t>, nk_reduce_minmax_f64_rvv,
                   nk_reduce_minmax_f64_serial);
    run_if_matches("reduce_minmax_i8_rvv", test_reduce_minmax_<i8_t, i8_t>, nk_reduce_minmax_i8_rvv,
                   nk_reduce_minmax_i8_serial);
    run_if_matches("reduce_minmax_u8_rvv", test_reduce_minmax_<u8_t, u8_t>, nk_reduce_minmax_u8_rvv,
                   nk_reduce_minmax_u8_serial);
    run_if_matches("reduce_minmax_i16_rvv", test_reduce_minmax_<i16_t, i16_t>, nk_reduce_minmax_i16_rvv,
                   nk_reduce_minmax_i16_serial);
    run_if_matches("reduce_minmax_u16_rvv", test_reduce_minmax_<u16_t, u16_t>, nk_reduce_minmax_u16_rvv,
                   nk_reduce_minmax_u16_serial);
    run_if_matches("reduce_minmax_i32_rvv", test_reduce_minmax_<i32_t, i32_t>, nk_reduce_minmax_i32_rvv,
                   nk_reduce_minmax_i32_serial);
    run_if_matches("reduce_minmax_u32_rvv", test_reduce_minmax_<u32_t, u32_t>, nk_reduce_minmax_u32_rvv,
                   nk_reduce_minmax_u32_serial);
    run_if_matches("reduce_minmax_i64_rvv", test_reduce_minmax_<i64_t, i64_t>, nk_reduce_minmax_i64_rvv,
                   nk_reduce_minmax_i64_serial);
    run_if_matches("reduce_minmax_u64_rvv", test_reduce_minmax_<u64_t, u64_t>, nk_reduce_minmax_u64_rvv,
                   nk_reduce_minmax_u64_serial);
    run_if_matches("reduce_minmax_bf16_rvv", test_reduce_minmax_<bf16_t, bf16_t>, nk_reduce_minmax_bf16_rvv,
                   nk_reduce_minmax_bf16_serial);
    run_if_matches("reduce_minmax_f16_rvv", test_reduce_minmax_<f16_t, f16_t>, nk_reduce_minmax_f16_rvv,
                   nk_reduce_minmax_f16_serial);
    run_if_matches("reduce_minmax_e4m3_rvv", test_reduce_minmax_<e4m3_t, e4m3_t>, nk_reduce_minmax_e4m3_rvv,
                   nk_reduce_minmax_e4m3_serial);
    run_if_matches("reduce_minmax_e5m2_rvv", test_reduce_minmax_<e5m2_t, e5m2_t>, nk_reduce_minmax_e5m2_rvv,
                   nk_reduce_minmax_e5m2_serial);
    run_if_matches("reduce_minmax_e2m3_rvv", test_reduce_minmax_<e2m3_t, e2m3_t>, nk_reduce_minmax_e2m3_rvv,
                   nk_reduce_minmax_e2m3_serial);
    run_if_matches("reduce_minmax_e3m2_rvv", test_reduce_minmax_<e3m2_t, e3m2_t>, nk_reduce_minmax_e3m2_rvv,
                   nk_reduce_minmax_e3m2_serial);
#endif
    run_if_matches("reduce_moments_f32_serial", test_reduce_moments_<f32_t, f64_t>, nk_reduce_moments_f32_serial,
                   nk_reduce_moments_f32_serial);
    run_if_matches("reduce_moments_f64_serial", test_reduce_moments_<f64_t, f64_t>, nk_reduce_moments_f64_serial,
                   nk_reduce_moments_f64_serial);
    run_if_matches("reduce_moments_i8_serial", test_reduce_moments_<i8_t, i64_t, u64_t>, nk_reduce_moments_i8_serial,
                   nk_reduce_moments_i8_serial);
    run_if_matches("reduce_moments_u8_serial", test_reduce_moments_<u8_t, u64_t>, nk_reduce_moments_u8_serial,
                   nk_reduce_moments_u8_serial);
    run_if_matches("reduce_moments_i16_serial", test_reduce_moments_<i16_t, i64_t, u64_t>, nk_reduce_moments_i16_serial,
                   nk_reduce_moments_i16_serial);
    run_if_matches("reduce_moments_u16_serial", test_reduce_moments_<u16_t, u64_t>, nk_reduce_moments_u16_serial,
                   nk_reduce_moments_u16_serial);
    run_if_matches("reduce_moments_i32_serial", test_reduce_moments_<i32_t, i64_t, u64_t>, nk_reduce_moments_i32_serial,
                   nk_reduce_moments_i32_serial);
    run_if_matches("reduce_moments_u32_serial", test_reduce_moments_<u32_t, u64_t>, nk_reduce_moments_u32_serial,
                   nk_reduce_moments_u32_serial);
    run_if_matches("reduce_moments_i64_serial", test_reduce_moments_<i64_t, i64_t, u64_t>, nk_reduce_moments_i64_serial,
                   nk_reduce_moments_i64_serial);
    run_if_matches("reduce_moments_u64_serial", test_reduce_moments_<u64_t, u64_t>, nk_reduce_moments_u64_serial,
                   nk_reduce_moments_u64_serial);
    run_if_matches("reduce_moments_f16_serial", test_reduce_moments_<f16_t, f32_t>, nk_reduce_moments_f16_serial,
                   nk_reduce_moments_f16_serial);
    run_if_matches("reduce_moments_bf16_serial", test_reduce_moments_<bf16_t, f32_t>, nk_reduce_moments_bf16_serial,
                   nk_reduce_moments_bf16_serial);
    run_if_matches("reduce_moments_e4m3_serial", test_reduce_moments_<e4m3_t, f32_t>, nk_reduce_moments_e4m3_serial,
                   nk_reduce_moments_e4m3_serial);
    run_if_matches("reduce_moments_e5m2_serial", test_reduce_moments_<e5m2_t, f32_t>, nk_reduce_moments_e5m2_serial,
                   nk_reduce_moments_e5m2_serial);
    run_if_matches("reduce_moments_e2m3_serial", test_reduce_moments_<e2m3_t, f32_t>, nk_reduce_moments_e2m3_serial,
                   nk_reduce_moments_e2m3_serial);
    run_if_matches("reduce_moments_e3m2_serial", test_reduce_moments_<e3m2_t, f32_t>, nk_reduce_moments_e3m2_serial,
                   nk_reduce_moments_e3m2_serial);
    run_if_matches("reduce_moments_i4_serial", test_reduce_moments_<i4x2_t, i64_t, u64_t>, nk_reduce_moments_i4_serial,
                   nk_reduce_moments_i4_serial);
    run_if_matches("reduce_moments_u4_serial", test_reduce_moments_<u4x2_t, u64_t>, nk_reduce_moments_u4_serial,
                   nk_reduce_moments_u4_serial);
    run_if_matches("reduce_moments_u1_serial", test_reduce_moments_<u1x8_t, u64_t>, nk_reduce_moments_u1_serial,
                   nk_reduce_moments_u1_serial);
    run_if_matches("reduce_minmax_f32_serial", test_reduce_minmax_<f32_t, f32_t>, nk_reduce_minmax_f32_serial,
                   nk_reduce_minmax_f32_serial);
    run_if_matches("reduce_minmax_f64_serial", test_reduce_minmax_<f64_t, f64_t>, nk_reduce_minmax_f64_serial,
                   nk_reduce_minmax_f64_serial);
    run_if_matches("reduce_minmax_i8_serial", test_reduce_minmax_<i8_t, i8_t>, nk_reduce_minmax_i8_serial,
                   nk_reduce_minmax_i8_serial);
    run_if_matches("reduce_minmax_u8_serial", test_reduce_minmax_<u8_t, u8_t>, nk_reduce_minmax_u8_serial,
                   nk_reduce_minmax_u8_serial);
    run_if_matches("reduce_minmax_i16_serial", test_reduce_minmax_<i16_t, i16_t>, nk_reduce_minmax_i16_serial,
                   nk_reduce_minmax_i16_serial);
    run_if_matches("reduce_minmax_u16_serial", test_reduce_minmax_<u16_t, u16_t>, nk_reduce_minmax_u16_serial,
                   nk_reduce_minmax_u16_serial);
    run_if_matches("reduce_minmax_i32_serial", test_reduce_minmax_<i32_t, i32_t>, nk_reduce_minmax_i32_serial,
                   nk_reduce_minmax_i32_serial);
    run_if_matches("reduce_minmax_u32_serial", test_reduce_minmax_<u32_t, u32_t>, nk_reduce_minmax_u32_serial,
                   nk_reduce_minmax_u32_serial);
    run_if_matches("reduce_minmax_i64_serial", test_reduce_minmax_<i64_t, i64_t>, nk_reduce_minmax_i64_serial,
                   nk_reduce_minmax_i64_serial);
    run_if_matches("reduce_minmax_u64_serial", test_reduce_minmax_<u64_t, u64_t>, nk_reduce_minmax_u64_serial,
                   nk_reduce_minmax_u64_serial);
    run_if_matches("reduce_minmax_f16_serial", test_reduce_minmax_<f16_t, f16_t>, nk_reduce_minmax_f16_serial,
                   nk_reduce_minmax_f16_serial);
    run_if_matches("reduce_minmax_bf16_serial", test_reduce_minmax_<bf16_t, bf16_t>, nk_reduce_minmax_bf16_serial,
                   nk_reduce_minmax_bf16_serial);
    run_if_matches("reduce_minmax_e4m3_serial", test_reduce_minmax_<e4m3_t, e4m3_t>, nk_reduce_minmax_e4m3_serial,
                   nk_reduce_minmax_e4m3_serial);
    run_if_matches("reduce_minmax_e5m2_serial", test_reduce_minmax_<e5m2_t, e5m2_t>, nk_reduce_minmax_e5m2_serial,
                   nk_reduce_minmax_e5m2_serial);
    run_if_matches("reduce_minmax_e2m3_serial", test_reduce_minmax_<e2m3_t, e2m3_t>, nk_reduce_minmax_e2m3_serial,
                   nk_reduce_minmax_e2m3_serial);
    run_if_matches("reduce_minmax_e3m2_serial", test_reduce_minmax_<e3m2_t, e3m2_t>, nk_reduce_minmax_e3m2_serial,
                   nk_reduce_minmax_e3m2_serial);
    run_if_matches("reduce_minmax_i4_serial", test_reduce_minmax_<i4x2_t, i8_t>, nk_reduce_minmax_i4_serial,
                   nk_reduce_minmax_i4_serial);
    run_if_matches("reduce_minmax_u4_serial", test_reduce_minmax_<u4x2_t, u8_t>, nk_reduce_minmax_u4_serial,
                   nk_reduce_minmax_u4_serial);
    run_if_matches("reduce_minmax_u1_serial", test_reduce_minmax_<u1x8_t, u8_t>, nk_reduce_minmax_u1_serial,
                   nk_reduce_minmax_u1_serial);
#endif
}

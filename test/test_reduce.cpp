/**
 *  @brief Reduction tests.
 *  @file test/test_reduce.cpp
 *  @author Ash Vardanian
 *  @date February 6, 2026
 */

#include "test.hpp"
#include "numkong/reduce.hpp"

/**
 *  @brief Unified reduce_add test for float types.
 *  Works with f32_t, f64_t, e4m3_t, e5m2_t wrapper types.
 */
template <typename scalar_type_>
error_stats_t test_reduce_add(typename scalar_type_::reduce_add_kernel_t kernel) {
    using scalar_t = scalar_type_;
    using raw_t = typename scalar_t::raw_t;
    using result_t = typename scalar_t::reduce_add_result_t;

    error_stats_t stats;
    std::mt19937 generator(global_config.seed);
    auto buffer = make_vector<scalar_t>(dense_dimensions);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(generator, buffer);

        result_t result;
        kernel(buffer.raw_values_data(), dense_dimensions, sizeof(raw_t), &result.raw_);

        f118_t reference;
        nk::reduce_add<scalar_t, f118_t, nk::no_simd_k>(buffer.values_data(), dense_dimensions, sizeof(raw_t),
                                                        &reference);

        stats.accumulate(result, reference);
    }
    return stats;
}

template <typename scalar_type_>
error_stats_t test_reduce_min(typename scalar_type_::reduce_extremum_kernel_t kernel) {
    using scalar_t = scalar_type_;
    using raw_t = typename scalar_t::raw_t;

    error_stats_t stats;
    std::mt19937 generator(global_config.seed);
    auto buffer = make_vector<scalar_t>(dense_dimensions);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(generator, buffer);

        scalar_t min_val;
        nk_size_t min_idx;
        kernel(buffer.raw_values_data(), dense_dimensions, sizeof(raw_t), &min_val.raw_, &min_idx);

        scalar_t ref_val;
        std::size_t ref_idx;
        nk::reduce_min<scalar_t, nk::no_simd_k>(buffer.values_data(), dense_dimensions, sizeof(raw_t), &ref_val,
                                                &ref_idx);

        stats.accumulate(min_val, ref_val);
    }
    return stats;
}

template <typename scalar_type_>
error_stats_t test_reduce_max(typename scalar_type_::reduce_extremum_kernel_t kernel) {
    using scalar_t = scalar_type_;
    using raw_t = typename scalar_t::raw_t;

    error_stats_t stats;
    std::mt19937 generator(global_config.seed);
    auto buffer = make_vector<scalar_t>(dense_dimensions);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(generator, buffer);

        scalar_t max_val;
        nk_size_t max_idx;
        kernel(buffer.raw_values_data(), dense_dimensions, sizeof(raw_t), &max_val.raw_, &max_idx);

        scalar_t ref_val;
        std::size_t ref_idx;
        nk::reduce_max<scalar_t, scalar_t, nk::no_simd_k>(buffer.values_data(), dense_dimensions, sizeof(raw_t),
                                                          &ref_val, &ref_idx);

        stats.accumulate(max_val, ref_val);
    }
    return stats;
}

void test_reduce() {
    std::puts("");
    std::printf("Reductions:\n");

#if NK_DYNAMIC_DISPATCH
    run_if_matches("reduce_add_f32", test_reduce_add<f32_t>, nk_reduce_add_f32);
    run_if_matches("reduce_add_f64", test_reduce_add<f64_t>, nk_reduce_add_f64);
    run_if_matches("reduce_add_i32", test_reduce_add<i32_t>, nk_reduce_add_i32);
    run_if_matches("reduce_add_e4m3", test_reduce_add<e4m3_t>, nk_reduce_add_e4m3);
    run_if_matches("reduce_add_e5m2", test_reduce_add<e5m2_t>, nk_reduce_add_e5m2);
    run_if_matches("reduce_add_e2m3", test_reduce_add<e2m3_t>, nk_reduce_add_e2m3);
    run_if_matches("reduce_add_e3m2", test_reduce_add<e3m2_t>, nk_reduce_add_e3m2);

    run_if_matches("reduce_min_f32", test_reduce_min<f32_t>, nk_reduce_min_f32);
    run_if_matches("reduce_min_f64", test_reduce_min<f64_t>, nk_reduce_min_f64);
    run_if_matches("reduce_min_i8", test_reduce_min<i8_t>, nk_reduce_min_i8);
    run_if_matches("reduce_min_u8", test_reduce_min<u8_t>, nk_reduce_min_u8);
    run_if_matches("reduce_min_i16", test_reduce_min<i16_t>, nk_reduce_min_i16);
    run_if_matches("reduce_min_u16", test_reduce_min<u16_t>, nk_reduce_min_u16);
    run_if_matches("reduce_min_i32", test_reduce_min<i32_t>, nk_reduce_min_i32);
    run_if_matches("reduce_min_u32", test_reduce_min<u32_t>, nk_reduce_min_u32);
    run_if_matches("reduce_min_i64", test_reduce_min<i64_t>, nk_reduce_min_i64);
    run_if_matches("reduce_min_u64", test_reduce_min<u64_t>, nk_reduce_min_u64);
    run_if_matches("reduce_min_e2m3", test_reduce_min<e2m3_t>, nk_reduce_min_e2m3);
    run_if_matches("reduce_min_e3m2", test_reduce_min<e3m2_t>, nk_reduce_min_e3m2);

    run_if_matches("reduce_max_f32", test_reduce_max<f32_t>, nk_reduce_max_f32);
    run_if_matches("reduce_max_f64", test_reduce_max<f64_t>, nk_reduce_max_f64);
    run_if_matches("reduce_max_i8", test_reduce_max<i8_t>, nk_reduce_max_i8);
    run_if_matches("reduce_max_u8", test_reduce_max<u8_t>, nk_reduce_max_u8);
    run_if_matches("reduce_max_i16", test_reduce_max<i16_t>, nk_reduce_max_i16);
    run_if_matches("reduce_max_u16", test_reduce_max<u16_t>, nk_reduce_max_u16);
    run_if_matches("reduce_max_i32", test_reduce_max<i32_t>, nk_reduce_max_i32);
    run_if_matches("reduce_max_u32", test_reduce_max<u32_t>, nk_reduce_max_u32);
    run_if_matches("reduce_max_i64", test_reduce_max<i64_t>, nk_reduce_max_i64);
    run_if_matches("reduce_max_u64", test_reduce_max<u64_t>, nk_reduce_max_u64);
    run_if_matches("reduce_max_e2m3", test_reduce_max<e2m3_t>, nk_reduce_max_e2m3);
    run_if_matches("reduce_max_e3m2", test_reduce_max<e3m2_t>, nk_reduce_max_e3m2);
#else
#if NK_TARGET_NEON
    run_if_matches("reduce_add_f32_neon", test_reduce_add<f32_t>, nk_reduce_add_f32_neon);
    run_if_matches("reduce_add_f64_neon", test_reduce_add<f64_t>, nk_reduce_add_f64_neon);
    run_if_matches("reduce_add_i32_neon", test_reduce_add<i32_t>, nk_reduce_add_i32_neon);
    run_if_matches("reduce_min_f32_neon", test_reduce_min<f32_t>, nk_reduce_min_f32_neon);
    run_if_matches("reduce_min_f64_neon", test_reduce_min<f64_t>, nk_reduce_min_f64_neon);
    run_if_matches("reduce_min_i8_neon", test_reduce_min<i8_t>, nk_reduce_min_i8_neon);
    run_if_matches("reduce_min_u8_neon", test_reduce_min<u8_t>, nk_reduce_min_u8_neon);
    run_if_matches("reduce_min_i16_neon", test_reduce_min<i16_t>, nk_reduce_min_i16_neon);
    run_if_matches("reduce_min_u16_neon", test_reduce_min<u16_t>, nk_reduce_min_u16_neon);
    run_if_matches("reduce_max_f32_neon", test_reduce_max<f32_t>, nk_reduce_max_f32_neon);
    run_if_matches("reduce_max_f64_neon", test_reduce_max<f64_t>, nk_reduce_max_f64_neon);
    run_if_matches("reduce_max_i8_neon", test_reduce_max<i8_t>, nk_reduce_max_i8_neon);
    run_if_matches("reduce_max_u8_neon", test_reduce_max<u8_t>, nk_reduce_max_u8_neon);
    run_if_matches("reduce_max_i16_neon", test_reduce_max<i16_t>, nk_reduce_max_i16_neon);
    run_if_matches("reduce_max_u16_neon", test_reduce_max<u16_t>, nk_reduce_max_u16_neon);
#endif
#if NK_TARGET_NEONFHM
    run_if_matches("reduce_add_e4m3_neonfhm", test_reduce_add<e4m3_t>, nk_reduce_add_e4m3_neonfhm);
    run_if_matches("reduce_add_e5m2_neonfhm", test_reduce_add<e5m2_t>, nk_reduce_add_e5m2_neonfhm);
#endif
#if NK_TARGET_HASWELL
    run_if_matches("reduce_add_f32_haswell", test_reduce_add<f32_t>, nk_reduce_add_f32_haswell);
    run_if_matches("reduce_add_f64_haswell", test_reduce_add<f64_t>, nk_reduce_add_f64_haswell);
    run_if_matches("reduce_add_i32_haswell", test_reduce_add<i32_t>, nk_reduce_add_i32_haswell);
    run_if_matches("reduce_add_e4m3_haswell", test_reduce_add<e4m3_t>, nk_reduce_add_e4m3_haswell);
    run_if_matches("reduce_add_e5m2_haswell", test_reduce_add<e5m2_t>, nk_reduce_add_e5m2_haswell);
    run_if_matches("reduce_add_e2m3_haswell", test_reduce_add<e2m3_t>, nk_reduce_add_e2m3_haswell);
    run_if_matches("reduce_add_e3m2_haswell", test_reduce_add<e3m2_t>, nk_reduce_add_e3m2_haswell);
    run_if_matches("reduce_min_f32_haswell", test_reduce_min<f32_t>, nk_reduce_min_f32_haswell);
    run_if_matches("reduce_min_f64_haswell", test_reduce_min<f64_t>, nk_reduce_min_f64_haswell);
    run_if_matches("reduce_min_i8_haswell", test_reduce_min<i8_t>, nk_reduce_min_i8_haswell);
    run_if_matches("reduce_min_u8_haswell", test_reduce_min<u8_t>, nk_reduce_min_u8_haswell);
    run_if_matches("reduce_min_i16_haswell", test_reduce_min<i16_t>, nk_reduce_min_i16_haswell);
    run_if_matches("reduce_min_u16_haswell", test_reduce_min<u16_t>, nk_reduce_min_u16_haswell);
    run_if_matches("reduce_min_i32_haswell", test_reduce_min<i32_t>, nk_reduce_min_i32_haswell);
    run_if_matches("reduce_min_u32_haswell", test_reduce_min<u32_t>, nk_reduce_min_u32_haswell);
    run_if_matches("reduce_min_i64_haswell", test_reduce_min<i64_t>, nk_reduce_min_i64_haswell);
    run_if_matches("reduce_min_u64_haswell", test_reduce_min<u64_t>, nk_reduce_min_u64_haswell);
    run_if_matches("reduce_min_e2m3_haswell", test_reduce_min<e2m3_t>, nk_reduce_min_e2m3_haswell);
    run_if_matches("reduce_min_e3m2_haswell", test_reduce_min<e3m2_t>, nk_reduce_min_e3m2_haswell);
    run_if_matches("reduce_max_f32_haswell", test_reduce_max<f32_t>, nk_reduce_max_f32_haswell);
    run_if_matches("reduce_max_f64_haswell", test_reduce_max<f64_t>, nk_reduce_max_f64_haswell);
    run_if_matches("reduce_max_i8_haswell", test_reduce_max<i8_t>, nk_reduce_max_i8_haswell);
    run_if_matches("reduce_max_u8_haswell", test_reduce_max<u8_t>, nk_reduce_max_u8_haswell);
    run_if_matches("reduce_max_i16_haswell", test_reduce_max<i16_t>, nk_reduce_max_i16_haswell);
    run_if_matches("reduce_max_u16_haswell", test_reduce_max<u16_t>, nk_reduce_max_u16_haswell);
    run_if_matches("reduce_max_i32_haswell", test_reduce_max<i32_t>, nk_reduce_max_i32_haswell);
    run_if_matches("reduce_max_u32_haswell", test_reduce_max<u32_t>, nk_reduce_max_u32_haswell);
    run_if_matches("reduce_max_i64_haswell", test_reduce_max<i64_t>, nk_reduce_max_i64_haswell);
    run_if_matches("reduce_max_u64_haswell", test_reduce_max<u64_t>, nk_reduce_max_u64_haswell);
    run_if_matches("reduce_max_e2m3_haswell", test_reduce_max<e2m3_t>, nk_reduce_max_e2m3_haswell);
    run_if_matches("reduce_max_e3m2_haswell", test_reduce_max<e3m2_t>, nk_reduce_max_e3m2_haswell);
#endif
#if NK_TARGET_SKYLAKE
    run_if_matches("reduce_add_f32_skylake", test_reduce_add<f32_t>, nk_reduce_add_f32_skylake);
    run_if_matches("reduce_add_f64_skylake", test_reduce_add<f64_t>, nk_reduce_add_f64_skylake);
    run_if_matches("reduce_add_i32_skylake", test_reduce_add<i32_t>, nk_reduce_add_i32_skylake);
    run_if_matches("reduce_min_f32_skylake", test_reduce_min<f32_t>, nk_reduce_min_f32_skylake);
    run_if_matches("reduce_min_f64_skylake", test_reduce_min<f64_t>, nk_reduce_min_f64_skylake);
    run_if_matches("reduce_min_i8_skylake", test_reduce_min<i8_t>, nk_reduce_min_i8_skylake);
    run_if_matches("reduce_min_u8_skylake", test_reduce_min<u8_t>, nk_reduce_min_u8_skylake);
    run_if_matches("reduce_min_i16_skylake", test_reduce_min<i16_t>, nk_reduce_min_i16_skylake);
    run_if_matches("reduce_min_u16_skylake", test_reduce_min<u16_t>, nk_reduce_min_u16_skylake);
    run_if_matches("reduce_min_i32_skylake", test_reduce_min<i32_t>, nk_reduce_min_i32_skylake);
    run_if_matches("reduce_min_u32_skylake", test_reduce_min<u32_t>, nk_reduce_min_u32_skylake);
    run_if_matches("reduce_min_i64_skylake", test_reduce_min<i64_t>, nk_reduce_min_i64_skylake);
    run_if_matches("reduce_min_u64_skylake", test_reduce_min<u64_t>, nk_reduce_min_u64_skylake);
    run_if_matches("reduce_min_e2m3_skylake", test_reduce_min<e2m3_t>, nk_reduce_min_e2m3_skylake);
    run_if_matches("reduce_min_e3m2_skylake", test_reduce_min<e3m2_t>, nk_reduce_min_e3m2_skylake);
    run_if_matches("reduce_max_f32_skylake", test_reduce_max<f32_t>, nk_reduce_max_f32_skylake);
    run_if_matches("reduce_max_f64_skylake", test_reduce_max<f64_t>, nk_reduce_max_f64_skylake);
    run_if_matches("reduce_max_i8_skylake", test_reduce_max<i8_t>, nk_reduce_max_i8_skylake);
    run_if_matches("reduce_max_u8_skylake", test_reduce_max<u8_t>, nk_reduce_max_u8_skylake);
    run_if_matches("reduce_max_i16_skylake", test_reduce_max<i16_t>, nk_reduce_max_i16_skylake);
    run_if_matches("reduce_max_u16_skylake", test_reduce_max<u16_t>, nk_reduce_max_u16_skylake);
    run_if_matches("reduce_max_i32_skylake", test_reduce_max<i32_t>, nk_reduce_max_i32_skylake);
    run_if_matches("reduce_max_u32_skylake", test_reduce_max<u32_t>, nk_reduce_max_u32_skylake);
    run_if_matches("reduce_max_i64_skylake", test_reduce_max<i64_t>, nk_reduce_max_i64_skylake);
    run_if_matches("reduce_max_u64_skylake", test_reduce_max<u64_t>, nk_reduce_max_u64_skylake);
    run_if_matches("reduce_max_e2m3_skylake", test_reduce_max<e2m3_t>, nk_reduce_max_e2m3_skylake);
    run_if_matches("reduce_max_e3m2_skylake", test_reduce_max<e3m2_t>, nk_reduce_max_e3m2_skylake);
#endif
#if NK_TARGET_RVV
    run_if_matches("reduce_add_f32_rvv", test_reduce_add<f32_t>, nk_reduce_add_f32_rvv);
    run_if_matches("reduce_add_f64_rvv", test_reduce_add<f64_t>, nk_reduce_add_f64_rvv);
    run_if_matches("reduce_add_i32_rvv", test_reduce_add<i32_t>, nk_reduce_add_i32_rvv);
    run_if_matches("reduce_add_e4m3_rvv", test_reduce_add<e4m3_t>, nk_reduce_add_e4m3_rvv);
    run_if_matches("reduce_add_e5m2_rvv", test_reduce_add<e5m2_t>, nk_reduce_add_e5m2_rvv);
    run_if_matches("reduce_add_e2m3_rvv", test_reduce_add<e2m3_t>, nk_reduce_add_e2m3_rvv);
    run_if_matches("reduce_add_e3m2_rvv", test_reduce_add<e3m2_t>, nk_reduce_add_e3m2_rvv);
    run_if_matches("reduce_min_f32_rvv", test_reduce_min<f32_t>, nk_reduce_min_f32_rvv);
    run_if_matches("reduce_min_f64_rvv", test_reduce_min<f64_t>, nk_reduce_min_f64_rvv);
    run_if_matches("reduce_min_i8_rvv", test_reduce_min<i8_t>, nk_reduce_min_i8_rvv);
    run_if_matches("reduce_min_u8_rvv", test_reduce_min<u8_t>, nk_reduce_min_u8_rvv);
    run_if_matches("reduce_max_f32_rvv", test_reduce_max<f32_t>, nk_reduce_max_f32_rvv);
    run_if_matches("reduce_max_f64_rvv", test_reduce_max<f64_t>, nk_reduce_max_f64_rvv);
    run_if_matches("reduce_max_i8_rvv", test_reduce_max<i8_t>, nk_reduce_max_i8_rvv);
    run_if_matches("reduce_max_u8_rvv", test_reduce_max<u8_t>, nk_reduce_max_u8_rvv);
#endif
    run_if_matches("reduce_add_f32_serial", test_reduce_add<f32_t>, nk_reduce_add_f32_serial);
    run_if_matches("reduce_add_f64_serial", test_reduce_add<f64_t>, nk_reduce_add_f64_serial);
    run_if_matches("reduce_add_i32_serial", test_reduce_add<i32_t>, nk_reduce_add_i32_serial);
    run_if_matches("reduce_add_e4m3_serial", test_reduce_add<e4m3_t>, nk_reduce_add_e4m3_serial);
    run_if_matches("reduce_add_e5m2_serial", test_reduce_add<e5m2_t>, nk_reduce_add_e5m2_serial);
    run_if_matches("reduce_add_e2m3_serial", test_reduce_add<e2m3_t>, nk_reduce_add_e2m3_serial);
    run_if_matches("reduce_add_e3m2_serial", test_reduce_add<e3m2_t>, nk_reduce_add_e3m2_serial);
    run_if_matches("reduce_min_f32_serial", test_reduce_min<f32_t>, nk_reduce_min_f32_serial);
    run_if_matches("reduce_min_f64_serial", test_reduce_min<f64_t>, nk_reduce_min_f64_serial);
    run_if_matches("reduce_min_i8_serial", test_reduce_min<i8_t>, nk_reduce_min_i8_serial);
    run_if_matches("reduce_min_u8_serial", test_reduce_min<u8_t>, nk_reduce_min_u8_serial);
    run_if_matches("reduce_min_i16_serial", test_reduce_min<i16_t>, nk_reduce_min_i16_serial);
    run_if_matches("reduce_min_u16_serial", test_reduce_min<u16_t>, nk_reduce_min_u16_serial);
    run_if_matches("reduce_min_i32_serial", test_reduce_min<i32_t>, nk_reduce_min_i32_serial);
    run_if_matches("reduce_min_u32_serial", test_reduce_min<u32_t>, nk_reduce_min_u32_serial);
    run_if_matches("reduce_min_i64_serial", test_reduce_min<i64_t>, nk_reduce_min_i64_serial);
    run_if_matches("reduce_min_u64_serial", test_reduce_min<u64_t>, nk_reduce_min_u64_serial);
    run_if_matches("reduce_min_e2m3_serial", test_reduce_min<e2m3_t>, nk_reduce_min_e2m3_serial);
    run_if_matches("reduce_min_e3m2_serial", test_reduce_min<e3m2_t>, nk_reduce_min_e3m2_serial);
    run_if_matches("reduce_max_f32_serial", test_reduce_max<f32_t>, nk_reduce_max_f32_serial);
    run_if_matches("reduce_max_f64_serial", test_reduce_max<f64_t>, nk_reduce_max_f64_serial);
    run_if_matches("reduce_max_i8_serial", test_reduce_max<i8_t>, nk_reduce_max_i8_serial);
    run_if_matches("reduce_max_u8_serial", test_reduce_max<u8_t>, nk_reduce_max_u8_serial);
    run_if_matches("reduce_max_i16_serial", test_reduce_max<i16_t>, nk_reduce_max_i16_serial);
    run_if_matches("reduce_max_u16_serial", test_reduce_max<u16_t>, nk_reduce_max_u16_serial);
    run_if_matches("reduce_max_i32_serial", test_reduce_max<i32_t>, nk_reduce_max_i32_serial);
    run_if_matches("reduce_max_u32_serial", test_reduce_max<u32_t>, nk_reduce_max_u32_serial);
    run_if_matches("reduce_max_i64_serial", test_reduce_max<i64_t>, nk_reduce_max_i64_serial);
    run_if_matches("reduce_max_u64_serial", test_reduce_max<u64_t>, nk_reduce_max_u64_serial);
    run_if_matches("reduce_max_e2m3_serial", test_reduce_max<e2m3_t>, nk_reduce_max_e2m3_serial);
    run_if_matches("reduce_max_e3m2_serial", test_reduce_max<e3m2_t>, nk_reduce_max_e3m2_serial);
#endif
}

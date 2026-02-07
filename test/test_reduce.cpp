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
#else
#if NK_TARGET_NEON
    run_if_matches("reduce_add_f32_neon", test_reduce_add<f32_t>, nk_reduce_add_f32_neon);
    run_if_matches("reduce_add_f64_neon", test_reduce_add<f64_t>, nk_reduce_add_f64_neon);
    run_if_matches("reduce_add_i32_neon", test_reduce_add<i32_t>, nk_reduce_add_i32_neon);
#endif
#if NK_TARGET_NEONFHM
    run_if_matches("reduce_add_neonfhm_e4m3", test_reduce_add<e4m3_t>, nk_reduce_add_e4m3_neonfhm);
    run_if_matches("reduce_add_neonfhm_e5m2", test_reduce_add<e5m2_t>, nk_reduce_add_e5m2_neonfhm);
#endif
#if NK_TARGET_HASWELL
    run_if_matches("reduce_add_f32_haswell", test_reduce_add<f32_t>, nk_reduce_add_f32_haswell);
    run_if_matches("reduce_add_f64_haswell", test_reduce_add<f64_t>, nk_reduce_add_f64_haswell);
    run_if_matches("reduce_add_i32_haswell", test_reduce_add<i32_t>, nk_reduce_add_i32_haswell);
    run_if_matches("reduce_add_e4m3_haswell", test_reduce_add<e4m3_t>, nk_reduce_add_e4m3_haswell);
    run_if_matches("reduce_add_e5m2_haswell", test_reduce_add<e5m2_t>, nk_reduce_add_e5m2_haswell);
#endif
#if NK_TARGET_SKYLAKE
    run_if_matches("reduce_add_f32_skylake", test_reduce_add<f32_t>, nk_reduce_add_f32_skylake);
    run_if_matches("reduce_add_f64_skylake", test_reduce_add<f64_t>, nk_reduce_add_f64_skylake);
    run_if_matches("reduce_add_i32_skylake", test_reduce_add<i32_t>, nk_reduce_add_i32_skylake);
#endif
    run_if_matches("reduce_add_f32_serial", test_reduce_add<f32_t>, nk_reduce_add_f32_serial);
    run_if_matches("reduce_add_f64_serial", test_reduce_add<f64_t>, nk_reduce_add_f64_serial);
    run_if_matches("reduce_add_i32_serial", test_reduce_add<i32_t>, nk_reduce_add_i32_serial);
    run_if_matches("reduce_add_e4m3_serial", test_reduce_add<e4m3_t>, nk_reduce_add_e4m3_serial);
    run_if_matches("reduce_add_e5m2_serial", test_reduce_add<e5m2_t>, nk_reduce_add_e5m2_serial);
#endif
}

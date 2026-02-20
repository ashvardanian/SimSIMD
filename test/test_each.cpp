/**
 *  @brief Elementwise operations tests.
 *  @file test/test_each.cpp
 *  @author Ash Vardanian
 *  @date December 28, 2025
 */

#include "test.hpp"
#include "numkong/each.hpp" // `nk::sum`, `nk::scale`, `nk::wsum`, `nk::fma`

/**
 *  @brief Unified test for elementwise sum: result[i] = a[i] + b[i]
 */
template <typename scalar_type_>
error_stats_t test_sum(typename scalar_type_::sum_kernel_t kernel) {
    using scalar_t = scalar_type_;
    using raw_t = typename scalar_t::raw_t;

    error_stats_t stats;
    std::mt19937 generator(global_config.seed);
    auto a = make_vector<scalar_t>(global_config.dense_dimensions),
         b = make_vector<scalar_t>(global_config.dense_dimensions);
    auto result = make_vector<scalar_t>(global_config.dense_dimensions),
         reference = make_vector<scalar_t>(global_config.dense_dimensions);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(generator, a);
        fill_random(generator, b);

        kernel(a.raw_values_data(), b.raw_values_data(), global_config.dense_dimensions, result.raw_values_data());
        nk::sum<scalar_t, nk::no_simd_k>(a.values_data(), b.values_data(), global_config.dense_dimensions,
                                         reference.values_data());

        for (std::size_t i = 0; i < global_config.dense_dimensions; i++) stats.accumulate(result[i], reference[i]);
    }
    return stats;
}

/**
 *  @brief Unified test for scale: result[i] = alpha * x[i] + beta
 */
template <typename scalar_type_>
error_stats_t test_scale(typename scalar_type_::scale_kernel_t kernel) {
    using scalar_t = scalar_type_;
    using raw_t = typename scalar_t::raw_t;
    using scale_t = typename scalar_t::scale_t;

    error_stats_t stats;
    std::mt19937 generator(global_config.seed);
    auto input = make_vector<scalar_t>(global_config.dense_dimensions);
    auto result = make_vector<scalar_t>(global_config.dense_dimensions),
         reference = make_vector<scalar_t>(global_config.dense_dimensions);
    std::uniform_real_distribution<scale_t> coef_dist(scale_t(-2.0), scale_t(2.0));

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(generator, input);

        scale_t alpha = coef_dist(generator);
        scale_t beta = coef_dist(generator);

        kernel(input.raw_values_data(), global_config.dense_dimensions, &alpha, &beta, result.raw_values_data());
        nk::scale<scalar_t, f118_t, nk::no_simd_k>(input.values_data(), global_config.dense_dimensions, &alpha, &beta,
                                                   reference.values_data());

        for (std::size_t i = 0; i < global_config.dense_dimensions; i++) stats.accumulate(result[i], reference[i]);
    }
    return stats;
}

/**
 *  @brief Unified test for weighted sum: result[i] = alpha * a[i] + beta * b[i]
 */
template <typename scalar_type_>
error_stats_t test_wsum(typename scalar_type_::wsum_kernel_t kernel) {
    using scalar_t = scalar_type_;
    using raw_t = typename scalar_t::raw_t;
    using scale_t = typename scalar_t::scale_t;

    error_stats_t stats;
    std::mt19937 generator(global_config.seed);
    auto a = make_vector<scalar_t>(global_config.dense_dimensions),
         b = make_vector<scalar_t>(global_config.dense_dimensions);
    auto result = make_vector<scalar_t>(global_config.dense_dimensions),
         reference = make_vector<scalar_t>(global_config.dense_dimensions);
    std::uniform_real_distribution<scale_t> coef_dist(scale_t(-2.0), scale_t(2.0));

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(generator, a);
        fill_random(generator, b);

        scale_t alpha = coef_dist(generator);
        scale_t beta = coef_dist(generator);

        kernel(a.raw_values_data(), b.raw_values_data(), global_config.dense_dimensions, &alpha, &beta,
               result.raw_values_data());
        nk::wsum<scalar_t, f118_t, nk::no_simd_k>(a.values_data(), b.values_data(), global_config.dense_dimensions,
                                                  &alpha, &beta, reference.values_data());

        for (std::size_t i = 0; i < global_config.dense_dimensions; i++) stats.accumulate(result[i], reference[i]);
    }
    return stats;
}

/**
 *  @brief Unified test for FMA: result[i] = alpha * a[i] * b[i] + beta * c[i]
 */
template <typename scalar_type_>
error_stats_t test_fma(typename scalar_type_::fma_kernel_t kernel) {
    using scalar_t = scalar_type_;
    using raw_t = typename scalar_t::raw_t;
    using scale_t = typename scalar_t::scale_t;

    error_stats_t stats;
    std::mt19937 generator(global_config.seed);
    auto a = make_vector<scalar_t>(global_config.dense_dimensions),
         b = make_vector<scalar_t>(global_config.dense_dimensions);
    auto c = make_vector<scalar_t>(global_config.dense_dimensions);
    auto result = make_vector<scalar_t>(global_config.dense_dimensions),
         reference = make_vector<scalar_t>(global_config.dense_dimensions);
    std::uniform_real_distribution<scale_t> coef_dist(scale_t(-2.0), scale_t(2.0));

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(generator, a);
        fill_random(generator, b);
        fill_random(generator, c);

        scale_t alpha = coef_dist(generator);
        scale_t beta = coef_dist(generator);

        kernel(a.raw_values_data(), b.raw_values_data(), c.raw_values_data(), global_config.dense_dimensions, &alpha,
               &beta, result.raw_values_data());
        nk::fma<scalar_t, f118_t, nk::no_simd_k>(a.values_data(), b.values_data(), global_config.dense_dimensions,
                                                 c.values_data(), &alpha, &beta, reference.values_data());

        for (std::size_t i = 0; i < global_config.dense_dimensions; i++) stats.accumulate(result[i], reference[i]);
    }
    return stats;
}

void test_elementwise() {
    std::puts("");
    std::printf("Elementwise Operations:\n");

#if NK_DYNAMIC_DISPATCH
    // Dynamic dispatch - only test the dispatcher itself
    run_if_matches("each_scale_f32", test_scale<f32_t>, nk_each_scale_f32);
    run_if_matches("each_sum_f32", test_sum<f32_t>, nk_each_sum_f32);
    run_if_matches("each_wsum_f32", test_wsum<f32_t>, nk_each_blend_f32);
    run_if_matches("each_fma_f32", test_fma<f32_t>, nk_each_fma_f32);
    run_if_matches("each_scale_e4m3", test_scale<e4m3_t>, nk_each_scale_e4m3);
    run_if_matches("each_scale_e5m2", test_scale<e5m2_t>, nk_each_scale_e5m2);
    run_if_matches("each_sum_e4m3", test_sum<e4m3_t>, nk_each_sum_e4m3);
    run_if_matches("each_sum_e5m2", test_sum<e5m2_t>, nk_each_sum_e5m2);
    run_if_matches("each_wsum_e4m3", test_wsum<e4m3_t>, nk_each_blend_e4m3);
    run_if_matches("each_wsum_e5m2", test_wsum<e5m2_t>, nk_each_blend_e5m2);
    run_if_matches("each_fma_e4m3", test_fma<e4m3_t>, nk_each_fma_e4m3);
    run_if_matches("each_fma_e5m2", test_fma<e5m2_t>, nk_each_fma_e5m2);
#else
    // Static compilation - test all available ISA variants

#if NK_TARGET_NEON
    run_if_matches("each_scale_f32_neon", test_scale<f32_t>, nk_each_scale_f32_neon);
    run_if_matches("each_sum_f32_neon", test_sum<f32_t>, nk_each_sum_f32_neon);
    run_if_matches("each_wsum_f32_neon", test_wsum<f32_t>, nk_each_blend_f32_neon);
    run_if_matches("each_fma_f32_neon", test_fma<f32_t>, nk_each_fma_f32_neon);
    run_if_matches("each_scale_e4m3_neon", test_scale<e4m3_t>, nk_each_scale_e4m3_neon);
    run_if_matches("each_scale_e5m2_neon", test_scale<e5m2_t>, nk_each_scale_e5m2_neon);
    run_if_matches("each_sum_e4m3_neon", test_sum<e4m3_t>, nk_each_sum_e4m3_neon);
    run_if_matches("each_sum_e5m2_neon", test_sum<e5m2_t>, nk_each_sum_e5m2_neon);
    run_if_matches("each_wsum_e4m3_neon", test_wsum<e4m3_t>, nk_each_blend_e4m3_neon);
    run_if_matches("each_wsum_e5m2_neon", test_wsum<e5m2_t>, nk_each_blend_e5m2_neon);
    run_if_matches("each_fma_e4m3_neon", test_fma<e4m3_t>, nk_each_fma_e4m3_neon);
    run_if_matches("each_fma_e5m2_neon", test_fma<e5m2_t>, nk_each_fma_e5m2_neon);
#endif // NK_TARGET_NEON

#if NK_TARGET_NEONHALF
#endif // NK_TARGET_NEONHALF

#if NK_TARGET_HASWELL
    run_if_matches("each_scale_f32_haswell", test_scale<f32_t>, nk_each_scale_f32_haswell);
    run_if_matches("each_sum_f32_haswell", test_sum<f32_t>, nk_each_sum_f32_haswell);
    run_if_matches("each_wsum_f32_haswell", test_wsum<f32_t>, nk_each_blend_f32_haswell);
    run_if_matches("each_fma_f32_haswell", test_fma<f32_t>, nk_each_fma_f32_haswell);
    run_if_matches("each_scale_e4m3_haswell", test_scale<e4m3_t>, nk_each_scale_e4m3_haswell);
    run_if_matches("each_scale_e5m2_haswell", test_scale<e5m2_t>, nk_each_scale_e5m2_haswell);
    run_if_matches("each_sum_e4m3_haswell", test_sum<e4m3_t>, nk_each_sum_e4m3_haswell);
    run_if_matches("each_sum_e5m2_haswell", test_sum<e5m2_t>, nk_each_sum_e5m2_haswell);
    run_if_matches("each_wsum_e4m3_haswell", test_wsum<e4m3_t>, nk_each_blend_e4m3_haswell);
    run_if_matches("each_wsum_e5m2_haswell", test_wsum<e5m2_t>, nk_each_blend_e5m2_haswell);
    run_if_matches("each_fma_e4m3_haswell", test_fma<e4m3_t>, nk_each_fma_e4m3_haswell);
    run_if_matches("each_fma_e5m2_haswell", test_fma<e5m2_t>, nk_each_fma_e5m2_haswell);
#endif // NK_TARGET_HASWELL

#if NK_TARGET_SKYLAKE
    run_if_matches("each_scale_f32_skylake", test_scale<f32_t>, nk_each_scale_f32_skylake);
    run_if_matches("each_sum_f32_skylake", test_sum<f32_t>, nk_each_sum_f32_skylake);
    run_if_matches("each_wsum_f32_skylake", test_wsum<f32_t>, nk_each_blend_f32_skylake);
    run_if_matches("each_fma_f32_skylake", test_fma<f32_t>, nk_each_fma_f32_skylake);
    run_if_matches("each_scale_e4m3_skylake", test_scale<e4m3_t>, nk_each_scale_e4m3_skylake);
    run_if_matches("each_scale_e5m2_skylake", test_scale<e5m2_t>, nk_each_scale_e5m2_skylake);
    run_if_matches("each_sum_e4m3_skylake", test_sum<e4m3_t>, nk_each_sum_e4m3_skylake);
    run_if_matches("each_sum_e5m2_skylake", test_sum<e5m2_t>, nk_each_sum_e5m2_skylake);
    run_if_matches("each_wsum_e4m3_skylake", test_wsum<e4m3_t>, nk_each_blend_e4m3_skylake);
    run_if_matches("each_wsum_e5m2_skylake", test_wsum<e5m2_t>, nk_each_blend_e5m2_skylake);
    run_if_matches("each_fma_e4m3_skylake", test_fma<e4m3_t>, nk_each_fma_e4m3_skylake);
    run_if_matches("each_fma_e5m2_skylake", test_fma<e5m2_t>, nk_each_fma_e5m2_skylake);
#endif // NK_TARGET_SKYLAKE

#if NK_TARGET_SAPPHIRE
    run_if_matches("each_sum_e4m3_sapphire", test_sum<e4m3_t>, nk_each_sum_e4m3_sapphire);
#endif // NK_TARGET_SAPPHIRE

    // Serial always runs - baseline test
    run_if_matches("each_scale_f32_serial", test_scale<f32_t>, nk_each_scale_f32_serial);
    run_if_matches("each_sum_f32_serial", test_sum<f32_t>, nk_each_sum_f32_serial);
    run_if_matches("each_wsum_f32_serial", test_wsum<f32_t>, nk_each_blend_f32_serial);
    run_if_matches("each_fma_f32_serial", test_fma<f32_t>, nk_each_fma_f32_serial);
    run_if_matches("each_scale_e4m3_serial", test_scale<e4m3_t>, nk_each_scale_e4m3_serial);
    run_if_matches("each_scale_e5m2_serial", test_scale<e5m2_t>, nk_each_scale_e5m2_serial);
    run_if_matches("each_sum_e4m3_serial", test_sum<e4m3_t>, nk_each_sum_e4m3_serial);
    run_if_matches("each_sum_e5m2_serial", test_sum<e5m2_t>, nk_each_sum_e5m2_serial);
    run_if_matches("each_wsum_e4m3_serial", test_wsum<e4m3_t>, nk_each_blend_e4m3_serial);
    run_if_matches("each_wsum_e5m2_serial", test_wsum<e5m2_t>, nk_each_blend_e5m2_serial);
    run_if_matches("each_fma_e4m3_serial", test_fma<e4m3_t>, nk_each_fma_e4m3_serial);
    run_if_matches("each_fma_e5m2_serial", test_fma<e5m2_t>, nk_each_fma_e5m2_serial);

#endif // NK_DYNAMIC_DISPATCH
}

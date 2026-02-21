/**
 *  @brief KL-divergence and Jensen-Shannon tests.
 *  @file test/test_probability.cpp
 *  @author Ash Vardanian
 *  @date December 28, 2025
 */

#include "test.hpp"
#include "numkong/probability.hpp" // `nk::kld`

/**
 *  @brief Template for KL divergence test.
 *  KLD requires probability distributions: all values > 0, sum to 1.
 */
template <typename scalar_type_>
error_stats_t test_kld(typename scalar_type_::probability_kernel_t kernel) {
    using scalar_t = scalar_type_;
    using raw_t = typename scalar_t::raw_t;
    using result_t = typename scalar_t::probability_result_t;

    error_stats_t stats;
    std::mt19937 generator(global_config.seed);
    auto p = make_vector<scalar_t>(global_config.dense_dimensions),
         q = make_vector<scalar_t>(global_config.dense_dimensions);

    for (auto start = test_start_time(); within_time_budget(start);) {
        nk::fill_probability(generator, p.values_data(), global_config.dense_dimensions);
        nk::fill_probability(generator, q.values_data(), global_config.dense_dimensions);

        result_t result;
        kernel(p.raw_values_data(), q.raw_values_data(), global_config.dense_dimensions, &result.raw_);

        f118_t reference;
        nk::kld<scalar_t, f118_t, nk::no_simd_k>(p.values_data(), q.values_data(), global_config.dense_dimensions,
                                                 &reference);

        stats.accumulate(result, reference);
    }

    return stats;
}

/**
 *  @brief Template for Jensen-Shannon divergence test.
 *  JSD requires probability distributions: all values > 0, sum to 1.
 */
template <typename scalar_type_>
error_stats_t test_jsd(typename scalar_type_::probability_kernel_t kernel) {
    using scalar_t = scalar_type_;
    using raw_t = typename scalar_t::raw_t;
    using result_t = typename scalar_t::probability_result_t;

    error_stats_t stats;
    std::mt19937 generator(global_config.seed);
    auto p = make_vector<scalar_t>(global_config.dense_dimensions),
         q = make_vector<scalar_t>(global_config.dense_dimensions);

    for (auto start = test_start_time(); within_time_budget(start);) {
        nk::fill_probability(generator, p.values_data(), global_config.dense_dimensions);
        nk::fill_probability(generator, q.values_data(), global_config.dense_dimensions);

        result_t result;
        kernel(p.raw_values_data(), q.raw_values_data(), global_config.dense_dimensions, &result.raw_);

        f118_t reference;
        nk::jsd<scalar_t, f118_t, nk::no_simd_k>(p.values_data(), q.values_data(), global_config.dense_dimensions,
                                                 &reference);

        stats.accumulate(result, reference);
    }

    return stats;
}

void test_probability() {
    std::puts("");
    std::printf("Probability Divergences:\n");

#if NK_DYNAMIC_DISPATCH
    run_if_matches("kld_f32", test_kld<f32_t>, nk_kld_f32);
    run_if_matches("kld_f64", test_kld<f64_t>, nk_kld_f64);
    run_if_matches("jsd_f32", test_jsd<f32_t>, nk_jsd_f32);
    run_if_matches("jsd_f64", test_jsd<f64_t>, nk_jsd_f64);
#else

#if NK_TARGET_NEON
    run_if_matches("kld_f32_neon", test_kld<f32_t>, nk_kld_f32_neon);
    run_if_matches("jsd_f32_neon", test_jsd<f32_t>, nk_jsd_f32_neon);
#endif // NK_TARGET_NEON

#if NK_TARGET_NEONHALF
    run_if_matches("kld_f16_neonhalf", test_kld<f16_t>, nk_kld_f16_neonhalf);
    run_if_matches("jsd_f16_neonhalf", test_jsd<f16_t>, nk_jsd_f16_neonhalf);
#endif // NK_TARGET_NEONHALF

#if NK_TARGET_HASWELL
    run_if_matches("kld_f16_haswell", test_kld<f16_t>, nk_kld_f16_haswell);
    run_if_matches("kld_f64_haswell", test_kld<f64_t>, nk_kld_f64_haswell);
    run_if_matches("jsd_f16_haswell", test_jsd<f16_t>, nk_jsd_f16_haswell);
    run_if_matches("jsd_f64_haswell", test_jsd<f64_t>, nk_jsd_f64_haswell);
#endif // NK_TARGET_HASWELL

#if NK_TARGET_SKYLAKE
    run_if_matches("kld_f32_skylake", test_kld<f32_t>, nk_kld_f32_skylake);
    run_if_matches("kld_f64_skylake", test_kld<f64_t>, nk_kld_f64_skylake);
    run_if_matches("jsd_f32_skylake", test_jsd<f32_t>, nk_jsd_f32_skylake);
    run_if_matches("jsd_f64_skylake", test_jsd<f64_t>, nk_jsd_f64_skylake);
#endif // NK_TARGET_SKYLAKE

#if NK_TARGET_SAPPHIRE
    run_if_matches("kld_f16_sapphire", test_kld<f16_t>, nk_kld_f16_sapphire);
    run_if_matches("jsd_f16_sapphire", test_jsd<f16_t>, nk_jsd_f16_sapphire);
#endif // NK_TARGET_SAPPHIRE

#if NK_TARGET_RVV
    run_if_matches("kld_f32_rvv", test_kld<f32_t>, nk_kld_f32_rvv);
    run_if_matches("kld_f64_rvv", test_kld<f64_t>, nk_kld_f64_rvv);
    run_if_matches("kld_f16_rvv", test_kld<f16_t>, nk_kld_f16_rvv);
    run_if_matches("kld_bf16_rvv", test_kld<bf16_t>, nk_kld_bf16_rvv);
    run_if_matches("jsd_f32_rvv", test_jsd<f32_t>, nk_jsd_f32_rvv);
    run_if_matches("jsd_f64_rvv", test_jsd<f64_t>, nk_jsd_f64_rvv);
    run_if_matches("jsd_f16_rvv", test_jsd<f16_t>, nk_jsd_f16_rvv);
    run_if_matches("jsd_bf16_rvv", test_jsd<bf16_t>, nk_jsd_bf16_rvv);
#endif // NK_TARGET_RVV

    // Serial always runs - baseline test
    run_if_matches("kld_f32_serial", test_kld<f32_t>, nk_kld_f32_serial);
    run_if_matches("kld_f64_serial", test_kld<f64_t>, nk_kld_f64_serial);
    run_if_matches("jsd_f32_serial", test_jsd<f32_t>, nk_jsd_f32_serial);
    run_if_matches("jsd_f64_serial", test_jsd<f64_t>, nk_jsd_f64_serial);

#endif // NK_DYNAMIC_DISPATCH
}

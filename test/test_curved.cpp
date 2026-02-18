/**
 *  @brief Bilinear and Mahalanobis tests.
 *  @file test/test_curved.cpp
 *  @author Ash Vardanian
 *  @date December 28, 2025
 */

#include "test.hpp"
#include "numkong/curved.hpp" // `nk::bilinear`

/**
 *  @brief Template for bilinear form test: a^T * M * b
 */
template <typename scalar_type_>
error_stats_t test_bilinear(typename scalar_type_::curved_kernel_t kernel) {
    using scalar_t = scalar_type_;
    using raw_t = typename scalar_t::raw_t;
    using result_t = typename scalar_t::curved_result_t;
    using reference_t = std::conditional_t<scalar_t::is_complex(), f118c_t, f118_t>;

    error_stats_t stats;
    std::mt19937 generator(global_config.seed);

    auto a = make_vector<scalar_t>(dense_dimensions), b = make_vector<scalar_t>(dense_dimensions);
    auto m = make_vector<scalar_t>(dense_dimensions * dense_dimensions);
    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(generator, a);
        fill_random(generator, b);
        fill_random(generator, m);

        result_t result;
        kernel(a.raw_values_data(), b.raw_values_data(), m.raw_values_data(), dense_dimensions, &result.raw_);

        reference_t reference;
        nk::bilinear<scalar_t, reference_t, nk::no_simd_k>(a.values_data(), b.values_data(), m.values_data(),
                                                           dense_dimensions, &reference);

        stats.accumulate(result, reference);
    }

    return stats;
}

/**
 *  @brief Template for Mahalanobis distance test: sqrt((a-b)^T * M * (a-b))
 */
template <typename scalar_type_>
error_stats_t test_mahalanobis(typename scalar_type_::curved_kernel_t kernel) {
    using scalar_t = scalar_type_;
    using raw_t = typename scalar_t::raw_t;
    using result_t = typename scalar_t::curved_result_t;

    error_stats_t stats;
    std::mt19937 generator(global_config.seed);

    auto a = make_vector<scalar_t>(dense_dimensions), b = make_vector<scalar_t>(dense_dimensions);
    auto m = make_vector<scalar_t>(dense_dimensions * dense_dimensions);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(generator, a);
        fill_random(generator, b);
        fill_random(generator, m);

        result_t result;
        kernel(a.raw_values_data(), b.raw_values_data(), m.raw_values_data(), dense_dimensions, &result.raw_);

        f118_t reference;
        nk::mahalanobis<scalar_t, f118_t, nk::no_simd_k>(a.values_data(), b.values_data(), m.values_data(),
                                                         dense_dimensions, &reference);

        stats.accumulate(result, reference);
    }
    return stats;
}

void test_curved() {
    std::puts("");
    std::printf("Curved/Bilinear Forms:\n");

#if NK_DYNAMIC_DISPATCH
    run_if_matches("bilinear_f32", test_bilinear<f32_t>, nk_bilinear_f32);
    run_if_matches("bilinear_f64", test_bilinear<f64_t>, nk_bilinear_f64);
    run_if_matches("bilinear_f32c", test_bilinear<f32c_t>, nk_bilinear_f32c);
    run_if_matches("bilinear_f64c", test_bilinear<f64c_t>, nk_bilinear_f64c);
    run_if_matches("mahalanobis_f32", test_mahalanobis<f32_t>, nk_mahalanobis_f32);
    run_if_matches("mahalanobis_f64", test_mahalanobis<f64_t>, nk_mahalanobis_f64);
#else

#if NK_TARGET_NEON
    run_if_matches("bilinear_f32_neon", test_bilinear<f32_t>, nk_bilinear_f32_neon);
    run_if_matches("bilinear_f32c_neon", test_bilinear<f32c_t>, nk_bilinear_f32c_neon);
    run_if_matches("mahalanobis_f32_neon", test_mahalanobis<f32_t>, nk_mahalanobis_f32_neon);
#endif // NK_TARGET_NEON

#if NK_TARGET_NEONHALF
    run_if_matches("bilinear_f16_neonhalf", test_bilinear<f16_t>, nk_bilinear_f16_neonhalf);
    run_if_matches("bilinear_f16c_neonhalf", test_bilinear<f16c_t>, nk_bilinear_f16c_neonhalf);
    run_if_matches("mahalanobis_f16_neonhalf", test_mahalanobis<f16_t>, nk_mahalanobis_f16_neonhalf);
#endif // NK_TARGET_NEONHALF

#if NK_TARGET_NEONBFDOT
    run_if_matches("bilinear_bf16_neonbfdot", test_bilinear<bf16_t>, nk_bilinear_bf16_neonbfdot);
    run_if_matches("bilinear_bf16c_neonbfdot", test_bilinear<bf16c_t>, nk_bilinear_bf16c_neonbfdot);
    run_if_matches("mahalanobis_bf16_neonbfdot", test_mahalanobis<bf16_t>, nk_mahalanobis_bf16_neonbfdot);
#endif // NK_TARGET_NEONBFDOT

#if NK_TARGET_SKYLAKE
    run_if_matches("bilinear_f32_skylake", test_bilinear<f32_t>, nk_bilinear_f32_skylake);
    run_if_matches("bilinear_f64_skylake", test_bilinear<f64_t>, nk_bilinear_f64_skylake);
    run_if_matches("bilinear_f32c_skylake", test_bilinear<f32c_t>, nk_bilinear_f32c_skylake);
    run_if_matches("bilinear_f64c_skylake", test_bilinear<f64c_t>, nk_bilinear_f64c_skylake);
    run_if_matches("mahalanobis_f32_skylake", test_mahalanobis<f32_t>, nk_mahalanobis_f32_skylake);
    run_if_matches("mahalanobis_f64_skylake", test_mahalanobis<f64_t>, nk_mahalanobis_f64_skylake);
#endif // NK_TARGET_SKYLAKE

#if NK_TARGET_SMEF64
    run_if_matches("bilinear_smef64_f32", test_bilinear<f32_t>, nk_bilinear_f32_smef64);
    run_if_matches("bilinear_smef64_f32c", test_bilinear<f32c_t>, nk_bilinear_f32c_smef64);
    run_if_matches("mahalanobis_f32_smef64", test_mahalanobis<f32_t>, nk_mahalanobis_f32_smef64);
    run_if_matches("bilinear_f64_smef64", test_bilinear<f64_t>, nk_bilinear_f64_smef64);
    run_if_matches("bilinear_f64c_smef64", test_bilinear<f64c_t>, nk_bilinear_f64c_smef64);
    run_if_matches("mahalanobis_f64_smef64", test_mahalanobis<f64_t>, nk_mahalanobis_f64_smef64);
#endif // NK_TARGET_SMEF64

    // Serial always runs - baseline test
    run_if_matches("bilinear_f32_serial", test_bilinear<f32_t>, nk_bilinear_f32_serial);
    run_if_matches("bilinear_f64_serial", test_bilinear<f64_t>, nk_bilinear_f64_serial);
    run_if_matches("bilinear_f32c_serial", test_bilinear<f32c_t>, nk_bilinear_f32c_serial);
    run_if_matches("bilinear_f64c_serial", test_bilinear<f64c_t>, nk_bilinear_f64c_serial);
    run_if_matches("mahalanobis_f32_serial", test_mahalanobis<f32_t>, nk_mahalanobis_f32_serial);
    run_if_matches("mahalanobis_f64_serial", test_mahalanobis<f64_t>, nk_mahalanobis_f64_serial);
    run_if_matches("bilinear_f16_serial", test_bilinear<f16_t>, nk_bilinear_f16_serial);
    run_if_matches("bilinear_f16c_serial", test_bilinear<f16c_t>, nk_bilinear_f16c_serial);
    run_if_matches("mahalanobis_f16_serial", test_mahalanobis<f16_t>, nk_mahalanobis_f16_serial);
    run_if_matches("bilinear_bf16_serial", test_bilinear<bf16_t>, nk_bilinear_bf16_serial);
    run_if_matches("bilinear_bf16c_serial", test_bilinear<bf16c_t>, nk_bilinear_bf16c_serial);
    run_if_matches("mahalanobis_bf16_serial", test_mahalanobis<bf16_t>, nk_mahalanobis_bf16_serial);

#endif // NK_DYNAMIC_DISPATCH
}

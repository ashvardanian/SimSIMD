/**
 *  @brief Trigonometry tests (sin, cos, atan).
 *  @file test/test_trigonometry.cpp
 *  @author Ash Vardanian
 *  @date February 6, 2026
 */

#include "test.hpp"
#include "numkong/trigonometry.hpp"

/**
 *  @brief Test sine approximation kernel against `nk::sin<scalar_t, f118_t, nk::no_simd_k>`.
 */
template <typename scalar_type_>
error_stats_t test_sin(typename scalar_type_::trigonometry_kernel_t kernel) {
    using scalar_t = scalar_type_;
    using raw_t = typename scalar_t::raw_t;

    error_stats_t stats;
    std::mt19937 generator(global_config.seed);
    auto inputs = make_vector<scalar_t>(dense_dimensions);
    auto outputs = make_vector<scalar_t>(dense_dimensions), reference = make_vector<scalar_t>(dense_dimensions);

    for (auto start = test_start_time(); within_time_budget(start);) {
        nk::fill_uniform(generator, inputs.values_data(), inputs.size_values(), -scalar_t::two_pi_k(),
                         scalar_t::two_pi_k());

        kernel(inputs.raw_values_data(), dense_dimensions, outputs.raw_values_data());
        nk::sin<scalar_t, f118_t, nk::no_simd_k>(inputs.values_data(), dense_dimensions, reference.values_data());

        for (std::size_t i = 0; i < dense_dimensions; i++) stats.accumulate(outputs[i], reference[i]);
    }
    return stats;
}

/**
 *  @brief Test cosine approximation kernel against `nk::cos<scalar_t, f118_t, nk::no_simd_k>`.
 */
template <typename scalar_type_>
error_stats_t test_cos(typename scalar_type_::trigonometry_kernel_t kernel) {
    using scalar_t = scalar_type_;
    using raw_t = typename scalar_t::raw_t;

    error_stats_t stats;
    std::mt19937 generator(global_config.seed);
    auto inputs = make_vector<scalar_t>(dense_dimensions);
    auto outputs = make_vector<scalar_t>(dense_dimensions), reference = make_vector<scalar_t>(dense_dimensions);

    for (auto start = test_start_time(); within_time_budget(start);) {
        nk::fill_uniform(generator, inputs.values_data(), inputs.size_values(), -scalar_t::two_pi_k(),
                         scalar_t::two_pi_k());

        kernel(inputs.raw_values_data(), dense_dimensions, outputs.raw_values_data());
        nk::cos<scalar_t, f118_t, nk::no_simd_k>(inputs.values_data(), dense_dimensions, reference.values_data());

        for (std::size_t i = 0; i < dense_dimensions; i++) stats.accumulate(outputs[i], reference[i]);
    }
    return stats;
}

/**
 *  @brief Test atan approximation kernel against `nk::atan<scalar_t, f118_t, nk::no_simd_k>`.
 */
template <typename scalar_type_>
error_stats_t test_atan(typename scalar_type_::trigonometry_kernel_t kernel) {
    using scalar_t = scalar_type_;
    using raw_t = typename scalar_t::raw_t;

    error_stats_t stats;
    std::mt19937 generator(global_config.seed);
    auto inputs = make_vector<scalar_t>(dense_dimensions);
    auto outputs = make_vector<scalar_t>(dense_dimensions), reference = make_vector<scalar_t>(dense_dimensions);

    for (auto start = test_start_time(); within_time_budget(start);) {
        nk::fill_uniform(generator, inputs.values_data(), inputs.size_values(), scalar_t(-10.0), scalar_t(10.0));

        kernel(inputs.raw_values_data(), dense_dimensions, outputs.raw_values_data());
        nk::atan<scalar_t, f118_t, nk::no_simd_k>(inputs.values_data(), dense_dimensions, reference.values_data());

        for (std::size_t i = 0; i < dense_dimensions; i++) stats.accumulate(outputs[i], reference[i]);
    }
    return stats;
}

void test_trigonometry() {
    std::puts("");
    std::printf("Trigonometry:\n");

#if NK_DYNAMIC_DISPATCH
    run_if_matches("each_sin_f32", test_sin<f32_t>, nk_each_sin_f32);
    run_if_matches("each_cos_f32", test_cos<f32_t>, nk_each_cos_f32);
    run_if_matches("each_atan_f32", test_atan<f32_t>, nk_each_atan_f32);
    run_if_matches("each_sin_f64", test_sin<f64_t>, nk_each_sin_f64);
    run_if_matches("each_cos_f64", test_cos<f64_t>, nk_each_cos_f64);
    run_if_matches("each_atan_f64", test_atan<f64_t>, nk_each_atan_f64);
#else

#if NK_TARGET_NEON
    run_if_matches("each_sin_f32_neon", test_sin<f32_t>, nk_each_sin_f32_neon);
    run_if_matches("each_cos_f32_neon", test_cos<f32_t>, nk_each_cos_f32_neon);
    run_if_matches("each_atan_f32_neon", test_atan<f32_t>, nk_each_atan_f32_neon);
    run_if_matches("each_sin_f64_neon", test_sin<f64_t>, nk_each_sin_f64_neon);
    run_if_matches("each_cos_f64_neon", test_cos<f64_t>, nk_each_cos_f64_neon);
    run_if_matches("each_atan_f64_neon", test_atan<f64_t>, nk_each_atan_f64_neon);
#endif

#if NK_TARGET_HASWELL
    run_if_matches("each_sin_f32_haswell", test_sin<f32_t>, nk_each_sin_f32_haswell);
    run_if_matches("each_cos_f32_haswell", test_cos<f32_t>, nk_each_cos_f32_haswell);
    run_if_matches("each_atan_f32_haswell", test_atan<f32_t>, nk_each_atan_f32_haswell);
    run_if_matches("each_sin_f64_haswell", test_sin<f64_t>, nk_each_sin_f64_haswell);
    run_if_matches("each_cos_f64_haswell", test_cos<f64_t>, nk_each_cos_f64_haswell);
    run_if_matches("each_atan_f64_haswell", test_atan<f64_t>, nk_each_atan_f64_haswell);
#endif

#if NK_TARGET_SKYLAKE
    run_if_matches("each_sin_f32_skylake", test_sin<f32_t>, nk_each_sin_f32_skylake);
    run_if_matches("each_cos_f32_skylake", test_cos<f32_t>, nk_each_cos_f32_skylake);
    run_if_matches("each_atan_f32_skylake", test_atan<f32_t>, nk_each_atan_f32_skylake);
    run_if_matches("each_sin_f64_skylake", test_sin<f64_t>, nk_each_sin_f64_skylake);
    run_if_matches("each_cos_f64_skylake", test_cos<f64_t>, nk_each_cos_f64_skylake);
    run_if_matches("each_atan_f64_skylake", test_atan<f64_t>, nk_each_atan_f64_skylake);
#endif

#if NK_TARGET_SAPPHIRE
    run_if_matches("each_sin_f16_sapphire", test_sin<f16_t>, nk_each_sin_f16_sapphire);
    run_if_matches("each_cos_f16_sapphire", test_cos<f16_t>, nk_each_cos_f16_sapphire);
    run_if_matches("each_atan_f16_sapphire", test_atan<f16_t>, nk_each_atan_f16_sapphire);
#endif

#if NK_TARGET_V128RELAXED
    run_if_matches("each_sin_f32_v128relaxed", test_sin<f32_t>, nk_each_sin_f32_v128relaxed);
    run_if_matches("each_cos_f32_v128relaxed", test_cos<f32_t>, nk_each_cos_f32_v128relaxed);
    run_if_matches("each_atan_f32_v128relaxed", test_atan<f32_t>, nk_each_atan_f32_v128relaxed);
    run_if_matches("each_sin_f64_v128relaxed", test_sin<f64_t>, nk_each_sin_f64_v128relaxed);
    run_if_matches("each_cos_f64_v128relaxed", test_cos<f64_t>, nk_each_cos_f64_v128relaxed);
    run_if_matches("each_atan_f64_v128relaxed", test_atan<f64_t>, nk_each_atan_f64_v128relaxed);
#endif // NK_TARGET_V128RELAXED

    run_if_matches("each_sin_f32_serial", test_sin<f32_t>, nk_each_sin_f32_serial);
    run_if_matches("each_cos_f32_serial", test_cos<f32_t>, nk_each_cos_f32_serial);
    run_if_matches("each_atan_f32_serial", test_atan<f32_t>, nk_each_atan_f32_serial);
    run_if_matches("each_sin_f64_serial", test_sin<f64_t>, nk_each_sin_f64_serial);
    run_if_matches("each_cos_f64_serial", test_cos<f64_t>, nk_each_cos_f64_serial);
    run_if_matches("each_atan_f64_serial", test_atan<f64_t>, nk_each_atan_f64_serial);
    run_if_matches("each_sin_f16_serial", test_sin<f16_t>, nk_each_sin_f16_serial);
    run_if_matches("each_cos_f16_serial", test_cos<f16_t>, nk_each_cos_f16_serial);
    run_if_matches("each_atan_f16_serial", test_atan<f16_t>, nk_each_atan_f16_serial);

#endif
}

/**
 *  @brief Haversine and Vincenty distance tests.
 *  @file test/test_geospatial.cpp
 *  @author Ash Vardanian
 *  @date December 28, 2025
 */

#include "test.hpp"
#include "numkong/geospatial.hpp" // `nk::haversine`

/**
 *  @brief Test Haversine distance.
 */
template <typename scalar_type_>
error_stats_t test_haversine(typename scalar_type_::haversine_kernel_t kernel) {
    using scalar_t = scalar_type_;
    using raw_t = typename scalar_t::raw_t;

    error_stats_t stats;
    std::mt19937 generator(global_config.seed);
    auto a_lats = make_vector<scalar_t>(dense_dimensions), a_lons = make_vector<scalar_t>(dense_dimensions);
    auto b_lats = make_vector<scalar_t>(dense_dimensions), b_lons = make_vector<scalar_t>(dense_dimensions);
    auto results = make_vector<scalar_t>(dense_dimensions), reference = make_vector<scalar_t>(dense_dimensions);

    for (auto start = test_start_time(); within_time_budget(start);) {
        nk::fill_coordinates(generator, a_lats.values_data(), a_lons.values_data(), dense_dimensions);
        nk::fill_coordinates(generator, b_lats.values_data(), b_lons.values_data(), dense_dimensions);

        kernel(a_lats.raw_values_data(), a_lons.raw_values_data(), b_lats.raw_values_data(), b_lons.raw_values_data(),
               dense_dimensions, results.raw_values_data());
        nk::haversine<scalar_t, f118_t, nk::no_simd_k>(a_lats.values_data(), a_lons.values_data(), b_lats.values_data(),
                                                       b_lons.values_data(), dense_dimensions, reference.values_data());

        for (std::size_t i = 0; i < dense_dimensions; i++) stats.accumulate(results[i], reference[i]);
    }
    return stats;
}

/**
 *  @brief Test Vincenty distance.
 */
template <typename scalar_type_>
error_stats_t test_vincenty(typename scalar_type_::haversine_kernel_t kernel) {
    using scalar_t = scalar_type_;
    using raw_t = typename scalar_t::raw_t;

    error_stats_t stats;
    std::mt19937 generator(global_config.seed);
    auto a_lats = make_vector<scalar_t>(dense_dimensions), a_lons = make_vector<scalar_t>(dense_dimensions);
    auto b_lats = make_vector<scalar_t>(dense_dimensions), b_lons = make_vector<scalar_t>(dense_dimensions);
    auto results = make_vector<scalar_t>(dense_dimensions), reference = make_vector<scalar_t>(dense_dimensions);

    for (auto start = test_start_time(); within_time_budget(start);) {
        nk::fill_coordinates(generator, a_lats.values_data(), a_lons.values_data(), dense_dimensions);
        nk::fill_coordinates(generator, b_lats.values_data(), b_lons.values_data(), dense_dimensions);

        kernel(a_lats.raw_values_data(), a_lons.raw_values_data(), b_lats.raw_values_data(), b_lons.raw_values_data(),
               dense_dimensions, results.raw_values_data());
        nk::vincenty<scalar_t, f118_t, nk::no_simd_k>(a_lats.values_data(), a_lons.values_data(), b_lats.values_data(),
                                                      b_lons.values_data(), dense_dimensions, reference.values_data());

        for (std::size_t i = 0; i < dense_dimensions; i++) stats.accumulate(results[i], reference[i]);
    }
    return stats;
}

void test_geospatial() {
    std::puts("");
    std::printf("Geospatial Functions:\n");

#if NK_DYNAMIC_DISPATCH
    run_if_matches("haversine_f64", test_haversine<f64_t>, nk_haversine_f64);
    run_if_matches("haversine_f32", test_haversine<f32_t>, nk_haversine_f32);
    run_if_matches("vincenty_f64", test_vincenty<f64_t>, nk_vincenty_f64);
    run_if_matches("vincenty_f32", test_vincenty<f32_t>, nk_vincenty_f32);
#else

#if NK_TARGET_NEON
    run_if_matches("haversine_f64_neon", test_haversine<f64_t>, nk_haversine_f64_neon);
    run_if_matches("haversine_f32_neon", test_haversine<f32_t>, nk_haversine_f32_neon);
    run_if_matches("vincenty_f64_neon", test_vincenty<f64_t>, nk_vincenty_f64_neon);
    run_if_matches("vincenty_f32_neon", test_vincenty<f32_t>, nk_vincenty_f32_neon);
#endif // NK_TARGET_NEON

#if NK_TARGET_HASWELL
    run_if_matches("haversine_f64_haswell", test_haversine<f64_t>, nk_haversine_f64_haswell);
    run_if_matches("haversine_f32_haswell", test_haversine<f32_t>, nk_haversine_f32_haswell);
    run_if_matches("vincenty_f64_haswell", test_vincenty<f64_t>, nk_vincenty_f64_haswell);
    run_if_matches("vincenty_f32_haswell", test_vincenty<f32_t>, nk_vincenty_f32_haswell);
#endif // NK_TARGET_HASWELL

#if NK_TARGET_SKYLAKE
    run_if_matches("haversine_f64_skylake", test_haversine<f64_t>, nk_haversine_f64_skylake);
    run_if_matches("haversine_f32_skylake", test_haversine<f32_t>, nk_haversine_f32_skylake);
    run_if_matches("vincenty_f64_skylake", test_vincenty<f64_t>, nk_vincenty_f64_skylake);
    run_if_matches("vincenty_f32_skylake", test_vincenty<f32_t>, nk_vincenty_f32_skylake);
#endif // NK_TARGET_SKYLAKE

#if NK_TARGET_V128RELAXED
    run_if_matches("haversine_f64_v128relaxed", test_haversine<f64_t>, nk_haversine_f64_v128relaxed);
    run_if_matches("haversine_f32_v128relaxed", test_haversine<f32_t>, nk_haversine_f32_v128relaxed);
    run_if_matches("vincenty_f64_v128relaxed", test_vincenty<f64_t>, nk_vincenty_f64_v128relaxed);
    run_if_matches("vincenty_f32_v128relaxed", test_vincenty<f32_t>, nk_vincenty_f32_v128relaxed);
#endif // NK_TARGET_V128RELAXED

    // Serial always runs - baseline test
    run_if_matches("haversine_f64_serial", test_haversine<f64_t>, nk_haversine_f64_serial);
    run_if_matches("haversine_f32_serial", test_haversine<f32_t>, nk_haversine_f32_serial);
    run_if_matches("vincenty_f64_serial", test_vincenty<f64_t>, nk_vincenty_f64_serial);
    run_if_matches("vincenty_f32_serial", test_vincenty<f32_t>, nk_vincenty_f32_serial);

#endif // NK_DYNAMIC_DISPATCH
}

/**
 *  @brief RMSD, Kabsch, and Umeyama alignment tests.
 *  @file test/test_mesh.cpp
 *  @author Ash Vardanian
 *  @date December 28, 2025
 */

#include "test.hpp"
#include "numkong/mesh.hpp" // `nk::rmsd`

/**
 *  @brief Test RMSD kernel.
 */
template <typename scalar_type_>
error_stats_t test_rmsd(typename scalar_type_::mesh_kernel_t kernel) {
    using scalar_t = scalar_type_;
    using output_t = typename scalar_t::dot_result_t;

    error_stats_t stats;
    std::mt19937 generator(global_config.seed);

    std::size_t n = mesh_points;
    auto a = make_vector<scalar_t>(n * 3), b = make_vector<scalar_t>(n * 3);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(generator, a);
        fill_random(generator, b);

        output_t a_centroid[3], b_centroid[3], rot[9], scale, result;
        kernel(a.raw_values_data(), b.raw_values_data(), n, &a_centroid[0].raw_, &b_centroid[0].raw_, &rot[0].raw_,
               &scale.raw_, &result.raw_);

        f118_t a_centroid_ref[3], b_centroid_ref[3], rot_ref[9], scale_ref, reference;
        nk::rmsd<scalar_t, f118_t, nk::no_simd_k>(a.values_data(), b.values_data(), n, a_centroid_ref, b_centroid_ref,
                                                  rot_ref, &scale_ref, &reference);

        stats.accumulate(result, reference);
    }
    return stats;
}

/**
 *  @brief Test Kabsch alignment kernel.
 */
template <typename scalar_type_>
error_stats_t test_kabsch(typename scalar_type_::mesh_kernel_t kernel) {
    using scalar_t = scalar_type_;
    using output_t = typename scalar_t::dot_result_t;

    error_stats_t stats;
    std::mt19937 generator(global_config.seed);

    std::size_t n = mesh_points;
    auto a = make_vector<scalar_t>(n * 3), b = make_vector<scalar_t>(n * 3);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(generator, a);
        fill_random(generator, b);

        output_t a_centroid[3], b_centroid[3], rot[9], scale, result;
        kernel(a.raw_values_data(), b.raw_values_data(), n, &a_centroid[0].raw_, &b_centroid[0].raw_, &rot[0].raw_,
               &scale.raw_, &result.raw_);

        f118_t a_centroid_ref[3], b_centroid_ref[3], rot_ref[9], scale_ref, reference;
        nk::kabsch<scalar_t, f118_t, nk::no_simd_k>(a.values_data(), b.values_data(), n, a_centroid_ref, b_centroid_ref,
                                                    rot_ref, &scale_ref, &reference);

        stats.accumulate(result, reference);
    }
    return stats;
}

/**
 *  @brief Test Umeyama alignment kernel.
 */
template <typename scalar_type_>
error_stats_t test_umeyama(typename scalar_type_::mesh_kernel_t kernel) {
    using scalar_t = scalar_type_;
    using output_t = typename scalar_t::dot_result_t;

    error_stats_t stats;
    std::mt19937 generator(global_config.seed);

    std::size_t n = mesh_points;
    auto a = make_vector<scalar_t>(n * 3), b = make_vector<scalar_t>(n * 3);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(generator, a);
        fill_random(generator, b);

        output_t a_centroid[3], b_centroid[3], rot[9], scale, result;
        kernel(a.raw_values_data(), b.raw_values_data(), n, &a_centroid[0].raw_, &b_centroid[0].raw_, &rot[0].raw_,
               &scale.raw_, &result.raw_);

        f118_t a_centroid_ref[3], b_centroid_ref[3], rot_ref[9], scale_ref, reference;
        nk::umeyama<scalar_t, f118_t, nk::no_simd_k>(a.values_data(), b.values_data(), n, a_centroid_ref,
                                                     b_centroid_ref, rot_ref, &scale_ref, &reference);

        stats.accumulate(result, reference);
    }
    return stats;
}

void test_mesh() {
    std::puts("");
    std::printf("Mesh Operations:\n");

#if NK_DYNAMIC_DISPATCH
    run_if_matches("rmsd_f64", test_rmsd<f64_t>, nk_rmsd_f64);
    run_if_matches("rmsd_f32", test_rmsd<f32_t>, nk_rmsd_f32);
    run_if_matches("kabsch_f64", test_kabsch<f64_t>, nk_kabsch_f64);
    run_if_matches("kabsch_f32", test_kabsch<f32_t>, nk_kabsch_f32);
    run_if_matches("umeyama_f64", test_umeyama<f64_t>, nk_umeyama_f64);
    run_if_matches("umeyama_f32", test_umeyama<f32_t>, nk_umeyama_f32);
#else

#if NK_TARGET_NEON
    run_if_matches("rmsd_f64_neon", test_rmsd<f64_t>, nk_rmsd_f64_neon);
    run_if_matches("rmsd_f32_neon", test_rmsd<f32_t>, nk_rmsd_f32_neon);
    run_if_matches("kabsch_f64_neon", test_kabsch<f64_t>, nk_kabsch_f64_neon);
    run_if_matches("kabsch_f32_neon", test_kabsch<f32_t>, nk_kabsch_f32_neon);
    run_if_matches("umeyama_f64_neon", test_umeyama<f64_t>, nk_umeyama_f64_neon);
    run_if_matches("umeyama_f32_neon", test_umeyama<f32_t>, nk_umeyama_f32_neon);
#endif // NK_TARGET_NEON

#if NK_TARGET_HASWELL
    run_if_matches("rmsd_f64_haswell", test_rmsd<f64_t>, nk_rmsd_f64_haswell);
    run_if_matches("rmsd_f32_haswell", test_rmsd<f32_t>, nk_rmsd_f32_haswell);
    run_if_matches("kabsch_f64_haswell", test_kabsch<f64_t>, nk_kabsch_f64_haswell);
    run_if_matches("kabsch_f32_haswell", test_kabsch<f32_t>, nk_kabsch_f32_haswell);
    run_if_matches("umeyama_f64_haswell", test_umeyama<f64_t>, nk_umeyama_f64_haswell);
    run_if_matches("umeyama_f32_haswell", test_umeyama<f32_t>, nk_umeyama_f32_haswell);
    run_if_matches("rmsd_f16_haswell", test_rmsd<f16_t>, nk_rmsd_f16_haswell);
    run_if_matches("kabsch_f16_haswell", test_kabsch<f16_t>, nk_kabsch_f16_haswell);
    run_if_matches("umeyama_f16_haswell", test_umeyama<f16_t>, nk_umeyama_f16_haswell);
    run_if_matches("rmsd_bf16_haswell", test_rmsd<bf16_t>, nk_rmsd_bf16_haswell);
    run_if_matches("kabsch_bf16_haswell", test_kabsch<bf16_t>, nk_kabsch_bf16_haswell);
    run_if_matches("umeyama_bf16_haswell", test_umeyama<bf16_t>, nk_umeyama_bf16_haswell);
#endif // NK_TARGET_HASWELL

#if NK_TARGET_SKYLAKE
    run_if_matches("rmsd_f64_skylake", test_rmsd<f64_t>, nk_rmsd_f64_skylake);
    run_if_matches("rmsd_f32_skylake", test_rmsd<f32_t>, nk_rmsd_f32_skylake);
    run_if_matches("kabsch_f64_skylake", test_kabsch<f64_t>, nk_kabsch_f64_skylake);
    run_if_matches("kabsch_f32_skylake", test_kabsch<f32_t>, nk_kabsch_f32_skylake);
    run_if_matches("umeyama_f64_skylake", test_umeyama<f64_t>, nk_umeyama_f64_skylake);
    run_if_matches("umeyama_f32_skylake", test_umeyama<f32_t>, nk_umeyama_f32_skylake);
#endif // NK_TARGET_SKYLAKE

    // Serial always runs - baseline test
    run_if_matches("rmsd_f64_serial", test_rmsd<f64_t>, nk_rmsd_f64_serial);
    run_if_matches("rmsd_f32_serial", test_rmsd<f32_t>, nk_rmsd_f32_serial);
    run_if_matches("kabsch_f64_serial", test_kabsch<f64_t>, nk_kabsch_f64_serial);
    run_if_matches("kabsch_f32_serial", test_kabsch<f32_t>, nk_kabsch_f32_serial);
    run_if_matches("umeyama_f64_serial", test_umeyama<f64_t>, nk_umeyama_f64_serial);
    run_if_matches("umeyama_f32_serial", test_umeyama<f32_t>, nk_umeyama_f32_serial);

#endif // NK_DYNAMIC_DISPATCH
}

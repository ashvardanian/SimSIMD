/**
 *  @brief Spatial distance tests.
 *  @file test/test_spatial.cpp
 *  @author Ash Vardanian
 *  @date February 6, 2026
 */

#include "test.hpp"
#include "numkong/spatial.hpp"

/**
 *  @brief Unified squared Euclidean distance test for float types.
 *  Works with f32_t, f64_t, f16_t, bf16_t wrapper types.
 */
template <typename scalar_type_>
error_stats_t test_sqeuclidean(typename scalar_type_::sqeuclidean_kernel_t kernel) {
    using scalar_t = scalar_type_;
    using raw_t = typename scalar_t::raw_t;
    using result_t = typename scalar_t::sqeuclidean_result_t;

    error_stats_t stats;
    std::mt19937 generator(global_config.seed);
    auto a = make_vector<scalar_t>(dense_dimensions), b = make_vector<scalar_t>(dense_dimensions);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(generator, a);
        fill_random(generator, b);

        result_t result;
        kernel(a.raw_values_data(), b.raw_values_data(), dense_dimensions, &result.raw_);

        f118_t reference;
        nk::sqeuclidean<scalar_t, f118_t, nk::no_simd_k>(a.values_data(), b.values_data(), dense_dimensions,
                                                         &reference);

        stats.accumulate(result, reference);
    }
    return stats;
}

/**
 *  @brief Unified angular (cosine) distance test for float types.
 *  Works with f32_t, f64_t, f16_t, bf16_t wrapper types.
 */
template <typename scalar_type_>
error_stats_t test_angular(typename scalar_type_::angular_kernel_t kernel) {
    using scalar_t = scalar_type_;
    using raw_t = typename scalar_t::raw_t;
    using result_t = typename scalar_t::angular_result_t;

    error_stats_t stats;
    std::mt19937 generator(global_config.seed);
    auto a = make_vector<scalar_t>(dense_dimensions), b = make_vector<scalar_t>(dense_dimensions);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(generator, a);
        fill_random(generator, b);

        result_t result;
        kernel(a.raw_values_data(), b.raw_values_data(), dense_dimensions, &result.raw_);

        f118_t reference;
        nk::angular<scalar_t, f118_t, nk::no_simd_k>(a.values_data(), b.values_data(), dense_dimensions, &reference);

        stats.accumulate(result, reference);
    }
    return stats;
}

void test_spatial() {
    std::puts("");
    std::printf("Spatial Distances:\n");

#if NK_DYNAMIC_DISPATCH
    // Dynamic dispatch - only test the dispatcher itself
    run_if_matches("sqeuclidean_f32", test_sqeuclidean<f32_t>, nk_sqeuclidean_f32);
    run_if_matches("sqeuclidean_f64", test_sqeuclidean<f64_t>, nk_sqeuclidean_f64);
    run_if_matches("sqeuclidean_f16", test_sqeuclidean<f16_t>, nk_sqeuclidean_f16);
    run_if_matches("sqeuclidean_bf16", test_sqeuclidean<bf16_t>, nk_sqeuclidean_bf16);
    run_if_matches("sqeuclidean_e2m3", test_sqeuclidean<e2m3_t>, nk_sqeuclidean_e2m3);
    run_if_matches("sqeuclidean_e3m2", test_sqeuclidean<e3m2_t>, nk_sqeuclidean_e3m2);
    run_if_matches("angular_f32", test_angular<f32_t>, nk_angular_f32);
    run_if_matches("angular_f64", test_angular<f64_t>, nk_angular_f64);
    run_if_matches("angular_f16", test_angular<f16_t>, nk_angular_f16);
    run_if_matches("angular_bf16", test_angular<bf16_t>, nk_angular_bf16);
    run_if_matches("angular_e2m3", test_angular<e2m3_t>, nk_angular_e2m3);
    run_if_matches("angular_e3m2", test_angular<e3m2_t>, nk_angular_e3m2);
    run_if_matches("sqeuclidean_i4", test_sqeuclidean<i4x2_t>, nk_sqeuclidean_i4);
    run_if_matches("sqeuclidean_u4", test_sqeuclidean<u4x2_t>, nk_sqeuclidean_u4);
    run_if_matches("angular_i4", test_angular<i4x2_t>, nk_angular_i4);
    run_if_matches("angular_u4", test_angular<u4x2_t>, nk_angular_u4);
#else
    // Static compilation - test all available ISA variants

#if NK_TARGET_NEON
    run_if_matches("sqeuclidean_f32_neon", test_sqeuclidean<f32_t>, nk_sqeuclidean_f32_neon);
    run_if_matches("sqeuclidean_f64_neon", test_sqeuclidean<f64_t>, nk_sqeuclidean_f64_neon);
    run_if_matches("angular_f32_neon", test_angular<f32_t>, nk_angular_f32_neon);
    run_if_matches("angular_f64_neon", test_angular<f64_t>, nk_angular_f64_neon);
#endif // NK_TARGET_NEON

#if NK_TARGET_NEONHALF
    run_if_matches("sqeuclidean_f16_neonhalf", test_sqeuclidean<f16_t>, nk_sqeuclidean_f16_neonhalf);
    run_if_matches("angular_f16_neonhalf", test_angular<f16_t>, nk_angular_f16_neonhalf);
#endif // NK_TARGET_NEONHALF

#if NK_TARGET_NEONBFDOT
    run_if_matches("sqeuclidean_bf16_neonbfdot", test_sqeuclidean<bf16_t>, nk_sqeuclidean_bf16_neonbfdot);
    run_if_matches("angular_bf16_neonbfdot", test_angular<bf16_t>, nk_angular_bf16_neonbfdot);
#endif // NK_TARGET_NEONBFDOT

#if NK_TARGET_HASWELL
    run_if_matches("sqeuclidean_f32_haswell", test_sqeuclidean<f32_t>, nk_sqeuclidean_f32_haswell);
    run_if_matches("sqeuclidean_f64_haswell", test_sqeuclidean<f64_t>, nk_sqeuclidean_f64_haswell);
    run_if_matches("sqeuclidean_f16_haswell", test_sqeuclidean<f16_t>, nk_sqeuclidean_f16_haswell);
    run_if_matches("sqeuclidean_bf16_haswell", test_sqeuclidean<bf16_t>, nk_sqeuclidean_bf16_haswell);
    run_if_matches("angular_f32_haswell", test_angular<f32_t>, nk_angular_f32_haswell);
    run_if_matches("angular_f64_haswell", test_angular<f64_t>, nk_angular_f64_haswell);
    run_if_matches("angular_f16_haswell", test_angular<f16_t>, nk_angular_f16_haswell);
    run_if_matches("angular_bf16_haswell", test_angular<bf16_t>, nk_angular_bf16_haswell);
#endif // NK_TARGET_HASWELL

#if NK_TARGET_SKYLAKE
    run_if_matches("sqeuclidean_f32_skylake", test_sqeuclidean<f32_t>, nk_sqeuclidean_f32_skylake);
    run_if_matches("sqeuclidean_f64_skylake", test_sqeuclidean<f64_t>, nk_sqeuclidean_f64_skylake);
    run_if_matches("angular_f32_skylake", test_angular<f32_t>, nk_angular_f32_skylake);
    run_if_matches("angular_f64_skylake", test_angular<f64_t>, nk_angular_f64_skylake);
#endif // NK_TARGET_SKYLAKE

#if NK_TARGET_ICELAKE
    run_if_matches("sqeuclidean_i4_icelake", test_sqeuclidean<i4x2_t>, nk_sqeuclidean_i4_icelake);
    run_if_matches("sqeuclidean_u4_icelake", test_sqeuclidean<u4x2_t>, nk_sqeuclidean_u4_icelake);
    run_if_matches("angular_i4_icelake", test_angular<i4x2_t>, nk_angular_i4_icelake);
    run_if_matches("angular_u4_icelake", test_angular<u4x2_t>, nk_angular_u4_icelake);
#endif // NK_TARGET_ICELAKE

#if NK_TARGET_RVV
    run_if_matches("sqeuclidean_f32_rvv", test_sqeuclidean<f32_t>, nk_sqeuclidean_f32_rvv);
    run_if_matches("sqeuclidean_f64_rvv", test_sqeuclidean<f64_t>, nk_sqeuclidean_f64_rvv);
    run_if_matches("sqeuclidean_f16_rvv", test_sqeuclidean<f16_t>, nk_sqeuclidean_f16_rvv);
    run_if_matches("sqeuclidean_bf16_rvv", test_sqeuclidean<bf16_t>, nk_sqeuclidean_bf16_rvv);
    run_if_matches("sqeuclidean_i4_rvv", test_sqeuclidean<i4x2_t>, nk_sqeuclidean_i4_rvv);
    run_if_matches("sqeuclidean_u4_rvv", test_sqeuclidean<u4x2_t>, nk_sqeuclidean_u4_rvv);
    run_if_matches("angular_f32_rvv", test_angular<f32_t>, nk_angular_f32_rvv);
    run_if_matches("angular_f64_rvv", test_angular<f64_t>, nk_angular_f64_rvv);
    run_if_matches("angular_f16_rvv", test_angular<f16_t>, nk_angular_f16_rvv);
    run_if_matches("angular_bf16_rvv", test_angular<bf16_t>, nk_angular_bf16_rvv);
    run_if_matches("angular_i4_rvv", test_angular<i4x2_t>, nk_angular_i4_rvv);
    run_if_matches("angular_u4_rvv", test_angular<u4x2_t>, nk_angular_u4_rvv);
#endif // NK_TARGET_RVV

#if NK_TARGET_V128RELAXED
    run_if_matches("sqeuclidean_f32_v128relaxed", test_sqeuclidean<f32_t>, nk_sqeuclidean_f32_v128relaxed);
    run_if_matches("sqeuclidean_f64_v128relaxed", test_sqeuclidean<f64_t>, nk_sqeuclidean_f64_v128relaxed);
    run_if_matches("sqeuclidean_f16_v128relaxed", test_sqeuclidean<f16_t>, nk_sqeuclidean_f16_v128relaxed);
    run_if_matches("sqeuclidean_bf16_v128relaxed", test_sqeuclidean<bf16_t>, nk_sqeuclidean_bf16_v128relaxed);
    run_if_matches("angular_f32_v128relaxed", test_angular<f32_t>, nk_angular_f32_v128relaxed);
    run_if_matches("angular_f64_v128relaxed", test_angular<f64_t>, nk_angular_f64_v128relaxed);
    run_if_matches("angular_f16_v128relaxed", test_angular<f16_t>, nk_angular_f16_v128relaxed);
    run_if_matches("angular_bf16_v128relaxed", test_angular<bf16_t>, nk_angular_bf16_v128relaxed);
#endif // NK_TARGET_V128RELAXED

#if NK_TARGET_RVVHALF
    run_if_matches("sqeuclidean_f16_rvvhalf", test_sqeuclidean<f16_t>, nk_sqeuclidean_f16_rvvhalf);
    run_if_matches("angular_f16_rvvhalf", test_angular<f16_t>, nk_angular_f16_rvvhalf);
#endif // NK_TARGET_RVVHALF

#if NK_TARGET_RVVBF16
    run_if_matches("sqeuclidean_bf16_rvvbf16", test_sqeuclidean<bf16_t>, nk_sqeuclidean_bf16_rvvbf16);
    run_if_matches("angular_bf16_rvvbf16", test_angular<bf16_t>, nk_angular_bf16_rvvbf16);
#endif // NK_TARGET_RVVBF16

    // Serial always runs - baseline test
    run_if_matches("sqeuclidean_f32_serial", test_sqeuclidean<f32_t>, nk_sqeuclidean_f32_serial);
    run_if_matches("sqeuclidean_f64_serial", test_sqeuclidean<f64_t>, nk_sqeuclidean_f64_serial);
    run_if_matches("sqeuclidean_f16_serial", test_sqeuclidean<f16_t>, nk_sqeuclidean_f16_serial);
    run_if_matches("sqeuclidean_bf16_serial", test_sqeuclidean<bf16_t>, nk_sqeuclidean_bf16_serial);
    run_if_matches("angular_f32_serial", test_angular<f32_t>, nk_angular_f32_serial);
    run_if_matches("angular_f64_serial", test_angular<f64_t>, nk_angular_f64_serial);
    run_if_matches("angular_f16_serial", test_angular<f16_t>, nk_angular_f16_serial);
    run_if_matches("angular_bf16_serial", test_angular<bf16_t>, nk_angular_bf16_serial);
    run_if_matches("sqeuclidean_i4_serial", test_sqeuclidean<i4x2_t>, nk_sqeuclidean_i4_serial);
    run_if_matches("sqeuclidean_u4_serial", test_sqeuclidean<u4x2_t>, nk_sqeuclidean_u4_serial);
    run_if_matches("angular_i4_serial", test_angular<i4x2_t>, nk_angular_i4_serial);
    run_if_matches("angular_u4_serial", test_angular<u4x2_t>, nk_angular_u4_serial);

#endif // NK_DYNAMIC_DISPATCH
}

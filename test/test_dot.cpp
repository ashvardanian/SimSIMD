/**
 *  @brief Dot product precision tests.
 *  @file test/test_dot.cpp
 *  @author Ash Vardanian
 *  @date December 28, 2025
 */

#include "test.hpp"
#include "numkong/dot.hpp" // `nk::dot`

/**
 *  @brief Unified dot product test for all types: float, integer, and complex.
 *  Works with f32_t, f64_t, f16_t, bf16_t, e4m3_t, e5m2_t, i8_t, u8_t, f32c_t, f64c_t.
 */
template <typename scalar_type_>
error_stats_t test_dot(typename scalar_type_::dot_kernel_t kernel) {
    using scalar_t = scalar_type_;
    using raw_t = typename scalar_t::raw_t;
    using result_t = typename scalar_t::dot_result_t;
    using reference_t = std::conditional_t<scalar_t::is_complex(), f118c_t, f118_t>;

    error_stats_t stats;
    std::mt19937 generator(global_config.seed);
    auto a = make_vector<scalar_t>(dense_dimensions), b = make_vector<scalar_t>(dense_dimensions);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(generator, a);
        fill_random(generator, b);

        result_t result;
        kernel(a.raw_values_data(), b.raw_values_data(), dense_dimensions, &result.raw_);

        reference_t reference;
        nk::dot<scalar_t, reference_t, nk::no_simd_k>(a.values_data(), b.values_data(), dense_dimensions, &reference);

        stats.accumulate(result, reference);
    }
    return stats;
}

/**
 *  @brief Conjugate dot product test for complex types (vdot = conj(a) * b).
 */
template <typename scalar_type_>
error_stats_t test_vdot(typename scalar_type_::vdot_kernel_t kernel) {
    using scalar_t = scalar_type_;
    using raw_t = typename scalar_t::raw_t;
    using result_t = typename scalar_t::vdot_result_t;

    error_stats_t stats;
    std::mt19937 generator(global_config.seed);
    auto a = make_vector<scalar_t>(dense_dimensions), b = make_vector<scalar_t>(dense_dimensions);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(generator, a);
        fill_random(generator, b);

        result_t result;
        kernel(a.raw_values_data(), b.raw_values_data(), dense_dimensions, &result.raw_);

        f118c_t reference;
        nk::vdot<scalar_t, f118c_t, nk::no_simd_k>(a.values_data(), b.values_data(), dense_dimensions, &reference);

        stats.accumulate(result, reference);
    }
    return stats;
}

void test_dot() {
    std::puts("");
    std::printf("Dot Products:\n");

#if NK_DYNAMIC_DISPATCH
    // Dynamic dispatch - only test the dispatcher itself
    run_if_matches("dot_f32", test_dot<f32_t>, nk_dot_f32);
    run_if_matches("dot_f64", test_dot<f64_t>, nk_dot_f64);
    run_if_matches("dot_f16", test_dot<f16_t>, nk_dot_f16);
    run_if_matches("dot_bf16", test_dot<bf16_t>, nk_dot_bf16);
    run_if_matches("dot_e4m3", test_dot<e4m3_t>, nk_dot_e4m3);
    run_if_matches("dot_e5m2", test_dot<e5m2_t>, nk_dot_e5m2);
    run_if_matches("dot_e2m3", test_dot<e2m3_t>, nk_dot_e2m3);
    run_if_matches("dot_e3m2", test_dot<e3m2_t>, nk_dot_e3m2);
    run_if_matches("dot_i8", test_dot<i8_t>, nk_dot_i8);
    run_if_matches("dot_u8", test_dot<u8_t>, nk_dot_u8);
    run_if_matches("dot_i4", test_dot<i4x2_t>, nk_dot_i4);
    run_if_matches("dot_u4", test_dot<u4x2_t>, nk_dot_u4);
    run_if_matches("dot_f32c", test_dot<f32c_t>, nk_dot_f32c);
    run_if_matches("vdot_f32c", test_vdot<f32c_t>, nk_vdot_f32c);
    run_if_matches("dot_f64c", test_dot<f64c_t>, nk_dot_f64c);
    run_if_matches("vdot_f64c", test_vdot<f64c_t>, nk_vdot_f64c);
    run_if_matches("dot_f16c", test_dot<f16c_t>, nk_dot_f16c);
    run_if_matches("vdot_f16c", test_vdot<f16c_t>, nk_vdot_f16c);
    run_if_matches("dot_bf16c", test_dot<bf16c_t>, nk_dot_bf16c);
    run_if_matches("vdot_bf16c", test_vdot<bf16c_t>, nk_vdot_bf16c);
#else
    // Static compilation - test all available ISA variants

#if NK_TARGET_NEON
    run_if_matches("dot_f32_neon", test_dot<f32_t>, nk_dot_f32_neon);
    run_if_matches("dot_f64_neon", test_dot<f64_t>, nk_dot_f64_neon);
    run_if_matches("dot_e4m3_neon", test_dot<e4m3_t>, nk_dot_e4m3_neon);
    run_if_matches("dot_e5m2_neon", test_dot<e5m2_t>, nk_dot_e5m2_neon);
    run_if_matches("dot_f32c_neon", test_dot<f32c_t>, nk_dot_f32c_neon);
    run_if_matches("vdot_f32c_neon", test_vdot<f32c_t>, nk_vdot_f32c_neon);
    run_if_matches("dot_f64c_neon", test_dot<f64c_t>, nk_dot_f64c_neon);
    run_if_matches("vdot_f64c_neon", test_vdot<f64c_t>, nk_vdot_f64c_neon);
#endif // NK_TARGET_NEON

#if NK_TARGET_NEONHALF
    run_if_matches("dot_f16_neonhalf", test_dot<f16_t>, nk_dot_f16_neonhalf);
    run_if_matches("dot_f16c_neonhalf", test_dot<f16c_t>, nk_dot_f16c_neonhalf);
    run_if_matches("vdot_f16c_neonhalf", test_vdot<f16c_t>, nk_vdot_f16c_neonhalf);
#endif // NK_TARGET_NEONHALF

#if NK_TARGET_NEONBFDOT
    run_if_matches("dot_bf16_neonbfdot", test_dot<bf16_t>, nk_dot_bf16_neonbfdot);
    run_if_matches("dot_bf16c_neonbfdot", test_dot<bf16c_t>, nk_dot_bf16c_neonbfdot);
    run_if_matches("vdot_bf16c_neonbfdot", test_vdot<bf16c_t>, nk_vdot_bf16c_neonbfdot);
#endif // NK_TARGET_NEONBFDOT

#if NK_TARGET_NEONSDOT
    run_if_matches("dot_i8_neonsdot", test_dot<i8_t>, nk_dot_i8_neonsdot);
    run_if_matches("dot_u8_neonsdot", test_dot<u8_t>, nk_dot_u8_neonsdot);
    run_if_matches("dot_i4_neonsdot", test_dot<i4x2_t>, nk_dot_i4_neonsdot);
    run_if_matches("dot_u4_neonsdot", test_dot<u4x2_t>, nk_dot_u4_neonsdot);
    run_if_matches("dot_e2m3_neonsdot", test_dot<e2m3_t>, nk_dot_e2m3_neonsdot);
#endif // NK_TARGET_NEONSDOT

#if NK_TARGET_HASWELL
    run_if_matches("dot_f32_haswell", test_dot<f32_t>, nk_dot_f32_haswell);
    run_if_matches("dot_f64_haswell", test_dot<f64_t>, nk_dot_f64_haswell);
    run_if_matches("dot_f16_haswell", test_dot<f16_t>, nk_dot_f16_haswell);
    run_if_matches("dot_bf16_haswell", test_dot<bf16_t>, nk_dot_bf16_haswell);
    run_if_matches("dot_e4m3_haswell", test_dot<e4m3_t>, nk_dot_e4m3_haswell);
    run_if_matches("dot_e5m2_haswell", test_dot<e5m2_t>, nk_dot_e5m2_haswell);
    run_if_matches("dot_e2m3_haswell", test_dot<e2m3_t>, nk_dot_e2m3_haswell);
    run_if_matches("dot_e3m2_haswell", test_dot<e3m2_t>, nk_dot_e3m2_haswell);
    run_if_matches("dot_i8_haswell", test_dot<i8_t>, nk_dot_i8_haswell);
    run_if_matches("dot_u8_haswell", test_dot<u8_t>, nk_dot_u8_haswell);
    run_if_matches("dot_f32c_haswell", test_dot<f32c_t>, nk_dot_f32c_haswell);
    run_if_matches("vdot_f32c_haswell", test_vdot<f32c_t>, nk_vdot_f32c_haswell);
    run_if_matches("dot_f16c_haswell", test_dot<f16c_t>, nk_dot_f16c_haswell);
    run_if_matches("vdot_f16c_haswell", test_vdot<f16c_t>, nk_vdot_f16c_haswell);
    run_if_matches("dot_bf16c_haswell", test_dot<bf16c_t>, nk_dot_bf16c_haswell);
    run_if_matches("vdot_bf16c_haswell", test_vdot<bf16c_t>, nk_vdot_bf16c_haswell);
    run_if_matches("dot_i4_haswell", test_dot<i4x2_t>, nk_dot_i4_haswell);
    run_if_matches("dot_u4_haswell", test_dot<u4x2_t>, nk_dot_u4_haswell);
#endif // NK_TARGET_HASWELL

#if NK_TARGET_SKYLAKE
    run_if_matches("dot_f32_skylake", test_dot<f32_t>, nk_dot_f32_skylake);
    run_if_matches("dot_f64_skylake", test_dot<f64_t>, nk_dot_f64_skylake);
    run_if_matches("dot_f16_skylake", test_dot<f16_t>, nk_dot_f16_skylake);
    run_if_matches("dot_bf16_skylake", test_dot<bf16_t>, nk_dot_bf16_skylake);
    run_if_matches("dot_e4m3_skylake", test_dot<e4m3_t>, nk_dot_e4m3_skylake);
    run_if_matches("dot_e5m2_skylake", test_dot<e5m2_t>, nk_dot_e5m2_skylake);
    run_if_matches("dot_e2m3_skylake", test_dot<e2m3_t>, nk_dot_e2m3_skylake);
    run_if_matches("dot_e3m2_skylake", test_dot<e3m2_t>, nk_dot_e3m2_skylake);
    run_if_matches("dot_i8_skylake", test_dot<i8_t>, nk_dot_i8_skylake);
    run_if_matches("dot_u8_skylake", test_dot<u8_t>, nk_dot_u8_skylake);
    run_if_matches("dot_f32c_skylake", test_dot<f32c_t>, nk_dot_f32c_skylake);
    run_if_matches("vdot_f32c_skylake", test_vdot<f32c_t>, nk_vdot_f32c_skylake);
    run_if_matches("dot_f64c_skylake", test_dot<f64c_t>, nk_dot_f64c_skylake);
    run_if_matches("vdot_f64c_skylake", test_vdot<f64c_t>, nk_vdot_f64c_skylake);
#endif // NK_TARGET_SKYLAKE

#if NK_TARGET_ICELAKE
    run_if_matches("dot_i8_icelake", test_dot<i8_t>, nk_dot_i8_icelake);
    run_if_matches("dot_u8_icelake", test_dot<u8_t>, nk_dot_u8_icelake);
    run_if_matches("dot_i4_icelake", test_dot<i4x2_t>, nk_dot_i4_icelake);
    run_if_matches("dot_u4_icelake", test_dot<u4x2_t>, nk_dot_u4_icelake);
    run_if_matches("dot_e2m3_icelake", test_dot<e2m3_t>, nk_dot_e2m3_icelake);
#endif // NK_TARGET_ICELAKE

#if NK_TARGET_GENOA
    run_if_matches("dot_bf16_genoa", test_dot<bf16_t>, nk_dot_bf16_genoa);
    run_if_matches("dot_e4m3_genoa", test_dot<e4m3_t>, nk_dot_e4m3_genoa);
    run_if_matches("dot_e5m2_genoa", test_dot<e5m2_t>, nk_dot_e5m2_genoa);
    run_if_matches("dot_e2m3_genoa", test_dot<e2m3_t>, nk_dot_e2m3_genoa);
    run_if_matches("dot_e3m2_genoa", test_dot<e3m2_t>, nk_dot_e3m2_genoa);
#endif // NK_TARGET_GENOA

#if NK_TARGET_SAPPHIRE
    run_if_matches("dot_e3m2_sapphire", test_dot<e3m2_t>, nk_dot_e3m2_sapphire);
#endif // NK_TARGET_SAPPHIRE

#if NK_TARGET_RVV
    run_if_matches("dot_i8_rvv", test_dot<i8_t>, nk_dot_i8_rvv);
    run_if_matches("dot_u8_rvv", test_dot<u8_t>, nk_dot_u8_rvv);
    run_if_matches("dot_f32_rvv", test_dot<f32_t>, nk_dot_f32_rvv);
    run_if_matches("dot_f64_rvv", test_dot<f64_t>, nk_dot_f64_rvv);
    run_if_matches("dot_f16_rvv", test_dot<f16_t>, nk_dot_f16_rvv);
    run_if_matches("dot_bf16_rvv", test_dot<bf16_t>, nk_dot_bf16_rvv);
    run_if_matches("dot_e4m3_rvv", test_dot<e4m3_t>, nk_dot_e4m3_rvv);
    run_if_matches("dot_e5m2_rvv", test_dot<e5m2_t>, nk_dot_e5m2_rvv);
    run_if_matches("dot_e2m3_rvv", test_dot<e2m3_t>, nk_dot_e2m3_rvv);
    run_if_matches("dot_e3m2_rvv", test_dot<e3m2_t>, nk_dot_e3m2_rvv);
    run_if_matches("dot_i4_rvv", test_dot<i4x2_t>, nk_dot_i4_rvv);
    run_if_matches("dot_u4_rvv", test_dot<u4x2_t>, nk_dot_u4_rvv);
#endif // NK_TARGET_RVV

#if NK_TARGET_V128RELAXED
    run_if_matches("dot_f32_v128relaxed", test_dot<f32_t>, nk_dot_f32_v128relaxed);
    run_if_matches("dot_f64_v128relaxed", test_dot<f64_t>, nk_dot_f64_v128relaxed);
    run_if_matches("dot_f16_v128relaxed", test_dot<f16_t>, nk_dot_f16_v128relaxed);
    run_if_matches("dot_bf16_v128relaxed", test_dot<bf16_t>, nk_dot_bf16_v128relaxed);
    run_if_matches("dot_i8_v128relaxed", test_dot<i8_t>, nk_dot_i8_v128relaxed);
    run_if_matches("dot_u8_v128relaxed", test_dot<u8_t>, nk_dot_u8_v128relaxed);
    run_if_matches("dot_e2m3_v128relaxed", test_dot<e2m3_t>, nk_dot_e2m3_v128relaxed);
#endif // NK_TARGET_V128RELAXED

#if NK_TARGET_RVVHALF
    run_if_matches("dot_f16_rvvhalf", test_dot<f16_t>, nk_dot_f16_rvvhalf);
    run_if_matches("dot_e3m2_rvvhalf", test_dot<e3m2_t>, nk_dot_e3m2_rvvhalf);
    run_if_matches("dot_e4m3_rvvhalf", test_dot<e4m3_t>, nk_dot_e4m3_rvvhalf);
    run_if_matches("dot_e5m2_rvvhalf", test_dot<e5m2_t>, nk_dot_e5m2_rvvhalf);
#endif // NK_TARGET_RVVHALF

#if NK_TARGET_RVVBF16
    run_if_matches("dot_bf16_rvvbf16", test_dot<bf16_t>, nk_dot_bf16_rvvbf16);
    run_if_matches("dot_e3m2_rvvbf16", test_dot<e3m2_t>, nk_dot_e3m2_rvvbf16);
    run_if_matches("dot_e4m3_rvvbf16", test_dot<e4m3_t>, nk_dot_e4m3_rvvbf16);
    run_if_matches("dot_e5m2_rvvbf16", test_dot<e5m2_t>, nk_dot_e5m2_rvvbf16);
#endif // NK_TARGET_RVVBF16

    // Serial always runs - baseline test
    run_if_matches("dot_f32_serial", test_dot<f32_t>, nk_dot_f32_serial);
    run_if_matches("dot_f64_serial", test_dot<f64_t>, nk_dot_f64_serial);
    run_if_matches("dot_f16_serial", test_dot<f16_t>, nk_dot_f16_serial);
    run_if_matches("dot_bf16_serial", test_dot<bf16_t>, nk_dot_bf16_serial);
    run_if_matches("dot_e4m3_serial", test_dot<e4m3_t>, nk_dot_e4m3_serial);
    run_if_matches("dot_e5m2_serial", test_dot<e5m2_t>, nk_dot_e5m2_serial);
    run_if_matches("dot_e2m3_serial", test_dot<e2m3_t>, nk_dot_e2m3_serial);
    run_if_matches("dot_e3m2_serial", test_dot<e3m2_t>, nk_dot_e3m2_serial);
    run_if_matches("dot_i8_serial", test_dot<i8_t>, nk_dot_i8_serial);
    run_if_matches("dot_u8_serial", test_dot<u8_t>, nk_dot_u8_serial);
    run_if_matches("dot_i4_serial", test_dot<i4x2_t>, nk_dot_i4_serial);
    run_if_matches("dot_u4_serial", test_dot<u4x2_t>, nk_dot_u4_serial);
    run_if_matches("dot_f32c_serial", test_dot<f32c_t>, nk_dot_f32c_serial);
    run_if_matches("vdot_f32c_serial", test_vdot<f32c_t>, nk_vdot_f32c_serial);
    run_if_matches("dot_f64c_serial", test_dot<f64c_t>, nk_dot_f64c_serial);
    run_if_matches("vdot_f64c_serial", test_vdot<f64c_t>, nk_vdot_f64c_serial);
    run_if_matches("dot_f16c_serial", test_dot<f16c_t>, nk_dot_f16c_serial);
    run_if_matches("vdot_f16c_serial", test_vdot<f16c_t>, nk_vdot_f16c_serial);
    run_if_matches("dot_bf16c_serial", test_dot<bf16c_t>, nk_dot_bf16c_serial);
    run_if_matches("vdot_bf16c_serial", test_vdot<bf16c_t>, nk_vdot_bf16c_serial);

#endif // NK_DYNAMIC_DISPATCH
    // BLAS/MKL/Accelerate precision comparisons are in test_cross_blas.cpp
}

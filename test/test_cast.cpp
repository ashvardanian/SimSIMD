/**
 *  @brief Type cast tests.
 *  @file test/test_cast.cpp
 *  @author Ash Vardanian
 *  @date February 6, 2026
 */

#include "test.hpp"
#include "numkong/cast.h"

using cast_t = void (*)(void const *, nk_dtype_t, nk_size_t, void *, nk_dtype_t);

/**
 *  @brief Test cast kernel against serial kernel.
 *  SIMD kernels must match serial output exactly (raw byte comparison).
 */
template <typename from_type_, typename to_type_>
error_stats_t test_cast(cast_t kernel) {
    error_stats_t stats;
    std::mt19937 generator(global_config.seed);

    auto src = make_vector<from_type_>(dense_dimensions);
    auto dst_simd = make_vector<to_type_>(dense_dimensions);
    auto dst_serial = make_vector<to_type_>(dense_dimensions);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(generator, src);

        nk_cast_serial(src.raw_values_data(), from_type_::dtype(), dense_dimensions, dst_serial.raw_values_data(),
                       to_type_::dtype());
        kernel(src.raw_values_data(), from_type_::dtype(), dense_dimensions, dst_simd.raw_values_data(),
               to_type_::dtype());

        for (std::size_t i = 0; i < dense_dimensions; ++i) stats.accumulate(dst_simd[i].raw_, dst_serial[i].raw_);
    }
    return stats;
}

void test_casts() {
    std::puts("");
    std::printf("Type Casts:\n");

#if NK_DYNAMIC_DISPATCH
    run_if_matches("cast_f32_to_f16", test_cast<f32_t, f16_t>, nk_cast);
    run_if_matches("cast_f16_to_f32", test_cast<f16_t, f32_t>, nk_cast);
    run_if_matches("cast_f32_to_bf16", test_cast<f32_t, bf16_t>, nk_cast);
    run_if_matches("cast_bf16_to_f32", test_cast<bf16_t, f32_t>, nk_cast);
    run_if_matches("cast_f32_to_e4m3", test_cast<f32_t, e4m3_t>, nk_cast);
    run_if_matches("cast_e4m3_to_f32", test_cast<e4m3_t, f32_t>, nk_cast);
    run_if_matches("cast_f32_to_e5m2", test_cast<f32_t, e5m2_t>, nk_cast);
    run_if_matches("cast_e5m2_to_f32", test_cast<e5m2_t, f32_t>, nk_cast);
    run_if_matches("cast_f32_to_e2m3", test_cast<f32_t, e2m3_t>, nk_cast);
    run_if_matches("cast_e2m3_to_f32", test_cast<e2m3_t, f32_t>, nk_cast);
    run_if_matches("cast_f32_to_e3m2", test_cast<f32_t, e3m2_t>, nk_cast);
    run_if_matches("cast_e3m2_to_f32", test_cast<e3m2_t, f32_t>, nk_cast);
    run_if_matches("cast_f64_to_f32", test_cast<f64_t, f32_t>, nk_cast);
    run_if_matches("cast_f32_to_f64", test_cast<f32_t, f64_t>, nk_cast);
#endif

#if NK_TARGET_HASWELL
    run_if_matches("cast_f32_to_f16_haswell", test_cast<f32_t, f16_t>, nk_cast_haswell);
    run_if_matches("cast_f16_to_f32_haswell", test_cast<f16_t, f32_t>, nk_cast_haswell);
    run_if_matches("cast_f32_to_bf16_haswell", test_cast<f32_t, bf16_t>, nk_cast_haswell);
    run_if_matches("cast_bf16_to_f32_haswell", test_cast<bf16_t, f32_t>, nk_cast_haswell);
    run_if_matches("cast_f32_to_e4m3_haswell", test_cast<f32_t, e4m3_t>, nk_cast_haswell);
    run_if_matches("cast_e4m3_to_f32_haswell", test_cast<e4m3_t, f32_t>, nk_cast_haswell);
    run_if_matches("cast_f32_to_e5m2_haswell", test_cast<f32_t, e5m2_t>, nk_cast_haswell);
    run_if_matches("cast_e5m2_to_f32_haswell", test_cast<e5m2_t, f32_t>, nk_cast_haswell);
#endif

#if NK_TARGET_SKYLAKE
    run_if_matches("cast_f32_to_f16_skylake", test_cast<f32_t, f16_t>, nk_cast_skylake);
    run_if_matches("cast_f16_to_f32_skylake", test_cast<f16_t, f32_t>, nk_cast_skylake);
    run_if_matches("cast_f32_to_bf16_skylake", test_cast<f32_t, bf16_t>, nk_cast_skylake);
    run_if_matches("cast_bf16_to_f32_skylake", test_cast<bf16_t, f32_t>, nk_cast_skylake);
    run_if_matches("cast_f32_to_e4m3_skylake", test_cast<f32_t, e4m3_t>, nk_cast_skylake);
    run_if_matches("cast_e4m3_to_f32_skylake", test_cast<e4m3_t, f32_t>, nk_cast_skylake);
    run_if_matches("cast_f32_to_e5m2_skylake", test_cast<f32_t, e5m2_t>, nk_cast_skylake);
    run_if_matches("cast_e5m2_to_f32_skylake", test_cast<e5m2_t, f32_t>, nk_cast_skylake);
    run_if_matches("cast_f16_to_bf16_skylake", test_cast<f16_t, bf16_t>, nk_cast_skylake);
    run_if_matches("cast_bf16_to_f16_skylake", test_cast<bf16_t, f16_t>, nk_cast_skylake);
    run_if_matches("cast_e4m3_to_f16_skylake", test_cast<e4m3_t, f16_t>, nk_cast_skylake);
    run_if_matches("cast_f16_to_e4m3_skylake", test_cast<f16_t, e4m3_t>, nk_cast_skylake);
    run_if_matches("cast_e5m2_to_f16_skylake", test_cast<e5m2_t, f16_t>, nk_cast_skylake);
    run_if_matches("cast_f16_to_e5m2_skylake", test_cast<f16_t, e5m2_t>, nk_cast_skylake);
    run_if_matches("cast_e4m3_to_bf16_skylake", test_cast<e4m3_t, bf16_t>, nk_cast_skylake);
    run_if_matches("cast_bf16_to_e4m3_skylake", test_cast<bf16_t, e4m3_t>, nk_cast_skylake);
    run_if_matches("cast_e5m2_to_bf16_skylake", test_cast<e5m2_t, bf16_t>, nk_cast_skylake);
    run_if_matches("cast_bf16_to_e5m2_skylake", test_cast<bf16_t, e5m2_t>, nk_cast_skylake);
#endif

#if NK_TARGET_ICELAKE

    run_if_matches("cast_e4m3_to_bf16_icelake", test_cast<e4m3_t, bf16_t>, nk_cast_icelake);
    run_if_matches("cast_bf16_to_e4m3_icelake", test_cast<bf16_t, e4m3_t>, nk_cast_icelake);
    run_if_matches("cast_e5m2_to_bf16_icelake", test_cast<e5m2_t, bf16_t>, nk_cast_icelake);
    run_if_matches("cast_bf16_to_e5m2_icelake", test_cast<bf16_t, e5m2_t>, nk_cast_icelake);
    run_if_matches("cast_e4m3_to_f16_icelake", test_cast<e4m3_t, f16_t>, nk_cast_icelake);
    run_if_matches("cast_e5m2_to_f16_icelake", test_cast<e5m2_t, f16_t>, nk_cast_icelake);
#endif

#if NK_TARGET_SAPPHIRE
    run_if_matches("cast_e4m3_to_f16_sapphire", test_cast<e4m3_t, f16_t>, nk_cast_sapphire);
    run_if_matches("cast_f16_to_e4m3_sapphire", test_cast<f16_t, e4m3_t>, nk_cast_sapphire);
    run_if_matches("cast_e5m2_to_f16_sapphire", test_cast<e5m2_t, f16_t>, nk_cast_sapphire);
    run_if_matches("cast_f16_to_e5m2_sapphire", test_cast<f16_t, e5m2_t>, nk_cast_sapphire);
#endif

#if NK_TARGET_NEON
    run_if_matches("cast_e4m3_to_f32_neon", test_cast<e4m3_t, f32_t>, nk_cast_neon);
    run_if_matches("cast_f32_to_e4m3_neon", test_cast<f32_t, e4m3_t>, nk_cast_neon);
    run_if_matches("cast_e5m2_to_f32_neon", test_cast<e5m2_t, f32_t>, nk_cast_neon);
    run_if_matches("cast_f32_to_e5m2_neon", test_cast<f32_t, e5m2_t>, nk_cast_neon);
#endif

#if NK_TARGET_RVV
    run_if_matches("cast_bf16_to_f32_rvv", test_cast<bf16_t, f32_t>, nk_cast_rvv);
    run_if_matches("cast_f32_to_bf16_rvv", test_cast<f32_t, bf16_t>, nk_cast_rvv);
    run_if_matches("cast_e4m3_to_f32_rvv", test_cast<e4m3_t, f32_t>, nk_cast_rvv);
    run_if_matches("cast_e5m2_to_f32_rvv", test_cast<e5m2_t, f32_t>, nk_cast_rvv);
#endif
}

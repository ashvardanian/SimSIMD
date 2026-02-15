/**
 *  @brief Batch operation tests - RVV ISA family (RISC-V Vector).
 *  @file test/test_cross_rvv.cpp
 *  @author Ash Vardanian
 *  @date February 15, 2026
 */
#include "test.hpp"
#include "test_cross.hpp"

void test_cross_rvv() {
#if NK_TARGET_RVV
    run_if_matches("dots_f32_rvv", test_dots<f32_t>, nk_dots_packed_size_f32_rvv, nk_dots_pack_f32_rvv,
                   nk_dots_packed_f32_rvv);
    run_if_matches("dots_f64_rvv", test_dots<f64_t>, nk_dots_packed_size_f64_rvv, nk_dots_pack_f64_rvv,
                   nk_dots_packed_f64_rvv);
    run_if_matches("dots_bf16_rvv", test_dots<bf16_t>, nk_dots_packed_size_bf16_rvv, nk_dots_pack_bf16_rvv,
                   nk_dots_packed_bf16_rvv);
    run_if_matches("dots_f16_rvv", test_dots<f16_t>, nk_dots_packed_size_f16_rvv, nk_dots_pack_f16_rvv,
                   nk_dots_packed_f16_rvv);
    run_if_matches("dots_i8_rvv", test_dots<i8_t>, nk_dots_packed_size_i8_rvv, nk_dots_pack_i8_rvv,
                   nk_dots_packed_i8_rvv);
    run_if_matches("dots_u8_rvv", test_dots<u8_t>, nk_dots_packed_size_u8_rvv, nk_dots_pack_u8_rvv,
                   nk_dots_packed_u8_rvv);
    run_if_matches("dots_e2m3_rvv", test_dots<e2m3_t>, nk_dots_packed_size_e2m3_rvv, nk_dots_pack_e2m3_rvv,
                   nk_dots_packed_e2m3_rvv);
    run_if_matches("dots_e3m2_rvv", test_dots<e3m2_t>, nk_dots_packed_size_e3m2_rvv, nk_dots_pack_e3m2_rvv,
                   nk_dots_packed_e3m2_rvv);
    run_if_matches("dots_e4m3_rvv", test_dots<e4m3_t>, nk_dots_packed_size_e4m3_rvv, nk_dots_pack_e4m3_rvv,
                   nk_dots_packed_e4m3_rvv);
    run_if_matches("dots_e5m2_rvv", test_dots<e5m2_t>, nk_dots_packed_size_e5m2_rvv, nk_dots_pack_e5m2_rvv,
                   nk_dots_packed_e5m2_rvv);

    run_if_matches("dots_symmetric_f32_rvv", test_dots_symmetric<f32_t>, nk_dots_symmetric_f32_rvv);
    run_if_matches("dots_symmetric_f64_rvv", test_dots_symmetric<f64_t>, nk_dots_symmetric_f64_rvv);
    run_if_matches("dots_symmetric_bf16_rvv", test_dots_symmetric<bf16_t>, nk_dots_symmetric_bf16_rvv);
    run_if_matches("dots_symmetric_f16_rvv", test_dots_symmetric<f16_t>, nk_dots_symmetric_f16_rvv);
    run_if_matches("dots_symmetric_i8_rvv", test_dots_symmetric<i8_t>, nk_dots_symmetric_i8_rvv);
    run_if_matches("dots_symmetric_u8_rvv", test_dots_symmetric<u8_t>, nk_dots_symmetric_u8_rvv);
    run_if_matches("dots_symmetric_e2m3_rvv", test_dots_symmetric<e2m3_t>, nk_dots_symmetric_e2m3_rvv);
    run_if_matches("dots_symmetric_e3m2_rvv", test_dots_symmetric<e3m2_t>, nk_dots_symmetric_e3m2_rvv);
    run_if_matches("dots_symmetric_e4m3_rvv", test_dots_symmetric<e4m3_t>, nk_dots_symmetric_e4m3_rvv);
    run_if_matches("dots_symmetric_e5m2_rvv", test_dots_symmetric<e5m2_t>, nk_dots_symmetric_e5m2_rvv);
#endif
}

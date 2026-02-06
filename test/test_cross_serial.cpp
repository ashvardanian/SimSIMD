/**
 *  @brief Batch operation tests - Serial fallback.
 *  @file test/test_cross_serial.cpp
 *  @author Ash Vardanian
 *  @date January 14, 2025
 */
#include "test.hpp"
#include "test_cross.hpp"

void test_cross_serial() {
    // Dynamic-dispatch tests (auto-selected best ISA)
#if NK_DYNAMIC_DISPATCH
    run_if_matches("dots_f32", test_dots<f32_t>, nk_dots_packed_size_f32, nk_dots_pack_f32, nk_dots_packed_f32);
    run_if_matches("dots_f64", test_dots<f64_t>, nk_dots_packed_size_f64, nk_dots_pack_f64, nk_dots_packed_f64);
    run_if_matches("dots_bf16", test_dots<bf16_t>, nk_dots_packed_size_bf16, nk_dots_pack_bf16, nk_dots_packed_bf16);
    run_if_matches("dots_f16", test_dots<f16_t>, nk_dots_packed_size_f16, nk_dots_pack_f16, nk_dots_packed_f16);
    run_if_matches("dots_i8", test_dots<i8_t>, nk_dots_packed_size_i8, nk_dots_pack_i8, nk_dots_packed_i8);
    run_if_matches("dots_u4", test_dots<u4x2_t>, nk_dots_packed_size_u4, nk_dots_pack_u4, nk_dots_packed_u4);
    run_if_matches("dots_i4", test_dots<i4x2_t>, nk_dots_packed_size_i4, nk_dots_pack_i4, nk_dots_packed_i4);

    run_if_matches("dots_symmetric_f32", test_dots_symmetric<f32_t>, nk_dots_symmetric_f32);
    run_if_matches("dots_symmetric_f64", test_dots_symmetric<f64_t>, nk_dots_symmetric_f64);
    run_if_matches("dots_symmetric_bf16", test_dots_symmetric<bf16_t>, nk_dots_symmetric_bf16);
    run_if_matches("dots_symmetric_f16", test_dots_symmetric<f16_t>, nk_dots_symmetric_f16);
    run_if_matches("dots_symmetric_i8", test_dots_symmetric<i8_t>, nk_dots_symmetric_i8);
    run_if_matches("dots_symmetric_u8", test_dots_symmetric<u8_t>, nk_dots_symmetric_u8);
    run_if_matches("dots_symmetric_i4", test_dots_symmetric<i4x2_t>, nk_dots_symmetric_i4);
    run_if_matches("dots_symmetric_u4", test_dots_symmetric<u4x2_t>, nk_dots_symmetric_u4);
    run_if_matches("dots_symmetric_e4m3", test_dots_symmetric<e4m3_t>, nk_dots_symmetric_e4m3);
    run_if_matches("dots_symmetric_e5m2", test_dots_symmetric<e5m2_t>, nk_dots_symmetric_e5m2);

    run_if_matches("hammings_u1", test_hammings<u1x8_t>, nk_hammings_packed_size_u1, nk_hammings_pack_u1,
                   nk_hammings_packed_u1);
    run_if_matches("hammings_symmetric_u1", test_hammings_symmetric<u1x8_t>, nk_hammings_symmetric_u1);
#endif

    // Serial always runs - baseline tests
    run_if_matches("dots_f32_serial", test_dots<f32_t>, nk_dots_packed_size_f32_serial, nk_dots_pack_f32_serial,
                   nk_dots_packed_f32_serial);
    run_if_matches("dots_f64_serial", test_dots<f64_t>, nk_dots_packed_size_f64_serial, nk_dots_pack_f64_serial,
                   nk_dots_packed_f64_serial);
    run_if_matches("dots_bf16_serial", test_dots<bf16_t>, nk_dots_packed_size_bf16_serial, nk_dots_pack_bf16_serial,
                   nk_dots_packed_bf16_serial);
    run_if_matches("dots_f16_serial", test_dots<f16_t>, nk_dots_packed_size_f16_serial, nk_dots_pack_f16_serial,
                   nk_dots_packed_f16_serial);
    run_if_matches("dots_i8_serial", test_dots<i8_t>, nk_dots_packed_size_i8_serial, nk_dots_pack_i8_serial,
                   nk_dots_packed_i8_serial);
    run_if_matches("dots_u8_serial", test_dots<u8_t>, nk_dots_packed_size_u8_serial, nk_dots_pack_u8_serial,
                   nk_dots_packed_u8_serial);
    run_if_matches("dots_e4m3_serial", test_dots<e4m3_t>, nk_dots_packed_size_e4m3_serial, nk_dots_pack_e4m3_serial,
                   nk_dots_packed_e4m3_serial);
    run_if_matches("dots_e5m2_serial", test_dots<e5m2_t>, nk_dots_packed_size_e5m2_serial, nk_dots_pack_e5m2_serial,
                   nk_dots_packed_e5m2_serial);
    run_if_matches("dots_e2m3_serial", test_dots<e2m3_t>, nk_dots_packed_size_e2m3_serial, nk_dots_pack_e2m3_serial,
                   nk_dots_packed_e2m3_serial);
    run_if_matches("dots_e3m2_serial", test_dots<e3m2_t>, nk_dots_packed_size_e3m2_serial, nk_dots_pack_e3m2_serial,
                   nk_dots_packed_e3m2_serial);
    run_if_matches("dots_u4_serial", test_dots<u4x2_t>, nk_dots_packed_size_u4_serial, nk_dots_pack_u4_serial,
                   nk_dots_packed_u4_serial);
    run_if_matches("dots_i4_serial", test_dots<i4x2_t>, nk_dots_packed_size_i4_serial, nk_dots_pack_i4_serial,
                   nk_dots_packed_i4_serial);

    // Serial symmetric tests
    run_if_matches("dots_symmetric_f32_serial", test_dots_symmetric<f32_t>, nk_dots_symmetric_f32_serial);
    run_if_matches("dots_symmetric_f64_serial", test_dots_symmetric<f64_t>, nk_dots_symmetric_f64_serial);
    run_if_matches("dots_symmetric_bf16_serial", test_dots_symmetric<bf16_t>, nk_dots_symmetric_bf16_serial);
    run_if_matches("dots_symmetric_f16_serial", test_dots_symmetric<f16_t>, nk_dots_symmetric_f16_serial);
    run_if_matches("dots_symmetric_i8_serial", test_dots_symmetric<i8_t>, nk_dots_symmetric_i8_serial);
    run_if_matches("dots_symmetric_u8_serial", test_dots_symmetric<u8_t>, nk_dots_symmetric_u8_serial);
    run_if_matches("dots_symmetric_e4m3_serial", test_dots_symmetric<e4m3_t>, nk_dots_symmetric_e4m3_serial);
    run_if_matches("dots_symmetric_e5m2_serial", test_dots_symmetric<e5m2_t>, nk_dots_symmetric_e5m2_serial);
    run_if_matches("dots_symmetric_e2m3_serial", test_dots_symmetric<e2m3_t>, nk_dots_symmetric_e2m3_serial);
    run_if_matches("dots_symmetric_e3m2_serial", test_dots_symmetric<e3m2_t>, nk_dots_symmetric_e3m2_serial);

    // Serial hammings tests
    run_if_matches("hammings_u1_serial", test_hammings<u1x8_t>, nk_hammings_packed_size_u1_serial,
                   nk_hammings_pack_u1_serial, nk_hammings_packed_u1_serial);
    run_if_matches("hammings_symmetric_u1_serial", test_hammings_symmetric<u1x8_t>, nk_hammings_symmetric_u1_serial);
}

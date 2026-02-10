/**
 *  @brief Batch operation tests - SME ISA.
 *  @file test/test_cross_sme.cpp
 *  @author Ash Vardanian
 *  @date January 14, 2025
 */
#include "test.hpp"
#include "test_cross.hpp"

void test_cross_sme() {
#if NK_TARGET_SME
    run_if_matches("dots_packed_f16_sme", test_dots<f16_t>, nk_dots_packed_size_f16_sme, nk_dots_pack_f16_sme,
                   nk_dots_packed_f16_sme);
    run_if_matches("dots_packed_bf16_sme", test_dots<bf16_t>, nk_dots_packed_size_bf16_sme, nk_dots_pack_bf16_sme,
                   nk_dots_packed_bf16_sme);
    run_if_matches("dots_packed_i8_sme", test_dots<i8_t>, nk_dots_packed_size_i8_sme, nk_dots_pack_i8_sme,
                   nk_dots_packed_i8_sme);
    run_if_matches("dots_packed_u8_sme", test_dots<u8_t>, nk_dots_packed_size_u8_sme, nk_dots_pack_u8_sme,
                   nk_dots_packed_u8_sme);
    run_if_matches("dots_packed_e4m3_sme", test_dots<e4m3_t>, nk_dots_packed_size_e4m3_sme, nk_dots_pack_e4m3_sme,
                   nk_dots_packed_e4m3_sme);
    run_if_matches("dots_packed_e5m2_sme", test_dots<e5m2_t>, nk_dots_packed_size_e5m2_sme, nk_dots_pack_e5m2_sme,
                   nk_dots_packed_e5m2_sme);
    run_if_matches("dots_packed_i4_sme", test_dots<i4x2_t>, nk_dots_packed_size_i4_sme, nk_dots_pack_i4_sme,
                   nk_dots_packed_i4_sme);
    run_if_matches("dots_packed_u4_sme", test_dots<u4x2_t>, nk_dots_packed_size_u4_sme, nk_dots_pack_u4_sme,
                   nk_dots_packed_u4_sme);

    run_if_matches("dots_symmetric_bf16_sme", test_dots_symmetric<bf16_t>, nk_dots_symmetric_bf16_sme);
    run_if_matches("dots_symmetric_f16_sme", test_dots_symmetric<f16_t>, nk_dots_symmetric_f16_sme);
    run_if_matches("dots_symmetric_i8_sme", test_dots_symmetric<i8_t>, nk_dots_symmetric_i8_sme);
    run_if_matches("dots_symmetric_u8_sme", test_dots_symmetric<u8_t>, nk_dots_symmetric_u8_sme);
    run_if_matches("dots_symmetric_i4_sme", test_dots_symmetric<i4x2_t>, nk_dots_symmetric_i4_sme);
    run_if_matches("dots_symmetric_u4_sme", test_dots_symmetric<u4x2_t>, nk_dots_symmetric_u4_sme);
    run_if_matches("dots_symmetric_e4m3_sme", test_dots_symmetric<e4m3_t>, nk_dots_symmetric_e4m3_sme);
    run_if_matches("dots_symmetric_e5m2_sme", test_dots_symmetric<e5m2_t>, nk_dots_symmetric_e5m2_sme);
#endif

#if NK_TARGET_SMEBI32
    run_if_matches("hammings_packed_u1_smebi32", test_hammings<u1x8_t>, nk_hammings_packed_size_u1_smebi32,
                   nk_hammings_pack_u1_smebi32, nk_hammings_packed_u1_smebi32);
    run_if_matches("hammings_symmetric_u1_smebi32", test_hammings_symmetric<u1x8_t>, nk_hammings_symmetric_u1_smebi32);
#endif

#if NK_TARGET_SMEF64
    run_if_matches("dots_packed_f32_smef64", test_dots<f32_t>, nk_dots_packed_size_f32_smef64, nk_dots_pack_f32_smef64,
                   nk_dots_packed_f32_smef64);
    run_if_matches("dots_packed_f64_smef64", test_dots<f64_t>, nk_dots_packed_size_f64_smef64, nk_dots_pack_f64_smef64,
                   nk_dots_packed_f64_smef64);
    run_if_matches("dots_symmetric_f32_smef64", test_dots_symmetric<f32_t>, nk_dots_symmetric_f32_smef64);
    run_if_matches("dots_symmetric_f64_smef64", test_dots_symmetric<f64_t>, nk_dots_symmetric_f64_smef64);
#endif
}

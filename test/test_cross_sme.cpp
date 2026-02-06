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
    // SME precision validation tests (Arm Scalable Matrix Extension)
    run_if_matches("dots_sme_f16", test_dots<f16_t>, nk_dots_packed_size_f16_sme, nk_dots_pack_f16_sme,
                   nk_dots_packed_f16_sme);
    run_if_matches("dots_sme_bf16", test_dots<bf16_t>, nk_dots_packed_size_bf16_sme, nk_dots_pack_bf16_sme,
                   nk_dots_packed_bf16_sme);
    run_if_matches("dots_sme_i8", test_dots<i8_t>, nk_dots_packed_size_i8_sme, nk_dots_pack_i8_sme,
                   nk_dots_packed_i8_sme);
    run_if_matches("dots_sme_u8", test_dots<u8_t>, nk_dots_packed_size_u8_sme, nk_dots_pack_u8_sme,
                   nk_dots_packed_u8_sme);
    run_if_matches("dots_sme_e4m3", test_dots<e4m3_t>, nk_dots_packed_size_e4m3_sme, nk_dots_pack_e4m3_sme,
                   nk_dots_packed_e4m3_sme);
    run_if_matches("dots_sme_e5m2", test_dots<e5m2_t>, nk_dots_packed_size_e5m2_sme, nk_dots_pack_e5m2_sme,
                   nk_dots_packed_e5m2_sme);

    // Symmetric tests
    run_if_matches("dots_symmetric_sme_bf16", test_dots_symmetric<bf16_t>, nk_dots_symmetric_bf16_sme);
    run_if_matches("dots_symmetric_sme_f16", test_dots_symmetric<f16_t>, nk_dots_symmetric_f16_sme);
    run_if_matches("dots_symmetric_sme_i8", test_dots_symmetric<i8_t>, nk_dots_symmetric_i8_sme);
    run_if_matches("dots_symmetric_sme_u8", test_dots_symmetric<u8_t>, nk_dots_symmetric_u8_sme);
    run_if_matches("dots_symmetric_sme_i4", test_dots_symmetric<i4x2_t>, nk_dots_symmetric_i4_sme);
    run_if_matches("dots_symmetric_sme_u4", test_dots_symmetric<u4x2_t>, nk_dots_symmetric_u4_sme);
    run_if_matches("dots_symmetric_sme_e4m3", test_dots_symmetric<e4m3_t>, nk_dots_symmetric_e4m3_sme);
    run_if_matches("dots_symmetric_sme_e5m2", test_dots_symmetric<e5m2_t>, nk_dots_symmetric_e5m2_sme);
#endif

#if NK_TARGET_SMEF64
    run_if_matches("dots_smef64_f32", test_dots<f32_t>, nk_dots_packed_size_f32_smef64, nk_dots_pack_f32_smef64,
                   nk_dots_packed_f32_smef64);
    run_if_matches("dots_smef64_f64", test_dots<f64_t>, nk_dots_packed_size_f64_smef64, nk_dots_pack_f64_smef64,
                   nk_dots_packed_f64_smef64);
#endif
}

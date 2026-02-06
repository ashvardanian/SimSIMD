/**
 *  @brief Batch operation tests - Arm ISA family.
 *  @file test/test_cross_arm.cpp
 *  @author Ash Vardanian
 *  @date February 6, 2026
 *
 *  Covers NEON, NEONHALF, NEONFHM, NEONBFDOT, NEONSDOT.
 */
#include "test.hpp"
#include "test_cross.hpp"

void test_cross_arm() {
#if NK_TARGET_NEON
    run_if_matches("dots_f32_neon", test_dots<f32_t>, nk_dots_packed_size_f32_neon, nk_dots_pack_f32_neon,
                   nk_dots_packed_f32_neon);
    run_if_matches("dots_f64_neon", test_dots<f64_t>, nk_dots_packed_size_f64_neon, nk_dots_pack_f64_neon,
                   nk_dots_packed_f64_neon);

    // Symmetric tests
    run_if_matches("dots_symmetric_f32_neon", test_dots_symmetric<f32_t>, nk_dots_symmetric_f32_neon);
    run_if_matches("dots_symmetric_f64_neon", test_dots_symmetric<f64_t>, nk_dots_symmetric_f64_neon);

    // Hammings tests
    run_if_matches("hammings_u1_neon", test_hammings<u1x8_t>, nk_hammings_packed_size_u1_neon, nk_hammings_pack_u1_neon,
                   nk_hammings_packed_u1_neon);
    run_if_matches("hammings_symmetric_u1_neon", test_hammings_symmetric<u1x8_t>, nk_hammings_symmetric_u1_neon);
#endif

#if NK_TARGET_NEONHALF
    run_if_matches("dots_f16_neonhalf", test_dots<f16_t>, nk_dots_packed_size_f16_neonhalf, nk_dots_pack_f16_neonhalf,
                   nk_dots_packed_f16_neonhalf);
    run_if_matches("dots_symmetric_f16_neonhalf", test_dots_symmetric<f16_t>, nk_dots_symmetric_f16_neonhalf);
#endif

#if NK_TARGET_NEONFHM
    run_if_matches("dots_f16_neonfhm", test_dots<f16_t>, nk_dots_packed_size_f16_neonfhm, nk_dots_pack_f16_neonfhm,
                   nk_dots_packed_f16_neonfhm);
    run_if_matches("dots_symmetric_f16_neonfhm", test_dots_symmetric<f16_t>, nk_dots_symmetric_f16_neonfhm);
    run_if_matches("dots_symmetric_e2m3_neonfhm", test_dots_symmetric<e2m3_t>, nk_dots_symmetric_e2m3_neonfhm);
    run_if_matches("dots_symmetric_e3m2_neonfhm", test_dots_symmetric<e3m2_t>, nk_dots_symmetric_e3m2_neonfhm);
#endif

#if NK_TARGET_NEONBFDOT
    run_if_matches("dots_bf16_neonbfdot", test_dots<bf16_t>, nk_dots_packed_size_bf16_neonbfdot,
                   nk_dots_pack_bf16_neonbfdot, nk_dots_packed_bf16_neonbfdot);
#endif

#if NK_TARGET_NEONSDOT
    run_if_matches("dots_i8_neonsdot", test_dots<i8_t>, nk_dots_packed_size_i8_neonsdot, nk_dots_pack_i8_neonsdot,
                   nk_dots_packed_i8_neonsdot);
    run_if_matches("dots_u8_neonsdot", test_dots<u8_t>, nk_dots_packed_size_u8_neonsdot, nk_dots_pack_u8_neonsdot,
                   nk_dots_packed_u8_neonsdot);
    run_if_matches("dots_i4_neonsdot", test_dots<i4x2_t>, nk_dots_packed_size_i4_neonsdot, nk_dots_pack_i4_neonsdot,
                   nk_dots_packed_i4_neonsdot);
    run_if_matches("dots_u4_neonsdot", test_dots<u4x2_t>, nk_dots_packed_size_u4_neonsdot, nk_dots_pack_u4_neonsdot,
                   nk_dots_packed_u4_neonsdot);

    // Symmetric tests
    run_if_matches("dots_symmetric_i8_neonsdot", test_dots_symmetric<i8_t>, nk_dots_symmetric_i8_neonsdot);
    run_if_matches("dots_symmetric_u8_neonsdot", test_dots_symmetric<u8_t>, nk_dots_symmetric_u8_neonsdot);
    run_if_matches("dots_symmetric_i4_neonsdot", test_dots_symmetric<i4x2_t>, nk_dots_symmetric_i4_neonsdot);
    run_if_matches("dots_symmetric_u4_neonsdot", test_dots_symmetric<u4x2_t>, nk_dots_symmetric_u4_neonsdot);
#endif
}

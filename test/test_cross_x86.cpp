/**
 *  @brief Batch operation tests - x86 ISA family.
 *  @file test/test_cross_x86.cpp
 *  @author Ash Vardanian
 *  @date February 6, 2026
 *
 *  Covers Haswell, Skylake, Ice Lake, Genoa, Sapphire.
 */
#include "test.hpp"
#include "test_cross.hpp"

void test_cross_x86() {
#if NK_TARGET_HASWELL
    run_if_matches("dots_f32_haswell", test_dots<f32_t>, nk_dots_packed_size_f32_haswell, nk_dots_pack_f32_haswell,
                   nk_dots_packed_f32_haswell);
    run_if_matches("dots_f64_haswell", test_dots<f64_t>, nk_dots_packed_size_f64_haswell, nk_dots_pack_f64_haswell,
                   nk_dots_packed_f64_haswell);
    run_if_matches("dots_e4m3_haswell", test_dots<e4m3_t>, nk_dots_packed_size_e4m3_haswell, nk_dots_pack_e4m3_haswell,
                   nk_dots_packed_e4m3_haswell);
    run_if_matches("dots_e5m2_haswell", test_dots<e5m2_t>, nk_dots_packed_size_e5m2_haswell, nk_dots_pack_e5m2_haswell,
                   nk_dots_packed_e5m2_haswell);
    run_if_matches("dots_e2m3_haswell", test_dots<e2m3_t>, nk_dots_packed_size_e2m3_haswell, nk_dots_pack_e2m3_haswell,
                   nk_dots_packed_e2m3_haswell);
    run_if_matches("dots_e3m2_haswell", test_dots<e3m2_t>, nk_dots_packed_size_e3m2_haswell, nk_dots_pack_e3m2_haswell,
                   nk_dots_packed_e3m2_haswell);

    run_if_matches("dots_symmetric_f32_haswell", test_dots_symmetric<f32_t>, nk_dots_symmetric_f32_haswell);
    run_if_matches("dots_symmetric_f64_haswell", test_dots_symmetric<f64_t>, nk_dots_symmetric_f64_haswell);
    run_if_matches("dots_symmetric_bf16_haswell", test_dots_symmetric<bf16_t>, nk_dots_symmetric_bf16_haswell);
    run_if_matches("dots_symmetric_f16_haswell", test_dots_symmetric<f16_t>, nk_dots_symmetric_f16_haswell);
    run_if_matches("dots_symmetric_i8_haswell", test_dots_symmetric<i8_t>, nk_dots_symmetric_i8_haswell);
    run_if_matches("dots_symmetric_u8_haswell", test_dots_symmetric<u8_t>, nk_dots_symmetric_u8_haswell);
    run_if_matches("dots_symmetric_e4m3_haswell", test_dots_symmetric<e4m3_t>, nk_dots_symmetric_e4m3_haswell);
    run_if_matches("dots_symmetric_e5m2_haswell", test_dots_symmetric<e5m2_t>, nk_dots_symmetric_e5m2_haswell);
    run_if_matches("dots_symmetric_e2m3_haswell", test_dots_symmetric<e2m3_t>, nk_dots_symmetric_e2m3_haswell);
    run_if_matches("dots_symmetric_e3m2_haswell", test_dots_symmetric<e3m2_t>, nk_dots_symmetric_e3m2_haswell);

    run_if_matches("hammings_u1_haswell", test_hammings<u1x8_t>, nk_hammings_packed_size_u1_haswell,
                   nk_hammings_pack_u1_haswell, nk_hammings_packed_u1_haswell);
    run_if_matches("hammings_symmetric_u1_haswell", test_hammings_symmetric<u1x8_t>, nk_hammings_symmetric_u1_haswell);
#endif

#if NK_TARGET_SKYLAKE
    run_if_matches("dots_f32_skylake", test_dots<f32_t>, nk_dots_packed_size_f32_skylake, nk_dots_pack_f32_skylake,
                   nk_dots_packed_f32_skylake);
    run_if_matches("dots_f64_skylake", test_dots<f64_t>, nk_dots_packed_size_f64_skylake, nk_dots_pack_f64_skylake,
                   nk_dots_packed_f64_skylake);
    run_if_matches("dots_e4m3_skylake", test_dots<e4m3_t>, nk_dots_packed_size_e4m3_skylake, nk_dots_pack_e4m3_skylake,
                   nk_dots_packed_e4m3_skylake);
    run_if_matches("dots_e5m2_skylake", test_dots<e5m2_t>, nk_dots_packed_size_e5m2_skylake, nk_dots_pack_e5m2_skylake,
                   nk_dots_packed_e5m2_skylake);
    run_if_matches("dots_e2m3_skylake", test_dots<e2m3_t>, nk_dots_packed_size_e2m3_skylake, nk_dots_pack_e2m3_skylake,
                   nk_dots_packed_e2m3_skylake);
    run_if_matches("dots_e3m2_skylake", test_dots<e3m2_t>, nk_dots_packed_size_e3m2_skylake, nk_dots_pack_e3m2_skylake,
                   nk_dots_packed_e3m2_skylake);

    run_if_matches("dots_symmetric_f32_skylake", test_dots_symmetric<f32_t>, nk_dots_symmetric_f32_skylake);
    run_if_matches("dots_symmetric_f64_skylake", test_dots_symmetric<f64_t>, nk_dots_symmetric_f64_skylake);
    run_if_matches("dots_symmetric_bf16_skylake", test_dots_symmetric<bf16_t>, nk_dots_symmetric_bf16_skylake);
    run_if_matches("dots_symmetric_f16_skylake", test_dots_symmetric<f16_t>, nk_dots_symmetric_f16_skylake);
    run_if_matches("dots_symmetric_e4m3_skylake", test_dots_symmetric<e4m3_t>, nk_dots_symmetric_e4m3_skylake);
    run_if_matches("dots_symmetric_e5m2_skylake", test_dots_symmetric<e5m2_t>, nk_dots_symmetric_e5m2_skylake);
    run_if_matches("dots_symmetric_e2m3_skylake", test_dots_symmetric<e2m3_t>, nk_dots_symmetric_e2m3_skylake);
    run_if_matches("dots_symmetric_e3m2_skylake", test_dots_symmetric<e3m2_t>, nk_dots_symmetric_e3m2_skylake);

#endif

#if NK_TARGET_ICELAKE
    run_if_matches("dots_i4_icelake", test_dots<i4x2_t>, nk_dots_packed_size_i4_icelake, nk_dots_pack_i4_icelake,
                   nk_dots_packed_i4_icelake);
    run_if_matches("dots_u4_icelake", test_dots<u4x2_t>, nk_dots_packed_size_u4_icelake, nk_dots_pack_u4_icelake,
                   nk_dots_packed_u4_icelake);
    run_if_matches("dots_i8_icelake", test_dots<i8_t>, nk_dots_packed_size_i8_icelake, nk_dots_pack_i8_icelake,
                   nk_dots_packed_i8_icelake);
    run_if_matches("dots_u8_icelake", test_dots<u8_t>, nk_dots_packed_size_u8_icelake, nk_dots_pack_u8_icelake,
                   nk_dots_packed_u8_icelake);

    run_if_matches("dots_symmetric_i8_icelake", test_dots_symmetric<i8_t>, nk_dots_symmetric_i8_icelake);
    run_if_matches("dots_symmetric_u8_icelake", test_dots_symmetric<u8_t>, nk_dots_symmetric_u8_icelake);
    run_if_matches("dots_symmetric_i4_icelake", test_dots_symmetric<i4x2_t>, nk_dots_symmetric_i4_icelake);
    run_if_matches("dots_symmetric_u4_icelake", test_dots_symmetric<u4x2_t>, nk_dots_symmetric_u4_icelake);

    run_if_matches("hammings_u1_icelake", test_hammings<u1x8_t>, nk_hammings_packed_size_u1_icelake,
                   nk_hammings_pack_u1_icelake, nk_hammings_packed_u1_icelake);
    run_if_matches("hammings_symmetric_u1_icelake", test_hammings_symmetric<u1x8_t>, nk_hammings_symmetric_u1_icelake);
#endif

#if NK_TARGET_GENOA
    run_if_matches("dots_bf16_genoa", test_dots<bf16_t>, nk_dots_packed_size_bf16_genoa, nk_dots_pack_bf16_genoa,
                   nk_dots_packed_bf16_genoa);
    run_if_matches("dots_e4m3_genoa", test_dots<e4m3_t>, nk_dots_packed_size_e4m3_genoa, nk_dots_pack_e4m3_genoa,
                   nk_dots_packed_e4m3_genoa);
    run_if_matches("dots_e5m2_genoa", test_dots<e5m2_t>, nk_dots_packed_size_e5m2_genoa, nk_dots_pack_e5m2_genoa,
                   nk_dots_packed_e5m2_genoa);
    run_if_matches("dots_e2m3_genoa", test_dots<e2m3_t>, nk_dots_packed_size_e2m3_genoa, nk_dots_pack_e2m3_genoa,
                   nk_dots_packed_e2m3_genoa);
    run_if_matches("dots_e3m2_genoa", test_dots<e3m2_t>, nk_dots_packed_size_e3m2_genoa, nk_dots_pack_e3m2_genoa,
                   nk_dots_packed_e3m2_genoa);

    run_if_matches("dots_symmetric_bf16_genoa", test_dots_symmetric<bf16_t>, nk_dots_symmetric_bf16_genoa);
    run_if_matches("dots_symmetric_e4m3_genoa", test_dots_symmetric<e4m3_t>, nk_dots_symmetric_e4m3_genoa);
    run_if_matches("dots_symmetric_e5m2_genoa", test_dots_symmetric<e5m2_t>, nk_dots_symmetric_e5m2_genoa);
    run_if_matches("dots_symmetric_e2m3_genoa", test_dots_symmetric<e2m3_t>, nk_dots_symmetric_e2m3_genoa);
    run_if_matches("dots_symmetric_e3m2_genoa", test_dots_symmetric<e3m2_t>, nk_dots_symmetric_e3m2_genoa);
#endif

#if NK_TARGET_SIERRA
    run_if_matches("dots_e2m3_sierra", test_dots<e2m3_t>, nk_dots_packed_size_e2m3_sierra, nk_dots_pack_e2m3_sierra,
                   nk_dots_packed_e2m3_sierra);
    run_if_matches("dots_i8_sierra", test_dots<i8_t>, nk_dots_packed_size_i8_sierra, nk_dots_pack_i8_sierra,
                   nk_dots_packed_i8_sierra);
    run_if_matches("dots_u8_sierra", test_dots<u8_t>, nk_dots_packed_size_u8_sierra, nk_dots_pack_u8_sierra,
                   nk_dots_packed_u8_sierra);

    run_if_matches("dots_symmetric_e2m3_sierra", test_dots_symmetric<e2m3_t>, nk_dots_symmetric_e2m3_sierra);
    run_if_matches("dots_symmetric_i8_sierra", test_dots_symmetric<i8_t>, nk_dots_symmetric_i8_sierra);
    run_if_matches("dots_symmetric_u8_sierra", test_dots_symmetric<u8_t>, nk_dots_symmetric_u8_sierra);
#endif
}

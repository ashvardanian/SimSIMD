/**
 *  @brief Batch operation tests - AMX ISA family (Sapphire Rapids AMX).
 *  @file test/test_cross_amx.cpp
 *  @author Ash Vardanian
 *  @date February 6, 2026
 */
#include "test.hpp"
#include "test_cross.hpp"

void test_cross_amx() {
#if NK_TARGET_SAPPHIREAMX
    run_if_matches("dots_bf16_sapphireamx", test_dots<bf16_t>, nk_dots_packed_size_bf16_sapphireamx,
                   nk_dots_pack_bf16_sapphireamx, nk_dots_packed_bf16_sapphireamx);
    run_if_matches("dots_i8_sapphireamx", test_dots<i8_t>, nk_dots_packed_size_i8_sapphireamx,
                   nk_dots_pack_i8_sapphireamx, nk_dots_packed_i8_sapphireamx);
    run_if_matches("dots_e4m3_sapphireamx", test_dots<e4m3_t>, nk_dots_packed_size_e4m3_sapphireamx,
                   nk_dots_pack_e4m3_sapphireamx, nk_dots_packed_e4m3_sapphireamx);
    run_if_matches("dots_e5m2_sapphireamx", test_dots<e5m2_t>, nk_dots_packed_size_e5m2_sapphireamx,
                   nk_dots_pack_e5m2_sapphireamx, nk_dots_packed_e5m2_sapphireamx);
    run_if_matches("dots_packed_e2m3_sapphireamx", test_dots<e2m3_t>, nk_dots_packed_size_e2m3_sapphireamx,
                   nk_dots_pack_e2m3_sapphireamx, nk_dots_packed_e2m3_sapphireamx);
    run_if_matches("dots_packed_e3m2_sapphireamx", test_dots<e3m2_t>, nk_dots_packed_size_e3m2_sapphireamx,
                   nk_dots_pack_e3m2_sapphireamx, nk_dots_packed_e3m2_sapphireamx);

    run_if_matches("dots_symmetric_bf16_sapphireamx", test_dots_symmetric<bf16_t>, nk_dots_symmetric_bf16_sapphireamx);
    run_if_matches("dots_symmetric_i8_sapphireamx", test_dots_symmetric<i8_t>, nk_dots_symmetric_i8_sapphireamx);
    run_if_matches("dots_symmetric_e2m3_sapphireamx", test_dots_symmetric<e2m3_t>, nk_dots_symmetric_e2m3_sapphireamx);
    run_if_matches("dots_symmetric_e3m2_sapphireamx", test_dots_symmetric<e3m2_t>, nk_dots_symmetric_e3m2_sapphireamx);
#endif
}

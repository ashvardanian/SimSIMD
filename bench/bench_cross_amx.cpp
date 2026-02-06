/**
 *  @brief Batch operation benchmarks - AMX ISA family (Sapphire Rapids AMX).
 *  @file bench/bench_cross_amx.cpp
 *  @author Ash Vardanian
 *  @date February 6, 2026
 */

#include "numkong/dot.h"
#include "numkong/dots.h"

#include "bench.hpp"

void bench_cross_amx() {
#if NK_TARGET_SAPPHIREAMX
    constexpr nk_dtype_t i8_k = nk_i8_k;
    constexpr nk_dtype_t i32_k = nk_i32_k;
    constexpr nk_dtype_t f32_k = nk_f32_k;
    constexpr nk_dtype_t bf16_k = nk_bf16_k;
    constexpr nk_dtype_t e4m3_k = nk_e4m3_k;
    constexpr nk_dtype_t e5m2_k = nk_e5m2_k;

    dots_<bf16_k, f32_k>("dots_packed_bf16_sapphireamx", nk_dots_packed_size_bf16_sapphireamx,
                         nk_dots_pack_bf16_sapphireamx, nk_dots_packed_bf16_sapphireamx);
    dots_<i8_k, i32_k>("dots_packed_i8_sapphireamx", nk_dots_packed_size_i8_sapphireamx, nk_dots_pack_i8_sapphireamx,
                       nk_dots_packed_i8_sapphireamx);
    dots_<e4m3_k, f32_k>("dots_packed_e4m3_sapphireamx", nk_dots_packed_size_e4m3_sapphireamx,
                         nk_dots_pack_e4m3_sapphireamx, nk_dots_packed_e4m3_sapphireamx);
    dots_<e5m2_k, f32_k>("dots_packed_e5m2_sapphireamx", nk_dots_packed_size_e5m2_sapphireamx,
                         nk_dots_pack_e5m2_sapphireamx, nk_dots_packed_e5m2_sapphireamx);
#endif
}

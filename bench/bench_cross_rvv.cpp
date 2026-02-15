/**
 *  @brief Batch operation benchmarks - RVV ISA family (RISC-V Vector).
 *  @file bench/bench_cross_rvv.cpp
 *  @author Ash Vardanian
 *  @date February 15, 2026
 */

#include "numkong/dot.h"
#include "numkong/dots.h"

#include "bench.hpp"

void bench_cross_rvv() {
    constexpr nk_dtype_t i8_k = nk_i8_k;
    constexpr nk_dtype_t u8_k = nk_u8_k;
    constexpr nk_dtype_t i32_k = nk_i32_k;
    constexpr nk_dtype_t u32_k = nk_u32_k;
    constexpr nk_dtype_t f64_k = nk_f64_k;
    constexpr nk_dtype_t f32_k = nk_f32_k;
    constexpr nk_dtype_t f16_k = nk_f16_k;
    constexpr nk_dtype_t bf16_k = nk_bf16_k;
    constexpr nk_dtype_t e4m3_k = nk_e4m3_k;
    constexpr nk_dtype_t e5m2_k = nk_e5m2_k;
    constexpr nk_dtype_t e2m3_k = nk_e2m3_k;
    constexpr nk_dtype_t e3m2_k = nk_e3m2_k;

#if NK_TARGET_RVV
    dots_<f32_k, f32_k>("dots_packed_f32_rvv", nk_dots_packed_size_f32_rvv, nk_dots_pack_f32_rvv,
                        nk_dots_packed_f32_rvv);
    dots_<f64_k, f64_k>("dots_packed_f64_rvv", nk_dots_packed_size_f64_rvv, nk_dots_pack_f64_rvv,
                        nk_dots_packed_f64_rvv);
    dots_<bf16_k, f32_k>("dots_packed_bf16_rvv", nk_dots_packed_size_bf16_rvv, nk_dots_pack_bf16_rvv,
                         nk_dots_packed_bf16_rvv);
    dots_<f16_k, f32_k>("dots_packed_f16_rvv", nk_dots_packed_size_f16_rvv, nk_dots_pack_f16_rvv,
                        nk_dots_packed_f16_rvv);
    dots_<i8_k, i32_k>("dots_packed_i8_rvv", nk_dots_packed_size_i8_rvv, nk_dots_pack_i8_rvv, nk_dots_packed_i8_rvv);
    dots_<u8_k, u32_k>("dots_packed_u8_rvv", nk_dots_packed_size_u8_rvv, nk_dots_pack_u8_rvv, nk_dots_packed_u8_rvv);
    dots_<e2m3_k, f32_k>("dots_packed_e2m3_rvv", nk_dots_packed_size_e2m3_rvv, nk_dots_pack_e2m3_rvv,
                         nk_dots_packed_e2m3_rvv);
    dots_<e3m2_k, f32_k>("dots_packed_e3m2_rvv", nk_dots_packed_size_e3m2_rvv, nk_dots_pack_e3m2_rvv,
                         nk_dots_packed_e3m2_rvv);
    dots_<e4m3_k, f32_k>("dots_packed_e4m3_rvv", nk_dots_packed_size_e4m3_rvv, nk_dots_pack_e4m3_rvv,
                         nk_dots_packed_e4m3_rvv);
    dots_<e5m2_k, f32_k>("dots_packed_e5m2_rvv", nk_dots_packed_size_e5m2_rvv, nk_dots_pack_e5m2_rvv,
                         nk_dots_packed_e5m2_rvv);
    dots_symmetric_<f32_k, f32_k>("dots_symmetric_f32_rvv", nk_dots_symmetric_f32_rvv);
    dots_symmetric_<f64_k, f64_k>("dots_symmetric_f64_rvv", nk_dots_symmetric_f64_rvv);
    dots_symmetric_<bf16_k, f32_k>("dots_symmetric_bf16_rvv", nk_dots_symmetric_bf16_rvv);
    dots_symmetric_<f16_k, f32_k>("dots_symmetric_f16_rvv", nk_dots_symmetric_f16_rvv);
    dots_symmetric_<i8_k, i32_k>("dots_symmetric_i8_rvv", nk_dots_symmetric_i8_rvv);
    dots_symmetric_<u8_k, u32_k>("dots_symmetric_u8_rvv", nk_dots_symmetric_u8_rvv);
    dots_symmetric_<e2m3_k, f32_k>("dots_symmetric_e2m3_rvv", nk_dots_symmetric_e2m3_rvv);
    dots_symmetric_<e3m2_k, f32_k>("dots_symmetric_e3m2_rvv", nk_dots_symmetric_e3m2_rvv);
    dots_symmetric_<e4m3_k, f32_k>("dots_symmetric_e4m3_rvv", nk_dots_symmetric_e4m3_rvv);
    dots_symmetric_<e5m2_k, f32_k>("dots_symmetric_e5m2_rvv", nk_dots_symmetric_e5m2_rvv);
#endif
}

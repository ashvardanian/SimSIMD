/**
 *  @brief Batch operation benchmarks - SME ISA.
 *  @file bench/bench_cross_sme.cpp
 *  @author Ash Vardanian
 *  @date January 14, 2025
 */

#include "numkong/dot.h"
#include "numkong/dots.h"
#include "numkong/sets.h"

#include "bench.hpp"

void bench_cross_sme() {
    constexpr nk_dtype_t f64_k = nk_f64_k;
    constexpr nk_dtype_t f32_k = nk_f32_k;
    constexpr nk_dtype_t u1_k = nk_u1_k;
    constexpr nk_dtype_t i4_k = nk_i4_k;
    constexpr nk_dtype_t u4_k = nk_u4_k;
    constexpr nk_dtype_t i8_k = nk_i8_k;
    constexpr nk_dtype_t u8_k = nk_u8_k;
    constexpr nk_dtype_t i32_k = nk_i32_k;
    constexpr nk_dtype_t u32_k = nk_u32_k;
    constexpr nk_dtype_t f16_k = nk_f16_k;
    constexpr nk_dtype_t bf16_k = nk_bf16_k;
    constexpr nk_dtype_t e4m3_k = nk_e4m3_k;
    constexpr nk_dtype_t e5m2_k = nk_e5m2_k;

#if NK_TARGET_SME
    dots_<f16_k, f32_k>("dots_packed_f16_sme", nk_dots_packed_size_f16_sme, nk_dots_pack_f16_sme,
                        nk_dots_packed_f16_sme);
    dots_<bf16_k, f32_k>("dots_packed_bf16_sme", nk_dots_packed_size_bf16_sme, nk_dots_pack_bf16_sme,
                         nk_dots_packed_bf16_sme);
    dots_<i8_k, i32_k>("dots_packed_i8_sme", nk_dots_packed_size_i8_sme, nk_dots_pack_i8_sme, nk_dots_packed_i8_sme);
    dots_<u8_k, u32_k>("dots_packed_u8_sme", nk_dots_packed_size_u8_sme, nk_dots_pack_u8_sme, nk_dots_packed_u8_sme);
    dots_<e4m3_k, f32_k>("dots_packed_e4m3_sme", nk_dots_packed_size_e4m3_sme, nk_dots_pack_e4m3_sme,
                         nk_dots_packed_e4m3_sme);
    dots_<e5m2_k, f32_k>("dots_packed_e5m2_sme", nk_dots_packed_size_e5m2_sme, nk_dots_pack_e5m2_sme,
                         nk_dots_packed_e5m2_sme);
    dots_<i4_k, i32_k>("dots_packed_i4_sme", nk_dots_packed_size_i4_sme, nk_dots_pack_i4_sme, nk_dots_packed_i4_sme);
    dots_<u4_k, u32_k>("dots_packed_u4_sme", nk_dots_packed_size_u4_sme, nk_dots_pack_u4_sme, nk_dots_packed_u4_sme);
    dots_symmetric_<bf16_k, f32_k>("dots_symmetric_bf16_sme", nk_dots_symmetric_bf16_sme);
    dots_symmetric_<f16_k, f32_k>("dots_symmetric_f16_sme", nk_dots_symmetric_f16_sme);
    dots_symmetric_<i8_k, i32_k>("dots_symmetric_i8_sme", nk_dots_symmetric_i8_sme);
    dots_symmetric_<u8_k, u32_k>("dots_symmetric_u8_sme", nk_dots_symmetric_u8_sme);
    dots_symmetric_<e4m3_k, f32_k>("dots_symmetric_e4m3_sme", nk_dots_symmetric_e4m3_sme);
    dots_symmetric_<e5m2_k, f32_k>("dots_symmetric_e5m2_sme", nk_dots_symmetric_e5m2_sme);
    dots_symmetric_<i4_k, i32_k>("dots_symmetric_i4_sme", nk_dots_symmetric_i4_sme);
    dots_symmetric_<u4_k, u32_k>("dots_symmetric_u4_sme", nk_dots_symmetric_u4_sme);
#endif

#if NK_TARGET_SMEBI32

    hammings_<u1_k, u32_k>("hammings_packed_u1_smebi32", nk_hammings_packed_size_u1_smebi32,
                           nk_hammings_pack_u1_smebi32, nk_hammings_packed_u1_smebi32);
    hammings_symmetric_<u1_k, u32_k>("hammings_symmetric_u1_smebi32", nk_hammings_symmetric_u1_smebi32);
#endif

#if NK_TARGET_SMEF64

    dots_<f32_k, f32_k>("dots_packed_f32_smef64", nk_dots_packed_size_f32_smef64, nk_dots_pack_f32_smef64,
                        nk_dots_packed_f32_smef64);
    dots_<f64_k, f64_k>("dots_packed_f64_smef64", nk_dots_packed_size_f64_smef64, nk_dots_pack_f64_smef64,
                        nk_dots_packed_f64_smef64);
    dots_symmetric_<f32_k, f32_k>("dots_symmetric_f32_smef64", nk_dots_symmetric_f32_smef64);
    dots_symmetric_<f64_k, f64_k>("dots_symmetric_f64_smef64", nk_dots_symmetric_f64_smef64);
#endif
}

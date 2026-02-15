/**
 *  @brief Batch operation benchmarks - Arm ISA family.
 *  @file bench/bench_cross_arm.cpp
 *  @author Ash Vardanian
 *  @date February 6, 2026
 *
 *  Covers NEON, NEONHALF, NEONFHM, NEONBFDOT, NEONSDOT.
 */

#include "numkong/dot.h"
#include "numkong/dots.h"
#include "numkong/sets.h"

#include "bench.hpp"

void bench_cross_arm() {
    constexpr nk_dtype_t f64_k = nk_f64_k;
    constexpr nk_dtype_t f32_k = nk_f32_k;
    constexpr nk_dtype_t f16_k = nk_f16_k;
    constexpr nk_dtype_t bf16_k = nk_bf16_k;
    constexpr nk_dtype_t e2m3_k = nk_e2m3_k;
    constexpr nk_dtype_t e5m2_k = nk_e5m2_k;
    constexpr nk_dtype_t e3m2_k = nk_e3m2_k;
    constexpr nk_dtype_t e4m3_k = nk_e4m3_k;
    constexpr nk_dtype_t i4_k = nk_i4_k;
    constexpr nk_dtype_t u4_k = nk_u4_k;
    constexpr nk_dtype_t i8_k = nk_i8_k;
    constexpr nk_dtype_t u8_k = nk_u8_k;
    constexpr nk_dtype_t i32_k = nk_i32_k;
    constexpr nk_dtype_t u32_k = nk_u32_k;

#if NK_TARGET_NEON

    dots_<f32_k, f32_k>("dots_packed_f32_neon", nk_dots_packed_size_f32_neon, nk_dots_pack_f32_neon,
                        nk_dots_packed_f32_neon);
    dots_<f64_k, f64_k>("dots_packed_f64_neon", nk_dots_packed_size_f64_neon, nk_dots_pack_f64_neon,
                        nk_dots_packed_f64_neon);
    dots_symmetric_<f32_k, f32_k>("dots_symmetric_f32_neon", nk_dots_symmetric_f32_neon);
    dots_symmetric_<f64_k, f64_k>("dots_symmetric_f64_neon", nk_dots_symmetric_f64_neon);
    hammings_<nk_u1_k, nk_u32_k>("hammings_u1_neon", nk_hammings_packed_size_u1_neon, nk_hammings_pack_u1_neon,
                                 nk_hammings_packed_u1_neon);
    hammings_symmetric_<nk_u1_k, nk_u32_k>("hammings_symmetric_u1_neon", nk_hammings_symmetric_u1_neon);

#endif

#if NK_TARGET_NEONHALF
    dots_<f16_k, f32_k>("dots_packed_f16_neonhalf", nk_dots_packed_size_f16_neonhalf, nk_dots_pack_f16_neonhalf,
                        nk_dots_packed_f16_neonhalf);
    dots_symmetric_<f16_k, f32_k>("dots_symmetric_f16_neonhalf", nk_dots_symmetric_f16_neonhalf);
#endif

#if NK_TARGET_NEONFHM

    dots_<f16_k, f32_k>("dots_packed_f16_neonfhm", nk_dots_packed_size_f16_neonfhm, nk_dots_pack_f16_neonfhm,
                        nk_dots_packed_f16_neonfhm);
    dots_<e2m3_k, f32_k>("dots_packed_e2m3_neonfhm", nk_dots_packed_size_e2m3_neonfhm, nk_dots_pack_e2m3_neonfhm,
                         nk_dots_packed_e2m3_neonfhm);
    dots_<e3m2_k, f32_k>("dots_packed_e3m2_neonfhm", nk_dots_packed_size_e3m2_neonfhm, nk_dots_pack_e3m2_neonfhm,
                         nk_dots_packed_e3m2_neonfhm);
    dots_<e4m3_k, f32_k>("dots_packed_e4m3_neonfhm", nk_dots_packed_size_e4m3_neonfhm, nk_dots_pack_e4m3_neonfhm,
                         nk_dots_packed_e4m3_neonfhm);
    dots_<e5m2_k, f32_k>("dots_packed_e5m2_neonfhm", nk_dots_packed_size_e5m2_neonfhm, nk_dots_pack_e5m2_neonfhm,
                         nk_dots_packed_e5m2_neonfhm);
    dots_symmetric_<f16_k, f32_k>("dots_symmetric_f16_neonfhm", nk_dots_symmetric_f16_neonfhm);
    dots_symmetric_<e2m3_k, f32_k>("dots_symmetric_e2m3_neonfhm", nk_dots_symmetric_e2m3_neonfhm);
    dots_symmetric_<e3m2_k, f32_k>("dots_symmetric_e3m2_neonfhm", nk_dots_symmetric_e3m2_neonfhm);
    dots_symmetric_<e4m3_k, f32_k>("dots_symmetric_e4m3_neonfhm", nk_dots_symmetric_e4m3_neonfhm);
    dots_symmetric_<e5m2_k, f32_k>("dots_symmetric_e5m2_neonfhm", nk_dots_symmetric_e5m2_neonfhm);
#endif

#if NK_TARGET_NEONBFDOT
    dots_<bf16_k, f32_k>("dots_packed_bf16_neonbfdot", nk_dots_packed_size_bf16_neonbfdot, nk_dots_pack_bf16_neonbfdot,
                         nk_dots_packed_bf16_neonbfdot);
    dots_symmetric_<bf16_k, f32_k>("dots_symmetric_bf16_neonbfdot", nk_dots_symmetric_bf16_neonbfdot);
#endif

#if NK_TARGET_NEONSDOT

    dots_<i8_k, i32_k>("dots_packed_i8_neonsdot", nk_dots_packed_size_i8_neonsdot, nk_dots_pack_i8_neonsdot,
                       nk_dots_packed_i8_neonsdot);
    dots_<u8_k, u32_k>("dots_packed_u8_neonsdot", nk_dots_packed_size_u8_neonsdot, nk_dots_pack_u8_neonsdot,
                       nk_dots_packed_u8_neonsdot);
    dots_<i4_k, i32_k>("dots_packed_i4_neonsdot", nk_dots_packed_size_i4_neonsdot, nk_dots_pack_i4_neonsdot,
                       nk_dots_packed_i4_neonsdot);
    dots_<u4_k, u32_k>("dots_packed_u4_neonsdot", nk_dots_packed_size_u4_neonsdot, nk_dots_pack_u4_neonsdot,
                       nk_dots_packed_u4_neonsdot);
    dots_symmetric_<i8_k, i32_k>("dots_symmetric_i8_neonsdot", nk_dots_symmetric_i8_neonsdot);
    dots_symmetric_<u8_k, u32_k>("dots_symmetric_u8_neonsdot", nk_dots_symmetric_u8_neonsdot);
    dots_symmetric_<i4_k, i32_k>("dots_symmetric_i4_neonsdot", nk_dots_symmetric_i4_neonsdot);
    dots_symmetric_<u4_k, u32_k>("dots_symmetric_u4_neonsdot", nk_dots_symmetric_u4_neonsdot);

#endif
}

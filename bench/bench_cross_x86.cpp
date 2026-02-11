/**
 *  @brief Batch operation benchmarks - x86 ISA family.
 *  @file bench/bench_cross_x86.cpp
 *  @author Ash Vardanian
 *  @date February 6, 2026
 *
 *  Covers Haswell, Skylake, Ice Lake, Genoa, Sapphire.
 */

#include "numkong/dot.h"
#include "numkong/dots.h"
#include "numkong/sets.h"

#include "bench.hpp"

void bench_cross_x86() {
    constexpr nk_dtype_t i4_k = nk_i4_k;
    constexpr nk_dtype_t u4_k = nk_u4_k;
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

#if NK_TARGET_HASWELL

    dots_<f32_k, f32_k>("dots_packed_f32_haswell", nk_dots_packed_size_f32_haswell, nk_dots_pack_f32_haswell,
                        nk_dots_packed_f32_haswell);
    dots_<f64_k, f64_k>("dots_packed_f64_haswell", nk_dots_packed_size_f64_haswell, nk_dots_pack_f64_haswell,
                        nk_dots_packed_f64_haswell);
    dots_<f16_k, f32_k>("dots_packed_f16_haswell", nk_dots_packed_size_f16_haswell, nk_dots_pack_f16_haswell,
                        nk_dots_packed_f16_haswell);
    dots_<bf16_k, f32_k>("dots_packed_bf16_haswell", nk_dots_packed_size_bf16_haswell, nk_dots_pack_bf16_haswell,
                         nk_dots_packed_bf16_haswell);
    dots_<e4m3_k, f32_k>("dots_packed_e4m3_haswell", nk_dots_packed_size_e4m3_haswell, nk_dots_pack_e4m3_haswell,
                         nk_dots_packed_e4m3_haswell);
    dots_<e5m2_k, f32_k>("dots_packed_e5m2_haswell", nk_dots_packed_size_e5m2_haswell, nk_dots_pack_e5m2_haswell,
                         nk_dots_packed_e5m2_haswell);
    dots_<e2m3_k, f32_k>("dots_packed_e2m3_haswell", nk_dots_packed_size_e2m3_haswell, nk_dots_pack_e2m3_haswell,
                         nk_dots_packed_e2m3_haswell);
    dots_<e3m2_k, f32_k>("dots_packed_e3m2_haswell", nk_dots_packed_size_e3m2_haswell, nk_dots_pack_e3m2_haswell,
                         nk_dots_packed_e3m2_haswell);
    dots_<i8_k, i32_k>("dots_packed_i8_haswell", nk_dots_packed_size_i8_haswell, nk_dots_pack_i8_haswell,
                       nk_dots_packed_i8_haswell);
    dots_<u8_k, u32_k>("dots_packed_u8_haswell", nk_dots_packed_size_u8_haswell, nk_dots_pack_u8_haswell,
                       nk_dots_packed_u8_haswell);
    dots_symmetric_<f32_k, f32_k>("dots_symmetric_f32_haswell", nk_dots_symmetric_f32_haswell);
    dots_symmetric_<f64_k, f64_k>("dots_symmetric_f64_haswell", nk_dots_symmetric_f64_haswell);
    dots_symmetric_<bf16_k, f32_k>("dots_symmetric_bf16_haswell", nk_dots_symmetric_bf16_haswell);
    dots_symmetric_<f16_k, f32_k>("dots_symmetric_f16_haswell", nk_dots_symmetric_f16_haswell);
    dots_symmetric_<i8_k, i32_k>("dots_symmetric_i8_haswell", nk_dots_symmetric_i8_haswell);
    dots_symmetric_<u8_k, u32_k>("dots_symmetric_u8_haswell", nk_dots_symmetric_u8_haswell);
    dots_symmetric_<e4m3_k, f32_k>("dots_symmetric_e4m3_haswell", nk_dots_symmetric_e4m3_haswell);
    dots_symmetric_<e5m2_k, f32_k>("dots_symmetric_e5m2_haswell", nk_dots_symmetric_e5m2_haswell);
    dots_symmetric_<e2m3_k, f32_k>("dots_symmetric_e2m3_haswell", nk_dots_symmetric_e2m3_haswell);
    dots_symmetric_<e3m2_k, f32_k>("dots_symmetric_e3m2_haswell", nk_dots_symmetric_e3m2_haswell);
    hammings_<nk_u1_k, nk_u32_k>("hammings_u1_haswell", nk_hammings_packed_size_u1_haswell, nk_hammings_pack_u1_haswell,
                                 nk_hammings_packed_u1_haswell);
    hammings_symmetric_<nk_u1_k, nk_u32_k>("hammings_symmetric_u1_haswell", nk_hammings_symmetric_u1_haswell);
#endif

#if NK_TARGET_SKYLAKE

    dots_<f32_k, f32_k>("dots_packed_f32_skylake", nk_dots_packed_size_f32_skylake, nk_dots_pack_f32_skylake,
                        nk_dots_packed_f32_skylake);
    dots_<f64_k, f64_k>("dots_packed_f64_skylake", nk_dots_packed_size_f64_skylake, nk_dots_pack_f64_skylake,
                        nk_dots_packed_f64_skylake);
    dots_<bf16_k, f32_k>("dots_packed_bf16_skylake", nk_dots_packed_size_bf16_skylake, nk_dots_pack_bf16_skylake,
                         nk_dots_packed_bf16_skylake);
    dots_<f16_k, f32_k>("dots_packed_f16_skylake", nk_dots_packed_size_f16_skylake, nk_dots_pack_f16_skylake,
                        nk_dots_packed_f16_skylake);
    dots_<e4m3_k, f32_k>("dots_packed_e4m3_skylake", nk_dots_packed_size_e4m3_skylake, nk_dots_pack_e4m3_skylake,
                         nk_dots_packed_e4m3_skylake);
    dots_<e5m2_k, f32_k>("dots_packed_e5m2_skylake", nk_dots_packed_size_e5m2_skylake, nk_dots_pack_e5m2_skylake,
                         nk_dots_packed_e5m2_skylake);
    dots_<e2m3_k, f32_k>("dots_packed_e2m3_skylake", nk_dots_packed_size_e2m3_skylake, nk_dots_pack_e2m3_skylake,
                         nk_dots_packed_e2m3_skylake);
    dots_<e3m2_k, f32_k>("dots_packed_e3m2_skylake", nk_dots_packed_size_e3m2_skylake, nk_dots_pack_e3m2_skylake,
                         nk_dots_packed_e3m2_skylake);
    dots_symmetric_<f32_k, f32_k>("dots_symmetric_f32_skylake", nk_dots_symmetric_f32_skylake);
    dots_symmetric_<f64_k, f64_k>("dots_symmetric_f64_skylake", nk_dots_symmetric_f64_skylake);
    dots_symmetric_<bf16_k, f32_k>("dots_symmetric_bf16_skylake", nk_dots_symmetric_bf16_skylake);
    dots_symmetric_<f16_k, f32_k>("dots_symmetric_f16_skylake", nk_dots_symmetric_f16_skylake);
    dots_symmetric_<e4m3_k, f32_k>("dots_symmetric_e4m3_skylake", nk_dots_symmetric_e4m3_skylake);
    dots_symmetric_<e5m2_k, f32_k>("dots_symmetric_e5m2_skylake", nk_dots_symmetric_e5m2_skylake);
    dots_symmetric_<e2m3_k, f32_k>("dots_symmetric_e2m3_skylake", nk_dots_symmetric_e2m3_skylake);
    dots_symmetric_<e3m2_k, f32_k>("dots_symmetric_e3m2_skylake", nk_dots_symmetric_e3m2_skylake);

#endif

#if NK_TARGET_ICELAKE

    dots_<i4_k, i32_k>("dots_packed_i4_icelake", nk_dots_packed_size_i4_icelake, nk_dots_pack_i4_icelake,
                       nk_dots_packed_i4_icelake);
    dots_<u4_k, u32_k>("dots_packed_u4_icelake", nk_dots_packed_size_u4_icelake, nk_dots_pack_u4_icelake,
                       nk_dots_packed_u4_icelake);
    dots_<i8_k, i32_k>("dots_packed_i8_icelake", nk_dots_packed_size_i8_icelake, nk_dots_pack_i8_icelake,
                       nk_dots_packed_i8_icelake);
    dots_<u8_k, u32_k>("dots_packed_u8_icelake", nk_dots_packed_size_u8_icelake, nk_dots_pack_u8_icelake,
                       nk_dots_packed_u8_icelake);
    dots_symmetric_<i8_k, i32_k>("dots_symmetric_i8_icelake", nk_dots_symmetric_i8_icelake);
    dots_symmetric_<u8_k, u32_k>("dots_symmetric_u8_icelake", nk_dots_symmetric_u8_icelake);
    dots_symmetric_<i4_k, i32_k>("dots_symmetric_i4_icelake", nk_dots_symmetric_i4_icelake);
    dots_symmetric_<u4_k, u32_k>("dots_symmetric_u4_icelake", nk_dots_symmetric_u4_icelake);
    hammings_<nk_u1_k, nk_u32_k>("hammings_u1_icelake", nk_hammings_packed_size_u1_icelake, nk_hammings_pack_u1_icelake,
                                 nk_hammings_packed_u1_icelake);
    hammings_symmetric_<nk_u1_k, nk_u32_k>("hammings_symmetric_u1_icelake", nk_hammings_symmetric_u1_icelake);

#endif

#if NK_TARGET_GENOA

    dots_<bf16_k, f32_k>("dots_packed_bf16_genoa", nk_dots_packed_size_bf16_genoa, nk_dots_pack_bf16_genoa,
                         nk_dots_packed_bf16_genoa);
    dots_<e4m3_k, f32_k>("dots_packed_e4m3_genoa", nk_dots_packed_size_e4m3_genoa, nk_dots_pack_e4m3_genoa,
                         nk_dots_packed_e4m3_genoa);
    dots_<e5m2_k, f32_k>("dots_packed_e5m2_genoa", nk_dots_packed_size_e5m2_genoa, nk_dots_pack_e5m2_genoa,
                         nk_dots_packed_e5m2_genoa);
    dots_<e2m3_k, f32_k>("dots_packed_e2m3_genoa", nk_dots_packed_size_e2m3_genoa, nk_dots_pack_e2m3_genoa,
                         nk_dots_packed_e2m3_genoa);
    dots_<e3m2_k, f32_k>("dots_packed_e3m2_genoa", nk_dots_packed_size_e3m2_genoa, nk_dots_pack_e3m2_genoa,
                         nk_dots_packed_e3m2_genoa);
    dots_symmetric_<bf16_k, f32_k>("dots_symmetric_bf16_genoa", nk_dots_symmetric_bf16_genoa);
    dots_symmetric_<e4m3_k, f32_k>("dots_symmetric_e4m3_genoa", nk_dots_symmetric_e4m3_genoa);
    dots_symmetric_<e5m2_k, f32_k>("dots_symmetric_e5m2_genoa", nk_dots_symmetric_e5m2_genoa);
    dots_symmetric_<e2m3_k, f32_k>("dots_symmetric_e2m3_genoa", nk_dots_symmetric_e2m3_genoa);
    dots_symmetric_<e3m2_k, f32_k>("dots_symmetric_e3m2_genoa", nk_dots_symmetric_e3m2_genoa);

#endif

#if NK_TARGET_SIERRA

    dots_<e2m3_k, f32_k>("dots_packed_e2m3_sierra", nk_dots_packed_size_e2m3_sierra, nk_dots_pack_e2m3_sierra,
                         nk_dots_packed_e2m3_sierra);
    dots_<i8_k, i32_k>("dots_packed_i8_sierra", nk_dots_packed_size_i8_sierra, nk_dots_pack_i8_sierra,
                       nk_dots_packed_i8_sierra);
    dots_<u8_k, u32_k>("dots_packed_u8_sierra", nk_dots_packed_size_u8_sierra, nk_dots_pack_u8_sierra,
                       nk_dots_packed_u8_sierra);
    dots_symmetric_<e2m3_k, f32_k>("dots_symmetric_e2m3_sierra", nk_dots_symmetric_e2m3_sierra);
    dots_symmetric_<i8_k, i32_k>("dots_symmetric_i8_sierra", nk_dots_symmetric_i8_sierra);
    dots_symmetric_<u8_k, u32_k>("dots_symmetric_u8_sierra", nk_dots_symmetric_u8_sierra);

#endif
}

/**
 *  @brief Spatial distance benchmarks (angular, sqeuclidean, euclidean).
 *  @file bench/bench_spatial.cpp
 *  @author Ash Vardanian
 *  @date March 14, 2023
 */

#include "numkong/spatial.h"

#include "bench.hpp"

void bench_spatial() {
    constexpr nk_dtype_t i4_k = nk_i4_k;
    constexpr nk_dtype_t u4_k = nk_u4_k;
    constexpr nk_dtype_t i8_k = nk_i8_k;
    constexpr nk_dtype_t u8_k = nk_u8_k;
    constexpr nk_dtype_t u32_k = nk_u32_k;
    constexpr nk_dtype_t f64_k = nk_f64_k;
    constexpr nk_dtype_t f32_k = nk_f32_k;
    constexpr nk_dtype_t f16_k = nk_f16_k;
    constexpr nk_dtype_t bf16_k = nk_bf16_k;
    constexpr nk_dtype_t e4m3_k = nk_e4m3_k;
    constexpr nk_dtype_t e5m2_k = nk_e5m2_k;
    constexpr nk_dtype_t e2m3_k = nk_e2m3_k;
    constexpr nk_dtype_t e3m2_k = nk_e3m2_k;

#if NK_TARGET_NEON
    dense_<f32_k, f32_k>("angular_f32_neon", nk_angular_f32_neon);
    dense_<f32_k, f32_k>("sqeuclidean_f32_neon", nk_sqeuclidean_f32_neon);
    dense_<f32_k, f32_k>("euclidean_f32_neon", nk_euclidean_f32_neon);
    dense_<f64_k, f64_k>("angular_f64_neon", nk_angular_f64_neon);
    dense_<f64_k, f64_k>("sqeuclidean_f64_neon", nk_sqeuclidean_f64_neon);
    dense_<f64_k, f64_k>("euclidean_f64_neon", nk_euclidean_f64_neon);
#endif

#if NK_TARGET_NEONSDOT
    dense_<i8_k, f32_k>("angular_i8_neonsdot", nk_angular_i8_neonsdot);
    dense_<i8_k, u32_k>("sqeuclidean_i8_neonsdot", nk_sqeuclidean_i8_neonsdot);
    dense_<i8_k, f32_k>("euclidean_i8_neonsdot", nk_euclidean_i8_neonsdot);
    dense_<u8_k, f32_k>("angular_u8_neonsdot", nk_angular_u8_neonsdot);
    dense_<u8_k, u32_k>("sqeuclidean_u8_neonsdot", nk_sqeuclidean_u8_neonsdot);
    dense_<u8_k, f32_k>("euclidean_u8_neonsdot", nk_euclidean_u8_neonsdot);
#endif

#if NK_TARGET_NEONHALF
    dense_<f16_k, f32_k>("angular_f16_neonhalf", nk_angular_f16_neonhalf);
    dense_<f16_k, f32_k>("sqeuclidean_f16_neonhalf", nk_sqeuclidean_f16_neonhalf);
    dense_<f16_k, f32_k>("euclidean_f16_neonhalf", nk_euclidean_f16_neonhalf);
#endif

#if NK_TARGET_NEONBFDOT
    dense_<bf16_k, f32_k>("angular_bf16_neonbfdot", nk_angular_bf16_neonbfdot);
    dense_<bf16_k, f32_k>("sqeuclidean_bf16_neonbfdot", nk_sqeuclidean_bf16_neonbfdot);
    dense_<bf16_k, f32_k>("euclidean_bf16_neonbfdot", nk_euclidean_bf16_neonbfdot);
#endif

#if NK_TARGET_SVE
    dense_<f32_k, f32_k>("angular_f32_sve", nk_angular_f32_sve);
    dense_<f32_k, f32_k>("sqeuclidean_f32_sve", nk_sqeuclidean_f32_sve);
    dense_<f32_k, f32_k>("euclidean_f32_sve", nk_euclidean_f32_sve);
    dense_<f64_k, f64_k>("angular_f64_sve", nk_angular_f64_sve);
    dense_<f64_k, f64_k>("sqeuclidean_f64_sve", nk_sqeuclidean_f64_sve);
    dense_<f64_k, f64_k>("euclidean_f64_sve", nk_euclidean_f64_sve);
#endif

#if NK_TARGET_SVEHALF
    dense_<f16_k, f32_k>("angular_f16_svehalf", nk_angular_f16_svehalf);
    dense_<f16_k, f32_k>("sqeuclidean_f16_svehalf", nk_sqeuclidean_f16_svehalf);
    dense_<f16_k, f32_k>("euclidean_f16_svehalf", nk_euclidean_f16_svehalf);
#endif

#if NK_TARGET_SVEBFDOT
    dense_<bf16_k, f32_k>("angular_bf16_svebfdot", nk_angular_bf16_svebfdot);
    dense_<bf16_k, f32_k>("sqeuclidean_bf16_svebfdot", nk_sqeuclidean_bf16_svebfdot);
    dense_<bf16_k, f32_k>("euclidean_bf16_svebfdot", nk_euclidean_bf16_svebfdot);
#endif

#if NK_TARGET_HASWELL
    dense_<f16_k, f32_k>("angular_f16_haswell", nk_angular_f16_haswell);
    dense_<f16_k, f32_k>("sqeuclidean_f16_haswell", nk_sqeuclidean_f16_haswell);
    dense_<f16_k, f32_k>("euclidean_f16_haswell", nk_euclidean_f16_haswell);
    dense_<bf16_k, f32_k>("angular_bf16_haswell", nk_angular_bf16_haswell);
    dense_<bf16_k, f32_k>("sqeuclidean_bf16_haswell", nk_sqeuclidean_bf16_haswell);
    dense_<bf16_k, f32_k>("euclidean_bf16_haswell", nk_euclidean_bf16_haswell);
    dense_<i8_k, f32_k>("angular_i8_haswell", nk_angular_i8_haswell);
    dense_<i8_k, u32_k>("sqeuclidean_i8_haswell", nk_sqeuclidean_i8_haswell);
    dense_<i8_k, f32_k>("euclidean_i8_haswell", nk_euclidean_i8_haswell);
    dense_<u8_k, f32_k>("angular_u8_haswell", nk_angular_u8_haswell);
    dense_<u8_k, u32_k>("sqeuclidean_u8_haswell", nk_sqeuclidean_u8_haswell);
    dense_<u8_k, f32_k>("euclidean_u8_haswell", nk_euclidean_u8_haswell);
#endif

#if NK_TARGET_SKYLAKE
    dense_<f32_k, f32_k>("angular_f32_skylake", nk_angular_f32_skylake);
    dense_<f32_k, f32_k>("sqeuclidean_f32_skylake", nk_sqeuclidean_f32_skylake);
    dense_<f32_k, f32_k>("euclidean_f32_skylake", nk_euclidean_f32_skylake);
    dense_<f64_k, f64_k>("angular_f64_skylake", nk_angular_f64_skylake);
    dense_<f64_k, f64_k>("sqeuclidean_f64_skylake", nk_sqeuclidean_f64_skylake);
    dense_<f64_k, f64_k>("euclidean_f64_skylake", nk_euclidean_f64_skylake);
    dense_<e4m3_k, f32_k>("angular_e4m3_skylake", nk_angular_e4m3_skylake);
    dense_<e4m3_k, f32_k>("sqeuclidean_e4m3_skylake", nk_sqeuclidean_e4m3_skylake);
    dense_<e4m3_k, f32_k>("euclidean_e4m3_skylake", nk_euclidean_e4m3_skylake);
    dense_<e5m2_k, f32_k>("angular_e5m2_skylake", nk_angular_e5m2_skylake);
    dense_<e5m2_k, f32_k>("sqeuclidean_e5m2_skylake", nk_sqeuclidean_e5m2_skylake);
    dense_<e5m2_k, f32_k>("euclidean_e5m2_skylake", nk_euclidean_e5m2_skylake);
    dense_<e2m3_k, f32_k>("angular_e2m3_skylake", nk_angular_e2m3_skylake);
    dense_<e2m3_k, f32_k>("sqeuclidean_e2m3_skylake", nk_sqeuclidean_e2m3_skylake);
    dense_<e2m3_k, f32_k>("euclidean_e2m3_skylake", nk_euclidean_e2m3_skylake);
    dense_<e3m2_k, f32_k>("angular_e3m2_skylake", nk_angular_e3m2_skylake);
    dense_<e3m2_k, f32_k>("sqeuclidean_e3m2_skylake", nk_sqeuclidean_e3m2_skylake);
    dense_<e3m2_k, f32_k>("euclidean_e3m2_skylake", nk_euclidean_e3m2_skylake);
#endif

#if NK_TARGET_ICELAKE
    dense_<i8_k, f32_k>("angular_i8_icelake", nk_angular_i8_icelake);
    dense_<i8_k, u32_k>("sqeuclidean_i8_icelake", nk_sqeuclidean_i8_icelake);
    dense_<i8_k, f32_k>("euclidean_i8_icelake", nk_euclidean_i8_icelake);
    dense_<u8_k, f32_k>("angular_u8_icelake", nk_angular_u8_icelake);
    dense_<u8_k, u32_k>("sqeuclidean_u8_icelake", nk_sqeuclidean_u8_icelake);
    dense_<u8_k, f32_k>("euclidean_u8_icelake", nk_euclidean_u8_icelake);
    dense_<i4_k, f32_k>("angular_i4_icelake", nk_angular_i4_icelake);
    dense_<i4_k, u32_k>("sqeuclidean_i4_icelake", nk_sqeuclidean_i4_icelake);
    dense_<i4_k, f32_k>("euclidean_i4_icelake", nk_euclidean_i4_icelake);
    dense_<u4_k, f32_k>("angular_u4_icelake", nk_angular_u4_icelake);
    dense_<u4_k, u32_k>("sqeuclidean_u4_icelake", nk_sqeuclidean_u4_icelake);
    dense_<u4_k, f32_k>("euclidean_u4_icelake", nk_euclidean_u4_icelake);
#endif

#if NK_TARGET_GENOA
    dense_<bf16_k, f32_k>("angular_bf16_genoa", nk_angular_bf16_genoa);
    dense_<bf16_k, f32_k>("sqeuclidean_bf16_genoa", nk_sqeuclidean_bf16_genoa);
    dense_<bf16_k, f32_k>("euclidean_bf16_genoa", nk_euclidean_bf16_genoa);
    dense_<e4m3_k, f32_k>("angular_e4m3_genoa", nk_angular_e4m3_genoa);
    dense_<e4m3_k, f32_k>("sqeuclidean_e4m3_genoa", nk_sqeuclidean_e4m3_genoa);
    dense_<e4m3_k, f32_k>("euclidean_e4m3_genoa", nk_euclidean_e4m3_genoa);
    dense_<e5m2_k, f32_k>("angular_e5m2_genoa", nk_angular_e5m2_genoa);
    dense_<e5m2_k, f32_k>("sqeuclidean_e5m2_genoa", nk_sqeuclidean_e5m2_genoa);
    dense_<e5m2_k, f32_k>("euclidean_e5m2_genoa", nk_euclidean_e5m2_genoa);
    dense_<e2m3_k, f32_k>("angular_e2m3_genoa", nk_angular_e2m3_genoa);
    dense_<e2m3_k, f32_k>("sqeuclidean_e2m3_genoa", nk_sqeuclidean_e2m3_genoa);
    dense_<e2m3_k, f32_k>("euclidean_e2m3_genoa", nk_euclidean_e2m3_genoa);
    dense_<e3m2_k, f32_k>("angular_e3m2_genoa", nk_angular_e3m2_genoa);
    dense_<e3m2_k, f32_k>("sqeuclidean_e3m2_genoa", nk_sqeuclidean_e3m2_genoa);
    dense_<e3m2_k, f32_k>("euclidean_e3m2_genoa", nk_euclidean_e3m2_genoa);
#endif

#if NK_TARGET_SAPPHIRE
    dense_<e4m3_k, f32_k>("euclidean_e4m3_sapphire", nk_euclidean_e4m3_sapphire);
    dense_<e4m3_k, f32_k>("sqeuclidean_e4m3_sapphire", nk_sqeuclidean_e4m3_sapphire);
#endif

#if NK_TARGET_RVV
    dense_<f32_k, f32_k>("sqeuclidean_f32_rvv", nk_sqeuclidean_f32_rvv);
    dense_<f64_k, f64_k>("sqeuclidean_f64_rvv", nk_sqeuclidean_f64_rvv);
    dense_<f32_k, f32_k>("angular_f32_rvv", nk_angular_f32_rvv);
    dense_<f64_k, f64_k>("angular_f64_rvv", nk_angular_f64_rvv);
#endif

#if NK_TARGET_V128RELAXED
    dense_<f32_k, f32_k>("sqeuclidean_f32_v128relaxed", nk_sqeuclidean_f32_v128relaxed);
    dense_<f64_k, f64_k>("sqeuclidean_f64_v128relaxed", nk_sqeuclidean_f64_v128relaxed);
    dense_<f16_k, f32_k>("sqeuclidean_f16_v128relaxed", nk_sqeuclidean_f16_v128relaxed);
    dense_<bf16_k, f32_k>("sqeuclidean_bf16_v128relaxed", nk_sqeuclidean_bf16_v128relaxed);
    dense_<f32_k, f32_k>("euclidean_f32_v128relaxed", nk_euclidean_f32_v128relaxed);
    dense_<f64_k, f64_k>("euclidean_f64_v128relaxed", nk_euclidean_f64_v128relaxed);
    dense_<f16_k, f32_k>("euclidean_f16_v128relaxed", nk_euclidean_f16_v128relaxed);
    dense_<bf16_k, f32_k>("euclidean_bf16_v128relaxed", nk_euclidean_bf16_v128relaxed);
    dense_<f32_k, f32_k>("angular_f32_v128relaxed", nk_angular_f32_v128relaxed);
    dense_<f64_k, f64_k>("angular_f64_v128relaxed", nk_angular_f64_v128relaxed);
    dense_<f16_k, f32_k>("angular_f16_v128relaxed", nk_angular_f16_v128relaxed);
    dense_<bf16_k, f32_k>("angular_bf16_v128relaxed", nk_angular_bf16_v128relaxed);
#endif

    // Serial fallbacks
    dense_<bf16_k, f32_k>("angular_bf16_serial", nk_angular_bf16_serial);
    dense_<bf16_k, f32_k>("sqeuclidean_bf16_serial", nk_sqeuclidean_bf16_serial);
    dense_<bf16_k, f32_k>("euclidean_bf16_serial", nk_euclidean_bf16_serial);
    dense_<e4m3_k, f32_k>("angular_e4m3_serial", nk_angular_e4m3_serial);
    dense_<e4m3_k, f32_k>("sqeuclidean_e4m3_serial", nk_sqeuclidean_e4m3_serial);
    dense_<e4m3_k, f32_k>("euclidean_e4m3_serial", nk_euclidean_e4m3_serial);
    dense_<e5m2_k, f32_k>("angular_e5m2_serial", nk_angular_e5m2_serial);
    dense_<e5m2_k, f32_k>("sqeuclidean_e5m2_serial", nk_sqeuclidean_e5m2_serial);
    dense_<e5m2_k, f32_k>("euclidean_e5m2_serial", nk_euclidean_e5m2_serial);
    dense_<e2m3_k, f32_k>("angular_e2m3_serial", nk_angular_e2m3_serial);
    dense_<e2m3_k, f32_k>("sqeuclidean_e2m3_serial", nk_sqeuclidean_e2m3_serial);
    dense_<e2m3_k, f32_k>("euclidean_e2m3_serial", nk_euclidean_e2m3_serial);
    dense_<e3m2_k, f32_k>("angular_e3m2_serial", nk_angular_e3m2_serial);
    dense_<e3m2_k, f32_k>("sqeuclidean_e3m2_serial", nk_sqeuclidean_e3m2_serial);
    dense_<e3m2_k, f32_k>("euclidean_e3m2_serial", nk_euclidean_e3m2_serial);
    dense_<f16_k, f32_k>("angular_f16_serial", nk_angular_f16_serial);
    dense_<f16_k, f32_k>("sqeuclidean_f16_serial", nk_sqeuclidean_f16_serial);
    dense_<f16_k, f32_k>("euclidean_f16_serial", nk_euclidean_f16_serial);
    dense_<f32_k, f32_k>("angular_f32_serial", nk_angular_f32_serial);
    dense_<f32_k, f32_k>("sqeuclidean_f32_serial", nk_sqeuclidean_f32_serial);
    dense_<f32_k, f32_k>("euclidean_f32_serial", nk_euclidean_f32_serial);
    dense_<f64_k, f64_k>("angular_f64_serial", nk_angular_f64_serial);
    dense_<f64_k, f64_k>("sqeuclidean_f64_serial", nk_sqeuclidean_f64_serial);
    dense_<f64_k, f64_k>("euclidean_f64_serial", nk_euclidean_f64_serial);
    dense_<i8_k, f32_k>("angular_i8_serial", nk_angular_i8_serial);
    dense_<i8_k, u32_k>("sqeuclidean_i8_serial", nk_sqeuclidean_i8_serial);
    dense_<i8_k, f32_k>("euclidean_i8_serial", nk_euclidean_i8_serial);
    dense_<u8_k, f32_k>("angular_u8_serial", nk_angular_u8_serial);
    dense_<u8_k, u32_k>("sqeuclidean_u8_serial", nk_sqeuclidean_u8_serial);
    dense_<u8_k, f32_k>("euclidean_u8_serial", nk_euclidean_u8_serial);
    dense_<i4_k, f32_k>("angular_i4_serial", nk_angular_i4_serial);
    dense_<i4_k, u32_k>("sqeuclidean_i4_serial", nk_sqeuclidean_i4_serial);
    dense_<i4_k, f32_k>("euclidean_i4_serial", nk_euclidean_i4_serial);
    dense_<u4_k, f32_k>("angular_u4_serial", nk_angular_u4_serial);
    dense_<u4_k, u32_k>("sqeuclidean_u4_serial", nk_sqeuclidean_u4_serial);
    dense_<u4_k, f32_k>("euclidean_u4_serial", nk_euclidean_u4_serial);
}

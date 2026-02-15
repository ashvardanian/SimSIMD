/**
 *  @brief KL-divergence and Jensen-Shannon divergence benchmarks.
 *  @file bench/bench_probability.cpp
 *  @author Ash Vardanian
 *  @date March 14, 2023
 */

#include "numkong/probability.h"

#include "bench.hpp"

void bench_probability() {
    constexpr nk_dtype_t f32_k = nk_f32_k;
    constexpr nk_dtype_t f16_k = nk_f16_k;
    constexpr nk_dtype_t bf16_k = nk_bf16_k;

#if NK_TARGET_NEON
    dense_<f32_k, f32_k>("kld_f32_neon", nk_kld_f32_neon);
    dense_<f32_k, f32_k>("jsd_f32_neon", nk_jsd_f32_neon);
#endif

#if NK_TARGET_NEONHALF
    dense_<f16_k, f32_k>("kld_f16_neonhalf", nk_kld_f16_neonhalf);
    dense_<f16_k, f32_k>("jsd_f16_neonhalf", nk_jsd_f16_neonhalf);
#endif

#if NK_TARGET_HASWELL
    dense_<f16_k, f32_k>("kld_f16_haswell", nk_kld_f16_haswell);
    dense_<f16_k, f32_k>("jsd_f16_haswell", nk_jsd_f16_haswell);
#endif

#if NK_TARGET_SKYLAKE
    dense_<f32_k, f32_k>("kld_f32_skylake", nk_kld_f32_skylake);
    dense_<f32_k, f32_k>("jsd_f32_skylake", nk_jsd_f32_skylake);
#endif

#if NK_TARGET_SAPPHIRE
    dense_<f16_k, f32_k>("kld_f16_sapphire", nk_kld_f16_sapphire);
    dense_<f16_k, f32_k>("jsd_f16_sapphire", nk_jsd_f16_sapphire);
#endif

    // Serial fallbacks
    dense_<bf16_k, f32_k>("kld_bf16_serial", nk_kld_bf16_serial);
    dense_<bf16_k, f32_k>("jsd_bf16_serial", nk_jsd_bf16_serial);
    dense_<f16_k, f32_k>("kld_f16_serial", nk_kld_f16_serial);
    dense_<f16_k, f32_k>("jsd_f16_serial", nk_jsd_f16_serial);
    dense_<f32_k, f32_k>("kld_f32_serial", nk_kld_f32_serial);
    dense_<f32_k, f32_k>("jsd_f32_serial", nk_jsd_f32_serial);
}

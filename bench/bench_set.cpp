/**
 *  @brief Binary set metric benchmarks (hamming, jaccard).
 *  @file bench/bench_set.cpp
 *  @author Ash Vardanian
 *  @date March 14, 2023
 */

#include "numkong/set.h"

#include "bench.hpp"

void bench_set() {
    constexpr nk_dtype_t u1_k = nk_u1_k;
    constexpr nk_dtype_t u8_k = nk_u8_k;
    constexpr nk_dtype_t u16_k = nk_u16_k;
    constexpr nk_dtype_t u32_k = nk_u32_k;
    constexpr nk_dtype_t f32_k = nk_f32_k;

#if NK_TARGET_NEON
    dense_<u1_k, u32_k>("hamming_u1_neon", nk_hamming_u1_neon);
    dense_<u1_k, f32_k>("jaccard_u1_neon", nk_jaccard_u1_neon);
#endif

#if NK_TARGET_SVE
    dense_<u1_k, u32_k>("hamming_u1_sve", nk_hamming_u1_sve);
    dense_<u1_k, f32_k>("jaccard_u1_sve", nk_jaccard_u1_sve);
#endif

#if NK_TARGET_HASWELL
    dense_<u1_k, u32_k>("hamming_u1_haswell", nk_hamming_u1_haswell);
    dense_<u1_k, f32_k>("jaccard_u1_haswell", nk_jaccard_u1_haswell);
#endif

#if NK_TARGET_ICELAKE
    dense_<u1_k, u32_k>("hamming_u1_icelake", nk_hamming_u1_icelake);
    dense_<u1_k, f32_k>("jaccard_u1_icelake", nk_jaccard_u1_icelake);
#endif

#if NK_TARGET_RVV
    dense_<u1_k, u32_k>("hamming_u1_rvv", nk_hamming_u1_rvv);
    dense_<u1_k, f32_k>("jaccard_u1_rvv", nk_jaccard_u1_rvv);
    dense_<u8_k, u32_k>("hamming_u8_rvv", nk_hamming_u8_rvv);
    dense_<u16_k, f32_k>("jaccard_u16_rvv", nk_jaccard_u16_rvv);
    dense_<u32_k, f32_k>("jaccard_u32_rvv", nk_jaccard_u32_rvv);
#endif

#if NK_TARGET_V128RELAXED
    dense_<u1_k, u32_k>("hamming_u1_v128relaxed", nk_hamming_u1_v128relaxed);
    dense_<u1_k, f32_k>("jaccard_u1_v128relaxed", nk_jaccard_u1_v128relaxed);
    dense_<u8_k, u32_k>("hamming_u8_v128relaxed", nk_hamming_u8_v128relaxed);
    dense_<u16_k, f32_k>("jaccard_u16_v128relaxed", nk_jaccard_u16_v128relaxed);
    dense_<u32_k, f32_k>("jaccard_u32_v128relaxed", nk_jaccard_u32_v128relaxed);
#endif

    // Serial fallbacks
    dense_<u1_k, u32_k>("hamming_u1_serial", nk_hamming_u1_serial);
    dense_<u1_k, f32_k>("jaccard_u1_serial", nk_jaccard_u1_serial);
}

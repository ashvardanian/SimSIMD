/**
 *  @brief SIMD-accelerated horizontal reduction operations for Intel Sapphire Rapids CPUs.
 *  @file include/numkong/reduce/sapphire.h
 *  @sa include/numkong/reduce.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 */
#ifndef NK_REDUCE_SAPPHIRE_H
#define NK_REDUCE_SAPPHIRE_H

#if NK_TARGET_X86_
#if NK_TARGET_SAPPHIRE
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "bmi2", "avx512bw", "avx512fp16")
#pragma clang attribute push(__attribute__((target("avx2,avx512f,avx512vl,bmi2,avx512bw,avx512fp16"))), \
                             apply_to = function)

#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

/** @brief Convert f32 scalar to f16 bit pattern using AVX512FP16. Returns i32 with f16 in lower 16 bits. */
NK_INTERNAL nk_i32_t nk_f32_to_f16_as_i16_sapphire_(nk_f32_t val) {
    return _mm_cvtsi128_si32(_mm_castph_si128(_mm_cvtss_sh(_mm_setzero_ph(), _mm_set_ss(val))));
}

#if defined(__cplusplus)
} // extern "C"
#endif

#pragma clang attribute pop
#pragma GCC pop_options
#endif // NK_TARGET_SAPPHIRE
#endif // NK_TARGET_X86_

#endif // NK_REDUCE_SAPPHIRE_H
/**
 *  @brief SIMD-accelerated Set Similarity Measures for LoongArch LASX (256-bit).
 *  @file include/numkong/set/loongsonasx.h
 *  @author Ash Vardanian
 *  @date March 23, 2026
 *
 *  @sa include/numkong/set.h
 *
 *  @section set_loongsonasx_instructions Key LASX Set Instructions
 *
 *      Intrinsic                    Instruction
 *      __lasx_xvld                  XVLD (256-bit unaligned load)
 *      __lasx_xvxor_v               XVXOR.V (bitwise XOR)
 *      __lasx_xvor_v                XVOR.V (bitwise OR)
 *      __lasx_xvand_v               XVAND.V (bitwise AND)
 *      __lasx_xvpcnt_d              XVPCNT.D (popcount per u64 element)
 *      __lasx_xvseq_b               XVSEQ.B (byte-wise equality, 0xFF/0x00)
 *      __lasx_xvmin_bu              XVMIN.BU (unsigned byte minimum)
 *      __lasx_xvhaddw_hu_bu         XVHADDW.HU.BU (horizontal pairwise add u8->u16)
 *      __lasx_xvhaddw_wu_hu         XVHADDW.WU.HU (horizontal pairwise add u16->u32)
 *      __lasx_xvhaddw_du_wu         XVHADDW.DU.WU (horizontal pairwise add u32->u64)
 *      __lasx_xvadd_d               XVADD.D (i64 addition)
 *      __lasx_xvpermi_q             XVPERMI.Q (extract/permute 128-bit lanes)
 *
 *  LASX provides per-element popcount at multiple widths (`xvpcnt_b/h/w/d`).
 *  For binary set operations we use `xvpcnt_d` which gives 4 x u64 popcount values
 *  directly, eliminating the need for horizontal byte-sum reduction chains.
 *
 *  For sorted integer set operations (jaccard_u16, jaccard_u32), SIMD provides limited
 *  benefit due to the inherently serial merge-based algorithm, so we delegate to the
 *  serial implementations.
 */
#ifndef NK_SET_LOONGSONASX_H
#define NK_SET_LOONGSONASX_H

#if NK_TARGET_LOONGARCH_
#if NK_TARGET_LOONGSONASX

#include "numkong/types.h"
#include "numkong/set/serial.h"      // `nk_u1x8_popcount_`, serial fallbacks
#include "numkong/dot/loongsonasx.h" // `nk_reduce_add_i32x8_loongsonasx_`

#if defined(__cplusplus)
extern "C" {
#endif

#pragma region - Reduction Helpers

/** @brief Horizontal sum of 4 u64 lanes in a 256-bit LASX register. */
NK_INTERNAL nk_u64_t nk_reduce_add_u64x4_loongsonasx_(__m256i sum_u64x4) {
    __m256i hi_u64x4 = __lasx_xvpermi_q(sum_u64x4, sum_u64x4, 0x11);
    __m256i sum_u64x2 = __lasx_xvadd_d(sum_u64x4, hi_u64x4);
    nk_b256_vec_t vec;
    vec.ymm = sum_u64x2;
    return vec.u64s[0] + vec.u64s[1];
}

/** @brief Horizontally sum all bytes in a 256-bit register as unsigned values.
 *
 *  Chains pairwise widening additions: u8->u16->u32->u64, then reduces 4 u64 lanes.
 */
NK_INTERNAL nk_u64_t nk_reduce_add_u8x32_loongsonasx_(__m256i v_u8x32) {
    __m256i sum_u16x16 = __lasx_xvhaddw_hu_bu(v_u8x32, v_u8x32);
    __m256i sum_u32x8 = __lasx_xvhaddw_wu_hu(sum_u16x16, sum_u16x16);
    __m256i sum_u64x4 = __lasx_xvhaddw_du_wu(sum_u32x8, sum_u32x8);
    return nk_reduce_add_u64x4_loongsonasx_(sum_u64x4);
}

#pragma endregion - Reduction Helpers

#pragma region - Binary Sets

NK_PUBLIC void nk_hamming_u1_loongsonasx(nk_u1x8_t const *a, nk_u1x8_t const *b, nk_size_t n, nk_u32_t *result) {
    nk_size_t n_bytes = nk_size_divide_round_up_(n, NK_BITS_PER_BYTE);
    __m256i count_u64x4 = __lasx_xvreplgr2vr_d(0);
    nk_size_t i = 0;

    for (; i + 32 <= n_bytes; i += 32) {
        __m256i a_u8x32 = __lasx_xvld(a + i, 0);
        __m256i b_u8x32 = __lasx_xvld(b + i, 0);
        __m256i xor_u8x32 = __lasx_xvxor_v(a_u8x32, b_u8x32);
        count_u64x4 = __lasx_xvadd_d(count_u64x4, __lasx_xvpcnt_d(xor_u8x32));
    }

    nk_b256_vec_t vec;
    vec.ymm = count_u64x4;
    nk_u64_t count = vec.u64s[0] + vec.u64s[1] + vec.u64s[2] + vec.u64s[3];

    for (; i < n_bytes; ++i) count += nk_u1x8_popcount_(a[i] ^ b[i]);
    *result = (nk_u32_t)count;
}

NK_PUBLIC void nk_jaccard_u1_loongsonasx(nk_u1x8_t const *a, nk_u1x8_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_size_t n_bytes = nk_size_divide_round_up_(n, NK_BITS_PER_BYTE);
    __m256i xor_count_u64x4 = __lasx_xvreplgr2vr_d(0);
    __m256i or_count_u64x4 = __lasx_xvreplgr2vr_d(0);
    nk_size_t i = 0;

    for (; i + 32 <= n_bytes; i += 32) {
        __m256i a_u8x32 = __lasx_xvld(a + i, 0);
        __m256i b_u8x32 = __lasx_xvld(b + i, 0);
        __m256i xor_u8x32 = __lasx_xvxor_v(a_u8x32, b_u8x32);
        __m256i or_u8x32 = __lasx_xvor_v(a_u8x32, b_u8x32);
        xor_count_u64x4 = __lasx_xvadd_d(xor_count_u64x4, __lasx_xvpcnt_d(xor_u8x32));
        or_count_u64x4 = __lasx_xvadd_d(or_count_u64x4, __lasx_xvpcnt_d(or_u8x32));
    }

    nk_b256_vec_t xor_vec, or_vec;
    xor_vec.ymm = xor_count_u64x4;
    or_vec.ymm = or_count_u64x4;
    nk_u64_t xor_count = xor_vec.u64s[0] + xor_vec.u64s[1] + xor_vec.u64s[2] + xor_vec.u64s[3];
    nk_u64_t or_count = or_vec.u64s[0] + or_vec.u64s[1] + or_vec.u64s[2] + or_vec.u64s[3];

    for (; i < n_bytes; ++i) {
        xor_count += nk_u1x8_popcount_(a[i] ^ b[i]);
        or_count += nk_u1x8_popcount_(a[i] | b[i]);
    }
    *result = (or_count != 0) ? (nk_f32_t)xor_count / (nk_f32_t)or_count : 0.0f;
}

#pragma endregion - Binary Sets

#pragma region - Integer Sets

NK_PUBLIC void nk_hamming_u8_loongsonasx(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u32_t *result) {
    __m256i count_u64x4 = __lasx_xvreplgr2vr_d(0);
    __m256i ones_u8x32 = __lasx_xvreplgr2vr_b(1);
    nk_size_t i = 0;

    for (; i + 32 <= n; i += 32) {
        __m256i a_u8x32 = __lasx_xvld(a + i, 0);
        __m256i b_u8x32 = __lasx_xvld(b + i, 0);
        __m256i xor_u8x32 = __lasx_xvxor_v(a_u8x32, b_u8x32);
        __m256i min_u8x32 = __lasx_xvmin_bu(xor_u8x32, ones_u8x32);
        __m256i sum_u16x16 = __lasx_xvhaddw_hu_bu(min_u8x32, min_u8x32);
        __m256i sum_u32x8 = __lasx_xvhaddw_wu_hu(sum_u16x16, sum_u16x16);
        __m256i sum_u64x4 = __lasx_xvhaddw_du_wu(sum_u32x8, sum_u32x8);
        count_u64x4 = __lasx_xvadd_d(count_u64x4, sum_u64x4);
    }

    nk_b256_vec_t vec;
    vec.ymm = count_u64x4;
    nk_u64_t count = vec.u64s[0] + vec.u64s[1] + vec.u64s[2] + vec.u64s[3];

    for (; i < n; ++i) count += (a[i] != b[i]);
    *result = (nk_u32_t)count;
}

#pragma endregion - Integer Sets

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_LOONGSONASX
#endif // NK_TARGET_LOONGARCH_
#endif // NK_SET_LOONGSONASX_H

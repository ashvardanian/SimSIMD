/**
 *  @brief NEON-accelerated Sparse Vector Operations.
 *  @file include/numkong/sparse/neon.h
 *  @author Ash Vardanian
 *  @date February 6, 2026
 *
 *  @sa include/numkong/sparse.h
 */
#ifndef NK_SPARSE_NEON_H
#define NK_SPARSE_NEON_H

#if NK_TARGET_ARM_
#if NK_TARGET_NEON

#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8-a"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8-a")
#endif

/**
 *  @brief  Uses `vshrn` to produce a bitmask, similar to `movemask` in SSE.
 *  https://community.arm.com/arm-community-blogs/b/infrastructure-solutions-blog/posts/porting-x86-vector-bitmask-optimizations-to-arm-neon
 */
NK_INTERNAL nk_u64_t nk_u8_to_u4_neon_(uint8x16_t vec) {
    return vget_lane_u64(vreinterpret_u64_u8(vshrn_n_u16(vreinterpretq_u16_u8(vec), 4)), 0);
}

NK_INTERNAL int nk_clz_u64_(nk_u64_t x) {
// On GCC and Clang use the builtin, otherwise use the generic implementation
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_clzll(x);
#else
    int n = 0;
    while ((x & 0x8000000000000000ull) == 0) n++, x <<= 1;
    return n;
#endif
}

NK_INTERNAL uint32x4_t nk_intersect_u32x4_neon_(uint32x4_t a, uint32x4_t b) {
    uint32x4_t b1 = vextq_u32(b, b, 1);
    uint32x4_t b2 = vextq_u32(b, b, 2);
    uint32x4_t b3 = vextq_u32(b, b, 3);
    uint32x4_t nm00 = vceqq_u32(a, b);
    uint32x4_t nm01 = vceqq_u32(a, b1);
    uint32x4_t nm02 = vceqq_u32(a, b2);
    uint32x4_t nm03 = vceqq_u32(a, b3);
    uint32x4_t nm = vorrq_u32(vorrq_u32(nm00, nm01), vorrq_u32(nm02, nm03));
    return nm;
}

NK_INTERNAL uint16x8_t nk_intersect_u16x8_neon_(uint16x8_t a, uint16x8_t b) {
    uint16x8_t b1 = vextq_u16(b, b, 1);
    uint16x8_t b2 = vextq_u16(b, b, 2);
    uint16x8_t b3 = vextq_u16(b, b, 3);
    uint16x8_t b4 = vextq_u16(b, b, 4);
    uint16x8_t b5 = vextq_u16(b, b, 5);
    uint16x8_t b6 = vextq_u16(b, b, 6);
    uint16x8_t b7 = vextq_u16(b, b, 7);
    uint16x8_t nm00 = vceqq_u16(a, b);
    uint16x8_t nm01 = vceqq_u16(a, b1);
    uint16x8_t nm02 = vceqq_u16(a, b2);
    uint16x8_t nm03 = vceqq_u16(a, b3);
    uint16x8_t nm04 = vceqq_u16(a, b4);
    uint16x8_t nm05 = vceqq_u16(a, b5);
    uint16x8_t nm06 = vceqq_u16(a, b6);
    uint16x8_t nm07 = vceqq_u16(a, b7);
    uint16x8_t nm = vorrq_u16(vorrq_u16(vorrq_u16(nm00, nm01), vorrq_u16(nm02, nm03)),
                              vorrq_u16(vorrq_u16(nm04, nm05), vorrq_u16(nm06, nm07)));
    return nm;
}

NK_PUBLIC void nk_sparse_intersect_u16_neon( //
    nk_u16_t const *a, nk_u16_t const *b,    //
    nk_size_t a_length, nk_size_t b_length,  //
    nk_u16_t *result, nk_size_t *count) {

    // NEON lacks compress-store, so fall back to serial for result output
    if (result) {
        nk_sparse_intersect_u16_serial(a, b, a_length, b_length, result, count);
        return;
    }

    // The baseline implementation for very small arrays (2 registers or less) can be quite simple:
    if (a_length < 32 && b_length < 32) {
        nk_sparse_intersect_u16_serial(a, b, a_length, b_length, result, count);
        return;
    }

    nk_u16_t const *const a_end = a + a_length;
    nk_u16_t const *const b_end = b + b_length;
    nk_b128_vec_t a_vec, b_vec;
    uint16x8_t c_counts_u16x8 = vdupq_n_u16(0);

    while (a + 8 <= a_end && b + 8 <= b_end) {
        a_vec.u16x8 = vld1q_u16(a);
        b_vec.u16x8 = vld1q_u16(b);

        // Intersecting registers with `nk_intersect_u16x8_neon_` involves a lot of shuffling
        // and comparisons, so we want to avoid it if the slices don't overlap at all..
        nk_u16_t a_min;
        nk_u16_t a_max = a_vec.u16s[7];
        nk_u16_t b_min = b_vec.u16s[0];
        nk_u16_t b_max = b_vec.u16s[7];

        // If the slices don't overlap, advance the appropriate pointer
        while (a_max < b_min && a + 16 <= a_end) {
            a += 8;
            a_vec.u16x8 = vld1q_u16(a);
            a_max = a_vec.u16s[7];
        }
        a_min = a_vec.u16s[0];
        while (b_max < a_min && b + 16 <= b_end) {
            b += 8;
            b_vec.u16x8 = vld1q_u16(b);
            b_max = b_vec.u16s[7];
        }
        b_min = b_vec.u16s[0];

        // Now we are likely to have some overlap, so we can intersect the registers.
        // We can do it by performing a population count at every cycle, but it's not the cheapest in terms of cycles.
        //
        //      nk_u64_t a_matches = __builtin_popcountll(
        //          nk_u8_to_u4_neon_(vreinterpretq_u8_u16(
        //              nk_intersect_u16x8_neon_(a_vec.u16x8, b_vec.u16x8))));
        //      c += a_matches / 8;
        //
        // Alternatively, we can we can transform match-masks into "ones", accumulate them between the cycles,
        // and merge all together in the end.
        uint16x8_t a_matches = nk_intersect_u16x8_neon_(a_vec.u16x8, b_vec.u16x8);
        c_counts_u16x8 = vaddq_u16(c_counts_u16x8, vandq_u16(a_matches, vdupq_n_u16(1)));

        // Counting leading zeros is tricky. On Arm we can use inline Assembly to get the result,
        // but MSVC doesn't support that:
        //
        //      NK_INTERNAL int nk_clz_u64_(nk_u64_t value) {
        //          nk_u64_t result;
        //          __asm__("clz %x0, %x1" : "=r"(result) : "r"(value));
        //          return (int)result;
        //      }
        //
        // Alternatively, we can use the `vclz_u32` NEON intrinsic.
        // It will compute the leading zeros number for both `a_step` and `b_step` in parallel.
        uint16x8_t a_max_u16x8 = vdupq_n_u16(a_max);
        uint16x8_t b_max_u16x8 = vdupq_n_u16(b_max);
        nk_u64_t a_step = nk_clz_u64_(nk_u8_to_u4_neon_( //
            vreinterpretq_u8_u16(vcleq_u16(a_vec.u16x8, b_max_u16x8))));
        nk_u64_t b_step = nk_clz_u64_(nk_u8_to_u4_neon_( //
            vreinterpretq_u8_u16(vcleq_u16(b_vec.u16x8, a_max_u16x8))));
        a += (64 - a_step) / 8;
        b += (64 - b_step) / 8;
    }

    nk_size_t tail_count = 0;
    nk_sparse_intersect_u16_serial(a, b, a_end - a, b_end - b, 0, &tail_count);
    *count = tail_count + (nk_size_t)vaddvq_u16(c_counts_u16x8);
}

NK_PUBLIC void nk_sparse_intersect_u32_neon( //
    nk_u32_t const *a, nk_u32_t const *b,    //
    nk_size_t a_length, nk_size_t b_length,  //
    nk_u32_t *result, nk_size_t *count) {

    // NEON lacks compress-store, so fall back to serial for result output
    if (result) {
        nk_sparse_intersect_u32_serial(a, b, a_length, b_length, result, count);
        return;
    }

    // The baseline implementation for very small arrays (2 registers or less) can be quite simple:
    if (a_length < 32 && b_length < 32) {
        nk_sparse_intersect_u32_serial(a, b, a_length, b_length, result, count);
        return;
    }

    nk_u32_t const *const a_end = a + a_length;
    nk_u32_t const *const b_end = b + b_length;
    nk_b128_vec_t a_vec, b_vec;
    uint32x4_t c_counts_u32x4 = vdupq_n_u32(0);

    while (a + 4 <= a_end && b + 4 <= b_end) {
        a_vec.u32x4 = vld1q_u32(a);
        b_vec.u32x4 = vld1q_u32(b);

        // Intersecting registers with `nk_intersect_u32x4_neon_` involves a lot of shuffling
        // and comparisons, so we want to avoid it if the slices don't overlap at all..
        nk_u32_t a_min;
        nk_u32_t a_max = a_vec.u32s[3];
        nk_u32_t b_min = b_vec.u32s[0];
        nk_u32_t b_max = b_vec.u32s[3];

        // If the slices don't overlap, advance the appropriate pointer
        while (a_max < b_min && a + 8 <= a_end) {
            a += 4;
            a_vec.u32x4 = vld1q_u32(a);
            a_max = a_vec.u32s[3];
        }
        a_min = a_vec.u32s[0];
        while (b_max < a_min && b + 8 <= b_end) {
            b += 4;
            b_vec.u32x4 = vld1q_u32(b);
            b_max = b_vec.u32s[3];
        }
        b_min = b_vec.u32s[0];

        // Now we are likely to have some overlap, so we can intersect the registers
        // We can do it by performing a population count at every cycle, but it's not the cheapest in terms of cycles.
        //
        //     nk_u64_t a_matches = __builtin_popcountll(
        //         nk_u8_to_u4_neon_(vreinterpretq_u8_u32(
        //             nk_intersect_u32x4_neon_(a_vec.u32x4, b_vec.u32x4))));
        //     c += a_matches / 16;
        //
        // Alternatively, we can we can transform match-masks into "ones", accumulate them between the cycles,
        // and merge all together in the end.
        uint32x4_t a_matches = nk_intersect_u32x4_neon_(a_vec.u32x4, b_vec.u32x4);
        c_counts_u32x4 = vaddq_u32(c_counts_u32x4, vandq_u32(a_matches, vdupq_n_u32(1)));

        uint32x4_t a_max_u32x4 = vdupq_n_u32(a_max);
        uint32x4_t b_max_u32x4 = vdupq_n_u32(b_max);
        nk_u64_t a_step = nk_clz_u64_(nk_u8_to_u4_neon_( //
            vreinterpretq_u8_u32(vcleq_u32(a_vec.u32x4, b_max_u32x4))));
        nk_u64_t b_step = nk_clz_u64_(nk_u8_to_u4_neon_( //
            vreinterpretq_u8_u32(vcleq_u32(b_vec.u32x4, a_max_u32x4))));
        a += (64 - a_step) / 16;
        b += (64 - b_step) / 16;
    }

    nk_size_t tail_count = 0;
    nk_sparse_intersect_u32_serial(a, b, a_end - a, b_end - b, 0, &tail_count);
    *count = tail_count + (nk_size_t)vaddvq_u32(c_counts_u32x4);
}

NK_INTERNAL uint64x2_t nk_intersect_u64x2_neon_(uint64x2_t a, uint64x2_t b) {
    uint64x2_t b1 = vextq_u64(b, b, 1);
    uint64x2_t nm00 = vceqq_u64(a, b);
    uint64x2_t nm01 = vceqq_u64(a, b1);
    uint64x2_t nm = vorrq_u64(nm00, nm01);
    return nm;
}

NK_PUBLIC void nk_sparse_intersect_u64_neon( //
    nk_u64_t const *a, nk_u64_t const *b,    //
    nk_size_t a_length, nk_size_t b_length,  //
    nk_u64_t *result, nk_size_t *count) {

    // NEON lacks compress-store, so fall back to serial for result output
    if (result) {
        nk_sparse_intersect_u64_serial(a, b, a_length, b_length, result, count);
        return;
    }

    // The baseline implementation for very small arrays (2 registers or less) can be quite simple:
    if (a_length < 8 && b_length < 8) {
        nk_sparse_intersect_u64_serial(a, b, a_length, b_length, result, count);
        return;
    }

    nk_u64_t const *const a_end = a + a_length;
    nk_u64_t const *const b_end = b + b_length;
    nk_b128_vec_t a_vec, b_vec;
    uint64x2_t c_counts_u64x2 = vdupq_n_u64(0);

    while (a + 2 <= a_end && b + 2 <= b_end) {
        a_vec.u64x2 = vld1q_u64(a);
        b_vec.u64x2 = vld1q_u64(b);

        // Intersecting registers with `nk_intersect_u64x2_neon_` involves comparisons,
        // so we want to avoid it if the slices don't overlap at all.
        nk_u64_t a_min;
        nk_u64_t a_max = a_vec.u64s[1];
        nk_u64_t b_min = b_vec.u64s[0];
        nk_u64_t b_max = b_vec.u64s[1];

        // If the slices don't overlap, advance the appropriate pointer
        while (a_max < b_min && a + 4 <= a_end) {
            a += 2;
            a_vec.u64x2 = vld1q_u64(a);
            a_max = a_vec.u64s[1];
        }
        a_min = a_vec.u64s[0];
        while (b_max < a_min && b + 4 <= b_end) {
            b += 2;
            b_vec.u64x2 = vld1q_u64(b);
            b_max = b_vec.u64s[1];
        }
        b_min = b_vec.u64s[0];

        // Now we are likely to have some overlap, so we can intersect the registers
        // Transform match-masks into "ones", accumulate them between the cycles,
        // and merge all together in the end.
        uint64x2_t a_matches = nk_intersect_u64x2_neon_(a_vec.u64x2, b_vec.u64x2);
        c_counts_u64x2 = vaddq_u64(c_counts_u64x2, vandq_u64(a_matches, vdupq_n_u64(1)));

        uint64x2_t a_max_u64x2 = vdupq_n_u64(a_max);
        uint64x2_t b_max_u64x2 = vdupq_n_u64(b_max);
        nk_u64_t a_step = nk_clz_u64_(nk_u8_to_u4_neon_( //
            vreinterpretq_u8_u64(vcleq_u64(a_vec.u64x2, b_max_u64x2))));
        nk_u64_t b_step = nk_clz_u64_(nk_u8_to_u4_neon_( //
            vreinterpretq_u8_u64(vcleq_u64(b_vec.u64x2, a_max_u64x2))));
        a += (64 - a_step) / 32;
        b += (64 - b_step) / 32;
    }

    nk_size_t tail_count = 0;
    nk_sparse_intersect_u64_serial(a, b, a_end - a, b_end - b, 0, &tail_count);
    *count = tail_count + (nk_size_t)vaddvq_u64(c_counts_u64x2);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_NEON
#endif // NK_TARGET_ARM_
#endif // NK_SPARSE_NEON_H

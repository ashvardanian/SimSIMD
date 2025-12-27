/**
 *  @brief SIMD-accelerated Spatial Similarity Measures optimized for Intel Ice Lake CPUs.
 *  @file include/numkong/spatial/ice.h
 *  @sa include/numkong/spatial.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 */
#ifndef NK_SPATIAL_ICE_H
#define NK_SPATIAL_ICE_H

#if _NK_TARGET_X86
#if NK_TARGET_ICE
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "bmi2", "avx512bw", "avx512vnni")
#pragma clang attribute push(__attribute__((target("avx2,avx512f,avx512vl,bmi2,avx512bw,avx512vnni"))), \
                             apply_to = function)

#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

NK_PUBLIC void nk_l2_i8_ice(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_u32_t d2;
    nk_l2sq_i8_ice(a, b, n, &d2);
    *result = _nk_sqrt_f32_haswell((nk_f32_t)d2);
}
NK_PUBLIC void nk_l2sq_i8_ice(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_u32_t *result) {
    __m512i distance_sq_i32x16 = _mm512_setzero_si512();
    __m512i a_i16x32, b_i16x32, diff_i16x32;

nk_l2sq_i8_ice_cycle:
    if (n < 32) { // TODO: Avoid early i16 upcast to step through 64 values at a time
        __mmask32 mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, n);
        a_i16x32 = _mm512_cvtepi8_epi16(_mm256_maskz_loadu_epi8(mask, a));
        b_i16x32 = _mm512_cvtepi8_epi16(_mm256_maskz_loadu_epi8(mask, b));
        n = 0;
    }
    else {
        a_i16x32 = _mm512_cvtepi8_epi16(_mm256_lddqu_si256((__m256i const *)a));
        b_i16x32 = _mm512_cvtepi8_epi16(_mm256_lddqu_si256((__m256i const *)b));
        a += 32, b += 32, n -= 32;
    }
    diff_i16x32 = _mm512_sub_epi16(a_i16x32, b_i16x32);
    distance_sq_i32x16 = _mm512_dpwssd_epi32(distance_sq_i32x16, diff_i16x32, diff_i16x32);
    if (n) goto nk_l2sq_i8_ice_cycle;

    *result = _mm512_reduce_add_epi32(distance_sq_i32x16);
}

NK_PUBLIC void nk_angular_i8_ice(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_f32_t *result) {

    __m512i dot_product_i32x16 = _mm512_setzero_si512();
    __m512i a_norm_sq_i32x16 = _mm512_setzero_si512();
    __m512i b_norm_sq_i32x16 = _mm512_setzero_si512();
    __m512i a_i16x32, b_i16x32;
nk_angular_i8_ice_cycle:
    if (n < 32) {
        __mmask32 mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, n);
        a_i16x32 = _mm512_cvtepi8_epi16(_mm256_maskz_loadu_epi8(mask, a));
        b_i16x32 = _mm512_cvtepi8_epi16(_mm256_maskz_loadu_epi8(mask, b));
        n = 0;
    }
    else {
        a_i16x32 = _mm512_cvtepi8_epi16(_mm256_lddqu_si256((__m256i const *)a));
        b_i16x32 = _mm512_cvtepi8_epi16(_mm256_lddqu_si256((__m256i const *)b));
        a += 32, b += 32, n -= 32;
    }

    // We can't directly use the `_mm512_dpbusd_epi32` intrinsic everywhere,
    // as it's asymmetric with respect to the sign of the input arguments:
    //
    //      Signed(ZeroExtend16(a.byte[4*j]) * SignExtend16(b.byte[4*j]))
    //
    // To compute the squares, we could just drop the sign bit of the second argument.
    // But this would lead to big-big problems on values like `-128`!
    // For dot-products we don't have the luxury of optimizing the sign bit away.
    // Assuming this is an approximate kernel (with reciprocal square root approximations)
    // in the end, we can allow clamping the value to [-127, 127] range.
    //
    // On Ice Lake:
    //
    //  1. `VPDPBUSDS (ZMM, ZMM, ZMM)` can only execute on port 0, with 5 cycle latency.
    //  2. `VPDPWSSDS (ZMM, ZMM, ZMM)` can also only execute on port 0, with 5 cycle latency.
    //  3. `VPMADDWD (ZMM, ZMM, ZMM)` can execute on ports 0 and 5, with 5 cycle latency.
    //
    // On Zen4 Genoa:
    //
    //  1. `VPDPBUSDS (ZMM, ZMM, ZMM)` can execute on ports 0 and 1, with 4 cycle latency.
    //  2. `VPDPWSSDS (ZMM, ZMM, ZMM)` can also execute on ports 0 and 1, with 4 cycle latency.
    //  3. `VPMADDWD (ZMM, ZMM, ZMM)` can execute on ports 0 and 1, with 3 cycle latency.
    //
    // The old solution was complex replied on 1. and 2.:
    //
    //    a_i8_abs_vec = _mm512_abs_epi8(a_i8_vec);
    //    b_i8_abs_vec = _mm512_abs_epi8(b_i8_vec);
    //    a2_i32_vec = _mm512_dpbusds_epi32(a2_i32_vec, a_i8_abs_vec, a_i8_abs_vec);
    //    b2_i32_vec = _mm512_dpbusds_epi32(b2_i32_vec, b_i8_abs_vec, b_i8_abs_vec);
    //    ab_i32_low_vec = _mm512_dpwssds_epi32(                      //
    //        ab_i32_low_vec,                                         //
    //        _mm512_cvtepi8_epi16(_mm512_castsi512_si256(a_i8_vec)), //
    //        _mm512_cvtepi8_epi16(_mm512_castsi512_si256(b_i8_vec)));
    //    ab_i32_high_vec = _mm512_dpwssds_epi32(                           //
    //        ab_i32_high_vec,                                              //
    //        _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(a_i8_vec, 1)), //
    //        _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(b_i8_vec, 1)));
    //
    // The new solution is simpler and relies on 3.:
    dot_product_i32x16 = _mm512_add_epi32(dot_product_i32x16, _mm512_madd_epi16(a_i16x32, b_i16x32));
    a_norm_sq_i32x16 = _mm512_add_epi32(a_norm_sq_i32x16, _mm512_madd_epi16(a_i16x32, a_i16x32));
    b_norm_sq_i32x16 = _mm512_add_epi32(b_norm_sq_i32x16, _mm512_madd_epi16(b_i16x32, b_i16x32));
    if (n) goto nk_angular_i8_ice_cycle;

    nk_i32_t dot_product_i32 = _mm512_reduce_add_epi32(dot_product_i32x16);
    nk_i32_t a_norm_sq_i32 = _mm512_reduce_add_epi32(a_norm_sq_i32x16);
    nk_i32_t b_norm_sq_i32 = _mm512_reduce_add_epi32(b_norm_sq_i32x16);
    *result = _nk_angular_normalize_f32_haswell(dot_product_i32, a_norm_sq_i32, b_norm_sq_i32);
}
NK_PUBLIC void nk_l2_u8_ice(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_u32_t d2;
    nk_l2sq_u8_ice(a, b, n, &d2);
    *result = _nk_sqrt_f32_haswell((nk_f32_t)d2);
}
NK_PUBLIC void nk_l2sq_u8_ice(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u32_t *result) {
    __m512i distance_sq_low_i32x16 = _mm512_setzero_si512();
    __m512i distance_sq_high_i32x16 = _mm512_setzero_si512();
    __m512i const zeros_i8x64 = _mm512_setzero_si512();
    __m512i diff_low_i16x32, diff_high_i16x32;
    __m512i a_u8x64, b_u8x64, diff_u8x64;

nk_l2sq_u8_ice_cycle:
    if (n < 64) {
        __mmask64 mask = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFF, n);
        a_u8x64 = _mm512_maskz_loadu_epi8(mask, a);
        b_u8x64 = _mm512_maskz_loadu_epi8(mask, b);
        n = 0;
    }
    else {
        a_u8x64 = _mm512_loadu_si512(a);
        b_u8x64 = _mm512_loadu_si512(b);
        a += 64, b += 64, n -= 64;
    }

    // Substracting unsigned vectors in AVX-512 is done by saturating subtraction:
    diff_u8x64 = _mm512_or_si512(_mm512_subs_epu8(a_u8x64, b_u8x64), _mm512_subs_epu8(b_u8x64, a_u8x64));
    diff_low_i16x32 = _mm512_unpacklo_epi8(diff_u8x64, zeros_i8x64);
    diff_high_i16x32 = _mm512_unpackhi_epi8(diff_u8x64, zeros_i8x64);

    // Multiply and accumulate at `int16` level, accumulate at `int32` level:
    distance_sq_low_i32x16 = _mm512_dpwssd_epi32(distance_sq_low_i32x16, diff_low_i16x32, diff_low_i16x32);
    distance_sq_high_i32x16 = _mm512_dpwssd_epi32(distance_sq_high_i32x16, diff_high_i16x32, diff_high_i16x32);
    if (n) goto nk_l2sq_u8_ice_cycle;

    *result = _mm512_reduce_add_epi32(_mm512_add_epi32(distance_sq_low_i32x16, distance_sq_high_i32x16));
}

NK_PUBLIC void nk_angular_u8_ice(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_f32_t *result) {

    __m512i dot_product_low_i32x16 = _mm512_setzero_si512();
    __m512i dot_product_high_i32x16 = _mm512_setzero_si512();
    __m512i a_norm_sq_low_i32x16 = _mm512_setzero_si512();
    __m512i a_norm_sq_high_i32x16 = _mm512_setzero_si512();
    __m512i b_norm_sq_low_i32x16 = _mm512_setzero_si512();
    __m512i b_norm_sq_high_i32x16 = _mm512_setzero_si512();
    __m512i const zeros_i8x64 = _mm512_setzero_si512();
    __m512i a_low_i16x32, a_high_i16x32, b_low_i16x32, b_high_i16x32;
    __m512i a_u8x64, b_u8x64;

nk_angular_u8_ice_cycle:
    if (n < 64) {
        __mmask64 mask = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFF, n);
        a_u8x64 = _mm512_maskz_loadu_epi8(mask, a);
        b_u8x64 = _mm512_maskz_loadu_epi8(mask, b);
        n = 0;
    }
    else {
        a_u8x64 = _mm512_loadu_si512(a);
        b_u8x64 = _mm512_loadu_si512(b);
        a += 64, b += 64, n -= 64;
    }

    // Upcast `uint8` to `int16`. Unlike the signed version, we can use the unpacking
    // instructions instead of extracts, as they are much faster and more efficient.
    a_low_i16x32 = _mm512_unpacklo_epi8(a_u8x64, zeros_i8x64);
    a_high_i16x32 = _mm512_unpackhi_epi8(a_u8x64, zeros_i8x64);
    b_low_i16x32 = _mm512_unpacklo_epi8(b_u8x64, zeros_i8x64);
    b_high_i16x32 = _mm512_unpackhi_epi8(b_u8x64, zeros_i8x64);

    // Multiply and accumulate as `int16`, accumulate products as `int32`:
    dot_product_low_i32x16 = _mm512_dpwssds_epi32(dot_product_low_i32x16, a_low_i16x32, b_low_i16x32);
    dot_product_high_i32x16 = _mm512_dpwssds_epi32(dot_product_high_i32x16, a_high_i16x32, b_high_i16x32);
    a_norm_sq_low_i32x16 = _mm512_dpwssds_epi32(a_norm_sq_low_i32x16, a_low_i16x32, a_low_i16x32);
    a_norm_sq_high_i32x16 = _mm512_dpwssds_epi32(a_norm_sq_high_i32x16, a_high_i16x32, a_high_i16x32);
    b_norm_sq_low_i32x16 = _mm512_dpwssds_epi32(b_norm_sq_low_i32x16, b_low_i16x32, b_low_i16x32);
    b_norm_sq_high_i32x16 = _mm512_dpwssds_epi32(b_norm_sq_high_i32x16, b_high_i16x32, b_high_i16x32);
    if (n) goto nk_angular_u8_ice_cycle;

    nk_i32_t dot_product_i32 = _mm512_reduce_add_epi32(
        _mm512_add_epi32(dot_product_low_i32x16, dot_product_high_i32x16));
    nk_i32_t a_norm_sq_i32 = _mm512_reduce_add_epi32(_mm512_add_epi32(a_norm_sq_low_i32x16, a_norm_sq_high_i32x16));
    nk_i32_t b_norm_sq_i32 = _mm512_reduce_add_epi32(_mm512_add_epi32(b_norm_sq_low_i32x16, b_norm_sq_high_i32x16));
    *result = _nk_angular_normalize_f32_haswell(dot_product_i32, a_norm_sq_i32, b_norm_sq_i32);
}

NK_PUBLIC void nk_l2_i4x2_ice(nk_i4x2_t const *a, nk_i4x2_t const *b, nk_size_t n_words, nk_f32_t *result) {
    nk_u32_t d2;
    nk_l2sq_i4x2_ice(a, b, n_words, &d2);
    *result = _nk_sqrt_f32_haswell((nk_f32_t)d2);
}
NK_PUBLIC void nk_l2sq_i4x2_ice(nk_i4x2_t const *a, nk_i4x2_t const *b, nk_size_t n_words, nk_u32_t *result) {

    // While `int8_t` covers the range [-128, 127], `int4_t` covers only [-8, 7].
    // The absolute difference between two 4-bit integers is at most 15 and it is always a `uint4_t` value!
    // Moreover, it's square is at most 225, which fits into `uint8_t` and can be computed with a single
    // lookup table. Accumulating those values is similar to checksumming, a piece of cake for SIMD!
    __m512i const i4_to_i8_lookup_vec = _mm512_set_epi8(        //
        -1, -2, -3, -4, -5, -6, -7, -8, 7, 6, 5, 4, 3, 2, 1, 0, //
        -1, -2, -3, -4, -5, -6, -7, -8, 7, 6, 5, 4, 3, 2, 1, 0, //
        -1, -2, -3, -4, -5, -6, -7, -8, 7, 6, 5, 4, 3, 2, 1, 0, //
        -1, -2, -3, -4, -5, -6, -7, -8, 7, 6, 5, 4, 3, 2, 1, 0);
    __m512i const u4_squares_lookup_vec = _mm512_set_epi8(                                        //
        (char)225, (char)196, (char)169, (char)144, 121, 100, 81, 64, 49, 36, 25, 16, 9, 4, 1, 0, //
        (char)225, (char)196, (char)169, (char)144, 121, 100, 81, 64, 49, 36, 25, 16, 9, 4, 1, 0, //
        (char)225, (char)196, (char)169, (char)144, 121, 100, 81, 64, 49, 36, 25, 16, 9, 4, 1, 0, //
        (char)225, (char)196, (char)169, (char)144, 121, 100, 81, 64, 49, 36, 25, 16, 9, 4, 1, 0);

    /// The mask used to take the low nibble of each byte.
    __m512i const i4_nibble_vec = _mm512_set1_epi8(0x0F);

    // Temporaries:
    __m512i a_i4x2_vec, b_i4x2_vec;
    __m512i a_i8_low_vec, a_i8_high_vec, b_i8_low_vec, b_i8_high_vec;
    __m512i d_u8_low_vec, d_u8_high_vec; //! Only the low 4 bits are actually used
    __m512i d2_u8_low_vec, d2_u8_high_vec;
    __m512i d2_u16_low_vec, d2_u16_high_vec;

    // Accumulators:
    __m512i d2_u32_vec = _mm512_setzero_si512();

nk_l2sq_i4x2_ice_cycle:
    if (n_words < 64) {
        __mmask64 mask = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFF, n_words);
        a_i4x2_vec = _mm512_maskz_loadu_epi8(mask, a);
        b_i4x2_vec = _mm512_maskz_loadu_epi8(mask, b);
        n_words = 0;
    }
    else {
        a_i4x2_vec = _mm512_loadu_epi8(a);
        b_i4x2_vec = _mm512_loadu_epi8(b);
        a += 64, b += 64, n_words -= 64;
    }

    // Unpack the 4-bit values into 8-bit values with an empty top nibble.
    a_i8_low_vec = _mm512_and_si512(a_i4x2_vec, i4_nibble_vec);
    a_i8_high_vec = _mm512_and_si512(_mm512_srli_epi64(a_i4x2_vec, 4), i4_nibble_vec);
    b_i8_low_vec = _mm512_and_si512(b_i4x2_vec, i4_nibble_vec);
    b_i8_high_vec = _mm512_and_si512(_mm512_srli_epi64(b_i4x2_vec, 4), i4_nibble_vec);
    a_i8_low_vec = _mm512_shuffle_epi8(i4_to_i8_lookup_vec, a_i8_low_vec);
    a_i8_high_vec = _mm512_shuffle_epi8(i4_to_i8_lookup_vec, a_i8_high_vec);
    b_i8_low_vec = _mm512_shuffle_epi8(i4_to_i8_lookup_vec, b_i8_low_vec);
    b_i8_high_vec = _mm512_shuffle_epi8(i4_to_i8_lookup_vec, b_i8_high_vec);

    // We can implement subtraction with a lookup table, or using `_mm512_sub_epi8`.
    d_u8_low_vec = _mm512_abs_epi8(_mm512_sub_epi8(a_i8_low_vec, b_i8_low_vec));
    d_u8_high_vec = _mm512_abs_epi8(_mm512_sub_epi8(a_i8_high_vec, b_i8_high_vec));

    // Now we can use the lookup table to compute the squares of the 4-bit unsigned integers
    // in the low nibbles of the `d_u8_low_vec` and `d_u8_high_vec` vectors.
    d2_u8_low_vec = _mm512_shuffle_epi8(u4_squares_lookup_vec, d_u8_low_vec);
    d2_u8_high_vec = _mm512_shuffle_epi8(u4_squares_lookup_vec, d_u8_high_vec);

    // Aggregating into 16-bit integers, we need to first upcast our 8-bit values to 16 bits.
    // After that, we will perform one more operation, upcasting further into 32-bit integers.
    d2_u16_low_vec =      //
        _mm512_add_epi16( //
            _mm512_unpacklo_epi8(d2_u8_low_vec, _mm512_setzero_si512()),
            _mm512_unpackhi_epi8(d2_u8_low_vec, _mm512_setzero_si512()));
    d2_u16_high_vec =     //
        _mm512_add_epi16( //
            _mm512_unpacklo_epi8(d2_u8_high_vec, _mm512_setzero_si512()),
            _mm512_unpackhi_epi8(d2_u8_high_vec, _mm512_setzero_si512()));
    d2_u32_vec = _mm512_add_epi32(d2_u32_vec, _mm512_unpacklo_epi16(d2_u16_low_vec, _mm512_setzero_si512()));
    d2_u32_vec = _mm512_add_epi32(d2_u32_vec, _mm512_unpacklo_epi16(d2_u16_high_vec, _mm512_setzero_si512()));
    if (n_words) goto nk_l2sq_i4x2_ice_cycle;

    // Finally, we can reduce the 16-bit integers to 32-bit integers and sum them up.
    int d2 = _mm512_reduce_add_epi32(d2_u32_vec);
    *result = d2;
}
NK_PUBLIC void nk_angular_i4x2_ice(nk_i4x2_t const *a, nk_i4x2_t const *b, nk_size_t n_words, nk_f32_t *result) {

    // We need to compose a lookup table for all the scalar products of 4-bit integers.
    // While `int8_t` covers the range [-128, 127], `int4_t` covers only [-8, 7].
    // Practically speaking, the product of two 4-bit signed integers is a 7-bit integer,
    // as the maximum absolute value of the product is `abs(-8 * -8) == 64`.
    //
    // To store 128 possible values of 2^7 bits we only need 128 single-byte scalars,
    // or just 2x ZMM registers. In that case our lookup will only take `vpermi2b` instruction,
    // easily inokable with `_mm512_permutex2var_epi8` intrinsic with latency of 6 on Sapphire Rapids.
    // The problem is converting 2d indices of our symmetric matrix into 1d offsets in the dense array.
    //
    // Alternatively, we can take the entire symmetric (16 x 16) matrix of products,
    // put into 4x ZMM registers, and use it with `_mm512_shuffle_epi8`, remembering
    // that it can only lookup with 128-bit lanes (16x 8-bit values).
    // That intrinsic has latency 1, but will need to be repeated and combined with
    // multiple iterations of `_mm512_shuffle_i64x2` that has latency 3.
    //
    // Altenatively, we can get down to 3 cycles per lookup with `vpermb` and `_mm512_permutexvar_epi8` intrinsics.
    // For that we can split our (16 x 16) matrix into 4x (8 x 8) submatrices, and use 4x ZMM registers.
    //
    // Still, all of those solutions are quite heavy compared to two parallel calls to `_mm512_dpbusds_epi32`
    // for the dot product. But we can still use the `_mm512_permutexvar_epi8` to compute the squares of the
    // 16 possible `int4_t` values faster.
    //
    // Here is how our `int4_t` range looks:
    //
    //      dec:     0   1   2   3   4   5   6   7  -8  -7  -6  -5  -4  -3  -2  -1
    //      hex:     0   1   2   3   4   5   6   7   8   9   A   B   C   D   E   F
    //
    // Squared:
    //
    //      dec2:    0   1   4   9  16  25  36  49  64  49  36  25  16   9   4   1
    //      hex2:    0   1   4   9  10  19  24  31  40  31  24  19  10   9   4   1
    //
    // Broadcast it to every lane, so that: `square(x) == _mm512_shuffle_epi8(i4_squares_lookup_vec, x)`.
    __m512i const i4_to_i8_lookup_vec = _mm512_set_epi8(        //
        -1, -2, -3, -4, -5, -6, -7, -8, 7, 6, 5, 4, 3, 2, 1, 0, //
        -1, -2, -3, -4, -5, -6, -7, -8, 7, 6, 5, 4, 3, 2, 1, 0, //
        -1, -2, -3, -4, -5, -6, -7, -8, 7, 6, 5, 4, 3, 2, 1, 0, //
        -1, -2, -3, -4, -5, -6, -7, -8, 7, 6, 5, 4, 3, 2, 1, 0);
    __m512i const i4_squares_lookup_vec = _mm512_set_epi8(       //
        1, 4, 9, 16, 25, 36, 49, 64, 49, 36, 25, 16, 9, 4, 1, 0, //
        1, 4, 9, 16, 25, 36, 49, 64, 49, 36, 25, 16, 9, 4, 1, 0, //
        1, 4, 9, 16, 25, 36, 49, 64, 49, 36, 25, 16, 9, 4, 1, 0, //
        1, 4, 9, 16, 25, 36, 49, 64, 49, 36, 25, 16, 9, 4, 1, 0);

    /// The mask used to take the low nibble of each byte.
    __m512i const i4_nibble_vec = _mm512_set1_epi8(0x0F);

    // Temporaries:
    __m512i a_i4x2_vec, b_i4x2_vec;
    __m512i a_i8_low_vec, a_i8_high_vec, b_i8_low_vec, b_i8_high_vec;
    __m512i a2_u8_vec, b2_u8_vec;

    // Accumulators:
    __m512i a2_u16_low_vec = _mm512_setzero_si512();
    __m512i a2_u16_high_vec = _mm512_setzero_si512();
    __m512i b2_u16_low_vec = _mm512_setzero_si512();
    __m512i b2_u16_high_vec = _mm512_setzero_si512();
    __m512i ab_i32_low_vec = _mm512_setzero_si512();
    __m512i ab_i32_high_vec = _mm512_setzero_si512();

nk_angular_i4x2_ice_cycle:
    if (n_words < 64) {
        __mmask64 mask = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFF, n_words);
        a_i4x2_vec = _mm512_maskz_loadu_epi8(mask, a);
        b_i4x2_vec = _mm512_maskz_loadu_epi8(mask, b);
        n_words = 0;
    }
    else {
        a_i4x2_vec = _mm512_loadu_epi8(a);
        b_i4x2_vec = _mm512_loadu_epi8(b);
        a += 64, b += 64, n_words -= 64;
    }

    // Unpack the 4-bit values into 8-bit values with an empty top nibble.
    // For now, they are not really 8-bit integers, as they are not sign-extended.
    // That part will come later, using the `i4_to_i8_lookup_vec` lookup.
    a_i8_low_vec = _mm512_and_si512(a_i4x2_vec, i4_nibble_vec);
    a_i8_high_vec = _mm512_and_si512(_mm512_srli_epi64(a_i4x2_vec, 4), i4_nibble_vec);
    b_i8_low_vec = _mm512_and_si512(b_i4x2_vec, i4_nibble_vec);
    b_i8_high_vec = _mm512_and_si512(_mm512_srli_epi64(b_i4x2_vec, 4), i4_nibble_vec);

    // Compute the squares of the 4-bit integers.
    // For symmetry we could have used 4 registers, aka "a2_i8_low_vec", "a2_i8_high_vec", "b2_i8_low_vec",
    // "b2_i8_high_vec". But the largest square value is just 64, so we can safely aggregate into 8-bit unsigned values.
    a2_u8_vec = _mm512_add_epi8(_mm512_shuffle_epi8(i4_squares_lookup_vec, a_i8_low_vec),
                                _mm512_shuffle_epi8(i4_squares_lookup_vec, a_i8_high_vec));
    b2_u8_vec = _mm512_add_epi8(_mm512_shuffle_epi8(i4_squares_lookup_vec, b_i8_low_vec),
                                _mm512_shuffle_epi8(i4_squares_lookup_vec, b_i8_high_vec));

    // We can safely aggregate into just 16-bit sums without overflow, if the vectors have less than:
    //      (2 scalars / byte) * (64 bytes / register) * (256 non-overflowing 8-bit additions in 16-bit intesgers)
    //      = 32'768 dimensions.
    //
    // We use saturated addition here to clearly inform in case of overflow.
    a2_u16_low_vec = _mm512_adds_epu16(a2_u16_low_vec, _mm512_cvtepu8_epi16(_mm512_castsi512_si256(a2_u8_vec)));
    a2_u16_high_vec = _mm512_adds_epu16(a2_u16_high_vec, _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64(a2_u8_vec, 1)));
    b2_u16_low_vec = _mm512_adds_epu16(b2_u16_low_vec, _mm512_cvtepu8_epi16(_mm512_castsi512_si256(a2_u8_vec)));
    b2_u16_high_vec = _mm512_adds_epu16(b2_u16_high_vec, _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64(a2_u8_vec, 1)));

    // Time to perform the proper sign extension of the 4-bit integers to 8-bit integers.
    a_i8_low_vec = _mm512_shuffle_epi8(i4_to_i8_lookup_vec, a_i8_low_vec);
    a_i8_high_vec = _mm512_shuffle_epi8(i4_to_i8_lookup_vec, a_i8_high_vec);
    b_i8_low_vec = _mm512_shuffle_epi8(i4_to_i8_lookup_vec, b_i8_low_vec);
    b_i8_high_vec = _mm512_shuffle_epi8(i4_to_i8_lookup_vec, b_i8_high_vec);

    // The same trick won't work for the primary dot-product, as the signs vector
    // components may differ significantly. So we have to use two `_mm512_dpwssds_epi32`
    // intrinsics instead, upcasting four chunks to 16-bit integers beforehand!
    // Alternatively, we can flip the signs of the second argument and use `_mm512_dpbusds_epi32`,
    // but it ends up taking more instructions.
    ab_i32_low_vec = _mm512_dpwssds_epi32(                          //
        ab_i32_low_vec,                                             //
        _mm512_cvtepi8_epi16(_mm512_castsi512_si256(a_i8_low_vec)), //
        _mm512_cvtepi8_epi16(_mm512_castsi512_si256(b_i8_low_vec)));
    ab_i32_low_vec = _mm512_dpwssds_epi32(                                //
        ab_i32_low_vec,                                                   //
        _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(a_i8_low_vec, 1)), //
        _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(b_i8_low_vec, 1)));
    ab_i32_high_vec = _mm512_dpwssds_epi32(                          //
        ab_i32_high_vec,                                             //
        _mm512_cvtepi8_epi16(_mm512_castsi512_si256(a_i8_high_vec)), //
        _mm512_cvtepi8_epi16(_mm512_castsi512_si256(b_i8_high_vec)));
    ab_i32_high_vec = _mm512_dpwssds_epi32(                                //
        ab_i32_high_vec,                                                   //
        _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(a_i8_high_vec, 1)), //
        _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(b_i8_high_vec, 1)));
    if (n_words) goto nk_angular_i4x2_ice_cycle;

    int ab = _mm512_reduce_add_epi32(_mm512_add_epi32(ab_i32_low_vec, ab_i32_high_vec));
    unsigned short a2_u16[32], b2_u16[32];
    _mm512_storeu_si512(a2_u16, _mm512_add_epi16(a2_u16_low_vec, a2_u16_high_vec));
    _mm512_storeu_si512(b2_u16, _mm512_add_epi16(b2_u16_low_vec, b2_u16_high_vec));
    unsigned int a2 = 0, b2 = 0;
    for (int i = 0; i < 32; ++i) a2 += a2_u16[i], b2 += b2_u16[i];
    *result = _nk_angular_normalize_f32_haswell(ab, a2, b2);
}

typedef nk_dot_i8x64_state_ice_t nk_angular_i8x64_state_ice_t;
NK_INTERNAL void nk_angular_i8x64_init_ice(nk_angular_i8x64_state_ice_t *state) { nk_dot_i8x64_init_ice(state); }
NK_INTERNAL void nk_angular_i8x64_update_ice(nk_angular_i8x64_state_ice_t *state, nk_b512_vec_t a, nk_b512_vec_t b) {
    nk_dot_i8x64_update_ice(state, a, b);
}
NK_INTERNAL void nk_angular_i8x64_finalize_ice(nk_angular_i8x64_state_ice_t const *state_a,
                                               nk_angular_i8x64_state_ice_t const *state_b,
                                               nk_angular_i8x64_state_ice_t const *state_c,
                                               nk_angular_i8x64_state_ice_t const *state_d, nk_f32_t query_norm,
                                               nk_f32_t target_norm_a, nk_f32_t target_norm_b, nk_f32_t target_norm_c,
                                               nk_f32_t target_norm_d, nk_f32_t *results) {
    // Extract all 4 dot products with single ILP-optimized call
    nk_i32_t dots_i32[4];
    nk_dot_i8x64_finalize_ice(state_a, state_b, state_c, state_d, dots_i32);

    // Convert dots to f32 and build vectors for parallel processing
    __m128 dots_f32x4 = _mm_cvtepi32_ps(_mm_loadu_si128((__m128i const *)dots_i32));
    __m128 query_norm_sq_f32x4 = _mm_set1_ps(query_norm * query_norm);
    __m128 target_norms_sq_f32x4 = _mm_set_ps(target_norm_d * target_norm_d, target_norm_c * target_norm_c,
                                              target_norm_b * target_norm_b, target_norm_a * target_norm_a);

    // products = query_norm_sq * target_norms_sq
    __m128 products_f32x4 = _mm_mul_ps(query_norm_sq_f32x4, target_norms_sq_f32x4);

    // rsqrt with Newton-Raphson refinement: x' = x * (1.5 - 0.5 * val * x * x)
    __m128 rsqrt_f32x4 = _mm_rsqrt_ps(products_f32x4);
    __m128 half_f32x4 = _mm_set1_ps(0.5f);
    __m128 three_halves_f32x4 = _mm_set1_ps(1.5f);
    __m128 rsqrt_sq_f32x4 = _mm_mul_ps(rsqrt_f32x4, rsqrt_f32x4);
    __m128 half_prod_f32x4 = _mm_mul_ps(half_f32x4, products_f32x4);
    __m128 muls_f32x4 = _mm_mul_ps(half_prod_f32x4, rsqrt_sq_f32x4);
    __m128 refinement_f32x4 = _mm_sub_ps(three_halves_f32x4, muls_f32x4);
    rsqrt_f32x4 = _mm_mul_ps(rsqrt_f32x4, refinement_f32x4);

    // normalized = dots * rsqrt(products)
    __m128 normalized_f32x4 = _mm_mul_ps(dots_f32x4, rsqrt_f32x4);

    // angular = 1 - normalized
    __m128 ones_f32x4 = _mm_set1_ps(1.0f);
    __m128 angular_f32x4 = _mm_sub_ps(ones_f32x4, normalized_f32x4);

    // Store results
    _mm_storeu_ps(results, angular_f32x4);
}

typedef nk_dot_i8x64_state_ice_t nk_l2_i8x64_state_ice_t;
NK_INTERNAL void nk_l2_i8x64_init_ice(nk_l2_i8x64_state_ice_t *state) { nk_dot_i8x64_init_ice(state); }
NK_INTERNAL void nk_l2_i8x64_update_ice(nk_l2_i8x64_state_ice_t *state, nk_b512_vec_t a, nk_b512_vec_t b) {
    nk_dot_i8x64_update_ice(state, a, b);
}
NK_INTERNAL void nk_l2_i8x64_finalize_ice(nk_l2_i8x64_state_ice_t const *state_a,
                                          nk_l2_i8x64_state_ice_t const *state_b,
                                          nk_l2_i8x64_state_ice_t const *state_c,
                                          nk_l2_i8x64_state_ice_t const *state_d, nk_f32_t query_norm,
                                          nk_f32_t target_norm_a, nk_f32_t target_norm_b, nk_f32_t target_norm_c,
                                          nk_f32_t target_norm_d, nk_f32_t *results) {
    // Extract all 4 dot products with single ILP-optimized call
    nk_i32_t dots_i32[4];
    nk_dot_i8x64_finalize_ice(state_a, state_b, state_c, state_d, dots_i32);

    // Convert dots to f32 and build vectors for parallel processing
    __m128 dots_f32x4 = _mm_cvtepi32_ps(_mm_loadu_si128((__m128i const *)dots_i32));
    __m128 query_norm_sq_f32x4 = _mm_set1_ps(query_norm * query_norm);
    __m128 target_norms_sq_f32x4 = _mm_set_ps(target_norm_d * target_norm_d, target_norm_c * target_norm_c,
                                              target_norm_b * target_norm_b, target_norm_a * target_norm_a);

    // L2 distance: sqrt(query_sq + target_sq - 2*dot)
    // dist_sq = query_norm_sq + target_norms_sq - 2*dots
    __m128 two_f32x4 = _mm_set1_ps(2.0f);
    __m128 two_dots_f32x4 = _mm_mul_ps(two_f32x4, dots_f32x4);
    __m128 sum_sq_f32x4 = _mm_add_ps(query_norm_sq_f32x4, target_norms_sq_f32x4);
    __m128 dist_sq_f32x4 = _mm_sub_ps(sum_sq_f32x4, two_dots_f32x4);

    // Clamp negatives to zero and take sqrt
    __m128 zeros_f32x4 = _mm_setzero_ps();
    dist_sq_f32x4 = _mm_max_ps(dist_sq_f32x4, zeros_f32x4);
    __m128 dist_f32x4 = _mm_sqrt_ps(dist_sq_f32x4);

    // Store results
    _mm_storeu_ps(results, dist_f32x4);
}

typedef nk_dot_u8x64_state_ice_t nk_angular_u8x64_state_ice_t;
NK_INTERNAL void nk_angular_u8x64_init_ice(nk_angular_u8x64_state_ice_t *state) { nk_dot_u8x64_init_ice(state); }
NK_INTERNAL void nk_angular_u8x64_update_ice(nk_angular_u8x64_state_ice_t *state, nk_b512_vec_t a, nk_b512_vec_t b) {
    nk_dot_u8x64_update_ice(state, a, b);
}
NK_INTERNAL void nk_angular_u8x64_finalize_ice(nk_angular_u8x64_state_ice_t const *state_a,
                                               nk_angular_u8x64_state_ice_t const *state_b,
                                               nk_angular_u8x64_state_ice_t const *state_c,
                                               nk_angular_u8x64_state_ice_t const *state_d, nk_f32_t query_norm,
                                               nk_f32_t target_norm_a, nk_f32_t target_norm_b, nk_f32_t target_norm_c,
                                               nk_f32_t target_norm_d, nk_f32_t *results) {
    // Extract all 4 dot products with single ILP-optimized call
    nk_u32_t dots_u32[4];
    nk_dot_u8x64_finalize_ice(state_a, state_b, state_c, state_d, dots_u32);

    // Convert dots to f32 (u32 values fit in f32 mantissa for typical vector lengths)
    __m128 dots_f32x4 = _mm_cvtepi32_ps(_mm_loadu_si128((__m128i const *)dots_u32));
    __m128 query_norm_sq_f32x4 = _mm_set1_ps(query_norm * query_norm);
    __m128 target_norms_sq_f32x4 = _mm_set_ps(target_norm_d * target_norm_d, target_norm_c * target_norm_c,
                                              target_norm_b * target_norm_b, target_norm_a * target_norm_a);

    // products = query_norm_sq * target_norms_sq
    __m128 products_f32x4 = _mm_mul_ps(query_norm_sq_f32x4, target_norms_sq_f32x4);

    // rsqrt with Newton-Raphson refinement: x' = x * (1.5 - 0.5 * val * x * x)
    __m128 rsqrt_f32x4 = _mm_rsqrt_ps(products_f32x4);
    __m128 half_f32x4 = _mm_set1_ps(0.5f);
    __m128 three_halves_f32x4 = _mm_set1_ps(1.5f);
    __m128 rsqrt_sq_f32x4 = _mm_mul_ps(rsqrt_f32x4, rsqrt_f32x4);
    __m128 half_prod_f32x4 = _mm_mul_ps(half_f32x4, products_f32x4);
    __m128 muls_f32x4 = _mm_mul_ps(half_prod_f32x4, rsqrt_sq_f32x4);
    __m128 refinement_f32x4 = _mm_sub_ps(three_halves_f32x4, muls_f32x4);
    rsqrt_f32x4 = _mm_mul_ps(rsqrt_f32x4, refinement_f32x4);

    // normalized = dots * rsqrt(products)
    __m128 normalized_f32x4 = _mm_mul_ps(dots_f32x4, rsqrt_f32x4);

    // angular = 1 - normalized
    __m128 ones_f32x4 = _mm_set1_ps(1.0f);
    __m128 angular_f32x4 = _mm_sub_ps(ones_f32x4, normalized_f32x4);

    // Store results
    _mm_storeu_ps(results, angular_f32x4);
}

typedef nk_dot_u8x64_state_ice_t nk_l2_u8x64_state_ice_t;
NK_INTERNAL void nk_l2_u8x64_init_ice(nk_l2_u8x64_state_ice_t *state) { nk_dot_u8x64_init_ice(state); }
NK_INTERNAL void nk_l2_u8x64_update_ice(nk_l2_u8x64_state_ice_t *state, nk_b512_vec_t a, nk_b512_vec_t b) {
    nk_dot_u8x64_update_ice(state, a, b);
}
NK_INTERNAL void nk_l2_u8x64_finalize_ice(nk_l2_u8x64_state_ice_t const *state_a,
                                          nk_l2_u8x64_state_ice_t const *state_b,
                                          nk_l2_u8x64_state_ice_t const *state_c,
                                          nk_l2_u8x64_state_ice_t const *state_d, nk_f32_t query_norm,
                                          nk_f32_t target_norm_a, nk_f32_t target_norm_b, nk_f32_t target_norm_c,
                                          nk_f32_t target_norm_d, nk_f32_t *results) {
    // Extract all 4 dot products with single ILP-optimized call
    nk_u32_t dots_u32[4];
    nk_dot_u8x64_finalize_ice(state_a, state_b, state_c, state_d, dots_u32);

    // Convert dots to f32 (u32 values fit in f32 mantissa for typical vector lengths)
    __m128 dots_f32x4 = _mm_cvtepi32_ps(_mm_loadu_si128((__m128i const *)dots_u32));
    __m128 query_norm_sq_f32x4 = _mm_set1_ps(query_norm * query_norm);
    __m128 target_norms_sq_f32x4 = _mm_set_ps(target_norm_d * target_norm_d, target_norm_c * target_norm_c,
                                              target_norm_b * target_norm_b, target_norm_a * target_norm_a);

    // L2 distance: sqrt(query_sq + target_sq - 2*dot)
    // dist_sq = query_norm_sq + target_norms_sq - 2*dots
    __m128 two_f32x4 = _mm_set1_ps(2.0f);
    __m128 two_dots_f32x4 = _mm_mul_ps(two_f32x4, dots_f32x4);
    __m128 sum_sq_f32x4 = _mm_add_ps(query_norm_sq_f32x4, target_norms_sq_f32x4);
    __m128 dist_sq_f32x4 = _mm_sub_ps(sum_sq_f32x4, two_dots_f32x4);

    // Clamp negatives to zero and take sqrt
    __m128 zeros_f32x4 = _mm_setzero_ps();
    dist_sq_f32x4 = _mm_max_ps(dist_sq_f32x4, zeros_f32x4);
    __m128 dist_f32x4 = _mm_sqrt_ps(dist_sq_f32x4);

    // Store results
    _mm_storeu_ps(results, dist_f32x4);
}

#if defined(__cplusplus)
} // extern "C"
#endif

#pragma clang attribute pop
#pragma GCC pop_options
#endif // NK_TARGET_ICE
#endif // _NK_TARGET_X86

#endif // NK_SPATIAL_ICE_H
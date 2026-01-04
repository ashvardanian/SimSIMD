/**
 *  @brief SIMD-accelerated Spatial Similarity Measures optimized for Intel Ice Lake CPUs.
 *  @file include/numkong/spatial/ice.h
 *  @sa include/numkong/spatial.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 */
#ifndef NK_SPATIAL_ICE_H
#define NK_SPATIAL_ICE_H

#if NK_TARGET_X86_
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
    *result = nk_sqrt_f32_haswell_((nk_f32_t)d2);
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
    *result = nk_angular_normalize_f32_haswell_(dot_product_i32, a_norm_sq_i32, b_norm_sq_i32);
}
NK_PUBLIC void nk_l2_u8_ice(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_u32_t d2;
    nk_l2sq_u8_ice(a, b, n, &d2);
    *result = nk_sqrt_f32_haswell_((nk_f32_t)d2);
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
    *result = nk_angular_normalize_f32_haswell_(dot_product_i32, a_norm_sq_i32, b_norm_sq_i32);
}

NK_PUBLIC void nk_l2_i4_ice(nk_i4x2_t const *a, nk_i4x2_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_u32_t d2;
    nk_l2sq_i4_ice(a, b, n, &d2);
    *result = nk_sqrt_f32_haswell_((nk_f32_t)d2);
}
NK_PUBLIC void nk_l2sq_i4_ice(nk_i4x2_t const *a, nk_i4x2_t const *b, nk_size_t n, nk_u32_t *result) {
    // i4 values are packed as nibbles: two 4-bit signed values per byte.
    // Parameter `n` is the number of 4-bit values (dimensions), not bytes.
    nk_size_t n_bytes = nk_size_divide_round_up_to_multiple_(n, 2);

    // While `int8_t` covers the range [-128, 127], `int4_t` covers only [-8, 7].
    // The absolute difference between two 4-bit integers is at most 15 and fits in `uint4_t`.
    // Moreover, its square is at most 225, which fits into `uint8_t`.
    //
    // Instead of using lookup tables for sign extension and squaring, we use arithmetic:
    //
    //  1. XOR trick for sign extension: `signed = (nibble ^ 8) - 8`
    //     Maps [0,7] -> [0,7] (positive) and [8,15] -> [-8,-1] (negative).
    //
    //  2. For L2 squared: |a-b|² = diff * diff, using `_mm512_dpbusd_epi32`.
    //     After computing signed difference and taking abs, the result fits in [0,15].
    //     We can then use DPBUSD to compute diff² efficiently without lookup tables.
    //
    // This approach avoids 8x VPSHUFB operations per iteration, replacing them with
    // arithmetic operations that distribute better across execution ports.
    __m512i const nibble_mask_u8x64 = _mm512_set1_epi8(0x0F);
    __m512i const eight_i8x64 = _mm512_set1_epi8(8);

    __m512i a_i4_vec, b_i4_vec;
    __m512i a_low_u8x64, a_high_u8x64, b_low_u8x64, b_high_u8x64;
    __m512i a_low_i8x64, a_high_i8x64, b_low_i8x64, b_high_i8x64;
    __m512i diff_low_u8x64, diff_high_u8x64;
    __m512i d2_i32x16 = _mm512_setzero_si512();

nk_l2sq_i4_ice_cycle:
    if (n_bytes < 64) {
        __mmask64 mask = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFF, n_bytes);
        a_i4_vec = _mm512_maskz_loadu_epi8(mask, a);
        b_i4_vec = _mm512_maskz_loadu_epi8(mask, b);
        n_bytes = 0;
    }
    else {
        a_i4_vec = _mm512_loadu_epi8(a);
        b_i4_vec = _mm512_loadu_epi8(b);
        a += 64, b += 64, n_bytes -= 64;
    }

    // Extract nibbles as unsigned [0,15]. VPSHUFB ignores high 4 bits of index,
    // so no AND needed for low nibbles when used with lookup, but we need it here.
    a_low_u8x64 = _mm512_and_si512(a_i4_vec, nibble_mask_u8x64);
    a_high_u8x64 = _mm512_and_si512(_mm512_srli_epi16(a_i4_vec, 4), nibble_mask_u8x64);
    b_low_u8x64 = _mm512_and_si512(b_i4_vec, nibble_mask_u8x64);
    b_high_u8x64 = _mm512_and_si512(_mm512_srli_epi16(b_i4_vec, 4), nibble_mask_u8x64);

    // Sign extend using XOR trick: signed = (nibble ^ 8) - 8
    a_low_i8x64 = _mm512_sub_epi8(_mm512_xor_si512(a_low_u8x64, eight_i8x64), eight_i8x64);
    a_high_i8x64 = _mm512_sub_epi8(_mm512_xor_si512(a_high_u8x64, eight_i8x64), eight_i8x64);
    b_low_i8x64 = _mm512_sub_epi8(_mm512_xor_si512(b_low_u8x64, eight_i8x64), eight_i8x64);
    b_high_i8x64 = _mm512_sub_epi8(_mm512_xor_si512(b_high_u8x64, eight_i8x64), eight_i8x64);

    // Compute |a - b| for each nibble pair. Result is unsigned in [0, 15].
    diff_low_u8x64 = _mm512_abs_epi8(_mm512_sub_epi8(a_low_i8x64, b_low_i8x64));
    diff_high_u8x64 = _mm512_abs_epi8(_mm512_sub_epi8(a_high_i8x64, b_high_i8x64));

    // Square and accumulate using DPBUSD: diff² = diff * diff.
    // DPBUSD computes u8*i8 products and sums groups of 4 into i32.
    // Since diff is in [0,15], it's safe for both u8 and i8 interpretation.
    d2_i32x16 = _mm512_dpbusd_epi32(d2_i32x16, diff_low_u8x64, diff_low_u8x64);
    d2_i32x16 = _mm512_dpbusd_epi32(d2_i32x16, diff_high_u8x64, diff_high_u8x64);
    if (n_bytes) goto nk_l2sq_i4_ice_cycle;

    *result = (nk_u32_t)_mm512_reduce_add_epi32(d2_i32x16);
}
NK_PUBLIC void nk_angular_i4_ice(nk_i4x2_t const *a, nk_i4x2_t const *b, nk_size_t n, nk_f32_t *result) {
    // i4 values are packed as nibbles: two 4-bit signed values per byte.
    // Parameter `n` is the number of 4-bit values (dimensions), not bytes.
    nk_size_t n_bytes = nk_size_divide_round_up_to_multiple_(n, 2);

    // Angular distance for signed 4-bit integers requires computing:
    //   1. Dot product: sum(a[i] * b[i])
    //   2. Squared norms: sum(a[i]²) and sum(b[i]²)
    //
    // For signed i4 values in [-8, 7], we use DPBUSD for everything by leveraging
    // an algebraic identity. Define x = a ^ 8 (XOR with 8), which maps:
    //   [0,7] -> [8,15] and [8,15] -> [0,7]
    //
    // The signed value is: a_signed = x - 8
    //
    // For two signed values:
    //   a_signed * b_signed = (ax - 8)(bx - 8) = ax*bx - 8*ax - 8*bx + 64
    //
    // Therefore:
    //   dot(a_signed, b_signed) = DPBUSD(ax, bx) - 8*(sum(ax) + sum(bx)) + 64*n
    //
    // This avoids all i8->i16 upcasts and uses DPBUSD directly on byte values!
    // For norms, we use |x|² = x², computing abs then squaring with DPBUSD.
    __m512i const nibble_mask_u8x64 = _mm512_set1_epi8(0x0F);
    __m512i const eight_i8x64 = _mm512_set1_epi8(8);
    __m512i const zeros_i8x64 = _mm512_setzero_si512();

    __m512i a_i4_vec, b_i4_vec;
    __m512i a_low_u8x64, a_high_u8x64, b_low_u8x64, b_high_u8x64;
    __m512i ax_low_u8x64, ax_high_u8x64, bx_low_u8x64, bx_high_u8x64;
    __m512i a_low_i8x64, a_high_i8x64, b_low_i8x64, b_high_i8x64;

    // Accumulators for dot product (using biased values) and correction sums
    __m512i ab_i32x16 = zeros_i8x64;
    __m512i ax_sum_i64x8 = zeros_i8x64;
    __m512i bx_sum_i64x8 = zeros_i8x64;
    // Accumulators for squared norms
    __m512i a2_i32x16 = zeros_i8x64;
    __m512i b2_i32x16 = zeros_i8x64;

nk_angular_i4_ice_cycle:
    if (n_bytes < 64) {
        __mmask64 mask = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFF, n_bytes);
        a_i4_vec = _mm512_maskz_loadu_epi8(mask, a);
        b_i4_vec = _mm512_maskz_loadu_epi8(mask, b);
        n_bytes = 0;
    }
    else {
        a_i4_vec = _mm512_loadu_epi8(a);
        b_i4_vec = _mm512_loadu_epi8(b);
        a += 64, b += 64, n_bytes -= 64;
    }

    // Extract nibbles as unsigned [0,15]
    a_low_u8x64 = _mm512_and_si512(a_i4_vec, nibble_mask_u8x64);
    a_high_u8x64 = _mm512_and_si512(_mm512_srli_epi16(a_i4_vec, 4), nibble_mask_u8x64);
    b_low_u8x64 = _mm512_and_si512(b_i4_vec, nibble_mask_u8x64);
    b_high_u8x64 = _mm512_and_si512(_mm512_srli_epi16(b_i4_vec, 4), nibble_mask_u8x64);

    // Compute biased values: ax = a ^ 8 (still in [0,15], just reordered)
    ax_low_u8x64 = _mm512_xor_si512(a_low_u8x64, eight_i8x64);
    ax_high_u8x64 = _mm512_xor_si512(a_high_u8x64, eight_i8x64);
    bx_low_u8x64 = _mm512_xor_si512(b_low_u8x64, eight_i8x64);
    bx_high_u8x64 = _mm512_xor_si512(b_high_u8x64, eight_i8x64);

    // Dot product using DPBUSD on biased values (correction applied at end)
    ab_i32x16 = _mm512_dpbusd_epi32(ab_i32x16, ax_low_u8x64, bx_low_u8x64);
    ab_i32x16 = _mm512_dpbusd_epi32(ab_i32x16, ax_high_u8x64, bx_high_u8x64);

    // Track sums for correction using SAD (sum of absolute differences with zero)
    ax_sum_i64x8 = _mm512_add_epi64(ax_sum_i64x8, _mm512_sad_epu8(ax_low_u8x64, zeros_i8x64));
    ax_sum_i64x8 = _mm512_add_epi64(ax_sum_i64x8, _mm512_sad_epu8(ax_high_u8x64, zeros_i8x64));
    bx_sum_i64x8 = _mm512_add_epi64(bx_sum_i64x8, _mm512_sad_epu8(bx_low_u8x64, zeros_i8x64));
    bx_sum_i64x8 = _mm512_add_epi64(bx_sum_i64x8, _mm512_sad_epu8(bx_high_u8x64, zeros_i8x64));

    // For norms: convert to signed, take abs, then square with DPBUSD
    a_low_i8x64 = _mm512_sub_epi8(ax_low_u8x64, eight_i8x64);
    a_high_i8x64 = _mm512_sub_epi8(ax_high_u8x64, eight_i8x64);
    b_low_i8x64 = _mm512_sub_epi8(bx_low_u8x64, eight_i8x64);
    b_high_i8x64 = _mm512_sub_epi8(bx_high_u8x64, eight_i8x64);

    __m512i a_low_abs_u8x64 = _mm512_abs_epi8(a_low_i8x64);
    __m512i a_high_abs_u8x64 = _mm512_abs_epi8(a_high_i8x64);
    __m512i b_low_abs_u8x64 = _mm512_abs_epi8(b_low_i8x64);
    __m512i b_high_abs_u8x64 = _mm512_abs_epi8(b_high_i8x64);

    // Squared norms: |x|² = x², use DPBUSD for efficient squaring
    a2_i32x16 = _mm512_dpbusd_epi32(a2_i32x16, a_low_abs_u8x64, a_low_abs_u8x64);
    a2_i32x16 = _mm512_dpbusd_epi32(a2_i32x16, a_high_abs_u8x64, a_high_abs_u8x64);
    b2_i32x16 = _mm512_dpbusd_epi32(b2_i32x16, b_low_abs_u8x64, b_low_abs_u8x64);
    b2_i32x16 = _mm512_dpbusd_epi32(b2_i32x16, b_high_abs_u8x64, b_high_abs_u8x64);
    if (n_bytes) goto nk_angular_i4_ice_cycle;

    // Apply algebraic correction for signed dot product:
    // signed_dot = DPBUSD(ax, bx) - 8*(sum(ax) + sum(bx)) + 64*n
    nk_i64_t ax_sum = _mm512_reduce_add_epi64(ax_sum_i64x8);
    nk_i64_t bx_sum = _mm512_reduce_add_epi64(bx_sum_i64x8);
    nk_i32_t ab_raw = _mm512_reduce_add_epi32(ab_i32x16);
    nk_i32_t ab = ab_raw - 8 * (nk_i32_t)(ax_sum + bx_sum) + 64 * (nk_i32_t)n;

    nk_i32_t a2 = _mm512_reduce_add_epi32(a2_i32x16);
    nk_i32_t b2 = _mm512_reduce_add_epi32(b2_i32x16);
    *result = nk_angular_normalize_f32_haswell_(ab, (nk_f32_t)a2, (nk_f32_t)b2);
}

NK_PUBLIC void nk_l2sq_u4_ice(nk_u4x2_t const *a, nk_u4x2_t const *b, nk_size_t n, nk_u32_t *result) {
    // u4 values are packed as nibbles: two 4-bit unsigned values per byte.
    // Parameter `n` is the number of 4-bit values (dimensions), not bytes.
    nk_size_t n_bytes = nk_size_divide_round_up_to_multiple_(n, 2);

    // For unsigned 4-bit integers in [0, 15], the L2 squared distance is straightforward:
    //   1. Extract nibbles as u8 values
    //   2. Compute |a - b| using saturating subtraction: max(a,b) - min(a,b) = (a ⊖ b) | (b ⊖ a)
    //   3. Square with DPBUSD: diff * diff
    //
    // No sign extension needed since values are unsigned.
    __m512i const nibble_mask_u8x64 = _mm512_set1_epi8(0x0F);

    __m512i a_u4_vec, b_u4_vec;
    __m512i a_low_u8x64, a_high_u8x64, b_low_u8x64, b_high_u8x64;
    __m512i diff_low_u8x64, diff_high_u8x64;
    __m512i d2_i32x16 = _mm512_setzero_si512();

nk_l2sq_u4_ice_cycle:
    if (n_bytes < 64) {
        __mmask64 mask = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFF, n_bytes);
        a_u4_vec = _mm512_maskz_loadu_epi8(mask, a);
        b_u4_vec = _mm512_maskz_loadu_epi8(mask, b);
        n_bytes = 0;
    }
    else {
        a_u4_vec = _mm512_loadu_epi8(a);
        b_u4_vec = _mm512_loadu_epi8(b);
        a += 64, b += 64, n_bytes -= 64;
    }

    // Extract nibbles as unsigned [0,15]
    a_low_u8x64 = _mm512_and_si512(a_u4_vec, nibble_mask_u8x64);
    a_high_u8x64 = _mm512_and_si512(_mm512_srli_epi16(a_u4_vec, 4), nibble_mask_u8x64);
    b_low_u8x64 = _mm512_and_si512(b_u4_vec, nibble_mask_u8x64);
    b_high_u8x64 = _mm512_and_si512(_mm512_srli_epi16(b_u4_vec, 4), nibble_mask_u8x64);

    // Absolute difference for unsigned: |a-b| = (a ⊖ b) | (b ⊖ a) where ⊖ is saturating sub
    diff_low_u8x64 = _mm512_or_si512(_mm512_subs_epu8(a_low_u8x64, b_low_u8x64),
                                     _mm512_subs_epu8(b_low_u8x64, a_low_u8x64));
    diff_high_u8x64 = _mm512_or_si512(_mm512_subs_epu8(a_high_u8x64, b_high_u8x64),
                                      _mm512_subs_epu8(b_high_u8x64, a_high_u8x64));

    // Square and accumulate using DPBUSD
    d2_i32x16 = _mm512_dpbusd_epi32(d2_i32x16, diff_low_u8x64, diff_low_u8x64);
    d2_i32x16 = _mm512_dpbusd_epi32(d2_i32x16, diff_high_u8x64, diff_high_u8x64);
    if (n_bytes) goto nk_l2sq_u4_ice_cycle;

    *result = (nk_u32_t)_mm512_reduce_add_epi32(d2_i32x16);
}
NK_PUBLIC void nk_l2_u4_ice(nk_u4x2_t const *a, nk_u4x2_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_u32_t d2;
    nk_l2sq_u4_ice(a, b, n, &d2);
    *result = nk_sqrt_f32_haswell_((nk_f32_t)d2);
}

NK_PUBLIC void nk_angular_u4_ice(nk_u4x2_t const *a, nk_u4x2_t const *b, nk_size_t n, nk_f32_t *result) {
    // u4 values are packed as nibbles: two 4-bit unsigned values per byte.
    // Parameter `n` is the number of 4-bit values (dimensions), not bytes.
    nk_size_t n_bytes = nk_size_divide_round_up_to_multiple_(n, 2);

    // Angular distance for unsigned 4-bit integers in [0, 15].
    // Since values are unsigned and small, we can use DPBUSD directly for both
    // dot product and norms without any sign handling.
    //
    // DPBUSD computes: ZeroExtend(a) * SignExtend(b), but for values in [0, 15],
    // sign extension is identity (no high bit set), so it works correctly.
    __m512i const nibble_mask_u8x64 = _mm512_set1_epi8(0x0F);
    __m512i const zeros_i8x64 = _mm512_setzero_si512();

    __m512i a_u4_vec, b_u4_vec;
    __m512i a_low_u8x64, a_high_u8x64, b_low_u8x64, b_high_u8x64;

    __m512i ab_i32x16 = zeros_i8x64;
    __m512i a2_i64x8 = zeros_i8x64;
    __m512i b2_i64x8 = zeros_i8x64;

nk_angular_u4_ice_cycle:
    if (n_bytes < 64) {
        __mmask64 mask = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFF, n_bytes);
        a_u4_vec = _mm512_maskz_loadu_epi8(mask, a);
        b_u4_vec = _mm512_maskz_loadu_epi8(mask, b);
        n_bytes = 0;
    }
    else {
        a_u4_vec = _mm512_loadu_epi8(a);
        b_u4_vec = _mm512_loadu_epi8(b);
        a += 64, b += 64, n_bytes -= 64;
    }

    // Extract nibbles as unsigned [0,15]
    a_low_u8x64 = _mm512_and_si512(a_u4_vec, nibble_mask_u8x64);
    a_high_u8x64 = _mm512_and_si512(_mm512_srli_epi16(a_u4_vec, 4), nibble_mask_u8x64);
    b_low_u8x64 = _mm512_and_si512(b_u4_vec, nibble_mask_u8x64);
    b_high_u8x64 = _mm512_and_si512(_mm512_srli_epi16(b_u4_vec, 4), nibble_mask_u8x64);

    // Dot product with DPBUSD (safe for unsigned [0,15])
    ab_i32x16 = _mm512_dpbusd_epi32(ab_i32x16, a_low_u8x64, b_low_u8x64);
    ab_i32x16 = _mm512_dpbusd_epi32(ab_i32x16, a_high_u8x64, b_high_u8x64);

    // Squared norms: compute a² per nibble using lookup table for efficiency
    // Squares lookup: 0->0, 1->1, 2->4, ..., 15->225
    __m512i const u4_squares_lookup_u8x64 = _mm512_set_epi8(
        (char)225, (char)196, (char)169, (char)144, 121, 100, 81, 64, 49, 36, 25, 16, 9, 4, 1, 0, //
        (char)225, (char)196, (char)169, (char)144, 121, 100, 81, 64, 49, 36, 25, 16, 9, 4, 1, 0, //
        (char)225, (char)196, (char)169, (char)144, 121, 100, 81, 64, 49, 36, 25, 16, 9, 4, 1, 0, //
        (char)225, (char)196, (char)169, (char)144, 121, 100, 81, 64, 49, 36, 25, 16, 9, 4, 1, 0);

    __m512i a2_lo_u8x64 = _mm512_shuffle_epi8(u4_squares_lookup_u8x64, a_low_u8x64);
    __m512i a2_hi_u8x64 = _mm512_shuffle_epi8(u4_squares_lookup_u8x64, a_high_u8x64);
    __m512i b2_lo_u8x64 = _mm512_shuffle_epi8(u4_squares_lookup_u8x64, b_low_u8x64);
    __m512i b2_hi_u8x64 = _mm512_shuffle_epi8(u4_squares_lookup_u8x64, b_high_u8x64);

    // Accumulate low and high squares separately using SAD to avoid u8 overflow
    a2_i64x8 = _mm512_add_epi64(a2_i64x8, _mm512_sad_epu8(a2_lo_u8x64, zeros_i8x64));
    a2_i64x8 = _mm512_add_epi64(a2_i64x8, _mm512_sad_epu8(a2_hi_u8x64, zeros_i8x64));
    b2_i64x8 = _mm512_add_epi64(b2_i64x8, _mm512_sad_epu8(b2_lo_u8x64, zeros_i8x64));
    b2_i64x8 = _mm512_add_epi64(b2_i64x8, _mm512_sad_epu8(b2_hi_u8x64, zeros_i8x64));
    if (n_bytes) goto nk_angular_u4_ice_cycle;

    nk_i32_t ab = _mm512_reduce_add_epi32(ab_i32x16);
    nk_i64_t a2 = _mm512_reduce_add_epi64(a2_i64x8);
    nk_i64_t b2 = _mm512_reduce_add_epi64(b2_i64x8);
    *result = nk_angular_normalize_f32_haswell_(ab, (nk_f32_t)a2, (nk_f32_t)b2);
}

typedef nk_dot_i8x32_state_ice_t nk_angular_i8x32_state_ice_t;
NK_INTERNAL void nk_angular_i8x32_init_ice(nk_angular_i8x32_state_ice_t *state) { nk_dot_i8x32_init_ice(state); }
NK_INTERNAL void nk_angular_i8x32_update_ice(nk_angular_i8x32_state_ice_t *state, nk_b256_vec_t a, nk_b256_vec_t b) {
    nk_dot_i8x32_update_ice(state, a, b);
}
NK_INTERNAL void nk_angular_i8x32_finalize_ice(nk_angular_i8x32_state_ice_t const *state_a,
                                               nk_angular_i8x32_state_ice_t const *state_b,
                                               nk_angular_i8x32_state_ice_t const *state_c,
                                               nk_angular_i8x32_state_ice_t const *state_d, nk_f32_t query_norm,
                                               nk_f32_t target_norm_a, nk_f32_t target_norm_b, nk_f32_t target_norm_c,
                                               nk_f32_t target_norm_d, nk_f32_t *results) {
    nk_b128_vec_t dots_vec;
    nk_dot_i8x32_finalize_ice(state_a, state_b, state_c, state_d, &dots_vec);
    nk_angular_f32x4_finalize_haswell_(_mm_cvtepi32_ps(dots_vec.xmm), query_norm, target_norm_a, target_norm_b,
                                       target_norm_c, target_norm_d, results);
}

typedef nk_dot_i8x32_state_ice_t nk_l2_i8x32_state_ice_t;
NK_INTERNAL void nk_l2_i8x32_init_ice(nk_l2_i8x32_state_ice_t *state) { nk_dot_i8x32_init_ice(state); }
NK_INTERNAL void nk_l2_i8x32_update_ice(nk_l2_i8x32_state_ice_t *state, nk_b256_vec_t a, nk_b256_vec_t b) {
    nk_dot_i8x32_update_ice(state, a, b);
}
NK_INTERNAL void nk_l2_i8x32_finalize_ice(nk_l2_i8x32_state_ice_t const *state_a,
                                          nk_l2_i8x32_state_ice_t const *state_b,
                                          nk_l2_i8x32_state_ice_t const *state_c,
                                          nk_l2_i8x32_state_ice_t const *state_d, nk_f32_t query_norm,
                                          nk_f32_t target_norm_a, nk_f32_t target_norm_b, nk_f32_t target_norm_c,
                                          nk_f32_t target_norm_d, nk_f32_t *results) {
    nk_b128_vec_t dots_vec;
    nk_dot_i8x32_finalize_ice(state_a, state_b, state_c, state_d, &dots_vec);
    nk_l2_f32x4_finalize_haswell_(_mm_cvtepi32_ps(dots_vec.xmm), query_norm, target_norm_a, target_norm_b,
                                  target_norm_c, target_norm_d, results);
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
    nk_b128_vec_t dots_vec;
    nk_dot_u8x64_finalize_ice(state_a, state_b, state_c, state_d, &dots_vec);
    nk_angular_f32x4_finalize_haswell_(_mm_cvtepi32_ps(dots_vec.xmm), query_norm, target_norm_a, target_norm_b,
                                       target_norm_c, target_norm_d, results);
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
    nk_b128_vec_t dots_vec;
    nk_dot_u8x64_finalize_ice(state_a, state_b, state_c, state_d, &dots_vec);
    nk_l2_f32x4_finalize_haswell_(_mm_cvtepi32_ps(dots_vec.xmm), query_norm, target_norm_a, target_norm_b,
                                  target_norm_c, target_norm_d, results);
}

#if defined(__cplusplus)
} // extern "C"
#endif

#pragma clang attribute pop
#pragma GCC pop_options
#endif // NK_TARGET_ICE
#endif // NK_TARGET_X86_

#endif // NK_SPATIAL_ICE_H

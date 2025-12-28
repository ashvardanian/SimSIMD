/**
 *  @brief SIMD-accelerated Spatial Similarity Measures optimized for Intel Haswell CPUs.
 *  @file include/numkong/spatial/haswell.h
 *  @sa include/numkong/spatial.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 */
#ifndef NK_SPATIAL_HASWELL_H
#define NK_SPATIAL_HASWELL_H

#if NK_TARGET_X86_
#if NK_TARGET_HASWELL
#pragma GCC push_options
#pragma GCC target("avx2", "f16c", "fma")
#pragma clang attribute push(__attribute__((target("avx2,f16c,fma"))), apply_to = function)

#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

NK_INTERNAL nk_f32_t nk_sqrt_f32_haswell_(nk_f32_t x) { return _mm_cvtss_f32(_mm_sqrt_ps(_mm_set_ss(x))); }
NK_INTERNAL nk_f64_t nk_sqrt_f64_haswell_(nk_f64_t x) { return _mm_cvtsd_f64(_mm_sqrt_pd(_mm_set_sd(x))); }

NK_INTERNAL nk_f64_t nk_angular_normalize_f64_haswell_(nk_f64_t ab, nk_f64_t a2, nk_f64_t b2) {

    // If both vectors have magnitude 0, the distance is 0.
    if (a2 == 0 && b2 == 0) return 0;
    // If any one of the vectors is 0, the square root of the product is 0,
    // the division is illformed, and the result is 1.
    else if (ab == 0) return 1;
    // We want to avoid the `nk_f32_approximate_inverse_square_root` due to high latency:
    // https://web.archive.org/web/20210208132927/http://assemblyrequired.crashworks.org/timing-square-root/
    // The latency of the native instruction is 4 cycles and it's broadly supported.
    // For single-precision floats it has a maximum relative error of 1.5*2^-12.
    // Higher precision isn't implemented on older CPUs. See `nk_angular_normalize_f64_skylake_` for that.
    __m128d squares = _mm_set_pd(a2, b2);
    __m128d rsqrts = _mm_cvtps_pd(_mm_rsqrt_ps(_mm_cvtpd_ps(squares)));
    // Newton-Raphson iteration for reciprocal square root:
    // https://en.wikipedia.org/wiki/Newton%27s_method
    rsqrts = _mm_add_pd( //
        _mm_mul_pd(_mm_set1_pd(1.5), rsqrts),
        _mm_mul_pd(_mm_mul_pd(_mm_mul_pd(squares, _mm_set1_pd(-0.5)), rsqrts), _mm_mul_pd(rsqrts, rsqrts)));

    nk_f64_t a2_reciprocal = _mm_cvtsd_f64(_mm_unpackhi_pd(rsqrts, rsqrts));
    nk_f64_t b2_reciprocal = _mm_cvtsd_f64(rsqrts);
    nk_f64_t result = 1 - ab * a2_reciprocal * b2_reciprocal;
    return result > 0 ? result : 0;
}

NK_INTERNAL nk_f32_t nk_angular_normalize_f32_haswell_(nk_f32_t ab, nk_f32_t a2, nk_f32_t b2) {

    // If both vectors have magnitude 0, the distance is 0.
    if (a2 == 0.0f && b2 == 0.0f) return 0.0f;
    // If any one of the vectors is 0, the square root of the product is 0,
    // the division is illformed, and the result is 1.
    else if (ab == 0.0f) return 1.0f;

    // Load the squares into an __m128 register for single-precision floating-point operations
    __m128 squares = _mm_set_ps(a2, b2, a2, b2); // We replicate to make use of full register

    // Compute the reciprocal square root of the squares using `_mm_rsqrt_ps` (single-precision)
    __m128 rsqrts = _mm_rsqrt_ps(squares);

    // Perform one iteration of Newton-Raphson refinement to improve the precision of rsqrt:
    // Formula: y' = y * (1.5 - 0.5 * x * y * y)
    __m128 half = _mm_set1_ps(0.5f);
    __m128 three_halves = _mm_set1_ps(1.5f);
    rsqrts = _mm_mul_ps(rsqrts,
                        _mm_sub_ps(three_halves, _mm_mul_ps(half, _mm_mul_ps(squares, _mm_mul_ps(rsqrts, rsqrts)))));

    // Extract the reciprocal square roots of a2 and b2 from the __m128 register
    nk_f32_t a2_reciprocal = _mm_cvtss_f32(_mm_shuffle_ps(rsqrts, rsqrts, _MM_SHUFFLE(0, 0, 0, 1)));
    nk_f32_t b2_reciprocal = _mm_cvtss_f32(rsqrts);

    // Calculate the angular distance: 1 - dot_product * a2_reciprocal * b2_reciprocal
    nk_f32_t result = 1.0f - ab * a2_reciprocal * b2_reciprocal;
    return result > 0 ? result : 0;
}

#pragma clang attribute pop
#pragma GCC pop_options
#endif // NK_TARGET_HASWELL
#endif // NK_TARGET_X86_

#if NK_TARGET_X86_
#if NK_TARGET_HASWELL
#pragma GCC push_options
#pragma GCC target("avx2", "f16c", "fma")
#pragma clang attribute push(__attribute__((target("avx2,f16c,fma"))), apply_to = function)

NK_PUBLIC void nk_l2_f16_haswell(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_l2sq_f16_haswell(a, b, n, result);
    *result = nk_sqrt_f32_haswell_(*result);
}
NK_PUBLIC void nk_l2sq_f16_haswell(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result) {
    __m256 a_f32x8, b_f32x8;
    __m256 distance_sq_f32x8 = _mm256_setzero_ps();

nk_l2sq_f16_haswell_cycle:
    if (n < 8) {
        a_f32x8 = nk_partial_load_f16x8_to_f32x8_haswell_(a, n);
        b_f32x8 = nk_partial_load_f16x8_to_f32x8_haswell_(b, n);
        n = 0;
    }
    else {
        a_f32x8 = _mm256_cvtph_ps(_mm_lddqu_si128((__m128i const *)a));
        b_f32x8 = _mm256_cvtph_ps(_mm_lddqu_si128((__m128i const *)b));
        n -= 8, a += 8, b += 8;
    }
    __m256 diff_f32x8 = _mm256_sub_ps(a_f32x8, b_f32x8);
    distance_sq_f32x8 = _mm256_fmadd_ps(diff_f32x8, diff_f32x8, distance_sq_f32x8);
    if (n) goto nk_l2sq_f16_haswell_cycle;

    *result = nk_reduce_add_f32x8_haswell_(distance_sq_f32x8);
}

NK_PUBLIC void nk_angular_f16_haswell(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result) {
    __m256 a_f32x8, b_f32x8;
    __m256 dot_product_f32x8 = _mm256_setzero_ps(), a_norm_sq_f32x8 = _mm256_setzero_ps(),
           b_norm_sq_f32x8 = _mm256_setzero_ps();

nk_angular_f16_haswell_cycle:
    if (n < 8) {
        a_f32x8 = nk_partial_load_f16x8_to_f32x8_haswell_(a, n);
        b_f32x8 = nk_partial_load_f16x8_to_f32x8_haswell_(b, n);
        n = 0;
    }
    else {
        a_f32x8 = _mm256_cvtph_ps(_mm_lddqu_si128((__m128i const *)a));
        b_f32x8 = _mm256_cvtph_ps(_mm_lddqu_si128((__m128i const *)b));
        n -= 8, a += 8, b += 8;
    }
    dot_product_f32x8 = _mm256_fmadd_ps(a_f32x8, b_f32x8, dot_product_f32x8);
    a_norm_sq_f32x8 = _mm256_fmadd_ps(a_f32x8, a_f32x8, a_norm_sq_f32x8);
    b_norm_sq_f32x8 = _mm256_fmadd_ps(b_f32x8, b_f32x8, b_norm_sq_f32x8);
    if (n) goto nk_angular_f16_haswell_cycle;

    nk_f32_t dot_product_f32 = nk_reduce_add_f32x8_haswell_(dot_product_f32x8);
    nk_f32_t a_norm_sq_f32 = nk_reduce_add_f32x8_haswell_(a_norm_sq_f32x8);
    nk_f32_t b_norm_sq_f32 = nk_reduce_add_f32x8_haswell_(b_norm_sq_f32x8);
    *result = nk_angular_normalize_f32_haswell_(dot_product_f32, a_norm_sq_f32, b_norm_sq_f32);
}

NK_PUBLIC void nk_l2_bf16_haswell(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_l2sq_bf16_haswell(a, b, n, result);
    *result = nk_sqrt_f32_haswell_(*result);
}
NK_PUBLIC void nk_l2sq_bf16_haswell(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result) {
    __m256 a_f32x8, b_f32x8;
    __m256 distance_sq_f32x8 = _mm256_setzero_ps();

nk_l2sq_bf16_haswell_cycle:
    if (n < 8) {
        a_f32x8 = nk_partial_load_bf16x8_to_f32x8_haswell_(a, n);
        b_f32x8 = nk_partial_load_bf16x8_to_f32x8_haswell_(b, n);
        n = 0;
    }
    else {
        a_f32x8 = nk_bf16x8_to_f32x8_haswell_(_mm_lddqu_si128((__m128i const *)a));
        b_f32x8 = nk_bf16x8_to_f32x8_haswell_(_mm_lddqu_si128((__m128i const *)b));
        n -= 8, a += 8, b += 8;
    }
    __m256 diff_f32x8 = _mm256_sub_ps(a_f32x8, b_f32x8);
    distance_sq_f32x8 = _mm256_fmadd_ps(diff_f32x8, diff_f32x8, distance_sq_f32x8);
    if (n) goto nk_l2sq_bf16_haswell_cycle;

    *result = nk_reduce_add_f32x8_haswell_(distance_sq_f32x8);
}

NK_PUBLIC void nk_angular_bf16_haswell(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result) {
    __m256 a_f32x8, b_f32x8;
    __m256 dot_product_f32x8 = _mm256_setzero_ps(), a_norm_sq_f32x8 = _mm256_setzero_ps(),
           b_norm_sq_f32x8 = _mm256_setzero_ps();

nk_angular_bf16_haswell_cycle:
    if (n < 8) {
        a_f32x8 = nk_partial_load_bf16x8_to_f32x8_haswell_(a, n);
        b_f32x8 = nk_partial_load_bf16x8_to_f32x8_haswell_(b, n);
        n = 0;
    }
    else {
        a_f32x8 = nk_bf16x8_to_f32x8_haswell_(_mm_lddqu_si128((__m128i const *)a));
        b_f32x8 = nk_bf16x8_to_f32x8_haswell_(_mm_lddqu_si128((__m128i const *)b));
        n -= 8, a += 8, b += 8;
    }
    dot_product_f32x8 = _mm256_fmadd_ps(a_f32x8, b_f32x8, dot_product_f32x8);
    a_norm_sq_f32x8 = _mm256_fmadd_ps(a_f32x8, a_f32x8, a_norm_sq_f32x8);
    b_norm_sq_f32x8 = _mm256_fmadd_ps(b_f32x8, b_f32x8, b_norm_sq_f32x8);
    if (n) goto nk_angular_bf16_haswell_cycle;

    nk_f32_t dot_product_f32 = nk_reduce_add_f32x8_haswell_(dot_product_f32x8);
    nk_f32_t a_norm_sq_f32 = nk_reduce_add_f32x8_haswell_(a_norm_sq_f32x8);
    nk_f32_t b_norm_sq_f32 = nk_reduce_add_f32x8_haswell_(b_norm_sq_f32x8);
    *result = nk_angular_normalize_f32_haswell_(dot_product_f32, a_norm_sq_f32, b_norm_sq_f32);
}

NK_PUBLIC void nk_l2_i8_haswell(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_u32_t distance_sq_u32;
    nk_l2sq_i8_haswell(a, b, n, &distance_sq_u32);
    *result = nk_sqrt_f32_haswell_((nk_f32_t)distance_sq_u32);
}
NK_PUBLIC void nk_l2sq_i8_haswell(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_u32_t *result) {

    __m256i distance_sq_low_i32x8 = _mm256_setzero_si256();
    __m256i distance_sq_high_i32x8 = _mm256_setzero_si256();

    nk_size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        __m256i a_i8x32 = _mm256_lddqu_si256((__m256i const *)(a + i));
        __m256i b_i8x32 = _mm256_lddqu_si256((__m256i const *)(b + i));

        // Sign extend `i8` to `i16`
        __m256i a_low_i16x16 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(a_i8x32));
        __m256i a_high_i16x16 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(a_i8x32, 1));
        __m256i b_low_i16x16 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(b_i8x32));
        __m256i b_high_i16x16 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(b_i8x32, 1));

        // Subtract
        // After this we will be squaring the values. The sign will be dropped
        // and each difference will be in the range [0, 255].
        __m256i diff_low_i16x16 = _mm256_sub_epi16(a_low_i16x16, b_low_i16x16);
        __m256i diff_high_i16x16 = _mm256_sub_epi16(a_high_i16x16, b_high_i16x16);

        // Accumulate into `i32` vectors
        distance_sq_low_i32x8 = _mm256_add_epi32(distance_sq_low_i32x8,
                                                 _mm256_madd_epi16(diff_low_i16x16, diff_low_i16x16));
        distance_sq_high_i32x8 = _mm256_add_epi32(distance_sq_high_i32x8,
                                                  _mm256_madd_epi16(diff_high_i16x16, diff_high_i16x16));
    }

    // Accumulate the 32-bit integers from `distance_sq_high_i32x8` and `distance_sq_low_i32x8`
    nk_i32_t distance_sq_i32 = nk_reduce_add_i32x8_haswell_(
        _mm256_add_epi32(distance_sq_low_i32x8, distance_sq_high_i32x8));

    // Take care of the tail:
    for (; i < n; ++i) {
        nk_i32_t diff_i32 = (nk_i32_t)(a[i]) - b[i];
        distance_sq_i32 += diff_i32 * diff_i32;
    }

    *result = (nk_u32_t)distance_sq_i32;
}

NK_PUBLIC void nk_angular_i8_haswell(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_f32_t *result) {

    __m256i dot_product_low_i32x8 = _mm256_setzero_si256();
    __m256i dot_product_high_i32x8 = _mm256_setzero_si256();
    __m256i a_norm_sq_low_i32x8 = _mm256_setzero_si256();
    __m256i a_norm_sq_high_i32x8 = _mm256_setzero_si256();
    __m256i b_norm_sq_low_i32x8 = _mm256_setzero_si256();
    __m256i b_norm_sq_high_i32x8 = _mm256_setzero_si256();

    // AVX2 has no instructions for 8-bit signed integer dot-products,
    // but it has a weird instruction for mixed signed-unsigned 8-bit dot-product.
    // So we need to normalize the first vector to its absolute value,
    // and shift the product sign into the second vector.
    //
    //      __m256i a_i8_abs_vec = _mm256_abs_epi8(a_i8_vec);
    //      __m256i b_i8_flipped_vec = _mm256_sign_epi8(b_i8_vec, a_i8_vec);
    //      __m256i ab_i16_vec = _mm256_maddubs_epi16(a_i8_abs_vec, b_i8_flipped_vec);
    //
    // The problem with this approach, however, is the `-128` value in the second vector.
    // Flipping its sign will do nothing, and the result will be incorrect.
    // This can easily lead to noticeable numerical errors in the final result.
    nk_size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        __m256i a_i8x32 = _mm256_lddqu_si256((__m256i const *)(a + i));
        __m256i b_i8x32 = _mm256_lddqu_si256((__m256i const *)(b + i));

        // Unpack `int8` to `int16`
        __m256i a_low_i16x16 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(a_i8x32, 0));
        __m256i a_high_i16x16 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(a_i8x32, 1));
        __m256i b_low_i16x16 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(b_i8x32, 0));
        __m256i b_high_i16x16 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(b_i8x32, 1));

        // Multiply and accumulate as `int16`, accumulate products as `int32`:
        dot_product_low_i32x8 = _mm256_add_epi32(dot_product_low_i32x8, _mm256_madd_epi16(a_low_i16x16, b_low_i16x16));
        dot_product_high_i32x8 = _mm256_add_epi32(dot_product_high_i32x8,
                                                  _mm256_madd_epi16(a_high_i16x16, b_high_i16x16));
        a_norm_sq_low_i32x8 = _mm256_add_epi32(a_norm_sq_low_i32x8, _mm256_madd_epi16(a_low_i16x16, a_low_i16x16));
        a_norm_sq_high_i32x8 = _mm256_add_epi32(a_norm_sq_high_i32x8, _mm256_madd_epi16(a_high_i16x16, a_high_i16x16));
        b_norm_sq_low_i32x8 = _mm256_add_epi32(b_norm_sq_low_i32x8, _mm256_madd_epi16(b_low_i16x16, b_low_i16x16));
        b_norm_sq_high_i32x8 = _mm256_add_epi32(b_norm_sq_high_i32x8, _mm256_madd_epi16(b_high_i16x16, b_high_i16x16));
    }

    // Further reduce to a single sum for each vector
    nk_i32_t dot_product_i32 = nk_reduce_add_i32x8_haswell_(
        _mm256_add_epi32(dot_product_low_i32x8, dot_product_high_i32x8));
    nk_i32_t a_norm_sq_i32 = nk_reduce_add_i32x8_haswell_(_mm256_add_epi32(a_norm_sq_low_i32x8, a_norm_sq_high_i32x8));
    nk_i32_t b_norm_sq_i32 = nk_reduce_add_i32x8_haswell_(_mm256_add_epi32(b_norm_sq_low_i32x8, b_norm_sq_high_i32x8));

    // Take care of the tail:
    for (; i < n; ++i) {
        nk_i32_t a_element_i32 = a[i], b_element_i32 = b[i];
        dot_product_i32 += a_element_i32 * b_element_i32;
        a_norm_sq_i32 += a_element_i32 * a_element_i32;
        b_norm_sq_i32 += b_element_i32 * b_element_i32;
    }

    *result = nk_angular_normalize_f32_haswell_(dot_product_i32, a_norm_sq_i32, b_norm_sq_i32);
}

NK_PUBLIC void nk_l2_u8_haswell(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_u32_t distance_sq_u32;
    nk_l2sq_u8_haswell(a, b, n, &distance_sq_u32);
    *result = nk_sqrt_f32_haswell_((nk_f32_t)distance_sq_u32);
}
NK_PUBLIC void nk_l2sq_u8_haswell(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u32_t *result) {

    __m256i distance_sq_low_i32x8 = _mm256_setzero_si256();
    __m256i distance_sq_high_i32x8 = _mm256_setzero_si256();
    __m256i const zeros_i8x32 = _mm256_setzero_si256();

    nk_size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        __m256i a_u8x32 = _mm256_lddqu_si256((__m256i const *)(a + i));
        __m256i b_u8x32 = _mm256_lddqu_si256((__m256i const *)(b + i));

        // Substracting unsigned vectors in AVX2 is done by saturating subtraction:
        __m256i diff_u8x32 = _mm256_or_si256(_mm256_subs_epu8(a_u8x32, b_u8x32), _mm256_subs_epu8(b_u8x32, a_u8x32));

        // Upcast `uint8` to `int16`. Unlike the signed version, we can use the unpacking
        // instructions instead of extracts, as they are much faster and more efficient.
        __m256i diff_low_i16x16 = _mm256_unpacklo_epi8(diff_u8x32, zeros_i8x32);
        __m256i diff_high_i16x16 = _mm256_unpackhi_epi8(diff_u8x32, zeros_i8x32);

        // Multiply and accumulate at `int16` level, accumulate at `int32` level:
        distance_sq_low_i32x8 = _mm256_add_epi32(distance_sq_low_i32x8,
                                                 _mm256_madd_epi16(diff_low_i16x16, diff_low_i16x16));
        distance_sq_high_i32x8 = _mm256_add_epi32(distance_sq_high_i32x8,
                                                  _mm256_madd_epi16(diff_high_i16x16, diff_high_i16x16));
    }

    // Accumulate the 32-bit integers from `distance_sq_high_i32x8` and `distance_sq_low_i32x8`
    nk_i32_t distance_sq_i32 = nk_reduce_add_i32x8_haswell_(
        _mm256_add_epi32(distance_sq_low_i32x8, distance_sq_high_i32x8));

    // Take care of the tail:
    for (; i < n; ++i) {
        nk_i32_t diff_i32 = (nk_i32_t)(a[i]) - b[i];
        distance_sq_i32 += diff_i32 * diff_i32;
    }

    *result = (nk_u32_t)distance_sq_i32;
}

NK_PUBLIC void nk_angular_u8_haswell(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_f32_t *result) {

    __m256i dot_product_low_i32x8 = _mm256_setzero_si256();
    __m256i dot_product_high_i32x8 = _mm256_setzero_si256();
    __m256i a_norm_sq_low_i32x8 = _mm256_setzero_si256();
    __m256i a_norm_sq_high_i32x8 = _mm256_setzero_si256();
    __m256i b_norm_sq_low_i32x8 = _mm256_setzero_si256();
    __m256i b_norm_sq_high_i32x8 = _mm256_setzero_si256();
    __m256i const zeros_i8x32 = _mm256_setzero_si256();

    // AVX2 has no instructions for 8-bit signed integer dot-products,
    // but it has a weird instruction for mixed signed-unsigned 8-bit dot-product.
    // So we need to normalize the first vector to its absolute value,
    // and shift the product sign into the second vector.
    //
    //      __m256i a_i8_abs_vec = _mm256_abs_epi8(a_i8_vec);
    //      __m256i b_i8_flipped_vec = _mm256_sign_epi8(b_i8_vec, a_i8_vec);
    //      __m256i ab_i16_vec = _mm256_maddubs_epi16(a_i8_abs_vec, b_i8_flipped_vec);
    //
    // The problem with this approach, however, is the `-128` value in the second vector.
    // Flipping its sign will do nothing, and the result will be incorrect.
    // This can easily lead to noticeable numerical errors in the final result.
    nk_size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        __m256i a_u8x32 = _mm256_lddqu_si256((__m256i const *)(a + i));
        __m256i b_u8x32 = _mm256_lddqu_si256((__m256i const *)(b + i));

        // Upcast `uint8` to `int16`. Unlike the signed version, we can use the unpacking
        // instructions instead of extracts, as they are much faster and more efficient.
        __m256i a_low_i16x16 = _mm256_unpacklo_epi8(a_u8x32, zeros_i8x32);
        __m256i a_high_i16x16 = _mm256_unpackhi_epi8(a_u8x32, zeros_i8x32);
        __m256i b_low_i16x16 = _mm256_unpacklo_epi8(b_u8x32, zeros_i8x32);
        __m256i b_high_i16x16 = _mm256_unpackhi_epi8(b_u8x32, zeros_i8x32);

        // Multiply and accumulate as `int16`, accumulate products as `int32`
        dot_product_low_i32x8 = _mm256_add_epi32(dot_product_low_i32x8, _mm256_madd_epi16(a_low_i16x16, b_low_i16x16));
        dot_product_high_i32x8 = _mm256_add_epi32(dot_product_high_i32x8,
                                                  _mm256_madd_epi16(a_high_i16x16, b_high_i16x16));
        a_norm_sq_low_i32x8 = _mm256_add_epi32(a_norm_sq_low_i32x8, _mm256_madd_epi16(a_low_i16x16, a_low_i16x16));
        a_norm_sq_high_i32x8 = _mm256_add_epi32(a_norm_sq_high_i32x8, _mm256_madd_epi16(a_high_i16x16, a_high_i16x16));
        b_norm_sq_low_i32x8 = _mm256_add_epi32(b_norm_sq_low_i32x8, _mm256_madd_epi16(b_low_i16x16, b_low_i16x16));
        b_norm_sq_high_i32x8 = _mm256_add_epi32(b_norm_sq_high_i32x8, _mm256_madd_epi16(b_high_i16x16, b_high_i16x16));
    }

    // Further reduce to a single sum for each vector
    nk_i32_t dot_product_i32 = nk_reduce_add_i32x8_haswell_(
        _mm256_add_epi32(dot_product_low_i32x8, dot_product_high_i32x8));
    nk_i32_t a_norm_sq_i32 = nk_reduce_add_i32x8_haswell_(_mm256_add_epi32(a_norm_sq_low_i32x8, a_norm_sq_high_i32x8));
    nk_i32_t b_norm_sq_i32 = nk_reduce_add_i32x8_haswell_(_mm256_add_epi32(b_norm_sq_low_i32x8, b_norm_sq_high_i32x8));

    // Take care of the tail:
    for (; i < n; ++i) {
        nk_i32_t a_element_i32 = a[i], b_element_i32 = b[i];
        dot_product_i32 += a_element_i32 * b_element_i32;
        a_norm_sq_i32 += a_element_i32 * a_element_i32;
        b_norm_sq_i32 += b_element_i32 * b_element_i32;
    }

    *result = nk_angular_normalize_f32_haswell_(dot_product_i32, a_norm_sq_i32, b_norm_sq_i32);
}

NK_PUBLIC void nk_l2_f32_haswell(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_l2sq_f32_haswell(a, b, n, result);
    *result = nk_sqrt_f32_haswell_(*result);
}
NK_PUBLIC void nk_l2sq_f32_haswell(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *result) {

    __m256 distance_sq_f32x8 = _mm256_setzero_ps();
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 a_f32x8 = _mm256_loadu_ps(a + i);
        __m256 b_f32x8 = _mm256_loadu_ps(b + i);
        __m256 diff_f32x8 = _mm256_sub_ps(a_f32x8, b_f32x8);
        distance_sq_f32x8 = _mm256_fmadd_ps(diff_f32x8, diff_f32x8, distance_sq_f32x8);
    }

    nk_f64_t distance_sq_f64 = nk_reduce_add_f32x8_haswell_(distance_sq_f32x8);
    for (; i < n; ++i) {
        nk_f32_t diff_f32 = a[i] - b[i];
        distance_sq_f64 += diff_f32 * diff_f32;
    }

    *result = distance_sq_f64;
}

NK_PUBLIC void nk_angular_f32_haswell(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *result) {

    __m256 dot_product_f32x8 = _mm256_setzero_ps();
    __m256 a_norm_sq_f32x8 = _mm256_setzero_ps();
    __m256 b_norm_sq_f32x8 = _mm256_setzero_ps();
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 a_f32x8 = _mm256_loadu_ps(a + i);
        __m256 b_f32x8 = _mm256_loadu_ps(b + i);
        dot_product_f32x8 = _mm256_fmadd_ps(a_f32x8, b_f32x8, dot_product_f32x8);
        a_norm_sq_f32x8 = _mm256_fmadd_ps(a_f32x8, a_f32x8, a_norm_sq_f32x8);
        b_norm_sq_f32x8 = _mm256_fmadd_ps(b_f32x8, b_f32x8, b_norm_sq_f32x8);
    }

    nk_f32_t dot_product_f32 = nk_reduce_add_f32x8_haswell_(dot_product_f32x8);
    nk_f32_t a_norm_sq_f32 = nk_reduce_add_f32x8_haswell_(a_norm_sq_f32x8);
    nk_f32_t b_norm_sq_f32 = nk_reduce_add_f32x8_haswell_(b_norm_sq_f32x8);
    for (; i < n; ++i) {
        nk_f32_t a_element_f32 = a[i], b_element_f32 = b[i];
        dot_product_f32 += a_element_f32 * b_element_f32;
        a_norm_sq_f32 += a_element_f32 * a_element_f32;
        b_norm_sq_f32 += b_element_f32 * b_element_f32;
    }
    *result = nk_angular_normalize_f32_haswell_(dot_product_f32, a_norm_sq_f32, b_norm_sq_f32);
}

NK_PUBLIC void nk_l2_f64_haswell(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result) {
    nk_l2sq_f64_haswell(a, b, n, result);
    *result = nk_sqrt_f64_haswell_(*result);
}
NK_PUBLIC void nk_l2sq_f64_haswell(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result) {

    __m256d distance_sq_f64x4 = _mm256_setzero_pd();
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d a_f64x4 = _mm256_loadu_pd(a + i);
        __m256d b_f64x4 = _mm256_loadu_pd(b + i);
        __m256d diff_f64x4 = _mm256_sub_pd(a_f64x4, b_f64x4);
        distance_sq_f64x4 = _mm256_fmadd_pd(diff_f64x4, diff_f64x4, distance_sq_f64x4);
    }

    nk_f64_t distance_sq_f64 = nk_reduce_add_f64x4_haswell_(distance_sq_f64x4);
    for (; i < n; ++i) {
        nk_f64_t diff_f64 = a[i] - b[i];
        distance_sq_f64 += diff_f64 * diff_f64;
    }

    *result = distance_sq_f64;
}

NK_PUBLIC void nk_angular_f64_haswell(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result) {

    __m256d dot_product_f64x4 = _mm256_setzero_pd();
    __m256d a_norm_sq_f64x4 = _mm256_setzero_pd();
    __m256d b_norm_sq_f64x4 = _mm256_setzero_pd();
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d a_f64x4 = _mm256_loadu_pd(a + i);
        __m256d b_f64x4 = _mm256_loadu_pd(b + i);
        dot_product_f64x4 = _mm256_fmadd_pd(a_f64x4, b_f64x4, dot_product_f64x4);
        a_norm_sq_f64x4 = _mm256_fmadd_pd(a_f64x4, a_f64x4, a_norm_sq_f64x4);
        b_norm_sq_f64x4 = _mm256_fmadd_pd(b_f64x4, b_f64x4, b_norm_sq_f64x4);
    }

    nk_f64_t dot_product_f64 = nk_reduce_add_f64x4_haswell_(dot_product_f64x4);
    nk_f64_t a_norm_sq_f64 = nk_reduce_add_f64x4_haswell_(a_norm_sq_f64x4);
    nk_f64_t b_norm_sq_f64 = nk_reduce_add_f64x4_haswell_(b_norm_sq_f64x4);
    for (; i < n; ++i) {
        nk_f64_t a_element_f64 = a[i], b_element_f64 = b[i];
        dot_product_f64 += a_element_f64 * b_element_f64;
        a_norm_sq_f64 += a_element_f64 * a_element_f64;
        b_norm_sq_f64 += b_element_f64 * b_element_f64;
    }
    *result = nk_angular_normalize_f64_haswell_(dot_product_f64, a_norm_sq_f64, b_norm_sq_f64);
}

typedef nk_dot_f32x8_state_haswell_t nk_angular_f32x8_state_haswell_t;
NK_INTERNAL void nk_angular_f32x8_init_haswell(nk_angular_f32x8_state_haswell_t *state) {
    nk_dot_f32x8_init_haswell(state);
}
NK_INTERNAL void nk_angular_f32x8_update_haswell(nk_angular_f32x8_state_haswell_t *state, nk_b256_vec_t a,
                                                 nk_b256_vec_t b) {
    nk_dot_f32x8_update_haswell(state, a, b);
}
NK_INTERNAL void nk_angular_f32x8_finalize_haswell(nk_angular_f32x8_state_haswell_t const *state_a,
                                                   nk_angular_f32x8_state_haswell_t const *state_b,
                                                   nk_angular_f32x8_state_haswell_t const *state_c,
                                                   nk_angular_f32x8_state_haswell_t const *state_d, nk_f32_t query_norm,
                                                   nk_f32_t target_norm_a, nk_f32_t target_norm_b,
                                                   nk_f32_t target_norm_c, nk_f32_t target_norm_d, nk_f32_t *results) {
    // Extract all 4 dot products with single ILP-optimized call
    nk_f32_t dots[4];
    nk_dot_f32x8_finalize_haswell(state_a, state_b, state_c, state_d, dots);

    // Build 256-bit F64 vectors for parallel processing
    __m256d dots_f64x4 = _mm256_set_pd((nk_f64_t)dots[3], (nk_f64_t)dots[2], //
                                       (nk_f64_t)dots[1], (nk_f64_t)dots[0]);
    __m256d query_norm_f64x4 = _mm256_set1_pd((nk_f64_t)query_norm);
    __m256d target_norms_f64x4 = _mm256_set_pd((nk_f64_t)target_norm_d, (nk_f64_t)target_norm_c,
                                               (nk_f64_t)target_norm_b, (nk_f64_t)target_norm_a);

    // Compute products = query_norm * target_norm for all 4
    __m256d products_f64x4 = _mm256_mul_pd(query_norm_f64x4, target_norms_f64x4);

    // Vectorized rsqrt with Newton-Raphson (approximate rsqrt not available for f64)
    // Use full sqrt: 1.0 - dot / sqrt(product)
    __m256d sqrt_products_f64x4 = _mm256_sqrt_pd(products_f64x4);
    __m256d normalized_f64x4 = _mm256_div_pd(dots_f64x4, sqrt_products_f64x4);
    __m256d ones_f64x4 = _mm256_set1_pd(1.0);
    __m256d angular_f64x4 = _mm256_sub_pd(ones_f64x4, normalized_f64x4);

    // Store results
    nk_b256_vec_t u;
    _mm256_storeu_pd(u.f64s, angular_f64x4);
    results[0] = (nk_f32_t)u.f64s[0];
    results[1] = (nk_f32_t)u.f64s[1];
    results[2] = (nk_f32_t)u.f64s[2];
    results[3] = (nk_f32_t)u.f64s[3];
}

typedef nk_dot_f32x8_state_haswell_t nk_l2_f32x8_state_haswell_t;
NK_INTERNAL void nk_l2_f32x8_init_haswell(nk_l2_f32x8_state_haswell_t *state) { nk_dot_f32x8_init_haswell(state); }
NK_INTERNAL void nk_l2_f32x8_update_haswell(nk_l2_f32x8_state_haswell_t *state, nk_b256_vec_t a, nk_b256_vec_t b) {
    nk_dot_f32x8_update_haswell(state, a, b);
}
NK_INTERNAL void nk_l2_f32x8_finalize_haswell(nk_l2_f32x8_state_haswell_t const *state_a,
                                              nk_l2_f32x8_state_haswell_t const *state_b,
                                              nk_l2_f32x8_state_haswell_t const *state_c,
                                              nk_l2_f32x8_state_haswell_t const *state_d, nk_f32_t query_norm,
                                              nk_f32_t target_norm_a, nk_f32_t target_norm_b, nk_f32_t target_norm_c,
                                              nk_f32_t target_norm_d, nk_f32_t *results) {
    // Extract all 4 dot products with single ILP-optimized call
    nk_f32_t dots[4];
    nk_dot_f32x8_finalize_haswell(state_a, state_b, state_c, state_d, dots);

    // Build 256-bit F64 vectors for parallel L2 distance: sqrt(q² + t² - 2*dot)
    __m256d dots_f64x4 = _mm256_set_pd((nk_f64_t)dots[3], (nk_f64_t)dots[2], //
                                       (nk_f64_t)dots[1], (nk_f64_t)dots[0]);
    __m256d query_norm_f64x4 = _mm256_set1_pd((nk_f64_t)query_norm);
    __m256d target_norms_f64x4 = _mm256_set_pd((nk_f64_t)target_norm_d, (nk_f64_t)target_norm_c,
                                               (nk_f64_t)target_norm_b, (nk_f64_t)target_norm_a);

    // Compute squared norms in parallel: q² and t² vectors
    __m256d query_sq_f64x4 = _mm256_mul_pd(query_norm_f64x4, query_norm_f64x4);
    __m256d target_sq_f64x4 = _mm256_mul_pd(target_norms_f64x4, target_norms_f64x4);

    // Compute distance squared: q² + t² - 2*dot using FMA
    __m256d two_f64x4 = _mm256_set1_pd(2.0);
    __m256d sum_sq_f64x4 = _mm256_add_pd(query_sq_f64x4, target_sq_f64x4);
    __m256d dist_sq_f64x4 = _mm256_fnmadd_pd(two_f64x4, dots_f64x4, sum_sq_f64x4);

    // Clamp negative to zero, then sqrt
    __m256d zeros_f64x4 = _mm256_setzero_pd();
    __m256d clamped_f64x4 = _mm256_max_pd(dist_sq_f64x4, zeros_f64x4);
    __m256d dist_f64x4 = _mm256_sqrt_pd(clamped_f64x4);

    // Store results (convert f64 → f32)
    nk_b256_vec_t u;
    _mm256_storeu_pd(u.f64s, dist_f64x4);
    results[0] = (nk_f32_t)u.f64s[0];
    results[1] = (nk_f32_t)u.f64s[1];
    results[2] = (nk_f32_t)u.f64s[2];
    results[3] = (nk_f32_t)u.f64s[3];
}

typedef nk_dot_f16x16_state_haswell_t nk_angular_f16x16_state_haswell_t;
NK_INTERNAL void nk_angular_f16x16_init_haswell(nk_angular_f16x16_state_haswell_t *state) {
    nk_dot_f16x16_init_haswell(state);
}
NK_INTERNAL void nk_angular_f16x16_update_haswell(nk_angular_f16x16_state_haswell_t *state, nk_b256_vec_t a,
                                                  nk_b256_vec_t b) {
    nk_dot_f16x16_update_haswell(state, a, b);
}
NK_INTERNAL void nk_angular_f16x16_finalize_haswell(nk_angular_f16x16_state_haswell_t const *state_a,
                                                    nk_angular_f16x16_state_haswell_t const *state_b,
                                                    nk_angular_f16x16_state_haswell_t const *state_c,
                                                    nk_angular_f16x16_state_haswell_t const *state_d,
                                                    nk_f32_t query_norm, nk_f32_t target_norm_a, nk_f32_t target_norm_b,
                                                    nk_f32_t target_norm_c, nk_f32_t target_norm_d, nk_f32_t *results) {
    // Extract all 4 dot products with single ILP-optimized call
    nk_f32_t dots[4];
    nk_dot_f16x16_finalize_haswell(state_a, state_b, state_c, state_d, dots);

    // Build 128-bit F32 vectors for parallel processing (F16 uses F32 accumulation)
    __m128 dots_f32x4 = _mm_loadu_ps(dots);
    __m128 query_norm_f32x4 = _mm_set1_ps(query_norm);
    __m128 target_norms_f32x4 = _mm_set_ps(target_norm_d, target_norm_c, target_norm_b, target_norm_a);

    // Compute products = query_norm * target_norm for all 4
    __m128 products_f32x4 = _mm_mul_ps(query_norm_f32x4, target_norms_f32x4);

    // Vectorized rsqrt with Newton-Raphson refinement for F32
    __m128 rsqrt_f32x4 = _mm_rsqrt_ps(products_f32x4);
    __m128 half_f32x4 = _mm_set1_ps(0.5f);
    __m128 three_f32x4 = _mm_set1_ps(3.0f);
    __m128 nr_f32x4 = _mm_mul_ps(products_f32x4, rsqrt_f32x4);
    nr_f32x4 = _mm_mul_ps(nr_f32x4, rsqrt_f32x4);
    nr_f32x4 = _mm_sub_ps(three_f32x4, nr_f32x4);
    rsqrt_f32x4 = _mm_mul_ps(_mm_mul_ps(half_f32x4, rsqrt_f32x4), nr_f32x4);

    // Compute angular: 1 - dot * rsqrt(product)
    __m128 normalized_f32x4 = _mm_mul_ps(dots_f32x4, rsqrt_f32x4);
    __m128 ones_f32x4 = _mm_set1_ps(1.0f);
    __m128 angular_f32x4 = _mm_sub_ps(ones_f32x4, normalized_f32x4);

    // Store results
    _mm_storeu_ps(results, angular_f32x4);
}

typedef nk_dot_f16x16_state_haswell_t nk_l2_f16x16_state_haswell_t;
NK_INTERNAL void nk_l2_f16x16_init_haswell(nk_l2_f16x16_state_haswell_t *state) { nk_dot_f16x16_init_haswell(state); }
NK_INTERNAL void nk_l2_f16x16_update_haswell(nk_l2_f16x16_state_haswell_t *state, nk_b256_vec_t a, nk_b256_vec_t b) {
    nk_dot_f16x16_update_haswell(state, a, b);
}
NK_INTERNAL void nk_l2_f16x16_finalize_haswell(nk_l2_f16x16_state_haswell_t const *state_a,
                                               nk_l2_f16x16_state_haswell_t const *state_b,
                                               nk_l2_f16x16_state_haswell_t const *state_c,
                                               nk_l2_f16x16_state_haswell_t const *state_d, nk_f32_t query_norm,
                                               nk_f32_t target_norm_a, nk_f32_t target_norm_b, nk_f32_t target_norm_c,
                                               nk_f32_t target_norm_d, nk_f32_t *results) {
    // Extract all 4 dot products with single ILP-optimized call
    nk_f32_t dots[4];
    nk_dot_f16x16_finalize_haswell(state_a, state_b, state_c, state_d, dots);

    // Build 128-bit F32 vectors for parallel L2 distance: sqrt(q² + t² - 2*dot)
    __m128 dots_f32x4 = _mm_loadu_ps(dots);
    __m128 query_norm_f32x4 = _mm_set1_ps(query_norm);
    __m128 target_norms_f32x4 = _mm_set_ps(target_norm_d, target_norm_c, target_norm_b, target_norm_a);

    // Compute squared norms in parallel
    __m128 query_sq_f32x4 = _mm_mul_ps(query_norm_f32x4, query_norm_f32x4);
    __m128 target_sq_f32x4 = _mm_mul_ps(target_norms_f32x4, target_norms_f32x4);

    // Compute distance squared: q² + t² - 2*dot using FMA
    __m128 two_f32x4 = _mm_set1_ps(2.0f);
    __m128 sum_sq_f32x4 = _mm_add_ps(query_sq_f32x4, target_sq_f32x4);
    __m128 dist_sq_f32x4 = _mm_fnmadd_ps(two_f32x4, dots_f32x4, sum_sq_f32x4);

    // Clamp negative to zero, then sqrt
    __m128 zeros_f32x4 = _mm_setzero_ps();
    __m128 clamped_f32x4 = _mm_max_ps(dist_sq_f32x4, zeros_f32x4);
    __m128 dist_f32x4 = _mm_sqrt_ps(clamped_f32x4);

    // Store results
    _mm_storeu_ps(results, dist_f32x4);
}

typedef nk_dot_bf16x16_state_haswell_t nk_angular_bf16x16_state_haswell_t;
NK_INTERNAL void nk_angular_bf16x16_init_haswell(nk_angular_bf16x16_state_haswell_t *state) {
    nk_dot_bf16x16_init_haswell(state);
}
NK_INTERNAL void nk_angular_bf16x16_update_haswell(nk_angular_bf16x16_state_haswell_t *state, nk_b256_vec_t a,
                                                   nk_b256_vec_t b) {
    nk_dot_bf16x16_update_haswell(state, a, b);
}
NK_INTERNAL void nk_angular_bf16x16_finalize_haswell(nk_angular_bf16x16_state_haswell_t const *state_a,
                                                     nk_angular_bf16x16_state_haswell_t const *state_b,
                                                     nk_angular_bf16x16_state_haswell_t const *state_c,
                                                     nk_angular_bf16x16_state_haswell_t const *state_d,
                                                     nk_f32_t query_norm, nk_f32_t target_norm_a,
                                                     nk_f32_t target_norm_b, nk_f32_t target_norm_c,
                                                     nk_f32_t target_norm_d, nk_f32_t *results) {
    // Extract all 4 dot products with single ILP-optimized call
    nk_f32_t dots[4];
    nk_dot_bf16x16_finalize_haswell(state_a, state_b, state_c, state_d, dots);

    // Build 128-bit F32 vectors for parallel processing (BF16 uses F32 accumulation)
    __m128 dots_f32x4 = _mm_loadu_ps(dots);
    __m128 query_norm_f32x4 = _mm_set1_ps(query_norm);
    __m128 target_norms_f32x4 = _mm_set_ps(target_norm_d, target_norm_c, target_norm_b, target_norm_a);

    // Compute products = query_norm * target_norm for all 4
    __m128 products_f32x4 = _mm_mul_ps(query_norm_f32x4, target_norms_f32x4);

    // Vectorized rsqrt with Newton-Raphson refinement for F32
    __m128 rsqrt_f32x4 = _mm_rsqrt_ps(products_f32x4);
    __m128 half_f32x4 = _mm_set1_ps(0.5f);
    __m128 three_f32x4 = _mm_set1_ps(3.0f);
    __m128 nr_f32x4 = _mm_mul_ps(products_f32x4, rsqrt_f32x4);
    nr_f32x4 = _mm_mul_ps(nr_f32x4, rsqrt_f32x4);
    nr_f32x4 = _mm_sub_ps(three_f32x4, nr_f32x4);
    rsqrt_f32x4 = _mm_mul_ps(_mm_mul_ps(half_f32x4, rsqrt_f32x4), nr_f32x4);

    // Compute angular: 1 - dot * rsqrt(product)
    __m128 normalized_f32x4 = _mm_mul_ps(dots_f32x4, rsqrt_f32x4);
    __m128 ones_f32x4 = _mm_set1_ps(1.0f);
    __m128 angular_f32x4 = _mm_sub_ps(ones_f32x4, normalized_f32x4);

    // Store results
    _mm_storeu_ps(results, angular_f32x4);
}

typedef nk_dot_bf16x16_state_haswell_t nk_l2_bf16x16_state_haswell_t;
NK_INTERNAL void nk_l2_bf16x16_init_haswell(nk_l2_bf16x16_state_haswell_t *state) {
    nk_dot_bf16x16_init_haswell(state);
}
NK_INTERNAL void nk_l2_bf16x16_update_haswell(nk_l2_bf16x16_state_haswell_t *state, nk_b256_vec_t a, nk_b256_vec_t b) {
    nk_dot_bf16x16_update_haswell(state, a, b);
}
NK_INTERNAL void nk_l2_bf16x16_finalize_haswell(nk_l2_bf16x16_state_haswell_t const *state_a,
                                                nk_l2_bf16x16_state_haswell_t const *state_b,
                                                nk_l2_bf16x16_state_haswell_t const *state_c,
                                                nk_l2_bf16x16_state_haswell_t const *state_d, nk_f32_t query_norm,
                                                nk_f32_t target_norm_a, nk_f32_t target_norm_b, nk_f32_t target_norm_c,
                                                nk_f32_t target_norm_d, nk_f32_t *results) {
    // Extract all 4 dot products with single ILP-optimized call
    nk_f32_t dots[4];
    nk_dot_bf16x16_finalize_haswell(state_a, state_b, state_c, state_d, dots);

    // Build 128-bit F32 vectors for parallel L2 distance: sqrt(q² + t² - 2*dot)
    __m128 dots_f32x4 = _mm_loadu_ps(dots);
    __m128 query_norm_f32x4 = _mm_set1_ps(query_norm);
    __m128 target_norms_f32x4 = _mm_set_ps(target_norm_d, target_norm_c, target_norm_b, target_norm_a);

    // Compute squared norms in parallel
    __m128 query_sq_f32x4 = _mm_mul_ps(query_norm_f32x4, query_norm_f32x4);
    __m128 target_sq_f32x4 = _mm_mul_ps(target_norms_f32x4, target_norms_f32x4);

    // Compute distance squared: q² + t² - 2*dot using FMA
    __m128 two_f32x4 = _mm_set1_ps(2.0f);
    __m128 sum_sq_f32x4 = _mm_add_ps(query_sq_f32x4, target_sq_f32x4);
    __m128 dist_sq_f32x4 = _mm_fnmadd_ps(two_f32x4, dots_f32x4, sum_sq_f32x4);

    // Clamp negative to zero, then sqrt
    __m128 zeros_f32x4 = _mm_setzero_ps();
    __m128 clamped_f32x4 = _mm_max_ps(dist_sq_f32x4, zeros_f32x4);
    __m128 dist_f32x4 = _mm_sqrt_ps(clamped_f32x4);

    // Store results
    _mm_storeu_ps(results, dist_f32x4);
}

typedef nk_dot_i8x32_state_haswell_t nk_angular_i8x32_state_haswell_t;
NK_INTERNAL void nk_angular_i8x32_init_haswell(nk_angular_i8x32_state_haswell_t *state) {
    nk_dot_i8x32_init_haswell(state);
}
NK_INTERNAL void nk_angular_i8x32_update_haswell(nk_angular_i8x32_state_haswell_t *state, nk_b256_vec_t a,
                                                 nk_b256_vec_t b) {
    nk_dot_i8x32_update_haswell(state, a, b);
}
NK_INTERNAL void nk_angular_i8x32_finalize_haswell(nk_angular_i8x32_state_haswell_t const *state_a,
                                                   nk_angular_i8x32_state_haswell_t const *state_b,
                                                   nk_angular_i8x32_state_haswell_t const *state_c,
                                                   nk_angular_i8x32_state_haswell_t const *state_d, nk_f32_t query_norm,
                                                   nk_f32_t target_norm_a, nk_f32_t target_norm_b,
                                                   nk_f32_t target_norm_c, nk_f32_t target_norm_d, nk_f32_t *results) {
    // Extract all 4 dot products with single ILP-optimized call (i32 output)
    nk_i32_t dots_i32[4];
    nk_dot_i8x32_finalize_haswell(state_a, state_b, state_c, state_d, dots_i32);

    // Build 128-bit F32 vectors for parallel processing (convert i32 → f32)
    __m128i dots_i32x4 = _mm_loadu_si128((__m128i const *)dots_i32);
    __m128 dots_f32x4 = _mm_cvtepi32_ps(dots_i32x4);
    __m128 query_norm_f32x4 = _mm_set1_ps(query_norm);
    __m128 target_norms_f32x4 = _mm_set_ps(target_norm_d, target_norm_c, target_norm_b, target_norm_a);

    // Compute products = query_norm * target_norm for all 4
    __m128 products_f32x4 = _mm_mul_ps(query_norm_f32x4, target_norms_f32x4);

    // Vectorized rsqrt with Newton-Raphson refinement
    __m128 rsqrt_f32x4 = _mm_rsqrt_ps(products_f32x4);
    __m128 half_f32x4 = _mm_set1_ps(0.5f);
    __m128 three_f32x4 = _mm_set1_ps(3.0f);
    __m128 nr_f32x4 = _mm_mul_ps(products_f32x4, rsqrt_f32x4);
    nr_f32x4 = _mm_mul_ps(nr_f32x4, rsqrt_f32x4);
    nr_f32x4 = _mm_sub_ps(three_f32x4, nr_f32x4);
    rsqrt_f32x4 = _mm_mul_ps(_mm_mul_ps(half_f32x4, rsqrt_f32x4), nr_f32x4);

    // Compute angular: 1 - dot * rsqrt(product)
    __m128 normalized_f32x4 = _mm_mul_ps(dots_f32x4, rsqrt_f32x4);
    __m128 ones_f32x4 = _mm_set1_ps(1.0f);
    __m128 angular_f32x4 = _mm_sub_ps(ones_f32x4, normalized_f32x4);

    // Store results
    _mm_storeu_ps(results, angular_f32x4);
}

typedef nk_dot_i8x32_state_haswell_t nk_l2_i8x32_state_haswell_t;
NK_INTERNAL void nk_l2_i8x32_init_haswell(nk_l2_i8x32_state_haswell_t *state) { nk_dot_i8x32_init_haswell(state); }
NK_INTERNAL void nk_l2_i8x32_update_haswell(nk_l2_i8x32_state_haswell_t *state, nk_b256_vec_t a, nk_b256_vec_t b) {
    nk_dot_i8x32_update_haswell(state, a, b);
}
NK_INTERNAL void nk_l2_i8x32_finalize_haswell(nk_l2_i8x32_state_haswell_t const *state_a,
                                              nk_l2_i8x32_state_haswell_t const *state_b,
                                              nk_l2_i8x32_state_haswell_t const *state_c,
                                              nk_l2_i8x32_state_haswell_t const *state_d, nk_f32_t query_norm,
                                              nk_f32_t target_norm_a, nk_f32_t target_norm_b, nk_f32_t target_norm_c,
                                              nk_f32_t target_norm_d, nk_f32_t *results) {
    // Extract all 4 dot products with single ILP-optimized call (i32 output)
    nk_i32_t dots_i32[4];
    nk_dot_i8x32_finalize_haswell(state_a, state_b, state_c, state_d, dots_i32);

    // Build 128-bit F32 vectors for parallel L2 distance: sqrt(q² + t² - 2*dot)
    __m128i dots_i32x4 = _mm_loadu_si128((__m128i const *)dots_i32);
    __m128 dots_f32x4 = _mm_cvtepi32_ps(dots_i32x4);
    __m128 query_norm_f32x4 = _mm_set1_ps(query_norm);
    __m128 target_norms_f32x4 = _mm_set_ps(target_norm_d, target_norm_c, target_norm_b, target_norm_a);

    // Compute squared norms in parallel
    __m128 query_sq_f32x4 = _mm_mul_ps(query_norm_f32x4, query_norm_f32x4);
    __m128 target_sq_f32x4 = _mm_mul_ps(target_norms_f32x4, target_norms_f32x4);

    // Compute distance squared: q² + t² - 2*dot using FMA
    __m128 two_f32x4 = _mm_set1_ps(2.0f);
    __m128 sum_sq_f32x4 = _mm_add_ps(query_sq_f32x4, target_sq_f32x4);
    __m128 dist_sq_f32x4 = _mm_fnmadd_ps(two_f32x4, dots_f32x4, sum_sq_f32x4);

    // Clamp negative to zero, then sqrt
    __m128 zeros_f32x4 = _mm_setzero_ps();
    __m128 clamped_f32x4 = _mm_max_ps(dist_sq_f32x4, zeros_f32x4);
    __m128 dist_f32x4 = _mm_sqrt_ps(clamped_f32x4);

    // Store results
    _mm_storeu_ps(results, dist_f32x4);
}

typedef nk_dot_u8x32_state_haswell_t nk_angular_u8x32_state_haswell_t;
NK_INTERNAL void nk_angular_u8x32_init_haswell(nk_angular_u8x32_state_haswell_t *state) {
    nk_dot_u8x32_init_haswell(state);
}
NK_INTERNAL void nk_angular_u8x32_update_haswell(nk_angular_u8x32_state_haswell_t *state, nk_b256_vec_t a,
                                                 nk_b256_vec_t b) {
    nk_dot_u8x32_update_haswell(state, a, b);
}
NK_INTERNAL void nk_angular_u8x32_finalize_haswell(nk_angular_u8x32_state_haswell_t const *state_a,
                                                   nk_angular_u8x32_state_haswell_t const *state_b,
                                                   nk_angular_u8x32_state_haswell_t const *state_c,
                                                   nk_angular_u8x32_state_haswell_t const *state_d, nk_f32_t query_norm,
                                                   nk_f32_t target_norm_a, nk_f32_t target_norm_b,
                                                   nk_f32_t target_norm_c, nk_f32_t target_norm_d, nk_f32_t *results) {
    // Extract all 4 dot products with single ILP-optimized call (u32 output)
    nk_u32_t dots_u32[4];
    nk_dot_u8x32_finalize_haswell(state_a, state_b, state_c, state_d, dots_u32);

    // Build 128-bit F32 vectors for parallel processing (convert u32 → f32 via i32)
    __m128i dots_i32x4 = _mm_loadu_si128((__m128i const *)dots_u32);
    __m128 dots_f32x4 = _mm_cvtepi32_ps(dots_i32x4);
    __m128 query_norm_f32x4 = _mm_set1_ps(query_norm);
    __m128 target_norms_f32x4 = _mm_set_ps(target_norm_d, target_norm_c, target_norm_b, target_norm_a);

    // Compute products = query_norm * target_norm for all 4
    __m128 products_f32x4 = _mm_mul_ps(query_norm_f32x4, target_norms_f32x4);

    // Vectorized rsqrt with Newton-Raphson refinement
    __m128 rsqrt_f32x4 = _mm_rsqrt_ps(products_f32x4);
    __m128 half_f32x4 = _mm_set1_ps(0.5f);
    __m128 three_f32x4 = _mm_set1_ps(3.0f);
    __m128 nr_f32x4 = _mm_mul_ps(products_f32x4, rsqrt_f32x4);
    nr_f32x4 = _mm_mul_ps(nr_f32x4, rsqrt_f32x4);
    nr_f32x4 = _mm_sub_ps(three_f32x4, nr_f32x4);
    rsqrt_f32x4 = _mm_mul_ps(_mm_mul_ps(half_f32x4, rsqrt_f32x4), nr_f32x4);

    // Compute angular: 1 - dot * rsqrt(product)
    __m128 normalized_f32x4 = _mm_mul_ps(dots_f32x4, rsqrt_f32x4);
    __m128 ones_f32x4 = _mm_set1_ps(1.0f);
    __m128 angular_f32x4 = _mm_sub_ps(ones_f32x4, normalized_f32x4);

    // Store results
    _mm_storeu_ps(results, angular_f32x4);
}

typedef nk_dot_u8x32_state_haswell_t nk_l2_u8x32_state_haswell_t;
NK_INTERNAL void nk_l2_u8x32_init_haswell(nk_l2_u8x32_state_haswell_t *state) { nk_dot_u8x32_init_haswell(state); }
NK_INTERNAL void nk_l2_u8x32_update_haswell(nk_l2_u8x32_state_haswell_t *state, nk_b256_vec_t a, nk_b256_vec_t b) {
    nk_dot_u8x32_update_haswell(state, a, b);
}
NK_INTERNAL void nk_l2_u8x32_finalize_haswell(nk_l2_u8x32_state_haswell_t const *state_a,
                                              nk_l2_u8x32_state_haswell_t const *state_b,
                                              nk_l2_u8x32_state_haswell_t const *state_c,
                                              nk_l2_u8x32_state_haswell_t const *state_d, nk_f32_t query_norm,
                                              nk_f32_t target_norm_a, nk_f32_t target_norm_b, nk_f32_t target_norm_c,
                                              nk_f32_t target_norm_d, nk_f32_t *results) {
    // Extract all 4 dot products with single ILP-optimized call (u32 output)
    nk_u32_t dots_u32[4];
    nk_dot_u8x32_finalize_haswell(state_a, state_b, state_c, state_d, dots_u32);

    // Build 128-bit F32 vectors for parallel L2 distance: sqrt(q² + t² - 2*dot)
    __m128i dots_i32x4 = _mm_loadu_si128((__m128i const *)dots_u32);
    __m128 dots_f32x4 = _mm_cvtepi32_ps(dots_i32x4);
    __m128 query_norm_f32x4 = _mm_set1_ps(query_norm);
    __m128 target_norms_f32x4 = _mm_set_ps(target_norm_d, target_norm_c, target_norm_b, target_norm_a);

    // Compute squared norms in parallel
    __m128 query_sq_f32x4 = _mm_mul_ps(query_norm_f32x4, query_norm_f32x4);
    __m128 target_sq_f32x4 = _mm_mul_ps(target_norms_f32x4, target_norms_f32x4);

    // Compute distance squared: q² + t² - 2*dot using FMA
    __m128 two_f32x4 = _mm_set1_ps(2.0f);
    __m128 sum_sq_f32x4 = _mm_add_ps(query_sq_f32x4, target_sq_f32x4);
    __m128 dist_sq_f32x4 = _mm_fnmadd_ps(two_f32x4, dots_f32x4, sum_sq_f32x4);

    // Clamp negative to zero, then sqrt
    __m128 zeros_f32x4 = _mm_setzero_ps();
    __m128 clamped_f32x4 = _mm_max_ps(dist_sq_f32x4, zeros_f32x4);
    __m128 dist_f32x4 = _mm_sqrt_ps(clamped_f32x4);

    // Store results
    _mm_storeu_ps(results, dist_f32x4);
}

#if defined(__cplusplus)
} // extern "C"
#endif

#pragma clang attribute pop
#pragma GCC pop_options
#endif // NK_TARGET_HASWELL
#endif // NK_TARGET_X86_

#endif // NK_SPATIAL_HASWELL_H
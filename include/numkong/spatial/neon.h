/**
 *  @brief SIMD-accelerated Spatial Similarity Measures for NEON.
 *  @file include/numkong/spatial/neon.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 *
 *  @sa include/numkong/spatial.h
 *
 *  @section neon_spatial_instructions Key NEON Spatial Instructions
 *
 *  ARM NEON instructions for distance computations:
 *
 *      Intrinsic         Instruction                   Latency     Throughput
 *                                                                  A76     M4+/V1+/Oryon
 *      vfmaq_f32         FMLA (V.4S, V.4S, V.4S)       4cy         2/cy    4/cy
 *      vmulq_f32         FMUL (V.4S, V.4S, V.4S)       3cy         2/cy    4/cy
 *      vaddq_f32         FADD (V.4S, V.4S, V.4S)       2cy         2/cy    4/cy
 *      vsubq_f32         FSUB (V.4S, V.4S, V.4S)       2cy         2/cy    4/cy
 *      vrsqrteq_f32      FRSQRTE (V.4S, V.4S)          2cy         2/cy    2/cy
 *      vsqrtq_f32        FSQRT (V.4S, V.4S)            9-12cy      0.25/cy 0.25/cy
 *      vrecpeq_f32       FRECPE (V.4S, V.4S)           2cy         2/cy    2/cy
 *
 *  FRSQRTE provides ~8-bit precision; two Newton-Raphson iterations via vrsqrtsq_f32 achieve
 *  ~23-bit precision, sufficient for f32. This is much faster than FSQRT (0.25/cy).
 *
 *  Distance computations (L2, angular) benefit from 2x throughput on 4-pipe cores (Apple M4+,
 *  Graviton3+, Oryon), but FSQRT remains slow on all cores. Use rsqrt+NR when precision allows.
 */
#ifndef NK_SPATIAL_NEON_H
#define NK_SPATIAL_NEON_H

#if NK_TARGET_ARM_
#if NK_TARGET_NEON
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8-a+simd"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8-a+simd")
#endif

#include "numkong/types.h"
#include "numkong/dot/neon.h" // For nk_dot_f32x2_state_neon_t

#if defined(__cplusplus)
extern "C" {
#endif

NK_INTERNAL nk_f32_t nk_f32_sqrt_neon(nk_f32_t x) { return vget_lane_f32(vsqrt_f32(vdup_n_f32(x)), 0); }
NK_INTERNAL nk_f64_t nk_f64_sqrt_neon(nk_f64_t x) { return vget_lane_f64(vsqrt_f64(vdup_n_f64(x)), 0); }

NK_INTERNAL nk_f32_t nk_angular_normalize_f32_neon_(nk_f32_t ab, nk_f32_t a2, nk_f32_t b2) {
    if (a2 == 0 && b2 == 0) return 0;
    if (ab == 0) return 1;
    nk_f32_t squares_arr[2] = {a2, b2};
    float32x2_t squares = vld1_f32(squares_arr);
    // Unlike x86, Arm NEON manuals don't explicitly mention the accuracy of their `rsqrt` approximation.
    // Third-party research suggests that it's less accurate than SSE instructions, having an error of 1.5×2⁻¹².
    // One or two rounds of Newton-Raphson refinement are recommended to improve the accuracy.
    // https://github.com/lighttransport/embree-aarch64/issues/24
    // https://github.com/lighttransport/embree-aarch64/blob/3f75f8cb4e553d13dced941b5fefd4c826835a6b/common/math/math.h#L137-L145
    float32x2_t rsqrts = vrsqrte_f32(squares);
    // Perform two rounds of Newton-Raphson refinement:
    // https://en.wikipedia.org/wiki/Newton%27s_method
    rsqrts = vmul_f32(rsqrts, vrsqrts_f32(vmul_f32(squares, rsqrts), rsqrts));
    rsqrts = vmul_f32(rsqrts, vrsqrts_f32(vmul_f32(squares, rsqrts), rsqrts));
    vst1_f32(squares_arr, rsqrts);
    nk_f32_t result = 1 - ab * squares_arr[0] * squares_arr[1];
    return result > 0 ? result : 0;
}

NK_INTERNAL nk_f64_t nk_angular_normalize_f64_neon_(nk_f64_t ab, nk_f64_t a2, nk_f64_t b2) {
    if (a2 == 0 && b2 == 0) return 0;
    if (ab == 0) return 1;
    nk_f64_t squares_arr[2] = {a2, b2};
    float64x2_t squares = vld1q_f64(squares_arr);

    // Unlike x86, Arm NEON manuals don't explicitly mention the accuracy of their `rsqrt` approximation.
    // Third-party research suggests that it's less accurate than SSE instructions, having an error of 1.5×2⁻¹².
    // One or two rounds of Newton-Raphson refinement are recommended to improve the accuracy.
    // https://github.com/lighttransport/embree-aarch64/issues/24
    // https://github.com/lighttransport/embree-aarch64/blob/3f75f8cb4e553d13dced941b5fefd4c826835a6b/common/math/math.h#L137-L145
    float64x2_t rsqrts_f64x2 = vrsqrteq_f64(squares);
    // Perform three rounds of Newton-Raphson refinement for f64 precision (~48 bits):
    // https://en.wikipedia.org/wiki/Newton%27s_method
    rsqrts_f64x2 = vmulq_f64(rsqrts_f64x2, vrsqrtsq_f64(vmulq_f64(squares, rsqrts_f64x2), rsqrts_f64x2));
    rsqrts_f64x2 = vmulq_f64(rsqrts_f64x2, vrsqrtsq_f64(vmulq_f64(squares, rsqrts_f64x2), rsqrts_f64x2));
    rsqrts_f64x2 = vmulq_f64(rsqrts_f64x2, vrsqrtsq_f64(vmulq_f64(squares, rsqrts_f64x2), rsqrts_f64x2));
    vst1q_f64(squares_arr, rsqrts_f64x2);
    nk_f64_t result = 1 - ab * squares_arr[0] * squares_arr[1];
    return result > 0 ? result : 0;
}

NK_PUBLIC void nk_sqeuclidean_f32_neon(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *result) {
    // Accumulate in f64 for numerical stability (2 f32s per iteration, avoids slow vget_low/high)
    float64x2_t sum_f64x2 = vdupq_n_f64(0);
    nk_size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        float32x2_t a_f32x2 = vld1_f32(a + i);
        float32x2_t b_f32x2 = vld1_f32(b + i);
        float32x2_t diff_f32x2 = vsub_f32(a_f32x2, b_f32x2);
        float64x2_t diff_f64x2 = vcvt_f64_f32(diff_f32x2);
        sum_f64x2 = vfmaq_f64(sum_f64x2, diff_f64x2, diff_f64x2);
    }
    nk_f64_t sum_f64 = vaddvq_f64(sum_f64x2);
    for (; i < n; ++i) {
        nk_f64_t diff_f64 = (nk_f64_t)a[i] - (nk_f64_t)b[i];
        sum_f64 += diff_f64 * diff_f64;
    }
    *result = (nk_f32_t)sum_f64;
}

NK_PUBLIC void nk_euclidean_f32_neon(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_sqeuclidean_f32_neon(a, b, n, result);
    *result = nk_f32_sqrt_neon(*result);
}

NK_PUBLIC void nk_angular_f32_neon(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *result) {
    // Accumulate in f64 for numerical stability (2 f32s per iteration, avoids slow vget_low/high)
    float64x2_t ab_f64x2 = vdupq_n_f64(0);
    float64x2_t a2_f64x2 = vdupq_n_f64(0);
    float64x2_t b2_f64x2 = vdupq_n_f64(0);
    nk_size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        float32x2_t a_f32x2 = vld1_f32(a + i);
        float32x2_t b_f32x2 = vld1_f32(b + i);
        float64x2_t a_f64x2 = vcvt_f64_f32(a_f32x2);
        float64x2_t b_f64x2 = vcvt_f64_f32(b_f32x2);
        ab_f64x2 = vfmaq_f64(ab_f64x2, a_f64x2, b_f64x2);
        a2_f64x2 = vfmaq_f64(a2_f64x2, a_f64x2, a_f64x2);
        b2_f64x2 = vfmaq_f64(b2_f64x2, b_f64x2, b_f64x2);
    }
    nk_f64_t ab_f64 = vaddvq_f64(ab_f64x2);
    nk_f64_t a2_f64 = vaddvq_f64(a2_f64x2);
    nk_f64_t b2_f64 = vaddvq_f64(b2_f64x2);
    for (; i < n; ++i) {
        nk_f64_t ai = (nk_f64_t)a[i], bi = (nk_f64_t)b[i];
        ab_f64 += ai * bi, a2_f64 += ai * ai, b2_f64 += bi * bi;
    }
    *result = (nk_f32_t)nk_angular_normalize_f64_neon_(ab_f64, a2_f64, b2_f64);
}

NK_PUBLIC void nk_sqeuclidean_f64_neon(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result) {
    float64x2_t sum_f64x2 = vdupq_n_f64(0);
    nk_size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        float64x2_t a_f64x2 = vld1q_f64(a + i);
        float64x2_t b_f64x2 = vld1q_f64(b + i);
        float64x2_t diff_f64x2 = vsubq_f64(a_f64x2, b_f64x2);
        sum_f64x2 = vfmaq_f64(sum_f64x2, diff_f64x2, diff_f64x2);
    }
    nk_f64_t sum_f64 = vaddvq_f64(sum_f64x2);
    for (; i < n; ++i) {
        nk_f64_t diff_f64 = a[i] - b[i];
        sum_f64 += diff_f64 * diff_f64;
    }
    *result = sum_f64;
}

NK_PUBLIC void nk_euclidean_f64_neon(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result) {
    nk_sqeuclidean_f64_neon(a, b, n, result);
    *result = nk_f64_sqrt_neon(*result);
}

NK_PUBLIC void nk_angular_f64_neon(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result) {
    // Dot2 (Ogita-Rump-Oishi) for cross-product ab (may have cancellation),
    // simple FMA for self-products a2/b2 (all positive, no cancellation)
    float64x2_t ab_sum_f64x2 = vdupq_n_f64(0);
    float64x2_t ab_compensation_f64x2 = vdupq_n_f64(0);
    float64x2_t a2_f64x2 = vdupq_n_f64(0);
    float64x2_t b2_f64x2 = vdupq_n_f64(0);
    nk_size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        float64x2_t a_f64x2 = vld1q_f64(a + i);
        float64x2_t b_f64x2 = vld1q_f64(b + i);
        // TwoProd for ab: product = a*b, error = fma(a,b,-product)
        float64x2_t product_f64x2 = vmulq_f64(a_f64x2, b_f64x2);
        float64x2_t product_error_f64x2 = vnegq_f64(vfmsq_f64(product_f64x2, a_f64x2, b_f64x2));
        // TwoSum: (t, q) = TwoSum(sum, product)
        float64x2_t t_f64x2 = vaddq_f64(ab_sum_f64x2, product_f64x2);
        float64x2_t z_f64x2 = vsubq_f64(t_f64x2, ab_sum_f64x2);
        float64x2_t sum_error_f64x2 = vaddq_f64(vsubq_f64(ab_sum_f64x2, vsubq_f64(t_f64x2, z_f64x2)),
                                                vsubq_f64(product_f64x2, z_f64x2));
        ab_sum_f64x2 = t_f64x2;
        ab_compensation_f64x2 = vaddq_f64(ab_compensation_f64x2, vaddq_f64(sum_error_f64x2, product_error_f64x2));
        // Simple FMA for self-products (no cancellation)
        a2_f64x2 = vfmaq_f64(a2_f64x2, a_f64x2, a_f64x2);
        b2_f64x2 = vfmaq_f64(b2_f64x2, b_f64x2, b_f64x2);
    }
    nk_f64_t ab_f64 = vaddvq_f64(vaddq_f64(ab_sum_f64x2, ab_compensation_f64x2));
    nk_f64_t a2_f64 = vaddvq_f64(a2_f64x2);
    nk_f64_t b2_f64 = vaddvq_f64(b2_f64x2);
    for (; i < n; ++i) {
        nk_f64_t ai = a[i], bi = b[i];
        ab_f64 += ai * bi, a2_f64 += ai * ai, b2_f64 += bi * bi;
    }
    *result = nk_angular_normalize_f64_neon_(ab_f64, a2_f64, b2_f64);
}

NK_PUBLIC void nk_sqeuclidean_e2m3_neon(nk_e2m3_t const *a, nk_e2m3_t const *b, nk_size_t n, nk_f32_t *result) {
    float32x4_t sum_f32x4 = vdupq_n_f32(0);
    nk_b32_vec_t a_vec, b_vec;

nk_sqeuclidean_e2m3_neon_cycle:
    if (n < 4) {
        a_vec = nk_partial_load_b8x4_serial_(a, n);
        b_vec = nk_partial_load_b8x4_serial_(b, n);
        n = 0;
    }
    else {
        nk_load_b32_serial_(a, &a_vec);
        nk_load_b32_serial_(b, &b_vec);
        a += 4, b += 4, n -= 4;
    }

    float32x4_t a_f32x4 = nk_e2m3x4_to_f32x4_neon_(a_vec);
    float32x4_t b_f32x4 = nk_e2m3x4_to_f32x4_neon_(b_vec);
    float32x4_t diff_f32x4 = vsubq_f32(a_f32x4, b_f32x4);
    sum_f32x4 = vfmaq_f32(sum_f32x4, diff_f32x4, diff_f32x4);
    if (n) goto nk_sqeuclidean_e2m3_neon_cycle;

    *result = vaddvq_f32(sum_f32x4);
}

NK_PUBLIC void nk_euclidean_e2m3_neon(nk_e2m3_t const *a, nk_e2m3_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_sqeuclidean_e2m3_neon(a, b, n, result);
    *result = nk_f32_sqrt_neon(*result);
}

NK_PUBLIC void nk_angular_e2m3_neon(nk_e2m3_t const *a, nk_e2m3_t const *b, nk_size_t n, nk_f32_t *result) {
    float32x4_t ab_f32x4 = vdupq_n_f32(0);
    float32x4_t a2_f32x4 = vdupq_n_f32(0);
    float32x4_t b2_f32x4 = vdupq_n_f32(0);
    nk_b32_vec_t a_vec, b_vec;

nk_angular_e2m3_neon_cycle:
    if (n < 4) {
        a_vec = nk_partial_load_b8x4_serial_(a, n);
        b_vec = nk_partial_load_b8x4_serial_(b, n);
        n = 0;
    }
    else {
        nk_load_b32_serial_(a, &a_vec);
        nk_load_b32_serial_(b, &b_vec);
        a += 4, b += 4, n -= 4;
    }

    float32x4_t a_f32x4 = nk_e2m3x4_to_f32x4_neon_(a_vec);
    float32x4_t b_f32x4 = nk_e2m3x4_to_f32x4_neon_(b_vec);
    ab_f32x4 = vfmaq_f32(ab_f32x4, a_f32x4, b_f32x4);
    a2_f32x4 = vfmaq_f32(a2_f32x4, a_f32x4, a_f32x4);
    b2_f32x4 = vfmaq_f32(b2_f32x4, b_f32x4, b_f32x4);
    if (n) goto nk_angular_e2m3_neon_cycle;

    nk_f32_t ab = vaddvq_f32(ab_f32x4);
    nk_f32_t a2 = vaddvq_f32(a2_f32x4);
    nk_f32_t b2 = vaddvq_f32(b2_f32x4);
    *result = nk_angular_normalize_f32_neon_(ab, a2, b2);
}

NK_PUBLIC void nk_sqeuclidean_e3m2_neon(nk_e3m2_t const *a, nk_e3m2_t const *b, nk_size_t n, nk_f32_t *result) {
    float32x4_t sum_f32x4 = vdupq_n_f32(0);
    nk_b32_vec_t a_vec, b_vec;

nk_sqeuclidean_e3m2_neon_cycle:
    if (n < 4) {
        a_vec = nk_partial_load_b8x4_serial_(a, n);
        b_vec = nk_partial_load_b8x4_serial_(b, n);
        n = 0;
    }
    else {
        nk_load_b32_serial_(a, &a_vec);
        nk_load_b32_serial_(b, &b_vec);
        a += 4, b += 4, n -= 4;
    }

    float32x4_t a_f32x4 = nk_e3m2x4_to_f32x4_neon_(a_vec);
    float32x4_t b_f32x4 = nk_e3m2x4_to_f32x4_neon_(b_vec);
    float32x4_t diff_f32x4 = vsubq_f32(a_f32x4, b_f32x4);
    sum_f32x4 = vfmaq_f32(sum_f32x4, diff_f32x4, diff_f32x4);
    if (n) goto nk_sqeuclidean_e3m2_neon_cycle;

    *result = vaddvq_f32(sum_f32x4);
}

NK_PUBLIC void nk_euclidean_e3m2_neon(nk_e3m2_t const *a, nk_e3m2_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_sqeuclidean_e3m2_neon(a, b, n, result);
    *result = nk_f32_sqrt_neon(*result);
}

NK_PUBLIC void nk_angular_e3m2_neon(nk_e3m2_t const *a, nk_e3m2_t const *b, nk_size_t n, nk_f32_t *result) {
    float32x4_t ab_f32x4 = vdupq_n_f32(0);
    float32x4_t a2_f32x4 = vdupq_n_f32(0);
    float32x4_t b2_f32x4 = vdupq_n_f32(0);
    nk_b32_vec_t a_vec, b_vec;

nk_angular_e3m2_neon_cycle:
    if (n < 4) {
        a_vec = nk_partial_load_b8x4_serial_(a, n);
        b_vec = nk_partial_load_b8x4_serial_(b, n);
        n = 0;
    }
    else {
        nk_load_b32_serial_(a, &a_vec);
        nk_load_b32_serial_(b, &b_vec);
        a += 4, b += 4, n -= 4;
    }

    float32x4_t a_f32x4 = nk_e3m2x4_to_f32x4_neon_(a_vec);
    float32x4_t b_f32x4 = nk_e3m2x4_to_f32x4_neon_(b_vec);
    ab_f32x4 = vfmaq_f32(ab_f32x4, a_f32x4, b_f32x4);
    a2_f32x4 = vfmaq_f32(a2_f32x4, a_f32x4, a_f32x4);
    b2_f32x4 = vfmaq_f32(b2_f32x4, b_f32x4, b_f32x4);
    if (n) goto nk_angular_e3m2_neon_cycle;

    nk_f32_t ab = vaddvq_f32(ab_f32x4);
    nk_f32_t a2 = vaddvq_f32(a2_f32x4);
    nk_f32_t b2 = vaddvq_f32(b2_f32x4);
    *result = nk_angular_normalize_f32_neon_(ab, a2, b2);
}

NK_PUBLIC void nk_sqeuclidean_e4m3_neon(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n, nk_f32_t *result) {
    float32x4_t sum_f32x4 = vdupq_n_f32(0);
    nk_b32_vec_t a_vec, b_vec;

nk_sqeuclidean_e4m3_neon_cycle:
    if (n < 4) {
        a_vec = nk_partial_load_b8x4_serial_(a, n);
        b_vec = nk_partial_load_b8x4_serial_(b, n);
        n = 0;
    }
    else {
        nk_load_b32_serial_(a, &a_vec);
        nk_load_b32_serial_(b, &b_vec);
        a += 4, b += 4, n -= 4;
    }

    float32x4_t a_f32x4 = nk_e4m3x4_to_f32x4_neon_(a_vec);
    float32x4_t b_f32x4 = nk_e4m3x4_to_f32x4_neon_(b_vec);

    float32x4_t diff_f32x4 = vsubq_f32(a_f32x4, b_f32x4);
    sum_f32x4 = vfmaq_f32(sum_f32x4, diff_f32x4, diff_f32x4);

    if (n) goto nk_sqeuclidean_e4m3_neon_cycle;

    *result = vaddvq_f32(sum_f32x4);
}

NK_PUBLIC void nk_euclidean_e4m3_neon(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_sqeuclidean_e4m3_neon(a, b, n, result);
    *result = nk_f32_sqrt_neon(*result);
}

NK_PUBLIC void nk_angular_e4m3_neon(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n, nk_f32_t *result) {
    float32x4_t ab_f32x4 = vdupq_n_f32(0);
    float32x4_t a2_f32x4 = vdupq_n_f32(0);
    float32x4_t b2_f32x4 = vdupq_n_f32(0);
    nk_b32_vec_t a_vec, b_vec;

nk_angular_e4m3_neon_cycle:
    if (n < 4) {
        a_vec = nk_partial_load_b8x4_serial_(a, n);
        b_vec = nk_partial_load_b8x4_serial_(b, n);
        n = 0;
    }
    else {
        nk_load_b32_serial_(a, &a_vec);
        nk_load_b32_serial_(b, &b_vec);
        a += 4, b += 4, n -= 4;
    }

    float32x4_t a_f32x4 = nk_e4m3x4_to_f32x4_neon_(a_vec);
    float32x4_t b_f32x4 = nk_e4m3x4_to_f32x4_neon_(b_vec);
    ab_f32x4 = vfmaq_f32(ab_f32x4, a_f32x4, b_f32x4);
    a2_f32x4 = vfmaq_f32(a2_f32x4, a_f32x4, a_f32x4);
    b2_f32x4 = vfmaq_f32(b2_f32x4, b_f32x4, b_f32x4);
    if (n) goto nk_angular_e4m3_neon_cycle;

    nk_f32_t ab = vaddvq_f32(ab_f32x4);
    nk_f32_t a2 = vaddvq_f32(a2_f32x4);
    nk_f32_t b2 = vaddvq_f32(b2_f32x4);
    *result = nk_angular_normalize_f32_neon_(ab, a2, b2);
}

NK_PUBLIC void nk_sqeuclidean_e5m2_neon(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n, nk_f32_t *result) {
    float32x4_t sum_f32x4 = vdupq_n_f32(0);
    nk_b32_vec_t a_vec, b_vec;

nk_sqeuclidean_e5m2_neon_cycle:
    if (n < 4) {
        a_vec = nk_partial_load_b8x4_serial_(a, n);
        b_vec = nk_partial_load_b8x4_serial_(b, n);
        n = 0;
    }
    else {
        nk_load_b32_serial_(a, &a_vec);
        nk_load_b32_serial_(b, &b_vec);
        a += 4, b += 4, n -= 4;
    }

    float32x4_t a_f32x4 = nk_e5m2x4_to_f32x4_neon_(a_vec);
    float32x4_t b_f32x4 = nk_e5m2x4_to_f32x4_neon_(b_vec);
    float32x4_t diff_f32x4 = vsubq_f32(a_f32x4, b_f32x4);
    sum_f32x4 = vfmaq_f32(sum_f32x4, diff_f32x4, diff_f32x4);
    if (n) goto nk_sqeuclidean_e5m2_neon_cycle;

    *result = vaddvq_f32(sum_f32x4);
}

NK_PUBLIC void nk_euclidean_e5m2_neon(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_sqeuclidean_e5m2_neon(a, b, n, result);
    *result = nk_f32_sqrt_neon(*result);
}

NK_PUBLIC void nk_angular_e5m2_neon(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n, nk_f32_t *result) {
    float32x4_t ab_f32x4 = vdupq_n_f32(0);
    float32x4_t a2_f32x4 = vdupq_n_f32(0);
    float32x4_t b2_f32x4 = vdupq_n_f32(0);
    nk_b32_vec_t a_vec, b_vec;

nk_angular_e5m2_neon_cycle:
    if (n < 4) {
        a_vec = nk_partial_load_b8x4_serial_(a, n);
        b_vec = nk_partial_load_b8x4_serial_(b, n);
        n = 0;
    }
    else {
        nk_load_b32_serial_(a, &a_vec);
        nk_load_b32_serial_(b, &b_vec);
        a += 4, b += 4, n -= 4;
    }

    float32x4_t a_f32x4 = nk_e5m2x4_to_f32x4_neon_(a_vec);
    float32x4_t b_f32x4 = nk_e5m2x4_to_f32x4_neon_(b_vec);
    ab_f32x4 = vfmaq_f32(ab_f32x4, a_f32x4, b_f32x4);
    a2_f32x4 = vfmaq_f32(a2_f32x4, a_f32x4, a_f32x4);
    b2_f32x4 = vfmaq_f32(b2_f32x4, b_f32x4, b_f32x4);
    if (n) goto nk_angular_e5m2_neon_cycle;

    nk_f32_t ab = vaddvq_f32(ab_f32x4);
    nk_f32_t a2 = vaddvq_f32(a2_f32x4);
    nk_f32_t b2 = vaddvq_f32(b2_f32x4);
    *result = nk_angular_normalize_f32_neon_(ab, a2, b2);
}

/** @brief Angular distance finalize: computes 1 − dot/√(‖query‖ × ‖target‖) for 4 pairs in f64. */
NK_INTERNAL void nk_angular_through_f64_finalize_neon_(float32x4_t dots_f32x4, nk_f32_t query_norm,
                                                       nk_f32_t target_norm_a, nk_f32_t target_norm_b,
                                                       nk_f32_t target_norm_c, nk_f32_t target_norm_d,
                                                       nk_f32_t *results) {
    // Build F64 vectors for parallel processing (2x float64x2_t for precision)
    float64x2_t dots_ab_f64x2 = vcvt_f64_f32(vget_low_f32(dots_f32x4));
    float64x2_t dots_cd_f64x2 = vcvt_f64_f32(vget_high_f32(dots_f32x4));

    nk_f64_t query_norm_sq = (nk_f64_t)query_norm * (nk_f64_t)query_norm;
    float64x2_t query_sq_f64x2 = vdupq_n_f64(query_norm_sq);

    float64x2_t target_norms_ab_f64x2 = {(nk_f64_t)target_norm_a, (nk_f64_t)target_norm_b};
    float64x2_t target_norms_cd_f64x2 = {(nk_f64_t)target_norm_c, (nk_f64_t)target_norm_d};
    float64x2_t target_sq_ab_f64x2 = vmulq_f64(target_norms_ab_f64x2, target_norms_ab_f64x2);
    float64x2_t target_sq_cd_f64x2 = vmulq_f64(target_norms_cd_f64x2, target_norms_cd_f64x2);

    // products = query_sq * target_sq
    float64x2_t products_ab_f64x2 = vmulq_f64(query_sq_f64x2, target_sq_ab_f64x2);
    float64x2_t products_cd_f64x2 = vmulq_f64(query_sq_f64x2, target_sq_cd_f64x2);

    // rsqrt with Newton-Raphson (2 iterations for ~48-bit precision)
    float64x2_t rsqrt_ab_f64x2 = vrsqrteq_f64(products_ab_f64x2);
    float64x2_t rsqrt_cd_f64x2 = vrsqrteq_f64(products_cd_f64x2);
    rsqrt_ab_f64x2 = vmulq_f64(rsqrt_ab_f64x2,
                               vrsqrtsq_f64(vmulq_f64(products_ab_f64x2, rsqrt_ab_f64x2), rsqrt_ab_f64x2));
    rsqrt_cd_f64x2 = vmulq_f64(rsqrt_cd_f64x2,
                               vrsqrtsq_f64(vmulq_f64(products_cd_f64x2, rsqrt_cd_f64x2), rsqrt_cd_f64x2));
    rsqrt_ab_f64x2 = vmulq_f64(rsqrt_ab_f64x2,
                               vrsqrtsq_f64(vmulq_f64(products_ab_f64x2, rsqrt_ab_f64x2), rsqrt_ab_f64x2));
    rsqrt_cd_f64x2 = vmulq_f64(rsqrt_cd_f64x2,
                               vrsqrtsq_f64(vmulq_f64(products_cd_f64x2, rsqrt_cd_f64x2), rsqrt_cd_f64x2));

    // angular = 1 − dot × rsqrt(product)
    float64x2_t ones_f64x2 = vdupq_n_f64(1.0);
    float64x2_t zeros_f64x2 = vdupq_n_f64(0.0);
    float64x2_t result_ab_f64x2 = vsubq_f64(ones_f64x2, vmulq_f64(dots_ab_f64x2, rsqrt_ab_f64x2));
    float64x2_t result_cd_f64x2 = vsubq_f64(ones_f64x2, vmulq_f64(dots_cd_f64x2, rsqrt_cd_f64x2));

    // Clamp to [0, inf)
    result_ab_f64x2 = vmaxq_f64(result_ab_f64x2, zeros_f64x2);
    result_cd_f64x2 = vmaxq_f64(result_cd_f64x2, zeros_f64x2);

    // Handle edge cases with vectorized selects
    uint64x2_t products_zero_ab_u64x2 = vceqq_f64(products_ab_f64x2, zeros_f64x2);
    uint64x2_t products_zero_cd_u64x2 = vceqq_f64(products_cd_f64x2, zeros_f64x2);
    uint64x2_t dots_zero_ab_u64x2 = vceqq_f64(dots_ab_f64x2, zeros_f64x2);
    uint64x2_t dots_zero_cd_u64x2 = vceqq_f64(dots_cd_f64x2, zeros_f64x2);

    // Both zero → result = 0; products zero but dots nonzero → result = 1
    uint64x2_t both_zero_ab_u64x2 = vandq_u64(products_zero_ab_u64x2, dots_zero_ab_u64x2);
    uint64x2_t both_zero_cd_u64x2 = vandq_u64(products_zero_cd_u64x2, dots_zero_cd_u64x2);
    result_ab_f64x2 = vbslq_f64(both_zero_ab_u64x2, zeros_f64x2, result_ab_f64x2);
    result_cd_f64x2 = vbslq_f64(both_zero_cd_u64x2, zeros_f64x2, result_cd_f64x2);

    uint64x2_t prod_zero_dot_nonzero_ab_u64x2 = vandq_u64(
        products_zero_ab_u64x2, vreinterpretq_u64_u32(vmvnq_u32(vreinterpretq_u32_u64(dots_zero_ab_u64x2))));
    uint64x2_t prod_zero_dot_nonzero_cd_u64x2 = vandq_u64(
        products_zero_cd_u64x2, vreinterpretq_u64_u32(vmvnq_u32(vreinterpretq_u32_u64(dots_zero_cd_u64x2))));
    result_ab_f64x2 = vbslq_f64(prod_zero_dot_nonzero_ab_u64x2, ones_f64x2, result_ab_f64x2);
    result_cd_f64x2 = vbslq_f64(prod_zero_dot_nonzero_cd_u64x2, ones_f64x2, result_cd_f64x2);

    // Convert to F32 and store
    float32x2_t result_ab_f32x2 = vcvt_f32_f64(result_ab_f64x2);
    float32x2_t result_cd_f32x2 = vcvt_f32_f64(result_cd_f64x2);
    vst1q_f32(results, vcombine_f32(result_ab_f32x2, result_cd_f32x2));
}

/** @brief L2 distance finalize: computes √(query²+target²−2 × dot) for 4 pairs in f64. */
NK_INTERNAL void nk_euclidean_through_f64_finalize_neon_(float32x4_t dots_f32x4, nk_f32_t query_norm,
                                                         nk_f32_t target_norm_a, nk_f32_t target_norm_b,
                                                         nk_f32_t target_norm_c, nk_f32_t target_norm_d,
                                                         nk_f32_t *results) {
    // Build F64 vectors
    float64x2_t dots_ab_f64x2 = vcvt_f64_f32(vget_low_f32(dots_f32x4));
    float64x2_t dots_cd_f64x2 = vcvt_f64_f32(vget_high_f32(dots_f32x4));

    nk_f64_t query_norm_sq = (nk_f64_t)query_norm * (nk_f64_t)query_norm;
    float64x2_t query_sq_f64x2 = vdupq_n_f64(query_norm_sq);

    float64x2_t target_norms_ab_f64x2 = {(nk_f64_t)target_norm_a, (nk_f64_t)target_norm_b};
    float64x2_t target_norms_cd_f64x2 = {(nk_f64_t)target_norm_c, (nk_f64_t)target_norm_d};
    float64x2_t target_sq_ab_f64x2 = vmulq_f64(target_norms_ab_f64x2, target_norms_ab_f64x2);
    float64x2_t target_sq_cd_f64x2 = vmulq_f64(target_norms_cd_f64x2, target_norms_cd_f64x2);

    // dist_sq = query_sq + target_sq − 2 × dot
    float64x2_t neg_two_f64x2 = vdupq_n_f64(-2.0);
    float64x2_t sum_sq_ab_f64x2 = vaddq_f64(query_sq_f64x2, target_sq_ab_f64x2);
    float64x2_t sum_sq_cd_f64x2 = vaddq_f64(query_sq_f64x2, target_sq_cd_f64x2);
    float64x2_t dist_sq_ab_f64x2 = vfmaq_f64(sum_sq_ab_f64x2, neg_two_f64x2, dots_ab_f64x2);
    float64x2_t dist_sq_cd_f64x2 = vfmaq_f64(sum_sq_cd_f64x2, neg_two_f64x2, dots_cd_f64x2);

    // Clamp and sqrt in f64
    float64x2_t zeros_f64x2 = vdupq_n_f64(0.0);
    dist_sq_ab_f64x2 = vmaxq_f64(dist_sq_ab_f64x2, zeros_f64x2);
    dist_sq_cd_f64x2 = vmaxq_f64(dist_sq_cd_f64x2, zeros_f64x2);
    float64x2_t dist_ab_f64x2 = vsqrtq_f64(dist_sq_ab_f64x2);
    float64x2_t dist_cd_f64x2 = vsqrtq_f64(dist_sq_cd_f64x2);

    // Convert to F32 and store
    float32x2_t dist_ab_f32x2 = vcvt_f32_f64(dist_ab_f64x2);
    float32x2_t dist_cd_f32x2 = vcvt_f32_f64(dist_cd_f64x2);
    vst1q_f32(results, vcombine_f32(dist_ab_f32x2, dist_cd_f32x2));
}

/** @brief Angular finalize with f32 numerics (for low-precision inputs f16/bf16/i8/u8). */
NK_INTERNAL void nk_angular_through_f32_finalize_neon_(float32x4_t dots_f32x4, nk_f32_t query_norm,
                                                       nk_f32_t target_norm_a, nk_f32_t target_norm_b,
                                                       nk_f32_t target_norm_c, nk_f32_t target_norm_d,
                                                       nk_f32_t *results) {
    float32x4_t query_norm_f32x4 = vdupq_n_f32(query_norm);
    float32x4_t target_norms_f32x4 = {target_norm_a, target_norm_b, target_norm_c, target_norm_d};
    float32x4_t products_f32x4 = vmulq_f32(query_norm_f32x4, target_norms_f32x4);

    // rsqrt with Newton-Raphson refinement (2 iterations)
    float32x4_t rsqrt_f32x4 = vrsqrteq_f32(products_f32x4);
    rsqrt_f32x4 = vmulq_f32(rsqrt_f32x4, vrsqrtsq_f32(vmulq_f32(products_f32x4, rsqrt_f32x4), rsqrt_f32x4));
    rsqrt_f32x4 = vmulq_f32(rsqrt_f32x4, vrsqrtsq_f32(vmulq_f32(products_f32x4, rsqrt_f32x4), rsqrt_f32x4));

    float32x4_t normalized_f32x4 = vmulq_f32(dots_f32x4, rsqrt_f32x4);
    float32x4_t result_f32x4 = vsubq_f32(vdupq_n_f32(1.0f), normalized_f32x4);

    // Clamp to [0, inf)
    result_f32x4 = vmaxq_f32(result_f32x4, vdupq_n_f32(0.0f));
    vst1q_f32(results, result_f32x4);
}

/** @brief L2 finalize with f32 numerics (for low-precision inputs f16/bf16/i8/u8). */
NK_INTERNAL void nk_euclidean_through_f32_finalize_neon_(float32x4_t dots_f32x4, nk_f32_t query_norm,
                                                         nk_f32_t target_norm_a, nk_f32_t target_norm_b,
                                                         nk_f32_t target_norm_c, nk_f32_t target_norm_d,
                                                         nk_f32_t *results) {
    float32x4_t query_norm_f32x4 = vdupq_n_f32(query_norm);
    float32x4_t target_norms_f32x4 = {target_norm_a, target_norm_b, target_norm_c, target_norm_d};
    float32x4_t query_sq_f32x4 = vmulq_f32(query_norm_f32x4, query_norm_f32x4);
    float32x4_t target_sq_f32x4 = vmulq_f32(target_norms_f32x4, target_norms_f32x4);
    float32x4_t sum_sq_f32x4 = vaddq_f32(query_sq_f32x4, target_sq_f32x4);
    // dist_sq = sum_sq − 2 × dot
    float32x4_t dist_sq_f32x4 = vfmsq_f32(sum_sq_f32x4, vdupq_n_f32(2.0f), dots_f32x4);
    // Clamp and sqrt
    dist_sq_f32x4 = vmaxq_f32(dist_sq_f32x4, vdupq_n_f32(0.0f));
    float32x4_t dist_f32x4 = vsqrtq_f32(dist_sq_f32x4);
    vst1q_f32(results, dist_f32x4);
}

typedef nk_dot_f32x2_state_neon_t nk_angular_f32x2_state_neon_t;
NK_INTERNAL void nk_angular_f32x2_init_neon(nk_angular_f32x2_state_neon_t *state) { nk_dot_f32x2_init_neon(state); }
NK_INTERNAL void nk_angular_f32x2_update_neon(nk_angular_f32x2_state_neon_t *state, nk_b64_vec_t a, nk_b64_vec_t b,
                                              nk_size_t depth_offset, nk_size_t active_dimensions) {
    nk_dot_f32x2_update_neon(state, a, b, depth_offset, active_dimensions);
}
NK_INTERNAL void nk_angular_f32x2_finalize_neon(nk_angular_f32x2_state_neon_t const *state_a,
                                                nk_angular_f32x2_state_neon_t const *state_b,
                                                nk_angular_f32x2_state_neon_t const *state_c,
                                                nk_angular_f32x2_state_neon_t const *state_d, nk_f32_t query_norm,
                                                nk_f32_t target_norm_a, nk_f32_t target_norm_b, nk_f32_t target_norm_c,
                                                nk_f32_t target_norm_d, nk_size_t total_dimensions, nk_f32_t *results) {
    nk_b128_vec_t dots_vec;
    nk_dot_f32x2_finalize_neon(state_a, state_b, state_c, state_d, total_dimensions, &dots_vec);
    nk_angular_through_f64_finalize_neon_(dots_vec.f32x4, query_norm, target_norm_a, target_norm_b, target_norm_c,
                                          target_norm_d, results);
}

typedef nk_dot_f32x2_state_neon_t nk_euclidean_f32x2_state_neon_t;
NK_INTERNAL void nk_euclidean_f32x2_init_neon(nk_euclidean_f32x2_state_neon_t *state) { nk_dot_f32x2_init_neon(state); }
NK_INTERNAL void nk_euclidean_f32x2_update_neon(nk_euclidean_f32x2_state_neon_t *state, nk_b64_vec_t a, nk_b64_vec_t b,
                                                nk_size_t depth_offset, nk_size_t active_dimensions) {
    nk_dot_f32x2_update_neon(state, a, b, depth_offset, active_dimensions);
}
NK_INTERNAL void nk_euclidean_f32x2_finalize_neon(nk_euclidean_f32x2_state_neon_t const *state_a,
                                                  nk_euclidean_f32x2_state_neon_t const *state_b,
                                                  nk_euclidean_f32x2_state_neon_t const *state_c,
                                                  nk_euclidean_f32x2_state_neon_t const *state_d, nk_f32_t query_norm,
                                                  nk_f32_t target_norm_a, nk_f32_t target_norm_b,
                                                  nk_f32_t target_norm_c, nk_f32_t target_norm_d,
                                                  nk_size_t total_dimensions, nk_f32_t *results) {
    nk_b128_vec_t dots_vec;
    nk_dot_f32x2_finalize_neon(state_a, state_b, state_c, state_d, total_dimensions, &dots_vec);
    nk_euclidean_through_f64_finalize_neon_(dots_vec.f32x4, query_norm, target_norm_a, target_norm_b, target_norm_c,
                                            target_norm_d, results);
}

#if defined(__cplusplus)
} // extern "C"
#endif

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // NK_TARGET_NEON
#endif // NK_TARGET_ARM_

#endif // NK_SPATIAL_NEON_H

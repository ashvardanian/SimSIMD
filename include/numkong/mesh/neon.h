/**
 *  @brief SIMD-accelerated mesh alignment functions optimized for Arm NEON-capable CPUs.
 *  @file include/numkong/mesh/neon.h
 *  @sa include/numkong/mesh.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 */
#ifndef NK_MESH_NEON_H
#define NK_MESH_NEON_H

#if NK_TARGET_ARM_
#if NK_TARGET_NEON
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8-a+simd"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8-a+simd")
#endif

#include "numkong/types.h"
#include "numkong/spatial/neon.h" // `nk_f32_sqrt_neon`

#if defined(__cplusplus)
extern "C" {
#endif

/*  Internal helper: Deinterleave 12 floats (4 xyz triplets) into separate x, y, z vectors.
 *  Uses NEON vld3q for efficient stride-3 deinterleaving.
 *
 *  Input: 12 contiguous floats [x0,y0,z0, x1,y1,z1, x2,y2,z2, x3,y3,z3]
 *  Output: x[4], y[4], z[4] vectors
 */
NK_INTERNAL void nk_deinterleave_f32x4_neon_(nk_f32_t const *ptr, float32x4_t *x_out, float32x4_t *y_out,
                                             float32x4_t *z_out) {
    float32x4x3_t xyz = vld3q_f32(ptr);
    *x_out = xyz.val[0];
    *y_out = xyz.val[1];
    *z_out = xyz.val[2];
}

/*  Internal helper: Deinterleave 6 f64 values (2 xyz triplets) into separate x, y, z vectors.
 *
 *  Input: 6 contiguous f64 [x0,y0,z0, x1,y1,z1]
 *  Output: x[2], y[2], z[2] vectors
 */
NK_INTERNAL void nk_deinterleave_f64x2_neon_(nk_f64_t const *ptr, float64x2_t *x_out, float64x2_t *y_out,
                                             float64x2_t *z_out) {
    // NEON doesn't have vld3q_f64, so we use vcombine to avoid stack round-trips
    // Load 2 xyz triplets: [x0,y0,z0, x1,y1,z1]
    *x_out = vcombine_f64(vld1_f64(&ptr[0]), vld1_f64(&ptr[3]));
    *y_out = vcombine_f64(vld1_f64(&ptr[1]), vld1_f64(&ptr[4]));
    *z_out = vcombine_f64(vld1_f64(&ptr[2]), vld1_f64(&ptr[5]));
}

/*  Internal helper: Compute sum of squared distances after applying rotation (and optional scale).
 *  Used by kabsch (scale=1.0) and umeyama (scale=computed_scale).
 *  Returns sum_squared, caller computes √(sum_squared / n).
 *
 *  Optimization: 2x loop unrolling with multiple accumulators hides FMA latency (3-7 cycles).
 */
NK_INTERNAL nk_f32_t nk_transformed_ssd_f32_neon_(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t const *r,
                                                  nk_f32_t scale, nk_f32_t centroid_a_x, nk_f32_t centroid_a_y,
                                                  nk_f32_t centroid_a_z, nk_f32_t centroid_b_x, nk_f32_t centroid_b_y,
                                                  nk_f32_t centroid_b_z) {
    // Broadcast scaled rotation matrix elements
    float32x4_t sr0_f32x4 = vdupq_n_f32(scale * r[0]), sr1_f32x4 = vdupq_n_f32(scale * r[1]),
                sr2_f32x4 = vdupq_n_f32(scale * r[2]);
    float32x4_t sr3_f32x4 = vdupq_n_f32(scale * r[3]), sr4_f32x4 = vdupq_n_f32(scale * r[4]),
                sr5_f32x4 = vdupq_n_f32(scale * r[5]);
    float32x4_t sr6_f32x4 = vdupq_n_f32(scale * r[6]), sr7_f32x4 = vdupq_n_f32(scale * r[7]),
                sr8_f32x4 = vdupq_n_f32(scale * r[8]);

    // Broadcast centroids
    float32x4_t centroid_a_x_f32x4 = vdupq_n_f32(centroid_a_x);
    float32x4_t centroid_a_y_f32x4 = vdupq_n_f32(centroid_a_y);
    float32x4_t centroid_a_z_f32x4 = vdupq_n_f32(centroid_a_z);
    float32x4_t centroid_b_x_f32x4 = vdupq_n_f32(centroid_b_x);
    float32x4_t centroid_b_y_f32x4 = vdupq_n_f32(centroid_b_y);
    float32x4_t centroid_b_z_f32x4 = vdupq_n_f32(centroid_b_z);

    // Two independent accumulators to hide FMA latency
    float32x4_t sum_squared_a_f32x4 = vdupq_n_f32(0);
    float32x4_t sum_squared_b_f32x4 = vdupq_n_f32(0);
    nk_size_t j = 0;

    // Main loop: process 8 points per iteration (2x unrolled)
    for (; j + 8 <= n; j += 8) {
        // First batch of 4 points
        float32x4_t a1_x, a1_y, a1_z, b1_x, b1_y, b1_z;
        nk_deinterleave_f32x4_neon_(a + j * 3, &a1_x, &a1_y, &a1_z);
        nk_deinterleave_f32x4_neon_(b + j * 3, &b1_x, &b1_y, &b1_z);

        // Second batch of 4 points
        float32x4_t a2_x, a2_y, a2_z, b2_x, b2_y, b2_z;
        nk_deinterleave_f32x4_neon_(a + (j + 4) * 3, &a2_x, &a2_y, &a2_z);
        nk_deinterleave_f32x4_neon_(b + (j + 4) * 3, &b2_x, &b2_y, &b2_z);

        // Center first batch
        float32x4_t pa1_x = vsubq_f32(a1_x, centroid_a_x_f32x4);
        float32x4_t pa1_y = vsubq_f32(a1_y, centroid_a_y_f32x4);
        float32x4_t pa1_z = vsubq_f32(a1_z, centroid_a_z_f32x4);
        float32x4_t pb1_x = vsubq_f32(b1_x, centroid_b_x_f32x4);
        float32x4_t pb1_y = vsubq_f32(b1_y, centroid_b_y_f32x4);
        float32x4_t pb1_z = vsubq_f32(b1_z, centroid_b_z_f32x4);

        // Center second batch
        float32x4_t pa2_x = vsubq_f32(a2_x, centroid_a_x_f32x4);
        float32x4_t pa2_y = vsubq_f32(a2_y, centroid_a_y_f32x4);
        float32x4_t pa2_z = vsubq_f32(a2_z, centroid_a_z_f32x4);
        float32x4_t pb2_x = vsubq_f32(b2_x, centroid_b_x_f32x4);
        float32x4_t pb2_y = vsubq_f32(b2_y, centroid_b_y_f32x4);
        float32x4_t pb2_z = vsubq_f32(b2_z, centroid_b_z_f32x4);

        // Rotate and scale first batch: ra1 = scale * R * pa1
        float32x4_t ra1_x = vfmaq_f32(vfmaq_f32(vmulq_f32(sr0_f32x4, pa1_x), sr1_f32x4, pa1_y), sr2_f32x4, pa1_z);
        float32x4_t ra1_y = vfmaq_f32(vfmaq_f32(vmulq_f32(sr3_f32x4, pa1_x), sr4_f32x4, pa1_y), sr5_f32x4, pa1_z);
        float32x4_t ra1_z = vfmaq_f32(vfmaq_f32(vmulq_f32(sr6_f32x4, pa1_x), sr7_f32x4, pa1_y), sr8_f32x4, pa1_z);

        // Rotate and scale second batch: ra2 = scale * R * pa2
        float32x4_t ra2_x = vfmaq_f32(vfmaq_f32(vmulq_f32(sr0_f32x4, pa2_x), sr1_f32x4, pa2_y), sr2_f32x4, pa2_z);
        float32x4_t ra2_y = vfmaq_f32(vfmaq_f32(vmulq_f32(sr3_f32x4, pa2_x), sr4_f32x4, pa2_y), sr5_f32x4, pa2_z);
        float32x4_t ra2_z = vfmaq_f32(vfmaq_f32(vmulq_f32(sr6_f32x4, pa2_x), sr7_f32x4, pa2_y), sr8_f32x4, pa2_z);

        // Deltas
        float32x4_t delta1_x = vsubq_f32(ra1_x, pb1_x);
        float32x4_t delta1_y = vsubq_f32(ra1_y, pb1_y);
        float32x4_t delta1_z = vsubq_f32(ra1_z, pb1_z);
        float32x4_t delta2_x = vsubq_f32(ra2_x, pb2_x);
        float32x4_t delta2_y = vsubq_f32(ra2_y, pb2_y);
        float32x4_t delta2_z = vsubq_f32(ra2_z, pb2_z);

        // Accumulate to independent accumulators
        sum_squared_a_f32x4 = vfmaq_f32(sum_squared_a_f32x4, delta1_x, delta1_x);
        sum_squared_b_f32x4 = vfmaq_f32(sum_squared_b_f32x4, delta2_x, delta2_x);
        sum_squared_a_f32x4 = vfmaq_f32(sum_squared_a_f32x4, delta1_y, delta1_y);
        sum_squared_b_f32x4 = vfmaq_f32(sum_squared_b_f32x4, delta2_y, delta2_y);
        sum_squared_a_f32x4 = vfmaq_f32(sum_squared_a_f32x4, delta1_z, delta1_z);
        sum_squared_b_f32x4 = vfmaq_f32(sum_squared_b_f32x4, delta2_z, delta2_z);
    }

    // Handle remaining 4 points
    if (j + 4 <= n) {
        float32x4_t a_x, a_y, a_z, b_x, b_y, b_z;
        nk_deinterleave_f32x4_neon_(a + j * 3, &a_x, &a_y, &a_z);
        nk_deinterleave_f32x4_neon_(b + j * 3, &b_x, &b_y, &b_z);

        float32x4_t pa_x = vsubq_f32(a_x, centroid_a_x_f32x4);
        float32x4_t pa_y = vsubq_f32(a_y, centroid_a_y_f32x4);
        float32x4_t pa_z = vsubq_f32(a_z, centroid_a_z_f32x4);
        float32x4_t pb_x = vsubq_f32(b_x, centroid_b_x_f32x4);
        float32x4_t pb_y = vsubq_f32(b_y, centroid_b_y_f32x4);
        float32x4_t pb_z = vsubq_f32(b_z, centroid_b_z_f32x4);

        float32x4_t ra_x = vfmaq_f32(vfmaq_f32(vmulq_f32(sr0_f32x4, pa_x), sr1_f32x4, pa_y), sr2_f32x4, pa_z);
        float32x4_t ra_y = vfmaq_f32(vfmaq_f32(vmulq_f32(sr3_f32x4, pa_x), sr4_f32x4, pa_y), sr5_f32x4, pa_z);
        float32x4_t ra_z = vfmaq_f32(vfmaq_f32(vmulq_f32(sr6_f32x4, pa_x), sr7_f32x4, pa_y), sr8_f32x4, pa_z);

        float32x4_t delta_x = vsubq_f32(ra_x, pb_x);
        float32x4_t delta_y = vsubq_f32(ra_y, pb_y);
        float32x4_t delta_z = vsubq_f32(ra_z, pb_z);

        sum_squared_a_f32x4 = vfmaq_f32(sum_squared_a_f32x4, delta_x, delta_x);
        sum_squared_a_f32x4 = vfmaq_f32(sum_squared_a_f32x4, delta_y, delta_y);
        sum_squared_a_f32x4 = vfmaq_f32(sum_squared_a_f32x4, delta_z, delta_z);
        j += 4;
    }

    // Combine accumulators and reduce
    float32x4_t sum_squared_f32x4 = vaddq_f32(sum_squared_a_f32x4, sum_squared_b_f32x4);
    nk_f32_t sum_squared = vaddvq_f32(sum_squared_f32x4);

    // Scalar tail
    for (; j < n; ++j) {
        nk_f32_t pa_x = a[j * 3 + 0] - centroid_a_x;
        nk_f32_t pa_y = a[j * 3 + 1] - centroid_a_y;
        nk_f32_t pa_z = a[j * 3 + 2] - centroid_a_z;
        nk_f32_t pb_x = b[j * 3 + 0] - centroid_b_x;
        nk_f32_t pb_y = b[j * 3 + 1] - centroid_b_y;
        nk_f32_t pb_z = b[j * 3 + 2] - centroid_b_z;

        nk_f32_t ra_x = scale * (r[0] * pa_x + r[1] * pa_y + r[2] * pa_z);
        nk_f32_t ra_y = scale * (r[3] * pa_x + r[4] * pa_y + r[5] * pa_z);
        nk_f32_t ra_z = scale * (r[6] * pa_x + r[7] * pa_y + r[8] * pa_z);

        nk_f32_t delta_x = ra_x - pb_x;
        nk_f32_t delta_y = ra_y - pb_y;
        nk_f32_t delta_z = ra_z - pb_z;
        sum_squared += delta_x * delta_x + delta_y * delta_y + delta_z * delta_z;
    }

    return sum_squared;
}

/*  Internal helper: Compute sum of squared distances for f64 after applying rotation (and optional scale).
 *  Note: rotation matrix r is f32 (from SVD), scale and data are f64.
 *
 *  Optimization: 2x loop unrolling with multiple accumulators hides FMA latency (3-7 cycles).
 */
NK_INTERNAL nk_f64_t nk_transformed_ssd_f64_neon_(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f32_t const *r,
                                                  nk_f64_t scale, nk_f64_t centroid_a_x, nk_f64_t centroid_a_y,
                                                  nk_f64_t centroid_a_z, nk_f64_t centroid_b_x, nk_f64_t centroid_b_y,
                                                  nk_f64_t centroid_b_z) {
    // Broadcast scaled rotation matrix elements (cast from f32)
    float64x2_t sr0_f64x2 = vdupq_n_f64(scale * r[0]), sr1_f64x2 = vdupq_n_f64(scale * r[1]),
                sr2_f64x2 = vdupq_n_f64(scale * r[2]);
    float64x2_t sr3_f64x2 = vdupq_n_f64(scale * r[3]), sr4_f64x2 = vdupq_n_f64(scale * r[4]),
                sr5_f64x2 = vdupq_n_f64(scale * r[5]);
    float64x2_t sr6_f64x2 = vdupq_n_f64(scale * r[6]), sr7_f64x2 = vdupq_n_f64(scale * r[7]),
                sr8_f64x2 = vdupq_n_f64(scale * r[8]);

    // Broadcast centroids
    float64x2_t centroid_a_x_f64x2 = vdupq_n_f64(centroid_a_x);
    float64x2_t centroid_a_y_f64x2 = vdupq_n_f64(centroid_a_y);
    float64x2_t centroid_a_z_f64x2 = vdupq_n_f64(centroid_a_z);
    float64x2_t centroid_b_x_f64x2 = vdupq_n_f64(centroid_b_x);
    float64x2_t centroid_b_y_f64x2 = vdupq_n_f64(centroid_b_y);
    float64x2_t centroid_b_z_f64x2 = vdupq_n_f64(centroid_b_z);

    // Two independent accumulators to hide FMA latency
    float64x2_t sum_squared_a_f64x2 = vdupq_n_f64(0);
    float64x2_t sum_squared_b_f64x2 = vdupq_n_f64(0);
    nk_size_t j = 0;

    // Main loop: process 4 points per iteration (2x unrolled, 2 points per batch)
    for (; j + 4 <= n; j += 4) {
        // First batch of 2 points
        float64x2_t a1_x, a1_y, a1_z, b1_x, b1_y, b1_z;
        nk_deinterleave_f64x2_neon_(a + j * 3, &a1_x, &a1_y, &a1_z);
        nk_deinterleave_f64x2_neon_(b + j * 3, &b1_x, &b1_y, &b1_z);

        // Second batch of 2 points
        float64x2_t a2_x, a2_y, a2_z, b2_x, b2_y, b2_z;
        nk_deinterleave_f64x2_neon_(a + (j + 2) * 3, &a2_x, &a2_y, &a2_z);
        nk_deinterleave_f64x2_neon_(b + (j + 2) * 3, &b2_x, &b2_y, &b2_z);

        // Center first batch
        float64x2_t pa1_x = vsubq_f64(a1_x, centroid_a_x_f64x2);
        float64x2_t pa1_y = vsubq_f64(a1_y, centroid_a_y_f64x2);
        float64x2_t pa1_z = vsubq_f64(a1_z, centroid_a_z_f64x2);
        float64x2_t pb1_x = vsubq_f64(b1_x, centroid_b_x_f64x2);
        float64x2_t pb1_y = vsubq_f64(b1_y, centroid_b_y_f64x2);
        float64x2_t pb1_z = vsubq_f64(b1_z, centroid_b_z_f64x2);

        // Center second batch
        float64x2_t pa2_x = vsubq_f64(a2_x, centroid_a_x_f64x2);
        float64x2_t pa2_y = vsubq_f64(a2_y, centroid_a_y_f64x2);
        float64x2_t pa2_z = vsubq_f64(a2_z, centroid_a_z_f64x2);
        float64x2_t pb2_x = vsubq_f64(b2_x, centroid_b_x_f64x2);
        float64x2_t pb2_y = vsubq_f64(b2_y, centroid_b_y_f64x2);
        float64x2_t pb2_z = vsubq_f64(b2_z, centroid_b_z_f64x2);

        // Rotate and scale first batch
        float64x2_t ra1_x = vfmaq_f64(vfmaq_f64(vmulq_f64(sr0_f64x2, pa1_x), sr1_f64x2, pa1_y), sr2_f64x2, pa1_z);
        float64x2_t ra1_y = vfmaq_f64(vfmaq_f64(vmulq_f64(sr3_f64x2, pa1_x), sr4_f64x2, pa1_y), sr5_f64x2, pa1_z);
        float64x2_t ra1_z = vfmaq_f64(vfmaq_f64(vmulq_f64(sr6_f64x2, pa1_x), sr7_f64x2, pa1_y), sr8_f64x2, pa1_z);

        // Rotate and scale second batch
        float64x2_t ra2_x = vfmaq_f64(vfmaq_f64(vmulq_f64(sr0_f64x2, pa2_x), sr1_f64x2, pa2_y), sr2_f64x2, pa2_z);
        float64x2_t ra2_y = vfmaq_f64(vfmaq_f64(vmulq_f64(sr3_f64x2, pa2_x), sr4_f64x2, pa2_y), sr5_f64x2, pa2_z);
        float64x2_t ra2_z = vfmaq_f64(vfmaq_f64(vmulq_f64(sr6_f64x2, pa2_x), sr7_f64x2, pa2_y), sr8_f64x2, pa2_z);

        // Deltas
        float64x2_t delta1_x = vsubq_f64(ra1_x, pb1_x);
        float64x2_t delta1_y = vsubq_f64(ra1_y, pb1_y);
        float64x2_t delta1_z = vsubq_f64(ra1_z, pb1_z);
        float64x2_t delta2_x = vsubq_f64(ra2_x, pb2_x);
        float64x2_t delta2_y = vsubq_f64(ra2_y, pb2_y);
        float64x2_t delta2_z = vsubq_f64(ra2_z, pb2_z);

        // Accumulate to independent accumulators (interleaved for latency hiding)
        sum_squared_a_f64x2 = vfmaq_f64(sum_squared_a_f64x2, delta1_x, delta1_x);
        sum_squared_b_f64x2 = vfmaq_f64(sum_squared_b_f64x2, delta2_x, delta2_x);
        sum_squared_a_f64x2 = vfmaq_f64(sum_squared_a_f64x2, delta1_y, delta1_y);
        sum_squared_b_f64x2 = vfmaq_f64(sum_squared_b_f64x2, delta2_y, delta2_y);
        sum_squared_a_f64x2 = vfmaq_f64(sum_squared_a_f64x2, delta1_z, delta1_z);
        sum_squared_b_f64x2 = vfmaq_f64(sum_squared_b_f64x2, delta2_z, delta2_z);
    }

    // Handle remaining 2 points
    if (j + 2 <= n) {
        float64x2_t a_x, a_y, a_z, b_x, b_y, b_z;
        nk_deinterleave_f64x2_neon_(a + j * 3, &a_x, &a_y, &a_z);
        nk_deinterleave_f64x2_neon_(b + j * 3, &b_x, &b_y, &b_z);

        float64x2_t pa_x = vsubq_f64(a_x, centroid_a_x_f64x2);
        float64x2_t pa_y = vsubq_f64(a_y, centroid_a_y_f64x2);
        float64x2_t pa_z = vsubq_f64(a_z, centroid_a_z_f64x2);
        float64x2_t pb_x = vsubq_f64(b_x, centroid_b_x_f64x2);
        float64x2_t pb_y = vsubq_f64(b_y, centroid_b_y_f64x2);
        float64x2_t pb_z = vsubq_f64(b_z, centroid_b_z_f64x2);

        float64x2_t ra_x = vfmaq_f64(vfmaq_f64(vmulq_f64(sr0_f64x2, pa_x), sr1_f64x2, pa_y), sr2_f64x2, pa_z);
        float64x2_t ra_y = vfmaq_f64(vfmaq_f64(vmulq_f64(sr3_f64x2, pa_x), sr4_f64x2, pa_y), sr5_f64x2, pa_z);
        float64x2_t ra_z = vfmaq_f64(vfmaq_f64(vmulq_f64(sr6_f64x2, pa_x), sr7_f64x2, pa_y), sr8_f64x2, pa_z);

        float64x2_t delta_x = vsubq_f64(ra_x, pb_x);
        float64x2_t delta_y = vsubq_f64(ra_y, pb_y);
        float64x2_t delta_z = vsubq_f64(ra_z, pb_z);

        sum_squared_a_f64x2 = vfmaq_f64(sum_squared_a_f64x2, delta_x, delta_x);
        sum_squared_a_f64x2 = vfmaq_f64(sum_squared_a_f64x2, delta_y, delta_y);
        sum_squared_a_f64x2 = vfmaq_f64(sum_squared_a_f64x2, delta_z, delta_z);
        j += 2;
    }

    // Combine accumulators and reduce
    float64x2_t sum_squared_f64x2 = vaddq_f64(sum_squared_a_f64x2, sum_squared_b_f64x2);
    nk_f64_t sum_squared = vaddvq_f64(sum_squared_f64x2);

    // Scalar tail
    for (; j < n; ++j) {
        nk_f64_t pa_x = a[j * 3 + 0] - centroid_a_x;
        nk_f64_t pa_y = a[j * 3 + 1] - centroid_a_y;
        nk_f64_t pa_z = a[j * 3 + 2] - centroid_a_z;
        nk_f64_t pb_x = b[j * 3 + 0] - centroid_b_x;
        nk_f64_t pb_y = b[j * 3 + 1] - centroid_b_y;
        nk_f64_t pb_z = b[j * 3 + 2] - centroid_b_z;

        nk_f64_t ra_x = scale * (r[0] * pa_x + r[1] * pa_y + r[2] * pa_z);
        nk_f64_t ra_y = scale * (r[3] * pa_x + r[4] * pa_y + r[5] * pa_z);
        nk_f64_t ra_z = scale * (r[6] * pa_x + r[7] * pa_y + r[8] * pa_z);

        nk_f64_t delta_x = ra_x - pb_x;
        nk_f64_t delta_y = ra_y - pb_y;
        nk_f64_t delta_z = ra_z - pb_z;
        sum_squared += delta_x * delta_x + delta_y * delta_y + delta_z * delta_z;
    }

    return sum_squared;
}

NK_PUBLIC void nk_rmsd_f32_neon(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result) {
    /* RMSD uses identity rotation and scale=1.0 */
    if (rotation) {
        rotation[0] = 1, rotation[1] = 0, rotation[2] = 0;
        rotation[3] = 0, rotation[4] = 1, rotation[5] = 0;
        rotation[6] = 0, rotation[7] = 0, rotation[8] = 1;
    }
    if (scale) *scale = 1.0f;

    float32x4_t const zeros_f32x4 = vdupq_n_f32(0);

    // Accumulators for centroids and squared differences
    float32x4_t sum_a_x_f32x4 = zeros_f32x4, sum_a_y_f32x4 = zeros_f32x4, sum_a_z_f32x4 = zeros_f32x4;
    float32x4_t sum_b_x_f32x4 = zeros_f32x4, sum_b_y_f32x4 = zeros_f32x4, sum_b_z_f32x4 = zeros_f32x4;
    float32x4_t sum_squared_x_f32x4 = zeros_f32x4, sum_squared_y_f32x4 = zeros_f32x4, sum_squared_z_f32x4 = zeros_f32x4;

    float32x4_t a_x_f32x4, a_y_f32x4, a_z_f32x4, b_x_f32x4, b_y_f32x4, b_z_f32x4;
    nk_size_t i = 0;

    // Main loop processing 4 points at a time
    for (; i + 4 <= n; i += 4) {
        nk_deinterleave_f32x4_neon_(a + i * 3, &a_x_f32x4, &a_y_f32x4, &a_z_f32x4);
        nk_deinterleave_f32x4_neon_(b + i * 3, &b_x_f32x4, &b_y_f32x4, &b_z_f32x4);

        sum_a_x_f32x4 = vaddq_f32(sum_a_x_f32x4, a_x_f32x4);
        sum_a_y_f32x4 = vaddq_f32(sum_a_y_f32x4, a_y_f32x4);
        sum_a_z_f32x4 = vaddq_f32(sum_a_z_f32x4, a_z_f32x4);
        sum_b_x_f32x4 = vaddq_f32(sum_b_x_f32x4, b_x_f32x4);
        sum_b_y_f32x4 = vaddq_f32(sum_b_y_f32x4, b_y_f32x4);
        sum_b_z_f32x4 = vaddq_f32(sum_b_z_f32x4, b_z_f32x4);

        float32x4_t delta_x_f32x4 = vsubq_f32(a_x_f32x4, b_x_f32x4);
        float32x4_t delta_y_f32x4 = vsubq_f32(a_y_f32x4, b_y_f32x4);
        float32x4_t delta_z_f32x4 = vsubq_f32(a_z_f32x4, b_z_f32x4);

        sum_squared_x_f32x4 = vfmaq_f32(sum_squared_x_f32x4, delta_x_f32x4, delta_x_f32x4);
        sum_squared_y_f32x4 = vfmaq_f32(sum_squared_y_f32x4, delta_y_f32x4, delta_y_f32x4);
        sum_squared_z_f32x4 = vfmaq_f32(sum_squared_z_f32x4, delta_z_f32x4, delta_z_f32x4);
    }

    // Reduce vectors to scalars
    nk_f32_t total_ax = vaddvq_f32(sum_a_x_f32x4);
    nk_f32_t total_ay = vaddvq_f32(sum_a_y_f32x4);
    nk_f32_t total_az = vaddvq_f32(sum_a_z_f32x4);
    nk_f32_t total_bx = vaddvq_f32(sum_b_x_f32x4);
    nk_f32_t total_by = vaddvq_f32(sum_b_y_f32x4);
    nk_f32_t total_bz = vaddvq_f32(sum_b_z_f32x4);
    nk_f32_t total_squared_x = vaddvq_f32(sum_squared_x_f32x4);
    nk_f32_t total_squared_y = vaddvq_f32(sum_squared_y_f32x4);
    nk_f32_t total_squared_z = vaddvq_f32(sum_squared_z_f32x4);

    // Scalar tail
    for (; i < n; ++i) {
        nk_f32_t ax = a[i * 3 + 0], ay = a[i * 3 + 1], az = a[i * 3 + 2];
        nk_f32_t bx = b[i * 3 + 0], by = b[i * 3 + 1], bz = b[i * 3 + 2];
        total_ax += ax;
        total_ay += ay;
        total_az += az;
        total_bx += bx;
        total_by += by;
        total_bz += bz;
        nk_f32_t delta_x = ax - bx, delta_y = ay - by, delta_z = az - bz;
        total_squared_x += delta_x * delta_x;
        total_squared_y += delta_y * delta_y;
        total_squared_z += delta_z * delta_z;
    }

    // Compute centroids
    nk_f32_t inv_n = 1.0f / (nk_f32_t)n;
    nk_f32_t centroid_a_x = total_ax * inv_n;
    nk_f32_t centroid_a_y = total_ay * inv_n;
    nk_f32_t centroid_a_z = total_az * inv_n;
    nk_f32_t centroid_b_x = total_bx * inv_n;
    nk_f32_t centroid_b_y = total_by * inv_n;
    nk_f32_t centroid_b_z = total_bz * inv_n;

    if (a_centroid) {
        a_centroid[0] = centroid_a_x;
        a_centroid[1] = centroid_a_y;
        a_centroid[2] = centroid_a_z;
    }
    if (b_centroid) {
        b_centroid[0] = centroid_b_x;
        b_centroid[1] = centroid_b_y;
        b_centroid[2] = centroid_b_z;
    }

    // Compute RMSD
    nk_f32_t mean_diff_x = centroid_a_x - centroid_b_x;
    nk_f32_t mean_diff_y = centroid_a_y - centroid_b_y;
    nk_f32_t mean_diff_z = centroid_a_z - centroid_b_z;
    nk_f32_t sum_squared = total_squared_x + total_squared_y + total_squared_z;
    nk_f32_t mean_diff_sq = mean_diff_x * mean_diff_x + mean_diff_y * mean_diff_y + mean_diff_z * mean_diff_z;

    *result = nk_f32_sqrt_neon(sum_squared * inv_n - mean_diff_sq);
}

NK_PUBLIC void nk_rmsd_f64_neon(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                                nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result) {
    /* RMSD uses identity rotation and scale=1.0 */
    if (rotation) {
        rotation[0] = 1, rotation[1] = 0, rotation[2] = 0;
        rotation[3] = 0, rotation[4] = 1, rotation[5] = 0;
        rotation[6] = 0, rotation[7] = 0, rotation[8] = 1;
    }
    if (scale) *scale = 1.0;

    float64x2_t const zeros_f64x2 = vdupq_n_f64(0);

    // Accumulators for centroids and squared differences
    float64x2_t sum_a_x_f64x2 = zeros_f64x2, sum_a_y_f64x2 = zeros_f64x2, sum_a_z_f64x2 = zeros_f64x2;
    float64x2_t sum_b_x_f64x2 = zeros_f64x2, sum_b_y_f64x2 = zeros_f64x2, sum_b_z_f64x2 = zeros_f64x2;
    float64x2_t sum_squared_x_f64x2 = zeros_f64x2, sum_squared_y_f64x2 = zeros_f64x2, sum_squared_z_f64x2 = zeros_f64x2;

    float64x2_t a_x_f64x2, a_y_f64x2, a_z_f64x2, b_x_f64x2, b_y_f64x2, b_z_f64x2;
    nk_size_t i = 0;

    // Main loop processing 2 points at a time
    for (; i + 2 <= n; i += 2) {
        nk_deinterleave_f64x2_neon_(a + i * 3, &a_x_f64x2, &a_y_f64x2, &a_z_f64x2);
        nk_deinterleave_f64x2_neon_(b + i * 3, &b_x_f64x2, &b_y_f64x2, &b_z_f64x2);

        sum_a_x_f64x2 = vaddq_f64(sum_a_x_f64x2, a_x_f64x2);
        sum_a_y_f64x2 = vaddq_f64(sum_a_y_f64x2, a_y_f64x2);
        sum_a_z_f64x2 = vaddq_f64(sum_a_z_f64x2, a_z_f64x2);
        sum_b_x_f64x2 = vaddq_f64(sum_b_x_f64x2, b_x_f64x2);
        sum_b_y_f64x2 = vaddq_f64(sum_b_y_f64x2, b_y_f64x2);
        sum_b_z_f64x2 = vaddq_f64(sum_b_z_f64x2, b_z_f64x2);

        float64x2_t delta_x_f64x2 = vsubq_f64(a_x_f64x2, b_x_f64x2);
        float64x2_t delta_y_f64x2 = vsubq_f64(a_y_f64x2, b_y_f64x2);
        float64x2_t delta_z_f64x2 = vsubq_f64(a_z_f64x2, b_z_f64x2);

        sum_squared_x_f64x2 = vfmaq_f64(sum_squared_x_f64x2, delta_x_f64x2, delta_x_f64x2);
        sum_squared_y_f64x2 = vfmaq_f64(sum_squared_y_f64x2, delta_y_f64x2, delta_y_f64x2);
        sum_squared_z_f64x2 = vfmaq_f64(sum_squared_z_f64x2, delta_z_f64x2, delta_z_f64x2);
    }

    // Reduce vectors to scalars
    nk_f64_t total_ax = vaddvq_f64(sum_a_x_f64x2);
    nk_f64_t total_ay = vaddvq_f64(sum_a_y_f64x2);
    nk_f64_t total_az = vaddvq_f64(sum_a_z_f64x2);
    nk_f64_t total_bx = vaddvq_f64(sum_b_x_f64x2);
    nk_f64_t total_by = vaddvq_f64(sum_b_y_f64x2);
    nk_f64_t total_bz = vaddvq_f64(sum_b_z_f64x2);
    nk_f64_t total_squared_x = vaddvq_f64(sum_squared_x_f64x2);
    nk_f64_t total_squared_y = vaddvq_f64(sum_squared_y_f64x2);
    nk_f64_t total_squared_z = vaddvq_f64(sum_squared_z_f64x2);

    // Scalar tail
    for (; i < n; ++i) {
        nk_f64_t ax = a[i * 3 + 0], ay = a[i * 3 + 1], az = a[i * 3 + 2];
        nk_f64_t bx = b[i * 3 + 0], by = b[i * 3 + 1], bz = b[i * 3 + 2];
        total_ax += ax;
        total_ay += ay;
        total_az += az;
        total_bx += bx;
        total_by += by;
        total_bz += bz;
        nk_f64_t delta_x = ax - bx, delta_y = ay - by, delta_z = az - bz;
        total_squared_x += delta_x * delta_x;
        total_squared_y += delta_y * delta_y;
        total_squared_z += delta_z * delta_z;
    }

    // Compute centroids
    nk_f64_t inv_n = 1.0 / (nk_f64_t)n;
    nk_f64_t centroid_a_x = total_ax * inv_n;
    nk_f64_t centroid_a_y = total_ay * inv_n;
    nk_f64_t centroid_a_z = total_az * inv_n;
    nk_f64_t centroid_b_x = total_bx * inv_n;
    nk_f64_t centroid_b_y = total_by * inv_n;
    nk_f64_t centroid_b_z = total_bz * inv_n;

    if (a_centroid) {
        a_centroid[0] = centroid_a_x;
        a_centroid[1] = centroid_a_y;
        a_centroid[2] = centroid_a_z;
    }
    if (b_centroid) {
        b_centroid[0] = centroid_b_x;
        b_centroid[1] = centroid_b_y;
        b_centroid[2] = centroid_b_z;
    }

    // Compute RMSD
    nk_f64_t mean_diff_x = centroid_a_x - centroid_b_x;
    nk_f64_t mean_diff_y = centroid_a_y - centroid_b_y;
    nk_f64_t mean_diff_z = centroid_a_z - centroid_b_z;
    nk_f64_t sum_squared = total_squared_x + total_squared_y + total_squared_z;
    nk_f64_t mean_diff_sq = mean_diff_x * mean_diff_x + mean_diff_y * mean_diff_y + mean_diff_z * mean_diff_z;

    *result = nk_f64_sqrt_neon(sum_squared * inv_n - mean_diff_sq);
}

NK_PUBLIC void nk_kabsch_f32_neon(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                  nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result) {
    float32x4_t const zeros_f32x4 = vdupq_n_f32(0);

    /*  2x unrolling with dual accumulators to hide FMA latency. */
    float32x4_t sum_a_x_a_f32x4 = zeros_f32x4, sum_a_y_a_f32x4 = zeros_f32x4, sum_a_z_a_f32x4 = zeros_f32x4;
    float32x4_t sum_b_x_a_f32x4 = zeros_f32x4, sum_b_y_a_f32x4 = zeros_f32x4, sum_b_z_a_f32x4 = zeros_f32x4;
    float32x4_t sum_a_x_b_f32x4 = zeros_f32x4, sum_a_y_b_f32x4 = zeros_f32x4, sum_a_z_b_f32x4 = zeros_f32x4;
    float32x4_t sum_b_x_b_f32x4 = zeros_f32x4, sum_b_y_b_f32x4 = zeros_f32x4, sum_b_z_b_f32x4 = zeros_f32x4;

    float32x4_t cov_xx_a_f32x4 = zeros_f32x4, cov_xy_a_f32x4 = zeros_f32x4, cov_xz_a_f32x4 = zeros_f32x4;
    float32x4_t cov_yx_a_f32x4 = zeros_f32x4, cov_yy_a_f32x4 = zeros_f32x4, cov_yz_a_f32x4 = zeros_f32x4;
    float32x4_t cov_zx_a_f32x4 = zeros_f32x4, cov_zy_a_f32x4 = zeros_f32x4, cov_zz_a_f32x4 = zeros_f32x4;
    float32x4_t cov_xx_b_f32x4 = zeros_f32x4, cov_xy_b_f32x4 = zeros_f32x4, cov_xz_b_f32x4 = zeros_f32x4;
    float32x4_t cov_yx_b_f32x4 = zeros_f32x4, cov_yy_b_f32x4 = zeros_f32x4, cov_yz_b_f32x4 = zeros_f32x4;
    float32x4_t cov_zx_b_f32x4 = zeros_f32x4, cov_zy_b_f32x4 = zeros_f32x4, cov_zz_b_f32x4 = zeros_f32x4;

    nk_size_t i = 0;
    float32x4_t a1_x_f32x4, a1_y_f32x4, a1_z_f32x4, b1_x_f32x4, b1_y_f32x4, b1_z_f32x4;
    float32x4_t a2_x_f32x4, a2_y_f32x4, a2_z_f32x4, b2_x_f32x4, b2_y_f32x4, b2_z_f32x4;

    // Main loop: 8 points per iteration (2x unrolled)
    for (; i + 8 <= n; i += 8) {
        nk_deinterleave_f32x4_neon_(a + i * 3, &a1_x_f32x4, &a1_y_f32x4, &a1_z_f32x4);
        nk_deinterleave_f32x4_neon_(b + i * 3, &b1_x_f32x4, &b1_y_f32x4, &b1_z_f32x4);
        nk_deinterleave_f32x4_neon_(a + (i + 4) * 3, &a2_x_f32x4, &a2_y_f32x4, &a2_z_f32x4);
        nk_deinterleave_f32x4_neon_(b + (i + 4) * 3, &b2_x_f32x4, &b2_y_f32x4, &b2_z_f32x4);

        // Interleaved accumulation to hide FMA latency
        sum_a_x_a_f32x4 = vaddq_f32(sum_a_x_a_f32x4, a1_x_f32x4);
        sum_a_x_b_f32x4 = vaddq_f32(sum_a_x_b_f32x4, a2_x_f32x4);
        sum_a_y_a_f32x4 = vaddq_f32(sum_a_y_a_f32x4, a1_y_f32x4);
        sum_a_y_b_f32x4 = vaddq_f32(sum_a_y_b_f32x4, a2_y_f32x4);
        sum_a_z_a_f32x4 = vaddq_f32(sum_a_z_a_f32x4, a1_z_f32x4);
        sum_a_z_b_f32x4 = vaddq_f32(sum_a_z_b_f32x4, a2_z_f32x4);
        sum_b_x_a_f32x4 = vaddq_f32(sum_b_x_a_f32x4, b1_x_f32x4);
        sum_b_x_b_f32x4 = vaddq_f32(sum_b_x_b_f32x4, b2_x_f32x4);
        sum_b_y_a_f32x4 = vaddq_f32(sum_b_y_a_f32x4, b1_y_f32x4);
        sum_b_y_b_f32x4 = vaddq_f32(sum_b_y_b_f32x4, b2_y_f32x4);
        sum_b_z_a_f32x4 = vaddq_f32(sum_b_z_a_f32x4, b1_z_f32x4);
        sum_b_z_b_f32x4 = vaddq_f32(sum_b_z_b_f32x4, b2_z_f32x4);

        cov_xx_a_f32x4 = vfmaq_f32(cov_xx_a_f32x4, a1_x_f32x4, b1_x_f32x4);
        cov_xx_b_f32x4 = vfmaq_f32(cov_xx_b_f32x4, a2_x_f32x4, b2_x_f32x4);
        cov_xy_a_f32x4 = vfmaq_f32(cov_xy_a_f32x4, a1_x_f32x4, b1_y_f32x4);
        cov_xy_b_f32x4 = vfmaq_f32(cov_xy_b_f32x4, a2_x_f32x4, b2_y_f32x4);
        cov_xz_a_f32x4 = vfmaq_f32(cov_xz_a_f32x4, a1_x_f32x4, b1_z_f32x4);
        cov_xz_b_f32x4 = vfmaq_f32(cov_xz_b_f32x4, a2_x_f32x4, b2_z_f32x4);
        cov_yx_a_f32x4 = vfmaq_f32(cov_yx_a_f32x4, a1_y_f32x4, b1_x_f32x4);
        cov_yx_b_f32x4 = vfmaq_f32(cov_yx_b_f32x4, a2_y_f32x4, b2_x_f32x4);
        cov_yy_a_f32x4 = vfmaq_f32(cov_yy_a_f32x4, a1_y_f32x4, b1_y_f32x4);
        cov_yy_b_f32x4 = vfmaq_f32(cov_yy_b_f32x4, a2_y_f32x4, b2_y_f32x4);
        cov_yz_a_f32x4 = vfmaq_f32(cov_yz_a_f32x4, a1_y_f32x4, b1_z_f32x4);
        cov_yz_b_f32x4 = vfmaq_f32(cov_yz_b_f32x4, a2_y_f32x4, b2_z_f32x4);
        cov_zx_a_f32x4 = vfmaq_f32(cov_zx_a_f32x4, a1_z_f32x4, b1_x_f32x4);
        cov_zx_b_f32x4 = vfmaq_f32(cov_zx_b_f32x4, a2_z_f32x4, b2_x_f32x4);
        cov_zy_a_f32x4 = vfmaq_f32(cov_zy_a_f32x4, a1_z_f32x4, b1_y_f32x4);
        cov_zy_b_f32x4 = vfmaq_f32(cov_zy_b_f32x4, a2_z_f32x4, b2_y_f32x4);
        cov_zz_a_f32x4 = vfmaq_f32(cov_zz_a_f32x4, a1_z_f32x4, b1_z_f32x4);
        cov_zz_b_f32x4 = vfmaq_f32(cov_zz_b_f32x4, a2_z_f32x4, b2_z_f32x4);
    }

    // 4-point tail
    for (; i + 4 <= n; i += 4) {
        nk_deinterleave_f32x4_neon_(a + i * 3, &a1_x_f32x4, &a1_y_f32x4, &a1_z_f32x4);
        nk_deinterleave_f32x4_neon_(b + i * 3, &b1_x_f32x4, &b1_y_f32x4, &b1_z_f32x4);
        sum_a_x_a_f32x4 = vaddq_f32(sum_a_x_a_f32x4, a1_x_f32x4);
        sum_a_y_a_f32x4 = vaddq_f32(sum_a_y_a_f32x4, a1_y_f32x4);
        sum_a_z_a_f32x4 = vaddq_f32(sum_a_z_a_f32x4, a1_z_f32x4);
        sum_b_x_a_f32x4 = vaddq_f32(sum_b_x_a_f32x4, b1_x_f32x4);
        sum_b_y_a_f32x4 = vaddq_f32(sum_b_y_a_f32x4, b1_y_f32x4);
        sum_b_z_a_f32x4 = vaddq_f32(sum_b_z_a_f32x4, b1_z_f32x4);
        cov_xx_a_f32x4 = vfmaq_f32(cov_xx_a_f32x4, a1_x_f32x4, b1_x_f32x4);
        cov_xy_a_f32x4 = vfmaq_f32(cov_xy_a_f32x4, a1_x_f32x4, b1_y_f32x4);
        cov_xz_a_f32x4 = vfmaq_f32(cov_xz_a_f32x4, a1_x_f32x4, b1_z_f32x4);
        cov_yx_a_f32x4 = vfmaq_f32(cov_yx_a_f32x4, a1_y_f32x4, b1_x_f32x4);
        cov_yy_a_f32x4 = vfmaq_f32(cov_yy_a_f32x4, a1_y_f32x4, b1_y_f32x4);
        cov_yz_a_f32x4 = vfmaq_f32(cov_yz_a_f32x4, a1_y_f32x4, b1_z_f32x4);
        cov_zx_a_f32x4 = vfmaq_f32(cov_zx_a_f32x4, a1_z_f32x4, b1_x_f32x4);
        cov_zy_a_f32x4 = vfmaq_f32(cov_zy_a_f32x4, a1_z_f32x4, b1_y_f32x4);
        cov_zz_a_f32x4 = vfmaq_f32(cov_zz_a_f32x4, a1_z_f32x4, b1_z_f32x4);
    }

    // Combine dual accumulators
    float32x4_t sum_a_x_f32x4 = vaddq_f32(sum_a_x_a_f32x4, sum_a_x_b_f32x4);
    float32x4_t sum_a_y_f32x4 = vaddq_f32(sum_a_y_a_f32x4, sum_a_y_b_f32x4);
    float32x4_t sum_a_z_f32x4 = vaddq_f32(sum_a_z_a_f32x4, sum_a_z_b_f32x4);
    float32x4_t sum_b_x_f32x4 = vaddq_f32(sum_b_x_a_f32x4, sum_b_x_b_f32x4);
    float32x4_t sum_b_y_f32x4 = vaddq_f32(sum_b_y_a_f32x4, sum_b_y_b_f32x4);
    float32x4_t sum_b_z_f32x4 = vaddq_f32(sum_b_z_a_f32x4, sum_b_z_b_f32x4);
    float32x4_t cov_xx_f32x4 = vaddq_f32(cov_xx_a_f32x4, cov_xx_b_f32x4);
    float32x4_t cov_xy_f32x4 = vaddq_f32(cov_xy_a_f32x4, cov_xy_b_f32x4);
    float32x4_t cov_xz_f32x4 = vaddq_f32(cov_xz_a_f32x4, cov_xz_b_f32x4);
    float32x4_t cov_yx_f32x4 = vaddq_f32(cov_yx_a_f32x4, cov_yx_b_f32x4);
    float32x4_t cov_yy_f32x4 = vaddq_f32(cov_yy_a_f32x4, cov_yy_b_f32x4);
    float32x4_t cov_yz_f32x4 = vaddq_f32(cov_yz_a_f32x4, cov_yz_b_f32x4);
    float32x4_t cov_zx_f32x4 = vaddq_f32(cov_zx_a_f32x4, cov_zx_b_f32x4);
    float32x4_t cov_zy_f32x4 = vaddq_f32(cov_zy_a_f32x4, cov_zy_b_f32x4);
    float32x4_t cov_zz_f32x4 = vaddq_f32(cov_zz_a_f32x4, cov_zz_b_f32x4);

    // Reduce vector accumulators
    nk_f32_t sum_a_x = vaddvq_f32(sum_a_x_f32x4);
    nk_f32_t sum_a_y = vaddvq_f32(sum_a_y_f32x4);
    nk_f32_t sum_a_z = vaddvq_f32(sum_a_z_f32x4);
    nk_f32_t sum_b_x = vaddvq_f32(sum_b_x_f32x4);
    nk_f32_t sum_b_y = vaddvq_f32(sum_b_y_f32x4);
    nk_f32_t sum_b_z = vaddvq_f32(sum_b_z_f32x4);

    nk_f32_t h00 = vaddvq_f32(cov_xx_f32x4);
    nk_f32_t h01 = vaddvq_f32(cov_xy_f32x4);
    nk_f32_t h02 = vaddvq_f32(cov_xz_f32x4);
    nk_f32_t h10 = vaddvq_f32(cov_yx_f32x4);
    nk_f32_t h11 = vaddvq_f32(cov_yy_f32x4);
    nk_f32_t h12 = vaddvq_f32(cov_yz_f32x4);
    nk_f32_t h20 = vaddvq_f32(cov_zx_f32x4);
    nk_f32_t h21 = vaddvq_f32(cov_zy_f32x4);
    nk_f32_t h22 = vaddvq_f32(cov_zz_f32x4);

    // Scalar tail
    for (; i < n; ++i) {
        nk_f32_t ax = a[i * 3 + 0], ay = a[i * 3 + 1], az = a[i * 3 + 2];
        nk_f32_t bx = b[i * 3 + 0], by = b[i * 3 + 1], bz = b[i * 3 + 2];
        sum_a_x += ax;
        sum_a_y += ay;
        sum_a_z += az;
        sum_b_x += bx;
        sum_b_y += by;
        sum_b_z += bz;
        h00 += ax * bx;
        h01 += ax * by;
        h02 += ax * bz;
        h10 += ay * bx;
        h11 += ay * by;
        h12 += ay * bz;
        h20 += az * bx;
        h21 += az * by;
        h22 += az * bz;
    }

    // Compute centroids
    nk_f32_t inv_n = 1.0f / (nk_f32_t)n;
    nk_f32_t centroid_a_x = sum_a_x * inv_n;
    nk_f32_t centroid_a_y = sum_a_y * inv_n;
    nk_f32_t centroid_a_z = sum_a_z * inv_n;
    nk_f32_t centroid_b_x = sum_b_x * inv_n;
    nk_f32_t centroid_b_y = sum_b_y * inv_n;
    nk_f32_t centroid_b_z = sum_b_z * inv_n;

    if (a_centroid) {
        a_centroid[0] = centroid_a_x;
        a_centroid[1] = centroid_a_y;
        a_centroid[2] = centroid_a_z;
    }
    if (b_centroid) {
        b_centroid[0] = centroid_b_x;
        b_centroid[1] = centroid_b_y;
        b_centroid[2] = centroid_b_z;
    }

    // Apply centering correction: H_centered = H - n * centroid_a * centroid_bᵀ
    h00 -= n * centroid_a_x * centroid_b_x;
    h01 -= n * centroid_a_x * centroid_b_y;
    h02 -= n * centroid_a_x * centroid_b_z;
    h10 -= n * centroid_a_y * centroid_b_x;
    h11 -= n * centroid_a_y * centroid_b_y;
    h12 -= n * centroid_a_y * centroid_b_z;
    h20 -= n * centroid_a_z * centroid_b_x;
    h21 -= n * centroid_a_z * centroid_b_y;
    h22 -= n * centroid_a_z * centroid_b_z;

    // Compute SVD and optimal rotation
    nk_f32_t cross_covariance[9] = {h00, h01, h02, h10, h11, h12, h20, h21, h22};
    nk_f32_t svd_u[9], svd_s[3], svd_v[9];
    nk_svd3x3_f32_(cross_covariance, svd_u, svd_s, svd_v);

    // R = V * Uᵀ
    nk_f32_t r[9];
    r[0] = svd_v[0] * svd_u[0] + svd_v[1] * svd_u[1] + svd_v[2] * svd_u[2];
    r[1] = svd_v[0] * svd_u[3] + svd_v[1] * svd_u[4] + svd_v[2] * svd_u[5];
    r[2] = svd_v[0] * svd_u[6] + svd_v[1] * svd_u[7] + svd_v[2] * svd_u[8];
    r[3] = svd_v[3] * svd_u[0] + svd_v[4] * svd_u[1] + svd_v[5] * svd_u[2];
    r[4] = svd_v[3] * svd_u[3] + svd_v[4] * svd_u[4] + svd_v[5] * svd_u[5];
    r[5] = svd_v[3] * svd_u[6] + svd_v[4] * svd_u[7] + svd_v[5] * svd_u[8];
    r[6] = svd_v[6] * svd_u[0] + svd_v[7] * svd_u[1] + svd_v[8] * svd_u[2];
    r[7] = svd_v[6] * svd_u[3] + svd_v[7] * svd_u[4] + svd_v[8] * svd_u[5];
    r[8] = svd_v[6] * svd_u[6] + svd_v[7] * svd_u[7] + svd_v[8] * svd_u[8];

    // Handle reflection: if det(R) < 0, negate third column of V and recompute R
    if (nk_det3x3_f32_(r) < 0) {
        svd_v[2] = -svd_v[2];
        svd_v[5] = -svd_v[5];
        svd_v[8] = -svd_v[8];
        r[0] = svd_v[0] * svd_u[0] + svd_v[1] * svd_u[1] + svd_v[2] * svd_u[2];
        r[1] = svd_v[0] * svd_u[3] + svd_v[1] * svd_u[4] + svd_v[2] * svd_u[5];
        r[2] = svd_v[0] * svd_u[6] + svd_v[1] * svd_u[7] + svd_v[2] * svd_u[8];
        r[3] = svd_v[3] * svd_u[0] + svd_v[4] * svd_u[1] + svd_v[5] * svd_u[2];
        r[4] = svd_v[3] * svd_u[3] + svd_v[4] * svd_u[4] + svd_v[5] * svd_u[5];
        r[5] = svd_v[3] * svd_u[6] + svd_v[4] * svd_u[7] + svd_v[5] * svd_u[8];
        r[6] = svd_v[6] * svd_u[0] + svd_v[7] * svd_u[1] + svd_v[8] * svd_u[2];
        r[7] = svd_v[6] * svd_u[3] + svd_v[7] * svd_u[4] + svd_v[8] * svd_u[5];
        r[8] = svd_v[6] * svd_u[6] + svd_v[7] * svd_u[7] + svd_v[8] * svd_u[8];
    }

    /* Output rotation matrix and scale=1.0 */
    if (rotation) {
        for (int j = 0; j < 9; ++j) rotation[j] = r[j];
    }
    if (scale) *scale = 1.0f;

    // Compute RMSD after optimal rotation
    nk_f32_t sum_squared = nk_transformed_ssd_f32_neon_(a, b, n, r, 1.0f, centroid_a_x, centroid_a_y, centroid_a_z,
                                                        centroid_b_x, centroid_b_y, centroid_b_z);
    *result = nk_f32_sqrt_neon(sum_squared * inv_n);
}

NK_PUBLIC void nk_kabsch_f64_neon(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                                  nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result) {
    float64x2_t const zeros_f64x2 = vdupq_n_f64(0);

    /*  2x unrolling with dual accumulators to hide FMA latency. */
    float64x2_t sum_a_x_a_f64x2 = zeros_f64x2, sum_a_y_a_f64x2 = zeros_f64x2, sum_a_z_a_f64x2 = zeros_f64x2;
    float64x2_t sum_b_x_a_f64x2 = zeros_f64x2, sum_b_y_a_f64x2 = zeros_f64x2, sum_b_z_a_f64x2 = zeros_f64x2;
    float64x2_t sum_a_x_b_f64x2 = zeros_f64x2, sum_a_y_b_f64x2 = zeros_f64x2, sum_a_z_b_f64x2 = zeros_f64x2;
    float64x2_t sum_b_x_b_f64x2 = zeros_f64x2, sum_b_y_b_f64x2 = zeros_f64x2, sum_b_z_b_f64x2 = zeros_f64x2;

    float64x2_t cov_xx_a_f64x2 = zeros_f64x2, cov_xy_a_f64x2 = zeros_f64x2, cov_xz_a_f64x2 = zeros_f64x2;
    float64x2_t cov_yx_a_f64x2 = zeros_f64x2, cov_yy_a_f64x2 = zeros_f64x2, cov_yz_a_f64x2 = zeros_f64x2;
    float64x2_t cov_zx_a_f64x2 = zeros_f64x2, cov_zy_a_f64x2 = zeros_f64x2, cov_zz_a_f64x2 = zeros_f64x2;
    float64x2_t cov_xx_b_f64x2 = zeros_f64x2, cov_xy_b_f64x2 = zeros_f64x2, cov_xz_b_f64x2 = zeros_f64x2;
    float64x2_t cov_yx_b_f64x2 = zeros_f64x2, cov_yy_b_f64x2 = zeros_f64x2, cov_yz_b_f64x2 = zeros_f64x2;
    float64x2_t cov_zx_b_f64x2 = zeros_f64x2, cov_zy_b_f64x2 = zeros_f64x2, cov_zz_b_f64x2 = zeros_f64x2;

    nk_size_t i = 0;
    float64x2_t a1_x_f64x2, a1_y_f64x2, a1_z_f64x2, b1_x_f64x2, b1_y_f64x2, b1_z_f64x2;
    float64x2_t a2_x_f64x2, a2_y_f64x2, a2_z_f64x2, b2_x_f64x2, b2_y_f64x2, b2_z_f64x2;

    // Main loop: 4 points per iteration (2x unrolled)
    for (; i + 4 <= n; i += 4) {
        nk_deinterleave_f64x2_neon_(a + i * 3, &a1_x_f64x2, &a1_y_f64x2, &a1_z_f64x2);
        nk_deinterleave_f64x2_neon_(b + i * 3, &b1_x_f64x2, &b1_y_f64x2, &b1_z_f64x2);
        nk_deinterleave_f64x2_neon_(a + (i + 2) * 3, &a2_x_f64x2, &a2_y_f64x2, &a2_z_f64x2);
        nk_deinterleave_f64x2_neon_(b + (i + 2) * 3, &b2_x_f64x2, &b2_y_f64x2, &b2_z_f64x2);

        // Interleaved accumulation
        sum_a_x_a_f64x2 = vaddq_f64(sum_a_x_a_f64x2, a1_x_f64x2);
        sum_a_x_b_f64x2 = vaddq_f64(sum_a_x_b_f64x2, a2_x_f64x2);
        sum_a_y_a_f64x2 = vaddq_f64(sum_a_y_a_f64x2, a1_y_f64x2);
        sum_a_y_b_f64x2 = vaddq_f64(sum_a_y_b_f64x2, a2_y_f64x2);
        sum_a_z_a_f64x2 = vaddq_f64(sum_a_z_a_f64x2, a1_z_f64x2);
        sum_a_z_b_f64x2 = vaddq_f64(sum_a_z_b_f64x2, a2_z_f64x2);
        sum_b_x_a_f64x2 = vaddq_f64(sum_b_x_a_f64x2, b1_x_f64x2);
        sum_b_x_b_f64x2 = vaddq_f64(sum_b_x_b_f64x2, b2_x_f64x2);
        sum_b_y_a_f64x2 = vaddq_f64(sum_b_y_a_f64x2, b1_y_f64x2);
        sum_b_y_b_f64x2 = vaddq_f64(sum_b_y_b_f64x2, b2_y_f64x2);
        sum_b_z_a_f64x2 = vaddq_f64(sum_b_z_a_f64x2, b1_z_f64x2);
        sum_b_z_b_f64x2 = vaddq_f64(sum_b_z_b_f64x2, b2_z_f64x2);

        cov_xx_a_f64x2 = vfmaq_f64(cov_xx_a_f64x2, a1_x_f64x2, b1_x_f64x2);
        cov_xx_b_f64x2 = vfmaq_f64(cov_xx_b_f64x2, a2_x_f64x2, b2_x_f64x2);
        cov_xy_a_f64x2 = vfmaq_f64(cov_xy_a_f64x2, a1_x_f64x2, b1_y_f64x2);
        cov_xy_b_f64x2 = vfmaq_f64(cov_xy_b_f64x2, a2_x_f64x2, b2_y_f64x2);
        cov_xz_a_f64x2 = vfmaq_f64(cov_xz_a_f64x2, a1_x_f64x2, b1_z_f64x2);
        cov_xz_b_f64x2 = vfmaq_f64(cov_xz_b_f64x2, a2_x_f64x2, b2_z_f64x2);
        cov_yx_a_f64x2 = vfmaq_f64(cov_yx_a_f64x2, a1_y_f64x2, b1_x_f64x2);
        cov_yx_b_f64x2 = vfmaq_f64(cov_yx_b_f64x2, a2_y_f64x2, b2_x_f64x2);
        cov_yy_a_f64x2 = vfmaq_f64(cov_yy_a_f64x2, a1_y_f64x2, b1_y_f64x2);
        cov_yy_b_f64x2 = vfmaq_f64(cov_yy_b_f64x2, a2_y_f64x2, b2_y_f64x2);
        cov_yz_a_f64x2 = vfmaq_f64(cov_yz_a_f64x2, a1_y_f64x2, b1_z_f64x2);
        cov_yz_b_f64x2 = vfmaq_f64(cov_yz_b_f64x2, a2_y_f64x2, b2_z_f64x2);
        cov_zx_a_f64x2 = vfmaq_f64(cov_zx_a_f64x2, a1_z_f64x2, b1_x_f64x2);
        cov_zx_b_f64x2 = vfmaq_f64(cov_zx_b_f64x2, a2_z_f64x2, b2_x_f64x2);
        cov_zy_a_f64x2 = vfmaq_f64(cov_zy_a_f64x2, a1_z_f64x2, b1_y_f64x2);
        cov_zy_b_f64x2 = vfmaq_f64(cov_zy_b_f64x2, a2_z_f64x2, b2_y_f64x2);
        cov_zz_a_f64x2 = vfmaq_f64(cov_zz_a_f64x2, a1_z_f64x2, b1_z_f64x2);
        cov_zz_b_f64x2 = vfmaq_f64(cov_zz_b_f64x2, a2_z_f64x2, b2_z_f64x2);
    }

    // 2-point tail
    for (; i + 2 <= n; i += 2) {
        nk_deinterleave_f64x2_neon_(a + i * 3, &a1_x_f64x2, &a1_y_f64x2, &a1_z_f64x2);
        nk_deinterleave_f64x2_neon_(b + i * 3, &b1_x_f64x2, &b1_y_f64x2, &b1_z_f64x2);
        sum_a_x_a_f64x2 = vaddq_f64(sum_a_x_a_f64x2, a1_x_f64x2);
        sum_a_y_a_f64x2 = vaddq_f64(sum_a_y_a_f64x2, a1_y_f64x2);
        sum_a_z_a_f64x2 = vaddq_f64(sum_a_z_a_f64x2, a1_z_f64x2);
        sum_b_x_a_f64x2 = vaddq_f64(sum_b_x_a_f64x2, b1_x_f64x2);
        sum_b_y_a_f64x2 = vaddq_f64(sum_b_y_a_f64x2, b1_y_f64x2);
        sum_b_z_a_f64x2 = vaddq_f64(sum_b_z_a_f64x2, b1_z_f64x2);
        cov_xx_a_f64x2 = vfmaq_f64(cov_xx_a_f64x2, a1_x_f64x2, b1_x_f64x2);
        cov_xy_a_f64x2 = vfmaq_f64(cov_xy_a_f64x2, a1_x_f64x2, b1_y_f64x2);
        cov_xz_a_f64x2 = vfmaq_f64(cov_xz_a_f64x2, a1_x_f64x2, b1_z_f64x2);
        cov_yx_a_f64x2 = vfmaq_f64(cov_yx_a_f64x2, a1_y_f64x2, b1_x_f64x2);
        cov_yy_a_f64x2 = vfmaq_f64(cov_yy_a_f64x2, a1_y_f64x2, b1_y_f64x2);
        cov_yz_a_f64x2 = vfmaq_f64(cov_yz_a_f64x2, a1_y_f64x2, b1_z_f64x2);
        cov_zx_a_f64x2 = vfmaq_f64(cov_zx_a_f64x2, a1_z_f64x2, b1_x_f64x2);
        cov_zy_a_f64x2 = vfmaq_f64(cov_zy_a_f64x2, a1_z_f64x2, b1_y_f64x2);
        cov_zz_a_f64x2 = vfmaq_f64(cov_zz_a_f64x2, a1_z_f64x2, b1_z_f64x2);
    }

    // Combine dual accumulators
    float64x2_t sum_a_x_f64x2 = vaddq_f64(sum_a_x_a_f64x2, sum_a_x_b_f64x2);
    float64x2_t sum_a_y_f64x2 = vaddq_f64(sum_a_y_a_f64x2, sum_a_y_b_f64x2);
    float64x2_t sum_a_z_f64x2 = vaddq_f64(sum_a_z_a_f64x2, sum_a_z_b_f64x2);
    float64x2_t sum_b_x_f64x2 = vaddq_f64(sum_b_x_a_f64x2, sum_b_x_b_f64x2);
    float64x2_t sum_b_y_f64x2 = vaddq_f64(sum_b_y_a_f64x2, sum_b_y_b_f64x2);
    float64x2_t sum_b_z_f64x2 = vaddq_f64(sum_b_z_a_f64x2, sum_b_z_b_f64x2);
    float64x2_t cov_xx_f64x2 = vaddq_f64(cov_xx_a_f64x2, cov_xx_b_f64x2);
    float64x2_t cov_xy_f64x2 = vaddq_f64(cov_xy_a_f64x2, cov_xy_b_f64x2);
    float64x2_t cov_xz_f64x2 = vaddq_f64(cov_xz_a_f64x2, cov_xz_b_f64x2);
    float64x2_t cov_yx_f64x2 = vaddq_f64(cov_yx_a_f64x2, cov_yx_b_f64x2);
    float64x2_t cov_yy_f64x2 = vaddq_f64(cov_yy_a_f64x2, cov_yy_b_f64x2);
    float64x2_t cov_yz_f64x2 = vaddq_f64(cov_yz_a_f64x2, cov_yz_b_f64x2);
    float64x2_t cov_zx_f64x2 = vaddq_f64(cov_zx_a_f64x2, cov_zx_b_f64x2);
    float64x2_t cov_zy_f64x2 = vaddq_f64(cov_zy_a_f64x2, cov_zy_b_f64x2);
    float64x2_t cov_zz_f64x2 = vaddq_f64(cov_zz_a_f64x2, cov_zz_b_f64x2);

    // Reduce vector accumulators
    nk_f64_t sum_a_x = vaddvq_f64(sum_a_x_f64x2);
    nk_f64_t sum_a_y = vaddvq_f64(sum_a_y_f64x2);
    nk_f64_t sum_a_z = vaddvq_f64(sum_a_z_f64x2);
    nk_f64_t sum_b_x = vaddvq_f64(sum_b_x_f64x2);
    nk_f64_t sum_b_y = vaddvq_f64(sum_b_y_f64x2);
    nk_f64_t sum_b_z = vaddvq_f64(sum_b_z_f64x2);

    nk_f64_t h00 = vaddvq_f64(cov_xx_f64x2);
    nk_f64_t h01 = vaddvq_f64(cov_xy_f64x2);
    nk_f64_t h02 = vaddvq_f64(cov_xz_f64x2);
    nk_f64_t h10 = vaddvq_f64(cov_yx_f64x2);
    nk_f64_t h11 = vaddvq_f64(cov_yy_f64x2);
    nk_f64_t h12 = vaddvq_f64(cov_yz_f64x2);
    nk_f64_t h20 = vaddvq_f64(cov_zx_f64x2);
    nk_f64_t h21 = vaddvq_f64(cov_zy_f64x2);
    nk_f64_t h22 = vaddvq_f64(cov_zz_f64x2);

    // Scalar tail
    for (; i < n; ++i) {
        nk_f64_t ax = a[i * 3 + 0], ay = a[i * 3 + 1], az = a[i * 3 + 2];
        nk_f64_t bx = b[i * 3 + 0], by = b[i * 3 + 1], bz = b[i * 3 + 2];
        sum_a_x += ax;
        sum_a_y += ay;
        sum_a_z += az;
        sum_b_x += bx;
        sum_b_y += by;
        sum_b_z += bz;
        h00 += ax * bx;
        h01 += ax * by;
        h02 += ax * bz;
        h10 += ay * bx;
        h11 += ay * by;
        h12 += ay * bz;
        h20 += az * bx;
        h21 += az * by;
        h22 += az * bz;
    }

    // Compute centroids
    nk_f64_t inv_n = 1.0 / (nk_f64_t)n;
    nk_f64_t centroid_a_x = sum_a_x * inv_n;
    nk_f64_t centroid_a_y = sum_a_y * inv_n;
    nk_f64_t centroid_a_z = sum_a_z * inv_n;
    nk_f64_t centroid_b_x = sum_b_x * inv_n;
    nk_f64_t centroid_b_y = sum_b_y * inv_n;
    nk_f64_t centroid_b_z = sum_b_z * inv_n;

    if (a_centroid) {
        a_centroid[0] = centroid_a_x;
        a_centroid[1] = centroid_a_y;
        a_centroid[2] = centroid_a_z;
    }
    if (b_centroid) {
        b_centroid[0] = centroid_b_x;
        b_centroid[1] = centroid_b_y;
        b_centroid[2] = centroid_b_z;
    }

    // Apply centering correction: H_centered = H - n * centroid_a * centroid_bᵀ
    h00 -= n * centroid_a_x * centroid_b_x;
    h01 -= n * centroid_a_x * centroid_b_y;
    h02 -= n * centroid_a_x * centroid_b_z;
    h10 -= n * centroid_a_y * centroid_b_x;
    h11 -= n * centroid_a_y * centroid_b_y;
    h12 -= n * centroid_a_y * centroid_b_z;
    h20 -= n * centroid_a_z * centroid_b_x;
    h21 -= n * centroid_a_z * centroid_b_y;
    h22 -= n * centroid_a_z * centroid_b_z;

    // Compute SVD and optimal rotation (using f32 SVD for performance)
    nk_f32_t cross_covariance[9] = {(nk_f32_t)h00, (nk_f32_t)h01, (nk_f32_t)h02, (nk_f32_t)h10, (nk_f32_t)h11,
                                    (nk_f32_t)h12, (nk_f32_t)h20, (nk_f32_t)h21, (nk_f32_t)h22};
    nk_f32_t svd_u[9], svd_s[3], svd_v[9];
    nk_svd3x3_f32_(cross_covariance, svd_u, svd_s, svd_v);

    // R = V * Uᵀ
    nk_f32_t r[9];
    r[0] = svd_v[0] * svd_u[0] + svd_v[1] * svd_u[1] + svd_v[2] * svd_u[2];
    r[1] = svd_v[0] * svd_u[3] + svd_v[1] * svd_u[4] + svd_v[2] * svd_u[5];
    r[2] = svd_v[0] * svd_u[6] + svd_v[1] * svd_u[7] + svd_v[2] * svd_u[8];
    r[3] = svd_v[3] * svd_u[0] + svd_v[4] * svd_u[1] + svd_v[5] * svd_u[2];
    r[4] = svd_v[3] * svd_u[3] + svd_v[4] * svd_u[4] + svd_v[5] * svd_u[5];
    r[5] = svd_v[3] * svd_u[6] + svd_v[4] * svd_u[7] + svd_v[5] * svd_u[8];
    r[6] = svd_v[6] * svd_u[0] + svd_v[7] * svd_u[1] + svd_v[8] * svd_u[2];
    r[7] = svd_v[6] * svd_u[3] + svd_v[7] * svd_u[4] + svd_v[8] * svd_u[5];
    r[8] = svd_v[6] * svd_u[6] + svd_v[7] * svd_u[7] + svd_v[8] * svd_u[8];

    // Handle reflection: if det(R) < 0, negate third column of V and recompute R
    if (nk_det3x3_f32_(r) < 0) {
        svd_v[2] = -svd_v[2];
        svd_v[5] = -svd_v[5];
        svd_v[8] = -svd_v[8];
        r[0] = svd_v[0] * svd_u[0] + svd_v[1] * svd_u[1] + svd_v[2] * svd_u[2];
        r[1] = svd_v[0] * svd_u[3] + svd_v[1] * svd_u[4] + svd_v[2] * svd_u[5];
        r[2] = svd_v[0] * svd_u[6] + svd_v[1] * svd_u[7] + svd_v[2] * svd_u[8];
        r[3] = svd_v[3] * svd_u[0] + svd_v[4] * svd_u[1] + svd_v[5] * svd_u[2];
        r[4] = svd_v[3] * svd_u[3] + svd_v[4] * svd_u[4] + svd_v[5] * svd_u[5];
        r[5] = svd_v[3] * svd_u[6] + svd_v[4] * svd_u[7] + svd_v[5] * svd_u[8];
        r[6] = svd_v[6] * svd_u[0] + svd_v[7] * svd_u[1] + svd_v[8] * svd_u[2];
        r[7] = svd_v[6] * svd_u[3] + svd_v[7] * svd_u[4] + svd_v[8] * svd_u[5];
        r[8] = svd_v[6] * svd_u[6] + svd_v[7] * svd_u[7] + svd_v[8] * svd_u[8];
    }

    /* Output rotation matrix and scale=1.0 */
    if (rotation) {
        for (int j = 0; j < 9; ++j) rotation[j] = (nk_f64_t)r[j];
    }
    if (scale) *scale = 1.0;

    // Compute RMSD after optimal rotation
    nk_f64_t sum_squared = nk_transformed_ssd_f64_neon_(a, b, n, r, 1.0, centroid_a_x, centroid_a_y, centroid_a_z,
                                                        centroid_b_x, centroid_b_y, centroid_b_z);
    *result = nk_f64_sqrt_neon(sum_squared * inv_n);
}

NK_PUBLIC void nk_umeyama_f32_neon(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                   nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result) {
    float32x4_t const zeros_f32x4 = vdupq_n_f32(0);

    /*  2x unrolling with dual accumulators to hide FMA latency. */
    float32x4_t sum_a_x_a_f32x4 = zeros_f32x4, sum_a_y_a_f32x4 = zeros_f32x4, sum_a_z_a_f32x4 = zeros_f32x4;
    float32x4_t sum_b_x_a_f32x4 = zeros_f32x4, sum_b_y_a_f32x4 = zeros_f32x4, sum_b_z_a_f32x4 = zeros_f32x4;
    float32x4_t sum_a_x_b_f32x4 = zeros_f32x4, sum_a_y_b_f32x4 = zeros_f32x4, sum_a_z_b_f32x4 = zeros_f32x4;
    float32x4_t sum_b_x_b_f32x4 = zeros_f32x4, sum_b_y_b_f32x4 = zeros_f32x4, sum_b_z_b_f32x4 = zeros_f32x4;

    float32x4_t cov_xx_a_f32x4 = zeros_f32x4, cov_xy_a_f32x4 = zeros_f32x4, cov_xz_a_f32x4 = zeros_f32x4;
    float32x4_t cov_yx_a_f32x4 = zeros_f32x4, cov_yy_a_f32x4 = zeros_f32x4, cov_yz_a_f32x4 = zeros_f32x4;
    float32x4_t cov_zx_a_f32x4 = zeros_f32x4, cov_zy_a_f32x4 = zeros_f32x4, cov_zz_a_f32x4 = zeros_f32x4;
    float32x4_t cov_xx_b_f32x4 = zeros_f32x4, cov_xy_b_f32x4 = zeros_f32x4, cov_xz_b_f32x4 = zeros_f32x4;
    float32x4_t cov_yx_b_f32x4 = zeros_f32x4, cov_yy_b_f32x4 = zeros_f32x4, cov_yz_b_f32x4 = zeros_f32x4;
    float32x4_t cov_zx_b_f32x4 = zeros_f32x4, cov_zy_b_f32x4 = zeros_f32x4, cov_zz_b_f32x4 = zeros_f32x4;
    float32x4_t variance_a_a_f32x4 = zeros_f32x4, variance_a_b_f32x4 = zeros_f32x4;

    nk_size_t i = 0;
    float32x4_t a1_x_f32x4, a1_y_f32x4, a1_z_f32x4, b1_x_f32x4, b1_y_f32x4, b1_z_f32x4;
    float32x4_t a2_x_f32x4, a2_y_f32x4, a2_z_f32x4, b2_x_f32x4, b2_y_f32x4, b2_z_f32x4;

    // Main loop: 8 points per iteration (2x unrolled)
    for (; i + 8 <= n; i += 8) {
        nk_deinterleave_f32x4_neon_(a + i * 3, &a1_x_f32x4, &a1_y_f32x4, &a1_z_f32x4);
        nk_deinterleave_f32x4_neon_(b + i * 3, &b1_x_f32x4, &b1_y_f32x4, &b1_z_f32x4);
        nk_deinterleave_f32x4_neon_(a + (i + 4) * 3, &a2_x_f32x4, &a2_y_f32x4, &a2_z_f32x4);
        nk_deinterleave_f32x4_neon_(b + (i + 4) * 3, &b2_x_f32x4, &b2_y_f32x4, &b2_z_f32x4);

        // Interleaved accumulation
        sum_a_x_a_f32x4 = vaddq_f32(sum_a_x_a_f32x4, a1_x_f32x4);
        sum_a_x_b_f32x4 = vaddq_f32(sum_a_x_b_f32x4, a2_x_f32x4);
        sum_a_y_a_f32x4 = vaddq_f32(sum_a_y_a_f32x4, a1_y_f32x4);
        sum_a_y_b_f32x4 = vaddq_f32(sum_a_y_b_f32x4, a2_y_f32x4);
        sum_a_z_a_f32x4 = vaddq_f32(sum_a_z_a_f32x4, a1_z_f32x4);
        sum_a_z_b_f32x4 = vaddq_f32(sum_a_z_b_f32x4, a2_z_f32x4);
        sum_b_x_a_f32x4 = vaddq_f32(sum_b_x_a_f32x4, b1_x_f32x4);
        sum_b_x_b_f32x4 = vaddq_f32(sum_b_x_b_f32x4, b2_x_f32x4);
        sum_b_y_a_f32x4 = vaddq_f32(sum_b_y_a_f32x4, b1_y_f32x4);
        sum_b_y_b_f32x4 = vaddq_f32(sum_b_y_b_f32x4, b2_y_f32x4);
        sum_b_z_a_f32x4 = vaddq_f32(sum_b_z_a_f32x4, b1_z_f32x4);
        sum_b_z_b_f32x4 = vaddq_f32(sum_b_z_b_f32x4, b2_z_f32x4);

        cov_xx_a_f32x4 = vfmaq_f32(cov_xx_a_f32x4, a1_x_f32x4, b1_x_f32x4);
        cov_xx_b_f32x4 = vfmaq_f32(cov_xx_b_f32x4, a2_x_f32x4, b2_x_f32x4);
        cov_xy_a_f32x4 = vfmaq_f32(cov_xy_a_f32x4, a1_x_f32x4, b1_y_f32x4);
        cov_xy_b_f32x4 = vfmaq_f32(cov_xy_b_f32x4, a2_x_f32x4, b2_y_f32x4);
        cov_xz_a_f32x4 = vfmaq_f32(cov_xz_a_f32x4, a1_x_f32x4, b1_z_f32x4);
        cov_xz_b_f32x4 = vfmaq_f32(cov_xz_b_f32x4, a2_x_f32x4, b2_z_f32x4);
        cov_yx_a_f32x4 = vfmaq_f32(cov_yx_a_f32x4, a1_y_f32x4, b1_x_f32x4);
        cov_yx_b_f32x4 = vfmaq_f32(cov_yx_b_f32x4, a2_y_f32x4, b2_x_f32x4);
        cov_yy_a_f32x4 = vfmaq_f32(cov_yy_a_f32x4, a1_y_f32x4, b1_y_f32x4);
        cov_yy_b_f32x4 = vfmaq_f32(cov_yy_b_f32x4, a2_y_f32x4, b2_y_f32x4);
        cov_yz_a_f32x4 = vfmaq_f32(cov_yz_a_f32x4, a1_y_f32x4, b1_z_f32x4);
        cov_yz_b_f32x4 = vfmaq_f32(cov_yz_b_f32x4, a2_y_f32x4, b2_z_f32x4);
        cov_zx_a_f32x4 = vfmaq_f32(cov_zx_a_f32x4, a1_z_f32x4, b1_x_f32x4);
        cov_zx_b_f32x4 = vfmaq_f32(cov_zx_b_f32x4, a2_z_f32x4, b2_x_f32x4);
        cov_zy_a_f32x4 = vfmaq_f32(cov_zy_a_f32x4, a1_z_f32x4, b1_y_f32x4);
        cov_zy_b_f32x4 = vfmaq_f32(cov_zy_b_f32x4, a2_z_f32x4, b2_y_f32x4);
        cov_zz_a_f32x4 = vfmaq_f32(cov_zz_a_f32x4, a1_z_f32x4, b1_z_f32x4);
        cov_zz_b_f32x4 = vfmaq_f32(cov_zz_b_f32x4, a2_z_f32x4, b2_z_f32x4);

        variance_a_a_f32x4 = vfmaq_f32(variance_a_a_f32x4, a1_x_f32x4, a1_x_f32x4);
        variance_a_b_f32x4 = vfmaq_f32(variance_a_b_f32x4, a2_x_f32x4, a2_x_f32x4);
        variance_a_a_f32x4 = vfmaq_f32(variance_a_a_f32x4, a1_y_f32x4, a1_y_f32x4);
        variance_a_b_f32x4 = vfmaq_f32(variance_a_b_f32x4, a2_y_f32x4, a2_y_f32x4);
        variance_a_a_f32x4 = vfmaq_f32(variance_a_a_f32x4, a1_z_f32x4, a1_z_f32x4);
        variance_a_b_f32x4 = vfmaq_f32(variance_a_b_f32x4, a2_z_f32x4, a2_z_f32x4);
    }

    // 4-point tail
    for (; i + 4 <= n; i += 4) {
        nk_deinterleave_f32x4_neon_(a + i * 3, &a1_x_f32x4, &a1_y_f32x4, &a1_z_f32x4);
        nk_deinterleave_f32x4_neon_(b + i * 3, &b1_x_f32x4, &b1_y_f32x4, &b1_z_f32x4);
        sum_a_x_a_f32x4 = vaddq_f32(sum_a_x_a_f32x4, a1_x_f32x4);
        sum_a_y_a_f32x4 = vaddq_f32(sum_a_y_a_f32x4, a1_y_f32x4);
        sum_a_z_a_f32x4 = vaddq_f32(sum_a_z_a_f32x4, a1_z_f32x4);
        sum_b_x_a_f32x4 = vaddq_f32(sum_b_x_a_f32x4, b1_x_f32x4);
        sum_b_y_a_f32x4 = vaddq_f32(sum_b_y_a_f32x4, b1_y_f32x4);
        sum_b_z_a_f32x4 = vaddq_f32(sum_b_z_a_f32x4, b1_z_f32x4);
        cov_xx_a_f32x4 = vfmaq_f32(cov_xx_a_f32x4, a1_x_f32x4, b1_x_f32x4);
        cov_xy_a_f32x4 = vfmaq_f32(cov_xy_a_f32x4, a1_x_f32x4, b1_y_f32x4);
        cov_xz_a_f32x4 = vfmaq_f32(cov_xz_a_f32x4, a1_x_f32x4, b1_z_f32x4);
        cov_yx_a_f32x4 = vfmaq_f32(cov_yx_a_f32x4, a1_y_f32x4, b1_x_f32x4);
        cov_yy_a_f32x4 = vfmaq_f32(cov_yy_a_f32x4, a1_y_f32x4, b1_y_f32x4);
        cov_yz_a_f32x4 = vfmaq_f32(cov_yz_a_f32x4, a1_y_f32x4, b1_z_f32x4);
        cov_zx_a_f32x4 = vfmaq_f32(cov_zx_a_f32x4, a1_z_f32x4, b1_x_f32x4);
        cov_zy_a_f32x4 = vfmaq_f32(cov_zy_a_f32x4, a1_z_f32x4, b1_y_f32x4);
        cov_zz_a_f32x4 = vfmaq_f32(cov_zz_a_f32x4, a1_z_f32x4, b1_z_f32x4);
        variance_a_a_f32x4 = vfmaq_f32(variance_a_a_f32x4, a1_x_f32x4, a1_x_f32x4);
        variance_a_a_f32x4 = vfmaq_f32(variance_a_a_f32x4, a1_y_f32x4, a1_y_f32x4);
        variance_a_a_f32x4 = vfmaq_f32(variance_a_a_f32x4, a1_z_f32x4, a1_z_f32x4);
    }

    // Combine dual accumulators
    float32x4_t sum_a_x_f32x4 = vaddq_f32(sum_a_x_a_f32x4, sum_a_x_b_f32x4);
    float32x4_t sum_a_y_f32x4 = vaddq_f32(sum_a_y_a_f32x4, sum_a_y_b_f32x4);
    float32x4_t sum_a_z_f32x4 = vaddq_f32(sum_a_z_a_f32x4, sum_a_z_b_f32x4);
    float32x4_t sum_b_x_f32x4 = vaddq_f32(sum_b_x_a_f32x4, sum_b_x_b_f32x4);
    float32x4_t sum_b_y_f32x4 = vaddq_f32(sum_b_y_a_f32x4, sum_b_y_b_f32x4);
    float32x4_t sum_b_z_f32x4 = vaddq_f32(sum_b_z_a_f32x4, sum_b_z_b_f32x4);
    float32x4_t cov_xx_f32x4 = vaddq_f32(cov_xx_a_f32x4, cov_xx_b_f32x4);
    float32x4_t cov_xy_f32x4 = vaddq_f32(cov_xy_a_f32x4, cov_xy_b_f32x4);
    float32x4_t cov_xz_f32x4 = vaddq_f32(cov_xz_a_f32x4, cov_xz_b_f32x4);
    float32x4_t cov_yx_f32x4 = vaddq_f32(cov_yx_a_f32x4, cov_yx_b_f32x4);
    float32x4_t cov_yy_f32x4 = vaddq_f32(cov_yy_a_f32x4, cov_yy_b_f32x4);
    float32x4_t cov_yz_f32x4 = vaddq_f32(cov_yz_a_f32x4, cov_yz_b_f32x4);
    float32x4_t cov_zx_f32x4 = vaddq_f32(cov_zx_a_f32x4, cov_zx_b_f32x4);
    float32x4_t cov_zy_f32x4 = vaddq_f32(cov_zy_a_f32x4, cov_zy_b_f32x4);
    float32x4_t cov_zz_f32x4 = vaddq_f32(cov_zz_a_f32x4, cov_zz_b_f32x4);
    float32x4_t variance_a_f32x4 = vaddq_f32(variance_a_a_f32x4, variance_a_b_f32x4);

    // Reduce vector accumulators
    nk_f32_t sum_a_x = vaddvq_f32(sum_a_x_f32x4);
    nk_f32_t sum_a_y = vaddvq_f32(sum_a_y_f32x4);
    nk_f32_t sum_a_z = vaddvq_f32(sum_a_z_f32x4);
    nk_f32_t sum_b_x = vaddvq_f32(sum_b_x_f32x4);
    nk_f32_t sum_b_y = vaddvq_f32(sum_b_y_f32x4);
    nk_f32_t sum_b_z = vaddvq_f32(sum_b_z_f32x4);
    nk_f32_t h00 = vaddvq_f32(cov_xx_f32x4);
    nk_f32_t h01 = vaddvq_f32(cov_xy_f32x4);
    nk_f32_t h02 = vaddvq_f32(cov_xz_f32x4);
    nk_f32_t h10 = vaddvq_f32(cov_yx_f32x4);
    nk_f32_t h11 = vaddvq_f32(cov_yy_f32x4);
    nk_f32_t h12 = vaddvq_f32(cov_yz_f32x4);
    nk_f32_t h20 = vaddvq_f32(cov_zx_f32x4);
    nk_f32_t h21 = vaddvq_f32(cov_zy_f32x4);
    nk_f32_t h22 = vaddvq_f32(cov_zz_f32x4);
    nk_f32_t sum_sq_a = vaddvq_f32(variance_a_f32x4);

    // Scalar tail
    for (; i < n; ++i) {
        nk_f32_t ax = a[i * 3 + 0], ay = a[i * 3 + 1], az = a[i * 3 + 2];
        nk_f32_t bx = b[i * 3 + 0], by = b[i * 3 + 1], bz = b[i * 3 + 2];
        sum_a_x += ax;
        sum_a_y += ay;
        sum_a_z += az;
        sum_b_x += bx;
        sum_b_y += by;
        sum_b_z += bz;
        h00 += ax * bx;
        h01 += ax * by;
        h02 += ax * bz;
        h10 += ay * bx;
        h11 += ay * by;
        h12 += ay * bz;
        h20 += az * bx;
        h21 += az * by;
        h22 += az * bz;
        sum_sq_a += ax * ax + ay * ay + az * az;
    }

    // Compute centroids
    nk_f32_t inv_n = 1.0f / (nk_f32_t)n;
    nk_f32_t centroid_a_x = sum_a_x * inv_n;
    nk_f32_t centroid_a_y = sum_a_y * inv_n;
    nk_f32_t centroid_a_z = sum_a_z * inv_n;
    nk_f32_t centroid_b_x = sum_b_x * inv_n;
    nk_f32_t centroid_b_y = sum_b_y * inv_n;
    nk_f32_t centroid_b_z = sum_b_z * inv_n;

    if (a_centroid) {
        a_centroid[0] = centroid_a_x;
        a_centroid[1] = centroid_a_y;
        a_centroid[2] = centroid_a_z;
    }
    if (b_centroid) {
        b_centroid[0] = centroid_b_x;
        b_centroid[1] = centroid_b_y;
        b_centroid[2] = centroid_b_z;
    }

    // Compute variance of A (centered)
    nk_f32_t centroid_sq = centroid_a_x * centroid_a_x + centroid_a_y * centroid_a_y + centroid_a_z * centroid_a_z;
    nk_f32_t var_a = sum_sq_a * inv_n - centroid_sq;

    // Apply centering correction: H_centered = H - n * centroid_a * centroid_bᵀ
    h00 -= n * centroid_a_x * centroid_b_x;
    h01 -= n * centroid_a_x * centroid_b_y;
    h02 -= n * centroid_a_x * centroid_b_z;
    h10 -= n * centroid_a_y * centroid_b_x;
    h11 -= n * centroid_a_y * centroid_b_y;
    h12 -= n * centroid_a_y * centroid_b_z;
    h20 -= n * centroid_a_z * centroid_b_x;
    h21 -= n * centroid_a_z * centroid_b_y;
    h22 -= n * centroid_a_z * centroid_b_z;

    // Compute SVD
    nk_f32_t cross_covariance[9] = {h00, h01, h02, h10, h11, h12, h20, h21, h22};
    nk_f32_t svd_u[9], svd_s[3], svd_v[9];
    nk_svd3x3_f32_(cross_covariance, svd_u, svd_s, svd_v);

    // R = V * Uᵀ
    nk_f32_t r[9];
    r[0] = svd_v[0] * svd_u[0] + svd_v[1] * svd_u[1] + svd_v[2] * svd_u[2];
    r[1] = svd_v[0] * svd_u[3] + svd_v[1] * svd_u[4] + svd_v[2] * svd_u[5];
    r[2] = svd_v[0] * svd_u[6] + svd_v[1] * svd_u[7] + svd_v[2] * svd_u[8];
    r[3] = svd_v[3] * svd_u[0] + svd_v[4] * svd_u[1] + svd_v[5] * svd_u[2];
    r[4] = svd_v[3] * svd_u[3] + svd_v[4] * svd_u[4] + svd_v[5] * svd_u[5];
    r[5] = svd_v[3] * svd_u[6] + svd_v[4] * svd_u[7] + svd_v[5] * svd_u[8];
    r[6] = svd_v[6] * svd_u[0] + svd_v[7] * svd_u[1] + svd_v[8] * svd_u[2];
    r[7] = svd_v[6] * svd_u[3] + svd_v[7] * svd_u[4] + svd_v[8] * svd_u[5];
    r[8] = svd_v[6] * svd_u[6] + svd_v[7] * svd_u[7] + svd_v[8] * svd_u[8];

    // Handle reflection and compute scale
    nk_f32_t det = nk_det3x3_f32_(r);
    nk_f32_t trace_d_s = svd_s[0] + svd_s[1] + (det < 0 ? -svd_s[2] : svd_s[2]);
    nk_f32_t computed_scale = trace_d_s / (n * var_a);

    if (det < 0) {
        svd_v[2] = -svd_v[2];
        svd_v[5] = -svd_v[5];
        svd_v[8] = -svd_v[8];
        r[0] = svd_v[0] * svd_u[0] + svd_v[1] * svd_u[1] + svd_v[2] * svd_u[2];
        r[1] = svd_v[0] * svd_u[3] + svd_v[1] * svd_u[4] + svd_v[2] * svd_u[5];
        r[2] = svd_v[0] * svd_u[6] + svd_v[1] * svd_u[7] + svd_v[2] * svd_u[8];
        r[3] = svd_v[3] * svd_u[0] + svd_v[4] * svd_u[1] + svd_v[5] * svd_u[2];
        r[4] = svd_v[3] * svd_u[3] + svd_v[4] * svd_u[4] + svd_v[5] * svd_u[5];
        r[5] = svd_v[3] * svd_u[6] + svd_v[4] * svd_u[7] + svd_v[5] * svd_u[8];
        r[6] = svd_v[6] * svd_u[0] + svd_v[7] * svd_u[1] + svd_v[8] * svd_u[2];
        r[7] = svd_v[6] * svd_u[3] + svd_v[7] * svd_u[4] + svd_v[8] * svd_u[5];
        r[8] = svd_v[6] * svd_u[6] + svd_v[7] * svd_u[7] + svd_v[8] * svd_u[8];
    }

    if (rotation) {
        for (int j = 0; j < 9; ++j) rotation[j] = r[j];
    }
    if (scale) *scale = computed_scale;

    // Compute RMSD after transformation
    nk_f32_t sum_squared = nk_transformed_ssd_f32_neon_(a, b, n, r, computed_scale, centroid_a_x, centroid_a_y,
                                                        centroid_a_z, centroid_b_x, centroid_b_y, centroid_b_z);
    *result = nk_f32_sqrt_neon(sum_squared * inv_n);
}

NK_PUBLIC void nk_umeyama_f64_neon(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                                   nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result) {
    float64x2_t const zeros_f64x2 = vdupq_n_f64(0);

    /*  2x unrolling with dual accumulators to hide FMA latency. */
    float64x2_t sum_a_x_a_f64x2 = zeros_f64x2, sum_a_y_a_f64x2 = zeros_f64x2, sum_a_z_a_f64x2 = zeros_f64x2;
    float64x2_t sum_b_x_a_f64x2 = zeros_f64x2, sum_b_y_a_f64x2 = zeros_f64x2, sum_b_z_a_f64x2 = zeros_f64x2;
    float64x2_t sum_a_x_b_f64x2 = zeros_f64x2, sum_a_y_b_f64x2 = zeros_f64x2, sum_a_z_b_f64x2 = zeros_f64x2;
    float64x2_t sum_b_x_b_f64x2 = zeros_f64x2, sum_b_y_b_f64x2 = zeros_f64x2, sum_b_z_b_f64x2 = zeros_f64x2;

    float64x2_t cov_xx_a_f64x2 = zeros_f64x2, cov_xy_a_f64x2 = zeros_f64x2, cov_xz_a_f64x2 = zeros_f64x2;
    float64x2_t cov_yx_a_f64x2 = zeros_f64x2, cov_yy_a_f64x2 = zeros_f64x2, cov_yz_a_f64x2 = zeros_f64x2;
    float64x2_t cov_zx_a_f64x2 = zeros_f64x2, cov_zy_a_f64x2 = zeros_f64x2, cov_zz_a_f64x2 = zeros_f64x2;
    float64x2_t cov_xx_b_f64x2 = zeros_f64x2, cov_xy_b_f64x2 = zeros_f64x2, cov_xz_b_f64x2 = zeros_f64x2;
    float64x2_t cov_yx_b_f64x2 = zeros_f64x2, cov_yy_b_f64x2 = zeros_f64x2, cov_yz_b_f64x2 = zeros_f64x2;
    float64x2_t cov_zx_b_f64x2 = zeros_f64x2, cov_zy_b_f64x2 = zeros_f64x2, cov_zz_b_f64x2 = zeros_f64x2;
    float64x2_t variance_a_a_f64x2 = zeros_f64x2, variance_a_b_f64x2 = zeros_f64x2;

    nk_size_t i = 0;
    float64x2_t a1_x_f64x2, a1_y_f64x2, a1_z_f64x2, b1_x_f64x2, b1_y_f64x2, b1_z_f64x2;
    float64x2_t a2_x_f64x2, a2_y_f64x2, a2_z_f64x2, b2_x_f64x2, b2_y_f64x2, b2_z_f64x2;

    // Main loop: 4 points per iteration (2x unrolled)
    for (; i + 4 <= n; i += 4) {
        nk_deinterleave_f64x2_neon_(a + i * 3, &a1_x_f64x2, &a1_y_f64x2, &a1_z_f64x2);
        nk_deinterleave_f64x2_neon_(b + i * 3, &b1_x_f64x2, &b1_y_f64x2, &b1_z_f64x2);
        nk_deinterleave_f64x2_neon_(a + (i + 2) * 3, &a2_x_f64x2, &a2_y_f64x2, &a2_z_f64x2);
        nk_deinterleave_f64x2_neon_(b + (i + 2) * 3, &b2_x_f64x2, &b2_y_f64x2, &b2_z_f64x2);

        // Interleaved accumulation
        sum_a_x_a_f64x2 = vaddq_f64(sum_a_x_a_f64x2, a1_x_f64x2);
        sum_a_x_b_f64x2 = vaddq_f64(sum_a_x_b_f64x2, a2_x_f64x2);
        sum_a_y_a_f64x2 = vaddq_f64(sum_a_y_a_f64x2, a1_y_f64x2);
        sum_a_y_b_f64x2 = vaddq_f64(sum_a_y_b_f64x2, a2_y_f64x2);
        sum_a_z_a_f64x2 = vaddq_f64(sum_a_z_a_f64x2, a1_z_f64x2);
        sum_a_z_b_f64x2 = vaddq_f64(sum_a_z_b_f64x2, a2_z_f64x2);
        sum_b_x_a_f64x2 = vaddq_f64(sum_b_x_a_f64x2, b1_x_f64x2);
        sum_b_x_b_f64x2 = vaddq_f64(sum_b_x_b_f64x2, b2_x_f64x2);
        sum_b_y_a_f64x2 = vaddq_f64(sum_b_y_a_f64x2, b1_y_f64x2);
        sum_b_y_b_f64x2 = vaddq_f64(sum_b_y_b_f64x2, b2_y_f64x2);
        sum_b_z_a_f64x2 = vaddq_f64(sum_b_z_a_f64x2, b1_z_f64x2);
        sum_b_z_b_f64x2 = vaddq_f64(sum_b_z_b_f64x2, b2_z_f64x2);

        cov_xx_a_f64x2 = vfmaq_f64(cov_xx_a_f64x2, a1_x_f64x2, b1_x_f64x2);
        cov_xx_b_f64x2 = vfmaq_f64(cov_xx_b_f64x2, a2_x_f64x2, b2_x_f64x2);
        cov_xy_a_f64x2 = vfmaq_f64(cov_xy_a_f64x2, a1_x_f64x2, b1_y_f64x2);
        cov_xy_b_f64x2 = vfmaq_f64(cov_xy_b_f64x2, a2_x_f64x2, b2_y_f64x2);
        cov_xz_a_f64x2 = vfmaq_f64(cov_xz_a_f64x2, a1_x_f64x2, b1_z_f64x2);
        cov_xz_b_f64x2 = vfmaq_f64(cov_xz_b_f64x2, a2_x_f64x2, b2_z_f64x2);
        cov_yx_a_f64x2 = vfmaq_f64(cov_yx_a_f64x2, a1_y_f64x2, b1_x_f64x2);
        cov_yx_b_f64x2 = vfmaq_f64(cov_yx_b_f64x2, a2_y_f64x2, b2_x_f64x2);
        cov_yy_a_f64x2 = vfmaq_f64(cov_yy_a_f64x2, a1_y_f64x2, b1_y_f64x2);
        cov_yy_b_f64x2 = vfmaq_f64(cov_yy_b_f64x2, a2_y_f64x2, b2_y_f64x2);
        cov_yz_a_f64x2 = vfmaq_f64(cov_yz_a_f64x2, a1_y_f64x2, b1_z_f64x2);
        cov_yz_b_f64x2 = vfmaq_f64(cov_yz_b_f64x2, a2_y_f64x2, b2_z_f64x2);
        cov_zx_a_f64x2 = vfmaq_f64(cov_zx_a_f64x2, a1_z_f64x2, b1_x_f64x2);
        cov_zx_b_f64x2 = vfmaq_f64(cov_zx_b_f64x2, a2_z_f64x2, b2_x_f64x2);
        cov_zy_a_f64x2 = vfmaq_f64(cov_zy_a_f64x2, a1_z_f64x2, b1_y_f64x2);
        cov_zy_b_f64x2 = vfmaq_f64(cov_zy_b_f64x2, a2_z_f64x2, b2_y_f64x2);
        cov_zz_a_f64x2 = vfmaq_f64(cov_zz_a_f64x2, a1_z_f64x2, b1_z_f64x2);
        cov_zz_b_f64x2 = vfmaq_f64(cov_zz_b_f64x2, a2_z_f64x2, b2_z_f64x2);

        variance_a_a_f64x2 = vfmaq_f64(variance_a_a_f64x2, a1_x_f64x2, a1_x_f64x2);
        variance_a_b_f64x2 = vfmaq_f64(variance_a_b_f64x2, a2_x_f64x2, a2_x_f64x2);
        variance_a_a_f64x2 = vfmaq_f64(variance_a_a_f64x2, a1_y_f64x2, a1_y_f64x2);
        variance_a_b_f64x2 = vfmaq_f64(variance_a_b_f64x2, a2_y_f64x2, a2_y_f64x2);
        variance_a_a_f64x2 = vfmaq_f64(variance_a_a_f64x2, a1_z_f64x2, a1_z_f64x2);
        variance_a_b_f64x2 = vfmaq_f64(variance_a_b_f64x2, a2_z_f64x2, a2_z_f64x2);
    }

    // 2-point tail
    for (; i + 2 <= n; i += 2) {
        nk_deinterleave_f64x2_neon_(a + i * 3, &a1_x_f64x2, &a1_y_f64x2, &a1_z_f64x2);
        nk_deinterleave_f64x2_neon_(b + i * 3, &b1_x_f64x2, &b1_y_f64x2, &b1_z_f64x2);
        sum_a_x_a_f64x2 = vaddq_f64(sum_a_x_a_f64x2, a1_x_f64x2);
        sum_a_y_a_f64x2 = vaddq_f64(sum_a_y_a_f64x2, a1_y_f64x2);
        sum_a_z_a_f64x2 = vaddq_f64(sum_a_z_a_f64x2, a1_z_f64x2);
        sum_b_x_a_f64x2 = vaddq_f64(sum_b_x_a_f64x2, b1_x_f64x2);
        sum_b_y_a_f64x2 = vaddq_f64(sum_b_y_a_f64x2, b1_y_f64x2);
        sum_b_z_a_f64x2 = vaddq_f64(sum_b_z_a_f64x2, b1_z_f64x2);
        cov_xx_a_f64x2 = vfmaq_f64(cov_xx_a_f64x2, a1_x_f64x2, b1_x_f64x2);
        cov_xy_a_f64x2 = vfmaq_f64(cov_xy_a_f64x2, a1_x_f64x2, b1_y_f64x2);
        cov_xz_a_f64x2 = vfmaq_f64(cov_xz_a_f64x2, a1_x_f64x2, b1_z_f64x2);
        cov_yx_a_f64x2 = vfmaq_f64(cov_yx_a_f64x2, a1_y_f64x2, b1_x_f64x2);
        cov_yy_a_f64x2 = vfmaq_f64(cov_yy_a_f64x2, a1_y_f64x2, b1_y_f64x2);
        cov_yz_a_f64x2 = vfmaq_f64(cov_yz_a_f64x2, a1_y_f64x2, b1_z_f64x2);
        cov_zx_a_f64x2 = vfmaq_f64(cov_zx_a_f64x2, a1_z_f64x2, b1_x_f64x2);
        cov_zy_a_f64x2 = vfmaq_f64(cov_zy_a_f64x2, a1_z_f64x2, b1_y_f64x2);
        cov_zz_a_f64x2 = vfmaq_f64(cov_zz_a_f64x2, a1_z_f64x2, b1_z_f64x2);
        variance_a_a_f64x2 = vfmaq_f64(variance_a_a_f64x2, a1_x_f64x2, a1_x_f64x2);
        variance_a_a_f64x2 = vfmaq_f64(variance_a_a_f64x2, a1_y_f64x2, a1_y_f64x2);
        variance_a_a_f64x2 = vfmaq_f64(variance_a_a_f64x2, a1_z_f64x2, a1_z_f64x2);
    }

    // Combine dual accumulators
    float64x2_t sum_a_x_f64x2 = vaddq_f64(sum_a_x_a_f64x2, sum_a_x_b_f64x2);
    float64x2_t sum_a_y_f64x2 = vaddq_f64(sum_a_y_a_f64x2, sum_a_y_b_f64x2);
    float64x2_t sum_a_z_f64x2 = vaddq_f64(sum_a_z_a_f64x2, sum_a_z_b_f64x2);
    float64x2_t sum_b_x_f64x2 = vaddq_f64(sum_b_x_a_f64x2, sum_b_x_b_f64x2);
    float64x2_t sum_b_y_f64x2 = vaddq_f64(sum_b_y_a_f64x2, sum_b_y_b_f64x2);
    float64x2_t sum_b_z_f64x2 = vaddq_f64(sum_b_z_a_f64x2, sum_b_z_b_f64x2);
    float64x2_t cov_xx_f64x2 = vaddq_f64(cov_xx_a_f64x2, cov_xx_b_f64x2);
    float64x2_t cov_xy_f64x2 = vaddq_f64(cov_xy_a_f64x2, cov_xy_b_f64x2);
    float64x2_t cov_xz_f64x2 = vaddq_f64(cov_xz_a_f64x2, cov_xz_b_f64x2);
    float64x2_t cov_yx_f64x2 = vaddq_f64(cov_yx_a_f64x2, cov_yx_b_f64x2);
    float64x2_t cov_yy_f64x2 = vaddq_f64(cov_yy_a_f64x2, cov_yy_b_f64x2);
    float64x2_t cov_yz_f64x2 = vaddq_f64(cov_yz_a_f64x2, cov_yz_b_f64x2);
    float64x2_t cov_zx_f64x2 = vaddq_f64(cov_zx_a_f64x2, cov_zx_b_f64x2);
    float64x2_t cov_zy_f64x2 = vaddq_f64(cov_zy_a_f64x2, cov_zy_b_f64x2);
    float64x2_t cov_zz_f64x2 = vaddq_f64(cov_zz_a_f64x2, cov_zz_b_f64x2);
    float64x2_t variance_a_f64x2 = vaddq_f64(variance_a_a_f64x2, variance_a_b_f64x2);

    // Reduce vector accumulators
    nk_f64_t sum_a_x = vaddvq_f64(sum_a_x_f64x2);
    nk_f64_t sum_a_y = vaddvq_f64(sum_a_y_f64x2);
    nk_f64_t sum_a_z = vaddvq_f64(sum_a_z_f64x2);
    nk_f64_t sum_b_x = vaddvq_f64(sum_b_x_f64x2);
    nk_f64_t sum_b_y = vaddvq_f64(sum_b_y_f64x2);
    nk_f64_t sum_b_z = vaddvq_f64(sum_b_z_f64x2);
    nk_f64_t h00 = vaddvq_f64(cov_xx_f64x2);
    nk_f64_t h01 = vaddvq_f64(cov_xy_f64x2);
    nk_f64_t h02 = vaddvq_f64(cov_xz_f64x2);
    nk_f64_t h10 = vaddvq_f64(cov_yx_f64x2);
    nk_f64_t h11 = vaddvq_f64(cov_yy_f64x2);
    nk_f64_t h12 = vaddvq_f64(cov_yz_f64x2);
    nk_f64_t h20 = vaddvq_f64(cov_zx_f64x2);
    nk_f64_t h21 = vaddvq_f64(cov_zy_f64x2);
    nk_f64_t h22 = vaddvq_f64(cov_zz_f64x2);
    nk_f64_t sum_sq_a = vaddvq_f64(variance_a_f64x2);

    // Scalar tail
    for (; i < n; ++i) {
        nk_f64_t ax = a[i * 3 + 0], ay = a[i * 3 + 1], az = a[i * 3 + 2];
        nk_f64_t bx = b[i * 3 + 0], by = b[i * 3 + 1], bz = b[i * 3 + 2];
        sum_a_x += ax;
        sum_a_y += ay;
        sum_a_z += az;
        sum_b_x += bx;
        sum_b_y += by;
        sum_b_z += bz;
        h00 += ax * bx;
        h01 += ax * by;
        h02 += ax * bz;
        h10 += ay * bx;
        h11 += ay * by;
        h12 += ay * bz;
        h20 += az * bx;
        h21 += az * by;
        h22 += az * bz;
        sum_sq_a += ax * ax + ay * ay + az * az;
    }

    // Compute centroids
    nk_f64_t inv_n = 1.0 / (nk_f64_t)n;
    nk_f64_t centroid_a_x = sum_a_x * inv_n;
    nk_f64_t centroid_a_y = sum_a_y * inv_n;
    nk_f64_t centroid_a_z = sum_a_z * inv_n;
    nk_f64_t centroid_b_x = sum_b_x * inv_n;
    nk_f64_t centroid_b_y = sum_b_y * inv_n;
    nk_f64_t centroid_b_z = sum_b_z * inv_n;

    if (a_centroid) {
        a_centroid[0] = centroid_a_x;
        a_centroid[1] = centroid_a_y;
        a_centroid[2] = centroid_a_z;
    }
    if (b_centroid) {
        b_centroid[0] = centroid_b_x;
        b_centroid[1] = centroid_b_y;
        b_centroid[2] = centroid_b_z;
    }

    // Compute variance of A (centered)
    nk_f64_t centroid_sq = centroid_a_x * centroid_a_x + centroid_a_y * centroid_a_y + centroid_a_z * centroid_a_z;
    nk_f64_t var_a = sum_sq_a * inv_n - centroid_sq;

    // Apply centering correction: H_centered = H - n * centroid_a * centroid_bᵀ
    h00 -= n * centroid_a_x * centroid_b_x;
    h01 -= n * centroid_a_x * centroid_b_y;
    h02 -= n * centroid_a_x * centroid_b_z;
    h10 -= n * centroid_a_y * centroid_b_x;
    h11 -= n * centroid_a_y * centroid_b_y;
    h12 -= n * centroid_a_y * centroid_b_z;
    h20 -= n * centroid_a_z * centroid_b_x;
    h21 -= n * centroid_a_z * centroid_b_y;
    h22 -= n * centroid_a_z * centroid_b_z;

    // Compute SVD (using f32 SVD for performance)
    nk_f32_t cross_covariance[9] = {(nk_f32_t)h00, (nk_f32_t)h01, (nk_f32_t)h02, (nk_f32_t)h10, (nk_f32_t)h11,
                                    (nk_f32_t)h12, (nk_f32_t)h20, (nk_f32_t)h21, (nk_f32_t)h22};
    nk_f32_t svd_u[9], svd_s[3], svd_v[9];
    nk_svd3x3_f32_(cross_covariance, svd_u, svd_s, svd_v);

    // R = V * Uᵀ
    nk_f32_t r[9];
    r[0] = svd_v[0] * svd_u[0] + svd_v[1] * svd_u[1] + svd_v[2] * svd_u[2];
    r[1] = svd_v[0] * svd_u[3] + svd_v[1] * svd_u[4] + svd_v[2] * svd_u[5];
    r[2] = svd_v[0] * svd_u[6] + svd_v[1] * svd_u[7] + svd_v[2] * svd_u[8];
    r[3] = svd_v[3] * svd_u[0] + svd_v[4] * svd_u[1] + svd_v[5] * svd_u[2];
    r[4] = svd_v[3] * svd_u[3] + svd_v[4] * svd_u[4] + svd_v[5] * svd_u[5];
    r[5] = svd_v[3] * svd_u[6] + svd_v[4] * svd_u[7] + svd_v[5] * svd_u[8];
    r[6] = svd_v[6] * svd_u[0] + svd_v[7] * svd_u[1] + svd_v[8] * svd_u[2];
    r[7] = svd_v[6] * svd_u[3] + svd_v[7] * svd_u[4] + svd_v[8] * svd_u[5];
    r[8] = svd_v[6] * svd_u[6] + svd_v[7] * svd_u[7] + svd_v[8] * svd_u[8];

    // Handle reflection and compute scale
    nk_f32_t det = nk_det3x3_f32_(r);
    nk_f64_t trace_d_s = svd_s[0] + svd_s[1] + (det < 0 ? -svd_s[2] : svd_s[2]);
    nk_f64_t computed_scale = trace_d_s / (n * var_a);

    if (det < 0) {
        svd_v[2] = -svd_v[2];
        svd_v[5] = -svd_v[5];
        svd_v[8] = -svd_v[8];
        r[0] = svd_v[0] * svd_u[0] + svd_v[1] * svd_u[1] + svd_v[2] * svd_u[2];
        r[1] = svd_v[0] * svd_u[3] + svd_v[1] * svd_u[4] + svd_v[2] * svd_u[5];
        r[2] = svd_v[0] * svd_u[6] + svd_v[1] * svd_u[7] + svd_v[2] * svd_u[8];
        r[3] = svd_v[3] * svd_u[0] + svd_v[4] * svd_u[1] + svd_v[5] * svd_u[2];
        r[4] = svd_v[3] * svd_u[3] + svd_v[4] * svd_u[4] + svd_v[5] * svd_u[5];
        r[5] = svd_v[3] * svd_u[6] + svd_v[4] * svd_u[7] + svd_v[5] * svd_u[8];
        r[6] = svd_v[6] * svd_u[0] + svd_v[7] * svd_u[1] + svd_v[8] * svd_u[2];
        r[7] = svd_v[6] * svd_u[3] + svd_v[7] * svd_u[4] + svd_v[8] * svd_u[5];
        r[8] = svd_v[6] * svd_u[6] + svd_v[7] * svd_u[7] + svd_v[8] * svd_u[8];
    }

    if (rotation) {
        for (int j = 0; j < 9; ++j) rotation[j] = (nk_f64_t)r[j];
    }
    if (scale) *scale = computed_scale;

    // Compute RMSD after transformation
    nk_f64_t sum_squared = nk_transformed_ssd_f64_neon_(a, b, n, r, computed_scale, centroid_a_x, centroid_a_y,
                                                        centroid_a_z, centroid_b_x, centroid_b_y, centroid_b_z);
    *result = nk_f64_sqrt_neon(sum_squared * inv_n);
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

#endif // NK_MESH_NEON_H

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
#pragma GCC push_options
#pragma GCC target("arch=armv8-a+simd")
#pragma clang attribute push(__attribute__((target("arch=armv8-a+simd"))), apply_to = function)

#include "numkong/types.h"
#include "numkong/spatial/neon.h" // For nk_sqrt_f32_neon_(), nk_sqrt_f64_neon_()

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
    // NEON doesn't have vld3q_f64, so we load manually
    nk_f64_t x0 = ptr[0], x1 = ptr[3];
    nk_f64_t y0 = ptr[1], y1 = ptr[4];
    nk_f64_t z0 = ptr[2], z1 = ptr[5];

    nk_f64_t x_arr[2] = {x0, x1};
    nk_f64_t y_arr[2] = {y0, y1};
    nk_f64_t z_arr[2] = {z0, z1};

    *x_out = vld1q_f64(x_arr);
    *y_out = vld1q_f64(y_arr);
    *z_out = vld1q_f64(z_arr);
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

    *result = nk_sqrt_f32_neon_(sum_squared * inv_n - mean_diff_sq);
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

    *result = nk_sqrt_f64_neon_(sum_squared * inv_n - mean_diff_sq);
}

NK_PUBLIC void nk_kabsch_f32_neon(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                  nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result) {
    float32x4_t const zeros_f32x4 = vdupq_n_f32(0);

    // Accumulators for centroids (f32)
    float32x4_t sum_a_x_f32x4 = zeros_f32x4, sum_a_y_f32x4 = zeros_f32x4, sum_a_z_f32x4 = zeros_f32x4;
    float32x4_t sum_b_x_f32x4 = zeros_f32x4, sum_b_y_f32x4 = zeros_f32x4, sum_b_z_f32x4 = zeros_f32x4;

    // Accumulators for covariance matrix (sum of outer products)
    float32x4_t cov_xx_f32x4 = zeros_f32x4, cov_xy_f32x4 = zeros_f32x4, cov_xz_f32x4 = zeros_f32x4;
    float32x4_t cov_yx_f32x4 = zeros_f32x4, cov_yy_f32x4 = zeros_f32x4, cov_yz_f32x4 = zeros_f32x4;
    float32x4_t cov_zx_f32x4 = zeros_f32x4, cov_zy_f32x4 = zeros_f32x4, cov_zz_f32x4 = zeros_f32x4;

    nk_size_t i = 0;
    float32x4_t a_x_f32x4, a_y_f32x4, a_z_f32x4, b_x_f32x4, b_y_f32x4, b_z_f32x4;

    // Fused single-pass: accumulate sums and outer products together
    for (; i + 4 <= n; i += 4) {
        nk_deinterleave_f32x4_neon_(a + i * 3, &a_x_f32x4, &a_y_f32x4, &a_z_f32x4);
        nk_deinterleave_f32x4_neon_(b + i * 3, &b_x_f32x4, &b_y_f32x4, &b_z_f32x4);

        // Accumulate centroids directly in f32
        sum_a_x_f32x4 = vaddq_f32(sum_a_x_f32x4, a_x_f32x4);
        sum_a_y_f32x4 = vaddq_f32(sum_a_y_f32x4, a_y_f32x4);
        sum_a_z_f32x4 = vaddq_f32(sum_a_z_f32x4, a_z_f32x4);
        sum_b_x_f32x4 = vaddq_f32(sum_b_x_f32x4, b_x_f32x4);
        sum_b_y_f32x4 = vaddq_f32(sum_b_y_f32x4, b_y_f32x4);
        sum_b_z_f32x4 = vaddq_f32(sum_b_z_f32x4, b_z_f32x4);

        // Accumulate outer products (raw, not centered) in f32
        cov_xx_f32x4 = vfmaq_f32(cov_xx_f32x4, a_x_f32x4, b_x_f32x4);
        cov_xy_f32x4 = vfmaq_f32(cov_xy_f32x4, a_x_f32x4, b_y_f32x4);
        cov_xz_f32x4 = vfmaq_f32(cov_xz_f32x4, a_x_f32x4, b_z_f32x4);
        cov_yx_f32x4 = vfmaq_f32(cov_yx_f32x4, a_y_f32x4, b_x_f32x4);
        cov_yy_f32x4 = vfmaq_f32(cov_yy_f32x4, a_y_f32x4, b_y_f32x4);
        cov_yz_f32x4 = vfmaq_f32(cov_yz_f32x4, a_y_f32x4, b_z_f32x4);
        cov_zx_f32x4 = vfmaq_f32(cov_zx_f32x4, a_z_f32x4, b_x_f32x4);
        cov_zy_f32x4 = vfmaq_f32(cov_zy_f32x4, a_z_f32x4, b_y_f32x4);
        cov_zz_f32x4 = vfmaq_f32(cov_zz_f32x4, a_z_f32x4, b_z_f32x4);
    }

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

    // Apply centering correction: H_centered = H - n * centroid_a * centroid_b^T
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
    nk_svd3x3__f32(cross_covariance, svd_u, svd_s, svd_v);

    // R = V * U^T
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
    if (nk_det3x3__f32(r) < 0) {
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
    nk_f32_t sum_squared = 0.0f;
    for (nk_size_t j = 0; j < n; ++j) {
        nk_f32_t pa[3], pb[3], ra[3];
        pa[0] = a[j * 3 + 0] - centroid_a_x;
        pa[1] = a[j * 3 + 1] - centroid_a_y;
        pa[2] = a[j * 3 + 2] - centroid_a_z;
        pb[0] = b[j * 3 + 0] - centroid_b_x;
        pb[1] = b[j * 3 + 1] - centroid_b_y;
        pb[2] = b[j * 3 + 2] - centroid_b_z;

        ra[0] = r[0] * pa[0] + r[1] * pa[1] + r[2] * pa[2];
        ra[1] = r[3] * pa[0] + r[4] * pa[1] + r[5] * pa[2];
        ra[2] = r[6] * pa[0] + r[7] * pa[1] + r[8] * pa[2];

        nk_f32_t delta_x = ra[0] - pb[0];
        nk_f32_t delta_y = ra[1] - pb[1];
        nk_f32_t delta_z = ra[2] - pb[2];
        sum_squared += delta_x * delta_x + delta_y * delta_y + delta_z * delta_z;
    }

    *result = nk_sqrt_f32_neon_(sum_squared * inv_n);
}

NK_PUBLIC void nk_kabsch_f64_neon(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                                  nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result) {
    float64x2_t const zeros_f64x2 = vdupq_n_f64(0);

    // Accumulators for centroids
    float64x2_t sum_a_x_f64x2 = zeros_f64x2, sum_a_y_f64x2 = zeros_f64x2, sum_a_z_f64x2 = zeros_f64x2;
    float64x2_t sum_b_x_f64x2 = zeros_f64x2, sum_b_y_f64x2 = zeros_f64x2, sum_b_z_f64x2 = zeros_f64x2;

    // Accumulators for covariance matrix (sum of outer products)
    float64x2_t cov_xx_f64x2 = zeros_f64x2, cov_xy_f64x2 = zeros_f64x2, cov_xz_f64x2 = zeros_f64x2;
    float64x2_t cov_yx_f64x2 = zeros_f64x2, cov_yy_f64x2 = zeros_f64x2, cov_yz_f64x2 = zeros_f64x2;
    float64x2_t cov_zx_f64x2 = zeros_f64x2, cov_zy_f64x2 = zeros_f64x2, cov_zz_f64x2 = zeros_f64x2;

    nk_size_t i = 0;
    float64x2_t a_x_f64x2, a_y_f64x2, a_z_f64x2, b_x_f64x2, b_y_f64x2, b_z_f64x2;

    // Fused single-pass
    for (; i + 2 <= n; i += 2) {
        nk_deinterleave_f64x2_neon_(a + i * 3, &a_x_f64x2, &a_y_f64x2, &a_z_f64x2);
        nk_deinterleave_f64x2_neon_(b + i * 3, &b_x_f64x2, &b_y_f64x2, &b_z_f64x2);

        sum_a_x_f64x2 = vaddq_f64(sum_a_x_f64x2, a_x_f64x2);
        sum_a_y_f64x2 = vaddq_f64(sum_a_y_f64x2, a_y_f64x2);
        sum_a_z_f64x2 = vaddq_f64(sum_a_z_f64x2, a_z_f64x2);
        sum_b_x_f64x2 = vaddq_f64(sum_b_x_f64x2, b_x_f64x2);
        sum_b_y_f64x2 = vaddq_f64(sum_b_y_f64x2, b_y_f64x2);
        sum_b_z_f64x2 = vaddq_f64(sum_b_z_f64x2, b_z_f64x2);

        cov_xx_f64x2 = vfmaq_f64(cov_xx_f64x2, a_x_f64x2, b_x_f64x2);
        cov_xy_f64x2 = vfmaq_f64(cov_xy_f64x2, a_x_f64x2, b_y_f64x2);
        cov_xz_f64x2 = vfmaq_f64(cov_xz_f64x2, a_x_f64x2, b_z_f64x2);
        cov_yx_f64x2 = vfmaq_f64(cov_yx_f64x2, a_y_f64x2, b_x_f64x2);
        cov_yy_f64x2 = vfmaq_f64(cov_yy_f64x2, a_y_f64x2, b_y_f64x2);
        cov_yz_f64x2 = vfmaq_f64(cov_yz_f64x2, a_y_f64x2, b_z_f64x2);
        cov_zx_f64x2 = vfmaq_f64(cov_zx_f64x2, a_z_f64x2, b_x_f64x2);
        cov_zy_f64x2 = vfmaq_f64(cov_zy_f64x2, a_z_f64x2, b_y_f64x2);
        cov_zz_f64x2 = vfmaq_f64(cov_zz_f64x2, a_z_f64x2, b_z_f64x2);
    }

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

    // Apply centering correction: H_centered = H - n * centroid_a * centroid_b^T
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
    nk_svd3x3__f32(cross_covariance, svd_u, svd_s, svd_v);

    // R = V * U^T
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
    if (nk_det3x3__f32(r) < 0) {
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

    // Compute RMSD after optimal rotation (use f64 for precision)
    nk_f64_t sum_squared = 0.0;
    for (nk_size_t j = 0; j < n; ++j) {
        nk_f64_t pa[3], pb[3], ra[3];
        pa[0] = a[j * 3 + 0] - centroid_a_x;
        pa[1] = a[j * 3 + 1] - centroid_a_y;
        pa[2] = a[j * 3 + 2] - centroid_a_z;
        pb[0] = b[j * 3 + 0] - centroid_b_x;
        pb[1] = b[j * 3 + 1] - centroid_b_y;
        pb[2] = b[j * 3 + 2] - centroid_b_z;

        ra[0] = r[0] * pa[0] + r[1] * pa[1] + r[2] * pa[2];
        ra[1] = r[3] * pa[0] + r[4] * pa[1] + r[5] * pa[2];
        ra[2] = r[6] * pa[0] + r[7] * pa[1] + r[8] * pa[2];

        nk_f64_t delta_x = ra[0] - pb[0];
        nk_f64_t delta_y = ra[1] - pb[1];
        nk_f64_t delta_z = ra[2] - pb[2];
        sum_squared += delta_x * delta_x + delta_y * delta_y + delta_z * delta_z;
    }

    *result = nk_sqrt_f64_neon_(sum_squared * inv_n);
}

NK_PUBLIC void nk_umeyama_f32_neon(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                   nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result) {
    // Fused single-pass: centroids, covariance, and variance of A using f32 numerics
    float32x4_t const zeros_f32x4 = vdupq_n_f32(0);

    float32x4_t sum_a_x_f32x4 = zeros_f32x4, sum_a_y_f32x4 = zeros_f32x4, sum_a_z_f32x4 = zeros_f32x4;
    float32x4_t sum_b_x_f32x4 = zeros_f32x4, sum_b_y_f32x4 = zeros_f32x4, sum_b_z_f32x4 = zeros_f32x4;
    float32x4_t cov_xx_f32x4 = zeros_f32x4, cov_xy_f32x4 = zeros_f32x4, cov_xz_f32x4 = zeros_f32x4;
    float32x4_t cov_yx_f32x4 = zeros_f32x4, cov_yy_f32x4 = zeros_f32x4, cov_yz_f32x4 = zeros_f32x4;
    float32x4_t cov_zx_f32x4 = zeros_f32x4, cov_zy_f32x4 = zeros_f32x4, cov_zz_f32x4 = zeros_f32x4;
    float32x4_t variance_a_f32x4 = zeros_f32x4;

    nk_size_t i = 0;
    float32x4_t a_x_f32x4, a_y_f32x4, a_z_f32x4, b_x_f32x4, b_y_f32x4, b_z_f32x4;

    for (; i + 4 <= n; i += 4) {
        nk_deinterleave_f32x4_neon_(a + i * 3, &a_x_f32x4, &a_y_f32x4, &a_z_f32x4);
        nk_deinterleave_f32x4_neon_(b + i * 3, &b_x_f32x4, &b_y_f32x4, &b_z_f32x4);

        // Accumulate centroids directly in f32
        sum_a_x_f32x4 = vaddq_f32(sum_a_x_f32x4, a_x_f32x4);
        sum_a_y_f32x4 = vaddq_f32(sum_a_y_f32x4, a_y_f32x4);
        sum_a_z_f32x4 = vaddq_f32(sum_a_z_f32x4, a_z_f32x4);
        sum_b_x_f32x4 = vaddq_f32(sum_b_x_f32x4, b_x_f32x4);
        sum_b_y_f32x4 = vaddq_f32(sum_b_y_f32x4, b_y_f32x4);
        sum_b_z_f32x4 = vaddq_f32(sum_b_z_f32x4, b_z_f32x4);

        // Accumulate outer products in f32
        cov_xx_f32x4 = vfmaq_f32(cov_xx_f32x4, a_x_f32x4, b_x_f32x4);
        cov_xy_f32x4 = vfmaq_f32(cov_xy_f32x4, a_x_f32x4, b_y_f32x4);
        cov_xz_f32x4 = vfmaq_f32(cov_xz_f32x4, a_x_f32x4, b_z_f32x4);
        cov_yx_f32x4 = vfmaq_f32(cov_yx_f32x4, a_y_f32x4, b_x_f32x4);
        cov_yy_f32x4 = vfmaq_f32(cov_yy_f32x4, a_y_f32x4, b_y_f32x4);
        cov_yz_f32x4 = vfmaq_f32(cov_yz_f32x4, a_y_f32x4, b_z_f32x4);
        cov_zx_f32x4 = vfmaq_f32(cov_zx_f32x4, a_z_f32x4, b_x_f32x4);
        cov_zy_f32x4 = vfmaq_f32(cov_zy_f32x4, a_z_f32x4, b_y_f32x4);
        cov_zz_f32x4 = vfmaq_f32(cov_zz_f32x4, a_z_f32x4, b_z_f32x4);

        // Accumulate variance of A
        variance_a_f32x4 = vfmaq_f32(variance_a_f32x4, a_x_f32x4, a_x_f32x4);
        variance_a_f32x4 = vfmaq_f32(variance_a_f32x4, a_y_f32x4, a_y_f32x4);
        variance_a_f32x4 = vfmaq_f32(variance_a_f32x4, a_z_f32x4, a_z_f32x4);
    }

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

    // Apply centering correction: H_centered = H - n * centroid_a * centroid_b^T
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
    nk_svd3x3__f32(cross_covariance, svd_u, svd_s, svd_v);

    // R = V * U^T
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
    nk_f32_t det = nk_det3x3__f32(r);
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
    nk_f32_t sum_squared = 0.0f;
    for (nk_size_t j = 0; j < n; ++j) {
        nk_f32_t pa[3], pb[3], ra[3];
        pa[0] = a[j * 3 + 0] - centroid_a_x;
        pa[1] = a[j * 3 + 1] - centroid_a_y;
        pa[2] = a[j * 3 + 2] - centroid_a_z;
        pb[0] = b[j * 3 + 0] - centroid_b_x;
        pb[1] = b[j * 3 + 1] - centroid_b_y;
        pb[2] = b[j * 3 + 2] - centroid_b_z;

        ra[0] = computed_scale * (r[0] * pa[0] + r[1] * pa[1] + r[2] * pa[2]);
        ra[1] = computed_scale * (r[3] * pa[0] + r[4] * pa[1] + r[5] * pa[2]);
        ra[2] = computed_scale * (r[6] * pa[0] + r[7] * pa[1] + r[8] * pa[2]);

        nk_f32_t delta_x = ra[0] - pb[0];
        nk_f32_t delta_y = ra[1] - pb[1];
        nk_f32_t delta_z = ra[2] - pb[2];
        sum_squared += delta_x * delta_x + delta_y * delta_y + delta_z * delta_z;
    }

    *result = nk_sqrt_f32_neon_(sum_squared * inv_n);
}

NK_PUBLIC void nk_umeyama_f64_neon(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                                   nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result) {
    // Fused single-pass: centroids, covariance, and variance of A using f64 numerics
    float64x2_t const zeros_f64x2 = vdupq_n_f64(0);

    float64x2_t sum_a_x_f64x2 = zeros_f64x2, sum_a_y_f64x2 = zeros_f64x2, sum_a_z_f64x2 = zeros_f64x2;
    float64x2_t sum_b_x_f64x2 = zeros_f64x2, sum_b_y_f64x2 = zeros_f64x2, sum_b_z_f64x2 = zeros_f64x2;
    float64x2_t cov_xx_f64x2 = zeros_f64x2, cov_xy_f64x2 = zeros_f64x2, cov_xz_f64x2 = zeros_f64x2;
    float64x2_t cov_yx_f64x2 = zeros_f64x2, cov_yy_f64x2 = zeros_f64x2, cov_yz_f64x2 = zeros_f64x2;
    float64x2_t cov_zx_f64x2 = zeros_f64x2, cov_zy_f64x2 = zeros_f64x2, cov_zz_f64x2 = zeros_f64x2;
    float64x2_t variance_a_f64x2 = zeros_f64x2;

    nk_size_t i = 0;
    float64x2_t a_x_f64x2, a_y_f64x2, a_z_f64x2, b_x_f64x2, b_y_f64x2, b_z_f64x2;

    for (; i + 2 <= n; i += 2) {
        nk_deinterleave_f64x2_neon_(a + i * 3, &a_x_f64x2, &a_y_f64x2, &a_z_f64x2);
        nk_deinterleave_f64x2_neon_(b + i * 3, &b_x_f64x2, &b_y_f64x2, &b_z_f64x2);

        // Accumulate centroids
        sum_a_x_f64x2 = vaddq_f64(sum_a_x_f64x2, a_x_f64x2);
        sum_a_y_f64x2 = vaddq_f64(sum_a_y_f64x2, a_y_f64x2);
        sum_a_z_f64x2 = vaddq_f64(sum_a_z_f64x2, a_z_f64x2);
        sum_b_x_f64x2 = vaddq_f64(sum_b_x_f64x2, b_x_f64x2);
        sum_b_y_f64x2 = vaddq_f64(sum_b_y_f64x2, b_y_f64x2);
        sum_b_z_f64x2 = vaddq_f64(sum_b_z_f64x2, b_z_f64x2);

        // Accumulate outer products
        cov_xx_f64x2 = vfmaq_f64(cov_xx_f64x2, a_x_f64x2, b_x_f64x2);
        cov_xy_f64x2 = vfmaq_f64(cov_xy_f64x2, a_x_f64x2, b_y_f64x2);
        cov_xz_f64x2 = vfmaq_f64(cov_xz_f64x2, a_x_f64x2, b_z_f64x2);
        cov_yx_f64x2 = vfmaq_f64(cov_yx_f64x2, a_y_f64x2, b_x_f64x2);
        cov_yy_f64x2 = vfmaq_f64(cov_yy_f64x2, a_y_f64x2, b_y_f64x2);
        cov_yz_f64x2 = vfmaq_f64(cov_yz_f64x2, a_y_f64x2, b_z_f64x2);
        cov_zx_f64x2 = vfmaq_f64(cov_zx_f64x2, a_z_f64x2, b_x_f64x2);
        cov_zy_f64x2 = vfmaq_f64(cov_zy_f64x2, a_z_f64x2, b_y_f64x2);
        cov_zz_f64x2 = vfmaq_f64(cov_zz_f64x2, a_z_f64x2, b_z_f64x2);

        // Accumulate variance of A
        variance_a_f64x2 = vfmaq_f64(variance_a_f64x2, a_x_f64x2, a_x_f64x2);
        variance_a_f64x2 = vfmaq_f64(variance_a_f64x2, a_y_f64x2, a_y_f64x2);
        variance_a_f64x2 = vfmaq_f64(variance_a_f64x2, a_z_f64x2, a_z_f64x2);
    }

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

    // Apply centering correction: H_centered = H - n * centroid_a * centroid_b^T
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
    nk_svd3x3__f32(cross_covariance, svd_u, svd_s, svd_v);

    // R = V * U^T
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
    nk_f32_t det = nk_det3x3__f32(r);
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
    nk_f64_t sum_squared = 0.0;
    for (nk_size_t j = 0; j < n; ++j) {
        nk_f64_t pa[3], pb[3], ra[3];
        pa[0] = a[j * 3 + 0] - centroid_a_x;
        pa[1] = a[j * 3 + 1] - centroid_a_y;
        pa[2] = a[j * 3 + 2] - centroid_a_z;
        pb[0] = b[j * 3 + 0] - centroid_b_x;
        pb[1] = b[j * 3 + 1] - centroid_b_y;
        pb[2] = b[j * 3 + 2] - centroid_b_z;

        ra[0] = computed_scale * (r[0] * pa[0] + r[1] * pa[1] + r[2] * pa[2]);
        ra[1] = computed_scale * (r[3] * pa[0] + r[4] * pa[1] + r[5] * pa[2]);
        ra[2] = computed_scale * (r[6] * pa[0] + r[7] * pa[1] + r[8] * pa[2]);

        nk_f64_t delta_x = ra[0] - pb[0];
        nk_f64_t delta_y = ra[1] - pb[1];
        nk_f64_t delta_z = ra[2] - pb[2];
        sum_squared += delta_x * delta_x + delta_y * delta_y + delta_z * delta_z;
    }

    *result = nk_sqrt_f64_neon_(sum_squared * inv_n);
}

#if defined(__cplusplus)
} // extern "C"
#endif

#pragma clang attribute pop
#pragma GCC pop_options
#endif // NK_TARGET_NEON
#endif // NK_TARGET_ARM_

#endif // NK_MESH_NEON_H

/**
 *  @brief SIMD-accelerated Dot Products for Real and Complex Numbers optimized for Intel Haswell CPUs.
 *  @file include/numkong/mesh/haswell.h
 *  @sa include/numkong/mesh.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 */
#ifndef NK_MESH_HASWELL_H
#define NK_MESH_HASWELL_H

#if _NK_TARGET_X86
#if NK_TARGET_HASWELL
#pragma GCC push_options
#pragma GCC target("avx2", "f16c", "fma")
#pragma clang attribute push(__attribute__((target("avx2,f16c,fma"))), apply_to = function)

#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

/*  Internal helper: Deinterleave 24 floats (8 xyz triplets) into separate x, y, z vectors.
 *  Uses AVX2 gather instructions for clean stride-3 access.
 *
 *  Input: 24 contiguous floats [x0,y0,z0, x1,y1,z1, ..., x7,y7,z7]
 *  Output: x[8], y[8], z[8] vectors
 */
NK_INTERNAL void _nk_deinterleave_f32x8_haswell(nk_f32_t const *ptr, __m256 *x_out, __m256 *y_out, __m256 *z_out) {
    // Gather indices: 0, 3, 6, 9, 12, 15, 18, 21 (stride 3)
    __m256i idx = _mm256_setr_epi32(0, 3, 6, 9, 12, 15, 18, 21);
    *x_out = _mm256_i32gather_ps(ptr + 0, idx, 4);
    *y_out = _mm256_i32gather_ps(ptr + 1, idx, 4);
    *z_out = _mm256_i32gather_ps(ptr + 2, idx, 4);
}

/*  Internal helper: Deinterleave 12 f64 values (4 xyz triplets) into separate x, y, z vectors.
 *  Uses scalar extraction for simplicity as AVX2 lacks efficient stride-3 gather for f64.
 *
 *  Input: 12 contiguous f64 [x0,y0,z0, x1,y1,z1, x2,y2,z2, x3,y3,z3]
 *  Output: x[4], y[4], z[4] vectors
 */
NK_INTERNAL void _nk_deinterleave_f64x4_haswell(nk_f64_t const *ptr, __m256d *x_out, __m256d *y_out, __m256d *z_out) {
    nk_f64_t x0 = ptr[0], x1 = ptr[3], x2 = ptr[6], x3 = ptr[9];
    nk_f64_t y0 = ptr[1], y1 = ptr[4], y2 = ptr[7], y3 = ptr[10];
    nk_f64_t z0 = ptr[2], z1 = ptr[5], z2 = ptr[8], z3 = ptr[11];

    *x_out = _mm256_setr_pd(x0, x1, x2, x3);
    *y_out = _mm256_setr_pd(y0, y1, y2, y3);
    *z_out = _mm256_setr_pd(z0, z1, z2, z3);
}

/* Horizontal reduction helpers moved to reduce.h:
 * - _nk_reduce_add_f32x8_haswell
 * - _nk_reduce_add_f64x4_haswell
 */

NK_PUBLIC void nk_rmsd_f32_haswell(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                   nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result) {
    /* RMSD uses identity rotation and scale=1.0 */
    if (rotation) {
        rotation[0] = 1, rotation[1] = 0, rotation[2] = 0;
        rotation[3] = 0, rotation[4] = 1, rotation[5] = 0;
        rotation[6] = 0, rotation[7] = 0, rotation[8] = 1;
    }
    if (scale) *scale = 1.0;
    // Optimized fused single-pass implementation using AVX2.
    // Computes centroids and squared differences in one pass.
    __m256 const zeros_f32x8 = _mm256_setzero_ps();

    // Accumulators for centroids and squared differences
    __m256 sum_a_x_f32x8 = zeros_f32x8, sum_a_y_f32x8 = zeros_f32x8, sum_a_z_f32x8 = zeros_f32x8;
    __m256 sum_b_x_f32x8 = zeros_f32x8, sum_b_y_f32x8 = zeros_f32x8, sum_b_z_f32x8 = zeros_f32x8;
    __m256 sum_squared_x_f32x8 = zeros_f32x8, sum_squared_y_f32x8 = zeros_f32x8, sum_squared_z_f32x8 = zeros_f32x8;

    __m256 a_x_f32x8, a_y_f32x8, a_z_f32x8, b_x_f32x8, b_y_f32x8, b_z_f32x8;
    nk_size_t i = 0;

    // Main loop with 2x unrolling
    for (; i + 16 <= n; i += 16) {
        // Iteration 0
        _nk_deinterleave_f32x8_haswell(a + i * 3, &a_x_f32x8, &a_y_f32x8, &a_z_f32x8);
        _nk_deinterleave_f32x8_haswell(b + i * 3, &b_x_f32x8, &b_y_f32x8, &b_z_f32x8);

        sum_a_x_f32x8 = _mm256_add_ps(sum_a_x_f32x8, a_x_f32x8);
        sum_a_y_f32x8 = _mm256_add_ps(sum_a_y_f32x8, a_y_f32x8);
        sum_a_z_f32x8 = _mm256_add_ps(sum_a_z_f32x8, a_z_f32x8);
        sum_b_x_f32x8 = _mm256_add_ps(sum_b_x_f32x8, b_x_f32x8);
        sum_b_y_f32x8 = _mm256_add_ps(sum_b_y_f32x8, b_y_f32x8);
        sum_b_z_f32x8 = _mm256_add_ps(sum_b_z_f32x8, b_z_f32x8);

        __m256 delta_x_f32x8 = _mm256_sub_ps(a_x_f32x8, b_x_f32x8);
        __m256 delta_y_f32x8 = _mm256_sub_ps(a_y_f32x8, b_y_f32x8);
        __m256 delta_z_f32x8 = _mm256_sub_ps(a_z_f32x8, b_z_f32x8);

        sum_squared_x_f32x8 = _mm256_fmadd_ps(delta_x_f32x8, delta_x_f32x8, sum_squared_x_f32x8);
        sum_squared_y_f32x8 = _mm256_fmadd_ps(delta_y_f32x8, delta_y_f32x8, sum_squared_y_f32x8);
        sum_squared_z_f32x8 = _mm256_fmadd_ps(delta_z_f32x8, delta_z_f32x8, sum_squared_z_f32x8);

        // Iteration 1
        __m256 a_x1_f32x8, a_y1_f32x8, a_z1_f32x8, b_x1_f32x8, b_y1_f32x8, b_z1_f32x8;
        _nk_deinterleave_f32x8_haswell(a + (i + 8) * 3, &a_x1_f32x8, &a_y1_f32x8, &a_z1_f32x8);
        _nk_deinterleave_f32x8_haswell(b + (i + 8) * 3, &b_x1_f32x8, &b_y1_f32x8, &b_z1_f32x8);

        sum_a_x_f32x8 = _mm256_add_ps(sum_a_x_f32x8, a_x1_f32x8);
        sum_a_y_f32x8 = _mm256_add_ps(sum_a_y_f32x8, a_y1_f32x8);
        sum_a_z_f32x8 = _mm256_add_ps(sum_a_z_f32x8, a_z1_f32x8);
        sum_b_x_f32x8 = _mm256_add_ps(sum_b_x_f32x8, b_x1_f32x8);
        sum_b_y_f32x8 = _mm256_add_ps(sum_b_y_f32x8, b_y1_f32x8);
        sum_b_z_f32x8 = _mm256_add_ps(sum_b_z_f32x8, b_z1_f32x8);

        __m256 delta_x1_f32x8 = _mm256_sub_ps(a_x1_f32x8, b_x1_f32x8);
        __m256 delta_y1_f32x8 = _mm256_sub_ps(a_y1_f32x8, b_y1_f32x8);
        __m256 delta_z1_f32x8 = _mm256_sub_ps(a_z1_f32x8, b_z1_f32x8);

        sum_squared_x_f32x8 = _mm256_fmadd_ps(delta_x1_f32x8, delta_x1_f32x8, sum_squared_x_f32x8);
        sum_squared_y_f32x8 = _mm256_fmadd_ps(delta_y1_f32x8, delta_y1_f32x8, sum_squared_y_f32x8);
        sum_squared_z_f32x8 = _mm256_fmadd_ps(delta_z1_f32x8, delta_z1_f32x8, sum_squared_z_f32x8);
    }

    // Handle 8-point remainder
    for (; i + 8 <= n; i += 8) {
        _nk_deinterleave_f32x8_haswell(a + i * 3, &a_x_f32x8, &a_y_f32x8, &a_z_f32x8);
        _nk_deinterleave_f32x8_haswell(b + i * 3, &b_x_f32x8, &b_y_f32x8, &b_z_f32x8);

        sum_a_x_f32x8 = _mm256_add_ps(sum_a_x_f32x8, a_x_f32x8);
        sum_a_y_f32x8 = _mm256_add_ps(sum_a_y_f32x8, a_y_f32x8);
        sum_a_z_f32x8 = _mm256_add_ps(sum_a_z_f32x8, a_z_f32x8);
        sum_b_x_f32x8 = _mm256_add_ps(sum_b_x_f32x8, b_x_f32x8);
        sum_b_y_f32x8 = _mm256_add_ps(sum_b_y_f32x8, b_y_f32x8);
        sum_b_z_f32x8 = _mm256_add_ps(sum_b_z_f32x8, b_z_f32x8);

        __m256 delta_x_f32x8 = _mm256_sub_ps(a_x_f32x8, b_x_f32x8);
        __m256 delta_y_f32x8 = _mm256_sub_ps(a_y_f32x8, b_y_f32x8);
        __m256 delta_z_f32x8 = _mm256_sub_ps(a_z_f32x8, b_z_f32x8);

        sum_squared_x_f32x8 = _mm256_fmadd_ps(delta_x_f32x8, delta_x_f32x8, sum_squared_x_f32x8);
        sum_squared_y_f32x8 = _mm256_fmadd_ps(delta_y_f32x8, delta_y_f32x8, sum_squared_y_f32x8);
        sum_squared_z_f32x8 = _mm256_fmadd_ps(delta_z_f32x8, delta_z_f32x8, sum_squared_z_f32x8);
    }

    // Reduce vectors to scalars
    nk_f32_t total_ax = _nk_reduce_add_f32x8_haswell(sum_a_x_f32x8);
    nk_f32_t total_ay = _nk_reduce_add_f32x8_haswell(sum_a_y_f32x8);
    nk_f32_t total_az = _nk_reduce_add_f32x8_haswell(sum_a_z_f32x8);
    nk_f32_t total_bx = _nk_reduce_add_f32x8_haswell(sum_b_x_f32x8);
    nk_f32_t total_by = _nk_reduce_add_f32x8_haswell(sum_b_y_f32x8);
    nk_f32_t total_bz = _nk_reduce_add_f32x8_haswell(sum_b_z_f32x8);
    nk_f32_t total_sq_x = _nk_reduce_add_f32x8_haswell(sum_squared_x_f32x8);
    nk_f32_t total_sq_y = _nk_reduce_add_f32x8_haswell(sum_squared_y_f32x8);
    nk_f32_t total_sq_z = _nk_reduce_add_f32x8_haswell(sum_squared_z_f32x8);

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
        total_sq_x += delta_x * delta_x;
        total_sq_y += delta_y * delta_y;
        total_sq_z += delta_z * delta_z;
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
    nk_f32_t sum_squared = total_sq_x + total_sq_y + total_sq_z;
    nk_f32_t mean_diff_sq = mean_diff_x * mean_diff_x + mean_diff_y * mean_diff_y + mean_diff_z * mean_diff_z;

    *result = NK_F32_SQRT((nk_distance_t)(sum_squared * inv_n - mean_diff_sq));
}

NK_PUBLIC void nk_rmsd_f64_haswell(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                                   nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result) {
    /* RMSD uses identity rotation and scale=1.0 */
    if (rotation) {
        rotation[0] = 1, rotation[1] = 0, rotation[2] = 0;
        rotation[3] = 0, rotation[4] = 1, rotation[5] = 0;
        rotation[6] = 0, rotation[7] = 0, rotation[8] = 1;
    }
    if (scale) *scale = 1.0;
    __m256d const zeros_f64x4 = _mm256_setzero_pd();

    // Accumulators for centroids and squared differences
    __m256d sum_a_x_f64x4 = zeros_f64x4, sum_a_y_f64x4 = zeros_f64x4, sum_a_z_f64x4 = zeros_f64x4;
    __m256d sum_b_x_f64x4 = zeros_f64x4, sum_b_y_f64x4 = zeros_f64x4, sum_b_z_f64x4 = zeros_f64x4;
    __m256d sum_squared_x_f64x4 = zeros_f64x4, sum_squared_y_f64x4 = zeros_f64x4, sum_squared_z_f64x4 = zeros_f64x4;

    __m256d a_x_f64x4, a_y_f64x4, a_z_f64x4, b_x_f64x4, b_y_f64x4, b_z_f64x4;
    nk_size_t i = 0;

    // Main loop with 2x unrolling
    for (; i + 8 <= n; i += 8) {
        // Iteration 0
        _nk_deinterleave_f64x4_haswell(a + i * 3, &a_x_f64x4, &a_y_f64x4, &a_z_f64x4);
        _nk_deinterleave_f64x4_haswell(b + i * 3, &b_x_f64x4, &b_y_f64x4, &b_z_f64x4);

        sum_a_x_f64x4 = _mm256_add_pd(sum_a_x_f64x4, a_x_f64x4);
        sum_a_y_f64x4 = _mm256_add_pd(sum_a_y_f64x4, a_y_f64x4);
        sum_a_z_f64x4 = _mm256_add_pd(sum_a_z_f64x4, a_z_f64x4);
        sum_b_x_f64x4 = _mm256_add_pd(sum_b_x_f64x4, b_x_f64x4);
        sum_b_y_f64x4 = _mm256_add_pd(sum_b_y_f64x4, b_y_f64x4);
        sum_b_z_f64x4 = _mm256_add_pd(sum_b_z_f64x4, b_z_f64x4);

        __m256d delta_x_f64x4 = _mm256_sub_pd(a_x_f64x4, b_x_f64x4);
        __m256d delta_y_f64x4 = _mm256_sub_pd(a_y_f64x4, b_y_f64x4);
        __m256d delta_z_f64x4 = _mm256_sub_pd(a_z_f64x4, b_z_f64x4);

        sum_squared_x_f64x4 = _mm256_fmadd_pd(delta_x_f64x4, delta_x_f64x4, sum_squared_x_f64x4);
        sum_squared_y_f64x4 = _mm256_fmadd_pd(delta_y_f64x4, delta_y_f64x4, sum_squared_y_f64x4);
        sum_squared_z_f64x4 = _mm256_fmadd_pd(delta_z_f64x4, delta_z_f64x4, sum_squared_z_f64x4);

        // Iteration 1
        __m256d a_x1_f64x4, a_y1_f64x4, a_z1_f64x4, b_x1_f64x4, b_y1_f64x4, b_z1_f64x4;
        _nk_deinterleave_f64x4_haswell(a + (i + 4) * 3, &a_x1_f64x4, &a_y1_f64x4, &a_z1_f64x4);
        _nk_deinterleave_f64x4_haswell(b + (i + 4) * 3, &b_x1_f64x4, &b_y1_f64x4, &b_z1_f64x4);

        sum_a_x_f64x4 = _mm256_add_pd(sum_a_x_f64x4, a_x1_f64x4);
        sum_a_y_f64x4 = _mm256_add_pd(sum_a_y_f64x4, a_y1_f64x4);
        sum_a_z_f64x4 = _mm256_add_pd(sum_a_z_f64x4, a_z1_f64x4);
        sum_b_x_f64x4 = _mm256_add_pd(sum_b_x_f64x4, b_x1_f64x4);
        sum_b_y_f64x4 = _mm256_add_pd(sum_b_y_f64x4, b_y1_f64x4);
        sum_b_z_f64x4 = _mm256_add_pd(sum_b_z_f64x4, b_z1_f64x4);

        __m256d delta_x1_f64x4 = _mm256_sub_pd(a_x1_f64x4, b_x1_f64x4);
        __m256d delta_y1_f64x4 = _mm256_sub_pd(a_y1_f64x4, b_y1_f64x4);
        __m256d delta_z1_f64x4 = _mm256_sub_pd(a_z1_f64x4, b_z1_f64x4);

        sum_squared_x_f64x4 = _mm256_fmadd_pd(delta_x1_f64x4, delta_x1_f64x4, sum_squared_x_f64x4);
        sum_squared_y_f64x4 = _mm256_fmadd_pd(delta_y1_f64x4, delta_y1_f64x4, sum_squared_y_f64x4);
        sum_squared_z_f64x4 = _mm256_fmadd_pd(delta_z1_f64x4, delta_z1_f64x4, sum_squared_z_f64x4);
    }

    // Handle 4-point remainder
    for (; i + 4 <= n; i += 4) {
        _nk_deinterleave_f64x4_haswell(a + i * 3, &a_x_f64x4, &a_y_f64x4, &a_z_f64x4);
        _nk_deinterleave_f64x4_haswell(b + i * 3, &b_x_f64x4, &b_y_f64x4, &b_z_f64x4);

        sum_a_x_f64x4 = _mm256_add_pd(sum_a_x_f64x4, a_x_f64x4);
        sum_a_y_f64x4 = _mm256_add_pd(sum_a_y_f64x4, a_y_f64x4);
        sum_a_z_f64x4 = _mm256_add_pd(sum_a_z_f64x4, a_z_f64x4);
        sum_b_x_f64x4 = _mm256_add_pd(sum_b_x_f64x4, b_x_f64x4);
        sum_b_y_f64x4 = _mm256_add_pd(sum_b_y_f64x4, b_y_f64x4);
        sum_b_z_f64x4 = _mm256_add_pd(sum_b_z_f64x4, b_z_f64x4);

        __m256d delta_x_f64x4 = _mm256_sub_pd(a_x_f64x4, b_x_f64x4);
        __m256d delta_y_f64x4 = _mm256_sub_pd(a_y_f64x4, b_y_f64x4);
        __m256d delta_z_f64x4 = _mm256_sub_pd(a_z_f64x4, b_z_f64x4);

        sum_squared_x_f64x4 = _mm256_fmadd_pd(delta_x_f64x4, delta_x_f64x4, sum_squared_x_f64x4);
        sum_squared_y_f64x4 = _mm256_fmadd_pd(delta_y_f64x4, delta_y_f64x4, sum_squared_y_f64x4);
        sum_squared_z_f64x4 = _mm256_fmadd_pd(delta_z_f64x4, delta_z_f64x4, sum_squared_z_f64x4);
    }

    // Reduce vectors to scalars
    nk_f64_t total_ax = _nk_reduce_add_f64x4_haswell(sum_a_x_f64x4);
    nk_f64_t total_ay = _nk_reduce_add_f64x4_haswell(sum_a_y_f64x4);
    nk_f64_t total_az = _nk_reduce_add_f64x4_haswell(sum_a_z_f64x4);
    nk_f64_t total_bx = _nk_reduce_add_f64x4_haswell(sum_b_x_f64x4);
    nk_f64_t total_by = _nk_reduce_add_f64x4_haswell(sum_b_y_f64x4);
    nk_f64_t total_bz = _nk_reduce_add_f64x4_haswell(sum_b_z_f64x4);
    nk_f64_t total_sq_x = _nk_reduce_add_f64x4_haswell(sum_squared_x_f64x4);
    nk_f64_t total_sq_y = _nk_reduce_add_f64x4_haswell(sum_squared_y_f64x4);
    nk_f64_t total_sq_z = _nk_reduce_add_f64x4_haswell(sum_squared_z_f64x4);

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
        total_sq_x += delta_x * delta_x;
        total_sq_y += delta_y * delta_y;
        total_sq_z += delta_z * delta_z;
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
    nk_f64_t sum_squared = total_sq_x + total_sq_y + total_sq_z;
    nk_f64_t mean_diff_sq = mean_diff_x * mean_diff_x + mean_diff_y * mean_diff_y + mean_diff_z * mean_diff_z;

    *result = NK_F64_SQRT(sum_squared * inv_n - mean_diff_sq);
}

NK_PUBLIC void nk_kabsch_f32_haswell(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                     nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result) {
    // Optimized fused single-pass implementation using AVX2.
    // Computes centroids and covariance matrix in one pass.
    __m256d const zeros_f64x4 = _mm256_setzero_pd();

    // Accumulators for centroids (f64 for precision)
    __m256d sum_a_x_f64x4 = zeros_f64x4, sum_a_y_f64x4 = zeros_f64x4, sum_a_z_f64x4 = zeros_f64x4;
    __m256d sum_b_x_f64x4 = zeros_f64x4, sum_b_y_f64x4 = zeros_f64x4, sum_b_z_f64x4 = zeros_f64x4;

    // Accumulators for covariance matrix (sum of outer products)
    __m256d cov_xx_f64x4 = zeros_f64x4, cov_xy_f64x4 = zeros_f64x4, cov_xz_f64x4 = zeros_f64x4;
    __m256d cov_yx_f64x4 = zeros_f64x4, cov_yy_f64x4 = zeros_f64x4, cov_yz_f64x4 = zeros_f64x4;
    __m256d cov_zx_f64x4 = zeros_f64x4, cov_zy_f64x4 = zeros_f64x4, cov_zz_f64x4 = zeros_f64x4;

    nk_size_t i = 0;
    __m256 a_x_f32x8, a_y_f32x8, a_z_f32x8, b_x_f32x8, b_y_f32x8, b_z_f32x8;

    // Fused single-pass: accumulate sums and outer products together
    for (; i + 8 <= n; i += 8) {
        _nk_deinterleave_f32x8_haswell(a + i * 3, &a_x_f32x8, &a_y_f32x8, &a_z_f32x8);
        _nk_deinterleave_f32x8_haswell(b + i * 3, &b_x_f32x8, &b_y_f32x8, &b_z_f32x8);

        // Convert to f64 - low 4 elements
        __m256d a_x_lo_f64x4 = _mm256_cvtps_pd(_mm256_castps256_ps128(a_x_f32x8));
        __m256d a_y_lo_f64x4 = _mm256_cvtps_pd(_mm256_castps256_ps128(a_y_f32x8));
        __m256d a_z_lo_f64x4 = _mm256_cvtps_pd(_mm256_castps256_ps128(a_z_f32x8));
        __m256d b_x_lo_f64x4 = _mm256_cvtps_pd(_mm256_castps256_ps128(b_x_f32x8));
        __m256d b_y_lo_f64x4 = _mm256_cvtps_pd(_mm256_castps256_ps128(b_y_f32x8));
        __m256d b_z_lo_f64x4 = _mm256_cvtps_pd(_mm256_castps256_ps128(b_z_f32x8));

        // Accumulate centroids
        sum_a_x_f64x4 = _mm256_add_pd(sum_a_x_f64x4, a_x_lo_f64x4);
        sum_a_y_f64x4 = _mm256_add_pd(sum_a_y_f64x4, a_y_lo_f64x4);
        sum_a_z_f64x4 = _mm256_add_pd(sum_a_z_f64x4, a_z_lo_f64x4);
        sum_b_x_f64x4 = _mm256_add_pd(sum_b_x_f64x4, b_x_lo_f64x4);
        sum_b_y_f64x4 = _mm256_add_pd(sum_b_y_f64x4, b_y_lo_f64x4);
        sum_b_z_f64x4 = _mm256_add_pd(sum_b_z_f64x4, b_z_lo_f64x4);

        // Accumulate outer products (raw, not centered)
        cov_xx_f64x4 = _mm256_fmadd_pd(a_x_lo_f64x4, b_x_lo_f64x4, cov_xx_f64x4);
        cov_xy_f64x4 = _mm256_fmadd_pd(a_x_lo_f64x4, b_y_lo_f64x4, cov_xy_f64x4);
        cov_xz_f64x4 = _mm256_fmadd_pd(a_x_lo_f64x4, b_z_lo_f64x4, cov_xz_f64x4);
        cov_yx_f64x4 = _mm256_fmadd_pd(a_y_lo_f64x4, b_x_lo_f64x4, cov_yx_f64x4);
        cov_yy_f64x4 = _mm256_fmadd_pd(a_y_lo_f64x4, b_y_lo_f64x4, cov_yy_f64x4);
        cov_yz_f64x4 = _mm256_fmadd_pd(a_y_lo_f64x4, b_z_lo_f64x4, cov_yz_f64x4);
        cov_zx_f64x4 = _mm256_fmadd_pd(a_z_lo_f64x4, b_x_lo_f64x4, cov_zx_f64x4);
        cov_zy_f64x4 = _mm256_fmadd_pd(a_z_lo_f64x4, b_y_lo_f64x4, cov_zy_f64x4);
        cov_zz_f64x4 = _mm256_fmadd_pd(a_z_lo_f64x4, b_z_lo_f64x4, cov_zz_f64x4);

        // High 4 elements
        __m256d a_x_hi_f64x4 = _mm256_cvtps_pd(_mm256_extractf128_ps(a_x_f32x8, 1));
        __m256d a_y_hi_f64x4 = _mm256_cvtps_pd(_mm256_extractf128_ps(a_y_f32x8, 1));
        __m256d a_z_hi_f64x4 = _mm256_cvtps_pd(_mm256_extractf128_ps(a_z_f32x8, 1));
        __m256d b_x_hi_f64x4 = _mm256_cvtps_pd(_mm256_extractf128_ps(b_x_f32x8, 1));
        __m256d b_y_hi_f64x4 = _mm256_cvtps_pd(_mm256_extractf128_ps(b_y_f32x8, 1));
        __m256d b_z_hi_f64x4 = _mm256_cvtps_pd(_mm256_extractf128_ps(b_z_f32x8, 1));

        sum_a_x_f64x4 = _mm256_add_pd(sum_a_x_f64x4, a_x_hi_f64x4);
        sum_a_y_f64x4 = _mm256_add_pd(sum_a_y_f64x4, a_y_hi_f64x4);
        sum_a_z_f64x4 = _mm256_add_pd(sum_a_z_f64x4, a_z_hi_f64x4);
        sum_b_x_f64x4 = _mm256_add_pd(sum_b_x_f64x4, b_x_hi_f64x4);
        sum_b_y_f64x4 = _mm256_add_pd(sum_b_y_f64x4, b_y_hi_f64x4);
        sum_b_z_f64x4 = _mm256_add_pd(sum_b_z_f64x4, b_z_hi_f64x4);

        cov_xx_f64x4 = _mm256_fmadd_pd(a_x_hi_f64x4, b_x_hi_f64x4, cov_xx_f64x4);
        cov_xy_f64x4 = _mm256_fmadd_pd(a_x_hi_f64x4, b_y_hi_f64x4, cov_xy_f64x4);
        cov_xz_f64x4 = _mm256_fmadd_pd(a_x_hi_f64x4, b_z_hi_f64x4, cov_xz_f64x4);
        cov_yx_f64x4 = _mm256_fmadd_pd(a_y_hi_f64x4, b_x_hi_f64x4, cov_yx_f64x4);
        cov_yy_f64x4 = _mm256_fmadd_pd(a_y_hi_f64x4, b_y_hi_f64x4, cov_yy_f64x4);
        cov_yz_f64x4 = _mm256_fmadd_pd(a_y_hi_f64x4, b_z_hi_f64x4, cov_yz_f64x4);
        cov_zx_f64x4 = _mm256_fmadd_pd(a_z_hi_f64x4, b_x_hi_f64x4, cov_zx_f64x4);
        cov_zy_f64x4 = _mm256_fmadd_pd(a_z_hi_f64x4, b_y_hi_f64x4, cov_zy_f64x4);
        cov_zz_f64x4 = _mm256_fmadd_pd(a_z_hi_f64x4, b_z_hi_f64x4, cov_zz_f64x4);
    }

    // Reduce vector accumulators
    nk_f64_t sum_a_x = _nk_reduce_add_f64x4_haswell(sum_a_x_f64x4);
    nk_f64_t sum_a_y = _nk_reduce_add_f64x4_haswell(sum_a_y_f64x4);
    nk_f64_t sum_a_z = _nk_reduce_add_f64x4_haswell(sum_a_z_f64x4);
    nk_f64_t sum_b_x = _nk_reduce_add_f64x4_haswell(sum_b_x_f64x4);
    nk_f64_t sum_b_y = _nk_reduce_add_f64x4_haswell(sum_b_y_f64x4);
    nk_f64_t sum_b_z = _nk_reduce_add_f64x4_haswell(sum_b_z_f64x4);

    nk_f64_t H00 = _nk_reduce_add_f64x4_haswell(cov_xx_f64x4);
    nk_f64_t H01 = _nk_reduce_add_f64x4_haswell(cov_xy_f64x4);
    nk_f64_t H02 = _nk_reduce_add_f64x4_haswell(cov_xz_f64x4);
    nk_f64_t H10 = _nk_reduce_add_f64x4_haswell(cov_yx_f64x4);
    nk_f64_t H11 = _nk_reduce_add_f64x4_haswell(cov_yy_f64x4);
    nk_f64_t H12 = _nk_reduce_add_f64x4_haswell(cov_yz_f64x4);
    nk_f64_t H20 = _nk_reduce_add_f64x4_haswell(cov_zx_f64x4);
    nk_f64_t H21 = _nk_reduce_add_f64x4_haswell(cov_zy_f64x4);
    nk_f64_t H22 = _nk_reduce_add_f64x4_haswell(cov_zz_f64x4);

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
        H00 += ax * bx;
        H01 += ax * by;
        H02 += ax * bz;
        H10 += ay * bx;
        H11 += ay * by;
        H12 += ay * bz;
        H20 += az * bx;
        H21 += az * by;
        H22 += az * bz;
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
        a_centroid[0] = (nk_f32_t)centroid_a_x;
        a_centroid[1] = (nk_f32_t)centroid_a_y;
        a_centroid[2] = (nk_f32_t)centroid_a_z;
    }
    if (b_centroid) {
        b_centroid[0] = (nk_f32_t)centroid_b_x;
        b_centroid[1] = (nk_f32_t)centroid_b_y;
        b_centroid[2] = (nk_f32_t)centroid_b_z;
    }

    // Apply centering correction: H_centered = H - n * centroid_a * centroid_b^T
    H00 -= n * centroid_a_x * centroid_b_x;
    H01 -= n * centroid_a_x * centroid_b_y;
    H02 -= n * centroid_a_x * centroid_b_z;
    H10 -= n * centroid_a_y * centroid_b_x;
    H11 -= n * centroid_a_y * centroid_b_y;
    H12 -= n * centroid_a_y * centroid_b_z;
    H20 -= n * centroid_a_z * centroid_b_x;
    H21 -= n * centroid_a_z * centroid_b_y;
    H22 -= n * centroid_a_z * centroid_b_z;

    // Compute SVD and optimal rotation
    nk_f32_t cross_covariance[9] = {(nk_f32_t)H00, (nk_f32_t)H01, (nk_f32_t)H02, (nk_f32_t)H10, (nk_f32_t)H11,
                                    (nk_f32_t)H12, (nk_f32_t)H20, (nk_f32_t)H21, (nk_f32_t)H22};
    nk_f32_t svd_u[9], svd_s[3], svd_v[9];
    _nk_svd3x3_f32(cross_covariance, svd_u, svd_s, svd_v);

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
    if (_nk_det3x3_f32(r) < 0) {
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
    if (scale) *scale = 1.0;

    // Compute RMSD after optimal rotation
    nk_f64_t sum_squared = 0.0;
    for (nk_size_t j = 0; j < n; ++j) {
        nk_f32_t pa[3], pb[3], ra[3];
        pa[0] = a[j * 3 + 0] - (nk_f32_t)centroid_a_x;
        pa[1] = a[j * 3 + 1] - (nk_f32_t)centroid_a_y;
        pa[2] = a[j * 3 + 2] - (nk_f32_t)centroid_a_z;
        pb[0] = b[j * 3 + 0] - (nk_f32_t)centroid_b_x;
        pb[1] = b[j * 3 + 1] - (nk_f32_t)centroid_b_y;
        pb[2] = b[j * 3 + 2] - (nk_f32_t)centroid_b_z;

        ra[0] = r[0] * pa[0] + r[1] * pa[1] + r[2] * pa[2];
        ra[1] = r[3] * pa[0] + r[4] * pa[1] + r[5] * pa[2];
        ra[2] = r[6] * pa[0] + r[7] * pa[1] + r[8] * pa[2];

        nk_f32_t delta_x = ra[0] - pb[0];
        nk_f32_t delta_y = ra[1] - pb[1];
        nk_f32_t delta_z = ra[2] - pb[2];
        sum_squared += delta_x * delta_x + delta_y * delta_y + delta_z * delta_z;
    }

    *result = NK_F32_SQRT(sum_squared * inv_n);
}

NK_PUBLIC void nk_kabsch_f64_haswell(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                                     nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result) {
    __m256d const zeros_f64x4 = _mm256_setzero_pd();

    // Accumulators for centroids
    __m256d sum_a_x_f64x4 = zeros_f64x4, sum_a_y_f64x4 = zeros_f64x4, sum_a_z_f64x4 = zeros_f64x4;
    __m256d sum_b_x_f64x4 = zeros_f64x4, sum_b_y_f64x4 = zeros_f64x4, sum_b_z_f64x4 = zeros_f64x4;

    // Accumulators for covariance matrix (sum of outer products)
    __m256d cov_xx_f64x4 = zeros_f64x4, cov_xy_f64x4 = zeros_f64x4, cov_xz_f64x4 = zeros_f64x4;
    __m256d cov_yx_f64x4 = zeros_f64x4, cov_yy_f64x4 = zeros_f64x4, cov_yz_f64x4 = zeros_f64x4;
    __m256d cov_zx_f64x4 = zeros_f64x4, cov_zy_f64x4 = zeros_f64x4, cov_zz_f64x4 = zeros_f64x4;

    nk_size_t i = 0;
    __m256d a_x_f64x4, a_y_f64x4, a_z_f64x4, b_x_f64x4, b_y_f64x4, b_z_f64x4;

    // Fused single-pass
    for (; i + 4 <= n; i += 4) {
        _nk_deinterleave_f64x4_haswell(a + i * 3, &a_x_f64x4, &a_y_f64x4, &a_z_f64x4);
        _nk_deinterleave_f64x4_haswell(b + i * 3, &b_x_f64x4, &b_y_f64x4, &b_z_f64x4);

        sum_a_x_f64x4 = _mm256_add_pd(sum_a_x_f64x4, a_x_f64x4);
        sum_a_y_f64x4 = _mm256_add_pd(sum_a_y_f64x4, a_y_f64x4);
        sum_a_z_f64x4 = _mm256_add_pd(sum_a_z_f64x4, a_z_f64x4);
        sum_b_x_f64x4 = _mm256_add_pd(sum_b_x_f64x4, b_x_f64x4);
        sum_b_y_f64x4 = _mm256_add_pd(sum_b_y_f64x4, b_y_f64x4);
        sum_b_z_f64x4 = _mm256_add_pd(sum_b_z_f64x4, b_z_f64x4);

        cov_xx_f64x4 = _mm256_fmadd_pd(a_x_f64x4, b_x_f64x4, cov_xx_f64x4);
        cov_xy_f64x4 = _mm256_fmadd_pd(a_x_f64x4, b_y_f64x4, cov_xy_f64x4);
        cov_xz_f64x4 = _mm256_fmadd_pd(a_x_f64x4, b_z_f64x4, cov_xz_f64x4);
        cov_yx_f64x4 = _mm256_fmadd_pd(a_y_f64x4, b_x_f64x4, cov_yx_f64x4);
        cov_yy_f64x4 = _mm256_fmadd_pd(a_y_f64x4, b_y_f64x4, cov_yy_f64x4);
        cov_yz_f64x4 = _mm256_fmadd_pd(a_y_f64x4, b_z_f64x4, cov_yz_f64x4);
        cov_zx_f64x4 = _mm256_fmadd_pd(a_z_f64x4, b_x_f64x4, cov_zx_f64x4);
        cov_zy_f64x4 = _mm256_fmadd_pd(a_z_f64x4, b_y_f64x4, cov_zy_f64x4);
        cov_zz_f64x4 = _mm256_fmadd_pd(a_z_f64x4, b_z_f64x4, cov_zz_f64x4);
    }

    // Reduce vector accumulators
    nk_f64_t sum_a_x = _nk_reduce_add_f64x4_haswell(sum_a_x_f64x4);
    nk_f64_t sum_a_y = _nk_reduce_add_f64x4_haswell(sum_a_y_f64x4);
    nk_f64_t sum_a_z = _nk_reduce_add_f64x4_haswell(sum_a_z_f64x4);
    nk_f64_t sum_b_x = _nk_reduce_add_f64x4_haswell(sum_b_x_f64x4);
    nk_f64_t sum_b_y = _nk_reduce_add_f64x4_haswell(sum_b_y_f64x4);
    nk_f64_t sum_b_z = _nk_reduce_add_f64x4_haswell(sum_b_z_f64x4);

    nk_f64_t H00 = _nk_reduce_add_f64x4_haswell(cov_xx_f64x4);
    nk_f64_t H01 = _nk_reduce_add_f64x4_haswell(cov_xy_f64x4);
    nk_f64_t H02 = _nk_reduce_add_f64x4_haswell(cov_xz_f64x4);
    nk_f64_t H10 = _nk_reduce_add_f64x4_haswell(cov_yx_f64x4);
    nk_f64_t H11 = _nk_reduce_add_f64x4_haswell(cov_yy_f64x4);
    nk_f64_t H12 = _nk_reduce_add_f64x4_haswell(cov_yz_f64x4);
    nk_f64_t H20 = _nk_reduce_add_f64x4_haswell(cov_zx_f64x4);
    nk_f64_t H21 = _nk_reduce_add_f64x4_haswell(cov_zy_f64x4);
    nk_f64_t H22 = _nk_reduce_add_f64x4_haswell(cov_zz_f64x4);

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
        H00 += ax * bx;
        H01 += ax * by;
        H02 += ax * bz;
        H10 += ay * bx;
        H11 += ay * by;
        H12 += ay * bz;
        H20 += az * bx;
        H21 += az * by;
        H22 += az * bz;
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
    H00 -= n * centroid_a_x * centroid_b_x;
    H01 -= n * centroid_a_x * centroid_b_y;
    H02 -= n * centroid_a_x * centroid_b_z;
    H10 -= n * centroid_a_y * centroid_b_x;
    H11 -= n * centroid_a_y * centroid_b_y;
    H12 -= n * centroid_a_y * centroid_b_z;
    H20 -= n * centroid_a_z * centroid_b_x;
    H21 -= n * centroid_a_z * centroid_b_y;
    H22 -= n * centroid_a_z * centroid_b_z;

    // Compute SVD and optimal rotation (using f32 SVD for performance)
    nk_f32_t cross_covariance[9] = {(nk_f32_t)H00, (nk_f32_t)H01, (nk_f32_t)H02, (nk_f32_t)H10, (nk_f32_t)H11,
                                    (nk_f32_t)H12, (nk_f32_t)H20, (nk_f32_t)H21, (nk_f32_t)H22};
    nk_f32_t svd_u[9], svd_s[3], svd_v[9];
    _nk_svd3x3_f32(cross_covariance, svd_u, svd_s, svd_v);

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
    if (_nk_det3x3_f32(r) < 0) {
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

    *result = NK_F64_SQRT(sum_squared * inv_n);
}

NK_PUBLIC void nk_umeyama_f32_haswell(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                      nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result) {
    // Fused single-pass: centroids, covariance, and variance of A
    __m256d const zeros_f64x4 = _mm256_setzero_pd();

    __m256d sum_a_x_f64x4 = zeros_f64x4, sum_a_y_f64x4 = zeros_f64x4, sum_a_z_f64x4 = zeros_f64x4;
    __m256d sum_b_x_f64x4 = zeros_f64x4, sum_b_y_f64x4 = zeros_f64x4, sum_b_z_f64x4 = zeros_f64x4;
    __m256d cov_xx_f64x4 = zeros_f64x4, cov_xy_f64x4 = zeros_f64x4, cov_xz_f64x4 = zeros_f64x4;
    __m256d cov_yx_f64x4 = zeros_f64x4, cov_yy_f64x4 = zeros_f64x4, cov_yz_f64x4 = zeros_f64x4;
    __m256d cov_zx_f64x4 = zeros_f64x4, cov_zy_f64x4 = zeros_f64x4, cov_zz_f64x4 = zeros_f64x4;
    __m256d variance_a_f64x4 = zeros_f64x4;

    nk_size_t i = 0;
    __m256 a_x_f32x8, a_y_f32x8, a_z_f32x8, b_x_f32x8, b_y_f32x8, b_z_f32x8;

    for (; i + 8 <= n; i += 8) {
        _nk_deinterleave_f32x8_haswell(a + i * 3, &a_x_f32x8, &a_y_f32x8, &a_z_f32x8);
        _nk_deinterleave_f32x8_haswell(b + i * 3, &b_x_f32x8, &b_y_f32x8, &b_z_f32x8);

        __m256d a_x_lo_f64x4 = _mm256_cvtps_pd(_mm256_castps256_ps128(a_x_f32x8));
        __m256d a_y_lo_f64x4 = _mm256_cvtps_pd(_mm256_castps256_ps128(a_y_f32x8));
        __m256d a_z_lo_f64x4 = _mm256_cvtps_pd(_mm256_castps256_ps128(a_z_f32x8));
        __m256d b_x_lo_f64x4 = _mm256_cvtps_pd(_mm256_castps256_ps128(b_x_f32x8));
        __m256d b_y_lo_f64x4 = _mm256_cvtps_pd(_mm256_castps256_ps128(b_y_f32x8));
        __m256d b_z_lo_f64x4 = _mm256_cvtps_pd(_mm256_castps256_ps128(b_z_f32x8));

        sum_a_x_f64x4 = _mm256_add_pd(sum_a_x_f64x4, a_x_lo_f64x4),
        sum_a_y_f64x4 = _mm256_add_pd(sum_a_y_f64x4, a_y_lo_f64x4);
        sum_a_z_f64x4 = _mm256_add_pd(sum_a_z_f64x4, a_z_lo_f64x4);
        sum_b_x_f64x4 = _mm256_add_pd(sum_b_x_f64x4, b_x_lo_f64x4),
        sum_b_y_f64x4 = _mm256_add_pd(sum_b_y_f64x4, b_y_lo_f64x4);
        sum_b_z_f64x4 = _mm256_add_pd(sum_b_z_f64x4, b_z_lo_f64x4);

        cov_xx_f64x4 = _mm256_fmadd_pd(a_x_lo_f64x4, b_x_lo_f64x4, cov_xx_f64x4),
        cov_xy_f64x4 = _mm256_fmadd_pd(a_x_lo_f64x4, b_y_lo_f64x4, cov_xy_f64x4);
        cov_xz_f64x4 = _mm256_fmadd_pd(a_x_lo_f64x4, b_z_lo_f64x4, cov_xz_f64x4);
        cov_yx_f64x4 = _mm256_fmadd_pd(a_y_lo_f64x4, b_x_lo_f64x4, cov_yx_f64x4),
        cov_yy_f64x4 = _mm256_fmadd_pd(a_y_lo_f64x4, b_y_lo_f64x4, cov_yy_f64x4);
        cov_yz_f64x4 = _mm256_fmadd_pd(a_y_lo_f64x4, b_z_lo_f64x4, cov_yz_f64x4);
        cov_zx_f64x4 = _mm256_fmadd_pd(a_z_lo_f64x4, b_x_lo_f64x4, cov_zx_f64x4),
        cov_zy_f64x4 = _mm256_fmadd_pd(a_z_lo_f64x4, b_y_lo_f64x4, cov_zy_f64x4);
        cov_zz_f64x4 = _mm256_fmadd_pd(a_z_lo_f64x4, b_z_lo_f64x4, cov_zz_f64x4);
        variance_a_f64x4 = _mm256_fmadd_pd(a_x_lo_f64x4, a_x_lo_f64x4, variance_a_f64x4);
        variance_a_f64x4 = _mm256_fmadd_pd(a_y_lo_f64x4, a_y_lo_f64x4, variance_a_f64x4);
        variance_a_f64x4 = _mm256_fmadd_pd(a_z_lo_f64x4, a_z_lo_f64x4, variance_a_f64x4);

        __m256d a_x_hi_f64x4 = _mm256_cvtps_pd(_mm256_extractf128_ps(a_x_f32x8, 1));
        __m256d a_y_hi_f64x4 = _mm256_cvtps_pd(_mm256_extractf128_ps(a_y_f32x8, 1));
        __m256d a_z_hi_f64x4 = _mm256_cvtps_pd(_mm256_extractf128_ps(a_z_f32x8, 1));
        __m256d b_x_hi_f64x4 = _mm256_cvtps_pd(_mm256_extractf128_ps(b_x_f32x8, 1));
        __m256d b_y_hi_f64x4 = _mm256_cvtps_pd(_mm256_extractf128_ps(b_y_f32x8, 1));
        __m256d b_z_hi_f64x4 = _mm256_cvtps_pd(_mm256_extractf128_ps(b_z_f32x8, 1));

        sum_a_x_f64x4 = _mm256_add_pd(sum_a_x_f64x4, a_x_hi_f64x4),
        sum_a_y_f64x4 = _mm256_add_pd(sum_a_y_f64x4, a_y_hi_f64x4);
        sum_a_z_f64x4 = _mm256_add_pd(sum_a_z_f64x4, a_z_hi_f64x4);
        sum_b_x_f64x4 = _mm256_add_pd(sum_b_x_f64x4, b_x_hi_f64x4),
        sum_b_y_f64x4 = _mm256_add_pd(sum_b_y_f64x4, b_y_hi_f64x4);
        sum_b_z_f64x4 = _mm256_add_pd(sum_b_z_f64x4, b_z_hi_f64x4);

        cov_xx_f64x4 = _mm256_fmadd_pd(a_x_hi_f64x4, b_x_hi_f64x4, cov_xx_f64x4),
        cov_xy_f64x4 = _mm256_fmadd_pd(a_x_hi_f64x4, b_y_hi_f64x4, cov_xy_f64x4);
        cov_xz_f64x4 = _mm256_fmadd_pd(a_x_hi_f64x4, b_z_hi_f64x4, cov_xz_f64x4);
        cov_yx_f64x4 = _mm256_fmadd_pd(a_y_hi_f64x4, b_x_hi_f64x4, cov_yx_f64x4),
        cov_yy_f64x4 = _mm256_fmadd_pd(a_y_hi_f64x4, b_y_hi_f64x4, cov_yy_f64x4);
        cov_yz_f64x4 = _mm256_fmadd_pd(a_y_hi_f64x4, b_z_hi_f64x4, cov_yz_f64x4);
        cov_zx_f64x4 = _mm256_fmadd_pd(a_z_hi_f64x4, b_x_hi_f64x4, cov_zx_f64x4),
        cov_zy_f64x4 = _mm256_fmadd_pd(a_z_hi_f64x4, b_y_hi_f64x4, cov_zy_f64x4);
        cov_zz_f64x4 = _mm256_fmadd_pd(a_z_hi_f64x4, b_z_hi_f64x4, cov_zz_f64x4);
        variance_a_f64x4 = _mm256_fmadd_pd(a_x_hi_f64x4, a_x_hi_f64x4, variance_a_f64x4);
        variance_a_f64x4 = _mm256_fmadd_pd(a_y_hi_f64x4, a_y_hi_f64x4, variance_a_f64x4);
        variance_a_f64x4 = _mm256_fmadd_pd(a_z_hi_f64x4, a_z_hi_f64x4, variance_a_f64x4);
    }

    // Reduce vector accumulators
    nk_f64_t sum_a_x = _nk_reduce_add_f64x4_haswell(sum_a_x_f64x4);
    nk_f64_t sum_a_y = _nk_reduce_add_f64x4_haswell(sum_a_y_f64x4);
    nk_f64_t sum_a_z = _nk_reduce_add_f64x4_haswell(sum_a_z_f64x4);
    nk_f64_t sum_b_x = _nk_reduce_add_f64x4_haswell(sum_b_x_f64x4);
    nk_f64_t sum_b_y = _nk_reduce_add_f64x4_haswell(sum_b_y_f64x4);
    nk_f64_t sum_b_z = _nk_reduce_add_f64x4_haswell(sum_b_z_f64x4);
    nk_f64_t H00 = _nk_reduce_add_f64x4_haswell(cov_xx_f64x4);
    nk_f64_t H01 = _nk_reduce_add_f64x4_haswell(cov_xy_f64x4);
    nk_f64_t H02 = _nk_reduce_add_f64x4_haswell(cov_xz_f64x4);
    nk_f64_t H10 = _nk_reduce_add_f64x4_haswell(cov_yx_f64x4);
    nk_f64_t H11 = _nk_reduce_add_f64x4_haswell(cov_yy_f64x4);
    nk_f64_t H12 = _nk_reduce_add_f64x4_haswell(cov_yz_f64x4);
    nk_f64_t H20 = _nk_reduce_add_f64x4_haswell(cov_zx_f64x4);
    nk_f64_t H21 = _nk_reduce_add_f64x4_haswell(cov_zy_f64x4);
    nk_f64_t H22 = _nk_reduce_add_f64x4_haswell(cov_zz_f64x4);
    nk_f64_t variance_a_sum = _nk_reduce_add_f64x4_haswell(variance_a_f64x4);

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
        H00 += ax * bx;
        H01 += ax * by;
        H02 += ax * bz;
        H10 += ay * bx;
        H11 += ay * by;
        H12 += ay * bz;
        H20 += az * bx;
        H21 += az * by;
        H22 += az * bz;
        variance_a_sum += ax * ax + ay * ay + az * az;
    }

    // Compute centroids
    nk_f64_t inv_n = 1.0 / (nk_f64_t)n;
    nk_f64_t centroid_a_x = sum_a_x * inv_n, centroid_a_y = sum_a_y * inv_n, centroid_a_z = sum_a_z * inv_n;
    nk_f64_t centroid_b_x = sum_b_x * inv_n, centroid_b_y = sum_b_y * inv_n, centroid_b_z = sum_b_z * inv_n;

    if (a_centroid)
        a_centroid[0] = (nk_f32_t)centroid_a_x, a_centroid[1] = (nk_f32_t)centroid_a_y,
        a_centroid[2] = (nk_f32_t)centroid_a_z;
    if (b_centroid)
        b_centroid[0] = (nk_f32_t)centroid_b_x, b_centroid[1] = (nk_f32_t)centroid_b_y,
        b_centroid[2] = (nk_f32_t)centroid_b_z;

    // Compute centered covariance and variance
    nk_f64_t variance_a = variance_a_sum * inv_n -
                          (centroid_a_x * centroid_a_x + centroid_a_y * centroid_a_y + centroid_a_z * centroid_a_z);

    nk_f32_t cross_covariance[9];
    cross_covariance[0] = (nk_f32_t)(H00 - n * centroid_a_x * centroid_b_x);
    cross_covariance[1] = (nk_f32_t)(H01 - n * centroid_a_x * centroid_b_y);
    cross_covariance[2] = (nk_f32_t)(H02 - n * centroid_a_x * centroid_b_z);
    cross_covariance[3] = (nk_f32_t)(H10 - n * centroid_a_y * centroid_b_x);
    cross_covariance[4] = (nk_f32_t)(H11 - n * centroid_a_y * centroid_b_y);
    cross_covariance[5] = (nk_f32_t)(H12 - n * centroid_a_y * centroid_b_z);
    cross_covariance[6] = (nk_f32_t)(H20 - n * centroid_a_z * centroid_b_x);
    cross_covariance[7] = (nk_f32_t)(H21 - n * centroid_a_z * centroid_b_y);
    cross_covariance[8] = (nk_f32_t)(H22 - n * centroid_a_z * centroid_b_z);

    // SVD
    nk_f32_t svd_u[9], svd_s[9], svd_v[9];
    _nk_svd3x3_f32(cross_covariance, svd_u, svd_s, svd_v);

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

    // Scale factor: c = trace(D*S) / (n * variance_a)
    nk_f32_t det = _nk_det3x3_f32(r);
    nk_f32_t d3 = det < 0 ? -1.0f : 1.0f;
    nk_f32_t trace_ds = svd_s[0] + svd_s[4] + d3 * svd_s[8];
    nk_f64_t c = (nk_f64_t)trace_ds / (n * variance_a);
    if (scale) *scale = c;

    // Handle reflection
    if (det < 0) {
        svd_v[2] = -svd_v[2], svd_v[5] = -svd_v[5], svd_v[8] = -svd_v[8];
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

    /* Output rotation matrix */
    if (rotation) {
        for (int j = 0; j < 9; ++j) rotation[j] = r[j];
    }

    // Compute RMSD with scaling using serial loop (simpler for Haswell)
    nk_f64_t sum_squared = 0.0;
    for (nk_size_t j = 0; j < n; ++j) {
        nk_f64_t pa[3], pb[3], ra[3];
        pa[0] = a[j * 3 + 0] - centroid_a_x;
        pa[1] = a[j * 3 + 1] - centroid_a_y;
        pa[2] = a[j * 3 + 2] - centroid_a_z;
        pb[0] = b[j * 3 + 0] - centroid_b_x;
        pb[1] = b[j * 3 + 1] - centroid_b_y;
        pb[2] = b[j * 3 + 2] - centroid_b_z;

        ra[0] = c * (r[0] * pa[0] + r[1] * pa[1] + r[2] * pa[2]);
        ra[1] = c * (r[3] * pa[0] + r[4] * pa[1] + r[5] * pa[2]);
        ra[2] = c * (r[6] * pa[0] + r[7] * pa[1] + r[8] * pa[2]);

        nk_f64_t delta_x = ra[0] - pb[0];
        nk_f64_t delta_y = ra[1] - pb[1];
        nk_f64_t delta_z = ra[2] - pb[2];
        sum_squared += delta_x * delta_x + delta_y * delta_y + delta_z * delta_z;
    }

    *result = NK_F32_SQRT(sum_squared * inv_n);
}

NK_PUBLIC void nk_umeyama_f64_haswell(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                                      nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result) {
    // Fused single-pass: centroids, covariance, and variance of A
    __m256d const zeros_f64x4 = _mm256_setzero_pd();

    __m256d sum_a_x_f64x4 = zeros_f64x4, sum_a_y_f64x4 = zeros_f64x4, sum_a_z_f64x4 = zeros_f64x4;
    __m256d sum_b_x_f64x4 = zeros_f64x4, sum_b_y_f64x4 = zeros_f64x4, sum_b_z_f64x4 = zeros_f64x4;
    __m256d cov_xx_f64x4 = zeros_f64x4, cov_xy_f64x4 = zeros_f64x4, cov_xz_f64x4 = zeros_f64x4;
    __m256d cov_yx_f64x4 = zeros_f64x4, cov_yy_f64x4 = zeros_f64x4, cov_yz_f64x4 = zeros_f64x4;
    __m256d cov_zx_f64x4 = zeros_f64x4, cov_zy_f64x4 = zeros_f64x4, cov_zz_f64x4 = zeros_f64x4;
    __m256d variance_a_f64x4 = zeros_f64x4;

    nk_size_t i = 0;
    __m256d a_x_f64x4, a_y_f64x4, a_z_f64x4, b_x_f64x4, b_y_f64x4, b_z_f64x4;

    for (; i + 4 <= n; i += 4) {
        _nk_deinterleave_f64x4_haswell(a + i * 3, &a_x_f64x4, &a_y_f64x4, &a_z_f64x4);
        _nk_deinterleave_f64x4_haswell(b + i * 3, &b_x_f64x4, &b_y_f64x4, &b_z_f64x4);

        sum_a_x_f64x4 = _mm256_add_pd(sum_a_x_f64x4, a_x_f64x4),
        sum_a_y_f64x4 = _mm256_add_pd(sum_a_y_f64x4, a_y_f64x4);
        sum_a_z_f64x4 = _mm256_add_pd(sum_a_z_f64x4, a_z_f64x4);
        sum_b_x_f64x4 = _mm256_add_pd(sum_b_x_f64x4, b_x_f64x4),
        sum_b_y_f64x4 = _mm256_add_pd(sum_b_y_f64x4, b_y_f64x4);
        sum_b_z_f64x4 = _mm256_add_pd(sum_b_z_f64x4, b_z_f64x4);

        cov_xx_f64x4 = _mm256_fmadd_pd(a_x_f64x4, b_x_f64x4, cov_xx_f64x4),
        cov_xy_f64x4 = _mm256_fmadd_pd(a_x_f64x4, b_y_f64x4, cov_xy_f64x4);
        cov_xz_f64x4 = _mm256_fmadd_pd(a_x_f64x4, b_z_f64x4, cov_xz_f64x4);
        cov_yx_f64x4 = _mm256_fmadd_pd(a_y_f64x4, b_x_f64x4, cov_yx_f64x4),
        cov_yy_f64x4 = _mm256_fmadd_pd(a_y_f64x4, b_y_f64x4, cov_yy_f64x4);
        cov_yz_f64x4 = _mm256_fmadd_pd(a_y_f64x4, b_z_f64x4, cov_yz_f64x4);
        cov_zx_f64x4 = _mm256_fmadd_pd(a_z_f64x4, b_x_f64x4, cov_zx_f64x4),
        cov_zy_f64x4 = _mm256_fmadd_pd(a_z_f64x4, b_y_f64x4, cov_zy_f64x4);
        cov_zz_f64x4 = _mm256_fmadd_pd(a_z_f64x4, b_z_f64x4, cov_zz_f64x4);
        variance_a_f64x4 = _mm256_fmadd_pd(a_x_f64x4, a_x_f64x4, variance_a_f64x4);
        variance_a_f64x4 = _mm256_fmadd_pd(a_y_f64x4, a_y_f64x4, variance_a_f64x4);
        variance_a_f64x4 = _mm256_fmadd_pd(a_z_f64x4, a_z_f64x4, variance_a_f64x4);
    }

    // Reduce vector accumulators
    nk_f64_t sum_a_x = _nk_reduce_add_f64x4_haswell(sum_a_x_f64x4);
    nk_f64_t sum_a_y = _nk_reduce_add_f64x4_haswell(sum_a_y_f64x4);
    nk_f64_t sum_a_z = _nk_reduce_add_f64x4_haswell(sum_a_z_f64x4);
    nk_f64_t sum_b_x = _nk_reduce_add_f64x4_haswell(sum_b_x_f64x4);
    nk_f64_t sum_b_y = _nk_reduce_add_f64x4_haswell(sum_b_y_f64x4);
    nk_f64_t sum_b_z = _nk_reduce_add_f64x4_haswell(sum_b_z_f64x4);
    nk_f64_t h00_s = _nk_reduce_add_f64x4_haswell(cov_xx_f64x4);
    nk_f64_t h01_s = _nk_reduce_add_f64x4_haswell(cov_xy_f64x4);
    nk_f64_t h02_s = _nk_reduce_add_f64x4_haswell(cov_xz_f64x4);
    nk_f64_t h10_s = _nk_reduce_add_f64x4_haswell(cov_yx_f64x4);
    nk_f64_t h11_s = _nk_reduce_add_f64x4_haswell(cov_yy_f64x4);
    nk_f64_t h12_s = _nk_reduce_add_f64x4_haswell(cov_yz_f64x4);
    nk_f64_t h20_s = _nk_reduce_add_f64x4_haswell(cov_zx_f64x4);
    nk_f64_t h21_s = _nk_reduce_add_f64x4_haswell(cov_zy_f64x4);
    nk_f64_t h22_s = _nk_reduce_add_f64x4_haswell(cov_zz_f64x4);
    nk_f64_t variance_a_sum = _nk_reduce_add_f64x4_haswell(variance_a_f64x4);

    // Scalar tail loop for remaining points
    for (; i < n; i++) {
        nk_f64_t ax = a[i * 3 + 0], ay = a[i * 3 + 1], az = a[i * 3 + 2];
        nk_f64_t bx = b[i * 3 + 0], by = b[i * 3 + 1], bz = b[i * 3 + 2];
        sum_a_x += ax;
        sum_a_y += ay;
        sum_a_z += az;
        sum_b_x += bx;
        sum_b_y += by;
        sum_b_z += bz;
        h00_s += ax * bx;
        h01_s += ax * by;
        h02_s += ax * bz;
        h10_s += ay * bx;
        h11_s += ay * by;
        h12_s += ay * bz;
        h20_s += az * bx;
        h21_s += az * by;
        h22_s += az * bz;
        variance_a_sum += ax * ax + ay * ay + az * az;
    }

    // Compute centroids
    nk_f64_t inv_n = 1.0 / (nk_f64_t)n;

    nk_f64_t centroid_a_x = sum_a_x * inv_n, centroid_a_y = sum_a_y * inv_n, centroid_a_z = sum_a_z * inv_n;
    nk_f64_t centroid_b_x = sum_b_x * inv_n, centroid_b_y = sum_b_y * inv_n, centroid_b_z = sum_b_z * inv_n;

    if (a_centroid) a_centroid[0] = centroid_a_x, a_centroid[1] = centroid_a_y, a_centroid[2] = centroid_a_z;
    if (b_centroid) b_centroid[0] = centroid_b_x, b_centroid[1] = centroid_b_y, b_centroid[2] = centroid_b_z;

    // Compute centered covariance and variance
    nk_f64_t variance_a = variance_a_sum * inv_n -
                          (centroid_a_x * centroid_a_x + centroid_a_y * centroid_a_y + centroid_a_z * centroid_a_z);

    nk_f32_t cross_covariance[9];
    cross_covariance[0] = (nk_f32_t)(h00_s - sum_a_x * sum_b_x * inv_n);
    cross_covariance[1] = (nk_f32_t)(h01_s - sum_a_x * sum_b_y * inv_n);
    cross_covariance[2] = (nk_f32_t)(h02_s - sum_a_x * sum_b_z * inv_n);
    cross_covariance[3] = (nk_f32_t)(h10_s - sum_a_y * sum_b_x * inv_n);
    cross_covariance[4] = (nk_f32_t)(h11_s - sum_a_y * sum_b_y * inv_n);
    cross_covariance[5] = (nk_f32_t)(h12_s - sum_a_y * sum_b_z * inv_n);
    cross_covariance[6] = (nk_f32_t)(h20_s - sum_a_z * sum_b_x * inv_n);
    cross_covariance[7] = (nk_f32_t)(h21_s - sum_a_z * sum_b_y * inv_n);
    cross_covariance[8] = (nk_f32_t)(h22_s - sum_a_z * sum_b_z * inv_n);

    // SVD
    nk_f32_t svd_u[9], svd_s[9], svd_v[9];
    _nk_svd3x3_f32(cross_covariance, svd_u, svd_s, svd_v);

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

    // Scale factor: c = trace(D*S) / (n * variance_a)
    nk_f32_t det = _nk_det3x3_f32(r);
    nk_f32_t d3 = det < 0 ? -1.0f : 1.0f;
    nk_f32_t trace_ds = svd_s[0] + svd_s[4] + d3 * svd_s[8];
    nk_f64_t c = (nk_f64_t)trace_ds / (n * variance_a);
    if (scale) *scale = c;

    // Handle reflection
    if (det < 0) {
        svd_v[2] = -svd_v[2], svd_v[5] = -svd_v[5], svd_v[8] = -svd_v[8];
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

    /* Output rotation matrix */
    if (rotation) {
        for (int j = 0; j < 9; ++j) rotation[j] = (nk_f64_t)r[j];
    }

    // Compute RMSD with scaling using serial loop
    nk_f64_t sum_squared = 0.0;
    for (nk_size_t j = 0; j < n; ++j) {
        nk_f64_t pa[3], pb[3], ra[3];
        pa[0] = a[j * 3 + 0] - centroid_a_x;
        pa[1] = a[j * 3 + 1] - centroid_a_y;
        pa[2] = a[j * 3 + 2] - centroid_a_z;
        pb[0] = b[j * 3 + 0] - centroid_b_x;
        pb[1] = b[j * 3 + 1] - centroid_b_y;
        pb[2] = b[j * 3 + 2] - centroid_b_z;

        ra[0] = c * (r[0] * pa[0] + r[1] * pa[1] + r[2] * pa[2]);
        ra[1] = c * (r[3] * pa[0] + r[4] * pa[1] + r[5] * pa[2]);
        ra[2] = c * (r[6] * pa[0] + r[7] * pa[1] + r[8] * pa[2]);

        nk_f64_t delta_x = ra[0] - pb[0];
        nk_f64_t delta_y = ra[1] - pb[1];
        nk_f64_t delta_z = ra[2] - pb[2];
        sum_squared += delta_x * delta_x + delta_y * delta_y + delta_z * delta_z;
    }

    *result = NK_F64_SQRT(sum_squared * inv_n);
}

#if defined(__cplusplus)
} // extern "C"
#endif

#pragma clang attribute pop
#pragma GCC pop_options
#endif // NK_TARGET_HASWELL
#endif // _NK_TARGET_X86

#endif // NK_MESH_HASWELL_H
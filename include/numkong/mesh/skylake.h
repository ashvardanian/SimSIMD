/**
 *  @brief SIMD-accelerated Dot Products for Real and Complex Numbers optimized for Intel Skylake-X CPUs.
 *  @file include/numkong/mesh/skylake.h
 *  @sa include/numkong/mesh.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 */
#ifndef NK_MESH_SKYLAKE_H
#define NK_MESH_SKYLAKE_H

#if NK_TARGET_X86_
#if NK_TARGET_SKYLAKE
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "avx512bw", "bmi2")
#pragma clang attribute push(__attribute__((target("avx2,avx512f,avx512vl,avx512bw,bmi2"))), apply_to = function)

#include "numkong/types.h"
#include "numkong/spatial/haswell.h"

#if defined(__cplusplus)
extern "C" {
#endif

/*  Internal helper: Deinterleave 48 floats (16 xyz triplets) into separate x, y, z vectors.
 *  Uses permutex2var shuffles instead of gather for ~1.8x speedup.
 *
 *  Input: 48 contiguous floats [x0,y0,z0, x1,y1,z1, ..., x15,y15,z15]
 *  Output: x[16], y[16], z[16] vectors
 *
 *  Implementation: Load 3 registers (r0,r1,r2), use 6 permutex2var ops to separate.
 *  Phase analysis: r0 starts at float 0 (phase 0), r1 at float 16 (phase 1), r2 at float 32 (phase 2)
 *
 *  X elements at memory positions: 0,3,6,9,12,15,18,21,24,27,30,33,36,39,42,45
 *    = r0[0,3,6,9,12,15], r1[2,5,8,11,14], r2[1,4,7,10,13]
 *  Y elements at memory positions: 1,4,7,10,13,16,19,22,25,28,31,34,37,40,43,46
 *    = r0[1,4,7,10,13], r1[0,3,6,9,12,15], r2[2,5,8,11,14]
 *  Z elements at memory positions: 2,5,8,11,14,17,20,23,26,29,32,35,38,41,44,47
 *    = r0[2,5,8,11,14], r1[1,4,7,10,13], r2[0,3,6,9,12,15]
 */
NK_INTERNAL void nk_deinterleave_f32x16_skylake_(                                            //
    nk_f32_t const *ptr, __m512 *x_f32x16_out, __m512 *y_f32x16_out, __m512 *z_f32x16_out) { //
    __m512 reg0_f32x16 = _mm512_loadu_ps(ptr);
    __m512 reg1_f32x16 = _mm512_loadu_ps(ptr + 16);
    __m512 reg2_f32x16 = _mm512_loadu_ps(ptr + 32);

    // X: reg0[0,3,6,9,12,15] + reg1[2,5,8,11,14] -> 11 elements, then + reg2[1,4,7,10,13] -> 16 elements
    // Indices for permutex2var: 0-15 = from first operand, 16-31 = from second operand
    __m512i idx_x_01_i32x16 = _mm512_setr_epi32(0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 0, 0, 0, 0, 0);
    __m512i idx_x_2_i32x16 = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 17, 20, 23, 26, 29);
    __m512 x01_f32x16 = _mm512_permutex2var_ps(reg0_f32x16, idx_x_01_i32x16, reg1_f32x16);
    *x_f32x16_out = _mm512_permutex2var_ps(x01_f32x16, idx_x_2_i32x16, reg2_f32x16);

    // Y: reg0[1,4,7,10,13] + reg1[0,3,6,9,12,15] -> 11 elements, then + reg2[2,5,8,11,14] -> 16 elements
    __m512i idx_y_01_i32x16 = _mm512_setr_epi32(1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 0, 0, 0, 0, 0);
    __m512i idx_y_2_i32x16 = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 18, 21, 24, 27, 30);
    __m512 y01_f32x16 = _mm512_permutex2var_ps(reg0_f32x16, idx_y_01_i32x16, reg1_f32x16);
    *y_f32x16_out = _mm512_permutex2var_ps(y01_f32x16, idx_y_2_i32x16, reg2_f32x16);

    // Z: reg0[2,5,8,11,14] + reg1[1,4,7,10,13] -> 10 elements, then + reg2[0,3,6,9,12,15] -> 16 elements
    __m512i idx_z_01_i32x16 = _mm512_setr_epi32(2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 0, 0, 0, 0, 0, 0);
    __m512i idx_z_2_i32x16 = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 19, 22, 25, 28, 31);
    __m512 z01_f32x16 = _mm512_permutex2var_ps(reg0_f32x16, idx_z_01_i32x16, reg1_f32x16);
    *z_f32x16_out = _mm512_permutex2var_ps(z01_f32x16, idx_z_2_i32x16, reg2_f32x16);
}

/*  Internal helper: Deinterleave 8 f64 3D points from xyz,xyz,xyz... to separate x,y,z vectors.
 *  Input: 24 consecutive f64 values (8 points * 3 coordinates)
 *  Output: Three __m512d vectors containing the x, y, z coordinates separately.
 */
NK_INTERNAL void nk_deinterleave_f64x8_skylake_(                                             //
    nk_f64_t const *ptr, __m512d *x_f64x8_out, __m512d *y_f64x8_out, __m512d *z_f64x8_out) { //
    __m512d reg0_f64x8 = _mm512_loadu_pd(ptr);                                               // elements 0-7
    __m512d reg1_f64x8 = _mm512_loadu_pd(ptr + 8);                                           // elements 8-15
    __m512d reg2_f64x8 = _mm512_loadu_pd(ptr + 16);                                          // elements 16-23

    // X: positions 0,3,6,9,12,15,18,21 -> reg0[0,3,6] + reg1[1,4,7] + reg2[2,5]
    __m512i idx_x_01_i64x8 = _mm512_setr_epi64(0, 3, 6, 9, 12, 15, 0, 0);
    __m512i idx_x_2_i64x8 = _mm512_setr_epi64(0, 1, 2, 3, 4, 5, 10, 13);
    __m512d x01_f64x8 = _mm512_permutex2var_pd(reg0_f64x8, idx_x_01_i64x8, reg1_f64x8);
    *x_f64x8_out = _mm512_permutex2var_pd(x01_f64x8, idx_x_2_i64x8, reg2_f64x8);

    // Y: positions 1,4,7,10,13,16,19,22 -> reg0[1,4,7] + reg1[2,5] + reg2[0,3,6]
    __m512i idx_y_01_i64x8 = _mm512_setr_epi64(1, 4, 7, 10, 13, 0, 0, 0);
    __m512i idx_y_2_i64x8 = _mm512_setr_epi64(0, 1, 2, 3, 4, 8, 11, 14);
    __m512d y01_f64x8 = _mm512_permutex2var_pd(reg0_f64x8, idx_y_01_i64x8, reg1_f64x8);
    *y_f64x8_out = _mm512_permutex2var_pd(y01_f64x8, idx_y_2_i64x8, reg2_f64x8);

    // Z: positions 2,5,8,11,14,17,20,23 -> reg0[2,5] + reg1[0,3,6] + reg2[1,4,7]
    __m512i idx_z_01_i64x8 = _mm512_setr_epi64(2, 5, 8, 11, 14, 0, 0, 0);
    __m512i idx_z_2_i64x8 = _mm512_setr_epi64(0, 1, 2, 3, 4, 9, 12, 15);
    __m512d z01_f64x8 = _mm512_permutex2var_pd(reg0_f64x8, idx_z_01_i64x8, reg1_f64x8);
    *z_f64x8_out = _mm512_permutex2var_pd(z01_f64x8, idx_z_2_i64x8, reg2_f64x8);
}

/*  Internal helper: Compute sum of squared distances after applying rotation (and optional scale).
 *  Used by kabsch (scale=1.0) and umeyama (scale=computed_scale).
 *  Returns sum_squared, caller computes sqrt(sum_squared / n).
 */
NK_INTERNAL nk_f32_t nk_transformed_ssd_f32_skylake_(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n,
                                                     nk_f32_t const *r, nk_f32_t scale, nk_f32_t centroid_a_x,
                                                     nk_f32_t centroid_a_y, nk_f32_t centroid_a_z,
                                                     nk_f32_t centroid_b_x, nk_f32_t centroid_b_y,
                                                     nk_f32_t centroid_b_z) {
    // Broadcast scaled rotation matrix elements
    __m512 sr0_f32x16 = _mm512_set1_ps(scale * r[0]), sr1_f32x16 = _mm512_set1_ps(scale * r[1]),
           sr2_f32x16 = _mm512_set1_ps(scale * r[2]);
    __m512 sr3_f32x16 = _mm512_set1_ps(scale * r[3]), sr4_f32x16 = _mm512_set1_ps(scale * r[4]),
           sr5_f32x16 = _mm512_set1_ps(scale * r[5]);
    __m512 sr6_f32x16 = _mm512_set1_ps(scale * r[6]), sr7_f32x16 = _mm512_set1_ps(scale * r[7]),
           sr8_f32x16 = _mm512_set1_ps(scale * r[8]);

    // Broadcast centroids
    __m512 centroid_a_x_f32x16 = _mm512_set1_ps(centroid_a_x);
    __m512 centroid_a_y_f32x16 = _mm512_set1_ps(centroid_a_y);
    __m512 centroid_a_z_f32x16 = _mm512_set1_ps(centroid_a_z);
    __m512 centroid_b_x_f32x16 = _mm512_set1_ps(centroid_b_x);
    __m512 centroid_b_y_f32x16 = _mm512_set1_ps(centroid_b_y);
    __m512 centroid_b_z_f32x16 = _mm512_set1_ps(centroid_b_z);

    __m512 sum_squared_f32x16 = _mm512_setzero_ps();
    __m512 a_x_f32x16, a_y_f32x16, a_z_f32x16, b_x_f32x16, b_y_f32x16, b_z_f32x16;
    nk_size_t j = 0;

    for (; j + 16 <= n; j += 16) {
        nk_deinterleave_f32x16_skylake_(a + j * 3, &a_x_f32x16, &a_y_f32x16, &a_z_f32x16);
        nk_deinterleave_f32x16_skylake_(b + j * 3, &b_x_f32x16, &b_y_f32x16, &b_z_f32x16);

        // Center points
        __m512 pa_x_f32x16 = _mm512_sub_ps(a_x_f32x16, centroid_a_x_f32x16);
        __m512 pa_y_f32x16 = _mm512_sub_ps(a_y_f32x16, centroid_a_y_f32x16);
        __m512 pa_z_f32x16 = _mm512_sub_ps(a_z_f32x16, centroid_a_z_f32x16);
        __m512 pb_x_f32x16 = _mm512_sub_ps(b_x_f32x16, centroid_b_x_f32x16);
        __m512 pb_y_f32x16 = _mm512_sub_ps(b_y_f32x16, centroid_b_y_f32x16);
        __m512 pb_z_f32x16 = _mm512_sub_ps(b_z_f32x16, centroid_b_z_f32x16);

        // Rotate and scale: ra = scale * R * pa
        __m512 ra_x_f32x16 = _mm512_fmadd_ps(
            sr2_f32x16, pa_z_f32x16, _mm512_fmadd_ps(sr1_f32x16, pa_y_f32x16, _mm512_mul_ps(sr0_f32x16, pa_x_f32x16)));
        __m512 ra_y_f32x16 = _mm512_fmadd_ps(
            sr5_f32x16, pa_z_f32x16, _mm512_fmadd_ps(sr4_f32x16, pa_y_f32x16, _mm512_mul_ps(sr3_f32x16, pa_x_f32x16)));
        __m512 ra_z_f32x16 = _mm512_fmadd_ps(
            sr8_f32x16, pa_z_f32x16, _mm512_fmadd_ps(sr7_f32x16, pa_y_f32x16, _mm512_mul_ps(sr6_f32x16, pa_x_f32x16)));

        // Delta and accumulate
        __m512 delta_x_f32x16 = _mm512_sub_ps(ra_x_f32x16, pb_x_f32x16);
        __m512 delta_y_f32x16 = _mm512_sub_ps(ra_y_f32x16, pb_y_f32x16);
        __m512 delta_z_f32x16 = _mm512_sub_ps(ra_z_f32x16, pb_z_f32x16);

        sum_squared_f32x16 = _mm512_fmadd_ps(delta_x_f32x16, delta_x_f32x16, sum_squared_f32x16);
        sum_squared_f32x16 = _mm512_fmadd_ps(delta_y_f32x16, delta_y_f32x16, sum_squared_f32x16);
        sum_squared_f32x16 = _mm512_fmadd_ps(delta_z_f32x16, delta_z_f32x16, sum_squared_f32x16);
    }

    nk_f32_t sum_squared = _mm512_reduce_add_ps(sum_squared_f32x16);

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
 *  Rotation matrix, scale and data are all f64 for full precision.
 */
NK_INTERNAL nk_f64_t nk_transformed_ssd_f64_skylake_(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n,
                                                     nk_f64_t const *r, nk_f64_t scale, nk_f64_t centroid_a_x,
                                                     nk_f64_t centroid_a_y, nk_f64_t centroid_a_z,
                                                     nk_f64_t centroid_b_x, nk_f64_t centroid_b_y,
                                                     nk_f64_t centroid_b_z) {
    // Broadcast scaled rotation matrix elements
    __m512d sr0_f64x8 = _mm512_set1_pd(scale * r[0]), sr1_f64x8 = _mm512_set1_pd(scale * r[1]),
            sr2_f64x8 = _mm512_set1_pd(scale * r[2]);
    __m512d sr3_f64x8 = _mm512_set1_pd(scale * r[3]), sr4_f64x8 = _mm512_set1_pd(scale * r[4]),
            sr5_f64x8 = _mm512_set1_pd(scale * r[5]);
    __m512d sr6_f64x8 = _mm512_set1_pd(scale * r[6]), sr7_f64x8 = _mm512_set1_pd(scale * r[7]),
            sr8_f64x8 = _mm512_set1_pd(scale * r[8]);

    // Broadcast centroids
    __m512d centroid_a_x_f64x8 = _mm512_set1_pd(centroid_a_x);
    __m512d centroid_a_y_f64x8 = _mm512_set1_pd(centroid_a_y);
    __m512d centroid_a_z_f64x8 = _mm512_set1_pd(centroid_a_z);
    __m512d centroid_b_x_f64x8 = _mm512_set1_pd(centroid_b_x);
    __m512d centroid_b_y_f64x8 = _mm512_set1_pd(centroid_b_y);
    __m512d centroid_b_z_f64x8 = _mm512_set1_pd(centroid_b_z);

    __m512d sum_squared_f64x8 = _mm512_setzero_pd();
    __m512d a_x_f64x8, a_y_f64x8, a_z_f64x8, b_x_f64x8, b_y_f64x8, b_z_f64x8;
    nk_size_t j = 0;

    for (; j + 8 <= n; j += 8) {
        nk_deinterleave_f64x8_skylake_(a + j * 3, &a_x_f64x8, &a_y_f64x8, &a_z_f64x8);
        nk_deinterleave_f64x8_skylake_(b + j * 3, &b_x_f64x8, &b_y_f64x8, &b_z_f64x8);

        // Center points
        __m512d pa_x_f64x8 = _mm512_sub_pd(a_x_f64x8, centroid_a_x_f64x8);
        __m512d pa_y_f64x8 = _mm512_sub_pd(a_y_f64x8, centroid_a_y_f64x8);
        __m512d pa_z_f64x8 = _mm512_sub_pd(a_z_f64x8, centroid_a_z_f64x8);
        __m512d pb_x_f64x8 = _mm512_sub_pd(b_x_f64x8, centroid_b_x_f64x8);
        __m512d pb_y_f64x8 = _mm512_sub_pd(b_y_f64x8, centroid_b_y_f64x8);
        __m512d pb_z_f64x8 = _mm512_sub_pd(b_z_f64x8, centroid_b_z_f64x8);

        // Rotate and scale: ra = scale * R * pa
        __m512d ra_x_f64x8 = _mm512_fmadd_pd(
            sr2_f64x8, pa_z_f64x8, _mm512_fmadd_pd(sr1_f64x8, pa_y_f64x8, _mm512_mul_pd(sr0_f64x8, pa_x_f64x8)));
        __m512d ra_y_f64x8 = _mm512_fmadd_pd(
            sr5_f64x8, pa_z_f64x8, _mm512_fmadd_pd(sr4_f64x8, pa_y_f64x8, _mm512_mul_pd(sr3_f64x8, pa_x_f64x8)));
        __m512d ra_z_f64x8 = _mm512_fmadd_pd(
            sr8_f64x8, pa_z_f64x8, _mm512_fmadd_pd(sr7_f64x8, pa_y_f64x8, _mm512_mul_pd(sr6_f64x8, pa_x_f64x8)));

        // Delta and accumulate
        __m512d delta_x_f64x8 = _mm512_sub_pd(ra_x_f64x8, pb_x_f64x8);
        __m512d delta_y_f64x8 = _mm512_sub_pd(ra_y_f64x8, pb_y_f64x8);
        __m512d delta_z_f64x8 = _mm512_sub_pd(ra_z_f64x8, pb_z_f64x8);

        sum_squared_f64x8 = _mm512_fmadd_pd(delta_x_f64x8, delta_x_f64x8, sum_squared_f64x8);
        sum_squared_f64x8 = _mm512_fmadd_pd(delta_y_f64x8, delta_y_f64x8, sum_squared_f64x8);
        sum_squared_f64x8 = _mm512_fmadd_pd(delta_z_f64x8, delta_z_f64x8, sum_squared_f64x8);
    }

    nk_f64_t sum_squared = _mm512_reduce_add_pd(sum_squared_f64x8);

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

NK_PUBLIC void nk_rmsd_f32_skylake(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                   nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result) {
    /* RMSD uses identity rotation and scale=1.0 */
    if (rotation) {
        rotation[0] = 1;
        rotation[1] = 0;
        rotation[2] = 0;
        rotation[3] = 0;
        rotation[4] = 1;
        rotation[5] = 0;
        rotation[6] = 0;
        rotation[7] = 0;
        rotation[8] = 1;
    }
    if (scale) *scale = 1.0;
    // Optimized fused single-pass implementation.
    // Computes centroids and squared differences in one pass using the identity:
    //   RMSD = sqrt(E[(a-mean_a) - (b-mean_b)]^2)
    //        = sqrt(E[(a-b)^2] - (mean_a - mean_b)^2)
    __m512i const gather_idx_i32x16 = _mm512_setr_epi32(0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45);
    __m512 const zeros_f32x16 = _mm512_setzero_ps();

    // Accumulators for centroids and squared differences
    __m512 sum_a_x_f32x16 = zeros_f32x16, sum_a_y_f32x16 = zeros_f32x16, sum_a_z_f32x16 = zeros_f32x16;
    __m512 sum_b_x_f32x16 = zeros_f32x16, sum_b_y_f32x16 = zeros_f32x16, sum_b_z_f32x16 = zeros_f32x16;
    __m512 sum_squared_x_f32x16 = zeros_f32x16, sum_squared_y_f32x16 = zeros_f32x16,
           sum_squared_z_f32x16 = zeros_f32x16;

    __m512 a_x_f32x16, a_y_f32x16, a_z_f32x16, b_x_f32x16, b_y_f32x16, b_z_f32x16;
    nk_size_t i = 0;

    // Main loop with 2x unrolling for better latency hiding
    for (; i + 32 <= n; i += 32) {
        // Iteration 0
        nk_deinterleave_f32x16_skylake_(a + i * 3, &a_x_f32x16, &a_y_f32x16, &a_z_f32x16);
        nk_deinterleave_f32x16_skylake_(b + i * 3, &b_x_f32x16, &b_y_f32x16, &b_z_f32x16);

        sum_a_x_f32x16 = _mm512_add_ps(sum_a_x_f32x16, a_x_f32x16);
        sum_a_y_f32x16 = _mm512_add_ps(sum_a_y_f32x16, a_y_f32x16);
        sum_a_z_f32x16 = _mm512_add_ps(sum_a_z_f32x16, a_z_f32x16);
        sum_b_x_f32x16 = _mm512_add_ps(sum_b_x_f32x16, b_x_f32x16);
        sum_b_y_f32x16 = _mm512_add_ps(sum_b_y_f32x16, b_y_f32x16);
        sum_b_z_f32x16 = _mm512_add_ps(sum_b_z_f32x16, b_z_f32x16);

        __m512 delta_x_f32x16 = _mm512_sub_ps(a_x_f32x16, b_x_f32x16);
        __m512 delta_y_f32x16 = _mm512_sub_ps(a_y_f32x16, b_y_f32x16);
        __m512 delta_z_f32x16 = _mm512_sub_ps(a_z_f32x16, b_z_f32x16);

        sum_squared_x_f32x16 = _mm512_fmadd_ps(delta_x_f32x16, delta_x_f32x16, sum_squared_x_f32x16);
        sum_squared_y_f32x16 = _mm512_fmadd_ps(delta_y_f32x16, delta_y_f32x16, sum_squared_y_f32x16);
        sum_squared_z_f32x16 = _mm512_fmadd_ps(delta_z_f32x16, delta_z_f32x16, sum_squared_z_f32x16);

        // Iteration 1
        __m512 a_x1_f32x16, a_y1_f32x16, a_z1_f32x16, b_x1_f32x16, b_y1_f32x16, b_z1_f32x16;
        nk_deinterleave_f32x16_skylake_(a + (i + 16) * 3, &a_x1_f32x16, &a_y1_f32x16, &a_z1_f32x16);
        nk_deinterleave_f32x16_skylake_(b + (i + 16) * 3, &b_x1_f32x16, &b_y1_f32x16, &b_z1_f32x16);

        sum_a_x_f32x16 = _mm512_add_ps(sum_a_x_f32x16, a_x1_f32x16);
        sum_a_y_f32x16 = _mm512_add_ps(sum_a_y_f32x16, a_y1_f32x16);
        sum_a_z_f32x16 = _mm512_add_ps(sum_a_z_f32x16, a_z1_f32x16);
        sum_b_x_f32x16 = _mm512_add_ps(sum_b_x_f32x16, b_x1_f32x16);
        sum_b_y_f32x16 = _mm512_add_ps(sum_b_y_f32x16, b_y1_f32x16);
        sum_b_z_f32x16 = _mm512_add_ps(sum_b_z_f32x16, b_z1_f32x16);

        __m512 delta_x1_f32x16 = _mm512_sub_ps(a_x1_f32x16, b_x1_f32x16);
        __m512 delta_y1_f32x16 = _mm512_sub_ps(a_y1_f32x16, b_y1_f32x16);
        __m512 delta_z1_f32x16 = _mm512_sub_ps(a_z1_f32x16, b_z1_f32x16);

        sum_squared_x_f32x16 = _mm512_fmadd_ps(delta_x1_f32x16, delta_x1_f32x16, sum_squared_x_f32x16);
        sum_squared_y_f32x16 = _mm512_fmadd_ps(delta_y1_f32x16, delta_y1_f32x16, sum_squared_y_f32x16);
        sum_squared_z_f32x16 = _mm512_fmadd_ps(delta_z1_f32x16, delta_z1_f32x16, sum_squared_z_f32x16);
    }

    // Handle 16-point remainder
    for (; i + 16 <= n; i += 16) {
        nk_deinterleave_f32x16_skylake_(a + i * 3, &a_x_f32x16, &a_y_f32x16, &a_z_f32x16);
        nk_deinterleave_f32x16_skylake_(b + i * 3, &b_x_f32x16, &b_y_f32x16, &b_z_f32x16);

        sum_a_x_f32x16 = _mm512_add_ps(sum_a_x_f32x16, a_x_f32x16);
        sum_a_y_f32x16 = _mm512_add_ps(sum_a_y_f32x16, a_y_f32x16);
        sum_a_z_f32x16 = _mm512_add_ps(sum_a_z_f32x16, a_z_f32x16);
        sum_b_x_f32x16 = _mm512_add_ps(sum_b_x_f32x16, b_x_f32x16);
        sum_b_y_f32x16 = _mm512_add_ps(sum_b_y_f32x16, b_y_f32x16);
        sum_b_z_f32x16 = _mm512_add_ps(sum_b_z_f32x16, b_z_f32x16);

        __m512 delta_x_f32x16 = _mm512_sub_ps(a_x_f32x16, b_x_f32x16);
        __m512 delta_y_f32x16 = _mm512_sub_ps(a_y_f32x16, b_y_f32x16);
        __m512 delta_z_f32x16 = _mm512_sub_ps(a_z_f32x16, b_z_f32x16);

        sum_squared_x_f32x16 = _mm512_fmadd_ps(delta_x_f32x16, delta_x_f32x16, sum_squared_x_f32x16);
        sum_squared_y_f32x16 = _mm512_fmadd_ps(delta_y_f32x16, delta_y_f32x16, sum_squared_y_f32x16);
        sum_squared_z_f32x16 = _mm512_fmadd_ps(delta_z_f32x16, delta_z_f32x16, sum_squared_z_f32x16);
    }

    // Tail: use masked gather
    if (i < n) {
        nk_size_t tail = n - i;
        __mmask16 mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, tail);
        nk_f32_t const *a_tail = a + i * 3;
        nk_f32_t const *b_tail = b + i * 3;

        a_x_f32x16 = _mm512_mask_i32gather_ps(zeros_f32x16, mask, gather_idx_i32x16, a_tail + 0, 4);
        a_y_f32x16 = _mm512_mask_i32gather_ps(zeros_f32x16, mask, gather_idx_i32x16, a_tail + 1, 4);
        a_z_f32x16 = _mm512_mask_i32gather_ps(zeros_f32x16, mask, gather_idx_i32x16, a_tail + 2, 4);
        b_x_f32x16 = _mm512_mask_i32gather_ps(zeros_f32x16, mask, gather_idx_i32x16, b_tail + 0, 4);
        b_y_f32x16 = _mm512_mask_i32gather_ps(zeros_f32x16, mask, gather_idx_i32x16, b_tail + 1, 4);
        b_z_f32x16 = _mm512_mask_i32gather_ps(zeros_f32x16, mask, gather_idx_i32x16, b_tail + 2, 4);

        sum_a_x_f32x16 = _mm512_add_ps(sum_a_x_f32x16, a_x_f32x16);
        sum_a_y_f32x16 = _mm512_add_ps(sum_a_y_f32x16, a_y_f32x16);
        sum_a_z_f32x16 = _mm512_add_ps(sum_a_z_f32x16, a_z_f32x16);
        sum_b_x_f32x16 = _mm512_add_ps(sum_b_x_f32x16, b_x_f32x16);
        sum_b_y_f32x16 = _mm512_add_ps(sum_b_y_f32x16, b_y_f32x16);
        sum_b_z_f32x16 = _mm512_add_ps(sum_b_z_f32x16, b_z_f32x16);

        __m512 delta_x_f32x16 = _mm512_sub_ps(a_x_f32x16, b_x_f32x16);
        __m512 delta_y_f32x16 = _mm512_sub_ps(a_y_f32x16, b_y_f32x16);
        __m512 delta_z_f32x16 = _mm512_sub_ps(a_z_f32x16, b_z_f32x16);

        sum_squared_x_f32x16 = _mm512_fmadd_ps(delta_x_f32x16, delta_x_f32x16, sum_squared_x_f32x16);
        sum_squared_y_f32x16 = _mm512_fmadd_ps(delta_y_f32x16, delta_y_f32x16, sum_squared_y_f32x16);
        sum_squared_z_f32x16 = _mm512_fmadd_ps(delta_z_f32x16, delta_z_f32x16, sum_squared_z_f32x16);
    }

    // Reduce and compute centroids
    nk_f32_t inv_n = 1.0f / (nk_f32_t)n;
    nk_f32_t centroid_a_x = _mm512_reduce_add_ps(sum_a_x_f32x16) * inv_n;
    nk_f32_t centroid_a_y = _mm512_reduce_add_ps(sum_a_y_f32x16) * inv_n;
    nk_f32_t centroid_a_z = _mm512_reduce_add_ps(sum_a_z_f32x16) * inv_n;
    nk_f32_t centroid_b_x = _mm512_reduce_add_ps(sum_b_x_f32x16) * inv_n;
    nk_f32_t centroid_b_y = _mm512_reduce_add_ps(sum_b_y_f32x16) * inv_n;
    nk_f32_t centroid_b_z = _mm512_reduce_add_ps(sum_b_z_f32x16) * inv_n;

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

    // Compute RMSD using the formula:
    // RMSD = sqrt(E[(a-b)^2] - (mean_a - mean_b)^2)
    nk_f32_t mean_diff_x = centroid_a_x - centroid_b_x;
    nk_f32_t mean_diff_y = centroid_a_y - centroid_b_y;
    nk_f32_t mean_diff_z = centroid_a_z - centroid_b_z;

    __m512 sum_squared_total_f32x16 = _mm512_add_ps(sum_squared_x_f32x16,
                                                    _mm512_add_ps(sum_squared_y_f32x16, sum_squared_z_f32x16));
    nk_f32_t sum_squared = _mm512_reduce_add_ps(sum_squared_total_f32x16);
    nk_f32_t mean_diff_sq = mean_diff_x * mean_diff_x + mean_diff_y * mean_diff_y + mean_diff_z * mean_diff_z;

    *result = nk_sqrt_f32_haswell_(sum_squared * inv_n - mean_diff_sq);
}

NK_PUBLIC void nk_kabsch_f32_skylake(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                     nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result) {
    // Optimized fused single-pass implementation using f32 accumulators.
    // Computes centroids and covariance matrix in one pass using the identity:
    //   H_ij = sum((a_i - mean_a) * (b_j - mean_b))
    //        = sum(a_i * b_j) - sum(a_i) * sum(b_j) / n
    __m512i const gather_idx_i32x16 = _mm512_setr_epi32(0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45);
    __m512 const zeros_f32x16 = _mm512_setzero_ps();

    // Accumulators for centroids (f32)
    __m512 sum_a_x_f32x16 = zeros_f32x16, sum_a_y_f32x16 = zeros_f32x16, sum_a_z_f32x16 = zeros_f32x16;
    __m512 sum_b_x_f32x16 = zeros_f32x16, sum_b_y_f32x16 = zeros_f32x16, sum_b_z_f32x16 = zeros_f32x16;

    // Accumulators for covariance matrix (sum of outer products, f32)
    __m512 cov_xx_f32x16 = zeros_f32x16, cov_xy_f32x16 = zeros_f32x16, cov_xz_f32x16 = zeros_f32x16;
    __m512 cov_yx_f32x16 = zeros_f32x16, cov_yy_f32x16 = zeros_f32x16, cov_yz_f32x16 = zeros_f32x16;
    __m512 cov_zx_f32x16 = zeros_f32x16, cov_zy_f32x16 = zeros_f32x16, cov_zz_f32x16 = zeros_f32x16;

    nk_size_t i = 0;
    __m512 a_x_f32x16, a_y_f32x16, a_z_f32x16, b_x_f32x16, b_y_f32x16, b_z_f32x16;

    // Fused single-pass: accumulate sums and outer products together
    for (; i + 16 <= n; i += 16) {
        nk_deinterleave_f32x16_skylake_(a + i * 3, &a_x_f32x16, &a_y_f32x16, &a_z_f32x16);
        nk_deinterleave_f32x16_skylake_(b + i * 3, &b_x_f32x16, &b_y_f32x16, &b_z_f32x16);

        // Accumulate centroids
        sum_a_x_f32x16 = _mm512_add_ps(sum_a_x_f32x16, a_x_f32x16);
        sum_a_y_f32x16 = _mm512_add_ps(sum_a_y_f32x16, a_y_f32x16);
        sum_a_z_f32x16 = _mm512_add_ps(sum_a_z_f32x16, a_z_f32x16);
        sum_b_x_f32x16 = _mm512_add_ps(sum_b_x_f32x16, b_x_f32x16);
        sum_b_y_f32x16 = _mm512_add_ps(sum_b_y_f32x16, b_y_f32x16);
        sum_b_z_f32x16 = _mm512_add_ps(sum_b_z_f32x16, b_z_f32x16);

        // Accumulate outer products (raw, not centered)
        cov_xx_f32x16 = _mm512_fmadd_ps(a_x_f32x16, b_x_f32x16, cov_xx_f32x16);
        cov_xy_f32x16 = _mm512_fmadd_ps(a_x_f32x16, b_y_f32x16, cov_xy_f32x16);
        cov_xz_f32x16 = _mm512_fmadd_ps(a_x_f32x16, b_z_f32x16, cov_xz_f32x16);
        cov_yx_f32x16 = _mm512_fmadd_ps(a_y_f32x16, b_x_f32x16, cov_yx_f32x16);
        cov_yy_f32x16 = _mm512_fmadd_ps(a_y_f32x16, b_y_f32x16, cov_yy_f32x16);
        cov_yz_f32x16 = _mm512_fmadd_ps(a_y_f32x16, b_z_f32x16, cov_yz_f32x16);
        cov_zx_f32x16 = _mm512_fmadd_ps(a_z_f32x16, b_x_f32x16, cov_zx_f32x16);
        cov_zy_f32x16 = _mm512_fmadd_ps(a_z_f32x16, b_y_f32x16, cov_zy_f32x16);
        cov_zz_f32x16 = _mm512_fmadd_ps(a_z_f32x16, b_z_f32x16, cov_zz_f32x16);
    }

    // Tail: masked gather for remaining points
    if (i < n) {
        nk_size_t tail = n - i;
        __mmask16 mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, tail);
        nk_f32_t const *a_tail = a + i * 3;
        nk_f32_t const *b_tail = b + i * 3;
        a_x_f32x16 = _mm512_mask_i32gather_ps(zeros_f32x16, mask, gather_idx_i32x16, a_tail + 0, 4);
        a_y_f32x16 = _mm512_mask_i32gather_ps(zeros_f32x16, mask, gather_idx_i32x16, a_tail + 1, 4);
        a_z_f32x16 = _mm512_mask_i32gather_ps(zeros_f32x16, mask, gather_idx_i32x16, a_tail + 2, 4);
        b_x_f32x16 = _mm512_mask_i32gather_ps(zeros_f32x16, mask, gather_idx_i32x16, b_tail + 0, 4);
        b_y_f32x16 = _mm512_mask_i32gather_ps(zeros_f32x16, mask, gather_idx_i32x16, b_tail + 1, 4);
        b_z_f32x16 = _mm512_mask_i32gather_ps(zeros_f32x16, mask, gather_idx_i32x16, b_tail + 2, 4);

        sum_a_x_f32x16 = _mm512_add_ps(sum_a_x_f32x16, a_x_f32x16);
        sum_a_y_f32x16 = _mm512_add_ps(sum_a_y_f32x16, a_y_f32x16);
        sum_a_z_f32x16 = _mm512_add_ps(sum_a_z_f32x16, a_z_f32x16);
        sum_b_x_f32x16 = _mm512_add_ps(sum_b_x_f32x16, b_x_f32x16);
        sum_b_y_f32x16 = _mm512_add_ps(sum_b_y_f32x16, b_y_f32x16);
        sum_b_z_f32x16 = _mm512_add_ps(sum_b_z_f32x16, b_z_f32x16);

        cov_xx_f32x16 = _mm512_fmadd_ps(a_x_f32x16, b_x_f32x16, cov_xx_f32x16);
        cov_xy_f32x16 = _mm512_fmadd_ps(a_x_f32x16, b_y_f32x16, cov_xy_f32x16);
        cov_xz_f32x16 = _mm512_fmadd_ps(a_x_f32x16, b_z_f32x16, cov_xz_f32x16);
        cov_yx_f32x16 = _mm512_fmadd_ps(a_y_f32x16, b_x_f32x16, cov_yx_f32x16);
        cov_yy_f32x16 = _mm512_fmadd_ps(a_y_f32x16, b_y_f32x16, cov_yy_f32x16);
        cov_yz_f32x16 = _mm512_fmadd_ps(a_y_f32x16, b_z_f32x16, cov_yz_f32x16);
        cov_zx_f32x16 = _mm512_fmadd_ps(a_z_f32x16, b_x_f32x16, cov_zx_f32x16);
        cov_zy_f32x16 = _mm512_fmadd_ps(a_z_f32x16, b_y_f32x16, cov_zy_f32x16);
        cov_zz_f32x16 = _mm512_fmadd_ps(a_z_f32x16, b_z_f32x16, cov_zz_f32x16);
    }

    // Reduce centroids
    nk_f32_t inv_n = 1.0f / (nk_f32_t)n;
    nk_f32_t sum_a_x = _mm512_reduce_add_ps(sum_a_x_f32x16);
    nk_f32_t sum_a_y = _mm512_reduce_add_ps(sum_a_y_f32x16);
    nk_f32_t sum_a_z = _mm512_reduce_add_ps(sum_a_z_f32x16);
    nk_f32_t sum_b_x = _mm512_reduce_add_ps(sum_b_x_f32x16);
    nk_f32_t sum_b_y = _mm512_reduce_add_ps(sum_b_y_f32x16);
    nk_f32_t sum_b_z = _mm512_reduce_add_ps(sum_b_z_f32x16);

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

    // Compute centered covariance matrix: H_ij = sum(a_i*b_j) - sum_a_i * sum_b_j / n
    nk_f32_t cross_covariance[9];
    cross_covariance[0] = _mm512_reduce_add_ps(cov_xx_f32x16) - sum_a_x * sum_b_x * inv_n;
    cross_covariance[1] = _mm512_reduce_add_ps(cov_xy_f32x16) - sum_a_x * sum_b_y * inv_n;
    cross_covariance[2] = _mm512_reduce_add_ps(cov_xz_f32x16) - sum_a_x * sum_b_z * inv_n;
    cross_covariance[3] = _mm512_reduce_add_ps(cov_yx_f32x16) - sum_a_y * sum_b_x * inv_n;
    cross_covariance[4] = _mm512_reduce_add_ps(cov_yy_f32x16) - sum_a_y * sum_b_y * inv_n;
    cross_covariance[5] = _mm512_reduce_add_ps(cov_yz_f32x16) - sum_a_y * sum_b_z * inv_n;
    cross_covariance[6] = _mm512_reduce_add_ps(cov_zx_f32x16) - sum_a_z * sum_b_x * inv_n;
    cross_covariance[7] = _mm512_reduce_add_ps(cov_zy_f32x16) - sum_a_z * sum_b_y * inv_n;
    cross_covariance[8] = _mm512_reduce_add_ps(cov_zz_f32x16) - sum_a_z * sum_b_z * inv_n;

    // Step 3: SVD
    nk_f32_t svd_u[9], svd_s[9], svd_v[9];
    nk_svd3x3_f32_(cross_covariance, svd_u, svd_s, svd_v);

    // Step 4: R = V * U^T
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

    // Handle reflection
    nk_f32_t det = nk_det3x3_f32_(r);
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

    /* Output rotation matrix and scale=1.0 */
    if (rotation) {
        for (int j = 0; j < 9; ++j) rotation[j] = r[j];
    }
    if (scale) *scale = 1.0f;

    // Compute RMSD after optimal rotation
    nk_f32_t sum_squared = nk_transformed_ssd_f32_skylake_(a, b, n, r, 1.0f, centroid_a_x, centroid_a_y, centroid_a_z,
                                                           centroid_b_x, centroid_b_y, centroid_b_z);
    *result = nk_sqrt_f32_haswell_(sum_squared * inv_n);
}

NK_PUBLIC void nk_rmsd_f64_skylake(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                                   nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result) {
    /* RMSD uses identity rotation and scale=1.0 */
    if (rotation) {
        rotation[0] = 1;
        rotation[1] = 0;
        rotation[2] = 0;
        rotation[3] = 0;
        rotation[4] = 1;
        rotation[5] = 0;
        rotation[6] = 0;
        rotation[7] = 0;
        rotation[8] = 1;
    }
    if (scale) *scale = 1.0;
    // Optimized fused single-pass implementation for f64.
    // Computes centroids and squared differences in one pass using the identity:
    //   RMSD = sqrt(E[(a-mean_a) - (b-mean_b)]^2)
    //        = sqrt(E[(a-b)^2] - (mean_a - mean_b)^2)
    __m512i const gather_idx_i64x8 = _mm512_setr_epi64(0, 3, 6, 9, 12, 15, 18, 21);
    __m512d const zeros_f64x8 = _mm512_setzero_pd();

    // Accumulators for centroids and squared differences
    __m512d sum_a_x_f64x8 = zeros_f64x8, sum_a_y_f64x8 = zeros_f64x8, sum_a_z_f64x8 = zeros_f64x8;
    __m512d sum_b_x_f64x8 = zeros_f64x8, sum_b_y_f64x8 = zeros_f64x8, sum_b_z_f64x8 = zeros_f64x8;
    __m512d sum_squared_x_f64x8 = zeros_f64x8, sum_squared_y_f64x8 = zeros_f64x8, sum_squared_z_f64x8 = zeros_f64x8;

    __m512d a_x_f64x8, a_y_f64x8, a_z_f64x8, b_x_f64x8, b_y_f64x8, b_z_f64x8;
    nk_size_t i = 0;

    // Main loop with 2x unrolling for better latency hiding
    for (; i + 16 <= n; i += 16) {
        // Iteration 0
        nk_deinterleave_f64x8_skylake_(a + i * 3, &a_x_f64x8, &a_y_f64x8, &a_z_f64x8);
        nk_deinterleave_f64x8_skylake_(b + i * 3, &b_x_f64x8, &b_y_f64x8, &b_z_f64x8);

        sum_a_x_f64x8 = _mm512_add_pd(sum_a_x_f64x8, a_x_f64x8),
        sum_a_y_f64x8 = _mm512_add_pd(sum_a_y_f64x8, a_y_f64x8),
        sum_a_z_f64x8 = _mm512_add_pd(sum_a_z_f64x8, a_z_f64x8);
        sum_b_x_f64x8 = _mm512_add_pd(sum_b_x_f64x8, b_x_f64x8),
        sum_b_y_f64x8 = _mm512_add_pd(sum_b_y_f64x8, b_y_f64x8),
        sum_b_z_f64x8 = _mm512_add_pd(sum_b_z_f64x8, b_z_f64x8);

        __m512d delta_x_f64x8 = _mm512_sub_pd(a_x_f64x8, b_x_f64x8),
                delta_y_f64x8 = _mm512_sub_pd(a_y_f64x8, b_y_f64x8),
                delta_z_f64x8 = _mm512_sub_pd(a_z_f64x8, b_z_f64x8);
        sum_squared_x_f64x8 = _mm512_fmadd_pd(delta_x_f64x8, delta_x_f64x8, sum_squared_x_f64x8);
        sum_squared_y_f64x8 = _mm512_fmadd_pd(delta_y_f64x8, delta_y_f64x8, sum_squared_y_f64x8);
        sum_squared_z_f64x8 = _mm512_fmadd_pd(delta_z_f64x8, delta_z_f64x8, sum_squared_z_f64x8);

        // Iteration 1
        __m512d a_x1_f64x8, a_y1_f64x8, a_z1_f64x8, b_x1_f64x8, b_y1_f64x8, b_z1_f64x8;
        nk_deinterleave_f64x8_skylake_(a + (i + 8) * 3, &a_x1_f64x8, &a_y1_f64x8, &a_z1_f64x8);
        nk_deinterleave_f64x8_skylake_(b + (i + 8) * 3, &b_x1_f64x8, &b_y1_f64x8, &b_z1_f64x8);

        sum_a_x_f64x8 = _mm512_add_pd(sum_a_x_f64x8, a_x1_f64x8),
        sum_a_y_f64x8 = _mm512_add_pd(sum_a_y_f64x8, a_y1_f64x8),
        sum_a_z_f64x8 = _mm512_add_pd(sum_a_z_f64x8, a_z1_f64x8);
        sum_b_x_f64x8 = _mm512_add_pd(sum_b_x_f64x8, b_x1_f64x8),
        sum_b_y_f64x8 = _mm512_add_pd(sum_b_y_f64x8, b_y1_f64x8),
        sum_b_z_f64x8 = _mm512_add_pd(sum_b_z_f64x8, b_z1_f64x8);

        __m512d delta_x1_f64x8 = _mm512_sub_pd(a_x1_f64x8, b_x1_f64x8),
                delta_y1_f64x8 = _mm512_sub_pd(a_y1_f64x8, b_y1_f64x8),
                delta_z1_f64x8 = _mm512_sub_pd(a_z1_f64x8, b_z1_f64x8);
        sum_squared_x_f64x8 = _mm512_fmadd_pd(delta_x1_f64x8, delta_x1_f64x8, sum_squared_x_f64x8);
        sum_squared_y_f64x8 = _mm512_fmadd_pd(delta_y1_f64x8, delta_y1_f64x8, sum_squared_y_f64x8);
        sum_squared_z_f64x8 = _mm512_fmadd_pd(delta_z1_f64x8, delta_z1_f64x8, sum_squared_z_f64x8);
    }

    // Handle 8-point remainder
    for (; i + 8 <= n; i += 8) {
        nk_deinterleave_f64x8_skylake_(a + i * 3, &a_x_f64x8, &a_y_f64x8, &a_z_f64x8);
        nk_deinterleave_f64x8_skylake_(b + i * 3, &b_x_f64x8, &b_y_f64x8, &b_z_f64x8);

        sum_a_x_f64x8 = _mm512_add_pd(sum_a_x_f64x8, a_x_f64x8),
        sum_a_y_f64x8 = _mm512_add_pd(sum_a_y_f64x8, a_y_f64x8),
        sum_a_z_f64x8 = _mm512_add_pd(sum_a_z_f64x8, a_z_f64x8);
        sum_b_x_f64x8 = _mm512_add_pd(sum_b_x_f64x8, b_x_f64x8),
        sum_b_y_f64x8 = _mm512_add_pd(sum_b_y_f64x8, b_y_f64x8),
        sum_b_z_f64x8 = _mm512_add_pd(sum_b_z_f64x8, b_z_f64x8);

        __m512d delta_x_f64x8 = _mm512_sub_pd(a_x_f64x8, b_x_f64x8),
                delta_y_f64x8 = _mm512_sub_pd(a_y_f64x8, b_y_f64x8),
                delta_z_f64x8 = _mm512_sub_pd(a_z_f64x8, b_z_f64x8);
        sum_squared_x_f64x8 = _mm512_fmadd_pd(delta_x_f64x8, delta_x_f64x8, sum_squared_x_f64x8);
        sum_squared_y_f64x8 = _mm512_fmadd_pd(delta_y_f64x8, delta_y_f64x8, sum_squared_y_f64x8);
        sum_squared_z_f64x8 = _mm512_fmadd_pd(delta_z_f64x8, delta_z_f64x8, sum_squared_z_f64x8);
    }

    // Tail: use masked gather
    if (i < n) {
        nk_size_t tail = n - i;
        __mmask8 mask = (__mmask8)_bzhi_u32(0xFF, tail);
        nk_f64_t const *a_tail = a + i * 3;
        nk_f64_t const *b_tail = b + i * 3;

        a_x_f64x8 = _mm512_mask_i64gather_pd(zeros_f64x8, mask, gather_idx_i64x8, a_tail + 0, 8);
        a_y_f64x8 = _mm512_mask_i64gather_pd(zeros_f64x8, mask, gather_idx_i64x8, a_tail + 1, 8);
        a_z_f64x8 = _mm512_mask_i64gather_pd(zeros_f64x8, mask, gather_idx_i64x8, a_tail + 2, 8);
        b_x_f64x8 = _mm512_mask_i64gather_pd(zeros_f64x8, mask, gather_idx_i64x8, b_tail + 0, 8);
        b_y_f64x8 = _mm512_mask_i64gather_pd(zeros_f64x8, mask, gather_idx_i64x8, b_tail + 1, 8);
        b_z_f64x8 = _mm512_mask_i64gather_pd(zeros_f64x8, mask, gather_idx_i64x8, b_tail + 2, 8);

        sum_a_x_f64x8 = _mm512_add_pd(sum_a_x_f64x8, a_x_f64x8),
        sum_a_y_f64x8 = _mm512_add_pd(sum_a_y_f64x8, a_y_f64x8),
        sum_a_z_f64x8 = _mm512_add_pd(sum_a_z_f64x8, a_z_f64x8);
        sum_b_x_f64x8 = _mm512_add_pd(sum_b_x_f64x8, b_x_f64x8),
        sum_b_y_f64x8 = _mm512_add_pd(sum_b_y_f64x8, b_y_f64x8),
        sum_b_z_f64x8 = _mm512_add_pd(sum_b_z_f64x8, b_z_f64x8);

        __m512d delta_x_f64x8 = _mm512_sub_pd(a_x_f64x8, b_x_f64x8),
                delta_y_f64x8 = _mm512_sub_pd(a_y_f64x8, b_y_f64x8),
                delta_z_f64x8 = _mm512_sub_pd(a_z_f64x8, b_z_f64x8);
        sum_squared_x_f64x8 = _mm512_fmadd_pd(delta_x_f64x8, delta_x_f64x8, sum_squared_x_f64x8);
        sum_squared_y_f64x8 = _mm512_fmadd_pd(delta_y_f64x8, delta_y_f64x8, sum_squared_y_f64x8);
        sum_squared_z_f64x8 = _mm512_fmadd_pd(delta_z_f64x8, delta_z_f64x8, sum_squared_z_f64x8);
    }

    // Reduce and compute centroids
    nk_f64_t inv_n = 1.0 / (nk_f64_t)n;
    nk_f64_t centroid_a_x = _mm512_reduce_add_pd(sum_a_x_f64x8) * inv_n;
    nk_f64_t centroid_a_y = _mm512_reduce_add_pd(sum_a_y_f64x8) * inv_n;
    nk_f64_t centroid_a_z = _mm512_reduce_add_pd(sum_a_z_f64x8) * inv_n;
    nk_f64_t centroid_b_x = _mm512_reduce_add_pd(sum_b_x_f64x8) * inv_n;
    nk_f64_t centroid_b_y = _mm512_reduce_add_pd(sum_b_y_f64x8) * inv_n;
    nk_f64_t centroid_b_z = _mm512_reduce_add_pd(sum_b_z_f64x8) * inv_n;

    if (a_centroid) a_centroid[0] = centroid_a_x, a_centroid[1] = centroid_a_y, a_centroid[2] = centroid_a_z;
    if (b_centroid) b_centroid[0] = centroid_b_x, b_centroid[1] = centroid_b_y, b_centroid[2] = centroid_b_z;

    // Compute RMSD using the formula:
    // RMSD = sqrt(E[(a-b)^2] - (mean_a - mean_b)^2)
    nk_f64_t mean_diff_x = centroid_a_x - centroid_b_x, mean_diff_y = centroid_a_y - centroid_b_y,
             mean_diff_z = centroid_a_z - centroid_b_z;
    __m512d sum_squared_total_f64x8 = _mm512_add_pd(sum_squared_x_f64x8,
                                                    _mm512_add_pd(sum_squared_y_f64x8, sum_squared_z_f64x8));
    nk_f64_t sum_squared = _mm512_reduce_add_pd(sum_squared_total_f64x8);
    nk_f64_t mean_diff_sq = mean_diff_x * mean_diff_x + mean_diff_y * mean_diff_y + mean_diff_z * mean_diff_z;

    *result = nk_sqrt_f64_haswell_(sum_squared * inv_n - mean_diff_sq);
}

NK_PUBLIC void nk_kabsch_f64_skylake(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                                     nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result) {
    // Optimized fused single-pass implementation for f64.
    // Computes centroids and covariance matrix in one pass using the identity:
    //   H_ij = sum((a_i - mean_a) * (b_j - mean_b))
    //        = sum(a_i * b_j) - sum(a_i) * sum(b_j) / n
    __m512i const gather_idx_i64x8 = _mm512_setr_epi64(0, 3, 6, 9, 12, 15, 18, 21);
    __m512d const zeros_f64x8 = _mm512_setzero_pd();

    // Accumulators for centroids
    __m512d sum_a_x_f64x8 = zeros_f64x8, sum_a_y_f64x8 = zeros_f64x8, sum_a_z_f64x8 = zeros_f64x8;
    __m512d sum_b_x_f64x8 = zeros_f64x8, sum_b_y_f64x8 = zeros_f64x8, sum_b_z_f64x8 = zeros_f64x8;

    // Accumulators for covariance matrix (sum of outer products)
    __m512d cov_xx_f64x8 = zeros_f64x8, cov_xy_f64x8 = zeros_f64x8, cov_xz_f64x8 = zeros_f64x8;
    __m512d cov_yx_f64x8 = zeros_f64x8, cov_yy_f64x8 = zeros_f64x8, cov_yz_f64x8 = zeros_f64x8;
    __m512d cov_zx_f64x8 = zeros_f64x8, cov_zy_f64x8 = zeros_f64x8, cov_zz_f64x8 = zeros_f64x8;

    nk_size_t i = 0;
    __m512d a_x_f64x8, a_y_f64x8, a_z_f64x8, b_x_f64x8, b_y_f64x8, b_z_f64x8;

    // Fused single-pass: accumulate sums and outer products together
    for (; i + 8 <= n; i += 8) {
        nk_deinterleave_f64x8_skylake_(a + i * 3, &a_x_f64x8, &a_y_f64x8, &a_z_f64x8);
        nk_deinterleave_f64x8_skylake_(b + i * 3, &b_x_f64x8, &b_y_f64x8, &b_z_f64x8);

        // Accumulate centroids
        sum_a_x_f64x8 = _mm512_add_pd(sum_a_x_f64x8, a_x_f64x8),
        sum_a_y_f64x8 = _mm512_add_pd(sum_a_y_f64x8, a_y_f64x8),
        sum_a_z_f64x8 = _mm512_add_pd(sum_a_z_f64x8, a_z_f64x8);
        sum_b_x_f64x8 = _mm512_add_pd(sum_b_x_f64x8, b_x_f64x8),
        sum_b_y_f64x8 = _mm512_add_pd(sum_b_y_f64x8, b_y_f64x8),
        sum_b_z_f64x8 = _mm512_add_pd(sum_b_z_f64x8, b_z_f64x8);

        // Accumulate outer products (raw, not centered)
        cov_xx_f64x8 = _mm512_fmadd_pd(a_x_f64x8, b_x_f64x8, cov_xx_f64x8),
        cov_xy_f64x8 = _mm512_fmadd_pd(a_x_f64x8, b_y_f64x8, cov_xy_f64x8),
        cov_xz_f64x8 = _mm512_fmadd_pd(a_x_f64x8, b_z_f64x8, cov_xz_f64x8);
        cov_yx_f64x8 = _mm512_fmadd_pd(a_y_f64x8, b_x_f64x8, cov_yx_f64x8),
        cov_yy_f64x8 = _mm512_fmadd_pd(a_y_f64x8, b_y_f64x8, cov_yy_f64x8),
        cov_yz_f64x8 = _mm512_fmadd_pd(a_y_f64x8, b_z_f64x8, cov_yz_f64x8);
        cov_zx_f64x8 = _mm512_fmadd_pd(a_z_f64x8, b_x_f64x8, cov_zx_f64x8),
        cov_zy_f64x8 = _mm512_fmadd_pd(a_z_f64x8, b_y_f64x8, cov_zy_f64x8),
        cov_zz_f64x8 = _mm512_fmadd_pd(a_z_f64x8, b_z_f64x8, cov_zz_f64x8);
    }

    // Tail: masked gather for remaining points
    if (i < n) {
        nk_size_t tail = n - i;
        __mmask8 mask = (__mmask8)_bzhi_u32(0xFF, tail);
        nk_f64_t const *a_tail = a + i * 3;
        nk_f64_t const *b_tail = b + i * 3;

        a_x_f64x8 = _mm512_mask_i64gather_pd(zeros_f64x8, mask, gather_idx_i64x8, a_tail + 0, 8);
        a_y_f64x8 = _mm512_mask_i64gather_pd(zeros_f64x8, mask, gather_idx_i64x8, a_tail + 1, 8);
        a_z_f64x8 = _mm512_mask_i64gather_pd(zeros_f64x8, mask, gather_idx_i64x8, a_tail + 2, 8);
        b_x_f64x8 = _mm512_mask_i64gather_pd(zeros_f64x8, mask, gather_idx_i64x8, b_tail + 0, 8);
        b_y_f64x8 = _mm512_mask_i64gather_pd(zeros_f64x8, mask, gather_idx_i64x8, b_tail + 1, 8);
        b_z_f64x8 = _mm512_mask_i64gather_pd(zeros_f64x8, mask, gather_idx_i64x8, b_tail + 2, 8);

        sum_a_x_f64x8 = _mm512_add_pd(sum_a_x_f64x8, a_x_f64x8),
        sum_a_y_f64x8 = _mm512_add_pd(sum_a_y_f64x8, a_y_f64x8),
        sum_a_z_f64x8 = _mm512_add_pd(sum_a_z_f64x8, a_z_f64x8);
        sum_b_x_f64x8 = _mm512_add_pd(sum_b_x_f64x8, b_x_f64x8),
        sum_b_y_f64x8 = _mm512_add_pd(sum_b_y_f64x8, b_y_f64x8),
        sum_b_z_f64x8 = _mm512_add_pd(sum_b_z_f64x8, b_z_f64x8);

        cov_xx_f64x8 = _mm512_fmadd_pd(a_x_f64x8, b_x_f64x8, cov_xx_f64x8),
        cov_xy_f64x8 = _mm512_fmadd_pd(a_x_f64x8, b_y_f64x8, cov_xy_f64x8),
        cov_xz_f64x8 = _mm512_fmadd_pd(a_x_f64x8, b_z_f64x8, cov_xz_f64x8);
        cov_yx_f64x8 = _mm512_fmadd_pd(a_y_f64x8, b_x_f64x8, cov_yx_f64x8),
        cov_yy_f64x8 = _mm512_fmadd_pd(a_y_f64x8, b_y_f64x8, cov_yy_f64x8),
        cov_yz_f64x8 = _mm512_fmadd_pd(a_y_f64x8, b_z_f64x8, cov_yz_f64x8);
        cov_zx_f64x8 = _mm512_fmadd_pd(a_z_f64x8, b_x_f64x8, cov_zx_f64x8),
        cov_zy_f64x8 = _mm512_fmadd_pd(a_z_f64x8, b_y_f64x8, cov_zy_f64x8),
        cov_zz_f64x8 = _mm512_fmadd_pd(a_z_f64x8, b_z_f64x8, cov_zz_f64x8);
    }

    // Reduce centroids
    nk_f64_t inv_n = 1.0 / (nk_f64_t)n;
    nk_f64_t sum_a_x = _mm512_reduce_add_pd(sum_a_x_f64x8), sum_a_y = _mm512_reduce_add_pd(sum_a_y_f64x8),
             sum_a_z = _mm512_reduce_add_pd(sum_a_z_f64x8);
    nk_f64_t sum_b_x = _mm512_reduce_add_pd(sum_b_x_f64x8), sum_b_y = _mm512_reduce_add_pd(sum_b_y_f64x8),
             sum_b_z = _mm512_reduce_add_pd(sum_b_z_f64x8);

    nk_f64_t centroid_a_x = sum_a_x * inv_n, centroid_a_y = sum_a_y * inv_n, centroid_a_z = sum_a_z * inv_n;
    nk_f64_t centroid_b_x = sum_b_x * inv_n, centroid_b_y = sum_b_y * inv_n, centroid_b_z = sum_b_z * inv_n;

    if (a_centroid) a_centroid[0] = centroid_a_x, a_centroid[1] = centroid_a_y, a_centroid[2] = centroid_a_z;
    if (b_centroid) b_centroid[0] = centroid_b_x, b_centroid[1] = centroid_b_y, b_centroid[2] = centroid_b_z;

    // Compute centered covariance matrix: H_ij = sum(a_i*b_j) - sum_a_i * sum_b_j / n
    nk_f64_t cross_covariance[9];
    cross_covariance[0] = _mm512_reduce_add_pd(cov_xx_f64x8) - sum_a_x * sum_b_x * inv_n;
    cross_covariance[1] = _mm512_reduce_add_pd(cov_xy_f64x8) - sum_a_x * sum_b_y * inv_n;
    cross_covariance[2] = _mm512_reduce_add_pd(cov_xz_f64x8) - sum_a_x * sum_b_z * inv_n;
    cross_covariance[3] = _mm512_reduce_add_pd(cov_yx_f64x8) - sum_a_y * sum_b_x * inv_n;
    cross_covariance[4] = _mm512_reduce_add_pd(cov_yy_f64x8) - sum_a_y * sum_b_y * inv_n;
    cross_covariance[5] = _mm512_reduce_add_pd(cov_yz_f64x8) - sum_a_y * sum_b_z * inv_n;
    cross_covariance[6] = _mm512_reduce_add_pd(cov_zx_f64x8) - sum_a_z * sum_b_x * inv_n;
    cross_covariance[7] = _mm512_reduce_add_pd(cov_zy_f64x8) - sum_a_z * sum_b_y * inv_n;
    cross_covariance[8] = _mm512_reduce_add_pd(cov_zz_f64x8) - sum_a_z * sum_b_z * inv_n;

    // SVD using f64 for full precision
    nk_f64_t svd_u[9], svd_s[3], svd_v[9];
    nk_svd3x3_f64_(cross_covariance, svd_u, svd_s, svd_v);

    // R = V * U^T
    nk_f64_t r[9];
    r[0] = svd_v[0] * svd_u[0] + svd_v[1] * svd_u[1] + svd_v[2] * svd_u[2];
    r[1] = svd_v[0] * svd_u[3] + svd_v[1] * svd_u[4] + svd_v[2] * svd_u[5];
    r[2] = svd_v[0] * svd_u[6] + svd_v[1] * svd_u[7] + svd_v[2] * svd_u[8];
    r[3] = svd_v[3] * svd_u[0] + svd_v[4] * svd_u[1] + svd_v[5] * svd_u[2];
    r[4] = svd_v[3] * svd_u[3] + svd_v[4] * svd_u[4] + svd_v[5] * svd_u[5];
    r[5] = svd_v[3] * svd_u[6] + svd_v[4] * svd_u[7] + svd_v[5] * svd_u[8];
    r[6] = svd_v[6] * svd_u[0] + svd_v[7] * svd_u[1] + svd_v[8] * svd_u[2];
    r[7] = svd_v[6] * svd_u[3] + svd_v[7] * svd_u[4] + svd_v[8] * svd_u[5];
    r[8] = svd_v[6] * svd_u[6] + svd_v[7] * svd_u[7] + svd_v[8] * svd_u[8];

    // Handle reflection
    if (nk_det3x3_f64_(r) < 0) {
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

    /* Output rotation matrix and scale=1.0 */
    if (rotation) {
        for (int j = 0; j < 9; ++j) rotation[j] = (nk_f64_t)r[j];
    }
    if (scale) *scale = 1.0;

    // Compute RMSD after optimal rotation
    nk_f64_t sum_squared = nk_transformed_ssd_f64_skylake_(a, b, n, r, 1.0, centroid_a_x, centroid_a_y, centroid_a_z,
                                                           centroid_b_x, centroid_b_y, centroid_b_z);
    *result = nk_sqrt_f64_haswell_(sum_squared * inv_n);
}

NK_PUBLIC void nk_umeyama_f32_skylake(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                      nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result) {
    // Fused single-pass: centroids, covariance, and variance of A using f32 accumulators
    __m512i const gather_idx_i32x16 = _mm512_setr_epi32(0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45);
    __m512 const zeros_f32x16 = _mm512_setzero_ps();

    // Accumulators for centroids (f32)
    __m512 sum_a_x_f32x16 = zeros_f32x16, sum_a_y_f32x16 = zeros_f32x16, sum_a_z_f32x16 = zeros_f32x16;
    __m512 sum_b_x_f32x16 = zeros_f32x16, sum_b_y_f32x16 = zeros_f32x16, sum_b_z_f32x16 = zeros_f32x16;
    __m512 cov_xx_f32x16 = zeros_f32x16, cov_xy_f32x16 = zeros_f32x16, cov_xz_f32x16 = zeros_f32x16;
    __m512 cov_yx_f32x16 = zeros_f32x16, cov_yy_f32x16 = zeros_f32x16, cov_yz_f32x16 = zeros_f32x16;
    __m512 cov_zx_f32x16 = zeros_f32x16, cov_zy_f32x16 = zeros_f32x16, cov_zz_f32x16 = zeros_f32x16;
    __m512 variance_a_f32x16 = zeros_f32x16;

    nk_size_t i = 0;
    __m512 a_x_f32x16, a_y_f32x16, a_z_f32x16, b_x_f32x16, b_y_f32x16, b_z_f32x16;

    for (; i + 16 <= n; i += 16) {
        nk_deinterleave_f32x16_skylake_(a + i * 3, &a_x_f32x16, &a_y_f32x16, &a_z_f32x16);
        nk_deinterleave_f32x16_skylake_(b + i * 3, &b_x_f32x16, &b_y_f32x16, &b_z_f32x16);

        sum_a_x_f32x16 = _mm512_add_ps(sum_a_x_f32x16, a_x_f32x16);
        sum_a_y_f32x16 = _mm512_add_ps(sum_a_y_f32x16, a_y_f32x16);
        sum_a_z_f32x16 = _mm512_add_ps(sum_a_z_f32x16, a_z_f32x16);
        sum_b_x_f32x16 = _mm512_add_ps(sum_b_x_f32x16, b_x_f32x16);
        sum_b_y_f32x16 = _mm512_add_ps(sum_b_y_f32x16, b_y_f32x16);
        sum_b_z_f32x16 = _mm512_add_ps(sum_b_z_f32x16, b_z_f32x16);

        cov_xx_f32x16 = _mm512_fmadd_ps(a_x_f32x16, b_x_f32x16, cov_xx_f32x16);
        cov_xy_f32x16 = _mm512_fmadd_ps(a_x_f32x16, b_y_f32x16, cov_xy_f32x16);
        cov_xz_f32x16 = _mm512_fmadd_ps(a_x_f32x16, b_z_f32x16, cov_xz_f32x16);
        cov_yx_f32x16 = _mm512_fmadd_ps(a_y_f32x16, b_x_f32x16, cov_yx_f32x16);
        cov_yy_f32x16 = _mm512_fmadd_ps(a_y_f32x16, b_y_f32x16, cov_yy_f32x16);
        cov_yz_f32x16 = _mm512_fmadd_ps(a_y_f32x16, b_z_f32x16, cov_yz_f32x16);
        cov_zx_f32x16 = _mm512_fmadd_ps(a_z_f32x16, b_x_f32x16, cov_zx_f32x16);
        cov_zy_f32x16 = _mm512_fmadd_ps(a_z_f32x16, b_y_f32x16, cov_zy_f32x16);
        cov_zz_f32x16 = _mm512_fmadd_ps(a_z_f32x16, b_z_f32x16, cov_zz_f32x16);
        variance_a_f32x16 = _mm512_fmadd_ps(a_x_f32x16, a_x_f32x16, variance_a_f32x16);
        variance_a_f32x16 = _mm512_fmadd_ps(a_y_f32x16, a_y_f32x16, variance_a_f32x16);
        variance_a_f32x16 = _mm512_fmadd_ps(a_z_f32x16, a_z_f32x16, variance_a_f32x16);
    }

    if (i < n) {
        nk_size_t tail = n - i;
        __mmask16 mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, tail);
        nk_f32_t const *a_tail = a + i * 3;
        nk_f32_t const *b_tail = b + i * 3;
        a_x_f32x16 = _mm512_mask_i32gather_ps(zeros_f32x16, mask, gather_idx_i32x16, a_tail + 0, 4);
        a_y_f32x16 = _mm512_mask_i32gather_ps(zeros_f32x16, mask, gather_idx_i32x16, a_tail + 1, 4);
        a_z_f32x16 = _mm512_mask_i32gather_ps(zeros_f32x16, mask, gather_idx_i32x16, a_tail + 2, 4);
        b_x_f32x16 = _mm512_mask_i32gather_ps(zeros_f32x16, mask, gather_idx_i32x16, b_tail + 0, 4);
        b_y_f32x16 = _mm512_mask_i32gather_ps(zeros_f32x16, mask, gather_idx_i32x16, b_tail + 1, 4);
        b_z_f32x16 = _mm512_mask_i32gather_ps(zeros_f32x16, mask, gather_idx_i32x16, b_tail + 2, 4);

        sum_a_x_f32x16 = _mm512_add_ps(sum_a_x_f32x16, a_x_f32x16);
        sum_a_y_f32x16 = _mm512_add_ps(sum_a_y_f32x16, a_y_f32x16);
        sum_a_z_f32x16 = _mm512_add_ps(sum_a_z_f32x16, a_z_f32x16);
        sum_b_x_f32x16 = _mm512_add_ps(sum_b_x_f32x16, b_x_f32x16);
        sum_b_y_f32x16 = _mm512_add_ps(sum_b_y_f32x16, b_y_f32x16);
        sum_b_z_f32x16 = _mm512_add_ps(sum_b_z_f32x16, b_z_f32x16);

        cov_xx_f32x16 = _mm512_fmadd_ps(a_x_f32x16, b_x_f32x16, cov_xx_f32x16);
        cov_xy_f32x16 = _mm512_fmadd_ps(a_x_f32x16, b_y_f32x16, cov_xy_f32x16);
        cov_xz_f32x16 = _mm512_fmadd_ps(a_x_f32x16, b_z_f32x16, cov_xz_f32x16);
        cov_yx_f32x16 = _mm512_fmadd_ps(a_y_f32x16, b_x_f32x16, cov_yx_f32x16);
        cov_yy_f32x16 = _mm512_fmadd_ps(a_y_f32x16, b_y_f32x16, cov_yy_f32x16);
        cov_yz_f32x16 = _mm512_fmadd_ps(a_y_f32x16, b_z_f32x16, cov_yz_f32x16);
        cov_zx_f32x16 = _mm512_fmadd_ps(a_z_f32x16, b_x_f32x16, cov_zx_f32x16);
        cov_zy_f32x16 = _mm512_fmadd_ps(a_z_f32x16, b_y_f32x16, cov_zy_f32x16);
        cov_zz_f32x16 = _mm512_fmadd_ps(a_z_f32x16, b_z_f32x16, cov_zz_f32x16);
        variance_a_f32x16 = _mm512_fmadd_ps(a_x_f32x16, a_x_f32x16, variance_a_f32x16);
        variance_a_f32x16 = _mm512_fmadd_ps(a_y_f32x16, a_y_f32x16, variance_a_f32x16);
        variance_a_f32x16 = _mm512_fmadd_ps(a_z_f32x16, a_z_f32x16, variance_a_f32x16);
    }

    // Reduce centroids
    nk_f32_t inv_n = 1.0f / (nk_f32_t)n;
    nk_f32_t sum_a_x = _mm512_reduce_add_ps(sum_a_x_f32x16), sum_a_y = _mm512_reduce_add_ps(sum_a_y_f32x16);
    nk_f32_t sum_a_z = _mm512_reduce_add_ps(sum_a_z_f32x16);
    nk_f32_t sum_b_x = _mm512_reduce_add_ps(sum_b_x_f32x16), sum_b_y = _mm512_reduce_add_ps(sum_b_y_f32x16);
    nk_f32_t sum_b_z = _mm512_reduce_add_ps(sum_b_z_f32x16);

    nk_f32_t centroid_a_x = sum_a_x * inv_n, centroid_a_y = sum_a_y * inv_n, centroid_a_z = sum_a_z * inv_n;
    nk_f32_t centroid_b_x = sum_b_x * inv_n, centroid_b_y = sum_b_y * inv_n, centroid_b_z = sum_b_z * inv_n;

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

    // Compute centered covariance and variance
    nk_f32_t variance_a_sum = _mm512_reduce_add_ps(variance_a_f32x16);
    nk_f32_t variance_a = variance_a_sum * inv_n -
                          (centroid_a_x * centroid_a_x + centroid_a_y * centroid_a_y + centroid_a_z * centroid_a_z);

    nk_f32_t cross_covariance[9];
    cross_covariance[0] = _mm512_reduce_add_ps(cov_xx_f32x16) - sum_a_x * sum_b_x * inv_n;
    cross_covariance[1] = _mm512_reduce_add_ps(cov_xy_f32x16) - sum_a_x * sum_b_y * inv_n;
    cross_covariance[2] = _mm512_reduce_add_ps(cov_xz_f32x16) - sum_a_x * sum_b_z * inv_n;
    cross_covariance[3] = _mm512_reduce_add_ps(cov_yx_f32x16) - sum_a_y * sum_b_x * inv_n;
    cross_covariance[4] = _mm512_reduce_add_ps(cov_yy_f32x16) - sum_a_y * sum_b_y * inv_n;
    cross_covariance[5] = _mm512_reduce_add_ps(cov_yz_f32x16) - sum_a_y * sum_b_z * inv_n;
    cross_covariance[6] = _mm512_reduce_add_ps(cov_zx_f32x16) - sum_a_z * sum_b_x * inv_n;
    cross_covariance[7] = _mm512_reduce_add_ps(cov_zy_f32x16) - sum_a_z * sum_b_y * inv_n;
    cross_covariance[8] = _mm512_reduce_add_ps(cov_zz_f32x16) - sum_a_z * sum_b_z * inv_n;

    // SVD
    nk_f32_t svd_u[9], svd_s[9], svd_v[9];
    nk_svd3x3_f32_(cross_covariance, svd_u, svd_s, svd_v);

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
    nk_f32_t det = nk_det3x3_f32_(r);
    nk_f32_t d3 = det < 0 ? -1.0f : 1.0f;
    nk_f32_t trace_ds = svd_s[0] + svd_s[4] + d3 * svd_s[8];
    nk_f32_t c = trace_ds / (n * variance_a);
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

    // Compute RMSD with scaling
    nk_f32_t sum_squared = nk_transformed_ssd_f32_skylake_(a, b, n, r, c, centroid_a_x, centroid_a_y, centroid_a_z,
                                                           centroid_b_x, centroid_b_y, centroid_b_z);
    *result = nk_sqrt_f32_haswell_(sum_squared * inv_n);
}

NK_PUBLIC void nk_umeyama_f64_skylake(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                                      nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result) {
    // Fused single-pass: centroids, covariance, and variance of A
    __m512i const gather_idx_i64x8 = _mm512_setr_epi64(0, 3, 6, 9, 12, 15, 18, 21);
    __m512d const zeros_f64x8 = _mm512_setzero_pd();

    __m512d sum_a_x_f64x8 = zeros_f64x8, sum_a_y_f64x8 = zeros_f64x8, sum_a_z_f64x8 = zeros_f64x8;
    __m512d sum_b_x_f64x8 = zeros_f64x8, sum_b_y_f64x8 = zeros_f64x8, sum_b_z_f64x8 = zeros_f64x8;
    __m512d cov_xx_f64x8 = zeros_f64x8, cov_xy_f64x8 = zeros_f64x8, cov_xz_f64x8 = zeros_f64x8;
    __m512d cov_yx_f64x8 = zeros_f64x8, cov_yy_f64x8 = zeros_f64x8, cov_yz_f64x8 = zeros_f64x8;
    __m512d cov_zx_f64x8 = zeros_f64x8, cov_zy_f64x8 = zeros_f64x8, cov_zz_f64x8 = zeros_f64x8;
    __m512d variance_a_f64x8 = zeros_f64x8;

    nk_size_t i = 0;
    __m512d a_x_f64x8, a_y_f64x8, a_z_f64x8, b_x_f64x8, b_y_f64x8, b_z_f64x8;

    for (; i + 8 <= n; i += 8) {
        nk_deinterleave_f64x8_skylake_(a + i * 3, &a_x_f64x8, &a_y_f64x8, &a_z_f64x8);
        nk_deinterleave_f64x8_skylake_(b + i * 3, &b_x_f64x8, &b_y_f64x8, &b_z_f64x8);

        sum_a_x_f64x8 = _mm512_add_pd(sum_a_x_f64x8, a_x_f64x8),
        sum_a_y_f64x8 = _mm512_add_pd(sum_a_y_f64x8, a_y_f64x8);
        sum_a_z_f64x8 = _mm512_add_pd(sum_a_z_f64x8, a_z_f64x8);
        sum_b_x_f64x8 = _mm512_add_pd(sum_b_x_f64x8, b_x_f64x8),
        sum_b_y_f64x8 = _mm512_add_pd(sum_b_y_f64x8, b_y_f64x8);
        sum_b_z_f64x8 = _mm512_add_pd(sum_b_z_f64x8, b_z_f64x8);

        cov_xx_f64x8 = _mm512_fmadd_pd(a_x_f64x8, b_x_f64x8, cov_xx_f64x8),
        cov_xy_f64x8 = _mm512_fmadd_pd(a_x_f64x8, b_y_f64x8, cov_xy_f64x8);
        cov_xz_f64x8 = _mm512_fmadd_pd(a_x_f64x8, b_z_f64x8, cov_xz_f64x8);
        cov_yx_f64x8 = _mm512_fmadd_pd(a_y_f64x8, b_x_f64x8, cov_yx_f64x8),
        cov_yy_f64x8 = _mm512_fmadd_pd(a_y_f64x8, b_y_f64x8, cov_yy_f64x8);
        cov_yz_f64x8 = _mm512_fmadd_pd(a_y_f64x8, b_z_f64x8, cov_yz_f64x8);
        cov_zx_f64x8 = _mm512_fmadd_pd(a_z_f64x8, b_x_f64x8, cov_zx_f64x8),
        cov_zy_f64x8 = _mm512_fmadd_pd(a_z_f64x8, b_y_f64x8, cov_zy_f64x8);
        cov_zz_f64x8 = _mm512_fmadd_pd(a_z_f64x8, b_z_f64x8, cov_zz_f64x8);
        variance_a_f64x8 = _mm512_fmadd_pd(a_x_f64x8, a_x_f64x8, variance_a_f64x8);
        variance_a_f64x8 = _mm512_fmadd_pd(a_y_f64x8, a_y_f64x8, variance_a_f64x8);
        variance_a_f64x8 = _mm512_fmadd_pd(a_z_f64x8, a_z_f64x8, variance_a_f64x8);
    }

    if (i < n) {
        nk_size_t tail = n - i;
        __mmask8 mask = (__mmask8)_bzhi_u32(0xFF, tail);
        nk_f64_t const *a_tail = a + i * 3;
        nk_f64_t const *b_tail = b + i * 3;

        a_x_f64x8 = _mm512_mask_i64gather_pd(zeros_f64x8, mask, gather_idx_i64x8, a_tail + 0, 8);
        a_y_f64x8 = _mm512_mask_i64gather_pd(zeros_f64x8, mask, gather_idx_i64x8, a_tail + 1, 8);
        a_z_f64x8 = _mm512_mask_i64gather_pd(zeros_f64x8, mask, gather_idx_i64x8, a_tail + 2, 8);
        b_x_f64x8 = _mm512_mask_i64gather_pd(zeros_f64x8, mask, gather_idx_i64x8, b_tail + 0, 8);
        b_y_f64x8 = _mm512_mask_i64gather_pd(zeros_f64x8, mask, gather_idx_i64x8, b_tail + 1, 8);
        b_z_f64x8 = _mm512_mask_i64gather_pd(zeros_f64x8, mask, gather_idx_i64x8, b_tail + 2, 8);

        sum_a_x_f64x8 = _mm512_add_pd(sum_a_x_f64x8, a_x_f64x8),
        sum_a_y_f64x8 = _mm512_add_pd(sum_a_y_f64x8, a_y_f64x8);
        sum_a_z_f64x8 = _mm512_add_pd(sum_a_z_f64x8, a_z_f64x8);
        sum_b_x_f64x8 = _mm512_add_pd(sum_b_x_f64x8, b_x_f64x8),
        sum_b_y_f64x8 = _mm512_add_pd(sum_b_y_f64x8, b_y_f64x8);
        sum_b_z_f64x8 = _mm512_add_pd(sum_b_z_f64x8, b_z_f64x8);

        cov_xx_f64x8 = _mm512_fmadd_pd(a_x_f64x8, b_x_f64x8, cov_xx_f64x8),
        cov_xy_f64x8 = _mm512_fmadd_pd(a_x_f64x8, b_y_f64x8, cov_xy_f64x8);
        cov_xz_f64x8 = _mm512_fmadd_pd(a_x_f64x8, b_z_f64x8, cov_xz_f64x8);
        cov_yx_f64x8 = _mm512_fmadd_pd(a_y_f64x8, b_x_f64x8, cov_yx_f64x8),
        cov_yy_f64x8 = _mm512_fmadd_pd(a_y_f64x8, b_y_f64x8, cov_yy_f64x8);
        cov_yz_f64x8 = _mm512_fmadd_pd(a_y_f64x8, b_z_f64x8, cov_yz_f64x8);
        cov_zx_f64x8 = _mm512_fmadd_pd(a_z_f64x8, b_x_f64x8, cov_zx_f64x8),
        cov_zy_f64x8 = _mm512_fmadd_pd(a_z_f64x8, b_y_f64x8, cov_zy_f64x8);
        cov_zz_f64x8 = _mm512_fmadd_pd(a_z_f64x8, b_z_f64x8, cov_zz_f64x8);
        variance_a_f64x8 = _mm512_fmadd_pd(a_x_f64x8, a_x_f64x8, variance_a_f64x8);
        variance_a_f64x8 = _mm512_fmadd_pd(a_y_f64x8, a_y_f64x8, variance_a_f64x8);
        variance_a_f64x8 = _mm512_fmadd_pd(a_z_f64x8, a_z_f64x8, variance_a_f64x8);
    }

    // Reduce centroids
    nk_f64_t inv_n = 1.0 / (nk_f64_t)n;
    nk_f64_t sum_a_x = _mm512_reduce_add_pd(sum_a_x_f64x8), sum_a_y = _mm512_reduce_add_pd(sum_a_y_f64x8);
    nk_f64_t sum_a_z = _mm512_reduce_add_pd(sum_a_z_f64x8);
    nk_f64_t sum_b_x = _mm512_reduce_add_pd(sum_b_x_f64x8), sum_b_y = _mm512_reduce_add_pd(sum_b_y_f64x8);
    nk_f64_t sum_b_z = _mm512_reduce_add_pd(sum_b_z_f64x8);

    nk_f64_t centroid_a_x = sum_a_x * inv_n, centroid_a_y = sum_a_y * inv_n, centroid_a_z = sum_a_z * inv_n;
    nk_f64_t centroid_b_x = sum_b_x * inv_n, centroid_b_y = sum_b_y * inv_n, centroid_b_z = sum_b_z * inv_n;

    if (a_centroid) a_centroid[0] = centroid_a_x, a_centroid[1] = centroid_a_y, a_centroid[2] = centroid_a_z;
    if (b_centroid) b_centroid[0] = centroid_b_x, b_centroid[1] = centroid_b_y, b_centroid[2] = centroid_b_z;

    // Compute centered covariance and variance
    nk_f64_t variance_a_sum = _mm512_reduce_add_pd(variance_a_f64x8);
    nk_f64_t variance_a = variance_a_sum * inv_n -
                          (centroid_a_x * centroid_a_x + centroid_a_y * centroid_a_y + centroid_a_z * centroid_a_z);

    // Compute centered covariance matrix: H_ij = sum(a_i*b_j) - sum_a_i * sum_b_j / n
    nk_f64_t cross_covariance[9];
    cross_covariance[0] = _mm512_reduce_add_pd(cov_xx_f64x8) - sum_a_x * sum_b_x * inv_n;
    cross_covariance[1] = _mm512_reduce_add_pd(cov_xy_f64x8) - sum_a_x * sum_b_y * inv_n;
    cross_covariance[2] = _mm512_reduce_add_pd(cov_xz_f64x8) - sum_a_x * sum_b_z * inv_n;
    cross_covariance[3] = _mm512_reduce_add_pd(cov_yx_f64x8) - sum_a_y * sum_b_x * inv_n;
    cross_covariance[4] = _mm512_reduce_add_pd(cov_yy_f64x8) - sum_a_y * sum_b_y * inv_n;
    cross_covariance[5] = _mm512_reduce_add_pd(cov_yz_f64x8) - sum_a_y * sum_b_z * inv_n;
    cross_covariance[6] = _mm512_reduce_add_pd(cov_zx_f64x8) - sum_a_z * sum_b_x * inv_n;
    cross_covariance[7] = _mm512_reduce_add_pd(cov_zy_f64x8) - sum_a_z * sum_b_y * inv_n;
    cross_covariance[8] = _mm512_reduce_add_pd(cov_zz_f64x8) - sum_a_z * sum_b_z * inv_n;

    // SVD using f64 for full precision
    nk_f64_t svd_u[9], svd_s[3], svd_v[9];
    nk_svd3x3_f64_(cross_covariance, svd_u, svd_s, svd_v);

    // R = V * U^T
    nk_f64_t r[9];
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
    nk_f64_t det = nk_det3x3_f64_(r);
    nk_f64_t d3 = det < 0 ? -1.0 : 1.0;
    nk_f64_t trace_ds = svd_s[0] + svd_s[1] + d3 * svd_s[2];
    nk_f64_t c = trace_ds / (n * variance_a);
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

    // Compute RMSD with scaling
    nk_f64_t sum_squared = nk_transformed_ssd_f64_skylake_(a, b, n, r, c, centroid_a_x, centroid_a_y, centroid_a_z,
                                                           centroid_b_x, centroid_b_y, centroid_b_z);
    *result = nk_sqrt_f64_haswell_(sum_squared * inv_n);
}

#if defined(__cplusplus)
} // extern "C"
#endif

#pragma clang attribute pop
#pragma GCC pop_options
#endif // NK_TARGET_SKYLAKE
#endif // NK_TARGET_X86_

#endif // NK_MESH_SKYLAKE_H
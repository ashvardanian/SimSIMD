/**
 *  @brief SIMD-accelerated Dot Products for Real and Complex Numbers optimized for Intel Haswell CPUs.
 *  @file include/numkong/mesh/haswell.h
 *  @sa include/numkong/mesh.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 *
 *  @section haswell_mesh_instructions Key AVX2 Mesh Instructions
 *
 *      Intrinsic                   Instruction                     Latency     Throughput  Ports
 *      _mm256_fmadd_ps             VFMADD (YMM, YMM, YMM)          5cy         0.5/cy      p01
 *      _mm256_hadd_ps              VHADDPS (YMM, YMM, YMM)         7cy         0.5/cy      p01+p5
 *      _mm256_permute2f128_ps      VPERM2F128 (YMM, YMM, YMM, I8)  3cy         1/cy        p5
 *      _mm256_extractf128_ps       VEXTRACTF128 (XMM, YMM, I8)     3cy         1/cy        p5
 *      _mm256_i32gather_ps         VGATHERDPS (YMM, M, YMM, YMM)   12cy        5/cy        p0+p23
 *
 *  Point cloud operations (centroid, covariance, Kabsch alignment) use gather instructions for
 *  stride-3 xyz deinterleaving. Multiple FMA accumulators hide the 5-cycle FMA latency. VHADDPS
 *  interleaves results across lanes, requiring additional shuffles for final scalar reduction.
 */
#ifndef NK_MESH_HASWELL_H
#define NK_MESH_HASWELL_H

#if NK_TARGET_X86_
#if NK_TARGET_HASWELL
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,f16c,fma,bmi,bmi2"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "f16c", "fma", "bmi", "bmi2")
#endif

#include "numkong/types.h"
#include "numkong/reduce/haswell.h"  // nk_reduce_add_f32x8_haswell_, nk_reduce_add_f64x4_haswell_
#include "numkong/spatial/haswell.h" // nk_sqrt_f32_haswell_, nk_sqrt_f64_haswell_

#if defined(__cplusplus)
extern "C" {
#endif

/*  Deinterleave 24 floats (8 xyz triplets) into separate x, y, z vectors.
 *  Uses AVX2 gather instructions for clean stride-3 access.
 *
 *  Input: 24 contiguous floats [x0,y0,z0, x1,y1,z1, ..., x7,y7,z7]
 *  Output: x[8], y[8], z[8] vectors
 */
NK_INTERNAL void nk_deinterleave_f32x8_haswell_(nk_f32_t const *ptr, __m256 *x_out, __m256 *y_out, __m256 *z_out) {
    // Gather indices: 0, 3, 6, 9, 12, 15, 18, 21 (stride 3)
    __m256i idx = _mm256_setr_epi32(0, 3, 6, 9, 12, 15, 18, 21);
    *x_out = _mm256_i32gather_ps(ptr + 0, idx, 4);
    *y_out = _mm256_i32gather_ps(ptr + 1, idx, 4);
    *z_out = _mm256_i32gather_ps(ptr + 2, idx, 4);
}

/*  Deinterleave 12 f64 values (4 xyz triplets) into separate x, y, z vectors.
 *  Uses scalar extraction for simplicity as AVX2 lacks efficient stride-3 gather for f64.
 *
 *  Input: 12 contiguous f64 [x0,y0,z0, x1,y1,z1, x2,y2,z2, x3,y3,z3]
 *  Output: x[4], y[4], z[4] vectors
 */
NK_INTERNAL void nk_deinterleave_f64x4_haswell_(nk_f64_t const *ptr, __m256d *x_out, __m256d *y_out, __m256d *z_out) {
    nk_f64_t x0 = ptr[0], x1 = ptr[3], x2 = ptr[6], x3 = ptr[9];
    nk_f64_t y0 = ptr[1], y1 = ptr[4], y2 = ptr[7], y3 = ptr[10];
    nk_f64_t z0 = ptr[2], z1 = ptr[5], z2 = ptr[8], z3 = ptr[11];

    *x_out = _mm256_setr_pd(x0, x1, x2, x3);
    *y_out = _mm256_setr_pd(y0, y1, y2, y3);
    *z_out = _mm256_setr_pd(z0, z1, z2, z3);
}

/* Horizontal reduction helpers moved to reduce.h:
 * - nk_reduce_add_f32x8_haswell_
 * - nk_reduce_add_f64x4_haswell_
 */

/*  Compute sum of squared distances after applying rotation (and optional scale).
 *  Used by kabsch (scale=1.0) and umeyama (scale=computed_scale).
 *  Returns sum_squared, caller computes sqrt(sum_squared / n).
 */
NK_INTERNAL nk_f32_t nk_transformed_ssd_f32_haswell_(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n,
                                                     nk_f32_t const *r, nk_f32_t scale, nk_f32_t centroid_a_x,
                                                     nk_f32_t centroid_a_y, nk_f32_t centroid_a_z,
                                                     nk_f32_t centroid_b_x, nk_f32_t centroid_b_y,
                                                     nk_f32_t centroid_b_z) {
    // Broadcast scaled rotation matrix elements
    __m256 sr0_f32x8 = _mm256_set1_ps(scale * r[0]), sr1_f32x8 = _mm256_set1_ps(scale * r[1]),
           sr2_f32x8 = _mm256_set1_ps(scale * r[2]);
    __m256 sr3_f32x8 = _mm256_set1_ps(scale * r[3]), sr4_f32x8 = _mm256_set1_ps(scale * r[4]),
           sr5_f32x8 = _mm256_set1_ps(scale * r[5]);
    __m256 sr6_f32x8 = _mm256_set1_ps(scale * r[6]), sr7_f32x8 = _mm256_set1_ps(scale * r[7]),
           sr8_f32x8 = _mm256_set1_ps(scale * r[8]);

    // Broadcast centroids
    __m256 centroid_a_x_f32x8 = _mm256_set1_ps(centroid_a_x);
    __m256 centroid_a_y_f32x8 = _mm256_set1_ps(centroid_a_y);
    __m256 centroid_a_z_f32x8 = _mm256_set1_ps(centroid_a_z);
    __m256 centroid_b_x_f32x8 = _mm256_set1_ps(centroid_b_x);
    __m256 centroid_b_y_f32x8 = _mm256_set1_ps(centroid_b_y);
    __m256 centroid_b_z_f32x8 = _mm256_set1_ps(centroid_b_z);

    __m256 sum_squared_f32x8 = _mm256_setzero_ps();
    __m256 a_x_f32x8, a_y_f32x8, a_z_f32x8, b_x_f32x8, b_y_f32x8, b_z_f32x8;
    nk_size_t j = 0;

    for (; j + 8 <= n; j += 8) {
        nk_deinterleave_f32x8_haswell_(a + j * 3, &a_x_f32x8, &a_y_f32x8, &a_z_f32x8);
        nk_deinterleave_f32x8_haswell_(b + j * 3, &b_x_f32x8, &b_y_f32x8, &b_z_f32x8);

        // Center points
        __m256 pa_x_f32x8 = _mm256_sub_ps(a_x_f32x8, centroid_a_x_f32x8);
        __m256 pa_y_f32x8 = _mm256_sub_ps(a_y_f32x8, centroid_a_y_f32x8);
        __m256 pa_z_f32x8 = _mm256_sub_ps(a_z_f32x8, centroid_a_z_f32x8);
        __m256 pb_x_f32x8 = _mm256_sub_ps(b_x_f32x8, centroid_b_x_f32x8);
        __m256 pb_y_f32x8 = _mm256_sub_ps(b_y_f32x8, centroid_b_y_f32x8);
        __m256 pb_z_f32x8 = _mm256_sub_ps(b_z_f32x8, centroid_b_z_f32x8);

        // Rotate and scale: ra = scale * R * pa
        __m256 ra_x_f32x8 = _mm256_fmadd_ps(
            sr2_f32x8, pa_z_f32x8, _mm256_fmadd_ps(sr1_f32x8, pa_y_f32x8, _mm256_mul_ps(sr0_f32x8, pa_x_f32x8)));
        __m256 ra_y_f32x8 = _mm256_fmadd_ps(
            sr5_f32x8, pa_z_f32x8, _mm256_fmadd_ps(sr4_f32x8, pa_y_f32x8, _mm256_mul_ps(sr3_f32x8, pa_x_f32x8)));
        __m256 ra_z_f32x8 = _mm256_fmadd_ps(
            sr8_f32x8, pa_z_f32x8, _mm256_fmadd_ps(sr7_f32x8, pa_y_f32x8, _mm256_mul_ps(sr6_f32x8, pa_x_f32x8)));

        // Delta and accumulate
        __m256 delta_x_f32x8 = _mm256_sub_ps(ra_x_f32x8, pb_x_f32x8);
        __m256 delta_y_f32x8 = _mm256_sub_ps(ra_y_f32x8, pb_y_f32x8);
        __m256 delta_z_f32x8 = _mm256_sub_ps(ra_z_f32x8, pb_z_f32x8);

        sum_squared_f32x8 = _mm256_fmadd_ps(delta_x_f32x8, delta_x_f32x8, sum_squared_f32x8);
        sum_squared_f32x8 = _mm256_fmadd_ps(delta_y_f32x8, delta_y_f32x8, sum_squared_f32x8);
        sum_squared_f32x8 = _mm256_fmadd_ps(delta_z_f32x8, delta_z_f32x8, sum_squared_f32x8);
    }

    nk_f32_t sum_squared = nk_reduce_add_f32x8_haswell_(sum_squared_f32x8);

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

/*  Compute sum of squared distances for f64 after applying rotation (and optional scale).
 *  Rotation matrix, scale and data are all f64 for full precision.
 */
NK_INTERNAL nk_f64_t nk_transformed_ssd_f64_haswell_(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n,
                                                     nk_f64_t const *r, nk_f64_t scale, nk_f64_t centroid_a_x,
                                                     nk_f64_t centroid_a_y, nk_f64_t centroid_a_z,
                                                     nk_f64_t centroid_b_x, nk_f64_t centroid_b_y,
                                                     nk_f64_t centroid_b_z) {
    // Broadcast scaled rotation matrix elements
    __m256d sr0_f64x4 = _mm256_set1_pd(scale * r[0]), sr1_f64x4 = _mm256_set1_pd(scale * r[1]),
            sr2_f64x4 = _mm256_set1_pd(scale * r[2]);
    __m256d sr3_f64x4 = _mm256_set1_pd(scale * r[3]), sr4_f64x4 = _mm256_set1_pd(scale * r[4]),
            sr5_f64x4 = _mm256_set1_pd(scale * r[5]);
    __m256d sr6_f64x4 = _mm256_set1_pd(scale * r[6]), sr7_f64x4 = _mm256_set1_pd(scale * r[7]),
            sr8_f64x4 = _mm256_set1_pd(scale * r[8]);

    // Broadcast centroids
    __m256d centroid_a_x_f64x4 = _mm256_set1_pd(centroid_a_x);
    __m256d centroid_a_y_f64x4 = _mm256_set1_pd(centroid_a_y);
    __m256d centroid_a_z_f64x4 = _mm256_set1_pd(centroid_a_z);
    __m256d centroid_b_x_f64x4 = _mm256_set1_pd(centroid_b_x);
    __m256d centroid_b_y_f64x4 = _mm256_set1_pd(centroid_b_y);
    __m256d centroid_b_z_f64x4 = _mm256_set1_pd(centroid_b_z);

    __m256d sum_squared_f64x4 = _mm256_setzero_pd();
    __m256d a_x_f64x4, a_y_f64x4, a_z_f64x4, b_x_f64x4, b_y_f64x4, b_z_f64x4;
    nk_size_t j = 0;

    for (; j + 4 <= n; j += 4) {
        nk_deinterleave_f64x4_haswell_(a + j * 3, &a_x_f64x4, &a_y_f64x4, &a_z_f64x4);
        nk_deinterleave_f64x4_haswell_(b + j * 3, &b_x_f64x4, &b_y_f64x4, &b_z_f64x4);

        // Center points
        __m256d pa_x_f64x4 = _mm256_sub_pd(a_x_f64x4, centroid_a_x_f64x4);
        __m256d pa_y_f64x4 = _mm256_sub_pd(a_y_f64x4, centroid_a_y_f64x4);
        __m256d pa_z_f64x4 = _mm256_sub_pd(a_z_f64x4, centroid_a_z_f64x4);
        __m256d pb_x_f64x4 = _mm256_sub_pd(b_x_f64x4, centroid_b_x_f64x4);
        __m256d pb_y_f64x4 = _mm256_sub_pd(b_y_f64x4, centroid_b_y_f64x4);
        __m256d pb_z_f64x4 = _mm256_sub_pd(b_z_f64x4, centroid_b_z_f64x4);

        // Rotate and scale: ra = scale * R * pa
        __m256d ra_x_f64x4 = _mm256_fmadd_pd(
            sr2_f64x4, pa_z_f64x4, _mm256_fmadd_pd(sr1_f64x4, pa_y_f64x4, _mm256_mul_pd(sr0_f64x4, pa_x_f64x4)));
        __m256d ra_y_f64x4 = _mm256_fmadd_pd(
            sr5_f64x4, pa_z_f64x4, _mm256_fmadd_pd(sr4_f64x4, pa_y_f64x4, _mm256_mul_pd(sr3_f64x4, pa_x_f64x4)));
        __m256d ra_z_f64x4 = _mm256_fmadd_pd(
            sr8_f64x4, pa_z_f64x4, _mm256_fmadd_pd(sr7_f64x4, pa_y_f64x4, _mm256_mul_pd(sr6_f64x4, pa_x_f64x4)));

        // Delta and accumulate
        __m256d delta_x_f64x4 = _mm256_sub_pd(ra_x_f64x4, pb_x_f64x4);
        __m256d delta_y_f64x4 = _mm256_sub_pd(ra_y_f64x4, pb_y_f64x4);
        __m256d delta_z_f64x4 = _mm256_sub_pd(ra_z_f64x4, pb_z_f64x4);

        sum_squared_f64x4 = _mm256_fmadd_pd(delta_x_f64x4, delta_x_f64x4, sum_squared_f64x4);
        sum_squared_f64x4 = _mm256_fmadd_pd(delta_y_f64x4, delta_y_f64x4, sum_squared_f64x4);
        sum_squared_f64x4 = _mm256_fmadd_pd(delta_z_f64x4, delta_z_f64x4, sum_squared_f64x4);
    }

    nk_f64_t sum_squared = nk_reduce_add_f64x4_haswell_(sum_squared_f64x4);

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
        nk_deinterleave_f32x8_haswell_(a + i * 3, &a_x_f32x8, &a_y_f32x8, &a_z_f32x8);
        nk_deinterleave_f32x8_haswell_(b + i * 3, &b_x_f32x8, &b_y_f32x8, &b_z_f32x8);

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
        nk_deinterleave_f32x8_haswell_(a + (i + 8) * 3, &a_x1_f32x8, &a_y1_f32x8, &a_z1_f32x8);
        nk_deinterleave_f32x8_haswell_(b + (i + 8) * 3, &b_x1_f32x8, &b_y1_f32x8, &b_z1_f32x8);

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
        nk_deinterleave_f32x8_haswell_(a + i * 3, &a_x_f32x8, &a_y_f32x8, &a_z_f32x8);
        nk_deinterleave_f32x8_haswell_(b + i * 3, &b_x_f32x8, &b_y_f32x8, &b_z_f32x8);

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
    nk_f32_t total_ax = nk_reduce_add_f32x8_haswell_(sum_a_x_f32x8);
    nk_f32_t total_ay = nk_reduce_add_f32x8_haswell_(sum_a_y_f32x8);
    nk_f32_t total_az = nk_reduce_add_f32x8_haswell_(sum_a_z_f32x8);
    nk_f32_t total_bx = nk_reduce_add_f32x8_haswell_(sum_b_x_f32x8);
    nk_f32_t total_by = nk_reduce_add_f32x8_haswell_(sum_b_y_f32x8);
    nk_f32_t total_bz = nk_reduce_add_f32x8_haswell_(sum_b_z_f32x8);
    nk_f32_t total_sq_x = nk_reduce_add_f32x8_haswell_(sum_squared_x_f32x8);
    nk_f32_t total_sq_y = nk_reduce_add_f32x8_haswell_(sum_squared_y_f32x8);
    nk_f32_t total_sq_z = nk_reduce_add_f32x8_haswell_(sum_squared_z_f32x8);

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

    *result = nk_sqrt_f32_haswell_(sum_squared * inv_n - mean_diff_sq);
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
        nk_deinterleave_f64x4_haswell_(a + i * 3, &a_x_f64x4, &a_y_f64x4, &a_z_f64x4);
        nk_deinterleave_f64x4_haswell_(b + i * 3, &b_x_f64x4, &b_y_f64x4, &b_z_f64x4);

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
        nk_deinterleave_f64x4_haswell_(a + (i + 4) * 3, &a_x1_f64x4, &a_y1_f64x4, &a_z1_f64x4);
        nk_deinterleave_f64x4_haswell_(b + (i + 4) * 3, &b_x1_f64x4, &b_y1_f64x4, &b_z1_f64x4);

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
        nk_deinterleave_f64x4_haswell_(a + i * 3, &a_x_f64x4, &a_y_f64x4, &a_z_f64x4);
        nk_deinterleave_f64x4_haswell_(b + i * 3, &b_x_f64x4, &b_y_f64x4, &b_z_f64x4);

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
    nk_f64_t total_ax = nk_reduce_add_f64x4_haswell_(sum_a_x_f64x4);
    nk_f64_t total_ay = nk_reduce_add_f64x4_haswell_(sum_a_y_f64x4);
    nk_f64_t total_az = nk_reduce_add_f64x4_haswell_(sum_a_z_f64x4);
    nk_f64_t total_bx = nk_reduce_add_f64x4_haswell_(sum_b_x_f64x4);
    nk_f64_t total_by = nk_reduce_add_f64x4_haswell_(sum_b_y_f64x4);
    nk_f64_t total_bz = nk_reduce_add_f64x4_haswell_(sum_b_z_f64x4);
    nk_f64_t total_sq_x = nk_reduce_add_f64x4_haswell_(sum_squared_x_f64x4);
    nk_f64_t total_sq_y = nk_reduce_add_f64x4_haswell_(sum_squared_y_f64x4);
    nk_f64_t total_sq_z = nk_reduce_add_f64x4_haswell_(sum_squared_z_f64x4);

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

    *result = nk_sqrt_f64_haswell_(sum_squared * inv_n - mean_diff_sq);
}

NK_PUBLIC void nk_kabsch_f32_haswell(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                     nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result) {
    // Optimized fused single-pass implementation using AVX2 with f32 numerics.
    // Computes centroids and covariance matrix in one pass.
    __m256 const zeros_f32x8 = _mm256_setzero_ps();

    // Accumulators for centroids (f32)
    __m256 sum_a_x_f32x8 = zeros_f32x8, sum_a_y_f32x8 = zeros_f32x8, sum_a_z_f32x8 = zeros_f32x8;
    __m256 sum_b_x_f32x8 = zeros_f32x8, sum_b_y_f32x8 = zeros_f32x8, sum_b_z_f32x8 = zeros_f32x8;

    // Accumulators for covariance matrix (sum of outer products)
    __m256 cov_xx_f32x8 = zeros_f32x8, cov_xy_f32x8 = zeros_f32x8, cov_xz_f32x8 = zeros_f32x8;
    __m256 cov_yx_f32x8 = zeros_f32x8, cov_yy_f32x8 = zeros_f32x8, cov_yz_f32x8 = zeros_f32x8;
    __m256 cov_zx_f32x8 = zeros_f32x8, cov_zy_f32x8 = zeros_f32x8, cov_zz_f32x8 = zeros_f32x8;

    nk_size_t i = 0;
    __m256 a_x_f32x8, a_y_f32x8, a_z_f32x8, b_x_f32x8, b_y_f32x8, b_z_f32x8;

    // Fused single-pass: accumulate sums and outer products together
    for (; i + 8 <= n; i += 8) {
        nk_deinterleave_f32x8_haswell_(a + i * 3, &a_x_f32x8, &a_y_f32x8, &a_z_f32x8);
        nk_deinterleave_f32x8_haswell_(b + i * 3, &b_x_f32x8, &b_y_f32x8, &b_z_f32x8);

        // Accumulate centroids directly in f32
        sum_a_x_f32x8 = _mm256_add_ps(sum_a_x_f32x8, a_x_f32x8);
        sum_a_y_f32x8 = _mm256_add_ps(sum_a_y_f32x8, a_y_f32x8);
        sum_a_z_f32x8 = _mm256_add_ps(sum_a_z_f32x8, a_z_f32x8);
        sum_b_x_f32x8 = _mm256_add_ps(sum_b_x_f32x8, b_x_f32x8);
        sum_b_y_f32x8 = _mm256_add_ps(sum_b_y_f32x8, b_y_f32x8);
        sum_b_z_f32x8 = _mm256_add_ps(sum_b_z_f32x8, b_z_f32x8);

        // Accumulate outer products (raw, not centered) in f32
        cov_xx_f32x8 = _mm256_fmadd_ps(a_x_f32x8, b_x_f32x8, cov_xx_f32x8);
        cov_xy_f32x8 = _mm256_fmadd_ps(a_x_f32x8, b_y_f32x8, cov_xy_f32x8);
        cov_xz_f32x8 = _mm256_fmadd_ps(a_x_f32x8, b_z_f32x8, cov_xz_f32x8);
        cov_yx_f32x8 = _mm256_fmadd_ps(a_y_f32x8, b_x_f32x8, cov_yx_f32x8);
        cov_yy_f32x8 = _mm256_fmadd_ps(a_y_f32x8, b_y_f32x8, cov_yy_f32x8);
        cov_yz_f32x8 = _mm256_fmadd_ps(a_y_f32x8, b_z_f32x8, cov_yz_f32x8);
        cov_zx_f32x8 = _mm256_fmadd_ps(a_z_f32x8, b_x_f32x8, cov_zx_f32x8);
        cov_zy_f32x8 = _mm256_fmadd_ps(a_z_f32x8, b_y_f32x8, cov_zy_f32x8);
        cov_zz_f32x8 = _mm256_fmadd_ps(a_z_f32x8, b_z_f32x8, cov_zz_f32x8);
    }

    // Reduce vector accumulators
    nk_f32_t sum_a_x = nk_reduce_add_f32x8_haswell_(sum_a_x_f32x8);
    nk_f32_t sum_a_y = nk_reduce_add_f32x8_haswell_(sum_a_y_f32x8);
    nk_f32_t sum_a_z = nk_reduce_add_f32x8_haswell_(sum_a_z_f32x8);
    nk_f32_t sum_b_x = nk_reduce_add_f32x8_haswell_(sum_b_x_f32x8);
    nk_f32_t sum_b_y = nk_reduce_add_f32x8_haswell_(sum_b_y_f32x8);
    nk_f32_t sum_b_z = nk_reduce_add_f32x8_haswell_(sum_b_z_f32x8);

    nk_f32_t h00 = nk_reduce_add_f32x8_haswell_(cov_xx_f32x8);
    nk_f32_t h01 = nk_reduce_add_f32x8_haswell_(cov_xy_f32x8);
    nk_f32_t h02 = nk_reduce_add_f32x8_haswell_(cov_xz_f32x8);
    nk_f32_t h10 = nk_reduce_add_f32x8_haswell_(cov_yx_f32x8);
    nk_f32_t h11 = nk_reduce_add_f32x8_haswell_(cov_yy_f32x8);
    nk_f32_t h12 = nk_reduce_add_f32x8_haswell_(cov_yz_f32x8);
    nk_f32_t h20 = nk_reduce_add_f32x8_haswell_(cov_zx_f32x8);
    nk_f32_t h21 = nk_reduce_add_f32x8_haswell_(cov_zy_f32x8);
    nk_f32_t h22 = nk_reduce_add_f32x8_haswell_(cov_zz_f32x8);

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

    // Compute SVD and optimal rotation (svd_s is 9-element diagonal matrix)
    nk_f32_t cross_covariance[9] = {h00, h01, h02, h10, h11, h12, h20, h21, h22};
    nk_f32_t svd_u[9], svd_s[9], svd_v[9];
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
    if (scale) *scale = 1.0;

    // Compute RMSD after optimal rotation
    nk_f32_t sum_squared = nk_transformed_ssd_f32_haswell_(a, b, n, r, 1.0f, centroid_a_x, centroid_a_y, centroid_a_z,
                                                           centroid_b_x, centroid_b_y, centroid_b_z);
    *result = nk_sqrt_f32_haswell_(sum_squared * inv_n);
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
        nk_deinterleave_f64x4_haswell_(a + i * 3, &a_x_f64x4, &a_y_f64x4, &a_z_f64x4);
        nk_deinterleave_f64x4_haswell_(b + i * 3, &b_x_f64x4, &b_y_f64x4, &b_z_f64x4);

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
    nk_f64_t sum_a_x = nk_reduce_add_f64x4_haswell_(sum_a_x_f64x4);
    nk_f64_t sum_a_y = nk_reduce_add_f64x4_haswell_(sum_a_y_f64x4);
    nk_f64_t sum_a_z = nk_reduce_add_f64x4_haswell_(sum_a_z_f64x4);
    nk_f64_t sum_b_x = nk_reduce_add_f64x4_haswell_(sum_b_x_f64x4);
    nk_f64_t sum_b_y = nk_reduce_add_f64x4_haswell_(sum_b_y_f64x4);
    nk_f64_t sum_b_z = nk_reduce_add_f64x4_haswell_(sum_b_z_f64x4);

    nk_f64_t H00 = nk_reduce_add_f64x4_haswell_(cov_xx_f64x4);
    nk_f64_t H01 = nk_reduce_add_f64x4_haswell_(cov_xy_f64x4);
    nk_f64_t H02 = nk_reduce_add_f64x4_haswell_(cov_xz_f64x4);
    nk_f64_t H10 = nk_reduce_add_f64x4_haswell_(cov_yx_f64x4);
    nk_f64_t H11 = nk_reduce_add_f64x4_haswell_(cov_yy_f64x4);
    nk_f64_t H12 = nk_reduce_add_f64x4_haswell_(cov_yz_f64x4);
    nk_f64_t H20 = nk_reduce_add_f64x4_haswell_(cov_zx_f64x4);
    nk_f64_t H21 = nk_reduce_add_f64x4_haswell_(cov_zy_f64x4);
    nk_f64_t H22 = nk_reduce_add_f64x4_haswell_(cov_zz_f64x4);

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

    // Apply centering correction: H_centered = H - n * centroid_a * centroid_bᵀ
    H00 -= n * centroid_a_x * centroid_b_x;
    H01 -= n * centroid_a_x * centroid_b_y;
    H02 -= n * centroid_a_x * centroid_b_z;
    H10 -= n * centroid_a_y * centroid_b_x;
    H11 -= n * centroid_a_y * centroid_b_y;
    H12 -= n * centroid_a_y * centroid_b_z;
    H20 -= n * centroid_a_z * centroid_b_x;
    H21 -= n * centroid_a_z * centroid_b_y;
    H22 -= n * centroid_a_z * centroid_b_z;

    // Compute SVD and optimal rotation using f64 precision (svd_s is 9-element diagonal matrix)
    nk_f64_t cross_covariance[9] = {H00, H01, H02, H10, H11, H12, H20, H21, H22};
    nk_f64_t svd_u[9], svd_s[9], svd_v[9];
    nk_svd3x3_f64_(cross_covariance, svd_u, svd_s, svd_v);

    // R = V * Uᵀ
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

    // Handle reflection: if det(R) < 0, negate third column of V and recompute R
    if (nk_det3x3_f64_(r) < 0) {
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
    nk_f64_t sum_squared = nk_transformed_ssd_f64_haswell_(a, b, n, r, 1.0, centroid_a_x, centroid_a_y, centroid_a_z,
                                                           centroid_b_x, centroid_b_y, centroid_b_z);
    *result = nk_sqrt_f64_haswell_(sum_squared * inv_n);
}

NK_PUBLIC void nk_umeyama_f32_haswell(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                      nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result) {
    // Fused single-pass: centroids, covariance, and variance of A using f32 numerics
    __m256 const zeros_f32x8 = _mm256_setzero_ps();

    __m256 sum_a_x_f32x8 = zeros_f32x8, sum_a_y_f32x8 = zeros_f32x8, sum_a_z_f32x8 = zeros_f32x8;
    __m256 sum_b_x_f32x8 = zeros_f32x8, sum_b_y_f32x8 = zeros_f32x8, sum_b_z_f32x8 = zeros_f32x8;
    __m256 cov_xx_f32x8 = zeros_f32x8, cov_xy_f32x8 = zeros_f32x8, cov_xz_f32x8 = zeros_f32x8;
    __m256 cov_yx_f32x8 = zeros_f32x8, cov_yy_f32x8 = zeros_f32x8, cov_yz_f32x8 = zeros_f32x8;
    __m256 cov_zx_f32x8 = zeros_f32x8, cov_zy_f32x8 = zeros_f32x8, cov_zz_f32x8 = zeros_f32x8;
    __m256 variance_a_f32x8 = zeros_f32x8;

    nk_size_t i = 0;
    __m256 a_x_f32x8, a_y_f32x8, a_z_f32x8, b_x_f32x8, b_y_f32x8, b_z_f32x8;

    for (; i + 8 <= n; i += 8) {
        nk_deinterleave_f32x8_haswell_(a + i * 3, &a_x_f32x8, &a_y_f32x8, &a_z_f32x8);
        nk_deinterleave_f32x8_haswell_(b + i * 3, &b_x_f32x8, &b_y_f32x8, &b_z_f32x8);

        // Accumulate centroids directly in f32
        sum_a_x_f32x8 = _mm256_add_ps(sum_a_x_f32x8, a_x_f32x8);
        sum_a_y_f32x8 = _mm256_add_ps(sum_a_y_f32x8, a_y_f32x8);
        sum_a_z_f32x8 = _mm256_add_ps(sum_a_z_f32x8, a_z_f32x8);
        sum_b_x_f32x8 = _mm256_add_ps(sum_b_x_f32x8, b_x_f32x8);
        sum_b_y_f32x8 = _mm256_add_ps(sum_b_y_f32x8, b_y_f32x8);
        sum_b_z_f32x8 = _mm256_add_ps(sum_b_z_f32x8, b_z_f32x8);

        // Accumulate outer products in f32
        cov_xx_f32x8 = _mm256_fmadd_ps(a_x_f32x8, b_x_f32x8, cov_xx_f32x8);
        cov_xy_f32x8 = _mm256_fmadd_ps(a_x_f32x8, b_y_f32x8, cov_xy_f32x8);
        cov_xz_f32x8 = _mm256_fmadd_ps(a_x_f32x8, b_z_f32x8, cov_xz_f32x8);
        cov_yx_f32x8 = _mm256_fmadd_ps(a_y_f32x8, b_x_f32x8, cov_yx_f32x8);
        cov_yy_f32x8 = _mm256_fmadd_ps(a_y_f32x8, b_y_f32x8, cov_yy_f32x8);
        cov_yz_f32x8 = _mm256_fmadd_ps(a_y_f32x8, b_z_f32x8, cov_yz_f32x8);
        cov_zx_f32x8 = _mm256_fmadd_ps(a_z_f32x8, b_x_f32x8, cov_zx_f32x8);
        cov_zy_f32x8 = _mm256_fmadd_ps(a_z_f32x8, b_y_f32x8, cov_zy_f32x8);
        cov_zz_f32x8 = _mm256_fmadd_ps(a_z_f32x8, b_z_f32x8, cov_zz_f32x8);

        // Accumulate variance of A
        variance_a_f32x8 = _mm256_fmadd_ps(a_x_f32x8, a_x_f32x8, variance_a_f32x8);
        variance_a_f32x8 = _mm256_fmadd_ps(a_y_f32x8, a_y_f32x8, variance_a_f32x8);
        variance_a_f32x8 = _mm256_fmadd_ps(a_z_f32x8, a_z_f32x8, variance_a_f32x8);
    }

    // Reduce vector accumulators
    nk_f32_t sum_a_x = nk_reduce_add_f32x8_haswell_(sum_a_x_f32x8);
    nk_f32_t sum_a_y = nk_reduce_add_f32x8_haswell_(sum_a_y_f32x8);
    nk_f32_t sum_a_z = nk_reduce_add_f32x8_haswell_(sum_a_z_f32x8);
    nk_f32_t sum_b_x = nk_reduce_add_f32x8_haswell_(sum_b_x_f32x8);
    nk_f32_t sum_b_y = nk_reduce_add_f32x8_haswell_(sum_b_y_f32x8);
    nk_f32_t sum_b_z = nk_reduce_add_f32x8_haswell_(sum_b_z_f32x8);
    nk_f32_t h00 = nk_reduce_add_f32x8_haswell_(cov_xx_f32x8);
    nk_f32_t h01 = nk_reduce_add_f32x8_haswell_(cov_xy_f32x8);
    nk_f32_t h02 = nk_reduce_add_f32x8_haswell_(cov_xz_f32x8);
    nk_f32_t h10 = nk_reduce_add_f32x8_haswell_(cov_yx_f32x8);
    nk_f32_t h11 = nk_reduce_add_f32x8_haswell_(cov_yy_f32x8);
    nk_f32_t h12 = nk_reduce_add_f32x8_haswell_(cov_yz_f32x8);
    nk_f32_t h20 = nk_reduce_add_f32x8_haswell_(cov_zx_f32x8);
    nk_f32_t h21 = nk_reduce_add_f32x8_haswell_(cov_zy_f32x8);
    nk_f32_t h22 = nk_reduce_add_f32x8_haswell_(cov_zz_f32x8);
    nk_f32_t variance_a_sum = nk_reduce_add_f32x8_haswell_(variance_a_f32x8);

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
        variance_a_sum += ax * ax + ay * ay + az * az;
    }

    // Compute centroids
    nk_f32_t inv_n = 1.0f / (nk_f32_t)n;
    nk_f32_t centroid_a_x = sum_a_x * inv_n, centroid_a_y = sum_a_y * inv_n, centroid_a_z = sum_a_z * inv_n;
    nk_f32_t centroid_b_x = sum_b_x * inv_n, centroid_b_y = sum_b_y * inv_n, centroid_b_z = sum_b_z * inv_n;

    if (a_centroid) a_centroid[0] = centroid_a_x, a_centroid[1] = centroid_a_y, a_centroid[2] = centroid_a_z;
    if (b_centroid) b_centroid[0] = centroid_b_x, b_centroid[1] = centroid_b_y, b_centroid[2] = centroid_b_z;

    // Compute centered covariance and variance
    nk_f32_t variance_a = variance_a_sum * inv_n -
                          (centroid_a_x * centroid_a_x + centroid_a_y * centroid_a_y + centroid_a_z * centroid_a_z);

    // Apply centering correction to covariance matrix
    h00 -= n * centroid_a_x * centroid_b_x;
    h01 -= n * centroid_a_x * centroid_b_y;
    h02 -= n * centroid_a_x * centroid_b_z;
    h10 -= n * centroid_a_y * centroid_b_x;
    h11 -= n * centroid_a_y * centroid_b_y;
    h12 -= n * centroid_a_y * centroid_b_z;
    h20 -= n * centroid_a_z * centroid_b_x;
    h21 -= n * centroid_a_z * centroid_b_y;
    h22 -= n * centroid_a_z * centroid_b_z;

    nk_f32_t cross_covariance[9] = {h00, h01, h02, h10, h11, h12, h20, h21, h22};

    // SVD
    nk_f32_t svd_u[9], svd_s[9], svd_v[9];
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

    // Scale factor: c = trace(D × S) / (n × variance(a))
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
    nk_f32_t sum_squared = nk_transformed_ssd_f32_haswell_(a, b, n, r, c, centroid_a_x, centroid_a_y, centroid_a_z,
                                                           centroid_b_x, centroid_b_y, centroid_b_z);
    *result = nk_sqrt_f32_haswell_(sum_squared * inv_n);
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
        nk_deinterleave_f64x4_haswell_(a + i * 3, &a_x_f64x4, &a_y_f64x4, &a_z_f64x4);
        nk_deinterleave_f64x4_haswell_(b + i * 3, &b_x_f64x4, &b_y_f64x4, &b_z_f64x4);

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
    nk_f64_t sum_a_x = nk_reduce_add_f64x4_haswell_(sum_a_x_f64x4);
    nk_f64_t sum_a_y = nk_reduce_add_f64x4_haswell_(sum_a_y_f64x4);
    nk_f64_t sum_a_z = nk_reduce_add_f64x4_haswell_(sum_a_z_f64x4);
    nk_f64_t sum_b_x = nk_reduce_add_f64x4_haswell_(sum_b_x_f64x4);
    nk_f64_t sum_b_y = nk_reduce_add_f64x4_haswell_(sum_b_y_f64x4);
    nk_f64_t sum_b_z = nk_reduce_add_f64x4_haswell_(sum_b_z_f64x4);
    nk_f64_t h00_s = nk_reduce_add_f64x4_haswell_(cov_xx_f64x4);
    nk_f64_t h01_s = nk_reduce_add_f64x4_haswell_(cov_xy_f64x4);
    nk_f64_t h02_s = nk_reduce_add_f64x4_haswell_(cov_xz_f64x4);
    nk_f64_t h10_s = nk_reduce_add_f64x4_haswell_(cov_yx_f64x4);
    nk_f64_t h11_s = nk_reduce_add_f64x4_haswell_(cov_yy_f64x4);
    nk_f64_t h12_s = nk_reduce_add_f64x4_haswell_(cov_yz_f64x4);
    nk_f64_t h20_s = nk_reduce_add_f64x4_haswell_(cov_zx_f64x4);
    nk_f64_t h21_s = nk_reduce_add_f64x4_haswell_(cov_zy_f64x4);
    nk_f64_t h22_s = nk_reduce_add_f64x4_haswell_(cov_zz_f64x4);
    nk_f64_t variance_a_sum = nk_reduce_add_f64x4_haswell_(variance_a_f64x4);

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

    nk_f64_t cross_covariance[9];
    cross_covariance[0] = h00_s - sum_a_x * sum_b_x * inv_n;
    cross_covariance[1] = h01_s - sum_a_x * sum_b_y * inv_n;
    cross_covariance[2] = h02_s - sum_a_x * sum_b_z * inv_n;
    cross_covariance[3] = h10_s - sum_a_y * sum_b_x * inv_n;
    cross_covariance[4] = h11_s - sum_a_y * sum_b_y * inv_n;
    cross_covariance[5] = h12_s - sum_a_y * sum_b_z * inv_n;
    cross_covariance[6] = h20_s - sum_a_z * sum_b_x * inv_n;
    cross_covariance[7] = h21_s - sum_a_z * sum_b_y * inv_n;
    cross_covariance[8] = h22_s - sum_a_z * sum_b_z * inv_n;

    // SVD using f64 for full precision (svd_s is 9-element diagonal matrix)
    nk_f64_t svd_u[9], svd_s[9], svd_v[9];
    nk_svd3x3_f64_(cross_covariance, svd_u, svd_s, svd_v);

    // R = V * Uᵀ
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

    // Scale factor: c = trace(D × S) / (n × variance(a))
    // svd_s diagonal: [0], [4], [8]
    nk_f64_t det = nk_det3x3_f64_(r);
    nk_f64_t d3 = det < 0 ? -1.0 : 1.0;
    nk_f64_t trace_ds = svd_s[0] + svd_s[4] + d3 * svd_s[8];
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
        for (int j = 0; j < 9; ++j) rotation[j] = r[j];
    }

    // Compute RMSD with scaling
    nk_f64_t sum_squared = nk_transformed_ssd_f64_haswell_(a, b, n, r, c, centroid_a_x, centroid_a_y, centroid_a_z,
                                                           centroid_b_x, centroid_b_y, centroid_b_z);
    *result = nk_sqrt_f64_haswell_(sum_squared * inv_n);
}

/*  Deinterleave 8 f16 xyz triplets (24 f16 values) and convert to 3 x __m256 f32.
 *  Uses scalar extraction for clean stride-3 access, then F16C conversion.
 *
 *  Input: 24 contiguous f16 [x0,y0,z0, x1,y1,z1, ..., x7,y7,z7]
 *  Output: x[8], y[8], z[8] vectors in f32
 */
NK_INTERNAL void nk_deinterleave_f16x8_to_f32x8_haswell_(nk_f16_t const *ptr, __m256 *x_out, __m256 *y_out,
                                                         __m256 *z_out) {
    // Extract x, y, z components with stride-3 access
    nk_b256_vec_t x_vec, y_vec, z_vec;
    x_vec.f16s[0] = ptr[0], x_vec.f16s[1] = ptr[3], x_vec.f16s[2] = ptr[6], x_vec.f16s[3] = ptr[9];
    x_vec.f16s[4] = ptr[12], x_vec.f16s[5] = ptr[15], x_vec.f16s[6] = ptr[18], x_vec.f16s[7] = ptr[21];
    y_vec.f16s[0] = ptr[1], y_vec.f16s[1] = ptr[4], y_vec.f16s[2] = ptr[7], y_vec.f16s[3] = ptr[10];
    y_vec.f16s[4] = ptr[13], y_vec.f16s[5] = ptr[16], y_vec.f16s[6] = ptr[19], y_vec.f16s[7] = ptr[22];
    z_vec.f16s[0] = ptr[2], z_vec.f16s[1] = ptr[5], z_vec.f16s[2] = ptr[8], z_vec.f16s[3] = ptr[11];
    z_vec.f16s[4] = ptr[14], z_vec.f16s[5] = ptr[17], z_vec.f16s[6] = ptr[20], z_vec.f16s[7] = ptr[23];
    // Convert f16 to f32 using F16C
    *x_out = _mm256_cvtph_ps(x_vec.xmms[0]);
    *y_out = _mm256_cvtph_ps(y_vec.xmms[0]);
    *z_out = _mm256_cvtph_ps(z_vec.xmms[0]);
}

/*  Deinterleave 8 bf16 xyz triplets (24 bf16 values) and convert to 3 x __m256 f32.
 *  Uses scalar extraction for clean stride-3 access, then bit-shift conversion.
 *
 *  Input: 24 contiguous bf16 [x0,y0,z0, x1,y1,z1, ..., x7,y7,z7]
 *  Output: x[8], y[8], z[8] vectors in f32
 */
NK_INTERNAL void nk_deinterleave_bf16x8_to_f32x8_haswell_(nk_bf16_t const *ptr, __m256 *x_out, __m256 *y_out,
                                                          __m256 *z_out) {
    // Extract x, y, z components with stride-3 access
    nk_b256_vec_t x_vec, y_vec, z_vec;
    x_vec.bf16s[0] = ptr[0], x_vec.bf16s[1] = ptr[3], x_vec.bf16s[2] = ptr[6], x_vec.bf16s[3] = ptr[9];
    x_vec.bf16s[4] = ptr[12], x_vec.bf16s[5] = ptr[15], x_vec.bf16s[6] = ptr[18], x_vec.bf16s[7] = ptr[21];
    y_vec.bf16s[0] = ptr[1], y_vec.bf16s[1] = ptr[4], y_vec.bf16s[2] = ptr[7], y_vec.bf16s[3] = ptr[10];
    y_vec.bf16s[4] = ptr[13], y_vec.bf16s[5] = ptr[16], y_vec.bf16s[6] = ptr[19], y_vec.bf16s[7] = ptr[22];
    z_vec.bf16s[0] = ptr[2], z_vec.bf16s[1] = ptr[5], z_vec.bf16s[2] = ptr[8], z_vec.bf16s[3] = ptr[11];
    z_vec.bf16s[4] = ptr[14], z_vec.bf16s[5] = ptr[17], z_vec.bf16s[6] = ptr[20], z_vec.bf16s[7] = ptr[23];
    // Convert bf16 to f32 by left-shifting 16 bits
    *x_out = nk_bf16x8_to_f32x8_haswell_(x_vec.xmms[0]);
    *y_out = nk_bf16x8_to_f32x8_haswell_(y_vec.xmms[0]);
    *z_out = nk_bf16x8_to_f32x8_haswell_(z_vec.xmms[0]);
}

/*  Compute sum of squared distances for f16 data after applying rotation (and optional scale).
 *  Loads f16 data, converts to f32 during processing.
 *  Note: rotation matrix r is f32 (from SVD), scale and computation done in f32.
 */
NK_INTERNAL nk_f32_t nk_transformed_ssd_f16_haswell_(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n,
                                                     nk_f32_t const *r, nk_f32_t scale, nk_f32_t centroid_a_x,
                                                     nk_f32_t centroid_a_y, nk_f32_t centroid_a_z,
                                                     nk_f32_t centroid_b_x, nk_f32_t centroid_b_y,
                                                     nk_f32_t centroid_b_z) {
    // Broadcast scaled rotation matrix elements
    __m256 sr0_f32x8 = _mm256_set1_ps(scale * r[0]), sr1_f32x8 = _mm256_set1_ps(scale * r[1]),
           sr2_f32x8 = _mm256_set1_ps(scale * r[2]);
    __m256 sr3_f32x8 = _mm256_set1_ps(scale * r[3]), sr4_f32x8 = _mm256_set1_ps(scale * r[4]),
           sr5_f32x8 = _mm256_set1_ps(scale * r[5]);
    __m256 sr6_f32x8 = _mm256_set1_ps(scale * r[6]), sr7_f32x8 = _mm256_set1_ps(scale * r[7]),
           sr8_f32x8 = _mm256_set1_ps(scale * r[8]);

    // Broadcast centroids
    __m256 centroid_a_x_f32x8 = _mm256_set1_ps(centroid_a_x);
    __m256 centroid_a_y_f32x8 = _mm256_set1_ps(centroid_a_y);
    __m256 centroid_a_z_f32x8 = _mm256_set1_ps(centroid_a_z);
    __m256 centroid_b_x_f32x8 = _mm256_set1_ps(centroid_b_x);
    __m256 centroid_b_y_f32x8 = _mm256_set1_ps(centroid_b_y);
    __m256 centroid_b_z_f32x8 = _mm256_set1_ps(centroid_b_z);

    __m256 sum_squared_f32x8 = _mm256_setzero_ps();
    __m256 a_x_f32x8, a_y_f32x8, a_z_f32x8, b_x_f32x8, b_y_f32x8, b_z_f32x8;
    nk_size_t j = 0;

    for (; j + 8 <= n; j += 8) {
        nk_deinterleave_f16x8_to_f32x8_haswell_(a + j * 3, &a_x_f32x8, &a_y_f32x8, &a_z_f32x8);
        nk_deinterleave_f16x8_to_f32x8_haswell_(b + j * 3, &b_x_f32x8, &b_y_f32x8, &b_z_f32x8);

        // Center points
        __m256 pa_x_f32x8 = _mm256_sub_ps(a_x_f32x8, centroid_a_x_f32x8);
        __m256 pa_y_f32x8 = _mm256_sub_ps(a_y_f32x8, centroid_a_y_f32x8);
        __m256 pa_z_f32x8 = _mm256_sub_ps(a_z_f32x8, centroid_a_z_f32x8);
        __m256 pb_x_f32x8 = _mm256_sub_ps(b_x_f32x8, centroid_b_x_f32x8);
        __m256 pb_y_f32x8 = _mm256_sub_ps(b_y_f32x8, centroid_b_y_f32x8);
        __m256 pb_z_f32x8 = _mm256_sub_ps(b_z_f32x8, centroid_b_z_f32x8);

        // Rotate and scale: ra = scale * R * pa
        __m256 ra_x_f32x8 = _mm256_fmadd_ps(
            sr2_f32x8, pa_z_f32x8, _mm256_fmadd_ps(sr1_f32x8, pa_y_f32x8, _mm256_mul_ps(sr0_f32x8, pa_x_f32x8)));
        __m256 ra_y_f32x8 = _mm256_fmadd_ps(
            sr5_f32x8, pa_z_f32x8, _mm256_fmadd_ps(sr4_f32x8, pa_y_f32x8, _mm256_mul_ps(sr3_f32x8, pa_x_f32x8)));
        __m256 ra_z_f32x8 = _mm256_fmadd_ps(
            sr8_f32x8, pa_z_f32x8, _mm256_fmadd_ps(sr7_f32x8, pa_y_f32x8, _mm256_mul_ps(sr6_f32x8, pa_x_f32x8)));

        // Delta and accumulate
        __m256 delta_x_f32x8 = _mm256_sub_ps(ra_x_f32x8, pb_x_f32x8);
        __m256 delta_y_f32x8 = _mm256_sub_ps(ra_y_f32x8, pb_y_f32x8);
        __m256 delta_z_f32x8 = _mm256_sub_ps(ra_z_f32x8, pb_z_f32x8);

        sum_squared_f32x8 = _mm256_fmadd_ps(delta_x_f32x8, delta_x_f32x8, sum_squared_f32x8);
        sum_squared_f32x8 = _mm256_fmadd_ps(delta_y_f32x8, delta_y_f32x8, sum_squared_f32x8);
        sum_squared_f32x8 = _mm256_fmadd_ps(delta_z_f32x8, delta_z_f32x8, sum_squared_f32x8);
    }

    nk_f32_t sum_squared = nk_reduce_add_f32x8_haswell_(sum_squared_f32x8);

    // Scalar tail
    for (; j < n; ++j) {
        nk_f32_t a_x_f32, a_y_f32, a_z_f32, b_x_f32, b_y_f32, b_z_f32;
        nk_f16_to_f32_haswell(&a[j * 3 + 0], &a_x_f32);
        nk_f16_to_f32_haswell(&a[j * 3 + 1], &a_y_f32);
        nk_f16_to_f32_haswell(&a[j * 3 + 2], &a_z_f32);
        nk_f16_to_f32_haswell(&b[j * 3 + 0], &b_x_f32);
        nk_f16_to_f32_haswell(&b[j * 3 + 1], &b_y_f32);
        nk_f16_to_f32_haswell(&b[j * 3 + 2], &b_z_f32);

        nk_f32_t pa_x = a_x_f32 - centroid_a_x;
        nk_f32_t pa_y = a_y_f32 - centroid_a_y;
        nk_f32_t pa_z = a_z_f32 - centroid_a_z;
        nk_f32_t pb_x = b_x_f32 - centroid_b_x;
        nk_f32_t pb_y = b_y_f32 - centroid_b_y;
        nk_f32_t pb_z = b_z_f32 - centroid_b_z;

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

/*  Compute sum of squared distances for bf16 data after applying rotation (and optional scale).
 *  Loads bf16 data, converts to f32 during processing.
 *  Note: rotation matrix r is f32 (from SVD), scale and computation done in f32.
 */
NK_INTERNAL nk_f32_t nk_transformed_ssd_bf16_haswell_(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n,
                                                      nk_f32_t const *r, nk_f32_t scale, nk_f32_t centroid_a_x,
                                                      nk_f32_t centroid_a_y, nk_f32_t centroid_a_z,
                                                      nk_f32_t centroid_b_x, nk_f32_t centroid_b_y,
                                                      nk_f32_t centroid_b_z) {
    // Broadcast scaled rotation matrix elements
    __m256 sr0_f32x8 = _mm256_set1_ps(scale * r[0]), sr1_f32x8 = _mm256_set1_ps(scale * r[1]),
           sr2_f32x8 = _mm256_set1_ps(scale * r[2]);
    __m256 sr3_f32x8 = _mm256_set1_ps(scale * r[3]), sr4_f32x8 = _mm256_set1_ps(scale * r[4]),
           sr5_f32x8 = _mm256_set1_ps(scale * r[5]);
    __m256 sr6_f32x8 = _mm256_set1_ps(scale * r[6]), sr7_f32x8 = _mm256_set1_ps(scale * r[7]),
           sr8_f32x8 = _mm256_set1_ps(scale * r[8]);

    // Broadcast centroids
    __m256 centroid_a_x_f32x8 = _mm256_set1_ps(centroid_a_x);
    __m256 centroid_a_y_f32x8 = _mm256_set1_ps(centroid_a_y);
    __m256 centroid_a_z_f32x8 = _mm256_set1_ps(centroid_a_z);
    __m256 centroid_b_x_f32x8 = _mm256_set1_ps(centroid_b_x);
    __m256 centroid_b_y_f32x8 = _mm256_set1_ps(centroid_b_y);
    __m256 centroid_b_z_f32x8 = _mm256_set1_ps(centroid_b_z);

    __m256 sum_squared_f32x8 = _mm256_setzero_ps();
    __m256 a_x_f32x8, a_y_f32x8, a_z_f32x8, b_x_f32x8, b_y_f32x8, b_z_f32x8;
    nk_size_t j = 0;

    for (; j + 8 <= n; j += 8) {
        nk_deinterleave_bf16x8_to_f32x8_haswell_(a + j * 3, &a_x_f32x8, &a_y_f32x8, &a_z_f32x8);
        nk_deinterleave_bf16x8_to_f32x8_haswell_(b + j * 3, &b_x_f32x8, &b_y_f32x8, &b_z_f32x8);

        // Center points
        __m256 pa_x_f32x8 = _mm256_sub_ps(a_x_f32x8, centroid_a_x_f32x8);
        __m256 pa_y_f32x8 = _mm256_sub_ps(a_y_f32x8, centroid_a_y_f32x8);
        __m256 pa_z_f32x8 = _mm256_sub_ps(a_z_f32x8, centroid_a_z_f32x8);
        __m256 pb_x_f32x8 = _mm256_sub_ps(b_x_f32x8, centroid_b_x_f32x8);
        __m256 pb_y_f32x8 = _mm256_sub_ps(b_y_f32x8, centroid_b_y_f32x8);
        __m256 pb_z_f32x8 = _mm256_sub_ps(b_z_f32x8, centroid_b_z_f32x8);

        // Rotate and scale: ra = scale * R * pa
        __m256 ra_x_f32x8 = _mm256_fmadd_ps(
            sr2_f32x8, pa_z_f32x8, _mm256_fmadd_ps(sr1_f32x8, pa_y_f32x8, _mm256_mul_ps(sr0_f32x8, pa_x_f32x8)));
        __m256 ra_y_f32x8 = _mm256_fmadd_ps(
            sr5_f32x8, pa_z_f32x8, _mm256_fmadd_ps(sr4_f32x8, pa_y_f32x8, _mm256_mul_ps(sr3_f32x8, pa_x_f32x8)));
        __m256 ra_z_f32x8 = _mm256_fmadd_ps(
            sr8_f32x8, pa_z_f32x8, _mm256_fmadd_ps(sr7_f32x8, pa_y_f32x8, _mm256_mul_ps(sr6_f32x8, pa_x_f32x8)));

        // Delta and accumulate
        __m256 delta_x_f32x8 = _mm256_sub_ps(ra_x_f32x8, pb_x_f32x8);
        __m256 delta_y_f32x8 = _mm256_sub_ps(ra_y_f32x8, pb_y_f32x8);
        __m256 delta_z_f32x8 = _mm256_sub_ps(ra_z_f32x8, pb_z_f32x8);

        sum_squared_f32x8 = _mm256_fmadd_ps(delta_x_f32x8, delta_x_f32x8, sum_squared_f32x8);
        sum_squared_f32x8 = _mm256_fmadd_ps(delta_y_f32x8, delta_y_f32x8, sum_squared_f32x8);
        sum_squared_f32x8 = _mm256_fmadd_ps(delta_z_f32x8, delta_z_f32x8, sum_squared_f32x8);
    }

    nk_f32_t sum_squared = nk_reduce_add_f32x8_haswell_(sum_squared_f32x8);

    // Scalar tail
    for (; j < n; ++j) {
        nk_f32_t a_x_f32, a_y_f32, a_z_f32, b_x_f32, b_y_f32, b_z_f32;
        nk_bf16_to_f32_serial(&a[j * 3 + 0], &a_x_f32);
        nk_bf16_to_f32_serial(&a[j * 3 + 1], &a_y_f32);
        nk_bf16_to_f32_serial(&a[j * 3 + 2], &a_z_f32);
        nk_bf16_to_f32_serial(&b[j * 3 + 0], &b_x_f32);
        nk_bf16_to_f32_serial(&b[j * 3 + 1], &b_y_f32);
        nk_bf16_to_f32_serial(&b[j * 3 + 2], &b_z_f32);

        nk_f32_t pa_x = a_x_f32 - centroid_a_x;
        nk_f32_t pa_y = a_y_f32 - centroid_a_y;
        nk_f32_t pa_z = a_z_f32 - centroid_a_z;
        nk_f32_t pb_x = b_x_f32 - centroid_b_x;
        nk_f32_t pb_y = b_y_f32 - centroid_b_y;
        nk_f32_t pb_z = b_z_f32 - centroid_b_z;

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

NK_PUBLIC void nk_rmsd_f16_haswell(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                   nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result) {
    /* RMSD uses identity rotation and scale=1.0 */
    if (rotation) {
        rotation[0] = 1, rotation[1] = 0, rotation[2] = 0;
        rotation[3] = 0, rotation[4] = 1, rotation[5] = 0;
        rotation[6] = 0, rotation[7] = 0, rotation[8] = 1;
    }
    if (scale) *scale = 1.0f;

    __m256 const zeros_f32x8 = _mm256_setzero_ps();

    // Accumulators for centroids and squared differences (all in f32)
    __m256 sum_a_x_f32x8 = zeros_f32x8, sum_a_y_f32x8 = zeros_f32x8, sum_a_z_f32x8 = zeros_f32x8;
    __m256 sum_b_x_f32x8 = zeros_f32x8, sum_b_y_f32x8 = zeros_f32x8, sum_b_z_f32x8 = zeros_f32x8;
    __m256 sum_squared_x_f32x8 = zeros_f32x8, sum_squared_y_f32x8 = zeros_f32x8, sum_squared_z_f32x8 = zeros_f32x8;

    __m256 a_x_f32x8, a_y_f32x8, a_z_f32x8, b_x_f32x8, b_y_f32x8, b_z_f32x8;
    nk_size_t i = 0;

    // Main loop processing 8 points at a time
    for (; i + 8 <= n; i += 8) {
        nk_deinterleave_f16x8_to_f32x8_haswell_(a + i * 3, &a_x_f32x8, &a_y_f32x8, &a_z_f32x8);
        nk_deinterleave_f16x8_to_f32x8_haswell_(b + i * 3, &b_x_f32x8, &b_y_f32x8, &b_z_f32x8);

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
    nk_f32_t total_ax = nk_reduce_add_f32x8_haswell_(sum_a_x_f32x8);
    nk_f32_t total_ay = nk_reduce_add_f32x8_haswell_(sum_a_y_f32x8);
    nk_f32_t total_az = nk_reduce_add_f32x8_haswell_(sum_a_z_f32x8);
    nk_f32_t total_bx = nk_reduce_add_f32x8_haswell_(sum_b_x_f32x8);
    nk_f32_t total_by = nk_reduce_add_f32x8_haswell_(sum_b_y_f32x8);
    nk_f32_t total_bz = nk_reduce_add_f32x8_haswell_(sum_b_z_f32x8);
    nk_f32_t total_sq_x = nk_reduce_add_f32x8_haswell_(sum_squared_x_f32x8);
    nk_f32_t total_sq_y = nk_reduce_add_f32x8_haswell_(sum_squared_y_f32x8);
    nk_f32_t total_sq_z = nk_reduce_add_f32x8_haswell_(sum_squared_z_f32x8);

    // Scalar tail
    for (; i < n; ++i) {
        nk_f32_t ax, ay, az, bx, by, bz;
        nk_f16_to_f32_haswell(&a[i * 3 + 0], &ax);
        nk_f16_to_f32_haswell(&a[i * 3 + 1], &ay);
        nk_f16_to_f32_haswell(&a[i * 3 + 2], &az);
        nk_f16_to_f32_haswell(&b[i * 3 + 0], &bx);
        nk_f16_to_f32_haswell(&b[i * 3 + 1], &by);
        nk_f16_to_f32_haswell(&b[i * 3 + 2], &bz);
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

    *result = nk_sqrt_f32_haswell_(sum_squared * inv_n - mean_diff_sq);
}

NK_PUBLIC void nk_rmsd_bf16_haswell(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                    nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result) {
    /* RMSD uses identity rotation and scale=1.0 */
    if (rotation) {
        rotation[0] = 1, rotation[1] = 0, rotation[2] = 0;
        rotation[3] = 0, rotation[4] = 1, rotation[5] = 0;
        rotation[6] = 0, rotation[7] = 0, rotation[8] = 1;
    }
    if (scale) *scale = 1.0f;

    __m256 const zeros_f32x8 = _mm256_setzero_ps();

    // Accumulators for centroids and squared differences (all in f32)
    __m256 sum_a_x_f32x8 = zeros_f32x8, sum_a_y_f32x8 = zeros_f32x8, sum_a_z_f32x8 = zeros_f32x8;
    __m256 sum_b_x_f32x8 = zeros_f32x8, sum_b_y_f32x8 = zeros_f32x8, sum_b_z_f32x8 = zeros_f32x8;
    __m256 sum_squared_x_f32x8 = zeros_f32x8, sum_squared_y_f32x8 = zeros_f32x8, sum_squared_z_f32x8 = zeros_f32x8;

    __m256 a_x_f32x8, a_y_f32x8, a_z_f32x8, b_x_f32x8, b_y_f32x8, b_z_f32x8;
    nk_size_t i = 0;

    // Main loop processing 8 points at a time
    for (; i + 8 <= n; i += 8) {
        nk_deinterleave_bf16x8_to_f32x8_haswell_(a + i * 3, &a_x_f32x8, &a_y_f32x8, &a_z_f32x8);
        nk_deinterleave_bf16x8_to_f32x8_haswell_(b + i * 3, &b_x_f32x8, &b_y_f32x8, &b_z_f32x8);

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
    nk_f32_t total_ax = nk_reduce_add_f32x8_haswell_(sum_a_x_f32x8);
    nk_f32_t total_ay = nk_reduce_add_f32x8_haswell_(sum_a_y_f32x8);
    nk_f32_t total_az = nk_reduce_add_f32x8_haswell_(sum_a_z_f32x8);
    nk_f32_t total_bx = nk_reduce_add_f32x8_haswell_(sum_b_x_f32x8);
    nk_f32_t total_by = nk_reduce_add_f32x8_haswell_(sum_b_y_f32x8);
    nk_f32_t total_bz = nk_reduce_add_f32x8_haswell_(sum_b_z_f32x8);
    nk_f32_t total_sq_x = nk_reduce_add_f32x8_haswell_(sum_squared_x_f32x8);
    nk_f32_t total_sq_y = nk_reduce_add_f32x8_haswell_(sum_squared_y_f32x8);
    nk_f32_t total_sq_z = nk_reduce_add_f32x8_haswell_(sum_squared_z_f32x8);

    // Scalar tail
    for (; i < n; ++i) {
        nk_f32_t ax, ay, az, bx, by, bz;
        nk_bf16_to_f32_serial(&a[i * 3 + 0], &ax);
        nk_bf16_to_f32_serial(&a[i * 3 + 1], &ay);
        nk_bf16_to_f32_serial(&a[i * 3 + 2], &az);
        nk_bf16_to_f32_serial(&b[i * 3 + 0], &bx);
        nk_bf16_to_f32_serial(&b[i * 3 + 1], &by);
        nk_bf16_to_f32_serial(&b[i * 3 + 2], &bz);
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

    *result = nk_sqrt_f32_haswell_(sum_squared * inv_n - mean_diff_sq);
}

NK_PUBLIC void nk_kabsch_f16_haswell(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                     nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result) {
    // Fused single-pass: load f16, convert to f32, compute centroids and covariance
    __m256 const zeros_f32x8 = _mm256_setzero_ps();

    // Accumulators for centroids (f32)
    __m256 sum_a_x_f32x8 = zeros_f32x8, sum_a_y_f32x8 = zeros_f32x8, sum_a_z_f32x8 = zeros_f32x8;
    __m256 sum_b_x_f32x8 = zeros_f32x8, sum_b_y_f32x8 = zeros_f32x8, sum_b_z_f32x8 = zeros_f32x8;

    // Accumulators for covariance matrix (sum of outer products)
    __m256 cov_xx_f32x8 = zeros_f32x8, cov_xy_f32x8 = zeros_f32x8, cov_xz_f32x8 = zeros_f32x8;
    __m256 cov_yx_f32x8 = zeros_f32x8, cov_yy_f32x8 = zeros_f32x8, cov_yz_f32x8 = zeros_f32x8;
    __m256 cov_zx_f32x8 = zeros_f32x8, cov_zy_f32x8 = zeros_f32x8, cov_zz_f32x8 = zeros_f32x8;

    nk_size_t i = 0;
    __m256 a_x_f32x8, a_y_f32x8, a_z_f32x8, b_x_f32x8, b_y_f32x8, b_z_f32x8;

    for (; i + 8 <= n; i += 8) {
        nk_deinterleave_f16x8_to_f32x8_haswell_(a + i * 3, &a_x_f32x8, &a_y_f32x8, &a_z_f32x8);
        nk_deinterleave_f16x8_to_f32x8_haswell_(b + i * 3, &b_x_f32x8, &b_y_f32x8, &b_z_f32x8);

        // Accumulate centroids
        sum_a_x_f32x8 = _mm256_add_ps(sum_a_x_f32x8, a_x_f32x8);
        sum_a_y_f32x8 = _mm256_add_ps(sum_a_y_f32x8, a_y_f32x8);
        sum_a_z_f32x8 = _mm256_add_ps(sum_a_z_f32x8, a_z_f32x8);
        sum_b_x_f32x8 = _mm256_add_ps(sum_b_x_f32x8, b_x_f32x8);
        sum_b_y_f32x8 = _mm256_add_ps(sum_b_y_f32x8, b_y_f32x8);
        sum_b_z_f32x8 = _mm256_add_ps(sum_b_z_f32x8, b_z_f32x8);

        // Accumulate outer products
        cov_xx_f32x8 = _mm256_fmadd_ps(a_x_f32x8, b_x_f32x8, cov_xx_f32x8);
        cov_xy_f32x8 = _mm256_fmadd_ps(a_x_f32x8, b_y_f32x8, cov_xy_f32x8);
        cov_xz_f32x8 = _mm256_fmadd_ps(a_x_f32x8, b_z_f32x8, cov_xz_f32x8);
        cov_yx_f32x8 = _mm256_fmadd_ps(a_y_f32x8, b_x_f32x8, cov_yx_f32x8);
        cov_yy_f32x8 = _mm256_fmadd_ps(a_y_f32x8, b_y_f32x8, cov_yy_f32x8);
        cov_yz_f32x8 = _mm256_fmadd_ps(a_y_f32x8, b_z_f32x8, cov_yz_f32x8);
        cov_zx_f32x8 = _mm256_fmadd_ps(a_z_f32x8, b_x_f32x8, cov_zx_f32x8);
        cov_zy_f32x8 = _mm256_fmadd_ps(a_z_f32x8, b_y_f32x8, cov_zy_f32x8);
        cov_zz_f32x8 = _mm256_fmadd_ps(a_z_f32x8, b_z_f32x8, cov_zz_f32x8);
    }

    // Reduce vector accumulators
    nk_f32_t sum_a_x = nk_reduce_add_f32x8_haswell_(sum_a_x_f32x8);
    nk_f32_t sum_a_y = nk_reduce_add_f32x8_haswell_(sum_a_y_f32x8);
    nk_f32_t sum_a_z = nk_reduce_add_f32x8_haswell_(sum_a_z_f32x8);
    nk_f32_t sum_b_x = nk_reduce_add_f32x8_haswell_(sum_b_x_f32x8);
    nk_f32_t sum_b_y = nk_reduce_add_f32x8_haswell_(sum_b_y_f32x8);
    nk_f32_t sum_b_z = nk_reduce_add_f32x8_haswell_(sum_b_z_f32x8);

    nk_f32_t h00 = nk_reduce_add_f32x8_haswell_(cov_xx_f32x8);
    nk_f32_t h01 = nk_reduce_add_f32x8_haswell_(cov_xy_f32x8);
    nk_f32_t h02 = nk_reduce_add_f32x8_haswell_(cov_xz_f32x8);
    nk_f32_t h10 = nk_reduce_add_f32x8_haswell_(cov_yx_f32x8);
    nk_f32_t h11 = nk_reduce_add_f32x8_haswell_(cov_yy_f32x8);
    nk_f32_t h12 = nk_reduce_add_f32x8_haswell_(cov_yz_f32x8);
    nk_f32_t h20 = nk_reduce_add_f32x8_haswell_(cov_zx_f32x8);
    nk_f32_t h21 = nk_reduce_add_f32x8_haswell_(cov_zy_f32x8);
    nk_f32_t h22 = nk_reduce_add_f32x8_haswell_(cov_zz_f32x8);

    // Scalar tail
    for (; i < n; ++i) {
        nk_f32_t ax, ay, az, bx, by, bz;
        nk_f16_to_f32_haswell(&a[i * 3 + 0], &ax);
        nk_f16_to_f32_haswell(&a[i * 3 + 1], &ay);
        nk_f16_to_f32_haswell(&a[i * 3 + 2], &az);
        nk_f16_to_f32_haswell(&b[i * 3 + 0], &bx);
        nk_f16_to_f32_haswell(&b[i * 3 + 1], &by);
        nk_f16_to_f32_haswell(&b[i * 3 + 2], &bz);
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
    nk_f32_t svd_u[9], svd_s[9], svd_v[9];
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
    nk_f32_t sum_squared = nk_transformed_ssd_f16_haswell_(a, b, n, r, 1.0f, centroid_a_x, centroid_a_y, centroid_a_z,
                                                           centroid_b_x, centroid_b_y, centroid_b_z);
    *result = nk_sqrt_f32_haswell_(sum_squared * inv_n);
}

NK_PUBLIC void nk_kabsch_bf16_haswell(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                      nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result) {
    // Fused single-pass: load bf16, convert to f32, compute centroids and covariance
    __m256 const zeros_f32x8 = _mm256_setzero_ps();

    // Accumulators for centroids (f32)
    __m256 sum_a_x_f32x8 = zeros_f32x8, sum_a_y_f32x8 = zeros_f32x8, sum_a_z_f32x8 = zeros_f32x8;
    __m256 sum_b_x_f32x8 = zeros_f32x8, sum_b_y_f32x8 = zeros_f32x8, sum_b_z_f32x8 = zeros_f32x8;

    // Accumulators for covariance matrix (sum of outer products)
    __m256 cov_xx_f32x8 = zeros_f32x8, cov_xy_f32x8 = zeros_f32x8, cov_xz_f32x8 = zeros_f32x8;
    __m256 cov_yx_f32x8 = zeros_f32x8, cov_yy_f32x8 = zeros_f32x8, cov_yz_f32x8 = zeros_f32x8;
    __m256 cov_zx_f32x8 = zeros_f32x8, cov_zy_f32x8 = zeros_f32x8, cov_zz_f32x8 = zeros_f32x8;

    nk_size_t i = 0;
    __m256 a_x_f32x8, a_y_f32x8, a_z_f32x8, b_x_f32x8, b_y_f32x8, b_z_f32x8;

    for (; i + 8 <= n; i += 8) {
        nk_deinterleave_bf16x8_to_f32x8_haswell_(a + i * 3, &a_x_f32x8, &a_y_f32x8, &a_z_f32x8);
        nk_deinterleave_bf16x8_to_f32x8_haswell_(b + i * 3, &b_x_f32x8, &b_y_f32x8, &b_z_f32x8);

        // Accumulate centroids
        sum_a_x_f32x8 = _mm256_add_ps(sum_a_x_f32x8, a_x_f32x8);
        sum_a_y_f32x8 = _mm256_add_ps(sum_a_y_f32x8, a_y_f32x8);
        sum_a_z_f32x8 = _mm256_add_ps(sum_a_z_f32x8, a_z_f32x8);
        sum_b_x_f32x8 = _mm256_add_ps(sum_b_x_f32x8, b_x_f32x8);
        sum_b_y_f32x8 = _mm256_add_ps(sum_b_y_f32x8, b_y_f32x8);
        sum_b_z_f32x8 = _mm256_add_ps(sum_b_z_f32x8, b_z_f32x8);

        // Accumulate outer products
        cov_xx_f32x8 = _mm256_fmadd_ps(a_x_f32x8, b_x_f32x8, cov_xx_f32x8);
        cov_xy_f32x8 = _mm256_fmadd_ps(a_x_f32x8, b_y_f32x8, cov_xy_f32x8);
        cov_xz_f32x8 = _mm256_fmadd_ps(a_x_f32x8, b_z_f32x8, cov_xz_f32x8);
        cov_yx_f32x8 = _mm256_fmadd_ps(a_y_f32x8, b_x_f32x8, cov_yx_f32x8);
        cov_yy_f32x8 = _mm256_fmadd_ps(a_y_f32x8, b_y_f32x8, cov_yy_f32x8);
        cov_yz_f32x8 = _mm256_fmadd_ps(a_y_f32x8, b_z_f32x8, cov_yz_f32x8);
        cov_zx_f32x8 = _mm256_fmadd_ps(a_z_f32x8, b_x_f32x8, cov_zx_f32x8);
        cov_zy_f32x8 = _mm256_fmadd_ps(a_z_f32x8, b_y_f32x8, cov_zy_f32x8);
        cov_zz_f32x8 = _mm256_fmadd_ps(a_z_f32x8, b_z_f32x8, cov_zz_f32x8);
    }

    // Reduce vector accumulators
    nk_f32_t sum_a_x = nk_reduce_add_f32x8_haswell_(sum_a_x_f32x8);
    nk_f32_t sum_a_y = nk_reduce_add_f32x8_haswell_(sum_a_y_f32x8);
    nk_f32_t sum_a_z = nk_reduce_add_f32x8_haswell_(sum_a_z_f32x8);
    nk_f32_t sum_b_x = nk_reduce_add_f32x8_haswell_(sum_b_x_f32x8);
    nk_f32_t sum_b_y = nk_reduce_add_f32x8_haswell_(sum_b_y_f32x8);
    nk_f32_t sum_b_z = nk_reduce_add_f32x8_haswell_(sum_b_z_f32x8);

    nk_f32_t h00 = nk_reduce_add_f32x8_haswell_(cov_xx_f32x8);
    nk_f32_t h01 = nk_reduce_add_f32x8_haswell_(cov_xy_f32x8);
    nk_f32_t h02 = nk_reduce_add_f32x8_haswell_(cov_xz_f32x8);
    nk_f32_t h10 = nk_reduce_add_f32x8_haswell_(cov_yx_f32x8);
    nk_f32_t h11 = nk_reduce_add_f32x8_haswell_(cov_yy_f32x8);
    nk_f32_t h12 = nk_reduce_add_f32x8_haswell_(cov_yz_f32x8);
    nk_f32_t h20 = nk_reduce_add_f32x8_haswell_(cov_zx_f32x8);
    nk_f32_t h21 = nk_reduce_add_f32x8_haswell_(cov_zy_f32x8);
    nk_f32_t h22 = nk_reduce_add_f32x8_haswell_(cov_zz_f32x8);

    // Scalar tail
    for (; i < n; ++i) {
        nk_f32_t ax, ay, az, bx, by, bz;
        nk_bf16_to_f32_serial(&a[i * 3 + 0], &ax);
        nk_bf16_to_f32_serial(&a[i * 3 + 1], &ay);
        nk_bf16_to_f32_serial(&a[i * 3 + 2], &az);
        nk_bf16_to_f32_serial(&b[i * 3 + 0], &bx);
        nk_bf16_to_f32_serial(&b[i * 3 + 1], &by);
        nk_bf16_to_f32_serial(&b[i * 3 + 2], &bz);
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
    nk_f32_t svd_u[9], svd_s[9], svd_v[9];
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
    nk_f32_t sum_squared = nk_transformed_ssd_bf16_haswell_(a, b, n, r, 1.0f, centroid_a_x, centroid_a_y, centroid_a_z,
                                                            centroid_b_x, centroid_b_y, centroid_b_z);
    *result = nk_sqrt_f32_haswell_(sum_squared * inv_n);
}

NK_PUBLIC void nk_umeyama_f16_haswell(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                      nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result) {
    // Fused single-pass: load f16, convert to f32, compute centroids, covariance, and variance
    __m256 const zeros_f32x8 = _mm256_setzero_ps();

    __m256 sum_a_x_f32x8 = zeros_f32x8, sum_a_y_f32x8 = zeros_f32x8, sum_a_z_f32x8 = zeros_f32x8;
    __m256 sum_b_x_f32x8 = zeros_f32x8, sum_b_y_f32x8 = zeros_f32x8, sum_b_z_f32x8 = zeros_f32x8;
    __m256 cov_xx_f32x8 = zeros_f32x8, cov_xy_f32x8 = zeros_f32x8, cov_xz_f32x8 = zeros_f32x8;
    __m256 cov_yx_f32x8 = zeros_f32x8, cov_yy_f32x8 = zeros_f32x8, cov_yz_f32x8 = zeros_f32x8;
    __m256 cov_zx_f32x8 = zeros_f32x8, cov_zy_f32x8 = zeros_f32x8, cov_zz_f32x8 = zeros_f32x8;
    __m256 variance_a_f32x8 = zeros_f32x8;

    nk_size_t i = 0;
    __m256 a_x_f32x8, a_y_f32x8, a_z_f32x8, b_x_f32x8, b_y_f32x8, b_z_f32x8;

    for (; i + 8 <= n; i += 8) {
        nk_deinterleave_f16x8_to_f32x8_haswell_(a + i * 3, &a_x_f32x8, &a_y_f32x8, &a_z_f32x8);
        nk_deinterleave_f16x8_to_f32x8_haswell_(b + i * 3, &b_x_f32x8, &b_y_f32x8, &b_z_f32x8);

        // Accumulate centroids
        sum_a_x_f32x8 = _mm256_add_ps(sum_a_x_f32x8, a_x_f32x8);
        sum_a_y_f32x8 = _mm256_add_ps(sum_a_y_f32x8, a_y_f32x8);
        sum_a_z_f32x8 = _mm256_add_ps(sum_a_z_f32x8, a_z_f32x8);
        sum_b_x_f32x8 = _mm256_add_ps(sum_b_x_f32x8, b_x_f32x8);
        sum_b_y_f32x8 = _mm256_add_ps(sum_b_y_f32x8, b_y_f32x8);
        sum_b_z_f32x8 = _mm256_add_ps(sum_b_z_f32x8, b_z_f32x8);

        // Accumulate outer products
        cov_xx_f32x8 = _mm256_fmadd_ps(a_x_f32x8, b_x_f32x8, cov_xx_f32x8);
        cov_xy_f32x8 = _mm256_fmadd_ps(a_x_f32x8, b_y_f32x8, cov_xy_f32x8);
        cov_xz_f32x8 = _mm256_fmadd_ps(a_x_f32x8, b_z_f32x8, cov_xz_f32x8);
        cov_yx_f32x8 = _mm256_fmadd_ps(a_y_f32x8, b_x_f32x8, cov_yx_f32x8);
        cov_yy_f32x8 = _mm256_fmadd_ps(a_y_f32x8, b_y_f32x8, cov_yy_f32x8);
        cov_yz_f32x8 = _mm256_fmadd_ps(a_y_f32x8, b_z_f32x8, cov_yz_f32x8);
        cov_zx_f32x8 = _mm256_fmadd_ps(a_z_f32x8, b_x_f32x8, cov_zx_f32x8);
        cov_zy_f32x8 = _mm256_fmadd_ps(a_z_f32x8, b_y_f32x8, cov_zy_f32x8);
        cov_zz_f32x8 = _mm256_fmadd_ps(a_z_f32x8, b_z_f32x8, cov_zz_f32x8);

        // Accumulate variance of A
        variance_a_f32x8 = _mm256_fmadd_ps(a_x_f32x8, a_x_f32x8, variance_a_f32x8);
        variance_a_f32x8 = _mm256_fmadd_ps(a_y_f32x8, a_y_f32x8, variance_a_f32x8);
        variance_a_f32x8 = _mm256_fmadd_ps(a_z_f32x8, a_z_f32x8, variance_a_f32x8);
    }

    // Reduce vector accumulators
    nk_f32_t sum_a_x = nk_reduce_add_f32x8_haswell_(sum_a_x_f32x8);
    nk_f32_t sum_a_y = nk_reduce_add_f32x8_haswell_(sum_a_y_f32x8);
    nk_f32_t sum_a_z = nk_reduce_add_f32x8_haswell_(sum_a_z_f32x8);
    nk_f32_t sum_b_x = nk_reduce_add_f32x8_haswell_(sum_b_x_f32x8);
    nk_f32_t sum_b_y = nk_reduce_add_f32x8_haswell_(sum_b_y_f32x8);
    nk_f32_t sum_b_z = nk_reduce_add_f32x8_haswell_(sum_b_z_f32x8);
    nk_f32_t h00 = nk_reduce_add_f32x8_haswell_(cov_xx_f32x8);
    nk_f32_t h01 = nk_reduce_add_f32x8_haswell_(cov_xy_f32x8);
    nk_f32_t h02 = nk_reduce_add_f32x8_haswell_(cov_xz_f32x8);
    nk_f32_t h10 = nk_reduce_add_f32x8_haswell_(cov_yx_f32x8);
    nk_f32_t h11 = nk_reduce_add_f32x8_haswell_(cov_yy_f32x8);
    nk_f32_t h12 = nk_reduce_add_f32x8_haswell_(cov_yz_f32x8);
    nk_f32_t h20 = nk_reduce_add_f32x8_haswell_(cov_zx_f32x8);
    nk_f32_t h21 = nk_reduce_add_f32x8_haswell_(cov_zy_f32x8);
    nk_f32_t h22 = nk_reduce_add_f32x8_haswell_(cov_zz_f32x8);
    nk_f32_t variance_a_sum = nk_reduce_add_f32x8_haswell_(variance_a_f32x8);

    // Scalar tail
    for (; i < n; ++i) {
        nk_f32_t ax, ay, az, bx, by, bz;
        nk_f16_to_f32_haswell(&a[i * 3 + 0], &ax);
        nk_f16_to_f32_haswell(&a[i * 3 + 1], &ay);
        nk_f16_to_f32_haswell(&a[i * 3 + 2], &az);
        nk_f16_to_f32_haswell(&b[i * 3 + 0], &bx);
        nk_f16_to_f32_haswell(&b[i * 3 + 1], &by);
        nk_f16_to_f32_haswell(&b[i * 3 + 2], &bz);
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
        variance_a_sum += ax * ax + ay * ay + az * az;
    }

    // Compute centroids
    nk_f32_t inv_n = 1.0f / (nk_f32_t)n;
    nk_f32_t centroid_a_x = sum_a_x * inv_n, centroid_a_y = sum_a_y * inv_n, centroid_a_z = sum_a_z * inv_n;
    nk_f32_t centroid_b_x = sum_b_x * inv_n, centroid_b_y = sum_b_y * inv_n, centroid_b_z = sum_b_z * inv_n;

    if (a_centroid) a_centroid[0] = centroid_a_x, a_centroid[1] = centroid_a_y, a_centroid[2] = centroid_a_z;
    if (b_centroid) b_centroid[0] = centroid_b_x, b_centroid[1] = centroid_b_y, b_centroid[2] = centroid_b_z;

    // Compute centered covariance and variance
    nk_f32_t variance_a = variance_a_sum * inv_n -
                          (centroid_a_x * centroid_a_x + centroid_a_y * centroid_a_y + centroid_a_z * centroid_a_z);

    // Apply centering correction to covariance matrix
    h00 -= n * centroid_a_x * centroid_b_x;
    h01 -= n * centroid_a_x * centroid_b_y;
    h02 -= n * centroid_a_x * centroid_b_z;
    h10 -= n * centroid_a_y * centroid_b_x;
    h11 -= n * centroid_a_y * centroid_b_y;
    h12 -= n * centroid_a_y * centroid_b_z;
    h20 -= n * centroid_a_z * centroid_b_x;
    h21 -= n * centroid_a_z * centroid_b_y;
    h22 -= n * centroid_a_z * centroid_b_z;

    nk_f32_t cross_covariance[9] = {h00, h01, h02, h10, h11, h12, h20, h21, h22};

    // SVD
    nk_f32_t svd_u[9], svd_s[9], svd_v[9];
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

    // Scale factor: c = trace(D × S) / (n × variance(a))
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
    nk_f32_t sum_squared = nk_transformed_ssd_f16_haswell_(a, b, n, r, c, centroid_a_x, centroid_a_y, centroid_a_z,
                                                           centroid_b_x, centroid_b_y, centroid_b_z);
    *result = nk_sqrt_f32_haswell_(sum_squared * inv_n);
}

NK_PUBLIC void nk_umeyama_bf16_haswell(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                       nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result) {
    // Fused single-pass: load bf16, convert to f32, compute centroids, covariance, and variance
    __m256 const zeros_f32x8 = _mm256_setzero_ps();

    __m256 sum_a_x_f32x8 = zeros_f32x8, sum_a_y_f32x8 = zeros_f32x8, sum_a_z_f32x8 = zeros_f32x8;
    __m256 sum_b_x_f32x8 = zeros_f32x8, sum_b_y_f32x8 = zeros_f32x8, sum_b_z_f32x8 = zeros_f32x8;
    __m256 cov_xx_f32x8 = zeros_f32x8, cov_xy_f32x8 = zeros_f32x8, cov_xz_f32x8 = zeros_f32x8;
    __m256 cov_yx_f32x8 = zeros_f32x8, cov_yy_f32x8 = zeros_f32x8, cov_yz_f32x8 = zeros_f32x8;
    __m256 cov_zx_f32x8 = zeros_f32x8, cov_zy_f32x8 = zeros_f32x8, cov_zz_f32x8 = zeros_f32x8;
    __m256 variance_a_f32x8 = zeros_f32x8;

    nk_size_t i = 0;
    __m256 a_x_f32x8, a_y_f32x8, a_z_f32x8, b_x_f32x8, b_y_f32x8, b_z_f32x8;

    for (; i + 8 <= n; i += 8) {
        nk_deinterleave_bf16x8_to_f32x8_haswell_(a + i * 3, &a_x_f32x8, &a_y_f32x8, &a_z_f32x8);
        nk_deinterleave_bf16x8_to_f32x8_haswell_(b + i * 3, &b_x_f32x8, &b_y_f32x8, &b_z_f32x8);

        // Accumulate centroids
        sum_a_x_f32x8 = _mm256_add_ps(sum_a_x_f32x8, a_x_f32x8);
        sum_a_y_f32x8 = _mm256_add_ps(sum_a_y_f32x8, a_y_f32x8);
        sum_a_z_f32x8 = _mm256_add_ps(sum_a_z_f32x8, a_z_f32x8);
        sum_b_x_f32x8 = _mm256_add_ps(sum_b_x_f32x8, b_x_f32x8);
        sum_b_y_f32x8 = _mm256_add_ps(sum_b_y_f32x8, b_y_f32x8);
        sum_b_z_f32x8 = _mm256_add_ps(sum_b_z_f32x8, b_z_f32x8);

        // Accumulate outer products
        cov_xx_f32x8 = _mm256_fmadd_ps(a_x_f32x8, b_x_f32x8, cov_xx_f32x8);
        cov_xy_f32x8 = _mm256_fmadd_ps(a_x_f32x8, b_y_f32x8, cov_xy_f32x8);
        cov_xz_f32x8 = _mm256_fmadd_ps(a_x_f32x8, b_z_f32x8, cov_xz_f32x8);
        cov_yx_f32x8 = _mm256_fmadd_ps(a_y_f32x8, b_x_f32x8, cov_yx_f32x8);
        cov_yy_f32x8 = _mm256_fmadd_ps(a_y_f32x8, b_y_f32x8, cov_yy_f32x8);
        cov_yz_f32x8 = _mm256_fmadd_ps(a_y_f32x8, b_z_f32x8, cov_yz_f32x8);
        cov_zx_f32x8 = _mm256_fmadd_ps(a_z_f32x8, b_x_f32x8, cov_zx_f32x8);
        cov_zy_f32x8 = _mm256_fmadd_ps(a_z_f32x8, b_y_f32x8, cov_zy_f32x8);
        cov_zz_f32x8 = _mm256_fmadd_ps(a_z_f32x8, b_z_f32x8, cov_zz_f32x8);

        // Accumulate variance of A
        variance_a_f32x8 = _mm256_fmadd_ps(a_x_f32x8, a_x_f32x8, variance_a_f32x8);
        variance_a_f32x8 = _mm256_fmadd_ps(a_y_f32x8, a_y_f32x8, variance_a_f32x8);
        variance_a_f32x8 = _mm256_fmadd_ps(a_z_f32x8, a_z_f32x8, variance_a_f32x8);
    }

    // Reduce vector accumulators
    nk_f32_t sum_a_x = nk_reduce_add_f32x8_haswell_(sum_a_x_f32x8);
    nk_f32_t sum_a_y = nk_reduce_add_f32x8_haswell_(sum_a_y_f32x8);
    nk_f32_t sum_a_z = nk_reduce_add_f32x8_haswell_(sum_a_z_f32x8);
    nk_f32_t sum_b_x = nk_reduce_add_f32x8_haswell_(sum_b_x_f32x8);
    nk_f32_t sum_b_y = nk_reduce_add_f32x8_haswell_(sum_b_y_f32x8);
    nk_f32_t sum_b_z = nk_reduce_add_f32x8_haswell_(sum_b_z_f32x8);
    nk_f32_t h00 = nk_reduce_add_f32x8_haswell_(cov_xx_f32x8);
    nk_f32_t h01 = nk_reduce_add_f32x8_haswell_(cov_xy_f32x8);
    nk_f32_t h02 = nk_reduce_add_f32x8_haswell_(cov_xz_f32x8);
    nk_f32_t h10 = nk_reduce_add_f32x8_haswell_(cov_yx_f32x8);
    nk_f32_t h11 = nk_reduce_add_f32x8_haswell_(cov_yy_f32x8);
    nk_f32_t h12 = nk_reduce_add_f32x8_haswell_(cov_yz_f32x8);
    nk_f32_t h20 = nk_reduce_add_f32x8_haswell_(cov_zx_f32x8);
    nk_f32_t h21 = nk_reduce_add_f32x8_haswell_(cov_zy_f32x8);
    nk_f32_t h22 = nk_reduce_add_f32x8_haswell_(cov_zz_f32x8);
    nk_f32_t variance_a_sum = nk_reduce_add_f32x8_haswell_(variance_a_f32x8);

    // Scalar tail
    for (; i < n; ++i) {
        nk_f32_t ax, ay, az, bx, by, bz;
        nk_bf16_to_f32_serial(&a[i * 3 + 0], &ax);
        nk_bf16_to_f32_serial(&a[i * 3 + 1], &ay);
        nk_bf16_to_f32_serial(&a[i * 3 + 2], &az);
        nk_bf16_to_f32_serial(&b[i * 3 + 0], &bx);
        nk_bf16_to_f32_serial(&b[i * 3 + 1], &by);
        nk_bf16_to_f32_serial(&b[i * 3 + 2], &bz);
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
        variance_a_sum += ax * ax + ay * ay + az * az;
    }

    // Compute centroids
    nk_f32_t inv_n = 1.0f / (nk_f32_t)n;
    nk_f32_t centroid_a_x = sum_a_x * inv_n, centroid_a_y = sum_a_y * inv_n, centroid_a_z = sum_a_z * inv_n;
    nk_f32_t centroid_b_x = sum_b_x * inv_n, centroid_b_y = sum_b_y * inv_n, centroid_b_z = sum_b_z * inv_n;

    if (a_centroid) a_centroid[0] = centroid_a_x, a_centroid[1] = centroid_a_y, a_centroid[2] = centroid_a_z;
    if (b_centroid) b_centroid[0] = centroid_b_x, b_centroid[1] = centroid_b_y, b_centroid[2] = centroid_b_z;

    // Compute centered covariance and variance
    nk_f32_t variance_a = variance_a_sum * inv_n -
                          (centroid_a_x * centroid_a_x + centroid_a_y * centroid_a_y + centroid_a_z * centroid_a_z);

    // Apply centering correction to covariance matrix
    h00 -= n * centroid_a_x * centroid_b_x;
    h01 -= n * centroid_a_x * centroid_b_y;
    h02 -= n * centroid_a_x * centroid_b_z;
    h10 -= n * centroid_a_y * centroid_b_x;
    h11 -= n * centroid_a_y * centroid_b_y;
    h12 -= n * centroid_a_y * centroid_b_z;
    h20 -= n * centroid_a_z * centroid_b_x;
    h21 -= n * centroid_a_z * centroid_b_y;
    h22 -= n * centroid_a_z * centroid_b_z;

    nk_f32_t cross_covariance[9] = {h00, h01, h02, h10, h11, h12, h20, h21, h22};

    // SVD
    nk_f32_t svd_u[9], svd_s[9], svd_v[9];
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

    // Scale factor: c = trace(D × S) / (n × variance(a))
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
    nk_f32_t sum_squared = nk_transformed_ssd_bf16_haswell_(a, b, n, r, c, centroid_a_x, centroid_a_y, centroid_a_z,
                                                            centroid_b_x, centroid_b_y, centroid_b_z);
    *result = nk_sqrt_f32_haswell_(sum_squared * inv_n);
}

#if defined(__cplusplus)
} // extern "C"
#endif

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // NK_TARGET_HASWELL
#endif // NK_TARGET_X86_

#endif // NK_MESH_HASWELL_H

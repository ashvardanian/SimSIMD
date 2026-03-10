/**
 *  @brief SIMD-accelerated Point Cloud Alignment for WASM Relaxed SIMD.
 *  @file include/numkong/mesh/v128relaxed.h
 *  @author Ash Vardanian
 *  @date March 10, 2026
 *
 *  @sa include/numkong/mesh.h
 *
 *  @section wasm_mesh_instructions Key WASM Relaxed SIMD Mesh Instructions
 *
 *  Point cloud operations use these WASM Relaxed SIMD instructions:
 *
 *      Intrinsic                       Operation
 *      wasm_f32x4_relaxed_madd         Fused multiply-add (4-way f32)
 *      wasm_f64x2_relaxed_madd         Fused multiply-add (2-way f64)
 *      wasm_f32x4_mul                  Multiply (4-way f32)
 *      wasm_f64x2_mul                  Multiply (2-way f64)
 *      wasm_f32x4_add/sub              Add/subtract (4-way f32)
 *      wasm_f64x2_add/sub              Add/subtract (2-way f64)
 *      wasm_f32x4_splat                Broadcast scalar to all lanes
 *      wasm_f64x2_splat                Broadcast scalar to all lanes
 *      wasm_i32x4_shuffle              Cross-vector lane permutation (f32)
 *      wasm_i64x2_shuffle              Cross-vector lane permutation (f64)
 *
 *  WASM lacks hardware stride-3 deinterleaving (no LD3 equivalent), so XYZ
 *  deinterleaving is done via shuffle chains. No dual-accumulator unrolling is
 *  used since WASM engines already handle instruction scheduling.
 */
#ifndef NK_MESH_V128RELAXED_H
#define NK_MESH_V128RELAXED_H

#if NK_TARGET_V128RELAXED

#include "numkong/types.h"
#include "numkong/reduce/v128relaxed.h"
#include "numkong/mesh/serial.h"

#include <math.h>

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("relaxed-simd"))), apply_to = function)
#endif

/*  Deinterleave 12 contiguous f32 values (4 XYZ triplets) into separate x, y, z vectors.
 *
 *  Input:  [x0,y0,z0, x1,y1,z1, x2,y2,z2, x3,y3,z3]
 *  Output: x[4], y[4], z[4] vectors
 */
NK_INTERNAL void nk_deinterleave_f32x4_v128relaxed_(nk_f32_t const *ptr, v128_t *xs_f32x4, v128_t *ys_f32x4,
                                                    v128_t *zs_f32x4) {
    v128_t v0_f32x4 = wasm_v128_load(ptr);     // x0 y0 z0 x1
    v128_t v1_f32x4 = wasm_v128_load(ptr + 4); // y1 z1 x2 y2
    v128_t v2_f32x4 = wasm_v128_load(ptr + 8); // z2 x3 y3 z3
    // x0 x1 x2 x3
    v128_t tmp01 = wasm_i32x4_shuffle(v0_f32x4, v1_f32x4, 0, 3, 6, 0); // x0 x1 x2 _
    *xs_f32x4 = wasm_i32x4_shuffle(tmp01, v2_f32x4, 0, 1, 2, 5);       // x0 x1 x2 x3
    // y0 y1 y2 y3
    v128_t tmp23 = wasm_i32x4_shuffle(v0_f32x4, v1_f32x4, 1, 4, 7, 0); // y0 y1 y2 _
    *ys_f32x4 = wasm_i32x4_shuffle(tmp23, v2_f32x4, 0, 1, 2, 6);       // y0 y1 y2 y3
    // z0 z1 z2 z3
    v128_t tmp45 = wasm_i32x4_shuffle(v0_f32x4, v1_f32x4, 2, 5, 0, 0); // z0 z1 _ _
    *zs_f32x4 = wasm_i32x4_shuffle(tmp45, v2_f32x4, 0, 1, 4, 7);       // z0 z1 z2 z3
}

/*  Deinterleave 6 contiguous f64 values (2 XYZ triplets) into separate x, y, z vectors.
 *
 *  Input:  [x0,y0,z0, x1,y1,z1]
 *  Output: x[2], y[2], z[2] vectors
 */
NK_INTERNAL void nk_deinterleave_f64x2_v128relaxed_(nk_f64_t const *ptr, v128_t *xs_f64x2, v128_t *ys_f64x2,
                                                    v128_t *zs_f64x2) {
    v128_t v0_f64x2 = wasm_v128_load(ptr);                    // x0 y0
    v128_t v1_f64x2 = wasm_v128_load(ptr + 2);                // z0 x1
    v128_t v2_f64x2 = wasm_v128_load(ptr + 4);                // y1 z1
    *xs_f64x2 = wasm_i64x2_shuffle(v0_f64x2, v1_f64x2, 0, 3); // x0 x1
    *ys_f64x2 = wasm_i64x2_shuffle(v0_f64x2, v2_f64x2, 1, 2); // y0 y1
    *zs_f64x2 = wasm_i64x2_shuffle(v1_f64x2, v2_f64x2, 0, 3); // z0 z1
}

/* Horizontal sum of all 4 f32 lanes. */
NK_INTERNAL nk_f32_t nk_hsum_f32x4_v128relaxed_(v128_t v) {
    return wasm_f32x4_extract_lane(v, 0) + wasm_f32x4_extract_lane(v, 1) + wasm_f32x4_extract_lane(v, 2) +
           wasm_f32x4_extract_lane(v, 3);
}

/* Horizontal sum of both f64 lanes. */
NK_INTERNAL nk_f64_t nk_hsum_f64x2_v128relaxed_(v128_t v) {
    return wasm_f64x2_extract_lane(v, 0) + wasm_f64x2_extract_lane(v, 1);
}

/*  Compute sum of squared distances for f32 after applying rotation (and optional scale).
 *  Used by kabsch (scale=1.0) and umeyama (scale=computed_scale).
 *  Returns sum_squared, caller computes sqrt(sum_squared / n).
 */
NK_INTERNAL nk_f32_t nk_transformed_ssd_f32_v128relaxed_(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n,
                                                         nk_f32_t const *r, nk_f32_t scale, nk_f32_t centroid_a_x,
                                                         nk_f32_t centroid_a_y, nk_f32_t centroid_a_z,
                                                         nk_f32_t centroid_b_x, nk_f32_t centroid_b_y,
                                                         nk_f32_t centroid_b_z) {
    // Broadcast scaled rotation matrix elements
    v128_t sr0_f32x4 = wasm_f32x4_splat(scale * r[0]), sr1_f32x4 = wasm_f32x4_splat(scale * r[1]),
           sr2_f32x4 = wasm_f32x4_splat(scale * r[2]);
    v128_t sr3_f32x4 = wasm_f32x4_splat(scale * r[3]), sr4_f32x4 = wasm_f32x4_splat(scale * r[4]),
           sr5_f32x4 = wasm_f32x4_splat(scale * r[5]);
    v128_t sr6_f32x4 = wasm_f32x4_splat(scale * r[6]), sr7_f32x4 = wasm_f32x4_splat(scale * r[7]),
           sr8_f32x4 = wasm_f32x4_splat(scale * r[8]);

    // Broadcast centroids
    v128_t centroid_a_x_f32x4 = wasm_f32x4_splat(centroid_a_x);
    v128_t centroid_a_y_f32x4 = wasm_f32x4_splat(centroid_a_y);
    v128_t centroid_a_z_f32x4 = wasm_f32x4_splat(centroid_a_z);
    v128_t centroid_b_x_f32x4 = wasm_f32x4_splat(centroid_b_x);
    v128_t centroid_b_y_f32x4 = wasm_f32x4_splat(centroid_b_y);
    v128_t centroid_b_z_f32x4 = wasm_f32x4_splat(centroid_b_z);

    v128_t sum_squared_f32x4 = wasm_f32x4_splat(0);
    nk_size_t j = 0;

    // Main loop: process 4 points per iteration
    for (; j + 4 <= n; j += 4) {
        v128_t a_x, a_y, a_z, b_x, b_y, b_z;
        nk_deinterleave_f32x4_v128relaxed_(a + j * 3, &a_x, &a_y, &a_z);
        nk_deinterleave_f32x4_v128relaxed_(b + j * 3, &b_x, &b_y, &b_z);

        v128_t pa_x = wasm_f32x4_sub(a_x, centroid_a_x_f32x4);
        v128_t pa_y = wasm_f32x4_sub(a_y, centroid_a_y_f32x4);
        v128_t pa_z = wasm_f32x4_sub(a_z, centroid_a_z_f32x4);
        v128_t pb_x = wasm_f32x4_sub(b_x, centroid_b_x_f32x4);
        v128_t pb_y = wasm_f32x4_sub(b_y, centroid_b_y_f32x4);
        v128_t pb_z = wasm_f32x4_sub(b_z, centroid_b_z_f32x4);

        // Rotate and scale: ra = scale * R * pa
        v128_t ra_x = wasm_f32x4_relaxed_madd(
            sr2_f32x4, pa_z, wasm_f32x4_relaxed_madd(sr1_f32x4, pa_y, wasm_f32x4_mul(sr0_f32x4, pa_x)));
        v128_t ra_y = wasm_f32x4_relaxed_madd(
            sr5_f32x4, pa_z, wasm_f32x4_relaxed_madd(sr4_f32x4, pa_y, wasm_f32x4_mul(sr3_f32x4, pa_x)));
        v128_t ra_z = wasm_f32x4_relaxed_madd(
            sr8_f32x4, pa_z, wasm_f32x4_relaxed_madd(sr7_f32x4, pa_y, wasm_f32x4_mul(sr6_f32x4, pa_x)));

        v128_t delta_x = wasm_f32x4_sub(ra_x, pb_x);
        v128_t delta_y = wasm_f32x4_sub(ra_y, pb_y);
        v128_t delta_z = wasm_f32x4_sub(ra_z, pb_z);

        sum_squared_f32x4 = wasm_f32x4_relaxed_madd(delta_x, delta_x, sum_squared_f32x4);
        sum_squared_f32x4 = wasm_f32x4_relaxed_madd(delta_y, delta_y, sum_squared_f32x4);
        sum_squared_f32x4 = wasm_f32x4_relaxed_madd(delta_z, delta_z, sum_squared_f32x4);
    }

    nk_f32_t sum_squared = nk_hsum_f32x4_v128relaxed_(sum_squared_f32x4);

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

/*  Compute sum of squared distances for f64 after applying rotation (and optional scale). */
NK_INTERNAL nk_f64_t nk_transformed_ssd_f64_v128relaxed_(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n,
                                                         nk_f64_t const *r, nk_f64_t scale, nk_f64_t centroid_a_x,
                                                         nk_f64_t centroid_a_y, nk_f64_t centroid_a_z,
                                                         nk_f64_t centroid_b_x, nk_f64_t centroid_b_y,
                                                         nk_f64_t centroid_b_z) {
    // Broadcast scaled rotation matrix elements
    v128_t sr0_f64x2 = wasm_f64x2_splat(scale * r[0]), sr1_f64x2 = wasm_f64x2_splat(scale * r[1]),
           sr2_f64x2 = wasm_f64x2_splat(scale * r[2]);
    v128_t sr3_f64x2 = wasm_f64x2_splat(scale * r[3]), sr4_f64x2 = wasm_f64x2_splat(scale * r[4]),
           sr5_f64x2 = wasm_f64x2_splat(scale * r[5]);
    v128_t sr6_f64x2 = wasm_f64x2_splat(scale * r[6]), sr7_f64x2 = wasm_f64x2_splat(scale * r[7]),
           sr8_f64x2 = wasm_f64x2_splat(scale * r[8]);

    // Broadcast centroids
    v128_t centroid_a_x_f64x2 = wasm_f64x2_splat(centroid_a_x);
    v128_t centroid_a_y_f64x2 = wasm_f64x2_splat(centroid_a_y);
    v128_t centroid_a_z_f64x2 = wasm_f64x2_splat(centroid_a_z);
    v128_t centroid_b_x_f64x2 = wasm_f64x2_splat(centroid_b_x);
    v128_t centroid_b_y_f64x2 = wasm_f64x2_splat(centroid_b_y);
    v128_t centroid_b_z_f64x2 = wasm_f64x2_splat(centroid_b_z);

    v128_t sum_squared_f64x2 = wasm_f64x2_splat(0);
    nk_size_t j = 0;

    // Main loop: process 2 points per iteration
    for (; j + 2 <= n; j += 2) {
        v128_t a_x, a_y, a_z, b_x, b_y, b_z;
        nk_deinterleave_f64x2_v128relaxed_(a + j * 3, &a_x, &a_y, &a_z);
        nk_deinterleave_f64x2_v128relaxed_(b + j * 3, &b_x, &b_y, &b_z);

        v128_t pa_x = wasm_f64x2_sub(a_x, centroid_a_x_f64x2);
        v128_t pa_y = wasm_f64x2_sub(a_y, centroid_a_y_f64x2);
        v128_t pa_z = wasm_f64x2_sub(a_z, centroid_a_z_f64x2);
        v128_t pb_x = wasm_f64x2_sub(b_x, centroid_b_x_f64x2);
        v128_t pb_y = wasm_f64x2_sub(b_y, centroid_b_y_f64x2);
        v128_t pb_z = wasm_f64x2_sub(b_z, centroid_b_z_f64x2);

        // Rotate and scale: ra = scale * R * pa
        v128_t ra_x = wasm_f64x2_relaxed_madd(
            sr2_f64x2, pa_z, wasm_f64x2_relaxed_madd(sr1_f64x2, pa_y, wasm_f64x2_mul(sr0_f64x2, pa_x)));
        v128_t ra_y = wasm_f64x2_relaxed_madd(
            sr5_f64x2, pa_z, wasm_f64x2_relaxed_madd(sr4_f64x2, pa_y, wasm_f64x2_mul(sr3_f64x2, pa_x)));
        v128_t ra_z = wasm_f64x2_relaxed_madd(
            sr8_f64x2, pa_z, wasm_f64x2_relaxed_madd(sr7_f64x2, pa_y, wasm_f64x2_mul(sr6_f64x2, pa_x)));

        v128_t delta_x = wasm_f64x2_sub(ra_x, pb_x);
        v128_t delta_y = wasm_f64x2_sub(ra_y, pb_y);
        v128_t delta_z = wasm_f64x2_sub(ra_z, pb_z);

        sum_squared_f64x2 = wasm_f64x2_relaxed_madd(delta_x, delta_x, sum_squared_f64x2);
        sum_squared_f64x2 = wasm_f64x2_relaxed_madd(delta_y, delta_y, sum_squared_f64x2);
        sum_squared_f64x2 = wasm_f64x2_relaxed_madd(delta_z, delta_z, sum_squared_f64x2);
    }

    nk_f64_t sum_squared = nk_hsum_f64x2_v128relaxed_(sum_squared_f64x2);

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

NK_PUBLIC void nk_rmsd_f32_v128relaxed(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                       nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result) {
    /* RMSD uses identity rotation and scale=1.0 */
    if (rotation) {
        rotation[0] = 1, rotation[1] = 0, rotation[2] = 0;
        rotation[3] = 0, rotation[4] = 1, rotation[5] = 0;
        rotation[6] = 0, rotation[7] = 0, rotation[8] = 1;
    }
    if (scale) *scale = 1.0f;

    v128_t const zeros_f32x4 = wasm_f32x4_splat(0);

    // Accumulators for centroids and squared differences
    v128_t sum_a_x_f32x4 = zeros_f32x4, sum_a_y_f32x4 = zeros_f32x4, sum_a_z_f32x4 = zeros_f32x4;
    v128_t sum_b_x_f32x4 = zeros_f32x4, sum_b_y_f32x4 = zeros_f32x4, sum_b_z_f32x4 = zeros_f32x4;
    v128_t sum_squared_x_f32x4 = zeros_f32x4, sum_squared_y_f32x4 = zeros_f32x4, sum_squared_z_f32x4 = zeros_f32x4;

    v128_t a_x_f32x4, a_y_f32x4, a_z_f32x4, b_x_f32x4, b_y_f32x4, b_z_f32x4;
    nk_size_t i = 0;

    // Main loop processing 4 points at a time
    for (; i + 4 <= n; i += 4) {
        nk_deinterleave_f32x4_v128relaxed_(a + i * 3, &a_x_f32x4, &a_y_f32x4, &a_z_f32x4);
        nk_deinterleave_f32x4_v128relaxed_(b + i * 3, &b_x_f32x4, &b_y_f32x4, &b_z_f32x4);

        sum_a_x_f32x4 = wasm_f32x4_add(sum_a_x_f32x4, a_x_f32x4);
        sum_a_y_f32x4 = wasm_f32x4_add(sum_a_y_f32x4, a_y_f32x4);
        sum_a_z_f32x4 = wasm_f32x4_add(sum_a_z_f32x4, a_z_f32x4);
        sum_b_x_f32x4 = wasm_f32x4_add(sum_b_x_f32x4, b_x_f32x4);
        sum_b_y_f32x4 = wasm_f32x4_add(sum_b_y_f32x4, b_y_f32x4);
        sum_b_z_f32x4 = wasm_f32x4_add(sum_b_z_f32x4, b_z_f32x4);

        v128_t delta_x_f32x4 = wasm_f32x4_sub(a_x_f32x4, b_x_f32x4);
        v128_t delta_y_f32x4 = wasm_f32x4_sub(a_y_f32x4, b_y_f32x4);
        v128_t delta_z_f32x4 = wasm_f32x4_sub(a_z_f32x4, b_z_f32x4);

        sum_squared_x_f32x4 = wasm_f32x4_relaxed_madd(delta_x_f32x4, delta_x_f32x4, sum_squared_x_f32x4);
        sum_squared_y_f32x4 = wasm_f32x4_relaxed_madd(delta_y_f32x4, delta_y_f32x4, sum_squared_y_f32x4);
        sum_squared_z_f32x4 = wasm_f32x4_relaxed_madd(delta_z_f32x4, delta_z_f32x4, sum_squared_z_f32x4);
    }

    // Reduce vectors to scalars
    nk_f32_t total_ax = nk_hsum_f32x4_v128relaxed_(sum_a_x_f32x4);
    nk_f32_t total_ay = nk_hsum_f32x4_v128relaxed_(sum_a_y_f32x4);
    nk_f32_t total_az = nk_hsum_f32x4_v128relaxed_(sum_a_z_f32x4);
    nk_f32_t total_bx = nk_hsum_f32x4_v128relaxed_(sum_b_x_f32x4);
    nk_f32_t total_by = nk_hsum_f32x4_v128relaxed_(sum_b_y_f32x4);
    nk_f32_t total_bz = nk_hsum_f32x4_v128relaxed_(sum_b_z_f32x4);
    nk_f32_t total_squared_x = nk_hsum_f32x4_v128relaxed_(sum_squared_x_f32x4);
    nk_f32_t total_squared_y = nk_hsum_f32x4_v128relaxed_(sum_squared_y_f32x4);
    nk_f32_t total_squared_z = nk_hsum_f32x4_v128relaxed_(sum_squared_z_f32x4);

    // Scalar tail
    for (; i < n; ++i) {
        nk_f32_t ax = a[i * 3 + 0], ay = a[i * 3 + 1], az = a[i * 3 + 2];
        nk_f32_t bx = b[i * 3 + 0], by = b[i * 3 + 1], bz = b[i * 3 + 2];
        total_ax += ax, total_ay += ay, total_az += az;
        total_bx += bx, total_by += by, total_bz += bz;
        nk_f32_t delta_x = ax - bx, delta_y = ay - by, delta_z = az - bz;
        total_squared_x += delta_x * delta_x;
        total_squared_y += delta_y * delta_y;
        total_squared_z += delta_z * delta_z;
    }

    // Compute centroids
    nk_f32_t inv_n = 1.0f / (nk_f32_t)n;
    nk_f32_t centroid_a_x = total_ax * inv_n, centroid_a_y = total_ay * inv_n, centroid_a_z = total_az * inv_n;
    nk_f32_t centroid_b_x = total_bx * inv_n, centroid_b_y = total_by * inv_n, centroid_b_z = total_bz * inv_n;
    if (a_centroid) a_centroid[0] = centroid_a_x, a_centroid[1] = centroid_a_y, a_centroid[2] = centroid_a_z;
    if (b_centroid) b_centroid[0] = centroid_b_x, b_centroid[1] = centroid_b_y, b_centroid[2] = centroid_b_z;

    // Compute RMSD
    nk_f32_t mean_diff_x = centroid_a_x - centroid_b_x;
    nk_f32_t mean_diff_y = centroid_a_y - centroid_b_y;
    nk_f32_t mean_diff_z = centroid_a_z - centroid_b_z;
    nk_f32_t sum_squared = total_squared_x + total_squared_y + total_squared_z;
    nk_f32_t mean_diff_sq = mean_diff_x * mean_diff_x + mean_diff_y * mean_diff_y + mean_diff_z * mean_diff_z;

    *result = sqrtf(sum_squared * inv_n - mean_diff_sq);
}

NK_PUBLIC void nk_rmsd_f64_v128relaxed(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                                       nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result) {
    /* RMSD uses identity rotation and scale=1.0 */
    if (rotation) {
        rotation[0] = 1, rotation[1] = 0, rotation[2] = 0;
        rotation[3] = 0, rotation[4] = 1, rotation[5] = 0;
        rotation[6] = 0, rotation[7] = 0, rotation[8] = 1;
    }
    if (scale) *scale = 1.0;

    v128_t const zeros_f64x2 = wasm_f64x2_splat(0);

    // Accumulators for centroids and squared differences
    v128_t sum_a_x_f64x2 = zeros_f64x2, sum_a_y_f64x2 = zeros_f64x2, sum_a_z_f64x2 = zeros_f64x2;
    v128_t sum_b_x_f64x2 = zeros_f64x2, sum_b_y_f64x2 = zeros_f64x2, sum_b_z_f64x2 = zeros_f64x2;
    v128_t sum_squared_x_f64x2 = zeros_f64x2, sum_squared_y_f64x2 = zeros_f64x2, sum_squared_z_f64x2 = zeros_f64x2;

    v128_t a_x_f64x2, a_y_f64x2, a_z_f64x2, b_x_f64x2, b_y_f64x2, b_z_f64x2;
    nk_size_t i = 0;

    // Main loop processing 2 points at a time
    for (; i + 2 <= n; i += 2) {
        nk_deinterleave_f64x2_v128relaxed_(a + i * 3, &a_x_f64x2, &a_y_f64x2, &a_z_f64x2);
        nk_deinterleave_f64x2_v128relaxed_(b + i * 3, &b_x_f64x2, &b_y_f64x2, &b_z_f64x2);

        sum_a_x_f64x2 = wasm_f64x2_add(sum_a_x_f64x2, a_x_f64x2);
        sum_a_y_f64x2 = wasm_f64x2_add(sum_a_y_f64x2, a_y_f64x2);
        sum_a_z_f64x2 = wasm_f64x2_add(sum_a_z_f64x2, a_z_f64x2);
        sum_b_x_f64x2 = wasm_f64x2_add(sum_b_x_f64x2, b_x_f64x2);
        sum_b_y_f64x2 = wasm_f64x2_add(sum_b_y_f64x2, b_y_f64x2);
        sum_b_z_f64x2 = wasm_f64x2_add(sum_b_z_f64x2, b_z_f64x2);

        v128_t delta_x_f64x2 = wasm_f64x2_sub(a_x_f64x2, b_x_f64x2);
        v128_t delta_y_f64x2 = wasm_f64x2_sub(a_y_f64x2, b_y_f64x2);
        v128_t delta_z_f64x2 = wasm_f64x2_sub(a_z_f64x2, b_z_f64x2);

        sum_squared_x_f64x2 = wasm_f64x2_relaxed_madd(delta_x_f64x2, delta_x_f64x2, sum_squared_x_f64x2);
        sum_squared_y_f64x2 = wasm_f64x2_relaxed_madd(delta_y_f64x2, delta_y_f64x2, sum_squared_y_f64x2);
        sum_squared_z_f64x2 = wasm_f64x2_relaxed_madd(delta_z_f64x2, delta_z_f64x2, sum_squared_z_f64x2);
    }

    // Reduce vectors to scalars
    nk_f64_t total_ax = nk_hsum_f64x2_v128relaxed_(sum_a_x_f64x2);
    nk_f64_t total_ay = nk_hsum_f64x2_v128relaxed_(sum_a_y_f64x2);
    nk_f64_t total_az = nk_hsum_f64x2_v128relaxed_(sum_a_z_f64x2);
    nk_f64_t total_bx = nk_hsum_f64x2_v128relaxed_(sum_b_x_f64x2);
    nk_f64_t total_by = nk_hsum_f64x2_v128relaxed_(sum_b_y_f64x2);
    nk_f64_t total_bz = nk_hsum_f64x2_v128relaxed_(sum_b_z_f64x2);
    nk_f64_t total_squared_x = nk_hsum_f64x2_v128relaxed_(sum_squared_x_f64x2);
    nk_f64_t total_squared_y = nk_hsum_f64x2_v128relaxed_(sum_squared_y_f64x2);
    nk_f64_t total_squared_z = nk_hsum_f64x2_v128relaxed_(sum_squared_z_f64x2);

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
    nk_f64_t centroid_a_x = total_ax * inv_n, centroid_a_y = total_ay * inv_n, centroid_a_z = total_az * inv_n;
    nk_f64_t centroid_b_x = total_bx * inv_n, centroid_b_y = total_by * inv_n, centroid_b_z = total_bz * inv_n;
    if (a_centroid) a_centroid[0] = centroid_a_x, a_centroid[1] = centroid_a_y, a_centroid[2] = centroid_a_z;
    if (b_centroid) b_centroid[0] = centroid_b_x, b_centroid[1] = centroid_b_y, b_centroid[2] = centroid_b_z;

    // Compute RMSD
    nk_f64_t mean_diff_x = centroid_a_x - centroid_b_x;
    nk_f64_t mean_diff_y = centroid_a_y - centroid_b_y;
    nk_f64_t mean_diff_z = centroid_a_z - centroid_b_z;
    nk_f64_t sum_squared = total_squared_x + total_squared_y + total_squared_z;
    nk_f64_t mean_diff_sq = mean_diff_x * mean_diff_x + mean_diff_y * mean_diff_y + mean_diff_z * mean_diff_z;

    *result = sqrt(sum_squared * inv_n - mean_diff_sq);
}

NK_PUBLIC void nk_kabsch_f32_v128relaxed(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                         nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result) {
    v128_t const zeros_f32x4 = wasm_f32x4_splat(0);

    // Single set of accumulators (1x unrolling)
    v128_t sum_a_x_f32x4 = zeros_f32x4, sum_a_y_f32x4 = zeros_f32x4, sum_a_z_f32x4 = zeros_f32x4;
    v128_t sum_b_x_f32x4 = zeros_f32x4, sum_b_y_f32x4 = zeros_f32x4, sum_b_z_f32x4 = zeros_f32x4;

    v128_t cov_xx_f32x4 = zeros_f32x4, cov_xy_f32x4 = zeros_f32x4, cov_xz_f32x4 = zeros_f32x4;
    v128_t cov_yx_f32x4 = zeros_f32x4, cov_yy_f32x4 = zeros_f32x4, cov_yz_f32x4 = zeros_f32x4;
    v128_t cov_zx_f32x4 = zeros_f32x4, cov_zy_f32x4 = zeros_f32x4, cov_zz_f32x4 = zeros_f32x4;

    nk_size_t i = 0;
    v128_t a_x_f32x4, a_y_f32x4, a_z_f32x4, b_x_f32x4, b_y_f32x4, b_z_f32x4;

    // Main loop: 4 points per iteration
    for (; i + 4 <= n; i += 4) {
        nk_deinterleave_f32x4_v128relaxed_(a + i * 3, &a_x_f32x4, &a_y_f32x4, &a_z_f32x4);
        nk_deinterleave_f32x4_v128relaxed_(b + i * 3, &b_x_f32x4, &b_y_f32x4, &b_z_f32x4);

        sum_a_x_f32x4 = wasm_f32x4_add(sum_a_x_f32x4, a_x_f32x4);
        sum_a_y_f32x4 = wasm_f32x4_add(sum_a_y_f32x4, a_y_f32x4);
        sum_a_z_f32x4 = wasm_f32x4_add(sum_a_z_f32x4, a_z_f32x4);
        sum_b_x_f32x4 = wasm_f32x4_add(sum_b_x_f32x4, b_x_f32x4);
        sum_b_y_f32x4 = wasm_f32x4_add(sum_b_y_f32x4, b_y_f32x4);
        sum_b_z_f32x4 = wasm_f32x4_add(sum_b_z_f32x4, b_z_f32x4);

        cov_xx_f32x4 = wasm_f32x4_relaxed_madd(a_x_f32x4, b_x_f32x4, cov_xx_f32x4);
        cov_xy_f32x4 = wasm_f32x4_relaxed_madd(a_x_f32x4, b_y_f32x4, cov_xy_f32x4);
        cov_xz_f32x4 = wasm_f32x4_relaxed_madd(a_x_f32x4, b_z_f32x4, cov_xz_f32x4);
        cov_yx_f32x4 = wasm_f32x4_relaxed_madd(a_y_f32x4, b_x_f32x4, cov_yx_f32x4);
        cov_yy_f32x4 = wasm_f32x4_relaxed_madd(a_y_f32x4, b_y_f32x4, cov_yy_f32x4);
        cov_yz_f32x4 = wasm_f32x4_relaxed_madd(a_y_f32x4, b_z_f32x4, cov_yz_f32x4);
        cov_zx_f32x4 = wasm_f32x4_relaxed_madd(a_z_f32x4, b_x_f32x4, cov_zx_f32x4);
        cov_zy_f32x4 = wasm_f32x4_relaxed_madd(a_z_f32x4, b_y_f32x4, cov_zy_f32x4);
        cov_zz_f32x4 = wasm_f32x4_relaxed_madd(a_z_f32x4, b_z_f32x4, cov_zz_f32x4);
    }

    // Reduce vector accumulators
    nk_f32_t sum_a_x = nk_hsum_f32x4_v128relaxed_(sum_a_x_f32x4);
    nk_f32_t sum_a_y = nk_hsum_f32x4_v128relaxed_(sum_a_y_f32x4);
    nk_f32_t sum_a_z = nk_hsum_f32x4_v128relaxed_(sum_a_z_f32x4);
    nk_f32_t sum_b_x = nk_hsum_f32x4_v128relaxed_(sum_b_x_f32x4);
    nk_f32_t sum_b_y = nk_hsum_f32x4_v128relaxed_(sum_b_y_f32x4);
    nk_f32_t sum_b_z = nk_hsum_f32x4_v128relaxed_(sum_b_z_f32x4);

    nk_f32_t h00 = nk_hsum_f32x4_v128relaxed_(cov_xx_f32x4);
    nk_f32_t h01 = nk_hsum_f32x4_v128relaxed_(cov_xy_f32x4);
    nk_f32_t h02 = nk_hsum_f32x4_v128relaxed_(cov_xz_f32x4);
    nk_f32_t h10 = nk_hsum_f32x4_v128relaxed_(cov_yx_f32x4);
    nk_f32_t h11 = nk_hsum_f32x4_v128relaxed_(cov_yy_f32x4);
    nk_f32_t h12 = nk_hsum_f32x4_v128relaxed_(cov_yz_f32x4);
    nk_f32_t h20 = nk_hsum_f32x4_v128relaxed_(cov_zx_f32x4);
    nk_f32_t h21 = nk_hsum_f32x4_v128relaxed_(cov_zy_f32x4);
    nk_f32_t h22 = nk_hsum_f32x4_v128relaxed_(cov_zz_f32x4);

    // Scalar tail
    for (; i < n; ++i) {
        nk_f32_t ax = a[i * 3 + 0], ay = a[i * 3 + 1], az = a[i * 3 + 2];
        nk_f32_t bx = b[i * 3 + 0], by = b[i * 3 + 1], bz = b[i * 3 + 2];
        sum_a_x += ax, sum_a_y += ay, sum_a_z += az;
        sum_b_x += bx, sum_b_y += by, sum_b_z += bz;
        h00 += ax * bx, h01 += ax * by, h02 += ax * bz;
        h10 += ay * bx, h11 += ay * by, h12 += ay * bz;
        h20 += az * bx, h21 += az * by, h22 += az * bz;
    }

    // Compute centroids
    nk_f32_t inv_n = 1.0f / (nk_f32_t)n;
    nk_f32_t centroid_a_x = sum_a_x * inv_n, centroid_a_y = sum_a_y * inv_n, centroid_a_z = sum_a_z * inv_n;
    nk_f32_t centroid_b_x = sum_b_x * inv_n, centroid_b_y = sum_b_y * inv_n, centroid_b_z = sum_b_z * inv_n;
    if (a_centroid) a_centroid[0] = centroid_a_x, a_centroid[1] = centroid_a_y, a_centroid[2] = centroid_a_z;
    if (b_centroid) b_centroid[0] = centroid_b_x, b_centroid[1] = centroid_b_y, b_centroid[2] = centroid_b_z;

    // Apply centering correction: H_centered = H - n * centroid_a * centroid_bT
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

    // R = V * UT
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
    nk_f32_t sum_squared = nk_transformed_ssd_f32_v128relaxed_(a, b, n, r, 1.0f, centroid_a_x, centroid_a_y,
                                                               centroid_a_z, centroid_b_x, centroid_b_y, centroid_b_z);
    *result = sqrtf(sum_squared * inv_n);
}

NK_PUBLIC void nk_kabsch_f64_v128relaxed(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                                         nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result) {
    v128_t const zeros_f64x2 = wasm_f64x2_splat(0);

    // Single set of accumulators (1x unrolling)
    v128_t sum_a_x_f64x2 = zeros_f64x2, sum_a_y_f64x2 = zeros_f64x2, sum_a_z_f64x2 = zeros_f64x2;
    v128_t sum_b_x_f64x2 = zeros_f64x2, sum_b_y_f64x2 = zeros_f64x2, sum_b_z_f64x2 = zeros_f64x2;

    v128_t cov_xx_f64x2 = zeros_f64x2, cov_xy_f64x2 = zeros_f64x2, cov_xz_f64x2 = zeros_f64x2;
    v128_t cov_yx_f64x2 = zeros_f64x2, cov_yy_f64x2 = zeros_f64x2, cov_yz_f64x2 = zeros_f64x2;
    v128_t cov_zx_f64x2 = zeros_f64x2, cov_zy_f64x2 = zeros_f64x2, cov_zz_f64x2 = zeros_f64x2;

    nk_size_t i = 0;
    v128_t a_x_f64x2, a_y_f64x2, a_z_f64x2, b_x_f64x2, b_y_f64x2, b_z_f64x2;

    // Main loop: 2 points per iteration
    for (; i + 2 <= n; i += 2) {
        nk_deinterleave_f64x2_v128relaxed_(a + i * 3, &a_x_f64x2, &a_y_f64x2, &a_z_f64x2);
        nk_deinterleave_f64x2_v128relaxed_(b + i * 3, &b_x_f64x2, &b_y_f64x2, &b_z_f64x2);

        sum_a_x_f64x2 = wasm_f64x2_add(sum_a_x_f64x2, a_x_f64x2);
        sum_a_y_f64x2 = wasm_f64x2_add(sum_a_y_f64x2, a_y_f64x2);
        sum_a_z_f64x2 = wasm_f64x2_add(sum_a_z_f64x2, a_z_f64x2);
        sum_b_x_f64x2 = wasm_f64x2_add(sum_b_x_f64x2, b_x_f64x2);
        sum_b_y_f64x2 = wasm_f64x2_add(sum_b_y_f64x2, b_y_f64x2);
        sum_b_z_f64x2 = wasm_f64x2_add(sum_b_z_f64x2, b_z_f64x2);

        cov_xx_f64x2 = wasm_f64x2_relaxed_madd(a_x_f64x2, b_x_f64x2, cov_xx_f64x2);
        cov_xy_f64x2 = wasm_f64x2_relaxed_madd(a_x_f64x2, b_y_f64x2, cov_xy_f64x2);
        cov_xz_f64x2 = wasm_f64x2_relaxed_madd(a_x_f64x2, b_z_f64x2, cov_xz_f64x2);
        cov_yx_f64x2 = wasm_f64x2_relaxed_madd(a_y_f64x2, b_x_f64x2, cov_yx_f64x2);
        cov_yy_f64x2 = wasm_f64x2_relaxed_madd(a_y_f64x2, b_y_f64x2, cov_yy_f64x2);
        cov_yz_f64x2 = wasm_f64x2_relaxed_madd(a_y_f64x2, b_z_f64x2, cov_yz_f64x2);
        cov_zx_f64x2 = wasm_f64x2_relaxed_madd(a_z_f64x2, b_x_f64x2, cov_zx_f64x2);
        cov_zy_f64x2 = wasm_f64x2_relaxed_madd(a_z_f64x2, b_y_f64x2, cov_zy_f64x2);
        cov_zz_f64x2 = wasm_f64x2_relaxed_madd(a_z_f64x2, b_z_f64x2, cov_zz_f64x2);
    }

    // Reduce vector accumulators
    nk_f64_t sum_a_x = nk_hsum_f64x2_v128relaxed_(sum_a_x_f64x2);
    nk_f64_t sum_a_y = nk_hsum_f64x2_v128relaxed_(sum_a_y_f64x2);
    nk_f64_t sum_a_z = nk_hsum_f64x2_v128relaxed_(sum_a_z_f64x2);
    nk_f64_t sum_b_x = nk_hsum_f64x2_v128relaxed_(sum_b_x_f64x2);
    nk_f64_t sum_b_y = nk_hsum_f64x2_v128relaxed_(sum_b_y_f64x2);
    nk_f64_t sum_b_z = nk_hsum_f64x2_v128relaxed_(sum_b_z_f64x2);

    nk_f64_t h00 = nk_hsum_f64x2_v128relaxed_(cov_xx_f64x2);
    nk_f64_t h01 = nk_hsum_f64x2_v128relaxed_(cov_xy_f64x2);
    nk_f64_t h02 = nk_hsum_f64x2_v128relaxed_(cov_xz_f64x2);
    nk_f64_t h10 = nk_hsum_f64x2_v128relaxed_(cov_yx_f64x2);
    nk_f64_t h11 = nk_hsum_f64x2_v128relaxed_(cov_yy_f64x2);
    nk_f64_t h12 = nk_hsum_f64x2_v128relaxed_(cov_yz_f64x2);
    nk_f64_t h20 = nk_hsum_f64x2_v128relaxed_(cov_zx_f64x2);
    nk_f64_t h21 = nk_hsum_f64x2_v128relaxed_(cov_zy_f64x2);
    nk_f64_t h22 = nk_hsum_f64x2_v128relaxed_(cov_zz_f64x2);

    // Scalar tail
    for (; i < n; ++i) {
        nk_f64_t ax = a[i * 3 + 0], ay = a[i * 3 + 1], az = a[i * 3 + 2];
        nk_f64_t bx = b[i * 3 + 0], by = b[i * 3 + 1], bz = b[i * 3 + 2];
        sum_a_x += ax, sum_a_y += ay, sum_a_z += az;
        sum_b_x += bx, sum_b_y += by, sum_b_z += bz;
        h00 += ax * bx, h01 += ax * by, h02 += ax * bz;
        h10 += ay * bx, h11 += ay * by, h12 += ay * bz;
        h20 += az * bx, h21 += az * by, h22 += az * bz;
    }

    // Compute centroids
    nk_f64_t inv_n = 1.0 / (nk_f64_t)n;
    nk_f64_t centroid_a_x = sum_a_x * inv_n, centroid_a_y = sum_a_y * inv_n, centroid_a_z = sum_a_z * inv_n;
    nk_f64_t centroid_b_x = sum_b_x * inv_n, centroid_b_y = sum_b_y * inv_n, centroid_b_z = sum_b_z * inv_n;
    if (a_centroid) a_centroid[0] = centroid_a_x, a_centroid[1] = centroid_a_y, a_centroid[2] = centroid_a_z;
    if (b_centroid) b_centroid[0] = centroid_b_x, b_centroid[1] = centroid_b_y, b_centroid[2] = centroid_b_z;

    // Apply centering correction: H_centered = H - n * centroid_a * centroid_bT
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
    nk_f64_t cross_covariance[9] = {h00, h01, h02, h10, h11, h12, h20, h21, h22};
    nk_f64_t svd_u[9], svd_s[9], svd_v[9];
    nk_svd3x3_f64_(cross_covariance, svd_u, svd_s, svd_v);

    // R = V * UT
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
    if (rotation)
        for (int j = 0; j < 9; ++j) rotation[j] = r[j];

    if (scale) *scale = 1.0;

    // Compute RMSD after optimal rotation
    nk_f64_t sum_squared = nk_transformed_ssd_f64_v128relaxed_(a, b, n, r, 1.0, centroid_a_x, centroid_a_y,
                                                               centroid_a_z, centroid_b_x, centroid_b_y, centroid_b_z);
    *result = sqrt(sum_squared * inv_n);
}

NK_PUBLIC void nk_umeyama_f32_v128relaxed(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                          nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result) {
    v128_t const zeros_f32x4 = wasm_f32x4_splat(0);

    // Single set of accumulators (1x unrolling)
    v128_t sum_a_x_f32x4 = zeros_f32x4, sum_a_y_f32x4 = zeros_f32x4, sum_a_z_f32x4 = zeros_f32x4;
    v128_t sum_b_x_f32x4 = zeros_f32x4, sum_b_y_f32x4 = zeros_f32x4, sum_b_z_f32x4 = zeros_f32x4;

    v128_t cov_xx_f32x4 = zeros_f32x4, cov_xy_f32x4 = zeros_f32x4, cov_xz_f32x4 = zeros_f32x4;
    v128_t cov_yx_f32x4 = zeros_f32x4, cov_yy_f32x4 = zeros_f32x4, cov_yz_f32x4 = zeros_f32x4;
    v128_t cov_zx_f32x4 = zeros_f32x4, cov_zy_f32x4 = zeros_f32x4, cov_zz_f32x4 = zeros_f32x4;
    v128_t variance_a_f32x4 = zeros_f32x4;

    nk_size_t i = 0;
    v128_t a_x_f32x4, a_y_f32x4, a_z_f32x4, b_x_f32x4, b_y_f32x4, b_z_f32x4;

    // Main loop: 4 points per iteration
    for (; i + 4 <= n; i += 4) {
        nk_deinterleave_f32x4_v128relaxed_(a + i * 3, &a_x_f32x4, &a_y_f32x4, &a_z_f32x4);
        nk_deinterleave_f32x4_v128relaxed_(b + i * 3, &b_x_f32x4, &b_y_f32x4, &b_z_f32x4);

        sum_a_x_f32x4 = wasm_f32x4_add(sum_a_x_f32x4, a_x_f32x4);
        sum_a_y_f32x4 = wasm_f32x4_add(sum_a_y_f32x4, a_y_f32x4);
        sum_a_z_f32x4 = wasm_f32x4_add(sum_a_z_f32x4, a_z_f32x4);
        sum_b_x_f32x4 = wasm_f32x4_add(sum_b_x_f32x4, b_x_f32x4);
        sum_b_y_f32x4 = wasm_f32x4_add(sum_b_y_f32x4, b_y_f32x4);
        sum_b_z_f32x4 = wasm_f32x4_add(sum_b_z_f32x4, b_z_f32x4);

        cov_xx_f32x4 = wasm_f32x4_relaxed_madd(a_x_f32x4, b_x_f32x4, cov_xx_f32x4);
        cov_xy_f32x4 = wasm_f32x4_relaxed_madd(a_x_f32x4, b_y_f32x4, cov_xy_f32x4);
        cov_xz_f32x4 = wasm_f32x4_relaxed_madd(a_x_f32x4, b_z_f32x4, cov_xz_f32x4);
        cov_yx_f32x4 = wasm_f32x4_relaxed_madd(a_y_f32x4, b_x_f32x4, cov_yx_f32x4);
        cov_yy_f32x4 = wasm_f32x4_relaxed_madd(a_y_f32x4, b_y_f32x4, cov_yy_f32x4);
        cov_yz_f32x4 = wasm_f32x4_relaxed_madd(a_y_f32x4, b_z_f32x4, cov_yz_f32x4);
        cov_zx_f32x4 = wasm_f32x4_relaxed_madd(a_z_f32x4, b_x_f32x4, cov_zx_f32x4);
        cov_zy_f32x4 = wasm_f32x4_relaxed_madd(a_z_f32x4, b_y_f32x4, cov_zy_f32x4);
        cov_zz_f32x4 = wasm_f32x4_relaxed_madd(a_z_f32x4, b_z_f32x4, cov_zz_f32x4);

        variance_a_f32x4 = wasm_f32x4_relaxed_madd(a_x_f32x4, a_x_f32x4, variance_a_f32x4);
        variance_a_f32x4 = wasm_f32x4_relaxed_madd(a_y_f32x4, a_y_f32x4, variance_a_f32x4);
        variance_a_f32x4 = wasm_f32x4_relaxed_madd(a_z_f32x4, a_z_f32x4, variance_a_f32x4);
    }

    // Reduce vector accumulators
    nk_f32_t sum_a_x = nk_hsum_f32x4_v128relaxed_(sum_a_x_f32x4);
    nk_f32_t sum_a_y = nk_hsum_f32x4_v128relaxed_(sum_a_y_f32x4);
    nk_f32_t sum_a_z = nk_hsum_f32x4_v128relaxed_(sum_a_z_f32x4);
    nk_f32_t sum_b_x = nk_hsum_f32x4_v128relaxed_(sum_b_x_f32x4);
    nk_f32_t sum_b_y = nk_hsum_f32x4_v128relaxed_(sum_b_y_f32x4);
    nk_f32_t sum_b_z = nk_hsum_f32x4_v128relaxed_(sum_b_z_f32x4);
    nk_f32_t h00 = nk_hsum_f32x4_v128relaxed_(cov_xx_f32x4);
    nk_f32_t h01 = nk_hsum_f32x4_v128relaxed_(cov_xy_f32x4);
    nk_f32_t h02 = nk_hsum_f32x4_v128relaxed_(cov_xz_f32x4);
    nk_f32_t h10 = nk_hsum_f32x4_v128relaxed_(cov_yx_f32x4);
    nk_f32_t h11 = nk_hsum_f32x4_v128relaxed_(cov_yy_f32x4);
    nk_f32_t h12 = nk_hsum_f32x4_v128relaxed_(cov_yz_f32x4);
    nk_f32_t h20 = nk_hsum_f32x4_v128relaxed_(cov_zx_f32x4);
    nk_f32_t h21 = nk_hsum_f32x4_v128relaxed_(cov_zy_f32x4);
    nk_f32_t h22 = nk_hsum_f32x4_v128relaxed_(cov_zz_f32x4);
    nk_f32_t sum_sq_a = nk_hsum_f32x4_v128relaxed_(variance_a_f32x4);

    // Scalar tail
    for (; i < n; ++i) {
        nk_f32_t ax = a[i * 3 + 0], ay = a[i * 3 + 1], az = a[i * 3 + 2];
        nk_f32_t bx = b[i * 3 + 0], by = b[i * 3 + 1], bz = b[i * 3 + 2];
        sum_a_x += ax, sum_a_y += ay, sum_a_z += az;
        sum_b_x += bx, sum_b_y += by, sum_b_z += bz;
        h00 += ax * bx, h01 += ax * by, h02 += ax * bz;
        h10 += ay * bx, h11 += ay * by, h12 += ay * bz;
        h20 += az * bx, h21 += az * by, h22 += az * bz;
        sum_sq_a += ax * ax + ay * ay + az * az;
    }

    // Compute centroids
    nk_f32_t inv_n = 1.0f / (nk_f32_t)n;
    nk_f32_t centroid_a_x = sum_a_x * inv_n, centroid_a_y = sum_a_y * inv_n, centroid_a_z = sum_a_z * inv_n;
    nk_f32_t centroid_b_x = sum_b_x * inv_n, centroid_b_y = sum_b_y * inv_n, centroid_b_z = sum_b_z * inv_n;
    if (a_centroid) a_centroid[0] = centroid_a_x, a_centroid[1] = centroid_a_y, a_centroid[2] = centroid_a_z;
    if (b_centroid) b_centroid[0] = centroid_b_x, b_centroid[1] = centroid_b_y, b_centroid[2] = centroid_b_z;

    // Compute variance of A (centered)
    nk_f32_t centroid_sq = centroid_a_x * centroid_a_x + centroid_a_y * centroid_a_y + centroid_a_z * centroid_a_z;
    nk_f32_t var_a = sum_sq_a * inv_n - centroid_sq;

    // Apply centering correction: H_centered = H - n * centroid_a * centroid_bT
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
    nk_f32_t svd_u[9], svd_s[9], svd_v[9];
    nk_svd3x3_f32_(cross_covariance, svd_u, svd_s, svd_v);

    // R = V * UT
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
    nk_f32_t trace_d_s = svd_s[0] + svd_s[4] + (det < 0 ? -svd_s[8] : svd_s[8]);
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

    if (rotation)
        for (int j = 0; j < 9; ++j) rotation[j] = r[j];
    if (scale) *scale = computed_scale;

    // Compute RMSD after transformation
    nk_f32_t sum_squared = nk_transformed_ssd_f32_v128relaxed_(a, b, n, r, computed_scale, centroid_a_x, centroid_a_y,
                                                               centroid_a_z, centroid_b_x, centroid_b_y, centroid_b_z);
    *result = sqrtf(sum_squared * inv_n);
}

NK_PUBLIC void nk_umeyama_f64_v128relaxed(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                                          nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result) {
    v128_t const zeros_f64x2 = wasm_f64x2_splat(0);

    // Single set of accumulators (1x unrolling)
    v128_t sum_a_x_f64x2 = zeros_f64x2, sum_a_y_f64x2 = zeros_f64x2, sum_a_z_f64x2 = zeros_f64x2;
    v128_t sum_b_x_f64x2 = zeros_f64x2, sum_b_y_f64x2 = zeros_f64x2, sum_b_z_f64x2 = zeros_f64x2;

    v128_t cov_xx_f64x2 = zeros_f64x2, cov_xy_f64x2 = zeros_f64x2, cov_xz_f64x2 = zeros_f64x2;
    v128_t cov_yx_f64x2 = zeros_f64x2, cov_yy_f64x2 = zeros_f64x2, cov_yz_f64x2 = zeros_f64x2;
    v128_t cov_zx_f64x2 = zeros_f64x2, cov_zy_f64x2 = zeros_f64x2, cov_zz_f64x2 = zeros_f64x2;
    v128_t variance_a_f64x2 = zeros_f64x2;

    nk_size_t i = 0;
    v128_t a_x_f64x2, a_y_f64x2, a_z_f64x2, b_x_f64x2, b_y_f64x2, b_z_f64x2;

    // Main loop: 2 points per iteration
    for (; i + 2 <= n; i += 2) {
        nk_deinterleave_f64x2_v128relaxed_(a + i * 3, &a_x_f64x2, &a_y_f64x2, &a_z_f64x2);
        nk_deinterleave_f64x2_v128relaxed_(b + i * 3, &b_x_f64x2, &b_y_f64x2, &b_z_f64x2);

        sum_a_x_f64x2 = wasm_f64x2_add(sum_a_x_f64x2, a_x_f64x2);
        sum_a_y_f64x2 = wasm_f64x2_add(sum_a_y_f64x2, a_y_f64x2);
        sum_a_z_f64x2 = wasm_f64x2_add(sum_a_z_f64x2, a_z_f64x2);
        sum_b_x_f64x2 = wasm_f64x2_add(sum_b_x_f64x2, b_x_f64x2);
        sum_b_y_f64x2 = wasm_f64x2_add(sum_b_y_f64x2, b_y_f64x2);
        sum_b_z_f64x2 = wasm_f64x2_add(sum_b_z_f64x2, b_z_f64x2);

        cov_xx_f64x2 = wasm_f64x2_relaxed_madd(a_x_f64x2, b_x_f64x2, cov_xx_f64x2);
        cov_xy_f64x2 = wasm_f64x2_relaxed_madd(a_x_f64x2, b_y_f64x2, cov_xy_f64x2);
        cov_xz_f64x2 = wasm_f64x2_relaxed_madd(a_x_f64x2, b_z_f64x2, cov_xz_f64x2);
        cov_yx_f64x2 = wasm_f64x2_relaxed_madd(a_y_f64x2, b_x_f64x2, cov_yx_f64x2);
        cov_yy_f64x2 = wasm_f64x2_relaxed_madd(a_y_f64x2, b_y_f64x2, cov_yy_f64x2);
        cov_yz_f64x2 = wasm_f64x2_relaxed_madd(a_y_f64x2, b_z_f64x2, cov_yz_f64x2);
        cov_zx_f64x2 = wasm_f64x2_relaxed_madd(a_z_f64x2, b_x_f64x2, cov_zx_f64x2);
        cov_zy_f64x2 = wasm_f64x2_relaxed_madd(a_z_f64x2, b_y_f64x2, cov_zy_f64x2);
        cov_zz_f64x2 = wasm_f64x2_relaxed_madd(a_z_f64x2, b_z_f64x2, cov_zz_f64x2);

        variance_a_f64x2 = wasm_f64x2_relaxed_madd(a_x_f64x2, a_x_f64x2, variance_a_f64x2);
        variance_a_f64x2 = wasm_f64x2_relaxed_madd(a_y_f64x2, a_y_f64x2, variance_a_f64x2);
        variance_a_f64x2 = wasm_f64x2_relaxed_madd(a_z_f64x2, a_z_f64x2, variance_a_f64x2);
    }

    // Reduce vector accumulators
    nk_f64_t sum_a_x = nk_hsum_f64x2_v128relaxed_(sum_a_x_f64x2);
    nk_f64_t sum_a_y = nk_hsum_f64x2_v128relaxed_(sum_a_y_f64x2);
    nk_f64_t sum_a_z = nk_hsum_f64x2_v128relaxed_(sum_a_z_f64x2);
    nk_f64_t sum_b_x = nk_hsum_f64x2_v128relaxed_(sum_b_x_f64x2);
    nk_f64_t sum_b_y = nk_hsum_f64x2_v128relaxed_(sum_b_y_f64x2);
    nk_f64_t sum_b_z = nk_hsum_f64x2_v128relaxed_(sum_b_z_f64x2);
    nk_f64_t h00 = nk_hsum_f64x2_v128relaxed_(cov_xx_f64x2);
    nk_f64_t h01 = nk_hsum_f64x2_v128relaxed_(cov_xy_f64x2);
    nk_f64_t h02 = nk_hsum_f64x2_v128relaxed_(cov_xz_f64x2);
    nk_f64_t h10 = nk_hsum_f64x2_v128relaxed_(cov_yx_f64x2);
    nk_f64_t h11 = nk_hsum_f64x2_v128relaxed_(cov_yy_f64x2);
    nk_f64_t h12 = nk_hsum_f64x2_v128relaxed_(cov_yz_f64x2);
    nk_f64_t h20 = nk_hsum_f64x2_v128relaxed_(cov_zx_f64x2);
    nk_f64_t h21 = nk_hsum_f64x2_v128relaxed_(cov_zy_f64x2);
    nk_f64_t h22 = nk_hsum_f64x2_v128relaxed_(cov_zz_f64x2);
    nk_f64_t sum_sq_a = nk_hsum_f64x2_v128relaxed_(variance_a_f64x2);

    // Scalar tail
    for (; i < n; ++i) {
        nk_f64_t ax = a[i * 3 + 0], ay = a[i * 3 + 1], az = a[i * 3 + 2];
        nk_f64_t bx = b[i * 3 + 0], by = b[i * 3 + 1], bz = b[i * 3 + 2];
        sum_a_x += ax, sum_a_y += ay, sum_a_z += az;
        sum_b_x += bx, sum_b_y += by, sum_b_z += bz;
        h00 += ax * bx, h01 += ax * by, h02 += ax * bz;
        h10 += ay * bx, h11 += ay * by, h12 += ay * bz;
        h20 += az * bx, h21 += az * by, h22 += az * bz;
        sum_sq_a += ax * ax + ay * ay + az * az;
    }

    // Compute centroids
    nk_f64_t inv_n = 1.0 / (nk_f64_t)n;
    nk_f64_t centroid_a_x = sum_a_x * inv_n, centroid_a_y = sum_a_y * inv_n, centroid_a_z = sum_a_z * inv_n;
    nk_f64_t centroid_b_x = sum_b_x * inv_n, centroid_b_y = sum_b_y * inv_n, centroid_b_z = sum_b_z * inv_n;
    if (a_centroid) a_centroid[0] = centroid_a_x, a_centroid[1] = centroid_a_y, a_centroid[2] = centroid_a_z;
    if (b_centroid) b_centroid[0] = centroid_b_x, b_centroid[1] = centroid_b_y, b_centroid[2] = centroid_b_z;

    // Compute variance of A (centered)
    nk_f64_t centroid_sq = centroid_a_x * centroid_a_x + centroid_a_y * centroid_a_y + centroid_a_z * centroid_a_z;
    nk_f64_t var_a = sum_sq_a * inv_n - centroid_sq;

    // Apply centering correction: H_centered = H - n * centroid_a * centroid_bT
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
    nk_f64_t cross_covariance[9] = {h00, h01, h02, h10, h11, h12, h20, h21, h22};
    nk_f64_t svd_u[9], svd_s[9], svd_v[9];
    nk_svd3x3_f64_(cross_covariance, svd_u, svd_s, svd_v);

    // R = V * UT
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

    // Handle reflection and compute scale
    nk_f64_t det = nk_det3x3_f64_(r);
    nk_f64_t trace_d_s = svd_s[0] + svd_s[4] + (det < 0 ? -svd_s[8] : svd_s[8]);
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

    if (rotation)
        for (int j = 0; j < 9; ++j) rotation[j] = r[j];
    if (scale) *scale = computed_scale;

    // Compute RMSD after transformation
    nk_f64_t sum_squared = nk_transformed_ssd_f64_v128relaxed_(a, b, n, r, computed_scale, centroid_a_x, centroid_a_y,
                                                               centroid_a_z, centroid_b_x, centroid_b_y, centroid_b_z);
    *result = sqrt(sum_squared * inv_n);
}

#if defined(__clang__)
#pragma clang attribute pop
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_V128RELAXED
#endif // NK_MESH_V128RELAXED_H

/**
 *  @brief SIMD-accelerated Point Cloud Alignment for Skylake.
 *  @file include/numkong/mesh/skylake.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 *
 *  @sa include/numkong/mesh.h
 *
 *  @section skylake_mesh_instructions Key AVX-512 Mesh Instructions
 *
 *      Intrinsic                   Instruction                     Latency     Throughput  Ports
 *      _mm512_fmadd_ps             VFMADD132PS (ZMM, ZMM, ZMM)     4cy         0.5/cy      p05
 *      _mm512_permutexvar_ps       VPERMPS (ZMM, ZMM, ZMM)         3cy         1/cy        p5
 *      _mm512_permutex2var_ps      VPERMT2PS (ZMM, ZMM, ZMM)       3cy         1/cy        p5
 *      _mm512_extractf32x8_ps      VEXTRACTF32X8 (YMM, ZMM, I8)    3cy         1/cy        p5
 *
 *  Point cloud operations use VPERMT2PS for stride-3 deinterleaving of xyz coordinates, avoiding
 *  expensive gather instructions. This achieves ~1.8x speedup over scalar deinterleaving. Dual FMA
 *  accumulators on Skylake-X server chips hide the 4cy latency for centroid and covariance computation.
 */
#ifndef NK_MESH_SKYLAKE_H
#define NK_MESH_SKYLAKE_H

#if NK_TARGET_X86_
#if NK_TARGET_SKYLAKE

#include "numkong/types.h"
#include "numkong/dot/skylake.h"
#include "numkong/mesh/serial.h"
#include "numkong/spatial/haswell.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,avx512f,avx512vl,avx512bw,avx512dq,f16c,fma,bmi,bmi2"))), \
                             apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "avx512bw", "avx512dq", "f16c", "fma", "bmi", "bmi2")
#endif

/*  Deinterleave 48 floats (16 xyz triplets) into separate x, y, z vectors.
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

    // X: reg0[0,3,6,9,12,15] + reg1[2,5,8,11,14] → 11 elements, then + reg2[1,4,7,10,13] → 16 elements
    // Indices for permutex2var: 0-15 = from first operand, 16-31 = from second operand
    __m512i idx_x_01_i32x16 = _mm512_setr_epi32(0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 0, 0, 0, 0, 0);
    __m512i idx_x_2_i32x16 = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 17, 20, 23, 26, 29);
    __m512 x01_f32x16 = _mm512_permutex2var_ps(reg0_f32x16, idx_x_01_i32x16, reg1_f32x16);
    *x_f32x16_out = _mm512_permutex2var_ps(x01_f32x16, idx_x_2_i32x16, reg2_f32x16);

    // Y: reg0[1,4,7,10,13] + reg1[0,3,6,9,12,15] → 11 elements, then + reg2[2,5,8,11,14] → 16 elements
    __m512i idx_y_01_i32x16 = _mm512_setr_epi32(1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 0, 0, 0, 0, 0);
    __m512i idx_y_2_i32x16 = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 18, 21, 24, 27, 30);
    __m512 y01_f32x16 = _mm512_permutex2var_ps(reg0_f32x16, idx_y_01_i32x16, reg1_f32x16);
    *y_f32x16_out = _mm512_permutex2var_ps(y01_f32x16, idx_y_2_i32x16, reg2_f32x16);

    // Z: reg0[2,5,8,11,14] + reg1[1,4,7,10,13] → 10 elements, then + reg2[0,3,6,9,12,15] → 16 elements
    __m512i idx_z_01_i32x16 = _mm512_setr_epi32(2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 0, 0, 0, 0, 0, 0);
    __m512i idx_z_2_i32x16 = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 19, 22, 25, 28, 31);
    __m512 z01_f32x16 = _mm512_permutex2var_ps(reg0_f32x16, idx_z_01_i32x16, reg1_f32x16);
    *z_f32x16_out = _mm512_permutex2var_ps(z01_f32x16, idx_z_2_i32x16, reg2_f32x16);
}

/*  Deinterleave 8 f64 3D points from xyz,xyz,xyz... to separate x,y,z vectors.
 *  Input: 24 consecutive f64 values (8 points * 3 coordinates)
 *  Output: Three __m512d vectors containing the x, y, z coordinates separately.
 */
NK_INTERNAL void nk_deinterleave_f64x8_skylake_(                                             //
    nk_f64_t const *ptr, __m512d *x_f64x8_out, __m512d *y_f64x8_out, __m512d *z_f64x8_out) { //
    __m512d reg0_f64x8 = _mm512_loadu_pd(ptr);                                               // elements 0-7
    __m512d reg1_f64x8 = _mm512_loadu_pd(ptr + 8);                                           // elements 8-15
    __m512d reg2_f64x8 = _mm512_loadu_pd(ptr + 16);                                          // elements 16-23

    // X: positions 0,3,6,9,12,15,18,21 → reg0[0,3,6] + reg1[1,4,7] + reg2[2,5]
    __m512i idx_x_01_i64x8 = _mm512_setr_epi64(0, 3, 6, 9, 12, 15, 0, 0);
    __m512i idx_x_2_i64x8 = _mm512_setr_epi64(0, 1, 2, 3, 4, 5, 10, 13);
    __m512d x01_f64x8 = _mm512_permutex2var_pd(reg0_f64x8, idx_x_01_i64x8, reg1_f64x8);
    *x_f64x8_out = _mm512_permutex2var_pd(x01_f64x8, idx_x_2_i64x8, reg2_f64x8);

    // Y: positions 1,4,7,10,13,16,19,22 → reg0[1,4,7] + reg1[2,5] + reg2[0,3,6]
    __m512i idx_y_01_i64x8 = _mm512_setr_epi64(1, 4, 7, 10, 13, 0, 0, 0);
    __m512i idx_y_2_i64x8 = _mm512_setr_epi64(0, 1, 2, 3, 4, 8, 11, 14);
    __m512d y01_f64x8 = _mm512_permutex2var_pd(reg0_f64x8, idx_y_01_i64x8, reg1_f64x8);
    *y_f64x8_out = _mm512_permutex2var_pd(y01_f64x8, idx_y_2_i64x8, reg2_f64x8);

    // Z: positions 2,5,8,11,14,17,20,23 → reg0[2,5] + reg1[0,3,6] + reg2[1,4,7]
    __m512i idx_z_01_i64x8 = _mm512_setr_epi64(2, 5, 8, 11, 14, 0, 0, 0);
    __m512i idx_z_2_i64x8 = _mm512_setr_epi64(0, 1, 2, 3, 4, 9, 12, 15);
    __m512d z01_f64x8 = _mm512_permutex2var_pd(reg0_f64x8, idx_z_01_i64x8, reg1_f64x8);
    *z_f64x8_out = _mm512_permutex2var_pd(z01_f64x8, idx_z_2_i64x8, reg2_f64x8);
}

NK_INTERNAL nk_f64_t nk_reduce_stable_f64x8_skylake_(__m512d values_f64x8) {
    nk_b512_vec_t values;
    values.zmm_pd = values_f64x8;
    nk_f64_t sum = 0.0, compensation = 0.0;
    for (nk_size_t lane_index = 0; lane_index != 8; ++lane_index)
        nk_accumulate_sum_f64_(&sum, &compensation, values.f64s[lane_index]);
    return sum + compensation;
}

NK_INTERNAL void nk_rotation_from_svd_f64_skylake_(nk_f64_t const *svd_u, nk_f64_t const *svd_v, nk_f64_t *rotation) {
    nk_rotation_from_svd_f64_serial_(svd_u, svd_v, rotation);
}

NK_INTERNAL void nk_accumulate_square_f64x8_skylake_(__m512d *sum_f64x8, __m512d *compensation_f64x8,
                                                     __m512d values_f64x8) {
    __m512d product_f64x8 = _mm512_mul_pd(values_f64x8, values_f64x8);
    __m512d product_error_f64x8 = _mm512_fmsub_pd(values_f64x8, values_f64x8, product_f64x8);
    __m512d tentative_sum_f64x8 = _mm512_add_pd(*sum_f64x8, product_f64x8);
    __m512d virtual_addend_f64x8 = _mm512_sub_pd(tentative_sum_f64x8, *sum_f64x8);
    __m512d sum_error_f64x8 = _mm512_add_pd(
        _mm512_sub_pd(*sum_f64x8, _mm512_sub_pd(tentative_sum_f64x8, virtual_addend_f64x8)),
        _mm512_sub_pd(product_f64x8, virtual_addend_f64x8));
    *sum_f64x8 = tentative_sum_f64x8;
    *compensation_f64x8 = _mm512_add_pd(*compensation_f64x8, _mm512_add_pd(sum_error_f64x8, product_error_f64x8));
}

/*  Compute sum of squared distances after applying rotation (and optional scale).
 *  Used by kabsch (scale=1.0) and umeyama (scale=computed_scale).
 *  Returns sum_squared, caller computes √(sum_squared / n).
 */
NK_INTERNAL nk_f64_t nk_transformed_ssd_f32_skylake_(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n,
                                                     nk_f64_t const *r, nk_f64_t scale, nk_f64_t centroid_a_x,
                                                     nk_f64_t centroid_a_y, nk_f64_t centroid_a_z,
                                                     nk_f64_t centroid_b_x, nk_f64_t centroid_b_y,
                                                     nk_f64_t centroid_b_z) {
    __m512d scaled_rotation_x_x_f64x8 = _mm512_set1_pd(scale * r[0]);
    __m512d scaled_rotation_x_y_f64x8 = _mm512_set1_pd(scale * r[1]);
    __m512d scaled_rotation_x_z_f64x8 = _mm512_set1_pd(scale * r[2]);
    __m512d scaled_rotation_y_x_f64x8 = _mm512_set1_pd(scale * r[3]);
    __m512d scaled_rotation_y_y_f64x8 = _mm512_set1_pd(scale * r[4]);
    __m512d scaled_rotation_y_z_f64x8 = _mm512_set1_pd(scale * r[5]);
    __m512d scaled_rotation_z_x_f64x8 = _mm512_set1_pd(scale * r[6]);
    __m512d scaled_rotation_z_y_f64x8 = _mm512_set1_pd(scale * r[7]);
    __m512d scaled_rotation_z_z_f64x8 = _mm512_set1_pd(scale * r[8]);
    __m512d centroid_a_x_f64x8 = _mm512_set1_pd(centroid_a_x), centroid_a_y_f64x8 = _mm512_set1_pd(centroid_a_y);
    __m512d centroid_a_z_f64x8 = _mm512_set1_pd(centroid_a_z), centroid_b_x_f64x8 = _mm512_set1_pd(centroid_b_x);
    __m512d centroid_b_y_f64x8 = _mm512_set1_pd(centroid_b_y), centroid_b_z_f64x8 = _mm512_set1_pd(centroid_b_z);
    __m512d sum_squared_f64x8 = _mm512_setzero_pd();
    __m512 a_x_f32x16, a_y_f32x16, a_z_f32x16, b_x_f32x16, b_y_f32x16, b_z_f32x16;
    nk_size_t index = 0;

    for (; index + 16 <= n; index += 16) {
        nk_deinterleave_f32x16_skylake_(a + index * 3, &a_x_f32x16, &a_y_f32x16, &a_z_f32x16),
            nk_deinterleave_f32x16_skylake_(b + index * 3, &b_x_f32x16, &b_y_f32x16, &b_z_f32x16);
        __m512d a_x_lower_f64x8 = _mm512_cvtps_pd(_mm512_castps512_ps256(a_x_f32x16));
        __m512d a_x_upper_f64x8 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(a_x_f32x16, 1));
        __m512d a_y_lower_f64x8 = _mm512_cvtps_pd(_mm512_castps512_ps256(a_y_f32x16));
        __m512d a_y_upper_f64x8 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(a_y_f32x16, 1));
        __m512d a_z_lower_f64x8 = _mm512_cvtps_pd(_mm512_castps512_ps256(a_z_f32x16));
        __m512d a_z_upper_f64x8 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(a_z_f32x16, 1));
        __m512d b_x_lower_f64x8 = _mm512_cvtps_pd(_mm512_castps512_ps256(b_x_f32x16));
        __m512d b_x_upper_f64x8 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(b_x_f32x16, 1));
        __m512d b_y_lower_f64x8 = _mm512_cvtps_pd(_mm512_castps512_ps256(b_y_f32x16));
        __m512d b_y_upper_f64x8 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(b_y_f32x16, 1));
        __m512d b_z_lower_f64x8 = _mm512_cvtps_pd(_mm512_castps512_ps256(b_z_f32x16));
        __m512d b_z_upper_f64x8 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(b_z_f32x16, 1));

        __m512d centered_a_x_lower_f64x8 = _mm512_sub_pd(a_x_lower_f64x8, centroid_a_x_f64x8);
        __m512d centered_a_x_upper_f64x8 = _mm512_sub_pd(a_x_upper_f64x8, centroid_a_x_f64x8);
        __m512d centered_a_y_lower_f64x8 = _mm512_sub_pd(a_y_lower_f64x8, centroid_a_y_f64x8);
        __m512d centered_a_y_upper_f64x8 = _mm512_sub_pd(a_y_upper_f64x8, centroid_a_y_f64x8);
        __m512d centered_a_z_lower_f64x8 = _mm512_sub_pd(a_z_lower_f64x8, centroid_a_z_f64x8);
        __m512d centered_a_z_upper_f64x8 = _mm512_sub_pd(a_z_upper_f64x8, centroid_a_z_f64x8);
        __m512d centered_b_x_lower_f64x8 = _mm512_sub_pd(b_x_lower_f64x8, centroid_b_x_f64x8);
        __m512d centered_b_x_upper_f64x8 = _mm512_sub_pd(b_x_upper_f64x8, centroid_b_x_f64x8);
        __m512d centered_b_y_lower_f64x8 = _mm512_sub_pd(b_y_lower_f64x8, centroid_b_y_f64x8);
        __m512d centered_b_y_upper_f64x8 = _mm512_sub_pd(b_y_upper_f64x8, centroid_b_y_f64x8);
        __m512d centered_b_z_lower_f64x8 = _mm512_sub_pd(b_z_lower_f64x8, centroid_b_z_f64x8);
        __m512d centered_b_z_upper_f64x8 = _mm512_sub_pd(b_z_upper_f64x8, centroid_b_z_f64x8);

        __m512d rotated_a_x_lower_f64x8 = _mm512_fmadd_pd(
            scaled_rotation_x_z_f64x8, centered_a_z_lower_f64x8,
            _mm512_fmadd_pd(scaled_rotation_x_y_f64x8, centered_a_y_lower_f64x8,
                            _mm512_mul_pd(scaled_rotation_x_x_f64x8, centered_a_x_lower_f64x8)));
        __m512d rotated_a_x_upper_f64x8 = _mm512_fmadd_pd(
            scaled_rotation_x_z_f64x8, centered_a_z_upper_f64x8,
            _mm512_fmadd_pd(scaled_rotation_x_y_f64x8, centered_a_y_upper_f64x8,
                            _mm512_mul_pd(scaled_rotation_x_x_f64x8, centered_a_x_upper_f64x8)));
        __m512d rotated_a_y_lower_f64x8 = _mm512_fmadd_pd(
            scaled_rotation_y_z_f64x8, centered_a_z_lower_f64x8,
            _mm512_fmadd_pd(scaled_rotation_y_y_f64x8, centered_a_y_lower_f64x8,
                            _mm512_mul_pd(scaled_rotation_y_x_f64x8, centered_a_x_lower_f64x8)));
        __m512d rotated_a_y_upper_f64x8 = _mm512_fmadd_pd(
            scaled_rotation_y_z_f64x8, centered_a_z_upper_f64x8,
            _mm512_fmadd_pd(scaled_rotation_y_y_f64x8, centered_a_y_upper_f64x8,
                            _mm512_mul_pd(scaled_rotation_y_x_f64x8, centered_a_x_upper_f64x8)));
        __m512d rotated_a_z_lower_f64x8 = _mm512_fmadd_pd(
            scaled_rotation_z_z_f64x8, centered_a_z_lower_f64x8,
            _mm512_fmadd_pd(scaled_rotation_z_y_f64x8, centered_a_y_lower_f64x8,
                            _mm512_mul_pd(scaled_rotation_z_x_f64x8, centered_a_x_lower_f64x8)));
        __m512d rotated_a_z_upper_f64x8 = _mm512_fmadd_pd(
            scaled_rotation_z_z_f64x8, centered_a_z_upper_f64x8,
            _mm512_fmadd_pd(scaled_rotation_z_y_f64x8, centered_a_y_upper_f64x8,
                            _mm512_mul_pd(scaled_rotation_z_x_f64x8, centered_a_x_upper_f64x8)));

        __m512d delta_x_lower_f64x8 = _mm512_sub_pd(rotated_a_x_lower_f64x8, centered_b_x_lower_f64x8);
        __m512d delta_x_upper_f64x8 = _mm512_sub_pd(rotated_a_x_upper_f64x8, centered_b_x_upper_f64x8);
        __m512d delta_y_lower_f64x8 = _mm512_sub_pd(rotated_a_y_lower_f64x8, centered_b_y_lower_f64x8);
        __m512d delta_y_upper_f64x8 = _mm512_sub_pd(rotated_a_y_upper_f64x8, centered_b_y_upper_f64x8);
        __m512d delta_z_lower_f64x8 = _mm512_sub_pd(rotated_a_z_lower_f64x8, centered_b_z_lower_f64x8);
        __m512d delta_z_upper_f64x8 = _mm512_sub_pd(rotated_a_z_upper_f64x8, centered_b_z_upper_f64x8);

        __m512d batch_sum_squared_f64x8 = _mm512_add_pd(_mm512_mul_pd(delta_x_lower_f64x8, delta_x_lower_f64x8),
                                                        _mm512_mul_pd(delta_x_upper_f64x8, delta_x_upper_f64x8));
        batch_sum_squared_f64x8 = _mm512_fmadd_pd(delta_y_lower_f64x8, delta_y_lower_f64x8, batch_sum_squared_f64x8);
        batch_sum_squared_f64x8 = _mm512_fmadd_pd(delta_y_upper_f64x8, delta_y_upper_f64x8, batch_sum_squared_f64x8);
        batch_sum_squared_f64x8 = _mm512_fmadd_pd(delta_z_lower_f64x8, delta_z_lower_f64x8, batch_sum_squared_f64x8);
        batch_sum_squared_f64x8 = _mm512_fmadd_pd(delta_z_upper_f64x8, delta_z_upper_f64x8, batch_sum_squared_f64x8);
        sum_squared_f64x8 = _mm512_add_pd(sum_squared_f64x8, batch_sum_squared_f64x8);
    }

    nk_f64_t sum_squared = _mm512_reduce_add_pd(sum_squared_f64x8);
    for (; index < n; ++index) {
        nk_f64_t centered_a_x = (nk_f64_t)a[index * 3 + 0] - centroid_a_x;
        nk_f64_t centered_a_y = (nk_f64_t)a[index * 3 + 1] - centroid_a_y;
        nk_f64_t centered_a_z = (nk_f64_t)a[index * 3 + 2] - centroid_a_z;
        nk_f64_t centered_b_x = (nk_f64_t)b[index * 3 + 0] - centroid_b_x;
        nk_f64_t centered_b_y = (nk_f64_t)b[index * 3 + 1] - centroid_b_y;
        nk_f64_t centered_b_z = (nk_f64_t)b[index * 3 + 2] - centroid_b_z;
        nk_f64_t rotated_a_x = scale * (r[0] * centered_a_x + r[1] * centered_a_y + r[2] * centered_a_z);
        nk_f64_t rotated_a_y = scale * (r[3] * centered_a_x + r[4] * centered_a_y + r[5] * centered_a_z);
        nk_f64_t rotated_a_z = scale * (r[6] * centered_a_x + r[7] * centered_a_y + r[8] * centered_a_z);
        nk_f64_t delta_x = rotated_a_x - centered_b_x, delta_y = rotated_a_y - centered_b_y,
                 delta_z = rotated_a_z - centered_b_z;
        sum_squared += delta_x * delta_x + delta_y * delta_y + delta_z * delta_z;
    }

    return sum_squared;
}

/*  Compute sum of squared distances for f64 after applying rotation (and optional scale).
 *  Rotation matrix, scale and data are all f64 for full precision.
 */
NK_INTERNAL nk_f64_t nk_transformed_ssd_f64_skylake_(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n,
                                                     nk_f64_t const *r, nk_f64_t scale, nk_f64_t centroid_a_x,
                                                     nk_f64_t centroid_a_y, nk_f64_t centroid_a_z,
                                                     nk_f64_t centroid_b_x, nk_f64_t centroid_b_y,
                                                     nk_f64_t centroid_b_z) {
    // Broadcast scaled rotation matrix elements
    __m512d scaled_rotation_x_x_f64x8 = _mm512_set1_pd(scale * r[0]);
    __m512d scaled_rotation_x_y_f64x8 = _mm512_set1_pd(scale * r[1]);
    __m512d scaled_rotation_x_z_f64x8 = _mm512_set1_pd(scale * r[2]);
    __m512d scaled_rotation_y_x_f64x8 = _mm512_set1_pd(scale * r[3]);
    __m512d scaled_rotation_y_y_f64x8 = _mm512_set1_pd(scale * r[4]);
    __m512d scaled_rotation_y_z_f64x8 = _mm512_set1_pd(scale * r[5]);
    __m512d scaled_rotation_z_x_f64x8 = _mm512_set1_pd(scale * r[6]);
    __m512d scaled_rotation_z_y_f64x8 = _mm512_set1_pd(scale * r[7]);
    __m512d scaled_rotation_z_z_f64x8 = _mm512_set1_pd(scale * r[8]);

    // Broadcast centroids
    __m512d centroid_a_x_f64x8 = _mm512_set1_pd(centroid_a_x);
    __m512d centroid_a_y_f64x8 = _mm512_set1_pd(centroid_a_y);
    __m512d centroid_a_z_f64x8 = _mm512_set1_pd(centroid_a_z);
    __m512d centroid_b_x_f64x8 = _mm512_set1_pd(centroid_b_x);
    __m512d centroid_b_y_f64x8 = _mm512_set1_pd(centroid_b_y);
    __m512d centroid_b_z_f64x8 = _mm512_set1_pd(centroid_b_z);

    __m512d sum_squared_f64x8 = _mm512_setzero_pd();
    __m512d sum_squared_compensation_f64x8 = _mm512_setzero_pd();
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
        __m512d ra_x_f64x8 = _mm512_fmadd_pd(scaled_rotation_x_z_f64x8, pa_z_f64x8,
                                             _mm512_fmadd_pd(scaled_rotation_x_y_f64x8, pa_y_f64x8,
                                                             _mm512_mul_pd(scaled_rotation_x_x_f64x8, pa_x_f64x8)));
        __m512d ra_y_f64x8 = _mm512_fmadd_pd(scaled_rotation_y_z_f64x8, pa_z_f64x8,
                                             _mm512_fmadd_pd(scaled_rotation_y_y_f64x8, pa_y_f64x8,
                                                             _mm512_mul_pd(scaled_rotation_y_x_f64x8, pa_x_f64x8)));
        __m512d ra_z_f64x8 = _mm512_fmadd_pd(scaled_rotation_z_z_f64x8, pa_z_f64x8,
                                             _mm512_fmadd_pd(scaled_rotation_z_y_f64x8, pa_y_f64x8,
                                                             _mm512_mul_pd(scaled_rotation_z_x_f64x8, pa_x_f64x8)));

        // Delta and accumulate
        __m512d delta_x_f64x8 = _mm512_sub_pd(ra_x_f64x8, pb_x_f64x8);
        __m512d delta_y_f64x8 = _mm512_sub_pd(ra_y_f64x8, pb_y_f64x8);
        __m512d delta_z_f64x8 = _mm512_sub_pd(ra_z_f64x8, pb_z_f64x8);

        nk_accumulate_square_f64x8_skylake_(&sum_squared_f64x8, &sum_squared_compensation_f64x8, delta_x_f64x8);
        nk_accumulate_square_f64x8_skylake_(&sum_squared_f64x8, &sum_squared_compensation_f64x8, delta_y_f64x8);
        nk_accumulate_square_f64x8_skylake_(&sum_squared_f64x8, &sum_squared_compensation_f64x8, delta_z_f64x8);
    }

    nk_f64_t sum_squared = nk_dot_stable_sum_f64x8_skylake_(sum_squared_f64x8, sum_squared_compensation_f64x8);
    nk_f64_t sum_squared_compensation = 0.0;

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
        nk_accumulate_square_f64_(&sum_squared, &sum_squared_compensation, delta_x);
        nk_accumulate_square_f64_(&sum_squared, &sum_squared_compensation, delta_y);
        nk_accumulate_square_f64_(&sum_squared, &sum_squared_compensation, delta_z);
    }

    return sum_squared + sum_squared_compensation;
}

NK_INTERNAL void nk_centroid_and_cross_covariance_f32_skylake_( //
    nk_f32_t const *a, nk_f32_t const *b, nk_size_t n,          //
    nk_f64_t *centroid_a_x, nk_f64_t *centroid_a_y, nk_f64_t *centroid_a_z, nk_f64_t *centroid_b_x,
    nk_f64_t *centroid_b_y, nk_f64_t *centroid_b_z, nk_f64_t cross_covariance_f64[9]) {
    __m512d sum_a_x_f64x8 = _mm512_setzero_pd(), sum_a_y_f64x8 = _mm512_setzero_pd();
    __m512d sum_a_z_f64x8 = _mm512_setzero_pd(), sum_b_x_f64x8 = _mm512_setzero_pd();
    __m512d sum_b_y_f64x8 = _mm512_setzero_pd(), sum_b_z_f64x8 = _mm512_setzero_pd();
    __m512d covariance_00_f64x8 = _mm512_setzero_pd(), covariance_01_f64x8 = _mm512_setzero_pd();
    __m512d covariance_02_f64x8 = _mm512_setzero_pd(), covariance_10_f64x8 = _mm512_setzero_pd();
    __m512d covariance_11_f64x8 = _mm512_setzero_pd(), covariance_12_f64x8 = _mm512_setzero_pd();
    __m512d covariance_20_f64x8 = _mm512_setzero_pd(), covariance_21_f64x8 = _mm512_setzero_pd();
    __m512d covariance_22_f64x8 = _mm512_setzero_pd();
    __m512 a_x_f32x16, a_y_f32x16, a_z_f32x16, b_x_f32x16, b_y_f32x16, b_z_f32x16;
    nk_size_t index = 0;

    for (; index + 16 <= n; index += 16) {
        nk_deinterleave_f32x16_skylake_(a + index * 3, &a_x_f32x16, &a_y_f32x16, &a_z_f32x16),
            nk_deinterleave_f32x16_skylake_(b + index * 3, &b_x_f32x16, &b_y_f32x16, &b_z_f32x16);
        __m512d a_x_lower_f64x8 = _mm512_cvtps_pd(_mm512_castps512_ps256(a_x_f32x16));
        __m512d a_x_upper_f64x8 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(a_x_f32x16, 1));
        __m512d a_y_lower_f64x8 = _mm512_cvtps_pd(_mm512_castps512_ps256(a_y_f32x16));
        __m512d a_y_upper_f64x8 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(a_y_f32x16, 1));
        __m512d a_z_lower_f64x8 = _mm512_cvtps_pd(_mm512_castps512_ps256(a_z_f32x16));
        __m512d a_z_upper_f64x8 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(a_z_f32x16, 1));
        __m512d b_x_lower_f64x8 = _mm512_cvtps_pd(_mm512_castps512_ps256(b_x_f32x16));
        __m512d b_x_upper_f64x8 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(b_x_f32x16, 1));
        __m512d b_y_lower_f64x8 = _mm512_cvtps_pd(_mm512_castps512_ps256(b_y_f32x16));
        __m512d b_y_upper_f64x8 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(b_y_f32x16, 1));
        __m512d b_z_lower_f64x8 = _mm512_cvtps_pd(_mm512_castps512_ps256(b_z_f32x16));
        __m512d b_z_upper_f64x8 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(b_z_f32x16, 1));

        sum_a_x_f64x8 = _mm512_add_pd(sum_a_x_f64x8, _mm512_add_pd(a_x_lower_f64x8, a_x_upper_f64x8)),
        sum_a_y_f64x8 = _mm512_add_pd(sum_a_y_f64x8, _mm512_add_pd(a_y_lower_f64x8, a_y_upper_f64x8)),
        sum_a_z_f64x8 = _mm512_add_pd(sum_a_z_f64x8, _mm512_add_pd(a_z_lower_f64x8, a_z_upper_f64x8));
        sum_b_x_f64x8 = _mm512_add_pd(sum_b_x_f64x8, _mm512_add_pd(b_x_lower_f64x8, b_x_upper_f64x8)),
        sum_b_y_f64x8 = _mm512_add_pd(sum_b_y_f64x8, _mm512_add_pd(b_y_lower_f64x8, b_y_upper_f64x8)),
        sum_b_z_f64x8 = _mm512_add_pd(sum_b_z_f64x8, _mm512_add_pd(b_z_lower_f64x8, b_z_upper_f64x8));
        covariance_00_f64x8 = _mm512_add_pd(covariance_00_f64x8,
                                            _mm512_add_pd(_mm512_mul_pd(a_x_lower_f64x8, b_x_lower_f64x8),
                                                          _mm512_mul_pd(a_x_upper_f64x8, b_x_upper_f64x8))),
        covariance_01_f64x8 = _mm512_add_pd(covariance_01_f64x8,
                                            _mm512_add_pd(_mm512_mul_pd(a_x_lower_f64x8, b_y_lower_f64x8),
                                                          _mm512_mul_pd(a_x_upper_f64x8, b_y_upper_f64x8))),
        covariance_02_f64x8 = _mm512_add_pd(covariance_02_f64x8,
                                            _mm512_add_pd(_mm512_mul_pd(a_x_lower_f64x8, b_z_lower_f64x8),
                                                          _mm512_mul_pd(a_x_upper_f64x8, b_z_upper_f64x8)));
        covariance_10_f64x8 = _mm512_add_pd(covariance_10_f64x8,
                                            _mm512_add_pd(_mm512_mul_pd(a_y_lower_f64x8, b_x_lower_f64x8),
                                                          _mm512_mul_pd(a_y_upper_f64x8, b_x_upper_f64x8))),
        covariance_11_f64x8 = _mm512_add_pd(covariance_11_f64x8,
                                            _mm512_add_pd(_mm512_mul_pd(a_y_lower_f64x8, b_y_lower_f64x8),
                                                          _mm512_mul_pd(a_y_upper_f64x8, b_y_upper_f64x8))),
        covariance_12_f64x8 = _mm512_add_pd(covariance_12_f64x8,
                                            _mm512_add_pd(_mm512_mul_pd(a_y_lower_f64x8, b_z_lower_f64x8),
                                                          _mm512_mul_pd(a_y_upper_f64x8, b_z_upper_f64x8)));
        covariance_20_f64x8 = _mm512_add_pd(covariance_20_f64x8,
                                            _mm512_add_pd(_mm512_mul_pd(a_z_lower_f64x8, b_x_lower_f64x8),
                                                          _mm512_mul_pd(a_z_upper_f64x8, b_x_upper_f64x8))),
        covariance_21_f64x8 = _mm512_add_pd(covariance_21_f64x8,
                                            _mm512_add_pd(_mm512_mul_pd(a_z_lower_f64x8, b_y_lower_f64x8),
                                                          _mm512_mul_pd(a_z_upper_f64x8, b_y_upper_f64x8))),
        covariance_22_f64x8 = _mm512_add_pd(covariance_22_f64x8,
                                            _mm512_add_pd(_mm512_mul_pd(a_z_lower_f64x8, b_z_lower_f64x8),
                                                          _mm512_mul_pd(a_z_upper_f64x8, b_z_upper_f64x8)));
    }

    nk_f64_t sum_a_x = _mm512_reduce_add_pd(sum_a_x_f64x8), sum_a_y = _mm512_reduce_add_pd(sum_a_y_f64x8),
             sum_a_z = _mm512_reduce_add_pd(sum_a_z_f64x8);
    nk_f64_t sum_b_x = _mm512_reduce_add_pd(sum_b_x_f64x8), sum_b_y = _mm512_reduce_add_pd(sum_b_y_f64x8),
             sum_b_z = _mm512_reduce_add_pd(sum_b_z_f64x8);
    nk_f64_t covariance_00 = _mm512_reduce_add_pd(covariance_00_f64x8),
             covariance_01 = _mm512_reduce_add_pd(covariance_01_f64x8),
             covariance_02 = _mm512_reduce_add_pd(covariance_02_f64x8);
    nk_f64_t covariance_10 = _mm512_reduce_add_pd(covariance_10_f64x8),
             covariance_11 = _mm512_reduce_add_pd(covariance_11_f64x8),
             covariance_12 = _mm512_reduce_add_pd(covariance_12_f64x8);
    nk_f64_t covariance_20 = _mm512_reduce_add_pd(covariance_20_f64x8),
             covariance_21 = _mm512_reduce_add_pd(covariance_21_f64x8),
             covariance_22 = _mm512_reduce_add_pd(covariance_22_f64x8);

    for (; index < n; ++index) {
        nk_f64_t a_x = a[index * 3 + 0], a_y = a[index * 3 + 1], a_z = a[index * 3 + 2];
        nk_f64_t b_x = b[index * 3 + 0], b_y = b[index * 3 + 1], b_z = b[index * 3 + 2];
        sum_a_x += a_x, sum_a_y += a_y, sum_a_z += a_z;
        sum_b_x += b_x, sum_b_y += b_y, sum_b_z += b_z;
        covariance_00 += a_x * b_x, covariance_01 += a_x * b_y, covariance_02 += a_x * b_z;
        covariance_10 += a_y * b_x, covariance_11 += a_y * b_y, covariance_12 += a_y * b_z;
        covariance_20 += a_z * b_x, covariance_21 += a_z * b_y, covariance_22 += a_z * b_z;
    }

    nk_f64_t inv_n = 1.0 / (nk_f64_t)n, n_f64 = (nk_f64_t)n;
    *centroid_a_x = sum_a_x * inv_n, *centroid_a_y = sum_a_y * inv_n, *centroid_a_z = sum_a_z * inv_n;
    *centroid_b_x = sum_b_x * inv_n, *centroid_b_y = sum_b_y * inv_n, *centroid_b_z = sum_b_z * inv_n;
    cross_covariance_f64[0] = covariance_00 - n_f64 * (*centroid_a_x) * (*centroid_b_x),
    cross_covariance_f64[1] = covariance_01 - n_f64 * (*centroid_a_x) * (*centroid_b_y),
    cross_covariance_f64[2] = covariance_02 - n_f64 * (*centroid_a_x) * (*centroid_b_z);
    cross_covariance_f64[3] = covariance_10 - n_f64 * (*centroid_a_y) * (*centroid_b_x),
    cross_covariance_f64[4] = covariance_11 - n_f64 * (*centroid_a_y) * (*centroid_b_y),
    cross_covariance_f64[5] = covariance_12 - n_f64 * (*centroid_a_y) * (*centroid_b_z);
    cross_covariance_f64[6] = covariance_20 - n_f64 * (*centroid_a_z) * (*centroid_b_x),
    cross_covariance_f64[7] = covariance_21 - n_f64 * (*centroid_a_z) * (*centroid_b_y),
    cross_covariance_f64[8] = covariance_22 - n_f64 * (*centroid_a_z) * (*centroid_b_z);
}

NK_PUBLIC void nk_rmsd_f32_skylake(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                   nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f64_t *result) {
    nk_f64_t identity[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    nk_f64_t centroid_a_x, centroid_a_y, centroid_a_z, centroid_b_x, centroid_b_y, centroid_b_z;
    nk_f64_t cross_covariance_f64[9];
    if (rotation)
        rotation[0] = 1, rotation[1] = 0, rotation[2] = 0, rotation[3] = 0, rotation[4] = 1, rotation[5] = 0,
        rotation[6] = 0, rotation[7] = 0, rotation[8] = 1;
    if (scale) *scale = 1.0f;
    nk_centroid_and_cross_covariance_f32_skylake_(a, b, n, &centroid_a_x, &centroid_a_y, &centroid_a_z, &centroid_b_x,
                                                  &centroid_b_y, &centroid_b_z, cross_covariance_f64);
    if (a_centroid)
        a_centroid[0] = (nk_f32_t)centroid_a_x, a_centroid[1] = (nk_f32_t)centroid_a_y,
        a_centroid[2] = (nk_f32_t)centroid_a_z;
    if (b_centroid)
        b_centroid[0] = (nk_f32_t)centroid_b_x, b_centroid[1] = (nk_f32_t)centroid_b_y,
        b_centroid[2] = (nk_f32_t)centroid_b_z;
    *result = nk_f64_sqrt_haswell(nk_transformed_ssd_f32_skylake_(a, b, n, identity, 1.0, centroid_a_x, centroid_a_y,
                                                                  centroid_a_z, centroid_b_x, centroid_b_y,
                                                                  centroid_b_z) /
                                  (nk_f64_t)n);
}

NK_PUBLIC void nk_kabsch_f32_skylake(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                     nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f64_t *result) {
    nk_f64_t centroid_a_x, centroid_a_y, centroid_a_z, centroid_b_x, centroid_b_y, centroid_b_z;
    nk_f64_t cross_covariance_f64[9];
    nk_centroid_and_cross_covariance_f32_skylake_(a, b, n, &centroid_a_x, &centroid_a_y, &centroid_a_z, &centroid_b_x,
                                                  &centroid_b_y, &centroid_b_z, cross_covariance_f64);
    if (a_centroid)
        a_centroid[0] = (nk_f32_t)centroid_a_x, a_centroid[1] = (nk_f32_t)centroid_a_y,
        a_centroid[2] = (nk_f32_t)centroid_a_z;
    if (b_centroid)
        b_centroid[0] = (nk_f32_t)centroid_b_x, b_centroid[1] = (nk_f32_t)centroid_b_y,
        b_centroid[2] = (nk_f32_t)centroid_b_z;
    if (scale) *scale = 1.0f;

    nk_f64_t svd_u[9], svd_s[9], svd_v[9], r[9];
    nk_svd3x3_f64_(cross_covariance_f64, svd_u, svd_s, svd_v);
    r[0] = svd_v[0] * svd_u[0] + svd_v[1] * svd_u[1] + svd_v[2] * svd_u[2];
    r[1] = svd_v[0] * svd_u[3] + svd_v[1] * svd_u[4] + svd_v[2] * svd_u[5];
    r[2] = svd_v[0] * svd_u[6] + svd_v[1] * svd_u[7] + svd_v[2] * svd_u[8];
    r[3] = svd_v[3] * svd_u[0] + svd_v[4] * svd_u[1] + svd_v[5] * svd_u[2];
    r[4] = svd_v[3] * svd_u[3] + svd_v[4] * svd_u[4] + svd_v[5] * svd_u[5];
    r[5] = svd_v[3] * svd_u[6] + svd_v[4] * svd_u[7] + svd_v[5] * svd_u[8];
    r[6] = svd_v[6] * svd_u[0] + svd_v[7] * svd_u[1] + svd_v[8] * svd_u[2];
    r[7] = svd_v[6] * svd_u[3] + svd_v[7] * svd_u[4] + svd_v[8] * svd_u[5];
    r[8] = svd_v[6] * svd_u[6] + svd_v[7] * svd_u[7] + svd_v[8] * svd_u[8];
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
    if (rotation)
        for (int index = 0; index != 9; ++index) rotation[index] = (nk_f32_t)r[index];
    *result = nk_f64_sqrt_haswell(nk_transformed_ssd_f32_skylake_(a, b, n, r, 1.0, centroid_a_x, centroid_a_y,
                                                                  centroid_a_z, centroid_b_x, centroid_b_y,
                                                                  centroid_b_z) /
                                  (nk_f64_t)n);
}

NK_PUBLIC void nk_rmsd_f64_skylake(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                                   nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result) {
    // RMSD uses identity rotation and scale=1.0.
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
    //   RMSD = √(E[(a-ā) - (b-b̄)]²)
    //        = √(E[(a-b)²] - (ā - b̄)²)
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

    // Reduce and compute centroids.
    nk_f64_t inv_n = 1.0 / (nk_f64_t)n;
    nk_f64_t total_ax = nk_reduce_stable_f64x8_skylake_(sum_a_x_f64x8), total_ax_compensation = 0.0;
    nk_f64_t total_ay = nk_reduce_stable_f64x8_skylake_(sum_a_y_f64x8), total_ay_compensation = 0.0;
    nk_f64_t total_az = nk_reduce_stable_f64x8_skylake_(sum_a_z_f64x8), total_az_compensation = 0.0;
    nk_f64_t total_bx = nk_reduce_stable_f64x8_skylake_(sum_b_x_f64x8), total_bx_compensation = 0.0;
    nk_f64_t total_by = nk_reduce_stable_f64x8_skylake_(sum_b_y_f64x8), total_by_compensation = 0.0;
    nk_f64_t total_bz = nk_reduce_stable_f64x8_skylake_(sum_b_z_f64x8), total_bz_compensation = 0.0;
    nk_f64_t total_squared_x = nk_reduce_stable_f64x8_skylake_(sum_squared_x_f64x8), total_squared_x_compensation = 0.0;
    nk_f64_t total_squared_y = nk_reduce_stable_f64x8_skylake_(sum_squared_y_f64x8), total_squared_y_compensation = 0.0;
    nk_f64_t total_squared_z = nk_reduce_stable_f64x8_skylake_(sum_squared_z_f64x8), total_squared_z_compensation = 0.0;

    for (; i < n; ++i) {
        nk_f64_t ax = a[i * 3 + 0], ay = a[i * 3 + 1], az = a[i * 3 + 2];
        nk_f64_t bx = b[i * 3 + 0], by = b[i * 3 + 1], bz = b[i * 3 + 2];
        nk_accumulate_sum_f64_(&total_ax, &total_ax_compensation, ax);
        nk_accumulate_sum_f64_(&total_ay, &total_ay_compensation, ay);
        nk_accumulate_sum_f64_(&total_az, &total_az_compensation, az);
        nk_accumulate_sum_f64_(&total_bx, &total_bx_compensation, bx);
        nk_accumulate_sum_f64_(&total_by, &total_by_compensation, by);
        nk_accumulate_sum_f64_(&total_bz, &total_bz_compensation, bz);
        nk_f64_t delta_x = ax - bx, delta_y = ay - by, delta_z = az - bz;
        nk_accumulate_square_f64_(&total_squared_x, &total_squared_x_compensation, delta_x);
        nk_accumulate_square_f64_(&total_squared_y, &total_squared_y_compensation, delta_y);
        nk_accumulate_square_f64_(&total_squared_z, &total_squared_z_compensation, delta_z);
    }

    total_ax += total_ax_compensation, total_ay += total_ay_compensation, total_az += total_az_compensation;
    total_bx += total_bx_compensation, total_by += total_by_compensation, total_bz += total_bz_compensation;
    total_squared_x += total_squared_x_compensation, total_squared_y += total_squared_y_compensation,
        total_squared_z += total_squared_z_compensation;

    nk_f64_t centroid_a_x = total_ax * inv_n, centroid_a_y = total_ay * inv_n, centroid_a_z = total_az * inv_n;
    nk_f64_t centroid_b_x = total_bx * inv_n, centroid_b_y = total_by * inv_n, centroid_b_z = total_bz * inv_n;

    if (a_centroid) a_centroid[0] = centroid_a_x, a_centroid[1] = centroid_a_y, a_centroid[2] = centroid_a_z;
    if (b_centroid) b_centroid[0] = centroid_b_x, b_centroid[1] = centroid_b_y, b_centroid[2] = centroid_b_z;

    // Compute RMSD using the formula:
    // RMSD = √(E[(a-b)²] - (ā - b̄)²).
    nk_f64_t mean_diff_x = centroid_a_x - centroid_b_x, mean_diff_y = centroid_a_y - centroid_b_y,
             mean_diff_z = centroid_a_z - centroid_b_z;
    nk_f64_t sum_squared = total_squared_x + total_squared_y + total_squared_z;
    nk_f64_t mean_diff_sq = mean_diff_x * mean_diff_x + mean_diff_y * mean_diff_y + mean_diff_z * mean_diff_z;

    *result = nk_f64_sqrt_haswell(sum_squared * inv_n - mean_diff_sq);
}

NK_PUBLIC void nk_kabsch_f64_skylake(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                                     nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result) {
    // Optimized fused single-pass implementation for f64.
    // Computes centroids and covariance matrix in one pass using the identity:
    //   Hᵢⱼ = Σ((aᵢ - ā) × (bⱼ - b̄))
    //       = Σ(aᵢ × bⱼ) - Σaᵢ × Σbⱼ / n
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

    // Reduce centroids and covariance.
    nk_f64_t inv_n = 1.0 / (nk_f64_t)n;
    nk_f64_t sum_a_x = nk_reduce_stable_f64x8_skylake_(sum_a_x_f64x8), sum_a_x_compensation = 0.0;
    nk_f64_t sum_a_y = nk_reduce_stable_f64x8_skylake_(sum_a_y_f64x8), sum_a_y_compensation = 0.0;
    nk_f64_t sum_a_z = nk_reduce_stable_f64x8_skylake_(sum_a_z_f64x8), sum_a_z_compensation = 0.0;
    nk_f64_t sum_b_x = nk_reduce_stable_f64x8_skylake_(sum_b_x_f64x8), sum_b_x_compensation = 0.0;
    nk_f64_t sum_b_y = nk_reduce_stable_f64x8_skylake_(sum_b_y_f64x8), sum_b_y_compensation = 0.0;
    nk_f64_t sum_b_z = nk_reduce_stable_f64x8_skylake_(sum_b_z_f64x8), sum_b_z_compensation = 0.0;
    nk_f64_t covariance_x_x = nk_reduce_stable_f64x8_skylake_(cov_xx_f64x8), covariance_x_x_compensation = 0.0;
    nk_f64_t covariance_x_y = nk_reduce_stable_f64x8_skylake_(cov_xy_f64x8), covariance_x_y_compensation = 0.0;
    nk_f64_t covariance_x_z = nk_reduce_stable_f64x8_skylake_(cov_xz_f64x8), covariance_x_z_compensation = 0.0;
    nk_f64_t covariance_y_x = nk_reduce_stable_f64x8_skylake_(cov_yx_f64x8), covariance_y_x_compensation = 0.0;
    nk_f64_t covariance_y_y = nk_reduce_stable_f64x8_skylake_(cov_yy_f64x8), covariance_y_y_compensation = 0.0;
    nk_f64_t covariance_y_z = nk_reduce_stable_f64x8_skylake_(cov_yz_f64x8), covariance_y_z_compensation = 0.0;
    nk_f64_t covariance_z_x = nk_reduce_stable_f64x8_skylake_(cov_zx_f64x8), covariance_z_x_compensation = 0.0;
    nk_f64_t covariance_z_y = nk_reduce_stable_f64x8_skylake_(cov_zy_f64x8), covariance_z_y_compensation = 0.0;
    nk_f64_t covariance_z_z = nk_reduce_stable_f64x8_skylake_(cov_zz_f64x8), covariance_z_z_compensation = 0.0;

    for (; i < n; ++i) {
        nk_f64_t ax = a[i * 3 + 0], ay = a[i * 3 + 1], az = a[i * 3 + 2];
        nk_f64_t bx = b[i * 3 + 0], by = b[i * 3 + 1], bz = b[i * 3 + 2];
        nk_accumulate_sum_f64_(&sum_a_x, &sum_a_x_compensation, ax);
        nk_accumulate_sum_f64_(&sum_a_y, &sum_a_y_compensation, ay);
        nk_accumulate_sum_f64_(&sum_a_z, &sum_a_z_compensation, az);
        nk_accumulate_sum_f64_(&sum_b_x, &sum_b_x_compensation, bx);
        nk_accumulate_sum_f64_(&sum_b_y, &sum_b_y_compensation, by);
        nk_accumulate_sum_f64_(&sum_b_z, &sum_b_z_compensation, bz);
        nk_accumulate_product_f64_(&covariance_x_x, &covariance_x_x_compensation, ax, bx);
        nk_accumulate_product_f64_(&covariance_x_y, &covariance_x_y_compensation, ax, by);
        nk_accumulate_product_f64_(&covariance_x_z, &covariance_x_z_compensation, ax, bz);
        nk_accumulate_product_f64_(&covariance_y_x, &covariance_y_x_compensation, ay, bx);
        nk_accumulate_product_f64_(&covariance_y_y, &covariance_y_y_compensation, ay, by);
        nk_accumulate_product_f64_(&covariance_y_z, &covariance_y_z_compensation, ay, bz);
        nk_accumulate_product_f64_(&covariance_z_x, &covariance_z_x_compensation, az, bx);
        nk_accumulate_product_f64_(&covariance_z_y, &covariance_z_y_compensation, az, by);
        nk_accumulate_product_f64_(&covariance_z_z, &covariance_z_z_compensation, az, bz);
    }

    sum_a_x += sum_a_x_compensation, sum_a_y += sum_a_y_compensation, sum_a_z += sum_a_z_compensation;
    sum_b_x += sum_b_x_compensation, sum_b_y += sum_b_y_compensation, sum_b_z += sum_b_z_compensation;
    covariance_x_x += covariance_x_x_compensation, covariance_x_y += covariance_x_y_compensation,
        covariance_x_z += covariance_x_z_compensation;
    covariance_y_x += covariance_y_x_compensation, covariance_y_y += covariance_y_y_compensation,
        covariance_y_z += covariance_y_z_compensation;
    covariance_z_x += covariance_z_x_compensation, covariance_z_y += covariance_z_y_compensation,
        covariance_z_z += covariance_z_z_compensation;

    nk_f64_t centroid_a_x = sum_a_x * inv_n, centroid_a_y = sum_a_y * inv_n, centroid_a_z = sum_a_z * inv_n;
    nk_f64_t centroid_b_x = sum_b_x * inv_n, centroid_b_y = sum_b_y * inv_n, centroid_b_z = sum_b_z * inv_n;

    if (a_centroid) a_centroid[0] = centroid_a_x, a_centroid[1] = centroid_a_y, a_centroid[2] = centroid_a_z;
    if (b_centroid) b_centroid[0] = centroid_b_x, b_centroid[1] = centroid_b_y, b_centroid[2] = centroid_b_z;

    // Compute centered covariance matrix: Hᵢⱼ = Σ(aᵢ×bⱼ) - Σaᵢ × Σbⱼ / n.
    nk_f64_t cross_covariance[9];
    cross_covariance[0] = covariance_x_x - sum_a_x * sum_b_x * inv_n;
    cross_covariance[1] = covariance_x_y - sum_a_x * sum_b_y * inv_n;
    cross_covariance[2] = covariance_x_z - sum_a_x * sum_b_z * inv_n;
    cross_covariance[3] = covariance_y_x - sum_a_y * sum_b_x * inv_n;
    cross_covariance[4] = covariance_y_y - sum_a_y * sum_b_y * inv_n;
    cross_covariance[5] = covariance_y_z - sum_a_y * sum_b_z * inv_n;
    cross_covariance[6] = covariance_z_x - sum_a_z * sum_b_x * inv_n;
    cross_covariance[7] = covariance_z_y - sum_a_z * sum_b_y * inv_n;
    cross_covariance[8] = covariance_z_z - sum_a_z * sum_b_z * inv_n;

    // SVD using f64 for full precision
    nk_f64_t svd_u[9], svd_s[9], svd_v[9];
    nk_svd3x3_f64_(cross_covariance, svd_u, svd_s, svd_v);

    nk_f64_t r[9];
    nk_rotation_from_svd_f64_skylake_(svd_u, svd_v, r);

    // Handle reflection
    if (nk_det3x3_f64_(r) < 0) {
        svd_v[2] = -svd_v[2], svd_v[5] = -svd_v[5], svd_v[8] = -svd_v[8];
        nk_rotation_from_svd_f64_skylake_(svd_u, svd_v, r);
    }

    // Output rotation matrix and scale=1.0.
    if (rotation) {
        for (int j = 0; j < 9; ++j) rotation[j] = (nk_f64_t)r[j];
    }
    if (scale) *scale = 1.0;

    // Compute RMSD after optimal rotation
    nk_f64_t sum_squared = nk_transformed_ssd_f64_skylake_(a, b, n, r, 1.0, centroid_a_x, centroid_a_y, centroid_a_z,
                                                           centroid_b_x, centroid_b_y, centroid_b_z);
    *result = nk_f64_sqrt_haswell(sum_squared * inv_n);
}

NK_INTERNAL void nk_centroid_and_cross_covariance_and_variance_f32_skylake_( //
    nk_f32_t const *a, nk_f32_t const *b, nk_size_t n,                       //
    nk_f64_t *centroid_a_x, nk_f64_t *centroid_a_y, nk_f64_t *centroid_a_z, nk_f64_t *centroid_b_x,
    nk_f64_t *centroid_b_y, nk_f64_t *centroid_b_z, nk_f64_t cross_covariance_f64[9], nk_f64_t *variance_a) {
    nk_centroid_and_cross_covariance_f32_skylake_(a, b, n, centroid_a_x, centroid_a_y, centroid_a_z, centroid_b_x,
                                                  centroid_b_y, centroid_b_z, cross_covariance_f64);
    __m512d variance_a_f64x8 = _mm512_setzero_pd();
    __m512 a_x_f32x16, a_y_f32x16, a_z_f32x16;
    nk_size_t index = 0;

    for (; index + 16 <= n; index += 16) {
        nk_deinterleave_f32x16_skylake_(a + index * 3, &a_x_f32x16, &a_y_f32x16, &a_z_f32x16);
        __m512d a_x_lower_f64x8 = _mm512_cvtps_pd(_mm512_castps512_ps256(a_x_f32x16));
        __m512d a_x_upper_f64x8 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(a_x_f32x16, 1));
        __m512d a_y_lower_f64x8 = _mm512_cvtps_pd(_mm512_castps512_ps256(a_y_f32x16));
        __m512d a_y_upper_f64x8 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(a_y_f32x16, 1));
        __m512d a_z_lower_f64x8 = _mm512_cvtps_pd(_mm512_castps512_ps256(a_z_f32x16));
        __m512d a_z_upper_f64x8 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(a_z_f32x16, 1));
        __m512d batch_norm_squared_f64x8 = _mm512_add_pd(_mm512_mul_pd(a_x_lower_f64x8, a_x_lower_f64x8),
                                                         _mm512_mul_pd(a_x_upper_f64x8, a_x_upper_f64x8));
        batch_norm_squared_f64x8 = _mm512_fmadd_pd(a_y_lower_f64x8, a_y_lower_f64x8, batch_norm_squared_f64x8);
        batch_norm_squared_f64x8 = _mm512_fmadd_pd(a_y_upper_f64x8, a_y_upper_f64x8, batch_norm_squared_f64x8);
        batch_norm_squared_f64x8 = _mm512_fmadd_pd(a_z_lower_f64x8, a_z_lower_f64x8, batch_norm_squared_f64x8);
        batch_norm_squared_f64x8 = _mm512_fmadd_pd(a_z_upper_f64x8, a_z_upper_f64x8, batch_norm_squared_f64x8);
        variance_a_f64x8 = _mm512_add_pd(variance_a_f64x8, batch_norm_squared_f64x8);
    }

    nk_f64_t variance_sum = _mm512_reduce_add_pd(variance_a_f64x8);
    for (; index < n; ++index) {
        nk_f64_t a_x = a[index * 3 + 0], a_y = a[index * 3 + 1], a_z = a[index * 3 + 2];
        variance_sum += a_x * a_x + a_y * a_y + a_z * a_z;
    }

    nk_f64_t inv_n = 1.0 / (nk_f64_t)n;
    *variance_a = variance_sum * inv_n - ((*centroid_a_x) * (*centroid_a_x) + (*centroid_a_y) * (*centroid_a_y) +
                                          (*centroid_a_z) * (*centroid_a_z));
}

NK_PUBLIC void nk_umeyama_f32_skylake(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                      nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f64_t *result) {
    nk_f64_t centroid_a_x, centroid_a_y, centroid_a_z, centroid_b_x, centroid_b_y, centroid_b_z, variance_a;
    nk_f64_t cross_covariance_f64[9];
    nk_centroid_and_cross_covariance_and_variance_f32_skylake_(a, b, n, &centroid_a_x, &centroid_a_y, &centroid_a_z,
                                                               &centroid_b_x, &centroid_b_y, &centroid_b_z,
                                                               cross_covariance_f64, &variance_a);
    if (a_centroid)
        a_centroid[0] = (nk_f32_t)centroid_a_x, a_centroid[1] = (nk_f32_t)centroid_a_y,
        a_centroid[2] = (nk_f32_t)centroid_a_z;
    if (b_centroid)
        b_centroid[0] = (nk_f32_t)centroid_b_x, b_centroid[1] = (nk_f32_t)centroid_b_y,
        b_centroid[2] = (nk_f32_t)centroid_b_z;

    nk_f64_t svd_u[9], svd_s[9], svd_v[9], r[9];
    nk_svd3x3_f64_(cross_covariance_f64, svd_u, svd_s, svd_v);
    r[0] = svd_v[0] * svd_u[0] + svd_v[1] * svd_u[1] + svd_v[2] * svd_u[2];
    r[1] = svd_v[0] * svd_u[3] + svd_v[1] * svd_u[4] + svd_v[2] * svd_u[5];
    r[2] = svd_v[0] * svd_u[6] + svd_v[1] * svd_u[7] + svd_v[2] * svd_u[8];
    r[3] = svd_v[3] * svd_u[0] + svd_v[4] * svd_u[1] + svd_v[5] * svd_u[2];
    r[4] = svd_v[3] * svd_u[3] + svd_v[4] * svd_u[4] + svd_v[5] * svd_u[5];
    r[5] = svd_v[3] * svd_u[6] + svd_v[4] * svd_u[7] + svd_v[5] * svd_u[8];
    r[6] = svd_v[6] * svd_u[0] + svd_v[7] * svd_u[1] + svd_v[8] * svd_u[2];
    r[7] = svd_v[6] * svd_u[3] + svd_v[7] * svd_u[4] + svd_v[8] * svd_u[5];
    r[8] = svd_v[6] * svd_u[6] + svd_v[7] * svd_u[7] + svd_v[8] * svd_u[8];

    nk_f64_t det = nk_det3x3_f64_(r);
    nk_f64_t trace_signed_singular_values = svd_s[0] + svd_s[4] + (det < 0 ? -svd_s[8] : svd_s[8]);
    nk_f64_t applied_scale = trace_signed_singular_values / ((nk_f64_t)n * variance_a);
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

    if (rotation)
        for (int index = 0; index != 9; ++index) rotation[index] = (nk_f32_t)r[index];
    if (scale) *scale = (nk_f32_t)applied_scale;
    *result = nk_f64_sqrt_haswell(nk_transformed_ssd_f32_skylake_(a, b, n, r, applied_scale, centroid_a_x, centroid_a_y,
                                                                  centroid_a_z, centroid_b_x, centroid_b_y,
                                                                  centroid_b_z) /
                                  (nk_f64_t)n);
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

    // Reduce centroids, covariance, and variance.
    nk_f64_t inv_n = 1.0 / (nk_f64_t)n;
    nk_f64_t sum_a_x = nk_reduce_stable_f64x8_skylake_(sum_a_x_f64x8), sum_a_x_compensation = 0.0;
    nk_f64_t sum_a_y = nk_reduce_stable_f64x8_skylake_(sum_a_y_f64x8), sum_a_y_compensation = 0.0;
    nk_f64_t sum_a_z = nk_reduce_stable_f64x8_skylake_(sum_a_z_f64x8), sum_a_z_compensation = 0.0;
    nk_f64_t sum_b_x = nk_reduce_stable_f64x8_skylake_(sum_b_x_f64x8), sum_b_x_compensation = 0.0;
    nk_f64_t sum_b_y = nk_reduce_stable_f64x8_skylake_(sum_b_y_f64x8), sum_b_y_compensation = 0.0;
    nk_f64_t sum_b_z = nk_reduce_stable_f64x8_skylake_(sum_b_z_f64x8), sum_b_z_compensation = 0.0;
    nk_f64_t covariance_x_x = nk_reduce_stable_f64x8_skylake_(cov_xx_f64x8), covariance_x_x_compensation = 0.0;
    nk_f64_t covariance_x_y = nk_reduce_stable_f64x8_skylake_(cov_xy_f64x8), covariance_x_y_compensation = 0.0;
    nk_f64_t covariance_x_z = nk_reduce_stable_f64x8_skylake_(cov_xz_f64x8), covariance_x_z_compensation = 0.0;
    nk_f64_t covariance_y_x = nk_reduce_stable_f64x8_skylake_(cov_yx_f64x8), covariance_y_x_compensation = 0.0;
    nk_f64_t covariance_y_y = nk_reduce_stable_f64x8_skylake_(cov_yy_f64x8), covariance_y_y_compensation = 0.0;
    nk_f64_t covariance_y_z = nk_reduce_stable_f64x8_skylake_(cov_yz_f64x8), covariance_y_z_compensation = 0.0;
    nk_f64_t covariance_z_x = nk_reduce_stable_f64x8_skylake_(cov_zx_f64x8), covariance_z_x_compensation = 0.0;
    nk_f64_t covariance_z_y = nk_reduce_stable_f64x8_skylake_(cov_zy_f64x8), covariance_z_y_compensation = 0.0;
    nk_f64_t covariance_z_z = nk_reduce_stable_f64x8_skylake_(cov_zz_f64x8), covariance_z_z_compensation = 0.0;
    nk_f64_t variance_a_sum = nk_reduce_stable_f64x8_skylake_(variance_a_f64x8), variance_a_compensation = 0.0;

    for (; i < n; ++i) {
        nk_f64_t ax = a[i * 3 + 0], ay = a[i * 3 + 1], az = a[i * 3 + 2];
        nk_f64_t bx = b[i * 3 + 0], by = b[i * 3 + 1], bz = b[i * 3 + 2];
        nk_accumulate_sum_f64_(&sum_a_x, &sum_a_x_compensation, ax);
        nk_accumulate_sum_f64_(&sum_a_y, &sum_a_y_compensation, ay);
        nk_accumulate_sum_f64_(&sum_a_z, &sum_a_z_compensation, az);
        nk_accumulate_sum_f64_(&sum_b_x, &sum_b_x_compensation, bx);
        nk_accumulate_sum_f64_(&sum_b_y, &sum_b_y_compensation, by);
        nk_accumulate_sum_f64_(&sum_b_z, &sum_b_z_compensation, bz);
        nk_accumulate_product_f64_(&covariance_x_x, &covariance_x_x_compensation, ax, bx);
        nk_accumulate_product_f64_(&covariance_x_y, &covariance_x_y_compensation, ax, by);
        nk_accumulate_product_f64_(&covariance_x_z, &covariance_x_z_compensation, ax, bz);
        nk_accumulate_product_f64_(&covariance_y_x, &covariance_y_x_compensation, ay, bx);
        nk_accumulate_product_f64_(&covariance_y_y, &covariance_y_y_compensation, ay, by);
        nk_accumulate_product_f64_(&covariance_y_z, &covariance_y_z_compensation, ay, bz);
        nk_accumulate_product_f64_(&covariance_z_x, &covariance_z_x_compensation, az, bx);
        nk_accumulate_product_f64_(&covariance_z_y, &covariance_z_y_compensation, az, by);
        nk_accumulate_product_f64_(&covariance_z_z, &covariance_z_z_compensation, az, bz);
        nk_accumulate_square_f64_(&variance_a_sum, &variance_a_compensation, ax);
        nk_accumulate_square_f64_(&variance_a_sum, &variance_a_compensation, ay);
        nk_accumulate_square_f64_(&variance_a_sum, &variance_a_compensation, az);
    }

    sum_a_x += sum_a_x_compensation, sum_a_y += sum_a_y_compensation, sum_a_z += sum_a_z_compensation;
    sum_b_x += sum_b_x_compensation, sum_b_y += sum_b_y_compensation, sum_b_z += sum_b_z_compensation;
    covariance_x_x += covariance_x_x_compensation, covariance_x_y += covariance_x_y_compensation,
        covariance_x_z += covariance_x_z_compensation;
    covariance_y_x += covariance_y_x_compensation, covariance_y_y += covariance_y_y_compensation,
        covariance_y_z += covariance_y_z_compensation;
    covariance_z_x += covariance_z_x_compensation, covariance_z_y += covariance_z_y_compensation,
        covariance_z_z += covariance_z_z_compensation;
    variance_a_sum += variance_a_compensation;

    nk_f64_t centroid_a_x = sum_a_x * inv_n, centroid_a_y = sum_a_y * inv_n, centroid_a_z = sum_a_z * inv_n;
    nk_f64_t centroid_b_x = sum_b_x * inv_n, centroid_b_y = sum_b_y * inv_n, centroid_b_z = sum_b_z * inv_n;

    if (a_centroid) a_centroid[0] = centroid_a_x, a_centroid[1] = centroid_a_y, a_centroid[2] = centroid_a_z;
    if (b_centroid) b_centroid[0] = centroid_b_x, b_centroid[1] = centroid_b_y, b_centroid[2] = centroid_b_z;

    // Compute centered covariance and variance.
    nk_f64_t variance_a = variance_a_sum * inv_n -
                          (centroid_a_x * centroid_a_x + centroid_a_y * centroid_a_y + centroid_a_z * centroid_a_z);

    // Compute centered covariance matrix: Hᵢⱼ = Σ(aᵢ×bⱼ) - Σaᵢ × Σbⱼ / n.
    nk_f64_t cross_covariance[9];
    cross_covariance[0] = covariance_x_x - sum_a_x * sum_b_x * inv_n;
    cross_covariance[1] = covariance_x_y - sum_a_x * sum_b_y * inv_n;
    cross_covariance[2] = covariance_x_z - sum_a_x * sum_b_z * inv_n;
    cross_covariance[3] = covariance_y_x - sum_a_y * sum_b_x * inv_n;
    cross_covariance[4] = covariance_y_y - sum_a_y * sum_b_y * inv_n;
    cross_covariance[5] = covariance_y_z - sum_a_y * sum_b_z * inv_n;
    cross_covariance[6] = covariance_z_x - sum_a_z * sum_b_x * inv_n;
    cross_covariance[7] = covariance_z_y - sum_a_z * sum_b_y * inv_n;
    cross_covariance[8] = covariance_z_z - sum_a_z * sum_b_z * inv_n;

    // SVD using f64 for full precision
    nk_f64_t svd_u[9], svd_s[9], svd_v[9];
    nk_svd3x3_f64_(cross_covariance, svd_u, svd_s, svd_v);

    nk_f64_t r[9];
    nk_rotation_from_svd_f64_skylake_(svd_u, svd_v, r);

    // Scale factor: c = trace(D × S) / (n × variance(a))
    nk_f64_t det = nk_det3x3_f64_(r);
    nk_f64_t d3 = det < 0 ? -1.0 : 1.0;
    nk_f64_t trace_ds = nk_sum_three_products_f64_(svd_s[0], 1.0, svd_s[4], 1.0, svd_s[8], d3);
    nk_f64_t c = trace_ds / (n * variance_a);
    if (scale) *scale = c;

    // Handle reflection
    if (det < 0) {
        svd_v[2] = -svd_v[2], svd_v[5] = -svd_v[5], svd_v[8] = -svd_v[8];
        nk_rotation_from_svd_f64_skylake_(svd_u, svd_v, r);
    }

    // Output rotation matrix.
    if (rotation) {
        for (int j = 0; j < 9; ++j) rotation[j] = (nk_f64_t)r[j];
    }

    // Compute RMSD with scaling
    nk_f64_t sum_squared = nk_transformed_ssd_f64_skylake_(a, b, n, r, c, centroid_a_x, centroid_a_y, centroid_a_z,
                                                           centroid_b_x, centroid_b_y, centroid_b_z);
    *result = nk_f64_sqrt_haswell(sum_squared * inv_n);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_SKYLAKE
#endif // NK_TARGET_X86_
#endif // NK_MESH_SKYLAKE_H

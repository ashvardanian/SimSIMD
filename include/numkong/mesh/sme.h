/**
 *  @brief SIMD-accelerated Point Cloud Alignment for SME.
 *  @file include/numkong/mesh/sme.h
 *  @author Ash Vardanian
 *  @date February 2026
 *
 *  @sa include/numkong/mesh.h
 *
 *  Uses ARM SME streaming SVE for mesh alignment with f32 inputs:
 *  - svld3_f32 hardware stride-3 deinterleaving (16 pts/iter at SVL=512)
 *  - svwhilelt_b32 predicated tail handling (no scalar remainder)
 *  - Single vector accumulator set (SVE width already 4x NEON, no unrolling needed)
 *
 *  @section sme_mesh_instructions Key Streaming SVE Instructions
 *
 *      Intrinsic                   Instruction                     Latency     Throughput
 *      svld3_f32                   LD3W (Z.S, P/Z, [Xn])           8-12cy      0.5/cy
 *      svmla_f32_x                FMLA (Z.S, P/M, Z.S, Z.S)       4cy         2/cy
 *      svadd_f32_x                FADD (Z.S, P/M, Z.S, Z.S)       3cy         2/cy
 *      svaddv_f32                 FADDV (S, P, Z.S)               6cy         1/cy
 *      svwhilelt_b32              WHILELT (P.S, Xn, Xm)           2cy         1/cy
 *      svcntw                     CNTW (Xd)                       1cy         2/cy
 */
#ifndef NK_MESH_SME_H
#define NK_MESH_SME_H

#if NK_TARGET_ARM_
#if NK_TARGET_SME

#include "numkong/types.h"
#include "numkong/mesh/serial.h" // `nk_svd3x3_f32_`, `nk_det3x3_f32_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("sme"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("+sme")
#endif

NK_INTERNAL nk_f32_t nk_f32_sqrt_sme_(nk_f32_t number_f32) {
    svbool_t predicate_first_b32 = svptrue_pat_b32(SV_VL1);
    return svaddv_f32(predicate_first_b32, svsqrt_f32_x(predicate_first_b32, svdup_f32(number_f32)));
}

NK_INTERNAL nk_f32_t nk_transformed_ssd_f32_sme_(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n,
                                                 nk_f32_t const *rotation, nk_f32_t scale, nk_f32_t centroid_a_x,
                                                 nk_f32_t centroid_a_y, nk_f32_t centroid_a_z, nk_f32_t centroid_b_x,
                                                 nk_f32_t centroid_b_y, nk_f32_t centroid_b_z) {
    // Broadcast scaled rotation matrix elements
    svfloat32_t scaled_r00_f32 = svdup_f32(scale * rotation[0]);
    svfloat32_t scaled_r01_f32 = svdup_f32(scale * rotation[1]);
    svfloat32_t scaled_r02_f32 = svdup_f32(scale * rotation[2]);
    svfloat32_t scaled_r10_f32 = svdup_f32(scale * rotation[3]);
    svfloat32_t scaled_r11_f32 = svdup_f32(scale * rotation[4]);
    svfloat32_t scaled_r12_f32 = svdup_f32(scale * rotation[5]);
    svfloat32_t scaled_r20_f32 = svdup_f32(scale * rotation[6]);
    svfloat32_t scaled_r21_f32 = svdup_f32(scale * rotation[7]);
    svfloat32_t scaled_r22_f32 = svdup_f32(scale * rotation[8]);

    // Broadcast centroids
    svfloat32_t centroid_a_x_f32 = svdup_f32(centroid_a_x);
    svfloat32_t centroid_a_y_f32 = svdup_f32(centroid_a_y);
    svfloat32_t centroid_a_z_f32 = svdup_f32(centroid_a_z);
    svfloat32_t centroid_b_x_f32 = svdup_f32(centroid_b_x);
    svfloat32_t centroid_b_y_f32 = svdup_f32(centroid_b_y);
    svfloat32_t centroid_b_z_f32 = svdup_f32(centroid_b_z);

    svfloat32_t sum_squared_f32 = svdup_f32(0);
    svbool_t predicate_body_b32 = svptrue_b32();
    nk_size_t index = 0;
    svbool_t predicate_tail_b32 = svwhilelt_b32((unsigned int)index, (unsigned int)n);
    while (svptest_first(predicate_body_b32, predicate_tail_b32)) {
        svfloat32x3_t a_xyz_f32x3 = svld3_f32(predicate_tail_b32, a + index * 3);
        svfloat32_t a_x_f32 = svget3_f32(a_xyz_f32x3, 0);
        svfloat32_t a_y_f32 = svget3_f32(a_xyz_f32x3, 1);
        svfloat32_t a_z_f32 = svget3_f32(a_xyz_f32x3, 2);

        svfloat32x3_t b_xyz_f32x3 = svld3_f32(predicate_tail_b32, b + index * 3);
        svfloat32_t b_x_f32 = svget3_f32(b_xyz_f32x3, 0);
        svfloat32_t b_y_f32 = svget3_f32(b_xyz_f32x3, 1);
        svfloat32_t b_z_f32 = svget3_f32(b_xyz_f32x3, 2);

        // Center
        svfloat32_t centered_a_x_f32 = svsub_f32_x(predicate_tail_b32, a_x_f32, centroid_a_x_f32);
        svfloat32_t centered_a_y_f32 = svsub_f32_x(predicate_tail_b32, a_y_f32, centroid_a_y_f32);
        svfloat32_t centered_a_z_f32 = svsub_f32_x(predicate_tail_b32, a_z_f32, centroid_a_z_f32);
        svfloat32_t centered_b_x_f32 = svsub_f32_x(predicate_tail_b32, b_x_f32, centroid_b_x_f32);
        svfloat32_t centered_b_y_f32 = svsub_f32_x(predicate_tail_b32, b_y_f32, centroid_b_y_f32);
        svfloat32_t centered_b_z_f32 = svsub_f32_x(predicate_tail_b32, b_z_f32, centroid_b_z_f32);

        // Rotate: rotated = scale * R * centered_a
        svfloat32_t rotated_a_x_f32 = svmla_f32_x(
            predicate_tail_b32,
            svmla_f32_x(predicate_tail_b32, svmul_f32_x(predicate_tail_b32, scaled_r00_f32, centered_a_x_f32),
                        scaled_r01_f32, centered_a_y_f32),
            scaled_r02_f32, centered_a_z_f32);
        svfloat32_t rotated_a_y_f32 = svmla_f32_x(
            predicate_tail_b32,
            svmla_f32_x(predicate_tail_b32, svmul_f32_x(predicate_tail_b32, scaled_r10_f32, centered_a_x_f32),
                        scaled_r11_f32, centered_a_y_f32),
            scaled_r12_f32, centered_a_z_f32);
        svfloat32_t rotated_a_z_f32 = svmla_f32_x(
            predicate_tail_b32,
            svmla_f32_x(predicate_tail_b32, svmul_f32_x(predicate_tail_b32, scaled_r20_f32, centered_a_x_f32),
                        scaled_r21_f32, centered_a_y_f32),
            scaled_r22_f32, centered_a_z_f32);

        // Delta
        svfloat32_t delta_x_f32 = svsub_f32_x(predicate_tail_b32, rotated_a_x_f32, centered_b_x_f32);
        svfloat32_t delta_y_f32 = svsub_f32_x(predicate_tail_b32, rotated_a_y_f32, centered_b_y_f32);
        svfloat32_t delta_z_f32 = svsub_f32_x(predicate_tail_b32, rotated_a_z_f32, centered_b_z_f32);

        // Accumulate
        sum_squared_f32 = svmla_f32_x(predicate_tail_b32, sum_squared_f32, delta_x_f32, delta_x_f32);
        sum_squared_f32 = svmla_f32_x(predicate_tail_b32, sum_squared_f32, delta_y_f32, delta_y_f32);
        sum_squared_f32 = svmla_f32_x(predicate_tail_b32, sum_squared_f32, delta_z_f32, delta_z_f32);

        index += svcntw();
        predicate_tail_b32 = svwhilelt_b32((unsigned int)index, (unsigned int)n);
    }

    return svaddv_f32(predicate_body_b32, sum_squared_f32);
}

__arm_locally_streaming static void nk_rmsd_f32_sme_kernel_(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n,
                                                            nk_f32_t *a_centroid, nk_f32_t *b_centroid,
                                                            nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result) {
    // RMSD uses identity rotation and scale=1.0
    if (rotation) {
        rotation[0] = 1, rotation[1] = 0, rotation[2] = 0;
        rotation[3] = 0, rotation[4] = 1, rotation[5] = 0;
        rotation[6] = 0, rotation[7] = 0, rotation[8] = 1;
    }
    if (scale) *scale = 1.0f;

    // SVE accumulators
    svfloat32_t sum_a_x_f32 = svdup_f32(0), sum_a_y_f32 = svdup_f32(0), sum_a_z_f32 = svdup_f32(0);
    svfloat32_t sum_b_x_f32 = svdup_f32(0), sum_b_y_f32 = svdup_f32(0), sum_b_z_f32 = svdup_f32(0);
    svfloat32_t sum_squared_x_f32 = svdup_f32(0), sum_squared_y_f32 = svdup_f32(0), sum_squared_z_f32 = svdup_f32(0);

    svbool_t predicate_body_b32 = svptrue_b32();
    nk_size_t index = 0;
    svbool_t predicate_tail_b32 = svwhilelt_b32((unsigned int)index, (unsigned int)n);
    while (svptest_first(predicate_body_b32, predicate_tail_b32)) {
        svfloat32x3_t a_xyz_f32x3 = svld3_f32(predicate_tail_b32, a + index * 3);
        svfloat32_t a_x_f32 = svget3_f32(a_xyz_f32x3, 0);
        svfloat32_t a_y_f32 = svget3_f32(a_xyz_f32x3, 1);
        svfloat32_t a_z_f32 = svget3_f32(a_xyz_f32x3, 2);

        svfloat32x3_t b_xyz_f32x3 = svld3_f32(predicate_tail_b32, b + index * 3);
        svfloat32_t b_x_f32 = svget3_f32(b_xyz_f32x3, 0);
        svfloat32_t b_y_f32 = svget3_f32(b_xyz_f32x3, 1);
        svfloat32_t b_z_f32 = svget3_f32(b_xyz_f32x3, 2);

        // Centroid sums
        sum_a_x_f32 = svadd_f32_x(predicate_tail_b32, sum_a_x_f32, a_x_f32);
        sum_a_y_f32 = svadd_f32_x(predicate_tail_b32, sum_a_y_f32, a_y_f32);
        sum_a_z_f32 = svadd_f32_x(predicate_tail_b32, sum_a_z_f32, a_z_f32);
        sum_b_x_f32 = svadd_f32_x(predicate_tail_b32, sum_b_x_f32, b_x_f32);
        sum_b_y_f32 = svadd_f32_x(predicate_tail_b32, sum_b_y_f32, b_y_f32);
        sum_b_z_f32 = svadd_f32_x(predicate_tail_b32, sum_b_z_f32, b_z_f32);

        // Squared differences
        svfloat32_t delta_x_f32 = svsub_f32_x(predicate_tail_b32, a_x_f32, b_x_f32);
        svfloat32_t delta_y_f32 = svsub_f32_x(predicate_tail_b32, a_y_f32, b_y_f32);
        svfloat32_t delta_z_f32 = svsub_f32_x(predicate_tail_b32, a_z_f32, b_z_f32);
        sum_squared_x_f32 = svmla_f32_x(predicate_tail_b32, sum_squared_x_f32, delta_x_f32, delta_x_f32);
        sum_squared_y_f32 = svmla_f32_x(predicate_tail_b32, sum_squared_y_f32, delta_y_f32, delta_y_f32);
        sum_squared_z_f32 = svmla_f32_x(predicate_tail_b32, sum_squared_z_f32, delta_z_f32, delta_z_f32);

        index += svcntw();
        predicate_tail_b32 = svwhilelt_b32((unsigned int)index, (unsigned int)n);
    }

    // Reduce to scalars
    nk_f32_t sum_a_x = svaddv_f32(predicate_body_b32, sum_a_x_f32);
    nk_f32_t sum_a_y = svaddv_f32(predicate_body_b32, sum_a_y_f32);
    nk_f32_t sum_a_z = svaddv_f32(predicate_body_b32, sum_a_z_f32);
    nk_f32_t sum_b_x = svaddv_f32(predicate_body_b32, sum_b_x_f32);
    nk_f32_t sum_b_y = svaddv_f32(predicate_body_b32, sum_b_y_f32);
    nk_f32_t sum_b_z = svaddv_f32(predicate_body_b32, sum_b_z_f32);
    nk_f32_t total_squared_x = svaddv_f32(predicate_body_b32, sum_squared_x_f32);
    nk_f32_t total_squared_y = svaddv_f32(predicate_body_b32, sum_squared_y_f32);
    nk_f32_t total_squared_z = svaddv_f32(predicate_body_b32, sum_squared_z_f32);

    // Compute centroids
    nk_f32_t inverse_n = 1.0f / (nk_f32_t)n;
    nk_f32_t centroid_a_x = sum_a_x * inverse_n;
    nk_f32_t centroid_a_y = sum_a_y * inverse_n;
    nk_f32_t centroid_a_z = sum_a_z * inverse_n;
    nk_f32_t centroid_b_x = sum_b_x * inverse_n;
    nk_f32_t centroid_b_y = sum_b_y * inverse_n;
    nk_f32_t centroid_b_z = sum_b_z * inverse_n;

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

    *result = nk_f32_sqrt_sme_(sum_squared * inverse_n - mean_diff_sq);
}

__arm_locally_streaming static void nk_kabsch_f32_sme_kernel_(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n,
                                                              nk_f32_t *a_centroid, nk_f32_t *b_centroid,
                                                              nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result) {
    // Phase 1: SVE cross-covariance loop
    svfloat32_t sum_a_x_f32 = svdup_f32(0), sum_a_y_f32 = svdup_f32(0), sum_a_z_f32 = svdup_f32(0);
    svfloat32_t sum_b_x_f32 = svdup_f32(0), sum_b_y_f32 = svdup_f32(0), sum_b_z_f32 = svdup_f32(0);
    svfloat32_t covariance_xx_f32 = svdup_f32(0), covariance_xy_f32 = svdup_f32(0), covariance_xz_f32 = svdup_f32(0);
    svfloat32_t covariance_yx_f32 = svdup_f32(0), covariance_yy_f32 = svdup_f32(0), covariance_yz_f32 = svdup_f32(0);
    svfloat32_t covariance_zx_f32 = svdup_f32(0), covariance_zy_f32 = svdup_f32(0), covariance_zz_f32 = svdup_f32(0);

    svbool_t predicate_body_b32 = svptrue_b32();
    nk_size_t index = 0;
    svbool_t predicate_tail_b32 = svwhilelt_b32((unsigned int)index, (unsigned int)n);
    while (svptest_first(predicate_body_b32, predicate_tail_b32)) {
        svfloat32x3_t a_xyz_f32x3 = svld3_f32(predicate_tail_b32, a + index * 3);
        svfloat32_t a_x_f32 = svget3_f32(a_xyz_f32x3, 0);
        svfloat32_t a_y_f32 = svget3_f32(a_xyz_f32x3, 1);
        svfloat32_t a_z_f32 = svget3_f32(a_xyz_f32x3, 2);

        svfloat32x3_t b_xyz_f32x3 = svld3_f32(predicate_tail_b32, b + index * 3);
        svfloat32_t b_x_f32 = svget3_f32(b_xyz_f32x3, 0);
        svfloat32_t b_y_f32 = svget3_f32(b_xyz_f32x3, 1);
        svfloat32_t b_z_f32 = svget3_f32(b_xyz_f32x3, 2);

        // Centroid sums
        sum_a_x_f32 = svadd_f32_x(predicate_tail_b32, sum_a_x_f32, a_x_f32);
        sum_a_y_f32 = svadd_f32_x(predicate_tail_b32, sum_a_y_f32, a_y_f32);
        sum_a_z_f32 = svadd_f32_x(predicate_tail_b32, sum_a_z_f32, a_z_f32);
        sum_b_x_f32 = svadd_f32_x(predicate_tail_b32, sum_b_x_f32, b_x_f32);
        sum_b_y_f32 = svadd_f32_x(predicate_tail_b32, sum_b_y_f32, b_y_f32);
        sum_b_z_f32 = svadd_f32_x(predicate_tail_b32, sum_b_z_f32, b_z_f32);

        // Cross-covariance
        covariance_xx_f32 = svmla_f32_x(predicate_tail_b32, covariance_xx_f32, a_x_f32, b_x_f32);
        covariance_xy_f32 = svmla_f32_x(predicate_tail_b32, covariance_xy_f32, a_x_f32, b_y_f32);
        covariance_xz_f32 = svmla_f32_x(predicate_tail_b32, covariance_xz_f32, a_x_f32, b_z_f32);
        covariance_yx_f32 = svmla_f32_x(predicate_tail_b32, covariance_yx_f32, a_y_f32, b_x_f32);
        covariance_yy_f32 = svmla_f32_x(predicate_tail_b32, covariance_yy_f32, a_y_f32, b_y_f32);
        covariance_yz_f32 = svmla_f32_x(predicate_tail_b32, covariance_yz_f32, a_y_f32, b_z_f32);
        covariance_zx_f32 = svmla_f32_x(predicate_tail_b32, covariance_zx_f32, a_z_f32, b_x_f32);
        covariance_zy_f32 = svmla_f32_x(predicate_tail_b32, covariance_zy_f32, a_z_f32, b_y_f32);
        covariance_zz_f32 = svmla_f32_x(predicate_tail_b32, covariance_zz_f32, a_z_f32, b_z_f32);

        index += svcntw();
        predicate_tail_b32 = svwhilelt_b32((unsigned int)index, (unsigned int)n);
    }

    // Reduce to scalars
    nk_f32_t sum_a_x = svaddv_f32(predicate_body_b32, sum_a_x_f32);
    nk_f32_t sum_a_y = svaddv_f32(predicate_body_b32, sum_a_y_f32);
    nk_f32_t sum_a_z = svaddv_f32(predicate_body_b32, sum_a_z_f32);
    nk_f32_t sum_b_x = svaddv_f32(predicate_body_b32, sum_b_x_f32);
    nk_f32_t sum_b_y = svaddv_f32(predicate_body_b32, sum_b_y_f32);
    nk_f32_t sum_b_z = svaddv_f32(predicate_body_b32, sum_b_z_f32);

    // Phase 2: scalar centroids and centering correction
    nk_f32_t inverse_n = 1.0f / (nk_f32_t)n;
    nk_f32_t centroid_a_x = sum_a_x * inverse_n;
    nk_f32_t centroid_a_y = sum_a_y * inverse_n;
    nk_f32_t centroid_a_z = sum_a_z * inverse_n;
    nk_f32_t centroid_b_x = sum_b_x * inverse_n;
    nk_f32_t centroid_b_y = sum_b_y * inverse_n;
    nk_f32_t centroid_b_z = sum_b_z * inverse_n;

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

    nk_f32_t cross_covariance[9];
    cross_covariance[0] = svaddv_f32(predicate_body_b32, covariance_xx_f32) - n * centroid_a_x * centroid_b_x;
    cross_covariance[1] = svaddv_f32(predicate_body_b32, covariance_xy_f32) - n * centroid_a_x * centroid_b_y;
    cross_covariance[2] = svaddv_f32(predicate_body_b32, covariance_xz_f32) - n * centroid_a_x * centroid_b_z;
    cross_covariance[3] = svaddv_f32(predicate_body_b32, covariance_yx_f32) - n * centroid_a_y * centroid_b_x;
    cross_covariance[4] = svaddv_f32(predicate_body_b32, covariance_yy_f32) - n * centroid_a_y * centroid_b_y;
    cross_covariance[5] = svaddv_f32(predicate_body_b32, covariance_yz_f32) - n * centroid_a_y * centroid_b_z;
    cross_covariance[6] = svaddv_f32(predicate_body_b32, covariance_zx_f32) - n * centroid_a_z * centroid_b_x;
    cross_covariance[7] = svaddv_f32(predicate_body_b32, covariance_zy_f32) - n * centroid_a_z * centroid_b_y;
    cross_covariance[8] = svaddv_f32(predicate_body_b32, covariance_zz_f32) - n * centroid_a_z * centroid_b_z;

    // Phase 3: SVD
    nk_f32_t svd_u[9], svd_s[9], svd_v[9];
    nk_svd3x3_f32_(cross_covariance, svd_u, svd_s, svd_v);

    // Phase 4: R = V * Uᵀ
    nk_f32_t rotation_matrix[9];
    rotation_matrix[0] = svd_v[0] * svd_u[0] + svd_v[1] * svd_u[1] + svd_v[2] * svd_u[2];
    rotation_matrix[1] = svd_v[0] * svd_u[3] + svd_v[1] * svd_u[4] + svd_v[2] * svd_u[5];
    rotation_matrix[2] = svd_v[0] * svd_u[6] + svd_v[1] * svd_u[7] + svd_v[2] * svd_u[8];
    rotation_matrix[3] = svd_v[3] * svd_u[0] + svd_v[4] * svd_u[1] + svd_v[5] * svd_u[2];
    rotation_matrix[4] = svd_v[3] * svd_u[3] + svd_v[4] * svd_u[4] + svd_v[5] * svd_u[5];
    rotation_matrix[5] = svd_v[3] * svd_u[6] + svd_v[4] * svd_u[7] + svd_v[5] * svd_u[8];
    rotation_matrix[6] = svd_v[6] * svd_u[0] + svd_v[7] * svd_u[1] + svd_v[8] * svd_u[2];
    rotation_matrix[7] = svd_v[6] * svd_u[3] + svd_v[7] * svd_u[4] + svd_v[8] * svd_u[5];
    rotation_matrix[8] = svd_v[6] * svd_u[6] + svd_v[7] * svd_u[7] + svd_v[8] * svd_u[8];

    // Handle reflection
    if (nk_det3x3_f32_(rotation_matrix) < 0) {
        svd_v[2] = -svd_v[2];
        svd_v[5] = -svd_v[5];
        svd_v[8] = -svd_v[8];
        rotation_matrix[0] = svd_v[0] * svd_u[0] + svd_v[1] * svd_u[1] + svd_v[2] * svd_u[2];
        rotation_matrix[1] = svd_v[0] * svd_u[3] + svd_v[1] * svd_u[4] + svd_v[2] * svd_u[5];
        rotation_matrix[2] = svd_v[0] * svd_u[6] + svd_v[1] * svd_u[7] + svd_v[2] * svd_u[8];
        rotation_matrix[3] = svd_v[3] * svd_u[0] + svd_v[4] * svd_u[1] + svd_v[5] * svd_u[2];
        rotation_matrix[4] = svd_v[3] * svd_u[3] + svd_v[4] * svd_u[4] + svd_v[5] * svd_u[5];
        rotation_matrix[5] = svd_v[3] * svd_u[6] + svd_v[4] * svd_u[7] + svd_v[5] * svd_u[8];
        rotation_matrix[6] = svd_v[6] * svd_u[0] + svd_v[7] * svd_u[1] + svd_v[8] * svd_u[2];
        rotation_matrix[7] = svd_v[6] * svd_u[3] + svd_v[7] * svd_u[4] + svd_v[8] * svd_u[5];
        rotation_matrix[8] = svd_v[6] * svd_u[6] + svd_v[7] * svd_u[7] + svd_v[8] * svd_u[8];
    }

    if (rotation) {
        for (int j = 0; j < 9; ++j) rotation[j] = rotation_matrix[j];
    }
    if (scale) *scale = 1.0f;

    // Phase 5: SVE transformed SSD
    nk_f32_t sum_squared = nk_transformed_ssd_f32_sme_(a, b, n, rotation_matrix, 1.0f, centroid_a_x, centroid_a_y,
                                                       centroid_a_z, centroid_b_x, centroid_b_y, centroid_b_z);

    // Phase 6: SVE sqrt
    *result = nk_f32_sqrt_sme_(sum_squared * inverse_n);
}

__arm_locally_streaming static void nk_umeyama_f32_sme_kernel_(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n,
                                                               nk_f32_t *a_centroid, nk_f32_t *b_centroid,
                                                               nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result) {
    // Phase 1: SVE cross-covariance loop with variance accumulator
    svfloat32_t sum_a_x_f32 = svdup_f32(0), sum_a_y_f32 = svdup_f32(0), sum_a_z_f32 = svdup_f32(0);
    svfloat32_t sum_b_x_f32 = svdup_f32(0), sum_b_y_f32 = svdup_f32(0), sum_b_z_f32 = svdup_f32(0);
    svfloat32_t covariance_xx_f32 = svdup_f32(0), covariance_xy_f32 = svdup_f32(0), covariance_xz_f32 = svdup_f32(0);
    svfloat32_t covariance_yx_f32 = svdup_f32(0), covariance_yy_f32 = svdup_f32(0), covariance_yz_f32 = svdup_f32(0);
    svfloat32_t covariance_zx_f32 = svdup_f32(0), covariance_zy_f32 = svdup_f32(0), covariance_zz_f32 = svdup_f32(0);
    svfloat32_t variance_a_f32 = svdup_f32(0);

    svbool_t predicate_body_b32 = svptrue_b32();
    nk_size_t index = 0;
    svbool_t predicate_tail_b32 = svwhilelt_b32((unsigned int)index, (unsigned int)n);
    while (svptest_first(predicate_body_b32, predicate_tail_b32)) {
        svfloat32x3_t a_xyz_f32x3 = svld3_f32(predicate_tail_b32, a + index * 3);
        svfloat32_t a_x_f32 = svget3_f32(a_xyz_f32x3, 0);
        svfloat32_t a_y_f32 = svget3_f32(a_xyz_f32x3, 1);
        svfloat32_t a_z_f32 = svget3_f32(a_xyz_f32x3, 2);

        svfloat32x3_t b_xyz_f32x3 = svld3_f32(predicate_tail_b32, b + index * 3);
        svfloat32_t b_x_f32 = svget3_f32(b_xyz_f32x3, 0);
        svfloat32_t b_y_f32 = svget3_f32(b_xyz_f32x3, 1);
        svfloat32_t b_z_f32 = svget3_f32(b_xyz_f32x3, 2);

        // Centroid sums
        sum_a_x_f32 = svadd_f32_x(predicate_tail_b32, sum_a_x_f32, a_x_f32);
        sum_a_y_f32 = svadd_f32_x(predicate_tail_b32, sum_a_y_f32, a_y_f32);
        sum_a_z_f32 = svadd_f32_x(predicate_tail_b32, sum_a_z_f32, a_z_f32);
        sum_b_x_f32 = svadd_f32_x(predicate_tail_b32, sum_b_x_f32, b_x_f32);
        sum_b_y_f32 = svadd_f32_x(predicate_tail_b32, sum_b_y_f32, b_y_f32);
        sum_b_z_f32 = svadd_f32_x(predicate_tail_b32, sum_b_z_f32, b_z_f32);

        // Cross-covariance
        covariance_xx_f32 = svmla_f32_x(predicate_tail_b32, covariance_xx_f32, a_x_f32, b_x_f32);
        covariance_xy_f32 = svmla_f32_x(predicate_tail_b32, covariance_xy_f32, a_x_f32, b_y_f32);
        covariance_xz_f32 = svmla_f32_x(predicate_tail_b32, covariance_xz_f32, a_x_f32, b_z_f32);
        covariance_yx_f32 = svmla_f32_x(predicate_tail_b32, covariance_yx_f32, a_y_f32, b_x_f32);
        covariance_yy_f32 = svmla_f32_x(predicate_tail_b32, covariance_yy_f32, a_y_f32, b_y_f32);
        covariance_yz_f32 = svmla_f32_x(predicate_tail_b32, covariance_yz_f32, a_y_f32, b_z_f32);
        covariance_zx_f32 = svmla_f32_x(predicate_tail_b32, covariance_zx_f32, a_z_f32, b_x_f32);
        covariance_zy_f32 = svmla_f32_x(predicate_tail_b32, covariance_zy_f32, a_z_f32, b_y_f32);
        covariance_zz_f32 = svmla_f32_x(predicate_tail_b32, covariance_zz_f32, a_z_f32, b_z_f32);

        // Variance of A: Σ(ax² + ay² + az²)
        variance_a_f32 = svmla_f32_x(predicate_tail_b32, variance_a_f32, a_x_f32, a_x_f32);
        variance_a_f32 = svmla_f32_x(predicate_tail_b32, variance_a_f32, a_y_f32, a_y_f32);
        variance_a_f32 = svmla_f32_x(predicate_tail_b32, variance_a_f32, a_z_f32, a_z_f32);

        index += svcntw();
        predicate_tail_b32 = svwhilelt_b32((unsigned int)index, (unsigned int)n);
    }

    // Reduce to scalars
    nk_f32_t sum_a_x = svaddv_f32(predicate_body_b32, sum_a_x_f32);
    nk_f32_t sum_a_y = svaddv_f32(predicate_body_b32, sum_a_y_f32);
    nk_f32_t sum_a_z = svaddv_f32(predicate_body_b32, sum_a_z_f32);
    nk_f32_t sum_b_x = svaddv_f32(predicate_body_b32, sum_b_x_f32);
    nk_f32_t sum_b_y = svaddv_f32(predicate_body_b32, sum_b_y_f32);
    nk_f32_t sum_b_z = svaddv_f32(predicate_body_b32, sum_b_z_f32);
    nk_f32_t sum_sq_a = svaddv_f32(predicate_body_b32, variance_a_f32);

    // Phase 2: scalar centroids and centering correction
    nk_f32_t inverse_n = 1.0f / (nk_f32_t)n;
    nk_f32_t centroid_a_x = sum_a_x * inverse_n;
    nk_f32_t centroid_a_y = sum_a_y * inverse_n;
    nk_f32_t centroid_a_z = sum_a_z * inverse_n;
    nk_f32_t centroid_b_x = sum_b_x * inverse_n;
    nk_f32_t centroid_b_y = sum_b_y * inverse_n;
    nk_f32_t centroid_b_z = sum_b_z * inverse_n;

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

    // Variance of A (centered)
    nk_f32_t centroid_sq = centroid_a_x * centroid_a_x + centroid_a_y * centroid_a_y + centroid_a_z * centroid_a_z;
    nk_f32_t var_a = sum_sq_a * inverse_n - centroid_sq;

    nk_f32_t cross_covariance[9];
    cross_covariance[0] = svaddv_f32(predicate_body_b32, covariance_xx_f32) - n * centroid_a_x * centroid_b_x;
    cross_covariance[1] = svaddv_f32(predicate_body_b32, covariance_xy_f32) - n * centroid_a_x * centroid_b_y;
    cross_covariance[2] = svaddv_f32(predicate_body_b32, covariance_xz_f32) - n * centroid_a_x * centroid_b_z;
    cross_covariance[3] = svaddv_f32(predicate_body_b32, covariance_yx_f32) - n * centroid_a_y * centroid_b_x;
    cross_covariance[4] = svaddv_f32(predicate_body_b32, covariance_yy_f32) - n * centroid_a_y * centroid_b_y;
    cross_covariance[5] = svaddv_f32(predicate_body_b32, covariance_yz_f32) - n * centroid_a_y * centroid_b_z;
    cross_covariance[6] = svaddv_f32(predicate_body_b32, covariance_zx_f32) - n * centroid_a_z * centroid_b_x;
    cross_covariance[7] = svaddv_f32(predicate_body_b32, covariance_zy_f32) - n * centroid_a_z * centroid_b_y;
    cross_covariance[8] = svaddv_f32(predicate_body_b32, covariance_zz_f32) - n * centroid_a_z * centroid_b_z;

    // Phase 3: SVD
    nk_f32_t svd_u[9], svd_s[9], svd_v[9];
    nk_svd3x3_f32_(cross_covariance, svd_u, svd_s, svd_v);

    // Phase 4: R = V * Uᵀ
    nk_f32_t rotation_matrix[9];
    rotation_matrix[0] = svd_v[0] * svd_u[0] + svd_v[1] * svd_u[1] + svd_v[2] * svd_u[2];
    rotation_matrix[1] = svd_v[0] * svd_u[3] + svd_v[1] * svd_u[4] + svd_v[2] * svd_u[5];
    rotation_matrix[2] = svd_v[0] * svd_u[6] + svd_v[1] * svd_u[7] + svd_v[2] * svd_u[8];
    rotation_matrix[3] = svd_v[3] * svd_u[0] + svd_v[4] * svd_u[1] + svd_v[5] * svd_u[2];
    rotation_matrix[4] = svd_v[3] * svd_u[3] + svd_v[4] * svd_u[4] + svd_v[5] * svd_u[5];
    rotation_matrix[5] = svd_v[3] * svd_u[6] + svd_v[4] * svd_u[7] + svd_v[5] * svd_u[8];
    rotation_matrix[6] = svd_v[6] * svd_u[0] + svd_v[7] * svd_u[1] + svd_v[8] * svd_u[2];
    rotation_matrix[7] = svd_v[6] * svd_u[3] + svd_v[7] * svd_u[4] + svd_v[8] * svd_u[5];
    rotation_matrix[8] = svd_v[6] * svd_u[6] + svd_v[7] * svd_u[7] + svd_v[8] * svd_u[8];

    // Handle reflection and compute scale
    nk_f32_t det = nk_det3x3_f32_(rotation_matrix);
    nk_f32_t trace_d_s = svd_s[0] + svd_s[4] + (det < 0 ? -svd_s[8] : svd_s[8]);
    nk_f32_t computed_scale = trace_d_s / (n * var_a);

    if (det < 0) {
        svd_v[2] = -svd_v[2];
        svd_v[5] = -svd_v[5];
        svd_v[8] = -svd_v[8];
        rotation_matrix[0] = svd_v[0] * svd_u[0] + svd_v[1] * svd_u[1] + svd_v[2] * svd_u[2];
        rotation_matrix[1] = svd_v[0] * svd_u[3] + svd_v[1] * svd_u[4] + svd_v[2] * svd_u[5];
        rotation_matrix[2] = svd_v[0] * svd_u[6] + svd_v[1] * svd_u[7] + svd_v[2] * svd_u[8];
        rotation_matrix[3] = svd_v[3] * svd_u[0] + svd_v[4] * svd_u[1] + svd_v[5] * svd_u[2];
        rotation_matrix[4] = svd_v[3] * svd_u[3] + svd_v[4] * svd_u[4] + svd_v[5] * svd_u[5];
        rotation_matrix[5] = svd_v[3] * svd_u[6] + svd_v[4] * svd_u[7] + svd_v[5] * svd_u[8];
        rotation_matrix[6] = svd_v[6] * svd_u[0] + svd_v[7] * svd_u[1] + svd_v[8] * svd_u[2];
        rotation_matrix[7] = svd_v[6] * svd_u[3] + svd_v[7] * svd_u[4] + svd_v[8] * svd_u[5];
        rotation_matrix[8] = svd_v[6] * svd_u[6] + svd_v[7] * svd_u[7] + svd_v[8] * svd_u[8];
    }

    if (rotation) {
        for (int j = 0; j < 9; ++j) rotation[j] = rotation_matrix[j];
    }
    if (scale) *scale = computed_scale;

    // Phase 5: SVE transformed SSD
    nk_f32_t sum_squared = nk_transformed_ssd_f32_sme_(a, b, n, rotation_matrix, computed_scale, centroid_a_x,
                                                       centroid_a_y, centroid_a_z, centroid_b_x, centroid_b_y,
                                                       centroid_b_z);

    // Phase 6: SVE sqrt
    *result = nk_f32_sqrt_sme_(sum_squared * inverse_n);
}

NK_PUBLIC void nk_rmsd_f32_sme(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                               nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result) {
    nk_rmsd_f32_sme_kernel_(a, b, n, a_centroid, b_centroid, rotation, scale, result);
}

NK_PUBLIC void nk_kabsch_f32_sme(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                 nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result) {
    nk_kabsch_f32_sme_kernel_(a, b, n, a_centroid, b_centroid, rotation, scale, result);
}

NK_PUBLIC void nk_umeyama_f32_sme(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                  nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result) {
    nk_umeyama_f32_sme_kernel_(a, b, n, a_centroid, b_centroid, rotation, scale, result);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_SME
#endif // NK_TARGET_ARM_
#endif // NK_MESH_SME_H

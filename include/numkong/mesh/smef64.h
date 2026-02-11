/**
 *  @brief SIMD-accelerated Point Cloud Alignment for SME F64.
 *  @file include/numkong/mesh/smef64.h
 *  @author Ash Vardanian
 *  @date February 2026
 *
 *  @sa include/numkong/mesh.h
 *
 *  Uses ARM SME streaming SVE with sme-f64f64 for mesh alignment with f64 inputs:
 *  - svld3_f64 hardware stride-3 deinterleaving (8 pts/iter at SVL=512)
 *  - svwhilelt_b64 predicated tail handling (no scalar remainder)
 *  - Full f64 SVD and rotation (fixes NEON f64 kernel precision bug)
 *
 *  @section smef64_mesh_instructions Key Streaming SVE Instructions
 *
 *      Intrinsic                   Instruction                     Latency     Throughput
 *      svld3_f64                   LD3D (Z.D, P/Z, [Xn])           8-12cy      0.5/cy
 *      svmla_f64_x                FMLA (Z.D, P/M, Z.D, Z.D)       4cy         2/cy
 *      svadd_f64_x                FADD (Z.D, P/M, Z.D, Z.D)       3cy         2/cy
 *      svaddv_f64                 FADDV (D, P, Z.D)               6cy         1/cy
 *      svwhilelt_b64              WHILELT (P.D, Xn, Xm)           2cy         1/cy
 *      svcntd                     CNTD (Xd)                       1cy         2/cy
 */
#ifndef NK_MESH_SMEF64_H
#define NK_MESH_SMEF64_H

#if NK_TARGET_ARM_
#if NK_TARGET_SMEF64

#include "numkong/types.h"
#include "numkong/mesh/serial.h" // `nk_svd3x3_f64_`, `nk_det3x3_f64_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("sme,sme-f64f64"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("+sme+sme-f64f64")
#endif

NK_INTERNAL nk_f64_t nk_f64_sqrt_smef64_(nk_f64_t number_f64) {
    svbool_t predicate_first_b64 = svptrue_pat_b64(SV_VL1);
    return svaddv_f64(predicate_first_b64, svsqrt_f64_x(predicate_first_b64, svdup_f64(number_f64)));
}

NK_INTERNAL nk_f64_t nk_transformed_ssd_f64_smef64_(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n,
                                                    nk_f64_t const *rotation, nk_f64_t scale, nk_f64_t centroid_a_x,
                                                    nk_f64_t centroid_a_y, nk_f64_t centroid_a_z, nk_f64_t centroid_b_x,
                                                    nk_f64_t centroid_b_y, nk_f64_t centroid_b_z) {
    // Broadcast scaled rotation matrix elements
    svfloat64_t scaled_r00_f64 = svdup_f64(scale * rotation[0]);
    svfloat64_t scaled_r01_f64 = svdup_f64(scale * rotation[1]);
    svfloat64_t scaled_r02_f64 = svdup_f64(scale * rotation[2]);
    svfloat64_t scaled_r10_f64 = svdup_f64(scale * rotation[3]);
    svfloat64_t scaled_r11_f64 = svdup_f64(scale * rotation[4]);
    svfloat64_t scaled_r12_f64 = svdup_f64(scale * rotation[5]);
    svfloat64_t scaled_r20_f64 = svdup_f64(scale * rotation[6]);
    svfloat64_t scaled_r21_f64 = svdup_f64(scale * rotation[7]);
    svfloat64_t scaled_r22_f64 = svdup_f64(scale * rotation[8]);

    // Broadcast centroids
    svfloat64_t centroid_a_x_f64 = svdup_f64(centroid_a_x);
    svfloat64_t centroid_a_y_f64 = svdup_f64(centroid_a_y);
    svfloat64_t centroid_a_z_f64 = svdup_f64(centroid_a_z);
    svfloat64_t centroid_b_x_f64 = svdup_f64(centroid_b_x);
    svfloat64_t centroid_b_y_f64 = svdup_f64(centroid_b_y);
    svfloat64_t centroid_b_z_f64 = svdup_f64(centroid_b_z);

    svfloat64_t sum_squared_f64 = svdup_f64(0);
    svbool_t predicate_body_b64 = svptrue_b64();
    nk_size_t index = 0;
    svbool_t predicate_tail_b64 = svwhilelt_b64(index, n);
    while (svptest_first(predicate_body_b64, predicate_tail_b64)) {
        svfloat64x3_t a_xyz_f64x3 = svld3_f64(predicate_tail_b64, a + index * 3);
        svfloat64_t a_x_f64 = svget3_f64(a_xyz_f64x3, 0);
        svfloat64_t a_y_f64 = svget3_f64(a_xyz_f64x3, 1);
        svfloat64_t a_z_f64 = svget3_f64(a_xyz_f64x3, 2);

        svfloat64x3_t b_xyz_f64x3 = svld3_f64(predicate_tail_b64, b + index * 3);
        svfloat64_t b_x_f64 = svget3_f64(b_xyz_f64x3, 0);
        svfloat64_t b_y_f64 = svget3_f64(b_xyz_f64x3, 1);
        svfloat64_t b_z_f64 = svget3_f64(b_xyz_f64x3, 2);

        // Center
        svfloat64_t centered_a_x_f64 = svsub_f64_x(predicate_tail_b64, a_x_f64, centroid_a_x_f64);
        svfloat64_t centered_a_y_f64 = svsub_f64_x(predicate_tail_b64, a_y_f64, centroid_a_y_f64);
        svfloat64_t centered_a_z_f64 = svsub_f64_x(predicate_tail_b64, a_z_f64, centroid_a_z_f64);
        svfloat64_t centered_b_x_f64 = svsub_f64_x(predicate_tail_b64, b_x_f64, centroid_b_x_f64);
        svfloat64_t centered_b_y_f64 = svsub_f64_x(predicate_tail_b64, b_y_f64, centroid_b_y_f64);
        svfloat64_t centered_b_z_f64 = svsub_f64_x(predicate_tail_b64, b_z_f64, centroid_b_z_f64);

        // Rotate: rotated = scale * R * centered_a
        svfloat64_t rotated_a_x_f64 = svmla_f64_x(
            predicate_tail_b64,
            svmla_f64_x(predicate_tail_b64, svmul_f64_x(predicate_tail_b64, scaled_r00_f64, centered_a_x_f64),
                        scaled_r01_f64, centered_a_y_f64),
            scaled_r02_f64, centered_a_z_f64);
        svfloat64_t rotated_a_y_f64 = svmla_f64_x(
            predicate_tail_b64,
            svmla_f64_x(predicate_tail_b64, svmul_f64_x(predicate_tail_b64, scaled_r10_f64, centered_a_x_f64),
                        scaled_r11_f64, centered_a_y_f64),
            scaled_r12_f64, centered_a_z_f64);
        svfloat64_t rotated_a_z_f64 = svmla_f64_x(
            predicate_tail_b64,
            svmla_f64_x(predicate_tail_b64, svmul_f64_x(predicate_tail_b64, scaled_r20_f64, centered_a_x_f64),
                        scaled_r21_f64, centered_a_y_f64),
            scaled_r22_f64, centered_a_z_f64);

        // Delta
        svfloat64_t delta_x_f64 = svsub_f64_x(predicate_tail_b64, rotated_a_x_f64, centered_b_x_f64);
        svfloat64_t delta_y_f64 = svsub_f64_x(predicate_tail_b64, rotated_a_y_f64, centered_b_y_f64);
        svfloat64_t delta_z_f64 = svsub_f64_x(predicate_tail_b64, rotated_a_z_f64, centered_b_z_f64);

        // Accumulate
        sum_squared_f64 = svmla_f64_x(predicate_tail_b64, sum_squared_f64, delta_x_f64, delta_x_f64);
        sum_squared_f64 = svmla_f64_x(predicate_tail_b64, sum_squared_f64, delta_y_f64, delta_y_f64);
        sum_squared_f64 = svmla_f64_x(predicate_tail_b64, sum_squared_f64, delta_z_f64, delta_z_f64);

        index += svcntd();
        predicate_tail_b64 = svwhilelt_b64(index, n);
    }

    return svaddv_f64(predicate_body_b64, sum_squared_f64);
}

__arm_locally_streaming static void nk_rmsd_f64_smef64_kernel_(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n,
                                                               nk_f64_t *a_centroid, nk_f64_t *b_centroid,
                                                               nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result) {
    // RMSD uses identity rotation and scale=1.0
    if (rotation) {
        rotation[0] = 1, rotation[1] = 0, rotation[2] = 0;
        rotation[3] = 0, rotation[4] = 1, rotation[5] = 0;
        rotation[6] = 0, rotation[7] = 0, rotation[8] = 1;
    }
    if (scale) *scale = 1.0;

    // SVE accumulators
    svfloat64_t sum_a_x_f64 = svdup_f64(0), sum_a_y_f64 = svdup_f64(0), sum_a_z_f64 = svdup_f64(0);
    svfloat64_t sum_b_x_f64 = svdup_f64(0), sum_b_y_f64 = svdup_f64(0), sum_b_z_f64 = svdup_f64(0);
    svfloat64_t sum_squared_x_f64 = svdup_f64(0), sum_squared_y_f64 = svdup_f64(0), sum_squared_z_f64 = svdup_f64(0);

    svbool_t predicate_body_b64 = svptrue_b64();
    nk_size_t index = 0;
    svbool_t predicate_tail_b64 = svwhilelt_b64(index, n);
    while (svptest_first(predicate_body_b64, predicate_tail_b64)) {
        svfloat64x3_t a_xyz_f64x3 = svld3_f64(predicate_tail_b64, a + index * 3);
        svfloat64_t a_x_f64 = svget3_f64(a_xyz_f64x3, 0);
        svfloat64_t a_y_f64 = svget3_f64(a_xyz_f64x3, 1);
        svfloat64_t a_z_f64 = svget3_f64(a_xyz_f64x3, 2);

        svfloat64x3_t b_xyz_f64x3 = svld3_f64(predicate_tail_b64, b + index * 3);
        svfloat64_t b_x_f64 = svget3_f64(b_xyz_f64x3, 0);
        svfloat64_t b_y_f64 = svget3_f64(b_xyz_f64x3, 1);
        svfloat64_t b_z_f64 = svget3_f64(b_xyz_f64x3, 2);

        // Centroid sums
        sum_a_x_f64 = svadd_f64_x(predicate_tail_b64, sum_a_x_f64, a_x_f64);
        sum_a_y_f64 = svadd_f64_x(predicate_tail_b64, sum_a_y_f64, a_y_f64);
        sum_a_z_f64 = svadd_f64_x(predicate_tail_b64, sum_a_z_f64, a_z_f64);
        sum_b_x_f64 = svadd_f64_x(predicate_tail_b64, sum_b_x_f64, b_x_f64);
        sum_b_y_f64 = svadd_f64_x(predicate_tail_b64, sum_b_y_f64, b_y_f64);
        sum_b_z_f64 = svadd_f64_x(predicate_tail_b64, sum_b_z_f64, b_z_f64);

        // Squared differences
        svfloat64_t delta_x_f64 = svsub_f64_x(predicate_tail_b64, a_x_f64, b_x_f64);
        svfloat64_t delta_y_f64 = svsub_f64_x(predicate_tail_b64, a_y_f64, b_y_f64);
        svfloat64_t delta_z_f64 = svsub_f64_x(predicate_tail_b64, a_z_f64, b_z_f64);
        sum_squared_x_f64 = svmla_f64_x(predicate_tail_b64, sum_squared_x_f64, delta_x_f64, delta_x_f64);
        sum_squared_y_f64 = svmla_f64_x(predicate_tail_b64, sum_squared_y_f64, delta_y_f64, delta_y_f64);
        sum_squared_z_f64 = svmla_f64_x(predicate_tail_b64, sum_squared_z_f64, delta_z_f64, delta_z_f64);

        index += svcntd();
        predicate_tail_b64 = svwhilelt_b64(index, n);
    }

    // Reduce to scalars
    nk_f64_t sum_a_x = svaddv_f64(predicate_body_b64, sum_a_x_f64);
    nk_f64_t sum_a_y = svaddv_f64(predicate_body_b64, sum_a_y_f64);
    nk_f64_t sum_a_z = svaddv_f64(predicate_body_b64, sum_a_z_f64);
    nk_f64_t sum_b_x = svaddv_f64(predicate_body_b64, sum_b_x_f64);
    nk_f64_t sum_b_y = svaddv_f64(predicate_body_b64, sum_b_y_f64);
    nk_f64_t sum_b_z = svaddv_f64(predicate_body_b64, sum_b_z_f64);
    nk_f64_t total_squared_x = svaddv_f64(predicate_body_b64, sum_squared_x_f64);
    nk_f64_t total_squared_y = svaddv_f64(predicate_body_b64, sum_squared_y_f64);
    nk_f64_t total_squared_z = svaddv_f64(predicate_body_b64, sum_squared_z_f64);

    // Compute centroids
    nk_f64_t inverse_n = 1.0 / (nk_f64_t)n;
    nk_f64_t centroid_a_x = sum_a_x * inverse_n;
    nk_f64_t centroid_a_y = sum_a_y * inverse_n;
    nk_f64_t centroid_a_z = sum_a_z * inverse_n;
    nk_f64_t centroid_b_x = sum_b_x * inverse_n;
    nk_f64_t centroid_b_y = sum_b_y * inverse_n;
    nk_f64_t centroid_b_z = sum_b_z * inverse_n;

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

    *result = nk_f64_sqrt_smef64_(sum_squared * inverse_n - mean_diff_sq);
}

__arm_locally_streaming static void nk_kabsch_f64_smef64_kernel_(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n,
                                                                 nk_f64_t *a_centroid, nk_f64_t *b_centroid,
                                                                 nk_f64_t *rotation, nk_f64_t *scale,
                                                                 nk_f64_t *result) {
    // Phase 1: SVE cross-covariance loop
    svfloat64_t sum_a_x_f64 = svdup_f64(0), sum_a_y_f64 = svdup_f64(0), sum_a_z_f64 = svdup_f64(0);
    svfloat64_t sum_b_x_f64 = svdup_f64(0), sum_b_y_f64 = svdup_f64(0), sum_b_z_f64 = svdup_f64(0);
    svfloat64_t covariance_xx_f64 = svdup_f64(0), covariance_xy_f64 = svdup_f64(0), covariance_xz_f64 = svdup_f64(0);
    svfloat64_t covariance_yx_f64 = svdup_f64(0), covariance_yy_f64 = svdup_f64(0), covariance_yz_f64 = svdup_f64(0);
    svfloat64_t covariance_zx_f64 = svdup_f64(0), covariance_zy_f64 = svdup_f64(0), covariance_zz_f64 = svdup_f64(0);

    svbool_t predicate_body_b64 = svptrue_b64();
    nk_size_t index = 0;
    svbool_t predicate_tail_b64 = svwhilelt_b64(index, n);
    while (svptest_first(predicate_body_b64, predicate_tail_b64)) {
        svfloat64x3_t a_xyz_f64x3 = svld3_f64(predicate_tail_b64, a + index * 3);
        svfloat64_t a_x_f64 = svget3_f64(a_xyz_f64x3, 0);
        svfloat64_t a_y_f64 = svget3_f64(a_xyz_f64x3, 1);
        svfloat64_t a_z_f64 = svget3_f64(a_xyz_f64x3, 2);

        svfloat64x3_t b_xyz_f64x3 = svld3_f64(predicate_tail_b64, b + index * 3);
        svfloat64_t b_x_f64 = svget3_f64(b_xyz_f64x3, 0);
        svfloat64_t b_y_f64 = svget3_f64(b_xyz_f64x3, 1);
        svfloat64_t b_z_f64 = svget3_f64(b_xyz_f64x3, 2);

        // Centroid sums
        sum_a_x_f64 = svadd_f64_x(predicate_tail_b64, sum_a_x_f64, a_x_f64);
        sum_a_y_f64 = svadd_f64_x(predicate_tail_b64, sum_a_y_f64, a_y_f64);
        sum_a_z_f64 = svadd_f64_x(predicate_tail_b64, sum_a_z_f64, a_z_f64);
        sum_b_x_f64 = svadd_f64_x(predicate_tail_b64, sum_b_x_f64, b_x_f64);
        sum_b_y_f64 = svadd_f64_x(predicate_tail_b64, sum_b_y_f64, b_y_f64);
        sum_b_z_f64 = svadd_f64_x(predicate_tail_b64, sum_b_z_f64, b_z_f64);

        // Cross-covariance
        covariance_xx_f64 = svmla_f64_x(predicate_tail_b64, covariance_xx_f64, a_x_f64, b_x_f64);
        covariance_xy_f64 = svmla_f64_x(predicate_tail_b64, covariance_xy_f64, a_x_f64, b_y_f64);
        covariance_xz_f64 = svmla_f64_x(predicate_tail_b64, covariance_xz_f64, a_x_f64, b_z_f64);
        covariance_yx_f64 = svmla_f64_x(predicate_tail_b64, covariance_yx_f64, a_y_f64, b_x_f64);
        covariance_yy_f64 = svmla_f64_x(predicate_tail_b64, covariance_yy_f64, a_y_f64, b_y_f64);
        covariance_yz_f64 = svmla_f64_x(predicate_tail_b64, covariance_yz_f64, a_y_f64, b_z_f64);
        covariance_zx_f64 = svmla_f64_x(predicate_tail_b64, covariance_zx_f64, a_z_f64, b_x_f64);
        covariance_zy_f64 = svmla_f64_x(predicate_tail_b64, covariance_zy_f64, a_z_f64, b_y_f64);
        covariance_zz_f64 = svmla_f64_x(predicate_tail_b64, covariance_zz_f64, a_z_f64, b_z_f64);

        index += svcntd();
        predicate_tail_b64 = svwhilelt_b64(index, n);
    }

    // Reduce to scalars
    nk_f64_t sum_a_x = svaddv_f64(predicate_body_b64, sum_a_x_f64);
    nk_f64_t sum_a_y = svaddv_f64(predicate_body_b64, sum_a_y_f64);
    nk_f64_t sum_a_z = svaddv_f64(predicate_body_b64, sum_a_z_f64);
    nk_f64_t sum_b_x = svaddv_f64(predicate_body_b64, sum_b_x_f64);
    nk_f64_t sum_b_y = svaddv_f64(predicate_body_b64, sum_b_y_f64);
    nk_f64_t sum_b_z = svaddv_f64(predicate_body_b64, sum_b_z_f64);

    // Phase 2: scalar centroids and centering correction
    nk_f64_t inverse_n = 1.0 / (nk_f64_t)n;
    nk_f64_t centroid_a_x = sum_a_x * inverse_n;
    nk_f64_t centroid_a_y = sum_a_y * inverse_n;
    nk_f64_t centroid_a_z = sum_a_z * inverse_n;
    nk_f64_t centroid_b_x = sum_b_x * inverse_n;
    nk_f64_t centroid_b_y = sum_b_y * inverse_n;
    nk_f64_t centroid_b_z = sum_b_z * inverse_n;

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

    nk_f64_t cross_covariance[9];
    cross_covariance[0] = svaddv_f64(predicate_body_b64, covariance_xx_f64) - n * centroid_a_x * centroid_b_x;
    cross_covariance[1] = svaddv_f64(predicate_body_b64, covariance_xy_f64) - n * centroid_a_x * centroid_b_y;
    cross_covariance[2] = svaddv_f64(predicate_body_b64, covariance_xz_f64) - n * centroid_a_x * centroid_b_z;
    cross_covariance[3] = svaddv_f64(predicate_body_b64, covariance_yx_f64) - n * centroid_a_y * centroid_b_x;
    cross_covariance[4] = svaddv_f64(predicate_body_b64, covariance_yy_f64) - n * centroid_a_y * centroid_b_y;
    cross_covariance[5] = svaddv_f64(predicate_body_b64, covariance_yz_f64) - n * centroid_a_y * centroid_b_z;
    cross_covariance[6] = svaddv_f64(predicate_body_b64, covariance_zx_f64) - n * centroid_a_z * centroid_b_x;
    cross_covariance[7] = svaddv_f64(predicate_body_b64, covariance_zy_f64) - n * centroid_a_z * centroid_b_y;
    cross_covariance[8] = svaddv_f64(predicate_body_b64, covariance_zz_f64) - n * centroid_a_z * centroid_b_z;

    // Phase 3: f64 SVD (fixes NEON precision bug — no narrowing to f32)
    nk_f64_t svd_u[9], svd_s[9], svd_v[9];
    nk_svd3x3_f64_(cross_covariance, svd_u, svd_s, svd_v);

    // Phase 4: R = V * Uᵀ (all f64)
    nk_f64_t rotation_matrix[9];
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
    if (nk_det3x3_f64_(rotation_matrix) < 0) {
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
    if (scale) *scale = 1.0;

    // Phase 5: SVE transformed SSD (all f64, no narrowing)
    nk_f64_t sum_squared = nk_transformed_ssd_f64_smef64_(a, b, n, rotation_matrix, 1.0, centroid_a_x, centroid_a_y,
                                                          centroid_a_z, centroid_b_x, centroid_b_y, centroid_b_z);

    // Phase 6: SVE sqrt
    *result = nk_f64_sqrt_smef64_(sum_squared * inverse_n);
}

__arm_locally_streaming static void nk_umeyama_f64_smef64_kernel_(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n,
                                                                  nk_f64_t *a_centroid, nk_f64_t *b_centroid,
                                                                  nk_f64_t *rotation, nk_f64_t *scale,
                                                                  nk_f64_t *result) {
    // Phase 1: SVE cross-covariance loop with variance accumulator
    svfloat64_t sum_a_x_f64 = svdup_f64(0), sum_a_y_f64 = svdup_f64(0), sum_a_z_f64 = svdup_f64(0);
    svfloat64_t sum_b_x_f64 = svdup_f64(0), sum_b_y_f64 = svdup_f64(0), sum_b_z_f64 = svdup_f64(0);
    svfloat64_t covariance_xx_f64 = svdup_f64(0), covariance_xy_f64 = svdup_f64(0), covariance_xz_f64 = svdup_f64(0);
    svfloat64_t covariance_yx_f64 = svdup_f64(0), covariance_yy_f64 = svdup_f64(0), covariance_yz_f64 = svdup_f64(0);
    svfloat64_t covariance_zx_f64 = svdup_f64(0), covariance_zy_f64 = svdup_f64(0), covariance_zz_f64 = svdup_f64(0);
    svfloat64_t variance_a_f64 = svdup_f64(0);

    svbool_t predicate_body_b64 = svptrue_b64();
    nk_size_t index = 0;
    svbool_t predicate_tail_b64 = svwhilelt_b64(index, n);
    while (svptest_first(predicate_body_b64, predicate_tail_b64)) {
        svfloat64x3_t a_xyz_f64x3 = svld3_f64(predicate_tail_b64, a + index * 3);
        svfloat64_t a_x_f64 = svget3_f64(a_xyz_f64x3, 0);
        svfloat64_t a_y_f64 = svget3_f64(a_xyz_f64x3, 1);
        svfloat64_t a_z_f64 = svget3_f64(a_xyz_f64x3, 2);

        svfloat64x3_t b_xyz_f64x3 = svld3_f64(predicate_tail_b64, b + index * 3);
        svfloat64_t b_x_f64 = svget3_f64(b_xyz_f64x3, 0);
        svfloat64_t b_y_f64 = svget3_f64(b_xyz_f64x3, 1);
        svfloat64_t b_z_f64 = svget3_f64(b_xyz_f64x3, 2);

        // Centroid sums
        sum_a_x_f64 = svadd_f64_x(predicate_tail_b64, sum_a_x_f64, a_x_f64);
        sum_a_y_f64 = svadd_f64_x(predicate_tail_b64, sum_a_y_f64, a_y_f64);
        sum_a_z_f64 = svadd_f64_x(predicate_tail_b64, sum_a_z_f64, a_z_f64);
        sum_b_x_f64 = svadd_f64_x(predicate_tail_b64, sum_b_x_f64, b_x_f64);
        sum_b_y_f64 = svadd_f64_x(predicate_tail_b64, sum_b_y_f64, b_y_f64);
        sum_b_z_f64 = svadd_f64_x(predicate_tail_b64, sum_b_z_f64, b_z_f64);

        // Cross-covariance
        covariance_xx_f64 = svmla_f64_x(predicate_tail_b64, covariance_xx_f64, a_x_f64, b_x_f64);
        covariance_xy_f64 = svmla_f64_x(predicate_tail_b64, covariance_xy_f64, a_x_f64, b_y_f64);
        covariance_xz_f64 = svmla_f64_x(predicate_tail_b64, covariance_xz_f64, a_x_f64, b_z_f64);
        covariance_yx_f64 = svmla_f64_x(predicate_tail_b64, covariance_yx_f64, a_y_f64, b_x_f64);
        covariance_yy_f64 = svmla_f64_x(predicate_tail_b64, covariance_yy_f64, a_y_f64, b_y_f64);
        covariance_yz_f64 = svmla_f64_x(predicate_tail_b64, covariance_yz_f64, a_y_f64, b_z_f64);
        covariance_zx_f64 = svmla_f64_x(predicate_tail_b64, covariance_zx_f64, a_z_f64, b_x_f64);
        covariance_zy_f64 = svmla_f64_x(predicate_tail_b64, covariance_zy_f64, a_z_f64, b_y_f64);
        covariance_zz_f64 = svmla_f64_x(predicate_tail_b64, covariance_zz_f64, a_z_f64, b_z_f64);

        // Variance of A: Σ(ax² + ay² + az²)
        variance_a_f64 = svmla_f64_x(predicate_tail_b64, variance_a_f64, a_x_f64, a_x_f64);
        variance_a_f64 = svmla_f64_x(predicate_tail_b64, variance_a_f64, a_y_f64, a_y_f64);
        variance_a_f64 = svmla_f64_x(predicate_tail_b64, variance_a_f64, a_z_f64, a_z_f64);

        index += svcntd();
        predicate_tail_b64 = svwhilelt_b64(index, n);
    }

    // Reduce to scalars
    nk_f64_t sum_a_x = svaddv_f64(predicate_body_b64, sum_a_x_f64);
    nk_f64_t sum_a_y = svaddv_f64(predicate_body_b64, sum_a_y_f64);
    nk_f64_t sum_a_z = svaddv_f64(predicate_body_b64, sum_a_z_f64);
    nk_f64_t sum_b_x = svaddv_f64(predicate_body_b64, sum_b_x_f64);
    nk_f64_t sum_b_y = svaddv_f64(predicate_body_b64, sum_b_y_f64);
    nk_f64_t sum_b_z = svaddv_f64(predicate_body_b64, sum_b_z_f64);
    nk_f64_t sum_sq_a = svaddv_f64(predicate_body_b64, variance_a_f64);

    // Phase 2: scalar centroids and centering correction
    nk_f64_t inverse_n = 1.0 / (nk_f64_t)n;
    nk_f64_t centroid_a_x = sum_a_x * inverse_n;
    nk_f64_t centroid_a_y = sum_a_y * inverse_n;
    nk_f64_t centroid_a_z = sum_a_z * inverse_n;
    nk_f64_t centroid_b_x = sum_b_x * inverse_n;
    nk_f64_t centroid_b_y = sum_b_y * inverse_n;
    nk_f64_t centroid_b_z = sum_b_z * inverse_n;

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
    nk_f64_t centroid_sq = centroid_a_x * centroid_a_x + centroid_a_y * centroid_a_y + centroid_a_z * centroid_a_z;
    nk_f64_t var_a = sum_sq_a * inverse_n - centroid_sq;

    nk_f64_t cross_covariance[9];
    cross_covariance[0] = svaddv_f64(predicate_body_b64, covariance_xx_f64) - n * centroid_a_x * centroid_b_x;
    cross_covariance[1] = svaddv_f64(predicate_body_b64, covariance_xy_f64) - n * centroid_a_x * centroid_b_y;
    cross_covariance[2] = svaddv_f64(predicate_body_b64, covariance_xz_f64) - n * centroid_a_x * centroid_b_z;
    cross_covariance[3] = svaddv_f64(predicate_body_b64, covariance_yx_f64) - n * centroid_a_y * centroid_b_x;
    cross_covariance[4] = svaddv_f64(predicate_body_b64, covariance_yy_f64) - n * centroid_a_y * centroid_b_y;
    cross_covariance[5] = svaddv_f64(predicate_body_b64, covariance_yz_f64) - n * centroid_a_y * centroid_b_z;
    cross_covariance[6] = svaddv_f64(predicate_body_b64, covariance_zx_f64) - n * centroid_a_z * centroid_b_x;
    cross_covariance[7] = svaddv_f64(predicate_body_b64, covariance_zy_f64) - n * centroid_a_z * centroid_b_y;
    cross_covariance[8] = svaddv_f64(predicate_body_b64, covariance_zz_f64) - n * centroid_a_z * centroid_b_z;

    // Phase 3: f64 SVD
    nk_f64_t svd_u[9], svd_s[9], svd_v[9];
    nk_svd3x3_f64_(cross_covariance, svd_u, svd_s, svd_v);

    // Phase 4: R = V * Uᵀ (all f64)
    nk_f64_t rotation_matrix[9];
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
    nk_f64_t det = nk_det3x3_f64_(rotation_matrix);
    nk_f64_t trace_d_s = svd_s[0] + svd_s[4] + (det < 0 ? -svd_s[8] : svd_s[8]);
    nk_f64_t computed_scale = trace_d_s / (n * var_a);

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

    // Phase 5: SVE transformed SSD (all f64)
    nk_f64_t sum_squared = nk_transformed_ssd_f64_smef64_(a, b, n, rotation_matrix, computed_scale, centroid_a_x,
                                                          centroid_a_y, centroid_a_z, centroid_b_x, centroid_b_y,
                                                          centroid_b_z);

    // Phase 6: SVE sqrt
    *result = nk_f64_sqrt_smef64_(sum_squared * inverse_n);
}

NK_PUBLIC void nk_rmsd_f64_smef64(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                                  nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result) {
    nk_rmsd_f64_smef64_kernel_(a, b, n, a_centroid, b_centroid, rotation, scale, result);
}

NK_PUBLIC void nk_kabsch_f64_smef64(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                                    nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result) {
    nk_kabsch_f64_smef64_kernel_(a, b, n, a_centroid, b_centroid, rotation, scale, result);
}

NK_PUBLIC void nk_umeyama_f64_smef64(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                                     nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result) {
    nk_umeyama_f64_smef64_kernel_(a, b, n, a_centroid, b_centroid, rotation, scale, result);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_SMEF64
#endif // NK_TARGET_ARM_
#endif // NK_MESH_SMEF64_H

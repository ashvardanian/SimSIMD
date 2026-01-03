/**
 *  @brief SIMD-accelerated Dot Products for Real and Complex Numbers optimized for SIMD-free CPUs.
 *  @file include/numkong/mesh/serial.h
 *  @sa include/numkong/mesh.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 */
#ifndef NK_MESH_SERIAL_H
#define NK_MESH_SERIAL_H
#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

/*  Constants for the McAdams 3x3 SVD algorithm.
 *  gamma = (sqrt(8) + 3)^2 / 4 = 5.828427124
 *  cstar = cos(pi/8), sstar = sin(pi/8)
 */
#define NK_SVD_GAMMA_F32   5.828427124f
#define NK_SVD_CSTAR_F32   0.923879532f
#define NK_SVD_SSTAR_F32   0.3826834323f
#define NK_SVD_EPSILON_F32 1e-6f

#define NK_SVD_GAMMA_F64   5.828427124746190
#define NK_SVD_CSTAR_F64   0.9238795325112867
#define NK_SVD_SSTAR_F64   0.3826834323650898
#define NK_SVD_EPSILON_F64 1e-12

/*  Type-Generic SVD Helper Macros
 *  These macros generate f32 and f64 versions of SVD helper functions
 *  used by the Kabsch and Umeyama algorithms.
 */

#define NK_MAKE_COND_SWAP(type)                                                        \
    NK_INTERNAL void nk_cond_swap__##type(int c, nk_##type##_t *x, nk_##type##_t *y) { \
        nk_##type##_t temp = *x;                                                       \
        *x = c ? *y : *x;                                                              \
        *y = c ? temp : *y;                                                            \
    }

#define NK_MAKE_COND_NEG_SWAP(type)                                                        \
    NK_INTERNAL void nk_cond_neg_swap__##type(int c, nk_##type##_t *x, nk_##type##_t *y) { \
        nk_##type##_t neg_x = -*x;                                                         \
        *x = c ? *y : *x;                                                                  \
        *y = c ? neg_x : *y;                                                               \
    }

#define NK_MAKE_APPROX_GIVENS_QUAT(type, gamma, cstar, sstar, compute_rsqrt)                                \
    NK_INTERNAL void nk_approx_givens_quat__##type(nk_##type##_t a11, nk_##type##_t a12, nk_##type##_t a22, \
                                                   nk_##type##_t *cos_half, nk_##type##_t *sin_half) {      \
        *cos_half = 2 * (a11 - a22), *sin_half = a12;                                                       \
        int use_givens = gamma * (*sin_half) * (*sin_half) < (*cos_half) * (*cos_half);                     \
        nk_##type##_t w = compute_rsqrt((*cos_half) * (*cos_half) + (*sin_half) * (*sin_half));             \
        *cos_half = use_givens ? w * (*cos_half) : cstar;                                                   \
        *sin_half = use_givens ? w * (*sin_half) : sstar;                                                   \
    }

#define NK_MAKE_JACOBI_CONJUGATION(type)                                                             \
    NK_INTERNAL void nk_jacobi_conjugation__##type(                                                  \
        int idx_x, int idx_y, int idx_z, nk_##type##_t *s11, nk_##type##_t *s21, nk_##type##_t *s22, \
        nk_##type##_t *s31, nk_##type##_t *s32, nk_##type##_t *s33, nk_##type##_t *quaternion) {     \
        nk_##type##_t cos_half, sin_half;                                                            \
        nk_approx_givens_quat__##type(*s11, *s21, *s22, &cos_half, &sin_half);                       \
        nk_##type##_t scale = cos_half * cos_half + sin_half * sin_half;                             \
        nk_##type##_t cos_theta = (cos_half * cos_half - sin_half * sin_half) / scale;               \
        nk_##type##_t sin_theta = (2 * sin_half * cos_half) / scale;                                 \
        nk_##type##_t s11_old = *s11, s21_old = *s21, s22_old = *s22;                                \
        nk_##type##_t s31_old = *s31, s32_old = *s32, s33_old = *s33;                                \
        *s11 = cos_theta * (cos_theta * s11_old + sin_theta * s21_old) +                             \
               sin_theta * (cos_theta * s21_old + sin_theta * s22_old);                              \
        *s21 = cos_theta * (-sin_theta * s11_old + cos_theta * s21_old) +                            \
               sin_theta * (-sin_theta * s21_old + cos_theta * s22_old);                             \
        *s22 = -sin_theta * (-sin_theta * s11_old + cos_theta * s21_old) +                           \
               cos_theta * (-sin_theta * s21_old + cos_theta * s22_old);                             \
        *s31 = cos_theta * s31_old + sin_theta * s32_old;                                            \
        *s32 = -sin_theta * s31_old + cos_theta * s32_old;                                           \
        *s33 = s33_old;                                                                              \
        /* Update quaternion accumulator */                                                          \
        nk_##type##_t quat_temp[3];                                                                  \
        quat_temp[0] = quaternion[0] * sin_half;                                                     \
        quat_temp[1] = quaternion[1] * sin_half;                                                     \
        quat_temp[2] = quaternion[2] * sin_half;                                                     \
        sin_half *= quaternion[3];                                                                   \
        quaternion[0] *= cos_half, quaternion[1] *= cos_half;                                        \
        quaternion[2] *= cos_half, quaternion[3] *= cos_half;                                        \
        quaternion[idx_z] += sin_half, quaternion[3] -= quat_temp[idx_z];                            \
        quaternion[idx_x] += quat_temp[idx_y], quaternion[idx_y] -= quat_temp[idx_x];                \
        /* Cyclic permutation of matrix elements */                                                  \
        s11_old = *s22, s21_old = *s32, s22_old = *s33;                                              \
        s31_old = *s21, s32_old = *s31, s33_old = *s11;                                              \
        *s11 = s11_old, *s21 = s21_old, *s22 = s22_old;                                              \
        *s31 = s31_old, *s32 = s32_old, *s33 = s33_old;                                              \
    }

#define NK_MAKE_QUAT_TO_MAT3(type)                                                               \
    NK_INTERNAL void nk_quat_to_mat3__##type(nk_##type##_t const *quat, nk_##type##_t *matrix) { \
        nk_##type##_t w = quat[3], x = quat[0], y = quat[1], z = quat[2];                        \
        nk_##type##_t q_xx = x * x, q_yy = y * y, q_zz = z * z;                                  \
        nk_##type##_t q_xz = x * z, q_xy = x * y, q_yz = y * z;                                  \
        nk_##type##_t q_wx = w * x, q_wy = w * y, q_wz = w * z;                                  \
        matrix[0] = 1 - 2 * (q_yy + q_zz), matrix[1] = 2 * (q_xy - q_wz);                        \
        matrix[2] = 2 * (q_xz + q_wy);                                                           \
        matrix[3] = 2 * (q_xy + q_wz), matrix[4] = 1 - 2 * (q_xx + q_zz);                        \
        matrix[5] = 2 * (q_yz - q_wx);                                                           \
        matrix[6] = 2 * (q_xz - q_wy), matrix[7] = 2 * (q_yz + q_wx);                            \
        matrix[8] = 1 - 2 * (q_xx + q_yy);                                                       \
    }

#define NK_MAKE_JACOBI_EIGENANALYSIS(type, compute_rsqrt)                                                        \
    NK_INTERNAL void nk_jacobi_eigenanalysis__##type(nk_##type##_t *s11, nk_##type##_t *s21, nk_##type##_t *s22, \
                                                     nk_##type##_t *s31, nk_##type##_t *s32, nk_##type##_t *s33, \
                                                     nk_##type##_t *quaternion) {                                \
        quaternion[0] = 0, quaternion[1] = 0, quaternion[2] = 0, quaternion[3] = 1;                              \
        /* 16 iterations for better convergence with repeated eigenvalues and identity-like matrices */          \
        for (int iter = 0; iter < 16; iter++) {                                                                  \
            nk_jacobi_conjugation__##type(0, 1, 2, s11, s21, s22, s31, s32, s33, quaternion);                    \
            nk_jacobi_conjugation__##type(1, 2, 0, s11, s21, s22, s31, s32, s33, quaternion);                    \
            nk_jacobi_conjugation__##type(2, 0, 1, s11, s21, s22, s31, s32, s33, quaternion);                    \
        }                                                                                                        \
        nk_##type##_t norm = compute_rsqrt(quaternion[0] * quaternion[0] + quaternion[1] * quaternion[1] +       \
                                           quaternion[2] * quaternion[2] + quaternion[3] * quaternion[3]);       \
        quaternion[0] *= norm, quaternion[1] *= norm;                                                            \
        quaternion[2] *= norm, quaternion[3] *= norm;                                                            \
    }

#define NK_MAKE_QR_GIVENS_QUAT(type, epsilon, compute_rsqrt)                                                \
    NK_INTERNAL void nk_qr_givens_quat__##type(nk_##type##_t a1, nk_##type##_t a2, nk_##type##_t *cos_half, \
                                               nk_##type##_t *sin_half) {                                   \
        nk_##type##_t a1_sq_plus_a2_sq = a1 * a1 + a2 * a2;                                                 \
        nk_##type##_t rho = a1_sq_plus_a2_sq * compute_rsqrt(a1_sq_plus_a2_sq);                             \
        rho = a1_sq_plus_a2_sq > epsilon ? rho : 0;                                                         \
        *sin_half = rho > epsilon ? a2 : 0;                                                                 \
        nk_##type##_t abs_a1 = a1 < 0 ? -a1 : a1;                                                           \
        nk_##type##_t max_rho = rho > epsilon ? rho : epsilon;                                              \
        *cos_half = abs_a1 + max_rho;                                                                       \
        int should_swap = a1 < 0;                                                                           \
        nk_cond_swap__##type(should_swap, sin_half, cos_half);                                              \
        nk_##type##_t w = compute_rsqrt((*cos_half) * (*cos_half) + (*sin_half) * (*sin_half));             \
        *cos_half *= w, *sin_half *= w;                                                                     \
    }

#define NK_MAKE_SORT_SINGULAR_VALUES(type)                                                 \
    NK_INTERNAL void nk_sort_singular_values__##type(nk_##type##_t *b, nk_##type##_t *v) { \
        nk_##type##_t rho1 = b[0] * b[0] + b[3] * b[3] + b[6] * b[6];                      \
        nk_##type##_t rho2 = b[1] * b[1] + b[4] * b[4] + b[7] * b[7];                      \
        nk_##type##_t rho3 = b[2] * b[2] + b[5] * b[5] + b[8] * b[8];                      \
        int should_swap;                                                                   \
        /* Sort columns by descending singular value magnitude */                          \
        should_swap = rho1 < rho2;                                                         \
        nk_cond_neg_swap__##type(should_swap, &b[0], &b[1]);                               \
        nk_cond_neg_swap__##type(should_swap, &v[0], &v[1]);                               \
        nk_cond_neg_swap__##type(should_swap, &b[3], &b[4]);                               \
        nk_cond_neg_swap__##type(should_swap, &v[3], &v[4]);                               \
        nk_cond_neg_swap__##type(should_swap, &b[6], &b[7]);                               \
        nk_cond_neg_swap__##type(should_swap, &v[6], &v[7]);                               \
        nk_cond_swap__##type(should_swap, &rho1, &rho2);                                   \
        should_swap = rho1 < rho3;                                                         \
        nk_cond_neg_swap__##type(should_swap, &b[0], &b[2]);                               \
        nk_cond_neg_swap__##type(should_swap, &v[0], &v[2]);                               \
        nk_cond_neg_swap__##type(should_swap, &b[3], &b[5]);                               \
        nk_cond_neg_swap__##type(should_swap, &v[3], &v[5]);                               \
        nk_cond_neg_swap__##type(should_swap, &b[6], &b[8]);                               \
        nk_cond_neg_swap__##type(should_swap, &v[6], &v[8]);                               \
        nk_cond_swap__##type(should_swap, &rho1, &rho3);                                   \
        should_swap = rho2 < rho3;                                                         \
        nk_cond_neg_swap__##type(should_swap, &b[1], &b[2]);                               \
        nk_cond_neg_swap__##type(should_swap, &v[1], &v[2]);                               \
        nk_cond_neg_swap__##type(should_swap, &b[4], &b[5]);                               \
        nk_cond_neg_swap__##type(should_swap, &v[4], &v[5]);                               \
        nk_cond_neg_swap__##type(should_swap, &b[7], &b[8]);                               \
        nk_cond_neg_swap__##type(should_swap, &v[7], &v[8]);                               \
    }

#define NK_MAKE_QR_DECOMPOSITION(type)                                                                               \
    NK_INTERNAL void nk_qr_decomposition__##type(nk_##type##_t const *input, nk_##type##_t *q, nk_##type##_t *r) {   \
        nk_##type##_t cos_half_1, sin_half_1;                                                                        \
        nk_##type##_t cos_half_2, sin_half_2;                                                                        \
        nk_##type##_t cos_half_3, sin_half_3;                                                                        \
        nk_##type##_t cos_theta, sin_theta;                                                                          \
        nk_##type##_t rotation_temp[9], matrix_temp[9];                                                              \
        /* First Givens rotation (zero input[3]) */                                                                  \
        nk_qr_givens_quat__##type(input[0], input[3], &cos_half_1, &sin_half_1);                                     \
        cos_theta = 1 - 2 * sin_half_1 * sin_half_1;                                                                 \
        sin_theta = 2 * cos_half_1 * sin_half_1;                                                                     \
        rotation_temp[0] = cos_theta * input[0] + sin_theta * input[3];                                              \
        rotation_temp[1] = cos_theta * input[1] + sin_theta * input[4];                                              \
        rotation_temp[2] = cos_theta * input[2] + sin_theta * input[5];                                              \
        rotation_temp[3] = -sin_theta * input[0] + cos_theta * input[3];                                             \
        rotation_temp[4] = -sin_theta * input[1] + cos_theta * input[4];                                             \
        rotation_temp[5] = -sin_theta * input[2] + cos_theta * input[5];                                             \
        rotation_temp[6] = input[6], rotation_temp[7] = input[7];                                                    \
        rotation_temp[8] = input[8];                                                                                 \
        /* Second Givens rotation (zero rotation_temp[6]) */                                                         \
        nk_qr_givens_quat__##type(rotation_temp[0], rotation_temp[6], &cos_half_2, &sin_half_2);                     \
        cos_theta = 1 - 2 * sin_half_2 * sin_half_2;                                                                 \
        sin_theta = 2 * cos_half_2 * sin_half_2;                                                                     \
        matrix_temp[0] = cos_theta * rotation_temp[0] + sin_theta * rotation_temp[6];                                \
        matrix_temp[1] = cos_theta * rotation_temp[1] + sin_theta * rotation_temp[7];                                \
        matrix_temp[2] = cos_theta * rotation_temp[2] + sin_theta * rotation_temp[8];                                \
        matrix_temp[3] = rotation_temp[3], matrix_temp[4] = rotation_temp[4];                                        \
        matrix_temp[5] = rotation_temp[5];                                                                           \
        matrix_temp[6] = -sin_theta * rotation_temp[0] + cos_theta * rotation_temp[6];                               \
        matrix_temp[7] = -sin_theta * rotation_temp[1] + cos_theta * rotation_temp[7];                               \
        matrix_temp[8] = -sin_theta * rotation_temp[2] + cos_theta * rotation_temp[8];                               \
        /* Third Givens rotation (zero matrix_temp[7]) */                                                            \
        nk_qr_givens_quat__##type(matrix_temp[4], matrix_temp[7], &cos_half_3, &sin_half_3);                         \
        cos_theta = 1 - 2 * sin_half_3 * sin_half_3;                                                                 \
        sin_theta = 2 * cos_half_3 * sin_half_3;                                                                     \
        r[0] = matrix_temp[0], r[1] = matrix_temp[1], r[2] = matrix_temp[2];                                         \
        r[3] = cos_theta * matrix_temp[3] + sin_theta * matrix_temp[6];                                              \
        r[4] = cos_theta * matrix_temp[4] + sin_theta * matrix_temp[7];                                              \
        r[5] = cos_theta * matrix_temp[5] + sin_theta * matrix_temp[8];                                              \
        r[6] = -sin_theta * matrix_temp[3] + cos_theta * matrix_temp[6];                                             \
        r[7] = -sin_theta * matrix_temp[4] + cos_theta * matrix_temp[7];                                             \
        r[8] = -sin_theta * matrix_temp[5] + cos_theta * matrix_temp[8];                                             \
        /* Construct Q = Q1 * Q2 * Q3 (closed-form expressions) */                                                   \
        nk_##type##_t sin_half_1_sq = sin_half_1 * sin_half_1;                                                       \
        nk_##type##_t sin_half_2_sq = sin_half_2 * sin_half_2;                                                       \
        nk_##type##_t sin_half_3_sq = sin_half_3 * sin_half_3;                                                       \
        q[0] = (-1 + 2 * sin_half_1_sq) * (-1 + 2 * sin_half_2_sq);                                                  \
        q[1] = 4 * cos_half_2 * cos_half_3 * (-1 + 2 * sin_half_1_sq) * sin_half_2 * sin_half_3 +                    \
               2 * cos_half_1 * sin_half_1 * (-1 + 2 * sin_half_3_sq);                                               \
        q[2] = 4 * cos_half_1 * cos_half_3 * sin_half_1 * sin_half_3 -                                               \
               2 * cos_half_2 * (-1 + 2 * sin_half_1_sq) * sin_half_2 * (-1 + 2 * sin_half_3_sq);                    \
        q[3] = 2 * cos_half_1 * sin_half_1 * (1 - 2 * sin_half_2_sq);                                                \
        q[4] = -8 * cos_half_1 * cos_half_2 * cos_half_3 * sin_half_1 * sin_half_2 * sin_half_3 +                    \
               (-1 + 2 * sin_half_1_sq) * (-1 + 2 * sin_half_3_sq);                                                  \
        q[5] = -2 * cos_half_3 * sin_half_3 + 4 * sin_half_1 *                                                       \
                                                  (cos_half_3 * sin_half_1 * sin_half_3 +                            \
                                                   cos_half_1 * cos_half_2 * sin_half_2 * (-1 + 2 * sin_half_3_sq)); \
        q[6] = 2 * cos_half_2 * sin_half_2;                                                                          \
        q[7] = 2 * cos_half_3 * (1 - 2 * sin_half_2_sq) * sin_half_3;                                                \
        q[8] = (-1 + 2 * sin_half_2_sq) * (-1 + 2 * sin_half_3_sq);                                                  \
    }

#define NK_MAKE_SVD3X3(type, compute_sqrt)                                                                   \
    NK_INTERNAL void nk_svd3x3_##type##_(nk_##type##_t const *a, nk_##type##_t *svd_u, nk_##type##_t *svd_s, \
                                         nk_##type##_t *svd_v) {                                             \
        /* Compute A^T * A (symmetric) */                                                                    \
        nk_##type##_t ata[9];                                                                                \
        ata[0] = a[0] * a[0] + a[3] * a[3] + a[6] * a[6];                                                    \
        ata[1] = a[0] * a[1] + a[3] * a[4] + a[6] * a[7];                                                    \
        ata[2] = a[0] * a[2] + a[3] * a[5] + a[6] * a[8];                                                    \
        ata[3] = ata[1];                                                                                     \
        ata[4] = a[1] * a[1] + a[4] * a[4] + a[7] * a[7];                                                    \
        ata[5] = a[1] * a[2] + a[4] * a[5] + a[7] * a[8];                                                    \
        ata[6] = ata[2];                                                                                     \
        ata[7] = ata[5];                                                                                     \
        ata[8] = a[2] * a[2] + a[5] * a[5] + a[8] * a[8];                                                    \
        /* Jacobi eigenanalysis of A^T * A */                                                                \
        nk_##type##_t quaternion[4];                                                                         \
        nk_jacobi_eigenanalysis__##type(&ata[0], &ata[1], &ata[4], &ata[2], &ata[5], &ata[8], quaternion);   \
        nk_quat_to_mat3__##type(quaternion, svd_v);                                                          \
        /* B = A * V */                                                                                      \
        nk_##type##_t product[9];                                                                            \
        product[0] = a[0] * svd_v[0] + a[1] * svd_v[3] + a[2] * svd_v[6];                                    \
        product[1] = a[0] * svd_v[1] + a[1] * svd_v[4] + a[2] * svd_v[7];                                    \
        product[2] = a[0] * svd_v[2] + a[1] * svd_v[5] + a[2] * svd_v[8];                                    \
        product[3] = a[3] * svd_v[0] + a[4] * svd_v[3] + a[5] * svd_v[6];                                    \
        product[4] = a[3] * svd_v[1] + a[4] * svd_v[4] + a[5] * svd_v[7];                                    \
        product[5] = a[3] * svd_v[2] + a[4] * svd_v[5] + a[5] * svd_v[8];                                    \
        product[6] = a[6] * svd_v[0] + a[7] * svd_v[3] + a[8] * svd_v[6];                                    \
        product[7] = a[6] * svd_v[1] + a[7] * svd_v[4] + a[8] * svd_v[7];                                    \
        product[8] = a[6] * svd_v[2] + a[7] * svd_v[5] + a[8] * svd_v[8];                                    \
        /* Sort singular values and update V */                                                              \
        nk_sort_singular_values__##type(product, svd_v);                                                     \
        /* Compute singular values from column norms of sorted B (before QR orthogonalizes them) */          \
        /* These are the true singular values: sqrt(||col_i||^2) */                                          \
        nk_##type##_t s1_sq = product[0] * product[0] + product[3] * product[3] + product[6] * product[6];   \
        nk_##type##_t s2_sq = product[1] * product[1] + product[4] * product[4] + product[7] * product[7];   \
        nk_##type##_t s3_sq = product[2] * product[2] + product[5] * product[5] + product[8] * product[8];   \
        /* QR decomposition: B = U * R (we only need U for the rotation) */                                  \
        nk_##type##_t qr_r[9];                                                                               \
        nk_qr_decomposition__##type(product, svd_u, qr_r);                                                   \
        /* Store singular values in diagonal of svd_s (rest is zero for compatibility) */                    \
        svd_s[0] = compute_sqrt(s1_sq), svd_s[1] = 0, svd_s[2] = 0;                                          \
        svd_s[3] = 0, svd_s[4] = compute_sqrt(s2_sq), svd_s[5] = 0;                                          \
        svd_s[6] = 0, svd_s[7] = 0, svd_s[8] = compute_sqrt(s3_sq);                                          \
    }

#define NK_MAKE_DET3X3(type)                                                             \
    NK_INTERNAL nk_##type##_t nk_det3x3_##type##_(nk_##type##_t const *m) {              \
        return m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) + \
               m[2] * (m[3] * m[7] - m[4] * m[6]);                                       \
    }

/* Generate f32 SVD helpers */
NK_MAKE_COND_SWAP(f32)
NK_MAKE_COND_NEG_SWAP(f32)
NK_MAKE_APPROX_GIVENS_QUAT(f32, NK_SVD_GAMMA_F32, NK_SVD_CSTAR_F32, NK_SVD_SSTAR_F32, nk_f32_rsqrt_serial)
NK_MAKE_JACOBI_CONJUGATION(f32)
NK_MAKE_QUAT_TO_MAT3(f32)
NK_MAKE_JACOBI_EIGENANALYSIS(f32, nk_f32_rsqrt_serial)
NK_MAKE_QR_GIVENS_QUAT(f32, NK_SVD_EPSILON_F32, nk_f32_rsqrt_serial)
NK_MAKE_SORT_SINGULAR_VALUES(f32)
NK_MAKE_QR_DECOMPOSITION(f32)
NK_MAKE_SVD3X3(f32, nk_f32_sqrt_serial)
NK_MAKE_DET3X3(f32)

/* Generate f64 SVD helpers */
NK_MAKE_COND_SWAP(f64)
NK_MAKE_COND_NEG_SWAP(f64)
NK_MAKE_APPROX_GIVENS_QUAT(f64, NK_SVD_GAMMA_F64, NK_SVD_CSTAR_F64, NK_SVD_SSTAR_F64, nk_f64_rsqrt_serial)
NK_MAKE_JACOBI_CONJUGATION(f64)
NK_MAKE_QUAT_TO_MAT3(f64)
NK_MAKE_JACOBI_EIGENANALYSIS(f64, nk_f64_rsqrt_serial)
NK_MAKE_QR_GIVENS_QUAT(f64, NK_SVD_EPSILON_F64, nk_f64_rsqrt_serial)
NK_MAKE_SORT_SINGULAR_VALUES(f64)
NK_MAKE_QR_DECOMPOSITION(f64)
NK_MAKE_SVD3X3(f64, nk_f64_sqrt_serial)
NK_MAKE_DET3X3(f64)

/*  RMSD (Root Mean Square Deviation) without optimal superposition.
 *  Simply computes the RMS of distances between corresponding points.
 */
#define NK_MAKE_RMSD(name, input_type, accumulator_type, output_type, load_and_convert, compute_sqrt)              \
    NK_PUBLIC void nk_rmsd_##input_type##_##name(nk_##input_type##_t const *a, nk_##input_type##_t const *b,       \
                                                 nk_size_t n, nk_##output_type##_t *a_centroid,                    \
                                                 nk_##output_type##_t *b_centroid, nk_##output_type##_t *rotation, \
                                                 nk_##output_type##_t *scale, nk_##output_type##_t *result) {      \
        nk_##accumulator_type##_t sum_a_x = 0, sum_a_y = 0, sum_a_z = 0;                                           \
        nk_##accumulator_type##_t sum_b_x = 0, sum_b_y = 0, sum_b_z = 0;                                           \
        nk_##accumulator_type##_t val_a_x, val_a_y, val_a_z, val_b_x, val_b_y, val_b_z;                            \
        for (nk_size_t i = 0; i < n; ++i) {                                                                        \
            load_and_convert(a + i * 3 + 0, &val_a_x), load_and_convert(a + i * 3 + 1, &val_a_y);                  \
            load_and_convert(a + i * 3 + 2, &val_a_z), load_and_convert(b + i * 3 + 0, &val_b_x);                  \
            load_and_convert(b + i * 3 + 1, &val_b_y), load_and_convert(b + i * 3 + 2, &val_b_z);                  \
            sum_a_x += val_a_x, sum_a_y += val_a_y, sum_a_z += val_a_z;                                            \
            sum_b_x += val_b_x, sum_b_y += val_b_y, sum_b_z += val_b_z;                                            \
        }                                                                                                          \
        nk_##accumulator_type##_t inv_n = (nk_##accumulator_type##_t)1.0 / n;                                      \
        nk_##accumulator_type##_t centroid_a_x = sum_a_x * inv_n;                                                  \
        nk_##accumulator_type##_t centroid_a_y = sum_a_y * inv_n;                                                  \
        nk_##accumulator_type##_t centroid_a_z = sum_a_z * inv_n;                                                  \
        nk_##accumulator_type##_t centroid_b_x = sum_b_x * inv_n;                                                  \
        nk_##accumulator_type##_t centroid_b_y = sum_b_y * inv_n;                                                  \
        nk_##accumulator_type##_t centroid_b_z = sum_b_z * inv_n;                                                  \
        if (a_centroid) {                                                                                          \
            a_centroid[0] = (nk_##output_type##_t)centroid_a_x;                                                    \
            a_centroid[1] = (nk_##output_type##_t)centroid_a_y;                                                    \
            a_centroid[2] = (nk_##output_type##_t)centroid_a_z;                                                    \
        }                                                                                                          \
        if (b_centroid) {                                                                                          \
            b_centroid[0] = (nk_##output_type##_t)centroid_b_x;                                                    \
            b_centroid[1] = (nk_##output_type##_t)centroid_b_y;                                                    \
            b_centroid[2] = (nk_##output_type##_t)centroid_b_z;                                                    \
        }                                                                                                          \
        /* RMSD uses identity rotation and scale=1.0 */                                                            \
        if (rotation) {                                                                                            \
            rotation[0] = 1, rotation[1] = 0, rotation[2] = 0;                                                     \
            rotation[3] = 0, rotation[4] = 1, rotation[5] = 0;                                                     \
            rotation[6] = 0, rotation[7] = 0, rotation[8] = 1;                                                     \
        }                                                                                                          \
        if (scale) *scale = 1.0;                                                                                   \
        nk_##accumulator_type##_t sum_squared = 0;                                                                 \
        for (nk_size_t i = 0; i < n; ++i) {                                                                        \
            load_and_convert(a + i * 3 + 0, &val_a_x), load_and_convert(b + i * 3 + 0, &val_b_x);                  \
            load_and_convert(a + i * 3 + 1, &val_a_y), load_and_convert(b + i * 3 + 1, &val_b_y);                  \
            load_and_convert(a + i * 3 + 2, &val_a_z), load_and_convert(b + i * 3 + 2, &val_b_z);                  \
            nk_##accumulator_type##_t dx = (val_a_x - centroid_a_x) - (val_b_x - centroid_b_x);                    \
            nk_##accumulator_type##_t dy = (val_a_y - centroid_a_y) - (val_b_y - centroid_b_y);                    \
            nk_##accumulator_type##_t dz = (val_a_z - centroid_a_z) - (val_b_z - centroid_b_z);                    \
            sum_squared += dx * dx + dy * dy + dz * dz;                                                            \
        }                                                                                                          \
        *result = compute_sqrt(sum_squared * inv_n);                                                               \
    }

/*  Kabsch algorithm for optimal rigid body superposition.
 *  Finds the rotation matrix R that minimizes RMSD between the two point sets.
 */
#define NK_MAKE_KABSCH(name, input_type, accumulator_type, output_type, svd_type, load_and_convert, compute_sqrt)    \
    NK_PUBLIC void nk_kabsch_##input_type##_##name(nk_##input_type##_t const *a, nk_##input_type##_t const *b,       \
                                                   nk_size_t n, nk_##output_type##_t *a_centroid,                    \
                                                   nk_##output_type##_t *b_centroid, nk_##output_type##_t *rotation, \
                                                   nk_##output_type##_t *scale, nk_##output_type##_t *result) {      \
        /* Step 1: Compute centroids */                                                                              \
        nk_##accumulator_type##_t sum_a_x = 0, sum_a_y = 0, sum_a_z = 0;                                             \
        nk_##accumulator_type##_t sum_b_x = 0, sum_b_y = 0, sum_b_z = 0;                                             \
        nk_##accumulator_type##_t val_a_x, val_a_y, val_a_z, val_b_x, val_b_y, val_b_z;                              \
        for (nk_size_t i = 0; i < n; ++i) {                                                                          \
            load_and_convert(a + i * 3 + 0, &val_a_x), load_and_convert(a + i * 3 + 1, &val_a_y);                    \
            load_and_convert(a + i * 3 + 2, &val_a_z), load_and_convert(b + i * 3 + 0, &val_b_x);                    \
            load_and_convert(b + i * 3 + 1, &val_b_y), load_and_convert(b + i * 3 + 2, &val_b_z);                    \
            sum_a_x += val_a_x, sum_a_y += val_a_y, sum_a_z += val_a_z;                                              \
            sum_b_x += val_b_x, sum_b_y += val_b_y, sum_b_z += val_b_z;                                              \
        }                                                                                                            \
        nk_##accumulator_type##_t inv_n = (nk_##accumulator_type##_t)1.0 / n;                                        \
        nk_##accumulator_type##_t centroid_a_x = sum_a_x * inv_n;                                                    \
        nk_##accumulator_type##_t centroid_a_y = sum_a_y * inv_n;                                                    \
        nk_##accumulator_type##_t centroid_a_z = sum_a_z * inv_n;                                                    \
        nk_##accumulator_type##_t centroid_b_x = sum_b_x * inv_n;                                                    \
        nk_##accumulator_type##_t centroid_b_y = sum_b_y * inv_n;                                                    \
        nk_##accumulator_type##_t centroid_b_z = sum_b_z * inv_n;                                                    \
        if (a_centroid) {                                                                                            \
            a_centroid[0] = (nk_##output_type##_t)centroid_a_x;                                                      \
            a_centroid[1] = (nk_##output_type##_t)centroid_a_y;                                                      \
            a_centroid[2] = (nk_##output_type##_t)centroid_a_z;                                                      \
        }                                                                                                            \
        if (b_centroid) {                                                                                            \
            b_centroid[0] = (nk_##output_type##_t)centroid_b_x;                                                      \
            b_centroid[1] = (nk_##output_type##_t)centroid_b_y;                                                      \
            b_centroid[2] = (nk_##output_type##_t)centroid_b_z;                                                      \
        }                                                                                                            \
        /* Step 2: Build 3x3 covariance matrix H = (A - centroid_A)^T * (B - centroid_B) */                          \
        nk_##accumulator_type##_t h[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};                                                \
        for (nk_size_t i = 0; i < n; ++i) {                                                                          \
            load_and_convert(a + i * 3 + 0, &val_a_x), load_and_convert(b + i * 3 + 0, &val_b_x);                    \
            load_and_convert(a + i * 3 + 1, &val_a_y), load_and_convert(b + i * 3 + 1, &val_b_y);                    \
            load_and_convert(a + i * 3 + 2, &val_a_z), load_and_convert(b + i * 3 + 2, &val_b_z);                    \
            val_a_x -= centroid_a_x, val_a_y -= centroid_a_y, val_a_z -= centroid_a_z;                               \
            val_b_x -= centroid_b_x, val_b_y -= centroid_b_y, val_b_z -= centroid_b_z;                               \
            h[0] += val_a_x * val_b_x, h[1] += val_a_x * val_b_y, h[2] += val_a_x * val_b_z;                         \
            h[3] += val_a_y * val_b_x, h[4] += val_a_y * val_b_y, h[5] += val_a_y * val_b_z;                         \
            h[6] += val_a_z * val_b_x, h[7] += val_a_z * val_b_y, h[8] += val_a_z * val_b_z;                         \
        }                                                                                                            \
        /* Convert to svd_type for SVD */                                                                            \
        nk_##svd_type##_t cross_covariance[9];                                                                       \
        for (int j = 0; j < 9; ++j) cross_covariance[j] = (nk_##svd_type##_t)h[j];                                   \
        /* Step 3: SVD of H = U * S * V^T */                                                                         \
        nk_##svd_type##_t svd_u[9], svd_s[9], svd_v[9];                                                              \
        nk_svd3x3_##svd_type##_(cross_covariance, svd_u, svd_s, svd_v);                                              \
        /* Step 4: R = V * U^T */                                                                                    \
        nk_##svd_type##_t rotation_matrix[9];                                                                        \
        rotation_matrix[0] = svd_v[0] * svd_u[0] + svd_v[1] * svd_u[1] + svd_v[2] * svd_u[2];                        \
        rotation_matrix[1] = svd_v[0] * svd_u[3] + svd_v[1] * svd_u[4] + svd_v[2] * svd_u[5];                        \
        rotation_matrix[2] = svd_v[0] * svd_u[6] + svd_v[1] * svd_u[7] + svd_v[2] * svd_u[8];                        \
        rotation_matrix[3] = svd_v[3] * svd_u[0] + svd_v[4] * svd_u[1] + svd_v[5] * svd_u[2];                        \
        rotation_matrix[4] = svd_v[3] * svd_u[3] + svd_v[4] * svd_u[4] + svd_v[5] * svd_u[5];                        \
        rotation_matrix[5] = svd_v[3] * svd_u[6] + svd_v[4] * svd_u[7] + svd_v[5] * svd_u[8];                        \
        rotation_matrix[6] = svd_v[6] * svd_u[0] + svd_v[7] * svd_u[1] + svd_v[8] * svd_u[2];                        \
        rotation_matrix[7] = svd_v[6] * svd_u[3] + svd_v[7] * svd_u[4] + svd_v[8] * svd_u[5];                        \
        rotation_matrix[8] = svd_v[6] * svd_u[6] + svd_v[7] * svd_u[7] + svd_v[8] * svd_u[8];                        \
        /* Handle reflection: if det(R) < 0, negate third column of V and recompute R */                             \
        nk_##svd_type##_t rotation_det = nk_det3x3_##svd_type##_(rotation_matrix);                                   \
        if (rotation_det < 0) {                                                                                      \
            svd_v[2] = -svd_v[2], svd_v[5] = -svd_v[5], svd_v[8] = -svd_v[8];                                        \
            rotation_matrix[0] = svd_v[0] * svd_u[0] + svd_v[1] * svd_u[1] + svd_v[2] * svd_u[2];                    \
            rotation_matrix[1] = svd_v[0] * svd_u[3] + svd_v[1] * svd_u[4] + svd_v[2] * svd_u[5];                    \
            rotation_matrix[2] = svd_v[0] * svd_u[6] + svd_v[1] * svd_u[7] + svd_v[2] * svd_u[8];                    \
            rotation_matrix[3] = svd_v[3] * svd_u[0] + svd_v[4] * svd_u[1] + svd_v[5] * svd_u[2];                    \
            rotation_matrix[4] = svd_v[3] * svd_u[3] + svd_v[4] * svd_u[4] + svd_v[5] * svd_u[5];                    \
            rotation_matrix[5] = svd_v[3] * svd_u[6] + svd_v[4] * svd_u[7] + svd_v[5] * svd_u[8];                    \
            rotation_matrix[6] = svd_v[6] * svd_u[0] + svd_v[7] * svd_u[1] + svd_v[8] * svd_u[2];                    \
            rotation_matrix[7] = svd_v[6] * svd_u[3] + svd_v[7] * svd_u[4] + svd_v[8] * svd_u[5];                    \
            rotation_matrix[8] = svd_v[6] * svd_u[6] + svd_v[7] * svd_u[7] + svd_v[8] * svd_u[8];                    \
        }                                                                                                            \
        /* Output rotation matrix and scale=1.0 */                                                                   \
        if (rotation) {                                                                                              \
            for (int j = 0; j < 9; ++j) rotation[j] = (nk_##output_type##_t)rotation_matrix[j];                      \
        }                                                                                                            \
        if (scale) *scale = 1.0;                                                                                     \
        /* Step 5: Compute RMSD after rotation */                                                                    \
        nk_##accumulator_type##_t sum_squared = 0;                                                                   \
        for (nk_size_t i = 0; i < n; ++i) {                                                                          \
            nk_##svd_type##_t point_a[3], point_b[3], rotated_point_a[3];                                            \
            load_and_convert(a + i * 3 + 0, &val_a_x), load_and_convert(a + i * 3 + 1, &val_a_y);                    \
            load_and_convert(a + i * 3 + 2, &val_a_z), load_and_convert(b + i * 3 + 0, &val_b_x);                    \
            load_and_convert(b + i * 3 + 1, &val_b_y), load_and_convert(b + i * 3 + 2, &val_b_z);                    \
            point_a[0] = (nk_##svd_type##_t)(val_a_x - centroid_a_x);                                                \
            point_a[1] = (nk_##svd_type##_t)(val_a_y - centroid_a_y);                                                \
            point_a[2] = (nk_##svd_type##_t)(val_a_z - centroid_a_z);                                                \
            point_b[0] = (nk_##svd_type##_t)(val_b_x - centroid_b_x);                                                \
            point_b[1] = (nk_##svd_type##_t)(val_b_y - centroid_b_y);                                                \
            point_b[2] = (nk_##svd_type##_t)(val_b_z - centroid_b_z);                                                \
            rotated_point_a[0] = rotation_matrix[0] * point_a[0] + rotation_matrix[1] * point_a[1] +                 \
                                 rotation_matrix[2] * point_a[2];                                                    \
            rotated_point_a[1] = rotation_matrix[3] * point_a[0] + rotation_matrix[4] * point_a[1] +                 \
                                 rotation_matrix[5] * point_a[2];                                                    \
            rotated_point_a[2] = rotation_matrix[6] * point_a[0] + rotation_matrix[7] * point_a[1] +                 \
                                 rotation_matrix[8] * point_a[2];                                                    \
            nk_##svd_type##_t dx = rotated_point_a[0] - point_b[0];                                                  \
            nk_##svd_type##_t dy = rotated_point_a[1] - point_b[1];                                                  \
            nk_##svd_type##_t dz = rotated_point_a[2] - point_b[2];                                                  \
            sum_squared += dx * dx + dy * dy + dz * dz;                                                              \
        }                                                                                                            \
        *result = compute_sqrt(sum_squared * inv_n);                                                                 \
    }

/*  Umeyama algorithm for optimal similarity transformation (rotation + uniform scale).
 *  Finds the rotation matrix R and scale factor c that minimizes ||c*R*A - B||.
 *  Reference: S. Umeyama, "Least-squares estimation of transformation parameters
 *  between two point patterns", IEEE TPAMI 1991.
 */
#define NK_MAKE_UMEYAMA(name, input_type, accumulator_type, output_type, svd_type, load_and_convert, compute_sqrt)    \
    NK_PUBLIC void nk_umeyama_##input_type##_##name(nk_##input_type##_t const *a, nk_##input_type##_t const *b,       \
                                                    nk_size_t n, nk_##output_type##_t *a_centroid,                    \
                                                    nk_##output_type##_t *b_centroid, nk_##output_type##_t *rotation, \
                                                    nk_##output_type##_t *scale, nk_##output_type##_t *result) {      \
        /* Step 1: Compute centroids */                                                                               \
        nk_##accumulator_type##_t sum_a_x = 0, sum_a_y = 0, sum_a_z = 0;                                              \
        nk_##accumulator_type##_t sum_b_x = 0, sum_b_y = 0, sum_b_z = 0;                                              \
        nk_##accumulator_type##_t val_a_x, val_a_y, val_a_z, val_b_x, val_b_y, val_b_z;                               \
        for (nk_size_t i = 0; i < n; ++i) {                                                                           \
            load_and_convert(a + i * 3 + 0, &val_a_x), load_and_convert(a + i * 3 + 1, &val_a_y);                     \
            load_and_convert(a + i * 3 + 2, &val_a_z), load_and_convert(b + i * 3 + 0, &val_b_x);                     \
            load_and_convert(b + i * 3 + 1, &val_b_y), load_and_convert(b + i * 3 + 2, &val_b_z);                     \
            sum_a_x += val_a_x, sum_a_y += val_a_y, sum_a_z += val_a_z;                                               \
            sum_b_x += val_b_x, sum_b_y += val_b_y, sum_b_z += val_b_z;                                               \
        }                                                                                                             \
        nk_##accumulator_type##_t inv_n = (nk_##accumulator_type##_t)1.0 / n;                                         \
        nk_##accumulator_type##_t centroid_a_x = sum_a_x * inv_n;                                                     \
        nk_##accumulator_type##_t centroid_a_y = sum_a_y * inv_n;                                                     \
        nk_##accumulator_type##_t centroid_a_z = sum_a_z * inv_n;                                                     \
        nk_##accumulator_type##_t centroid_b_x = sum_b_x * inv_n;                                                     \
        nk_##accumulator_type##_t centroid_b_y = sum_b_y * inv_n;                                                     \
        nk_##accumulator_type##_t centroid_b_z = sum_b_z * inv_n;                                                     \
        if (a_centroid) {                                                                                             \
            a_centroid[0] = (nk_##output_type##_t)centroid_a_x;                                                       \
            a_centroid[1] = (nk_##output_type##_t)centroid_a_y;                                                       \
            a_centroid[2] = (nk_##output_type##_t)centroid_a_z;                                                       \
        }                                                                                                             \
        if (b_centroid) {                                                                                             \
            b_centroid[0] = (nk_##output_type##_t)centroid_b_x;                                                       \
            b_centroid[1] = (nk_##output_type##_t)centroid_b_y;                                                       \
            b_centroid[2] = (nk_##output_type##_t)centroid_b_z;                                                       \
        }                                                                                                             \
        /* Step 2: Build covariance matrix H and compute variance of A */                                             \
        nk_##accumulator_type##_t h[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};                                                 \
        nk_##accumulator_type##_t variance_a = 0;                                                                     \
        for (nk_size_t i = 0; i < n; ++i) {                                                                           \
            load_and_convert(a + i * 3 + 0, &val_a_x), load_and_convert(b + i * 3 + 0, &val_b_x);                     \
            load_and_convert(a + i * 3 + 1, &val_a_y), load_and_convert(b + i * 3 + 1, &val_b_y);                     \
            load_and_convert(a + i * 3 + 2, &val_a_z), load_and_convert(b + i * 3 + 2, &val_b_z);                     \
            val_a_x -= centroid_a_x, val_a_y -= centroid_a_y, val_a_z -= centroid_a_z;                                \
            val_b_x -= centroid_b_x, val_b_y -= centroid_b_y, val_b_z -= centroid_b_z;                                \
            variance_a += val_a_x * val_a_x + val_a_y * val_a_y + val_a_z * val_a_z;                                  \
            h[0] += val_a_x * val_b_x, h[1] += val_a_x * val_b_y, h[2] += val_a_x * val_b_z;                          \
            h[3] += val_a_y * val_b_x, h[4] += val_a_y * val_b_y, h[5] += val_a_y * val_b_z;                          \
            h[6] += val_a_z * val_b_x, h[7] += val_a_z * val_b_y, h[8] += val_a_z * val_b_z;                          \
        }                                                                                                             \
        variance_a *= inv_n;                                                                                          \
        /* Convert to svd_type for SVD */                                                                             \
        nk_##svd_type##_t cross_covariance[9];                                                                        \
        for (int j = 0; j < 9; ++j) cross_covariance[j] = (nk_##svd_type##_t)h[j];                                    \
        /* Step 3: SVD of H = U * S * V^T */                                                                          \
        nk_##svd_type##_t svd_u[9], svd_s[9], svd_v[9];                                                               \
        nk_svd3x3_##svd_type##_(cross_covariance, svd_u, svd_s, svd_v);                                               \
        /* Step 4: R = V * U^T */                                                                                     \
        nk_##svd_type##_t rotation_matrix[9];                                                                         \
        rotation_matrix[0] = svd_v[0] * svd_u[0] + svd_v[1] * svd_u[1] + svd_v[2] * svd_u[2];                         \
        rotation_matrix[1] = svd_v[0] * svd_u[3] + svd_v[1] * svd_u[4] + svd_v[2] * svd_u[5];                         \
        rotation_matrix[2] = svd_v[0] * svd_u[6] + svd_v[1] * svd_u[7] + svd_v[2] * svd_u[8];                         \
        rotation_matrix[3] = svd_v[3] * svd_u[0] + svd_v[4] * svd_u[1] + svd_v[5] * svd_u[2];                         \
        rotation_matrix[4] = svd_v[3] * svd_u[3] + svd_v[4] * svd_u[4] + svd_v[5] * svd_u[5];                         \
        rotation_matrix[5] = svd_v[3] * svd_u[6] + svd_v[4] * svd_u[7] + svd_v[5] * svd_u[8];                         \
        rotation_matrix[6] = svd_v[6] * svd_u[0] + svd_v[7] * svd_u[1] + svd_v[8] * svd_u[2];                         \
        rotation_matrix[7] = svd_v[6] * svd_u[3] + svd_v[7] * svd_u[4] + svd_v[8] * svd_u[5];                         \
        rotation_matrix[8] = svd_v[6] * svd_u[6] + svd_v[7] * svd_u[7] + svd_v[8] * svd_u[8];                         \
        /* Handle reflection and compute scale: c = trace(D*S) / variance_a */                                        \
        /* D = diag(1, 1, det(R)), svd_s contains proper positive singular values on diagonal */                      \
        nk_##svd_type##_t rotation_det = nk_det3x3_##svd_type##_(rotation_matrix);                                    \
        nk_##svd_type##_t sign_det = rotation_det < 0 ? (nk_##svd_type##_t) - 1.0 : (nk_##svd_type##_t)1.0;           \
        nk_##svd_type##_t trace_scaled_s = svd_s[0] + svd_s[4] + sign_det * svd_s[8];                                 \
        nk_##accumulator_type##_t scale_factor = (nk_##accumulator_type##_t)trace_scaled_s /                          \
                                                 ((nk_##accumulator_type##_t)n * variance_a);                         \
        if (scale) *scale = scale_factor;                                                                             \
        if (rotation_det < 0) {                                                                                       \
            svd_v[2] = -svd_v[2], svd_v[5] = -svd_v[5], svd_v[8] = -svd_v[8];                                         \
            rotation_matrix[0] = svd_v[0] * svd_u[0] + svd_v[1] * svd_u[1] + svd_v[2] * svd_u[2];                     \
            rotation_matrix[1] = svd_v[0] * svd_u[3] + svd_v[1] * svd_u[4] + svd_v[2] * svd_u[5];                     \
            rotation_matrix[2] = svd_v[0] * svd_u[6] + svd_v[1] * svd_u[7] + svd_v[2] * svd_u[8];                     \
            rotation_matrix[3] = svd_v[3] * svd_u[0] + svd_v[4] * svd_u[1] + svd_v[5] * svd_u[2];                     \
            rotation_matrix[4] = svd_v[3] * svd_u[3] + svd_v[4] * svd_u[4] + svd_v[5] * svd_u[5];                     \
            rotation_matrix[5] = svd_v[3] * svd_u[6] + svd_v[4] * svd_u[7] + svd_v[5] * svd_u[8];                     \
            rotation_matrix[6] = svd_v[6] * svd_u[0] + svd_v[7] * svd_u[1] + svd_v[8] * svd_u[2];                     \
            rotation_matrix[7] = svd_v[6] * svd_u[3] + svd_v[7] * svd_u[4] + svd_v[8] * svd_u[5];                     \
            rotation_matrix[8] = svd_v[6] * svd_u[6] + svd_v[7] * svd_u[7] + svd_v[8] * svd_u[8];                     \
        }                                                                                                             \
        /* Output rotation matrix */                                                                                  \
        if (rotation) {                                                                                               \
            for (int j = 0; j < 9; ++j) rotation[j] = (nk_##output_type##_t)rotation_matrix[j];                       \
        }                                                                                                             \
        /* Step 5: Compute RMSD after similarity transform: ||c*R*a - b|| */                                          \
        nk_##accumulator_type##_t sum_squared = 0;                                                                    \
        for (nk_size_t i = 0; i < n; ++i) {                                                                           \
            nk_##svd_type##_t point_a[3], point_b[3], rotated_point_a[3];                                             \
            load_and_convert(a + i * 3 + 0, &val_a_x), load_and_convert(a + i * 3 + 1, &val_a_y);                     \
            load_and_convert(a + i * 3 + 2, &val_a_z), load_and_convert(b + i * 3 + 0, &val_b_x);                     \
            load_and_convert(b + i * 3 + 1, &val_b_y), load_and_convert(b + i * 3 + 2, &val_b_z);                     \
            point_a[0] = (nk_##svd_type##_t)(val_a_x - centroid_a_x);                                                 \
            point_a[1] = (nk_##svd_type##_t)(val_a_y - centroid_a_y);                                                 \
            point_a[2] = (nk_##svd_type##_t)(val_a_z - centroid_a_z);                                                 \
            point_b[0] = (nk_##svd_type##_t)(val_b_x - centroid_b_x);                                                 \
            point_b[1] = (nk_##svd_type##_t)(val_b_y - centroid_b_y);                                                 \
            point_b[2] = (nk_##svd_type##_t)(val_b_z - centroid_b_z);                                                 \
            rotated_point_a[0] = (nk_##svd_type##_t)scale_factor *                                                    \
                                 (rotation_matrix[0] * point_a[0] + rotation_matrix[1] * point_a[1] +                 \
                                  rotation_matrix[2] * point_a[2]);                                                   \
            rotated_point_a[1] = (nk_##svd_type##_t)scale_factor *                                                    \
                                 (rotation_matrix[3] * point_a[0] + rotation_matrix[4] * point_a[1] +                 \
                                  rotation_matrix[5] * point_a[2]);                                                   \
            rotated_point_a[2] = (nk_##svd_type##_t)scale_factor *                                                    \
                                 (rotation_matrix[6] * point_a[0] + rotation_matrix[7] * point_a[1] +                 \
                                  rotation_matrix[8] * point_a[2]);                                                   \
            nk_##svd_type##_t dx = rotated_point_a[0] - point_b[0];                                                   \
            nk_##svd_type##_t dy = rotated_point_a[1] - point_b[1];                                                   \
            nk_##svd_type##_t dz = rotated_point_a[2] - point_b[2];                                                   \
            sum_squared += dx * dx + dy * dy + dz * dz;                                                               \
        }                                                                                                             \
        *result = compute_sqrt(sum_squared * inv_n);                                                                  \
    }

NK_MAKE_RMSD(serial, f64, f64, f64, nk_assign_from_to_, nk_f64_sqrt_serial)         // nk_rmsd_f64_serial
NK_MAKE_KABSCH(serial, f64, f64, f64, f64, nk_assign_from_to_, nk_f64_sqrt_serial)  // nk_kabsch_f64_serial
NK_MAKE_UMEYAMA(serial, f64, f64, f64, f64, nk_assign_from_to_, nk_f64_sqrt_serial) // nk_umeyama_f64_serial

NK_MAKE_RMSD(serial, f32, f32, f32, nk_assign_from_to_, nk_f32_sqrt_serial)         // nk_rmsd_f32_serial
NK_MAKE_KABSCH(serial, f32, f32, f32, f32, nk_assign_from_to_, nk_f32_sqrt_serial)  // nk_kabsch_f32_serial
NK_MAKE_UMEYAMA(serial, f32, f32, f32, f32, nk_assign_from_to_, nk_f32_sqrt_serial) // nk_umeyama_f32_serial

NK_MAKE_RMSD(serial, f16, f32, f32, nk_f16_to_f32_serial, nk_f32_sqrt_serial)         // nk_rmsd_f16_serial
NK_MAKE_KABSCH(serial, f16, f32, f32, f32, nk_f16_to_f32_serial, nk_f32_sqrt_serial)  // nk_kabsch_f16_serial
NK_MAKE_UMEYAMA(serial, f16, f32, f32, f32, nk_f16_to_f32_serial, nk_f32_sqrt_serial) // nk_umeyama_f16_serial

NK_MAKE_RMSD(serial, bf16, f32, f32, nk_bf16_to_f32_serial, nk_f32_sqrt_serial)         // nk_rmsd_bf16_serial
NK_MAKE_KABSCH(serial, bf16, f32, f32, f32, nk_bf16_to_f32_serial, nk_f32_sqrt_serial)  // nk_kabsch_bf16_serial
NK_MAKE_UMEYAMA(serial, bf16, f32, f32, f32, nk_bf16_to_f32_serial, nk_f32_sqrt_serial) // nk_umeyama_bf16_serial

NK_MAKE_RMSD(accurate, f32, f64, f64, nk_assign_from_to_, nk_f64_sqrt_serial)        // nk_rmsd_f32_accurate
NK_MAKE_KABSCH(accurate, f32, f64, f64, f64, nk_assign_from_to_, nk_f64_sqrt_serial) // nk_kabsch_f32_accurate
NK_MAKE_UMEYAMA(accurate, f32, f64, f64, f64, nk_assign_from_to_,
                nk_f64_sqrt_serial) // nk_umeyama_f32_accurate

NK_MAKE_RMSD(accurate, f16, f64, f64, nk_f16_to_f64_, nk_f64_sqrt_serial)         // nk_rmsd_f16_accurate
NK_MAKE_KABSCH(accurate, f16, f64, f64, f64, nk_f16_to_f64_, nk_f64_sqrt_serial)  // nk_kabsch_f16_accurate
NK_MAKE_UMEYAMA(accurate, f16, f64, f64, f64, nk_f16_to_f64_, nk_f64_sqrt_serial) // nk_umeyama_f16_accurate

NK_MAKE_RMSD(accurate, bf16, f64, f64, nk_bf16_to_f64_, nk_f64_sqrt_serial) // nk_rmsd_bf16_accurate
NK_MAKE_KABSCH(accurate, bf16, f64, f64, f64, nk_bf16_to_f64_,
               nk_f64_sqrt_serial) // nk_kabsch_bf16_accurate
NK_MAKE_UMEYAMA(accurate, bf16, f64, f64, f64, nk_bf16_to_f64_,
                nk_f64_sqrt_serial) // nk_umeyama_bf16_accurate

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_MESH_SERIAL_H
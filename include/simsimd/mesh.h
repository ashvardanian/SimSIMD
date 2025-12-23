/**
 *  @file       mesh.h
 *  @brief      SIMD-accelerated similarity measures for meshes and rigid 3D bodies.
 *  @author     Ash Vardanian
 *  @date       June 19, 2024
 *
 *  Contains:
 *  - Root Mean Square Deviation (RMSD) for rigid body superposition
 *  - Kabsch algorithm for optimal rigid body superposition
 *
 *  For datatypes:
 *  - 64-bit IEEE-754 floating point
 *  - 32-bit IEEE-754 floating point
 *  - 16-bit IEEE-754 floating point
 *  - 16-bit brain-floating point
 *
 *  For hardware architectures:
 *  - x86 (AVX2, AVX512)
 *  - Arm (NEON, SVE)
 *
 *  The Kabsch algorithm finds the optimal rotation matrix that minimizes RMSD between two point sets.
 *  It uses singular value decomposition (SVD) of a 3x3 covariance matrix.
 *
 *  The 3x3 SVD implementation is based on the McAdams et al. paper:
 *  "Computing the Singular Value Decomposition of 3x3 matrices with minimal branching
 *  and elementary floating point operations", University of Wisconsin - Madison TR1690, 2011.
 *
 *  x86 intrinsics: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/
 *  Arm intrinsics: https://developer.arm.com/architectures/instruction-sets/intrinsics/
 */
#ifndef SIMSIMD_MESH_H
#define SIMSIMD_MESH_H

#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

// clang-format off

/*  Serial backends for all numeric types.
 *  By default they use 32-bit arithmetic, unless the arguments themselves contain 64-bit floats.
 *  For double-precision computation check out the "*_accurate" variants of those "*_serial" functions.
 */
SIMSIMD_PUBLIC void simsimd_rmsd_f64_serial(simsimd_f64_t const* a, simsimd_f64_t const* b, simsimd_size_t n, simsimd_f64_t* a_centroid, simsimd_f64_t* b_centroid, simsimd_distance_t* result);
SIMSIMD_PUBLIC void simsimd_kabsch_f64_serial(simsimd_f64_t const* a, simsimd_f64_t const* b, simsimd_size_t n, simsimd_f64_t* a_centroid, simsimd_f64_t* b_centroid, simsimd_distance_t* result);

SIMSIMD_PUBLIC void simsimd_rmsd_f32_serial(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t n, simsimd_f32_t* a_centroid, simsimd_f32_t* b_centroid, simsimd_distance_t* result);
SIMSIMD_PUBLIC void simsimd_kabsch_f32_serial(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t n, simsimd_f32_t* a_centroid, simsimd_f32_t* b_centroid, simsimd_distance_t* result);

SIMSIMD_PUBLIC void simsimd_rmsd_f16_serial(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_size_t n, simsimd_f16_t* a_centroid, simsimd_f16_t* b_centroid, simsimd_distance_t* result);
SIMSIMD_PUBLIC void simsimd_kabsch_f16_serial(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_size_t n, simsimd_f16_t* a_centroid, simsimd_f16_t* b_centroid, simsimd_distance_t* result);

SIMSIMD_PUBLIC void simsimd_rmsd_bf16_serial(simsimd_bf16_t const* a, simsimd_bf16_t const* b, simsimd_size_t n, simsimd_bf16_t* a_centroid, simsimd_bf16_t* b_centroid, simsimd_distance_t* result);
SIMSIMD_PUBLIC void simsimd_kabsch_bf16_serial(simsimd_bf16_t const* a, simsimd_bf16_t const* b, simsimd_size_t n, simsimd_bf16_t* a_centroid, simsimd_bf16_t* b_centroid, simsimd_distance_t* result);

/*  Double-precision serial backends for all numeric types.
 *  For single-precision computation check out the "*_serial" counterparts of those "*_accurate" functions.
 */
SIMSIMD_PUBLIC void simsimd_rmsd_f32_accurate(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t n, simsimd_f32_t* a_centroid, simsimd_f32_t* b_centroid, simsimd_distance_t* result);
SIMSIMD_PUBLIC void simsimd_kabsch_f32_accurate(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t n, simsimd_f32_t* a_centroid, simsimd_f32_t* b_centroid, simsimd_distance_t* result);

SIMSIMD_PUBLIC void simsimd_rmsd_f16_accurate(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_size_t n, simsimd_f16_t* a_centroid, simsimd_f16_t* b_centroid, simsimd_distance_t* result);
SIMSIMD_PUBLIC void simsimd_kabsch_f16_accurate(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_size_t n, simsimd_f16_t* a_centroid, simsimd_f16_t* b_centroid, simsimd_distance_t* result);

SIMSIMD_PUBLIC void simsimd_rmsd_bf16_accurate(simsimd_bf16_t const* a, simsimd_bf16_t const* b, simsimd_size_t n, simsimd_bf16_t* a_centroid, simsimd_bf16_t* b_centroid, simsimd_distance_t* result);
SIMSIMD_PUBLIC void simsimd_kabsch_bf16_accurate(simsimd_bf16_t const* a, simsimd_bf16_t const* b, simsimd_size_t n, simsimd_bf16_t* a_centroid, simsimd_bf16_t* b_centroid, simsimd_distance_t* result);

/*  SIMD-powered backends for AVX512 CPUs of Skylake generation and newer.
 */
SIMSIMD_PUBLIC void simsimd_rmsd_f32_skylake(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t n, simsimd_f32_t* a_centroid, simsimd_f32_t* b_centroid, simsimd_distance_t* result);
SIMSIMD_PUBLIC void simsimd_kabsch_f32_skylake(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t n, simsimd_f32_t* a_centroid, simsimd_f32_t* b_centroid, simsimd_distance_t* result);
SIMSIMD_PUBLIC void simsimd_rmsd_f64_skylake(simsimd_f64_t const* a, simsimd_f64_t const* b, simsimd_size_t n, simsimd_f64_t* a_centroid, simsimd_f64_t* b_centroid, simsimd_distance_t* result);
SIMSIMD_PUBLIC void simsimd_kabsch_f64_skylake(simsimd_f64_t const* a, simsimd_f64_t const* b, simsimd_size_t n, simsimd_f64_t* a_centroid, simsimd_f64_t* b_centroid, simsimd_distance_t* result);

// clang-format on

/*  Constants for the McAdams 3x3 SVD algorithm.
 *  gamma = (sqrt(8) + 3)^2 / 4 = 5.828427124
 *  cstar = cos(pi/8), sstar = sin(pi/8)
 */
#define SIMSIMD_SVD_GAMMA_F32 5.828427124f
#define SIMSIMD_SVD_CSTAR_F32 0.923879532f
#define SIMSIMD_SVD_SSTAR_F32 0.3826834323f
#define SIMSIMD_SVD_EPSILON_F32 1e-6f

#define SIMSIMD_SVD_GAMMA_F64 5.828427124
#define SIMSIMD_SVD_CSTAR_F64 0.923879532
#define SIMSIMD_SVD_SSTAR_F64 0.3826834323
#define SIMSIMD_SVD_EPSILON_F64 1e-12

/*  Internal helper: Fast reciprocal square root using the "magic number" method.
 *  Two Newton-Raphson iterations for high accuracy (~1e-7 relative error).
 */
SIMSIMD_INTERNAL simsimd_f32_t _simsimd_rsqrt_f32(simsimd_f32_t x) {
    simsimd_f32_t xhalf = 0.5f * x;
    union {
        simsimd_f32_t f;
        simsimd_i32_t i;
    } u;
    u.f = x;
    u.i = 0x5f375a82 - (u.i >> 1);
    u.f = u.f * (1.5f - xhalf * u.f * u.f); // First iteration: ~1e-3 relative error
    u.f = u.f * (1.5f - xhalf * u.f * u.f); // Second iteration: ~1e-7 relative error
    return u.f;
}

/*  Internal helper: Conditional swap without branching.
 */
SIMSIMD_INTERNAL void _simsimd_cond_swap_f32(int c, simsimd_f32_t *x, simsimd_f32_t *y) {
    simsimd_f32_t z = *x;
    *x = c ? *y : *x;
    *y = c ? z : *y;
}

/*  Internal helper: Conditional negating swap without branching.
 */
SIMSIMD_INTERNAL void _simsimd_cond_neg_swap_f32(int c, simsimd_f32_t *x, simsimd_f32_t *y) {
    simsimd_f32_t z = -*x;
    *x = c ? *y : *x;
    *y = c ? z : *y;
}

/*  Internal helper: Approximate Givens quaternion for Jacobi rotation.
 */
SIMSIMD_INTERNAL void _simsimd_approx_givens_quat_f32(       //
    simsimd_f32_t a11, simsimd_f32_t a12, simsimd_f32_t a22, //
    simsimd_f32_t *cos_half, simsimd_f32_t *sin_half) {      //
    *cos_half = 2 * (a11 - a22), *sin_half = a12;
    int use_givens = SIMSIMD_SVD_GAMMA_F32 * (*sin_half) * (*sin_half) < (*cos_half) * (*cos_half);
    simsimd_f32_t w = _simsimd_rsqrt_f32((*cos_half) * (*cos_half) + (*sin_half) * (*sin_half));
    *cos_half = use_givens ? w * (*cos_half) : SIMSIMD_SVD_CSTAR_F32;
    *sin_half = use_givens ? w * (*sin_half) : SIMSIMD_SVD_SSTAR_F32;
}

/*  Internal helper: Jacobi conjugation step for symmetric eigenvalue problem.
 */
SIMSIMD_INTERNAL void _simsimd_jacobi_conjugation_f32(          //
    int x, int y, int z,                                        //
    simsimd_f32_t *s11, simsimd_f32_t *s21, simsimd_f32_t *s22, //
    simsimd_f32_t *s31, simsimd_f32_t *s32, simsimd_f32_t *s33, //
    simsimd_f32_t *q_v) {                                       //
    simsimd_f32_t cos_half, sin_half;
    _simsimd_approx_givens_quat_f32(*s11, *s21, *s22, &cos_half, &sin_half);

    simsimd_f32_t scale = cos_half * cos_half + sin_half * sin_half;
    simsimd_f32_t cos_theta = (cos_half * cos_half - sin_half * sin_half) / scale;
    simsimd_f32_t sin_theta = (2 * sin_half * cos_half) / scale;

    simsimd_f32_t s11_tmp = *s11, s21_tmp = *s21, s22_tmp = *s22;
    simsimd_f32_t s31_tmp = *s31, s32_tmp = *s32, s33_tmp = *s33;

    *s11 = cos_theta * (cos_theta * s11_tmp + sin_theta * s21_tmp) +
           sin_theta * (cos_theta * s21_tmp + sin_theta * s22_tmp);
    *s21 = cos_theta * (-sin_theta * s11_tmp + cos_theta * s21_tmp) +
           sin_theta * (-sin_theta * s21_tmp + cos_theta * s22_tmp);
    *s22 = -sin_theta * (-sin_theta * s11_tmp + cos_theta * s21_tmp) +
           cos_theta * (-sin_theta * s21_tmp + cos_theta * s22_tmp);
    *s31 = cos_theta * s31_tmp + sin_theta * s32_tmp;
    *s32 = -sin_theta * s31_tmp + cos_theta * s32_tmp;
    *s33 = s33_tmp;

    // Update quaternion accumulator
    simsimd_f32_t quat_tmp[3];
    quat_tmp[0] = q_v[0] * sin_half, quat_tmp[1] = q_v[1] * sin_half, quat_tmp[2] = q_v[2] * sin_half;
    sin_half *= q_v[3];
    q_v[0] *= cos_half, q_v[1] *= cos_half, q_v[2] *= cos_half, q_v[3] *= cos_half;
    q_v[z] += sin_half, q_v[3] -= quat_tmp[z], q_v[x] += quat_tmp[y], q_v[y] -= quat_tmp[x];

    // Cyclic permutation of matrix elements
    s11_tmp = *s22, s21_tmp = *s32, s22_tmp = *s33, s31_tmp = *s21, s32_tmp = *s31, s33_tmp = *s11;
    *s11 = s11_tmp, *s21 = s21_tmp, *s22 = s22_tmp, *s31 = s31_tmp, *s32 = s32_tmp, *s33 = s33_tmp;
}

/*  Internal helper: Convert quaternion to 3x3 rotation matrix.
 */
SIMSIMD_INTERNAL void _simsimd_quat_to_mat3_f32(simsimd_f32_t const *q_v, simsimd_f32_t *m) {
    simsimd_f32_t w = q_v[3], x = q_v[0], y = q_v[1], z = q_v[2];
    simsimd_f32_t q_xx = x * x, q_yy = y * y, q_zz = z * z;
    simsimd_f32_t q_xz = x * z, q_xy = x * y, q_yz = y * z;
    simsimd_f32_t q_wx = w * x, q_wy = w * y, q_wz = w * z;
    m[0] = 1 - 2 * (q_yy + q_zz), m[1] = 2 * (q_xy - q_wz), m[2] = 2 * (q_xz + q_wy);
    m[3] = 2 * (q_xy + q_wz), m[4] = 1 - 2 * (q_xx + q_zz), m[5] = 2 * (q_yz - q_wx);
    m[6] = 2 * (q_xz - q_wy), m[7] = 2 * (q_yz + q_wx), m[8] = 1 - 2 * (q_xx + q_yy);
}

/*  Internal helper: Jacobi eigenanalysis for symmetric 3x3 matrix.
 *  4 iterations of cyclic Jacobi rotations.
 */
SIMSIMD_INTERNAL void _simsimd_jacobi_eigenanalysis_f32(        //
    simsimd_f32_t *s11, simsimd_f32_t *s21, simsimd_f32_t *s22, //
    simsimd_f32_t *s31, simsimd_f32_t *s32, simsimd_f32_t *s33, //
    simsimd_f32_t *q_v) {                                       //
    q_v[0] = 0, q_v[1] = 0, q_v[2] = 0, q_v[3] = 1;
    for (int i = 0; i < 4; i++) {
        _simsimd_jacobi_conjugation_f32(0, 1, 2, s11, s21, s22, s31, s32, s33, q_v);
        _simsimd_jacobi_conjugation_f32(1, 2, 0, s11, s21, s22, s31, s32, s33, q_v);
        _simsimd_jacobi_conjugation_f32(2, 0, 1, s11, s21, s22, s31, s32, s33, q_v);
    }
    // Normalize quaternion to ensure orthogonal rotation matrix
    simsimd_f32_t norm = _simsimd_rsqrt_f32(q_v[0] * q_v[0] + q_v[1] * q_v[1] + q_v[2] * q_v[2] + q_v[3] * q_v[3]);
    q_v[0] *= norm, q_v[1] *= norm, q_v[2] *= norm, q_v[3] *= norm;
}

/*  Internal helper: QR Givens quaternion for rotation.
 */
SIMSIMD_INTERNAL void _simsimd_qr_givens_quat_f32(      //
    simsimd_f32_t a1, simsimd_f32_t a2,                 //
    simsimd_f32_t *cos_half, simsimd_f32_t *sin_half) { //
    simsimd_f32_t a1_sq_plus_a2_sq = a1 * a1 + a2 * a2;
    simsimd_f32_t rho = a1_sq_plus_a2_sq * _simsimd_rsqrt_f32(a1_sq_plus_a2_sq); // sqrt(a1^2 + a2^2)
    rho = a1_sq_plus_a2_sq > SIMSIMD_SVD_EPSILON_F32 ? rho : 0;

    *sin_half = rho > SIMSIMD_SVD_EPSILON_F32 ? a2 : 0;
    simsimd_f32_t abs_a1 = a1 < 0 ? -a1 : a1;
    simsimd_f32_t max_rho = rho > SIMSIMD_SVD_EPSILON_F32 ? rho : SIMSIMD_SVD_EPSILON_F32;
    *cos_half = abs_a1 + max_rho;

    int should_swap = a1 < 0;
    _simsimd_cond_swap_f32(should_swap, sin_half, cos_half);

    simsimd_f32_t w = _simsimd_rsqrt_f32((*cos_half) * (*cos_half) + (*sin_half) * (*sin_half));
    *cos_half *= w, *sin_half *= w;
}

/*  Internal helper: Sort singular values and corresponding V columns.
 */
SIMSIMD_INTERNAL void _simsimd_sort_singular_values_f32(simsimd_f32_t *b, simsimd_f32_t *v) {
    simsimd_f32_t rho1 = b[0] * b[0] + b[3] * b[3] + b[6] * b[6];
    simsimd_f32_t rho2 = b[1] * b[1] + b[4] * b[4] + b[7] * b[7];
    simsimd_f32_t rho3 = b[2] * b[2] + b[5] * b[5] + b[8] * b[8];
    int should_swap;
    // Sort columns by descending singular value magnitude (bubble sort with 3 comparisons)
    should_swap = rho1 < rho2;
    _simsimd_cond_neg_swap_f32(should_swap, &b[0], &b[1]), _simsimd_cond_neg_swap_f32(should_swap, &v[0], &v[1]);
    _simsimd_cond_neg_swap_f32(should_swap, &b[3], &b[4]), _simsimd_cond_neg_swap_f32(should_swap, &v[3], &v[4]);
    _simsimd_cond_neg_swap_f32(should_swap, &b[6], &b[7]), _simsimd_cond_neg_swap_f32(should_swap, &v[6], &v[7]);
    _simsimd_cond_swap_f32(should_swap, &rho1, &rho2);

    should_swap = rho1 < rho3;
    _simsimd_cond_neg_swap_f32(should_swap, &b[0], &b[2]), _simsimd_cond_neg_swap_f32(should_swap, &v[0], &v[2]);
    _simsimd_cond_neg_swap_f32(should_swap, &b[3], &b[5]), _simsimd_cond_neg_swap_f32(should_swap, &v[3], &v[5]);
    _simsimd_cond_neg_swap_f32(should_swap, &b[6], &b[8]), _simsimd_cond_neg_swap_f32(should_swap, &v[6], &v[8]);
    _simsimd_cond_swap_f32(should_swap, &rho1, &rho3);

    should_swap = rho2 < rho3;
    _simsimd_cond_neg_swap_f32(should_swap, &b[1], &b[2]), _simsimd_cond_neg_swap_f32(should_swap, &v[1], &v[2]);
    _simsimd_cond_neg_swap_f32(should_swap, &b[4], &b[5]), _simsimd_cond_neg_swap_f32(should_swap, &v[4], &v[5]);
    _simsimd_cond_neg_swap_f32(should_swap, &b[7], &b[8]), _simsimd_cond_neg_swap_f32(should_swap, &v[7], &v[8]);
}

/*  Internal helper: QR decomposition of 3x3 matrix using Givens rotations.
 *  Outputs Q (orthogonal) and R (upper triangular).
 */
SIMSIMD_INTERNAL void _simsimd_qr_decomposition_f32(simsimd_f32_t const *b, simsimd_f32_t *q, simsimd_f32_t *r) {
    simsimd_f32_t cos_half_1, sin_half_1, cos_half_2, sin_half_2, cos_half_3, sin_half_3;
    simsimd_f32_t cos_theta, sin_theta;
    simsimd_f32_t r_tmp[9], b_tmp[9];

    // First Givens rotation (zero b[3])
    _simsimd_qr_givens_quat_f32(b[0], b[3], &cos_half_1, &sin_half_1);
    cos_theta = 1 - 2 * sin_half_1 * sin_half_1, sin_theta = 2 * cos_half_1 * sin_half_1;
    r_tmp[0] = cos_theta * b[0] + sin_theta * b[3], r_tmp[1] = cos_theta * b[1] + sin_theta * b[4],
    r_tmp[2] = cos_theta * b[2] + sin_theta * b[5];
    r_tmp[3] = -sin_theta * b[0] + cos_theta * b[3], r_tmp[4] = -sin_theta * b[1] + cos_theta * b[4],
    r_tmp[5] = -sin_theta * b[2] + cos_theta * b[5];
    r_tmp[6] = b[6], r_tmp[7] = b[7], r_tmp[8] = b[8];

    // Second Givens rotation (zero r_tmp[6])
    _simsimd_qr_givens_quat_f32(r_tmp[0], r_tmp[6], &cos_half_2, &sin_half_2);
    cos_theta = 1 - 2 * sin_half_2 * sin_half_2, sin_theta = 2 * cos_half_2 * sin_half_2;
    b_tmp[0] = cos_theta * r_tmp[0] + sin_theta * r_tmp[6], b_tmp[1] = cos_theta * r_tmp[1] + sin_theta * r_tmp[7],
    b_tmp[2] = cos_theta * r_tmp[2] + sin_theta * r_tmp[8];
    b_tmp[3] = r_tmp[3], b_tmp[4] = r_tmp[4], b_tmp[5] = r_tmp[5];
    b_tmp[6] = -sin_theta * r_tmp[0] + cos_theta * r_tmp[6], b_tmp[7] = -sin_theta * r_tmp[1] + cos_theta * r_tmp[7],
    b_tmp[8] = -sin_theta * r_tmp[2] + cos_theta * r_tmp[8];

    // Third Givens rotation (zero b_tmp[7])
    _simsimd_qr_givens_quat_f32(b_tmp[4], b_tmp[7], &cos_half_3, &sin_half_3);
    cos_theta = 1 - 2 * sin_half_3 * sin_half_3, sin_theta = 2 * cos_half_3 * sin_half_3;
    r[0] = b_tmp[0], r[1] = b_tmp[1], r[2] = b_tmp[2];
    r[3] = cos_theta * b_tmp[3] + sin_theta * b_tmp[6], r[4] = cos_theta * b_tmp[4] + sin_theta * b_tmp[7],
    r[5] = cos_theta * b_tmp[5] + sin_theta * b_tmp[8];
    r[6] = -sin_theta * b_tmp[3] + cos_theta * b_tmp[6], r[7] = -sin_theta * b_tmp[4] + cos_theta * b_tmp[7],
    r[8] = -sin_theta * b_tmp[5] + cos_theta * b_tmp[8];

    // Construct Q = Q1 * Q2 * Q3 (using closed-form expressions for efficiency)
    simsimd_f32_t sin_half_1_sq = sin_half_1 * sin_half_1, sin_half_2_sq = sin_half_2 * sin_half_2,
                  sin_half_3_sq = sin_half_3 * sin_half_3;
    q[0] = (-1 + 2 * sin_half_1_sq) * (-1 + 2 * sin_half_2_sq);
    q[1] = 4 * cos_half_2 * cos_half_3 * (-1 + 2 * sin_half_1_sq) * sin_half_2 * sin_half_3 +
           2 * cos_half_1 * sin_half_1 * (-1 + 2 * sin_half_3_sq);
    q[2] = 4 * cos_half_1 * cos_half_3 * sin_half_1 * sin_half_3 -
           2 * cos_half_2 * (-1 + 2 * sin_half_1_sq) * sin_half_2 * (-1 + 2 * sin_half_3_sq);
    q[3] = 2 * cos_half_1 * sin_half_1 * (1 - 2 * sin_half_2_sq);
    q[4] = -8 * cos_half_1 * cos_half_2 * cos_half_3 * sin_half_1 * sin_half_2 * sin_half_3 +
           (-1 + 2 * sin_half_1_sq) * (-1 + 2 * sin_half_3_sq);
    q[5] = -2 * cos_half_3 * sin_half_3 +
           4 * sin_half_1 *
               (cos_half_3 * sin_half_1 * sin_half_3 + cos_half_1 * cos_half_2 * sin_half_2 * (-1 + 2 * sin_half_3_sq));
    q[6] = 2 * cos_half_2 * sin_half_2;
    q[7] = 2 * cos_half_3 * (1 - 2 * sin_half_2_sq) * sin_half_3;
    q[8] = (-1 + 2 * sin_half_2_sq) * (-1 + 2 * sin_half_3_sq);
}

/*  Internal helper: Compute SVD of 3x3 matrix A = U * S * V^T
 *  Using the McAdams algorithm with fixed 4 Jacobi iterations.
 */
SIMSIMD_INTERNAL void _simsimd_svd3x3_f32(simsimd_f32_t const *a, simsimd_f32_t *u, simsimd_f32_t *s,
                                          simsimd_f32_t *v) {
    // Compute A^T * A (symmetric)
    simsimd_f32_t ata[9];
    ata[0] = a[0] * a[0] + a[3] * a[3] + a[6] * a[6];
    ata[1] = a[0] * a[1] + a[3] * a[4] + a[6] * a[7];
    ata[2] = a[0] * a[2] + a[3] * a[5] + a[6] * a[8];
    ata[3] = ata[1];
    ata[4] = a[1] * a[1] + a[4] * a[4] + a[7] * a[7];
    ata[5] = a[1] * a[2] + a[4] * a[5] + a[7] * a[8];
    ata[6] = ata[2];
    ata[7] = ata[5];
    ata[8] = a[2] * a[2] + a[5] * a[5] + a[8] * a[8];

    // Jacobi eigenanalysis of A^T * A
    simsimd_f32_t q_v[4];
    _simsimd_jacobi_eigenanalysis_f32(&ata[0], &ata[1], &ata[4], &ata[2], &ata[5], &ata[8], q_v);
    _simsimd_quat_to_mat3_f32(q_v, v);

    // B = A * V
    simsimd_f32_t b[9];
    b[0] = a[0] * v[0] + a[1] * v[3] + a[2] * v[6];
    b[1] = a[0] * v[1] + a[1] * v[4] + a[2] * v[7];
    b[2] = a[0] * v[2] + a[1] * v[5] + a[2] * v[8];
    b[3] = a[3] * v[0] + a[4] * v[3] + a[5] * v[6];
    b[4] = a[3] * v[1] + a[4] * v[4] + a[5] * v[7];
    b[5] = a[3] * v[2] + a[4] * v[5] + a[5] * v[8];
    b[6] = a[6] * v[0] + a[7] * v[3] + a[8] * v[6];
    b[7] = a[6] * v[1] + a[7] * v[4] + a[8] * v[7];
    b[8] = a[6] * v[2] + a[7] * v[5] + a[8] * v[8];

    // Sort singular values and update V
    _simsimd_sort_singular_values_f32(b, v);

    // QR decomposition: B = U * S
    _simsimd_qr_decomposition_f32(b, u, s);
}

/*  Internal helper: Compute determinant of 3x3 matrix.
 */
SIMSIMD_INTERNAL simsimd_f32_t _simsimd_det3x3_f32(simsimd_f32_t const *m) {
    return m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) + m[2] * (m[3] * m[7] - m[4] * m[6]);
}

/*  RMSD (Root Mean Square Deviation) without optimal superposition.
 *  Simply computes the RMS of distances between corresponding points.
 */
#define SIMSIMD_MAKE_RMSD(name, input_type, accumulator_type, load_and_convert)                                   \
    SIMSIMD_PUBLIC void simsimd_rmsd_##input_type##_##name(                                                       \
        simsimd_##input_type##_t const *a, simsimd_##input_type##_t const *b, simsimd_size_t n,                   \
        simsimd_##input_type##_t *a_centroid, simsimd_##input_type##_t *b_centroid, simsimd_distance_t *result) { \
        simsimd_##accumulator_type##_t ax = 0, ay = 0, az = 0;                                                    \
        simsimd_##accumulator_type##_t bx = 0, by = 0, bz = 0;                                                    \
        for (simsimd_size_t i = 0; i < n; ++i) {                                                                  \
            ax += load_and_convert(a + i * 3 + 0);                                                                \
            ay += load_and_convert(a + i * 3 + 1);                                                                \
            az += load_and_convert(a + i * 3 + 2);                                                                \
            bx += load_and_convert(b + i * 3 + 0);                                                                \
            by += load_and_convert(b + i * 3 + 1);                                                                \
            bz += load_and_convert(b + i * 3 + 2);                                                                \
        }                                                                                                         \
        simsimd_##accumulator_type##_t inv_n = (simsimd_##accumulator_type##_t)1.0 / n;                           \
        ax *= inv_n;                                                                                              \
        ay *= inv_n;                                                                                              \
        az *= inv_n;                                                                                              \
        bx *= inv_n;                                                                                              \
        by *= inv_n;                                                                                              \
        bz *= inv_n;                                                                                              \
        if (a_centroid) {                                                                                         \
            a_centroid[0] = (simsimd_##input_type##_t)ax;                                                         \
            a_centroid[1] = (simsimd_##input_type##_t)ay;                                                         \
            a_centroid[2] = (simsimd_##input_type##_t)az;                                                         \
        }                                                                                                         \
        if (b_centroid) {                                                                                         \
            b_centroid[0] = (simsimd_##input_type##_t)bx;                                                         \
            b_centroid[1] = (simsimd_##input_type##_t)by;                                                         \
            b_centroid[2] = (simsimd_##input_type##_t)bz;                                                         \
        }                                                                                                         \
        simsimd_##accumulator_type##_t sum_sq = 0;                                                                \
        for (simsimd_size_t i = 0; i < n; ++i) {                                                                  \
            simsimd_##accumulator_type##_t dx =                                                                   \
                (load_and_convert(a + i * 3 + 0) - ax) - (load_and_convert(b + i * 3 + 0) - bx);                  \
            simsimd_##accumulator_type##_t dy =                                                                   \
                (load_and_convert(a + i * 3 + 1) - ay) - (load_and_convert(b + i * 3 + 1) - by);                  \
            simsimd_##accumulator_type##_t dz =                                                                   \
                (load_and_convert(a + i * 3 + 2) - az) - (load_and_convert(b + i * 3 + 2) - bz);                  \
            sum_sq += dx * dx + dy * dy + dz * dz;                                                                \
        }                                                                                                         \
        *result = SIMSIMD_SQRT(sum_sq * inv_n);                                                                   \
    }

/*  Kabsch algorithm for optimal rigid body superposition.
 *  Finds the rotation matrix R that minimizes RMSD between the two point sets.
 */
#define SIMSIMD_MAKE_KABSCH(name, input_type, accumulator_type, load_and_convert)                                 \
    SIMSIMD_PUBLIC void simsimd_kabsch_##input_type##_##name(                                                     \
        simsimd_##input_type##_t const *a, simsimd_##input_type##_t const *b, simsimd_size_t n,                   \
        simsimd_##input_type##_t *a_centroid, simsimd_##input_type##_t *b_centroid, simsimd_distance_t *result) { \
        /* Step 1: Compute centroids */                                                                           \
        simsimd_##accumulator_type##_t ax = 0, ay = 0, az = 0;                                                    \
        simsimd_##accumulator_type##_t bx = 0, by = 0, bz = 0;                                                    \
        for (simsimd_size_t i = 0; i < n; ++i) {                                                                  \
            ax += load_and_convert(a + i * 3 + 0);                                                                \
            ay += load_and_convert(a + i * 3 + 1);                                                                \
            az += load_and_convert(a + i * 3 + 2);                                                                \
            bx += load_and_convert(b + i * 3 + 0);                                                                \
            by += load_and_convert(b + i * 3 + 1);                                                                \
            bz += load_and_convert(b + i * 3 + 2);                                                                \
        }                                                                                                         \
        simsimd_##accumulator_type##_t inv_n = (simsimd_##accumulator_type##_t)1.0 / n;                           \
        ax *= inv_n;                                                                                              \
        ay *= inv_n;                                                                                              \
        az *= inv_n;                                                                                              \
        bx *= inv_n;                                                                                              \
        by *= inv_n;                                                                                              \
        bz *= inv_n;                                                                                              \
        if (a_centroid) {                                                                                         \
            a_centroid[0] = (simsimd_##input_type##_t)ax;                                                         \
            a_centroid[1] = (simsimd_##input_type##_t)ay;                                                         \
            a_centroid[2] = (simsimd_##input_type##_t)az;                                                         \
        }                                                                                                         \
        if (b_centroid) {                                                                                         \
            b_centroid[0] = (simsimd_##input_type##_t)bx;                                                         \
            b_centroid[1] = (simsimd_##input_type##_t)by;                                                         \
            b_centroid[2] = (simsimd_##input_type##_t)bz;                                                         \
        }                                                                                                         \
        /* Step 2: Build 3x3 covariance matrix H = (A - centroid_A)^T * (B - centroid_B) */                       \
        /* Use accumulator_type for high-precision accumulation */                                                \
        simsimd_##accumulator_type##_t h_acc[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};                                    \
        for (simsimd_size_t i = 0; i < n; ++i) {                                                                  \
            simsimd_##accumulator_type##_t pa[3], pb[3];                                                          \
            pa[0] = load_and_convert(a + i * 3 + 0) - ax;                                                         \
            pa[1] = load_and_convert(a + i * 3 + 1) - ay;                                                         \
            pa[2] = load_and_convert(a + i * 3 + 2) - az;                                                         \
            pb[0] = load_and_convert(b + i * 3 + 0) - bx;                                                         \
            pb[1] = load_and_convert(b + i * 3 + 1) - by;                                                         \
            pb[2] = load_and_convert(b + i * 3 + 2) - bz;                                                         \
            h_acc[0] += pa[0] * pb[0];                                                                            \
            h_acc[1] += pa[0] * pb[1];                                                                            \
            h_acc[2] += pa[0] * pb[2];                                                                            \
            h_acc[3] += pa[1] * pb[0];                                                                            \
            h_acc[4] += pa[1] * pb[1];                                                                            \
            h_acc[5] += pa[1] * pb[2];                                                                            \
            h_acc[6] += pa[2] * pb[0];                                                                            \
            h_acc[7] += pa[2] * pb[1];                                                                            \
            h_acc[8] += pa[2] * pb[2];                                                                            \
        }                                                                                                         \
        /* Convert to f32 for SVD (SVD precision is adequate at f32) */                                           \
        simsimd_f32_t h[9];                                                                                       \
        for (int j = 0; j < 9; ++j) h[j] = (simsimd_f32_t)h_acc[j];                                               \
        /* Step 3: SVD of H = U * S * V^T */                                                                      \
        simsimd_f32_t u[9], s[9], v[9];                                                                           \
        _simsimd_svd3x3_f32(h, u, s, v);                                                                          \
        /* Step 4: R = V * U^T */                                                                                 \
        simsimd_f32_t r[9];                                                                                       \
        r[0] = v[0] * u[0] + v[1] * u[1] + v[2] * u[2];                                                           \
        r[1] = v[0] * u[3] + v[1] * u[4] + v[2] * u[5];                                                           \
        r[2] = v[0] * u[6] + v[1] * u[7] + v[2] * u[8];                                                           \
        r[3] = v[3] * u[0] + v[4] * u[1] + v[5] * u[2];                                                           \
        r[4] = v[3] * u[3] + v[4] * u[4] + v[5] * u[5];                                                           \
        r[5] = v[3] * u[6] + v[4] * u[7] + v[5] * u[8];                                                           \
        r[6] = v[6] * u[0] + v[7] * u[1] + v[8] * u[2];                                                           \
        r[7] = v[6] * u[3] + v[7] * u[4] + v[8] * u[5];                                                           \
        r[8] = v[6] * u[6] + v[7] * u[7] + v[8] * u[8];                                                           \
        /* Handle reflection: if det(R) < 0, negate third column of V and recompute R */                          \
        simsimd_f32_t det = _simsimd_det3x3_f32(r);                                                               \
        if (det < 0) {                                                                                            \
            v[2] = -v[2];                                                                                         \
            v[5] = -v[5];                                                                                         \
            v[8] = -v[8];                                                                                         \
            r[0] = v[0] * u[0] + v[1] * u[1] + v[2] * u[2];                                                       \
            r[1] = v[0] * u[3] + v[1] * u[4] + v[2] * u[5];                                                       \
            r[2] = v[0] * u[6] + v[1] * u[7] + v[2] * u[8];                                                       \
            r[3] = v[3] * u[0] + v[4] * u[1] + v[5] * u[2];                                                       \
            r[4] = v[3] * u[3] + v[4] * u[4] + v[5] * u[5];                                                       \
            r[5] = v[3] * u[6] + v[4] * u[7] + v[5] * u[8];                                                       \
            r[6] = v[6] * u[0] + v[7] * u[1] + v[8] * u[2];                                                       \
            r[7] = v[6] * u[3] + v[7] * u[4] + v[8] * u[5];                                                       \
            r[8] = v[6] * u[6] + v[7] * u[7] + v[8] * u[8];                                                       \
        }                                                                                                         \
        /* Step 5: Compute RMSD after rotation */                                                                 \
        simsimd_##accumulator_type##_t sum_sq = 0;                                                                \
        for (simsimd_size_t i = 0; i < n; ++i) {                                                                  \
            simsimd_f32_t pa[3], pb[3], ra[3];                                                                    \
            pa[0] = (simsimd_f32_t)(load_and_convert(a + i * 3 + 0) - ax);                                        \
            pa[1] = (simsimd_f32_t)(load_and_convert(a + i * 3 + 1) - ay);                                        \
            pa[2] = (simsimd_f32_t)(load_and_convert(a + i * 3 + 2) - az);                                        \
            pb[0] = (simsimd_f32_t)(load_and_convert(b + i * 3 + 0) - bx);                                        \
            pb[1] = (simsimd_f32_t)(load_and_convert(b + i * 3 + 1) - by);                                        \
            pb[2] = (simsimd_f32_t)(load_and_convert(b + i * 3 + 2) - bz);                                        \
            ra[0] = r[0] * pa[0] + r[1] * pa[1] + r[2] * pa[2];                                                   \
            ra[1] = r[3] * pa[0] + r[4] * pa[1] + r[5] * pa[2];                                                   \
            ra[2] = r[6] * pa[0] + r[7] * pa[1] + r[8] * pa[2];                                                   \
            simsimd_f32_t dx = ra[0] - pb[0];                                                                     \
            simsimd_f32_t dy = ra[1] - pb[1];                                                                     \
            simsimd_f32_t dz = ra[2] - pb[2];                                                                     \
            sum_sq += dx * dx + dy * dy + dz * dz;                                                                \
        }                                                                                                         \
        *result = SIMSIMD_SQRT(sum_sq * inv_n);                                                                   \
    }

SIMSIMD_MAKE_RMSD(serial, f64, f64, SIMSIMD_DEREFERENCE)   // simsimd_rmsd_f64_serial
SIMSIMD_MAKE_KABSCH(serial, f64, f64, SIMSIMD_DEREFERENCE) // simsimd_kabsch_f64_serial

SIMSIMD_MAKE_RMSD(serial, f32, f32, SIMSIMD_DEREFERENCE)   // simsimd_rmsd_f32_serial
SIMSIMD_MAKE_KABSCH(serial, f32, f32, SIMSIMD_DEREFERENCE) // simsimd_kabsch_f32_serial

SIMSIMD_MAKE_RMSD(serial, f16, f32, SIMSIMD_F16_TO_F32)   // simsimd_rmsd_f16_serial
SIMSIMD_MAKE_KABSCH(serial, f16, f32, SIMSIMD_F16_TO_F32) // simsimd_kabsch_f16_serial

SIMSIMD_MAKE_RMSD(serial, bf16, f32, SIMSIMD_BF16_TO_F32)   // simsimd_rmsd_bf16_serial
SIMSIMD_MAKE_KABSCH(serial, bf16, f32, SIMSIMD_BF16_TO_F32) // simsimd_kabsch_bf16_serial

SIMSIMD_MAKE_RMSD(accurate, f32, f64, SIMSIMD_DEREFERENCE)   // simsimd_rmsd_f32_accurate
SIMSIMD_MAKE_KABSCH(accurate, f32, f64, SIMSIMD_DEREFERENCE) // simsimd_kabsch_f32_accurate

SIMSIMD_MAKE_RMSD(accurate, f16, f64, SIMSIMD_F16_TO_F32)   // simsimd_rmsd_f16_accurate
SIMSIMD_MAKE_KABSCH(accurate, f16, f64, SIMSIMD_F16_TO_F32) // simsimd_kabsch_f16_accurate

SIMSIMD_MAKE_RMSD(accurate, bf16, f64, SIMSIMD_BF16_TO_F32)   // simsimd_rmsd_bf16_accurate
SIMSIMD_MAKE_KABSCH(accurate, bf16, f64, SIMSIMD_BF16_TO_F32) // simsimd_kabsch_bf16_accurate

#if _SIMSIMD_TARGET_X86
#if SIMSIMD_TARGET_SKYLAKE
#pragma GCC push_options
#pragma GCC target("avx512f", "avx512vl", "avx512bw", "bmi2")
#pragma clang attribute push(__attribute__((target("avx512f,avx512vl,avx512bw,bmi2"))), apply_to = function)

#include <immintrin.h>

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
SIMSIMD_INTERNAL void _simsimd_deinterleave_f32x16_skylake(simsimd_f32_t const *ptr, __m512 *x_out, __m512 *y_out,
                                                           __m512 *z_out) {
    __m512 r0 = _mm512_loadu_ps(ptr);
    __m512 r1 = _mm512_loadu_ps(ptr + 16);
    __m512 r2 = _mm512_loadu_ps(ptr + 32);

    // X: r0[0,3,6,9,12,15] + r1[2,5,8,11,14] -> 11 elements, then + r2[1,4,7,10,13] -> 16 elements
    // Indices for permutex2var: 0-15 = from first operand, 16-31 = from second operand
    __m512i idx_x_01 = _mm512_setr_epi32(0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 0, 0, 0, 0, 0);
    __m512i idx_x_2 = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 17, 20, 23, 26, 29);
    __m512 x01 = _mm512_permutex2var_ps(r0, idx_x_01, r1);
    *x_out = _mm512_permutex2var_ps(x01, idx_x_2, r2);

    // Y: r0[1,4,7,10,13] + r1[0,3,6,9,12,15] -> 11 elements, then + r2[2,5,8,11,14] -> 16 elements
    __m512i idx_y_01 = _mm512_setr_epi32(1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 0, 0, 0, 0, 0);
    __m512i idx_y_2 = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 18, 21, 24, 27, 30);
    __m512 y01 = _mm512_permutex2var_ps(r0, idx_y_01, r1);
    *y_out = _mm512_permutex2var_ps(y01, idx_y_2, r2);

    // Z: r0[2,5,8,11,14] + r1[1,4,7,10,13] -> 10 elements, then + r2[0,3,6,9,12,15] -> 16 elements
    __m512i idx_z_01 = _mm512_setr_epi32(2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 0, 0, 0, 0, 0, 0);
    __m512i idx_z_2 = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 19, 22, 25, 28, 31);
    __m512 z01 = _mm512_permutex2var_ps(r0, idx_z_01, r1);
    *z_out = _mm512_permutex2var_ps(z01, idx_z_2, r2);
}

/*  Internal helper: Deinterleave 8 f64 3D points from xyz,xyz,xyz... to separate x,y,z vectors.
 *  Input: 24 consecutive f64 values (8 points * 3 coordinates)
 *  Output: Three __m512d vectors containing the x, y, z coordinates separately.
 */
SIMSIMD_INTERNAL void _simsimd_deinterleave_f64x8_skylake(simsimd_f64_t const *ptr, __m512d *x_out, __m512d *y_out,
                                                          __m512d *z_out) {
    __m512d r0 = _mm512_loadu_pd(ptr);      // elements 0-7
    __m512d r1 = _mm512_loadu_pd(ptr + 8);  // elements 8-15
    __m512d r2 = _mm512_loadu_pd(ptr + 16); // elements 16-23

    // X: positions 0,3,6,9,12,15,18,21 -> r0[0,3,6] + r1[1,4,7] + r2[2,5]
    __m512i idx_x_01 = _mm512_setr_epi64(0, 3, 6, 9, 12, 15, 0, 0);
    __m512i idx_x_2 = _mm512_setr_epi64(0, 1, 2, 3, 4, 5, 10, 13);
    __m512d x01 = _mm512_permutex2var_pd(r0, idx_x_01, r1);
    *x_out = _mm512_permutex2var_pd(x01, idx_x_2, r2);

    // Y: positions 1,4,7,10,13,16,19,22 -> r0[1,4,7] + r1[2,5] + r2[0,3,6]
    __m512i idx_y_01 = _mm512_setr_epi64(1, 4, 7, 10, 13, 0, 0, 0);
    __m512i idx_y_2 = _mm512_setr_epi64(0, 1, 2, 3, 4, 8, 11, 14);
    __m512d y01 = _mm512_permutex2var_pd(r0, idx_y_01, r1);
    *y_out = _mm512_permutex2var_pd(y01, idx_y_2, r2);

    // Z: positions 2,5,8,11,14,17,20,23 -> r0[2,5] + r1[0,3,6] + r2[1,4,7]
    __m512i idx_z_01 = _mm512_setr_epi64(2, 5, 8, 11, 14, 0, 0, 0);
    __m512i idx_z_2 = _mm512_setr_epi64(0, 1, 2, 3, 4, 9, 12, 15);
    __m512d z01 = _mm512_permutex2var_pd(r0, idx_z_01, r1);
    *z_out = _mm512_permutex2var_pd(z01, idx_z_2, r2);
}

SIMSIMD_PUBLIC void simsimd_rmsd_f32_skylake(simsimd_f32_t const *a, simsimd_f32_t const *b, simsimd_size_t n,
                                             simsimd_f32_t *a_centroid, simsimd_f32_t *b_centroid,
                                             simsimd_distance_t *result) {
    // Optimized fused single-pass implementation.
    // Computes centroids and squared differences in one pass using the identity:
    //   RMSD = sqrt(E[(a-mean_a) - (b-mean_b)]^2)
    //        = sqrt(E[(a-b)^2] - (mean_a - mean_b)^2)
    __m512i const gather_idx = _mm512_setr_epi32(0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45);
    __m512 const zeros = _mm512_setzero_ps();

    // Accumulators for centroids and squared differences
    __m512 sum_ax = zeros, sum_ay = zeros, sum_az = zeros;
    __m512 sum_bx = zeros, sum_by = zeros, sum_bz = zeros;
    __m512 sum_sq_x = zeros, sum_sq_y = zeros, sum_sq_z = zeros;

    __m512 a_x, a_y, a_z, b_x, b_y, b_z;
    simsimd_size_t i = 0;

    // Main loop with 2x unrolling for better latency hiding
    for (; i + 32 <= n; i += 32) {
        // Iteration 0
        _simsimd_deinterleave_f32x16_skylake(a + i * 3, &a_x, &a_y, &a_z);
        _simsimd_deinterleave_f32x16_skylake(b + i * 3, &b_x, &b_y, &b_z);

        sum_ax = _mm512_add_ps(sum_ax, a_x);
        sum_ay = _mm512_add_ps(sum_ay, a_y);
        sum_az = _mm512_add_ps(sum_az, a_z);
        sum_bx = _mm512_add_ps(sum_bx, b_x);
        sum_by = _mm512_add_ps(sum_by, b_y);
        sum_bz = _mm512_add_ps(sum_bz, b_z);

        __m512 dx = _mm512_sub_ps(a_x, b_x);
        __m512 dy = _mm512_sub_ps(a_y, b_y);
        __m512 dz = _mm512_sub_ps(a_z, b_z);

        sum_sq_x = _mm512_fmadd_ps(dx, dx, sum_sq_x);
        sum_sq_y = _mm512_fmadd_ps(dy, dy, sum_sq_y);
        sum_sq_z = _mm512_fmadd_ps(dz, dz, sum_sq_z);

        // Iteration 1
        __m512 a_x1, a_y1, a_z1, b_x1, b_y1, b_z1;
        _simsimd_deinterleave_f32x16_skylake(a + (i + 16) * 3, &a_x1, &a_y1, &a_z1);
        _simsimd_deinterleave_f32x16_skylake(b + (i + 16) * 3, &b_x1, &b_y1, &b_z1);

        sum_ax = _mm512_add_ps(sum_ax, a_x1);
        sum_ay = _mm512_add_ps(sum_ay, a_y1);
        sum_az = _mm512_add_ps(sum_az, a_z1);
        sum_bx = _mm512_add_ps(sum_bx, b_x1);
        sum_by = _mm512_add_ps(sum_by, b_y1);
        sum_bz = _mm512_add_ps(sum_bz, b_z1);

        __m512 dx1 = _mm512_sub_ps(a_x1, b_x1);
        __m512 dy1 = _mm512_sub_ps(a_y1, b_y1);
        __m512 dz1 = _mm512_sub_ps(a_z1, b_z1);

        sum_sq_x = _mm512_fmadd_ps(dx1, dx1, sum_sq_x);
        sum_sq_y = _mm512_fmadd_ps(dy1, dy1, sum_sq_y);
        sum_sq_z = _mm512_fmadd_ps(dz1, dz1, sum_sq_z);
    }

    // Handle 16-point remainder
    for (; i + 16 <= n; i += 16) {
        _simsimd_deinterleave_f32x16_skylake(a + i * 3, &a_x, &a_y, &a_z);
        _simsimd_deinterleave_f32x16_skylake(b + i * 3, &b_x, &b_y, &b_z);

        sum_ax = _mm512_add_ps(sum_ax, a_x);
        sum_ay = _mm512_add_ps(sum_ay, a_y);
        sum_az = _mm512_add_ps(sum_az, a_z);
        sum_bx = _mm512_add_ps(sum_bx, b_x);
        sum_by = _mm512_add_ps(sum_by, b_y);
        sum_bz = _mm512_add_ps(sum_bz, b_z);

        __m512 dx = _mm512_sub_ps(a_x, b_x);
        __m512 dy = _mm512_sub_ps(a_y, b_y);
        __m512 dz = _mm512_sub_ps(a_z, b_z);

        sum_sq_x = _mm512_fmadd_ps(dx, dx, sum_sq_x);
        sum_sq_y = _mm512_fmadd_ps(dy, dy, sum_sq_y);
        sum_sq_z = _mm512_fmadd_ps(dz, dz, sum_sq_z);
    }

    // Tail: use masked gather
    if (i < n) {
        simsimd_size_t tail = n - i;
        __mmask16 mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, tail);
        simsimd_f32_t const *a_tail = a + i * 3;
        simsimd_f32_t const *b_tail = b + i * 3;

        a_x = _mm512_mask_i32gather_ps(zeros, mask, gather_idx, a_tail + 0, 4);
        a_y = _mm512_mask_i32gather_ps(zeros, mask, gather_idx, a_tail + 1, 4);
        a_z = _mm512_mask_i32gather_ps(zeros, mask, gather_idx, a_tail + 2, 4);
        b_x = _mm512_mask_i32gather_ps(zeros, mask, gather_idx, b_tail + 0, 4);
        b_y = _mm512_mask_i32gather_ps(zeros, mask, gather_idx, b_tail + 1, 4);
        b_z = _mm512_mask_i32gather_ps(zeros, mask, gather_idx, b_tail + 2, 4);

        sum_ax = _mm512_add_ps(sum_ax, a_x);
        sum_ay = _mm512_add_ps(sum_ay, a_y);
        sum_az = _mm512_add_ps(sum_az, a_z);
        sum_bx = _mm512_add_ps(sum_bx, b_x);
        sum_by = _mm512_add_ps(sum_by, b_y);
        sum_bz = _mm512_add_ps(sum_bz, b_z);

        __m512 dx = _mm512_sub_ps(a_x, b_x);
        __m512 dy = _mm512_sub_ps(a_y, b_y);
        __m512 dz = _mm512_sub_ps(a_z, b_z);

        sum_sq_x = _mm512_fmadd_ps(dx, dx, sum_sq_x);
        sum_sq_y = _mm512_fmadd_ps(dy, dy, sum_sq_y);
        sum_sq_z = _mm512_fmadd_ps(dz, dz, sum_sq_z);
    }

    // Reduce and compute centroids
    simsimd_f32_t inv_n = 1.0f / (simsimd_f32_t)n;
    simsimd_f32_t a_cx = _mm512_reduce_add_ps(sum_ax) * inv_n;
    simsimd_f32_t a_cy = _mm512_reduce_add_ps(sum_ay) * inv_n;
    simsimd_f32_t a_cz = _mm512_reduce_add_ps(sum_az) * inv_n;
    simsimd_f32_t b_cx = _mm512_reduce_add_ps(sum_bx) * inv_n;
    simsimd_f32_t b_cy = _mm512_reduce_add_ps(sum_by) * inv_n;
    simsimd_f32_t b_cz = _mm512_reduce_add_ps(sum_bz) * inv_n;

    if (a_centroid) {
        a_centroid[0] = a_cx;
        a_centroid[1] = a_cy;
        a_centroid[2] = a_cz;
    }
    if (b_centroid) {
        b_centroid[0] = b_cx;
        b_centroid[1] = b_cy;
        b_centroid[2] = b_cz;
    }

    // Compute RMSD using the formula:
    // RMSD = sqrt(E[(a-b)^2] - (mean_a - mean_b)^2)
    simsimd_f32_t mean_diff_x = a_cx - b_cx;
    simsimd_f32_t mean_diff_y = a_cy - b_cy;
    simsimd_f32_t mean_diff_z = a_cz - b_cz;

    __m512 sum_sq_total = _mm512_add_ps(sum_sq_x, _mm512_add_ps(sum_sq_y, sum_sq_z));
    simsimd_f32_t sum_sq = _mm512_reduce_add_ps(sum_sq_total);
    simsimd_f32_t mean_diff_sq = mean_diff_x * mean_diff_x + mean_diff_y * mean_diff_y + mean_diff_z * mean_diff_z;

    *result = SIMSIMD_SQRT((simsimd_distance_t)(sum_sq * inv_n - mean_diff_sq));
}

SIMSIMD_PUBLIC void simsimd_kabsch_f32_skylake(simsimd_f32_t const *a, simsimd_f32_t const *b, simsimd_size_t n,
                                               simsimd_f32_t *a_centroid, simsimd_f32_t *b_centroid,
                                               simsimd_distance_t *result) {
    // Optimized fused single-pass implementation.
    // Computes centroids and covariance matrix in one pass using the identity:
    //   H_ij = sum((a_i - mean_a) * (b_j - mean_b))
    //        = sum(a_i * b_j) - sum(a_i) * sum(b_j) / n
    __m512i const gather_idx = _mm512_setr_epi32(0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45);
    __m512 const zeros_f32 = _mm512_setzero_ps();
    __m512d const zeros_f64 = _mm512_setzero_pd();

    // Accumulators for centroids
    __m512d a_x_sum = zeros_f64, a_y_sum = zeros_f64, a_z_sum = zeros_f64;
    __m512d b_x_sum = zeros_f64, b_y_sum = zeros_f64, b_z_sum = zeros_f64;

    // Accumulators for covariance matrix (sum of outer products)
    __m512d h00 = zeros_f64, h01 = zeros_f64, h02 = zeros_f64;
    __m512d h10 = zeros_f64, h11 = zeros_f64, h12 = zeros_f64;
    __m512d h20 = zeros_f64, h21 = zeros_f64, h22 = zeros_f64;

    simsimd_size_t i = 0;
    __m512 a_x_vec, a_y_vec, a_z_vec, b_x_vec, b_y_vec, b_z_vec;

    // Fused single-pass: accumulate sums and outer products together
    for (; i + 16 <= n; i += 16) {
        _simsimd_deinterleave_f32x16_skylake(a + i * 3, &a_x_vec, &a_y_vec, &a_z_vec);
        _simsimd_deinterleave_f32x16_skylake(b + i * 3, &b_x_vec, &b_y_vec, &b_z_vec);

        // Convert to f64 - low 8 elements
        __m512d a_x_lo = _mm512_cvtps_pd(_mm512_castps512_ps256(a_x_vec));
        __m512d a_y_lo = _mm512_cvtps_pd(_mm512_castps512_ps256(a_y_vec));
        __m512d a_z_lo = _mm512_cvtps_pd(_mm512_castps512_ps256(a_z_vec));
        __m512d b_x_lo = _mm512_cvtps_pd(_mm512_castps512_ps256(b_x_vec));
        __m512d b_y_lo = _mm512_cvtps_pd(_mm512_castps512_ps256(b_y_vec));
        __m512d b_z_lo = _mm512_cvtps_pd(_mm512_castps512_ps256(b_z_vec));

        // Accumulate centroids
        a_x_sum = _mm512_add_pd(a_x_sum, a_x_lo);
        a_y_sum = _mm512_add_pd(a_y_sum, a_y_lo);
        a_z_sum = _mm512_add_pd(a_z_sum, a_z_lo);
        b_x_sum = _mm512_add_pd(b_x_sum, b_x_lo);
        b_y_sum = _mm512_add_pd(b_y_sum, b_y_lo);
        b_z_sum = _mm512_add_pd(b_z_sum, b_z_lo);

        // Accumulate outer products (raw, not centered)
        h00 = _mm512_fmadd_pd(a_x_lo, b_x_lo, h00);
        h01 = _mm512_fmadd_pd(a_x_lo, b_y_lo, h01);
        h02 = _mm512_fmadd_pd(a_x_lo, b_z_lo, h02);
        h10 = _mm512_fmadd_pd(a_y_lo, b_x_lo, h10);
        h11 = _mm512_fmadd_pd(a_y_lo, b_y_lo, h11);
        h12 = _mm512_fmadd_pd(a_y_lo, b_z_lo, h12);
        h20 = _mm512_fmadd_pd(a_z_lo, b_x_lo, h20);
        h21 = _mm512_fmadd_pd(a_z_lo, b_y_lo, h21);
        h22 = _mm512_fmadd_pd(a_z_lo, b_z_lo, h22);

        // High 8 elements
        __m512d a_x_hi = _mm512_cvtps_pd(_mm512_extractf32x8_ps(a_x_vec, 1));
        __m512d a_y_hi = _mm512_cvtps_pd(_mm512_extractf32x8_ps(a_y_vec, 1));
        __m512d a_z_hi = _mm512_cvtps_pd(_mm512_extractf32x8_ps(a_z_vec, 1));
        __m512d b_x_hi = _mm512_cvtps_pd(_mm512_extractf32x8_ps(b_x_vec, 1));
        __m512d b_y_hi = _mm512_cvtps_pd(_mm512_extractf32x8_ps(b_y_vec, 1));
        __m512d b_z_hi = _mm512_cvtps_pd(_mm512_extractf32x8_ps(b_z_vec, 1));

        a_x_sum = _mm512_add_pd(a_x_sum, a_x_hi);
        a_y_sum = _mm512_add_pd(a_y_sum, a_y_hi);
        a_z_sum = _mm512_add_pd(a_z_sum, a_z_hi);
        b_x_sum = _mm512_add_pd(b_x_sum, b_x_hi);
        b_y_sum = _mm512_add_pd(b_y_sum, b_y_hi);
        b_z_sum = _mm512_add_pd(b_z_sum, b_z_hi);

        h00 = _mm512_fmadd_pd(a_x_hi, b_x_hi, h00);
        h01 = _mm512_fmadd_pd(a_x_hi, b_y_hi, h01);
        h02 = _mm512_fmadd_pd(a_x_hi, b_z_hi, h02);
        h10 = _mm512_fmadd_pd(a_y_hi, b_x_hi, h10);
        h11 = _mm512_fmadd_pd(a_y_hi, b_y_hi, h11);
        h12 = _mm512_fmadd_pd(a_y_hi, b_z_hi, h12);
        h20 = _mm512_fmadd_pd(a_z_hi, b_x_hi, h20);
        h21 = _mm512_fmadd_pd(a_z_hi, b_y_hi, h21);
        h22 = _mm512_fmadd_pd(a_z_hi, b_z_hi, h22);
    }

    // Tail: masked gather for remaining points
    if (i < n) {
        simsimd_size_t tail = n - i;
        __mmask16 mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, tail);
        simsimd_f32_t const *a_tail = a + i * 3;
        simsimd_f32_t const *b_tail = b + i * 3;
        a_x_vec = _mm512_mask_i32gather_ps(zeros_f32, mask, gather_idx, a_tail + 0, 4);
        a_y_vec = _mm512_mask_i32gather_ps(zeros_f32, mask, gather_idx, a_tail + 1, 4);
        a_z_vec = _mm512_mask_i32gather_ps(zeros_f32, mask, gather_idx, a_tail + 2, 4);
        b_x_vec = _mm512_mask_i32gather_ps(zeros_f32, mask, gather_idx, b_tail + 0, 4);
        b_y_vec = _mm512_mask_i32gather_ps(zeros_f32, mask, gather_idx, b_tail + 1, 4);
        b_z_vec = _mm512_mask_i32gather_ps(zeros_f32, mask, gather_idx, b_tail + 2, 4);

        __m512d a_x_lo = _mm512_cvtps_pd(_mm512_castps512_ps256(a_x_vec));
        __m512d a_y_lo = _mm512_cvtps_pd(_mm512_castps512_ps256(a_y_vec));
        __m512d a_z_lo = _mm512_cvtps_pd(_mm512_castps512_ps256(a_z_vec));
        __m512d b_x_lo = _mm512_cvtps_pd(_mm512_castps512_ps256(b_x_vec));
        __m512d b_y_lo = _mm512_cvtps_pd(_mm512_castps512_ps256(b_y_vec));
        __m512d b_z_lo = _mm512_cvtps_pd(_mm512_castps512_ps256(b_z_vec));

        a_x_sum = _mm512_add_pd(a_x_sum, a_x_lo);
        a_y_sum = _mm512_add_pd(a_y_sum, a_y_lo);
        a_z_sum = _mm512_add_pd(a_z_sum, a_z_lo);
        b_x_sum = _mm512_add_pd(b_x_sum, b_x_lo);
        b_y_sum = _mm512_add_pd(b_y_sum, b_y_lo);
        b_z_sum = _mm512_add_pd(b_z_sum, b_z_lo);

        h00 = _mm512_fmadd_pd(a_x_lo, b_x_lo, h00);
        h01 = _mm512_fmadd_pd(a_x_lo, b_y_lo, h01);
        h02 = _mm512_fmadd_pd(a_x_lo, b_z_lo, h02);
        h10 = _mm512_fmadd_pd(a_y_lo, b_x_lo, h10);
        h11 = _mm512_fmadd_pd(a_y_lo, b_y_lo, h11);
        h12 = _mm512_fmadd_pd(a_y_lo, b_z_lo, h12);
        h20 = _mm512_fmadd_pd(a_z_lo, b_x_lo, h20);
        h21 = _mm512_fmadd_pd(a_z_lo, b_y_lo, h21);
        h22 = _mm512_fmadd_pd(a_z_lo, b_z_lo, h22);

        __m512d a_x_hi = _mm512_cvtps_pd(_mm512_extractf32x8_ps(a_x_vec, 1));
        __m512d a_y_hi = _mm512_cvtps_pd(_mm512_extractf32x8_ps(a_y_vec, 1));
        __m512d a_z_hi = _mm512_cvtps_pd(_mm512_extractf32x8_ps(a_z_vec, 1));
        __m512d b_x_hi = _mm512_cvtps_pd(_mm512_extractf32x8_ps(b_x_vec, 1));
        __m512d b_y_hi = _mm512_cvtps_pd(_mm512_extractf32x8_ps(b_y_vec, 1));
        __m512d b_z_hi = _mm512_cvtps_pd(_mm512_extractf32x8_ps(b_z_vec, 1));

        a_x_sum = _mm512_add_pd(a_x_sum, a_x_hi);
        a_y_sum = _mm512_add_pd(a_y_sum, a_y_hi);
        a_z_sum = _mm512_add_pd(a_z_sum, a_z_hi);
        b_x_sum = _mm512_add_pd(b_x_sum, b_x_hi);
        b_y_sum = _mm512_add_pd(b_y_sum, b_y_hi);
        b_z_sum = _mm512_add_pd(b_z_sum, b_z_hi);

        h00 = _mm512_fmadd_pd(a_x_hi, b_x_hi, h00);
        h01 = _mm512_fmadd_pd(a_x_hi, b_y_hi, h01);
        h02 = _mm512_fmadd_pd(a_x_hi, b_z_hi, h02);
        h10 = _mm512_fmadd_pd(a_y_hi, b_x_hi, h10);
        h11 = _mm512_fmadd_pd(a_y_hi, b_y_hi, h11);
        h12 = _mm512_fmadd_pd(a_y_hi, b_z_hi, h12);
        h20 = _mm512_fmadd_pd(a_z_hi, b_x_hi, h20);
        h21 = _mm512_fmadd_pd(a_z_hi, b_y_hi, h21);
        h22 = _mm512_fmadd_pd(a_z_hi, b_z_hi, h22);
    }

    // Reduce centroids
    simsimd_f64_t inv_n_d = 1.0 / (simsimd_f64_t)n;
    simsimd_f64_t sum_ax = _mm512_reduce_add_pd(a_x_sum);
    simsimd_f64_t sum_ay = _mm512_reduce_add_pd(a_y_sum);
    simsimd_f64_t sum_az = _mm512_reduce_add_pd(a_z_sum);
    simsimd_f64_t sum_bx = _mm512_reduce_add_pd(b_x_sum);
    simsimd_f64_t sum_by = _mm512_reduce_add_pd(b_y_sum);
    simsimd_f64_t sum_bz = _mm512_reduce_add_pd(b_z_sum);

    simsimd_f32_t a_cx = (simsimd_f32_t)(sum_ax * inv_n_d);
    simsimd_f32_t a_cy = (simsimd_f32_t)(sum_ay * inv_n_d);
    simsimd_f32_t a_cz = (simsimd_f32_t)(sum_az * inv_n_d);
    simsimd_f32_t b_cx = (simsimd_f32_t)(sum_bx * inv_n_d);
    simsimd_f32_t b_cy = (simsimd_f32_t)(sum_by * inv_n_d);
    simsimd_f32_t b_cz = (simsimd_f32_t)(sum_bz * inv_n_d);

    if (a_centroid) {
        a_centroid[0] = a_cx;
        a_centroid[1] = a_cy;
        a_centroid[2] = a_cz;
    }
    if (b_centroid) {
        b_centroid[0] = b_cx;
        b_centroid[1] = b_cy;
        b_centroid[2] = b_cz;
    }

    // Compute centered covariance matrix: H_ij = sum(a_i*b_j) - sum_a_i * sum_b_j / n
    simsimd_f32_t h[9];
    h[0] = (simsimd_f32_t)(_mm512_reduce_add_pd(h00) - sum_ax * sum_bx * inv_n_d);
    h[1] = (simsimd_f32_t)(_mm512_reduce_add_pd(h01) - sum_ax * sum_by * inv_n_d);
    h[2] = (simsimd_f32_t)(_mm512_reduce_add_pd(h02) - sum_ax * sum_bz * inv_n_d);
    h[3] = (simsimd_f32_t)(_mm512_reduce_add_pd(h10) - sum_ay * sum_bx * inv_n_d);
    h[4] = (simsimd_f32_t)(_mm512_reduce_add_pd(h11) - sum_ay * sum_by * inv_n_d);
    h[5] = (simsimd_f32_t)(_mm512_reduce_add_pd(h12) - sum_ay * sum_bz * inv_n_d);
    h[6] = (simsimd_f32_t)(_mm512_reduce_add_pd(h20) - sum_az * sum_bx * inv_n_d);
    h[7] = (simsimd_f32_t)(_mm512_reduce_add_pd(h21) - sum_az * sum_by * inv_n_d);
    h[8] = (simsimd_f32_t)(_mm512_reduce_add_pd(h22) - sum_az * sum_bz * inv_n_d);

    // Step 3: SVD
    simsimd_f32_t u[9], s[9], v[9];
    _simsimd_svd3x3_f32(h, u, s, v);

    // Step 4: R = V * U^T
    simsimd_f32_t r[9];
    r[0] = v[0] * u[0] + v[1] * u[1] + v[2] * u[2];
    r[1] = v[0] * u[3] + v[1] * u[4] + v[2] * u[5];
    r[2] = v[0] * u[6] + v[1] * u[7] + v[2] * u[8];
    r[3] = v[3] * u[0] + v[4] * u[1] + v[5] * u[2];
    r[4] = v[3] * u[3] + v[4] * u[4] + v[5] * u[5];
    r[5] = v[3] * u[6] + v[4] * u[7] + v[5] * u[8];
    r[6] = v[6] * u[0] + v[7] * u[1] + v[8] * u[2];
    r[7] = v[6] * u[3] + v[7] * u[4] + v[8] * u[5];
    r[8] = v[6] * u[6] + v[7] * u[7] + v[8] * u[8];

    // Handle reflection
    simsimd_f32_t det = _simsimd_det3x3_f32(r);
    if (det < 0) {
        v[2] = -v[2];
        v[5] = -v[5];
        v[8] = -v[8];
        r[0] = v[0] * u[0] + v[1] * u[1] + v[2] * u[2];
        r[1] = v[0] * u[3] + v[1] * u[4] + v[2] * u[5];
        r[2] = v[0] * u[6] + v[1] * u[7] + v[2] * u[8];
        r[3] = v[3] * u[0] + v[4] * u[1] + v[5] * u[2];
        r[4] = v[3] * u[3] + v[4] * u[4] + v[5] * u[5];
        r[5] = v[3] * u[6] + v[4] * u[7] + v[5] * u[8];
        r[6] = v[6] * u[0] + v[7] * u[1] + v[8] * u[2];
        r[7] = v[6] * u[3] + v[7] * u[4] + v[8] * u[5];
        r[8] = v[6] * u[6] + v[7] * u[7] + v[8] * u[8];
    }

    // Step 5: Compute RMSD after rotation using shuffle-based deinterleave
    __m512d sum_sq_vec = zeros_f64;

    __m512 r0_vec = _mm512_set1_ps(r[0]), r1_vec = _mm512_set1_ps(r[1]), r2_vec = _mm512_set1_ps(r[2]);
    __m512 r3_vec = _mm512_set1_ps(r[3]), r4_vec = _mm512_set1_ps(r[4]), r5_vec = _mm512_set1_ps(r[5]);
    __m512 r6_vec = _mm512_set1_ps(r[6]), r7_vec = _mm512_set1_ps(r[7]), r8_vec = _mm512_set1_ps(r[8]);
    __m512 a_cx_vec = _mm512_set1_ps(a_cx), a_cy_vec = _mm512_set1_ps(a_cy), a_cz_vec = _mm512_set1_ps(a_cz);
    __m512 b_cx_vec = _mm512_set1_ps(b_cx), b_cy_vec = _mm512_set1_ps(b_cy), b_cz_vec = _mm512_set1_ps(b_cz);

    // Main loop with shuffle-based deinterleave
    for (i = 0; i + 16 <= n; i += 16) {
        _simsimd_deinterleave_f32x16_skylake(a + i * 3, &a_x_vec, &a_y_vec, &a_z_vec);
        _simsimd_deinterleave_f32x16_skylake(b + i * 3, &b_x_vec, &b_y_vec, &b_z_vec);

        // Center points
        a_x_vec = _mm512_sub_ps(a_x_vec, a_cx_vec);
        a_y_vec = _mm512_sub_ps(a_y_vec, a_cy_vec);
        a_z_vec = _mm512_sub_ps(a_z_vec, a_cz_vec);
        b_x_vec = _mm512_sub_ps(b_x_vec, b_cx_vec);
        b_y_vec = _mm512_sub_ps(b_y_vec, b_cy_vec);
        b_z_vec = _mm512_sub_ps(b_z_vec, b_cz_vec);

        // R * a_centered
        __m512 ra_x_vec =
            _mm512_fmadd_ps(r0_vec, a_x_vec, _mm512_fmadd_ps(r1_vec, a_y_vec, _mm512_mul_ps(r2_vec, a_z_vec)));
        __m512 ra_y_vec =
            _mm512_fmadd_ps(r3_vec, a_x_vec, _mm512_fmadd_ps(r4_vec, a_y_vec, _mm512_mul_ps(r5_vec, a_z_vec)));
        __m512 ra_z_vec =
            _mm512_fmadd_ps(r6_vec, a_x_vec, _mm512_fmadd_ps(r7_vec, a_y_vec, _mm512_mul_ps(r8_vec, a_z_vec)));

        __m512 dx_vec = _mm512_sub_ps(ra_x_vec, b_x_vec);
        __m512 dy_vec = _mm512_sub_ps(ra_y_vec, b_y_vec);
        __m512 dz_vec = _mm512_sub_ps(ra_z_vec, b_z_vec);

        // Accumulate in f64 for precision - low 8 elements
        __m512d dx_lo = _mm512_cvtps_pd(_mm512_castps512_ps256(dx_vec));
        __m512d dy_lo = _mm512_cvtps_pd(_mm512_castps512_ps256(dy_vec));
        __m512d dz_lo = _mm512_cvtps_pd(_mm512_castps512_ps256(dz_vec));
        sum_sq_vec = _mm512_fmadd_pd(dx_lo, dx_lo, sum_sq_vec);
        sum_sq_vec = _mm512_fmadd_pd(dy_lo, dy_lo, sum_sq_vec);
        sum_sq_vec = _mm512_fmadd_pd(dz_lo, dz_lo, sum_sq_vec);
        // High 8 elements
        __m512d dx_hi = _mm512_cvtps_pd(_mm512_extractf32x8_ps(dx_vec, 1));
        __m512d dy_hi = _mm512_cvtps_pd(_mm512_extractf32x8_ps(dy_vec, 1));
        __m512d dz_hi = _mm512_cvtps_pd(_mm512_extractf32x8_ps(dz_vec, 1));
        sum_sq_vec = _mm512_fmadd_pd(dx_hi, dx_hi, sum_sq_vec);
        sum_sq_vec = _mm512_fmadd_pd(dy_hi, dy_hi, sum_sq_vec);
        sum_sq_vec = _mm512_fmadd_pd(dz_hi, dz_hi, sum_sq_vec);
    }

    // Tail with masked gather
    if (i < n) {
        simsimd_size_t tail = n - i;
        __mmask16 mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, tail);
        simsimd_f32_t const *a_tail = a + i * 3;
        simsimd_f32_t const *b_tail = b + i * 3;
        a_x_vec = _mm512_mask_i32gather_ps(zeros_f32, mask, gather_idx, a_tail + 0, 4);
        a_y_vec = _mm512_mask_i32gather_ps(zeros_f32, mask, gather_idx, a_tail + 1, 4);
        a_z_vec = _mm512_mask_i32gather_ps(zeros_f32, mask, gather_idx, a_tail + 2, 4);
        b_x_vec = _mm512_mask_i32gather_ps(zeros_f32, mask, gather_idx, b_tail + 0, 4);
        b_y_vec = _mm512_mask_i32gather_ps(zeros_f32, mask, gather_idx, b_tail + 1, 4);
        b_z_vec = _mm512_mask_i32gather_ps(zeros_f32, mask, gather_idx, b_tail + 2, 4);

        a_x_vec = _mm512_sub_ps(a_x_vec, a_cx_vec);
        a_y_vec = _mm512_sub_ps(a_y_vec, a_cy_vec);
        a_z_vec = _mm512_sub_ps(a_z_vec, a_cz_vec);
        b_x_vec = _mm512_sub_ps(b_x_vec, b_cx_vec);
        b_y_vec = _mm512_sub_ps(b_y_vec, b_cy_vec);
        b_z_vec = _mm512_sub_ps(b_z_vec, b_cz_vec);

        __m512 ra_x_vec =
            _mm512_fmadd_ps(r0_vec, a_x_vec, _mm512_fmadd_ps(r1_vec, a_y_vec, _mm512_mul_ps(r2_vec, a_z_vec)));
        __m512 ra_y_vec =
            _mm512_fmadd_ps(r3_vec, a_x_vec, _mm512_fmadd_ps(r4_vec, a_y_vec, _mm512_mul_ps(r5_vec, a_z_vec)));
        __m512 ra_z_vec =
            _mm512_fmadd_ps(r6_vec, a_x_vec, _mm512_fmadd_ps(r7_vec, a_y_vec, _mm512_mul_ps(r8_vec, a_z_vec)));

        __m512 dx_vec = _mm512_sub_ps(ra_x_vec, b_x_vec);
        __m512 dy_vec = _mm512_sub_ps(ra_y_vec, b_y_vec);
        __m512 dz_vec = _mm512_sub_ps(ra_z_vec, b_z_vec);

        __m512d dx_lo = _mm512_cvtps_pd(_mm512_castps512_ps256(dx_vec));
        __m512d dy_lo = _mm512_cvtps_pd(_mm512_castps512_ps256(dy_vec));
        __m512d dz_lo = _mm512_cvtps_pd(_mm512_castps512_ps256(dz_vec));
        sum_sq_vec = _mm512_fmadd_pd(dx_lo, dx_lo, sum_sq_vec);
        sum_sq_vec = _mm512_fmadd_pd(dy_lo, dy_lo, sum_sq_vec);
        sum_sq_vec = _mm512_fmadd_pd(dz_lo, dz_lo, sum_sq_vec);
        __m512d dx_hi = _mm512_cvtps_pd(_mm512_extractf32x8_ps(dx_vec, 1));
        __m512d dy_hi = _mm512_cvtps_pd(_mm512_extractf32x8_ps(dy_vec, 1));
        __m512d dz_hi = _mm512_cvtps_pd(_mm512_extractf32x8_ps(dz_vec, 1));
        sum_sq_vec = _mm512_fmadd_pd(dx_hi, dx_hi, sum_sq_vec);
        sum_sq_vec = _mm512_fmadd_pd(dy_hi, dy_hi, sum_sq_vec);
        sum_sq_vec = _mm512_fmadd_pd(dz_hi, dz_hi, sum_sq_vec);
    }

    *result = SIMSIMD_SQRT((simsimd_distance_t)_mm512_reduce_add_pd(sum_sq_vec) * inv_n_d);
}

SIMSIMD_PUBLIC void simsimd_rmsd_f64_skylake(simsimd_f64_t const *a, simsimd_f64_t const *b, simsimd_size_t n,
                                             simsimd_f64_t *a_centroid, simsimd_f64_t *b_centroid,
                                             simsimd_distance_t *result) {
    // Optimized fused single-pass implementation for f64.
    // Computes centroids and squared differences in one pass using the identity:
    //   RMSD = sqrt(E[(a-mean_a) - (b-mean_b)]^2)
    //        = sqrt(E[(a-b)^2] - (mean_a - mean_b)^2)
    __m512i const gather_idx = _mm512_setr_epi64(0, 3, 6, 9, 12, 15, 18, 21);
    __m512d const zeros = _mm512_setzero_pd();

    // Accumulators for centroids and squared differences
    __m512d sum_ax = zeros, sum_ay = zeros, sum_az = zeros;
    __m512d sum_bx = zeros, sum_by = zeros, sum_bz = zeros;
    __m512d sum_sq_x = zeros, sum_sq_y = zeros, sum_sq_z = zeros;

    __m512d a_x, a_y, a_z, b_x, b_y, b_z;
    simsimd_size_t i = 0;

    // Main loop with 2x unrolling for better latency hiding
    for (; i + 16 <= n; i += 16) {
        // Iteration 0
        _simsimd_deinterleave_f64x8_skylake(a + i * 3, &a_x, &a_y, &a_z);
        _simsimd_deinterleave_f64x8_skylake(b + i * 3, &b_x, &b_y, &b_z);

        sum_ax = _mm512_add_pd(sum_ax, a_x), sum_ay = _mm512_add_pd(sum_ay, a_y), sum_az = _mm512_add_pd(sum_az, a_z);
        sum_bx = _mm512_add_pd(sum_bx, b_x), sum_by = _mm512_add_pd(sum_by, b_y), sum_bz = _mm512_add_pd(sum_bz, b_z);

        __m512d dx = _mm512_sub_pd(a_x, b_x), dy = _mm512_sub_pd(a_y, b_y), dz = _mm512_sub_pd(a_z, b_z);
        sum_sq_x = _mm512_fmadd_pd(dx, dx, sum_sq_x);
        sum_sq_y = _mm512_fmadd_pd(dy, dy, sum_sq_y);
        sum_sq_z = _mm512_fmadd_pd(dz, dz, sum_sq_z);

        // Iteration 1
        __m512d a_x1, a_y1, a_z1, b_x1, b_y1, b_z1;
        _simsimd_deinterleave_f64x8_skylake(a + (i + 8) * 3, &a_x1, &a_y1, &a_z1);
        _simsimd_deinterleave_f64x8_skylake(b + (i + 8) * 3, &b_x1, &b_y1, &b_z1);

        sum_ax = _mm512_add_pd(sum_ax, a_x1), sum_ay = _mm512_add_pd(sum_ay, a_y1),
        sum_az = _mm512_add_pd(sum_az, a_z1);
        sum_bx = _mm512_add_pd(sum_bx, b_x1), sum_by = _mm512_add_pd(sum_by, b_y1),
        sum_bz = _mm512_add_pd(sum_bz, b_z1);

        __m512d dx1 = _mm512_sub_pd(a_x1, b_x1), dy1 = _mm512_sub_pd(a_y1, b_y1), dz1 = _mm512_sub_pd(a_z1, b_z1);
        sum_sq_x = _mm512_fmadd_pd(dx1, dx1, sum_sq_x);
        sum_sq_y = _mm512_fmadd_pd(dy1, dy1, sum_sq_y);
        sum_sq_z = _mm512_fmadd_pd(dz1, dz1, sum_sq_z);
    }

    // Handle 8-point remainder
    for (; i + 8 <= n; i += 8) {
        _simsimd_deinterleave_f64x8_skylake(a + i * 3, &a_x, &a_y, &a_z);
        _simsimd_deinterleave_f64x8_skylake(b + i * 3, &b_x, &b_y, &b_z);

        sum_ax = _mm512_add_pd(sum_ax, a_x), sum_ay = _mm512_add_pd(sum_ay, a_y), sum_az = _mm512_add_pd(sum_az, a_z);
        sum_bx = _mm512_add_pd(sum_bx, b_x), sum_by = _mm512_add_pd(sum_by, b_y), sum_bz = _mm512_add_pd(sum_bz, b_z);

        __m512d dx = _mm512_sub_pd(a_x, b_x), dy = _mm512_sub_pd(a_y, b_y), dz = _mm512_sub_pd(a_z, b_z);
        sum_sq_x = _mm512_fmadd_pd(dx, dx, sum_sq_x);
        sum_sq_y = _mm512_fmadd_pd(dy, dy, sum_sq_y);
        sum_sq_z = _mm512_fmadd_pd(dz, dz, sum_sq_z);
    }

    // Tail: use masked gather
    if (i < n) {
        simsimd_size_t tail = n - i;
        __mmask8 mask = (__mmask8)_bzhi_u32(0xFF, tail);
        simsimd_f64_t const *a_tail = a + i * 3;
        simsimd_f64_t const *b_tail = b + i * 3;

        a_x = _mm512_mask_i64gather_pd(zeros, mask, gather_idx, a_tail + 0, 8);
        a_y = _mm512_mask_i64gather_pd(zeros, mask, gather_idx, a_tail + 1, 8);
        a_z = _mm512_mask_i64gather_pd(zeros, mask, gather_idx, a_tail + 2, 8);
        b_x = _mm512_mask_i64gather_pd(zeros, mask, gather_idx, b_tail + 0, 8);
        b_y = _mm512_mask_i64gather_pd(zeros, mask, gather_idx, b_tail + 1, 8);
        b_z = _mm512_mask_i64gather_pd(zeros, mask, gather_idx, b_tail + 2, 8);

        sum_ax = _mm512_add_pd(sum_ax, a_x), sum_ay = _mm512_add_pd(sum_ay, a_y), sum_az = _mm512_add_pd(sum_az, a_z);
        sum_bx = _mm512_add_pd(sum_bx, b_x), sum_by = _mm512_add_pd(sum_by, b_y), sum_bz = _mm512_add_pd(sum_bz, b_z);

        __m512d dx = _mm512_sub_pd(a_x, b_x), dy = _mm512_sub_pd(a_y, b_y), dz = _mm512_sub_pd(a_z, b_z);
        sum_sq_x = _mm512_fmadd_pd(dx, dx, sum_sq_x);
        sum_sq_y = _mm512_fmadd_pd(dy, dy, sum_sq_y);
        sum_sq_z = _mm512_fmadd_pd(dz, dz, sum_sq_z);
    }

    // Reduce and compute centroids
    simsimd_f64_t inv_n = 1.0 / (simsimd_f64_t)n;
    simsimd_f64_t a_cx = _mm512_reduce_add_pd(sum_ax) * inv_n;
    simsimd_f64_t a_cy = _mm512_reduce_add_pd(sum_ay) * inv_n;
    simsimd_f64_t a_cz = _mm512_reduce_add_pd(sum_az) * inv_n;
    simsimd_f64_t b_cx = _mm512_reduce_add_pd(sum_bx) * inv_n;
    simsimd_f64_t b_cy = _mm512_reduce_add_pd(sum_by) * inv_n;
    simsimd_f64_t b_cz = _mm512_reduce_add_pd(sum_bz) * inv_n;

    if (a_centroid) a_centroid[0] = a_cx, a_centroid[1] = a_cy, a_centroid[2] = a_cz;
    if (b_centroid) b_centroid[0] = b_cx, b_centroid[1] = b_cy, b_centroid[2] = b_cz;

    // Compute RMSD using the formula:
    // RMSD = sqrt(E[(a-b)^2] - (mean_a - mean_b)^2)
    simsimd_f64_t mean_diff_x = a_cx - b_cx, mean_diff_y = a_cy - b_cy, mean_diff_z = a_cz - b_cz;
    __m512d sum_sq_total = _mm512_add_pd(sum_sq_x, _mm512_add_pd(sum_sq_y, sum_sq_z));
    simsimd_f64_t sum_sq = _mm512_reduce_add_pd(sum_sq_total);
    simsimd_f64_t mean_diff_sq = mean_diff_x * mean_diff_x + mean_diff_y * mean_diff_y + mean_diff_z * mean_diff_z;

    *result = SIMSIMD_SQRT((simsimd_distance_t)(sum_sq * inv_n - mean_diff_sq));
}

SIMSIMD_PUBLIC void simsimd_kabsch_f64_skylake(simsimd_f64_t const *a, simsimd_f64_t const *b, simsimd_size_t n,
                                               simsimd_f64_t *a_centroid, simsimd_f64_t *b_centroid,
                                               simsimd_distance_t *result) {
    // Optimized fused single-pass implementation for f64.
    // Computes centroids and covariance matrix in one pass using the identity:
    //   H_ij = sum((a_i - mean_a) * (b_j - mean_b))
    //        = sum(a_i * b_j) - sum(a_i) * sum(b_j) / n
    __m512i const gather_idx = _mm512_setr_epi64(0, 3, 6, 9, 12, 15, 18, 21);
    __m512d const zeros = _mm512_setzero_pd();

    // Accumulators for centroids
    __m512d a_x_sum = zeros, a_y_sum = zeros, a_z_sum = zeros;
    __m512d b_x_sum = zeros, b_y_sum = zeros, b_z_sum = zeros;

    // Accumulators for covariance matrix (sum of outer products)
    __m512d h00 = zeros, h01 = zeros, h02 = zeros;
    __m512d h10 = zeros, h11 = zeros, h12 = zeros;
    __m512d h20 = zeros, h21 = zeros, h22 = zeros;

    simsimd_size_t i = 0;
    __m512d a_x, a_y, a_z, b_x, b_y, b_z;

    // Fused single-pass: accumulate sums and outer products together
    for (; i + 8 <= n; i += 8) {
        _simsimd_deinterleave_f64x8_skylake(a + i * 3, &a_x, &a_y, &a_z);
        _simsimd_deinterleave_f64x8_skylake(b + i * 3, &b_x, &b_y, &b_z);

        // Accumulate centroids
        a_x_sum = _mm512_add_pd(a_x_sum, a_x), a_y_sum = _mm512_add_pd(a_y_sum, a_y),
        a_z_sum = _mm512_add_pd(a_z_sum, a_z);
        b_x_sum = _mm512_add_pd(b_x_sum, b_x), b_y_sum = _mm512_add_pd(b_y_sum, b_y),
        b_z_sum = _mm512_add_pd(b_z_sum, b_z);

        // Accumulate outer products (raw, not centered)
        h00 = _mm512_fmadd_pd(a_x, b_x, h00), h01 = _mm512_fmadd_pd(a_x, b_y, h01),
        h02 = _mm512_fmadd_pd(a_x, b_z, h02);
        h10 = _mm512_fmadd_pd(a_y, b_x, h10), h11 = _mm512_fmadd_pd(a_y, b_y, h11),
        h12 = _mm512_fmadd_pd(a_y, b_z, h12);
        h20 = _mm512_fmadd_pd(a_z, b_x, h20), h21 = _mm512_fmadd_pd(a_z, b_y, h21),
        h22 = _mm512_fmadd_pd(a_z, b_z, h22);
    }

    // Tail: masked gather for remaining points
    if (i < n) {
        simsimd_size_t tail = n - i;
        __mmask8 mask = (__mmask8)_bzhi_u32(0xFF, tail);
        simsimd_f64_t const *a_tail = a + i * 3;
        simsimd_f64_t const *b_tail = b + i * 3;

        a_x = _mm512_mask_i64gather_pd(zeros, mask, gather_idx, a_tail + 0, 8);
        a_y = _mm512_mask_i64gather_pd(zeros, mask, gather_idx, a_tail + 1, 8);
        a_z = _mm512_mask_i64gather_pd(zeros, mask, gather_idx, a_tail + 2, 8);
        b_x = _mm512_mask_i64gather_pd(zeros, mask, gather_idx, b_tail + 0, 8);
        b_y = _mm512_mask_i64gather_pd(zeros, mask, gather_idx, b_tail + 1, 8);
        b_z = _mm512_mask_i64gather_pd(zeros, mask, gather_idx, b_tail + 2, 8);

        a_x_sum = _mm512_add_pd(a_x_sum, a_x), a_y_sum = _mm512_add_pd(a_y_sum, a_y),
        a_z_sum = _mm512_add_pd(a_z_sum, a_z);
        b_x_sum = _mm512_add_pd(b_x_sum, b_x), b_y_sum = _mm512_add_pd(b_y_sum, b_y),
        b_z_sum = _mm512_add_pd(b_z_sum, b_z);

        h00 = _mm512_fmadd_pd(a_x, b_x, h00), h01 = _mm512_fmadd_pd(a_x, b_y, h01),
        h02 = _mm512_fmadd_pd(a_x, b_z, h02);
        h10 = _mm512_fmadd_pd(a_y, b_x, h10), h11 = _mm512_fmadd_pd(a_y, b_y, h11),
        h12 = _mm512_fmadd_pd(a_y, b_z, h12);
        h20 = _mm512_fmadd_pd(a_z, b_x, h20), h21 = _mm512_fmadd_pd(a_z, b_y, h21),
        h22 = _mm512_fmadd_pd(a_z, b_z, h22);
    }

    // Reduce centroids
    simsimd_f64_t inv_n = 1.0 / (simsimd_f64_t)n;
    simsimd_f64_t sum_ax = _mm512_reduce_add_pd(a_x_sum), sum_ay = _mm512_reduce_add_pd(a_y_sum),
                  sum_az = _mm512_reduce_add_pd(a_z_sum);
    simsimd_f64_t sum_bx = _mm512_reduce_add_pd(b_x_sum), sum_by = _mm512_reduce_add_pd(b_y_sum),
                  sum_bz = _mm512_reduce_add_pd(b_z_sum);

    simsimd_f64_t a_cx = sum_ax * inv_n, a_cy = sum_ay * inv_n, a_cz = sum_az * inv_n;
    simsimd_f64_t b_cx = sum_bx * inv_n, b_cy = sum_by * inv_n, b_cz = sum_bz * inv_n;

    if (a_centroid) a_centroid[0] = a_cx, a_centroid[1] = a_cy, a_centroid[2] = a_cz;
    if (b_centroid) b_centroid[0] = b_cx, b_centroid[1] = b_cy, b_centroid[2] = b_cz;

    // Compute centered covariance matrix: H_ij = sum(a_i*b_j) - sum_a_i * sum_b_j / n
    simsimd_f32_t h[9];
    h[0] = (simsimd_f32_t)(_mm512_reduce_add_pd(h00) - sum_ax * sum_bx * inv_n);
    h[1] = (simsimd_f32_t)(_mm512_reduce_add_pd(h01) - sum_ax * sum_by * inv_n);
    h[2] = (simsimd_f32_t)(_mm512_reduce_add_pd(h02) - sum_ax * sum_bz * inv_n);
    h[3] = (simsimd_f32_t)(_mm512_reduce_add_pd(h10) - sum_ay * sum_bx * inv_n);
    h[4] = (simsimd_f32_t)(_mm512_reduce_add_pd(h11) - sum_ay * sum_by * inv_n);
    h[5] = (simsimd_f32_t)(_mm512_reduce_add_pd(h12) - sum_ay * sum_bz * inv_n);
    h[6] = (simsimd_f32_t)(_mm512_reduce_add_pd(h20) - sum_az * sum_bx * inv_n);
    h[7] = (simsimd_f32_t)(_mm512_reduce_add_pd(h21) - sum_az * sum_by * inv_n);
    h[8] = (simsimd_f32_t)(_mm512_reduce_add_pd(h22) - sum_az * sum_bz * inv_n);

    // SVD (f32 is sufficient for rotation matrix)
    simsimd_f32_t u[9], s[9], v[9];
    _simsimd_svd3x3_f32(h, u, s, v);

    // R = V * U^T
    simsimd_f32_t r[9];
    r[0] = v[0] * u[0] + v[1] * u[1] + v[2] * u[2];
    r[1] = v[0] * u[3] + v[1] * u[4] + v[2] * u[5];
    r[2] = v[0] * u[6] + v[1] * u[7] + v[2] * u[8];
    r[3] = v[3] * u[0] + v[4] * u[1] + v[5] * u[2];
    r[4] = v[3] * u[3] + v[4] * u[4] + v[5] * u[5];
    r[5] = v[3] * u[6] + v[4] * u[7] + v[5] * u[8];
    r[6] = v[6] * u[0] + v[7] * u[1] + v[8] * u[2];
    r[7] = v[6] * u[3] + v[7] * u[4] + v[8] * u[5];
    r[8] = v[6] * u[6] + v[7] * u[7] + v[8] * u[8];

    // Handle reflection
    if (_simsimd_det3x3_f32(r) < 0) {
        v[2] = -v[2], v[5] = -v[5], v[8] = -v[8];
        r[0] = v[0] * u[0] + v[1] * u[1] + v[2] * u[2];
        r[1] = v[0] * u[3] + v[1] * u[4] + v[2] * u[5];
        r[2] = v[0] * u[6] + v[1] * u[7] + v[2] * u[8];
        r[3] = v[3] * u[0] + v[4] * u[1] + v[5] * u[2];
        r[4] = v[3] * u[3] + v[4] * u[4] + v[5] * u[5];
        r[5] = v[3] * u[6] + v[4] * u[7] + v[5] * u[8];
        r[6] = v[6] * u[0] + v[7] * u[1] + v[8] * u[2];
        r[7] = v[6] * u[3] + v[7] * u[4] + v[8] * u[5];
        r[8] = v[6] * u[6] + v[7] * u[7] + v[8] * u[8];
    }

    // Compute RMSD after rotation using f64 throughout
    __m512d sum_sq_vec = zeros;
    __m512d r0_vec = _mm512_set1_pd(r[0]), r1_vec = _mm512_set1_pd(r[1]), r2_vec = _mm512_set1_pd(r[2]);
    __m512d r3_vec = _mm512_set1_pd(r[3]), r4_vec = _mm512_set1_pd(r[4]), r5_vec = _mm512_set1_pd(r[5]);
    __m512d r6_vec = _mm512_set1_pd(r[6]), r7_vec = _mm512_set1_pd(r[7]), r8_vec = _mm512_set1_pd(r[8]);
    __m512d a_cx_vec = _mm512_set1_pd(a_cx), a_cy_vec = _mm512_set1_pd(a_cy), a_cz_vec = _mm512_set1_pd(a_cz);
    __m512d b_cx_vec = _mm512_set1_pd(b_cx), b_cy_vec = _mm512_set1_pd(b_cy), b_cz_vec = _mm512_set1_pd(b_cz);

    for (i = 0; i + 8 <= n; i += 8) {
        _simsimd_deinterleave_f64x8_skylake(a + i * 3, &a_x, &a_y, &a_z);
        _simsimd_deinterleave_f64x8_skylake(b + i * 3, &b_x, &b_y, &b_z);

        // Center points
        a_x = _mm512_sub_pd(a_x, a_cx_vec), a_y = _mm512_sub_pd(a_y, a_cy_vec), a_z = _mm512_sub_pd(a_z, a_cz_vec);
        b_x = _mm512_sub_pd(b_x, b_cx_vec), b_y = _mm512_sub_pd(b_y, b_cy_vec), b_z = _mm512_sub_pd(b_z, b_cz_vec);

        // R * a_centered
        __m512d ra_x = _mm512_fmadd_pd(r0_vec, a_x, _mm512_fmadd_pd(r1_vec, a_y, _mm512_mul_pd(r2_vec, a_z)));
        __m512d ra_y = _mm512_fmadd_pd(r3_vec, a_x, _mm512_fmadd_pd(r4_vec, a_y, _mm512_mul_pd(r5_vec, a_z)));
        __m512d ra_z = _mm512_fmadd_pd(r6_vec, a_x, _mm512_fmadd_pd(r7_vec, a_y, _mm512_mul_pd(r8_vec, a_z)));

        __m512d dx = _mm512_sub_pd(ra_x, b_x), dy = _mm512_sub_pd(ra_y, b_y), dz = _mm512_sub_pd(ra_z, b_z);
        sum_sq_vec = _mm512_fmadd_pd(dx, dx, sum_sq_vec);
        sum_sq_vec = _mm512_fmadd_pd(dy, dy, sum_sq_vec);
        sum_sq_vec = _mm512_fmadd_pd(dz, dz, sum_sq_vec);
    }

    // Tail with masked gather
    if (i < n) {
        simsimd_size_t tail = n - i;
        __mmask8 mask = (__mmask8)_bzhi_u32(0xFF, tail);
        simsimd_f64_t const *a_tail = a + i * 3;
        simsimd_f64_t const *b_tail = b + i * 3;

        a_x = _mm512_mask_i64gather_pd(zeros, mask, gather_idx, a_tail + 0, 8);
        a_y = _mm512_mask_i64gather_pd(zeros, mask, gather_idx, a_tail + 1, 8);
        a_z = _mm512_mask_i64gather_pd(zeros, mask, gather_idx, a_tail + 2, 8);
        b_x = _mm512_mask_i64gather_pd(zeros, mask, gather_idx, b_tail + 0, 8);
        b_y = _mm512_mask_i64gather_pd(zeros, mask, gather_idx, b_tail + 1, 8);
        b_z = _mm512_mask_i64gather_pd(zeros, mask, gather_idx, b_tail + 2, 8);

        a_x = _mm512_sub_pd(a_x, a_cx_vec), a_y = _mm512_sub_pd(a_y, a_cy_vec), a_z = _mm512_sub_pd(a_z, a_cz_vec);
        b_x = _mm512_sub_pd(b_x, b_cx_vec), b_y = _mm512_sub_pd(b_y, b_cy_vec), b_z = _mm512_sub_pd(b_z, b_cz_vec);

        __m512d ra_x = _mm512_fmadd_pd(r0_vec, a_x, _mm512_fmadd_pd(r1_vec, a_y, _mm512_mul_pd(r2_vec, a_z)));
        __m512d ra_y = _mm512_fmadd_pd(r3_vec, a_x, _mm512_fmadd_pd(r4_vec, a_y, _mm512_mul_pd(r5_vec, a_z)));
        __m512d ra_z = _mm512_fmadd_pd(r6_vec, a_x, _mm512_fmadd_pd(r7_vec, a_y, _mm512_mul_pd(r8_vec, a_z)));

        __m512d dx = _mm512_sub_pd(ra_x, b_x), dy = _mm512_sub_pd(ra_y, b_y), dz = _mm512_sub_pd(ra_z, b_z);
        sum_sq_vec = _mm512_fmadd_pd(dx, dx, sum_sq_vec);
        sum_sq_vec = _mm512_fmadd_pd(dy, dy, sum_sq_vec);
        sum_sq_vec = _mm512_fmadd_pd(dz, dz, sum_sq_vec);
    }

    *result = SIMSIMD_SQRT((simsimd_distance_t)_mm512_reduce_add_pd(sum_sq_vec) * inv_n);
}

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_SKYLAKE
#endif // _SIMSIMD_TARGET_X86

#ifdef __cplusplus
}
#endif

#endif // SIMSIMD_MESH_H

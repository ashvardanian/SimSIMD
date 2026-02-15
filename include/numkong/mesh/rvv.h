/**
 *  @brief SIMD-accelerated Mesh Operations for RISC-V.
 *  @file include/numkong/mesh/rvv.h
 *  @author Ash Vardanian
 *  @date February 6, 2026
 *
 *  @sa include/numkong/mesh.h
 *
 *  RVV mesh operations leverage:
 *
 *  - `vlseg3e32`/`vlseg3e64`: deinterleave xyz triplets in hardware
 *  - `vfwcvt`/`vfwmacc`: widening FMA for f32→f64 accumulation
 *  - `vfredusum`: single-instruction horizontal reduction
 *  - Serial SVD/determinant from mesh/serial.h for fixed 3×3 matrix operations
 *
 *  Fused helpers minimize data passes:
 *
 *  - `nk_bicentroid_*_rvv_`: both centroids in a single pass (used by RMSD)
 *  - `nk_centroid_and_cross_covariance_*_rvv_`: centroids + H in one pass (Kabsch)
 *  - `nk_centroid_and_cross_covariance_and_variance_*_rvv_`: + variance (Umeyama)
 *
 *  Math for fused centroid+covariance:
 *    H[i][j] = Σ (a[i] - ca[i]) * (b[j] - cb[j])
 *            = Σ a[i]*b[j] - n*ca[i]*cb[j]
 *  So we accumulate raw Σ a[i]*b[j] in the loop, then fix up after.
 *
 *  Key RVV-specific optimizations (vs. scalar or x86 backends):
 *
 *  - Deferred horizontal reduction in bicentroid: per-lane `vfwadd_wv` (f32)
 *    or `vfadd_vv` (f64) accumulation across loop iterations, with a single
 *    `vfredusum` after the loop — eliminates 6 `vfredusum` per iteration.
 *  - `vfwmacc_vv` in f32 SSD: accumulates widened squared distances per-lane
 *    (dx²+dy²+dz²) before a single reduction — saves 2 `vfredusum` per iteration.
 *  - Vectorized R = V×Uᵀ via `vfmul_vf`/`vfmacc_vf`: each output row computed
 *    as a 3-element vector dot product — 15 vector ops vs 45 scalar ops.
 *  - `vfncvt_f_f_w` for f64→f32 narrowing of H matrix before SVD.
 */
#ifndef NK_MESH_RVV_H
#define NK_MESH_RVV_H

#if NK_TARGET_RISCV_
#if NK_TARGET_RVV

#include "numkong/types.h"
#include "numkong/spatial/rvv.h" // `nk_f32_sqrt_rvv`, `nk_f64_sqrt_rvv`
#include "numkong/mesh/serial.h" // `nk_svd3x3_f32_`, `nk_svd3x3_f64_`, `nk_det3x3_f32_`, `nk_det3x3_f64_`

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=+v"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=+v")
#endif

#if defined(__cplusplus)
extern "C" {
#endif

/**
 *  @brief Compute centroids of two f32 point clouds in a single pass.
 *
 *  Reads both clouds simultaneously, accumulating 6 sums (3 per cloud) in f64.
 *  Reduces RMSD from 3 passes to 2 (bicentroid + SSD).
 *  Uses per-lane `vfwadd_wv` accumulation with deferred `vfredusum` after the loop.
 */
NK_INTERNAL void nk_bicentroid_f32_rvv_(               //
    nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, //
    nk_f64_t *ca_x, nk_f64_t *ca_y, nk_f64_t *ca_z,    //
    nk_f64_t *cb_x, nk_f64_t *cb_y, nk_f64_t *cb_z) {
    nk_size_t vlmax = __riscv_vsetvlmax_e64m2();
    vfloat64m2_t sum_a_x_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, vlmax);
    vfloat64m2_t sum_a_y_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, vlmax);
    vfloat64m2_t sum_a_z_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, vlmax);
    vfloat64m2_t sum_b_x_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, vlmax);
    vfloat64m2_t sum_b_y_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, vlmax);
    vfloat64m2_t sum_b_z_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, vlmax);
    nk_f32_t const *a_ptr = a, *b_ptr = b;
    nk_size_t remaining = n;
    for (nk_size_t vl; remaining > 0; remaining -= vl, a_ptr += vl * 3, b_ptr += vl * 3) {
        vl = __riscv_vsetvl_e32m1(remaining);
        vfloat32m1x3_t a_f32m1x3 = __riscv_vlseg3e32_v_f32m1x3(a_ptr, vl);
        sum_a_x_f64m2 = __riscv_vfwadd_wv_f64m2(sum_a_x_f64m2, __riscv_vget_v_f32m1x3_f32m1(a_f32m1x3, 0), vl);
        sum_a_y_f64m2 = __riscv_vfwadd_wv_f64m2(sum_a_y_f64m2, __riscv_vget_v_f32m1x3_f32m1(a_f32m1x3, 1), vl);
        sum_a_z_f64m2 = __riscv_vfwadd_wv_f64m2(sum_a_z_f64m2, __riscv_vget_v_f32m1x3_f32m1(a_f32m1x3, 2), vl);
        vfloat32m1x3_t b_f32m1x3 = __riscv_vlseg3e32_v_f32m1x3(b_ptr, vl);
        sum_b_x_f64m2 = __riscv_vfwadd_wv_f64m2(sum_b_x_f64m2, __riscv_vget_v_f32m1x3_f32m1(b_f32m1x3, 0), vl);
        sum_b_y_f64m2 = __riscv_vfwadd_wv_f64m2(sum_b_y_f64m2, __riscv_vget_v_f32m1x3_f32m1(b_f32m1x3, 1), vl);
        sum_b_z_f64m2 = __riscv_vfwadd_wv_f64m2(sum_b_z_f64m2, __riscv_vget_v_f32m1x3_f32m1(b_f32m1x3, 2), vl);
    }
    vfloat64m1_t zero_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, 1);
    nk_f64_t inv_n = 1.0 / (nk_f64_t)n;
    *ca_x = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m2_f64m1(sum_a_x_f64m2, zero_f64m1, vlmax)) * inv_n;
    *ca_y = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m2_f64m1(sum_a_y_f64m2, zero_f64m1, vlmax)) * inv_n;
    *ca_z = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m2_f64m1(sum_a_z_f64m2, zero_f64m1, vlmax)) * inv_n;
    *cb_x = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m2_f64m1(sum_b_x_f64m2, zero_f64m1, vlmax)) * inv_n;
    *cb_y = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m2_f64m1(sum_b_y_f64m2, zero_f64m1, vlmax)) * inv_n;
    *cb_z = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m2_f64m1(sum_b_z_f64m2, zero_f64m1, vlmax)) * inv_n;
}

/**
 *  @brief Compute centroids of two f64 point clouds in a single pass.
 *  Uses per-lane `vfadd_vv` accumulation with deferred `vfredusum` after the loop.
 */
NK_INTERNAL void nk_bicentroid_f64_rvv_(               //
    nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, //
    nk_f64_t *ca_x, nk_f64_t *ca_y, nk_f64_t *ca_z,    //
    nk_f64_t *cb_x, nk_f64_t *cb_y, nk_f64_t *cb_z) {
    nk_size_t vlmax = __riscv_vsetvlmax_e64m1();
    vfloat64m1_t sum_a_x_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    vfloat64m1_t sum_a_y_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    vfloat64m1_t sum_a_z_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    vfloat64m1_t sum_b_x_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    vfloat64m1_t sum_b_y_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    vfloat64m1_t sum_b_z_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    nk_f64_t const *a_ptr = a, *b_ptr = b;
    nk_size_t remaining = n;
    for (nk_size_t vl; remaining > 0; remaining -= vl, a_ptr += vl * 3, b_ptr += vl * 3) {
        vl = __riscv_vsetvl_e64m1(remaining);
        vfloat64m1x3_t a_f64m1x3 = __riscv_vlseg3e64_v_f64m1x3(a_ptr, vl);
        sum_a_x_f64m1 = __riscv_vfadd_vv_f64m1(sum_a_x_f64m1, __riscv_vget_v_f64m1x3_f64m1(a_f64m1x3, 0), vl);
        sum_a_y_f64m1 = __riscv_vfadd_vv_f64m1(sum_a_y_f64m1, __riscv_vget_v_f64m1x3_f64m1(a_f64m1x3, 1), vl);
        sum_a_z_f64m1 = __riscv_vfadd_vv_f64m1(sum_a_z_f64m1, __riscv_vget_v_f64m1x3_f64m1(a_f64m1x3, 2), vl);
        vfloat64m1x3_t b_f64m1x3 = __riscv_vlseg3e64_v_f64m1x3(b_ptr, vl);
        sum_b_x_f64m1 = __riscv_vfadd_vv_f64m1(sum_b_x_f64m1, __riscv_vget_v_f64m1x3_f64m1(b_f64m1x3, 0), vl);
        sum_b_y_f64m1 = __riscv_vfadd_vv_f64m1(sum_b_y_f64m1, __riscv_vget_v_f64m1x3_f64m1(b_f64m1x3, 1), vl);
        sum_b_z_f64m1 = __riscv_vfadd_vv_f64m1(sum_b_z_f64m1, __riscv_vget_v_f64m1x3_f64m1(b_f64m1x3, 2), vl);
    }
    vfloat64m1_t zero_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, 1);
    nk_f64_t inv_n = 1.0 / (nk_f64_t)n;
    *ca_x = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m1_f64m1(sum_a_x_f64m1, zero_f64m1, vlmax)) * inv_n;
    *ca_y = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m1_f64m1(sum_a_y_f64m1, zero_f64m1, vlmax)) * inv_n;
    *ca_z = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m1_f64m1(sum_a_z_f64m1, zero_f64m1, vlmax)) * inv_n;
    *cb_x = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m1_f64m1(sum_b_x_f64m1, zero_f64m1, vlmax)) * inv_n;
    *cb_y = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m1_f64m1(sum_b_y_f64m1, zero_f64m1, vlmax)) * inv_n;
    *cb_z = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m1_f64m1(sum_b_z_f64m1, zero_f64m1, vlmax)) * inv_n;
}

/**
 *  @brief Compute centroids and cross-covariance matrix in a single pass (f32).
 *
 *  Accumulates raw Σ a[i]*b[j] and Σ a[i], Σ b[j] simultaneously, then:
 *    ca = Σa / n,  cb = Σb / n
 *    H[i][j] = raw[i][j] - n * ca[i] * cb[j]
 *
 *  Reduces Kabsch from 4 passes to 2 (fused centroid+covariance + SSD).
 *  Cross-products use per-lane `vfwmacc_vv` accumulation (vfloat64m2_t) with
 *  deferred `vfredusum` after the loop — eliminates 9 reductions per iteration.
 */
NK_INTERNAL void nk_centroid_and_cross_covariance_f32_rvv_( //
    nk_f32_t const *a, nk_f32_t const *b, nk_size_t n,      //
    nk_f64_t *ca_x, nk_f64_t *ca_y, nk_f64_t *ca_z,         //
    nk_f64_t *cb_x, nk_f64_t *cb_y, nk_f64_t *cb_z,         //
    nk_f64_t h[9]) {
    // 6 centroid accumulators
    vfloat64m1_t sum_a_x_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, 1), sum_a_y_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, 1);
    vfloat64m1_t sum_a_z_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, 1);
    vfloat64m1_t sum_b_x_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, 1), sum_b_y_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, 1);
    vfloat64m1_t sum_b_z_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, 1);
    // 9 raw cross-product accumulators: Σ a[i]*b[j]
    nk_size_t vlmax = __riscv_vsetvlmax_e64m2();
    vfloat64m2_t cross_00_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, vlmax), cross_01_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, vlmax);
    vfloat64m2_t cross_02_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, vlmax), cross_10_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, vlmax);
    vfloat64m2_t cross_11_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, vlmax), cross_12_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, vlmax);
    vfloat64m2_t cross_20_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, vlmax), cross_21_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, vlmax);
    vfloat64m2_t cross_22_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, vlmax);
    nk_f32_t const *a_ptr = a, *b_ptr = b;
    nk_size_t remaining = n;
    for (nk_size_t vl; remaining > 0; remaining -= vl, a_ptr += vl * 3, b_ptr += vl * 3) {
        vl = __riscv_vsetvl_e32m1(remaining);
        vfloat32m1x3_t a_f32m1x3 = __riscv_vlseg3e32_v_f32m1x3(a_ptr, vl);
        vfloat32m1_t a_x_f32m1 = __riscv_vget_v_f32m1x3_f32m1(a_f32m1x3, 0);
        vfloat32m1_t a_y_f32m1 = __riscv_vget_v_f32m1x3_f32m1(a_f32m1x3, 1);
        vfloat32m1_t a_z_f32m1 = __riscv_vget_v_f32m1x3_f32m1(a_f32m1x3, 2);
        vfloat32m1x3_t b_f32m1x3 = __riscv_vlseg3e32_v_f32m1x3(b_ptr, vl);
        vfloat32m1_t b_x_f32m1 = __riscv_vget_v_f32m1x3_f32m1(b_f32m1x3, 0);
        vfloat32m1_t b_y_f32m1 = __riscv_vget_v_f32m1x3_f32m1(b_f32m1x3, 1);
        vfloat32m1_t b_z_f32m1 = __riscv_vget_v_f32m1x3_f32m1(b_f32m1x3, 2);
        // Centroid sums (widened to f64)
        sum_a_x_f64m1 = __riscv_vfredusum_vs_f64m2_f64m1(__riscv_vfwcvt_f_f_v_f64m2(a_x_f32m1, vl), sum_a_x_f64m1, vl);
        sum_a_y_f64m1 = __riscv_vfredusum_vs_f64m2_f64m1(__riscv_vfwcvt_f_f_v_f64m2(a_y_f32m1, vl), sum_a_y_f64m1, vl);
        sum_a_z_f64m1 = __riscv_vfredusum_vs_f64m2_f64m1(__riscv_vfwcvt_f_f_v_f64m2(a_z_f32m1, vl), sum_a_z_f64m1, vl);
        sum_b_x_f64m1 = __riscv_vfredusum_vs_f64m2_f64m1(__riscv_vfwcvt_f_f_v_f64m2(b_x_f32m1, vl), sum_b_x_f64m1, vl);
        sum_b_y_f64m1 = __riscv_vfredusum_vs_f64m2_f64m1(__riscv_vfwcvt_f_f_v_f64m2(b_y_f32m1, vl), sum_b_y_f64m1, vl);
        sum_b_z_f64m1 = __riscv_vfredusum_vs_f64m2_f64m1(__riscv_vfwcvt_f_f_v_f64m2(b_z_f32m1, vl), sum_b_z_f64m1, vl);
        // Raw cross-products: Σ a[i]*b[j], widened to f64
        cross_00_f64m2 = __riscv_vfwmacc_vv_f64m2(cross_00_f64m2, a_x_f32m1, b_x_f32m1, vl);
        cross_01_f64m2 = __riscv_vfwmacc_vv_f64m2(cross_01_f64m2, a_x_f32m1, b_y_f32m1, vl);
        cross_02_f64m2 = __riscv_vfwmacc_vv_f64m2(cross_02_f64m2, a_x_f32m1, b_z_f32m1, vl);
        cross_10_f64m2 = __riscv_vfwmacc_vv_f64m2(cross_10_f64m2, a_y_f32m1, b_x_f32m1, vl);
        cross_11_f64m2 = __riscv_vfwmacc_vv_f64m2(cross_11_f64m2, a_y_f32m1, b_y_f32m1, vl);
        cross_12_f64m2 = __riscv_vfwmacc_vv_f64m2(cross_12_f64m2, a_y_f32m1, b_z_f32m1, vl);
        cross_20_f64m2 = __riscv_vfwmacc_vv_f64m2(cross_20_f64m2, a_z_f32m1, b_x_f32m1, vl);
        cross_21_f64m2 = __riscv_vfwmacc_vv_f64m2(cross_21_f64m2, a_z_f32m1, b_y_f32m1, vl);
        cross_22_f64m2 = __riscv_vfwmacc_vv_f64m2(cross_22_f64m2, a_z_f32m1, b_z_f32m1, vl);
    }
    // Compute centroids
    nk_f64_t inv_n = 1.0 / (nk_f64_t)n;
    nk_f64_t ca_x_ = __riscv_vfmv_f_s_f64m1_f64(sum_a_x_f64m1) * inv_n;
    nk_f64_t ca_y_ = __riscv_vfmv_f_s_f64m1_f64(sum_a_y_f64m1) * inv_n;
    nk_f64_t ca_z_ = __riscv_vfmv_f_s_f64m1_f64(sum_a_z_f64m1) * inv_n;
    nk_f64_t cb_x_ = __riscv_vfmv_f_s_f64m1_f64(sum_b_x_f64m1) * inv_n;
    nk_f64_t cb_y_ = __riscv_vfmv_f_s_f64m1_f64(sum_b_y_f64m1) * inv_n;
    nk_f64_t cb_z_ = __riscv_vfmv_f_s_f64m1_f64(sum_b_z_f64m1) * inv_n;
    *ca_x = ca_x_;
    *ca_y = ca_y_;
    *ca_z = ca_z_;
    *cb_x = cb_x_;
    *cb_y = cb_y_;
    *cb_z = cb_z_;
    // Fix up: H[i][j] = raw[i][j] - n * ca[i] * cb[j]
    vfloat64m1_t zero_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, 1);
    nk_f64_t n_f64 = (nk_f64_t)n;
    h[0] = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m2_f64m1(cross_00_f64m2, zero_f64m1, vlmax)) - n_f64 * ca_x_ * cb_x_;
    h[1] = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m2_f64m1(cross_01_f64m2, zero_f64m1, vlmax)) - n_f64 * ca_x_ * cb_y_;
    h[2] = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m2_f64m1(cross_02_f64m2, zero_f64m1, vlmax)) - n_f64 * ca_x_ * cb_z_;
    h[3] = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m2_f64m1(cross_10_f64m2, zero_f64m1, vlmax)) - n_f64 * ca_y_ * cb_x_;
    h[4] = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m2_f64m1(cross_11_f64m2, zero_f64m1, vlmax)) - n_f64 * ca_y_ * cb_y_;
    h[5] = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m2_f64m1(cross_12_f64m2, zero_f64m1, vlmax)) - n_f64 * ca_y_ * cb_z_;
    h[6] = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m2_f64m1(cross_20_f64m2, zero_f64m1, vlmax)) - n_f64 * ca_z_ * cb_x_;
    h[7] = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m2_f64m1(cross_21_f64m2, zero_f64m1, vlmax)) - n_f64 * ca_z_ * cb_y_;
    h[8] = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m2_f64m1(cross_22_f64m2, zero_f64m1, vlmax)) - n_f64 * ca_z_ * cb_z_;
}

/**
 *  @brief Compute centroids and cross-covariance matrix in a single pass (f64).
 *
 *  Per-lane `vfadd_vv`/`vfmacc_vv` accumulation with deferred `vfredusum` after the loop
 *  — eliminates 15 horizontal reductions per iteration.
 */
NK_INTERNAL void nk_centroid_and_cross_covariance_f64_rvv_( //
    nk_f64_t const *a, nk_f64_t const *b, nk_size_t n,      //
    nk_f64_t *ca_x, nk_f64_t *ca_y, nk_f64_t *ca_z,         //
    nk_f64_t *cb_x, nk_f64_t *cb_y, nk_f64_t *cb_z,         //
    nk_f64_t h[9]) {
    nk_size_t vlmax = __riscv_vsetvlmax_e64m1();
    vfloat64m1_t sum_a_x_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax), sum_a_y_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    vfloat64m1_t sum_a_z_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    vfloat64m1_t sum_b_x_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax), sum_b_y_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    vfloat64m1_t sum_b_z_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    vfloat64m1_t cross_00_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax), cross_01_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    vfloat64m1_t cross_02_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax), cross_10_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    vfloat64m1_t cross_11_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax), cross_12_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    vfloat64m1_t cross_20_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax), cross_21_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    vfloat64m1_t cross_22_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    nk_f64_t const *a_ptr = a, *b_ptr = b;
    nk_size_t remaining = n;
    for (nk_size_t vl; remaining > 0; remaining -= vl, a_ptr += vl * 3, b_ptr += vl * 3) {
        vl = __riscv_vsetvl_e64m1(remaining);
        vfloat64m1x3_t a_f64m1x3 = __riscv_vlseg3e64_v_f64m1x3(a_ptr, vl);
        vfloat64m1_t a_x_f64m1 = __riscv_vget_v_f64m1x3_f64m1(a_f64m1x3, 0);
        vfloat64m1_t a_y_f64m1 = __riscv_vget_v_f64m1x3_f64m1(a_f64m1x3, 1);
        vfloat64m1_t a_z_f64m1 = __riscv_vget_v_f64m1x3_f64m1(a_f64m1x3, 2);
        vfloat64m1x3_t b_f64m1x3 = __riscv_vlseg3e64_v_f64m1x3(b_ptr, vl);
        vfloat64m1_t b_x_f64m1 = __riscv_vget_v_f64m1x3_f64m1(b_f64m1x3, 0);
        vfloat64m1_t b_y_f64m1 = __riscv_vget_v_f64m1x3_f64m1(b_f64m1x3, 1);
        vfloat64m1_t b_z_f64m1 = __riscv_vget_v_f64m1x3_f64m1(b_f64m1x3, 2);
        sum_a_x_f64m1 = __riscv_vfadd_vv_f64m1(sum_a_x_f64m1, a_x_f64m1, vl);
        sum_a_y_f64m1 = __riscv_vfadd_vv_f64m1(sum_a_y_f64m1, a_y_f64m1, vl);
        sum_a_z_f64m1 = __riscv_vfadd_vv_f64m1(sum_a_z_f64m1, a_z_f64m1, vl);
        sum_b_x_f64m1 = __riscv_vfadd_vv_f64m1(sum_b_x_f64m1, b_x_f64m1, vl);
        sum_b_y_f64m1 = __riscv_vfadd_vv_f64m1(sum_b_y_f64m1, b_y_f64m1, vl);
        sum_b_z_f64m1 = __riscv_vfadd_vv_f64m1(sum_b_z_f64m1, b_z_f64m1, vl);
        cross_00_f64m1 = __riscv_vfmacc_vv_f64m1(cross_00_f64m1, a_x_f64m1, b_x_f64m1, vl);
        cross_01_f64m1 = __riscv_vfmacc_vv_f64m1(cross_01_f64m1, a_x_f64m1, b_y_f64m1, vl);
        cross_02_f64m1 = __riscv_vfmacc_vv_f64m1(cross_02_f64m1, a_x_f64m1, b_z_f64m1, vl);
        cross_10_f64m1 = __riscv_vfmacc_vv_f64m1(cross_10_f64m1, a_y_f64m1, b_x_f64m1, vl);
        cross_11_f64m1 = __riscv_vfmacc_vv_f64m1(cross_11_f64m1, a_y_f64m1, b_y_f64m1, vl);
        cross_12_f64m1 = __riscv_vfmacc_vv_f64m1(cross_12_f64m1, a_y_f64m1, b_z_f64m1, vl);
        cross_20_f64m1 = __riscv_vfmacc_vv_f64m1(cross_20_f64m1, a_z_f64m1, b_x_f64m1, vl);
        cross_21_f64m1 = __riscv_vfmacc_vv_f64m1(cross_21_f64m1, a_z_f64m1, b_y_f64m1, vl);
        cross_22_f64m1 = __riscv_vfmacc_vv_f64m1(cross_22_f64m1, a_z_f64m1, b_z_f64m1, vl);
    }
    // Compute centroids
    vfloat64m1_t zero_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, 1);
    nk_f64_t inv_n = 1.0 / (nk_f64_t)n;
    nk_f64_t ca_x_ = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m1_f64m1(sum_a_x_f64m1, zero_f64m1, vlmax)) * inv_n;
    nk_f64_t ca_y_ = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m1_f64m1(sum_a_y_f64m1, zero_f64m1, vlmax)) * inv_n;
    nk_f64_t ca_z_ = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m1_f64m1(sum_a_z_f64m1, zero_f64m1, vlmax)) * inv_n;
    nk_f64_t cb_x_ = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m1_f64m1(sum_b_x_f64m1, zero_f64m1, vlmax)) * inv_n;
    nk_f64_t cb_y_ = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m1_f64m1(sum_b_y_f64m1, zero_f64m1, vlmax)) * inv_n;
    nk_f64_t cb_z_ = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m1_f64m1(sum_b_z_f64m1, zero_f64m1, vlmax)) * inv_n;
    *ca_x = ca_x_;
    *ca_y = ca_y_;
    *ca_z = ca_z_;
    *cb_x = cb_x_;
    *cb_y = cb_y_;
    *cb_z = cb_z_;
    nk_f64_t n_f64 = (nk_f64_t)n;
    h[0] = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m1_f64m1(cross_00_f64m1, zero_f64m1, vlmax)) - n_f64 * ca_x_ * cb_x_;
    h[1] = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m1_f64m1(cross_01_f64m1, zero_f64m1, vlmax)) - n_f64 * ca_x_ * cb_y_;
    h[2] = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m1_f64m1(cross_02_f64m1, zero_f64m1, vlmax)) - n_f64 * ca_x_ * cb_z_;
    h[3] = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m1_f64m1(cross_10_f64m1, zero_f64m1, vlmax)) - n_f64 * ca_y_ * cb_x_;
    h[4] = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m1_f64m1(cross_11_f64m1, zero_f64m1, vlmax)) - n_f64 * ca_y_ * cb_y_;
    h[5] = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m1_f64m1(cross_12_f64m1, zero_f64m1, vlmax)) - n_f64 * ca_y_ * cb_z_;
    h[6] = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m1_f64m1(cross_20_f64m1, zero_f64m1, vlmax)) - n_f64 * ca_z_ * cb_x_;
    h[7] = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m1_f64m1(cross_21_f64m1, zero_f64m1, vlmax)) - n_f64 * ca_z_ * cb_y_;
    h[8] = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m1_f64m1(cross_22_f64m1, zero_f64m1, vlmax)) - n_f64 * ca_z_ * cb_z_;
}

/**
 *  @brief Compute centroids, cross-covariance, and variance_a in a single pass (f32).
 *
 *  Same as centroid_and_cross_covariance but also computes:
 *    variance_a = (1/n) * Σ ||a[i] - ca||²
 *               = (1/n) * (Σ ||a[i]||² - n * ||ca||²)
 *
 *  Cross-products use per-lane `vfwmacc_vv` accumulation (vfloat64m2_t) with
 *  deferred `vfredusum` after the loop — eliminates 9 reductions per iteration.
 */
NK_INTERNAL void nk_centroid_and_cross_covariance_and_variance_f32_rvv_( //
    nk_f32_t const *a, nk_f32_t const *b, nk_size_t n,                   //
    nk_f64_t *ca_x, nk_f64_t *ca_y, nk_f64_t *ca_z,                      //
    nk_f64_t *cb_x, nk_f64_t *cb_y, nk_f64_t *cb_z,                      //
    nk_f64_t h[9], nk_f64_t *variance_a) {
    vfloat64m1_t sum_a_x_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, 1), sum_a_y_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, 1);
    vfloat64m1_t sum_a_z_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, 1);
    vfloat64m1_t sum_b_x_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, 1), sum_b_y_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, 1);
    vfloat64m1_t sum_b_z_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, 1);
    nk_size_t vlmax = __riscv_vsetvlmax_e64m2();
    vfloat64m2_t cross_00_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, vlmax), cross_01_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, vlmax);
    vfloat64m2_t cross_02_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, vlmax), cross_10_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, vlmax);
    vfloat64m2_t cross_11_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, vlmax), cross_12_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, vlmax);
    vfloat64m2_t cross_20_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, vlmax), cross_21_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, vlmax);
    vfloat64m2_t cross_22_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, vlmax);
    vfloat64m1_t sum_norm_squared_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, 1); // Σ ||a[i]||²
    nk_f32_t const *a_ptr = a, *b_ptr = b;
    nk_size_t remaining = n;
    for (nk_size_t vl; remaining > 0; remaining -= vl, a_ptr += vl * 3, b_ptr += vl * 3) {
        vl = __riscv_vsetvl_e32m1(remaining);
        vfloat32m1x3_t a_f32m1x3 = __riscv_vlseg3e32_v_f32m1x3(a_ptr, vl);
        vfloat32m1_t a_x_f32m1 = __riscv_vget_v_f32m1x3_f32m1(a_f32m1x3, 0);
        vfloat32m1_t a_y_f32m1 = __riscv_vget_v_f32m1x3_f32m1(a_f32m1x3, 1);
        vfloat32m1_t a_z_f32m1 = __riscv_vget_v_f32m1x3_f32m1(a_f32m1x3, 2);
        vfloat32m1x3_t b_f32m1x3 = __riscv_vlseg3e32_v_f32m1x3(b_ptr, vl);
        vfloat32m1_t b_x_f32m1 = __riscv_vget_v_f32m1x3_f32m1(b_f32m1x3, 0);
        vfloat32m1_t b_y_f32m1 = __riscv_vget_v_f32m1x3_f32m1(b_f32m1x3, 1);
        vfloat32m1_t b_z_f32m1 = __riscv_vget_v_f32m1x3_f32m1(b_f32m1x3, 2);
        sum_a_x_f64m1 = __riscv_vfredusum_vs_f64m2_f64m1(__riscv_vfwcvt_f_f_v_f64m2(a_x_f32m1, vl), sum_a_x_f64m1, vl);
        sum_a_y_f64m1 = __riscv_vfredusum_vs_f64m2_f64m1(__riscv_vfwcvt_f_f_v_f64m2(a_y_f32m1, vl), sum_a_y_f64m1, vl);
        sum_a_z_f64m1 = __riscv_vfredusum_vs_f64m2_f64m1(__riscv_vfwcvt_f_f_v_f64m2(a_z_f32m1, vl), sum_a_z_f64m1, vl);
        sum_b_x_f64m1 = __riscv_vfredusum_vs_f64m2_f64m1(__riscv_vfwcvt_f_f_v_f64m2(b_x_f32m1, vl), sum_b_x_f64m1, vl);
        sum_b_y_f64m1 = __riscv_vfredusum_vs_f64m2_f64m1(__riscv_vfwcvt_f_f_v_f64m2(b_y_f32m1, vl), sum_b_y_f64m1, vl);
        sum_b_z_f64m1 = __riscv_vfredusum_vs_f64m2_f64m1(__riscv_vfwcvt_f_f_v_f64m2(b_z_f32m1, vl), sum_b_z_f64m1, vl);
        cross_00_f64m2 = __riscv_vfwmacc_vv_f64m2(cross_00_f64m2, a_x_f32m1, b_x_f32m1, vl);
        cross_01_f64m2 = __riscv_vfwmacc_vv_f64m2(cross_01_f64m2, a_x_f32m1, b_y_f32m1, vl);
        cross_02_f64m2 = __riscv_vfwmacc_vv_f64m2(cross_02_f64m2, a_x_f32m1, b_z_f32m1, vl);
        cross_10_f64m2 = __riscv_vfwmacc_vv_f64m2(cross_10_f64m2, a_y_f32m1, b_x_f32m1, vl);
        cross_11_f64m2 = __riscv_vfwmacc_vv_f64m2(cross_11_f64m2, a_y_f32m1, b_y_f32m1, vl);
        cross_12_f64m2 = __riscv_vfwmacc_vv_f64m2(cross_12_f64m2, a_y_f32m1, b_z_f32m1, vl);
        cross_20_f64m2 = __riscv_vfwmacc_vv_f64m2(cross_20_f64m2, a_z_f32m1, b_x_f32m1, vl);
        cross_21_f64m2 = __riscv_vfwmacc_vv_f64m2(cross_21_f64m2, a_z_f32m1, b_y_f32m1, vl);
        cross_22_f64m2 = __riscv_vfwmacc_vv_f64m2(cross_22_f64m2, a_z_f32m1, b_z_f32m1, vl);
        // Variance: Σ (a_x² + a_y² + a_z²) — raw, not centered
        vfloat64m2_t norm_squared_f64m2 = __riscv_vfwmul_vv_f64m2(a_x_f32m1, a_x_f32m1, vl);
        norm_squared_f64m2 = __riscv_vfwmacc_vv_f64m2(norm_squared_f64m2, a_y_f32m1, a_y_f32m1, vl);
        norm_squared_f64m2 = __riscv_vfwmacc_vv_f64m2(norm_squared_f64m2, a_z_f32m1, a_z_f32m1, vl);
        sum_norm_squared_f64m1 = __riscv_vfredusum_vs_f64m2_f64m1(norm_squared_f64m2, sum_norm_squared_f64m1, vl);
    }
    nk_f64_t inv_n = 1.0 / (nk_f64_t)n;
    nk_f64_t ca_x_ = __riscv_vfmv_f_s_f64m1_f64(sum_a_x_f64m1) * inv_n;
    nk_f64_t ca_y_ = __riscv_vfmv_f_s_f64m1_f64(sum_a_y_f64m1) * inv_n;
    nk_f64_t ca_z_ = __riscv_vfmv_f_s_f64m1_f64(sum_a_z_f64m1) * inv_n;
    nk_f64_t cb_x_ = __riscv_vfmv_f_s_f64m1_f64(sum_b_x_f64m1) * inv_n;
    nk_f64_t cb_y_ = __riscv_vfmv_f_s_f64m1_f64(sum_b_y_f64m1) * inv_n;
    nk_f64_t cb_z_ = __riscv_vfmv_f_s_f64m1_f64(sum_b_z_f64m1) * inv_n;
    *ca_x = ca_x_;
    *ca_y = ca_y_;
    *ca_z = ca_z_;
    *cb_x = cb_x_;
    *cb_y = cb_y_;
    *cb_z = cb_z_;
    vfloat64m1_t zero_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, 1);
    nk_f64_t n_f64 = (nk_f64_t)n;
    h[0] = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m2_f64m1(cross_00_f64m2, zero_f64m1, vlmax)) - n_f64 * ca_x_ * cb_x_;
    h[1] = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m2_f64m1(cross_01_f64m2, zero_f64m1, vlmax)) - n_f64 * ca_x_ * cb_y_;
    h[2] = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m2_f64m1(cross_02_f64m2, zero_f64m1, vlmax)) - n_f64 * ca_x_ * cb_z_;
    h[3] = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m2_f64m1(cross_10_f64m2, zero_f64m1, vlmax)) - n_f64 * ca_y_ * cb_x_;
    h[4] = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m2_f64m1(cross_11_f64m2, zero_f64m1, vlmax)) - n_f64 * ca_y_ * cb_y_;
    h[5] = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m2_f64m1(cross_12_f64m2, zero_f64m1, vlmax)) - n_f64 * ca_y_ * cb_z_;
    h[6] = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m2_f64m1(cross_20_f64m2, zero_f64m1, vlmax)) - n_f64 * ca_z_ * cb_x_;
    h[7] = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m2_f64m1(cross_21_f64m2, zero_f64m1, vlmax)) - n_f64 * ca_z_ * cb_y_;
    h[8] = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m2_f64m1(cross_22_f64m2, zero_f64m1, vlmax)) - n_f64 * ca_z_ * cb_z_;
    // variance_a = (1/n) * (Σ ||a[i]||² - n * ||ca||²)
    *variance_a = __riscv_vfmv_f_s_f64m1_f64(sum_norm_squared_f64m1) * inv_n -
                  (ca_x_ * ca_x_ + ca_y_ * ca_y_ + ca_z_ * ca_z_);
}

/**
 *  @brief Compute centroids, cross-covariance, and variance_a in a single pass (f64).
 *
 *  Per-lane `vfadd_vv`/`vfmacc_vv` accumulation with deferred `vfredusum` after the loop
 *  — eliminates 16 horizontal reductions per iteration.
 */
NK_INTERNAL void nk_centroid_and_cross_covariance_and_variance_f64_rvv_( //
    nk_f64_t const *a, nk_f64_t const *b, nk_size_t n,                   //
    nk_f64_t *ca_x, nk_f64_t *ca_y, nk_f64_t *ca_z,                      //
    nk_f64_t *cb_x, nk_f64_t *cb_y, nk_f64_t *cb_z,                      //
    nk_f64_t h[9], nk_f64_t *variance_a) {
    nk_size_t vlmax = __riscv_vsetvlmax_e64m1();
    vfloat64m1_t sum_a_x_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax), sum_a_y_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    vfloat64m1_t sum_a_z_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    vfloat64m1_t sum_b_x_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax), sum_b_y_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    vfloat64m1_t sum_b_z_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    vfloat64m1_t cross_00_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax), cross_01_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    vfloat64m1_t cross_02_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax), cross_10_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    vfloat64m1_t cross_11_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax), cross_12_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    vfloat64m1_t cross_20_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax), cross_21_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    vfloat64m1_t cross_22_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    vfloat64m1_t sum_norm_squared_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    nk_f64_t const *a_ptr = a, *b_ptr = b;
    nk_size_t remaining = n;
    for (nk_size_t vl; remaining > 0; remaining -= vl, a_ptr += vl * 3, b_ptr += vl * 3) {
        vl = __riscv_vsetvl_e64m1(remaining);
        vfloat64m1x3_t a_f64m1x3 = __riscv_vlseg3e64_v_f64m1x3(a_ptr, vl);
        vfloat64m1_t a_x_f64m1 = __riscv_vget_v_f64m1x3_f64m1(a_f64m1x3, 0);
        vfloat64m1_t a_y_f64m1 = __riscv_vget_v_f64m1x3_f64m1(a_f64m1x3, 1);
        vfloat64m1_t a_z_f64m1 = __riscv_vget_v_f64m1x3_f64m1(a_f64m1x3, 2);
        vfloat64m1x3_t b_f64m1x3 = __riscv_vlseg3e64_v_f64m1x3(b_ptr, vl);
        vfloat64m1_t b_x_f64m1 = __riscv_vget_v_f64m1x3_f64m1(b_f64m1x3, 0);
        vfloat64m1_t b_y_f64m1 = __riscv_vget_v_f64m1x3_f64m1(b_f64m1x3, 1);
        vfloat64m1_t b_z_f64m1 = __riscv_vget_v_f64m1x3_f64m1(b_f64m1x3, 2);
        sum_a_x_f64m1 = __riscv_vfadd_vv_f64m1(sum_a_x_f64m1, a_x_f64m1, vl);
        sum_a_y_f64m1 = __riscv_vfadd_vv_f64m1(sum_a_y_f64m1, a_y_f64m1, vl);
        sum_a_z_f64m1 = __riscv_vfadd_vv_f64m1(sum_a_z_f64m1, a_z_f64m1, vl);
        sum_b_x_f64m1 = __riscv_vfadd_vv_f64m1(sum_b_x_f64m1, b_x_f64m1, vl);
        sum_b_y_f64m1 = __riscv_vfadd_vv_f64m1(sum_b_y_f64m1, b_y_f64m1, vl);
        sum_b_z_f64m1 = __riscv_vfadd_vv_f64m1(sum_b_z_f64m1, b_z_f64m1, vl);
        cross_00_f64m1 = __riscv_vfmacc_vv_f64m1(cross_00_f64m1, a_x_f64m1, b_x_f64m1, vl);
        cross_01_f64m1 = __riscv_vfmacc_vv_f64m1(cross_01_f64m1, a_x_f64m1, b_y_f64m1, vl);
        cross_02_f64m1 = __riscv_vfmacc_vv_f64m1(cross_02_f64m1, a_x_f64m1, b_z_f64m1, vl);
        cross_10_f64m1 = __riscv_vfmacc_vv_f64m1(cross_10_f64m1, a_y_f64m1, b_x_f64m1, vl);
        cross_11_f64m1 = __riscv_vfmacc_vv_f64m1(cross_11_f64m1, a_y_f64m1, b_y_f64m1, vl);
        cross_12_f64m1 = __riscv_vfmacc_vv_f64m1(cross_12_f64m1, a_y_f64m1, b_z_f64m1, vl);
        cross_20_f64m1 = __riscv_vfmacc_vv_f64m1(cross_20_f64m1, a_z_f64m1, b_x_f64m1, vl);
        cross_21_f64m1 = __riscv_vfmacc_vv_f64m1(cross_21_f64m1, a_z_f64m1, b_y_f64m1, vl);
        cross_22_f64m1 = __riscv_vfmacc_vv_f64m1(cross_22_f64m1, a_z_f64m1, b_z_f64m1, vl);
        vfloat64m1_t norm_squared_f64m1 = __riscv_vfmul_vv_f64m1(a_x_f64m1, a_x_f64m1, vl);
        norm_squared_f64m1 = __riscv_vfmacc_vv_f64m1(norm_squared_f64m1, a_y_f64m1, a_y_f64m1, vl);
        norm_squared_f64m1 = __riscv_vfmacc_vv_f64m1(norm_squared_f64m1, a_z_f64m1, a_z_f64m1, vl);
        sum_norm_squared_f64m1 = __riscv_vfadd_vv_f64m1(sum_norm_squared_f64m1, norm_squared_f64m1, vl);
    }
    vfloat64m1_t zero_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, 1);
    nk_f64_t inv_n = 1.0 / (nk_f64_t)n;
    nk_f64_t ca_x_ = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m1_f64m1(sum_a_x_f64m1, zero_f64m1, vlmax)) * inv_n;
    nk_f64_t ca_y_ = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m1_f64m1(sum_a_y_f64m1, zero_f64m1, vlmax)) * inv_n;
    nk_f64_t ca_z_ = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m1_f64m1(sum_a_z_f64m1, zero_f64m1, vlmax)) * inv_n;
    nk_f64_t cb_x_ = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m1_f64m1(sum_b_x_f64m1, zero_f64m1, vlmax)) * inv_n;
    nk_f64_t cb_y_ = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m1_f64m1(sum_b_y_f64m1, zero_f64m1, vlmax)) * inv_n;
    nk_f64_t cb_z_ = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m1_f64m1(sum_b_z_f64m1, zero_f64m1, vlmax)) * inv_n;
    *ca_x = ca_x_;
    *ca_y = ca_y_;
    *ca_z = ca_z_;
    *cb_x = cb_x_;
    *cb_y = cb_y_;
    *cb_z = cb_z_;
    nk_f64_t n_f64 = (nk_f64_t)n;
    h[0] = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m1_f64m1(cross_00_f64m1, zero_f64m1, vlmax)) - n_f64 * ca_x_ * cb_x_;
    h[1] = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m1_f64m1(cross_01_f64m1, zero_f64m1, vlmax)) - n_f64 * ca_x_ * cb_y_;
    h[2] = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m1_f64m1(cross_02_f64m1, zero_f64m1, vlmax)) - n_f64 * ca_x_ * cb_z_;
    h[3] = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m1_f64m1(cross_10_f64m1, zero_f64m1, vlmax)) - n_f64 * ca_y_ * cb_x_;
    h[4] = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m1_f64m1(cross_11_f64m1, zero_f64m1, vlmax)) - n_f64 * ca_y_ * cb_y_;
    h[5] = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m1_f64m1(cross_12_f64m1, zero_f64m1, vlmax)) - n_f64 * ca_y_ * cb_z_;
    h[6] = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m1_f64m1(cross_20_f64m1, zero_f64m1, vlmax)) - n_f64 * ca_z_ * cb_x_;
    h[7] = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m1_f64m1(cross_21_f64m1, zero_f64m1, vlmax)) - n_f64 * ca_z_ * cb_y_;
    h[8] = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m1_f64m1(cross_22_f64m1, zero_f64m1, vlmax)) - n_f64 * ca_z_ * cb_z_;
    *variance_a = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m1_f64m1(sum_norm_squared_f64m1, zero_f64m1, vlmax)) * inv_n -
                  (ca_x_ * ca_x_ + ca_y_ * ca_y_ + ca_z_ * ca_z_);
}

NK_INTERNAL nk_f64_t nk_transformed_ssd_f32_rvv_(      //
    nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, //
    nk_f32_t const *r, nk_f32_t scale,                 //
    nk_f32_t ca_x, nk_f32_t ca_y, nk_f32_t ca_z,       //
    nk_f32_t cb_x, nk_f32_t cb_y, nk_f32_t cb_z) {
    nk_f32_t sr0 = scale * r[0], sr1 = scale * r[1], sr2 = scale * r[2];
    nk_f32_t sr3 = scale * r[3], sr4 = scale * r[4], sr5 = scale * r[5];
    nk_f32_t sr6 = scale * r[6], sr7 = scale * r[7], sr8 = scale * r[8];
    vfloat64m1_t sum_distance_squared_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, 1);
    nk_f32_t const *a_ptr = a, *b_ptr = b;
    nk_size_t remaining = n;
    for (nk_size_t vl; remaining > 0; remaining -= vl, a_ptr += vl * 3, b_ptr += vl * 3) {
        vl = __riscv_vsetvl_e32m1(remaining);
        vfloat32m1x3_t a_f32m1x3 = __riscv_vlseg3e32_v_f32m1x3(a_ptr, vl);
        vfloat32m1_t centered_a_x_f32m1 = __riscv_vfsub_vf_f32m1(__riscv_vget_v_f32m1x3_f32m1(a_f32m1x3, 0), ca_x, vl);
        vfloat32m1_t centered_a_y_f32m1 = __riscv_vfsub_vf_f32m1(__riscv_vget_v_f32m1x3_f32m1(a_f32m1x3, 1), ca_y, vl);
        vfloat32m1_t centered_a_z_f32m1 = __riscv_vfsub_vf_f32m1(__riscv_vget_v_f32m1x3_f32m1(a_f32m1x3, 2), ca_z, vl);
        vfloat32m1_t rotated_a_x_f32m1 = __riscv_vfmul_vf_f32m1(centered_a_x_f32m1, sr0, vl);
        rotated_a_x_f32m1 = __riscv_vfmacc_vf_f32m1(rotated_a_x_f32m1, sr1, centered_a_y_f32m1, vl);
        rotated_a_x_f32m1 = __riscv_vfmacc_vf_f32m1(rotated_a_x_f32m1, sr2, centered_a_z_f32m1, vl);
        vfloat32m1_t rotated_a_y_f32m1 = __riscv_vfmul_vf_f32m1(centered_a_x_f32m1, sr3, vl);
        rotated_a_y_f32m1 = __riscv_vfmacc_vf_f32m1(rotated_a_y_f32m1, sr4, centered_a_y_f32m1, vl);
        rotated_a_y_f32m1 = __riscv_vfmacc_vf_f32m1(rotated_a_y_f32m1, sr5, centered_a_z_f32m1, vl);
        vfloat32m1_t rotated_a_z_f32m1 = __riscv_vfmul_vf_f32m1(centered_a_x_f32m1, sr6, vl);
        rotated_a_z_f32m1 = __riscv_vfmacc_vf_f32m1(rotated_a_z_f32m1, sr7, centered_a_y_f32m1, vl);
        rotated_a_z_f32m1 = __riscv_vfmacc_vf_f32m1(rotated_a_z_f32m1, sr8, centered_a_z_f32m1, vl);
        vfloat32m1x3_t b_f32m1x3 = __riscv_vlseg3e32_v_f32m1x3(b_ptr, vl);
        vfloat32m1_t centered_b_x_f32m1 = __riscv_vfsub_vf_f32m1(__riscv_vget_v_f32m1x3_f32m1(b_f32m1x3, 0), cb_x, vl);
        vfloat32m1_t centered_b_y_f32m1 = __riscv_vfsub_vf_f32m1(__riscv_vget_v_f32m1x3_f32m1(b_f32m1x3, 1), cb_y, vl);
        vfloat32m1_t centered_b_z_f32m1 = __riscv_vfsub_vf_f32m1(__riscv_vget_v_f32m1x3_f32m1(b_f32m1x3, 2), cb_z, vl);
        vfloat32m1_t delta_x_f32m1 = __riscv_vfsub_vv_f32m1(rotated_a_x_f32m1, centered_b_x_f32m1, vl);
        vfloat32m1_t delta_y_f32m1 = __riscv_vfsub_vv_f32m1(rotated_a_y_f32m1, centered_b_y_f32m1, vl);
        vfloat32m1_t delta_z_f32m1 = __riscv_vfsub_vv_f32m1(rotated_a_z_f32m1, centered_b_z_f32m1, vl);
        vfloat64m2_t distance_squared_f64m2 = __riscv_vfwmul_vv_f64m2(delta_x_f32m1, delta_x_f32m1, vl);
        distance_squared_f64m2 = __riscv_vfwmacc_vv_f64m2(distance_squared_f64m2, delta_y_f32m1, delta_y_f32m1, vl);
        distance_squared_f64m2 = __riscv_vfwmacc_vv_f64m2(distance_squared_f64m2, delta_z_f32m1, delta_z_f32m1, vl);
        sum_distance_squared_f64m1 = __riscv_vfredusum_vs_f64m2_f64m1(distance_squared_f64m2,
                                                                      sum_distance_squared_f64m1, vl);
    }
    return __riscv_vfmv_f_s_f64m1_f64(sum_distance_squared_f64m1);
}

NK_INTERNAL nk_f64_t nk_transformed_ssd_f64_rvv_(      //
    nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, //
    nk_f64_t const *r, nk_f64_t scale,                 //
    nk_f64_t ca_x, nk_f64_t ca_y, nk_f64_t ca_z,       //
    nk_f64_t cb_x, nk_f64_t cb_y, nk_f64_t cb_z) {
    nk_f64_t sr0 = scale * r[0], sr1 = scale * r[1], sr2 = scale * r[2];
    nk_f64_t sr3 = scale * r[3], sr4 = scale * r[4], sr5 = scale * r[5];
    nk_f64_t sr6 = scale * r[6], sr7 = scale * r[7], sr8 = scale * r[8];
    vfloat64m1_t sum_distance_squared_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, 1);
    nk_f64_t const *a_ptr = a, *b_ptr = b;
    nk_size_t remaining = n;
    for (nk_size_t vl; remaining > 0; remaining -= vl, a_ptr += vl * 3, b_ptr += vl * 3) {
        vl = __riscv_vsetvl_e64m1(remaining);
        vfloat64m1x3_t a_f64m1x3 = __riscv_vlseg3e64_v_f64m1x3(a_ptr, vl);
        vfloat64m1_t centered_a_x_f64m1 = __riscv_vfsub_vf_f64m1(__riscv_vget_v_f64m1x3_f64m1(a_f64m1x3, 0), ca_x, vl);
        vfloat64m1_t centered_a_y_f64m1 = __riscv_vfsub_vf_f64m1(__riscv_vget_v_f64m1x3_f64m1(a_f64m1x3, 1), ca_y, vl);
        vfloat64m1_t centered_a_z_f64m1 = __riscv_vfsub_vf_f64m1(__riscv_vget_v_f64m1x3_f64m1(a_f64m1x3, 2), ca_z, vl);
        vfloat64m1_t rotated_a_x_f64m1 = __riscv_vfmul_vf_f64m1(centered_a_x_f64m1, sr0, vl);
        rotated_a_x_f64m1 = __riscv_vfmacc_vf_f64m1(rotated_a_x_f64m1, sr1, centered_a_y_f64m1, vl);
        rotated_a_x_f64m1 = __riscv_vfmacc_vf_f64m1(rotated_a_x_f64m1, sr2, centered_a_z_f64m1, vl);
        vfloat64m1_t rotated_a_y_f64m1 = __riscv_vfmul_vf_f64m1(centered_a_x_f64m1, sr3, vl);
        rotated_a_y_f64m1 = __riscv_vfmacc_vf_f64m1(rotated_a_y_f64m1, sr4, centered_a_y_f64m1, vl);
        rotated_a_y_f64m1 = __riscv_vfmacc_vf_f64m1(rotated_a_y_f64m1, sr5, centered_a_z_f64m1, vl);
        vfloat64m1_t rotated_a_z_f64m1 = __riscv_vfmul_vf_f64m1(centered_a_x_f64m1, sr6, vl);
        rotated_a_z_f64m1 = __riscv_vfmacc_vf_f64m1(rotated_a_z_f64m1, sr7, centered_a_y_f64m1, vl);
        rotated_a_z_f64m1 = __riscv_vfmacc_vf_f64m1(rotated_a_z_f64m1, sr8, centered_a_z_f64m1, vl);
        vfloat64m1x3_t b_f64m1x3 = __riscv_vlseg3e64_v_f64m1x3(b_ptr, vl);
        vfloat64m1_t centered_b_x_f64m1 = __riscv_vfsub_vf_f64m1(__riscv_vget_v_f64m1x3_f64m1(b_f64m1x3, 0), cb_x, vl);
        vfloat64m1_t centered_b_y_f64m1 = __riscv_vfsub_vf_f64m1(__riscv_vget_v_f64m1x3_f64m1(b_f64m1x3, 1), cb_y, vl);
        vfloat64m1_t centered_b_z_f64m1 = __riscv_vfsub_vf_f64m1(__riscv_vget_v_f64m1x3_f64m1(b_f64m1x3, 2), cb_z, vl);
        vfloat64m1_t delta_x_f64m1 = __riscv_vfsub_vv_f64m1(rotated_a_x_f64m1, centered_b_x_f64m1, vl);
        vfloat64m1_t delta_y_f64m1 = __riscv_vfsub_vv_f64m1(rotated_a_y_f64m1, centered_b_y_f64m1, vl);
        vfloat64m1_t delta_z_f64m1 = __riscv_vfsub_vv_f64m1(rotated_a_z_f64m1, centered_b_z_f64m1, vl);
        vfloat64m1_t distance_squared_f64m1 = __riscv_vfmul_vv_f64m1(delta_x_f64m1, delta_x_f64m1, vl);
        distance_squared_f64m1 = __riscv_vfmacc_vv_f64m1(distance_squared_f64m1, delta_y_f64m1, delta_y_f64m1, vl);
        distance_squared_f64m1 = __riscv_vfmacc_vv_f64m1(distance_squared_f64m1, delta_z_f64m1, delta_z_f64m1, vl);
        sum_distance_squared_f64m1 = __riscv_vfredusum_vs_f64m1_f64m1(distance_squared_f64m1,
                                                                      sum_distance_squared_f64m1, vl);
    }
    return __riscv_vfmv_f_s_f64m1_f64(sum_distance_squared_f64m1);
}

/** @brief Compute R = V * Uᵀ from SVD factors (f32), vectorized with `vfmul_vf`/`vfmacc_vf`. */
NK_INTERNAL void nk_rotation_from_svd_f32_rvv_( //
    nk_f32_t *svd_u, nk_f32_t *svd_v, nk_f32_t r[9]) {
    nk_size_t vl3 = __riscv_vsetvl_e32m1(3);
    vfloat32m1_t u_row0_f32m1 = __riscv_vle32_v_f32m1(svd_u + 0, vl3);
    vfloat32m1_t u_row1_f32m1 = __riscv_vle32_v_f32m1(svd_u + 3, vl3);
    vfloat32m1_t u_row2_f32m1 = __riscv_vle32_v_f32m1(svd_u + 6, vl3);
    // Row 0: R[0..2] = V[0]*U_row0 + V[1]*U_row1 + V[2]*U_row2
    vfloat32m1_t rotation_row_f32m1 = __riscv_vfmul_vf_f32m1(u_row0_f32m1, svd_v[0], vl3);
    rotation_row_f32m1 = __riscv_vfmacc_vf_f32m1(rotation_row_f32m1, svd_v[1], u_row1_f32m1, vl3);
    rotation_row_f32m1 = __riscv_vfmacc_vf_f32m1(rotation_row_f32m1, svd_v[2], u_row2_f32m1, vl3);
    __riscv_vse32_v_f32m1(r + 0, rotation_row_f32m1, vl3);
    // Row 1: R[3..5]
    rotation_row_f32m1 = __riscv_vfmul_vf_f32m1(u_row0_f32m1, svd_v[3], vl3);
    rotation_row_f32m1 = __riscv_vfmacc_vf_f32m1(rotation_row_f32m1, svd_v[4], u_row1_f32m1, vl3);
    rotation_row_f32m1 = __riscv_vfmacc_vf_f32m1(rotation_row_f32m1, svd_v[5], u_row2_f32m1, vl3);
    __riscv_vse32_v_f32m1(r + 3, rotation_row_f32m1, vl3);
    // Row 2: R[6..8]
    rotation_row_f32m1 = __riscv_vfmul_vf_f32m1(u_row0_f32m1, svd_v[6], vl3);
    rotation_row_f32m1 = __riscv_vfmacc_vf_f32m1(rotation_row_f32m1, svd_v[7], u_row1_f32m1, vl3);
    rotation_row_f32m1 = __riscv_vfmacc_vf_f32m1(rotation_row_f32m1, svd_v[8], u_row2_f32m1, vl3);
    __riscv_vse32_v_f32m1(r + 6, rotation_row_f32m1, vl3);
}

/** @brief Compute R = V * Uᵀ from SVD factors (f64), vectorized with `vfmul_vf`/`vfmacc_vf`. */
NK_INTERNAL void nk_rotation_from_svd_f64_rvv_( //
    nk_f64_t *svd_u, nk_f64_t *svd_v, nk_f64_t r[9]) {
    nk_size_t vl3 = __riscv_vsetvl_e64m1(3);
    vfloat64m1_t u_row0_f64m1 = __riscv_vle64_v_f64m1(svd_u + 0, vl3);
    vfloat64m1_t u_row1_f64m1 = __riscv_vle64_v_f64m1(svd_u + 3, vl3);
    vfloat64m1_t u_row2_f64m1 = __riscv_vle64_v_f64m1(svd_u + 6, vl3);
    // Row 0: R[0..2] = V[0]*U_row0 + V[1]*U_row1 + V[2]*U_row2
    vfloat64m1_t rotation_row_f64m1 = __riscv_vfmul_vf_f64m1(u_row0_f64m1, svd_v[0], vl3);
    rotation_row_f64m1 = __riscv_vfmacc_vf_f64m1(rotation_row_f64m1, svd_v[1], u_row1_f64m1, vl3);
    rotation_row_f64m1 = __riscv_vfmacc_vf_f64m1(rotation_row_f64m1, svd_v[2], u_row2_f64m1, vl3);
    __riscv_vse64_v_f64m1(r + 0, rotation_row_f64m1, vl3);
    // Row 1: R[3..5]
    rotation_row_f64m1 = __riscv_vfmul_vf_f64m1(u_row0_f64m1, svd_v[3], vl3);
    rotation_row_f64m1 = __riscv_vfmacc_vf_f64m1(rotation_row_f64m1, svd_v[4], u_row1_f64m1, vl3);
    rotation_row_f64m1 = __riscv_vfmacc_vf_f64m1(rotation_row_f64m1, svd_v[5], u_row2_f64m1, vl3);
    __riscv_vse64_v_f64m1(r + 3, rotation_row_f64m1, vl3);
    // Row 2: R[6..8]
    rotation_row_f64m1 = __riscv_vfmul_vf_f64m1(u_row0_f64m1, svd_v[6], vl3);
    rotation_row_f64m1 = __riscv_vfmacc_vf_f64m1(rotation_row_f64m1, svd_v[7], u_row1_f64m1, vl3);
    rotation_row_f64m1 = __riscv_vfmacc_vf_f64m1(rotation_row_f64m1, svd_v[8], u_row2_f64m1, vl3);
    __riscv_vse64_v_f64m1(r + 6, rotation_row_f64m1, vl3);
}

NK_PUBLIC void nk_rmsd_f32_rvv(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                               nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result) {
    nk_f32_t identity[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    if (rotation)
        for (int j = 0; j < 9; ++j) rotation[j] = identity[j];
    if (scale) *scale = 1.0f;
    nk_f64_t ca_x, ca_y, ca_z, cb_x, cb_y, cb_z;
    nk_bicentroid_f32_rvv_(a, b, n, &ca_x, &ca_y, &ca_z, &cb_x, &cb_y, &cb_z);
    if (a_centroid) a_centroid[0] = (nk_f32_t)ca_x, a_centroid[1] = (nk_f32_t)ca_y, a_centroid[2] = (nk_f32_t)ca_z;
    if (b_centroid) b_centroid[0] = (nk_f32_t)cb_x, b_centroid[1] = (nk_f32_t)cb_y, b_centroid[2] = (nk_f32_t)cb_z;
    nk_f64_t ssd = nk_transformed_ssd_f32_rvv_(a, b, n, identity, 1.0f, (nk_f32_t)ca_x, (nk_f32_t)ca_y, (nk_f32_t)ca_z,
                                               (nk_f32_t)cb_x, (nk_f32_t)cb_y, (nk_f32_t)cb_z);
    *result = nk_f32_sqrt_rvv((nk_f32_t)(ssd / (nk_f64_t)n));
}

NK_PUBLIC void nk_rmsd_f64_rvv(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                               nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result) {
    nk_f64_t identity[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    if (rotation)
        for (int j = 0; j < 9; ++j) rotation[j] = identity[j];
    if (scale) *scale = 1.0;
    nk_f64_t ca_x, ca_y, ca_z, cb_x, cb_y, cb_z;
    nk_bicentroid_f64_rvv_(a, b, n, &ca_x, &ca_y, &ca_z, &cb_x, &cb_y, &cb_z);
    if (a_centroid) a_centroid[0] = ca_x, a_centroid[1] = ca_y, a_centroid[2] = ca_z;
    if (b_centroid) b_centroid[0] = cb_x, b_centroid[1] = cb_y, b_centroid[2] = cb_z;
    nk_f64_t ssd = nk_transformed_ssd_f64_rvv_(a, b, n, identity, 1.0, ca_x, ca_y, ca_z, cb_x, cb_y, cb_z);
    *result = nk_f64_sqrt_rvv(ssd / (nk_f64_t)n);
}

NK_PUBLIC void nk_kabsch_f32_rvv(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                 nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result) {
    if (scale) *scale = 1.0f;
    nk_f64_t ca_x, ca_y, ca_z, cb_x, cb_y, cb_z;
    nk_f64_t h[9];
    nk_centroid_and_cross_covariance_f32_rvv_(a, b, n, &ca_x, &ca_y, &ca_z, &cb_x, &cb_y, &cb_z, h);
    if (a_centroid) a_centroid[0] = (nk_f32_t)ca_x, a_centroid[1] = (nk_f32_t)ca_y, a_centroid[2] = (nk_f32_t)ca_z;
    if (b_centroid) b_centroid[0] = (nk_f32_t)cb_x, b_centroid[1] = (nk_f32_t)cb_y, b_centroid[2] = (nk_f32_t)cb_z;
    nk_f32_t hf[9];
    for (nk_size_t i = 0, vl; i < 9; i += vl) {
        vl = __riscv_vsetvl_e64m2(9 - i);
        vfloat64m2_t covariance_f64m2 = __riscv_vle64_v_f64m2(h + i, vl);
        __riscv_vse32_v_f32m1(hf + i, __riscv_vfncvt_f_f_w_f32m1(covariance_f64m2, vl), vl);
    }
    nk_f32_t svd_u[9], svd_s[9], svd_v[9];
    nk_svd3x3_f32_(hf, svd_u, svd_s, svd_v);
    nk_f32_t r[9];
    nk_rotation_from_svd_f32_rvv_(svd_u, svd_v, r);
    if (nk_det3x3_f32_(r) < 0) {
        svd_v[2] = -svd_v[2], svd_v[5] = -svd_v[5], svd_v[8] = -svd_v[8];
        nk_rotation_from_svd_f32_rvv_(svd_u, svd_v, r);
    }
    if (rotation)
        for (int j = 0; j < 9; ++j) rotation[j] = r[j];
    nk_f64_t ssd = nk_transformed_ssd_f32_rvv_(a, b, n, r, 1.0f, (nk_f32_t)ca_x, (nk_f32_t)ca_y, (nk_f32_t)ca_z,
                                               (nk_f32_t)cb_x, (nk_f32_t)cb_y, (nk_f32_t)cb_z);
    *result = nk_f32_sqrt_rvv((nk_f32_t)(ssd / (nk_f64_t)n));
}

NK_PUBLIC void nk_kabsch_f64_rvv(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                                 nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result) {
    if (scale) *scale = 1.0;
    nk_f64_t ca_x, ca_y, ca_z, cb_x, cb_y, cb_z;
    nk_f64_t h[9];
    nk_centroid_and_cross_covariance_f64_rvv_(a, b, n, &ca_x, &ca_y, &ca_z, &cb_x, &cb_y, &cb_z, h);
    if (a_centroid) a_centroid[0] = ca_x, a_centroid[1] = ca_y, a_centroid[2] = ca_z;
    if (b_centroid) b_centroid[0] = cb_x, b_centroid[1] = cb_y, b_centroid[2] = cb_z;
    nk_f64_t svd_u[9], svd_s[9], svd_v[9];
    nk_svd3x3_f64_(h, svd_u, svd_s, svd_v);
    nk_f64_t r[9];
    nk_rotation_from_svd_f64_rvv_(svd_u, svd_v, r);
    if (nk_det3x3_f64_(r) < 0) {
        svd_v[2] = -svd_v[2], svd_v[5] = -svd_v[5], svd_v[8] = -svd_v[8];
        nk_rotation_from_svd_f64_rvv_(svd_u, svd_v, r);
    }
    if (rotation)
        for (int j = 0; j < 9; ++j) rotation[j] = r[j];
    nk_f64_t ssd = nk_transformed_ssd_f64_rvv_(a, b, n, r, 1.0, ca_x, ca_y, ca_z, cb_x, cb_y, cb_z);
    *result = nk_f64_sqrt_rvv(ssd / (nk_f64_t)n);
}

NK_PUBLIC void nk_umeyama_f32_rvv(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                  nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result) {
    nk_f64_t ca_x, ca_y, ca_z, cb_x, cb_y, cb_z;
    nk_f64_t h[9], variance_a;
    nk_centroid_and_cross_covariance_and_variance_f32_rvv_(a, b, n, &ca_x, &ca_y, &ca_z, &cb_x, &cb_y, &cb_z, h,
                                                           &variance_a);
    if (a_centroid) a_centroid[0] = (nk_f32_t)ca_x, a_centroid[1] = (nk_f32_t)ca_y, a_centroid[2] = (nk_f32_t)ca_z;
    if (b_centroid) b_centroid[0] = (nk_f32_t)cb_x, b_centroid[1] = (nk_f32_t)cb_y, b_centroid[2] = (nk_f32_t)cb_z;
    nk_f32_t hf[9];
    for (nk_size_t i = 0, vl; i < 9; i += vl) {
        vl = __riscv_vsetvl_e64m2(9 - i);
        vfloat64m2_t covariance_f64m2 = __riscv_vle64_v_f64m2(h + i, vl);
        __riscv_vse32_v_f32m1(hf + i, __riscv_vfncvt_f_f_w_f32m1(covariance_f64m2, vl), vl);
    }
    nk_f32_t svd_u[9], svd_s[9], svd_v[9];
    nk_svd3x3_f32_(hf, svd_u, svd_s, svd_v);
    nk_f32_t r[9];
    nk_rotation_from_svd_f32_rvv_(svd_u, svd_v, r);
    nk_f32_t det = nk_det3x3_f32_(r);
    nk_f32_t sign_det = det < 0 ? -1.0f : 1.0f;
    nk_f32_t trace_ds = svd_s[0] + svd_s[4] + sign_det * svd_s[8];
    nk_f32_t scale_factor = trace_ds / ((nk_f32_t)n * (nk_f32_t)variance_a);
    if (scale) *scale = scale_factor;
    if (det < 0) {
        svd_v[2] = -svd_v[2], svd_v[5] = -svd_v[5], svd_v[8] = -svd_v[8];
        nk_rotation_from_svd_f32_rvv_(svd_u, svd_v, r);
    }
    if (rotation)
        for (int j = 0; j < 9; ++j) rotation[j] = r[j];
    nk_f64_t ssd = nk_transformed_ssd_f32_rvv_(a, b, n, r, scale_factor, (nk_f32_t)ca_x, (nk_f32_t)ca_y, (nk_f32_t)ca_z,
                                               (nk_f32_t)cb_x, (nk_f32_t)cb_y, (nk_f32_t)cb_z);
    *result = nk_f32_sqrt_rvv((nk_f32_t)(ssd / (nk_f64_t)n));
}

NK_PUBLIC void nk_umeyama_f64_rvv(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                                  nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result) {
    nk_f64_t ca_x, ca_y, ca_z, cb_x, cb_y, cb_z;
    nk_f64_t h[9], variance_a;
    nk_centroid_and_cross_covariance_and_variance_f64_rvv_(a, b, n, &ca_x, &ca_y, &ca_z, &cb_x, &cb_y, &cb_z, h,
                                                           &variance_a);
    if (a_centroid) a_centroid[0] = ca_x, a_centroid[1] = ca_y, a_centroid[2] = ca_z;
    if (b_centroid) b_centroid[0] = cb_x, b_centroid[1] = cb_y, b_centroid[2] = cb_z;
    nk_f64_t svd_u[9], svd_s[9], svd_v[9];
    nk_svd3x3_f64_(h, svd_u, svd_s, svd_v);
    nk_f64_t r[9];
    nk_rotation_from_svd_f64_rvv_(svd_u, svd_v, r);
    nk_f64_t det = nk_det3x3_f64_(r);
    nk_f64_t sign_det = det < 0 ? -1.0 : 1.0;
    nk_f64_t trace_ds = svd_s[0] + svd_s[4] + sign_det * svd_s[8];
    nk_f64_t scale_factor = trace_ds / ((nk_f64_t)n * variance_a);
    if (scale) *scale = scale_factor;
    if (det < 0) {
        svd_v[2] = -svd_v[2], svd_v[5] = -svd_v[5], svd_v[8] = -svd_v[8];
        nk_rotation_from_svd_f64_rvv_(svd_u, svd_v, r);
    }
    if (rotation)
        for (int j = 0; j < 9; ++j) rotation[j] = r[j];
    nk_f64_t ssd = nk_transformed_ssd_f64_rvv_(a, b, n, r, scale_factor, ca_x, ca_y, ca_z, cb_x, cb_y, cb_z);
    *result = nk_f64_sqrt_rvv(ssd / (nk_f64_t)n);
}

NK_PUBLIC void nk_rmsd_f16_rvv(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                               nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result) {
    nk_rmsd_f16_serial(a, b, n, a_centroid, b_centroid, rotation, scale, result);
}

NK_PUBLIC void nk_kabsch_f16_rvv(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                 nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result) {
    nk_kabsch_f16_serial(a, b, n, a_centroid, b_centroid, rotation, scale, result);
}

NK_PUBLIC void nk_umeyama_f16_rvv(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                  nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result) {
    nk_umeyama_f16_serial(a, b, n, a_centroid, b_centroid, rotation, scale, result);
}

NK_PUBLIC void nk_rmsd_bf16_rvv(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result) {
    nk_rmsd_bf16_serial(a, b, n, a_centroid, b_centroid, rotation, scale, result);
}

NK_PUBLIC void nk_kabsch_bf16_rvv(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                  nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result) {
    nk_kabsch_bf16_serial(a, b, n, a_centroid, b_centroid, rotation, scale, result);
}

NK_PUBLIC void nk_umeyama_bf16_rvv(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                   nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result) {
    nk_umeyama_bf16_serial(a, b, n, a_centroid, b_centroid, rotation, scale, result);
}

#if defined(__cplusplus)
} // extern "C"
#endif

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#endif // NK_TARGET_RVV
#endif // NK_TARGET_RISCV_
#endif // NK_MESH_RVV_H

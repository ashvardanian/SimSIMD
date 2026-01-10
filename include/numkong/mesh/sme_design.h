/**
 *  @file include/numkong/mesh/sme_design.h
 *  @brief DESIGN DOCUMENT - SME-accelerated Kabsch/Umeyama algorithms.
 *
 *  This file contains the design for SME-accelerated mesh alignment algorithms.
 *  NOT FOR PRODUCTION USE - This is a design exploration file.
 *
 *  @section mesh_sme_instructions ARM SME/Streaming SVE Instructions
 *
 *      Intrinsic/Attribute             Instruction                     Latency         Throughput
 *      __arm_locally_streaming         SMSTART/SMSTOP (scoped)         ~50-100cy       -
 *      __arm_streaming                 SMSTART (enter streaming mode)  ~50-100cy       -
 *      __arm_streaming (exit)          SMSTOP (exit streaming mode)    ~50-100cy       -
 *      __arm_new("za")                 ZA tile allocation              0cy (compile-time)
 *      __arm_inout("za")               ZA tile read/write              0cy (compile-time)
 *      svmopa_za32_f32_m               FMOPA (ZA.S, P/M, Z.S, Z.S)     16cy (amortized over 16x16 tile)
 *      svld3_f32                       LD3W (Z.S, P/Z, [Xn])           8-12cy          0.5/cy
 *      svld1_f32                       LD1W (Z.S, P/Z, [Xn])           4-6cy           2/cy
 *      svmla_f32_x                     FMLA (Z.S, P/M, Z.S, Z.S)       4cy             2/cy
 *      svadd_f32_x                     FADD (Z.S, P/M, Z.S, Z.S)       3cy             2/cy
 *      svaddv_f32                      FADDV (S, P, Z.S)               6cy             1/cy
 *      svdup_f32                       DUP (Z.S, #imm)                 1cy             2/cy
 *      svptrue_b32                     PTRUE (P.S, pattern)            1cy             2/cy
 *      svcntw                          CNTW (Xd)                       1cy             2/cy
 *
 *  @section neon_vs_streaming_sve NEON vs Streaming SVE Performance Tradeoffs
 *
 *  For 3D mesh alignment (Kabsch/Umeyama), choosing between NEON and Streaming SVE depends on dataset size:
 *
 *  @code
 *  // NEON approach (current): 2.25N cycles for N points
 *  // - 6N scalar loads (interleaved XYZ)
 *  // - 9N FMAs with 4-wide vectors
 *  // - Simple, low overhead
 *
 *  // Streaming SVE approach: 1.2N cycles for N points
 *  // - svld3 hardware deinterleaving (~1 cycle per 16 points)
 *  // - 9 vector FMAs per 16 points
 *  // - Mode enter/exit overhead (~100 cycles)
 *  // - Best for N ≥ 1000 points
 *  @endcode
 *
 *  **Recommendation:**
 *  - N < 1000: Use NEON (avoid streaming mode overhead)
 *  - N ≥ 1000: Use Streaming SVE (1.9x speedup from efficient deinterleaving)
 *
 *  The 3x3 SVD step is too small to benefit from SME (16x16 minimum tile size).
 *
 *  ============================================================================
 *  SME OPPORTUNITY FOR KABSCH/UMEYAMA
 *  ============================================================================
 *
 *  The Kabsch and Umeyama algorithms compute a 3x3 cross-covariance matrix:
 *
 *      H = ∑ᵢ₌₀ⁿ⁻¹ (aᵢ - ā) ⨯ (bᵢ - b̄)ᵀ
 *
 *  This is a sum of outer products of 3D vectors, which maps directly to SME's
 *  `svmopa` (outer product accumulation) instruction.
 *
 *  CURRENT APPROACH (NEON/AVX):
 *  - 9 scalar accumulators (h00, h01, h02, h10, h11, h12, h20, h21, h22)
 *  - Process points with interleaved XYZ layout
 *  - FMA for each element: hᵢⱼ += (aᵢ - āᵢ) × (bⱼ - b̄ⱼ)
 *
 *  SME APPROACH:
 *  - Use ZA32 tiles (16x16 F32) to accumulate outer products
 *  - Reformat data to Structure of Arrays (SoA): [all X's][all Y's][all Z's]
 *  - Compute 3x3 = 9 element-wise products, but SME wants vector outer products
 *
 *  CHALLENGE: 3D points don't naturally fit SME's vector outer product model.
 *  SME expects: outer_product(vec_a[32], vec_b[32]) → tile[16x16]
 *  We have: outer_product(point_a[3], point_b[3]) → cov[3x3]
 *
 *  ============================================================================
 *  APPROACH 1: Batched Outer Products (RECOMMENDED)
 *  ============================================================================
 *
 *  Reformulate as batched computation:
 *
 *  For N points, partition X, Y, Z into vectors:
 *    Aₓ = [a₀.x, a₁.x, ..., a₃₁.x]  (32 elements for F16, 16 for F32)
 *    Aᵧ = [a₀.y, a₁.y, ..., a₃₁.y]
 *    Aᵧ = [a₀.z, a₁.z, ..., a₃₁.z]
 *    Bₓ = [b₀.x, b₁.x, ..., b₃₁.x]
 *    etc.
 *
 *  Then cross-covariance elements are:
 *    H₀₀ = ∑(Aₓ × Bₓ)
 *    H₀₁ = ∑(Aₓ × Bᵧ)
 *    ...
 *    H₂₂ = ∑(Aᵧ × Bᵧ)
 *
 *  This is 9 dot products, not outer products. SME's advantage is less clear here.
 *
 *  However, we can use SME for the SVD computation which follows, or use
 *  streaming SVE for efficient deinterleaving and FMA operations.
 *
 *  ============================================================================
 *  APPROACH 2: Streaming SVE for Deinterleaving + NEON FMA
 *  ============================================================================
 *
 *  Use streaming SVE loads (svld3) to deinterleave XYZ efficiently:
 *
 *  ```c
 *  __arm_locally_streaming
 *  void compute_cross_covariance_streaming_(...) {
 *      svbool_t ptrue = svptrue_b32();
 *      svfloat32_t sum_ax = svdup_f32(0), sum_ay = svdup_f32(0), sum_az = svdup_f32(0);
 *      svfloat32_t sum_bx = svdup_f32(0), sum_by = svdup_f32(0), sum_bz = svdup_f32(0);
 *      svfloat32_t h00 = svdup_f32(0), h01 = svdup_f32(0), h02 = svdup_f32(0);
 *      svfloat32_t h10 = svdup_f32(0), h11 = svdup_f32(0), h12 = svdup_f32(0);
 *      svfloat32_t h20 = svdup_f32(0), h21 = svdup_f32(0), h22 = svdup_f32(0);
 *
 *      for (size_t i = 0; i < n; i += svcntw()) {
 *          // Load interleaved XYZ and deinterleave
 *          svfloat32x3_t a_xyz = svld3_f32(ptrue, a + i * 3);
 *          svfloat32x3_t b_xyz = svld3_f32(ptrue, b + i * 3);
 *
 *          svfloat32_t ax = svget3(a_xyz, 0);
 *          svfloat32_t ay = svget3(a_xyz, 1);
 *          svfloat32_t az = svget3(a_xyz, 2);
 *          svfloat32_t bx = svget3(b_xyz, 0);
 *          svfloat32_t by = svget3(b_xyz, 1);
 *          svfloat32_t bz = svget3(b_xyz, 2);
 *
 *          // Accumulate sums for centroid
 *          sum_ax = svadd_f32_x(ptrue, sum_ax, ax);
 *          sum_ay = svadd_f32_x(ptrue, sum_ay, ay);
 *          // ...
 *
 *          // Accumulate cross products
 *          h00 = svmla_f32_x(ptrue, h00, ax, bx);  // += ax × bx
 *          h01 = svmla_f32_x(ptrue, h01, ax, by);  // += ax × by
 *          h02 = svmla_f32_x(ptrue, h02, ax, bz);  // += ax × bz
 *          h10 = svmla_f32_x(ptrue, h10, ay, bx);
 *          // ... 9 accumulators total
 *      }
 *
 *      // Horizontal reduce
 *      float H00 = svaddv_f32(ptrue, h00) - ... // subtract centroid correction
 *  }
 *  ```
 *
 *  BENEFIT: svld3 efficiently deinterleaves XYZ data in streaming mode.
 *  512-bit SVL = 16 F32 values = process 16 points per iteration.
 *
 *  ============================================================================
 *  APPROACH 3: SME for Small Matrix Operations (SVD)
 *  ============================================================================
 *
 *  After computing the 3x3 cross-covariance H, the algorithms perform SVD:
 *
 *      H = U × S × Vᵀ
 *      R = V × Uᵀ (with reflection handling)
 *
 *  The 3x3 SVD is currently implemented using iterative Jacobi rotations.
 *  SME could accelerate this if we had larger matrices, but for 3x3 the
 *  overhead of entering streaming mode exceeds any benefit.
 *
 *  ============================================================================
 *  PERFORMANCE ESTIMATE
 *  ============================================================================
 *
 *  For N 3D points:
 *
 *  CURRENT (NEON):
 *  - Loads: 2N * 3 = 6N scalar loads
 *  - FMAs: 9 * N = 9N FMAs (for cross-covariance)
 *  - With 4-wide NEON: ~2.25N cycles for FMAs
 *
 *  SME STREAMING SVE:
 *  - svld3 deinterleaves in hardware, ~1 cycle per 16 points
 *  - 9 svmla operations per 16 points
 *  - Reduction overhead at end
 *  - Estimated: ~0.6N cycles for loads + ~0.6N for FMAs = ~1.2N cycles
 *  - ~1.9x speedup potential
 *
 *  Note: For small N (< 100 points), streaming mode enter/exit overhead
 *  may dominate. Best for N > 1000 points.
 *
 *  ============================================================================
 *  IMPLEMENTATION RECOMMENDATION
 *  ============================================================================
 *
 *  1. For N < 1000: Use existing NEON implementation (avoid SME overhead)
 *
 *  2. For N ≥ 1000: Use Streaming SVE with svld3 deinterleaving
 *     - Efficient hardware deinterleaving
 *     - 16 points per iteration (F32) or 32 points (F16)
 *     - 9 vector FMAs for cross-covariance
 *
 *  3. The 3x3 SVD is too small to benefit from SME
 *     - Keep scalar/NEON implementation
 *     - 3x3 = 9 elements, SME tiles are 16x16 minimum
 *
 *  ============================================================================
 *  SKELETON IMPLEMENTATION
 *  ============================================================================
 */

#ifndef NK_MESH_SME_DESIGN_H
#define NK_MESH_SME_DESIGN_H

#ifdef NK_TARGET_SME

#include <arm_sme.h>
#include "numkong/types.h"

/**
 *  @brief Compute cross-covariance matrix using Streaming SVE.
 *
 *  Uses svld3 for efficient XYZ deinterleaving.
 *  Returns 9-element array [h00, h01, h02, h10, h11, h12, h20, h21, h22].
 *
 *  @param a First point cloud (source), n*3 interleaved [x0,y0,z0, x1,y1,z1, ...]
 *  @param b Second point cloud (target), n*3 interleaved
 *  @param n Number of 3D points
 *  @param[out] cross_cov Output 3x3 cross-covariance matrix (row-major)
 *  @param[out] centroid_a Output centroid of a [3]
 *  @param[out] centroid_b Output centroid of b [3]
 */
__arm_locally_streaming static void nk_mesh_cross_covariance_sve_(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n,
                                                                  nk_f32_t *cross_cov, nk_f32_t *centroid_a,
                                                                  nk_f32_t *centroid_b) {

    svbool_t const ptrue = svptrue_b32();
    nk_size_t const vec_len = svcntw(); // 16 for 512-bit SVL

    // Accumulators for sums (for centroid)
    svfloat32_t sum_ax = svdup_f32(0.0f), sum_ay = svdup_f32(0.0f), sum_az = svdup_f32(0.0f);
    svfloat32_t sum_bx = svdup_f32(0.0f), sum_by = svdup_f32(0.0f), sum_bz = svdup_f32(0.0f);

    // Accumulators for raw cross products (before centroid correction)
    svfloat32_t h00 = svdup_f32(0.0f), h01 = svdup_f32(0.0f), h02 = svdup_f32(0.0f);
    svfloat32_t h10 = svdup_f32(0.0f), h11 = svdup_f32(0.0f), h12 = svdup_f32(0.0f);
    svfloat32_t h20 = svdup_f32(0.0f), h21 = svdup_f32(0.0f), h22 = svdup_f32(0.0f);

    // Main loop - process vec_len points per iteration
    nk_size_t i = 0;
    for (; i + vec_len <= n; i += vec_len) {
        // Load interleaved XYZ and deinterleave using svld3
        // NOTE: svld3 requires contiguous interleaved data
        svfloat32x3_t a_xyz = svld3_f32(ptrue, a + i * 3);
        svfloat32x3_t b_xyz = svld3_f32(ptrue, b + i * 3);

        svfloat32_t ax = svget3_f32(a_xyz, 0);
        svfloat32_t ay = svget3_f32(a_xyz, 1);
        svfloat32_t az = svget3_f32(a_xyz, 2);
        svfloat32_t bx = svget3_f32(b_xyz, 0);
        svfloat32_t by = svget3_f32(b_xyz, 1);
        svfloat32_t bz = svget3_f32(b_xyz, 2);

        // Accumulate sums for centroid computation
        sum_ax = svadd_f32_x(ptrue, sum_ax, ax);
        sum_ay = svadd_f32_x(ptrue, sum_ay, ay);
        sum_az = svadd_f32_x(ptrue, sum_az, az);
        sum_bx = svadd_f32_x(ptrue, sum_bx, bx);
        sum_by = svadd_f32_x(ptrue, sum_by, by);
        sum_bz = svadd_f32_x(ptrue, sum_bz, bz);

        // Accumulate cross products: Hᵢⱼ = ∑(aᵢ × bⱼ)
        // Row 0: a.x * b
        h00 = svmla_f32_x(ptrue, h00, ax, bx);
        h01 = svmla_f32_x(ptrue, h01, ax, by);
        h02 = svmla_f32_x(ptrue, h02, ax, bz);
        // Row 1: a.y * b
        h10 = svmla_f32_x(ptrue, h10, ay, bx);
        h11 = svmla_f32_x(ptrue, h11, ay, by);
        h12 = svmla_f32_x(ptrue, h12, ay, bz);
        // Row 2: a.z * b
        h20 = svmla_f32_x(ptrue, h20, az, bx);
        h21 = svmla_f32_x(ptrue, h21, az, by);
        h22 = svmla_f32_x(ptrue, h22, az, bz);
    }

    // Horizontal reductions
    nk_f32_t total_ax = svaddv_f32(ptrue, sum_ax);
    nk_f32_t total_ay = svaddv_f32(ptrue, sum_ay);
    nk_f32_t total_az = svaddv_f32(ptrue, sum_az);
    nk_f32_t total_bx = svaddv_f32(ptrue, sum_bx);
    nk_f32_t total_by = svaddv_f32(ptrue, sum_by);
    nk_f32_t total_bz = svaddv_f32(ptrue, sum_bz);

    nk_f32_t H00 = svaddv_f32(ptrue, h00);
    nk_f32_t H01 = svaddv_f32(ptrue, h01);
    nk_f32_t H02 = svaddv_f32(ptrue, h02);
    nk_f32_t H10 = svaddv_f32(ptrue, h10);
    nk_f32_t H11 = svaddv_f32(ptrue, h11);
    nk_f32_t H12 = svaddv_f32(ptrue, h12);
    nk_f32_t H20 = svaddv_f32(ptrue, h20);
    nk_f32_t H21 = svaddv_f32(ptrue, h21);
    nk_f32_t H22 = svaddv_f32(ptrue, h22);

    // Handle tail (remaining points)
    for (; i < n; i++) {
        nk_f32_t ax = a[i * 3 + 0], ay = a[i * 3 + 1], az = a[i * 3 + 2];
        nk_f32_t bx = b[i * 3 + 0], by = b[i * 3 + 1], bz = b[i * 3 + 2];

        total_ax += ax;
        total_ay += ay;
        total_az += az;
        total_bx += bx;
        total_by += by;
        total_bz += bz;

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
    nk_f32_t inv_n = 1.0f / (nk_f32_t)n;
    centroid_a[0] = total_ax * inv_n;
    centroid_a[1] = total_ay * inv_n;
    centroid_a[2] = total_az * inv_n;
    centroid_b[0] = total_bx * inv_n;
    centroid_b[1] = total_by * inv_n;
    centroid_b[2] = total_bz * inv_n;

    // Apply centroid correction: H_corrected = H_raw - n × (ā ⊗ b̄)
    cross_cov[0] = H00 - total_ax * centroid_b[0];
    cross_cov[1] = H01 - total_ax * centroid_b[1];
    cross_cov[2] = H02 - total_ax * centroid_b[2];
    cross_cov[3] = H10 - total_ay * centroid_b[0];
    cross_cov[4] = H11 - total_ay * centroid_b[1];
    cross_cov[5] = H12 - total_ay * centroid_b[2];
    cross_cov[6] = H20 - total_az * centroid_b[0];
    cross_cov[7] = H21 - total_az * centroid_b[1];
    cross_cov[8] = H22 - total_az * centroid_b[2];
}

/**
 *  @brief Full Kabsch implementation using Streaming SVE for cross-covariance.
 *
 *  This function computes the optimal rotation matrix R that minimizes:
 *      ‖R × (a - ā) - (b - b̄)‖
 *
 *  @param a First point cloud (source), n*3 interleaved
 *  @param b Second point cloud (target), n*3 interleaved
 *  @param n Number of 3D points
 *  @param[out] rotation Output 3x3 rotation matrix (row-major, 9 values)
 *  @param[out] translation Output translation vector (b_centroid - R*a_centroid)
 *  @param[out] rmsd Root mean square deviation after alignment
 */
__arm_locally_streaming __arm_new("za") static void nk_mesh_kabsch_sve_(nk_f32_t const *a, nk_f32_t const *b,
                                                                        nk_size_t n, nk_f32_t *rotation,
                                                                        nk_f32_t *translation, nk_f32_t *rmsd) {

    nk_f32_t cross_cov[9];
    nk_f32_t centroid_a[3], centroid_b[3];

    // Step 1: Compute cross-covariance using Streaming SVE
    nk_mesh_cross_covariance_sve_(a, b, n, cross_cov, centroid_a, centroid_b);

    // Step 2: SVD of 3x3 cross-covariance
    // Use existing scalar SVD implementation (3x3 is too small for SME)
    nk_f32_t svd_u[9], svd_s[3], svd_v[9];
    // nk_svd3x3_f32_(cross_cov, svd_u, svd_s, svd_v);  // External function

    // Step 3: R = V × Uᵀ (with reflection handling if det < 0)
    // ... (same as existing implementation)

    // Step 4: Compute translation = b̄ - R × ā
    // ... (same as existing implementation)

    // Step 5: Compute RMSD
    // ... (same as existing implementation)
}

/**
 *  PERFORMANCE SUMMARY
 *  ===================
 *
 *  Test case: 10,000 3D points (F32)
 *
 *  NEON baseline:
 *  - Cross-covariance: ~15,000 cycles
 *  - SVD (3x3): ~500 cycles
 *  - Total: ~15,500 cycles
 *
 *  Streaming SVE:
 *  - Mode enter/exit: ~100 cycles
 *  - Cross-covariance: ~8,000 cycles (svld3 + 9 svmla per 16 points)
 *  - SVD: ~500 cycles (same)
 *  - Total: ~8,600 cycles
 *
 *  Expected speedup: ~1.8x for large point clouds
 *
 *  For small point clouds (N < 100): NEON is faster due to mode switching overhead.
 */

#endif // NK_TARGET_SME
#endif // NK_MESH_SME_DESIGN_H

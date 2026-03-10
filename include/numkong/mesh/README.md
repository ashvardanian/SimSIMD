# Point Cloud Alignment in NumKong

NumKong implements RMSD, Kabsch, and Umeyama algorithms for rigid-body superposition of 3D point clouds.
RMSD measures alignment quality, Kabsch finds the optimal rotation minimizing RMSD, and Umeyama extends Kabsch with uniform scaling.
Used in structural biology (protein alignment), robotics (point cloud registration), and computer graphics (mesh registration).

Centroid:

```math
\bar{a} = \frac{1}{n}\sum a_i
```

Cross-covariance matrix:

```math
H = \sum (a_i - \bar{a})(b_i - \bar{b})^T
```

SVD-based rotation:

```math
H = U \Sigma V^T, \quad R = V U^T
```

Umeyama scale factor:

```math
s = \frac{\text{tr}(\Sigma)}{n \cdot \sigma_a^2}
```

RMSD after alignment:

```math
\text{RMSD} = \sqrt{\frac{1}{n}\sum \|s \cdot R(a_i - \bar{a}) - (b_i - \bar{b})\|^2}
```

Reformulating as Python pseudocode:

```python
import numpy as np

def kabsch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_c, b_c = a - a.mean(0), b - b.mean(0)
    H = a_c.T @ b_c
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R

def umeyama(a: np.ndarray, b: np.ndarray) -> tuple:
    a_c, b_c = a - a.mean(0), b - b.mean(0)
    H = a_c.T @ b_c
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    scale = S.sum() / (len(a) * np.var(a_c))
    return R, scale

def rmsd(a: np.ndarray, b: np.ndarray) -> float:
    return np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1)))
```

## Input & Output Types

| Input Type | Output Type | Description                                    |
| ---------- | ----------- | ---------------------------------------------- |
| `f64`      | `f64`       | 64-bit IEEE 754 double precision               |
| `f32`      | `f32`       | 32-bit IEEE 754 single precision               |
| `f16`      | `f32`       | 16-bit IEEE 754 half precision, widened output |
| `bf16`     | `f32`       | 16-bit brain float, widened output             |

## Optimizations

### McAdams Branching-Free 3×3 SVD

`nk_kabsch_f32_serial`, `nk_kabsch_f64_haswell`, `nk_umeyama_f32_neon` use a Jacobi eigenanalysis with fixed 16 iterations (no convergence check) for deterministic behavior.
Quaternion-accumulated rotations: each Jacobi sweep updates a 4-element quaternion instead of recomputing eigenvectors.
Approximate Givens angles via `nk_approximate_givens_quaternion_` — a γ-threshold test selects between computed angles and precomputed cos(π/8), sin(π/8) constants.
Cyclic permutation of matrix elements avoids explicit sorting of eigenvalues.

### Stride-3 Deinterleaving

Point clouds are stored interleaved as [x₀,y₀,z₀, x₁,y₁,z₁, ...].
NEON uses `vld3q_f32` to hardware-deinterleave 4 XYZ triplets in one instruction — no gather needed.
Haswell uses `_mm256_i32gather_ps` with indices [0,3,6,9,12,15,18,21] to load 8 x-coordinates from 8 points.
RVV uses indexed loads with dynamic stride to adapt to variable vector length.

### Reflection Correction

`nk_kabsch_f32_haswell`, `nk_kabsch_f64_skylake` check for improper rotations (det(R) = -1, reflections) after computing R = V·Uᵀ.
If det(R) is negative, the last column of V is flipped.
This ensures the output is always a proper rotation matrix (det = +1).

### Pre-Scaled Rotation for Umeyama

`nk_umeyama_f32_haswell`, `nk_umeyama_f64_skylake` fold the computed scale factor into the rotation matrix before applying to points.
`sr[i] = scale * r[i]` is computed once and broadcast — avoiding a per-point scalar multiply.

### Why SME and SVE Were Removed

SME mesh kernels (`nk_rmsd_f32_sme`, `nk_kabsch_f32_sme`, `nk_umeyama_f32_sme`, plus f64 variants) were implemented in 1,052 lines across `sme.h` and `smef64.h` (commit `0e0bc30c`) and removed 4 days later (commit `f55e9a71`).
The fundamental mismatch: the algorithm computes a 3×3 cross-covariance matrix $H = \sum (a_i - \bar{a})(b_i - \bar{b})^T$ — a sum of outer products of 3D vectors.
SME's `FMOPA` operates on SVL-wide vectors (16+ elements at SVL=512), but the outer products here are 3×3 — the tile is 99.6% wasted (9 useful cells out of 256).
Three approaches were explored in a design document (`sme_design.h`, 398 lines):
(1) batched outer products — reformulates as 9 independent dot products but loses SME's outer-product strength, falling back to what NEON already does;
(2) streaming SVE with `svld3` — hardware stride-3 deinterleaving processes 16 points per iteration vs NEON's 4, but `SMSTART`/`SMSTOP` mode transitions cost ~100 cycles and the 3×3 SVD step cannot use streaming mode at all;
(3) SME for SVD — the 3×3 matrix cannot fill even one 16×16 tile.
Performance estimates from the design document: NEON baseline ~2.25N cycles for N points; streaming SVE ~1.2N cycles but with ~100-cycle mode transition overhead — for typical protein alignment workloads (N = 100–500 atoms), the overhead dominates.
SVE mesh kernels (`sve.h`, `svehalf.h`, 112 lines total) were removed in the same commit — variable vector length added complexity without clear benefit over fixed-width NEON for the 3D point cloud problem.

## Performance

The following performance tables are produced by manually re-running `nk_test` and `nk_bench` included internal tools to measure both accuracy and throughput at different input shapes.
The input size is controlled by the `NK_MESH_POINTS` environment variable and set to 256, 1024, and 4096 points.
Each alignment computes centroids, covariance, and a 3×3 SVD over $N$ point pairs, so cost is $O(N)$ per alignment with a large constant.
The throughput is measured in mp/s as millions of 3D points aligned per second.
Accuracy is reported as ULP (units in last place), the number of representable floating-point values between the result and the exact answer.
Each kernel runs for at least 20 seconds per configuration.
Benchmark threads are pinned to specific cores; on machines with heterogeneous core types (e.g., Apple P/E cores), only the fastest cores are used.
Workloads that significantly degrade CPU frequencies (Intel AMX, Apple SME) run in separate passes to avoid affecting throughput measurements of other kernels.

### Intel Sapphire Rapids

#### Native

| Kernel                    |                      256 |                     1024 |                     4096 |
| :------------------------ | -----------------------: | -----------------------: | -----------------------: |
| __f64__                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_rmsd_f64_serial`      |        220 mp/s, 1.4 ulp |        276 mp/s, 2.6 ulp |        198 mp/s, 5.2 ulp |
| `nk_kabsch_f64_serial`    |         49 mp/s, 1.4 ulp |         96 mp/s, 2.7 ulp |        104 mp/s, 5.1 ulp |
| `nk_umeyama_f64_serial`   |         51 mp/s, 1.0 ulp |         94 mp/s, 1.9 ulp |        103 mp/s, 3.6 ulp |
| `nk_rmsd_f64_haswell`     |        181 mp/s, 0.3 ulp |        456 mp/s, 0.4 ulp |        237 mp/s, 0.8 ulp |
| `nk_kabsch_f64_haswell`   |         69 mp/s, 0.9 ulp |        160 mp/s, 1.3 ulp |        172 mp/s, 2.3 ulp |
| `nk_umeyama_f64_haswell`  |         56 mp/s, 0.4 ulp |        160 mp/s, 0.8 ulp |        171 mp/s, 1.6 ulp |
| `nk_rmsd_f64_skylake`     |        463 mp/s, 0.3 ulp |        560 mp/s, 0.3 ulp |        254 mp/s, 0.4 ulp |
| `nk_kabsch_f64_skylake`   |         91 mp/s, 0.7 ulp |        201 mp/s, 0.9 ulp |        180 mp/s, 1.3 ulp |
| `nk_umeyama_f64_skylake`  |         88 mp/s, 0.2 ulp |        206 mp/s, 0.4 ulp |        183 mp/s, 0.8 ulp |
| __f32__                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_rmsd_f32_serial`      |        294 mp/s, 1.4 ulp |        469 mp/s, 2.6 ulp |        362 mp/s, 5.2 ulp |
| `nk_kabsch_f32_serial`    |         64 mp/s, 1.5 ulp |        129 mp/s, 2.6 ulp |        143 mp/s, 5.2 ulp |
| `nk_umeyama_f32_serial`   |         61 mp/s, 0.9 ulp |        119 mp/s, 1.8 ulp |        136 mp/s, 3.7 ulp |
| `nk_rmsd_f32_haswell`     |        581 mp/s, 0.3 ulp |        861 mp/s, 0.3 ulp |        520 mp/s, 0.4 ulp |
| `nk_kabsch_f32_haswell`   |        106 mp/s, 0.7 ulp |        248 mp/s, 0.9 ulp |        277 mp/s, 1.3 ulp |
| `nk_umeyama_f32_haswell`  |        100 mp/s, 0.2 ulp |        249 mp/s, 0.4 ulp |        271 mp/s, 0.8 ulp |
| `nk_rmsd_f32_skylake`     |        808 mp/s, 0.3 ulp |      1,256 mp/s, 0.3 ulp |        612 mp/s, 0.3 ulp |
| `nk_kabsch_f32_skylake`   |        124 mp/s, 0.7 ulp |        341 mp/s, 0.7 ulp |        377 mp/s, 0.9 ulp |
| `nk_umeyama_f32_skylake`  |        118 mp/s, 0.2 ulp |        340 mp/s, 0.3 ulp |        363 mp/s, 0.4 ulp |
| __bf16__                  | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_rmsd_bf16_haswell`    |        491 mp/s, 0.3 ulp |        836 mp/s, 3.6 ulp |       820 mp/s, 12.8 ulp |
| `nk_kabsch_bf16_haswell`  |         51 mp/s, 0.7 ulp |         74 mp/s, 0.9 ulp |         81 mp/s, 1.3 ulp |
| `nk_umeyama_bf16_haswell` |         51 mp/s, 0.3 ulp |         74 mp/s, 0.4 ulp |         80 mp/s, 0.8 ulp |
| __f16__                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_rmsd_f16_haswell`     |        392 mp/s, 0.3 ulp |        793 mp/s, 0.7 ulp |        794 mp/s, 2.4 ulp |
| `nk_kabsch_f16_haswell`   |        149 mp/s, 0.7 ulp |        363 mp/s, 0.9 ulp |        398 mp/s, 1.3 ulp |
| `nk_umeyama_f16_haswell`  |        159 mp/s, 0.3 ulp |        361 mp/s, 0.4 ulp |        367 mp/s, 0.8 ulp |

#### WASM

Measured with Wasmtime v42 (Cranelift backend).

| Kernel                        |                      256 |                     1024 |                     4096 |
| :---------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f64__                       | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_rmsd_f64_serial`          |          ? mp/s, 1.5 ulp |          ? mp/s, 2.7 ulp |          ? mp/s, 4.8 ulp |
| `nk_rmsd_f64_v128relaxed`     |          ? mp/s, 0.4 ulp |          ? mp/s, 0.8 ulp |          ? mp/s, 1.5 ulp |
| `nk_kabsch_f64_serial`        |          ? mp/s, 1.6 ulp |          ? mp/s, 2.2 ulp |          ? mp/s, 3.9 ulp |
| `nk_kabsch_f64_v128relaxed`   |          ? mp/s, 1.2 ulp |          ? mp/s, 2.1 ulp |          ? mp/s, 4.9 ulp |
| `nk_umeyama_f64_serial`       |          ? mp/s, 1.0 ulp |          ? mp/s, 2.0 ulp |          ? mp/s, 3.2 ulp |
| `nk_umeyama_f64_v128relaxed`  |          ? mp/s, 0.8 ulp |          ? mp/s, 1.5 ulp |          ? mp/s, 3.1 ulp |
| __f32__                       | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_rmsd_f32_serial`          |          ? mp/s, 1.4 ulp |          ? mp/s, 2.8 ulp |          ? mp/s, 5.2 ulp |
| `nk_rmsd_f32_v128relaxed`     |          ? mp/s, 0.3 ulp |          ? mp/s, 0.4 ulp |          ? mp/s, 0.6 ulp |
| `nk_kabsch_f32_serial`        |          ? mp/s, 1.4 ulp |          ? mp/s, 2.7 ulp |          ? mp/s, 4.9 ulp |
| `nk_kabsch_f32_v128relaxed`   |          ? mp/s, 0.8 ulp |          ? mp/s, 1.3 ulp |          ? mp/s, 2.3 ulp |
| `nk_umeyama_f32_serial`       |          ? mp/s, 1.0 ulp |          ? mp/s, 1.7 ulp |          ? mp/s, 4.4 ulp |
| `nk_umeyama_f32_v128relaxed`  |          ? mp/s, 0.4 ulp |          ? mp/s, 0.8 ulp |          ? mp/s, 1.6 ulp |
| __bf16__                      | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_rmsd_bf16_serial`         |            ? mp/s, ? ulp |            ? mp/s, ? ulp |            ? mp/s, ? ulp |
| `nk_rmsd_bf16_v128relaxed`    |            ? mp/s, ? ulp |            ? mp/s, ? ulp |            ? mp/s, ? ulp |
| `nk_kabsch_bf16_serial`       |            ? mp/s, ? ulp |            ? mp/s, ? ulp |            ? mp/s, ? ulp |
| `nk_kabsch_bf16_v128relaxed`  |            ? mp/s, ? ulp |            ? mp/s, ? ulp |            ? mp/s, ? ulp |
| `nk_umeyama_bf16_serial`      |            ? mp/s, ? ulp |            ? mp/s, ? ulp |            ? mp/s, ? ulp |
| `nk_umeyama_bf16_v128relaxed` |            ? mp/s, ? ulp |            ? mp/s, ? ulp |            ? mp/s, ? ulp |
| __f16__                       | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_rmsd_f16_serial`          |            ? mp/s, ? ulp |            ? mp/s, ? ulp |            ? mp/s, ? ulp |
| `nk_rmsd_f16_v128relaxed`     |            ? mp/s, ? ulp |            ? mp/s, ? ulp |            ? mp/s, ? ulp |
| `nk_kabsch_f16_serial`        |            ? mp/s, ? ulp |            ? mp/s, ? ulp |            ? mp/s, ? ulp |
| `nk_kabsch_f16_v128relaxed`   |            ? mp/s, ? ulp |            ? mp/s, ? ulp |            ? mp/s, ? ulp |
| `nk_umeyama_f16_serial`       |            ? mp/s, ? ulp |            ? mp/s, ? ulp |            ? mp/s, ? ulp |
| `nk_umeyama_f16_v128relaxed`  |            ? mp/s, ? ulp |            ? mp/s, ? ulp |            ? mp/s, ? ulp |

### Apple M4 Pro

#### Native

| Kernel                   |                      256 |                     1024 |                     4096 |
| :----------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f64__                  | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_rmsd_f64_serial`     |            ? mp/s, ? ulp |            ? mp/s, ? ulp |            ? mp/s, ? ulp |
| `nk_kabsch_f64_serial`   |            ? mp/s, ? ulp |            ? mp/s, ? ulp |            ? mp/s, ? ulp |
| `nk_umeyama_f64_serial`  |            ? mp/s, ? ulp |            ? mp/s, ? ulp |            ? mp/s, ? ulp |
| `nk_rmsd_f64_neon`       |            ? mp/s, ? ulp |            ? mp/s, ? ulp |            ? mp/s, ? ulp |
| `nk_kabsch_f64_neon`     |            ? mp/s, ? ulp |            ? mp/s, ? ulp |            ? mp/s, ? ulp |
| `nk_umeyama_f64_neon`    |            ? mp/s, ? ulp |            ? mp/s, ? ulp |            ? mp/s, ? ulp |
| __f32__                  | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_rmsd_f32_serial`     |            ? mp/s, ? ulp |            ? mp/s, ? ulp |            ? mp/s, ? ulp |
| `nk_kabsch_f32_serial`   |            ? mp/s, ? ulp |            ? mp/s, ? ulp |            ? mp/s, ? ulp |
| `nk_umeyama_f32_serial`  |            ? mp/s, ? ulp |            ? mp/s, ? ulp |            ? mp/s, ? ulp |
| `nk_rmsd_f32_neon`       |            ? mp/s, ? ulp |            ? mp/s, ? ulp |            ? mp/s, ? ulp |
| `nk_kabsch_f32_neon`     |            ? mp/s, ? ulp |            ? mp/s, ? ulp |            ? mp/s, ? ulp |
| `nk_umeyama_f32_neon`    |            ? mp/s, ? ulp |            ? mp/s, ? ulp |            ? mp/s, ? ulp |
| __bf16__                 | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_rmsd_bf16_serial`    |            ? mp/s, ? ulp |            ? mp/s, ? ulp |            ? mp/s, ? ulp |
| `nk_kabsch_bf16_serial`  |            ? mp/s, ? ulp |            ? mp/s, ? ulp |            ? mp/s, ? ulp |
| `nk_umeyama_bf16_serial` |            ? mp/s, ? ulp |            ? mp/s, ? ulp |            ? mp/s, ? ulp |
| __f16__                  | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_rmsd_f16_serial`     |            ? mp/s, ? ulp |            ? mp/s, ? ulp |            ? mp/s, ? ulp |
| `nk_kabsch_f16_serial`   |            ? mp/s, ? ulp |            ? mp/s, ? ulp |            ? mp/s, ? ulp |
| `nk_umeyama_f16_serial`  |            ? mp/s, ? ulp |            ? mp/s, ? ulp |            ? mp/s, ? ulp |

#### WASM

Measured with Wasmtime v42 (Cranelift backend).

| Kernel                        |                      256 |                     1024 |                     4096 |
| :---------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f64__                       | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_rmsd_f64_serial`          |            ? mp/s, ? ulp |            ? mp/s, ? ulp |            ? mp/s, ? ulp |
| `nk_rmsd_f64_v128relaxed`     |            ? mp/s, ? ulp |            ? mp/s, ? ulp |            ? mp/s, ? ulp |
| `nk_kabsch_f64_serial`        |            ? mp/s, ? ulp |            ? mp/s, ? ulp |            ? mp/s, ? ulp |
| `nk_kabsch_f64_v128relaxed`   |            ? mp/s, ? ulp |            ? mp/s, ? ulp |            ? mp/s, ? ulp |
| `nk_umeyama_f64_serial`       |            ? mp/s, ? ulp |            ? mp/s, ? ulp |            ? mp/s, ? ulp |
| `nk_umeyama_f64_v128relaxed`  |            ? mp/s, ? ulp |            ? mp/s, ? ulp |            ? mp/s, ? ulp |
| __f32__                       | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_rmsd_f32_serial`          |            ? mp/s, ? ulp |            ? mp/s, ? ulp |            ? mp/s, ? ulp |
| `nk_rmsd_f32_v128relaxed`     |            ? mp/s, ? ulp |            ? mp/s, ? ulp |            ? mp/s, ? ulp |
| `nk_kabsch_f32_serial`        |            ? mp/s, ? ulp |            ? mp/s, ? ulp |            ? mp/s, ? ulp |
| `nk_kabsch_f32_v128relaxed`   |            ? mp/s, ? ulp |            ? mp/s, ? ulp |            ? mp/s, ? ulp |
| `nk_umeyama_f32_serial`       |            ? mp/s, ? ulp |            ? mp/s, ? ulp |            ? mp/s, ? ulp |
| `nk_umeyama_f32_v128relaxed`  |            ? mp/s, ? ulp |            ? mp/s, ? ulp |            ? mp/s, ? ulp |
| __bf16__                      | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_rmsd_bf16_serial`         |            ? mp/s, ? ulp |            ? mp/s, ? ulp |            ? mp/s, ? ulp |
| `nk_rmsd_bf16_v128relaxed`    |            ? mp/s, ? ulp |            ? mp/s, ? ulp |            ? mp/s, ? ulp |
| `nk_kabsch_bf16_serial`       |            ? mp/s, ? ulp |            ? mp/s, ? ulp |            ? mp/s, ? ulp |
| `nk_kabsch_bf16_v128relaxed`  |            ? mp/s, ? ulp |            ? mp/s, ? ulp |            ? mp/s, ? ulp |
| `nk_umeyama_bf16_serial`      |            ? mp/s, ? ulp |            ? mp/s, ? ulp |            ? mp/s, ? ulp |
| `nk_umeyama_bf16_v128relaxed` |            ? mp/s, ? ulp |            ? mp/s, ? ulp |            ? mp/s, ? ulp |
| __f16__                       | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_rmsd_f16_serial`          |            ? mp/s, ? ulp |            ? mp/s, ? ulp |            ? mp/s, ? ulp |
| `nk_rmsd_f16_v128relaxed`     |            ? mp/s, ? ulp |            ? mp/s, ? ulp |            ? mp/s, ? ulp |
| `nk_kabsch_f16_serial`        |            ? mp/s, ? ulp |            ? mp/s, ? ulp |            ? mp/s, ? ulp |
| `nk_kabsch_f16_v128relaxed`   |            ? mp/s, ? ulp |            ? mp/s, ? ulp |            ? mp/s, ? ulp |
| `nk_umeyama_f16_serial`       |            ? mp/s, ? ulp |            ? mp/s, ? ulp |            ? mp/s, ? ulp |
| `nk_umeyama_f16_v128relaxed`  |            ? mp/s, ? ulp |            ? mp/s, ? ulp |            ? mp/s, ? ulp |

# Batched Distance Matrices in NumKong

NumKong implements batched distance matrix computation via pre-packed dot products plus normalization. Angular distance and Euclidean distance are computed from the packed dot product output without materializing an intermediate C matrix.

Angular distance from pre-packed dot products:

```math
D_{ij} = 1 - \frac{C_{ij}}{\sqrt{\|A_i\|^2 \cdot \|B_j\|^2}}
```

Euclidean distance from pre-packed dot products:

```math
D_{ij} = \sqrt{\|A_i\|^2 + \|B_j\|^2 - 2 C_{ij}}
```

Reformulating as Python pseudocode:

```python
import numpy as np

def angulars_packed(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    dots = a @ b.T
    a_norms = np.sum(a ** 2, axis=1, keepdims=True)
    b_norms = np.sum(b ** 2, axis=1, keepdims=True)
    return 1 - dots / np.sqrt(a_norms * b_norms.T)

def euclideans_packed(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    dots = a @ b.T
    a_norms = np.sum(a ** 2, axis=1, keepdims=True)
    b_norms = np.sum(b ** 2, axis=1, keepdims=True)
    return np.sqrt(np.maximum(a_norms + b_norms.T - 2 * dots, 0))
```

## Input & Output Types

| Input Type | Output Type | Description                                    |
| ---------- | ----------- | ---------------------------------------------- |
| `f64`      | `f64`       | 64-bit IEEE 754 double precision               |
| `f32`      | `f32`       | 32-bit IEEE 754 single precision               |
| `f16`      | `f32`       | 16-bit IEEE 754 half precision, widened output |
| `bf16`     | `f32`       | 16-bit brain float, widened output             |
| `e4m3`     | `f32`       | 8-bit FP8: 4 exponent, 3 mantissa bits         |
| `e5m2`     | `f32`       | 8-bit FP8: 5 exponent, 2 mantissa bits         |
| `e2m3`     | `f32`       | 8-bit MX format: 2 exponent, 3 mantissa bits   |
| `e3m2`     | `f32`       | 8-bit MX format: 3 exponent, 2 mantissa bits   |
| `i8`       | `f32`       | 8-bit signed integers, float output            |
| `u8`       | `f32`       | 8-bit unsigned integers, float output          |
| `i4`       | `f32`       | 4-bit signed integers, float output            |
| `u4`       | `f32`       | 4-bit unsigned integers, float output          |

## Optimizations

### Distance-from-Dot Algebraic Reduction

`nk_angulars_packed_f32_haswell`, `nk_angulars_packed_f32_skylake`, `nk_euclideans_packed_f32_haswell`, `nk_euclideans_packed_f32_skylake` derive distance matrices from pre-packed dot product output without materializing an intermediate result matrix.
Angular distance rewrites as $1 - \text{dot}(a,b) \cdot \text{rsqrt}(\|a\|^2 \cdot \|b\|^2)$, converting two separate square roots and a division into one rsqrt and one multiply.
Euclidean distance expands the identity $\|a - b\|^2 = \|a\|^2 + \|b\|^2 - 2 \cdot \text{dot}(a,b)$, requiring only one final sqrt per output element.
Both formulas decompose into: (1) a batched GEMM for all M×N dot products, (2) per-vector squared norms precomputed once during packing.
The singular `spatial/` kernels compute these three sums ($\sum a_i b_i$, $\sum a_i^2$, $\sum b_i^2$) in a single pass with three interleaved accumulators; the batched `spatials/` kernels separate them — norms are computed once per vector during packing, and dots come from the GEMM — trading register pressure for amortized cost across the full M×N output.

### Serial vs Vectorized Sqrt and Rsqrt Cost

`nk_angular_through_f32_from_dot_serial_` uses the Quake 3 fast inverse square root (magic constant `0x5F375A86`, three Newton-Raphson iterations, ~34.9 correct bits for f32) to compute `dot * rsqrt(query_norm * target_norm)`.
`nk_angular_through_f32_from_dot_haswell_` replaces this with hardware `_mm_rsqrt_ps` (~12-bit approximation, 5cy latency, 1/cy on port 0) plus one Newton-Raphson refinement step (~22–24 correct bits).
`nk_euclidean_through_f32_from_dot_serial_` computes `sqrt(x)` as `x * rsqrt(x)` — reusing the same rsqrt path.
`nk_euclidean_through_f32_from_dot_haswell_` uses exact `_mm_sqrt_ps` (11cy latency, 7cy throughput for XMM) instead of the rsqrt approximation — the subtraction $\|a\|^2 + \|b\|^2 - 2 \cdot \text{dot}$ can produce values near zero where rsqrt error would be amplified by the subsequent multiply.
For f64, all backends use exact division and sqrt — no fast rsqrt approximation, since reaching 52 mantissa bits of precision would need 4+ Newton-Raphson iterations, negating the speed advantage.
The 4-wide finalizer batching amortizes these costs: one rsqrt or sqrt call processes 4 output elements simultaneously, hiding the latency behind the GEMM tile's computation.

### Norm Precomputation in Packed Buffers

`nk_dots_pack_f32_serial`, `nk_dots_pack_f32_haswell`, `nk_dots_pack_bf16_haswell` compute per-column squared norms $\|b_j\|^2 = \sum_k b_{jk}^2 = \text{dot}(b_j, b_j)$ during the packing step via `nk_reduce_moments_*` primitives.
The squared norm is a self-dot-product — already a byproduct of touching every element for type conversion and layout transformation.
Angular and Euclidean finalizers read norms from packed buffer metadata, eliminating a separate O(N·K) norm pass over B.

## Performance

Controlled by `NK_MATRIX_HEIGHT`, `NK_MATRIX_WIDTH`, `NK_MATRIX_DEPTH`.
All values are set to the same value for products of two square-shaped matrices.
Columns show for matrixes with 256, 1024, and 4096 sides.

### Intel Sapphire Rapids

#### Native

| Kernel                                     |            256³ |           1024³ |           4096³ |
| :----------------------------------------- | --------------: | --------------: | --------------: |
| __f64__                                    |                 |                 |                 |
| `nk_angulars_packed_f64_serial`            | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_symmetric_f64_serial`         | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_packed_f64_serial`          | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_symmetric_f64_serial`       | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_packed_f64_haswell`           | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_symmetric_f64_haswell`        | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_packed_f64_haswell`         | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_symmetric_f64_haswell`      | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_packed_f64_skylake`           | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_symmetric_f64_skylake`        | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_packed_f64_skylake`         | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_symmetric_f64_skylake`      | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| __f32__                                    |                 |                 |                 |
| `nk_angulars_packed_f32_serial`            | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_symmetric_f32_serial`         | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_packed_f32_serial`          | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_symmetric_f32_serial`       | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_packed_f32_haswell`           | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_symmetric_f32_haswell`        | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_packed_f32_haswell`         | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_symmetric_f32_haswell`      | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_packed_f32_skylake`           | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_symmetric_f32_skylake`        | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_packed_f32_skylake`         | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_symmetric_f32_skylake`      | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| __bf16__                                   |                 |                 |                 |
| `nk_angulars_packed_bf16_serial`           | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_symmetric_bf16_serial`        | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_packed_bf16_serial`         | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_symmetric_bf16_serial`      | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_packed_bf16_haswell`          | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_symmetric_bf16_haswell`       | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_packed_bf16_haswell`        | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_symmetric_bf16_haswell`     | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_packed_bf16_skylake`          | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_symmetric_bf16_skylake`       | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_packed_bf16_skylake`        | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_symmetric_bf16_skylake`     | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_packed_bf16_genoa`            | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_symmetric_bf16_genoa`         | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_packed_bf16_genoa`          | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_symmetric_bf16_genoa`       | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_packed_bf16_sapphireamx`      | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_symmetric_bf16_sapphireamx`   | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_packed_bf16_sapphireamx`    | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_symmetric_bf16_sapphireamx` | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| __f16__                                    |                 |                 |                 |
| `nk_angulars_packed_f16_serial`            | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_symmetric_f16_serial`         | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_packed_f16_serial`          | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_symmetric_f16_serial`       | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_packed_f16_haswell`           | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_symmetric_f16_haswell`        | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_packed_f16_haswell`         | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_symmetric_f16_haswell`      | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_packed_f16_skylake`           | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_symmetric_f16_skylake`        | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_packed_f16_skylake`         | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_symmetric_f16_skylake`      | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| __e5m2__                                   |                 |                 |                 |
| `nk_angulars_packed_e5m2_serial`           | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_symmetric_e5m2_serial`        | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_packed_e5m2_serial`         | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_symmetric_e5m2_serial`      | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_packed_e5m2_haswell`          | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_symmetric_e5m2_haswell`       | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_packed_e5m2_haswell`        | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_symmetric_e5m2_haswell`     | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_packed_e5m2_skylake`          | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_symmetric_e5m2_skylake`       | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_packed_e5m2_skylake`        | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_symmetric_e5m2_skylake`     | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_packed_e5m2_genoa`            | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_symmetric_e5m2_genoa`         | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_packed_e5m2_genoa`          | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_symmetric_e5m2_genoa`       | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_packed_e5m2_sapphireamx`      | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_symmetric_e5m2_sapphireamx`   | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_packed_e5m2_sapphireamx`    | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_symmetric_e5m2_sapphireamx` | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| __e4m3__                                   |                 |                 |                 |
| `nk_angulars_packed_e4m3_serial`           | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_symmetric_e4m3_serial`        | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_packed_e4m3_serial`         | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_symmetric_e4m3_serial`      | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_packed_e4m3_haswell`          | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_symmetric_e4m3_haswell`       | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_packed_e4m3_haswell`        | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_symmetric_e4m3_haswell`     | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_packed_e4m3_skylake`          | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_symmetric_e4m3_skylake`       | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_packed_e4m3_skylake`        | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_symmetric_e4m3_skylake`     | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_packed_e4m3_genoa`            | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_symmetric_e4m3_genoa`         | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_packed_e4m3_genoa`          | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_symmetric_e4m3_genoa`       | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_packed_e4m3_sapphireamx`      | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_symmetric_e4m3_sapphireamx`   | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_packed_e4m3_sapphireamx`    | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_symmetric_e4m3_sapphireamx` | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| __e3m2__                                   |                 |                 |                 |
| `nk_angulars_packed_e3m2_serial`           | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_symmetric_e3m2_serial`        | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_packed_e3m2_serial`         | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_symmetric_e3m2_serial`      | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_packed_e3m2_haswell`          | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_symmetric_e3m2_haswell`       | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_packed_e3m2_haswell`        | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_symmetric_e3m2_haswell`     | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_packed_e3m2_skylake`          | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_symmetric_e3m2_skylake`       | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_packed_e3m2_skylake`        | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_symmetric_e3m2_skylake`     | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_packed_e3m2_genoa`            | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_symmetric_e3m2_genoa`         | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_packed_e3m2_genoa`          | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_symmetric_e3m2_genoa`       | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_packed_e3m2_sapphireamx`      | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_symmetric_e3m2_sapphireamx`   | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_packed_e3m2_sapphireamx`    | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_symmetric_e3m2_sapphireamx` | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| __e2m3__                                   |                 |                 |                 |
| `nk_angulars_packed_e2m3_serial`           | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_symmetric_e2m3_serial`        | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_packed_e2m3_serial`         | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_symmetric_e2m3_serial`      | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_packed_e2m3_haswell`          | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_symmetric_e2m3_haswell`       | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_packed_e2m3_haswell`        | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_symmetric_e2m3_haswell`     | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_packed_e2m3_skylake`          | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_symmetric_e2m3_skylake`       | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_packed_e2m3_skylake`        | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_symmetric_e2m3_skylake`     | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_packed_e2m3_genoa`            | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_symmetric_e2m3_genoa`         | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_packed_e2m3_genoa`          | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_symmetric_e2m3_genoa`       | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_packed_e2m3_sapphireamx`      | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_symmetric_e2m3_sapphireamx`   | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_packed_e2m3_sapphireamx`    | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_symmetric_e2m3_sapphireamx` | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| __i8__                                     |                 |                 |                 |
| `nk_angulars_packed_i8_serial`             | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_symmetric_i8_serial`          | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_packed_i8_serial`           | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_symmetric_i8_serial`        | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_packed_i8_haswell`            | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_symmetric_i8_haswell`         | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_packed_i8_haswell`          | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_symmetric_i8_haswell`       | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_packed_i8_icelake`            | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_symmetric_i8_icelake`         | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_packed_i8_icelake`          | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_symmetric_i8_icelake`       | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_packed_i8_sapphireamx`        | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_symmetric_i8_sapphireamx`     | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_packed_i8_sapphireamx`      | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_symmetric_i8_sapphireamx`   | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_packed_i8_alder`              | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_symmetric_i8_alder`           | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_packed_i8_alder`            | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_symmetric_i8_alder`         | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_packed_i8_sierra`             | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_symmetric_i8_sierra`          | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_packed_i8_sierra`           | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_symmetric_i8_sierra`        | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| __u8__                                     |                 |                 |                 |
| `nk_angulars_packed_u8_serial`             | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_symmetric_u8_serial`          | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_packed_u8_serial`           | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_symmetric_u8_serial`        | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_packed_u8_haswell`            | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_symmetric_u8_haswell`         | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_packed_u8_haswell`          | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_symmetric_u8_haswell`       | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_packed_u8_icelake`            | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_symmetric_u8_icelake`         | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_packed_u8_icelake`          | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_symmetric_u8_icelake`       | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_packed_u8_sapphireamx`        | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_symmetric_u8_sapphireamx`     | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_packed_u8_sapphireamx`      | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_symmetric_u8_sapphireamx`   | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_packed_u8_alder`              | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_symmetric_u8_alder`           | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_packed_u8_alder`            | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_symmetric_u8_alder`         | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_packed_u8_sierra`             | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_symmetric_u8_sierra`          | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_packed_u8_sierra`           | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_symmetric_u8_sierra`        | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| __i4__                                     |                 |                 |                 |
| `nk_angulars_packed_i4_serial`             | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_symmetric_i4_serial`          | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_packed_i4_serial`           | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_symmetric_i4_serial`        | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_packed_i4_icelake`            | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_symmetric_i4_icelake`         | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_packed_i4_icelake`          | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_symmetric_i4_icelake`       | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| __u4__                                     |                 |                 |                 |
| `nk_angulars_packed_u4_serial`             | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_symmetric_u4_serial`          | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_packed_u4_serial`           | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_symmetric_u4_serial`        | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_packed_u4_icelake`            | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_symmetric_u4_icelake`         | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_packed_u4_icelake`          | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_symmetric_u4_icelake`       | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |

#### V8 (Chromium)

| Kernel                                     |            256³ |           1024³ |           4096³ |
| :----------------------------------------- | --------------: | --------------: | --------------: |
| __f64__                                    |                 |                 |                 |
| `nk_angulars_packed_f64_v128relaxed`       | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_symmetric_f64_v128relaxed`    | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_packed_f64_v128relaxed`     | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_symmetric_f64_v128relaxed`  | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| __f32__                                    |                 |                 |                 |
| `nk_angulars_packed_f32_v128relaxed`       | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_symmetric_f32_v128relaxed`    | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_packed_f32_v128relaxed`     | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_symmetric_f32_v128relaxed`  | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| __bf16__                                   |                 |                 |                 |
| `nk_angulars_packed_bf16_v128relaxed`      | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_symmetric_bf16_v128relaxed`   | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_packed_bf16_v128relaxed`    | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_symmetric_bf16_v128relaxed` | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| __e2m3__                                   |                 |                 |                 |
| `nk_angulars_packed_e2m3_v128relaxed`      | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_symmetric_e2m3_v128relaxed`   | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_packed_e2m3_v128relaxed`    | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_symmetric_e2m3_v128relaxed` | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| __i8__                                     |                 |                 |                 |
| `nk_angulars_packed_i8_v128relaxed`        | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_symmetric_i8_v128relaxed`     | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_packed_i8_v128relaxed`      | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_symmetric_i8_v128relaxed`   | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| __u8__                                     |                 |                 |                 |
| `nk_angulars_packed_u8_v128relaxed`        | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_symmetric_u8_v128relaxed`     | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_packed_u8_v128relaxed`      | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_symmetric_u8_v128relaxed`   | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |

#### Wasmtime (Cranelift)

| Kernel                                     |            256³ |           1024³ |           4096³ |
| :----------------------------------------- | --------------: | --------------: | --------------: |
| __f64__                                    |                 |                 |                 |
| `nk_angulars_packed_f64_v128relaxed`       | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_symmetric_f64_v128relaxed`    | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_packed_f64_v128relaxed`     | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_symmetric_f64_v128relaxed`  | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| __f32__                                    |                 |                 |                 |
| `nk_angulars_packed_f32_v128relaxed`       | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_symmetric_f32_v128relaxed`    | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_packed_f32_v128relaxed`     | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_symmetric_f32_v128relaxed`  | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| __bf16__                                   |                 |                 |                 |
| `nk_angulars_packed_bf16_v128relaxed`      | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_symmetric_bf16_v128relaxed`   | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_packed_bf16_v128relaxed`    | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_symmetric_bf16_v128relaxed` | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| __e2m3__                                   |                 |                 |                 |
| `nk_angulars_packed_e2m3_v128relaxed`      | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_symmetric_e2m3_v128relaxed`   | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_packed_e2m3_v128relaxed`    | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_symmetric_e2m3_v128relaxed` | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| __i8__                                     |                 |                 |                 |
| `nk_angulars_packed_i8_v128relaxed`        | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_symmetric_i8_v128relaxed`     | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_packed_i8_v128relaxed`      | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_symmetric_i8_v128relaxed`   | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| __u8__                                     |                 |                 |                 |
| `nk_angulars_packed_u8_v128relaxed`        | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_symmetric_u8_v128relaxed`     | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_packed_u8_v128relaxed`      | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_symmetric_u8_v128relaxed`   | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |

### Apple M4 Pro

#### Native

| Kernel                                   |            256³ |           1024³ |           4096³ |
| :--------------------------------------- | --------------: | --------------: | --------------: |
| __f64__                                  |                 |                 |                 |
| `nk_angulars_packed_f64_serial`          | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_symmetric_f64_serial`       | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_packed_f64_serial`        | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_symmetric_f64_serial`     | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_packed_f64_neon`            | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_symmetric_f64_neon`         | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_packed_f64_neon`          | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_symmetric_f64_neon`       | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_packed_f64_smef64`          | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_symmetric_f64_smef64`       | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_packed_f64_smef64`        | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_symmetric_f64_smef64`     | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| __f32__                                  |                 |                 |                 |
| `nk_angulars_packed_f32_serial`          | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_symmetric_f32_serial`       | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_packed_f32_serial`        | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_symmetric_f32_serial`     | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_packed_f32_neon`            | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_symmetric_f32_neon`         | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_packed_f32_neon`          | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_symmetric_f32_neon`       | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_packed_f32_smef64`          | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_symmetric_f32_smef64`       | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_packed_f32_smef64`        | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_symmetric_f32_smef64`     | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| __bf16__                                 |                 |                 |                 |
| `nk_angulars_packed_bf16_serial`         | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_symmetric_bf16_serial`      | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_packed_bf16_serial`       | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_symmetric_bf16_serial`    | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_packed_bf16_neonbfdot`      | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_symmetric_bf16_neonbfdot`   | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_packed_bf16_neonbfdot`    | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_symmetric_bf16_neonbfdot` | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_packed_bf16_sme`            | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_symmetric_bf16_sme`         | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_packed_bf16_sme`          | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_symmetric_bf16_sme`       | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| __f16__                                  |                 |                 |                 |
| `nk_angulars_packed_f16_serial`          | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_symmetric_f16_serial`       | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_packed_f16_serial`        | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_symmetric_f16_serial`     | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_packed_f16_neonhalf`        | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_symmetric_f16_neonhalf`     | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_packed_f16_neonhalf`      | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_symmetric_f16_neonhalf`   | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_packed_f16_neonfhm`         | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_symmetric_f16_neonfhm`      | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_packed_f16_neonfhm`       | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_symmetric_f16_neonfhm`    | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_packed_f16_sme`             | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_symmetric_f16_sme`          | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_packed_f16_sme`           | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_symmetric_f16_sme`        | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| __e5m2__                                 |                 |                 |                 |
| `nk_angulars_packed_e5m2_serial`         | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_symmetric_e5m2_serial`      | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_packed_e5m2_serial`       | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_symmetric_e5m2_serial`    | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_packed_e5m2_neonfhm`        | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_symmetric_e5m2_neonfhm`     | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_packed_e5m2_neonfhm`      | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_symmetric_e5m2_neonfhm`   | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_packed_e5m2_sme`            | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_symmetric_e5m2_sme`         | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_packed_e5m2_sme`          | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_symmetric_e5m2_sme`       | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| __e4m3__                                 |                 |                 |                 |
| `nk_angulars_packed_e4m3_serial`         | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_symmetric_e4m3_serial`      | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_packed_e4m3_serial`       | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_symmetric_e4m3_serial`    | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_packed_e4m3_neonfhm`        | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_symmetric_e4m3_neonfhm`     | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_packed_e4m3_neonfhm`      | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_symmetric_e4m3_neonfhm`   | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_packed_e4m3_sme`            | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_symmetric_e4m3_sme`         | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_packed_e4m3_sme`          | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_symmetric_e4m3_sme`       | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| __e3m2__                                 |                 |                 |                 |
| `nk_angulars_packed_e3m2_serial`         | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_symmetric_e3m2_serial`      | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_packed_e3m2_serial`       | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_symmetric_e3m2_serial`    | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_packed_e3m2_neonfhm`        | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_symmetric_e3m2_neonfhm`     | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_packed_e3m2_neonfhm`      | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_symmetric_e3m2_neonfhm`   | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_packed_e3m2_sme`            | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_symmetric_e3m2_sme`         | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_packed_e3m2_sme`          | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_symmetric_e3m2_sme`       | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| __e2m3__                                 |                 |                 |                 |
| `nk_angulars_packed_e2m3_serial`         | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_symmetric_e2m3_serial`      | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_packed_e2m3_serial`       | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_symmetric_e2m3_serial`    | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_packed_e2m3_neonfhm`        | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_symmetric_e2m3_neonfhm`     | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_packed_e2m3_neonfhm`      | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_symmetric_e2m3_neonfhm`   | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_packed_e2m3_sme`            | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_symmetric_e2m3_sme`         | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_packed_e2m3_sme`          | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_symmetric_e2m3_sme`       | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| __i8__                                   |                 |                 |                 |
| `nk_angulars_packed_i8_serial`           | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_symmetric_i8_serial`        | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_packed_i8_serial`         | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_symmetric_i8_serial`      | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_packed_i8_neonsdot`         | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_symmetric_i8_neonsdot`      | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_packed_i8_neonsdot`       | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_symmetric_i8_neonsdot`    | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_packed_i8_sme`              | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_symmetric_i8_sme`           | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_packed_i8_sme`            | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_symmetric_i8_sme`         | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| __u8__                                   |                 |                 |                 |
| `nk_angulars_packed_u8_serial`           | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_symmetric_u8_serial`        | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_packed_u8_serial`         | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_symmetric_u8_serial`      | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_packed_u8_neonsdot`         | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_symmetric_u8_neonsdot`      | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_packed_u8_neonsdot`       | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_symmetric_u8_neonsdot`    | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_packed_u8_sme`              | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_symmetric_u8_sme`           | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_packed_u8_sme`            | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_symmetric_u8_sme`         | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| __i4__                                   |                 |                 |                 |
| `nk_angulars_packed_i4_serial`           | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_symmetric_i4_serial`        | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_packed_i4_serial`         | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_symmetric_i4_serial`      | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_packed_i4_neonsdot`         | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_symmetric_i4_neonsdot`      | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_packed_i4_neonsdot`       | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_symmetric_i4_neonsdot`    | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_packed_i4_sme`              | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_symmetric_i4_sme`           | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_packed_i4_sme`            | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_symmetric_i4_sme`         | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| __u4__                                   |                 |                 |                 |
| `nk_angulars_packed_u4_serial`           | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_symmetric_u4_serial`        | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_packed_u4_serial`         | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_symmetric_u4_serial`      | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_packed_u4_neonsdot`         | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_symmetric_u4_neonsdot`      | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_packed_u4_neonsdot`       | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_symmetric_u4_neonsdot`    | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_packed_u4_sme`              | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_symmetric_u4_sme`           | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_packed_u4_sme`            | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_symmetric_u4_sme`         | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |

#### V8 (Chromium)

| Kernel                                     |            256³ |           1024³ |           4096³ |
| :----------------------------------------- | --------------: | --------------: | --------------: |
| __f64__                                    |                 |                 |                 |
| `nk_angulars_packed_f64_v128relaxed`       | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_symmetric_f64_v128relaxed`    | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_packed_f64_v128relaxed`     | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_symmetric_f64_v128relaxed`  | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| __f32__                                    |                 |                 |                 |
| `nk_angulars_packed_f32_v128relaxed`       | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_symmetric_f32_v128relaxed`    | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_packed_f32_v128relaxed`     | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_symmetric_f32_v128relaxed`  | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| __bf16__                                   |                 |                 |                 |
| `nk_angulars_packed_bf16_v128relaxed`      | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_symmetric_bf16_v128relaxed`   | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_packed_bf16_v128relaxed`    | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_symmetric_bf16_v128relaxed` | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| __e2m3__                                   |                 |                 |                 |
| `nk_angulars_packed_e2m3_v128relaxed`      | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_symmetric_e2m3_v128relaxed`   | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_packed_e2m3_v128relaxed`    | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_symmetric_e2m3_v128relaxed` | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| __i8__                                     |                 |                 |                 |
| `nk_angulars_packed_i8_v128relaxed`        | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_symmetric_i8_v128relaxed`     | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_packed_i8_v128relaxed`      | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_symmetric_i8_v128relaxed`   | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| __u8__                                     |                 |                 |                 |
| `nk_angulars_packed_u8_v128relaxed`        | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_symmetric_u8_v128relaxed`     | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_packed_u8_v128relaxed`      | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_symmetric_u8_v128relaxed`   | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |

#### Wasmtime (Cranelift)

| Kernel                                     |            256³ |           1024³ |           4096³ |
| :----------------------------------------- | --------------: | --------------: | --------------: |
| __f64__                                    |                 |                 |                 |
| `nk_angulars_packed_f64_v128relaxed`       | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_symmetric_f64_v128relaxed`    | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_packed_f64_v128relaxed`     | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_symmetric_f64_v128relaxed`  | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| __f32__                                    |                 |                 |                 |
| `nk_angulars_packed_f32_v128relaxed`       | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_symmetric_f32_v128relaxed`    | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_packed_f32_v128relaxed`     | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_symmetric_f32_v128relaxed`  | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| __bf16__                                   |                 |                 |                 |
| `nk_angulars_packed_bf16_v128relaxed`      | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_symmetric_bf16_v128relaxed`   | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_packed_bf16_v128relaxed`    | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_symmetric_bf16_v128relaxed` | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| __e2m3__                                   |                 |                 |                 |
| `nk_angulars_packed_e2m3_v128relaxed`      | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_symmetric_e2m3_v128relaxed`   | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_packed_e2m3_v128relaxed`    | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_symmetric_e2m3_v128relaxed` | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| __i8__                                     |                 |                 |                 |
| `nk_angulars_packed_i8_v128relaxed`        | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_symmetric_i8_v128relaxed`     | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_packed_i8_v128relaxed`      | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_symmetric_i8_v128relaxed`   | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| __u8__                                     |                 |                 |                 |
| `nk_angulars_packed_u8_v128relaxed`        | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_angulars_symmetric_u8_v128relaxed`     | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_packed_u8_v128relaxed`      | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_euclideans_symmetric_u8_v128relaxed`   | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |

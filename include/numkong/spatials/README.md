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

### Intel Sapphire Rapids

#### Native

<table>
<tr>
  <th>Kernel</th>
  <th>256³</th>
  <th>1024³</th>
  <th>4096³</th>
</tr>
<tr><td colspan="4"><b>f64</b></td></tr>
<tr>
  <td><code>nk_angulars_packed_f64_serial</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_angulars_symmetric_f64_serial</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_euclideans_packed_f64_serial</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_euclideans_symmetric_f64_serial</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_angulars_packed_f64_haswell</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_angulars_symmetric_f64_haswell</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_euclideans_packed_f64_haswell</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_euclideans_symmetric_f64_haswell</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_angulars_packed_f64_skylake</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_angulars_symmetric_f64_skylake</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_euclideans_packed_f64_skylake</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_euclideans_symmetric_f64_skylake</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_angulars_packed_f64_sapphireamx</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_angulars_symmetric_f64_sapphireamx</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_euclideans_packed_f64_sapphireamx</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_euclideans_symmetric_f64_sapphireamx</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>f32</b></td></tr>
<tr>
  <td><code>nk_angulars_packed_f32_serial</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_angulars_symmetric_f32_serial</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_euclideans_packed_f32_serial</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_euclideans_symmetric_f32_serial</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_angulars_packed_f32_haswell</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_angulars_symmetric_f32_haswell</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_euclideans_packed_f32_haswell</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_euclideans_symmetric_f32_haswell</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_angulars_packed_f32_skylake</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_angulars_symmetric_f32_skylake</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_euclideans_packed_f32_skylake</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_euclideans_symmetric_f32_skylake</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_angulars_packed_f32_icelake</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_angulars_symmetric_f32_icelake</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_euclideans_packed_f32_icelake</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_euclideans_symmetric_f32_icelake</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_angulars_packed_f32_sapphireamx</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_angulars_symmetric_f32_sapphireamx</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_euclideans_packed_f32_sapphireamx</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_euclideans_symmetric_f32_sapphireamx</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>bf16</b></td></tr>
<tr>
  <td><code>nk_angulars_packed_bf16_serial</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_angulars_symmetric_bf16_serial</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_euclideans_packed_bf16_serial</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_euclideans_symmetric_bf16_serial</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_angulars_packed_bf16_haswell</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_angulars_symmetric_bf16_haswell</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_euclideans_packed_bf16_haswell</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_euclideans_symmetric_bf16_haswell</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_angulars_packed_bf16_skylake</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_angulars_symmetric_bf16_skylake</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_euclideans_packed_bf16_skylake</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_euclideans_symmetric_bf16_skylake</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_angulars_packed_bf16_genoa</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_angulars_symmetric_bf16_genoa</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_euclideans_packed_bf16_genoa</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_euclideans_symmetric_bf16_genoa</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_angulars_packed_bf16_sapphireamx</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_angulars_symmetric_bf16_sapphireamx</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_euclideans_packed_bf16_sapphireamx</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_euclideans_symmetric_bf16_sapphireamx</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>f16</b></td></tr>
<tr>
  <td><code>nk_angulars_packed_f16_serial</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_angulars_symmetric_f16_serial</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_euclideans_packed_f16_serial</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_euclideans_symmetric_f16_serial</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_angulars_packed_f16_haswell</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_angulars_symmetric_f16_haswell</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_euclideans_packed_f16_haswell</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_euclideans_symmetric_f16_haswell</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_angulars_packed_f16_skylake</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_angulars_symmetric_f16_skylake</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_euclideans_packed_f16_skylake</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_euclideans_symmetric_f16_skylake</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>e4m3</b></td></tr>
<tr>
  <td><code>nk_angulars_packed_e4m3_serial</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_angulars_symmetric_e4m3_serial</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_euclideans_packed_e4m3_serial</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_euclideans_symmetric_e4m3_serial</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_angulars_packed_e4m3_haswell</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_angulars_symmetric_e4m3_haswell</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_euclideans_packed_e4m3_haswell</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_euclideans_symmetric_e4m3_haswell</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_angulars_packed_e4m3_skylake</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_angulars_symmetric_e4m3_skylake</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_euclideans_packed_e4m3_skylake</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_euclideans_symmetric_e4m3_skylake</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
</table>

#### V8 (Chromium)

<table>
<tr>
  <th>Kernel</th>
  <th>256³</th>
  <th>1024³</th>
  <th>4096³</th>
</tr>
<tr><td colspan="4"><b>f64</b></td></tr>
<tr>
  <td><code>nk_angulars_packed_f64_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_angulars_symmetric_f64_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_euclideans_packed_f64_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_euclideans_symmetric_f64_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>f32</b></td></tr>
<tr>
  <td><code>nk_angulars_packed_f32_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_angulars_symmetric_f32_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_euclideans_packed_f32_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_euclideans_symmetric_f32_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>bf16</b></td></tr>
<tr>
  <td><code>nk_angulars_packed_bf16_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_angulars_symmetric_bf16_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_euclideans_packed_bf16_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_euclideans_symmetric_bf16_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>f16</b></td></tr>
<tr>
  <td><code>nk_angulars_packed_f16_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_angulars_symmetric_f16_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_euclideans_packed_f16_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_euclideans_symmetric_f16_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
</table>

#### Wasmtime (Cranelift)

<table>
<tr>
  <th>Kernel</th>
  <th>256³</th>
  <th>1024³</th>
  <th>4096³</th>
</tr>
<tr><td colspan="4"><b>f64</b></td></tr>
<tr>
  <td><code>nk_angulars_packed_f64_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_angulars_symmetric_f64_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_euclideans_packed_f64_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_euclideans_symmetric_f64_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>f32</b></td></tr>
<tr>
  <td><code>nk_angulars_packed_f32_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_angulars_symmetric_f32_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_euclideans_packed_f32_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_euclideans_symmetric_f32_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>bf16</b></td></tr>
<tr>
  <td><code>nk_angulars_packed_bf16_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_angulars_symmetric_bf16_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_euclideans_packed_bf16_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_euclideans_symmetric_bf16_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>f16</b></td></tr>
<tr>
  <td><code>nk_angulars_packed_f16_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_angulars_symmetric_f16_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_euclideans_packed_f16_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_euclideans_symmetric_f16_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
</table>

### Apple M4 Pro

#### Native

<table>
<tr>
  <th>Kernel</th>
  <th>256³</th>
  <th>1024³</th>
  <th>4096³</th>
</tr>
<tr><td colspan="4"><b>f64</b></td></tr>
<tr>
  <td><code>nk_angulars_packed_f64_serial</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_angulars_symmetric_f64_serial</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_euclideans_packed_f64_serial</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_euclideans_symmetric_f64_serial</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_angulars_packed_f64_neon</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_angulars_symmetric_f64_neon</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_euclideans_packed_f64_neon</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_euclideans_symmetric_f64_neon</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_angulars_packed_f64_sme</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_angulars_symmetric_f64_sme</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_euclideans_packed_f64_sme</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_euclideans_symmetric_f64_sme</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>f32</b></td></tr>
<tr>
  <td><code>nk_angulars_packed_f32_serial</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_angulars_symmetric_f32_serial</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_euclideans_packed_f32_serial</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_euclideans_symmetric_f32_serial</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_angulars_packed_f32_neon</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_angulars_symmetric_f32_neon</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_euclideans_packed_f32_neon</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_euclideans_symmetric_f32_neon</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_angulars_packed_f32_sme</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_angulars_symmetric_f32_sme</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_euclideans_packed_f32_sme</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_euclideans_symmetric_f32_sme</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>bf16</b></td></tr>
<tr>
  <td><code>nk_angulars_packed_bf16_serial</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_angulars_symmetric_bf16_serial</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_euclideans_packed_bf16_serial</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_euclideans_symmetric_bf16_serial</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_angulars_packed_bf16_neonbfdot</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_angulars_symmetric_bf16_neonbfdot</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_euclideans_packed_bf16_neonbfdot</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_euclideans_symmetric_bf16_neonbfdot</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_angulars_packed_bf16_smehalf</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_angulars_symmetric_bf16_smehalf</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_euclideans_packed_bf16_smehalf</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_euclideans_symmetric_bf16_smehalf</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>f16</b></td></tr>
<tr>
  <td><code>nk_angulars_packed_f16_serial</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_angulars_symmetric_f16_serial</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_euclideans_packed_f16_serial</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_euclideans_symmetric_f16_serial</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_angulars_packed_f16_neon</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_angulars_symmetric_f16_neon</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_euclideans_packed_f16_neon</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_euclideans_symmetric_f16_neon</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_angulars_packed_f16_neonfhm</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_angulars_symmetric_f16_neonfhm</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_euclideans_packed_f16_neonfhm</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_euclideans_symmetric_f16_neonfhm</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>e4m3</b></td></tr>
<tr>
  <td><code>nk_angulars_packed_e4m3_serial</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_angulars_symmetric_e4m3_serial</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_euclideans_packed_e4m3_serial</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_euclideans_symmetric_e4m3_serial</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_angulars_packed_e4m3_neon</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_angulars_symmetric_e4m3_neon</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_euclideans_packed_e4m3_neon</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_euclideans_symmetric_e4m3_neon</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
</table>

#### V8 (Chromium)

<table>
<tr>
  <th>Kernel</th>
  <th>256³</th>
  <th>1024³</th>
  <th>4096³</th>
</tr>
<tr><td colspan="4"><b>f64</b></td></tr>
<tr>
  <td><code>nk_angulars_packed_f64_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_angulars_symmetric_f64_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_euclideans_packed_f64_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_euclideans_symmetric_f64_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>f32</b></td></tr>
<tr>
  <td><code>nk_angulars_packed_f32_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_angulars_symmetric_f32_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_euclideans_packed_f32_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_euclideans_symmetric_f32_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>bf16</b></td></tr>
<tr>
  <td><code>nk_angulars_packed_bf16_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_angulars_symmetric_bf16_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_euclideans_packed_bf16_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_euclideans_symmetric_bf16_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>f16</b></td></tr>
<tr>
  <td><code>nk_angulars_packed_f16_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_angulars_symmetric_f16_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_euclideans_packed_f16_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_euclideans_symmetric_f16_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
</table>

#### Wasmtime (Cranelift)

<table>
<tr>
  <th>Kernel</th>
  <th>256³</th>
  <th>1024³</th>
  <th>4096³</th>
</tr>
<tr><td colspan="4"><b>f64</b></td></tr>
<tr>
  <td><code>nk_angulars_packed_f64_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_angulars_symmetric_f64_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_euclideans_packed_f64_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_euclideans_symmetric_f64_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>f32</b></td></tr>
<tr>
  <td><code>nk_angulars_packed_f32_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_angulars_symmetric_f32_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_euclideans_packed_f32_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_euclideans_symmetric_f32_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>bf16</b></td></tr>
<tr>
  <td><code>nk_angulars_packed_bf16_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_angulars_symmetric_bf16_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_euclideans_packed_bf16_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_euclideans_symmetric_bf16_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>f16</b></td></tr>
<tr>
  <td><code>nk_angulars_packed_f16_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_angulars_symmetric_f16_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_euclideans_packed_f16_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_euclideans_symmetric_f16_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
</table>

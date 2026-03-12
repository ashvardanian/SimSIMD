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

The following performance tables are produced by manually re-running `nk_test` and `nk_bench` included internal tools to measure both accuracy and throughput at different input shapes.
The input size is controlled by `NK_MATRIX_HEIGHT`, `NK_MATRIX_WIDTH`, and `NK_MATRIX_DEPTH` environment variables, all set to the same value for batched distance computations over square matrices.
Columns show throughput for 256³, 1024³, and 4096³ configurations.
The throughput is measured in GSO/s as Giga scalar operations per second, with $\text{ops} = 2 \cdot M \cdot N \cdot K$ complexity for computing $M \times N$ pairwise distances over $K$-dimensional vectors.
Accuracy is reported as ULP (units in last place), the number of representable floating-point values between the result and the exact answer.
Each kernel runs for at least 20 seconds per configuration.
Benchmark threads are pinned to specific cores; on machines with heterogeneous core types (e.g., Apple P/E cores), only the fastest cores are used.
Workloads that significantly degrade CPU frequencies (Intel AMX, Apple SME) run in separate passes to avoid affecting throughput measurements of other kernels.

### Intel Sapphire Rapids

#### Native

| Kernel                                     |                     256³ |                    1024³ |                    4096³ |
| :----------------------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f64__                                    | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_f64_serial`            |       0.578 gso/s, 0 ulp |       0.691 gso/s, 0 ulp |       0.787 gso/s, 0 ulp |
| `nk_angulars_symmetric_f64_serial`         |       0.477 gso/s, 0 ulp |       0.569 gso/s, 0 ulp |        1.24 gso/s, 0 ulp |
| `nk_euclideans_packed_f64_serial`          |     0.569 gso/s, 0.6 ulp |     0.692 gso/s, 0.6 ulp |     0.775 gso/s, 0.6 ulp |
| `nk_euclideans_symmetric_f64_serial`       |     0.477 gso/s, 0.6 ulp |     0.562 gso/s, 0.6 ulp |      1.26 gso/s, 0.3 ulp |
| `nk_angulars_packed_f64_haswell`           |        5.61 gso/s, 0 ulp |        5.90 gso/s, 0 ulp |        2.97 gso/s, 0 ulp |
| `nk_angulars_symmetric_f64_haswell`        |        5.32 gso/s, 0 ulp |        5.71 gso/s, 0 ulp |        6.34 gso/s, 0 ulp |
| `nk_euclideans_packed_f64_haswell`         |      5.72 gso/s, 0.2 ulp |      6.00 gso/s, 0.2 ulp |      5.96 gso/s, 0.2 ulp |
| `nk_euclideans_symmetric_f64_haswell`      |      5.35 gso/s, 0.2 ulp |      5.88 gso/s, 0.2 ulp |      11.6 gso/s, 0.2 ulp |
| `nk_angulars_packed_f64_skylake`           |        7.61 gso/s, 0 ulp |        4.84 gso/s, 0 ulp |        9.10 gso/s, 0 ulp |
| `nk_angulars_symmetric_f64_skylake`        |        4.57 gso/s, 0 ulp |        5.31 gso/s, 0 ulp |        15.9 gso/s, 0 ulp |
| `nk_euclideans_packed_f64_skylake`         |      4.76 gso/s, 0.2 ulp |      4.93 gso/s, 0.2 ulp |      8.66 gso/s, 0.2 ulp |
| `nk_euclideans_symmetric_f64_skylake`      |      4.47 gso/s, 0.2 ulp |      4.81 gso/s, 0.2 ulp |      13.5 gso/s, 0.2 ulp |
| __f32__                                    | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_f32_serial`            |      15.0 gso/s, 0.1 ulp |      16.3 gso/s, 0.1 ulp |      16.4 gso/s, 0.1 ulp |
| `nk_angulars_symmetric_f32_serial`         |      3.86 gso/s, 0.1 ulp |      4.29 gso/s, 0.1 ulp |      8.62 gso/s, 0.1 ulp |
| `nk_euclideans_packed_f32_serial`          |      15.3 gso/s, 0.6 ulp |      17.0 gso/s, 0.5 ulp |      17.0 gso/s, 0.5 ulp |
| `nk_euclideans_symmetric_f32_serial`       |      3.97 gso/s, 0.6 ulp |      4.16 gso/s, 0.5 ulp |      8.38 gso/s, 0.3 ulp |
| `nk_angulars_packed_f32_haswell`           |        29.3 gso/s, 0 ulp |        31.8 gso/s, 0 ulp |        21.3 gso/s, 0 ulp |
| `nk_angulars_symmetric_f32_haswell`        |        15.7 gso/s, 0 ulp |        17.4 gso/s, 0 ulp |        22.1 gso/s, 0 ulp |
| `nk_euclideans_packed_f32_haswell`         |      29.1 gso/s, 0.2 ulp |      31.6 gso/s, 0.2 ulp |      31.2 gso/s, 0.2 ulp |
| `nk_euclideans_symmetric_f32_haswell`      |      15.8 gso/s, 0.2 ulp |      17.5 gso/s, 0.2 ulp |      35.9 gso/s, 0.2 ulp |
| `nk_angulars_packed_f32_skylake`           |        34.3 gso/s, 0 ulp |        23.9 gso/s, 0 ulp |        39.5 gso/s, 0 ulp |
| `nk_angulars_symmetric_f32_skylake`        |        15.1 gso/s, 0 ulp |        18.7 gso/s, 0 ulp |        52.8 gso/s, 0 ulp |
| `nk_euclideans_packed_f32_skylake`         |      18.9 gso/s, 0.2 ulp |      22.1 gso/s, 0.2 ulp |      38.1 gso/s, 0.2 ulp |
| `nk_euclideans_symmetric_f32_skylake`      |      14.0 gso/s, 0.2 ulp |      17.3 gso/s, 0.2 ulp |      51.3 gso/s, 0.2 ulp |
| __bf16__                                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_bf16_serial`           |        1.18 gso/s, 0 ulp |        1.21 gso/s, 0 ulp |      1.19 gso/s, 0.1 ulp |
| `nk_angulars_symmetric_bf16_serial`        |        1.19 gso/s, 0 ulp |        1.18 gso/s, 0 ulp |        2.35 gso/s, 0 ulp |
| `nk_euclideans_packed_bf16_serial`         |      1.20 gso/s, 0.6 ulp |      1.18 gso/s, 0.6 ulp |      1.16 gso/s, 6.0 ulp |
| `nk_euclideans_symmetric_bf16_serial`      |      1.11 gso/s, 0.6 ulp |      1.14 gso/s, 0.6 ulp |      2.34 gso/s, 0.4 ulp |
| `nk_angulars_packed_bf16_haswell`          |        52.3 gso/s, 0 ulp |        64.6 gso/s, 0 ulp |      35.8 gso/s, 0.1 ulp |
| `nk_angulars_symmetric_bf16_haswell`       |        29.8 gso/s, 0 ulp |        36.1 gso/s, 0 ulp |        72.8 gso/s, 0 ulp |
| `nk_euclideans_packed_bf16_haswell`        |      53.6 gso/s, 0.2 ulp |      66.3 gso/s, 0.3 ulp |      69.0 gso/s, 5.8 ulp |
| `nk_euclideans_symmetric_bf16_haswell`     |      29.5 gso/s, 0.2 ulp |      36.3 gso/s, 0.3 ulp |      75.1 gso/s, 0.3 ulp |
| `nk_angulars_packed_bf16_skylake`          |        68.2 gso/s, 0 ulp |        59.9 gso/s, 0 ulp |      90.9 gso/s, 0.1 ulp |
| `nk_angulars_symmetric_bf16_skylake`       |        24.0 gso/s, 0 ulp |        30.7 gso/s, 0 ulp |         112 gso/s, 0 ulp |
| `nk_euclideans_packed_bf16_skylake`        |      48.0 gso/s, 0.2 ulp |      62.0 gso/s, 0.3 ulp |      81.3 gso/s, 5.8 ulp |
| `nk_euclideans_symmetric_bf16_skylake`     |      25.5 gso/s, 0.2 ulp |      30.6 gso/s, 0.3 ulp |       103 gso/s, 0.3 ulp |
| `nk_angulars_packed_bf16_genoa`            |        38.8 gso/s, 0 ulp |        55.2 gso/s, 0 ulp |        59.2 gso/s, 0 ulp |
| `nk_angulars_symmetric_bf16_genoa`         |        31.4 gso/s, 0 ulp |        36.2 gso/s, 0 ulp |        92.7 gso/s, 0 ulp |
| `nk_euclideans_packed_bf16_genoa`          |      38.6 gso/s, 0.2 ulp |      60.8 gso/s, 0.3 ulp |      54.5 gso/s, 0.3 ulp |
| `nk_euclideans_symmetric_bf16_genoa`       |      34.7 gso/s, 0.2 ulp |      46.6 gso/s, 0.3 ulp |       152 gso/s, 0.3 ulp |
| `nk_angulars_packed_bf16_sapphireamx`      |         326 gso/s, 0 ulp |         583 gso/s, 0 ulp |         651 gso/s, 0 ulp |
| `nk_angulars_symmetric_bf16_sapphireamx`   |        79.1 gso/s, 0 ulp |         117 gso/s, 0 ulp |         131 gso/s, 0 ulp |
| `nk_euclideans_packed_bf16_sapphireamx`    |       319 gso/s, 0.3 ulp |       601 gso/s, 0.3 ulp |       681 gso/s, 0.3 ulp |
| `nk_euclideans_symmetric_bf16_sapphireamx` |      75.0 gso/s, 0.3 ulp |       118 gso/s, 0.3 ulp |       126 gso/s, 0.3 ulp |
| __f16__                                    | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_f16_serial`            |      7.46 gso/s, 0.1 ulp |      7.97 gso/s, 0.1 ulp |      8.12 gso/s, 0.1 ulp |
| `nk_angulars_symmetric_f16_serial`         |      4.04 gso/s, 0.1 ulp |      4.09 gso/s, 0.1 ulp |      8.13 gso/s, 0.1 ulp |
| `nk_euclideans_packed_f16_serial`          |      7.69 gso/s, 0.7 ulp |      7.73 gso/s, 1.1 ulp |      8.34 gso/s, 0.6 ulp |
| `nk_euclideans_symmetric_f16_serial`       |      4.08 gso/s, 0.7 ulp |      4.19 gso/s, 1.1 ulp |      8.23 gso/s, 0.5 ulp |
| `nk_angulars_packed_f16_haswell`           |      58.9 gso/s, 0.1 ulp |      70.4 gso/s, 0.1 ulp |      38.6 gso/s, 0.1 ulp |
| `nk_angulars_symmetric_f16_haswell`        |      30.7 gso/s, 0.1 ulp |      42.4 gso/s, 0.1 ulp |      88.6 gso/s, 0.1 ulp |
| `nk_euclideans_packed_f16_haswell`         |      60.8 gso/s, 0.4 ulp |      71.3 gso/s, 0.9 ulp |      71.8 gso/s, 0.5 ulp |
| `nk_euclideans_symmetric_f16_haswell`      |      31.7 gso/s, 0.4 ulp |      40.0 gso/s, 0.9 ulp |      88.1 gso/s, 0.3 ulp |
| `nk_angulars_packed_f16_skylake`           |      63.7 gso/s, 0.1 ulp |      54.1 gso/s, 0.1 ulp |        93.2 gso/s, 0 ulp |
| `nk_angulars_symmetric_f16_skylake`        |      22.9 gso/s, 0.1 ulp |      32.4 gso/s, 0.1 ulp |         112 gso/s, 0 ulp |
| `nk_euclideans_packed_f16_skylake`         |      39.2 gso/s, 0.4 ulp |      62.5 gso/s, 0.9 ulp |      83.6 gso/s, 0.5 ulp |
| `nk_euclideans_symmetric_f16_skylake`      |      23.5 gso/s, 0.4 ulp |      33.5 gso/s, 0.9 ulp |      99.3 gso/s, 0.3 ulp |
| __e5m2__                                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_e5m2_serial`           |       0.587 gso/s, 0 ulp |       0.553 gso/s, 0 ulp |       0.563 gso/s, 0 ulp |
| `nk_angulars_symmetric_e5m2_serial`        |       0.446 gso/s, 0 ulp |       0.427 gso/s, 0 ulp |       0.847 gso/s, 0 ulp |
| `nk_euclideans_packed_e5m2_serial`         |     0.576 gso/s, 0.5 ulp |     0.571 gso/s, 0.5 ulp |     0.557 gso/s, 0.2 ulp |
| `nk_euclideans_symmetric_e5m2_serial`      |     0.424 gso/s, 0.5 ulp |     0.437 gso/s, 0.5 ulp |     0.836 gso/s, 0.2 ulp |
| `nk_angulars_packed_e5m2_haswell`          |        27.3 gso/s, 0 ulp |        29.9 gso/s, 0 ulp |        12.1 gso/s, 0 ulp |
| `nk_angulars_symmetric_e5m2_haswell`       |        15.1 gso/s, 0 ulp |        16.3 gso/s, 0 ulp |        32.8 gso/s, 0 ulp |
| `nk_euclideans_packed_e5m2_haswell`        |        27.3 gso/s, 0 ulp |        28.8 gso/s, 0 ulp |        29.6 gso/s, 0 ulp |
| `nk_euclideans_symmetric_e5m2_haswell`     |        14.7 gso/s, 0 ulp |        15.3 gso/s, 0 ulp |        33.4 gso/s, 0 ulp |
| `nk_angulars_packed_e5m2_skylake`          |        18.1 gso/s, 0 ulp |        22.8 gso/s, 0 ulp |        36.6 gso/s, 0 ulp |
| `nk_angulars_symmetric_e5m2_skylake`       |        12.0 gso/s, 0 ulp |        12.6 gso/s, 0 ulp |        41.9 gso/s, 0 ulp |
| `nk_euclideans_packed_e5m2_skylake`        |        18.6 gso/s, 0 ulp |        20.8 gso/s, 0 ulp |        36.3 gso/s, 0 ulp |
| `nk_euclideans_symmetric_e5m2_skylake`     |        12.0 gso/s, 0 ulp |        13.6 gso/s, 0 ulp |        32.6 gso/s, 0 ulp |
| `nk_angulars_packed_e5m2_genoa`            |        24.6 gso/s, 0 ulp |        29.5 gso/s, 0 ulp |        34.7 gso/s, 0 ulp |
| `nk_angulars_symmetric_e5m2_genoa`         |        16.0 gso/s, 0 ulp |        19.2 gso/s, 0 ulp |        51.7 gso/s, 0 ulp |
| `nk_euclideans_packed_e5m2_genoa`          |        27.0 gso/s, 0 ulp |        33.5 gso/s, 0 ulp |        34.2 gso/s, 0 ulp |
| `nk_euclideans_symmetric_e5m2_genoa`       |        16.6 gso/s, 0 ulp |        19.9 gso/s, 0 ulp |        67.2 gso/s, 0 ulp |
| `nk_angulars_packed_e5m2_sapphireamx`      |         216 gso/s, 0 ulp |         382 gso/s, 0 ulp |         400 gso/s, 0 ulp |
| `nk_angulars_symmetric_e5m2_sapphireamx`   |        48.8 gso/s, 0 ulp |        70.6 gso/s, 0 ulp |        72.9 gso/s, 0 ulp |
| `nk_euclideans_packed_e5m2_sapphireamx`    |         227 gso/s, 0 ulp |         390 gso/s, 0 ulp |         425 gso/s, 0 ulp |
| `nk_euclideans_symmetric_e5m2_sapphireamx` |        49.3 gso/s, 0 ulp |        70.5 gso/s, 0 ulp |        70.8 gso/s, 0 ulp |
| __e4m3__                                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_e4m3_serial`           |       0.479 gso/s, 0 ulp |       0.473 gso/s, 0 ulp |       0.485 gso/s, 0 ulp |
| `nk_angulars_symmetric_e4m3_serial`        |       0.395 gso/s, 0 ulp |       0.390 gso/s, 0 ulp |       0.795 gso/s, 0 ulp |
| `nk_euclideans_packed_e4m3_serial`         |     0.467 gso/s, 0.5 ulp |     0.484 gso/s, 0.5 ulp |     0.480 gso/s, 0.5 ulp |
| `nk_euclideans_symmetric_e4m3_serial`      |     0.395 gso/s, 0.5 ulp |     0.395 gso/s, 0.5 ulp |     0.781 gso/s, 0.3 ulp |
| `nk_angulars_packed_e4m3_haswell`          |        20.0 gso/s, 0 ulp |        22.0 gso/s, 0 ulp |        9.50 gso/s, 0 ulp |
| `nk_angulars_symmetric_e4m3_haswell`       |        11.0 gso/s, 0 ulp |        11.6 gso/s, 0 ulp |        22.6 gso/s, 0 ulp |
| `nk_euclideans_packed_e4m3_haswell`        |        20.6 gso/s, 0 ulp |        22.2 gso/s, 0 ulp |      21.8 gso/s, 0.2 ulp |
| `nk_euclideans_symmetric_e4m3_haswell`     |        11.0 gso/s, 0 ulp |        11.4 gso/s, 0 ulp |      22.4 gso/s, 0.1 ulp |
| `nk_angulars_packed_e4m3_skylake`          |        17.1 gso/s, 0 ulp |        20.4 gso/s, 0 ulp |        30.4 gso/s, 0 ulp |
| `nk_angulars_symmetric_e4m3_skylake`       |        9.31 gso/s, 0 ulp |        9.51 gso/s, 0 ulp |        34.2 gso/s, 0 ulp |
| `nk_euclideans_packed_e4m3_skylake`        |        18.1 gso/s, 0 ulp |        16.9 gso/s, 0 ulp |      29.1 gso/s, 0.2 ulp |
| `nk_euclideans_symmetric_e4m3_skylake`     |        8.32 gso/s, 0 ulp |        9.35 gso/s, 0 ulp |      28.7 gso/s, 0.1 ulp |
| `nk_angulars_packed_e4m3_genoa`            |        29.1 gso/s, 0 ulp |        30.1 gso/s, 0 ulp |        37.3 gso/s, 0 ulp |
| `nk_angulars_symmetric_e4m3_genoa`         |        17.1 gso/s, 0 ulp |        18.1 gso/s, 0 ulp |        46.4 gso/s, 0 ulp |
| `nk_euclideans_packed_e4m3_genoa`          |        25.8 gso/s, 0 ulp |        32.3 gso/s, 0 ulp |      35.8 gso/s, 0.1 ulp |
| `nk_euclideans_symmetric_e4m3_genoa`       |        17.6 gso/s, 0 ulp |        22.6 gso/s, 0 ulp |      68.4 gso/s, 0.1 ulp |
| `nk_angulars_packed_e4m3_sapphireamx`      |         221 gso/s, 0 ulp |         378 gso/s, 0 ulp |         404 gso/s, 0 ulp |
| `nk_angulars_symmetric_e4m3_sapphireamx`   |        49.3 gso/s, 0 ulp |        70.7 gso/s, 0 ulp |        71.6 gso/s, 0 ulp |
| `nk_euclideans_packed_e4m3_sapphireamx`    |       220 gso/s, 0.1 ulp |       391 gso/s, 0.1 ulp |       417 gso/s, 0.1 ulp |
| `nk_euclideans_symmetric_e4m3_sapphireamx` |      48.2 gso/s, 0.1 ulp |      70.6 gso/s, 0.1 ulp |      72.3 gso/s, 0.1 ulp |
| __e3m2__                                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_e3m2_serial`           |       0.554 gso/s, 0 ulp |       0.524 gso/s, 0 ulp |       0.534 gso/s, 0 ulp |
| `nk_angulars_symmetric_e3m2_serial`        |       0.439 gso/s, 0 ulp |       0.427 gso/s, 0 ulp |       0.839 gso/s, 0 ulp |
| `nk_euclideans_packed_e3m2_serial`         |     0.556 gso/s, 0.5 ulp |     0.549 gso/s, 0.5 ulp |     0.509 gso/s, 0.2 ulp |
| `nk_euclideans_symmetric_e3m2_serial`      |     0.413 gso/s, 0.5 ulp |     0.427 gso/s, 0.5 ulp |     0.829 gso/s, 0.2 ulp |
| `nk_angulars_packed_e3m2_haswell`          |        29.9 gso/s, 0 ulp |        31.0 gso/s, 0 ulp |        12.8 gso/s, 0 ulp |
| `nk_angulars_symmetric_e3m2_haswell`       |        26.9 gso/s, 0 ulp |        32.2 gso/s, 0 ulp |        63.1 gso/s, 0 ulp |
| `nk_euclideans_packed_e3m2_haswell`        |        28.8 gso/s, 0 ulp |        31.2 gso/s, 0 ulp |        32.5 gso/s, 0 ulp |
| `nk_euclideans_symmetric_e3m2_haswell`     |        28.0 gso/s, 0 ulp |        30.9 gso/s, 0 ulp |        58.5 gso/s, 0 ulp |
| `nk_angulars_packed_e3m2_skylake`          |        27.5 gso/s, 0 ulp |        22.8 gso/s, 0 ulp |        42.2 gso/s, 0 ulp |
| `nk_angulars_symmetric_e3m2_skylake`       |        21.0 gso/s, 0 ulp |        23.7 gso/s, 0 ulp |        86.5 gso/s, 0 ulp |
| `nk_euclideans_packed_e3m2_skylake`        |        24.4 gso/s, 0 ulp |        21.9 gso/s, 0 ulp |        38.1 gso/s, 0 ulp |
| `nk_euclideans_symmetric_e3m2_skylake`     |        21.3 gso/s, 0 ulp |        21.9 gso/s, 0 ulp |        78.8 gso/s, 0 ulp |
| `nk_angulars_packed_e3m2_genoa`            |        28.6 gso/s, 0 ulp |        44.6 gso/s, 0 ulp |        44.1 gso/s, 0 ulp |
| `nk_angulars_symmetric_e3m2_genoa`         |        20.3 gso/s, 0 ulp |        24.2 gso/s, 0 ulp |        58.5 gso/s, 0 ulp |
| `nk_euclideans_packed_e3m2_genoa`          |        32.1 gso/s, 0 ulp |        41.2 gso/s, 0 ulp |        57.9 gso/s, 0 ulp |
| `nk_euclideans_symmetric_e3m2_genoa`       |        20.9 gso/s, 0 ulp |        27.1 gso/s, 0 ulp |        86.8 gso/s, 0 ulp |
| `nk_angulars_packed_e3m2_sapphireamx`      |         247 gso/s, 0 ulp |         450 gso/s, 0 ulp |         489 gso/s, 0 ulp |
| `nk_angulars_symmetric_e3m2_sapphireamx`   |        61.2 gso/s, 0 ulp |         102 gso/s, 0 ulp |        94.1 gso/s, 0 ulp |
| `nk_euclideans_packed_e3m2_sapphireamx`    |         236 gso/s, 0 ulp |         459 gso/s, 0 ulp |         477 gso/s, 0 ulp |
| `nk_euclideans_symmetric_e3m2_sapphireamx` |        60.1 gso/s, 0 ulp |         103 gso/s, 0 ulp |        94.6 gso/s, 0 ulp |
| __e2m3__                                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_e2m3_serial`           |       0.332 gso/s, 0 ulp |       0.325 gso/s, 0 ulp |       0.320 gso/s, 0 ulp |
| `nk_angulars_symmetric_e2m3_serial`        |       0.298 gso/s, 0 ulp |       0.305 gso/s, 0 ulp |       0.568 gso/s, 0 ulp |
| `nk_euclideans_packed_e2m3_serial`         |     0.324 gso/s, 0.5 ulp |     0.310 gso/s, 0.5 ulp |     0.313 gso/s, 0.2 ulp |
| `nk_euclideans_symmetric_e2m3_serial`      |     0.293 gso/s, 0.5 ulp |     0.295 gso/s, 0.5 ulp |     0.586 gso/s, 0.2 ulp |
| `nk_angulars_packed_e2m3_haswell`          |        51.9 gso/s, 0 ulp |        59.5 gso/s, 0 ulp |        24.8 gso/s, 0 ulp |
| `nk_angulars_symmetric_e2m3_haswell`       |        46.9 gso/s, 0 ulp |        57.4 gso/s, 0 ulp |         122 gso/s, 0 ulp |
| `nk_euclideans_packed_e2m3_haswell`        |        55.6 gso/s, 0 ulp |        63.2 gso/s, 0 ulp |        63.6 gso/s, 0 ulp |
| `nk_euclideans_symmetric_e2m3_haswell`     |        44.7 gso/s, 0 ulp |        55.4 gso/s, 0 ulp |         110 gso/s, 0 ulp |
| `nk_angulars_packed_e2m3_skylake`          |        42.9 gso/s, 0 ulp |        52.4 gso/s, 0 ulp |        81.9 gso/s, 0 ulp |
| `nk_angulars_symmetric_e2m3_skylake`       |        29.8 gso/s, 0 ulp |        45.2 gso/s, 0 ulp |         158 gso/s, 0 ulp |
| `nk_euclideans_packed_e2m3_skylake`        |        38.6 gso/s, 0 ulp |        44.5 gso/s, 0 ulp |        76.3 gso/s, 0 ulp |
| `nk_euclideans_symmetric_e2m3_skylake`     |        32.4 gso/s, 0 ulp |        47.7 gso/s, 0 ulp |         126 gso/s, 0 ulp |
| `nk_angulars_packed_e2m3_genoa`            |        26.1 gso/s, 0 ulp |        37.1 gso/s, 0 ulp |        42.3 gso/s, 0 ulp |
| `nk_angulars_symmetric_e2m3_genoa`         |        20.0 gso/s, 0 ulp |        24.4 gso/s, 0 ulp |        61.8 gso/s, 0 ulp |
| `nk_euclideans_packed_e2m3_genoa`          |        33.0 gso/s, 0 ulp |        43.7 gso/s, 0 ulp |        56.9 gso/s, 0 ulp |
| `nk_euclideans_symmetric_e2m3_genoa`       |        21.6 gso/s, 0 ulp |        26.3 gso/s, 0 ulp |        87.3 gso/s, 0 ulp |
| `nk_angulars_packed_e2m3_sapphireamx`      |         366 gso/s, 0 ulp |         999 gso/s, 0 ulp |        1031 gso/s, 0 ulp |
| `nk_angulars_symmetric_e2m3_sapphireamx`   |        89.7 gso/s, 0 ulp |         209 gso/s, 0 ulp |         208 gso/s, 0 ulp |
| `nk_euclideans_packed_e2m3_sapphireamx`    |         346 gso/s, 0 ulp |         994 gso/s, 0 ulp |        1045 gso/s, 0 ulp |
| `nk_euclideans_symmetric_e2m3_sapphireamx` |        88.6 gso/s, 0 ulp |         215 gso/s, 0 ulp |         203 gso/s, 0 ulp |
| __i8__                                     | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_i8_serial`             |        8.84 gso/s, 0 ulp |        9.49 gso/s, 0 ulp |        10.1 gso/s, 0 ulp |
| `nk_angulars_symmetric_i8_serial`          |        4.40 gso/s, 0 ulp |        4.45 gso/s, 0 ulp |        9.58 gso/s, ? ulp |
| `nk_euclideans_packed_i8_serial`           |      8.64 gso/s, 0.4 ulp |      9.84 gso/s, 0.4 ulp |      9.94 gso/s, 0.4 ulp |
| `nk_euclideans_symmetric_i8_serial`        |      4.47 gso/s, 0.4 ulp |      4.64 gso/s, 0.4 ulp |        9.15 gso/s, ? ulp |
| `nk_angulars_packed_i8_haswell`            |        72.8 gso/s, 0 ulp |        94.0 gso/s, 0 ulp |         109 gso/s, 0 ulp |
| `nk_angulars_symmetric_i8_haswell`         |        51.1 gso/s, 0 ulp |        77.3 gso/s, 0 ulp |         159 gso/s, ? ulp |
| `nk_euclideans_packed_i8_haswell`          |        82.8 gso/s, 0 ulp |        94.3 gso/s, 0 ulp |         104 gso/s, 0 ulp |
| `nk_euclideans_symmetric_i8_haswell`       |        53.8 gso/s, 0 ulp |        77.0 gso/s, 0 ulp |         169 gso/s, ? ulp |
| `nk_angulars_packed_i8_icelake`            |        88.7 gso/s, 0 ulp |         293 gso/s, 0 ulp |         174 gso/s, 0 ulp |
| `nk_angulars_symmetric_i8_icelake`         |        62.1 gso/s, 0 ulp |         122 gso/s, 0 ulp |         262 gso/s, ? ulp |
| `nk_euclideans_packed_i8_icelake`          |        94.2 gso/s, 0 ulp |         164 gso/s, 0 ulp |         188 gso/s, 0 ulp |
| `nk_euclideans_symmetric_i8_icelake`       |        62.2 gso/s, 0 ulp |         145 gso/s, 0 ulp |         247 gso/s, ? ulp |
| `nk_angulars_packed_i8_sapphireamx`        |         448 gso/s, 0 ulp |        1253 gso/s, 0 ulp |        1234 gso/s, 0 ulp |
| `nk_angulars_symmetric_i8_sapphireamx`     |         103 gso/s, 0 ulp |         261 gso/s, 0 ulp |         233 gso/s, 0 ulp |
| `nk_euclideans_packed_i8_sapphireamx`      |         433 gso/s, 0 ulp |        1234 gso/s, 0 ulp |        1248 gso/s, 0 ulp |
| `nk_euclideans_symmetric_i8_sapphireamx`   |         101 gso/s, 0 ulp |         262 gso/s, 0 ulp |         228 gso/s, 0 ulp |
| __u8__                                     | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_u8_serial`             |      12.2 gso/s, 0.3 ulp |      12.8 gso/s, 0.3 ulp |      13.0 gso/s, 0.3 ulp |
| `nk_angulars_symmetric_u8_serial`          |      4.48 gso/s, 0.3 ulp |      4.73 gso/s, 0.3 ulp |        9.50 gso/s, ? ulp |
| `nk_euclideans_packed_u8_serial`           |      12.0 gso/s, 0.5 ulp |      13.1 gso/s, 0.5 ulp |      13.4 gso/s, 0.6 ulp |
| `nk_euclideans_symmetric_u8_serial`        |      4.52 gso/s, 0.5 ulp |      4.69 gso/s, 0.5 ulp |        9.65 gso/s, ? ulp |
| `nk_angulars_packed_u8_haswell`            |      52.9 gso/s, 0.3 ulp |      85.2 gso/s, 0.3 ulp |      77.6 gso/s, 0.3 ulp |
| `nk_angulars_symmetric_u8_haswell`         |      39.6 gso/s, 0.3 ulp |      65.8 gso/s, 0.3 ulp |         130 gso/s, ? ulp |
| `nk_euclideans_packed_u8_haswell`          |      55.4 gso/s, 0.5 ulp |      87.0 gso/s, 0.5 ulp |       103 gso/s, 0.6 ulp |
| `nk_euclideans_symmetric_u8_haswell`       |      39.2 gso/s, 0.5 ulp |      72.9 gso/s, 0.5 ulp |         157 gso/s, ? ulp |
| `nk_angulars_packed_u8_icelake`            |      89.2 gso/s, 0.3 ulp |       293 gso/s, 0.3 ulp |       171 gso/s, 0.3 ulp |
| `nk_angulars_symmetric_u8_icelake`         |      62.0 gso/s, 0.3 ulp |       119 gso/s, 0.3 ulp |         224 gso/s, ? ulp |
| `nk_euclideans_packed_u8_icelake`          |        94.1 gso/s, 0 ulp |         168 gso/s, 0 ulp |         195 gso/s, 0 ulp |
| `nk_euclideans_symmetric_u8_icelake`       |        61.5 gso/s, 0 ulp |         117 gso/s, 0 ulp |         294 gso/s, ? ulp |
| `nk_angulars_packed_u8_sapphireamx`        |       436 gso/s, 0.2 ulp |      1194 gso/s, 0.2 ulp |      1208 gso/s, 0.2 ulp |
| `nk_angulars_symmetric_u8_sapphireamx`     |       101 gso/s, 0.2 ulp |       258 gso/s, 0.2 ulp |       231 gso/s, 0.2 ulp |
| `nk_euclideans_packed_u8_sapphireamx`      |         433 gso/s, 0 ulp |        1207 gso/s, 0 ulp |        1241 gso/s, 0 ulp |
| `nk_euclideans_symmetric_u8_sapphireamx`   |         102 gso/s, 0 ulp |         259 gso/s, 0 ulp |         228 gso/s, 0 ulp |
| __i4__                                     | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_i4_serial`             |        3.79 gso/s, ? ulp |        3.83 gso/s, ? ulp |        4.06 gso/s, ? ulp |
| `nk_angulars_symmetric_i4_serial`          |        3.52 gso/s, ? ulp |        3.58 gso/s, ? ulp |        7.08 gso/s, ? ulp |
| `nk_euclideans_packed_i4_serial`           |        3.69 gso/s, ? ulp |        3.91 gso/s, ? ulp |        3.76 gso/s, ? ulp |
| `nk_euclideans_symmetric_i4_serial`        |        3.45 gso/s, ? ulp |        3.64 gso/s, ? ulp |        6.99 gso/s, ? ulp |
| `nk_angulars_packed_i4_icelake`            |        65.5 gso/s, ? ulp |         203 gso/s, ? ulp |         122 gso/s, ? ulp |
| `nk_angulars_symmetric_i4_icelake`         |        61.7 gso/s, ? ulp |         140 gso/s, ? ulp |         339 gso/s, ? ulp |
| `nk_euclideans_packed_i4_icelake`          |        66.4 gso/s, ? ulp |         138 gso/s, ? ulp |         123 gso/s, ? ulp |
| `nk_euclideans_symmetric_i4_icelake`       |        61.2 gso/s, ? ulp |         129 gso/s, ? ulp |         265 gso/s, ? ulp |
| __u4__                                     | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_u4_serial`             |        5.49 gso/s, ? ulp |        5.60 gso/s, ? ulp |        5.78 gso/s, ? ulp |
| `nk_angulars_symmetric_u4_serial`          |        5.18 gso/s, ? ulp |        5.57 gso/s, ? ulp |        11.5 gso/s, ? ulp |
| `nk_euclideans_packed_u4_serial`           |        5.23 gso/s, ? ulp |        5.50 gso/s, ? ulp |        5.64 gso/s, ? ulp |
| `nk_euclideans_symmetric_u4_serial`        |        5.22 gso/s, ? ulp |        5.47 gso/s, ? ulp |        11.1 gso/s, ? ulp |
| `nk_angulars_packed_u4_icelake`            |        92.2 gso/s, ? ulp |         274 gso/s, ? ulp |         176 gso/s, ? ulp |
| `nk_angulars_symmetric_u4_icelake`         |        71.9 gso/s, ? ulp |         169 gso/s, ? ulp |         311 gso/s, ? ulp |
| `nk_euclideans_packed_u4_icelake`          |        94.1 gso/s, ? ulp |         175 gso/s, ? ulp |         178 gso/s, ? ulp |
| `nk_euclideans_symmetric_u4_icelake`       |        72.2 gso/s, ? ulp |         150 gso/s, ? ulp |         287 gso/s, ? ulp |

#### WASM

Measured with Wasmtime v42 (Cranelift backend).

| Kernel                                     |                     256³ |                    1024³ |                    4096³ |
| :----------------------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f64__                                    | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_f64_serial`            |        1.38 gso/s, 0 ulp |        1.37 gso/s, 0 ulp |        1.36 gso/s, 0 ulp |
| `nk_angulars_symmetric_f64_serial`         |       0.267 gso/s, 0 ulp |       0.268 gso/s, 0 ulp |       0.258 gso/s, 0 ulp |
| `nk_euclideans_packed_f64_serial`          |      1.41 gso/s, 0.6 ulp |      1.37 gso/s, 0.6 ulp |      1.36 gso/s, 0.6 ulp |
| `nk_euclideans_symmetric_f64_serial`       |     0.272 gso/s, 0.6 ulp |     0.271 gso/s, 0.5 ulp |     0.161 gso/s, 0.5 ulp |
| `nk_angulars_packed_f64_v128relaxed`       |      10.9 gso/s, 0.1 ulp |      10.9 gso/s, 0.1 ulp |      10.9 gso/s, 0.1 ulp |
| `nk_angulars_symmetric_f64_v128relaxed`    |     0.238 gso/s, 0.1 ulp |     0.240 gso/s, 0.1 ulp |     0.271 gso/s, 0.1 ulp |
| `nk_euclideans_packed_f64_v128relaxed`     |      11.0 gso/s, 0.6 ulp |      11.2 gso/s, 0.6 ulp |      11.2 gso/s, 0.6 ulp |
| `nk_euclideans_symmetric_f64_v128relaxed`  |    0.0463 gso/s, 0.6 ulp |    0.0465 gso/s, 0.5 ulp |   0.00806 gso/s, 0.5 ulp |
| __f32__                                    | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_f32_serial`            |      11.2 gso/s, 0.1 ulp |      11.5 gso/s, 0.1 ulp |      11.0 gso/s, 0.1 ulp |
| `nk_angulars_symmetric_f32_serial`         |     0.252 gso/s, 0.1 ulp |     0.167 gso/s, 0.1 ulp |    0.0527 gso/s, 0.1 ulp |
| `nk_euclideans_packed_f32_serial`          |      11.7 gso/s, 0.6 ulp |      11.4 gso/s, 0.6 ulp |      11.1 gso/s, 0.5 ulp |
| `nk_euclideans_symmetric_f32_serial`       |     0.132 gso/s, 0.5 ulp |     0.229 gso/s, 0.5 ulp |     0.207 gso/s, 0.5 ulp |
| `nk_angulars_packed_f32_v128relaxed`       |      22.0 gso/s, 0.1 ulp |      22.1 gso/s, 0.1 ulp |      22.3 gso/s, 0.1 ulp |
| `nk_angulars_symmetric_f32_v128relaxed`    |     0.246 gso/s, 0.1 ulp |     0.213 gso/s, 0.1 ulp |    0.0556 gso/s, 0.1 ulp |
| `nk_euclideans_packed_f32_v128relaxed`     |      22.4 gso/s, 0.2 ulp |      22.4 gso/s, 0.2 ulp |      22.5 gso/s, 0.2 ulp |
| `nk_euclideans_symmetric_f32_v128relaxed`  |     0.172 gso/s, 0.2 ulp |     0.209 gso/s, 0.2 ulp |     0.210 gso/s, 0.2 ulp |
| __bf16__                                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_bf16_serial`           |        4.44 gso/s, 0 ulp |      4.42 gso/s, 0.2 ulp |      4.26 gso/s, 0.6 ulp |
| `nk_angulars_symmetric_bf16_serial`        |      0.0853 gso/s, 0 ulp |     0.277 gso/s, 0.2 ulp |    0.0455 gso/s, 0.6 ulp |
| `nk_euclideans_packed_bf16_serial`         |      4.40 gso/s, 0.7 ulp |      4.38 gso/s, 6.1 ulp |       4.30 gso/s, 32 ulp |
| `nk_euclideans_symmetric_bf16_serial`      |     0.197 gso/s, 0.6 ulp |    0.0838 gso/s, 5.3 ulp |      0.161 gso/s, 28 ulp |
| `nk_angulars_packed_bf16_v128relaxed`      |        24.7 gso/s, 0 ulp |      23.3 gso/s, 0.2 ulp |      23.7 gso/s, 0.6 ulp |
| `nk_angulars_symmetric_bf16_v128relaxed`   |       0.128 gso/s, 0 ulp |    0.0145 gso/s, 0.2 ulp |     0.246 gso/s, 0.6 ulp |
| `nk_euclideans_packed_bf16_v128relaxed`    |      25.3 gso/s, 0.7 ulp |      23.6 gso/s, 6.1 ulp |       23.7 gso/s, 32 ulp |
| `nk_euclideans_symmetric_bf16_v128relaxed` |     0.127 gso/s, 0.3 ulp |     0.130 gso/s, 5.1 ulp |     0.0523 gso/s, 28 ulp |
| __e2m3__                                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_e2m3_serial`           |        2.66 gso/s, 0 ulp |        2.71 gso/s, 0 ulp |        2.63 gso/s, 0 ulp |
| `nk_angulars_symmetric_e2m3_serial`        |      0.0400 gso/s, 0 ulp |      0.0413 gso/s, 0 ulp |       0.238 gso/s, 0 ulp |
| `nk_euclideans_packed_e2m3_serial`         |      2.74 gso/s, 0.5 ulp |      2.70 gso/s, 0.5 ulp |      2.67 gso/s, 0.5 ulp |
| `nk_euclideans_symmetric_e2m3_serial`      |    0.0403 gso/s, 0.5 ulp |    0.0411 gso/s, 0.4 ulp |    0.0401 gso/s, 0.4 ulp |
| `nk_angulars_packed_e2m3_v128relaxed`      |        18.4 gso/s, 0 ulp |        18.6 gso/s, 0 ulp |        18.5 gso/s, 0 ulp |
| `nk_angulars_symmetric_e2m3_v128relaxed`   |      0.0559 gso/s, 0 ulp |      0.0180 gso/s, 0 ulp |       0.131 gso/s, 0 ulp |
| `nk_euclideans_packed_e2m3_v128relaxed`    |        18.5 gso/s, 0 ulp |        18.7 gso/s, 0 ulp |        18.1 gso/s, 0 ulp |
| `nk_euclideans_symmetric_e2m3_v128relaxed` |       0.206 gso/s, 0 ulp |      0.0170 gso/s, 0 ulp |      0.0554 gso/s, 0 ulp |
| __i8__                                     | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_i8_serial`             |        4.73 gso/s, 0 ulp |        4.81 gso/s, 0 ulp |        4.59 gso/s, 0 ulp |
| `nk_angulars_symmetric_i8_serial`          |     0.00447 gso/s, 0 ulp |       0.198 gso/s, 0 ulp |       0.190 gso/s, 0 ulp |
| `nk_euclideans_packed_i8_serial`           |      4.77 gso/s, 0.5 ulp |      4.80 gso/s, 0.4 ulp |      4.65 gso/s, 0.4 ulp |
| `nk_euclideans_symmetric_i8_serial`        |     0.201 gso/s, 0.5 ulp |    0.0819 gso/s, 0.4 ulp |    0.0823 gso/s, 0.4 ulp |
| `nk_angulars_packed_i8_v128relaxed`        |        31.6 gso/s, 0 ulp |        31.7 gso/s, 0 ulp |        31.1 gso/s, 0 ulp |
| `nk_angulars_symmetric_i8_v128relaxed`     |      0.0304 gso/s, 0 ulp |      0.0680 gso/s, 0 ulp |       0.298 gso/s, 0 ulp |
| `nk_euclideans_packed_i8_v128relaxed`      |        31.5 gso/s, 0 ulp |        32.3 gso/s, 0 ulp |        30.8 gso/s, 0 ulp |
| `nk_euclideans_symmetric_i8_v128relaxed`   |       0.224 gso/s, 0 ulp |       0.222 gso/s, 0 ulp |       0.143 gso/s, 0 ulp |
| __u8__                                     | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_u8_serial`             |      5.11 gso/s, 0.4 ulp |      5.30 gso/s, 0.3 ulp |      5.03 gso/s, 0.3 ulp |
| `nk_angulars_symmetric_u8_serial`          |     0.118 gso/s, 0.4 ulp |   0.00461 gso/s, 0.3 ulp |    0.0838 gso/s, 0.3 ulp |
| `nk_euclideans_packed_u8_serial`           |      5.24 gso/s, 0.5 ulp |      5.22 gso/s, 0.5 ulp |      5.07 gso/s, 0.5 ulp |
| `nk_euclideans_symmetric_u8_serial`        |     0.229 gso/s, 0.5 ulp |    0.0829 gso/s, 0.5 ulp |     0.224 gso/s, 0.5 ulp |
| `nk_angulars_packed_u8_v128relaxed`        |        79.4 gso/s, ? ulp |        79.6 gso/s, ? ulp |        77.9 gso/s, ? ulp |
| `nk_angulars_symmetric_u8_v128relaxed`     |     0.00453 gso/s, ? ulp |       0.198 gso/s, ? ulp |       0.274 gso/s, ? ulp |
| `nk_euclideans_packed_u8_v128relaxed`      |        78.1 gso/s, ? ulp |        81.6 gso/s, ? ulp |        78.1 gso/s, ? ulp |
| `nk_euclideans_symmetric_u8_v128relaxed`   |       0.201 gso/s, ? ulp |      0.0814 gso/s, ? ulp |      0.0831 gso/s, ? ulp |
| __i4__                                     | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_i4_serial`             |        7.59 gso/s, ? ulp |        7.51 gso/s, ? ulp |        7.48 gso/s, ? ulp |
| `nk_angulars_symmetric_i4_serial`          |       0.163 gso/s, ? ulp |       0.274 gso/s, ? ulp |       0.158 gso/s, ? ulp |
| `nk_euclideans_packed_i4_serial`           |        7.79 gso/s, ? ulp |        7.85 gso/s, ? ulp |        7.38 gso/s, ? ulp |
| `nk_euclideans_symmetric_i4_serial`        |       0.159 gso/s, ? ulp |       0.165 gso/s, ? ulp |       0.160 gso/s, ? ulp |
| __u4__                                     | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_u4_serial`             |        5.84 gso/s, ? ulp |        5.78 gso/s, ? ulp |        5.62 gso/s, ? ulp |
| `nk_angulars_symmetric_u4_serial`          |       0.160 gso/s, ? ulp |       0.270 gso/s, ? ulp |      0.0430 gso/s, ? ulp |
| `nk_euclideans_packed_u4_serial`           |        5.80 gso/s, ? ulp |        5.83 gso/s, ? ulp |        5.63 gso/s, ? ulp |
| `nk_euclideans_symmetric_u4_serial`        |       0.157 gso/s, ? ulp |       0.267 gso/s, ? ulp |       0.156 gso/s, ? ulp |

### Apple M4 Pro

#### Native

| Kernel                                   |                     256³ |                    1024³ |                    4096³ |
| :--------------------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f64__                                  | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_f64_serial`          |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_angulars_symmetric_f64_serial`       |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_euclideans_packed_f64_serial`        |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_euclideans_symmetric_f64_serial`     |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_angulars_packed_f64_neon`            |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_angulars_symmetric_f64_neon`         |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_euclideans_packed_f64_neon`          |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_euclideans_symmetric_f64_neon`       |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_angulars_packed_f64_smef64`          |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_angulars_symmetric_f64_smef64`       |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_euclideans_packed_f64_smef64`        |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_euclideans_symmetric_f64_smef64`     |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| __f32__                                  | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_f32_serial`          |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_angulars_symmetric_f32_serial`       |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_euclideans_packed_f32_serial`        |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_euclideans_symmetric_f32_serial`     |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_angulars_packed_f32_neon`            |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_angulars_symmetric_f32_neon`         |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_euclideans_packed_f32_neon`          |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_euclideans_symmetric_f32_neon`       |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_angulars_packed_f32_smef64`          |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_angulars_symmetric_f32_smef64`       |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_euclideans_packed_f32_smef64`        |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_euclideans_symmetric_f32_smef64`     |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| __bf16__                                 | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_bf16_serial`         |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_angulars_symmetric_bf16_serial`      |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_euclideans_packed_bf16_serial`       |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_euclideans_symmetric_bf16_serial`    |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_angulars_packed_bf16_neonbfdot`      |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_angulars_symmetric_bf16_neonbfdot`   |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_euclideans_packed_bf16_neonbfdot`    |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_euclideans_symmetric_bf16_neonbfdot` |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_angulars_packed_bf16_sme`            |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_angulars_symmetric_bf16_sme`         |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_euclideans_packed_bf16_sme`          |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_euclideans_symmetric_bf16_sme`       |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| __f16__                                  | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_f16_serial`          |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_angulars_symmetric_f16_serial`       |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_euclideans_packed_f16_serial`        |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_euclideans_symmetric_f16_serial`     |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_angulars_packed_f16_neonhalf`        |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_angulars_symmetric_f16_neonhalf`     |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_euclideans_packed_f16_neonhalf`      |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_euclideans_symmetric_f16_neonhalf`   |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_angulars_packed_f16_neonfhm`         |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_angulars_symmetric_f16_neonfhm`      |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_euclideans_packed_f16_neonfhm`       |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_euclideans_symmetric_f16_neonfhm`    |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_angulars_packed_f16_sme`             |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_angulars_symmetric_f16_sme`          |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_euclideans_packed_f16_sme`           |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_euclideans_symmetric_f16_sme`        |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| __e5m2__                                 | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_e5m2_serial`         |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_angulars_symmetric_e5m2_serial`      |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_euclideans_packed_e5m2_serial`       |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_euclideans_symmetric_e5m2_serial`    |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_angulars_packed_e5m2_neonfhm`        |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_angulars_symmetric_e5m2_neonfhm`     |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_euclideans_packed_e5m2_neonfhm`      |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_euclideans_symmetric_e5m2_neonfhm`   |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_angulars_packed_e5m2_sme`            |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_angulars_symmetric_e5m2_sme`         |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_euclideans_packed_e5m2_sme`          |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_euclideans_symmetric_e5m2_sme`       |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| __e4m3__                                 | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_e4m3_serial`         |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_angulars_symmetric_e4m3_serial`      |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_euclideans_packed_e4m3_serial`       |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_euclideans_symmetric_e4m3_serial`    |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_angulars_packed_e4m3_neonfhm`        |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_angulars_symmetric_e4m3_neonfhm`     |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_euclideans_packed_e4m3_neonfhm`      |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_euclideans_symmetric_e4m3_neonfhm`   |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_angulars_packed_e4m3_sme`            |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_angulars_symmetric_e4m3_sme`         |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_euclideans_packed_e4m3_sme`          |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_euclideans_symmetric_e4m3_sme`       |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| __e3m2__                                 | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_e3m2_serial`         |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_angulars_symmetric_e3m2_serial`      |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_euclideans_packed_e3m2_serial`       |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_euclideans_symmetric_e3m2_serial`    |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_angulars_packed_e3m2_neonfhm`        |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_angulars_symmetric_e3m2_neonfhm`     |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_euclideans_packed_e3m2_neonfhm`      |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_euclideans_symmetric_e3m2_neonfhm`   |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_angulars_packed_e3m2_sme`            |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_angulars_symmetric_e3m2_sme`         |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_euclideans_packed_e3m2_sme`          |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_euclideans_symmetric_e3m2_sme`       |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| __e2m3__                                 | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_e2m3_serial`         |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_angulars_symmetric_e2m3_serial`      |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_euclideans_packed_e2m3_serial`       |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_euclideans_symmetric_e2m3_serial`    |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_angulars_packed_e2m3_neonfhm`        |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_angulars_symmetric_e2m3_neonfhm`     |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_euclideans_packed_e2m3_neonfhm`      |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_euclideans_symmetric_e2m3_neonfhm`   |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_angulars_packed_e2m3_sme`            |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_angulars_symmetric_e2m3_sme`         |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_euclideans_packed_e2m3_sme`          |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_euclideans_symmetric_e2m3_sme`       |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| __i8__                                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_i8_serial`           |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_angulars_symmetric_i8_serial`        |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_euclideans_packed_i8_serial`         |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_euclideans_symmetric_i8_serial`      |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_angulars_packed_i8_neonsdot`         |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_angulars_symmetric_i8_neonsdot`      |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_euclideans_packed_i8_neonsdot`       |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_euclideans_symmetric_i8_neonsdot`    |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_angulars_packed_i8_sme`              |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_angulars_symmetric_i8_sme`           |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_euclideans_packed_i8_sme`            |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_euclideans_symmetric_i8_sme`         |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| __u8__                                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_u8_serial`           |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_angulars_symmetric_u8_serial`        |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_euclideans_packed_u8_serial`         |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_euclideans_symmetric_u8_serial`      |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_angulars_packed_u8_neonsdot`         |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_angulars_symmetric_u8_neonsdot`      |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_euclideans_packed_u8_neonsdot`       |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_euclideans_symmetric_u8_neonsdot`    |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_angulars_packed_u8_sme`              |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_angulars_symmetric_u8_sme`           |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_euclideans_packed_u8_sme`            |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_euclideans_symmetric_u8_sme`         |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| __i4__                                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_i4_serial`           |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_angulars_symmetric_i4_serial`        |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_euclideans_packed_i4_serial`         |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_euclideans_symmetric_i4_serial`      |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_angulars_packed_i4_neonsdot`         |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_angulars_symmetric_i4_neonsdot`      |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_euclideans_packed_i4_neonsdot`       |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_euclideans_symmetric_i4_neonsdot`    |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_angulars_packed_i4_sme`              |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_angulars_symmetric_i4_sme`           |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_euclideans_packed_i4_sme`            |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_euclideans_symmetric_i4_sme`         |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| __u4__                                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_u4_serial`           |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_angulars_symmetric_u4_serial`        |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_euclideans_packed_u4_serial`         |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_euclideans_symmetric_u4_serial`      |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_angulars_packed_u4_neonsdot`         |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_angulars_symmetric_u4_neonsdot`      |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_euclideans_packed_u4_neonsdot`       |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_euclideans_symmetric_u4_neonsdot`    |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_angulars_packed_u4_sme`              |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_angulars_symmetric_u4_sme`           |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_euclideans_packed_u4_sme`            |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_euclideans_symmetric_u4_sme`         |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |

#### WASM

Measured with Wasmtime v42 (Cranelift backend).

| Kernel                                     |                     256³ |                    1024³ |                    4096³ |
| :----------------------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f64__                                    | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_f64_serial`            |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_angulars_packed_f64_v128relaxed`       |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_angulars_symmetric_f64_serial`         |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_angulars_symmetric_f64_v128relaxed`    |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_euclideans_packed_f64_serial`          |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_euclideans_packed_f64_v128relaxed`     |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_euclideans_symmetric_f64_serial`       |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_euclideans_symmetric_f64_v128relaxed`  |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| __f32__                                    | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_f32_serial`            |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_angulars_packed_f32_v128relaxed`       |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_angulars_symmetric_f32_serial`         |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_angulars_symmetric_f32_v128relaxed`    |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_euclideans_packed_f32_serial`          |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_euclideans_packed_f32_v128relaxed`     |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_euclideans_symmetric_f32_serial`       |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_euclideans_symmetric_f32_v128relaxed`  |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| __bf16__                                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_bf16_serial`           |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_angulars_packed_bf16_v128relaxed`      |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_angulars_symmetric_bf16_serial`        |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_angulars_symmetric_bf16_v128relaxed`   |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_euclideans_packed_bf16_serial`         |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_euclideans_packed_bf16_v128relaxed`    |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_euclideans_symmetric_bf16_serial`      |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_euclideans_symmetric_bf16_v128relaxed` |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| __e2m3__                                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_e2m3_serial`           |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_angulars_packed_e2m3_v128relaxed`      |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_angulars_symmetric_e2m3_serial`        |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_angulars_symmetric_e2m3_v128relaxed`   |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_euclideans_packed_e2m3_serial`         |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_euclideans_packed_e2m3_v128relaxed`    |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_euclideans_symmetric_e2m3_serial`      |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_euclideans_symmetric_e2m3_v128relaxed` |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| __i8__                                     | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_i8_serial`             |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_angulars_packed_i8_v128relaxed`        |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_angulars_symmetric_i8_serial`          |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_angulars_symmetric_i8_v128relaxed`     |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_euclideans_packed_i8_serial`           |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_euclideans_packed_i8_v128relaxed`      |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_euclideans_symmetric_i8_serial`        |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_euclideans_symmetric_i8_v128relaxed`   |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| __u8__                                     | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_u8_serial`             |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_angulars_packed_u8_v128relaxed`        |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_angulars_symmetric_u8_serial`          |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_angulars_symmetric_u8_v128relaxed`     |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_euclideans_packed_u8_serial`           |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_euclideans_packed_u8_v128relaxed`      |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_euclideans_symmetric_u8_serial`        |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_euclideans_symmetric_u8_v128relaxed`   |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |

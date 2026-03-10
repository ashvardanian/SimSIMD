# Curved Space Distances in NumKong

NumKong implements distance functions for curved metric spaces: bilinear forms compute $a^T C b$ for an arbitrary metric tensor $C$, while Mahalanobis distance generalizes Euclidean distance to account for correlations between dimensions.
Complex bilinear forms extend this to Hermitian inner products.
These operations are central to Gaussian process inference, metric learning, and statistical distance measures.

The bilinear form for real vectors is:

```math
\text{bilinear}(a, b, C) = a^T C b = \sum_{i=0}^{n-1} \sum_{j=0}^{n-1} a_i \cdot c_{ij} \cdot b_j
```

The Mahalanobis distance is:

```math
\text{mahalanobis}(a, b, C) = \sqrt{(a - b)^T C (a - b)}
```

For complex vectors, the bilinear form uses the conjugate transpose:

```math
\text{bilinear}(a, b, C) = a^H C b = \sum_{i=0}^{n-1} \sum_{j=0}^{n-1} \bar{a_i} \cdot c_{ij} \cdot b_j
```

Reformulating as Python pseudocode:

```python
import numpy as np

def bilinear(a: np.ndarray, b: np.ndarray, C: np.ndarray) -> float:
    return a @ C @ b

def mahalanobis(a: np.ndarray, b: np.ndarray, C: np.ndarray) -> float:
    diff = a - b
    return np.sqrt(diff @ C @ diff)

def bilinear_complex(a: np.ndarray, b: np.ndarray, C: np.ndarray) -> complex:
    return np.conj(a) @ C @ b
```

## Input & Output Types

Real bilinear and Mahalanobis:

| Input Type | Output Type | Description                                    |
| ---------- | ----------- | ---------------------------------------------- |
| `f64`      | `f64`       | 64-bit IEEE 754 double precision               |
| `f32`      | `f32`       | 32-bit IEEE 754 single precision               |
| `f16`      | `f32`       | 16-bit IEEE 754 half precision, widened output |
| `bf16`     | `f32`       | 16-bit brain float, widened output             |

Complex bilinear:

| Input Type | Output Type | Description                                |
| ---------- | ----------- | ------------------------------------------ |
| `f64c`     | `f64c`      | 64-bit complex pairs                       |
| `f32c`     | `f32c`      | 32-bit complex pairs                       |
| `f16c`     | `f32c`      | 16-bit complex pairs, widened output       |
| `bf16c`    | `f32c`      | 16-bit brain complex pairs, widened output |

## Optimizations

### Row-Major Streaming with Nested Dot2

`nk_bilinear_f64_skylake`, `nk_mahalanobis_f64_skylake` decompose the bilinear form $a^T C b$ as $\sum_i a_i \cdot \text{dot}(C_i, b)$ where $C_i$ is the $i$-th row of the metric tensor.
Each inner dot product uses Dot2 compensation — TwoProd via FMA captures the rounding error of each $c_{ij} \cdot b_j$ product exactly, and a TwoSum chain propagates it through the accumulator.
The outer sum over rows uses a second level of compensation, tracking the rounding error of each $a_i \cdot r_i$ accumulation.
This nested structure gives $O(n)$ cache-friendly sequential access to the $n \times n$ matrix $C$, since each row is read once and discarded.
`nk_bilinear_f32_neon`, `nk_bilinear_f32_skylake`, `nk_mahalanobis_f32_neon`, `nk_mahalanobis_f32_skylake` use the same row-major streaming pattern but accumulate in `f64` instead of Dot2, which provides sufficient precision for `f32` inputs.

### SME Outer-Product Accumulation

`nk_bilinear_f32_smef64`, `nk_bilinear_f64_smef64`, `nk_bilinear_f32c_smef64`, `nk_bilinear_f64c_smef64`, `nk_mahalanobis_f32_smef64`, `nk_mahalanobis_f64_smef64` use the Scalable Matrix Extension to compute the bilinear form as an outer-product accumulation.
Each `FMOPA` instruction performs a rank-1 update $a_i \cdot b^T$ into the SME ZA tile array, and the matrix $C$ is streamed row-by-row and multiplied into the accumulator.
This is fundamentally different from the row-major dot approach — it reformulates $a^T C b$ as a matrix-multiply problem where SME's 2D tile registers can exploit the matrix engine's throughput.
For dimensions that align to the tile size, this approach achieves near-peak throughput; dimensions that do not align fall back to NEON for cleanup of the residual elements.

### Complex Bilinear Decomposition

`nk_bilinear_f32c_neon`, `nk_bilinear_f32c_skylake`, `nk_bilinear_f64c_skylake` compute $a^H C b$ where each element involves 4 real multiplications from the complex product $\bar{a_i} \cdot c_{ij} \cdot b_j$.
The kernel decomposes this into real and imaginary dot products over rows of $C$: for each row $i$, it computes the real part as $a_{i,re} \cdot \text{dot}(C_i, b)_{re} + a_{i,im} \cdot \text{dot}(C_i, b)_{im}$ and the imaginary part with the conjugation baked in as sign flips.
This fuses the conjugation of $a$ into the sign of the cross terms rather than explicitly negating the imaginary components, saving one negate operation per element.

## Performance

The following performance tables are produced by manually re-running `nk_test` and `nk_bench` included internal tools to measure both accuracy and throughput at different input shapes.
The input size is controlled by the `NK_CURVED_DIMENSIONS` environment variable.
The metric tensor is a square matrix of side $N$, so each bilinear form $\mathbf{x}^\top M \mathbf{x}$ has $O(N^2)$ arithmetic complexity.
Columns show matrix side length: 256², 1024², 4096².
The throughput is measured in GSO/s as Giga scalar operations per second.
Accuracy is reported as ULP (units in last place), the number of representable floating-point values between the result and the exact answer.
Each kernel runs for at least 20 seconds per configuration.
Benchmark threads are pinned to specific cores; on machines with heterogeneous core types (e.g., Apple P/E cores), only the fastest cores are used.
Workloads that significantly degrade CPU frequencies (Intel AMX, Apple SME) run in separate passes to avoid affecting throughput measurements of other kernels.

### Intel Sapphire Rapids

#### Native

| Kernel                        |                     256² |                    1024² |                    4096² |
| :---------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f64c__                      | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_bilinear_f64c_serial`     |     0.170 gso/s, 1.9 ulp |     0.109 gso/s, 1.8 ulp |     0.106 gso/s, 2.0 ulp |
| `nk_bilinear_f64c_skylake`    |     0.884 gso/s, 5.1 ulp |     0.865 gso/s, 4.8 ulp |     0.790 gso/s, 5.1 ulp |
| __f32c__                      | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_bilinear_f32c_serial`     |        1.30 gso/s, 0 ulp |       0.992 gso/s, 0 ulp |        1.05 gso/s, 0 ulp |
| `nk_bilinear_f32c_skylake`    |        2.01 gso/s, 0 ulp |        1.86 gso/s, 0 ulp |        1.81 gso/s, 0 ulp |
| __bf16c__                     | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_bilinear_bf16c_serial`    |      0.149 gso/s, 18 ulp |      0.115 gso/s, 18 ulp |      0.104 gso/s, 19 ulp |
| `nk_bilinear_bf16c_genoa`     |       4.33 gso/s, 19 ulp |       3.98 gso/s, 18 ulp |       4.04 gso/s, 19 ulp |
| __f16c__                      | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_bilinear_f16c_serial`     |      0.576 gso/s, 34 ulp |      0.583 gso/s, 35 ulp |      0.582 gso/s, 37 ulp |
| __f64__                       | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_bilinear_f64_serial`      |     0.587 gso/s, 1.5 ulp |     0.370 gso/s, 1.4 ulp |     0.372 gso/s, 1.4 ulp |
| `nk_mahalanobis_f64_serial`   |     0.556 gso/s, 0.3 ulp |     0.332 gso/s, 0.3 ulp |     0.307 gso/s, 0.2 ulp |
| `nk_bilinear_f64_skylake`     |      2.09 gso/s, 3.0 ulp |      2.03 gso/s, 2.9 ulp |      1.78 gso/s, 2.9 ulp |
| `nk_mahalanobis_f64_skylake`  |      2.11 gso/s, 0.2 ulp |      1.94 gso/s, 0.2 ulp |      2.14 gso/s, 0.2 ulp |
| __f32__                       | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_bilinear_f32_serial`      |        2.51 gso/s, 0 ulp |        1.94 gso/s, 0 ulp |        2.12 gso/s, 0 ulp |
| `nk_mahalanobis_f32_serial`   |        2.19 gso/s, 0 ulp |        1.52 gso/s, 0 ulp |        1.60 gso/s, 0 ulp |
| `nk_bilinear_f32_haswell`     |        3.60 gso/s, 0 ulp |        3.84 gso/s, 0 ulp |        3.73 gso/s, 0 ulp |
| `nk_mahalanobis_f32_haswell`  |        3.41 gso/s, 0 ulp |        3.48 gso/s, 0 ulp |        3.31 gso/s, 0 ulp |
| `nk_bilinear_f32_skylake`     |        4.30 gso/s, 0 ulp |        4.01 gso/s, 0 ulp |        3.79 gso/s, 0 ulp |
| `nk_mahalanobis_f32_skylake`  |        3.80 gso/s, 0 ulp |        3.59 gso/s, 0 ulp |        3.78 gso/s, 0 ulp |
| __bf16__                      | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_bilinear_bf16_serial`     |      0.311 gso/s, 17 ulp |      0.208 gso/s, 17 ulp |      0.209 gso/s, 18 ulp |
| `nk_mahalanobis_bf16_serial`  |     0.197 gso/s, 2.7 ulp |     0.169 gso/s, 2.7 ulp |     0.169 gso/s, 2.8 ulp |
| `nk_bilinear_bf16_haswell`    |       8.20 gso/s, 23 ulp |       7.17 gso/s, 24 ulp |       5.44 gso/s, 23 ulp |
| `nk_mahalanobis_bf16_haswell` |      6.98 gso/s, 0.9 ulp |      6.51 gso/s, 0.8 ulp |      6.03 gso/s, 0.9 ulp |
| `nk_bilinear_bf16_genoa`      |       10.8 gso/s, 25 ulp |       9.54 gso/s, 27 ulp |       9.55 gso/s, 24 ulp |
| `nk_mahalanobis_bf16_genoa`   |       8.11 gso/s, 9K ulp |       7.36 gso/s, 9K ulp |       7.76 gso/s, 9K ulp |
| __f16__                       | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_bilinear_f16_serial`      |      0.647 gso/s, 37 ulp |      0.646 gso/s, 34 ulp |      0.665 gso/s, 35 ulp |
| `nk_mahalanobis_f16_serial`   |     0.494 gso/s, 2.2 ulp |     0.505 gso/s, 2.2 ulp |     0.503 gso/s, 2.2 ulp |
| `nk_bilinear_f16_haswell`     |       8.13 gso/s, 27 ulp |       7.48 gso/s, 34 ulp |       7.23 gso/s, 26 ulp |
| `nk_mahalanobis_f16_haswell`  |      8.07 gso/s, 0.8 ulp |      6.87 gso/s, 0.8 ulp |      5.08 gso/s, 0.8 ulp |

### Apple M4 Pro

#### Native

| Kernel                          |                     256² |                    1024² |                    4096² |
| :------------------------------ | -----------------------: | -----------------------: | -----------------------: |
| __f64c__                        | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_bilinear_f64c_serial`       |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| __f32c__                        | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_bilinear_f32c_serial`       |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_bilinear_f32c_neon`         |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| __bf16c__                       | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_bilinear_bf16c_serial`      |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_bilinear_bf16c_neonbfdot`   |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| __f16c__                        | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_bilinear_f16c_serial`       |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_bilinear_f16c_neonhalf`     |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| __f64__                         | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_bilinear_f64_serial`        |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_mahalanobis_f64_serial`     |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| __f32__                         | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_bilinear_f32_serial`        |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_mahalanobis_f32_serial`     |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_bilinear_f32_neon`          |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_mahalanobis_f32_neon`       |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| __bf16__                        | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_bilinear_bf16_serial`       |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_mahalanobis_bf16_serial`    |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_bilinear_bf16_neonbfdot`    |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_mahalanobis_bf16_neonbfdot` |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| __f16__                         | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_bilinear_f16_serial`        |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_mahalanobis_f16_serial`     |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_bilinear_f16_neonhalf`      |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_mahalanobis_f16_neonhalf`   |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |

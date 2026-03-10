# Spatial Distances in NumKong

NumKong implements spatial distance functions for dense vectors: squared Euclidean distance, Euclidean distance, and angular (cosine) distance.
These are the most widely used metrics in nearest-neighbor search, clustering, and dimensionality reduction, covering every numeric type supported by the library.

Squared Euclidean distance measures the sum of squared element-wise differences:

```math
\text{sqeuclidean}(a, b) = \sum_{i=0}^{n-1} (a_i - b_i)^2
```

Euclidean distance is the square root of the squared Euclidean distance:

```math
\text{euclidean}(a, b) = \sqrt{\sum_{i=0}^{n-1} (a_i - b_i)^2}
```

Angular distance (cosine distance) measures the angle between two vectors:

```math
\text{angular}(a, b) = 1 - \frac{\sum_{i=0}^{n-1} a_i \cdot b_i}{\sqrt{\sum_{i=0}^{n-1} a_i^2} \cdot \sqrt{\sum_{i=0}^{n-1} b_i^2}}
```

Reformulating as Python pseudocode:

```python
import numpy as np

def sqeuclidean(a: np.ndarray, b: np.ndarray) -> float:
    return np.sum((a - b) ** 2)

def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return np.sqrt(np.sum((a - b) ** 2))

def angular(a: np.ndarray, b: np.ndarray) -> float:
    ab = np.dot(a, b)
    a2 = np.dot(a, a)
    b2 = np.dot(b, b)
    if a2 == 0 and b2 == 0: return 0
    if ab == 0: return 1
    return 1 - ab / (np.sqrt(a2) * np.sqrt(b2))
```

## Input & Output Types

| Input Type | Output Type | Description                                    |
| ---------- | ----------- | ---------------------------------------------- |
| `f64`      | `f64`       | 64-bit IEEE 754 double precision               |
| `f32`      | `f32`       | 32-bit IEEE 754 single precision               |
| `f16`      | `f32`       | 16-bit IEEE 754 half precision, widened output |
| `bf16`     | `f32`       | 16-bit brain float, widened output             |
| `e5m2`     | `f32`       | 8-bit FP8: 5 exponent, 2 mantissa bits         |
| `e4m3`     | `f32`       | 8-bit FP8: 4 exponent, 3 mantissa bits         |
| `e3m2`     | `f32`       | 8-bit MX format: 3 exponent, 2 mantissa bits   |
| `e2m3`     | `f32`       | 8-bit MX format: 2 exponent, 3 mantissa bits   |
| `i8`       | `f32`       | 8-bit signed integers                          |
| `u8`       | `f32`       | 8-bit unsigned integers                        |
| `i4`       | `f32`       | 4-bit signed integers, packed nibble pairs     |
| `u4`       | `f32`       | 4-bit unsigned integers, packed nibble pairs   |

## Optimizations

### Three-Accumulator Angular Pattern

`nk_angular_f32_haswell`, `nk_angular_f32_skylake`, `nk_angular_f32_neon` compute cosine distance as $1 - ab / (\sqrt{a^2} \cdot \sqrt{b^2})$, requiring three concurrent dot products in a single pass: $\sum a_i b_i$, $\sum a_i^2$, and $\sum b_i^2$.
All spatial angular kernels interleave these three FMA streams so that each vector element is loaded once and immediately contributes to all three accumulators.
This triples register pressure compared to a plain dot product — on Haswell with 16 YMM registers, three independent 4-register accumulator chains leave only 4 registers for temporaries.
The single-pass design is essential because reading two vectors of length $n$ once costs $2n$ cache line fetches, while a three-pass approach would cost $6n$.

### Reciprocal Square Root with Newton-Raphson Refinement

`nk_angular_f32_haswell`, `nk_angular_f64_haswell`, `nk_angular_f32_neon`, `nk_angular_f64_neon` compute the final normalization via in-hardware reciprocal square root estimates refined by Newton-Raphson iteration.
The iteration formula is $x_{n+1} = x_n \cdot (3 - d \cdot x_n^2) / 2$, where $d$ is the value whose reciprocal square root is needed.
NEON `vrsqrte` + `vrsqrts` performs one refinement step, reaching roughly 22 bits of precision.
Haswell `VRSQRT14` provides $2^{-14}$ relative error and one Newton-Raphson step doubles the precision to approximately 28 bits.
Skylake `VRSQRT28` achieves $2^{-28}$ accuracy directly, eliminating the need for a refinement step entirely.
This reciprocal square root is needed for both euclidean distance ($\sqrt{d}$ via $d \cdot \text{rsqrt}(d)$) and angular distance ($1/\sqrt{a^2} \cdot 1/\sqrt{b^2}$).

### Absolute Differences for Integer Types

`nk_sqeuclidean_i8_haswell`, `nk_sqeuclidean_u8_haswell`, `nk_sqeuclidean_i8_skylake`, `nk_sqeuclidean_u8_skylake` compute squared Euclidean distance by first obtaining element-wise absolute differences, then squaring and accumulating.
For signed `i8`, XOR with `0x80` converts the range from [-128, 127] to unsigned [0, 255], then saturating subtract in both directions followed by OR gives $|a - b|$:

```
bias_a = _mm256_xor_si256(a, 0x80)
bias_b = _mm256_xor_si256(b, 0x80)
abs_diff = _mm256_or_si256(_mm256_subs_epu8(bias_a, bias_b), _mm256_subs_epu8(bias_b, bias_a))
```

For unsigned `u8`, the same saturating subtract trick works without the XOR bias.
The absolute differences are then zero-extended via `VPUNPCKLBW`/`VPUNPCKHBW` (1 cycle, cheaper than `VPMOVZXBW`) and squared+accumulated via `VPMADDWD`, which computes $d_i^2 + d_{i+1}^2$ in one instruction.

### Masked Neumaier Compensation on Skylake

`nk_sqeuclidean_f64_skylake` uses `VGETEXP`-based Neumaier TwoSum inside AVX-512 masked loops.
The mask register tracks which lanes are active, handling tail elements when the vector length is not a multiple of the SIMD width.
The compensation term accumulates the low-order rounding errors from each addition, and because the mask propagates through both the main sum and the compensation update, even the final partial iteration maintains full Neumaier accuracy.
This avoids the need for a separate scalar tail loop that would otherwise lose the compensated error tracking.

## Performance

The following performance tables are produced by manually re-running `nk_test` and `nk_bench` included internal tools to measure both accuracy and throughput at different input shapes.
The input size is controlled by the `NK_DENSE_DIMENSIONS` environment variable and set to 256, 1024, and 4096 elements.
The throughput is measured in GB/s as the number of input bytes per second.
Accuracy is reported as ULP (units in last place), the number of representable floating-point values between the result and the exact answer.
Each kernel runs for at least 20 seconds per configuration.
Benchmark threads are pinned to specific cores; on machines with heterogeneous core types (e.g., Apple P/E cores), only the fastest cores are used.
Workloads that significantly degrade CPU frequencies (Intel AMX, Apple SME) run in separate passes to avoid affecting throughput measurements of other kernels.

### Intel Sapphire Rapids

#### Native

| Kernel                         |                      256 |                     1024 |                     4096 |
| :----------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f64__                        | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_f64_serial`    |       8.31 gb/s, 0.1 ulp |         8.14 gb/s, 0 ulp |         8.24 gb/s, 0 ulp |
| `nk_euclidean_f64_serial`      |       7.30 gb/s, 0.6 ulp |       8.08 gb/s, 0.5 ulp |       8.00 gb/s, 0.5 ulp |
| `nk_angular_f64_serial`        |         2.97 gb/s, 0 ulp |         2.94 gb/s, 0 ulp |         3.17 gb/s, 0 ulp |
| `nk_sqeuclidean_f64_skylake`   |       31.6 gb/s, 0.4 ulp |       31.4 gb/s, 0.7 ulp |       29.2 gb/s, 1.3 ulp |
| `nk_euclidean_f64_skylake`     |       29.9 gb/s, 0.3 ulp |       31.1 gb/s, 0.4 ulp |       28.7 gb/s, 0.7 ulp |
| `nk_angular_f64_skylake`       |         25.8 gb/s, 0 ulp |         28.1 gb/s, 0 ulp |         23.9 gb/s, 0 ulp |
| __f32__                        | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_f32_serial`    |         4.15 gb/s, 0 ulp |         3.89 gb/s, 0 ulp |         4.01 gb/s, 0 ulp |
| `nk_euclidean_f32_serial`      |       3.78 gb/s, 0.1 ulp |       3.94 gb/s, 0.1 ulp |       4.02 gb/s, 0.1 ulp |
| `nk_angular_f32_serial`        |         1.38 gb/s, 0 ulp |         1.39 gb/s, 0 ulp |         1.49 gb/s, 0 ulp |
| `nk_sqeuclidean_f32_skylake`   |         36.9 gb/s, 0 ulp |         29.4 gb/s, 0 ulp |         28.1 gb/s, 0 ulp |
| `nk_euclidean_f32_skylake`     |       34.9 gb/s, 0.1 ulp |       29.7 gb/s, 0.1 ulp |       28.1 gb/s, 0.1 ulp |
| `nk_angular_f32_skylake`       |         32.1 gb/s, 0 ulp |         28.0 gb/s, 0 ulp |         26.8 gb/s, 0 ulp |
| __bf16__                       | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_bf16_serial`   |        0.575 gb/s, 0 ulp |        0.533 gb/s, 0 ulp |        0.552 gb/s, 0 ulp |
| `nk_euclidean_bf16_serial`     |      0.524 gb/s, 0.5 ulp |      0.533 gb/s, 0.5 ulp |      0.541 gb/s, 0.5 ulp |
| `nk_angular_bf16_serial`       |        0.467 gb/s, 0 ulp |        0.403 gb/s, 0 ulp |        0.443 gb/s, 0 ulp |
| `nk_sqeuclidean_bf16_haswell`  |       30.0 gb/s, 0.5 ulp |       19.3 gb/s, 7.4 ulp |        18.6 gb/s, 27 ulp |
| `nk_euclidean_bf16_haswell`    |       25.6 gb/s, 0.3 ulp |       18.8 gb/s, 4.1 ulp |        18.4 gb/s, 15 ulp |
| `nk_angular_bf16_haswell`      |         21.5 gb/s, 0 ulp |         18.4 gb/s, 0 ulp |       18.3 gb/s, 0.2 ulp |
| `nk_sqeuclidean_bf16_genoa`    |       27.8 gb/s, 0.3 ulp |       24.3 gb/s, 0.5 ulp |      25.8 gb/s, 10.5 ulp |
| `nk_euclidean_bf16_genoa`      |       26.7 gb/s, 0.2 ulp |       24.2 gb/s, 0.3 ulp |       26.1 gb/s, 5.8 ulp |
| `nk_angular_bf16_genoa`        |         36.5 gb/s, 0 ulp |         29.4 gb/s, 0 ulp |       27.1 gb/s, 0.1 ulp |
| __f16__                        | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_f16_serial`    |      0.986 gb/s, 0.1 ulp |      0.917 gb/s, 0.1 ulp |      0.919 gb/s, 0.1 ulp |
| `nk_euclidean_f16_serial`      |      0.936 gb/s, 0.6 ulp |      0.918 gb/s, 0.5 ulp |      0.944 gb/s, 0.5 ulp |
| `nk_angular_f16_serial`        |        0.908 gb/s, 0 ulp |        0.847 gb/s, 0 ulp |        0.800 gb/s, 0 ulp |
| `nk_sqeuclidean_f16_haswell`   |       24.7 gb/s, 0.4 ulp |       22.5 gb/s, 1.4 ulp |       16.2 gb/s, 5.1 ulp |
| `nk_euclidean_f16_haswell`     |       27.2 gb/s, 0.3 ulp |       21.7 gb/s, 0.8 ulp |       19.4 gb/s, 2.8 ulp |
| `nk_angular_f16_haswell`       |       20.1 gb/s, 0.1 ulp |       19.2 gb/s, 0.1 ulp |       18.7 gb/s, 0.1 ulp |
| __e5m2__                       | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_e5m2_serial`   |        0.498 gb/s, 0 ulp |        0.467 gb/s, 0 ulp |        0.482 gb/s, 0 ulp |
| `nk_euclidean_e5m2_serial`     |      0.480 gb/s, 0.5 ulp |      0.470 gb/s, 0.5 ulp |      0.506 gb/s, 0.5 ulp |
| `nk_angular_e5m2_serial`       |        0.206 gb/s, 0 ulp |        0.224 gb/s, 0 ulp |        0.231 gb/s, 0 ulp |
| `nk_sqeuclidean_e5m2_skylake`  |         6.69 gb/s, 0 ulp |         6.03 gb/s, 0 ulp |         6.01 gb/s, 0 ulp |
| `nk_euclidean_e5m2_skylake`    |         5.76 gb/s, 0 ulp |         5.98 gb/s, 0 ulp |         6.01 gb/s, 0 ulp |
| `nk_angular_e5m2_skylake`      |         5.55 gb/s, 0 ulp |         5.55 gb/s, 0 ulp |         5.59 gb/s, 0 ulp |
| `nk_sqeuclidean_e5m2_genoa`    |         6.14 gb/s, 0 ulp |         6.25 gb/s, 0 ulp |         5.85 gb/s, 0 ulp |
| `nk_euclidean_e5m2_genoa`      |         5.55 gb/s, 0 ulp |         5.87 gb/s, 0 ulp |         6.03 gb/s, 0 ulp |
| `nk_angular_e5m2_genoa`        |         8.90 gb/s, 0 ulp |         9.51 gb/s, 0 ulp |         9.59 gb/s, 0 ulp |
| __e4m3__                       | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_e4m3_serial`   |        0.316 gb/s, 0 ulp |        0.275 gb/s, 0 ulp |        0.288 gb/s, 0 ulp |
| `nk_euclidean_e4m3_serial`     |      0.292 gb/s, 0.5 ulp |      0.278 gb/s, 0.5 ulp |      0.273 gb/s, 0.5 ulp |
| `nk_angular_e4m3_serial`       |        0.175 gb/s, 0 ulp |        0.167 gb/s, 0 ulp |        0.179 gb/s, 0 ulp |
| `nk_sqeuclidean_e4m3_skylake`  |         4.84 gb/s, 0 ulp |         4.65 gb/s, 0 ulp |       4.68 gb/s, 0.2 ulp |
| `nk_euclidean_e4m3_skylake`    |         4.50 gb/s, 0 ulp |         4.72 gb/s, 0 ulp |       4.59 gb/s, 0.2 ulp |
| `nk_angular_e4m3_skylake`      |         4.86 gb/s, 0 ulp |         4.49 gb/s, 0 ulp |         4.46 gb/s, 0 ulp |
| `nk_sqeuclidean_e4m3_genoa`    |         5.98 gb/s, 0 ulp |         6.26 gb/s, 0 ulp |       6.07 gb/s, 0.2 ulp |
| `nk_euclidean_e4m3_genoa`      |         5.96 gb/s, 0 ulp |         6.12 gb/s, 0 ulp |       5.98 gb/s, 0.2 ulp |
| `nk_angular_e4m3_genoa`        |         8.82 gb/s, 0 ulp |         9.92 gb/s, 0 ulp |         9.37 gb/s, 0 ulp |
| `nk_sqeuclidean_e4m3_sapphire` |         4.89 gb/s, 0 ulp |         4.75 gb/s, 0 ulp |       4.70 gb/s, 0.2 ulp |
| `nk_euclidean_e4m3_sapphire`   |         4.71 gb/s, 0 ulp |         4.74 gb/s, 0 ulp |       4.73 gb/s, 0.2 ulp |
| __e3m2__                       | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_e3m2_serial`   |        0.510 gb/s, 0 ulp |        0.486 gb/s, 0 ulp |        0.495 gb/s, 0 ulp |
| `nk_euclidean_e3m2_serial`     |      0.491 gb/s, 0.5 ulp |      0.487 gb/s, 0.5 ulp |      0.478 gb/s, 0.4 ulp |
| `nk_angular_e3m2_serial`       |        0.211 gb/s, 0 ulp |        0.219 gb/s, 0 ulp |        0.215 gb/s, 0 ulp |
| `nk_sqeuclidean_e3m2_skylake`  |         6.58 gb/s, 0 ulp |         5.87 gb/s, 0 ulp |         5.92 gb/s, 0 ulp |
| `nk_euclidean_e3m2_skylake`    |         5.79 gb/s, 0 ulp |         5.98 gb/s, 0 ulp |         5.94 gb/s, 0 ulp |
| `nk_angular_e3m2_skylake`      |         5.98 gb/s, 0 ulp |         5.88 gb/s, 0 ulp |         5.54 gb/s, 0 ulp |
| `nk_sqeuclidean_e3m2_genoa`    |         7.68 gb/s, 0 ulp |         8.27 gb/s, 0 ulp |         7.99 gb/s, 0 ulp |
| `nk_euclidean_e3m2_genoa`      |         7.60 gb/s, 0 ulp |         8.44 gb/s, 0 ulp |         8.39 gb/s, 0 ulp |
| `nk_angular_e3m2_genoa`        |         11.5 gb/s, 0 ulp |         13.0 gb/s, 0 ulp |         12.6 gb/s, 0 ulp |
| `nk_sqeuclidean_e3m2_sapphire` |        24.2 gb/s, 1K ulp |        22.2 gb/s, 1K ulp |        21.4 gb/s, 1K ulp |
| `nk_euclidean_e3m2_sapphire`   |       20.0 gb/s, 563 ulp |       21.6 gb/s, 569 ulp |       21.6 gb/s, 568 ulp |
| `nk_angular_e3m2_sapphire`     |        14.1 gb/s, 11 ulp |       17.2 gb/s, 5.8 ulp |       18.4 gb/s, 3.1 ulp |
| __e2m3__                       | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_e2m3_serial`   |        0.554 gb/s, 0 ulp |        0.497 gb/s, 0 ulp |        0.494 gb/s, 0 ulp |
| `nk_euclidean_e2m3_serial`     |      0.499 gb/s, 0.5 ulp |      0.466 gb/s, 0.5 ulp |      0.502 gb/s, 0.5 ulp |
| `nk_angular_e2m3_serial`       |        0.211 gb/s, 0 ulp |        0.218 gb/s, 0 ulp |        0.221 gb/s, 0 ulp |
| `nk_sqeuclidean_e2m3_skylake`  |         6.58 gb/s, 0 ulp |         6.09 gb/s, 0 ulp |         5.83 gb/s, 0 ulp |
| `nk_euclidean_e2m3_skylake`    |         5.77 gb/s, 0 ulp |         5.91 gb/s, 0 ulp |         5.92 gb/s, 0 ulp |
| `nk_angular_e2m3_skylake`      |         6.03 gb/s, 0 ulp |         5.74 gb/s, 0 ulp |         5.61 gb/s, 0 ulp |
| `nk_sqeuclidean_e2m3_genoa`    |         7.66 gb/s, 0 ulp |         8.20 gb/s, 0 ulp |         8.25 gb/s, 0 ulp |
| `nk_euclidean_e2m3_genoa`      |         7.47 gb/s, 0 ulp |         8.17 gb/s, 0 ulp |         8.03 gb/s, 0 ulp |
| `nk_angular_e2m3_genoa`        |         11.3 gb/s, 0 ulp |         12.6 gb/s, 0 ulp |         12.4 gb/s, 0 ulp |
| `nk_sqeuclidean_e2m3_sapphire` |       23.9 gb/s, 352 ulp |       22.1 gb/s, 273 ulp |       21.9 gb/s, 254 ulp |
| `nk_euclidean_e2m3_sapphire`   |       19.7 gb/s, 192 ulp |       21.7 gb/s, 150 ulp |       21.8 gb/s, 140 ulp |
| `nk_angular_e2m3_sapphire`     |       14.3 gb/s, 2.8 ulp |       16.3 gb/s, 1.4 ulp |       18.1 gb/s, 0.7 ulp |
| __i8__                         | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_i8_serial`     |                35.8 gb/s |                25.8 gb/s |                19.3 gb/s |
| `nk_euclidean_i8_serial`       |                26.8 gb/s |                23.3 gb/s |                19.0 gb/s |
| `nk_angular_i8_serial`         |                7.97 gb/s |                8.16 gb/s |                8.14 gb/s |
| `nk_sqeuclidean_i8_haswell`    |                26.4 gb/s |                33.0 gb/s |                26.1 gb/s |
| `nk_euclidean_i8_haswell`      |                32.6 gb/s |                31.6 gb/s |                26.2 gb/s |
| `nk_angular_i8_haswell`        |                19.9 gb/s |                24.8 gb/s |                22.7 gb/s |
| `nk_sqeuclidean_i8_icelake`    |                58.7 gb/s |                44.1 gb/s |                28.9 gb/s |
| `nk_euclidean_i8_icelake`      |                48.9 gb/s |                41.3 gb/s |                29.2 gb/s |
| `nk_angular_i8_icelake`        |                24.7 gb/s |                29.9 gb/s |                26.0 gb/s |
| __u8__                         | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_u8_serial`     |                11.7 gb/s |                11.1 gb/s |                10.5 gb/s |
| `nk_euclidean_u8_serial`       |                10.5 gb/s |                11.2 gb/s |                10.5 gb/s |
| `nk_angular_u8_serial`         |                8.07 gb/s |                8.16 gb/s |                7.90 gb/s |
| `nk_sqeuclidean_u8_haswell`    |                44.3 gb/s |                34.2 gb/s |                27.0 gb/s |
| `nk_euclidean_u8_haswell`      |                36.7 gb/s |                34.3 gb/s |                27.2 gb/s |
| `nk_angular_u8_haswell`        |                22.0 gb/s |                27.4 gb/s |                23.8 gb/s |
| `nk_sqeuclidean_u8_icelake`    |                65.7 gb/s |                46.5 gb/s |                30.0 gb/s |
| `nk_euclidean_u8_icelake`      |                57.7 gb/s |                43.5 gb/s |                30.1 gb/s |
| `nk_angular_u8_icelake`        |                28.6 gb/s |                33.1 gb/s |                27.9 gb/s |
| __i4__                         | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_i4_serial`     |                15.8 gb/s |                16.3 gb/s |                14.4 gb/s |
| `nk_euclidean_i4_serial`       |                11.1 gb/s |                14.9 gb/s |                14.6 gb/s |
| `nk_angular_i4_serial`         |                5.90 gb/s |                6.47 gb/s |                6.23 gb/s |
| `nk_sqeuclidean_i4_icelake`    |                42.2 gb/s |                46.4 gb/s |                27.9 gb/s |
| `nk_euclidean_i4_icelake`      |                33.6 gb/s |                37.7 gb/s |                27.1 gb/s |
| `nk_angular_i4_icelake`        |                11.3 gb/s |                18.9 gb/s |                20.8 gb/s |
| __u4__                         | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_u4_serial`     |                15.8 gb/s |                16.6 gb/s |                14.6 gb/s |
| `nk_euclidean_u4_serial`       |                11.2 gb/s |                14.8 gb/s |                14.7 gb/s |
| `nk_angular_u4_serial`         |                5.48 gb/s |                6.45 gb/s |                6.27 gb/s |
| `nk_sqeuclidean_u4_icelake`    |                38.5 gb/s |                45.3 gb/s |                27.2 gb/s |
| `nk_euclidean_u4_icelake`      |                31.2 gb/s |                35.8 gb/s |                27.4 gb/s |
| `nk_angular_u4_icelake`        |                14.0 gb/s |                28.0 gb/s |                23.9 gb/s |

#### WASM

Measured with Wasmtime v42 (Cranelift backend).

| Kernel                            |                      256 |                     1024 |                     4096 |
| :-------------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f64__                           | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_f64_serial`       |          ? gb/s, 0.1 ulp |            ? gb/s, 0 ulp |            ? gb/s, 0 ulp |
| `nk_euclidean_f64_serial`         |          ? gb/s, 0.6 ulp |          ? gb/s, 0.6 ulp |          ? gb/s, 0.6 ulp |
| `nk_angular_f64_serial`           |            ? gb/s, 0 ulp |            ? gb/s, 0 ulp |            ? gb/s, 0 ulp |
| `nk_sqeuclidean_f64_v128relaxed`  |          ? gb/s, 1.3 ulp |          ? gb/s, 2.6 ulp |          ? gb/s, 4.5 ulp |
| `nk_euclidean_f64_v128relaxed`    |          ? gb/s, 0.7 ulp |          ? gb/s, 1.4 ulp |          ? gb/s, 2.5 ulp |
| `nk_angular_f64_v128relaxed`      |          ? gb/s, 0.1 ulp |          ? gb/s, 0.1 ulp |          ? gb/s, 0.1 ulp |
| __f32__                           | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_f32_serial`       |            ? gb/s, 0 ulp |            ? gb/s, 0 ulp |            ? gb/s, 0 ulp |
| `nk_euclidean_f32_serial`         |          ? gb/s, 0.1 ulp |          ? gb/s, 0.1 ulp |          ? gb/s, 0.2 ulp |
| `nk_angular_f32_serial`           |            ? gb/s, 0 ulp |            ? gb/s, 0 ulp |            ? gb/s, 0 ulp |
| `nk_sqeuclidean_f32_v128relaxed`  |          ? gb/s, 0.7 ulp |          ? gb/s, 1.3 ulp |          ? gb/s, 2.6 ulp |
| `nk_euclidean_f32_v128relaxed`    |          ? gb/s, 0.4 ulp |          ? gb/s, 0.7 ulp |          ? gb/s, 1.4 ulp |
| `nk_angular_f32_v128relaxed`      |            ? gb/s, 0 ulp |            ? gb/s, 0 ulp |            ? gb/s, 0 ulp |
| __bf16__                          | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_bf16_serial`      |            ? gb/s, 0 ulp |            ? gb/s, 0 ulp |            ? gb/s, 0 ulp |
| `nk_euclidean_bf16_serial`        |          ? gb/s, 0.6 ulp |          ? gb/s, 0.5 ulp |          ? gb/s, 0.5 ulp |
| `nk_angular_bf16_serial`          |            ? gb/s, 0 ulp |            ? gb/s, 0 ulp |            ? gb/s, 0 ulp |
| `nk_sqeuclidean_bf16_v128relaxed` |          ? gb/s, 0.9 ulp |           ? gb/s, 13 ulp |           ? gb/s, 21 ulp |
| `nk_euclidean_bf16_v128relaxed`   |          ? gb/s, 0.5 ulp |          ? gb/s, 6.9 ulp |           ? gb/s, 11 ulp |
| `nk_angular_bf16_v128relaxed`     |            ? gb/s, 0 ulp |          ? gb/s, 0.3 ulp |          ? gb/s, 0.6 ulp |
| __f16__                           | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_f16_serial`       |          ? gb/s, 0.1 ulp |          ? gb/s, 0.1 ulp |          ? gb/s, 0.1 ulp |
| `nk_euclidean_f16_serial`         |          ? gb/s, 0.6 ulp |          ? gb/s, 0.6 ulp |          ? gb/s, 0.5 ulp |
| `nk_angular_f16_serial`           |            ? gb/s, 0 ulp |            ? gb/s, 0 ulp |            ? gb/s, 0 ulp |
| `nk_sqeuclidean_f16_v128relaxed`  |          ? gb/s, 0.9 ulp |          ? gb/s, 3.6 ulp |          ? gb/s, 9.8 ulp |
| `nk_euclidean_f16_v128relaxed`    |          ? gb/s, 0.5 ulp |          ? gb/s, 2.1 ulp |          ? gb/s, 5.4 ulp |
| `nk_angular_f16_v128relaxed`      |          ? gb/s, 0.1 ulp |          ? gb/s, 0.1 ulp |          ? gb/s, 0.1 ulp |
| __i8__                            | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_i8_serial`        |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_euclidean_i8_serial`          |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_angular_i8_serial`            |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_sqeuclidean_i8_v128relaxed`   |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_euclidean_i8_v128relaxed`     |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_angular_i8_v128relaxed`       |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| __u8__                            | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_u8_serial`        |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_euclidean_u8_serial`          |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_angular_u8_serial`            |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_sqeuclidean_u8_v128relaxed`   |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_euclidean_u8_v128relaxed`     |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_angular_u8_v128relaxed`       |                   0 gb/s |                   0 gb/s |                   0 gb/s |

### Apple M4 Max

#### Native

| Kernel                          |                      256 |                     1024 |                     4096 |
| :------------------------------ | -----------------------: | -----------------------: | -----------------------: |
| __f64__                         | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_f64_serial`     |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_euclidean_f64_serial`       |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_angular_f64_serial`         |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_sqeuclidean_f64_neon`       |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_euclidean_f64_neon`         |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_angular_f64_neon`           |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| __f32__                         | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_f32_serial`     |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_euclidean_f32_serial`       |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_angular_f32_serial`         |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_sqeuclidean_f32_neon`       |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_euclidean_f32_neon`         |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_angular_f32_neon`           |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| __bf16__                        | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_bf16_serial`    |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_euclidean_bf16_serial`      |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_angular_bf16_serial`        |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_sqeuclidean_bf16_neonbfdot` |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_euclidean_bf16_neonbfdot`   |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_angular_bf16_neonbfdot`     |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| __f16__                         | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_f16_serial`     |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_euclidean_f16_serial`       |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_angular_f16_serial`         |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_sqeuclidean_f16_neonhalf`   |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_euclidean_f16_neonhalf`     |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_angular_f16_neonhalf`       |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| __e5m2__                        | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_e5m2_serial`    |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_euclidean_e5m2_serial`      |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_angular_e5m2_serial`        |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| __e4m3__                        | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_e4m3_serial`    |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_euclidean_e4m3_serial`      |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_angular_e4m3_serial`        |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| __e3m2__                        | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_e3m2_serial`    |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_euclidean_e3m2_serial`      |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_angular_e3m2_serial`        |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_sqeuclidean_e3m2_neon`      |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_euclidean_e3m2_neon`        |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_angular_e3m2_neon`          |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| __e2m3__                        | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_e2m3_serial`    |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_euclidean_e2m3_serial`      |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_angular_e2m3_serial`        |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_sqeuclidean_e2m3_neon`      |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_euclidean_e2m3_neon`        |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_angular_e2m3_neon`          |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| __i8__                          | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_i8_serial`      |                   ? gb/s |                   ? gb/s |                   ? gb/s |
| `nk_euclidean_i8_serial`        |                   ? gb/s |                   ? gb/s |                   ? gb/s |
| `nk_angular_i8_serial`          |                   ? gb/s |                   ? gb/s |                   ? gb/s |
| `nk_sqeuclidean_i8_neonsdot`    |                   ? gb/s |                   ? gb/s |                   ? gb/s |
| `nk_euclidean_i8_neonsdot`      |                   ? gb/s |                   ? gb/s |                   ? gb/s |
| `nk_angular_i8_neonsdot`        |                   ? gb/s |                   ? gb/s |                   ? gb/s |
| __u8__                          | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_u8_serial`      |                   ? gb/s |                   ? gb/s |                   ? gb/s |
| `nk_euclidean_u8_serial`        |                   ? gb/s |                   ? gb/s |                   ? gb/s |
| `nk_angular_u8_serial`          |                   ? gb/s |                   ? gb/s |                   ? gb/s |
| `nk_sqeuclidean_u8_neonsdot`    |                   ? gb/s |                   ? gb/s |                   ? gb/s |
| `nk_euclidean_u8_neonsdot`      |                   ? gb/s |                   ? gb/s |                   ? gb/s |
| `nk_angular_u8_neonsdot`        |                   ? gb/s |                   ? gb/s |                   ? gb/s |
| __i4__                          | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_i4_serial`      |                   ? gb/s |                   ? gb/s |                   ? gb/s |
| `nk_euclidean_i4_serial`        |                   ? gb/s |                   ? gb/s |                   ? gb/s |
| `nk_angular_i4_serial`          |                   ? gb/s |                   ? gb/s |                   ? gb/s |
| __u4__                          | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_u4_serial`      |                   ? gb/s |                   ? gb/s |                   ? gb/s |
| `nk_euclidean_u4_serial`        |                   ? gb/s |                   ? gb/s |                   ? gb/s |
| `nk_angular_u4_serial`          |                   ? gb/s |                   ? gb/s |                   ? gb/s |

#### WASM

Measured with Wasmtime v42 (Cranelift backend).

| Kernel                            |                      256 |                     1024 |                     4096 |
| :-------------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f64__                           | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_f64_serial`       |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_sqeuclidean_f64_v128relaxed`  |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_euclidean_f64_serial`         |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_euclidean_f64_v128relaxed`    |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_angular_f64_serial`           |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_angular_f64_v128relaxed`      |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| __f32__                           | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_f32_serial`       |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_sqeuclidean_f32_v128relaxed`  |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_euclidean_f32_serial`         |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_euclidean_f32_v128relaxed`    |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_angular_f32_serial`           |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_angular_f32_v128relaxed`      |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| __bf16__                          | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_bf16_serial`      |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_sqeuclidean_bf16_v128relaxed` |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_euclidean_bf16_serial`        |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_euclidean_bf16_v128relaxed`   |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_angular_bf16_serial`          |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_angular_bf16_v128relaxed`     |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| __f16__                           | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_f16_serial`       |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_sqeuclidean_f16_v128relaxed`  |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_euclidean_f16_serial`         |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_euclidean_f16_v128relaxed`    |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_angular_f16_serial`           |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_angular_f16_v128relaxed`      |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| __i8__                            | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_i8_serial`        |                   ? gb/s |                   ? gb/s |                   ? gb/s |
| `nk_sqeuclidean_i8_v128relaxed`   |                   ? gb/s |                   ? gb/s |                   ? gb/s |
| `nk_euclidean_i8_serial`          |                   ? gb/s |                   ? gb/s |                   ? gb/s |
| `nk_euclidean_i8_v128relaxed`     |                   ? gb/s |                   ? gb/s |                   ? gb/s |
| `nk_angular_i8_serial`            |                   ? gb/s |                   ? gb/s |                   ? gb/s |
| `nk_angular_i8_v128relaxed`       |                   ? gb/s |                   ? gb/s |                   ? gb/s |
| __u8__                            | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_u8_serial`        |                   ? gb/s |                   ? gb/s |                   ? gb/s |
| `nk_sqeuclidean_u8_v128relaxed`   |                   ? gb/s |                   ? gb/s |                   ? gb/s |
| `nk_euclidean_u8_serial`          |                   ? gb/s |                   ? gb/s |                   ? gb/s |
| `nk_euclidean_u8_v128relaxed`     |                   ? gb/s |                   ? gb/s |                   ? gb/s |
| `nk_angular_u8_serial`            |                   ? gb/s |                   ? gb/s |                   ? gb/s |
| `nk_angular_u8_v128relaxed`       |                   ? gb/s |                   ? gb/s |                   ? gb/s |

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

Controlled by `NK_DENSE_DIMENSIONS`.
Columns show 256, 1024, 4096 elements.

### Intel Sapphire Rapids

#### Native

| Kernel                         |           256 |          1024 |          4096 |
| :----------------------------- | ------------: | ------------: | ------------: |
| __f64__                        |               |               |               |
| `nk_sqeuclidean_f64_serial`    | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_euclidean_f64_serial`      | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_angular_f64_serial`        | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_sqeuclidean_f64_haswell`   | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_euclidean_f64_haswell`     | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_angular_f64_haswell`       | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_sqeuclidean_f64_skylake`   | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_euclidean_f64_skylake`     | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_angular_f64_skylake`       | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| __f32__                        |               |               |               |
| `nk_sqeuclidean_f32_serial`    | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_euclidean_f32_serial`      | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_angular_f32_serial`        | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_sqeuclidean_f32_haswell`   | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_euclidean_f32_haswell`     | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_angular_f32_haswell`       | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_sqeuclidean_f32_skylake`   | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_euclidean_f32_skylake`     | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_angular_f32_skylake`       | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| __bf16__                       |               |               |               |
| `nk_sqeuclidean_bf16_serial`   | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_euclidean_bf16_serial`     | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_angular_bf16_serial`       | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_sqeuclidean_bf16_haswell`  | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_euclidean_bf16_haswell`    | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_angular_bf16_haswell`      | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_sqeuclidean_bf16_genoa`    | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_euclidean_bf16_genoa`      | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_angular_bf16_genoa`        | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| __f16__                        |               |               |               |
| `nk_sqeuclidean_f16_serial`    | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_euclidean_f16_serial`      | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_angular_f16_serial`        | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_sqeuclidean_f16_haswell`   | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_euclidean_f16_haswell`     | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_angular_f16_haswell`       | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_sqeuclidean_f16_skylake`   | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_euclidean_f16_skylake`     | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_angular_f16_skylake`       | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| __e5m2__                       |               |               |               |
| `nk_sqeuclidean_e5m2_serial`   | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_euclidean_e5m2_serial`     | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_angular_e5m2_serial`       | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_sqeuclidean_e5m2_skylake`  | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_euclidean_e5m2_skylake`    | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_angular_e5m2_skylake`      | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_sqeuclidean_e5m2_genoa`    | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_euclidean_e5m2_genoa`      | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_angular_e5m2_genoa`        | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| __e4m3__                       |               |               |               |
| `nk_sqeuclidean_e4m3_serial`   | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_euclidean_e4m3_serial`     | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_angular_e4m3_serial`       | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_sqeuclidean_e4m3_skylake`  | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_euclidean_e4m3_skylake`    | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_angular_e4m3_skylake`      | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_sqeuclidean_e4m3_genoa`    | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_euclidean_e4m3_genoa`      | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_angular_e4m3_genoa`        | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_sqeuclidean_e4m3_sapphire` | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_euclidean_e4m3_sapphire`   | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| __e3m2__                       |               |               |               |
| `nk_sqeuclidean_e3m2_serial`   | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_euclidean_e3m2_serial`     | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_angular_e3m2_serial`       | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_sqeuclidean_e3m2_haswell`  | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_euclidean_e3m2_haswell`    | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_angular_e3m2_haswell`      | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_sqeuclidean_e3m2_skylake`  | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_euclidean_e3m2_skylake`    | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_angular_e3m2_skylake`      | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_sqeuclidean_e3m2_genoa`    | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_euclidean_e3m2_genoa`      | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_angular_e3m2_genoa`        | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_sqeuclidean_e3m2_sapphire` | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_euclidean_e3m2_sapphire`   | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_angular_e3m2_sapphire`     | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| __e2m3__                       |               |               |               |
| `nk_sqeuclidean_e2m3_serial`   | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_euclidean_e2m3_serial`     | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_angular_e2m3_serial`       | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_sqeuclidean_e2m3_haswell`  | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_euclidean_e2m3_haswell`    | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_angular_e2m3_haswell`      | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_sqeuclidean_e2m3_skylake`  | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_euclidean_e2m3_skylake`    | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_angular_e2m3_skylake`      | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_sqeuclidean_e2m3_genoa`    | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_euclidean_e2m3_genoa`      | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_angular_e2m3_genoa`        | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_sqeuclidean_e2m3_sapphire` | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_euclidean_e2m3_sapphire`   | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_angular_e2m3_sapphire`     | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| __i8__                         |               |               |               |
| `nk_sqeuclidean_i8_serial`     |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_euclidean_i8_serial`       |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_angular_i8_serial`         |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_sqeuclidean_i8_haswell`    |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_euclidean_i8_haswell`      |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_angular_i8_haswell`        |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_sqeuclidean_i8_icelake`    |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_euclidean_i8_icelake`      |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_angular_i8_icelake`        |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_angular_i8_alder`          |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_angular_i8_sierra`         |        0 GB/s |        0 GB/s |        0 GB/s |
| __u8__                         |               |               |               |
| `nk_sqeuclidean_u8_serial`     |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_euclidean_u8_serial`       |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_angular_u8_serial`         |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_sqeuclidean_u8_haswell`    |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_euclidean_u8_haswell`      |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_angular_u8_haswell`        |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_sqeuclidean_u8_icelake`    |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_euclidean_u8_icelake`      |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_angular_u8_icelake`        |        0 GB/s |        0 GB/s |        0 GB/s |
| __i4__                         |               |               |               |
| `nk_sqeuclidean_i4_serial`     |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_euclidean_i4_serial`       |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_angular_i4_serial`         |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_sqeuclidean_i4_icelake`    |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_euclidean_i4_icelake`      |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_angular_i4_icelake`        |        0 GB/s |        0 GB/s |        0 GB/s |
| __u4__                         |               |               |               |
| `nk_sqeuclidean_u4_serial`     |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_euclidean_u4_serial`       |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_angular_u4_serial`         |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_sqeuclidean_u4_icelake`    |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_euclidean_u4_icelake`      |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_angular_u4_icelake`        |        0 GB/s |        0 GB/s |        0 GB/s |

#### V8

| Kernel                            |           256 |          1024 |          4096 |
| :-------------------------------- | ------------: | ------------: | ------------: |
| __f64__                           |               |               |               |
| `nk_sqeuclidean_f64_v128relaxed`  | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_euclidean_f64_v128relaxed`    | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_angular_f64_v128relaxed`      | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| __f32__                           |               |               |               |
| `nk_sqeuclidean_f32_v128relaxed`  | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_euclidean_f32_v128relaxed`    | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_angular_f32_v128relaxed`      | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| __bf16__                          |               |               |               |
| `nk_sqeuclidean_bf16_v128relaxed` | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_euclidean_bf16_v128relaxed`   | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_angular_bf16_v128relaxed`     | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| __f16__                           |               |               |               |
| `nk_sqeuclidean_f16_v128relaxed`  | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_euclidean_f16_v128relaxed`    | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_angular_f16_v128relaxed`      | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| __i8__                            |               |               |               |
| `nk_sqeuclidean_i8_v128relaxed`   |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_euclidean_i8_v128relaxed`     |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_angular_i8_v128relaxed`       |        0 GB/s |        0 GB/s |        0 GB/s |
| __u8__                            |               |               |               |
| `nk_sqeuclidean_u8_v128relaxed`   |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_euclidean_u8_v128relaxed`     |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_angular_u8_v128relaxed`       |        0 GB/s |        0 GB/s |        0 GB/s |

#### Wasmtime

| Kernel                            |           256 |          1024 |          4096 |
| :-------------------------------- | ------------: | ------------: | ------------: |
| __f64__                           |               |               |               |
| `nk_sqeuclidean_f64_v128relaxed`  | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_euclidean_f64_v128relaxed`    | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_angular_f64_v128relaxed`      | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| __f32__                           |               |               |               |
| `nk_sqeuclidean_f32_v128relaxed`  | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_euclidean_f32_v128relaxed`    | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_angular_f32_v128relaxed`      | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| __bf16__                          |               |               |               |
| `nk_sqeuclidean_bf16_v128relaxed` | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_euclidean_bf16_v128relaxed`   | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_angular_bf16_v128relaxed`     | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| __f16__                           |               |               |               |
| `nk_sqeuclidean_f16_v128relaxed`  | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_euclidean_f16_v128relaxed`    | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_angular_f16_v128relaxed`      | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| __i8__                            |               |               |               |
| `nk_sqeuclidean_i8_v128relaxed`   |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_euclidean_i8_v128relaxed`     |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_angular_i8_v128relaxed`       |        0 GB/s |        0 GB/s |        0 GB/s |
| __u8__                            |               |               |               |
| `nk_sqeuclidean_u8_v128relaxed`   |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_euclidean_u8_v128relaxed`     |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_angular_u8_v128relaxed`       |        0 GB/s |        0 GB/s |        0 GB/s |

### Apple M4 Max

#### Native

| Kernel                          |           256 |          1024 |          4096 |
| :------------------------------ | ------------: | ------------: | ------------: |
| __f64__                         |               |               |               |
| `nk_sqeuclidean_f64_serial`     | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_euclidean_f64_serial`       | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_angular_f64_serial`         | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_sqeuclidean_f64_neon`       | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_euclidean_f64_neon`         | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_angular_f64_neon`           | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| __f32__                         |               |               |               |
| `nk_sqeuclidean_f32_serial`     | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_euclidean_f32_serial`       | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_angular_f32_serial`         | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_sqeuclidean_f32_neon`       | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_euclidean_f32_neon`         | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_angular_f32_neon`           | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| __bf16__                        |               |               |               |
| `nk_sqeuclidean_bf16_serial`    | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_euclidean_bf16_serial`      | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_angular_bf16_serial`        | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_sqeuclidean_bf16_neonbfdot` | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_euclidean_bf16_neonbfdot`   | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_angular_bf16_neonbfdot`     | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| __f16__                         |               |               |               |
| `nk_sqeuclidean_f16_serial`     | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_euclidean_f16_serial`       | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_angular_f16_serial`         | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_sqeuclidean_f16_neonhalf`   | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_euclidean_f16_neonhalf`     | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_angular_f16_neonhalf`       | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| __e5m2__                        |               |               |               |
| `nk_sqeuclidean_e5m2_serial`    | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_euclidean_e5m2_serial`      | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_angular_e5m2_serial`        | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| __e4m3__                        |               |               |               |
| `nk_sqeuclidean_e4m3_serial`    | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_euclidean_e4m3_serial`      | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_angular_e4m3_serial`        | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| __e3m2__                        |               |               |               |
| `nk_sqeuclidean_e3m2_serial`    | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_euclidean_e3m2_serial`      | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_angular_e3m2_serial`        | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_sqeuclidean_e3m2_neon`      | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_euclidean_e3m2_neon`        | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_angular_e3m2_neon`          | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| __e2m3__                        |               |               |               |
| `nk_sqeuclidean_e2m3_serial`    | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_euclidean_e2m3_serial`      | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_angular_e2m3_serial`        | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_sqeuclidean_e2m3_neon`      | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_euclidean_e2m3_neon`        | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_angular_e2m3_neon`          | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| __i8__                          |               |               |               |
| `nk_sqeuclidean_i8_serial`      |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_euclidean_i8_serial`        |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_angular_i8_serial`          |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_sqeuclidean_i8_neonsdot`    |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_euclidean_i8_neonsdot`      |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_angular_i8_neonsdot`        |        0 GB/s |        0 GB/s |        0 GB/s |
| __u8__                          |               |               |               |
| `nk_sqeuclidean_u8_serial`      |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_euclidean_u8_serial`        |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_angular_u8_serial`          |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_sqeuclidean_u8_neonsdot`    |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_euclidean_u8_neonsdot`      |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_angular_u8_neonsdot`        |        0 GB/s |        0 GB/s |        0 GB/s |
| __i4__                          |               |               |               |
| `nk_sqeuclidean_i4_serial`      |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_euclidean_i4_serial`        |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_angular_i4_serial`          |        0 GB/s |        0 GB/s |        0 GB/s |
| __u4__                          |               |               |               |
| `nk_sqeuclidean_u4_serial`      |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_euclidean_u4_serial`        |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_angular_u4_serial`          |        0 GB/s |        0 GB/s |        0 GB/s |

#### V8

| Kernel                            |           256 |          1024 |          4096 |
| :-------------------------------- | ------------: | ------------: | ------------: |
| __f64__                           |               |               |               |
| `nk_sqeuclidean_f64_v128relaxed`  | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_euclidean_f64_v128relaxed`    | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_angular_f64_v128relaxed`      | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| __f32__                           |               |               |               |
| `nk_sqeuclidean_f32_v128relaxed`  | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_euclidean_f32_v128relaxed`    | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_angular_f32_v128relaxed`      | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| __bf16__                          |               |               |               |
| `nk_sqeuclidean_bf16_v128relaxed` | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_euclidean_bf16_v128relaxed`   | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_angular_bf16_v128relaxed`     | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| __f16__                           |               |               |               |
| `nk_sqeuclidean_f16_v128relaxed`  | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_euclidean_f16_v128relaxed`    | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_angular_f16_v128relaxed`      | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| __i8__                            |               |               |               |
| `nk_sqeuclidean_i8_v128relaxed`   |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_euclidean_i8_v128relaxed`     |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_angular_i8_v128relaxed`       |        0 GB/s |        0 GB/s |        0 GB/s |
| __u8__                            |               |               |               |
| `nk_sqeuclidean_u8_v128relaxed`   |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_euclidean_u8_v128relaxed`     |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_angular_u8_v128relaxed`       |        0 GB/s |        0 GB/s |        0 GB/s |

#### Wasmtime

| Kernel                            |           256 |          1024 |          4096 |
| :-------------------------------- | ------------: | ------------: | ------------: |
| __f64__                           |               |               |               |
| `nk_sqeuclidean_f64_v128relaxed`  | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_euclidean_f64_v128relaxed`    | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_angular_f64_v128relaxed`      | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| __f32__                           |               |               |               |
| `nk_sqeuclidean_f32_v128relaxed`  | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_euclidean_f32_v128relaxed`    | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_angular_f32_v128relaxed`      | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| __bf16__                          |               |               |               |
| `nk_sqeuclidean_bf16_v128relaxed` | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_euclidean_bf16_v128relaxed`   | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_angular_bf16_v128relaxed`     | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| __f16__                           |               |               |               |
| `nk_sqeuclidean_f16_v128relaxed`  | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_euclidean_f16_v128relaxed`    | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_angular_f16_v128relaxed`      | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| __i8__                            |               |               |               |
| `nk_sqeuclidean_i8_v128relaxed`   |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_euclidean_i8_v128relaxed`     |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_angular_i8_v128relaxed`       |        0 GB/s |        0 GB/s |        0 GB/s |
| __u8__                            |               |               |               |
| `nk_sqeuclidean_u8_v128relaxed`   |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_euclidean_u8_v128relaxed`     |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_angular_u8_v128relaxed`       |        0 GB/s |        0 GB/s |        0 GB/s |

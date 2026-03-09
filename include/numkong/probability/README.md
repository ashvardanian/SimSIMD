# Divergence Measures for Probability Distributions in NumKong

NumKong implements divergence functions between discrete probability distributions: Kullback-Leibler divergence measures the information lost when one distribution approximates another, while Jensen-Shannon divergence provides a symmetric and bounded alternative.
These are used in variational inference, topic modeling, and distribution comparison tasks.

Kullback-Leibler divergence from $P$ to $Q$:

```math
\text{KLD}(P \| Q) = \sum_{i=0}^{n-1} P(i) \log_2 \frac{P(i)}{Q(i)}
```

Jensen-Shannon divergence symmetrizes KLD through a mixture:

```math
\text{JSD}(P, Q) = \frac{1}{2} \text{KLD}(P \| M) + \frac{1}{2} \text{KLD}(Q \| M)
```

where $M = \frac{P + Q}{2}$.

Reformulating as Python pseudocode:

```python
import numpy as np

def kld(p: np.ndarray, q: np.ndarray) -> float:
    mask = p > 0
    return np.sum(p[mask] * np.log2(p[mask] / q[mask]))

def jsd(p: np.ndarray, q: np.ndarray) -> float:
    m = (p + q) / 2
    return (kld(p, m) + kld(q, m)) / 2
```

## Input & Output Types

| Input Type | Output Type | Description                                    |
| ---------- | ----------- | ---------------------------------------------- |
| `f64`      | `f64`       | 64-bit IEEE 754 double precision               |
| `f32`      | `f32`       | 32-bit IEEE 754 single precision               |
| `f16`      | `f32`       | 16-bit IEEE 754 half precision, widened output |
| `bf16`     | `f32`       | 16-bit brain float, widened output             |

## Optimizations

### SIMD Log2 Approximation

`nk_kld_f32_skylake`, `nk_jsd_f32_skylake` use `VGETEXP` and `VGETMANT` to decompose floating-point values into exponent and mantissa components, then apply a polynomial approximation to the mantissa to compute $\log_2$.
The pipeline on Skylake is:

```
exponent = VGETEXPPS(x)
mantissa = VGETMANTPS(x, normalize_to_[1,2)) - 1
log2(x) ≈ exponent + polynomial(mantissa)
```

`VGETEXP` extracts the unbiased exponent as a float, while `VGETMANT` normalizes the mantissa to $[1, 2)$.
A degree-4 minimax polynomial over the normalized mantissa completes the approximation.
These instructions handle subnormals correctly without extra integer bit manipulation.

`nk_kld_f32_neon`, `nk_jsd_f32_neon`, `nk_kld_f16_haswell`, `nk_jsd_f16_haswell` use integer bit extraction instead:

```
exponent = (reinterpret_as_int(x) >> 23) - 127
mantissa = reinterpret_as_float((reinterpret_as_int(x) & 0x7FFFFF) | 0x3F800000) - 1
log2(x) ≈ exponent + c₁·m + c₂·m² + c₃·m³ + c₄·m⁴ + c₅·m⁵
```

This approach reinterprets the float as an integer, shifts out the mantissa bits to obtain the exponent, then masks and recombines to produce a normalized mantissa in $[1, 2)$.
It works on any ISA with integer-float reinterpretation and avoids the need for specialized exponent/mantissa instructions.

### Kahan Compensated Summation for F64

`nk_kld_f64_haswell`, `nk_jsd_f64_haswell` use Kahan compensated summation to maintain a running correction term alongside the accumulator.
The Kahan update for each divergence term is:

```
compensated_term = divergence_term - correction
tentative_sum    = accumulator + compensated_term
correction       = (tentative_sum - accumulator) - compensated_term
accumulator      = tentative_sum
```

After each $P(i) \log_2(P(i) / Q(i))$ term is computed, `correction` captures the low-order bits lost in the addition, and the next iteration subtracts this correction from the new term before adding it to the accumulator.
This keeps the accumulated error bounded by $O(1)$ ULP regardless of vector length, rather than the $O(n)$ ULP growth of naive summation.

## Performance

Controlled by `NK_DENSE_DIMENSIONS`.
Columns show 256, 1024, 4096 elements.

### Intel Sapphire Rapids

#### Native

| Kernel               |           256 |          1024 |          4096 |
| :------------------- | ------------: | ------------: | ------------: |
| __f64__              |               |               |               |
| `nk_kld_f64_serial`  | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_jsd_f64_serial`  | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_kld_f64_haswell` | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_jsd_f64_haswell` | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_kld_f64_skylake` | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_jsd_f64_skylake` | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| __f32__              |               |               |               |
| `nk_kld_f32_serial`  | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_jsd_f32_serial`  | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_kld_f32_skylake` | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_jsd_f32_skylake` | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| __bf16__             |               |               |               |
| `nk_kld_bf16_serial` | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_jsd_bf16_serial` | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| __f16__              |               |               |               |
| `nk_kld_f16_serial`  | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_jsd_f16_serial`  | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_kld_f16_haswell` | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_jsd_f16_haswell` | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_kld_f16_skylake` | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_jsd_f16_skylake` | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |

### Apple M4 Pro

#### Native

| Kernel                |           256 |          1024 |          4096 |
| :-------------------- | ------------: | ------------: | ------------: |
| __f64__               |               |               |               |
| `nk_kld_f64_serial`   | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_jsd_f64_serial`   | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| __f32__               |               |               |               |
| `nk_kld_f32_serial`   | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_jsd_f32_serial`   | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_kld_f32_neon`     | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_jsd_f32_neon`     | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| __bf16__              |               |               |               |
| `nk_kld_bf16_serial`  | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_jsd_bf16_serial`  | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| __f16__               |               |               |               |
| `nk_kld_f16_serial`   | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_jsd_f16_serial`   | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_kld_f16_neonhalf` | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_jsd_f16_neonhalf` | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |

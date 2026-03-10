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

The following performance tables are produced by manually re-running `nk_test` and `nk_bench` included internal tools to measure both accuracy and throughput at different input shapes.
The input size is controlled by the `NK_DENSE_DIMENSIONS` environment variable and set to 256, 1024, and 4096 elements.
The throughput is measured in GB/s as the number of input bytes per second.
Accuracy is reported as ULP (units in last place), the number of representable floating-point values between the result and the exact answer.
Each kernel runs for at least 20 seconds per configuration.
Benchmark threads are pinned to specific cores; on machines with heterogeneous core types (e.g., Apple P/E cores), only the fastest cores are used.
Workloads that significantly degrade CPU frequencies (Intel AMX, Apple SME) run in separate passes to avoid affecting throughput measurements of other kernels.

### Intel Sapphire Rapids

#### Native

| Kernel               |                      256 |                     1024 |                     4096 |
| :------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f64__              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_kld_f64_serial`  |     0.928 gb/s, 5.6K ulp |       1.18 gb/s, 25K ulp |      0.842 gb/s, 99K ulp |
| `nk_jsd_f64_serial`  |      0.512 gb/s, 0.4 ulp |      0.578 gb/s, 0.4 ulp |      0.427 gb/s, 0.5 ulp |
| `nk_kld_f64_haswell` |      9.00 gb/s, 5.6K ulp |       9.07 gb/s, 25K ulp |       8.85 gb/s, 99K ulp |
| `nk_jsd_f64_haswell` |       4.70 gb/s, 1.7 ulp |       4.88 gb/s, 1.4 ulp |       4.79 gb/s, 1.4 ulp |
| `nk_kld_f64_skylake` |      10.7 gb/s, 5.6K ulp |       11.7 gb/s, 25K ulp |       8.31 gb/s, 99K ulp |
| `nk_jsd_f64_skylake` |       5.82 gb/s, 1.7 ulp |       6.01 gb/s, 1.4 ulp |       4.59 gb/s, 1.4 ulp |
| __f32__              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_kld_f32_serial`  |       0.610 gb/s, 1K ulp |       0.714 gb/s, 5K ulp |      0.652 gb/s, 18K ulp |
| `nk_jsd_f32_serial`  |      0.322 gb/s, 0.4 ulp |      0.383 gb/s, 0.4 ulp |      0.324 gb/s, 4.6 ulp |
| `nk_kld_f32_skylake` |        14.8 gb/s, 1K ulp |        17.0 gb/s, 5K ulp |       17.7 gb/s, 18K ulp |
| `nk_jsd_f32_skylake` |       9.04 gb/s, 6.5 ulp |       8.86 gb/s, 6.9 ulp |        7.01 gb/s, 11 ulp |
| __bf16__             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_kld_bf16_serial` |       0.155 gb/s, 1K ulp |       0.209 gb/s, 5K ulp |      0.191 gb/s, 18K ulp |
| `nk_jsd_bf16_serial` |      0.097 gb/s, 1.5 ulp |      0.121 gb/s, 2.8 ulp |      0.112 gb/s, 9.1 ulp |
| __f16__              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_kld_f16_serial`  |       0.181 gb/s, 1K ulp |       0.181 gb/s, 5K ulp |      0.169 gb/s, 18K ulp |
| `nk_jsd_f16_serial`  |      0.185 gb/s, 1.4 ulp |      0.218 gb/s, 2.8 ulp |      0.204 gb/s, 8.1 ulp |
| `nk_kld_f16_haswell` |        8.82 gb/s, 1K ulp |        8.68 gb/s, 5K ulp |       8.42 gb/s, 18K ulp |
| `nk_jsd_f16_haswell` |       4.38 gb/s, 6.5 ulp |       4.53 gb/s, 6.9 ulp |        4.29 gb/s, 11 ulp |
| `nk_kld_f16_skylake` |        9.87 gb/s, 1K ulp |        9.47 gb/s, 5K ulp |       7.25 gb/s, 18K ulp |
| `nk_jsd_f16_skylake` |       4.60 gb/s, 6.5 ulp |       5.00 gb/s, 6.9 ulp |        3.87 gb/s, 11 ulp |

#### WASM

Measured with Wasmtime v42 (Cranelift backend).

| Kernel               |                      256 |                     1024 |                     4096 |
| :------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f64__              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_kld_f64_serial`  |         ? gb/s, 5.7K ulp |          ? gb/s, 23K ulp |          ? gb/s, 99K ulp |
| `nk_jsd_f64_serial`  |          ? gb/s, 0.5 ulp |          ? gb/s, 0.4 ulp |          ? gb/s, 0.7 ulp |
| __f32__              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_kld_f32_serial`  |           ? gb/s, 1K ulp |           ? gb/s, 5K ulp |          ? gb/s, 18K ulp |
| `nk_jsd_f32_serial`  |          ? gb/s, 0.5 ulp |          ? gb/s, 0.4 ulp |          ? gb/s, 4.3 ulp |
| __bf16__             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_kld_bf16_serial` |           ? gb/s, 1K ulp |           ? gb/s, 5K ulp |          ? gb/s, 18K ulp |
| `nk_jsd_bf16_serial` |          ? gb/s, 1.3 ulp |          ? gb/s, 3.2 ulp |          ? gb/s, 8.3 ulp |
| __f16__              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_kld_f16_serial`  |           ? gb/s, 1K ulp |           ? gb/s, 5K ulp |          ? gb/s, 18K ulp |
| `nk_jsd_f16_serial`  |          ? gb/s, 1.5 ulp |          ? gb/s, 2.4 ulp |          ? gb/s, 9.2 ulp |

### Apple M4 Pro

#### Native

| Kernel                |                      256 |                     1024 |                     4096 |
| :-------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f64__               | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_kld_f64_serial`   |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_jsd_f64_serial`   |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| __f32__               | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_kld_f32_serial`   |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_jsd_f32_serial`   |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_kld_f32_neon`     |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_jsd_f32_neon`     |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| __bf16__              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_kld_bf16_serial`  |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_jsd_bf16_serial`  |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| __f16__               | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_kld_f16_serial`   |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_jsd_f16_serial`   |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_kld_f16_neonhalf` |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_jsd_f16_neonhalf` |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |

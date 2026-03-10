# Trigonometric Functions in NumKong

NumKong implements element-wise trigonometric functions -- sine, cosine, and arc tangent -- with ~3 ulp error bounds for f32 and faithful rounding for f64.
Each function operates on dense vectors, reading input angles (radians) and writing output values of the same length.
The implementations derive from SLEEF (SIMD Library for Evaluating Elementary Functions), adapted for NumKong's ISA dispatch and type system.

Sine:

```math
\text{sin}: \mathbb{R} \to [-1, 1]
```

Cosine:

```math
\text{cos}: \mathbb{R} \to [-1, 1]
```

Arc tangent:

```math
\text{atan}: \mathbb{R} \to \left(-\frac{\pi}{2}, \frac{\pi}{2}\right)
```

Reformulating as Python pseudocode:

```python
import numpy as np

def sin(a: np.ndarray) -> np.ndarray:
    return np.sin(a)

def cos(a: np.ndarray) -> np.ndarray:
    return np.cos(a)

def atan(a: np.ndarray) -> np.ndarray:
    return np.arctan(a)
```

## Input & Output Types

| Input Type | Output Type | Description                                      |
| ---------- | ----------- | ------------------------------------------------ |
| `f64`      | `f64`       | 64-bit IEEE 754 double precision                 |
| `f32`      | `f32`       | 32-bit IEEE 754 single precision                 |
| `f16`      | `f16`       | 16-bit half precision, widened to f32 internally |

## Optimizations

### Cody-Waite Range Reduction

All trigonometric kernels reduce the input angle to $[-\pi/4, \pi/4]$ before polynomial evaluation using Cody-Waite argument reduction.
The constant $\pi$ is split into high and low parts ($\pi_{\text{hi}} + \pi_{\text{lo}}$) to maintain precision during the subtraction $x - n\pi$: `reduced = (x - n * pi_hi) - n * pi_lo`.
Single-part subtraction would lose ~3 bits of precision for large multiples of $\pi$; the two-part split preserves the full mantissa.
The quadrant index $n = \text{round}(x / \pi)$ selects which trigonometric identity to apply (sin-cos swap, sign flip) via a 2-bit branch.

### Minimax Polynomial Approximation

`nk_each_sin_f32_serial`, `nk_each_cos_f32_serial` evaluate degree-9 minimax polynomials via Horner's method after range reduction.
The polynomial coefficients are precomputed to minimize maximum error over $[-\pi/4, \pi/4]$ -- Chebyshev-optimal, not Taylor truncation.
Horner evaluation: `p = c9*x^2 + c7; p = p*x^2 + c5; p = p*x^2 + c3; p = p*x^2 + c1; p = p*x` -- 4 FMA operations plus 1 multiply for the final odd-power term.
`nk_each_sin_f64_serial` uses degree-19 polynomials for 52-bit mantissa coverage.

### Vectorized Polynomial Evaluation

`nk_each_sin_f32_haswell`, `nk_each_cos_f32_skylake` evaluate the same polynomial on 8 (AVX2) or 16 (AVX-512) elements simultaneously.
Range reduction, quadrant selection, and polynomial evaluation all operate on packed vectors -- the only scalar operation is the final sign correction via `VBLENDVPS` with the quadrant mask.
`nk_each_sin_f32_neon` processes 4 elements per iteration using `vfmaq_f32` for the Horner chain.
WASM v128relaxed (`nk_each_sin_f32_v128relaxed`) uses `f32x4.relaxed_madd` for the FMA steps, achieving ~2x throughput over strict `f32x4.mul` + `f32x4.add` sequences.

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

| Kernel                     |                      256 |                     1024 |                     4096 |
| :------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f64__                    | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sin_f64_serial`   |       1.18 gb/s, 0.2 ulp |       1.06 gb/s, 0.2 ulp |       1.10 gb/s, 0.2 ulp |
| `nk_each_cos_f64_serial`   |       1.24 gb/s, 0.3 ulp |       1.12 gb/s, 0.3 ulp |       1.14 gb/s, 0.3 ulp |
| `nk_each_atan_f64_serial`  |      0.332 gb/s, 0.3 ulp |      0.307 gb/s, 0.3 ulp |      0.309 gb/s, 0.3 ulp |
| `nk_each_sin_f64_haswell`  |       7.20 gb/s, 0.2 ulp |       6.83 gb/s, 0.2 ulp |       6.55 gb/s, 0.2 ulp |
| `nk_each_cos_f64_haswell`  |       7.44 gb/s, 0.3 ulp |       6.63 gb/s, 0.3 ulp |       6.54 gb/s, 0.3 ulp |
| `nk_each_atan_f64_haswell` |       6.07 gb/s, 0.3 ulp |       5.22 gb/s, 0.3 ulp |       5.26 gb/s, 0.3 ulp |
| `nk_each_sin_f64_skylake`  |       12.2 gb/s, 0.2 ulp |       11.5 gb/s, 0.2 ulp |       8.78 gb/s, 0.2 ulp |
| `nk_each_cos_f64_skylake`  |       11.7 gb/s, 0.3 ulp |       10.7 gb/s, 0.3 ulp |       10.2 gb/s, 0.3 ulp |
| `nk_each_atan_f64_skylake` |       9.16 gb/s, 0.3 ulp |       7.60 gb/s, 0.3 ulp |       7.28 gb/s, 0.3 ulp |
| __f32__                    | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sin_f32_serial`   |       10.9 gb/s, 4.9 ulp |       9.65 gb/s, 4.9 ulp |       9.93 gb/s, 4.9 ulp |
| `nk_each_cos_f32_serial`   |        10.1 gb/s, 16 ulp |        9.02 gb/s, 16 ulp |      9.60 gb/s, 16.4 ulp |
| `nk_each_atan_f32_serial`  |      0.687 gb/s, 0.4 ulp |      0.661 gb/s, 0.4 ulp |      0.680 gb/s, 0.4 ulp |
| `nk_each_sin_f32_haswell`  |       13.9 gb/s, 4.9 ulp |       12.2 gb/s, 4.9 ulp |       12.0 gb/s, 4.9 ulp |
| `nk_each_cos_f32_haswell`  |        13.1 gb/s, 16 ulp |        11.6 gb/s, 16 ulp |      11.1 gb/s, 16.4 ulp |
| `nk_each_atan_f32_haswell` |       8.74 gb/s, 0.4 ulp |       10.3 gb/s, 0.4 ulp |       10.3 gb/s, 0.4 ulp |
| `nk_each_sin_f32_skylake`  |       17.0 gb/s, 4.9 ulp |       12.9 gb/s, 4.9 ulp |       13.0 gb/s, 4.9 ulp |
| `nk_each_cos_f32_skylake`  |        16.9 gb/s, 16 ulp |        13.0 gb/s, 16 ulp |      12.0 gb/s, 16.3 ulp |
| `nk_each_atan_f32_skylake` |       14.1 gb/s, 0.4 ulp |       11.9 gb/s, 0.4 ulp |       10.2 gb/s, 0.4 ulp |
| __f16__                    | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sin_f16_serial`   |      0.113 gb/s, 1.3 ulp |      0.106 gb/s, 1.3 ulp |      0.110 gb/s, 1.3 ulp |
| `nk_each_cos_f16_serial`   |       0.109 gb/s, 12 ulp |       0.104 gb/s, 12 ulp |     0.105 gb/s, 12.3 ulp |
| `nk_each_atan_f16_serial`  |      0.021 gb/s, 6.4 ulp |      0.019 gb/s, 6.4 ulp |      0.020 gb/s, 6.4 ulp |
| `nk_each_sin_f16_skylake`  |        10.7 gb/s, 8K ulp |        9.92 gb/s, 8K ulp |        9.59 gb/s, 8K ulp |
| `nk_each_cos_f16_skylake`  |        10.0 gb/s, 8K ulp |        9.43 gb/s, 8K ulp |        9.04 gb/s, 8K ulp |
| `nk_each_atan_f16_skylake` |       8.20 gb/s, 16K ulp |       8.20 gb/s, 16K ulp |       7.73 gb/s, 16K ulp |

#### WASM

Measured with Wasmtime v42 (Cranelift backend).

| Kernel                         |                      256 |                     1024 |                     4096 |
| :----------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f64__                        | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sin_f64_serial`       |          ? gb/s, 0.2 ulp |          ? gb/s, 0.2 ulp |          ? gb/s, 0.2 ulp |
| `nk_each_cos_f64_serial`       |          ? gb/s, 0.3 ulp |          ? gb/s, 0.3 ulp |          ? gb/s, 0.3 ulp |
| `nk_each_atan_f64_serial`      |          ? gb/s, 0.3 ulp |          ? gb/s, 0.3 ulp |          ? gb/s, 0.3 ulp |
| `nk_each_sin_f64_v128relaxed`  |          ? gb/s, 0.2 ulp |          ? gb/s, 0.2 ulp |          ? gb/s, 0.2 ulp |
| `nk_each_cos_f64_v128relaxed`  |          ? gb/s, 0.3 ulp |          ? gb/s, 0.3 ulp |          ? gb/s, 0.3 ulp |
| `nk_each_atan_f64_v128relaxed` |          ? gb/s, 0.3 ulp |          ? gb/s, 0.3 ulp |          ? gb/s, 0.3 ulp |
| __f32__                        | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sin_f32_serial`       |          ? gb/s, 4.8 ulp |          ? gb/s, 4.8 ulp |          ? gb/s, 4.8 ulp |
| `nk_each_cos_f32_serial`       |           ? gb/s, 16 ulp |           ? gb/s, 16 ulp |           ? gb/s, 16 ulp |
| `nk_each_atan_f32_serial`      |          ? gb/s, 0.4 ulp |          ? gb/s, 0.4 ulp |          ? gb/s, 0.4 ulp |
| `nk_each_sin_f32_v128relaxed`  |           ? gb/s, 20 ulp |           ? gb/s, 19 ulp |           ? gb/s, 19 ulp |
| `nk_each_cos_f32_v128relaxed`  |           ? gb/s, 20 ulp |           ? gb/s, 20 ulp |           ? gb/s, 20 ulp |
| `nk_each_atan_f32_v128relaxed` |          ? gb/s, 0.4 ulp |          ? gb/s, 0.4 ulp |          ? gb/s, 0.4 ulp |

### Apple M4 Pro

#### Native

| Kernel                    |                      256 |                     1024 |                     4096 |
| :------------------------ | -----------------------: | -----------------------: | -----------------------: |
| __f64__                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sin_f64_serial`  |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_each_cos_f64_serial`  |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_each_atan_f64_serial` |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_each_sin_f64_neon`    |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_each_cos_f64_neon`    |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_each_atan_f64_neon`   |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| __f32__                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sin_f32_serial`  |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_each_cos_f32_serial`  |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_each_atan_f32_serial` |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_each_sin_f32_neon`    |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_each_cos_f32_neon`    |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_each_atan_f32_neon`   |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| __f16__                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sin_f16_serial`  |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_each_cos_f16_serial`  |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_each_atan_f16_serial` |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |

#### WASM

Measured with Wasmtime v42 (Cranelift backend).

| Kernel                         |                      256 |                     1024 |                     4096 |
| :----------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f64__                        | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sin_f64_serial`       |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_each_sin_f64_v128relaxed`  |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_each_cos_f64_serial`       |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_each_cos_f64_v128relaxed`  |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_each_atan_f64_serial`      |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_each_atan_f64_v128relaxed` |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| __f32__                        | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sin_f32_serial`       |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_each_sin_f32_v128relaxed`  |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_each_cos_f32_serial`       |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_each_cos_f32_v128relaxed`  |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_each_atan_f32_serial`      |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_each_atan_f32_v128relaxed` |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |

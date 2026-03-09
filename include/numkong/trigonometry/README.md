# Trigonometric Functions in NumKong

NumKong implements element-wise trigonometric functions -- sine, cosine, and arc tangent -- with ~3 ULP error bounds for f32 and faithful rounding for f64.
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

### Intel Sapphire Rapids

#### Native

<table>
<tr>
  <th>Kernel</th>
  <th>256</th>
  <th>1024</th>
  <th>4096</th>
</tr>
<tr><td colspan="4"><b>sin f64</b></td></tr>
<tr>
  <td><code>nk_each_sin_f64_haswell</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_each_sin_f64_skylake</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>sin f32</b></td></tr>
<tr>
  <td><code>nk_each_sin_f32_haswell</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_each_sin_f32_skylake</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>sin f16</b></td></tr>
<tr>
  <td><code>nk_each_sin_f16_skylake</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>cos f64</b></td></tr>
<tr>
  <td><code>nk_each_cos_f64_haswell</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_each_cos_f64_skylake</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>cos f32</b></td></tr>
<tr>
  <td><code>nk_each_cos_f32_haswell</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_each_cos_f32_skylake</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>cos f16</b></td></tr>
<tr>
  <td><code>nk_each_cos_f16_skylake</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>atan f64</b></td></tr>
<tr>
  <td><code>nk_each_atan_f64_haswell</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_each_atan_f64_skylake</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>atan f32</b></td></tr>
<tr>
  <td><code>nk_each_atan_f32_haswell</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_each_atan_f32_skylake</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>atan f16</b></td></tr>
<tr>
  <td><code>nk_each_atan_f16_skylake</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
</table>

#### V8 (Chromium)

<table>
<tr>
  <th>Kernel</th>
  <th>256</th>
  <th>1024</th>
  <th>4096</th>
</tr>
<tr><td colspan="4"><b>sin f64</b></td></tr>
<tr>
  <td><code>nk_each_sin_f64_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>sin f32</b></td></tr>
<tr>
  <td><code>nk_each_sin_f32_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>cos f64</b></td></tr>
<tr>
  <td><code>nk_each_cos_f64_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>cos f32</b></td></tr>
<tr>
  <td><code>nk_each_cos_f32_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>atan f64</b></td></tr>
<tr>
  <td><code>nk_each_atan_f64_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>atan f32</b></td></tr>
<tr>
  <td><code>nk_each_atan_f32_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
</table>

#### Wasmtime (Cranelift)

<table>
<tr>
  <th>Kernel</th>
  <th>256</th>
  <th>1024</th>
  <th>4096</th>
</tr>
<tr><td colspan="4"><b>sin f64</b></td></tr>
<tr>
  <td><code>nk_each_sin_f64_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>sin f32</b></td></tr>
<tr>
  <td><code>nk_each_sin_f32_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>cos f64</b></td></tr>
<tr>
  <td><code>nk_each_cos_f64_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>cos f32</b></td></tr>
<tr>
  <td><code>nk_each_cos_f32_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>atan f64</b></td></tr>
<tr>
  <td><code>nk_each_atan_f64_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>atan f32</b></td></tr>
<tr>
  <td><code>nk_each_atan_f32_v128relaxed</code></td>
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
  <th>256</th>
  <th>1024</th>
  <th>4096</th>
</tr>
<tr><td colspan="4"><b>sin f64</b></td></tr>
<tr>
  <td><code>nk_each_sin_f64_neon</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>sin f32</b></td></tr>
<tr>
  <td><code>nk_each_sin_f32_neon</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>cos f64</b></td></tr>
<tr>
  <td><code>nk_each_cos_f64_neon</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>cos f32</b></td></tr>
<tr>
  <td><code>nk_each_cos_f32_neon</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>atan f64</b></td></tr>
<tr>
  <td><code>nk_each_atan_f64_neon</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>atan f32</b></td></tr>
<tr>
  <td><code>nk_each_atan_f32_neon</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
</table>

#### V8 (Chromium)

<table>
<tr>
  <th>Kernel</th>
  <th>256</th>
  <th>1024</th>
  <th>4096</th>
</tr>
<tr><td colspan="4"><b>sin f64</b></td></tr>
<tr>
  <td><code>nk_each_sin_f64_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>sin f32</b></td></tr>
<tr>
  <td><code>nk_each_sin_f32_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>cos f64</b></td></tr>
<tr>
  <td><code>nk_each_cos_f64_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>cos f32</b></td></tr>
<tr>
  <td><code>nk_each_cos_f32_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>atan f64</b></td></tr>
<tr>
  <td><code>nk_each_atan_f64_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>atan f32</b></td></tr>
<tr>
  <td><code>nk_each_atan_f32_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
</table>

#### Wasmtime (Cranelift)

<table>
<tr>
  <th>Kernel</th>
  <th>256</th>
  <th>1024</th>
  <th>4096</th>
</tr>
<tr><td colspan="4"><b>sin f64</b></td></tr>
<tr>
  <td><code>nk_each_sin_f64_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>sin f32</b></td></tr>
<tr>
  <td><code>nk_each_sin_f32_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>cos f64</b></td></tr>
<tr>
  <td><code>nk_each_cos_f64_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>cos f32</b></td></tr>
<tr>
  <td><code>nk_each_cos_f32_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>atan f64</b></td></tr>
<tr>
  <td><code>nk_each_atan_f64_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>atan f32</b></td></tr>
<tr>
  <td><code>nk_each_atan_f32_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
</table>

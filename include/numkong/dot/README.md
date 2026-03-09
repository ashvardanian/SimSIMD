# Vector-Vector Dot Products in NumKong

NumKong implements dot products for every numeric type supported by the library, as the most important building block of higher-level functionality for vectors and higher rank tensors.

Dot product for real numbers and integers is defined as:

```math
\text{dot}(a, b) = \sum_{i=0}^{n-1} a_i \cdot b_i
```

For complex numbers, the dot product expands via the distributive property of complex multiplication:

```math
\text{dot}(a, b) = \sum_{i=0}^{n-1} (a_{i,re} \cdot b_{i,re} - a_{i,im} \cdot b_{i,im}) + j \sum_{i=0}^{n-1} (a_{i,re} \cdot b_{i,im} + a_{i,im} \cdot b_{i,re})
```

The conjugate dot product negates the imaginary part of $b$:

```math
\text{vdot}(a, b) = \sum_{i=0}^{n-1} a_i \cdot \bar{b_i} = \sum_{i=0}^{n-1} (a_{i,re} \cdot b_{i,re} + a_{i,im} \cdot b_{i,im}) + j \sum_{i=0}^{n-1} (a_{i,im} \cdot b_{i,re} - a_{i,re} \cdot b_{i,im})
```

Where $\bar{b_i}$ is the complex conjugate of $b_i$.
Reformulating as Python pseudocode for interleaved real/imaginary scalar arrays:

```python
def dot_real(a: List[number], b: List[number]) -> number:
    return sum(ai * bi for ai, bi in zip(a, b))

def dot_complex(a: List[number], b: List[number]) -> Tuple[number, number]:
    a_re, a_im = a[0::2], a[1::2]
    b_re, b_im = b[0::2], b[1::2]
    ab_re = sum(ar * br - ai * bi for ar, ai, br, bi in zip(a_re, a_im, b_re, b_im))
    ab_im = sum(ar * bi + ai * br for ar, ai, br, bi in zip(a_re, a_im, b_re, b_im))
    return ab_re, ab_im

def vdot_complex(a: List[number], b: List[number]) -> Tuple[number, number]:
    a_re, a_im = a[0::2], a[1::2]
    b_re, b_im = b[0::2], b[1::2]
    ab_re = sum(ar * br + ai * bi for ar, ai, br, bi in zip(a_re, a_im, b_re, b_im))
    ab_im = sum(ai * br - ar * bi for ar, ai, br, bi in zip(a_re, a_im, b_re, b_im))
    return ab_re, ab_im
```

## Input & Output Types

Real and integer dot products:

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
| `i8`       | `i32`       | 8-bit signed integers                          |
| `u8`       | `u32`       | 8-bit unsigned integers                        |
| `i4`       | `i32`       | 4-bit signed integers, packed nibble pairs     |
| `u4`       | `u32`       | 4-bit unsigned integers, packed nibble pairs   |
| `u1`       | `u32`       | 1-bit binary packed octets, popcount of AND    |

Complex dot products (both `dot` and `vdot`):

| Input Type | Output Type | Description                                |
| ---------- | ----------- | ------------------------------------------ |
| `f64c`     | `f64c`      | 64-bit complex pairs                       |
| `f32c`     | `f32c`      | 32-bit complex pairs                       |
| `f16c`     | `f32c`      | 16-bit complex pairs, widened output       |
| `bf16c`    | `f32c`      | 16-bit brain complex pairs, widened output |

## Optimizations

### Compensated Arithmetic for Large Floats

`nk_dot_f64_serial` uses Neumaier compensated summation — tracking a correction term adjusted by magnitude comparison at each step.
`nk_dot_f64_haswell`, `nk_dot_f64_skylake`, `nk_dot_f64_sve` implement the Dot2 algorithm by Ogita, Rump, and Oishi: TwoProd via FMA captures the rounding error of each product exactly, and a TwoSum chain propagates it through the accumulator.
On SVE, the final horizontal reduction uses `svtbl` to extract upper halves at each tree level, applying TwoSum at every stage.
The serial path uses Neumaier because it processes one element at a time and can cheaply branch on magnitudes.
Dot2 avoids those branches entirely — TwoProd and TwoSum are pure arithmetic with no comparisons, mapping naturally to wide SIMD where branching per lane is impossible.

### Lookup Tables for Mini-Floats

`nk_dot_e2m3_haswell`, `nk_dot_e3m2_haswell`, `nk_dot_e2m3_skylake`, `nk_dot_e3m2_skylake` encode 32 MX format values into scaled integers via dual 16-entry LUTs loaded into vector registers.
The low 4 magnitude bits index `VPSHUFB`, bit 4 selects between the lower and upper table via blending, and the results feed into `VPMADDUBSW` + `VPMADDWD` chains with a final $\div 256$ scaling.
`nk_dot_e2m3_sapphire`, `nk_dot_e3m2_sapphire` replace this with a single 64-entry signed LUT via `VPERMUTEX2VAR`, where the sign bit naturally selects between positive and negative tables.
They accumulate in native FP16 via `VFMADD_PH` and flush to FP32 every 128 elements to avoid overflow.

### Algebraic Domain Shifting

`nk_dot_i8_icelake`, `nk_dot_u8_icelake` work around `VPDPBUSD` requiring u8 × i8 operands.
For i8 × i8, one operand is XORed with `0x80` to shift to unsigned, and the correction $128 \cdot \sum b_i$ is computed via `VPSADBW`, which runs on port 5 and avoids contention with `DPBUSD` on ports 0-1.
`nk_dot_i4_icelake` extends this to packed nibbles using the identity $(a'-8)(b'-8) = a' b' - 8(a'+b') + 64$ — two `VPDPBUSD` calls handle low and high nibbles separately, with SAD-based correction.
`nk_dot_i8_v128relaxed`, `nk_dot_u8_v128relaxed` face an even tighter constraint: WASM's `i32x4_relaxed_dot_i8x16_i7x16_add` computes i8 × i7, so the sign bit of one operand must be masked off entirely.
For i8 × i8, the sign bit of $b$ is cleared to produce a 7-bit value, and a windowed correction $-128 \cdot \sum_{b_i < 0} a_i$ is accumulated in i16 and flushed every 127 iterations to prevent overflow.
For u8 × u8, $b$ is XORed with `0x80` to shift into signed range, same as Ice Lake, with the correction $128 \cdot \sum a_i$ computed via pairwise widening adds.

### Widening Fusion Through BF16 on x86

`nk_dot_e4m3_genoa`, `nk_dot_e5m2_genoa`, `nk_dot_e2m3_genoa`, `nk_dot_e3m2_genoa` convert FP8/MX values to BF16, then accumulate via `VDPBF16PS` — repurposing Genoa's BF16 dot-product hardware for types it was never designed for.
Each `VDPBF16PS` fuses two BF16 multiply-adds per 32-bit lane at 6-cycle throughput.
`nk_dot_bf16c_genoa` uses the same instruction for complex BF16, preparing operands with `VPSHUFB` for lane swapping and `VPXORD` with `0x80000000` for sign flips before feeding into `VDPBF16PS`.

### Widening Fusion Through F16 on Arm

`nk_dot_f16_neonfhm`, `nk_dot_f16c_neonfhm`, `nk_dot_e2m3_neonfhm`, `nk_dot_e3m2_neonfhm` use the ARMv8.4-FHM instructions `FMLAL`/`FMLSL`, which fuse FP16-to-FP32 conversion with multiply-accumulate in a single operation.
`vfmlalq_low_f16` and `vfmlalq_high_f16` process the lower and upper 4 elements of an 8-wide FP16 vector respectively.
For complex dot products, `FMLSL` provides the subtraction path $a_{re} b_{im} - a_{im} b_{re}$ without a separate negate step.

### Widening Chains on RISC-V

`nk_dot_i8_rvv`, `nk_dot_u8_rvv` use `vwmul` for i8 × i8 → i16 widening multiply followed by `vwadd` to widen-accumulate into i32 — a two-stage chain that naturally prevents overflow.
`nk_dot_bf16_rvvbf16` uses the Zvfbfwma extension's `vfwmaccbf16` for fused bf16 × bf16 → f32 widening multiply-accumulate.
`nk_dot_e4m3_rvvbf16`, `nk_dot_e5m2_rvvbf16` convert FP8 to BF16 via 256-entry LUTs, then feed the same `vfwmaccbf16` path.

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
<tr><td colspan="4"><b>f64c</b></td></tr>
<tr>
  <td><code>nk_dot_f64c_skylake</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_vdot_f64c_skylake</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>f32c</b></td></tr>
<tr>
  <td><code>nk_dot_f32c_haswell</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_vdot_f32c_haswell</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_dot_f32c_skylake</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_vdot_f32c_skylake</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>bf16c</b></td></tr>
<tr>
  <td><code>nk_dot_bf16c_haswell</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_vdot_bf16c_haswell</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>f16c</b></td></tr>
<tr>
  <td><code>nk_dot_f16c_haswell</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_vdot_f16c_haswell</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>f64</b></td></tr>
<tr>
  <td><code>nk_dot_f64_haswell</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_dot_f64_skylake</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>f32</b></td></tr>
<tr>
  <td><code>nk_dot_f32_haswell</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_dot_f32_skylake</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>bf16</b></td></tr>
<tr>
  <td><code>nk_dot_bf16_haswell</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_dot_bf16_skylake</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>f16</b></td></tr>
<tr>
  <td><code>nk_dot_f16_haswell</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_dot_f16_skylake</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>e5m2</b></td></tr>
<tr>
  <td><code>nk_dot_e5m2_haswell</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_dot_e5m2_skylake</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>e4m3</b></td></tr>
<tr>
  <td><code>nk_dot_e4m3_haswell</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_dot_e4m3_skylake</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>e3m2</b></td></tr>
<tr>
  <td><code>nk_dot_e3m2_haswell</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_dot_e3m2_skylake</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_dot_e3m2_icelake</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>e2m3</b></td></tr>
<tr>
  <td><code>nk_dot_e2m3_haswell</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_dot_e2m3_skylake</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_dot_e2m3_icelake</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>i8</b></td></tr>
<tr>
  <td><code>nk_dot_i8_haswell</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_dot_i8_skylake</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_dot_i8_icelake</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>u8</b></td></tr>
<tr>
  <td><code>nk_dot_u8_haswell</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_dot_u8_skylake</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_dot_u8_icelake</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>i4</b></td></tr>
<tr>
  <td><code>nk_dot_i4_haswell</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_dot_i4_icelake</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>u4</b></td></tr>
<tr>
  <td><code>nk_dot_u4_haswell</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_dot_u4_icelake</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>u1</b></td></tr>
<tr>
  <td><code>nk_dot_u1_haswell</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_dot_u1_icelake</code></td>
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
<tr><td colspan="4"><b>f64</b></td></tr>
<tr>
  <td><code>nk_dot_f64_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>f32</b></td></tr>
<tr>
  <td><code>nk_dot_f32_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>bf16</b></td></tr>
<tr>
  <td><code>nk_dot_bf16_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>f16</b></td></tr>
<tr>
  <td><code>nk_dot_f16_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>e3m2</b></td></tr>
<tr>
  <td><code>nk_dot_e3m2_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>e2m3</b></td></tr>
<tr>
  <td><code>nk_dot_e2m3_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>i8</b></td></tr>
<tr>
  <td><code>nk_dot_i8_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>u8</b></td></tr>
<tr>
  <td><code>nk_dot_u8_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>u1</b></td></tr>
<tr>
  <td><code>nk_dot_u1_v128relaxed</code></td>
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
<tr><td colspan="4"><b>f64</b></td></tr>
<tr>
  <td><code>nk_dot_f64_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>f32</b></td></tr>
<tr>
  <td><code>nk_dot_f32_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>bf16</b></td></tr>
<tr>
  <td><code>nk_dot_bf16_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>f16</b></td></tr>
<tr>
  <td><code>nk_dot_f16_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>e3m2</b></td></tr>
<tr>
  <td><code>nk_dot_e3m2_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>e2m3</b></td></tr>
<tr>
  <td><code>nk_dot_e2m3_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>i8</b></td></tr>
<tr>
  <td><code>nk_dot_i8_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>u8</b></td></tr>
<tr>
  <td><code>nk_dot_u8_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>u1</b></td></tr>
<tr>
  <td><code>nk_dot_u1_v128relaxed</code></td>
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
<tr><td colspan="4"><b>f64c</b></td></tr>
<tr>
  <td><code>nk_dot_f64c_neon</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_vdot_f64c_neon</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>f32c</b></td></tr>
<tr>
  <td><code>nk_dot_f32c_neon</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_vdot_f32c_neon</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>bf16c</b></td></tr>
<tr>
  <td><code>nk_dot_bf16c_neonbfdot</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_vdot_bf16c_neonbfdot</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>f16c</b></td></tr>
<tr>
  <td><code>nk_dot_f16c_neonhalf</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_vdot_f16c_neonhalf</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_dot_f16c_neonfhm</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_vdot_f16c_neonfhm</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>f64</b></td></tr>
<tr>
  <td><code>nk_dot_f64_neon</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>f32</b></td></tr>
<tr>
  <td><code>nk_dot_f32_neon</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>bf16</b></td></tr>
<tr>
  <td><code>nk_dot_bf16_neon</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_dot_bf16_neonbfdot</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>f16</b></td></tr>
<tr>
  <td><code>nk_dot_f16_neonhalf</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_dot_f16_neonfhm</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>e5m2</b></td></tr>
<tr>
  <td><code>nk_dot_e5m2_neon</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_dot_e5m2_neonfhm</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_dot_e5m2_neonbfdot</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>e4m3</b></td></tr>
<tr>
  <td><code>nk_dot_e4m3_neon</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_dot_e4m3_neonfhm</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_dot_e4m3_neonbfdot</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>e3m2</b></td></tr>
<tr>
  <td><code>nk_dot_e3m2_neon</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_dot_e3m2_neonfhm</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_dot_e3m2_neonsdot</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>e2m3</b></td></tr>
<tr>
  <td><code>nk_dot_e2m3_neon</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_dot_e2m3_neonfhm</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_dot_e2m3_neonsdot</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>i8</b></td></tr>
<tr>
  <td><code>nk_dot_i8_neonsdot</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>u8</b></td></tr>
<tr>
  <td><code>nk_dot_u8_neonsdot</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>i4</b></td></tr>
<tr>
  <td><code>nk_dot_i4_neonsdot</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>u4</b></td></tr>
<tr>
  <td><code>nk_dot_u4_neonsdot</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>u1</b></td></tr>
<tr>
  <td><code>nk_dot_u1_neon</code></td>
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
<tr><td colspan="4"><b>f64</b></td></tr>
<tr>
  <td><code>nk_dot_f64_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>f32</b></td></tr>
<tr>
  <td><code>nk_dot_f32_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>bf16</b></td></tr>
<tr>
  <td><code>nk_dot_bf16_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>f16</b></td></tr>
<tr>
  <td><code>nk_dot_f16_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>e3m2</b></td></tr>
<tr>
  <td><code>nk_dot_e3m2_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>e2m3</b></td></tr>
<tr>
  <td><code>nk_dot_e2m3_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>i8</b></td></tr>
<tr>
  <td><code>nk_dot_i8_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>u8</b></td></tr>
<tr>
  <td><code>nk_dot_u8_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>u1</b></td></tr>
<tr>
  <td><code>nk_dot_u1_v128relaxed</code></td>
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
<tr><td colspan="4"><b>f64</b></td></tr>
<tr>
  <td><code>nk_dot_f64_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>f32</b></td></tr>
<tr>
  <td><code>nk_dot_f32_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>bf16</b></td></tr>
<tr>
  <td><code>nk_dot_bf16_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>f16</b></td></tr>
<tr>
  <td><code>nk_dot_f16_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>e3m2</b></td></tr>
<tr>
  <td><code>nk_dot_e3m2_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>e2m3</b></td></tr>
<tr>
  <td><code>nk_dot_e2m3_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>i8</b></td></tr>
<tr>
  <td><code>nk_dot_i8_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>u8</b></td></tr>
<tr>
  <td><code>nk_dot_u8_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>u1</b></td></tr>
<tr>
  <td><code>nk_dot_u1_v128relaxed</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
</table>

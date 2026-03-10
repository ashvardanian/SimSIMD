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

### Deferred Sign-Flip in Complex Dot Products

The Haswell bf16c/f16c/f32c kernels compute $\sum (a_r b_r - a_i b_i)$ without per-pair subtraction.
Instead, two accumulators collect interleaved products $[a_r b_r, a_i b_i, \ldots]$ and $[a_r b_i, a_i b_r, \ldots]$, and a post-loop XOR flips the sign of every odd lane to produce the subtraction.
This gives one FMA per accumulator per iteration, but each lane grows $O(n)$ while the true result is $O(\sqrt{n})$.
The f32c kernel absorbs this via f64 accumulators; Genoa's `VDPBF16PS` and ARM's `FMLSL` pair terms naturally.
For bf16c/f16c on Haswell the accumulator is f32, so the $O(\log n)$ precision loss from lane separation is visible in max ULP at large $n$, though mean ULP remains low.

### Widening Fusion Through F16 on Arm

`nk_dot_f16_neonfhm`, `nk_dot_f16c_neonfhm`, `nk_dot_e2m3_neonfhm`, `nk_dot_e3m2_neonfhm` use the ARMv8.4-FHM instructions `FMLAL`/`FMLSL`, which fuse FP16-to-FP32 conversion with multiply-accumulate in a single operation.
`vfmlalq_low_f16` and `vfmlalq_high_f16` process the lower and upper 4 elements of an 8-wide FP16 vector respectively.
For complex dot products, `FMLSL` provides the subtraction path $a_{re} b_{im} - a_{im} b_{re}$ without a separate negate step.

### Widening Chains on RISC-V

`nk_dot_i8_rvv`, `nk_dot_u8_rvv` use `vwmul` for i8 × i8 → i16 widening multiply followed by `vwadd` to widen-accumulate into i32 — a two-stage chain that naturally prevents overflow.
`nk_dot_bf16_rvvbf16` uses the Zvfbfwma extension's `vfwmaccbf16` for fused bf16 × bf16 → f32 widening multiply-accumulate.
`nk_dot_e4m3_rvvbf16`, `nk_dot_e5m2_rvvbf16` convert FP8 to BF16 via 256-entry LUTs, then feed the same `vfwmaccbf16` path.

## Performance

The following performance tables are produced by manually re-running `nk_test` and `nk_bench` included internal tools to measure both accuracy and throughput at different input shapes.
The input size is controlled by the `NK_DENSE_DIMENSIONS` environment variable and set to 256, 1024, and 4096 elements.
The throughput is measured in gb/s as the number of bytes read per second amortized for a large batch of vector pairs.
Accuracy is reported as ULP (units in last place), the number of representable floating-point values between the result and the exact answer.
Each kernel runs for at least 20 seconds per configuration.
Benchmark threads are pinned to specific cores; on machines with heterogeneous core types (e.g., Apple P/E cores), only the fastest cores are used.
Workloads that significantly degrade CPU frequencies (Intel AMX, Apple SME) run in separate passes to avoid affecting throughput measurements of other kernels.

### Intel Sapphire Rapids

#### Native

| Kernel                   |                      256 |                     1024 |                     4096 |
| :----------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f64c__                 | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_f64c_with_blas`  |        29.9 gb/s, 25 ulp |        29.4 gb/s, 97 ulp |        15.1 gb/s, 32 ulp |
| `nk_vdot_f64c_with_blas` |        30.1 gb/s, 18 ulp |        29.5 gb/s, 17 ulp |        15.2 gb/s, 25 ulp |
| `nk_dot_f64c_serial`     |       5.29 gb/s, 5.2 ulp |       6.37 gb/s, 8.9 ulp |       6.50 gb/s, 3.2 ulp |
| `nk_vdot_f64c_serial`    |       5.23 gb/s, 4.3 ulp |       6.31 gb/s, 3.0 ulp |       6.55 gb/s, 3.0 ulp |
| `nk_dot_f64c_skylake`    |         23.7 gb/s, 0 ulp |         23.5 gb/s, 0 ulp |         12.5 gb/s, 0 ulp |
| `nk_vdot_f64c_skylake`   |         24.6 gb/s, 0 ulp |         23.5 gb/s, 0 ulp |         13.5 gb/s, 0 ulp |
| __f32c__                 | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_f32c_with_blas`  |       30.4 gb/s, 8.6 ulp |        29.4 gb/s, 13 ulp |        24.6 gb/s, 19 ulp |
| `nk_vdot_f32c_with_blas` |        30.3 gb/s, 11 ulp |        29.6 gb/s, 14 ulp |        25.1 gb/s, 21 ulp |
| `nk_dot_f32c_serial`     |        22.8 gb/s, 18 ulp |        21.8 gb/s, 29 ulp |        19.1 gb/s, 55 ulp |
| `nk_vdot_f32c_serial`    |        22.9 gb/s, 21 ulp |        21.8 gb/s, 5K ulp |        20.3 gb/s, 68 ulp |
| `nk_dot_f32c_haswell`    |         22.8 gb/s, 0 ulp |         22.1 gb/s, 0 ulp |         17.6 gb/s, 0 ulp |
| `nk_vdot_f32c_haswell`   |         22.9 gb/s, 0 ulp |         21.8 gb/s, 0 ulp |         18.2 gb/s, 0 ulp |
| `nk_dot_f32c_skylake`    |         27.5 gb/s, 0 ulp |         27.6 gb/s, 0 ulp |         22.1 gb/s, 0 ulp |
| `nk_vdot_f32c_skylake`   |         27.2 gb/s, 0 ulp |         27.6 gb/s, 0 ulp |         17.8 gb/s, 0 ulp |
| __bf16c__                | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_bf16c_serial`    |      0.609 gb/s, 0.1 ulp |      0.604 gb/s, 3.2 ulp |       0.618 gb/s, 13 ulp |
| `nk_vdot_bf16c_serial`   |      0.627 gb/s, 0.2 ulp |      0.618 gb/s, 3.7 ulp |       0.608 gb/s, 14 ulp |
| `nk_dot_bf16c_haswell`   |         22.6 gb/s, 0 ulp |       18.1 gb/s, 1.3 ulp |        19.7 gb/s, 9K ulp |
| `nk_vdot_bf16c_haswell`  |       22.2 gb/s, 0.1 ulp |       18.4 gb/s, 1.3 ulp |       19.7 gb/s, 6.2 ulp |
| `nk_dot_bf16c_genoa`     |         37.5 gb/s, 0 ulp |       29.2 gb/s, 1.1 ulp |       30.1 gb/s, 3.2 ulp |
| `nk_vdot_bf16c_genoa`    |       37.2 gb/s, 0.2 ulp |       29.5 gb/s, 1.2 ulp |       30.3 gb/s, 7.3 ulp |
| __f16c__                 | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_f16c_serial`     |        2.01 gb/s, 17 ulp |        2.00 gb/s, 34 ulp |        2.01 gb/s, 60 ulp |
| `nk_vdot_f16c_serial`    |        1.65 gb/s, 19 ulp |        1.62 gb/s, 30 ulp |        1.63 gb/s, 90 ulp |
| `nk_dot_f16c_haswell`    |        24.3 gb/s, 13 ulp |        19.4 gb/s, 24 ulp |        20.1 gb/s, 52 ulp |
| `nk_vdot_f16c_haswell`   |        24.3 gb/s, 14 ulp |        19.4 gb/s, 25 ulp |        20.2 gb/s, 71 ulp |
| __f64__                  | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_f64_with_blas`   |       31.5 gb/s, 6.9 ulp |       30.3 gb/s, 9.3 ulp |        27.5 gb/s, 20 ulp |
| `nk_dot_f64_serial`      |       5.47 gb/s, 2.3 ulp |       6.29 gb/s, 2.2 ulp |       6.95 gb/s, 2.3 ulp |
| `nk_dot_f64_haswell`     |         24.4 gb/s, 0 ulp |         24.6 gb/s, 0 ulp |         19.0 gb/s, 0 ulp |
| `nk_dot_f64_skylake`     |         26.3 gb/s, 0 ulp |         27.9 gb/s, 0 ulp |         24.6 gb/s, 0 ulp |
| __f32__                  | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_f32_serial`      |        16.5 gb/s, 1K ulp |        12.9 gb/s, 36 ulp |        12.7 gb/s, 63 ulp |
| `nk_dot_f32_with_blas`   |        51.3 gb/s, 14 ulp |        30.7 gb/s, 14 ulp |        33.0 gb/s, 15 ulp |
| `nk_dot_f32_haswell`     |         29.7 gb/s, 0 ulp |         22.7 gb/s, 0 ulp |         24.6 gb/s, 0 ulp |
| `nk_dot_f32_skylake`     |         38.3 gb/s, 0 ulp |         29.2 gb/s, 0 ulp |         28.9 gb/s, 0 ulp |
| __bf16__                 | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_bf16_serial`     |        0.622 gb/s, 0 ulp |      0.657 gb/s, 1.1 ulp |      0.639 gb/s, 6.0 ulp |
| `nk_dot_bf16_haswell`    |         30.7 gb/s, 0 ulp |       19.8 gb/s, 0.4 ulp |       19.4 gb/s, 5.0 ulp |
| `nk_dot_bf16_skylake`    |         49.7 gb/s, 0 ulp |       28.6 gb/s, 0.4 ulp |       29.2 gb/s, 3.3 ulp |
| `nk_dot_bf16_genoa`      |         78.7 gb/s, 0 ulp |       31.0 gb/s, 0.4 ulp |       31.7 gb/s, 2.9 ulp |
| __f16__                  | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_f16_serial`      |        1.30 gb/s, 14 ulp |        1.32 gb/s, 28 ulp |        1.30 gb/s, 54 ulp |
| `nk_dot_f16_haswell`     |       30.0 gb/s, 9.4 ulp |        22.6 gb/s, 16 ulp |        20.5 gb/s, 31 ulp |
| `nk_dot_f16_skylake`     |       52.4 gb/s, 8.0 ulp |        29.2 gb/s, 11 ulp |        29.8 gb/s, 23 ulp |
| __e5m2__                 | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_e5m2_serial`     |         1.01 gb/s, 0 ulp |        0.910 gb/s, 0 ulp |        0.958 gb/s, 0 ulp |
| `nk_dot_e5m2_haswell`    |         4.86 gb/s, 0 ulp |         4.84 gb/s, 0 ulp |         4.82 gb/s, 0 ulp |
| `nk_dot_e5m2_skylake`    |         6.17 gb/s, 0 ulp |         6.21 gb/s, 0 ulp |         6.12 gb/s, 0 ulp |
| `nk_dot_e5m2_genoa`      |         12.4 gb/s, 0 ulp |         12.9 gb/s, 0 ulp |         12.5 gb/s, 0 ulp |
| __e4m3__                 | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_e4m3_serial`     |        0.386 gb/s, 0 ulp |        0.363 gb/s, 0 ulp |        0.384 gb/s, 0 ulp |
| `nk_dot_e4m3_haswell`    |         3.13 gb/s, 0 ulp |         3.17 gb/s, 0 ulp |         3.18 gb/s, 0 ulp |
| `nk_dot_e4m3_skylake`    |         4.89 gb/s, 0 ulp |         5.01 gb/s, 0 ulp |         4.80 gb/s, 0 ulp |
| `nk_dot_e4m3_genoa`      |         12.1 gb/s, 0 ulp |         12.9 gb/s, 0 ulp |         12.6 gb/s, 0 ulp |
| __e3m2__                 | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_e3m2_serial`     |        0.991 gb/s, 0 ulp |        0.921 gb/s, 0 ulp |        0.944 gb/s, 0 ulp |
| `nk_dot_e3m2_haswell`    |         12.0 gb/s, 0 ulp |         12.1 gb/s, 0 ulp |         11.7 gb/s, 0 ulp |
| `nk_dot_e3m2_icelake`    |         22.8 gb/s, 0 ulp |         24.4 gb/s, 0 ulp |         22.6 gb/s, 0 ulp |
| __e2m3__                 | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_e2m3_serial`     |        0.982 gb/s, 0 ulp |        0.957 gb/s, 0 ulp |        0.952 gb/s, 0 ulp |
| `nk_dot_e2m3_haswell`    |         19.4 gb/s, 0 ulp |         19.8 gb/s, 0 ulp |         18.7 gb/s, 0 ulp |
| `nk_dot_e2m3_icelake`    |         54.6 gb/s, 0 ulp |         43.7 gb/s, 0 ulp |         31.4 gb/s, 0 ulp |
| __i8__                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_i8_serial`       |                17.6 gb/s |                17.0 gb/s |                16.0 gb/s |
| `nk_dot_i8_haswell`      |                43.7 gb/s |                35.3 gb/s |                28.7 gb/s |
| `nk_dot_i8_skylake`      |                54.8 gb/s |                37.8 gb/s |                30.3 gb/s |
| `nk_dot_i8_icelake`      |                65.4 gb/s |                46.2 gb/s |                31.9 gb/s |
| __u8__                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_u8_serial`       |                17.8 gb/s |                16.8 gb/s |                16.1 gb/s |
| `nk_dot_u8_haswell`      |                47.7 gb/s |                37.0 gb/s |                29.6 gb/s |
| `nk_dot_u8_skylake`      |                54.6 gb/s |                39.2 gb/s |                30.2 gb/s |
| `nk_dot_u8_icelake`      |                68.3 gb/s |                48.6 gb/s |                32.0 gb/s |
| __i4__                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_i4_serial`       |                9.53 gb/s |                11.6 gb/s |                11.6 gb/s |
| `nk_dot_i4_haswell`      |                8.31 gb/s |                8.64 gb/s |                8.31 gb/s |
| `nk_dot_i4_icelake`      |                25.9 gb/s |                36.2 gb/s |                27.5 gb/s |
| __u4__                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_u4_serial`       |                10.4 gb/s |                12.3 gb/s |                12.1 gb/s |
| `nk_dot_u4_haswell`      |                15.0 gb/s |                15.9 gb/s |                14.8 gb/s |
| `nk_dot_u4_icelake`      |                48.3 gb/s |                60.8 gb/s |                30.8 gb/s |
| __u1__                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_u1_serial`       |                3.85 gb/s |                4.69 gb/s |                4.99 gb/s |
| `nk_dot_u1_haswell`      |                14.0 gb/s |                43.6 gb/s |                71.3 gb/s |
| `nk_dot_u1_icelake`      |                22.2 gb/s |                70.2 gb/s |                 110 gb/s |

#### WASM

Measured with Wasmtime v42 (Cranelift backend).

| Kernel                     |                      256 |                     1024 |                     4096 |
| :------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f64c__                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_f64c_serial`       |          ? gb/s, 3.4 ulp |          ? gb/s, 3.7 ulp |          ? gb/s, 7.5 ulp |
| `nk_vdot_f64c_serial`      |          ? gb/s, 3.6 ulp |          ? gb/s, 4.4 ulp |          ? gb/s, 5.4 ulp |
| `nk_dot_f64c_v128relaxed`  |           ? gb/s, 13 ulp |           ? gb/s, 31 ulp |           ? gb/s, 38 ulp |
| `nk_vdot_f64c_v128relaxed` |           ? gb/s, 20 ulp |           ? gb/s, 21 ulp |           ? gb/s, 47 ulp |
| __f32c__                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_f32c_serial`       |           ? gb/s, 14 ulp |           ? gb/s, 30 ulp |           ? gb/s, 60 ulp |
| `nk_vdot_f32c_serial`      |           ? gb/s, 17 ulp |           ? gb/s, 29 ulp |           ? gb/s, 63 ulp |
| `nk_dot_f32c_v128relaxed`  |            ? gb/s, 0 ulp |            ? gb/s, 0 ulp |            ? gb/s, 0 ulp |
| `nk_vdot_f32c_v128relaxed` |            ? gb/s, 0 ulp |            ? gb/s, 0 ulp |            ? gb/s, 0 ulp |
| __bf16c__                  | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_bf16c_serial`      |          ? gb/s, 0.2 ulp |          ? gb/s, 1.8 ulp |           ? gb/s, 13 ulp |
| `nk_vdot_bf16c_serial`     |          ? gb/s, 0.1 ulp |          ? gb/s, 3.0 ulp |          ? gb/s, 7.2 ulp |
| __f16c__                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_f16c_serial`       |           ? gb/s, 11 ulp |           ? gb/s, 32 ulp |           ? gb/s, 36 ulp |
| `nk_vdot_f16c_serial`      |           ? gb/s, 15 ulp |           ? gb/s, 16 ulp |           ? gb/s, 29 ulp |
| __f64__                    | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_f64_serial`        |          ? gb/s, 2.3 ulp |          ? gb/s, 4.5 ulp |          ? gb/s, 3.5 ulp |
| `nk_dot_f64_v128relaxed`   |          ? gb/s, 2.6 ulp |          ? gb/s, 5.3 ulp |          ? gb/s, 3.8 ulp |
| __f32__                    | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_f32_serial`        |           ? gb/s, 16 ulp |           ? gb/s, 69 ulp |          ? gb/s, 104 ulp |
| `nk_dot_f32_v128relaxed`   |            ? gb/s, 0 ulp |            ? gb/s, 0 ulp |            ? gb/s, 0 ulp |
| __bf16__                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_bf16_serial`       |            ? gb/s, 0 ulp |          ? gb/s, 1.0 ulp |          ? gb/s, 6.3 ulp |
| `nk_dot_bf16_v128relaxed`  |            ? gb/s, 0 ulp |          ? gb/s, 0.4 ulp |          ? gb/s, 3.7 ulp |
| __f16__                    | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_f16_serial`        |           ? gb/s, 14 ulp |           ? gb/s, 29 ulp |           ? gb/s, 57 ulp |
| `nk_dot_f16_v128relaxed`   |          ? gb/s, 8.9 ulp |           ? gb/s, 23 ulp |           ? gb/s, 35 ulp |
| __e5m2__                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_e5m2_serial`       |            ? gb/s, 0 ulp |            ? gb/s, 0 ulp |            ? gb/s, 0 ulp |
| `nk_dot_e5m2_v128relaxed`  |            ? gb/s, 0 ulp |            ? gb/s, 0 ulp |            ? gb/s, 0 ulp |
| __e4m3__                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_e4m3_serial`       |            ? gb/s, 0 ulp |            ? gb/s, 0 ulp |            ? gb/s, 0 ulp |
| `nk_dot_e4m3_v128relaxed`  |            ? gb/s, 0 ulp |            ? gb/s, 0 ulp |            ? gb/s, 0 ulp |
| __e3m2__                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_e3m2_serial`       |            ? gb/s, 0 ulp |            ? gb/s, 0 ulp |            ? gb/s, 0 ulp |
| `nk_dot_e3m2_v128relaxed`  |            ? gb/s, 0 ulp |            ? gb/s, 0 ulp |            ? gb/s, 0 ulp |
| __e2m3__                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_e2m3_serial`       |            ? gb/s, 0 ulp |            ? gb/s, 0 ulp |            ? gb/s, 0 ulp |
| `nk_dot_e2m3_v128relaxed`  |            ? gb/s, 0 ulp |            ? gb/s, 0 ulp |            ? gb/s, 0 ulp |
| __i8__                     | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_i8_serial`         |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_dot_i8_v128relaxed`    |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| __u8__                     | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_u8_serial`         |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_dot_u8_v128relaxed`    |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| __i4__                     | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_i4_serial`         |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_dot_i4_v128relaxed`    |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| __u4__                     | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_u4_serial`         |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_dot_u4_v128relaxed`    |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| __u1__                     | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_u1_serial`         |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_dot_u1_v128relaxed`    |                   0 gb/s |                   0 gb/s |                   0 gb/s |

### Apple M4 Pro

#### Native

| Kernel                    |                      256 |                     1024 |                     4096 |
| :------------------------ | -----------------------: | -----------------------: | -----------------------: |
| __f64c__                  | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_f64c_serial`      |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_vdot_f64c_serial`     |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_dot_f64c_neon`        |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_vdot_f64c_neon`       |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| __f32c__                  | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_f32c_serial`      |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_vdot_f32c_serial`     |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_dot_f32c_neon`        |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_vdot_f32c_neon`       |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| __bf16c__                 | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_bf16c_serial`     |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_vdot_bf16c_serial`    |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_dot_bf16c_neonbfdot`  |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_vdot_bf16c_neonbfdot` |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| __f16c__                  | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_f16c_serial`      |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_vdot_f16c_serial`     |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_dot_f16c_neonhalf`    |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_vdot_f16c_neonhalf`   |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_dot_f16c_neonfhm`     |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_vdot_f16c_neonfhm`    |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| __f64__                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_f64_serial`       |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_dot_f64_neon`         |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| __f32__                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_f32_serial`       |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_dot_f32_neon`         |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| __bf16__                  | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_bf16_serial`      |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_dot_bf16_neonbfdot`   |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| __f16__                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_f16_serial`       |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_dot_f16_neonhalf`     |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_dot_f16_neonfhm`      |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| __e5m2__                  | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_e5m2_serial`      |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_dot_e5m2_neon`        |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| __e4m3__                  | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_e4m3_serial`      |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_dot_e4m3_neon`        |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| __e3m2__                  | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_e3m2_serial`      |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_dot_e3m2_neonsdot`    |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_dot_e3m2_neonfhm`     |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| __e2m3__                  | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_e2m3_serial`      |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_dot_e2m3_neonsdot`    |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_dot_e2m3_neonfhm`     |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| __i8__                    | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_i8_serial`        |                   ? gb/s |                   ? gb/s |                   ? gb/s |
| `nk_dot_i8_neonsdot`      |                   ? gb/s |                   ? gb/s |                   ? gb/s |
| __u8__                    | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_u8_serial`        |                   ? gb/s |                   ? gb/s |                   ? gb/s |
| `nk_dot_u8_neonsdot`      |                   ? gb/s |                   ? gb/s |                   ? gb/s |
| __i4__                    | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_i4_serial`        |                   ? gb/s |                   ? gb/s |                   ? gb/s |
| `nk_dot_i4_neonsdot`      |                   ? gb/s |                   ? gb/s |                   ? gb/s |
| __u4__                    | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_u4_serial`        |                   ? gb/s |                   ? gb/s |                   ? gb/s |
| `nk_dot_u4_neonsdot`      |                   ? gb/s |                   ? gb/s |                   ? gb/s |
| __u1__                    | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_u1_serial`        |                   ? gb/s |                   ? gb/s |                   ? gb/s |
| `nk_dot_u1_neon`          |                   ? gb/s |                   ? gb/s |                   ? gb/s |

#### WASM

Measured with Wasmtime v42 (Cranelift backend).

| Kernel                     |                      256 |                     1024 |                     4096 |
| :------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f64c__                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_f64c_serial`       |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_vdot_f64c_serial`      |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_dot_f64c_v128relaxed`  |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_vdot_f64c_v128relaxed` |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| __f32c__                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_f32c_serial`       |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_vdot_f32c_serial`      |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_dot_f32c_v128relaxed`  |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_vdot_f32c_v128relaxed` |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| __bf16c__                  | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_bf16c_serial`      |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_vdot_bf16c_serial`     |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| __f16c__                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_f16c_serial`       |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_vdot_f16c_serial`      |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| __f64__                    | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_f64_serial`        |          ? gb/s, 2.3 ulp |          ? gb/s, 4.5 ulp |          ? gb/s, 3.5 ulp |
| `nk_dot_f64_v128relaxed`   |          ? gb/s, 2.6 ulp |          ? gb/s, 5.3 ulp |          ? gb/s, 3.8 ulp |
| __f32__                    | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_f32_serial`        |           ? gb/s, 16 ulp |           ? gb/s, 69 ulp |          ? gb/s, 104 ulp |
| `nk_dot_f32_v128relaxed`   |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| __bf16__                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_bf16_serial`       |            ? gb/s, ? ulp |          ? gb/s, 1.0 ulp |          ? gb/s, 6.3 ulp |
| `nk_dot_bf16_v128relaxed`  |            ? gb/s, ? ulp |          ? gb/s, 0.4 ulp |          ? gb/s, 3.7 ulp |
| __f16__                    | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_f16_serial`        |           ? gb/s, 14 ulp |           ? gb/s, 29 ulp |           ? gb/s, 57 ulp |
| `nk_dot_f16_v128relaxed`   |          ? gb/s, 8.9 ulp |           ? gb/s, 23 ulp |           ? gb/s, 35 ulp |
| __e5m2__                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_e5m2_serial`       |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_dot_e5m2_v128relaxed`  |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| __e4m3__                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_e4m3_serial`       |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_dot_e4m3_v128relaxed`  |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| __e3m2__                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_e3m2_serial`       |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_dot_e3m2_v128relaxed`  |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| __e2m3__                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_e2m3_serial`       |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| `nk_dot_e2m3_v128relaxed`  |            ? gb/s, ? ulp |            ? gb/s, ? ulp |            ? gb/s, ? ulp |
| __i8__                     | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_i8_serial`         |                   ? gb/s |                   ? gb/s |                   ? gb/s |
| `nk_dot_i8_v128relaxed`    |                   ? gb/s |                   ? gb/s |                   ? gb/s |
| __u8__                     | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_u8_serial`         |                   ? gb/s |                   ? gb/s |                   ? gb/s |
| `nk_dot_u8_v128relaxed`    |                   ? gb/s |                   ? gb/s |                   ? gb/s |
| __i4__                     | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_i4_serial`         |                   ? gb/s |                   ? gb/s |                   ? gb/s |
| `nk_dot_i4_v128relaxed`    |                   ? gb/s |                   ? gb/s |                   ? gb/s |
| __u4__                     | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_u4_serial`         |                   ? gb/s |                   ? gb/s |                   ? gb/s |
| `nk_dot_u4_v128relaxed`    |                   ? gb/s |                   ? gb/s |                   ? gb/s |
| __u1__                     | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dot_u1_serial`         |                   ? gb/s |                   ? gb/s |                   ? gb/s |
| `nk_dot_u1_v128relaxed`    |                   ? gb/s |                   ? gb/s |                   ? gb/s |

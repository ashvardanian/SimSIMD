# Scalar Math Primitives in NumKong

NumKong provides single-element math operations -- square root, reciprocal square root, fused multiply-add, and saturating integer arithmetic -- with per-ISA implementations.
These primitives serve as building blocks for vectorized kernels: distance finalizers call `nk_f32_rsqrt` for angular normalization, packing routines call `nk_f32_sqrt` for norm computation.
Ordering functions (`nk_f16_order`, `nk_bf16_order`, `nk_e4m3_order`) convert floating-point bit patterns to integers that sort in the same order as the original floats.

Reciprocal square root:

```math
\text{rsqrt}(x) = \frac{1}{\sqrt{x}}
```

Fused multiply-add:

```math
\text{fma}(a, b, c) = a \cdot b + c
```

Saturating addition:

```math
\text{sat\_add}(a, b) = \text{clamp}(a + b, \text{T\_MIN}, \text{T\_MAX})
```

Reformulating as Python pseudocode:

```python
import numpy as np

def rsqrt(x: float) -> float:
    return 1.0 / np.sqrt(x)

def fma(a: float, b: float, c: float) -> float:
    return a * b + c

def saturating_add(a: int, b: int, bits: int, signed: bool) -> int:
    lo, hi = (-(1 << (bits-1)), (1 << (bits-1)) - 1) if signed else (0, (1 << bits) - 1)
    return max(lo, min(a + b, hi))
```

## Input & Output Types

| Input Type | Output Type | Description                                     |
| ---------- | ----------- | ----------------------------------------------- |
| `f64`      | `f64`       | sqrt, rsqrt, fma for 64-bit doubles             |
| `f32`      | `f32`       | sqrt, rsqrt, fma for 32-bit floats              |
| `f16`      | `f16`       | sqrt, rsqrt, fma for 16-bit halfs               |
| `i8`       | `i8`        | Saturating add and multiply                     |
| `u8`       | `u8`        | Saturating add and multiply                     |
| `i16`      | `i16`       | Saturating add and multiply                     |
| `u16`      | `u16`       | Saturating add and multiply                     |
| `i32`      | `i32`       | Saturating add and multiply                     |
| `u32`      | `u32`       | Saturating add and multiply                     |
| `i64`      | `i64`       | Saturating add and multiply                     |
| `u64`      | `u64`       | Saturating add and multiply                     |
| `i4x2`     | `i4x2`      | Saturating add for packed signed nibble pairs   |
| `u4x2`     | `u4x2`      | Saturating add for packed unsigned nibble pairs |
| `f16`      | `u16`       | Ordering: bit pattern to sortable integer       |
| `bf16`     | `u16`       | Ordering: bit pattern to sortable integer       |
| `e4m3`     | `u8`        | Ordering: bit pattern to sortable integer       |
| `e5m2`     | `u8`        | Ordering: bit pattern to sortable integer       |
| `e2m3`     | `u8`        | Ordering: bit pattern to sortable integer       |
| `e3m2`     | `u8`        | Ordering: bit pattern to sortable integer       |

## Optimizations

### Quake 3 Fast Inverse Square Root

`nk_f32_rsqrt_serial` uses the classic bit-manipulation trick: reinterpret f32 bits as i32, compute `0x5F375A86 - (bits >> 1)`, reinterpret back to f32, then refine with 3 Newton-Raphson iterations reaching ~34.9 correct bits.
Each Newton-Raphson iteration: `y = y * (1.5f - 0.5f * x * y * y)` -- 2 multiplies and 1 subtract, ~4cy per iteration.
`nk_f32_rsqrt_haswell` replaces this with hardware `VRSQRT14PS` ($2^{-14}$ relative error, ~4cy latency) plus one Newton-Raphson refinement (~22-24 correct bits).
`nk_f64_rsqrt_serial` uses the f64 magic constant `0x5FE6EB50C7B537A9` with 4 iterations for 52-bit mantissa coverage.

### Dekker Error-Free Multiplication for FMA

`nk_f32_fma_serial` emulates fused multiply-add on platforms without hardware FMA using Dekker's algorithm: splits each operand into high and low halves via `a_hi = (a * 134217729.0f) - ((a * 134217729.0f) - a)`, then computes the exact product error term.
The magic constant $134217729 = 2^{27} + 1$ splits a 24-bit mantissa into two 12-bit halves that multiply without rounding.
`nk_f32_fma_haswell` uses hardware `VFMADD231SS` -- single instruction, single cycle, exact to the last bit.

### Float-to-Integer Ordering

`nk_f16_order_serial`, `nk_bf16_order_serial`, `nk_e4m3_order_serial` convert floating-point bit patterns to unsigned integers that preserve the total order.
Positive floats are already ordered by their bit patterns; negative floats need bit inversion: `if (bits & sign_bit) bits = ~bits; else bits ^= sign_bit`.
This enables integer comparison instructions (`VPCMPUD`) for floating-point sorting without branching -- used by `nk_reduce_minmax_*` for FP8 and sub-32-bit types that lack native SIMD comparison.

## Performance

Scalar primitives operate on single elements and are not independently benchmarked.
Their performance is captured within the vector kernels that call them.

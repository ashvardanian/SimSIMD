# Element-Wise Arithmetic in NumKong

NumKong implements element-wise vector arithmetic: addition, scaling, blending, and fused multiply-add across all supported numeric types.
Each operation reads one to three input vectors and writes one output vector of the same length, with scalar coefficients $\alpha$ and $\beta$ controlling linear combinations.
Mixed-precision workflows use narrower input types (f16, bf16, FP8) with f32 intermediate computation and narrowed output.

Sum (addition):

```math
\text{result}_i = a_i + b_i
```

Scale:

```math
\text{result}_i = \alpha \cdot a_i + \beta
```

Blend:

```math
\text{result}_i = \alpha \cdot a_i + \beta \cdot b_i
```

Fused multiply-add:

```math
\text{result}_i = \alpha \cdot a_i \cdot b_i + \beta \cdot c_i
```

Reformulating as Python pseudocode:

```python
import numpy as np

def fma(a: np.ndarray, b: np.ndarray, c: np.ndarray,
        alpha: float = 1.0, beta: float = 1.0) -> np.ndarray:
    return alpha * a * b + beta * c

def scale(a: np.ndarray, alpha: float = 1.0, beta: float = 0.0) -> np.ndarray:
    return alpha * a + beta

def add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a + b

def blend(a: np.ndarray, b: np.ndarray,
          alpha: float = 1.0, beta: float = 1.0) -> np.ndarray:
    return alpha * a + beta * b
```

## Input & Output Types

Real and integer element-wise operations:

| Input Type | Output Type | Description                            |
| ---------- | ----------- | -------------------------------------- |
| `f64`      | `f64`       | 64-bit IEEE 754 double precision       |
| `f32`      | `f32`       | 32-bit IEEE 754 single precision       |
| `f16`      | `f16`       | 16-bit IEEE 754 half precision         |
| `bf16`     | `bf16`      | 16-bit brain float                     |
| `e4m3`     | `e4m3`      | 8-bit FP8: 4 exponent, 3 mantissa bits |
| `e5m2`     | `e5m2`      | 8-bit FP8: 5 exponent, 2 mantissa bits |
| `i8`       | `i8`        | 8-bit signed integers, saturating      |
| `u8`       | `u8`        | 8-bit unsigned integers, saturating    |
| `i16`      | `i16`       | 16-bit signed integers                 |
| `u16`      | `u16`       | 16-bit unsigned integers               |
| `i32`      | `i32`       | 32-bit signed integers                 |
| `u32`      | `u32`       | 32-bit unsigned integers               |
| `i64`      | `i64`       | 64-bit signed integers                 |
| `u64`      | `u64`       | 64-bit unsigned integers               |

Complex element-wise operations:

| Input Type | Output Type | Description          |
| ---------- | ----------- | -------------------- |
| `f64c`     | `f64c`      | 64-bit complex pairs |
| `f32c`     | `f32c`      | 32-bit complex pairs |

## Optimizations

### Widening-Narrowing Pipeline for Sub-32-bit Types

`nk_each_fma_f16_haswell`, `nk_each_blend_bf16_neonbfdot`, `nk_each_scale_e4m3_haswell` widen inputs to f32 before arithmetic, then narrow the result back to the original type.
The widen-compute-narrow pipeline costs 2 extra conversion instructions per element but guarantees f32-precision intermediate results -- critical for FMA where naive f16 multiplication would lose 5+ bits of mantissa.
Haswell processes 8 f16 elements per cycle: `VCVTPH2PS` (widen) -> `VFMADD231PS` (FMA) -> `VCVTPS2PH` (narrow), fully pipelined across 3 execution ports.

### Saturating Integer Arithmetic

`nk_each_sum_i8_haswell`, `nk_each_sum_u8_neonhalf` use saturating addition -- clamping to type bounds instead of wrapping on overflow.
Haswell uses `VPADDSB` / `VPADDUSB` for signed/unsigned 8-bit saturation in a single instruction (32 elements per cycle at YMM width).
Serial fallback implements saturation via branch-free min/max: `result = min(max(a + b, TYPE_MIN), TYPE_MAX)` with overflow detection through sign-bit comparison.

### Complex Number Layout

`nk_each_fma_f32c_serial`, `nk_each_blend_f64c_serial` operate on interleaved real/imaginary pairs: `[re0, im0, re1, im1, ...]`.
Addition and scaling treat complex vectors as 2N-length real vectors -- no special handling needed.
FMA requires cross-lane operations for the imaginary part: `re(a*b) = re(a)*re(b) - im(a)*im(b)`, implemented via `VFMADDSUB231PS` which alternates add/subtract across even/odd lanes.

## Performance

Controlled by `NK_DENSE_DIMENSIONS`.
Columns show 256, 1024, 4096 elements.

### Intel Sapphire Rapids

#### Native

| Kernel                       |           256 |          1024 |          4096 |
| :--------------------------- | ------------: | ------------: | ------------: |
| __f64__                      |               |               |               |
| `nk_each_sum_f64_serial`     | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_sum_f64_haswell`    | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_sum_f64_skylake`    | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_scale_f64_serial`   | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_scale_f64_haswell`  | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_scale_f64_skylake`  | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_blend_f64_serial`   | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_blend_f64_haswell`  | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_blend_f64_skylake`  | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_fma_f64_serial`     | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_fma_f64_haswell`    | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_fma_f64_skylake`    | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| __f32__                      |               |               |               |
| `nk_each_sum_f32_serial`     | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_sum_f32_haswell`    | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_sum_f32_skylake`    | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_scale_f32_serial`   | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_scale_f32_haswell`  | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_scale_f32_skylake`  | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_blend_f32_serial`   | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_blend_f32_haswell`  | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_blend_f32_skylake`  | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_fma_f32_serial`     | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_fma_f32_haswell`    | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_fma_f32_skylake`    | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| __bf16__                     |               |               |               |
| `nk_each_sum_bf16_serial`    | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_sum_bf16_haswell`   | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_sum_bf16_skylake`   | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_scale_bf16_serial`  | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_scale_bf16_haswell` | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_scale_bf16_skylake` | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_blend_bf16_serial`  | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_blend_bf16_haswell` | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_blend_bf16_skylake` | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_fma_bf16_serial`    | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_fma_bf16_haswell`   | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_fma_bf16_skylake`   | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| __f16__                      |               |               |               |
| `nk_each_sum_f16_serial`     | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_sum_f16_haswell`    | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_sum_f16_sapphire`   | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_scale_f16_serial`   | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_scale_f16_haswell`  | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_blend_f16_serial`   | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_blend_f16_haswell`  | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_fma_f16_serial`     | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_fma_f16_haswell`    | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| __e4m3__                     |               |               |               |
| `nk_each_sum_e4m3_serial`    | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_scale_e4m3_serial`  | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_blend_e4m3_serial`  | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_fma_e4m3_serial`    | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| __e5m2__                     |               |               |               |
| `nk_each_sum_e5m2_serial`    | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_scale_e5m2_serial`  | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_blend_e5m2_serial`  | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_fma_e5m2_serial`    | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| __e2m3__                     |               |               |               |
| `nk_each_sum_e2m3_serial`    | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_scale_e2m3_serial`  | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_blend_e2m3_serial`  | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_fma_e2m3_serial`    | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| __e3m2__                     |               |               |               |
| `nk_each_sum_e3m2_serial`    | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_scale_e3m2_serial`  | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_blend_e3m2_serial`  | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_fma_e3m2_serial`    | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| __i8__                       |               |               |               |
| `nk_each_sum_i8_serial`      |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_sum_i8_haswell`     |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_sum_i8_icelake`     |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_scale_i8_serial`    |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_scale_i8_haswell`   |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_scale_i8_skylake`   |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_scale_i8_sapphire`  |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_blend_i8_serial`    |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_blend_i8_haswell`   |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_blend_i8_sapphire`  |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_fma_i8_serial`      |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_fma_i8_haswell`     |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_fma_i8_skylake`     |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_fma_i8_sapphire`    |        0 GB/s |        0 GB/s |        0 GB/s |
| __u8__                       |               |               |               |
| `nk_each_sum_u8_serial`      |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_sum_u8_haswell`     |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_sum_u8_icelake`     |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_scale_u8_serial`    |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_scale_u8_haswell`   |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_scale_u8_skylake`   |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_scale_u8_sapphire`  |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_blend_u8_serial`    |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_blend_u8_haswell`   |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_blend_u8_sapphire`  |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_fma_u8_serial`      |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_fma_u8_haswell`     |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_fma_u8_skylake`     |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_fma_u8_sapphire`    |        0 GB/s |        0 GB/s |        0 GB/s |
| __i16__                      |               |               |               |
| `nk_each_sum_i16_serial`     |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_sum_i16_haswell`    |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_sum_i16_icelake`    |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_scale_i16_serial`   |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_scale_i16_haswell`  |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_scale_i16_skylake`  |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_blend_i16_serial`   |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_fma_i16_serial`     |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_fma_i16_haswell`    |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_fma_i16_skylake`    |        0 GB/s |        0 GB/s |        0 GB/s |
| __u16__                      |               |               |               |
| `nk_each_sum_u16_serial`     |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_sum_u16_haswell`    |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_sum_u16_icelake`    |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_scale_u16_serial`   |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_scale_u16_haswell`  |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_scale_u16_skylake`  |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_blend_u16_serial`   |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_fma_u16_serial`     |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_fma_u16_haswell`    |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_fma_u16_skylake`    |        0 GB/s |        0 GB/s |        0 GB/s |
| __i32__                      |               |               |               |
| `nk_each_sum_i32_serial`     |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_sum_i32_haswell`    |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_sum_i32_icelake`    |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_scale_i32_serial`   |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_scale_i32_haswell`  |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_scale_i32_skylake`  |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_blend_i32_serial`   |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_fma_i32_serial`     |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_fma_i32_haswell`    |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_fma_i32_skylake`    |        0 GB/s |        0 GB/s |        0 GB/s |
| __u32__                      |               |               |               |
| `nk_each_sum_u32_serial`     |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_sum_u32_haswell`    |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_sum_u32_icelake`    |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_scale_u32_serial`   |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_scale_u32_haswell`  |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_scale_u32_skylake`  |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_blend_u32_serial`   |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_fma_u32_serial`     |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_fma_u32_haswell`    |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_fma_u32_skylake`    |        0 GB/s |        0 GB/s |        0 GB/s |
| __i64__                      |               |               |               |
| `nk_each_sum_i64_serial`     |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_sum_i64_icelake`    |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_scale_i64_serial`   |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_scale_i64_skylake`  |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_blend_i64_serial`   |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_fma_i64_serial`     |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_fma_i64_skylake`    |        0 GB/s |        0 GB/s |        0 GB/s |
| __u64__                      |               |               |               |
| `nk_each_sum_u64_serial`     |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_sum_u64_icelake`    |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_scale_u64_serial`   |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_scale_u64_skylake`  |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_blend_u64_serial`   |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_fma_u64_serial`     |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_fma_u64_skylake`    |        0 GB/s |        0 GB/s |        0 GB/s |
| __f64c__                     |               |               |               |
| `nk_each_scale_f64c_serial`  | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_scale_f64c_haswell` | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_scale_f64c_skylake` | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_blend_f64c_serial`  | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_blend_f64c_haswell` | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_blend_f64c_skylake` | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_fma_f64c_serial`    | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_fma_f64c_haswell`   | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_fma_f64c_skylake`   | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| __f32c__                     |               |               |               |
| `nk_each_scale_f32c_serial`  | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_scale_f32c_haswell` | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_scale_f32c_skylake` | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_blend_f32c_serial`  | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_blend_f32c_haswell` | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_blend_f32c_skylake` | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_fma_f32c_serial`    | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_fma_f32c_haswell`   | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_fma_f32c_skylake`   | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |

### Apple M4 Pro

#### Native

| Kernel                         |           256 |          1024 |          4096 |
| :----------------------------- | ------------: | ------------: | ------------: |
| __f64__                        |               |               |               |
| `nk_each_sum_f64_serial`       | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_sum_f64_neon`         | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_scale_f64_serial`     | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_scale_f64_neon`       | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_blend_f64_serial`     | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_blend_f64_neon`       | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_fma_f64_serial`       | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_fma_f64_neon`         | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| __f32__                        |               |               |               |
| `nk_each_sum_f32_serial`       | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_sum_f32_neon`         | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_scale_f32_serial`     | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_scale_f32_neon`       | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_blend_f32_serial`     | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_blend_f32_neon`       | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_fma_f32_serial`       | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_fma_f32_neon`         | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| __bf16__                       |               |               |               |
| `nk_each_sum_bf16_serial`      | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_sum_bf16_neonbfdot`   | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_scale_bf16_serial`    | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_scale_bf16_neonbfdot` | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_blend_bf16_serial`    | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_blend_bf16_neonbfdot` | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_fma_bf16_serial`      | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_fma_bf16_neonbfdot`   | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| __f16__                        |               |               |               |
| `nk_each_sum_f16_serial`       | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_sum_f16_neonhalf`     | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_scale_f16_serial`     | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_scale_f16_neonhalf`   | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_blend_f16_serial`     | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_blend_f16_neonhalf`   | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_fma_f16_serial`       | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_fma_f16_neonhalf`     | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| __e4m3__                       |               |               |               |
| `nk_each_sum_e4m3_serial`      | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_sum_e4m3_neon`        | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_scale_e4m3_serial`    | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_scale_e4m3_neon`      | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_blend_e4m3_serial`    | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_blend_e4m3_neon`      | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_fma_e4m3_serial`      | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_fma_e4m3_neon`        | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| __e5m2__                       |               |               |               |
| `nk_each_sum_e5m2_serial`      | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_sum_e5m2_neon`        | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_scale_e5m2_serial`    | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_scale_e5m2_neon`      | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_blend_e5m2_serial`    | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_blend_e5m2_neon`      | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_fma_e5m2_serial`      | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_fma_e5m2_neon`        | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| __e2m3__                       |               |               |               |
| `nk_each_sum_e2m3_serial`      | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_scale_e2m3_serial`    | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_blend_e2m3_serial`    | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_fma_e2m3_serial`      | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| __e3m2__                       |               |               |               |
| `nk_each_sum_e3m2_serial`      | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_scale_e3m2_serial`    | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_blend_e3m2_serial`    | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_fma_e3m2_serial`      | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| __i8__                         |               |               |               |
| `nk_each_sum_i8_serial`        |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_sum_i8_neonhalf`      |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_scale_i8_serial`      |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_scale_i8_neonhalf`    |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_blend_i8_serial`      |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_blend_i8_neonhalf`    |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_fma_i8_serial`        |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_fma_i8_neonhalf`      |        0 GB/s |        0 GB/s |        0 GB/s |
| __u8__                         |               |               |               |
| `nk_each_sum_u8_serial`        |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_sum_u8_neonhalf`      |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_scale_u8_serial`      |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_scale_u8_neonhalf`    |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_blend_u8_serial`      |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_blend_u8_neonhalf`    |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_fma_u8_serial`        |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_fma_u8_neonhalf`      |        0 GB/s |        0 GB/s |        0 GB/s |
| __i16__                        |               |               |               |
| `nk_each_sum_i16_serial`       |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_sum_i16_neon`         |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_scale_i16_serial`     |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_scale_i16_neon`       |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_blend_i16_serial`     |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_fma_i16_serial`       |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_fma_i16_neon`         |        0 GB/s |        0 GB/s |        0 GB/s |
| __u16__                        |               |               |               |
| `nk_each_sum_u16_serial`       |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_sum_u16_neon`         |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_scale_u16_serial`     |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_scale_u16_neon`       |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_blend_u16_serial`     |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_fma_u16_serial`       |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_fma_u16_neon`         |        0 GB/s |        0 GB/s |        0 GB/s |
| __i32__                        |               |               |               |
| `nk_each_sum_i32_serial`       |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_sum_i32_neon`         |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_scale_i32_serial`     |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_scale_i32_neon`       |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_blend_i32_serial`     |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_fma_i32_serial`       |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_fma_i32_neon`         |        0 GB/s |        0 GB/s |        0 GB/s |
| __u32__                        |               |               |               |
| `nk_each_sum_u32_serial`       |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_sum_u32_neon`         |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_scale_u32_serial`     |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_scale_u32_neon`       |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_blend_u32_serial`     |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_fma_u32_serial`       |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_fma_u32_neon`         |        0 GB/s |        0 GB/s |        0 GB/s |
| __i64__                        |               |               |               |
| `nk_each_sum_i64_serial`       |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_sum_i64_neon`         |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_scale_i64_serial`     |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_scale_i64_neon`       |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_blend_i64_serial`     |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_fma_i64_serial`       |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_fma_i64_neon`         |        0 GB/s |        0 GB/s |        0 GB/s |
| __u64__                        |               |               |               |
| `nk_each_sum_u64_serial`       |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_sum_u64_neon`         |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_scale_u64_serial`     |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_scale_u64_neon`       |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_blend_u64_serial`     |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_fma_u64_serial`       |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_each_fma_u64_neon`         |        0 GB/s |        0 GB/s |        0 GB/s |
| __f64c__                       |               |               |               |
| `nk_each_scale_f64c_serial`    | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_scale_f64c_neon`      | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_blend_f64c_serial`    | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_blend_f64c_neon`      | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_fma_f64c_serial`      | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_fma_f64c_neon`        | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| __f32c__                       |               |               |               |
| `nk_each_scale_f32c_serial`    | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_scale_f32c_neon`      | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_blend_f32c_serial`    | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_blend_f32c_neon`      | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_fma_f32c_serial`      | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_each_fma_f32c_neon`        | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |

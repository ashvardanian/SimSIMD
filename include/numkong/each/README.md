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

The following performance tables are produced by manually re-running `nk_test` and `nk_bench` included internal tools to measure both accuracy and throughput at different input shapes.
The input size is controlled by the `NK_DENSE_DIMENSIONS` environment variable and set to 256, 1024, and 4096 elements.
The throughput is measured in GB/s as the number of input bytes read per second.
Accuracy is reported as ULP (units in last place), the number of representable floating-point values between the result and the exact answer.
Each kernel runs for at least 20 seconds per configuration.
Benchmark threads are pinned to specific cores; on machines with heterogeneous core types (e.g., Apple P/E cores), only the fastest cores are used.
Workloads that significantly degrade CPU frequencies (Intel AMX, Apple SME) run in separate passes to avoid affecting throughput measurements of other kernels.

### Intel Sapphire Rapids

#### Native

| Kernel                        |                      256 |                     1024 |                     4096 |
| :---------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f64__                       | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_f64_serial`      |         16.7 gb/s, 0 ulp |         17.2 gb/s, 0 ulp |         10.9 gb/s, 0 ulp |
| `nk_each_sum_f64_haswell`     |         14.7 gb/s, 0 ulp |         10.2 gb/s, 0 ulp |         7.99 gb/s, 0 ulp |
| `nk_each_sum_f64_skylake`     |         16.4 gb/s, 0 ulp |         16.7 gb/s, 0 ulp |         8.59 gb/s, 0 ulp |
| `nk_sum_f64_with_blas`        |                12.1 gb/s |                8.63 gb/s |                6.83 gb/s |
| `nk_each_scale_f64_serial`    |         11.4 gb/s, 0 ulp |         12.5 gb/s, 0 ulp |         7.99 gb/s, 0 ulp |
| `nk_each_scale_f64_haswell`   |         9.61 gb/s, 0 ulp |         9.22 gb/s, 0 ulp |         5.08 gb/s, 0 ulp |
| `nk_each_scale_f64_skylake`   |         11.2 gb/s, 0 ulp |         11.9 gb/s, 0 ulp |         6.30 gb/s, 0 ulp |
| `nk_each_blend_f64_serial`    |      16.4 gb/s, 461K ulp |      16.4 gb/s, 811K ulp |       11.8 gb/s, 53K ulp |
| `nk_each_blend_f64_haswell`   |       13.4 gb/s, 1.5 ulp |       11.2 gb/s, 1.5 ulp |       7.85 gb/s, 1.1 ulp |
| `nk_each_blend_f64_skylake`   |       16.5 gb/s, 1.7 ulp |       15.9 gb/s, 1.5 ulp |       8.55 gb/s, 1.1 ulp |
| `nk_each_blend_f64_with_blas` |                11.3 gb/s |                8.80 gb/s |                6.71 gb/s |
| `nk_each_fma_f64_serial`      |      20.1 gb/s, 360K ulp |      20.8 gb/s, 123K ulp |     11.7 gb/s, 2897K ulp |
| `nk_each_fma_f64_haswell`     |       16.8 gb/s, 1.5 ulp |       11.1 gb/s, 1.5 ulp |       9.42 gb/s, 2.8 ulp |
| `nk_each_fma_f64_skylake`     |       19.8 gb/s, 1.4 ulp |       20.4 gb/s, 1.5 ulp |       11.6 gb/s, 2.7 ulp |
| __f32__                       | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_f32_serial`      |         16.9 gb/s, 0 ulp |         18.8 gb/s, 0 ulp |         15.7 gb/s, 0 ulp |
| `nk_each_sum_f32_haswell`     |         14.6 gb/s, 0 ulp |         14.3 gb/s, 0 ulp |         8.49 gb/s, 0 ulp |
| `nk_each_sum_f32_skylake`     |         16.7 gb/s, 0 ulp |         17.9 gb/s, 0 ulp |         16.5 gb/s, 0 ulp |
| `nk_sum_f32_with_blas`        |                11.6 gb/s |                12.8 gb/s |                7.57 gb/s |
| `nk_each_scale_f32_serial`    |         10.8 gb/s, 0 ulp |         13.0 gb/s, 0 ulp |         12.1 gb/s, 0 ulp |
| `nk_each_scale_f32_haswell`   |         9.26 gb/s, 0 ulp |         10.2 gb/s, 0 ulp |         6.22 gb/s, 0 ulp |
| `nk_each_scale_f32_skylake`   |         12.3 gb/s, 0 ulp |         11.9 gb/s, 0 ulp |         12.9 gb/s, 0 ulp |
| `nk_each_blend_f32_serial`    |       16.2 gb/s, 351 ulp |       18.7 gb/s, 2.0 ulp |       17.4 gb/s, 1.4 ulp |
| `nk_each_blend_f32_haswell`   |       14.5 gb/s, 2.3 ulp |       14.2 gb/s, 2.1 ulp |       8.16 gb/s, 1.3 ulp |
| `nk_each_blend_f32_skylake`   |       15.7 gb/s, 1.9 ulp |       17.2 gb/s, 1.8 ulp |       15.9 gb/s, 1.3 ulp |
| `nk_each_blend_f32_with_blas` |                10.3 gb/s |                9.60 gb/s |                6.69 gb/s |
| `nk_each_fma_f32_serial`      |       20.1 gb/s, 1.4 ulp |       18.7 gb/s, 2.1 ulp |       19.0 gb/s, 1.6 ulp |
| `nk_each_fma_f32_haswell`     |       18.4 gb/s, 1.4 ulp |       15.3 gb/s, 1.8 ulp |       9.52 gb/s, 1.5 ulp |
| `nk_each_fma_f32_skylake`     |       20.8 gb/s, 1.4 ulp |       19.3 gb/s, 1.7 ulp |       16.5 gb/s, 1.5 ulp |
| __bf16__                      | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_bf16_serial`     |        0.175 gb/s, 0 ulp |        0.178 gb/s, 0 ulp |        0.173 gb/s, 0 ulp |
| `nk_each_sum_bf16_haswell`    |         7.97 gb/s, 0 ulp |         9.33 gb/s, 0 ulp |         10.3 gb/s, 0 ulp |
| `nk_each_sum_bf16_skylake`    |         11.4 gb/s, 0 ulp |         11.2 gb/s, 0 ulp |         14.1 gb/s, 0 ulp |
| `nk_each_scale_bf16_serial`   |        0.128 gb/s, 0 ulp |        0.119 gb/s, 0 ulp |        0.132 gb/s, 0 ulp |
| `nk_each_scale_bf16_haswell`  |         6.08 gb/s, 0 ulp |         6.55 gb/s, 0 ulp |         6.92 gb/s, 0 ulp |
| `nk_each_scale_bf16_skylake`  |         7.43 gb/s, 0 ulp |         8.04 gb/s, 0 ulp |         8.45 gb/s, 0 ulp |
| `nk_each_blend_bf16_serial`   |        0.211 gb/s, 0 ulp |        0.204 gb/s, 0 ulp |        0.224 gb/s, 0 ulp |
| `nk_each_blend_bf16_haswell`  |       8.58 gb/s, 2.2 ulp |       9.44 gb/s, 1.5 ulp |       10.2 gb/s, 1.5 ulp |
| `nk_each_blend_bf16_skylake`  |       10.3 gb/s, 2.3 ulp |       11.9 gb/s, 1.3 ulp |       13.4 gb/s, 1.5 ulp |
| `nk_each_fma_bf16_serial`     |        0.264 gb/s, 0 ulp |        0.260 gb/s, 0 ulp |        0.256 gb/s, 0 ulp |
| `nk_each_fma_bf16_haswell`    |       10.9 gb/s, 1.5 ulp |       10.3 gb/s, 0.9 ulp |       11.4 gb/s, 1.0 ulp |
| `nk_each_fma_bf16_skylake`    |       14.1 gb/s, 1.2 ulp |       13.0 gb/s, 0.7 ulp |       15.8 gb/s, 1.1 ulp |
| __f16__                       | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_f16_serial`      |         33.7 gb/s, 0 ulp |         16.1 gb/s, 0 ulp |         18.8 gb/s, 0 ulp |
| `nk_each_sum_f16_haswell`     |         14.4 gb/s, 0 ulp |         11.8 gb/s, 0 ulp |         9.84 gb/s, 0 ulp |
| `nk_each_sum_f16_sapphire`    |         39.2 gb/s, 0 ulp |         17.0 gb/s, 0 ulp |         18.9 gb/s, 0 ulp |
| `nk_each_scale_f16_serial`    |        0.423 gb/s, 0 ulp |        0.282 gb/s, 0 ulp |        0.409 gb/s, 0 ulp |
| `nk_each_scale_f16_haswell`   |         8.92 gb/s, 0 ulp |         8.59 gb/s, 0 ulp |         8.15 gb/s, 0 ulp |
| `nk_each_scale_f16_skylake`   |         17.0 gb/s, 0 ulp |         10.7 gb/s, 0 ulp |         12.1 gb/s, 0 ulp |
| `nk_each_blend_f16_serial`    |      0.769 gb/s, 1.3 ulp |      0.669 gb/s, 1.6 ulp |      0.792 gb/s, 1.5 ulp |
| `nk_each_blend_f16_haswell`   |       13.5 gb/s, 1.2 ulp |       11.0 gb/s, 1.4 ulp |       11.7 gb/s, 1.5 ulp |
| `nk_each_blend_f16_skylake`   |       16.9 gb/s, 1.3 ulp |       13.9 gb/s, 1.4 ulp |       14.2 gb/s, 1.1 ulp |
| `nk_each_fma_f16_serial`      |      0.965 gb/s, 1.0 ulp |      0.787 gb/s, 1.1 ulp |      0.952 gb/s, 1.2 ulp |
| `nk_each_fma_f16_haswell`     |       15.2 gb/s, 1.4 ulp |       13.6 gb/s, 1.0 ulp |       15.7 gb/s, 1.1 ulp |
| `nk_each_fma_f16_skylake`     |       16.3 gb/s, 1.3 ulp |       16.2 gb/s, 1.3 ulp |       15.3 gb/s, 1.1 ulp |
| __e4m3__                      | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_e4m3_serial`     |       0.0966 gb/s, 0 ulp |       0.0943 gb/s, 0 ulp |       0.0946 gb/s, 0 ulp |
| `nk_each_sum_e4m3_haswell`    |        0.895 gb/s, 0 ulp |        0.772 gb/s, 0 ulp |        0.824 gb/s, 0 ulp |
| `nk_each_sum_e4m3_skylake`    |         1.48 gb/s, 0 ulp |         1.34 gb/s, 0 ulp |         1.40 gb/s, 0 ulp |
| `nk_each_sum_e4m3_sapphire`   |         2.12 gb/s, 0 ulp |         1.89 gb/s, 0 ulp |         2.13 gb/s, 0 ulp |
| `nk_each_scale_e4m3_serial`   |       0.0550 gb/s, 0 ulp |       0.0543 gb/s, 0 ulp |       0.0570 gb/s, 0 ulp |
| `nk_each_scale_e4m3_haswell`  |        0.495 gb/s, 0 ulp |        0.532 gb/s, 0 ulp |        0.540 gb/s, 0 ulp |
| `nk_each_scale_e4m3_skylake`  |         1.05 gb/s, 0 ulp |         1.02 gb/s, 0 ulp |         1.10 gb/s, 0 ulp |
| `nk_each_blend_e4m3_serial`   |       0.0889 gb/s, 0 ulp |       0.0927 gb/s, 0 ulp |       0.0876 gb/s, 0 ulp |
| `nk_each_blend_e4m3_haswell`  |      0.807 gb/s, 0.6 ulp |        0.756 gb/s, 0 ulp |        0.789 gb/s, 0 ulp |
| `nk_each_blend_e4m3_skylake`  |         1.50 gb/s, 0 ulp |         1.44 gb/s, 0 ulp |         1.48 gb/s, 0 ulp |
| `nk_each_fma_e4m3_serial`     |        0.120 gb/s, 0 ulp |      0.118 gb/s, 0.9 ulp |        0.115 gb/s, 0 ulp |
| `nk_each_fma_e4m3_haswell`    |        0.989 gb/s, 0 ulp |        0.909 gb/s, 0 ulp |      0.967 gb/s, 0.5 ulp |
| `nk_each_fma_e4m3_skylake`    |         1.89 gb/s, 0 ulp |         1.74 gb/s, 0 ulp |         1.80 gb/s, 0 ulp |
| __e5m2__                      | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_e5m2_serial`     |        0.115 gb/s, 0 ulp |        0.114 gb/s, 0 ulp |        0.110 gb/s, 0 ulp |
| `nk_each_sum_e5m2_haswell`    |         1.04 gb/s, 0 ulp |        0.987 gb/s, 0 ulp |        0.972 gb/s, 0 ulp |
| `nk_each_sum_e5m2_skylake`    |         1.71 gb/s, 0 ulp |         1.76 gb/s, 0 ulp |         1.79 gb/s, 0 ulp |
| `nk_each_scale_e5m2_serial`   |       0.0593 gb/s, 0 ulp |       0.0630 gb/s, 0 ulp |       0.0630 gb/s, 0 ulp |
| `nk_each_scale_e5m2_haswell`  |        0.601 gb/s, 0 ulp |        0.611 gb/s, 0 ulp |        0.588 gb/s, 0 ulp |
| `nk_each_scale_e5m2_skylake`  |         1.11 gb/s, 0 ulp |         1.11 gb/s, 0 ulp |         1.12 gb/s, 0 ulp |
| `nk_each_blend_e5m2_serial`   |        0.108 gb/s, 0 ulp |        0.113 gb/s, 0 ulp |       0.114 gb/s, 50 ulp |
| `nk_each_blend_e5m2_haswell`  |        0.999 gb/s, 0 ulp |        0.895 gb/s, 0 ulp |        0.951 gb/s, 0 ulp |
| `nk_each_blend_e5m2_skylake`  |         1.77 gb/s, 0 ulp |         1.65 gb/s, 0 ulp |         1.72 gb/s, 0 ulp |
| `nk_each_fma_e5m2_serial`     |      0.155 gb/s, 5.1 ulp |        0.146 gb/s, 0 ulp |        0.149 gb/s, 0 ulp |
| `nk_each_fma_e5m2_haswell`    |         1.24 gb/s, 0 ulp |         1.19 gb/s, 0 ulp |         1.25 gb/s, 0 ulp |
| `nk_each_fma_e5m2_skylake`    |         2.36 gb/s, 0 ulp |         2.01 gb/s, 0 ulp |         2.11 gb/s, 0 ulp |
| __e2m3__                      | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_e2m3_serial`     |        0.109 gb/s, 0 ulp |        0.105 gb/s, 0 ulp |        0.110 gb/s, 0 ulp |
| `nk_each_scale_e2m3_serial`   |       0.0500 gb/s, 0 ulp |       0.0474 gb/s, 0 ulp |       0.0495 gb/s, 0 ulp |
| `nk_each_blend_e2m3_serial`   |       0.0864 gb/s, 0 ulp |       0.0888 gb/s, 0 ulp |       0.0908 gb/s, 0 ulp |
| `nk_each_fma_e2m3_serial`     |        0.133 gb/s, 0 ulp |        0.128 gb/s, 0 ulp |        0.128 gb/s, 0 ulp |
| __e3m2__                      | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_e3m2_serial`     |        0.118 gb/s, 0 ulp |        0.115 gb/s, 0 ulp |        0.106 gb/s, 0 ulp |
| `nk_each_scale_e3m2_serial`   |       0.0574 gb/s, 0 ulp |       0.0555 gb/s, 0 ulp |       0.0548 gb/s, 0 ulp |
| `nk_each_blend_e3m2_serial`   |        0.110 gb/s, 0 ulp |        0.105 gb/s, 0 ulp |       0.0956 gb/s, 0 ulp |
| `nk_each_fma_e3m2_serial`     |        0.151 gb/s, 0 ulp |        0.149 gb/s, 0 ulp |        0.138 gb/s, 0 ulp |
| __i8__                        | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_i8_serial`       |                21.7 gb/s |                15.9 gb/s |                12.5 gb/s |
| `nk_each_sum_i8_haswell`      |                37.7 gb/s |                15.7 gb/s |                16.5 gb/s |
| `nk_each_sum_i8_icelake`      |                47.2 gb/s |                17.7 gb/s |                17.7 gb/s |
| `nk_each_scale_i8_serial`     |                2.34 gb/s |                1.50 gb/s |                1.70 gb/s |
| `nk_each_scale_i8_haswell`    |                3.91 gb/s |                3.93 gb/s |                3.64 gb/s |
| `nk_each_scale_i8_skylake`    |                6.74 gb/s |                6.70 gb/s |                6.93 gb/s |
| `nk_each_scale_i8_sapphire`   |                23.0 gb/s |                11.4 gb/s |                10.8 gb/s |
| `nk_each_blend_i8_serial`     |                3.66 gb/s |                2.23 gb/s |                2.60 gb/s |
| `nk_each_blend_i8_haswell`    |                5.95 gb/s |                5.37 gb/s |                6.37 gb/s |
| `nk_each_blend_i8_sapphire`   |                32.4 gb/s |                17.7 gb/s |                15.2 gb/s |
| `nk_each_fma_i8_serial`       |                4.49 gb/s |                2.63 gb/s |                2.98 gb/s |
| `nk_each_fma_i8_haswell`      |                7.36 gb/s |                6.84 gb/s |                7.15 gb/s |
| `nk_each_fma_i8_skylake`      |                11.2 gb/s |                9.45 gb/s |                10.1 gb/s |
| `nk_each_fma_i8_sapphire`     |                32.2 gb/s |                21.2 gb/s |                18.7 gb/s |
| __u8__                        | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_u8_serial`       |                17.0 gb/s |                13.7 gb/s |                12.5 gb/s |
| `nk_each_sum_u8_haswell`      |                42.6 gb/s |                15.7 gb/s |                15.4 gb/s |
| `nk_each_sum_u8_icelake`      |                45.9 gb/s |                17.3 gb/s |                17.4 gb/s |
| `nk_each_scale_u8_serial`     |                2.11 gb/s |                2.03 gb/s |                1.86 gb/s |
| `nk_each_scale_u8_haswell`    |                3.91 gb/s |                3.89 gb/s |                4.27 gb/s |
| `nk_each_scale_u8_skylake`    |                6.93 gb/s |                5.99 gb/s |                6.70 gb/s |
| `nk_each_scale_u8_sapphire`   |                24.7 gb/s |                12.1 gb/s |                11.9 gb/s |
| `nk_each_blend_u8_serial`     |                3.23 gb/s |                2.62 gb/s |                3.43 gb/s |
| `nk_each_blend_u8_haswell`    |                4.87 gb/s |                5.10 gb/s |                5.61 gb/s |
| `nk_each_blend_u8_sapphire`   |                39.8 gb/s |                18.1 gb/s |                16.5 gb/s |
| `nk_each_fma_u8_serial`       |                3.19 gb/s |                3.92 gb/s |                4.54 gb/s |
| `nk_each_fma_u8_haswell`      |                6.98 gb/s |                6.29 gb/s |                7.62 gb/s |
| `nk_each_fma_u8_skylake`      |                9.66 gb/s |                9.21 gb/s |                10.3 gb/s |
| `nk_each_fma_u8_sapphire`     |                25.3 gb/s |                21.2 gb/s |                19.0 gb/s |
| __i16__                       | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_i16_serial`      |                12.0 gb/s |                11.6 gb/s |                14.2 gb/s |
| `nk_each_sum_i16_haswell`     |                26.1 gb/s |                16.2 gb/s |                16.8 gb/s |
| `nk_each_sum_i16_icelake`     |                39.3 gb/s |                18.2 gb/s |                18.0 gb/s |
| `nk_each_scale_i16_serial`    |                3.19 gb/s |                4.25 gb/s |                3.77 gb/s |
| `nk_each_scale_i16_haswell`   |                7.07 gb/s |                7.42 gb/s |                7.69 gb/s |
| `nk_each_scale_i16_skylake`   |                12.8 gb/s |                8.94 gb/s |                10.5 gb/s |
| `nk_each_blend_i16_serial`    |                6.00 gb/s |                6.04 gb/s |                6.37 gb/s |
| `nk_each_fma_i16_serial`      |                8.37 gb/s |                7.64 gb/s |                7.07 gb/s |
| `nk_each_fma_i16_haswell`     |                10.7 gb/s |                12.5 gb/s |                13.4 gb/s |
| `nk_each_fma_i16_skylake`     |                18.3 gb/s |                15.7 gb/s |                19.0 gb/s |
| __u16__                       | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_u16_serial`      |                12.2 gb/s |                11.8 gb/s |                13.7 gb/s |
| `nk_each_sum_u16_haswell`     |                24.1 gb/s |                14.4 gb/s |                17.7 gb/s |
| `nk_each_sum_u16_icelake`     |                39.7 gb/s |                18.2 gb/s |                18.4 gb/s |
| `nk_each_scale_u16_serial`    |                4.44 gb/s |                5.28 gb/s |                5.48 gb/s |
| `nk_each_scale_u16_haswell`   |                7.82 gb/s |                7.21 gb/s |                7.66 gb/s |
| `nk_each_scale_u16_skylake`   |                15.5 gb/s |                11.6 gb/s |                9.04 gb/s |
| `nk_each_blend_u16_serial`    |                7.84 gb/s |                7.08 gb/s |                9.15 gb/s |
| `nk_each_fma_u16_serial`      |                9.01 gb/s |                8.19 gb/s |                9.37 gb/s |
| `nk_each_fma_u16_haswell`     |                11.2 gb/s |                12.2 gb/s |                13.9 gb/s |
| `nk_each_fma_u16_skylake`     |                22.6 gb/s |                20.8 gb/s |                17.5 gb/s |
| __i32__                       | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_i32_serial`      |                12.6 gb/s |                12.7 gb/s |                10.1 gb/s |
| `nk_each_sum_i32_haswell`     |                14.6 gb/s |                15.3 gb/s |                13.9 gb/s |
| `nk_each_sum_i32_icelake`     |                17.1 gb/s |                18.5 gb/s |                15.4 gb/s |
| `nk_each_scale_i32_serial`    |                3.04 gb/s |                3.28 gb/s |                3.44 gb/s |
| `nk_each_scale_i32_haswell`   |                8.94 gb/s |                8.72 gb/s |                8.09 gb/s |
| `nk_each_scale_i32_skylake`   |                11.1 gb/s |                12.6 gb/s |                11.3 gb/s |
| `nk_each_blend_i32_serial`    |                4.99 gb/s |                5.99 gb/s |                5.06 gb/s |
| `nk_each_fma_i32_serial`      |                6.33 gb/s |                6.38 gb/s |                6.40 gb/s |
| `nk_each_fma_i32_haswell`     |                13.5 gb/s |                16.4 gb/s |                11.2 gb/s |
| `nk_each_fma_i32_skylake`     |                20.1 gb/s |                21.3 gb/s |                16.0 gb/s |
| __u32__                       | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_u32_serial`      |                13.6 gb/s |                11.6 gb/s |                9.94 gb/s |
| `nk_each_sum_u32_haswell`     |                16.0 gb/s |                17.7 gb/s |                12.7 gb/s |
| `nk_each_sum_u32_icelake`     |                17.4 gb/s |                19.0 gb/s |                14.8 gb/s |
| `nk_each_scale_u32_serial`    |                2.13 gb/s |                3.22 gb/s |                2.74 gb/s |
| `nk_each_scale_u32_haswell`   |                8.20 gb/s |                9.40 gb/s |                9.18 gb/s |
| `nk_each_scale_u32_skylake`   |                10.6 gb/s |                12.9 gb/s |                11.1 gb/s |
| `nk_each_blend_u32_serial`    |                3.76 gb/s |                5.22 gb/s |                5.78 gb/s |
| `nk_each_fma_u32_serial`      |                4.78 gb/s |                5.63 gb/s |                8.43 gb/s |
| `nk_each_fma_u32_haswell`     |                13.7 gb/s |                16.1 gb/s |                12.1 gb/s |
| `nk_each_fma_u32_skylake`     |                20.2 gb/s |                21.1 gb/s |                15.2 gb/s |
| __i64__                       | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_i64_serial`      |                15.9 gb/s |                15.8 gb/s |                10.3 gb/s |
| `nk_each_sum_i64_icelake`     |                17.7 gb/s |                19.2 gb/s |                11.1 gb/s |
| `nk_each_scale_i64_serial`    |                7.44 gb/s |                9.17 gb/s |                8.72 gb/s |
| `nk_each_scale_i64_skylake`   |                11.8 gb/s |                13.8 gb/s |                9.10 gb/s |
| `nk_each_blend_i64_serial`    |                10.9 gb/s |                14.3 gb/s |                10.5 gb/s |
| `nk_each_fma_i64_serial`      |                13.5 gb/s |                19.0 gb/s |                11.8 gb/s |
| `nk_each_fma_i64_skylake`     |                21.7 gb/s |                22.2 gb/s |                11.8 gb/s |
| __u64__                       | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_u64_serial`      |                14.7 gb/s |                16.9 gb/s |                10.5 gb/s |
| `nk_each_sum_u64_icelake`     |                18.1 gb/s |                19.2 gb/s |                12.4 gb/s |
| `nk_each_scale_u64_serial`    |                7.99 gb/s |                9.79 gb/s |                8.82 gb/s |
| `nk_each_scale_u64_skylake`   |                11.6 gb/s |                13.9 gb/s |                7.24 gb/s |
| `nk_each_blend_u64_serial`    |                11.9 gb/s |                16.5 gb/s |                13.8 gb/s |
| `nk_each_fma_u64_serial`      |                15.3 gb/s |                21.6 gb/s |                14.0 gb/s |
| `nk_each_fma_u64_skylake`     |                21.6 gb/s |                21.8 gb/s |                11.4 gb/s |
| __f64c__                      | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_scale_f64c_serial`   |       10.6 gb/s, 3.4 ulp |       7.04 gb/s, 2.6 ulp |       5.63 gb/s, 2.0 ulp |
| `nk_each_scale_f64c_haswell`  |       9.89 gb/s, 3.5 ulp |       6.45 gb/s, 2.0 ulp |       5.90 gb/s, 2.1 ulp |
| `nk_each_scale_f64c_skylake`  |       9.29 gb/s, 3.5 ulp |       6.01 gb/s, 2.0 ulp |       5.67 gb/s, 2.0 ulp |
| `nk_each_blend_f64c_serial`   |       16.2 gb/s, 2.5 ulp |       8.80 gb/s, 2.4 ulp |       8.23 gb/s, 2.6 ulp |
| `nk_each_blend_f64c_haswell`  |       14.0 gb/s, 2.5 ulp |       8.26 gb/s, 2.4 ulp |       8.59 gb/s, 2.7 ulp |
| `nk_each_blend_f64c_skylake`  |       14.3 gb/s, 2.5 ulp |       8.82 gb/s, 2.7 ulp |       7.33 gb/s, 2.7 ulp |
| `nk_each_fma_f64c_serial`     |       17.4 gb/s, 4.5 ulp |       10.1 gb/s, 3.2 ulp |       9.45 gb/s, 2.8 ulp |
| `nk_each_fma_f64c_haswell`    |       14.5 gb/s, 4.9 ulp |       8.86 gb/s, 3.1 ulp |       9.31 gb/s, 2.7 ulp |
| `nk_each_fma_f64c_skylake`    |       16.1 gb/s, 4.0 ulp |       8.93 gb/s, 3.4 ulp |       10.4 gb/s, 2.8 ulp |
| __f32c__                      | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_scale_f32c_serial`   |       10.1 gb/s, 2.1 ulp |       9.28 gb/s, 1.8 ulp |       6.20 gb/s, 2.1 ulp |
| `nk_each_scale_f32c_haswell`  |       8.81 gb/s, 2.1 ulp |       8.87 gb/s, 1.8 ulp |       7.73 gb/s, 2.1 ulp |
| `nk_each_scale_f32c_skylake`  |       9.67 gb/s, 2.2 ulp |       7.61 gb/s, 1.8 ulp |       7.09 gb/s, 2.1 ulp |
| `nk_each_blend_f32c_serial`   |       14.6 gb/s, 6.9 ulp |       10.8 gb/s, 2.8 ulp |       8.43 gb/s, 8.7 ulp |
| `nk_each_blend_f32c_haswell`  |       13.0 gb/s, 8.7 ulp |       12.1 gb/s, 2.9 ulp |       10.5 gb/s, 9.7 ulp |
| `nk_each_blend_f32c_skylake`  |       14.0 gb/s, 7.7 ulp |       9.28 gb/s, 2.7 ulp |      9.47 gb/s, 10.2 ulp |
| `nk_each_fma_f32c_serial`     |       15.7 gb/s, 8.8 ulp |      11.96 gb/s, 3.9 ulp |       9.33 gb/s, 8.5 ulp |
| `nk_each_fma_f32c_haswell`    |       14.1 gb/s, 6.6 ulp |       10.3 gb/s, 2.9 ulp |       9.77 gb/s, 7.2 ulp |
| `nk_each_fma_f32c_skylake`    |       15.5 gb/s, 9.2 ulp |       10.3 gb/s, 3.8 ulp |       9.95 gb/s, 8.3 ulp |

### Apple M4

#### Native

| Kernel                         |                      256 |                     1024 |                     4096 |
| :----------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f64__                        | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_f64_serial`       |         19.0 gb/s, 0 ulp |         14.0 gb/s, 0 ulp |         14.8 gb/s, 0 ulp |
| `nk_each_sum_f64_neon`         |         14.4 gb/s, 0 ulp |         16.6 gb/s, 0 ulp |         23.0 gb/s, 0 ulp |
| `nk_each_scale_f64_serial`     |         35.8 gb/s, 0 ulp |         16.3 gb/s, 0 ulp |         17.4 gb/s, 0 ulp |
| `nk_each_scale_f64_neon`       |         46.2 gb/s, 0 ulp |         25.8 gb/s, 0 ulp |         33.0 gb/s, 0 ulp |
| `nk_each_blend_f64_serial`     |         18.3 gb/s, 0 ulp |         12.4 gb/s, 0 ulp |         13.0 gb/s, 0 ulp |
| `nk_each_blend_f64_neon`       |       34.0 gb/s, 1.9 ulp |       17.8 gb/s, 2.7 ulp |       24.0 gb/s, 1.8 ulp |
| `nk_each_fma_f64_serial`       |         12.3 gb/s, 0 ulp |         9.51 gb/s, 0 ulp |         10.1 gb/s, 0 ulp |
| `nk_each_fma_f64_neon`         |       23.5 gb/s, 1.4 ulp |       12.9 gb/s, 1.6 ulp |       19.5 gb/s, 1.6 ulp |
| __f32__                        | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_f32_serial`       |         21.1 gb/s, 0 ulp |         10.4 gb/s, 0 ulp |         12.9 gb/s, 0 ulp |
| `nk_each_sum_f32_neon`         |         32.7 gb/s, 0 ulp |         34.7 gb/s, 0 ulp |         28.1 gb/s, 0 ulp |
| `nk_each_scale_f32_serial`     |         24.9 gb/s, 0 ulp |         26.8 gb/s, 0 ulp |         17.7 gb/s, 0 ulp |
| `nk_each_scale_f32_neon`       |         50.4 gb/s, 0 ulp |         42.2 gb/s, 0 ulp |         56.5 gb/s, 0 ulp |
| `nk_each_blend_f32_serial`     |        16.8 gb/s, 26 ulp |        9.57 gb/s, 26 ulp |       13.7 gb/s, 2.0 ulp |
| `nk_each_blend_f32_neon`       |       16.0 gb/s, 1.7 ulp |       9.46 gb/s, 1.6 ulp |       11.0 gb/s, 1.6 ulp |
| `nk_each_fma_f32_serial`       |       13.6 gb/s, 2.1 ulp |       7.80 gb/s, 2.5 ulp |       9.92 gb/s, 2.2 ulp |
| `nk_each_fma_f32_neon`         |       13.2 gb/s, 2.1 ulp |        8.85 gb/s, 21 ulp |       8.78 gb/s, 1.8 ulp |
| __bf16__                       | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_bf16_serial`      |         10.6 gb/s, 0 ulp |         9.80 gb/s, 0 ulp |         10.6 gb/s, 0 ulp |
| `nk_each_sum_bf16_neonbfdot`   |         24.7 gb/s, 0 ulp |         24.4 gb/s, 0 ulp |         23.4 gb/s, 0 ulp |
| `nk_each_scale_bf16_serial`    |         13.1 gb/s, 0 ulp |         11.0 gb/s, 0 ulp |         12.9 gb/s, 0 ulp |
| `nk_each_scale_bf16_neonbfdot` |         28.3 gb/s, 0 ulp |         27.2 gb/s, 0 ulp |         19.5 gb/s, 0 ulp |
| `nk_each_blend_bf16_serial`    |        9.13 gb/s, 28 ulp |        8.36 gb/s, 26 ulp |       8.42 gb/s, 2.2 ulp |
| `nk_each_blend_bf16_neonbfdot` |        11.6 gb/s, 29 ulp |        9.57 gb/s, 29 ulp |       8.71 gb/s, 2.2 ulp |
| `nk_each_fma_bf16_serial`      |       7.68 gb/s, 2.1 ulp |       6.56 gb/s, 2.0 ulp |        6.37 gb/s, 33 ulp |
| `nk_each_fma_bf16_neonbfdot`   |       8.74 gb/s, 1.2 ulp |       7.53 gb/s, 1.2 ulp |       6.22 gb/s, 1.5 ulp |
| __f16__                        | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_f16_serial`       |         19.0 gb/s, 0 ulp |         17.1 gb/s, 0 ulp |         11.0 gb/s, 0 ulp |
| `nk_each_sum_f16_neonhalf`     |         28.9 gb/s, 0 ulp |         33.8 gb/s, 0 ulp |         20.2 gb/s, 0 ulp |
| `nk_each_scale_f16_serial`     |         19.4 gb/s, 0 ulp |         18.1 gb/s, 0 ulp |         14.2 gb/s, 0 ulp |
| `nk_each_scale_f16_neonhalf`   |     51.4 gb/s, 87.3K ulp |     49.7 gb/s, 84.8K ulp |     36.1 gb/s, 87.3K ulp |
| `nk_each_blend_f16_serial`     |       12.0 gb/s, 2.0 ulp |       10.3 gb/s, 2.0 ulp |       9.46 gb/s, 2.3 ulp |
| `nk_each_blend_f16_neonhalf`   |     18.0 gb/s, 91.6K ulp |     15.5 gb/s, 92.6K ulp |     10.1 gb/s, 91.9K ulp |
| `nk_each_fma_f16_serial`       |       9.94 gb/s, 2.1 ulp |       8.29 gb/s, 1.8 ulp |       8.52 gb/s, 2.2 ulp |
| `nk_each_fma_f16_neonhalf`     |     14.7 gb/s, 97.1K ulp |     11.1 gb/s, 96.7K ulp |     10.7 gb/s, 99.2K ulp |
| __e4m3__                       | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_e4m3_serial`      |        0.122 gb/s, 0 ulp |        0.123 gb/s, 0 ulp |        0.125 gb/s, 0 ulp |
| `nk_each_sum_e4m3_neon`        |        0.856 gb/s, 0 ulp |        0.896 gb/s, 0 ulp |        0.909 gb/s, 0 ulp |
| `nk_each_scale_e4m3_serial`    |        0.108 gb/s, 0 ulp |        0.109 gb/s, 0 ulp |        0.109 gb/s, 0 ulp |
| `nk_each_scale_e4m3_neon`      |         1.03 gb/s, 0 ulp |         1.11 gb/s, 0 ulp |         1.12 gb/s, 0 ulp |
| `nk_each_blend_e4m3_serial`    |     0.0907 gb/s, 0.4 ulp |     0.0907 gb/s, 1.1 ulp |     0.0885 gb/s, 2.8 ulp |
| `nk_each_blend_e4m3_neon`      |      0.845 gb/s, 0.1 ulp |        0.905 gb/s, 0 ulp |        0.910 gb/s, 0 ulp |
| `nk_each_fma_e4m3_serial`      |     0.0808 gb/s, 0.7 ulp |     0.0795 gb/s, 0.6 ulp |     0.0790 gb/s, 2.2 ulp |
| `nk_each_fma_e4m3_neon`        |      0.707 gb/s, 0.1 ulp |      0.766 gb/s, 0.6 ulp |      0.765 gb/s, 1.0 ulp |
| __e5m2__                       | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_e5m2_serial`      |        0.163 gb/s, 0 ulp |        0.160 gb/s, 0 ulp |        0.161 gb/s, 0 ulp |
| `nk_each_sum_e5m2_neon`        |         1.70 gb/s, 0 ulp |         1.87 gb/s, 0 ulp |         1.88 gb/s, 0 ulp |
| `nk_each_scale_e5m2_serial`    |        0.137 gb/s, 0 ulp |        0.134 gb/s, 0 ulp |        0.134 gb/s, 0 ulp |
| `nk_each_scale_e5m2_neon`      |         1.78 gb/s, 0 ulp |         1.94 gb/s, 0 ulp |         1.96 gb/s, 0 ulp |
| `nk_each_blend_e5m2_serial`    |        0.126 gb/s, 0 ulp |        0.123 gb/s, 0 ulp |      0.123 gb/s, 4.9 ulp |
| `nk_each_blend_e5m2_neon`      |       1.62 gb/s, 0.7 ulp |         1.79 gb/s, 0 ulp |         1.79 gb/s, 0 ulp |
| `nk_each_fma_e5m2_serial`      |      0.117 gb/s, 1.9 ulp |      0.114 gb/s, 0.9 ulp |      0.114 gb/s, 4.0 ulp |
| `nk_each_fma_e5m2_neon`        |         1.48 gb/s, 0 ulp |       1.62 gb/s, 1.3 ulp |         1.62 gb/s, 0 ulp |
| __e2m3__                       | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_e2m3_serial`      |        0.124 gb/s, ? ulp |        0.117 gb/s, ? ulp |        0.115 gb/s, ? ulp |
| `nk_each_scale_e2m3_serial`    |       0.0793 gb/s, ? ulp |       0.0769 gb/s, ? ulp |       0.0769 gb/s, ? ulp |
| `nk_each_blend_e2m3_serial`    |       0.0856 gb/s, ? ulp |       0.0828 gb/s, ? ulp |       0.0828 gb/s, ? ulp |
| `nk_each_fma_e2m3_serial`      |       0.0884 gb/s, ? ulp |       0.0842 gb/s, ? ulp |       0.0852 gb/s, ? ulp |
| __e3m2__                       | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_e3m2_serial`      |        0.181 gb/s, ? ulp |        0.175 gb/s, ? ulp |        0.173 gb/s, ? ulp |
| `nk_each_scale_e3m2_serial`    |        0.124 gb/s, ? ulp |        0.117 gb/s, ? ulp |        0.117 gb/s, ? ulp |
| `nk_each_blend_e3m2_serial`    |        0.123 gb/s, ? ulp |        0.116 gb/s, ? ulp |        0.115 gb/s, ? ulp |
| `nk_each_fma_e3m2_serial`      |        0.118 gb/s, ? ulp |        0.110 gb/s, ? ulp |        0.109 gb/s, ? ulp |
| __i8__                         | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_i8_serial`        |                20.6 gb/s |                24.2 gb/s |                12.6 gb/s |
| `nk_each_sum_i8_neonhalf`      |                41.5 gb/s |                40.2 gb/s |                36.6 gb/s |
| `nk_each_scale_i8_serial`      |                2.88 gb/s |                2.85 gb/s |                2.84 gb/s |
| `nk_each_scale_i8_neonhalf`    |                24.4 gb/s |                22.7 gb/s |                12.0 gb/s |
| `nk_each_blend_i8_serial`      |                2.29 gb/s |                2.32 gb/s |                2.24 gb/s |
| `nk_each_blend_i8_neonhalf`    |                8.83 gb/s |                9.48 gb/s |                8.68 gb/s |
| `nk_each_fma_i8_serial`        |                2.02 gb/s |                1.97 gb/s |                1.77 gb/s |
| `nk_each_fma_i8_neonhalf`      |                7.27 gb/s |                6.97 gb/s |                4.79 gb/s |
| __u8__                         | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_u8_serial`        |                4.46 gb/s |                4.49 gb/s |                4.09 gb/s |
| `nk_each_sum_u8_neonhalf`      |                35.0 gb/s |                38.1 gb/s |                20.2 gb/s |
| `nk_each_scale_u8_serial`      |                3.38 gb/s |                3.41 gb/s |                3.34 gb/s |
| `nk_each_scale_u8_neonhalf`    |                24.4 gb/s |                22.7 gb/s |                21.0 gb/s |
| `nk_each_blend_u8_serial`      |                2.59 gb/s |                2.53 gb/s |                2.48 gb/s |
| `nk_each_blend_u8_neonhalf`    |                9.21 gb/s |                8.53 gb/s |                8.00 gb/s |
| `nk_each_fma_u8_serial`        |                2.17 gb/s |                2.16 gb/s |                1.86 gb/s |
| `nk_each_fma_u8_neonhalf`      |                7.29 gb/s |                7.11 gb/s |                4.72 gb/s |
| __i16__                        | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_i16_serial`       |                20.2 gb/s |                24.3 gb/s |                11.1 gb/s |
| `nk_each_sum_i16_neon`         |                31.4 gb/s |                36.8 gb/s |                36.7 gb/s |
| `nk_each_scale_i16_serial`     |                5.36 gb/s |                5.35 gb/s |                5.49 gb/s |
| `nk_each_scale_i16_neon`       |                24.9 gb/s |                24.5 gb/s |                19.9 gb/s |
| `nk_each_blend_i16_serial`     |                4.23 gb/s |                3.87 gb/s |                4.08 gb/s |
| `nk_each_fma_i16_serial`       |                3.74 gb/s |                3.31 gb/s |                3.16 gb/s |
| `nk_each_fma_i16_neon`         |                11.5 gb/s |                11.7 gb/s |                11.7 gb/s |
| __u16__                        | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_u16_serial`       |                7.95 gb/s |                7.73 gb/s |                6.73 gb/s |
| `nk_each_sum_u16_neon`         |                32.0 gb/s |                39.3 gb/s |                35.9 gb/s |
| `nk_each_scale_u16_serial`     |                6.43 gb/s |                6.36 gb/s |                6.07 gb/s |
| `nk_each_scale_u16_neon`       |                25.0 gb/s |                24.5 gb/s |                25.7 gb/s |
| `nk_each_blend_u16_serial`     |                5.07 gb/s |                4.95 gb/s |                4.73 gb/s |
| `nk_each_fma_u16_serial`       |                4.41 gb/s |                4.03 gb/s |                3.73 gb/s |
| `nk_each_fma_u16_neon`         |                11.5 gb/s |                11.8 gb/s |                10.7 gb/s |
| __i32__                        | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_i32_serial`       |                20.9 gb/s |                14.3 gb/s |                12.6 gb/s |
| `nk_each_sum_i32_neon`         |                38.8 gb/s |                39.0 gb/s |                36.2 gb/s |
| `nk_each_scale_i32_serial`     |                4.34 gb/s |                4.14 gb/s |                4.24 gb/s |
| `nk_each_scale_i32_neon`       |                22.6 gb/s |                25.6 gb/s |                25.6 gb/s |
| `nk_each_blend_i32_serial`     |                4.02 gb/s |                3.57 gb/s |                3.84 gb/s |
| `nk_each_fma_i32_serial`       |                3.47 gb/s |                3.01 gb/s |                3.35 gb/s |
| `nk_each_fma_i32_neon`         |                11.5 gb/s |                11.7 gb/s |                11.7 gb/s |
| __u32__                        | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_u32_serial`       |                13.1 gb/s |                13.4 gb/s |                9.76 gb/s |
| `nk_each_sum_u32_neon`         |                33.2 gb/s |                28.1 gb/s |                27.0 gb/s |
| `nk_each_scale_u32_serial`     |                4.72 gb/s |                4.76 gb/s |                5.56 gb/s |
| `nk_each_scale_u32_neon`       |                22.5 gb/s |                25.8 gb/s |                25.6 gb/s |
| `nk_each_blend_u32_serial`     |                4.70 gb/s |                4.38 gb/s |                4.70 gb/s |
| `nk_each_fma_u32_serial`       |                4.03 gb/s |                4.04 gb/s |                3.98 gb/s |
| `nk_each_fma_u32_neon`         |                11.4 gb/s |                11.8 gb/s |                10.7 gb/s |
| __i64__                        | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_i64_serial`       |                16.7 gb/s |                15.5 gb/s |                11.2 gb/s |
| `nk_each_sum_i64_neon`         |                38.5 gb/s |                39.4 gb/s |                28.6 gb/s |
| `nk_each_scale_i64_serial`     |                6.94 gb/s |                7.26 gb/s |                8.77 gb/s |
| `nk_each_scale_i64_neon`       |                50.0 gb/s |                50.6 gb/s |                44.5 gb/s |
| `nk_each_blend_i64_serial`     |                9.02 gb/s |                7.98 gb/s |                8.72 gb/s |
| `nk_each_fma_i64_serial`       |                7.19 gb/s |                7.29 gb/s |                7.32 gb/s |
| `nk_each_fma_i64_neon`         |                28.4 gb/s |                28.2 gb/s |                20.6 gb/s |
| __u64__                        | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_u64_serial`       |                17.8 gb/s |                11.8 gb/s |                11.7 gb/s |
| `nk_each_sum_u64_neon`         |                39.2 gb/s |                37.7 gb/s |                28.6 gb/s |
| `nk_each_scale_u64_serial`     |                9.80 gb/s |                8.70 gb/s |                11.8 gb/s |
| `nk_each_scale_u64_neon`       |                38.6 gb/s |                50.9 gb/s |                43.7 gb/s |
| `nk_each_blend_u64_serial`     |                9.31 gb/s |                8.52 gb/s |                8.76 gb/s |
| `nk_each_fma_u64_serial`       |                7.59 gb/s |                7.91 gb/s |                6.65 gb/s |
| `nk_each_fma_u64_neon`         |                19.1 gb/s |                27.7 gb/s |                20.6 gb/s |
| __f64c__                       | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_scale_f64c_serial`    |       15.6 gb/s, 2.2 ulp |       15.8 gb/s, 1.9 ulp |       19.7 gb/s, 2.2 ulp |
| `nk_each_scale_f64c_neon`      |       24.1 gb/s, 1.5 ulp |       51.2 gb/s, 1.5 ulp |       35.5 gb/s, 1.3 ulp |
| `nk_each_blend_f64c_serial`    |       12.2 gb/s, 4.2 ulp |       10.3 gb/s, 3.0 ulp |       17.4 gb/s, 2.6 ulp |
| `nk_each_blend_f64c_neon`      |       17.2 gb/s, 3.2 ulp |       24.7 gb/s, 3.0 ulp |       23.1 gb/s, 2.2 ulp |
| `nk_each_fma_f64c_serial`      |       6.73 gb/s, 3.4 ulp |       8.11 gb/s, 5.2 ulp |       13.0 gb/s, 2.5 ulp |
| `nk_each_fma_f64c_neon`        |       12.5 gb/s, 3.2 ulp |       18.8 gb/s, 3.0 ulp |       17.8 gb/s, 2.4 ulp |
| __f32c__                       | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_scale_f32c_serial`    |        23.1 gb/s, 17 ulp |       13.3 gb/s, 2.4 ulp |        13.6 gb/s, 17 ulp |
| `nk_each_scale_f32c_neon`      |       18.6 gb/s, 1.8 ulp |       38.3 gb/s, 1.6 ulp |       32.9 gb/s, 1.6 ulp |
| `nk_each_blend_f32c_serial`    |       13.7 gb/s, 2.4 ulp |       8.74 gb/s, 2.6 ulp |        12.4 gb/s, 28 ulp |
| `nk_each_blend_f32c_neon`      |       26.9 gb/s, 2.2 ulp |       21.8 gb/s, 2.2 ulp |       22.4 gb/s, 3.2 ulp |
| `nk_each_fma_f32c_serial`      |       10.2 gb/s, 3.1 ulp |        8.04 gb/s, 78 ulp |       9.05 gb/s, 3.5 ulp |
| `nk_each_fma_f32c_neon`        |       22.4 gb/s, 2.9 ulp |       17.1 gb/s, 4.1 ulp |        17.8 gb/s, 81 ulp |

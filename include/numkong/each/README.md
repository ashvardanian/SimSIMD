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

| Kernel                       |                      256 |                     1024 |                     4096 |
| :--------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f64__                      | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_f64_serial`     |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_each_sum_f64_haswell`    |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_each_sum_f64_skylake`    |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_each_scale_f64_serial`   |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_each_scale_f64_haswell`  |         12.4 gb/s, 0 ulp |         13.1 gb/s, 0 ulp |         7.87 gb/s, 0 ulp |
| `nk_each_scale_f64_skylake`  |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_each_blend_f64_serial`   |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_each_blend_f64_haswell`  |       17.6 gb/s, 1.4 ulp |       18.1 gb/s, 1.5 ulp |       12.9 gb/s, 1.8 ulp |
| `nk_each_blend_f64_skylake`  |       18.5 gb/s, 1.4 ulp |       18.0 gb/s, 1.5 ulp |       11.7 gb/s, 1.8 ulp |
| `nk_each_fma_f64_serial`     |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_each_fma_f64_haswell`    |       21.5 gb/s, 1.9 ulp |       20.9 gb/s, 1.8 ulp |       14.3 gb/s, 1.8 ulp |
| `nk_each_fma_f64_skylake`    |       23.3 gb/s, 1.9 ulp |       22.2 gb/s, 1.8 ulp |       13.1 gb/s, 1.8 ulp |
| __f32__                      | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_f32_serial`     |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_each_sum_f32_haswell`    |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_each_sum_f32_skylake`    |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_each_scale_f32_serial`   |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_each_scale_f32_haswell`  |         12.4 gb/s, 0 ulp |         12.4 gb/s, 0 ulp |         13.2 gb/s, 0 ulp |
| `nk_each_scale_f32_skylake`  |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_each_blend_f32_serial`   |         0 gb/s, 52.7 ulp |          0 gb/s, 1.5 ulp |         0 gb/s, 11.4 ulp |
| `nk_each_blend_f32_haswell`  |       18.5 gb/s, 1.9 ulp |       18.3 gb/s, 1.5 ulp |       16.8 gb/s, 2.0 ulp |
| `nk_each_blend_f32_skylake`  |       18.5 gb/s, 1.9 ulp |       17.3 gb/s, 1.5 ulp |      17.7 gb/s, 10.5 ulp |
| `nk_each_fma_f32_serial`     |          0 gb/s, 2.2 ulp |          0 gb/s, 2.1 ulp |          0 gb/s, 2.1 ulp |
| `nk_each_fma_f32_haswell`    |       23.2 gb/s, 2.2 ulp |       21.6 gb/s, 1.9 ulp |       16.6 gb/s, 1.8 ulp |
| `nk_each_fma_f32_skylake`    |       23.1 gb/s, 2.2 ulp |       22.0 gb/s, 1.9 ulp |       17.6 gb/s, 1.9 ulp |
| __bf16__                     | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_bf16_serial`    |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_each_sum_bf16_haswell`   |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_each_sum_bf16_skylake`   |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_each_scale_bf16_serial`  |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_each_scale_bf16_haswell` |         9.51 gb/s, 0 ulp |         10.1 gb/s, 0 ulp |         8.11 gb/s, 0 ulp |
| `nk_each_scale_bf16_skylake` |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_each_blend_bf16_serial`  |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_each_blend_bf16_haswell` |      14.1 gb/s, 44.5 ulp |       13.1 gb/s, 1.4 ulp |       14.6 gb/s, 1.6 ulp |
| `nk_each_blend_bf16_skylake` |      19.7 gb/s, 38.4 ulp |       13.2 gb/s, 6.0 ulp |       17.9 gb/s, 1.6 ulp |
| `nk_each_fma_bf16_serial`    |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_each_fma_bf16_haswell`   |       17.7 gb/s, 1.7 ulp |      15.7 gb/s, 15.2 ulp |      14.8 gb/s, 35.9 ulp |
| `nk_each_fma_bf16_skylake`   |       22.1 gb/s, 1.4 ulp |      17.2 gb/s, 15.7 ulp |      20.6 gb/s, 34.9 ulp |
| __f16__                      | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_f16_serial`     |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_each_sum_f16_haswell`    |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_each_sum_f16_sapphire`   |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_each_scale_f16_serial`   |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_each_scale_f16_haswell`  |         18.8 gb/s, 0 ulp |         11.6 gb/s, 0 ulp |         13.0 gb/s, 0 ulp |
| `nk_each_blend_f16_serial`   |      0.854 gb/s, 1.3 ulp |      0.801 gb/s, 1.5 ulp |      0.807 gb/s, 7.2 ulp |
| `nk_each_blend_f16_haswell`  |       23.3 gb/s, 1.3 ulp |       16.3 gb/s, 6.3 ulp |      16.8 gb/s, 11.1 ulp |
| `nk_each_fma_f16_serial`     |       1.01 gb/s, 1.1 ulp |      0.949 gb/s, 1.3 ulp |       0.98 gb/s, 1.3 ulp |
| `nk_each_fma_f16_haswell`    |       24.2 gb/s, 1.2 ulp |       18.5 gb/s, 1.3 ulp |       19.4 gb/s, 8.8 ulp |
| __e4m3__                     | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_e4m3_serial`    |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_each_scale_e4m3_serial`  |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_each_blend_e4m3_serial`  |         0 gb/s, 31.3 ulp |          0 gb/s, 0.3 ulp |          0 gb/s, 0.3 ulp |
| `nk_each_fma_e4m3_serial`    |          0 gb/s, 0.3 ulp |          0 gb/s, 0.4 ulp |          0 gb/s, 0.7 ulp |
| __e5m2__                     | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_e5m2_serial`    |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_each_scale_e5m2_serial`  |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_each_blend_e5m2_serial`  |          0 gb/s, 0.5 ulp |         0 gb/s, 15.7 ulp |          0 gb/s, 2.8 ulp |
| `nk_each_fma_e5m2_serial`    |          0 gb/s, 1.5 ulp |          0 gb/s, 0.3 ulp |          0 gb/s, 1.0 ulp |
| __e2m3__                     | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_e2m3_serial`    |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_each_scale_e2m3_serial`  |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_each_blend_e2m3_serial`  |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_each_fma_e2m3_serial`    |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| __e3m2__                     | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_e3m2_serial`    |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_each_scale_e3m2_serial`  |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_each_blend_e3m2_serial`  |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_each_fma_e3m2_serial`    |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| __i8__                       | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_i8_serial`      |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_each_sum_i8_haswell`     |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_each_sum_i8_icelake`     |                48.0 gb/s |                17.1 gb/s |                18.5 gb/s |
| `nk_each_scale_i8_serial`    |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_each_scale_i8_haswell`   |                7.61 gb/s |                7.02 gb/s |                6.74 gb/s |
| `nk_each_scale_i8_skylake`   |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_each_scale_i8_sapphire`  |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_each_blend_i8_serial`    |                3.86 gb/s |                2.66 gb/s |                3.56 gb/s |
| `nk_each_blend_i8_haswell`   |                11.3 gb/s |                9.82 gb/s |                9.39 gb/s |
| `nk_each_blend_i8_sapphire`  |                31.1 gb/s |                13.7 gb/s |                18.0 gb/s |
| `nk_each_fma_i8_serial`      |                4.80 gb/s |                3.53 gb/s |                4.54 gb/s |
| `nk_each_fma_i8_haswell`     |                13.9 gb/s |                11.1 gb/s |                11.1 gb/s |
| `nk_each_fma_i8_skylake`     |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_each_fma_i8_sapphire`    |                36.1 gb/s |                16.4 gb/s |                21.3 gb/s |
| __u8__                       | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_u8_serial`      |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_each_sum_u8_haswell`     |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_each_sum_u8_icelake`     |                47.5 gb/s |                17.1 gb/s |                18.6 gb/s |
| `nk_each_scale_u8_serial`    |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_each_scale_u8_haswell`   |                7.52 gb/s |                6.97 gb/s |                6.38 gb/s |
| `nk_each_scale_u8_skylake`   |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_each_scale_u8_sapphire`  |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_each_blend_u8_serial`    |                4.93 gb/s |                3.22 gb/s |                4.49 gb/s |
| `nk_each_blend_u8_haswell`   |                10.9 gb/s |                9.37 gb/s |                7.91 gb/s |
| `nk_each_blend_u8_sapphire`  |                40.8 gb/s |                15.0 gb/s |                18.3 gb/s |
| `nk_each_fma_u8_serial`      |                6.24 gb/s |                3.93 gb/s |                5.61 gb/s |
| `nk_each_fma_u8_haswell`     |                13.1 gb/s |                11.3 gb/s |                11.4 gb/s |
| `nk_each_fma_u8_skylake`     |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_each_fma_u8_sapphire`    |                41.8 gb/s |                16.5 gb/s |                21.3 gb/s |
| __i16__                      | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_i16_serial`     |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_each_sum_i16_haswell`    |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_each_sum_i16_icelake`    |                39.3 gb/s |                17.2 gb/s |                19.2 gb/s |
| `nk_each_scale_i16_serial`   |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_each_scale_i16_haswell`  |                12.9 gb/s |                10.4 gb/s |                10.1 gb/s |
| `nk_each_scale_i16_skylake`  |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_each_blend_i16_serial`   |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_each_fma_i16_serial`     |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_each_fma_i16_haswell`    |                18.6 gb/s |                16.5 gb/s |                17.9 gb/s |
| `nk_each_fma_i16_skylake`    |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| __u16__                      | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_u16_serial`     |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_each_sum_u16_haswell`    |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_each_sum_u16_icelake`    |                39.8 gb/s |                17.3 gb/s |                19.1 gb/s |
| `nk_each_scale_u16_serial`   |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_each_scale_u16_haswell`  |                10.2 gb/s |                10.8 gb/s |                11.7 gb/s |
| `nk_each_scale_u16_skylake`  |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_each_blend_u16_serial`   |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_each_fma_u16_serial`     |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_each_fma_u16_haswell`    |                18.6 gb/s |                16.5 gb/s |                19.2 gb/s |
| `nk_each_fma_u16_skylake`    |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| __i32__                      | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_i32_serial`     |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_each_sum_i32_haswell`    |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_each_sum_i32_icelake`    |                18.7 gb/s |                17.8 gb/s |                18.2 gb/s |
| `nk_each_scale_i32_serial`   |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_each_scale_i32_haswell`  |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_each_scale_i32_skylake`  |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_each_blend_i32_serial`   |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_each_fma_i32_serial`     |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_each_fma_i32_haswell`    |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_each_fma_i32_skylake`    |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| __u32__                      | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_u32_serial`     |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_each_sum_u32_haswell`    |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_each_sum_u32_icelake`    |                18.8 gb/s |                17.8 gb/s |                18.5 gb/s |
| `nk_each_scale_u32_serial`   |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_each_scale_u32_haswell`  |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_each_scale_u32_skylake`  |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_each_blend_u32_serial`   |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_each_fma_u32_serial`     |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_each_fma_u32_haswell`    |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_each_fma_u32_skylake`    |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| __i64__                      | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_i64_serial`     |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_each_sum_i64_icelake`    |                18.8 gb/s |                18.5 gb/s |                13.9 gb/s |
| `nk_each_scale_i64_serial`   |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_each_scale_i64_skylake`  |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_each_blend_i64_serial`   |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_each_fma_i64_serial`     |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_each_fma_i64_skylake`    |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| __u64__                      | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_u64_serial`     |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_each_sum_u64_icelake`    |                18.9 gb/s |                18.7 gb/s |                13.9 gb/s |
| `nk_each_scale_u64_serial`   |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_each_scale_u64_skylake`  |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_each_blend_u64_serial`   |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_each_fma_u64_serial`     |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_each_fma_u64_skylake`    |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| __f64c__                     | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_scale_f64c_serial`  |          0 gb/s, 3.0 ulp |          0 gb/s, 2.3 ulp |          0 gb/s, 2.1 ulp |
| `nk_each_scale_f64c_haswell` |          0 gb/s, 4.3 ulp |          0 gb/s, 2.3 ulp |          0 gb/s, 2.1 ulp |
| `nk_each_scale_f64c_skylake` |          0 gb/s, 3.0 ulp |          0 gb/s, 2.3 ulp |          0 gb/s, 2.1 ulp |
| `nk_each_blend_f64c_serial`  |          0 gb/s, 2.9 ulp |          0 gb/s, 3.7 ulp |          0 gb/s, 3.6 ulp |
| `nk_each_blend_f64c_haswell` |          0 gb/s, 3.1 ulp |          0 gb/s, 3.7 ulp |          0 gb/s, 3.5 ulp |
| `nk_each_blend_f64c_skylake` |          0 gb/s, 2.9 ulp |          0 gb/s, 3.7 ulp |          0 gb/s, 3.6 ulp |
| `nk_each_fma_f64c_serial`    |          0 gb/s, 3.6 ulp |          0 gb/s, 3.6 ulp |          0 gb/s, 3.7 ulp |
| `nk_each_fma_f64c_haswell`   |          0 gb/s, 3.4 ulp |          0 gb/s, 3.6 ulp |          0 gb/s, 3.7 ulp |
| `nk_each_fma_f64c_skylake`   |          0 gb/s, 3.6 ulp |          0 gb/s, 3.6 ulp |          0 gb/s, 3.7 ulp |
| __f32c__                     | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_scale_f32c_serial`  |          0 gb/s, 2.3 ulp |         0 gb/s, 17.2 ulp |          0 gb/s, 7.2 ulp |
| `nk_each_scale_f32c_haswell` |          0 gb/s, 2.3 ulp |         0 gb/s, 16.8 ulp |          0 gb/s, 7.1 ulp |
| `nk_each_scale_f32c_skylake` |          0 gb/s, 2.3 ulp |         0 gb/s, 18.9 ulp |          0 gb/s, 7.1 ulp |
| `nk_each_blend_f32c_serial`  |         0 gb/s, 19.5 ulp |          0 gb/s, 7.3 ulp |         0 gb/s, 20.2 ulp |
| `nk_each_blend_f32c_haswell` |         0 gb/s, 21.9 ulp |          0 gb/s, 7.2 ulp |         0 gb/s, 21.4 ulp |
| `nk_each_blend_f32c_skylake` |         0 gb/s, 20.0 ulp |          0 gb/s, 7.4 ulp |         0 gb/s, 20.0 ulp |
| `nk_each_fma_f32c_serial`    |          0 gb/s, 3.6 ulp |         0 gb/s, 28.7 ulp |         0 gb/s, 29.0 ulp |
| `nk_each_fma_f32c_haswell`   |          0 gb/s, 3.6 ulp |         0 gb/s, 28.0 ulp |         0 gb/s, 28.4 ulp |
| `nk_each_fma_f32c_skylake`   |          0 gb/s, 3.5 ulp |         0 gb/s, 28.8 ulp |         0 gb/s, 28.1 ulp |

### Apple M4 Pro

#### Native

| Kernel                         |                      256 |                     1024 |                     4096 |
| :----------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f64__                        | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_f64_serial`       |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_each_sum_f64_neon`         |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_each_scale_f64_serial`     |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_each_scale_f64_neon`       |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_each_blend_f64_serial`     |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_each_blend_f64_neon`       |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_each_fma_f64_serial`       |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_each_fma_f64_neon`         |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| __f32__                        | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_f32_serial`       |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_each_sum_f32_neon`         |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_each_scale_f32_serial`     |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_each_scale_f32_neon`       |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_each_blend_f32_serial`     |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_each_blend_f32_neon`       |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_each_fma_f32_serial`       |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_each_fma_f32_neon`         |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| __bf16__                       | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_bf16_serial`      |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_each_sum_bf16_neonbfdot`   |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_each_scale_bf16_serial`    |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_each_scale_bf16_neonbfdot` |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_each_blend_bf16_serial`    |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_each_blend_bf16_neonbfdot` |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_each_fma_bf16_serial`      |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_each_fma_bf16_neonbfdot`   |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| __f16__                        | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_f16_serial`       |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_each_sum_f16_neonhalf`     |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_each_scale_f16_serial`     |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_each_scale_f16_neonhalf`   |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_each_blend_f16_serial`     |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_each_blend_f16_neonhalf`   |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_each_fma_f16_serial`       |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_each_fma_f16_neonhalf`     |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| __e4m3__                       | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_e4m3_serial`      |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_each_sum_e4m3_neon`        |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_each_scale_e4m3_serial`    |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_each_scale_e4m3_neon`      |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_each_blend_e4m3_serial`    |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_each_blend_e4m3_neon`      |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_each_fma_e4m3_serial`      |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_each_fma_e4m3_neon`        |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| __e5m2__                       | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_e5m2_serial`      |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_each_sum_e5m2_neon`        |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_each_scale_e5m2_serial`    |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_each_scale_e5m2_neon`      |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_each_blend_e5m2_serial`    |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_each_blend_e5m2_neon`      |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_each_fma_e5m2_serial`      |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_each_fma_e5m2_neon`        |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| __e2m3__                       | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_e2m3_serial`      |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_each_scale_e2m3_serial`    |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_each_blend_e2m3_serial`    |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_each_fma_e2m3_serial`      |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| __e3m2__                       | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_e3m2_serial`      |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_each_scale_e3m2_serial`    |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_each_blend_e3m2_serial`    |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_each_fma_e3m2_serial`      |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| __i8__                         | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_i8_serial`        |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_each_sum_i8_neonhalf`      |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_each_scale_i8_serial`      |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_each_scale_i8_neonhalf`    |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_each_blend_i8_serial`      |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_each_blend_i8_neonhalf`    |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_each_fma_i8_serial`        |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_each_fma_i8_neonhalf`      |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| __u8__                         | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_u8_serial`        |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_each_sum_u8_neonhalf`      |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_each_scale_u8_serial`      |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_each_scale_u8_neonhalf`    |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_each_blend_u8_serial`      |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_each_blend_u8_neonhalf`    |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_each_fma_u8_serial`        |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_each_fma_u8_neonhalf`      |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| __i16__                        | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_i16_serial`       |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_each_sum_i16_neon`         |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_each_scale_i16_serial`     |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_each_scale_i16_neon`       |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_each_blend_i16_serial`     |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_each_fma_i16_serial`       |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_each_fma_i16_neon`         |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| __u16__                        | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_u16_serial`       |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_each_sum_u16_neon`         |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_each_scale_u16_serial`     |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_each_scale_u16_neon`       |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_each_blend_u16_serial`     |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_each_fma_u16_serial`       |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_each_fma_u16_neon`         |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| __i32__                        | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_i32_serial`       |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_each_sum_i32_neon`         |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_each_scale_i32_serial`     |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_each_scale_i32_neon`       |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_each_blend_i32_serial`     |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_each_fma_i32_serial`       |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_each_fma_i32_neon`         |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| __u32__                        | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_u32_serial`       |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_each_sum_u32_neon`         |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_each_scale_u32_serial`     |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_each_scale_u32_neon`       |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_each_blend_u32_serial`     |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_each_fma_u32_serial`       |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_each_fma_u32_neon`         |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| __i64__                        | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_i64_serial`       |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_each_sum_i64_neon`         |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_each_scale_i64_serial`     |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_each_scale_i64_neon`       |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_each_blend_i64_serial`     |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_each_fma_i64_serial`       |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_each_fma_i64_neon`         |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| __u64__                        | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_sum_u64_serial`       |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_each_sum_u64_neon`         |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_each_scale_u64_serial`     |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_each_scale_u64_neon`       |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_each_blend_u64_serial`     |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_each_fma_u64_serial`       |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_each_fma_u64_neon`         |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| __f64c__                       | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_scale_f64c_serial`    |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_each_scale_f64c_neon`      |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_each_blend_f64c_serial`    |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_each_blend_f64c_neon`      |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_each_fma_f64c_serial`      |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_each_fma_f64c_neon`        |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| __f32c__                       | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_each_scale_f32c_serial`    |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_each_scale_f32c_neon`      |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_each_blend_f32c_serial`    |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_each_blend_f32c_neon`      |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_each_fma_f32c_serial`      |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_each_fma_f32c_neon`        |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |

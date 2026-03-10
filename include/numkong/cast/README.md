# Type Conversions in NumKong

NumKong implements bidirectional type conversions between all supported numeric formats through f32 as a hub type.
Conversions cover IEEE 754 floats (f16, f32, f64), brain float (bf16), FP8 formats (e4m3, e5m2, e2m3, e3m2), and integers (i8-i64, u8-u64, packed i4x2/u4x2).
All conversions use round-to-nearest-even (RNE) for narrowing and exact widening where the target format has sufficient range and precision.

BF16 relates to F32 by truncation with rounding:

```math
\text{bf16} \approx \text{f32} \gg 16
```

With RNE tie-breaking to preserve the least significant bit of the truncated result.

F16 range and precision:

```math
\text{f16} \in [-65504, 65504], \quad \text{min positive normal} = 2^{-14}
```

Reformulating as Python pseudocode:

```python
import numpy as np

def cast(a: np.ndarray, target_dtype: np.dtype) -> np.ndarray:
    return a.astype(target_dtype)
```

## Input & Output Types

Float-to-float conversions:

| Input Type | Output Type | Description                                |
| ---------- | ----------- | ------------------------------------------ |
| `f64`      | `f32`       | 64-bit to 32-bit, narrowing with RNE       |
| `f32`      | `f64`       | 32-bit to 64-bit, exact widening           |
| `f32`      | `f16`       | 32-bit to 16-bit half precision            |
| `f16`      | `f32`       | 16-bit half to 32-bit, exact widening      |
| `f32`      | `bf16`      | 32-bit to brain float, truncation with RNE |
| `bf16`     | `f32`       | Brain float to 32-bit, exact widening      |

Float-to-FP8 conversions:

| Input Type | Output Type | Description                                |
| ---------- | ----------- | ------------------------------------------ |
| `f32`      | `e4m3`      | 32-bit to FP8: 4 exponent, 3 mantissa bits |
| `e4m3`     | `f32`       | FP8 to 32-bit, exact via lookup table      |
| `f32`      | `e5m2`      | 32-bit to FP8: 5 exponent, 2 mantissa bits |
| `e5m2`     | `f32`       | FP8 to 32-bit, exact via lookup table      |
| `f32`      | `e2m3`      | 32-bit to MX: 2 exponent, 3 mantissa bits  |
| `e2m3`     | `f32`       | MX to 32-bit, exact via lookup table       |
| `f32`      | `e3m2`      | 32-bit to MX: 3 exponent, 2 mantissa bits  |
| `e3m2`     | `f32`       | MX to 32-bit, exact via lookup table       |

Float-to-integer conversions:

| Input Type | Output Type | Description                         |
| ---------- | ----------- | ----------------------------------- |
| `f32`      | `i8`        | Clamped to [-128, 127], rounded     |
| `f32`      | `u8`        | Clamped to [0, 255], rounded        |
| `f32`      | `i16`       | Clamped to [-32768, 32767], rounded |
| `f32`      | `u16`       | Clamped to [0, 65535], rounded      |
| `f64`      | `i32`       | Clamped to i32 range, rounded       |
| `f64`      | `u32`       | Clamped to u32 range, rounded       |
| `f64`      | `i64`       | Clamped to i64 range, rounded       |
| `f64`      | `u64`       | Clamped to u64 range, rounded       |

Packed sub-byte conversions:

| Input Type | Output Type | Description                                      |
| ---------- | ----------- | ------------------------------------------------ |
| `i4x2`     | `i8`        | Signed 4-bit pair to two signed 8-bit values     |
| `u4x2`     | `u8`        | Unsigned 4-bit pair to two unsigned 8-bit values |

## Optimizations

### Lookup Tables for FP8 and Mini-Floats

`nk_e4m3_to_f32_serial`, `nk_e5m2_to_f32_serial`, `nk_e2m3_to_f32_serial`, `nk_e3m2_to_f32_serial` use 256-entry precomputed lookup tables -- each 8-bit input indexes directly into an f32 result array.
The reverse direction (`nk_f32_to_e4m3_serial`) uses clamping + rounding: clamp to format range, multiply by scale, round-to-nearest, cast to u8.
SIMD backends (`nk_cast_haswell`, `nk_cast_skylake`) use `VPGATHERDD` to perform 8 or 16 simultaneous table lookups from the same 256-entry table.
AVX-512 gathers on Skylake achieve ~3cy throughput per 16-element lookup vs ~8cy on Haswell for 8-element gathers.

### BF16 as Truncated F32

`nk_bf16_to_f32_serial` zero-extends by left-shifting 16 bits -- exact, no rounding error, single-cycle on all platforms.
`nk_f32_to_bf16_serial` right-shifts with round-to-nearest-even: adds a rounding bias of `0x7FFF + ((bits >> 16) & 1)` before truncating, matching the IEEE 754 RNE tie-breaking rule.
NEON backend uses `vreinterpretq_u16_u8` + `vzip` for zero-extension; Haswell uses `VPSLLD` / `VPSRLD` shifts.

### F16C Hardware Conversion

`nk_f16_to_f32_haswell`, `nk_f32_to_f16_haswell` use the F16C extension instructions `VCVTPH2PS` / `VCVTPS2PH` -- single-instruction conversion of 8 elements with correct denormal handling, NaN propagation, and RNE rounding.
The serial fallback (`nk_f16_to_f32_serial`) must handle denormals via explicit exponent/mantissa extraction and conditional re-normalization -- ~15 integer ops per element vs 1 instruction with F16C.
AVX-512 (`nk_cast_skylake`) doubles throughput to 16 elements per instruction.

## Performance

The following performance tables are produced by manually running `nk_bench` included internal tools to measure the throughput at different input shapes.
The input size is controlled by the `NK_DENSE_DIMENSIONS` environment variable and set to 256, 1024, and 4096 elements.
The throughput is measured in GB/s as the number of bytes read and written per second, with ↓ for downcasts and ↑ for upcasts.
Each kernel runs for at least 5 seconds per configuration.
Benchmark threads are pinned to specific cores; on machines with heterogeneous core types (e.g., Apple P/E cores), only the fastest cores are used.
Workloads that significantly degrade CPU frequencies (Intel AMX, Apple SME) run in separate passes to avoid affecting throughput measurements of other kernels.

### Intel Sapphire Rapids

#### Native

| Kernel             |        ↓ 256 |         ↓ 1K |         ↓ 4K |        ↑ 256 |         ↑ 1K |         ↑ 4K |
| :----------------- | -----------: | -----------: | -----------: | -----------: | -----------: | -----------: |
| __f32 ↔ bf16__     | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial`   |   0.542 gb/s |   0.521 gb/s |   0.553 gb/s |    1.10 gb/s |    1.12 gb/s |    1.17 gb/s |
| `nk_cast_haswell`  |    40.8 gb/s |    52.4 gb/s |    55.1 gb/s |    27.7 gb/s |    43.2 gb/s |    46.3 gb/s |
| `nk_cast_skylake`  |    23.6 gb/s |    44.8 gb/s |    46.8 gb/s |    37.6 gb/s |    60.1 gb/s |    61.3 gb/s |
| `nk_cast_icelake`  |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |
| `nk_cast_sapphire` |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |
| __f32 ↔ f16__      | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial`   |    6.28 gb/s |    6.36 gb/s |    6.20 gb/s |    2.93 gb/s |    2.95 gb/s |    2.89 gb/s |
| `nk_cast_haswell`  |    50.2 gb/s |     106 gb/s |     105 gb/s |    31.7 gb/s |    60.2 gb/s |    66.1 gb/s |
| `nk_cast_skylake`  |    38.0 gb/s |    56.6 gb/s |    39.4 gb/s |    39.7 gb/s |    58.3 gb/s |    43.7 gb/s |
| `nk_cast_icelake`  |    51.8 gb/s |    60.2 gb/s |    54.3 gb/s |    52.2 gb/s |    57.7 gb/s |    60.6 gb/s |
| `nk_cast_sapphire` |    48.9 gb/s |    60.8 gb/s |    54.3 gb/s |    51.0 gb/s |    56.4 gb/s |    57.4 gb/s |
| __f32 ↔ e5m2__     | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial`   |   0.785 gb/s |   0.725 gb/s |   0.569 gb/s |    2.62 gb/s |    2.57 gb/s |    2.69 gb/s |
| `nk_cast_haswell`  |    7.93 gb/s |    8.39 gb/s |    5.44 gb/s |    12.6 gb/s |    17.9 gb/s |    10.6 gb/s |
| `nk_cast_skylake`  |    10.3 gb/s |    10.8 gb/s |    10.0 gb/s |    27.2 gb/s |    28.6 gb/s |    28.0 gb/s |
| `nk_cast_icelake`  |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |
| `nk_cast_sapphire` |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |
| __f32 ↔ e4m3__     | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial`   |   0.653 gb/s |   0.623 gb/s |   0.445 gb/s |    1.51 gb/s |    1.43 gb/s |    1.44 gb/s |
| `nk_cast_haswell`  |    6.74 gb/s |    7.35 gb/s |    6.68 gb/s |    10.4 gb/s |    12.1 gb/s |    7.47 gb/s |
| `nk_cast_skylake`  |    7.70 gb/s |    9.83 gb/s |    9.79 gb/s |    17.3 gb/s |    23.2 gb/s |    22.2 gb/s |
| `nk_cast_icelake`  |    8.51 gb/s |    9.01 gb/s |    9.43 gb/s |    17.8 gb/s |    20.5 gb/s |    21.4 gb/s |
| `nk_cast_sapphire` |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |
| __f32 ↔ e3m2__     | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial`   |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |
| `nk_cast_haswell`  |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |
| `nk_cast_skylake`  |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |
| `nk_cast_icelake`  |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |
| `nk_cast_sapphire` |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |
| __f32 ↔ e2m3__     | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial`   |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |
| `nk_cast_haswell`  |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |
| `nk_cast_skylake`  |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |
| `nk_cast_icelake`  |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |
| `nk_cast_sapphire` |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |
| __f32 ↔ i16__      | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial`   |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |
| `nk_cast_haswell`  |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |
| `nk_cast_skylake`  |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |
| `nk_cast_icelake`  |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |
| `nk_cast_sapphire` |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |
| __f32 ↔ u16__      | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial`   |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |
| `nk_cast_haswell`  |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |
| `nk_cast_skylake`  |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |
| `nk_cast_icelake`  |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |
| `nk_cast_sapphire` |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |
| __f32 ↔ i8__       | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial`   |    3.22 gb/s |    3.62 gb/s |    3.40 gb/s |    5.41 gb/s |    5.65 gb/s |    5.73 gb/s |
| `nk_cast_haswell`  |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |
| `nk_cast_skylake`  |    13.0 gb/s |    13.2 gb/s |    13.9 gb/s |    22.1 gb/s |    23.4 gb/s |    22.9 gb/s |
| `nk_cast_icelake`  |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |
| `nk_cast_sapphire` |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |
| __f32 ↔ u8__       | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial`   |    4.44 gb/s |    4.58 gb/s |    5.84 gb/s |    7.45 gb/s |    7.20 gb/s |       ? gb/s |
| `nk_cast_haswell`  |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |
| `nk_cast_skylake`  |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |
| `nk_cast_icelake`  |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |
| `nk_cast_sapphire` |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |
| __f64 ↔ f32__      | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial`   |    11.6 gb/s |    12.2 gb/s |    12.3 gb/s |    12.1 gb/s |    12.9 gb/s |    13.2 gb/s |
| `nk_cast_haswell`  |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |
| `nk_cast_skylake`  |    52.1 gb/s |    59.4 gb/s |    53.8 gb/s |    54.4 gb/s |    65.9 gb/s |    60.6 gb/s |
| `nk_cast_icelake`  |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |
| `nk_cast_sapphire` |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |
| __f64 ↔ i64__      | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial`   |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |
| `nk_cast_haswell`  |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |
| `nk_cast_skylake`  |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |
| `nk_cast_icelake`  |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |
| `nk_cast_sapphire` |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |
| __f64 ↔ u64__      | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial`   |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |
| `nk_cast_haswell`  |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |
| `nk_cast_skylake`  |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |
| `nk_cast_icelake`  |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |
| `nk_cast_sapphire` |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |
| __f64 ↔ i32__      | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial`   |    3.71 gb/s |    3.97 gb/s |    3.71 gb/s |    11.6 gb/s |    12.3 gb/s |    12.6 gb/s |
| `nk_cast_haswell`  |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |
| `nk_cast_skylake`  |    38.7 gb/s |    48.1 gb/s |    45.9 gb/s |    54.1 gb/s |    64.2 gb/s |    60.8 gb/s |
| `nk_cast_icelake`  |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |
| `nk_cast_sapphire` |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |
| __f64 ↔ u32__      | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial`   |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |
| `nk_cast_haswell`  |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |
| `nk_cast_skylake`  |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |
| `nk_cast_icelake`  |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |
| `nk_cast_sapphire` |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |

#### WASM

Measured with Wasmtime v42 (Cranelift backend).

| Kernel                |        ↓ 256 |         ↓ 1K |         ↓ 4K |        ↑ 256 |         ↑ 1K |         ↑ 4K |
| :-------------------- | -----------: | -----------: | -----------: | -----------: | -----------: | -----------: |
| __f32 ↔ bf16__        | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial`      |       0 gb/s |       0 gb/s |       0 gb/s |       0 gb/s |       0 gb/s |       0 gb/s |
| `nk_cast_v128relaxed` |       0 gb/s |       0 gb/s |       0 gb/s |       0 gb/s |       0 gb/s |       0 gb/s |
| __f32 ↔ f16__         | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial`      |       0 gb/s |       0 gb/s |       0 gb/s |       0 gb/s |       0 gb/s |       0 gb/s |
| `nk_cast_v128relaxed` |       0 gb/s |       0 gb/s |       0 gb/s |       0 gb/s |       0 gb/s |       0 gb/s |
| __f32 ↔ e5m2__        | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial`      |       0 gb/s |       0 gb/s |       0 gb/s |       0 gb/s |       0 gb/s |       0 gb/s |
| `nk_cast_v128relaxed` |       0 gb/s |       0 gb/s |       0 gb/s |       0 gb/s |       0 gb/s |       0 gb/s |
| __f32 ↔ e4m3__        | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial`      |       0 gb/s |       0 gb/s |       0 gb/s |       0 gb/s |       0 gb/s |       0 gb/s |
| `nk_cast_v128relaxed` |       0 gb/s |       0 gb/s |       0 gb/s |       0 gb/s |       0 gb/s |       0 gb/s |

### Apple M4 Pro

#### Native

| Kernel           |        ↓ 256 |         ↓ 1K |         ↓ 4K |        ↑ 256 |         ↑ 1K |         ↑ 4K |
| :--------------- | -----------: | -----------: | -----------: | -----------: | -----------: | -----------: |
| __f32 ↔ bf16__   | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial` |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |
| `nk_cast_neon`   |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |
| __f32 ↔ f16__    | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial` |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |
| `nk_cast_neon`   |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |
| __f32 ↔ e5m2__   | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial` |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |
| `nk_cast_neon`   |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |
| __f32 ↔ e4m3__   | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial` |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |
| `nk_cast_neon`   |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |
| __f32 ↔ e3m2__   | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial` |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |
| `nk_cast_neon`   |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |
| __f32 ↔ e2m3__   | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial` |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |
| `nk_cast_neon`   |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |
| __f32 ↔ i16__    | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial` |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |
| `nk_cast_neon`   |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |
| __f32 ↔ u16__    | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial` |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |
| `nk_cast_neon`   |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |
| __f32 ↔ i8__     | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial` |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |
| `nk_cast_neon`   |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |
| __f32 ↔ u8__     | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial` |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |
| `nk_cast_neon`   |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |
| __f64 ↔ f32__    | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial` |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |
| `nk_cast_neon`   |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |
| __f64 ↔ i64__    | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial` |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |
| `nk_cast_neon`   |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |
| __f64 ↔ u64__    | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial` |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |
| `nk_cast_neon`   |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |
| __f64 ↔ i32__    | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial` |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |
| `nk_cast_neon`   |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |
| __f64 ↔ u32__    | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial` |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |
| `nk_cast_neon`   |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |

#### WASM

Measured with Wasmtime v42 (Cranelift backend).

| Kernel                |        ↓ 256 |         ↓ 1K |         ↓ 4K |        ↑ 256 |         ↑ 1K |         ↑ 4K |
| :-------------------- | -----------: | -----------: | -----------: | -----------: | -----------: | -----------: |
| __f32 ↔ bf16__        | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial`      |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |
| `nk_cast_v128relaxed` |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |
| __f32 ↔ f16__         | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial`      |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |
| `nk_cast_v128relaxed` |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |
| __f32 ↔ e5m2__        | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial`      |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |
| `nk_cast_v128relaxed` |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |
| __f32 ↔ e4m3__        | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial`      |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |
| `nk_cast_v128relaxed` |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |

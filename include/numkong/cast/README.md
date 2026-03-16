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
| `e4m3`     | `f32`       | FP8 to 32-bit, exact via integer-add       |
| `f32`      | `e5m2`      | 32-bit to FP8: 5 exponent, 2 mantissa bits |
| `e5m2`     | `f32`       | FP8 to 32-bit, exact via integer-add       |
| `f32`      | `e2m3`      | 32-bit to MX: 2 exponent, 3 mantissa bits  |
| `e2m3`     | `f32`       | MX to 32-bit, exact via integer-add        |
| `f32`      | `e3m2`      | 32-bit to MX: 3 exponent, 2 mantissa bits  |
| `e3m2`     | `f32`       | MX to 32-bit, exact via integer-add        |

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

### FP8/FP6 Upcast via Integer-Add (FTZ-Safe)

`nk_e4m3_to_f32_serial`, `nk_e5m2_to_f32_serial`, `nk_e2m3_to_f32_serial`, `nk_e3m2_to_f32_serial` use 256-entry precomputed lookup tables -- each 8-bit input indexes directly into an f32 result array.
The reverse direction (`nk_f32_to_e4m3_serial`) uses clamping + rounding: clamp to format range, multiply by scale, round-to-nearest, cast to u8.

SIMD backends use a Giesen-inspired integer-add trick instead of the original magic-multiply approach.
The magic-multiply method places magnitude bits as a denormal f32 mantissa, then float-multiplies by a power of 2 to rebias the exponent.
This breaks when FTZ/DAZ is set (as `nk_configure_thread` intentionally does for performance), flushing denormal intermediates to zero.
The integer-add approach replaces the float multiply with an integer add for normal values (`shifted + (127-bias)<<23`), and uses `cvt_int_to_float * scale` for subnormal values.
Both paths produce only normal f32 intermediates — no denormals, correct under any FTZ/DAZ setting.

On AVX-512, masked operations (`_mm512_mask_cvtepi32_ps`, `_mm512_mask_mul_ps`, `_mm512_mask_or_epi32`) fold the subnormal blend and inf/NaN fixup into single instructions.
On RVV, merge-undisturbed masked intrinsics (`_mu` variants) achieve the same optimization.

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
| `nk_cast_icelake`  |    21.4 gb/s |    26.0 gb/s |    27.2 gb/s |    32.6 gb/s |    39.4 gb/s |    44.3 gb/s |
| `nk_cast_sapphire` |    21.5 gb/s |    21.1 gb/s |    49.5 gb/s |    39.2 gb/s |    38.3 gb/s |    56.3 gb/s |
| __f32 ↔ f16__      | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial`   |    6.28 gb/s |    6.36 gb/s |    6.20 gb/s |    2.93 gb/s |    2.95 gb/s |    2.89 gb/s |
| `nk_cast_haswell`  |    50.2 gb/s |     106 gb/s |     105 gb/s |    31.7 gb/s |    60.2 gb/s |    66.1 gb/s |
| `nk_cast_skylake`  |    38.0 gb/s |    56.6 gb/s |    39.4 gb/s |    39.7 gb/s |    58.3 gb/s |    43.7 gb/s |
| `nk_cast_icelake`  |    51.8 gb/s |    60.2 gb/s |    54.3 gb/s |    52.2 gb/s |    57.7 gb/s |    60.6 gb/s |
| `nk_cast_sapphire` |    31.8 gb/s |    33.8 gb/s |    38.8 gb/s |    35.0 gb/s |    33.6 gb/s |    51.5 gb/s |
| __f32 ↔ e5m2__     | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial`   |   0.785 gb/s |   0.725 gb/s |   0.569 gb/s |    4.60 gb/s |    4.75 gb/s |    4.25 gb/s |
| `nk_cast_haswell`  |    7.93 gb/s |    8.39 gb/s |    5.44 gb/s |    14.5 gb/s |    14.9 gb/s |    15.7 gb/s |
| `nk_cast_skylake`  |    10.3 gb/s |    10.8 gb/s |    10.0 gb/s |    26.0 gb/s |    28.8 gb/s |    28.8 gb/s |
| `nk_cast_icelake`  |    5.07 gb/s |    4.96 gb/s |    6.08 gb/s |    24.5 gb/s |    30.8 gb/s |    28.0 gb/s |
| `nk_cast_sapphire` |    7.81 gb/s |    5.25 gb/s |    10.7 gb/s |    26.2 gb/s |    30.5 gb/s |    29.3 gb/s |
| __f32 ↔ e4m3__     | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial`   |   0.653 gb/s |   0.623 gb/s |   0.445 gb/s |    2.80 gb/s |    3.00 gb/s |    2.97 gb/s |
| `nk_cast_haswell`  |    6.74 gb/s |    7.35 gb/s |    6.68 gb/s |    13.9 gb/s |    14.3 gb/s |    13.7 gb/s |
| `nk_cast_skylake`  |    7.70 gb/s |    9.83 gb/s |    9.79 gb/s |    24.7 gb/s |    26.6 gb/s |    28.5 gb/s |
| `nk_cast_icelake`  |    8.51 gb/s |    9.01 gb/s |    9.43 gb/s |    24.8 gb/s |    26.4 gb/s |    27.9 gb/s |
| `nk_cast_sapphire` |    4.98 gb/s |    4.90 gb/s |    8.56 gb/s |    25.0 gb/s |    26.8 gb/s |    26.7 gb/s |
| __f32 ↔ e3m2__     | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial`   |   0.863 gb/s |    1.44 gb/s |    1.21 gb/s |    4.23 gb/s |    4.27 gb/s |    4.49 gb/s |
| `nk_cast_haswell`  |    4.70 gb/s |    5.04 gb/s |    5.00 gb/s |    15.0 gb/s |    16.1 gb/s |    16.2 gb/s |
| `nk_cast_skylake`  |    6.34 gb/s |    6.37 gb/s |    6.46 gb/s |    29.3 gb/s |    36.6 gb/s |    34.6 gb/s |
| `nk_cast_icelake`  |    5.34 gb/s |    5.10 gb/s |    6.36 gb/s |    30.2 gb/s |    36.8 gb/s |    34.3 gb/s |
| `nk_cast_sapphire` |    8.78 gb/s |    9.93 gb/s |    7.02 gb/s |    30.1 gb/s |    34.1 gb/s |    34.8 gb/s |
| __f32 ↔ e2m3__     | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial`   |   0.941 gb/s |    1.39 gb/s |   0.688 gb/s |    4.37 gb/s |    4.86 gb/s |    4.94 gb/s |
| `nk_cast_haswell`  |    4.76 gb/s |    4.51 gb/s |    5.00 gb/s |    14.3 gb/s |    15.9 gb/s |    16.6 gb/s |
| `nk_cast_skylake`  |    6.55 gb/s |    6.54 gb/s |    6.42 gb/s |    29.6 gb/s |    33.6 gb/s |    35.8 gb/s |
| `nk_cast_icelake`  |    5.03 gb/s |    6.41 gb/s |    6.44 gb/s |    29.7 gb/s |    33.9 gb/s |    36.4 gb/s |
| `nk_cast_sapphire` |    9.95 gb/s |    8.90 gb/s |    9.17 gb/s |    27.8 gb/s |    33.6 gb/s |    36.0 gb/s |
| __f32 ↔ i16__      | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial`   |    1.99 gb/s |    2.02 gb/s |    2.04 gb/s |    4.59 gb/s |    4.63 gb/s |    4.68 gb/s |
| `nk_cast_haswell`  |    46.4 gb/s |    51.8 gb/s |    53.0 gb/s |    19.8 gb/s |    21.0 gb/s |    21.9 gb/s |
| `nk_cast_skylake`  |    31.0 gb/s |    34.2 gb/s |    36.7 gb/s |    48.7 gb/s |    58.5 gb/s |    61.1 gb/s |
| __f32 ↔ u16__      | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial`   |    3.19 gb/s |    3.13 gb/s |    3.14 gb/s |    4.60 gb/s |    4.82 gb/s |    4.75 gb/s |
| `nk_cast_haswell`  |    36.4 gb/s |    43.6 gb/s |    48.4 gb/s |    19.1 gb/s |    20.6 gb/s |    21.2 gb/s |
| `nk_cast_skylake`  |    32.0 gb/s |    36.1 gb/s |    37.3 gb/s |    48.4 gb/s |    55.0 gb/s |    59.5 gb/s |
| __f32 ↔ i8__       | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial`   |    3.22 gb/s |    3.62 gb/s |    3.40 gb/s |    5.41 gb/s |    5.65 gb/s |    5.73 gb/s |
| `nk_cast_haswell`  |    21.6 gb/s |    25.5 gb/s |    27.5 gb/s |    12.8 gb/s |    13.6 gb/s |    14.0 gb/s |
| `nk_cast_skylake`  |    13.0 gb/s |    13.2 gb/s |    13.9 gb/s |    22.1 gb/s |    23.4 gb/s |    22.9 gb/s |
| `nk_cast_icelake`  |    14.2 gb/s |    16.4 gb/s |    21.5 gb/s |    25.4 gb/s |    29.4 gb/s |    34.8 gb/s |
| `nk_cast_sapphire` |    26.0 gb/s |    27.3 gb/s |    19.5 gb/s |    33.1 gb/s |    48.9 gb/s |    49.4 gb/s |
| __f32 ↔ u8__       | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial`   |    4.44 gb/s |    4.58 gb/s |    5.84 gb/s |    7.45 gb/s |    7.20 gb/s |    4.24 gb/s |
| `nk_cast_haswell`  |    41.2 gb/s |    42.2 gb/s |    41.4 gb/s |    17.9 gb/s |    19.2 gb/s |    20.8 gb/s |
| `nk_cast_skylake`  |    27.8 gb/s |    31.1 gb/s |    33.4 gb/s |    39.8 gb/s |    48.7 gb/s |    51.5 gb/s |
| __f64 ↔ f32__      | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial`   |    11.6 gb/s |    12.2 gb/s |    12.3 gb/s |    12.1 gb/s |    12.9 gb/s |    13.2 gb/s |
| `nk_cast_skylake`  |    52.1 gb/s |    59.4 gb/s |    53.8 gb/s |    54.4 gb/s |    65.9 gb/s |    60.6 gb/s |
| __f64 ↔ i64__      | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial`   |    5.30 gb/s |    5.21 gb/s |    5.21 gb/s |    15.4 gb/s |    16.1 gb/s |    14.0 gb/s |
| `nk_cast_skylake`  |    8.73 gb/s |    9.81 gb/s |    9.03 gb/s |    25.3 gb/s |    26.8 gb/s |    20.3 gb/s |
| __f64 ↔ u64__      | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial`   |    9.17 gb/s |    8.55 gb/s |    8.57 gb/s |    16.3 gb/s |    15.1 gb/s |    15.0 gb/s |
| `nk_cast_skylake`  |    13.8 gb/s |    14.5 gb/s |    15.4 gb/s |    25.5 gb/s |    28.1 gb/s |    19.6 gb/s |
| __f64 ↔ i32__      | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial`   |    3.71 gb/s |    3.97 gb/s |    3.71 gb/s |    11.6 gb/s |    12.3 gb/s |    12.6 gb/s |
| `nk_cast_skylake`  |    38.7 gb/s |    48.1 gb/s |    45.9 gb/s |    54.1 gb/s |    64.2 gb/s |    60.8 gb/s |
| __f64 ↔ u32__      | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial`   |    6.37 gb/s |    6.16 gb/s |    6.08 gb/s |    10.9 gb/s |    11.9 gb/s |    10.3 gb/s |
| `nk_cast_skylake`  |    46.6 gb/s |    48.9 gb/s |    49.5 gb/s |    50.2 gb/s |    60.5 gb/s |    62.3 gb/s |

#### WASM

Measured with Wasmtime v42 (Cranelift backend).

| Kernel           |        ↓ 256 |         ↓ 1K |            ↓ 4K |        ↑ 256 |         ↑ 1K |            ↑ 4K |
| :--------------- | -----------: | -----------: | --------------: | -----------: | -----------: | --------------: |
| __f32 ↔ bf16__   | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░ |
| `nk_cast_serial` |       0 gb/s |       0 gb/s |       1.63 gb/s |       0 gb/s |       0 gb/s |       2.21 gb/s |
| __f32 ↔ f16__    | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░ |
| `nk_cast_serial` |       0 gb/s |       0 gb/s |      0.436 gb/s |       0 gb/s |       0 gb/s |       1.19 gb/s |
| __f32 ↔ e5m2__   | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░ |
| `nk_cast_serial` |       0 gb/s |       0 gb/s |      0.294 gb/s |       0 gb/s |       0 gb/s |       1.45 gb/s |
| __f32 ↔ e4m3__   | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░ |
| `nk_cast_serial` |       0 gb/s |       0 gb/s |      0.239 gb/s |       0 gb/s |       0 gb/s |      0.746 gb/s |

### Apple M4

#### Native

| Kernel           |        ↓ 256 |         ↓ 1K |         ↓ 4K |        ↑ 256 |         ↑ 1K |         ↑ 4K |
| :--------------- | -----------: | -----------: | -----------: | -----------: | -----------: | -----------: |
| __f32 ↔ bf16__   | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial` |    10.2 gb/s |    10.6 gb/s |    10.7 gb/s |    8.06 gb/s |    8.34 gb/s |    8.32 gb/s |
| `nk_cast_neon`   |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |
| __f32 ↔ f16__    | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial` |    10.9 gb/s |    11.3 gb/s |    11.4 gb/s |    8.40 gb/s |    8.62 gb/s |    8.70 gb/s |
| `nk_cast_neon`   |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |
| __f32 ↔ e5m2__   | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial` |    1.65 gb/s |    1.52 gb/s |    1.36 gb/s |    5.96 gb/s |    6.08 gb/s |    6.11 gb/s |
| `nk_cast_neon`   |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |
| __f32 ↔ e4m3__   | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial` |    1.49 gb/s |    1.36 gb/s |    1.24 gb/s |    4.96 gb/s |    5.05 gb/s |    4.81 gb/s |
| `nk_cast_neon`   |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |
| __f32 ↔ e3m2__   | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial` |    2.17 gb/s |    2.13 gb/s |    1.97 gb/s |    5.90 gb/s |    6.02 gb/s |    6.07 gb/s |
| `nk_cast_neon`   |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |
| __f32 ↔ e2m3__   | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial` |    2.54 gb/s |    2.45 gb/s |    2.23 gb/s |    5.88 gb/s |    6.11 gb/s |    6.10 gb/s |
| `nk_cast_neon`   |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |
| __f32 ↔ i16__    | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial` |    6.13 gb/s |    5.99 gb/s |    6.10 gb/s |    8.29 gb/s |    8.53 gb/s |    8.58 gb/s |
| `nk_cast_neon`   |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |
| __f32 ↔ u16__    | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial` |    5.36 gb/s |    5.01 gb/s |    4.49 gb/s |    8.43 gb/s |    8.64 gb/s |    8.76 gb/s |
| `nk_cast_neon`   |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |
| __f32 ↔ i8__     | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial` |    4.83 gb/s |    4.89 gb/s |    5.09 gb/s |    6.67 gb/s |    6.92 gb/s |    7.08 gb/s |
| `nk_cast_neon`   |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |
| __f32 ↔ u8__     | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial` |    4.31 gb/s |    4.10 gb/s |    3.62 gb/s |    7.03 gb/s |    7.21 gb/s |    7.28 gb/s |
| `nk_cast_neon`   |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |
| __f64 ↔ f32__    | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial` |    17.3 gb/s |    17.8 gb/s |    18.1 gb/s |    17.9 gb/s |    18.5 gb/s |    18.5 gb/s |
| `nk_cast_neon`   |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |
| __f64 ↔ i64__    | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial` |    16.8 gb/s |    17.2 gb/s |    17.0 gb/s |    23.9 gb/s |    24.7 gb/s |    24.8 gb/s |
| `nk_cast_neon`   |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |
| __f64 ↔ u64__    | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial` |    13.5 gb/s |    12.8 gb/s |    11.3 gb/s |    24.4 gb/s |    25.0 gb/s |    25.1 gb/s |
| `nk_cast_neon`   |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |
| __f64 ↔ i32__    | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial` |    12.1 gb/s |    12.4 gb/s |    12.6 gb/s |    18.2 gb/s |    18.9 gb/s |    19.2 gb/s |
| `nk_cast_neon`   |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |
| __f64 ↔ u32__    | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial` |    10.9 gb/s |    10.6 gb/s |    9.58 gb/s |    17.6 gb/s |    18.0 gb/s |    18.1 gb/s |
| `nk_cast_neon`   |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |       ? gb/s |

#### WASM

Measured with Wasmtime v42 (Cranelift backend).

| Kernel           |        ↓ 256 |         ↓ 1K |         ↓ 4K |        ↑ 256 |         ↑ 1K |         ↑ 4K |
| :--------------- | -----------: | -----------: | -----------: | -----------: | -----------: | -----------: |
| __f32 ↔ bf16__   | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial` |    3.39 gb/s |    7.29 gb/s |    11.2 gb/s |    3.08 gb/s |    6.20 gb/s |    8.81 gb/s |
| __f32 ↔ f16__    | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial` |   0.605 gb/s |   0.952 gb/s |    1.22 gb/s |    2.36 gb/s |    4.71 gb/s |    7.31 gb/s |
| __f32 ↔ e5m2__   | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial` |   0.752 gb/s |    1.84 gb/s |    1.80 gb/s |    2.24 gb/s |    6.32 gb/s |    6.31 gb/s |
| __f32 ↔ e4m3__   | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ | ░░░░░░░░░░░░ |
| `nk_cast_serial` |   0.623 gb/s |    1.61 gb/s |    1.50 gb/s |    1.68 gb/s |    4.35 gb/s |    4.28 gb/s |

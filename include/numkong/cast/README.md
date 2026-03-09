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

### Intel Sapphire Rapids

#### Native

<table>
<tr>
  <th>Kernel</th>
  <th>256</th>
  <th>1024</th>
  <th>4096</th>
</tr>
<tr><td colspan="4"><b>f32-f16</b></td></tr>
<tr>
  <td><code>nk_cast_haswell</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_cast_skylake</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_cast_sapphire</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>f16-f32</b></td></tr>
<tr>
  <td><code>nk_cast_haswell</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_cast_skylake</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_cast_sapphire</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>f32-bf16</b></td></tr>
<tr>
  <td><code>nk_cast_haswell</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_cast_skylake</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>bf16-f32</b></td></tr>
<tr>
  <td><code>nk_cast_haswell</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_cast_skylake</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>f32-e4m3</b></td></tr>
<tr>
  <td><code>nk_cast_haswell</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_cast_skylake</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>e4m3-f32</b></td></tr>
<tr>
  <td><code>nk_cast_haswell</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_cast_skylake</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>f32-e5m2</b></td></tr>
<tr>
  <td><code>nk_cast_haswell</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_cast_skylake</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>e5m2-f32</b></td></tr>
<tr>
  <td><code>nk_cast_haswell</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr>
  <td><code>nk_cast_skylake</code></td>
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
<tr><td colspan="4"><b>f32-f16</b></td></tr>
<tr>
  <td><code>nk_cast_neon</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>f16-f32</b></td></tr>
<tr>
  <td><code>nk_cast_neon</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>f32-bf16</b></td></tr>
<tr>
  <td><code>nk_cast_neon</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>bf16-f32</b></td></tr>
<tr>
  <td><code>nk_cast_neon</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>f32-e4m3</b></td></tr>
<tr>
  <td><code>nk_cast_neon</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>e4m3-f32</b></td></tr>
<tr>
  <td><code>nk_cast_neon</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>f32-e5m2</b></td></tr>
<tr>
  <td><code>nk_cast_neon</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
<tr><td colspan="4"><b>e5m2-f32</b></td></tr>
<tr>
  <td><code>nk_cast_neon</code></td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
  <td>0 GB/s<br>0 ULP, 0%</td>
</tr>
</table>

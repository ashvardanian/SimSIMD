# Horizontal Reductions in NumKong

NumKong implements single-pass horizontal reductions over dense vectors: statistical moments (sum + sum-of-squares) and extrema (min + max with argmin + argmax).
Both reductions traverse the input once, producing scalar outputs with compensated arithmetic for numerical stability.
The only module with full stride support -- `stride_bytes` controls the byte distance between consecutive logical elements, enabling column extraction from row-major matrices and strided array views without copying.
Used internally by packing routines for norm precomputation and by distance kernels for normalization.

Moments:

```math
\text{sum} = \sum a_i, \quad \text{sumsq} = \sum a_i^2
```

Min-max:

```math
\text{min} = \min_i a_i, \quad \text{argmin} = \arg\min_i a_i
```

Reformulating as Python pseudocode:

```python
import numpy as np

def moments(a: np.ndarray) -> tuple[float, float]:
    return np.sum(a), np.sum(a ** 2)

def minmax(a: np.ndarray) -> tuple[float, int, float, int]:
    return np.min(a), np.argmin(a), np.max(a), np.argmax(a)
```

## Input & Output Types

Float reductions:

| Input Type | Output Type | Description                           |
| ---------- | ----------- | ------------------------------------- |
| `f64`      | `f64`       | 64-bit double precision               |
| `f32`      | `f32`       | 32-bit single precision               |
| `f16`      | `f32`       | 16-bit half precision, widened output |
| `bf16`     | `f32`       | 16-bit brain float, widened output    |

FP8 reductions:

| Input Type | Output Type | Description                                  |
| ---------- | ----------- | -------------------------------------------- |
| `e4m3`     | `f32`       | 8-bit FP8: 4 exponent, 3 mantissa bits       |
| `e5m2`     | `f32`       | 8-bit FP8: 5 exponent, 2 mantissa bits       |
| `e2m3`     | `f32`       | 8-bit MX format: 2 exponent, 3 mantissa bits |
| `e3m2`     | `f32`       | 8-bit MX format: 3 exponent, 2 mantissa bits |

Integer reductions:

| Input Type | Output Type | Description                        |
| ---------- | ----------- | ---------------------------------- |
| `i8`       | `i64`       | 8-bit signed, widened to 64-bit    |
| `u8`       | `u64`       | 8-bit unsigned, widened to 64-bit  |
| `i16`      | `i64`       | 16-bit signed, widened to 64-bit   |
| `u16`      | `u64`       | 16-bit unsigned, widened to 64-bit |
| `i32`      | `i64`       | 32-bit signed, widened to 64-bit   |
| `u32`      | `u64`       | 32-bit unsigned, widened to 64-bit |
| `i64`      | `i64`       | 64-bit signed                      |
| `u64`      | `u64`       | 64-bit unsigned                    |

Sub-byte reductions:

| Input Type | Output Type | Description                               |
| ---------- | ----------- | ----------------------------------------- |
| `i4`       | `i64`       | 4-bit signed nibbles, widened to 64-bit   |
| `u4`       | `u64`       | 4-bit unsigned nibbles, widened to 64-bit |
| `u1`       | `u64`       | 1-bit binary packed octets                |

## Optimizations

### Strided Access Across Backends

Reductions accept a `stride_bytes` parameter specifying the byte distance between consecutive logical elements -- the only NumKong module where loads far outnumber stores (N loads, 2-4 scalar stores), making arbitrary strides practical.
Serial iterates with byte-pointer arithmetic: `ptr += stride_bytes` per element.
NEON uses hardware de-interleaving loads (`vld2q_f32`, `vld3q_f32`, `vld4q_f32`) for small integer strides (2-4 elements apart), extracting column 0 from interleaved data in a single instruction.
Haswell/Skylake use blend masks for small strides and `_mm256_i32gather_ps` / `_mm512_i32gather_pd` hardware gathers for larger strides -- 8cy per gather on Haswell, ~5cy on Skylake for 16-element gathers.
RVV uses native strided loads (`__riscv_vlse32_v_f32m1`) that accept arbitrary byte strides directly in the load instruction -- no gather overhead, no stride-dependent branching.

### Kahan-Neumaier Compensated Summation

`nk_reduce_moments_f32_serial`, `nk_reduce_moments_f32_haswell` use Neumaier's variant of Kahan summation -- maintaining a running compensation term that captures rounding errors.
Standard pairwise summation accumulates $O(\sqrt{n})$ ULP error for n elements; Neumaier compensation bounds error to $O(1)$ ULP regardless of vector length.
The serial path uses Neumaier's adaptive branch: `if (abs(sum) >= abs(val))` selects the larger summand first, minimizing relative error in the compensation term.
SIMD backends (`nk_reduce_moments_f32_haswell`) carry 8 independent compensation lanes in a YMM register -- computing `round_error = tentative - sum; correction = (sum - (tentative - round_error)) + (val - round_error)` without branches, folding all lanes into a single scalar correction at the end.

### Fused Moments in a Single Pass

`nk_reduce_moments_f32_haswell`, `nk_reduce_moments_f64_skylake` compute sum and sum-of-squares simultaneously -- one load feeds both a `VADDPS` (sum accumulator) and a `VFMADD231PS` (square accumulator).
Two accumulators share the same loaded data, halving memory bandwidth compared to separate sum + norm passes.
The squared-norm $\|a\|^2 = \sum a_i^2$ is a self-dot-product, reused by packing routines (`nk_dots_pack_f32_haswell`) to precompute per-vector norms during layout transformation.
For f16/bf16/FP8 inputs, all backends widen to f32 before accumulation -- NEON FHM (`nk_reduce_moments_e4m3_neonfhm`) converts e4m3->f16 via lookup, then uses `vfmlalq_low_f16` to fuse the f16->f32 widening with the FMA into the f32 accumulator.

### Integer Saturation in Sum-of-Squares

Integer moments accumulate sums in the widest available type: i8/u8/i16 inputs produce i64/u64 outputs.
Sums use widening addition chains -- NEON uses pairwise widening (`vpaddlq_s16` -> u32 -> u64 stages); Haswell biases i8 inputs with 0x80 and uses unsigned SAD (`_mm256_sad_epu8`) for the sum, correcting by subtracting $128 \times \text{count}$ at the end.
Sum-of-squares can overflow u64 when squaring large i32 values -- backends use explicit saturating multiply: checks if `abs(val) < 2^32` (square fits in u64), otherwise saturates to I64_MAX.
Haswell emulates u64 saturating add via XOR-based unsigned comparison: flip sign bits to convert unsigned overflow detection into a signed comparison, then OR with the overflow mask to produce all-ones on saturation.

### Recursive Blocking for Counter Overflow

All SIMD backends use loop iteration counters narrower than `nk_size_t` to save register pressure -- u8 for i8 minmax lanes, u16 for f32 moments lanes.
When `count` exceeds the counter's range x lane count (e.g., Haswell f32: $256 \times 8 = 2048$ elements for u8 counters), the reduction splits recursively: process the left half, process the right half, combine results with saturating arithmetic.
Block caps vary by backend and element width: Haswell i8 minmax uses u8 loop counters (cap = $256 \times 32 = 8192$); Skylake f32 moments uses u16 counters (cap = $65536 \times 16 = 1048576$).
The recursive split is invisible to the caller -- the public API accepts arbitrary `count` values; internal dispatch chooses between single-pass and recursive based on the cap.

### Index Tracking at Different Register Scales

Argmin/argmax requires tracking both values and their positions -- but indices need wider storage than values (u64 for arbitrary-length vectors, vs u8/u16/f32 for data).
Haswell i8 minmax tracks iteration counters in u8 lanes (same width as data) -- after the loop, the winning lane's counter is multiplied by the lane count and added to the lane index within the register to reconstruct the global position.
RVV uses u64m2 registers (LMUL=2) for indices alongside f32m1 for values -- the wider index register holds one 64-bit position per f32 lane, enabling direct merge without post-loop reconstruction.
NEON uses same-width counters (u8x16 for i8x16 minmax), limiting block size to $256 \times 16 = 4096$ elements before recursive splitting.

### NaN-Aware Extrema Tracking

`nk_reduce_minmax_f32_haswell`, `nk_reduce_minmax_f64_skylake` use IEEE ordered-quiet comparisons (`_CMP_LT_OQ`, `_CMP_GT_OQ`) -- returning false when either operand is NaN, so NaN inputs never replace the running extremum.
Tail elements beyond the vector-aligned portion are masked by loading into a NaN-filled register via `_mm256_mask_loadu_ps(nan_vec, mask, ptr)` -- NaN tails cannot win any comparison, eliminating out-of-bounds artifacts.
If all inputs are NaN, the sentinels remain (min = F32_MAX, max = F32_MIN) and indices are set to NK_SIZE_MAX, signaling no valid extremum.
The final horizontal reduction across lanes uses pairwise `VSHUFPS` + `VMINPS` chains -- 3 shuffles for a 256-bit register, $O(\log_2 w)$ for width $w$.

## Performance

Controlled by `NK_DENSE_DIMENSIONS`.
Columns show 256, 1024, 4096 elements.

### Intel Sapphire Rapids

#### Native

| Kernel                           |           256 |          1024 |          4096 |
| :------------------------------- | ------------: | ------------: | ------------: |
| __f64__                          |               |               |               |
| `nk_reduce_moments_f64_serial`   | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_reduce_minmax_f64_serial`    | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_reduce_moments_f64_haswell`  | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_reduce_minmax_f64_haswell`   | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_reduce_moments_f64_skylake`  | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_reduce_minmax_f64_skylake`   | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| __f32__                          |               |               |               |
| `nk_reduce_moments_f32_serial`   | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_reduce_minmax_f32_serial`    | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_reduce_moments_f32_haswell`  | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_reduce_minmax_f32_haswell`   | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_reduce_moments_f32_skylake`  | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_reduce_minmax_f32_skylake`   | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| __bf16__                         |               |               |               |
| `nk_reduce_moments_bf16_serial`  | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_reduce_minmax_bf16_serial`   | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_reduce_moments_bf16_haswell` | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_reduce_minmax_bf16_haswell`  | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_reduce_moments_bf16_skylake` | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_reduce_minmax_bf16_skylake`  | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_reduce_moments_bf16_genoa`   | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| __f16__                          |               |               |               |
| `nk_reduce_moments_f16_serial`   | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_reduce_minmax_f16_serial`    | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_reduce_moments_f16_haswell`  | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_reduce_minmax_f16_haswell`   | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_reduce_moments_f16_skylake`  | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_reduce_minmax_f16_skylake`   | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| __e5m2__                         |               |               |               |
| `nk_reduce_moments_e5m2_serial`  | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_reduce_minmax_e5m2_serial`   | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_reduce_moments_e5m2_haswell` | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_reduce_minmax_e5m2_haswell`  | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_reduce_moments_e5m2_skylake` | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_reduce_minmax_e5m2_skylake`  | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_reduce_moments_e5m2_genoa`   | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| __e4m3__                         |               |               |               |
| `nk_reduce_moments_e4m3_serial`  | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_reduce_minmax_e4m3_serial`   | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_reduce_moments_e4m3_haswell` | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_reduce_minmax_e4m3_haswell`  | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_reduce_moments_e4m3_skylake` | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_reduce_minmax_e4m3_skylake`  | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_reduce_moments_e4m3_genoa`   | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| __e3m2__                         |               |               |               |
| `nk_reduce_moments_e3m2_serial`  | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_reduce_minmax_e3m2_serial`   | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_reduce_moments_e3m2_haswell` | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_reduce_minmax_e3m2_haswell`  | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_reduce_moments_e3m2_skylake` | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_reduce_minmax_e3m2_skylake`  | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_reduce_moments_e3m2_icelake` | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_reduce_moments_e3m2_alder`   | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| __e2m3__                         |               |               |               |
| `nk_reduce_moments_e2m3_serial`  | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_reduce_minmax_e2m3_serial`   | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_reduce_moments_e2m3_haswell` | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_reduce_minmax_e2m3_haswell`  | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_reduce_moments_e2m3_skylake` | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_reduce_minmax_e2m3_skylake`  | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_reduce_moments_e2m3_icelake` | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_reduce_moments_e2m3_sierra`  | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| __i8__                           |               |               |               |
| `nk_reduce_moments_i8_serial`    |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_minmax_i8_serial`     |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_moments_i8_haswell`   |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_minmax_i8_haswell`    |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_moments_i8_skylake`   |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_minmax_i8_skylake`    |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_moments_i8_icelake`   |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_moments_i8_sierra`    |        0 GB/s |        0 GB/s |        0 GB/s |
| __u8__                           |               |               |               |
| `nk_reduce_moments_u8_serial`    |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_minmax_u8_serial`     |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_moments_u8_haswell`   |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_minmax_u8_haswell`    |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_moments_u8_skylake`   |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_minmax_u8_skylake`    |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_moments_u8_icelake`   |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_moments_u8_alder`     |        0 GB/s |        0 GB/s |        0 GB/s |
| __i4__                           |               |               |               |
| `nk_reduce_moments_i4_serial`    |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_minmax_i4_serial`     |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_moments_i4_haswell`   |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_moments_i4_skylake`   |        0 GB/s |        0 GB/s |        0 GB/s |
| __u4__                           |               |               |               |
| `nk_reduce_moments_u4_serial`    |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_minmax_u4_serial`     |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_moments_u4_haswell`   |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_moments_u4_skylake`   |        0 GB/s |        0 GB/s |        0 GB/s |
| __u1__                           |               |               |               |
| `nk_reduce_moments_u1_serial`    |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_minmax_u1_serial`     |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_moments_u1_haswell`   |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_moments_u1_skylake`   |        0 GB/s |        0 GB/s |        0 GB/s |
| __i16__                          |               |               |               |
| `nk_reduce_moments_i16_serial`   |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_minmax_i16_serial`    |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_moments_i16_haswell`  |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_minmax_i16_haswell`   |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_moments_i16_skylake`  |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_minmax_i16_skylake`   |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_moments_i16_icelake`  |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_moments_i16_alder`    |        0 GB/s |        0 GB/s |        0 GB/s |
| __u16__                          |               |               |               |
| `nk_reduce_moments_u16_serial`   |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_minmax_u16_serial`    |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_moments_u16_haswell`  |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_minmax_u16_haswell`   |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_moments_u16_skylake`  |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_minmax_u16_skylake`   |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_moments_u16_alder`    |        0 GB/s |        0 GB/s |        0 GB/s |
| __i32__                          |               |               |               |
| `nk_reduce_moments_i32_serial`   |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_minmax_i32_serial`    |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_moments_i32_haswell`  |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_minmax_i32_haswell`   |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_moments_i32_skylake`  |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_minmax_i32_skylake`   |        0 GB/s |        0 GB/s |        0 GB/s |
| __u32__                          |               |               |               |
| `nk_reduce_moments_u32_serial`   |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_minmax_u32_serial`    |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_moments_u32_haswell`  |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_minmax_u32_haswell`   |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_moments_u32_skylake`  |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_minmax_u32_skylake`   |        0 GB/s |        0 GB/s |        0 GB/s |
| __i64__                          |               |               |               |
| `nk_reduce_moments_i64_serial`   |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_minmax_i64_serial`    |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_moments_i64_haswell`  |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_minmax_i64_haswell`   |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_moments_i64_skylake`  |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_minmax_i64_skylake`   |        0 GB/s |        0 GB/s |        0 GB/s |
| __u64__                          |               |               |               |
| `nk_reduce_moments_u64_serial`   |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_minmax_u64_serial`    |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_moments_u64_haswell`  |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_minmax_u64_haswell`   |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_moments_u64_skylake`  |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_minmax_u64_skylake`   |        0 GB/s |        0 GB/s |        0 GB/s |

#### V8

| Kernel                               |           256 |          1024 |          4096 |
| :----------------------------------- | ------------: | ------------: | ------------: |
| __f64__                              |               |               |               |
| `nk_reduce_moments_f64_v128relaxed`  | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_reduce_minmax_f64_v128relaxed`   | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| __f32__                              |               |               |               |
| `nk_reduce_moments_f32_v128relaxed`  | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_reduce_minmax_f32_v128relaxed`   | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| __bf16__                             |               |               |               |
| `nk_reduce_moments_bf16_v128relaxed` | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_reduce_minmax_bf16_v128relaxed`  | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| __f16__                              |               |               |               |
| `nk_reduce_moments_f16_v128relaxed`  | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_reduce_minmax_f16_v128relaxed`   | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| __e5m2__                             |               |               |               |
| `nk_reduce_moments_e5m2_v128relaxed` | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_reduce_minmax_e5m2_v128relaxed`  | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| __e4m3__                             |               |               |               |
| `nk_reduce_moments_e4m3_v128relaxed` | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_reduce_minmax_e4m3_v128relaxed`  | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| __e3m2__                             |               |               |               |
| `nk_reduce_moments_e3m2_v128relaxed` | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_reduce_minmax_e3m2_v128relaxed`  | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| __e2m3__                             |               |               |               |
| `nk_reduce_moments_e2m3_v128relaxed` | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_reduce_minmax_e2m3_v128relaxed`  | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| __i8__                               |               |               |               |
| `nk_reduce_moments_i8_v128relaxed`   |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_minmax_i8_v128relaxed`    |        0 GB/s |        0 GB/s |        0 GB/s |
| __u8__                               |               |               |               |
| `nk_reduce_moments_u8_v128relaxed`   |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_minmax_u8_v128relaxed`    |        0 GB/s |        0 GB/s |        0 GB/s |
| __i16__                              |               |               |               |
| `nk_reduce_moments_i16_v128relaxed`  |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_minmax_i16_v128relaxed`   |        0 GB/s |        0 GB/s |        0 GB/s |
| __u16__                              |               |               |               |
| `nk_reduce_moments_u16_v128relaxed`  |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_minmax_u16_v128relaxed`   |        0 GB/s |        0 GB/s |        0 GB/s |
| __i32__                              |               |               |               |
| `nk_reduce_moments_i32_v128relaxed`  |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_minmax_i32_v128relaxed`   |        0 GB/s |        0 GB/s |        0 GB/s |
| __u32__                              |               |               |               |
| `nk_reduce_moments_u32_v128relaxed`  |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_minmax_u32_v128relaxed`   |        0 GB/s |        0 GB/s |        0 GB/s |
| __i64__                              |               |               |               |
| `nk_reduce_moments_i64_v128relaxed`  |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_minmax_i64_v128relaxed`   |        0 GB/s |        0 GB/s |        0 GB/s |
| __u64__                              |               |               |               |
| `nk_reduce_moments_u64_v128relaxed`  |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_minmax_u64_v128relaxed`   |        0 GB/s |        0 GB/s |        0 GB/s |

#### Wasmtime

| Kernel                               |           256 |          1024 |          4096 |
| :----------------------------------- | ------------: | ------------: | ------------: |
| __f64__                              |               |               |               |
| `nk_reduce_moments_f64_v128relaxed`  | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_reduce_minmax_f64_v128relaxed`   | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| __f32__                              |               |               |               |
| `nk_reduce_moments_f32_v128relaxed`  | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_reduce_minmax_f32_v128relaxed`   | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| __bf16__                             |               |               |               |
| `nk_reduce_moments_bf16_v128relaxed` | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_reduce_minmax_bf16_v128relaxed`  | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| __f16__                              |               |               |               |
| `nk_reduce_moments_f16_v128relaxed`  | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_reduce_minmax_f16_v128relaxed`   | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| __e5m2__                             |               |               |               |
| `nk_reduce_moments_e5m2_v128relaxed` | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_reduce_minmax_e5m2_v128relaxed`  | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| __e4m3__                             |               |               |               |
| `nk_reduce_moments_e4m3_v128relaxed` | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_reduce_minmax_e4m3_v128relaxed`  | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| __e3m2__                             |               |               |               |
| `nk_reduce_moments_e3m2_v128relaxed` | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_reduce_minmax_e3m2_v128relaxed`  | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| __e2m3__                             |               |               |               |
| `nk_reduce_moments_e2m3_v128relaxed` | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_reduce_minmax_e2m3_v128relaxed`  | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| __i8__                               |               |               |               |
| `nk_reduce_moments_i8_v128relaxed`   |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_minmax_i8_v128relaxed`    |        0 GB/s |        0 GB/s |        0 GB/s |
| __u8__                               |               |               |               |
| `nk_reduce_moments_u8_v128relaxed`   |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_minmax_u8_v128relaxed`    |        0 GB/s |        0 GB/s |        0 GB/s |
| __i16__                              |               |               |               |
| `nk_reduce_moments_i16_v128relaxed`  |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_minmax_i16_v128relaxed`   |        0 GB/s |        0 GB/s |        0 GB/s |
| __u16__                              |               |               |               |
| `nk_reduce_moments_u16_v128relaxed`  |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_minmax_u16_v128relaxed`   |        0 GB/s |        0 GB/s |        0 GB/s |
| __i32__                              |               |               |               |
| `nk_reduce_moments_i32_v128relaxed`  |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_minmax_i32_v128relaxed`   |        0 GB/s |        0 GB/s |        0 GB/s |
| __u32__                              |               |               |               |
| `nk_reduce_moments_u32_v128relaxed`  |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_minmax_u32_v128relaxed`   |        0 GB/s |        0 GB/s |        0 GB/s |
| __i64__                              |               |               |               |
| `nk_reduce_moments_i64_v128relaxed`  |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_minmax_i64_v128relaxed`   |        0 GB/s |        0 GB/s |        0 GB/s |
| __u64__                              |               |               |               |
| `nk_reduce_moments_u64_v128relaxed`  |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_minmax_u64_v128relaxed`   |        0 GB/s |        0 GB/s |        0 GB/s |

### Apple M4 Pro

#### Native

| Kernel                             |           256 |          1024 |          4096 |
| :--------------------------------- | ------------: | ------------: | ------------: |
| __f64__                            |               |               |               |
| `nk_reduce_moments_f64_serial`     | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_reduce_minmax_f64_serial`      | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_reduce_moments_f64_neon`       | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_reduce_minmax_f64_neon`        | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| __f32__                            |               |               |               |
| `nk_reduce_moments_f32_serial`     | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_reduce_minmax_f32_serial`      | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_reduce_moments_f32_neon`       | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_reduce_minmax_f32_neon`        | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| __bf16__                           |               |               |               |
| `nk_reduce_moments_bf16_serial`    | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_reduce_minmax_bf16_serial`     | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_reduce_moments_bf16_neonbfdot` | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_reduce_minmax_bf16_neonbfdot`  | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| __f16__                            |               |               |               |
| `nk_reduce_moments_f16_serial`     | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_reduce_minmax_f16_serial`      | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_reduce_moments_f16_neonhalf`   | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_reduce_minmax_f16_neonhalf`    | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| __e5m2__                           |               |               |               |
| `nk_reduce_moments_e5m2_serial`    | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_reduce_minmax_e5m2_serial`     | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_reduce_moments_e5m2_neonfhm`   | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_reduce_minmax_e5m2_neonfhm`    | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| __e4m3__                           |               |               |               |
| `nk_reduce_moments_e4m3_serial`    | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_reduce_minmax_e4m3_serial`     | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_reduce_moments_e4m3_neonfhm`   | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_reduce_minmax_e4m3_neonfhm`    | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| __e3m2__                           |               |               |               |
| `nk_reduce_moments_e3m2_serial`    | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_reduce_minmax_e3m2_serial`     | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| __e2m3__                           |               |               |               |
| `nk_reduce_moments_e2m3_serial`    | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_reduce_minmax_e2m3_serial`     | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_reduce_moments_e2m3_neonsdot`  | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| __i8__                             |               |               |               |
| `nk_reduce_moments_i8_serial`      |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_minmax_i8_serial`       |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_moments_i8_neon`        |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_minmax_i8_neon`         |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_moments_i8_neonsdot`    |        0 GB/s |        0 GB/s |        0 GB/s |
| __u8__                             |               |               |               |
| `nk_reduce_moments_u8_serial`      |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_minmax_u8_serial`       |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_moments_u8_neon`        |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_minmax_u8_neon`         |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_moments_u8_neonsdot`    |        0 GB/s |        0 GB/s |        0 GB/s |
| __i4__                             |               |               |               |
| `nk_reduce_moments_i4_serial`      |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_minmax_i4_serial`       |        0 GB/s |        0 GB/s |        0 GB/s |
| __u4__                             |               |               |               |
| `nk_reduce_moments_u4_serial`      |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_minmax_u4_serial`       |        0 GB/s |        0 GB/s |        0 GB/s |
| __u1__                             |               |               |               |
| `nk_reduce_moments_u1_serial`      |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_minmax_u1_serial`       |        0 GB/s |        0 GB/s |        0 GB/s |
| __i16__                            |               |               |               |
| `nk_reduce_moments_i16_serial`     |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_minmax_i16_serial`      |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_moments_i16_neon`       |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_minmax_i16_neon`        |        0 GB/s |        0 GB/s |        0 GB/s |
| __u16__                            |               |               |               |
| `nk_reduce_moments_u16_serial`     |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_minmax_u16_serial`      |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_moments_u16_neon`       |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_minmax_u16_neon`        |        0 GB/s |        0 GB/s |        0 GB/s |
| __i32__                            |               |               |               |
| `nk_reduce_moments_i32_serial`     |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_minmax_i32_serial`      |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_moments_i32_neon`       |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_minmax_i32_neon`        |        0 GB/s |        0 GB/s |        0 GB/s |
| __u32__                            |               |               |               |
| `nk_reduce_moments_u32_serial`     |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_minmax_u32_serial`      |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_moments_u32_neon`       |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_minmax_u32_neon`        |        0 GB/s |        0 GB/s |        0 GB/s |
| __i64__                            |               |               |               |
| `nk_reduce_moments_i64_serial`     |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_minmax_i64_serial`      |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_moments_i64_neon`       |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_minmax_i64_neon`        |        0 GB/s |        0 GB/s |        0 GB/s |
| __u64__                            |               |               |               |
| `nk_reduce_moments_u64_serial`     |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_minmax_u64_serial`      |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_moments_u64_neon`       |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_minmax_u64_neon`        |        0 GB/s |        0 GB/s |        0 GB/s |

#### V8

| Kernel                               |           256 |          1024 |          4096 |
| :----------------------------------- | ------------: | ------------: | ------------: |
| __f64__                              |               |               |               |
| `nk_reduce_moments_f64_v128relaxed`  | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_reduce_minmax_f64_v128relaxed`   | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| __f32__                              |               |               |               |
| `nk_reduce_moments_f32_v128relaxed`  | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_reduce_minmax_f32_v128relaxed`   | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| __bf16__                             |               |               |               |
| `nk_reduce_moments_bf16_v128relaxed` | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_reduce_minmax_bf16_v128relaxed`  | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| __f16__                              |               |               |               |
| `nk_reduce_moments_f16_v128relaxed`  | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_reduce_minmax_f16_v128relaxed`   | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| __e5m2__                             |               |               |               |
| `nk_reduce_moments_e5m2_v128relaxed` | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_reduce_minmax_e5m2_v128relaxed`  | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| __e4m3__                             |               |               |               |
| `nk_reduce_moments_e4m3_v128relaxed` | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_reduce_minmax_e4m3_v128relaxed`  | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| __e3m2__                             |               |               |               |
| `nk_reduce_moments_e3m2_v128relaxed` | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_reduce_minmax_e3m2_v128relaxed`  | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| __e2m3__                             |               |               |               |
| `nk_reduce_moments_e2m3_v128relaxed` | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_reduce_minmax_e2m3_v128relaxed`  | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| __i8__                               |               |               |               |
| `nk_reduce_moments_i8_v128relaxed`   |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_minmax_i8_v128relaxed`    |        0 GB/s |        0 GB/s |        0 GB/s |
| __u8__                               |               |               |               |
| `nk_reduce_moments_u8_v128relaxed`   |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_minmax_u8_v128relaxed`    |        0 GB/s |        0 GB/s |        0 GB/s |
| __i16__                              |               |               |               |
| `nk_reduce_moments_i16_v128relaxed`  |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_minmax_i16_v128relaxed`   |        0 GB/s |        0 GB/s |        0 GB/s |
| __u16__                              |               |               |               |
| `nk_reduce_moments_u16_v128relaxed`  |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_minmax_u16_v128relaxed`   |        0 GB/s |        0 GB/s |        0 GB/s |
| __i32__                              |               |               |               |
| `nk_reduce_moments_i32_v128relaxed`  |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_minmax_i32_v128relaxed`   |        0 GB/s |        0 GB/s |        0 GB/s |
| __u32__                              |               |               |               |
| `nk_reduce_moments_u32_v128relaxed`  |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_minmax_u32_v128relaxed`   |        0 GB/s |        0 GB/s |        0 GB/s |
| __i64__                              |               |               |               |
| `nk_reduce_moments_i64_v128relaxed`  |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_minmax_i64_v128relaxed`   |        0 GB/s |        0 GB/s |        0 GB/s |
| __u64__                              |               |               |               |
| `nk_reduce_moments_u64_v128relaxed`  |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_minmax_u64_v128relaxed`   |        0 GB/s |        0 GB/s |        0 GB/s |

#### Wasmtime

| Kernel                               |           256 |          1024 |          4096 |
| :----------------------------------- | ------------: | ------------: | ------------: |
| __f64__                              |               |               |               |
| `nk_reduce_moments_f64_v128relaxed`  | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_reduce_minmax_f64_v128relaxed`   | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| __f32__                              |               |               |               |
| `nk_reduce_moments_f32_v128relaxed`  | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_reduce_minmax_f32_v128relaxed`   | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| __bf16__                             |               |               |               |
| `nk_reduce_moments_bf16_v128relaxed` | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_reduce_minmax_bf16_v128relaxed`  | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| __f16__                              |               |               |               |
| `nk_reduce_moments_f16_v128relaxed`  | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_reduce_minmax_f16_v128relaxed`   | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| __e5m2__                             |               |               |               |
| `nk_reduce_moments_e5m2_v128relaxed` | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_reduce_minmax_e5m2_v128relaxed`  | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| __e4m3__                             |               |               |               |
| `nk_reduce_moments_e4m3_v128relaxed` | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_reduce_minmax_e4m3_v128relaxed`  | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| __e3m2__                             |               |               |               |
| `nk_reduce_moments_e3m2_v128relaxed` | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_reduce_minmax_e3m2_v128relaxed`  | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| __e2m3__                             |               |               |               |
| `nk_reduce_moments_e2m3_v128relaxed` | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| `nk_reduce_minmax_e2m3_v128relaxed`  | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP | 0 GB/s, 0 ULP |
| __i8__                               |               |               |               |
| `nk_reduce_moments_i8_v128relaxed`   |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_minmax_i8_v128relaxed`    |        0 GB/s |        0 GB/s |        0 GB/s |
| __u8__                               |               |               |               |
| `nk_reduce_moments_u8_v128relaxed`   |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_minmax_u8_v128relaxed`    |        0 GB/s |        0 GB/s |        0 GB/s |
| __i16__                              |               |               |               |
| `nk_reduce_moments_i16_v128relaxed`  |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_minmax_i16_v128relaxed`   |        0 GB/s |        0 GB/s |        0 GB/s |
| __u16__                              |               |               |               |
| `nk_reduce_moments_u16_v128relaxed`  |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_minmax_u16_v128relaxed`   |        0 GB/s |        0 GB/s |        0 GB/s |
| __i32__                              |               |               |               |
| `nk_reduce_moments_i32_v128relaxed`  |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_minmax_i32_v128relaxed`   |        0 GB/s |        0 GB/s |        0 GB/s |
| __u32__                              |               |               |               |
| `nk_reduce_moments_u32_v128relaxed`  |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_minmax_u32_v128relaxed`   |        0 GB/s |        0 GB/s |        0 GB/s |
| __i64__                              |               |               |               |
| `nk_reduce_moments_i64_v128relaxed`  |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_minmax_i64_v128relaxed`   |        0 GB/s |        0 GB/s |        0 GB/s |
| __u64__                              |               |               |               |
| `nk_reduce_moments_u64_v128relaxed`  |        0 GB/s |        0 GB/s |        0 GB/s |
| `nk_reduce_minmax_u64_v128relaxed`   |        0 GB/s |        0 GB/s |        0 GB/s |

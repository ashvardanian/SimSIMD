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

The following performance tables are produced by manually re-running `nk_test` and `nk_bench` included internal tools to measure both accuracy and throughput at different input shapes.
The input size is controlled by the `NK_DENSE_DIMENSIONS` environment variable and set to 256, 1024, and 4096 elements.
The throughput is measured in GB/s as the number of input bytes per second.
Accuracy is reported as ULP (units in last place), the number of representable floating-point values between the result and the exact answer.
Each kernel runs for at least 20 seconds per configuration.
Benchmark threads are pinned to specific cores; on machines with heterogeneous core types (e.g., Apple P/E cores), only the fastest cores are used.
Workloads that significantly degrade CPU frequencies (Intel AMX, Apple SME) run in separate passes to avoid affecting throughput measurements of other kernels.

### Intel Sapphire Rapids

#### Native

| Kernel                           |                      256 |                     1024 |                     4096 |
| :------------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f64__                          | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_f64_serial`   |         2.57 gb/s, 0 ulp |         2.72 gb/s, 0 ulp |         1.70 gb/s, 0 ulp |
| `nk_reduce_minmax_f64_serial`    |         6.71 gb/s, 0 ulp |         6.28 gb/s, 0 ulp |         4.73 gb/s, 0 ulp |
| `nk_reduce_moments_f64_haswell`  |       16.5 gb/s, 0.2 ulp |       15.6 gb/s, 0.3 ulp |       9.07 gb/s, 0.2 ulp |
| `nk_reduce_minmax_f64_haswell`   |         10.7 gb/s, 0 ulp |         10.4 gb/s, 0 ulp |         7.93 gb/s, 0 ulp |
| `nk_reduce_moments_f64_skylake`  |       22.7 gb/s, 0.2 ulp |       21.3 gb/s, 0.6 ulp |       14.2 gb/s, 0.4 ulp |
| `nk_reduce_minmax_f64_skylake`   |         32.4 gb/s, 0 ulp |         26.0 gb/s, 0 ulp |         19.5 gb/s, 0 ulp |
| __f32__                          | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_f32_serial`   |        0.630 gb/s, 0 ulp |        0.591 gb/s, 0 ulp |        0.469 gb/s, 0 ulp |
| `nk_reduce_minmax_f32_serial`    |         3.29 gb/s, 0 ulp |         3.11 gb/s, 0 ulp |         2.46 gb/s, 0 ulp |
| `nk_reduce_moments_f32_haswell`  |       30.2 gb/s, 0.8 ulp |       27.5 gb/s, 3.9 ulp |       20.0 gb/s, 4.7 ulp |
| `nk_reduce_minmax_f32_haswell`   |         10.3 gb/s, 0 ulp |         10.7 gb/s, 0 ulp |         10.2 gb/s, 0 ulp |
| `nk_reduce_moments_f32_skylake`  |       37.8 gb/s, 0.5 ulp |       30.8 gb/s, 3.0 ulp |       20.1 gb/s, 6.2 ulp |
| `nk_reduce_minmax_f32_skylake`   |         31.5 gb/s, 0 ulp |         29.3 gb/s, 0 ulp |         18.1 gb/s, 0 ulp |
| __bf16__                         | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_bf16_serial`  |        0.390 gb/s, 0 ulp |        0.386 gb/s, 0 ulp |        0.227 gb/s, 0 ulp |
| `nk_reduce_minmax_bf16_serial`   |        0.906 gb/s, 0 ulp |        0.961 gb/s, 0 ulp |        0.567 gb/s, 0 ulp |
| `nk_reduce_moments_bf16_haswell` |         13.6 gb/s, 0 ulp |         14.6 gb/s, 0 ulp |       10.5 gb/s, 0.4 ulp |
| `nk_reduce_minmax_bf16_haswell`  |         7.39 gb/s, 0 ulp |         10.2 gb/s, 0 ulp |         7.55 gb/s, 0 ulp |
| `nk_reduce_moments_bf16_skylake` |         29.7 gb/s, 0 ulp |         26.8 gb/s, 0 ulp |       16.1 gb/s, 0.2 ulp |
| `nk_reduce_minmax_bf16_skylake`  |         11.7 gb/s, 0 ulp |         19.5 gb/s, 0 ulp |         15.5 gb/s, 0 ulp |
| `nk_reduce_moments_bf16_genoa`   |         30.6 gb/s, 0 ulp |         27.8 gb/s, 0 ulp |       17.1 gb/s, 0.2 ulp |
| __f16__                          | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_f16_serial`   |        0.540 gb/s, 0 ulp |        0.501 gb/s, 0 ulp |        0.367 gb/s, 0 ulp |
| `nk_reduce_minmax_f16_serial`    |        0.921 gb/s, 0 ulp |        0.956 gb/s, 0 ulp |        0.674 gb/s, 0 ulp |
| `nk_reduce_moments_f16_haswell`  |         15.1 gb/s, 0 ulp |         16.4 gb/s, 0 ulp |       11.6 gb/s, 0.1 ulp |
| `nk_reduce_minmax_f16_haswell`   |         9.20 gb/s, 0 ulp |         11.3 gb/s, 0 ulp |         8.52 gb/s, 0 ulp |
| `nk_reduce_moments_f16_skylake`  |         30.7 gb/s, 0 ulp |         30.1 gb/s, 0 ulp |         17.8 gb/s, 0 ulp |
| `nk_reduce_minmax_f16_skylake`   |         17.5 gb/s, 0 ulp |         21.8 gb/s, 0 ulp |         17.9 gb/s, 0 ulp |
| __e5m2__                         | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_e5m2_serial`  |        0.261 gb/s, 0 ulp |        0.274 gb/s, 0 ulp |        0.159 gb/s, 0 ulp |
| `nk_reduce_minmax_e5m2_serial`   |        0.439 gb/s, 0 ulp |        0.428 gb/s, 0 ulp |        0.303 gb/s, 0 ulp |
| `nk_reduce_moments_e5m2_haswell` |         3.75 gb/s, 0 ulp |         4.41 gb/s, 0 ulp |         4.37 gb/s, 0 ulp |
| `nk_reduce_minmax_e5m2_haswell`  |         7.40 gb/s, 0 ulp |         9.87 gb/s, 0 ulp |         7.00 gb/s, 0 ulp |
| `nk_reduce_moments_e5m2_skylake` |         5.56 gb/s, 0 ulp |         5.60 gb/s, 0 ulp |         3.87 gb/s, 0 ulp |
| `nk_reduce_minmax_e5m2_skylake`  |         9.48 gb/s, 0 ulp |         16.8 gb/s, 0 ulp |         15.4 gb/s, 0 ulp |
| `nk_reduce_moments_e5m2_genoa`   |         7.83 gb/s, 0 ulp |         8.58 gb/s, 0 ulp |         6.63 gb/s, 0 ulp |
| __e4m3__                         | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_e4m3_serial`  |        0.211 gb/s, 0 ulp |        0.212 gb/s, 0 ulp |        0.149 gb/s, 0 ulp |
| `nk_reduce_minmax_e4m3_serial`   |        0.462 gb/s, 0 ulp |        0.447 gb/s, 0 ulp |        0.273 gb/s, 0 ulp |
| `nk_reduce_moments_e4m3_haswell` |         2.86 gb/s, 0 ulp |         3.11 gb/s, 0 ulp |         2.92 gb/s, 0 ulp |
| `nk_reduce_minmax_e4m3_haswell`  |         8.03 gb/s, 0 ulp |         11.6 gb/s, 0 ulp |         6.14 gb/s, 0 ulp |
| `nk_reduce_moments_e4m3_skylake` |         4.50 gb/s, 0 ulp |         4.58 gb/s, 0 ulp |         3.37 gb/s, 0 ulp |
| `nk_reduce_minmax_e4m3_skylake`  |         9.25 gb/s, 0 ulp |         16.6 gb/s, 0 ulp |         14.1 gb/s, 0 ulp |
| `nk_reduce_moments_e4m3_genoa`   |         7.58 gb/s, 0 ulp |         8.47 gb/s, 0 ulp |         7.08 gb/s, 0 ulp |
| __e3m2__                         | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_e3m2_serial`  |        0.219 gb/s, 0 ulp |        0.259 gb/s, 0 ulp |        0.138 gb/s, 0 ulp |
| `nk_reduce_minmax_e3m2_serial`   |        0.473 gb/s, 0 ulp |        0.452 gb/s, 0 ulp |        0.441 gb/s, 0 ulp |
| `nk_reduce_moments_e3m2_haswell` |         3.98 gb/s, 0 ulp |         4.47 gb/s, 0 ulp |         4.28 gb/s, 0 ulp |
| `nk_reduce_minmax_e3m2_haswell`  |         9.45 gb/s, 0 ulp |         15.1 gb/s, 0 ulp |         7.25 gb/s, 0 ulp |
| `nk_reduce_moments_e3m2_skylake` |         5.47 gb/s, 0 ulp |         5.60 gb/s, 0 ulp |         4.16 gb/s, 0 ulp |
| `nk_reduce_minmax_e3m2_skylake`  |         11.2 gb/s, 0 ulp |         20.8 gb/s, 0 ulp |         18.7 gb/s, 0 ulp |
| `nk_reduce_moments_e3m2_icelake` |         14.2 gb/s, 0 ulp |         15.0 gb/s, 0 ulp |         9.90 gb/s, 0 ulp |
| `nk_reduce_moments_e3m2_alder`   |         9.57 gb/s, 0 ulp |         11.3 gb/s, 0 ulp |         10.7 gb/s, 0 ulp |
| __e2m3__                         | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_e2m3_serial`  |        0.201 gb/s, 0 ulp |        0.208 gb/s, 0 ulp |        0.132 gb/s, 0 ulp |
| `nk_reduce_minmax_e2m3_serial`   |        0.467 gb/s, 0 ulp |        0.441 gb/s, 0 ulp |        0.376 gb/s, 0 ulp |
| `nk_reduce_moments_e2m3_haswell` |         4.33 gb/s, 0 ulp |         4.28 gb/s, 0 ulp |         4.29 gb/s, 0 ulp |
| `nk_reduce_minmax_e2m3_haswell`  |         9.55 gb/s, 0 ulp |         14.9 gb/s, 0 ulp |         9.11 gb/s, 0 ulp |
| `nk_reduce_moments_e2m3_skylake` |         5.41 gb/s, 0 ulp |         5.64 gb/s, 0 ulp |         4.51 gb/s, 0 ulp |
| `nk_reduce_minmax_e2m3_skylake`  |         10.7 gb/s, 0 ulp |         20.7 gb/s, 0 ulp |         17.4 gb/s, 0 ulp |
| `nk_reduce_moments_e2m3_icelake` |         26.7 gb/s, 0 ulp |         29.4 gb/s, 0 ulp |         24.9 gb/s, 0 ulp |
| __i8__                           | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_i8_serial`    |                2.73 gb/s |                2.66 gb/s |                2.54 gb/s |
| `nk_reduce_minmax_i8_serial`     |               0.797 gb/s |               0.964 gb/s |               0.685 gb/s |
| `nk_reduce_moments_i8_haswell`   |                20.3 gb/s |                23.9 gb/s |                18.2 gb/s |
| `nk_reduce_minmax_i8_haswell`    |                8.87 gb/s |                15.5 gb/s |                15.7 gb/s |
| `nk_reduce_moments_i8_skylake`   |                23.6 gb/s |                32.2 gb/s |                19.5 gb/s |
| `nk_reduce_minmax_i8_skylake`    |                10.5 gb/s |                26.1 gb/s |                18.0 gb/s |
| `nk_reduce_moments_i8_icelake`   |                27.1 gb/s |                39.2 gb/s |                19.2 gb/s |
| __u8__                           | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_u8_serial`    |                2.78 gb/s |                2.63 gb/s |                2.34 gb/s |
| `nk_reduce_minmax_u8_serial`     |               0.827 gb/s |               0.944 gb/s |               0.729 gb/s |
| `nk_reduce_moments_u8_haswell`   |                21.4 gb/s |                24.6 gb/s |                22.7 gb/s |
| `nk_reduce_minmax_u8_haswell`    |                10.5 gb/s |                14.8 gb/s |                16.1 gb/s |
| `nk_reduce_moments_u8_skylake`   |                23.2 gb/s |                34.9 gb/s |                23.9 gb/s |
| `nk_reduce_minmax_u8_skylake`    |                11.4 gb/s |                21.4 gb/s |                16.3 gb/s |
| `nk_reduce_moments_u8_icelake`   |                29.6 gb/s |                42.4 gb/s |                23.9 gb/s |
| `nk_reduce_moments_u8_alder`     |                19.5 gb/s |                24.5 gb/s |                23.0 gb/s |
| __i4__                           | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_i4_serial`    |               0.796 gb/s |               0.705 gb/s |               0.421 gb/s |
| `nk_reduce_minmax_i4_serial`     |               0.314 gb/s |               0.342 gb/s |               0.331 gb/s |
| `nk_reduce_moments_i4_haswell`   |                8.67 gb/s |                16.9 gb/s |                18.1 gb/s |
| `nk_reduce_moments_i4_skylake`   |                17.0 gb/s |                22.8 gb/s |                13.8 gb/s |
| __u4__                           | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_u4_serial`    |                1.04 gb/s |               0.784 gb/s |               0.622 gb/s |
| `nk_reduce_minmax_u4_serial`     |               0.388 gb/s |               0.368 gb/s |               0.357 gb/s |
| `nk_reduce_moments_u4_haswell`   |                9.04 gb/s |                19.2 gb/s |                21.1 gb/s |
| `nk_reduce_moments_u4_skylake`   |                18.3 gb/s |                25.7 gb/s |                19.1 gb/s |
| __u1__                           | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_u1_serial`    |                2.12 gb/s |                1.92 gb/s |                1.38 gb/s |
| `nk_reduce_minmax_u1_serial`     |                6.06 gb/s |                21.3 gb/s |                80.8 gb/s |
| `nk_reduce_moments_u1_haswell`   |                5.03 gb/s |                14.1 gb/s |                20.5 gb/s |
| `nk_reduce_moments_u1_skylake`   |                8.91 gb/s |                24.1 gb/s |                21.7 gb/s |
| __i16__                          | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_i16_serial`   |                3.31 gb/s |                3.82 gb/s |                2.67 gb/s |
| `nk_reduce_minmax_i16_serial`    |                1.58 gb/s |                1.85 gb/s |                1.17 gb/s |
| `nk_reduce_moments_i16_haswell`  |                24.6 gb/s |                27.2 gb/s |                23.3 gb/s |
| `nk_reduce_minmax_i16_haswell`   |                13.7 gb/s |                13.0 gb/s |                12.7 gb/s |
| `nk_reduce_moments_i16_skylake`  |                37.2 gb/s |                34.5 gb/s |                22.9 gb/s |
| `nk_reduce_minmax_i16_skylake`   |                22.4 gb/s |                27.0 gb/s |                15.9 gb/s |
| `nk_reduce_moments_i16_icelake`  |                39.3 gb/s |                35.0 gb/s |                19.2 gb/s |
| `nk_reduce_moments_i16_alder`    |                16.0 gb/s |                16.9 gb/s |                16.4 gb/s |
| __u16__                          | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_u16_serial`   |                4.23 gb/s |                3.89 gb/s |                2.60 gb/s |
| `nk_reduce_minmax_u16_serial`    |                1.66 gb/s |                1.44 gb/s |                1.16 gb/s |
| `nk_reduce_moments_u16_haswell`  |                10.3 gb/s |                12.0 gb/s |                11.1 gb/s |
| `nk_reduce_minmax_u16_haswell`   |                13.4 gb/s |                13.3 gb/s |                10.3 gb/s |
| `nk_reduce_moments_u16_skylake`  |                19.9 gb/s |                20.0 gb/s |                11.3 gb/s |
| `nk_reduce_minmax_u16_skylake`   |                18.0 gb/s |                25.0 gb/s |                21.6 gb/s |
| `nk_reduce_moments_u16_alder`    |                11.9 gb/s |                12.1 gb/s |                11.8 gb/s |
| __i32__                          | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_i32_serial`   |                4.17 gb/s |                4.06 gb/s |                2.63 gb/s |
| `nk_reduce_minmax_i32_serial`    |                3.04 gb/s |                3.71 gb/s |                2.48 gb/s |
| `nk_reduce_moments_i32_haswell`  |                8.70 gb/s |                10.1 gb/s |                9.31 gb/s |
| `nk_reduce_minmax_i32_haswell`   |                13.9 gb/s |                13.5 gb/s |                10.5 gb/s |
| `nk_reduce_moments_i32_skylake`  |                15.6 gb/s |                15.9 gb/s |                9.77 gb/s |
| `nk_reduce_minmax_i32_skylake`   |                32.2 gb/s |                28.3 gb/s |                17.3 gb/s |
| __u32__                          | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_u32_serial`   |                4.67 gb/s |                5.26 gb/s |                3.83 gb/s |
| `nk_reduce_minmax_u32_serial`    |                3.15 gb/s |                3.54 gb/s |                2.89 gb/s |
| `nk_reduce_moments_u32_haswell`  |                8.67 gb/s |                7.43 gb/s |                7.17 gb/s |
| `nk_reduce_minmax_u32_haswell`   |                13.0 gb/s |                13.6 gb/s |                9.97 gb/s |
| `nk_reduce_moments_u32_skylake`  |                25.1 gb/s |                22.4 gb/s |                15.1 gb/s |
| `nk_reduce_minmax_u32_skylake`   |                31.7 gb/s |                28.3 gb/s |                18.5 gb/s |
| __i64__                          | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_i64_serial`   |                4.68 gb/s |                4.35 gb/s |                2.85 gb/s |
| `nk_reduce_minmax_i64_serial`    |                6.67 gb/s |                5.60 gb/s |                4.56 gb/s |
| `nk_reduce_moments_i64_haswell`  |                13.0 gb/s |                12.5 gb/s |                11.6 gb/s |
| `nk_reduce_minmax_i64_haswell`   |                11.7 gb/s |                11.1 gb/s |                8.49 gb/s |
| `nk_reduce_moments_i64_skylake`  |                24.3 gb/s |                21.9 gb/s |                14.9 gb/s |
| `nk_reduce_minmax_i64_skylake`   |                33.1 gb/s |                26.3 gb/s |                17.9 gb/s |
| __u64__                          | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_u64_serial`   |                2.75 gb/s |                2.56 gb/s |                1.85 gb/s |
| `nk_reduce_minmax_u64_serial`    |                5.42 gb/s |                7.14 gb/s |                4.98 gb/s |
| `nk_reduce_moments_u64_haswell`  |                14.1 gb/s |                13.2 gb/s |                12.2 gb/s |
| `nk_reduce_minmax_u64_haswell`   |                11.5 gb/s |                11.2 gb/s |                8.33 gb/s |
| `nk_reduce_moments_u64_skylake`  |                28.7 gb/s |                23.9 gb/s |                14.5 gb/s |
| `nk_reduce_minmax_u64_skylake`   |                32.8 gb/s |                25.7 gb/s |                20.1 gb/s |

#### WASM

Measured with Wasmtime v42 (Cranelift backend).

| Kernel                               |                      256 |                     1024 |                     4096 |
| :----------------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f64__                              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_f64_v128relaxed`  |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_reduce_minmax_f64_v128relaxed`   |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| __f32__                              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_f32_v128relaxed`  |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_reduce_minmax_f32_v128relaxed`   |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| __bf16__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_bf16_v128relaxed` |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_reduce_minmax_bf16_v128relaxed`  |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| __f16__                              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_f16_v128relaxed`  |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_reduce_minmax_f16_v128relaxed`   |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| __e5m2__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_e5m2_v128relaxed` |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_reduce_minmax_e5m2_v128relaxed`  |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| __e4m3__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_e4m3_v128relaxed` |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_reduce_minmax_e4m3_v128relaxed`  |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| __e3m2__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_e3m2_v128relaxed` |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_reduce_minmax_e3m2_v128relaxed`  |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| __e2m3__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_e2m3_v128relaxed` |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_reduce_minmax_e2m3_v128relaxed`  |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| __i8__                               | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_i8_v128relaxed`   |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_reduce_minmax_i8_v128relaxed`    |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| __u8__                               | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_u8_v128relaxed`   |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_reduce_minmax_u8_v128relaxed`    |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| __i16__                              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_i16_v128relaxed`  |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_reduce_minmax_i16_v128relaxed`   |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| __u16__                              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_u16_v128relaxed`  |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_reduce_minmax_u16_v128relaxed`   |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| __i32__                              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_i32_v128relaxed`  |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_reduce_minmax_i32_v128relaxed`   |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| __u32__                              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_u32_v128relaxed`  |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_reduce_minmax_u32_v128relaxed`   |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| __i64__                              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_i64_v128relaxed`  |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_reduce_minmax_i64_v128relaxed`   |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| __u64__                              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_u64_v128relaxed`  |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_reduce_minmax_u64_v128relaxed`   |                   0 gb/s |                   0 gb/s |                   0 gb/s |

### Apple M4 Pro

#### Native

| Kernel                             |                      256 |                     1024 |                     4096 |
| :--------------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f64__                            | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_f64_serial`     |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_reduce_minmax_f64_serial`      |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_reduce_moments_f64_neon`       |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_reduce_minmax_f64_neon`        |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| __f32__                            | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_f32_serial`     |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_reduce_minmax_f32_serial`      |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_reduce_moments_f32_neon`       |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_reduce_minmax_f32_neon`        |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| __bf16__                           | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_bf16_serial`    |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_reduce_minmax_bf16_serial`     |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_reduce_moments_bf16_neonbfdot` |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_reduce_minmax_bf16_neonbfdot`  |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| __f16__                            | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_f16_serial`     |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_reduce_minmax_f16_serial`      |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_reduce_moments_f16_neonhalf`   |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_reduce_minmax_f16_neonhalf`    |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| __e5m2__                           | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_e5m2_serial`    |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_reduce_minmax_e5m2_serial`     |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_reduce_moments_e5m2_neonfhm`   |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_reduce_minmax_e5m2_neonfhm`    |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| __e4m3__                           | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_e4m3_serial`    |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_reduce_minmax_e4m3_serial`     |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_reduce_moments_e4m3_neonfhm`   |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_reduce_minmax_e4m3_neonfhm`    |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| __e3m2__                           | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_e3m2_serial`    |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_reduce_minmax_e3m2_serial`     |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| __e2m3__                           | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_e2m3_serial`    |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_reduce_minmax_e2m3_serial`     |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_reduce_moments_e2m3_neonsdot`  |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| __i8__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_i8_serial`      |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_reduce_minmax_i8_serial`       |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_reduce_moments_i8_neon`        |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_reduce_minmax_i8_neon`         |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_reduce_moments_i8_neonsdot`    |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| __u8__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_u8_serial`      |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_reduce_minmax_u8_serial`       |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_reduce_moments_u8_neon`        |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_reduce_minmax_u8_neon`         |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_reduce_moments_u8_neonsdot`    |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| __i4__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_i4_serial`      |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_reduce_minmax_i4_serial`       |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| __u4__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_u4_serial`      |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_reduce_minmax_u4_serial`       |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| __u1__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_u1_serial`      |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_reduce_minmax_u1_serial`       |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| __i16__                            | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_i16_serial`     |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_reduce_minmax_i16_serial`      |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_reduce_moments_i16_neon`       |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_reduce_minmax_i16_neon`        |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| __u16__                            | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_u16_serial`     |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_reduce_minmax_u16_serial`      |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_reduce_moments_u16_neon`       |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_reduce_minmax_u16_neon`        |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| __i32__                            | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_i32_serial`     |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_reduce_minmax_i32_serial`      |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_reduce_moments_i32_neon`       |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_reduce_minmax_i32_neon`        |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| __u32__                            | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_u32_serial`     |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_reduce_minmax_u32_serial`      |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_reduce_moments_u32_neon`       |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_reduce_minmax_u32_neon`        |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| __i64__                            | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_i64_serial`     |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_reduce_minmax_i64_serial`      |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_reduce_moments_i64_neon`       |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_reduce_minmax_i64_neon`        |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| __u64__                            | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_u64_serial`     |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_reduce_minmax_u64_serial`      |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_reduce_moments_u64_neon`       |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_reduce_minmax_u64_neon`        |                   0 gb/s |                   0 gb/s |                   0 gb/s |

#### WASM

Measured with Wasmtime v42 (Cranelift backend).

| Kernel                               |                      256 |                     1024 |                     4096 |
| :----------------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f64__                              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_f64_v128relaxed`  |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_reduce_minmax_f64_v128relaxed`   |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| __f32__                              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_f32_v128relaxed`  |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_reduce_minmax_f32_v128relaxed`   |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| __bf16__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_bf16_v128relaxed` |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_reduce_minmax_bf16_v128relaxed`  |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| __f16__                              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_f16_v128relaxed`  |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_reduce_minmax_f16_v128relaxed`   |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| __e5m2__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_e5m2_v128relaxed` |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_reduce_minmax_e5m2_v128relaxed`  |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| __e4m3__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_e4m3_v128relaxed` |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_reduce_minmax_e4m3_v128relaxed`  |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| __e3m2__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_e3m2_v128relaxed` |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_reduce_minmax_e3m2_v128relaxed`  |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| __e2m3__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_e2m3_v128relaxed` |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_reduce_minmax_e2m3_v128relaxed`  |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| __i8__                               | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_i8_v128relaxed`   |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_reduce_minmax_i8_v128relaxed`    |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| __u8__                               | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_u8_v128relaxed`   |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_reduce_minmax_u8_v128relaxed`    |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| __i16__                              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_i16_v128relaxed`  |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_reduce_minmax_i16_v128relaxed`   |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| __u16__                              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_u16_v128relaxed`  |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_reduce_minmax_u16_v128relaxed`   |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| __i32__                              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_i32_v128relaxed`  |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_reduce_minmax_i32_v128relaxed`   |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| __u32__                              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_u32_v128relaxed`  |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_reduce_minmax_u32_v128relaxed`   |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| __i64__                              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_i64_v128relaxed`  |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_reduce_minmax_i64_v128relaxed`   |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| __u64__                              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_reduce_moments_u64_v128relaxed`  |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_reduce_minmax_u64_v128relaxed`   |                   0 gb/s |                   0 gb/s |                   0 gb/s |

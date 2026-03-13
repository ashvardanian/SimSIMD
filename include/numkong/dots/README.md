# Batched Dot Products in NumKong

NumKong implements batched GEMM computing C = A × Bᵀ (packed) and C = A × Aᵀ (symmetric). B is pre-packed once and reused across queries. This is the foundation for the spatials, sets, and maxsim modules.

Packed dot product computes the full cross-product matrix:

```math
C_{ij} = \sum_{k} A_{ik} \cdot B_{jk}^T
```

Symmetric dot product uses the same matrix for both operands:

```math
C_{ij} = \sum_{k} A_{ik} \cdot A_{jk}
```

Reformulating as Python pseudocode:

```python
import numpy as np

def dots_packed(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a @ b.T

def dots_symmetric(a: np.ndarray) -> np.ndarray:
    return a @ a.T
```

## Input & Output Types

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

## Optimizations

### B Matrix Pre-Packing with Stride Breaking

`nk_dots_pack_f32_serial`, `nk_dots_pack_f32_haswell`, `nk_dots_pack_bf16_haswell`, `nk_dots_pack_i8_haswell` pre-pack the B matrix into a contiguous buffer optimized for streaming access during GEMM.
Power-of-2 stride detection — when `stride_bytes & (stride_bytes - 1) == 0` — adds `depth_simd_dimensions` padding to avoid cache associativity conflicts on set-associative caches.
Type conversion is amortized into the pack step: bf16→f32, f16→f32, and FP8→f32 conversions happen once during packing instead of per-row during GEMM.
A 64-byte header stores metadata: column count, depth dimensions, and padded depth.
Row grouping (`group_size=16`) zero-pads partial groups at matrix edges for uniform SIMD processing.

### Tiled Register Accumulation

`nk_dots_packed_f32_haswell`, `nk_dots_packed_f32_skylake`, `nk_dots_packed_f32_neon` use a 4×4 tile kernel with 16 accumulators to handle ~80% of the work.
A 1×8 tile kernel with 8 accumulators handles edge rows that don't fill a full 4-row tile.
No depth blocking is used — the kernel relies on hardware prefetch for streaming A/B access patterns.
Row loads are amortized across multiple dot products: each A row is loaded once and multiplied against 4 B columns per tile pass.

### AMX 2D Tile Engine

`nk_dots_packed_bf16_sapphireamx`, `nk_dots_packed_f32_sapphireamx`, `nk_dots_packed_i8_sapphireamx` use Intel AMX's 8 tile registers (TMM0–TMM7), each 1 KB (16 rows × 64 bytes).
Convention: TMM0–1 hold A tiles, TMM2–3 hold B tiles, TMM4–7 are C accumulators — giving a 2×2 output tile (32×32 f32 results) per tile pass.
`TDPBF16PS tmm_c, tmm_a, tmm_b` performs a 16×16 outer product with 32 bf16 multiply-adds per cell (16×16×32 = 8,192 MACs per instruction).
Each A row contains 16 bf16 pairs interleaved as [a₀, a₁, a₀, a₁, ...] and B columns as [b₀, b₁, b₀, b₁, ...] — the hardware consumes two bf16 elements per slot, accumulating into f32.
`TDPBSSD tmm_c, tmm_a, tmm_b` does the same for i8: 64 bytes per row gives 16×16×64 = 16,384 i8 MACs per instruction.
i8 data is quad-interleaved: [a₀, a₁, a₂, a₃, a₀, a₁, a₂, a₃, ...] so the hardware can consume four i8 elements per 32-bit slot.
Tile configuration via `LDTILECFG` sets row counts and column byte-widths per tile — allows undersized tiles at matrix edges without masking.
Morton Z-curve ordering for tile traversal improves cache reuse when both A and B exceed L2.
This eliminates the explicit M×N×K loop nesting and register file pressure of vector ISAs — the entire dot-product reduction happens inside the tile instruction.

### SME Outer-Product Streaming

`nk_dots_packed_f32_sme`, `nk_dots_packed_bf16_smehalf`, `nk_dots_packed_f64_smef64` use Arm's SME ZA tile array (up to 4 named tiles ZA0–ZA3 in 32-bit mode, each SVL×SVL elements).
`FMOPA za, pn/m, pm/m, zn.s, zm.s` computes a full SVL×SVL rank-1 update in one instruction — one row of A times one row of B, accumulated into ZA.
ZA0 time-shares between data staging and accumulation: A rows are loaded horizontally into ZA0 (`st1w {za0h.s[ws]}, ...`), then read vertically (`svread_ver_za32_f32_m`) to produce transposed column vectors for B.
This avoids explicit transpose operations — the tile's 2D addressing provides free transposition.
ZA1–ZA3 serve as accumulators while ZA0 stages the next data.
A 3-column-tile fast path handles B column count ≤ 3×SVL using ZA1–ZA3 as three separate accumulator tiles, avoiding spill/reload cycles.
For wider B, the kernel falls back to multi-pass accumulation with ZA store/load between passes.
`BFMOPA` for bf16 uses the same outer-product pattern but with bf16→f32 widening — 2× the depth per instruction vs f32 `FMOPA`.
`SMSTART`/`SMSTOP` streaming mode transitions cost ~50–100 cycles, amortized across the full M×N output.
Ozaki splitting for f64 (`nk_dots_packed_f64_smef64`) splits each f64 into 3 mantissa-masked f32 slices, computes 6 FMOPAs (all cross-products of 3×2 slices) into 3 ZA accumulators, then reconstructs the f64 result — achieving f64 precision using f32 tile hardware.

### Compensated Integer GEMM

`nk_dots_packed_i8_icelake`, `nk_dots_packed_u8_icelake`, `nk_dots_packed_i8_haswell` work around the unsigned×signed operand requirement of integer dot-product instructions.
`VPDPBUSD` (Ice Lake+) computes u8×i8 dot products accumulating directly to i32 — but requires one unsigned and one signed operand.
For signed×signed (i8×i8), one operand is XOR'd with `0x80` to shift to unsigned range, introducing a bias of $128 \cdot \sum_k b_k$ per output element.
Rather than computing the bias correction per-element inside the inner loop (requiring extra registers for running sums), the B column sums $\sum_k b_k$ are pre-computed once during packing and stored in the packed buffer metadata.
The inner loop only needs the `VPDPBUSD` accumulator — the bias subtraction is a single post-loop correction: `result[i][j] -= 128 * b_column_sum[j]`.
This reduces per-accumulator state from 2 registers (dot + running sum) to 1 register (dot only), freeing registers for more accumulators in the 4×4 tile.
Haswell fallback uses `VPMADDUBSW` (u8×i8→i16) + `VPMADDWD` (i16→i32), a two-instruction chain with i16 intermediate overflow risk — quantization ranges must be tighter ([-79, 79] vs [-127, 127]).

### 4-Way Finalizer Amortization

All packed and symmetric kernels across the dots, spatials, and sets modules share a finalizer-based design.
The 4×4 tile accumulates 16 dot products in registers, then stores results 4-wide via `nk_b128_vec_t` — a union of `f32[4]`, `i32[4]`, `u32[4]` fitting a 128-bit register.
A finalizer function pointer processes 4 results simultaneously, amortizing horizontal reductions and type conversions:

```
// 4-wide finalizer signature
void finalizer(nk_b128_vec_t dots,          // 4 dot products
               nk_f32_t query_norm,         // precomputed query squared-norm
               nk_b128_vec_t target_norms,  // 4 target squared-norms
               nk_b128_vec_t *results)      // 4 output distances

// Angular: 4 divisions + 4 subtractions in one call
results->f32s[i] = 1 - dots.f32s[i] / sqrt(query_norm * target_norms.f32s[i])

// Euclidean: 4 sqrt(a² + b² - 2ab) in one call
results->f32s[i] = sqrt(query_norm + target_norms.f32s[i] - 2 * dots.f32s[i])
```

The 4×4 tile emits 4 rows of 4 results each — the finalizer is called 4 times per tile, once per query row.
For the 1×8 edge tile, two finalizer calls handle 8 results.
This design decouples the GEMM loop from the distance metric: the same tiled accumulation code serves dots, spatials, and sets by swapping only the finalizer function pointer.

## Performance

The following performance tables are produced by manually re-running `nk_test` and `nk_bench` included internal tools to measure both accuracy and throughput at different input shapes.
The input size is controlled by `NK_MATRIX_HEIGHT`, `NK_MATRIX_WIDTH`, and `NK_MATRIX_DEPTH` environment variables, all set to the same value for products of two square matrices.
Columns show throughput for 256³, 1024³, and 4096³ matrix products.
The throughput is measured in GSO/s as Giga scalar operations per second, with $\text{ops} = 2 \cdot M \cdot N \cdot K$ arithmetic complexity for an $M \times K$ by $K \times N$ product.
Accuracy is reported as mean ULP (units in last place) unless noted otherwise — the average number of representable floating-point values between the result and the exact answer.
Each kernel runs for at least 20 seconds per configuration.
Benchmark threads are pinned to specific cores; on machines with heterogeneous core types (e.g., Apple P/E cores), only the fastest cores are used.
Workloads that significantly degrade CPU frequencies (Intel AMX, Apple SME) run in separate passes to avoid affecting throughput measurements of other kernels.

### Intel Sapphire Rapids

#### Native

| Kernel                               |                     256³ |                    1024³ |                    4096³ |
| :----------------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f64__                              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_f64_with_blas`       |       63.0 gso/s, 16 ulp |       75.0 gso/s, 58 ulp |     80.1 gso/s, 56.2 ulp |
| `nk_dots_packed_f64_with_mkl`        |       62.0 gso/s, 16 ulp |       75.5 gso/s, 58 ulp |     80.1 gso/s, 56.2 ulp |
| `nk_dots_symmetric_f64_with_blas`    |       53.4 gso/s, 13 ulp |       64.0 gso/s, 30 ulp |     76.9 gso/s, 50.8 ulp |
| `nk_dots_packed_f64_serial`          |       0.393 gso/s, 2 ulp |     0.489 gso/s, 4.6 ulp |     0.488 gso/s, 5.9 ulp |
| `nk_dots_symmetric_f64_serial`       |       0.346 gso/s, 2 ulp |     0.357 gso/s, 2.9 ulp |     0.574 gso/s, 3.9 ulp |
| `nk_dots_packed_f64_haswell`         |        3.67 gso/s, 0 ulp |        3.70 gso/s, 0 ulp |        4.32 gso/s, 0 ulp |
| `nk_dots_symmetric_f64_haswell`      |        5.09 gso/s, 0 ulp |        3.40 gso/s, 0 ulp |        12.0 gso/s, 0 ulp |
| `nk_dots_packed_f64_skylake`         |        7.86 gso/s, 0 ulp |        3.41 gso/s, 0 ulp |        9.38 gso/s, 0 ulp |
| `nk_dots_symmetric_f64_skylake`      |        4.66 gso/s, 0 ulp |        5.16 gso/s, 0 ulp |        10.9 gso/s, 0 ulp |
| __f32__                              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_f32_with_blas`       |        126 gso/s, 18 ulp |        143 gso/s, 30 ulp |       150 gso/s, 267 ulp |
| `nk_dots_packed_f32_with_mkl`        |        120 gso/s, 18 ulp |        133 gso/s, 30 ulp |         151 gso/s, ? ulp |
| `nk_dots_symmetric_f32_with_blas`    |        103 gso/s, 23 ulp |        125 gso/s, 39 ulp |       151 gso/s, 260 ulp |
| `nk_dots_packed_f32_serial`          |       13.3 gso/s, 21 ulp |       13.6 gso/s, 30 ulp |      13.2 gso/s, 725 ulp |
| `nk_dots_symmetric_f32_serial`       |       3.44 gso/s, 35 ulp |       3.52 gso/s, 32 ulp |      6.05 gso/s, 284 ulp |
| `nk_dots_packed_f32_haswell`         |        21.7 gso/s, 0 ulp |        23.7 gso/s, 0 ulp |        24.3 gso/s, 0 ulp |
| `nk_dots_symmetric_f32_haswell`      |        15.9 gso/s, 0 ulp |        6.52 gso/s, 0 ulp |        38.2 gso/s, 0 ulp |
| `nk_dots_packed_f32_skylake`         |        32.4 gso/s, 0 ulp |        13.6 gso/s, 0 ulp |        43.3 gso/s, 0 ulp |
| `nk_dots_symmetric_f32_skylake`      |        17.4 gso/s, 0 ulp |        20.9 gso/s, 0 ulp |        33.2 gso/s, 0 ulp |
| __bf16__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_bf16_serial`         |       0.842 gso/s, 0 ulp |     0.824 gso/s, 0.5 ulp |     0.825 gso/s, 5.4 ulp |
| `nk_dots_symmetric_bf16_serial`      |       0.808 gso/s, 0 ulp |     0.759 gso/s, 0.9 ulp |      1.74 gso/s, 5.4 ulp |
| `nk_dots_packed_bf16_haswell`        |        39.4 gso/s, 0 ulp |      43.1 gso/s, 0.4 ulp |      44.7 gso/s, 4.5 ulp |
| `nk_dots_symmetric_bf16_haswell`     |        29.6 gso/s, 0 ulp |      28.6 gso/s, 0.3 ulp |      79.4 gso/s, 4.2 ulp |
| `nk_dots_packed_bf16_skylake`        |        80.4 gso/s, 0 ulp |      68.4 gso/s, 0.3 ulp |      90.8 gso/s, 3.7 ulp |
| `nk_dots_symmetric_bf16_skylake`     |        30.5 gso/s, 0 ulp |      37.7 gso/s, 0.3 ulp |      82.5 gso/s, 3.2 ulp |
| `nk_dots_packed_bf16_genoa`          |        43.4 gso/s, 0 ulp |      28.7 gso/s, 0.3 ulp |      46.1 gso/s, 3.5 ulp |
| `nk_dots_symmetric_bf16_genoa`       |        34.5 gso/s, 0 ulp |      24.0 gso/s, 0.5 ulp |       148 gso/s, 3.5 ulp |
| `nk_dots_packed_bf16_sapphireamx`    |         256 gso/s, 0 ulp |       365 gso/s, 0.7 ulp |       692 gso/s, 5.8 ulp |
| `nk_dots_packed_bf16_with_mkl`       |         188 gso/s, 0 ulp |       503 gso/s, 0.7 ulp |       836 gso/s, 5.8 ulp |
| `nk_dots_symmetric_bf16_sapphireamx` |        80.7 gso/s, 0 ulp |       119 gso/s, 0.5 ulp |       120 gso/s, 5.8 ulp |
| __f16__                              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_f16_serial`          |       4.44 gso/s, 14 ulp |       4.42 gso/s, 40 ulp |      4.40 gso/s, 326 ulp |
| `nk_dots_symmetric_f16_serial`       |      3.66 gso/s, 8.9 ulp |       3.44 gso/s, 25 ulp |     5.06 gso/s, 55.6 ulp |
| `nk_dots_packed_f16_haswell`         |       38.1 gso/s, 12 ulp |       47.0 gso/s, 22 ulp |      48.7 gso/s, 374 ulp |
| `nk_dots_symmetric_f16_haswell`      |      32.4 gso/s, 7.7 ulp |       27.8 gso/s, 32 ulp |      93.8 gso/s, 486 ulp |
| `nk_dots_packed_f16_skylake`         |      68.6 gso/s, 7.3 ulp |        102 gso/s, 21 ulp |      97.9 gso/s, 138 ulp |
| `nk_dots_symmetric_f16_skylake`      |      30.4 gso/s, 5.9 ulp |       38.4 gso/s, 25 ulp |       72.7 gso/s, 32 ulp |
| `nk_dots_packed_f16_with_mkl`        |        133 gso/s, 17 ulp |        135 gso/s, 31 ulp |      143 gso/s, 39.5 ulp |
| __e5m2__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_e5m2_serial`         |        2.51 gso/s, 0 ulp |        2.57 gso/s, 0 ulp |        4.82 gso/s, 0 ulp |
| `nk_dots_symmetric_e5m2_serial`      |        2.47 gso/s, 0 ulp |        2.54 gso/s, 0 ulp |        5.06 gso/s, 0 ulp |
| `nk_dots_packed_e5m2_haswell`        |        15.5 gso/s, 0 ulp |        15.7 gso/s, 0 ulp |        20.6 gso/s, 0 ulp |
| `nk_dots_symmetric_e5m2_haswell`     |        14.3 gso/s, 0 ulp |        13.1 gso/s, 0 ulp |        34.4 gso/s, 0 ulp |
| `nk_dots_packed_e5m2_skylake`        |        23.6 gso/s, 0 ulp |        26.7 gso/s, 0 ulp |        40.2 gso/s, 0 ulp |
| `nk_dots_symmetric_e5m2_skylake`     |        13.5 gso/s, 0 ulp |        14.7 gso/s, 0 ulp |        27.4 gso/s, 0 ulp |
| `nk_dots_packed_e5m2_genoa`          |        24.4 gso/s, 0 ulp |        30.9 gso/s, 0 ulp |        51.2 gso/s, 0 ulp |
| `nk_dots_symmetric_e5m2_genoa`       |        20.2 gso/s, 0 ulp |        21.1 gso/s, 0 ulp |        68.8 gso/s, 0 ulp |
| `nk_dots_packed_e5m2_sapphireamx`    |         165 gso/s, 0 ulp |         401 gso/s, 0 ulp |         426 gso/s, 0 ulp |
| __e4m3__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_e4m3_serial`         |       0.326 gso/s, 0 ulp |       0.334 gso/s, 0 ulp |       0.318 gso/s, 0 ulp |
| `nk_dots_symmetric_e4m3_serial`      |       0.307 gso/s, 0 ulp |       0.315 gso/s, 0 ulp |       0.618 gso/s, 0 ulp |
| `nk_dots_packed_e4m3_haswell`        |        11.9 gso/s, 0 ulp |        11.0 gso/s, 0 ulp |        13.3 gso/s, 0 ulp |
| `nk_dots_symmetric_e4m3_haswell`     |        10.7 gso/s, 0 ulp |        9.79 gso/s, 0 ulp |        24.1 gso/s, 0 ulp |
| `nk_dots_packed_e4m3_skylake`        |        25.0 gso/s, 0 ulp |        19.7 gso/s, 0 ulp |        32.9 gso/s, 0 ulp |
| `nk_dots_symmetric_e4m3_skylake`     |        10.3 gso/s, 0 ulp |        12.0 gso/s, 0 ulp |        21.0 gso/s, 0 ulp |
| `nk_dots_packed_e4m3_genoa`          |        27.4 gso/s, 0 ulp |        21.0 gso/s, 0 ulp |        35.0 gso/s, 0 ulp |
| `nk_dots_symmetric_e4m3_genoa`       |        19.9 gso/s, 0 ulp |        20.4 gso/s, 0 ulp |        67.2 gso/s, 0 ulp |
| `nk_dots_packed_e4m3_sapphireamx`    |         166 gso/s, 0 ulp |         221 gso/s, 0 ulp |         440 gso/s, 0 ulp |
| __e3m2__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_e3m2_serial`         |        2.34 gso/s, 0 ulp |        2.53 gso/s, 0 ulp |        2.80 gso/s, 0 ulp |
| `nk_dots_symmetric_e3m2_serial`      |        1.88 gso/s, 0 ulp |        2.04 gso/s, 0 ulp |        3.96 gso/s, 0 ulp |
| `nk_dots_packed_e3m2_haswell`        |        32.0 gso/s, 0 ulp |        19.5 gso/s, 0 ulp |        20.2 gso/s, 0 ulp |
| `nk_dots_symmetric_e3m2_haswell`     |        13.4 gso/s, 0 ulp |        24.6 gso/s, 0 ulp |        63.2 gso/s, 0 ulp |
| `nk_dots_packed_e3m2_skylake`        |        25.8 gso/s, 0 ulp |        27.4 gso/s, 0 ulp |        41.7 gso/s, 0 ulp |
| `nk_dots_symmetric_e3m2_skylake`     |        26.8 gso/s, 0 ulp |        29.5 gso/s, 0 ulp |        62.3 gso/s, 0 ulp |
| `nk_dots_packed_e3m2_genoa`          |        30.0 gso/s, 0 ulp |        29.4 gso/s, 0 ulp |        57.7 gso/s, 0 ulp |
| `nk_dots_symmetric_e3m2_genoa`       |        39.5 gso/s, 0 ulp |        27.5 gso/s, 0 ulp |        88.0 gso/s, 0 ulp |
| `nk_dots_packed_e3m2_sapphireamx`    |         269 gso/s, 0 ulp |         177 gso/s, 0 ulp |         499 gso/s, 0 ulp |
| `nk_dots_symmetric_e3m2_sapphireamx` |        63.4 gso/s, 0 ulp |        37.6 gso/s, 0 ulp |        88.0 gso/s, 0 ulp |
| __e2m3__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_e2m3_serial`         |        2.33 gso/s, 0 ulp |        2.47 gso/s, 0 ulp |        4.82 gso/s, 0 ulp |
| `nk_dots_symmetric_e2m3_serial`      |        1.94 gso/s, 0 ulp |        2.02 gso/s, 0 ulp |        4.05 gso/s, 0 ulp |
| `nk_dots_packed_e2m3_haswell`        |        53.7 gso/s, 0 ulp |        35.8 gso/s, 0 ulp |        36.7 gso/s, 0 ulp |
| `nk_dots_symmetric_e2m3_haswell`     |        21.5 gso/s, 0 ulp |        53.5 gso/s, 0 ulp |         127 gso/s, 0 ulp |
| `nk_dots_packed_e2m3_skylake`        |        44.0 gso/s, 0 ulp |        55.5 gso/s, 0 ulp |        86.5 gso/s, 0 ulp |
| `nk_dots_symmetric_e2m3_skylake`     |        37.8 gso/s, 0 ulp |        51.6 gso/s, 0 ulp |        91.3 gso/s, 0 ulp |
| `nk_dots_packed_e2m3_genoa`          |        30.5 gso/s, 0 ulp |        24.5 gso/s, 0 ulp |        59.2 gso/s, 0 ulp |
| `nk_dots_symmetric_e2m3_genoa`       |        24.4 gso/s, 0 ulp |        25.6 gso/s, 0 ulp |        94.8 gso/s, 0 ulp |
| `nk_dots_packed_e2m3_sapphireamx`    |         398 gso/s, 0 ulp |        1.17 tso/s, 0 ulp |        1.10 tso/s, 0 ulp |
| `nk_dots_symmetric_e2m3_sapphireamx` |        94.8 gso/s, 0 ulp |         201 gso/s, 0 ulp |         199 gso/s, 0 ulp |
| `nk_dots_packed_e2m3_alder`          |        73.3 gso/s, 0 ulp |        40.4 gso/s, 0 ulp |        88.6 gso/s, 0 ulp |
| `nk_dots_symmetric_e2m3_alder`       |        34.2 gso/s, 0 ulp |        44.5 gso/s, 0 ulp |         166 gso/s, 0 ulp |
| __i8__                               | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_i8_serial`           |               6.44 gso/s |               6.62 gso/s |               7.44 gso/s |
| `nk_dots_symmetric_i8_serial`        |               2.93 gso/s |               2.99 gso/s |               5.83 gso/s |
| `nk_dots_packed_i8_haswell`          |               48.0 gso/s |               88.3 gso/s |               99.7 gso/s |
| `nk_dots_symmetric_i8_haswell`       |               49.7 gso/s |               61.6 gso/s |                170 gso/s |
| `nk_dots_packed_i8_icelake`          |                121 gso/s |                201 gso/s |                209 gso/s |
| `nk_dots_symmetric_i8_icelake`       |               66.7 gso/s |                107 gso/s |                350 gso/s |
| `nk_dots_packed_i8_sapphireamx`      |                365 gso/s |                700 gso/s |               1.35 tso/s |
| `nk_dots_packed_i8u8_with_mkl`       |                261 gso/s |                633 gso/s |               1.64 tso/s |
| `nk_dots_symmetric_i8_sapphireamx`   |                109 gso/s |                262 gso/s |                221 gso/s |
| `nk_dots_packed_i8_alder`            |                176 gso/s |                120 gso/s |                265 gso/s |
| `nk_dots_symmetric_i8_alder`         |               59.1 gso/s |               97.3 gso/s |                348 gso/s |
| __u8__                               | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_u8_serial`           |               7.45 gso/s |               7.79 gso/s |               7.88 gso/s |
| `nk_dots_symmetric_u8_serial`        |               2.81 gso/s |               2.91 gso/s |               5.35 gso/s |
| `nk_dots_packed_u8_haswell`          |               50.6 gso/s |               53.6 gso/s |                100 gso/s |
| `nk_dots_symmetric_u8_haswell`       |               56.8 gso/s |               28.7 gso/s |                171 gso/s |
| `nk_dots_packed_u8_icelake`          |                117 gso/s |                195 gso/s |                222 gso/s |
| `nk_dots_symmetric_u8_icelake`       |               60.0 gso/s |                102 gso/s |                215 gso/s |
| `nk_dots_packed_u8_alder`            |               90.2 gso/s |                126 gso/s |                271 gso/s |
| `nk_dots_symmetric_u8_alder`         |               54.5 gso/s |               90.4 gso/s |                359 gso/s |
| __i4__                               | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_i4_serial`           |               2.43 gso/s |               2.43 gso/s |               2.24 gso/s |
| `nk_dots_symmetric_i4_serial`        |               2.26 gso/s |               2.13 gso/s |               4.44 gso/s |
| `nk_dots_packed_i4_icelake`          |               90.8 gso/s |                125 gso/s |                137 gso/s |
| `nk_dots_symmetric_i4_icelake`       |               68.7 gso/s |                106 gso/s |                351 gso/s |
| __u4__                               | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_u4_serial`           |               3.27 gso/s |               3.37 gso/s |               3.33 gso/s |
| `nk_dots_symmetric_u4_serial`        |               3.02 gso/s |               3.06 gso/s |               6.13 gso/s |
| `nk_dots_packed_u4_icelake`          |                123 gso/s |                176 gso/s |                196 gso/s |
| `nk_dots_symmetric_u4_icelake`       |               82.1 gso/s |                129 gso/s |                403 gso/s |
| __u1__                               | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_u1_haswell`          |                144 gso/s |                159 gso/s |                356 gso/s |
| `nk_dots_symmetric_u1_haswell`       |                112 gso/s |                152 gso/s |                707 gso/s |
| `nk_dots_packed_u1_icelake`          |                176 gso/s |                446 gso/s |                869 gso/s |
| `nk_dots_symmetric_u1_icelake`       |                116 gso/s |                295 gso/s |               1.35 tso/s |
| __i16__                              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_i16_with_mkl`        |                261 gso/s |                322 gso/s |                365 gso/s |

#### WASM

Measured with Wasmtime v42 (Cranelift backend).

| Kernel                               |                     256³ |                    1024³ |                    4096³ |
| :----------------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f64__                              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_f64_serial`          |       0.596 gso/s, 3 ulp |       0.701 gso/s, 5 ulp |       0.797 gso/s, 6 ulp |
| `nk_dots_symmetric_f64_serial`       |       0.497 gso/s, 4 ulp |     0.271 gso/s, 2.9 ulp |     0.272 gso/s, 3.9 ulp |
| `nk_dots_packed_f64_v128relaxed`     |      12.0 gso/s, 9.9 ulp |       11.8 gso/s, 52 ulp |       12.1 gso/s, 33 ulp |
| `nk_dots_symmetric_f64_v128relaxed`  |      0.126 gso/s, 11 ulp |     0.164 gso/s, 124 ulp |     0.0850 gso/s, 38 ulp |
| __f32__                              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_f32_serial`          |       17.0 gso/s, 19 ulp |       16.8 gso/s, 30 ulp |      17.5 gso/s, 725 ulp |
| `nk_dots_symmetric_f32_serial`       |       4.26 gso/s, 20 ulp |       4.27 gso/s, 29 ulp |    0.0953 gso/s, 284 ulp |
| `nk_dots_packed_f32_v128relaxed`     |       23.5 gso/s, 11 ulp |       23.2 gso/s, 19 ulp |       23.5 gso/s, 31 ulp |
| `nk_dots_symmetric_f32_v128relaxed`  |      0.209 gso/s, 12 ulp |     0.0965 gso/s, 50 ulp |      0.208 gso/s, 37 ulp |
| __bf16__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_bf16_serial`         |      1.18 gso/s, 0.1 ulp |      1.19 gso/s, 0.5 ulp |        1.22 gso/s, 5 ulp |
| `nk_dots_symmetric_bf16_serial`      |        1.17 gso/s, 0 ulp |      1.18 gso/s, 0.6 ulp |     0.279 gso/s, 5.4 ulp |
| `nk_dots_packed_bf16_v128relaxed`    |        24.8 gso/s, 0 ulp |      25.6 gso/s, 0.5 ulp |      25.3 gso/s, 6.8 ulp |
| `nk_dots_symmetric_bf16_v128relaxed` |     0.174 gso/s, 0.1 ulp |     0.226 gso/s, 1.4 ulp |      0.243 gso/s, 11 ulp |
| __e5m2__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_e5m2_serial`         |       0.537 gso/s, 0 ulp |       0.576 gso/s, 0 ulp |        2.71 gso/s, 0 ulp |
| `nk_dots_symmetric_e5m2_serial`      |       0.377 gso/s, 0 ulp |       0.437 gso/s, 0 ulp |      0.0410 gso/s, 0 ulp |
| `nk_dots_packed_e5m2_v128relaxed`    |        6.42 gso/s, ? ulp |        6.63 gso/s, ? ulp |        6.40 gso/s, ? ulp |
| `nk_dots_symmetric_e5m2_v128relaxed` |        5.54 gso/s, 0 ulp |        5.69 gso/s, 0 ulp |        5.76 gso/s, 0 ulp |
| __e4m3__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_e4m3_serial`         |       0.469 gso/s, 0 ulp |       0.441 gso/s, 0 ulp |       0.475 gso/s, 0 ulp |
| `nk_dots_symmetric_e4m3_serial`      |       0.394 gso/s, 0 ulp |       0.393 gso/s, 0 ulp |      0.0390 gso/s, 0 ulp |
| `nk_dots_packed_e4m3_v128relaxed`    |        3.55 gso/s, ? ulp |        4.98 gso/s, ? ulp |        4.89 gso/s, ? ulp |
| `nk_dots_symmetric_e4m3_v128relaxed` |        4.36 gso/s, 0 ulp |        4.32 gso/s, 0 ulp |        4.58 gso/s, 0 ulp |
| __e2m3__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_e2m3_serial`         |       0.334 gso/s, 0 ulp |       0.324 gso/s, 0 ulp |        2.76 gso/s, 0 ulp |
| `nk_dots_symmetric_e2m3_serial`      |       0.302 gso/s, 0 ulp |       0.297 gso/s, 0 ulp |      0.0411 gso/s, 0 ulp |
| `nk_dots_packed_e2m3_v128relaxed`    |        18.8 gso/s, 0 ulp |        19.1 gso/s, 0 ulp |        19.0 gso/s, 0 ulp |
| `nk_dots_symmetric_e2m3_v128relaxed` |       0.288 gso/s, 0 ulp |      0.0950 gso/s, 0 ulp |      0.0953 gso/s, 0 ulp |
| __i8__                               | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_i8_serial`           |               9.61 gso/s |               9.66 gso/s |               4.88 gso/s |
| `nk_dots_symmetric_i8_serial`        |               4.69 gso/s |               4.73 gso/s |              0.227 gso/s |
| `nk_dots_packed_i8_v128relaxed`      |               32.1 gso/s |               31.9 gso/s |               33.4 gso/s |
| `nk_dots_symmetric_i8_v128relaxed`   |              0.186 gso/s |              0.302 gso/s |              0.258 gso/s |
| __u8__                               | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_u8_serial`           |               13.1 gso/s |               13.7 gso/s |               5.26 gso/s |
| `nk_dots_symmetric_u8_serial`        |               4.52 gso/s |               4.77 gso/s |             0.0433 gso/s |
| `nk_dots_packed_u8_v128relaxed`      |               82.7 gso/s |               84.3 gso/s |               84.1 gso/s |
| `nk_dots_symmetric_u8_v128relaxed`   |              0.124 gso/s |              0.192 gso/s |              0.162 gso/s |
| __i4__                               | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_i4_serial`           |               3.93 gso/s |               3.90 gso/s |               7.85 gso/s |
| `nk_dots_symmetric_i4_serial`        |               3.47 gso/s |               3.52 gso/s |              0.274 gso/s |
| `nk_dots_packed_i4_v128relaxed`      |               7.80 gso/s |               10.3 gso/s |               10.1 gso/s |
| `nk_dots_symmetric_i4_v128relaxed`   |               20.2 gso/s |               20.4 gso/s |               20.4 gso/s |
| __u4__                               | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_u4_serial`           |               5.76 gso/s |               5.78 gso/s |               5.90 gso/s |
| `nk_dots_symmetric_u4_serial`        |               5.25 gso/s |               5.71 gso/s |              0.271 gso/s |
| `nk_dots_packed_u4_v128relaxed`      |               52.8 gso/s |               71.6 gso/s |               70.8 gso/s |
| `nk_dots_symmetric_u4_v128relaxed`   |               28.6 gso/s |               29.1 gso/s |               28.9 gso/s |
| __u1__                               | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_u1_serial`           |                  0 gso/s |                  0 gso/s |                  0 gso/s |
| `nk_dots_packed_u1_v128relaxed`      |                301 gso/s |                305 gso/s |                303 gso/s |
| `nk_dots_symmetric_u1_serial`        |                  0 gso/s |                  0 gso/s |                  0 gso/s |
| `nk_dots_symmetric_u1_v128relaxed`   |              0.137 gso/s |              0.252 gso/s |              0.199 gso/s |

### Apple M4

#### Native

| Kernel                             |                      256³ |                     1024³ |                     4096³ |
| :--------------------------------- | ------------------------: | ------------------------: | ------------------------: |
| __f64__                            |  ░░░░░░░░░░░░░░░░░░░░░░░░ |  ░░░░░░░░░░░░░░░░░░░░░░░░ |  ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_f64_serial`        |        0.596 gso/s, 3 ulp |        0.701 gso/s, 5 ulp |        0.797 gso/s, 6 ulp |
| `nk_dots_symmetric_f64_serial`     |        0.497 gso/s, 4 ulp |        0.568 gso/s, 3 ulp |         1.50 gso/s, ? ulp |
| `nk_dots_packed_f64_neon`          |         4.23 gso/s, 0 ulp |         4.26 gso/s, 0 ulp |         4.22 gso/s, 0 ulp |
| `nk_dots_symmetric_f64_neon`       |         1.91 gso/s, 0 ulp |         1.95 gso/s, 0 ulp |         3.96 gso/s, ? ulp |
| `nk_dots_packed_f64_smef64`        |       24.9 gso/s, 1.5 ulp |            ? gso/s, ? ulp |       39.6 gso/s, 0.9 ulp |
| `nk_dots_symmetric_f64_smef64`     |       5.33 gso/s, 1.5 ulp |            ? gso/s, ? ulp |       5.45 gso/s, 1.1 ulp |
| __f32__                            |  ░░░░░░░░░░░░░░░░░░░░░░░░ |  ░░░░░░░░░░░░░░░░░░░░░░░░ |  ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_f32_serial`        |        17.0 gso/s, 19 ulp |        16.8 gso/s, 30 ulp |       17.5 gso/s, 725 ulp |
| `nk_dots_symmetric_f32_serial`     |        4.26 gso/s, 20 ulp |        4.27 gso/s, 29 ulp |      6.22 gso/s, 557M ulp |
| `nk_dots_packed_f32_neon`          |         28.3 gso/s, 0 ulp |         29.0 gso/s, 0 ulp |         29.4 gso/s, 0 ulp |
| `nk_dots_symmetric_f32_neon`       |         4.21 gso/s, 0 ulp |         4.25 gso/s, 0 ulp |         8.38 gso/s, 0 ulp |
| `nk_dots_packed_f32_smef64`        |          185 gso/s, 0 ulp |            ? gso/s, ? ulp |          209 gso/s, 0 ulp |
| `nk_dots_symmetric_f32_smef64`     |         18.8 gso/s, 0 ulp |            ? gso/s, ? ulp |         23.1 gso/s, 0 ulp |
| __bf16__                           |  ░░░░░░░░░░░░░░░░░░░░░░░░ |  ░░░░░░░░░░░░░░░░░░░░░░░░ |  ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_bf16_serial`       |       1.18 gso/s, 0.1 ulp |       1.19 gso/s, 0.5 ulp |         1.22 gso/s, 5 ulp |
| `nk_dots_symmetric_bf16_serial`    |         1.17 gso/s, 0 ulp |       1.18 gso/s, 0.6 ulp |      10.6 gso/s, 557M ulp |
| `nk_dots_packed_bf16_neonbfdot`    |         42.2 gso/s, 0 ulp |       42.5 gso/s, 0.5 ulp |         43.3 gso/s, ? ulp |
| `nk_dots_symmetric_bf16_neonbfdot` |         15.0 gso/s, 0 ulp |       15.2 gso/s, 0.5 ulp |         30.3 gso/s, ? ulp |
| `nk_dots_packed_bf16_sme`          |          543 gso/s, 0 ulp |      1,120 gso/s, 4.2 ulp |        479 gso/s, 3.8 ulp |
| `nk_dots_symmetric_bf16_sme`       |        104 gso/s, 0.1 ulp |        165 gso/s, 1.8 ulp |       94.4 gso/s, 1.8 ulp |
| __f16__                            |  ░░░░░░░░░░░░░░░░░░░░░░░░ |  ░░░░░░░░░░░░░░░░░░░░░░░░ |  ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_f16_serial`        |       8.26 gso/s, 204 ulp |        8.04 gso/s, 36 ulp |       8.60 gso/s, 326 ulp |
| `nk_dots_symmetric_f16_serial`     |        4.21 gso/s, 13 ulp |        4.17 gso/s, 29 ulp |      15.6 gso/s, 557M ulp |
| `nk_dots_packed_f16_neonhalf`      |      56.9 gso/s, 16.8 ulp |      58.0 gso/s, 25.5 ulp |         58.3 gso/s, ? ulp |
| `nk_dots_symmetric_f16_neonhalf`   |      8.30 gso/s, 11.5 ulp |      8.35 gso/s, 24.9 ulp |         16.9 gso/s, ? ulp |
| `nk_dots_packed_f16_neonfhm`       |      73.8 gso/s, 16.7 ulp |      73.4 gso/s, 25.5 ulp |         75.1 gso/s, ? ulp |
| `nk_dots_symmetric_f16_neonfhm`    |      14.9 gso/s, 11.5 ulp |      15.2 gso/s, 24.9 ulp |         30.4 gso/s, ? ulp |
| `nk_dots_packed_f16_sme`           |       550 gso/s, 14.8 ulp |     1,120 gso/s, 28.2 ulp |       477 gso/s, 28.2 ulp |
| `nk_dots_symmetric_f16_sme`        |      96.3 gso/s, 12.6 ulp |       165 gso/s, 24.4 ulp |      95.0 gso/s, 24.4 ulp |
| __e5m2__                           |  ░░░░░░░░░░░░░░░░░░░░░░░░ |  ░░░░░░░░░░░░░░░░░░░░░░░░ |  ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_e5m2_serial`       |        0.537 gso/s, 0 ulp |        0.576 gso/s, 0 ulp |         9.81 gso/s, 0 ulp |
| `nk_dots_symmetric_e5m2_serial`    |        0.377 gso/s, 0 ulp |        0.437 gso/s, 0 ulp |      5.44 gso/s, 557M ulp |
| `nk_dots_packed_e5m2_neonfhm`      |         66.7 gso/s, 0 ulp |         67.7 gso/s, 0 ulp |         68.7 gso/s, 0 ulp |
| `nk_dots_symmetric_e5m2_neonfhm`   |         24.6 gso/s, 0 ulp |         26.7 gso/s, 0 ulp |         53.1 gso/s, 0 ulp |
| `nk_dots_packed_e5m2_sme`          |          502 gso/s, 0 ulp |        1,100 gso/s, 0 ulp |          578 gso/s, 0 ulp |
| `nk_dots_symmetric_e5m2_sme`       |          132 gso/s, 0 ulp |          163 gso/s, 0 ulp |          102 gso/s, 0 ulp |
| __e4m3__                           |  ░░░░░░░░░░░░░░░░░░░░░░░░ |  ░░░░░░░░░░░░░░░░░░░░░░░░ |  ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_e4m3_serial`       |        0.469 gso/s, 0 ulp |        0.441 gso/s, 0 ulp |        0.475 gso/s, 0 ulp |
| `nk_dots_symmetric_e4m3_serial`    |        0.394 gso/s, 0 ulp |        0.393 gso/s, 0 ulp |     0.859 gso/s, 557M ulp |
| `nk_dots_packed_e4m3_neonfhm`      |         22.2 gso/s, 0 ulp |         22.4 gso/s, 0 ulp |         22.6 gso/s, 0 ulp |
| `nk_dots_symmetric_e4m3_neonfhm`   |         12.5 gso/s, 0 ulp |         13.0 gso/s, 0 ulp |         25.7 gso/s, 0 ulp |
| `nk_dots_packed_e4m3_sme`          |          269 gso/s, 0 ulp |          475 gso/s, 0 ulp |          211 gso/s, 0 ulp |
| `nk_dots_symmetric_e4m3_sme`       |         27.5 gso/s, 0 ulp |         33.4 gso/s, 0 ulp |         25.5 gso/s, 0 ulp |
| __e3m2__                           |  ░░░░░░░░░░░░░░░░░░░░░░░░ |  ░░░░░░░░░░░░░░░░░░░░░░░░ |  ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_e3m2_serial`       |        0.546 gso/s, 0 ulp |        0.564 gso/s, 0 ulp |         8.81 gso/s, 0 ulp |
| `nk_dots_symmetric_e3m2_serial`    |        0.435 gso/s, 0 ulp |        0.433 gso/s, 0 ulp |      5.38 gso/s, 557M ulp |
| `nk_dots_packed_e3m2_neonfhm`      |         44.0 gso/s, 0 ulp |         44.5 gso/s, 0 ulp |         44.7 gso/s, 0 ulp |
| `nk_dots_symmetric_e3m2_neonfhm`   |         19.4 gso/s, 0 ulp |         20.1 gso/s, 0 ulp |         41.7 gso/s, 0 ulp |
| `nk_dots_packed_e3m2_sme`          |    251 gso/s, 946,000 ulp |  446 gso/s, 1,200,000 ulp |  221 gso/s, 1,200,000 ulp |
| `nk_dots_symmetric_e3m2_sme`       | 27.9 gso/s, 1,660,000 ulp | 31.5 gso/s, 2,310,000 ulp | 23.3 gso/s, 2,310,000 ulp |
| __e2m3__                           |  ░░░░░░░░░░░░░░░░░░░░░░░░ |  ░░░░░░░░░░░░░░░░░░░░░░░░ |  ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_e2m3_serial`       |        0.334 gso/s, 0 ulp |        0.324 gso/s, 0 ulp |         8.86 gso/s, 0 ulp |
| `nk_dots_symmetric_e2m3_serial`    |        0.302 gso/s, 0 ulp |        0.297 gso/s, 0 ulp |      5.44 gso/s, 557M ulp |
| `nk_dots_packed_e2m3_neonfhm`      |         34.6 gso/s, 0 ulp |         34.6 gso/s, 0 ulp |         35.1 gso/s, 0 ulp |
| `nk_dots_symmetric_e2m3_neonfhm`   |         16.5 gso/s, 0 ulp |         17.3 gso/s, 0 ulp |         34.0 gso/s, 0 ulp |
| `nk_dots_packed_e2m3_sme`          |          752 gso/s, 0 ulp |        2,010 gso/s, 0 ulp |          813 gso/s, 0 ulp |
| `nk_dots_symmetric_e2m3_sme`       |          132 gso/s, 0 ulp |          170 gso/s, 0 ulp |          108 gso/s, 0 ulp |
| __i8__                             |  ░░░░░░░░░░░░░░░░░░░░░░░░ |  ░░░░░░░░░░░░░░░░░░░░░░░░ |  ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_i8_serial`         |                9.61 gso/s |                9.66 gso/s |                13.7 gso/s |
| `nk_dots_symmetric_i8_serial`      |                4.69 gso/s |                4.73 gso/s |                8.99 gso/s |
| `nk_dots_packed_i8_neonsdot`       |                 240 gso/s |                 260 gso/s |                 282 gso/s |
| `nk_dots_symmetric_i8_neonsdot`    |                30.3 gso/s |                32.8 gso/s |                63.0 gso/s |
| `nk_dots_packed_i8_sme`            |               1,010 gso/s |               2,380 gso/s |               1,310 gso/s |
| `nk_dots_symmetric_i8_sme`         |                 186 gso/s |                 366 gso/s |                 213 gso/s |
| __u8__                             |  ░░░░░░░░░░░░░░░░░░░░░░░░ |  ░░░░░░░░░░░░░░░░░░░░░░░░ |  ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_u8_serial`         |                13.1 gso/s |                13.7 gso/s |                10.5 gso/s |
| `nk_dots_symmetric_u8_serial`      |                4.52 gso/s |                4.77 gso/s |                10.4 gso/s |
| `nk_dots_packed_u8_neonsdot`       |                 238 gso/s |                 262 gso/s |                 272 gso/s |
| `nk_dots_symmetric_u8_neonsdot`    |                30.3 gso/s |                32.4 gso/s |                63.5 gso/s |
| `nk_dots_packed_u8_sme`            |               1,180 gso/s |               2,380 gso/s |               1,290 gso/s |
| `nk_dots_symmetric_u8_sme`         |                 201 gso/s |                 366 gso/s |                 209 gso/s |
| __i4__                             |  ░░░░░░░░░░░░░░░░░░░░░░░░ |  ░░░░░░░░░░░░░░░░░░░░░░░░ |  ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_i4_serial`         |                3.93 gso/s |                3.90 gso/s |                18.4 gso/s |
| `nk_dots_symmetric_i4_serial`      |                3.47 gso/s |                3.52 gso/s |                9.52 gso/s |
| `nk_dots_packed_i4_neonsdot`       |                 185 gso/s |                 197 gso/s |                 189 gso/s |
| `nk_dots_symmetric_i4_neonsdot`    |                51.0 gso/s |                58.1 gso/s |                 121 gso/s |
| `nk_dots_packed_i4_sme`            |               1,150 gso/s |               2,200 gso/s |               1,290 gso/s |
| `nk_dots_symmetric_i4_sme`         |                 236 gso/s |                 421 gso/s |                 235 gso/s |
| __u4__                             |  ░░░░░░░░░░░░░░░░░░░░░░░░ |  ░░░░░░░░░░░░░░░░░░░░░░░░ |  ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_u4_serial`         |                5.76 gso/s |                5.78 gso/s |                17.7 gso/s |
| `nk_dots_symmetric_u4_serial`      |                5.25 gso/s |                5.71 gso/s |                10.4 gso/s |
| `nk_dots_packed_u4_neonsdot`       |                 209 gso/s |                 229 gso/s |                 218 gso/s |
| `nk_dots_symmetric_u4_neonsdot`    |                51.9 gso/s |                59.7 gso/s |                 120 gso/s |
| `nk_dots_packed_u4_sme`            |               1,190 gso/s |               2,250 gso/s |               1,100 gso/s |
| `nk_dots_symmetric_u4_sme`         |                 182 gso/s |                   ? gso/s |                 244 gso/s |
| __u1__                             |  ░░░░░░░░░░░░░░░░░░░░░░░░ |  ░░░░░░░░░░░░░░░░░░░░░░░░ |  ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_u1_serial`         |                   ? gso/s |                   ? gso/s |                   ? gso/s |
| `nk_dots_symmetric_u1_serial`      |                   ? gso/s |                   ? gso/s |                   ? gso/s |
| `nk_dots_packed_u1_neon`           |                 571 gso/s |                 685 gso/s |                 732 gso/s |
| `nk_dots_symmetric_u1_neon`        |                 125 gso/s |                 210 gso/s |                 495 gso/s |
| `nk_dots_packed_u1_smebi32`        |                   ? gso/s |                   ? gso/s |                   ? gso/s |
| `nk_dots_symmetric_u1_smebi32`     |                   ? gso/s |                   ? gso/s |                   ? gso/s |

#### WASM

Measured with Wasmtime v42 (Cranelift backend).

| Kernel                               |                     256³ |                    1024³ |                    4096³ |
| :----------------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f64__                              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_f64_serial`          |        2.11 gso/s, 3 ulp |        4.67 gso/s, 5 ulp |      5.37 gso/s, 2.2 ulp |
| `nk_dots_symmetric_f64_serial`       |        1.89 gso/s, 4 ulp |        3.21 gso/s, 3 ulp |      5.62 gso/s, 2.4 ulp |
| `nk_dots_packed_f64_v128relaxed`     |     34.9 gso/s, 32.4 ulp |     32.5 gso/s, 32.4 ulp |     38.4 gso/s, 32.4 ulp |
| `nk_dots_symmetric_f64_v128relaxed`  |     10.2 gso/s, 37.6 ulp |     10.4 gso/s, 37.6 ulp |     10.9 gso/s, 37.6 ulp |
| __f32__                              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_f32_serial`          |       9.09 gso/s, 19 ulp |       17.2 gso/s, 30 ulp |     26.3 gso/s, 41.7 ulp |
| `nk_dots_symmetric_f32_serial`       |       6.90 gso/s, 20 ulp |       18.2 gso/s, 29 ulp |     18.2 gso/s, 58.8 ulp |
| `nk_dots_packed_f32_v128relaxed`     |     61.5 gso/s, 44.1 ulp |     65.7 gso/s, 44.1 ulp |     68.0 gso/s, 44.1 ulp |
| `nk_dots_symmetric_f32_v128relaxed`  |     19.5 gso/s, 48.2 ulp |     20.3 gso/s, 48.2 ulp |     20.4 gso/s, 48.2 ulp |
| __bf16__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_bf16_serial`         |      7.36 gso/s, 0.1 ulp |      17.8 gso/s, 0.5 ulp |      21.2 gso/s, 1.3 ulp |
| `nk_dots_symmetric_bf16_serial`      |        8.24 gso/s, 0 ulp |      26.9 gso/s, 0.6 ulp |      26.9 gso/s, 1.1 ulp |
| `nk_dots_packed_bf16_v128relaxed`    |      52.7 gso/s, 1.4 ulp |      55.1 gso/s, 1.4 ulp |      59.0 gso/s, 1.4 ulp |
| `nk_dots_symmetric_bf16_v128relaxed` |      16.8 gso/s, 1.3 ulp |      16.3 gso/s, 1.3 ulp |      18.0 gso/s, 1.3 ulp |
| __f16__                              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_f16_serial`          |     0.615 gso/s, 204 ulp |       1.42 gso/s, 36 ulp |     1.67 gso/s, 25.9 ulp |
| `nk_dots_symmetric_f16_serial`       |      0.528 gso/s, 13 ulp |       1.38 gso/s, 29 ulp |     1.46 gso/s, 27.9 ulp |
| `nk_dots_packed_f16_v128relaxed`     |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_dots_symmetric_f16_v128relaxed`  |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| __e5m2__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_e5m2_serial`         |        1.96 gso/s, 0 ulp |        4.57 gso/s, 0 ulp |        5.72 gso/s, 0 ulp |
| `nk_dots_symmetric_e5m2_serial`      |        2.78 gso/s, 0 ulp |        7.59 gso/s, 0 ulp |        8.02 gso/s, 0 ulp |
| `nk_dots_packed_e5m2_v128relaxed`    |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_dots_symmetric_e5m2_v128relaxed` |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| __e4m3__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_e4m3_serial`         |       0.340 gso/s, 0 ulp |       0.717 gso/s, 0 ulp |       0.864 gso/s, 0 ulp |
| `nk_dots_symmetric_e4m3_serial`      |       0.331 gso/s, 0 ulp |       0.822 gso/s, 0 ulp |       0.874 gso/s, 0 ulp |
| `nk_dots_packed_e4m3_v128relaxed`    |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_dots_symmetric_e4m3_v128relaxed` |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| __e3m2__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_e3m2_serial`         |        1.90 gso/s, 0 ulp |        4.38 gso/s, 0 ulp |        5.66 gso/s, 0 ulp |
| `nk_dots_symmetric_e3m2_serial`      |        2.72 gso/s, 0 ulp |        7.33 gso/s, 0 ulp |        7.70 gso/s, 0 ulp |
| `nk_dots_packed_e3m2_v128relaxed`    |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_dots_symmetric_e3m2_v128relaxed` |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| __e2m3__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_e2m3_serial`         |        1.93 gso/s, 0 ulp |        4.42 gso/s, 0 ulp |        5.66 gso/s, 0 ulp |
| `nk_dots_symmetric_e2m3_serial`      |        2.71 gso/s, 0 ulp |        7.31 gso/s, 0 ulp |        7.70 gso/s, 0 ulp |
| `nk_dots_packed_e2m3_v128relaxed`    |        34.8 gso/s, 0 ulp |        35.0 gso/s, 0 ulp |        38.6 gso/s, 0 ulp |
| `nk_dots_symmetric_e2m3_v128relaxed` |        32.8 gso/s, 0 ulp |        35.3 gso/s, 0 ulp |        38.2 gso/s, 0 ulp |
| __i8__                               | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_i8_serial`           |               4.60 gso/s |               11.1 gso/s |               13.8 gso/s |
| `nk_dots_symmetric_i8_serial`        |               6.57 gso/s |               17.2 gso/s |               18.4 gso/s |
| `nk_dots_packed_i8_v128relaxed`      |               47.1 gso/s |               48.8 gso/s |               52.7 gso/s |
| `nk_dots_symmetric_i8_v128relaxed`   |               43.1 gso/s |               41.8 gso/s |               52.7 gso/s |
| __u8__                               | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_u8_serial`           |               4.67 gso/s |               11.4 gso/s |               14.1 gso/s |
| `nk_dots_symmetric_u8_serial`        |               7.18 gso/s |               17.0 gso/s |               18.6 gso/s |
| `nk_dots_packed_u8_v128relaxed`      |               78.1 gso/s |               81.4 gso/s |               89.3 gso/s |
| `nk_dots_symmetric_u8_v128relaxed`   |               57.3 gso/s |               75.3 gso/s |               83.6 gso/s |
| __i4__                               | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_i4_serial`           |               7.06 gso/s |               18.5 gso/s |               19.7 gso/s |
| `nk_dots_symmetric_i4_serial`        |               5.43 gso/s |               12.5 gso/s |               13.1 gso/s |
| `nk_dots_packed_i4_v128relaxed`      |                  ? gso/s |                  ? gso/s |                  ? gso/s |
| `nk_dots_symmetric_i4_v128relaxed`   |                  ? gso/s |                  ? gso/s |                  ? gso/s |
| __u4__                               | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_u4_serial`           |               5.17 gso/s |               14.0 gso/s |               15.0 gso/s |
| `nk_dots_symmetric_u4_serial`        |               5.26 gso/s |               11.5 gso/s |               12.1 gso/s |
| `nk_dots_packed_u4_v128relaxed`      |                  ? gso/s |                  ? gso/s |                  ? gso/s |
| `nk_dots_symmetric_u4_v128relaxed`   |                  ? gso/s |                  ? gso/s |                  ? gso/s |
| __u1__                               | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_u1_serial`           |                  ? gso/s |                  ? gso/s |                  ? gso/s |
| `nk_dots_symmetric_u1_serial`        |                  ? gso/s |                  ? gso/s |                  ? gso/s |
| `nk_dots_packed_u1_v128relaxed`      |                603 gso/s |                736 gso/s |                872 gso/s |
| `nk_dots_symmetric_u1_v128relaxed`   |                182 gso/s |                359 gso/s |                497 gso/s |

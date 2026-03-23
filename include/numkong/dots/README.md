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

The Sapphire Rapids AMX backends for `bf16`, mini-floats, `i8`, and `u8` use Intel AMX's 8 tile registers (TMM0–TMM7), each 1 KB (16 rows × 64 bytes).
Convention: TMM0–1 hold A tiles, TMM2–3 hold B tiles, TMM4–7 are C accumulators — giving a 2×2 output tile (32×32 f32 results) per tile pass.
`TDPBF16PS tmm_c, tmm_a, tmm_b` performs a 16×16 outer product with 32 bf16 multiply-adds per cell (16×16×32 = 8,192 MACs per instruction).
Each A row contains 16 bf16 pairs interleaved as [a₀, a₁, a₀, a₁, ...] and B columns as [b₀, b₁, b₀, b₁, ...] — the hardware consumes two bf16 elements per slot, accumulating into f32.
`TDPBSSD tmm_c, tmm_a, tmm_b` does the same for i8: 64 bytes per row gives 16×16×64 = 16,384 i8 MACs per instruction.
i8 data is quad-interleaved: [a₀, a₁, a₂, a₃, a₀, a₁, a₂, a₃, ...] so the hardware can consume four i8 elements per 32-bit slot.
Tile configuration via `LDTILECFG` sets row counts and column byte-widths per tile — allows undersized tiles at matrix edges without masking.
Morton Z-curve ordering for tile traversal improves cache reuse when both A and B exceed L2.
This eliminates the explicit M×N×K loop nesting and register file pressure of vector ISAs — the entire dot-product reduction happens inside the tile instruction.

### SME Outer-Product Streaming

`nk_dots_packed_f32_smef64`, `nk_dots_packed_bf16_sme`, `nk_dots_packed_f64_smef64` use Arm's SME ZA tile array (up to 4 named tiles ZA0–ZA3 in 32-bit mode, each SVL×SVL elements).
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
Rows marked `🧩` use external BLAS or MKL baselines rather than NumKong kernels.
Each kernel runs for at least 20 seconds per configuration.
Benchmark threads are pinned to specific cores; on machines with heterogeneous core types (e.g., Apple P/E cores), only the fastest cores are used.
Workloads that significantly degrade CPU frequencies (Intel AMX, Apple SME) run in separate passes to avoid affecting throughput measurements of other kernels.

### Intel Sapphire Rapids

#### Native

| Kernel                               |                     256³ |                    1024³ |                    4096³ |
| :----------------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f64__                              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `dots_packed_f64_with_blas` 🧩        |       58.7 gso/s, 16 ulp |       73.1 gso/s, 58 ulp |     73.8 gso/s, 56.2 ulp |
| `dots_packed_f64_with_mkl` 🧩         |       59.9 gso/s, 16 ulp |       73.7 gso/s, 58 ulp |     73.3 gso/s, 56.2 ulp |
| `dots_symmetric_f64_with_blas` 🧩     |       50.8 gso/s, 13 ulp |       70.4 gso/s, 30 ulp |       74 gso/s, 50.8 ulp |
| `nk_dots_packed_f64_serial`          |       0.393 gso/s, 2 ulp |     0.489 gso/s, 4.6 ulp |     0.488 gso/s, 5.9 ulp |
| `nk_dots_symmetric_f64_serial`       |       0.346 gso/s, 2 ulp |     0.357 gso/s, 2.9 ulp |     0.574 gso/s, 3.9 ulp |
| `nk_dots_packed_f64_haswell`         |        5.56 gso/s, 0 ulp |        5.97 gso/s, 0 ulp |        6.15 gso/s, 0 ulp |
| `nk_dots_symmetric_f64_haswell`      |         5.1 gso/s, 0 ulp |        5.71 gso/s, 0 ulp |        11.5 gso/s, 0 ulp |
| `nk_dots_packed_f64_skylake`         |        8.05 gso/s, 0 ulp |        8.69 gso/s, 0 ulp |        8.93 gso/s, 0 ulp |
| `nk_dots_symmetric_f64_skylake`      |        7.52 gso/s, 0 ulp |        8.88 gso/s, 0 ulp |        17.6 gso/s, 0 ulp |
| __f32__                              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `dots_packed_f32_with_blas` 🧩        |        113 gso/s, 18 ulp |        139 gso/s, 30 ulp |       147 gso/s, 267 ulp |
| `dots_symmetric_f32_with_blas` 🧩     |       94.5 gso/s, 23 ulp |        126 gso/s, 39 ulp |       146 gso/s, 260 ulp |
| `nk_dots_packed_f32_serial`          |      9.89 gso/s, 5.3 ulp |     10.2 gso/s, 11.8 ulp |        10.1 gso/s, ? ulp |
| `nk_dots_symmetric_f32_serial`       |     6.30 gso/s, 11.1 ulp |     6.57 gso/s, 13.4 ulp |        6.53 gso/s, ? ulp |
| `nk_dots_packed_f32_haswell`         |        30.1 gso/s, 0 ulp |        31.6 gso/s, 0 ulp |        31.9 gso/s, 0 ulp |
| `nk_dots_symmetric_f32_haswell`      |        21.4 gso/s, 0 ulp |        26.2 gso/s, 0 ulp |        53.3 gso/s, 0 ulp |
| `nk_dots_packed_f32_skylake`         |          35 gso/s, 0 ulp |        38.6 gso/s, 0 ulp |        39.5 gso/s, 0 ulp |
| `nk_dots_symmetric_f32_skylake`      |        26.6 gso/s, 0 ulp |        30.5 gso/s, 0 ulp |          62 gso/s, 0 ulp |
| __bf16__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `dots_packed_bf16_with_mkl` 🧩        |         190 gso/s, 0 ulp |       531 gso/s, 0.7 ulp |       865 gso/s, 5.8 ulp |
| `nk_dots_packed_bf16_serial`         |       0.842 gso/s, 0 ulp |     0.824 gso/s, 0.5 ulp |     0.825 gso/s, 5.4 ulp |
| `nk_dots_symmetric_bf16_serial`      |       0.808 gso/s, 0 ulp |     0.759 gso/s, 0.9 ulp |      1.74 gso/s, 5.4 ulp |
| `nk_dots_packed_bf16_haswell`        |        57.4 gso/s, 0 ulp |      66.5 gso/s, 0.4 ulp |      67.1 gso/s, 4.5 ulp |
| `nk_dots_symmetric_bf16_haswell`     |        39.5 gso/s, 0 ulp |      50.8 gso/s, 0.3 ulp |       111 gso/s, 4.2 ulp |
| `nk_dots_packed_bf16_skylake`        |        73.8 gso/s, 0 ulp |      90.1 gso/s, 0.3 ulp |        90 gso/s, 3.7 ulp |
| `nk_dots_symmetric_bf16_skylake`     |        52.7 gso/s, 0 ulp |      58.5 gso/s, 0.3 ulp |       127 gso/s, 3.2 ulp |
| `nk_dots_packed_bf16_genoa`          |        64.1 gso/s, 0 ulp |      85.3 gso/s, 0.3 ulp |      90.3 gso/s, 3.5 ulp |
| `nk_dots_symmetric_bf16_genoa`       |        58.1 gso/s, 0 ulp |      61.3 gso/s, 0.5 ulp |       133 gso/s, 3.5 ulp |
| `nk_dots_packed_bf16_sapphireamx`    |         391 gso/s, 0 ulp |       531 gso/s, 0.7 ulp |       604 gso/s, 5.8 ulp |
| `nk_dots_symmetric_bf16_sapphireamx` |        81.6 gso/s, 0 ulp |       120 gso/s, 0.5 ulp |       124 gso/s, 5.8 ulp |
| __f16__                              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `dots_packed_f16_with_mkl` 🧩         |        123 gso/s, 17 ulp |        138 gso/s, 31 ulp |      138 gso/s, 39.5 ulp |
| `nk_dots_packed_f16_serial`          |       4.44 gso/s, 14 ulp |       4.42 gso/s, 40 ulp |      4.40 gso/s, 326 ulp |
| `nk_dots_symmetric_f16_serial`       |      3.66 gso/s, 8.9 ulp |       3.44 gso/s, 25 ulp |     5.06 gso/s, 55.6 ulp |
| `nk_dots_packed_f16_haswell`         |       63.4 gso/s, 12 ulp |       72.4 gso/s, 22 ulp |      71.8 gso/s, 374 ulp |
| `nk_dots_symmetric_f16_haswell`      |      39.9 gso/s, 7.7 ulp |       55.7 gso/s, 32 ulp |       127 gso/s, 486 ulp |
| `nk_dots_packed_f16_skylake`         |      74.3 gso/s, 7.3 ulp |       98.7 gso/s, 21 ulp |      85.4 gso/s, 138 ulp |
| `nk_dots_symmetric_f16_skylake`      |        53 gso/s, 5.9 ulp |       59.3 gso/s, 25 ulp |        133 gso/s, 32 ulp |
| __e5m2__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_e5m2_serial`         |        4.00 gso/s, 0 ulp |        4.79 gso/s, 0 ulp |        4.80 gso/s, 0 ulp |
| `nk_dots_symmetric_e5m2_serial`      |        3.90 gso/s, 0 ulp |        4.17 gso/s, 0 ulp |        5.06 gso/s, 0 ulp |
| `nk_dots_packed_e5m2_haswell`        |        38.6 gso/s, 0 ulp |        42.3 gso/s, 0 ulp |        41.4 gso/s, 0 ulp |
| `nk_dots_symmetric_e5m2_haswell`     |        19.2 gso/s, 0 ulp |        23.1 gso/s, 0 ulp |        22.1 gso/s, 0 ulp |
| `nk_dots_packed_e5m2_skylake`        |        45.2 gso/s, 0 ulp |        52.1 gso/s, 0 ulp |        54.1 gso/s, 0 ulp |
| `nk_dots_symmetric_e5m2_skylake`     |        25.0 gso/s, 0 ulp |        27.7 gso/s, 0 ulp |        28.8 gso/s, 0 ulp |
| `nk_dots_packed_e5m2_genoa`          |        44.0 gso/s, 0 ulp |        50.6 gso/s, 0 ulp |        49.5 gso/s, 0 ulp |
| `nk_dots_symmetric_e5m2_genoa`       |        30.4 gso/s, 0 ulp |        33.9 gso/s, 0 ulp |        34.6 gso/s, 0 ulp |
| `nk_dots_packed_e5m2_sapphireamx`    |         261 gso/s, 0 ulp |         419 gso/s, 0 ulp |         435 gso/s, 0 ulp |
| __e4m3__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_e4m3_serial`         |       0.433 gso/s, 0 ulp |       0.478 gso/s, 0 ulp |       0.415 gso/s, 0 ulp |
| `nk_dots_symmetric_e4m3_serial`      |       0.408 gso/s, 0 ulp |       0.376 gso/s, 0 ulp |       0.618 gso/s, 0 ulp |
| `nk_dots_packed_e4m3_haswell`        |        29.3 gso/s, 0 ulp |        30.9 gso/s, 0 ulp |        32.1 gso/s, 0 ulp |
| `nk_dots_symmetric_e4m3_haswell`     |        14.6 gso/s, 0 ulp |        15.8 gso/s, 0 ulp |        16.1 gso/s, 0 ulp |
| `nk_dots_packed_e4m3_skylake`        |        35.2 gso/s, 0 ulp |        38.5 gso/s, 0 ulp |        37.9 gso/s, 0 ulp |
| `nk_dots_symmetric_e4m3_skylake`     |        19.8 gso/s, 0 ulp |        21.5 gso/s, 0 ulp |        21.9 gso/s, 0 ulp |
| `nk_dots_packed_e4m3_genoa`          |        42.8 gso/s, 0 ulp |        50.0 gso/s, 0 ulp |        50.3 gso/s, 0 ulp |
| `nk_dots_symmetric_e4m3_genoa`       |        30.7 gso/s, 0 ulp |        33.5 gso/s, 0 ulp |        34.4 gso/s, 0 ulp |
| `nk_dots_packed_e4m3_sapphireamx`    |         260 gso/s, 0 ulp |         425 gso/s, 0 ulp |         423 gso/s, 0 ulp |
| __e3m2__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_e3m2_serial`         |        4.38 gso/s, 0 ulp |        4.14 gso/s, 0 ulp |        4.72 gso/s, 0 ulp |
| `nk_dots_symmetric_e3m2_serial`      |        3.28 gso/s, 0 ulp |        3.57 gso/s, 0 ulp |        3.96 gso/s, 0 ulp |
| `nk_dots_packed_e3m2_haswell`        |        31.6 gso/s, 0 ulp |        32.3 gso/s, 0 ulp |        33.8 gso/s, 0 ulp |
| `nk_dots_symmetric_e3m2_haswell`     |        29.4 gso/s, 0 ulp |        32.7 gso/s, 0 ulp |        31.4 gso/s, 0 ulp |
| `nk_dots_packed_e3m2_skylake`        |        39.3 gso/s, 0 ulp |        44.3 gso/s, 0 ulp |        44.4 gso/s, 0 ulp |
| `nk_dots_symmetric_e3m2_skylake`     |        39.2 gso/s, 0 ulp |        46.0 gso/s, 0 ulp |        47.1 gso/s, 0 ulp |
| `nk_dots_packed_e3m2_sapphireamx`    |         276 gso/s, 0 ulp |         490 gso/s, 0 ulp |         483 gso/s, 0 ulp |
| `nk_dots_symmetric_e3m2_sapphireamx` |        63.3 gso/s, 0 ulp |         104 gso/s, 0 ulp |        91.6 gso/s, 0 ulp |
| __e2m3__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_e2m3_serial`         |        4.79 gso/s, 0 ulp |        5.06 gso/s, 0 ulp |        4.35 gso/s, 0 ulp |
| `nk_dots_symmetric_e2m3_serial`      |        3.24 gso/s, 0 ulp |        3.74 gso/s, 0 ulp |        4.05 gso/s, 0 ulp |
| `nk_dots_packed_e2m3_haswell`        |        59.0 gso/s, 0 ulp |        62.5 gso/s, 0 ulp |        65.6 gso/s, 0 ulp |
| `nk_dots_symmetric_e2m3_haswell`     |        50.8 gso/s, 0 ulp |        62.4 gso/s, 0 ulp |        64.6 gso/s, 0 ulp |
| `nk_dots_packed_e2m3_skylake`        |        70.2 gso/s, 0 ulp |        81.9 gso/s, 0 ulp |        85.5 gso/s, 0 ulp |
| `nk_dots_symmetric_e2m3_skylake`     |        65.5 gso/s, 0 ulp |        84.2 gso/s, 0 ulp |        85.4 gso/s, 0 ulp |
| `nk_dots_packed_e2m3_sapphireamx`    |         426 gso/s, 0 ulp |       1,192 gso/s, 0 ulp |       1,072 gso/s, 0 ulp |
| `nk_dots_symmetric_e2m3_sapphireamx` |        96.3 gso/s, 0 ulp |         209 gso/s, 0 ulp |         195 gso/s, 0 ulp |
| `nk_dots_packed_e2m3_alder`          |        72.2 gso/s, 0 ulp |        78.3 gso/s, 0 ulp |        85.8 gso/s, 0 ulp |
| `nk_dots_symmetric_e2m3_alder`       |        63.3 gso/s, 0 ulp |        73.2 gso/s, 0 ulp |        78.6 gso/s, 0 ulp |
| __i8__                               | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `dots_packed_i8u8_with_mkl` 🧩        |                250 gso/s |                627 gso/s |              1,670 gso/s |
| `nk_dots_packed_i8_serial`           |               6.44 gso/s |               6.62 gso/s |               7.44 gso/s |
| `nk_dots_symmetric_i8_serial`        |               2.93 gso/s |               2.99 gso/s |               5.83 gso/s |
| `nk_dots_packed_i8_haswell`          |               87.7 gso/s |                104 gso/s |                108 gso/s |
| `nk_dots_symmetric_i8_haswell`       |                 64 gso/s |               80.9 gso/s |                173 gso/s |
| `nk_dots_packed_i8_icelake`          |                191 gso/s |                326 gso/s |                410 gso/s |
| `nk_dots_symmetric_i8_icelake`       |               79.2 gso/s |                303 gso/s |                760 gso/s |
| `nk_dots_packed_i8_sapphireamx`      |                547 gso/s |              1,610 gso/s |              1,300 gso/s |
| `nk_dots_symmetric_i8_sapphireamx`   |                112 gso/s |                266 gso/s |                221 gso/s |
| `nk_dots_packed_i8_alder`            |                180 gso/s |                229 gso/s |                270 gso/s |
| `nk_dots_symmetric_i8_alder`         |                108 gso/s |                218 gso/s |                263 gso/s |
| __u8__                               | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_u8_serial`           |               7.45 gso/s |               7.79 gso/s |               7.88 gso/s |
| `nk_dots_symmetric_u8_serial`        |               2.81 gso/s |               2.91 gso/s |               5.35 gso/s |
| `nk_dots_packed_u8_haswell`          |                 88 gso/s |                102 gso/s |                107 gso/s |
| `nk_dots_symmetric_u8_haswell`       |               64.3 gso/s |               79.8 gso/s |                181 gso/s |
| `nk_dots_packed_u8_icelake`          |                194 gso/s |                329 gso/s |                402 gso/s |
| `nk_dots_symmetric_u8_icelake`       |               83.9 gso/s |                300 gso/s |                755 gso/s |
| `nk_dots_packed_u8_sapphireamx`      |                550 gso/s |              1,680 gso/s |              1,330 gso/s |
| `nk_dots_symmetric_u8_sapphireamx`   |                113 gso/s |                270 gso/s |                223 gso/s |
| `nk_dots_packed_u8_alder`            |                181 gso/s |                230 gso/s |                266 gso/s |
| `nk_dots_symmetric_u8_alder`         |                108 gso/s |                216 gso/s |                257 gso/s |
| __i4__                               | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_i4_serial`           |               2.43 gso/s |               2.43 gso/s |               2.24 gso/s |
| `nk_dots_symmetric_i4_serial`        |               2.26 gso/s |               2.13 gso/s |               4.44 gso/s |
| `nk_dots_packed_i4_icelake`          |                135 gso/s |                211 gso/s |                254 gso/s |
| `nk_dots_symmetric_i4_icelake`       |               78.7 gso/s |                252 gso/s |                581 gso/s |
| __u4__                               | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_u4_serial`           |               3.27 gso/s |               3.37 gso/s |               3.33 gso/s |
| `nk_dots_symmetric_u4_serial`        |               3.02 gso/s |               3.06 gso/s |               6.13 gso/s |
| `nk_dots_packed_u4_icelake`          |                152 gso/s |                302 gso/s |                387 gso/s |
| `nk_dots_symmetric_u4_icelake`       |               97.3 gso/s |                311 gso/s |                697 gso/s |
| __u1__                               | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_u1_haswell`          |                225 gso/s |                261 gso/s |                344 gso/s |
| `nk_dots_symmetric_u1_haswell`       |                122 gso/s |                277 gso/s |                756 gso/s |
| `nk_dots_packed_u1_icelake`          |                196 gso/s |                750 gso/s |              1,390 gso/s |
| `nk_dots_symmetric_u1_icelake`       |                171 gso/s |                661 gso/s |              2,500 gso/s |

#### WASM

Measured with Wasmtime v42 (Cranelift backend).

| Kernel                               |                     256³ |                    1024³ |                    4096³ |
| :----------------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f64__                              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_f64_serial`          |     0.947 gso/s, 3.4 ulp |     0.969 gso/s, 2.4 ulp |       0.969 gso/s, 0 ulp |
| `nk_dots_symmetric_f64_serial`       |     0.957 gso/s, 3.7 ulp |      1.11 gso/s, 2.5 ulp |        1.16 gso/s, 0 ulp |
| `nk_dots_packed_f64_v128relaxed`     |     2.73 gso/s, 23.6 ulp |     2.79 gso/s, 32.5 ulp |      2.81 gso/s, 3.9 ulp |
| `nk_dots_symmetric_f64_v128relaxed`  |     2.01 gso/s, 21.6 ulp |     2.55 gso/s, 41.2 ulp |      2.77 gso/s, 2.9 ulp |
| __f32__                              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_f32_serial`          |     4.27 gso/s, 14.6 ulp |     4.35 gso/s, 28.6 ulp |     4.47 gso/s, 25.3 ulp |
| `nk_dots_symmetric_f32_serial`       |     3.13 gso/s, 11.5 ulp |     5.09 gso/s, 34.8 ulp |     5.78 gso/s, 44.7 ulp |
| `nk_dots_packed_f32_v128relaxed`     |     10.4 gso/s, 12.9 ulp |     10.6 gso/s, 26.5 ulp |     10.9 gso/s, 39.7 ulp |
| `nk_dots_symmetric_f32_v128relaxed`  |     3.73 gso/s, 10.3 ulp |     6.27 gso/s, 28.6 ulp |     7.43 gso/s, 76.2 ulp |
| __bf16__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_bf16_serial`         |        4.33 gso/s, 0 ulp |      4.46 gso/s, 0.4 ulp |      4.45 gso/s, 9.5 ulp |
| `nk_dots_symmetric_bf16_serial`      |        3.76 gso/s, 0 ulp |      6.36 gso/s, 0.5 ulp |      7.43 gso/s, 4.9 ulp |
| `nk_dots_packed_bf16_v128relaxed`    |        23.2 gso/s, 0 ulp |      24.5 gso/s, 0.4 ulp |      24.9 gso/s, 6.8 ulp |
| `nk_dots_symmetric_bf16_v128relaxed` |        4.92 gso/s, 0 ulp |      10.5 gso/s, 0.5 ulp |      13.7 gso/s, 4.9 ulp |
| __f16__                              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_f16_serial`          |       4.33 gso/s, 26 ulp |       4.46 gso/s, 26 ulp |       4.45 gso/s, 26 ulp |
| `nk_dots_symmetric_f16_serial`       |       3.76 gso/s, 28 ulp |       6.36 gso/s, 28 ulp |       7.43 gso/s, 28 ulp |
| `nk_dots_packed_f16_v128relaxed`     |       7.39 gso/s, 27 ulp |       7.36 gso/s, 27 ulp |       7.45 gso/s, 27 ulp |
| `nk_dots_symmetric_f16_v128relaxed`  |       3.70 gso/s, 28 ulp |       3.83 gso/s, 28 ulp |       3.87 gso/s, 28 ulp |
| __e5m2__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_e5m2_serial`         |        2.63 gso/s, 0 ulp |        2.69 gso/s, 0 ulp |        2.70 gso/s, 0 ulp |
| `nk_dots_symmetric_e5m2_serial`      |        1.62 gso/s, 0 ulp |        2.04 gso/s, 0 ulp |        2.16 gso/s, 0 ulp |
| `nk_dots_packed_e5m2_v128relaxed`    |        6.25 gso/s, 0 ulp |        6.50 gso/s, 0 ulp |        6.55 gso/s, 0 ulp |
| `nk_dots_symmetric_e5m2_v128relaxed` |        3.37 gso/s, 0 ulp |        5.23 gso/s, 0 ulp |        6.06 gso/s, 0 ulp |
| __e4m3__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_e4m3_serial`         |       0.348 gso/s, 0 ulp |       0.345 gso/s, 0 ulp |       0.345 gso/s, 0 ulp |
| `nk_dots_symmetric_e4m3_serial`      |       0.321 gso/s, 0 ulp |       0.340 gso/s, 0 ulp |       0.345 gso/s, 0 ulp |
| `nk_dots_packed_e4m3_v128relaxed`    |        4.80 gso/s, 0 ulp |        4.92 gso/s, 0 ulp |        4.96 gso/s, 0 ulp |
| `nk_dots_symmetric_e4m3_v128relaxed` |        2.85 gso/s, 0 ulp |        4.17 gso/s, 0 ulp |        4.62 gso/s, 0 ulp |
| __e2m3__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_e2m3_serial`         |        2.63 gso/s, 0 ulp |        2.69 gso/s, 0 ulp |        2.71 gso/s, 0 ulp |
| `nk_dots_symmetric_e2m3_serial`      |        1.62 gso/s, 0 ulp |        2.06 gso/s, 0 ulp |        2.14 gso/s, 0 ulp |
| `nk_dots_packed_e2m3_v128relaxed`    |        17.2 gso/s, 0 ulp |        18.2 gso/s, 0 ulp |        18.7 gso/s, 0 ulp |
| `nk_dots_symmetric_e2m3_v128relaxed` |        5.35 gso/s, 0 ulp |        11.6 gso/s, 0 ulp |        16.3 gso/s, 0 ulp |
| __i8__                               | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_i8_serial`           |               4.40 gso/s |               4.54 gso/s |               4.73 gso/s |
| `nk_dots_symmetric_i8_serial`        |               2.74 gso/s |               3.89 gso/s |               4.29 gso/s |
| `nk_dots_packed_i8_v128relaxed`      |               36.5 gso/s |               38.5 gso/s |               41.1 gso/s |
| `nk_dots_symmetric_i8_v128relaxed`   |               29.2 gso/s |               36.3 gso/s |               39.2 gso/s |
| __u8__                               | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_u8_serial`           |               4.94 gso/s |               5.14 gso/s |               4.88 gso/s |
| `nk_dots_symmetric_u8_serial`        |               2.74 gso/s |               3.94 gso/s |               4.40 gso/s |
| `nk_dots_packed_u8_v128relaxed`      |               35.2 gso/s |               37.7 gso/s |               40.5 gso/s |
| `nk_dots_symmetric_u8_v128relaxed`   |               21.0 gso/s |               26.6 gso/s |               28.6 gso/s |
| __i4__                               | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_i4_serial`           |               6.34 gso/s |               6.40 gso/s |               6.59 gso/s |
| `nk_dots_symmetric_i4_serial`        |               2.70 gso/s |               3.76 gso/s |               4.13 gso/s |
| `nk_dots_packed_i4_v128relaxed`      |               9.81 gso/s |               10.3 gso/s |               10.4 gso/s |
| `nk_dots_symmetric_i4_v128relaxed`   |               4.95 gso/s |               15.6 gso/s |               32.8 gso/s |
| __u4__                               | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_u4_serial`           |               5.61 gso/s |               5.76 gso/s |               5.79 gso/s |
| `nk_dots_symmetric_u4_serial`        |               3.01 gso/s |               4.34 gso/s |               4.94 gso/s |
| `nk_dots_packed_u4_v128relaxed`      |               58.6 gso/s |               71.0 gso/s |               76.5 gso/s |
| `nk_dots_symmetric_u4_v128relaxed`   |               6.97 gso/s |               21.9 gso/s |               46.7 gso/s |
| __u1__                               | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_u1_serial`           |               96.2 gso/s |                143 gso/s |                151 gso/s |
| `nk_dots_packed_u1_v128relaxed`      |                166 gso/s |                280 gso/s |                294 gso/s |
| `nk_dots_symmetric_u1_serial`        |               7.42 gso/s |               27.9 gso/s |               87.3 gso/s |
| `nk_dots_symmetric_u1_v128relaxed`   |               7.35 gso/s |               27.5 gso/s |               81.9 gso/s |

### Apple M4

#### Native

| Kernel                             |                     256³ |                    1024³ |                    4096³ |
| :--------------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f64__                            | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_f64_serial`        |        1.82 gso/s, 3 ulp |        1.81 gso/s, 5 ulp |        1.82 gso/s, 6 ulp |
| `nk_dots_symmetric_f64_serial`     |        1.40 gso/s, 0 ulp |        1.42 gso/s, 0 ulp |        1.42 gso/s, 0 ulp |
| `nk_dots_packed_f64_neon`          |        5.62 gso/s, 0 ulp |        5.48 gso/s, 0 ulp |        5.21 gso/s, 0 ulp |
| `nk_dots_symmetric_f64_neon`       |        4.46 gso/s, 0 ulp |        4.94 gso/s, 0 ulp |        5.71 gso/s, 0 ulp |
| `nk_dots_packed_f64_smef64`        |      13.9 gso/s, 1.5 ulp |      12.1 gso/s, 1.1 ulp |      12.9 gso/s, 0.9 ulp |
| `nk_dots_symmetric_f64_smef64`     |      5.18 gso/s, 1.5 ulp |      5.06 gso/s, 1.2 ulp |      4.46 gso/s, 1.1 ulp |
| __f32__                            | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_f32_serial`        |       10.4 gso/s, 19 ulp |       10.6 gso/s, 30 ulp |      11.2 gso/s, 725 ulp |
| `nk_dots_symmetric_f32_serial`     |      8.34 gso/s, 3.1 ulp |     8.64 gso/s, 12.8 ulp |     8.96 gso/s, 39.9 ulp |
| `nk_dots_packed_f32_neon`          |        40.7 gso/s, 0 ulp |        40.1 gso/s, 0 ulp |        41.5 gso/s, 0 ulp |
| `nk_dots_symmetric_f32_neon`       |      10.0 gso/s, 4.6 ulp |     10.2 gso/s, 17.7 ulp |       10.1 gso/s, 59 ulp |
| `nk_dots_packed_f32_smef64`        |        42.8 gso/s, 0 ulp |       57.8 gso/s, 15 ulp |        50.7 gso/s, 0 ulp |
| `nk_dots_symmetric_f32_smef64`     |      15.0 gso/s, 4.3 ulp |     15.4 gso/s, 19.0 ulp |        10.2 gso/s, 0 ulp |
| __bf16__                           | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_bf16_serial`       |      17.0 gso/s, 0.1 ulp |      17.6 gso/s, 0.5 ulp |        17.2 gso/s, 5 ulp |
| `nk_dots_symmetric_bf16_serial`    |     13.5 gso/s, 0.01 ulp |      13.4 gso/s, 0.7 ulp |      16.6 gso/s, 115 ulp |
| `nk_dots_packed_bf16_neonbfdot`    |        57.7 gso/s, 0 ulp |      58.1 gso/s, 0.5 ulp |      58.8 gso/s, 7.2 ulp |
| `nk_dots_symmetric_bf16_neonbfdot` |        38.9 gso/s, 0 ulp |      39.1 gso/s, 0.5 ulp |        37.7 gso/s, ? ulp |
| `nk_dots_packed_bf16_sme`          |         437 gso/s, 0 ulp |       571 gso/s, 4.2 ulp |       507 gso/s, 3.8 ulp |
| `nk_dots_symmetric_bf16_sme`       |      106 gso/s, 0.07 ulp |      94.2 gso/s, 1.2 ulp |      90.0 gso/s, 1.8 ulp |
| __f16__                            | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_f16_serial`        |      13.1 gso/s, 204 ulp |       13.4 gso/s, 36 ulp |      14.1 gso/s, 326 ulp |
| `nk_dots_symmetric_f16_serial`     |       21.7 gso/s, 13 ulp |     18.0 gso/s, 24.6 ulp |      26.2 gso/s, 506 ulp |
| `nk_dots_packed_f16_neonhalf`      |     76.6 gso/s, 16.8 ulp |     75.5 gso/s, 25.5 ulp |      82.3 gso/s, 618 ulp |
| `nk_dots_symmetric_f16_neonhalf`   |     20.0 gso/s, 12.1 ulp |     20.4 gso/s, 25.0 ulp |      20.7 gso/s, 506 ulp |
| `nk_dots_packed_f16_neonfhm`       |      111 gso/s, 16.7 ulp |      110 gso/s, 25.5 ulp |       108 gso/s, 618 ulp |
| `nk_dots_symmetric_f16_neonfhm`    |     35.3 gso/s, 12.1 ulp |     36.4 gso/s, 25.0 ulp |      36.7 gso/s, 506 ulp |
| `nk_dots_packed_f16_sme`           |      461 gso/s, 14.8 ulp |      484 gso/s, 28.2 ulp |      476 gso/s, 28.2 ulp |
| `nk_dots_symmetric_f16_sme`        |     98.6 gso/s, 12.1 ulp |     95.2 gso/s, 23.8 ulp |     88.4 gso/s, 24.4 ulp |
| __e5m2__                           | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_e5m2_serial`       |        12.1 gso/s, 0 ulp |        11.8 gso/s, 0 ulp |        13.7 gso/s, 0 ulp |
| `nk_dots_symmetric_e5m2_serial`    |        7.88 gso/s, 0 ulp |        8.23 gso/s, 0 ulp |        8.31 gso/s, 0 ulp |
| `nk_dots_packed_e5m2_neonfhm`      |        95.2 gso/s, 0 ulp |        98.3 gso/s, 0 ulp |        99.3 gso/s, 0 ulp |
| `nk_dots_symmetric_e5m2_neonfhm`   |        61.0 gso/s, 0 ulp |        67.7 gso/s, 0 ulp |        76.3 gso/s, 0 ulp |
| `nk_dots_packed_e5m2_sme`          |         327 gso/s, 0 ulp |       1,120 gso/s, 0 ulp |         552 gso/s, 0 ulp |
| `nk_dots_symmetric_e5m2_sme`       |        70.4 gso/s, 0 ulp |        66.3 gso/s, 0 ulp |         120 gso/s, 0 ulp |
| __e4m3__                           | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_e4m3_serial`       |        1.19 gso/s, 0 ulp |        1.20 gso/s, 0 ulp |        1.20 gso/s, 0 ulp |
| `nk_dots_symmetric_e4m3_serial`    |        1.23 gso/s, 0 ulp |        1.24 gso/s, 0 ulp |   1.24 gso/s, 0.0001 ulp |
| `nk_dots_packed_e4m3_neonfhm`      |        28.7 gso/s, 0 ulp |        29.6 gso/s, 0 ulp |        26.7 gso/s, 0 ulp |
| `nk_dots_symmetric_e4m3_neonfhm`   |        36.1 gso/s, 0 ulp |        37.4 gso/s, 0 ulp |        37.6 gso/s, 0 ulp |
| `nk_dots_packed_e4m3_sme`          |         168 gso/s, 0 ulp |         219 gso/s, 0 ulp |         181 gso/s, 0 ulp |
| `nk_dots_symmetric_e4m3_sme`       |        25.6 gso/s, 0 ulp |        23.2 gso/s, 0 ulp |        35.1 gso/s, 0 ulp |
| __e3m2__                           | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_e3m2_serial`       |        12.0 gso/s, 0 ulp |        11.7 gso/s, 0 ulp |        11.4 gso/s, 0 ulp |
| `nk_dots_symmetric_e3m2_serial`    |        7.56 gso/s, 0 ulp |        7.75 gso/s, 0 ulp |        7.76 gso/s, 0 ulp |
| `nk_dots_packed_e3m2_sme`          |        15.4 gso/s, 0 ulp |        11.9 gso/s, 0 ulp |        12.3 gso/s, 0 ulp |
| `nk_dots_symmetric_e3m2_sme`       |        1.59 gso/s, 0 ulp |        1.67 gso/s, 0 ulp |        1.79 gso/s, 0 ulp |
| __e2m3__                           | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_e2m3_serial`       |        13.3 gso/s, 0 ulp |        11.5 gso/s, 0 ulp |        14.6 gso/s, 0 ulp |
| `nk_dots_symmetric_e2m3_serial`    |        7.85 gso/s, 0 ulp |        8.23 gso/s, 0 ulp |        8.30 gso/s, 0 ulp |
| `nk_dots_packed_e2m3_sme`          |         518 gso/s, 0 ulp |         740 gso/s, 0 ulp |       1,017 gso/s, 0 ulp |
| `nk_dots_symmetric_e2m3_sme`       |        65.1 gso/s, 0 ulp |        79.6 gso/s, 0 ulp |        98.4 gso/s, 0 ulp |
| __i8__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_i8_serial`         |               17.0 gso/s |               17.3 gso/s |               17.3 gso/s |
| `nk_dots_symmetric_i8_serial`      |               13.9 gso/s |               14.1 gso/s |               14.3 gso/s |
| `nk_dots_packed_i8_neonsdot`       |                301 gso/s |                365 gso/s |                389 gso/s |
| `nk_dots_symmetric_i8_neonsdot`    |               72.8 gso/s |               79.7 gso/s |               79.9 gso/s |
| `nk_dots_packed_i8_sme`            |                854 gso/s |                904 gso/s |              1,206 gso/s |
| `nk_dots_symmetric_i8_sme`         |                156 gso/s |                195 gso/s |                154 gso/s |
| __u8__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_u8_serial`         |               15.4 gso/s |               16.0 gso/s |               16.0 gso/s |
| `nk_dots_symmetric_u8_serial`      |               15.8 gso/s |               16.2 gso/s |               16.2 gso/s |
| `nk_dots_packed_u8_neonsdot`       |                329 gso/s |                401 gso/s |                387 gso/s |
| `nk_dots_symmetric_u8_neonsdot`    |               72.5 gso/s |               79.8 gso/s |               80.4 gso/s |
| `nk_dots_packed_u8_sme`            |                907 gso/s |                970 gso/s |              1,224 gso/s |
| `nk_dots_symmetric_u8_sme`         |                167 gso/s |                204 gso/s |                163 gso/s |
| __i4__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_i4_serial`         |               17.3 gso/s |               17.6 gso/s |               18.9 gso/s |
| `nk_dots_symmetric_i4_serial`      |               15.1 gso/s |               15.3 gso/s |               15.3 gso/s |
| `nk_dots_packed_i4_neonsdot`       |                255 gso/s |                278 gso/s |                283 gso/s |
| `nk_dots_symmetric_i4_neonsdot`    |                117 gso/s |                144 gso/s |                149 gso/s |
| `nk_dots_packed_i4_sme`            |              1,028 gso/s |                960 gso/s |              1,105 gso/s |
| `nk_dots_symmetric_i4_sme`         |                310 gso/s |                249 gso/s |                393 gso/s |
| __u4__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_u4_serial`         |               19.1 gso/s |               19.7 gso/s |               19.4 gso/s |
| `nk_dots_symmetric_u4_serial`      |               16.1 gso/s |               16.4 gso/s |               16.1 gso/s |
| `nk_dots_packed_u4_neonsdot`       |                290 gso/s |                320 gso/s |                328 gso/s |
| `nk_dots_symmetric_u4_neonsdot`    |                125 gso/s |                144 gso/s |                149 gso/s |
| `nk_dots_packed_u4_sme`            |              1,230 gso/s |              1,013 gso/s |              1,246 gso/s |
| `nk_dots_symmetric_u4_sme`         |                340 gso/s |                285 gso/s |                401 gso/s |
| __u1__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_u1_serial`         |                342 gso/s |                429 gso/s |                447 gso/s |
| `nk_dots_symmetric_u1_serial`      |                226 gso/s |                342 gso/s |                357 gso/s |
| `nk_dots_packed_u1_neon`           |                795 gso/s |                931 gso/s |                983 gso/s |
| `nk_dots_symmetric_u1_neon`        |                290 gso/s |                483 gso/s |                561 gso/s |
| `nk_dots_packed_u1_smebi32`        |                998 gso/s |              4,251 gso/s |              7,369 gso/s |
| `nk_dots_symmetric_u1_smebi32`     |               53.7 gso/s |                278 gso/s |                548 gso/s |

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
| `nk_dots_packed_e5m2_v128relaxed`    |        11.5 gso/s, ? ulp |        11.6 gso/s, ? ulp |        11.6 gso/s, ? ulp |
| `nk_dots_symmetric_e5m2_v128relaxed` |        11.7 gso/s, 0 ulp |        12.2 gso/s, 0 ulp |        12.3 gso/s, 0 ulp |
| __e4m3__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_e4m3_serial`         |       0.340 gso/s, 0 ulp |       0.717 gso/s, 0 ulp |       0.864 gso/s, 0 ulp |
| `nk_dots_symmetric_e4m3_serial`      |       0.331 gso/s, 0 ulp |       0.822 gso/s, 0 ulp |       0.874 gso/s, 0 ulp |
| `nk_dots_packed_e4m3_v128relaxed`    |        9.44 gso/s, ? ulp |        9.57 gso/s, ? ulp |        9.58 gso/s, ? ulp |
| `nk_dots_symmetric_e4m3_v128relaxed` |        9.46 gso/s, 0 ulp |        10.0 gso/s, 0 ulp |        10.2 gso/s, 0 ulp |
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
| `nk_dots_packed_u8_v128relaxed`      |               24.5 gso/s |               25.3 gso/s |               25.6 gso/s |
| `nk_dots_symmetric_u8_v128relaxed`   |               20.3 gso/s |               23.5 gso/s |               24.4 gso/s |
| __i4__                               | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_i4_serial`           |               7.06 gso/s |               18.5 gso/s |               19.7 gso/s |
| `nk_dots_symmetric_i4_serial`        |               5.43 gso/s |               12.5 gso/s |               13.1 gso/s |
| `nk_dots_packed_i4_v128relaxed`      |               26.2 gso/s |               24.3 gso/s |               22.8 gso/s |
| `nk_dots_symmetric_i4_v128relaxed`   |               54.2 gso/s |               67.9 gso/s |               73.4 gso/s |
| __u4__                               | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_u4_serial`           |               5.17 gso/s |               14.0 gso/s |               15.0 gso/s |
| `nk_dots_symmetric_u4_serial`        |               5.26 gso/s |               11.5 gso/s |               12.1 gso/s |
| `nk_dots_packed_u4_v128relaxed`      |               78.0 gso/s |               82.7 gso/s |               84.0 gso/s |
| `nk_dots_symmetric_u4_v128relaxed`   |               66.9 gso/s |               81.4 gso/s |               85.3 gso/s |
| __u1__                               | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_u1_serial`           |                209 gso/s |                250 gso/s |                270 gso/s |
| `nk_dots_symmetric_u1_serial`        |                165 gso/s |                314 gso/s |                395 gso/s |
| `nk_dots_packed_u1_v128relaxed`      |                603 gso/s |                736 gso/s |                872 gso/s |
| `nk_dots_symmetric_u1_v128relaxed`   |                182 gso/s |                359 gso/s |                497 gso/s |

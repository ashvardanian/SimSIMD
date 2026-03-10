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
Accuracy is reported as ULP (units in last place), the number of representable floating-point values between the result and the exact answer.
Each kernel runs for at least 20 seconds per configuration.
Benchmark threads are pinned to specific cores; on machines with heterogeneous core types (e.g., Apple P/E cores), only the fastest cores are used.
Workloads that significantly degrade CPU frequencies (Intel AMX, Apple SME) run in separate passes to avoid affecting throughput measurements of other kernels.

### Intel Sapphire Rapids

#### Native

| Kernel                               |                     256³ |                    1024³ |                    4096³ |
| :----------------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f64__                              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_f64_with_blas`       |       62.5 gso/s, 19 ulp |       69.4 gso/s, 38 ulp |        50.7 gso/s, ? ulp |
| `nk_dots_symmetric_f64_with_blas`    |       53.0 gso/s, 32 ulp |       65.4 gso/s, 35 ulp |       49.7 gso/s, 51 ulp |
| `nk_dots_packed_f64_serial`          |     0.596 gso/s, 2.7 ulp |     0.699 gso/s, 4.6 ulp |       0.797 gso/s, 6 ulp |
| `nk_dots_symmetric_f64_serial`       |     0.497 gso/s, 3.6 ulp |     0.583 gso/s, 3.3 ulp |     0.857 gso/s, 2.3 ulp |
| `nk_dots_packed_f64_haswell`         |        5.95 gso/s, 0 ulp |        5.98 gso/s, 0 ulp |        3.50 gso/s, 0 ulp |
| `nk_dots_symmetric_f64_haswell`      |        5.47 gso/s, 0 ulp |        5.83 gso/s, 0 ulp |        11.5 gso/s, 0 ulp |
| `nk_dots_packed_f64_skylake`         |        8.57 gso/s, 0 ulp |        8.70 gso/s, 0 ulp |        8.74 gso/s, 0 ulp |
| `nk_dots_symmetric_f64_skylake`      |        7.57 gso/s, 0 ulp |        8.21 gso/s, 0 ulp |        15.3 gso/s, 0 ulp |
| __f32__                              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_f32_with_blas`       |       117 gso/s, 128 ulp |       137 gso/s, 328 ulp |         101 gso/s, ? ulp |
| `nk_dots_symmetric_f32_with_blas`    |       97.9 gso/s, 24 ulp |        125 gso/s, 37 ulp |      98.1 gso/s, 260 ulp |
| `nk_dots_packed_f32_serial`          |       17.0 gso/s, 19 ulp |       17.3 gso/s, 30 ulp |      17.5 gso/s, 725 ulp |
| `nk_dots_symmetric_f32_serial`       |       4.26 gso/s, 20 ulp |       4.36 gso/s, 29 ulp |       6.85 gso/s, 44 ulp |
| `nk_dots_packed_f32_haswell`         |        30.9 gso/s, 0 ulp |        32.0 gso/s, 0 ulp |        22.9 gso/s, 0 ulp |
| `nk_dots_symmetric_f32_haswell`      |        16.5 gso/s, 0 ulp |        18.4 gso/s, 0 ulp |        35.9 gso/s, 0 ulp |
| `nk_dots_packed_f32_skylake`         |        37.3 gso/s, 0 ulp |        39.7 gso/s, 0 ulp |        39.6 gso/s, 0 ulp |
| `nk_dots_symmetric_f32_skylake`      |        24.9 gso/s, 0 ulp |        27.7 gso/s, 0 ulp |        54.7 gso/s, ? ulp |
| __bf16__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_bf16_serial`         |      1.18 gso/s, 0.1 ulp |      1.19 gso/s, 0.5 ulp |        1.22 gso/s, 5 ulp |
| `nk_dots_symmetric_bf16_serial`      |        1.17 gso/s, 0 ulp |      1.17 gso/s, 0.6 ulp |      1.51 gso/s, 2.2 ulp |
| `nk_dots_packed_bf16_haswell`        |        57.4 gso/s, 0 ulp |      67.3 gso/s, 0.4 ulp |      43.0 gso/s, 4.5 ulp |
| `nk_dots_symmetric_bf16_haswell`     |        31.1 gso/s, 0 ulp |      37.1 gso/s, 0.3 ulp |      75.7 gso/s, 1.9 ulp |
| `nk_dots_packed_bf16_skylake`        |        78.4 gso/s, 0 ulp |      89.1 gso/s, 0.3 ulp |      91.4 gso/s, 3.7 ulp |
| `nk_dots_symmetric_bf16_skylake`     |        40.0 gso/s, 0 ulp |      52.3 gso/s, 0.3 ulp |       102 gso/s, 1.8 ulp |
| `nk_dots_packed_bf16_genoa`          |        66.8 gso/s, 0 ulp |      85.0 gso/s, 0.3 ulp |        49.3 gso/s, ? ulp |
| `nk_dots_symmetric_bf16_genoa`       |        54.0 gso/s, 0 ulp |      67.7 gso/s, 0.4 ulp |      95.0 gso/s, 1.0 ulp |
| `nk_dots_packed_bf16_sapphireamx`    |         267 gso/s, 0 ulp |       389 gso/s, 0.6 ulp |       628 gso/s, 5.8 ulp |
| `nk_dots_symmetric_bf16_sapphireamx` |        57.1 gso/s, 0 ulp |      90.3 gso/s, 0.6 ulp |       115 gso/s, 5.8 ulp |
| __f16__                              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_f16_serial`          |      8.26 gso/s, 204 ulp |       8.22 gso/s, 36 ulp |      8.60 gso/s, 326 ulp |
| `nk_dots_symmetric_f16_serial`       |       4.21 gso/s, 13 ulp |       4.30 gso/s, 29 ulp |       7.30 gso/s, 35 ulp |
| `nk_dots_packed_f16_haswell`         |      64.0 gso/s, 437 ulp |       73.1 gso/s, 24 ulp |      43.8 gso/s, 374 ulp |
| `nk_dots_symmetric_f16_haswell`      |       32.1 gso/s, 10 ulp |       44.4 gso/s, 37 ulp |       88.8 gso/s, 22 ulp |
| `nk_dots_packed_f16_skylake`         |      76.3 gso/s, 8.7 ulp |       93.4 gso/s, 17 ulp |      87.7 gso/s, 138 ulp |
| `nk_dots_symmetric_f16_skylake`      |      44.4 gso/s, 9.0 ulp |       58.3 gso/s, 20 ulp |        119 gso/s, 21 ulp |
| __e5m2__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_e5m2_serial`         |       0.537 gso/s, 0 ulp |       0.537 gso/s, 0 ulp |       0.297 gso/s, 0 ulp |
| `nk_dots_symmetric_e5m2_serial`      |       0.377 gso/s, 0 ulp |       0.378 gso/s, 0 ulp |       0.687 gso/s, 0 ulp |
| `nk_dots_packed_e5m2_haswell`        |        28.9 gso/s, 0 ulp |        30.7 gso/s, 0 ulp |        14.7 gso/s, 0 ulp |
| `nk_dots_symmetric_e5m2_haswell`     |        15.3 gso/s, 0 ulp |        16.4 gso/s, 0 ulp |        33.4 gso/s, 0 ulp |
| `nk_dots_packed_e5m2_skylake`        |        35.1 gso/s, 0 ulp |        36.1 gso/s, 0 ulp |        38.5 gso/s, 0 ulp |
| `nk_dots_symmetric_e5m2_skylake`     |        20.7 gso/s, 0 ulp |        20.8 gso/s, 0 ulp |        39.9 gso/s, 0 ulp |
| `nk_dots_packed_e5m2_genoa`          |        43.1 gso/s, 0 ulp |        47.7 gso/s, 0 ulp |        30.2 gso/s, ? ulp |
| `nk_dots_symmetric_e5m2_genoa`       |        31.9 gso/s, 0 ulp |        31.8 gso/s, 0 ulp |        47.5 gso/s, 0 ulp |
| `nk_dots_packed_e5m2_sapphireamx`    |         211 gso/s, 0 ulp |         252 gso/s, 0 ulp |         411 gso/s, 0 ulp |
| __e4m3__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_e4m3_serial`         |       0.469 gso/s, 0 ulp |       0.458 gso/s, 0 ulp |       0.475 gso/s, 0 ulp |
| `nk_dots_symmetric_e4m3_serial`      |       0.394 gso/s, 0 ulp |       0.401 gso/s, 0 ulp |       0.668 gso/s, 0 ulp |
| `nk_dots_packed_e4m3_haswell`        |        21.4 gso/s, 0 ulp |        22.3 gso/s, 0 ulp |        8.36 gso/s, 0 ulp |
| `nk_dots_symmetric_e4m3_haswell`     |        11.2 gso/s, 0 ulp |        11.2 gso/s, 0 ulp |        23.0 gso/s, 0 ulp |
| `nk_dots_packed_e4m3_skylake`        |        28.9 gso/s, 0 ulp |        30.9 gso/s, 0 ulp |        30.5 gso/s, 0 ulp |
| `nk_dots_symmetric_e4m3_skylake`     |        16.8 gso/s, 0 ulp |        17.3 gso/s, 0 ulp |        31.1 gso/s, 0 ulp |
| `nk_dots_packed_e4m3_genoa`          |        44.0 gso/s, 0 ulp |        47.9 gso/s, 0 ulp |        22.7 gso/s, ? ulp |
| `nk_dots_symmetric_e4m3_genoa`       |        31.5 gso/s, 0 ulp |        30.7 gso/s, 0 ulp |        46.4 gso/s, 0 ulp |
| `nk_dots_packed_e4m3_sapphireamx`    |         211 gso/s, 0 ulp |         255 gso/s, 0 ulp |         444 gso/s, 0 ulp |
| __e3m2__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_e3m2_serial`         |       0.546 gso/s, 0 ulp |       0.568 gso/s, 0 ulp |       0.260 gso/s, 0 ulp |
| `nk_dots_symmetric_e3m2_serial`      |       0.435 gso/s, 0 ulp |       0.433 gso/s, 0 ulp |       0.565 gso/s, 0 ulp |
| `nk_dots_packed_e3m2_haswell`        |        31.1 gso/s, 0 ulp |        33.1 gso/s, 0 ulp |        13.9 gso/s, 0 ulp |
| `nk_dots_symmetric_e3m2_haswell`     |        28.5 gso/s, 0 ulp |        30.2 gso/s, 0 ulp |        62.1 gso/s, 0 ulp |
| `nk_dots_packed_e3m2_skylake`        |        38.5 gso/s, 0 ulp |        40.9 gso/s, 0 ulp |        43.9 gso/s, 0 ulp |
| `nk_dots_symmetric_e3m2_skylake`     |        39.3 gso/s, 0 ulp |        41.8 gso/s, 0 ulp |        91.0 gso/s, 0 ulp |
| `nk_dots_packed_e3m2_genoa`          |        51.5 gso/s, 0 ulp |        57.2 gso/s, 0 ulp |        36.8 gso/s, ? ulp |
| `nk_dots_symmetric_e3m2_genoa`       |        39.8 gso/s, 0 ulp |        41.1 gso/s, 0 ulp |        52.9 gso/s, 0 ulp |
| `nk_dots_packed_e3m2_sapphireamx`    |         195 gso/s, 0 ulp |         302 gso/s, 0 ulp |         418 gso/s, 0 ulp |
| `nk_dots_symmetric_e3m2_sapphireamx` |        47.0 gso/s, 0 ulp |        65.8 gso/s, 0 ulp |        87.1 gso/s, 0 ulp |
| __e2m3__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_e2m3_serial`         |       0.334 gso/s, 0 ulp |       0.335 gso/s, 0 ulp |       0.206 gso/s, 0 ulp |
| `nk_dots_symmetric_e2m3_serial`      |       0.302 gso/s, 0 ulp |       0.297 gso/s, 0 ulp |       0.435 gso/s, 0 ulp |
| `nk_dots_packed_e2m3_haswell`        |        57.6 gso/s, 0 ulp |        61.5 gso/s, 0 ulp |        30.1 gso/s, 0 ulp |
| `nk_dots_symmetric_e2m3_haswell`     |        50.0 gso/s, 0 ulp |        59.0 gso/s, 0 ulp |         123 gso/s, 0 ulp |
| `nk_dots_packed_e2m3_skylake`        |        69.5 gso/s, 0 ulp |        81.2 gso/s, 0 ulp |        82.3 gso/s, 0 ulp |
| `nk_dots_symmetric_e2m3_skylake`     |        62.4 gso/s, 0 ulp |        78.1 gso/s, 0 ulp |         166 gso/s, 0 ulp |
| `nk_dots_packed_e2m3_genoa`          |        51.6 gso/s, 0 ulp |        56.6 gso/s, 0 ulp |        34.9 gso/s, ? ulp |
| `nk_dots_symmetric_e2m3_genoa`       |        39.9 gso/s, 0 ulp |        41.7 gso/s, 0 ulp |        55.4 gso/s, 0 ulp |
| `nk_dots_packed_e2m3_sapphireamx`    |         295 gso/s, 0 ulp |         647 gso/s, 0 ulp |        1078 gso/s, 0 ulp |
| `nk_dots_symmetric_e2m3_sapphireamx` |        68.4 gso/s, 0 ulp |         144 gso/s, 0 ulp |         182 gso/s, 0 ulp |
| __i8__                               | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_i8_serial`           |               9.61 gso/s |               9.90 gso/s |               7.86 gso/s |
| `nk_dots_symmetric_i8_serial`        |               4.69 gso/s |               4.73 gso/s |               4.66 gso/s |
| `nk_dots_packed_i8_haswell`          |               88.9 gso/s |                101 gso/s |               45.4 gso/s |
| `nk_dots_symmetric_i8_haswell`       |               57.6 gso/s |               82.6 gso/s |                171 gso/s |
| `nk_dots_packed_i8_icelake`          |                193 gso/s |                316 gso/s |                211 gso/s |
| `nk_dots_symmetric_i8_icelake`       |                101 gso/s |                177 gso/s |                338 gso/s |
| `nk_dots_packed_i8_sapphireamx`      |                443 gso/s |                888 gso/s |               1260 gso/s |
| `nk_dots_symmetric_i8_sapphireamx`   |               78.9 gso/s |                187 gso/s |                216 gso/s |
| __u8__                               | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_u8_serial`           |               13.1 gso/s |               13.6 gso/s |               9.34 gso/s |
| `nk_dots_symmetric_u8_serial`        |               4.52 gso/s |               4.77 gso/s |               4.68 gso/s |
| `nk_dots_packed_u8_haswell`          |               89.7 gso/s |                101 gso/s |               41.7 gso/s |
| `nk_dots_symmetric_u8_haswell`       |               57.9 gso/s |               81.1 gso/s |                168 gso/s |
| `nk_dots_packed_u8_icelake`          |                199 gso/s |                333 gso/s |                186 gso/s |
| `nk_dots_symmetric_u8_icelake`       |                102 gso/s |                184 gso/s |                335 gso/s |
| __i4__                               | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_i4_serial`           |               3.93 gso/s |               4.01 gso/s |               2.38 gso/s |
| `nk_dots_symmetric_i4_serial`        |               3.47 gso/s |               3.52 gso/s |               3.29 gso/s |
| `nk_dots_packed_i4_icelake`          |                138 gso/s |                213 gso/s |                140 gso/s |
| `nk_dots_symmetric_i4_icelake`       |                121 gso/s |                237 gso/s |                356 gso/s |
| __u4__                               | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_u4_serial`           |               5.76 gso/s |               5.52 gso/s |               3.27 gso/s |
| `nk_dots_symmetric_u4_serial`        |               5.25 gso/s |               5.71 gso/s |               4.78 gso/s |
| `nk_dots_packed_u4_icelake`          |                185 gso/s |                296 gso/s |                211 gso/s |
| `nk_dots_symmetric_u4_icelake`       |                142 gso/s |                291 gso/s |                428 gso/s |
| __u1__                               | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_u1_haswell`          |                220 gso/s |                249 gso/s |                334 gso/s |
| `nk_dots_symmetric_u1_haswell`       |                127 gso/s |                271 gso/s |                682 gso/s |
| `nk_dots_packed_u1_icelake`          |                250 gso/s |                742 gso/s |                881 gso/s |
| `nk_dots_symmetric_u1_icelake`       |                174 gso/s |                653 gso/s |               1610 gso/s |

#### WASM

Measured with Wasmtime v42 (Cranelift backend).

| Kernel                               |                     256³ |                    1024³ |                    4096³ |
| :----------------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f64__                              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_f64_serial`          |       0.596 gso/s, 3 ulp |       0.701 gso/s, 5 ulp |       0.797 gso/s, 6 ulp |
| `nk_dots_symmetric_f64_serial`       |       0.497 gso/s, 4 ulp |       0.568 gso/s, 3 ulp |         ? gso/s, 2.4 ulp |
| `nk_dots_packed_f64_v128relaxed`     |         ? gso/s, 9.9 ulp |          ? gso/s, 52 ulp |          ? gso/s, 33 ulp |
| `nk_dots_symmetric_f64_v128relaxed`  |          ? gso/s, 11 ulp |         ? gso/s, 124 ulp |          ? gso/s, 38 ulp |
| __f32__                              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_f32_serial`          |       17.0 gso/s, 19 ulp |       16.8 gso/s, 30 ulp |      17.5 gso/s, 725 ulp |
| `nk_dots_symmetric_f32_serial`       |       4.26 gso/s, 20 ulp |       4.27 gso/s, 29 ulp |          ? gso/s, 59 ulp |
| `nk_dots_packed_f32_v128relaxed`     |          ? gso/s, 11 ulp |          ? gso/s, 19 ulp |          ? gso/s, 31 ulp |
| `nk_dots_symmetric_f32_v128relaxed`  |          ? gso/s, 12 ulp |          ? gso/s, 50 ulp |          ? gso/s, 37 ulp |
| __bf16__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_bf16_serial`         |      1.18 gso/s, 0.1 ulp |      1.19 gso/s, 0.5 ulp |        1.22 gso/s, 5 ulp |
| `nk_dots_symmetric_bf16_serial`      |        1.17 gso/s, 0 ulp |      1.18 gso/s, 0.6 ulp |         ? gso/s, 1.1 ulp |
| `nk_dots_packed_bf16_v128relaxed`    |           ? gso/s, 0 ulp |         ? gso/s, 0.5 ulp |         ? gso/s, 6.8 ulp |
| `nk_dots_symmetric_bf16_v128relaxed` |         ? gso/s, 0.1 ulp |         ? gso/s, 1.4 ulp |          ? gso/s, 11 ulp |
| __e5m2__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_e5m2_serial`         |       0.537 gso/s, 0 ulp |       0.576 gso/s, 0 ulp |           ? gso/s, 0 ulp |
| `nk_dots_symmetric_e5m2_serial`      |       0.377 gso/s, 0 ulp |       0.437 gso/s, 0 ulp |           ? gso/s, 0 ulp |
| `nk_dots_packed_e5m2_v128relaxed`    |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_dots_symmetric_e5m2_v128relaxed` |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| __e4m3__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_e4m3_serial`         |       0.469 gso/s, 0 ulp |       0.441 gso/s, 0 ulp |       0.475 gso/s, 0 ulp |
| `nk_dots_symmetric_e4m3_serial`      |       0.394 gso/s, 0 ulp |       0.393 gso/s, 0 ulp |           ? gso/s, 0 ulp |
| `nk_dots_packed_e4m3_v128relaxed`    |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_dots_symmetric_e4m3_v128relaxed` |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| __e2m3__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_e2m3_serial`         |       0.334 gso/s, 0 ulp |       0.324 gso/s, 0 ulp |           ? gso/s, 0 ulp |
| `nk_dots_symmetric_e2m3_serial`      |       0.302 gso/s, 0 ulp |       0.297 gso/s, 0 ulp |           ? gso/s, 0 ulp |
| `nk_dots_packed_e2m3_v128relaxed`    |           ? gso/s, 0 ulp |           ? gso/s, 0 ulp |           ? gso/s, 0 ulp |
| `nk_dots_symmetric_e2m3_v128relaxed` |           ? gso/s, 0 ulp |           ? gso/s, 0 ulp |           ? gso/s, 0 ulp |
| __i8__                               | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_i8_serial`           |               9.61 gso/s |               9.66 gso/s |                  0 gso/s |
| `nk_dots_symmetric_i8_serial`        |               4.69 gso/s |               4.73 gso/s |                  0 gso/s |
| `nk_dots_packed_i8_v128relaxed`      |                  0 gso/s |                  0 gso/s |                  0 gso/s |
| `nk_dots_symmetric_i8_v128relaxed`   |                  0 gso/s |                  0 gso/s |                  0 gso/s |
| __u8__                               | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_u8_serial`           |               13.1 gso/s |               13.7 gso/s |                  0 gso/s |
| `nk_dots_symmetric_u8_serial`        |               4.52 gso/s |               4.77 gso/s |                  0 gso/s |
| `nk_dots_packed_u8_v128relaxed`      |                  0 gso/s |                  0 gso/s |                  0 gso/s |
| `nk_dots_symmetric_u8_v128relaxed`   |                  0 gso/s |                  0 gso/s |                  0 gso/s |
| __i4__                               | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_i4_serial`           |               3.93 gso/s |               3.90 gso/s |                  0 gso/s |
| `nk_dots_symmetric_i4_serial`        |               3.47 gso/s |               3.52 gso/s |                  0 gso/s |
| `nk_dots_packed_i4_v128relaxed`      |                  0 gso/s |                  0 gso/s |                  0 gso/s |
| `nk_dots_symmetric_i4_v128relaxed`   |                  0 gso/s |                  0 gso/s |                  0 gso/s |
| __u4__                               | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_u4_serial`           |               5.76 gso/s |               5.78 gso/s |                  0 gso/s |
| `nk_dots_symmetric_u4_serial`        |               5.25 gso/s |               5.71 gso/s |                  0 gso/s |
| `nk_dots_packed_u4_v128relaxed`      |                  0 gso/s |                  0 gso/s |                  0 gso/s |
| `nk_dots_symmetric_u4_v128relaxed`   |                  0 gso/s |                  0 gso/s |                  0 gso/s |
| __u1__                               | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_u1_serial`           |                  0 gso/s |                  0 gso/s |                  0 gso/s |
| `nk_dots_packed_u1_v128relaxed`      |                  0 gso/s |                  0 gso/s |                  0 gso/s |
| `nk_dots_symmetric_u1_serial`        |                  0 gso/s |                  0 gso/s |                  0 gso/s |
| `nk_dots_symmetric_u1_v128relaxed`   |                  0 gso/s |                  0 gso/s |                  0 gso/s |

### Apple M4 Pro

#### Native

| Kernel                             |                     256³ |                    1024³ |                    4096³ |
| :--------------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f64__                            | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_f64_serial`        |       0.596 gso/s, 3 ulp |       0.701 gso/s, 5 ulp |       0.797 gso/s, 6 ulp |
| `nk_dots_symmetric_f64_serial`     |       0.497 gso/s, 4 ulp |       0.568 gso/s, 3 ulp |           ? gso/s, ? ulp |
| `nk_dots_packed_f64_neon`          |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_dots_symmetric_f64_neon`       |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_dots_packed_f64_smef64`        |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_dots_symmetric_f64_smef64`     |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| __f32__                            | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_f32_serial`        |       17.0 gso/s, 19 ulp |       16.8 gso/s, 30 ulp |      17.5 gso/s, 725 ulp |
| `nk_dots_symmetric_f32_serial`     |       4.26 gso/s, 20 ulp |       4.27 gso/s, 29 ulp |           ? gso/s, ? ulp |
| `nk_dots_packed_f32_neon`          |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_dots_symmetric_f32_neon`       |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_dots_packed_f32_smef64`        |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_dots_symmetric_f32_smef64`     |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| __bf16__                           | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_bf16_serial`       |      1.18 gso/s, 0.1 ulp |      1.19 gso/s, 0.5 ulp |        1.22 gso/s, 5 ulp |
| `nk_dots_symmetric_bf16_serial`    |        1.17 gso/s, 0 ulp |      1.18 gso/s, 0.6 ulp |           ? gso/s, ? ulp |
| `nk_dots_packed_bf16_neonbfdot`    |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_dots_symmetric_bf16_neonbfdot` |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_dots_packed_bf16_sme`          |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_dots_symmetric_bf16_sme`       |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| __f16__                            | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_f16_serial`        |      8.26 gso/s, 204 ulp |       8.04 gso/s, 36 ulp |      8.60 gso/s, 326 ulp |
| `nk_dots_symmetric_f16_serial`     |       4.21 gso/s, 13 ulp |       4.17 gso/s, 29 ulp |           ? gso/s, ? ulp |
| `nk_dots_packed_f16_neonhalf`      |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_dots_symmetric_f16_neonhalf`   |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_dots_packed_f16_neonfhm`       |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_dots_symmetric_f16_neonfhm`    |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_dots_packed_f16_sme`           |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_dots_symmetric_f16_sme`        |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| __e5m2__                           | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_e5m2_serial`       |       0.537 gso/s, 0 ulp |       0.576 gso/s, 0 ulp |           ? gso/s, ? ulp |
| `nk_dots_symmetric_e5m2_serial`    |       0.377 gso/s, 0 ulp |       0.437 gso/s, 0 ulp |           ? gso/s, ? ulp |
| `nk_dots_packed_e5m2_neonfhm`      |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_dots_symmetric_e5m2_neonfhm`   |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_dots_packed_e5m2_sme`          |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_dots_symmetric_e5m2_sme`       |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| __e4m3__                           | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_e4m3_serial`       |       0.469 gso/s, 0 ulp |       0.441 gso/s, 0 ulp |       0.475 gso/s, 0 ulp |
| `nk_dots_symmetric_e4m3_serial`    |       0.394 gso/s, 0 ulp |       0.393 gso/s, 0 ulp |           ? gso/s, ? ulp |
| `nk_dots_packed_e4m3_neonfhm`      |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_dots_symmetric_e4m3_neonfhm`   |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_dots_packed_e4m3_sme`          |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_dots_symmetric_e4m3_sme`       |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| __e3m2__                           | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_e3m2_serial`       |       0.546 gso/s, 0 ulp |       0.564 gso/s, 0 ulp |           ? gso/s, ? ulp |
| `nk_dots_symmetric_e3m2_serial`    |       0.435 gso/s, 0 ulp |       0.433 gso/s, 0 ulp |           ? gso/s, ? ulp |
| `nk_dots_packed_e3m2_neonfhm`      |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_dots_symmetric_e3m2_neonfhm`   |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_dots_packed_e3m2_sme`          |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_dots_symmetric_e3m2_sme`       |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| __e2m3__                           | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_e2m3_serial`       |       0.334 gso/s, 0 ulp |       0.324 gso/s, 0 ulp |           ? gso/s, ? ulp |
| `nk_dots_symmetric_e2m3_serial`    |       0.302 gso/s, 0 ulp |       0.297 gso/s, 0 ulp |           ? gso/s, ? ulp |
| `nk_dots_packed_e2m3_neonfhm`      |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_dots_symmetric_e2m3_neonfhm`   |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_dots_packed_e2m3_sme`          |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_dots_symmetric_e2m3_sme`       |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| __i8__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_i8_serial`         |               9.61 gso/s |               9.66 gso/s |                  ? gso/s |
| `nk_dots_symmetric_i8_serial`      |               4.69 gso/s |               4.73 gso/s |                  ? gso/s |
| `nk_dots_packed_i8_neonsdot`       |                  ? gso/s |                  ? gso/s |                  ? gso/s |
| `nk_dots_symmetric_i8_neonsdot`    |                  ? gso/s |                  ? gso/s |                  ? gso/s |
| `nk_dots_packed_i8_sme`            |                  ? gso/s |                  ? gso/s |                  ? gso/s |
| `nk_dots_symmetric_i8_sme`         |                  ? gso/s |                  ? gso/s |                  ? gso/s |
| __u8__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_u8_serial`         |               13.1 gso/s |               13.7 gso/s |                  ? gso/s |
| `nk_dots_symmetric_u8_serial`      |               4.52 gso/s |               4.77 gso/s |                  ? gso/s |
| `nk_dots_packed_u8_neonsdot`       |                  ? gso/s |                  ? gso/s |                  ? gso/s |
| `nk_dots_symmetric_u8_neonsdot`    |                  ? gso/s |                  ? gso/s |                  ? gso/s |
| `nk_dots_packed_u8_sme`            |                  ? gso/s |                  ? gso/s |                  ? gso/s |
| `nk_dots_symmetric_u8_sme`         |                  ? gso/s |                  ? gso/s |                  ? gso/s |
| __i4__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_i4_serial`         |               3.93 gso/s |               3.90 gso/s |                  ? gso/s |
| `nk_dots_symmetric_i4_serial`      |               3.47 gso/s |               3.52 gso/s |                  ? gso/s |
| `nk_dots_packed_i4_neonsdot`       |                  ? gso/s |                  ? gso/s |                  ? gso/s |
| `nk_dots_symmetric_i4_neonsdot`    |                  ? gso/s |                  ? gso/s |                  ? gso/s |
| `nk_dots_packed_i4_sme`            |                  ? gso/s |                  ? gso/s |                  ? gso/s |
| `nk_dots_symmetric_i4_sme`         |                  ? gso/s |                  ? gso/s |                  ? gso/s |
| __u4__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_u4_serial`         |               5.76 gso/s |               5.78 gso/s |                  ? gso/s |
| `nk_dots_symmetric_u4_serial`      |               5.25 gso/s |               5.71 gso/s |                  ? gso/s |
| `nk_dots_packed_u4_neonsdot`       |                  ? gso/s |                  ? gso/s |                  ? gso/s |
| `nk_dots_symmetric_u4_neonsdot`    |                  ? gso/s |                  ? gso/s |                  ? gso/s |
| `nk_dots_packed_u4_sme`            |                  ? gso/s |                  ? gso/s |                  ? gso/s |
| `nk_dots_symmetric_u4_sme`         |                  ? gso/s |                  ? gso/s |                  ? gso/s |
| __u1__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_u1_serial`         |                  ? gso/s |                  ? gso/s |                  ? gso/s |
| `nk_dots_symmetric_u1_serial`      |                  ? gso/s |                  ? gso/s |                  ? gso/s |
| `nk_dots_packed_u1_neon`           |                  ? gso/s |                  ? gso/s |                  ? gso/s |
| `nk_dots_symmetric_u1_neon`        |                  ? gso/s |                  ? gso/s |                  ? gso/s |
| `nk_dots_packed_u1_smebi32`        |                  ? gso/s |                  ? gso/s |                  ? gso/s |
| `nk_dots_symmetric_u1_smebi32`     |                  ? gso/s |                  ? gso/s |                  ? gso/s |

#### WASM

Measured with Wasmtime v42 (Cranelift backend).

| Kernel                               |                     256³ |                    1024³ |                    4096³ |
| :----------------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f64__                              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_f64_serial`          |       0.596 gso/s, 3 ulp |       0.701 gso/s, 5 ulp |       0.797 gso/s, 6 ulp |
| `nk_dots_packed_f64_v128relaxed`     |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_dots_symmetric_f64_serial`       |       0.497 gso/s, 4 ulp |       0.568 gso/s, 3 ulp |           ? gso/s, ? ulp |
| `nk_dots_symmetric_f64_v128relaxed`  |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| __f32__                              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_f32_serial`          |       17.0 gso/s, 19 ulp |       16.8 gso/s, 30 ulp |      17.5 gso/s, 725 ulp |
| `nk_dots_packed_f32_v128relaxed`     |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_dots_symmetric_f32_serial`       |       4.26 gso/s, 20 ulp |       4.27 gso/s, 29 ulp |           ? gso/s, ? ulp |
| `nk_dots_symmetric_f32_v128relaxed`  |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| __bf16__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_bf16_serial`         |      1.18 gso/s, 0.1 ulp |      1.19 gso/s, 0.5 ulp |        1.22 gso/s, 5 ulp |
| `nk_dots_packed_bf16_v128relaxed`    |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_dots_symmetric_bf16_serial`      |        1.17 gso/s, 0 ulp |      1.18 gso/s, 0.6 ulp |           ? gso/s, ? ulp |
| `nk_dots_symmetric_bf16_v128relaxed` |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| __e5m2__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_e5m2_serial`         |       0.537 gso/s, 0 ulp |       0.576 gso/s, 0 ulp |           ? gso/s, ? ulp |
| `nk_dots_packed_e5m2_v128relaxed`    |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_dots_symmetric_e5m2_serial`      |       0.377 gso/s, 0 ulp |       0.437 gso/s, 0 ulp |           ? gso/s, ? ulp |
| `nk_dots_symmetric_e5m2_v128relaxed` |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| __e4m3__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_e4m3_serial`         |       0.469 gso/s, 0 ulp |       0.441 gso/s, 0 ulp |       0.475 gso/s, 0 ulp |
| `nk_dots_packed_e4m3_v128relaxed`    |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_dots_symmetric_e4m3_serial`      |       0.394 gso/s, 0 ulp |       0.393 gso/s, 0 ulp |           ? gso/s, ? ulp |
| `nk_dots_symmetric_e4m3_v128relaxed` |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| __e2m3__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_e2m3_serial`         |       0.334 gso/s, 0 ulp |       0.324 gso/s, 0 ulp |           ? gso/s, ? ulp |
| `nk_dots_packed_e2m3_v128relaxed`    |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_dots_symmetric_e2m3_serial`      |       0.302 gso/s, 0 ulp |       0.297 gso/s, 0 ulp |           ? gso/s, ? ulp |
| `nk_dots_symmetric_e2m3_v128relaxed` |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| __i8__                               | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_i8_serial`           |               9.61 gso/s |               9.66 gso/s |                  ? gso/s |
| `nk_dots_packed_i8_v128relaxed`      |                  ? gso/s |                  ? gso/s |                  ? gso/s |
| `nk_dots_symmetric_i8_serial`        |               4.69 gso/s |               4.73 gso/s |                  ? gso/s |
| `nk_dots_symmetric_i8_v128relaxed`   |                  ? gso/s |                  ? gso/s |                  ? gso/s |
| __u8__                               | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_u8_serial`           |               13.1 gso/s |               13.7 gso/s |                  ? gso/s |
| `nk_dots_packed_u8_v128relaxed`      |                  ? gso/s |                  ? gso/s |                  ? gso/s |
| `nk_dots_symmetric_u8_serial`        |               4.52 gso/s |               4.77 gso/s |                  ? gso/s |
| `nk_dots_symmetric_u8_v128relaxed`   |                  ? gso/s |                  ? gso/s |                  ? gso/s |
| __i4__                               | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_i4_serial`           |               3.93 gso/s |               3.90 gso/s |                  ? gso/s |
| `nk_dots_packed_i4_v128relaxed`      |                  ? gso/s |                  ? gso/s |                  ? gso/s |
| `nk_dots_symmetric_i4_serial`        |               3.47 gso/s |               3.52 gso/s |                  ? gso/s |
| `nk_dots_symmetric_i4_v128relaxed`   |                  ? gso/s |                  ? gso/s |                  ? gso/s |
| __u4__                               | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_u4_serial`           |               5.76 gso/s |               5.78 gso/s |                  ? gso/s |
| `nk_dots_packed_u4_v128relaxed`      |                  ? gso/s |                  ? gso/s |                  ? gso/s |
| `nk_dots_symmetric_u4_serial`        |               5.25 gso/s |               5.71 gso/s |                  ? gso/s |
| `nk_dots_symmetric_u4_v128relaxed`   |                  ? gso/s |                  ? gso/s |                  ? gso/s |
| __u1__                               | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_u1_serial`           |                  ? gso/s |                  ? gso/s |                  ? gso/s |
| `nk_dots_packed_u1_v128relaxed`      |                  ? gso/s |                  ? gso/s |                  ? gso/s |
| `nk_dots_symmetric_u1_serial`        |                  ? gso/s |                  ? gso/s |                  ? gso/s |
| `nk_dots_symmetric_u1_v128relaxed`   |                  ? gso/s |                  ? gso/s |                  ? gso/s |

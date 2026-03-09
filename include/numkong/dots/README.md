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

Controlled by `NK_MATRIX_HEIGHT`, `NK_MATRIX_WIDTH`, `NK_MATRIX_DEPTH`.
All values are set to the same value for products of two square-shaped matrices.
Columns show for matrixes with 256, 1024, and 4096 sides.

### Intel Sapphire Rapids

#### Native

| Kernel                               |            256³ |           1024³ |           4096³ |
| :----------------------------------- | --------------: | --------------: | --------------: |
| __f64__                              |                 |                 |                 |
| `nk_dots_packed_f64_serial`          | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_symmetric_f64_serial`       | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_packed_f64_haswell`         | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_symmetric_f64_haswell`      | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_packed_f64_skylake`         | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_symmetric_f64_skylake`      | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| __f32__                              |                 |                 |                 |
| `nk_dots_packed_f32_serial`          | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_symmetric_f32_serial`       | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_packed_f32_haswell`         | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_symmetric_f32_haswell`      | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_packed_f32_skylake`         | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_symmetric_f32_skylake`      | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| __bf16__                             |                 |                 |                 |
| `nk_dots_packed_bf16_serial`         | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_symmetric_bf16_serial`      | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_packed_bf16_haswell`        | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_symmetric_bf16_haswell`     | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_packed_bf16_skylake`        | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_symmetric_bf16_skylake`     | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_packed_bf16_genoa`          | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_symmetric_bf16_genoa`       | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_packed_bf16_sapphireamx`    | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_symmetric_bf16_sapphireamx` | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| __f16__                              |                 |                 |                 |
| `nk_dots_packed_f16_serial`          | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_symmetric_f16_serial`       | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_packed_f16_haswell`         | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_symmetric_f16_haswell`      | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_packed_f16_skylake`         | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_symmetric_f16_skylake`      | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| __e5m2__                             |                 |                 |                 |
| `nk_dots_packed_e5m2_serial`         | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_symmetric_e5m2_serial`      | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_packed_e5m2_haswell`        | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_symmetric_e5m2_haswell`     | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_packed_e5m2_skylake`        | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_symmetric_e5m2_skylake`     | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_packed_e5m2_genoa`          | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_symmetric_e5m2_genoa`       | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_packed_e5m2_sapphireamx`    | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_symmetric_e5m2_sapphireamx` | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| __e4m3__                             |                 |                 |                 |
| `nk_dots_packed_e4m3_serial`         | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_symmetric_e4m3_serial`      | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_packed_e4m3_haswell`        | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_symmetric_e4m3_haswell`     | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_packed_e4m3_skylake`        | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_symmetric_e4m3_skylake`     | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_packed_e4m3_genoa`          | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_symmetric_e4m3_genoa`       | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_packed_e4m3_sapphireamx`    | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_symmetric_e4m3_sapphireamx` | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| __e3m2__                             |                 |                 |                 |
| `nk_dots_packed_e3m2_serial`         | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_symmetric_e3m2_serial`      | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_packed_e3m2_haswell`        | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_symmetric_e3m2_haswell`     | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_packed_e3m2_skylake`        | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_symmetric_e3m2_skylake`     | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_packed_e3m2_genoa`          | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_symmetric_e3m2_genoa`       | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_packed_e3m2_sapphireamx`    | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_symmetric_e3m2_sapphireamx` | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| __e2m3__                             |                 |                 |                 |
| `nk_dots_packed_e2m3_serial`         | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_symmetric_e2m3_serial`      | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_packed_e2m3_haswell`        | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_symmetric_e2m3_haswell`     | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_packed_e2m3_skylake`        | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_symmetric_e2m3_skylake`     | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_packed_e2m3_genoa`          | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_symmetric_e2m3_genoa`       | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_packed_e2m3_sapphireamx`    | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_symmetric_e2m3_sapphireamx` | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_packed_e2m3_alder`          | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_symmetric_e2m3_alder`       | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_packed_e2m3_sierra`         | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_symmetric_e2m3_sierra`      | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| __i8__                               |                 |                 |                 |
| `nk_dots_packed_i8_serial`           |        0 GTOP/s |        0 GTOP/s |        0 GTOP/s |
| `nk_dots_symmetric_i8_serial`        |        0 GTOP/s |        0 GTOP/s |        0 GTOP/s |
| `nk_dots_packed_i8_haswell`          |        0 GTOP/s |        0 GTOP/s |        0 GTOP/s |
| `nk_dots_symmetric_i8_haswell`       |        0 GTOP/s |        0 GTOP/s |        0 GTOP/s |
| `nk_dots_packed_i8_icelake`          |        0 GTOP/s |        0 GTOP/s |        0 GTOP/s |
| `nk_dots_symmetric_i8_icelake`       |        0 GTOP/s |        0 GTOP/s |        0 GTOP/s |
| `nk_dots_packed_i8_sapphireamx`      |        0 GTOP/s |        0 GTOP/s |        0 GTOP/s |
| `nk_dots_symmetric_i8_sapphireamx`   |        0 GTOP/s |        0 GTOP/s |        0 GTOP/s |
| __u8__                               |                 |                 |                 |
| `nk_dots_packed_u8_serial`           |        0 GTOP/s |        0 GTOP/s |        0 GTOP/s |
| `nk_dots_symmetric_u8_serial`        |        0 GTOP/s |        0 GTOP/s |        0 GTOP/s |
| `nk_dots_packed_u8_haswell`          |        0 GTOP/s |        0 GTOP/s |        0 GTOP/s |
| `nk_dots_symmetric_u8_haswell`       |        0 GTOP/s |        0 GTOP/s |        0 GTOP/s |
| `nk_dots_packed_u8_icelake`          |        0 GTOP/s |        0 GTOP/s |        0 GTOP/s |
| `nk_dots_symmetric_u8_icelake`       |        0 GTOP/s |        0 GTOP/s |        0 GTOP/s |
| `nk_dots_packed_u8_sapphireamx`      |        0 GTOP/s |        0 GTOP/s |        0 GTOP/s |
| `nk_dots_symmetric_u8_sapphireamx`   |        0 GTOP/s |        0 GTOP/s |        0 GTOP/s |
| `nk_dots_symmetric_u8_alder`         |        0 GTOP/s |        0 GTOP/s |        0 GTOP/s |
| `nk_dots_symmetric_u8_sierra`        |        0 GTOP/s |        0 GTOP/s |        0 GTOP/s |
| __i4__                               |                 |                 |                 |
| `nk_dots_packed_i4_serial`           |        0 GTOP/s |        0 GTOP/s |        0 GTOP/s |
| `nk_dots_symmetric_i4_serial`        |        0 GTOP/s |        0 GTOP/s |        0 GTOP/s |
| `nk_dots_packed_i4_haswell`          |        0 GTOP/s |        0 GTOP/s |        0 GTOP/s |
| `nk_dots_symmetric_i4_haswell`       |        0 GTOP/s |        0 GTOP/s |        0 GTOP/s |
| `nk_dots_packed_i4_icelake`          |        0 GTOP/s |        0 GTOP/s |        0 GTOP/s |
| `nk_dots_symmetric_i4_icelake`       |        0 GTOP/s |        0 GTOP/s |        0 GTOP/s |
| __u4__                               |                 |                 |                 |
| `nk_dots_packed_u4_serial`           |        0 GTOP/s |        0 GTOP/s |        0 GTOP/s |
| `nk_dots_symmetric_u4_serial`        |        0 GTOP/s |        0 GTOP/s |        0 GTOP/s |
| `nk_dots_packed_u4_haswell`          |        0 GTOP/s |        0 GTOP/s |        0 GTOP/s |
| `nk_dots_symmetric_u4_haswell`       |        0 GTOP/s |        0 GTOP/s |        0 GTOP/s |
| `nk_dots_packed_u4_icelake`          |        0 GTOP/s |        0 GTOP/s |        0 GTOP/s |
| `nk_dots_symmetric_u4_icelake`       |        0 GTOP/s |        0 GTOP/s |        0 GTOP/s |
| __u1__                               |                 |                 |                 |
| `nk_dots_packed_u1_serial`           |        0 GTOP/s |        0 GTOP/s |        0 GTOP/s |
| `nk_dots_symmetric_u1_serial`        |        0 GTOP/s |        0 GTOP/s |        0 GTOP/s |
| `nk_dots_packed_u1_haswell`          |        0 GTOP/s |        0 GTOP/s |        0 GTOP/s |
| `nk_dots_symmetric_u1_haswell`       |        0 GTOP/s |        0 GTOP/s |        0 GTOP/s |
| `nk_dots_packed_u1_icelake`          |        0 GTOP/s |        0 GTOP/s |        0 GTOP/s |
| `nk_dots_symmetric_u1_icelake`       |        0 GTOP/s |        0 GTOP/s |        0 GTOP/s |

#### V8 (Chromium)

| Kernel                               |            256³ |           1024³ |           4096³ |
| :----------------------------------- | --------------: | --------------: | --------------: |
| __f64__                              |                 |                 |                 |
| `nk_dots_packed_f64_v128relaxed`     | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_symmetric_f64_v128relaxed`  | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| __f32__                              |                 |                 |                 |
| `nk_dots_packed_f32_v128relaxed`     | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_symmetric_f32_v128relaxed`  | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| __bf16__                             |                 |                 |                 |
| `nk_dots_packed_bf16_v128relaxed`    | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_symmetric_bf16_v128relaxed` | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| __e5m2__                             |                 |                 |                 |
| `nk_dots_packed_e5m2_v128relaxed`    | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_symmetric_e5m2_v128relaxed` | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| __e4m3__                             |                 |                 |                 |
| `nk_dots_packed_e4m3_v128relaxed`    | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_symmetric_e4m3_v128relaxed` | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| __e2m3__                             |                 |                 |                 |
| `nk_dots_packed_e2m3_v128relaxed`    | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_symmetric_e2m3_v128relaxed` | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| __i8__                               |                 |                 |                 |
| `nk_dots_packed_i8_v128relaxed`      |        0 GTOP/s |        0 GTOP/s |        0 GTOP/s |
| `nk_dots_symmetric_i8_v128relaxed`   |        0 GTOP/s |        0 GTOP/s |        0 GTOP/s |
| __u8__                               |                 |                 |                 |
| `nk_dots_packed_u8_v128relaxed`      |        0 GTOP/s |        0 GTOP/s |        0 GTOP/s |
| `nk_dots_symmetric_u8_v128relaxed`   |        0 GTOP/s |        0 GTOP/s |        0 GTOP/s |
| __i4__                               |                 |                 |                 |
| `nk_dots_packed_i4_v128relaxed`      |        0 GTOP/s |        0 GTOP/s |        0 GTOP/s |
| `nk_dots_symmetric_i4_v128relaxed`   |        0 GTOP/s |        0 GTOP/s |        0 GTOP/s |
| __u4__                               |                 |                 |                 |
| `nk_dots_packed_u4_v128relaxed`      |        0 GTOP/s |        0 GTOP/s |        0 GTOP/s |
| `nk_dots_symmetric_u4_v128relaxed`   |        0 GTOP/s |        0 GTOP/s |        0 GTOP/s |

#### Wasmtime (Cranelift)

| Kernel                               |            256³ |           1024³ |           4096³ |
| :----------------------------------- | --------------: | --------------: | --------------: |
| __f64__                              |                 |                 |                 |
| `nk_dots_packed_f64_v128relaxed`     | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_symmetric_f64_v128relaxed`  | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| __f32__                              |                 |                 |                 |
| `nk_dots_packed_f32_v128relaxed`     | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_symmetric_f32_v128relaxed`  | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| __bf16__                             |                 |                 |                 |
| `nk_dots_packed_bf16_v128relaxed`    | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_symmetric_bf16_v128relaxed` | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| __e5m2__                             |                 |                 |                 |
| `nk_dots_packed_e5m2_v128relaxed`    | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_symmetric_e5m2_v128relaxed` | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| __e4m3__                             |                 |                 |                 |
| `nk_dots_packed_e4m3_v128relaxed`    | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_symmetric_e4m3_v128relaxed` | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| __e2m3__                             |                 |                 |                 |
| `nk_dots_packed_e2m3_v128relaxed`    | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_symmetric_e2m3_v128relaxed` | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| __i8__                               |                 |                 |                 |
| `nk_dots_packed_i8_v128relaxed`      |        0 GTOP/s |        0 GTOP/s |        0 GTOP/s |
| `nk_dots_symmetric_i8_v128relaxed`   |        0 GTOP/s |        0 GTOP/s |        0 GTOP/s |
| __u8__                               |                 |                 |                 |
| `nk_dots_packed_u8_v128relaxed`      |        0 GTOP/s |        0 GTOP/s |        0 GTOP/s |
| `nk_dots_symmetric_u8_v128relaxed`   |        0 GTOP/s |        0 GTOP/s |        0 GTOP/s |
| __i4__                               |                 |                 |                 |
| `nk_dots_packed_i4_v128relaxed`      |        0 GTOP/s |        0 GTOP/s |        0 GTOP/s |
| `nk_dots_symmetric_i4_v128relaxed`   |        0 GTOP/s |        0 GTOP/s |        0 GTOP/s |
| __u4__                               |                 |                 |                 |
| `nk_dots_packed_u4_v128relaxed`      |        0 GTOP/s |        0 GTOP/s |        0 GTOP/s |
| `nk_dots_symmetric_u4_v128relaxed`   |        0 GTOP/s |        0 GTOP/s |        0 GTOP/s |

### Apple M4 Pro

#### Native

| Kernel                             |            256³ |           1024³ |           4096³ |
| :--------------------------------- | --------------: | --------------: | --------------: |
| __f64__                            |                 |                 |                 |
| `nk_dots_packed_f64_serial`        | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_symmetric_f64_serial`     | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_packed_f64_neon`          | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_symmetric_f64_neon`       | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_packed_f64_smef64`        | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_symmetric_f64_smef64`     | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| __f32__                            |                 |                 |                 |
| `nk_dots_packed_f32_serial`        | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_symmetric_f32_serial`     | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_packed_f32_neon`          | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_symmetric_f32_neon`       | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_packed_f32_smef64`        | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_symmetric_f32_smef64`     | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| __bf16__                           |                 |                 |                 |
| `nk_dots_packed_bf16_serial`       | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_symmetric_bf16_serial`    | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_packed_bf16_neonbfdot`    | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_symmetric_bf16_neonbfdot` | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_packed_bf16_sme`          | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_symmetric_bf16_sme`       | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| __f16__                            |                 |                 |                 |
| `nk_dots_packed_f16_serial`        | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_symmetric_f16_serial`     | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_packed_f16_neonhalf`      | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_symmetric_f16_neonhalf`   | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_packed_f16_neonfhm`       | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_symmetric_f16_neonfhm`    | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_packed_f16_sme`           | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_symmetric_f16_sme`        | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| __e5m2__                           |                 |                 |                 |
| `nk_dots_packed_e5m2_serial`       | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_symmetric_e5m2_serial`    | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_packed_e5m2_neonfhm`      | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_symmetric_e5m2_neonfhm`   | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_packed_e5m2_sme`          | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_symmetric_e5m2_sme`       | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| __e4m3__                           |                 |                 |                 |
| `nk_dots_packed_e4m3_serial`       | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_symmetric_e4m3_serial`    | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_packed_e4m3_neonfhm`      | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_symmetric_e4m3_neonfhm`   | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_packed_e4m3_sme`          | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_symmetric_e4m3_sme`       | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| __e3m2__                           |                 |                 |                 |
| `nk_dots_packed_e3m2_serial`       | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_symmetric_e3m2_serial`    | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_packed_e3m2_neonfhm`      | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_symmetric_e3m2_neonfhm`   | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_packed_e3m2_sme`          | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_symmetric_e3m2_sme`       | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| __e2m3__                           |                 |                 |                 |
| `nk_dots_packed_e2m3_serial`       | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_symmetric_e2m3_serial`    | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_packed_e2m3_neonfhm`      | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_symmetric_e2m3_neonfhm`   | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_packed_e2m3_sme`          | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_symmetric_e2m3_sme`       | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| __i8__                             |                 |                 |                 |
| `nk_dots_packed_i8_serial`         |        0 GTOP/s |        0 GTOP/s |        0 GTOP/s |
| `nk_dots_symmetric_i8_serial`      |        0 GTOP/s |        0 GTOP/s |        0 GTOP/s |
| `nk_dots_packed_i8_neonsdot`       |        0 GTOP/s |        0 GTOP/s |        0 GTOP/s |
| `nk_dots_symmetric_i8_neonsdot`    |        0 GTOP/s |        0 GTOP/s |        0 GTOP/s |
| `nk_dots_packed_i8_sme`            |        0 GTOP/s |        0 GTOP/s |        0 GTOP/s |
| `nk_dots_symmetric_i8_sme`         |        0 GTOP/s |        0 GTOP/s |        0 GTOP/s |
| __u8__                             |                 |                 |                 |
| `nk_dots_packed_u8_serial`         |        0 GTOP/s |        0 GTOP/s |        0 GTOP/s |
| `nk_dots_symmetric_u8_serial`      |        0 GTOP/s |        0 GTOP/s |        0 GTOP/s |
| `nk_dots_packed_u8_neonsdot`       |        0 GTOP/s |        0 GTOP/s |        0 GTOP/s |
| `nk_dots_symmetric_u8_neonsdot`    |        0 GTOP/s |        0 GTOP/s |        0 GTOP/s |
| `nk_dots_packed_u8_sme`            |        0 GTOP/s |        0 GTOP/s |        0 GTOP/s |
| `nk_dots_symmetric_u8_sme`         |        0 GTOP/s |        0 GTOP/s |        0 GTOP/s |
| __i4__                             |                 |                 |                 |
| `nk_dots_packed_i4_serial`         |        0 GTOP/s |        0 GTOP/s |        0 GTOP/s |
| `nk_dots_symmetric_i4_serial`      |        0 GTOP/s |        0 GTOP/s |        0 GTOP/s |
| `nk_dots_packed_i4_neonsdot`       |        0 GTOP/s |        0 GTOP/s |        0 GTOP/s |
| `nk_dots_symmetric_i4_neonsdot`    |        0 GTOP/s |        0 GTOP/s |        0 GTOP/s |
| `nk_dots_packed_i4_sme`            |        0 GTOP/s |        0 GTOP/s |        0 GTOP/s |
| `nk_dots_symmetric_i4_sme`         |        0 GTOP/s |        0 GTOP/s |        0 GTOP/s |
| __u4__                             |                 |                 |                 |
| `nk_dots_packed_u4_serial`         |        0 GTOP/s |        0 GTOP/s |        0 GTOP/s |
| `nk_dots_symmetric_u4_serial`      |        0 GTOP/s |        0 GTOP/s |        0 GTOP/s |
| `nk_dots_packed_u4_neonsdot`       |        0 GTOP/s |        0 GTOP/s |        0 GTOP/s |
| `nk_dots_symmetric_u4_neonsdot`    |        0 GTOP/s |        0 GTOP/s |        0 GTOP/s |
| `nk_dots_packed_u4_sme`            |        0 GTOP/s |        0 GTOP/s |        0 GTOP/s |
| `nk_dots_symmetric_u4_sme`         |        0 GTOP/s |        0 GTOP/s |        0 GTOP/s |
| __u1__                             |                 |                 |                 |
| `nk_dots_packed_u1_serial`         |        0 GTOP/s |        0 GTOP/s |        0 GTOP/s |
| `nk_dots_symmetric_u1_serial`      |        0 GTOP/s |        0 GTOP/s |        0 GTOP/s |
| `nk_dots_packed_u1_neon`           |        0 GTOP/s |        0 GTOP/s |        0 GTOP/s |
| `nk_dots_symmetric_u1_neon`        |        0 GTOP/s |        0 GTOP/s |        0 GTOP/s |
| `nk_dots_packed_u1_smebi32`        |        0 GTOP/s |        0 GTOP/s |        0 GTOP/s |
| `nk_dots_symmetric_u1_smebi32`     |        0 GTOP/s |        0 GTOP/s |        0 GTOP/s |

#### V8 (Chromium)

| Kernel                               |            256³ |           1024³ |           4096³ |
| :----------------------------------- | --------------: | --------------: | --------------: |
| __f64__                              |                 |                 |                 |
| `nk_dots_packed_f64_v128relaxed`     | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_symmetric_f64_v128relaxed`  | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| __f32__                              |                 |                 |                 |
| `nk_dots_packed_f32_v128relaxed`     | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_symmetric_f32_v128relaxed`  | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| __bf16__                             |                 |                 |                 |
| `nk_dots_packed_bf16_v128relaxed`    | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_symmetric_bf16_v128relaxed` | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| __e5m2__                             |                 |                 |                 |
| `nk_dots_packed_e5m2_v128relaxed`    | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_symmetric_e5m2_v128relaxed` | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| __e4m3__                             |                 |                 |                 |
| `nk_dots_packed_e4m3_v128relaxed`    | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_symmetric_e4m3_v128relaxed` | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| __e2m3__                             |                 |                 |                 |
| `nk_dots_packed_e2m3_v128relaxed`    | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_symmetric_e2m3_v128relaxed` | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| __i8__                               |                 |                 |                 |
| `nk_dots_packed_i8_v128relaxed`      |        0 GTOP/s |        0 GTOP/s |        0 GTOP/s |
| `nk_dots_symmetric_i8_v128relaxed`   |        0 GTOP/s |        0 GTOP/s |        0 GTOP/s |
| __u8__                               |                 |                 |                 |
| `nk_dots_packed_u8_v128relaxed`      |        0 GTOP/s |        0 GTOP/s |        0 GTOP/s |
| `nk_dots_symmetric_u8_v128relaxed`   |        0 GTOP/s |        0 GTOP/s |        0 GTOP/s |
| __i4__                               |                 |                 |                 |
| `nk_dots_packed_i4_v128relaxed`      |        0 GTOP/s |        0 GTOP/s |        0 GTOP/s |
| `nk_dots_symmetric_i4_v128relaxed`   |        0 GTOP/s |        0 GTOP/s |        0 GTOP/s |
| __u4__                               |                 |                 |                 |
| `nk_dots_packed_u4_v128relaxed`      |        0 GTOP/s |        0 GTOP/s |        0 GTOP/s |
| `nk_dots_symmetric_u4_v128relaxed`   |        0 GTOP/s |        0 GTOP/s |        0 GTOP/s |

#### Wasmtime (Cranelift)

| Kernel                               |            256³ |           1024³ |           4096³ |
| :----------------------------------- | --------------: | --------------: | --------------: |
| __f64__                              |                 |                 |                 |
| `nk_dots_packed_f64_v128relaxed`     | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_symmetric_f64_v128relaxed`  | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| __f32__                              |                 |                 |                 |
| `nk_dots_packed_f32_v128relaxed`     | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_symmetric_f32_v128relaxed`  | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| __bf16__                             |                 |                 |                 |
| `nk_dots_packed_bf16_v128relaxed`    | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_symmetric_bf16_v128relaxed` | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| __e5m2__                             |                 |                 |                 |
| `nk_dots_packed_e5m2_v128relaxed`    | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_symmetric_e5m2_v128relaxed` | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| __e4m3__                             |                 |                 |                 |
| `nk_dots_packed_e4m3_v128relaxed`    | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_symmetric_e4m3_v128relaxed` | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| __e2m3__                             |                 |                 |                 |
| `nk_dots_packed_e2m3_v128relaxed`    | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_dots_symmetric_e2m3_v128relaxed` | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| __i8__                               |                 |                 |                 |
| `nk_dots_packed_i8_v128relaxed`      |        0 GTOP/s |        0 GTOP/s |        0 GTOP/s |
| `nk_dots_symmetric_i8_v128relaxed`   |        0 GTOP/s |        0 GTOP/s |        0 GTOP/s |
| __u8__                               |                 |                 |                 |
| `nk_dots_packed_u8_v128relaxed`      |        0 GTOP/s |        0 GTOP/s |        0 GTOP/s |
| `nk_dots_symmetric_u8_v128relaxed`   |        0 GTOP/s |        0 GTOP/s |        0 GTOP/s |
| __i4__                               |                 |                 |                 |
| `nk_dots_packed_i4_v128relaxed`      |        0 GTOP/s |        0 GTOP/s |        0 GTOP/s |
| `nk_dots_symmetric_i4_v128relaxed`   |        0 GTOP/s |        0 GTOP/s |        0 GTOP/s |
| __u4__                               |                 |                 |                 |
| `nk_dots_packed_u4_v128relaxed`      |        0 GTOP/s |        0 GTOP/s |        0 GTOP/s |
| `nk_dots_symmetric_u4_v128relaxed`   |        0 GTOP/s |        0 GTOP/s |        0 GTOP/s |

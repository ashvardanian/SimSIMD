# Sparse Vector Operations in NumKong

NumKong implements set intersection and weighted dot products for sparse vectors stored as sorted arrays of unique indices with optional associated weights.
Set intersection counts common elements between two sorted index arrays; sparse dot product sums the products of weights at matching indices.
Used in inverted-index search, sparse feature matching, and graph intersection queries.
The separate index/weight stream design makes these primitives composable into batched sparse operations and future sparse GEMM workloads.

Set intersection:

```math
|A \cap B| = |\{i : i \in A \land i \in B\}|
```

Sparse dot product:

```math
\text{dot}(a, b) = \sum_{i \in A \cap B} w_a(i) \cdot w_b(i)
```

Reformulating as Python pseudocode:

```python
import numpy as np

def intersect(a_indices: np.ndarray, b_indices: np.ndarray) -> int:
    return len(np.intersect1d(a_indices, b_indices))

def sparse_dot(a_indices: np.ndarray, a_weights: np.ndarray,
               b_indices: np.ndarray, b_weights: np.ndarray) -> float:
    common = np.intersect1d(a_indices, b_indices, return_indices=True)
    return np.dot(a_weights[common[1]], b_weights[common[2]])
```

## Input & Output Types

| Input Type | Output Type | Description                                  |
| ---------- | ----------- | -------------------------------------------- |
| `u16`      | `u64`       | 16-bit index intersection count              |
| `u32`      | `u64`       | 32-bit index intersection count              |
| `u64`      | `u64`       | 64-bit index intersection count              |
| `u32+f32`  | `f32`       | Sparse dot with 32-bit indices, f32 weights  |
| `u16+bf16` | `f32`       | Sparse dot with 16-bit indices, bf16 weights |

## Optimizations

### Adaptive Merge vs Galloping Search

`nk_sparse_intersect_u32_serial` selects between linear merge and galloping (exponential) search based on length ratio: when `longer_length > 64 * shorter_length`, galloping search over the longer array is used.
Linear merge advances two pointers in lockstep at $O(|A| + |B|)$ using branch-free conditional increments: `i += ai < bj; j += ai >= bj` -- no branch misprediction penalty.
Galloping binary-searches the longer array for each element of the shorter at $O(|A| \cdot \log |B|)$: an exponential probe doubles the search range until the target is bracketed, then binary search narrows within.
The crossover at 64x length ratio balances the per-element cost of binary search ($\log_2 |B|$ comparisons) against linear scan's single comparison per advance -- the threshold was chosen empirically, as cache locality favors linear merge at moderate ratios.

### Broadcast-Compare SIMD Intersection on x86

`nk_sparse_intersect_u32_icelake` loads 16 indices from each array into ZMM registers, then rotates one register through multiple positions, comparing each rotation against the other with `VPCMPEQD` to test all 16x16 = 256 pairs.
The rotation approach uses `_mm512_shuffle_epi32` with permutation constants (`_MM_PERM_ADCB`, etc.) to cycle elements through comparison positions -- contending for port 5 (~1cy per shuffle, ~3cy for `_mm512_alignr_epi32`).
Match counts are extracted via `_mm_popcnt_u32` on the comparison masks, accumulating intersection size without materializing matched elements.
Before each 16x16 comparison block, a fast overlap check (`a_max < b_min || b_max < a_min`) skips non-overlapping register loads entirely -- critical for sparse workloads where most pairs have disjoint index ranges.
No native `_mm512_2intersect_epi16` instruction exists in any x86 ISA -- u16 intersection must convert indices to u32 before comparison, halving effective throughput.

### VP2INTERSECT on AMD Turin

`nk_sparse_intersect_u32_turin` uses the `VP2INTERSECT` instruction (Zen5), which produces _two_ 16-bit masks in a single operation -- one indicating which elements of A matched any element of B, and vice versa.
This replaces the entire shuffle-rotate-compare sequence from Ice Lake with a single instruction, eliminating port-5 contention entirely.
Even on Turin, u16 intersection requires zero-extending to u32 first -- no `VP2INTERSECT` variant operates on 16-bit elements.
For u64, `_mm512_2intersect_epi64` processes 8x8 = 64 pairs per instruction -- half the throughput of u32 but still far faster than the Ice Lake shuffle approach.

### SVE2 Set Membership via svmatch and svhistcnt

`nk_sparse_intersect_u16_sve2` uses the `svmatch_u16` instruction -- true hardware set membership testing that matches each element against a 128-bit lane of candidates.
However, `svmatch` only operates on u8 and u16 -- no u32 or u64 variant exists in SVE2.
For u32/u64, `nk_sparse_intersect_u32_sve2` uses `svhistcnt_u32_z` (histogram count): this computes a prefix-match count for each element against preceding elements in the combined register, and a reverse pass captures the upper triangle -- ORing both halves yields the full intersection mask.
NEON (`nk_sparse_intersect_u32_neon`) lacks compress-store entirely -- when intersection results must be materialized (not just counted), the kernel falls back to serial extraction, using `vclz_u32` (count leading zeros) to compute pointer advance steps from comparison masks.

### BF16 Weights in Sparse Dot Products

`nk_sparse_dot_u16bf16_sve2` loads bf16 weights alongside u16 indices, selecting matching weights via `svsel_s16` after intersection detection, then accumulating with `svbfdot_f32` -- a single instruction that multiplies bf16 pairs and adds to an f32 accumulator.
`nk_sparse_dot_u16bf16_turin` zero-extends u16 indices to u32 for `VP2INTERSECT`, then compresses matching bf16 weights with `VPCOMPRESSW` and accumulates via `_mm256_dpbf16_ps` (6cy latency on Genoa).
BF16 weights halve memory traffic compared to f32 (16-bit vs 32-bit per weight) while preserving sufficient precision for learned sparse attention weights and embedding lookups.
The index/weight stream separation enables type-independent intersection (u16/u32/u64 indices) with type-specific accumulation (bf16 or f32 weights) -- the same intersection code path serves both weight types.

### Implications for Sparse GEMM

Current sparse operations handle inner-product dot products -- one pair of sparse vectors at a time.
Extending to batched sparse GEMM (SpMM, SpGEMM) would require simultaneous intersection of multiple sparse vectors -- the broadcast-compare pattern scales naturally, since one document vector's indices can be broadcast against multiple query vectors' indices in the same ZMM/SVE registers.
The 64x galloping threshold is tuned for individual vector pairs; batched workloads with different sparsity patterns per row would benefit from adaptive per-pair threshold selection.
Hardware support remains the bottleneck: no ISA provides native sparse outer-product instructions, and `VP2INTERSECT` exists only on AMD Zen5+ -- Intel Tiger Lake's implementation had 36-41cy latency, making it slower than the manual shuffle approach on Ice Lake.

## Performance

Controlled by `NK_SPARSE_FIRST_LENGTH`, `NK_SPARSE_SECOND_LENGTH`, `NK_SPARSE_INTERSECTION`.
Columns show 1%, 50%, 95% intersection ratio with both `NK_SPARSE_FIRST_LENGTH` and `NK_SPARSE_SECOND_LENGTH` set to the same value of 4096.

### Intel Sapphire Rapids

#### Native

| Kernel                            |     1% |    50% |    95% |
| :-------------------------------- | -----: | -----: | -----: |
| __u16__                           |        |        |        |
| `nk_sparse_intersect_u16_serial`  | 0 GB/s | 0 GB/s | 0 GB/s |
| `nk_sparse_intersect_u16_icelake` | 0 GB/s | 0 GB/s | 0 GB/s |
| `nk_sparse_dot_u16bf16_serial`    | 0 GB/s | 0 GB/s | 0 GB/s |
| __u32__                           |        |        |        |
| `nk_sparse_intersect_u32_serial`  | 0 GB/s | 0 GB/s | 0 GB/s |
| `nk_sparse_intersect_u32_icelake` | 0 GB/s | 0 GB/s | 0 GB/s |
| `nk_sparse_dot_u32f32_serial`     | 0 GB/s | 0 GB/s | 0 GB/s |
| `nk_sparse_dot_u32f32_icelake`    | 0 GB/s | 0 GB/s | 0 GB/s |
| __u64__                           |        |        |        |
| `nk_sparse_intersect_u64_serial`  | 0 GB/s | 0 GB/s | 0 GB/s |
| `nk_sparse_intersect_u64_icelake` | 0 GB/s | 0 GB/s | 0 GB/s |

### Apple M4 Pro

#### Native

| Kernel                           |     1% |    50% |    95% |
| :------------------------------- | -----: | -----: | -----: |
| __u16__                          |        |        |        |
| `nk_sparse_intersect_u16_serial` | 0 GB/s | 0 GB/s | 0 GB/s |
| `nk_sparse_intersect_u16_neon`   | 0 GB/s | 0 GB/s | 0 GB/s |
| `nk_sparse_dot_u16bf16_serial`   | 0 GB/s | 0 GB/s | 0 GB/s |
| __u32__                          |        |        |        |
| `nk_sparse_intersect_u32_serial` | 0 GB/s | 0 GB/s | 0 GB/s |
| `nk_sparse_intersect_u32_neon`   | 0 GB/s | 0 GB/s | 0 GB/s |
| `nk_sparse_dot_u32f32_serial`    | 0 GB/s | 0 GB/s | 0 GB/s |
| __u64__                          |        |        |        |
| `nk_sparse_intersect_u64_serial` | 0 GB/s | 0 GB/s | 0 GB/s |
| `nk_sparse_intersect_u64_neon`   | 0 GB/s | 0 GB/s | 0 GB/s |

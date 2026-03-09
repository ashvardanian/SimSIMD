# MaxSim Late-Interaction Scoring in NumKong

NumKong implements ColBERT-style late-interaction scoring: the MaxSim score sums, over each query token, the minimum angular distance to any document token. A two-stage coarse-to-fine strategy uses i8-quantized screening to find the best document per query, then full-precision refinement computes the final angular distance.

MaxSim score:

```math
\text{MaxSim}(Q, D) = \sum_{i=0}^{m-1} \min_{j=0}^{n-1} \text{angular}(q_i, d_j)
```

Coarse screening finds the best document via i8 dot products as a proxy for argmin angular:

```math
j^* = \arg\max_j \text{dot}_{\text{i8}}(q_i, d_j)
```

Full-precision refinement:

```math
\text{angular}(q_i, d_{j^*}) = 1 - \frac{\text{dot}(q_i, d_{j^*})}{\|q_i\| \cdot \|d_{j^*}\|}
```

Reformulating as Python pseudocode:

```python
import numpy as np

def maxsim(queries: np.ndarray, documents: np.ndarray) -> float:
    score = 0.0
    for q in queries:
        dots = documents @ q
        best = np.argmax(dots)
        d = documents[best]
        angular = 1 - np.dot(q, d) / (np.linalg.norm(q) * np.linalg.norm(d))
        score += angular
    return score
```

## Input & Output Types

| Input Type | Output Type | Description                        |
| ---------- | ----------- | ---------------------------------- |
| `bf16`     | `f32`       | 16-bit brain float, widened output |
| `f32`      | `f32`       | 32-bit IEEE 754 single precision   |
| `f16`      | `f32`       | 16-bit IEEE 754 half precision     |

## Optimizations

### Dual Pre-Packing Advantage

`nk_maxsim_packed_bf16_sme`, `nk_maxsim_packed_f32_sme` benefit from having _both_ query and document matrices pre-packed into identical contiguous formats, unlike `nk_dots_packed` where only B is pre-packed and A is accessed with arbitrary stride.
In the dots GEMM, one ZA tile must be reserved for A-side staging (loading unpacked A rows into the tile array), leaving 3 ZA tiles for accumulation.
With both sides pre-packed, all 4 ZA tiles (ZA0–ZA3) serve as accumulators — a +33% increase in MOPA throughput.
No output matrix materialization: dots_packed writes a full M×N f32 result matrix, while maxsim reduces each query row to a single argmax index in-flight, eliminating the M×N memory round-trip.
Benchmark data (Apple M4, SVL=512):

| Dimensions           | dots_packed GEMM | maxsim fused | GEMM speedup | End-to-end |
| -------------------- | ---------------- | ------------ | ------------ | ---------- |
| 32×128×128 (ColBERT) | 840 GFLOPS       | 1516 GFLOPS  | 1.81×        | 5.10×      |
| 32×256×128           | 1037 GFLOPS      | 1591 GFLOPS  | 1.53×        | 5.17×      |
| 64×512×128           | 1016 GFLOPS      | 1651 GFLOPS  | 1.62×        | 5.42×      |
| 32×128×256           | 859 GFLOPS       | 1725 GFLOPS  | 2.01×        | 4.06×      |
| 32×1024×768 (BERT)   | 1124 GFLOPS      | 1932 GFLOPS  | 1.72×        | 2.61×      |

End-to-end speedup (5×) exceeds GEMM-only speedup (1.5–2×) because maxsim eliminates output materialization and fuses argmax+angular refinement into the tile extraction loop.

### Two-Stage Coarse-to-Fine Scoring

All backends use i8-quantized coarse screening at O(m·n·k) with 1 byte/element instead of 2–4, followed by full-precision refinement at O(m·k) for only the winning pairs.
Break-even at ~4 documents per query — beyond that, coarse screening dominates and the i8 bandwidth advantage compounds.

### ISA-Specific Quantization Ranges

Haswell uses [-79, 79] — `VPMADDUBSW` produces i16 intermediates, must avoid saturation (2×depth×79 < 32767).
Alder Lake and Ice Lake use [-127, 127] — `VPDPBUSD` accumulates directly to i32, no i16 bottleneck.
WASM v128relaxed uses [-63, 63] — `i32x4_relaxed_dot_i8x16_i7x16_add` requires 7-bit operands.
Serial uses [-127, 127].

### XOR-0x80 Bias Correction

`nk_maxsim_packed_bf16_haswell`, `nk_maxsim_packed_f32_alder`, `nk_maxsim_packed_f32_icelake` work around the unsigned×signed operand requirement of `VPMADDUBSW` and `VPDPBUSD`.
Both query and document are signed after quantization, so queries are XOR'd with `0x80` to shift to unsigned range.
Post-multiply correction subtracts $128 \cdot \text{sum\_i8}(d_j)$ per document, where sums are precomputed in packed metadata.

### Vertical Column Extraction on SME

`nk_maxsim_packed_bf16_sme`, `nk_maxsim_packed_f32_sme` accumulate Q×D dot products into ZA tiles (4 tiles ZA0–ZA3, each SVL×SVL).
The argmax operation needs to find the best document for each query.
The naive approach reads rows horizontally (`svread_hor_za32`) and reduces each row with `svmaxv` — but `svmaxv` is a horizontal reduction costing ~8 cycles on typical SVE implementations.
Vertical column extraction flips the access pattern: `svread_ver_za32_f32_m` reads one _column_ of ZA, returning one dot-product score per query for a single document.
Element-wise `svcmpgt_f32` + `svsel_f32` (~1 cycle each) update the running maximum across all queries simultaneously.
For 32 queries × 256 documents: horizontal approach = 32 × 256 × `svmaxv` = 8,192 horizontal reductions; vertical approach = 256 column reads × 1 element-wise `svmax` = 256 vertical reads + 256 comparisons (~270 cycles vs ~2,048 cycles for the argmax phase alone).
The argmax index is tracked in-flight using `svsel` to conditionally update an index vector alongside the maximum values — no separate argmax pass needed.
After finding the best document index per query, full-precision angular refinement uses the originals stored in the packed buffer's third region.

### Three-Region Packed Buffer

All backends use a three-region packed buffer layout: [Header 64B] [i8 vectors, 64B-aligned] [metadata, 64B-aligned] [originals, 64B-aligned].
Per-vector metadata (12 bytes) stores quantization scale, i8 sum (for bias correction), and inverse norm (for angular finalization).
The originals region stores full-precision vectors for refinement via existing `nk_dot_*` primitives.

## Performance

Controlled by `NK_MATRIX_HEIGHT`, `NK_MATRIX_WIDTH`, `NK_MATRIX_DEPTH`.
All values are set to the same value for products of two square-shaped matrices.
Columns show for matrixes with 256, 1024, and 4096 sides.

### Intel Sapphire Rapids

#### Native

| Kernel                              |            256³ |           1024³ |           4096³ |
| :---------------------------------- | --------------: | --------------: | --------------: |
| __bf16__                            |                 |                 |                 |
| `nk_maxsim_packed_bf16_serial`      | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_maxsim_packed_bf16_haswell`     | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_maxsim_packed_bf16_genoa`       | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_maxsim_packed_bf16_sapphireamx` | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_maxsim_packed_bf16_alder`       | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| __f32__                             |                 |                 |                 |
| `nk_maxsim_packed_f32_serial`       | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_maxsim_packed_f32_haswell`      | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_maxsim_packed_f32_icelake`      | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_maxsim_packed_f32_sapphireamx`  | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_maxsim_packed_f32_alder`        | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| __f16__                             |                 |                 |                 |
| `nk_maxsim_packed_f16_serial`       | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_maxsim_packed_f16_haswell`      | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_maxsim_packed_f16_icelake`      | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_maxsim_packed_f16_sapphireamx`  | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_maxsim_packed_f16_alder`        | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |

#### V8 (Chromium)

| Kernel                              |            256³ |           1024³ |           4096³ |
| :---------------------------------- | --------------: | --------------: | --------------: |
| __bf16__                            |                 |                 |                 |
| `nk_maxsim_packed_bf16_v128relaxed` | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| __f32__                             |                 |                 |                 |
| `nk_maxsim_packed_f32_v128relaxed`  | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| __f16__                             |                 |                 |                 |
| `nk_maxsim_packed_f16_v128relaxed`  | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |

#### Wasmtime (Cranelift)

| Kernel                              |            256³ |           1024³ |           4096³ |
| :---------------------------------- | --------------: | --------------: | --------------: |
| __bf16__                            |                 |                 |                 |
| `nk_maxsim_packed_bf16_v128relaxed` | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| __f32__                             |                 |                 |                 |
| `nk_maxsim_packed_f32_v128relaxed`  | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| __f16__                             |                 |                 |                 |
| `nk_maxsim_packed_f16_v128relaxed`  | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |

### Apple M4 Pro

#### Native

| Kernel                           |            256³ |           1024³ |           4096³ |
| :------------------------------- | --------------: | --------------: | --------------: |
| __bf16__                         |                 |                 |                 |
| `nk_maxsim_packed_bf16_serial`   | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_maxsim_packed_bf16_neonsdot` | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_maxsim_packed_bf16_sme`      | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| __f32__                          |                 |                 |                 |
| `nk_maxsim_packed_f32_serial`    | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_maxsim_packed_f32_neonsdot`  | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_maxsim_packed_f32_sme`       | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| __f16__                          |                 |                 |                 |
| `nk_maxsim_packed_f16_serial`    | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_maxsim_packed_f16_neonsdot`  | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| `nk_maxsim_packed_f16_sme`       | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |

#### V8 (Chromium)

| Kernel                              |            256³ |           1024³ |           4096³ |
| :---------------------------------- | --------------: | --------------: | --------------: |
| __bf16__                            |                 |                 |                 |
| `nk_maxsim_packed_bf16_v128relaxed` | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| __f32__                             |                 |                 |                 |
| `nk_maxsim_packed_f32_v128relaxed`  | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| __f16__                             |                 |                 |                 |
| `nk_maxsim_packed_f16_v128relaxed`  | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |

#### Wasmtime (Cranelift)

| Kernel                              |            256³ |           1024³ |           4096³ |
| :---------------------------------- | --------------: | --------------: | --------------: |
| __bf16__                            |                 |                 |                 |
| `nk_maxsim_packed_bf16_v128relaxed` | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| __f32__                             |                 |                 |                 |
| `nk_maxsim_packed_f32_v128relaxed`  | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |
| __f16__                             |                 |                 |                 |
| `nk_maxsim_packed_f16_v128relaxed`  | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP | 0 GTOP/s, 0 ULP |

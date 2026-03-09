# Batched Set Distances in NumKong

NumKong implements batched M×N Hamming and Jaccard distance matrices for binary vectors. The module reuses the dots u1 packing and GEMM infrastructure, converting popcount-of-AND dot products to set distances via precomputed norms.

Hamming distance from batched dot products:

```math
D_{ij} = \|A_i\|_1 + \|B_j\|_1 - 2 \cdot \text{dot}(A_i, B_j)
```

Where dot = popcount(AND), measuring intersection size.

Jaccard distance from batched dot products:

```math
D_{ij} = 1 - \frac{\text{dot}(A_i, B_j)}{\|A_i\|_1 + \|B_j\|_1 - \text{dot}(A_i, B_j)}
```

Reformulating as Python pseudocode:

```python
import numpy as np

def hammings_packed(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    dots = np.array([[np.unpackbits(np.bitwise_and(ai, bj)).sum()
                      for bj in b] for ai in a])
    a_pop = np.array([np.unpackbits(ai).sum() for ai in a])[:, None]
    b_pop = np.array([np.unpackbits(bj).sum() for bj in b])[None, :]
    return a_pop + b_pop - 2 * dots

def jaccards_packed(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    dots = np.array([[np.unpackbits(np.bitwise_and(ai, bj)).sum()
                      for bj in b] for ai in a])
    a_pop = np.array([np.unpackbits(ai).sum() for ai in a])[:, None]
    b_pop = np.array([np.unpackbits(bj).sum() for bj in b])[None, :]
    union = a_pop + b_pop - dots
    return np.where(union > 0, 1.0 - dots / union, 0.0)
```

## Input & Output Types

| Input Type | Output Type | Description                            |
| ---------- | ----------- | -------------------------------------- |
| `u1`       | `u32`       | Binary Hamming distance, packed octets |
| `u1`       | `f32`       | Binary Jaccard distance, packed octets |

## Optimizations

### Hamming and Jaccard from Intersection Counts

`nk_hammings_packed_u1_serial`, `nk_hammings_packed_u1_haswell`, `nk_jaccards_packed_u1_serial`, `nk_jaccards_packed_u1_haswell` reuse the dots u1 GEMM output where each dot product $\text{dot}(a, b) = \text{popcount}(a \mathbin{\&} b) = |A \cap B|$ counts intersection bits.
The L1 norm of a binary vector is its popcount: $|A| = \text{popcount}(a) = \|a\|_1$.
By inclusion-exclusion, $|A \cup B| = |A| + |B| - |A \cap B|$.
Hamming distance counts positions where exactly one bit is set: $D_H = |A| + |B| - 2|A \cap B| = \text{popcount}(a \oplus b)$.
Finalizer `nk_hamming_u32x4_from_dot_serial_` computes `pop_a + pop_b - 2 * dot` in pure u32 arithmetic — no division, no float conversion, no sqrt.
Jaccard distance: $D_J = 1 - \frac{|A \cap B|}{|A \cup B|} = 1 - \frac{\text{dot}}{\text{pop}_a + \text{pop}_b - \text{dot}}$.
Finalizer `nk_jaccard_f32x4_from_dot_serial_` requires u32→f32 cast plus f32 division (~11cy latency on Haswell), making it ~3× more expensive per element than Hamming's integer subtraction chain.
Per-column popcount norms ($\|a\|_1$, $\|b\|_1$) are precomputed during packing and stored in packed buffer metadata, avoiding per-pair recomputation.

### SME Binary Outer-Product Accumulation

`nk_hammings_packed_u1_smebi32`, `nk_jaccards_packed_u1_smebi32` use the `BMOPA` instruction which computes $\text{popcount}(\text{XNOR}(a, b))$ — counting _matching_ bits in a single outer-product operation over 16×16 output tiles with 512-bit depth chunks.
This is fundamentally different from the AND+POPCNT used by scalar/NEON/x86 kernels, which count _intersection_ bits.
Hamming from `BMOPA`: $D_H = \text{depth\_bits} - \text{popcount}(\text{XNOR})$, since XOR popcount (differing bits) is the Hamming distance directly — no per-vector norm correction needed.
Jaccard from `BMOPA`: must convert matching-bit counts to intersection via $|A \cap B| = (\text{popcount}(\text{XNOR}) - (\text{depth\_bits} - |A| - |B|)) / 2$, then apply the Jaccard formula — more arithmetic than the AND-based path.
Streaming mode overhead (~50–100 cycles for `SMSTART`/`SMSTOP`) is amortized across the full M×N output.

## Performance

Controlled by `NK_MATRIX_HEIGHT`, `NK_MATRIX_WIDTH`, `NK_MATRIX_DEPTH`.
All values are set to the same value for products of two square-shaped matrices.
Columns show for matrixes with 256, 1024, and 4096 sides.

### Intel Sapphire Rapids

#### Native

| Kernel                             |     256³ |    1024³ |    4096³ |
| :--------------------------------- | -------: | -------: | -------: |
| __u1__                             |          |          |          |
| `nk_hammings_packed_u1_serial`     | 0 GTOP/s | 0 GTOP/s | 0 GTOP/s |
| `nk_hammings_symmetric_u1_serial`  | 0 GTOP/s | 0 GTOP/s | 0 GTOP/s |
| `nk_jaccards_packed_u1_serial`     | 0 GTOP/s | 0 GTOP/s | 0 GTOP/s |
| `nk_jaccards_symmetric_u1_serial`  | 0 GTOP/s | 0 GTOP/s | 0 GTOP/s |
| `nk_hammings_packed_u1_haswell`    | 0 GTOP/s | 0 GTOP/s | 0 GTOP/s |
| `nk_hammings_symmetric_u1_haswell` | 0 GTOP/s | 0 GTOP/s | 0 GTOP/s |
| `nk_jaccards_packed_u1_haswell`    | 0 GTOP/s | 0 GTOP/s | 0 GTOP/s |
| `nk_jaccards_symmetric_u1_haswell` | 0 GTOP/s | 0 GTOP/s | 0 GTOP/s |
| `nk_hammings_packed_u1_icelake`    | 0 GTOP/s | 0 GTOP/s | 0 GTOP/s |
| `nk_hammings_symmetric_u1_icelake` | 0 GTOP/s | 0 GTOP/s | 0 GTOP/s |
| `nk_jaccards_packed_u1_icelake`    | 0 GTOP/s | 0 GTOP/s | 0 GTOP/s |
| `nk_jaccards_symmetric_u1_icelake` | 0 GTOP/s | 0 GTOP/s | 0 GTOP/s |

### Apple M4 Pro

#### Native

| Kernel                             |     256³ |    1024³ |    4096³ |
| :--------------------------------- | -------: | -------: | -------: |
| __u1__                             |          |          |          |
| `nk_hammings_packed_u1_serial`     | 0 GTOP/s | 0 GTOP/s | 0 GTOP/s |
| `nk_hammings_symmetric_u1_serial`  | 0 GTOP/s | 0 GTOP/s | 0 GTOP/s |
| `nk_jaccards_packed_u1_serial`     | 0 GTOP/s | 0 GTOP/s | 0 GTOP/s |
| `nk_jaccards_symmetric_u1_serial`  | 0 GTOP/s | 0 GTOP/s | 0 GTOP/s |
| `nk_hammings_packed_u1_neon`       | 0 GTOP/s | 0 GTOP/s | 0 GTOP/s |
| `nk_hammings_symmetric_u1_neon`    | 0 GTOP/s | 0 GTOP/s | 0 GTOP/s |
| `nk_jaccards_packed_u1_neon`       | 0 GTOP/s | 0 GTOP/s | 0 GTOP/s |
| `nk_jaccards_symmetric_u1_neon`    | 0 GTOP/s | 0 GTOP/s | 0 GTOP/s |
| `nk_hammings_packed_u1_smebi32`    | 0 GTOP/s | 0 GTOP/s | 0 GTOP/s |
| `nk_hammings_symmetric_u1_smebi32` | 0 GTOP/s | 0 GTOP/s | 0 GTOP/s |
| `nk_jaccards_packed_u1_smebi32`    | 0 GTOP/s | 0 GTOP/s | 0 GTOP/s |
| `nk_jaccards_symmetric_u1_smebi32` | 0 GTOP/s | 0 GTOP/s | 0 GTOP/s |


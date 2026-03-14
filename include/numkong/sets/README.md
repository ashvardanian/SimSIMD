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

The following performance tables are produced by manually re-running `nk_test` and `nk_bench` included internal tools to measure both accuracy and throughput at different input shapes.
The input size is controlled by `NK_MATRIX_HEIGHT`, `NK_MATRIX_WIDTH`, and `NK_MATRIX_DEPTH` environment variables, all set to the same value for batched set operations over square matrices.
Columns show throughput for 256³, 1024³, and 4096³ configurations.
The throughput is measured in GSO/s as Giga scalar operations per second.
Accuracy is reported as ULP (units in last place), the number of representable floating-point values between the result and the exact answer.
Each kernel runs for at least 20 seconds per configuration.
Benchmark threads are pinned to specific cores; on machines with heterogeneous core types (e.g., Apple P/E cores), only the fastest cores are used.
Workloads that significantly degrade CPU frequencies (Intel AMX, Apple SME) run in separate passes to avoid affecting throughput measurements of other kernels.

### Intel Sapphire Rapids

#### Native

| Kernel                             |                     256³ |                    1024³ |                    4096³ |
| :--------------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __u1__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_hammings_packed_u1_serial`     |                109 gso/s |                162 gso/s |                284 gso/s |
| `nk_hammings_symmetric_u1_serial`  |               39.7 gso/s |                133 gso/s |                325 gso/s |
| `nk_jaccards_packed_u1_serial`     |        54.8 gso/s, 0 ulp |         128 gso/s, 0 ulp |         259 gso/s, 0 ulp |
| `nk_jaccards_symmetric_u1_serial`  |        29.8 gso/s, 0 ulp |         110 gso/s, 0 ulp |         292 gso/s, 0 ulp |
| `nk_hammings_packed_u1_haswell`    |               62.3 gso/s |                129 gso/s |                167 gso/s |
| `nk_hammings_symmetric_u1_haswell` |               40.1 gso/s |                136 gso/s |                350 gso/s |
| `nk_jaccards_packed_u1_haswell`    |      53.5 gso/s, 0.3 ulp |       123 gso/s, 0.3 ulp |       170 gso/s, 0.3 ulp |
| `nk_jaccards_symmetric_u1_haswell` |      36.7 gso/s, 0.3 ulp |       135 gso/s, 0.3 ulp |       339 gso/s, 0.3 ulp |
| `nk_hammings_packed_u1_icelake`    |               73.9 gso/s |                340 gso/s |                634 gso/s |
| `nk_hammings_symmetric_u1_icelake` |               52.6 gso/s |                298 gso/s |                942 gso/s |
| `nk_jaccards_packed_u1_icelake`    |      65.6 gso/s, 0.3 ulp |       323 gso/s, 0.3 ulp |       638 gso/s, 0.3 ulp |
| `nk_jaccards_symmetric_u1_icelake` |      47.4 gso/s, 0.3 ulp |       289 gso/s, 0.3 ulp |       894 gso/s, 0.3 ulp |

#### WASM

Measured with Wasmtime v42 (Cranelift backend).

| Kernel                                 |                     256³ |                    1024³ |                    4096³ |
| :------------------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __u1__                                 | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_hammings_packed_u1_serial`         |               35.9 gso/s |               66.6 gso/s |               74.1 gso/s |
| `nk_hammings_packed_u1_v128relaxed`    |               75.9 gso/s |                134 gso/s |                145 gso/s |
| `nk_hammings_symmetric_u1_serial`      |               3.53 gso/s |               13.3 gso/s |               79.4 gso/s |
| `nk_hammings_symmetric_u1_v128relaxed` |               3.62 gso/s |               13.7 gso/s |               80.9 gso/s |
| `nk_jaccards_packed_u1_serial`         |        32.6 gso/s, 0 ulp |        59.9 gso/s, 0 ulp |        72.2 gso/s, 0 ulp |
| `nk_jaccards_packed_u1_v128relaxed`    |        66.9 gso/s, 0 ulp |         129 gso/s, 0 ulp |         142 gso/s, 0 ulp |
| `nk_jaccards_symmetric_u1_serial`      |        3.49 gso/s, 0 ulp |        13.1 gso/s, 0 ulp |        77.9 gso/s, ? ulp |
| `nk_jaccards_symmetric_u1_v128relaxed` |        3.63 gso/s, 0 ulp |        13.8 gso/s, 0 ulp |        81.0 gso/s, ? ulp |

### Apple M4

#### Native

| Kernel                             |                     256³ |                    1024³ |                    4096³ |
| :--------------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __u1__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_hammings_packed_u1_serial`     |                154 gso/s |                204 gso/s |                221 gso/s |
| `nk_hammings_symmetric_u1_serial`  |                101 gso/s |                159 gso/s |                172 gso/s |
| `nk_jaccards_packed_u1_serial`     |         116 gso/s, 0 ulp |         203 gso/s, 0 ulp |         232 gso/s, 0 ulp |
| `nk_jaccards_symmetric_u1_serial`  |        86.3 gso/s, 0 ulp |         157 gso/s, 0 ulp |         176 gso/s, 0 ulp |
| `nk_hammings_packed_u1_neon`       |                315 gso/s |                428 gso/s |                481 gso/s |
| `nk_hammings_symmetric_u1_neon`    |                132 gso/s |                240 gso/s |                294 gso/s |
| `nk_jaccards_packed_u1_neon`       |       266 gso/s, 8.6 ulp |       416 gso/s, 8.6 ulp |       488 gso/s, 8.6 ulp |
| `nk_jaccards_symmetric_u1_neon`    |       129 gso/s, 8.5 ulp |       242 gso/s, 8.5 ulp |       294 gso/s, 8.5 ulp |
| `nk_hammings_packed_u1_smebi32`    |              1,420 gso/s |              2,928 gso/s |              4,027 gso/s |
| `nk_hammings_symmetric_u1_smebi32` |                629 gso/s |              1,438 gso/s |              1,111 gso/s |
| `nk_jaccards_packed_u1_smebi32`    |         273 gso/s, 0 ulp |       1,381 gso/s, 0 ulp |       3,280 gso/s, 0 ulp |
| `nk_jaccards_symmetric_u1_smebi32` |        45.1 gso/s, 0 ulp |         267 gso/s, 0 ulp |         618 gso/s, 0 ulp |

#### WASM

Measured with Wasmtime v42 (Cranelift backend).

| Kernel                                 |                     256³ |                    1024³ |                    4096³ |
| :------------------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __u1__                                 | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_hammings_packed_u1_serial`         |               35.2 gso/s |               47.6 gso/s |               52.8 gso/s |
| `nk_hammings_symmetric_u1_serial`      |               25.4 gso/s |               51.5 gso/s |                129 gso/s |
| `nk_jaccards_packed_u1_serial`         |        30.9 gso/s, 0 ulp |        46.0 gso/s, 0 ulp |        52.7 gso/s, 0 ulp |
| `nk_jaccards_symmetric_u1_serial`      |        22.8 gso/s, 0 ulp |        48.9 gso/s, 0 ulp |         123 gso/s, 0 ulp |
| `nk_hammings_packed_u1_v128relaxed`    |                102 gso/s |                144 gso/s |                160 gso/s |
| `nk_hammings_symmetric_u1_v128relaxed` |               28.2 gso/s |               61.7 gso/s |                175 gso/s |
| `nk_jaccards_packed_u1_v128relaxed`    |        91.2 gso/s, 0 ulp |         140 gso/s, 0 ulp |         172 gso/s, 0 ulp |
| `nk_jaccards_symmetric_u1_v128relaxed` |        26.9 gso/s, 0 ulp |        60.3 gso/s, 0 ulp |         177 gso/s, 0 ulp |

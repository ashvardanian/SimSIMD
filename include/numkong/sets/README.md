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
| `nk_hammings_packed_u1_serial`     |                116 gso/s |                125 gso/s |                279 gso/s |
| `nk_hammings_symmetric_u1_serial`  |               41.3 gso/s |               73.9 gso/s |                300 gso/s |
| `nk_jaccards_packed_u1_serial`     |        56.9 gso/s, 0 ulp |        99.1 gso/s, 0 ulp |         248 gso/s, 0 ulp |
| `nk_jaccards_symmetric_u1_serial`  |        33.2 gso/s, 0 ulp |        59.7 gso/s, 0 ulp |   277 gso/s, 6520000 ulp |
| `nk_hammings_packed_u1_haswell`    |               66.8 gso/s |               90.1 gso/s |                166 gso/s |
| `nk_hammings_symmetric_u1_haswell` |               44.3 gso/s |               77.7 gso/s |                299 gso/s |
| `nk_jaccards_packed_u1_haswell`    |      58.2 gso/s, 0.3 ulp |      74.8 gso/s, 0.3 ulp |       104 gso/s, 0.3 ulp |
| `nk_jaccards_symmetric_u1_haswell` |      41.5 gso/s, 0.3 ulp |      72.3 gso/s, 0.3 ulp |   315 gso/s, 6520000 ulp |
| `nk_hammings_packed_u1_icelake`    |               92.7 gso/s |                242 gso/s |                611 gso/s |
| `nk_hammings_symmetric_u1_icelake` |               62.5 gso/s |                182 gso/s |                903 gso/s |
| `nk_jaccards_packed_u1_icelake`    |      81.1 gso/s, 0.3 ulp |       220 gso/s, 0.3 ulp |       600 gso/s, 0.3 ulp |
| `nk_jaccards_symmetric_u1_icelake` |      53.9 gso/s, 0.3 ulp |       250 gso/s, 0.3 ulp |   694 gso/s, 6570000 ulp |

#### WASM

Measured with Wasmtime v42 (Cranelift backend).

| Kernel                                 |                     256³ |                    1024³ |                    4096³ |
| :------------------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __u1__                                 | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_hammings_packed_u1_serial`         |                  0 gso/s |                  0 gso/s |                  0 gso/s |
| `nk_hammings_packed_u1_v128relaxed`    |                  0 gso/s |                  0 gso/s |                  0 gso/s |
| `nk_hammings_symmetric_u1_serial`      |                  0 gso/s |                  0 gso/s |                  0 gso/s |
| `nk_hammings_symmetric_u1_v128relaxed` |                  0 gso/s |                  0 gso/s |                  0 gso/s |
| `nk_jaccards_packed_u1_serial`         |           ? gso/s, 0 ulp |           ? gso/s, 0 ulp |           ? gso/s, 0 ulp |
| `nk_jaccards_packed_u1_v128relaxed`    |           ? gso/s, 0 ulp |           ? gso/s, 0 ulp |           ? gso/s, 0 ulp |
| `nk_jaccards_symmetric_u1_serial`      |           ? gso/s, 0 ulp |           ? gso/s, 0 ulp |           ? gso/s, 0 ulp |
| `nk_jaccards_symmetric_u1_v128relaxed` |           ? gso/s, 0 ulp |           ? gso/s, 0 ulp |           ? gso/s, 0 ulp |

### Apple M4 Pro

#### Native

| Kernel                             |                     256³ |                    1024³ |                    4096³ |
| :--------------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __u1__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_hammings_packed_u1_serial`     |                  0 gso/s |                  0 gso/s |                  0 gso/s |
| `nk_hammings_symmetric_u1_serial`  |                  0 gso/s |                  0 gso/s |                  0 gso/s |
| `nk_jaccards_packed_u1_serial`     |           0 gso/s, 0 ulp |           0 gso/s, 0 ulp |           0 gso/s, 0 ulp |
| `nk_jaccards_symmetric_u1_serial`  |           0 gso/s, 0 ulp |           0 gso/s, 0 ulp |           0 gso/s, 0 ulp |
| `nk_hammings_packed_u1_neon`       |                  0 gso/s |                  0 gso/s |                  0 gso/s |
| `nk_hammings_symmetric_u1_neon`    |                  0 gso/s |                  0 gso/s |                  0 gso/s |
| `nk_jaccards_packed_u1_neon`       |           0 gso/s, 0 ulp |           0 gso/s, 0 ulp |           0 gso/s, 0 ulp |
| `nk_jaccards_symmetric_u1_neon`    |           0 gso/s, 0 ulp |           0 gso/s, 0 ulp |           0 gso/s, 0 ulp |
| `nk_hammings_packed_u1_smebi32`    |                  0 gso/s |                  0 gso/s |                  0 gso/s |
| `nk_hammings_symmetric_u1_smebi32` |                  0 gso/s |                  0 gso/s |                  0 gso/s |
| `nk_jaccards_packed_u1_smebi32`    |           0 gso/s, 0 ulp |           0 gso/s, 0 ulp |           0 gso/s, 0 ulp |
| `nk_jaccards_symmetric_u1_smebi32` |           0 gso/s, 0 ulp |           0 gso/s, 0 ulp |           0 gso/s, 0 ulp |

#### WASM

Measured with Wasmtime v42 (Cranelift backend).

| Kernel                                 |                     256³ |                    1024³ |                    4096³ |
| :------------------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __u1__                                 | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_hammings_packed_u1_serial`         |                  ? gso/s |                  ? gso/s |                  ? gso/s |
| `nk_hammings_packed_u1_v128relaxed`    |                  ? gso/s |                  ? gso/s |                  ? gso/s |
| `nk_hammings_symmetric_u1_serial`      |                  ? gso/s |                  ? gso/s |                  ? gso/s |
| `nk_hammings_symmetric_u1_v128relaxed` |                  ? gso/s |                  ? gso/s |                  ? gso/s |
| `nk_jaccards_packed_u1_serial`         |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_jaccards_packed_u1_v128relaxed`    |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_jaccards_symmetric_u1_serial`      |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_jaccards_symmetric_u1_v128relaxed` |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |


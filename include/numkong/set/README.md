# Set Similarity Measures in NumKong

NumKong implements set similarity functions for binary and integer vectors: Hamming distance measures the number of differing elements, while Jaccard distance measures the complement of the intersection-over-union ratio.
These are fundamental to locality-sensitive hashing, MinHash sketches, and binary feature matching.

Hamming distance counts the number of positions where elements differ.
For binary vectors packed as octets, this is the popcount of the XOR.
For byte-level vectors, it counts the number of mismatched bytes:

```math
\text{hamming}(a, b) = \sum_{i=0}^{n-1} [a_i \neq b_i]
```

Jaccard distance measures the dissimilarity of two sets.
For binary vectors, the intersection and union are computed via bitwise AND and OR with popcount:

```math
\text{jaccard}(a, b) = 1 - \frac{|A \cap B|}{|A \cup B|} = 1 - \frac{\text{popcount}(a \mathbin{\&} b)}{\text{popcount}(a \mathbin{|} b)}
```

For word-level vectors (MinHash signatures), Jaccard similarity is the fraction of matching elements:

```math
\text{jaccard}(a, b) = 1 - \frac{\sum_{i=0}^{n-1} [a_i = b_i]}{n}
```

Reformulating as Python pseudocode:

```python
import numpy as np

def hamming_bits(a: np.ndarray, b: np.ndarray) -> int:
    return np.unpackbits(np.bitwise_xor(a, b)).sum()

def jaccard_bits(a: np.ndarray, b: np.ndarray) -> float:
    intersection = np.unpackbits(np.bitwise_and(a, b)).sum()
    union = np.unpackbits(np.bitwise_or(a, b)).sum()
    return 1 - intersection / union if union else 0

def jaccard_words(a: np.ndarray, b: np.ndarray) -> float:
    return 1 - np.mean(a == b)
```

## Input & Output Types

| Input Type | Output Type | Description                                 |
| ---------- | ----------- | ------------------------------------------- |
| `u1`       | `u32`       | Binary Hamming distance, packed octets      |
| `u1`       | `f32`       | Binary Jaccard distance, packed octets      |
| `u8`       | `u32`       | Byte-level Hamming distance                 |
| `u16`      | `f32`       | Word-level Jaccard distance, 16-bit MinHash |
| `u32`      | `f32`       | Word-level Jaccard distance, 32-bit MinHash |

## Optimizations

### Harley-Seal Carry-Save Adders for U1

`nk_hamming_u1_haswell`, `nk_jaccard_u1_haswell` amortize the cost of popcount by using Harley-Seal carry-save adder trees.
Instead of computing popcount on every XOR/AND/OR result independently, three intermediate values are combined through a full-adder circuit:

```
ones  = a ^ b ^ c
twos  = (a & b) | (c & (a ^ b))
```

This circuit takes three popcount inputs and produces a ones and twos accumulator, where `twos` has double the weight of `ones`.
By chaining two levels, a fours accumulator is also produced, so the actual `VPSHUFB`-based popcount is called only on the final accumulated ones, twos, and fours values.
The total number of popcount operations is reduced by roughly a factor of three compared to computing popcount on every vector independently.

### Native VPOPCNTQ on Ice Lake

`nk_hamming_u1_icelake`, `nk_jaccard_u1_icelake` use `VPOPCNTQ` on 512-bit vectors, which directly produces per-quadword population counts for 8 quadwords at once.
This single instruction replaces the entire nibble-LUT + Harley-Seal pipeline used on Haswell.
The kernels batch 16 vectors before horizontal reduction to minimize `VPSADBW` overhead, accumulating the per-quadword counts into a running total via `VPADDQ`.

### Jaccard via Precomputed Norms

`nk_jaccard_u1_haswell`, `nk_jaccard_u1_icelake` exploit the identity $|A \cup B| = |A| + |B| - |A \cap B|$ to avoid computing both AND-popcount and OR-popcount in the inner loop.
When vector norms (popcount of each vector) are precomputed and passed via the streaming API, only the intersection popcount is needed per pair, halving the work in the critical path.

### Byte Hamming via VPSADBW

`nk_hamming_u8_haswell`, `nk_hamming_u8_icelake` compute byte-level Hamming distance using XOR to produce per-byte difference indicators, then `VPSADBW` against zero to horizontally sum the nonzero bytes.
XOR produces 0 for equal bytes and nonzero for different ones, and `VPSADBW` sums the absolute values of byte differences within each 64-bit lane.
Since XOR results are either 0 or nonzero (not necessarily 1), the kernel masks XOR output through `VPMIN` with a vector of ones to clamp each byte to 0 or 1 before feeding `VPSADBW`.

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

| Kernel                   |                      256 |                     1024 |                     4096 |
| :----------------------- | -----------------------: | -----------------------: | -----------------------: |
| __u1__                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_hamming_u1_serial`   |                2.74 gb/s |                2.50 gb/s |                4.14 gb/s |
| `nk_jaccard_u1_serial`   |         1.80 gb/s, 0 ulp |         1.51 gb/s, 0 ulp |         2.26 gb/s, 0 ulp |
| `nk_hamming_u1_haswell`  |                14.7 gb/s |                39.8 gb/s |                64.0 gb/s |
| `nk_jaccard_u1_haswell`  |         10.1 gb/s, 0 ulp |         27.0 gb/s, 0 ulp |         44.7 gb/s, 0 ulp |
| `nk_hamming_u1_icelake`  |                22.2 gb/s |                66.2 gb/s |                 100 gb/s |
| `nk_jaccard_u1_icelake`  |         12.2 gb/s, 0 ulp |         40.3 gb/s, 0 ulp |         69.8 gb/s, 0 ulp |
| __u8__                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_hamming_u8_serial`   |                21.8 gb/s |                20.9 gb/s |                19.6 gb/s |
| `nk_hamming_u8_haswell`  |                41.0 gb/s |                30.8 gb/s |                27.4 gb/s |
| `nk_hamming_u8_icelake`  |                93.4 gb/s |                48.1 gb/s |                31.3 gb/s |
| __u16__                  | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_jaccard_u16_serial`  |         53.3 gb/s, 0 ulp |         27.6 gb/s, 0 ulp |         28.6 gb/s, 0 ulp |
| `nk_jaccard_u16_haswell` |         35.5 gb/s, 0 ulp |         25.3 gb/s, 0 ulp |         27.2 gb/s, 0 ulp |
| `nk_jaccard_u16_icelake` |         95.2 gb/s, 0 ulp |         28.5 gb/s, 0 ulp |         31.4 gb/s, 0 ulp |
| __u32__                  | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_jaccard_u32_serial`  |         48.4 gb/s, 0 ulp |         27.7 gb/s, 0 ulp |         29.3 gb/s, 0 ulp |
| `nk_jaccard_u32_haswell` |         33.2 gb/s, 0 ulp |         25.0 gb/s, 0 ulp |         28.0 gb/s, 0 ulp |
| `nk_jaccard_u32_icelake` |         40.2 gb/s, 0 ulp |         27.6 gb/s, 0 ulp |         30.8 gb/s, 0 ulp |

#### WASM

Measured with Wasmtime v42 (Cranelift backend).

| Kernel                       |                      256 |                     1024 |                     4096 |
| :--------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __u1__                       | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_hamming_u1_v128relaxed`  |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_jaccard_u1_v128relaxed`  |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| __u8__                       | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_hamming_u8_v128relaxed`  |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| __u16__                      | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_jaccard_u16_v128relaxed` |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| __u32__                      | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_jaccard_u32_v128relaxed` |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |

### Apple M4 Pro

#### Native

| Kernel                  |                      256 |                     1024 |                     4096 |
| :---------------------- | -----------------------: | -----------------------: | -----------------------: |
| __u1__                  | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_hamming_u1_serial`  |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_jaccard_u1_serial`  |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_hamming_u1_neon`    |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_jaccard_u1_neon`    |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| __u8__                  | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_hamming_u8_serial`  |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_hamming_u8_neon`    |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| __u16__                 | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_jaccard_u16_serial` |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_jaccard_u16_neon`   |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| __u32__                 | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_jaccard_u32_serial` |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| `nk_jaccard_u32_neon`   |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |

#### WASM

Measured with Wasmtime v42 (Cranelift backend).

| Kernel                       |                      256 |                     1024 |                     4096 |
| :--------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __u1__                       | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_hamming_u1_v128relaxed`  |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| `nk_jaccard_u1_v128relaxed`  |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| __u8__                       | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_hamming_u8_v128relaxed`  |                   0 gb/s |                   0 gb/s |                   0 gb/s |
| __u16__                      | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_jaccard_u16_v128relaxed` |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |
| __u32__                      | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_jaccard_u32_v128relaxed` |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |            0 gb/s, 0 ulp |

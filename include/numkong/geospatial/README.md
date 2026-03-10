# Geospatial Distances in NumKong

NumKong implements geodesic distance functions for points on Earth's surface: Haversine computes great-circle distance on a perfect sphere, while Vincenty solves the inverse geodesic problem on the WGS-84 oblate spheroid.
Both operate on arrays of latitude/longitude pairs in radians and produce distances in meters.

The Haversine formula computes the great-circle distance between two points:

```math
\text{haversine}(\phi_1, \lambda_1, \phi_2, \lambda_2) = 2R \arcsin\sqrt{\sin^2\frac{\phi_2 - \phi_1}{2} + \cos\phi_1 \cos\phi_2 \sin^2\frac{\lambda_2 - \lambda_1}{2}}
```

where $R$ is Earth's mean radius and $(\phi, \lambda)$ are latitude and longitude in radians.

Vincenty's formula solves the inverse geodesic problem on an oblate spheroid, iteratively refining the reduced latitude difference until convergence:

```math
\text{vincenty}(\phi_1, \lambda_1, \phi_2, \lambda_2) = b \cdot A \cdot (\sigma - \Delta\sigma)
```

where $a$ and $b$ are the equatorial and polar semi-axes of the WGS-84 ellipsoid, $\sigma$ is the angular separation, and $\Delta\sigma$ is the correction term computed through iterative convergence.

Reformulating as Python pseudocode:

```python
import numpy as np

def haversine(lat1, lon1, lat2, lon2, R=6371000):
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))
```

Input coordinates are in radians, output distances are in meters.

## Input & Output Types

| Input Type | Output Type | Description             |
| ---------- | ----------- | ----------------------- |
| `f64`      | `f64`       | 64-bit double precision |
| `f32`      | `f32`       | 32-bit single precision |

## Optimizations

### Trigonometric Polynomial Approximations

`nk_haversine_f32_haswell`, `nk_haversine_f32_skylake`, `nk_haversine_f32_neon` replace `libm` sin, cos, atan2, and asin with SIMD polynomial approximations achieving approximately 3.5 ULP accuracy.
Range reduction maps the input angle to $[-\pi/4, \pi/4]$ using Cody-Waite extended precision constants, then an odd-degree minimax polynomial evaluates sin and an even-degree polynomial evaluates cos.
The `f32` kernels use 5-term polynomials while `f64` kernels use 11-term polynomials for the extra precision required by double-precision inputs.
This avoids the latency of scalar `libm` calls — each trigonometric evaluation would otherwise serialize through a single execution port, while the polynomial chains pipeline across multiple FMA units.

### Vincenty Iterative Convergence with Masked Lanes

`nk_vincenty_f64_haswell`, `nk_vincenty_f64_skylake`, `nk_vincenty_f64_neon` implement the full Vincenty inverse formula with up to 100 iterations and a convergence threshold of $10^{-12}$ radians (approximately 6 micrometers on Earth's surface).
Each SIMD lane may converge at a different iteration count, so the kernel accumulates a `converged_mask` via `_mm256_or_pd(converged_mask, newly_converged)` and selectively freezes converged lanes with `_mm256_blendv_pd(lambda_new, lambda, converged_mask)`.
Early exit uses `_mm256_movemask_pd` — when all 4 bits (for `f64`) or 8 bits (for `f32`) are set, the loop breaks.
Coincident points and equatorial edge cases are handled by blending safe values (ones) into the intermediate terms to avoid division by zero, without requiring branches that would diverge across SIMD lanes.

### Haversine Without Final Arc Conversion

`nk_haversine_f32_haswell`, `nk_haversine_f64_haswell` support a similarity mode where the haversine formula involves $2R \cdot \text{asin}(\sqrt{h})$ and the intermediate value $h = \sin^2(\Delta\phi/2) + \cos\phi_1 \cos\phi_2 \cdot \sin^2(\Delta\lambda/2)$ is monotonic with distance.
For ranking and comparison use cases, comparing $h$ values directly produces the same ordering as comparing full Haversine distances, since both asin and sqrt are monotonically increasing.
This eliminates the two most expensive operations in the pipeline.
The kernels compute the full distance by default, but the streaming API can optionally skip the final conversion when only relative ordering is needed.

## Performance

The following performance tables are produced by manually re-running `nk_test` and `nk_bench` included internal tools to measure both accuracy and throughput at different input shapes.
The input size is controlled by the `NK_GEOSPATIAL_MAX_ANGLE` environment variable and set to ≤1°, ≤30°, and ≤180° maximum angular separation between pairs of coordinates.
The larger the angular separation between pairs, the longer the algorithm may take to converge and the higher the error.
The throughput is measured in MP/s as the number of Millions of pairwise point distances computed per second - amortized for a large batch size, with `NK_DENSE_DIMENSIONS=1536` by default.
Accuracy is reported as max absolute error in meters using SI prefixes (nm, µm, mm, m), measuring the distance difference from Vincenty's formula computed at double-double (f118) precision.
Each kernel runs for at least 20 seconds per configuration.
Benchmark threads are pinned to specific cores; on machines with heterogeneous core types (e.g., Apple P/E cores), only the fastest cores are used.
Workloads that significantly degrade CPU frequencies (Intel AMX, Apple SME) run in separate passes to avoid affecting throughput measurements of other kernels.

### Intel Sapphire Rapids

#### Native

| Kernel                     |                      ≤1° |                     ≤30° |                    ≤180° |
| :------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f64__                    | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_haversine_f64_serial`  |       61.2 mp/s, 1.12 km |       59.2 mp/s, 32.8 km |        66.3 mp/s, 148 km |
| `nk_vincenty_f64_serial`   |       18.0 mp/s, 2.07 nm |       15.1 mp/s, 2.79 nm |        14.6 mp/s, 594 nm |
| `nk_haversine_f64_haswell` |      2,340 mp/s, 1.12 km |      2,420 mp/s, 32.8 km |       2,250 mp/s, 148 km |
| `nk_vincenty_f64_haswell`  |        375 mp/s, 1.96 nm |        288 mp/s, 2.79 nm |         229 mp/s, 594 nm |
| `nk_haversine_f64_skylake` |      3,480 mp/s, 1.12 km |      3,810 mp/s, 32.8 km |       3,230 mp/s, 148 km |
| `nk_vincenty_f64_skylake`  |        623 mp/s, 6.32 µm |        593 mp/s, 6.16 µm |        342 mp/s, 6.32 µm |
| __f32__                    | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_haversine_f32_serial`  |      786 mp/s, 20,000 km |        826 mp/s, 32.8 km |         964 mp/s, 146 km |
| `nk_vincenty_f32_serial`   |     49.0 mp/s, 20,000 km |        36.4 mp/s, 12.2 m |        28.9 mp/s, 22.0 m |
| `nk_haversine_f32_haswell` |    4,200 mp/s, 20,000 km |      4,440 mp/s, 32.8 km |       3,440 mp/s, 146 km |
| `nk_vincenty_f32_haswell`  |         891 mp/s, 11.7 m |         704 mp/s, 12.5 m |         268 mp/s, 22.0 m |
| `nk_haversine_f32_skylake` |    5,920 mp/s, 20,000 km |      6,550 mp/s, 32.8 km |       5,260 mp/s, 146 km |
| `nk_vincenty_f32_skylake`  |       1,290 mp/s, 11.7 m |       1,350 mp/s, 12.5 m |         325 mp/s, 22.0 m |

#### WASM

Measured with Wasmtime v42 (Cranelift backend).

| Kernel                         |                      ≤1° |                     ≤30° |                    ≤180° |
| :----------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f64__                        | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_haversine_f64_serial`      |              ? mp/s, ? m |              ? mp/s, ? m |              ? mp/s, ? m |
| `nk_vincenty_f64_serial`       |              ? mp/s, ? m |              ? mp/s, ? m |              ? mp/s, ? m |
| `nk_haversine_f64_v128relaxed` |              ? mp/s, ? m |              ? mp/s, ? m |              ? mp/s, ? m |
| `nk_vincenty_f64_v128relaxed`  |              ? mp/s, ? m |              ? mp/s, ? m |              ? mp/s, ? m |
| __f32__                        | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_haversine_f32_serial`      |              ? mp/s, ? m |              ? mp/s, ? m |              ? mp/s, ? m |
| `nk_vincenty_f32_serial`       |              ? mp/s, ? m |              ? mp/s, ? m |              ? mp/s, ? m |
| `nk_haversine_f32_v128relaxed` |              ? mp/s, ? m |              ? mp/s, ? m |              ? mp/s, ? m |
| `nk_vincenty_f32_v128relaxed`  |              ? mp/s, ? m |              ? mp/s, ? m |              ? mp/s, ? m |

### Apple M4 Pro

#### Native

| Kernel                    |                      ≤1° |                     ≤30° |                    ≤180° |
| :------------------------ | -----------------------: | -----------------------: | -----------------------: |
| __f64__                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_haversine_f64_serial` |              ? mp/s, ? m |              ? mp/s, ? m |              ? mp/s, ? m |
| `nk_vincenty_f64_serial`  |              ? mp/s, ? m |              ? mp/s, ? m |              ? mp/s, ? m |
| `nk_haversine_f64_neon`   |              ? mp/s, ? m |              ? mp/s, ? m |              ? mp/s, ? m |
| `nk_vincenty_f64_neon`    |              ? mp/s, ? m |              ? mp/s, ? m |              ? mp/s, ? m |
| __f32__                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_haversine_f32_serial` |              ? mp/s, ? m |              ? mp/s, ? m |              ? mp/s, ? m |
| `nk_vincenty_f32_serial`  |              ? mp/s, ? m |              ? mp/s, ? m |              ? mp/s, ? m |
| `nk_haversine_f32_neon`   |              ? mp/s, ? m |              ? mp/s, ? m |              ? mp/s, ? m |
| `nk_vincenty_f32_neon`    |              ? mp/s, ? m |              ? mp/s, ? m |              ? mp/s, ? m |

#### WASM

Measured with Wasmtime v42 (Cranelift backend).

| Kernel                         |                      ≤1° |                     ≤30° |                    ≤180° |
| :----------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f64__                        | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_haversine_f64_serial`      |              ? mp/s, ? m |              ? mp/s, ? m |              ? mp/s, ? m |
| `nk_vincenty_f64_serial`       |              ? mp/s, ? m |              ? mp/s, ? m |              ? mp/s, ? m |
| `nk_haversine_f64_v128relaxed` |              ? mp/s, ? m |              ? mp/s, ? m |              ? mp/s, ? m |
| `nk_vincenty_f64_v128relaxed`  |              ? mp/s, ? m |              ? mp/s, ? m |              ? mp/s, ? m |
| __f32__                        | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_haversine_f32_serial`      |              ? mp/s, ? m |              ? mp/s, ? m |              ? mp/s, ? m |
| `nk_vincenty_f32_serial`       |              ? mp/s, ? m |              ? mp/s, ? m |              ? mp/s, ? m |
| `nk_haversine_f32_v128relaxed` |              ? mp/s, ? m |              ? mp/s, ? m |              ? mp/s, ? m |
| `nk_vincenty_f32_v128relaxed`  |              ? mp/s, ? m |              ? mp/s, ? m |              ? mp/s, ? m |

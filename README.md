# SimSIMD

SIMD-accelerated similarity measures, metrics, distance functions for x86 and Arm.
Tuned for Machine Learning applications, mid-size vectors with 100-1024 dimensions.
For Cosine (Angular) distance, the most common metric in AI, one can expect following performance.

| Method | Vectors | Any Length | Speed on 256b | Speed on 1024b |
| :----- | :------ | :--------- | ------------: | -------------: |
| Serial | `f32`   | ✅          |        5 GB/s |         5 GB/s |
| SVE    | `f32`   | ✅          |       34 GB/s |        40 GB/s |
| SVE    | `f16`   | ✅          |       28 GB/s |        35 GB/s |
| NEON   | `f16`   | ❌          |       16 GB/s |        18 GB/s |

> The benchmarks were done on Arm-based "Graviton 3" CPUs powering AWS `c7g.metal` instances.
> We only use Arm NEON implementation with vectors lengths that are multiples of 128 bits, avoiding any additional head or tail `for` loops for misaligned data.
> By default, we use GCC12, `-O3`, `-march=native` for benchmarks.
> Serial versions imply auto-vectorization pragmas.

Need something like this in your CMake-based project?

```cmake
FetchContent_Declare(
    simsimd
    GIT_REPOSITORY https://github.com/ashvardanian/simsimd.git
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(simsimd)
include_directories(${simsimd_SOURCE_DIR}/include)
```

Want to use in Python with [USearch](https://github.com/unum-cloud/usearch)?

```py
from usearch import Index, CompiledMetric, MetricKind, MetricSignature
from simsimd import to_int, cos_f32x4_neon

metric = CompiledMetric(
    pointer=to_int(cos_f32x4_neon),
    kind=MetricKind.Cos,
    signature=MetricSignature.ArrayArraySize,
)

index = Index(256, metric=metric)
```

## Available Metrics

In the C99 interface all function are prepended with `simsimd_` namespace prefix.
Signature defines the number of arguments:

- two pointers, and length,
- two pointers.

The latter is obviously intended for cases, where the number of dimensions is hard-coded.
Constraints define the limitations on the number of dimensions that an argument vector can have.

| Name                    | Signature | ISA Extesion |  Constraints   |
| :---------------------- | :-------: | :----------: | :------------: |
| `dot_f32_sve`           |    ✳️✳️📏    |   Arm SVE    |                |
| `dot_f32x4_neon`        |    ✳️✳️📏    |   Arm NEON   |  `d % 4 == 0`  |
| `cos_f32_sve`           |    ✳️✳️📏    |   Arm SVE    |                |
| `cos_f16_sve`           |    ✳️✳️📏    |   Arm SVE    |                |
| `cos_f16x4_neon`        |    ✳️✳️📏    |   Arm NEON   |  `d % 4 == 0`  |
| `cos_i8x16_neon`        |    ✳️✳️📏    |   Arm NEON   | `d % 16 == 0`  |
| `cos_f32x4_neon`        |    ✳️✳️📏    |   Arm NEON   |  `d % 4 == 0`  |
| `cos_f16x16_avx512`     |    ✳️✳️📏    | x86 AVX-512  | `d % 16 == 0`  |
| `cos_f32x4_avx2`        |    ✳️✳️📏    |   x86 AVX2   |  `d % 4 == 0`  |
| `l2sq_f32_sve`          |    ✳️✳️📏    |   Arm SVE    |                |
| `l2sq_f16_sve`          |    ✳️✳️📏    |   Arm SVE    |                |
| `hamming_b1x8_sve`      |    ✳️✳️📏    |   Arm SVE    |  `d % 8 == 0`  |
| `hamming_b1x128_sve`    |    ✳️✳️📏    |   Arm SVE    | `d % 128 == 0` |
| `hamming_b1x128_avx512` |    ✳️✳️📏    | x86 AVX-512  | `d % 128 == 0` |
| `tanimoto_b1x8_naive`   |    ✳️✳️📏    |              |  `d % 8 == 0`  |
| `tanimoto_maccs_naive`  |    ✳️✳️     |              |   `d == 166`   |
| `tanimoto_maccs_neon`   |    ✳️✳️     |   Arm NEON   |   `d == 166`   |
| `tanimoto_maccs_sve`    |    ✳️✳️     |   Arm SVE    |   `d == 166`   |
| `tanimoto_maccs_avx512` |    ✳️✳️     | x86 AVX-512  |   `d == 166`   |

## Benchmarks

The benchmarks are repeated for every function with a different number of cores involved.
Light-weight distance functions would be memory bound, implying that multi-core performance may be lower, if the bus bandwidth is not enough to saturate all the cores.
Similarly, heavy-weight distance functions running on all cores may result in CPU frequency downclocking.
This is well illustrated by the single-core performance of `i9-13950HX`, equipped with DDR5 memory.

| Method                 | Threads | Speed on 1024b |
| :--------------------- | ------: | -------------: |
| `tanimoto_maccs_naive` |       1 |       2.8 Gb/s |
| `tanimoto_maccs_naive` |      32 |       1.2 Gb/s |
| `cos_f32_naive`        |       1 |      15.3 Gb/s |
| `cos_f32_naive`        |      32 |       4.5 Gb/s |
| `dot_f32x4_avx2`       |       1 |      96.2 Gb/s |
| `dot_f32x4_avx2`       |      32 |      23.6 Gb/s |
| `cos_f32x4_avx2`       |       1 |      56.3 Gb/s |
| `cos_f32x4_avx2`       |      32 |      13.9 Gb/s |

To replicate on your hardware, please run following on Linux:

```sh
git clone https://github.com/ashvardanian/SimSIMD.git && cd SimSIMD
cmake -DCMAKE_BUILD_TYPE=Release -DSIMSIMD_BUILD_BENCHMARKS=1 -B ./build && make -C ./build && ./build/simsimd_bench
```

MacOS:

```sh
brew install llvm
git clone https://github.com/ashvardanian/SimSIMD.git && cd SimSIMD
cmake -B ./build \
    -DCMAKE_C_COMPILER="/opt/homebrew/opt/llvm/bin/clang" \
    -DCMAKE_CXX_COMPILER="/opt/homebrew/opt/llvm/bin/clang++" \
    -DSIMSIMD_BUILD_BENCHMARKS=1 \
    && \
    make -C ./build -j && ./build/simsimd_bench
```

Install and test locally:

```sh
pip install -e . && pytest python/test.py -s -x
```

# SimSIMD

SIMD-accelerated similarity measures, metrics, distance functions for x86 and Arm.
Tuned for Machine Learning applications, mid-size vectors with 100-512 dimensions.

| Distance          | Serial |     x86 AVX     | Arm NEON | Arm SVE |
| :---------------- | :----: | :-------------: | :------: | :-----: |
| `f32` Dot Product |   ✅    |    AVX2-FMA     |    ✅     |    ✅    |
| `f32` Cosine      |   ✅    |    AVX2-FMA     |    ✅     |    ✅    |
| `f32` Euclidean   |   ✅    |        ❌        |    ❌     |    ✅    |
|                   |        |                 |          |         |
| `f16` Dot Product |   ✅    |        ❌        |    ❌     |    ✅    |
| `f16` Euclidean   |   ✅    |        ❌        |    ❌     |    ✅    |
|                   |        |                 |          |         |
| `u1` Hamming      |   ✅    | AVX512VPOPCNTDQ |    ✅     |    ✅    |

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

## Benchmarks

By default, we use GCC12, `-O3`, `-march=native` for benchmarks.
Serial versions imply auto-vectorization pragmas.

---

Cosine distance performance on a single core of a 64-core Arm-based "Graviton 3" CPUs powering AWS `c7g.metal` instances:

| Method | Vectors    | Any Length | Performance |
| :----- | :--------- | :--------- | ----------: |
| Serial | `f32` x16  | ✅          |      5 GB/s |
| Serial | `f32` x256 | ✅          |      5 GB/s |
|        |            |            |             |
| NEON   | `f32` x16  | ❌          |     18 GB/s |
| NEON   | `f32` x256 | ❌          |     29 GB/s |
|        |            |            |             |
| SVE    | `f32` x16  | ✅          |     15 GB/s |
| SVE    | `f32` x256 | ✅          |     39 GB/s |

We only use Arm NEON implementation with vectors lengths that are multiples of 128 bits, avoiding any additional head or tail `for` loops for misaligned data.
SVE looses to NEON on very short vectors, but outperforms on longer sequences.

---

On the x86 AMD Zen2 cores making up the 64-core Threadripper PRO 3995WX the numbers are:

| Method  | Vectors    | Any Length | Performance |
| :------ | :--------- | :--------- | ----------: |
| Serial  | `f32` x16  | ✅          |      9 GB/s |
| Serial  | `f32` x256 | ✅          |     10 GB/s |
|         |            |            |             |
| AVX-FMA | `f32` x16  | ❌          |     20 GB/s |
| AVX-FMA | `f32` x256 | ❌          |     24 GB/s |

The gap between auto-vectorized code and directly using 128-bit registers is much less pronounced.
With AVX2 and 256-bit registers the results should be better, but would be less broadly applicable.

---

To replicate on your hardware, please run following on Linux:

```sh
cmake -DCMAKE_BUILD_TYPE=Release -DSIMSIMD_BUILD_BENCHMARKS=1 -B ./build && make -C ./build && ./build/simsimd_bench
```

MacOS:

```sh
brew install llvm
cmake -B ./build \
    -DCMAKE_C_COMPILER="/opt/homebrew/opt/llvm/bin/clang" \
    -DCMAKE_CXX_COMPILER="/opt/homebrew/opt/llvm/bin/clang++" \
    -DSIMSIMD_BUILD_BENCHMARKS=1 \
    && \
    make -C ./build -j && ./build/simsimd_bench
```
# SimSIMD 📏

## The `scipy.spatial.distance` of a healthy person

SimSIMD implements most commonly used similarity measures using SIMD intrinsics, that few compilers can produce, and even fewer do it well.
This includes conventional AVX2 instructions on x86 and NEON on Arm, as well as lesser known AVX-512 instructions on x86 and Scalable Vector Extensions on Arm.
They are tuned for Machine Learning applications and mid-size vectors with 100-1024 dimensions.
It includes the Euclidean (L2), Inner Product, and Cosine (Angular) distances implemented for `f32`, `f16`, and `i8` vectors.

## Benchmarks

Let's assume you have 10'000 embeddings obtained from OpenAI Ada API.
Those have 1536 dimensions.
You are running on the Apple M2 Pro Arm CPU with NEON support.

| Conventional                         | SimSIMD               | `f32` improvement | `f16` improvement | `i8` improvement |
| :----------------------------------- | :-------------------- | ----------------: | ----------------: | ---------------: |
| `scipy.spatial.distance.cosine`      | `simsimd.cosine`      |          __39 x__ |          __84 x__ |        __196 x__ |
| `scipy.spatial.distance.sqeuclidean` | `simsimd.sqeuclidean` |           __8 x__ |          __25 x__ |         __22 x__ |
| `numpy.inner`                        | `simsimd.inner`       |           __3 x__ |          __10 x__ |         __18 x__ |

On modern Intel Sapphire Rapids platform we've obtained the following numbers, comparing SimSIMD to autovectorized-code using GCC 12.
As it can be clearly seen, auto-vectorization works well for single-precision `float` and single-byte `int8_t`, but it fails on `_Float16`, supported by the C language since 2011.

|               | GCC 12 `f32` | GCC 12 `f16` | SimSIMD AVX-512 `f16` | `f16` improvement |
| :------------ | -----------: | -----------: | --------------------: | ----------------: |
| `cosine`      |     3.28 M/s |   336.29 k/s |              6.88 M/s |          __20 x__ |
| `sqeuclidean` |     4.62 M/s |   147.25 k/s |              5.32 M/s |          __36 x__ |
| `inner`       |     3.81 M/s |   192.02 k/s |              5.99 M/s |          __31 x__ |

Using SVE on Arm and masked loads on AVX-512, the implementations almost entirely avoid scalar code.
The Python binding is equally efficient avoiding PyBind11, SWIG or any other high-level tools, and using CPython C API directly, but still avoiding expensive functions, like the `PyArg_ParseTuple`.

## Using in Python

```sh
pip install simsimd
```

Computing the distance between two vectors:

```py
import simsimd, numpy

vec1 = numpy.random.randn(1536).astype(np.float32)
vec2 = numpy.random.randn(1536).astype(np.float32)
dist = simsimd.cosine(vec1, vec2)
```

Computing the distance between two batches:

```py
batch1 = numpy.random.randn(100, 1536).astype(np.float32)
batch2 = numpy.random.randn(100, 1536).astype(np.float32)
dist = simsimd.cosine(batch1, batch2)
```

Computing the distances between all possible pairs of rows in two matrices, like [`scipy.spatial.distance.cdist`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html).

```py
matrix1 = numpy.random.randn(1000, 1536).astype(np.float32)
matrix2 = numpy.random.randn(10, 1536).astype(np.float32)
distances = simsimd.cdist(matrix1, matrix2, metric="cosine")
```

By default, all the function use a single CPU core.
On Linux, you can also pass the `threads=0` argument to parallelize across all available cores, or any custom number.

```py
distances = simsimd.cdist(matrix1, matrix2, metric="cosine", threads=0)
```

Computing all pairwise distances between all rows of a single matrix, like [`scipy.spatial.distance.pdist`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html).

```py
distances = simsimd.pdist(matrix1, metric="cosine", threads=0)
```

You can also print supported hardware backends using `simsimd.get_capabilites()`.

### Using Python API with USearch

Want to use it in Python with [USearch](https://github.com/unum-cloud/usearch)?

```py
from usearch import Index, CompiledMetric, MetricKind, MetricSignature
from simsimd import pointer_to_sqeuclidean, pointer_to_cosine, pointer_to_inner

metric = CompiledMetric(
    pointer=pointer_to_cosine("f16"),
    kind=MetricKind.Cos,
    signature=MetricSignature.ArrayArraySize,
)

index = Index(256, metric=metric)
```

## Using in C

To use the `_Float16` functionality, you may need C 11.
For the rest - C 99 is enough.
To integrate with CMake-based project:

```cmake
FetchContent_Declare(
    simsimd
    GIT_REPOSITORY https://github.com/ashvardanian/simsimd.git
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(simsimd)
include_directories(${simsimd_SOURCE_DIR}/include)
```

To enable SimSIMD integration in USearch, just compile it with `USEARCH_USE_SIMSIMD=1`.
It's the default behaviour on most platforms.

## Roadmap

- [ ] Expose Hamming and Tanimoto distances in Python.
- [ ] Intel AMX instructions. Intrinsics only work in Intel's most recent compiler.

Wanna join the development?
Use this command to rerun the experiments:

```sh
cmake -DCMAKE_BUILD_TYPE=Release -DSIMSIMD_BUILD_BENCHMARKS=1 -B ./build_release && make -C ./build_release && ./build_release/simsimd_bench
```

Install and test with PyTest locally:

```sh
pip install -e . && pytest python/test.py -s -x
```

# SimSIMD 📏

<div>
<a href="https://pypi.org/project/simsimd/"> <img alt="PyPI - Downloads" src="https://img.shields.io/pypi/dm/simsimd?label=pypi%20downloads"> </a>
<a href="https://www.npmjs.com/package/simsimd"> <img alt="npm" src="https://img.shields.io/npm/dy/simsimd?label=npm%20dowloads"> </a>
<img alt="GitHub code size in bytes" src="https://img.shields.io/github/languages/code-size/ashvardanian/simsimd">
</div>

## Efficient Alternative to [`scipy.spatial.distance`][scipy] and [`numpy.inner`][numpy]

SimSIMD leverages SIMD intrinsics, capabilities that only select compilers effectively utilize. This framework supports conventional AVX2 instructions on x86, NEON on Arm, as well as __rare__ AVX-512 FP16 instructions on x86 and Scalable Vector Extensions (SVE) on Arm. Designed specifically for Machine Learning contexts, it's optimized for handling high-dimensional vector embeddings.

- ✅ __3-200x faster__ than NumPy and SciPy distance functions.
- ✅ Euclidean (L2), Inner Product, and Cosine (Angular) spatial distances.
- ✅ Hamming (~ Manhattan) and Jaccard (~ Tanimoto) binary distances.
- ✅ Kullback-Leibler and Jensen–Shannon divergences for probability distributions.
- ✅ Single-precision `f32`, half-precision `f16`, `i8`, and binary vectors.
- ✅ Compatible with GCC and Clang on MacOS and Linux, and MinGW on Windows.
- ✅ Compatible with NumPy, PyTorch, TensorFlow, and other tensors.
- ✅ Has __no dependencies__, not even LibC.
- ✅ [JavaScript API](#using-simsimd-in-javascript).
- ✅ [C API](#using-simsimd-in-c).

[scipy]: https://docs.scipy.org/doc/scipy/reference/spatial.distance.html#module-scipy.spatial.distance
[numpy]: https://numpy.org/doc/stable/reference/generated/numpy.inner.html

## Benchmarks

### Apple M2 Pro

Given 1000 embeddings from OpenAI Ada API with 1536 dimensions, running on the Apple M2 Pro Arm CPU with NEON support, here's how SimSIMD performs against conventional methods:

| Kind           | `f32` improvement | `f16` improvement | `i8` improvement | Conventional method                    | SimSIMD         |
| :------------- | ----------------: | ----------------: | ---------------: | :------------------------------------- | :-------------- |
| Cosine         |          __32 x__ |          __79 x__ |        __133 x__ | `scipy.spatial.distance.cosine`        | `cosine`        |
| Euclidean ²    |           __5 x__ |          __26 x__ |         __17 x__ | `scipy.spatial.distance.sqeuclidean`   | `sqeuclidean`   |
| Inner Product  |           __2 x__ |           __9 x__ |         __18 x__ | `numpy.inner`                          | `inner`         |
| Jensen Shannon |          __31 x__ |          __53 x__ |                  | `scipy.spatial.distance.jensenshannon` | `jensenshannon` |

### Intel Sapphire Rapids

On the Intel Sapphire Rapids platform, SimSIMD was benchmarked against auto-vectorized code using GCC 12. GCC handles single-precision `float`, but might not be the best choice for `int8` and `_Float16` arrays, which has been part of the C language since 2011.

| Kind           | GCC 12 `f32` | GCC 12 `f16` | SimSIMD `f16` | `f16` improvement |
| :------------- | -----------: | -----------: | ------------: | ----------------: |
| Cosine         |     3.28 M/s | _336.29 k/s_ |    _6.88 M/s_ |          __20 x__ |
| Euclidean ²    |     4.62 M/s | _147.25 k/s_ |    _5.32 M/s_ |          __36 x__ |
| Inner Product  |     3.81 M/s | _192.02 k/s_ |    _5.99 M/s_ |          __31 x__ |
| Jensen Shannon |     1.18 M/s |  _18.13 k/s_ |    _2.14 M/s_ |         __118 x__ |

__Technical Insights__:

- [Uses Arm SVE and x86 AVX-512's masked loads to eliminate tail `for`-loops](https://ashvardanian.com/posts/simsimd-faster-scipy/#tails-of-the-past-the-significance-of-masked-loads).
- [Uses AVX-512 FP16 for half-precision operations, that few compilers vectorize](https://ashvardanian.com/posts/simsimd-faster-scipy/#the-challenge-of-f16).
- [Substitutes LibC's `sqrt` calls with bithacks using Jan Kadlec's constant](https://ashvardanian.com/posts/simsimd-faster-scipy/#bonus-section-bypassing-sqrt-and-libc-dependencies).
- Avoids slow PyBind11 and SWIG, directly using the CPython C API.
- Avoids slow `PyArg_ParseTuple` and manually unpacks argument tuples.

__Broader Benchmarking Results__:

- [Apple M2 Pro](https://ashvardanian.com/posts/simsimd-faster-scipy/#appendix-1-performance-on-apple-m2-pro).
- [4th Gen Intel Xeon Platinum](https://ashvardanian.com/posts/simsimd-faster-scipy/#appendix-2-performance-on-4th-gen-intel-xeon-platinum-8480).
- [AWS Graviton 3](https://ashvardanian.com/posts/simsimd-faster-scipy/#appendix-3-performance-on-aws-graviton-3).

## Using in Python

### Installation

```sh
pip install simsimd
```

### Distance Between 2 Vectors

```py
import simsimd
import numpy as np

vec1 = np.random.randn(1536).astype(np.float32)
vec2 = np.random.randn(1536).astype(np.float32)
dist = simsimd.cosine(vec1, vec2)
```

Supported functions include `cosine`, `inner`, `sqeuclidean`, `hamming`, and `jaccard`.

### Distance Between 2 Batches

```py
batch1 = np.random.randn(100, 1536).astype(np.float32)
batch2 = np.random.randn(100, 1536).astype(np.float32)
dist = simsimd.cosine(batch1, batch2)
```

If either batch has more than one vector, the other batch must have one or same number of vectors.
If it contains just one, the value is broadcasted.

### All Pairwise Distances

For calculating distances between all possible pairs of rows across two matrices (akin to [`scipy.spatial.distance.cdist`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html)):

```py
matrix1 = np.random.randn(1000, 1536).astype(np.float32)
matrix2 = np.random.randn(10, 1536).astype(np.float32)
distances = simsimd.cdist(matrix1, matrix2, metric="cosine")
```

### Multithreading

By default, computations use a single CPU core. To optimize and utilize all CPU cores on Linux systems, add the `threads=0` argument. Alternatively, specify a custom number of threads:

```py
distances = simsimd.cdist(matrix1, matrix2, metric="cosine", threads=0)
```

### Hardware Backend Capabilities

To view a list of hardware backends that SimSIMD supports:

```py
print(simsimd.get_capabilities())
```

### Using Python API with USearch

Want to use it in Python with [USearch](https://github.com/unum-cloud/usearch)?
You can wrap the raw C function pointers SimSIMD backends into a `CompiledMetric`, and pass it to USearch, similar to how it handles Numba's JIT-compiled code.

```py
from usearch.index import Index, CompiledMetric, MetricKind, MetricSignature
from simsimd import pointer_to_sqeuclidean, pointer_to_cosine, pointer_to_inner

metric = CompiledMetric(
    pointer=pointer_to_cosine("f16"),
    kind=MetricKind.Cos,
    signature=MetricSignature.ArrayArraySize,
)

index = Index(256, metric=metric)
```

## Using SimSIMD in JavaScript

After you add `simsimd` as a dependency and `npm install`, you will be able to call SimSIMD function on various `TypedArray` variants:

```js
const { sqeuclidean, cosine, inner, hamming, jaccard } = require('simsimd');

const vectorA = new Float32Array([1.0, 2.0, 3.0]);
const vectorB = new Float32Array([4.0, 5.0, 6.0]);

const distance = sqeuclidean(vectorA, vectorB);
console.log('Squared Euclidean Distance:', distance);
```

## Using SimSIMD in C

If you're aiming to utilize the `_Float16` functionality with SimSIMD, ensure your development environment is compatible with C 11. For other functionalities of SimSIMD, C 99 compatibility will suffice.

For integration within a CMake-based project, add the following segment to your `CMakeLists.txt`:

```cmake
FetchContent_Declare(
    simsimd
    GIT_REPOSITORY https://github.com/ashvardanian/simsimd.git
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(simsimd)
include_directories(${simsimd_SOURCE_DIR}/include)
```

Stay updated with the latest advancements by always using the most recent compiler available for your platform. This ensures that you benefit from the newest intrinsics.

Should you wish to integrate SimSIMD within USearch, simply compile USearch with the flag `USEARCH_USE_SIMSIMD=1`. Notably, this is the default setting on the majority of platforms.

## Benchmarking and Contributing

__To rerun experiments__ utilize the following command:

```sh
cmake -DCMAKE_BUILD_TYPE=Release -DSIMSIMD_BUILD_BENCHMARKS=1 -B ./build_release
cmake --build build_release --config Release
./build_release/simsimd_bench
./build_release/simsimd_bench --benchmark_filter=js
```

__To test and benchmark with Python bindings__:

```sh
pip install -e .
pytest python/test.py -s -x 
python python/bench.py --n 1000 --ndim 1000000 # batch size and dimensions
```

__To test and benchmark JavaScript bindings__:

```sh
npm install --dev
npm test
npm run bench
```

__To test and benchmark GoLang bindings__:

```sh
cd golang
go test # To test
go test -run=^$ -bench=. -benchmem # To benchmark
```

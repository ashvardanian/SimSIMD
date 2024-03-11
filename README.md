# SimSIMD 📏

<div>
<a href="https://pepy.tech/project/simsimd">
    <img alt="PyPI" src="https://static.pepy.tech/personalized-badge/simsimd?period=total&units=abbreviation&left_color=black&right_color=blue&left_text=SimSIMD%20Python%20installs">
</a>
<a href="https://www.npmjs.com/package/simsimd">
    <img alt="npm" src="https://img.shields.io/npm/dy/simsimd?label=npm%20dowloads">
</a>
<a href="https://crates.io/crates/simsimd">
    <img alt="rust" src="https://img.shields.io/crates/d/simsimd?logo=rust" />
</a>
<img alt="GitHub code size in bytes" src="https://img.shields.io/github/languages/code-size/ashvardanian/simsimd">
<a href="https://github.com/ashvardanian/SimSIMD/actions/workflows/release.yml">
    <img alt="GitHub Actions Ubuntu" src="https://img.shields.io/github/actions/workflow/status/ashvardanian/SimSIMD/release.yml?branch=main&label=Ubuntu&logo=github&color=blue">
</a>
<a href="https://github.com/ashvardanian/SimSIMD/actions/workflows/release.yml">
    <img alt="GitHub Actions Windows" src="https://img.shields.io/github/actions/workflow/status/ashvardanian/SimSIMD/release.yml?branch=main&label=Windows&logo=windows&color=blue">
</a>
<a href="https://github.com/ashvardanian/SimSIMD/actions/workflows/release.yml">
    <img alt="GitHub Actions MacOS" src="https://img.shields.io/github/actions/workflow/status/ashvardanian/SimSIMD/release.yml?branch=main&label=MacOS&logo=apple&color=blue">
</a>
<a href="https://github.com/ashvardanian/SimSIMD/actions/workflows/release.yml">
    <img alt="GitHub Actions CentOS Linux" src="https://img.shields.io/github/actions/workflow/status/ashvardanian/SimSIMD/release.yml?branch=main&label=CentOS&logo=centos&color=blue">
</a>

</div>


## Hardware-Accelerated Similarity Metrics and Distance Functions

- Zero-dependency [header-only C 99](#using-simsimd-in-c) library.
- Handles `f64` double-, `f32` single-, and `f16` half-precision, `i8` integral, and binary vectors.
- Targets ARM NEON, SVE, x86 AVX2, AVX-512 (VNNI, FP16) hardware backends.
- Bindings for [Python](#using-simsimd-in-python), [Rust](#using-simsimd-in-rust) and [JavaScript](#using-simsimd-in-javascript).
- __Up to 200x faster__ than [`scipy.spatial.distance`][scipy] and [`numpy.inner`][numpy].
- [Supports more Python versions][compatibility] than SciPy and NumPy.
- Zero-copy compatible with NumPy, PyTorch, TensorFlow, and other tensors.
- Used in [USearch](https://github.com/unum-cloud/usearch) and several DBMS products.

__Implemented distance functions__ include:

- Euclidean (L2), Inner Distance, and Cosine (Angular) spatial distances.
- Hamming (~ Manhattan) and Jaccard (~ Tanimoto) binary distances.
- Kullback-Leibler and Jensen–Shannon divergences for probability distributions.

[scipy]: https://docs.scipy.org/doc/scipy/reference/spatial.distance.html#module-scipy.spatial.distance
[numpy]: https://numpy.org/doc/stable/reference/generated/numpy.inner.html
[compatibility]: https://pypi.org/project/simsimd/#files

__Technical Insights__ and related articles:

- [Uses Horner's method for polynomial approximations, beating GCC 12 by 119x](https://ashvardanian.com/posts/gcc-12-vs-avx512fp16/).
- [Uses Arm SVE and x86 AVX-512's masked loads to eliminate tail `for`-loops](https://ashvardanian.com/posts/simsimd-faster-scipy/#tails-of-the-past-the-significance-of-masked-loads).
- [Uses AVX-512 FP16 for half-precision operations, that few compilers vectorize](https://ashvardanian.com/posts/simsimd-faster-scipy/#the-challenge-of-f16).
- [Substitutes LibC's `sqrt` calls with bithacks using Jan Kadlec's constant](https://ashvardanian.com/posts/simsimd-faster-scipy/#bonus-section-bypassing-sqrt-and-libc-dependencies).
- [For Python avoids slow PyBind11, SWIG, and even `PyArg_ParseTuple` for speed](https://ashvardanian.com/posts/pybind11-cpython-tutorial/).
- [For JavaScript uses typed arrays and NAPI for zero-copy calls](https://ashvardanian.com/posts/javascript-ai-vector-search/).

## Benchmarks

### Apple M2 Pro

Given 1000 embeddings from OpenAI Ada API with 1536 dimensions, running on the Apple M2 Pro Arm CPU with NEON support, here's how SimSIMD performs against conventional methods:

| Kind           | `f32` improvement | `f16` improvement | `i8` improvement | Conventional method                    | SimSIMD         |
| :------------- | ----------------: | ----------------: | ---------------: | :------------------------------------- | :-------------- |
| Cosine         |          __32 x__ |          __79 x__ |        __133 x__ | `scipy.spatial.distance.cosine`        | `cosine`        |
| Euclidean ²    |           __5 x__ |          __26 x__ |         __17 x__ | `scipy.spatial.distance.sqeuclidean`   | `sqeuclidean`   |
| Inner Distance |           __2 x__ |           __9 x__ |         __18 x__ | `numpy.inner`                          | `inner`         |
| Jensen Shannon |          __31 x__ |          __53 x__ |                  | `scipy.spatial.distance.jensenshannon` | `jensenshannon` |

### Intel Sapphire Rapids

On the Intel Sapphire Rapids platform, SimSIMD was benchmarked against auto-vectorized code using GCC 12. GCC handles single-precision `float` but might not be the best choice for `int8` and `_Float16` arrays, which has been part of the C language since 2011.

| Kind           | GCC 12 `f32` | GCC 12 `f16` | SimSIMD `f16` | `f16` improvement |
| :------------- | -----------: | -----------: | ------------: | ----------------: |
| Cosine         |     3.28 M/s | _336.29 k/s_ |    _6.88 M/s_ |          __20 x__ |
| Euclidean ²    |     4.62 M/s | _147.25 k/s_ |    _5.32 M/s_ |          __36 x__ |
| Inner Distance |     3.81 M/s | _192.02 k/s_ |    _5.99 M/s_ |          __31 x__ |
| Jensen Shannon |     1.18 M/s |  _18.13 k/s_ |    _2.14 M/s_ |         __118 x__ |

__Broader Benchmarking Results__:

- [Apple M2 Pro](https://ashvardanian.com/posts/simsimd-faster-scipy/#appendix-1-performance-on-apple-m2-pro).
- [4th Gen Intel Xeon Platinum](https://ashvardanian.com/posts/simsimd-faster-scipy/#appendix-2-performance-on-4th-gen-intel-xeon-platinum-8480).
- [AWS Graviton 3](https://ashvardanian.com/posts/simsimd-faster-scipy/#appendix-3-performance-on-aws-graviton-3).

## Using SimSIMD in Python

The package is intended to replace the usage of `numpy.inner`, `numpy.dot`, and `scipy.spatial.distance`.
Aside from drastic performance improvements, SimSIMD significantly improves accuracy in mixed precision setups.
NumPy and SciPy, processing `i8` or `f16` vectors, will use the same types for accumulators, while SimSIMD can combine `i8` enumeration, `i16` multiplication, and `i32` accumulation to entirely avoid overflows.
The same applies to processing `f16` values with `f32` precision.

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
Unlike SciPy, SimSIMD allows explicitly stating the precision of the input vectors, which is especially useful for mixed-precision setups.

```py
dist = simsimd.cosine(vec1, vec2, "i8")
dist = simsimd.cosine(vec1, vec2, "f16")
dist = simsimd.cosine(vec1, vec2, "f32")
dist = simsimd.cosine(vec1, vec2, "f64")
```

### Distance Between 2 Batches

```py
batch1 = np.random.randn(100, 1536).astype(np.float32)
batch2 = np.random.randn(100, 1536).astype(np.float32)
dist = simsimd.cosine(batch1, batch2)
```

If either batch has more than one vector, the other batch must have one or the same number of vectors.
If it contains just one, the value is broadcasted.

### All Pairwise Distances

For calculating distances between all possible pairs of rows across two matrices (akin to [`scipy.spatial.distance.cdist`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html)).
The resulting object will have a type `DistancesTensor`, zero-copy compatible with NumPy and other libraries.

```py
import numpy as np
from simsimd import cdist, DistancesTensor

matrix1 = np.random.randn(1000, 1536).astype(np.float32)
matrix2 = np.random.randn(10, 1536).astype(np.float32)
distances: DistancesTensor = simsimd.cdist(matrix1, matrix2, metric="cosine") # zero-copy
distances_array: np.ndarray = np.array(distances, copy=True) # now managed by NumPy
```

### Multithreading

By default, computations use a single CPU core.
To optimize and utilize all CPU cores on Linux systems, add the `threads=0` argument.
Alternatively, specify a custom number of threads:

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
You can wrap the raw C function pointers SimSIMD backends into a `CompiledMetric` and pass it to USearch, similar to how it handles Numba's JIT-compiled code.

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

## Using SimSIMD in Rust

To install, add the following to your `Cargo.toml`:

```toml
[dependencies]
simsimd = "..."
```

Before using the SimSIMD library, ensure you have imported the necessary traits and types into your Rust source file.
The library provides several traits for different distance/similarity kinds - `SpatialSimilarity`, `BinarySimilarity`, and `ProbabilitySimilarity`.

```rust
use simsimd::SpatialSimilarity;

fn main() {
    let vector_a: Vec<f32> = vec![1.0, 2.0, 3.0];
    let vector_b: Vec<f32> = vec![4.0, 5.0, 6.0];

    // Compute the cosine similarity between vector_a and vector_b
    let cosine_similarity = f32::cosine(&vector_a, &vector_b)
        .expect("Vectors must be of the same length");

    println!("Cosine Similarity: {}", cosine_similarity);

    // Compute the squared Euclidean distance between vector_a and vector_b
    let sq_euclidean_distance = f32::sqeuclidean(&vector_a, &vector_b)
        .expect("Vectors must be of the same length");

    println!("Squared Euclidean Distance: {}", sq_euclidean_distance);
}
```

Similarly, one can compute bit-level distance functions between slices of unsigned integers:

```rust
use simsimd::BinarySimilarity;

fn main() {
    let vector_a = &[0b11110000, 0b00001111, 0b10101010];
    let vector_b = &[0b11110000, 0b00001111, 0b01010101];

    // Compute the Hamming distance between vector_a and vector_b
    let hamming_distance = u8::hamming(&vector_a, &vector_b)
        .expect("Vectors must be of the same length");

    println!("Hamming Distance: {}", hamming_distance);

    // Compute the Jaccard distance between vector_a and vector_b
    let jaccard_distance = u8::jaccard(&vector_a, &vector_b)
        .expect("Vectors must be of the same length");

    println!("Jaccard Distance: {}", jaccard_distance);
}
```

Rust has no native support for half-precision floating-point numbers, but SimSIMD provides a `f16` type for this purpose.
It doesn't have any functionality and is a `transparent` wrapper around `u16`, so it can be used with `half`, or any other half-precision library.

```rust
use simsimd::SpatialSimilarity;
use simsimd::f16 as SimF16;
use half::f16 as HalfF16;

fn main() {
    let vector_a: Vec<HalfF16> = ...
    let vector_b: Vec<HalfF16> = ...

    let buffer_a: &[SimF16] = unsafe { std::slice::from_raw_parts(a_half.as_ptr() as *const SimF16, a_half.len()) };
    let buffer_b: &[SimF16] = unsafe { std::slice::from_raw_parts(b_half.as_ptr() as *const SimF16, b_half.len()) };

    // Compute the cosine similarity between vector_a and vector_b
    let cosine_similarity = SimF16::cosine(&vector_a, &vector_b)
        .expect("Vectors must be of the same length");

    println!("Cosine Similarity: {}", cosine_similarity);
}
```

## Using SimSIMD in JavaScript

To install, choose one of the following options depending on your environment:

- `npm install --save simsimd`
- `yarn add simsimd`
- `pnpm add simsimd`
- `bun install simsimd`

The package is distributed with prebuilt binaries for Node.js v10 and above for Linux (x86_64, arm64), macOS (x86_64, arm64), and Windows (i386,x86_64).

If your platform is not supported, you can build the package from source via `npm run build`. This will automatically happen unless you install the package with `--ignore-scripts` flag or use Bun.

After you install it, you will be able to call the SimSIMD functions on various `TypedArray` variants:

```js
const { sqeuclidean, cosine, inner, hamming, jaccard } = require('simsimd');

const vectorA = new Float32Array([1.0, 2.0, 3.0]);
const vectorB = new Float32Array([4.0, 5.0, 6.0]);

const distance = sqeuclidean(vectorA, vectorB);
console.log('Squared Euclidean Distance:', distance);
```

## Using SimSIMD in C

For integration within a CMake-based project, add the following segment to your `CMakeLists.txt`:

```cmake
FetchContent_Declare(
    simsimd
    GIT_REPOSITORY https://github.com/ashvardanian/simsimd.git
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(simsimd)
```

If you're aiming to utilize the `_Float16` functionality with SimSIMD, ensure your development environment is compatible with C 11.
For other functionalities of SimSIMD, C 99 compatibility will suffice.
A minimal usage example would be:

```c
#include <simsimd/simsimd.h>

int main() {
    simsimd_f32_t vector_a[1536];
    simsimd_f32_t vector_b[1536];
    simsimd_f32_t distance = simsimd_avx512_f32_cos(vector_a, vector_b, 1536);
    return 0;
}
```

All of the functions names follow the same pattern: `simsimd_{backend}_{type}_{metric}`.

- The backend can be `avx512`, `avx2`, `neon`, or `sve`.
- The type can be `f64`, `f32`, `f16`, `i8`, or `b8`.
- The metric can be `cos`, `ip`, `l2sq`, `hamming`, `jaccard`, `kl`, or `js`.

In case you want to avoid hard-coding the backend, you can use the `simsimd_metric_punned_t` to pun the function pointer, and `simsimd_capabilities` function to get the available backends at runtime.

![MathKong banner](https://github.com/ashvardanian/ashvardanian/blob/master/repositories/MathKong.jpg?raw=true)

Computing dot-products, similarity measures, and distances between low- and high-dimensional vectors is ubiquitous in Machine Learning, Scientific Computing, Geospatial Analysis, and Information Retrieval.
These algorithms generally have linear complexity in time, constant or linear complexity in space, and are data-parallel.
In other words, they are easily parallelizable and vectorizable and often available in packages like BLAS (level 1) and LAPACK, as well as higher-level `numpy` and `scipy` Python libraries.
Ironically, even with decades of evolution in compilers and numerical computing, [most libraries can be 3x - 1'000x slower than hardware potential][benchmarks] even on the most popular hardware, like 64-bit x86 and Arm CPUs.
Moreover, most lack mixed-precision support, which is crucial for modern AI!
The rare few that support minimal mixed precision, run only on one platform, and are vendor-locked, by companies like Intel and Nvidia.
MathKong provides an alternative.
1️⃣ MathKong functions are practically as fast as `memcpy`.
2️⃣ Unlike BLAS, most kernels are designed for mixed-precision and bit-level operations.
3️⃣ MathKong often [ships more binaries than NumPy][compatibility] and has more backends than most BLAS implementations, and more high-level interfaces than most libraries.

[benchmarks]: https://ashvardanian.com/posts/mathkong-faster-scipy
[compatibility]: https://pypi.org/project/mathkong/#files

<div>
<a href="https://pepy.tech/project/mathkong">
    <img alt="PyPI" src="https://static.pepy.tech/personalized-badge/mathkong?period=total&units=abbreviation&left_color=black&right_color=blue&left_text=MathKong%20Python%20installs" />
</a>
<a href="https://www.npmjs.com/package/mathkong">
    <img alt="npm" src="https://img.shields.io/npm/dy/mathkong?label=JavaScript%20NPM%20installs" />
</a>
<a href="https://crates.io/crates/mathkong">
    <img alt="rust" src="https://img.shields.io/crates/d/mathkong?label=Rust%20Crate%20installs" />
</a>
<img alt="GitHub code size in bytes" src="https://img.shields.io/github/languages/code-size/ashvardanian/mathkong">
<a href="https://github.com/ashvardanian/MathKong/actions/workflows/release.yml">
    <img alt="GitHub Actions Ubuntu" src="https://img.shields.io/github/actions/workflow/status/ashvardanian/MathKong/release.yml?branch=main&label=Ubuntu&logo=github&color=blue">
</a>
<a href="https://github.com/ashvardanian/MathKong/actions/workflows/release.yml">
    <img alt="GitHub Actions Windows" src="https://img.shields.io/github/actions/workflow/status/ashvardanian/MathKong/release.yml?branch=main&label=Windows&logo=windows&color=blue">
</a>
<a href="https://github.com/ashvardanian/MathKong/actions/workflows/release.yml">
    <img alt="GitHub Actions macOS" src="https://img.shields.io/github/actions/workflow/status/ashvardanian/MathKong/release.yml?branch=main&label=macOS&logo=apple&color=blue">
</a>
<a href="https://github.com/ashvardanian/MathKong/actions/workflows/release.yml">
    <img alt="GitHub Actions CentOS Linux" src="https://img.shields.io/github/actions/workflow/status/ashvardanian/MathKong/release.yml?branch=main&label=CentOS&logo=centos&color=blue">
</a>

</div>

## Features

__MathKong__ (Arabic: "سيمسيم دي") is a mixed-precision math library of __over 450 SIMD-optimized kernels__ extensively used in AI, Search, and DBMS workloads.
Named after the iconic ["Open Sesame"](https://en.wikipedia.org/wiki/Open_sesame) command that opened doors to treasure in _Ali Baba and the Forty Thieves_, SimSimd can help you 10x the cost-efficiency of your computational pipelines.
Implemented distance functions include:

- Euclidean (L2) and Cosine (Angular) spatial distances for Vector Search. _[docs][docs-spatial]_
- Dot-Products for real & complex vectors for DSP & Quantum computing. _[docs][docs-dot]_
- Hamming (~ Manhattan) and Jaccard (~ Tanimoto) bit-level distances. _[docs][docs-binary]_
- Set Intersections for Sparse Vectors and Text Analysis. _[docs][docs-sparse]_
- Mahalanobis distance and Quadratic forms for Scientific Computing. _[docs][docs-curved]_
- Kullback-Leibler and Jensen–Shannon divergences for probability distributions. _[docs][docs-probability]_
- Fused-Multiply-Add (FMA) and Weighted Sums to replace BLAS level 1 functions. _[docs][docs-elementwise]_
- For Levenshtein, Needleman–Wunsch, and Smith-Waterman, check [StringZilla][stringzilla].
- 🔜 Haversine and Vincenty's formulae for Geospatial Analysis.

[docs-spatial]: #cosine-similarity-reciprocal-square-root-and-newton-raphson-iteration
[docs-curved]: #curved-spaces-mahalanobis-distance-and-bilinear-quadratic-forms
[docs-sparse]: #set-intersection-galloping-and-binary-search
[docs-binary]: https://github.com/ashvardanian/MathKong/pull/138
[docs-dot]: #complex-dot-products-conjugate-dot-products-and-complex-numbers
[docs-probability]: #logarithms-in-kullback-leibler--jensenshannon-divergences
[docs-elementwise]: #mixed-precision-in-fused-multiply-add-and-weighted-sums
[scipy]: https://docs.scipy.org/doc/scipy/reference/spatial.distance.html#module-scipy.spatial.distance
[numpy]: https://numpy.org/doc/stable/reference/generated/numpy.inner.html
[stringzilla]: https://github.com/ashvardanian/stringzilla

Moreover, MathKong...

- handles `float64`, `float32`, `float16`, and `bfloat16` real & complex vectors.
- handles `int8` integral, `int4` sub-byte, and `b8` binary vectors.
- handles sparse `uint32` and `uint16` sets, and weighted sparse vectors.
- is a zero-dependency [header-only C 99](#using-mathkong-in-c) library.
- has [Python](#using-mathkong-in-python), [Rust](#using-mathkong-in-rust), [JS](#using-mathkong-in-javascript), and [Swift](#using-mathkong-in-swift) bindings.
- has Arm backends for NEON, Scalable Vector Extensions (SVE), and SVE2.
- has x86 backends for Haswell, Skylake, Ice Lake, Genoa, and Sapphire Rapids.
- with both compile-time and runtime CPU feature detection easily integrates anywhere!

Due to the high-level of fragmentation of SIMD support in different x86 CPUs, MathKong generally uses the names of select Intel CPU generations for its backends.
They, however, also work on AMD CPUs.
Intel Haswell is compatible with AMD Zen 1/2/3, while AMD Genoa Zen 4 covers AVX-512 instructions added to Intel Skylake and Ice Lake.
You can learn more about the technical implementation details in the following blog-posts:

- [Uses Horner's method for polynomial approximations, beating GCC 12 by 119x](https://ashvardanian.com/posts/gcc-12-vs-avx512fp16/).
- [Uses Arm SVE and x86 AVX-512's masked loads to eliminate tail `for`-loops](https://ashvardanian.com/posts/mathkong-faster-scipy/#tails-of-the-past-the-significance-of-masked-loads).
- [Substitutes libc's `sqrt` with Newton Raphson iterations](https://github.com/ashvardanian/MathKong/releases/tag/v5.4.0).
- [Uses Galloping and SVE2 histograms to intersect sparse vectors](https://ashvardanian.com/posts/simd-set-intersections-sve2-avx512/).
- For Python: [avoids slow PyBind11, SWIG, & `PyArg_ParseTuple`](https://ashvardanian.com/posts/pybind11-cpython-tutorial/) [using faster calling convention](https://ashvardanian.com/posts/discount-on-keyword-arguments-in-python/).
- For JavaScript: [uses typed arrays and NAPI for zero-copy calls](https://ashvardanian.com/posts/javascript-ai-vector-search/).

## Benchmarks

<table style="width: 100%; text-align: center; table-layout: fixed;">
  <colgroup>
    <col style="width: 33%;">
    <col style="width: 33%;">
    <col style="width: 33%;">
  </colgroup>
  <tr>
    <th align="center">NumPy</th>
    <th align="center">C 99</th>
    <th align="center">MathKong</th>
  </tr>
  <!-- Cosine distances with different precision levels -->
  <tr>
    <td colspan="4" align="center">cosine distances between 1536d vectors in <code>int8</code></td>
  </tr>
  <tr>
    <td align="center"> <!-- scipy.spatial.distance.cosine -->
      🚧 overflows<br/>
    </td>
    <td align="center"> <!-- serial -->
      <span style="color:#ABABAB;">x86:</span> <b>10,548,600</b> ops/s<br/>
      <span style="color:#ABABAB;">arm:</span> <b>11,379,300</b> ops/s
    </td>
    <td align="center"> <!-- mathkong -->
      <span style="color:#ABABAB;">x86:</span> <b>16,151,800</b> ops/s<br/>
      <span style="color:#ABABAB;">arm:</span> <b>13,524,000</b> ops/s
    </td>
  </tr>
  <tr>
    <td colspan="4" align="center">cosine distances between 1536d vectors in <code>bfloat16</code></td>
  </tr>
  <tr>
    <td align="center"> <!-- scipy.spatial.distance.cosine -->
      🚧 not supported<br/>
    </td>
    <td align="center"> <!-- serial -->
      <span style="color:#ABABAB;">x86:</span> <b>119,835</b> ops/s<br/>
      <span style="color:#ABABAB;">arm:</span> <b>403,909</b> ops/s
    </td>
    <td align="center"> <!-- mathkong -->
      <span style="color:#ABABAB;">x86:</span> <b>9,738,540</b> ops/s<br/>
      <span style="color:#ABABAB;">arm:</span> <b>4,881,900</b> ops/s
    </td>
  </tr>
  <tr>
    <td colspan="4" align="center">cosine distances between 1536d vectors in <code>float16</code></td>
  </tr>
  <tr>
    <td align="center"> <!-- scipy.spatial.distance.cosine -->
      <span style="color:#ABABAB;">x86:</span> <b>40,481</b> ops/s<br/>
      <span style="color:#ABABAB;">arm:</span> <b>21,451</b> ops/s
    </td>
    <td align="center"> <!-- serial -->
      <span style="color:#ABABAB;">x86:</span> <b>501,310</b> ops/s<br/>
      <span style="color:#ABABAB;">arm:</span> <b>871,963</b> ops/s
    </td>
    <td align="center"> <!-- mathkong -->
      <span style="color:#ABABAB;">x86:</span> <b>7,627,600</b> ops/s<br/>
      <span style="color:#ABABAB;">arm:</span> <b>3,316,810</b> ops/s
    </td>
  </tr>
  <tr>
    <td colspan="4" align="center">cosine distances between 1536d vectors in <code>float32</code></td>
  </tr>
  <tr>
    <td align="center"> <!-- scipy.spatial.distance.cosine -->
      <span style="color:#ABABAB;">x86:</span> <b>253,902</b> ops/s<br/>
      <span style="color:#ABABAB;">arm:</span> <b>46,394</b> ops/s
    </td>
    <td align="center"> <!-- serial -->
      <span style="color:#ABABAB;">x86:</span> <b>882,484</b> ops/s<br/>
      <span style="color:#ABABAB;">arm:</span> <b>399,661</b> ops/s
    </td>
    <td align="center"> <!-- mathkong -->
      <span style="color:#ABABAB;">x86:</span> <b>8,202,910</b> ops/s<br/>
      <span style="color:#ABABAB;">arm:</span> <b>3,400,620</b> ops/s
    </td>
  </tr>
  <tr>
    <td colspan="4" align="center">cosine distances between 1536d vectors in <code>float64</code></td>
  </tr>
  <tr>
    <td align="center"> <!-- scipy.spatial.distance.cosine -->
      <span style="color:#ABABAB;">x86:</span> <b>212,421</b> ops/s<br/>
      <span style="color:#ABABAB;">arm:</span> <b>52,904</b> ops/s
    </td>
    <td align="center"> <!-- serial -->
      <span style="color:#ABABAB;">x86:</span> <b>839,301</b> ops/s<br/>
      <span style="color:#ABABAB;">arm:</span> <b>837,126</b> ops/s
    </td>
    <td align="center"> <!-- mathkong -->
      <span style="color:#ABABAB;">x86:</span> <b>1,538,530</b> ops/s<br/>
      <span style="color:#ABABAB;">arm:</span> <b>1,678,920</b> ops/s
    </td>
  </tr>

  <!-- Euclidean distance with different precision level -->
  <tr>
    <td colspan="4" align="center">euclidean distance between 1536d vectors in <code>int8</code></td>
  </tr>
  <tr>
    <td align="center"> <!-- scipy.spatial.distance.sqeuclidean -->
      <span style="color:#ABABAB;">x86:</span> <b>252,113</b> ops/s<br/>
      <span style="color:#ABABAB;">arm:</span> <b>177,443</b> ops/s
    </td>
    <td align="center"> <!-- serial -->
      <span style="color:#ABABAB;">x86:</span> <b>6,690,110</b> ops/s<br/>
      <span style="color:#ABABAB;">arm:</span> <b>4,114,160</b> ops/s
    </td>
    <td align="center"> <!-- mathkong -->
      <span style="color:#ABABAB;">x86:</span> <b>18,989,000</b> ops/s<br/>
      <span style="color:#ABABAB;">arm:</span> <b>18,878,200</b> ops/s
    </td>
  </tr>
  <tr>
    <td colspan="4" align="center">euclidean distance between 1536d vectors in <code>bfloat16</code></td>
  </tr>
  <tr>
    <td align="center"> <!-- scipy.spatial.distance.sqeuclidean -->
      🚧 not supported<br/>
    </td>
    <td align="center"> <!-- serial -->
      <span style="color:#ABABAB;">x86:</span> <b>119,842</b> ops/s<br/>
      <span style="color:#ABABAB;">arm:</span> <b>1,049,230</b> ops/s
    </td>
    <td align="center"> <!-- mathkong -->
      <span style="color:#ABABAB;">x86:</span> <b>9,727,210</b> ops/s<br/>
      <span style="color:#ABABAB;">arm:</span> <b>4,233,420</b> ops/s
    </td>
  </tr>
  <tr>
    <td colspan="4" align="center">euclidean distance between 1536d vectors in <code>float16</code></td>
  </tr>
  <tr>
    <td align="center"> <!-- scipy.spatial.distance.sqeuclidean -->
      <span style="color:#ABABAB;">x86:</span> <b>54,621</b> ops/s<br/>
      <span style="color:#ABABAB;">arm:</span> <b>71,793</b> ops/s
    </td>
    <td align="center"> <!-- serial -->
      <span style="color:#ABABAB;">x86:</span> <b>196,413</b> ops/s<br/>
      <span style="color:#ABABAB;">arm:</span> <b>911,370</b> ops/s
    </td>
    <td align="center"> <!-- mathkong -->
      <span style="color:#ABABAB;">x86:</span> <b>19,466,800</b> ops/s<br/>
      <span style="color:#ABABAB;">arm:</span> <b>3,522,760</b> ops/s
    </td>
  </tr>
  <tr>
    <td colspan="4" align="center">euclidean distance between 1536d vectors in <code>float32</code></td>
  </tr>
  <tr>
    <td align="center"> <!-- scipy.spatial.distance.sqeuclidean -->
      <span style="color:#ABABAB;">x86:</span> <b>424,944</b> ops/s<br/>
      <span style="color:#ABABAB;">arm:</span> <b>292,629</b> ops/s
    </td>
    <td align="center"> <!-- serial -->
      <span style="color:#ABABAB;">x86:</span> <b>1,295,210</b> ops/s<br/>
      <span style="color:#ABABAB;">arm:</span> <b>1,055,940</b> ops/s
    </td>
    <td align="center"> <!-- mathkong -->
      <span style="color:#ABABAB;">x86:</span> <b>8,924,100</b> ops/s<br/>
      <span style="color:#ABABAB;">arm:</span> <b>3,602,650</b> ops/s
    </td>
  </tr>
  <tr>
    <td colspan="4" align="center">euclidean distance between 1536d vectors in <code>float64</code></td>
  </tr>
  <tr>
    <td align="center"> <!-- scipy.spatial.distance.sqeuclidean -->
      <span style="color:#ABABAB;">x86:</span> <b>334,929</b> ops/s<br/>
      <span style="color:#ABABAB;">arm:</span> <b>237,505</b> ops/s
    </td>
    <td align="center"> <!-- serial -->
      <span style="color:#ABABAB;">x86:</span> <b>1,215,190</b> ops/s<br/>
      <span style="color:#ABABAB;">arm:</span> <b>905,782</b> ops/s
    </td>
    <td align="center"> <!-- mathkong -->
      <span style="color:#ABABAB;">x86:</span> <b>1,701,740</b> ops/s<br/>
      <span style="color:#ABABAB;">arm:</span> <b>1,735,840</b> ops/s
    </td>
  </tr>
  <!-- Bilinear forms -->
  <!-- Sparse set intersections -->
</table>

> For benchmarks we mostly use 1536-dimensional vectors, like the embeddings produced by the OpenAI Ada API.
> The code was compiled with GCC 12, using glibc v2.35.
> The benchmarks performed on Arm-based Graviton3 AWS `c7g` instances and `r7iz` Intel Sapphire Rapids.
> Most modern Arm-based 64-bit CPUs will have similar relative speedups.
> Variance withing x86 CPUs will be larger.

Similar speedups are often observed even when compared to BLAS and LAPACK libraries underlying most numerical computing libraries, including NumPy and SciPy in Python.
Broader benchmarking results:

- [Apple M2 Pro](https://ashvardanian.com/posts/mathkong-faster-scipy/#appendix-1-performance-on-apple-m2-pro).
- [Intel Sapphire Rapids](https://ashvardanian.com/posts/mathkong-faster-scipy/#appendix-2-performance-on-4th-gen-intel-xeon-platinum-8480).
- [AWS Graviton 3](https://ashvardanian.com/posts/mathkong-faster-scipy/#appendix-3-performance-on-aws-graviton-3).

## Using MathKong in Python

The package is intended to replace the usage of `numpy.inner`, `numpy.dot`, and `scipy.spatial.distance`.
Aside from drastic performance improvements, MathKong significantly improves accuracy in mixed precision setups.
NumPy and SciPy, processing `int8`, `uint8` or `float16` vectors, will use the same types for accumulators, while MathKong can combine `int8` enumeration, `int16` multiplication, and `int32` accumulation to avoid overflows entirely.
The same applies to processing `float16` and `bfloat16` values with `float32` precision.

### Installation

Use the following snippet to install MathKong and list available hardware acceleration options available on your machine:

```sh
pip install mathkong
python -c "import mathkong; print(mathkong.get_capabilities())"   # for hardware introspection
python -c "import mathkong; help(mathkong)"                       # for documentation
```

With precompiled binaries, MathKong ships `.pyi` interface files for type hinting and static analysis.
You can check all the available functions in [`python/annotations/__init__.pyi`](https://github.com/ashvardanian/MathKong/blob/main/python/annotations/__init__.pyi).

### One-to-One Distance

```py
import mathkong
import numpy as np

vec1 = np.random.randn(1536).astype(np.float32)
vec2 = np.random.randn(1536).astype(np.float32)
dist = mathkong.angular(vec1, vec2)
```

Supported functions include `cosine`, `inner`, `sqeuclidean`, `hamming`, `jaccard`, `kulbackleibler`, `jensenshannon`, and `intersect`.
Dot products are supported for both real and complex numbers:

```py
vec1 = np.random.randn(768).astype(np.float64) + 1j * np.random.randn(768).astype(np.float64)
vec2 = np.random.randn(768).astype(np.float64) + 1j * np.random.randn(768).astype(np.float64)

dist = mathkong.dot(vec1.astype(np.complex128), vec2.astype(np.complex128))
dist = mathkong.dot(vec1.astype(np.complex64), vec2.astype(np.complex64))
dist = mathkong.vdot(vec1.astype(np.complex64), vec2.astype(np.complex64)) # conjugate, same as `np.vdot`
```

Unlike SciPy, MathKong allows explicitly stating the precision of the input vectors, which is especially useful for mixed-precision setups.
The `dtype` argument can be passed both by name and as a positional argument:

```py
dist = mathkong.angular(vec1, vec2, "int8")
dist = mathkong.angular(vec1, vec2, "float16")
dist = mathkong.angular(vec1, vec2, "float32")
dist = mathkong.angular(vec1, vec2, "float64")
dist = mathkong.hamming(vec1, vec2, "bin8")
```

Binary distance functions are computed at a bit-level.
Meaning a vector of 10x 8-bit integers will be treated as a sequence of 80 individual bits or dimensions.
This differs from NumPy, that can't handle smaller-than-byte types, but you can still avoid the `bin8` argument by reinterpreting the vector as booleans:

```py
vec1 = np.random.randint(2, size=80).astype(np.uint8).packbits().view(np.bool_)
vec2 = np.random.randint(2, size=80).astype(np.uint8).packbits().view(np.bool_)
hamming_distance = mathkong.hamming(vec1, vec2)
jaccard_distance = mathkong.jaccard(vec1, vec2)
```

With other frameworks, like PyTorch, one can get a richer type-system than NumPy, but the lack of good CPython interoperability makes it hard to pass data without copies.
Here is an example of using MathKong with PyTorch to compute the cosine similarity between two `bfloat16` vectors:

```py
import numpy as np
buf1 = np.empty(8, dtype=np.uint16)
buf2 = np.empty(8, dtype=np.uint16)

# View the same memory region with PyTorch and randomize it
import torch
vec1 = torch.asarray(memoryview(buf1), copy=False).view(torch.bfloat16)
vec2 = torch.asarray(memoryview(buf2), copy=False).view(torch.bfloat16)
torch.randn(8, out=vec1)
torch.randn(8, out=vec2)

# Both libs will look into the same memory buffers and report the same results
dist_slow = 1 - torch.nn.functional.cosine_similarity(vec1, vec2, dim=0)
dist_fast = mathkong.angular(buf1, buf2, "bfloat16")
```

It also allows using MathKong for half-precision complex numbers, which NumPy does not support.
For that, view data as continuous even-length `np.float16` vectors and override type-resolution with `complex32` string.

```py
vec1 = np.random.randn(1536).astype(np.float16)
vec2 = np.random.randn(1536).astype(np.float16)
simd.dot(vec1, vec2, "complex32")
simd.vdot(vec1, vec2, "complex32")
```

When dealing with sparse representations and integer sets, you can apply the `intersect` function to two 1-dimensional arrays of `uint16` or `uint32` integers:

```py
from random import randint
import numpy as np
import mathkong as simd

length1, length2 = randint(1, 100), randint(1, 100)
vec1 = np.sort(np.random.randint(0, 1000, length1).astype(np.uint16))
vec2 = np.sort(np.random.randint(0, 1000, length2).astype(np.uint16))

slow_result = len(np.intersect1d(vec1, vec2))
fast_result = simd.intersect(vec1, vec2)
assert slow_result == fast_result
```

### One-to-Many Distances

Every distance function can be used not only for one-to-one but also one-to-many and many-to-many distance calculations.
For one-to-many:

```py
vec1 = np.random.randn(1536).astype(np.float32) # rank 1 tensor
batch1 = np.random.randn(1, 1536).astype(np.float32) # rank 2 tensor
batch2 = np.random.randn(100, 1536).astype(np.float32)

dist_rank1 = mathkong.angular(vec1, batch2)
dist_rank2 = mathkong.angular(batch1, batch2)
```

### Many-to-Many Distances

All distance functions in MathKong can be used to compute many-to-many distances.
For two batches of 100 vectors to compute 100 distances, one would call it like this:

```py
batch1 = np.random.randn(100, 1536).astype(np.float32)
batch2 = np.random.randn(100, 1536).astype(np.float32)
dist = mathkong.angular(batch1, batch2)
```

Input matrices must have identical shapes.
This functionality isn't natively present in NumPy or SciPy, and generally requires creating intermediate arrays, which is inefficient and memory-consuming.

### Many-to-Many All-Pairs Distances

One can use MathKong to compute distances between all possible pairs of rows across two matrices (akin to [`scipy.spatial.distance.cdist`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html)).
The resulting object will have a type `DistancesTensor`, zero-copy compatible with NumPy and other libraries.
For two arrays of 10 and 1,000 entries, the resulting tensor will have 10,000 cells:

```py
import numpy as np
from mathkong import cdist, DistancesTensor

matrix1 = np.random.randn(1000, 1536).astype(np.float32)
matrix2 = np.random.randn(10, 1536).astype(np.float32)
distances: DistancesTensor = mathkong.cdist(matrix1, matrix2, metric="cosine")   # zero-copy, managed by MathKong
distances_array: np.ndarray = np.array(distances, copy=True)                    # now managed by NumPy
```

### Element-wise Kernels

MathKong also provides mixed-precision element-wise kernels, where the input vectors and the output have the same numeric type, but the intermediate accumulators are of a higher precision.

```py
import numpy as np
from mathkong import fma, wsum

# Let's take two FullHD video frames
first_frame = np.random.randn(1920 * 1024).astype(np.uint8)
second_frame = np.random.randn(1920 * 1024).astype(np.uint8)
average_frame = np.empty_like(first_frame)
wsum(first_frame, second_frame, alpha=0.5, beta=0.5, out=average_frame)

# Slow analog with NumPy:
slow_average_frame = (0.5 * first_frame + 0.5 * second_frame).astype(np.uint8)
```

Similarly, the `fma` takes three arguments and computes the fused multiply-add operation.
In applications like Machine Learning you may also benefit from using the "brain-float" format not natively supported by NumPy.
In 3D Graphics, for example, we can use FMA to compute the [Phong shading model](https://en.wikipedia.org/wiki/Phong_shading):

```py
# Assume a FullHD frame with random values for simplicity
light_intensity = np.random.rand(1920 * 1080).astype(np.float16)  # Intensity of light on each pixel
diffuse_component = np.random.rand(1920 * 1080).astype(np.float16)  # Diffuse reflectance on the surface
specular_component = np.random.rand(1920 * 1080).astype(np.float16)  # Specular reflectance for highlights
output_color = np.empty_like(light_intensity)  # Array to store the resulting color intensity

# Define the scaling factors for diffuse and specular contributions
alpha = 0.7  # Weight for the diffuse component
beta = 0.3   # Weight for the specular component

# Formula: color = alpha * light_intensity * diffuse_component + beta * specular_component
fma(light_intensity, diffuse_component, specular_component,
    dtype="float16", # Optional, unless it can't be inferred from the input
    alpha=alpha, beta=beta, out=output_color)

# Slow analog with NumPy for comparison
slow_output_color = (alpha * light_intensity * diffuse_component + beta * specular_component).astype(np.float16)
```

### Multithreading and Memory Usage

By default, computations use a single CPU core.
To override this behavior, use the `threads` argument.
Set it to `0` to use all available CPU cores and let the underlying C library manage the thread pool.
Here is an example of dealing with large sets of binary vectors:

```py
ndim = 1536 # OpenAI Ada embeddings
matrix1 = np.packbits(np.random.randint(2, size=(10_000, ndim)).astype(np.uint8))
matrix2 = np.packbits(np.random.randint(2, size=(1_000, ndim)).astype(np.uint8))

distances = mathkong.cdist(matrix1, matrix2,
    metric="hamming",   # Unlike SciPy, MathKong doesn't divide by the number of dimensions
    out_dtype="uint8",  # so we can use `uint8` instead of `float64` to save memory.
    threads=0,          # Use all CPU cores with OpenMP.
    dtype="bin8",       # Override input argument type to `bin8` eight-bit words.
)
```

Alternatively, when using free-threading Python 3.13t builds, one can combine single-threaded MathKong operations with Python's `concurrent.futures.ThreadPoolExecutor` to parallelize the computations.
By default, the output distances will be stored in double-precision `float64` floating-point numbers.
That behavior may not be space-efficient, especially if you are computing the hamming distance between short binary vectors, that will generally fit into 8x smaller `uint8` or `uint16` types.
To override this behavior, use the `out_dtype` argument, or consider pre-allocating the output array and passing it to the `out` argument.
A more complete example may look like this:

```py
from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor
from mathkong import cosine
import numpy as np

# Generate large dataset
vectors_a = np.random.rand(100_000, 1536).astype(np.float32)
vectors_b = np.random.rand(100_000, 1536).astype(np.float32)
distances = np.zeros((100_000,), dtype=np.float32)

def compute_batch(start_idx, end_idx):
    batch_a = vectors_a[start_idx:end_idx]
    batch_b = vectors_b[start_idx:end_idx]
    cosine(batch_a, batch_b, out=distances[start_idx:end_idx])

# Use all CPU cores with true parallelism (no GIL!)
num_threads = cpu_count()
chunk_size = len(vectors_a) // num_threads

with ThreadPoolExecutor(max_workers=num_threads) as executor:
    futures = []
    for i in range(num_threads):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size if i < num_threads - 1 else len(vectors_a)
        futures.append(executor.submit(compute_batch, start_idx, end_idx))

    # Collect results from all threads
    results = [future.result() for future in futures]
```

### Half-Precision Brain-Float Numbers

The "brain-float-16" is a popular machine learning format.
It's broadly supported in hardware and is very machine-friendly, but software support is still lagging behind.
[Unlike NumPy](https://github.com/numpy/numpy/issues/19808), you can already use `bf16` datatype in MathKong.
Luckily, to downcast `f32` to `bf16` you only have to drop the last 16 bits:

```py
import numpy as np
import mathkong as simd

a = np.random.randn(ndim).astype(np.float32)
b = np.random.randn(ndim).astype(np.float32)

# NumPy doesn't natively support brain-float, so we need a trick!
# Luckily, it's very easy to reduce the representation accuracy
# by simply masking the low 16-bits of our 32-bit single-precision
# numbers. We can also add `0x8000` to round the numbers.
a_f32rounded = ((a.view(np.uint32) + 0x8000) & 0xFFFF0000).view(np.float32)
b_f32rounded = ((b.view(np.uint32) + 0x8000) & 0xFFFF0000).view(np.float32)

# To represent them as brain-floats, we need to drop the second half
a_bf16 = np.right_shift(a_f32rounded.view(np.uint32), 16).astype(np.uint16)
b_bf16 = np.right_shift(b_f32rounded.view(np.uint32), 16).astype(np.uint16)

# Now we can compare the results
expected = np.inner(a_f32rounded, b_f32rounded)
result = simd.inner(a_bf16, b_bf16, "bf16")
```

### Helper Functions

You can turn specific backends on or off depending on the exact environment.
A common case may be avoiding AVX-512 on older AMD CPUs and [Intel Ice Lake](https://travisdowns.github.io/blog/2020/08/19/icl-avx512-freq.html) CPUs to ensure the CPU doesn't change the frequency license and throttle performance.

```py
$ mathkong.get_capabilities()
> {'serial': True, 'neon': False, 'sve': False, 'neon_f16': False, 'sve_f16': False, 'neon_bf16': False, 'sve_bf16': False, 'neon_i8': False, 'sve_i8': False, 'haswell': True, 'skylake': True, 'ice': True, 'genoa': True, 'sapphire': True, 'turin': True}
$ mathkong.disable_capability("sapphire")
$ mathkong.enable_capability("sapphire")
```

### Using Python API with USearch

Want to use it in Python with [USearch](https://github.com/unum-cloud/usearch)?
You can wrap the raw C function pointers MathKong backends into a `CompiledMetric` and pass it to USearch, similar to how it handles Numba's JIT-compiled code.

```py
from usearch.index import Index, CompiledMetric, MetricKind, MetricSignature
from mathkong import pointer_to_sqeuclidean, pointer_to_cosine, pointer_to_inner

metric = CompiledMetric(
    pointer=pointer_to_cosine("f16"),
    kind=MetricKind.Cos,
    signature=MetricSignature.ArrayArraySize,
)

index = Index(256, metric=metric)
```

## Using MathKong in Rust

To install, add the following to your `Cargo.toml`:

```toml
[dependencies]
mathkong = "..."
```

Before using the MathKong library, ensure you have imported the necessary traits and types into your Rust source file.
The library provides several traits for different distance/similarity kinds - `SpatialSimilarity`, `BinarySimilarity`, and `ProbabilitySimilarity`.

### Spatial Similarity: Cosine and Euclidean Distances

```rust
use mathkong::SpatialSimilarity;

fn main() {
    let vector_a: Vec<f32> = vec![1.0, 2.0, 3.0];
    let vector_b: Vec<f32> = vec![4.0, 5.0, 6.0];

    // Compute the cosine distance between vectors
    let cosine_distance = f32::cosine(&vector_a, &vector_b)
        .expect("Vectors must be of the same length");

    println!("Cosine Distance: {}", cosine_distance);

    // Compute the squared Euclidean distance between vectors
    let sq_euclidean_distance = f32::sqeuclidean(&vector_a, &vector_b)
        .expect("Vectors must be of the same length");

    println!("Squared Euclidean Distance: {}", sq_euclidean_distance);
}
```

Spatial similarity functions are available for `f64`, `f32`, `f16`, and `i8` types.

### Dot-Products: Inner and Complex Inner Products

```rust
use mathkong::SpatialSimilarity;
use mathkong::ComplexProducts;

fn main() {
    // Complex vectors have interleaved real & imaginary components
    let vector_a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let vector_b: Vec<f32> = vec![5.0, 6.0, 7.0, 8.0];

    // Compute the inner product between vectors
    let inner_product = SpatialSimilarity::dot(&vector_a, &vector_b)
        .expect("Vectors must be of the same length");

    println!("Inner Product: {}", inner_product);

    // Compute the complex inner product between vectors
    let complex_inner_product = ComplexProducts::dot(&vector_a, &vector_b)
        .expect("Vectors must be of the same length");

    let complex_conjugate_inner_product = ComplexProducts::vdot(&vector_a, &vector_b)
        .expect("Vectors must be of the same length");

    println!("Complex Inner Product: {:?}", complex_inner_product); // -18, 69
    println!("Complex C. Inner Product: {:?}", complex_conjugate_inner_product); // 70, -8
}
```

Complex inner products are available for `f64`, `f32`, and `f16` types.

### Probability Distributions: Jensen-Shannon and Kullback-Leibler Divergences

```rust
use mathkong::ProbabilitySimilarity;

fn main() {
    let vector_a: Vec<f32> = vec![1.0, 2.0, 3.0];
    let vector_b: Vec<f32> = vec![4.0, 5.0, 6.0];

    let jensen_shannon = f32::jensenshannon(&vector_a, &vector_b)
        .expect("Vectors must be of the same length");

    println!("Jensen-Shannon Divergence: {}", jensen_shannon);

    let kullback_leibler = f32::kullbackleibler(&vector_a, &vector_b)
        .expect("Vectors must be of the same length");

    println!("Kullback-Leibler Divergence: {}", kullback_leibler);
}
```

Probability similarity functions are available for `f64`, `f32`, and `f16` types.

### Binary Similarity: Hamming and Jaccard Distances

Similar to spatial distances, one can compute bit-level distance functions between slices of unsigned integers:

```rust
use mathkong::BinarySimilarity;

fn main() {
    let vector_a = &[0b11110000, 0b00001111, 0b10101010];
    let vector_b = &[0b11110000, 0b00001111, 0b01010101];

    // Compute the Hamming distance between vectors
    let hamming_distance = u8::hamming(&vector_a, &vector_b)
        .expect("Vectors must be of the same length");

    println!("Hamming Distance: {}", hamming_distance);

    // Compute the Jaccard distance between vectors
    let jaccard_distance = u8::jaccard(&vector_a, &vector_b)
        .expect("Vectors must be of the same length");

    println!("Jaccard Distance: {}", jaccard_distance);
}
```

Binary similarity functions are available only for `u8` types.

### Half-Precision Floating-Point Numbers

Rust has no native support for half-precision floating-point numbers, but MathKong provides a `f16` type with built-in conversion methods.
The underlying `u16` representation is publicly accessible for direct bit manipulation.

```rust
use mathkong::{SpatialSimilarity, f16};

fn main() {
    // Create f16 vectors using built-in conversion methods
    let vector_a: Vec<f16> = vec![1.0, 2.0, 3.0].iter().map(|&x| f16::from_f32(x)).collect();
    let vector_b: Vec<f16> = vec![4.0, 5.0, 6.0].iter().map(|&x| f16::from_f32(x)).collect();

    // Compute the cosine distance
    let cosine_distance = f16::cosine(&vector_a, &vector_b)
        .expect("Vectors must be of the same length");
    
    println!("Cosine Distance: {}", cosine_distance);

    // Direct bit manipulation
    let half = f16::from_f32(3.14159);
    let bits = half.0; // Access raw u16 representation
    let reconstructed = f16(bits);
    
    // Convert back to f32
    let float_value = half.to_f32();
}
```

For interoperability with the `half` crate:

```rust
use mathkong::{SpatialSimilarity, f16 as SimF16};
use half::f16 as HalfF16;

fn main() {
    let vector_a: Vec<HalfF16> = vec![1.0, 2.0, 3.0].iter().map(|&x| HalfF16::from_f32(x)).collect();
    let vector_b: Vec<HalfF16> = vec![4.0, 5.0, 6.0].iter().map(|&x| HalfF16::from_f32(x)).collect();

    // Safe reinterpret cast due to identical memory layout
    let buffer_a: &[SimF16] = unsafe { std::slice::from_raw_parts(vector_a.as_ptr() as *const SimF16, vector_a.len()) };
    let buffer_b: &[SimF16] = unsafe { std::slice::from_raw_parts(vector_b.as_ptr() as *const SimF16, vector_b.len()) };

    let cosine_distance = SimF16::cosine(buffer_a, buffer_b)
        .expect("Vectors must be of the same length");

    println!("Cosine Distance: {}", cosine_distance);
}
```

### Half-Precision Brain-Float Numbers

The "brain-float-16" is a popular machine learning format.
It's broadly supported in hardware and is very machine-friendly, but software support is still lagging behind.
[Unlike NumPy](https://github.com/numpy/numpy/issues/19808), you can already use `bf16` datatype in MathKong.
MathKong provides a `bf16` type with built-in conversion methods and direct bit access.

```rust
use mathkong::{SpatialSimilarity, bf16};

fn main() {
    // Create bf16 vectors using built-in conversion methods
    let vector_a: Vec<bf16> = vec![1.0, 2.0, 3.0].iter().map(|&x| bf16::from_f32(x)).collect();
    let vector_b: Vec<bf16> = vec![4.0, 5.0, 6.0].iter().map(|&x| bf16::from_f32(x)).collect();

    // Compute the cosine similarity
    let cosine_distance = bf16::cosine(&vector_a, &vector_b)
        .expect("Vectors must be of the same length");
    
    println!("Cosine Distance: {}", cosine_distance);

    // Direct bit manipulation
    let brain_half = bf16::from_f32(3.14159);
    let bits = brain_half.0; // Access raw u16 representation
    let reconstructed = bf16(bits);
    
    // Convert back to f32
    let float_value = brain_half.to_f32();

    // Compare precision differences
    let original = 3.14159_f32;
    let f16_roundtrip = f16::from_f32(original).to_f32();
    let bf16_roundtrip = bf16::from_f32(original).to_f32();
    
    println!("Original: {}", original);
    println!("f16 roundtrip: {}", f16_roundtrip);
    println!("bf16 roundtrip: {}", bf16_roundtrip);
}
```

### Dynamic Dispatch in Rust

MathKong provides a [dynamic dispatch](#dynamic-dispatch) mechanism to select the most advanced micro-kernel for the current CPU.
You can query supported backends and use the `MathKong::capabilities` function to select the best one.

```rust
println!("uses neon: {}", capabilities::uses_neon());
println!("uses sve: {}", capabilities::uses_sve());
println!("uses haswell: {}", capabilities::uses_haswell());
println!("uses skylake: {}", capabilities::uses_skylake());
println!("uses ice: {}", capabilities::uses_ice());
println!("uses genoa: {}", capabilities::uses_genoa());
println!("uses sapphire: {}", capabilities::uses_sapphire());
println!("uses turin: {}", capabilities::uses_turin());
println!("uses sierra: {}", capabilities::uses_sierra());
```

## Using MathKong in JavaScript

To install, choose one of the following options depending on your environment:

- `npm install --save mathkong`
- `yarn add mathkong`
- `pnpm add mathkong`
- `bun install mathkong`

The package is distributed with prebuilt binaries, but if your platform is not supported, you can build the package from the source via `npm run build`.
This will automatically happen unless you install the package with the `--ignore-scripts` flag or use Bun.
After you install it, you will be able to call the MathKong functions on various `TypedArray` variants:

```js
const { sqeuclidean, cosine, inner, hamming, jaccard } = require("mathkong");

const vectorA = new Float32Array([1.0, 2.0, 3.0]);
const vectorB = new Float32Array([4.0, 5.0, 6.0]);

const distance = sqeuclidean(vectorA, vectorB);
console.log("Squared Euclidean Distance:", distance);
```

Other numeric types and precision levels are supported as well.
For double-precision floating-point numbers, use `Float64Array`:

```js
const vectorA = new Float64Array([1.0, 2.0, 3.0]);
const vectorB = new Float64Array([4.0, 5.0, 6.0]);
const distance = cosine(vectorA, vectorB);
```

When doing machine learning and vector search with high-dimensional vectors you may want to quantize them to 8-bit integers.
You may want to project values from the $[-1, 1]$ range to the $[-127, 127]$ range and then cast them to `Int8Array`:

```js
const quantizedVectorA = new Int8Array(vectorA.map((v) => v * 127));
const quantizedVectorB = new Int8Array(vectorB.map((v) => v * 127));
const distance = cosine(quantizedVectorA, quantizedVectorB);
```

A more extreme quantization case would be to use binary vectors.
You can map all positive values to `1` and all negative values and zero to `0`, packing eight values into a single byte.
After that, Hamming and Jaccard distances can be computed.

```js
const { toBinary, hamming } = require("mathkong");

const binaryVectorA = toBinary(vectorA);
const binaryVectorB = toBinary(vectorB);
const distance = hamming(binaryVectorA, binaryVectorB);
```

## Using MathKong in Swift

To install, simply add the following dependency to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/ashvardanian/mathkong")
]
```

The package provides the most common spatial metrics for `Int8`, `Float16`, `Float32`, and `Float64` vectors.

```swift
import MathKong

let vectorA: [Int8] = [1, 2, 3]
let vectorB: [Int8] = [4, 5, 6]

let dotProduct = vectorA.dot(vectorB)           // Computes the dot product
let angularDistance = vectorA.angular(vectorB)  // Computes the angular distance
let sqEuclidean = vectorA.sqeuclidean(vectorB)  // Computes the squared Euclidean distance
```

## Using MathKong in C

For integration within a CMake-based project, add the following segment to your `CMakeLists.txt`:

```cmake
FetchContent_Declare(
    mathkong
    GIT_REPOSITORY https://github.com/ashvardanian/mathkong.git
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(mathkong)
```

After that, you can use the MathKong library in your C code in several ways.
Simplest of all, you can include the headers, and the compiler will automatically select the most recent CPU extensions that MathKong will use.

```c
#include <mathkong/mathkong.h>

int main() {
    mathkong_flush_denormals(); // Optional, to avoid performance penalties on denormal numbers

    mathkong_f32_t vector_a[1536];
    mathkong_f32_t vector_b[1536];
    mathkong_kernel_punned_t distance_function = mathkong_metric_punned(
        mathkong_angular_k,  // Metric kind, like the angular cosine distance
        mathkong_f32_k,      // Data type, like: f16, f32, f64, i8, b8, complex variants, etc.
        mathkong_cap_any_k); // Which CPU capabilities are we allowed to use
    mathkong_distance_t distance;
    mathkong_metric_dense_punned_t metric = (mathkong_metric_dense_punned_t)metric_punned;
    metric(vector_a, vector_b, 1536, &distance);
    return 0;
}
```

### Dynamic Dispatch in C

To avoid hard-coding the backend, you can rely on `c/lib.c` to prepackage all possible backends in one binary, and select the most recent CPU features at runtime.
That feature of the C library is called [dynamic dispatch](#dynamic-dispatch) and is extensively used in the Python, JavaScript, and Rust bindings.
To test which CPU features are available on the machine at runtime, use the following APIs:

```c
int uses_dynamic_dispatch = mathkong_uses_dynamic_dispatch(); // Check if dynamic dispatch was enabled
mathkong_capability_t capabilities = mathkong_capabilities();  // Returns a bitmask

int uses_neon = mathkong_uses_neon();
int uses_sve = mathkong_uses_sve();
int uses_haswell = mathkong_uses_haswell();
int uses_skylake = mathkong_uses_skylake();
int uses_ice = mathkong_uses_ice();
int uses_genoa = mathkong_uses_genoa();
int uses_sapphire = mathkong_uses_sapphire();
```

To override compilation settings and switch between runtime and compile-time dispatch, define the following macro:

```c
#define SIMSIMD_DYNAMIC_DISPATCH 1 // or 0
```

### Spatial Distances: Angular and Euclidean Distances

```c
#include <mathkong/mathkong.h>

int main() {
    mathkong_i8_t i8[1536];
    mathkong_i8_t u8[1536];
    mathkong_f64_t f64s[1536];
    mathkong_f32_t f32s[1536];
    mathkong_f16_t f16s[1536];
    mathkong_bf16_t bf16s[1536];
    mathkong_distance_t distance;

    // Angular distance between two vectors
    mathkong_angular_i8(i8s, i8s, 1536, &distance);
    mathkong_angular_u8(u8s, u8s, 1536, &distance);
    mathkong_angular_f16(f16s, f16s, 1536, &distance);
    mathkong_angular_f32(f32s, f32s, 1536, &distance);
    mathkong_angular_f64(f64s, f64s, 1536, &distance);
    mathkong_angular_bf16(bf16s, bf16s, 1536, &distance);

    // Euclidean distance between two vectors
    mathkong_l2sq_i8(i8s, i8s, 1536, &distance);
    mathkong_l2sq_u8(u8s, u8s, 1536, &distance);
    mathkong_l2sq_f16(f16s, f16s, 1536, &distance);
    mathkong_l2sq_f32(f32s, f32s, 1536, &distance);
    mathkong_l2sq_f64(f64s, f64s, 1536, &distance);
    mathkong_l2sq_bf16(bf16s, bf16s, 1536, &distance);

    return 0;
}
```

### Dot-Products: Inner and Complex Inner Products

```c
#include <mathkong/mathkong.h>

int main() {
    // MathKong provides "sized" type-aliases without relying on `stdint.h`
    mathkong_i8_t i8[1536];
    mathkong_i8_t u8[1536];
    mathkong_f16_t f16s[1536];
    mathkong_f32_t f32s[1536];
    mathkong_f64_t f64s[1536];
    mathkong_bf16_t bf16s[1536];
    mathkong_distance_t product;

    // Inner product between two real vectors
    mathkong_dot_i8(i8s, i8s, 1536, &product);
    mathkong_dot_u8(u8s, u8s, 1536, &product);
    mathkong_dot_f16(f16s, f16s, 1536, &product);
    mathkong_dot_f32(f32s, f32s, 1536, &product);
    mathkong_dot_f64(f64s, f64s, 1536, &product);
    mathkong_dot_bf16(bf16s, bf16s, 1536, &product);

    // MathKong provides complex types with `real` and `imag` fields
    mathkong_f64c_t f64s[768];
    mathkong_f32c_t f32s[768];
    mathkong_f16c_t f16s[768];
    mathkong_bf16c_t bf16s[768];
    mathkong_distance_t products[2]; // real and imaginary parts

    // Complex inner product between two vectors
    mathkong_dot_f16c(f16cs, f16cs, 768, &products[0]);
    mathkong_dot_f32c(f32cs, f32cs, 768, &products[0]);
    mathkong_dot_f64c(f64cs, f64cs, 768, &products[0]);
    mathkong_dot_bf16c(bf16cs, bf16cs, 768, &products[0]);

    // Complex conjugate inner product between two vectors
    mathkong_vdot_f16c(f16cs, f16cs, 768, &products[0]);
    mathkong_vdot_f32c(f32cs, f32cs, 768, &products[0]);
    mathkong_vdot_f64c(f64cs, f64cs, 768, &products[0]);
    mathkong_vdot_bf16c(bf16cs, bf16cs, 768, &products[0]);
    return 0;
}
```

### Binary Distances: Hamming and Jaccard Distances

```c
#include <mathkong/mathkong.h>

int main() {
    mathkong_b8_t b8s[1536 / 8]; // 8 bits per word
    mathkong_distance_t distance;
    mathkong_hamming_b8(b8s, b8s, 1536 / 8, &distance);
    mathkong_jaccard_b8(b8s, b8s, 1536 / 8, &distance);
    return 0;
}
```

### Probability Distributions: Jensen-Shannon and Kullback-Leibler Divergences

```c
#include <mathkong/mathkong.h>

int main() {
    mathkong_f64_t f64s[1536];
    mathkong_f32_t f32s[1536];
    mathkong_f16_t f16s[1536];
    mathkong_distance_t divergence;

    // Jensen-Shannon divergence between two vectors
    mathkong_js_f16(f16s, f16s, 1536, &divergence);
    mathkong_js_f32(f32s, f32s, 1536, &divergence);
    mathkong_js_f64(f64s, f64s, 1536, &divergence);

    // Kullback-Leibler divergence between two vectors
    mathkong_kl_f16(f16s, f16s, 1536, &divergence);
    mathkong_kl_f32(f32s, f32s, 1536, &divergence);
    mathkong_kl_f64(f64s, f64s, 1536, &divergence);
    return 0;
}
```

### Half-Precision Floating-Point Numbers

If you aim to utilize the `_Float16` functionality with MathKong, ensure your development environment is compatible with C 11.
For other MathKong functionalities, C 99 compatibility will suffice.
To explicitly disable half-precision support, define the following macro before imports:

```c
#define SIMSIMD_NATIVE_F16 0 // or 1
#define SIMSIMD_NATIVE_BF16 0 // or 1
#include <mathkong/mathkong.h>
```

### Compilation Settings and Debugging

`SIMSIMD_DYNAMIC_DISPATCH`:

> By default, MathKong is a header-only library.
> But if you are running on different generations of devices, it makes sense to pre-compile the library for all supported generations at once, and dispatch at runtime.
> This flag does just that and is used to produce the `mathkong.so` shared library, as well as the Python and other bindings.

For Arm: `SIMSIMD_TARGET_NEON`, `SIMSIMD_TARGET_SVE`, `SIMSIMD_TARGET_SVE2`, `SIMSIMD_TARGET_NEON_F16`, `SIMSIMD_TARGET_SVE_F16`, `SIMSIMD_TARGET_NEON_BF16`, `SIMSIMD_TARGET_SVE_BF16`.
For x86: (`SIMSIMD_TARGET_HASWELL`, `SIMSIMD_TARGET_SKYLAKE`, `SIMSIMD_TARGET_ICE`, `SIMSIMD_TARGET_GENOA`, `SIMSIMD_TARGET_SAPPHIRE`, `SIMSIMD_TARGET_TURIN`, `SIMSIMD_TARGET_SIERRA`.

> By default, MathKong automatically infers the target architecture and pre-compiles as many kernels as possible.
> In some cases, you may want to explicitly disable some of the kernels.
> Most often it's due to compiler support issues, like the lack of some recent intrinsics or low-precision numeric types.
> In other cases, you may want to disable some kernels to speed up the compilation process and trim the binary size.

`SIMSIMD_SQRT`, `SIMSIMD_RSQRT`, `SIMSIMD_LOG`:

> By default, for __non__-SIMD backends, MathKong may use `libc` functions like `sqrt` and `log`.
> Those are generally very accurate, but slow, and introduce a dependency on the C standard library.
> To avoid that you can override those definitions with your custom implementations, like: `#define SIMSIMD_RSQRT(x) (1 / sqrt(x))`.

## Algorithms & Design Decisions 📚

In general there are a few principles that MathKong follows:

- Avoid loop unrolling.
- Never allocate memory.
- Never throw exceptions or set `errno`.
- Keep all function arguments the size of the pointer.
- Avoid returning from public interfaces, use out-arguments instead.
- Don't over-optimize for old CPUs and single- and double-precision floating-point numbers.
- Prioritize mixed-precision and integer operations, and new ISA extensions.
- Prefer saturated arithmetic and avoid overflows.

Possibly, in the future:

- Best effort computation silencing `NaN` components in low-precision inputs.
- Detect overflows and report the distance with a "signaling" `NaN`.

Last, but not the least - don't build unless there is a demand for it.
So if you have a specific use-case, please open an issue or a pull request, and ideally, bring in more users with similar needs.

### Cosine Similarity, Reciprocal Square Root, and Newton-Raphson Iteration

The cosine similarity is the most common and straightforward metric used in machine learning and information retrieval.
Interestingly, there are multiple ways to shoot yourself in the foot when computing it.
The cosine similarity is the inverse of the cosine distance, which is the cosine of the angle between two vectors.

```math
\text{CosineSimilarity}(a, b) = \frac{a \cdot b}{\|a\| \cdot \|b\|}
```

```math
\text{CosineDistance}(a, b) = 1 - \frac{a \cdot b}{\|a\| \cdot \|b\|}
```

In NumPy terms, MathKong implementation is similar to:

```python
import numpy as np

def cos_numpy(a: np.ndarray, b: np.ndarray) -> float:
    ab, a2, b2 = np.dot(a, b), np.dot(a, a), np.dot(b, b) # Fused in MathKong
    if a2 == 0 and b2 == 0: result = 0                    # Same in SciPy
    elif ab == 0: result = 1                              # Division by zero error in SciPy
    else: result = 1 - ab / (sqrt(a2) * sqrt(b2))         # Bigger rounding error in SciPy
    return result
```

In SciPy, however, the cosine distance is computed as `1 - ab / np.sqrt(a2 * b2)`.
It handles the edge case of a zero and non-zero argument pair differently, resulting in a division by zero error.
It's not only less efficient, but also less accurate, given how the reciprocal square roots are computed.
The C standard library provides the `sqrt` function, which is generally very accurate, but slow.
The `rsqrt` in-hardware implementations are faster, but have different accuracy characteristics.

- SSE `rsqrtps` and AVX `vrsqrtps`: $1.5 \times 2^{-12}$ maximal relative error.
- AVX-512 `vrsqrt14pd` instruction: $2^{-14}$ maximal relative error.
- NEON `frsqrte` instruction has no documented error bounds, but [can be][arm-rsqrt] $2^{-3}$.

[arm-rsqrt]: https://gist.github.com/ashvardanian/5e5cf585d63f8ab6d240932313c75411

To overcome the limitations of the `rsqrt` instruction, MathKong uses the Newton-Raphson iteration to refine the initial estimate for high-precision floating-point numbers.
It can be defined as:

```math
x_{n+1} = x_n \cdot (3 - x_n \cdot x_n) / 2
```

On 1536-dimensional inputs on Intel Sapphire Rapids CPU a single such iteration can result in a 2-3 orders of magnitude relative error reduction:

| Datatype   |         NumPy Error | MathKong w/out Iteration |            MathKong |
| :--------- | ------------------: | -----------------------: | ------------------: |
| `bfloat16` | 1.89e-08 ± 1.59e-08 |      3.07e-07 ± 3.09e-07 | 3.53e-09 ± 2.70e-09 |
| `float16`  | 1.67e-02 ± 1.44e-02 |      2.68e-05 ± 1.95e-05 | 2.02e-05 ± 1.39e-05 |
| `float32`  | 2.21e-08 ± 1.65e-08 |      3.47e-07 ± 3.49e-07 | 3.77e-09 ± 2.84e-09 |
| `float64`  | 0.00e+00 ± 0.00e+00 |      3.80e-07 ± 4.50e-07 | 1.35e-11 ± 1.85e-11 |

### Curved Spaces, Mahalanobis Distance, and Bilinear Quadratic Forms

The Mahalanobis distance is a generalization of the Euclidean distance, which takes into account the covariance of the data.
It's very similar in its form to the bilinear form, which is a generalization of the dot product.

```math
\text{BilinearForm}(a, b, M) = a^T M b
```

```math
\text{Mahalanobis}(a, b, M) = \sqrt{(a - b)^T M^{-1} (a - b)}
```

Bilinear Forms can be seen as one of the most important linear algebraic operations, surprisingly missing in BLAS and LAPACK.
They are versatile and appear in various domains:

- In Quantum Mechanics, the expectation value of an observable $A$ in a state $\psi$ is given by $\langle \psi | A | \psi \rangle$, which is a bilinear form.
- In Machine Learning, in Support Vector Machines (SVMs), bilinear forms define kernel functions that measure similarity between data points.
- In Differential Geometry, the metric tensor, which defines distances and angles on a manifold, is a bilinear form on the tangent space.
- In Economics, payoff functions in certain Game Theoretic problems can be modeled as bilinear forms of players' strategies.
- In Physics, interactions between electric and magnetic fields can be expressed using bilinear forms.

Broad applications aside, the lack of a specialized primitive for bilinear forms in BLAS and LAPACK means significant performance overhead.
A $vector * matrix * vector$ product is a scalar, whereas its constituent parts ($vector * matrix$ and $matrix * vector$) are vectors:

- They need memory to be stored in: $O(n)$ allocation.
- The data will be written to memory and read back, wasting CPU cycles.

MathKong doesn't produce intermediate vector results, like `a @ M @ b`, but computes the bilinear form directly.

### Set Intersection, Galloping, and Binary Search

The set intersection operation is generally defined as the number of elements that are common between two sets, represented as sorted arrays of integers.
The most common way to compute it is a linear scan:

```c
size_t intersection_size(int *a, int *b, size_t n, size_t m) {
    size_t i = 0, j = 0, count = 0;
    while (i < n && j < m) {
        if (a[i] < b[j]) i++;
        else if (a[i] > b[j]) j++;
        else i++, j++, count++;
    }
    return count;
}
```

Alternatively, one can use the binary search to find the elements in the second array that are present in the first one.
On every step the checked region of the second array is halved, which is called the _galloping search_.
It's faster, but only when large arrays of very different sizes are intersected.
Third approach is to use the SIMD instructions to compare multiple elements at once:

- Using string-intersection instructions on x86, like `pcmpestrm`.
- Using integer-intersection instructions in AVX-512, like `vp2intersectd`.
- Using vanilla equality checks present in all SIMD instruction sets.

After benchmarking, the last approach was chosen, as it's the most flexible and often the fastest.

### Complex Dot Products, Conjugate Dot Products, and Complex Numbers

Complex dot products are a generalization of the dot product to complex numbers.
They are supported by most BLAS packages, but almost never in mixed precision.
MathKong defines `dot` and `vdot` kernels as:

```math
\text{dot}(a, b) = \sum_{i=0}^{n-1} a_i \cdot b_i
```

```math
\text{vdot}(a, b) = \sum_{i=0}^{n-1} a_i \cdot \bar{b_i}
```

Where $\bar{b_i}$ is the complex conjugate of $b_i$.
Putting that into Python code for scalar arrays:

```python
def dot(a: List[number], b: List[number]) -> number:
    a_real, a_imaginary = a[0::2], a[1::2]
    b_real, b_imaginary = b[0::2], b[1::2]
    ab_real, ab_imaginary = 0, 0
    for ar, ai, br, bi in zip(a_real, a_imaginary, b_real, b_imaginary):
        ab_real += ar * br - ai * bi
        ab_imaginary += ar * bi + ai * br
    return ab_real, ab_imaginary

def vdot(a: List[number], b: List[number]) -> number:
    a_real, a_imaginary = a[0::2], a[1::2]
    b_real, b_imaginary = b[0::2], b[1::2]
    ab_real, ab_imaginary = 0, 0
    for ar, ai, br, bi in zip(a_real, a_imaginary, b_real, b_imaginary):
        ab_real += ar * br + ai * bi
        ab_imaginary += ar * bi - ai * br
    return ab_real, ab_imaginary
```

### Logarithms in Kullback-Leibler & Jensen–Shannon Divergences

The Kullback-Leibler divergence is a measure of how one probability distribution diverges from a second, expected probability distribution.
Jensen-Shannon divergence is a symmetrized and smoothed version of the Kullback-Leibler divergence, which can be used as a distance metric between probability distributions.

```math
\text{KL}(P || Q) = \sum_{i} P(i) \log \frac{P(i)}{Q(i)}
```

```math
\text{JS}(P, Q) = \frac{1}{2} \text{KL}(P || M) + \frac{1}{2} \text{KL}(Q || M), M = \frac{P + Q}{2}
```

Both functions are defined for non-negative numbers, and the logarithm is a key part of their computation.

### Mixed Precision in Fused-Multiply-Add and Weighted Sums

The "Fused-Multiply-Add" (FMA) operation is a single operation that combines element-wise multiplication and addition with different scaling factors.
The "Weighted Sum" is it's simplified variant without element-wise multiplication.
The "Sum" operation is a further simplified variant without scaling factors, and "Scale" is the unary equivalent of FMA:

```math
\text{Scale}_i(A, \alpha, \beta) = \alpha \cdot A_i + \beta
```

```math
\text{Sum}_i(A, B) = A_i + B_i
```

```math
\text{WSum}_i(A, B, \alpha, \beta) = \alpha \cdot A_i + \beta \cdot B_i
```

```math
\text{FMA}_i(A, B, C, \alpha, \beta) = \alpha \cdot A_i \cdot B_i + \beta \cdot C_i
```

In NumPy terms, the implementation __may__ look like:

```py
import numpy as np

def scale(A: np.ndarray, /, Alpha: float, Beta: float) -> np.ndarray:
    return (Alpha * A + Beta).astype(A.dtype)

def sum(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    assert A.dtype == B.dtype, "Input types must match and affect the output style"
    return (A + B).astype(A.dtype)

def wsum(A: np.ndarray, B: np.ndarray, /, Alpha: float, Beta: float) -> np.ndarray:
    assert A.dtype == B.dtype, "Input types must match and affect the output style"
    return (Alpha * A + Beta * B).astype(A.dtype)

def fma(A: np.ndarray, B: np.ndarray, C: np.ndarray, /, Alpha: float, Beta: float) -> np.ndarray:
    assert A.dtype == B.dtype and A.dtype == C.dtype, "Input types must match and affect the output style"
    return (Alpha * A * B + Beta * C).astype(A.dtype)
```

The tricky part is implementing those operations in mixed precision, where the scaling factors are of different precision than the input and output vectors.
MathKong uses double-precision floating-point scaling factors for any input and output precision, including `i8` and `u8` integers and `f16` and `bf16` floats.
Depending on the generation of the CPU, given native support for `f16` addition and multiplication, the `f16` temporaries are used for `i8` and `u8` multiplication, scaling, and addition.
For `bf16`, native support is generally limited to dot-products with subsequent partial accumulation, which is not enough for the FMA and WSum operations, so `f32` is used as a temporary.

### Auto-Vectorization & Loop Unrolling

On the Intel Sapphire Rapids platform, MathKong was benchmarked against auto-vectorized code using GCC 12.
GCC handles single-precision `float` but might not be the best choice for `int8` and `_Float16` arrays, which have been part of the C language since 2011.

| Kind                      | GCC 12 `f32` | GCC 12 `f16` | MathKong `f16` | `f16` improvement |
| :------------------------ | -----------: | -----------: | -------------: | ----------------: |
| Inner Product             |    3,810 K/s |      192 K/s |      5,990 K/s |          __31 x__ |
| Cosine Distance           |    3,280 K/s |      336 K/s |      6,880 K/s |          __20 x__ |
| Euclidean Distance ²      |    4,620 K/s |      147 K/s |      5,320 K/s |          __36 x__ |
| Jensen-Shannon Divergence |    1,180 K/s |       18 K/s |      2,140 K/s |         __118 x__ |

### Dynamic Dispatch

Most popular software is precompiled and distributed with fairly conservative CPU optimizations, to ensure compatibility with older hardware.
Database Management platforms, like ClickHouse, and Web Browsers, like Google Chrome,need to run on billions of devices, and they can't afford to be picky about the CPU features.
For such users MathKong provides a dynamic dispatch mechanism, which selects the most advanced micro-kernel for the current CPU at runtime.

<table>
  <tr>
    <th>Subset</th>
    <th>F</th>
    <th>CD</th>
    <th>ER</th>
    <th>PF</th>
    <th>4FMAPS</th>
    <th>4VNNIW</th>
    <th>VPOPCNTDQ</th>
    <th>VL</th>
    <th>DQ</th>
    <th>BW</th>
    <th>IFMA</th>
    <th>VBMI</th>
    <th>VNNI</th>
    <th>BF16</th>
    <th>VBMI2</th>
    <th>BITALG</th>
    <th>VPCLMULQDQ</th>
    <th>GFNI</th>
    <th>VAES</th>
    <th>VP2INTERSECT</th>
    <th>FP16</th>
  </tr>
  <tr>
    <td><a href="https://en.wikipedia.org/wiki/Xeon_Phi#Knights_Landing">Knights Landing</a> (Xeon Phi x200, 2016)</td>
    <td colspan="2" rowspan="9" style="background:#9EFF9E;color:black;vertical-align:middle;text-align:center;">Yes</td>
    <td colspan="2" rowspan="2" style="background:#9EFF9E;color:black;vertical-align:middle;text-align:center;">Yes</td>
    <td colspan="17" style="background:#FFC7C7;color:black;vertical-align:middle;text-align:center;">No</td>
  </tr>
  <tr>
    <td><a href="https://en.wikipedia.org/wiki/Xeon_Phi#Knights_Mill">Knights Mill</a> (Xeon Phi x205, 2017)</td>
    <td colspan="3" style="background:#9EFF9E;color:black;vertical-align:middle;text-align:center;">Yes</td>
    <td colspan="14" style="background:#FFC7C7;color:black;vertical-align:middle;text-align:center;">No</td>
  </tr>
  <tr>
    <td>
      <a href="https://en.wikipedia.org/wiki/Skylake_(microarchitecture)#Skylake-SP_(14_nm)_Scalable_Performance">Skylake-SP</a>, 
      <a href="https://en.wikipedia.org/wiki/Skylake_(microarchitecture)#Mainstream_desktop_processors">Skylake-X</a> (2017)
    </td>
    <td colspan="4" rowspan="11" style="background:#FFC7C7;color:black;vertical-align:middle;text-align:center;">No</td>
    <td rowspan="4" style="background:#FFC7C7;color:black;vertical-align:middle;text-align:center;">No</td>
    <td colspan="3" rowspan="4" style="background:#9EFF9E;color:black;vertical-align:middle;text-align:center;">Yes</td>
    <td colspan="11" style="background:#FFC7C7;color:black;vertical-align:middle;text-align:center;">No</td>
  </tr>
  <tr>
    <td><a href="https://en.wikipedia.org/wiki/Cannon_Lake_(microarchitecture)">Cannon Lake</a> (2018)</td>
    <td colspan="2" style="background:#9EFF9E;color:black;vertical-align:middle;text-align:center;">Yes</td>
    <td colspan="9" style="background:#FFC7C7;color:black;vertical-align:middle;text-align:center;">No</td>
  </tr>
  <tr>
    <td><a href="https://en.wikipedia.org/wiki/Cascade_Lake_(microarchitecture)">Cascade Lake</a> (2019)</td>
    <td colspan="2" rowspan="2" style="background:#FFC7C7;color:black;vertical-align:middle;text-align:center;">No</td>
    <td rowspan="2" style="background:#9EFF9E;color:black;vertical-align:middle;text-align:center;">Yes</td>
    <td colspan="8" style="background:#FFC7C7;color:black;vertical-align:middle;text-align:center;">No</td>
  </tr>
  <tr>
    <td><a href="https://en.wikipedia.org/wiki/Cooper_Lake_(microarchitecture)">Cooper Lake</a> (2020)</td>
    <td style="background:#9EFF9E;color:black;vertical-align:middle;text-align:center;">Yes</td>
    <td colspan="7" style="background:#FFC7C7;color:black;vertical-align:middle;text-align:center;">No</td>
  </tr>
  <tr>
    <td><a href="https://en.wikipedia.org/wiki/Ice_Lake_(microarchitecture)">Ice Lake</a> (2019)</td>
    <td colspan="7" rowspan="3" style="background:#9EFF9E;color:black;vertical-align:middle;text-align:center;">Yes</td>
    <td rowspan="3" style="background:#FFC7C7;color:black;vertical-align:middle;text-align:center;">No</td>
    <td colspan="5" rowspan="3" style="background:#9EFF9E;color:black;vertical-align:middle;text-align:center;">Yes</td>
    <td colspan="2" style="background:#FFC7C7;color:black;vertical-align:middle;text-align:center;">No</td>
  </tr>
  <tr>
    <td><a href="https://en.wikipedia.org/wiki/Tiger_Lake_(microarchitecture)">Tiger Lake</a> (2020)</td>
    <td style="background:#9EFF9E;color:black;vertical-align:middle;text-align:center;">Yes</td>
    <td style="background:#FFC7C7;color:black;vertical-align:middle;text-align:center;">No</td>
  </tr>
  <tr>
    <td><a href="https://en.wikipedia.org/wiki/Rocket_Lake">Rocket Lake</a> (2021)</td>
    <td colspan="2" style="background:#FFC7C7;color:black;vertical-align:middle;text-align:center;">No</td>
  </tr>
  <tr>
    <td><a href="https://en.wikipedia.org/wiki/Alder_Lake">Alder Lake</a> (2021)</td>
    <td colspan="2" style="background:#FFB;color:black;vertical-align:middle;text-align:center;">Partial</td>
    <td colspan="15" style="background:#FFB;color:black;vertical-align:middle;text-align:center;">Partial</td>
  </tr>
  <tr>
    <td><a href="https://en.wikipedia.org/wiki/Zen_4">Zen 4</a> (2022)</td>
    <td colspan="2" rowspan="3" style="background:#9EFF9E;color:black;vertical-align:middle;text-align:center;">Yes</td>
    <td colspan="13" rowspan="3" style="background:#9EFF9E;color:black;vertical-align:middle;text-align:center;">Yes</td>
    <td colspan="2" style="background:#FFC7C7;color:black;vertical-align:middle;text-align:center;">No</td>
  </tr>
  <tr>
    <td><a href="https://en.wikipedia.org/wiki/Sapphire_Rapids_(microprocessor)">Sapphire Rapids</a> (2023)</td>
    <td style="background:#FFC7C7;color:black;vertical-align:middle;text-align:center;">No</td>
    <td style="background:#9EFF9E;color:black;vertical-align:middle;text-align:center;">Yes</td>
  </tr>
  <tr>
    <td><a href="https://en.wikipedia.org/wiki/Zen_5">Zen 5</a> (2024)</td>
    <td style="background:#9EFF9E;color:black;vertical-align:middle;text-align:center;">Yes</td>
    <td style="background:#FFC7C7;color:black;vertical-align:middle;text-align:center;">No</td>
  </tr>
</table>

You can compile MathKong on an old CPU, like Intel Haswell, and run it on a new one, like AMD Genoa, and it will automatically use the most advanced instructions available.
Reverse is also true, you can compile on a new CPU and run on an old one, and it will automatically fall back to the most basic instructions.
Moreover, the very first time you prove for CPU capabilities with `mathkong_capabilities()`, it initializes the dynamic dispatch mechanism, and all subsequent calls will be faster and won't face race conditions in multi-threaded environments.

## Target Specific Backends

MathKong exposes all kernels for all backends, and you can select the most advanced one for the current CPU without relying on built-in dispatch mechanisms.
That's handy for testing and benchmarking, but also in case you want to dispatch a very specific kernel for a very specific CPU, bypassing MathKong assignment logic.
All of the function names follow the same pattern: `mathkong_{function}_{type}_{backend}`.

- The backend can be `serial`, `haswell`, `skylake`, `ice`, `genoa`, `sapphire`, `turin`, `neon`, or `sve`.
- The type can be `f64`, `f32`, `f16`, `bf16`, `f64c`, `f32c`, `f16c`, `bf16c`, `i8`, or `b8`.
- The function can be `dot`, `vdot`, `cos`, `l2sq`, `hamming`, `jaccard`, `kl`, `js`, or `intersect`.

To avoid hard-coding the backend, you can use the `mathkong_kernel_punned_t` to pun the function pointer and the `mathkong_capabilities` function to get the available backends at runtime.
To match all the function names, consider a RegEx:

```regex
SIMSIMD_PUBLIC void mathkong_\w+_\w+_\w+\(
```

On Linux, you can use the following command to list all unique functions:

```sh
$ grep -oP 'SIMSIMD_PUBLIC void mathkong_\w+_\w+_\w+\(' include/mathkong/*.h | sort | uniq
> include/mathkong/binary.h:SIMSIMD_PUBLIC void mathkong_hamming_b8_haswell(
> include/mathkong/binary.h:SIMSIMD_PUBLIC void mathkong_hamming_b8_ice(
> include/mathkong/binary.h:SIMSIMD_PUBLIC void mathkong_hamming_b8_neon(
> include/mathkong/binary.h:SIMSIMD_PUBLIC void mathkong_hamming_b8_serial(
> include/mathkong/binary.h:SIMSIMD_PUBLIC void mathkong_hamming_b8_sve(
> include/mathkong/binary.h:SIMSIMD_PUBLIC void mathkong_jaccard_b8_haswell(
> include/mathkong/binary.h:SIMSIMD_PUBLIC void mathkong_jaccard_b8_ice(
> include/mathkong/binary.h:SIMSIMD_PUBLIC void mathkong_jaccard_b8_neon(
> include/mathkong/binary.h:SIMSIMD_PUBLIC void mathkong_jaccard_b8_serial(
> include/mathkong/binary.h:SIMSIMD_PUBLIC void mathkong_jaccard_b8_sve(
```

## License

Feel free to use the project under Apache 2.0 or the Three-clause BSD license at your preference.

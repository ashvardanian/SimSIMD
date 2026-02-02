![NumKong banner](https://github.com/ashvardanian/ashvardanian/blob/master/repositories/NumKong.jpg?raw=true)

Computing dot-products, similarity measures, and distances between low- and high-dimensional vectors is ubiquitous in Machine Learning, Scientific Computing, Geospatial Analysis, and Information Retrieval.
These algorithms generally have linear complexity in time, constant or linear complexity in space, and are data-parallel.
In other words, it is easily parallelizable and vectorizable and often available in packages like BLAS (level 1) and LAPACK, as well as higher-level `numpy` and `scipy` Python libraries.
Ironically, even with decades of evolution in compilers and numerical computing, [most libraries can be 3-200x slower than hardware potential][benchmarks] even on the most popular hardware, like 64-bit x86 and Arm CPUs.
Moreover, most lack mixed-precision support, which is crucial for modern AI!
The rare few that support minimal mixed precision, run only on one platform, and are vendor-locked, by companies like Intel and Nvidia.
NumKong provides an alternative.
1️⃣ NumKong functions are practically as fast as `memcpy`.
2️⃣ Unlike BLAS, most kernels are designed for mixed-precision and bit-level operations.
3️⃣ NumKong often [ships more binaries than NumPy][compatibility] and has more backends than most BLAS implementations, and more high-level interfaces than most libraries.

[benchmarks]: https://ashvardanian.com/posts/numkong-faster-scipy
[compatibility]: https://pypi.org/project/numkong/#files

<div>
<a href="https://pepy.tech/project/numkong">
    <img alt="PyPI" src="https://static.pepy.tech/personalized-badge/numkong?period=total&units=abbreviation&left_color=black&right_color=blue&left_text=NumKong%20Python%20installs" />
</a>
<a href="https://www.npmjs.com/package/numkong">
    <img alt="npm" src="https://img.shields.io/npm/dy/numkong?label=JavaScript%20NPM%20installs" />
</a>
<a href="https://crates.io/crates/numkong">
    <img alt="rust" src="https://img.shields.io/crates/d/numkong?label=Rust%20Crate%20installs" />
</a>
<img alt="GitHub code size in bytes" src="https://img.shields.io/github/languages/code-size/ashvardanian/numkong">
<a href="https://github.com/ashvardanian/NumKong/actions/workflows/release.yml">
    <img alt="GitHub Actions Ubuntu" src="https://img.shields.io/github/actions/workflow/status/ashvardanian/NumKong/release.yml?branch=main&label=Ubuntu&logo=github&color=blue">
</a>
<a href="https://github.com/ashvardanian/NumKong/actions/workflows/release.yml">
    <img alt="GitHub Actions Windows" src="https://img.shields.io/github/actions/workflow/status/ashvardanian/NumKong/release.yml?branch=main&label=Windows&logo=windows&color=blue">
</a>
<a href="https://github.com/ashvardanian/NumKong/actions/workflows/release.yml">
    <img alt="GitHub Actions macOS" src="https://img.shields.io/github/actions/workflow/status/ashvardanian/NumKong/release.yml?branch=main&label=macOS&logo=apple&color=blue">
</a>
<a href="https://github.com/ashvardanian/NumKong/actions/workflows/release.yml">
    <img alt="GitHub Actions CentOS Linux" src="https://img.shields.io/github/actions/workflow/status/ashvardanian/NumKong/release.yml?branch=main&label=CentOS&logo=centos&color=blue">
</a>

</div>

## Features

__NumKong__ (Arabic: "سيمسيم دي") is a mixed-precision math library of __over 350 SIMD-optimized kernels__ extensively used in AI, Search, and DBMS workloads.
Named after the iconic ["Open Sesame"](https://en.wikipedia.org/wiki/Open_sesame) command that opened doors to treasure in _Ali Baba and the Forty Thieves_, NumKong can help you 10x the cost-efficiency of your computational pipelines.
Implemented distance functions include:

- Euclidean (L2) and Cosine (Angular) spatial distances for Vector Search. _[docs][docs-spatial]_
- Dot-Products for real & complex vectors for DSP & Quantum computing. _[docs][docs-dot]_
- Hamming (~ Manhattan) and Jaccard (~ Tanimoto) bit-level distances. _[docs][docs-binary]_
- Set Intersections for Sparse Vectors and Text Analysis. _[docs][docs-sparse]_
- Mahalanobis distance and Quadratic forms for Scientific Computing. _[docs][docs-curved]_
- Kullback-Leibler and Jensen–Shannon divergences for probability distributions. _[docs][docs-probability]_
- Fused-Multiply-Add (FMA) and Weighted Sums to replace BLAS level 1 functions. _[docs][docs-fma]_
- For Levenshtein, Needleman–Wunsch, and Smith-Waterman, check [StringZilla][stringzilla].
- 🔜 Haversine and Vincenty's formulae for Geospatial Analysis.

[docs-spatial]: #cosine-similarity-reciprocal-square-root-and-newton-raphson-iteration
[docs-curved]: #curved-spaces-mahalanobis-distance-and-bilinear-quadratic-forms
[docs-sparse]: #set-intersection-galloping-and-binary-search
[docs-binary]: https://github.com/ashvardanian/NumKong/pull/138
[docs-dot]: #complex-dot-products-conjugate-dot-products-and-complex-numbers
[docs-probability]: #logarithms-in-kullback-leibler--jensenshannon-divergences
[docs-fma]: #mixed-precision-in-fused-multiply-add-and-weighted-sums
[scipy]: https://docs.scipy.org/doc/scipy/reference/spatial.distance.html#module-scipy.spatial.distance
[numpy]: https://numpy.org/doc/stable/reference/generated/numpy.inner.html
[stringzilla]: https://github.com/ashvardanian/stringzilla

Moreover, NumKong...

- handles `float64`, `float32`, `float16`, and `bfloat16` real & complex vectors.
- handles `float8_e4m3`, `float8_e5m2`, and other esotheric AI datatypes.
- handles `int8` integral, `int4` sub-byte, and `uint1` binary vectors.
- handles sparse `uint32` and `uint16` sets, and weighted sparse vectors.
- is a zero-dependency [header-only C 99](#using-numkong-in-c) library.
- has [Python](#using-numkong-in-python), [Rust](#using-numkong-in-rust), [JS](#using-numkong-in-javascript), and [Swift](#using-numkong-in-swift) bindings.
- has Arm backends for NEON, Scalable Vector Extensions (SVE), and SVE2.
- has x86 backends for Haswell, Skylake, Ice Lake, Genoa, and Sapphire Rapids.
- has RISC-V backends for Vector extensions (SpaceMIT, SiFive, XuanTie).
- has WebAssembly backend with Relaxed SIMD for browser and edge deployments.
- with both compile-time and runtime CPU feature detection easily integrates anywhere!

Due to the high-level of fragmentation of SIMD support in different x86 CPUs, NumKong generally uses the names of select Intel CPU generations for its backends.
They, however, also work on AMD CPUs.
Intel Haswell is compatible with AMD Zen 1/2/3, while AMD Genoa Zen 4 covers AVX-512 instructions added to Intel Skylake and Ice Lake.
You can learn more about the technical implementation details in the following blog-posts:

- [Uses Horner's method for polynomial approximations, beating GCC 12 by 119x](https://ashvardanian.com/posts/gcc-12-vs-avx512fp16/).
- [Uses Arm SVE and x86 AVX-512's masked loads to eliminate tail `for`-loops](https://ashvardanian.com/posts/numkong-faster-scipy/#tails-of-the-past-the-significance-of-masked-loads).
- [Substitutes libc's `sqrt` with Newton Raphson iterations](https://github.com/ashvardanian/NumKong/releases/tag/v5.4.0).
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
    <th align="center">NumKong</th>
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
    <td align="center"> <!-- numkong -->
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
    <td align="center"> <!-- numkong -->
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
    <td align="center"> <!-- numkong -->
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
    <td align="center"> <!-- numkong -->
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
    <td align="center"> <!-- numkong -->
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
    <td align="center"> <!-- numkong -->
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
    <td align="center"> <!-- numkong -->
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
    <td align="center"> <!-- numkong -->
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
    <td align="center"> <!-- numkong -->
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
    <td align="center"> <!-- numkong -->
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
> Variance within x86 CPUs will be larger.

Similar speedups are often observed even when compared to BLAS and LAPACK libraries underlying most numerical computing libraries, including NumPy and SciPy in Python.
Broader benchmarking results:

- [Apple M2 Pro](https://ashvardanian.com/posts/numkong-faster-scipy/#appendix-1-performance-on-apple-m2-pro).
- [Intel Sapphire Rapids](https://ashvardanian.com/posts/numkong-faster-scipy/#appendix-2-performance-on-4th-gen-intel-xeon-platinum-8480).
- [AWS Graviton 3](https://ashvardanian.com/posts/numkong-faster-scipy/#appendix-3-performance-on-aws-graviton-3).


## Numeric Types

### Float64

For double-precision numerics, NumKong deviates from most BLAS-like libraries by leveraging __stable compensated summation algorithms__ that track numerical errors separately. On serial paths, we use __Neumaier's algorithm__ (1974), an improvement over Kahan-Babuška that correctly handles cases where added terms are larger than the running sum, achieving O(1) error growth instead of O(n). On SIMD paths with FMA support, we implement the __Dot2 algorithm__ (Ogita-Rump-Oishi, 2005) which maintains separate error compensators for both multiplication and accumulation, capturing rounding errors via `TwoProd` and `TwoSum` operations.

Beyond dot products, Float64 enables __Mahalanobis distance__ computation for statistical analysis and __bilinear forms__ for quantum computing, both using the same compensated summation to preserve precision through long arithmetic circuits. For probability distributions, __Kullback-Leibler__ and __Jensen-Shannon divergences__ benefit from F64's 52-bit mantissa when computing `log(p/q)` ratios that approach machine epsilon. The precision overhead is worthwhile: on 1024³ GEMM operations, NumKong's compensated F64 achieves 10-50x smaller ULP errors than Intel MKL despite running at 3x lower throughput, making it ideal for scientific computing where numerical stability matters more than raw speed.

```c
// Dot2 TwoProd: Capture multiplication rounding error
h = a * b;
r = fma(a, b, -h);  // Extracts rounding error

// Dot2 TwoSum: Capture addition rounding error
t = sum + product;
e = (sum - t) + product;  // Compensator term
```

### Float32

For traditional single-precision inputs, NumKong uses __double-precision FMA__ to reduce numerical error in long arithmetic circuits. The SIMD implementations load 4 F32 values, upcast to F64 for full-precision multiplication and accumulation, then downcast only during finalization. This avoids catastrophic cancellation at minimal cost since modern CPUs have dedicated F64 vector units operating at nearly the same throughput as F32.

For __complex dot products__, NumKong employs a clever optimization that doubles throughput: instead of using separate FMA and FMS (fused multiply-subtract) instructions for the real component, we defer sign flips until after the accumulation loop completes, applying a single bitwise XOR to flip sign bits. This eliminates execution port contention—allowing dual FMA units to run at full capacity—and improves throughput from 2.5 GB/s to 5 GB/s on Haswell. The transformation is simple: compute `a_r×b_r + a_i×b_i` (treating negatives as positive), then XOR with `0x80000000` to flip every second element's sign bit after the loop.

NumKong also handles __curved space metrics__ like the Mahalanobis distance `√((a-b)ᵀ C (a-b))` where C is a metric tensor, applying the same compensated accumulation strategies to preserve precision when multiplying small differences by large inverse covariance matrices. For __probability distributions__, we compute KL divergence using log2 decomposition via `VGETEXP` (extract exponent) and polynomial approximations for the mantissa, avoiding expensive transcendental function calls.

```c
// Complex multiply optimization: XOR sign flip AFTER loop
for (...) {
    sum_real = fma(a, b, sum_real);  // No sign flip in loop
    sum_imag = fma(a, b_swapped, sum_imag);
}
sum_real = xor(sum_real, 0x80000000);  // Single XOR after loop
```

### BFloat16

Brain Float is not an IEEE 754 standard type, but it has become the __universal recommendation__ for AI-related applications due to its unique properties: on old CPUs, upcasting BF16 to F32 requires just an unpack and left-shift by 16 bits (essentially free), while on new CPUs, both Arm and x86 provide widening mixed-precision dot products via __DPBF16PS__ (AVX-512 on Genoa/Sapphire Rapids) and __BFDOT__ (NEON on ARMv8.6-A Graviton 3+). These instructions compute two BF16 products per lane, accumulating directly into F32 registers, achieving 2-4x higher throughput than separate conversion and multiply-add.

The format itself prioritizes __dynamic range over precision__: BF16 shares Float32's 8-bit exponent but truncates the mantissa to 7 bits, allowing it to represent the same range (±3.4×10³⁸) with coarser granularity. This makes it perfect for neural network weights and activations where large dynamic range matters more than precision near zero. NumKong's Float8 types (E4M3/E5M2) upcast to BF16 before using DPBF16PS, effectively creating a three-tier precision hierarchy: F8 for storage, BF16 for compute, F32 for accumulation.

Beyond dot products, BF16 excels at __cosine similarity__ where division by magnitudes benefits from the wide exponent range, and __element-wise operations__ like `α×a + β` where the BF16→F32 upcast happens once per element rather than repeatedly. For __L2 distance__, the formula `√(‖a‖² + ‖b‖² - 2ab)` accumulates into F32 then applies Newton-Raphson reciprocal sqrt, achieving 9x speedup over NumPy while maintaining better numerical accuracy.

| Platform         | BF16 Instruction | Latency | Throughput | Elements/Cycle |
| ---------------- | ---------------- | ------- | ---------- | -------------- |
| Genoa AVX-512    | DPBF16PS         | 6 cy    | 2/cy       | 32 BF16/cy     |
| Graviton 3 NEON  | BFDOT            | 4 cy    | 2/cy       | 8 BF16/cy      |
| Haswell (compat) | Upcast→FMA       | 8 cy    | 1/cy       | 8 BF16/cy      |

### Float16

IEEE 754 half-precision floating point (FP16) is widely supported across modern hardware with 1 sign bit, 5 exponent bits (bias=15), and 10 mantissa bits, giving a range of ±65504. Unlike BFloat16 which prioritizes exponent range, FP16 prioritizes __precision over range__ (10 vs 7 mantissa bits), making it better suited for values near zero, gradients during training, and scientific computing where relative error matters more than dynamic range.

On x86, older CPUs use __F16C extensions__ (Ivy Bridge+) providing `VCVTPH2PS` for fast F16→F32 conversion, then standard FMA for computation. Newer CPUs (Sapphire Rapids+) add native __AVX-512-FP16__ with dedicated F16 arithmetic, though NumKong still widens to F32 for accumulation to preserve precision. On Arm, the story is richer: basic ARMv8.2-A provides `FCVTL` (3-cycle conversion) followed by `FMLA` (4-cycle FMA), but ARMv8.4-A adds __FMLAL/FMLAL2__ instructions for fused F16→F32 widening multiply-accumulate, reducing the total latency from 7 cycles to 4 cycles and achieving 20-48% speedup.

FP16 serves as the ideal __accumulator for Float6 operations__, accurately representing ~20-50 products of E2M3FN or E3M2FN pairs before overflow becomes a concern. For __cosine similarity__, FP16's precision advantage shows clearly: computing `1 - ab/(‖a‖‖b‖)` with BF16 produces ~10⁻⁶ error, while FP16 achieves ~10⁻⁵ error—closer to F32's ~10⁻⁷. The __reciprocal square root__ optimization uses either `VRSQRTPS` (x86) or `FRSQRTE` (ARM) for initial approximation, then refines with Newton-Raphson: one iteration on x86 (hardware gives 12-bit accuracy), two iterations on ARM (hardware gives only 8-bit accuracy).

```c
// ARM NEON: FMLAL widening multiply-accumulate (4cy latency)
float32x4_t acc = vfmlalq_low_f16(acc, a_f16x8, b_f16x8);   // Elements 0-3
acc = vfmlalq_high_f16(acc, a_f16x8, b_f16x8);  // Elements 4-7

// vs older approach (7cy latency total)
float32x4_t a_f32 = vcvt_f32_f16(vget_low_f16(a_f16x8));  // 3cy
acc = vfmaq_f32(acc, a_f32, b_f32);  // 4cy
```

### Float8

8-bit floating point types in NumKong follow the OCP (Open Compute Project) standard and come in two flavors designed for different use cases. __E4M3FN__ (1 sign + 4 exponent + 3 mantissa bits, range ±448) has no infinities—using all-ones exponent for NaN—making it preferred for __training__ where precision near zero matters more than handling overflow. __E5M2FN__ (1 sign + 5 exponent + 2 mantissa bits, range ±57344) supports infinities and provides wider dynamic range, making it better for __inference__ where weights can take extreme values but precision requirements are relaxed.

The key to NumKong's Float8 performance is the __upcast-then-compute strategy__: on x86 Genoa/Sapphire Rapids, E4M3/E5M2 values upcast to BFloat16 via lookup tables, then use native __DPBF16PS__ instructions for 2-per-lane dot products accumulating to F32. On Arm Graviton 3+ with ARMv8.6-A, the same BF16 upcast happens via NEON table lookups, then __BFDOT__ instructions complete the computation. Older platforms without DPBF16/BFDOT convert F8→F32 directly and use standard FMA, sacrificing 20-30% throughput but maintaining compatibility.

Float8 types are critical for __LLM inference__ where memory bandwidth dominates: storing 405B parameters in FP8 (405 GB) instead of BF16 (810 GB) halves DRAM traffic, and on-chip F8→BF16 conversion happens faster than the memory fetch would complete anyway. NVIDIA H100 and AMD MI300X both expose native FP8 tensor cores, while NumKong's CPU implementation serves edge deployment and mixed-CPU-GPU pipelines. For __GEMM operations__, the packed matrix format stores F8 values contiguously, and microkernels unpack 32-64 elements per iteration for SIMD processing.

| Format | Range  | Mantissa | Use Case             | Hardware Support   |
| ------ | ------ | -------- | -------------------- | ------------------ |
| E4M3FN | ±448   | 3 bits   | Training (gradients) | H100, MI300, Genoa |
| E5M2FN | ±57344 | 2 bits   | Inference (weights)  | H100, MI300, Genoa |



### Float6

6-bit floating point types are some of the smallest representations that have been shown to be accurate enough for LLM inference and potentially training.
They come in 2 flavors: E3M2FN and E2M3FN.
Despite sub-byte size, to simplify addressing, the values are typically padded to 8 bits.
A smaller range compare to Float8 variants allows using smaller accumulators in dot products, such as the canonical IEEE Float16.
Float16 can accurately represent:

- sum of around 50 products of E3M2FN pairs,
- sum of around 20 products of E2M3FN pairs.

On Intel Sapphire Rapids it allows leveraging `_mm512_fmadd_ph` FMAs as opposed to `_mm512_dpbf16_ps`.
The former, however, has lower throughput, requires periodic Float32 upcasts, and is supported on fewer machines.
Moreover, the cost of upcasting from Float6 to Float16 and to BFloat16 is identical - be it via table lookups or bit hacks.
On Arm, however, NEON FHM extensions bring widening `FMLAL` dot-products for Float16 - both faster and more widely available than `BFDOT` instructions for BFloat16.

### Int8

Both signed and unsigned 8-bit integers are supported extensively across dot products, angular distances, set operations, and reductions. The most sophisticated optimization is __algebraic transformation for symmetric operations__: on platforms like Ice Lake with AVX-512 VNNI, the native instruction __DPBUSD__ is asymmetric (unsigned × signed → signed), yet NumKong exploits it for both i8×i8 and u8×u8 via clever bit manipulation.

For __signed i8×i8__, we convert the signed operand to unsigned via XOR with `0x80`, compute `DPBUSD(a⊕0x80, b) = (a+128)×b`, then subtract a correction term `128×sum(b)` to recover the true result. This achieves __1.36x speedup__ over the naive approach because DPBUSD runs on port 0 while the correction sum accumulates on port 5 in parallel, eliminating the cvtepi8_epi16 bottleneck that serializes on port 5. For __unsigned u8×u8__, we XOR the second operand to make it signed, compute `DPBUSD(a, b⊕0x80) = a×(b-128)`, then add correction `128×sum(a)` using the fast SAD instruction, achieving __1.92x speedup__ by eliminating four unpack operations.

On older platforms without VNNI (Haswell, Skylake), NumKong falls back to __VPMADDUBSW__ (u8×i8→i16) followed by __VPMADDWD__ (i16×1→i32), processing 16 elements per iteration instead of 32-64, but still maintaining competitive performance through careful port utilization. Sierra Forest's AVX-VNNI variant further improves on Ice Lake by 30-40% via better instruction scheduling and lower latency. Long arithmetic circuits use __32-bit accumulators__ to prevent overflow: even with max values (±127 or 0-255), 16 million products fit safely in i32 range.

```c
// Asymmetric transform for i8×i8 using DPBUSD (unsigned×signed)
a_unsigned = a XOR 0x80;           // Convert signed→unsigned
result = DPBUSD(a_unsigned, b);    // Computes (a+128)×b
correction = 128 * sum(b);         // Parallel on different port
final = result - correction;       // True a×b value (1.36x faster!)
```

### Int4

Both signed and unsigned 4-bit nibble-sized integers are supported for __fast AI inference__ workloads, following the same quantization strategies as ONNX and TensorFlow Lite. The challenge with Int4 is __sub-byte addressing__: two nibbles pack into each byte, requiring careful extraction and sign extension. NumKong uses bitmasks to isolate low nibbles `(byte & 0x0F)` and high nibbles `(byte >> 4)`, then for signed i4, applies the transformation `(nibble ⊕ 8) - 8` to map the unsigned range [0,15] to signed range [-8,7].

For __dot products__, NumKong maintains __separate accumulators__ for low and high nibbles, processing them independently then summing at finalization. This avoids expensive nibble-interleaving operations and allows SIMD lanes to work in parallel. The memory savings are substantial: a 7B parameter LLM in Int4 requires 3.5 GB (vs 7 GB in Int8, 14 GB in FP16), fitting entirely in consumer GPU memory and enabling edge deployment. For __GEMM operations__, the packed matrix format stores nibbles contiguously, and unpacking happens during the microkernel's register-blocking phase, amortizing the extraction cost across multiple dot products.

```c
// Nibble extraction and sign extension for i4
low_nibble = byte & 0x0F;             // Mask lower 4 bits
high_nibble = (byte >> 4) & 0x0F;     // Shift and mask upper 4 bits
i4_value = (low_nibble ^ 8) - 8;      // Maps [0,15] to [-8,7]
```


## Algorithms & Design Decisions 📚

In general there are a few principles that NumKong follows:

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

In NumPy terms, NumKong implementation is similar to:

```python
import numpy as np

def cos_numpy(a: np.ndarray, b: np.ndarray) -> float:
    ab, a2, b2 = np.dot(a, b), np.dot(a, a), np.dot(b, b) # Fused in NumKong
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

To overcome the limitations of the `rsqrt` instruction, NumKong uses the Newton-Raphson iteration to refine the initial estimate for high-precision floating-point numbers.
It can be defined as:

```math
x_{n+1} = x_n \cdot (3 - x_n \cdot x_n) / 2
```

On 1536-dimensional inputs on Intel Sapphire Rapids CPU a single such iteration can result in a 2-3 orders of magnitude relative error reduction:

| Datatype   |         NumPy Error | NumKong w/out Iteration |             NumKong |
| :--------- | ------------------: | ----------------------: | ------------------: |
| `bfloat16` | 1.89e-08 ± 1.59e-08 |     3.07e-07 ± 3.09e-07 | 3.53e-09 ± 2.70e-09 |
| `float16`  | 1.67e-02 ± 1.44e-02 |     2.68e-05 ± 1.95e-05 | 2.02e-05 ± 1.39e-05 |
| `float32`  | 2.21e-08 ± 1.65e-08 |     3.47e-07 ± 3.49e-07 | 3.77e-09 ± 2.84e-09 |
| `float64`  | 0.00e+00 ± 0.00e+00 |     3.80e-07 ± 4.50e-07 | 1.35e-11 ± 1.85e-11 |

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

NumKong doesn't produce intermediate vector results, like `a @ M @ b`, but computes the bilinear form directly.

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
NumKong defines `dot` and `vdot` kernels as:

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

The Fused-Multiply-Add (FMA) operation is a single operation that combines element-wise multiplication and addition with different scaling factors.
The Weighted Sum is its simplified variant without element-wise multiplication.

```math
\text{FMA}_i(A, B, C, \alpha, \beta) = \alpha \cdot A_i \cdot B_i + \beta \cdot C_i
```

```math
\text{WSum}_i(A, B, \alpha, \beta) = \alpha \cdot A_i + \beta \cdot B_i
```

In NumPy terms, the implementation may look like:

```py
import numpy as np
def wsum(A: np.ndarray, B: np.ndarray, /, Alpha: float, Beta: float) -> np.ndarray:
    assert A.dtype == B.dtype, "Input types must match and affect the output style"
    return (Alpha * A + Beta * B).astype(A.dtype)
def fma(A: np.ndarray, B: np.ndarray, C: np.ndarray, /, Alpha: float, Beta: float) -> np.ndarray:
    assert A.dtype == B.dtype and A.dtype == C.dtype, "Input types must match and affect the output style"
    return (Alpha * A * B + Beta * C).astype(A.dtype)
```

The tricky part is implementing those operations in mixed precision, where the scaling factors are of different precision than the input and output vectors.
NumKong uses double-precision floating-point scaling factors for any input and output precision, including `i8` and `u8` integers and `f16` and `bf16` floats.
Depending on the generation of the CPU, given native support for `f16` addition and multiplication, the `f16` temporaries are used for `i8` and `u8` multiplication, scaling, and addition.
For `bf16`, native support is generally limited to dot-products with subsequent partial accumulation, which is not enough for the FMA and WSum operations, so `f32` is used as a temporary.

### Auto-Vectorization & Loop Unrolling

On the Intel Sapphire Rapids platform, NumKong was benchmarked against auto-vectorized code using GCC 12.
GCC handles single-precision `float` but might not be the best choice for `int8` and `_Float16` arrays, which have been part of the C language since 2011.

| Kind                      | GCC 12 `f32` | GCC 12 `f16` | NumKong `f16` | `f16` improvement |
| :------------------------ | -----------: | -----------: | ------------: | ----------------: |
| Inner Product             |    3,810 K/s |      192 K/s |     5,990 K/s |          __31 x__ |
| Cosine Distance           |    3,280 K/s |      336 K/s |     6,880 K/s |          __20 x__ |
| Euclidean Distance ²      |    4,620 K/s |      147 K/s |     5,320 K/s |          __36 x__ |
| Jensen-Shannon Divergence |    1,180 K/s |       18 K/s |     2,140 K/s |         __118 x__ |

### Dynamic Dispatch

Most popular software is precompiled and distributed with fairly conservative CPU optimizations, to ensure compatibility with older hardware.
Database Management platforms, like ClickHouse, and Web Browsers, like Google Chrome,need to run on billions of devices, and they can't afford to be picky about the CPU features.
For such users NumKong provides a dynamic dispatch mechanism, which selects the most advanced micro-kernel for the current CPU at runtime.

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

You can compile NumKong on an old CPU, like Intel Haswell, and run it on a new one, like AMD Genoa, and it will automatically use the most advanced instructions available.
Reverse is also true, you can compile on a new CPU and run on an old one, and it will automatically fall back to the most basic instructions.
Moreover, the very first time you prove for CPU capabilities with `nk_capabilities()`, it initializes the dynamic dispatch mechanism, and all subsequent calls will be faster and won't face race conditions in multi-threaded environments.

## Target Specific Backends

NumKong exposes all kernels for all backends, and you can select the most advanced one for the current CPU without relying on built-in dispatch mechanisms.
That's handy for testing and benchmarking, but also in case you want to dispatch a very specific kernel for a very specific CPU, bypassing NumKong assignment logic.
All of the function names follow the same pattern: `nk_{function}_{type}_{backend}`.

- The backend can be `serial`, `haswell`, `skylake`, `ice`, `genoa`, `sapphire`, `turin`, `neon`, or `sve`.
- The type can be `f64`, `f32`, `f16`, `bf16`, `f64c`, `f32c`, `f16c`, `bf16c`, `i8`, or `u1`.
- The function can be `dot`, `vdot`, `cos`, `sqeuclidean`, `hamming`, `jaccard`, `kld`, `jsd`, or `intersect`.

To avoid hard-coding the backend, you can use the `nk_kernel_punned_t` to pun the function pointer and the `nk_capabilities` function to get the available backends at runtime.
To match all the function names, consider a RegEx:

```regex
NK_PUBLIC void nk_\w+_\w+_\w+\(
```

On Linux, you can use the following command to list all unique functions:

```sh
$ grep -oP 'NK_PUBLIC void nk_\w+_\w+_\w+\(' include/numkong/*.h | sort | uniq
> include/numkong/set.h:NK_PUBLIC void nk_hamming_u1_haswell(
> include/numkong/set.h:NK_PUBLIC void nk_hamming_u1_icelake(
> include/numkong/set.h:NK_PUBLIC void nk_hamming_u1_neon(
> include/numkong/set.h:NK_PUBLIC void nk_hamming_u1_serial(
> include/numkong/set.h:NK_PUBLIC void nk_hamming_u1_sve(
> include/numkong/set.h:NK_PUBLIC void nk_jaccard_u1_haswell(
> include/numkong/set.h:NK_PUBLIC void nk_jaccard_u1_icelake(
> include/numkong/set.h:NK_PUBLIC void nk_jaccard_u1_neon(
> include/numkong/set.h:NK_PUBLIC void nk_jaccard_u1_serial(
> include/numkong/set.h:NK_PUBLIC void nk_jaccard_u1_sve(
```

## License

Feel free to use the project under Apache 2.0 or the Three-clause BSD license at your preference.

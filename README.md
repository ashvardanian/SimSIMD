# NumKong: Mixed Precision for All

NumKong (previously SimSIMD) is a latency-oriented mixed-precision BLAS-like numerics & linear-algebra library designed for portability and multimodal retrieval tasks.
It's one of the largest collections of SIMD kernels online, totalling over 1500 endpoints powering the [Unum](https://www.unum.cloud/)'s open-source [USearch](https://github.com/unum-cloud/usearch) search engine and the DBMS & AI products leveraging it.

![NumKong banner](https://github.com/ashvardanian/ashvardanian/blob/master/repositories/NumKong-v7.jpg?raw=true)

NumKong covers everything — from exotic GPU-only 6-bit floating-point types in LLMs to 64-bit complex floating-point numbers in Scientific Computing — with rigorous testing of numerical stability bounds for various distributions of input data:

<div align="center">
<pre><code>
┌───────────────────────────────┬──────────────────┬─────────────────────────────┬──────────────────┐
│          Operations           │    Datatypes     │          Backends           │    Ecosystems    │
├───────────────────────────────┼──────────────────┼─────────────────────────────┼──────────────────┤
│ Vector-Vector                 │ <a href="#">Bits &amp; Ints</a>      │ <a href="#">x86</a>                         │ Core             │
│   <a href="#">dot</a> · <a href="#">angular</a> · <a href="#">euclidean</a>   │   u1 · u4 · u8   │   Haswell · Alder Lake      │   <a href="#">C 99</a>           │
│   hamming · kld · jsd · …     │   i4 · i8        │   Sierra Forest · Skylake   │                  │
│                               │                  │   Ice Lake · Genoa · Turin  │ Primary          │
│ <a href="#">Matrix-Matrix</a>                 │ <a href="#">Mini-floats</a>      │   Sapphire Rapids ·         │   <a href="#">C++ 23</a>         │
│   dots_packed · dots_symmetric│   e2m3 · e3m2    │   Granite Rapids            │   <a href="#">Python 3</a>       │
│   euclideans_packed · …       │   e4m3 · e5m2    │                             │   <a href="#">Rust</a>           │
│                               │                  │ <a href="#">Arm</a>                         │                  │
│ Quadratic                     │ <a href="#">Halfs &amp; Classics</a> │   NEON · NEONHalf · NEONFhm │ Additional       │
│   <a href="#">bilinear</a> · mahalanobis      │   f16 · bf16     │   NEONBFDot · NEONSDot      │   <a href="#">Swift</a> · <a href="#">JS</a>     │
│                               │   f32 · f64      │   SVE · SVEHalf · SVEBfDot  │   <a href="#">Go</a>             │
│ <a href="#">Geospatial</a> &amp; <a href="#">Geometric</a>        │                  │   SVESDot · SVE2            │                  │
│   haversine · vincenty        │ <a href="#">Complex</a>:         │   SME · SMEF64 · SMEBI32   │ <a href="#">Tools</a>            │
│   rmsd · kabsch · umeyama · … │   f16c · bf16c   │                             │   <a href="#">Tests</a>          │
│                               │   f32c · f64c    │ <a href="#">RISC-V</a>                      │   <a href="#">Benchmarks</a>     │
│ Bespoke                       │                  │   RVV · RVVHalf             │   <a href="https://github.com/ashvardanian/NumWars">NumWars</a>        │
│   <a href="#">fma</a> · blend · <a href="#">sin</a> · <a href="#">cast</a>    │                  │   RVVBf16 · RVVBB           │                  │
│   <a href="#">reduce_moments</a> · <a href="#">sparse_dot</a> │                  │                             │                  │
│   <a href="#">maxsim</a> · intersect · …     │                  │ <a href="#">WASM</a>                        │                  │
│                               │                  │   V128Relaxed               │                  │
└───────────────────────────────┴──────────────────┴─────────────────────────────┴──────────────────┘
</code></pre>
</div>

Not every combination of Datatype x Operation x Backend is implemented — only the ones that unlock interesting new opportunities.
The `icelake` capability level doesn't get a `dot_bf16` variant, for example, and falls through to `dot_bf16_skylake`.
Every operation has a `serial` fallback, but even the 6-bit or 8-bit floats supported by zero CPUs today won't be evaluated sequentially — they use lookup tables and bit-twiddling hacks tuned per backend.

The library makes many opinionated (and documented) design decisions around saturation, rounding, and numerical stability.
Because of that, NumKong's `f32` and `f64` GEMM-like kernels can be 5x slower than the go-to BLAS, but 50x more numerically accurate.
It also respects hardware popularity — on Arm, `f16` is the most common path for mini-floats; on x86, `bf16` is more widely available.
In most SDKs, you compile and ship once — the best kernel for each ISA variant is selected at runtime.
In C and C++, all kernels can be accessed directly and inlined — avoiding every layer of indirection.
The rest of this document unpacks the functionality and the logic behind the design decisions, starting with a few performance highlights.
For language-specific "Quick Start" guides or dedicated testing and benchmarking reports, please refer to table above.

## Performance & Numerical Stability Together

It's very hard to pick a baseline for performance comparison for NumKong, because pretty much no software uses mixed precision.
Both BLAS libraries and PyTorch-like frameworks don't have the same breadth of numeric types and rarely leverage multi-precision numerics.
Of smaller types, NumPy & SciPy can deal with `i8` and `f16`... overflowing on almost every operation, outputting dot products for both in the same `i8` and `f16`.
Unlike them, NumKong will return `i32` and `f32` values for those input types.
Even assuming mixed precision is typically much more expensive, here's what the latency looks like for OpenAI-style 1536-dimensional vector embeddings:

> Single dot product, M calls/s, higher is better. Apple M2 Pro, Python 3.12.

|                    | NumPy | PyTorch |  JAX | NumKong |
| :----------------- | ----: | ------: | ---: | ------: |
| `f32` dot products |   2.1 |     1.5 |  0.1 |     0.9 |
| `f16` dot products |   0.4 |     1.6 |  0.2 |     1.7 |
| `i8` dot products  |   1.0 |     0.3 |  0.2 |     2.8 |

NumKong's `f32` path is intentionally slower — it uses mixed-precision accumulation (Kahan summation and compensated dot products) for better numerical stability.
The `f16` and `i8` paths are where mixed precision shines: NumKong returns `f32` and `i32` results without overflow, while NumPy truncates to the input type.

But raw call latency isn't the whole picture.
Numerical stability matters just as much — here's what happens when you accumulate 1536 `f16` values:

> Max absolute error vs f64 reference, 1000 trials. Apple M2 Pro.

| Metric              | NumPy (f16 accumulator) | NumKong (f32 accumulator) |
| :------------------ | ----------------------: | ------------------------: |
| Max absolute error  |                  1.0008 |                    0.0011 |
| Mean absolute error |                  0.4987 |                    0.0003 |
| NumKong advantage   |                         |            **878x** lower |

So you might be thinking — this must be the heaviest project of them all?

> Binary size of Python shared libraries (`.so` / `.dylib`).

| Package |   Size |
| :------ | -----: |
| NumKong | 2.3 MB |
| NumPy   | 6.9 MB |
| JAX     | 232 MB |

For a broader comparison across languages and runtimes, see the [NumWars](https://github.com/ashvardanian/NumWars) benchmarking suite.

## Project Organization & Design Decisions 📚

At a high-level, NumKong, like StringZilla, USearch, UCall, and other related projects, has a similar performance & portability mindset, that is hard to combine with individual repositories.
That kind of logic doesn't fit well into `<operation>/README.md` or `<language>/README.md` files, even though those are a much better home for low-level semantics and micro-kernel details.

In general there are a few principles that NumKong follows:

- Avoid loop unrolling and scalar tails.
- Don't manage threads and be compatible with any parallelism models.
- Don't manage memory and be compatible with arbitrary allocators & alignment.
- Don't constrain ourselves to traditional BLAS-like Matrix Multiplication APIs.
- Don't throw exceptions and pass values by pointers.
- Prefer saturated arithmetic and avoid overflows, where needed.
- Cover most modern CPUs with flexible dispatch and wait for them to converge with GPUs.

### Auto-Vectorization & Loop Unrolling

Most "optimized SIMD code" is a 2–4x unrolled data-parallel `for`-loop over `f32` arrays with a serial scalar tail for the last few elements:

```c
float boring_dot_product_f32(float const *a, float const *b, size_t n) {
    __m256 sum0 = _mm256_setzero_ps(), sum1 = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        sum0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), sum0);
        sum1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), sum1);
    }
    float result = _mm256_reduce_add_ps(_mm256_add_ps(sum0, sum1));
    for (; i < n; i++) result += a[i] * b[i]; // serial tail
    return result;
}
```

This kind of unrolling has been historically the most recommended missing "optimization" for NumKong, and it's intentionally avoided.

__Modern CPUs already "unroll" in hardware.__
Out-of-order engines with reorder buffers of 320–630 entries (Zen 4: 320, Golden Cove: 512, Apple Firestorm: ~630) can keep a dozen of loop iterations in-flight simultaneously.
The physical register file is much larger than the ISA-visible architectural registers — Skylake has ~180 physical integer registers behind 16 architectural GPRs, and ~168 physical vector registers behind 32 architectural ZMMs.
The register renaming unit maps the same `zmm0` in iteration N and iteration N+1 to different physical registers, extracting cross-iteration parallelism automatically — exactly the benefit that source-level unrolling was historically supposed to provide.

__Unrolling actively hurts at NumKong's scale.__
Every unrolled copy is a distinct instruction in the binary.
With 1,500+ kernel endpoints across 30+ backends, even 2x unrolling would inflate the `.text` section by megabytes — directly impacting install size for Python wheels, NPM packages, and Rust crates.
Larger loop bodies also increase instruction-cache and micro-op-cache pressure; Agner Fog also recommends:

> _"avoid loop unrolling where possible in order to economize the use of the micro-op cache"_.

Meaning, a loop that spills out of the uop cache falls back to the slower legacy decoder, making the "optimized" version slower than the compact original.
For a header-only library, unrolling also compounds __compilation time__: register allocation is NP-hard (reducible to graph coloring), and unrolling multiplies the number of simultaneously live ranges the allocator must consider, increasing compile time super-linearly across every translation unit that includes the headers.

__Serial tails are a correctness hazard.__
The leftover elements after the last full SIMD chunk run through a scalar loop that silently drops FMA fusion, compensated accumulation, and saturating arithmetic — producing results with different numerical properties than the SIMD body.
NumKong often uses masked loads instead (`_mm512_maskz_loadu_ps` on AVX-512, predicated `svld1_f32` on SVE), processing every element through the same arithmetic path regardless of alignment.
It's not exactly orthogonal to loop-unrolling, but makes a different kernel layout more compatible.

__The real performance gap is elsewhere.__
On Intel Sapphire Rapids, NumKong was benchmarked against auto-vectorized code compiled with GCC 12.
GCC handles single-precision `float` competently, but struggles with `_Float16` and other mixed-precision paths:

| Kind                      | GCC 12 `f32` | GCC 12 `f16` | NumKong `f16` | `f16` improvement |
| :------------------------ | -----------: | -----------: | ------------: | ----------------: |
| Inner Product             |    3,810 K/s |      192 K/s |     5,990 K/s |          __31 x__ |
| Cosine Distance           |    3,280 K/s |      336 K/s |     6,880 K/s |          __20 x__ |
| Euclidean Distance ²      |    4,620 K/s |      147 K/s |     5,320 K/s |          __36 x__ |
| Jensen-Shannon Divergence |    1,180 K/s |       18 K/s |     2,140 K/s |         __118 x__ |

NumKong's `f16` kernels are faster than GCC's `f32` output — not because of unrolling, but because they use F16C conversion instructions, widening FMA pipelines, and compensated accumulation that no compiler will synthesize from a plain `for` loop.
The same story repeats for `bf16`, `e4m3`, `i8`, and `i4`: these types require algorithmic transformations — lookup tables, algebraic domain shifts, asymmetric VNNI tricks — that live beyond the reach of auto-vectorization.

### Parallelism & Multi-Threading

BLAS libraries traditionally manage their own thread pools.
[OpenBLAS](https://github.com/OpenMathLib/OpenBLAS/blob/develop/USAGE.md) spawns threads controlled by `OPENBLAS_NUM_THREADS`, [Intel MKL](https://www.intel.com/content/www/us/en/docs/onemkl/developer-guide-linux/2025-1/techniques-to-set-the-number-of-threads.html) forks its own OpenMP runtime via `MKL_NUM_THREADS`, and [Apple Accelerate](https://developer.apple.com/documentation/accelerate/blas) delegates to GCD.
This works in isolation — but the moment your application adds its own parallelism (joblib, std::thread, Tokio, GCD, OpenMP), you get __thread oversubscription__: MKL spawns 8 threads inside each of your 8 joblib workers, producing 64 threads on 8 cores, thrashing caches and stalling on context switches.
The Python ecosystem has built [entire libraries](https://github.com/joblib/threadpoolctl) just to work around this problem, and [scikit-learn's documentation](https://scikit-learn.org/stable/computing/parallelism.html) devotes a full page to managing the interaction between joblib parallelism and BLAS thread pools.

NumKong takes a different position: __the numerics layer should not own threads__.
Modern hardware makes the "spawn N threads and split evenly" model increasingly untenable:

- __Server-grade CPUs__ have hundreds of cores split across sockets, chiplets, and tiles, resulting in dozens of physical NUMA domains with vastly different memory access latencies.
  A thread pool that ignores NUMA topology will spend more time on remote memory stalls than on arithmetic.
- __Consumer-grade CPUs__ pack heterogeneous Quality-of-Service core types on the same die — Intel P-cores and E-cores run at different frequencies and sometimes support different ISA extensions.
  A naive work-split gives equal chunks to fast and slow cores, and the whole task stalls waiting for the slowest partition.
- __Real-time operating systems__ in robotics and edge AI cannot afford to yield the main thread to a BLAS-managed pool.
  These systems need deterministic latency, not maximum throughput.

Instead, NumKong exposes __row-range parameters__ that let the caller partition work across any threading model.
For GEMM-shaped `dots_packed`, this is straightforward — pass a slice of A's rows and the full packed B to compute the corresponding slice of C.
For SYRK-shaped `dots_symmetric`, explicit `start_row` / `end_row` parameters control which rows of the symmetric output matrix a given thread computes.
The GIL is released around every kernel call, making NumKong compatible with `concurrent.futures`, `multiprocessing`, or any other parallelism model:

```python
import concurrent.futures, numkong as nk, numpy as np

vectors, num_threads = np.random.randn(1000, 768).astype(np.float32), 4
output = nk.zeros((1000, 1000), dtype="float32")

def compute_slice(t):
    start = t * (len(vectors) // num_threads)
    end = start + len(vectors) // num_threads if t < num_threads - 1 else len(vectors)
    nk.dots_symmetric(vectors, out=output, start_row=start, end_row=end)

with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as pool:
    list(pool.map(compute_slice, range(num_threads)))
```

For users who want a ready-made low-latency thread pool without the oversubscription baggage of OpenMP, we recommend [Fork Union](https://github.com/ashvardanian/ForkUnion) — a minimalist fork-join library for C, C++, and Rust that avoids mutexes, CAS atomics, and dynamic allocations on the critical path, with optional NUMA pinning on Linux.

### Memory Allocation & Management

BLAS libraries typically allocate internal buffers during GEMM — [OpenBLAS](https://github.com/OpenMathLib/OpenBLAS) packs matrices into L2/L3-sized panels via per-thread buffer pools backed by `mmap` or `shmget`.
This hidden allocation has caused real problems: [14 lock/unlock pairs per small GEMM call](https://github.com/xianyi/OpenBLAS/issues/478) throttling 12-thread scaling to 2x, [silently incorrect results](https://github.com/xianyi/OpenBLAS/issues/1844) from thread-unsafe allocation in `np.dot`, and [deadlocks after `fork()`](https://github.com/numpy/numpy/issues/30092) due to mutex state not being reset in child processes.
The [BLASFEO](https://github.com/giaf/blasfeo) library was created specifically for embedded model-predictive control where `malloc` during computation is unacceptable.

NumKong __never allocates memory__.
Following the same philosophy as [Intel MKL's packed GEMM API](https://www.intel.com/content/www/us/en/developer/articles/technical/introducing-the-new-packed-apis-for-gemm.html) (`cblas_sgemm_pack_get_size` → `cblas_sgemm_pack` → `cblas_sgemm_compute`), NumKong exposes a three-phase interface — `nk_dots_packed_size` → `nk_dots_pack` → `nk_dots_packed` — where the caller owns the buffer and NumKong only fills it.

The reason GEMM libraries repack matrices at all is that every hardware target has a different preferred layout — Intel AMX expects B in a [VNNI-interleaved](https://www.intel.com/content/www/us/en/developer/articles/code-sample/advanced-matrix-extensions-intrinsics-functions.html) tile format (pairs of BF16 values packed into DWORDs across the K dimension), while Arm SME wants column vectors for its [FMOPA outer-product](https://developer.arm.com/documentation/ddi0602/latest/SME-Instructions) instructions.
Since GEMM is $O(N^3)$ and repacking is $O(N^2)$, the cost is asymptotically free — but the allocation and locking overhead is not.

NumKong's `nk_dots_pack` performs five transformations beyond simple reordering:

- __Type pre-conversion__ — mini-floats (e4m3, bf16, etc.) are upcast to the compute type once during packing, not on every GEMM call.
  This amortizes the conversion cost across all rows of A that will be multiplied against the packed B.
- __SIMD depth padding__ — rows are zero-padded to the SIMD vector width (16 for AVX-512 f32, 64 for AVX-512 i8), allowing inner loops to load without boundary checks.
- __Power-of-2 stride breaking__ — when the padded row stride is a power of 2, one extra SIMD step of padding is added.
  Power-of-2 strides cause [cache set aliasing](https://en.algorithmica.org/hpc/cpu-cache/associativity/) where consecutive rows map to the same cache sets, effectively shrinking usable L1/L2 capacity — stride-256 traversals can be [~10x slower](https://en.algorithmica.org/hpc/cpu-cache/associativity/) than stride-257.
- __Per-column norm precomputation__ — squared norms ($\|b_j\|^2$) are computed and stored alongside the packed data, so distance kernels (`angulars_packed`, `euclideans_packed`) can reuse them without a separate pass.
- __ISA-specific tile layout__ — AMX packing interleaves BF16 pairs into 16×32 tiles matching `TDPBF16PS` expectations; SME packing arranges vectors at SVE granularity for `FMOPA` outer products; generic backends use simple column-major with depth padding.

```python
import numkong as nk, numpy as np

right_matrix = np.random.randn(1000, 768).astype(np.float16)
right_packed = nk.dots_pack(right_matrix, dtype="float16")                        # pack once
for query_batch in stream: results = nk.dots_packed(query_batch, right_packed)    # reuse many times
```

### The Evolution of Matrix Multiplication APIs

The classic BLAS GEMM computes $C = \alpha A B + \beta C$ for f32/f64 matrices.
It's a powerful primitive, but the workloads that dominate modern compute — LLM inference, vector search, quantum simulation — expose three ways in which the traditional GEMM interface falls short.

__Frozen weights justify separating packing from computation.__
During LLM inference, a very large share of GEMM calls use a static weight matrix — weights don't change after loading.
This makes offline repacking a one-time cost amortized over the entire serving lifetime: [NVIDIA's TurboMind](https://arxiv.org/pdf/2508.15601) explicitly splits GEMM into offline weight packing (hardware-aware layout conversion) and online mixed-precision computation, and [Intel MKL's packed GEMM API](https://www.intel.com/content/www/us/en/developer/articles/technical/introducing-the-new-packed-apis-for-gemm.html) exposes the same two-phase pattern.
NumKong's `nk_dots_pack` → `nk_dots_packed` follows this philosophy — pack the weight matrix once, reuse it across all queries.

__Mixed precision demands more than an epilogue addition.__
Modern transformer layers operate in a precision sandwich: weights stored in bf16/fp8, [GEMM accumulated in f32](https://pytorch.org/blog/what-every-user-should-know-about-mixed-precision-training-in-pytorch/), output downcast back to bf16 for the next layer.
Between GEMM calls, [LayerNorm or RMSNorm](https://arxiv.org/html/2409.12951v2) re-normalizes hidden states, so the next layer is often much closer to an angular or normalized similarity computation than to a plain raw dot product.
[nGPT](https://arxiv.org/html/2410.01131v1) takes this to its logical conclusion: all vectors live on the unit hypersphere, and every matrix-vector product is a pure angular distance.
This means many "GEMM" workloads in production are semantically closer to many-to-many angular distance computation — which is exactly what NumKong's `angulars_packed` and `euclideans_packed` kernels compute directly, fusing norm handling and type conversion into a single pass.

__The GEMM-for-distances trick has real costs.__
A common shortcut in vector search is to decompose pairwise Euclidean distance as $\|a - b\|^2 = \|a\|^2 + \|b\|^2 - 2 \langle a, b \rangle$, precompute norms, and call `sgemm` for the inner-product matrix.
Both [FAISS](https://github.com/facebookresearch/faiss/wiki/Implementation-notes) and [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.euclidean_distances.html) use this approach — and both document its limitations.
Scikit-learn's docs warn of _"catastrophic cancellation"_ in the subtraction; this has caused [real bugs](https://github.com/scikit-learn/scikit-learn/issues/9354) with ~37% error on near-identical float32 vectors.
The $O(N^2)$ postprocessing pass (adding norms, square roots, divisions) is not free either — [NVIDIA's RAFT](https://github.com/rapidsai/raft/pull/339) measured a __20–25% speedup__ from fusing it into the GEMM epilogue.
Even [FAISS switches to direct SIMD](https://github.com/facebookresearch/faiss/wiki/Implementation-notes) when the query count drops below 20.
And the standard BLAS interface was never designed for sub-byte types — [no vendor supports int4](https://research.colfax-intl.com/cutlass-tutorial-sub-byte-gemm-on-nvidia-blackwell-gpus/), and sub-byte types cannot even be strided without bit-level repacking.

__Some operations need more than GEMM + postprocessing.__
NumKong implements several GEMM-shaped operations where the "epilogue" is too complex for a simple addition:

- __Bilinear forms__ ($a^T C b$) in quantum computing compute a [scalar expectation value](https://phys.libretexts.org/Bookshelves/Quantum_Mechanics/Advanced_Quantum_Mechanics_(Kok)/10:_Pauli_Spin_Matrices/10.2:_Expectation_Values) — the naive approach materializes an $N$-dimensional intermediate vector $Cb$, but NumKong's `nk_bilinear` streams through rows of $C$ with nested compensated dot products, never allocating beyond registers.
  For complex-valued quantum states, where the intermediate would be a 2N-element complex vector, the savings double.
- __MaxSim scoring__ for [ColBERT-style late-interaction retrieval](https://github.com/stanford-futuredata/ColBERT) computes $\sum_i \min_j \text{angular}(q_i, d_j)$ — a sum-of-min-distances across token pairs.
  A GEMM would produce the full $M \times N$ similarity matrix, but NumKong's `nk_maxsim_packed` fuses a coarse i8-quantized screening with full-precision angular refinement on winning pairs only, __packing both query and document matrices__ to enable all 4 SME tiles as accumulators (+33% throughput vs `dots_packed`).
  [PLAID](https://ar5iv.labs.arxiv.org/html/2205.09707) and [maxsim-cpu](https://www.mixedbread.com/blog/maxsim-cpu) have independently shown that dedicated MaxSim kernels outperform the GEMM decomposition by 5–10x.

NumKong treats these as first-class operations — `dots_packed`, `euclideans_packed`, `angulars_packed`, `nk_bilinear`, `nk_maxsim_packed` — rather than decomposing everything into GEMM + postprocessing.
More operations are being considered as well.

### Saturating, Rounding, Stable Arithmetic, Square Roots, and Picking FP6 Over FP8

Floating-point arithmetic on computers [is not associative](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html): $(a + b) + c \neq a + (b + c)$ in general, and the standard advice — "upcast to wider types" — often isn't enough while universally sacrificing performance.
NumKong makes opinionated, operation-specific decisions about where to spend precision and where to economize, rather than applying one IEEE rule uniformly.

__Saturation depends on the operation.__
A reduction over a 4 GB array of `i8` values contains ~4 billion elements — but [i32 wrapping overflow](https://cedardb.com/blog/overflow_handling/) occurs after just ~17 million i8 summands ($127 \times 16.9\text{M} > 2^{31}$).
Reductions in NumKong use saturating arithmetic because the input can be arbitrarily long.
Matrix multiplications don't need saturation because GEMM depth rarely exceeds tens of thousands — well within i32 range.
x86 provides no saturating 32-bit SIMD add ([only byte/word variants](https://www.felixcloutier.com/x86/paddb:paddw:paddd:paddq)), so NumKong implements saturation via overflow detection with XOR-based unsigned comparison on platforms that lack native support.

__Square roots & special math ops are platform-specific.__
Angular distance requires $1/\sqrt{\|a\|^2 \cdot \|b\|^2}$ — but the cost of computing this normalization varies dramatically across hardware.
x86 `VSQRTPS` takes [~12 cycles](https://uops.info/html-lat/SKX/VSQRTPS_XMM_XMM-Measurements.html), followed by `VDIVPS` at ~11 cycles — totalling ~23 cycles for a precise `1/sqrt(x)`.
The `VRSQRT14PS` alternative starts with a [14-bit estimate in ~4 cycles](https://www.intel.com/content/www/us/en/developer/articles/code-sample/reference-implementations-for-ia-approximation-instructions-vrcp14-vrsqrt14-vrcp28-vrsqrt28-vexp2.html), then one Newton-Raphson iteration ($y = y \cdot (1.5 - 0.5 x y^2)$, ~4 more cycles) reaches full f32 precision — a __~3x speedup__.
ARM's `FRSQRTE` provides only [~8 bits](https://github.com/DLTcollab/sse2neon/issues/526), requiring __two__ Newton-Raphson iterations to match.
NumKong selects the iteration count per platform so the final ULP bound is consistent across ISAs, rather than exposing different precision to different users.

__E2M3 and E3M2 can outperform E4M3 and E5M2.__
As described in the [Numeric Types](#e5m2-e4m3-e3m2--e2m3-quarter-precision-inputs) section, 6-bit [MX formats](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) can be scaled to exact integers that fit in `i8`/`i16`, enabling integer `VPDPBUSD` / `SDOT` accumulation instead of the floating-point pipeline.
E4M3 and E5M2 cannot use this path — their wider range overflows practical integer types.
Without the integer path, E5M2 falls back to f32 accumulation where its [2-bit mantissa](https://developer.nvidia.com/blog/floating-point-8-an-introduction-to-efficient-lower-precision-ai-training/) creates a [catastrophic cancellation risk](https://www.ac.uma.es/arith2024/papers/Fused%20FP8%204-Way%20Dot%20Product%20with%20Scaling%20and%20FP32%20Accumulation.pdf) that E2M3's integer path avoids entirely:

|         |  _i_ = 0 | _i_ = 1 |  _i_ = 2 |   _i_ = 3 |  _i_ = 4 |  _i_ = 5 |  _i_ = 6 |  _i_ = 7 |
| ------- | -------: | ------: | -------: | --------: | -------: | -------: | -------: | -------: |
| _aᵢ_    |  0.00122 |   20480 | −0.00122 |       1.5 | −0.00586 |    −3072 |     −640 |  0.00146 |
| _bᵢ_    |      −40 |     320 |    −1280 |  −7.63e⁻⁵ |        0 | 0.000427 |    10240 | −4.58e⁻⁵ |
| _aᵢ·bᵢ_ | −0.04883 | 6553600 |   1.5625 | −0.000114 |        0 |  −1.3125 | −6553600 |      ≈ 0 |

> __Why F32 accumulation fails here.__
> The accurate sum of these 8 products is ≈ 0.201.
> After two `vfmaq_f32` calls, the 4 accumulator lanes hold pairwise products: lanes 1 and 2 carry values around ±6.5 M.
> At that magnitude the F32 ULP is 0.5 — so the small meaningful terms (−0.049, 1.563, −1.313, −0.0001) are all below one ULP and get absorbed during pairwise reduction.
> The large terms then cancel exactly to zero, and the information is gone.
> Final F32 result: __0.0__ instead of __0.201__.

Every such decision — saturation thresholds, Newton-Raphson iteration counts, integer vs floating-point paths — is documented per operation and per type in the [module-specific READMEs](include/numkong/).

### Calling Convention & Error Handling

NumKong never throws exceptions, never sets `errno`, and never calls `setjmp`/`longjmp` — [exceptions bloat call sites with unwind tables](https://monkeywritescode.blogspot.com/p/c-exceptions-under-hood.html) and are invisible to C, Python, Rust, Swift, Go, and JavaScript FFI; `errno` is thread-local state whose [storage model varies across C runtimes](https://en.cppreference.com/w/c/error/errno).
Instead, every function takes inputs as `const` pointers, writes outputs through caller-provided pointers, and returns `void`:

```c
void nk_dot_f32(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *result);
void nk_dot_bf16(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result);
```

Pointers eliminate implicit casts for types with platform-dependent storage.
`nk_f16_t` and `nk_bf16_t` resolve to native `__fp16` / `__bf16` when available but fall back to `unsigned short` otherwise — if passed by value, the compiler would silently apply integer promotion instead of preserving the bit pattern.
Passing by pointer keeps the representation opaque: kernels read raw and convert explicitly when needed, so the same binary works regardless of whether the compiler understands `_Float16`.

The only place that requires error signaling is [dynamic dispatch](#compile-time-and-run-time-dispatch) — looking up the best kernel for the current CPU at runtime.
When no kernel matches, the dispatcher sets the [capabilities mask](c/dispatch.h) to zero and fills the function pointer with [`nk_error_`](c/numkong.c), a stub that writes `0xFF` into the output — `NaN` for floats, `−1` for signed integers, `TYPE_MAX` for unsigned.

### Compile-Time and Run-Time Dispatch

NumKong provides two dispatch mechanisms.
__Compile-time dispatch__ selects the fastest kernel supported by the target platform at build time — thinner binaries, no indirection overhead, but requires knowing your deployment hardware.
__Run-time dispatch__ compiles every supported kernel into the binary and picks the best one on the target machine via `nk_capabilities()` — one pointer indirection per call, but a single binary runs everywhere.
The run-time path is common in DBMS products (ClickHouse), web browsers (Chromium), and other upstream projects that ship to heterogeneous fleets.

All kernel names follow the pattern `nk_{operation}_{type}_{backend}`, and you can use `nk_kernel_punned_t` with `nk_capabilities()` to resolve them at runtime without hard-coding the backend.
The first call to `nk_capabilities()` initializes the dispatch table; all subsequent calls are lock-free.

## Numeric Types

### Float64: Double Precision Inputs

For double-precision numerics, NumKong deviates from most BLAS-like libraries by leveraging __stable compensated summation algorithms__ that track numerical errors separately. On serial paths, we use __Neumaier's algorithm__ (1974), an improvement over Kahan-Babuška that correctly handles cases where added terms are larger than the running sum, achieving O(1) error growth instead of O(n).
On SIMD paths with FMA support, we implement the __Dot2 algorithm__ (Ogita-Rump-Oishi, 2005) which maintains separate error compensators for both multiplication and accumulation, capturing rounding errors via `TwoProd` and `TwoSum` operations.

Beyond dot products, Float64 enables __Mahalanobis distance__ computation for statistical analysis and __bilinear forms__ for quantum computing, both using the same compensated summation to preserve precision through long arithmetic circuits.
For probability distributions, __Kullback-Leibler__ and __Jensen-Shannon divergences__ benefit from F64's 52-bit mantissa when computing `log(p/q)` ratios that approach machine epsilon.
The precision overhead is worthwhile: on 1024³ GEMM operations, NumKong's compensated F64 achieves 10-50x smaller ULP errors than Intel MKL despite running at 3x lower throughput, making it ideal for scientific computing where numerical stability matters more than raw speed.

```c
// Dot2 TwoProd: Capture multiplication rounding error
h = a * b;
r = fma(a, b, -h);  // Extracts rounding error

// Dot2 TwoSum: Capture addition rounding error
t = sum + product;
e = (sum - t) + product;  // Compensator term
```

### Float32: Single Precision Inputs

For traditional single-precision inputs, NumKong uses __double-precision FMA__ to reduce numerical error in long arithmetic circuits.
The SIMD implementations load 4 F32 values, upcast to F64 for full-precision multiplication and accumulation, then downcast only during finalization.
This avoids catastrophic cancellation at minimal cost since modern CPUs have dedicated F64 vector units operating at nearly the same throughput as F32.

For __complex dot products__, NumKong employs a clever optimization that doubles throughput: instead of using separate FMA and FMS (fused multiply-subtract) instructions for the real component, we defer sign flips until after the accumulation loop completes, applying a single bitwise XOR to flip sign bits.
This eliminates execution port contention—allowing dual FMA units to run at full capacity—and improves throughput from 2.5 GB/s to 5 GB/s on Haswell.
The transformation is simple: compute `a_r × b_r + a_i × b_i` (treating negatives as positive), then XOR with `0x80000000` to flip every second element's sign bit after the loop.

NumKong also handles __curved space metrics__ like the Mahalanobis distance `√((a-b)ᵀ C (a-b))` where C is a metric tensor, applying the same compensated accumulation strategies to preserve precision when multiplying small differences by large inverse covariance matrices.
For __probability distributions__, we compute KL divergence using log2 decomposition via `VGETEXP` (extract exponent) and polynomial approximations for the mantissa, avoiding expensive transcendental function calls.

```c
// Complex multiply optimization: XOR sign flip AFTER loop
for (...) {
    sum_real = fma(a, b, sum_real);  // No sign flip in loop
    sum_imag = fma(a, b_swapped, sum_imag);
}
sum_real = xor(sum_real, 0x80000000);  // Single XOR after loop
```

### BFloat16 & Float16: Half Precision Inputs

Brain Float is not an IEEE 754 standard type, but it has become the __universal recommendation__ for AI-related applications due to its unique properties: on old CPUs, upcasting BF16 to F32 requires just an unpack and left-shift by 16 bits (essentially free), while on new CPUs, both Arm and x86 provide widening mixed-precision dot products via __DPBF16PS__ (AVX-512 on Genoa/Sapphire Rapids) and __BFDOT__ (NEON on ARMv8.6-A Graviton 3+).
These instructions compute two BF16 products per lane, accumulating directly into F32 registers, achieving 2-4x higher throughput than separate conversion and multiply-add.

The format itself prioritizes __dynamic range over precision__: BF16 shares Float32's 8-bit exponent but truncates the mantissa to 7 bits, allowing it to represent the same range (±3.4×10³⁸) with coarser granularity.
This makes it perfect for neural network weights and activations where large dynamic range matters more than precision near zero.
NumKong's Float8 types (E4M3/E5M2) upcast to BF16 before using DPBF16PS, effectively creating a three-tier precision hierarchy: F8 for storage, BF16 for compute, F32 for accumulation.

Beyond dot products, BF16 excels at __cosine similarity__ where division by magnitudes benefits from the wide exponent range, and __element-wise operations__ like `α×a + β` where the BF16→F32 upcast happens once per element rather than repeatedly.
For __L2 distance__, the formula `√(‖a‖² + ‖b‖² - 2ab)` accumulates into F32 then applies Newton-Raphson reciprocal sqrt, achieving 9x speedup over NumPy while maintaining better numerical accuracy.

| Platform         | BF16 Instruction | Latency | Throughput | Elements/Cycle |
| ---------------- | ---------------- | ------- | ---------- | -------------- |
| Genoa AVX-512    | DPBF16PS         | 6 cy    | 2/cy       | 32 BF16/cy     |
| Graviton 3 NEON  | BFDOT            | 4 cy    | 2/cy       | 8 BF16/cy      |
| Haswell (compat) | Upcast→FMA       | 8 cy    | 1/cy       | 8 BF16/cy      |

IEEE 754 half-precision floating point (FP16) is widely supported across modern hardware with 1 sign bit, 5 exponent bits (bias=15), and 10 mantissa bits, giving a range of ±65504.
Unlike BFloat16 which prioritizes exponent range, FP16 prioritizes __precision over range__ (10 vs 7 mantissa bits), making it better suited for values near zero, gradients during training, and scientific computing where relative error matters more than dynamic range.

On x86, older CPUs use __F16C extensions__ (Ivy Bridge+) providing `VCVTPH2PS` for fast F16→F32 conversion, then standard FMA for computation.
Newer CPUs (Sapphire Rapids+) add native __AVX-512-FP16__ with dedicated F16 arithmetic, though NumKong still widens to F32 for accumulation to preserve precision.
On Arm, the story is richer: basic ARMv8.2-A provides `FCVTL` (3-cycle conversion) followed by `FMLA` (4-cycle FMA), but ARMv8.4-A adds __FMLAL/FMLAL2__ instructions for fused F16→F32 widening multiply-accumulate, reducing the total latency from 7 cycles to 4 cycles and achieving 20-48% speedup.

FP16 serves as the ideal __accumulator for Float6 operations__, accurately representing ~20-50 products of E2M3FN or E3M2FN pairs before overflow becomes a concern.
For __cosine similarity__, FP16's precision advantage shows clearly: computing `1 - ab/(‖a‖‖b‖)` with BF16 produces ~10⁻⁶ error, while FP16 achieves ~10⁻⁵ error—closer to F32's ~10⁻⁷.
The __reciprocal square root__ optimization uses either `VRSQRTPS` (x86) or `FRSQRTE` (ARM) for initial approximation, then refines with Newton-Raphson: one iteration on x86 (hardware gives 12-bit accuracy), two iterations on ARM (hardware gives only 8-bit accuracy).

```c
// ARM NEON: FMLAL widening multiply-accumulate (4cy latency)
float32x4_t acc = vfmlalq_low_f16(acc, a_f16x8, b_f16x8);   // Elements 0-3
acc = vfmlalq_high_f16(acc, a_f16x8, b_f16x8);  // Elements 4-7

// vs older approach (7cy latency total)
float32x4_t a_f32 = vcvt_f32_f16(vget_low_f16(a_f16x8));  // 3cy
acc = vfmaq_f32(acc, a_f32, b_f32);  // 4cy
```

### E5M2, E4M3, E3M2, & E2M3: Quarter Precision Inputs

8-bit floating point types in NumKong follow the OCP (Open Compute Project) standard and come in two flavors designed for different use cases.
__E4M3FN__ (1 sign + 4 exponent + 3 mantissa bits, range ±448) has no infinities—using all-ones exponent for NaN—making it preferred for __training__ where precision near zero matters more than handling overflow.
__E5M2FN__ (1 sign + 5 exponent + 2 mantissa bits, range ±57344) supports infinities and provides wider dynamic range, making it better for __inference__ where weights can take extreme values but precision requirements are relaxed.

The key to NumKong's Float8 performance is the __upcast-then-compute strategy__: on x86 Genoa/Sapphire Rapids, E4M3/E5M2 values upcast to BFloat16 via lookup tables, then use native __DPBF16PS__ instructions for 2-per-lane dot products accumulating to F32.
On Arm Graviton 3+ with ARMv8.6-A, the same BF16 upcast happens via NEON table lookups, then __BFDOT__ instructions complete the computation.
Older platforms without DPBF16/BFDOT convert F8→F32 directly and use standard FMA, sacrificing 20-30% throughput but maintaining compatibility.

Float8 types are critical for __LLM inference__ where memory bandwidth dominates: storing 405B parameters in FP8 (405 GB) instead of BF16 (810 GB) halves DRAM traffic, and on-chip F8→BF16 conversion happens faster than the memory fetch would complete anyway.
NVIDIA H100 and AMD MI300X both expose native FP8 tensor cores, while NumKong's CPU implementation serves edge deployment and mixed-CPU-GPU pipelines.
For __GEMM operations__, the packed matrix format stores F8 values contiguously, and microkernels unpack 32-64 elements per iteration for SIMD processing.

| Format | Range  | Mantissa | Use Case             | Hardware Support   |
| ------ | ------ | -------- | -------------------- | ------------------ |
| E4M3FN | ±448   | 3 bits   | Training (gradients) | H100, MI300, Genoa |
| E5M2FN | ±57344 | 2 bits   | Inference (weights)  | H100, MI300, Genoa |

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

---

E4M3 and E5M2 cannot use this path.
E4M3 scaled by 16 reaches 7,680 — too large for i8, barely fitting i16 with a 128-entry table.
E5M2's range (±57,344) makes the scaled product exceed i32 entirely.
Without the integer path, E5M2 falls back to f32 accumulation — where its [2-bit mantissa (only 4 values per binade)](https://developer.nvidia.com/blog/floating-point-8-an-introduction-to-efficient-lower-precision-ai-training/) creates a [catastrophic cancellation risk](https://www.ac.uma.es/arith2024/papers/Fused%20FP8%204-Way%20Dot%20Product%20with%20Scaling%20and%20FP32%20Accumulation.pdf) that E2M3's integer path avoids completely:

|         |  _i_ = 0 | _i_ = 1 |  _i_ = 2 |   _i_ = 3 |  _i_ = 4 |  _i_ = 5 |  _i_ = 6 |  _i_ = 7 |
| ------- | -------: | ------: | -------: | --------: | -------: | -------: | -------: | -------: |
| _aᵢ_    |  0.00122 |   20480 | −0.00122 |       1.5 | −0.00586 |    −3072 |     −640 |  0.00146 |
| _bᵢ_    |      −40 |     320 |    −1280 |  −7.63e⁻⁵ |        0 | 0.000427 |    10240 | −4.58e⁻⁵ |
| _aᵢ·bᵢ_ | −0.04883 | 6553600 |   1.5625 | −0.000114 |        0 |  −1.3125 | −6553600 |      ≈ 0 |

> __Why F32 accumulation fails here.__
> The accurate sum of these 8 products is ≈ 0.201.
> After two `vfmaq_f32` calls, the 4 accumulator lanes hold pairwise products: lanes 1 and 2 carry values around ±6.5 M.
> At that magnitude the F32 ULP is 0.5 — so the small meaningful terms (−0.049, 1.563, −1.313, −0.0001) are all below one ULP and get absorbed during pairwise reduction.
> The large terms then cancel exactly to zero, and the information is gone.
> Final F32 result: __0.0__ instead of __0.201__.


### Int8

Both signed and unsigned 8-bit integers are supported extensively across dot products, angular distances, set operations, and reductions.
The most sophisticated optimization is __algebraic transformation for symmetric operations__: on platforms like Ice Lake with AVX-512 VNNI, the native instruction __DPBUSD__ is asymmetric (unsigned × signed → signed), yet NumKong exploits it for both i8×i8 and u8×u8 via clever bit manipulation.

For __signed i8×i8__, we convert the signed operand to unsigned via XOR with `0x80`, compute `DPBUSD(a⊕0x80, b) = (a+128)×b`, then subtract a correction term `128×sum(b)` to recover the true result.
This achieves __1.36x speedup__ over the naive approach because DPBUSD runs on port 0 while the correction sum accumulates on port 5 in parallel, eliminating the cvtepi8_epi16 bottleneck that serializes on port 5.
For __unsigned u8×u8__, we XOR the second operand to make it signed, compute `DPBUSD(a, b⊕0x80) = a×(b-128)`, then add correction `128×sum(a)` using the fast SAD instruction, achieving __1.92x speedup__ by eliminating four unpack operations.

On older platforms without VNNI (Haswell, Skylake), NumKong falls back to __VPMADDUBSW__ (u8×i8→i16) followed by __VPMADDWD__ (i16×1→i32), processing 16 elements per iteration instead of 32-64, but still maintaining competitive performance through careful port utilization.
Sierra Forest's AVX-VNNI variant further improves on Ice Lake by 30-40% via better instruction scheduling and lower latency.
Long arithmetic circuits use __32-bit accumulators__ to prevent overflow: even with max values (±127 or 0-255), 16 million products fit safely in i32 range.

```c
// Asymmetric transform for i8×i8 using DPBUSD (unsigned×signed)
a_unsigned = a XOR 0x80;           // Convert signed→unsigned
result = DPBUSD(a_unsigned, b);    // Computes (a+128)×b
correction = 128 * sum(b);         // Parallel on different port
final = result - correction;       // True a×b value (1.36x faster!)
```

### Int4

Both signed and unsigned 4-bit nibble-sized integers are supported for __fast AI inference__ workloads, following the same quantization strategies as ONNX and TensorFlow Lite.
The challenge with Int4 is __sub-byte addressing__: two nibbles pack into each byte, requiring careful extraction and sign extension.
NumKong uses bitmasks to isolate low nibbles `(byte & 0x0F)` and high nibbles `(byte >> 4)`, then for signed i4, applies the transformation `(nibble ⊕ 8) - 8` to map the unsigned range [0,15] to signed range [-8,7].

For __dot products__, NumKong maintains __separate accumulators__ for low and high nibbles, processing them independently then summing at finalization.
This avoids expensive nibble-interleaving operations and allows SIMD lanes to work in parallel.
The memory savings are substantial: a 7B parameter LLM in Int4 requires 3.5 GB (vs 7 GB in Int8, 14 GB in FP16), fitting entirely in consumer GPU memory and enabling edge deployment.
For __GEMM operations__, the packed matrix format stores nibbles contiguously, and unpacking happens during the microkernel's register-blocking phase, amortizing the extraction cost across multiple dot products.

```c
// Nibble extraction and sign extension for i4
low_nibble = byte & 0x0F;             // Mask lower 4 bits
high_nibble = (byte >> 4) & 0x0F;     // Shift and mask upper 4 bits
i4_value = (low_nibble ^ 8) - 8;      // Maps [0,15] to [-8,7]
```




## License

Feel free to use the project under Apache 2.0 or the Three-clause BSD license at your preference.

# NumKong for Rust

NumKong's Rust crate keeps most of the native kernel surface while expressing it in Rust-native terms.
Rust is the cleanest place to use NumKong when you want static typing, explicit ownership, and strong container APIs without giving up mixed precision.
Traits cover scalar metric families.
`Tensor`, `Vector`, and packed matrix types cover the higher-level workflows.
Custom allocators, low-precision storage wrappers, and explicit row-contiguity checks stay visible instead of being hidden behind a dynamic runtime.
The crate makes the storage policy and result promotion visible.
That matters for fp16, bf16, fp8, packed bits, and strided reductions.

## Quickstart

```rust
use numkong::{configure_thread, Dot};

fn main() {
    configure_thread();
    let a = [1.0_f32, 2.0, 3.0];
    let b = [4.0_f32, 5.0, 6.0];
    let dot = f32::dot(&a, &b).unwrap();
    println!("dot={dot}");
}
```

## Highlights

This is the most complete high-level SDK after Python.
It is the best fit if you want most of the native breadth without dropping into a manual FFI layer.

__Trait-first scalar API.__
`Type::operation(&a, &b)` stays compact and predictable.
__Allocator-aware tensors.__
`Tensor`, `PackedMatrix`, and `MaxSimPackedMatrix` can use custom allocators.
__Storage-first low precision.__
`f16`, `bf16`, fp8, fp6, and packed integer wrappers are first-class types.
__Matrix kernels with explicit contracts.__
Packed and symmetric kernels validate shapes and row contiguity.
__No hidden thread pool.__
Parallel helpers remain host-controlled.
__Fork Union support.__
The `parallel` feature is the intended native orchestration layer.

## Ecosystem Comparison

| Feature                  | NumKong                                                                         | nalgebra                          | ndarray                          |
| ------------------------ | ------------------------------------------------------------------------------- | --------------------------------- | -------------------------------- |
| Operation families       | dots, distances, binary, probability, geospatial, mesh, elementwise, reductions | linear algebra, decompositions    | general n-dimensional arithmetic |
| SIMD acceleration        | runtime-dispatched across x86, ARM, RISC-V                                      | optional SIMD via `simba`         | none built-in                    |
| Type support             | `f64`, `f32`, `f16`, `bf16`, fp8, fp6, `i8`, `u8`, packed bits                  | `f64`, `f32`, generic `RealField` | `f64`, `f32`, integer types      |
| Mixed-precision widening | automatic per-operation promotions                                              | manual                            | manual                           |
| Custom allocators        | `Tensor`, `PackedMatrix`, `MaxSimPackedMatrix`                                  | `VecStorage` with global only     | global only                      |
| Packed matrix reuse      | pack once, reuse across batches                                                 | no equivalent                     | no equivalent                    |
| Symmetric kernels        | `SYRK`-like, skips duplicate pairs                                              | manual triangular access          | no equivalent                    |
| Parallel helpers         | Fork Union, row-range partitioning                                              | Rayon via `par-rayon` feature     | `par_azip` via `rayon`           |

NumKong validates `f16` and `bf16` interop against the `half` crate in its own test suite.
That lets you move between ecosystem-standard half types and NumKong's kernel-facing wrappers without ambiguity.

## Installation

Minimal:

```toml
[dependencies]
numkong = "6.5.15"
```

With host-side parallel helpers:

```toml
[dependencies]
numkong = { version = "6.5.15", features = ["parallel", "std"] }
```

## Compilation and Backend Selection

The crate uses the `cc` build system to compile the C backend with `NK_DYNAMIC_DISPATCH=1` automatically.
All supported backends for the target architecture are compiled into a single binary and selected at runtime.

The two Cargo features are `std`, which enables standard library support, and `parallel`, which adds host-side orchestration via Fork Union and implies `std`.

Backend selection follows the target architecture.
ARM gets NEON, SVE, and SME, with SME available on Linux, FreeBSD, and macOS.
x86-64 gets Haswell (AVX2), Skylake/Icelake/Sapphire Rapids AVX-512 variants, and AMX on Linux and Windows only.
RISC-V gets RVV backends on Linux and FreeBSD.
WASM gets relaxed v128.

Individual backends can be disabled through environment variables.
Any `NK_TARGET_*` variable set to `0` or `false` disables that backend.
Backends not explicitly disabled are enabled by default for the target platform.

```sh
NK_TARGET_NEON=0 cargo build
NK_TARGET_SVE=0 NK_TARGET_SME=0 cargo build
```

If a backend fails to compile, the build system automatically disables it and retries with the remaining backends.
A warning is emitted for each disabled backend.

## Dynamic Dispatch and Capabilities

`configure_thread` enables CPU-specific acceleration features such as Intel AMX.
It must be called once per thread before using AMX operations.

```rust
use numkong::{available, configure_thread, cap};

let caps = available();
configure_thread();

if caps & cap::SAPPHIREAMX != 0 {
    println!("AMX available");
}
```

Call `configure_thread` at the start of every thread that will use AMX operations.
In a thread-pool setting, each worker thread needs its own call.
The function is idempotent and cheap to call more than once on the same thread.

## Core Traits

The crate root re-exports the main metric families:

- `Dot`, `VDot`, `Angular`, `Euclidean`
- `Hamming`, `Jaccard`
- `KullbackLeibler`, `JensenShannon`
- `Haversine`, `Vincenty`
- `Bilinear`, `Mahalanobis`
- `ReduceMoments`, `ReduceMinMax`
- `EachScale`, `EachSum`, `EachBlend`, `EachFMA`

The standard call shape is:

```rust
use numkong::{Dot, JensenShannon, Jaccard, u1x8};

let a = [1.0_f32, 2.0, 3.0];
let b = [4.0_f32, 5.0, 6.0];
let dot = f32::dot(&a, &b).unwrap();

let bits_a = [u1x8(0b11110000), u1x8(0b00001111)];
let bits_b = [u1x8(0b11110000), u1x8(0b11110000)];
let jaccard = u1x8::jaccard(&bits_a, &bits_b).unwrap();

let p = [0.2_f32, 0.3, 0.5];
let q = [0.1_f32, 0.3, 0.6];
let jsd = f32::jensenshannon(&p, &q).unwrap();

println!("{dot} {jaccard} {jsd}");
```

## Dot Products

Dot products span real, complex, quantized, and packed storage types.

```rust
use numkong::{Dot, VDot, f32c};

let a = [f32c { re: 1.0, im: 2.0 }, f32c { re: 3.0, im: 4.0 }];
let b = [f32c { re: 5.0, im: 6.0 }, f32c { re: 7.0, im: 8.0 }];

let dot = f32c::dot(&a, &b).unwrap();
let vdot = f32c::vdot(&a, &b).unwrap(); // like numpy.vdot, conjugated

println!("{dot:?} {vdot:?}");
```

## Dense Distances

The dense spatial family covers `sqeuclidean`, `euclidean`, and `angular`.
The main value over naive loops is the combination of SIMD and safer accumulation policy.

```rust
use numkong::Euclidean;

let a = [1_i8, 2, 3, 4];
let b = [4_i8, 3, 2, 1];

let distance = i8::euclidean(&a, &b).unwrap(); // widened output, not int8
println!("{distance}");
```

## Scalar Types and Promotions

The scalar wrappers are storage-first types.
They are not decorative aliases over `f32`.

| Type   | Layout        | Bytes | Range        | Inf | NaN |
| ------ | ------------- | ----- | ------------ | --- | --- |
| `f16`  | 1+5+10        | 2     | ±65504       | yes | yes |
| `bf16` | 1+8+7         | 2     | ±3.4×10³⁸    | yes | yes |
| `e4m3` | 1+4+3         | 1     | ±448         | no  | yes |
| `e5m2` | 1+5+2         | 1     | ±57344       | yes | yes |
| `e2m3` | 1+2+3 (6 bit) | 1     | ±7.5         | no  | no  |
| `e3m2` | 1+3+2 (6 bit) | 1     | ±28          | no  | no  |
| `u1x8` | 8 packed bits | 1     | 0–1 per bit  | —   | —   |
| `u4x2` | 2×4-bit uint  | 1     | 0–15 per nib | —   | —   |
| `i4x2` | 2×4-bit int   | 1     | −8–7 per nib | —   | —   |

The trait hierarchy documents intent:

- `StorageElement` — raw storable element type.
- `NumberLike` — adds numeric conversion and ordering.
- `FloatConvertible` — adds unpacking and float-domain conversion.

The output type is intentionally wider than the storage type for many operations.
For example, `i8::dot` returns `i32`.
`f32::dot` returns a wider accumulator type.
Moments reductions widen even more aggressively.

## Set Similarity

Packed-binary metrics work on packed words instead of boolean slices.
That is the right model once the workload is "semantic hash" rather than "array of booleans".

```rust
use numkong::{Hamming, Jaccard, u1x8};

let a = [u1x8(0b10101010), u1x8(0b11110000)];
let b = [u1x8(0b10101110), u1x8(0b11000000)];
let hamming = u1x8::hamming(&a, &b).unwrap();
let jaccard = u1x8::jaccard(&a, &b).unwrap();
```

Integer set Jaccard works on sorted arrays of integer identifiers.

```rust
let set_a = [1_u32, 3, 5, 7, 9];
let set_b = [3_u32, 5, 8, 9, 10];
let jaccard_sets = u32::jaccard(&set_a, &set_b).unwrap();
assert!(jaccard_sets > 0.0 && jaccard_sets < 1.0); // |A ∩ B| / |A ∪ B|
```

## Probability Metrics

```rust
use numkong::{JensenShannon, KullbackLeibler};

let p = [0.2_f32, 0.3, 0.5], q = [0.1_f32, 0.3, 0.6];
let kl_forward = f32::kullbackleibler(&p, &q).unwrap();
let kl_reverse = f32::kullbackleibler(&q, &p).unwrap();
assert!(kl_forward != kl_reverse); // KLD is asymmetric

let js_forward = f32::jensenshannon(&p, &q).unwrap();
let js_reverse = f32::jensenshannon(&q, &p).unwrap();
assert!((js_forward - js_reverse).abs() < 1e-6, "JSD is symmetric");
```

## Geospatial Metrics

Inputs are latitudes and longitudes in radians.
Outputs are meters.

```rust
use numkong::{Haversine, Vincenty};

// Statue of Liberty (40.6892°N, 74.0445°W) → Big Ben (51.5007°N, 0.1246°W)
let liberty_lat = [0.7101605100_f64], liberty_lon = [-1.2923203180_f64];
let big_ben_lat = [0.8988567821_f64], big_ben_lon = [-0.0021746802_f64];
let mut distance = [0.0_f64; 1];
f64::vincenty(&liberty_lat, &liberty_lon, &big_ben_lat, &big_ben_lon, &mut distance).unwrap();  // ≈ 5,589,857 m (ellipsoidal, baseline)
f64::haversine(&liberty_lat, &liberty_lon, &big_ben_lat, &big_ben_lon, &mut distance).unwrap(); // ≈ 5,543,723 m (spherical, ~46 km less)

// Vincenty in f32 — drifts ~2 m from f64
let liberty_lat32 = [0.7101605100_f32], liberty_lon32 = [-1.2923203180_f32];
let big_ben_lat32 = [0.8988567821_f32], big_ben_lon32 = [-0.0021746802_f32];
let mut distance_f32 = [0.0_f32; 1];
f32::vincenty(&liberty_lat32, &liberty_lon32, &big_ben_lat32, &big_ben_lon32, &mut distance_f32).unwrap(); // ≈ 5,589,859 m (+2 m drift)
```

## Curved Metrics

Curved-space kernels combine vectors with an extra metric tensor or covariance inverse.

```rust
use numkong::{Bilinear, Mahalanobis, f32c};

// Complex bilinear form: aᴴ M b
let a = [f32c { re: 1.0, im: 0.0 }; 16];
let b = [f32c { re: 0.0, im: 1.0 }; 16];
let metric = [f32c { re: 1.0, im: 0.0 }; 16 * 16];
let bilinear = f32c::bilinear(&a, &b, &metric).unwrap();

// Real Mahalanobis distance: √((a−b)ᵀ M⁻¹ (a−b))
let x = [1.0_f32; 32];
let y = [2.0_f32; 32];
let mut inv_cov = vec![0.0_f32; 32 * 32];
for i in 0..32 { inv_cov[i * 32 + i] = 1.0; } // identity matrix
let distance = f32::mahalanobis(&x, &y, &inv_cov).unwrap();
```

## Vectors, Tensors, Views, and Spans

The container model is unusual enough that it needs direct documentation.

- `Vector<T>` owns one-dimensional storage.
- `VectorView<'a, T>` is an immutable borrowed view.
- `VectorSpan<'a, T>` is a mutable borrowed view.
- `Tensor<T, A, MAX_RANK>` owns N-dimensional storage and can use a custom allocator.
- `TensorView` and `TensorSpan` are the borrowed forms.
- `Matrix<T>` is a rank-2 alias over `Tensor<T, _, 2>`.

The allocator story is explicit.
`Tensor` and `PackedMatrix` default to `Global`.
The underlying layout uses `SIMD_ALIGNMENT == 64` for owned allocations.
That does _not_ mean callers must align their source buffers manually.
It means owned outputs and packed payloads are allocated in a SIMD-friendly way when the crate owns them.

```rust
use numkong::{SliceRange, Tensor};

let t = Tensor::<f32>::try_from_slice(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[3, 3]).unwrap();
assert_eq!(t.shape(), &[3, 3]);
assert_eq!(t.row(1), Some(&[3.0, 4.0, 5.0][..]));

// Transpose
let view = t.view().transpose();
assert_eq!(view.shape(), &[3, 3]);

// Column slicing — produces a strided view
let col1 = t.view().slice(&[SliceRange::full(), SliceRange::index(1)]).unwrap();
assert_eq!(col1.shape(), &[3]);

// Scalar element access
let val = t.view().get(&[1, 2]).unwrap(); // row 1, col 2 → 5.0

// Reduction on a sliced view
let idx = col1.try_argmin_all().unwrap(); // index of the minimum in the second column
```

The main layout rules are:

- General slicing and transposition are supported by views.
- Elementwise and many reduction kernels accept strided views.
- Matrix-style kernels require rank-2 inputs with contiguous rows.
- A tensor can be non-contiguous overall and still have contiguous rows.
- Some reductions have SIMD kernels for strided lanes.
- Some backends still fall back depending on alignment and dtype.

## Elementwise Operations

Elementwise kernels live on tensors and views.
They are not a promise that every arbitrary strided view gets the same SIMD path on every backend.

```rust
use numkong::Tensor;

let a = Tensor::<f32>::try_from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
let b = Tensor::<f32>::try_full(&[2, 2], 2.0).unwrap();

let blended = a.view().try_blend_tensor(&b.view(), 0.25, 0.75).unwrap();
let sines = blended.sin().unwrap();

assert_eq!(sines.shape(), &[2, 2]);
```

## Trigonometry

The trigonometric kernels share the tensor and view surface.
They are useful both directly and as a sanity check that the container path is not just about matrix kernels.

```rust
use numkong::Tensor;

let a = Tensor::<f32>::try_from_slice(&[0.0, 1.0, 2.0, 3.0], &[2, 2]).unwrap();
let c = a.cos().unwrap();
let s = a.sin().unwrap();

assert_eq!(c.shape(), &[2, 2]);
assert_eq!(s.shape(), &[2, 2]);
```

## Moments Reductions

Moments reductions return both sum and sum-of-squares.
That is the right building block for norms and variance-like workflows.

```rust
use numkong::{ReduceMoments, Tensor};

let narrow = Tensor::<u8>::try_full(&[1024], 255).unwrap();
let (sum, sumsq) = narrow.try_moments_all().unwrap();

assert!(sum > 255);      // a naive u8 accumulation would overflow immediately
assert!(sumsq > 255u64); // same for sum-of-squares
```

The important documentation point is not just "wider outputs exist".
It is that the API makes the widened outputs part of the type story.

## Min/Max Reductions

Min/max reductions are a separate family because they stress layout differently.
They also expose the unusual strength of NumKong's reduction backends: some strided lanes still hit SIMD kernels.

```rust
use numkong::Tensor;

let t = Tensor::<f32>::try_from_slice(&[
    3.0, 0.0, 7.0,
    1.0, 2.0, 5.0,
    4.0, -1.0, 6.0,
], &[3, 3]).unwrap();

let second_column = t.view().slice(&[SliceRange::full(), SliceRange::index(1)]).unwrap();
let idx = second_column.try_argmin_all().unwrap();

assert_eq!(idx, 2);
```


## Sparse Operations and Intersections

Sparse helpers cover both sorted-index intersection and weighted sparse dot products.

```rust
use numkong::{SparseIntersect, SparseDot};

let a_idx = [1_u32, 3, 5, 7], b_idx = [3_u32, 4, 5, 8];
let count = u32::sparse_intersection_size(&a_idx, &b_idx);
assert_eq!(count, 2); // indices 3 and 5

let a_weights = [1.0_f32, 2.0, 3.0, 4.0], b_weights = [5.0_f32, 6.0, 7.0, 8.0];
let dot = u32::sparse_dot(&a_idx, &b_idx, &a_weights, &b_weights).unwrap();
assert!(dot > 0.0); // weighted dot over shared indices
```

## Packed Matrix Kernels for GEMM-Like Workloads

Packed kernels are the main "matrix throughput" path in the crate.
They are `GEMM`-like in workload shape.
They are not a thin BLAS clone.

```rust
use numkong::{PackedMatrix, Tensor};

let a = Tensor::<f32>::try_full(&[1024, 512], 1.0).unwrap();
let b = Tensor::<f32>::try_full(&[256, 512], 1.0).unwrap();

let b_packed = PackedMatrix::try_pack(&b).unwrap();
let c = a.dots_packed(&b_packed);

assert_eq!(c.shape(), &[1024, 256]);
```

The useful economics are:

- pack `B` once
- reuse it across many `A` batches
- convert or pad once during packing instead of on every multiply
- reuse precomputed norms for `angulars_packed` and `euclideans_packed`

The crate checks row contiguity because these kernels assume contiguous rows.
Caller-side source alignment is not required.
The owned packed buffer handles its own aligned allocation internally.

## Symmetric Kernels for SYRK-Like Workloads

Symmetric kernels are for self-similarity and self-distance.
They are `SYRK`-like in shape.
They avoid duplicate `(i, j)` and `(j, i)` work.

```rust
use numkong::Tensor;

let vectors = Tensor::<f32>::try_full(&[100, 768], 1.0).unwrap();
let gram = vectors.view().try_dots_symmetric().unwrap();

assert_eq!(gram.shape(), &[100, 100]);
```

This family is also where row-window partitioning becomes the natural parallel model.
That is structurally different from packed `GEMM`-style work against a shared packed RHS.

## MaxSim and ColBERT-Style Late Interaction

MaxSim is the late-interaction primitive used by systems such as [ColBERT](https://arxiv.org/abs/2004.12832).
It is not "just another matrix multiply".

```rust
use numkong::{MaxSimPackedMatrix, Tensor};

let queries = Tensor::<f32>::try_full(&[4, 16], 1.0).unwrap();
let docs = Tensor::<f32>::try_full(&[8, 16], 1.0).unwrap();

let queries_packed = queries.view().try_maxsim_pack().unwrap();
let docs_packed = docs.view().try_maxsim_pack().unwrap();
let score = queries_packed.score(&docs_packed);

assert!(score.is_finite());
```

## Geometric Mesh Alignment

Mesh alignment returns transforms, scales, and RMSD values.
That is a different API shape from the scalar metric families.

```rust
use numkong::MeshAlignment;

let source = [[0.0_f32, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
let target = [[0.0_f32, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];

let result = f32::kabsch(&source, &target).unwrap();
assert!(result.rmsd < 1e-6);
assert!((result.scale - 1.0).abs() < 1e-6);

// Umeyama with known 2x scaling
let scaled = [[0.0_f32, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]];
let result = f32::umeyama(&source, &scaled).unwrap();
assert!(result.rmsd < 1e-6);
assert!((result.scale - 2.0).abs() < 0.01);
```

## Parallelism and Fork Union

NumKong does not own a thread pool.
The `parallel` feature adds host-side orchestration helpers via [Fork Union](https://github.com/ashvardanian/ForkUnion), not a hidden scheduler.

```rust
use numkong::{PackedMatrix, Tensor};
use fork_union::ThreadPool;

let a = Tensor::<f32>::try_full(&[4096, 768], 1.0).unwrap();
let b = Tensor::<f32>::try_full(&[8192, 768], 1.0).unwrap();
let mut pool = ThreadPool::try_spawn(4).unwrap();

// GEMM-like: rows of A partitioned across threads, one shared packed B
let b_packed = PackedMatrix::try_pack(&b).unwrap();
let c = a.dots_packed_parallel(&b_packed, &mut pool);
assert_eq!(c.shape(), &[4096, 8192]);

// SYRK-like: row windows of one square output partitioned across threads
let gram = a.dots_symmetric_parallel(&mut pool);
assert_eq!(gram.shape(), &[4096, 4096]);
```

Rayon or a manual thread pool can still work if the rest of your application already depends on them.

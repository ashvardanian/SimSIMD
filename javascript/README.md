# NumKong for JavaScript

NumKong's JavaScript package brings low-latency vector kernels to Node and Bun-style runtimes, targeting the space between handwritten loops over `TypedArray`s and much heavier tensor frameworks.
It keeps the JS surface intentionally compact: dense distances, dot products, binary metrics, probability divergences, dtype-tagged low-precision storage, typed views, and runtime capability inspection.
The binding does not include symmetric matrix kernels or MaxSim; use the Python or Rust SDK for those workloads.

## Quickstart

```ts
import { dot } from "numkong";

const a = new Float32Array([1, 2, 3]);
const b = new Float32Array([4, 5, 6]);
console.log(dot(a, b)); // 32
```

## Highlights

This SDK is deliberately smaller than Python or Rust.
Its job is to make the hot vector kernels easy to use from modern JavaScript runtimes.

__TypedArray-first API.__
Standard `Float32Array`, `Float64Array`, `Int8Array`, and `Uint8Array` work directly.
__DType tags for exotic storage.__
`f16`, `bf16`, fp8, and packed bits stay explicit.
__Owned and borrowed views.__
`Vector`, `VectorView`, and base tensor wrappers preserve dtype metadata.
__Portable runtime story.__
The same package can target native addons and WASM runtimes.
__No fake tensor-framework scope.__
This binding stays centered on the vector families it actually exports.

## Ecosystem Comparison

| Feature                 | NumKong                         | onnxruntime-node | tensorflow.js     | TypedArray loops |
| ----------------------- | ------------------------------- | ---------------- | ----------------- | ---------------- |
| Vector distances        | sqeuclidean, euclidean, angular | not included     | limited set       | manual loops     |
| Dot products            | SIMD-accelerated, dtype-tagged  | not included     | `matMul`-based    | manual loops     |
| Binary metrics          | Hamming, Jaccard on packed bits | not included     | not included      | manual loops     |
| Probability divergences | KL, JS on all float types       | not included     | limited           | manual loops     |
| Low-precision types     | `f16`, `bf16`, `e4m3`, `e5m2`   | not included     | not included      | not included     |
| SIMD acceleration       | native addon                    | WASM SIMD        | WASM SIMD / WebGL | none             |
| Portable WASM fallback  | yes                             | yes              | yes               | n/a              |
| Bundle size             | small                           | large            | large             | zero             |

## Installation

The package targets Node `>= 22`.

```sh
npm install numkong
yarn add numkong
pnpm add numkong
bun add numkong
```

If you build from source, the package uses `node-gyp-build` on install and TypeScript sources under `javascript/`.

## Dot Products

Dot products are separate from distances because dtype tagging and low-precision storage matter more here.

```ts
import { dot, inner } from "numkong";

const a = new Float32Array([1, 2, 3]);
const b = new Float32Array([4, 5, 6]);

console.log(dot(a, b));
console.log(inner(a, b)); // alias for ecosystem familiarity
```

For non-native numeric layouts, pass an explicit `DType` or wrap the storage in a typed NumKong view.

## Dense Distances

Dense distance entrypoints work directly on the standard numeric `TypedArray` types.

```ts
import { sqeuclidean, euclidean, angular } from "numkong";

const a = new Float32Array([1, 2, 3]);
const b = new Float32Array([4, 5, 6]);

console.log(sqeuclidean(a, b)); // equivalent shape to a manual sum((a[i] - b[i]) ** 2)
console.log(euclidean(a, b));
console.log(angular(a, b));
```

When the storage type is one of the standard JS typed arrays, dtype inference is automatic.

## Binary Metrics

Binary metrics operate on packed storage rather than on generic boolean arrays.

```ts
import { toBinary, hamming, jaccard } from "numkong";

const a = toBinary(new Float32Array([1, -2, 3, -4, 5, -6, 7, -8]));
const b = toBinary(new Float32Array([1, 2, -3, -4, 5, 6, -7, -8]));

console.log(hamming(a, b));
console.log(jaccard(a, b));
```

That is a better model for sign-quantized embeddings and semantic hashes than looping over JS booleans.

## Probability Metrics

```ts
import { kullbackleibler, jensenshannon } from "numkong";

const p = new Float32Array([0.2, 0.3, 0.5]);
const q = new Float32Array([0.1, 0.3, 0.6]);

console.log(kullbackleibler(p, q));
console.log(jensenshannon(p, q));
```

These are direct divergence kernels, not scalar JS loops hidden behind a helper.

## DType Tags and Low-Precision Arrays

JavaScript has no built-in `f16`, `bf16`, fp8, or packed-bit numeric model.
NumKong handles that with explicit dtype tags and wrapper arrays.

The supported low-precision types and their bit layouts are:

- `Float16Array`: 1 sign + 5 exponent + 10 mantissa bits, 2 bytes per element, range ±65504, supports Inf and NaN
- `BFloat16Array`: 1 sign + 8 exponent + 7 mantissa bits, 2 bytes per element, full `f32` dynamic range, supports Inf and NaN
- `E4M3Array`: 1 sign + 4 exponent + 3 mantissa bits, 1 byte per element, range ±448, no Inf, NaN is 0x7F only
- `E5M2Array`: 1 sign + 5 exponent + 2 mantissa bits, 1 byte per element, range ±57344, supports Inf and NaN
- `BinaryArray`: packed bits in bytes, 8 elements per byte

Use `.byteLength` for the exact payload size.

```ts
import { Float16Array, E4M3Array, DType, dot, angular } from "numkong";

const a16 = new Float16Array([1, 2, 3]);
const b16 = new Float16Array([4, 5, 6]);
console.log(dot(a16, b16, DType.F16));
console.log(a16.byteLength);

const a8 = new E4M3Array([1, 2, 3]);
const b8 = new E4M3Array([4, 5, 6]);
console.log(angular(a8, b8, DType.E4M3));
console.log(a8.byteLength);
```

If the underlying storage is a raw `Uint16Array` or `Uint8Array`, JS itself cannot know whether those bytes mean integers, `f16`, `bf16`, fp8, or packed bits.
That is exactly when the `DType` tag becomes mandatory.

## Vector Views and Owned Buffers

The wrapper hierarchy exists to keep dtype and ownership explicit across addon and WASM boundaries.

- `TensorBase` carries `buffer`, `byteOffset`, and `dtype`
- `VectorBase` adds rank-1 semantics
- `VectorView` is a zero-copy borrowed wrapper over existing memory
- `Vector` owns its `ArrayBuffer`

```ts
import { VectorView, Vector, DType, dot } from "numkong";

const raw = new Uint16Array([0x3c00, 0x4000, 0x4200]); // f16 payload, not uint16 values
const view = VectorView.from(raw, DType.F16);

const owned = new Vector(3, DType.F32);
owned.toTypedArray().set(new Float32Array([4, 5, 6]));

console.log(dot(view, view));
console.log(dot(owned, owned));
console.log(owned.byteLength);
```

Use `VectorView` when the bytes already live somewhere else.
Use `Vector` when NumKong should own the storage.

## Scalar Conversion Helpers

The package also exposes scalar conversion helpers for the exotic formats:

- `castF16ToF32`
- `castF32ToF16`
- `castBF16ToF32`
- `castF32ToBF16`
- `castE4M3ToF32`
- `castF32ToE4M3`
- `castE5M2ToF32`
- `castF32ToE5M2`
- `cast`

Those are useful both for diagnostics and for building your own typed views over raw payload buffers.

## Capabilities and Runtime Selection

Capability detection is explicit:

```ts
import { Capability, getCapabilities, hasCapability } from "numkong";

const caps = getCapabilities();

console.log(caps);
console.log(hasCapability(Capability.HASWELL));
console.log(hasCapability(Capability.NEON));
```

The exact bitmask depends on whether you are running the native addon or a WASM runtime.

There is no `configure_thread` call in the JS binding.
Thread configuration is managed internally by the native addon or WASM runtime.

## Native Addon and WASM Runtimes

The top-level package is native-first.
It loads the compiled addon through `node-gyp-build`.

The repository also ships WASM wrappers.
Those are useful for portable or sandboxed environments.
They should not be described as feature-complete replacements for the native-heavy SDKs.

Practical guidance:

- Use the native addon for the lowest host-call latency.
- Use the WASM path when portability matters more than absolute latency.
- Keep your expectations scoped to the vector-oriented API that this binding actually exports.

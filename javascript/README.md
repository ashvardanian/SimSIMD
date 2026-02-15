# NumKong for JavaScript

Portable mixed-precision BLAS-like vector math library with SIMD acceleration for x86, ARM, RISC-V, and WASM.

## Installation

```sh
npm install numkong
yarn add numkong
pnpm add numkong
bun install numkong
```

The package ships with prebuilt binaries.
If your platform isn't covered, `npm run build` compiles from source automatically (unless you pass `--ignore-scripts`).

## Quickstart

```js
import { dot, angular, euclidean } from 'numkong';

const a = new Float32Array([1.0, 2.0, 3.0]);
const b = new Float32Array([4.0, 5.0, 6.0]);

console.log(dot(a, b));        // 32
console.log(euclidean(a, b));  // 5.196...
console.log(angular(a, b));    // ~0.0253 radians
```

## Data Types

NumKong supports 11 numeric formats via the `DType` enum.
JavaScript has no native f16, bf16, or fp8 — so for those types, the backing storage holds opaque bit patterns (`Uint16Array` for 16-bit, `Uint8Array` for 8-bit).
Indexing into the backing array gives you raw bits, not usable floats.
To work with values, use the [encoding helpers](#encoding-helpers) or — more commonly — pass the data through [tensor containers](#tensor-types) directly to distance functions.

| `DType`      | Bits | Backing Storage | C Type      | Use Case                                   |
| ------------ | ---- | --------------- | ----------- | ------------------------------------------ |
| `DType.F64`  | 64   | `Float64Array`  | `nk_f64_t`  | High precision scientific computing        |
| `DType.F32`  | 32   | `Float32Array`  | `nk_f32_t`  | Standard ML embeddings                     |
| `DType.F16`  | 16   | `Uint16Array`   | `nk_f16_t`  | GPU inference, mobile ML                   |
| `DType.BF16` | 16   | `Uint16Array`   | `nk_bf16_t` | TPU training, mixed precision              |
| `DType.E4M3` | 8    | `Uint8Array`    | `nk_e4m3_t` | H100 inference (4-bit exp, 3-bit mantissa) |
| `DType.E5M2` | 8    | `Uint8Array`    | `nk_e5m2_t` | H100 training (5-bit exp, 2-bit mantissa)  |
| `DType.E2M3` | 6    | `Uint8Array`    | `nk_e2m3_t` | FP6 (2-bit exp, 3-bit mantissa)            |
| `DType.E3M2` | 6    | `Uint8Array`    | `nk_e3m2_t` | FP6 (3-bit exp, 2-bit mantissa)            |
| `DType.I8`   | 8    | `Int8Array`     | `nk_i8_t`   | Quantized models                           |
| `DType.U8`   | 8    | `Uint8Array`    | `nk_u8_t`   | Quantized models                           |
| `DType.U1`   | 1    | `Uint8Array`    | `nk_u1x8_t` | Binary embeddings, semantic hashing        |

`DType` is a regular (non-const) numeric enum — `DType.F32 === 1`, `DType[1] === "F32"`.
Distance functions use integer switch dispatch internally (compiles to a jump table, no string overhead).

## Usage Examples

__Basic operations with native types:__

```js
import { dot, angular } from 'numkong';

// Standard 32-bit floats
const a = new Float32Array([1.0, 2.0, 3.0]);
const b = new Float32Array([4.0, 5.0, 6.0]);
console.log(dot(a, b));  // 32.0

// 64-bit doubles for high precision
const c = new Float64Array([1.0, 2.0, 3.0]);
const d = new Float64Array([4.0, 5.0, 6.0]);
console.log(angular(c, d));  // ~0.0253 radians
```

__Custom data types (with tensor containers):__

```js
import { VectorView, DType, dot, angular } from 'numkong';

// Wrap f16 data in a VectorView — dtype tag means no third argument needed
const a_f16 = VectorView.from(new Uint16Array([...f16Bits]), DType.F16);
const b_f16 = VectorView.from(new Uint16Array([...f16Bits]), DType.F16);
console.log(dot(a_f16, b_f16)); // dtype carried by the view
```

__Custom data types (with encoding helpers + explicit dtype):__

```js
import { Float16Array, E4M3Array, DType, dot, angular } from 'numkong';

// Float16Array encodes f32 → f16 bits on construction
const a_f16 = new Float16Array([1.0, 2.0, 3.0]);
const b_f16 = new Float16Array([4.0, 5.0, 6.0]);
console.log(dot(a_f16, b_f16, DType.F16)); // explicit dtype for raw TypedArray

// FP8 E4M3 format
const a_fp8 = new E4M3Array([1.0, 2.0, 3.0]);
const b_fp8 = new E4M3Array([4.0, 5.0, 6.0]);
console.log(angular(a_fp8, b_fp8, DType.E4M3));
```

__Type conversions:__

```js
import { castF16ToF32, castF32ToF16, castBF16ToF32, castF32ToBF16 } from 'numkong';

// f16 ↔ f32 conversion (uses hardware acceleration on supported CPUs)
const f32_value = 3.14159;
const f16_bits = castF32ToF16(f32_value);       // Returns uint16 bit representation
const f32_decoded = castF16ToF32(f16_bits);     // Decode back to f32

// bf16 ↔ f32 conversion
const bf16_bits = castF32ToBF16(f32_value);
const bf16_decoded = castBF16ToF32(bf16_bits);

// FP8 conversions (e4m3 and e5m2)
import { castE4M3ToF32, castF32ToE4M3 } from 'numkong';
const e4m3_bits = castF32ToE4M3(f32_value);
const e4m3_decoded = castE4M3ToF32(e4m3_bits);
```

__Binary vectors (u1):__

```js
import { hamming, jaccard, toBinary, BinaryArray } from 'numkong';

// Quantize floats to bit-packed binary
const binaryA = toBinary(new Float32Array([1, -2, 3, -4, 5, -6, 7, -8]));
const binaryB = toBinary(new Float32Array([1, 2, -3, -4, 5, 6, -7, -8]));
console.log(hamming(binaryA, binaryB)); // Hamming distance
console.log(jaccard(binaryA, binaryB)); // Jaccard distance

// Or use BinaryArray directly
const a = new BinaryArray(128); // 128 bits
a.setBit(0, 1);
a.setBit(5, 1);
```

## Tensor Types

JavaScript `TypedArray` objects don't carry dtype metadata — a `Uint16Array` could be f16, bf16, or raw u16.
NumKong's tensor containers solve this: they pair an `ArrayBuffer` region with a `DType` tag, so distance functions know which SIMD kernel to dispatch without a string argument.

```
TensorBase (abstract)                     buffer, byteOffset, dtype
├── VectorBase (abstract, rank = 1)       + length
│   ├── VectorView                        non-owning (like std::span<T>)
│   └── Vector                            owning (like std::vector<T>)
└── MatrixBase (abstract, rank = 2)       + rows, cols, rowStride, colStride
```

All fields are embedded — zero dynamic allocation.
Every distance function accepts `TensorBase`, so `VectorView`, `Vector`, and any future `MatrixView` all work.

### Classes

| Class        | Ownership  | Key Fields                      | Use Case                              |
| ------------ | ---------- | ------------------------------- | ------------------------------------- |
| `TensorBase` | abstract   | `buffer`, `byteOffset`, `dtype` | Base — distance functions accept this |
| `VectorBase` | abstract   | + `length`                      | Rank-1 interface                      |
| `VectorView` | non-owning | (same)                          | Wrap existing memory, WASM interop    |
| `Vector`     | owning     | (same, owns `ArrayBuffer`)      | Allocate vectors, store results       |
| `MatrixBase` | abstract   | + `rows`, `cols`, strides       | Stub for future matmul                |

### When to use what

| Situation                                 | Recommendation                                                  |
| ----------------------------------------- | --------------------------------------------------------------- |
| Distance on `Float32Array`/`Float64Array` | Pass directly — auto-detected                                   |
| Distance on f16/bf16/fp8 data             | `VectorView.from(arr, DType.F16)` — carries dtype               |
| WASM: data already on heap                | `new VectorView(memory.buffer, offset, len, dtype)` — zero-copy |
| Allocate a fresh buffer                   | `new Vector(len, DType.F32)`, then `.toTypedArray()` to mutate  |
| Deep-copy into owned memory               | `Vector.fromView(view)` or `Vector.fromTypedArray(arr)`         |

### VectorView — wrap existing data without copying

```js
import { VectorView, DType, dot } from 'numkong';

const arr = new Float32Array([1, 2, 3, 4]);
const view = VectorView.from(arr);              // zero-copy, infers DType.F32

console.log(view.dtype === DType.F32);          // true
console.log(view.length);                       // 4
console.log(view.bytesPerElement);              // 4
console.log(view.byteLength);                   // 16
console.log(dot(view, view));                   // 30
```

### Vector — allocate and own memory

```js
import { Vector, DType, euclidean } from 'numkong';

const a = new Vector(3, DType.F32);
const arr = a.toTypedArray();                   // Float32Array view (zero-copy)
arr[0] = 1; arr[1] = 2; arr[2] = 3;             // mutate through TypedArray

const b = Vector.fromTypedArray(new Float32Array([4, 5, 6]));
console.log(euclidean(a, b));                   // 5.196...
```

### Convert between them

```js
import { Vector, VectorView } from 'numkong';

// Vector IS-A VectorBase IS-A TensorBase — pass to any distance function
const vec = Vector.fromTypedArray(new Float32Array([1, 2, 3]));

// VectorView → Vector (copies data, new owned buffer)
const view = VectorView.from(new Float32Array([4, 5, 6]));
const owned = Vector.fromView(view);
```

### Cross-module WASM interop (zero-copy)

```js
import { VectorView, DType } from 'numkong';

const memory = new WebAssembly.Memory({ initial: 256, maximum: 4096 });
const nk = await createNumKongModule({ wasmMemory: memory });

// Another WASM module writes output at some offset in shared memory
const output = new VectorView(memory.buffer, outputOffset, 768, DType.F32);

// Zero-copy distance — no data copied between modules
import * as wasm from 'numkong/numkong-wasm';
wasm.initWasm(nk);
const similarity = wasm.dot(output, queryView);
```

### Encoding helpers

NumKong also provides `Float16Array`, `BFloat16Array`, `E4M3Array`, and `E5M2Array` — `TypedArray` subclasses that handle per-element encode/decode (e.g. `Float16Array.getFloat32(i)`, `.toFloat32Array()`).
These are useful for element-wise inspection or construction, but for distance computation you typically wrap the underlying buffer in a `VectorView` and pass it directly — the SIMD kernel operates on the raw bits, not decoded floats.

## Distance Functions

| Function                           | Types               | Description             |
| ---------------------------------- | ------------------- | ----------------------- |
| `dot`, `inner`                     | all except u1       | Inner product           |
| `angular`                          | all except u1       | Cosine distance         |
| `sqeuclidean`, `euclidean`         | all except u1       | L2 distance             |
| `hamming`, `jaccard`               | u1                  | Binary distances        |
| `kullbackleibler`, `jensenshannon` | f64, f32, f16, bf16 | Probability divergences |

> __Note:__ f16, bf16, e4m3, e5m2, e2m3, e3m2 require passing the dtype as a third argument:
> `dot(a_f16, b_f16, DType.F16)`

## Capability Detection

NumKong exposes runtime SIMD capabilities to JavaScript:

```js
import { getCapabilities, hasCapability, Capability } from 'numkong';

// Get bitmask of available capabilities
const caps = getCapabilities();
console.log(`Capabilities: 0x${caps.toString(16)}`);

// Check for specific capabilities
if (hasCapability(Capability.HASWELL))      console.log('AVX2 available');
if (hasCapability(Capability.NEON))         console.log('ARM NEON available');
if (hasCapability(Capability.V128RELAXED))  console.log('WASM Relaxed SIMD available');
```

## WASM Support

NumKong supports 4 WASM runtime environments:

- __Emscripten__ (Node.js/Browser) - uses `EM_ASM` for JavaScript-based SIMD detection
- __WASI__ (Wasmer/Wasmtime) - uses host-provided imports for standalone runtime detection
- __Browser__ (Playwright) - automated browser testing with Chromium
- __Wasmtime__ (Rust) - integrated Rust tests via cargo

### Building WASM Targets

__Emscripten (for Node.js and Browser):__

```bash
# Install Emscripten
git clone https://github.com/emscripten-core/emsdk.git ~/emsdk
cd ~/emsdk && ./emsdk install latest && ./emsdk activate latest
source ~/emsdk/emsdk_env.sh

# Build
cmake -B build-wasm -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-wasm.cmake
cmake --build build-wasm
```

__WASI (for Wasmer and Wasmtime):__

```bash
# Install WASI-SDK
wget https://github.com/WebAssembly/wasi-sdk/releases/download/wasi-sdk-24/wasi-sdk-24.0-x86_64-linux.tar.gz
tar xf wasi-sdk-24.0-x86_64-linux.tar.gz -C ~
mv ~/wasi-sdk-24.0-x86_64-linux ~/wasi-sdk

# Build
export WASI_SDK_PATH=~/wasi-sdk
cmake -B build-wasi -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-wasi.cmake -DNK_BUILD_WASI=ON
cmake --build build-wasi
```

### Testing WASM Builds

```bash
# Emscripten runtime (Node.js)
NK_RUNTIME=emscripten node --test test/test-wasm.mjs

# WASI runtime (Node.js polyfill)
NK_RUNTIME=wasi-node node --test test/test-wasm.mjs

# Browser (Playwright)
npx playwright install --with-deps chromium
npm run test:wasm:browser

# Wasmer
wasmer run build-wasi/test.wasm

# Wasmtime (via Rust)
cargo test --features wasm-runtime wasm_runtime

# All WASM tests
npm run test:wasm:all
```

### WASI Import Requirements

When instantiating WASI modules, the host must provide SIMD detection imports:

```js
// Node.js WASI
const instance = await WebAssembly.instantiate(wasmBytes, {
    wasi_snapshot_preview1: wasi.wasiImport,
    env: {
        nk_has_v128: () => WebAssembly.validate(simd128TestBytes) ? 1 : 0,
        nk_has_relaxed: () => WebAssembly.validate(relaxedTestBytes) ? 1 : 0
    }
});
```

```rust
// Rust Wasmtime
linker.func_wrap("env", "nk_has_v128", || -> i32 { 1 })?;
linker.func_wrap("env", "nk_has_relaxed", || -> i32 { 1 })?;
```

## Benchmarking

```bash
npm run bench:native                            # Benchmark native Node.js addon
npm run bench:all                               # All runtimes
NK_DIMENSIONS=768 npm run bench:native          # Custom dimensions
NK_FILTER="dot|angular" npm run bench:native    # Filter functions
```

For full benchmarking documentation, see [../bench/README.md](../bench/README.md).

## Development

### Node.js

```bash
nvm install 20
npm install -g typescript
npm install --save-dev @types/node
npm run build-js                        # Build TypeScript
npm test                                # Run tests
npm run bench                           # Run benchmarks
```

### Deno

```bash
deno test -A
```

### Bun

```bash
bun install
bun test ./scripts/test.mjs
```

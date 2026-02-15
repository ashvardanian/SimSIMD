/**
 * @fileoverview NumKong - Portable mixed-precision BLAS-like vector math library
 *
 * NumKong provides SIMD-accelerated distance metrics and vector operations for
 * x86, ARM, RISC-V, and WASM platforms. The library automatically detects and uses
 * the best available SIMD instruction set at runtime.
 *
 * @module numkong
 * @author Ash Vardanian
 *
 * @example
 * ```typescript
 * import { dot, euclidean, Float16Array } from 'numkong';
 *
 * // Auto-detected types
 * const a = new Float32Array([1, 2, 3]);
 * const b = new Float32Array([4, 5, 6]);
 * dot(a, b);        // 32
 * euclidean(a, b);  // 5.196...
 *
 * // Custom types with explicit dtype
 * const c = new Float16Array([1, 2, 3]);
 * const d = new Float16Array([4, 5, 6]);
 * dot(c, d, DType.F16); // 32
 * ```
 */

import build from "node-gyp-build";
import * as path from "node:path";
import { existsSync } from "node:fs";
import { getFileName, getRoot } from "bindings";
import { setConversionFunctions, Float16Array, BFloat16Array, E4M3Array, E5M2Array, BinaryArray, TensorBase, VectorBase, VectorView, Vector, MatrixBase, DType, dtypeToString } from "./dtypes.js";

let compiled: any;

try {
  let builddir = getBuildDir(getDirName());
  compiled = build(builddir);

  // Initialize conversion functions for dtypes.ts
  setConversionFunctions({
    castF16ToF32: compiled.castF16ToF32,
    castF32ToF16: compiled.castF32ToF16,
    castBF16ToF32: compiled.castBF16ToF32,
    castF32ToBF16: compiled.castF32ToBF16,
    castE4M3ToF32: compiled.castE4M3ToF32,
    castF32ToE4M3: compiled.castF32ToE4M3,
    castE5M2ToF32: compiled.castE5M2ToF32,
    castF32ToE5M2: compiled.castF32ToE5M2,
    cast: compiled.cast,
  });
} catch (e) {
  // Native addon not available
  // For WASM usage, import the Emscripten module directly (see test/test-wasm.mjs)
  throw new Error(
    "NumKong native addon not found. Build with `npm run build` or use WASM " +
    "by importing the Emscripten module directly. See test/test-wasm.mjs for examples."
  );
}

/**
 * CPU capability bit masks in chronological order (by first commercial silicon).
 * Use these with getCapabilities() to check for specific SIMD support.
 */
export const Capability = {
  SERIAL: 1n << 0n,          // Always: Fallback
  NEON: 1n << 1n,            // 2013: ARM NEON
  HASWELL: 1n << 2n,         // 2013: Intel AVX2
  SKYLAKE: 1n << 3n,         // 2017: Intel AVX-512
  NEONHALF: 1n << 4n,        // 2017: ARM NEON FP16
  NEONSDOT: 1n << 5n,        // 2017: ARM NEON i8 dot
  NEONFHM: 1n << 6n,         // 2018: ARM NEON FP16 FML
  ICELAKE: 1n << 7n,         // 2019: Intel AVX-512 VNNI
  GENOA: 1n << 8n,           // 2020: Intel/AMD AVX-512 BF16
  NEONBFDOT: 1n << 9n,       // 2020: ARM NEON BF16
  SVE: 1n << 10n,            // 2020: ARM SVE
  SVEHALF: 1n << 11n,        // 2020: ARM SVE FP16
  SVESDOT: 1n << 12n,        // 2020: ARM SVE i8 dot
  SIERRA: 1n << 13n,         // 2021: Intel AVX2+VNNI
  SVEBFDOT: 1n << 14n,       // 2021: ARM SVE BF16
  SVE2: 1n << 15n,           // 2022: ARM SVE2
  V128RELAXED: 1n << 16n,    // 2022: WASM Relaxed SIMD
  SAPPHIRE: 1n << 17n,       // 2023: Intel AVX-512 FP16
  SAPPHIREAMX: 1n << 18n,    // 2023: Intel Sapphire AMX
  RVV: 1n << 19n,            // 2023: RISC-V Vector
  RVVHALF: 1n << 20n,        // 2023: RISC-V Zvfh
  RVVBF16: 1n << 21n,        // 2023: RISC-V Zvfbfwma
  GRANITEAMX: 1n << 22n,     // 2024: Intel Granite AMX FP16
  TURIN: 1n << 23n,          // 2024: AMD Turin AVX-512 CD
  SME: 1n << 24n,            // 2024: ARM SME
  SME2: 1n << 25n,           // 2024: ARM SME2
  SMEF64: 1n << 26n,         // 2024: ARM SME F64
  SMEFA64: 1n << 27n,        // 2024: ARM SME FA64
  SVE2P1: 1n << 28n,         // 2025+: ARM SVE2.1
  SME2P1: 1n << 29n,         // 2025+: ARM SME2.1
  SMEHALF: 1n << 30n,        // 2025+: ARM SME F16F16
  SMEBF16: 1n << 31n,        // 2025+: ARM SME B16B16
  SMELUT2: 1n << 32n,        // 2025+: ARM SME LUTv2
  RVVBB: 1n << 33n,          // 2025+: RISC-V Zvbb
} as const;

/* #region Custom Numeric Types */

export { Float16Array, BFloat16Array, E4M3Array, E5M2Array, BinaryArray, TensorBase, VectorBase, VectorView, Vector, MatrixBase };

/* #endregion Custom Numeric Types */

/* #region Type Conversion Functions */

/** Convert a single FP16 value (as uint16 bits) to FP32 */
export const castF16ToF32 = compiled.castF16ToF32;
/** Convert a single FP32 value to FP16 (returns uint16 bits) */
export const castF32ToF16 = compiled.castF32ToF16;
/** Convert a single BF16 value (as uint16 bits) to FP32 */
export const castBF16ToF32 = compiled.castBF16ToF32;
/** Convert a single FP32 value to BF16 (returns uint16 bits) */
export const castF32ToBF16 = compiled.castF32ToBF16;
/** Convert a single E4M3 value (as uint8 bits) to FP32 */
export const castE4M3ToF32 = compiled.castE4M3ToF32;
/** Convert a single FP32 value to E4M3 (returns uint8 bits) */
export const castF32ToE4M3 = compiled.castF32ToE4M3;
/** Convert a single E5M2 value (as uint8 bits) to FP32 */
export const castE5M2ToF32 = compiled.castE5M2ToF32;
/** Convert a single FP32 value to E5M2 (returns uint8 bits) */
export const castF32ToE5M2 = compiled.castF32ToE5M2;
/** Bulk conversion between different numeric types (modifies destination array in-place) */
export const cast = compiled.cast;

/* #endregion Type Conversion Functions */

/* #region Types */

export { DType };

/**
 * @brief Numeric arrays supported by distance metrics with auto-detected dtype.
 *
 * These standard TypedArrays are auto-detected by the N-API binding.
 */
export type NumericArray = Float64Array | Float32Array | Int8Array | Uint8Array;

/**
 * @brief Extended array types supported by distance metrics with explicit dtype parameter.
 *
 * Includes Uint16Array (backing type for Float16Array and BFloat16Array) in addition
 * to the auto-detected types. Pass a dtype string as the third argument to distance
 * functions when using custom types.
 */
export type DistanceArray = Float64Array | Float32Array | Int8Array | Uint8Array | Uint16Array;

/**
 * @brief Union type for all array types (including custom types for conversions)
 */
export type NumKongArray =
  | Float64Array
  | Float32Array
  | Float16Array
  | BFloat16Array
  | E4M3Array
  | E5M2Array
  | Int8Array
  | Uint8Array
  | BinaryArray;

/* #endregion Types */

/**
 * @brief Extract a TypedArray from a TensorBase for the N-API backend.
 *
 * The native backend doesn't benefit from zero-copy TensorBase (Node.js TypedArrays
 * already share process memory), but accepting TensorBase keeps the API uniform.
 */
function unwrapTensor(input: TensorBase): { arr: DistanceArray; dtype: DType } {
  switch (input.dtype) {
    case DType.F64: return { arr: new Float64Array(input.buffer, input.byteOffset, input.length), dtype: input.dtype };
    case DType.F32: return { arr: new Float32Array(input.buffer, input.byteOffset, input.length), dtype: input.dtype };
    case DType.F16: case DType.BF16: return { arr: new Uint16Array(input.buffer, input.byteOffset, input.length), dtype: input.dtype };
    case DType.I8: return { arr: new Int8Array(input.buffer, input.byteOffset, input.length), dtype: input.dtype };
    case DType.U8: case DType.U1: return { arr: new Uint8Array(input.buffer, input.byteOffset, input.length), dtype: input.dtype };
    default: return { arr: new Uint8Array(input.buffer, input.byteOffset, input.length), dtype: input.dtype };
  }
}

/* #region Distance Metrics and Similarity Functions */

/**
 * @brief Returns the runtime-detected SIMD capabilities as a bitmask.
 *
 * The bitmask includes flags for various SIMD instruction sets like AVX2, AVX-512,
 * ARM NEON, ARM SVE, ARM SME, RISC-V Vector, and WASM SIMD extensions.
 * Use with Capability constants to check for specific instruction sets.
 *
 * @returns {bigint} Bitmask of capability flags (use with Capability constants)
 *
 * @code{.ts}
 * import { getCapabilities, Capability } from 'numkong';
 *
 * const caps = getCapabilities();
 * console.log(`Capabilities: 0x${caps.toString(16)}`);
 *
 * // Check for specific SIMD support
 * if (caps & Capability.HASWELL) {
 *   console.log('AVX2 available');
 * }
 * @endcode
 */
export const getCapabilities = (): bigint => {
  return compiled.getCapabilities();
};

/**
 * @brief Checks if a specific SIMD capability is available at runtime.
 *
 * This is a convenience wrapper around getCapabilities() that tests for a single capability.
 *
 * @param {bigint} cap - Capability flag to check (from Capability constants)
 * @returns {boolean} True if the capability is available, false otherwise
 *
 * @code{.ts}
 * import { hasCapability, Capability } from 'numkong';
 *
 * if (hasCapability(Capability.HASWELL)) {
 *   console.log('Intel AVX2 (Haswell) available');
 * }
 * if (hasCapability(Capability.NEON)) {
 *   console.log('ARM NEON available');
 * }
 * if (hasCapability(Capability.V128RELAXED)) {
 *   console.log('WASM Relaxed SIMD available');
 * }
 * @endcode
 */
export const hasCapability = (cap: bigint): boolean => {
  return (getCapabilities() & cap) !== 0n;
};

/**
 * @brief Computes the squared Euclidean distance between two vectors.
 * @param a - The first vector.
 * @param b - The second vector (must match the type of a).
 * @param dtype - Optional dtype string for custom types (e.g. 'f16', 'bf16', 'e4m3').
 * @returns {number} The squared Euclidean distance between vectors a and b.
 */
export function sqeuclidean(a: NumericArray, b: NumericArray): number;
export function sqeuclidean(a: DistanceArray, b: DistanceArray, dtype: DType): number;
export function sqeuclidean(a: TensorBase, b: TensorBase): number;
export function sqeuclidean(a: DistanceArray | TensorBase, b: DistanceArray | TensorBase, dtype?: DType): number {
  if (a instanceof TensorBase) { const u = unwrapTensor(a), v = unwrapTensor(b as TensorBase); return compiled.sqeuclidean(u.arr, v.arr, dtypeToString(u.dtype)); }
  return dtype !== undefined ? compiled.sqeuclidean(a, b, dtypeToString(dtype)) : compiled.sqeuclidean(a, b);
}

/**
 * @brief Computes the Euclidean distance between two vectors.
 * @param a - The first vector.
 * @param b - The second vector (must match the type of a).
 * @param dtype - Optional dtype string for custom types (e.g. 'f16', 'bf16', 'e4m3').
 * @returns {number} The Euclidean distance between vectors a and b.
 */
export function euclidean(a: NumericArray, b: NumericArray): number;
export function euclidean(a: DistanceArray, b: DistanceArray, dtype: DType): number;
export function euclidean(a: TensorBase, b: TensorBase): number;
export function euclidean(a: DistanceArray | TensorBase, b: DistanceArray | TensorBase, dtype?: DType): number {
  if (a instanceof TensorBase) { const u = unwrapTensor(a), v = unwrapTensor(b as TensorBase); return compiled.euclidean(u.arr, v.arr, dtypeToString(u.dtype)); }
  return dtype !== undefined ? compiled.euclidean(a, b, dtypeToString(dtype)) : compiled.euclidean(a, b);
}

/**
 * @brief Computes the angular distance between two vectors.
 * @param a - The first vector.
 * @param b - The second vector (must match the type of a).
 * @param dtype - Optional dtype string for custom types (e.g. 'f16', 'bf16', 'e4m3').
 * @returns {number} The angular distance between vectors a and b.
 */
export function angular(a: NumericArray, b: NumericArray): number;
export function angular(a: DistanceArray, b: DistanceArray, dtype: DType): number;
export function angular(a: TensorBase, b: TensorBase): number;
export function angular(a: DistanceArray | TensorBase, b: DistanceArray | TensorBase, dtype?: DType): number {
  if (a instanceof TensorBase) { const u = unwrapTensor(a), v = unwrapTensor(b as TensorBase); return compiled.angular(u.arr, v.arr, dtypeToString(u.dtype)); }
  return dtype !== undefined ? compiled.angular(a, b, dtypeToString(dtype)) : compiled.angular(a, b);
}

/**
 * @brief Computes the inner product of two vectors (same as dot product).
 * @param a - The first vector.
 * @param b - The second vector (must match the type of a).
 * @param dtype - Optional dtype string for custom types (e.g. 'f16', 'bf16', 'e4m3').
 * @returns {number} The inner product of vectors a and b.
 */
export function inner(a: NumericArray, b: NumericArray): number;
export function inner(a: DistanceArray, b: DistanceArray, dtype: DType): number;
export function inner(a: TensorBase, b: TensorBase): number;
export function inner(a: DistanceArray | TensorBase, b: DistanceArray | TensorBase, dtype?: DType): number {
  if (a instanceof TensorBase) { const u = unwrapTensor(a), v = unwrapTensor(b as TensorBase); return compiled.inner(u.arr, v.arr, dtypeToString(u.dtype)); }
  return dtype !== undefined ? compiled.inner(a, b, dtypeToString(dtype)) : compiled.inner(a, b);
}

/**
 * @brief Computes the dot product of two vectors (same as inner product).
 * @param a - The first vector.
 * @param b - The second vector (must match the type of a).
 * @param dtype - Optional dtype string for custom types (e.g. 'f16', 'bf16', 'e4m3').
 * @returns {number} The dot product of vectors a and b.
 */
export function dot(a: NumericArray, b: NumericArray): number;
export function dot(a: DistanceArray, b: DistanceArray, dtype: DType): number;
export function dot(a: TensorBase, b: TensorBase): number;
export function dot(a: DistanceArray | TensorBase, b: DistanceArray | TensorBase, dtype?: DType): number {
  if (a instanceof TensorBase) { const u = unwrapTensor(a), v = unwrapTensor(b as TensorBase); return compiled.dot(u.arr, v.arr, dtypeToString(u.dtype)); }
  return dtype !== undefined ? compiled.dot(a, b, dtypeToString(dtype)) : compiled.dot(a, b);
}

/**
 * @brief Computes the bitwise Hamming distance between two vectors.
 *
 * Both vectors are treated as bit-packed (u1 dtype), where each byte contains 8 bits.
 * Use toBinary() to convert numeric arrays to bit-packed format.
 *
 * @param {Uint8Array | BinaryArray} a - The first bit-packed vector.
 * @param {Uint8Array | BinaryArray} b - The second bit-packed vector.
 * @returns {number} The Hamming distance (number of differing bits) between vectors a and b.
 */
export const hamming = (a: Uint8Array | BinaryArray | TensorBase, b: Uint8Array | BinaryArray | TensorBase): number => {
  if (a instanceof TensorBase) { const u = unwrapTensor(a), v = unwrapTensor(b as TensorBase); return compiled.hamming(u.arr, v.arr); }
  return compiled.hamming(a, b);
};

/**
 * @brief Computes the bitwise Jaccard distance between two vectors.
 *
 * Both vectors are treated as bit-packed (u1 dtype), where each byte contains 8 bits.
 * Use toBinary() to convert numeric arrays to bit-packed format.
 *
 * @param {Uint8Array | BinaryArray} a - The first bit-packed vector.
 * @param {Uint8Array | BinaryArray} b - The second bit-packed vector.
 * @returns {number} The Jaccard distance (1 - Jaccard similarity) between vectors a and b.
 */
export const jaccard = (a: Uint8Array | BinaryArray | TensorBase, b: Uint8Array | BinaryArray | TensorBase): number => {
  if (a instanceof TensorBase) { const u = unwrapTensor(a), v = unwrapTensor(b as TensorBase); return compiled.jaccard(u.arr, v.arr); }
  return compiled.jaccard(a, b);
};

/**
 * @brief Computes the Kullback-Leibler divergence between two probability distributions.
 *
 * Both vectors must represent valid probability distributions (non-negative, sum to 1).
 * Supports f64, f32 (auto-detected) and f16, bf16 (with explicit dtype).
 *
 * @param a - The first probability distribution.
 * @param b - The second probability distribution (must match the type of a).
 * @param dtype - Optional dtype string for custom types (e.g. 'f16', 'bf16').
 * @returns {number} The Kullback-Leibler divergence KL(a || b) = Σ a[i] * log(a[i] / b[i]).
 */
export function kullbackleibler(a: Float64Array | Float32Array, b: Float64Array | Float32Array): number;
export function kullbackleibler(a: Float64Array | Float32Array | Uint16Array, b: Float64Array | Float32Array | Uint16Array, dtype: DType): number;
export function kullbackleibler(a: TensorBase, b: TensorBase): number;
export function kullbackleibler(a: Float64Array | Float32Array | Uint16Array | TensorBase, b: Float64Array | Float32Array | Uint16Array | TensorBase, dtype?: DType): number {
  if (a instanceof TensorBase) { const u = unwrapTensor(a), v = unwrapTensor(b as TensorBase); return compiled.kullbackleibler(u.arr, v.arr, dtypeToString(u.dtype)); }
  return dtype !== undefined ? compiled.kullbackleibler(a, b, dtypeToString(dtype)) : compiled.kullbackleibler(a, b);
}

/**
 * @brief Computes the Jensen-Shannon divergence between two probability distributions.
 *
 * Both vectors must represent valid probability distributions (non-negative, sum to 1).
 * Supports f64, f32 (auto-detected) and f16, bf16 (with explicit dtype).
 * JSD is the symmetrized version of KL divergence.
 *
 * @param a - The first probability distribution.
 * @param b - The second probability distribution (must match the type of a).
 * @param dtype - Optional dtype string for custom types (e.g. 'f16', 'bf16').
 * @returns {number} The Jensen-Shannon divergence JS(a, b) = 0.5 * (KL(a || m) + KL(b || m)) where m = (a + b) / 2.
 */
export function jensenshannon(a: Float64Array | Float32Array, b: Float64Array | Float32Array): number;
export function jensenshannon(a: Float64Array | Float32Array | Uint16Array, b: Float64Array | Float32Array | Uint16Array, dtype: DType): number;
export function jensenshannon(a: TensorBase, b: TensorBase): number;
export function jensenshannon(a: Float64Array | Float32Array | Uint16Array | TensorBase, b: Float64Array | Float32Array | Uint16Array | TensorBase, dtype?: DType): number {
  if (a instanceof TensorBase) { const u = unwrapTensor(a), v = unwrapTensor(b as TensorBase); return compiled.jensenshannon(u.arr, v.arr, dtypeToString(u.dtype)); }
  return dtype !== undefined ? compiled.jensenshannon(a, b, dtypeToString(dtype)) : compiled.jensenshannon(a, b);
}

/**
 * @brief Quantizes a numeric vector into a bit-packed binary representation.
 *
 * Converts each element to a single bit: 1 for positive values, 0 for non-positive values.
 * The bits are packed into bytes (8 bits per byte) in big-endian bit order within each byte.
 * This is the required format for hamming() and jaccard() distance functions.
 *
 * @param {Float32Array | Float64Array | Int8Array} vector - The vector to quantize and pack.
 * @returns {Uint8Array} A bit-packed array where each byte contains 8 binary values.
 *
 * @code{.ts}
 * const vec = new Float32Array([1.5, -2.3, 0.0, 3.1, -1.0, 2.0, 0.5, -0.5]);
 * const binary = toBinary(vec);
 * // Result: Uint8Array([0b10010110]) = [0x96]
 * //   bits: [1, 0, 0, 1, 0, 1, 1, 0] for elements [+, -, 0, +, -, +, +, -]
 *
 * // Use with Hamming distance
 * const a = toBinary(new Float32Array([1, 2, 3]));
 * const b = toBinary(new Float32Array([1, -2, 3]));
 * const dist = hamming(a, b); // Counts differing bits
 * @endcode
 */
export const toBinary = (vector: Float32Array | Float64Array | Int8Array): Uint8Array => {
  const byteLength = Math.ceil(vector.length / 8);
  const packedVector = new Uint8Array(byteLength);

  for (let i = 0; i < vector.length; i++) {
    if (vector[i] > 0) {
      const byteIndex = Math.floor(i / 8);
      const bitPosition = 7 - (i % 8);
      packedVector[byteIndex] |= (1 << bitPosition);
    }
  }

  return packedVector;
};

/* #endregion Distance Metrics and Similarity Functions */

export default {
  dot,
  inner,
  sqeuclidean,
  euclidean,
  angular,
  hamming,
  jaccard,
  kullbackleibler,
  jensenshannon,
  toBinary,
  Float16Array,
  BFloat16Array,
  E4M3Array,
  E5M2Array,
  BinaryArray,
  TensorBase,
  VectorBase,
  VectorView,
  Vector,
  MatrixBase,
  castF16ToF32,
  castF32ToF16,
  castBF16ToF32,
  castF32ToBF16,
  castE4M3ToF32,
  castF32ToE4M3,
  castE5M2ToF32,
  castF32ToE5M2,
  cast,
};

/**
 * @brief Finds the directory where the native build of the numkong module is located.
 * @param {string} dir - The directory to start the search from.
 */
function getBuildDir(dir: string) {
  if (existsSync(path.join(dir, "build"))) return dir;
  if (existsSync(path.join(dir, "prebuilds"))) return dir;
  if (path.basename(dir) === ".next") {
    // special case for next.js on custom node (not vercel)
    const sideways = path.join(dir, "..", "node_modules", "numkong");
    if (existsSync(sideways)) return getBuildDir(sideways);
  }
  if (dir === "/") throw new Error("Could not find native build for numkong");
  return getBuildDir(path.join(dir, ".."));
}

function getDirName() {
  try {
    if (__dirname) return __dirname;
  } catch (e) { }
  return getRoot(getFileName());
}

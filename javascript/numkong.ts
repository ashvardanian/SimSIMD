import build from "node-gyp-build";
import * as path from "node:path";
import { existsSync } from "node:fs";
import { getFileName, getRoot } from "bindings";
import * as fallback from "./fallback.js";

let compiled: any;

try {
  let builddir = getBuildDir(getDirName());
  compiled = build(builddir);
} catch (e) {
  compiled = fallback;
  console.warn(
    "It seems like your environment doesn't support the native numkong module, so we are providing a JS fallback."
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
} as const;

/**
 * @brief Computes the squared Euclidean distance between two vectors.
 * @param {Float64Array|Float32Array|Int8Array|Uint8Array} a - The first vector.
 * @param {Float64Array|Float32Array|Int8Array|Uint8Array} b - The second vector.
 * @returns {number} The squared Euclidean distance between vectors a and b.
 */
export const sqeuclidean = (
  a: Float64Array | Float32Array | Int8Array | Uint8Array,
  b: Float64Array | Float32Array | Int8Array | Uint8Array
): number => {
  return compiled.sqeuclidean(a, b);
};

/**
 * @brief Computes the Euclidean distance between two vectors.
 * @param {Float64Array|Float32Array|Int8Array|Uint8Array} a - The first vector.
 * @param {Float64Array|Float32Array|Int8Array|Uint8Array} b - The second vector.
 * @returns {number} The Euclidean distance between vectors a and b.
 */
export const euclidean = (
  a: Float64Array | Float32Array | Int8Array | Uint8Array,
  b: Float64Array | Float32Array | Int8Array | Uint8Array
): number => {
  return compiled.euclidean(a, b);
};

/**
 * @brief Computes the angular distance between two vectors.
 * @param {Float64Array|Float32Array|Int8Array|Uint8Array} a - The first vector.
 * @param {Float64Array|Float32Array|Int8Array|Uint8Array} b - The second vector.
 * @returns {number} The angular distance between vectors a and b.
 */
export const angular = (
  a: Float64Array | Float32Array | Int8Array | Uint8Array,
  b: Float64Array | Float32Array | Int8Array | Uint8Array
): number => {
  return compiled.angular(a, b);
};

/**
 * @brief Computes the inner product of two vectors (same as dot product).
 * @param {Float64Array|Float32Array|Int8Array|Uint8Array} a - The first vector.
 * @param {Float64Array|Float32Array|Int8Array|Uint8Array} b - The second vector.
 * @returns {number} The inner product of vectors a and b.
 */
export const inner = (
  a: Float64Array | Float32Array | Int8Array | Uint8Array,
  b: Float64Array | Float32Array | Int8Array | Uint8Array
): number => {
  return compiled.inner(a, b);
};

/**
 * @brief Computes the dot product of two vectors (same as inner product).
 * @param {Float64Array|Float32Array|Int8Array|Uint8Array} a - The first vector.
 * @param {Float64Array|Float32Array|Int8Array|Uint8Array} b - The second vector.
 * @returns {number} The dot product of vectors a and b.
 */
export const dot = (
  a: Float64Array | Float32Array | Int8Array | Uint8Array,
  b: Float64Array | Float32Array | Int8Array | Uint8Array
): number => {
  return compiled.dot(a, b);
};

/**
 * @brief Computes the bitwise Hamming distance between two vectors.
 * @param {Uint8Array} a - The first vector.
 * @param {Uint8Array} b - The second vector.
 * @returns {number} The Hamming distance between vectors a and b.
 */
export const hamming = (a: Uint8Array, b: Uint8Array): number => {
  return compiled.hamming(a, b);
};

/**
 * @brief Computes the bitwise Jaccard distance between two vectors.
 * @param {Uint8Array} a - The first vector.
 * @param {Uint8Array} b - The second vector.
 * @returns {number} The Jaccard distance between vectors a and b.
 */
export const jaccard = (a: Uint8Array, b: Uint8Array): number => {
  return compiled.jaccard(a, b);
};

/**
 * @brief Computes the Kullback-Leibler divergence between two vectors.
 * @param {Float64Array|Float32Array} a - The first vector.
 * @param {Float64Array|Float32Array} b - The second vector.
 * @returns {number} The Kullback-Leibler divergence between vectors a and b.
 */
export const kullbackleibler = (a: Float64Array | Float32Array, b: Float64Array | Float32Array): number => {
  return compiled.kullbackleibler(a, b);
};

/**
 * @brief Computes the Jensen-Shannon divergence between two vectors.
 * @param {Float64Array|Float32Array} a - The first vector.
 * @param {Float64Array|Float32Array} b - The second vector.
 * @returns {number} The Jensen-Shannon divergence between vectors a and b.
 */
export const jensenshannon = (a: Float64Array | Float32Array, b: Float64Array | Float32Array): number => {
  return compiled.jensenshannon(a, b);
};

/**
 * Quantizes a floating-point vector into a binary vector (1 for positive values, 0 for non-positive values) and packs the result into a Uint8Array, where each element represents 8 binary values from the original vector.
 * This function is useful for preparing data for bitwise distance computations, such as Hamming or Jaccard indices.
 * 
 * @param {Float32Array | Float64Array | Int8Array} vector The floating-point vector to be quantized and packed.
 * @returns {Uint8Array} A Uint8Array where each byte represents 8 binary quantized values from the input vector.
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

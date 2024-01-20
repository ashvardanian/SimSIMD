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
    "It seems like your environment does't support the native simsimd module, so we are providing a JS fallback."
  );
}

/**
 * @brief Computes the squared Euclidean distance between two vectors.
 * @param {Float32Array|Int8Array} a - The first vector.
 * @param {Float32Array|Int8Array} b - The second vector.
 * @returns {number} The squared Euclidean distance between vectors a and b.
 */
export const sqeuclidean = (
  a: Float32Array | Int8Array,
  b: Float32Array | Int8Array
): number => {
  return compiled.sqeuclidean(a, b);
};

/**
 * @brief Computes the cosine distance between two vectors.
 * @param {Float32Array|Int8Array} a - The first vector.
 * @param {Float32Array|Int8Array} b - The second vector.
 * @returns {number} The cosine distance between vectors a and b.
 */
export const cosine = (
  a: Float32Array | Int8Array,
  b: Float32Array | Int8Array
): number => {
  return compiled.cosine(a, b);
};

/**
 * @brief Computes the inner product of two vectors.
 * @param {Float32Array} a - The first vector.
 * @param {Float32Array} b - The second vector.
 * @returns {number} The inner product of vectors a and b.
 */
export const inner = (a: Float32Array, b: Float32Array): number => {
  return compiled.inner(a, b);
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
 * @brief Computes the bitwise Jaccard similarity coefficient between two vectors.
 * @param {Uint8Array} a - The first vector.
 * @param {Uint8Array} b - The second vector.
 * @returns {number} The Jaccard similarity coefficient between vectors a and b.
 */
export const jaccard = (a: Uint8Array, b: Uint8Array): number => {
  return compiled.jaccard(a, b);
};

/**
 * @brief Computes the kullbackleibler similarity coefficient between two vectors.
 * @param {Float32Array} a - The first vector.
 * @param {Float32Array} b - The second vector.
 * @returns {number} The Jaccard similarity coefficient between vectors a and b.
 */
export const kullbackleibler = (a: Float32Array, b: Float32Array): number => {
  return compiled.kullbackleibler(a, b);
};

/**
 * @brief Computes the jensenshannon similarity coefficient between two vectors.
 * @param {Float32Array} a - The first vector.
 * @param {Float32Array} b - The second vector.
 * @returns {number} The Jaccard similarity coefficient between vectors a and b.
 */
export const jensenshannon = (a: Float32Array, b: Float32Array): number => {
  return compiled.jensenshannon(a, b);
};

export default {
  sqeuclidean,
  cosine,
  inner,
  hamming,
  jaccard,
  kullbackleibler,
  jensenshannon,
};

// utility functions to help find native builds

function getBuildDir(dir: string) {
  if (existsSync(path.join(dir, "build"))) return dir;
  if (existsSync(path.join(dir, "prebuilds"))) return dir;
  if (path.basename(dir) === ".next") {
    // special case for next.js on custom node (not vercel)
    const sideways = path.join(dir, "..", "node_modules", "simsimd");
    if (existsSync(sideways)) return getBuildDir(sideways);
  }
  if (dir === "/") throw new Error("Could not find native build for simsimd");
  return getBuildDir(path.join(dir, ".."));
}

function getDirName() {
  try {
    if (__dirname) return __dirname;
  } catch (e) {}
  return getRoot(getFileName());
}

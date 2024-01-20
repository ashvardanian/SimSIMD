console.warn(
  "It seems like your environment does't support the native simsimd module, so we are providing a JS fallback."
);

/**
 * @brief Computes the inner product of two vectors.
 * @param {Float32Array} a - The first vector.
 * @param {Float32Array} b - The second vector.
 * @returns {number} The inner product of vectors a and b.
 */
export function inner(a: Float32Array, b: Float32Array): number {
  if (a.length !== b.length) {
    throw new Error("Vectors must have the same length");
  }

  let result = 0;
  for (let i = 0; i < a.length; i++) {
    result += a[i] * b[i];
  }
  return 1 - result;
}

/**
 * @brief Computes the squared Euclidean distance between two vectors.
 * @param {Float32Array|Int8Array} a - The first vector.
 * @param {Float32Array|Int8Array} b - The second vector.
 * @returns {number} The squared Euclidean distance between vectors a and b.
 */
export function sqeuclidean(
  a: Float32Array | Int8Array,
  b: Float32Array | Int8Array
): number {
  if (a.length !== b.length) {
    throw new Error("Vectors must have the same length");
  }

  let result = 0;
  for (let i = 0; i < a.length; i++) {
    result += (a[i] - b[i]) * (a[i] - b[i]);
  }
  return result;
}

/**
 * @brief Computes the cosine distance between two vectors.
 * @param {Float32Array|Int8Array} a - The first vector.
 * @param {Float32Array|Int8Array} b - The second vector.
 * @returns {number} The cosine distance between vectors a and b.
 */
export function cosine(
  a: Float32Array | Int8Array,
  b: Float32Array | Int8Array
): number {
  if (a.length !== b.length) {
    throw new Error("Vectors must have the same length");
  }

  let dotProduct = 0;
  let magnitudeA = 0;
  let magnitudeB = 0;

  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i];
    magnitudeA += a[i] * a[i];
    magnitudeB += b[i] * b[i];
  }

  magnitudeA = Math.sqrt(magnitudeA);
  magnitudeB = Math.sqrt(magnitudeB);

  if (magnitudeA === 0 || magnitudeB === 0) {
    console.warn(
      "Warning: One of the magnitudes is zero. Cosine similarity is undefined."
    );
    return 0;
  }

  return 1 - dotProduct / (magnitudeA * magnitudeB);
}

/**
 * @brief Computes the bitwise Hamming distance between two vectors.
 * @param {Uint8Array} a - The first vector.
 * @param {Uint8Array} b - The second vector.
 * @returns {number} The Hamming distance between vectors a and b.
 */
export const hamming = (a: Uint8Array, b: Uint8Array): number => {
  throw new Error("Not implemented");
};

/**
 * @brief Computes the bitwise Jaccard similarity coefficient between two vectors.
 * @param {Uint8Array} a - The first vector.
 * @param {Uint8Array} b - The second vector.
 * @returns {number} The Jaccard similarity coefficient between vectors a and b.
 */
export const jaccard = (a: Uint8Array, b: Uint8Array): number => {
  throw new Error("Not implemented");
};

/**
 * @brief Computes the kullbackleibler similarity coefficient between two vectors.
 * @param {Float32Array} a - The first vector.
 * @param {Float32Array} b - The second vector.
 * @returns {number} The Jaccard similarity coefficient between vectors a and b.
 */
export const kullbackleibler = (a: Float32Array, b: Float32Array): number => {
  throw new Error("Not implemented");
};

/**
 * @brief Computes the jensenshannon similarity coefficient between two vectors.
 * @param {Float32Array} a - The first vector.
 * @param {Float32Array} b - The second vector.
 * @returns {number} The Jaccard similarity coefficient between vectors a and b.
 */
export const jensenshannon = (a: Float32Array, b: Float32Array): number => {
  throw new Error("Not implemented");
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

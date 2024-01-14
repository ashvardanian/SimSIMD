/**
 * @brief Computes the squared Euclidean distance between two vectors.
 * @param {Float32Array|Int8Array} a - The first vector.
 * @param {Float32Array|Int8Array} b - The second vector.
 * @returns {number} The squared Euclidean distance between vectors a and b.
 */
export declare const sqeuclidean: (a: Float32Array | Int8Array, b: Float32Array | Int8Array) => number;
/**
 * @brief Computes the cosine similarity distance between two vectors.
 * @param {Float32Array|Int8Array} a - The first vector.
 * @param {Float32Array|Int8Array} b - The second vector.
 * @returns {number} The cosine similarity distance between vectors a and b.
 */
export declare const cosine: (a: Float32Array | Int8Array, b: Float32Array | Int8Array) => number;
/**
 * @brief Computes the inner product of two vectors.
 * @param {Float32Array} a - The first vector.
 * @param {Float32Array} b - The second vector.
 * @returns {number} The inner product of vectors a and b.
 */
export declare const inner: (a: Float32Array, b: Float32Array) => number;
/**
 * @brief Computes the bitwise Hamming distance between two vectors.
 * @param {Uint8Array} a - The first vector.
 * @param {Uint8Array} b - The second vector.
 * @returns {number} The Hamming distance between vectors a and b.
 */
export declare const hamming: (a: Uint8Array, b: Uint8Array) => number;
/**
 * @brief Computes the bitwise Jaccard similarity coefficient between two vectors.
 * @param {Uint8Array} a - The first vector.
 * @param {Uint8Array} b - The second vector.
 * @returns {number} The Jaccard similarity coefficient between vectors a and b.
 */
export declare const jaccard: (a: Uint8Array, b: Uint8Array) => number;

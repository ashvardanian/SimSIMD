const build = require('node-gyp-build');
const path = require('path');
const compiled = build(path.resolve(__dirname, '..'));

module.exports = {

    /**
     * @brief Computes the squared Euclidean distance between two vectors.
     * @param {Float32Array|Int8Array} a - The first vector.
     * @param {Float32Array|Int8Array} b - The second vector.
     * @returns {number} The squared Euclidean distance between vectors a and b.
     */
    sqeuclidean: compiled.sqeuclidean,

    /**
     * @brief Computes the cosine similarity between two vectors.
     * @param {Float32Array|Int8Array} a - The first vector.
     * @param {Float32Array|Int8Array} b - The second vector.
     * @returns {number} The cosine similarity between vectors a and b.
     */
    cosine: compiled.cosine,

    /**
     * @brief Computes the inner product of two vectors.
     * @param {Float32Array} a - The first vector.
     * @param {Float32Array} b - The second vector.
     * @returns {number} The inner product of vectors a and b.
     */
    inner: compiled.inner,

    /**
     * @brief Computes the bitwise Hamming distance between two vectors.
     * @param {Uint8Array} a - The first vector.
     * @param {Uint8Array} b - The second vector.
     * @returns {number} The Hamming distance between vectors a and b.
     */
    hamming: compiled.hamming,

    /**
     * @brief Computes the bitwise Jaccard similarity coefficient between two vectors.
     * @param {Uint8Array} a - The first vector.
     * @param {Uint8Array} b - The second vector.
     * @returns {number} The Jaccard similarity coefficient between vectors a and b.
     */
    jaccard: compiled.jaccard,

};

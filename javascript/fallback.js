function fallbackWarning() {
  console.warn(
    "It seems like your environment does't support the native simsimd module, so we are providing a JS fallback."
  );
}

function inner_distance(arr1, arr2) {
  fallbackWarning();
  return 1 - arr1.reduce((acc, val, i) => acc + val * arr2[i], 0);
}

function sqeuclidean(arr1, arr2) {
  fallbackWarning();
  return Math.hypot(...Object.keys(arr1).map((k) => arr2[k] - arr1[k])) ** 2;
}

function cosine(arr1, arr2) {
  fallbackWarning();
  let dotProduct = arr1.reduce((acc, val, i) => acc + val * arr2[i], 0);
  let magnitudeA = Math.sqrt(arr1.reduce((acc, val) => acc + val * val, 0));
  let magnitudeB = Math.sqrt(arr2.reduce((acc, val) => acc + val * val, 0));
  return dotProduct / (magnitudeA * magnitudeB);
}

module.exports = {
  /**
   * Computes the inner distance between two arrays.
   * @param {number[]} arr1 - The first array.
   * @param {number[]} arr2 - The second array.
   * @returns {number} The inner distance between arr1 and arr2.
   */
  inner_distance: inner_distance,

  /**
   * Computes the squared Euclidean distance between two arrays.
   * @param {number[]} arr1 - The first array.
   * @param {number[]} arr2 - The second array.
   * @returns {number} The squared Euclidean distance between arr1 and arr2.
   */
  sqeuclidean: sqeuclidean,
  /**
   * Computes the cosine similarity between two arrays.
   * @param {number[]} arr1 - The first array.
   * @param {number[]} arr2 - The second array.
   * @returns {number} The cosine similarity between arr1 and arr2.
   */
  cosine: cosine,
};

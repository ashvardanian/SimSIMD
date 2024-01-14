function fallbackWarning() {
  console.warn(
    "It seems like your environment does't support the native simsimd module, so we are providing a JS fallback."
  );
}

function inner_distance(arr1, arr2) {
  fallbackWarning();
  if (arr1.length !== arr2.length) {
    throw new Error("Vectors must have the same length");
  }

  return 1 - arr1.reduce((acc, val, i) => acc + val * arr2[i], 0);
}

function sqeuclidean(arr1, arr2) {
  fallbackWarning();
  if (arr1.length !== arr2.length) {
    throw new Error("Vectors must have the same length");
  }

  let sum = 0;
  for (let i = 0; i < arr1.length; i++) {
    sum += (arr2[i] - arr1[i]) ** 2;
  }
  return sum;
}

function cosine(arr1, arr2) {
  fallbackWarning();
  if (arr1.length !== arr2.length) {
    throw new Error("Vectors must have the same length");
  }

  const dotProduct = arr1.reduce((acc, val, i) => acc + val * arr2[i], 0);
  const magnitudeA = Math.sqrt(arr1.reduce((acc, val) => acc + val ** 2, 0));
  const magnitudeB = Math.sqrt(arr2.reduce((acc, val) => acc + val ** 2, 0));

  if (magnitudeA === 0 || magnitudeB === 0) {
    console.warn(
      "Warning: One of the magnitudes is zero. Cosine similarity is undefined."
    );
    return 0;
  }

  return 1 - dotProduct / (magnitudeA * magnitudeB);
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
   * Computes the cosine similarity distance between two arrays.
   * @param {number[]} arr1 - The first array.
   * @param {number[]} arr2 - The second array.
   * @returns {number} The cosine similarity distance between arr1 and arr2.
   */
  cosine: cosine,
};

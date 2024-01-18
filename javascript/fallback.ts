console.warn(
  "It seems like your environment does't support the native simsimd module, so we are providing a JS fallback."
);

/**
 * Computes the inner distance between two arrays.
 * @param {number[]} arr1 - The first array.
 * @param {number[]} arr2 - The second array.
 * @returns {number} The inner distance between arr1 and arr2.
 */
export function inner_distance(arr1: number[], arr2: number[]): number {
  if (arr1.length !== arr2.length) {
    throw new Error("Vectors must have the same length");
  }

  let result = 0;
  for (let i = 0; i < arr1.length; i++) {
    result += arr1[i] * arr2[i];
  }
  return 1 - result;
}

/**
 * Computes the squared Euclidean distance between two arrays.
 * @param {number[]} arr1 - The first array.
 * @param {number[]} arr2 - The second array.
 * @returns {number} The squared Euclidean distance between arr1 and arr2.
 */
export function sqeuclidean(arr1: number[], arr2: number[]): number {
  if (arr1.length !== arr2.length) {
    throw new Error("Vectors must have the same length");
  }

  let result = 0;
  for (let i = 0; i < arr1.length; i++) {
    result += (arr1[i] - arr2[i]) * (arr1[i] - arr2[i]);
  }
  return result;
}

/**
 * Computes the cosine distance between two arrays.
 * @param {number[]} arr1 - The first array.
 * @param {number[]} arr2 - The second array.
 * @returns {number} The cosine distance between arr1 and arr2.
 */
export function cosine(arr1: number[], arr2: number[]): number {
  if (arr1.length !== arr2.length) {
    throw new Error("Vectors must have the same length");
  }

  let dotProduct = 0;
  let magnitudeA = 0;
  let magnitudeB = 0;

  for (let i = 0; i < arr1.length; i++) {
    dotProduct += arr1[i] * arr2[i];
    magnitudeA += arr1[i] * arr1[i];
    magnitudeB += arr2[i] * arr2[i];
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

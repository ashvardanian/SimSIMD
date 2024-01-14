/**
 * Computes the inner distance between two arrays.
 * @param {number[]} arr1 - The first array.
 * @param {number[]} arr2 - The second array.
 * @returns {number} The inner distance between arr1 and arr2.
 */
export function inner_distance(arr1: number[], arr2: number[]): number {
  return 1 - arr1.reduce((acc, val, i) => acc + val * arr2[i], 0);
}

/**
 * Computes the squared Euclidean distance between two arrays.
 * @param {number[]} arr1 - The first array.
 * @param {number[]} arr2 - The second array.
 * @returns {number} The squared Euclidean distance between arr1 and arr2.
 */
export function sqeuclidean(arr1: number[], arr2: number[]): number {
  return Math.hypot(...Object.keys(arr1).map((k: any) => arr2[k] - arr1[k])) ** 2;
}

/**
 * Computes the cosine similarity between two arrays.
 * @param {number[]} arr1 - The first array.
 * @param {number[]} arr2 - The second array.
 * @returns {number} The cosine similarity between arr1 and arr2.
 */
export function cosine(arr1: number[], arr2: number[]) {
  let dotProduct = arr1.reduce((acc, val, i) => acc + val * arr2[i], 0);
  let magnitudeA = Math.sqrt(arr1.reduce((acc, val) => acc + val * val, 0));
  let magnitudeB = Math.sqrt(arr2.reduce((acc, val) => acc + val * val, 0));
  return dotProduct / (magnitudeA * magnitudeB);
}

/**
 * Computes the Hamming distance between two Uint8Arrays.
 * @param {Uint8Array} arr1 - The first Uint8Array.
 * @param {Uint8Array} arr2 - The second Uint8Array.
 * @returns {number} The Hamming distance between arr1 and arr2.
 */
function hamming(arr1: Uint8Array, arr2: Uint8Array): number {
  let distance = 0;
  for (let i = 0; i < arr1.length; i++) {
    let xor = arr1[i] ^ arr2[i];
    for (let j = 0; j < 8; j++) {
      distance += (xor >> j) & 1;
    }
  }
  return distance;
}


module.exports {
  
}
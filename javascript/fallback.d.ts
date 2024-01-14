/**
 * Computes the inner distance between two arrays.
 * @param {number[]} arr1 - The first array.
 * @param {number[]} arr2 - The second array.
 * @returns {number} The inner distance between arr1 and arr2.
 */
export declare function inner_distance(arr1: number[], arr2: number[]): number;

/**
 * Computes the squared Euclidean distance between two arrays.
 * @param {number[]} arr1 - The first array.
 * @param {number[]} arr2 - The second array.
 * @returns {number} The squared Euclidean distance between arr1 and arr2.
 */
export declare function sqeuclidean(arr1: number[], arr2: number[]): number;

/**
 * Computes the cosine similarity distance between two arrays.
 * @param {number[]} arr1 - The first array.
 * @param {number[]} arr2 - The second array.
 * @returns {number} The cosine similarity distance between arr1 and arr2.
 */
export declare function cosine(arr1: number[], arr2: number[]): number;

/** 
* Computes the Hamming distance between two Uint8Arrays.
* @param {Uint8Array} arr1 - The first Uint8Array.
* @param {Uint8Array} arr2 - The second Uint8Array.
* @returns {number} The Hamming distance between arr1 and arr2.
*/
export declare function hamming(arr1: Uint8Array, arr2: Uint8Array): number;

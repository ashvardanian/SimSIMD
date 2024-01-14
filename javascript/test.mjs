import test from "node:test";
import assert from "node:assert";
import build from "node-gyp-build";
import process from "node:process";

const simsimd = build(process.cwd());

import { sqeuclidean, cosine, inner_distance } from "./fallback.js";

function assertAlmostEqual(actual, expected, tolerance = 1e-6) {
  const lowerBound = expected - tolerance;
  const upperBound = expected + tolerance;
  assert(
    actual >= lowerBound && actual <= upperBound,
    `Expected ${actual} to be almost equal to ${expected}`
  );
}

test("Distance from itself", () => {
  const f32s = new Float32Array([1.0, 2.0, 3.0]);
  assertAlmostEqual(simsimd.sqeuclidean(f32s, f32s), 0.0, 0.01);
  assertAlmostEqual(simsimd.cosine(f32s, f32s), 0.0, 0.01);

  const f32sNormalized = new Float32Array([
    1.0 / Math.sqrt(14),
    2.0 / Math.sqrt(14),
    3.0 / Math.sqrt(14),
  ]);
  assertAlmostEqual(simsimd.inner(f32sNormalized, f32sNormalized), 0.0, 0.01);

  const f32sDistribution = new Float32Array([1.0 / 6, 2.0 / 6, 3.0 / 6]);
  assertAlmostEqual(
    simsimd.kullbackleibler(f32sDistribution, f32sDistribution),
    0.0,
    0.01
  );
  assertAlmostEqual(
    simsimd.jensenshannon(f32sDistribution, f32sDistribution),
    0.0,
    0.01
  );

  const u8s = new Uint8Array([1, 2, 3]);
  assertAlmostEqual(simsimd.hamming(u8s, u8s), 0.0, 0.01);
  assertAlmostEqual(simsimd.jaccard(u8s, u8s), 0.0, 0.01);
});

test("Distance from itself JS", () => {
  const arr = [1.0, 2.0, 3.0];
  assertAlmostEqual(sqeuclidean(arr, arr), 0.0, 0.01);
  assertAlmostEqual(cosine(arr, arr), 0.0, 0.01);
//   assertAlmostEqual(inner_distance(arr, arr), 0.0, 0.01);

  // const f32sNormalized = new Float32Array([1.0 / Math.sqrt(14), 2.0 / Math.sqrt(14), 3.0 / Math.sqrt(14)]);
  // assertAlmostEqual(simsimd.inner(f32sNormalized, f32sNormalized), 0.0, 0.01);

  // const f32sDistribution = new Float32Array([1.0 / 6, 2.0 / 6, 3.0 / 6]);
  // assertAlmostEqual(simsimd.kullbackleibler(f32sDistribution, f32sDistribution), 0.0, 0.01);
  // assertAlmostEqual(simsimd.jensenshannon(f32sDistribution, f32sDistribution), 0.0, 0.01);

  // const u8s = new Uint8Array([1, 2, 3]);
  // assertAlmostEqual(simsimd.hamming(u8s, u8s), 0.0, 0.01);
  // assertAlmostEqual(simsimd.jaccard(u8s, u8s), 0.0, 0.01);
});

const f32Array1 = new Float32Array([1.0, 2.0, 3.0]);
const f32Array2 = new Float32Array([4.0, 5.0, 6.0]);

const fallbackArray1 = [1.0, 2.0, 3.0];
const fallbackArray2 = [4.0, 5.0, 6.0];

test("Squared Euclidean Distance", () => {
  const result = simsimd.sqeuclidean(f32Array1, f32Array2);
  assertAlmostEqual(result, 27.0, 0.01);
});

test("Inner Product", () => {
  const result = simsimd.inner(f32Array1, f32Array2);
  assertAlmostEqual(result, -31.0, 0.01);
});

test("Cosine Similarity", () => {
  const result = simsimd.cosine(f32Array1, f32Array2);
  assertAlmostEqual(result, 0.029, 0.01);
});

test("Squared Euclidean Distance JS", () => {
  const result = sqeuclidean(fallbackArray1, fallbackArray2);
  assertAlmostEqual(result, 27.0, 0.01);
});

test("Inner Product JS", () => {
  const result = inner_distance(fallbackArray1, fallbackArray2);
  assertAlmostEqual(result, -31.0, 0.01);
});

test("Cosine Similarity JS", () => {
  const result = cosine(fallbackArray1, fallbackArray2);
  assertAlmostEqual(result, 0.029, 0.01);
});

test("Squared Euclidean Distance JS vs C", () => {
  const result = simsimd.sqeuclidean(f32Array1, f32Array2);
  const resultjs = sqeuclidean(fallbackArray1, fallbackArray2);
  assertAlmostEqual(result, resultjs, 0.01);
});

test("Inner Distance JS vs C", () => {
  const result = simsimd.inner(f32Array1, f32Array2);
  const resultjs = inner_distance(fallbackArray1, fallbackArray2);
  assertAlmostEqual(result, resultjs, 0.01);
});

test("Cosine Similarity JS vs C", () => {
  const result = simsimd.cosine(f32Array1, f32Array2);
  const resultjs = cosine(fallbackArray1, fallbackArray2);
  assertAlmostEqual(result, resultjs, 0.01);
});

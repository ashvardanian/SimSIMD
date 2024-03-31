// Currently the builds are expected to run only on Node.js,
// but Deno tests pass as well.
//
// Bun supports `node:assert`, but not `node:test`.
// Using `require` we can make the tests compatible with Bun.
//
//    const isBun = typeof Bun !== "undefined";
//    let assert, test;
//    if (isBun) {
//      assert = require('node:assert');
//      test = require('bun:test');
//    } else {
//      assert = require('node:assert');
//      test = require('node:test');
//    }
//
// That, however, leads to other issues, like the following:
//
//    require is not defined in ES module scope, you can use import instead
//
// https://bun.sh/docs/runtime/nodejs-apis
// https://bun.sh/guides/util/detect-bun
import test from "node:test";
import assert from "node:assert";

import * as simsimd from "./dist/esm/simsimd.js";

import * as fallback from "./dist/esm/fallback.js";

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

  const f64s = new Float64Array([1.0, 2.0, 3.0]);
  assertAlmostEqual(simsimd.sqeuclidean(f64s, f64s), 0.0, 0.01);
  assertAlmostEqual(simsimd.cosine(f64s, f64s), 0.0, 0.01);

  const f32sNormalized = new Float32Array([
    1.0 / Math.sqrt(14),
    2.0 / Math.sqrt(14),
    3.0 / Math.sqrt(14),
  ]);
  assertAlmostEqual(simsimd.inner(f32sNormalized, f32sNormalized), 1.0, 0.01);

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

test("Distance from itself JS fallback", () => {
  const f32s = new Float32Array([1.0, 2.0, 3.0]);

  assertAlmostEqual(fallback.sqeuclidean(f32s, f32s), 0.0, 0.01);
  assertAlmostEqual(fallback.cosine(f32s, f32s), 0.0, 0.01);

  const arrNormalized = new Float32Array([
    1.0 / Math.sqrt(14),
    2.0 / Math.sqrt(14),
    3.0 / Math.sqrt(14),
  ]);
  assertAlmostEqual(fallback.inner(arrNormalized, arrNormalized), 1.0, 0.01);

  const f32sDistribution = new Float32Array([1.0 / 6, 2.0 / 6, 3.0 / 6]);
  assertAlmostEqual(
    fallback.kullbackleibler(f32sDistribution, f32sDistribution),
    0.0,
    0.01
  );
  assertAlmostEqual(
    fallback.jensenshannon(f32sDistribution, f32sDistribution),
    0.0,
    0.01
  );

  const u8s = new Uint8Array([1, 2, 3]);
  assertAlmostEqual(fallback.hamming(u8s, u8s), 0.0, 0.01);
  assertAlmostEqual(fallback.jaccard(u8s, u8s), 0.0, 0.01);
});

const f32Array1 = new Float32Array([1.0, 2.0, 3.0]);
const f32Array2 = new Float32Array([4.0, 5.0, 6.0]);

test("Squared Euclidean Distance", () => {
  const result = simsimd.sqeuclidean(f32Array1, f32Array2);
  assertAlmostEqual(result, 27.0, 0.01);
});

test("Inner Distance", () => {
  const result = simsimd.inner(f32Array1, f32Array2);
  assertAlmostEqual(result, 32.0, 0.01);
});

test("Cosine Similarity", () => {
  const result = simsimd.cosine(f32Array1, f32Array2);
  assertAlmostEqual(result, 0.029, 0.01);
});

test("Squared Euclidean Distance JS", () => {
  const result = fallback.sqeuclidean(f32Array1, f32Array2);
  assertAlmostEqual(result, 27.0, 0.01);
});

test("Inner Distance JS", () => {
  const result = fallback.inner(f32Array1, f32Array2);
  assertAlmostEqual(result, 32.0, 0.01);
});

test("Cosine Similarity JS", () => {
  const result = fallback.cosine(f32Array1, f32Array2);
  assertAlmostEqual(result, 0.029, 0.01);
});

test("Squared Euclidean Distance C vs JS", () => {
  const result = simsimd.sqeuclidean(f32Array1, f32Array2);
  const resultjs = fallback.sqeuclidean(f32Array1, f32Array2);
  assertAlmostEqual(resultjs, result, 0.01);
});

test("Inner Distance C vs JS", () => {
  const result = simsimd.inner(f32Array1, f32Array2);
  const resultjs = fallback.inner(f32Array1, f32Array2);
  assertAlmostEqual(resultjs, result, 0.01);
});

test("Cosine Similarity C vs JS", () => {
  const result = simsimd.cosine(f32Array1, f32Array2);
  const resultjs = fallback.cosine(f32Array1, f32Array2);
  assertAlmostEqual(resultjs, result, 0.01);
});

test("Hamming C vs JS", () => {
  const u8s = new Uint8Array([1, 2, 3]);
  const result = simsimd.hamming(u8s, u8s);
  const resultjs = fallback.hamming(u8s, u8s);
  assertAlmostEqual(resultjs, result, 0.01);
});

test("Jaccard C vs JS", () => {
  const u8s = new Uint8Array([1, 2, 3]);
  const result = simsimd.jaccard(u8s, u8s);
  const resultjs = fallback.jaccard(u8s, u8s);
  assertAlmostEqual(resultjs, result, 0.01);
});

test("Kullback-Leibler C vs JS", () => {
  const f32sDistribution = new Float32Array([1.0 / 6, 2.0 / 6, 3.0 / 6]);
  const result = simsimd.kullbackleibler(f32sDistribution, f32sDistribution);
  const resultjs = fallback.kullbackleibler(f32sDistribution, f32sDistribution);
  assertAlmostEqual(resultjs, result, 0.01);
});

test("Jensen-Shannon C vs JS", () => {
  const f32sDistribution = new Float32Array([1.0 / 6, 2.0 / 6, 3.0 / 6]);
  const result = simsimd.jensenshannon(f32sDistribution, f32sDistribution);
  const resultjs = fallback.jensenshannon(f32sDistribution, f32sDistribution);
  assertAlmostEqual(resultjs, result, 0.01);
});

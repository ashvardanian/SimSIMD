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

import * as numkong from "../javascript/dist/esm/numkong.js";

function assertAlmostEqual(actual, expected, tolerance = 1e-6) {
  const lowerBound = expected - tolerance;
  const upperBound = expected + tolerance;
  assert(actual >= lowerBound && actual <= upperBound, `Expected ${actual} to be almost equal to ${expected}`);
}

test("Distance from itself", () => {
  const f32s = new Float32Array([1.0, 2.0, 3.0]);
  assertAlmostEqual(numkong.sqeuclidean(f32s, f32s), 0.0, 0.01);
  assertAlmostEqual(numkong.angular(f32s, f32s), 0.0, 0.01);

  const f64s = new Float64Array([1.0, 2.0, 3.0]);
  assertAlmostEqual(numkong.sqeuclidean(f64s, f64s), 0.0, 0.01);
  assertAlmostEqual(numkong.angular(f64s, f64s), 0.0, 0.01);

  const f32sNormalized = new Float32Array([1 / Math.sqrt(14), 2 / Math.sqrt(14), 3 / Math.sqrt(14)]);
  assertAlmostEqual(numkong.inner(f32sNormalized, f32sNormalized), 1.0, 0.01);

  const f32sHistogram = new Float32Array([1.0 / 6, 2.0 / 6, 3.0 / 6]);
  assertAlmostEqual(numkong.kullbackleibler(f32sHistogram, f32sHistogram), 0.0, 0.01);
  assertAlmostEqual(numkong.jensenshannon(f32sHistogram, f32sHistogram), 0.0, 0.01);

  const u8s = new Uint8Array([1, 2, 3]);
  assertAlmostEqual(numkong.hamming(u8s, u8s), 0.0, 0.01);
  assertAlmostEqual(numkong.jaccard(u8s, u8s), 0.0, 0.01);
});

test("Squared Euclidean Distance", () => {
  const f64sOne = new Float64Array([1.0, 2.0, 3.0]);
  const f64sTwo = new Float64Array([4.0, 5.0, 6.0]);
  assertAlmostEqual(numkong.sqeuclidean(f64sOne, f64sTwo), 27.0, 0.01);

  const f32sOne = new Float32Array([1.0, 2.0, 3.0]);
  const f32sTwo = new Float32Array([4.0, 5.0, 6.0]);
  assertAlmostEqual(numkong.sqeuclidean(f32sOne, f32sTwo), 27.0, 0.01);

  const u8sOne = new Uint8Array([1, 2, 3]);
  const u8sTwo = new Uint8Array([4, 5, 6]);
  assertAlmostEqual(numkong.sqeuclidean(u8sOne, u8sTwo), 27.0, 0.01);

  const i8sOne = new Int8Array([1, 2, 3]);
  const i8sTwo = new Int8Array([4, 5, 6]);
  assertAlmostEqual(numkong.sqeuclidean(i8sOne, i8sTwo), 27.0, 0.01);
});

test("Euclidean Distance", () => {
  const f64sOne = new Float64Array([1.0, 2.0, 3.0]);
  const f64sTwo = new Float64Array([4.0, 5.0, 6.0]);
  assertAlmostEqual(numkong.euclidean(f64sOne, f64sTwo), 5.2, 0.01);

  const f32sOne = new Float32Array([1.0, 2.0, 3.0]);
  const f32sTwo = new Float32Array([4.0, 5.0, 6.0]);
  assertAlmostEqual(numkong.euclidean(f32sOne, f32sTwo), 5.2, 0.01);

  const u8sOne = new Uint8Array([1, 2, 3]);
  const u8sTwo = new Uint8Array([4, 5, 6]);
  assertAlmostEqual(numkong.euclidean(u8sOne, u8sTwo), 5.2, 0.01);

  const i8sOne = new Int8Array([1, 2, 3]);
  const i8sTwo = new Int8Array([4, 5, 6]);
  assertAlmostEqual(numkong.euclidean(i8sOne, i8sTwo), 5.2, 0.01);
});

test("Inner Product", () => {
  const f64sOne = new Float64Array([1.0, 2.0, 3.0]);
  const f64sTwo = new Float64Array([4.0, 5.0, 6.0]);
  assertAlmostEqual(numkong.inner(f64sOne, f64sTwo), 32.0, 0.01);

  const f32sOne = new Float32Array([1.0, 2.0, 3.0]);
  const f32sTwo = new Float32Array([4.0, 5.0, 6.0]);
  assertAlmostEqual(numkong.inner(f32sOne, f32sTwo), 32.0, 0.01);

  const u8sOne = new Uint8Array([1, 2, 3]);
  const u8sTwo = new Uint8Array([4, 5, 6]);
  assertAlmostEqual(numkong.inner(u8sOne, u8sTwo), 32.0, 0.01);

  const i8sOne = new Int8Array([1, 2, 3]);
  const i8sTwo = new Int8Array([4, 5, 6]);
  assertAlmostEqual(numkong.inner(i8sOne, i8sTwo), 32.0, 0.01);
});

test("Angular Distance", () => {
  const f64sOne = new Float64Array([1.0, 2.0, 3.0]);
  const f64sTwo = new Float64Array([4.0, 5.0, 6.0]);
  assertAlmostEqual(numkong.angular(f64sOne, f64sTwo), 0.03, 0.01);

  const f32sOne = new Float32Array([1.0, 2.0, 3.0]);
  const f32sTwo = new Float32Array([4.0, 5.0, 6.0]);
  assertAlmostEqual(numkong.angular(f32sOne, f32sTwo), 0.03, 0.01);

  const u8sOne = new Uint8Array([1, 2, 3]);
  const u8sTwo = new Uint8Array([4, 5, 6]);
  assertAlmostEqual(numkong.angular(u8sOne, u8sTwo), 0.03, 0.01);

  const i8sOne = new Int8Array([1, 2, 3]);
  const i8sTwo = new Int8Array([4, 5, 6]);
  assertAlmostEqual(numkong.angular(i8sOne, i8sTwo), 0.03, 0.01);
});

test("Kullback-Leibler", () => {
  const f64sOne = new Float64Array([1.0 / 6, 2.0 / 6, 3.0 / 6]);
  const f64sTwo = new Float64Array([4.0 / 17, 5.0 / 17, 6.0 / 17]);
  assertAlmostEqual(numkong.kullbackleibler(f64sOne, f64sTwo), 0.16, 0.01);

  const f32sOne = new Float32Array([1.0 / 6, 2.0 / 6, 3.0 / 6]);
  const f32sTwo = new Float32Array([4.0 / 17, 5.0 / 17, 6.0 / 17]);
  assertAlmostEqual(numkong.kullbackleibler(f32sOne, f32sTwo), 0.16, 0.01);
});

test("Jensen-Shannon", () => {
  const f64sOne = new Float64Array([1.0 / 6, 2.0 / 6, 3.0 / 6]);
  const f64sTwo = new Float64Array([4.0 / 17, 5.0 / 17, 6.0 / 17]);
  assertAlmostEqual(numkong.jensenshannon(f64sOne, f64sTwo), 0.095, 0.01);

  const f32sOne = new Float32Array([1.0 / 6, 2.0 / 6, 3.0 / 6]);
  const f32sTwo = new Float32Array([4.0 / 17, 5.0 / 17, 6.0 / 17]);
  assertAlmostEqual(numkong.jensenshannon(f32sOne, f32sTwo), 0.095, 0.01);
});

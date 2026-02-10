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
import * as fallback from "../javascript/dist/esm/fallback.js";

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
  assertAlmostEqual(numkong.sqeuclidean(f32s, f32s), 0.0, 0.01);
  assertAlmostEqual(numkong.angular(f32s, f32s), 0.0, 0.01);
  assertAlmostEqual(fallback.sqeuclidean(f32s, f32s), 0.0, 0.01);
  assertAlmostEqual(fallback.angular(f32s, f32s), 0.0, 0.01);

  const f64s = new Float64Array([1.0, 2.0, 3.0]);
  assertAlmostEqual(numkong.sqeuclidean(f64s, f64s), 0.0, 0.01);
  assertAlmostEqual(numkong.angular(f64s, f64s), 0.0, 0.01);
  assertAlmostEqual(fallback.sqeuclidean(f64s, f64s), 0.0, 0.01);
  assertAlmostEqual(fallback.angular(f64s, f64s), 0.0, 0.01);

  const f32sNormalized = new Float32Array([
    1 / Math.sqrt(14),
    2 / Math.sqrt(14),
    3 / Math.sqrt(14),
  ]);
  assertAlmostEqual(numkong.inner(f32sNormalized, f32sNormalized), 1.0, 0.01);
  assertAlmostEqual(fallback.inner(f32sNormalized, f32sNormalized), 1.0, 0.01);

  const f32sHistogram = new Float32Array([1.0 / 6, 2.0 / 6, 3.0 / 6]);
  assertAlmostEqual(
    numkong.kullbackleibler(f32sHistogram, f32sHistogram),
    0.0,
    0.01
  );
  assertAlmostEqual(
    numkong.jensenshannon(f32sHistogram, f32sHistogram),
    0.0,
    0.01
  );
  assertAlmostEqual(
    fallback.kullbackleibler(f32sHistogram, f32sHistogram),
    0.0,
    0.01
  );
  assertAlmostEqual(
    fallback.jensenshannon(f32sHistogram, f32sHistogram),
    0.0,
    0.01
  );

  const u8s = new Uint8Array([1, 2, 3]);
  assertAlmostEqual(numkong.hamming(u8s, u8s), 0.0, 0.01);
  assertAlmostEqual(numkong.jaccard(u8s, u8s), 0.0, 0.01);
});

test("Distance from itself JS fallback", () => {
  const f32s = new Float32Array([1.0, 2.0, 3.0]);

  assertAlmostEqual(fallback.sqeuclidean(f32s, f32s), 0.0, 0.01);
  assertAlmostEqual(fallback.angular(f32s, f32s), 0.0, 0.01);

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

test("Squared Euclidean Distance", () => {
  const f64sOne = new Float64Array([1.0, 2.0, 3.0]);
  const f64sTwo = new Float64Array([4.0, 5.0, 6.0]);
  const f64sResult = numkong.sqeuclidean(f64sOne, f64sTwo);
  const f64sResultJS = fallback.sqeuclidean(f64sOne, f64sTwo);
  assertAlmostEqual(f64sResultJS, 27.0, 0.01);
  assertAlmostEqual(f64sResult, 27.0, 0.01);

  const f32sOne = new Float32Array([1.0, 2.0, 3.0]);
  const f32sTwo = new Float32Array([4.0, 5.0, 6.0]);
  const f32sResult = numkong.sqeuclidean(f32sOne, f32sTwo);
  const f32sResultJS = fallback.sqeuclidean(f32sOne, f32sTwo);
  assertAlmostEqual(f32sResultJS, 27.0, 0.01);
  assertAlmostEqual(f32sResult, 27.0, 0.01);

  const u8sOne = new Uint8Array([1, 2, 3]);
  const u8sTwo = new Uint8Array([4, 5, 6]);
  const u8sResult = numkong.sqeuclidean(u8sOne, u8sTwo);
  const u8sResultJS = fallback.sqeuclidean(u8sOne, u8sTwo);
  assertAlmostEqual(u8sResultJS, 27.0, 0.01);
  assertAlmostEqual(u8sResult, 27.0, 0.01);

  const i8sOne = new Int8Array([1, 2, 3]);
  const i8sTwo = new Int8Array([4, 5, 6]);
  const i8sResult = numkong.sqeuclidean(i8sOne, i8sTwo);
  const i8sResultJS = fallback.sqeuclidean(i8sOne, i8sTwo);
  assertAlmostEqual(i8sResultJS, 27.0, 0.01);
  assertAlmostEqual(i8sResult, 27.0, 0.01);
});

test("Euclidean Distance", () => {
  const f64sOne = new Float64Array([1.0, 2.0, 3.0]);
  const f64sTwo = new Float64Array([4.0, 5.0, 6.0]);
  const f64sResult = numkong.euclidean(f64sOne, f64sTwo);
  const f64sResultJS = fallback.euclidean(f64sOne, f64sTwo);
  assertAlmostEqual(f64sResultJS, 5.2, 0.01);
  assertAlmostEqual(f64sResult, 5.2, 0.01);

  const f32sOne = new Float32Array([1.0, 2.0, 3.0]);
  const f32sTwo = new Float32Array([4.0, 5.0, 6.0]);
  const f32sResult = numkong.euclidean(f32sOne, f32sTwo);
  const f32sResultJS = fallback.euclidean(f32sOne, f32sTwo);
  assertAlmostEqual(f32sResultJS, 5.2, 0.01);
  assertAlmostEqual(f32sResult, 5.2, 0.01);

  const u8sOne = new Uint8Array([1, 2, 3]);
  const u8sTwo = new Uint8Array([4, 5, 6]);
  const u8sResult = numkong.euclidean(u8sOne, u8sTwo);
  const u8sResultJS = fallback.euclidean(u8sOne, u8sTwo);
  assertAlmostEqual(u8sResultJS, 5.2, 0.01);
  assertAlmostEqual(u8sResult, 5.2, 0.01);

  const i8sOne = new Int8Array([1, 2, 3]);
  const i8sTwo = new Int8Array([4, 5, 6]);
  const i8sResult = numkong.euclidean(i8sOne, i8sTwo);
  const i8sResultJS = fallback.euclidean(i8sOne, i8sTwo);
  assertAlmostEqual(i8sResultJS, 5.2, 0.01);
  assertAlmostEqual(i8sResult, 5.2, 0.01);
});

test("Inner Product", () => {
  const f64sOne = new Float64Array([1.0, 2.0, 3.0]);
  const f64sTwo = new Float64Array([4.0, 5.0, 6.0]);
  const f64sResult = numkong.inner(f64sOne, f64sTwo);
  const f64sResultJS = fallback.inner(f64sOne, f64sTwo);
  assertAlmostEqual(f64sResultJS, 32.0, 0.01);
  assertAlmostEqual(f64sResult, 32.0, 0.01);

  const f32sOne = new Float32Array([1.0, 2.0, 3.0]);
  const f32sTwo = new Float32Array([4.0, 5.0, 6.0]);
  const f32sResult = numkong.inner(f32sOne, f32sTwo);
  const f32sResultJS = fallback.inner(f32sOne, f32sTwo);
  assertAlmostEqual(f32sResultJS, 32.0, 0.01);
  assertAlmostEqual(f32sResult, 32.0, 0.01);

  const u8sOne = new Uint8Array([1, 2, 3]);
  const u8sTwo = new Uint8Array([4, 5, 6]);
  const u8sResult = numkong.inner(u8sOne, u8sTwo);
  const u8sResultJS = fallback.inner(u8sOne, u8sTwo);
  assertAlmostEqual(u8sResultJS, 32.0, 0.01);
  assertAlmostEqual(u8sResult, 32.0, 0.01);

  const i8sOne = new Int8Array([1, 2, 3]);
  const i8sTwo = new Int8Array([4, 5, 6]);
  const i8sResult = numkong.inner(i8sOne, i8sTwo);
  const i8sResultJS = fallback.inner(i8sOne, i8sTwo);
  assertAlmostEqual(i8sResultJS, 32.0, 0.01);
  assertAlmostEqual(i8sResult, 32.0, 0.01);
});

test("Angular Distance", () => {
  const f64sOne = new Float64Array([1.0, 2.0, 3.0]);
  const f64sTwo = new Float64Array([4.0, 5.0, 6.0]);
  const f64sResult = numkong.angular(f64sOne, f64sTwo);
  const f64sResultJS = fallback.angular(f64sOne, f64sTwo);
  assertAlmostEqual(f64sResultJS, 0.03, 0.01);
  assertAlmostEqual(f64sResult, 0.03, 0.01);

  const f32sOne = new Float32Array([1.0, 2.0, 3.0]);
  const f32sTwo = new Float32Array([4.0, 5.0, 6.0]);
  const f32sResult = numkong.angular(f32sOne, f32sTwo);
  const f32sResultJS = fallback.angular(f32sOne, f32sTwo);
  assertAlmostEqual(f32sResultJS, 0.03, 0.01);
  assertAlmostEqual(f32sResult, 0.03, 0.01);

  const u8sOne = new Uint8Array([1, 2, 3]);
  const u8sTwo = new Uint8Array([4, 5, 6]);
  const u8sResult = numkong.angular(u8sOne, u8sTwo);
  const u8sResultJS = fallback.angular(u8sOne, u8sTwo);
  assertAlmostEqual(u8sResultJS, 0.03, 0.01);
  assertAlmostEqual(u8sResult, 0.03, 0.01);

  const i8sOne = new Int8Array([1, 2, 3]);
  const i8sTwo = new Int8Array([4, 5, 6]);
  const i8sResult = numkong.angular(i8sOne, i8sTwo);
  const i8sResultJS = fallback.angular(i8sOne, i8sTwo);
  assertAlmostEqual(i8sResultJS, 0.03, 0.01);
  assertAlmostEqual(i8sResult, 0.03, 0.01);
});

test("Kullback-Leibler", () => {
  const f64sOne = new Float64Array([1.0 / 6, 2.0 / 6, 3.0 / 6]);
  const f64sTwo = new Float64Array([4.0 / 17, 5.0 / 17, 6.0 / 17]);
  const f64sResult = numkong.kullbackleibler(f64sOne, f64sTwo);
  const f64sResultJS = fallback.kullbackleibler(f64sOne, f64sTwo);
  assertAlmostEqual(f64sResultJS, 0.16, 0.01);
  assertAlmostEqual(f64sResult, 0.16, 0.01);

  const f32sOne = new Float32Array([1.0 / 6, 2.0 / 6, 3.0 / 6]);
  const f32sTwo = new Float32Array([4.0 / 17, 5.0 / 17, 6.0 / 17]);
  const f32sResult = numkong.kullbackleibler(f32sOne, f32sTwo);
  const f32sResultJS = fallback.kullbackleibler(f32sOne, f32sTwo);
  assertAlmostEqual(f32sResultJS, 0.16, 0.01);
  assertAlmostEqual(f32sResult, 0.16, 0.01);
});

test("Jensen-Shannon", () => {
  const f64sOne = new Float64Array([1.0 / 6, 2.0 / 6, 3.0 / 6]);
  const f64sTwo = new Float64Array([4.0 / 17, 5.0 / 17, 6.0 / 17]);
  const f64sResult = numkong.jensenshannon(f64sOne, f64sTwo);
  const f64sResultJS = fallback.jensenshannon(f64sOne, f64sTwo);
  assertAlmostEqual(f64sResultJS, 0.095, 0.01);
  assertAlmostEqual(f64sResult, 0.095, 0.01);

  const f32sOne = new Float32Array([1.0 / 6, 2.0 / 6, 3.0 / 6]);
  const f32sTwo = new Float32Array([4.0 / 17, 5.0 / 17, 6.0 / 17]);
  const f32sResult = numkong.jensenshannon(f32sOne, f32sTwo);
  const f32sResultJS = fallback.jensenshannon(f32sOne, f32sTwo);
  assertAlmostEqual(f32sResultJS, 0.095, 0.01);
  assertAlmostEqual(f32sResult, 0.095, 0.01);
});

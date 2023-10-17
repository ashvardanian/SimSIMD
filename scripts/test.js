import test from 'node:test';
import bindings from 'bindings';
import assert from 'node:assert';

const simsimd = bindings('simsimd');

const typedArray1 = new Float32Array([1.0, 2.0, 3.0]);
const typedArray2 = new Float32Array([4.0, 5.0, 6.0]);

function assertAlmostEqual(actual, expected, tolerance = 1e-6) {
    const lowerBound = expected - tolerance;
    const upperBound = expected + tolerance;
    assert(actual >= lowerBound && actual <= upperBound, `Expected ${actual} to be almost equal to ${expected}`);
}

test('Squared Euclidean Distance', () => {
    const result = simsimd.sqeuclidean(typedArray1, typedArray2);
    assertAlmostEqual(result, 27.0, 0.01);
});

test('Inner Product', () => {
    const result = simsimd.inner(typedArray1, typedArray2);
    assertAlmostEqual(result, -31.0, 0.01);
});

test('Cosine Similarity', () => {
    const result = simsimd.cosine(typedArray1, typedArray2);
    assertAlmostEqual(result, 0.029, 0.01);
});

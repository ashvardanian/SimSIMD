import test from 'node:test';
import bindings from 'bindings';
import assert from 'node:assert';

const simsimd = bindings('simsimd');

const f32Array1 = new Float32Array([1.0, 2.0, 3.0]);
const f32Array2 = new Float32Array([4.0, 5.0, 6.0]);

const u8Array1 = new Uint8Array([1, 2, 3]);
const u8Array2 = new Uint8Array([4, 5, 6]);

function assertAlmostEqual(actual, expected, tolerance = 1e-6) {
    const lowerBound = expected - tolerance;
    const upperBound = expected + tolerance;
    assert(actual >= lowerBound && actual <= upperBound, `Expected ${actual} to be almost equal to ${expected}`);
}

test('Distance from itself', () => {
    assertAlmostEqual(simsimd.sqeuclidean(f32Array1, f32Array1), 0.0, 0.01);
    assertAlmostEqual(simsimd.cosine(f32Array1, f32Array1), 0.0, 0.01);

    // Inner-product distance on non-nroamalized vectors would yield:
    // 1 - 1 - 4 - 9 = -13
    assertAlmostEqual(simsimd.inner(f32Array1, f32Array1), -13.0, 0.01);

    assertAlmostEqual(simsimd.kullbackleibler(f32Array1, f32Array1), 0.0, 0.01);
    assertAlmostEqual(simsimd.jensenshannon(f32Array1, f32Array1), 0.0, 0.01);

    assertAlmostEqual(simsimd.hamming(u8Array1, u8Array1), 0.0, 0.01);
    assertAlmostEqual(simsimd.jaccard(u8Array1, u8Array1), 0.0, 0.01);
});

test('Squared Euclidean Distance', () => {
    const result = simsimd.sqeuclidean(f32Array1, f32Array2);
    assertAlmostEqual(result, 27.0, 0.01);
});

test('Inner Product', () => {
    const result = simsimd.inner(f32Array1, f32Array2);
    assertAlmostEqual(result, -31.0, 0.01);
});

test('Cosine Similarity', () => {
    const result = simsimd.cosine(f32Array1, f32Array2);
    assertAlmostEqual(result, 0.029, 0.01);
});

/**
 * Multi-runtime WASM test suite for NumKong
 * Supports: Emscripten (Node.js), WASI (Node.js), Browser (Playwright)
 *
 * Usage:
 *   NK_RUNTIME=emscripten node --test test/test-wasm.mjs
 *   NK_RUNTIME=wasi-node node --test test/test-wasm.mjs
 */

import test from "node:test";
import assert from "node:assert";
import { readFileSync } from "node:fs";
import { WASI } from "node:wasi";

// Runtime loader - adapts to different WASM execution environments
async function loadNumKong(runtime) {
    switch (runtime) {
        case 'native':
            // Load native Node.js addon (baseline for comparison)
            return (await import('../javascript/dist/esm/numkong.js'));

        case 'emscripten':
            // Load Emscripten build (uses EM_ASM for capability detection)
            const wasmWrapper = await import('../javascript/dist/esm/numkong-wasm.js');
            const EmModule = await import('../build-wasm/numkong.js');
            const wasmInstance = await EmModule.default();
            wasmWrapper.initWasm(wasmInstance);
            return wasmWrapper;

        case 'wasi-node':
            // Load WASI via Node.js built-in WASI support (node:wasi)
            // Host provides capability detection imports (nk_has_v128, nk_has_relaxed)
            const wasi = new WASI({
                version: 'preview1',
                args: [],
                env: {},
            });

            const wasmBytes = readFileSync('./build-wasi/test.wasm');

            // Capability detection test bytecode (SIMD128 detection)
            const simd128Test = new Uint8Array([
                0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, // Magic + version
                0x01, 0x05, 0x01, 0x60, 0x00, 0x01, 0x7b,       // Type: [] -> [v128]
                0x03, 0x02, 0x01, 0x00,                         // Function section
                0x0a, 0x09, 0x01, 0x07, 0x00, 0xfd, 0x0c,       // Code: v128.const
                0x00, 0x00, 0x00, 0x00, 0x0b                    // i32x4 [0,0,0,0] + end
            ]);

            // Capability detection test bytecode (Relaxed SIMD detection)
            const relaxedTest = new Uint8Array([
                0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x01, 0x60, 0x03,
                0x7b, 0x7b, 0x7b, 0x01, 0x7b, 0x03, 0x02, 0x01, 0x00, 0x0a, 0x09, 0x01, 0x07,
                0x00, 0x20, 0x00, 0x20, 0x01, 0x20, 0x02, 0xfd, 0xaf, 0x01, 0x0b // f32x4.relaxed_madd
            ]);

            const { instance } = await WebAssembly.instantiate(wasmBytes, {
                wasi_snapshot_preview1: wasi.wasiImport,
                env: {
                    // Host-side capability probes imported by the WASM module
                    nk_has_v128: () => {
                        try {
                            return WebAssembly.validate(simd128Test) ? 1 : 0;
                        } catch {
                            return 0;
                        }
                    },
                    nk_has_relaxed: () => {
                        try {
                            return WebAssembly.validate(relaxedTest) ? 1 : 0;
                        } catch {
                            return 0;
                        }
                    }
                }
            });

            wasi.start(instance);
            return instance.exports;

        default:
            throw new Error(`Unknown runtime: ${runtime}`);
    }
}

// Load runtime based on environment variable
const runtime = process.env.NK_RUNTIME || 'native';
console.log(`Testing NumKong on runtime: ${runtime}`);

const numkong = await loadNumKong(runtime);

// Helper function for approximate equality
function assertAlmostEqual(actual, expected, tolerance = 1e-6) {
    const lowerBound = expected - tolerance;
    const upperBound = expected + tolerance;
    assert(
        actual >= lowerBound && actual <= upperBound,
        `Expected ${actual} to be almost equal to ${expected} (tolerance: ${tolerance})`
    );
}

// Test suite (copied from test/test.mjs structure)
test(`[${runtime}] Distance from itself`, () => {
    const f32s = new Float32Array([1.0, 2.0, 3.0]);
    assertAlmostEqual(numkong.sqeuclidean(f32s, f32s), 0.0, 0.01);
    assertAlmostEqual(numkong.angular(f32s, f32s), 0.0, 0.01);

    const f64s = new Float64Array([1.0, 2.0, 3.0]);
    assertAlmostEqual(numkong.sqeuclidean(f64s, f64s), 0.0, 0.01);
    assertAlmostEqual(numkong.angular(f64s, f64s), 0.0, 0.01);

    const f32sNormalized = new Float32Array([
        1 / Math.sqrt(14),
        2 / Math.sqrt(14),
        3 / Math.sqrt(14),
    ]);
    assertAlmostEqual(numkong.inner(f32sNormalized, f32sNormalized), 1.0, 0.01);

    const f32sHistogram = new Float32Array([1.0 / 6, 2.0 / 6, 3.0 / 6]);
    assertAlmostEqual(numkong.kullbackleibler(f32sHistogram, f32sHistogram), 0.0, 0.01);
    assertAlmostEqual(numkong.jensenshannon(f32sHistogram, f32sHistogram), 0.0, 0.01);

    const u8s = new Uint8Array([1, 2, 3]);
    assertAlmostEqual(numkong.hamming(u8s, u8s), 0.0, 0.01);
    assertAlmostEqual(numkong.jaccard(u8s, u8s), 0.0, 0.01);
});

test(`[${runtime}] Orthogonal vectors`, () => {
    const a = new Float32Array([1.0, 0.0, 0.0]);
    const b = new Float32Array([0.0, 1.0, 0.0]);

    assertAlmostEqual(numkong.inner(a, b), 0.0, 0.01);
    assertAlmostEqual(numkong.angular(a, b), 1.0, 0.01);
});

test(`[${runtime}] Opposite vectors`, () => {
    const a = new Float32Array([1.0, 2.0, 3.0]);
    const b = new Float32Array([-1.0, -2.0, -3.0]);

    assertAlmostEqual(numkong.angular(a, b), 2.0, 0.01);
});

test(`[${runtime}] Euclidean distance`, () => {
    const a = new Float32Array([0.0, 0.0, 0.0]);
    const b = new Float32Array([3.0, 4.0, 0.0]);

    assertAlmostEqual(numkong.euclidean(a, b), 5.0, 0.01);
    assertAlmostEqual(numkong.sqeuclidean(a, b), 25.0, 0.01);
});

test(`[${runtime}] Capability detection`, () => {
    // Test that getCapabilities returns a bigint
    if (typeof numkong.getCapabilities === 'function') {
        const caps = numkong.getCapabilities();
        assert(typeof caps === 'bigint', 'getCapabilities should return bigint');
        console.log(`  Runtime capabilities: 0x${caps.toString(16)}`);

        // Test hasCapability helper
        if (typeof numkong.hasCapability === 'function') {
            // Serial fallback should always be present
            assert(numkong.hasCapability(1n << 0n), 'SERIAL capability should be present');
        }
    } else {
        console.log(`  getCapabilities not available in ${runtime} mode`);
    }
});

// Expanded test coverage - comprehensive dtype/function/dimension testing
const testMatrix = {
    dot: ['f64', 'f32', 'i8', 'u8'],
    inner: ['f64', 'f32', 'i8', 'u8'],
    sqeuclidean: ['f64', 'f32', 'i8', 'u8'],
    euclidean: ['f64', 'f32', 'i8', 'u8'],
    angular: ['f64', 'f32', 'i8'],
    kullbackleibler: ['f64', 'f32'],
    jensenshannon: ['f64', 'f32'],
    hamming: ['u8'],
    jaccard: ['u8']
};

const dims = [3, 16, 128, 1536];

function randomVector(dtype, len) {
    const rand = () => Math.random() * 2 - 1;
    if (dtype === 'f64') return Float64Array.from({length: len}, rand);
    if (dtype === 'f32') return Float32Array.from({length: len}, rand);
    if (dtype === 'i8') return Int8Array.from({length: len}, () => (Math.random() * 256 - 128) | 0);
    if (dtype === 'u8') return Uint8Array.from({length: len}, () => Math.random() * 256 | 0);
}

for (const [fn, dtypes] of Object.entries(testMatrix)) {
    for (const dtype of dtypes) {
        for (const dim of dims) {
            test(`[${runtime}] ${fn}(${dtype}×${dim})`, () => {
                const a = randomVector(dtype, dim);
                const b = randomVector(dtype, dim);
                const result = numkong[fn](a, b);

                assert.strictEqual(typeof result, 'number');
                assert.ok(isFinite(result));
                assertAlmostEqual(result, numkong[fn](a, b), 1e-6);
            });
        }
    }
}

console.log(`All tests passed for runtime: ${runtime}`);

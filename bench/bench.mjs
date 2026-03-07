#!/usr/bin/env node
/**
 * @brief Unified multi-runtime benchmark and reporting for NumKong
 * @file bench/bench.mjs
 * @author Claude & Ash Vardanian
 * @date February 4, 2026
 *
 * All-in-one benchmark suite supporting multiple runtimes and report generation.
 *
 * Environment variables:
 * - NK_DIMENSIONS: Vector dimensionality (default: 1536)
 * - NK_ITERATIONS: Benchmark iterations (default: 1000)
 * - NK_FILTER: Regex to filter tests (default: .*)
 * - NK_RUNTIME: Runtime to use (default: native)
 * - NK_SEED: Random seed (default: 42)
 *
 * Commands:
 * - node bench.mjs              → Run benchmarks for configured runtime
 * - node bench.mjs --report     → Generate comparison report from results
 * - node bench.mjs --browser    → Run browser benchmarks via Playwright
 */

import Benchmark from 'benchmark';
import { readFileSync, writeFileSync, readdirSync, existsSync, mkdirSync } from 'node:fs';
import { WASI } from 'node:wasi';
import build from 'node-gyp-build';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const rootDir = path.join(__dirname, '..');
const resultsDir = path.join(__dirname, 'results');

// ============================================================================
// Configuration
// ============================================================================

const CONFIG = {
    dimensions: parseInt(process.env.NK_DIMENSIONS || '1536'),
    iterations: parseInt(process.env.NK_ITERATIONS || '1000'),
    filter: new RegExp(process.env.NK_FILTER || '.*'),
    runtime: process.env.NK_RUNTIME || 'native',
    seed: parseInt(process.env.NK_SEED || '42')
};

const BENCHMARK_MATRIX = {
    functions: ['dot', 'inner', 'angular', 'sqeuclidean', 'euclidean', 'hamming', 'jaccard', 'kullbackleibler', 'jensenshannon'],
    dtypes: ['f64', 'f32', 'f16', 'bf16', 'e4m3', 'e5m2', 'i8', 'u8', 'u1'],
};

const FUNCTION_DTYPE_SUPPORT = {
    'dot': ['f64', 'f32', 'f16', 'bf16', 'e4m3', 'e5m2', 'i8', 'u8'],
    'inner': ['f64', 'f32', 'f16', 'bf16', 'e4m3', 'e5m2', 'i8', 'u8'],
    'angular': ['f64', 'f32', 'f16', 'bf16', 'e4m3', 'e5m2', 'i8'],
    'sqeuclidean': ['f64', 'f32', 'f16', 'bf16', 'e4m3', 'e5m2', 'i8', 'u8'],
    'euclidean': ['f64', 'f32', 'f16', 'bf16', 'e4m3', 'e5m2', 'i8', 'u8'],
    'hamming': ['u1'],
    'jaccard': ['u1'],
    'kullbackleibler': ['f64', 'f32'],
    'jensenshannon': ['f64', 'f32']
};

const RUNTIME_INFO = {
    native: {
        name: 'Native Node.js Addon',
        description: 'Compiled C code with SIMD optimizations',
        expectedPerformance: '1.00x (baseline)',
    },
    emscripten: {
        name: 'Emscripten WASM',
        description: 'WebAssembly with SIMD support',
        expectedPerformance: '0.60-0.80x of native',
    },
    wasi: {
        name: 'WASI',
        description: 'WebAssembly System Interface',
        expectedPerformance: '0.50-0.70x of native',
    },
    browser: {
        name: 'Browser (Chromium)',
        description: 'WebAssembly in browser context',
        expectedPerformance: '0.40-0.60x of native',
    }
};

// ============================================================================
// Runtime Loading
// ============================================================================

async function loadNumKong(runtime) {
    switch (runtime) {
        case 'native':
            return loadNative();
        case 'emscripten':
            return loadEmscripten();
        case 'wasi':
            return loadWASI();
        case 'browser':
            throw new Error('Browser runtime requires --browser flag');
        default:
            throw new Error(`Unknown runtime: ${runtime}. Use: native, emscripten, wasi, or browser`);
    }
}

function loadNative() {
    try {
        const builddir = rootDir;
        const numkong = build(builddir);
        console.log('✓ Loaded NumKong native addon');
        return numkong;
    } catch (e) {
        throw new Error(`Failed to load native addon: ${e.message}`);
    }
}

async function loadEmscripten() {
    try {
        const wasmPath = path.join(rootDir, 'build-wasm', 'numkong_js.js');
        const wasmModule = await import(wasmPath);
        const numkong = await wasmModule.default();
        console.log('✓ Loaded NumKong Emscripten WASM');
        return numkong;
    } catch (e) {
        throw new Error(`Failed to load Emscripten WASM: ${e.message}\nBuild with: cmake -B build-wasm -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-wasm.cmake && cmake --build build-wasm`);
    }
}

async function loadWASI() {
    try {
        const wasiPath = path.join(rootDir, 'build-wasi', 'test.wasm');

        // SIMD capability test bytecode
        const simd128Test = new Uint8Array([
            0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
            0x01, 0x05, 0x01, 0x60, 0x00, 0x01, 0x7b, 0x03,
            0x02, 0x01, 0x00, 0x0a, 0x0a, 0x01, 0x08, 0x00,
            0x41, 0x00, 0xfd, 0x0f, 0x0b
        ]);

        const relaxedTest = new Uint8Array([
            0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
            0x01, 0x05, 0x01, 0x60, 0x00, 0x01, 0x7b, 0x03,
            0x02, 0x01, 0x00, 0x0a, 0x0a, 0x01, 0x08, 0x00,
            0x41, 0x00, 0xfd, 0x81, 0x01, 0x0b
        ]);

        const hasV128 = WebAssembly.validate(simd128Test) ? 1 : 0;
        const hasRelaxed = WebAssembly.validate(relaxedTest) ? 1 : 0;

        console.log(`  WASM SIMD capabilities: v128=${hasV128}, relaxed=${hasRelaxed}`);

        const wasi = new WASI({
            version: 'preview1',
            args: [],
            env: {},
        });

        const wasmBytes = readFileSync(wasiPath);

        const { instance } = await WebAssembly.instantiate(wasmBytes, {
            wasi_snapshot_preview1: wasi.wasiImport,
            env: {
                nk_has_v128: () => hasV128,
                nk_has_relaxed: () => hasRelaxed
            }
        });

        wasi.start(instance);
        console.log('✓ Loaded NumKong WASI');
        return instance.exports;
    } catch (e) {
        throw new Error(`Failed to load WASI: ${e.message}\nBuild with: cmake -B build-wasi -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-wasi.cmake -DNK_BUILD_TEST=ON && cmake --build build-wasi`);
    }
}

// ============================================================================
// Test Data Generation
// ============================================================================

class Random {
    constructor(seed) {
        this.seed = seed;
    }

    next() {
        this.seed = (this.seed * 1103515245 + 12345) & 0x7fffffff;
        return this.seed / 0x7fffffff;
    }
}

function generateTestData(dtype, length, seed) {
    const rng = new Random(seed);

    switch (dtype) {
        case 'f64':
            return Float64Array.from({ length }, () => rng.next() * 2 - 1);
        case 'f32':
            return Float32Array.from({ length }, () => rng.next() * 2 - 1);
        case 'f16':
        case 'bf16':
        case 'e4m3':
        case 'e5m2':
            return Float32Array.from({ length }, () => rng.next() * 2 - 1);
        case 'i8':
            return Int8Array.from({ length }, () => Math.floor(rng.next() * 256) - 128);
        case 'u8':
            return Uint8Array.from({ length }, () => Math.floor(rng.next() * 256));
        case 'u1':
            const byteLength = Math.ceil(length / 8);
            return Uint8Array.from({ length: byteLength }, () => Math.floor(rng.next() * 256));
        default:
            throw new Error(`Unknown dtype: ${dtype}`);
    }
}

// ============================================================================
// Benchmark Execution
// ============================================================================

async function runBenchmarks() {
    console.log('╔════════════════════════════════════════════════════════════════╗');
    console.log('║          NumKong JavaScript Multi-Runtime Benchmarks          ║');
    console.log('╚════════════════════════════════════════════════════════════════╝\n');

    console.log('Configuration:');
    console.log(`  Dimensions: ${CONFIG.dimensions}`);
    console.log(`  Iterations: ${CONFIG.iterations}`);
    console.log(`  Runtime: ${CONFIG.runtime}`);
    console.log(`  Filter: ${CONFIG.filter.source}`);
    console.log(`  Seed: ${CONFIG.seed}\n`);

    const runtimeInfo = RUNTIME_INFO[CONFIG.runtime];
    console.log(`Runtime: ${runtimeInfo.name}`);
    console.log(`  ${runtimeInfo.description}`);
    console.log(`  Expected performance: ${runtimeInfo.expectedPerformance}\n`);

    // Load NumKong module
    let numkong;
    try {
        numkong = await loadNumKong(CONFIG.runtime);
    } catch (e) {
        console.error(`❌ Failed to load runtime: ${e.message}`);
        process.exit(1);
    }

    const results = [];

    // Iterate through benchmark matrix
    for (const func of BENCHMARK_MATRIX.functions) {
        const supportedDtypes = FUNCTION_DTYPE_SUPPORT[func] || [];

        for (const dtype of supportedDtypes) {
            const testName = `${func}-${dtype}`;

            // Apply filter
            if (!CONFIG.filter.test(testName)) {
                continue;
            }

            // Generate test data
            const a = generateTestData(dtype, CONFIG.dimensions, CONFIG.seed);
            const b = generateTestData(dtype, CONFIG.dimensions, CONFIG.seed + 1);

            // Run benchmark
            await new Promise((resolve) => {
                const suite = new Benchmark.Suite();

                suite.add(testName, () => {
                    numkong[func](a, b);
                });

                suite.on('cycle', (event) => {
                    const bench = event.target;
                    const opsPerSec = bench.hz;
                    const meanTime = bench.stats.mean * 1000; // ms
                    const stdDev = bench.stats.deviation * 1000; // ms

                    console.log(`✓ ${testName.padEnd(30)} ${opsPerSec.toFixed(0).padStart(10)} ops/sec  (${meanTime.toFixed(3)}ms ± ${stdDev.toFixed(3)}ms)`);

                    results.push({
                        function: func,
                        dtype: dtype,
                        runtime: CONFIG.runtime,
                        dimensions: CONFIG.dimensions,
                        opsPerSec: opsPerSec,
                        meanTime: meanTime,
                        stdDev: stdDev,
                        timestamp: new Date().toISOString()
                    });
                });

                suite.on('complete', resolve);

                suite.run({ async: false });
            });
        }
    }

    // Save results
    mkdirSync(resultsDir, { recursive: true });

    const resultsFile = path.join(resultsDir, `${CONFIG.runtime}.json`);
    writeFileSync(resultsFile, JSON.stringify(results, null, 2));

    console.log(`\n✅ Benchmark complete! Results saved to: ${resultsFile}`);
    console.log(`   Total tests run: ${results.length}`);

    return results;
}

// ============================================================================
// Browser Benchmarks (Playwright)
// ============================================================================

async function runBrowserBenchmarks() {
    console.log('╔════════════════════════════════════════════════════════════════╗');
    console.log('║             NumKong Browser Benchmark Runner                  ║');
    console.log('╚════════════════════════════════════════════════════════════════╝\n');

    console.log('Configuration:');
    console.log(`  Dimensions: ${CONFIG.dimensions}`);
    console.log(`  Filter: ${CONFIG.filter.source}`);
    console.log(`  Seed: ${CONFIG.seed}\n`);

    // Check if WASM build exists
    const wasmPath = path.join(rootDir, 'build-wasm', 'numkong_js.js');
    if (!existsSync(wasmPath)) {
        console.error('❌ Emscripten WASM build not found!');
        console.error(`   Expected: ${wasmPath}`);
        console.error('   Build it with:');
        console.error('     source ~/emsdk/emsdk_env.sh');
        console.error('     cmake -B build-wasm -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-wasm.cmake');
        console.error('     cmake --build build-wasm');
        process.exit(1);
    }

    console.log('✓ Found Emscripten WASM build');

    // Dynamic import of Playwright
    console.log('\nLaunching Chromium...');
    const { chromium } = await import('playwright');

    const browser = await chromium.launch({
        headless: true,
        args: [
            '--enable-features=WebAssemblySimd',
            '--enable-features=WebAssemblyRelaxedSimd',
        ]
    });

    const context = await browser.newContext();
    const page = await context.newPage();

    // Set up console logging from browser
    page.on('console', msg => {
        const text = msg.text();
        if (text.includes('Error') || text.includes('Failed')) {
            console.error(`  Browser: ${text}`);
        }
    });

    // Build URL with query parameters
    const benchmarkUrl = `file://${path.join(__dirname, 'bench-browser.html')}?dimensions=${CONFIG.dimensions}&filter=${encodeURIComponent(CONFIG.filter.source)}&seed=${CONFIG.seed}`;

    console.log('✓ Chromium launched');
    console.log(`\nNavigating to: ${benchmarkUrl}`);

    try {
        await page.goto(benchmarkUrl);
        console.log('✓ Page loaded');
        console.log('\nRunning benchmarks (this may take several minutes)...\n');

        // Wait for benchmarks to complete (timeout: 10 minutes)
        await page.waitForFunction(
            () => window.benchmarkComplete === true,
            { timeout: 600000 }
        );

        // Check for errors
        const error = await page.evaluate(() => window.benchmarkError);
        if (error) {
            throw new Error(error);
        }

        // Get results
        const results = await page.evaluate(() => window.benchmarkResults);

        if (!results || results.length === 0) {
            throw new Error('No benchmark results returned');
        }

        console.log(`\n✅ Browser benchmarks complete! Total tests: ${results.length}`);

        // Print summary
        console.log('\nQuick Summary:');
        const avgOps = results.reduce((sum, r) => sum + r.opsPerSec, 0) / results.length;
        console.log(`  Average: ${avgOps.toFixed(0)} ops/sec (${results.length} tests)`);

        // Save results
        mkdirSync(resultsDir, { recursive: true });
        const resultsFile = path.join(resultsDir, 'browser.json');
        writeFileSync(resultsFile, JSON.stringify(results, null, 2));

        console.log(`\n✓ Results saved to: ${resultsFile}`);
        console.log('\nGenerate comparison report with: node bench.mjs --report');

    } catch (error) {
        console.error(`\n❌ Benchmark failed: ${error.message}`);

        // Take screenshot for debugging
        const screenshotPath = path.join(resultsDir, 'error-screenshot.png');
        mkdirSync(path.dirname(screenshotPath), { recursive: true });
        await page.screenshot({ path: screenshotPath });
        console.error(`   Screenshot saved to: ${screenshotPath}`);

        await browser.close();
        process.exit(1);
    }

    await browser.close();
}

// ============================================================================
// Report Generation
// ============================================================================

function loadAllResults() {
    if (!existsSync(resultsDir)) {
        throw new Error(`Results directory not found: ${resultsDir}\nRun benchmarks first with: node bench.mjs`);
    }

    const files = readdirSync(resultsDir).filter(f => f.endsWith('.json'));
    const allResults = {};

    for (const file of files) {
        const runtime = path.basename(file, '.json');
        const filepath = path.join(resultsDir, file);
        const data = JSON.parse(readFileSync(filepath, 'utf-8'));
        allResults[runtime] = data;
    }

    return allResults;
}

function generateMarkdownReport(allResults) {
    const runtimes = Object.keys(allResults);

    if (runtimes.length === 0) {
        return '# No benchmark results found\n\nRun benchmarks with: node bench.mjs';
    }

    // Get configuration from first result
    const firstResult = allResults[runtimes[0]][0];
    const dimensions = firstResult?.dimensions || 'unknown';
    const timestamp = firstResult?.timestamp ? new Date(firstResult.timestamp).toLocaleString() : 'unknown';

    let report = '# NumKong JavaScript Benchmark Results\n\n';
    report += '## Configuration\n\n';
    report += `- **Dimensions**: ${dimensions}\n`;
    report += `- **Date**: ${timestamp}\n`;
    report += `- **Node.js**: ${process.version}\n`;
    report += `- **Runtimes**: ${runtimes.join(', ')}\n\n`;

    // Group results by function and dtype
    const grouped = {};

    for (const runtime of runtimes) {
        for (const result of allResults[runtime]) {
            const key = `${result.function}-${result.dtype}`;
            if (!grouped[key]) {
                grouped[key] = {
                    function: result.function,
                    dtype: result.dtype,
                    results: {}
                };
            }
            grouped[key].results[runtime] = result.opsPerSec;
        }
    }

    // Generate comparison tables by function
    const functions = [...new Set(Object.values(grouped).map(g => g.function))];

    for (const func of functions) {
        report += `## ${func.charAt(0).toUpperCase() + func.slice(1)}\n\n`;
        report += '| DType | ' + runtimes.map(r => r.charAt(0).toUpperCase() + r.slice(1)).join(' | ') + ' | Best |\n';
        report += '|-------|' + runtimes.map(() => '--------').join('|') + '|------|\n';

        const funcResults = Object.values(grouped).filter(g => g.function === func);

        for (const { dtype, results } of funcResults) {
            const values = runtimes.map(r => results[r]);
            const maxVal = Math.max(...values.filter(v => v !== undefined));
            const bestRuntime = runtimes[values.indexOf(maxVal)];

            const row = [
                dtype,
                ...runtimes.map(r => {
                    const val = results[r];
                    if (!val) return '-';
                    const formatted = val >= 1000 ? `${(val / 1000).toFixed(1)}k` : val.toFixed(0);
                    return val === maxVal ? `**${formatted}**` : formatted;
                }),
                bestRuntime
            ];

            report += '| ' + row.join(' | ') + ' |\n';
        }

        report += '\n';
    }

    // Summary section
    report += '## Summary\n\n';

    if (runtimes.includes('native')) {
        const nativeResults = allResults['native'];
        const avgNative = nativeResults.reduce((sum, r) => sum + r.opsPerSec, 0) / nativeResults.length;

        for (const runtime of runtimes) {
            if (runtime === 'native') continue;

            const runtimeResults = allResults[runtime];
            const avgRuntime = runtimeResults.reduce((sum, r) => sum + r.opsPerSec, 0) / runtimeResults.length;
            const ratio = (avgRuntime / avgNative * 100).toFixed(0);

            report += `- **${runtime}**: ${ratio}% of native performance\n`;
        }
    }

    report += '\n---\n\n';
    report += `*Generated with NumKong benchmark suite on ${new Date().toLocaleString()}*\n`;

    return report;
}

function generateJSONSummary(allResults) {
    const summary = {
        timestamp: new Date().toISOString(),
        node_version: process.version,
        runtimes: {}
    };

    for (const [runtime, results] of Object.entries(allResults)) {
        const opsPerSec = results.map(r => r.opsPerSec);
        summary.runtimes[runtime] = {
            count: results.length,
            avgOpsPerSec: opsPerSec.reduce((a, b) => a + b, 0) / opsPerSec.length,
            minOpsPerSec: Math.min(...opsPerSec),
            maxOpsPerSec: Math.max(...opsPerSec)
        };
    }

    return summary;
}

function generateReport() {
    console.log('╔════════════════════════════════════════════════════════════════╗');
    console.log('║             NumKong Benchmark Report Generator                ║');
    console.log('╚════════════════════════════════════════════════════════════════╝\n');

    try {
        const allResults = loadAllResults();
        const runtimes = Object.keys(allResults);

        console.log(`Found results for ${runtimes.length} runtime(s): ${runtimes.join(', ')}\n`);

        // Generate markdown report
        const markdown = generateMarkdownReport(allResults);
        const markdownPath = path.join(resultsDir, 'report.md');
        writeFileSync(markdownPath, markdown);
        console.log(`✓ Markdown report: ${markdownPath}`);

        // Generate JSON summary
        const summary = generateJSONSummary(allResults);
        const summaryPath = path.join(resultsDir, 'summary.json');
        writeFileSync(summaryPath, JSON.stringify(summary, null, 2));
        console.log(`✓ JSON summary: ${summaryPath}`);

        console.log('\n✅ Report generation complete!');

        // Print quick summary
        console.log('\nQuick Summary:');
        for (const [runtime, stats] of Object.entries(summary.runtimes)) {
            console.log(`  ${runtime}: ${stats.avgOpsPerSec.toFixed(0)} ops/sec avg (${stats.count} tests)`);
        }

    } catch (e) {
        console.error(`❌ Error: ${e.message}`);
        process.exit(1);
    }
}

// ============================================================================
// Main Entry Point
// ============================================================================

async function main() {
    const args = process.argv.slice(2);

    if (args.includes('--report')) {
        // Generate report from existing results
        generateReport();
    } else if (args.includes('--browser')) {
        // Run browser benchmarks
        await runBrowserBenchmarks();
    } else {
        // Run benchmarks for configured runtime
        await runBenchmarks();
    }
}

// Run if called directly
if (import.meta.url === `file://${process.argv[1]}`) {
    main().catch((e) => {
        console.error('❌ Fatal error:', e.message);
        process.exit(1);
    });
}

export { runBenchmarks, generateReport, loadNumKong };

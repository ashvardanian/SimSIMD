const benchmark = require('benchmark');

// Assuming the vectors are of the same length
function cosineDistance(a, b) {
    let dotProduct = 0;
    let magA = 0;
    let magB = 0;
    for (let i = 0; i < a.length; i++) {
        dotProduct += a[i] * b[i];
        magA += a[i] * a[i];
        magB += b[i] * b[i];
    }
    return 1 - (dotProduct / (Math.sqrt(magA) * Math.sqrt(magB)));
}


// Generate random data for testing
const dimensions = 1536;  // Adjust dimensions as needed
const array1 = Array.from({ length: dimensions }, () => Math.random() * 100);
const array2 = Array.from({ length: dimensions }, () => Math.random() * 100);
const floatArray1 = new Float32Array(array1);
const floatArray2 = new Float32Array(array2);


// Create benchmark suite
const singleSuite = new benchmark.Suite('Single Vector Processing');

// Single-vector processing benchmarks
singleSuite

    // Pure JavaScript
    .add('Array of Numbers', () => {
        cosineDistance(array1, array2);
    })
    .add('TypedArray of Float32', () => {
        cosineDistance(floatArray1, floatArray2);
    })
    .on('cycle', (event) => {
        if (event.target.error) {
            console.error(String(event.target.error));
        } else {
            console.log(String(event.target));
        }
    })
    .on('complete', () => {
        console.log('Fastest Single-Vector Processing is ' + singleSuite.filter('fastest').map('name'));
    })
    .run({
        noCache: true,
        async: false,
    });

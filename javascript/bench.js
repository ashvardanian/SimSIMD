import benchmark from 'benchmark';
import bindings from 'bindings';

const simsimd = bindings('simsimd');

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
const size = 1536;  // Adjust size as needed
const array1 = Array.from({ length: size }, () => Math.random() * 100);
const array2 = Array.from({ length: size }, () => Math.random() * 100);
const floatArray1 = new Float32Array(array1);
const floatArray2 = new Float32Array(array2);
const intArray1 = new Int8Array(array1);
const intArray2 = new Int8Array(array2);

// Create benchmark suite
const suite = new benchmark.Suite();

suite
    .add('Plain Float JavaScript Arrays', () => {
        cosineDistance(array1, array2);
    })
    .add('TypedArray Float without SimSIMD', () => {
        cosineDistance(floatArray1, floatArray2);
    })
    .add('TypedArray Int without SimSIMD', () => {
        cosineDistance(intArray1, intArray2);
    })
    .add('TypedArray Float with SimSIMD', () => {
        simsimd.cosine(floatArray1, floatArray2);
    })
    .add('TypedArray Int with SimSIMD', () => {
        simsimd.cosine(intArray1, intArray2);
    })
    .on('cycle', (event) => {
        console.log(String(event.target));
    })
    .on('complete', () => {
        console.log('Fastest is ' + suite.filter('fastest').map('name'));
    })
    .run({
        noCache: true,
        async: false,
    });

# NumKong for JavaScript

## Installation

To install, choose one of the following options depending on your environment:

- `npm install --save numkong`
- `yarn add numkong`
- `pnpm add numkong`
- `bun install numkong`

The package is distributed with prebuilt binaries, but if your platform is not supported, you can build the package from the source via `npm run build`.
This will automatically happen unless you install the package with the `--ignore-scripts` flag or use Bun.
After you install it, you will be able to call the NumKong functions on various `TypedArray` variants:

```js
const { sqeuclidean, angular, inner, hamming, jaccard } = require("numkong");

const vectorA = new Float32Array([1.0, 2.0, 3.0]);
const vectorB = new Float32Array([4.0, 5.0, 6.0]);

const distance = sqeuclidean(vectorA, vectorB);
console.log("Squared Euclidean Distance:", distance);
```

Other numeric types and precision levels are supported as well.
For double-precision floating-point numbers, use `Float64Array`:

```js
const vectorA = new Float64Array([1.0, 2.0, 3.0]);
const vectorB = new Float64Array([4.0, 5.0, 6.0]);
const distance = angular(vectorA, vectorB);
```

When doing machine learning and vector search with high-dimensional vectors you may want to quantize them to 8-bit integers.
You may want to project values from the $[-1, 1]$ range to the $[-127, 127]$ range and then cast them to `Int8Array`:

```js
const quantizedVectorA = new Int8Array(vectorA.map((v) => v * 127));
const quantizedVectorB = new Int8Array(vectorB.map((v) => v * 127));
const distance = angular(quantizedVectorA, quantizedVectorB);
```

A more extreme quantization case would be to use binary vectors.
You can map all positive values to `1` and all negative values and zero to `0`, packing eight values into a single byte.
After that, Hamming and Jaccard distances can be computed.

```js
const { toBinary, hamming } = require("numkong");

const binaryVectorA = toBinary(vectorA);
const binaryVectorB = toBinary(vectorB);
const distance = hamming(binaryVectorA, binaryVectorB);
```

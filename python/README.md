# NumKong for Python

The package is intended to replace the usage of `numpy.inner`, `numpy.dot`, and `scipy.spatial.distance`.
Aside from drastic performance improvements, NumKong significantly improves accuracy in mixed precision setups.
NumPy and SciPy, processing `int8`, `uint8` or `float16` vectors, will use the same types for accumulators, while NumKong can combine `int8` enumeration, `int16` multiplication, and `int32` accumulation to avoid overflows entirely.
The same applies to processing `float16` and `bfloat16` values with `float32` precision.

## Installation

Use the following snippet to install NumKong and list hardware acceleration options available on your machine:

```sh
pip install numkong
python -c "import numkong; print(numkong.get_capabilities())"   # for hardware introspection
python -c "import numkong; help(numkong)"                       # for documentation
```

With precompiled binaries, NumKong ships `.pyi` interface files for type hinting and static analysis.
You can check all the available functions in [`python/annotations/__init__.pyi`](https://github.com/ashvardanian/NumKong/blob/main/python/annotations/__init__.pyi).

## One-to-One Distance

```py
import numkong
import numpy as np

vec1 = np.random.randn(1536).astype(np.float32)
vec2 = np.random.randn(1536).astype(np.float32)
dist = numkong.angular(vec1, vec2)
```

Supported functions include `angular`, `inner`, `sqeuclidean`, `hamming`, `jaccard`, `kullbackleibler`, `jensenshannon`, and `intersect`.
Dot products are supported for both real and complex numbers:

```py
vec1 = np.random.randn(768).astype(np.float64) + 1j * np.random.randn(768).astype(np.float64)
vec2 = np.random.randn(768).astype(np.float64) + 1j * np.random.randn(768).astype(np.float64)

dist = numkong.dot(vec1.astype(np.complex128), vec2.astype(np.complex128))
dist = numkong.dot(vec1.astype(np.complex64), vec2.astype(np.complex64))
dist = numkong.vdot(vec1.astype(np.complex64), vec2.astype(np.complex64)) # conjugate, same as `np.vdot`
```

Unlike SciPy, NumKong allows explicitly stating the precision of the input vectors, which is especially useful for mixed-precision setups.
The `dtype` argument can be passed both by name and as a positional argument:

```py
dist = numkong.angular(vec1, vec2, "int8")
dist = numkong.angular(vec1, vec2, "float16")
dist = numkong.angular(vec1, vec2, "float32")
dist = numkong.angular(vec1, vec2, "float64")
dist = numkong.hamming(vec1, vec2, "bin8")
```

Binary distance functions are computed at a bit-level.
Meaning a vector of 10x 8-bit integers will be treated as a sequence of 80 individual bits or dimensions.
This differs from NumPy, that can't handle smaller-than-byte types, but you can still avoid the `bin8` argument by reinterpreting the vector as booleans:

```py
vec1 = np.random.randint(2, size=80).astype(np.uint8).packbits().view(np.bool_)
vec2 = np.random.randint(2, size=80).astype(np.uint8).packbits().view(np.bool_)
hamming_distance = numkong.hamming(vec1, vec2)
jaccard_distance = numkong.jaccard(vec1, vec2)
```

With other frameworks, like PyTorch, one can get a richer type-system than NumPy, but the lack of good CPython interoperability makes it hard to pass data without copies.
Here is an example of using NumKong with PyTorch to compute the cosine similarity between two `bfloat16` vectors:

```py
import numpy as np
buf1 = np.empty(8, dtype=np.uint16)
buf2 = np.empty(8, dtype=np.uint16)

# View the same memory region with PyTorch and randomize it
import torch
vec1 = torch.asarray(memoryview(buf1), copy=False).view(torch.bfloat16)
vec2 = torch.asarray(memoryview(buf2), copy=False).view(torch.bfloat16)
torch.randn(8, out=vec1)
torch.randn(8, out=vec2)

# Both libs will look into the same memory buffers and report the same results
dist_slow = 1 - torch.nn.functional.cosine_similarity(vec1, vec2, dim=0)
dist_fast = numkong.angular(buf1, buf2, "bfloat16")
```

It also allows using NumKong for half-precision complex numbers, which NumPy does not support.
For that, view data as continuous even-length `np.float16` vectors and override type-resolution with `complex32` string.

```py
vec1 = np.random.randn(1536).astype(np.float16)
vec2 = np.random.randn(1536).astype(np.float16)
nk.dot(vec1, vec2, "complex32")
nk.vdot(vec1, vec2, "complex32")
```

When dealing with sparse representations and integer sets, you can apply the `intersect` function to two 1-dimensional arrays of `uint16` or `uint32` integers:

```py
from random import randint
import numpy as np
import numkong as nk

length1, length2 = randint(1, 100), randint(1, 100)
vec1 = np.sort(np.random.randint(0, 1000, length1).astype(np.uint16))
vec2 = np.sort(np.random.randint(0, 1000, length2).astype(np.uint16))

slow_result = len(np.intersect1d(vec1, vec2))
fast_result = nk.intersect(vec1, vec2)
assert slow_result == fast_result
```

## One-to-Many Distances

Every distance function can be used not only for one-to-one but also one-to-many and many-to-many distance calculations.
For one-to-many:

```py
vec1 = np.random.randn(1536).astype(np.float32) # rank 1 tensor
batch1 = np.random.randn(1, 1536).astype(np.float32) # rank 2 tensor
batch2 = np.random.randn(100, 1536).astype(np.float32)

dist_rank1 = numkong.angular(vec1, batch2)
dist_rank2 = numkong.angular(batch1, batch2)
```

## Many-to-Many Distances

All distance functions in NumKong can be used to compute many-to-many distances.
For two batches of 100 vectors to compute 100 distances, one would call it like this:

```py
batch1 = np.random.randn(100, 1536).astype(np.float32)
batch2 = np.random.randn(100, 1536).astype(np.float32)
dist = numkong.angular(batch1, batch2)
```

Input matrices must have identical shapes.
This functionality isn't natively present in NumPy or SciPy, and generally requires creating intermediate arrays, which is inefficient and memory-consuming.

## Many-to-Many All-Pairs Distances

One can use NumKong to compute distances between all possible pairs of rows across two matrices (akin to [`scipy.spatial.distance.cdist`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html)).
The resulting object will have a type `DistancesTensor`, zero-copy compatible with NumPy and other libraries.
For two arrays of 10 and 1,000 entries, the resulting tensor will have 10,000 cells:

```py
import numpy as np
from numkong import cdist, DistancesTensor

matrix1 = np.random.randn(1000, 1536).astype(np.float32)
matrix2 = np.random.randn(10, 1536).astype(np.float32)
distances: DistancesTensor = numkong.cdist(matrix1, matrix2, metric="cosine")   # zero-copy, managed by NumKong
distances_array: np.ndarray = np.array(distances, copy=True)                    # now managed by NumPy
```

## Element-wise Kernels

NumKong also provides mixed-precision element-wise kernels, where the input vectors and the output have the same numeric type, but the intermediate accumulators are of a higher precision.

```py
import numpy as np
from numkong import fma, wsum

# Let's take two FullHD video frames
first_frame = np.random.randn(1920 * 1024).astype(np.uint8)
second_frame = np.random.randn(1920 * 1024).astype(np.uint8)
average_frame = np.empty_like(first_frame)
wsum(first_frame, second_frame, alpha=0.5, beta=0.5, out=average_frame)

# Slow analog with NumPy:
slow_average_frame = (0.5 * first_frame + 0.5 * second_frame).astype(np.uint8)
```

Similarly, the `fma` takes three arguments and computes the fused multiply-add operation.
In applications like Machine Learning you may also benefit from using the "brain-float" format not natively supported by NumPy.
In 3D Graphics, for example, we can use FMA to compute the [Phong shading model](https://en.wikipedia.org/wiki/Phong_shading):

```py
# Assume a FullHD frame with random values for simplicity
light_intensity = np.random.rand(1920 * 1080).astype(np.float16)  # Intensity of light on each pixel
diffuse_component = np.random.rand(1920 * 1080).astype(np.float16)  # Diffuse reflectance on the surface
specular_component = np.random.rand(1920 * 1080).astype(np.float16)  # Specular reflectance for highlights
output_color = np.empty_like(light_intensity)  # Array to store the resulting color intensity

# Define the scaling factors for diffuse and specular contributions
alpha = 0.7  # Weight for the diffuse component
beta = 0.3   # Weight for the specular component

# Formula: color = alpha * light_intensity * diffuse_component + beta * specular_component
fma(light_intensity, diffuse_component, specular_component,
    dtype="float16", # Optional, unless it can't be inferred from the input
    alpha=alpha, beta=beta, out=output_color)

# Slow analog with NumPy for comparison
slow_output_color = (alpha * light_intensity * diffuse_component + beta * specular_component).astype(np.float16)
```

## Multithreading and Memory Usage

By default, computations use a single CPU core.
To override this behavior, use the `threads` argument.
Set it to `0` to use all available CPU cores and let the underlying C library manage the thread pool.
Here is an example of dealing with large sets of binary vectors:

```py
ndim = 1536 # OpenAI Ada embeddings
matrix1 = np.packbits(np.random.randint(2, size=(10_000, ndim)).astype(np.uint8))
matrix2 = np.packbits(np.random.randint(2, size=(1_000, ndim)).astype(np.uint8))

distances = numkong.cdist(matrix1, matrix2,
    metric="hamming",   # Unlike SciPy, NumKong doesn't divide by the number of dimensions
    out_dtype="uint8",  # so we can use `uint8` instead of `float64` to save memory.
    threads=0,          # Use all CPU cores with OpenMP.
    dtype="bin8",       # Override input argument type to `bin8` eight-bit words.
)
```

Alternatively, when using free-threading Python 3.13t builds, one can combine single-threaded NumKong operations with Python's `concurrent.futures.ThreadPoolExecutor` to parallelize the computations.
By default, the output distances will be stored in double-precision `float64` floating-point numbers.
That behavior may not be space-efficient, especially if you are computing the hamming distance between short binary vectors, that will generally fit into 8x smaller `uint8` or `uint16` types.
To override this behavior, use the `out_dtype` argument, or consider pre-allocating the output array and passing it to the `out` argument.
A more complete example may look like this:

```py
from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor
from numkong import angular
import numpy as np

# Generate large dataset
vectors_a = np.random.rand(100_000, 1536).astype(np.float32)
vectors_b = np.random.rand(100_000, 1536).astype(np.float32)
distances = np.zeros((100_000,), dtype=np.float32)

def compute_batch(start_idx, end_idx):
    batch_a = vectors_a[start_idx:end_idx]
    batch_b = vectors_b[start_idx:end_idx]
    angular(batch_a, batch_b, out=distances[start_idx:end_idx])

# Use all CPU cores with true parallelism (no GIL!)
num_threads = cpu_count()
chunk_size = len(vectors_a) // num_threads

with ThreadPoolExecutor(max_workers=num_threads) as executor:
    futures = []
    for i in range(num_threads):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size if i < num_threads - 1 else len(vectors_a)
        futures.append(executor.submit(compute_batch, start_idx, end_idx))

    # Collect results from all threads
    results = [future.result() for future in futures]
```

## Half-Precision Brain-Float Numbers

The "brain-float-16" is a popular machine learning format.
It's broadly supported in hardware and is very machine-friendly, but software support is still lagging behind.
[Unlike NumPy](https://github.com/numpy/numpy/issues/19808), you can already use `bf16` datatype in NumKong.
Luckily, to downcast `f32` to `bf16` you only have to drop the last 16 bits:

```py
import numpy as np
import numkong as nk

ndim = 1536
a = np.random.randn(ndim).astype(np.float32)
b = np.random.randn(ndim).astype(np.float32)

# NumPy doesn't natively support brain-float, so we need a trick!
# Luckily, it's very easy to reduce the representation accuracy
# by simply masking the low 16-bits of our 32-bit single-precision
# numbers. We can also add `0x8000` to round the numbers.
a_f32rounded = ((a.view(np.uint32) + 0x8000) & 0xFFFF0000).view(np.float32)
b_f32rounded = ((b.view(np.uint32) + 0x8000) & 0xFFFF0000).view(np.float32)

# To represent them as brain-floats, we need to drop the second half
a_bf16 = np.right_shift(a_f32rounded.view(np.uint32), 16).astype(np.uint16)
b_bf16 = np.right_shift(b_f32rounded.view(np.uint32), 16).astype(np.uint16)

# Now we can compare the results
expected = np.inner(a_f32rounded, b_f32rounded)
result = nk.inner(a_bf16, b_bf16, "bf16")
```

## Helper Functions

You can turn specific backends on or off depending on the exact environment.
A common case may be avoiding AVX-512 on older AMD CPUs and [Intel Ice Lake](https://travisdowns.github.io/blog/2020/08/19/icl-avx512-freq.html) CPUs to ensure the CPU doesn't change the frequency license and throttle performance.

```py
$ numkong.get_capabilities()
> {'serial': True, 'neon': False, 'sve': False, 'neonhalf': False, 'svehalf': False, 'neonbfdot': False, 'svebfdot': False, 'neonsdot': False, 'svesdot': False, 'haswell': True, 'skylake': True, 'ice': True, 'genoa': True, 'sapphire': True, 'turin': True}
$ numkong.disable_capability("sapphire")
$ numkong.enable_capability("sapphire")
```

## Using Python API with USearch

Want to use it in Python with [USearch](https://github.com/unum-cloud/usearch)?
You can wrap the raw C function pointers NumKong backends into a `CompiledMetric` and pass it to USearch, similar to how it handles Numba's JIT-compiled code.

```py
from usearch.index import Index, CompiledMetric, MetricKind, MetricSignature
from numkong import pointer_to_sqeuclidean, pointer_to_angular, pointer_to_inner

metric = CompiledMetric(
    pointer=pointer_to_angular("f16"),
    kind=MetricKind.Cos,
    signature=MetricSignature.ArrayArraySize,
)

index = Index(256, metric=metric)
```

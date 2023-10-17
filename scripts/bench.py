import timeit
import argparse

import numpy as np
import scipy.spatial.distance as spd
import simsimd as simd

# Argument parsing
parser = argparse.ArgumentParser(description="Benchmark SimSIMD vs. SciPy")
parser.add_argument(
    "--n", type=int, default=1000, help="Number of vectors (default: 1000)"
)
parser.add_argument(
    "--ndim", type=int, default=1536, help="Number of dimensions (default: 1536)"
)
args = parser.parse_args()

count = args.n
ndim = args.ndim


# Benchmark function
def benchmark(func, A, B, count=1):
    start_time = timeit.default_timer()
    func(A, B)
    end_time = timeit.default_timer()
    return (end_time - start_time) / count


def wrap_rowwise(conventional_func):
    def wrapped(A, B):
        for i in range(A.shape[0]):
            conventional_func(A[i], B[i])

    return wrapped


print()
print("# Benchmarking SimSIMD vs. SciPy")
print()

print("- Vector dimensions:", ndim)
print("- Vectors count:", count)

try:
    caps = [cap for cap, enabled in simd.get_capabilities().items() if enabled]
    print("- Hardware capabilities:", ", ".join(caps))

    deps: dict = np.show_config(mode="dicts").get("Build Dependencies")
    print("- NumPy BLAS dependency:", deps["blas"]["name"])
    print("- NumPy LAPACK dependency:", deps["lapack"]["name"])

except:
    pass

count = 1000
ndim = 1536

generators = {
    np.float32: lambda: np.random.randn(count, ndim).astype(np.float32),
    np.float16: lambda: np.random.randn(count, ndim).astype(np.float16),
    np.int8: lambda: np.random.randint(-100, 100, (count, ndim), np.int8),
    np.uint8: lambda: np.packbits(
        np.random.randint(0, 2, (count, ndim), np.uint8), axis=0
    ),
}

dtype_names = {
    np.float32: "f32",
    np.float16: "f16",
    np.int8: "i8",
    np.uint8: "u8",
}


print()
print("## Between 2 Vectors, Batch Size: 1")
print()

# Table headers
print(
    "| Datatype | Method                |                Ops/s |        SimSIMD Ops/s | SimSIMD Improvement |"
)
print(
    "| :------- | :-------------------- | -------------------: | -------------------: | ------------------: |"
)

# Benchmark functions
funcs = [
    ("scipy.cosine", spd.cosine, simd.cosine, [np.float32, np.float16, np.int8]),
    (
        "scipy.sqeuclidean",
        spd.sqeuclidean,
        simd.sqeuclidean,
        [np.float32, np.float16, np.int8],
    ),
    ("numpy.inner", np.inner, simd.inner, [np.float32, np.float16, np.int8]),
    ("scipy.hamming", spd.hamming, simd.hamming, [np.uint8]),
    ("scipy.jaccard", spd.jaccard, simd.jaccard, [np.uint8]),
]


for name, conventional_func, simd_func, dtypes in funcs:
    for dtype in dtypes:
        A = generators[dtype]()
        B = generators[dtype]()

        conventional_time = benchmark(wrap_rowwise(conventional_func), A, B, count)
        simd_time = benchmark(wrap_rowwise(simd_func), A, B, count)

        conventional_ops = 1 / conventional_time
        simd_ops = 1 / simd_time
        improvement = conventional_time / simd_time

        func_name = "`{}`".format(name)
        dtype_name = "`{}`".format(dtype_names[dtype])
        print(
            f"| {dtype_name:8} | {func_name:21} | {conventional_ops:20,.0f} | {simd_ops:20,.0f} | {improvement:17.2f} x |"
        )

print()


print("## Between 2 Vectors, Batch Size: {:,}".format(count))
print()

# Table headers
print(
    "| Datatype | Method                |                Ops/s |        SimSIMD Ops/s | SimSIMD Improvement |"
)
print(
    "| :------- | :-------------------- | -------------------: | -------------------: | ------------------: |"
)

# Benchmark functions
funcs = [
    ("scipy.cosine", spd.cosine, simd.cosine, [np.float32, np.float16, np.int8]),
    (
        "scipy.sqeuclidean",
        spd.sqeuclidean,
        simd.sqeuclidean,
        [np.float32, np.float16, np.int8],
    ),
    ("numpy.inner", np.inner, simd.inner, [np.float32, np.float16, np.int8]),
    ("scipy.hamming", spd.hamming, simd.hamming, [np.uint8]),
    ("scipy.jaccard", spd.jaccard, simd.jaccard, [np.uint8]),
]


for name, conventional_func, simd_func, dtypes in funcs:
    for dtype in dtypes:
        A = generators[dtype]()
        B = generators[dtype]()

        conventional_time = benchmark(wrap_rowwise(conventional_func), A, B, count)
        simd_time = benchmark(simd_func, A, B, count)

        conventional_ops = 1 / conventional_time
        simd_ops = 1 / simd_time
        improvement = conventional_time / simd_time

        func_name = "`{}`".format(name)
        dtype_name = "`{}`".format(dtype_names[dtype])
        print(
            f"| {dtype_name:8} | {func_name:21} | {conventional_ops:20,.0f} | {simd_ops:20,.0f} | {improvement:17.2f} x |"
        )

print()


# Benchmark functions for cdist
print()
print("## Between All Pairs of Vectors (`cdist`), Batch Size: {:,}".format(count))
print()


cdist_funcs = [
    (
        "scipy.cosine",
        lambda A, B: spd.cdist(A, B, "cosine"),
        lambda A, B: simd.cdist(A, B, "cosine"),
        [np.float32, np.float16, np.int8],
    ),
    (
        "scipy.sqeuclidean",
        lambda A, B: spd.cdist(A, B, "sqeuclidean"),
        lambda A, B: simd.cdist(A, B, "sqeuclidean"),
        [np.float32, np.float16, np.int8],
    ),
    (
        "numpy.inner",
        lambda A, B: 1 - np.dot(A, B.T),
        lambda A, B: simd.cdist(A, B, "inner"),
        [np.float32, np.float16, np.int8],
    ),
    (
        "scipy.hamming",
        lambda A, B: spd.cdist(A, B, "hamming"),
        lambda A, B: simd.cdist(A, B, "hamming"),
        [np.uint8],
    ),
    (
        "scipy.jaccard",
        lambda A, B: spd.cdist(A, B, "jaccard"),
        lambda A, B: simd.cdist(A, B, "jaccard"),
        [np.uint8],
    ),
]


# Table headers
print(
    "| Datatype | Method                |                Ops/s |        SimSIMD Ops/s | SimSIMD Improvement |"
)
print(
    "| :------- | :-------------------- | -------------------: | -------------------: | ------------------: |"
)

for name, conventional_func, simd_func, dtypes in cdist_funcs:
    for dtype in dtypes:
        A = generators[dtype]()
        B = generators[dtype]()

        conventional_time = benchmark(conventional_func, A, B)
        simd_time = benchmark(simd_func, A, B)

        conventional_ops = count**2 / conventional_time
        simd_ops = count**2 / simd_time
        improvement = conventional_time / simd_time

        func_name = "`{}`".format(name)
        dtype_name = "`{}`".format(dtype_names[dtype])
        print(
            f"| {dtype_name:8} | {func_name:21} | {conventional_ops:20,.0f} | {simd_ops:20,.0f} | {improvement:17.2f} x |"
        )

print()

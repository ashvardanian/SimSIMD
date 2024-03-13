import timeit
import argparse

import numpy as np
import scipy as sp
import sklearn as sk
import scipy.spatial.distance as spd
import scipy.special as scs
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

# Set to ignore all floating-point errors
np.seterr(all="ignore")


def benchmark(func, A, B, count=1):
    """Time the amount of time it takes to run a function and return the average time per run in seconds."""
    start_time = timeit.default_timer()
    func(A, B)
    end_time = timeit.default_timer()
    return (end_time - start_time) / count


def wrap_rowwise(baseline_func):
    """Wrap a function to apply it row-wise to rows of two matrices."""

    def wrapped(A, B):
        for i in range(A.shape[0]):
            baseline_func(A[i], B[i])

    return wrapped


def print_makrkdown_row(dtype, name, baseline_time, simd_time):
    """Print a formatted row for the markdown table."""
    baseline_ops = f"{1 / baseline_time:,.0f}" if baseline_time is not None else "💥"
    simd_ops = f"{1 / simd_time:,.0f}" if simd_time is not None else "💥"
    improvement = (
        f"{baseline_time / simd_time:,.2f} x" if simd_time and baseline_time else "🤷"
    )

    func_name = "`{}`".format(name)
    dtype_name = "`{}`".format(dtype_names[dtype])

    # Print the formatted line
    print(
        f"| {dtype_name:8} | {func_name:21} | {baseline_ops:20} | {simd_ops:20} | {improvement:17} |"
    )


def raise_(ex):
    """Utility function to allow raising exceptions in lambda functions."""
    raise ex


print()
print("# Benchmarking SimSIMD vs. SciPy")
print()

print("- Vector dimensions:", ndim)
print("- Vectors count:", count)

try:
    caps = [cap for cap, enabled in simd.get_capabilities().items() if enabled]
    print("- Hardware capabilities:", ", ".join(caps))

    # Log versions of SimSIMD, NumPy, SciPy, and scikit-learn
    print(f"- SimSIMD version: {simd.__version__}")
    print(f"- SciPy version: {sp.__version__}")
    print(f"- scikit-learn version: {sk.__version__}")
    print(f"- NumPy version: {np.__version__}")

    deps: dict = np.show_config(mode="dicts").get("Build Dependencies")
    print("-- NumPy BLAS dependency:", deps["blas"]["name"])
    print("-- NumPy LAPACK dependency:", deps["lapack"]["name"])
except Exception as e:
    print(f"An error occurred: {e}")

count = 1000
ndim = 1536

generators = {
    np.complex128: lambda: (
        np.random.randn(count, ndim // 2).astype(np.float64)
        + 1j * np.random.randn(count, ndim // 2).astype(np.float64)
    ).view(np.complex128),
    np.complex64: lambda: (
        np.random.randn(count, ndim // 2).astype(np.float32)
        + 1j * np.random.randn(count, ndim // 2).astype(np.float32)
    ).view(np.complex64),
    "complex32": lambda: np.random.randn(count, ndim).astype(np.float16),
    np.float64: lambda: np.random.randn(count, ndim).astype(np.float64),
    np.float32: lambda: np.random.randn(count, ndim).astype(np.float32),
    np.float16: lambda: np.random.randn(count, ndim).astype(np.float16),
    np.int8: lambda: np.random.randint(
        -100, high=100, size=(count, ndim), dtype=np.int8
    ),
    np.uint8: lambda: np.packbits(
        np.random.randint(0, high=2, size=(count, ndim), dtype=np.uint8), axis=0
    ),
}

dtype_names = {
    np.complex128: "f64c",
    np.complex64: "f32c",
    "complex32": "f16c",
    np.float64: "f64",
    np.float32: "f32",
    np.float16: "f16",
    np.int8: "i8",
    np.uint8: "b8",
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
    (
        "scipy.cosine",
        spd.cosine,
        simd.cosine,
        [np.float64, np.float32, np.float16, np.int8],
    ),
    (
        "scipy.sqeuclidean",
        spd.sqeuclidean,
        simd.sqeuclidean,
        [np.float64, np.float32, np.float16, np.int8],
    ),
    (
        "numpy.dot",
        np.dot,
        simd.dot,
        [np.float64, np.float32, np.float16, np.int8, np.complex64, np.complex128],
    ),
    (
        "numpy.dot",
        lambda A, B: raise_(NotImplementedError("Not implemented for complex32")),
        lambda A, B: simd.dot(A, B, "complex32"),
        ["complex32"],
    ),
    (
        "numpy.vdot",
        np.vdot,
        simd.vdot,
        [np.complex64, np.complex128],
    ),
    (
        "scipy.jensenshannon",
        spd.jensenshannon,
        simd.jensenshannon,
        [np.float64, np.float32, np.float16],
    ),
    (
        "scipy.kl_div",
        scs.kl_div,
        simd.kullbackleibler,
        [np.float64, np.float32, np.float16],
    ),
    ("scipy.hamming", spd.hamming, simd.hamming, [np.uint8]),
    ("scipy.jaccard", spd.jaccard, simd.jaccard, [np.uint8]),
]


for name, baseline_func, simd_func, dtypes in funcs:
    for dtype in dtypes:
        A = generators[dtype]()
        B = generators[dtype]()
        baseline_time = None
        simd_time = None

        # Try obtaining the measurements
        try:
            baseline_time = benchmark(wrap_rowwise(baseline_func), A, B, count)
        except Exception as e:
            # raise RuntimeError(str(e) + " for %s(%s)" % (name, str(dtype))) from e
            pass

        try:
            simd_time = benchmark(wrap_rowwise(simd_func), A, B, count)
        except Exception as e:
            # raise RuntimeError(str(e) + " for %s(%s)" % (name, str(dtype))) from e
            pass

        print_makrkdown_row(dtype, name, baseline_time, simd_time)

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
    (
        "scipy.cosine",
        wrap_rowwise(spd.cosine),
        simd.cosine,
        [np.float64, np.float32, np.float16, np.int8],
    ),
    (
        "scipy.sqeuclidean",
        wrap_rowwise(spd.sqeuclidean),
        simd.sqeuclidean,
        [np.float64, np.float32, np.float16, np.int8],
    ),
    (
        "numpy.dot",
        lambda A, B: np.sum(A * B, axis=1),
        simd.dot,
        [np.float64, np.float32, np.float16, np.int8, np.complex64, np.complex128],
    ),
    (
        "scipy.jensenshannon",
        wrap_rowwise(spd.jensenshannon),
        simd.jensenshannon,
        [np.float64, np.float32, np.float16],
    ),
    (
        "scipy.kl_div",
        wrap_rowwise(scs.kl_div),
        simd.kullbackleibler,
        [np.float64, np.float32, np.float16],
    ),
    ("scipy.hamming", wrap_rowwise(spd.hamming), simd.hamming, [np.uint8]),
    ("scipy.jaccard", wrap_rowwise(spd.jaccard), simd.jaccard, [np.uint8]),
]


for name, baseline_func, simd_func, dtypes in funcs:
    for dtype in dtypes:
        A = generators[dtype]()
        B = generators[dtype]()
        baseline_time = None
        simd_time = None

        try:
            baseline_time = benchmark(baseline_func, A, B, count)
        except Exception:
            pass
        try:
            simd_time = benchmark(simd_func, A, B, count)
        except Exception:
            pass

        print_makrkdown_row(dtype, name, baseline_time, simd_time)


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
        "numpy.dot",
        lambda A, B: np.dot(A, B.T),
        lambda A, B: simd.cdist(A, B, "dot"),
        [np.float32, np.float16, np.int8, np.complex64, np.complex128],
    ),
    (
        "scipy.jensenshannon",
        lambda A, B: spd.cdist(A, B, "jensenshannon"),
        lambda A, B: simd.cdist(A, B, "jensenshannon"),
        [np.float32, np.float16],
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

for name, baseline_func, simd_func, dtypes in cdist_funcs:
    for dtype in dtypes:
        A = generators[dtype]()
        B = generators[dtype]()
        baseline_time = None
        simd_time = None

        try:
            baseline_time = benchmark(baseline_func, A, B)
        except Exception:
            pass
        try:
            simd_time = benchmark(simd_func, A, B)
        except Exception:
            pass

        print_makrkdown_row(
            dtype,
            name,
            baseline_time / count**2 if baseline_time else None,
            simd_time / count**2 if simd_time else None,
        )

print()

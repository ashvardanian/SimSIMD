#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# This script benchmarks the performance of SimSIMD against other libraries,
# such as NumPy, SciPy, scikit-learn, PyTorch, TensorFlow, and JAX.
# It can operate in 2 modes: Batch mode and All-Pairs mode.
import os
import timeit
import argparse
from typing import List, Union
from dataclasses import dataclass

import simsimd as simd

metric_names = [
    "dot",  # Dot product
    "spatial",  # Euclidean and Cosine distance
    "binary",  # Hamming and Jaccard distance for binary vectors
    "probability",  # Jensen-Shannon and Kullback-Leibler divergences for probability distributions
    "sparse",  # Intersection of two sparse integer sets, with float/int weights
]
dtype_names = [
    "bits",  #! Not supported by SciPy
    "int8",  #! Presented as supported, but overflows most of the time
    "uint16",
    "uint32",
    "float16",
    "float32",
    "float64",
    "bfloat16",  #! Not supported by NumPy
    "complex32",  #! Not supported by NumPy
    "complex64",
    "complex128",
]

# Argument parsing
parser = argparse.ArgumentParser(description="Benchmark SimSIMD vs. other libraries")
parser.add_argument(
    "--ndim",
    type=int,
    default=1536,
    help="""
        Number of dimensions in vectors (default: 1536)
                    
        For binary vectors (e.g., Hamming, Jaccard), this is the number of bits.
        In case of SimSIMD, the inputs will be treated at the bit-level.
        Other packages will be matching/comparing 8-bit integers.
        The volume of exchanged data will be identical, but the results will differ.
        """,
)
parser.add_argument(
    "-n",
    "--count",
    type=int,
    default=1,
    help="""
        Number of vectors per batch (default: 1)
        
        By default, when set to 1 the benchmark will generate many vectors of size (ndim, )
        and call the functions on pairs of single vectors: both directly, and through `cdist`.
        Alternatively, for larger batch sizes the benchmark will generate two matrices of 
        size (n, ndim) and compute:
         
        - batch mode: (n) distances between vectors in identical rows of the two matrices,
        - all-pairs mode: (n^2) distances between all pairs of vectors in the two matrices via `cdist`.
        """,
)
parser.add_argument(
    "--metric",
    choices=["all", *metric_names],
    default="all",
    help="Distance metric to use, profiles everything by default",
)
parser.add_argument(
    "--dtype",
    choices=["all", *dtype_names],
    default="all",
    help="Defines numeric types to benchmark, profiles everything by default",
)
parser.add_argument("--scipy", action="store_true", help="Profile SciPy, must be installed")
parser.add_argument("--scikit", action="store_true", help="Profile scikit-learn, must be installed")
parser.add_argument("--torch", action="store_true", help="Profile PyTorch, must be installed")
parser.add_argument("--tf", action="store_true", help="Profile TensorFlow, must be installed")
parser.add_argument("--jax", action="store_true", help="Profile JAX, must be installed")
args = parser.parse_args()

args.count = 10
args.scipy = True
args.scikit = True
args.torch = True
args.tf = True
args.jax = True

# Conditionally import libraries based on arguments
import numpy as np

# Set to ignore all floating-point errors
np.seterr(all="ignore")

if args.scipy:
    import scipy as sp
    import scipy.spatial.distance as spd
    import scipy.special as scs

if args.scikit:
    import sklearn as sk
    import sklearn.metrics.pairwise as skp
if args.torch:
    import torch
if args.tf:
    # Disable TensorFlow warning messages
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # This hides INFO and WARNING messages

    import tensorflow as tf

    # This will show only ERROR messages, not WARNING messages.
    # Additionally, to filter out oneDNN related warnings, you might need to:
    tf.get_logger().setLevel("FATAL")

if args.jax:
    import jax.numpy as jnp
    import jax

args = parser.parse_args()
count = args.count
ndim = args.ndim
assert count > 0, "Number of vectors per batch must be greater than 0"
assert ndim > 0, "Number of dimensions must be greater than 0"

dtypes_profiled = set([args.dtype] if args.dtype != "all" else dtype_names)
metrics_profiled = set([args.metric] if args.metric != "all" else metric_names)


@dataclass
class BatchKernel:
    name: str
    baseline_one_to_one_func: callable
    baseline_many_to_many_func: callable
    simsimd_func: callable
    dtypes: list
    tensor_type: callable


@dataclass
class AllPairsKernel:
    name: str
    baseline_func: callable
    simsimd_func: callable
    dtypes: list


@dataclass
class Result:
    dtype: str
    name: str
    baseline_calls_per_sec: float
    simsimd_calls_per_sec: float
    bytes_per_vector: int
    distance_calculations: int
    error: str = None


def filter_dtypes(*dtypes) -> list:
    """Filter out unsupported data types."""
    return [dtype for dtype in dtypes if dtype in dtypes_profiled]


def benchmark(func, A, B, count=1):
    """Time the amount of time it takes to run a function and return the average time per run in seconds."""
    start_time = timeit.default_timer()
    func(A, B)
    end_time = timeit.default_timer()
    return (end_time - start_time) / count


def wrap_row_calls(baseline_one_to_one_func):
    """Wrap a function to apply it row-wise to rows of two matrices."""

    def wrapped(A, B):
        for i in range(A.shape[0]):
            baseline_one_to_one_func(A[i], B[i])

    return wrapped


def print_markdown_text(dtype, name, baseline_ops, simd_ops, improvement):
    print(f"| {dtype:8} | {name:30} | {baseline_ops:18} | {simd_ops:18} | {improvement:17} |")


def print_markdown_row(dtype, name, baseline_time, simsimd_time):
    """Print a formatted row for the markdown table."""
    baseline_ops = f"{1 / baseline_time:,.0f}" if baseline_time is not None else "💥"
    simd_ops = f"{1 / simsimd_time:,.0f}" if simsimd_time is not None else "💥"
    improvement = f"{baseline_time / simsimd_time:,.2f} x" if simsimd_time and baseline_time else "🤷"
    name = f"`{name}`"
    dtype = f"`{dtype}`"
    print_markdown_text(dtype, name, baseline_ops, simd_ops, improvement)


def raise_(ex):
    """Utility function to allow raising exceptions in lambda functions."""
    raise ex


# Print the benchmarking environment details
print("# Benchmarking SimSIMD")
print("- Vector dimensions:", ndim)
print("- Vectors count:", count)
print("- Metrics:", ", ".join(metrics_profiled))
print("- Datatypes:", ", ".join(dtypes_profiled))
try:
    caps = [cap for cap, enabled in simd.get_capabilities().items() if enabled]
    print("- Hardware capabilities:", ", ".join(caps))

    # Log versions of SimSIMD, NumPy, SciPy, and scikit-learn
    print(f"- SimSIMD version: {simd.__version__}")
    print(f"- NumPy version: {np.__version__}")

    if args.scipy:
        print(f"- SciPy version: {sp.__version__}")
    if args.scikit:
        print(f"- scikit-learn version: {sk.__version__}")
    if args.torch:
        print(f"- PyTorch version: {torch.__version__}")
    if args.tf:
        print(f"- TensorFlow version: {tf.__version__}")
    if args.jax:
        print(f"- JAX version: {jax.__version__}")

    deps: dict = np.show_config(mode="dicts").get("Build Dependencies")
    print("-- NumPy BLAS dependency:", deps["blas"]["name"])
    print("-- NumPy LAPACK dependency:", deps["lapack"]["name"])
except Exception as e:
    print(f"An error occurred: {e}")


print()
print("## One-to-One Vector Operations")
print()


generators = {
    "complex128": lambda: (
        np.random.randn(count, ndim // 2).astype(np.float64) + 1j * np.random.randn(count, ndim // 2).astype(np.float64)
    ).view(np.complex128),
    "complex64": lambda: (
        np.random.randn(count, ndim // 2).astype(np.float32) + 1j * np.random.randn(count, ndim // 2).astype(np.float32)
    ).view(np.complex64),
    "complex32": lambda: np.random.randn(count, ndim).astype(np.float16),
    "float64": lambda: np.random.randn(count, ndim).astype(np.float64),
    "float32": lambda: np.random.randn(count, ndim).astype(np.float32),
    "float16": lambda: np.random.randn(count, ndim).astype(np.float16),
    "bfloat16": lambda: np.random.randint(0, high=256, size=(count, ndim), dtype=np.int16),
    "int8": lambda: np.random.randint(-100, high=100, size=(count, ndim), dtype=np.int8),
    "bits": lambda: np.packbits(np.random.randint(0, high=2, size=(count, ndim), dtype=np.uint8), axis=0),
}

dtype_names = {
    "complex128": "f64c",
    "complex64": "f32c",
    "complex32": "f16c",
    "float64": "f64",
    "float32": "f32",
    "float16": "f16",
    "bfloat16": "bf16",
    "int8": "i8",
    "bits": "b8",
}

# Table headers
print_markdown_text("Datatype", "Method", "Ops/s", "SimSIMD Ops/s", "SimSIMD Improvement")
print_markdown_text(":-------", ":------", "-----:", "-------------:", "------------------:")


# Benchmark functions
batch_kernels: List[BatchKernel] = []

if "dot" in metrics_profiled:
    batch_kernels.extend(
        [
            BatchKernel(
                "numpy.dot",
                np.dot,
                lambda A, B: np.sum(A * B, axis=1),
                simd.dot,
                filter_dtypes("float64", "float32", "float16", "int8", "complex64", "complex128"),
                np.array,
            ),
            BatchKernel(
                "numpy.dot",
                lambda A, B: raise_(NotImplementedError("Not implemented for complex32")),
                lambda A, B: raise_(NotImplementedError("Not implemented for complex32")),
                lambda A, B: simd.dot(A, B, "complex32"),
                filter_dtypes("complex32"),
                np.array,
            ),
            BatchKernel(
                "numpy.dot",
                lambda A, B: raise_(NotImplementedError("Not implemented for bfloat16")),
                lambda A, B: raise_(NotImplementedError("Not implemented for bfloat16")),
                lambda A, B: simd.dot(A, B, "bfloat16"),
                filter_dtypes("bfloat16"),
                np.array,
            ),
            BatchKernel(
                "numpy.vdot",
                np.vdot,
                wrap_row_calls(np.vdot),
                simd.vdot,
                filter_dtypes("complex64", "complex128"),
                np.array,
            ),
        ]
    )

if "spatial" in metrics_profiled and args.scipy:
    batch_kernels.extend(
        [
            BatchKernel(
                "scipy.cosine",
                spd.cosine,
                wrap_row_calls(spd.cosine),
                simd.cosine,
                filter_dtypes("float64", "float32", "float16", "int8"),
                np.array,
            ),
            BatchKernel(
                "scipy.cosine",
                lambda A, B: raise_(NotImplementedError(f"Not implemented for bfloat16")),
                lambda A, B: raise_(NotImplementedError(f"Not implemented for bfloat16")),
                lambda A, B: simd.cosine(A, B, "bfloat16"),
                filter_dtypes("bfloat16"),
                np.array,
            ),
            BatchKernel(
                "scipy.sqeuclidean",
                spd.sqeuclidean,
                wrap_row_calls(spd.sqeuclidean),
                simd.sqeuclidean,
                filter_dtypes("float64", "float32", "float16", "int8"),
                np.array,
            ),
        ]
    )

if "probability" in metrics_profiled and args.scipy:
    batch_kernels.extend(
        [
            BatchKernel(
                "scipy.jensenshannon",
                spd.jensenshannon,
                wrap_row_calls(spd.jensenshannon),
                simd.jensenshannon,
                filter_dtypes("float64", "float32", "float16"),
                np.array,
            ),
            BatchKernel(
                "scipy.kl_div",
                scs.kl_div,
                wrap_row_calls(scs.kl_div),
                simd.kullbackleibler,
                filter_dtypes("float64", "float32", "float16"),
                np.array,
            ),
        ]
    )

if "binary" in metrics_profiled and args.scipy:
    batch_kernels.extend(
        [
            BatchKernel(
                "scipy.hamming",
                spd.hamming,
                wrap_row_calls(spd.hamming),
                lambda a, b: simd.hamming(a, b, "b8"),
                filter_dtypes("bits"),
                np.array,
            ),
            BatchKernel(
                "scipy.jaccard",
                spd.jaccard,
                wrap_row_calls(spd.jaccard),
                lambda a, b: simd.jaccard(a, b, "b8"),
                filter_dtypes("bits"),
                np.array,
            ),
        ]
    )

if "spatial" in metrics_profiled and args.scikit:
    batch_kernels.extend(
        [
            BatchKernel(
                "sklearn.cosine_similarity",
                lambda A, B: skp.cosine_similarity(A.reshape(1, ndim), B.reshape(1, ndim)),
                skp.paired_cosine_distances,
                simd.cosine,
                filter_dtypes("float64", "float32", "float16", "int8"),
                np.array,
            ),
            BatchKernel(
                "sklearn.euclidean_distances",
                lambda A, B: skp.euclidean_distances(A.reshape(1, ndim), B.reshape(1, ndim)),
                skp.paired_euclidean_distances,
                simd.sqeuclidean,
                filter_dtypes("float64", "float32", "float16", "int8"),
                np.array,
            ),
        ]
    )

if "dot" in metrics_profiled and args.tf:
    batch_kernels.extend(
        [
            BatchKernel(
                "tensorflow.tensordot",
                lambda A, B: tf.tensordot(A, B, axes=1),
                lambda A, B: raise_(NotImplementedError("Should be an easy patch")),  # TODO
                simd.dot,
                filter_dtypes("float64", "float32", "float16", "int8"),
                tf.convert_to_tensor,
            )
        ]
    )

if "dot" in metrics_profiled and args.jax:
    batch_kernels.extend(
        [
            BatchKernel(
                "jax.numpy.dot",
                lambda A, B: jnp.dot(A, B).block_until_ready(),
                lambda A, B: raise_(NotImplementedError("Should be an easy patch")),  # TODO
                simd.dot,
                filter_dtypes("float64", "float32", "float16", "int8"),
                jnp.array,
            )
        ]
    )

if "dot" in metrics_profiled and args.torch:
    batch_kernels.extend(
        [
            BatchKernel(
                "torch.dot",
                lambda A, B: torch.dot(A, B).item(),
                lambda A, B: raise_(NotImplementedError("Should be an easy patch")),  # TODO
                simd.dot,
                filter_dtypes("float64", "float32", "float16", "int8"),
                torch.tensor,
            ),
        ]
    )


for kernel in batch_kernels:
    for dtype in kernel.dtypes:
        A = generators[dtype]()
        B = generators[dtype]()
        if count == 1:
            A = A.flatten()
            B = B.flatten()

        At = kernel.tensor_type(A)
        Bt = kernel.tensor_type(B)
        baseline_time = None
        simsimd_time = None
        baseline_one_to_one_func = kernel.baseline_one_to_one_func if count == 1 else kernel.baseline_many_to_many_func
        simsimd_func = kernel.simsimd_func

        # Try obtaining the measurements
        try:
            baseline_time = benchmark(baseline_one_to_one_func, At, Bt, count)
        except NotImplementedError:
            pass
        except ValueError:
            pass  #! This happens often during overflows
        except RuntimeError:
            pass  #! This happens often during overflows
        except Exception as e:
            raise RuntimeError(str(e) + " for %s(%s)" % (kernel.name, str(dtype))) from e

        try:
            simsimd_time = benchmark(simsimd_func, A, B, count)
        except NotImplementedError:
            pass
        except Exception as e:
            raise RuntimeError(str(e) + " for %s(%s)" % (kernel.name, str(dtype))) from e

        print_markdown_row(dtype, kernel.name, baseline_time, simsimd_time)

print()


# Benchmark functions for cdist
print()
print("## Between All Pairs of Vectors (`cdist`), Batch Size: {:,}".format(count))
print()


all_pairs_kernels: List[AllPairsKernel] = []

if "dot" in metrics_profiled:
    all_pairs_kernels.extend(
        [
            AllPairsKernel(
                "numpy.dot",
                lambda A, B: np.dot(A, B.T),
                lambda A, B: simd.cdist(A, B, metric="dot"),
                filter_dtypes("float32", "float16", "int8", "complex64", "complex128"),
            ),
        ]
    )

if "spatial" in metrics_profiled and args.scipy:
    all_pairs_kernels.extend(
        [
            AllPairsKernel(
                "scipy.cosine",
                lambda A, B: spd.cdist(A, B, "cosine"),
                lambda A, B: simd.cdist(A, B, metric="cosine"),
                filter_dtypes("float32", "float16", "int8"),
            ),
            AllPairsKernel(
                "scipy.sqeuclidean",
                lambda A, B: spd.cdist(A, B, "sqeuclidean"),
                lambda A, B: simd.cdist(A, B, metric="sqeuclidean"),
                filter_dtypes("float32", "float16", "int8"),
            ),
            AllPairsKernel(
                "scipy.jensenshannon",
                lambda A, B: spd.cdist(A, B, "jensenshannon"),
                lambda A, B: simd.cdist(A, B, metric="jensenshannon"),
                filter_dtypes("float32", "float16"),
            ),
            AllPairsKernel(
                "scipy.hamming",
                lambda A, B: spd.cdist(A, B, "hamming"),
                lambda A, B: simd.cdist(A, B, metric="hamming", dtype="bits"),
                filter_dtypes("bits"),
            ),
            AllPairsKernel(
                "scipy.jaccard",
                lambda A, B: spd.cdist(A, B, "jaccard"),
                lambda A, B: simd.cdist(A, B, metric="jaccard", dtype="bits"),
                filter_dtypes("bits"),
            ),
        ]
    )


# Table headers
print_markdown_text("Datatype", "Method", "Ops/s", "SimSIMD Ops/s", "SimSIMD Improvement")
print_markdown_text(":-------", ":------", "-----:", "-------------:", "------------------:")

for kernel in all_pairs_kernels:
    for dtype in kernel.dtypes:
        A = generators[dtype]()
        B = generators[dtype]()
        baseline_time = None
        simsimd_time = None

        try:
            baseline_time = benchmark(kernel.baseline_func, A, B)
        except Exception as e:
            raise e
        try:
            simsimd_time = benchmark(kernel.simsimd_func, A, B)
        except Exception as e:
            raise e

        print_markdown_row(
            dtype,
            kernel.name,
            baseline_time / count**2 if baseline_time else None,
            simsimd_time / count**2 if simsimd_time else None,
        )

print()

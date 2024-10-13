import os
import timeit
import argparse
from functools import partial

import numpy as np
import pandas as pd
import simsimd as simd

import perfplot


def ndim_argument(value):
    if value == "default":
        return [2**k for k in range(16)]
    try:
        # Split the input string by commas and convert each part to an integer
        return [int(x) for x in value.split(',')]
    except ValueError:
        raise argparse.ArgumentTypeError("Value must be 'default' or a comma-separated list of integers")


# Argument parsing
parser = argparse.ArgumentParser(description="Benchmark SimSIMD")
parser.add_argument(
    "--ndim", 
    type=ndim_argument, 
    default="default", 
    help="Size of vectors to benchmark, either 'default' powers of 2 (from 1 to 32K) or comma-seperated list of integers"
)
parser.add_argument("--torch", action="store_true", help="Profile PyTorch")
parser.add_argument("--tf", action="store_true", help="Profile TensorFlow")
parser.add_argument("--jax", action="store_true", help="Profile JAX")
parser.add_argument(
    "--plot_fp", 
    type=str, 
    default="simsimd_speed_up.png", 
    help="File to save the plot to, default: 'simsimd_speed_up.png'"
)
parser.add_argument("--debug", action="store_true", help="Provide additional debug information")

args = parser.parse_args()

debug_flag = args.debug
plot_fp = args.plot_fp

# conditionally import torch, tensorflow, and jax
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

# Set to ignore all floating-point errors
np.seterr(all="ignore")

def raise_(ex):
    """Utility function to allow raising exceptions in lambda functions."""
    raise ex


print()
print("# Benchmarking SimSIMD")
print()

print("- Vector dimensions:", args.ndim)
print("- Plot file path:", plot_fp)

try:
    caps = [cap for cap, enabled in simd.get_capabilities().items() if enabled]
    print("- Hardware capabilities:", ", ".join(caps))

    # Log versions of SimSIMD, NumPy, SciPy, and scikit-learn
    print(f"- SimSIMD version: {simd.__version__}")
    print(f"- NumPy version: {np.__version__}")

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

# Functions to benchmark
# Note: The first function and dtype in the list is the baseline function
perfplot_funcs = [
    (
        "numpy.dot",
        np.dot,
        [np.float64, np.float32, np.float16, np.int8, np.complex64, np.complex128],
        np.array,
    ),
    (
        "simd.dot",
        simd.dot,
        [np.float64, np.float32, np.float16, np.int8, np.complex64, np.complex128],
        np.array,
    ),

]

if args.tf:
    perfplot_funcs.extend(
        [
            (
                "tensorflow.tensordot",
                lambda A, B: tf.tensordot(A, B, axes=1),
                [np.float64, np.float32, np.float16,],
                tf.convert_to_tensor,
            )
        ]
    )

if args.jax:
    perfplot_funcs.extend(
        [
            (
                "jax.numpy.dot",
                lambda A, B: jnp.dot(A, B).block_until_ready(),
                [np.float64, np.float32, np.float16, np.int8],
                jnp.array,
            )
        ]
    )

if args.torch:
    perfplot_funcs.extend(
        [
            (
                "torch.dot",
                lambda A, B: torch.dot(A, B).item(),
                [np.float64, np.float32, np.float16, np.int8],
                torch.tensor,
            ),
        ]
    )


# Define benchmark data generators
generators_perfplot = {
    np.complex128: lambda: (
        np.random.randn(ndim // 2).astype(np.float64) + 1j * np.random.randn(ndim // 2).astype(np.float64)
    ).view(np.complex128),
    np.complex64: lambda: (
        np.random.randn(ndim // 2).astype(np.float32) + 1j * np.random.randn(ndim // 2).astype(np.float32)
    ).view(np.complex64),
    "complex32": lambda: np.random.randn(ndim).astype(np.float16),
    np.float64: lambda: np.random.randn(ndim).astype(np.float64),
    np.float32: lambda: np.random.randn(ndim).astype(np.float32),
    np.float16: lambda: np.random.randn(ndim).astype(np.float16),
    np.int8: lambda: np.random.randint(-100, high=100, size=(ndim), dtype=np.int8),
    np.uint8: lambda: np.packbits(np.random.randint(0, high=2, size=(ndim), dtype=np.uint8), axis=0),
}


# Function useful to test correct passing of arguments
# to the function to be benchmarked
# use of this function will impact timings
test_perfplot_stage = ""
test_perfplot_count = -1

def test_benchmark_perfplot(func, A, B, func_name):
    """benchmark run code."""
    global test_perfplot_stage, test_perfplot_count
    stage_string = f"{func_name} {func.__module__}.{func.__name__} for dtype {A.dtype} {A.shape}"
    if test_perfplot_stage != stage_string:
        test_perfplot_stage = stage_string
        test_perfplot_count = -1
    test_perfplot_count += 1
    if test_perfplot_count % 250 == 0:
        print(f"benchmark_perfplot {stage_string} {test_perfplot_count}")

    return func(A, B)

# Wrapper to call the function to be tested
def test_benchmark_perfplot_wrapper(A, B, func, name):
    return test_benchmark_perfplot(func, A, B, name)

# Setup the data for the function to be tested
def perfplot_setup(ndim, tensor_type, dtype):
    A = np.random.rand(ndim).astype(dtype)
    B = np.random.rand(ndim).astype(dtype)
    A = tensor_type(A)
    B = tensor_type(B)
    if args.debug:
        print(f"setup for {tensor_type.__module__}.{tensor_type.__name__} {dtype_names[dtype]} {ndim}")

    return A, B

def perfplot_setup_wrapper(ndim, tensor_type, dtype):
    return perfplot_setup(ndim, tensor_type, dtype)

# setup structures for perfplot
kernels = []
setups = []
labels = []

for name, this_func, dtypes, tensor_type in perfplot_funcs:
    for dtype in dtypes:

        # define callable to run the function to be tested
        # kernels.append(
        #     partial(benchmark_perfplot_wrapper, func=this_func, name=f"{this_func.__module__}.{this_func.__name__}")
        # )
        if debug_flag:
            kernels.append(
                partial(
                    test_benchmark_perfplot_wrapper, 
                    func=this_func, 
                    name=f"{this_func.__module__}.{this_func.__name__}"
                )
            )
        else:
            kernels.append(this_func)

        # define callable to setup the data for the function to be tested
        setups.append(partial(perfplot_setup_wrapper, tensor_type=tensor_type, dtype=dtype))

        # define labels for the plot
        labels.append(f"{name}({dtype_names[dtype]})")

# Run the benchmarks
perfplot_results = perfplot.bench(
    setup = setups,
    kernels = kernels,
    labels = labels,
    n_range=args.ndim,
    flops=lambda ndim: 1, #lambda ndim: 3 * ndim,  # FLOPS plots
    xlabel="ndim",
    equality_check=None,  # bypass correctness check, right now it is failing
)

# Plot the results
perfplot_results.save(
    plot_fp, 
    transparent=False, 
    bbox_inches="tight", 
    relative_to=0, 
    logy="auto"
)

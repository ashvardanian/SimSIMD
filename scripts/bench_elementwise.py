#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module: bench_elementwise.py

This script benchmarks the performance of SimSIMD against other libraries,
such as NumPy, PyTorch, TensorFlow, and JAX on element-wise operations.
It applies not only to tensors/arrays of identical shape, but also to
broadcasting operations along different dimensions.
"""
import os
import time
import argparse
from typing import List, Generator, Union, Tuple, Dict
from dataclasses import dataclass
from itertools import product, chain


#! Before all else, ensure that we use only one thread for each library
os.environ["OMP_NUM_THREADS"] = "1"  # OpenMP
os.environ["MKL_NUM_THREADS"] = "1"  # MKL
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # NumExpr
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # Accelerate
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # OpenBLAS

# NumPy and SimSIMD are obligatory for benchmarking
import numpy as np
import simsimd as simd
import tabulate

# Set to ignore all floating-point errors
np.seterr(all="ignore")


operation_names = [
    "add",  # A + B
    "multiply",  # A * B
]
dtype_names = [
    ("int8", "int8"),  #! Presented as supported, but overflows most of the time
    ("int16", "int16"),
    ("int32", "int32"),
    ("int64", "int64"),
    ("uint8", "uint8"),  #! Presented as supported, but overflows most of the time
    ("uint16", "uint16"),
    ("uint32", "uint32"),
    ("uint64", "uint64"),
    ("bfloat16", "bfloat16"),  #! Not supported by NumPy
    ("float16", "float16"),
    ("float32", "float32"),
    ("float64", "float64"),
    # Common mixed precision variants:
    ("float32", "float64"),  # f32f64
    ("float16", "float32"),  # f16f32
    ("float16", "float64"),  # f16f64
    ("bfloat16", "float32"),  # bf16f32
    ("bfloat16", "float64"),  # bf16f64
    ("uint8", "int16"),  # u8i16
    ("uint16", "int32"),  # u16i32
    ("int8", "uint16"),  # i8u16
    ("int16", "uint32"),  # i16u32
]

dtype_names_supported = [(x, y) for x, y in dtype_names if x != "bfloat16" and y != "bfloat16"]

shapes = [
    ((1,), (1,)),
    ((100000,), (100000,)),
    ((512, 512), (512, 512)),
    ((77, 77, 77), (77, 77, 77)),
    ((31, 31, 31, 31), (31, 31, 31, 31)),
    ((7, 7, 7, 7, 7), (7, 7, 7, 7, 7)),
    ((8, 8, 8, 8, 8), (8, 8, 8, 8, 8)),
    ((4, 4, 4, 4, 4, 4), (4, 4, 4, 4, 4, 4)),
    ((2, 2, 2, 2, 2, 2, 2), (2, 2, 2, 2, 2, 2, 2)),
    ((2, 2, 2, 2, 2, 2, 2, 2), (2, 2, 2, 2, 2, 2, 2, 2)),
    ((2, 2, 2, 2, 2, 2, 2, 2, 2), (2, 2, 2, 2, 2, 2, 2, 2, 2)),
    ((2, 2, 2, 2, 2, 2, 2, 2, 2, 2), (2, 2, 2, 2, 2, 2, 2, 2, 2, 2)),
    ((2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2), (2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2)),
    ((2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2), (2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2)),
    ((4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4), (1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1)),
]


@dataclass
class Kernel:
    """Data class to store information about a numeric kernel."""

    name: str
    first_dtype: str
    second_dtype: str
    baseline_func: callable
    simsimd_func: callable
    tensor_type: callable = np.array


def serial_broadcast_shape(shape1, shape2):
    # Computes the broadcasted shape following NumPy broadcasting rules
    result_shape = []
    len1, len2 = len(shape1), len(shape2)
    for i in range(max(len1, len2)):
        dim1 = shape1[-(i + 1)] if i < len1 else 1
        dim2 = shape2[-(i + 1)] if i < len2 else 1
        if dim1 == dim2 or dim1 == 1 or dim2 == 1:
            result_shape.insert(0, max(dim1, dim2))
        else:
            raise ValueError(f"Shapes {shape1} and {shape2} are not compatible")
    return tuple(result_shape)


def serial_ndindex(shape):
    if not shape:
        yield ()
        return
    for idx in range(shape[0]):
        for rest in serial_ndindex(shape[1:]):
            yield (idx,) + rest


def serial_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    shape_a = a.shape
    shape_b = b.shape
    output_shape = serial_broadcast_shape(shape_a, shape_b)

    # Pad shapes with ones on the left to align them with the output shape
    len_output = len(output_shape)
    len_a = len(shape_a)
    len_b = len(shape_b)
    padded_shape_a = (1,) * (len_output - len_a) + shape_a
    padded_shape_b = (1,) * (len_output - len_b) + shape_b

    # Prepare the output array
    output = np.zeros(output_shape, dtype=np.result_type(a, b))

    # Iterate over all possible indices in the output array
    for idx in serial_ndindex(output_shape):
        idx_a = []
        idx_b = []
        # Map output indices to input indices, considering broadcasting
        for idx_i, dim_a, dim_b in zip(idx, padded_shape_a, padded_shape_b):
            idx_a.append(idx_i if dim_a != 1 else 0)
            idx_b.append(idx_i if dim_b != 1 else 0)
        # Adjust indices to match the dimensions of 'a' and 'b'
        idx_a = tuple(idx_a[-len_a:])
        idx_b = tuple(idx_b[-len_b:])
        # Perform the addition
        val_a = a[idx_a]
        val_b = b[idx_b]
        output[idx] = val_a + val_b

    return output


def serial_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    shape_a = a.shape
    shape_b = b.shape
    output_shape = serial_broadcast_shape(shape_a, shape_b)

    # Pad shapes with ones on the left to align them with the output shape
    len_output = len(output_shape)
    len_a = len(shape_a)
    len_b = len(shape_b)
    padded_shape_a = (1,) * (len_output - len_a) + shape_a
    padded_shape_b = (1,) * (len_output - len_b) + shape_b

    # Prepare the output array
    output = np.zeros(output_shape, dtype=np.result_type(a, b))

    # Iterate over all possible indices in the output array
    for idx in serial_ndindex(output_shape):
        idx_a = []
        idx_b = []
        # Map output indices to input indices, considering broadcasting
        for idx_i, dim_a, dim_b in zip(idx, padded_shape_a, padded_shape_b):
            idx_a.append(idx_i if dim_a != 1 else 0)
            idx_b.append(idx_i if dim_b != 1 else 0)
        # Adjust indices to match the dimensions of 'a' and 'b'
        idx_a = tuple(idx_a[-len_a:])
        idx_b = tuple(idx_b[-len_b:])
        # Perform the addition
        val_a = a[idx_a]
        val_b = b[idx_b]
        output[idx] = val_a * val_b

    return output


def yield_kernels(
    operation_names: List[str],
    dtype_names: List[Tuple[str]],
    include_serial: bool = False,
    include_torch: bool = False,
    include_tf: bool = False,
    include_jax: bool = False,
) -> Generator[Kernel, None, None]:
    """Yield a list of kernels to latency."""

    if include_torch:
        import torch
    if include_tf:
        # Disable TensorFlow warning messages
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # This hides INFO and WARNING messages

        import tensorflow as tf

        # This will show only ERROR messages, not WARNING messages.
        # Additionally, to filter out oneDNN related warnings, you might need to:
        tf.get_logger().setLevel("FATAL")

    if include_jax:
        import jax
        import jax.numpy as jnp

    def raise_(ex):
        """Utility function to allow raising exceptions in lambda functions."""
        raise ex

    def for_dtypes(
        name: str,
        dtypes: List[Tuple[str]],
        baseline_func: callable,
        simsimd_func: callable,
        tensor_type: callable = np.array,
    ) -> list:
        """Filter out unsupported data types."""
        return [
            Kernel(
                name=name,
                baseline_func=baseline_func,
                simsimd_func=simsimd_func,
                tensor_type=tensor_type,
                first_dtype=first_dtype,
                second_dtype=second_dtype,
            )
            for first_dtype, second_dtype in dtypes
        ]

    if "add" in operation_names:
        yield from for_dtypes(
            "numpy.add",
            dtype_names_supported,
            np.add,
            simd.add,
        )
        yield from for_dtypes(
            "numpy.add",
            [("bfloat16", "bfloat16")],
            lambda A, B: raise_(NotImplementedError("Not implemented for bfloat16")),
            lambda A, B: simd.add(A, B, a_dtype="bfloat16", b_dtype="bfloat16", out_dtype="bfloat16"),
        )
    if "add" in operation_names and include_serial:
        yield from for_dtypes(
            "serial.add",
            dtype_names_supported,
            serial_add,
            simd.add,
        )
    if "multiply" in operation_names:
        yield from for_dtypes(
            "numpy.multiply",
            dtype_names_supported,
            np.multiply,
            simd.multiply,
        )
    if "multiply" in operation_names and include_serial:
        yield from for_dtypes(
            "serial.multiply",
            dtype_names_supported,
            serial_multiply,
            simd.multiply,
        )


@dataclass
class Result:
    first_dtype: str
    second_dtype: str
    name: str
    shape_first: tuple
    shape_second: tuple
    baseline_seconds: Union[float, Exception]
    simsimd_seconds: Union[float, Exception]
    bytes_per_input: int
    invocations: int


def random_matrix(shape: tuple, dtype: str) -> np.ndarray:
    if dtype == "bfloat16":
        return np.random.randint(0, high=256, size=shape, dtype=np.int16)
    if np.issubdtype(np.dtype(dtype), np.integer):
        return np.random.randint(0, high=10, size=shape, dtype=dtype)
    else:
        return np.random.rand(*shape).astype(dtype)


def latency(func, A, B, iterations: int = 1, warmup: int = 0) -> float:
    """Time the amount of time it takes to run a function and return the average time per run in seconds."""
    while warmup > 0:
        func(A, B)
        warmup -= 1
    start_time = time.time_ns()
    while iterations > 0:
        func(A, B)
        iterations -= 1
    end_time = time.time_ns()
    return (end_time - start_time) / 1e9


def yield_results(
    first_shape: tuple,
    second_shape: tuple,
    kernels: List[Kernel],
    warmup: int = 0,
    time_limit: float = 1.0,
) -> Generator[Result, None, None]:
    # For each of the present data types, we may want to pre-generate several random matrices
    count_matrices_per_dtype = 8
    count_repetitions_per_matrix = 3  # This helps dampen the effect of time-measurement itself

    # Let's cache the matrices for each data type and shape
    matrices_per_dtype_and_shape: Dict[Tuple[str, tuple]] = {}
    all_dtypes = set(kernel.first_dtype for kernel in kernels) | set(kernel.second_dtype for kernel in kernels)
    for dtype, shape in product(all_dtypes, (first_shape, second_shape)):
        matrix_key = (dtype, shape)
        if dtype in matrices_per_dtype_and_shape:
            continue
        matrices = [random_matrix(shape, dtype) for _ in range(count_matrices_per_dtype)]
        matrices_per_dtype_and_shape[matrix_key] = matrices[]

    # For each kernel, repeat benchmarks for each data type
    for kernel in kernels:
        first_matrix_key = (kernel.first_dtype, first_shape)
        first_matrices_numpy = matrices_per_dtype_and_shape[first_matrix_key]
        first_matrices_converted = [kernel.tensor_type(m) for m in first_matrices_numpy]

        second_matrix_key = (kernel.second_dtype, second_shape)
        second_matrices_numpy = matrices_per_dtype_and_shape[second_matrix_key]
        second_matrices_converted = [kernel.tensor_type(m) for m in second_matrices_numpy]

        baseline_func = kernel.baseline_func
        simsimd_func = kernel.simsimd_func
        result = Result(
            kernel.first_dtype,
            kernel.second_dtype,
            kernel.name,
            baseline_seconds=0,
            simsimd_seconds=0,
            shape_first=first_shape,
            shape_second=second_shape,
            bytes_per_input=max(first_matrices_numpy[0].nbytes, second_matrices_numpy[0].nbytes),
            invocations=count_matrices_per_dtype * count_repetitions_per_matrix,
        )

        # Try obtaining the baseline measurements
        try:
            for i, j in product(range(count_matrices_per_dtype), range(count_matrices_per_dtype)):
                result.baseline_seconds += latency(
                    baseline_func,
                    first_matrices_converted[i],
                    second_matrices_converted[j],
                    count_repetitions_per_matrix,
                    warmup,
                )
                if result.baseline_seconds > time_limit:
                    break
        except NotImplementedError as e:
            result.baseline_seconds = e
        except ValueError as e:
            result.baseline_seconds = e  #! This happens often during overflows
        except RuntimeError as e:
            result.baseline_seconds = e  #! This happens often during overflows
        except Exception as e:
            # This is an unexpected exception... once you face it, please report it
            raise RuntimeError(str(e) + " for %s(%s)" % (kernel.name, str(kernel.dtype))) from e

        # Try obtaining the SimSIMD measurements
        try:
            for i, j in product(range(count_matrices_per_dtype), range(count_matrices_per_dtype)):
                result.simsimd_seconds += latency(
                    simsimd_func,
                    first_matrices_numpy[i],
                    second_matrices_numpy[j],
                    count_repetitions_per_matrix,
                    warmup,
                )
                if result.simsimd_seconds > time_limit:
                    break
        except NotImplementedError as e:
            result.simsimd_seconds = e
        except Exception as e:
            # This is an unexpected exception... once you face it, please report it
            raise RuntimeError(str(e) + " for %s(%s)" % (kernel.name, str(kernel.dtype))) from e

        yield result


def result_to_row(result: Result) -> List[str]:
    dtype_cell = f"`{result.first_dtype}` & `{result.second_dtype}`"
    name_cell = f"`{result.name}`"
    baseline_cell = "💥"
    simsimd_cell = "💥"
    improvement_cell = "🤷"

    if isinstance(result.baseline_seconds, float):
        ops_per_second = result.invocations / result.baseline_seconds
        gbs_per_second = result.bytes_per_input * ops_per_second / 1e9
        baseline_cell = f"{ops_per_second:,.0f} ops/s, {gbs_per_second:,.3f} GB/s"
    if isinstance(result.simsimd_seconds, float):
        ops_per_second = result.invocations / result.simsimd_seconds
        gbs_per_second = result.bytes_per_input * ops_per_second / 1e9
        simsimd_cell = f"{ops_per_second:,.0f} ops/s, {gbs_per_second:,.3f} GB/s"
    if isinstance(result.baseline_seconds, float) and isinstance(result.simsimd_seconds, float):
        improvement_cell = f"{result.baseline_seconds / result.simsimd_seconds:,.2f} x"

    return [dtype_cell, name_cell, baseline_cell, simsimd_cell, improvement_cell]


def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description="Benchmark SimSIMD and other libraries")
    parser.add_argument(
        "--operation",
        choices=["all", *operation_names],
        default="all",
        help="Operation to benchmark, profiles everything by default",
    )
    parser.add_argument(
        "--dtype",
        choices=["all", *dtype_names],
        default="all",
        help="Defines numeric types to latency, profiles everything by default",
    )
    parser.add_argument("--torch", action="store_true", help="Profile PyTorch, must be installed")
    parser.add_argument("--tf", action="store_true", help="Profile TensorFlow, must be installed")
    parser.add_argument("--jax", action="store_true", help="Profile JAX, must be installed")
    parser.add_argument(
        "--time-limit",
        type=float,
        default=1.0,
        help="Maximum time in seconds to run each latency (default: 1.0)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="""
        Number of warm-up runs before timing (default: 0)
        
        This will greatly affect the results for all heavy libraries relying on JIT compilation
        or lazy computational graphs (e.g., TensorFlow, PyTorch, JAX).
        """,
    )
    args = parser.parse_args()
    dtypes_profiled = []
    if args.dtype != "all":
        dtypes_profiled = [
            (first_dtype, second_dtype)
            for first_dtype, second_dtype in dtype_names
            if first_dtype == args.dtype or second_dtype == args.dtype
        ]
    else:
        dtypes_profiled = dtype_names
    operation_names_profiled = set([args.operation] if args.operation != "all" else operation_names)

    print("# Benchmarking SimSIMD")
    print("- Operations:", ", ".join(operation_names_profiled))
    print("- Datatypes:", ", ".join([f"{a} & {b}" for a, b in dtypes_profiled]))
    try:
        caps = [cap for cap, enabled in simd.get_capabilities().items() if enabled]
        print("- Hardware capabilities:", ", ".join(caps))

        # Log versions of SimSIMD, NumPy, SciPy, and scikit-learn
        print(f"- SimSIMD version: {simd.__version__}")
        print(f"- NumPy version: {np.__version__}")

        if args.torch:
            import torch

            print(f"- PyTorch version: {torch.__version__}")
        if args.tf:
            import tensorflow as tf

            print(f"- TensorFlow version: {tf.__version__}")
        if args.jax:
            import jax

            print(f"- JAX version: {jax.__version__}")

        deps: dict = np.show_config(mode="dicts").get("Build Dependencies")
        print("-- NumPy BLAS dependency:", deps["blas"]["name"])
        print("-- NumPy LAPACK dependency:", deps["lapack"]["name"])
    except Exception as e:
        print(f"An error occurred: {e}")

    kernels: List[Kernel] = list(
        yield_kernels(
            operation_names_profiled,
            dtypes_profiled,
            include_torch=args.torch,
            include_tf=args.tf,
            include_jax=args.jax,
        )
    )

    for shape in shapes:
        first_shape, second_shape = shape
        print(f"## Shapes: {first_shape} and {second_shape}")

        results = yield_results(first_shape, second_shape, kernels, warmup=args.warmup, time_limit=args.time_limit)
        columns_headers = ["Data Type", "Method", "Baseline", "SimSIMD", "Improvement"]
        results_rows = []
        for result in results:
            result_row = result_to_row(result)
            results_rows.append(result_row)

        print(tabulate.tabulate(results_rows, headers=columns_headers))

        # TODO:Benchmark with non-continuous inputs
        # For that we will patch the shapes incrementing the extents of the first one by 3,
        # and the extents of the second one by 5, to make sure they are not multiples of
        # the scalar width.
        # first_shape = tuple(x + 3 for x in first_shape)
        # second_shape = tuple(x + 5 for x in second_shape)


if __name__ == "__main__":
    main()

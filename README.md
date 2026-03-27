# NumKong: Mixed Precision for All

NumKong (previously SimSIMD) is a portable mixed-precision math library with over 2000 kernels for x86, Arm, RISC-V, and WASM.
It covers numeric types from 6-bit floats to 64-bit complex numbers, hardened against in-house 118-bit extended-precision baselines.
Built alongside the [USearch](https://github.com/unum-cloud/usearch) vector-search engine, it provides wider accumulators to avoid the overflow and precision loss typical of naive same-type arithmetic.

![NumKong banner](https://github.com/ashvardanian/ashvardanian/blob/master/repositories/NumKong-v7.png?raw=true)

## Quick Start

NumKong is a header-only C library with zero dependencies. To use it in Python:

```bash
pip install numkong
```

In C or C++, simply include the headers from `include/numkong/`. For other languages:
- **Rust:** `cargo add numkong`
- **JavaScript:** `npm install numkong`
- **Go:** `go get github.com/unum-cloud/numkong/go`

## Latency, Throughput, & Numerical Stability

Most libraries return dot products in the __same type as the input__ — Float16 × Float16 → Float16, Int8 × Int8 → Int8.
This leads to quiet overflow: a 2048-dimensional `i8` dot product can reach ±10 million, but `i8` maxes out at 127.
NumKong promotes to wider accumulators — Float16 → Float32, BFloat16 → Float32, Int8 → Int32, Float32 → Float64 — so results stay in range.

> Single 2048-d dot product on Intel [Sapphire Rapids](https://en.wikipedia.org/wiki/Sapphire_Rapids), single-threaded.
> Each cell shows __gso/s, mean relative error__ vs higher-precision reference.
> gso/s = Giga Scalar Operations per Second — a more suitable name than GFLOP/s when counting both integer and floating-point work.
> NumPy 2.4, PyTorch 2.10, JAX 0.9.

| Input  |        NumPy + OpenBLAS |           PyTorch + MKL |                     JAX |               NumKong |
| :----- | ----------------------: | ----------------------: | ----------------------: | --------------------: |
|        |          ░░░░░░░░░░░░░░ |          ░░░░░░░░░░░░░░ |          ░░░░░░░░░░░░░░ |        ░░░░░░░░░░░░░░ |
| `f64`  |    2.0 gso/s, 1e-15 err |    0.6 gso/s, 1e-15 err |    0.4 gso/s, 1e-14 err |  5.8 gso/s, 1e-16 err |
| `f32`  |     1.5 gso/s, 2e-6 err |     0.6 gso/s, 2e-6 err |     0.4 gso/s, 5e-6 err |   7.1 gso/s, 2e-7 err |
| `bf16` |                       — |     0.5 gso/s, 1.9% err |     0.5 gso/s, 1.9% err |   9.7 gso/s, 1.8% err |
| `f16`  |    0.2 gso/s, 0.25% err |    0.5 gso/s, 0.25% err |    0.4 gso/s, 0.25% err | 11.5 gso/s, 0.24% err |
| `e5m2` |                       — |     0.7 gso/s, 4.6% err |     0.5 gso/s, 4.6% err |     7.1 gso/s, 0% err |
| `i8`   | 1.1 gso/s, __overflow__ | 0.5 gso/s, __overflow__ | 0.5 gso/s, __overflow__ |    14.8 gso/s, 0% err |

A fair objection: PyTorch and JAX are designed for throughput, not single-call latency.
They lower execution graphs through [XLA](https://openxla.org/) or vendored BLAS libraries like [Intel MKL](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html) and Nvidia [cuBLAS](https://developer.nvidia.com/cublas).
So here's the same comparison on a throughput-oriented workload — matrix multiplication:

> Matrix multiplication (2048 × 2048) × (2048 × 2048) on Intel Sapphire Rapids, single-threaded.
> gso/s = Giga Scalar Operations per Second, same format.
>
# Contributing

To keep the quality of the code high, we have a set of [guidelines](https://github.com/unum-cloud).

- [What's the procedure?](https://github.com/unum-cloud/awesome/blob/main/Workflow.md#organizing-software-development)
- [How to organize branches?](https://github.com/unum-cloud/awesome/blob/main/Workflow.md#branches)
- [How to style commits?](https://github.com/unum-cloud/awesome/blob/main/Workflow.md#commits)

## C and C++

To rerun experiments utilize the following command:

```sh
sudo apt install libopenblas-dev # BLAS installation is optional, but recommended for benchmarks
cmake -D CMAKE_BUILD_TYPE=Release -D SIMSIMD_BUILD_TESTS=1 -D SIMSIMD_BUILD_BENCHMARKS=1 -D SIMSIMD_BUILD_BENCHMARKS_WITH_CBLAS=1 -B build_release
cmake --build build_release --config Release
build_release/simsimd_bench
build_release/simsimd_bench --benchmark_filter=js
build_release/simsimd_test_run_time
build_release/simsimd_test_compile_time # no need to run this one, it's just a compile-time test
```

To utilize `f16` instructions, use GCC 12 or newer, or Clang 16 or newer.
To install them on Ubuntu 22.04, use:

```sh
sudo apt install gcc-12 g++-12
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 100
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 100
```

On MacOS it's recommended to use Homebrew and install Clang, as opposed to "Apple Clang".
Replacing the default compiler is not recommended, as it may break the system, but you can pass it as an environment variable:

```sh
brew install llvm
cmake -D CMAKE_BUILD_TYPE=Release -D SIMSIMD_BUILD_TESTS=1 \
    -D CMAKE_C_COMPILER="$(brew --prefix llvm)/bin/clang" \
    -D CMAKE_CXX_COMPILER="$(brew --prefix llvm)/bin/clang++" \
    -B build_release
cmake --build build_release --config Release
```

## Python

Testing:

```sh
pip install -e .                             # to install the package in editable mode
pip install pytest pytest-repeat tabulate    # testing dependencies
pytest scripts/test.py -s -x -Wd             # to run tests

# to check supported SIMD instructions:
python -c "import simsimd; print(simsimd.get_capabilities())" 
```

Here, `-s` will output the logs.
The `-x` will stop on the first failure.
The `-Wd` will silence overflows and runtime warnings.

When building on MacOS, same as with C/C++, use non-Apple Clang version:

```sh
brew install llvm
CC=$(brew --prefix llvm)/bin/clang CXX=$(brew --prefix llvm)/bin/clang++ pip install -e .
```

Benchmarking:

```sh
pip install numpy scipy scikit-learn                 # for comparison baselines
python scripts/bench_vectors.py                      # to run default benchmarks
python scripts/bench_vectors.py --n 1000 --ndim 1536 # batch size and dimensions
```

You can also benchmark against other libraries, filter the numeric types, and distance metrics:

```sh
$ python scripts/bench_vectors.py --help
> usage: bench.py [-h] [--ndim NDIM] [-n COUNT]
>                 [--metric {all,dot,spatial,binary,probability,sparse}]
>                 [--dtype {all,bits,int8,uint16,uint32,float16,float32,float64,bfloat16,complex32,complex64,complex128}] 
>                 [--scipy] [--scikit] [--torch] [--tf] [--jax]
> 
> Benchmark SimSIMD vs. other libraries
> 
> optional arguments:
>   -h, --help            show this help message and exit
>   --ndim NDIM           Number of dimensions in vectors (default: 1536) For binary vectors (e.g., Hamming, Jaccard), this is the number of bits. In
>                         case of SimSIMD, the inputs will be treated at the bit-level. Other packages will be matching/comparing 8-bit integers. The
>                         volume of exchanged data will be identical, but the results will differ.
>   -n COUNT, --count COUNT
>                         Number of vectors per batch (default: 1) By default, when set to 1 the benchmark will generate many vectors of size (ndim, )
>                         and call the functions on pairs of single vectors: both directly, and through `cdist`. Alternatively, for larger batch sizes
>                         the benchmark will generate two matrices of size (n, ndim) and compute: - batch mode: (n) distances between vectors in
>                         identical rows of the two matrices, - all-pairs mode: (n^2) distances between all pairs of vectors in the two matrices via
>                         `cdist`.
>   --metric {all,dot,spatial,binary,probability,sparse}
>                         Distance metric to use, profiles everything by default
>   --dtype {all,bits,int8,uint16,uint32,float16,float32,float64,bfloat16,complex32,complex64,complex128}
>                         Defines numeric types to benchmark, profiles everything by default
>   --scipy               Profile SciPy, must be installed
>   --scikit              Profile scikit-learn, must be installed
>   --torch               Profile PyTorch, must be installed
>   --tf                  Profile TensorFlow, must be installed
>   --jax                 Profile JAX, must be installed
```


Before merging your changes you may want to test your changes against the entire matrix of Python versions USearch supports.
For that you need the `cibuildwheel`, which is tricky to use on MacOS and Windows, as it would target just the local environment.
Still, if you have Docker running on any desktop OS, you can use it to build and test the Python bindings for all Python versions for Linux:

```sh
pip install cibuildwheel
cibuildwheel
cibuildwheel --platform linux                   # works on any OS and builds all Linux backends
cibuildwheel --platform linux --archs x86_64    # 64-bit x86, the most common on desktop and servers
cibuildwheel --platform linux --archs aarch64   # 64-bit Arm for mobile devices, Apple M-series, and AWS Graviton
cibuildwheel --platform linux --archs i686      # 32-bit Linux
cibuildwheel --platform linux --archs s390x     # emulating big-endian IBM Z
cibuildwheel --platform macos                   # works only on MacOS
cibuildwheel --platform windows                 # works only on Windows
```

You may need root privileges for multi-architecture builds:

```sh
sudo $(which cibuildwheel) --platform linux
```

On Windows and MacOS, to avoid frequent path resolution issues, you may want to use:

```sh
python -m cibuildwheel --platform windows
```

## Rust

```sh
cargo test -p simsimd
cargo test -p simsimd -- --nocapture # To see the output
cargo bench
open target/criterion/report/index.html
```

## JavaScript

If you don't have NPM installed:

```sh
wget -qO- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
nvm install 20
```

Testing and benchmarking:

```sh
npm install -g typescript
npm run build-js
npm test
npm run bench
```

Running with Deno:

```sh
deno test --allow-read
```

Running with Bun:

```sh
npm install -g bun
bun test
```

## GoLang

```sh
cd golang
go test # To test
go test -run=^$ -bench=. -benchmem # To benchmark
```


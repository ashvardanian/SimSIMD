# Contributing

## C and C++

To rerun experiments utilize the following command:

```sh
sudo apt install libopenblas-dev # BLAS installation is optional, but recommended for benchmarks
cmake -D CMAKE_BUILD_TYPE=Release \
    -D SIMSIMD_BUILD_TESTS=1 -D SIMSIMD_BUILD_BENCHMARKS=1 \
    -D SIMSIMD_BUILD_BENCHMARKS_WITH_CBLAS=1 -D SIMSIMD_BUILD_WITH_OPENMP=1 \
    -B build_release
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

Similarly, using Clang on Linux:

```sh
sudo apt install clang
cmake -D CMAKE_BUILD_TYPE=Release \
    -D SIMSIMD_BUILD_TESTS=1 -D SIMSIMD_BUILD_BENCHMARKS=1 \
    -D SIMSIMD_BUILD_BENCHMARKS_WITH_CBLAS=1 -D SIMSIMD_BUILD_WITH_OPENMP=0 \
    -D CMAKE_C_COMPILER=clang -D CMAKE_CXX_COMPILER=clang++ \
    -B build_release
cmake --build build_release --config Release
```

I'd recommend putting the following breakpoints:

- `__asan::ReportGenericError` - to detect illegal memory accesses.
- `__GI_exit` - to stop at exit points - the end of running any executable.
- `__builtin_unreachable` - to catch all the places where the code is expected to be unreachable.

## Python

Testing:

```sh
pip install -e .                             # to install the package in editable mode
pip install pytest pytest-repeat tabulate    # testing dependencies
pytest python/test.py -s -x -Wd                 # to run tests

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
pip install numpy scipy scikit-learn        # for comparison baselines
python python/bench.py                      # to run default benchmarks
python python/bench.py --n 1000 --ndim 1536 # batch size and dimensions
```

You can also benchmark against other libraries:

```sh
$ python python/bench.py --help
> usage: bench.py [-h] [--n N] [--ndim NDIM] [--scipy] [--scikit] [--torch] [--tf] [--jax]
> 
> Benchmark SimSIMD vs. other libraries
> 
> options:
>   -h, --help   show this help message and exit
>   --n N        Number of vectors (default: 1000)
>   --ndim NDIM  Number of dimensions (default: 1536)
>   --scipy      Profile SciPy
>   --scikit     Profile scikit-learn
>   --torch      Profile PyTorch
>   --tf         Profile TensorFlow
>   --jax        Profile JAX
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


# Contributing

## C and C++

To rerun experiments utilize the following command:

```sh
sudo apt install libopenblas-dev # BLAS installation is optional
cmake -DCMAKE_BUILD_TYPE=Release -DSIMSIMD_BUILD_BENCHMARKS=1 -DSIMSIMD_BUILD_TESTS=1 -B ./build_release
cmake --build build_release --config Release
build_release/simsimd_bench
build_release/simsimd_bench --benchmark_filter=js
build_release/simsimd_test_run_time
build_release/simsimd_test_compile_time
```

## Python

Testing:

```sh
pip install -e .                    # to install the package in editable mode
pip install pytest pytest-repeat    # testing dependencies
pytest python/test.py -s -x -Wd     # to run tests
```

Here, `-s` will output the logs.
The `-x` will stop on the first failure.
The `-Wd` will silence overflows and runtime warnings.

Benchmarking:

```sh
pip install numpy scipy scikit-learn    # for comparison baselines
python python/bench.py                  # to run default benchmarks
python python/bench.py --n 1000 --ndim 1000000 # batch size and dimensions
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

You may need root previligies for multi-architecture builds:

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
open ./target/criterion/report/index.html
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


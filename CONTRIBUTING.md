# Contributing

To keep the quality of the code high, we follow the [coding style and conventions](https://github.com/ashvardanian/ashvardanian/blob/master/CONTRIBUTING.md) shared across multiple projects — covering Git history, C++ and Python formatting, dependency management, and documentation.

## Directory Tree

```
include/numkong/          C and C++ headers — one .h per kernel family, one .hpp per C++ API
include/numkong/*/        Per-ISA kernel implementations — serial, haswell, neon, rvv, sme, etc.
c/                        Runtime dispatch layer — one dispatch_*.c per dtype
test/                     C++ precision tests — see test/README.md
bench/                    C++ Google Benchmark suite and JS bench runner — see bench/README.md
python/                   CPython extension, no SWIG or PyBind11
javascript/               Node.js native addon + Emscripten WASM + TypeScript API
rust/                     Rust FFI bindings
swift/                    Swift Package Manager bindings
golang/                   Go cgo bindings
cmake/                    Toolchain files for cross-compilation — WASM, WASI, RISC-V, AArch64
```

## C and C++

### Building

```sh
cmake -B build_release -D CMAKE_BUILD_TYPE=Release \
      -D NK_BUILD_TEST=1 \
      -D NK_BUILD_BENCH=1 \
      -D NK_COMPARE_TO_BLAS=1
cmake --build build_release --config Release --parallel
build_release/nk_bench
build_release/nk_test
```

| CMake Flag             | Default            | Description                                     |
| ---------------------- | ------------------ | ----------------------------------------------- |
| `NK_BUILD_TEST`        | `OFF`              | Compile precision tests with ULP error analysis |
| `NK_BUILD_BENCH`       | `OFF`              | Compile micro-benchmarks                        |
| `NK_BUILD_SHARED`      | `ON`, if top-level | Compile dynamic library                         |
| `NK_BUILD_SHARED_TEST` | `OFF`              | Compile tests against the shared library        |
| `NK_COMPARE_TO_BLAS`   | `AUTO`             | Include OpenBLAS or Apple Accelerate            |
| `NK_COMPARE_TO_MKL`    | `AUTO`             | Include Intel MKL                               |

### Compiler Requirements

| ISA Family                              | GCC | Clang | AppleClang     | MSVC        |
| --------------------------------------- | --- | ----- | -------------- | ----------- |
| Base — serial, NEON, AVX2               | 9+  | 10+   | Any            | 2019+       |
| Float16 — NEONHalf, Sapphire FP16, Zvfh | 12+ | 16+   | Any            | 2022 17.14+ |
| AVX-512 — Skylake, Ice Lake             | 9+  | 10+   | N/A            | 2019+       |
| AVX-512BF16 — Genoa                     | 12+ | 16+   | N/A            | 2022 17.14+ |
| Intel AMX — Sapphire, Granite           | 14+ | 18+   | N/A            | 2022 17.14+ |
| Arm SME/SME2                            | 14+ | 18+   | 16+ / Xcode 16 | N/A         |
| RISC-V Vector — RVV 1.0                 | 13+ | 17+   | N/A            | N/A         |
| RVV + Zvfh/Zvfbfwma/Zvbb                | 14+ | 18+   | N/A            | N/A         |

To install on Ubuntu 22.04:

```sh
sudo apt install gcc-12 g++-12
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 100
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 100
```

### Cross-Compilation

NumKong ships 8 toolchain files in `cmake/` for cross-compiling to non-native targets.
Tests and benchmarks run transparently under QEMU via `CMAKE_CROSSCOMPILING_EMULATOR`.

| Target                    | Toolchain File                  | Emulator                | Prerequisites                                                                                    |
| ------------------------- | ------------------------------- | ----------------------- | ------------------------------------------------------------------------------------------------ |
| ARM64 Linux               | `toolchain-aarch64-gnu.cmake`   | `qemu-aarch64 -cpu max` | `gcc-aarch64-linux-gnu`, `qemu-user`                                                             |
| RISC-V 64 GCC             | `toolchain-riscv64-gnu.cmake`   | `qemu-riscv64 -cpu max` | [riscv-gnu-toolchain](https://github.com/riscv-collab/riscv-gnu-toolchain/releases), `qemu-user` |
| RISC-V 64 LLVM            | `toolchain-riscv64-llvm.cmake`  | `qemu-riscv64 -cpu max` | LLVM 17+, `RISCV_SYSROOT`                                                                        |
| Android ARM64             | `toolchain-android-arm64.cmake` | —                       | `ANDROID_NDK_ROOT`                                                                               |
| x86_64 from Apple Silicon | `toolchain-x86_64-llvm.cmake`   | `arch -x86_64`          | Homebrew LLVM                                                                                    |
| WASM Emscripten           | `toolchain-wasm.cmake`          | Node.js                 | Emscripten 3.1.27+                                                                               |
| WASM64 Memory64           | `toolchain-wasm64.cmake`        | Node.js                 | Emscripten 3.1.35+                                                                               |
| WASI                      | `toolchain-wasi.cmake`          | Wasmtime / Wasmer       | WASI SDK 24+                                                                                     |

Set `NK_IN_QEMU=1` to relax half-precision accuracy thresholds under emulation.

__ARM64 Linux__

```sh
cmake -B build_arm64 -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-aarch64-gnu.cmake
cmake --build build_arm64 --parallel
```

To build and run tests under emulation, see [test/README.md](test/README.md#cross-compilation).

Default arch: `armv9-a+sve2+fp16+bf16+i8mm+dotprod+fp16fml`.

__RISC-V 64 with GCC__

```sh
cmake -B build_riscv -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-riscv64-gnu.cmake
cmake --build build_riscv --parallel
```

To build and run tests under emulation, see [test/README.md](test/README.md#cross-compilation).

Default arch: `rv64gcv_zvfh_zvfbfwma_zvbb`.

__RISC-V 64 with LLVM__

```sh
export RISCV_SYSROOT=/path/to/riscv-sysroot
cmake -B build_riscv_llvm -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-riscv64-llvm.cmake
cmake --build build_riscv_llvm --parallel
```

To build and run tests under emulation, see [test/README.md](test/README.md#cross-compilation).

__Android ARM64__

```sh
cmake -B build_android -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-android-arm64.cmake
cmake --build build_android --parallel
```

To build and run tests under emulation, see [test/README.md](test/README.md#cross-compilation).

__WASM via Emscripten__

```sh
source ~/emsdk/emsdk_env.sh
cmake -B build-wasm -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-wasm.cmake
cmake --build build-wasm --parallel
```

For wasm64 — Memory64:

```sh
cmake -B build-wasm64 -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-wasm64.cmake
cmake --build build-wasm64 --parallel
```

__WASI__

```sh
export WASI_SDK_PATH=~/wasi-sdk-24.0-x86_64-linux
cmake -B build-wasi -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-wasi.cmake
cmake --build build-wasi --parallel
```

__iOS Simulator via Xcode__

```sh
xcodebuild test -scheme NumKong -destination 'platform=iOS Simulator,name=iPhone 16'
```

__x86_64 from Apple Silicon__

```sh
cmake -B build_x86 -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-x86_64-llvm.cmake
cmake --build build_x86 --parallel
```

### macOS

With Apple Clang and Homebrew OpenBLAS:

```sh
brew install openblas
cmake -B build_release -D CMAKE_BUILD_TYPE=Release \
      -D NK_BUILD_TEST=1 \
      -D NK_BUILD_BENCH=1 \
      -D NK_COMPARE_TO_BLAS=1 \
      -D CMAKE_PREFIX_PATH="$(brew --prefix openblas)" \
      -D CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES="$(brew --prefix openblas)/include"
cmake --build build_release --config Release --parallel
```

With Homebrew Clang — recommended for full ISA support:

```sh
brew install llvm openblas
unset DEVELOPER_DIR
cmake -B build_release -D CMAKE_BUILD_TYPE=Release \
      -D NK_BUILD_TEST=1 \
      -D NK_BUILD_BENCH=1 \
      -D NK_COMPARE_TO_BLAS=1 \
      -D CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES="$(brew --prefix openblas)/include" \
      -D CMAKE_C_LINK_FLAGS="-L$(xcrun --sdk macosx --show-sdk-path)/usr/lib" \
      -D CMAKE_EXE_LINKER_FLAGS="-L$(xcrun --sdk macosx --show-sdk-path)/usr/lib" \
      -D CMAKE_C_COMPILER="$(brew --prefix llvm)/bin/clang" \
      -D CMAKE_CXX_COMPILER="$(brew --prefix llvm)/bin/clang++" \
      -D CMAKE_OSX_SYSROOT="$(xcrun --sdk macosx --show-sdk-path)" \
      -D CMAKE_OSX_DEPLOYMENT_TARGET=$(sw_vers -productVersion)
cmake --build build_release --config Release --parallel
```

### BLAS Threading

When benchmarking with BLAS cross-validation, disable multi-threading in BLAS libraries to avoid interference — see [bench/README.md](bench/README.md#environment-variables) for the `*_NUM_THREADS` variables.

### Debugger Breakpoints

Useful breakpoints for debugging:

- `__asan::ReportGenericError` — illegal memory accesses.
- `__GI_exit` — exit points at end of any executable.
- `__builtin_unreachable` — unexpected code paths.
- `_sz_assert_failure` — StringZilla logic assertions.

See [test/README.md](test/README.md) for test framework details and [bench/README.md](bench/README.md) for benchmark configuration.

## Python

Python bindings are implemented using pure CPython, so you wouldn't need to install SWIG, PyBind11, or any other third-party library.
Still, you need a virtual environment.
If you already have one:

```sh
pip install -e .                             # build locally from source
pip install pytest pytest-repeat tabulate    # testing dependencies
pytest test/ -s -x -Wd                       # to run tests

# to check supported SIMD instructions:
python -c "import numkong; print(numkong.get_capabilities())"
```

Alternatively, use `uv` to create the virtual environment.

```sh
uv venv --python 3.13t          # or your preferred version
source .venv/bin/activate       # activate the environment
uv pip install -e .             # build locally from source

# to run GIL-related tests in a free-threaded environment:
uv pip install pytest pytest-repeat tabulate numpy scipy
PYTHON_GIL=0 python -m pytest test/ -s -x -Wd -k gil
```

Here, `-s` will output the logs.
The `-x` will stop on the first failure.
The `-Wd` will silence overflows and runtime warnings.

When building on macOS, same as with C/C++, use non-Apple Clang version:

```sh
brew install llvm
CC=$(brew --prefix llvm)/bin/clang CXX=$(brew --prefix llvm)/bin/clang++ pip install -e .
```

Before merging your changes you may want to test your changes against the entire matrix of Python versions NumKong supports.
For that you need the `cibuildwheel`, which is tricky to use on macOS and Windows, as it would target just the local environment.
Still, if you have Docker running on any desktop OS, you can use it to build and test the Python bindings for all Python versions for Linux:

```sh
pip install cibuildwheel
cibuildwheel
cibuildwheel --platform linux                   # works on any OS and builds all Linux backends
cibuildwheel --platform linux --archs x86_64    # 64-bit x86, the most common on desktop and servers
cibuildwheel --platform linux --archs aarch64   # 64-bit Arm for mobile devices, Apple M-series, and AWS Graviton
cibuildwheel --platform linux --archs i686      # 32-bit Linux
cibuildwheel --platform macos                   # works only on macOS
cibuildwheel --platform windows                 # works only on Windows
```

You may need root privileges for multi-architecture builds:

```sh
sudo $(which cibuildwheel) --platform linux
```

On Windows and macOS, to avoid frequent path resolution issues, you may want to use:

```sh
python -m cibuildwheel --platform windows
```

## Rust

```sh
cargo test -p numkong
cargo test -p numkong -- --nocapture # To see the output
```

To automatically detect the Minimum Supported Rust Version — MSRV:

```sh
cargo +stable install cargo-msrv
cargo msrv find --ignore-lockfile
```

Please avoid the temptation of using macros in this Rust code.

## JavaScript

See [javascript/README.md](javascript/README.md) for JavaScript/TypeScript development, WASM support, and API documentation.

Quick reference:

```sh
npm run build-js        # Build TypeScript
npm test                # Run tests
npm run bench           # Run benchmarks
```

## Swift

```sh
swift build && swift test -v
```

Running Swift on Linux requires a couple of extra steps, as the Swift compiler is not available in the default repositories.
Please get the most recent Swift tarball from the [official website](https://www.swift.org/install/).
At the time of writing, for 64-bit Arm CPU running Ubuntu 22.04, the following commands would work:

```bash
wget https://download.swift.org/swift-5.9.2-release/ubuntu2204-aarch64/swift-5.9.2-RELEASE/swift-5.9.2-RELEASE-ubuntu22.04-aarch64.tar.gz
tar xzf swift-5.9.2-RELEASE-ubuntu22.04-aarch64.tar.gz
sudo mv swift-5.9.2-RELEASE-ubuntu22.04-aarch64 /usr/share/swift
echo "export PATH=/usr/share/swift/usr/bin:$PATH" >> ~/.bashrc
source ~/.bashrc
```

You can check the available images on [`swift.org/download` page](https://www.swift.org/download/#releases).
For x86 CPUs, the following commands would work:

```bash
wget https://download.swift.org/swift-5.9.2-release/ubuntu2204/swift-5.9.2-RELEASE/swift-5.9.2-RELEASE-ubuntu22.04.tar.gz
tar xzf swift-5.9.2-RELEASE-ubuntu22.04.tar.gz
sudo mv swift-5.9.2-RELEASE-ubuntu22.04 /usr/share/swift
echo "export PATH=/usr/share/swift/usr/bin:$PATH" >> ~/.bashrc
source ~/.bashrc
```

Alternatively, on Linux, the official Swift Docker image can be used for builds and tests:

```bash
sudo docker run --rm -v "$PWD:/workspace" -w /workspace swift:5.9 /bin/bash -cl "swift build -c release --static-swift-stdlib && swift test -c release --enable-test-discovery"
```

## GoLang

```sh
cd golang
go test # To test
go test -run=^$ -bench=. -benchmem # To benchmark
```

## Adding a New Kernel Family

To add a new operation family, for example `foo`:

1. __C header__: create `include/numkong/foo.h` with serial implementation and dispatch function signatures.
2. __ISA implementations__: add `include/numkong/foo/serial.h`, `foo/neon.h`, `foo/haswell.h`, etc.
3. __Dispatch layer__: add entries to the appropriate `c/dispatch_*.c` files for each dtype the kernel supports.
4. __C++ wrapper__: create `include/numkong/foo.hpp` with the typed C++ API.
5. __Test__: create `test/test_foo.cpp` with precision validation against `f118_t` references.
6. __Benchmark__: create `bench/bench_foo.cpp` with Google Benchmark harness.
7. __Cross-platform tests__: add entries to `test/test_cross.hpp` and the relevant `test_cross_*.cpp` files.
8. __CMakeLists.txt__: wire the new source files into the `nk_test` and `nk_bench` targets.
9. __Language bindings__: update `python/numkong.c`, `javascript/numkong.c`, `rust/numkong.rs`, etc. as needed.

## Adding a Backend Kernel to an Existing Family

For primary kernels, every backend implementation should be wired in five places beyond the backend header itself:

1. __Forward declaration__: add the `NK_PUBLIC` declaration with the matching `@copydoc` in the first half of `include/numkong/<family>.h`.
2. __Compile-time dispatch__: add the `#if !NK_DYNAMIC_DISPATCH` branch in the second half of `include/numkong/<family>.h`.
3. __Run-time dispatch__: add the dtype-specific entry to the relevant `c/dispatch_*.c` table.
4. __Precision tests__: register the kernel in `nk_test`, usually in the existing `test/test_<family>.cpp` suite.
5. __Benchmarks__: register the kernel in `nk_bench`, usually in the existing `bench/bench_<family>.cpp` suite.

Use the existing family suite unless the kernel introduces a genuinely new test shape.
The rule is about coverage and reachability, not about creating a brand new source file for every symbol.

There are two intentional exceptions:

- `cast`: the family-level `nk_cast_*` kernels follow the same header/dispatch/test/bench rule, but scalar conversion helpers are wired through `c/dispatch_other.c` and are covered through `test/test_cast.cpp` and `bench/bench_cast.cpp`.
- `scalar`: scalar helpers are centrally declared in `include/numkong/scalar.h`, wired through `c/dispatch_other.c`, and currently do not follow the per-helper `nk_test` and `nk_bench` registration pattern.

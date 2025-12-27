# NumKong for C

## Installation

For integration within a CMake-based project, add the following segment to your `CMakeLists.txt`:

```cmake
FetchContent_Declare(
    numkong
    GIT_REPOSITORY https://github.com/ashvardanian/numkong.git
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(numkong)
```

After that, you can use the NumKong library in your C code in several ways.
Simplest of all, you can include the headers, and the compiler will automatically select the most recent CPU extensions that NumKong will use.

```c
#include <numkong/numkong.h>

int main() {
    nk_capability_t caps = nk_capabilities();
    nk_flush_denormals(caps); // Avoid denormal penalties, enable AMX if available

    nk_f32_t vector_a[1536];
    nk_f32_t vector_b[1536];
    nk_kernel_punned_t metric_punned = nk_metric_punned(
        nk_metric_angular_k, // Metric kind, like the angular distance
        nk_datatype_f32_k,   // Data type, like: f16, f32, f64, i8, b8, and complex variants
        nk_cap_any_k);       // Which CPU capabilities are we allowed to use
    nk_distance_t distance;
    nk_metric_dense_punned_t metric = (nk_metric_dense_punned_t)metric_punned;
    metric(vector_a, vector_b, 1536, &distance);
    return 0;
}
```

## Dynamic Dispatch in C

To avoid hard-coding the backend, you can rely on `c/lib.c` to prepackage all possible backends in one binary, and select the most recent CPU features at runtime.
That feature of the C library is called [dynamic dispatch](#dynamic-dispatch) and is extensively used in the Python, JavaScript, and Rust bindings.
To test which CPU features are available on the machine at runtime, use the following APIs:

```c
int uses_dynamic_dispatch = nk_uses_dynamic_dispatch(); // Check if dynamic dispatch was enabled
nk_capability_t capabilities = nk_capabilities();  // Returns a bitmask

int uses_neon = nk_uses_neon();
int uses_sve = nk_uses_sve();
int uses_haswell = nk_uses_haswell();
int uses_skylake = nk_uses_skylake();
int uses_ice = nk_uses_ice();
int uses_genoa = nk_uses_genoa();
int uses_sapphire = nk_uses_sapphire();
```

To override compilation settings and switch between runtime and compile-time dispatch, define the following macro:

```c
#define NK_DYNAMIC_DISPATCH 1 // or 0
```

## Spatial Distances: Angular and Euclidean Distances

```c
#include <numkong/numkong.h>

int main() {
    nk_i8_t i8s[1536];
    nk_u8_t u8s[1536];
    nk_f64_t f64s[1536];
    nk_f32_t f32s[1536];
    nk_f16_t f16s[1536];
    nk_bf16_t bf16s[1536];
    nk_distance_t distance;

    // Angular distance between two vectors
    nk_angular_i8(i8s, i8s, 1536, &distance);
    nk_angular_u8(u8s, u8s, 1536, &distance);
    nk_angular_f16(f16s, f16s, 1536, &distance);
    nk_angular_f32(f32s, f32s, 1536, &distance);
    nk_angular_f64(f64s, f64s, 1536, &distance);
    nk_angular_bf16(bf16s, bf16s, 1536, &distance);

    // Euclidean distance between two vectors
    nk_l2sq_i8(i8s, i8s, 1536, &distance);
    nk_l2sq_u8(u8s, u8s, 1536, &distance);
    nk_l2sq_f16(f16s, f16s, 1536, &distance);
    nk_l2sq_f32(f32s, f32s, 1536, &distance);
    nk_l2sq_f64(f64s, f64s, 1536, &distance);
    nk_l2sq_bf16(bf16s, bf16s, 1536, &distance);

    return 0;
}
```

## Dot-Products: Inner and Complex Inner Products

```c
#include <numkong/numkong.h>

int main() {
    // NumKong provides "sized" type-aliases without relying on `stdint.h`
    nk_i8_t i8[1536];
    nk_i8_t u8[1536];
    nk_f16_t f16s[1536];
    nk_f32_t f32s[1536];
    nk_f64_t f64s[1536];
    nk_bf16_t bf16s[1536];
    nk_distance_t product;

    // Inner product between two real vectors
    nk_dot_i8(i8s, i8s, 1536, &product);
    nk_dot_u8(u8s, u8s, 1536, &product);
    nk_dot_f16(f16s, f16s, 1536, &product);
    nk_dot_f32(f32s, f32s, 1536, &product);
    nk_dot_f64(f64s, f64s, 1536, &product);
    nk_dot_bf16(bf16s, bf16s, 1536, &product);

    // NumKong provides complex types with `real` and `imag` fields
    nk_f64c_t f64cs[768];
    nk_f32c_t f32cs[768];
    nk_f16c_t f16cs[768];
    nk_bf16c_t bf16cs[768];
    nk_distance_t products[2]; // real and imaginary parts

    // Complex inner product between two vectors
    nk_dot_f16c(f16cs, f16cs, 768, &products[0]);
    nk_dot_f32c(f32cs, f32cs, 768, &products[0]);
    nk_dot_f64c(f64cs, f64cs, 768, &products[0]);
    nk_dot_bf16c(bf16cs, bf16cs, 768, &products[0]);

    // Complex conjugate inner product between two vectors
    nk_vdot_f16c(f16cs, f16cs, 768, &products[0]);
    nk_vdot_f32c(f32cs, f32cs, 768, &products[0]);
    nk_vdot_f64c(f64cs, f64cs, 768, &products[0]);
    nk_vdot_bf16c(bf16cs, bf16cs, 768, &products[0]);
    return 0;
}
```

## Binary Distances: Hamming and Jaccard Distances

```c
#include <numkong/numkong.h>

int main() {
    nk_b8_t b8s[1536 / 8]; // 8 bits per word
    nk_distance_t distance;
    nk_hamming_b8(b8s, b8s, 1536 / 8, &distance);
    nk_jaccard_b8(b8s, b8s, 1536 / 8, &distance);
    return 0;
}
```

## Probability Distributions: Jensen-Shannon and Kullback-Leibler Divergences

```c
#include <numkong/numkong.h>

int main() {
    nk_f64_t f64s[1536];
    nk_f32_t f32s[1536];
    nk_f16_t f16s[1536];
    nk_distance_t divergence;

    // Jensen-Shannon divergence between two vectors
    nk_jsd_f16(f16s, f16s, 1536, &divergence);
    nk_jsd_f32(f32s, f32s, 1536, &divergence);
    nk_jsd_f64(f64s, f64s, 1536, &divergence);

    // Kullback-Leibler divergence between two vectors
    nk_kld_f16(f16s, f16s, 1536, &divergence);
    nk_kld_f32(f32s, f32s, 1536, &divergence);
    nk_kld_f64(f64s, f64s, 1536, &divergence);
    return 0;
}
```

## Half-Precision Floating-Point Numbers

If you aim to utilize the `_Float16` functionality with NumKong, ensure your development environment is compatible with C 11.
For other NumKong functionalities, C 99 compatibility will suffice.
To explicitly disable half-precision support, define the following macro before imports:

```c
#define NK_NATIVE_F16 0 // or 1
#define NK_NATIVE_BF16 0 // or 1
#include <numkong/numkong.h>
```

## Compilation Settings and Debugging

`NK_DYNAMIC_DISPATCH`.

> By default, NumKong is a header-only library.
> But if you are running on different generations of devices, it makes sense to pre-compile the library for all supported generations at once, and dispatch at runtime.
> This flag does just that and is used to produce the `numkong.so` shared library, as well as the Python and other bindings.

For Arm: `NK_TARGET_NEON`, `NK_TARGET_SVE`, `NK_TARGET_SVE2`, `NK_TARGET_NEON_F16`, `NK_TARGET_SVE_F16`, `NK_TARGET_NEON_BF16`, `NK_TARGET_SVE_BF16`.
For x86: `NK_TARGET_HASWELL`, `NK_TARGET_SKYLAKE`, `NK_TARGET_ICE`, `NK_TARGET_GENOA`, `NK_TARGET_SAPPHIRE`, `NK_TARGET_TURIN`, `NK_TARGET_SIERRA`.

> By default, NumKong automatically infers the target architecture and pre-compiles as many kernels as possible.
> In some cases, you may want to explicitly disable some of the kernels.
> Most often it's due to compiler support issues, like the lack of some recent intrinsics or low-precision numeric types.
> In other cases, you may want to disable some kernels to speed up the compilation process and trim the binary size.

For single-precision math operations: `NK_F32_SQRT`, `NK_F32_RSQRT`, `NK_F32_LOG`, `NK_F32_TAN`, `NK_F32_ABS`.
For double-precision math operations: `NK_F64_SQRT`, `NK_F64_RSQRT`, `NK_F64_LOG`, `NK_F64_TAN`, `NK_F64_ABS`.

> By default, for __non__-SIMD backends, NumKong may use `libc` functions like `sqrt` and `log`.
> Those are generally very accurate, but slow, and introduce a dependency on the C standard library.
> To avoid that you can override those definitions with your custom implementations, like: `#define NK_F32_RSQRT(x) (1 / sqrt(x))`.

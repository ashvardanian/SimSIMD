/**
 *  @file       matmul.h
 *  @brief      SIMD-accelerated mixed-precision Matrix Multiplication kernels.
 *  @author     Ash Vardanian
 *  @date       September 14, 2024
 *
 *  Contains:
 *  - General Matrix Multiplication (GEMM) for Real and Integral numbers.
 *
 *  For datatypes:
 *  - 64-bit IEEE floating point numbers
 *  - 32-bit IEEE floating point numbers
 *  - 16-bit IEEE floating point numbers
 *  - 16-bit brain floating point numbers
 *  - 8-bit signed integers
 *
 *  For hardware architectures:
 *  - x86 (AVX2, AVX512, AMX)
 *  - Arm (NEON, SVE, SME)
 *
 *  x86 intrinsics: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/
 *  Arm intrinsics: https://developer.arm.com/architectures/instruction-sets/intrinsics/
 *
 *  Matrix Multiplication in 40 lines of C by Sergey Slotin: https://en.algorithmica.org/hpc/algorithms/matmul/
 *  LLaMA Now Goes Faster on CPUs by Justine Tunney: https://justine.lol/matmul/
 *  LLM.int8 quantization for PyTorch: https://github.com/bitsandbytes-foundation/bitsandbytes
 */
#ifndef SIMSIMD_MATMUL_H
#define SIMSIMD_MATMUL_H

#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

/*  Serial backends for all numeric types.
 *  By default they use 32-bit arithmetic, unless the arguments themselves contain 64-bit floats.
 *  For double-precision computation check out the "*_accurate" variants of those "*_serial" functions.
 */
SIMSIMD_PUBLIC void simsimd_matmul_f64_serial(                                              //
    simsimd_f64_t const* a, simsimd_size_t lda, simsimd_f64_t const* b, simsimd_size_t ldb, //
    simsimd_size_t a_rows, simsimd_size_t a_cols, simsimd_size_t b_cols, simsimd_f64_t* c, simsimd_size_t ldc);
SIMSIMD_PUBLIC void simsimd_matmul_f32_serial(                                              //
    simsimd_f32_t const* a, simsimd_size_t lda, simsimd_f32_t const* b, simsimd_size_t ldb, //
    simsimd_size_t a_rows, simsimd_size_t a_cols, simsimd_size_t b_cols, simsimd_f32_t* c, simsimd_size_t ldc);
SIMSIMD_PUBLIC void simsimd_matmul_f16_serial(                                              //
    simsimd_f16_t const* a, simsimd_size_t lda, simsimd_f16_t const* b, simsimd_size_t ldb, //
    simsimd_size_t a_rows, simsimd_size_t a_cols, simsimd_size_t b_cols, simsimd_f16_t* c, simsimd_size_t ldc);
SIMSIMD_PUBLIC void simsimd_matmul_bf16_serial(                                               //
    simsimd_bf16_t const* a, simsimd_size_t lda, simsimd_bf16_t const* b, simsimd_size_t ldb, //
    simsimd_size_t a_rows, simsimd_size_t a_cols, simsimd_size_t b_cols, simsimd_bf16_t* c, simsimd_size_t ldc);
SIMSIMD_PUBLIC void simsimd_matmul_i8_serial(                                             //
    simsimd_i8_t const* a, simsimd_size_t lda, simsimd_i8_t const* b, simsimd_size_t ldb, //
    simsimd_size_t a_rows, simsimd_size_t a_cols, simsimd_size_t b_cols, simsimd_i8_t* c, simsimd_size_t ldc);

/*  Double-precision serial backends for all numeric types.
 *  For single-precision computation check out the "*_serial" counterparts of those "*_accurate" functions.
 */
SIMSIMD_PUBLIC void simsimd_matmul_f32_accurate(                                            //
    simsimd_f32_t const* a, simsimd_size_t lda, simsimd_f32_t const* b, simsimd_size_t ldb, //
    simsimd_size_t a_rows, simsimd_size_t a_cols, simsimd_size_t b_cols, simsimd_f32_t* c, simsimd_size_t ldc);
SIMSIMD_PUBLIC void simsimd_matmul_f16_accurate(                                            //
    simsimd_f16_t const* a, simsimd_size_t lda, simsimd_f16_t const* b, simsimd_size_t ldb, //
    simsimd_size_t a_rows, simsimd_size_t a_cols, simsimd_size_t b_cols, simsimd_f16_t* c, simsimd_size_t ldc);
SIMSIMD_PUBLIC void simsimd_matmul_bf16_accurate(                                             //
    simsimd_bf16_t const* a, simsimd_size_t lda, simsimd_bf16_t const* b, simsimd_size_t ldb, //
    simsimd_size_t a_rows, simsimd_size_t a_cols, simsimd_size_t b_cols, simsimd_bf16_t* c, simsimd_size_t ldc);

/*  SIMD-powered backends for Arm NEON, mostly using 32-bit arithmetic over 128-bit words.
 *  By far the most portable backend, covering most Arm v8 devices, over a billion phones, and almost all
 *  server CPUs produced before 2023.
 */
SIMSIMD_PUBLIC void simsimd_matmul_f32_neon(                                                //
    simsimd_f32_t const* a, simsimd_size_t lda, simsimd_f32_t const* b, simsimd_size_t ldb, //
    simsimd_size_t a_rows, simsimd_size_t a_cols, simsimd_size_t b_cols, simsimd_f32_t* c, simsimd_size_t ldc);
SIMSIMD_PUBLIC void simsimd_matmul_f16_neon(                                                //
    simsimd_f16_t const* a, simsimd_size_t lda, simsimd_f16_t const* b, simsimd_size_t ldb, //
    simsimd_size_t a_rows, simsimd_size_t a_cols, simsimd_size_t b_cols, simsimd_f16_t* c, simsimd_size_t ldc);
SIMSIMD_PUBLIC void simsimd_matmul_bf16_neon(                                                 //
    simsimd_bf16_t const* a, simsimd_size_t lda, simsimd_bf16_t const* b, simsimd_size_t ldb, //
    simsimd_size_t a_rows, simsimd_size_t a_cols, simsimd_size_t b_cols, simsimd_bf16_t* c, simsimd_size_t ldc);
SIMSIMD_PUBLIC void simsimd_matmul_i8_neon(                                               //
    simsimd_i8_t const* a, simsimd_size_t lda, simsimd_i8_t const* b, simsimd_size_t ldb, //
    simsimd_size_t a_rows, simsimd_size_t a_cols, simsimd_size_t b_cols, simsimd_i8_t* c, simsimd_size_t ldc);

/*  SIMD-powered backends for Arm SVE, mostly using 32-bit arithmetic over variable-length platform-defined word sizes.
 *  Designed for Arm Graviton 3, Microsoft Cobalt, as well as Nvidia Grace and newer Ampere Altra CPUs.
 */
SIMSIMD_PUBLIC void simsimd_matmul_f32_sve(                                                 //
    simsimd_f32_t const* a, simsimd_size_t lda, simsimd_f32_t const* b, simsimd_size_t ldb, //
    simsimd_size_t a_rows, simsimd_size_t a_cols, simsimd_size_t b_cols, simsimd_f32_t* c, simsimd_size_t ldc);
SIMSIMD_PUBLIC void simsimd_matmul_f16_sve(                                                 //
    simsimd_f16_t const* a, simsimd_size_t lda, simsimd_f16_t const* b, simsimd_size_t ldb, //
    simsimd_size_t a_rows, simsimd_size_t a_cols, simsimd_size_t b_cols, simsimd_f16_t* c, simsimd_size_t ldc);
SIMSIMD_PUBLIC void simsimd_matmul_bf16_sve(                                                //
    simsimd_f16_t const* a, simsimd_size_t lda, simsimd_f16_t const* b, simsimd_size_t ldb, //
    simsimd_size_t a_rows, simsimd_size_t a_cols, simsimd_size_t b_cols, simsimd_f16_t* c, simsimd_size_t ldc);

/*  SIMD-powered backends for AVX2 CPUs of Haswell generation and newer, using 32-bit arithmetic over 256-bit words.
 *  First demonstrated in 2011, at least one Haswell-based processor was still being sold in 2022 — the Pentium G3420.
 *  Practically all modern x86 CPUs support AVX2, FMA, and F16C, making it a perfect baseline for SIMD algorithms.
 *  On other hand, there is no need to implement AVX2 versions of `f32` and `f64` functions, as those are
 *  properly vectorized by recent compilers.
 */
SIMSIMD_PUBLIC void simsimd_matmul_f32_haswell(                                             //
    simsimd_f32_t const* a, simsimd_size_t lda, simsimd_f32_t const* b, simsimd_size_t ldb, //
    simsimd_size_t a_rows, simsimd_size_t a_cols, simsimd_size_t b_cols, simsimd_f32_t* c, simsimd_size_t ldc);
SIMSIMD_PUBLIC void simsimd_matmul_f16_haswell(                                             //
    simsimd_f16_t const* a, simsimd_size_t lda, simsimd_f16_t const* b, simsimd_size_t ldb, //
    simsimd_size_t a_rows, simsimd_size_t a_cols, simsimd_size_t b_cols, simsimd_f16_t* c, simsimd_size_t ldc);
SIMSIMD_PUBLIC void simsimd_matmul_bf16_haswell(                                              //
    simsimd_bf16_t const* a, simsimd_size_t lda, simsimd_bf16_t const* b, simsimd_size_t ldb, //
    simsimd_size_t a_rows, simsimd_size_t a_cols, simsimd_size_t b_cols, simsimd_bf16_t* c, simsimd_size_t ldc);
SIMSIMD_PUBLIC void simsimd_matmul_i8_haswell(                                            //
    simsimd_i8_t const* a, simsimd_size_t lda, simsimd_i8_t const* b, simsimd_size_t ldb, //
    simsimd_size_t a_rows, simsimd_size_t a_cols, simsimd_size_t b_cols, simsimd_i8_t* c, simsimd_size_t ldc);

/*  SIMD-powered backends for various generations of AVX512 CPUs.
 *  Skylake is handy, as it supports masked loads and other operations, avoiding the need for the tail loop.
 *  Ice Lake added VNNI, VPOPCNTDQ, IFMA, VBMI, VAES, GFNI, VBMI2, BITALG, VPCLMULQDQ, and other extensions for integral
 * operations. Genoa added only BF16. Sapphire Rapids added tiled matrix operations in AMX, that we can use for `i8` and
 * `bf16` types.
 */
SIMSIMD_PUBLIC void simsimd_matmul_f32_skylake(                                             //
    simsimd_f32_t const* a, simsimd_size_t lda, simsimd_f32_t const* b, simsimd_size_t ldb, //
    simsimd_size_t a_rows, simsimd_size_t a_cols, simsimd_size_t b_cols, simsimd_f32_t* c, simsimd_size_t ldc);
SIMSIMD_PUBLIC void simsimd_matmul_i8_ice(                                                //
    simsimd_i8_t const* a, simsimd_size_t lda, simsimd_i8_t const* b, simsimd_size_t ldb, //
    simsimd_size_t a_rows, simsimd_size_t a_cols, simsimd_size_t b_cols, simsimd_i8_t* c, simsimd_size_t ldc);
SIMSIMD_PUBLIC void simsimd_matmul_bf16_genoa(                                                //
    simsimd_bf16_t const* a, simsimd_size_t lda, simsimd_bf16_t const* b, simsimd_size_t ldb, //
    simsimd_size_t a_rows, simsimd_size_t a_cols, simsimd_size_t b_cols, simsimd_bf16_t* c, simsimd_size_t ldc);
SIMSIMD_PUBLIC void simsimd_matmul_f16_sapphire(                                            //
    simsimd_f16_t const* a, simsimd_size_t lda, simsimd_f16_t const* b, simsimd_size_t ldb, //
    simsimd_size_t a_rows, simsimd_size_t a_cols, simsimd_size_t b_cols, simsimd_f16_t* c, simsimd_size_t ldc);
SIMSIMD_PUBLIC void simsimd_matmul_bf16_sapphire(                                             //
    simsimd_bf16_t const* a, simsimd_size_t lda, simsimd_bf16_t const* b, simsimd_size_t ldb, //
    simsimd_size_t a_rows, simsimd_size_t a_cols, simsimd_size_t b_cols, simsimd_bf16_t* c, simsimd_size_t ldc);
SIMSIMD_PUBLIC void simsimd_matmul_i8_sapphire(                                               //
    simsimd_bf16_t const* a, simsimd_size_t lda, simsimd_bf16_t const* b, simsimd_size_t ldb, //
    simsimd_size_t a_rows, simsimd_size_t a_cols, simsimd_size_t b_cols, simsimd_bf16_t* c, simsimd_size_t ldc);

#define SIMSIMD_MAKE_MATMUL(name, input_type, accumulator_type, output_type, load_and_convert, convert_and_store)      \
    void simsimd_matmul_##input_type##_##name(simsimd_##input_type##_t const* a, simsimd_size_t lda,                   \
                                              simsimd_##input_type##_t const* b, simsimd_size_t ldb,                   \
                                              simsimd_size_t a_rows, simsimd_size_t a_cols, simsimd_size_t b_cols,     \
                                              simsimd_##output_type##_t* c, simsimd_size_t ldc) {                      \
        for (simsimd_size_t i = 0; i < a_rows; ++i) {                                                                  \
            for (simsimd_size_t j = 0; j < b_cols; ++j) {                                                              \
                simsimd_##accumulator_type##_t sum = 0;                                                                \
                for (simsimd_size_t k = 0; k < a_cols; ++k) {                                                          \
                    simsimd_##accumulator_type##_t aik = load_and_convert(&a[i * lda + k]);                            \
                    simsimd_##accumulator_type##_t bkj = load_and_convert(&b[k * ldb + j]);                            \
                    sum += aik * bkj;                                                                                  \
                }                                                                                                      \
                convert_and_store(sum, &c[i * ldc + j]);                                                               \
            }                                                                                                          \
        }                                                                                                              \
    }

#define SIMSIMD_MAKE_TILED(name, input_type, accumulator_type, output_type, load_and_convert, convert_and_store,       \
                           tile_size)                                                                                  \
    void simsimd_matmul_##input_type##_##name(simsimd_##input_type##_t const* a, simsimd_size_t lda,                   \
                                              simsimd_##input_type##_t const* b, simsimd_size_t ldb,                   \
                                              simsimd_size_t a_rows, simsimd_size_t a_cols, simsimd_size_t b_cols,     \
                                              simsimd_##output_type##_t* c, simsimd_size_t ldc) {                      \
        _Pragma("omp parallel for collapse(2) schedule(static)");                                                      \
        for (simsimd_size_t ii = 0; ii < a_rows; ii += tile_size) {                                                    \
            for (simsimd_size_t jj = 0; jj < b_cols; jj += tile_size) {                                                \
                for (simsimd_size_t kk = 0; kk < a_cols; kk += tile_size) {                                            \
                    simsimd_size_t i_max = (ii + tile_size < a_rows) ? (ii + tile_size) : a_rows;                      \
                    simsimd_size_t j_max = (jj + tile_size < b_cols) ? (jj + tile_size) : b_cols;                      \
                    simsimd_size_t k_max = (kk + tile_size < a_cols) ? (kk + tile_size) : a_cols;                      \
                    for (simsimd_size_t i = ii; i < i_max; ++i) {                                                      \
                        for (simsimd_size_t j = jj; j < j_max; ++j) {                                                  \
                            simsimd_##accumulator_type##_t sum = 0;                                                    \
                            for (simsimd_size_t k = kk; k < k_max; ++k) {                                              \
                                simsimd_##accumulator_type##_t aik = load_and_convert(&a[i * lda + k]);                \
                                simsimd_##accumulator_type##_t bkj = load_and_convert(&b[k * ldb + j]);                \
                                sum += aik * bkj;                                                                      \
                            }                                                                                          \
                            convert_and_store(sum, &c[i * ldc + j]);                                                   \
                        }                                                                                              \
                    }                                                                                                  \
                }                                                                                                      \
            }                                                                                                          \
        }                                                                                                              \
    }

SIMSIMD_MAKE_TILED(serial, f64, f64, f64, SIMSIMD_DEREFERENCE, SIMSIMD_EXPORT, 16)        // simsimd_matmul_f64_serial
SIMSIMD_MAKE_TILED(serial, f32, f32, f32, SIMSIMD_DEREFERENCE, SIMSIMD_EXPORT, 16)        // simsimd_matmul_f32_seria
SIMSIMD_MAKE_TILED(serial, f16, f32, f16, SIMSIMD_F16_TO_F32, SIMSIMD_F32_TO_F16, 16)     // simsimd_matmul_f16_serial
SIMSIMD_MAKE_TILED(serial, bf16, f32, bf16, SIMSIMD_BF16_TO_F32, SIMSIMD_F32_TO_BF16, 16) // simsimd_matmul_bf16_serial
SIMSIMD_MAKE_TILED(serial, i8, i64, i8, SIMSIMD_DEREFERENCE, SIMSIMD_EXPORT, 16)          // simsimd_matmul_i8_serial
SIMSIMD_MAKE_TILED(accurate, f32, f64, f32, SIMSIMD_DEREFERENCE, SIMSIMD_EXPORT, 16)      // simsimd_matmul_f32_accurate
SIMSIMD_MAKE_TILED(accurate, f16, f64, f16, SIMSIMD_F16_TO_F32, SIMSIMD_F32_TO_F16, 16)   // simsimd_matmul_f16_accurate
SIMSIMD_MAKE_TILED(accurate, bf16, f64, bf16, SIMSIMD_BF16_TO_F32, SIMSIMD_F32_TO_BF16,
                   16) // simsimd_matmul_bf16_accurate

#if SIMSIMD_TARGET_ARM
#if SIMSIMD_TARGET_NEON
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+simd")
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+simd"))), apply_to = function)

#pragma clang attribute pop
#pragma GCC pop_options
#endif

#if SIMSIMD_TARGET_NEON_F16
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+simd+fp16")
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+simd+fp16"))), apply_to = function)

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_NEON_F16

#if SIMSIMD_TARGET_NEON_BF16
#pragma GCC push_options
#pragma GCC target("arch=armv8.6-a+simd+bf16")
#pragma clang attribute push(__attribute__((target("arch=armv8.6-a+simd+bf16"))), apply_to = function)

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_NEON_BF16

#if SIMSIMD_TARGET_SVE

#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+sve")
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+sve"))), apply_to = function)

#pragma clang attribute pop
#pragma GCC pop_options

#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+sve+fp16")
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+sve+fp16"))), apply_to = function)

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_SVE
#endif // SIMSIMD_TARGET_ARM

#if SIMSIMD_TARGET_X86
#if SIMSIMD_TARGET_HASWELL
#pragma GCC push_options
#pragma GCC target("avx2", "f16c", "fma")
#pragma clang attribute push(__attribute__((target("avx2,f16c,fma"))), apply_to = function)

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_HASWELL

#if SIMSIMD_TARGET_SKYLAKE
#pragma GCC push_options
#pragma GCC target("avx512f", "avx512vl", "avx512bw", "bmi2")
#pragma clang attribute push(__attribute__((target("avx512f,avx512vl,avx512bw,bmi2"))), apply_to = function)

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_SKYLAKE

#if SIMSIMD_TARGET_GENOA
#pragma GCC push_options
#pragma GCC target("avx512f", "avx512vl", "bmi2", "avx512bw", "avx512bf16")
#pragma clang attribute push(__attribute__((target("avx512f,avx512vl,bmi2,avx512bw,avx512bf16"))), apply_to = function)

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_GENOA

#if SIMSIMD_TARGET_SAPPHIRE
#pragma GCC push_options
#pragma GCC target("avx512f", "avx512vl", "bmi2", "avx512bw", "avx512fp16")
#pragma clang attribute push(__attribute__((target("avx512f,avx512vl,bmi2,avx512bw,avx512fp16"))), apply_to = function)

SIMSIMD_PUBLIC void simsimd_matmul_bf16_sapphire( //
    simsimd_bf16_t const* a, simsimd_size_t lda, simsimd_bf16_t const* b, simsimd_size_t ldb, simsimd_size_t a_rows,
    simsimd_size_t a_cols, simsimd_size_t b_cols, simsimd_bf16_t* c, simsimd_size_t ldc) {}

SIMSIMD_PUBLIC void simsimd_matmul_i8_sapphire( //
    simsimd_bf16_t const* a, simsimd_size_t lda, simsimd_bf16_t const* b, simsimd_size_t ldb, simsimd_size_t a_rows,
    simsimd_size_t a_cols, simsimd_size_t b_cols, simsimd_bf16_t* c, simsimd_size_t ldc) {}

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_SAPPHIRE

#if SIMSIMD_TARGET_ICE
#pragma GCC push_options
#pragma GCC target("avx512f", "avx512vl", "bmi2", "avx512bw", "avx512vnni")
#pragma clang attribute push(__attribute__((target("avx512f,avx512vl,bmi2,avx512bw,avx512vnni"))), apply_to = function)

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_ICE
#endif // SIMSIMD_TARGET_X86

#ifdef __cplusplus
}
#endif

#endif

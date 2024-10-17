/**
 *  @file       fma.h
 *  @brief      SIMD-accelerated mixed-precision Fused-Multiply-Add operations.
 *  @author     Ash Vardanian
 *  @date       October 16, 2024
 *
 *  Contains following element-wise operations:
 *  - Weighted Sum: Oq[i] = Alpha * X[i] + Beta * Z[i]
 *  - FMA or Fused-Multiply-Add: O[i] = Alpha * X[i] * Y[i] + Beta * Z[i]
 *
 *  For datatypes:
 *  - 64-bit IEEE floating point numbers
 *  - 32-bit IEEE floating point numbers
 *  - 16-bit IEEE floating point numbers
 *  - 16-bit brain floating point numbers
 *  - 8-bit unsigned integers
 *  - 8-bit signed integers
 *
 *  For hardware architectures:
 *  - Arm: NEON, SVE
 *  - x86: Haswell, Ice Lake, Skylake, Genoa, Sapphire
 *
 *  x86 intrinsics: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/
 *  Arm intrinsics: https://developer.arm.com/architectures/instruction-sets/intrinsics/
 */
#ifndef SIMSIMD_FMA_H
#define SIMSIMD_FMA_H

#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
}
#endif

#endif

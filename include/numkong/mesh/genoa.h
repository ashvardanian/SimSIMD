/**
 *  @brief SIMD-accelerated Dot Products for Real and Complex Numbers optimized for AMD Genoa CPUs.
 *  @file include/numkong/mesh/genoa.h
 *  @sa include/numkong/mesh.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 */
#ifndef NK_MESH_GENOA_H
#define NK_MESH_GENOA_H

#if _NK_TARGET_X86
#if NK_TARGET_GENOA
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "bmi2", "avx512bw", "avx512bf16")
#pragma clang attribute push(__attribute__((target("avx2,avx512f,avx512vl,bmi2,avx512bw,avx512bf16"))), \
                             apply_to = function)

#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#pragma clang attribute pop
#pragma GCC pop_options
#endif // NK_TARGET_GENOA
#endif // _NK_TARGET_X86

#endif // NK_MESH_GENOA_H
/**
 *  @file       sparse.h
 *  @brief      SIMD-accelerated functions for Sparse Vectors.
 *  @author     Ash Vardanian
 *  @date       March 21, 2024
 *
 *  Contains:
 *  - Set Intersection ~ Jaccard Distance
 *
 *  For datatypes:
 *  - u16 & i16: for vocabularies under 64 K tokens
 *  - u32 & i32: for vocabularies under 4 B tokens
 *
 *  For hardware architectures:
 *  - Arm (SVE)
 *  - x86 (AVX512)
 *
 *  Interestingly, to implement sparse distances and products, the most important function
 *  is analogous to `std::set_intersection`, that outputs the intersection of two sorted
 *  sequences. The naive implementation of that function would look like:
 *
 *      std::size_t intersection = 0;
 *      while (i != a_length && j != b_length) {
 *          scalar_t ai = a[i];
 *          scalar_t bj = b[j];
 *          intersection += ai == bj;
 *          i += ai < bj;
 *          j += ai >= bj;
 *      }
 *
 *  Assuming we are dealing with sparse arrays, most of the time we are just evaluating
 *  branches and skipping entries. So what if we could skip multiple entries at a time
 *  searching for the next chunk, where an intersection is possible.
 *
 *  x86 intrinsics: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/
 *  Arm intrinsics: https://developer.arm.com/architectures/instruction-sets/intrinsics/
 */
#ifndef SIMSIMD_SPARSE_H
#define SIMSIMD_SPARSE_H

#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

// clang-format off

/*  Implements the serial set intersection algorithm, similar to `std::set_intersection in C++ STL`,
 *  but uses clever galloping logic, if the arrays significantly differ in size.
 */
inline static void simsimd_intersect_u16_serial(simsimd_u16_t const* a, simsimd_u16_t const* b, simsimd_size_t a_length, simsimd_size_t b_length, simsimd_distance_t* results);
inline static void simsimd_intersect_u32_serial(simsimd_u32_t const* a, simsimd_u32_t const* b, simsimd_size_t a_length, simsimd_size_t b_length, simsimd_distance_t* results);

/*  Implements the most naive set intersection algorithm, similar to `std::set_intersection in C++ STL`,
 *  naively enumerating the elements of two arrays.
 */
inline static void simsimd_intersect_u16_accurate(simsimd_u16_t const* a, simsimd_u16_t const* b, simsimd_size_t a_length, simsimd_size_t b_length, simsimd_distance_t* results);
inline static void simsimd_intersect_u32_accurate(simsimd_u32_t const* a, simsimd_u32_t const* b, simsimd_size_t a_length, simsimd_size_t b_length, simsimd_distance_t* results);

/*  SIMD-powered backends for Arm NEON, mostly using 32-bit arithmetic over 128-bit words.
 *  By far the most portable backend, covering most Arm v8 devices, over a billion phones, and almost all
 *  server CPUs produced before 2023.
 */
inline static void simsimd_intersect_u32_neon(simsimd_u32_t const* a, simsimd_u32_t const* b, simsimd_size_t a_length, simsimd_size_t b_length, simsimd_distance_t* results);
inline static void simsimd_intersect_u16_neon(simsimd_u16_t const* a, simsimd_u16_t const* b, simsimd_size_t a_length, simsimd_size_t b_length, simsimd_distance_t* results);

/*  SIMD-powered backends for Arm SVE, mostly using 32-bit arithmetic over variable-length platform-defined word sizes.
 *  Designed for Arm Graviton 3, Microsoft Cobalt, as well as Nvidia Grace and newer Ampere Altra CPUs.
 */
inline static void simsimd_intersect_u32_sve(simsimd_u32_t const* a, simsimd_u32_t const* b, simsimd_size_t a_length, simsimd_size_t b_length, simsimd_distance_t* results);
inline static void simsimd_intersect_u16_sve(simsimd_u16_t const* a, simsimd_u16_t const* b, simsimd_size_t a_length, simsimd_size_t b_length, simsimd_distance_t* results);

/*  SIMD-powered backends for AVX2 CPUs of Haswell generation and newer, using 32-bit arithmetic over 256-bit words.
 *  First demonstrated in 2011, at least one Haswell-based processor was still being sold in 2022 — the Pentium G3420.
 *  Practically all modern x86 CPUs support AVX2, FMA, and F16C, making it a perfect baseline for SIMD algorithms.
 *  On other hand, there is no need to implement AVX2 versions of `f32` and `f64` functions, as those are
 *  properly vectorized by recent compilers.
 */
inline static void simsimd_intersect_u32_haswell(simsimd_u32_t const* a, simsimd_u32_t const* b, simsimd_size_t a_length, simsimd_size_t b_length, simsimd_distance_t* results);
inline static void simsimd_intersect_u16_haswell(simsimd_u16_t const* a, simsimd_u16_t const* b, simsimd_size_t a_length, simsimd_size_t b_length, simsimd_distance_t* results);

/*  SIMD-powered backends for various generations of AVX512 CPUs.
 *  Skylake is handy, as it supports masked loads and other operations, avoiding the need for the tail loop.
 *  Ice Lake added VNNI, VPOPCNTDQ, IFMA, VBMI, VAES, GFNI, VBMI2, BITALG, VPCLMULQDQ, and other extensions for integral operations.
 *  Sapphire Rapids added tiled matrix operations, but we are most interested in the new mixed-precision FMA instructions.
 */
inline static void simsimd_intersect_u32_skylake(simsimd_u32_t const* a, simsimd_u32_t const* b, simsimd_size_t a_length, simsimd_size_t b_length, simsimd_distance_t* results);
inline static void simsimd_intersect_u16_skylake(simsimd_u16_t const* a, simsimd_u16_t const* b, simsimd_size_t a_length, simsimd_size_t b_length, simsimd_distance_t* results);
// clang-format on

#define SIMSIMD_MAKE_INTERSECT_LINEAR(name, input_type, accumulator_type)                                              \
    inline static void simsimd_intersect_##input_type##_##name(                                                        \
        simsimd_##input_type##_t const* a, simsimd_##input_type##_t const* b, simsimd_size_t a_length,                 \
        simsimd_size_t b_length, simsimd_distance_t* result) {                                                         \
        simsimd_##accumulator_type##_t intersection = 0;                                                               \
        simsimd_size_t i = 0, j = 0;                                                                                   \
        while (i != a_length && j != b_length) {                                                                       \
            simsimd_##input_type##_t ai = a[i];                                                                        \
            simsimd_##input_type##_t bj = b[j];                                                                        \
            intersection += ai == bj;                                                                                  \
            i += ai < bj;                                                                                              \
            j += ai >= bj;                                                                                             \
        }                                                                                                              \
        *result = intersection;                                                                                        \
    }

SIMSIMD_MAKE_INTERSECT_LINEAR(accurate, u16, size) // simsimd_intersect_u16_accurate
SIMSIMD_MAKE_INTERSECT_LINEAR(accurate, u32, size) // simsimd_intersect_u32_accurate

#define SIMSIMD_MAKE_INTERSECT_GALLOPING(name, input_type, accumulator_type)                                           \
    inline static simsimd_size_t simsimd_galloping_search_##input_type(simsimd_##input_type##_t const* b,              \
                                                                       simsimd_size_t start, simsimd_size_t b_length,  \
                                                                       simsimd_##input_type##_t val) {                 \
        simsimd_size_t low = start;                                                                                    \
        simsimd_size_t high = start + 1;                                                                               \
        while (high < b_length && b[high] < val) {                                                                     \
            low = high;                                                                                                \
            high = (2 * high < b_length) ? 2 * high : b_length;                                                        \
        }                                                                                                              \
        while (low < high) {                                                                                           \
            simsimd_size_t mid = low + (high - low) / 2;                                                               \
            if (b[mid] < val) {                                                                                        \
                low = mid + 1;                                                                                         \
            } else {                                                                                                   \
                high = mid;                                                                                            \
            }                                                                                                          \
        }                                                                                                              \
        return low;                                                                                                    \
    }                                                                                                                  \
                                                                                                                       \
    inline static void simsimd_intersect_##input_type##_##name(                                                        \
        simsimd_##input_type##_t const* a, simsimd_##input_type##_t const* b, simsimd_size_t a_length,                 \
        simsimd_size_t b_length, simsimd_distance_t* result) {                                                         \
        /* Swap arrays if necessary, as we want "b" to be larger than "a" */                                           \
        if (a_length > b_length) {                                                                                     \
            simsimd_##input_type##_t const* temp = a;                                                                  \
            a = b;                                                                                                     \
            b = temp;                                                                                                  \
            simsimd_size_t temp_length = a_length;                                                                     \
            a_length = b_length;                                                                                       \
            b_length = temp_length;                                                                                    \
        }                                                                                                              \
                                                                                                                       \
        /* Use accurate implementation if galloping is not beneficial */                                               \
        if (b_length < 64 * a_length) {                                                                                \
            simsimd_intersect_##input_type##_accurate(a, b, a_length, b_length, result);                               \
            return;                                                                                                    \
        }                                                                                                              \
                                                                                                                       \
        /* Perform galloping, shrinking the target range */                                                            \
        simsimd_##accumulator_type##_t intersection = 0;                                                               \
        simsimd_size_t j = 0;                                                                                          \
        for (simsimd_size_t i = 0; i < a_length; ++i) {                                                                \
            simsimd_##input_type##_t ai = a[i];                                                                        \
            j = simsimd_galloping_search_##input_type(b, j, b_length, ai);                                             \
            if (j < b_length && b[j] == ai) {                                                                          \
                intersection++;                                                                                        \
            }                                                                                                          \
        }                                                                                                              \
        *result = intersection;                                                                                        \
    }

SIMSIMD_MAKE_INTERSECT_GALLOPING(serial, u16, size) // simsimd_intersect_u16_serial
SIMSIMD_MAKE_INTERSECT_GALLOPING(serial, u32, size) // simsimd_intersect_u32_serial

#if SIMSIMD_TARGET_X86
#if SIMSIMD_TARGET_ICE
#pragma GCC push_options
#pragma GCC target("avx512f", "avx512vl", "bmi2", "avx512bw")
#pragma clang attribute push(__attribute__((target("avx512f,avx512vl,bmi2,avx512bw"))), apply_to = function)

inline static void simsimd_intersect_u16_skylake(simsimd_u16_t const* shorter, simsimd_u16_t const* longer,
                                                 simsimd_size_t shorter_length, simsimd_size_t longer_length,
                                                 simsimd_distance_t* results) {
    simsimd_size_t intersection_count = 0;
    simsimd_size_t shorter_idx = 0, longer_idx = 0;
    simsimd_size_t longer_load_size = 32;

    while (shorter_idx < shorter_length && longer_idx < longer_length) {
        // Load `shorter_member` and broadcast it to shorter vector, load `longer_members_vec` from memory.
        simsimd_size_t longer_remaining = longer_length - longer_idx;
        simsimd_u16_t shorter_member = shorter[shorter_idx];
        __m512i shorter_member_vec = _mm512_set1_epi16(shorter_member);
        __m512i longer_members_vec;
        if (longer_remaining >= 32) {
            longer_members_vec = _mm512_loadu_si512((__m512i const*)(longer + longer_idx));
        } else {
            longer_load_size = longer_remaining;
            longer_members_vec = _mm512_maskz_loadu_epi16((__mmask32)_bzhi_u32(0xFFFFFFFF, longer_remaining),
                                                          (__m512i const*)(longer + longer_idx));
        }

        // Compare `shorter_member` with each element in `longer_members_vec`,
        // and jump to the position of the last match.
        __mmask32 equal_mask = _mm512_cmpeq_epi16_mask(shorter_member_vec, longer_members_vec);
        simsimd_size_t equal_count = _mm_popcnt_u32(equal_mask);
        intersection_count += equal_count;
        __mmask32 smaller_mask = _mm512_cmplt_epu16_mask(longer_members_vec, shorter_member_vec);
        simsimd_size_t smaller_count = _mm_popcnt_u32(smaller_mask);

        // We move forward in a shorter array if only there was at least one entry in the register, that was larger.
        shorter_idx += (longer_load_size - smaller_count - equal_count) != 0;
        longer_idx += smaller_count + equal_count;

        // At any given cycle, take one entry from shorter array and compare it with multiple from the longer array.
        // For that, we need to swap the arrays if necessary.
        if ((shorter_length - shorter_idx) > (longer_length - longer_idx)) {
            simsimd_u16_t const* temp_array = shorter;
            shorter = longer, longer = temp_array;
            simsimd_size_t temp_length = shorter_length;
            shorter_length = longer_length, longer_length = temp_length;
            simsimd_size_t temp_idx = shorter_idx;
            shorter_idx = longer_idx, longer_idx = temp_idx;
        }
    }
    *results = intersection_count;
}

inline static void simsimd_intersect_u32_skylake(simsimd_u32_t const* shorter, simsimd_u32_t const* longer,
                                                 simsimd_size_t shorter_length, simsimd_size_t longer_length,
                                                 simsimd_distance_t* results) {
    simsimd_size_t intersection_count = 0;
    simsimd_size_t shorter_idx = 0, longer_idx = 0;
    simsimd_size_t longer_load_size = 16; // Loading 16 x u32 = 512 bits

    while (shorter_idx < shorter_length && longer_idx < longer_length) {
        // Load `shorter_member` and broadcast it to shorter vector, load `longer_members_vec` from memory.
        simsimd_size_t longer_remaining = longer_length - longer_idx;
        simsimd_u32_t shorter_member = shorter[shorter_idx];
        __m512i shorter_member_vec = _mm512_set1_epi32(shorter_member);
        __m512i longer_members_vec;
        if (longer_remaining >= 16) {
            longer_members_vec = _mm512_loadu_si512((__m512i const*)(longer + longer_idx));
        } else {
            longer_load_size = longer_remaining;
            longer_members_vec = _mm512_maskz_loadu_epi32((__mmask16)_bzhi_u32(0xFFFF, longer_remaining),
                                                          (__m512i const*)(longer + longer_idx));
        }

        // Compare `shorter_member` with each element in `longer_members_vec`,
        // and jump to the position of the last match.
        __mmask16 equal_mask = _mm512_cmpeq_epi32_mask(shorter_member_vec, longer_members_vec);
        simsimd_size_t equal_count = _mm_popcnt_u32(equal_mask);
        intersection_count += equal_count;
        __mmask16 smaller_mask = _mm512_cmplt_epu32_mask(longer_members_vec, shorter_member_vec);
        simsimd_size_t smaller_count = _mm_popcnt_u32(smaller_mask);

        // We move forward in a shorter array if only there was at least one entry in the register, that was larger.
        shorter_idx += (longer_load_size - smaller_count - equal_count) != 0;
        longer_idx += smaller_count + equal_count;

        // At any given cycle, take one entry from shorter array and compare it with multiple from the longer array.
        // For that, we need to swap the arrays if necessary.
        if ((shorter_length - shorter_idx) > (longer_length - longer_idx)) {
            simsimd_u32_t const* temp_array = shorter;
            shorter = longer, longer = temp_array;
            simsimd_size_t temp_length = shorter_length;
            shorter_length = longer_length, longer_length = temp_length;
            simsimd_size_t temp_idx = shorter_idx;
            shorter_idx = longer_idx, longer_idx = temp_idx;
        }
    }
    *results = intersection_count;
}

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_ICE
#endif // SIMSIMD_TARGET_X86

#ifdef __cplusplus
}
#endif

#endif

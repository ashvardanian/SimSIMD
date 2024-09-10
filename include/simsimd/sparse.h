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
 *  - u16: for vocabularies under 64 K tokens
 *  - u32: for vocabularies under 4 B tokens
 *
 *  For hardware architectures:
 *  - x86 (AVX512)
 *  - Arm (SVE)
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
SIMSIMD_PUBLIC void simsimd_intersect_u16_serial(simsimd_u16_t const* a, simsimd_u16_t const* b, simsimd_size_t a_length, simsimd_size_t b_length, simsimd_distance_t* results);
SIMSIMD_PUBLIC void simsimd_intersect_u32_serial(simsimd_u32_t const* a, simsimd_u32_t const* b, simsimd_size_t a_length, simsimd_size_t b_length, simsimd_distance_t* results);

/*  Implements the most naive set intersection algorithm, similar to `std::set_intersection in C++ STL`,
 *  naively enumerating the elements of two arrays.
 */
SIMSIMD_PUBLIC void simsimd_intersect_u16_accurate(simsimd_u16_t const* a, simsimd_u16_t const* b, simsimd_size_t a_length, simsimd_size_t b_length, simsimd_distance_t* results);
SIMSIMD_PUBLIC void simsimd_intersect_u32_accurate(simsimd_u32_t const* a, simsimd_u32_t const* b, simsimd_size_t a_length, simsimd_size_t b_length, simsimd_distance_t* results);

/*  SIMD-powered backends for Arm SVE, mostly using 32-bit arithmetic over variable-length platform-defined word sizes.
 *  Designed for Arm Graviton 3, Microsoft Cobalt, as well as Nvidia Grace and newer Ampere Altra CPUs.
 */
SIMSIMD_PUBLIC void simsimd_intersect_u32_sve(simsimd_u32_t const* a, simsimd_u32_t const* b, simsimd_size_t a_length, simsimd_size_t b_length, simsimd_distance_t* results);
SIMSIMD_PUBLIC void simsimd_intersect_u16_sve(simsimd_u16_t const* a, simsimd_u16_t const* b, simsimd_size_t a_length, simsimd_size_t b_length, simsimd_distance_t* results);

/*  SIMD-powered backends for various generations of AVX512 CPUs.
 *  Skylake is handy, as it supports masked loads and other operations, avoiding the need for the tail loop.
 */
SIMSIMD_PUBLIC void simsimd_intersect_u32_ice(simsimd_u32_t const* a, simsimd_u32_t const* b, simsimd_size_t a_length, simsimd_size_t b_length, simsimd_distance_t* results);
SIMSIMD_PUBLIC void simsimd_intersect_u16_ice(simsimd_u16_t const* a, simsimd_u16_t const* b, simsimd_size_t a_length, simsimd_size_t b_length, simsimd_distance_t* results);
// clang-format on

#define SIMSIMD_MAKE_INTERSECT_LINEAR(name, input_type, accumulator_type)                                              \
    SIMSIMD_PUBLIC void simsimd_intersect_##input_type##_##name(                                                       \
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
    SIMSIMD_PUBLIC simsimd_size_t simsimd_galloping_search_##input_type(simsimd_##input_type##_t const* array,         \
                                                                        simsimd_size_t start, simsimd_size_t length,   \
                                                                        simsimd_##input_type##_t val) {                \
        simsimd_size_t low = start;                                                                                    \
        simsimd_size_t high = start + 1;                                                                               \
        while (high < length && array[high] < val) {                                                                   \
            low = high;                                                                                                \
            high = (2 * high < length) ? 2 * high : length;                                                            \
        }                                                                                                              \
        while (low < high) {                                                                                           \
            simsimd_size_t mid = low + (high - low) / 2;                                                               \
            if (array[mid] < val) {                                                                                    \
                low = mid + 1;                                                                                         \
            } else {                                                                                                   \
                high = mid;                                                                                            \
            }                                                                                                          \
        }                                                                                                              \
        return low;                                                                                                    \
    }                                                                                                                  \
                                                                                                                       \
    SIMSIMD_PUBLIC void simsimd_intersect_##input_type##_##name(                                                       \
        simsimd_##input_type##_t const* shorter, simsimd_##input_type##_t const* longer,                               \
        simsimd_size_t shorter_length, simsimd_size_t longer_length, simsimd_distance_t* result) {                     \
        /* Swap arrays if necessary, as we want "longer" to be larger than "shorter" */                                \
        if (longer_length < shorter_length) {                                                                          \
            simsimd_##input_type##_t const* temp = shorter;                                                            \
            shorter = longer;                                                                                          \
            longer = temp;                                                                                             \
            simsimd_size_t temp_length = shorter_length;                                                               \
            shorter_length = longer_length;                                                                            \
            longer_length = temp_length;                                                                               \
        }                                                                                                              \
                                                                                                                       \
        /* Use the accurate implementation if galloping is not beneficial */                                           \
        if (longer_length < 64 * shorter_length) {                                                                     \
            simsimd_intersect_##input_type##_accurate(shorter, longer, shorter_length, longer_length, result);         \
            return;                                                                                                    \
        }                                                                                                              \
                                                                                                                       \
        /* Perform galloping, shrinking the target range */                                                            \
        simsimd_##accumulator_type##_t intersection = 0;                                                               \
        simsimd_size_t j = 0;                                                                                          \
        for (simsimd_size_t i = 0; i < shorter_length; ++i) {                                                          \
            simsimd_##input_type##_t shorter_i = shorter[i];                                                           \
            j = simsimd_galloping_search_##input_type(longer, j, longer_length, shorter_i);                            \
            if (j < longer_length && longer[j] == shorter_i) {                                                         \
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
#pragma GCC target("avx512f", "avx512vl", "bmi2", "lzcnt", "popcnt", "avx512bw", "avx512vbmi2")
#pragma clang attribute push(__attribute__((target("avx512f,avx512vl,bmi2,lzcnt,popcnt,avx512bw,avx512vbmi2"))),       \
                             apply_to = function)

/*  The AVX-512 implementations are inspired by the "Faster-Than-Native Alternatives
 *  for x86 VP2INTERSECT Instructions" paper by Guille Diez-Canas, 2022.
 *
 *      https://github.com/mozonaut/vp2intersect
 *      https://arxiv.org/pdf/2112.06342.pdf
 *
 *  For R&D purposes, it's important to keep the following latencies in mind:
 *
 *   - `_mm512_permutex_epi64` - needs F - 3 cycles latency
 *   - `_mm512_shuffle_epi8` - needs BW - 1 cycle latency
 *   - `_mm512_permutexvar_epi16` - needs BW - 4-6 cycles latency
 *   - `_mm512_permutexvar_epi8` - needs VBMI - 3 cycles latency
 */

SIMSIMD_INTERNAL simsimd_u32_t _mm512_2intersect_epi16_mask(__m512i a, __m512i b) {
    __m512i a1 = _mm512_alignr_epi32(a, a, 4);
    __m512i a2 = _mm512_alignr_epi32(a, a, 8);
    __m512i a3 = _mm512_alignr_epi32(a, a, 12);

    __m512i b1 = _mm512_shuffle_epi32(b, _MM_PERM_ADCB);
    __m512i b2 = _mm512_shuffle_epi32(b, _MM_PERM_BADC);
    __m512i b3 = _mm512_shuffle_epi32(b, _MM_PERM_CBAD);

    __m512i b01 = _mm512_shrdi_epi32(b, b, 16);
    __m512i b11 = _mm512_shrdi_epi32(b1, b1, 16);
    __m512i b21 = _mm512_shrdi_epi32(b2, b2, 16);
    __m512i b31 = _mm512_shrdi_epi32(b3, b3, 16);

    __mmask32 nm00 = _mm512_cmpneq_epi16_mask(a, b);
    __mmask32 nm01 = _mm512_cmpneq_epi16_mask(a1, b);
    __mmask32 nm02 = _mm512_cmpneq_epi16_mask(a2, b);
    __mmask32 nm03 = _mm512_cmpneq_epi16_mask(a3, b);

    __mmask32 nm10 = _mm512_mask_cmpneq_epi16_mask(nm00, a, b01);
    __mmask32 nm11 = _mm512_mask_cmpneq_epi16_mask(nm01, a1, b01);
    __mmask32 nm12 = _mm512_mask_cmpneq_epi16_mask(nm02, a2, b01);
    __mmask32 nm13 = _mm512_mask_cmpneq_epi16_mask(nm03, a3, b01);

    __mmask32 nm20 = _mm512_mask_cmpneq_epi16_mask(nm10, a, b1);
    __mmask32 nm21 = _mm512_mask_cmpneq_epi16_mask(nm11, a1, b1);
    __mmask32 nm22 = _mm512_mask_cmpneq_epi16_mask(nm12, a2, b1);
    __mmask32 nm23 = _mm512_mask_cmpneq_epi16_mask(nm13, a3, b1);

    __mmask32 nm30 = _mm512_mask_cmpneq_epi16_mask(nm20, a, b11);
    __mmask32 nm31 = _mm512_mask_cmpneq_epi16_mask(nm21, a1, b11);
    __mmask32 nm32 = _mm512_mask_cmpneq_epi16_mask(nm22, a2, b11);
    __mmask32 nm33 = _mm512_mask_cmpneq_epi16_mask(nm23, a3, b11);

    __mmask32 nm40 = _mm512_mask_cmpneq_epi16_mask(nm30, a, b2);
    __mmask32 nm41 = _mm512_mask_cmpneq_epi16_mask(nm31, a1, b2);
    __mmask32 nm42 = _mm512_mask_cmpneq_epi16_mask(nm32, a2, b2);
    __mmask32 nm43 = _mm512_mask_cmpneq_epi16_mask(nm33, a3, b2);

    __mmask32 nm50 = _mm512_mask_cmpneq_epi16_mask(nm40, a, b21);
    __mmask32 nm51 = _mm512_mask_cmpneq_epi16_mask(nm41, a1, b21);
    __mmask32 nm52 = _mm512_mask_cmpneq_epi16_mask(nm42, a2, b21);
    __mmask32 nm53 = _mm512_mask_cmpneq_epi16_mask(nm43, a3, b21);

    __mmask32 nm60 = _mm512_mask_cmpneq_epi16_mask(nm50, a, b3);
    __mmask32 nm61 = _mm512_mask_cmpneq_epi16_mask(nm51, a1, b3);
    __mmask32 nm62 = _mm512_mask_cmpneq_epi16_mask(nm52, a2, b3);
    __mmask32 nm63 = _mm512_mask_cmpneq_epi16_mask(nm53, a3, b3);

    __mmask32 nm70 = _mm512_mask_cmpneq_epi16_mask(nm60, a, b31);
    __mmask32 nm71 = _mm512_mask_cmpneq_epi16_mask(nm61, a1, b31);
    __mmask32 nm72 = _mm512_mask_cmpneq_epi16_mask(nm62, a2, b31);
    __mmask32 nm73 = _mm512_mask_cmpneq_epi16_mask(nm63, a3, b31);

    return ~(simsimd_u32_t)(nm70 & simsimd_u32_rol(nm71, 8) & simsimd_u32_rol(nm72, 16) & simsimd_u32_ror(nm73, 8));
}

SIMSIMD_INTERNAL simsimd_u16_t _mm512_2intersect_epi32_mask(__m512i a, __m512i b) {
    __m512i a1 = _mm512_alignr_epi32(a, a, 4);
    __m512i b1 = _mm512_shuffle_epi32(b, _MM_PERM_ADCB);
    __mmask16 nm00 = _mm512_cmpneq_epi32_mask(a, b);

    __m512i a2 = _mm512_alignr_epi32(a, a, 8);
    __m512i a3 = _mm512_alignr_epi32(a, a, 12);
    __mmask16 nm01 = _mm512_cmpneq_epi32_mask(a1, b);
    __mmask16 nm02 = _mm512_cmpneq_epi32_mask(a2, b);

    __mmask16 nm03 = _mm512_cmpneq_epi32_mask(a3, b);
    __mmask16 nm10 = _mm512_mask_cmpneq_epi32_mask(nm00, a, b1);
    __mmask16 nm11 = _mm512_mask_cmpneq_epi32_mask(nm01, a1, b1);

    __m512i b2 = _mm512_shuffle_epi32(b, _MM_PERM_BADC);
    __mmask16 nm12 = _mm512_mask_cmpneq_epi32_mask(nm02, a2, b1);
    __mmask16 nm13 = _mm512_mask_cmpneq_epi32_mask(nm03, a3, b1);
    __mmask16 nm20 = _mm512_mask_cmpneq_epi32_mask(nm10, a, b2);

    __m512i b3 = _mm512_shuffle_epi32(b, _MM_PERM_CBAD);
    __mmask16 nm21 = _mm512_mask_cmpneq_epi32_mask(nm11, a1, b2);
    __mmask16 nm22 = _mm512_mask_cmpneq_epi32_mask(nm12, a2, b2);
    __mmask16 nm23 = _mm512_mask_cmpneq_epi32_mask(nm13, a3, b2);

    __mmask16 nm0 = _mm512_mask_cmpneq_epi32_mask(nm20, a, b3);
    __mmask16 nm1 = _mm512_mask_cmpneq_epi32_mask(nm21, a1, b3);
    __mmask16 nm2 = _mm512_mask_cmpneq_epi32_mask(nm22, a2, b3);
    __mmask16 nm3 = _mm512_mask_cmpneq_epi32_mask(nm23, a3, b3);

    return ~(simsimd_u16_t)(nm0 & simsimd_u16_rol(nm1, 4) & simsimd_u16_rol(nm2, 8) & simsimd_u16_ror(nm3, 4));
}

SIMSIMD_PUBLIC void simsimd_intersect_u16_ice(simsimd_u16_t const* a, simsimd_u16_t const* b, simsimd_size_t a_length,
                                              simsimd_size_t b_length, simsimd_distance_t* results) {

    // The baseline implementation for very small arrays (2 registers or less) can be quite simple:
    if (a_length < 64 && b_length < 64) {
        simsimd_intersect_u16_serial(a, b, a_length, b_length, results);
        return;
    }

    simsimd_u16_t const* const a_end = a + a_length;
    simsimd_u16_t const* const b_end = b + b_length;
    simsimd_size_t c = 0;
    union vec_t {
        __m512i zmm;
        simsimd_u16_t u16[32];
        simsimd_u8_t u8[64];
    } a_vec, b_vec;

    while (a + 32 < a_end && b + 32 < b_end) {
        a_vec.zmm = _mm512_loadu_si512((__m512i const*)a);
        b_vec.zmm = _mm512_loadu_si512((__m512i const*)b);

        // Intersecting registers with `_mm512_2intersect_epi16_mask` involves a lot of shuffling
        // and comparisons, so we want to avoid it if the slices don't overlap at all..
        simsimd_u16_t a_min;
        simsimd_u16_t a_max = a_vec.u16[31];
        simsimd_u16_t b_min = b_vec.u16[0];
        simsimd_u16_t b_max = b_vec.u16[31];

        // If the slices don't overlap, advance the appropriate pointer
        while (a_max < b_min && a + 64 < a_end) {
            a += 32;
            a_vec.zmm = _mm512_loadu_si512((__m512i const*)a);
            a_max = a_vec.u16[31];
        }
        a_min = a_vec.u16[0];
        while (b_max < a_min && b + 64 < b_end) {
            b += 32;
            b_vec.zmm = _mm512_loadu_si512((__m512i const*)b);
            b_max = b_vec.u16[31];
        }
        b_min = b_vec.u16[0];

        // Now we are likely to have some overlap, so we can intersect the registers
        __mmask32 a_matches = _mm512_2intersect_epi16_mask(a_vec.zmm, b_vec.zmm);

        // The paper also contained a very nice procedure for exporting the matches,
        // but we don't need it here:
        //      _mm512_mask_compressstoreu_epi16(c, a_matches, a_vec);
        c += _mm_popcnt_u32(a_matches); // The `_popcnt32` symbol isn't recognized by MSVC

        __m512i a_last_broadcasted = _mm512_set1_epi16(*(short const*)&a_max);
        __m512i b_last_broadcasted = _mm512_set1_epi16(*(short const*)&b_max);
        __mmask32 a_step_mask = _mm512_cmple_epu16_mask(a_vec.zmm, b_last_broadcasted);
        __mmask32 b_step_mask = _mm512_cmple_epu16_mask(b_vec.zmm, a_last_broadcasted);
        a += 32 - _lzcnt_u32((simsimd_u32_t)a_step_mask);
        b += 32 - _lzcnt_u32((simsimd_u32_t)b_step_mask);
    }

    simsimd_intersect_u16_serial(a, b, a_end - a, b_end - b, results);
    *results += c;
}

SIMSIMD_PUBLIC void simsimd_intersect_u32_ice(simsimd_u32_t const* a, simsimd_u32_t const* b, simsimd_size_t a_length,
                                              simsimd_size_t b_length, simsimd_distance_t* results) {

    // The baseline implementation for very small arrays (2 registers or less) can be quite simple:
    if (a_length < 32 && b_length < 32) {
        simsimd_intersect_u32_serial(a, b, a_length, b_length, results);
        return;
    }

    simsimd_u32_t const* const a_end = a + a_length;
    simsimd_u32_t const* const b_end = b + b_length;
    simsimd_size_t c = 0;
    union vec_t {
        __m512i zmm;
        simsimd_u32_t u32[16];
        simsimd_u8_t u8[64];
    } a_vec, b_vec;

    while (a + 16 < a_end && b + 16 < b_end) {
        a_vec.zmm = _mm512_loadu_si512((__m512i const*)a);
        b_vec.zmm = _mm512_loadu_si512((__m512i const*)b);

        // Intersecting registers with `_mm512_2intersect_epi32_mask` involves a lot of shuffling
        // and comparisons, so we want to avoid it if the slices don't overlap at all..
        simsimd_u32_t a_min;
        simsimd_u32_t a_max = a_vec.u32[15];
        simsimd_u32_t b_min = b_vec.u32[0];
        simsimd_u32_t b_max = b_vec.u32[15];

        // If the slices don't overlap, advance the appropriate pointer
        while (a_max < b_min && a + 32 < a_end) {
            a += 16;
            a_vec.zmm = _mm512_loadu_si512((__m512i const*)a);
            a_max = a_vec.u32[15];
        }
        a_min = a_vec.u32[0];
        while (b_max < a_min && b + 32 < b_end) {
            b += 16;
            b_vec.zmm = _mm512_loadu_si512((__m512i const*)b);
            b_max = b_vec.u32[15];
        }
        b_min = b_vec.u32[0];

        // Now we are likely to have some overlap, so we can intersect the registers
        __mmask16 a_matches = _mm512_2intersect_epi32_mask(a_vec.zmm, b_vec.zmm);

        // The paper also contained a very nice procedure for exporting the matches,
        // but we don't need it here:
        //      _mm512_mask_compressstoreu_epi32(c, a_matches, a_vec);
        c += _mm_popcnt_u32(a_matches); // The `_popcnt32` symbol isn't recognized by MSVC

        __m512i a_last_broadcasted = _mm512_set1_epi32(*(int const*)&a_max);
        __m512i b_last_broadcasted = _mm512_set1_epi32(*(int const*)&b_max);
        __mmask16 a_step_mask = _mm512_cmple_epu32_mask(a_vec.zmm, b_last_broadcasted);
        __mmask16 b_step_mask = _mm512_cmple_epu32_mask(b_vec.zmm, a_last_broadcasted);
        a += 32 - _lzcnt_u32((simsimd_u32_t)a_step_mask);
        b += 32 - _lzcnt_u32((simsimd_u32_t)b_step_mask);
    }

    simsimd_intersect_u32_serial(a, b, a_end - a, b_end - b, results);
    *results += c;
}

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_ICE
#endif // SIMSIMD_TARGET_X86

#if SIMSIMD_TARGET_ARM
#if SIMSIMD_TARGET_SVE

#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+sve")
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+sve"))), apply_to = function)

SIMSIMD_PUBLIC void simsimd_intersect_u16_sve(simsimd_u16_t const* shorter, simsimd_u16_t const* longer,
                                              simsimd_size_t shorter_length, simsimd_size_t longer_length,
                                              simsimd_distance_t* results) {

    // Temporarily disable SVE: https://github.com/ashvardanian/SimSIMD/issues/168
    simsimd_intersect_u16_serial(shorter, longer, shorter_length, longer_length, results);
    return;

    // SVE implementations with 128-bit registers can only fit 8x 16-bit words,
    // making this kernel quite inefficient. Let's aim for registers of 256 bits and larger.
    simsimd_size_t longer_load_size = svcnth();
    if (longer_load_size < 16) {
        simsimd_intersect_u16_serial(shorter, longer, shorter_length, longer_length, results);
        return;
    }

    simsimd_size_t intersection_count = 0;
    simsimd_size_t shorter_idx = 0, longer_idx = 0;
    while (shorter_idx < shorter_length && longer_idx < longer_length) {
        // Load `shorter_member` and broadcast it, load `longer_members_vec` from memory
        simsimd_size_t longer_remaining = longer_length - longer_idx;
        simsimd_u16_t shorter_member = shorter[shorter_idx];
        svbool_t pg = svwhilelt_b16_u64(longer_idx, longer_length);
        svuint16_t shorter_member_vec = svdup_n_u16(shorter_member);
        svuint16_t longer_members_vec = svld1_u16(pg, longer + longer_idx);

        // Compare `shorter_member` with each element in `longer_members_vec`
        svbool_t equal_mask = svcmpeq_u16(pg, shorter_member_vec, longer_members_vec);
        simsimd_size_t equal_count = svcntp_b16(svptrue_b16(), equal_mask);
        intersection_count += equal_count;

        // Count the number of elements in `longer_members_vec` that are less than `shorter_member`
        svbool_t smaller_mask = svcmplt_u16(pg, longer_members_vec, shorter_member_vec);
        simsimd_size_t smaller_count = svcntp_b16(svptrue_b16(), smaller_mask);

        // Advance pointers
        longer_load_size = longer_remaining < longer_load_size ? longer_remaining : longer_load_size;
        shorter_idx += (longer_load_size - smaller_count - equal_count) != 0;
        longer_idx += smaller_count + equal_count;

        // Swap arrays if necessary
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

SIMSIMD_PUBLIC void simsimd_intersect_u32_sve(simsimd_u32_t const* shorter, simsimd_u32_t const* longer,
                                              simsimd_size_t shorter_length, simsimd_size_t longer_length,
                                              simsimd_distance_t* results) {

    // Temporarily disable SVE: https://github.com/ashvardanian/SimSIMD/issues/168
    simsimd_intersect_u32_serial(shorter, longer, shorter_length, longer_length, results);
    return;

    // SVE implementations with 128-bit registers can only fit 4x 32-bit words,
    // making this kernel quite inefficient. Let's aim for registers of 256 bits and larger.
    simsimd_size_t longer_load_size = svcntw();
    if (longer_load_size < 8) {
        simsimd_intersect_u32_serial(shorter, longer, shorter_length, longer_length, results);
        return;
    }

    simsimd_size_t intersection_count = 0;
    simsimd_size_t shorter_idx = 0, longer_idx = 0;
    while (shorter_idx < shorter_length && longer_idx < longer_length) {
        // Load `shorter_member` and broadcast it, load `longer_members_vec` from memory
        simsimd_size_t longer_remaining = longer_length - longer_idx;
        simsimd_u32_t shorter_member = shorter[shorter_idx];
        svbool_t pg = svwhilelt_b32_u64(longer_idx, longer_length);
        svuint32_t shorter_member_vec = svdup_n_u32(shorter_member);
        svuint32_t longer_members_vec = svld1_u32(pg, longer + longer_idx);

        // Compare `shorter_member` with each element in `longer_members_vec`
        svbool_t equal_mask = svcmpeq_u32(pg, shorter_member_vec, longer_members_vec);
        simsimd_size_t equal_count = svcntp_b32(svptrue_b32(), equal_mask);
        intersection_count += equal_count;

        // Count the number of elements in `longer_members_vec` that are less than `shorter_member`
        svbool_t smaller_mask = svcmplt_u32(pg, longer_members_vec, shorter_member_vec);
        simsimd_size_t smaller_count = svcntp_b32(svptrue_b32(), smaller_mask);

        // Advance pointers
        longer_load_size = longer_remaining < longer_load_size ? longer_remaining : longer_load_size;
        shorter_idx += (longer_load_size - smaller_count - equal_count) != 0;
        longer_idx += smaller_count + equal_count;

        // Swap arrays if necessary
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
#endif // SIMSIMD_TARGET_SVE
#endif // SIMSIMD_TARGET_ARM

#ifdef __cplusplus
}
#endif

#endif

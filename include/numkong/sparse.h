/**
 *  @brief SIMD-accelerated sparse set operations and dot products.
 *  @file include/numkong/sparse.h
 *  @author Ash Vardanian
 *  @date March 21, 2024
 *
 *  Contains:
 *
 *  - Set intersection for sorted unique arrays → `u32` count
 *  - Sparse dot products for weighted sparse vectors → `f32` product
 *
 *  For dtypes:
 *
 *  - `u16`: indices for vocabularies under 64 thousand tokens
 *  - `u32`: indices for vocabularies under 4 billion tokens
 *  - `u64`: indices for trillion-scale combinatorics and graphs
 *  - `u16` indices + `bf16` weights → `f32` product
 *  - `u32` indices + `f32` weights → `f32` product
 *
 *  For hardware architectures:
 *
 *  - Arm: NEON, SVE2
 *  - x86: Ice Lake, Turin
 *
 *  @section intersection_algorithm Intersection by Merge
 *
 *  The core primitive is analogous to `std::set_intersection`, taking two sorted arrays
 *  of unique values and producing the intersection size:
 *
 *      std::size_t intersection_size = 0;
 *      while (i != a_length && j != b_length) {
 *          scalar_t ai = a[i], bj = b[j];
 *          intersection_size += ai == bj;
 *          i += ai < bj;
 *          j += ai ≥ bj;
 *      }
 *
 *  Weighted sparse dot-products follow the same merge loop, but accumulate a product
 *  for matching indices:
 *
 *      double product = 0;
 *      while (i != a_length && j != b_length) {
 *          scalar_t ai = a[i], bj = b[j];
 *          product += ai == bj ? a_weights[i] * b_weights[j] : 0;
 *          i += ai < bj;
 *          j += ai ≥ bj;
 *      }
 *
 *  @section galloping_search Galloping vs Linear
 *
 *  When the arrays are highly imbalanced, linear merge wastes cycles skipping elements.
 *  The serial implementation switches to a galloping search to jump over large gaps.
 *
 *  @section x86_instructions Relevant x86 Instructions
 *
 *  The Ice Lake kernels are shuffle/compare heavy; their throughput is often gated by port 5.
 *  On Genoa, many integer ops dual-issue on FP ports, often improving throughput despite higher latency.
 *
 *      Intrinsic                       Instruction                      Ice           Genoa
 *      _mm512_shuffle_epi32            VPSHUFD (ZMM, ZMM, I8)           1c @ p5       1c @ p123
 *      _mm512_mask_cmpneq_epi32_mask   VPCMPD (K, ZMM, ZMM, I8)         3c @ p5       5c @ p01
 *      _mm512_alignr_epi32             VALIGND (ZMM, ZMM, ZMM, I8)      3c @ p5       6c @ p12
 *      _mm512_conflict_epi32           VPCONFLICTD (ZMM, ZMM)           26c @ p0/5    7c @ p01/12
 *      _mm256_maskz_compress_epi16     VPCOMPRESSW (YMM, K, YMM)        3-6c @ p5     4-8c @ p01/12
 *      _mm256_dpwssds_epi32            VPDPWSSDS (YMM, K, YMM, YMM)     4-5c @ p01    4c @ p01
 *      _mm256_dpbf16_ps                VDPBF16PS (YMM, YMM, YMM)        n/a           6c @ p01
 *
 *  VP2INTERSECTD is unsupported on Ice Lake and not yet covered by uops.info for Zen5/Turin.
 *  Tiger Lake measures ~36-41c @ p5 for ZMM variants, which is why we always avoid it on Intel.
 *
 *  @section references References
 *
 *  - uops.info: https://uops.info/
 *  - Intel Intrinsics Guide: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/
 *  - Arm Intrinsics Reference: https://developer.arm.com/architectures/instruction-sets/intrinsics/
 *  - vp2intersect experiments: https://github.com/mozonaut/vp2intersect
 *  - Diez-Canas "Faster-Than-Native Alternatives for x86 VP2INTERSECT Instructions":
 *    https://arxiv.org/pdf/2112.06342.pdf
 *
 */
#ifndef NK_SPARSE_H
#define NK_SPARSE_H

#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

/**
 *  @brief Set intersection between two sorted u16 arrays.
 *
 *  @param[in] a The first sorted array of indices.
 *  @param[in] b The second sorted array of indices.
 *  @param[in] a_length The number of elements in the first array.
 *  @param[in] b_length The number of elements in the second array.
 *  @param[out] result Output buffer for intersection elements, or NULL to count only.
 *  @param[out] count The output intersection count.
 *
 *  @note Inputs must be sorted in ascending order and contain unique elements.
 */
NK_DYNAMIC void nk_sparse_intersect_u16( //
    nk_u16_t const *a, nk_u16_t const *b, nk_size_t a_length, nk_size_t b_length, nk_u16_t *result, nk_size_t *count);

/**
 *  @brief Set intersection between two sorted u32 arrays.
 *
 *  @param[in] a The first sorted array of indices.
 *  @param[in] b The second sorted array of indices.
 *  @param[in] a_length The number of elements in the first array.
 *  @param[in] b_length The number of elements in the second array.
 *  @param[out] result Output buffer for intersection elements, or NULL to count only.
 *  @param[out] count The output intersection count.
 *
 *  @note Inputs must be sorted in ascending order and contain unique elements.
 */
NK_DYNAMIC void nk_sparse_intersect_u32( //
    nk_u32_t const *a, nk_u32_t const *b, nk_size_t a_length, nk_size_t b_length, nk_u32_t *result, nk_size_t *count);

/**
 *  @brief Set intersection between two sorted u64 arrays.
 *
 *  @param[in] a The first sorted array of indices.
 *  @param[in] b The second sorted array of indices.
 *  @param[in] a_length The number of elements in the first array.
 *  @param[in] b_length The number of elements in the second array.
 *  @param[out] result Output buffer for intersection elements, or NULL to count only.
 *  @param[out] count The output intersection count.
 *
 *  @note Inputs must be sorted in ascending order and contain unique elements.
 */
NK_DYNAMIC void nk_sparse_intersect_u64( //
    nk_u64_t const *a, nk_u64_t const *b, nk_size_t a_length, nk_size_t b_length, nk_u64_t *result, nk_size_t *count);

/**
 *  @brief Sparse dot-product over u16 indices with bf16 weights.
 *
 *  @param[in] a The first sorted array of indices.
 *  @param[in] b The second sorted array of indices.
 *  @param[in] a_weights The bf16 weights for the first array.
 *  @param[in] b_weights The bf16 weights for the second array.
 *  @param[in] a_length The number of elements in the first array.
 *  @param[in] b_length The number of elements in the second array.
 *  @param[out] product The output dot product.
 *
 *  @note Inputs must be sorted in ascending order and contain unique elements.
 */
NK_DYNAMIC void nk_sparse_dot_u16bf16( //
    nk_u16_t const *a, nk_u16_t const *b, nk_bf16_t const *a_weights, nk_bf16_t const *b_weights, nk_size_t a_length,
    nk_size_t b_length, nk_f32_t *product);

/**
 *  @brief Sparse dot-product over u32 indices with f32 weights.
 *
 *  @param[in] a The first sorted array of indices.
 *  @param[in] b The second sorted array of indices.
 *  @param[in] a_weights The f32 weights for the first array.
 *  @param[in] b_weights The f32 weights for the second array.
 *  @param[in] a_length The number of elements in the first array.
 *  @param[in] b_length The number of elements in the second array.
 *  @param[out] product The output dot product.
 *
 *  @note Inputs must be sorted in ascending order and contain unique elements.
 */
NK_DYNAMIC void nk_sparse_dot_u32f32( //
    nk_u32_t const *a, nk_u32_t const *b, nk_f32_t const *a_weights, nk_f32_t const *b_weights, nk_size_t a_length,
    nk_size_t b_length, nk_f32_t *product);

/** @copydoc nk_sparse_intersect_u16 */
NK_PUBLIC void nk_sparse_intersect_u16_serial(nk_u16_t const *a, nk_u16_t const *b, nk_size_t a_length,
                                              nk_size_t b_length, nk_u16_t *result, nk_size_t *count);
/** @copydoc nk_sparse_intersect_u32 */
NK_PUBLIC void nk_sparse_intersect_u32_serial(nk_u32_t const *a, nk_u32_t const *b, nk_size_t a_length,
                                              nk_size_t b_length, nk_u32_t *result, nk_size_t *count);
/** @copydoc nk_sparse_intersect_u64 */
NK_PUBLIC void nk_sparse_intersect_u64_serial(nk_u64_t const *a, nk_u64_t const *b, nk_size_t a_length,
                                              nk_size_t b_length, nk_u64_t *result, nk_size_t *count);
/** @copydoc nk_sparse_dot_u16bf16 */
NK_PUBLIC void nk_sparse_dot_u16bf16_serial(nk_u16_t const *a, nk_u16_t const *b, nk_bf16_t const *a_weights,
                                            nk_bf16_t const *b_weights, nk_size_t a_length, nk_size_t b_length,
                                            nk_f32_t *product);
/** @copydoc nk_sparse_dot_u32f32 */
NK_PUBLIC void nk_sparse_dot_u32f32_serial(nk_u32_t const *a, nk_u32_t const *b, nk_f32_t const *a_weights,
                                           nk_f32_t const *b_weights, nk_size_t a_length, nk_size_t b_length,
                                           nk_f32_t *product);

#if NK_TARGET_NEON
/** @copydoc nk_sparse_intersect_u16 */
NK_PUBLIC void nk_sparse_intersect_u16_neon(nk_u16_t const *a, nk_u16_t const *b, nk_size_t a_length,
                                            nk_size_t b_length, nk_u16_t *result, nk_size_t *count);
/** @copydoc nk_sparse_intersect_u32 */
NK_PUBLIC void nk_sparse_intersect_u32_neon(nk_u32_t const *a, nk_u32_t const *b, nk_size_t a_length,
                                            nk_size_t b_length, nk_u32_t *result, nk_size_t *count);
/** @copydoc nk_sparse_intersect_u64 */
NK_PUBLIC void nk_sparse_intersect_u64_neon(nk_u64_t const *a, nk_u64_t const *b, nk_size_t a_length,
                                            nk_size_t b_length, nk_u64_t *result, nk_size_t *count);
#endif // NK_TARGET_NEON

#if NK_TARGET_SVE2
/** @copydoc nk_sparse_intersect_u16 */
NK_PUBLIC void nk_sparse_intersect_u16_sve2(nk_u16_t const *a, nk_u16_t const *b, nk_size_t a_length,
                                            nk_size_t b_length, nk_u16_t *result, nk_size_t *count);
/** @copydoc nk_sparse_intersect_u32 */
NK_PUBLIC void nk_sparse_intersect_u32_sve2(nk_u32_t const *a, nk_u32_t const *b, nk_size_t a_length,
                                            nk_size_t b_length, nk_u32_t *result, nk_size_t *count);
/** @copydoc nk_sparse_intersect_u64 */
NK_PUBLIC void nk_sparse_intersect_u64_sve2(nk_u64_t const *a, nk_u64_t const *b, nk_size_t a_length,
                                            nk_size_t b_length, nk_u64_t *result, nk_size_t *count);
/** @copydoc nk_sparse_dot_u32f32 */
NK_PUBLIC void nk_sparse_dot_u32f32_sve2(nk_u32_t const *a, nk_u32_t const *b, nk_f32_t const *a_weights,
                                         nk_f32_t const *b_weights, nk_size_t a_length, nk_size_t b_length,
                                         nk_f32_t *product);
#endif // NK_TARGET_SVE2

#if NK_TARGET_SVE2 && NK_TARGET_SVEBFDOT
/** @copydoc nk_sparse_dot_u16bf16 */
NK_PUBLIC void nk_sparse_dot_u16bf16_sve2(nk_u16_t const *a, nk_u16_t const *b, nk_bf16_t const *a_weights,
                                          nk_bf16_t const *b_weights, nk_size_t a_length, nk_size_t b_length,
                                          nk_f32_t *product);
#endif // NK_TARGET_SVE2 && NK_TARGET_SVEBFDOT

#if NK_TARGET_ICE
/** @copydoc nk_sparse_intersect_u16 */
NK_PUBLIC void nk_sparse_intersect_u16_ice(nk_u16_t const *a, nk_u16_t const *b, nk_size_t a_length, nk_size_t b_length,
                                           nk_u16_t *result, nk_size_t *count);
/** @copydoc nk_sparse_intersect_u32 */
NK_PUBLIC void nk_sparse_intersect_u32_ice(nk_u32_t const *a, nk_u32_t const *b, nk_size_t a_length, nk_size_t b_length,
                                           nk_u32_t *result, nk_size_t *count);
/** @copydoc nk_sparse_intersect_u64 */
NK_PUBLIC void nk_sparse_intersect_u64_ice(nk_u64_t const *a, nk_u64_t const *b, nk_size_t a_length, nk_size_t b_length,
                                           nk_u64_t *result, nk_size_t *count);
/** @copydoc nk_sparse_dot_u32f32 */
NK_PUBLIC void nk_sparse_dot_u32f32_ice(nk_u32_t const *a, nk_u32_t const *b, nk_f32_t const *a_weights,
                                        nk_f32_t const *b_weights, nk_size_t a_length, nk_size_t b_length,
                                        nk_f32_t *product);
#endif // NK_TARGET_ICE

#if NK_TARGET_TURIN
/** @copydoc nk_sparse_intersect_u16 */
NK_PUBLIC void nk_sparse_intersect_u16_turin(nk_u16_t const *a, nk_u16_t const *b, nk_size_t a_length,
                                             nk_size_t b_length, nk_u16_t *result, nk_size_t *count);
/** @copydoc nk_sparse_intersect_u32 */
NK_PUBLIC void nk_sparse_intersect_u32_turin(nk_u32_t const *a, nk_u32_t const *b, nk_size_t a_length,
                                             nk_size_t b_length, nk_u32_t *result, nk_size_t *count);
/** @copydoc nk_sparse_intersect_u64 */
NK_PUBLIC void nk_sparse_intersect_u64_turin(nk_u64_t const *a, nk_u64_t const *b, nk_size_t a_length,
                                             nk_size_t b_length, nk_u64_t *result, nk_size_t *count);
/** @copydoc nk_sparse_dot_u16bf16 */
NK_PUBLIC void nk_sparse_dot_u16bf16_turin(nk_u16_t const *a, nk_u16_t const *b, nk_bf16_t const *a_weights,
                                           nk_bf16_t const *b_weights, nk_size_t a_length, nk_size_t b_length,
                                           nk_f32_t *product);
/** @copydoc nk_sparse_dot_u32f32 */
NK_PUBLIC void nk_sparse_dot_u32f32_turin(nk_u32_t const *a, nk_u32_t const *b, nk_f32_t const *a_weights,
                                          nk_f32_t const *b_weights, nk_size_t a_length, nk_size_t b_length,
                                          nk_f32_t *product);
#endif // NK_TARGET_TURIN

#define nk_define_sparse_intersect_(input_type)                                                                      \
    NK_PUBLIC nk_size_t nk_sparse_intersect_##input_type##_galloping_search_(                                        \
        nk_##input_type##_t const *array, nk_size_t start, nk_size_t length, nk_##input_type##_t val) {              \
        nk_size_t low = start;                                                                                       \
        nk_size_t high = start + 1;                                                                                  \
        while (high < length && array[high] < val) {                                                                 \
            low = high;                                                                                              \
            high = (2 * high < length) ? 2 * high : length;                                                          \
        }                                                                                                            \
        while (low < high) {                                                                                         \
            nk_size_t mid = low + (high - low) / 2;                                                                  \
            if (array[mid] < val) { low = mid + 1; }                                                                 \
            else { high = mid; }                                                                                     \
        }                                                                                                            \
        return low;                                                                                                  \
    }                                                                                                                \
    NK_PUBLIC nk_size_t nk_sparse_intersect_##input_type##_linear_scan_(                                             \
        nk_##input_type##_t const *a, nk_##input_type##_t const *b, nk_size_t a_length, nk_size_t b_length,          \
        nk_##input_type##_t *result) {                                                                               \
        nk_size_t intersection_size = 0;                                                                             \
        nk_size_t i = 0, j = 0;                                                                                      \
        while (i != a_length && j != b_length) {                                                                     \
            nk_##input_type##_t ai = a[i];                                                                           \
            nk_##input_type##_t bj = b[j];                                                                           \
            if (ai == bj) {                                                                                          \
                if (result) result[intersection_size] = ai;                                                          \
                intersection_size++;                                                                                 \
            }                                                                                                        \
            i += ai <= bj;                                                                                           \
            j += ai >= bj;                                                                                           \
        }                                                                                                            \
        return intersection_size;                                                                                    \
    }                                                                                                                \
    NK_PUBLIC void nk_sparse_intersect_##input_type##_serial(                                                        \
        nk_##input_type##_t const *shorter, nk_##input_type##_t const *longer, nk_size_t shorter_length,             \
        nk_size_t longer_length, nk_##input_type##_t *result, nk_size_t *count) {                                    \
        /* Swap arrays if necessary, as we want "longer" to be larger than "shorter" */                              \
        if (longer_length < shorter_length) {                                                                        \
            nk_##input_type##_t const *temp = shorter;                                                               \
            shorter = longer;                                                                                        \
            longer = temp;                                                                                           \
            nk_size_t temp_length = shorter_length;                                                                  \
            shorter_length = longer_length;                                                                          \
            longer_length = temp_length;                                                                             \
        }                                                                                                            \
                                                                                                                     \
        /* Use the accurate implementation if galloping is not beneficial */                                         \
        if (longer_length < 64 * shorter_length) {                                                                   \
            *count = nk_sparse_intersect_##input_type##_linear_scan_(shorter, longer, shorter_length, longer_length, \
                                                                     result);                                        \
            return;                                                                                                  \
        }                                                                                                            \
                                                                                                                     \
        /* Perform galloping, shrinking the target range */                                                          \
        nk_size_t intersection_size = 0;                                                                             \
        nk_size_t j = 0;                                                                                             \
        for (nk_size_t i = 0; i < shorter_length; ++i) {                                                             \
            nk_##input_type##_t shorter_i = shorter[i];                                                              \
            j = nk_sparse_intersect_##input_type##_galloping_search_(longer, j, longer_length, shorter_i);           \
            if (j < longer_length && longer[j] == shorter_i) {                                                       \
                if (result) result[intersection_size] = shorter_i;                                                   \
                intersection_size++;                                                                                 \
            }                                                                                                        \
        }                                                                                                            \
        *count = intersection_size;                                                                                  \
    }

#define nk_define_sparse_dot_(input_type, weight_type, accumulator_type, load_and_convert)                  \
    NK_PUBLIC void nk_sparse_dot_##input_type##weight_type##_serial(                                        \
        nk_##input_type##_t const *a, nk_##input_type##_t const *b, nk_##weight_type##_t const *a_weights,  \
        nk_##weight_type##_t const *b_weights, nk_size_t a_length, nk_size_t b_length, nk_f32_t *product) { \
        nk_##accumulator_type##_t weights_product = 0, awi, bwi;                                            \
        nk_size_t i = 0, j = 0;                                                                             \
        while (i != a_length && j != b_length) {                                                            \
            nk_##input_type##_t ai = a[i];                                                                  \
            nk_##input_type##_t bj = b[j];                                                                  \
            int matches = ai == bj;                                                                         \
            load_and_convert(a_weights + i, &awi);                                                          \
            load_and_convert(b_weights + j, &bwi);                                                          \
            weights_product += matches * awi * bwi;                                                         \
            i += ai < bj;                                                                                   \
            j += ai >= bj;                                                                                  \
        }                                                                                                   \
        *product = (nk_f32_t)weights_product;                                                               \
    }

nk_define_sparse_intersect_(u16) // nk_sparse_intersect_u16_serial
nk_define_sparse_intersect_(u32) // nk_sparse_intersect_u32_serial
nk_define_sparse_intersect_(u64) // nk_sparse_intersect_u64_serial

nk_define_sparse_dot_(u16, bf16, f32, nk_bf16_to_f32_serial) // nk_sparse_dot_u16bf16_serial
nk_define_sparse_dot_(u32, f32, f32, nk_assign_from_to_)     // nk_sparse_dot_u32f32_serial

/*  The AVX-512 implementations are inspired by the "Faster-Than-Native Alternatives
 *  for x86 VP2INTERSECT Instructions" paper by Guille Diez-Canas, 2022.
 *
 *      https://github.com/mozonaut/vp2intersect
 *      https://arxiv.org/pdf/2112.06342.pdf
 *
 *  For R&D purposes, it's important to keep the following latencies in mind:
 *
 *   - `_mm512_permutex_epi64` (VPERMQ) - needs F - 3 cy latency, 1 cy throughput @ p5
 *   - `_mm512_shuffle_epi8` (VPSHUFB) - needs BW - 1 cy latency, 1 cy throughput @ p5
 *   - `_mm512_permutexvar_epi16` (VPERMW) - needs BW - 4-6 cy latency, 1 cy throughput @ p5
 *   - `_mm512_permutexvar_epi8` (VPERMB) - needs VBMI - 3 cy latency, 1 cy throughput @ p5
 */
#if NK_TARGET_X86_
#if NK_TARGET_ICE
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,avx512f,avx512vl,bmi2,lzcnt,popcnt,avx512bw,avx512vbmi2"))), \
                             apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "bmi2", "lzcnt", "popcnt", "avx512bw", "avx512vbmi2")
#endif

/**
 *  @brief  Analogous to `_mm512_2intersect_epi16_mask`, but compatible with Ice Lake CPUs,
 *          slightly faster than the native Tiger Lake implementation, but returns only one mask.
 */
NK_INTERNAL nk_u32_t nk_intersect_u16x32_ice_(__m512i a, __m512i b) {
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

    return ~(nk_u32_t)(nm70 & nk_u32_rol(nm71, 8) & nk_u32_rol(nm72, 16) & nk_u32_ror(nm73, 8));
}

/**
 *  @brief  Analogous to `_mm512_2intersect_epi32`, but compatible with Ice Lake CPUs,
 *          slightly faster than the native Tiger Lake implementation, but returns only one mask.
 */
NK_INTERNAL nk_u16_t nk_intersect_u32x16_ice_(__m512i a, __m512i b) {
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

    return ~(nk_u16_t)(nm0 & nk_u16_rol(nm1, 4) & nk_u16_rol(nm2, 8) & nk_u16_ror(nm3, 4));
}

NK_PUBLIC void nk_sparse_intersect_u16_ice( //
    nk_u16_t const *a, nk_u16_t const *b,   //
    nk_size_t a_length, nk_size_t b_length, //
    nk_u16_t *result, nk_size_t *count) {

    // The baseline implementation for very small arrays (2 registers or less) can be quite simple:
    if (a_length < 64 && b_length < 64) {
        nk_sparse_intersect_u16_serial(a, b, a_length, b_length, result, count);
        return;
    }

    nk_u16_t const *const a_end = a + a_length;
    nk_u16_t const *const b_end = b + b_length;
    nk_size_t c = 0;
    nk_b512_vec_t a_vec, b_vec;

    while (a + 32 <= a_end && b + 32 <= b_end) {
        a_vec.zmm = _mm512_loadu_si512((__m512i const *)a);
        b_vec.zmm = _mm512_loadu_si512((__m512i const *)b);

        // Intersecting registers with `nk_intersect_u16x32_ice_` involves a lot of shuffling
        // and comparisons, so we want to avoid it if the slices don't overlap at all..
        nk_u16_t a_min;
        nk_u16_t a_max = a_vec.u16s[31];
        nk_u16_t b_min = b_vec.u16s[0];
        nk_u16_t b_max = b_vec.u16s[31];

        // If the slices don't overlap, advance the appropriate pointer
        while (a_max < b_min && a + 64 <= a_end) {
            a += 32;
            a_vec.zmm = _mm512_loadu_si512((__m512i const *)a);
            a_max = a_vec.u16s[31];
        }
        a_min = a_vec.u16s[0];
        while (b_max < a_min && b + 64 <= b_end) {
            b += 32;
            b_vec.zmm = _mm512_loadu_si512((__m512i const *)b);
            b_max = b_vec.u16s[31];
        }
        b_min = b_vec.u16s[0];

        __m512i a_max_u16x32 = _mm512_set1_epi16(*(short const *)&a_max);
        __m512i b_max_u16x32 = _mm512_set1_epi16(*(short const *)&b_max);
        __mmask32 a_step_mask = _mm512_cmple_epu16_mask(a_vec.zmm, b_max_u16x32);
        __mmask32 b_step_mask = _mm512_cmple_epu16_mask(b_vec.zmm, a_max_u16x32);
        a += 32 - _lzcnt_u32((nk_u32_t)a_step_mask);
        b += 32 - _lzcnt_u32((nk_u32_t)b_step_mask);

        // Now we are likely to have some overlap, so we can intersect the registers
        __mmask32 a_matches = nk_intersect_u16x32_ice_(a_vec.zmm, b_vec.zmm);

        // Export matches if result buffer is provided
        if (result) { _mm512_mask_compressstoreu_epi16(result + c, a_matches, a_vec.zmm); }
        c += _mm_popcnt_u32(a_matches); // MSVC has no `_popcnt32`
    }

    nk_size_t tail_count = 0;
    nk_sparse_intersect_u16_serial(a, b, a_end - a, b_end - b, result ? result + c : NULL, &tail_count);
    *count = c + tail_count;
}

NK_PUBLIC void nk_sparse_intersect_u32_ice( //
    nk_u32_t const *a, nk_u32_t const *b,   //
    nk_size_t a_length, nk_size_t b_length, //
    nk_u32_t *result, nk_size_t *count) {

    // The baseline implementation for very small arrays (2 registers or less) can be quite simple:
    if (a_length < 32 && b_length < 32) {
        nk_sparse_intersect_u32_serial(a, b, a_length, b_length, result, count);
        return;
    }

    nk_u32_t const *const a_end = a + a_length;
    nk_u32_t const *const b_end = b + b_length;
    nk_size_t c = 0;
    nk_b512_vec_t a_vec, b_vec;

    while (a + 16 <= a_end && b + 16 <= b_end) {
        a_vec.zmm = _mm512_loadu_si512((__m512i const *)a);
        b_vec.zmm = _mm512_loadu_si512((__m512i const *)b);

        // Intersecting registers with `nk_intersect_u32x16_ice_` involves a lot of shuffling
        // and comparisons, so we want to avoid it if the slices don't overlap at all..
        nk_u32_t a_min;
        nk_u32_t a_max = a_vec.u32s[15];
        nk_u32_t b_min = b_vec.u32s[0];
        nk_u32_t b_max = b_vec.u32s[15];

        // If the slices don't overlap, advance the appropriate pointer
        while (a_max < b_min && a + 32 <= a_end) {
            a += 16;
            a_vec.zmm = _mm512_loadu_si512((__m512i const *)a);
            a_max = a_vec.u32s[15];
        }
        a_min = a_vec.u32s[0];
        while (b_max < a_min && b + 32 <= b_end) {
            b += 16;
            b_vec.zmm = _mm512_loadu_si512((__m512i const *)b);
            b_max = b_vec.u32s[15];
        }
        b_min = b_vec.u32s[0];

        __m512i a_max_u32x16 = _mm512_set1_epi32(*(int const *)&a_max);
        __m512i b_max_u32x16 = _mm512_set1_epi32(*(int const *)&b_max);
        __mmask16 a_step_mask = _mm512_cmple_epu32_mask(a_vec.zmm, b_max_u32x16);
        __mmask16 b_step_mask = _mm512_cmple_epu32_mask(b_vec.zmm, a_max_u32x16);
        a += 32 - _lzcnt_u32((nk_u32_t)a_step_mask);
        b += 32 - _lzcnt_u32((nk_u32_t)b_step_mask);

        // Now we are likely to have some overlap, so we can intersect the registers
        __mmask16 a_matches = nk_intersect_u32x16_ice_(a_vec.zmm, b_vec.zmm);

        // Export matches if result buffer is provided
        if (result) { _mm512_mask_compressstoreu_epi32(result + c, a_matches, a_vec.zmm); }
        c += _mm_popcnt_u32(a_matches); // MSVC has no `_popcnt32`
    }

    nk_size_t tail_count = 0;
    nk_sparse_intersect_u32_serial(a, b, a_end - a, b_end - b, result ? result + c : NULL, &tail_count);
    *count = c + tail_count;
}

/**
 *  @brief  Analogous to `_mm512_2intersect_epi64`, but compatible with Ice Lake CPUs,
 *          returns only one mask indicating which elements in `a` have a match in `b`.
 */
NK_INTERNAL nk_u8_t nk_intersect_u64x8_ice_(__m512i a, __m512i b) {
    __m512i a1 = _mm512_alignr_epi64(a, a, 2);
    __m512i b1 = _mm512_shuffle_i64x2(b, b, _MM_SHUFFLE(2, 1, 0, 3));
    __mmask8 nm00 = _mm512_cmpneq_epi64_mask(a, b);

    __m512i a2 = _mm512_alignr_epi64(a, a, 4);
    __m512i a3 = _mm512_alignr_epi64(a, a, 6);
    __mmask8 nm01 = _mm512_cmpneq_epi64_mask(a1, b);
    __mmask8 nm02 = _mm512_cmpneq_epi64_mask(a2, b);

    __mmask8 nm03 = _mm512_cmpneq_epi64_mask(a3, b);
    __mmask8 nm10 = _mm512_mask_cmpneq_epi64_mask(nm00, a, b1);
    __mmask8 nm11 = _mm512_mask_cmpneq_epi64_mask(nm01, a1, b1);

    __m512i b2 = _mm512_shuffle_i64x2(b, b, _MM_SHUFFLE(1, 0, 3, 2));
    __mmask8 nm12 = _mm512_mask_cmpneq_epi64_mask(nm02, a2, b1);
    __mmask8 nm13 = _mm512_mask_cmpneq_epi64_mask(nm03, a3, b1);
    __mmask8 nm20 = _mm512_mask_cmpneq_epi64_mask(nm10, a, b2);

    __m512i b3 = _mm512_shuffle_i64x2(b, b, _MM_SHUFFLE(0, 3, 2, 1));
    __mmask8 nm21 = _mm512_mask_cmpneq_epi64_mask(nm11, a1, b2);
    __mmask8 nm22 = _mm512_mask_cmpneq_epi64_mask(nm12, a2, b2);
    __mmask8 nm23 = _mm512_mask_cmpneq_epi64_mask(nm13, a3, b2);

    __mmask8 nm0 = _mm512_mask_cmpneq_epi64_mask(nm20, a, b3);
    __mmask8 nm1 = _mm512_mask_cmpneq_epi64_mask(nm21, a1, b3);
    __mmask8 nm2 = _mm512_mask_cmpneq_epi64_mask(nm22, a2, b3);
    __mmask8 nm3 = _mm512_mask_cmpneq_epi64_mask(nm23, a3, b3);

    return ~(nk_u8_t)(nm0 & nk_u8_rol(nm1, 2) & nk_u8_rol(nm2, 4) & nk_u8_ror(nm3, 2));
}

NK_PUBLIC void nk_sparse_intersect_u64_ice( //
    nk_u64_t const *a, nk_u64_t const *b,   //
    nk_size_t a_length, nk_size_t b_length, //
    nk_u64_t *result, nk_size_t *count) {

    // The baseline implementation for very small arrays (2 registers or less) can be quite simple:
    if (a_length < 16 && b_length < 16) {
        nk_sparse_intersect_u64_serial(a, b, a_length, b_length, result, count);
        return;
    }

    nk_u64_t const *const a_end = a + a_length;
    nk_u64_t const *const b_end = b + b_length;
    nk_size_t c = 0;
    nk_b512_vec_t a_vec, b_vec;

    while (a + 8 <= a_end && b + 8 <= b_end) {
        a_vec.zmm = _mm512_loadu_si512((__m512i const *)a);
        b_vec.zmm = _mm512_loadu_si512((__m512i const *)b);

        // Intersecting registers with `nk_intersect_u64x8_ice_` involves a lot of shuffling
        // and comparisons, so we want to avoid it if the slices don't overlap at all.
        nk_u64_t a_min;
        nk_u64_t a_max = a_vec.u64s[7];
        nk_u64_t b_min = b_vec.u64s[0];
        nk_u64_t b_max = b_vec.u64s[7];

        // If the slices don't overlap, advance the appropriate pointer
        while (a_max < b_min && a + 16 <= a_end) {
            a += 8;
            a_vec.zmm = _mm512_loadu_si512((__m512i const *)a);
            a_max = a_vec.u64s[7];
        }
        a_min = a_vec.u64s[0];
        while (b_max < a_min && b + 16 <= b_end) {
            b += 8;
            b_vec.zmm = _mm512_loadu_si512((__m512i const *)b);
            b_max = b_vec.u64s[7];
        }
        b_min = b_vec.u64s[0];

        __m512i a_max_u64x8 = _mm512_set1_epi64(*(long long const *)&a_max);
        __m512i b_max_u64x8 = _mm512_set1_epi64(*(long long const *)&b_max);
        __mmask8 a_step_mask = _mm512_cmple_epu64_mask(a_vec.zmm, b_max_u64x8);
        __mmask8 b_step_mask = _mm512_cmple_epu64_mask(b_vec.zmm, a_max_u64x8);
        a += 16 - _lzcnt_u32((nk_u32_t)a_step_mask | 0x100);
        b += 16 - _lzcnt_u32((nk_u32_t)b_step_mask | 0x100);

        // Now we are likely to have some overlap, so we can intersect the registers
        __mmask8 a_matches = nk_intersect_u64x8_ice_(a_vec.zmm, b_vec.zmm);

        // Export matches if result buffer is provided
        if (result) { _mm512_mask_compressstoreu_epi64(result + c, a_matches, a_vec.zmm); }
        c += _mm_popcnt_u32(a_matches); // MSVC has no `_popcnt32`
    }

    nk_size_t tail_count = 0;
    nk_sparse_intersect_u64_serial(a, b, a_end - a, b_end - b, result ? result + c : NULL, &tail_count);
    *count = c + tail_count;
}

NK_PUBLIC void nk_sparse_dot_u32f32_ice(                  //
    nk_u32_t const *a, nk_u32_t const *b,                 //
    nk_f32_t const *a_weights, nk_f32_t const *b_weights, //
    nk_size_t a_length, nk_size_t b_length, nk_f32_t *product) {

    // The baseline implementation for very small arrays (2 registers or less) can be quite simple:
    if (a_length < 32 && b_length < 32) {
        nk_sparse_dot_u32f32_serial(a, b, a_weights, b_weights, a_length, b_length, product);
        return;
    }

    nk_u32_t const *const a_end = a + a_length;
    nk_u32_t const *const b_end = b + b_length;
    __m512 product_f32x16 = _mm512_setzero_ps();
    nk_b512_vec_t a_vec, b_vec;

    while (a + 16 <= a_end && b + 16 <= b_end) {
        a_vec.zmm = _mm512_loadu_si512((__m512i const *)a);
        b_vec.zmm = _mm512_loadu_si512((__m512i const *)b);

        // Intersecting registers with `nk_intersect_u32x16_ice_` involves a lot of shuffling
        // and comparisons, so we want to avoid it if the slices don't overlap at all.
        nk_u32_t a_min;
        nk_u32_t a_max = a_vec.u32s[15];
        nk_u32_t b_min = b_vec.u32s[0];
        nk_u32_t b_max = b_vec.u32s[15];

        // If the slices don't overlap, advance the appropriate pointer
        while (a_max < b_min && a + 32 <= a_end) {
            a += 16;
            a_weights += 16;
            a_vec.zmm = _mm512_loadu_si512((__m512i const *)a);
            a_max = a_vec.u32s[15];
        }
        a_min = a_vec.u32s[0];
        while (b_max < a_min && b + 32 <= b_end) {
            b += 16;
            b_weights += 16;
            b_vec.zmm = _mm512_loadu_si512((__m512i const *)b);
            b_max = b_vec.u32s[15];
        }
        b_min = b_vec.u32s[0];

        __m512i a_max_u32x16 = _mm512_set1_epi32(*(int const *)&a_max);
        __m512i b_max_u32x16 = _mm512_set1_epi32(*(int const *)&b_max);
        __mmask16 a_step_mask = _mm512_cmple_epu32_mask(a_vec.zmm, b_max_u32x16);
        __mmask16 b_step_mask = _mm512_cmple_epu32_mask(b_vec.zmm, a_max_u32x16);
        a += 32 - _lzcnt_u32((nk_u32_t)a_step_mask);
        a_weights += 32 - _lzcnt_u32((nk_u32_t)a_step_mask);
        b += 32 - _lzcnt_u32((nk_u32_t)b_step_mask);
        b_weights += 32 - _lzcnt_u32((nk_u32_t)b_step_mask);

        // Now we are likely to have some overlap, so we can intersect the registers
        __mmask16 a_matches = nk_intersect_u32x16_ice_(a_vec.zmm, b_vec.zmm);
        __mmask16 b_matches = nk_intersect_u32x16_ice_(b_vec.zmm, a_vec.zmm);
        if (a_matches == 0) continue;

        // Load and compress matching weights
        __m512 a_weights_f32x16 = _mm512_loadu_ps(a_weights - (32 - _lzcnt_u32((nk_u32_t)a_step_mask)));
        __m512 b_weights_f32x16 = _mm512_loadu_ps(b_weights - (32 - _lzcnt_u32((nk_u32_t)b_step_mask)));
        __m512 a_matched_f32x16 = _mm512_maskz_compress_ps(a_matches, a_weights_f32x16);
        __m512 b_matched_f32x16 = _mm512_maskz_compress_ps(b_matches, b_weights_f32x16);

        // FMA accumulation
        product_f32x16 = _mm512_fmadd_ps(a_matched_f32x16, b_matched_f32x16, product_f32x16);
    }

    nk_f32_t tail_product = 0;
    nk_sparse_dot_u32f32_serial(a, b, a_weights, b_weights, a_end - a, b_end - b, &tail_product);
    *product = _mm512_reduce_add_ps(product_f32x16) + tail_product;
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // NK_TARGET_ICE

#if NK_TARGET_TURIN
#if defined(__clang__)
#pragma clang attribute push(                                                                                                    \
    __attribute__((target(                                                                                                       \
        "avx2,avx512f,avx512vl,bmi,bmi2,lzcnt,popcnt,avx512bw,avx512vbmi2,avx512bf16,avx512vnni,avx512vp2intersect,avx512dq"))), \
    apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "bmi", "bmi2", "lzcnt", "popcnt", "avx512bw", "avx512vbmi2", \
                   "avx512bf16", "avx512vnni", "avx512vp2intersect", "avx512dq")
#endif

NK_PUBLIC void nk_sparse_intersect_u16_turin( //
    nk_u16_t const *a, nk_u16_t const *b,     //
    nk_size_t a_length, nk_size_t b_length,   //
    nk_u16_t *result, nk_size_t *count) {

    //! There is no such thing as `_mm512_2intersect_epi16`, only the 32-bit variant!
    //! So instead of jumping through 32 entries at a time, like on Ice Lake, we will
    //! step through 16 entries at a time.
    nk_u16_t const *const a_end = a + a_length;
    nk_u16_t const *const b_end = b + b_length;
    nk_size_t c = 0;
    nk_b256_vec_t a_vec, b_vec;

    // Broadcast index for last element (hoisted outside loop)
    __m256i const last_idx = _mm256_set1_epi16(15);
    while (a + 16 <= a_end && b + 16 <= b_end) {
        a_vec.ymm = _mm256_loadu_si256((__m256i const *)a);
        b_vec.ymm = _mm256_loadu_si256((__m256i const *)b);

        // Intersect the registers
        __m512i a_i32x16 = _mm512_cvtepu16_epi32(a_vec.ymm);
        __m512i b_i32x16 = _mm512_cvtepu16_epi32(b_vec.ymm);
        __mmask16 a_matches_any_in_b, b_matches_any_in_a;
        _mm512_2intersect_epi32(a_i32x16, b_i32x16, &a_matches_any_in_b, &b_matches_any_in_a);

        // Export matches if result buffer is provided
        if (result) { _mm256_mask_compressstoreu_epi16(result + c, a_matches_any_in_b, a_vec.ymm); }
        c += _mm_popcnt_u32(a_matches_any_in_b); // MSVC has no `_popcnt32`

        __m256i a_max_u16x16 = _mm256_permutexvar_epi16(last_idx, a_vec.ymm);
        __m256i b_max_u16x16 = _mm256_permutexvar_epi16(last_idx, b_vec.ymm);
        __mmask16 a_step_mask = _mm256_cmple_epu16_mask(a_vec.ymm, b_max_u16x16);
        __mmask16 b_step_mask = _mm256_cmple_epu16_mask(b_vec.ymm, a_max_u16x16);
        a += _tzcnt_u32(~(nk_u32_t)a_step_mask | 0x10000);
        b += _tzcnt_u32(~(nk_u32_t)b_step_mask | 0x10000);
    }

    nk_size_t tail_count = 0;
    nk_sparse_intersect_u16_serial(a, b, a_end - a, b_end - b, result ? result + c : NULL, &tail_count);
    *count = c + tail_count;
}

NK_PUBLIC void nk_sparse_intersect_u32_turin( //
    nk_u32_t const *a, nk_u32_t const *b,     //
    nk_size_t a_length, nk_size_t b_length,   //
    nk_u32_t *result, nk_size_t *count) {

    nk_u32_t const *const a_end = a + a_length;
    nk_u32_t const *const b_end = b + b_length;
    nk_size_t c = 0;
    nk_b512_vec_t a_vec, b_vec;

    // Broadcast index for last element (hoisted outside loop)
    __m512i const last_idx = _mm512_set1_epi32(15);
    while (a + 16 <= a_end && b + 16 <= b_end) {
        a_vec.zmm = _mm512_loadu_si512((__m512i const *)a);
        b_vec.zmm = _mm512_loadu_si512((__m512i const *)b);

        // Intersect the registers
        __mmask16 a_matches_any_in_b, b_matches_any_in_a;
        _mm512_2intersect_epi32(a_vec.zmm, b_vec.zmm, &a_matches_any_in_b, &b_matches_any_in_a);

        // Export matches if result buffer is provided
        if (result) { _mm512_mask_compressstoreu_epi32(result + c, a_matches_any_in_b, a_vec.zmm); }
        c += _mm_popcnt_u32(a_matches_any_in_b); // MSVC has no `_popcnt32`

        // Pure SIMD broadcasts - no scalar extraction needed
        __m512i a_max_u32x16 = _mm512_permutexvar_epi32(last_idx, a_vec.zmm);
        __m512i b_max_u32x16 = _mm512_permutexvar_epi32(last_idx, b_vec.zmm);
        __mmask16 a_step_mask = _mm512_cmple_epu32_mask(a_vec.zmm, b_max_u32x16);
        __mmask16 b_step_mask = _mm512_cmple_epu32_mask(b_vec.zmm, a_max_u32x16);
        a += _tzcnt_u32(~(nk_u32_t)a_step_mask | 0x10000);
        b += _tzcnt_u32(~(nk_u32_t)b_step_mask | 0x10000);
    }

    nk_size_t tail_count = 0;
    nk_sparse_intersect_u32_serial(a, b, a_end - a, b_end - b, result ? result + c : NULL, &tail_count);
    *count = c + tail_count;
}

NK_PUBLIC void nk_sparse_intersect_u64_turin( //
    nk_u64_t const *a, nk_u64_t const *b,     //
    nk_size_t a_length, nk_size_t b_length,   //
    nk_u64_t *result, nk_size_t *count) {

    nk_u64_t const *const a_end = a + a_length;
    nk_u64_t const *const b_end = b + b_length;
    nk_size_t c = 0;
    nk_b512_vec_t a_vec, b_vec;

    // Broadcast index for last element (hoisted outside loop)
    __m512i const last_idx = _mm512_set1_epi64(7);
    while (a + 8 <= a_end && b + 8 <= b_end) {
        a_vec.zmm = _mm512_loadu_si512((__m512i const *)a);
        b_vec.zmm = _mm512_loadu_si512((__m512i const *)b);

        // Intersect the registers
        __mmask8 a_matches_any_in_b, b_matches_any_in_a;
        _mm512_2intersect_epi64(a_vec.zmm, b_vec.zmm, &a_matches_any_in_b, &b_matches_any_in_a);

        // Export matches if result buffer is provided
        if (result) { _mm512_mask_compressstoreu_epi64(result + c, a_matches_any_in_b, a_vec.zmm); }
        c += _mm_popcnt_u32(a_matches_any_in_b); // MSVC has no `_popcnt32`

        // Pure SIMD broadcasts - no scalar extraction needed
        __m512i a_max_u64x8 = _mm512_permutexvar_epi64(last_idx, a_vec.zmm);
        __m512i b_max_u64x8 = _mm512_permutexvar_epi64(last_idx, b_vec.zmm);
        __mmask8 a_step_mask = _mm512_cmple_epu64_mask(a_vec.zmm, b_max_u64x8);
        __mmask8 b_step_mask = _mm512_cmple_epu64_mask(b_vec.zmm, a_max_u64x8);
        a += _tzcnt_u32(~(nk_u32_t)a_step_mask | 0x100);
        b += _tzcnt_u32(~(nk_u32_t)b_step_mask | 0x100);
    }

    nk_size_t tail_count = 0;
    nk_sparse_intersect_u64_serial(a, b, a_end - a, b_end - b, result ? result + c : NULL, &tail_count);
    *count = c + tail_count;
}

NK_PUBLIC void nk_sparse_dot_u16bf16_turin(                 //
    nk_u16_t const *a, nk_u16_t const *b,                   //
    nk_bf16_t const *a_weights, nk_bf16_t const *b_weights, //
    nk_size_t a_length, nk_size_t b_length,                 //
    nk_f32_t *product) {

    // The baseline implementation for very small arrays (2 registers or less) can be quite simple:
    if (a_length < 64 && b_length < 64) {
        nk_sparse_dot_u16bf16_serial(a, b, a_weights, b_weights, a_length, b_length, product);
        return;
    }

    //! There is no such thing as `_mm512_2intersect_epi16`, only the 32-bit variant!
    //! So instead of jumping through 32 entries at a time, like on Ice Lake, we will
    //! step through 16 entries at a time.
    nk_u16_t const *const a_end = a + a_length;
    nk_u16_t const *const b_end = b + b_length;
    nk_b256_vec_t a_vec, b_vec;
    __m256 product_f32x8 = _mm256_setzero_ps();

    // Broadcast index for last element (hoisted outside loop)
    __m256i const last_idx = _mm256_set1_epi16(15);
    while (a + 16 <= a_end && b + 16 <= b_end) {
        a_vec.ymm = _mm256_loadu_si256((__m256i const *)a);
        b_vec.ymm = _mm256_loadu_si256((__m256i const *)b);

        // Intersecting registers with `_mm512_2intersect_epi16_mask` involves a lot of shuffling
        // and comparisons, so we want to avoid it if the slices don't overlap at all..
        nk_u16_t a_min;
        nk_u16_t a_max = a_vec.u16s[15];
        nk_u16_t b_min = b_vec.u16s[0];
        nk_u16_t b_max = b_vec.u16s[15];

        // If the slices don't overlap, advance the appropriate pointer
        while (a_max < b_min && a + 32 <= a_end) {
            a += 16, a_weights += 16;
            a_vec.ymm = _mm256_loadu_si256((__m256i const *)a);
            a_max = a_vec.u16s[15];
        }
        a_min = a_vec.u16s[0];
        while (b_max < a_min && b + 32 <= b_end) {
            b += 16, b_weights += 16;
            b_vec.ymm = _mm256_loadu_si256((__m256i const *)b);
            b_max = b_vec.u16s[15];
        }
        b_min = b_vec.u16s[0];

        // Now we are likely to have some overlap, so we can intersect the registers
        __m512i a_i32x16 = _mm512_cvtepu16_epi32(a_vec.ymm);
        __m512i b_i32x16 = _mm512_cvtepu16_epi32(b_vec.ymm);
        __mmask16 a_matches_any_in_b, b_matches_any_in_a;
        _mm512_2intersect_epi32(a_i32x16, b_i32x16, &a_matches_any_in_b, &b_matches_any_in_a);

        // Load and shift all the relevant weights to the start of the vector before doing the dot product
        if (a_matches_any_in_b) {
            __m256i a_weights_bf16x16 = _mm256_loadu_si256((__m256i const *)a_weights);
            a_weights_bf16x16 = _mm256_maskz_compress_epi16(a_matches_any_in_b, a_weights_bf16x16);
            __m256i b_weights_bf16x16 = _mm256_loadu_si256((__m256i const *)b_weights);
            b_weights_bf16x16 = _mm256_maskz_compress_epi16(b_matches_any_in_a, b_weights_bf16x16);
            product_f32x8 = _mm256_dpbf16_ps(product_f32x8, (__m256bh)a_weights_bf16x16, (__m256bh)b_weights_bf16x16);
        }

        __m256i a_max_u16x16 = _mm256_permutexvar_epi16(last_idx, a_vec.ymm);
        __m256i b_max_u16x16 = _mm256_permutexvar_epi16(last_idx, b_vec.ymm);
        __mmask16 a_step_mask = _mm256_cmple_epu16_mask(a_vec.ymm, b_max_u16x16);
        __mmask16 b_step_mask = _mm256_cmple_epu16_mask(b_vec.ymm, a_max_u16x16);
        nk_size_t a_step = _tzcnt_u32(~(nk_u32_t)a_step_mask | 0x10000);
        nk_size_t b_step = _tzcnt_u32(~(nk_u32_t)b_step_mask | 0x10000);
        a += a_step, a_weights += a_step;
        b += b_step, b_weights += b_step;
    }
    nk_f32_t tail_product = 0;
    nk_sparse_dot_u16bf16_serial(a, b, a_weights, b_weights, a_end - a, b_end - b, &tail_product);
    *product = tail_product + _mm512_reduce_add_ps(_mm512_insertf32x8(_mm512_setzero_ps(), product_f32x8, 0));
}

NK_PUBLIC void nk_sparse_dot_u32f32_turin(                //
    nk_u32_t const *a, nk_u32_t const *b,                 //
    nk_f32_t const *a_weights, nk_f32_t const *b_weights, //
    nk_size_t a_length, nk_size_t b_length,               //
    nk_f32_t *product) {

    // The baseline implementation for very small arrays (2 registers or less) can be quite simple:
    if (a_length < 32 && b_length < 32) {
        nk_sparse_dot_u32f32_serial(a, b, a_weights, b_weights, a_length, b_length, product);
        return;
    }

    // Native VP2INTERSECTD works directly on u32 - no conversion needed!
    nk_u32_t const *const a_end = a + a_length;
    nk_u32_t const *const b_end = b + b_length;
    __m512 product_f32x16 = _mm512_setzero_ps();
    nk_b512_vec_t a_vec, b_vec;

    while (a + 16 <= a_end && b + 16 <= b_end) {
        a_vec.zmm = _mm512_loadu_si512((__m512i const *)a);
        b_vec.zmm = _mm512_loadu_si512((__m512i const *)b);

        // Avoid expensive intersection if slices don't overlap at all
        nk_u32_t a_min;
        nk_u32_t a_max = a_vec.u32s[15];
        nk_u32_t b_min = b_vec.u32s[0];
        nk_u32_t b_max = b_vec.u32s[15];

        // If the slices don't overlap, advance the appropriate pointer
        while (a_max < b_min && a + 32 <= a_end) {
            a += 16, a_weights += 16;
            a_vec.zmm = _mm512_loadu_si512((__m512i const *)a);
            a_max = a_vec.u32s[15];
        }
        a_min = a_vec.u32s[0];
        while (b_max < a_min && b + 32 <= b_end) {
            b += 16, b_weights += 16;
            b_vec.zmm = _mm512_loadu_si512((__m512i const *)b);
            b_max = b_vec.u32s[15];
        }
        b_min = b_vec.u32s[0];

        // Native u32 intersection - no conversion needed!
        __mmask16 a_matches, b_matches;
        _mm512_2intersect_epi32(a_vec.zmm, b_vec.zmm, &a_matches, &b_matches);

        // Load and compress matching weights, then FMA
        if (a_matches) {
            __m512 a_weights_f32x16 = _mm512_loadu_ps(a_weights);
            __m512 b_weights_f32x16 = _mm512_loadu_ps(b_weights);
            __m512 a_matched_f32x16 = _mm512_maskz_compress_ps(a_matches, a_weights_f32x16);
            __m512 b_matched_f32x16 = _mm512_maskz_compress_ps(b_matches, b_weights_f32x16);
            product_f32x16 = _mm512_fmadd_ps(a_matched_f32x16, b_matched_f32x16, product_f32x16);
        }

        __m512i a_max_u32x16 = _mm512_set1_epi32(*(int const *)&a_max);
        __m512i b_max_u32x16 = _mm512_set1_epi32(*(int const *)&b_max);
        __mmask16 a_step_mask = _mm512_cmple_epu32_mask(a_vec.zmm, b_max_u32x16);
        __mmask16 b_step_mask = _mm512_cmple_epu32_mask(b_vec.zmm, a_max_u32x16);
        nk_size_t a_step = _tzcnt_u32(~(nk_u32_t)a_step_mask | 0x10000);
        nk_size_t b_step = _tzcnt_u32(~(nk_u32_t)b_step_mask | 0x10000);
        a += a_step, a_weights += a_step;
        b += b_step, b_weights += b_step;
    }

    nk_f32_t tail_product = 0;
    nk_sparse_dot_u32f32_serial(a, b, a_weights, b_weights, a_end - a, b_end - b, &tail_product);
    *product = _mm512_reduce_add_ps(product_f32x16) + tail_product;
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // NK_TARGET_TURIN
#endif // NK_TARGET_X86_

#if NK_TARGET_ARM_
#if NK_TARGET_NEON
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8-a"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8-a")
#endif

/**
 *  @brief  Uses `vshrn` to produce a bitmask, similar to `movemask` in SSE.
 *  https://community.arm.com/arm-community-blogs/b/infrastructure-solutions-blog/posts/porting-x86-vector-bitmask-optimizations-to-arm-neon
 */
NK_INTERNAL nk_u64_t nk_u8_to_u4_neon_(uint8x16_t vec) {
    return vget_lane_u64(vreinterpret_u64_u8(vshrn_n_u16(vreinterpretq_u16_u8(vec), 4)), 0);
}

NK_INTERNAL int nk_clz_u64_(nk_u64_t x) {
// On GCC and Clang use the builtin, otherwise use the generic implementation
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_clzll(x);
#else
    int n = 0;
    while ((x & 0x8000000000000000ull) == 0) n++, x <<= 1;
    return n;
#endif
}

NK_INTERNAL uint32x4_t nk_intersect_u32x4_neon_(uint32x4_t a, uint32x4_t b) {
    uint32x4_t b1 = vextq_u32(b, b, 1);
    uint32x4_t b2 = vextq_u32(b, b, 2);
    uint32x4_t b3 = vextq_u32(b, b, 3);
    uint32x4_t nm00 = vceqq_u32(a, b);
    uint32x4_t nm01 = vceqq_u32(a, b1);
    uint32x4_t nm02 = vceqq_u32(a, b2);
    uint32x4_t nm03 = vceqq_u32(a, b3);
    uint32x4_t nm = vorrq_u32(vorrq_u32(nm00, nm01), vorrq_u32(nm02, nm03));
    return nm;
}

NK_INTERNAL uint16x8_t nk_intersect_u16x8_neon_(uint16x8_t a, uint16x8_t b) {
    uint16x8_t b1 = vextq_u16(b, b, 1);
    uint16x8_t b2 = vextq_u16(b, b, 2);
    uint16x8_t b3 = vextq_u16(b, b, 3);
    uint16x8_t b4 = vextq_u16(b, b, 4);
    uint16x8_t b5 = vextq_u16(b, b, 5);
    uint16x8_t b6 = vextq_u16(b, b, 6);
    uint16x8_t b7 = vextq_u16(b, b, 7);
    uint16x8_t nm00 = vceqq_u16(a, b);
    uint16x8_t nm01 = vceqq_u16(a, b1);
    uint16x8_t nm02 = vceqq_u16(a, b2);
    uint16x8_t nm03 = vceqq_u16(a, b3);
    uint16x8_t nm04 = vceqq_u16(a, b4);
    uint16x8_t nm05 = vceqq_u16(a, b5);
    uint16x8_t nm06 = vceqq_u16(a, b6);
    uint16x8_t nm07 = vceqq_u16(a, b7);
    uint16x8_t nm = vorrq_u16(vorrq_u16(vorrq_u16(nm00, nm01), vorrq_u16(nm02, nm03)),
                              vorrq_u16(vorrq_u16(nm04, nm05), vorrq_u16(nm06, nm07)));
    return nm;
}

NK_PUBLIC void nk_sparse_intersect_u16_neon( //
    nk_u16_t const *a, nk_u16_t const *b,    //
    nk_size_t a_length, nk_size_t b_length,  //
    nk_u16_t *result, nk_size_t *count) {

    // NEON lacks compress-store, so fall back to serial for result output
    if (result) {
        nk_sparse_intersect_u16_serial(a, b, a_length, b_length, result, count);
        return;
    }

    // The baseline implementation for very small arrays (2 registers or less) can be quite simple:
    if (a_length < 32 && b_length < 32) {
        nk_sparse_intersect_u16_serial(a, b, a_length, b_length, result, count);
        return;
    }

    nk_u16_t const *const a_end = a + a_length;
    nk_u16_t const *const b_end = b + b_length;
    nk_b128_vec_t a_vec, b_vec;
    uint16x8_t c_counts_u16x8 = vdupq_n_u16(0);

    while (a + 8 <= a_end && b + 8 <= b_end) {
        a_vec.u16x8 = vld1q_u16(a);
        b_vec.u16x8 = vld1q_u16(b);

        // Intersecting registers with `nk_intersect_u16x8_neon_` involves a lot of shuffling
        // and comparisons, so we want to avoid it if the slices don't overlap at all..
        nk_u16_t a_min;
        nk_u16_t a_max = a_vec.u16s[7];
        nk_u16_t b_min = b_vec.u16s[0];
        nk_u16_t b_max = b_vec.u16s[7];

        // If the slices don't overlap, advance the appropriate pointer
        while (a_max < b_min && a + 16 <= a_end) {
            a += 8;
            a_vec.u16x8 = vld1q_u16(a);
            a_max = a_vec.u16s[7];
        }
        a_min = a_vec.u16s[0];
        while (b_max < a_min && b + 16 <= b_end) {
            b += 8;
            b_vec.u16x8 = vld1q_u16(b);
            b_max = b_vec.u16s[7];
        }
        b_min = b_vec.u16s[0];

        // Now we are likely to have some overlap, so we can intersect the registers.
        // We can do it by performing a population count at every cycle, but it's not the cheapest in terms of cycles.
        //
        //      nk_u64_t a_matches = __builtin_popcountll(
        //          nk_u8_to_u4_neon_(vreinterpretq_u8_u16(
        //              nk_intersect_u16x8_neon_(a_vec.u16x8, b_vec.u16x8))));
        //      c += a_matches / 8;
        //
        // Alternatively, we can we can transform match-masks into "ones", accumulate them between the cycles,
        // and merge all together in the end.
        uint16x8_t a_matches = nk_intersect_u16x8_neon_(a_vec.u16x8, b_vec.u16x8);
        c_counts_u16x8 = vaddq_u16(c_counts_u16x8, vandq_u16(a_matches, vdupq_n_u16(1)));

        // Counting leading zeros is tricky. On Arm we can use inline Assembly to get the result,
        // but MSVC doesn't support that:
        //
        //      NK_INTERNAL int nk_clz_u64_(nk_u64_t value) {
        //          nk_u64_t result;
        //          __asm__("clz %x0, %x1" : "=r"(result) : "r"(value));
        //          return (int)result;
        //      }
        //
        // Alternatively, we can use the `vclz_u32` NEON intrinsic.
        // It will compute the leading zeros number for both `a_step` and `b_step` in parallel.
        uint16x8_t a_max_u16x8 = vdupq_n_u16(a_max);
        uint16x8_t b_max_u16x8 = vdupq_n_u16(b_max);
        nk_u64_t a_step = nk_clz_u64_(nk_u8_to_u4_neon_( //
            vreinterpretq_u8_u16(vcleq_u16(a_vec.u16x8, b_max_u16x8))));
        nk_u64_t b_step = nk_clz_u64_(nk_u8_to_u4_neon_( //
            vreinterpretq_u8_u16(vcleq_u16(b_vec.u16x8, a_max_u16x8))));
        a += (64 - a_step) / 8;
        b += (64 - b_step) / 8;
    }

    nk_size_t tail_count = 0;
    nk_sparse_intersect_u16_serial(a, b, a_end - a, b_end - b, NULL, &tail_count);
    *count = tail_count + (nk_size_t)vaddvq_u16(c_counts_u16x8);
}

NK_PUBLIC void nk_sparse_intersect_u32_neon( //
    nk_u32_t const *a, nk_u32_t const *b,    //
    nk_size_t a_length, nk_size_t b_length,  //
    nk_u32_t *result, nk_size_t *count) {

    // NEON lacks compress-store, so fall back to serial for result output
    if (result) {
        nk_sparse_intersect_u32_serial(a, b, a_length, b_length, result, count);
        return;
    }

    // The baseline implementation for very small arrays (2 registers or less) can be quite simple:
    if (a_length < 32 && b_length < 32) {
        nk_sparse_intersect_u32_serial(a, b, a_length, b_length, result, count);
        return;
    }

    nk_u32_t const *const a_end = a + a_length;
    nk_u32_t const *const b_end = b + b_length;
    nk_b128_vec_t a_vec, b_vec;
    uint32x4_t c_counts_u32x4 = vdupq_n_u32(0);

    while (a + 4 <= a_end && b + 4 <= b_end) {
        a_vec.u32x4 = vld1q_u32(a);
        b_vec.u32x4 = vld1q_u32(b);

        // Intersecting registers with `nk_intersect_u32x4_neon_` involves a lot of shuffling
        // and comparisons, so we want to avoid it if the slices don't overlap at all..
        nk_u32_t a_min;
        nk_u32_t a_max = a_vec.u32s[3];
        nk_u32_t b_min = b_vec.u32s[0];
        nk_u32_t b_max = b_vec.u32s[3];

        // If the slices don't overlap, advance the appropriate pointer
        while (a_max < b_min && a + 8 <= a_end) {
            a += 4;
            a_vec.u32x4 = vld1q_u32(a);
            a_max = a_vec.u32s[3];
        }
        a_min = a_vec.u32s[0];
        while (b_max < a_min && b + 8 <= b_end) {
            b += 4;
            b_vec.u32x4 = vld1q_u32(b);
            b_max = b_vec.u32s[3];
        }
        b_min = b_vec.u32s[0];

        // Now we are likely to have some overlap, so we can intersect the registers
        // We can do it by performing a population count at every cycle, but it's not the cheapest in terms of cycles.
        //
        //     nk_u64_t a_matches = __builtin_popcountll(
        //         nk_u8_to_u4_neon_(vreinterpretq_u8_u32(
        //             nk_intersect_u32x4_neon_(a_vec.u32x4, b_vec.u32x4))));
        //     c += a_matches / 16;
        //
        // Alternatively, we can we can transform match-masks into "ones", accumulate them between the cycles,
        // and merge all together in the end.
        uint32x4_t a_matches = nk_intersect_u32x4_neon_(a_vec.u32x4, b_vec.u32x4);
        c_counts_u32x4 = vaddq_u32(c_counts_u32x4, vandq_u32(a_matches, vdupq_n_u32(1)));

        uint32x4_t a_max_u32x4 = vdupq_n_u32(a_max);
        uint32x4_t b_max_u32x4 = vdupq_n_u32(b_max);
        nk_u64_t a_step = nk_clz_u64_(nk_u8_to_u4_neon_( //
            vreinterpretq_u8_u32(vcleq_u32(a_vec.u32x4, b_max_u32x4))));
        nk_u64_t b_step = nk_clz_u64_(nk_u8_to_u4_neon_( //
            vreinterpretq_u8_u32(vcleq_u32(b_vec.u32x4, a_max_u32x4))));
        a += (64 - a_step) / 16;
        b += (64 - b_step) / 16;
    }

    nk_size_t tail_count = 0;
    nk_sparse_intersect_u32_serial(a, b, a_end - a, b_end - b, NULL, &tail_count);
    *count = tail_count + (nk_size_t)vaddvq_u32(c_counts_u32x4);
}

NK_INTERNAL uint64x2_t nk_intersect_u64x2_neon_(uint64x2_t a, uint64x2_t b) {
    uint64x2_t b1 = vextq_u64(b, b, 1);
    uint64x2_t nm00 = vceqq_u64(a, b);
    uint64x2_t nm01 = vceqq_u64(a, b1);
    uint64x2_t nm = vorrq_u64(nm00, nm01);
    return nm;
}

NK_PUBLIC void nk_sparse_intersect_u64_neon( //
    nk_u64_t const *a, nk_u64_t const *b,    //
    nk_size_t a_length, nk_size_t b_length,  //
    nk_u64_t *result, nk_size_t *count) {

    // NEON lacks compress-store, so fall back to serial for result output
    if (result) {
        nk_sparse_intersect_u64_serial(a, b, a_length, b_length, result, count);
        return;
    }

    // The baseline implementation for very small arrays (2 registers or less) can be quite simple:
    if (a_length < 8 && b_length < 8) {
        nk_sparse_intersect_u64_serial(a, b, a_length, b_length, result, count);
        return;
    }

    nk_u64_t const *const a_end = a + a_length;
    nk_u64_t const *const b_end = b + b_length;
    nk_b128_vec_t a_vec, b_vec;
    uint64x2_t c_counts_u64x2 = vdupq_n_u64(0);

    while (a + 2 <= a_end && b + 2 <= b_end) {
        a_vec.u64x2 = vld1q_u64(a);
        b_vec.u64x2 = vld1q_u64(b);

        // Intersecting registers with `nk_intersect_u64x2_neon_` involves comparisons,
        // so we want to avoid it if the slices don't overlap at all.
        nk_u64_t a_min;
        nk_u64_t a_max = a_vec.u64s[1];
        nk_u64_t b_min = b_vec.u64s[0];
        nk_u64_t b_max = b_vec.u64s[1];

        // If the slices don't overlap, advance the appropriate pointer
        while (a_max < b_min && a + 4 <= a_end) {
            a += 2;
            a_vec.u64x2 = vld1q_u64(a);
            a_max = a_vec.u64s[1];
        }
        a_min = a_vec.u64s[0];
        while (b_max < a_min && b + 4 <= b_end) {
            b += 2;
            b_vec.u64x2 = vld1q_u64(b);
            b_max = b_vec.u64s[1];
        }
        b_min = b_vec.u64s[0];

        // Now we are likely to have some overlap, so we can intersect the registers
        // Transform match-masks into "ones", accumulate them between the cycles,
        // and merge all together in the end.
        uint64x2_t a_matches = nk_intersect_u64x2_neon_(a_vec.u64x2, b_vec.u64x2);
        c_counts_u64x2 = vaddq_u64(c_counts_u64x2, vandq_u64(a_matches, vdupq_n_u64(1)));

        uint64x2_t a_max_u64x2 = vdupq_n_u64(a_max);
        uint64x2_t b_max_u64x2 = vdupq_n_u64(b_max);
        nk_u64_t a_step = nk_clz_u64_(nk_u8_to_u4_neon_( //
            vreinterpretq_u8_u64(vcleq_u64(a_vec.u64x2, b_max_u64x2))));
        nk_u64_t b_step = nk_clz_u64_(nk_u8_to_u4_neon_( //
            vreinterpretq_u8_u64(vcleq_u64(b_vec.u64x2, a_max_u64x2))));
        a += (64 - a_step) / 32;
        b += (64 - b_step) / 32;
    }

    nk_size_t tail_count = 0;
    nk_sparse_intersect_u64_serial(a, b, a_end - a, b_end - b, NULL, &tail_count);
    *count = tail_count + (nk_size_t)vaddvq_u64(c_counts_u64x2);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // NK_TARGET_NEON

/*  SVE2 introduces many new integer-oriented instructions, extending some of the NEON functionality
 *  to variable-length SVE registers. Those include "compare multiple" intrinsics:
 *
 *  - `svmatch[_u16]` that matches each scalar in first vector against all members of a 128-bit lane in the second.
 *  - `svhistcnt[_s32]_z` does something similar, performing an inclusive prefix scan.
 *  - `svtbx[_u16]` does extended table lookup
 *
 *  Other notable instructions:
 *
 *  - `DUP`: Broadcast indexed predicate element
 *    https://developer.arm.com/documentation/ddi0602/2021-06/SVE-Instructions/DUP--predicate---Broadcast-indexed-predicate-element-?lang=en
 *  - `SCLAMP` and `UCLAMP`: clamp values, i.e. combined min+max
 *    https://developer.arm.com/documentation/ddi0602/2021-06/SVE-Instructions/SCLAMP--Signed-clamp-to-minimum-maximum-vector-?lang=en
 *    https://developer.arm.com/documentation/ddi0602/2021-06/SVE-Instructions/UCLAMP--Unsigned-clamp-to-minimum-maximum-vector-?lang=en
 *  - `TBLQ`: Table lookup quadword
 *    https://developer.arm.com/documentation/ddi0602/2022-12/SVE-Instructions/TBLQ--Programmable-table-lookup-within-each-quadword-vector-segment--zeroing--?lang=en
 *
 *  Great resources for SVE2 intrinsics:
 *
 *  > ARM’s Scalable Vector Extensions: A Critical Look at SVE2 For Integer Workloads
 *    https://gist.github.com/zingaburga/805669eb891c820bd220418ee3f0d6bd
 */
#if NK_TARGET_SVE2
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+sve+sve2"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+sve+sve2")
#endif

NK_PUBLIC void nk_sparse_intersect_u16_sve2( //
    nk_u16_t const *a, nk_u16_t const *b,    //
    nk_size_t a_length, nk_size_t b_length,  //
    nk_u16_t *result, nk_size_t *count) {

    // A single SVE lane is 128 bits wide, so one lane fits 8 values.
    nk_size_t const register_size = svcnth();
    nk_size_t const lanes_count = register_size / 8;
    nk_size_t a_idx = 0, b_idx = 0;
    nk_size_t c = 0;

    while (a_idx < a_length && b_idx < b_length) {
        // Load `a_member` and broadcast it, load `b_members_vec` from memory
        svbool_t a_progress = svwhilelt_b16_u64(a_idx, a_length);
        svbool_t b_progress = svwhilelt_b16_u64(b_idx, b_length);
        svuint16_t a_vec = svld1_u16(a_progress, a + a_idx);
        svuint16_t b_vec = svld1_u16(b_progress, b + b_idx);

        // Intersecting registers with `svmatch_u16` involves a lot of shuffling
        // and comparisons, so we want to avoid it if the slices don't overlap at all..
        nk_u16_t a_min;
        nk_u16_t a_max = svlastb(a_progress, a_vec);
        nk_u16_t b_min = svlasta(svpfalse_b(), b_vec);
        nk_u16_t b_max = svlastb(b_progress, b_vec);

        // If the slices don't overlap, advance the appropriate pointer
        while (a_max < b_min && (a_idx + register_size) <= a_length) {
            a_idx += register_size;
            a_progress = svwhilelt_b16_u64(a_idx, a_length);
            a_vec = svld1_u16(a_progress, a + a_idx);
            a_max = svlastb(a_progress, a_vec);
        }
        a_min = svlasta(svpfalse_b(), a_vec);
        while (b_max < a_min && (b_idx + register_size) <= b_length) {
            b_idx += register_size;
            b_progress = svwhilelt_b16_u64(b_idx, b_length);
            b_vec = svld1_u16(b_progress, b + b_idx);
            b_max = svlastb(b_progress, b_vec);
        }
        b_min = svlasta(svpfalse_b(), b_vec);

        // Before we evaluate the intersection size, obfurscating the order in `b_vec`,
        // let's estimate how much we will need to advance the pointers afterwards.
        // For that, we don't even need to broadcast the values in SVE, as the whole
        // register can be compared against a scalar:
        //
        //      svuint16_t a_last_broadcasted =  svdup_n_u16(a_max);
        //      svuint16_t b_last_broadcasted =  svdup_n_u16(b_max);
        svbool_t a_mask = svcmple_n_u16(a_progress, a_vec, b_max);
        svbool_t b_mask = svcmple_n_u16(b_progress, b_vec, a_max);
        nk_u64_t a_step = svcntp_b16(a_progress, a_mask);
        nk_u64_t b_step = svcntp_b16(b_progress, b_mask);

        // Compare `a_vec` with each lane of `b_vec`
        svbool_t equal_mask = svmatch_u16(a_progress, a_vec, b_vec);
        for (nk_size_t i = 1; i < lanes_count; i++) {
            b_vec = svext_u16(b_vec, b_vec, 8);
            equal_mask = svorr_z(svptrue_b16(), equal_mask, svmatch_u16(a_progress, a_vec, b_vec));
        }
        nk_size_t equal_count = svcntp_b16(svptrue_b16(), equal_mask);

        // Use SVE2 svcompact to compress matching elements and store to result buffer
        if (result) {
            svuint16_t compacted = svcompact_u16(equal_mask, a_vec);
            svbool_t store_predicate = svwhilelt_b16_u64(0, equal_count);
            svst1_u16(store_predicate, result + c, compacted);
        }

        // Advance
        a_idx += a_step;
        b_idx += b_step;
        c += equal_count;
    }
    *count = c;
}

NK_PUBLIC void nk_sparse_intersect_u32_sve2( //
    nk_u32_t const *a, nk_u32_t const *b,    //
    nk_size_t a_length, nk_size_t b_length,  //
    nk_u32_t *result, nk_size_t *count) {

    // A single SVE lane is 128 bits wide, so one lane fits 4 values.
    nk_size_t const register_size = svcntw();
    nk_size_t const lanes_count = register_size / 4;
    nk_size_t a_idx = 0, b_idx = 0;
    nk_size_t c = 0;

    while (a_idx < a_length && b_idx < b_length) {
        // Load `a_member` and broadcast it, load `b_members_vec` from memory
        svbool_t a_progress = svwhilelt_b32_u64(a_idx, a_length);
        svbool_t b_progress = svwhilelt_b32_u64(b_idx, b_length);
        svuint32_t a_vec = svld1_u32(a_progress, a + a_idx);
        svuint32_t b_vec = svld1_u32(b_progress, b + b_idx);

        // Intersecting registers with `svmatch_u16` involves a lot of shuffling
        // and comparisons, so we want to avoid it if the slices don't overlap at all..
        nk_u32_t a_min;
        nk_u32_t a_max = svlastb(a_progress, a_vec);
        nk_u32_t b_min = svlasta(svpfalse_b(), b_vec);
        nk_u32_t b_max = svlastb(b_progress, b_vec);

        // If the slices don't overlap, advance the appropriate pointer
        while (a_max < b_min && (a_idx + register_size) <= a_length) {
            a_idx += register_size;
            a_progress = svwhilelt_b32_u64(a_idx, a_length);
            a_vec = svld1_u32(a_progress, a + a_idx);
            a_max = svlastb(a_progress, a_vec);
        }
        a_min = svlasta(svpfalse_b(), a_vec);
        while (b_max < a_min && (b_idx + register_size) <= b_length) {
            b_idx += register_size;
            b_progress = svwhilelt_b32_u64(b_idx, b_length);
            b_vec = svld1_u32(b_progress, b + b_idx);
            b_max = svlastb(b_progress, b_vec);
        }
        b_min = svlasta(svpfalse_b(), b_vec);

        // Before we evaluate the intersection size, obfurscating the order in `b_vec`,
        // let's estimate how much we will need to advance the pointers afterwards.
        // For that, we don't even need to broadcast the values in SVE, as the whole
        // register can be compared against a scalar:
        //
        //      svuint32_t a_last_broadcasted =  svdup_n_u32(a_max);
        //      svuint32_t b_last_broadcasted =  svdup_n_u32(b_max);
        svbool_t a_mask = svcmple_n_u32(a_progress, a_vec, b_max);
        svbool_t b_mask = svcmple_n_u32(b_progress, b_vec, a_max);
        nk_u64_t a_step = svcntp_b32(a_progress, a_mask);
        nk_u64_t b_step = svcntp_b32(b_progress, b_mask);

        // Comparing `a_vec` with each lane of `b_vec` can't be done with `svmatch`,
        // the same way as in `nk_sparse_intersect_u16_sve2`, as that instruction is only
        // available for 8-bit and 16-bit integers.
        //
        //      svbool_t equal_mask = svpfalse_b();
        //      for (nk_size_t i = 0; i < register_size; i++) {
        //          equal_mask = svorr_z(svptrue_b32(), equal_mask, svcmpeq_u32(a_progress, a_vec, b_vec));
        //          b_vec = svext_u32(b_vec, b_vec, 1);
        //      }
        //      nk_size_t equal_count = svcntp_b32(a_progress, equal_mask);
        //
        // Alternatively, one can use histogram instructions, like `svhistcnt_u32_z`.
        // They practically compute the prefix-matching count, which is equivalent to
        // the lower triangle of the row-major intersection matrix.
        // To compute the upper triangle, we can reverse (with `svrev_b32`) the order of
        // elements and repeat the operation, accumulating the results for top and bottom.
        // Let's look at 4x element registers as an example:
        //
        //      ⊐ α = {A, B, C, D}, β = {X, Y, Z, W}:
        //
        //      hist(α, β):           hist(α_rev, β_rev):
        //
        //        X Y Z W               W Z Y X
        //      A 1 0 0 0             D 1 0 0 0
        //      B 1 1 0 0             C 1 1 0 0
        //      C 1 1 1 0             B 1 1 1 0
        //      D 1 1 1 1             A 1 1 1 1
        //
        svuint32_t hist_lower = svhistcnt_u32_z(a_progress, a_vec, b_vec);
        svuint32_t a_rev_vec = svrev_u32(a_vec);
        svuint32_t b_rev_vec = svrev_u32(b_vec);
        svuint32_t hist_upper = svrev_u32(svhistcnt_u32_z(svptrue_b32(), a_rev_vec, b_rev_vec));
        svuint32_t hist = svorr_u32_x(a_progress, hist_lower, hist_upper);
        svbool_t equal_mask = svcmpne_n_u32(a_progress, hist, 0);
        nk_size_t equal_count = svcntp_b32(a_progress, equal_mask);

        // Use SVE2 svcompact to compress matching elements and store to result buffer
        if (result) {
            svuint32_t compacted = svcompact_u32(equal_mask, a_vec);
            svbool_t store_predicate = svwhilelt_b32_u64(0, equal_count);
            svst1_u32(store_predicate, result + c, compacted);
        }

        // Advance
        a_idx += a_step;
        b_idx += b_step;
        c += equal_count;
    }
    *count = c;
}

NK_PUBLIC void nk_sparse_intersect_u64_sve2( //
    nk_u64_t const *a, nk_u64_t const *b,    //
    nk_size_t a_length, nk_size_t b_length,  //
    nk_u64_t *result, nk_size_t *count) {

    // A single SVE lane is 128 bits wide, so one lane fits 2 values.
    nk_size_t const register_size = svcntd();
    nk_size_t const lanes_count = register_size / 2;
    nk_size_t a_idx = 0, b_idx = 0;
    nk_size_t c = 0;

    while (a_idx < a_length && b_idx < b_length) {
        // Load `a_member` and broadcast it, load `b_members_vec` from memory
        svbool_t a_progress = svwhilelt_b64_u64(a_idx, a_length);
        svbool_t b_progress = svwhilelt_b64_u64(b_idx, b_length);
        svuint64_t a_vec = svld1_u64(a_progress, a + a_idx);
        svuint64_t b_vec = svld1_u64(b_progress, b + b_idx);

        // Intersecting registers involves comparisons,
        // so we want to avoid it if the slices don't overlap at all.
        nk_u64_t a_min;
        nk_u64_t a_max = svlastb(a_progress, a_vec);
        nk_u64_t b_min = svlasta(svpfalse_b(), b_vec);
        nk_u64_t b_max = svlastb(b_progress, b_vec);

        // If the slices don't overlap, advance the appropriate pointer
        while (a_max < b_min && (a_idx + register_size) <= a_length) {
            a_idx += register_size;
            a_progress = svwhilelt_b64_u64(a_idx, a_length);
            a_vec = svld1_u64(a_progress, a + a_idx);
            a_max = svlastb(a_progress, a_vec);
        }
        a_min = svlasta(svpfalse_b(), a_vec);
        while (b_max < a_min && (b_idx + register_size) <= b_length) {
            b_idx += register_size;
            b_progress = svwhilelt_b64_u64(b_idx, b_length);
            b_vec = svld1_u64(b_progress, b + b_idx);
            b_max = svlastb(b_progress, b_vec);
        }
        b_min = svlasta(svpfalse_b(), b_vec);

        // Estimate how much we will need to advance the pointers afterwards.
        svbool_t a_mask = svcmple_n_u64(a_progress, a_vec, b_max);
        svbool_t b_mask = svcmple_n_u64(b_progress, b_vec, a_max);
        nk_u64_t a_step = svcntp_b64(a_progress, a_mask);
        nk_u64_t b_step = svcntp_b64(b_progress, b_mask);

        // Use histogram instructions like `svhistcnt_u64_z` to compute intersection.
        // They compute the prefix-matching count, equivalent to the lower triangle
        // of the row-major intersection matrix.
        svuint64_t hist_lower = svhistcnt_u64_z(a_progress, a_vec, b_vec);
        svuint64_t a_rev_vec = svrev_u64(a_vec);
        svuint64_t b_rev_vec = svrev_u64(b_vec);
        svuint64_t hist_upper = svrev_u64(svhistcnt_u64_z(svptrue_b64(), a_rev_vec, b_rev_vec));
        svuint64_t hist = svorr_u64_x(a_progress, hist_lower, hist_upper);
        svbool_t equal_mask = svcmpne_n_u64(a_progress, hist, 0);
        nk_size_t equal_count = svcntp_b64(a_progress, equal_mask);

        // Use SVE2 svcompact to compress matching elements and store to result buffer
        if (result) {
            svuint64_t compacted = svcompact_u64(equal_mask, a_vec);
            svbool_t store_predicate = svwhilelt_b64_u64(0, equal_count);
            svst1_u64(store_predicate, result + c, compacted);
        }

        // Advance
        a_idx += a_step;
        b_idx += b_step;
        c += equal_count;
    }
    *count = c;
}

NK_PUBLIC void nk_sparse_dot_u32f32_sve2(                 //
    nk_u32_t const *a, nk_u32_t const *b,                 //
    nk_f32_t const *a_weights, nk_f32_t const *b_weights, //
    nk_size_t a_length, nk_size_t b_length,               //
    nk_f32_t *product) {

    // A single SVE lane is 128 bits wide, so one lane fits 4 values.
    nk_size_t const register_size = svcntw();
    nk_size_t const lanes_count = register_size / 4;
    nk_size_t a_idx = 0, b_idx = 0;
    svfloat32_t product_f32_sve = svdup_f32(0.f);

    while (a_idx < a_length && b_idx < b_length) {
        // Load indices with progress predicates
        svbool_t a_progress = svwhilelt_b32_u64(a_idx, a_length);
        svbool_t b_progress = svwhilelt_b32_u64(b_idx, b_length);
        svuint32_t a_u32_sve = svld1_u32(a_progress, a + a_idx);
        svuint32_t b_u32_sve = svld1_u32(b_progress, b + b_idx);

        // Avoid expensive intersection if slices don't overlap at all
        nk_u32_t a_min;
        nk_u32_t a_max = svlastb(a_progress, a_u32_sve);
        nk_u32_t b_min = svlasta(svpfalse_b(), b_u32_sve);
        nk_u32_t b_max = svlastb(b_progress, b_u32_sve);

        // If the slices don't overlap, advance the appropriate pointer
        while (a_max < b_min && (a_idx + register_size) <= a_length) {
            a_idx += register_size;
            a_progress = svwhilelt_b32_u64(a_idx, a_length);
            a_u32_sve = svld1_u32(a_progress, a + a_idx);
            a_max = svlastb(a_progress, a_u32_sve);
        }
        a_min = svlasta(svpfalse_b(), a_u32_sve);
        while (b_max < a_min && (b_idx + register_size) <= b_length) {
            b_idx += register_size;
            b_progress = svwhilelt_b32_u64(b_idx, b_length);
            b_u32_sve = svld1_u32(b_progress, b + b_idx);
            b_max = svlastb(b_progress, b_u32_sve);
        }
        b_min = svlasta(svpfalse_b(), b_u32_sve);

        // Calculate step sizes before modifying vectors
        svbool_t a_mask = svcmple_n_u32(a_progress, a_u32_sve, b_max);
        svbool_t b_mask = svcmple_n_u32(b_progress, b_u32_sve, a_max);
        nk_u64_t a_step = svcntp_b32(a_progress, a_mask);
        nk_u64_t b_step = svcntp_b32(b_progress, b_mask);

        // Use histogram-based intersection (svmatch_u32 doesn't exist)
        svuint32_t hist_lower = svhistcnt_u32_z(a_progress, a_u32_sve, b_u32_sve);
        svuint32_t a_rev_u32_sve = svrev_u32(a_u32_sve);
        svuint32_t b_rev_u32_sve = svrev_u32(b_u32_sve);
        svuint32_t hist_upper = svrev_u32(svhistcnt_u32_z(svptrue_b32(), a_rev_u32_sve, b_rev_u32_sve));
        svuint32_t hist = svorr_u32_x(a_progress, hist_lower, hist_upper);
        svbool_t a_equal_mask = svcmpne_n_u32(a_progress, hist, 0);

        // Load weights and mask by intersection
        svfloat32_t a_weights_f32_sve = svld1_f32(a_progress, a_weights + a_idx);
        svfloat32_t b_weights_f32_sve = svld1_f32(b_progress, b_weights + b_idx);

        // For each position in a that matches something in b, we need the corresponding b weight.
        // Use lane-by-lane matching for dot product.
        for (nk_size_t i = 0; i < lanes_count; i++) {
            // Check which elements of a match the current rotation of b
            svbool_t equal_lane = svcmpeq_u32(a_progress, a_u32_sve, b_u32_sve);
            // Multiply matching weights and accumulate
            svfloat32_t b_equal_weights = svsel_f32(equal_lane, b_weights_f32_sve, svdup_f32(0.f));
            product_f32_sve = svmla_f32_x(a_progress, product_f32_sve, a_weights_f32_sve, b_equal_weights);
            // Rotate b vectors
            b_u32_sve = svext_u32(b_u32_sve, b_u32_sve, 4);
            b_weights_f32_sve = svext_f32(b_weights_f32_sve, b_weights_f32_sve, 4);
        }

        // Advance
        a_idx += a_step;
        b_idx += b_step;
    }
    *product = svaddv_f32(svptrue_b32(), product_f32_sve);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // NK_TARGET_SVE2

#if NK_TARGET_SVE2 && NK_TARGET_SVEBFDOT
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8.6-a+sve+sve2+bf16"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8.6-a+sve+sve2+bf16")
#endif

NK_PUBLIC void nk_sparse_dot_u16bf16_sve2(                  //
    nk_u16_t const *a, nk_u16_t const *b,                   //
    nk_bf16_t const *a_weights, nk_bf16_t const *b_weights, //
    nk_size_t a_length, nk_size_t b_length,                 //
    nk_f32_t *product) {

    // A single SVE lane is 128 bits wide, so one lane fits 8 values.
    nk_size_t const register_size = svcnth();
    nk_size_t const lanes_count = register_size / 8;
    nk_size_t a_idx = 0, b_idx = 0;
    svfloat32_t product_vec = svdupq_n_f32(0.f, 0.f, 0.f, 0.f);

    while (a_idx < a_length && b_idx < b_length) {
        // Load `a_member` and broadcast it, load `b_members_vec` from memory
        svbool_t a_progress = svwhilelt_b16_u64(a_idx, a_length);
        svbool_t b_progress = svwhilelt_b16_u64(b_idx, b_length);
        svuint16_t a_vec = svld1_u16(a_progress, a + a_idx);
        svuint16_t b_vec = svld1_u16(b_progress, b + b_idx);

        // Intersecting registers with `svmatch_u16` involves a lot of shuffling
        // and comparisons, so we want to avoid it if the slices don't overlap at all..
        nk_u16_t a_min;
        nk_u16_t a_max = svlastb(a_progress, a_vec);
        nk_u16_t b_min = svlasta(svpfalse_b(), b_vec);
        nk_u16_t b_max = svlastb(b_progress, b_vec);

        // If the slices don't overlap, advance the appropriate pointer
        while (a_max < b_min && (a_idx + register_size) <= a_length) {
            a_idx += register_size;
            a_progress = svwhilelt_b16_u64(a_idx, a_length);
            a_vec = svld1_u16(a_progress, a + a_idx);
            a_max = svlastb(a_progress, a_vec);
        }
        a_min = svlasta(svpfalse_b(), a_vec);
        while (b_max < a_min && (b_idx + register_size) <= b_length) {
            b_idx += register_size;
            b_progress = svwhilelt_b16_u64(b_idx, b_length);
            b_vec = svld1_u16(b_progress, b + b_idx);
            b_max = svlastb(b_progress, b_vec);
        }
        b_min = svlasta(svpfalse_b(), b_vec);

        // Before we evaluate the intersection size, obfurscating the order in `b_vec`,
        // let's estimate how much we will need to advance the pointers afterwards.
        // For that, we don't even need to broadcast the values in SVE, as the whole
        // register can be compared against a scalar:
        //
        //      svuint16_t a_last_broadcasted =  svdup_n_u16(a_max);
        //      svuint16_t b_last_broadcasted =  svdup_n_u16(b_max);
        svbool_t a_mask = svcmple_n_u16(a_progress, a_vec, b_max);
        svbool_t b_mask = svcmple_n_u16(b_progress, b_vec, a_max);
        nk_u64_t a_step = svcntp_b16(a_progress, a_mask);
        nk_u64_t b_step = svcntp_b16(b_progress, b_mask);

        // Compare `a_vec` with each lane of `b_vec`
        svbfloat16_t a_weights_vec = svld1_bf16(a_progress, (__bf16 const *)a_weights + a_idx);
        svbfloat16_t b_weights_vec = svld1_bf16(b_progress, (__bf16 const *)b_weights + b_idx);
        for (nk_size_t i = 0; i < lanes_count; i++) {
            svbool_t equal_mask = svmatch_u16(a_progress, a_vec, b_vec);
            //! The `svsel_bf16` intrinsic is broken in many compilers, not returning the correct type.
            //! So we reinterprete floats as integers and apply `svsel_s16`, but the `svreinterpret_s16_bs16`
            //! and `svreinterpret_bf16_s16` are not always properly defined!
            svint16_t b_equal_weights_vec = svsel_s16(equal_mask, svreinterpret_s16_bf16(b_weights_vec),
                                                      svdup_n_s16(0));
            product_vec = svbfdot_f32(product_vec, a_weights_vec, svreinterpret_bf16_s16(b_equal_weights_vec));
            b_vec = svext_u16(b_vec, b_vec, 8);
        }

        // Advance
        a_idx += a_step;
        b_idx += b_step;
    }
    *product = svaddv_f32(svptrue_b32(), product_vec);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // NK_TARGET_SVE2 && NK_TARGET_SVEBFDOT
#endif // NK_TARGET_ARM_

#if !NK_DYNAMIC_DISPATCH

NK_PUBLIC void nk_sparse_intersect_u16(nk_u16_t const *a, nk_u16_t const *b, nk_size_t a_length, nk_size_t b_length,
                                       nk_u16_t *result, nk_size_t *count) {
#if NK_TARGET_SVE2
    nk_sparse_intersect_u16_sve2(a, b, a_length, b_length, result, count);
#elif NK_TARGET_NEON
    nk_sparse_intersect_u16_neon(a, b, a_length, b_length, result, count);
#elif NK_TARGET_TURIN
    nk_sparse_intersect_u16_turin(a, b, a_length, b_length, result, count);
#elif NK_TARGET_ICE
    nk_sparse_intersect_u16_ice(a, b, a_length, b_length, result, count);
#else
    nk_sparse_intersect_u16_serial(a, b, a_length, b_length, result, count);
#endif
}

NK_PUBLIC void nk_sparse_intersect_u32(nk_u32_t const *a, nk_u32_t const *b, nk_size_t a_length, nk_size_t b_length,
                                       nk_u32_t *result, nk_size_t *count) {
#if NK_TARGET_SVE2
    nk_sparse_intersect_u32_sve2(a, b, a_length, b_length, result, count);
#elif NK_TARGET_NEON
    nk_sparse_intersect_u32_neon(a, b, a_length, b_length, result, count);
#elif NK_TARGET_TURIN
    nk_sparse_intersect_u32_turin(a, b, a_length, b_length, result, count);
#elif NK_TARGET_ICE
    nk_sparse_intersect_u32_ice(a, b, a_length, b_length, result, count);
#else
    nk_sparse_intersect_u32_serial(a, b, a_length, b_length, result, count);
#endif
}

NK_PUBLIC void nk_sparse_intersect_u64(nk_u64_t const *a, nk_u64_t const *b, nk_size_t a_length, nk_size_t b_length,
                                       nk_u64_t *result, nk_size_t *count) {
#if NK_TARGET_SVE2
    nk_sparse_intersect_u64_sve2(a, b, a_length, b_length, result, count);
#elif NK_TARGET_NEON
    nk_sparse_intersect_u64_neon(a, b, a_length, b_length, result, count);
#elif NK_TARGET_TURIN
    nk_sparse_intersect_u64_turin(a, b, a_length, b_length, result, count);
#elif NK_TARGET_ICE
    nk_sparse_intersect_u64_ice(a, b, a_length, b_length, result, count);
#else
    nk_sparse_intersect_u64_serial(a, b, a_length, b_length, result, count);
#endif
}

NK_PUBLIC void nk_sparse_dot_u16bf16(nk_u16_t const *a, nk_u16_t const *b, nk_bf16_t const *a_weights,
                                     nk_bf16_t const *b_weights, nk_size_t a_length, nk_size_t b_length,
                                     nk_f32_t *product) {
#if NK_TARGET_SVE2 && NK_TARGET_SVEBFDOT
    nk_sparse_dot_u16bf16_sve2(a, b, a_weights, b_weights, a_length, b_length, product);
#elif NK_TARGET_TURIN
    nk_sparse_dot_u16bf16_turin(a, b, a_weights, b_weights, a_length, b_length, product);
#else
    nk_sparse_dot_u16bf16_serial(a, b, a_weights, b_weights, a_length, b_length, product);
#endif
}

NK_PUBLIC void nk_sparse_dot_u32f32(nk_u32_t const *a, nk_u32_t const *b, nk_f32_t const *a_weights,
                                    nk_f32_t const *b_weights, nk_size_t a_length, nk_size_t b_length,
                                    nk_f32_t *product) {
#if NK_TARGET_SVE2
    nk_sparse_dot_u32f32_sve2(a, b, a_weights, b_weights, a_length, b_length, product);
#elif NK_TARGET_TURIN
    nk_sparse_dot_u32f32_turin(a, b, a_weights, b_weights, a_length, b_length, product);
#elif NK_TARGET_ICE
    nk_sparse_dot_u32f32_ice(a, b, a_weights, b_weights, a_length, b_length, product);
#else
    nk_sparse_dot_u32f32_serial(a, b, a_weights, b_weights, a_length, b_length, product);
#endif
}

#endif // !NK_DYNAMIC_DISPATCH

#if defined(__cplusplus)
}
#endif

#endif

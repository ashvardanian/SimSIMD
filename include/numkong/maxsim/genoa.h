/**
 *  @brief SIMD-accelerated MaxSim (ColBERT late-interaction) for Genoa.
 *  @file include/numkong/maxsim/genoa.h
 *  @author Ash Vardanian
 *  @date February 17, 2026
 *
 *  @sa include/numkong/maxsim.h
 *
 *  Uses AVX-512 VNNI (VPDPBUSD) for coarse i8 screening and VDPBF16PS for bf16 refinement.
 *  VPDPBUSD computes 4 groups of (u8 x i8) -> i32 per 128-bit lane, processing 64 i8 pairs
 *  per ZMM register operation. Bias correction via XOR with 0x80 converts signed queries
 *  to unsigned, then subtracts 128 * sum(document_i8) after the depth loop.
 *
 *  4x4 register tiling: 4 queries x 4 documents = 16 ZMM accumulators per depth loop.
 *  Each document load is amortized across 4 VPDPBUSDs, and each query load across 4 documents.
 *
 *      Intrinsic                   Instruction     Genoa (Zen4)
 *      _mm512_dpbusd_epi32         VPDPBUSD        4cy @ p01 (512-bit)
 *      _mm512_dpbf16_ps            VDPBF16PS       6cy @ p01 (512-bit)
 */
#ifndef NK_MAXSIM_GENOA_H
#define NK_MAXSIM_GENOA_H

#if NK_TARGET_X86_
#if NK_TARGET_GENOA

#include "numkong/types.h"
#include "numkong/maxsim/serial.h" // nk_maxsim_packed_header_t, nk_maxsim_vector_metadata_t
#include "numkong/dot/genoa.h"     // nk_dot_bf16_genoa
#include "numkong/dot/haswell.h"   // nk_dot_f32_haswell

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(                                                                                   \
    __attribute__((target("avx2,avx512f,avx512vl,avx512bw,avx512dq,avx512vnni,avx512bf16,f16c,fma,bmi,bmi2"))), \
    apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "avx512bw", "avx512dq", "avx512vnni", "avx512bf16", "f16c", "fma", \
                   "bmi", "bmi2")
#endif

NK_PUBLIC nk_size_t nk_maxsim_packed_size_bf16_genoa(nk_size_t n, nk_size_t k) {
    return nk_maxsim_packed_size_(n, k, sizeof(nk_bf16_t), 64);
}

NK_PUBLIC nk_size_t nk_maxsim_packed_size_f32_genoa(nk_size_t n, nk_size_t k) {
    return nk_maxsim_packed_size_(n, k, sizeof(nk_f32_t), 64);
}

NK_PUBLIC void nk_maxsim_pack_bf16_genoa( //
    nk_bf16_t const *vectors, nk_size_t n, nk_size_t k, nk_size_t stride, void *packed) {

    nk_size_t const original_element_bytes = sizeof(nk_bf16_t);

    // Compute padded depth (depth_simd_dimensions = 64 for genoa)
    nk_size_t depth_i8_padded = nk_size_round_up_to_multiple_(k, 64);
    if ((depth_i8_padded & (depth_i8_padded - 1)) == 0 && depth_i8_padded > 0) depth_i8_padded += 64;

    // Compute region offsets
    nk_size_t const header_size = sizeof(nk_maxsim_packed_header_t);
    nk_size_t const i8_region_size = nk_size_round_up_to_multiple_(n * depth_i8_padded, 64);
    nk_size_t const metadata_region_size = nk_size_round_up_to_multiple_(n * sizeof(nk_maxsim_vector_metadata_t), 64);
    nk_size_t const original_stride = nk_size_round_up_to_multiple_(k * original_element_bytes, 64);

    nk_size_t const offset_i8_data = header_size;
    nk_size_t const offset_metadata = offset_i8_data + i8_region_size;
    nk_size_t const offset_original_data = offset_metadata + metadata_region_size;

    // Write header
    nk_maxsim_packed_header_t *header = (nk_maxsim_packed_header_t *)packed;
    header->vector_count = (nk_u32_t)n;
    header->depth_dimensions = (nk_u32_t)k;
    header->depth_i8_padded = (nk_u32_t)depth_i8_padded;
    header->original_element_bytes = (nk_u32_t)original_element_bytes;
    header->offset_i8_data = (nk_u32_t)offset_i8_data;
    header->offset_metadata = (nk_u32_t)offset_metadata;
    header->offset_original_data = (nk_u32_t)offset_original_data;
    header->original_stride_bytes = (nk_u32_t)original_stride;
    for (int r = 0; r < 8; r++) header->reserved[r] = 0;

    nk_i8_t *i8_data = (nk_i8_t *)((char *)packed + offset_i8_data);
    nk_maxsim_vector_metadata_t *metadata = (nk_maxsim_vector_metadata_t *)((char *)packed + offset_metadata);
    char *originals = (char *)packed + offset_original_data;

    for (nk_size_t vec_index = 0; vec_index < n; vec_index++) {
        nk_bf16_t const *src_row = (nk_bf16_t const *)((char const *)vectors + vec_index * stride);
        nk_i8_t *dst_i8_row = i8_data + vec_index * depth_i8_padded;

        // Pass 1: Find absmax and convert to f32
        nk_f32_t absmax_f32 = 0.0f;
        for (nk_size_t d = 0; d < k; d++) {
            nk_f32_t val_f32;
            nk_bf16_to_f32_serial(&src_row[d], &val_f32);
            nk_f32_t abs_val = nk_f32_abs_(val_f32);
            if (abs_val > absmax_f32) absmax_f32 = abs_val;
        }

        nk_f32_t scale_f32 = absmax_f32 / 127.0f;
        if (scale_f32 == 0.0f) scale_f32 = 1.0f;

        // Pass 2: Quantize to i8 and compute sum
        nk_i32_t sum_i8_i32 = 0;
        for (nk_size_t d = 0; d < k; d++) {
            nk_f32_t val_f32;
            nk_bf16_to_f32_serial(&src_row[d], &val_f32);
            nk_f32_t scaled = val_f32 / scale_f32;

            nk_i32_t rounded;
            if (scaled >= 0.0f) rounded = (nk_i32_t)(scaled + 0.5f);
            else rounded = (nk_i32_t)(scaled - 0.5f);
            if (rounded > 127) rounded = 127;
            if (rounded < -127) rounded = -127;

            dst_i8_row[d] = (nk_i8_t)rounded;
            sum_i8_i32 += rounded;
        }

        // Zero-pad remainder
        for (nk_size_t d = k; d < depth_i8_padded; d++) dst_i8_row[d] = 0;

        // Store metadata
        metadata[vec_index].scale_f32 = scale_f32;
        metadata[vec_index].sum_i8_i32 = sum_i8_i32;

        // Copy original bf16 vector to originals region
        char *dst_original = originals + vec_index * original_stride;
        nk_copy_bytes_(dst_original, src_row, k * original_element_bytes);
        for (nk_size_t b = k * original_element_bytes; b < original_stride; b++) dst_original[b] = 0;
    }
}

NK_PUBLIC void nk_maxsim_pack_f32_genoa( //
    nk_f32_t const *vectors, nk_size_t n, nk_size_t k, nk_size_t stride, void *packed) {

    nk_size_t const original_element_bytes = sizeof(nk_f32_t);

    // Compute padded depth (depth_simd_dimensions = 64 for genoa)
    nk_size_t depth_i8_padded = nk_size_round_up_to_multiple_(k, 64);
    if ((depth_i8_padded & (depth_i8_padded - 1)) == 0 && depth_i8_padded > 0) depth_i8_padded += 64;

    // Compute region offsets
    nk_size_t const header_size = sizeof(nk_maxsim_packed_header_t);
    nk_size_t const i8_region_size = nk_size_round_up_to_multiple_(n * depth_i8_padded, 64);
    nk_size_t const metadata_region_size = nk_size_round_up_to_multiple_(n * sizeof(nk_maxsim_vector_metadata_t), 64);
    nk_size_t const original_stride = nk_size_round_up_to_multiple_(k * original_element_bytes, 64);

    nk_size_t const offset_i8_data = header_size;
    nk_size_t const offset_metadata = offset_i8_data + i8_region_size;
    nk_size_t const offset_original_data = offset_metadata + metadata_region_size;

    // Write header
    nk_maxsim_packed_header_t *header = (nk_maxsim_packed_header_t *)packed;
    header->vector_count = (nk_u32_t)n;
    header->depth_dimensions = (nk_u32_t)k;
    header->depth_i8_padded = (nk_u32_t)depth_i8_padded;
    header->original_element_bytes = (nk_u32_t)original_element_bytes;
    header->offset_i8_data = (nk_u32_t)offset_i8_data;
    header->offset_metadata = (nk_u32_t)offset_metadata;
    header->offset_original_data = (nk_u32_t)offset_original_data;
    header->original_stride_bytes = (nk_u32_t)original_stride;
    for (int r = 0; r < 8; r++) header->reserved[r] = 0;

    nk_i8_t *i8_data = (nk_i8_t *)((char *)packed + offset_i8_data);
    nk_maxsim_vector_metadata_t *metadata = (nk_maxsim_vector_metadata_t *)((char *)packed + offset_metadata);
    char *originals = (char *)packed + offset_original_data;

    for (nk_size_t vec_index = 0; vec_index < n; vec_index++) {
        nk_f32_t const *src_row = (nk_f32_t const *)((char const *)vectors + vec_index * stride);
        nk_i8_t *dst_i8_row = i8_data + vec_index * depth_i8_padded;

        // Pass 1: Find absmax
        nk_f32_t absmax_f32 = 0.0f;
        for (nk_size_t d = 0; d < k; d++) {
            nk_f32_t abs_val = nk_f32_abs_(src_row[d]);
            if (abs_val > absmax_f32) absmax_f32 = abs_val;
        }

        nk_f32_t scale_f32 = absmax_f32 / 127.0f;
        if (scale_f32 == 0.0f) scale_f32 = 1.0f;

        // Pass 2: Quantize to i8 and compute sum
        nk_i32_t sum_i8_i32 = 0;
        for (nk_size_t d = 0; d < k; d++) {
            nk_f32_t scaled = src_row[d] / scale_f32;

            nk_i32_t rounded;
            if (scaled >= 0.0f) rounded = (nk_i32_t)(scaled + 0.5f);
            else rounded = (nk_i32_t)(scaled - 0.5f);
            if (rounded > 127) rounded = 127;
            if (rounded < -127) rounded = -127;

            dst_i8_row[d] = (nk_i8_t)rounded;
            sum_i8_i32 += rounded;
        }

        // Zero-pad remainder
        for (nk_size_t d = k; d < depth_i8_padded; d++) dst_i8_row[d] = 0;

        // Store metadata
        metadata[vec_index].scale_f32 = scale_f32;
        metadata[vec_index].sum_i8_i32 = sum_i8_i32;

        // Copy original f32 vector to originals region
        char *dst_original = originals + vec_index * original_stride;
        nk_copy_bytes_(dst_original, src_row, k * original_element_bytes);
        for (nk_size_t b = k * original_element_bytes; b < original_stride; b++) dst_original[b] = 0;
    }
}

/** @brief Reduces 4 ZMM i32x16 accumulators to a single __m128i with 4 horizontal sums. */
NK_INTERNAL __m128i nk_maxsim_reduce_i32x16x4_genoa_(           //
    __m512i accumulator_a_i32x16, __m512i accumulator_b_i32x16, //
    __m512i accumulator_c_i32x16, __m512i accumulator_d_i32x16) {
    // Step 1: 16 -> 8 (extract high 256-bit half and add to low half)
    __m256i sum_a_i32x8 = _mm256_add_epi32(_mm512_castsi512_si256(accumulator_a_i32x16),
                                           _mm512_extracti32x8_epi32(accumulator_a_i32x16, 1));
    __m256i sum_b_i32x8 = _mm256_add_epi32(_mm512_castsi512_si256(accumulator_b_i32x16),
                                           _mm512_extracti32x8_epi32(accumulator_b_i32x16, 1));
    __m256i sum_c_i32x8 = _mm256_add_epi32(_mm512_castsi512_si256(accumulator_c_i32x16),
                                           _mm512_extracti32x8_epi32(accumulator_c_i32x16, 1));
    __m256i sum_d_i32x8 = _mm256_add_epi32(_mm512_castsi512_si256(accumulator_d_i32x16),
                                           _mm512_extracti32x8_epi32(accumulator_d_i32x16, 1));
    // Step 2: 8 -> 4 (extract high 128-bit half and add to low half)
    __m128i sum_a_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(sum_a_i32x8), _mm256_extracti128_si256(sum_a_i32x8, 1));
    __m128i sum_b_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(sum_b_i32x8), _mm256_extracti128_si256(sum_b_i32x8, 1));
    __m128i sum_c_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(sum_c_i32x8), _mm256_extracti128_si256(sum_c_i32x8, 1));
    __m128i sum_d_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(sum_d_i32x8), _mm256_extracti128_si256(sum_d_i32x8, 1));
    // Step 3: 4x4 transpose + reduce -> [sum_a, sum_b, sum_c, sum_d]
    __m128i transpose_ab_low_i32x4 = _mm_unpacklo_epi32(sum_a_i32x4, sum_b_i32x4);
    __m128i transpose_cd_low_i32x4 = _mm_unpacklo_epi32(sum_c_i32x4, sum_d_i32x4);
    __m128i transpose_ab_high_i32x4 = _mm_unpackhi_epi32(sum_a_i32x4, sum_b_i32x4);
    __m128i transpose_cd_high_i32x4 = _mm_unpackhi_epi32(sum_c_i32x4, sum_d_i32x4);
    __m128i sum_lane_0_i32x4 = _mm_unpacklo_epi64(transpose_ab_low_i32x4, transpose_cd_low_i32x4);
    __m128i sum_lane_1_i32x4 = _mm_unpackhi_epi64(transpose_ab_low_i32x4, transpose_cd_low_i32x4);
    __m128i sum_lane_2_i32x4 = _mm_unpacklo_epi64(transpose_ab_high_i32x4, transpose_cd_high_i32x4);
    __m128i sum_lane_3_i32x4 = _mm_unpackhi_epi64(transpose_ab_high_i32x4, transpose_cd_high_i32x4);
    return _mm_add_epi32(_mm_add_epi32(sum_lane_0_i32x4, sum_lane_1_i32x4),
                         _mm_add_epi32(sum_lane_2_i32x4, sum_lane_3_i32x4));
}

NK_PUBLIC nk_f32_t nk_maxsim_packed_bf16_genoa( //
    void const *q_packed, void const *d_packed, //
    nk_size_t n_q, nk_size_t n_d, nk_size_t depth) {

    nk_maxsim_packed_header_t const *q_header = (nk_maxsim_packed_header_t const *)q_packed;
    nk_maxsim_packed_header_t const *d_header = (nk_maxsim_packed_header_t const *)d_packed;

    nk_size_t const depth_i8_padded = q_header->depth_i8_padded;

    nk_i8_t const *q_i8 = (nk_i8_t const *)((char const *)q_packed + q_header->offset_i8_data);
    nk_i8_t const *d_i8 = (nk_i8_t const *)((char const *)d_packed + d_header->offset_i8_data);
    nk_maxsim_vector_metadata_t const *d_metadata = (nk_maxsim_vector_metadata_t const *)((char const *)d_packed +
                                                                                          d_header->offset_metadata);
    char const *q_originals = (char const *)q_packed + q_header->offset_original_data;
    char const *d_originals = (char const *)d_packed + d_header->offset_original_data;
    nk_size_t const q_original_stride = q_header->original_stride_bytes;
    nk_size_t const d_original_stride = d_header->original_stride_bytes;

    __m512i const xor_mask_u8x64 = _mm512_set1_epi8((char)0x80);
    nk_f32_t total_f32 = 0.0f;

    // Primary path: 4-query grouping
    nk_size_t query_block_start_index = 0;
    for (; query_block_start_index + 4 <= n_q; query_block_start_index += 4) {
        __m128i running_max_i32x4 = _mm_set1_epi32(NK_I32_MIN);
        __m128i running_argmax_i32x4 = _mm_setzero_si128();

        // 4x4 document blocking
        nk_size_t document_block_start_index = 0;
        for (; document_block_start_index + 4 <= n_d; document_block_start_index += 4) {
            // 16 ZMM accumulators: [query_idx][doc_idx]
            __m512i accumulator_tiles_i32x16[4][4];
            for (int q = 0; q < 4; q++)
                for (int d = 0; d < 4; d++) accumulator_tiles_i32x16[q][d] = _mm512_setzero_si512();

            // Depth loop: 4 queries x 4 documents per step
            for (nk_size_t depth_index = 0; depth_index < depth_i8_padded; depth_index += 64) {
                // Load and XOR 4 query vectors (reused for all 4 documents)
                __m512i query_biased_u8x64_0 = _mm512_xor_si512(
                    _mm512_loadu_si512(
                        (__m512i const *)(q_i8 + (query_block_start_index + 0) * depth_i8_padded + depth_index)),
                    xor_mask_u8x64);
                __m512i query_biased_u8x64_1 = _mm512_xor_si512(
                    _mm512_loadu_si512(
                        (__m512i const *)(q_i8 + (query_block_start_index + 1) * depth_i8_padded + depth_index)),
                    xor_mask_u8x64);
                __m512i query_biased_u8x64_2 = _mm512_xor_si512(
                    _mm512_loadu_si512(
                        (__m512i const *)(q_i8 + (query_block_start_index + 2) * depth_i8_padded + depth_index)),
                    xor_mask_u8x64);
                __m512i query_biased_u8x64_3 = _mm512_xor_si512(
                    _mm512_loadu_si512(
                        (__m512i const *)(q_i8 + (query_block_start_index + 3) * depth_i8_padded + depth_index)),
                    xor_mask_u8x64);

                // Document 0: 4 VPDPBUSDs
                __m512i document_i8x64 = _mm512_loadu_si512(
                    (__m512i const *)(d_i8 + (document_block_start_index + 0) * depth_i8_padded + depth_index));
                accumulator_tiles_i32x16[0][0] = _mm512_dpbusd_epi32(accumulator_tiles_i32x16[0][0],
                                                                     query_biased_u8x64_0, document_i8x64);
                accumulator_tiles_i32x16[1][0] = _mm512_dpbusd_epi32(accumulator_tiles_i32x16[1][0],
                                                                     query_biased_u8x64_1, document_i8x64);
                accumulator_tiles_i32x16[2][0] = _mm512_dpbusd_epi32(accumulator_tiles_i32x16[2][0],
                                                                     query_biased_u8x64_2, document_i8x64);
                accumulator_tiles_i32x16[3][0] = _mm512_dpbusd_epi32(accumulator_tiles_i32x16[3][0],
                                                                     query_biased_u8x64_3, document_i8x64);

                // Document 1: 4 VPDPBUSDs
                document_i8x64 = _mm512_loadu_si512(
                    (__m512i const *)(d_i8 + (document_block_start_index + 1) * depth_i8_padded + depth_index));
                accumulator_tiles_i32x16[0][1] = _mm512_dpbusd_epi32(accumulator_tiles_i32x16[0][1],
                                                                     query_biased_u8x64_0, document_i8x64);
                accumulator_tiles_i32x16[1][1] = _mm512_dpbusd_epi32(accumulator_tiles_i32x16[1][1],
                                                                     query_biased_u8x64_1, document_i8x64);
                accumulator_tiles_i32x16[2][1] = _mm512_dpbusd_epi32(accumulator_tiles_i32x16[2][1],
                                                                     query_biased_u8x64_2, document_i8x64);
                accumulator_tiles_i32x16[3][1] = _mm512_dpbusd_epi32(accumulator_tiles_i32x16[3][1],
                                                                     query_biased_u8x64_3, document_i8x64);

                // Document 2: 4 VPDPBUSDs
                document_i8x64 = _mm512_loadu_si512(
                    (__m512i const *)(d_i8 + (document_block_start_index + 2) * depth_i8_padded + depth_index));
                accumulator_tiles_i32x16[0][2] = _mm512_dpbusd_epi32(accumulator_tiles_i32x16[0][2],
                                                                     query_biased_u8x64_0, document_i8x64);
                accumulator_tiles_i32x16[1][2] = _mm512_dpbusd_epi32(accumulator_tiles_i32x16[1][2],
                                                                     query_biased_u8x64_1, document_i8x64);
                accumulator_tiles_i32x16[2][2] = _mm512_dpbusd_epi32(accumulator_tiles_i32x16[2][2],
                                                                     query_biased_u8x64_2, document_i8x64);
                accumulator_tiles_i32x16[3][2] = _mm512_dpbusd_epi32(accumulator_tiles_i32x16[3][2],
                                                                     query_biased_u8x64_3, document_i8x64);

                // Document 3: 4 VPDPBUSDs
                document_i8x64 = _mm512_loadu_si512(
                    (__m512i const *)(d_i8 + (document_block_start_index + 3) * depth_i8_padded + depth_index));
                accumulator_tiles_i32x16[0][3] = _mm512_dpbusd_epi32(accumulator_tiles_i32x16[0][3],
                                                                     query_biased_u8x64_0, document_i8x64);
                accumulator_tiles_i32x16[1][3] = _mm512_dpbusd_epi32(accumulator_tiles_i32x16[1][3],
                                                                     query_biased_u8x64_1, document_i8x64);
                accumulator_tiles_i32x16[2][3] = _mm512_dpbusd_epi32(accumulator_tiles_i32x16[2][3],
                                                                     query_biased_u8x64_2, document_i8x64);
                accumulator_tiles_i32x16[3][3] = _mm512_dpbusd_epi32(accumulator_tiles_i32x16[3][3],
                                                                     query_biased_u8x64_3, document_i8x64);
            }

            // Reduce each query's 4 doc accumulators -> __m128i with [dot_d0, dot_d1, dot_d2, dot_d3]
            __m128i query_0_coarse_dots_i32x4 = nk_maxsim_reduce_i32x16x4_genoa_(
                accumulator_tiles_i32x16[0][0], accumulator_tiles_i32x16[0][1], accumulator_tiles_i32x16[0][2],
                accumulator_tiles_i32x16[0][3]);
            __m128i query_1_coarse_dots_i32x4 = nk_maxsim_reduce_i32x16x4_genoa_(
                accumulator_tiles_i32x16[1][0], accumulator_tiles_i32x16[1][1], accumulator_tiles_i32x16[1][2],
                accumulator_tiles_i32x16[1][3]);
            __m128i query_2_coarse_dots_i32x4 = nk_maxsim_reduce_i32x16x4_genoa_(
                accumulator_tiles_i32x16[2][0], accumulator_tiles_i32x16[2][1], accumulator_tiles_i32x16[2][2],
                accumulator_tiles_i32x16[2][3]);
            __m128i query_3_coarse_dots_i32x4 = nk_maxsim_reduce_i32x16x4_genoa_(
                accumulator_tiles_i32x16[3][0], accumulator_tiles_i32x16[3][1], accumulator_tiles_i32x16[3][2],
                accumulator_tiles_i32x16[3][3]);

            // Bias correct per document: subtract 128 * sum_i8 for each of 4 documents
            __m128i bias_correction_i32x4 = _mm_set_epi32(128 * d_metadata[document_block_start_index + 3].sum_i8_i32,
                                                          128 * d_metadata[document_block_start_index + 2].sum_i8_i32,
                                                          128 * d_metadata[document_block_start_index + 1].sum_i8_i32,
                                                          128 * d_metadata[document_block_start_index + 0].sum_i8_i32);
            query_0_coarse_dots_i32x4 = _mm_sub_epi32(query_0_coarse_dots_i32x4, bias_correction_i32x4);
            query_1_coarse_dots_i32x4 = _mm_sub_epi32(query_1_coarse_dots_i32x4, bias_correction_i32x4);
            query_2_coarse_dots_i32x4 = _mm_sub_epi32(query_2_coarse_dots_i32x4, bias_correction_i32x4);
            query_3_coarse_dots_i32x4 = _mm_sub_epi32(query_3_coarse_dots_i32x4, bias_correction_i32x4);

            // 4x4 transpose: [query][doc] -> [doc][query] for vectorized argmax
            __m128i transpose_queries_01_low_i32x4 = _mm_unpacklo_epi32(query_0_coarse_dots_i32x4,
                                                                        query_1_coarse_dots_i32x4);
            __m128i transpose_queries_23_low_i32x4 = _mm_unpacklo_epi32(query_2_coarse_dots_i32x4,
                                                                        query_3_coarse_dots_i32x4);
            __m128i transpose_queries_01_high_i32x4 = _mm_unpackhi_epi32(query_0_coarse_dots_i32x4,
                                                                         query_1_coarse_dots_i32x4);
            __m128i transpose_queries_23_high_i32x4 = _mm_unpackhi_epi32(query_2_coarse_dots_i32x4,
                                                                         query_3_coarse_dots_i32x4);
            // doc_N_dots_i32x4 = [q0_dot_dN, q1_dot_dN, q2_dot_dN, q3_dot_dN]
            __m128i document_0_dots_i32x4 = _mm_unpacklo_epi64(transpose_queries_01_low_i32x4,
                                                               transpose_queries_23_low_i32x4);
            __m128i document_1_dots_i32x4 = _mm_unpackhi_epi64(transpose_queries_01_low_i32x4,
                                                               transpose_queries_23_low_i32x4);
            __m128i document_2_dots_i32x4 = _mm_unpacklo_epi64(transpose_queries_01_high_i32x4,
                                                               transpose_queries_23_high_i32x4);
            __m128i document_3_dots_i32x4 = _mm_unpackhi_epi64(transpose_queries_01_high_i32x4,
                                                               transpose_queries_23_high_i32x4);

            // Branchless SIMD argmax: update running max for all 4 queries simultaneously
            __m128i comparison_mask_i32x4, document_index_i32x4;

            comparison_mask_i32x4 = _mm_cmpgt_epi32(document_0_dots_i32x4, running_max_i32x4);
            document_index_i32x4 = _mm_set1_epi32((int)(document_block_start_index + 0));
            running_max_i32x4 = _mm_blendv_epi8(running_max_i32x4, document_0_dots_i32x4, comparison_mask_i32x4);
            running_argmax_i32x4 = _mm_blendv_epi8(running_argmax_i32x4, document_index_i32x4, comparison_mask_i32x4);

            comparison_mask_i32x4 = _mm_cmpgt_epi32(document_1_dots_i32x4, running_max_i32x4);
            document_index_i32x4 = _mm_set1_epi32((int)(document_block_start_index + 1));
            running_max_i32x4 = _mm_blendv_epi8(running_max_i32x4, document_1_dots_i32x4, comparison_mask_i32x4);
            running_argmax_i32x4 = _mm_blendv_epi8(running_argmax_i32x4, document_index_i32x4, comparison_mask_i32x4);

            comparison_mask_i32x4 = _mm_cmpgt_epi32(document_2_dots_i32x4, running_max_i32x4);
            document_index_i32x4 = _mm_set1_epi32((int)(document_block_start_index + 2));
            running_max_i32x4 = _mm_blendv_epi8(running_max_i32x4, document_2_dots_i32x4, comparison_mask_i32x4);
            running_argmax_i32x4 = _mm_blendv_epi8(running_argmax_i32x4, document_index_i32x4, comparison_mask_i32x4);

            comparison_mask_i32x4 = _mm_cmpgt_epi32(document_3_dots_i32x4, running_max_i32x4);
            document_index_i32x4 = _mm_set1_epi32((int)(document_block_start_index + 3));
            running_max_i32x4 = _mm_blendv_epi8(running_max_i32x4, document_3_dots_i32x4, comparison_mask_i32x4);
            running_argmax_i32x4 = _mm_blendv_epi8(running_argmax_i32x4, document_index_i32x4, comparison_mask_i32x4);
        }

        // Document tail: 4x1 (remaining n_d % 4 documents)
        for (nk_size_t document_index = document_block_start_index; document_index < n_d; document_index++) {
            nk_i8_t const *document_i8_row = d_i8 + document_index * depth_i8_padded;

            __m512i accumulator_i32x16_0 = _mm512_setzero_si512();
            __m512i accumulator_i32x16_1 = _mm512_setzero_si512();
            __m512i accumulator_i32x16_2 = _mm512_setzero_si512();
            __m512i accumulator_i32x16_3 = _mm512_setzero_si512();

            for (nk_size_t depth_index = 0; depth_index < depth_i8_padded; depth_index += 64) {
                __m512i document_i8x64 = _mm512_loadu_si512((__m512i const *)(document_i8_row + depth_index));

                __m512i query_biased_u8x64_0 = _mm512_xor_si512(
                    _mm512_loadu_si512(
                        (__m512i const *)(q_i8 + (query_block_start_index + 0) * depth_i8_padded + depth_index)),
                    xor_mask_u8x64);
                accumulator_i32x16_0 = _mm512_dpbusd_epi32(accumulator_i32x16_0, query_biased_u8x64_0, document_i8x64);

                __m512i query_biased_u8x64_1 = _mm512_xor_si512(
                    _mm512_loadu_si512(
                        (__m512i const *)(q_i8 + (query_block_start_index + 1) * depth_i8_padded + depth_index)),
                    xor_mask_u8x64);
                accumulator_i32x16_1 = _mm512_dpbusd_epi32(accumulator_i32x16_1, query_biased_u8x64_1, document_i8x64);

                __m512i query_biased_u8x64_2 = _mm512_xor_si512(
                    _mm512_loadu_si512(
                        (__m512i const *)(q_i8 + (query_block_start_index + 2) * depth_i8_padded + depth_index)),
                    xor_mask_u8x64);
                accumulator_i32x16_2 = _mm512_dpbusd_epi32(accumulator_i32x16_2, query_biased_u8x64_2, document_i8x64);

                __m512i query_biased_u8x64_3 = _mm512_xor_si512(
                    _mm512_loadu_si512(
                        (__m512i const *)(q_i8 + (query_block_start_index + 3) * depth_i8_padded + depth_index)),
                    xor_mask_u8x64);
                accumulator_i32x16_3 = _mm512_dpbusd_epi32(accumulator_i32x16_3, query_biased_u8x64_3, document_i8x64);
            }

            // Reduce 4 accumulators to 4 scalar dots, bias correct, SIMD argmax update
            nk_i32_t bias_correction_i32 = 128 * d_metadata[document_index].sum_i8_i32;
            __m128i coarse_dots_i32x4 = _mm_set_epi32(
                _mm512_reduce_add_epi32(accumulator_i32x16_3) - bias_correction_i32,
                _mm512_reduce_add_epi32(accumulator_i32x16_2) - bias_correction_i32,
                _mm512_reduce_add_epi32(accumulator_i32x16_1) - bias_correction_i32,
                _mm512_reduce_add_epi32(accumulator_i32x16_0) - bias_correction_i32);

            __m128i comparison_mask_i32x4 = _mm_cmpgt_epi32(coarse_dots_i32x4, running_max_i32x4);
            __m128i document_index_i32x4 = _mm_set1_epi32((int)document_index);
            running_max_i32x4 = _mm_blendv_epi8(running_max_i32x4, coarse_dots_i32x4, comparison_mask_i32x4);
            running_argmax_i32x4 = _mm_blendv_epi8(running_argmax_i32x4, document_index_i32x4, comparison_mask_i32x4);
        }

        // Extract argmax indices and refine
        nk_u32_t best_document_index_0_u32 = (nk_u32_t)_mm_extract_epi32(running_argmax_i32x4, 0);
        nk_u32_t best_document_index_1_u32 = (nk_u32_t)_mm_extract_epi32(running_argmax_i32x4, 1);
        nk_u32_t best_document_index_2_u32 = (nk_u32_t)_mm_extract_epi32(running_argmax_i32x4, 2);
        nk_u32_t best_document_index_3_u32 = (nk_u32_t)_mm_extract_epi32(running_argmax_i32x4, 3);

        // Phase 2: Full-precision bf16 refinement for 4 winning pairs
        nk_f32_t result_f32;
        nk_dot_bf16_genoa((nk_bf16_t const *)(q_originals + (query_block_start_index + 0) * q_original_stride),
                          (nk_bf16_t const *)(d_originals + best_document_index_0_u32 * d_original_stride), depth,
                          &result_f32);
        total_f32 += result_f32;
        nk_dot_bf16_genoa((nk_bf16_t const *)(q_originals + (query_block_start_index + 1) * q_original_stride),
                          (nk_bf16_t const *)(d_originals + best_document_index_1_u32 * d_original_stride), depth,
                          &result_f32);
        total_f32 += result_f32;
        nk_dot_bf16_genoa((nk_bf16_t const *)(q_originals + (query_block_start_index + 2) * q_original_stride),
                          (nk_bf16_t const *)(d_originals + best_document_index_2_u32 * d_original_stride), depth,
                          &result_f32);
        total_f32 += result_f32;
        nk_dot_bf16_genoa((nk_bf16_t const *)(q_originals + (query_block_start_index + 3) * q_original_stride),
                          (nk_bf16_t const *)(d_originals + best_document_index_3_u32 * d_original_stride), depth,
                          &result_f32);
        total_f32 += result_f32;
    }

    // Query tail: 1x1 (remaining n_q % 4 queries)
    for (nk_size_t query_index = query_block_start_index; query_index < n_q; query_index++) {
        nk_i8_t const *query_i8_row = q_i8 + query_index * depth_i8_padded;
        nk_i32_t running_max_i32 = NK_I32_MIN;
        nk_u32_t running_argmax_u32 = 0;

        for (nk_size_t document_index = 0; document_index < n_d; document_index++) {
            nk_i8_t const *document_i8_row = d_i8 + document_index * depth_i8_padded;
            __m512i accumulator_i32x16 = _mm512_setzero_si512();

            for (nk_size_t depth_index = 0; depth_index < depth_i8_padded; depth_index += 64) {
                __m512i document_i8x64 = _mm512_loadu_si512((__m512i const *)(document_i8_row + depth_index));
                __m512i query_i8x64 = _mm512_loadu_si512((__m512i const *)(query_i8_row + depth_index));
                __m512i query_biased_u8x64 = _mm512_xor_si512(query_i8x64, xor_mask_u8x64);
                accumulator_i32x16 = _mm512_dpbusd_epi32(accumulator_i32x16, query_biased_u8x64, document_i8x64);
            }

            nk_i32_t coarse_dot_i32 = _mm512_reduce_add_epi32(accumulator_i32x16) -
                                      128 * d_metadata[document_index].sum_i8_i32;

            if (coarse_dot_i32 > running_max_i32) {
                running_max_i32 = coarse_dot_i32;
                running_argmax_u32 = (nk_u32_t)document_index;
            }
        }

        nk_bf16_t const *q_original = (nk_bf16_t const *)(q_originals + query_index * q_original_stride);
        nk_bf16_t const *d_original = (nk_bf16_t const *)(d_originals + running_argmax_u32 * d_original_stride);
        nk_f32_t result_f32;
        nk_dot_bf16_genoa(q_original, d_original, depth, &result_f32);
        total_f32 += result_f32;
    }

    return total_f32;
}

NK_PUBLIC nk_f32_t nk_maxsim_packed_f32_genoa(  //
    void const *q_packed, void const *d_packed, //
    nk_size_t n_q, nk_size_t n_d, nk_size_t depth) {

    nk_maxsim_packed_header_t const *q_header = (nk_maxsim_packed_header_t const *)q_packed;
    nk_maxsim_packed_header_t const *d_header = (nk_maxsim_packed_header_t const *)d_packed;

    nk_size_t const depth_i8_padded = q_header->depth_i8_padded;

    nk_i8_t const *q_i8 = (nk_i8_t const *)((char const *)q_packed + q_header->offset_i8_data);
    nk_i8_t const *d_i8 = (nk_i8_t const *)((char const *)d_packed + d_header->offset_i8_data);
    nk_maxsim_vector_metadata_t const *d_metadata = (nk_maxsim_vector_metadata_t const *)((char const *)d_packed +
                                                                                          d_header->offset_metadata);
    char const *q_originals = (char const *)q_packed + q_header->offset_original_data;
    char const *d_originals = (char const *)d_packed + d_header->offset_original_data;
    nk_size_t const q_original_stride = q_header->original_stride_bytes;
    nk_size_t const d_original_stride = d_header->original_stride_bytes;

    __m512i const xor_mask_u8x64 = _mm512_set1_epi8((char)0x80);
    nk_f32_t total_f32 = 0.0f;

    // Primary path: 4-query grouping
    nk_size_t query_block_start_index = 0;
    for (; query_block_start_index + 4 <= n_q; query_block_start_index += 4) {
        __m128i running_max_i32x4 = _mm_set1_epi32(NK_I32_MIN);
        __m128i running_argmax_i32x4 = _mm_setzero_si128();

        // 4x4 document blocking
        nk_size_t document_block_start_index = 0;
        for (; document_block_start_index + 4 <= n_d; document_block_start_index += 4) {
            __m512i accumulator_tiles_i32x16[4][4];
            for (int q = 0; q < 4; q++)
                for (int d = 0; d < 4; d++) accumulator_tiles_i32x16[q][d] = _mm512_setzero_si512();

            for (nk_size_t depth_index = 0; depth_index < depth_i8_padded; depth_index += 64) {
                __m512i query_biased_u8x64_0 = _mm512_xor_si512(
                    _mm512_loadu_si512(
                        (__m512i const *)(q_i8 + (query_block_start_index + 0) * depth_i8_padded + depth_index)),
                    xor_mask_u8x64);
                __m512i query_biased_u8x64_1 = _mm512_xor_si512(
                    _mm512_loadu_si512(
                        (__m512i const *)(q_i8 + (query_block_start_index + 1) * depth_i8_padded + depth_index)),
                    xor_mask_u8x64);
                __m512i query_biased_u8x64_2 = _mm512_xor_si512(
                    _mm512_loadu_si512(
                        (__m512i const *)(q_i8 + (query_block_start_index + 2) * depth_i8_padded + depth_index)),
                    xor_mask_u8x64);
                __m512i query_biased_u8x64_3 = _mm512_xor_si512(
                    _mm512_loadu_si512(
                        (__m512i const *)(q_i8 + (query_block_start_index + 3) * depth_i8_padded + depth_index)),
                    xor_mask_u8x64);

                __m512i document_i8x64 = _mm512_loadu_si512(
                    (__m512i const *)(d_i8 + (document_block_start_index + 0) * depth_i8_padded + depth_index));
                accumulator_tiles_i32x16[0][0] = _mm512_dpbusd_epi32(accumulator_tiles_i32x16[0][0],
                                                                     query_biased_u8x64_0, document_i8x64);
                accumulator_tiles_i32x16[1][0] = _mm512_dpbusd_epi32(accumulator_tiles_i32x16[1][0],
                                                                     query_biased_u8x64_1, document_i8x64);
                accumulator_tiles_i32x16[2][0] = _mm512_dpbusd_epi32(accumulator_tiles_i32x16[2][0],
                                                                     query_biased_u8x64_2, document_i8x64);
                accumulator_tiles_i32x16[3][0] = _mm512_dpbusd_epi32(accumulator_tiles_i32x16[3][0],
                                                                     query_biased_u8x64_3, document_i8x64);

                document_i8x64 = _mm512_loadu_si512(
                    (__m512i const *)(d_i8 + (document_block_start_index + 1) * depth_i8_padded + depth_index));
                accumulator_tiles_i32x16[0][1] = _mm512_dpbusd_epi32(accumulator_tiles_i32x16[0][1],
                                                                     query_biased_u8x64_0, document_i8x64);
                accumulator_tiles_i32x16[1][1] = _mm512_dpbusd_epi32(accumulator_tiles_i32x16[1][1],
                                                                     query_biased_u8x64_1, document_i8x64);
                accumulator_tiles_i32x16[2][1] = _mm512_dpbusd_epi32(accumulator_tiles_i32x16[2][1],
                                                                     query_biased_u8x64_2, document_i8x64);
                accumulator_tiles_i32x16[3][1] = _mm512_dpbusd_epi32(accumulator_tiles_i32x16[3][1],
                                                                     query_biased_u8x64_3, document_i8x64);

                document_i8x64 = _mm512_loadu_si512(
                    (__m512i const *)(d_i8 + (document_block_start_index + 2) * depth_i8_padded + depth_index));
                accumulator_tiles_i32x16[0][2] = _mm512_dpbusd_epi32(accumulator_tiles_i32x16[0][2],
                                                                     query_biased_u8x64_0, document_i8x64);
                accumulator_tiles_i32x16[1][2] = _mm512_dpbusd_epi32(accumulator_tiles_i32x16[1][2],
                                                                     query_biased_u8x64_1, document_i8x64);
                accumulator_tiles_i32x16[2][2] = _mm512_dpbusd_epi32(accumulator_tiles_i32x16[2][2],
                                                                     query_biased_u8x64_2, document_i8x64);
                accumulator_tiles_i32x16[3][2] = _mm512_dpbusd_epi32(accumulator_tiles_i32x16[3][2],
                                                                     query_biased_u8x64_3, document_i8x64);

                document_i8x64 = _mm512_loadu_si512(
                    (__m512i const *)(d_i8 + (document_block_start_index + 3) * depth_i8_padded + depth_index));
                accumulator_tiles_i32x16[0][3] = _mm512_dpbusd_epi32(accumulator_tiles_i32x16[0][3],
                                                                     query_biased_u8x64_0, document_i8x64);
                accumulator_tiles_i32x16[1][3] = _mm512_dpbusd_epi32(accumulator_tiles_i32x16[1][3],
                                                                     query_biased_u8x64_1, document_i8x64);
                accumulator_tiles_i32x16[2][3] = _mm512_dpbusd_epi32(accumulator_tiles_i32x16[2][3],
                                                                     query_biased_u8x64_2, document_i8x64);
                accumulator_tiles_i32x16[3][3] = _mm512_dpbusd_epi32(accumulator_tiles_i32x16[3][3],
                                                                     query_biased_u8x64_3, document_i8x64);
            }

            __m128i query_0_coarse_dots_i32x4 = nk_maxsim_reduce_i32x16x4_genoa_(
                accumulator_tiles_i32x16[0][0], accumulator_tiles_i32x16[0][1], accumulator_tiles_i32x16[0][2],
                accumulator_tiles_i32x16[0][3]);
            __m128i query_1_coarse_dots_i32x4 = nk_maxsim_reduce_i32x16x4_genoa_(
                accumulator_tiles_i32x16[1][0], accumulator_tiles_i32x16[1][1], accumulator_tiles_i32x16[1][2],
                accumulator_tiles_i32x16[1][3]);
            __m128i query_2_coarse_dots_i32x4 = nk_maxsim_reduce_i32x16x4_genoa_(
                accumulator_tiles_i32x16[2][0], accumulator_tiles_i32x16[2][1], accumulator_tiles_i32x16[2][2],
                accumulator_tiles_i32x16[2][3]);
            __m128i query_3_coarse_dots_i32x4 = nk_maxsim_reduce_i32x16x4_genoa_(
                accumulator_tiles_i32x16[3][0], accumulator_tiles_i32x16[3][1], accumulator_tiles_i32x16[3][2],
                accumulator_tiles_i32x16[3][3]);

            __m128i bias_correction_i32x4 = _mm_set_epi32(128 * d_metadata[document_block_start_index + 3].sum_i8_i32,
                                                          128 * d_metadata[document_block_start_index + 2].sum_i8_i32,
                                                          128 * d_metadata[document_block_start_index + 1].sum_i8_i32,
                                                          128 * d_metadata[document_block_start_index + 0].sum_i8_i32);
            query_0_coarse_dots_i32x4 = _mm_sub_epi32(query_0_coarse_dots_i32x4, bias_correction_i32x4);
            query_1_coarse_dots_i32x4 = _mm_sub_epi32(query_1_coarse_dots_i32x4, bias_correction_i32x4);
            query_2_coarse_dots_i32x4 = _mm_sub_epi32(query_2_coarse_dots_i32x4, bias_correction_i32x4);
            query_3_coarse_dots_i32x4 = _mm_sub_epi32(query_3_coarse_dots_i32x4, bias_correction_i32x4);

            __m128i transpose_queries_01_low_i32x4 = _mm_unpacklo_epi32(query_0_coarse_dots_i32x4,
                                                                        query_1_coarse_dots_i32x4);
            __m128i transpose_queries_23_low_i32x4 = _mm_unpacklo_epi32(query_2_coarse_dots_i32x4,
                                                                        query_3_coarse_dots_i32x4);
            __m128i transpose_queries_01_high_i32x4 = _mm_unpackhi_epi32(query_0_coarse_dots_i32x4,
                                                                         query_1_coarse_dots_i32x4);
            __m128i transpose_queries_23_high_i32x4 = _mm_unpackhi_epi32(query_2_coarse_dots_i32x4,
                                                                         query_3_coarse_dots_i32x4);
            __m128i document_0_dots_i32x4 = _mm_unpacklo_epi64(transpose_queries_01_low_i32x4,
                                                               transpose_queries_23_low_i32x4);
            __m128i document_1_dots_i32x4 = _mm_unpackhi_epi64(transpose_queries_01_low_i32x4,
                                                               transpose_queries_23_low_i32x4);
            __m128i document_2_dots_i32x4 = _mm_unpacklo_epi64(transpose_queries_01_high_i32x4,
                                                               transpose_queries_23_high_i32x4);
            __m128i document_3_dots_i32x4 = _mm_unpackhi_epi64(transpose_queries_01_high_i32x4,
                                                               transpose_queries_23_high_i32x4);

            __m128i comparison_mask_i32x4, document_index_i32x4;

            comparison_mask_i32x4 = _mm_cmpgt_epi32(document_0_dots_i32x4, running_max_i32x4);
            document_index_i32x4 = _mm_set1_epi32((int)(document_block_start_index + 0));
            running_max_i32x4 = _mm_blendv_epi8(running_max_i32x4, document_0_dots_i32x4, comparison_mask_i32x4);
            running_argmax_i32x4 = _mm_blendv_epi8(running_argmax_i32x4, document_index_i32x4, comparison_mask_i32x4);

            comparison_mask_i32x4 = _mm_cmpgt_epi32(document_1_dots_i32x4, running_max_i32x4);
            document_index_i32x4 = _mm_set1_epi32((int)(document_block_start_index + 1));
            running_max_i32x4 = _mm_blendv_epi8(running_max_i32x4, document_1_dots_i32x4, comparison_mask_i32x4);
            running_argmax_i32x4 = _mm_blendv_epi8(running_argmax_i32x4, document_index_i32x4, comparison_mask_i32x4);

            comparison_mask_i32x4 = _mm_cmpgt_epi32(document_2_dots_i32x4, running_max_i32x4);
            document_index_i32x4 = _mm_set1_epi32((int)(document_block_start_index + 2));
            running_max_i32x4 = _mm_blendv_epi8(running_max_i32x4, document_2_dots_i32x4, comparison_mask_i32x4);
            running_argmax_i32x4 = _mm_blendv_epi8(running_argmax_i32x4, document_index_i32x4, comparison_mask_i32x4);

            comparison_mask_i32x4 = _mm_cmpgt_epi32(document_3_dots_i32x4, running_max_i32x4);
            document_index_i32x4 = _mm_set1_epi32((int)(document_block_start_index + 3));
            running_max_i32x4 = _mm_blendv_epi8(running_max_i32x4, document_3_dots_i32x4, comparison_mask_i32x4);
            running_argmax_i32x4 = _mm_blendv_epi8(running_argmax_i32x4, document_index_i32x4, comparison_mask_i32x4);
        }

        // Document tail: 4x1
        for (nk_size_t document_index = document_block_start_index; document_index < n_d; document_index++) {
            nk_i8_t const *document_i8_row = d_i8 + document_index * depth_i8_padded;

            __m512i accumulator_i32x16_0 = _mm512_setzero_si512();
            __m512i accumulator_i32x16_1 = _mm512_setzero_si512();
            __m512i accumulator_i32x16_2 = _mm512_setzero_si512();
            __m512i accumulator_i32x16_3 = _mm512_setzero_si512();

            for (nk_size_t depth_index = 0; depth_index < depth_i8_padded; depth_index += 64) {
                __m512i document_i8x64 = _mm512_loadu_si512((__m512i const *)(document_i8_row + depth_index));

                __m512i query_biased_u8x64_0 = _mm512_xor_si512(
                    _mm512_loadu_si512(
                        (__m512i const *)(q_i8 + (query_block_start_index + 0) * depth_i8_padded + depth_index)),
                    xor_mask_u8x64);
                accumulator_i32x16_0 = _mm512_dpbusd_epi32(accumulator_i32x16_0, query_biased_u8x64_0, document_i8x64);

                __m512i query_biased_u8x64_1 = _mm512_xor_si512(
                    _mm512_loadu_si512(
                        (__m512i const *)(q_i8 + (query_block_start_index + 1) * depth_i8_padded + depth_index)),
                    xor_mask_u8x64);
                accumulator_i32x16_1 = _mm512_dpbusd_epi32(accumulator_i32x16_1, query_biased_u8x64_1, document_i8x64);

                __m512i query_biased_u8x64_2 = _mm512_xor_si512(
                    _mm512_loadu_si512(
                        (__m512i const *)(q_i8 + (query_block_start_index + 2) * depth_i8_padded + depth_index)),
                    xor_mask_u8x64);
                accumulator_i32x16_2 = _mm512_dpbusd_epi32(accumulator_i32x16_2, query_biased_u8x64_2, document_i8x64);

                __m512i query_biased_u8x64_3 = _mm512_xor_si512(
                    _mm512_loadu_si512(
                        (__m512i const *)(q_i8 + (query_block_start_index + 3) * depth_i8_padded + depth_index)),
                    xor_mask_u8x64);
                accumulator_i32x16_3 = _mm512_dpbusd_epi32(accumulator_i32x16_3, query_biased_u8x64_3, document_i8x64);
            }

            nk_i32_t bias_correction_i32 = 128 * d_metadata[document_index].sum_i8_i32;
            __m128i coarse_dots_i32x4 = _mm_set_epi32(
                _mm512_reduce_add_epi32(accumulator_i32x16_3) - bias_correction_i32,
                _mm512_reduce_add_epi32(accumulator_i32x16_2) - bias_correction_i32,
                _mm512_reduce_add_epi32(accumulator_i32x16_1) - bias_correction_i32,
                _mm512_reduce_add_epi32(accumulator_i32x16_0) - bias_correction_i32);

            __m128i comparison_mask_i32x4 = _mm_cmpgt_epi32(coarse_dots_i32x4, running_max_i32x4);
            __m128i document_index_i32x4 = _mm_set1_epi32((int)document_index);
            running_max_i32x4 = _mm_blendv_epi8(running_max_i32x4, coarse_dots_i32x4, comparison_mask_i32x4);
            running_argmax_i32x4 = _mm_blendv_epi8(running_argmax_i32x4, document_index_i32x4, comparison_mask_i32x4);
        }

        nk_u32_t best_document_index_0_u32 = (nk_u32_t)_mm_extract_epi32(running_argmax_i32x4, 0);
        nk_u32_t best_document_index_1_u32 = (nk_u32_t)_mm_extract_epi32(running_argmax_i32x4, 1);
        nk_u32_t best_document_index_2_u32 = (nk_u32_t)_mm_extract_epi32(running_argmax_i32x4, 2);
        nk_u32_t best_document_index_3_u32 = (nk_u32_t)_mm_extract_epi32(running_argmax_i32x4, 3);

        nk_f32_t result_f32;
        nk_dot_f32_haswell((nk_f32_t const *)(q_originals + (query_block_start_index + 0) * q_original_stride),
                           (nk_f32_t const *)(d_originals + best_document_index_0_u32 * d_original_stride), depth,
                           &result_f32);
        total_f32 += result_f32;
        nk_dot_f32_haswell((nk_f32_t const *)(q_originals + (query_block_start_index + 1) * q_original_stride),
                           (nk_f32_t const *)(d_originals + best_document_index_1_u32 * d_original_stride), depth,
                           &result_f32);
        total_f32 += result_f32;
        nk_dot_f32_haswell((nk_f32_t const *)(q_originals + (query_block_start_index + 2) * q_original_stride),
                           (nk_f32_t const *)(d_originals + best_document_index_2_u32 * d_original_stride), depth,
                           &result_f32);
        total_f32 += result_f32;
        nk_dot_f32_haswell((nk_f32_t const *)(q_originals + (query_block_start_index + 3) * q_original_stride),
                           (nk_f32_t const *)(d_originals + best_document_index_3_u32 * d_original_stride), depth,
                           &result_f32);
        total_f32 += result_f32;
    }

    // Query tail: 1x1
    for (nk_size_t query_index = query_block_start_index; query_index < n_q; query_index++) {
        nk_i8_t const *query_i8_row = q_i8 + query_index * depth_i8_padded;
        nk_i32_t running_max_i32 = NK_I32_MIN;
        nk_u32_t running_argmax_u32 = 0;

        for (nk_size_t document_index = 0; document_index < n_d; document_index++) {
            nk_i8_t const *document_i8_row = d_i8 + document_index * depth_i8_padded;
            __m512i accumulator_i32x16 = _mm512_setzero_si512();

            for (nk_size_t depth_index = 0; depth_index < depth_i8_padded; depth_index += 64) {
                __m512i document_i8x64 = _mm512_loadu_si512((__m512i const *)(document_i8_row + depth_index));
                __m512i query_i8x64 = _mm512_loadu_si512((__m512i const *)(query_i8_row + depth_index));
                __m512i query_biased_u8x64 = _mm512_xor_si512(query_i8x64, xor_mask_u8x64);
                accumulator_i32x16 = _mm512_dpbusd_epi32(accumulator_i32x16, query_biased_u8x64, document_i8x64);
            }

            nk_i32_t coarse_dot_i32 = _mm512_reduce_add_epi32(accumulator_i32x16) -
                                      128 * d_metadata[document_index].sum_i8_i32;

            if (coarse_dot_i32 > running_max_i32) {
                running_max_i32 = coarse_dot_i32;
                running_argmax_u32 = (nk_u32_t)document_index;
            }
        }

        nk_f32_t const *q_original = (nk_f32_t const *)(q_originals + query_index * q_original_stride);
        nk_f32_t const *d_original = (nk_f32_t const *)(d_originals + running_argmax_u32 * d_original_stride);
        nk_f32_t result_f32;
        nk_dot_f32_haswell(q_original, d_original, depth, &result_f32);
        total_f32 += result_f32;
    }

    return total_f32;
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_GENOA
#endif // NK_TARGET_X86_
#endif // NK_MAXSIM_GENOA_H

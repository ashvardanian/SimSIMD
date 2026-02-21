/**
 *  @brief SWAR-accelerated MaxSim (ColBERT late-interaction) for SIMD-free CPUs.
 *  @file include/numkong/maxsim/serial.h
 *  @author Ash Vardanian
 *  @date February 17, 2026
 *
 *  @sa include/numkong/maxsim.h
 *
 *  Defines the packed buffer header and per-vector metadata structures used by all MaxSim ISA backends,
 *  plus scalar reference implementations for correctness validation.
 *
 *  MaxSim computes: result = Σᵢ maxⱼ dot(qᵢ, dⱼ) — ColBERT late-interaction scoring.
 *
 *  Strategy: coarse i8-quantized screening with running argmax, then full-precision refinement
 *  of the winning (query, document) pairs via existing nk_dot_* primitives.
 *
 *  @section packed_layout Packed Buffer Layout
 *
 *  [Header 64B] [i8 vectors, 64B-aligned] [metadata, 64B-aligned] [originals row-major, 64B-aligned]
 *
 *  - i8 region: row-major with padded depth for SIMD alignment
 *  - Metadata region: vector_count x 8 bytes (scale + sum per vector)
 *  - Originals region: row-major bf16 or f32, stride padded to 64B for nk_dot_* calls
 */
#ifndef NK_MAXSIM_SERIAL_H
#define NK_MAXSIM_SERIAL_H

#include "numkong/types.h"
#include "numkong/cast/serial.h" // nk_bf16_to_f32_serial
#include "numkong/dot/serial.h"  // nk_dot_bf16_serial, nk_dot_f32_serial

#if defined(__cplusplus)
extern "C" {
#endif

/**
 *  @brief Packed buffer header (64 bytes, cache-line aligned).
 *  Stored at the beginning of every maxsim packed buffer.
 */
typedef struct {
    nk_u32_t vector_count;           ///< Number of vectors packed
    nk_u32_t depth_dimensions;       ///< Logical depth (number of elements per vector)
    nk_u32_t depth_i8_padded;        ///< Padded i8 depth in bytes (SIMD-aligned)
    nk_u32_t original_element_bytes; ///< 2 for bf16, 4 for f32
    nk_u32_t offset_i8_data;         ///< Byte offset from buffer start to i8 region
    nk_u32_t offset_metadata;        ///< Byte offset from buffer start to metadata region
    nk_u32_t offset_original_data;   ///< Byte offset from buffer start to originals region
    nk_u32_t original_stride_bytes;  ///< Row stride in bytes for originals region
    nk_u32_t reserved[8];            ///< Padding to 64 bytes
} nk_maxsim_packed_header_t;

NK_STATIC_ASSERT(sizeof(nk_maxsim_packed_header_t) == 64, nk_maxsim_packed_header_must_be_64_bytes);

/**
 *  @brief Per-vector quantization metadata (8 bytes).
 *  Stored in the metadata region of the packed buffer, one per vector.
 */
typedef struct {
    nk_f32_t scale_f32;  ///< Quantization scale: absmax / 127
    nk_i32_t sum_i8_i32; ///< Sum of all i8 quantized elements (for VPMADDUBSW bias correction)
} nk_maxsim_vector_metadata_t;

NK_STATIC_ASSERT(sizeof(nk_maxsim_vector_metadata_t) == 8, nk_maxsim_vector_metadata_must_be_8_bytes);

/**
 *  @brief Computes padded i8 depth and total packed buffer size for maxsim.
 *
 *  Layout: header + i8 data (64B-aligned) + metadata (64B-aligned) + originals (64B-aligned)
 *
 *  @param vector_count Number of vectors to pack.
 *  @param depth Number of elements per vector.
 *  @param original_element_bytes Size of each original element (2 for bf16, 4 for f32).
 *  @param depth_simd_dimensions SIMD width for i8 depth padding (1 for serial).
 */
NK_INTERNAL nk_size_t nk_maxsim_packed_size_( //
    nk_size_t vector_count, nk_size_t depth,  //
    nk_size_t original_element_bytes, nk_size_t depth_simd_dimensions) {

    // Step 1: Pad i8 depth to SIMD width
    nk_size_t depth_i8_padded = nk_size_round_up_to_multiple_(depth, depth_simd_dimensions);

    // Step 2: Break power-of-2 strides for cache associativity
    if ((depth_i8_padded & (depth_i8_padded - 1)) == 0 && depth_i8_padded > 0) depth_i8_padded += depth_simd_dimensions;

    // Step 3: Calculate region sizes
    nk_size_t const header_size = sizeof(nk_maxsim_packed_header_t);
    nk_size_t const i8_region_size = nk_size_round_up_to_multiple_(vector_count * depth_i8_padded, 64);
    nk_size_t const metadata_region_size = nk_size_round_up_to_multiple_(
        vector_count * sizeof(nk_maxsim_vector_metadata_t), 64);
    nk_size_t const original_stride = nk_size_round_up_to_multiple_(depth * original_element_bytes, 64);
    nk_size_t const originals_region_size = vector_count * original_stride;

    return header_size + i8_region_size + metadata_region_size + originals_region_size;
}

NK_PUBLIC nk_size_t nk_maxsim_packed_size_bf16_serial(nk_size_t n, nk_size_t k) {
    return nk_maxsim_packed_size_(n, k, sizeof(nk_bf16_t), 1);
}

NK_PUBLIC nk_size_t nk_maxsim_packed_size_f32_serial(nk_size_t n, nk_size_t k) {
    return nk_maxsim_packed_size_(n, k, sizeof(nk_f32_t), 1);
}

NK_PUBLIC void nk_maxsim_pack_bf16_serial( //
    nk_bf16_t const *vectors, nk_size_t n, nk_size_t k, nk_size_t stride, void *packed) {

    nk_size_t const original_element_bytes = sizeof(nk_bf16_t);

    // Compute padded depth (depth_simd_dimensions = 1 for serial)
    nk_size_t depth_i8_padded = k;
    if ((depth_i8_padded & (depth_i8_padded - 1)) == 0 && depth_i8_padded > 0) depth_i8_padded += 1;

    // Compute region offsets
    nk_size_t const header_size = sizeof(nk_maxsim_packed_header_t);
    nk_size_t const i8_region_size = nk_size_round_up_to_multiple_(n * depth_i8_padded, 64);
    nk_size_t const metadata_region_size = nk_size_round_up_to_multiple_(
        n * sizeof(nk_maxsim_vector_metadata_t), 64);
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

NK_PUBLIC void nk_maxsim_pack_f32_serial( //
    nk_f32_t const *vectors, nk_size_t n, nk_size_t k, nk_size_t stride, void *packed) {

    nk_size_t const original_element_bytes = sizeof(nk_f32_t);

    // Compute padded depth (depth_simd_dimensions = 1 for serial)
    nk_size_t depth_i8_padded = k;
    if ((depth_i8_padded & (depth_i8_padded - 1)) == 0 && depth_i8_padded > 0) depth_i8_padded += 1;

    // Compute region offsets
    nk_size_t const header_size = sizeof(nk_maxsim_packed_header_t);
    nk_size_t const i8_region_size = nk_size_round_up_to_multiple_(n * depth_i8_padded, 64);
    nk_size_t const metadata_region_size = nk_size_round_up_to_multiple_(
        n * sizeof(nk_maxsim_vector_metadata_t), 64);
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

NK_PUBLIC nk_f32_t nk_maxsim_packed_bf16_serial( //
    void const *q_packed, void const *d_packed,  //
    nk_size_t n_q, nk_size_t n_d, nk_size_t depth) {

    nk_maxsim_packed_header_t const *q_header = (nk_maxsim_packed_header_t const *)q_packed;
    nk_maxsim_packed_header_t const *d_header = (nk_maxsim_packed_header_t const *)d_packed;

    nk_size_t const depth_i8_padded = q_header->depth_i8_padded;

    nk_i8_t const *q_i8 = (nk_i8_t const *)((char const *)q_packed + q_header->offset_i8_data);
    nk_i8_t const *d_i8 = (nk_i8_t const *)((char const *)d_packed + d_header->offset_i8_data);
    char const *q_originals = (char const *)q_packed + q_header->offset_original_data;
    char const *d_originals = (char const *)d_packed + d_header->offset_original_data;
    nk_size_t const q_original_stride = q_header->original_stride_bytes;
    nk_size_t const d_original_stride = d_header->original_stride_bytes;

    nk_f32_t total_f32 = 0.0f;

    // Primary path: 4-query grouping
    nk_size_t query_block_start_index = 0;
    for (; query_block_start_index + 4 <= n_q; query_block_start_index += 4) {
        nk_i32_t running_max_i32[4] = {NK_I32_MIN, NK_I32_MIN, NK_I32_MIN, NK_I32_MIN};
        nk_u32_t running_argmax_u32[4] = {0, 0, 0, 0};

        for (nk_size_t document_index = 0; document_index < n_d; document_index++) {
            nk_i8_t const *document_i8_row = d_i8 + document_index * depth_i8_padded;
            nk_i32_t accumulator_i32[4] = {0, 0, 0, 0};

            for (nk_size_t k = 0; k < depth_i8_padded; k++) {
                nk_i32_t d_val = (nk_i32_t)document_i8_row[k];
                accumulator_i32[0] += (nk_i32_t)q_i8[(query_block_start_index + 0) * depth_i8_padded + k] * d_val;
                accumulator_i32[1] += (nk_i32_t)q_i8[(query_block_start_index + 1) * depth_i8_padded + k] * d_val;
                accumulator_i32[2] += (nk_i32_t)q_i8[(query_block_start_index + 2) * depth_i8_padded + k] * d_val;
                accumulator_i32[3] += (nk_i32_t)q_i8[(query_block_start_index + 3) * depth_i8_padded + k] * d_val;
            }

            for (int q = 0; q < 4; q++) {
                if (accumulator_i32[q] > running_max_i32[q]) {
                    running_max_i32[q] = accumulator_i32[q];
                    running_argmax_u32[q] = (nk_u32_t)document_index;
                }
            }
        }

        // Phase 2: Full-precision bf16 refinement for 4 winning pairs
        for (int q = 0; q < 4; q++) {
            nk_bf16_t const *q_original = (nk_bf16_t const *)(q_originals +
                                                              (query_block_start_index + q) * q_original_stride);
            nk_bf16_t const *d_original = (nk_bf16_t const *)(d_originals + running_argmax_u32[q] * d_original_stride);
            nk_f32_t result_f32;
            nk_dot_bf16_serial(q_original, d_original, depth, &result_f32);
            total_f32 += result_f32;
        }
    }

    // Edge path: remaining 1-3 queries
    for (nk_size_t query_index = query_block_start_index; query_index < n_q; query_index++) {
        nk_i8_t const *query_i8_row = q_i8 + query_index * depth_i8_padded;
        nk_i32_t running_max_i32 = NK_I32_MIN;
        nk_u32_t running_argmax_u32 = 0;

        for (nk_size_t document_index = 0; document_index < n_d; document_index++) {
            nk_i8_t const *document_i8_row = d_i8 + document_index * depth_i8_padded;
            nk_i32_t accumulator_i32 = 0;

            for (nk_size_t k = 0; k < depth_i8_padded; k++)
                accumulator_i32 += (nk_i32_t)query_i8_row[k] * (nk_i32_t)document_i8_row[k];

            if (accumulator_i32 > running_max_i32) {
                running_max_i32 = accumulator_i32;
                running_argmax_u32 = (nk_u32_t)document_index;
            }
        }

        nk_bf16_t const *q_original = (nk_bf16_t const *)(q_originals + query_index * q_original_stride);
        nk_bf16_t const *d_original = (nk_bf16_t const *)(d_originals + running_argmax_u32 * d_original_stride);
        nk_f32_t result_f32;
        nk_dot_bf16_serial(q_original, d_original, depth, &result_f32);
        total_f32 += result_f32;
    }

    return total_f32;
}

NK_PUBLIC nk_f32_t nk_maxsim_packed_f32_serial( //
    void const *q_packed, void const *d_packed, //
    nk_size_t n_q, nk_size_t n_d, nk_size_t depth) {

    nk_maxsim_packed_header_t const *q_header = (nk_maxsim_packed_header_t const *)q_packed;
    nk_maxsim_packed_header_t const *d_header = (nk_maxsim_packed_header_t const *)d_packed;

    nk_size_t const depth_i8_padded = q_header->depth_i8_padded;

    nk_i8_t const *q_i8 = (nk_i8_t const *)((char const *)q_packed + q_header->offset_i8_data);
    nk_i8_t const *d_i8 = (nk_i8_t const *)((char const *)d_packed + d_header->offset_i8_data);
    char const *q_originals = (char const *)q_packed + q_header->offset_original_data;
    char const *d_originals = (char const *)d_packed + d_header->offset_original_data;
    nk_size_t const q_original_stride = q_header->original_stride_bytes;
    nk_size_t const d_original_stride = d_header->original_stride_bytes;

    nk_f32_t total_f32 = 0.0f;

    // Primary path: 4-query grouping
    nk_size_t query_block_start_index = 0;
    for (; query_block_start_index + 4 <= n_q; query_block_start_index += 4) {
        nk_i32_t running_max_i32[4] = {NK_I32_MIN, NK_I32_MIN, NK_I32_MIN, NK_I32_MIN};
        nk_u32_t running_argmax_u32[4] = {0, 0, 0, 0};

        for (nk_size_t document_index = 0; document_index < n_d; document_index++) {
            nk_i8_t const *document_i8_row = d_i8 + document_index * depth_i8_padded;
            nk_i32_t accumulator_i32[4] = {0, 0, 0, 0};

            for (nk_size_t k = 0; k < depth_i8_padded; k++) {
                nk_i32_t d_val = (nk_i32_t)document_i8_row[k];
                accumulator_i32[0] += (nk_i32_t)q_i8[(query_block_start_index + 0) * depth_i8_padded + k] * d_val;
                accumulator_i32[1] += (nk_i32_t)q_i8[(query_block_start_index + 1) * depth_i8_padded + k] * d_val;
                accumulator_i32[2] += (nk_i32_t)q_i8[(query_block_start_index + 2) * depth_i8_padded + k] * d_val;
                accumulator_i32[3] += (nk_i32_t)q_i8[(query_block_start_index + 3) * depth_i8_padded + k] * d_val;
            }

            for (int q = 0; q < 4; q++) {
                if (accumulator_i32[q] > running_max_i32[q]) {
                    running_max_i32[q] = accumulator_i32[q];
                    running_argmax_u32[q] = (nk_u32_t)document_index;
                }
            }
        }

        for (int q = 0; q < 4; q++) {
            nk_f32_t const *q_original = (nk_f32_t const *)(q_originals +
                                                            (query_block_start_index + q) * q_original_stride);
            nk_f32_t const *d_original = (nk_f32_t const *)(d_originals + running_argmax_u32[q] * d_original_stride);
            nk_f32_t result_f32;
            nk_dot_f32_serial(q_original, d_original, depth, &result_f32);
            total_f32 += result_f32;
        }
    }

    // Edge path: remaining 1-3 queries
    for (nk_size_t query_index = query_block_start_index; query_index < n_q; query_index++) {
        nk_i8_t const *query_i8_row = q_i8 + query_index * depth_i8_padded;
        nk_i32_t running_max_i32 = NK_I32_MIN;
        nk_u32_t running_argmax_u32 = 0;

        for (nk_size_t document_index = 0; document_index < n_d; document_index++) {
            nk_i8_t const *document_i8_row = d_i8 + document_index * depth_i8_padded;
            nk_i32_t accumulator_i32 = 0;

            for (nk_size_t k = 0; k < depth_i8_padded; k++)
                accumulator_i32 += (nk_i32_t)query_i8_row[k] * (nk_i32_t)document_i8_row[k];

            if (accumulator_i32 > running_max_i32) {
                running_max_i32 = accumulator_i32;
                running_argmax_u32 = (nk_u32_t)document_index;
            }
        }

        nk_f32_t const *q_original = (nk_f32_t const *)(q_originals + query_index * q_original_stride);
        nk_f32_t const *d_original = (nk_f32_t const *)(d_originals + running_argmax_u32 * d_original_stride);
        nk_f32_t result_f32;
        nk_dot_f32_serial(q_original, d_original, depth, &result_f32);
        total_f32 += result_f32;
    }

    return total_f32;
}

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_MAXSIM_SERIAL_H

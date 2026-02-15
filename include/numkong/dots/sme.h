/**
 *  @brief SIMD-accelerated Batched Dot Products for SME.
 *  @file include/numkong/dots/sme.h
 *  @author Ash Vardanian
 *  @date January 2, 2026
 *
 *  @sa include/numkong/dots.h
 *
 *  Uses ARM Scalable Matrix Extension (SME) for efficient matrix multiplication
 *  with `ZA32` tiles supporting `f16`, `bf16`, `i8`, `u8`, and `e4m3` input types:
 *
 *  - `svmopa_za32_f16_m`: `f16` × `f16` outer product accumulate to `f32`
 *  - `svmopa_za32_bf16_m`: `bf16` × `bf16` outer product accumulate to `f32`
 *  - `svmopa_za32_s8_m`: `i8` × `i8` outer product accumulate to `i32`
 *  - `svmopa_za32_u8_m`: `u8` × `u8` outer product accumulate to `u32`
 *
 *  SME tile dimensions (for SVL=512, i.e., Apple M4):
 *
 *  - `ZA32` tile: 16 × 16 `f32`/`i32` elements (1KB)
 *  - `f16`/`bf16` vectors: 32 elements per SVE vector
 *  - `i8`/`u8` vectors: 64 elements per SVE vector
 *  - `f32`/`i32` vectors: 16 elements per SVE vector
 *
 *  Output pattern: Each `svmopa` accumulates a 16 × 16 tile from input vectors.
 *  We process multiple ZA tiles (0-3) to form larger output blocks.
 *
 *  Performance characteristics (Apple M4):
 *
 *  - `f16` → `f32` peak: ~2 TFLOPS per core
 *  - `bf16` → `f32` peak: ~2 TFLOPS per core
 *  - `i8` → `i32` peak: ~2 TOPS per core
 *  - Streaming mode has different register set from normal NEON
 *
 *  Acceleration opportunities:
 *
 *  - Pre-pack B matrix for column-major access: avoids transpose overhead
 *  - Tile along M/N dimensions: cache efficiency
 *  - Use multiple ZA tiles: 2×2 output blocking
 *
 *  @section dots_sme_instructions ARM SME Instructions
 *
 *      Intrinsic                       Instruction                     Latency     Throughput
 *      `svmopa_za32_f16_m`             `FMOPA` (ZA.S, P/M, Z.H, Z.H)   16cy        amortized
 *      `svmopa_za32_bf16_m`            `BFMOPA` (ZA.S, P/M, Z.H, Z.H)  16cy        amortized
 *      `svmopa_za32_s8_m`              `SMOPA` (ZA.S, P/M, Z.B, Z.B)   16cy        amortized
 *      `svmopa_za32_u8_m`              `UMOPA` (ZA.S, P/M, Z.B, Z.B)   16cy        amortized
 *      `svzero_za`                     `ZERO` (ZA)                     2cy         1/cy
 *      `svld1_hor_za32`                `LD1W` (ZA.S[Ws, #imm], P/Z)    4-6cy       1/cy
 *      `svst1_hor_za32`                `ST1W` (ZA.S[Ws, #imm], P)      4cy         1/cy
 *      `__arm_streaming`               `SMSTART`                       ~50-100cy
 *      `__arm_streaming` (exit)        `SMSTOP`                        ~50-100cy
 *      `__arm_new("za")`               ZA tile allocation              0cy
 *      `svcntw`                        `CNTW` (Xd)                     1cy         2/cy
 *      `svcnth`                        `CNTH` (Xd)                     1cy         2/cy
 */
#ifndef NK_DOTS_SME_H
#define NK_DOTS_SME_H

#if NK_TARGET_ARM_
#if NK_TARGET_SME

#include "numkong/types.h"
#include "numkong/cast/serial.h" // `nk_e4m3_to_f16_serial`, `nk_e5m2_to_f16_serial`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("sme"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("+sme")
#endif

/**
 *  SME-specific packed buffer header (64-byte aligned).
 *  Layout optimized for SME outer product access patterns with predicate-based edge handling.
 */
typedef struct {
    nk_u32_t column_tile_count; // ⌈columns/tile_dimension⌉: number of column tiles
    nk_u32_t depth_tile_count;  // ⌈depth/depth_tile_size⌉: number of depth tiles
    nk_u32_t columns;           // actual N dimension for predicates
    nk_u32_t depth;             // actual K dimension for predicates
    nk_u32_t svl_bytes;         // SVL in bytes at pack time: for validation
    nk_u32_t reserved[11];      // padding to 64 bytes
} nk_dots_sme_packed_header_t;

/*  Selective ZA tile zeroing masks for `svzero_mask_za(mask)`.
 *  These zero individual ZA.S tiles without destroying other accumulators.
 *  ZA.D tiles (8 tiles, 8x8 each): ZA0.D = mask bit 0, ..., ZA7.D = mask bit 7.
 */
enum {
    nk_sme_zero_za32_tile_0_ = 0x11,
    nk_sme_zero_za32_tile_1_ = 0x22,
    nk_sme_zero_za32_tiles_123_ = 0xEE, /* Accumulators only (preserves ZA0 staging) */
    nk_sme_zero_za64_tile_0_ = 0x01,
    nk_sme_zero_za64_tile_1_ = 0x02,
    nk_sme_zero_za64_tiles_1_7_ = 0xFE, /* Accumulators ZA1-7.D (preserves ZA0.D staging) */
};

/*
 *  f16/bf16 → f32 GEMM using FMOPA/BFMOPA with ZA32 tiles.
 *
 *  Tile layout (SVL=512, Apple M4):
 *  - ZA32 output tile: 16 × 16 f32 elements (1 KB)
 *  - Input vectors: 32 f16/bf16 elements (SVL/16)
 *  - Depth per FMOPA: 2 f16 pairs → 1 f32 (widening 2:1)
 *  - FMOPA predicates: b16 (input granularity), not b32
 *  - 4-tile path: ZA0-ZA3 process 4 column tiles simultaneously
 */
#pragma region Half Precision Floats

NK_PUBLIC nk_size_t nk_dots_packed_size_f16_sme(nk_size_t n, nk_size_t k) {
    nk_size_t const expansion = 2;               // FMOPA f16 → f32: 2 f16 pairs per f32 output
    nk_size_t const tile_dimension = svcntsw();  // ZA32 tile dim: 16
    nk_size_t const vector_elements = svcntsh(); // f16 elements per SVE vector: 32
    nk_size_t const column_tile_count = nk_size_divide_round_up_(n, tile_dimension);
    nk_size_t const depth_step_count = nk_size_divide_round_up_(k, expansion);

    nk_size_t size = sizeof(nk_dots_sme_packed_header_t);
    size += column_tile_count * depth_step_count * vector_elements * sizeof(nk_f16_t);
    return size;
}

NK_PUBLIC nk_size_t nk_dots_packed_size_bf16_sme(nk_size_t n, nk_size_t k) { return nk_dots_packed_size_f16_sme(n, k); }

NK_PUBLIC void nk_dots_pack_f16_sme(             //
    nk_f16_t const *b, nk_size_t n, nk_size_t k, //
    nk_size_t b_stride, void *b_packed) {

    nk_size_t const expansion = 2;               // FMOPA f16 → f32: 2 f16 pairs per f32 output
    nk_size_t const tile_dimension = svcntsw();  // ZA32 tile dim: 16
    nk_size_t const vector_elements = svcntsh(); // f16 per SVE vector: 32 = tile_dimension * expansion
    nk_size_t const b_stride_elements = b_stride / sizeof(nk_f16_t);

    nk_size_t const column_tile_count = nk_size_divide_round_up_(n, tile_dimension);
    nk_size_t const depth_step_count = nk_size_divide_round_up_(k, expansion);
    nk_size_t const total_vectors = column_tile_count * depth_step_count;

    nk_dots_sme_packed_header_t *header = (nk_dots_sme_packed_header_t *)b_packed;
    header->column_tile_count = (nk_u32_t)column_tile_count;
    header->depth_tile_count = (nk_u32_t)depth_step_count;
    header->columns = (nk_u32_t)n;
    header->depth = (nk_u32_t)k;
    header->svl_bytes = (nk_u32_t)(svcntsw() * sizeof(nk_f32_t));

    nk_f16_t *tiles_ptr = (nk_f16_t *)((char *)b_packed + sizeof(nk_dots_sme_packed_header_t));

    // Zero-initialize all packed data (partial vectors stay zero-padded)
    for (nk_size_t i = 0; i < total_vectors * vector_elements; i++) { tiles_ptr[i] = 0; }

    // Pack in interleaved format for FMOPA:
    // For each column tile and depth step, produce one SVE vector where:
    //   packed_vec[expansion * d + k] = B[column_tile*tile_dim + d, depth_step*expansion + k]
    // d = 0..tile_dim-1 (B row within column tile), k = 0..expansion-1 (depth sub-element)
    for (nk_size_t column_tile = 0; column_tile < column_tile_count; column_tile++) {
        for (nk_size_t depth_step = 0; depth_step < depth_step_count; depth_step++) {
            nk_size_t const vec_index = column_tile * depth_step_count + depth_step;
            nk_f16_t *vec_output = tiles_ptr + vec_index * vector_elements;

            nk_size_t const b_row_start = column_tile * tile_dimension;
            nk_size_t const depth_base = depth_step * expansion;
            nk_size_t const rows_to_pack = (b_row_start + tile_dimension <= n) ? tile_dimension : (n - b_row_start);

            for (nk_size_t column_in_tile = 0; column_in_tile < rows_to_pack; column_in_tile++) {
                for (nk_size_t sub_element = 0; sub_element < expansion; sub_element++) {
                    nk_size_t const depth_idx = depth_base + sub_element;
                    if (depth_idx < k) {
                        nk_size_t const src_idx = (b_row_start + column_in_tile) * b_stride_elements + depth_idx;
                        vec_output[expansion * column_in_tile + sub_element] = b[src_idx];
                    }
                }
            }
        }
    }
}

NK_PUBLIC void nk_dots_pack_bf16_sme(             //
    nk_bf16_t const *b, nk_size_t n, nk_size_t k, //
    nk_size_t b_stride, void *b_packed) {

    nk_size_t const expansion = 2;
    nk_size_t const tile_dimension = svcntsw();
    nk_size_t const vector_elements = svcntsh();
    nk_size_t const b_stride_elements = b_stride / sizeof(nk_bf16_t);

    nk_size_t const column_tile_count = nk_size_divide_round_up_(n, tile_dimension);
    nk_size_t const depth_step_count = nk_size_divide_round_up_(k, expansion);
    nk_size_t const total_vectors = column_tile_count * depth_step_count;

    nk_dots_sme_packed_header_t *header = (nk_dots_sme_packed_header_t *)b_packed;
    header->column_tile_count = (nk_u32_t)column_tile_count;
    header->depth_tile_count = (nk_u32_t)depth_step_count;
    header->columns = (nk_u32_t)n;
    header->depth = (nk_u32_t)k;
    header->svl_bytes = (nk_u32_t)(svcntsw() * sizeof(nk_f32_t));

    nk_bf16_t *tiles_ptr = (nk_bf16_t *)((char *)b_packed + sizeof(nk_dots_sme_packed_header_t));

    for (nk_size_t i = 0; i < total_vectors * vector_elements; i++) { tiles_ptr[i] = 0; }

    // Interleaved packing: packed_vec[expansion * d + k] = B[column_tile*tile_dim + d, depth_step*expansion + k]
    for (nk_size_t column_tile = 0; column_tile < column_tile_count; column_tile++) {
        for (nk_size_t depth_step = 0; depth_step < depth_step_count; depth_step++) {
            nk_size_t const vec_index = column_tile * depth_step_count + depth_step;
            nk_bf16_t *vec_output = tiles_ptr + vec_index * vector_elements;

            nk_size_t const b_row_start = column_tile * tile_dimension;
            nk_size_t const depth_base = depth_step * expansion;
            nk_size_t const rows_to_pack = (b_row_start + tile_dimension <= n) ? tile_dimension : (n - b_row_start);

            for (nk_size_t column_in_tile = 0; column_in_tile < rows_to_pack; column_in_tile++) {
                for (nk_size_t sub_element = 0; sub_element < expansion; sub_element++) {
                    nk_size_t const depth_idx = depth_base + sub_element;
                    if (depth_idx < k) {
                        nk_size_t const src_idx = (b_row_start + column_in_tile) * b_stride_elements + depth_idx;
                        vec_output[expansion * column_in_tile + sub_element] = b[src_idx];
                    }
                }
            }
        }
    }
}

/**
 *  `f16` → `f32` GEMM core kernel using SME outer products.
 *
 *  FMOPA f16 → f32 semantics: ZA[s][d] += Σ(k=0..1) Zn[2s+k] * Zm[2d+k]
 *  So Zn is interpreted as a 16×2 sub-matrix and Zm as a 2×16 sub-matrix.
 *
 *  For correct GEMM C[i][j] = Σ_k A[i][k]*B[j][k]:
 *  - Zn[2*s+k] = A[row_start+s, depth_base+k]  (gather 2 depth elements per A row)
 *  - Zm[2*d+k] = B[column_start+d, depth_base+k]   (pre-packed interleaved)
 *  - Loop over depth in steps of 2 (expansion factor)
 */
__arm_locally_streaming __arm_new("za") static void nk_dots_packed_f16_sme_kernel_( //
    nk_f16_t const *a, void const *b_packed, nk_f32_t *c,                           //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,                             //
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) {

    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_size_t const column_tile_count = header->column_tile_count;
    nk_size_t const depth_step_count = header->depth_tile_count;

    nk_size_t const expansion = 2;
    nk_size_t const tile_dimension = svcntw();              // 16: ZA32 tile dimension
    nk_size_t const vector_elements = svcnth();             // 32: f16 elements per SVE vector
    nk_size_t const depth_steps_per_batch = tile_dimension; // 16 depth steps per ZA0 load

    nk_f16_t const *b_packed_base = (nk_f16_t const *)((char const *)b_packed + sizeof(nk_dots_sme_packed_header_t));

    svbool_t const full_predicate_b16 = svptrue_b16();
    svbool_t const full_predicate_b32 = svptrue_b32();

    // ZA tile assignment:
    //   ZA0.S = A data staging (horizontal load, vertical read)
    //   ZA1.S, ZA2.S, ZA3.S = MOPA accumulation (3 column tiles simultaneously)

    nk_size_t const row_tile_count = nk_size_divide_round_up_(rows, tile_dimension);

    for (nk_size_t row_tile_index = 0; row_tile_index < row_tile_count; row_tile_index++) {
        nk_size_t const row_start = row_tile_index * tile_dimension;
        nk_size_t const rows_remaining = (row_start + tile_dimension <= rows) ? tile_dimension : (rows - row_start);
        svbool_t const row_predicate_b16 = svwhilelt_b16((uint32_t)0, (uint32_t)(rows_remaining * expansion));
        svbool_t const row_predicate_b32 = svwhilelt_b32((uint32_t)0, (uint32_t)rows_remaining);

        nk_size_t column_tile_index = 0;

        // Fast path: 3 column tiles at a time using ZA1-ZA3 (ZA0 = staging)
        for (; column_tile_index + 3 <= column_tile_count; column_tile_index += 3) {
            svzero_mask_za(nk_sme_zero_za32_tiles_123_); // zero accumulators only

            for (nk_size_t depth_batch_start = 0; depth_batch_start < depth_step_count;
                 depth_batch_start += depth_steps_per_batch) {
                nk_size_t const depth_batch_end = (depth_batch_start + depth_steps_per_batch < depth_step_count)
                                                      ? depth_batch_start + depth_steps_per_batch
                                                      : depth_step_count;
                nk_size_t const batch_size = depth_batch_end - depth_batch_start;

                svzero_mask_za(nk_sme_zero_za32_tile_0_); // clear staging ZA0 only

                // Load A rows into ZA0.S horizontally (f32 words = 2 packed f16 each)
                svbool_t const batch_predicate_b32 = svwhilelt_b32((uint32_t)0, (uint32_t)batch_size);
                for (nk_size_t row_in_tile = 0; row_in_tile < rows_remaining; row_in_tile++) {
                    nk_size_t const a_row = row_start + row_in_tile;
                    nk_f32_t const *a_row_ptr = (nk_f32_t const *)(a + a_row * a_stride_elements +
                                                                   depth_batch_start * expansion);
                    svld1_hor_za32(0, row_in_tile, batch_predicate_b32, a_row_ptr);
                }

                // For each depth step in batch: vertical read = transpose = perfect interleaving
                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {

                    svfloat32_t a_column_f32 = svread_ver_za32_f32_m(svdup_f32(0.0f), row_predicate_b32, 0, depth_step);
                    svfloat16_t a_interleaved_vector_f16 = svreinterpret_f16_f32(a_column_f32);

                    // Load B vectors for 3 column tiles
                    nk_f16_t const *b_packed_ptr_0 = b_packed_base + ((column_tile_index + 0) * depth_step_count +
                                                                      depth_batch_start + depth_step) *
                                                                         vector_elements;
                    nk_f16_t const *b_packed_ptr_1 = b_packed_base + ((column_tile_index + 1) * depth_step_count +
                                                                      depth_batch_start + depth_step) *
                                                                         vector_elements;
                    nk_f16_t const *b_packed_ptr_2 = b_packed_base + ((column_tile_index + 2) * depth_step_count +
                                                                      depth_batch_start + depth_step) *
                                                                         vector_elements;
                    svfloat16_t b_packed_vector_0 = svld1_f16(full_predicate_b16, (float16_t const *)b_packed_ptr_0);
                    svfloat16_t b_packed_vector_1 = svld1_f16(full_predicate_b16, (float16_t const *)b_packed_ptr_1);
                    svfloat16_t b_packed_vector_2 = svld1_f16(full_predicate_b16, (float16_t const *)b_packed_ptr_2);

                    // 3 MOPA calls into ZA1, ZA2, ZA3
                    svmopa_za32_f16_m(1, row_predicate_b16, full_predicate_b16, a_interleaved_vector_f16,
                                      b_packed_vector_0);
                    svmopa_za32_f16_m(2, row_predicate_b16, full_predicate_b16, a_interleaved_vector_f16,
                                      b_packed_vector_1);
                    svmopa_za32_f16_m(3, row_predicate_b16, full_predicate_b16, a_interleaved_vector_f16,
                                      b_packed_vector_2);
                }
            }

            // Extract from ZA1, ZA2, ZA3 (accumulated across ALL batches)
            for (nk_size_t row = 0; row < rows_remaining; row++) {
                nk_size_t const cs0 = (column_tile_index + 0) * tile_dimension;
                nk_size_t const cs1 = (column_tile_index + 1) * tile_dimension;
                nk_size_t const cs2 = (column_tile_index + 2) * tile_dimension;
                svst1_hor_za32(1, row, full_predicate_b32, c + (row_start + row) * c_stride_elements + cs0);
                svst1_hor_za32(2, row, full_predicate_b32, c + (row_start + row) * c_stride_elements + cs1);
                svst1_hor_za32(3, row, full_predicate_b32, c + (row_start + row) * c_stride_elements + cs2);
            }
        }

        // Remainder: 1 column tile at a time using ZA1 only
        for (; column_tile_index < column_tile_count; column_tile_index++) {
            nk_size_t const column_start = column_tile_index * tile_dimension;
            nk_size_t const cols_remaining = (column_start + tile_dimension <= columns) ? tile_dimension
                                                                                        : (columns - column_start);
            svbool_t const column_predicate_b32 = svwhilelt_b32((uint32_t)0, (uint32_t)cols_remaining);
            svbool_t const column_predicate_b16 = svwhilelt_b16((uint32_t)0, (uint32_t)(cols_remaining * expansion));

            svzero_mask_za(nk_sme_zero_za32_tile_1_); // zero ZA1 accumulator only

            for (nk_size_t depth_batch_start = 0; depth_batch_start < depth_step_count;
                 depth_batch_start += depth_steps_per_batch) {
                nk_size_t const depth_batch_end = (depth_batch_start + depth_steps_per_batch < depth_step_count)
                                                      ? depth_batch_start + depth_steps_per_batch
                                                      : depth_step_count;
                nk_size_t const batch_size = depth_batch_end - depth_batch_start;

                svzero_mask_za(nk_sme_zero_za32_tile_0_); // clear staging ZA0

                svbool_t const batch_predicate_b32 = svwhilelt_b32((uint32_t)0, (uint32_t)batch_size);
                for (nk_size_t row_in_tile = 0; row_in_tile < rows_remaining; row_in_tile++) {
                    nk_size_t const a_row = row_start + row_in_tile;
                    nk_f32_t const *a_row_ptr = (nk_f32_t const *)(a + a_row * a_stride_elements +
                                                                   depth_batch_start * expansion);
                    svld1_hor_za32(0, row_in_tile, batch_predicate_b32, a_row_ptr);
                }

                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {

                    svfloat32_t a_column_f32 = svread_ver_za32_f32_m(svdup_f32(0.0f), row_predicate_b32, 0, depth_step);
                    svfloat16_t a_interleaved_vector_f16 = svreinterpret_f16_f32(a_column_f32);

                    nk_f16_t const *b_packed_ptr = b_packed_base + (column_tile_index * depth_step_count +
                                                                    depth_batch_start + depth_step) *
                                                                       vector_elements;
                    svfloat16_t b_packed_vector = svld1_f16(column_predicate_b16, (float16_t const *)b_packed_ptr);

                    svmopa_za32_f16_m(1, row_predicate_b16, column_predicate_b16, a_interleaved_vector_f16,
                                      b_packed_vector);
                }
            }

            for (nk_size_t row = 0; row < rows_remaining; row++) {
                svst1_hor_za32(1, row, column_predicate_b32, c + (row_start + row) * c_stride_elements + column_start);
            }
        }
    }
}

/**
 *  `bf16` → `f32` GEMM core kernel using SME outer products.
 *  Same interleaved algorithm as f16 kernel, using BFMOPA bf16 → f32.
 */
__arm_locally_streaming __arm_new("za") static void nk_dots_packed_bf16_sme_kernel_( //
    nk_bf16_t const *a, void const *b_packed, nk_f32_t *c,                           //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,                              //
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) {

    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_size_t const column_tile_count = header->column_tile_count;
    nk_size_t const depth_step_count = header->depth_tile_count;

    nk_size_t const expansion = 2;
    nk_size_t const tile_dimension = svcntw();
    nk_size_t const vector_elements = svcnth();
    nk_size_t const depth_steps_per_batch = tile_dimension;

    nk_bf16_t const *b_packed_base = (nk_bf16_t const *)((char const *)b_packed + sizeof(nk_dots_sme_packed_header_t));

    svbool_t const full_predicate_b16 = svptrue_b16();
    svbool_t const full_predicate_b32 = svptrue_b32();

    nk_size_t const row_tile_count = nk_size_divide_round_up_(rows, tile_dimension);

    for (nk_size_t row_tile_index = 0; row_tile_index < row_tile_count; row_tile_index++) {
        nk_size_t const row_start = row_tile_index * tile_dimension;
        nk_size_t const rows_remaining = (row_start + tile_dimension <= rows) ? tile_dimension : (rows - row_start);
        svbool_t const row_predicate_b16 = svwhilelt_b16((uint32_t)0, (uint32_t)(rows_remaining * expansion));
        svbool_t const row_predicate_b32 = svwhilelt_b32((uint32_t)0, (uint32_t)rows_remaining);

        nk_size_t column_tile_index = 0;

        // Fast path: 3 column tiles using ZA1-ZA3 (ZA0 = staging)
        for (; column_tile_index + 3 <= column_tile_count; column_tile_index += 3) {
            svzero_mask_za(nk_sme_zero_za32_tiles_123_);

            for (nk_size_t depth_batch_start = 0; depth_batch_start < depth_step_count;
                 depth_batch_start += depth_steps_per_batch) {
                nk_size_t const depth_batch_end = (depth_batch_start + depth_steps_per_batch < depth_step_count)
                                                      ? depth_batch_start + depth_steps_per_batch
                                                      : depth_step_count;
                nk_size_t const batch_size = depth_batch_end - depth_batch_start;

                svzero_mask_za(nk_sme_zero_za32_tile_0_);

                svbool_t const batch_predicate_b32 = svwhilelt_b32((uint32_t)0, (uint32_t)batch_size);
                for (nk_size_t row_in_tile = 0; row_in_tile < rows_remaining; row_in_tile++) {
                    nk_size_t const a_row = row_start + row_in_tile;
                    nk_f32_t const *a_row_ptr = (nk_f32_t const *)(a + a_row * a_stride_elements +
                                                                   depth_batch_start * expansion);
                    svld1_hor_za32(0, row_in_tile, batch_predicate_b32, a_row_ptr);
                }

                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {

                    svfloat32_t a_column_f32 = svread_ver_za32_f32_m(svdup_f32(0.0f), row_predicate_b32, 0, depth_step);
                    svbfloat16_t a_interleaved_vector_bf16 = svreinterpret_bf16_f32(a_column_f32);

                    nk_bf16_t const *b_packed_ptr_0 = b_packed_base + ((column_tile_index + 0) * depth_step_count +
                                                                       depth_batch_start + depth_step) *
                                                                          vector_elements;
                    nk_bf16_t const *b_packed_ptr_1 = b_packed_base + ((column_tile_index + 1) * depth_step_count +
                                                                       depth_batch_start + depth_step) *
                                                                          vector_elements;
                    nk_bf16_t const *b_packed_ptr_2 = b_packed_base + ((column_tile_index + 2) * depth_step_count +
                                                                       depth_batch_start + depth_step) *
                                                                          vector_elements;
                    svbfloat16_t b_packed_vector_0 = svld1_bf16(full_predicate_b16, (bfloat16_t const *)b_packed_ptr_0);
                    svbfloat16_t b_packed_vector_1 = svld1_bf16(full_predicate_b16, (bfloat16_t const *)b_packed_ptr_1);
                    svbfloat16_t b_packed_vector_2 = svld1_bf16(full_predicate_b16, (bfloat16_t const *)b_packed_ptr_2);

                    svmopa_za32_bf16_m(1, row_predicate_b16, full_predicate_b16, a_interleaved_vector_bf16,
                                       b_packed_vector_0);
                    svmopa_za32_bf16_m(2, row_predicate_b16, full_predicate_b16, a_interleaved_vector_bf16,
                                       b_packed_vector_1);
                    svmopa_za32_bf16_m(3, row_predicate_b16, full_predicate_b16, a_interleaved_vector_bf16,
                                       b_packed_vector_2);
                }
            }

            for (nk_size_t row = 0; row < rows_remaining; row++) {
                nk_size_t const cs0 = (column_tile_index + 0) * tile_dimension;
                nk_size_t const cs1 = (column_tile_index + 1) * tile_dimension;
                nk_size_t const cs2 = (column_tile_index + 2) * tile_dimension;
                svst1_hor_za32(1, row, full_predicate_b32, c + (row_start + row) * c_stride_elements + cs0);
                svst1_hor_za32(2, row, full_predicate_b32, c + (row_start + row) * c_stride_elements + cs1);
                svst1_hor_za32(3, row, full_predicate_b32, c + (row_start + row) * c_stride_elements + cs2);
            }
        }

        // Remainder: 1 column tile using ZA1
        for (; column_tile_index < column_tile_count; column_tile_index++) {
            nk_size_t const column_start = column_tile_index * tile_dimension;
            nk_size_t const cols_remaining = (column_start + tile_dimension <= columns) ? tile_dimension
                                                                                        : (columns - column_start);
            svbool_t const column_predicate_b32 = svwhilelt_b32((uint32_t)0, (uint32_t)cols_remaining);
            svbool_t const column_predicate_b16 = svwhilelt_b16((uint32_t)0, (uint32_t)(cols_remaining * expansion));

            svzero_mask_za(nk_sme_zero_za32_tile_1_);

            for (nk_size_t depth_batch_start = 0; depth_batch_start < depth_step_count;
                 depth_batch_start += depth_steps_per_batch) {
                nk_size_t const depth_batch_end = (depth_batch_start + depth_steps_per_batch < depth_step_count)
                                                      ? depth_batch_start + depth_steps_per_batch
                                                      : depth_step_count;
                nk_size_t const batch_size = depth_batch_end - depth_batch_start;

                svzero_mask_za(nk_sme_zero_za32_tile_0_);

                svbool_t const batch_predicate_b32 = svwhilelt_b32((uint32_t)0, (uint32_t)batch_size);
                for (nk_size_t row_in_tile = 0; row_in_tile < rows_remaining; row_in_tile++) {
                    nk_size_t const a_row = row_start + row_in_tile;
                    nk_f32_t const *a_row_ptr = (nk_f32_t const *)(a + a_row * a_stride_elements +
                                                                   depth_batch_start * expansion);
                    svld1_hor_za32(0, row_in_tile, batch_predicate_b32, a_row_ptr);
                }

                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {

                    svfloat32_t a_column_f32 = svread_ver_za32_f32_m(svdup_f32(0.0f), row_predicate_b32, 0, depth_step);
                    svbfloat16_t a_interleaved_vector_bf16 = svreinterpret_bf16_f32(a_column_f32);

                    nk_bf16_t const *b_packed_ptr = b_packed_base + (column_tile_index * depth_step_count +
                                                                     depth_batch_start + depth_step) *
                                                                        vector_elements;
                    svbfloat16_t b_packed_vector = svld1_bf16(column_predicate_b16, (bfloat16_t const *)b_packed_ptr);

                    svmopa_za32_bf16_m(1, row_predicate_b16, column_predicate_b16, a_interleaved_vector_bf16,
                                       b_packed_vector);
                }
            }

            for (nk_size_t row = 0; row < rows_remaining; row++) {
                svst1_hor_za32(1, row, column_predicate_b32, c + (row_start + row) * c_stride_elements + column_start);
            }
        }
    }
}

NK_PUBLIC void nk_dots_packed_f16_sme(                    //
    nk_f16_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,   //
    nk_size_t a_stride, nk_size_t c_stride) {

    nk_size_t const a_stride_elements = a_stride / sizeof(nk_f16_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);

    nk_dots_packed_f16_sme_kernel_(a, b_packed, c, rows, columns, depth, a_stride_elements, c_stride_elements);
}

NK_PUBLIC void nk_dots_packed_bf16_sme(                    //
    nk_bf16_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,    //
    nk_size_t a_stride, nk_size_t c_stride) {

    nk_size_t const a_stride_elements = a_stride / sizeof(nk_bf16_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);

    nk_dots_packed_bf16_sme_kernel_(a, b_packed, c, rows, columns, depth, a_stride_elements, c_stride_elements);
}

/**
 *   `f16` × `f16` → `f32` symmetric kernel using MOPA self-GEMM.
 *   Time-shares ZA0 for both A and B transposition: loads A horizontally,
 *   pre-reads A columns into Z registers, then reloads ZA0 with B data
 *   per column tile. Eliminates all scalar B-packing loops.
 */
__arm_locally_streaming __arm_new("za") static void nk_dots_symmetric_f16_sme_kernel_(
    nk_f16_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride_elements, nk_f32_t *result,
    nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) {

    nk_size_t const expansion = 2;
    nk_size_t const tile_dimension = svcntw(); // 16
    nk_size_t const depth_step_count = nk_size_divide_round_up_(depth, expansion);
    nk_size_t const depth_steps_per_batch = tile_dimension; // 16

    svbool_t const full_predicate_b16 = svptrue_b16();
    svbool_t const full_predicate_b32 = svptrue_b32();

    NK_ALIGN64 nk_f32_t a_buffer[16][16];

    nk_size_t const row_end = row_start + row_count;
    nk_size_t const column_tile_count = nk_size_divide_round_up_(n_vectors, tile_dimension);

    for (nk_size_t row_tile_start = row_start; row_tile_start < row_end && row_tile_start < n_vectors;
         row_tile_start += tile_dimension) {
        nk_size_t const rows_remaining = (row_tile_start + tile_dimension <= row_end) ? tile_dimension
                                                                                      : (row_end - row_tile_start);
        nk_size_t const rows_clamped = (row_tile_start + rows_remaining <= n_vectors) ? rows_remaining
                                                                                      : (n_vectors - row_tile_start);
        svbool_t const row_predicate_b16 = svwhilelt_b16((uint32_t)0, (uint32_t)(rows_clamped * expansion));
        svbool_t const row_predicate_b32 = svwhilelt_b32((uint32_t)0, (uint32_t)rows_clamped);

        nk_size_t column_tile_index = 0;

        // Fast path: 3 column tiles at a time
        for (; column_tile_index + 3 <= column_tile_count; column_tile_index += 3) {
            svzero_mask_za(nk_sme_zero_za32_tiles_123_);

            for (nk_size_t depth_batch_start = 0; depth_batch_start < depth_step_count;
                 depth_batch_start += depth_steps_per_batch) {
                nk_size_t const depth_batch_end = (depth_batch_start + depth_steps_per_batch < depth_step_count)
                                                      ? depth_batch_start + depth_steps_per_batch
                                                      : depth_step_count;
                nk_size_t const batch_size = depth_batch_end - depth_batch_start;

                // ZA transpose for A rows (identical to packed kernel)
                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                svbool_t const batch_predicate_b32 = svwhilelt_b32((uint32_t)0, (uint32_t)batch_size);
                for (nk_size_t row_in_tile = 0; row_in_tile < rows_clamped; row_in_tile++) {
                    nk_size_t const row_abs = row_tile_start + row_in_tile;
                    nk_f32_t const *a_row_ptr = (nk_f32_t const *)(vectors + row_abs * stride_elements +
                                                                   depth_batch_start * expansion);
                    svld1_hor_za32(0, row_in_tile, batch_predicate_b32, a_row_ptr);
                }

                // Save A columns from ZA0 to stack buffer
                for (nk_size_t s = 0; s < batch_size; s++)
                    svst1_f32(full_predicate_b32, a_buffer[s],
                              svread_ver_za32_f32_m(svdup_f32(0), row_predicate_b32, 0, s));

                // Load B column tile 0 into ZA0, vertical read + FMOPA into ZA1
                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t column = 0; column < tile_dimension; column++) {
                    nk_size_t const column_abs = (column_tile_index + 0) * tile_dimension + column;
                    if (column_abs < n_vectors) {
                        nk_f32_t const *b_row = (nk_f32_t const *)(vectors + column_abs * stride_elements +
                                                                   depth_batch_start * expansion);
                        svld1_hor_za32(0, column, batch_predicate_b32, b_row);
                    }
                }
                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {
                    svfloat16_t a_interleaved = svreinterpret_f16_f32(
                        svld1_f32(full_predicate_b32, a_buffer[depth_step]));
                    svfloat32_t b_column_f32 = svread_ver_za32_f32_m(svdup_f32(0.0f), full_predicate_b32, 0,
                                                                     depth_step);
                    svfloat16_t b_interleaved = svreinterpret_f16_f32(b_column_f32);
                    svmopa_za32_f16_m(1, row_predicate_b16, full_predicate_b16, a_interleaved, b_interleaved);
                }

                // Load B column tile 1 into ZA0, vertical read + FMOPA into ZA2
                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t column = 0; column < tile_dimension; column++) {
                    nk_size_t const column_abs = (column_tile_index + 1) * tile_dimension + column;
                    if (column_abs < n_vectors) {
                        nk_f32_t const *b_row = (nk_f32_t const *)(vectors + column_abs * stride_elements +
                                                                   depth_batch_start * expansion);
                        svld1_hor_za32(0, column, batch_predicate_b32, b_row);
                    }
                }
                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {
                    svfloat16_t a_interleaved = svreinterpret_f16_f32(
                        svld1_f32(full_predicate_b32, a_buffer[depth_step]));
                    svfloat32_t b_column_f32 = svread_ver_za32_f32_m(svdup_f32(0.0f), full_predicate_b32, 0,
                                                                     depth_step);
                    svfloat16_t b_interleaved = svreinterpret_f16_f32(b_column_f32);
                    svmopa_za32_f16_m(2, row_predicate_b16, full_predicate_b16, a_interleaved, b_interleaved);
                }

                // Load B column tile 2 into ZA0, vertical read + FMOPA into ZA3
                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t column = 0; column < tile_dimension; column++) {
                    nk_size_t const column_abs = (column_tile_index + 2) * tile_dimension + column;
                    if (column_abs < n_vectors) {
                        nk_f32_t const *b_row = (nk_f32_t const *)(vectors + column_abs * stride_elements +
                                                                   depth_batch_start * expansion);
                        svld1_hor_za32(0, column, batch_predicate_b32, b_row);
                    }
                }
                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {
                    svfloat16_t a_interleaved = svreinterpret_f16_f32(
                        svld1_f32(full_predicate_b32, a_buffer[depth_step]));
                    svfloat32_t b_column_f32 = svread_ver_za32_f32_m(svdup_f32(0.0f), full_predicate_b32, 0,
                                                                     depth_step);
                    svfloat16_t b_interleaved = svreinterpret_f16_f32(b_column_f32);
                    svmopa_za32_f16_m(3, row_predicate_b16, full_predicate_b16, a_interleaved, b_interleaved);
                }
            }

            // Extract results from ZA1-3
            for (nk_size_t row = 0; row < rows_clamped; row++) {
                nk_size_t const row_abs = row_tile_start + row;
                nk_f32_t *result_row = result + row_abs * result_stride_elements;
                svst1_hor_za32(1, row, full_predicate_b32, result_row + (column_tile_index + 0) * tile_dimension);
                svst1_hor_za32(2, row, full_predicate_b32, result_row + (column_tile_index + 1) * tile_dimension);
                svst1_hor_za32(3, row, full_predicate_b32, result_row + (column_tile_index + 2) * tile_dimension);
            }
        }

        // Remainder: 1 column tile at a time
        for (; column_tile_index < column_tile_count; column_tile_index++) {
            nk_size_t const column_tile_start = column_tile_index * tile_dimension;
            nk_size_t const cols_remaining = (column_tile_start + tile_dimension <= n_vectors)
                                                 ? tile_dimension
                                                 : (n_vectors - column_tile_start);
            svbool_t const column_predicate_b32 = svwhilelt_b32((uint32_t)0, (uint32_t)cols_remaining);
            svbool_t const column_predicate_b16 = svwhilelt_b16((uint32_t)0, (uint32_t)(cols_remaining * expansion));

            svzero_mask_za(nk_sme_zero_za32_tile_1_);

            for (nk_size_t depth_batch_start = 0; depth_batch_start < depth_step_count;
                 depth_batch_start += depth_steps_per_batch) {
                nk_size_t const depth_batch_end = (depth_batch_start + depth_steps_per_batch < depth_step_count)
                                                      ? depth_batch_start + depth_steps_per_batch
                                                      : depth_step_count;
                nk_size_t const batch_size = depth_batch_end - depth_batch_start;

                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                svbool_t const batch_predicate_b32 = svwhilelt_b32((uint32_t)0, (uint32_t)batch_size);
                for (nk_size_t row_in_tile = 0; row_in_tile < rows_clamped; row_in_tile++) {
                    nk_size_t const row_abs = row_tile_start + row_in_tile;
                    nk_f32_t const *a_row_ptr = (nk_f32_t const *)(vectors + row_abs * stride_elements +
                                                                   depth_batch_start * expansion);
                    svld1_hor_za32(0, row_in_tile, batch_predicate_b32, a_row_ptr);
                }

                // Save A columns from ZA0 to stack buffer
                for (nk_size_t s = 0; s < batch_size; s++)
                    svst1_f32(full_predicate_b32, a_buffer[s],
                              svread_ver_za32_f32_m(svdup_f32(0), row_predicate_b32, 0, s));

                // Load B column tile into ZA0, vertical read + FMOPA into ZA1
                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t column = 0; column < tile_dimension; column++) {
                    nk_size_t const column_abs = column_tile_start + column;
                    if (column_abs < n_vectors) {
                        nk_f32_t const *b_row = (nk_f32_t const *)(vectors + column_abs * stride_elements +
                                                                   depth_batch_start * expansion);
                        svld1_hor_za32(0, column, batch_predicate_b32, b_row);
                    }
                }
                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {
                    svfloat16_t a_interleaved = svreinterpret_f16_f32(
                        svld1_f32(full_predicate_b32, a_buffer[depth_step]));
                    svfloat32_t b_column_f32 = svread_ver_za32_f32_m(svdup_f32(0.0f), column_predicate_b32, 0,
                                                                     depth_step);
                    svfloat16_t b_interleaved = svreinterpret_f16_f32(b_column_f32);
                    svmopa_za32_f16_m(1, row_predicate_b16, column_predicate_b16, a_interleaved, b_interleaved);
                }
            }

            for (nk_size_t row = 0; row < rows_clamped; row++) {
                nk_size_t const row_abs = row_tile_start + row;
                svst1_hor_za32(1, row, column_predicate_b32,
                               result + row_abs * result_stride_elements + column_tile_start);
            }
        }

        // Mirror: result[j][i] = result[i][j] for j in [row_tile_start, row_tile_start + rows_clamped)
        for (nk_size_t row = 0; row < rows_clamped; row++) {
            nk_size_t const i = row_tile_start + row;
            for (nk_size_t j = 0; j < i && j < n_vectors; j++) {
                result[j * result_stride_elements + i] = result[i * result_stride_elements + j];
            }
        }
    }
}

NK_PUBLIC void nk_dots_symmetric_f16_sme(nk_f16_t const *vectors, nk_size_t n_vectors, nk_size_t depth,
                                         nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                         nk_size_t row_start, nk_size_t row_count) {

    nk_size_t const stride_elements = stride / sizeof(nk_f16_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_dots_symmetric_f16_sme_kernel_(vectors, n_vectors, depth, stride_elements, result, result_stride_elements,
                                      row_start, row_count);
}

__arm_locally_streaming __arm_new("za") static void nk_dots_symmetric_bf16_sme_kernel_(
    nk_bf16_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride_elements, nk_f32_t *result,
    nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) {

    nk_size_t const expansion = 2;
    nk_size_t const tile_dimension = svcntw();
    nk_size_t const depth_step_count = nk_size_divide_round_up_(depth, expansion);
    nk_size_t const depth_steps_per_batch = tile_dimension;

    svbool_t const full_predicate_b16 = svptrue_b16();
    svbool_t const full_predicate_b32 = svptrue_b32();

    NK_ALIGN64 nk_f32_t a_buffer[16][16];

    nk_size_t const row_end = row_start + row_count;
    nk_size_t const column_tile_count = nk_size_divide_round_up_(n_vectors, tile_dimension);

    for (nk_size_t row_tile_start = row_start; row_tile_start < row_end && row_tile_start < n_vectors;
         row_tile_start += tile_dimension) {
        nk_size_t const rows_remaining = (row_tile_start + tile_dimension <= row_end) ? tile_dimension
                                                                                      : (row_end - row_tile_start);
        nk_size_t const rows_clamped = (row_tile_start + rows_remaining <= n_vectors) ? rows_remaining
                                                                                      : (n_vectors - row_tile_start);
        svbool_t const row_predicate_b16 = svwhilelt_b16((uint32_t)0, (uint32_t)(rows_clamped * expansion));
        svbool_t const row_predicate_b32 = svwhilelt_b32((uint32_t)0, (uint32_t)rows_clamped);

        nk_size_t column_tile_index = 0;

        for (; column_tile_index + 3 <= column_tile_count; column_tile_index += 3) {
            svzero_mask_za(nk_sme_zero_za32_tiles_123_);

            for (nk_size_t depth_batch_start = 0; depth_batch_start < depth_step_count;
                 depth_batch_start += depth_steps_per_batch) {
                nk_size_t const depth_batch_end = (depth_batch_start + depth_steps_per_batch < depth_step_count)
                                                      ? depth_batch_start + depth_steps_per_batch
                                                      : depth_step_count;
                nk_size_t const batch_size = depth_batch_end - depth_batch_start;

                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                svbool_t const batch_predicate_b32 = svwhilelt_b32((uint32_t)0, (uint32_t)batch_size);
                for (nk_size_t row_in_tile = 0; row_in_tile < rows_clamped; row_in_tile++) {
                    nk_size_t const row_abs = row_tile_start + row_in_tile;
                    nk_f32_t const *a_row_ptr = (nk_f32_t const *)(vectors + row_abs * stride_elements +
                                                                   depth_batch_start * expansion);
                    svld1_hor_za32(0, row_in_tile, batch_predicate_b32, a_row_ptr);
                }

                for (nk_size_t s = 0; s < batch_size; s++)
                    svst1_f32(full_predicate_b32, a_buffer[s],
                              svread_ver_za32_f32_m(svdup_f32(0), row_predicate_b32, 0, s));

                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t column = 0; column < tile_dimension; column++) {
                    nk_size_t const column_abs = (column_tile_index + 0) * tile_dimension + column;
                    if (column_abs < n_vectors) {
                        nk_f32_t const *b_row = (nk_f32_t const *)(vectors + column_abs * stride_elements +
                                                                   depth_batch_start * expansion);
                        svld1_hor_za32(0, column, batch_predicate_b32, b_row);
                    }
                }
                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {
                    svbfloat16_t a_interleaved = svreinterpret_bf16_f32(
                        svld1_f32(full_predicate_b32, a_buffer[depth_step]));
                    svfloat32_t b_column_f32 = svread_ver_za32_f32_m(svdup_f32(0.0f), full_predicate_b32, 0,
                                                                     depth_step);
                    svbfloat16_t b_interleaved = svreinterpret_bf16_f32(b_column_f32);
                    svmopa_za32_bf16_m(1, row_predicate_b16, full_predicate_b16, a_interleaved, b_interleaved);
                }

                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t column = 0; column < tile_dimension; column++) {
                    nk_size_t const column_abs = (column_tile_index + 1) * tile_dimension + column;
                    if (column_abs < n_vectors) {
                        nk_f32_t const *b_row = (nk_f32_t const *)(vectors + column_abs * stride_elements +
                                                                   depth_batch_start * expansion);
                        svld1_hor_za32(0, column, batch_predicate_b32, b_row);
                    }
                }
                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {
                    svbfloat16_t a_interleaved = svreinterpret_bf16_f32(
                        svld1_f32(full_predicate_b32, a_buffer[depth_step]));
                    svfloat32_t b_column_f32 = svread_ver_za32_f32_m(svdup_f32(0.0f), full_predicate_b32, 0,
                                                                     depth_step);
                    svbfloat16_t b_interleaved = svreinterpret_bf16_f32(b_column_f32);
                    svmopa_za32_bf16_m(2, row_predicate_b16, full_predicate_b16, a_interleaved, b_interleaved);
                }

                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t column = 0; column < tile_dimension; column++) {
                    nk_size_t const column_abs = (column_tile_index + 2) * tile_dimension + column;
                    if (column_abs < n_vectors) {
                        nk_f32_t const *b_row = (nk_f32_t const *)(vectors + column_abs * stride_elements +
                                                                   depth_batch_start * expansion);
                        svld1_hor_za32(0, column, batch_predicate_b32, b_row);
                    }
                }
                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {
                    svbfloat16_t a_interleaved = svreinterpret_bf16_f32(
                        svld1_f32(full_predicate_b32, a_buffer[depth_step]));
                    svfloat32_t b_column_f32 = svread_ver_za32_f32_m(svdup_f32(0.0f), full_predicate_b32, 0,
                                                                     depth_step);
                    svbfloat16_t b_interleaved = svreinterpret_bf16_f32(b_column_f32);
                    svmopa_za32_bf16_m(3, row_predicate_b16, full_predicate_b16, a_interleaved, b_interleaved);
                }
            }

            for (nk_size_t row = 0; row < rows_clamped; row++) {
                nk_size_t const row_abs = row_tile_start + row;
                nk_f32_t *result_row = result + row_abs * result_stride_elements;
                svst1_hor_za32(1, row, full_predicate_b32, result_row + (column_tile_index + 0) * tile_dimension);
                svst1_hor_za32(2, row, full_predicate_b32, result_row + (column_tile_index + 1) * tile_dimension);
                svst1_hor_za32(3, row, full_predicate_b32, result_row + (column_tile_index + 2) * tile_dimension);
            }
        }

        for (; column_tile_index < column_tile_count; column_tile_index++) {
            nk_size_t const column_tile_start = column_tile_index * tile_dimension;
            nk_size_t const cols_remaining = (column_tile_start + tile_dimension <= n_vectors)
                                                 ? tile_dimension
                                                 : (n_vectors - column_tile_start);
            svbool_t const column_predicate_b32 = svwhilelt_b32((uint32_t)0, (uint32_t)cols_remaining);
            svbool_t const column_predicate_b16 = svwhilelt_b16((uint32_t)0, (uint32_t)(cols_remaining * expansion));

            svzero_mask_za(nk_sme_zero_za32_tile_1_);

            for (nk_size_t depth_batch_start = 0; depth_batch_start < depth_step_count;
                 depth_batch_start += depth_steps_per_batch) {
                nk_size_t const depth_batch_end = (depth_batch_start + depth_steps_per_batch < depth_step_count)
                                                      ? depth_batch_start + depth_steps_per_batch
                                                      : depth_step_count;
                nk_size_t const batch_size = depth_batch_end - depth_batch_start;

                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                svbool_t const batch_predicate_b32 = svwhilelt_b32((uint32_t)0, (uint32_t)batch_size);
                for (nk_size_t row_in_tile = 0; row_in_tile < rows_clamped; row_in_tile++) {
                    nk_size_t const row_abs = row_tile_start + row_in_tile;
                    nk_f32_t const *a_row_ptr = (nk_f32_t const *)(vectors + row_abs * stride_elements +
                                                                   depth_batch_start * expansion);
                    svld1_hor_za32(0, row_in_tile, batch_predicate_b32, a_row_ptr);
                }

                for (nk_size_t s = 0; s < batch_size; s++)
                    svst1_f32(full_predicate_b32, a_buffer[s],
                              svread_ver_za32_f32_m(svdup_f32(0), row_predicate_b32, 0, s));

                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t column = 0; column < tile_dimension; column++) {
                    nk_size_t const column_abs = column_tile_start + column;
                    if (column_abs < n_vectors) {
                        nk_f32_t const *b_row = (nk_f32_t const *)(vectors + column_abs * stride_elements +
                                                                   depth_batch_start * expansion);
                        svld1_hor_za32(0, column, batch_predicate_b32, b_row);
                    }
                }
                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {
                    svbfloat16_t a_interleaved = svreinterpret_bf16_f32(
                        svld1_f32(full_predicate_b32, a_buffer[depth_step]));
                    svfloat32_t b_column_f32 = svread_ver_za32_f32_m(svdup_f32(0.0f), column_predicate_b32, 0,
                                                                     depth_step);
                    svbfloat16_t b_interleaved = svreinterpret_bf16_f32(b_column_f32);
                    svmopa_za32_bf16_m(1, row_predicate_b16, column_predicate_b16, a_interleaved, b_interleaved);
                }
            }

            for (nk_size_t row = 0; row < rows_clamped; row++) {
                nk_size_t const row_abs = row_tile_start + row;
                svst1_hor_za32(1, row, column_predicate_b32,
                               result + row_abs * result_stride_elements + column_tile_start);
            }
        }

        for (nk_size_t row = 0; row < rows_clamped; row++) {
            nk_size_t const i = row_tile_start + row;
            for (nk_size_t j = 0; j < i && j < n_vectors; j++) {
                result[j * result_stride_elements + i] = result[i * result_stride_elements + j];
            }
        }
    }
}

NK_PUBLIC void nk_dots_symmetric_bf16_sme(nk_bf16_t const *vectors, nk_size_t n_vectors, nk_size_t depth,
                                          nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                          nk_size_t row_start, nk_size_t row_count) {

    nk_size_t const stride_elements = stride / sizeof(nk_bf16_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_dots_symmetric_bf16_sme_kernel_(vectors, n_vectors, depth, stride_elements, result, result_stride_elements,
                                       row_start, row_count);
}

#pragma endregion

/*
 *  `i8` × `i8` → `i32` GEMM using SME outer products.
 *
 *  Uses `svmopa_za32_s8_m` for signed 8-bit integer outer product accumulate.
 *  Available on Apple M4 (SME_I8I32 = 1).
 *
 *  Tile dimensions for `i8` → `i32` (512-bit SVL):
 *  - Input vectors: 64 `i8` elements (SVL/8 = 64)
 *  - Output tile: 16 × 16 `i32` elements (`ZA32`)
 *  - Each output `i32` is a dot product of 4 `i8` pairs
 *
 *  Expected performance: ~2 TOPS (4× `f16` due to 4:1 element packing)
 */

#pragma region Signed Integers

NK_PUBLIC nk_size_t nk_dots_packed_size_i8_sme(nk_size_t n, nk_size_t k) {
    nk_size_t const expansion = 4;               // SMOPA i8→i32: 4 i8 pairs per i32 output
    nk_size_t const tile_dimension = svcntsw();  // ZA32 tile dim: 16
    nk_size_t const vector_elements = svcntsb(); // i8 elements per SVE vector: 64
    nk_size_t const column_tile_count = nk_size_divide_round_up_(n, tile_dimension);
    nk_size_t const depth_step_count = nk_size_divide_round_up_(k, expansion);

    nk_size_t size = sizeof(nk_dots_sme_packed_header_t);
    size += column_tile_count * depth_step_count * vector_elements * sizeof(nk_i8_t);
    return size;
}

NK_PUBLIC void nk_dots_pack_i8_sme(             //
    nk_i8_t const *b, nk_size_t n, nk_size_t k, //
    nk_size_t b_stride, void *b_packed) {

    nk_size_t const expansion = 4;
    nk_size_t const tile_dimension = svcntsw();
    nk_size_t const vector_elements = svcntsb(); // 64 = tile_dimension * expansion
    nk_size_t const b_stride_elements = b_stride / sizeof(nk_i8_t);

    nk_size_t const column_tile_count = nk_size_divide_round_up_(n, tile_dimension);
    nk_size_t const depth_step_count = nk_size_divide_round_up_(k, expansion);
    nk_size_t const total_vectors = column_tile_count * depth_step_count;

    nk_dots_sme_packed_header_t *header = (nk_dots_sme_packed_header_t *)b_packed;
    header->column_tile_count = (nk_u32_t)column_tile_count;
    header->depth_tile_count = (nk_u32_t)depth_step_count;
    header->columns = (nk_u32_t)n;
    header->depth = (nk_u32_t)k;
    header->svl_bytes = (nk_u32_t)(svcntsw() * sizeof(nk_i32_t));

    nk_i8_t *tiles_ptr = (nk_i8_t *)((char *)b_packed + sizeof(nk_dots_sme_packed_header_t));

    for (nk_size_t i = 0; i < total_vectors * vector_elements; i++) { tiles_ptr[i] = 0; }

    // Interleaved packing: packed_vec[expansion * d + k] = B[column_tile*tile_dim + d, depth_step*expansion + k]
    for (nk_size_t column_tile = 0; column_tile < column_tile_count; column_tile++) {
        for (nk_size_t depth_step = 0; depth_step < depth_step_count; depth_step++) {
            nk_size_t const vec_index = column_tile * depth_step_count + depth_step;
            nk_i8_t *vec_output = tiles_ptr + vec_index * vector_elements;

            nk_size_t const b_row_start = column_tile * tile_dimension;
            nk_size_t const depth_base = depth_step * expansion;
            nk_size_t const rows_to_pack = (b_row_start + tile_dimension <= n) ? tile_dimension : (n - b_row_start);

            for (nk_size_t column_in_tile = 0; column_in_tile < rows_to_pack; column_in_tile++) {
                for (nk_size_t sub_element = 0; sub_element < expansion; sub_element++) {
                    nk_size_t const depth_idx = depth_base + sub_element;
                    if (depth_idx < k) {
                        nk_size_t const src_idx = (b_row_start + column_in_tile) * b_stride_elements + depth_idx;
                        vec_output[expansion * column_in_tile + sub_element] = b[src_idx];
                    }
                }
            }
        }
    }
}

/**
 *  `i8` × `i8` → `i32` GEMM core kernel using SME outer products.
 *
 *  SMOPA i8→i32 semantics: ZA[s][d] += Σ(k=0..3) Zn[4s+k] * Zm[4d+k]
 *  So Zn is interpreted as a 16×4 sub-matrix and Zm as a 4×16 sub-matrix.
 *
 *  For correct GEMM C[i][j] = Σ_k A[i][k]*B[j][k]:
 *  - Zn[4*s+k] = A[row_start+s, depth_base+k]  (gather 4 depth elements per A row)
 *  - Zm[4*d+k] = B[column_start+d, depth_base+k]   (pre-packed interleaved)
 *  - Loop over depth in steps of 4 (expansion factor)
 */
__arm_locally_streaming __arm_new("za") static void nk_dots_packed_i8_sme_kernel_( //
    nk_i8_t const *a, void const *b_packed, nk_i32_t *c,                           //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,                            //
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) {

    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_size_t const column_tile_count = header->column_tile_count;
    nk_size_t const depth_step_count = header->depth_tile_count;

    nk_size_t const expansion = 4;
    nk_size_t const tile_dimension = svcntw();              // 16: ZA32 tile dimension
    nk_size_t const vector_elements = svcntb();             // 64: i8 elements per SVE vector
    nk_size_t const depth_steps_per_batch = tile_dimension; // 16 steps = 64 depth elements per ZA0 load

    nk_i8_t const *b_packed_base = (nk_i8_t const *)((char const *)b_packed + sizeof(nk_dots_sme_packed_header_t));

    svbool_t const full_predicate_b8 = svptrue_b8();
    svbool_t const full_predicate_b32 = svptrue_b32();

    nk_size_t const row_tile_count = nk_size_divide_round_up_(rows, tile_dimension);

    for (nk_size_t row_tile_index = 0; row_tile_index < row_tile_count; row_tile_index++) {
        nk_size_t const row_start = row_tile_index * tile_dimension;
        nk_size_t const rows_remaining = (row_start + tile_dimension <= rows) ? tile_dimension : (rows - row_start);
        svbool_t const row_predicate_b8 = svwhilelt_b8((uint32_t)0, (uint32_t)(rows_remaining * expansion));
        svbool_t const row_predicate_b32 = svwhilelt_b32((uint32_t)0, (uint32_t)rows_remaining);

        nk_size_t column_tile_index = 0;

        // Fast path: 3 column tiles using ZA1-ZA3 (ZA0 = staging)
        for (; column_tile_index + 3 <= column_tile_count; column_tile_index += 3) {
            svzero_mask_za(nk_sme_zero_za32_tiles_123_);

            for (nk_size_t depth_batch_start = 0; depth_batch_start < depth_step_count;
                 depth_batch_start += depth_steps_per_batch) {
                nk_size_t const depth_batch_end = (depth_batch_start + depth_steps_per_batch < depth_step_count)
                                                      ? depth_batch_start + depth_steps_per_batch
                                                      : depth_step_count;
                nk_size_t const batch_size = depth_batch_end - depth_batch_start;

                svzero_mask_za(nk_sme_zero_za32_tile_0_);

                // Load A rows into ZA0.S (each f32 word = 4 packed i8 bytes)
                svbool_t const batch_predicate_b32 = svwhilelt_b32((uint32_t)0, (uint32_t)batch_size);
                for (nk_size_t row_in_tile = 0; row_in_tile < rows_remaining; row_in_tile++) {
                    nk_size_t const a_row = row_start + row_in_tile;
                    nk_f32_t const *a_row_ptr = (nk_f32_t const *)(a + a_row * a_stride_elements +
                                                                   depth_batch_start * expansion);
                    svld1_hor_za32(0, row_in_tile, batch_predicate_b32, a_row_ptr);
                }

                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {

                    // Vertical read at f32 granularity produces [row0_k0..k3, row1_k0..k3, ...]
                    svint32_t a_column_i32 = svread_ver_za32_s32_m(svdup_s32(0), row_predicate_b32, 0, depth_step);
                    svint8_t a_interleaved_vector_i8 = svreinterpret_s8_s32(a_column_i32);

                    nk_i8_t const *b_packed_ptr_0 = b_packed_base + ((column_tile_index + 0) * depth_step_count +
                                                                     depth_batch_start + depth_step) *
                                                                        vector_elements;
                    nk_i8_t const *b_packed_ptr_1 = b_packed_base + ((column_tile_index + 1) * depth_step_count +
                                                                     depth_batch_start + depth_step) *
                                                                        vector_elements;
                    nk_i8_t const *b_packed_ptr_2 = b_packed_base + ((column_tile_index + 2) * depth_step_count +
                                                                     depth_batch_start + depth_step) *
                                                                        vector_elements;
                    svint8_t b_packed_vector_0 = svld1_s8(full_predicate_b8, b_packed_ptr_0);
                    svint8_t b_packed_vector_1 = svld1_s8(full_predicate_b8, b_packed_ptr_1);
                    svint8_t b_packed_vector_2 = svld1_s8(full_predicate_b8, b_packed_ptr_2);

                    svmopa_za32_s8_m(1, row_predicate_b8, full_predicate_b8, a_interleaved_vector_i8,
                                     b_packed_vector_0);
                    svmopa_za32_s8_m(2, row_predicate_b8, full_predicate_b8, a_interleaved_vector_i8,
                                     b_packed_vector_1);
                    svmopa_za32_s8_m(3, row_predicate_b8, full_predicate_b8, a_interleaved_vector_i8,
                                     b_packed_vector_2);
                }
            }

            for (nk_size_t row = 0; row < rows_remaining; row++) {
                nk_size_t const cs0 = (column_tile_index + 0) * tile_dimension;
                nk_size_t const cs1 = (column_tile_index + 1) * tile_dimension;
                nk_size_t const cs2 = (column_tile_index + 2) * tile_dimension;
                svst1_hor_za32(1, row, full_predicate_b32, c + (row_start + row) * c_stride_elements + cs0);
                svst1_hor_za32(2, row, full_predicate_b32, c + (row_start + row) * c_stride_elements + cs1);
                svst1_hor_za32(3, row, full_predicate_b32, c + (row_start + row) * c_stride_elements + cs2);
            }
        }

        // Remainder: 1 column tile using ZA1
        for (; column_tile_index < column_tile_count; column_tile_index++) {
            nk_size_t const column_start = column_tile_index * tile_dimension;
            nk_size_t const cols_remaining = (column_start + tile_dimension <= columns) ? tile_dimension
                                                                                        : (columns - column_start);
            svbool_t const column_predicate_b32 = svwhilelt_b32((uint32_t)0, (uint32_t)cols_remaining);
            svbool_t const column_predicate_b8 = svwhilelt_b8((uint32_t)0, (uint32_t)(cols_remaining * expansion));

            svzero_mask_za(nk_sme_zero_za32_tile_1_);

            for (nk_size_t depth_batch_start = 0; depth_batch_start < depth_step_count;
                 depth_batch_start += depth_steps_per_batch) {
                nk_size_t const depth_batch_end = (depth_batch_start + depth_steps_per_batch < depth_step_count)
                                                      ? depth_batch_start + depth_steps_per_batch
                                                      : depth_step_count;
                nk_size_t const batch_size = depth_batch_end - depth_batch_start;

                svzero_mask_za(nk_sme_zero_za32_tile_0_);

                svbool_t const batch_predicate_b32 = svwhilelt_b32((uint32_t)0, (uint32_t)batch_size);
                for (nk_size_t row_in_tile = 0; row_in_tile < rows_remaining; row_in_tile++) {
                    nk_size_t const a_row = row_start + row_in_tile;
                    nk_f32_t const *a_row_ptr = (nk_f32_t const *)(a + a_row * a_stride_elements +
                                                                   depth_batch_start * expansion);
                    svld1_hor_za32(0, row_in_tile, batch_predicate_b32, a_row_ptr);
                }

                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {

                    svint32_t a_column_i32 = svread_ver_za32_s32_m(svdup_s32(0), row_predicate_b32, 0, depth_step);
                    svint8_t a_interleaved_vector_i8 = svreinterpret_s8_s32(a_column_i32);

                    nk_i8_t const *b_packed_ptr = b_packed_base + (column_tile_index * depth_step_count +
                                                                   depth_batch_start + depth_step) *
                                                                      vector_elements;
                    svint8_t b_packed_vector = svld1_s8(column_predicate_b8, b_packed_ptr);

                    svmopa_za32_s8_m(1, row_predicate_b8, column_predicate_b8, a_interleaved_vector_i8,
                                     b_packed_vector);
                }
            }

            for (nk_size_t row = 0; row < rows_remaining; row++) {
                svst1_hor_za32(1, row, column_predicate_b32, c + (row_start + row) * c_stride_elements + column_start);
            }
        }
    }
}

NK_PUBLIC void nk_dots_packed_i8_sme(                    //
    nk_i8_t const *a, void const *b_packed, nk_i32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,  //
    nk_size_t a_stride, nk_size_t c_stride) {

    nk_size_t const a_stride_elements = a_stride / sizeof(nk_i8_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_i32_t);

    nk_dots_packed_i8_sme_kernel_(a, b_packed, c, rows, columns, depth, a_stride_elements, c_stride_elements);
}

// expansion=4: each i32 word packs 4 i8 values
__arm_locally_streaming __arm_new("za") static void nk_dots_symmetric_i8_sme_kernel_(
    nk_i8_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride_elements, nk_i32_t *result,
    nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) {

    nk_size_t const expansion = 4;
    nk_size_t const tile_dimension = svcntw();
    nk_size_t const depth_step_count = nk_size_divide_round_up_(depth, expansion);
    nk_size_t const depth_steps_per_batch = tile_dimension;

    svbool_t const full_predicate_b8 = svptrue_b8();
    svbool_t const full_predicate_b32 = svptrue_b32();

    NK_ALIGN64 nk_i32_t a_buffer[16][16];

    nk_size_t const row_end = row_start + row_count;
    nk_size_t const column_tile_count = nk_size_divide_round_up_(n_vectors, tile_dimension);

    for (nk_size_t row_tile_start = row_start; row_tile_start < row_end && row_tile_start < n_vectors;
         row_tile_start += tile_dimension) {
        nk_size_t const rows_remaining = (row_tile_start + tile_dimension <= row_end) ? tile_dimension
                                                                                      : (row_end - row_tile_start);
        nk_size_t const rows_clamped = (row_tile_start + rows_remaining <= n_vectors) ? rows_remaining
                                                                                      : (n_vectors - row_tile_start);
        svbool_t const row_predicate_b8 = svwhilelt_b8((uint32_t)0, (uint32_t)(rows_clamped * expansion));
        svbool_t const row_predicate_b32 = svwhilelt_b32((uint32_t)0, (uint32_t)rows_clamped);

        nk_size_t column_tile_index = 0;

        for (; column_tile_index + 3 <= column_tile_count; column_tile_index += 3) {
            svzero_mask_za(nk_sme_zero_za32_tiles_123_);

            for (nk_size_t depth_batch_start = 0; depth_batch_start < depth_step_count;
                 depth_batch_start += depth_steps_per_batch) {
                nk_size_t const depth_batch_end = (depth_batch_start + depth_steps_per_batch < depth_step_count)
                                                      ? depth_batch_start + depth_steps_per_batch
                                                      : depth_step_count;
                nk_size_t const batch_size = depth_batch_end - depth_batch_start;

                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                svbool_t const batch_predicate_b32 = svwhilelt_b32((uint32_t)0, (uint32_t)batch_size);
                for (nk_size_t row_in_tile = 0; row_in_tile < rows_clamped; row_in_tile++) {
                    nk_size_t const row_abs = row_tile_start + row_in_tile;
                    nk_f32_t const *a_row_ptr = (nk_f32_t const *)(vectors + row_abs * stride_elements +
                                                                   depth_batch_start * expansion);
                    svld1_hor_za32(0, row_in_tile, batch_predicate_b32, a_row_ptr);
                }

                for (nk_size_t s = 0; s < batch_size; s++)
                    svst1_s32(full_predicate_b32, a_buffer[s],
                              svread_ver_za32_s32_m(svdup_s32(0), row_predicate_b32, 0, s));

                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t column = 0; column < tile_dimension; column++) {
                    nk_size_t const column_abs = (column_tile_index + 0) * tile_dimension + column;
                    if (column_abs < n_vectors) {
                        nk_f32_t const *b_row = (nk_f32_t const *)(vectors + column_abs * stride_elements +
                                                                   depth_batch_start * expansion);
                        svld1_hor_za32(0, column, batch_predicate_b32, b_row);
                    }
                }
                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {
                    svint8_t a_interleaved = svreinterpret_s8_s32(svld1_s32(full_predicate_b32, a_buffer[depth_step]));
                    svint32_t b_column_i32 = svread_ver_za32_s32_m(svdup_s32(0), full_predicate_b32, 0, depth_step);
                    svint8_t b_interleaved = svreinterpret_s8_s32(b_column_i32);
                    svmopa_za32_s8_m(1, row_predicate_b8, full_predicate_b8, a_interleaved, b_interleaved);
                }

                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t column = 0; column < tile_dimension; column++) {
                    nk_size_t const column_abs = (column_tile_index + 1) * tile_dimension + column;
                    if (column_abs < n_vectors) {
                        nk_f32_t const *b_row = (nk_f32_t const *)(vectors + column_abs * stride_elements +
                                                                   depth_batch_start * expansion);
                        svld1_hor_za32(0, column, batch_predicate_b32, b_row);
                    }
                }
                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {
                    svint8_t a_interleaved = svreinterpret_s8_s32(svld1_s32(full_predicate_b32, a_buffer[depth_step]));
                    svint32_t b_column_i32 = svread_ver_za32_s32_m(svdup_s32(0), full_predicate_b32, 0, depth_step);
                    svint8_t b_interleaved = svreinterpret_s8_s32(b_column_i32);
                    svmopa_za32_s8_m(2, row_predicate_b8, full_predicate_b8, a_interleaved, b_interleaved);
                }

                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t column = 0; column < tile_dimension; column++) {
                    nk_size_t const column_abs = (column_tile_index + 2) * tile_dimension + column;
                    if (column_abs < n_vectors) {
                        nk_f32_t const *b_row = (nk_f32_t const *)(vectors + column_abs * stride_elements +
                                                                   depth_batch_start * expansion);
                        svld1_hor_za32(0, column, batch_predicate_b32, b_row);
                    }
                }
                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {
                    svint8_t a_interleaved = svreinterpret_s8_s32(svld1_s32(full_predicate_b32, a_buffer[depth_step]));
                    svint32_t b_column_i32 = svread_ver_za32_s32_m(svdup_s32(0), full_predicate_b32, 0, depth_step);
                    svint8_t b_interleaved = svreinterpret_s8_s32(b_column_i32);
                    svmopa_za32_s8_m(3, row_predicate_b8, full_predicate_b8, a_interleaved, b_interleaved);
                }
            }

            for (nk_size_t row = 0; row < rows_clamped; row++) {
                nk_size_t const row_abs = row_tile_start + row;
                nk_f32_t *result_row = (nk_f32_t *)(result + row_abs * result_stride_elements);
                svst1_hor_za32(1, row, full_predicate_b32, result_row + (column_tile_index + 0) * tile_dimension);
                svst1_hor_za32(2, row, full_predicate_b32, result_row + (column_tile_index + 1) * tile_dimension);
                svst1_hor_za32(3, row, full_predicate_b32, result_row + (column_tile_index + 2) * tile_dimension);
            }
        }

        for (; column_tile_index < column_tile_count; column_tile_index++) {
            nk_size_t const column_tile_start = column_tile_index * tile_dimension;
            nk_size_t const cols_remaining = (column_tile_start + tile_dimension <= n_vectors)
                                                 ? tile_dimension
                                                 : (n_vectors - column_tile_start);
            svbool_t const column_predicate_b32 = svwhilelt_b32((uint32_t)0, (uint32_t)cols_remaining);
            svbool_t const column_predicate_b8 = svwhilelt_b8((uint32_t)0, (uint32_t)(cols_remaining * expansion));

            svzero_mask_za(nk_sme_zero_za32_tile_1_);

            for (nk_size_t depth_batch_start = 0; depth_batch_start < depth_step_count;
                 depth_batch_start += depth_steps_per_batch) {
                nk_size_t const depth_batch_end = (depth_batch_start + depth_steps_per_batch < depth_step_count)
                                                      ? depth_batch_start + depth_steps_per_batch
                                                      : depth_step_count;
                nk_size_t const batch_size = depth_batch_end - depth_batch_start;

                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                svbool_t const batch_predicate_b32 = svwhilelt_b32((uint32_t)0, (uint32_t)batch_size);
                for (nk_size_t row_in_tile = 0; row_in_tile < rows_clamped; row_in_tile++) {
                    nk_size_t const row_abs = row_tile_start + row_in_tile;
                    nk_f32_t const *a_row_ptr = (nk_f32_t const *)(vectors + row_abs * stride_elements +
                                                                   depth_batch_start * expansion);
                    svld1_hor_za32(0, row_in_tile, batch_predicate_b32, a_row_ptr);
                }

                for (nk_size_t s = 0; s < batch_size; s++)
                    svst1_s32(full_predicate_b32, a_buffer[s],
                              svread_ver_za32_s32_m(svdup_s32(0), row_predicate_b32, 0, s));

                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t column = 0; column < tile_dimension; column++) {
                    nk_size_t const column_abs = column_tile_start + column;
                    if (column_abs < n_vectors) {
                        nk_f32_t const *b_row = (nk_f32_t const *)(vectors + column_abs * stride_elements +
                                                                   depth_batch_start * expansion);
                        svld1_hor_za32(0, column, batch_predicate_b32, b_row);
                    }
                }
                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {
                    svint8_t a_interleaved = svreinterpret_s8_s32(svld1_s32(full_predicate_b32, a_buffer[depth_step]));
                    svint32_t b_column_i32 = svread_ver_za32_s32_m(svdup_s32(0), column_predicate_b32, 0, depth_step);
                    svint8_t b_interleaved = svreinterpret_s8_s32(b_column_i32);
                    svmopa_za32_s8_m(1, row_predicate_b8, column_predicate_b8, a_interleaved, b_interleaved);
                }
            }

            for (nk_size_t row = 0; row < rows_clamped; row++) {
                nk_size_t const row_abs = row_tile_start + row;
                svst1_hor_za32(1, row, column_predicate_b32,
                               (nk_f32_t *)(result + row_abs * result_stride_elements) + column_tile_start);
            }
        }

        for (nk_size_t row = 0; row < rows_clamped; row++) {
            nk_size_t const i = row_tile_start + row;
            for (nk_size_t j = 0; j < i && j < n_vectors; j++) {
                result[j * result_stride_elements + i] = result[i * result_stride_elements + j];
            }
        }
    }
}

NK_PUBLIC void nk_dots_symmetric_i8_sme(nk_i8_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride,
                                        nk_i32_t *result, nk_size_t result_stride, nk_size_t row_start,
                                        nk_size_t row_count) {

    nk_size_t const stride_elements = stride / sizeof(nk_i8_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_i32_t);
    nk_dots_symmetric_i8_sme_kernel_(vectors, n_vectors, depth, stride_elements, result, result_stride_elements,
                                     row_start, row_count);
}

#pragma endregion

/*
 *  e4m3 × e4m3 → f32 GEMM using inline SSVE conversion + FMOPA.
 *
 *  Pipeline: e4m3 bytes → svunpklo → arithmetic → f16 → FMOPA → f32
 *  - Load: 64 bytes via svld1_u8, convert lower 32 to f16 inline
 *  - Accumulate: FMOPA f16 → f32 into ZA32 tiles
 *  - No memory round-trip for format conversion
 *  - FMOPA predicates: b16 (f16 input granularity)
 */
#pragma region Quarter Precision E4M3

/**
 *  E4M3 subnormal → F16 lookup table (8 entries).
 *  Index by 3-bit mantissa (0..7) to get the F16 bit pattern for mant/512.
 *  For indices >= 8 (normal values), SVE TBL returns 0 which is ignored by SEL.
 */
static nk_u16_t const nk_e4m3_subnorm_f16_lut_[8] = {
    0x0000, // mant=0: 0.0
    0x1800, // mant=1: 1/512
    0x1C00, // mant=2: 2/512
    0x1E00, // mant=3: 3/512
    0x2000, // mant=4: 4/512
    0x2100, // mant=5: 5/512
    0x2200, // mant=6: 6/512
    0x2300, // mant=7: 7/512
};

/**
 *  Inline `e4m3` → `f16` conversion returning `svfloat16_t` for direct use in GEMM.
 *  This avoids memory round-trip when used inside a streaming kernel.
 *
 *  Uses an 8-entry TBL lookup for subnormals instead of UCVTF+FMUL, saving 3 instructions
 *  (18 → 15). The `subnorm_lut` parameter should be loaded once per kernel entry via:
 *    svuint16_t subnorm_lut = svld1_u16(svwhilelt_b16(0u, 8u), nk_e4m3_subnorm_f16_lut_);
 *
 *  @param pg16 Predicate for 16-bit elements: use `svptrue_b16()`
 *  @param bytes Pre-loaded 64 bytes: `svuint8_t` from `svld1_u8`
 *  @param subnorm_lut Pre-loaded 8-entry subnormal LUT as `svuint16_t`
 *  @return 32 `f16` values as `svfloat16_t`: from lower 32 bytes
 */
NK_INTERNAL svfloat16_t nk_e4m3x_to_f16x_ssve_(svbool_t pg16, svuint8_t bytes, svuint16_t subnorm_lut) {
    svuint16_t vals = svunpklo_u16(bytes); // 1: UUNPKLO

    svuint16_t sign = svlsl_n_u16_x(pg16, svand_n_u16_x(pg16, vals, 0x80), 8); // 2-3: AND+LSL
    svuint16_t mag = svand_n_u16_x(pg16, vals, 0x7F);                          // 4: AND

    // Normal path: sign | ((mag << 7) + 0x2000)
    svuint16_t normal = svadd_n_u16_x(pg16, svlsl_n_u16_x(pg16, mag, 7), 0x2000); // 5-6: LSL+ADD
    normal = svorr_u16_x(pg16, normal, sign);                                     // 7: ORR

    // Subnormal path: 8-entry TBL (mag 0..7 for subnorms; mag>7 → TBL returns 0, ignored by SEL)
    svuint16_t subnorm = svorr_u16_x(pg16, svtbl_u16(subnorm_lut, mag), sign); // 8-9: TBL+ORR

    svbool_t is_subnorm = svcmpeq_n_u16(pg16, svand_n_u16_x(pg16, vals, 0x78), 0); // 10-11: AND+CMPEQ
    // NaN path: mag == 0x7F → f16 NaN
    svbool_t is_nan = svcmpeq_n_u16(pg16, mag, 0x7F);           // 12: CMPEQ
    svuint16_t nan_val = svorr_n_u16_x(pg16, sign, 0x7E00);     // 13: ORR
    svuint16_t result = svsel_u16(is_subnorm, subnorm, normal); // 14: SEL
    result = svsel_u16(is_nan, nan_val, result);                // 15: SEL

    return svreinterpret_f16_u16(result);
}

/**
 *  Inline `e5m2` → `f16` conversion returning `svfloat16_t` for direct use in GEMM.
 *  This avoids memory round-trip when used inside a streaming kernel.
 *
 *  E5M2 format: S EEEEE MM (1+5+2 bits, bias=15, range [-57344, 57344])
 *  F16 format:  S EEEEE MMMMMMMMMM (1+5+10 bits, bias=15)
 *
 *  Since E5M2 and F16 share the same exponent bias (15), normal values convert
 *  by simply shifting the magnitude left by 8 bits.
 *
 *  @param pg16 Predicate for 16-bit elements (use svptrue_b16())
 *  @param bytes Pre-loaded 64 bytes (svuint8_t from svld1_u8)
 *  @return 32 F16 values as svfloat16_t (from lower 32 bytes)
 */
NK_INTERNAL svfloat16_t nk_e5m2x_to_f16x_ssve_(svbool_t pg16, svuint8_t bytes) {
    // E5M2 and F16 share the same exponent bias (15), sign position, exponent width,
    // and mantissa field alignment. The conversion f16 = byte << 8 is exact for ALL
    // 256 values including subnormals, infinity, and NaN.
    return svreinterpret_f16_u16(svlsl_n_u16_x(pg16, svunpklo_u16(bytes), 8));
}

/**
 *  Fused `e4m3` × `e4m3` → `f32` GEMM kernel using interleaved FMOPA.
 *  Converts `e4m3` → `f16` on-the-fly for A, B is pre-converted during packing.
 */
__arm_locally_streaming __arm_new("za") __attribute__((noinline)) static void nk_dots_packed_e4m3_sme_kernel_( //
    nk_e4m3_t const *a, void const *b_packed, nk_f32_t *c,                                                     //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,                                                        //
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) {

    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_size_t const column_tile_count = header->column_tile_count;
    nk_size_t const depth_step_count = header->depth_tile_count;

    nk_size_t const expansion = 2;
    nk_size_t const tile_dimension = svcntw();
    nk_size_t const vector_elements = svcnth();
    nk_size_t const depth_steps_per_batch = tile_dimension;

    nk_f16_t const *b_packed_base = (nk_f16_t const *)((char const *)b_packed + sizeof(nk_dots_sme_packed_header_t));

    svbool_t const full_predicate_b16 = svptrue_b16();
    svbool_t const full_predicate_b8 = svptrue_b8();
    svbool_t const full_predicate_b32 = svptrue_b32();

    // Load e4m3 subnormal LUT once for all conversions in this kernel
    svuint16_t const subnorm_lut = svld1_u16(svwhilelt_b16(0u, 8u), nk_e4m3_subnorm_f16_lut_);

    // Stack buffer for e4m3 → f16 conversion (32 f16 per row, up to 16 rows)
    NK_ALIGN64 nk_f16_t a_converted_f16x32[16][32];

    nk_size_t const row_tile_count = nk_size_divide_round_up_(rows, tile_dimension);

    for (nk_size_t row_tile_index = 0; row_tile_index < row_tile_count; row_tile_index++) {
        nk_size_t const row_start = row_tile_index * tile_dimension;
        nk_size_t const rows_remaining = (row_start + tile_dimension <= rows) ? tile_dimension : (rows - row_start);
        svbool_t const row_predicate_b16 = svwhilelt_b16((uint32_t)0, (uint32_t)(rows_remaining * expansion));
        svbool_t const row_predicate_b32 = svwhilelt_b32((uint32_t)0, (uint32_t)rows_remaining);

        nk_size_t column_tile_index = 0;

        // Fast path: 3 column tiles using ZA1-ZA3 (ZA0 = staging)
        for (; column_tile_index + 3 <= column_tile_count; column_tile_index += 3) {
            svzero_mask_za(nk_sme_zero_za32_tiles_123_);

            for (nk_size_t depth_batch_start = 0; depth_batch_start < depth_step_count;
                 depth_batch_start += depth_steps_per_batch) {
                nk_size_t const depth_batch_end = (depth_batch_start + depth_steps_per_batch < depth_step_count)
                                                      ? depth_batch_start + depth_steps_per_batch
                                                      : depth_step_count;
                nk_size_t const batch_size = depth_batch_end - depth_batch_start;

                // Convert e4m3 → f16 for each A row in this batch, then load into ZA0
                svzero_mask_za(nk_sme_zero_za32_tile_0_);

                svbool_t const batch_predicate_b32 = svwhilelt_b32((uint32_t)0, (uint32_t)batch_size);
                for (nk_size_t row_in_tile = 0; row_in_tile < rows_remaining; row_in_tile++) {
                    nk_size_t const a_row = row_start + row_in_tile;
                    // Load raw e4m3 bytes and convert to f16 using vectorized conversion
                    nk_e4m3_t const *a_src = a + a_row * a_stride_elements + depth_batch_start * expansion;
                    svuint8_t raw_bytes = svld1_u8(full_predicate_b8, (uint8_t const *)a_src);
                    svfloat16_t converted_f16 = nk_e4m3x_to_f16x_ssve_(full_predicate_b16, raw_bytes, subnorm_lut);
                    svst1_f16(full_predicate_b16, (float16_t *)a_converted_f16x32[row_in_tile], converted_f16);
                    // Load converted f16 row into ZA0 as f32 words
                    svld1_hor_za32(0, row_in_tile, batch_predicate_b32,
                                   (nk_f32_t const *)a_converted_f16x32[row_in_tile]);
                }

                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {

                    svfloat32_t a_column_f32 = svread_ver_za32_f32_m(svdup_f32(0.0f), row_predicate_b32, 0, depth_step);
                    svfloat16_t a_interleaved_vector_f16 = svreinterpret_f16_f32(a_column_f32);

                    nk_f16_t const *b_packed_ptr_0 = b_packed_base + ((column_tile_index + 0) * depth_step_count +
                                                                      depth_batch_start + depth_step) *
                                                                         vector_elements;
                    nk_f16_t const *b_packed_ptr_1 = b_packed_base + ((column_tile_index + 1) * depth_step_count +
                                                                      depth_batch_start + depth_step) *
                                                                         vector_elements;
                    nk_f16_t const *b_packed_ptr_2 = b_packed_base + ((column_tile_index + 2) * depth_step_count +
                                                                      depth_batch_start + depth_step) *
                                                                         vector_elements;
                    svfloat16_t b_packed_vector_0 = svld1_f16(full_predicate_b16, (float16_t const *)b_packed_ptr_0);
                    svfloat16_t b_packed_vector_1 = svld1_f16(full_predicate_b16, (float16_t const *)b_packed_ptr_1);
                    svfloat16_t b_packed_vector_2 = svld1_f16(full_predicate_b16, (float16_t const *)b_packed_ptr_2);

                    svmopa_za32_f16_m(1, row_predicate_b16, full_predicate_b16, a_interleaved_vector_f16,
                                      b_packed_vector_0);
                    svmopa_za32_f16_m(2, row_predicate_b16, full_predicate_b16, a_interleaved_vector_f16,
                                      b_packed_vector_1);
                    svmopa_za32_f16_m(3, row_predicate_b16, full_predicate_b16, a_interleaved_vector_f16,
                                      b_packed_vector_2);
                }
            }

            for (nk_size_t row = 0; row < rows_remaining; row++) {
                nk_size_t const cs0 = (column_tile_index + 0) * tile_dimension;
                nk_size_t const cs1 = (column_tile_index + 1) * tile_dimension;
                nk_size_t const cs2 = (column_tile_index + 2) * tile_dimension;
                svst1_hor_za32(1, row, full_predicate_b32, c + (row_start + row) * c_stride_elements + cs0);
                svst1_hor_za32(2, row, full_predicate_b32, c + (row_start + row) * c_stride_elements + cs1);
                svst1_hor_za32(3, row, full_predicate_b32, c + (row_start + row) * c_stride_elements + cs2);
            }
        }

        // Remainder: 1 column tile using ZA1
        for (; column_tile_index < column_tile_count; column_tile_index++) {
            nk_size_t const column_start = column_tile_index * tile_dimension;
            nk_size_t const cols_remaining = (column_start + tile_dimension <= columns) ? tile_dimension
                                                                                        : (columns - column_start);
            svbool_t const column_predicate_b32 = svwhilelt_b32((uint32_t)0, (uint32_t)cols_remaining);
            svbool_t const column_predicate_b16 = svwhilelt_b16((uint32_t)0, (uint32_t)(cols_remaining * expansion));

            svzero_mask_za(nk_sme_zero_za32_tile_1_);

            for (nk_size_t depth_batch_start = 0; depth_batch_start < depth_step_count;
                 depth_batch_start += depth_steps_per_batch) {
                nk_size_t const depth_batch_end = (depth_batch_start + depth_steps_per_batch < depth_step_count)
                                                      ? depth_batch_start + depth_steps_per_batch
                                                      : depth_step_count;
                nk_size_t const batch_size = depth_batch_end - depth_batch_start;

                svzero_mask_za(nk_sme_zero_za32_tile_0_);

                svbool_t const batch_predicate_b32 = svwhilelt_b32((uint32_t)0, (uint32_t)batch_size);
                for (nk_size_t row_in_tile = 0; row_in_tile < rows_remaining; row_in_tile++) {
                    nk_size_t const a_row = row_start + row_in_tile;
                    nk_e4m3_t const *a_src = a + a_row * a_stride_elements + depth_batch_start * expansion;
                    svuint8_t raw_bytes = svld1_u8(full_predicate_b8, (uint8_t const *)a_src);
                    svfloat16_t converted_f16 = nk_e4m3x_to_f16x_ssve_(full_predicate_b16, raw_bytes, subnorm_lut);
                    svst1_f16(full_predicate_b16, (float16_t *)a_converted_f16x32[row_in_tile], converted_f16);
                    svld1_hor_za32(0, row_in_tile, batch_predicate_b32,
                                   (nk_f32_t const *)a_converted_f16x32[row_in_tile]);
                }

                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {

                    svfloat32_t a_column_f32 = svread_ver_za32_f32_m(svdup_f32(0.0f), row_predicate_b32, 0, depth_step);
                    svfloat16_t a_interleaved_vector_f16 = svreinterpret_f16_f32(a_column_f32);

                    nk_f16_t const *b_packed_ptr = b_packed_base + (column_tile_index * depth_step_count +
                                                                    depth_batch_start + depth_step) *
                                                                       vector_elements;
                    svfloat16_t b_packed_vector = svld1_f16(column_predicate_b16, (float16_t const *)b_packed_ptr);

                    svmopa_za32_f16_m(1, row_predicate_b16, column_predicate_b16, a_interleaved_vector_f16,
                                      b_packed_vector);
                }
            }

            for (nk_size_t row = 0; row < rows_remaining; row++) {
                svst1_hor_za32(1, row, column_predicate_b32, c + (row_start + row) * c_stride_elements + column_start);
            }
        }
    }
}

NK_PUBLIC nk_size_t nk_dots_packed_size_e4m3_sme(nk_size_t n, nk_size_t k) {
    // Uses `f16` format for packed data
    return nk_dots_packed_size_f16_sme(n, k);
}

NK_PUBLIC void nk_dots_pack_e4m3_sme(             //
    nk_e4m3_t const *b, nk_size_t n, nk_size_t k, //
    nk_size_t b_stride, void *b_packed) {

    nk_size_t const expansion = 2;
    nk_size_t const tile_dimension = svcntsw();
    nk_size_t const vector_elements = svcntsh();
    nk_size_t const b_stride_elements = b_stride / sizeof(nk_e4m3_t);

    nk_size_t const column_tile_count = nk_size_divide_round_up_(n, tile_dimension);
    nk_size_t const depth_step_count = nk_size_divide_round_up_(k, expansion);
    nk_size_t const total_vectors = column_tile_count * depth_step_count;

    nk_dots_sme_packed_header_t *header = (nk_dots_sme_packed_header_t *)b_packed;
    header->column_tile_count = (nk_u32_t)column_tile_count;
    header->depth_tile_count = (nk_u32_t)depth_step_count;
    header->columns = (nk_u32_t)n;
    header->depth = (nk_u32_t)k;
    header->svl_bytes = (nk_u32_t)(svcntsw() * sizeof(nk_f32_t));

    nk_f16_t *tiles_ptr = (nk_f16_t *)((char *)b_packed + sizeof(nk_dots_sme_packed_header_t));

    for (nk_size_t i = 0; i < total_vectors * vector_elements; i++) tiles_ptr[i] = 0;

    // Interleaved packing with e4m3 → f16 conversion
    for (nk_size_t column_tile = 0; column_tile < column_tile_count; column_tile++) {
        for (nk_size_t depth_step = 0; depth_step < depth_step_count; depth_step++) {
            nk_size_t const vec_index = column_tile * depth_step_count + depth_step;
            nk_f16_t *vec_output = tiles_ptr + vec_index * vector_elements;

            nk_size_t const b_row_start = column_tile * tile_dimension;
            nk_size_t const depth_base = depth_step * expansion;
            nk_size_t const rows_to_pack = (b_row_start + tile_dimension <= n) ? tile_dimension : (n - b_row_start);

            for (nk_size_t column_in_tile = 0; column_in_tile < rows_to_pack; column_in_tile++) {
                for (nk_size_t sub_element = 0; sub_element < expansion; sub_element++) {
                    nk_size_t const depth_idx = depth_base + sub_element;
                    if (depth_idx < k) {
                        nk_size_t const src_idx = (b_row_start + column_in_tile) * b_stride_elements + depth_idx;
                        nk_e4m3_to_f16_serial(&b[src_idx], &vec_output[expansion * column_in_tile + sub_element]);
                    }
                }
            }
        }
    }
}

NK_PUBLIC void nk_dots_packed_e4m3_sme(                    //
    nk_e4m3_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,    //
    nk_size_t a_stride, nk_size_t c_stride) {

    nk_size_t const a_stride_elements = a_stride / sizeof(nk_e4m3_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);

    nk_dots_packed_e4m3_sme_kernel_(a, b_packed, c, rows, columns, depth, a_stride_elements, c_stride_elements);
}

/**
 * `e4m3` × `e4m3` → `f32` symmetric kernel using MOPA self-GEMM.
 *  Time-shares ZA0 for both A and B transposition with e4m3 → f16 conversion.
 *  Pre-reads A columns into Z registers, then reloads ZA0 with converted B data
 *  per column tile. Eliminates all scalar B-packing loops.
 */
__arm_locally_streaming __arm_new("za") static void nk_dots_symmetric_e4m3_sme_kernel_(
    nk_e4m3_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride_elements, nk_f32_t *result,
    nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) {

    nk_size_t const expansion = 2;
    nk_size_t const tile_dimension = svcntw();
    nk_size_t const depth_step_count = nk_size_divide_round_up_(depth, expansion);
    nk_size_t const depth_steps_per_batch = tile_dimension;

    svbool_t const full_predicate_b16 = svptrue_b16();
    svbool_t const full_predicate_b8 = svptrue_b8();
    svbool_t const full_predicate_b32 = svptrue_b32();

    // Load e4m3 subnormal LUT once for all conversions in this kernel
    svuint16_t const subnorm_lut = svld1_u16(svwhilelt_b16(0u, 8u), nk_e4m3_subnorm_f16_lut_);

    NK_ALIGN64 nk_f16_t a_converted_f16x32[16][32];
    NK_ALIGN64 nk_f16_t b_bounce_f16x32[32];
    NK_ALIGN64 nk_f32_t a_buffer[16][16];

    nk_size_t const row_end = row_start + row_count;
    nk_size_t const column_tile_count = nk_size_divide_round_up_(n_vectors, tile_dimension);

    for (nk_size_t row_tile_start = row_start; row_tile_start < row_end && row_tile_start < n_vectors;
         row_tile_start += tile_dimension) {
        nk_size_t const rows_clamped = (row_tile_start + tile_dimension <= row_end) ? tile_dimension
                                                                                    : (row_end - row_tile_start);
        nk_size_t const rows_actual = (row_tile_start + rows_clamped <= n_vectors) ? rows_clamped
                                                                                   : (n_vectors - row_tile_start);
        svbool_t const row_predicate_b16 = svwhilelt_b16((uint32_t)0, (uint32_t)(rows_actual * expansion));
        svbool_t const row_predicate_b32 = svwhilelt_b32((uint32_t)0, (uint32_t)rows_actual);

        nk_size_t column_tile_index = 0;

        for (; column_tile_index + 3 <= column_tile_count; column_tile_index += 3) {
            svzero_mask_za(nk_sme_zero_za32_tiles_123_);

            for (nk_size_t depth_batch_start = 0; depth_batch_start < depth_step_count;
                 depth_batch_start += depth_steps_per_batch) {
                nk_size_t const depth_batch_end = (depth_batch_start + depth_steps_per_batch < depth_step_count)
                                                      ? depth_batch_start + depth_steps_per_batch
                                                      : depth_step_count;
                nk_size_t const batch_size = depth_batch_end - depth_batch_start;

                // ZA transpose for A rows: convert e4m3 → f16 then load
                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                svbool_t const batch_predicate_b32 = svwhilelt_b32((uint32_t)0, (uint32_t)batch_size);
                for (nk_size_t row_in_tile = 0; row_in_tile < rows_actual; row_in_tile++) {
                    nk_size_t const row_abs = row_tile_start + row_in_tile;
                    nk_e4m3_t const *a_src = vectors + row_abs * stride_elements + depth_batch_start * expansion;
                    svuint8_t raw_bytes = svld1_u8(full_predicate_b8, (uint8_t const *)a_src);
                    svfloat16_t converted_f16 = nk_e4m3x_to_f16x_ssve_(full_predicate_b16, raw_bytes, subnorm_lut);
                    svst1_f16(full_predicate_b16, (float16_t *)a_converted_f16x32[row_in_tile], converted_f16);
                    svld1_hor_za32(0, row_in_tile, batch_predicate_b32,
                                   (nk_f32_t const *)a_converted_f16x32[row_in_tile]);
                }

                // Save A columns from ZA0 to stack buffer
                for (nk_size_t s = 0; s < batch_size; s++)
                    svst1_f32(full_predicate_b32, a_buffer[s],
                              svread_ver_za32_f32_m(svdup_f32(0), row_predicate_b32, 0, s));

                svbool_t const depth_predicate_b8 = svwhilelt_b8((uint32_t)0, (uint32_t)(batch_size * expansion));
                svbool_t const depth_predicate_b16 = svwhilelt_b16((uint32_t)0, (uint32_t)(batch_size * expansion));

                // Load B column tile 0 into ZA0, vertical read + FMOPA into ZA1
                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t column = 0; column < tile_dimension; column++) {
                    nk_size_t const column_abs = (column_tile_index + 0) * tile_dimension + column;
                    if (column_abs < n_vectors) {
                        nk_e4m3_t const *src = &vectors[column_abs * stride_elements + depth_batch_start * expansion];
                        svuint8_t raw = svld1_u8(depth_predicate_b8, (uint8_t const *)src);
                        svfloat16_t converted_f16 = nk_e4m3x_to_f16x_ssve_(depth_predicate_b16, raw, subnorm_lut);
                        svst1_f16(depth_predicate_b16, (float16_t *)b_bounce_f16x32, converted_f16);
                        svld1_hor_za32(0, column, batch_predicate_b32, (nk_f32_t const *)b_bounce_f16x32);
                    }
                }
                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {
                    svfloat16_t a_interleaved = svreinterpret_f16_f32(
                        svld1_f32(full_predicate_b32, a_buffer[depth_step]));
                    svfloat32_t b_column_f32 = svread_ver_za32_f32_m(svdup_f32(0.0f), full_predicate_b32, 0,
                                                                     depth_step);
                    svfloat16_t b_interleaved = svreinterpret_f16_f32(b_column_f32);
                    svmopa_za32_f16_m(1, row_predicate_b16, full_predicate_b16, a_interleaved, b_interleaved);
                }

                // Load B column tile 1 into ZA0, vertical read + FMOPA into ZA2
                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t column = 0; column < tile_dimension; column++) {
                    nk_size_t const column_abs = (column_tile_index + 1) * tile_dimension + column;
                    if (column_abs < n_vectors) {
                        nk_e4m3_t const *src = &vectors[column_abs * stride_elements + depth_batch_start * expansion];
                        svuint8_t raw = svld1_u8(depth_predicate_b8, (uint8_t const *)src);
                        svfloat16_t converted_f16 = nk_e4m3x_to_f16x_ssve_(depth_predicate_b16, raw, subnorm_lut);
                        svst1_f16(depth_predicate_b16, (float16_t *)b_bounce_f16x32, converted_f16);
                        svld1_hor_za32(0, column, batch_predicate_b32, (nk_f32_t const *)b_bounce_f16x32);
                    }
                }
                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {
                    svfloat16_t a_interleaved = svreinterpret_f16_f32(
                        svld1_f32(full_predicate_b32, a_buffer[depth_step]));
                    svfloat32_t b_column_f32 = svread_ver_za32_f32_m(svdup_f32(0.0f), full_predicate_b32, 0,
                                                                     depth_step);
                    svfloat16_t b_interleaved = svreinterpret_f16_f32(b_column_f32);
                    svmopa_za32_f16_m(2, row_predicate_b16, full_predicate_b16, a_interleaved, b_interleaved);
                }

                // Load B column tile 2 into ZA0, vertical read + FMOPA into ZA3
                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t column = 0; column < tile_dimension; column++) {
                    nk_size_t const column_abs = (column_tile_index + 2) * tile_dimension + column;
                    if (column_abs < n_vectors) {
                        nk_e4m3_t const *src = &vectors[column_abs * stride_elements + depth_batch_start * expansion];
                        svuint8_t raw = svld1_u8(depth_predicate_b8, (uint8_t const *)src);
                        svfloat16_t converted_f16 = nk_e4m3x_to_f16x_ssve_(depth_predicate_b16, raw, subnorm_lut);
                        svst1_f16(depth_predicate_b16, (float16_t *)b_bounce_f16x32, converted_f16);
                        svld1_hor_za32(0, column, batch_predicate_b32, (nk_f32_t const *)b_bounce_f16x32);
                    }
                }
                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {
                    svfloat16_t a_interleaved = svreinterpret_f16_f32(
                        svld1_f32(full_predicate_b32, a_buffer[depth_step]));
                    svfloat32_t b_column_f32 = svread_ver_za32_f32_m(svdup_f32(0.0f), full_predicate_b32, 0,
                                                                     depth_step);
                    svfloat16_t b_interleaved = svreinterpret_f16_f32(b_column_f32);
                    svmopa_za32_f16_m(3, row_predicate_b16, full_predicate_b16, a_interleaved, b_interleaved);
                }
            }

            for (nk_size_t row = 0; row < rows_actual; row++) {
                nk_size_t const row_abs = row_tile_start + row;
                nk_f32_t *result_row = result + row_abs * result_stride_elements;
                svst1_hor_za32(1, row, full_predicate_b32, result_row + (column_tile_index + 0) * tile_dimension);
                svst1_hor_za32(2, row, full_predicate_b32, result_row + (column_tile_index + 1) * tile_dimension);
                svst1_hor_za32(3, row, full_predicate_b32, result_row + (column_tile_index + 2) * tile_dimension);
            }
        }

        for (; column_tile_index < column_tile_count; column_tile_index++) {
            nk_size_t const column_tile_start = column_tile_index * tile_dimension;
            nk_size_t const cols_remaining = (column_tile_start + tile_dimension <= n_vectors)
                                                 ? tile_dimension
                                                 : (n_vectors - column_tile_start);
            svbool_t const column_predicate_b32 = svwhilelt_b32((uint32_t)0, (uint32_t)cols_remaining);
            svbool_t const column_predicate_b16 = svwhilelt_b16((uint32_t)0, (uint32_t)(cols_remaining * expansion));

            svzero_mask_za(nk_sme_zero_za32_tile_1_);

            for (nk_size_t depth_batch_start = 0; depth_batch_start < depth_step_count;
                 depth_batch_start += depth_steps_per_batch) {
                nk_size_t const depth_batch_end = (depth_batch_start + depth_steps_per_batch < depth_step_count)
                                                      ? depth_batch_start + depth_steps_per_batch
                                                      : depth_step_count;
                nk_size_t const batch_size = depth_batch_end - depth_batch_start;

                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                svbool_t const batch_predicate_b32 = svwhilelt_b32((uint32_t)0, (uint32_t)batch_size);
                for (nk_size_t row_in_tile = 0; row_in_tile < rows_actual; row_in_tile++) {
                    nk_size_t const row_abs = row_tile_start + row_in_tile;
                    nk_e4m3_t const *a_src = vectors + row_abs * stride_elements + depth_batch_start * expansion;
                    svuint8_t raw_bytes = svld1_u8(full_predicate_b8, (uint8_t const *)a_src);
                    svfloat16_t converted_f16 = nk_e4m3x_to_f16x_ssve_(full_predicate_b16, raw_bytes, subnorm_lut);
                    svst1_f16(full_predicate_b16, (float16_t *)a_converted_f16x32[row_in_tile], converted_f16);
                    svld1_hor_za32(0, row_in_tile, batch_predicate_b32,
                                   (nk_f32_t const *)a_converted_f16x32[row_in_tile]);
                }

                // Save A columns from ZA0 to stack buffer
                for (nk_size_t s = 0; s < batch_size; s++)
                    svst1_f32(full_predicate_b32, a_buffer[s],
                              svread_ver_za32_f32_m(svdup_f32(0), row_predicate_b32, 0, s));

                svbool_t const depth_predicate_b8 = svwhilelt_b8((uint32_t)0, (uint32_t)(batch_size * expansion));
                svbool_t const depth_predicate_b16 = svwhilelt_b16((uint32_t)0, (uint32_t)(batch_size * expansion));

                // Load B column tile into ZA0, vertical read + FMOPA into ZA1
                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t column = 0; column < tile_dimension; column++) {
                    nk_size_t const column_abs = column_tile_start + column;
                    if (column_abs < n_vectors) {
                        nk_e4m3_t const *src = &vectors[column_abs * stride_elements + depth_batch_start * expansion];
                        svuint8_t raw = svld1_u8(depth_predicate_b8, (uint8_t const *)src);
                        svfloat16_t converted_f16 = nk_e4m3x_to_f16x_ssve_(depth_predicate_b16, raw, subnorm_lut);
                        svst1_f16(depth_predicate_b16, (float16_t *)b_bounce_f16x32, converted_f16);
                        svld1_hor_za32(0, column, batch_predicate_b32, (nk_f32_t const *)b_bounce_f16x32);
                    }
                }
                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {
                    svfloat16_t a_interleaved = svreinterpret_f16_f32(
                        svld1_f32(full_predicate_b32, a_buffer[depth_step]));
                    svfloat32_t b_column_f32 = svread_ver_za32_f32_m(svdup_f32(0.0f), column_predicate_b32, 0,
                                                                     depth_step);
                    svfloat16_t b_interleaved = svreinterpret_f16_f32(b_column_f32);
                    svmopa_za32_f16_m(1, row_predicate_b16, column_predicate_b16, a_interleaved, b_interleaved);
                }
            }

            for (nk_size_t row = 0; row < rows_actual; row++) {
                nk_size_t const row_abs = row_tile_start + row;
                svst1_hor_za32(1, row, column_predicate_b32,
                               result + row_abs * result_stride_elements + column_tile_start);
            }
        }

        for (nk_size_t row = 0; row < rows_actual; row++) {
            nk_size_t const i = row_tile_start + row;
            for (nk_size_t j = 0; j < i && j < n_vectors; j++)
                result[j * result_stride_elements + i] = result[i * result_stride_elements + j];
        }
    }
}

NK_PUBLIC void nk_dots_symmetric_e4m3_sme(nk_e4m3_t const *vectors, nk_size_t n_vectors, nk_size_t depth,
                                          nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                          nk_size_t row_start, nk_size_t row_count) {

    nk_size_t const stride_elements = stride / sizeof(nk_e4m3_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_dots_symmetric_e4m3_sme_kernel_(vectors, n_vectors, depth, stride_elements, result, result_stride_elements,
                                       row_start, row_count);
}

#pragma endregion

/*
 *  e5m2 × e5m2 → f32 GEMM using inline SSVE conversion + FMOPA.
 *
 *  Pipeline: e5m2 bytes → svunpklo → arithmetic → f16 → FMOPA → f32
 *  - Same tile layout as e4m3 (both convert to f16 before FMOPA)
 *  - E5M2 shares F16 exponent bias (15), so normal conversion is a shift
 *  - Handles infinity (mag=0x7C) and NaN (mag>0x7C)
 */
#pragma region Quarter Precision E5M2

/**
 *  Fused `e5m2` × `e5m2` → `f32` GEMM kernel using interleaved FMOPA.
 *  Converts `e5m2` → `f16` on-the-fly for A, B is pre-converted during packing.
 */
__arm_locally_streaming __arm_new("za") __attribute__((noinline)) static void nk_dots_packed_e5m2_sme_kernel_( //
    nk_e5m2_t const *a, void const *b_packed, nk_f32_t *c,                                                     //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,                                                        //
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) {

    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_size_t const column_tile_count = header->column_tile_count;
    nk_size_t const depth_step_count = header->depth_tile_count;

    nk_size_t const expansion = 2;
    nk_size_t const tile_dimension = svcntw();
    nk_size_t const vector_elements = svcnth();
    nk_size_t const depth_steps_per_batch = tile_dimension;

    nk_f16_t const *b_packed_base = (nk_f16_t const *)((char const *)b_packed + sizeof(nk_dots_sme_packed_header_t));

    svbool_t const full_predicate_b16 = svptrue_b16();
    svbool_t const full_predicate_b8 = svptrue_b8();
    svbool_t const full_predicate_b32 = svptrue_b32();

    NK_ALIGN64 nk_f16_t a_converted_f16x32[16][32];

    nk_size_t const row_tile_count = nk_size_divide_round_up_(rows, tile_dimension);

    for (nk_size_t row_tile_index = 0; row_tile_index < row_tile_count; row_tile_index++) {
        nk_size_t const row_start = row_tile_index * tile_dimension;
        nk_size_t const rows_remaining = (row_start + tile_dimension <= rows) ? tile_dimension : (rows - row_start);
        svbool_t const row_predicate_b16 = svwhilelt_b16((uint32_t)0, (uint32_t)(rows_remaining * expansion));
        svbool_t const row_predicate_b32 = svwhilelt_b32((uint32_t)0, (uint32_t)rows_remaining);

        nk_size_t column_tile_index = 0;

        // Fast path: 3 column tiles using ZA1-ZA3 (ZA0 = staging)
        for (; column_tile_index + 3 <= column_tile_count; column_tile_index += 3) {
            svzero_mask_za(nk_sme_zero_za32_tiles_123_);

            for (nk_size_t depth_batch_start = 0; depth_batch_start < depth_step_count;
                 depth_batch_start += depth_steps_per_batch) {
                nk_size_t const depth_batch_end = (depth_batch_start + depth_steps_per_batch < depth_step_count)
                                                      ? depth_batch_start + depth_steps_per_batch
                                                      : depth_step_count;
                nk_size_t const batch_size = depth_batch_end - depth_batch_start;

                svzero_mask_za(nk_sme_zero_za32_tile_0_);

                svbool_t const batch_predicate_b32 = svwhilelt_b32((uint32_t)0, (uint32_t)batch_size);
                for (nk_size_t row_in_tile = 0; row_in_tile < rows_remaining; row_in_tile++) {
                    nk_size_t const a_row = row_start + row_in_tile;
                    // Vectorized e5m2 → f16 conversion (2 instructions)
                    nk_e5m2_t const *a_src = a + a_row * a_stride_elements + depth_batch_start * expansion;
                    svuint8_t raw_bytes = svld1_u8(full_predicate_b8, (uint8_t const *)a_src);
                    svfloat16_t converted_f16 = nk_e5m2x_to_f16x_ssve_(full_predicate_b16, raw_bytes);
                    svst1_f16(full_predicate_b16, (float16_t *)a_converted_f16x32[row_in_tile], converted_f16);
                    svld1_hor_za32(0, row_in_tile, batch_predicate_b32,
                                   (nk_f32_t const *)a_converted_f16x32[row_in_tile]);
                }

                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {

                    svfloat32_t a_column_f32 = svread_ver_za32_f32_m(svdup_f32(0.0f), row_predicate_b32, 0, depth_step);
                    svfloat16_t a_interleaved_vector_f16 = svreinterpret_f16_f32(a_column_f32);

                    nk_f16_t const *b_packed_ptr_0 = b_packed_base + ((column_tile_index + 0) * depth_step_count +
                                                                      depth_batch_start + depth_step) *
                                                                         vector_elements;
                    nk_f16_t const *b_packed_ptr_1 = b_packed_base + ((column_tile_index + 1) * depth_step_count +
                                                                      depth_batch_start + depth_step) *
                                                                         vector_elements;
                    nk_f16_t const *b_packed_ptr_2 = b_packed_base + ((column_tile_index + 2) * depth_step_count +
                                                                      depth_batch_start + depth_step) *
                                                                         vector_elements;
                    svfloat16_t b_packed_vector_0 = svld1_f16(full_predicate_b16, (float16_t const *)b_packed_ptr_0);
                    svfloat16_t b_packed_vector_1 = svld1_f16(full_predicate_b16, (float16_t const *)b_packed_ptr_1);
                    svfloat16_t b_packed_vector_2 = svld1_f16(full_predicate_b16, (float16_t const *)b_packed_ptr_2);

                    svmopa_za32_f16_m(1, row_predicate_b16, full_predicate_b16, a_interleaved_vector_f16,
                                      b_packed_vector_0);
                    svmopa_za32_f16_m(2, row_predicate_b16, full_predicate_b16, a_interleaved_vector_f16,
                                      b_packed_vector_1);
                    svmopa_za32_f16_m(3, row_predicate_b16, full_predicate_b16, a_interleaved_vector_f16,
                                      b_packed_vector_2);
                }
            }

            for (nk_size_t row = 0; row < rows_remaining; row++) {
                nk_size_t const cs0 = (column_tile_index + 0) * tile_dimension;
                nk_size_t const cs1 = (column_tile_index + 1) * tile_dimension;
                nk_size_t const cs2 = (column_tile_index + 2) * tile_dimension;
                svst1_hor_za32(1, row, full_predicate_b32, c + (row_start + row) * c_stride_elements + cs0);
                svst1_hor_za32(2, row, full_predicate_b32, c + (row_start + row) * c_stride_elements + cs1);
                svst1_hor_za32(3, row, full_predicate_b32, c + (row_start + row) * c_stride_elements + cs2);
            }
        }

        // Remainder: 1 column tile using ZA1
        for (; column_tile_index < column_tile_count; column_tile_index++) {
            nk_size_t const column_start = column_tile_index * tile_dimension;
            nk_size_t const cols_remaining = (column_start + tile_dimension <= columns) ? tile_dimension
                                                                                        : (columns - column_start);
            svbool_t const column_predicate_b32 = svwhilelt_b32((uint32_t)0, (uint32_t)cols_remaining);
            svbool_t const column_predicate_b16 = svwhilelt_b16((uint32_t)0, (uint32_t)(cols_remaining * expansion));

            svzero_mask_za(nk_sme_zero_za32_tile_1_);

            for (nk_size_t depth_batch_start = 0; depth_batch_start < depth_step_count;
                 depth_batch_start += depth_steps_per_batch) {
                nk_size_t const depth_batch_end = (depth_batch_start + depth_steps_per_batch < depth_step_count)
                                                      ? depth_batch_start + depth_steps_per_batch
                                                      : depth_step_count;
                nk_size_t const batch_size = depth_batch_end - depth_batch_start;

                svzero_mask_za(nk_sme_zero_za32_tile_0_);

                svbool_t const batch_predicate_b32 = svwhilelt_b32((uint32_t)0, (uint32_t)batch_size);
                for (nk_size_t row_in_tile = 0; row_in_tile < rows_remaining; row_in_tile++) {
                    nk_size_t const a_row = row_start + row_in_tile;
                    nk_e5m2_t const *a_src = a + a_row * a_stride_elements + depth_batch_start * expansion;
                    svuint8_t raw_bytes = svld1_u8(full_predicate_b8, (uint8_t const *)a_src);
                    svfloat16_t converted_f16 = nk_e5m2x_to_f16x_ssve_(full_predicate_b16, raw_bytes);
                    svst1_f16(full_predicate_b16, (float16_t *)a_converted_f16x32[row_in_tile], converted_f16);
                    svld1_hor_za32(0, row_in_tile, batch_predicate_b32,
                                   (nk_f32_t const *)a_converted_f16x32[row_in_tile]);
                }

                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {

                    svfloat32_t a_column_f32 = svread_ver_za32_f32_m(svdup_f32(0.0f), row_predicate_b32, 0, depth_step);
                    svfloat16_t a_interleaved_vector_f16 = svreinterpret_f16_f32(a_column_f32);

                    nk_f16_t const *b_packed_ptr = b_packed_base + (column_tile_index * depth_step_count +
                                                                    depth_batch_start + depth_step) *
                                                                       vector_elements;
                    svfloat16_t b_packed_vector = svld1_f16(column_predicate_b16, (float16_t const *)b_packed_ptr);

                    svmopa_za32_f16_m(1, row_predicate_b16, column_predicate_b16, a_interleaved_vector_f16,
                                      b_packed_vector);
                }
            }

            for (nk_size_t row = 0; row < rows_remaining; row++) {
                svst1_hor_za32(1, row, column_predicate_b32, c + (row_start + row) * c_stride_elements + column_start);
            }
        }
    }
}

NK_PUBLIC nk_size_t nk_dots_packed_size_e5m2_sme(nk_size_t n, nk_size_t k) { return nk_dots_packed_size_f16_sme(n, k); }

NK_PUBLIC void nk_dots_pack_e5m2_sme(nk_e5m2_t const *b, nk_size_t n, nk_size_t k, nk_size_t b_stride, void *b_packed) {

    nk_size_t const expansion = 2;
    nk_size_t const tile_dimension = svcntsw();
    nk_size_t const vector_elements = svcntsh();
    nk_size_t const b_stride_elements = b_stride / sizeof(nk_e5m2_t);

    nk_size_t const column_tile_count = nk_size_divide_round_up_(n, tile_dimension);
    nk_size_t const depth_step_count = nk_size_divide_round_up_(k, expansion);
    nk_size_t const total_vectors = column_tile_count * depth_step_count;

    nk_dots_sme_packed_header_t *header = (nk_dots_sme_packed_header_t *)b_packed;
    header->column_tile_count = (nk_u32_t)column_tile_count;
    header->depth_tile_count = (nk_u32_t)depth_step_count;
    header->columns = (nk_u32_t)n;
    header->depth = (nk_u32_t)k;
    header->svl_bytes = (nk_u32_t)(svcntsw() * sizeof(nk_f32_t));

    nk_f16_t *tiles_ptr = (nk_f16_t *)((char *)b_packed + sizeof(nk_dots_sme_packed_header_t));

    for (nk_size_t i = 0; i < total_vectors * vector_elements; i++) tiles_ptr[i] = 0;

    // Interleaved packing with e5m2 → f16 conversion
    for (nk_size_t column_tile = 0; column_tile < column_tile_count; column_tile++) {
        for (nk_size_t depth_step = 0; depth_step < depth_step_count; depth_step++) {
            nk_size_t const vec_index = column_tile * depth_step_count + depth_step;
            nk_f16_t *vec_output = tiles_ptr + vec_index * vector_elements;

            nk_size_t const b_row_start = column_tile * tile_dimension;
            nk_size_t const depth_base = depth_step * expansion;
            nk_size_t const rows_to_pack = (b_row_start + tile_dimension <= n) ? tile_dimension : (n - b_row_start);

            for (nk_size_t column_in_tile = 0; column_in_tile < rows_to_pack; column_in_tile++) {
                for (nk_size_t sub_element = 0; sub_element < expansion; sub_element++) {
                    nk_size_t const depth_idx = depth_base + sub_element;
                    if (depth_idx < k) {
                        nk_size_t const src_idx = (b_row_start + column_in_tile) * b_stride_elements + depth_idx;
                        nk_e5m2_to_f16_serial(&b[src_idx], &vec_output[expansion * column_in_tile + sub_element]);
                    }
                }
            }
        }
    }
}

/*  `e5m2` × `e5m2` → `f32` GEMM: public interface.
 *  Predicate-based edge handling eliminates scalar fallbacks.
 */
NK_PUBLIC void nk_dots_packed_e5m2_sme(                    //
    nk_e5m2_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,    //
    nk_size_t a_stride, nk_size_t c_stride) {

    nk_size_t const a_stride_elements = a_stride / sizeof(nk_e5m2_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);

    nk_dots_packed_e5m2_sme_kernel_(a, b_packed, c, rows, columns, depth, a_stride_elements, c_stride_elements);
}

/**
 * `e5m2` × `e5m2` → `f32` symmetric kernel using MOPA self-GEMM.
 *  Time-shares ZA0 for both A and B transposition with e5m2 → f16 conversion.
 *  Pre-reads A columns into Z registers, then reloads ZA0 with converted B data
 *  per column tile. Eliminates all scalar B-packing loops.
 */
__arm_locally_streaming __arm_new("za") static void nk_dots_symmetric_e5m2_sme_kernel_(
    nk_e5m2_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride_elements, nk_f32_t *result,
    nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) {

    nk_size_t const expansion = 2;
    nk_size_t const tile_dimension = svcntw();
    nk_size_t const depth_step_count = nk_size_divide_round_up_(depth, expansion);
    nk_size_t const depth_steps_per_batch = tile_dimension;

    svbool_t const full_predicate_b16 = svptrue_b16();
    svbool_t const full_predicate_b8 = svptrue_b8();
    svbool_t const full_predicate_b32 = svptrue_b32();

    NK_ALIGN64 nk_f16_t a_converted_f16x32[16][32];
    NK_ALIGN64 nk_f16_t b_bounce_f16x32[32];
    NK_ALIGN64 nk_f32_t a_buffer[16][16];

    nk_size_t const row_end = row_start + row_count;
    nk_size_t const column_tile_count = nk_size_divide_round_up_(n_vectors, tile_dimension);

    for (nk_size_t row_tile_start = row_start; row_tile_start < row_end && row_tile_start < n_vectors;
         row_tile_start += tile_dimension) {
        nk_size_t const rows_clamped = (row_tile_start + tile_dimension <= row_end) ? tile_dimension
                                                                                    : (row_end - row_tile_start);
        nk_size_t const rows_actual = (row_tile_start + rows_clamped <= n_vectors) ? rows_clamped
                                                                                   : (n_vectors - row_tile_start);
        svbool_t const row_predicate_b16 = svwhilelt_b16((uint32_t)0, (uint32_t)(rows_actual * expansion));
        svbool_t const row_predicate_b32 = svwhilelt_b32((uint32_t)0, (uint32_t)rows_actual);

        nk_size_t column_tile_index = 0;

        for (; column_tile_index + 3 <= column_tile_count; column_tile_index += 3) {
            svzero_mask_za(nk_sme_zero_za32_tiles_123_);

            for (nk_size_t depth_batch_start = 0; depth_batch_start < depth_step_count;
                 depth_batch_start += depth_steps_per_batch) {
                nk_size_t const depth_batch_end = (depth_batch_start + depth_steps_per_batch < depth_step_count)
                                                      ? depth_batch_start + depth_steps_per_batch
                                                      : depth_step_count;
                nk_size_t const batch_size = depth_batch_end - depth_batch_start;

                // ZA transpose for A rows: convert e5m2 → f16 then load
                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                svbool_t const batch_predicate_b32 = svwhilelt_b32((uint32_t)0, (uint32_t)batch_size);
                for (nk_size_t row_in_tile = 0; row_in_tile < rows_actual; row_in_tile++) {
                    nk_size_t const row_abs = row_tile_start + row_in_tile;
                    nk_e5m2_t const *a_src = vectors + row_abs * stride_elements + depth_batch_start * expansion;
                    svuint8_t raw_bytes = svld1_u8(full_predicate_b8, (uint8_t const *)a_src);
                    svfloat16_t converted_f16 = nk_e5m2x_to_f16x_ssve_(full_predicate_b16, raw_bytes);
                    svst1_f16(full_predicate_b16, (float16_t *)a_converted_f16x32[row_in_tile], converted_f16);
                    svld1_hor_za32(0, row_in_tile, batch_predicate_b32,
                                   (nk_f32_t const *)a_converted_f16x32[row_in_tile]);
                }

                // Save A columns from ZA0 to stack buffer
                for (nk_size_t s = 0; s < batch_size; s++)
                    svst1_f32(full_predicate_b32, a_buffer[s],
                              svread_ver_za32_f32_m(svdup_f32(0), row_predicate_b32, 0, s));

                svbool_t const depth_predicate_b8 = svwhilelt_b8((uint32_t)0, (uint32_t)(batch_size * expansion));
                svbool_t const depth_predicate_b16 = svwhilelt_b16((uint32_t)0, (uint32_t)(batch_size * expansion));

                // Load B column tile 0 into ZA0, vertical read + FMOPA into ZA1
                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t column = 0; column < tile_dimension; column++) {
                    nk_size_t const column_abs = (column_tile_index + 0) * tile_dimension + column;
                    if (column_abs < n_vectors) {
                        nk_e5m2_t const *src = &vectors[column_abs * stride_elements + depth_batch_start * expansion];
                        svuint8_t raw = svld1_u8(depth_predicate_b8, (uint8_t const *)src);
                        svfloat16_t converted_f16 = nk_e5m2x_to_f16x_ssve_(depth_predicate_b16, raw);
                        svst1_f16(depth_predicate_b16, (float16_t *)b_bounce_f16x32, converted_f16);
                        svld1_hor_za32(0, column, batch_predicate_b32, (nk_f32_t const *)b_bounce_f16x32);
                    }
                }
                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {
                    svfloat16_t a_interleaved = svreinterpret_f16_f32(
                        svld1_f32(full_predicate_b32, a_buffer[depth_step]));
                    svfloat32_t b_column_f32 = svread_ver_za32_f32_m(svdup_f32(0.0f), full_predicate_b32, 0,
                                                                     depth_step);
                    svfloat16_t b_interleaved = svreinterpret_f16_f32(b_column_f32);
                    svmopa_za32_f16_m(1, row_predicate_b16, full_predicate_b16, a_interleaved, b_interleaved);
                }

                // Load B column tile 1 into ZA0, vertical read + FMOPA into ZA2
                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t column = 0; column < tile_dimension; column++) {
                    nk_size_t const column_abs = (column_tile_index + 1) * tile_dimension + column;
                    if (column_abs < n_vectors) {
                        nk_e5m2_t const *src = &vectors[column_abs * stride_elements + depth_batch_start * expansion];
                        svuint8_t raw = svld1_u8(depth_predicate_b8, (uint8_t const *)src);
                        svfloat16_t converted_f16 = nk_e5m2x_to_f16x_ssve_(depth_predicate_b16, raw);
                        svst1_f16(depth_predicate_b16, (float16_t *)b_bounce_f16x32, converted_f16);
                        svld1_hor_za32(0, column, batch_predicate_b32, (nk_f32_t const *)b_bounce_f16x32);
                    }
                }
                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {
                    svfloat16_t a_interleaved = svreinterpret_f16_f32(
                        svld1_f32(full_predicate_b32, a_buffer[depth_step]));
                    svfloat32_t b_column_f32 = svread_ver_za32_f32_m(svdup_f32(0.0f), full_predicate_b32, 0,
                                                                     depth_step);
                    svfloat16_t b_interleaved = svreinterpret_f16_f32(b_column_f32);
                    svmopa_za32_f16_m(2, row_predicate_b16, full_predicate_b16, a_interleaved, b_interleaved);
                }

                // Load B column tile 2 into ZA0, vertical read + FMOPA into ZA3
                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t column = 0; column < tile_dimension; column++) {
                    nk_size_t const column_abs = (column_tile_index + 2) * tile_dimension + column;
                    if (column_abs < n_vectors) {
                        nk_e5m2_t const *src = &vectors[column_abs * stride_elements + depth_batch_start * expansion];
                        svuint8_t raw = svld1_u8(depth_predicate_b8, (uint8_t const *)src);
                        svfloat16_t converted_f16 = nk_e5m2x_to_f16x_ssve_(depth_predicate_b16, raw);
                        svst1_f16(depth_predicate_b16, (float16_t *)b_bounce_f16x32, converted_f16);
                        svld1_hor_za32(0, column, batch_predicate_b32, (nk_f32_t const *)b_bounce_f16x32);
                    }
                }
                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {
                    svfloat16_t a_interleaved = svreinterpret_f16_f32(
                        svld1_f32(full_predicate_b32, a_buffer[depth_step]));
                    svfloat32_t b_column_f32 = svread_ver_za32_f32_m(svdup_f32(0.0f), full_predicate_b32, 0,
                                                                     depth_step);
                    svfloat16_t b_interleaved = svreinterpret_f16_f32(b_column_f32);
                    svmopa_za32_f16_m(3, row_predicate_b16, full_predicate_b16, a_interleaved, b_interleaved);
                }
            }

            for (nk_size_t row = 0; row < rows_actual; row++) {
                nk_size_t const row_abs = row_tile_start + row;
                nk_f32_t *result_row = result + row_abs * result_stride_elements;
                svst1_hor_za32(1, row, full_predicate_b32, result_row + (column_tile_index + 0) * tile_dimension);
                svst1_hor_za32(2, row, full_predicate_b32, result_row + (column_tile_index + 1) * tile_dimension);
                svst1_hor_za32(3, row, full_predicate_b32, result_row + (column_tile_index + 2) * tile_dimension);
            }
        }

        for (; column_tile_index < column_tile_count; column_tile_index++) {
            nk_size_t const column_tile_start = column_tile_index * tile_dimension;
            nk_size_t const cols_remaining = (column_tile_start + tile_dimension <= n_vectors)
                                                 ? tile_dimension
                                                 : (n_vectors - column_tile_start);
            svbool_t const column_predicate_b32 = svwhilelt_b32((uint32_t)0, (uint32_t)cols_remaining);
            svbool_t const column_predicate_b16 = svwhilelt_b16((uint32_t)0, (uint32_t)(cols_remaining * expansion));

            svzero_mask_za(nk_sme_zero_za32_tile_1_);

            for (nk_size_t depth_batch_start = 0; depth_batch_start < depth_step_count;
                 depth_batch_start += depth_steps_per_batch) {
                nk_size_t const depth_batch_end = (depth_batch_start + depth_steps_per_batch < depth_step_count)
                                                      ? depth_batch_start + depth_steps_per_batch
                                                      : depth_step_count;
                nk_size_t const batch_size = depth_batch_end - depth_batch_start;

                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                svbool_t const batch_predicate_b32 = svwhilelt_b32((uint32_t)0, (uint32_t)batch_size);
                for (nk_size_t row_in_tile = 0; row_in_tile < rows_actual; row_in_tile++) {
                    nk_size_t const row_abs = row_tile_start + row_in_tile;
                    nk_e5m2_t const *a_src = vectors + row_abs * stride_elements + depth_batch_start * expansion;
                    svuint8_t raw_bytes = svld1_u8(full_predicate_b8, (uint8_t const *)a_src);
                    svfloat16_t converted_f16 = nk_e5m2x_to_f16x_ssve_(full_predicate_b16, raw_bytes);
                    svst1_f16(full_predicate_b16, (float16_t *)a_converted_f16x32[row_in_tile], converted_f16);
                    svld1_hor_za32(0, row_in_tile, batch_predicate_b32,
                                   (nk_f32_t const *)a_converted_f16x32[row_in_tile]);
                }

                // Save A columns from ZA0 to stack buffer
                for (nk_size_t s = 0; s < batch_size; s++)
                    svst1_f32(full_predicate_b32, a_buffer[s],
                              svread_ver_za32_f32_m(svdup_f32(0), row_predicate_b32, 0, s));

                svbool_t const depth_predicate_b8 = svwhilelt_b8((uint32_t)0, (uint32_t)(batch_size * expansion));
                svbool_t const depth_predicate_b16 = svwhilelt_b16((uint32_t)0, (uint32_t)(batch_size * expansion));

                // Load B column tile into ZA0, vertical read + FMOPA into ZA1
                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t column = 0; column < tile_dimension; column++) {
                    nk_size_t const column_abs = column_tile_start + column;
                    if (column_abs < n_vectors) {
                        nk_e5m2_t const *src = &vectors[column_abs * stride_elements + depth_batch_start * expansion];
                        svuint8_t raw = svld1_u8(depth_predicate_b8, (uint8_t const *)src);
                        svfloat16_t converted_f16 = nk_e5m2x_to_f16x_ssve_(depth_predicate_b16, raw);
                        svst1_f16(depth_predicate_b16, (float16_t *)b_bounce_f16x32, converted_f16);
                        svld1_hor_za32(0, column, batch_predicate_b32, (nk_f32_t const *)b_bounce_f16x32);
                    }
                }
                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {
                    svfloat16_t a_interleaved = svreinterpret_f16_f32(
                        svld1_f32(full_predicate_b32, a_buffer[depth_step]));
                    svfloat32_t b_column_f32 = svread_ver_za32_f32_m(svdup_f32(0.0f), column_predicate_b32, 0,
                                                                     depth_step);
                    svfloat16_t b_interleaved = svreinterpret_f16_f32(b_column_f32);
                    svmopa_za32_f16_m(1, row_predicate_b16, column_predicate_b16, a_interleaved, b_interleaved);
                }
            }

            for (nk_size_t row = 0; row < rows_actual; row++) {
                nk_size_t const row_abs = row_tile_start + row;
                svst1_hor_za32(1, row, column_predicate_b32,
                               result + row_abs * result_stride_elements + column_tile_start);
            }
        }

        for (nk_size_t row = 0; row < rows_actual; row++) {
            nk_size_t const i = row_tile_start + row;
            for (nk_size_t j = 0; j < i && j < n_vectors; j++)
                result[j * result_stride_elements + i] = result[i * result_stride_elements + j];
        }
    }
}

NK_PUBLIC void nk_dots_symmetric_e5m2_sme(nk_e5m2_t const *vectors, nk_size_t n_vectors, nk_size_t depth,
                                          nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                          nk_size_t row_start, nk_size_t row_count) {

    nk_size_t const stride_elements = stride / sizeof(nk_e5m2_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_dots_symmetric_e5m2_sme_kernel_(vectors, n_vectors, depth, stride_elements, result, result_stride_elements,
                                       row_start, row_count);
}

#pragma endregion

/*
 *  `e2m3` × `e2m3` → `f32` GEMM using `i8` SME outer products.
 *
 *  E2M3 format: S MMMMM (1+2+3 = 6 bits, stored in a byte with upper 2 bits unused).
 *  Values are converted to signed `i8` via a 32-entry magnitude LUT, then processed
 *  with `svmopa_za32_s8_m`. The `i32` accumulator is converted to `f32` and scaled
 *  by 1/256 to recover the true floating-point dot product.
 *
 *  This gives 2x throughput over the `f16` path since `i8` SMOPA processes 4 elements
 *  per `i32` word vs 2 `f16` elements per `f32` word.
 *
 *  Tile dimensions (SVL=512, Apple M4):
 *  - Input vectors: 64 `i8` elements (after conversion)
 *  - Output tile: 16 × 16 `i32` → `f32` elements (`ZA32`)
 *  - Each output word accumulates 4 `i8` pairs, scaled by 1/256
 */

#pragma region Quarter Precision E2M3

/**
 *  Inline `e2m3` → signed `i8` conversion returning `svint8_t` for direct use in GEMM.
 *  Uses a 32-entry magnitude LUT via SVE TBL instruction.
 *
 *  E2M3 encoding: bit 5 = sign, bits 4:0 = magnitude index (0..27 used).
 *  LUT maps magnitude index → unsigned integer value, then sign is applied.
 *
 *  @param pg8       Predicate for 8-bit elements
 *  @param raw_bytes Pre-loaded e2m3 bytes as `svuint8_t`
 *  @return          Signed `i8` values as `svint8_t`
 */
NK_INTERNAL svint8_t nk_e2m3x_to_i8x_ssve_(svbool_t pg8, svuint8_t raw_bytes) {
    // 32-entry magnitude LUT, replicated for SVE TBL (handles SVL > 256 bits)
    static NK_ALIGN64 nk_u8_t const lut_data[64] = {0,  2,  4,  6,  8,  10, 12, 14, 16, 20,  24,  28,  32, 36, 40, 44,
                                                    48, 52, 56, 60, 64, 72, 80, 88, 96, 104, 112, 120, 0,  0,  0,  0,
                                                    0,  2,  4,  6,  8,  10, 12, 14, 16, 20,  24,  28,  32, 36, 40, 44,
                                                    48, 52, 56, 60, 64, 72, 80, 88, 96, 104, 112, 120, 0,  0,  0,  0};
    svuint8_t lut_vector = svld1_u8(svptrue_b8(), lut_data);
    svuint8_t magnitude = svand_n_u8_x(pg8, raw_bytes, 0x1F);
    svuint8_t unsigned_val = svtbl_u8(lut_vector, magnitude);
    svuint8_t sign_bits = svand_n_u8_x(pg8, raw_bytes, 0x20);
    svbool_t negate = svcmpne_n_u8(pg8, sign_bits, 0);
    svint8_t positive = svreinterpret_s8_u8(unsigned_val);
    svint8_t negated = svneg_s8_x(pg8, positive);
    return svsel_s8(negate, negated, positive);
}

/**
 *  Fused `e2m3` × `e2m3` → `f32` GEMM kernel using interleaved SMOPA.
 *  Converts `e2m3` → `i8` on-the-fly for A, B is pre-converted during packing.
 *  Accumulates in `i32` via `svmopa_za32_s8_m`, then converts to `f32` with 1/256 scaling.
 */
__arm_locally_streaming __arm_new("za") __attribute__((noinline)) static void nk_dots_packed_e2m3_sme_kernel_( //
    nk_e2m3_t const *a, void const *b_packed, nk_f32_t *c,                                                     //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,                                                        //
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) {

    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_size_t const column_tile_count = header->column_tile_count;
    nk_size_t const depth_step_count = header->depth_tile_count;

    nk_size_t const expansion = 4;              // SMOPA i8→i32: 4 i8 pairs per i32 output
    nk_size_t const tile_dimension = svcntw();  // 16: ZA32 tile dimension
    nk_size_t const vector_elements = svcntb(); // 64: i8 elements per SVE vector
    nk_size_t const depth_steps_per_batch = tile_dimension;

    nk_i8_t const *b_packed_base = (nk_i8_t const *)((char const *)b_packed + sizeof(nk_dots_sme_packed_header_t));

    svbool_t const full_predicate_b8 = svptrue_b8();
    svbool_t const full_predicate_b32 = svptrue_b32();

    // Stack buffer for e2m3 -> i8 conversion (64 i8 per row, up to 16 rows)
    NK_ALIGN64 nk_i8_t a_converted_i8x64[16][64];

    nk_size_t const row_tile_count = nk_size_divide_round_up_(rows, tile_dimension);

    for (nk_size_t row_tile_index = 0; row_tile_index < row_tile_count; row_tile_index++) {
        nk_size_t const row_start = row_tile_index * tile_dimension;
        nk_size_t const rows_remaining = (row_start + tile_dimension <= rows) ? tile_dimension : (rows - row_start);
        svbool_t const row_predicate_b8 = svwhilelt_b8((uint32_t)0, (uint32_t)(rows_remaining * expansion));
        svbool_t const row_predicate_b32 = svwhilelt_b32((uint32_t)0, (uint32_t)rows_remaining);

        nk_size_t column_tile_index = 0;

        // Fast path: 3 column tiles using ZA1-ZA3 (ZA0 = staging)
        for (; column_tile_index + 3 <= column_tile_count; column_tile_index += 3) {
            svzero_mask_za(nk_sme_zero_za32_tiles_123_);

            for (nk_size_t depth_batch_start = 0; depth_batch_start < depth_step_count;
                 depth_batch_start += depth_steps_per_batch) {
                nk_size_t const depth_batch_end = (depth_batch_start + depth_steps_per_batch < depth_step_count)
                                                      ? depth_batch_start + depth_steps_per_batch
                                                      : depth_step_count;
                nk_size_t const batch_size = depth_batch_end - depth_batch_start;

                // Convert e2m3 -> i8 for each A row in this batch, then load into ZA0
                svzero_mask_za(nk_sme_zero_za32_tile_0_);

                svbool_t const batch_predicate_b32 = svwhilelt_b32((uint32_t)0, (uint32_t)batch_size);
                for (nk_size_t row_in_tile = 0; row_in_tile < rows_remaining; row_in_tile++) {
                    nk_size_t const a_row = row_start + row_in_tile;
                    // Load raw e2m3 bytes and convert to i8 using vectorized conversion
                    nk_e2m3_t const *a_src = a + a_row * a_stride_elements + depth_batch_start * expansion;
                    svuint8_t raw_bytes = svld1_u8(full_predicate_b8, (uint8_t const *)a_src);
                    svint8_t converted_i8 = nk_e2m3x_to_i8x_ssve_(full_predicate_b8, raw_bytes);
                    svst1_s8(full_predicate_b8, a_converted_i8x64[row_in_tile], converted_i8);
                    // Load converted i8 row into ZA0 as f32 words (each f32 word = 4 packed i8 bytes)
                    svld1_hor_za32(0, row_in_tile, batch_predicate_b32,
                                   (nk_f32_t const *)a_converted_i8x64[row_in_tile]);
                }

                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {

                    // Vertical read at f32 granularity produces [row0_k0..k3, row1_k0..k3, ...]
                    svint32_t a_column_i32 = svread_ver_za32_s32_m(svdup_s32(0), row_predicate_b32, 0, depth_step);
                    svint8_t a_interleaved_vector_i8 = svreinterpret_s8_s32(a_column_i32);

                    nk_i8_t const *b_packed_ptr_0 = b_packed_base + ((column_tile_index + 0) * depth_step_count +
                                                                     depth_batch_start + depth_step) *
                                                                        vector_elements;
                    nk_i8_t const *b_packed_ptr_1 = b_packed_base + ((column_tile_index + 1) * depth_step_count +
                                                                     depth_batch_start + depth_step) *
                                                                        vector_elements;
                    nk_i8_t const *b_packed_ptr_2 = b_packed_base + ((column_tile_index + 2) * depth_step_count +
                                                                     depth_batch_start + depth_step) *
                                                                        vector_elements;
                    svint8_t b_packed_vector_0 = svld1_s8(full_predicate_b8, b_packed_ptr_0);
                    svint8_t b_packed_vector_1 = svld1_s8(full_predicate_b8, b_packed_ptr_1);
                    svint8_t b_packed_vector_2 = svld1_s8(full_predicate_b8, b_packed_ptr_2);

                    svmopa_za32_s8_m(1, row_predicate_b8, full_predicate_b8, a_interleaved_vector_i8,
                                     b_packed_vector_0);
                    svmopa_za32_s8_m(2, row_predicate_b8, full_predicate_b8, a_interleaved_vector_i8,
                                     b_packed_vector_1);
                    svmopa_za32_s8_m(3, row_predicate_b8, full_predicate_b8, a_interleaved_vector_i8,
                                     b_packed_vector_2);
                }
            }

            // Store results: convert i32 → f32 with 1/256 scaling
            for (nk_size_t row = 0; row < rows_remaining; row++) {
                nk_size_t const row_offset = (row_start + row) * c_stride_elements;
                nk_size_t const cs0 = (column_tile_index + 0) * tile_dimension;
                nk_size_t const cs1 = (column_tile_index + 1) * tile_dimension;
                nk_size_t const cs2 = (column_tile_index + 2) * tile_dimension;

                svint32_t row_i32_1 = svread_hor_za32_s32_m(svdup_s32(0), full_predicate_b32, 1, row);
                svint32_t row_i32_2 = svread_hor_za32_s32_m(svdup_s32(0), full_predicate_b32, 2, row);
                svint32_t row_i32_3 = svread_hor_za32_s32_m(svdup_s32(0), full_predicate_b32, 3, row);
                svfloat32_t row_f32_1 = svmul_n_f32_x(full_predicate_b32,
                                                      svcvt_f32_s32_x(full_predicate_b32, row_i32_1), 1.0f / 256.0f);
                svfloat32_t row_f32_2 = svmul_n_f32_x(full_predicate_b32,
                                                      svcvt_f32_s32_x(full_predicate_b32, row_i32_2), 1.0f / 256.0f);
                svfloat32_t row_f32_3 = svmul_n_f32_x(full_predicate_b32,
                                                      svcvt_f32_s32_x(full_predicate_b32, row_i32_3), 1.0f / 256.0f);
                svst1_f32(full_predicate_b32, c + row_offset + cs0, row_f32_1);
                svst1_f32(full_predicate_b32, c + row_offset + cs1, row_f32_2);
                svst1_f32(full_predicate_b32, c + row_offset + cs2, row_f32_3);
            }
        }

        // Remainder: 1 column tile using ZA1
        for (; column_tile_index < column_tile_count; column_tile_index++) {
            nk_size_t const column_start = column_tile_index * tile_dimension;
            nk_size_t const cols_remaining = (column_start + tile_dimension <= columns) ? tile_dimension
                                                                                        : (columns - column_start);
            svbool_t const column_predicate_b32 = svwhilelt_b32((uint32_t)0, (uint32_t)cols_remaining);
            svbool_t const column_predicate_b8 = svwhilelt_b8((uint32_t)0, (uint32_t)(cols_remaining * expansion));

            svzero_mask_za(nk_sme_zero_za32_tile_1_);

            for (nk_size_t depth_batch_start = 0; depth_batch_start < depth_step_count;
                 depth_batch_start += depth_steps_per_batch) {
                nk_size_t const depth_batch_end = (depth_batch_start + depth_steps_per_batch < depth_step_count)
                                                      ? depth_batch_start + depth_steps_per_batch
                                                      : depth_step_count;
                nk_size_t const batch_size = depth_batch_end - depth_batch_start;

                svzero_mask_za(nk_sme_zero_za32_tile_0_);

                svbool_t const batch_predicate_b32 = svwhilelt_b32((uint32_t)0, (uint32_t)batch_size);
                for (nk_size_t row_in_tile = 0; row_in_tile < rows_remaining; row_in_tile++) {
                    nk_size_t const a_row = row_start + row_in_tile;
                    nk_e2m3_t const *a_src = a + a_row * a_stride_elements + depth_batch_start * expansion;
                    svuint8_t raw_bytes = svld1_u8(full_predicate_b8, (uint8_t const *)a_src);
                    svint8_t converted_i8 = nk_e2m3x_to_i8x_ssve_(full_predicate_b8, raw_bytes);
                    svst1_s8(full_predicate_b8, a_converted_i8x64[row_in_tile], converted_i8);
                    svld1_hor_za32(0, row_in_tile, batch_predicate_b32,
                                   (nk_f32_t const *)a_converted_i8x64[row_in_tile]);
                }

                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {

                    svint32_t a_column_i32 = svread_ver_za32_s32_m(svdup_s32(0), row_predicate_b32, 0, depth_step);
                    svint8_t a_interleaved_vector_i8 = svreinterpret_s8_s32(a_column_i32);

                    nk_i8_t const *b_packed_ptr = b_packed_base + (column_tile_index * depth_step_count +
                                                                   depth_batch_start + depth_step) *
                                                                      vector_elements;
                    svint8_t b_packed_vector = svld1_s8(column_predicate_b8, b_packed_ptr);

                    svmopa_za32_s8_m(1, row_predicate_b8, column_predicate_b8, a_interleaved_vector_i8,
                                     b_packed_vector);
                }
            }

            // Store results: convert i32 → f32 with 1/256 scaling
            for (nk_size_t row = 0; row < rows_remaining; row++) {
                svint32_t row_i32 = svread_hor_za32_s32_m(svdup_s32(0), column_predicate_b32, 1, row);
                svfloat32_t row_f32 = svmul_n_f32_x(column_predicate_b32,
                                                    svcvt_f32_s32_x(column_predicate_b32, row_i32), 1.0f / 256.0f);
                svst1_f32(column_predicate_b32, c + (row_start + row) * c_stride_elements + column_start, row_f32);
            }
        }
    }
}

NK_PUBLIC nk_size_t nk_dots_packed_size_e2m3_sme(nk_size_t n, nk_size_t k) {
    // Uses `i8` format for packed data (same tile geometry as i8)
    return nk_dots_packed_size_i8_sme(n, k);
}

NK_PUBLIC void nk_dots_pack_e2m3_sme(             //
    nk_e2m3_t const *b, nk_size_t n, nk_size_t k, //
    nk_size_t b_stride, void *b_packed) {

    nk_size_t const expansion = 4;
    nk_size_t const tile_dimension = svcntsw();
    nk_size_t const vector_elements = svcntsb(); // 64 = tile_dimension * expansion
    nk_size_t const b_stride_elements = b_stride / sizeof(nk_e2m3_t);

    nk_size_t const column_tile_count = nk_size_divide_round_up_(n, tile_dimension);
    nk_size_t const depth_step_count = nk_size_divide_round_up_(k, expansion);
    nk_size_t const total_vectors = column_tile_count * depth_step_count;

    nk_dots_sme_packed_header_t *header = (nk_dots_sme_packed_header_t *)b_packed;
    header->column_tile_count = (nk_u32_t)column_tile_count;
    header->depth_tile_count = (nk_u32_t)depth_step_count;
    header->columns = (nk_u32_t)n;
    header->depth = (nk_u32_t)k;
    header->svl_bytes = (nk_u32_t)(svcntsw() * sizeof(nk_i32_t));

    nk_i8_t *tiles_ptr = (nk_i8_t *)((char *)b_packed + sizeof(nk_dots_sme_packed_header_t));

    for (nk_size_t i = 0; i < total_vectors * vector_elements; i++) { tiles_ptr[i] = 0; }

    // Scalar e2m3 → signed i8 magnitude LUT
    static nk_u8_t const lut_magnitude[32] = {0,  2,  4,  6,  8,  10, 12, 14, 16, 20,  24,  28,  32, 36, 40, 44,
                                              48, 52, 56, 60, 64, 72, 80, 88, 96, 104, 112, 120, 0,  0,  0,  0};

    // Interleaved packing with e2m3 → i8 conversion
    for (nk_size_t column_tile = 0; column_tile < column_tile_count; column_tile++) {
        for (nk_size_t depth_step = 0; depth_step < depth_step_count; depth_step++) {
            nk_size_t const vec_index = column_tile * depth_step_count + depth_step;
            nk_i8_t *vec_output = tiles_ptr + vec_index * vector_elements;

            nk_size_t const b_row_start = column_tile * tile_dimension;
            nk_size_t const depth_base = depth_step * expansion;
            nk_size_t const rows_to_pack = (b_row_start + tile_dimension <= n) ? tile_dimension : (n - b_row_start);

            for (nk_size_t column_in_tile = 0; column_in_tile < rows_to_pack; column_in_tile++) {
                for (nk_size_t sub_element = 0; sub_element < expansion; sub_element++) {
                    nk_size_t const depth_idx = depth_base + sub_element;
                    if (depth_idx < k) {
                        nk_size_t const src_idx = (b_row_start + column_in_tile) * b_stride_elements + depth_idx;
                        nk_u8_t raw = b[src_idx];
                        nk_i8_t val = (nk_i8_t)lut_magnitude[raw & 0x1F];
                        if (raw & 0x20) val = -val;
                        vec_output[expansion * column_in_tile + sub_element] = val;
                    }
                }
            }
        }
    }
}

NK_PUBLIC void nk_dots_packed_e2m3_sme(                    //
    nk_e2m3_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,    //
    nk_size_t a_stride, nk_size_t c_stride) {

    nk_size_t const a_stride_elements = a_stride / sizeof(nk_e2m3_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);

    nk_dots_packed_e2m3_sme_kernel_(a, b_packed, c, rows, columns, depth, a_stride_elements, c_stride_elements);
}

/**
 *  `e2m3` × `e2m3` → `f32` symmetric kernel using SMOPA self-GEMM.
 *  Time-shares ZA0 for both A and B transposition with e2m3 → i8 conversion.
 *  Pre-reads A columns into Z registers, then reloads ZA0 with converted B data
 *  per column tile. Accumulates in i32, converts to f32 with 1/256 scaling.
 */
__arm_locally_streaming __arm_new("za") static void nk_dots_symmetric_e2m3_sme_kernel_(
    nk_e2m3_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride_elements, nk_f32_t *result,
    nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) {

    nk_size_t const expansion = 4;
    nk_size_t const tile_dimension = svcntw();
    nk_size_t const depth_step_count = nk_size_divide_round_up_(depth, expansion);
    nk_size_t const depth_steps_per_batch = tile_dimension;

    svbool_t const full_predicate_b8 = svptrue_b8();
    svbool_t const full_predicate_b32 = svptrue_b32();

    NK_ALIGN64 nk_i8_t a_converted_i8x64[16][64];
    NK_ALIGN64 nk_i8_t b_bounce_i8x64[64];
    NK_ALIGN64 nk_i32_t a_buffer[16][16];

    nk_size_t const row_end = row_start + row_count;
    nk_size_t const column_tile_count = nk_size_divide_round_up_(n_vectors, tile_dimension);

    for (nk_size_t row_tile_start = row_start; row_tile_start < row_end && row_tile_start < n_vectors;
         row_tile_start += tile_dimension) {
        nk_size_t const rows_clamped = (row_tile_start + tile_dimension <= row_end) ? tile_dimension
                                                                                    : (row_end - row_tile_start);
        nk_size_t const rows_actual = (row_tile_start + rows_clamped <= n_vectors) ? rows_clamped
                                                                                   : (n_vectors - row_tile_start);
        svbool_t const row_predicate_b8 = svwhilelt_b8((uint32_t)0, (uint32_t)(rows_actual * expansion));
        svbool_t const row_predicate_b32 = svwhilelt_b32((uint32_t)0, (uint32_t)rows_actual);

        nk_size_t column_tile_index = 0;

        for (; column_tile_index + 3 <= column_tile_count; column_tile_index += 3) {
            svzero_mask_za(nk_sme_zero_za32_tiles_123_);

            for (nk_size_t depth_batch_start = 0; depth_batch_start < depth_step_count;
                 depth_batch_start += depth_steps_per_batch) {
                nk_size_t const depth_batch_end = (depth_batch_start + depth_steps_per_batch < depth_step_count)
                                                      ? depth_batch_start + depth_steps_per_batch
                                                      : depth_step_count;
                nk_size_t const batch_size = depth_batch_end - depth_batch_start;

                // ZA transpose for A rows: convert e2m3 → i8 then load
                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                svbool_t const batch_predicate_b32 = svwhilelt_b32((uint32_t)0, (uint32_t)batch_size);
                for (nk_size_t row_in_tile = 0; row_in_tile < rows_actual; row_in_tile++) {
                    nk_size_t const row_abs = row_tile_start + row_in_tile;
                    nk_e2m3_t const *a_src = vectors + row_abs * stride_elements + depth_batch_start * expansion;
                    svuint8_t raw_bytes = svld1_u8(full_predicate_b8, (uint8_t const *)a_src);
                    svint8_t converted_i8 = nk_e2m3x_to_i8x_ssve_(full_predicate_b8, raw_bytes);
                    svst1_s8(full_predicate_b8, a_converted_i8x64[row_in_tile], converted_i8);
                    svld1_hor_za32(0, row_in_tile, batch_predicate_b32,
                                   (nk_f32_t const *)a_converted_i8x64[row_in_tile]);
                }

                // Save A columns from ZA0 to stack buffer
                for (nk_size_t s = 0; s < batch_size; s++)
                    svst1_s32(full_predicate_b32, a_buffer[s],
                              svread_ver_za32_s32_m(svdup_s32(0), row_predicate_b32, 0, s));

                svbool_t const depth_predicate_b8 = svwhilelt_b8((uint32_t)0, (uint32_t)(batch_size * expansion));

                // Load B column tile 0 into ZA0, vertical read + SMOPA into ZA1
                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t column = 0; column < tile_dimension; column++) {
                    nk_size_t const column_abs = (column_tile_index + 0) * tile_dimension + column;
                    if (column_abs < n_vectors) {
                        nk_e2m3_t const *src = &vectors[column_abs * stride_elements + depth_batch_start * expansion];
                        svuint8_t raw = svld1_u8(depth_predicate_b8, (uint8_t const *)src);
                        svint8_t converted_i8 = nk_e2m3x_to_i8x_ssve_(depth_predicate_b8, raw);
                        svst1_s8(depth_predicate_b8, b_bounce_i8x64, converted_i8);
                        svld1_hor_za32(0, column, batch_predicate_b32, (nk_f32_t const *)b_bounce_i8x64);
                    }
                }
                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {
                    svint8_t a_interleaved = svreinterpret_s8_s32(svld1_s32(full_predicate_b32, a_buffer[depth_step]));
                    svint32_t b_column_i32 = svread_ver_za32_s32_m(svdup_s32(0), full_predicate_b32, 0, depth_step);
                    svint8_t b_interleaved = svreinterpret_s8_s32(b_column_i32);
                    svmopa_za32_s8_m(1, row_predicate_b8, full_predicate_b8, a_interleaved, b_interleaved);
                }

                // Load B column tile 1 into ZA0, vertical read + SMOPA into ZA2
                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t column = 0; column < tile_dimension; column++) {
                    nk_size_t const column_abs = (column_tile_index + 1) * tile_dimension + column;
                    if (column_abs < n_vectors) {
                        nk_e2m3_t const *src = &vectors[column_abs * stride_elements + depth_batch_start * expansion];
                        svuint8_t raw = svld1_u8(depth_predicate_b8, (uint8_t const *)src);
                        svint8_t converted_i8 = nk_e2m3x_to_i8x_ssve_(depth_predicate_b8, raw);
                        svst1_s8(depth_predicate_b8, b_bounce_i8x64, converted_i8);
                        svld1_hor_za32(0, column, batch_predicate_b32, (nk_f32_t const *)b_bounce_i8x64);
                    }
                }
                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {
                    svint8_t a_interleaved = svreinterpret_s8_s32(svld1_s32(full_predicate_b32, a_buffer[depth_step]));
                    svint32_t b_column_i32 = svread_ver_za32_s32_m(svdup_s32(0), full_predicate_b32, 0, depth_step);
                    svint8_t b_interleaved = svreinterpret_s8_s32(b_column_i32);
                    svmopa_za32_s8_m(2, row_predicate_b8, full_predicate_b8, a_interleaved, b_interleaved);
                }

                // Load B column tile 2 into ZA0, vertical read + SMOPA into ZA3
                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t column = 0; column < tile_dimension; column++) {
                    nk_size_t const column_abs = (column_tile_index + 2) * tile_dimension + column;
                    if (column_abs < n_vectors) {
                        nk_e2m3_t const *src = &vectors[column_abs * stride_elements + depth_batch_start * expansion];
                        svuint8_t raw = svld1_u8(depth_predicate_b8, (uint8_t const *)src);
                        svint8_t converted_i8 = nk_e2m3x_to_i8x_ssve_(depth_predicate_b8, raw);
                        svst1_s8(depth_predicate_b8, b_bounce_i8x64, converted_i8);
                        svld1_hor_za32(0, column, batch_predicate_b32, (nk_f32_t const *)b_bounce_i8x64);
                    }
                }
                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {
                    svint8_t a_interleaved = svreinterpret_s8_s32(svld1_s32(full_predicate_b32, a_buffer[depth_step]));
                    svint32_t b_column_i32 = svread_ver_za32_s32_m(svdup_s32(0), full_predicate_b32, 0, depth_step);
                    svint8_t b_interleaved = svreinterpret_s8_s32(b_column_i32);
                    svmopa_za32_s8_m(3, row_predicate_b8, full_predicate_b8, a_interleaved, b_interleaved);
                }
            }

            // Store results: convert i32 → f32 with 1/256 scaling
            for (nk_size_t row = 0; row < rows_actual; row++) {
                nk_size_t const row_abs = row_tile_start + row;
                nk_f32_t *result_row = result + row_abs * result_stride_elements;

                svint32_t row_i32_1 = svread_hor_za32_s32_m(svdup_s32(0), full_predicate_b32, 1, row);
                svint32_t row_i32_2 = svread_hor_za32_s32_m(svdup_s32(0), full_predicate_b32, 2, row);
                svint32_t row_i32_3 = svread_hor_za32_s32_m(svdup_s32(0), full_predicate_b32, 3, row);
                svfloat32_t row_f32_1 = svmul_n_f32_x(full_predicate_b32,
                                                      svcvt_f32_s32_x(full_predicate_b32, row_i32_1), 1.0f / 256.0f);
                svfloat32_t row_f32_2 = svmul_n_f32_x(full_predicate_b32,
                                                      svcvt_f32_s32_x(full_predicate_b32, row_i32_2), 1.0f / 256.0f);
                svfloat32_t row_f32_3 = svmul_n_f32_x(full_predicate_b32,
                                                      svcvt_f32_s32_x(full_predicate_b32, row_i32_3), 1.0f / 256.0f);
                svst1_f32(full_predicate_b32, result_row + (column_tile_index + 0) * tile_dimension, row_f32_1);
                svst1_f32(full_predicate_b32, result_row + (column_tile_index + 1) * tile_dimension, row_f32_2);
                svst1_f32(full_predicate_b32, result_row + (column_tile_index + 2) * tile_dimension, row_f32_3);
            }
        }

        for (; column_tile_index < column_tile_count; column_tile_index++) {
            nk_size_t const column_tile_start = column_tile_index * tile_dimension;
            nk_size_t const cols_remaining = (column_tile_start + tile_dimension <= n_vectors)
                                                 ? tile_dimension
                                                 : (n_vectors - column_tile_start);
            svbool_t const column_predicate_b32 = svwhilelt_b32((uint32_t)0, (uint32_t)cols_remaining);
            svbool_t const column_predicate_b8 = svwhilelt_b8((uint32_t)0, (uint32_t)(cols_remaining * expansion));

            svzero_mask_za(nk_sme_zero_za32_tile_1_);

            for (nk_size_t depth_batch_start = 0; depth_batch_start < depth_step_count;
                 depth_batch_start += depth_steps_per_batch) {
                nk_size_t const depth_batch_end = (depth_batch_start + depth_steps_per_batch < depth_step_count)
                                                      ? depth_batch_start + depth_steps_per_batch
                                                      : depth_step_count;
                nk_size_t const batch_size = depth_batch_end - depth_batch_start;

                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                svbool_t const batch_predicate_b32 = svwhilelt_b32((uint32_t)0, (uint32_t)batch_size);
                for (nk_size_t row_in_tile = 0; row_in_tile < rows_actual; row_in_tile++) {
                    nk_size_t const row_abs = row_tile_start + row_in_tile;
                    nk_e2m3_t const *a_src = vectors + row_abs * stride_elements + depth_batch_start * expansion;
                    svuint8_t raw_bytes = svld1_u8(full_predicate_b8, (uint8_t const *)a_src);
                    svint8_t converted_i8 = nk_e2m3x_to_i8x_ssve_(full_predicate_b8, raw_bytes);
                    svst1_s8(full_predicate_b8, a_converted_i8x64[row_in_tile], converted_i8);
                    svld1_hor_za32(0, row_in_tile, batch_predicate_b32,
                                   (nk_f32_t const *)a_converted_i8x64[row_in_tile]);
                }

                // Save A columns from ZA0 to stack buffer
                for (nk_size_t s = 0; s < batch_size; s++)
                    svst1_s32(full_predicate_b32, a_buffer[s],
                              svread_ver_za32_s32_m(svdup_s32(0), row_predicate_b32, 0, s));

                svbool_t const depth_predicate_b8 = svwhilelt_b8((uint32_t)0, (uint32_t)(batch_size * expansion));

                // Load B column tile into ZA0, vertical read + SMOPA into ZA1
                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t column = 0; column < tile_dimension; column++) {
                    nk_size_t const column_abs = column_tile_start + column;
                    if (column_abs < n_vectors) {
                        nk_e2m3_t const *src = &vectors[column_abs * stride_elements + depth_batch_start * expansion];
                        svuint8_t raw = svld1_u8(depth_predicate_b8, (uint8_t const *)src);
                        svint8_t converted_i8 = nk_e2m3x_to_i8x_ssve_(depth_predicate_b8, raw);
                        svst1_s8(depth_predicate_b8, b_bounce_i8x64, converted_i8);
                        svld1_hor_za32(0, column, batch_predicate_b32, (nk_f32_t const *)b_bounce_i8x64);
                    }
                }
                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {
                    svint8_t a_interleaved = svreinterpret_s8_s32(svld1_s32(full_predicate_b32, a_buffer[depth_step]));
                    svint32_t b_column_i32 = svread_ver_za32_s32_m(svdup_s32(0), column_predicate_b32, 0, depth_step);
                    svint8_t b_interleaved = svreinterpret_s8_s32(b_column_i32);
                    svmopa_za32_s8_m(1, row_predicate_b8, column_predicate_b8, a_interleaved, b_interleaved);
                }
            }

            // Store results: convert i32 → f32 with 1/256 scaling
            for (nk_size_t row = 0; row < rows_actual; row++) {
                nk_size_t const row_abs = row_tile_start + row;
                svint32_t row_i32 = svread_hor_za32_s32_m(svdup_s32(0), column_predicate_b32, 1, row);
                svfloat32_t row_f32 = svmul_n_f32_x(column_predicate_b32,
                                                    svcvt_f32_s32_x(column_predicate_b32, row_i32), 1.0f / 256.0f);
                svst1_f32(column_predicate_b32, result + row_abs * result_stride_elements + column_tile_start, row_f32);
            }
        }

        for (nk_size_t row = 0; row < rows_actual; row++) {
            nk_size_t const i = row_tile_start + row;
            for (nk_size_t j = 0; j < i && j < n_vectors; j++)
                result[j * result_stride_elements + i] = result[i * result_stride_elements + j];
        }
    }
}

NK_PUBLIC void nk_dots_symmetric_e2m3_sme(nk_e2m3_t const *vectors, nk_size_t n_vectors, nk_size_t depth,
                                          nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                          nk_size_t row_start, nk_size_t row_count) {

    nk_size_t const stride_elements = stride / sizeof(nk_e2m3_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_dots_symmetric_e2m3_sme_kernel_(vectors, n_vectors, depth, stride_elements, result, result_stride_elements,
                                       row_start, row_count);
}

#pragma endregion

/*
 *  e3m2 × e3m2 → f32 GEMM using inline SSVE conversion + FMOPA.
 *
 *  Pipeline: e3m2 bytes → extract sign/exp/mant → rebuild f16 → FMOPA → f32
 *  - Same tile layout as e5m2 (both convert to f16 before FMOPA)
 *  - E3M2 has 3-bit exponent (bias=3), 2-bit mantissa; requires rebiasing to f16
 *  - Handles subnormals (exp=0, mant!=0) via integer→float conversion + scaling
 *  - No infinity or NaN in E3M2FN format
 */
#pragma region Quarter Precision E3M2

/**
 *  Inline `e3m2` → `f16` conversion returning `svfloat16_t` for direct use in GEMM.
 *
 *  E3M2 encoding (6 bits in low byte):
 *    bit 5 = sign, bits 4:2 = exponent (bias=3), bits 1:0 = mantissa
 *
 *  Conversion to f16 (1+5+10 bits, bias=15):
 *    Normal (exp!=0): f16 = sign16 | ((exp3 + 12) << 10) | (mant2 << 8)
 *    Subnormal (exp==0, mant!=0): convert mantissa to f16 float, multiply by 2^(-4)
 *    Zero (exp==0, mant==0): signed zero
 *
 *  @param pg16      Predicate for 16-bit elements
 *  @param bytes     Pre-loaded bytes (svuint8_t from svld1_u8)
 *  @return          F16 values as svfloat16_t (from lower half of bytes via unpack)
 */
NK_INTERNAL svfloat16_t nk_e3m2x_to_f16x_ssve_(svbool_t pg16, svuint8_t bytes) {
    svuint16_t vals = svunpklo_u16(bytes);

    // Extract sign (bit 5 → bit 15), exponent (bits 4:2), mantissa (bits 1:0)
    svuint16_t sign = svlsl_n_u16_x(pg16, svand_n_u16_x(pg16, vals, 0x20), 10); // bit5 → bit15
    svuint16_t exp3 = svand_n_u16_x(pg16, svlsr_n_u16_x(pg16, vals, 2), 0x07);
    svuint16_t mant = svand_n_u16_x(pg16, vals, 0x03);

    // Normal path: f16 = sign | ((exp3 + 12) << 10) | (mant << 8)
    svuint16_t normal = svorr_u16_x(pg16, svlsl_n_u16_x(pg16, svadd_n_u16_x(pg16, exp3, 12), 10),
                                    svlsl_n_u16_x(pg16, mant, 8));
    normal = svorr_u16_x(pg16, normal, sign);

    // Subnormal path: value = mant * 2^(-4), where 2^(-4) in f16 = 0x2800
    svfloat16_t mant_f16 = svcvt_f16_u16_x(pg16, mant);
    svfloat16_t scale = svreinterpret_f16_u16(svdup_n_u16(0x2800)); // 2^(-4)
    svfloat16_t subnorm_abs = svmul_f16_x(pg16, mant_f16, scale);
    svuint16_t subnorm = svorr_u16_x(pg16, svreinterpret_u16_f16(subnorm_abs), sign);

    // Select: subnormal when exp3 == 0, normal otherwise
    svbool_t is_subnorm = svcmpeq_n_u16(pg16, exp3, 0);
    svuint16_t result = svsel_u16(is_subnorm, subnorm, normal);

    return svreinterpret_f16_u16(result);
}

/**
 *  Fused `e3m2` × `e3m2` → `f32` GEMM kernel using interleaved FMOPA.
 *  Converts `e3m2` → `f16` on-the-fly for A, B is pre-converted during packing.
 */
__arm_locally_streaming __arm_new("za") __attribute__((noinline)) static void nk_dots_packed_e3m2_sme_kernel_( //
    nk_e3m2_t const *a, void const *b_packed, nk_f32_t *c,                                                     //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,                                                        //
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) {

    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_size_t const column_tile_count = header->column_tile_count;
    nk_size_t const depth_step_count = header->depth_tile_count;

    nk_size_t const expansion = 2;
    nk_size_t const tile_dimension = svcntw();
    nk_size_t const vector_elements = svcnth();
    nk_size_t const depth_steps_per_batch = tile_dimension;

    nk_f16_t const *b_packed_base = (nk_f16_t const *)((char const *)b_packed + sizeof(nk_dots_sme_packed_header_t));

    svbool_t const full_predicate_b16 = svptrue_b16();
    svbool_t const full_predicate_b8 = svptrue_b8();
    svbool_t const full_predicate_b32 = svptrue_b32();

    NK_ALIGN64 nk_f16_t a_converted_f16x32[16][32];

    nk_size_t const row_tile_count = nk_size_divide_round_up_(rows, tile_dimension);

    for (nk_size_t row_tile_index = 0; row_tile_index < row_tile_count; row_tile_index++) {
        nk_size_t const row_start = row_tile_index * tile_dimension;
        nk_size_t const rows_remaining = (row_start + tile_dimension <= rows) ? tile_dimension : (rows - row_start);
        svbool_t const row_predicate_b16 = svwhilelt_b16((uint32_t)0, (uint32_t)(rows_remaining * expansion));
        svbool_t const row_predicate_b32 = svwhilelt_b32((uint32_t)0, (uint32_t)rows_remaining);

        nk_size_t column_tile_index = 0;

        // Fast path: 3 column tiles using ZA1-ZA3 (ZA0 = staging)
        for (; column_tile_index + 3 <= column_tile_count; column_tile_index += 3) {
            svzero_mask_za(nk_sme_zero_za32_tiles_123_);

            for (nk_size_t depth_batch_start = 0; depth_batch_start < depth_step_count;
                 depth_batch_start += depth_steps_per_batch) {
                nk_size_t const depth_batch_end = (depth_batch_start + depth_steps_per_batch < depth_step_count)
                                                      ? depth_batch_start + depth_steps_per_batch
                                                      : depth_step_count;
                nk_size_t const batch_size = depth_batch_end - depth_batch_start;

                svzero_mask_za(nk_sme_zero_za32_tile_0_);

                svbool_t const batch_predicate_b32 = svwhilelt_b32((uint32_t)0, (uint32_t)batch_size);
                for (nk_size_t row_in_tile = 0; row_in_tile < rows_remaining; row_in_tile++) {
                    nk_size_t const a_row = row_start + row_in_tile;
                    // Vectorized e3m2 → f16 conversion
                    nk_e3m2_t const *a_src = a + a_row * a_stride_elements + depth_batch_start * expansion;
                    svuint8_t raw_bytes = svld1_u8(full_predicate_b8, (uint8_t const *)a_src);
                    svfloat16_t converted_f16 = nk_e3m2x_to_f16x_ssve_(full_predicate_b16, raw_bytes);
                    svst1_f16(full_predicate_b16, (float16_t *)a_converted_f16x32[row_in_tile], converted_f16);
                    svld1_hor_za32(0, row_in_tile, batch_predicate_b32,
                                   (nk_f32_t const *)a_converted_f16x32[row_in_tile]);
                }

                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {

                    svfloat32_t a_column_f32 = svread_ver_za32_f32_m(svdup_f32(0.0f), row_predicate_b32, 0, depth_step);
                    svfloat16_t a_interleaved_vector_f16 = svreinterpret_f16_f32(a_column_f32);

                    nk_f16_t const *b_packed_ptr_0 = b_packed_base + ((column_tile_index + 0) * depth_step_count +
                                                                      depth_batch_start + depth_step) *
                                                                         vector_elements;
                    nk_f16_t const *b_packed_ptr_1 = b_packed_base + ((column_tile_index + 1) * depth_step_count +
                                                                      depth_batch_start + depth_step) *
                                                                         vector_elements;
                    nk_f16_t const *b_packed_ptr_2 = b_packed_base + ((column_tile_index + 2) * depth_step_count +
                                                                      depth_batch_start + depth_step) *
                                                                         vector_elements;
                    svfloat16_t b_packed_vector_0 = svld1_f16(full_predicate_b16, (float16_t const *)b_packed_ptr_0);
                    svfloat16_t b_packed_vector_1 = svld1_f16(full_predicate_b16, (float16_t const *)b_packed_ptr_1);
                    svfloat16_t b_packed_vector_2 = svld1_f16(full_predicate_b16, (float16_t const *)b_packed_ptr_2);

                    svmopa_za32_f16_m(1, row_predicate_b16, full_predicate_b16, a_interleaved_vector_f16,
                                      b_packed_vector_0);
                    svmopa_za32_f16_m(2, row_predicate_b16, full_predicate_b16, a_interleaved_vector_f16,
                                      b_packed_vector_1);
                    svmopa_za32_f16_m(3, row_predicate_b16, full_predicate_b16, a_interleaved_vector_f16,
                                      b_packed_vector_2);
                }
            }

            for (nk_size_t row = 0; row < rows_remaining; row++) {
                nk_size_t const cs0 = (column_tile_index + 0) * tile_dimension;
                nk_size_t const cs1 = (column_tile_index + 1) * tile_dimension;
                nk_size_t const cs2 = (column_tile_index + 2) * tile_dimension;
                svst1_hor_za32(1, row, full_predicate_b32, c + (row_start + row) * c_stride_elements + cs0);
                svst1_hor_za32(2, row, full_predicate_b32, c + (row_start + row) * c_stride_elements + cs1);
                svst1_hor_za32(3, row, full_predicate_b32, c + (row_start + row) * c_stride_elements + cs2);
            }
        }

        // Remainder: 1 column tile using ZA1
        for (; column_tile_index < column_tile_count; column_tile_index++) {
            nk_size_t const column_start = column_tile_index * tile_dimension;
            nk_size_t const cols_remaining = (column_start + tile_dimension <= columns) ? tile_dimension
                                                                                        : (columns - column_start);
            svbool_t const column_predicate_b32 = svwhilelt_b32((uint32_t)0, (uint32_t)cols_remaining);
            svbool_t const column_predicate_b16 = svwhilelt_b16((uint32_t)0, (uint32_t)(cols_remaining * expansion));

            svzero_mask_za(nk_sme_zero_za32_tile_1_);

            for (nk_size_t depth_batch_start = 0; depth_batch_start < depth_step_count;
                 depth_batch_start += depth_steps_per_batch) {
                nk_size_t const depth_batch_end = (depth_batch_start + depth_steps_per_batch < depth_step_count)
                                                      ? depth_batch_start + depth_steps_per_batch
                                                      : depth_step_count;
                nk_size_t const batch_size = depth_batch_end - depth_batch_start;

                svzero_mask_za(nk_sme_zero_za32_tile_0_);

                svbool_t const batch_predicate_b32 = svwhilelt_b32((uint32_t)0, (uint32_t)batch_size);
                for (nk_size_t row_in_tile = 0; row_in_tile < rows_remaining; row_in_tile++) {
                    nk_size_t const a_row = row_start + row_in_tile;
                    nk_e3m2_t const *a_src = a + a_row * a_stride_elements + depth_batch_start * expansion;
                    svuint8_t raw_bytes = svld1_u8(full_predicate_b8, (uint8_t const *)a_src);
                    svfloat16_t converted_f16 = nk_e3m2x_to_f16x_ssve_(full_predicate_b16, raw_bytes);
                    svst1_f16(full_predicate_b16, (float16_t *)a_converted_f16x32[row_in_tile], converted_f16);
                    svld1_hor_za32(0, row_in_tile, batch_predicate_b32,
                                   (nk_f32_t const *)a_converted_f16x32[row_in_tile]);
                }

                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {

                    svfloat32_t a_column_f32 = svread_ver_za32_f32_m(svdup_f32(0.0f), row_predicate_b32, 0, depth_step);
                    svfloat16_t a_interleaved_vector_f16 = svreinterpret_f16_f32(a_column_f32);

                    nk_f16_t const *b_packed_ptr = b_packed_base + (column_tile_index * depth_step_count +
                                                                    depth_batch_start + depth_step) *
                                                                       vector_elements;
                    svfloat16_t b_packed_vector = svld1_f16(column_predicate_b16, (float16_t const *)b_packed_ptr);

                    svmopa_za32_f16_m(1, row_predicate_b16, column_predicate_b16, a_interleaved_vector_f16,
                                      b_packed_vector);
                }
            }

            for (nk_size_t row = 0; row < rows_remaining; row++) {
                svst1_hor_za32(1, row, column_predicate_b32, c + (row_start + row) * c_stride_elements + column_start);
            }
        }
    }
}

NK_PUBLIC nk_size_t nk_dots_packed_size_e3m2_sme(nk_size_t n, nk_size_t k) { return nk_dots_packed_size_f16_sme(n, k); }

NK_PUBLIC void nk_dots_pack_e3m2_sme(nk_e3m2_t const *b, nk_size_t n, nk_size_t k, nk_size_t b_stride, void *b_packed) {

    nk_size_t const expansion = 2;
    nk_size_t const tile_dimension = svcntsw();
    nk_size_t const vector_elements = svcntsh();
    nk_size_t const b_stride_elements = b_stride / sizeof(nk_e3m2_t);

    nk_size_t const column_tile_count = nk_size_divide_round_up_(n, tile_dimension);
    nk_size_t const depth_step_count = nk_size_divide_round_up_(k, expansion);
    nk_size_t const total_vectors = column_tile_count * depth_step_count;

    nk_dots_sme_packed_header_t *header = (nk_dots_sme_packed_header_t *)b_packed;
    header->column_tile_count = (nk_u32_t)column_tile_count;
    header->depth_tile_count = (nk_u32_t)depth_step_count;
    header->columns = (nk_u32_t)n;
    header->depth = (nk_u32_t)k;
    header->svl_bytes = (nk_u32_t)(svcntsw() * sizeof(nk_f32_t));

    nk_f16_t *tiles_ptr = (nk_f16_t *)((char *)b_packed + sizeof(nk_dots_sme_packed_header_t));

    for (nk_size_t i = 0; i < total_vectors * vector_elements; i++) tiles_ptr[i] = 0;

    // Interleaved packing with e3m2 → f16 conversion (via f32 intermediate)
    for (nk_size_t column_tile = 0; column_tile < column_tile_count; column_tile++) {
        for (nk_size_t depth_step = 0; depth_step < depth_step_count; depth_step++) {
            nk_size_t const vec_index = column_tile * depth_step_count + depth_step;
            nk_f16_t *vec_output = tiles_ptr + vec_index * vector_elements;

            nk_size_t const b_row_start = column_tile * tile_dimension;
            nk_size_t const depth_base = depth_step * expansion;
            nk_size_t const rows_to_pack = (b_row_start + tile_dimension <= n) ? tile_dimension : (n - b_row_start);

            for (nk_size_t column_in_tile = 0; column_in_tile < rows_to_pack; column_in_tile++) {
                for (nk_size_t sub_element = 0; sub_element < expansion; sub_element++) {
                    nk_size_t const depth_idx = depth_base + sub_element;
                    if (depth_idx < k) {
                        nk_size_t const src_idx = (b_row_start + column_in_tile) * b_stride_elements + depth_idx;
                        nk_f32_t temp_f32;
                        nk_e3m2_to_f32_serial(&b[src_idx], &temp_f32);
                        nk_f32_to_f16_serial(&temp_f32, &vec_output[expansion * column_in_tile + sub_element]);
                    }
                }
            }
        }
    }
}

/*  `e3m2` × `e3m2` → `f32` GEMM: public interface.
 *  Predicate-based edge handling eliminates scalar fallbacks.
 */
NK_PUBLIC void nk_dots_packed_e3m2_sme(                    //
    nk_e3m2_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,    //
    nk_size_t a_stride, nk_size_t c_stride) {

    nk_size_t const a_stride_elements = a_stride / sizeof(nk_e3m2_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);

    nk_dots_packed_e3m2_sme_kernel_(a, b_packed, c, rows, columns, depth, a_stride_elements, c_stride_elements);
}

/**
 * `e3m2` × `e3m2` → `f32` symmetric kernel using MOPA self-GEMM.
 *  Time-shares ZA0 for both A and B transposition with e3m2 → f16 conversion.
 *  Pre-reads A columns into Z registers, then reloads ZA0 with converted B data
 *  per column tile. Eliminates all scalar B-packing loops.
 */
__arm_locally_streaming __arm_new("za") static void nk_dots_symmetric_e3m2_sme_kernel_(
    nk_e3m2_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride_elements, nk_f32_t *result,
    nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) {

    nk_size_t const expansion = 2;
    nk_size_t const tile_dimension = svcntw();
    nk_size_t const depth_step_count = nk_size_divide_round_up_(depth, expansion);
    nk_size_t const depth_steps_per_batch = tile_dimension;

    svbool_t const full_predicate_b16 = svptrue_b16();
    svbool_t const full_predicate_b8 = svptrue_b8();
    svbool_t const full_predicate_b32 = svptrue_b32();

    NK_ALIGN64 nk_f16_t a_converted_f16x32[16][32];
    NK_ALIGN64 nk_f16_t b_bounce_f16x32[32];
    NK_ALIGN64 nk_f32_t a_buffer[16][16];

    nk_size_t const row_end = row_start + row_count;
    nk_size_t const column_tile_count = nk_size_divide_round_up_(n_vectors, tile_dimension);

    for (nk_size_t row_tile_start = row_start; row_tile_start < row_end && row_tile_start < n_vectors;
         row_tile_start += tile_dimension) {
        nk_size_t const rows_clamped = (row_tile_start + tile_dimension <= row_end) ? tile_dimension
                                                                                    : (row_end - row_tile_start);
        nk_size_t const rows_actual = (row_tile_start + rows_clamped <= n_vectors) ? rows_clamped
                                                                                   : (n_vectors - row_tile_start);
        svbool_t const row_predicate_b16 = svwhilelt_b16((uint32_t)0, (uint32_t)(rows_actual * expansion));
        svbool_t const row_predicate_b32 = svwhilelt_b32((uint32_t)0, (uint32_t)rows_actual);

        nk_size_t column_tile_index = 0;

        for (; column_tile_index + 3 <= column_tile_count; column_tile_index += 3) {
            svzero_mask_za(nk_sme_zero_za32_tiles_123_);

            for (nk_size_t depth_batch_start = 0; depth_batch_start < depth_step_count;
                 depth_batch_start += depth_steps_per_batch) {
                nk_size_t const depth_batch_end = (depth_batch_start + depth_steps_per_batch < depth_step_count)
                                                      ? depth_batch_start + depth_steps_per_batch
                                                      : depth_step_count;
                nk_size_t const batch_size = depth_batch_end - depth_batch_start;

                // ZA transpose for A rows: convert e3m2 → f16 then load
                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                svbool_t const batch_predicate_b32 = svwhilelt_b32((uint32_t)0, (uint32_t)batch_size);
                for (nk_size_t row_in_tile = 0; row_in_tile < rows_actual; row_in_tile++) {
                    nk_size_t const row_abs = row_tile_start + row_in_tile;
                    nk_e3m2_t const *a_src = vectors + row_abs * stride_elements + depth_batch_start * expansion;
                    svuint8_t raw_bytes = svld1_u8(full_predicate_b8, (uint8_t const *)a_src);
                    svfloat16_t converted_f16 = nk_e3m2x_to_f16x_ssve_(full_predicate_b16, raw_bytes);
                    svst1_f16(full_predicate_b16, (float16_t *)a_converted_f16x32[row_in_tile], converted_f16);
                    svld1_hor_za32(0, row_in_tile, batch_predicate_b32,
                                   (nk_f32_t const *)a_converted_f16x32[row_in_tile]);
                }

                // Save A columns from ZA0 to stack buffer
                for (nk_size_t s = 0; s < batch_size; s++)
                    svst1_f32(full_predicate_b32, a_buffer[s],
                              svread_ver_za32_f32_m(svdup_f32(0), row_predicate_b32, 0, s));

                svbool_t const depth_predicate_b8 = svwhilelt_b8((uint32_t)0, (uint32_t)(batch_size * expansion));
                svbool_t const depth_predicate_b16 = svwhilelt_b16((uint32_t)0, (uint32_t)(batch_size * expansion));

                // Load B column tile 0 into ZA0, vertical read + FMOPA into ZA1
                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t column = 0; column < tile_dimension; column++) {
                    nk_size_t const column_abs = (column_tile_index + 0) * tile_dimension + column;
                    if (column_abs < n_vectors) {
                        nk_e3m2_t const *src = &vectors[column_abs * stride_elements + depth_batch_start * expansion];
                        svuint8_t raw = svld1_u8(depth_predicate_b8, (uint8_t const *)src);
                        svfloat16_t converted_f16 = nk_e3m2x_to_f16x_ssve_(depth_predicate_b16, raw);
                        svst1_f16(depth_predicate_b16, (float16_t *)b_bounce_f16x32, converted_f16);
                        svld1_hor_za32(0, column, batch_predicate_b32, (nk_f32_t const *)b_bounce_f16x32);
                    }
                }
                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {
                    svfloat16_t a_interleaved = svreinterpret_f16_f32(
                        svld1_f32(full_predicate_b32, a_buffer[depth_step]));
                    svfloat32_t b_column_f32 = svread_ver_za32_f32_m(svdup_f32(0.0f), full_predicate_b32, 0,
                                                                     depth_step);
                    svfloat16_t b_interleaved = svreinterpret_f16_f32(b_column_f32);
                    svmopa_za32_f16_m(1, row_predicate_b16, full_predicate_b16, a_interleaved, b_interleaved);
                }

                // Load B column tile 1 into ZA0, vertical read + FMOPA into ZA2
                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t column = 0; column < tile_dimension; column++) {
                    nk_size_t const column_abs = (column_tile_index + 1) * tile_dimension + column;
                    if (column_abs < n_vectors) {
                        nk_e3m2_t const *src = &vectors[column_abs * stride_elements + depth_batch_start * expansion];
                        svuint8_t raw = svld1_u8(depth_predicate_b8, (uint8_t const *)src);
                        svfloat16_t converted_f16 = nk_e3m2x_to_f16x_ssve_(depth_predicate_b16, raw);
                        svst1_f16(depth_predicate_b16, (float16_t *)b_bounce_f16x32, converted_f16);
                        svld1_hor_za32(0, column, batch_predicate_b32, (nk_f32_t const *)b_bounce_f16x32);
                    }
                }
                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {
                    svfloat16_t a_interleaved = svreinterpret_f16_f32(
                        svld1_f32(full_predicate_b32, a_buffer[depth_step]));
                    svfloat32_t b_column_f32 = svread_ver_za32_f32_m(svdup_f32(0.0f), full_predicate_b32, 0,
                                                                     depth_step);
                    svfloat16_t b_interleaved = svreinterpret_f16_f32(b_column_f32);
                    svmopa_za32_f16_m(2, row_predicate_b16, full_predicate_b16, a_interleaved, b_interleaved);
                }

                // Load B column tile 2 into ZA0, vertical read + FMOPA into ZA3
                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t column = 0; column < tile_dimension; column++) {
                    nk_size_t const column_abs = (column_tile_index + 2) * tile_dimension + column;
                    if (column_abs < n_vectors) {
                        nk_e3m2_t const *src = &vectors[column_abs * stride_elements + depth_batch_start * expansion];
                        svuint8_t raw = svld1_u8(depth_predicate_b8, (uint8_t const *)src);
                        svfloat16_t converted_f16 = nk_e3m2x_to_f16x_ssve_(depth_predicate_b16, raw);
                        svst1_f16(depth_predicate_b16, (float16_t *)b_bounce_f16x32, converted_f16);
                        svld1_hor_za32(0, column, batch_predicate_b32, (nk_f32_t const *)b_bounce_f16x32);
                    }
                }
                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {
                    svfloat16_t a_interleaved = svreinterpret_f16_f32(
                        svld1_f32(full_predicate_b32, a_buffer[depth_step]));
                    svfloat32_t b_column_f32 = svread_ver_za32_f32_m(svdup_f32(0.0f), full_predicate_b32, 0,
                                                                     depth_step);
                    svfloat16_t b_interleaved = svreinterpret_f16_f32(b_column_f32);
                    svmopa_za32_f16_m(3, row_predicate_b16, full_predicate_b16, a_interleaved, b_interleaved);
                }
            }

            for (nk_size_t row = 0; row < rows_actual; row++) {
                nk_size_t const row_abs = row_tile_start + row;
                nk_f32_t *result_row = result + row_abs * result_stride_elements;
                svst1_hor_za32(1, row, full_predicate_b32, result_row + (column_tile_index + 0) * tile_dimension);
                svst1_hor_za32(2, row, full_predicate_b32, result_row + (column_tile_index + 1) * tile_dimension);
                svst1_hor_za32(3, row, full_predicate_b32, result_row + (column_tile_index + 2) * tile_dimension);
            }
        }

        for (; column_tile_index < column_tile_count; column_tile_index++) {
            nk_size_t const column_tile_start = column_tile_index * tile_dimension;
            nk_size_t const cols_remaining = (column_tile_start + tile_dimension <= n_vectors)
                                                 ? tile_dimension
                                                 : (n_vectors - column_tile_start);
            svbool_t const column_predicate_b32 = svwhilelt_b32((uint32_t)0, (uint32_t)cols_remaining);
            svbool_t const column_predicate_b16 = svwhilelt_b16((uint32_t)0, (uint32_t)(cols_remaining * expansion));

            svzero_mask_za(nk_sme_zero_za32_tile_1_);

            for (nk_size_t depth_batch_start = 0; depth_batch_start < depth_step_count;
                 depth_batch_start += depth_steps_per_batch) {
                nk_size_t const depth_batch_end = (depth_batch_start + depth_steps_per_batch < depth_step_count)
                                                      ? depth_batch_start + depth_steps_per_batch
                                                      : depth_step_count;
                nk_size_t const batch_size = depth_batch_end - depth_batch_start;

                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                svbool_t const batch_predicate_b32 = svwhilelt_b32((uint32_t)0, (uint32_t)batch_size);
                for (nk_size_t row_in_tile = 0; row_in_tile < rows_actual; row_in_tile++) {
                    nk_size_t const row_abs = row_tile_start + row_in_tile;
                    nk_e3m2_t const *a_src = vectors + row_abs * stride_elements + depth_batch_start * expansion;
                    svuint8_t raw_bytes = svld1_u8(full_predicate_b8, (uint8_t const *)a_src);
                    svfloat16_t converted_f16 = nk_e3m2x_to_f16x_ssve_(full_predicate_b16, raw_bytes);
                    svst1_f16(full_predicate_b16, (float16_t *)a_converted_f16x32[row_in_tile], converted_f16);
                    svld1_hor_za32(0, row_in_tile, batch_predicate_b32,
                                   (nk_f32_t const *)a_converted_f16x32[row_in_tile]);
                }

                // Save A columns from ZA0 to stack buffer
                for (nk_size_t s = 0; s < batch_size; s++)
                    svst1_f32(full_predicate_b32, a_buffer[s],
                              svread_ver_za32_f32_m(svdup_f32(0), row_predicate_b32, 0, s));

                svbool_t const depth_predicate_b8 = svwhilelt_b8((uint32_t)0, (uint32_t)(batch_size * expansion));
                svbool_t const depth_predicate_b16 = svwhilelt_b16((uint32_t)0, (uint32_t)(batch_size * expansion));

                // Load B column tile into ZA0, vertical read + FMOPA into ZA1
                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t column = 0; column < tile_dimension; column++) {
                    nk_size_t const column_abs = column_tile_start + column;
                    if (column_abs < n_vectors) {
                        nk_e3m2_t const *src = &vectors[column_abs * stride_elements + depth_batch_start * expansion];
                        svuint8_t raw = svld1_u8(depth_predicate_b8, (uint8_t const *)src);
                        svfloat16_t converted_f16 = nk_e3m2x_to_f16x_ssve_(depth_predicate_b16, raw);
                        svst1_f16(depth_predicate_b16, (float16_t *)b_bounce_f16x32, converted_f16);
                        svld1_hor_za32(0, column, batch_predicate_b32, (nk_f32_t const *)b_bounce_f16x32);
                    }
                }
                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {
                    svfloat16_t a_interleaved = svreinterpret_f16_f32(
                        svld1_f32(full_predicate_b32, a_buffer[depth_step]));
                    svfloat32_t b_column_f32 = svread_ver_za32_f32_m(svdup_f32(0.0f), column_predicate_b32, 0,
                                                                     depth_step);
                    svfloat16_t b_interleaved = svreinterpret_f16_f32(b_column_f32);
                    svmopa_za32_f16_m(1, row_predicate_b16, column_predicate_b16, a_interleaved, b_interleaved);
                }
            }

            for (nk_size_t row = 0; row < rows_actual; row++) {
                nk_size_t const row_abs = row_tile_start + row;
                svst1_hor_za32(1, row, column_predicate_b32,
                               result + row_abs * result_stride_elements + column_tile_start);
            }
        }

        for (nk_size_t row = 0; row < rows_actual; row++) {
            nk_size_t const i = row_tile_start + row;
            for (nk_size_t j = 0; j < i && j < n_vectors; j++)
                result[j * result_stride_elements + i] = result[i * result_stride_elements + j];
        }
    }
}

NK_PUBLIC void nk_dots_symmetric_e3m2_sme(nk_e3m2_t const *vectors, nk_size_t n_vectors, nk_size_t depth,
                                          nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                          nk_size_t row_start, nk_size_t row_count) {

    nk_size_t const stride_elements = stride / sizeof(nk_e3m2_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_dots_symmetric_e3m2_sme_kernel_(vectors, n_vectors, depth, stride_elements, result, result_stride_elements,
                                       row_start, row_count);
}

#pragma endregion

/*
 *  `u8` × `u8` → `u32` GEMM using SME outer products.
 *
 *  Uses `svmopa_za32_u8_m` for unsigned 8-bit integer outer product accumulate.
 *  Available on Apple M4 (SME_I8I32 = 1, covers both signed and unsigned).
 *
 *  Tile dimensions identical to `i8` → `i32` (512-bit SVL):
 *  - Input vectors: 64 `u8` elements (SVL/8 = 64)
 *  - Output tile: 16 × 16 `u32` elements (`ZA32`)
 *  - Each output `u32` is a dot product of 4 `u8` pairs
 */

#pragma region Unsigned Integers

NK_PUBLIC nk_size_t nk_dots_packed_size_u8_sme(nk_size_t n, nk_size_t k) { return nk_dots_packed_size_i8_sme(n, k); }

NK_PUBLIC void nk_dots_pack_u8_sme(             //
    nk_u8_t const *b, nk_size_t n, nk_size_t k, //
    nk_size_t b_stride, void *b_packed) {

    nk_size_t const expansion = 4;
    nk_size_t const tile_dimension = svcntsw();
    nk_size_t const vector_elements = svcntsb();
    nk_size_t const b_stride_elements = b_stride / sizeof(nk_u8_t);

    nk_size_t const column_tile_count = nk_size_divide_round_up_(n, tile_dimension);
    nk_size_t const depth_step_count = nk_size_divide_round_up_(k, expansion);
    nk_size_t const total_vectors = column_tile_count * depth_step_count;

    nk_dots_sme_packed_header_t *header = (nk_dots_sme_packed_header_t *)b_packed;
    header->column_tile_count = (nk_u32_t)column_tile_count;
    header->depth_tile_count = (nk_u32_t)depth_step_count;
    header->columns = (nk_u32_t)n;
    header->depth = (nk_u32_t)k;
    header->svl_bytes = (nk_u32_t)(svcntsw() * sizeof(nk_u32_t));

    nk_u8_t *tiles_ptr = (nk_u8_t *)((char *)b_packed + sizeof(nk_dots_sme_packed_header_t));

    for (nk_size_t i = 0; i < total_vectors * vector_elements; i++) tiles_ptr[i] = 0;

    // Interleaved packing: packed_vec[expansion * d + k] = B[column_tile*tile_dim + d, depth_step*expansion + k]
    for (nk_size_t column_tile = 0; column_tile < column_tile_count; column_tile++) {
        for (nk_size_t depth_step = 0; depth_step < depth_step_count; depth_step++) {
            nk_size_t const vec_index = column_tile * depth_step_count + depth_step;
            nk_u8_t *vec_output = tiles_ptr + vec_index * vector_elements;

            nk_size_t const b_row_start = column_tile * tile_dimension;
            nk_size_t const depth_base = depth_step * expansion;
            nk_size_t const rows_to_pack = (b_row_start + tile_dimension <= n) ? tile_dimension : (n - b_row_start);

            for (nk_size_t column_in_tile = 0; column_in_tile < rows_to_pack; column_in_tile++) {
                for (nk_size_t sub_element = 0; sub_element < expansion; sub_element++) {
                    nk_size_t const depth_idx = depth_base + sub_element;
                    if (depth_idx < k) {
                        nk_size_t const src_idx = (b_row_start + column_in_tile) * b_stride_elements + depth_idx;
                        vec_output[expansion * column_in_tile + sub_element] = b[src_idx];
                    }
                }
            }
        }
    }
}

/**
 *  `u8` × `u8` → `u32` GEMM core kernel using SME outer products.
 *  Same interleaved algorithm as i8 kernel, using UMOPA u8→u32.
 */
__arm_locally_streaming __arm_new("za") static void nk_dots_packed_u8_sme_kernel_( //
    nk_u8_t const *a, void const *b_packed, nk_u32_t *c,                           //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,                            //
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) {

    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_size_t const column_tile_count = header->column_tile_count;
    nk_size_t const depth_step_count = header->depth_tile_count;

    nk_size_t const expansion = 4;
    nk_size_t const tile_dimension = svcntw();
    nk_size_t const vector_elements = svcntb();
    nk_size_t const depth_steps_per_batch = tile_dimension;

    nk_u8_t const *b_packed_base = (nk_u8_t const *)((char const *)b_packed + sizeof(nk_dots_sme_packed_header_t));

    svbool_t const full_predicate_b8 = svptrue_b8();
    svbool_t const full_predicate_b32 = svptrue_b32();

    nk_size_t const row_tile_count = nk_size_divide_round_up_(rows, tile_dimension);

    for (nk_size_t row_tile_index = 0; row_tile_index < row_tile_count; row_tile_index++) {
        nk_size_t const row_start = row_tile_index * tile_dimension;
        nk_size_t const rows_remaining = (row_start + tile_dimension <= rows) ? tile_dimension : (rows - row_start);
        svbool_t const row_predicate_b8 = svwhilelt_b8((uint32_t)0, (uint32_t)(rows_remaining * expansion));
        svbool_t const row_predicate_b32 = svwhilelt_b32((uint32_t)0, (uint32_t)rows_remaining);

        nk_size_t column_tile_index = 0;

        // Fast path: 3 column tiles using ZA1-ZA3 (ZA0 = staging)
        for (; column_tile_index + 3 <= column_tile_count; column_tile_index += 3) {
            svzero_mask_za(nk_sme_zero_za32_tiles_123_);

            for (nk_size_t depth_batch_start = 0; depth_batch_start < depth_step_count;
                 depth_batch_start += depth_steps_per_batch) {
                nk_size_t const depth_batch_end = (depth_batch_start + depth_steps_per_batch < depth_step_count)
                                                      ? depth_batch_start + depth_steps_per_batch
                                                      : depth_step_count;
                nk_size_t const batch_size = depth_batch_end - depth_batch_start;

                svzero_mask_za(nk_sme_zero_za32_tile_0_);

                svbool_t const batch_predicate_b32 = svwhilelt_b32((uint32_t)0, (uint32_t)batch_size);
                for (nk_size_t row_in_tile = 0; row_in_tile < rows_remaining; row_in_tile++) {
                    nk_size_t const a_row = row_start + row_in_tile;
                    nk_f32_t const *a_row_ptr = (nk_f32_t const *)(a + a_row * a_stride_elements +
                                                                   depth_batch_start * expansion);
                    svld1_hor_za32(0, row_in_tile, batch_predicate_b32, a_row_ptr);
                }

                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {

                    svuint32_t a_column_u32 = svread_ver_za32_u32_m(svdup_u32(0), row_predicate_b32, 0, depth_step);
                    svuint8_t a_interleaved_vector_u8 = svreinterpret_u8_u32(a_column_u32);

                    nk_u8_t const *b_packed_ptr_0 = b_packed_base + ((column_tile_index + 0) * depth_step_count +
                                                                     depth_batch_start + depth_step) *
                                                                        vector_elements;
                    nk_u8_t const *b_packed_ptr_1 = b_packed_base + ((column_tile_index + 1) * depth_step_count +
                                                                     depth_batch_start + depth_step) *
                                                                        vector_elements;
                    nk_u8_t const *b_packed_ptr_2 = b_packed_base + ((column_tile_index + 2) * depth_step_count +
                                                                     depth_batch_start + depth_step) *
                                                                        vector_elements;
                    svuint8_t b_packed_vector_0 = svld1_u8(full_predicate_b8, b_packed_ptr_0);
                    svuint8_t b_packed_vector_1 = svld1_u8(full_predicate_b8, b_packed_ptr_1);
                    svuint8_t b_packed_vector_2 = svld1_u8(full_predicate_b8, b_packed_ptr_2);

                    svmopa_za32_u8_m(1, row_predicate_b8, full_predicate_b8, a_interleaved_vector_u8,
                                     b_packed_vector_0);
                    svmopa_za32_u8_m(2, row_predicate_b8, full_predicate_b8, a_interleaved_vector_u8,
                                     b_packed_vector_1);
                    svmopa_za32_u8_m(3, row_predicate_b8, full_predicate_b8, a_interleaved_vector_u8,
                                     b_packed_vector_2);
                }
            }

            for (nk_size_t row = 0; row < rows_remaining; row++) {
                nk_size_t const cs0 = (column_tile_index + 0) * tile_dimension;
                nk_size_t const cs1 = (column_tile_index + 1) * tile_dimension;
                nk_size_t const cs2 = (column_tile_index + 2) * tile_dimension;
                svst1_hor_za32(1, row, full_predicate_b32,
                               (nk_i32_t *)(c + (row_start + row) * c_stride_elements + cs0));
                svst1_hor_za32(2, row, full_predicate_b32,
                               (nk_i32_t *)(c + (row_start + row) * c_stride_elements + cs1));
                svst1_hor_za32(3, row, full_predicate_b32,
                               (nk_i32_t *)(c + (row_start + row) * c_stride_elements + cs2));
            }
        }

        // Remainder: 1 column tile using ZA1
        for (; column_tile_index < column_tile_count; column_tile_index++) {
            nk_size_t const column_start = column_tile_index * tile_dimension;
            nk_size_t const cols_remaining = (column_start + tile_dimension <= columns) ? tile_dimension
                                                                                        : (columns - column_start);
            svbool_t const column_predicate_b32 = svwhilelt_b32((uint32_t)0, (uint32_t)cols_remaining);
            svbool_t const column_predicate_b8 = svwhilelt_b8((uint32_t)0, (uint32_t)(cols_remaining * expansion));

            svzero_mask_za(nk_sme_zero_za32_tile_1_);

            for (nk_size_t depth_batch_start = 0; depth_batch_start < depth_step_count;
                 depth_batch_start += depth_steps_per_batch) {
                nk_size_t const depth_batch_end = (depth_batch_start + depth_steps_per_batch < depth_step_count)
                                                      ? depth_batch_start + depth_steps_per_batch
                                                      : depth_step_count;
                nk_size_t const batch_size = depth_batch_end - depth_batch_start;

                svzero_mask_za(nk_sme_zero_za32_tile_0_);

                svbool_t const batch_predicate_b32 = svwhilelt_b32((uint32_t)0, (uint32_t)batch_size);
                for (nk_size_t row_in_tile = 0; row_in_tile < rows_remaining; row_in_tile++) {
                    nk_size_t const a_row = row_start + row_in_tile;
                    nk_f32_t const *a_row_ptr = (nk_f32_t const *)(a + a_row * a_stride_elements +
                                                                   depth_batch_start * expansion);
                    svld1_hor_za32(0, row_in_tile, batch_predicate_b32, a_row_ptr);
                }

                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {

                    svuint32_t a_column_u32 = svread_ver_za32_u32_m(svdup_u32(0), row_predicate_b32, 0, depth_step);
                    svuint8_t a_interleaved_vector_u8 = svreinterpret_u8_u32(a_column_u32);

                    nk_u8_t const *b_packed_ptr = b_packed_base + (column_tile_index * depth_step_count +
                                                                   depth_batch_start + depth_step) *
                                                                      vector_elements;
                    svuint8_t b_packed_vector = svld1_u8(column_predicate_b8, b_packed_ptr);

                    svmopa_za32_u8_m(1, row_predicate_b8, column_predicate_b8, a_interleaved_vector_u8,
                                     b_packed_vector);
                }
            }

            for (nk_size_t row = 0; row < rows_remaining; row++) {
                svst1_hor_za32(1, row, column_predicate_b32,
                               (nk_i32_t *)(c + (row_start + row) * c_stride_elements + column_start));
            }
        }
    }
}

NK_PUBLIC void nk_dots_packed_u8_sme(                    //
    nk_u8_t const *a, void const *b_packed, nk_u32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,  //
    nk_size_t a_stride, nk_size_t c_stride) {

    nk_size_t const a_stride_elements = a_stride / sizeof(nk_u8_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_u32_t);

    nk_dots_packed_u8_sme_kernel_(a, b_packed, c, rows, columns, depth, a_stride_elements, c_stride_elements);
}

// expansion=4: each u32 word packs 4 u8 values
__arm_locally_streaming __arm_new("za") static void nk_dots_symmetric_u8_sme_kernel_(
    nk_u8_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride_elements, nk_u32_t *result,
    nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) {

    nk_size_t const expansion = 4;
    nk_size_t const tile_dimension = svcntw();
    nk_size_t const depth_step_count = nk_size_divide_round_up_(depth, expansion);
    nk_size_t const depth_steps_per_batch = tile_dimension;

    svbool_t const full_predicate_b8 = svptrue_b8();
    svbool_t const full_predicate_b32 = svptrue_b32();

    NK_ALIGN64 nk_u32_t a_buffer[16][16];

    nk_size_t const row_end = row_start + row_count;
    nk_size_t const column_tile_count = nk_size_divide_round_up_(n_vectors, tile_dimension);

    for (nk_size_t row_tile_start = row_start; row_tile_start < row_end && row_tile_start < n_vectors;
         row_tile_start += tile_dimension) {
        nk_size_t const rows_remaining = (row_tile_start + tile_dimension <= row_end) ? tile_dimension
                                                                                      : (row_end - row_tile_start);
        nk_size_t const rows_clamped = (row_tile_start + rows_remaining <= n_vectors) ? rows_remaining
                                                                                      : (n_vectors - row_tile_start);
        svbool_t const row_predicate_b8 = svwhilelt_b8((uint32_t)0, (uint32_t)(rows_clamped * expansion));
        svbool_t const row_predicate_b32 = svwhilelt_b32((uint32_t)0, (uint32_t)rows_clamped);

        nk_size_t column_tile_index = 0;

        for (; column_tile_index + 3 <= column_tile_count; column_tile_index += 3) {
            svzero_mask_za(nk_sme_zero_za32_tiles_123_);

            for (nk_size_t depth_batch_start = 0; depth_batch_start < depth_step_count;
                 depth_batch_start += depth_steps_per_batch) {
                nk_size_t const depth_batch_end = (depth_batch_start + depth_steps_per_batch < depth_step_count)
                                                      ? depth_batch_start + depth_steps_per_batch
                                                      : depth_step_count;
                nk_size_t const batch_size = depth_batch_end - depth_batch_start;

                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                svbool_t const batch_predicate_b32 = svwhilelt_b32((uint32_t)0, (uint32_t)batch_size);
                for (nk_size_t row_in_tile = 0; row_in_tile < rows_clamped; row_in_tile++) {
                    nk_size_t const row_abs = row_tile_start + row_in_tile;
                    nk_f32_t const *a_row_ptr = (nk_f32_t const *)(vectors + row_abs * stride_elements +
                                                                   depth_batch_start * expansion);
                    svld1_hor_za32(0, row_in_tile, batch_predicate_b32, a_row_ptr);
                }

                for (nk_size_t s = 0; s < batch_size; s++)
                    svst1_u32(full_predicate_b32, a_buffer[s],
                              svread_ver_za32_u32_m(svdup_u32(0), row_predicate_b32, 0, s));

                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t column = 0; column < tile_dimension; column++) {
                    nk_size_t const column_abs = (column_tile_index + 0) * tile_dimension + column;
                    if (column_abs < n_vectors) {
                        nk_f32_t const *b_row = (nk_f32_t const *)(vectors + column_abs * stride_elements +
                                                                   depth_batch_start * expansion);
                        svld1_hor_za32(0, column, batch_predicate_b32, b_row);
                    }
                }
                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {
                    svuint8_t a_interleaved = svreinterpret_u8_u32(svld1_u32(full_predicate_b32, a_buffer[depth_step]));
                    svuint32_t b_column_u32 = svread_ver_za32_u32_m(svdup_u32(0), full_predicate_b32, 0, depth_step);
                    svuint8_t b_interleaved = svreinterpret_u8_u32(b_column_u32);
                    svmopa_za32_u8_m(1, row_predicate_b8, full_predicate_b8, a_interleaved, b_interleaved);
                }

                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t column = 0; column < tile_dimension; column++) {
                    nk_size_t const column_abs = (column_tile_index + 1) * tile_dimension + column;
                    if (column_abs < n_vectors) {
                        nk_f32_t const *b_row = (nk_f32_t const *)(vectors + column_abs * stride_elements +
                                                                   depth_batch_start * expansion);
                        svld1_hor_za32(0, column, batch_predicate_b32, b_row);
                    }
                }
                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {
                    svuint8_t a_interleaved = svreinterpret_u8_u32(svld1_u32(full_predicate_b32, a_buffer[depth_step]));
                    svuint32_t b_column_u32 = svread_ver_za32_u32_m(svdup_u32(0), full_predicate_b32, 0, depth_step);
                    svuint8_t b_interleaved = svreinterpret_u8_u32(b_column_u32);
                    svmopa_za32_u8_m(2, row_predicate_b8, full_predicate_b8, a_interleaved, b_interleaved);
                }

                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t column = 0; column < tile_dimension; column++) {
                    nk_size_t const column_abs = (column_tile_index + 2) * tile_dimension + column;
                    if (column_abs < n_vectors) {
                        nk_f32_t const *b_row = (nk_f32_t const *)(vectors + column_abs * stride_elements +
                                                                   depth_batch_start * expansion);
                        svld1_hor_za32(0, column, batch_predicate_b32, b_row);
                    }
                }
                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {
                    svuint8_t a_interleaved = svreinterpret_u8_u32(svld1_u32(full_predicate_b32, a_buffer[depth_step]));
                    svuint32_t b_column_u32 = svread_ver_za32_u32_m(svdup_u32(0), full_predicate_b32, 0, depth_step);
                    svuint8_t b_interleaved = svreinterpret_u8_u32(b_column_u32);
                    svmopa_za32_u8_m(3, row_predicate_b8, full_predicate_b8, a_interleaved, b_interleaved);
                }
            }

            for (nk_size_t row = 0; row < rows_clamped; row++) {
                nk_size_t const row_abs = row_tile_start + row;
                nk_f32_t *result_row = (nk_f32_t *)(result + row_abs * result_stride_elements);
                svst1_hor_za32(1, row, full_predicate_b32, result_row + (column_tile_index + 0) * tile_dimension);
                svst1_hor_za32(2, row, full_predicate_b32, result_row + (column_tile_index + 1) * tile_dimension);
                svst1_hor_za32(3, row, full_predicate_b32, result_row + (column_tile_index + 2) * tile_dimension);
            }
        }

        for (; column_tile_index < column_tile_count; column_tile_index++) {
            nk_size_t const column_tile_start = column_tile_index * tile_dimension;
            nk_size_t const cols_remaining = (column_tile_start + tile_dimension <= n_vectors)
                                                 ? tile_dimension
                                                 : (n_vectors - column_tile_start);
            svbool_t const column_predicate_b32 = svwhilelt_b32((uint32_t)0, (uint32_t)cols_remaining);
            svbool_t const column_predicate_b8 = svwhilelt_b8((uint32_t)0, (uint32_t)(cols_remaining * expansion));

            svzero_mask_za(nk_sme_zero_za32_tile_1_);

            for (nk_size_t depth_batch_start = 0; depth_batch_start < depth_step_count;
                 depth_batch_start += depth_steps_per_batch) {
                nk_size_t const depth_batch_end = (depth_batch_start + depth_steps_per_batch < depth_step_count)
                                                      ? depth_batch_start + depth_steps_per_batch
                                                      : depth_step_count;
                nk_size_t const batch_size = depth_batch_end - depth_batch_start;

                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                svbool_t const batch_predicate_b32 = svwhilelt_b32((uint32_t)0, (uint32_t)batch_size);
                for (nk_size_t row_in_tile = 0; row_in_tile < rows_clamped; row_in_tile++) {
                    nk_size_t const row_abs = row_tile_start + row_in_tile;
                    nk_f32_t const *a_row_ptr = (nk_f32_t const *)(vectors + row_abs * stride_elements +
                                                                   depth_batch_start * expansion);
                    svld1_hor_za32(0, row_in_tile, batch_predicate_b32, a_row_ptr);
                }

                for (nk_size_t s = 0; s < batch_size; s++)
                    svst1_u32(full_predicate_b32, a_buffer[s],
                              svread_ver_za32_u32_m(svdup_u32(0), row_predicate_b32, 0, s));

                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t column = 0; column < tile_dimension; column++) {
                    nk_size_t const column_abs = column_tile_start + column;
                    if (column_abs < n_vectors) {
                        nk_f32_t const *b_row = (nk_f32_t const *)(vectors + column_abs * stride_elements +
                                                                   depth_batch_start * expansion);
                        svld1_hor_za32(0, column, batch_predicate_b32, b_row);
                    }
                }
                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {
                    svuint8_t a_interleaved = svreinterpret_u8_u32(svld1_u32(full_predicate_b32, a_buffer[depth_step]));
                    svuint32_t b_column_u32 = svread_ver_za32_u32_m(svdup_u32(0), column_predicate_b32, 0, depth_step);
                    svuint8_t b_interleaved = svreinterpret_u8_u32(b_column_u32);
                    svmopa_za32_u8_m(1, row_predicate_b8, column_predicate_b8, a_interleaved, b_interleaved);
                }
            }

            for (nk_size_t row = 0; row < rows_clamped; row++) {
                nk_size_t const row_abs = row_tile_start + row;
                svst1_hor_za32(1, row, column_predicate_b32,
                               (nk_f32_t *)(result + row_abs * result_stride_elements) + column_tile_start);
            }
        }

        for (nk_size_t row = 0; row < rows_clamped; row++) {
            nk_size_t const i = row_tile_start + row;
            for (nk_size_t j = 0; j < i && j < n_vectors; j++) {
                result[j * result_stride_elements + i] = result[i * result_stride_elements + j];
            }
        }
    }
}

NK_PUBLIC void nk_dots_symmetric_u8_sme(nk_u8_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride,
                                        nk_u32_t *result, nk_size_t result_stride, nk_size_t row_start,
                                        nk_size_t row_count) {

    nk_size_t const stride_elements = stride / sizeof(nk_u8_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_u32_t);
    nk_dots_symmetric_u8_sme_kernel_(vectors, n_vectors, depth, stride_elements, result, result_stride_elements,
                                     row_start, row_count);
}

#pragma endregion

/*
 *  4-bit integer GEMM (u4, i4) using direct mask-and-double-MOPA.
 *
 *  Each byte packs two 4-bit values (nk_u4x2_t / nk_i4x2_t): byte = (hi << 4) | lo.
 *  Pack functions copy raw packed bytes into the tiled layout (no nibble→byte expansion).
 *  Kernels load packed bytes directly into ZA0, then split into low/high nibbles in
 *  registers and issue 2 MOPAs per depth step:
 *
 *    u4: a_low = a & 0x0F,  a_high = a >> 4   → UMOPA(tile, a_low, b_low) + UMOPA(tile, a_high, b_high)
 *    i4: a_low = (a<<4)>>4, a_high = a >> 4    → SMOPA(tile, a_low, b_low) + SMOPA(tile, a_high, b_high)
 *
 *  This eliminates the bounce buffer entirely, halves memory bandwidth, and matches
 *  i8/u8 throughput since the dominant cost is 2 MOPAs per step × ceil(depth/8) steps
 *  = ceil(depth/4) total MOPAs — same as the unpacked path.
 */

#pragma region Nibble Integers

NK_PUBLIC nk_size_t nk_dots_packed_size_u4_sme(nk_size_t n, nk_size_t k) {
    nk_size_t const tile_dimension = svcntsw();
    nk_size_t const vector_elements = svcntsb();
    nk_size_t const packed_depth = nk_size_divide_round_up_(k, 2);
    nk_size_t const depth_step_count = nk_size_divide_round_up_(packed_depth, 4);
    nk_size_t const column_tile_count = nk_size_divide_round_up_(n, tile_dimension);
    nk_size_t const vector_pair_stride = 2 * vector_elements;
    nk_size_t size = sizeof(nk_dots_sme_packed_header_t);
    size += column_tile_count * depth_step_count * vector_pair_stride * sizeof(nk_u8_t);
    return size;
}

NK_PUBLIC void nk_dots_pack_u4_sme(               //
    nk_u4x2_t const *b, nk_size_t n, nk_size_t k, //
    nk_size_t b_stride, void *b_packed) {

    nk_size_t const tile_dimension = svcntsw();
    nk_size_t const vector_elements = svcntsb();
    nk_size_t const vector_pair_stride = 2 * vector_elements;
    nk_size_t const b_stride_bytes = b_stride / sizeof(nk_u4x2_t);
    nk_size_t const packed_depth = nk_size_divide_round_up_(k, 2);
    nk_size_t const depth_step_count = nk_size_divide_round_up_(packed_depth, 4);
    nk_size_t const column_tile_count = nk_size_divide_round_up_(n, tile_dimension);
    nk_size_t const total_vectors = column_tile_count * depth_step_count;

    nk_dots_sme_packed_header_t *header = (nk_dots_sme_packed_header_t *)b_packed;
    header->column_tile_count = (nk_u32_t)column_tile_count;
    header->depth_tile_count = (nk_u32_t)depth_step_count;
    header->columns = (nk_u32_t)n;
    header->depth = (nk_u32_t)k;
    header->svl_bytes = (nk_u32_t)(svcntsw() * sizeof(nk_u32_t));

    nk_u8_t *tiles_ptr = (nk_u8_t *)((char *)b_packed + sizeof(nk_dots_sme_packed_header_t));

    for (nk_size_t i = 0; i < total_vectors * vector_pair_stride; i++) tiles_ptr[i] = 0;

    for (nk_size_t column_tile = 0; column_tile < column_tile_count; column_tile++) {
        for (nk_size_t depth_step = 0; depth_step < depth_step_count; depth_step++) {
            nk_size_t const vec_index = column_tile * depth_step_count + depth_step;
            nk_u8_t *vec_low = tiles_ptr + vec_index * vector_pair_stride;
            nk_u8_t *vec_high = vec_low + vector_elements;

            nk_size_t const b_row_start = column_tile * tile_dimension;
            nk_size_t const depth_byte_base = depth_step * 4;
            nk_size_t const rows_to_pack = (b_row_start + tile_dimension <= n) ? tile_dimension : (n - b_row_start);

            for (nk_size_t column_in_tile = 0; column_in_tile < rows_to_pack; column_in_tile++) {
                for (nk_size_t sub = 0; sub < 4; sub++) {
                    nk_size_t const byte_idx = depth_byte_base + sub;
                    if (byte_idx < packed_depth) {
                        nk_size_t const src_idx = (b_row_start + column_in_tile) * b_stride_bytes + byte_idx;
                        nk_u8_t val = ((nk_u8_t const *)b)[src_idx];
                        nk_u8_t low_nibble = val & 0x0F;
                        nk_u8_t high_nibble = val >> 4;
                        if (byte_idx == packed_depth - 1 && (k & 1)) high_nibble = 0;
                        nk_size_t const slot = 4 * column_in_tile + sub;
                        vec_low[slot] = low_nibble;
                        vec_high[slot] = high_nibble;
                    }
                }
            }
        }
    }
}

/**
 *  `u4` × `u4` → `u32` packed GEMM kernel using SME UMOPA with direct mask-and-double-MOPA.
 *  A input is nibble-packed — loaded directly into ZA0, split into low/high nibbles in registers.
 *  B input is pre-split low/high nibble vectors from nk_dots_pack_u4_sme.
 *  Two UMOPAs per depth step: one for low nibbles, one for high nibbles.
 */
__arm_locally_streaming __arm_new("za") static void nk_dots_packed_u4_sme_kernel_( //
    nk_u4x2_t const *a, void const *b_packed, nk_u32_t *c,                         //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,                            //
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) {

    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_size_t const column_tile_count = header->column_tile_count;
    nk_size_t const depth_step_count = header->depth_tile_count;

    nk_size_t const expansion = 4;
    nk_size_t const tile_dimension = svcntw();
    nk_size_t const vector_elements = svcntb();
    nk_size_t const vector_pair_stride = 2 * vector_elements;
    nk_size_t const depth_steps_per_batch = tile_dimension;
    nk_size_t const packed_depth = nk_size_divide_round_up_(depth, 2);

    nk_u8_t const *b_packed_base = (nk_u8_t const *)((char const *)b_packed + sizeof(nk_dots_sme_packed_header_t));

    svbool_t const full_predicate_b8 = svptrue_b8();
    svbool_t const full_predicate_b32 = svptrue_b32();

    nk_size_t const row_tile_count = nk_size_divide_round_up_(rows, tile_dimension);

    for (nk_size_t row_tile_index = 0; row_tile_index < row_tile_count; row_tile_index++) {
        nk_size_t const row_start = row_tile_index * tile_dimension;
        nk_size_t const rows_remaining = (row_start + tile_dimension <= rows) ? tile_dimension : (rows - row_start);
        svbool_t const row_predicate_b8 = svwhilelt_b8((uint32_t)0, (uint32_t)(rows_remaining * expansion));
        svbool_t const row_predicate_b32 = svwhilelt_b32((uint32_t)0, (uint32_t)rows_remaining);

        nk_size_t column_tile_index = 0;

        for (; column_tile_index + 3 <= column_tile_count; column_tile_index += 3) {
            svzero_mask_za(nk_sme_zero_za32_tiles_123_);

            for (nk_size_t depth_batch_start = 0; depth_batch_start < depth_step_count;
                 depth_batch_start += depth_steps_per_batch) {
                nk_size_t const depth_batch_end = (depth_batch_start + depth_steps_per_batch < depth_step_count)
                                                      ? depth_batch_start + depth_steps_per_batch
                                                      : depth_step_count;
                nk_size_t const batch_size = depth_batch_end - depth_batch_start;

                svzero_mask_za(nk_sme_zero_za32_tile_0_);

                svbool_t const batch_predicate_b32 = svwhilelt_b32((uint32_t)0, (uint32_t)batch_size);
                nk_size_t const byte_offset = depth_batch_start * 4;

                for (nk_size_t row_in_tile = 0; row_in_tile < rows_remaining; row_in_tile++) {
                    nk_size_t const a_row = row_start + row_in_tile;
                    nk_u8_t const *src = (nk_u8_t const *)(a + a_row * a_stride_elements) + byte_offset;
                    svld1_hor_za32(0, row_in_tile, batch_predicate_b32, (nk_f32_t const *)src);
                }

                // Last-step ZA0 fixup: mask invalid bytes and odd-depth high nibble
                if (depth_batch_start + batch_size >= depth_step_count) {
                    nk_size_t const last_step = batch_size - 1;
                    nk_size_t const global_step = depth_batch_start + last_step;
                    nk_size_t const valid_bytes = packed_depth - global_step * 4;
                    if (valid_bytes < 4 || (depth & 1)) {
                        nk_u32_t mask = (valid_bytes >= 4) ? 0xFFFFFFFFu : ((1u << (valid_bytes * 8)) - 1u);
                        if (depth & 1) {
                            nk_u32_t nibble_clear = ~((nk_u32_t)0xF0 << ((valid_bytes - 1) * 8));
                            mask &= nibble_clear;
                        }
                        for (nk_size_t row_in_tile = 0; row_in_tile < rows_remaining; row_in_tile++) {
                            svuint32_t column = svread_ver_za32_u32_m(svdup_u32(0), row_predicate_b32, 0, last_step);
                            column = svand_n_u32_x(row_predicate_b32, column, mask);
                            svwrite_ver_za32_u32_m(0, last_step, row_predicate_b32, column);
                        }
                    }
                }

                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {

                    svuint32_t a_column_u32 = svread_ver_za32_u32_m(svdup_u32(0), row_predicate_b32, 0, depth_step);
                    svuint8_t a_packed_nibbles_u8 = svreinterpret_u8_u32(a_column_u32);
                    svuint8_t a_low_nibbles = svand_n_u8_x(full_predicate_b8, a_packed_nibbles_u8, 0x0F);
                    svuint8_t a_high_nibbles = svlsr_n_u8_x(full_predicate_b8, a_packed_nibbles_u8, 4);

                    nk_u8_t const *b_column_pair_0 = b_packed_base + ((column_tile_index + 0) * depth_step_count +
                                                                      depth_batch_start + depth_step) *
                                                                         vector_pair_stride;
                    nk_u8_t const *b_column_pair_1 = b_packed_base + ((column_tile_index + 1) * depth_step_count +
                                                                      depth_batch_start + depth_step) *
                                                                         vector_pair_stride;
                    nk_u8_t const *b_column_pair_2 = b_packed_base + ((column_tile_index + 2) * depth_step_count +
                                                                      depth_batch_start + depth_step) *
                                                                         vector_pair_stride;

                    svuint8_t b_low_nibbles_0 = svld1_u8(full_predicate_b8, b_column_pair_0);
                    svuint8_t b_high_nibbles_0 = svld1_u8(full_predicate_b8, b_column_pair_0 + vector_elements);
                    svuint8_t b_low_nibbles_1 = svld1_u8(full_predicate_b8, b_column_pair_1);
                    svuint8_t b_high_nibbles_1 = svld1_u8(full_predicate_b8, b_column_pair_1 + vector_elements);
                    svuint8_t b_low_nibbles_2 = svld1_u8(full_predicate_b8, b_column_pair_2);
                    svuint8_t b_high_nibbles_2 = svld1_u8(full_predicate_b8, b_column_pair_2 + vector_elements);

                    svmopa_za32_u8_m(1, row_predicate_b8, full_predicate_b8, a_low_nibbles, b_low_nibbles_0);
                    svmopa_za32_u8_m(1, row_predicate_b8, full_predicate_b8, a_high_nibbles, b_high_nibbles_0);
                    svmopa_za32_u8_m(2, row_predicate_b8, full_predicate_b8, a_low_nibbles, b_low_nibbles_1);
                    svmopa_za32_u8_m(2, row_predicate_b8, full_predicate_b8, a_high_nibbles, b_high_nibbles_1);
                    svmopa_za32_u8_m(3, row_predicate_b8, full_predicate_b8, a_low_nibbles, b_low_nibbles_2);
                    svmopa_za32_u8_m(3, row_predicate_b8, full_predicate_b8, a_high_nibbles, b_high_nibbles_2);
                }
            }

            for (nk_size_t row = 0; row < rows_remaining; row++) {
                nk_u32_t *c_row = c + (row_start + row) * c_stride_elements;
                nk_size_t const cs0 = (column_tile_index + 0) * tile_dimension;
                nk_size_t const cs1 = (column_tile_index + 1) * tile_dimension;
                nk_size_t const cs2 = (column_tile_index + 2) * tile_dimension;
                svst1_hor_za32(1, row, full_predicate_b32, (nk_f32_t *)(c_row + cs0));
                svst1_hor_za32(2, row, full_predicate_b32, (nk_f32_t *)(c_row + cs1));
                svst1_hor_za32(3, row, full_predicate_b32, (nk_f32_t *)(c_row + cs2));
            }
        }

        for (; column_tile_index < column_tile_count; column_tile_index++) {
            nk_size_t const column_start = column_tile_index * tile_dimension;
            nk_size_t const cols_remaining = (column_start + tile_dimension <= columns) ? tile_dimension
                                                                                        : (columns - column_start);
            svbool_t const column_predicate_b32 = svwhilelt_b32((uint32_t)0, (uint32_t)cols_remaining);
            svbool_t const column_predicate_b8 = svwhilelt_b8((uint32_t)0, (uint32_t)(cols_remaining * expansion));

            svzero_mask_za(nk_sme_zero_za32_tile_1_);

            for (nk_size_t depth_batch_start = 0; depth_batch_start < depth_step_count;
                 depth_batch_start += depth_steps_per_batch) {
                nk_size_t const depth_batch_end = (depth_batch_start + depth_steps_per_batch < depth_step_count)
                                                      ? depth_batch_start + depth_steps_per_batch
                                                      : depth_step_count;
                nk_size_t const batch_size = depth_batch_end - depth_batch_start;

                svzero_mask_za(nk_sme_zero_za32_tile_0_);

                svbool_t const batch_predicate_b32 = svwhilelt_b32((uint32_t)0, (uint32_t)batch_size);
                nk_size_t const byte_offset = depth_batch_start * 4;

                for (nk_size_t row_in_tile = 0; row_in_tile < rows_remaining; row_in_tile++) {
                    nk_size_t const a_row = row_start + row_in_tile;
                    nk_u8_t const *src = (nk_u8_t const *)(a + a_row * a_stride_elements) + byte_offset;
                    svld1_hor_za32(0, row_in_tile, batch_predicate_b32, (nk_f32_t const *)src);
                }

                // Last-step ZA0 fixup
                if (depth_batch_start + batch_size >= depth_step_count) {
                    nk_size_t const last_step = batch_size - 1;
                    nk_size_t const global_step = depth_batch_start + last_step;
                    nk_size_t const valid_bytes = packed_depth - global_step * 4;
                    if (valid_bytes < 4 || (depth & 1)) {
                        nk_u32_t mask = (valid_bytes >= 4) ? 0xFFFFFFFFu : ((1u << (valid_bytes * 8)) - 1u);
                        if (depth & 1) {
                            nk_u32_t nibble_clear = ~((nk_u32_t)0xF0 << ((valid_bytes - 1) * 8));
                            mask &= nibble_clear;
                        }
                        for (nk_size_t row_in_tile = 0; row_in_tile < rows_remaining; row_in_tile++) {
                            svuint32_t column = svread_ver_za32_u32_m(svdup_u32(0), row_predicate_b32, 0, last_step);
                            column = svand_n_u32_x(row_predicate_b32, column, mask);
                            svwrite_ver_za32_u32_m(0, last_step, row_predicate_b32, column);
                        }
                    }
                }

                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {

                    svuint32_t a_column_u32 = svread_ver_za32_u32_m(svdup_u32(0), row_predicate_b32, 0, depth_step);
                    svuint8_t a_packed_nibbles_u8 = svreinterpret_u8_u32(a_column_u32);
                    svuint8_t a_low_nibbles = svand_n_u8_x(full_predicate_b8, a_packed_nibbles_u8, 0x0F);
                    svuint8_t a_high_nibbles = svlsr_n_u8_x(full_predicate_b8, a_packed_nibbles_u8, 4);

                    nk_u8_t const *b_column_pair = b_packed_base + (column_tile_index * depth_step_count +
                                                                    depth_batch_start + depth_step) *
                                                                       vector_pair_stride;
                    svuint8_t b_low_nibbles = svld1_u8(column_predicate_b8, b_column_pair);
                    svuint8_t b_high_nibbles = svld1_u8(column_predicate_b8, b_column_pair + vector_elements);

                    svmopa_za32_u8_m(1, row_predicate_b8, column_predicate_b8, a_low_nibbles, b_low_nibbles);
                    svmopa_za32_u8_m(1, row_predicate_b8, column_predicate_b8, a_high_nibbles, b_high_nibbles);
                }
            }

            for (nk_size_t row = 0; row < rows_remaining; row++) {
                svst1_hor_za32(1, row, column_predicate_b32,
                               (nk_f32_t *)(c + (row_start + row) * c_stride_elements + column_start));
            }
        }
    }
}

NK_PUBLIC void nk_dots_packed_u4_sme(                      //
    nk_u4x2_t const *a, void const *b_packed, nk_u32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,    //
    nk_size_t a_stride, nk_size_t c_stride) {

    nk_size_t const a_stride_elements = a_stride / sizeof(nk_u4x2_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_u32_t);
    nk_dots_packed_u4_sme_kernel_(a, b_packed, c, rows, columns, depth, a_stride_elements, c_stride_elements);
}

NK_PUBLIC nk_size_t nk_dots_packed_size_i4_sme(nk_size_t n, nk_size_t k) {
    nk_size_t const tile_dimension = svcntsw();
    nk_size_t const vector_elements = svcntsb();
    nk_size_t const packed_depth = nk_size_divide_round_up_(k, 2);
    nk_size_t const depth_step_count = nk_size_divide_round_up_(packed_depth, 4);
    nk_size_t const column_tile_count = nk_size_divide_round_up_(n, tile_dimension);
    nk_size_t const vector_pair_stride = 2 * vector_elements;
    nk_size_t size = sizeof(nk_dots_sme_packed_header_t);
    size += column_tile_count * depth_step_count * vector_pair_stride * sizeof(nk_u8_t);
    return size;
}

NK_PUBLIC void nk_dots_pack_i4_sme(               //
    nk_i4x2_t const *b, nk_size_t n, nk_size_t k, //
    nk_size_t b_stride, void *b_packed) {

    nk_size_t const tile_dimension = svcntsw();
    nk_size_t const vector_elements = svcntsb();
    nk_size_t const vector_pair_stride = 2 * vector_elements;
    nk_size_t const b_stride_bytes = b_stride / sizeof(nk_i4x2_t);
    nk_size_t const packed_depth = nk_size_divide_round_up_(k, 2);
    nk_size_t const depth_step_count = nk_size_divide_round_up_(packed_depth, 4);
    nk_size_t const column_tile_count = nk_size_divide_round_up_(n, tile_dimension);
    nk_size_t const total_vectors = column_tile_count * depth_step_count;

    nk_dots_sme_packed_header_t *header = (nk_dots_sme_packed_header_t *)b_packed;
    header->column_tile_count = (nk_u32_t)column_tile_count;
    header->depth_tile_count = (nk_u32_t)depth_step_count;
    header->columns = (nk_u32_t)n;
    header->depth = (nk_u32_t)k;
    header->svl_bytes = (nk_u32_t)(svcntsw() * sizeof(nk_i32_t));

    nk_u8_t *tiles_ptr = (nk_u8_t *)((char *)b_packed + sizeof(nk_dots_sme_packed_header_t));

    for (nk_size_t i = 0; i < total_vectors * vector_pair_stride; i++) tiles_ptr[i] = 0;

    for (nk_size_t column_tile = 0; column_tile < column_tile_count; column_tile++) {
        for (nk_size_t depth_step = 0; depth_step < depth_step_count; depth_step++) {
            nk_size_t const vec_index = column_tile * depth_step_count + depth_step;
            nk_u8_t *vec_low = tiles_ptr + vec_index * vector_pair_stride;
            nk_u8_t *vec_high = vec_low + vector_elements;

            nk_size_t const b_row_start = column_tile * tile_dimension;
            nk_size_t const depth_byte_base = depth_step * 4;
            nk_size_t const rows_to_pack = (b_row_start + tile_dimension <= n) ? tile_dimension : (n - b_row_start);

            for (nk_size_t column_in_tile = 0; column_in_tile < rows_to_pack; column_in_tile++) {
                for (nk_size_t sub = 0; sub < 4; sub++) {
                    nk_size_t const byte_idx = depth_byte_base + sub;
                    if (byte_idx < packed_depth) {
                        nk_size_t const src_idx = (b_row_start + column_in_tile) * b_stride_bytes + byte_idx;
                        nk_u8_t val = ((nk_u8_t const *)b)[src_idx];
                        nk_u8_t low_nibble = val & 0x0F;
                        nk_u8_t high_nibble = (val >> 4) & 0x0F;
                        if (byte_idx == packed_depth - 1 && (k & 1)) high_nibble = 0;
                        // Sign-extend 4-bit to 8-bit: (nibble ^ 8) - 8
                        nk_i8_t low_extended = (nk_i8_t)((low_nibble ^ 8) - 8);
                        nk_i8_t high_extended = (nk_i8_t)((high_nibble ^ 8) - 8);
                        nk_size_t const slot = 4 * column_in_tile + sub;
                        vec_low[slot] = (nk_u8_t)low_extended;
                        vec_high[slot] = (nk_u8_t)high_extended;
                    }
                }
            }
        }
    }
}

/**
 *  `i4` × `i4` → `i32` packed GEMM kernel using SME SMOPA with direct mask-and-double-MOPA.
 *  A input is nibble-packed — loaded directly into ZA0, sign-extended via LSL+ASR in registers.
 *  B input is pre-split sign-extended nibble vectors from nk_dots_pack_i4_sme.
 *  Two SMOPAs per depth step: one for low nibbles, one for high nibbles.
 */
__arm_locally_streaming __arm_new("za") static void nk_dots_packed_i4_sme_kernel_( //
    nk_i4x2_t const *a, void const *b_packed, nk_i32_t *c,                         //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,                            //
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) {

    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_size_t const column_tile_count = header->column_tile_count;
    nk_size_t const depth_step_count = header->depth_tile_count;

    nk_size_t const expansion = 4;
    nk_size_t const tile_dimension = svcntw();
    nk_size_t const vector_elements = svcntb();
    nk_size_t const vector_pair_stride = 2 * vector_elements;
    nk_size_t const depth_steps_per_batch = tile_dimension;
    nk_size_t const packed_depth = nk_size_divide_round_up_(depth, 2);

    nk_u8_t const *b_packed_base = (nk_u8_t const *)((char const *)b_packed + sizeof(nk_dots_sme_packed_header_t));

    svbool_t const full_predicate_b8 = svptrue_b8();
    svbool_t const full_predicate_b32 = svptrue_b32();

    nk_size_t const row_tile_count = nk_size_divide_round_up_(rows, tile_dimension);

    for (nk_size_t row_tile_index = 0; row_tile_index < row_tile_count; row_tile_index++) {
        nk_size_t const row_start = row_tile_index * tile_dimension;
        nk_size_t const rows_remaining = (row_start + tile_dimension <= rows) ? tile_dimension : (rows - row_start);
        svbool_t const row_predicate_b8 = svwhilelt_b8((uint32_t)0, (uint32_t)(rows_remaining * expansion));
        svbool_t const row_predicate_b32 = svwhilelt_b32((uint32_t)0, (uint32_t)rows_remaining);

        nk_size_t column_tile_index = 0;

        for (; column_tile_index + 3 <= column_tile_count; column_tile_index += 3) {
            svzero_mask_za(nk_sme_zero_za32_tiles_123_);

            for (nk_size_t depth_batch_start = 0; depth_batch_start < depth_step_count;
                 depth_batch_start += depth_steps_per_batch) {
                nk_size_t const depth_batch_end = (depth_batch_start + depth_steps_per_batch < depth_step_count)
                                                      ? depth_batch_start + depth_steps_per_batch
                                                      : depth_step_count;
                nk_size_t const batch_size = depth_batch_end - depth_batch_start;

                svzero_mask_za(nk_sme_zero_za32_tile_0_);

                svbool_t const batch_predicate_b32 = svwhilelt_b32((uint32_t)0, (uint32_t)batch_size);
                nk_size_t const byte_offset = depth_batch_start * 4;

                for (nk_size_t row_in_tile = 0; row_in_tile < rows_remaining; row_in_tile++) {
                    nk_size_t const a_row = row_start + row_in_tile;
                    nk_u8_t const *src = (nk_u8_t const *)(a + a_row * a_stride_elements) + byte_offset;
                    svld1_hor_za32(0, row_in_tile, batch_predicate_b32, (nk_f32_t const *)src);
                }

                // Last-step ZA0 fixup: mask invalid bytes and odd-depth high nibble
                if (depth_batch_start + batch_size >= depth_step_count) {
                    nk_size_t const last_step = batch_size - 1;
                    nk_size_t const global_step = depth_batch_start + last_step;
                    nk_size_t const valid_bytes = packed_depth - global_step * 4;
                    if (valid_bytes < 4 || (depth & 1)) {
                        nk_u32_t mask = (valid_bytes >= 4) ? 0xFFFFFFFFu : ((1u << (valid_bytes * 8)) - 1u);
                        if (depth & 1) {
                            nk_u32_t nibble_clear = ~((nk_u32_t)0xF0 << ((valid_bytes - 1) * 8));
                            mask &= nibble_clear;
                        }
                        for (nk_size_t row_in_tile = 0; row_in_tile < rows_remaining; row_in_tile++) {
                            svuint32_t column = svread_ver_za32_u32_m(svdup_u32(0), row_predicate_b32, 0, last_step);
                            column = svand_n_u32_x(row_predicate_b32, column, mask);
                            svwrite_ver_za32_u32_m(0, last_step, row_predicate_b32, column);
                        }
                    }
                }

                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {

                    svint32_t a_column_i32 = svread_ver_za32_s32_m(svdup_s32(0), row_predicate_b32, 0, depth_step);
                    svint8_t a_packed_nibbles_s8 = svreinterpret_s8_s32(a_column_i32);
                    svint8_t a_low_nibbles = svasr_n_s8_x(full_predicate_b8,
                                                          svlsl_n_s8_x(full_predicate_b8, a_packed_nibbles_s8, 4), 4);
                    svint8_t a_high_nibbles = svasr_n_s8_x(full_predicate_b8, a_packed_nibbles_s8, 4);

                    nk_i8_t const *b_column_pair_0 = (nk_i8_t const *)(b_packed_base +
                                                                       ((column_tile_index + 0) * depth_step_count +
                                                                        depth_batch_start + depth_step) *
                                                                           vector_pair_stride);
                    nk_i8_t const *b_column_pair_1 = (nk_i8_t const *)(b_packed_base +
                                                                       ((column_tile_index + 1) * depth_step_count +
                                                                        depth_batch_start + depth_step) *
                                                                           vector_pair_stride);
                    nk_i8_t const *b_column_pair_2 = (nk_i8_t const *)(b_packed_base +
                                                                       ((column_tile_index + 2) * depth_step_count +
                                                                        depth_batch_start + depth_step) *
                                                                           vector_pair_stride);

                    svint8_t b_low_nibbles_0 = svld1_s8(full_predicate_b8, b_column_pair_0);
                    svint8_t b_high_nibbles_0 = svld1_s8(full_predicate_b8, b_column_pair_0 + vector_elements);
                    svint8_t b_low_nibbles_1 = svld1_s8(full_predicate_b8, b_column_pair_1);
                    svint8_t b_high_nibbles_1 = svld1_s8(full_predicate_b8, b_column_pair_1 + vector_elements);
                    svint8_t b_low_nibbles_2 = svld1_s8(full_predicate_b8, b_column_pair_2);
                    svint8_t b_high_nibbles_2 = svld1_s8(full_predicate_b8, b_column_pair_2 + vector_elements);

                    svmopa_za32_s8_m(1, row_predicate_b8, full_predicate_b8, a_low_nibbles, b_low_nibbles_0);
                    svmopa_za32_s8_m(1, row_predicate_b8, full_predicate_b8, a_high_nibbles, b_high_nibbles_0);
                    svmopa_za32_s8_m(2, row_predicate_b8, full_predicate_b8, a_low_nibbles, b_low_nibbles_1);
                    svmopa_za32_s8_m(2, row_predicate_b8, full_predicate_b8, a_high_nibbles, b_high_nibbles_1);
                    svmopa_za32_s8_m(3, row_predicate_b8, full_predicate_b8, a_low_nibbles, b_low_nibbles_2);
                    svmopa_za32_s8_m(3, row_predicate_b8, full_predicate_b8, a_high_nibbles, b_high_nibbles_2);
                }
            }

            for (nk_size_t row = 0; row < rows_remaining; row++) {
                nk_i32_t *c_row = c + (row_start + row) * c_stride_elements;
                nk_size_t const cs0 = (column_tile_index + 0) * tile_dimension;
                nk_size_t const cs1 = (column_tile_index + 1) * tile_dimension;
                nk_size_t const cs2 = (column_tile_index + 2) * tile_dimension;
                svst1_hor_za32(1, row, full_predicate_b32, (nk_f32_t *)(c_row + cs0));
                svst1_hor_za32(2, row, full_predicate_b32, (nk_f32_t *)(c_row + cs1));
                svst1_hor_za32(3, row, full_predicate_b32, (nk_f32_t *)(c_row + cs2));
            }
        }

        for (; column_tile_index < column_tile_count; column_tile_index++) {
            nk_size_t const column_start = column_tile_index * tile_dimension;
            nk_size_t const cols_remaining = (column_start + tile_dimension <= columns) ? tile_dimension
                                                                                        : (columns - column_start);
            svbool_t const column_predicate_b32 = svwhilelt_b32((uint32_t)0, (uint32_t)cols_remaining);
            svbool_t const column_predicate_b8 = svwhilelt_b8((uint32_t)0, (uint32_t)(cols_remaining * expansion));

            svzero_mask_za(nk_sme_zero_za32_tile_1_);

            for (nk_size_t depth_batch_start = 0; depth_batch_start < depth_step_count;
                 depth_batch_start += depth_steps_per_batch) {
                nk_size_t const depth_batch_end = (depth_batch_start + depth_steps_per_batch < depth_step_count)
                                                      ? depth_batch_start + depth_steps_per_batch
                                                      : depth_step_count;
                nk_size_t const batch_size = depth_batch_end - depth_batch_start;

                svzero_mask_za(nk_sme_zero_za32_tile_0_);

                svbool_t const batch_predicate_b32 = svwhilelt_b32((uint32_t)0, (uint32_t)batch_size);
                nk_size_t const byte_offset = depth_batch_start * 4;

                for (nk_size_t row_in_tile = 0; row_in_tile < rows_remaining; row_in_tile++) {
                    nk_size_t const a_row = row_start + row_in_tile;
                    nk_u8_t const *src = (nk_u8_t const *)(a + a_row * a_stride_elements) + byte_offset;
                    svld1_hor_za32(0, row_in_tile, batch_predicate_b32, (nk_f32_t const *)src);
                }

                // Last-step ZA0 fixup
                if (depth_batch_start + batch_size >= depth_step_count) {
                    nk_size_t const last_step = batch_size - 1;
                    nk_size_t const global_step = depth_batch_start + last_step;
                    nk_size_t const valid_bytes = packed_depth - global_step * 4;
                    if (valid_bytes < 4 || (depth & 1)) {
                        nk_u32_t mask = (valid_bytes >= 4) ? 0xFFFFFFFFu : ((1u << (valid_bytes * 8)) - 1u);
                        if (depth & 1) {
                            nk_u32_t nibble_clear = ~((nk_u32_t)0xF0 << ((valid_bytes - 1) * 8));
                            mask &= nibble_clear;
                        }
                        for (nk_size_t row_in_tile = 0; row_in_tile < rows_remaining; row_in_tile++) {
                            svuint32_t column = svread_ver_za32_u32_m(svdup_u32(0), row_predicate_b32, 0, last_step);
                            column = svand_n_u32_x(row_predicate_b32, column, mask);
                            svwrite_ver_za32_u32_m(0, last_step, row_predicate_b32, column);
                        }
                    }
                }

                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {

                    svint32_t a_column_i32 = svread_ver_za32_s32_m(svdup_s32(0), row_predicate_b32, 0, depth_step);
                    svint8_t a_packed_nibbles_s8 = svreinterpret_s8_s32(a_column_i32);
                    svint8_t a_low_nibbles = svasr_n_s8_x(full_predicate_b8,
                                                          svlsl_n_s8_x(full_predicate_b8, a_packed_nibbles_s8, 4), 4);
                    svint8_t a_high_nibbles = svasr_n_s8_x(full_predicate_b8, a_packed_nibbles_s8, 4);

                    nk_i8_t const *b_column_pair =
                        (nk_i8_t const *)(b_packed_base +
                                          (column_tile_index * depth_step_count + depth_batch_start + depth_step) *
                                              vector_pair_stride);
                    svint8_t b_low_nibbles = svld1_s8(column_predicate_b8, b_column_pair);
                    svint8_t b_high_nibbles = svld1_s8(column_predicate_b8, b_column_pair + vector_elements);

                    svmopa_za32_s8_m(1, row_predicate_b8, column_predicate_b8, a_low_nibbles, b_low_nibbles);
                    svmopa_za32_s8_m(1, row_predicate_b8, column_predicate_b8, a_high_nibbles, b_high_nibbles);
                }
            }

            for (nk_size_t row = 0; row < rows_remaining; row++) {
                svst1_hor_za32(1, row, column_predicate_b32,
                               (nk_f32_t *)(c + (row_start + row) * c_stride_elements + column_start));
            }
        }
    }
}

NK_PUBLIC void nk_dots_packed_i4_sme(                      //
    nk_i4x2_t const *a, void const *b_packed, nk_i32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,    //
    nk_size_t a_stride, nk_size_t c_stride) {

    nk_size_t const a_stride_elements = a_stride / sizeof(nk_i4x2_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_i32_t);
    nk_dots_packed_i4_sme_kernel_(a, b_packed, c, rows, columns, depth, a_stride_elements, c_stride_elements);
}

/**
 *  `u4` × `u4` → `u32` symmetric kernel using SME UMOPA with direct mask-and-double-MOPA.
 *  Loads packed nibble bytes directly into ZA0, splits into low/high nibbles in registers,
 *  issues 2 UMOPAs per depth step. ZA0 = staging tile, ZA1-ZA3 = accumulators.
 */
__arm_locally_streaming __arm_new("za") static void nk_dots_symmetric_u4_sme_kernel_(
    nk_u4x2_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride_elements, nk_u32_t *result,
    nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) {

    nk_size_t const expansion = 4;
    nk_size_t const tile_dimension = svcntw();
    nk_size_t const packed_depth = nk_size_divide_round_up_(depth, 2);
    nk_size_t const depth_step_count = nk_size_divide_round_up_(packed_depth, 4);
    nk_size_t const depth_steps_per_batch = tile_dimension;

    svbool_t const full_predicate_b8 = svptrue_b8();
    svbool_t const full_predicate_b32 = svptrue_b32();

    NK_ALIGN64 nk_u32_t a_buffer[16][16];

    nk_size_t const row_end = row_start + row_count;
    nk_size_t const column_tile_count = nk_size_divide_round_up_(n_vectors, tile_dimension);

    for (nk_size_t row_tile_start = row_start; row_tile_start < row_end && row_tile_start < n_vectors;
         row_tile_start += tile_dimension) {
        nk_size_t const rows_remaining = (row_tile_start + tile_dimension <= row_end) ? tile_dimension
                                                                                      : (row_end - row_tile_start);
        nk_size_t const rows_clamped = (row_tile_start + rows_remaining <= n_vectors) ? rows_remaining
                                                                                      : (n_vectors - row_tile_start);
        svbool_t const row_predicate_b8 = svwhilelt_b8((uint32_t)0, (uint32_t)(rows_clamped * expansion));
        svbool_t const row_predicate_b32 = svwhilelt_b32((uint32_t)0, (uint32_t)rows_clamped);

        nk_size_t column_tile_index = 0;

        for (; column_tile_index + 3 <= column_tile_count; column_tile_index += 3) {
            svzero_mask_za(nk_sme_zero_za32_tiles_123_);

            for (nk_size_t depth_batch_start = 0; depth_batch_start < depth_step_count;
                 depth_batch_start += depth_steps_per_batch) {
                nk_size_t const depth_batch_end = (depth_batch_start + depth_steps_per_batch < depth_step_count)
                                                      ? depth_batch_start + depth_steps_per_batch
                                                      : depth_step_count;
                nk_size_t const batch_size = depth_batch_end - depth_batch_start;

                // Load A rows into ZA0 (packed bytes directly from source)
                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                svbool_t const batch_predicate_b32 = svwhilelt_b32((uint32_t)0, (uint32_t)batch_size);
                nk_size_t const byte_offset = depth_batch_start * 4;

                for (nk_size_t row_in_tile = 0; row_in_tile < rows_clamped; row_in_tile++) {
                    nk_size_t const row_abs = row_tile_start + row_in_tile;
                    nk_u8_t const *src = (nk_u8_t const *)(vectors + row_abs * stride_elements) + byte_offset;
                    svld1_hor_za32(0, row_in_tile, batch_predicate_b32, (nk_f32_t const *)src);
                }

                // Last-step ZA0 fixup for A
                if (depth_batch_start + batch_size >= depth_step_count) {
                    nk_size_t const last_step = batch_size - 1;
                    nk_size_t const global_step = depth_batch_start + last_step;
                    nk_size_t const valid_bytes = packed_depth - global_step * 4;
                    if (valid_bytes < 4 || (depth & 1)) {
                        nk_u32_t mask = (valid_bytes >= 4) ? 0xFFFFFFFFu : ((1u << (valid_bytes * 8)) - 1u);
                        if (depth & 1) {
                            nk_u32_t nibble_clear = ~((nk_u32_t)0xF0 << ((valid_bytes - 1) * 8));
                            mask &= nibble_clear;
                        }
                        for (nk_size_t row_in_tile = 0; row_in_tile < rows_clamped; row_in_tile++) {
                            svuint32_t column = svread_ver_za32_u32_m(svdup_u32(0), row_predicate_b32, 0, last_step);
                            column = svand_n_u32_x(row_predicate_b32, column, mask);
                            svwrite_ver_za32_u32_m(0, last_step, row_predicate_b32, column);
                        }
                    }
                }

                // Cache A columns
                for (nk_size_t s = 0; s < batch_size; s++)
                    svst1_u32(full_predicate_b32, a_buffer[s],
                              svread_ver_za32_u32_m(svdup_u32(0), row_predicate_b32, 0, s));

                // B column tile 0
                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t column = 0; column < tile_dimension; column++) {
                    nk_size_t const column_abs = (column_tile_index + 0) * tile_dimension + column;
                    if (column_abs < n_vectors) {
                        nk_u8_t const *src = (nk_u8_t const *)(vectors + column_abs * stride_elements) + byte_offset;
                        svld1_hor_za32(0, column, batch_predicate_b32, (nk_f32_t const *)src);
                    }
                }
                if (depth_batch_start + batch_size >= depth_step_count) {
                    nk_size_t const last_step = batch_size - 1;
                    nk_size_t const global_step = depth_batch_start + last_step;
                    nk_size_t const valid_bytes = packed_depth - global_step * 4;
                    if (valid_bytes < 4 || (depth & 1)) {
                        nk_u32_t mask = (valid_bytes >= 4) ? 0xFFFFFFFFu : ((1u << (valid_bytes * 8)) - 1u);
                        if (depth & 1) {
                            nk_u32_t nibble_clear = ~((nk_u32_t)0xF0 << ((valid_bytes - 1) * 8));
                            mask &= nibble_clear;
                        }
                        svuint32_t column = svread_ver_za32_u32_m(svdup_u32(0), full_predicate_b32, 0, last_step);
                        column = svand_n_u32_x(full_predicate_b32, column, mask);
                        svwrite_ver_za32_u32_m(0, last_step, full_predicate_b32, column);
                    }
                }
                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {
                    svuint8_t a_u8 = svreinterpret_u8_u32(svld1_u32(full_predicate_b32, a_buffer[depth_step]));
                    svuint32_t b_column_u32 = svread_ver_za32_u32_m(svdup_u32(0), full_predicate_b32, 0, depth_step);
                    svuint8_t b_u8 = svreinterpret_u8_u32(b_column_u32);
                    svuint8_t a_low = svand_n_u8_x(full_predicate_b8, a_u8, 0x0F);
                    svuint8_t a_high = svlsr_n_u8_x(full_predicate_b8, a_u8, 4);
                    svuint8_t b_low = svand_n_u8_x(full_predicate_b8, b_u8, 0x0F);
                    svuint8_t b_high = svlsr_n_u8_x(full_predicate_b8, b_u8, 4);
                    svmopa_za32_u8_m(1, row_predicate_b8, full_predicate_b8, a_low, b_low);
                    svmopa_za32_u8_m(1, row_predicate_b8, full_predicate_b8, a_high, b_high);
                }

                // B column tile 1
                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t column = 0; column < tile_dimension; column++) {
                    nk_size_t const column_abs = (column_tile_index + 1) * tile_dimension + column;
                    if (column_abs < n_vectors) {
                        nk_u8_t const *src = (nk_u8_t const *)(vectors + column_abs * stride_elements) + byte_offset;
                        svld1_hor_za32(0, column, batch_predicate_b32, (nk_f32_t const *)src);
                    }
                }
                if (depth_batch_start + batch_size >= depth_step_count) {
                    nk_size_t const last_step = batch_size - 1;
                    nk_size_t const global_step = depth_batch_start + last_step;
                    nk_size_t const valid_bytes = packed_depth - global_step * 4;
                    if (valid_bytes < 4 || (depth & 1)) {
                        nk_u32_t mask = (valid_bytes >= 4) ? 0xFFFFFFFFu : ((1u << (valid_bytes * 8)) - 1u);
                        if (depth & 1) {
                            nk_u32_t nibble_clear = ~((nk_u32_t)0xF0 << ((valid_bytes - 1) * 8));
                            mask &= nibble_clear;
                        }
                        svuint32_t column = svread_ver_za32_u32_m(svdup_u32(0), full_predicate_b32, 0, last_step);
                        column = svand_n_u32_x(full_predicate_b32, column, mask);
                        svwrite_ver_za32_u32_m(0, last_step, full_predicate_b32, column);
                    }
                }
                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {
                    svuint8_t a_u8 = svreinterpret_u8_u32(svld1_u32(full_predicate_b32, a_buffer[depth_step]));
                    svuint32_t b_column_u32 = svread_ver_za32_u32_m(svdup_u32(0), full_predicate_b32, 0, depth_step);
                    svuint8_t b_u8 = svreinterpret_u8_u32(b_column_u32);
                    svuint8_t a_low = svand_n_u8_x(full_predicate_b8, a_u8, 0x0F);
                    svuint8_t a_high = svlsr_n_u8_x(full_predicate_b8, a_u8, 4);
                    svuint8_t b_low = svand_n_u8_x(full_predicate_b8, b_u8, 0x0F);
                    svuint8_t b_high = svlsr_n_u8_x(full_predicate_b8, b_u8, 4);
                    svmopa_za32_u8_m(2, row_predicate_b8, full_predicate_b8, a_low, b_low);
                    svmopa_za32_u8_m(2, row_predicate_b8, full_predicate_b8, a_high, b_high);
                }

                // B column tile 2
                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t column = 0; column < tile_dimension; column++) {
                    nk_size_t const column_abs = (column_tile_index + 2) * tile_dimension + column;
                    if (column_abs < n_vectors) {
                        nk_u8_t const *src = (nk_u8_t const *)(vectors + column_abs * stride_elements) + byte_offset;
                        svld1_hor_za32(0, column, batch_predicate_b32, (nk_f32_t const *)src);
                    }
                }
                if (depth_batch_start + batch_size >= depth_step_count) {
                    nk_size_t const last_step = batch_size - 1;
                    nk_size_t const global_step = depth_batch_start + last_step;
                    nk_size_t const valid_bytes = packed_depth - global_step * 4;
                    if (valid_bytes < 4 || (depth & 1)) {
                        nk_u32_t mask = (valid_bytes >= 4) ? 0xFFFFFFFFu : ((1u << (valid_bytes * 8)) - 1u);
                        if (depth & 1) {
                            nk_u32_t nibble_clear = ~((nk_u32_t)0xF0 << ((valid_bytes - 1) * 8));
                            mask &= nibble_clear;
                        }
                        svuint32_t column = svread_ver_za32_u32_m(svdup_u32(0), full_predicate_b32, 0, last_step);
                        column = svand_n_u32_x(full_predicate_b32, column, mask);
                        svwrite_ver_za32_u32_m(0, last_step, full_predicate_b32, column);
                    }
                }
                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {
                    svuint8_t a_u8 = svreinterpret_u8_u32(svld1_u32(full_predicate_b32, a_buffer[depth_step]));
                    svuint32_t b_column_u32 = svread_ver_za32_u32_m(svdup_u32(0), full_predicate_b32, 0, depth_step);
                    svuint8_t b_u8 = svreinterpret_u8_u32(b_column_u32);
                    svuint8_t a_low = svand_n_u8_x(full_predicate_b8, a_u8, 0x0F);
                    svuint8_t a_high = svlsr_n_u8_x(full_predicate_b8, a_u8, 4);
                    svuint8_t b_low = svand_n_u8_x(full_predicate_b8, b_u8, 0x0F);
                    svuint8_t b_high = svlsr_n_u8_x(full_predicate_b8, b_u8, 4);
                    svmopa_za32_u8_m(3, row_predicate_b8, full_predicate_b8, a_low, b_low);
                    svmopa_za32_u8_m(3, row_predicate_b8, full_predicate_b8, a_high, b_high);
                }
            }

            for (nk_size_t row = 0; row < rows_clamped; row++) {
                nk_size_t const row_abs = row_tile_start + row;
                nk_f32_t *result_row = (nk_f32_t *)(result + row_abs * result_stride_elements);
                svst1_hor_za32(1, row, full_predicate_b32, result_row + (column_tile_index + 0) * tile_dimension);
                svst1_hor_za32(2, row, full_predicate_b32, result_row + (column_tile_index + 1) * tile_dimension);
                svst1_hor_za32(3, row, full_predicate_b32, result_row + (column_tile_index + 2) * tile_dimension);
            }
        }

        for (; column_tile_index < column_tile_count; column_tile_index++) {
            nk_size_t const column_tile_start = column_tile_index * tile_dimension;
            nk_size_t const cols_remaining = (column_tile_start + tile_dimension <= n_vectors)
                                                 ? tile_dimension
                                                 : (n_vectors - column_tile_start);
            svbool_t const column_predicate_b32 = svwhilelt_b32((uint32_t)0, (uint32_t)cols_remaining);
            svbool_t const column_predicate_b8 = svwhilelt_b8((uint32_t)0, (uint32_t)(cols_remaining * expansion));

            svzero_mask_za(nk_sme_zero_za32_tile_1_);

            for (nk_size_t depth_batch_start = 0; depth_batch_start < depth_step_count;
                 depth_batch_start += depth_steps_per_batch) {
                nk_size_t const depth_batch_end = (depth_batch_start + depth_steps_per_batch < depth_step_count)
                                                      ? depth_batch_start + depth_steps_per_batch
                                                      : depth_step_count;
                nk_size_t const batch_size = depth_batch_end - depth_batch_start;

                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                svbool_t const batch_predicate_b32 = svwhilelt_b32((uint32_t)0, (uint32_t)batch_size);
                nk_size_t const byte_offset = depth_batch_start * 4;

                for (nk_size_t row_in_tile = 0; row_in_tile < rows_clamped; row_in_tile++) {
                    nk_size_t const row_abs = row_tile_start + row_in_tile;
                    nk_u8_t const *src = (nk_u8_t const *)(vectors + row_abs * stride_elements) + byte_offset;
                    svld1_hor_za32(0, row_in_tile, batch_predicate_b32, (nk_f32_t const *)src);
                }

                // Last-step ZA0 fixup for A
                if (depth_batch_start + batch_size >= depth_step_count) {
                    nk_size_t const last_step = batch_size - 1;
                    nk_size_t const global_step = depth_batch_start + last_step;
                    nk_size_t const valid_bytes = packed_depth - global_step * 4;
                    if (valid_bytes < 4 || (depth & 1)) {
                        nk_u32_t mask = (valid_bytes >= 4) ? 0xFFFFFFFFu : ((1u << (valid_bytes * 8)) - 1u);
                        if (depth & 1) {
                            nk_u32_t nibble_clear = ~((nk_u32_t)0xF0 << ((valid_bytes - 1) * 8));
                            mask &= nibble_clear;
                        }
                        for (nk_size_t row_in_tile = 0; row_in_tile < rows_clamped; row_in_tile++) {
                            svuint32_t column = svread_ver_za32_u32_m(svdup_u32(0), row_predicate_b32, 0, last_step);
                            column = svand_n_u32_x(row_predicate_b32, column, mask);
                            svwrite_ver_za32_u32_m(0, last_step, row_predicate_b32, column);
                        }
                    }
                }

                for (nk_size_t s = 0; s < batch_size; s++)
                    svst1_u32(full_predicate_b32, a_buffer[s],
                              svread_ver_za32_u32_m(svdup_u32(0), row_predicate_b32, 0, s));

                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t column = 0; column < tile_dimension; column++) {
                    nk_size_t const column_abs = column_tile_start + column;
                    if (column_abs < n_vectors) {
                        nk_u8_t const *src = (nk_u8_t const *)(vectors + column_abs * stride_elements) + byte_offset;
                        svld1_hor_za32(0, column, batch_predicate_b32, (nk_f32_t const *)src);
                    }
                }
                // Last-step ZA0 fixup for B
                if (depth_batch_start + batch_size >= depth_step_count) {
                    nk_size_t const last_step = batch_size - 1;
                    nk_size_t const global_step = depth_batch_start + last_step;
                    nk_size_t const valid_bytes = packed_depth - global_step * 4;
                    if (valid_bytes < 4 || (depth & 1)) {
                        nk_u32_t mask = (valid_bytes >= 4) ? 0xFFFFFFFFu : ((1u << (valid_bytes * 8)) - 1u);
                        if (depth & 1) {
                            nk_u32_t nibble_clear = ~((nk_u32_t)0xF0 << ((valid_bytes - 1) * 8));
                            mask &= nibble_clear;
                        }
                        svuint32_t column = svread_ver_za32_u32_m(svdup_u32(0), column_predicate_b32, 0, last_step);
                        column = svand_n_u32_x(column_predicate_b32, column, mask);
                        svwrite_ver_za32_u32_m(0, last_step, column_predicate_b32, column);
                    }
                }
                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {
                    svuint8_t a_u8 = svreinterpret_u8_u32(svld1_u32(full_predicate_b32, a_buffer[depth_step]));
                    svuint32_t b_column_u32 = svread_ver_za32_u32_m(svdup_u32(0), column_predicate_b32, 0, depth_step);
                    svuint8_t b_u8 = svreinterpret_u8_u32(b_column_u32);
                    svuint8_t a_low = svand_n_u8_x(full_predicate_b8, a_u8, 0x0F);
                    svuint8_t a_high = svlsr_n_u8_x(full_predicate_b8, a_u8, 4);
                    svuint8_t b_low = svand_n_u8_x(full_predicate_b8, b_u8, 0x0F);
                    svuint8_t b_high = svlsr_n_u8_x(full_predicate_b8, b_u8, 4);
                    svmopa_za32_u8_m(1, row_predicate_b8, column_predicate_b8, a_low, b_low);
                    svmopa_za32_u8_m(1, row_predicate_b8, column_predicate_b8, a_high, b_high);
                }
            }

            for (nk_size_t row = 0; row < rows_clamped; row++) {
                nk_size_t const row_abs = row_tile_start + row;
                svst1_hor_za32(1, row, column_predicate_b32,
                               (nk_f32_t *)(result + row_abs * result_stride_elements) + column_tile_start);
            }
        }

        for (nk_size_t row = 0; row < rows_clamped; row++) {
            nk_size_t const i = row_tile_start + row;
            for (nk_size_t j = 0; j < i && j < n_vectors; j++) {
                result[j * result_stride_elements + i] = result[i * result_stride_elements + j];
            }
        }
    }
}

NK_PUBLIC void nk_dots_symmetric_u4_sme(nk_u4x2_t const *vectors, nk_size_t n_vectors, nk_size_t depth,
                                        nk_size_t stride, nk_u32_t *result, nk_size_t result_stride,
                                        nk_size_t row_start, nk_size_t row_count) {

    nk_size_t const stride_elements = stride / sizeof(nk_u4x2_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_u32_t);
    nk_dots_symmetric_u4_sme_kernel_(vectors, n_vectors, depth, stride_elements, result, result_stride_elements,
                                     row_start, row_count);
}

/**
 *  `i4` × `i4` → `i32` symmetric kernel using SME SMOPA with direct mask-and-double-MOPA.
 *  Loads packed nibble bytes directly into ZA0, sign-extends via LSL+ASR in registers,
 *  issues 2 SMOPAs per depth step. ZA0 = staging tile, ZA1-ZA3 = accumulators.
 */
__arm_locally_streaming __arm_new("za") static void nk_dots_symmetric_i4_sme_kernel_(
    nk_i4x2_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride_elements, nk_i32_t *result,
    nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) {

    nk_size_t const expansion = 4;
    nk_size_t const tile_dimension = svcntw();
    nk_size_t const packed_depth = nk_size_divide_round_up_(depth, 2);
    nk_size_t const depth_step_count = nk_size_divide_round_up_(packed_depth, 4);
    nk_size_t const depth_steps_per_batch = tile_dimension;

    svbool_t const full_predicate_b8 = svptrue_b8();
    svbool_t const full_predicate_b32 = svptrue_b32();

    NK_ALIGN64 nk_u32_t a_buffer[16][16];

    nk_size_t const row_end = row_start + row_count;
    nk_size_t const column_tile_count = nk_size_divide_round_up_(n_vectors, tile_dimension);

    for (nk_size_t row_tile_start = row_start; row_tile_start < row_end && row_tile_start < n_vectors;
         row_tile_start += tile_dimension) {
        nk_size_t const rows_remaining = (row_tile_start + tile_dimension <= row_end) ? tile_dimension
                                                                                      : (row_end - row_tile_start);
        nk_size_t const rows_clamped = (row_tile_start + rows_remaining <= n_vectors) ? rows_remaining
                                                                                      : (n_vectors - row_tile_start);
        svbool_t const row_predicate_b8 = svwhilelt_b8((uint32_t)0, (uint32_t)(rows_clamped * expansion));
        svbool_t const row_predicate_b32 = svwhilelt_b32((uint32_t)0, (uint32_t)rows_clamped);

        nk_size_t column_tile_index = 0;

        for (; column_tile_index + 3 <= column_tile_count; column_tile_index += 3) {
            svzero_mask_za(nk_sme_zero_za32_tiles_123_);

            for (nk_size_t depth_batch_start = 0; depth_batch_start < depth_step_count;
                 depth_batch_start += depth_steps_per_batch) {
                nk_size_t const depth_batch_end = (depth_batch_start + depth_steps_per_batch < depth_step_count)
                                                      ? depth_batch_start + depth_steps_per_batch
                                                      : depth_step_count;
                nk_size_t const batch_size = depth_batch_end - depth_batch_start;

                // Load A rows into ZA0 (packed bytes directly from source)
                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                svbool_t const batch_predicate_b32 = svwhilelt_b32((uint32_t)0, (uint32_t)batch_size);
                nk_size_t const byte_offset = depth_batch_start * 4;

                for (nk_size_t row_in_tile = 0; row_in_tile < rows_clamped; row_in_tile++) {
                    nk_size_t const row_abs = row_tile_start + row_in_tile;
                    nk_u8_t const *src = (nk_u8_t const *)(vectors + row_abs * stride_elements) + byte_offset;
                    svld1_hor_za32(0, row_in_tile, batch_predicate_b32, (nk_f32_t const *)src);
                }

                // Last-step ZA0 fixup for A
                if (depth_batch_start + batch_size >= depth_step_count) {
                    nk_size_t const last_step = batch_size - 1;
                    nk_size_t const global_step = depth_batch_start + last_step;
                    nk_size_t const valid_bytes = packed_depth - global_step * 4;
                    if (valid_bytes < 4 || (depth & 1)) {
                        nk_u32_t mask = (valid_bytes >= 4) ? 0xFFFFFFFFu : ((1u << (valid_bytes * 8)) - 1u);
                        if (depth & 1) {
                            nk_u32_t nibble_clear = ~((nk_u32_t)0xF0 << ((valid_bytes - 1) * 8));
                            mask &= nibble_clear;
                        }
                        for (nk_size_t row_in_tile = 0; row_in_tile < rows_clamped; row_in_tile++) {
                            svuint32_t column = svread_ver_za32_u32_m(svdup_u32(0), row_predicate_b32, 0, last_step);
                            column = svand_n_u32_x(row_predicate_b32, column, mask);
                            svwrite_ver_za32_u32_m(0, last_step, row_predicate_b32, column);
                        }
                    }
                }

                // Cache A columns
                for (nk_size_t s = 0; s < batch_size; s++)
                    svst1_u32(full_predicate_b32, a_buffer[s],
                              svread_ver_za32_u32_m(svdup_u32(0), row_predicate_b32, 0, s));

                // B column tile 0
                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t column = 0; column < tile_dimension; column++) {
                    nk_size_t const column_abs = (column_tile_index + 0) * tile_dimension + column;
                    if (column_abs < n_vectors) {
                        nk_u8_t const *src = (nk_u8_t const *)(vectors + column_abs * stride_elements) + byte_offset;
                        svld1_hor_za32(0, column, batch_predicate_b32, (nk_f32_t const *)src);
                    }
                }
                if (depth_batch_start + batch_size >= depth_step_count) {
                    nk_size_t const last_step = batch_size - 1;
                    nk_size_t const global_step = depth_batch_start + last_step;
                    nk_size_t const valid_bytes = packed_depth - global_step * 4;
                    if (valid_bytes < 4 || (depth & 1)) {
                        nk_u32_t mask = (valid_bytes >= 4) ? 0xFFFFFFFFu : ((1u << (valid_bytes * 8)) - 1u);
                        if (depth & 1) {
                            nk_u32_t nibble_clear = ~((nk_u32_t)0xF0 << ((valid_bytes - 1) * 8));
                            mask &= nibble_clear;
                        }
                        svuint32_t column = svread_ver_za32_u32_m(svdup_u32(0), full_predicate_b32, 0, last_step);
                        column = svand_n_u32_x(full_predicate_b32, column, mask);
                        svwrite_ver_za32_u32_m(0, last_step, full_predicate_b32, column);
                    }
                }
                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {
                    svint8_t a_s8 = svreinterpret_s8_u32(svld1_u32(full_predicate_b32, a_buffer[depth_step]));
                    svint32_t b_column_i32 = svread_ver_za32_s32_m(svdup_s32(0), full_predicate_b32, 0, depth_step);
                    svint8_t b_s8 = svreinterpret_s8_s32(b_column_i32);
                    svint8_t a_low = svasr_n_s8_x(full_predicate_b8, svlsl_n_s8_x(full_predicate_b8, a_s8, 4), 4);
                    svint8_t a_high = svasr_n_s8_x(full_predicate_b8, a_s8, 4);
                    svint8_t b_low = svasr_n_s8_x(full_predicate_b8, svlsl_n_s8_x(full_predicate_b8, b_s8, 4), 4);
                    svint8_t b_high = svasr_n_s8_x(full_predicate_b8, b_s8, 4);
                    svmopa_za32_s8_m(1, row_predicate_b8, full_predicate_b8, a_low, b_low);
                    svmopa_za32_s8_m(1, row_predicate_b8, full_predicate_b8, a_high, b_high);
                }

                // B column tile 1
                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t column = 0; column < tile_dimension; column++) {
                    nk_size_t const column_abs = (column_tile_index + 1) * tile_dimension + column;
                    if (column_abs < n_vectors) {
                        nk_u8_t const *src = (nk_u8_t const *)(vectors + column_abs * stride_elements) + byte_offset;
                        svld1_hor_za32(0, column, batch_predicate_b32, (nk_f32_t const *)src);
                    }
                }
                if (depth_batch_start + batch_size >= depth_step_count) {
                    nk_size_t const last_step = batch_size - 1;
                    nk_size_t const global_step = depth_batch_start + last_step;
                    nk_size_t const valid_bytes = packed_depth - global_step * 4;
                    if (valid_bytes < 4 || (depth & 1)) {
                        nk_u32_t mask = (valid_bytes >= 4) ? 0xFFFFFFFFu : ((1u << (valid_bytes * 8)) - 1u);
                        if (depth & 1) {
                            nk_u32_t nibble_clear = ~((nk_u32_t)0xF0 << ((valid_bytes - 1) * 8));
                            mask &= nibble_clear;
                        }
                        svuint32_t column = svread_ver_za32_u32_m(svdup_u32(0), full_predicate_b32, 0, last_step);
                        column = svand_n_u32_x(full_predicate_b32, column, mask);
                        svwrite_ver_za32_u32_m(0, last_step, full_predicate_b32, column);
                    }
                }
                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {
                    svint8_t a_s8 = svreinterpret_s8_u32(svld1_u32(full_predicate_b32, a_buffer[depth_step]));
                    svint32_t b_column_i32 = svread_ver_za32_s32_m(svdup_s32(0), full_predicate_b32, 0, depth_step);
                    svint8_t b_s8 = svreinterpret_s8_s32(b_column_i32);
                    svint8_t a_low = svasr_n_s8_x(full_predicate_b8, svlsl_n_s8_x(full_predicate_b8, a_s8, 4), 4);
                    svint8_t a_high = svasr_n_s8_x(full_predicate_b8, a_s8, 4);
                    svint8_t b_low = svasr_n_s8_x(full_predicate_b8, svlsl_n_s8_x(full_predicate_b8, b_s8, 4), 4);
                    svint8_t b_high = svasr_n_s8_x(full_predicate_b8, b_s8, 4);
                    svmopa_za32_s8_m(2, row_predicate_b8, full_predicate_b8, a_low, b_low);
                    svmopa_za32_s8_m(2, row_predicate_b8, full_predicate_b8, a_high, b_high);
                }

                // B column tile 2
                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t column = 0; column < tile_dimension; column++) {
                    nk_size_t const column_abs = (column_tile_index + 2) * tile_dimension + column;
                    if (column_abs < n_vectors) {
                        nk_u8_t const *src = (nk_u8_t const *)(vectors + column_abs * stride_elements) + byte_offset;
                        svld1_hor_za32(0, column, batch_predicate_b32, (nk_f32_t const *)src);
                    }
                }
                if (depth_batch_start + batch_size >= depth_step_count) {
                    nk_size_t const last_step = batch_size - 1;
                    nk_size_t const global_step = depth_batch_start + last_step;
                    nk_size_t const valid_bytes = packed_depth - global_step * 4;
                    if (valid_bytes < 4 || (depth & 1)) {
                        nk_u32_t mask = (valid_bytes >= 4) ? 0xFFFFFFFFu : ((1u << (valid_bytes * 8)) - 1u);
                        if (depth & 1) {
                            nk_u32_t nibble_clear = ~((nk_u32_t)0xF0 << ((valid_bytes - 1) * 8));
                            mask &= nibble_clear;
                        }
                        svuint32_t column = svread_ver_za32_u32_m(svdup_u32(0), full_predicate_b32, 0, last_step);
                        column = svand_n_u32_x(full_predicate_b32, column, mask);
                        svwrite_ver_za32_u32_m(0, last_step, full_predicate_b32, column);
                    }
                }
                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {
                    svint8_t a_s8 = svreinterpret_s8_u32(svld1_u32(full_predicate_b32, a_buffer[depth_step]));
                    svint32_t b_column_i32 = svread_ver_za32_s32_m(svdup_s32(0), full_predicate_b32, 0, depth_step);
                    svint8_t b_s8 = svreinterpret_s8_s32(b_column_i32);
                    svint8_t a_low = svasr_n_s8_x(full_predicate_b8, svlsl_n_s8_x(full_predicate_b8, a_s8, 4), 4);
                    svint8_t a_high = svasr_n_s8_x(full_predicate_b8, a_s8, 4);
                    svint8_t b_low = svasr_n_s8_x(full_predicate_b8, svlsl_n_s8_x(full_predicate_b8, b_s8, 4), 4);
                    svint8_t b_high = svasr_n_s8_x(full_predicate_b8, b_s8, 4);
                    svmopa_za32_s8_m(3, row_predicate_b8, full_predicate_b8, a_low, b_low);
                    svmopa_za32_s8_m(3, row_predicate_b8, full_predicate_b8, a_high, b_high);
                }
            }

            for (nk_size_t row = 0; row < rows_clamped; row++) {
                nk_size_t const row_abs = row_tile_start + row;
                nk_f32_t *result_row = (nk_f32_t *)(result + row_abs * result_stride_elements);
                svst1_hor_za32(1, row, full_predicate_b32, result_row + (column_tile_index + 0) * tile_dimension);
                svst1_hor_za32(2, row, full_predicate_b32, result_row + (column_tile_index + 1) * tile_dimension);
                svst1_hor_za32(3, row, full_predicate_b32, result_row + (column_tile_index + 2) * tile_dimension);
            }
        }

        for (; column_tile_index < column_tile_count; column_tile_index++) {
            nk_size_t const column_tile_start = column_tile_index * tile_dimension;
            nk_size_t const cols_remaining = (column_tile_start + tile_dimension <= n_vectors)
                                                 ? tile_dimension
                                                 : (n_vectors - column_tile_start);
            svbool_t const column_predicate_b32 = svwhilelt_b32((uint32_t)0, (uint32_t)cols_remaining);
            svbool_t const column_predicate_b8 = svwhilelt_b8((uint32_t)0, (uint32_t)(cols_remaining * expansion));

            svzero_mask_za(nk_sme_zero_za32_tile_1_);

            for (nk_size_t depth_batch_start = 0; depth_batch_start < depth_step_count;
                 depth_batch_start += depth_steps_per_batch) {
                nk_size_t const depth_batch_end = (depth_batch_start + depth_steps_per_batch < depth_step_count)
                                                      ? depth_batch_start + depth_steps_per_batch
                                                      : depth_step_count;
                nk_size_t const batch_size = depth_batch_end - depth_batch_start;

                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                svbool_t const batch_predicate_b32 = svwhilelt_b32((uint32_t)0, (uint32_t)batch_size);
                nk_size_t const byte_offset = depth_batch_start * 4;

                for (nk_size_t row_in_tile = 0; row_in_tile < rows_clamped; row_in_tile++) {
                    nk_size_t const row_abs = row_tile_start + row_in_tile;
                    nk_u8_t const *src = (nk_u8_t const *)(vectors + row_abs * stride_elements) + byte_offset;
                    svld1_hor_za32(0, row_in_tile, batch_predicate_b32, (nk_f32_t const *)src);
                }

                // Last-step ZA0 fixup for A
                if (depth_batch_start + batch_size >= depth_step_count) {
                    nk_size_t const last_step = batch_size - 1;
                    nk_size_t const global_step = depth_batch_start + last_step;
                    nk_size_t const valid_bytes = packed_depth - global_step * 4;
                    if (valid_bytes < 4 || (depth & 1)) {
                        nk_u32_t mask = (valid_bytes >= 4) ? 0xFFFFFFFFu : ((1u << (valid_bytes * 8)) - 1u);
                        if (depth & 1) {
                            nk_u32_t nibble_clear = ~((nk_u32_t)0xF0 << ((valid_bytes - 1) * 8));
                            mask &= nibble_clear;
                        }
                        for (nk_size_t row_in_tile = 0; row_in_tile < rows_clamped; row_in_tile++) {
                            svuint32_t column = svread_ver_za32_u32_m(svdup_u32(0), row_predicate_b32, 0, last_step);
                            column = svand_n_u32_x(row_predicate_b32, column, mask);
                            svwrite_ver_za32_u32_m(0, last_step, row_predicate_b32, column);
                        }
                    }
                }

                for (nk_size_t s = 0; s < batch_size; s++)
                    svst1_u32(full_predicate_b32, a_buffer[s],
                              svread_ver_za32_u32_m(svdup_u32(0), row_predicate_b32, 0, s));

                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t column = 0; column < tile_dimension; column++) {
                    nk_size_t const column_abs = column_tile_start + column;
                    if (column_abs < n_vectors) {
                        nk_u8_t const *src = (nk_u8_t const *)(vectors + column_abs * stride_elements) + byte_offset;
                        svld1_hor_za32(0, column, batch_predicate_b32, (nk_f32_t const *)src);
                    }
                }
                // Last-step ZA0 fixup for B
                if (depth_batch_start + batch_size >= depth_step_count) {
                    nk_size_t const last_step = batch_size - 1;
                    nk_size_t const global_step = depth_batch_start + last_step;
                    nk_size_t const valid_bytes = packed_depth - global_step * 4;
                    if (valid_bytes < 4 || (depth & 1)) {
                        nk_u32_t mask = (valid_bytes >= 4) ? 0xFFFFFFFFu : ((1u << (valid_bytes * 8)) - 1u);
                        if (depth & 1) {
                            nk_u32_t nibble_clear = ~((nk_u32_t)0xF0 << ((valid_bytes - 1) * 8));
                            mask &= nibble_clear;
                        }
                        svuint32_t column = svread_ver_za32_u32_m(svdup_u32(0), column_predicate_b32, 0, last_step);
                        column = svand_n_u32_x(column_predicate_b32, column, mask);
                        svwrite_ver_za32_u32_m(0, last_step, column_predicate_b32, column);
                    }
                }
                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {
                    svint8_t a_s8 = svreinterpret_s8_u32(svld1_u32(full_predicate_b32, a_buffer[depth_step]));
                    svint32_t b_column_i32 = svread_ver_za32_s32_m(svdup_s32(0), column_predicate_b32, 0, depth_step);
                    svint8_t b_s8 = svreinterpret_s8_s32(b_column_i32);
                    svint8_t a_low = svasr_n_s8_x(full_predicate_b8, svlsl_n_s8_x(full_predicate_b8, a_s8, 4), 4);
                    svint8_t a_high = svasr_n_s8_x(full_predicate_b8, a_s8, 4);
                    svint8_t b_low = svasr_n_s8_x(full_predicate_b8, svlsl_n_s8_x(full_predicate_b8, b_s8, 4), 4);
                    svint8_t b_high = svasr_n_s8_x(full_predicate_b8, b_s8, 4);
                    svmopa_za32_s8_m(1, row_predicate_b8, column_predicate_b8, a_low, b_low);
                    svmopa_za32_s8_m(1, row_predicate_b8, column_predicate_b8, a_high, b_high);
                }
            }

            for (nk_size_t row = 0; row < rows_clamped; row++) {
                nk_size_t const row_abs = row_tile_start + row;
                svst1_hor_za32(1, row, column_predicate_b32,
                               (nk_f32_t *)(result + row_abs * result_stride_elements) + column_tile_start);
            }
        }

        for (nk_size_t row = 0; row < rows_clamped; row++) {
            nk_size_t const i = row_tile_start + row;
            for (nk_size_t j = 0; j < i && j < n_vectors; j++) {
                result[j * result_stride_elements + i] = result[i * result_stride_elements + j];
            }
        }
    }
}

NK_PUBLIC void nk_dots_symmetric_i4_sme(nk_i4x2_t const *vectors, nk_size_t n_vectors, nk_size_t depth,
                                        nk_size_t stride, nk_i32_t *result, nk_size_t result_stride,
                                        nk_size_t row_start, nk_size_t row_count) {

    nk_size_t const stride_elements = stride / sizeof(nk_i4x2_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_i32_t);
    nk_dots_symmetric_i4_sme_kernel_(vectors, n_vectors, depth, stride_elements, result, result_stride_elements,
                                     row_start, row_count);
}

#pragma endregion
#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_SME
#endif // NK_TARGET_ARM_
#endif // NK_DOTS_SME_H

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
    //   packed_vec[expansion * d + k] = B[col_tile*tile_dim + d, depth_step*expansion + k]
    // d = 0..tile_dim-1 (B row within column tile), k = 0..expansion-1 (depth sub-element)
    for (nk_size_t col_tile = 0; col_tile < column_tile_count; col_tile++) {
        for (nk_size_t depth_step = 0; depth_step < depth_step_count; depth_step++) {
            nk_size_t const vec_index = col_tile * depth_step_count + depth_step;
            nk_f16_t *vec_output = tiles_ptr + vec_index * vector_elements;

            nk_size_t const b_row_start = col_tile * tile_dimension;
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

    // Interleaved packing: packed_vec[expansion * d + k] = B[col_tile*tile_dim + d, depth_step*expansion + k]
    for (nk_size_t col_tile = 0; col_tile < column_tile_count; col_tile++) {
        for (nk_size_t depth_step = 0; depth_step < depth_step_count; depth_step++) {
            nk_size_t const vec_index = col_tile * depth_step_count + depth_step;
            nk_bf16_t *vec_output = tiles_ptr + vec_index * vector_elements;

            nk_size_t const b_row_start = col_tile * tile_dimension;
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
 *  - Zm[2*d+k] = B[col_start+d, depth_base+k]   (pre-packed interleaved)
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

    nk_f16_t const *b_vecs = (nk_f16_t const *)((char const *)b_packed + sizeof(nk_dots_sme_packed_header_t));

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
                for (nk_size_t row_within_tile = 0; row_within_tile < rows_remaining; row_within_tile++) {
                    nk_size_t const a_row = row_start + row_within_tile;
                    nk_f32_t const *a_row_ptr = (nk_f32_t const *)(a + a_row * a_stride_elements +
                                                                   depth_batch_start * expansion);
                    svld1_hor_za32(0, row_within_tile, batch_predicate_b32, a_row_ptr);
                }

                // For each depth step in batch: vertical read = transpose = perfect interleaving
                for (nk_size_t step_within_batch = 0; step_within_batch < batch_size; step_within_batch++) {
                    nk_size_t const ds = depth_batch_start + step_within_batch;
                    svfloat32_t a_column_f32 = svread_ver_za32_f32_m(svdup_f32(0.0f), row_predicate_b32, 0,
                                                                     step_within_batch);
                    svfloat16_t a_interleaved_vector_f16 = svreinterpret_f16_f32(a_column_f32);

                    // Load B vectors for 3 column tiles
                    nk_f16_t const *bv0 = b_vecs + ((column_tile_index + 0) * depth_step_count + ds) * vector_elements;
                    nk_f16_t const *bv1 = b_vecs + ((column_tile_index + 1) * depth_step_count + ds) * vector_elements;
                    nk_f16_t const *bv2 = b_vecs + ((column_tile_index + 2) * depth_step_count + ds) * vector_elements;
                    svfloat16_t b_packed_vector_f16_0 = svld1_f16(full_predicate_b16, (float16_t const *)bv0);
                    svfloat16_t b_packed_vector_f16_1 = svld1_f16(full_predicate_b16, (float16_t const *)bv1);
                    svfloat16_t b_packed_vector_f16_2 = svld1_f16(full_predicate_b16, (float16_t const *)bv2);

                    // 3 MOPA calls into ZA1, ZA2, ZA3
                    svmopa_za32_f16_m(1, row_predicate_b16, full_predicate_b16, a_interleaved_vector_f16,
                                      b_packed_vector_f16_0);
                    svmopa_za32_f16_m(2, row_predicate_b16, full_predicate_b16, a_interleaved_vector_f16,
                                      b_packed_vector_f16_1);
                    svmopa_za32_f16_m(3, row_predicate_b16, full_predicate_b16, a_interleaved_vector_f16,
                                      b_packed_vector_f16_2);
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
            nk_size_t const col_start = column_tile_index * tile_dimension;
            nk_size_t const cols_remaining = (col_start + tile_dimension <= columns) ? tile_dimension
                                                                                     : (columns - col_start);
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
                for (nk_size_t row_within_tile = 0; row_within_tile < rows_remaining; row_within_tile++) {
                    nk_size_t const a_row = row_start + row_within_tile;
                    nk_f32_t const *a_row_ptr = (nk_f32_t const *)(a + a_row * a_stride_elements +
                                                                   depth_batch_start * expansion);
                    svld1_hor_za32(0, row_within_tile, batch_predicate_b32, a_row_ptr);
                }

                for (nk_size_t step_within_batch = 0; step_within_batch < batch_size; step_within_batch++) {
                    nk_size_t const ds = depth_batch_start + step_within_batch;
                    svfloat32_t a_column_f32 = svread_ver_za32_f32_m(svdup_f32(0.0f), row_predicate_b32, 0,
                                                                     step_within_batch);
                    svfloat16_t a_interleaved_vector_f16 = svreinterpret_f16_f32(a_column_f32);

                    nk_f16_t const *bv = b_vecs + (column_tile_index * depth_step_count + ds) * vector_elements;
                    svfloat16_t b_packed_vector_f16 = svld1_f16(column_predicate_b16, (float16_t const *)bv);

                    svmopa_za32_f16_m(1, row_predicate_b16, column_predicate_b16, a_interleaved_vector_f16,
                                      b_packed_vector_f16);
                }
            }

            for (nk_size_t row = 0; row < rows_remaining; row++) {
                svst1_hor_za32(1, row, column_predicate_b32, c + (row_start + row) * c_stride_elements + col_start);
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

    nk_bf16_t const *b_vecs = (nk_bf16_t const *)((char const *)b_packed + sizeof(nk_dots_sme_packed_header_t));

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
                for (nk_size_t row_within_tile = 0; row_within_tile < rows_remaining; row_within_tile++) {
                    nk_size_t const a_row = row_start + row_within_tile;
                    nk_f32_t const *a_row_ptr = (nk_f32_t const *)(a + a_row * a_stride_elements +
                                                                   depth_batch_start * expansion);
                    svld1_hor_za32(0, row_within_tile, batch_predicate_b32, a_row_ptr);
                }

                for (nk_size_t step_within_batch = 0; step_within_batch < batch_size; step_within_batch++) {
                    nk_size_t const ds = depth_batch_start + step_within_batch;
                    svfloat32_t a_column_f32 = svread_ver_za32_f32_m(svdup_f32(0.0f), row_predicate_b32, 0,
                                                                     step_within_batch);
                    svbfloat16_t a_interleaved_vector_bf16 = svreinterpret_bf16_f32(a_column_f32);

                    nk_bf16_t const *bv0 = b_vecs + ((column_tile_index + 0) * depth_step_count + ds) * vector_elements;
                    nk_bf16_t const *bv1 = b_vecs + ((column_tile_index + 1) * depth_step_count + ds) * vector_elements;
                    nk_bf16_t const *bv2 = b_vecs + ((column_tile_index + 2) * depth_step_count + ds) * vector_elements;
                    svbfloat16_t b_packed_vector_bf16_0 = svld1_bf16(full_predicate_b16, (bfloat16_t const *)bv0);
                    svbfloat16_t b_packed_vector_bf16_1 = svld1_bf16(full_predicate_b16, (bfloat16_t const *)bv1);
                    svbfloat16_t b_packed_vector_bf16_2 = svld1_bf16(full_predicate_b16, (bfloat16_t const *)bv2);

                    svmopa_za32_bf16_m(1, row_predicate_b16, full_predicate_b16, a_interleaved_vector_bf16,
                                       b_packed_vector_bf16_0);
                    svmopa_za32_bf16_m(2, row_predicate_b16, full_predicate_b16, a_interleaved_vector_bf16,
                                       b_packed_vector_bf16_1);
                    svmopa_za32_bf16_m(3, row_predicate_b16, full_predicate_b16, a_interleaved_vector_bf16,
                                       b_packed_vector_bf16_2);
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
            nk_size_t const col_start = column_tile_index * tile_dimension;
            nk_size_t const cols_remaining = (col_start + tile_dimension <= columns) ? tile_dimension
                                                                                     : (columns - col_start);
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
                for (nk_size_t row_within_tile = 0; row_within_tile < rows_remaining; row_within_tile++) {
                    nk_size_t const a_row = row_start + row_within_tile;
                    nk_f32_t const *a_row_ptr = (nk_f32_t const *)(a + a_row * a_stride_elements +
                                                                   depth_batch_start * expansion);
                    svld1_hor_za32(0, row_within_tile, batch_predicate_b32, a_row_ptr);
                }

                for (nk_size_t step_within_batch = 0; step_within_batch < batch_size; step_within_batch++) {
                    nk_size_t const ds = depth_batch_start + step_within_batch;
                    svfloat32_t a_column_f32 = svread_ver_za32_f32_m(svdup_f32(0.0f), row_predicate_b32, 0,
                                                                     step_within_batch);
                    svbfloat16_t a_interleaved_vector_bf16 = svreinterpret_bf16_f32(a_column_f32);

                    nk_bf16_t const *bv = b_vecs + (column_tile_index * depth_step_count + ds) * vector_elements;
                    svbfloat16_t b_packed_vector_bf16 = svld1_bf16(column_predicate_b16, (bfloat16_t const *)bv);

                    svmopa_za32_bf16_m(1, row_predicate_b16, column_predicate_b16, a_interleaved_vector_bf16,
                                       b_packed_vector_bf16);
                }
            }

            for (nk_size_t row = 0; row < rows_remaining; row++) {
                svst1_hor_za32(1, row, column_predicate_b32, c + (row_start + row) * c_stride_elements + col_start);
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
                for (nk_size_t col = 0; col < tile_dimension; col++) {
                    nk_size_t const col_abs = (column_tile_index + 0) * tile_dimension + col;
                    if (col_abs < n_vectors) {
                        nk_f32_t const *b_row = (nk_f32_t const *)(vectors + col_abs * stride_elements +
                                                                   depth_batch_start * expansion);
                        svld1_hor_za32(0, col, batch_predicate_b32, b_row);
                    }
                }
                for (nk_size_t step = 0; step < batch_size; step++) {
                    svfloat16_t a_interleaved = svreinterpret_f16_f32(svld1_f32(full_predicate_b32, a_buffer[step]));
                    svfloat32_t b_col_f32 = svread_ver_za32_f32_m(svdup_f32(0.0f), full_predicate_b32, 0, step);
                    svfloat16_t b_interleaved = svreinterpret_f16_f32(b_col_f32);
                    svmopa_za32_f16_m(1, row_predicate_b16, full_predicate_b16, a_interleaved, b_interleaved);
                }

                // Load B column tile 1 into ZA0, vertical read + FMOPA into ZA2
                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t col = 0; col < tile_dimension; col++) {
                    nk_size_t const col_abs = (column_tile_index + 1) * tile_dimension + col;
                    if (col_abs < n_vectors) {
                        nk_f32_t const *b_row = (nk_f32_t const *)(vectors + col_abs * stride_elements +
                                                                   depth_batch_start * expansion);
                        svld1_hor_za32(0, col, batch_predicate_b32, b_row);
                    }
                }
                for (nk_size_t step = 0; step < batch_size; step++) {
                    svfloat16_t a_interleaved = svreinterpret_f16_f32(svld1_f32(full_predicate_b32, a_buffer[step]));
                    svfloat32_t b_col_f32 = svread_ver_za32_f32_m(svdup_f32(0.0f), full_predicate_b32, 0, step);
                    svfloat16_t b_interleaved = svreinterpret_f16_f32(b_col_f32);
                    svmopa_za32_f16_m(2, row_predicate_b16, full_predicate_b16, a_interleaved, b_interleaved);
                }

                // Load B column tile 2 into ZA0, vertical read + FMOPA into ZA3
                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t col = 0; col < tile_dimension; col++) {
                    nk_size_t const col_abs = (column_tile_index + 2) * tile_dimension + col;
                    if (col_abs < n_vectors) {
                        nk_f32_t const *b_row = (nk_f32_t const *)(vectors + col_abs * stride_elements +
                                                                   depth_batch_start * expansion);
                        svld1_hor_za32(0, col, batch_predicate_b32, b_row);
                    }
                }
                for (nk_size_t step = 0; step < batch_size; step++) {
                    svfloat16_t a_interleaved = svreinterpret_f16_f32(svld1_f32(full_predicate_b32, a_buffer[step]));
                    svfloat32_t b_col_f32 = svread_ver_za32_f32_m(svdup_f32(0.0f), full_predicate_b32, 0, step);
                    svfloat16_t b_interleaved = svreinterpret_f16_f32(b_col_f32);
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
            nk_size_t const col_tile_start = column_tile_index * tile_dimension;
            nk_size_t const cols_remaining = (col_tile_start + tile_dimension <= n_vectors)
                                                 ? tile_dimension
                                                 : (n_vectors - col_tile_start);
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
                for (nk_size_t col = 0; col < tile_dimension; col++) {
                    nk_size_t const col_abs = col_tile_start + col;
                    if (col_abs < n_vectors) {
                        nk_f32_t const *b_row = (nk_f32_t const *)(vectors + col_abs * stride_elements +
                                                                   depth_batch_start * expansion);
                        svld1_hor_za32(0, col, batch_predicate_b32, b_row);
                    }
                }
                for (nk_size_t step = 0; step < batch_size; step++) {
                    svfloat16_t a_interleaved = svreinterpret_f16_f32(svld1_f32(full_predicate_b32, a_buffer[step]));
                    svfloat32_t b_col_f32 = svread_ver_za32_f32_m(svdup_f32(0.0f), column_predicate_b32, 0, step);
                    svfloat16_t b_interleaved = svreinterpret_f16_f32(b_col_f32);
                    svmopa_za32_f16_m(1, row_predicate_b16, column_predicate_b16, a_interleaved, b_interleaved);
                }
            }

            for (nk_size_t row = 0; row < rows_clamped; row++) {
                nk_size_t const row_abs = row_tile_start + row;
                svst1_hor_za32(1, row, column_predicate_b32,
                               result + row_abs * result_stride_elements + col_tile_start);
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
                for (nk_size_t col = 0; col < tile_dimension; col++) {
                    nk_size_t const col_abs = (column_tile_index + 0) * tile_dimension + col;
                    if (col_abs < n_vectors) {
                        nk_f32_t const *b_row = (nk_f32_t const *)(vectors + col_abs * stride_elements +
                                                                   depth_batch_start * expansion);
                        svld1_hor_za32(0, col, batch_predicate_b32, b_row);
                    }
                }
                for (nk_size_t step = 0; step < batch_size; step++) {
                    svbfloat16_t a_interleaved = svreinterpret_bf16_f32(svld1_f32(full_predicate_b32, a_buffer[step]));
                    svfloat32_t b_col_f32 = svread_ver_za32_f32_m(svdup_f32(0.0f), full_predicate_b32, 0, step);
                    svbfloat16_t b_interleaved = svreinterpret_bf16_f32(b_col_f32);
                    svmopa_za32_bf16_m(1, row_predicate_b16, full_predicate_b16, a_interleaved, b_interleaved);
                }

                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t col = 0; col < tile_dimension; col++) {
                    nk_size_t const col_abs = (column_tile_index + 1) * tile_dimension + col;
                    if (col_abs < n_vectors) {
                        nk_f32_t const *b_row = (nk_f32_t const *)(vectors + col_abs * stride_elements +
                                                                   depth_batch_start * expansion);
                        svld1_hor_za32(0, col, batch_predicate_b32, b_row);
                    }
                }
                for (nk_size_t step = 0; step < batch_size; step++) {
                    svbfloat16_t a_interleaved = svreinterpret_bf16_f32(svld1_f32(full_predicate_b32, a_buffer[step]));
                    svfloat32_t b_col_f32 = svread_ver_za32_f32_m(svdup_f32(0.0f), full_predicate_b32, 0, step);
                    svbfloat16_t b_interleaved = svreinterpret_bf16_f32(b_col_f32);
                    svmopa_za32_bf16_m(2, row_predicate_b16, full_predicate_b16, a_interleaved, b_interleaved);
                }

                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t col = 0; col < tile_dimension; col++) {
                    nk_size_t const col_abs = (column_tile_index + 2) * tile_dimension + col;
                    if (col_abs < n_vectors) {
                        nk_f32_t const *b_row = (nk_f32_t const *)(vectors + col_abs * stride_elements +
                                                                   depth_batch_start * expansion);
                        svld1_hor_za32(0, col, batch_predicate_b32, b_row);
                    }
                }
                for (nk_size_t step = 0; step < batch_size; step++) {
                    svbfloat16_t a_interleaved = svreinterpret_bf16_f32(svld1_f32(full_predicate_b32, a_buffer[step]));
                    svfloat32_t b_col_f32 = svread_ver_za32_f32_m(svdup_f32(0.0f), full_predicate_b32, 0, step);
                    svbfloat16_t b_interleaved = svreinterpret_bf16_f32(b_col_f32);
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
            nk_size_t const col_tile_start = column_tile_index * tile_dimension;
            nk_size_t const cols_remaining = (col_tile_start + tile_dimension <= n_vectors)
                                                 ? tile_dimension
                                                 : (n_vectors - col_tile_start);
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
                for (nk_size_t col = 0; col < tile_dimension; col++) {
                    nk_size_t const col_abs = col_tile_start + col;
                    if (col_abs < n_vectors) {
                        nk_f32_t const *b_row = (nk_f32_t const *)(vectors + col_abs * stride_elements +
                                                                   depth_batch_start * expansion);
                        svld1_hor_za32(0, col, batch_predicate_b32, b_row);
                    }
                }
                for (nk_size_t step = 0; step < batch_size; step++) {
                    svbfloat16_t a_interleaved = svreinterpret_bf16_f32(svld1_f32(full_predicate_b32, a_buffer[step]));
                    svfloat32_t b_col_f32 = svread_ver_za32_f32_m(svdup_f32(0.0f), column_predicate_b32, 0, step);
                    svbfloat16_t b_interleaved = svreinterpret_bf16_f32(b_col_f32);
                    svmopa_za32_bf16_m(1, row_predicate_b16, column_predicate_b16, a_interleaved, b_interleaved);
                }
            }

            for (nk_size_t row = 0; row < rows_clamped; row++) {
                nk_size_t const row_abs = row_tile_start + row;
                svst1_hor_za32(1, row, column_predicate_b32,
                               result + row_abs * result_stride_elements + col_tile_start);
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

    // Interleaved packing: packed_vec[expansion * d + k] = B[col_tile*tile_dim + d, depth_step*expansion + k]
    for (nk_size_t col_tile = 0; col_tile < column_tile_count; col_tile++) {
        for (nk_size_t depth_step = 0; depth_step < depth_step_count; depth_step++) {
            nk_size_t const vec_index = col_tile * depth_step_count + depth_step;
            nk_i8_t *vec_output = tiles_ptr + vec_index * vector_elements;

            nk_size_t const b_row_start = col_tile * tile_dimension;
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
 *  - Zm[4*d+k] = B[col_start+d, depth_base+k]   (pre-packed interleaved)
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

    nk_i8_t const *b_vecs = (nk_i8_t const *)((char const *)b_packed + sizeof(nk_dots_sme_packed_header_t));

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
                for (nk_size_t row_within_tile = 0; row_within_tile < rows_remaining; row_within_tile++) {
                    nk_size_t const a_row = row_start + row_within_tile;
                    nk_f32_t const *a_row_ptr = (nk_f32_t const *)(a + a_row * a_stride_elements +
                                                                   depth_batch_start * expansion);
                    svld1_hor_za32(0, row_within_tile, batch_predicate_b32, a_row_ptr);
                }

                for (nk_size_t step_within_batch = 0; step_within_batch < batch_size; step_within_batch++) {
                    nk_size_t const ds = depth_batch_start + step_within_batch;
                    // Vertical read at f32 granularity produces [row0_k0..k3, row1_k0..k3, ...]
                    svint32_t a_column_i32 = svread_ver_za32_s32_m(svdup_s32(0), row_predicate_b32, 0,
                                                                   step_within_batch);
                    svint8_t a_interleaved_vector_i8 = svreinterpret_s8_s32(a_column_i32);

                    nk_i8_t const *bv0 = b_vecs + ((column_tile_index + 0) * depth_step_count + ds) * vector_elements;
                    nk_i8_t const *bv1 = b_vecs + ((column_tile_index + 1) * depth_step_count + ds) * vector_elements;
                    nk_i8_t const *bv2 = b_vecs + ((column_tile_index + 2) * depth_step_count + ds) * vector_elements;
                    svint8_t b_packed_vector_i8_0 = svld1_s8(full_predicate_b8, bv0);
                    svint8_t b_packed_vector_i8_1 = svld1_s8(full_predicate_b8, bv1);
                    svint8_t b_packed_vector_i8_2 = svld1_s8(full_predicate_b8, bv2);

                    svmopa_za32_s8_m(1, row_predicate_b8, full_predicate_b8, a_interleaved_vector_i8,
                                     b_packed_vector_i8_0);
                    svmopa_za32_s8_m(2, row_predicate_b8, full_predicate_b8, a_interleaved_vector_i8,
                                     b_packed_vector_i8_1);
                    svmopa_za32_s8_m(3, row_predicate_b8, full_predicate_b8, a_interleaved_vector_i8,
                                     b_packed_vector_i8_2);
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
            nk_size_t const col_start = column_tile_index * tile_dimension;
            nk_size_t const cols_remaining = (col_start + tile_dimension <= columns) ? tile_dimension
                                                                                     : (columns - col_start);
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
                for (nk_size_t row_within_tile = 0; row_within_tile < rows_remaining; row_within_tile++) {
                    nk_size_t const a_row = row_start + row_within_tile;
                    nk_f32_t const *a_row_ptr = (nk_f32_t const *)(a + a_row * a_stride_elements +
                                                                   depth_batch_start * expansion);
                    svld1_hor_za32(0, row_within_tile, batch_predicate_b32, a_row_ptr);
                }

                for (nk_size_t step_within_batch = 0; step_within_batch < batch_size; step_within_batch++) {
                    nk_size_t const ds = depth_batch_start + step_within_batch;
                    svint32_t a_column_i32 = svread_ver_za32_s32_m(svdup_s32(0), row_predicate_b32, 0,
                                                                   step_within_batch);
                    svint8_t a_interleaved_vector_i8 = svreinterpret_s8_s32(a_column_i32);

                    nk_i8_t const *bv = b_vecs + (column_tile_index * depth_step_count + ds) * vector_elements;
                    svint8_t b_packed_vector_i8 = svld1_s8(column_predicate_b8, bv);

                    svmopa_za32_s8_m(1, row_predicate_b8, column_predicate_b8, a_interleaved_vector_i8,
                                     b_packed_vector_i8);
                }
            }

            for (nk_size_t row = 0; row < rows_remaining; row++) {
                svst1_hor_za32(1, row, column_predicate_b32, c + (row_start + row) * c_stride_elements + col_start);
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
                for (nk_size_t col = 0; col < tile_dimension; col++) {
                    nk_size_t const col_abs = (column_tile_index + 0) * tile_dimension + col;
                    if (col_abs < n_vectors) {
                        nk_f32_t const *b_row = (nk_f32_t const *)(vectors + col_abs * stride_elements +
                                                                   depth_batch_start * expansion);
                        svld1_hor_za32(0, col, batch_predicate_b32, b_row);
                    }
                }
                for (nk_size_t step = 0; step < batch_size; step++) {
                    svint8_t a_interleaved = svreinterpret_s8_s32(svld1_s32(full_predicate_b32, a_buffer[step]));
                    svint32_t b_col_i32 = svread_ver_za32_s32_m(svdup_s32(0), full_predicate_b32, 0, step);
                    svint8_t b_interleaved = svreinterpret_s8_s32(b_col_i32);
                    svmopa_za32_s8_m(1, row_predicate_b8, full_predicate_b8, a_interleaved, b_interleaved);
                }

                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t col = 0; col < tile_dimension; col++) {
                    nk_size_t const col_abs = (column_tile_index + 1) * tile_dimension + col;
                    if (col_abs < n_vectors) {
                        nk_f32_t const *b_row = (nk_f32_t const *)(vectors + col_abs * stride_elements +
                                                                   depth_batch_start * expansion);
                        svld1_hor_za32(0, col, batch_predicate_b32, b_row);
                    }
                }
                for (nk_size_t step = 0; step < batch_size; step++) {
                    svint8_t a_interleaved = svreinterpret_s8_s32(svld1_s32(full_predicate_b32, a_buffer[step]));
                    svint32_t b_col_i32 = svread_ver_za32_s32_m(svdup_s32(0), full_predicate_b32, 0, step);
                    svint8_t b_interleaved = svreinterpret_s8_s32(b_col_i32);
                    svmopa_za32_s8_m(2, row_predicate_b8, full_predicate_b8, a_interleaved, b_interleaved);
                }

                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t col = 0; col < tile_dimension; col++) {
                    nk_size_t const col_abs = (column_tile_index + 2) * tile_dimension + col;
                    if (col_abs < n_vectors) {
                        nk_f32_t const *b_row = (nk_f32_t const *)(vectors + col_abs * stride_elements +
                                                                   depth_batch_start * expansion);
                        svld1_hor_za32(0, col, batch_predicate_b32, b_row);
                    }
                }
                for (nk_size_t step = 0; step < batch_size; step++) {
                    svint8_t a_interleaved = svreinterpret_s8_s32(svld1_s32(full_predicate_b32, a_buffer[step]));
                    svint32_t b_col_i32 = svread_ver_za32_s32_m(svdup_s32(0), full_predicate_b32, 0, step);
                    svint8_t b_interleaved = svreinterpret_s8_s32(b_col_i32);
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
            nk_size_t const col_tile_start = column_tile_index * tile_dimension;
            nk_size_t const cols_remaining = (col_tile_start + tile_dimension <= n_vectors)
                                                 ? tile_dimension
                                                 : (n_vectors - col_tile_start);
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
                for (nk_size_t col = 0; col < tile_dimension; col++) {
                    nk_size_t const col_abs = col_tile_start + col;
                    if (col_abs < n_vectors) {
                        nk_f32_t const *b_row = (nk_f32_t const *)(vectors + col_abs * stride_elements +
                                                                   depth_batch_start * expansion);
                        svld1_hor_za32(0, col, batch_predicate_b32, b_row);
                    }
                }
                for (nk_size_t step = 0; step < batch_size; step++) {
                    svint8_t a_interleaved = svreinterpret_s8_s32(svld1_s32(full_predicate_b32, a_buffer[step]));
                    svint32_t b_col_i32 = svread_ver_za32_s32_m(svdup_s32(0), column_predicate_b32, 0, step);
                    svint8_t b_interleaved = svreinterpret_s8_s32(b_col_i32);
                    svmopa_za32_s8_m(1, row_predicate_b8, column_predicate_b8, a_interleaved, b_interleaved);
                }
            }

            for (nk_size_t row = 0; row < rows_clamped; row++) {
                nk_size_t const row_abs = row_tile_start + row;
                svst1_hor_za32(1, row, column_predicate_b32,
                               (nk_f32_t *)(result + row_abs * result_stride_elements) + col_tile_start);
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
 *  Inline `e4m3` → `f16` conversion returning `svfloat16_t` for direct use in GEMM.
 *  This avoids memory round-trip when used inside a streaming kernel.
 *
 *  @param pg16   Predicate for 16-bit elements: use `svptrue_b16()`
 *  @param bytes  Pre-loaded 64 bytes: `svuint8_t` from `svld1_u8`
 *  @return       32 `f16` values as `svfloat16_t`: from lower 32 bytes
 */
NK_INTERNAL svfloat16_t nk_e4m3x_to_f16x_ssve_(svbool_t pg16, svuint8_t bytes) {
    svuint16_t vals = svunpklo_u16(bytes);

    svuint16_t sign = svlsl_n_u16_x(pg16, svand_n_u16_x(pg16, vals, 0x80), 8);
    svuint16_t mag = svand_n_u16_x(pg16, vals, 0x7F);
    svuint16_t mant = svand_n_u16_x(pg16, vals, 0x07);

    // Normal path: F16 = sign | ((mag << 7) + 0x2000)
    svuint16_t normal = svadd_n_u16_x(pg16, svlsl_n_u16_x(pg16, mag, 7), 0x2000);
    normal = svorr_u16_x(pg16, normal, sign);

    // Subnormal path: `mant` × (1/512) where 1/512 = 0x1800 in `f16`
    svfloat16_t mant_f16 = svcvt_f16_u16_x(pg16, mant);
    svfloat16_t scale = svreinterpret_f16_u16(svdup_n_u16(0x1800));
    svfloat16_t subnorm_abs = svmul_f16_x(pg16, mant_f16, scale);
    svuint16_t subnorm = svorr_u16_x(pg16, svreinterpret_u16_f16(subnorm_abs), sign);

    svbool_t is_subnorm = svcmpeq_n_u16(pg16, svand_n_u16_x(pg16, vals, 0x78), 0);
    svbool_t is_nan = svcmpeq_n_u16(pg16, mag, 0x7F);
    svuint16_t nan_val = svorr_n_u16_x(pg16, sign, 0x7E00);

    svuint16_t result = svsel_u16(is_subnorm, subnorm, normal);
    result = svsel_u16(is_nan, nan_val, result);

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
 *  @param pg16   Predicate for 16-bit elements (use svptrue_b16())
 *  @param bytes  Pre-loaded 64 bytes (svuint8_t from svld1_u8)
 *  @return       32 F16 values as svfloat16_t (from lower 32 bytes)
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

    nk_f16_t const *b_vecs = (nk_f16_t const *)((char const *)b_packed + sizeof(nk_dots_sme_packed_header_t));

    svbool_t const full_predicate_b16 = svptrue_b16();
    svbool_t const full_predicate_b8 = svptrue_b8();
    svbool_t const full_predicate_b32 = svptrue_b32();

    // Stack buffer for e4m3 -> f16 conversion (32 f16 per row, up to 16 rows)
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

                // Convert e4m3 -> f16 for each A row in this batch, then load into ZA0
                svzero_mask_za(nk_sme_zero_za32_tile_0_);

                svbool_t const batch_predicate_b32 = svwhilelt_b32((uint32_t)0, (uint32_t)batch_size);
                for (nk_size_t row_within_tile = 0; row_within_tile < rows_remaining; row_within_tile++) {
                    nk_size_t const a_row = row_start + row_within_tile;
                    // Load raw e4m3 bytes and convert to f16 using vectorized conversion
                    nk_e4m3_t const *a_src = a + a_row * a_stride_elements + depth_batch_start * expansion;
                    svuint8_t raw_bytes = svld1_u8(full_predicate_b8, (uint8_t const *)a_src);
                    svfloat16_t converted_f16 = nk_e4m3x_to_f16x_ssve_(full_predicate_b16, raw_bytes);
                    svst1_f16(full_predicate_b16, (float16_t *)a_converted_f16x32[row_within_tile], converted_f16);
                    // Load converted f16 row into ZA0 as f32 words
                    svld1_hor_za32(0, row_within_tile, batch_predicate_b32,
                                   (nk_f32_t const *)a_converted_f16x32[row_within_tile]);
                }

                for (nk_size_t step_within_batch = 0; step_within_batch < batch_size; step_within_batch++) {
                    nk_size_t const ds = depth_batch_start + step_within_batch;
                    svfloat32_t a_column_f32 = svread_ver_za32_f32_m(svdup_f32(0.0f), row_predicate_b32, 0,
                                                                     step_within_batch);
                    svfloat16_t a_interleaved_vector_f16 = svreinterpret_f16_f32(a_column_f32);

                    nk_f16_t const *bv0 = b_vecs + ((column_tile_index + 0) * depth_step_count + ds) * vector_elements;
                    nk_f16_t const *bv1 = b_vecs + ((column_tile_index + 1) * depth_step_count + ds) * vector_elements;
                    nk_f16_t const *bv2 = b_vecs + ((column_tile_index + 2) * depth_step_count + ds) * vector_elements;
                    svfloat16_t b_packed_vector_f16_0 = svld1_f16(full_predicate_b16, (float16_t const *)bv0);
                    svfloat16_t b_packed_vector_f16_1 = svld1_f16(full_predicate_b16, (float16_t const *)bv1);
                    svfloat16_t b_packed_vector_f16_2 = svld1_f16(full_predicate_b16, (float16_t const *)bv2);

                    svmopa_za32_f16_m(1, row_predicate_b16, full_predicate_b16, a_interleaved_vector_f16,
                                      b_packed_vector_f16_0);
                    svmopa_za32_f16_m(2, row_predicate_b16, full_predicate_b16, a_interleaved_vector_f16,
                                      b_packed_vector_f16_1);
                    svmopa_za32_f16_m(3, row_predicate_b16, full_predicate_b16, a_interleaved_vector_f16,
                                      b_packed_vector_f16_2);
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
            nk_size_t const col_start = column_tile_index * tile_dimension;
            nk_size_t const cols_remaining = (col_start + tile_dimension <= columns) ? tile_dimension
                                                                                     : (columns - col_start);
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
                for (nk_size_t row_within_tile = 0; row_within_tile < rows_remaining; row_within_tile++) {
                    nk_size_t const a_row = row_start + row_within_tile;
                    nk_e4m3_t const *a_src = a + a_row * a_stride_elements + depth_batch_start * expansion;
                    svuint8_t raw_bytes = svld1_u8(full_predicate_b8, (uint8_t const *)a_src);
                    svfloat16_t converted_f16 = nk_e4m3x_to_f16x_ssve_(full_predicate_b16, raw_bytes);
                    svst1_f16(full_predicate_b16, (float16_t *)a_converted_f16x32[row_within_tile], converted_f16);
                    svld1_hor_za32(0, row_within_tile, batch_predicate_b32,
                                   (nk_f32_t const *)a_converted_f16x32[row_within_tile]);
                }

                for (nk_size_t step_within_batch = 0; step_within_batch < batch_size; step_within_batch++) {
                    nk_size_t const ds = depth_batch_start + step_within_batch;
                    svfloat32_t a_column_f32 = svread_ver_za32_f32_m(svdup_f32(0.0f), row_predicate_b32, 0,
                                                                     step_within_batch);
                    svfloat16_t a_interleaved_vector_f16 = svreinterpret_f16_f32(a_column_f32);

                    nk_f16_t const *bv = b_vecs + (column_tile_index * depth_step_count + ds) * vector_elements;
                    svfloat16_t b_packed_vector_f16 = svld1_f16(column_predicate_b16, (float16_t const *)bv);

                    svmopa_za32_f16_m(1, row_predicate_b16, column_predicate_b16, a_interleaved_vector_f16,
                                      b_packed_vector_f16);
                }
            }

            for (nk_size_t row = 0; row < rows_remaining; row++) {
                svst1_hor_za32(1, row, column_predicate_b32, c + (row_start + row) * c_stride_elements + col_start);
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
    for (nk_size_t col_tile = 0; col_tile < column_tile_count; col_tile++) {
        for (nk_size_t depth_step = 0; depth_step < depth_step_count; depth_step++) {
            nk_size_t const vec_index = col_tile * depth_step_count + depth_step;
            nk_f16_t *vec_output = tiles_ptr + vec_index * vector_elements;

            nk_size_t const b_row_start = col_tile * tile_dimension;
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
                    svfloat16_t converted_f16 = nk_e4m3x_to_f16x_ssve_(full_predicate_b16, raw_bytes);
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
                for (nk_size_t col = 0; col < tile_dimension; col++) {
                    nk_size_t const col_abs = (column_tile_index + 0) * tile_dimension + col;
                    if (col_abs < n_vectors) {
                        nk_e4m3_t const *src = &vectors[col_abs * stride_elements + depth_batch_start * expansion];
                        svuint8_t raw = svld1_u8(depth_predicate_b8, (uint8_t const *)src);
                        svfloat16_t cvt = nk_e4m3x_to_f16x_ssve_(depth_predicate_b16, raw);
                        svst1_f16(depth_predicate_b16, (float16_t *)b_bounce_f16x32, cvt);
                        svld1_hor_za32(0, col, batch_predicate_b32, (nk_f32_t const *)b_bounce_f16x32);
                    }
                }
                for (nk_size_t step = 0; step < batch_size; step++) {
                    svfloat16_t a_interleaved = svreinterpret_f16_f32(svld1_f32(full_predicate_b32, a_buffer[step]));
                    svfloat32_t b_col_f32 = svread_ver_za32_f32_m(svdup_f32(0.0f), full_predicate_b32, 0, step);
                    svfloat16_t b_interleaved = svreinterpret_f16_f32(b_col_f32);
                    svmopa_za32_f16_m(1, row_predicate_b16, full_predicate_b16, a_interleaved, b_interleaved);
                }

                // Load B column tile 1 into ZA0, vertical read + FMOPA into ZA2
                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t col = 0; col < tile_dimension; col++) {
                    nk_size_t const col_abs = (column_tile_index + 1) * tile_dimension + col;
                    if (col_abs < n_vectors) {
                        nk_e4m3_t const *src = &vectors[col_abs * stride_elements + depth_batch_start * expansion];
                        svuint8_t raw = svld1_u8(depth_predicate_b8, (uint8_t const *)src);
                        svfloat16_t cvt = nk_e4m3x_to_f16x_ssve_(depth_predicate_b16, raw);
                        svst1_f16(depth_predicate_b16, (float16_t *)b_bounce_f16x32, cvt);
                        svld1_hor_za32(0, col, batch_predicate_b32, (nk_f32_t const *)b_bounce_f16x32);
                    }
                }
                for (nk_size_t step = 0; step < batch_size; step++) {
                    svfloat16_t a_interleaved = svreinterpret_f16_f32(svld1_f32(full_predicate_b32, a_buffer[step]));
                    svfloat32_t b_col_f32 = svread_ver_za32_f32_m(svdup_f32(0.0f), full_predicate_b32, 0, step);
                    svfloat16_t b_interleaved = svreinterpret_f16_f32(b_col_f32);
                    svmopa_za32_f16_m(2, row_predicate_b16, full_predicate_b16, a_interleaved, b_interleaved);
                }

                // Load B column tile 2 into ZA0, vertical read + FMOPA into ZA3
                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t col = 0; col < tile_dimension; col++) {
                    nk_size_t const col_abs = (column_tile_index + 2) * tile_dimension + col;
                    if (col_abs < n_vectors) {
                        nk_e4m3_t const *src = &vectors[col_abs * stride_elements + depth_batch_start * expansion];
                        svuint8_t raw = svld1_u8(depth_predicate_b8, (uint8_t const *)src);
                        svfloat16_t cvt = nk_e4m3x_to_f16x_ssve_(depth_predicate_b16, raw);
                        svst1_f16(depth_predicate_b16, (float16_t *)b_bounce_f16x32, cvt);
                        svld1_hor_za32(0, col, batch_predicate_b32, (nk_f32_t const *)b_bounce_f16x32);
                    }
                }
                for (nk_size_t step = 0; step < batch_size; step++) {
                    svfloat16_t a_interleaved = svreinterpret_f16_f32(svld1_f32(full_predicate_b32, a_buffer[step]));
                    svfloat32_t b_col_f32 = svread_ver_za32_f32_m(svdup_f32(0.0f), full_predicate_b32, 0, step);
                    svfloat16_t b_interleaved = svreinterpret_f16_f32(b_col_f32);
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
            nk_size_t const col_tile_start = column_tile_index * tile_dimension;
            nk_size_t const cols_remaining = (col_tile_start + tile_dimension <= n_vectors)
                                                 ? tile_dimension
                                                 : (n_vectors - col_tile_start);
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
                    svfloat16_t converted_f16 = nk_e4m3x_to_f16x_ssve_(full_predicate_b16, raw_bytes);
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
                for (nk_size_t col = 0; col < tile_dimension; col++) {
                    nk_size_t const col_abs = col_tile_start + col;
                    if (col_abs < n_vectors) {
                        nk_e4m3_t const *src = &vectors[col_abs * stride_elements + depth_batch_start * expansion];
                        svuint8_t raw = svld1_u8(depth_predicate_b8, (uint8_t const *)src);
                        svfloat16_t cvt = nk_e4m3x_to_f16x_ssve_(depth_predicate_b16, raw);
                        svst1_f16(depth_predicate_b16, (float16_t *)b_bounce_f16x32, cvt);
                        svld1_hor_za32(0, col, batch_predicate_b32, (nk_f32_t const *)b_bounce_f16x32);
                    }
                }
                for (nk_size_t step = 0; step < batch_size; step++) {
                    svfloat16_t a_interleaved = svreinterpret_f16_f32(svld1_f32(full_predicate_b32, a_buffer[step]));
                    svfloat32_t b_col_f32 = svread_ver_za32_f32_m(svdup_f32(0.0f), column_predicate_b32, 0, step);
                    svfloat16_t b_interleaved = svreinterpret_f16_f32(b_col_f32);
                    svmopa_za32_f16_m(1, row_predicate_b16, column_predicate_b16, a_interleaved, b_interleaved);
                }
            }

            for (nk_size_t row = 0; row < rows_actual; row++) {
                nk_size_t const row_abs = row_tile_start + row;
                svst1_hor_za32(1, row, column_predicate_b32,
                               result + row_abs * result_stride_elements + col_tile_start);
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

    nk_f16_t const *b_vecs = (nk_f16_t const *)((char const *)b_packed + sizeof(nk_dots_sme_packed_header_t));

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
                for (nk_size_t row_within_tile = 0; row_within_tile < rows_remaining; row_within_tile++) {
                    nk_size_t const a_row = row_start + row_within_tile;
                    // Vectorized e5m2 -> f16 conversion (2 instructions)
                    nk_e5m2_t const *a_src = a + a_row * a_stride_elements + depth_batch_start * expansion;
                    svuint8_t raw_bytes = svld1_u8(full_predicate_b8, (uint8_t const *)a_src);
                    svfloat16_t converted_f16 = nk_e5m2x_to_f16x_ssve_(full_predicate_b16, raw_bytes);
                    svst1_f16(full_predicate_b16, (float16_t *)a_converted_f16x32[row_within_tile], converted_f16);
                    svld1_hor_za32(0, row_within_tile, batch_predicate_b32,
                                   (nk_f32_t const *)a_converted_f16x32[row_within_tile]);
                }

                for (nk_size_t step_within_batch = 0; step_within_batch < batch_size; step_within_batch++) {
                    nk_size_t const ds = depth_batch_start + step_within_batch;
                    svfloat32_t a_column_f32 = svread_ver_za32_f32_m(svdup_f32(0.0f), row_predicate_b32, 0,
                                                                     step_within_batch);
                    svfloat16_t a_interleaved_vector_f16 = svreinterpret_f16_f32(a_column_f32);

                    nk_f16_t const *bv0 = b_vecs + ((column_tile_index + 0) * depth_step_count + ds) * vector_elements;
                    nk_f16_t const *bv1 = b_vecs + ((column_tile_index + 1) * depth_step_count + ds) * vector_elements;
                    nk_f16_t const *bv2 = b_vecs + ((column_tile_index + 2) * depth_step_count + ds) * vector_elements;
                    svfloat16_t b_packed_vector_f16_0 = svld1_f16(full_predicate_b16, (float16_t const *)bv0);
                    svfloat16_t b_packed_vector_f16_1 = svld1_f16(full_predicate_b16, (float16_t const *)bv1);
                    svfloat16_t b_packed_vector_f16_2 = svld1_f16(full_predicate_b16, (float16_t const *)bv2);

                    svmopa_za32_f16_m(1, row_predicate_b16, full_predicate_b16, a_interleaved_vector_f16,
                                      b_packed_vector_f16_0);
                    svmopa_za32_f16_m(2, row_predicate_b16, full_predicate_b16, a_interleaved_vector_f16,
                                      b_packed_vector_f16_1);
                    svmopa_za32_f16_m(3, row_predicate_b16, full_predicate_b16, a_interleaved_vector_f16,
                                      b_packed_vector_f16_2);
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
            nk_size_t const col_start = column_tile_index * tile_dimension;
            nk_size_t const cols_remaining = (col_start + tile_dimension <= columns) ? tile_dimension
                                                                                     : (columns - col_start);
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
                for (nk_size_t row_within_tile = 0; row_within_tile < rows_remaining; row_within_tile++) {
                    nk_size_t const a_row = row_start + row_within_tile;
                    nk_e5m2_t const *a_src = a + a_row * a_stride_elements + depth_batch_start * expansion;
                    svuint8_t raw_bytes = svld1_u8(full_predicate_b8, (uint8_t const *)a_src);
                    svfloat16_t converted_f16 = nk_e5m2x_to_f16x_ssve_(full_predicate_b16, raw_bytes);
                    svst1_f16(full_predicate_b16, (float16_t *)a_converted_f16x32[row_within_tile], converted_f16);
                    svld1_hor_za32(0, row_within_tile, batch_predicate_b32,
                                   (nk_f32_t const *)a_converted_f16x32[row_within_tile]);
                }

                for (nk_size_t step_within_batch = 0; step_within_batch < batch_size; step_within_batch++) {
                    nk_size_t const ds = depth_batch_start + step_within_batch;
                    svfloat32_t a_column_f32 = svread_ver_za32_f32_m(svdup_f32(0.0f), row_predicate_b32, 0,
                                                                     step_within_batch);
                    svfloat16_t a_interleaved_vector_f16 = svreinterpret_f16_f32(a_column_f32);

                    nk_f16_t const *bv = b_vecs + (column_tile_index * depth_step_count + ds) * vector_elements;
                    svfloat16_t b_packed_vector_f16 = svld1_f16(column_predicate_b16, (float16_t const *)bv);

                    svmopa_za32_f16_m(1, row_predicate_b16, column_predicate_b16, a_interleaved_vector_f16,
                                      b_packed_vector_f16);
                }
            }

            for (nk_size_t row = 0; row < rows_remaining; row++) {
                svst1_hor_za32(1, row, column_predicate_b32, c + (row_start + row) * c_stride_elements + col_start);
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
    for (nk_size_t col_tile = 0; col_tile < column_tile_count; col_tile++) {
        for (nk_size_t depth_step = 0; depth_step < depth_step_count; depth_step++) {
            nk_size_t const vec_index = col_tile * depth_step_count + depth_step;
            nk_f16_t *vec_output = tiles_ptr + vec_index * vector_elements;

            nk_size_t const b_row_start = col_tile * tile_dimension;
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
                for (nk_size_t col = 0; col < tile_dimension; col++) {
                    nk_size_t const col_abs = (column_tile_index + 0) * tile_dimension + col;
                    if (col_abs < n_vectors) {
                        nk_e5m2_t const *src = &vectors[col_abs * stride_elements + depth_batch_start * expansion];
                        svuint8_t raw = svld1_u8(depth_predicate_b8, (uint8_t const *)src);
                        svfloat16_t cvt = nk_e5m2x_to_f16x_ssve_(depth_predicate_b16, raw);
                        svst1_f16(depth_predicate_b16, (float16_t *)b_bounce_f16x32, cvt);
                        svld1_hor_za32(0, col, batch_predicate_b32, (nk_f32_t const *)b_bounce_f16x32);
                    }
                }
                for (nk_size_t step = 0; step < batch_size; step++) {
                    svfloat16_t a_interleaved = svreinterpret_f16_f32(svld1_f32(full_predicate_b32, a_buffer[step]));
                    svfloat32_t b_col_f32 = svread_ver_za32_f32_m(svdup_f32(0.0f), full_predicate_b32, 0, step);
                    svfloat16_t b_interleaved = svreinterpret_f16_f32(b_col_f32);
                    svmopa_za32_f16_m(1, row_predicate_b16, full_predicate_b16, a_interleaved, b_interleaved);
                }

                // Load B column tile 1 into ZA0, vertical read + FMOPA into ZA2
                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t col = 0; col < tile_dimension; col++) {
                    nk_size_t const col_abs = (column_tile_index + 1) * tile_dimension + col;
                    if (col_abs < n_vectors) {
                        nk_e5m2_t const *src = &vectors[col_abs * stride_elements + depth_batch_start * expansion];
                        svuint8_t raw = svld1_u8(depth_predicate_b8, (uint8_t const *)src);
                        svfloat16_t cvt = nk_e5m2x_to_f16x_ssve_(depth_predicate_b16, raw);
                        svst1_f16(depth_predicate_b16, (float16_t *)b_bounce_f16x32, cvt);
                        svld1_hor_za32(0, col, batch_predicate_b32, (nk_f32_t const *)b_bounce_f16x32);
                    }
                }
                for (nk_size_t step = 0; step < batch_size; step++) {
                    svfloat16_t a_interleaved = svreinterpret_f16_f32(svld1_f32(full_predicate_b32, a_buffer[step]));
                    svfloat32_t b_col_f32 = svread_ver_za32_f32_m(svdup_f32(0.0f), full_predicate_b32, 0, step);
                    svfloat16_t b_interleaved = svreinterpret_f16_f32(b_col_f32);
                    svmopa_za32_f16_m(2, row_predicate_b16, full_predicate_b16, a_interleaved, b_interleaved);
                }

                // Load B column tile 2 into ZA0, vertical read + FMOPA into ZA3
                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t col = 0; col < tile_dimension; col++) {
                    nk_size_t const col_abs = (column_tile_index + 2) * tile_dimension + col;
                    if (col_abs < n_vectors) {
                        nk_e5m2_t const *src = &vectors[col_abs * stride_elements + depth_batch_start * expansion];
                        svuint8_t raw = svld1_u8(depth_predicate_b8, (uint8_t const *)src);
                        svfloat16_t cvt = nk_e5m2x_to_f16x_ssve_(depth_predicate_b16, raw);
                        svst1_f16(depth_predicate_b16, (float16_t *)b_bounce_f16x32, cvt);
                        svld1_hor_za32(0, col, batch_predicate_b32, (nk_f32_t const *)b_bounce_f16x32);
                    }
                }
                for (nk_size_t step = 0; step < batch_size; step++) {
                    svfloat16_t a_interleaved = svreinterpret_f16_f32(svld1_f32(full_predicate_b32, a_buffer[step]));
                    svfloat32_t b_col_f32 = svread_ver_za32_f32_m(svdup_f32(0.0f), full_predicate_b32, 0, step);
                    svfloat16_t b_interleaved = svreinterpret_f16_f32(b_col_f32);
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
            nk_size_t const col_tile_start = column_tile_index * tile_dimension;
            nk_size_t const cols_remaining = (col_tile_start + tile_dimension <= n_vectors)
                                                 ? tile_dimension
                                                 : (n_vectors - col_tile_start);
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
                for (nk_size_t col = 0; col < tile_dimension; col++) {
                    nk_size_t const col_abs = col_tile_start + col;
                    if (col_abs < n_vectors) {
                        nk_e5m2_t const *src = &vectors[col_abs * stride_elements + depth_batch_start * expansion];
                        svuint8_t raw = svld1_u8(depth_predicate_b8, (uint8_t const *)src);
                        svfloat16_t cvt = nk_e5m2x_to_f16x_ssve_(depth_predicate_b16, raw);
                        svst1_f16(depth_predicate_b16, (float16_t *)b_bounce_f16x32, cvt);
                        svld1_hor_za32(0, col, batch_predicate_b32, (nk_f32_t const *)b_bounce_f16x32);
                    }
                }
                for (nk_size_t step = 0; step < batch_size; step++) {
                    svfloat16_t a_interleaved = svreinterpret_f16_f32(svld1_f32(full_predicate_b32, a_buffer[step]));
                    svfloat32_t b_col_f32 = svread_ver_za32_f32_m(svdup_f32(0.0f), column_predicate_b32, 0, step);
                    svfloat16_t b_interleaved = svreinterpret_f16_f32(b_col_f32);
                    svmopa_za32_f16_m(1, row_predicate_b16, column_predicate_b16, a_interleaved, b_interleaved);
                }
            }

            for (nk_size_t row = 0; row < rows_actual; row++) {
                nk_size_t const row_abs = row_tile_start + row;
                svst1_hor_za32(1, row, column_predicate_b32,
                               result + row_abs * result_stride_elements + col_tile_start);
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

    // Interleaved packing: packed_vec[expansion * d + k] = B[col_tile*tile_dim + d, depth_step*expansion + k]
    for (nk_size_t col_tile = 0; col_tile < column_tile_count; col_tile++) {
        for (nk_size_t depth_step = 0; depth_step < depth_step_count; depth_step++) {
            nk_size_t const vec_index = col_tile * depth_step_count + depth_step;
            nk_u8_t *vec_output = tiles_ptr + vec_index * vector_elements;

            nk_size_t const b_row_start = col_tile * tile_dimension;
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

    nk_u8_t const *b_vecs = (nk_u8_t const *)((char const *)b_packed + sizeof(nk_dots_sme_packed_header_t));

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
                for (nk_size_t row_within_tile = 0; row_within_tile < rows_remaining; row_within_tile++) {
                    nk_size_t const a_row = row_start + row_within_tile;
                    nk_f32_t const *a_row_ptr = (nk_f32_t const *)(a + a_row * a_stride_elements +
                                                                   depth_batch_start * expansion);
                    svld1_hor_za32(0, row_within_tile, batch_predicate_b32, a_row_ptr);
                }

                for (nk_size_t step_within_batch = 0; step_within_batch < batch_size; step_within_batch++) {
                    nk_size_t const ds = depth_batch_start + step_within_batch;
                    svuint32_t a_column_u32 = svread_ver_za32_u32_m(svdup_u32(0), row_predicate_b32, 0,
                                                                    step_within_batch);
                    svuint8_t a_interleaved_vector_u8 = svreinterpret_u8_u32(a_column_u32);

                    nk_u8_t const *bv0 = b_vecs + ((column_tile_index + 0) * depth_step_count + ds) * vector_elements;
                    nk_u8_t const *bv1 = b_vecs + ((column_tile_index + 1) * depth_step_count + ds) * vector_elements;
                    nk_u8_t const *bv2 = b_vecs + ((column_tile_index + 2) * depth_step_count + ds) * vector_elements;
                    svuint8_t b_packed_vector_u8_0 = svld1_u8(full_predicate_b8, bv0);
                    svuint8_t b_packed_vector_u8_1 = svld1_u8(full_predicate_b8, bv1);
                    svuint8_t b_packed_vector_u8_2 = svld1_u8(full_predicate_b8, bv2);

                    svmopa_za32_u8_m(1, row_predicate_b8, full_predicate_b8, a_interleaved_vector_u8,
                                     b_packed_vector_u8_0);
                    svmopa_za32_u8_m(2, row_predicate_b8, full_predicate_b8, a_interleaved_vector_u8,
                                     b_packed_vector_u8_1);
                    svmopa_za32_u8_m(3, row_predicate_b8, full_predicate_b8, a_interleaved_vector_u8,
                                     b_packed_vector_u8_2);
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
            nk_size_t const col_start = column_tile_index * tile_dimension;
            nk_size_t const cols_remaining = (col_start + tile_dimension <= columns) ? tile_dimension
                                                                                     : (columns - col_start);
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
                for (nk_size_t row_within_tile = 0; row_within_tile < rows_remaining; row_within_tile++) {
                    nk_size_t const a_row = row_start + row_within_tile;
                    nk_f32_t const *a_row_ptr = (nk_f32_t const *)(a + a_row * a_stride_elements +
                                                                   depth_batch_start * expansion);
                    svld1_hor_za32(0, row_within_tile, batch_predicate_b32, a_row_ptr);
                }

                for (nk_size_t step_within_batch = 0; step_within_batch < batch_size; step_within_batch++) {
                    nk_size_t const ds = depth_batch_start + step_within_batch;
                    svuint32_t a_column_u32 = svread_ver_za32_u32_m(svdup_u32(0), row_predicate_b32, 0,
                                                                    step_within_batch);
                    svuint8_t a_interleaved_vector_u8 = svreinterpret_u8_u32(a_column_u32);

                    nk_u8_t const *bv = b_vecs + (column_tile_index * depth_step_count + ds) * vector_elements;
                    svuint8_t b_packed_vector_u8 = svld1_u8(column_predicate_b8, bv);

                    svmopa_za32_u8_m(1, row_predicate_b8, column_predicate_b8, a_interleaved_vector_u8,
                                     b_packed_vector_u8);
                }
            }

            for (nk_size_t row = 0; row < rows_remaining; row++) {
                svst1_hor_za32(1, row, column_predicate_b32,
                               (nk_i32_t *)(c + (row_start + row) * c_stride_elements + col_start));
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
                for (nk_size_t col = 0; col < tile_dimension; col++) {
                    nk_size_t const col_abs = (column_tile_index + 0) * tile_dimension + col;
                    if (col_abs < n_vectors) {
                        nk_f32_t const *b_row = (nk_f32_t const *)(vectors + col_abs * stride_elements +
                                                                   depth_batch_start * expansion);
                        svld1_hor_za32(0, col, batch_predicate_b32, b_row);
                    }
                }
                for (nk_size_t step = 0; step < batch_size; step++) {
                    svuint8_t a_interleaved = svreinterpret_u8_u32(svld1_u32(full_predicate_b32, a_buffer[step]));
                    svuint32_t b_col_u32 = svread_ver_za32_u32_m(svdup_u32(0), full_predicate_b32, 0, step);
                    svuint8_t b_interleaved = svreinterpret_u8_u32(b_col_u32);
                    svmopa_za32_u8_m(1, row_predicate_b8, full_predicate_b8, a_interleaved, b_interleaved);
                }

                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t col = 0; col < tile_dimension; col++) {
                    nk_size_t const col_abs = (column_tile_index + 1) * tile_dimension + col;
                    if (col_abs < n_vectors) {
                        nk_f32_t const *b_row = (nk_f32_t const *)(vectors + col_abs * stride_elements +
                                                                   depth_batch_start * expansion);
                        svld1_hor_za32(0, col, batch_predicate_b32, b_row);
                    }
                }
                for (nk_size_t step = 0; step < batch_size; step++) {
                    svuint8_t a_interleaved = svreinterpret_u8_u32(svld1_u32(full_predicate_b32, a_buffer[step]));
                    svuint32_t b_col_u32 = svread_ver_za32_u32_m(svdup_u32(0), full_predicate_b32, 0, step);
                    svuint8_t b_interleaved = svreinterpret_u8_u32(b_col_u32);
                    svmopa_za32_u8_m(2, row_predicate_b8, full_predicate_b8, a_interleaved, b_interleaved);
                }

                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t col = 0; col < tile_dimension; col++) {
                    nk_size_t const col_abs = (column_tile_index + 2) * tile_dimension + col;
                    if (col_abs < n_vectors) {
                        nk_f32_t const *b_row = (nk_f32_t const *)(vectors + col_abs * stride_elements +
                                                                   depth_batch_start * expansion);
                        svld1_hor_za32(0, col, batch_predicate_b32, b_row);
                    }
                }
                for (nk_size_t step = 0; step < batch_size; step++) {
                    svuint8_t a_interleaved = svreinterpret_u8_u32(svld1_u32(full_predicate_b32, a_buffer[step]));
                    svuint32_t b_col_u32 = svread_ver_za32_u32_m(svdup_u32(0), full_predicate_b32, 0, step);
                    svuint8_t b_interleaved = svreinterpret_u8_u32(b_col_u32);
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
            nk_size_t const col_tile_start = column_tile_index * tile_dimension;
            nk_size_t const cols_remaining = (col_tile_start + tile_dimension <= n_vectors)
                                                 ? tile_dimension
                                                 : (n_vectors - col_tile_start);
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
                for (nk_size_t col = 0; col < tile_dimension; col++) {
                    nk_size_t const col_abs = col_tile_start + col;
                    if (col_abs < n_vectors) {
                        nk_f32_t const *b_row = (nk_f32_t const *)(vectors + col_abs * stride_elements +
                                                                   depth_batch_start * expansion);
                        svld1_hor_za32(0, col, batch_predicate_b32, b_row);
                    }
                }
                for (nk_size_t step = 0; step < batch_size; step++) {
                    svuint8_t a_interleaved = svreinterpret_u8_u32(svld1_u32(full_predicate_b32, a_buffer[step]));
                    svuint32_t b_col_u32 = svread_ver_za32_u32_m(svdup_u32(0), column_predicate_b32, 0, step);
                    svuint8_t b_interleaved = svreinterpret_u8_u32(b_col_u32);
                    svmopa_za32_u8_m(1, row_predicate_b8, column_predicate_b8, a_interleaved, b_interleaved);
                }
            }

            for (nk_size_t row = 0; row < rows_clamped; row++) {
                nk_size_t const row_abs = row_tile_start + row;
                svst1_hor_za32(1, row, column_predicate_b32,
                               (nk_f32_t *)(result + row_abs * result_stride_elements) + col_tile_start);
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
 *  4-bit integer GEMM (u4, i4) using nibble→byte unpacking + SMOPA/UMOPA.
 *
 *  Each byte packs two 4-bit values (nk_u4x2_t / nk_i4x2_t).
 *  Unpacking: extract low/high nibbles via AND/LSR (unsigned) or LSL+ASR (signed).
 *  Pack functions unpack to i8/u8 at pack time; kernels reuse the i8/u8 MOPA path.
 *  Symmetric kernels unpack on-the-fly via bounce buffer before ZA0 load.
 */

#pragma region Nibble Integers

NK_PUBLIC nk_size_t nk_dots_packed_size_u4_sme(nk_size_t n, nk_size_t k) { return nk_dots_packed_size_u8_sme(n, k); }

NK_PUBLIC void nk_dots_pack_u4_sme(               //
    nk_u4x2_t const *b, nk_size_t n, nk_size_t k, //
    nk_size_t b_stride, void *b_packed) {

    nk_size_t const expansion = 4;
    nk_size_t const tile_dimension = svcntsw();
    nk_size_t const vector_elements = svcntsb();
    nk_size_t const b_stride_elements = b_stride / sizeof(nk_u4x2_t);

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

    for (nk_size_t col_tile = 0; col_tile < column_tile_count; col_tile++) {
        for (nk_size_t depth_step = 0; depth_step < depth_step_count; depth_step++) {
            nk_size_t const vec_index = col_tile * depth_step_count + depth_step;
            nk_u8_t *vec_output = tiles_ptr + vec_index * vector_elements;

            nk_size_t const b_row_start = col_tile * tile_dimension;
            nk_size_t const depth_base = depth_step * expansion;
            nk_size_t const rows_to_pack = (b_row_start + tile_dimension <= n) ? tile_dimension : (n - b_row_start);

            for (nk_size_t column_in_tile = 0; column_in_tile < rows_to_pack; column_in_tile++) {
                for (nk_size_t sub_element = 0; sub_element < expansion; sub_element++) {
                    nk_size_t const depth_idx = depth_base + sub_element;
                    if (depth_idx < k) {
                        nk_size_t const byte_idx = (b_row_start + column_in_tile) * b_stride_elements + depth_idx / 2;
                        nk_u8_t byte_val = ((nk_u8_t const *)b)[byte_idx];
                        nk_u8_t nibble = (depth_idx & 1) ? (byte_val >> 4) : (byte_val & 0x0F);
                        vec_output[expansion * column_in_tile + sub_element] = nibble;
                    }
                }
            }
        }
    }
}

/**
 *  `u4` × `u4` → `u32` packed GEMM kernel using SME UMOPA with nibble→u8 unpacking.
 *  A input is nibble-packed — unpacked on-the-fly via bounce buffer before ZA0 load.
 *  B input is pre-packed u8 from nk_dots_pack_u4_sme — loaded directly.
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
    nk_size_t const depth_steps_per_batch = tile_dimension;
    nk_size_t const packed_depth = nk_size_divide_round_up_(depth, 2);

    nk_u8_t const *b_vecs = (nk_u8_t const *)((char const *)b_packed + sizeof(nk_dots_sme_packed_header_t));

    svbool_t const full_predicate_b8 = svptrue_b8();
    svbool_t const full_predicate_b32 = svptrue_b32();

    NK_ALIGN64 nk_u8_t bounce[64];

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
                nk_size_t const nibble_bytes = batch_size * 2;
                nk_size_t const nibble_offset = depth_batch_start * 2;

                for (nk_size_t row_within_tile = 0; row_within_tile < rows_remaining; row_within_tile++) {
                    nk_size_t const a_row = row_start + row_within_tile;
                    nk_u8_t const *src = (nk_u8_t const *)(a + a_row * a_stride_elements) + nibble_offset;
                    nk_size_t const avail = (nibble_offset < packed_depth) ? (packed_depth - nibble_offset) : 0;
                    nk_size_t const load_bytes = (nibble_bytes < avail) ? nibble_bytes : avail;
                    svbool_t const nibble_pred = svwhilelt_b8((uint32_t)0, (uint32_t)load_bytes);
                    for (nk_size_t i = 0; i < 64; i++) bounce[i] = 0;
                    svuint8_t packed = svld1_u8(nibble_pred, src);
                    svuint8_t low = svand_n_u8_x(nibble_pred, packed, 0x0F);
                    svuint8_t high = svlsr_n_u8_x(nibble_pred, packed, 4);
                    svuint8_t unpacked = svzip1_u8(low, high);
                    svst1_u8(full_predicate_b8, bounce, unpacked);
                    svld1_hor_za32(0, row_within_tile, batch_predicate_b32, (nk_f32_t const *)bounce);
                }

                for (nk_size_t step_within_batch = 0; step_within_batch < batch_size; step_within_batch++) {
                    nk_size_t const ds = depth_batch_start + step_within_batch;
                    svuint32_t a_column_u32 = svread_ver_za32_u32_m(svdup_u32(0), row_predicate_b32, 0,
                                                                    step_within_batch);
                    svuint8_t a_interleaved_vector_u8 = svreinterpret_u8_u32(a_column_u32);

                    nk_u8_t const *bv0 = b_vecs + ((column_tile_index + 0) * depth_step_count + ds) * vector_elements;
                    nk_u8_t const *bv1 = b_vecs + ((column_tile_index + 1) * depth_step_count + ds) * vector_elements;
                    nk_u8_t const *bv2 = b_vecs + ((column_tile_index + 2) * depth_step_count + ds) * vector_elements;

                    svmopa_za32_u8_m(1, row_predicate_b8, full_predicate_b8, a_interleaved_vector_u8,
                                     svld1_u8(full_predicate_b8, bv0));
                    svmopa_za32_u8_m(2, row_predicate_b8, full_predicate_b8, a_interleaved_vector_u8,
                                     svld1_u8(full_predicate_b8, bv1));
                    svmopa_za32_u8_m(3, row_predicate_b8, full_predicate_b8, a_interleaved_vector_u8,
                                     svld1_u8(full_predicate_b8, bv2));
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
            nk_size_t const col_start = column_tile_index * tile_dimension;
            nk_size_t const cols_remaining = (col_start + tile_dimension <= columns) ? tile_dimension
                                                                                     : (columns - col_start);
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
                nk_size_t const nibble_bytes = batch_size * 2;
                nk_size_t const nibble_offset = depth_batch_start * 2;

                for (nk_size_t row_within_tile = 0; row_within_tile < rows_remaining; row_within_tile++) {
                    nk_size_t const a_row = row_start + row_within_tile;
                    nk_u8_t const *src = (nk_u8_t const *)(a + a_row * a_stride_elements) + nibble_offset;
                    nk_size_t const avail = (nibble_offset < packed_depth) ? (packed_depth - nibble_offset) : 0;
                    nk_size_t const load_bytes = (nibble_bytes < avail) ? nibble_bytes : avail;
                    svbool_t const nibble_pred = svwhilelt_b8((uint32_t)0, (uint32_t)load_bytes);
                    for (nk_size_t i = 0; i < 64; i++) bounce[i] = 0;
                    svuint8_t packed = svld1_u8(nibble_pred, src);
                    svuint8_t low = svand_n_u8_x(nibble_pred, packed, 0x0F);
                    svuint8_t high = svlsr_n_u8_x(nibble_pred, packed, 4);
                    svuint8_t unpacked = svzip1_u8(low, high);
                    svst1_u8(full_predicate_b8, bounce, unpacked);
                    svld1_hor_za32(0, row_within_tile, batch_predicate_b32, (nk_f32_t const *)bounce);
                }

                for (nk_size_t step_within_batch = 0; step_within_batch < batch_size; step_within_batch++) {
                    nk_size_t const ds = depth_batch_start + step_within_batch;
                    svuint32_t a_column_u32 = svread_ver_za32_u32_m(svdup_u32(0), row_predicate_b32, 0,
                                                                    step_within_batch);
                    svuint8_t a_interleaved_vector_u8 = svreinterpret_u8_u32(a_column_u32);

                    nk_u8_t const *bv = b_vecs + (column_tile_index * depth_step_count + ds) * vector_elements;
                    svmopa_za32_u8_m(1, row_predicate_b8, column_predicate_b8, a_interleaved_vector_u8,
                                     svld1_u8(column_predicate_b8, bv));
                }
            }

            for (nk_size_t row = 0; row < rows_remaining; row++) {
                svst1_hor_za32(1, row, column_predicate_b32,
                               (nk_f32_t *)(c + (row_start + row) * c_stride_elements + col_start));
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

NK_PUBLIC nk_size_t nk_dots_packed_size_i4_sme(nk_size_t n, nk_size_t k) { return nk_dots_packed_size_i8_sme(n, k); }

NK_PUBLIC void nk_dots_pack_i4_sme(               //
    nk_i4x2_t const *b, nk_size_t n, nk_size_t k, //
    nk_size_t b_stride, void *b_packed) {

    nk_size_t const expansion = 4;
    nk_size_t const tile_dimension = svcntsw();
    nk_size_t const vector_elements = svcntsb();
    nk_size_t const b_stride_elements = b_stride / sizeof(nk_i4x2_t);

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

    for (nk_size_t i = 0; i < total_vectors * vector_elements; i++) tiles_ptr[i] = 0;

    for (nk_size_t col_tile = 0; col_tile < column_tile_count; col_tile++) {
        for (nk_size_t depth_step = 0; depth_step < depth_step_count; depth_step++) {
            nk_size_t const vec_index = col_tile * depth_step_count + depth_step;
            nk_i8_t *vec_output = tiles_ptr + vec_index * vector_elements;

            nk_size_t const b_row_start = col_tile * tile_dimension;
            nk_size_t const depth_base = depth_step * expansion;
            nk_size_t const rows_to_pack = (b_row_start + tile_dimension <= n) ? tile_dimension : (n - b_row_start);

            for (nk_size_t column_in_tile = 0; column_in_tile < rows_to_pack; column_in_tile++) {
                for (nk_size_t sub_element = 0; sub_element < expansion; sub_element++) {
                    nk_size_t const depth_idx = depth_base + sub_element;
                    if (depth_idx < k) {
                        nk_size_t const byte_idx = (b_row_start + column_in_tile) * b_stride_elements + depth_idx / 2;
                        nk_u8_t byte_val = ((nk_u8_t const *)b)[byte_idx];
                        nk_u8_t nibble = (depth_idx & 1) ? (byte_val >> 4) : (byte_val & 0x0F);
                        // Sign extend: treat nibble as signed 4-bit value
                        vec_output[expansion * column_in_tile + sub_element] = (nk_i8_t)((nibble ^ 8) - 8);
                    }
                }
            }
        }
    }
}

/**
 *  `i4` × `i4` → `i32` packed GEMM kernel using SME SMOPA with nibble→i8 unpacking.
 *  A input is nibble-packed — unpacked on-the-fly via bounce buffer before ZA0 load.
 *  B input is pre-packed i8 from nk_dots_pack_i4_sme — loaded directly.
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
    nk_size_t const depth_steps_per_batch = tile_dimension;
    nk_size_t const packed_depth = nk_size_divide_round_up_(depth, 2);

    nk_i8_t const *b_vecs = (nk_i8_t const *)((char const *)b_packed + sizeof(nk_dots_sme_packed_header_t));

    svbool_t const full_predicate_b8 = svptrue_b8();
    svbool_t const full_predicate_b32 = svptrue_b32();

    NK_ALIGN64 nk_i8_t bounce[64];

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
                nk_size_t const nibble_bytes = batch_size * 2;
                nk_size_t const nibble_offset = depth_batch_start * 2;

                for (nk_size_t row_within_tile = 0; row_within_tile < rows_remaining; row_within_tile++) {
                    nk_size_t const a_row = row_start + row_within_tile;
                    nk_i8_t const *src = (nk_i8_t const *)(a + a_row * a_stride_elements) + nibble_offset;
                    nk_size_t const avail = (nibble_offset < packed_depth) ? (packed_depth - nibble_offset) : 0;
                    nk_size_t const load_bytes = (nibble_bytes < avail) ? nibble_bytes : avail;
                    svbool_t const nibble_pred = svwhilelt_b8((uint32_t)0, (uint32_t)load_bytes);
                    for (nk_size_t i = 0; i < 64; i++) bounce[i] = 0;
                    svint8_t packed = svld1_s8(nibble_pred, src);
                    svint8_t low = svasr_n_s8_x(nibble_pred, svlsl_n_s8_x(nibble_pred, packed, 4), 4);
                    svint8_t high = svasr_n_s8_x(nibble_pred, packed, 4);
                    svint8_t unpacked = svzip1_s8(low, high);
                    svst1_s8(full_predicate_b8, bounce, unpacked);
                    svld1_hor_za32(0, row_within_tile, batch_predicate_b32, (nk_f32_t const *)bounce);
                }

                for (nk_size_t step_within_batch = 0; step_within_batch < batch_size; step_within_batch++) {
                    nk_size_t const ds = depth_batch_start + step_within_batch;
                    svint32_t a_column_i32 = svread_ver_za32_s32_m(svdup_s32(0), row_predicate_b32, 0,
                                                                   step_within_batch);
                    svint8_t a_interleaved_vector_i8 = svreinterpret_s8_s32(a_column_i32);

                    nk_i8_t const *bv0 = b_vecs + ((column_tile_index + 0) * depth_step_count + ds) * vector_elements;
                    nk_i8_t const *bv1 = b_vecs + ((column_tile_index + 1) * depth_step_count + ds) * vector_elements;
                    nk_i8_t const *bv2 = b_vecs + ((column_tile_index + 2) * depth_step_count + ds) * vector_elements;

                    svmopa_za32_s8_m(1, row_predicate_b8, full_predicate_b8, a_interleaved_vector_i8,
                                     svld1_s8(full_predicate_b8, bv0));
                    svmopa_za32_s8_m(2, row_predicate_b8, full_predicate_b8, a_interleaved_vector_i8,
                                     svld1_s8(full_predicate_b8, bv1));
                    svmopa_za32_s8_m(3, row_predicate_b8, full_predicate_b8, a_interleaved_vector_i8,
                                     svld1_s8(full_predicate_b8, bv2));
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
            nk_size_t const col_start = column_tile_index * tile_dimension;
            nk_size_t const cols_remaining = (col_start + tile_dimension <= columns) ? tile_dimension
                                                                                     : (columns - col_start);
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
                nk_size_t const nibble_bytes = batch_size * 2;
                nk_size_t const nibble_offset = depth_batch_start * 2;

                for (nk_size_t row_within_tile = 0; row_within_tile < rows_remaining; row_within_tile++) {
                    nk_size_t const a_row = row_start + row_within_tile;
                    nk_i8_t const *src = (nk_i8_t const *)(a + a_row * a_stride_elements) + nibble_offset;
                    nk_size_t const avail = (nibble_offset < packed_depth) ? (packed_depth - nibble_offset) : 0;
                    nk_size_t const load_bytes = (nibble_bytes < avail) ? nibble_bytes : avail;
                    svbool_t const nibble_pred = svwhilelt_b8((uint32_t)0, (uint32_t)load_bytes);
                    for (nk_size_t i = 0; i < 64; i++) bounce[i] = 0;
                    svint8_t packed = svld1_s8(nibble_pred, src);
                    svint8_t low = svasr_n_s8_x(nibble_pred, svlsl_n_s8_x(nibble_pred, packed, 4), 4);
                    svint8_t high = svasr_n_s8_x(nibble_pred, packed, 4);
                    svint8_t unpacked = svzip1_s8(low, high);
                    svst1_s8(full_predicate_b8, bounce, unpacked);
                    svld1_hor_za32(0, row_within_tile, batch_predicate_b32, (nk_f32_t const *)bounce);
                }

                for (nk_size_t step_within_batch = 0; step_within_batch < batch_size; step_within_batch++) {
                    nk_size_t const ds = depth_batch_start + step_within_batch;
                    svint32_t a_column_i32 = svread_ver_za32_s32_m(svdup_s32(0), row_predicate_b32, 0,
                                                                   step_within_batch);
                    svint8_t a_interleaved_vector_i8 = svreinterpret_s8_s32(a_column_i32);

                    nk_i8_t const *bv = b_vecs + (column_tile_index * depth_step_count + ds) * vector_elements;
                    svmopa_za32_s8_m(1, row_predicate_b8, column_predicate_b8, a_interleaved_vector_i8,
                                     svld1_s8(column_predicate_b8, bv));
                }
            }

            for (nk_size_t row = 0; row < rows_remaining; row++) {
                svst1_hor_za32(1, row, column_predicate_b32,
                               (nk_f32_t *)(c + (row_start + row) * c_stride_elements + col_start));
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
 *  `u4` × `u4` → `u32` symmetric kernel using SME UMOPA with nibble→byte unpacking.
 *  Unpacks 4-bit values to u8 via bounce buffer, then uses UMOPA u8→u32 outer products.
 *  ZA0 = staging tile, ZA1-ZA3 = accumulators (3-tile fast path).
 */
__arm_locally_streaming __arm_new("za") static void nk_dots_symmetric_u4_sme_kernel_(
    nk_u4x2_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride_elements, nk_u32_t *result,
    nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) {

    nk_size_t const expansion = 4;
    nk_size_t const tile_dimension = svcntw();
    nk_size_t const depth_step_count = nk_size_divide_round_up_(depth, expansion);
    nk_size_t const depth_steps_per_batch = tile_dimension;
    nk_size_t const packed_depth = nk_size_divide_round_up_(depth, 2);

    svbool_t const full_predicate_b8 = svptrue_b8();
    svbool_t const full_predicate_b32 = svptrue_b32();

    NK_ALIGN64 nk_u32_t a_buffer[16][16];
    NK_ALIGN64 nk_u8_t bounce[64];

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

                // Unpack A nibbles → u8 bounce → ZA0
                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                svbool_t const batch_predicate_b32 = svwhilelt_b32((uint32_t)0, (uint32_t)batch_size);
                nk_size_t const nibble_bytes = batch_size * 2;
                nk_size_t const nibble_offset = depth_batch_start * 2;

                for (nk_size_t row_in_tile = 0; row_in_tile < rows_clamped; row_in_tile++) {
                    nk_size_t const row_abs = row_tile_start + row_in_tile;
                    nk_u8_t const *src = (nk_u8_t const *)(vectors + row_abs * stride_elements) + nibble_offset;
                    nk_size_t const avail = (nibble_offset < packed_depth) ? (packed_depth - nibble_offset) : 0;
                    nk_size_t const load_bytes = (nibble_bytes < avail) ? nibble_bytes : avail;
                    svbool_t const nibble_pred = svwhilelt_b8((uint32_t)0, (uint32_t)load_bytes);
                    for (nk_size_t i = 0; i < 64; i++) bounce[i] = 0;
                    svuint8_t packed = svld1_u8(nibble_pred, src);
                    svuint8_t low = svand_n_u8_x(nibble_pred, packed, 0x0F);
                    svuint8_t high = svlsr_n_u8_x(nibble_pred, packed, 4);
                    svuint8_t unpacked = svzip1_u8(low, high);
                    svst1_u8(full_predicate_b8, bounce, unpacked);
                    svld1_hor_za32(0, row_in_tile, batch_predicate_b32, (nk_f32_t const *)bounce);
                }

                for (nk_size_t s = 0; s < batch_size; s++)
                    svst1_u32(full_predicate_b32, a_buffer[s],
                              svread_ver_za32_u32_m(svdup_u32(0), row_predicate_b32, 0, s));

                // B column tile 0
                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t col = 0; col < tile_dimension; col++) {
                    nk_size_t const col_abs = (column_tile_index + 0) * tile_dimension + col;
                    if (col_abs < n_vectors) {
                        nk_u8_t const *src = (nk_u8_t const *)(vectors + col_abs * stride_elements) + nibble_offset;
                        nk_size_t const avail = (nibble_offset < packed_depth) ? (packed_depth - nibble_offset) : 0;
                        nk_size_t const load_bytes = (nibble_bytes < avail) ? nibble_bytes : avail;
                        svbool_t const nibble_pred = svwhilelt_b8((uint32_t)0, (uint32_t)load_bytes);
                        for (nk_size_t i = 0; i < 64; i++) bounce[i] = 0;
                        svuint8_t packed = svld1_u8(nibble_pred, src);
                        svuint8_t low = svand_n_u8_x(nibble_pred, packed, 0x0F);
                        svuint8_t high = svlsr_n_u8_x(nibble_pred, packed, 4);
                        svuint8_t unpacked = svzip1_u8(low, high);
                        svst1_u8(full_predicate_b8, bounce, unpacked);
                        svld1_hor_za32(0, col, batch_predicate_b32, (nk_f32_t const *)bounce);
                    }
                }
                for (nk_size_t step = 0; step < batch_size; step++) {
                    svuint8_t a_interleaved = svreinterpret_u8_u32(svld1_u32(full_predicate_b32, a_buffer[step]));
                    svuint32_t b_col_u32 = svread_ver_za32_u32_m(svdup_u32(0), full_predicate_b32, 0, step);
                    svuint8_t b_interleaved = svreinterpret_u8_u32(b_col_u32);
                    svmopa_za32_u8_m(1, row_predicate_b8, full_predicate_b8, a_interleaved, b_interleaved);
                }

                // B column tile 1
                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t col = 0; col < tile_dimension; col++) {
                    nk_size_t const col_abs = (column_tile_index + 1) * tile_dimension + col;
                    if (col_abs < n_vectors) {
                        nk_u8_t const *src = (nk_u8_t const *)(vectors + col_abs * stride_elements) + nibble_offset;
                        nk_size_t const avail = (nibble_offset < packed_depth) ? (packed_depth - nibble_offset) : 0;
                        nk_size_t const load_bytes = (nibble_bytes < avail) ? nibble_bytes : avail;
                        svbool_t const nibble_pred = svwhilelt_b8((uint32_t)0, (uint32_t)load_bytes);
                        for (nk_size_t i = 0; i < 64; i++) bounce[i] = 0;
                        svuint8_t packed = svld1_u8(nibble_pred, src);
                        svuint8_t low = svand_n_u8_x(nibble_pred, packed, 0x0F);
                        svuint8_t high = svlsr_n_u8_x(nibble_pred, packed, 4);
                        svuint8_t unpacked = svzip1_u8(low, high);
                        svst1_u8(full_predicate_b8, bounce, unpacked);
                        svld1_hor_za32(0, col, batch_predicate_b32, (nk_f32_t const *)bounce);
                    }
                }
                for (nk_size_t step = 0; step < batch_size; step++) {
                    svuint8_t a_interleaved = svreinterpret_u8_u32(svld1_u32(full_predicate_b32, a_buffer[step]));
                    svuint32_t b_col_u32 = svread_ver_za32_u32_m(svdup_u32(0), full_predicate_b32, 0, step);
                    svuint8_t b_interleaved = svreinterpret_u8_u32(b_col_u32);
                    svmopa_za32_u8_m(2, row_predicate_b8, full_predicate_b8, a_interleaved, b_interleaved);
                }

                // B column tile 2
                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t col = 0; col < tile_dimension; col++) {
                    nk_size_t const col_abs = (column_tile_index + 2) * tile_dimension + col;
                    if (col_abs < n_vectors) {
                        nk_u8_t const *src = (nk_u8_t const *)(vectors + col_abs * stride_elements) + nibble_offset;
                        nk_size_t const avail = (nibble_offset < packed_depth) ? (packed_depth - nibble_offset) : 0;
                        nk_size_t const load_bytes = (nibble_bytes < avail) ? nibble_bytes : avail;
                        svbool_t const nibble_pred = svwhilelt_b8((uint32_t)0, (uint32_t)load_bytes);
                        for (nk_size_t i = 0; i < 64; i++) bounce[i] = 0;
                        svuint8_t packed = svld1_u8(nibble_pred, src);
                        svuint8_t low = svand_n_u8_x(nibble_pred, packed, 0x0F);
                        svuint8_t high = svlsr_n_u8_x(nibble_pred, packed, 4);
                        svuint8_t unpacked = svzip1_u8(low, high);
                        svst1_u8(full_predicate_b8, bounce, unpacked);
                        svld1_hor_za32(0, col, batch_predicate_b32, (nk_f32_t const *)bounce);
                    }
                }
                for (nk_size_t step = 0; step < batch_size; step++) {
                    svuint8_t a_interleaved = svreinterpret_u8_u32(svld1_u32(full_predicate_b32, a_buffer[step]));
                    svuint32_t b_col_u32 = svread_ver_za32_u32_m(svdup_u32(0), full_predicate_b32, 0, step);
                    svuint8_t b_interleaved = svreinterpret_u8_u32(b_col_u32);
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
            nk_size_t const col_tile_start = column_tile_index * tile_dimension;
            nk_size_t const cols_remaining = (col_tile_start + tile_dimension <= n_vectors)
                                                 ? tile_dimension
                                                 : (n_vectors - col_tile_start);
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
                nk_size_t const nibble_bytes = batch_size * 2;
                nk_size_t const nibble_offset = depth_batch_start * 2;

                for (nk_size_t row_in_tile = 0; row_in_tile < rows_clamped; row_in_tile++) {
                    nk_size_t const row_abs = row_tile_start + row_in_tile;
                    nk_u8_t const *src = (nk_u8_t const *)(vectors + row_abs * stride_elements) + nibble_offset;
                    nk_size_t const avail = (nibble_offset < packed_depth) ? (packed_depth - nibble_offset) : 0;
                    nk_size_t const load_bytes = (nibble_bytes < avail) ? nibble_bytes : avail;
                    svbool_t const nibble_pred = svwhilelt_b8((uint32_t)0, (uint32_t)load_bytes);
                    for (nk_size_t i = 0; i < 64; i++) bounce[i] = 0;
                    svuint8_t packed = svld1_u8(nibble_pred, src);
                    svuint8_t low = svand_n_u8_x(nibble_pred, packed, 0x0F);
                    svuint8_t high = svlsr_n_u8_x(nibble_pred, packed, 4);
                    svuint8_t unpacked = svzip1_u8(low, high);
                    svst1_u8(full_predicate_b8, bounce, unpacked);
                    svld1_hor_za32(0, row_in_tile, batch_predicate_b32, (nk_f32_t const *)bounce);
                }

                for (nk_size_t s = 0; s < batch_size; s++)
                    svst1_u32(full_predicate_b32, a_buffer[s],
                              svread_ver_za32_u32_m(svdup_u32(0), row_predicate_b32, 0, s));

                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t col = 0; col < tile_dimension; col++) {
                    nk_size_t const col_abs = col_tile_start + col;
                    if (col_abs < n_vectors) {
                        nk_u8_t const *src = (nk_u8_t const *)(vectors + col_abs * stride_elements) + nibble_offset;
                        nk_size_t const avail = (nibble_offset < packed_depth) ? (packed_depth - nibble_offset) : 0;
                        nk_size_t const load_bytes = (nibble_bytes < avail) ? nibble_bytes : avail;
                        svbool_t const nibble_pred = svwhilelt_b8((uint32_t)0, (uint32_t)load_bytes);
                        for (nk_size_t i = 0; i < 64; i++) bounce[i] = 0;
                        svuint8_t packed = svld1_u8(nibble_pred, src);
                        svuint8_t low = svand_n_u8_x(nibble_pred, packed, 0x0F);
                        svuint8_t high = svlsr_n_u8_x(nibble_pred, packed, 4);
                        svuint8_t unpacked = svzip1_u8(low, high);
                        svst1_u8(full_predicate_b8, bounce, unpacked);
                        svld1_hor_za32(0, col, batch_predicate_b32, (nk_f32_t const *)bounce);
                    }
                }
                for (nk_size_t step = 0; step < batch_size; step++) {
                    svuint8_t a_interleaved = svreinterpret_u8_u32(svld1_u32(full_predicate_b32, a_buffer[step]));
                    svuint32_t b_col_u32 = svread_ver_za32_u32_m(svdup_u32(0), column_predicate_b32, 0, step);
                    svuint8_t b_interleaved = svreinterpret_u8_u32(b_col_u32);
                    svmopa_za32_u8_m(1, row_predicate_b8, column_predicate_b8, a_interleaved, b_interleaved);
                }
            }

            for (nk_size_t row = 0; row < rows_clamped; row++) {
                nk_size_t const row_abs = row_tile_start + row;
                svst1_hor_za32(1, row, column_predicate_b32,
                               (nk_f32_t *)(result + row_abs * result_stride_elements) + col_tile_start);
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
 *  `i4` × `i4` → `i32` symmetric kernel using SME SMOPA with nibble→byte unpacking.
 *  Unpacks 4-bit values to i8 via sign-extending bounce buffer, then uses SMOPA i8→i32.
 *  ZA0 = staging tile, ZA1-ZA3 = accumulators (3-tile fast path).
 */
__arm_locally_streaming __arm_new("za") static void nk_dots_symmetric_i4_sme_kernel_(
    nk_i4x2_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride_elements, nk_i32_t *result,
    nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) {

    nk_size_t const expansion = 4;
    nk_size_t const tile_dimension = svcntw();
    nk_size_t const depth_step_count = nk_size_divide_round_up_(depth, expansion);
    nk_size_t const depth_steps_per_batch = tile_dimension;
    nk_size_t const packed_depth = nk_size_divide_round_up_(depth, 2);

    svbool_t const full_predicate_b8 = svptrue_b8();
    svbool_t const full_predicate_b32 = svptrue_b32();

    NK_ALIGN64 nk_i32_t a_buffer[16][16];
    NK_ALIGN64 nk_i8_t bounce[64];

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

                // Unpack A nibbles → i8 bounce → ZA0
                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                svbool_t const batch_predicate_b32 = svwhilelt_b32((uint32_t)0, (uint32_t)batch_size);
                nk_size_t const nibble_bytes = batch_size * 2;
                nk_size_t const nibble_offset = depth_batch_start * 2;

                for (nk_size_t row_in_tile = 0; row_in_tile < rows_clamped; row_in_tile++) {
                    nk_size_t const row_abs = row_tile_start + row_in_tile;
                    nk_i8_t const *src = (nk_i8_t const *)(vectors + row_abs * stride_elements) + nibble_offset;
                    nk_size_t const avail = (nibble_offset < packed_depth) ? (packed_depth - nibble_offset) : 0;
                    nk_size_t const load_bytes = (nibble_bytes < avail) ? nibble_bytes : avail;
                    svbool_t const nibble_pred = svwhilelt_b8((uint32_t)0, (uint32_t)load_bytes);
                    for (nk_size_t i = 0; i < 64; i++) bounce[i] = 0;
                    svint8_t packed = svld1_s8(nibble_pred, src);
                    svint8_t low = svasr_n_s8_x(nibble_pred, svlsl_n_s8_x(nibble_pred, packed, 4), 4);
                    svint8_t high = svasr_n_s8_x(nibble_pred, packed, 4);
                    svint8_t unpacked = svzip1_s8(low, high);
                    svst1_s8(full_predicate_b8, bounce, unpacked);
                    svld1_hor_za32(0, row_in_tile, batch_predicate_b32, (nk_f32_t const *)bounce);
                }

                for (nk_size_t s = 0; s < batch_size; s++)
                    svst1_s32(full_predicate_b32, a_buffer[s],
                              svread_ver_za32_s32_m(svdup_s32(0), row_predicate_b32, 0, s));

                // B column tile 0
                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t col = 0; col < tile_dimension; col++) {
                    nk_size_t const col_abs = (column_tile_index + 0) * tile_dimension + col;
                    if (col_abs < n_vectors) {
                        nk_i8_t const *src = (nk_i8_t const *)(vectors + col_abs * stride_elements) + nibble_offset;
                        nk_size_t const avail = (nibble_offset < packed_depth) ? (packed_depth - nibble_offset) : 0;
                        nk_size_t const load_bytes = (nibble_bytes < avail) ? nibble_bytes : avail;
                        svbool_t const nibble_pred = svwhilelt_b8((uint32_t)0, (uint32_t)load_bytes);
                        for (nk_size_t i = 0; i < 64; i++) bounce[i] = 0;
                        svint8_t packed = svld1_s8(nibble_pred, src);
                        svint8_t low = svasr_n_s8_x(nibble_pred, svlsl_n_s8_x(nibble_pred, packed, 4), 4);
                        svint8_t high = svasr_n_s8_x(nibble_pred, packed, 4);
                        svint8_t unpacked = svzip1_s8(low, high);
                        svst1_s8(full_predicate_b8, bounce, unpacked);
                        svld1_hor_za32(0, col, batch_predicate_b32, (nk_f32_t const *)bounce);
                    }
                }
                for (nk_size_t step = 0; step < batch_size; step++) {
                    svint8_t a_interleaved = svreinterpret_s8_s32(svld1_s32(full_predicate_b32, a_buffer[step]));
                    svint32_t b_col_i32 = svread_ver_za32_s32_m(svdup_s32(0), full_predicate_b32, 0, step);
                    svint8_t b_interleaved = svreinterpret_s8_s32(b_col_i32);
                    svmopa_za32_s8_m(1, row_predicate_b8, full_predicate_b8, a_interleaved, b_interleaved);
                }

                // B column tile 1
                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t col = 0; col < tile_dimension; col++) {
                    nk_size_t const col_abs = (column_tile_index + 1) * tile_dimension + col;
                    if (col_abs < n_vectors) {
                        nk_i8_t const *src = (nk_i8_t const *)(vectors + col_abs * stride_elements) + nibble_offset;
                        nk_size_t const avail = (nibble_offset < packed_depth) ? (packed_depth - nibble_offset) : 0;
                        nk_size_t const load_bytes = (nibble_bytes < avail) ? nibble_bytes : avail;
                        svbool_t const nibble_pred = svwhilelt_b8((uint32_t)0, (uint32_t)load_bytes);
                        for (nk_size_t i = 0; i < 64; i++) bounce[i] = 0;
                        svint8_t packed = svld1_s8(nibble_pred, src);
                        svint8_t low = svasr_n_s8_x(nibble_pred, svlsl_n_s8_x(nibble_pred, packed, 4), 4);
                        svint8_t high = svasr_n_s8_x(nibble_pred, packed, 4);
                        svint8_t unpacked = svzip1_s8(low, high);
                        svst1_s8(full_predicate_b8, bounce, unpacked);
                        svld1_hor_za32(0, col, batch_predicate_b32, (nk_f32_t const *)bounce);
                    }
                }
                for (nk_size_t step = 0; step < batch_size; step++) {
                    svint8_t a_interleaved = svreinterpret_s8_s32(svld1_s32(full_predicate_b32, a_buffer[step]));
                    svint32_t b_col_i32 = svread_ver_za32_s32_m(svdup_s32(0), full_predicate_b32, 0, step);
                    svint8_t b_interleaved = svreinterpret_s8_s32(b_col_i32);
                    svmopa_za32_s8_m(2, row_predicate_b8, full_predicate_b8, a_interleaved, b_interleaved);
                }

                // B column tile 2
                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t col = 0; col < tile_dimension; col++) {
                    nk_size_t const col_abs = (column_tile_index + 2) * tile_dimension + col;
                    if (col_abs < n_vectors) {
                        nk_i8_t const *src = (nk_i8_t const *)(vectors + col_abs * stride_elements) + nibble_offset;
                        nk_size_t const avail = (nibble_offset < packed_depth) ? (packed_depth - nibble_offset) : 0;
                        nk_size_t const load_bytes = (nibble_bytes < avail) ? nibble_bytes : avail;
                        svbool_t const nibble_pred = svwhilelt_b8((uint32_t)0, (uint32_t)load_bytes);
                        for (nk_size_t i = 0; i < 64; i++) bounce[i] = 0;
                        svint8_t packed = svld1_s8(nibble_pred, src);
                        svint8_t low = svasr_n_s8_x(nibble_pred, svlsl_n_s8_x(nibble_pred, packed, 4), 4);
                        svint8_t high = svasr_n_s8_x(nibble_pred, packed, 4);
                        svint8_t unpacked = svzip1_s8(low, high);
                        svst1_s8(full_predicate_b8, bounce, unpacked);
                        svld1_hor_za32(0, col, batch_predicate_b32, (nk_f32_t const *)bounce);
                    }
                }
                for (nk_size_t step = 0; step < batch_size; step++) {
                    svint8_t a_interleaved = svreinterpret_s8_s32(svld1_s32(full_predicate_b32, a_buffer[step]));
                    svint32_t b_col_i32 = svread_ver_za32_s32_m(svdup_s32(0), full_predicate_b32, 0, step);
                    svint8_t b_interleaved = svreinterpret_s8_s32(b_col_i32);
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
            nk_size_t const col_tile_start = column_tile_index * tile_dimension;
            nk_size_t const cols_remaining = (col_tile_start + tile_dimension <= n_vectors)
                                                 ? tile_dimension
                                                 : (n_vectors - col_tile_start);
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
                nk_size_t const nibble_bytes = batch_size * 2;
                nk_size_t const nibble_offset = depth_batch_start * 2;

                for (nk_size_t row_in_tile = 0; row_in_tile < rows_clamped; row_in_tile++) {
                    nk_size_t const row_abs = row_tile_start + row_in_tile;
                    nk_i8_t const *src = (nk_i8_t const *)(vectors + row_abs * stride_elements) + nibble_offset;
                    nk_size_t const avail = (nibble_offset < packed_depth) ? (packed_depth - nibble_offset) : 0;
                    nk_size_t const load_bytes = (nibble_bytes < avail) ? nibble_bytes : avail;
                    svbool_t const nibble_pred = svwhilelt_b8((uint32_t)0, (uint32_t)load_bytes);
                    for (nk_size_t i = 0; i < 64; i++) bounce[i] = 0;
                    svint8_t packed = svld1_s8(nibble_pred, src);
                    svint8_t low = svasr_n_s8_x(nibble_pred, svlsl_n_s8_x(nibble_pred, packed, 4), 4);
                    svint8_t high = svasr_n_s8_x(nibble_pred, packed, 4);
                    svint8_t unpacked = svzip1_s8(low, high);
                    svst1_s8(full_predicate_b8, bounce, unpacked);
                    svld1_hor_za32(0, row_in_tile, batch_predicate_b32, (nk_f32_t const *)bounce);
                }

                for (nk_size_t s = 0; s < batch_size; s++)
                    svst1_s32(full_predicate_b32, a_buffer[s],
                              svread_ver_za32_s32_m(svdup_s32(0), row_predicate_b32, 0, s));

                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t col = 0; col < tile_dimension; col++) {
                    nk_size_t const col_abs = col_tile_start + col;
                    if (col_abs < n_vectors) {
                        nk_i8_t const *src = (nk_i8_t const *)(vectors + col_abs * stride_elements) + nibble_offset;
                        nk_size_t const avail = (nibble_offset < packed_depth) ? (packed_depth - nibble_offset) : 0;
                        nk_size_t const load_bytes = (nibble_bytes < avail) ? nibble_bytes : avail;
                        svbool_t const nibble_pred = svwhilelt_b8((uint32_t)0, (uint32_t)load_bytes);
                        for (nk_size_t i = 0; i < 64; i++) bounce[i] = 0;
                        svint8_t packed = svld1_s8(nibble_pred, src);
                        svint8_t low = svasr_n_s8_x(nibble_pred, svlsl_n_s8_x(nibble_pred, packed, 4), 4);
                        svint8_t high = svasr_n_s8_x(nibble_pred, packed, 4);
                        svint8_t unpacked = svzip1_s8(low, high);
                        svst1_s8(full_predicate_b8, bounce, unpacked);
                        svld1_hor_za32(0, col, batch_predicate_b32, (nk_f32_t const *)bounce);
                    }
                }
                for (nk_size_t step = 0; step < batch_size; step++) {
                    svint8_t a_interleaved = svreinterpret_s8_s32(svld1_s32(full_predicate_b32, a_buffer[step]));
                    svint32_t b_col_i32 = svread_ver_za32_s32_m(svdup_s32(0), column_predicate_b32, 0, step);
                    svint8_t b_interleaved = svreinterpret_s8_s32(b_col_i32);
                    svmopa_za32_s8_m(1, row_predicate_b8, column_predicate_b8, a_interleaved, b_interleaved);
                }
            }

            for (nk_size_t row = 0; row < rows_clamped; row++) {
                nk_size_t const row_abs = row_tile_start + row;
                svst1_hor_za32(1, row, column_predicate_b32,
                               (nk_f32_t *)(result + row_abs * result_stride_elements) + col_tile_start);
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

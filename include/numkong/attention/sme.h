/**
 *  @brief FlashAttention-style kernels for SME.
 *  @file include/numkong/attention/sme.h
 *  @author Ash Vardanian
 *  @date January 11, 2026
 *
 *  @sa include/numkong/attention.h
 *
 *  This file implements FlashAttention-2 style scaled dot-product attention (SDPA) optimized
 *  for ARM SME instructions on Apple M4 and similar processors. The kernel computes:
 *
 *      O = softmax(Q × Kᵀ / √d) × V
 *
 *  Key features:
 *  - Online softmax: Mathematically exact, processes KV blocks incrementally
 *  - Pre-packed KV cache: Amortizes packing cost for repeated inference
 *  - GQA/MQA support: Different num_heads and num_kv_heads for grouped-query attention
 *  - Pure Streaming SVE: No NEON intrinsics for non-linear operations
 *
 *  Target models (2025):
 *  - Kimi K2:       head_dim=112, 64 heads, MHA, 128K context
 *  - LLaMA 3.1 405B: head_dim=128, 128 heads, 16 KV heads (GQA 8:1), 128K context
 *  - Qwen 2.5 72B:  head_dim=128, 64 heads, 8 KV heads (GQA 8:1), 32K context
 *
 *  SME tile dimensions (for SVL=512, i.e., Apple M4):
 *  - ZA32 tile: 16 x 16 f32 elements (1KB)
 *  - bf16/f16 vectors: 32 elements per SVE vector
 *
 *  Expected performance (Apple M4):
 *  - bf16 prefill (query_len=64, kv_len=4K): 400-500 GFLOPS
 *  - bf16 decode (query_len=1, kv_len=4K): 200-300 GFLOPS
 *
 *  Block sizes:
 *  - Bᵣ = 16 (query block rows, matches ZA32 tile height)
 *  - Bᶜ = 16 (KV block columns, fits 16×16 scores)
 */
#ifndef NK_ATTENTION_SME_H
#define NK_ATTENTION_SME_H

#if NK_TARGET_ARM_
#if NK_TARGET_SME

#include "numkong/types.h"

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
 *  @brief Convert bf16 vector to f32 in registers (streaming SVE compatible).
 *
 *  BF16 is the upper 16 bits of F32, so we:
 *  1. Reinterpret bf16 as u16
 *  2. Zero-extend to u32 (unpklo for lower half)
 *  3. Shift left by 16 to place in f32 exponent+mantissa position
 *  4. Reinterpret as f32
 */
NK_INTERNAL svfloat32_t nk_bf16_to_f32_sve_(svbool_t pg, svbfloat16_t x) __arm_streaming {
    svuint16_t u16 = svreinterpret_u16_bf16(x);
    svuint32_t u32 = svunpklo_u32(u16);
    u32 = svlsl_n_u32_x(pg, u32, 16);
    return svreinterpret_f32_u32(u32);
}

/**
 *  @brief Convert f32 vector to bf16 in registers with rounding (streaming SVE compatible).
 *
 *  1. Reinterpret f32 as u32
 *  2. Add rounding bias (0x8000) for round-to-nearest
 *  3. Shift right by 16
 *  4. Narrow to u16 and reinterpret as bf16
 */
NK_INTERNAL svbfloat16_t nk_f32_to_bf16_sve_(svbool_t pg, svfloat32_t x) __arm_streaming {
    svuint32_t u32 = svreinterpret_u32_f32(x);
    u32 = svadd_n_u32_x(pg, u32, 0x8000); // Round to nearest
    u32 = svlsr_n_u32_x(pg, u32, 16);
    svuint16_t u16 = svuzp1_u16(svreinterpret_u16_u32(u32), svreinterpret_u16_u32(u32));
    return svreinterpret_bf16_u16(u16);
}

/**
 *  @brief Packed KV cache header for attention (64-byte aligned).
 *
 *  Layout in memory:
 *  [header: 64 bytes][K tiles: variable][V tiles: variable]
 */
typedef struct {
    nk_u32_t num_kv_heads;    ///< Number of K/V heads (for GQA, may differ from Q heads)
    nk_u32_t head_dim;        ///< Original head dimension (64, 112, 128)
    nk_u32_t head_dim_padded; ///< Padded to multiple of 32 for SME
    nk_u32_t seq_len;         ///< Current sequence length
    nk_u32_t max_seq_len;     ///< Maximum sequence length (for pre-allocation)
    nk_u32_t k_offset;        ///< Byte offset to K data from header start
    nk_u32_t v_offset;        ///< Byte offset to V data from header start
    nk_u32_t reserved[9];     ///< Pad to 64 bytes
} nk_attention_sme_kv_packed_header_t;

/**
 *  @brief Fast exp approximation in Streaming SVE.
 *
 *  Uses Cody-Waite range reduction + Horner polynomial (degree 4).
 *  Accuracy: ~0.1% relative error, acceptable for softmax normalization.
 *
 *  @param pg Active predicate
 *  @param x Input vector
 *  @return exp(x) approximation
 */
NK_INTERNAL svfloat32_t nk_exp_f32_sve_(svbool_t pg, svfloat32_t x) __arm_streaming {
    // Constants for Cody-Waite range reduction
    svfloat32_t log2e = svdup_f32(1.4426950408889634f);
    svfloat32_t ln2_hi = svdup_f32(0.693145751953125f);
    svfloat32_t ln2_lo = svdup_f32(1.42860682030941723212e-6f);

    // Clamp to avoid overflow/underflow
    svfloat32_t max_x = svdup_f32(88.3762626647949f);
    svfloat32_t min_x = svdup_f32(-87.3365447504021f);
    x = svmax_f32_m(pg, svmin_f32_m(pg, x, max_x), min_x);

    // n = round(x / ln(2))
    svfloat32_t n = svrintn_f32_m(svundef_f32(), pg, svmul_f32_m(pg, x, log2e));

    // r = x - n × ln(2) using Cody-Waite for precision
    svfloat32_t r = svmsb_f32_m(pg, n, ln2_hi, x);
    r = svmsb_f32_m(pg, n, ln2_lo, r);

    // Polynomial approximation for exp(r): degree 4
    // exp(r) ≈ 1 + r + r²/2 + r³/6 + r⁴/24
    svfloat32_t p = svdup_f32(4.1666666667e-2f);            // 1/24
    p = svmad_f32_m(pg, p, r, svdup_f32(1.6666666667e-1f)); // 1/6
    p = svmad_f32_m(pg, p, r, svdup_f32(5.0000000000e-1f)); // 1/2
    p = svmad_f32_m(pg, p, r, svdup_f32(1.0f));             // 1
    p = svmad_f32_m(pg, p, r, svdup_f32(1.0f));             // 1

    // Reconstruct: exp(x) = 2ⁿ × exp(r)
    // 2ⁿ via IEEE 754 exponent manipulation
    svint32_t ni = svcvt_s32_f32_m(svundef_s32(), pg, n);
    ni = svadd_s32_m(pg, ni, svdup_s32(127));
    ni = svlsl_n_s32_m(pg, ni, 23);
    svfloat32_t pow2n = svreinterpret_f32_s32(ni);

    return svmul_f32_m(pg, p, pow2n);
}

/* Horizontal f32 sum (streaming SVE). */
NK_INTERNAL nk_f32_t nk_reduce_add_f32_sve_(svbool_t pg, svfloat32_t x) __arm_streaming { return svaddv_f32(pg, x); }

/* Horizontal f32 max (streaming SVE). */
NK_INTERNAL nk_f32_t nk_reduce_max_f32_sve_(svbool_t pg, svfloat32_t x) __arm_streaming { return svmaxv_f32(pg, x); }

NK_PUBLIC nk_size_t nk_attention_packed_kv_size_bf16_sme(nk_size_t num_kv_heads, nk_size_t head_dim,
                                                         nk_size_t max_seq_len) {
    // Pad head_dim to multiple of 32 for SME
    nk_size_t head_dim_padded = nk_size_round_up_to_multiple_(head_dim, 32);

    // K and V each: [num_kv_heads, seq_len, head_dim_padded] in bf16
    nk_size_t kv_size = num_kv_heads * max_seq_len * head_dim_padded * sizeof(nk_bf16_t);

    // Header + K + V
    return sizeof(nk_attention_sme_kv_packed_header_t) + 2 * kv_size;
}

NK_PUBLIC nk_size_t nk_attention_packed_kv_size_f16_sme(nk_size_t num_kv_heads, nk_size_t head_dim,
                                                        nk_size_t max_seq_len) {
    return nk_attention_packed_kv_size_bf16_sme(num_kv_heads, head_dim, max_seq_len);
}

NK_PUBLIC void nk_attention_pack_kv_bf16_sme(nk_bf16_t const *k, nk_bf16_t const *v, nk_size_t num_kv_heads,
                                             nk_size_t head_dim, nk_size_t seq_len, nk_size_t k_stride,
                                             nk_size_t v_stride, void *kv_packed) {

    nk_attention_sme_kv_packed_header_t *header = (nk_attention_sme_kv_packed_header_t *)kv_packed;
    nk_size_t head_dim_padded = nk_size_round_up_to_multiple_(head_dim, 32);

    // Initialize header
    header->num_kv_heads = (nk_u32_t)num_kv_heads;
    header->head_dim = (nk_u32_t)head_dim;
    header->head_dim_padded = (nk_u32_t)head_dim_padded;
    header->seq_len = (nk_u32_t)seq_len;
    header->k_offset = sizeof(nk_attention_sme_kv_packed_header_t);

    nk_size_t kv_head_size = seq_len * head_dim_padded * sizeof(nk_bf16_t);
    header->v_offset = header->k_offset + (nk_u32_t)(num_kv_heads * kv_head_size);

    nk_bf16_t *k_packed = (nk_bf16_t *)((char *)kv_packed + header->k_offset);
    nk_bf16_t *v_packed = (nk_bf16_t *)((char *)kv_packed + header->v_offset);

    // Pack K and V: copy with padding
    for (nk_size_t h = 0; h < num_kv_heads; h++) {
        nk_bf16_t const *k_head = k + h * k_stride;
        nk_bf16_t const *v_head = v + h * v_stride;
        nk_bf16_t *k_out = k_packed + h * seq_len * head_dim_padded;
        nk_bf16_t *v_out = v_packed + h * seq_len * head_dim_padded;

        for (nk_size_t s = 0; s < seq_len; s++) {
            // Copy valid head_dim elements
            for (nk_size_t d = 0; d < head_dim; d++) {
                k_out[s * head_dim_padded + d] = k_head[s * head_dim + d];
                v_out[s * head_dim_padded + d] = v_head[s * head_dim + d];
            }
            // Zero padding using predicated SVE stores
            nk_size_t pad_len = head_dim_padded - head_dim;
            if (pad_len > 0) {
                svbfloat16_t zeros_bf16 = svdup_bf16(0);
                for (nk_size_t d = head_dim; d < head_dim_padded; d += svcnth()) {
                    svbool_t pg = svwhilelt_b16((nk_u32_t)(d - head_dim), (nk_u32_t)pad_len);
                    svst1_bf16(pg, (bfloat16_t *)(k_out + s * head_dim_padded + d), zeros_bf16);
                    svst1_bf16(pg, (bfloat16_t *)(v_out + s * head_dim_padded + d), zeros_bf16);
                }
            }
        }
    }
}

NK_PUBLIC void nk_attention_pack_kv_f16_sme(nk_f16_t const *k, nk_f16_t const *v, nk_size_t num_kv_heads,
                                            nk_size_t head_dim, nk_size_t seq_len, nk_size_t k_stride,
                                            nk_size_t v_stride, void *kv_packed) {

    nk_attention_sme_kv_packed_header_t *header = (nk_attention_sme_kv_packed_header_t *)kv_packed;
    nk_size_t head_dim_padded = nk_size_round_up_to_multiple_(head_dim, 32);

    header->num_kv_heads = (nk_u32_t)num_kv_heads;
    header->head_dim = (nk_u32_t)head_dim;
    header->head_dim_padded = (nk_u32_t)head_dim_padded;
    header->seq_len = (nk_u32_t)seq_len;
    header->k_offset = sizeof(nk_attention_sme_kv_packed_header_t);

    nk_size_t kv_head_size = seq_len * head_dim_padded * sizeof(nk_f16_t);
    header->v_offset = header->k_offset + (nk_u32_t)(num_kv_heads * kv_head_size);

    nk_f16_t *k_packed = (nk_f16_t *)((char *)kv_packed + header->k_offset);
    nk_f16_t *v_packed = (nk_f16_t *)((char *)kv_packed + header->v_offset);

    for (nk_size_t h = 0; h < num_kv_heads; h++) {
        nk_f16_t const *k_head = k + h * k_stride;
        nk_f16_t const *v_head = v + h * v_stride;
        nk_f16_t *k_out = k_packed + h * seq_len * head_dim_padded;
        nk_f16_t *v_out = v_packed + h * seq_len * head_dim_padded;

        for (nk_size_t s = 0; s < seq_len; s++) {
            for (nk_size_t d = 0; d < head_dim; d++) {
                k_out[s * head_dim_padded + d] = k_head[s * head_dim + d];
                v_out[s * head_dim_padded + d] = v_head[s * head_dim + d];
            }
            // Zero padding using predicated SVE stores
            nk_size_t pad_len = head_dim_padded - head_dim;
            if (pad_len > 0) {
                svfloat16_t zeros_f16 = svdup_f16(0);
                for (nk_size_t d = head_dim; d < head_dim_padded; d += svcnth()) {
                    svbool_t pg = svwhilelt_b16((nk_u32_t)(d - head_dim), (nk_u32_t)pad_len);
                    svst1_f16(pg, (float16_t *)(k_out + s * head_dim_padded + d), zeros_f16);
                    svst1_f16(pg, (float16_t *)(v_out + s * head_dim_padded + d), zeros_f16);
                }
            }
        }
    }
}

/**
 *  @brief Internal: bf16 attention kernel - OPTIMIZED version.
 *
 *  FlashAttention-2 algorithm with optimizations:
 *  - State (row_max, row_sum) kept in SVE registers instead of arrays
 *  - Vectorized initialization of o_acc array
 *  - FMLA-based dot products for Q×Kᵀ (proper vectorization)
 *  - Batched exp calls for softmax
 *  - svdup_lane for weight broadcasting (no store-load cycle)
 *  - Predicated SVE loops for unified body/tail handling
 */
__arm_locally_streaming __arm_new("za") static void nk_attention_bf16_sme_kernel_(
    nk_bf16_t const *q, // [query_len, head_dim]
    nk_bf16_t const *k, // [kv_len, head_dim_padded]
    nk_bf16_t const *v, // [kv_len, head_dim_padded]
    nk_bf16_t *output,  // [query_len, head_dim]
    nk_size_t query_len, nk_size_t kv_len, nk_size_t head_dim, nk_size_t head_dim_padded, nk_f32_t scale) {

    svbool_t const ptrue_s = svptrue_b32();
    nk_size_t const Bc = 16;
    nk_size_t const valid_q = (query_len < 16) ? query_len : 16;

    // State in SVE registers instead of arrays.
    svfloat32_t row_max_v = svdup_f32(NK_F32_MIN); // All 16 row maxes
    svfloat32_t row_sum_v = svdup_f32(0.0f);       // All 16 row sums

    // Output accumulator - still needs array but with vectorized init
    NK_ALIGN64 nk_f32_t o_acc[16 * 256];

    //  Vectorized init instead of scalar loop.
    svfloat32_t zero_v = svdup_f32(0.0f);
    for (nk_size_t i = 0; i < 16 * head_dim_padded; i += svcntw()) { svst1_f32(ptrue_s, o_acc + i, zero_v); }

    // Temporary for scores - stored in ZA tile 0
    NK_ALIGN64 nk_f32_t scores[16][16];

    // Process KV in blocks of Bc=16
    for (nk_size_t kv_start = 0; kv_start < kv_len; kv_start += Bc) {
        nk_size_t const valid_kv = ((kv_start + Bc) <= kv_len) ? Bc : (kv_len - kv_start);

        // Q×Kᵀ using FMLA with 8× ki unroll.
        // Reuses Q row across 8 ki values, reducing Q load overhead
        // For Bc=16, this gives exactly 2 iterations per qi
        for (nk_size_t qi = 0; qi < valid_q; qi++) {
            nk_size_t ki = 0;

            // Unrolled loop: process 8 ki at a time
            for (; ki + 8 <= valid_kv; ki += 8) {
                svfloat32_t dot0 = svdup_f32(0.0f), dot1 = svdup_f32(0.0f);
                svfloat32_t dot2 = svdup_f32(0.0f), dot3 = svdup_f32(0.0f);
                svfloat32_t dot4 = svdup_f32(0.0f), dot5 = svdup_f32(0.0f);
                svfloat32_t dot6 = svdup_f32(0.0f), dot7 = svdup_f32(0.0f);

                for (nk_size_t d = 0; d < head_dim; d += svcntw()) {
                    svbool_t pg = svwhilelt_b32((nk_u32_t)d, (nk_u32_t)head_dim);

                    // Load Q once, convert to f32
                    svbfloat16_t q_bf16 = svld1_bf16(pg, (bfloat16_t const *)(q + qi * head_dim + d));
                    svfloat32_t q_f32 = nk_bf16_to_f32_sve_(pg, q_bf16);

                    // Load 8 K rows and accumulate
                    svbfloat16_t k0 = svld1_bf16(pg,
                                                 (bfloat16_t const *)(k + (kv_start + ki + 0) * head_dim_padded + d));
                    svbfloat16_t k1 = svld1_bf16(pg,
                                                 (bfloat16_t const *)(k + (kv_start + ki + 1) * head_dim_padded + d));
                    svbfloat16_t k2 = svld1_bf16(pg,
                                                 (bfloat16_t const *)(k + (kv_start + ki + 2) * head_dim_padded + d));
                    svbfloat16_t k3 = svld1_bf16(pg,
                                                 (bfloat16_t const *)(k + (kv_start + ki + 3) * head_dim_padded + d));
                    svbfloat16_t k4 = svld1_bf16(pg,
                                                 (bfloat16_t const *)(k + (kv_start + ki + 4) * head_dim_padded + d));
                    svbfloat16_t k5 = svld1_bf16(pg,
                                                 (bfloat16_t const *)(k + (kv_start + ki + 5) * head_dim_padded + d));
                    svbfloat16_t k6 = svld1_bf16(pg,
                                                 (bfloat16_t const *)(k + (kv_start + ki + 6) * head_dim_padded + d));
                    svbfloat16_t k7 = svld1_bf16(pg,
                                                 (bfloat16_t const *)(k + (kv_start + ki + 7) * head_dim_padded + d));

                    dot0 = svmla_f32_x(pg, dot0, q_f32, nk_bf16_to_f32_sve_(pg, k0));
                    dot1 = svmla_f32_x(pg, dot1, q_f32, nk_bf16_to_f32_sve_(pg, k1));
                    dot2 = svmla_f32_x(pg, dot2, q_f32, nk_bf16_to_f32_sve_(pg, k2));
                    dot3 = svmla_f32_x(pg, dot3, q_f32, nk_bf16_to_f32_sve_(pg, k3));
                    dot4 = svmla_f32_x(pg, dot4, q_f32, nk_bf16_to_f32_sve_(pg, k4));
                    dot5 = svmla_f32_x(pg, dot5, q_f32, nk_bf16_to_f32_sve_(pg, k5));
                    dot6 = svmla_f32_x(pg, dot6, q_f32, nk_bf16_to_f32_sve_(pg, k6));
                    dot7 = svmla_f32_x(pg, dot7, q_f32, nk_bf16_to_f32_sve_(pg, k7));
                }

                // Reduce all 8, scale, and batch store using SVE
                NK_ALIGN64 nk_f32_t temp_scores[8];
                temp_scores[0] = svaddv_f32(ptrue_s, dot0);
                temp_scores[1] = svaddv_f32(ptrue_s, dot1);
                temp_scores[2] = svaddv_f32(ptrue_s, dot2);
                temp_scores[3] = svaddv_f32(ptrue_s, dot3);
                temp_scores[4] = svaddv_f32(ptrue_s, dot4);
                temp_scores[5] = svaddv_f32(ptrue_s, dot5);
                temp_scores[6] = svaddv_f32(ptrue_s, dot6);
                temp_scores[7] = svaddv_f32(ptrue_s, dot7);

                // Vectorized scale and store (8 elements = partial SVE vector for SVL=512)
                svbool_t pg8 = svwhilelt_b32((nk_u32_t)0, (nk_u32_t)8);
                svfloat32_t scores_vec = svld1_f32(pg8, temp_scores);
                scores_vec = svmul_f32_x(pg8, scores_vec, svdup_f32(scale));
                svst1_f32(pg8, &scores[qi][ki], scores_vec);
            }

            // Handle remaining ki values (0-7 remaining)
            for (; ki < valid_kv; ki++) {
                svfloat32_t dot = svdup_f32(0.0f);
                for (nk_size_t d = 0; d < head_dim; d += svcntw()) {
                    svbool_t pg = svwhilelt_b32((nk_u32_t)d, (nk_u32_t)head_dim);
                    svbfloat16_t q_bf16 = svld1_bf16(pg, (bfloat16_t const *)(q + qi * head_dim + d));
                    svbfloat16_t k_bf16 = svld1_bf16(pg,
                                                     (bfloat16_t const *)(k + (kv_start + ki) * head_dim_padded + d));
                    svfloat32_t q_f32 = nk_bf16_to_f32_sve_(pg, q_bf16);
                    svfloat32_t k_f32 = nk_bf16_to_f32_sve_(pg, k_bf16);
                    dot = svmla_f32_x(pg, dot, q_f32, k_f32);
                }
                scores[qi][ki] = svaddv_f32(ptrue_s, dot) * scale;
            }
        }

        // Compute block maxes for all rows at once.
        NK_ALIGN64 nk_f32_t block_max_arr[16];
        for (nk_size_t qi = 0; qi < valid_q; qi++) {
            svfloat32_t row_scores = svld1_f32(ptrue_s, scores[qi]);
            block_max_arr[qi] = svmaxv_f32(ptrue_s, row_scores);
        }
        svfloat32_t block_max_v = svld1_f32(ptrue_s, block_max_arr);

        // Compute new max: element-wise max of row_max_v and block_max_v
        svfloat32_t new_max_v = svmax_f32_x(ptrue_s, row_max_v, block_max_v);

        // Batch correction factors (one exp call).
        svfloat32_t correction_v = nk_exp_f32_sve_(ptrue_s, svsub_f32_x(ptrue_s, row_max_v, new_max_v));

        // Update row_sum_v with corrections (all 16 at once)
        row_sum_v = svmul_f32_x(ptrue_s, row_sum_v, correction_v);

        // Store corrections for o_acc rescaling
        NK_ALIGN64 nk_f32_t corrections[16];
        svst1_f32(ptrue_s, corrections, correction_v);

        // Rescale o_acc for each query row
        for (nk_size_t qi = 0; qi < valid_q; qi++) {
            nk_f32_t corr = corrections[qi];
            svfloat32_t corr_vec = svdup_f32(corr);
            for (nk_size_t d = 0; d < head_dim; d += svcntw()) {
                svbool_t pg = svwhilelt_b32((nk_u32_t)d, (nk_u32_t)head_dim);
                svfloat32_t o = svld1_f32(pg, o_acc + qi * head_dim_padded + d);
                o = svmul_f32_x(pg, o, corr_vec);
                svst1_f32(pg, o_acc + qi * head_dim_padded + d, o);
            }
        }

        //  Compute softmax weights (16 exp calls, each vectorized).
        NK_ALIGN64 nk_f32_t new_max_arr[16];
        svst1_f32(ptrue_s, new_max_arr, new_max_v);

        NK_ALIGN64 nk_f32_t all_weights[16][16];
        NK_ALIGN64 nk_f32_t row_sum_deltas[16];

        for (nk_size_t qi = 0; qi < valid_q; qi++) {
            svfloat32_t row_scores = svld1_f32(ptrue_s, scores[qi]);
            svfloat32_t max_broadcast = svdup_f32(new_max_arr[qi]);
            svfloat32_t weights = nk_exp_f32_sve_(ptrue_s, svsub_f32_x(ptrue_s, row_scores, max_broadcast));

            // Quantize weights to bf16 precision
            svbfloat16_t w_bf16 = nk_f32_to_bf16_sve_(ptrue_s, weights);
            weights = nk_bf16_to_f32_sve_(ptrue_s, w_bf16);

            svst1_f32(ptrue_s, all_weights[qi], weights);
            row_sum_deltas[qi] = svaddv_f32(ptrue_s, weights);
        }

        // Update row_sum_v with deltas
        svfloat32_t deltas_v = svld1_f32(ptrue_s, row_sum_deltas);
        row_sum_v = svadd_f32_x(ptrue_s, row_sum_v, deltas_v);

        // Update row_max_v
        row_max_v = new_max_v;

        // nd 6: P×V with specialized decode path.
        if (valid_q == 1) {
            // Decode mode: keep accumulators in registers, reduce memory traffic
            // Process all ki for each depth chunk, accumulate in register
            for (nk_size_t d = 0; d < head_dim; d += svcntw()) {
                svbool_t pg = svwhilelt_b32((nk_u32_t)d, (nk_u32_t)head_dim);
                svfloat32_t acc = svld1_f32(pg, o_acc + d); // Load once

                for (nk_size_t ki = 0; ki < valid_kv; ki++) {
                    svbfloat16_t v_bf16 = svld1_bf16(pg,
                                                     (bfloat16_t const *)(v + (kv_start + ki) * head_dim_padded + d));
                    svfloat32_t v_f32 = nk_bf16_to_f32_sve_(pg, v_bf16);
                    svfloat32_t w_vec = svdup_f32(all_weights[0][ki]);
                    acc = svmla_f32_x(pg, acc, w_vec, v_f32);
                }

                svst1_f32(pg, o_acc + d, acc); // Store once
            }
        }
        else {
            // Prefill mode: ki-first for better V row reuse across queries
            for (nk_size_t ki = 0; ki < valid_kv; ki++) {
                for (nk_size_t d = 0; d < head_dim; d += svcntw()) {
                    svbool_t pg = svwhilelt_b32((nk_u32_t)d, (nk_u32_t)head_dim);

                    // Load V row once (reused for all qi)
                    svbfloat16_t v_bf16 = svld1_bf16(pg,
                                                     (bfloat16_t const *)(v + (kv_start + ki) * head_dim_padded + d));
                    svfloat32_t v_f32 = nk_bf16_to_f32_sve_(pg, v_bf16);

                    // Accumulate for all query rows
                    for (nk_size_t qi = 0; qi < valid_q; qi++) {
                        nk_f32_t w = all_weights[qi][ki];
                        svfloat32_t w_vec = svdup_f32(w);

                        svfloat32_t acc = svld1_f32(pg, o_acc + qi * head_dim_padded + d);
                        acc = svmla_f32_x(pg, acc, w_vec, v_f32);
                        svst1_f32(pg, o_acc + qi * head_dim_padded + d, acc);
                    }
                }
            }
        }
    }

    // Final: Vectorized normalization.
    NK_ALIGN64 nk_f32_t final_sums[16];
    svst1_f32(ptrue_s, final_sums, row_sum_v);

    for (nk_size_t qi = 0; qi < valid_q; qi++) {
        nk_f32_t inv_sum = (final_sums[qi] > 0.0f) ? (1.0f / final_sums[qi]) : 0.0f;
        svfloat32_t inv_sum_v = svdup_f32(inv_sum);

        for (nk_size_t d = 0; d < head_dim; d += svcntw()) {
            svbool_t pg = svwhilelt_b32((nk_u32_t)d, (nk_u32_t)head_dim);

            svfloat32_t o = svmul_f32_x(pg, svld1_f32(pg, o_acc + qi * head_dim_padded + d), inv_sum_v);
            svbfloat16_t o_bf16 = nk_f32_to_bf16_sve_(pg, o);
            svst1_bf16(pg, (bfloat16_t *)(output + qi * head_dim + d), o_bf16);
        }
    }
}

NK_PUBLIC void nk_attention_bf16_sme(nk_bf16_t const *q, void const *kv_packed, nk_bf16_t *output, nk_size_t num_heads,
                                     nk_size_t num_kv_heads, nk_size_t query_len, nk_size_t kv_len, nk_size_t head_dim,
                                     nk_f32_t scale) {

    nk_attention_sme_kv_packed_header_t const *header = (nk_attention_sme_kv_packed_header_t const *)kv_packed;
    nk_size_t head_dim_padded = header->head_dim_padded;
    nk_size_t kv_head_stride = kv_len * head_dim_padded;

    nk_bf16_t const *k_packed = (nk_bf16_t const *)((char const *)kv_packed + header->k_offset);
    nk_bf16_t const *v_packed = (nk_bf16_t const *)((char const *)kv_packed + header->v_offset);

    nk_size_t group_size = (num_kv_heads > 0) ? num_heads / num_kv_heads : 1;

    // Process each query head
    for (nk_size_t q_head = 0; q_head < num_heads; q_head++) {
        nk_size_t kv_head = q_head / group_size;

        nk_bf16_t const *q_ptr = q + q_head * query_len * head_dim;
        nk_bf16_t const *k_ptr = k_packed + kv_head * kv_head_stride;
        nk_bf16_t const *v_ptr = v_packed + kv_head * kv_head_stride;
        nk_bf16_t *out_ptr = output + q_head * query_len * head_dim;

        // Process queries in blocks of 16 (matches ZA tile dimension)
        for (nk_size_t q_start = 0; q_start < query_len; q_start += 16) {
            nk_size_t q_block_len = (q_start + 16 < query_len) ? 16 : (query_len - q_start);

            nk_attention_bf16_sme_kernel_(q_ptr + q_start * head_dim, k_ptr, v_ptr, out_ptr + q_start * head_dim,
                                          q_block_len, kv_len, head_dim, head_dim_padded, scale);
        }
    }
}

/**
 *  @brief Internal: f16 attention kernel using SME outer products and streaming SVE.
 *
 *  FlashAttention-2 algorithm with:
 *  - SME outer products for Q×Kᵀ computation
 *  - Vectorized softmax using nk_exp_f32_sve_ (no __builtin_expf)
 *  - F32 → F16 downcast before P×V multiplication
 *  - Predicated SVE loops for unified body/tail handling
 */
__arm_locally_streaming __arm_new("za") static void nk_attention_f16_sme_kernel_(
    nk_f16_t const *q, nk_f16_t const *k, nk_f16_t const *v, nk_f16_t *output, nk_size_t query_len, nk_size_t kv_len,
    nk_size_t head_dim, nk_size_t head_dim_padded, nk_f32_t scale) {

    svbool_t const ptrue_s = svptrue_b32();
    svbool_t const ptrue_h = svptrue_b16();
    nk_size_t const Bc = 16;
    nk_size_t const valid_q = (query_len < 16) ? query_len : 16;

    // State arrays (NOT sizeless types - those can't go in structs/arrays)
    NK_ALIGN64 nk_f32_t row_max[16];
    NK_ALIGN64 nk_f32_t row_sum[16];
    NK_ALIGN64 nk_f32_t o_acc[16 * 256]; // Max head_dim=256

    // Initialize state using vectorized SVE stores
    svst1_f32(ptrue_s, row_max, svdup_f32(NK_F32_MIN));
    svst1_f32(ptrue_s, row_sum, svdup_f32(0.0f));

    // Vectorized init of o_acc using SVE loop
    svfloat32_t zero_vec = svdup_f32(0.0f);
    for (nk_size_t i = 0; i < 16 * head_dim_padded; i += svcntw()) { svst1_f32(ptrue_s, o_acc + i, zero_vec); }

    // Process KV in blocks of Bc=16
    for (nk_size_t kv_start = 0; kv_start < kv_len; kv_start += Bc) {
        nk_size_t const valid_kv = ((kv_start + Bc) <= kv_len) ? Bc : (kv_len - kv_start);

        // Phase 1: Q×Kᵀ using SME outer products.
        // Compute scores[16, 16] = Q[16, d] × K[16, d]ᵀ into ZA tile 0
        svzero_za();

        // Accumulate over depth dimension in chunks of 32 (f16 vector width)
        for (nk_size_t d = 0; d < head_dim_padded; d += 32) {
            // For each query row, load Q slice and corresponding K slice, do outer product
            for (nk_size_t qi = 0; qi < valid_q; qi++) {
                // Load Q[qi, d:d+32] as f16 vector
                svfloat16_t q_vec = svld1_f16(ptrue_h, (float16_t const *)(q + qi * head_dim + d));

                // For this depth slice, accumulate outer products with all K rows in block
                for (nk_size_t ki = 0; ki < valid_kv; ki++) {
                    // Load K[kv_start+ki, d:d+32] as f16 vector
                    svfloat16_t k_vec = svld1_f16(ptrue_h,
                                                  (float16_t const *)(k + (kv_start + ki) * head_dim_padded + d));

                    // Outer product accumulate: ZA[qi, ki] += dot(q_vec, k_vec)
                    // Note: svmopa does 2-way widening, so we use it row-by-row
                    svmopa_za32_f16_m(0, ptrue_s, ptrue_s, q_vec, k_vec);
                }
            }
        }

        // Phase 2: Extract scores, apply online softmax, accumulate P×V.
        // Process each query row: extract score row, compute softmax weights, multiply by V
        NK_ALIGN64 nk_f32_t score_row[16];
        NK_ALIGN64 nk_f32_t weights_f32[16];

        // Temporary arrays to collect per-row updates for batch application
        NK_ALIGN64 nk_f32_t block_maxes[16];
        NK_ALIGN64 nk_f32_t new_maxes[16];
        NK_ALIGN64 nk_f32_t corrections[16];
        NK_ALIGN64 nk_f32_t sum_deltas[16];
        NK_ALIGN64 nk_f32_t all_weights[16][16];

        for (nk_size_t qi = 0; qi < valid_q; qi++) {
            // Extract score row from ZA tile
            svst1_hor_za32(0, (nk_u32_t)qi, ptrue_s, score_row);

            // Scale scores and find row max (vectorized)
            svfloat32_t s_vec = svmul_f32_x(ptrue_s, svld1_f32(ptrue_s, score_row), svdup_f32(scale));
            svst1_f32(ptrue_s, score_row, s_vec); // Store scaled scores
            block_maxes[qi] = svmaxv_f32(ptrue_s, s_vec);
        }

        // Vectorized max update for all rows at once.
        svfloat32_t old_max_vec = svld1_f32(ptrue_s, row_max);
        svfloat32_t block_max_vec = svld1_f32(ptrue_s, block_maxes);
        svfloat32_t new_max_vec = svmax_f32_x(ptrue_s, old_max_vec, block_max_vec);
        svst1_f32(ptrue_s, new_maxes, new_max_vec);

        // Vectorized correction factor computation.
        svfloat32_t correction_vec = nk_exp_f32_sve_(ptrue_s, svsub_f32_x(ptrue_s, old_max_vec, new_max_vec));
        svst1_f32(ptrue_s, corrections, correction_vec);

        // Vectorized row_sum rescaling.
        svfloat32_t row_sum_vec = svld1_f32(ptrue_s, row_sum);
        row_sum_vec = svmul_f32_x(ptrue_s, row_sum_vec, correction_vec);

        // Process each row for o_acc rescaling and weight computation
        for (nk_size_t qi = 0; qi < valid_q; qi++) {
            nk_f32_t correction = corrections[qi];
            nk_f32_t new_max = new_maxes[qi];

            // Rescale old output accumulator (vectorized over head_dim with predicates)
            svfloat32_t corr_bcast = svdup_f32(correction);
            for (nk_size_t d = 0; d < head_dim; d += svcntw()) {
                svbool_t pg = svwhilelt_b32((nk_u32_t)d, (nk_u32_t)head_dim);
                svfloat32_t o = svld1_f32(pg, o_acc + qi * head_dim_padded + d);
                o = svmul_f32_x(pg, o, corr_bcast);
                svst1_f32(pg, o_acc + qi * head_dim_padded + d, o);
            }

            // Re-extract score row and compute softmax weights
            svst1_hor_za32(0, (nk_u32_t)qi, ptrue_s, score_row);
            svfloat32_t s_vec = svmul_f32_x(ptrue_s, svld1_f32(ptrue_s, score_row), svdup_f32(scale));
            svfloat32_t p_vec = nk_exp_f32_sve_(ptrue_s, svsub_f32_x(ptrue_s, s_vec, svdup_f32(new_max)));

            // Collect sum delta for later batch update
            sum_deltas[qi] = svaddv_f32(ptrue_s, p_vec);

            // Downcast weights F32 → F16 → F32 (quantization step required before V multiplication)
            svfloat16_t weights_f16 = svcvt_f16_f32_x(ptrue_s, p_vec);
            svfloat32_t weights_quantized = svcvt_f32_f16_x(ptrue_s, weights_f16);
            svst1_f32(ptrue_s, all_weights[qi], weights_quantized);
        }

        // Vectorized row_sum update with deltas.
        svfloat32_t delta_vec = svld1_f32(ptrue_s, sum_deltas);
        row_sum_vec = svadd_f32_x(ptrue_s, row_sum_vec, delta_vec);
        svst1_f32(ptrue_s, row_sum, row_sum_vec);

        // Update row_max.
        svst1_f32(ptrue_s, row_max, new_max_vec);

        // Phase 3: Accumulate P×V for all query rows.
        for (nk_size_t qi = 0; qi < valid_q; qi++) {
            for (nk_size_t ki = 0; ki < valid_kv; ki++) {
                nk_f32_t w = all_weights[qi][ki];
                svfloat32_t w_vec = svdup_f32(w);

                // Vectorized over head_dim with predicated loop (unified body/tail)
                for (nk_size_t d = 0; d < head_dim; d += svcntw()) {
                    svbool_t pg = svwhilelt_b32((nk_u32_t)d, (nk_u32_t)head_dim);

                    // Load V row slice, convert to f32
                    svfloat16_t v_f16 = svld1_f16(pg, (float16_t const *)(v + (kv_start + ki) * head_dim_padded + d));
                    svfloat32_t v_f32 = svcvt_f32_f16_x(pg, v_f16);

                    // Load accumulator, multiply-add, store
                    svfloat32_t acc = svld1_f32(pg, o_acc + qi * head_dim_padded + d);
                    acc = svmla_f32_x(pg, acc, w_vec, v_f32);
                    svst1_f32(pg, o_acc + qi * head_dim_padded + d, acc);
                }
            }
        }
    }

    // Final: Vectorized normalization with predicates.
    // Compute inverse sums for all rows at once using SVE
    svfloat32_t final_sum_vec = svld1_f32(ptrue_s, row_sum);
    svfloat32_t ones_vec = svdup_f32(1.0f);
    svfloat32_t zeros_vec = svdup_f32(0.0f);

    // inv_sum = (row_sum > 0) ? 1/row_sum : 0 using svsel
    svbool_t sum_positive = svcmpgt_f32(ptrue_s, final_sum_vec, zeros_vec);
    svfloat32_t inv_sum_vec = svsel_f32(sum_positive, svdiv_f32_x(ptrue_s, ones_vec, final_sum_vec), zeros_vec);

    // Store inverse sums for per-row access
    NK_ALIGN64 nk_f32_t inv_sums[16];
    svst1_f32(ptrue_s, inv_sums, inv_sum_vec);

    for (nk_size_t qi = 0; qi < valid_q; qi++) {
        svfloat32_t inv_sum_v = svdup_f32(inv_sums[qi]);

        for (nk_size_t d = 0; d < head_dim; d += svcntw()) {
            svbool_t pg = svwhilelt_b32((nk_u32_t)d, (nk_u32_t)head_dim);

            // Load f32 accumulator, scale by 1/sum
            svfloat32_t o = svmul_f32_x(pg, svld1_f32(pg, o_acc + qi * head_dim_padded + d), inv_sum_v);

            // Convert to f16 and store
            svfloat16_t o_f16 = svcvt_f16_f32_x(pg, o);
            svst1_f16(pg, (float16_t *)(output + qi * head_dim + d), o_f16);
        }
    }
}

NK_PUBLIC void nk_attention_f16_sme(nk_f16_t const *q, void const *kv_packed, nk_f16_t *output, nk_size_t num_heads,
                                    nk_size_t num_kv_heads, nk_size_t query_len, nk_size_t kv_len, nk_size_t head_dim,
                                    nk_f32_t scale) {

    nk_attention_sme_kv_packed_header_t const *header = (nk_attention_sme_kv_packed_header_t const *)kv_packed;
    nk_size_t head_dim_padded = header->head_dim_padded;
    nk_size_t kv_head_stride = kv_len * head_dim_padded;

    nk_f16_t const *k_packed = (nk_f16_t const *)((char const *)kv_packed + header->k_offset);
    nk_f16_t const *v_packed = (nk_f16_t const *)((char const *)kv_packed + header->v_offset);

    nk_size_t group_size = (num_kv_heads > 0) ? num_heads / num_kv_heads : 1;

    for (nk_size_t q_head = 0; q_head < num_heads; q_head++) {
        nk_size_t kv_head = q_head / group_size;

        nk_f16_t const *q_ptr = q + q_head * query_len * head_dim;
        nk_f16_t const *k_ptr = k_packed + kv_head * kv_head_stride;
        nk_f16_t const *v_ptr = v_packed + kv_head * kv_head_stride;
        nk_f16_t *out_ptr = output + q_head * query_len * head_dim;

        for (nk_size_t q_start = 0; q_start < query_len; q_start += 16) {
            nk_size_t q_block_len = (q_start + 16 < query_len) ? 16 : (query_len - q_start);

            nk_attention_f16_sme_kernel_(q_ptr + q_start * head_dim, k_ptr, v_ptr, out_ptr + q_start * head_dim,
                                         q_block_len, kv_len, head_dim, head_dim_padded, scale);
        }
    }
}

NK_PUBLIC void nk_attention_causal_bf16_sme(nk_bf16_t const *q, void const *kv_packed, nk_bf16_t *output,
                                            nk_size_t num_heads, nk_size_t num_kv_heads, nk_size_t query_len,
                                            nk_size_t kv_len, nk_size_t head_dim, nk_f32_t scale) {
    // TODO: Implement proper causal masking with block skipping
    // For now, delegate to full attention (correct for decode where query_len=1)
    nk_attention_bf16_sme(q, kv_packed, output, num_heads, num_kv_heads, query_len, kv_len, head_dim, scale);
}

NK_PUBLIC void nk_attention_causal_f16_sme(nk_f16_t const *q, void const *kv_packed, nk_f16_t *output,
                                           nk_size_t num_heads, nk_size_t num_kv_heads, nk_size_t query_len,
                                           nk_size_t kv_len, nk_size_t head_dim, nk_f32_t scale) {
    // TODO: Implement proper causal masking with block skipping
    // For now, delegate to full attention (correct for decode where query_len=1)
    nk_attention_f16_sme(q, kv_packed, output, num_heads, num_kv_heads, query_len, kv_len, head_dim, scale);
}

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
#endif // NK_ATTENTION_SME_H

/**
 *  @brief SIMD-accelerated MaxSim (ColBERT Late Interaction).
 *  @file include/numkong/maxsim.h
 *  @author Ash Vardanian
 *  @date February 17, 2026
 *
 *  Computes angular distance late-interaction: result = Σᵢ minⱼ angular(qᵢ, dⱼ).
 *  Angular distance = 1 - dot(q, d) / sqrt(||q||² × ||d||²), clamped >= 0.
 *
 *  Strategy: coarse i8-quantized screening with running argmax (dot as proxy for argmin angular),
 *  then full-precision refinement of the winning pairs via nk_dot_* primitives,
 *  finalized with angular distance and accumulated with f64.
 *
 *  It implements several operations:
 *
 *  - "maxsim_packed" - computing MaxSim where both Q and D are pre-packed into optimal form
 *  - "maxsim_packed_size" - estimating the memory requirements for external malloc
 *  - "maxsim_pack" - performing the pre-processing (quantization + original copy)
 *
 *  @section maxsim_api Two-Phase API
 *
 *  @code{.c}
 *  // Pack query and document matrices
 *  nk_size_t q_bytes = nk_maxsim_packed_size_bf16(n_q, depth);
 *  nk_size_t d_bytes = nk_maxsim_packed_size_bf16(n_d, depth);
 *  void *q_packed = malloc(q_bytes);
 *  void *d_packed = malloc(d_bytes);
 *  nk_maxsim_pack_bf16(queries, n_q, depth, depth * sizeof(nk_bf16_t), q_packed);
 *  nk_maxsim_pack_bf16(documents, n_d, depth, depth * sizeof(nk_bf16_t), d_packed);
 *
 *  // Compute MaxSim score
 *  nk_f32_t score = nk_maxsim_packed_bf16(q_packed, d_packed, n_q, n_d, depth);
 *  @endcode
 *
 *  @section maxsim_packed_layout Packed Buffer Layout
 *
 *  [Header 64B] [i8 vectors, 64B-aligned] [metadata, 64B-aligned] [originals row-major, 64B-aligned]
 *
 *  The packed format is backend-specific: different ISAs use different i8 depth padding
 *  and clamp ranges. Pack with the matching ISA's pack function.
 *
 *  @section maxsim_isa_support ISA Support
 *
 *  Currently implemented:
 *  - Serial: scalar reference (all platforms)
 *  - Haswell: AVX2 VPMADDUBSW coarse [-79,79] + bias correction (bf16/f32/f16)
 *  - Icelake: AVX-512 VNNI VPDPBUSD coarse (f32/f16)
 *  - Genoa: AVX-512 VNNI coarse + VDPBF16PS refinement (bf16 only)
 *  - NEONSDOT: ARM SDOT (vdotq_s32) coarse, no bias correction (bf16/f32/f16)
 *  - SME: ARM fused BFMOPA (existing, unchanged)
 */
#ifndef NK_MAXSIM_H
#define NK_MAXSIM_H

#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

/**
 *  @brief Returns packed buffer size in bytes for a maxsim vector set.
 *  @param[in] n The number of vectors to pack.
 *  @param[in] k The number of dimensions per vector (depth).
 *  @note The packed layout is backend-specific and must be produced by the matching pack function.
 */
NK_DYNAMIC nk_size_t nk_maxsim_packed_size_bf16(nk_size_t n, nk_size_t k);
/** @copydoc nk_maxsim_packed_size_bf16 */
NK_DYNAMIC nk_size_t nk_maxsim_packed_size_f32(nk_size_t n, nk_size_t k);
/** @copydoc nk_maxsim_packed_size_bf16 */
NK_DYNAMIC nk_size_t nk_maxsim_packed_size_f16(nk_size_t n, nk_size_t k);

/**
 *  @brief Packs vectors into a backend-specific layout for maxsim computation.
 *  @param[in] vectors The input vectors in row-major order.
 *  @param[in] n The number of vectors.
 *  @param[in] k The number of dimensions per vector (depth).
 *  @param[in] stride The row stride in bytes for the input vectors.
 *  @param[out] packed The output packed buffer from nk_maxsim_packed_size_bf16.
 */
NK_DYNAMIC void nk_maxsim_pack_bf16(nk_bf16_t const *vectors, nk_size_t n, nk_size_t k, nk_size_t stride, void *packed);
/** @copydoc nk_maxsim_pack_bf16 */
NK_DYNAMIC void nk_maxsim_pack_f32(nk_f32_t const *vectors, nk_size_t n, nk_size_t k, nk_size_t stride, void *packed);
/** @copydoc nk_maxsim_pack_bf16 */
NK_DYNAMIC void nk_maxsim_pack_f16(nk_f16_t const *vectors, nk_size_t n, nk_size_t k, nk_size_t stride, void *packed);

/**
 *  @brief Computes angular distance late-interaction on pre-packed vectors.
 *  Returns Σᵢ minⱼ angular(qᵢ, dⱼ) where angular = 1 - dot / sqrt(||q||² × ||d||²).
 *
 *  @param[in] q_packed Packed query vectors (from nk_maxsim_pack_bf16).
 *  @param[in] d_packed Packed document vectors (from nk_maxsim_pack_bf16).
 *  @param[in] n_q Number of query vectors.
 *  @param[in] n_d Number of document vectors.
 *  @param[in] depth Number of dimensions per vector.
 *  @return Sum of per-query minimum angular distances.
 */
NK_DYNAMIC nk_f32_t nk_maxsim_packed_bf16(void const *q_packed, void const *d_packed, nk_size_t n_q, nk_size_t n_d,
                                          nk_size_t depth);
/** @copydoc nk_maxsim_packed_bf16 */
NK_DYNAMIC nk_f32_t nk_maxsim_packed_f32(void const *q_packed, void const *d_packed, nk_size_t n_q, nk_size_t n_d,
                                         nk_size_t depth);
/** @copydoc nk_maxsim_packed_bf16 */
NK_DYNAMIC nk_f32_t nk_maxsim_packed_f16(void const *q_packed, void const *d_packed, nk_size_t n_q, nk_size_t n_d,
                                         nk_size_t depth);

// Serial (always available)
/** @copydoc nk_maxsim_packed_size_bf16 */
NK_PUBLIC nk_size_t nk_maxsim_packed_size_bf16_serial(nk_size_t n, nk_size_t k);
/** @copydoc nk_maxsim_packed_size_bf16 */
NK_PUBLIC nk_size_t nk_maxsim_packed_size_f32_serial(nk_size_t n, nk_size_t k);
/** @copydoc nk_maxsim_packed_size_bf16 */
NK_PUBLIC nk_size_t nk_maxsim_packed_size_f16_serial(nk_size_t n, nk_size_t k);
/** @copydoc nk_maxsim_pack_bf16 */
NK_PUBLIC void nk_maxsim_pack_bf16_serial(nk_bf16_t const *vectors, nk_size_t n, nk_size_t k, nk_size_t stride,
                                          void *packed);
/** @copydoc nk_maxsim_pack_bf16 */
NK_PUBLIC void nk_maxsim_pack_f32_serial(nk_f32_t const *vectors, nk_size_t n, nk_size_t k, nk_size_t stride,
                                         void *packed);
/** @copydoc nk_maxsim_pack_bf16 */
NK_PUBLIC void nk_maxsim_pack_f16_serial(nk_f16_t const *vectors, nk_size_t n, nk_size_t k, nk_size_t stride,
                                         void *packed);
/** @copydoc nk_maxsim_packed_bf16 */
NK_PUBLIC nk_f32_t nk_maxsim_packed_bf16_serial(void const *q_packed, void const *d_packed, nk_size_t n_q,
                                                nk_size_t n_d, nk_size_t depth);
/** @copydoc nk_maxsim_packed_bf16 */
NK_PUBLIC nk_f32_t nk_maxsim_packed_f32_serial(void const *q_packed, void const *d_packed, nk_size_t n_q, nk_size_t n_d,
                                               nk_size_t depth);
/** @copydoc nk_maxsim_packed_bf16 */
NK_PUBLIC nk_f32_t nk_maxsim_packed_f16_serial(void const *q_packed, void const *d_packed, nk_size_t n_q, nk_size_t n_d,
                                               nk_size_t depth);

#if NK_TARGET_ICELAKE
/** @copydoc nk_maxsim_packed_size_bf16 */
NK_PUBLIC nk_size_t nk_maxsim_packed_size_f32_icelake(nk_size_t n, nk_size_t k);
/** @copydoc nk_maxsim_packed_size_bf16 */
NK_PUBLIC nk_size_t nk_maxsim_packed_size_f16_icelake(nk_size_t n, nk_size_t k);
/** @copydoc nk_maxsim_pack_bf16 */
NK_PUBLIC void nk_maxsim_pack_f32_icelake(nk_f32_t const *vectors, nk_size_t n, nk_size_t k, nk_size_t stride,
                                          void *packed);
/** @copydoc nk_maxsim_pack_bf16 */
NK_PUBLIC void nk_maxsim_pack_f16_icelake(nk_f16_t const *vectors, nk_size_t n, nk_size_t k, nk_size_t stride,
                                          void *packed);
/** @copydoc nk_maxsim_packed_bf16 */
NK_PUBLIC nk_f32_t nk_maxsim_packed_f32_icelake(void const *q_packed, void const *d_packed, nk_size_t n_q,
                                                nk_size_t n_d, nk_size_t depth);
/** @copydoc nk_maxsim_packed_bf16 */
NK_PUBLIC nk_f32_t nk_maxsim_packed_f16_icelake(void const *q_packed, void const *d_packed, nk_size_t n_q,
                                                nk_size_t n_d, nk_size_t depth);
#endif // NK_TARGET_ICELAKE

#if NK_TARGET_GENOA
/** @copydoc nk_maxsim_packed_size_bf16 */
NK_PUBLIC nk_size_t nk_maxsim_packed_size_bf16_genoa(nk_size_t n, nk_size_t k);
/** @copydoc nk_maxsim_pack_bf16 */
NK_PUBLIC void nk_maxsim_pack_bf16_genoa(nk_bf16_t const *vectors, nk_size_t n, nk_size_t k, nk_size_t stride,
                                         void *packed);
/** @copydoc nk_maxsim_packed_bf16 */
NK_PUBLIC nk_f32_t nk_maxsim_packed_bf16_genoa(void const *q_packed, void const *d_packed, nk_size_t n_q, nk_size_t n_d,
                                               nk_size_t depth);
#endif // NK_TARGET_GENOA

#if NK_TARGET_HASWELL
/** @copydoc nk_maxsim_packed_size_bf16 */
NK_PUBLIC nk_size_t nk_maxsim_packed_size_bf16_haswell(nk_size_t n, nk_size_t k);
/** @copydoc nk_maxsim_packed_size_bf16 */
NK_PUBLIC nk_size_t nk_maxsim_packed_size_f32_haswell(nk_size_t n, nk_size_t k);
/** @copydoc nk_maxsim_packed_size_bf16 */
NK_PUBLIC nk_size_t nk_maxsim_packed_size_f16_haswell(nk_size_t n, nk_size_t k);
/** @copydoc nk_maxsim_pack_bf16 */
NK_PUBLIC void nk_maxsim_pack_bf16_haswell(nk_bf16_t const *vectors, nk_size_t n, nk_size_t k, nk_size_t stride,
                                           void *packed);
/** @copydoc nk_maxsim_pack_bf16 */
NK_PUBLIC void nk_maxsim_pack_f32_haswell(nk_f32_t const *vectors, nk_size_t n, nk_size_t k, nk_size_t stride,
                                          void *packed);
/** @copydoc nk_maxsim_pack_bf16 */
NK_PUBLIC void nk_maxsim_pack_f16_haswell(nk_f16_t const *vectors, nk_size_t n, nk_size_t k, nk_size_t stride,
                                          void *packed);
/** @copydoc nk_maxsim_packed_bf16 */
NK_PUBLIC nk_f32_t nk_maxsim_packed_bf16_haswell(void const *q_packed, void const *d_packed, nk_size_t n_q,
                                                 nk_size_t n_d, nk_size_t depth);
/** @copydoc nk_maxsim_packed_bf16 */
NK_PUBLIC nk_f32_t nk_maxsim_packed_f32_haswell(void const *q_packed, void const *d_packed, nk_size_t n_q,
                                                nk_size_t n_d, nk_size_t depth);
/** @copydoc nk_maxsim_packed_bf16 */
NK_PUBLIC nk_f32_t nk_maxsim_packed_f16_haswell(void const *q_packed, void const *d_packed, nk_size_t n_q,
                                                nk_size_t n_d, nk_size_t depth);
#endif // NK_TARGET_HASWELL

#if NK_TARGET_NEONSDOT
/** @copydoc nk_maxsim_packed_size_bf16 */
NK_PUBLIC nk_size_t nk_maxsim_packed_size_bf16_neonsdot(nk_size_t n, nk_size_t k);
/** @copydoc nk_maxsim_packed_size_bf16 */
NK_PUBLIC nk_size_t nk_maxsim_packed_size_f32_neonsdot(nk_size_t n, nk_size_t k);
/** @copydoc nk_maxsim_packed_size_bf16 */
NK_PUBLIC nk_size_t nk_maxsim_packed_size_f16_neonsdot(nk_size_t n, nk_size_t k);
/** @copydoc nk_maxsim_pack_bf16 */
NK_PUBLIC void nk_maxsim_pack_bf16_neonsdot(nk_bf16_t const *vectors, nk_size_t n, nk_size_t k, nk_size_t stride,
                                            void *packed);
/** @copydoc nk_maxsim_pack_bf16 */
NK_PUBLIC void nk_maxsim_pack_f32_neonsdot(nk_f32_t const *vectors, nk_size_t n, nk_size_t k, nk_size_t stride,
                                           void *packed);
/** @copydoc nk_maxsim_pack_bf16 */
NK_PUBLIC void nk_maxsim_pack_f16_neonsdot(nk_f16_t const *vectors, nk_size_t n, nk_size_t k, nk_size_t stride,
                                           void *packed);
/** @copydoc nk_maxsim_packed_bf16 */
NK_PUBLIC nk_f32_t nk_maxsim_packed_bf16_neonsdot(void const *q_packed, void const *d_packed, nk_size_t n_q,
                                                  nk_size_t n_d, nk_size_t depth);
/** @copydoc nk_maxsim_packed_bf16 */
NK_PUBLIC nk_f32_t nk_maxsim_packed_f32_neonsdot(void const *q_packed, void const *d_packed, nk_size_t n_q,
                                                 nk_size_t n_d, nk_size_t depth);
/** @copydoc nk_maxsim_packed_bf16 */
NK_PUBLIC nk_f32_t nk_maxsim_packed_f16_neonsdot(void const *q_packed, void const *d_packed, nk_size_t n_q,
                                                 nk_size_t n_d, nk_size_t depth);
#endif // NK_TARGET_NEONSDOT

#if NK_TARGET_SME
/** @copydoc nk_maxsim_packed_size_bf16 */
NK_PUBLIC nk_size_t nk_maxsim_packed_size_bf16_sme(nk_size_t n, nk_size_t k);
/** @copydoc nk_maxsim_packed_size_bf16 */
NK_PUBLIC nk_size_t nk_maxsim_packed_size_f16_sme(nk_size_t n, nk_size_t k);
/** @copydoc nk_maxsim_packed_size_bf16 */
NK_PUBLIC nk_size_t nk_maxsim_packed_size_f32_sme(nk_size_t n, nk_size_t k);
/** @copydoc nk_maxsim_pack_bf16 */
NK_PUBLIC void nk_maxsim_pack_bf16_sme(nk_bf16_t const *vectors, nk_size_t n, nk_size_t k, nk_size_t stride,
                                       void *packed);
/** @copydoc nk_maxsim_pack_bf16 */
NK_PUBLIC void nk_maxsim_pack_f16_sme(nk_f16_t const *vectors, nk_size_t n, nk_size_t k, nk_size_t stride,
                                      void *packed);
/** @copydoc nk_maxsim_pack_bf16 */
NK_PUBLIC void nk_maxsim_pack_f32_sme(nk_f32_t const *vectors, nk_size_t n, nk_size_t k, nk_size_t stride,
                                      void *packed);
/** @copydoc nk_maxsim_packed_bf16 */
NK_PUBLIC nk_f32_t nk_maxsim_packed_bf16_sme(void const *q_packed, void const *d_packed, nk_size_t n_q, nk_size_t n_d,
                                             nk_size_t depth);
/** @copydoc nk_maxsim_packed_bf16 */
NK_PUBLIC nk_f32_t nk_maxsim_packed_f16_sme(void const *q_packed, void const *d_packed, nk_size_t n_q, nk_size_t n_d,
                                            nk_size_t depth);
/** @copydoc nk_maxsim_packed_bf16 */
NK_PUBLIC nk_f32_t nk_maxsim_packed_f32_sme(void const *q_packed, void const *d_packed, nk_size_t n_q, nk_size_t n_d,
                                            nk_size_t depth);
#endif // NK_TARGET_SME

#if defined(__cplusplus)
} // extern "C"
#endif

#include "numkong/maxsim/serial.h"
#include "numkong/maxsim/haswell.h"
#include "numkong/maxsim/icelake.h"
#include "numkong/maxsim/genoa.h"
#include "numkong/maxsim/neonsdot.h"
#include "numkong/maxsim/sme.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if !NK_DYNAMIC_DISPATCH

NK_PUBLIC nk_size_t nk_maxsim_packed_size_bf16(nk_size_t n, nk_size_t k) {
#if NK_TARGET_SME
    return nk_maxsim_packed_size_bf16_sme(n, k);
#elif NK_TARGET_GENOA
    return nk_maxsim_packed_size_bf16_genoa(n, k);
#elif NK_TARGET_HASWELL
    return nk_maxsim_packed_size_bf16_haswell(n, k);
#elif NK_TARGET_NEONSDOT
    return nk_maxsim_packed_size_bf16_neonsdot(n, k);
#else
    return nk_maxsim_packed_size_bf16_serial(n, k);
#endif
}

NK_PUBLIC nk_size_t nk_maxsim_packed_size_f32(nk_size_t n, nk_size_t k) {
#if NK_TARGET_SME
    return nk_maxsim_packed_size_f32_sme(n, k);
#elif NK_TARGET_ICELAKE
    return nk_maxsim_packed_size_f32_icelake(n, k);
#elif NK_TARGET_HASWELL
    return nk_maxsim_packed_size_f32_haswell(n, k);
#elif NK_TARGET_NEONSDOT
    return nk_maxsim_packed_size_f32_neonsdot(n, k);
#else
    return nk_maxsim_packed_size_f32_serial(n, k);
#endif
}

NK_PUBLIC nk_size_t nk_maxsim_packed_size_f16(nk_size_t n, nk_size_t k) {
#if NK_TARGET_SME
    return nk_maxsim_packed_size_f16_sme(n, k);
#elif NK_TARGET_ICELAKE
    return nk_maxsim_packed_size_f16_icelake(n, k);
#elif NK_TARGET_HASWELL
    return nk_maxsim_packed_size_f16_haswell(n, k);
#elif NK_TARGET_NEONSDOT
    return nk_maxsim_packed_size_f16_neonsdot(n, k);
#else
    return nk_maxsim_packed_size_f16_serial(n, k);
#endif
}

NK_PUBLIC void nk_maxsim_pack_bf16(nk_bf16_t const *vectors, nk_size_t n, nk_size_t k, nk_size_t stride, void *packed) {
#if NK_TARGET_SME
    nk_maxsim_pack_bf16_sme(vectors, n, k, stride, packed);
#elif NK_TARGET_GENOA
    nk_maxsim_pack_bf16_genoa(vectors, n, k, stride, packed);
#elif NK_TARGET_HASWELL
    nk_maxsim_pack_bf16_haswell(vectors, n, k, stride, packed);
#elif NK_TARGET_NEONSDOT
    nk_maxsim_pack_bf16_neonsdot(vectors, n, k, stride, packed);
#else
    nk_maxsim_pack_bf16_serial(vectors, n, k, stride, packed);
#endif
}

NK_PUBLIC void nk_maxsim_pack_f32(nk_f32_t const *vectors, nk_size_t n, nk_size_t k, nk_size_t stride, void *packed) {
#if NK_TARGET_SME
    nk_maxsim_pack_f32_sme(vectors, n, k, stride, packed);
#elif NK_TARGET_ICELAKE
    nk_maxsim_pack_f32_icelake(vectors, n, k, stride, packed);
#elif NK_TARGET_HASWELL
    nk_maxsim_pack_f32_haswell(vectors, n, k, stride, packed);
#elif NK_TARGET_NEONSDOT
    nk_maxsim_pack_f32_neonsdot(vectors, n, k, stride, packed);
#else
    nk_maxsim_pack_f32_serial(vectors, n, k, stride, packed);
#endif
}

NK_PUBLIC void nk_maxsim_pack_f16(nk_f16_t const *vectors, nk_size_t n, nk_size_t k, nk_size_t stride, void *packed) {
#if NK_TARGET_SME
    nk_maxsim_pack_f16_sme(vectors, n, k, stride, packed);
#elif NK_TARGET_ICELAKE
    nk_maxsim_pack_f16_icelake(vectors, n, k, stride, packed);
#elif NK_TARGET_HASWELL
    nk_maxsim_pack_f16_haswell(vectors, n, k, stride, packed);
#elif NK_TARGET_NEONSDOT
    nk_maxsim_pack_f16_neonsdot(vectors, n, k, stride, packed);
#else
    nk_maxsim_pack_f16_serial(vectors, n, k, stride, packed);
#endif
}

NK_PUBLIC nk_f32_t nk_maxsim_packed_bf16(void const *q_packed, void const *d_packed, nk_size_t n_q, nk_size_t n_d,
                                         nk_size_t depth) {
#if NK_TARGET_SME
    return nk_maxsim_packed_bf16_sme(q_packed, d_packed, n_q, n_d, depth);
#elif NK_TARGET_GENOA
    return nk_maxsim_packed_bf16_genoa(q_packed, d_packed, n_q, n_d, depth);
#elif NK_TARGET_HASWELL
    return nk_maxsim_packed_bf16_haswell(q_packed, d_packed, n_q, n_d, depth);
#elif NK_TARGET_NEONSDOT
    return nk_maxsim_packed_bf16_neonsdot(q_packed, d_packed, n_q, n_d, depth);
#else
    return nk_maxsim_packed_bf16_serial(q_packed, d_packed, n_q, n_d, depth);
#endif
}

NK_PUBLIC nk_f32_t nk_maxsim_packed_f32(void const *q_packed, void const *d_packed, nk_size_t n_q, nk_size_t n_d,
                                        nk_size_t depth) {
#if NK_TARGET_SME
    return nk_maxsim_packed_f32_sme(q_packed, d_packed, n_q, n_d, depth);
#elif NK_TARGET_ICELAKE
    return nk_maxsim_packed_f32_icelake(q_packed, d_packed, n_q, n_d, depth);
#elif NK_TARGET_HASWELL
    return nk_maxsim_packed_f32_haswell(q_packed, d_packed, n_q, n_d, depth);
#elif NK_TARGET_NEONSDOT
    return nk_maxsim_packed_f32_neonsdot(q_packed, d_packed, n_q, n_d, depth);
#else
    return nk_maxsim_packed_f32_serial(q_packed, d_packed, n_q, n_d, depth);
#endif
}

NK_PUBLIC nk_f32_t nk_maxsim_packed_f16(void const *q_packed, void const *d_packed, nk_size_t n_q, nk_size_t n_d,
                                        nk_size_t depth) {
#if NK_TARGET_SME
    return nk_maxsim_packed_f16_sme(q_packed, d_packed, n_q, n_d, depth);
#elif NK_TARGET_ICELAKE
    return nk_maxsim_packed_f16_icelake(q_packed, d_packed, n_q, n_d, depth);
#elif NK_TARGET_HASWELL
    return nk_maxsim_packed_f16_haswell(q_packed, d_packed, n_q, n_d, depth);
#elif NK_TARGET_NEONSDOT
    return nk_maxsim_packed_f16_neonsdot(q_packed, d_packed, n_q, n_d, depth);
#else
    return nk_maxsim_packed_f16_serial(q_packed, d_packed, n_q, n_d, depth);
#endif
}

#endif // !NK_DYNAMIC_DISPATCH

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_MAXSIM_H

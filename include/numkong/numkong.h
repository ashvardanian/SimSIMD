/**
 *  @brief SIMD-accelerated Similarity Measures and Distance Functions.
 *  @file include/numkong.h
 *  @author Ash Vardanian
 *  @date March 14, 2023
 *
 *  Umbrella header that includes all domain-specific kernel headers
 *  and the runtime capability detection infrastructure.
 */

#ifndef NK_NUMKONG_H
#define NK_NUMKONG_H

#include "numkong/capabilities.h" // Runtime detection
#include "numkong/cast.h"         // Type Conversions
#include "numkong/set.h"          // Hamming, Jaccard
#include "numkong/curved.h"       // Mahalanobis, Bilinear Forms
#include "numkong/dot.h"          // Inner (dot) product, and its conjugate
#include "numkong/dots.h"         // GEMM-style MxN batched dot-products
#include "numkong/each.h"         // Weighted Sum, Fused-Multiply-Add
#include "numkong/geospatial.h"   // Haversine and Vincenty
#include "numkong/mesh.h"         // RMSD, Kabsch, Umeyama
#include "numkong/probability.h"  // Kullback-Leibler, Jensen-Shannon
#include "numkong/reduce.h"       // Horizontal MinMax & Moments reductions
#include "numkong/sets.h"         // Hamming & Jaccard distances for binary sets
#include "numkong/sparse.h"       // Set Intersections and Sparse Dot Products
#include "numkong/spatial.h"      // Euclidean, Angular
#include "numkong/trigonometry.h" // Sin, Cos, Atan

#if defined(__cplusplus)
extern "C" {
#endif

/**
 *  @brief  Returns the output dtype for a given metric kind and input dtype.
 */
NK_PUBLIC nk_dtype_t nk_kernel_output_dtype(nk_kernel_kind_t kind, nk_dtype_t input) {
    switch (kind) {
    case nk_kernel_dot_k:
    case nk_kernel_vdot_k: return nk_dot_output_dtype(input);
    case nk_kernel_angular_k: return nk_angular_output_dtype(input);
    case nk_kernel_sqeuclidean_k: return nk_sqeuclidean_output_dtype(input);
    case nk_kernel_euclidean_k: return nk_euclidean_output_dtype(input);
    case nk_kernel_bilinear_k: return nk_bilinear_output_dtype(input);
    case nk_kernel_mahalanobis_k: return nk_mahalanobis_output_dtype(input);
    case nk_kernel_hamming_k: return nk_hamming_output_dtype(input);
    case nk_kernel_jaccard_k: return nk_jaccard_output_dtype(input);
    case nk_kernel_haversine_k: return nk_haversine_output_dtype(input);
    case nk_kernel_vincenty_k: return nk_vincenty_output_dtype(input);
    case nk_kernel_kld_k:
    case nk_kernel_jsd_k: return nk_probability_output_dtype(input);
    case nk_kernel_rmsd_k: return nk_rmsd_output_dtype(input);
    case nk_kernel_kabsch_k: return nk_kabsch_output_dtype(input);
    case nk_kernel_umeyama_k: return nk_umeyama_output_dtype(input);
    default: return nk_dtype_unknown_k;
    }
}

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_NUMKONG_H

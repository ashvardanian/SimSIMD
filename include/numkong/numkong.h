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
#include "numkong/reduce.h"       // Horizontal reductions: sum, min, max
#include "numkong/sets.h"         // Hamming & Jaccard distances for binary sets
#include "numkong/sparse.h"       // Intersect
#include "numkong/spatial.h"      // L2, Angular
#include "numkong/trigonometry.h" // Sin, Cos, Atan

#endif // NK_NUMKONG_H

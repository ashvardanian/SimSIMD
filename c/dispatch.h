/**
 *  @brief Common Definitions for Dispatch Files.
 *  @file c/dispatch.h
 *  @author Ash Vardanian
 *  @date February 3, 2026
 */
#ifndef NK_DISPATCH_H
#define NK_DISPATCH_H

#define NK_DYNAMIC_DISPATCH 1
#define NK_NATIVE_F16       0
#define NK_NATIVE_BF16      0

/*  Depending on the Operating System, the following intrinsics are available
 *  on recent compiler toolchains:
 *
 *  - Linux: everything is available in GCC 12+ and Clang 16+.
 *  - Windows - MSVC: everything except Sapphire Rapids and ARM SVE.
 *  - macOS - Apple Clang: only Arm NEON and x86 AVX2 Haswell extensions are available.
 */
#if !defined(NK_TARGET_NEON) && (defined(__APPLE__) || defined(__linux__))
#define NK_TARGET_NEON 1
#endif
#if !defined(NK_TARGET_SVE) && (defined(__linux__))
#define NK_TARGET_SVE 1
#endif
#if !defined(NK_TARGET_SVE2) && (defined(__linux__))
#define NK_TARGET_SVE2 1
#endif
#if !defined(NK_TARGET_HASWELL) && (defined(_MSC_VER) || defined(__APPLE__) || defined(__linux__))
#define NK_TARGET_HASWELL 1
#endif
#if !defined(NK_TARGET_SKYLAKE) && (defined(_MSC_VER) || defined(__linux__))
#define NK_TARGET_SKYLAKE 1
#endif
#if !defined(NK_TARGET_ICELAKE) && (defined(_MSC_VER) || defined(__linux__))
#define NK_TARGET_ICELAKE 1
#endif
#if !defined(NK_TARGET_GENOA) && (defined(__linux__))
#define NK_TARGET_GENOA 1
#endif
#if !defined(NK_TARGET_SAPPHIRE) && (defined(__linux__))
#define NK_TARGET_SAPPHIRE 1
#endif
#if !defined(NK_TARGET_TURIN) && (defined(__linux__))
#define NK_TARGET_TURIN 1
#endif
#if !defined(NK_TARGET_SIERRA) && (defined(__linux__))
#define NK_TARGET_SIERRA 1
#endif

#include <numkong/numkong.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward declaration of dispatch table type (same structure as in numkong.c)
typedef struct {
    // Dot products
    nk_metric_dense_punned_t dot_i8;
    nk_metric_dense_punned_t dot_u8;
    nk_metric_dense_punned_t dot_i4;
    nk_metric_dense_punned_t dot_u4;
    nk_metric_dense_punned_t dot_f16;
    nk_metric_dense_punned_t dot_bf16;
    nk_metric_dense_punned_t dot_f32;
    nk_metric_dense_punned_t dot_f64;
    nk_metric_dense_punned_t dot_f16c;
    nk_metric_dense_punned_t dot_bf16c;
    nk_metric_dense_punned_t dot_f32c;
    nk_metric_dense_punned_t dot_f64c;
    nk_metric_dense_punned_t dot_e4m3;
    nk_metric_dense_punned_t dot_e5m2;
    nk_metric_dense_punned_t dot_e2m3;
    nk_metric_dense_punned_t dot_e3m2;
    nk_metric_dense_punned_t vdot_f16c;
    nk_metric_dense_punned_t vdot_bf16c;
    nk_metric_dense_punned_t vdot_f32c;
    nk_metric_dense_punned_t vdot_f64c;
    // Angular distances
    nk_metric_dense_punned_t angular_i8;
    nk_metric_dense_punned_t angular_u8;
    nk_metric_dense_punned_t angular_i4;
    nk_metric_dense_punned_t angular_u4;
    nk_metric_dense_punned_t angular_f16;
    nk_metric_dense_punned_t angular_bf16;
    nk_metric_dense_punned_t angular_f32;
    nk_metric_dense_punned_t angular_f64;
    nk_metric_dense_punned_t angular_e4m3;
    nk_metric_dense_punned_t angular_e5m2;
    nk_metric_dense_punned_t angular_e2m3;
    nk_metric_dense_punned_t angular_e3m2;
    // Squared Euclidean distances
    nk_metric_dense_punned_t sqeuclidean_i8;
    nk_metric_dense_punned_t sqeuclidean_u8;
    nk_metric_dense_punned_t sqeuclidean_i4;
    nk_metric_dense_punned_t sqeuclidean_u4;
    nk_metric_dense_punned_t sqeuclidean_f16;
    nk_metric_dense_punned_t sqeuclidean_bf16;
    nk_metric_dense_punned_t sqeuclidean_f32;
    nk_metric_dense_punned_t sqeuclidean_f64;
    nk_metric_dense_punned_t sqeuclidean_e4m3;
    nk_metric_dense_punned_t sqeuclidean_e5m2;
    nk_metric_dense_punned_t sqeuclidean_e2m3;
    nk_metric_dense_punned_t sqeuclidean_e3m2;
    // Euclidean distances
    nk_metric_dense_punned_t euclidean_i8;
    nk_metric_dense_punned_t euclidean_u8;
    nk_metric_dense_punned_t euclidean_i4;
    nk_metric_dense_punned_t euclidean_u4;
    nk_metric_dense_punned_t euclidean_f16;
    nk_metric_dense_punned_t euclidean_bf16;
    nk_metric_dense_punned_t euclidean_f32;
    nk_metric_dense_punned_t euclidean_f64;
    nk_metric_dense_punned_t euclidean_e4m3;
    nk_metric_dense_punned_t euclidean_e5m2;
    nk_metric_dense_punned_t euclidean_e2m3;
    nk_metric_dense_punned_t euclidean_e3m2;
    // Geospatial distances
    nk_metric_geospatial_punned_t haversine_f64;
    nk_metric_geospatial_punned_t haversine_f32;
    nk_metric_geospatial_punned_t vincenty_f64;
    nk_metric_geospatial_punned_t vincenty_f32;
    // Binary distances
    nk_metric_dense_punned_t hamming_u1;
    nk_metric_dense_punned_t jaccard_u1;
    nk_metric_dense_punned_t jaccard_u32;
    nk_metric_dense_punned_t hamming_u8;
    nk_metric_dense_punned_t jaccard_u16;
    // Probability distributions
    nk_metric_dense_punned_t kld_f16;
    nk_metric_dense_punned_t kld_bf16;
    nk_metric_dense_punned_t kld_f32;
    nk_metric_dense_punned_t kld_f64;
    nk_metric_dense_punned_t jsd_f16;
    nk_metric_dense_punned_t jsd_bf16;
    nk_metric_dense_punned_t jsd_f32;
    nk_metric_dense_punned_t jsd_f64;
    // Sparse sets
    nk_sparse_intersect_punned_t sparse_intersect_u16;
    nk_sparse_intersect_punned_t sparse_intersect_u32;
    nk_sparse_intersect_punned_t sparse_intersect_u64;
    nk_sparse_dot_punned_t sparse_dot_u16bf16;
    nk_sparse_dot_punned_t sparse_dot_u32f32;
    // Curved spaces
    nk_metric_curved_punned_t bilinear_f64;
    nk_metric_curved_punned_t bilinear_f64c;
    nk_metric_curved_punned_t mahalanobis_f64;
    nk_metric_curved_punned_t bilinear_f32;
    nk_metric_curved_punned_t bilinear_f32c;
    nk_metric_curved_punned_t mahalanobis_f32;
    nk_metric_curved_punned_t bilinear_f16;
    nk_metric_curved_punned_t bilinear_f16c;
    nk_metric_curved_punned_t mahalanobis_f16;
    nk_metric_curved_punned_t bilinear_bf16;
    nk_metric_curved_punned_t bilinear_bf16c;
    nk_metric_curved_punned_t mahalanobis_bf16;
    // Element-wise FMA
    nk_each_fma_punned_t each_fma_f32;
    nk_each_fma_punned_t each_fma_f16;
    nk_each_fma_punned_t each_fma_f64;
    nk_each_fma_punned_t each_fma_bf16;
    nk_each_fma_punned_t each_fma_i8;
    nk_each_fma_punned_t each_fma_u8;
    nk_each_fma_punned_t each_fma_e4m3;
    nk_each_fma_punned_t each_fma_e5m2;
    nk_each_fma_punned_t each_fma_e2m3;
    nk_each_fma_punned_t each_fma_e3m2;
    nk_each_fma_punned_t each_fma_i16;
    nk_each_fma_punned_t each_fma_u16;
    nk_each_fma_punned_t each_fma_i32;
    nk_each_fma_punned_t each_fma_u32;
    nk_each_fma_punned_t each_fma_i64;
    nk_each_fma_punned_t each_fma_u64;
    // Element-wise blend
    nk_each_blend_punned_t each_blend_f64;
    nk_each_blend_punned_t each_blend_f32;
    nk_each_blend_punned_t each_blend_f16;
    nk_each_blend_punned_t each_blend_bf16;
    nk_each_blend_punned_t each_blend_i8;
    nk_each_blend_punned_t each_blend_u8;
    nk_each_blend_punned_t each_blend_e4m3;
    nk_each_blend_punned_t each_blend_e5m2;
    nk_each_blend_punned_t each_blend_e2m3;
    nk_each_blend_punned_t each_blend_e3m2;
    nk_each_blend_punned_t each_blend_i16;
    nk_each_blend_punned_t each_blend_u16;
    nk_each_blend_punned_t each_blend_i32;
    nk_each_blend_punned_t each_blend_u32;
    nk_each_blend_punned_t each_blend_i64;
    nk_each_blend_punned_t each_blend_u64;
    // Element-wise scale
    nk_each_scale_punned_t each_scale_f64;
    nk_each_scale_punned_t each_scale_f32;
    nk_each_scale_punned_t each_scale_f16;
    nk_each_scale_punned_t each_scale_bf16;
    nk_each_scale_punned_t each_scale_i8;
    nk_each_scale_punned_t each_scale_u8;
    nk_each_scale_punned_t each_scale_i16;
    nk_each_scale_punned_t each_scale_u16;
    nk_each_scale_punned_t each_scale_i32;
    nk_each_scale_punned_t each_scale_u32;
    nk_each_scale_punned_t each_scale_i64;
    nk_each_scale_punned_t each_scale_u64;
    nk_each_scale_punned_t each_scale_e4m3;
    nk_each_scale_punned_t each_scale_e5m2;
    nk_each_scale_punned_t each_scale_e2m3;
    nk_each_scale_punned_t each_scale_e3m2;
    // Element-wise sum
    nk_each_sum_punned_t each_sum_f64;
    nk_each_sum_punned_t each_sum_f32;
    nk_each_sum_punned_t each_sum_f16;
    nk_each_sum_punned_t each_sum_bf16;
    nk_each_sum_punned_t each_sum_i8;
    nk_each_sum_punned_t each_sum_u8;
    nk_each_sum_punned_t each_sum_i16;
    nk_each_sum_punned_t each_sum_u16;
    nk_each_sum_punned_t each_sum_i32;
    nk_each_sum_punned_t each_sum_u32;
    nk_each_sum_punned_t each_sum_i64;
    nk_each_sum_punned_t each_sum_u64;
    nk_each_sum_punned_t each_sum_e4m3;
    nk_each_sum_punned_t each_sum_e5m2;
    nk_each_sum_punned_t each_sum_e2m3;
    nk_each_sum_punned_t each_sum_e3m2;
    // Trigonometry
    nk_kernel_trigonometry_punned_t each_sin_f32;
    nk_kernel_trigonometry_punned_t each_sin_f64;
    nk_kernel_trigonometry_punned_t each_cos_f32;
    nk_kernel_trigonometry_punned_t each_cos_f64;
    nk_kernel_trigonometry_punned_t each_atan_f32;
    nk_kernel_trigonometry_punned_t each_atan_f64;
    // Mesh alignment
    nk_metric_mesh_punned_t rmsd_f32;
    nk_metric_mesh_punned_t rmsd_f64;
    nk_metric_mesh_punned_t kabsch_f32;
    nk_metric_mesh_punned_t kabsch_f64;
    nk_metric_mesh_punned_t umeyama_f32;
    nk_metric_mesh_punned_t umeyama_f64;
    // Reduce add
    nk_kernel_reduce_add_punned_t reduce_add_f32;
    nk_kernel_reduce_add_punned_t reduce_add_f64;
    nk_kernel_reduce_add_punned_t reduce_add_i8;
    nk_kernel_reduce_add_punned_t reduce_add_u8;
    nk_kernel_reduce_add_punned_t reduce_add_i16;
    nk_kernel_reduce_add_punned_t reduce_add_u16;
    nk_kernel_reduce_add_punned_t reduce_add_i32;
    nk_kernel_reduce_add_punned_t reduce_add_u32;
    nk_kernel_reduce_add_punned_t reduce_add_i64;
    nk_kernel_reduce_add_punned_t reduce_add_u64;
    nk_kernel_reduce_add_punned_t reduce_add_f16;
    nk_kernel_reduce_add_punned_t reduce_add_bf16;
    nk_kernel_reduce_add_punned_t reduce_add_e4m3;
    nk_kernel_reduce_add_punned_t reduce_add_e5m2;
    nk_kernel_reduce_add_punned_t reduce_add_e2m3;
    nk_kernel_reduce_add_punned_t reduce_add_e3m2;
    // Reduce minmax
    nk_kernel_reduce_minmax_punned_t reduce_min_f32;
    nk_kernel_reduce_minmax_punned_t reduce_max_f32;
    nk_kernel_reduce_minmax_punned_t reduce_min_f64;
    nk_kernel_reduce_minmax_punned_t reduce_max_f64;
    nk_kernel_reduce_minmax_punned_t reduce_min_i8;
    nk_kernel_reduce_minmax_punned_t reduce_max_i8;
    nk_kernel_reduce_minmax_punned_t reduce_min_u8;
    nk_kernel_reduce_minmax_punned_t reduce_max_u8;
    nk_kernel_reduce_minmax_punned_t reduce_min_i16;
    nk_kernel_reduce_minmax_punned_t reduce_max_i16;
    nk_kernel_reduce_minmax_punned_t reduce_min_u16;
    nk_kernel_reduce_minmax_punned_t reduce_max_u16;
    nk_kernel_reduce_minmax_punned_t reduce_min_i32;
    nk_kernel_reduce_minmax_punned_t reduce_max_i32;
    nk_kernel_reduce_minmax_punned_t reduce_min_u32;
    nk_kernel_reduce_minmax_punned_t reduce_max_u32;
    nk_kernel_reduce_minmax_punned_t reduce_min_i64;
    nk_kernel_reduce_minmax_punned_t reduce_max_i64;
    nk_kernel_reduce_minmax_punned_t reduce_min_u64;
    nk_kernel_reduce_minmax_punned_t reduce_max_u64;
    nk_kernel_reduce_minmax_punned_t reduce_min_f16;
    nk_kernel_reduce_minmax_punned_t reduce_max_f16;
    nk_kernel_reduce_minmax_punned_t reduce_min_bf16;
    nk_kernel_reduce_minmax_punned_t reduce_max_bf16;
    nk_kernel_reduce_minmax_punned_t reduce_min_e4m3;
    nk_kernel_reduce_minmax_punned_t reduce_max_e4m3;
    nk_kernel_reduce_minmax_punned_t reduce_min_e5m2;
    nk_kernel_reduce_minmax_punned_t reduce_max_e5m2;
    nk_kernel_reduce_minmax_punned_t reduce_min_e2m3;
    nk_kernel_reduce_minmax_punned_t reduce_max_e2m3;
    nk_kernel_reduce_minmax_punned_t reduce_min_e3m2;
    nk_kernel_reduce_minmax_punned_t reduce_max_e3m2;
    // Dots packed size
    nk_dots_packed_size_punned_t dots_packed_size_f32;
    nk_dots_packed_size_punned_t dots_packed_size_f64;
    nk_dots_packed_size_punned_t dots_packed_size_f16;
    nk_dots_packed_size_punned_t dots_packed_size_bf16;
    nk_dots_packed_size_punned_t dots_packed_size_i8;
    nk_dots_packed_size_punned_t dots_packed_size_u8;
    nk_dots_packed_size_punned_t dots_packed_size_e4m3;
    nk_dots_packed_size_punned_t dots_packed_size_e5m2;
    nk_dots_packed_size_punned_t dots_packed_size_e2m3;
    nk_dots_packed_size_punned_t dots_packed_size_e3m2;
    nk_dots_packed_size_punned_t dots_packed_size_u1;
    nk_dots_packed_size_punned_t dots_packed_size_u4;
    nk_dots_packed_size_punned_t dots_packed_size_i4;
    // Dots pack
    nk_dots_pack_punned_t dots_pack_f32;
    nk_dots_pack_punned_t dots_pack_f64;
    nk_dots_pack_punned_t dots_pack_f16;
    nk_dots_pack_punned_t dots_pack_bf16;
    nk_dots_pack_punned_t dots_pack_i8;
    nk_dots_pack_punned_t dots_pack_u8;
    nk_dots_pack_punned_t dots_pack_e4m3;
    nk_dots_pack_punned_t dots_pack_e5m2;
    nk_dots_pack_punned_t dots_pack_e2m3;
    nk_dots_pack_punned_t dots_pack_e3m2;
    nk_dots_pack_punned_t dots_pack_u1;
    nk_dots_pack_punned_t dots_pack_u4;
    nk_dots_pack_punned_t dots_pack_i4;
    // Dots packed
    nk_dots_punned_t dots_packed_f32;
    nk_dots_punned_t dots_packed_f64;
    nk_dots_punned_t dots_packed_f16;
    nk_dots_punned_t dots_packed_bf16;
    nk_dots_punned_t dots_packed_i8;
    nk_dots_punned_t dots_packed_u8;
    nk_dots_punned_t dots_packed_e4m3;
    nk_dots_punned_t dots_packed_e5m2;
    nk_dots_punned_t dots_packed_e2m3;
    nk_dots_punned_t dots_packed_e3m2;
    nk_dots_punned_t dots_packed_u1;
    nk_dots_punned_t dots_packed_u4;
    nk_dots_punned_t dots_packed_i4;
    // Dots symmetric
    nk_dots_symmetric_punned_t dots_symmetric_f32;
    nk_dots_symmetric_punned_t dots_symmetric_f64;
    nk_dots_symmetric_punned_t dots_symmetric_f16;
    nk_dots_symmetric_punned_t dots_symmetric_bf16;
    nk_dots_symmetric_punned_t dots_symmetric_i8;
    nk_dots_symmetric_punned_t dots_symmetric_u8;
    nk_dots_symmetric_punned_t dots_symmetric_e4m3;
    nk_dots_symmetric_punned_t dots_symmetric_e5m2;
    nk_dots_symmetric_punned_t dots_symmetric_e2m3;
    nk_dots_symmetric_punned_t dots_symmetric_e3m2;
    nk_dots_symmetric_punned_t dots_symmetric_u4;
    nk_dots_symmetric_punned_t dots_symmetric_i4;
    // Hammings
    nk_hammings_packed_size_punned_t hammings_packed_size_u1;
    nk_hammings_pack_punned_t hammings_pack_u1;
    nk_hammings_punned_t hammings_packed_u1;
    nk_hammings_symmetric_punned_t hammings_symmetric_u1;
    // Type casting
    nk_kernel_cast_punned_t cast;
    // Scalar conversions (single element)
    void (*f16_to_f32)(nk_f16_t const *, nk_f32_t *);
    void (*f32_to_f16)(nk_f32_t const *, nk_f16_t *);
    void (*bf16_to_f32)(nk_bf16_t const *, nk_f32_t *);
    void (*f32_to_bf16)(nk_f32_t const *, nk_bf16_t *);
    void (*e4m3_to_f32)(nk_e4m3_t const *, nk_f32_t *);
    void (*f32_to_e4m3)(nk_f32_t const *, nk_e4m3_t *);
    void (*e5m2_to_f32)(nk_e5m2_t const *, nk_f32_t *);
    void (*f32_to_e5m2)(nk_f32_t const *, nk_e5m2_t *);
    void (*e2m3_to_f32)(nk_e2m3_t const *, nk_f32_t *);
    void (*f32_to_e2m3)(nk_f32_t const *, nk_e2m3_t *);
    void (*e3m2_to_f32)(nk_e3m2_t const *, nk_f32_t *);
    void (*f32_to_e3m2)(nk_f32_t const *, nk_e3m2_t *);
} nk_implementations_t;

// Global dispatch table - defined in numkong.c
extern nk_implementations_t nk_dispatch_table;

// Error handlers - defined in numkong.c
extern void nk_error_dense_(void const *, void const *, nk_size_t, void *);
extern void nk_error_sparse_intersect_(void const *, void const *, nk_size_t, nk_size_t, void *, nk_size_t *);
extern void nk_error_sparse_dot_(void const *, void const *, void const *, void const *, nk_size_t, nk_size_t, void *);
extern void nk_error_curved_(void const *, void const *, void const *, nk_size_t, void *);
extern void nk_error_geospatial_(void const *, void const *, void const *, void const *, nk_size_t, void *);
extern void nk_error_each_fma_(void const *, void const *, void const *, nk_size_t, void const *, void const *, void *);
extern void nk_error_each_blend_(void const *, void const *, nk_size_t, void const *, void const *, void *);
extern void nk_error_each_scale_(void const *, nk_size_t, void const *, void const *, void *);
extern void nk_error_each_sum_(void const *, void const *, nk_size_t, void *);
extern void nk_error_trigonometry_(void const *, nk_size_t, void *);
extern void nk_error_mesh_(void const *, void const *, nk_size_t, void *, void *, void *, void *, void *);
extern void nk_error_reduce_add_(void const *, nk_size_t, nk_size_t, void *);
extern void nk_error_reduce_minmax_(void const *, nk_size_t, nk_size_t, void *, nk_size_t *);
extern nk_size_t nk_error_packed_size_(nk_size_t, nk_size_t);
extern void nk_error_pack_(void const *, nk_size_t, nk_size_t, nk_size_t, void *);
extern void nk_error_dots_(void const *, void const *, void *, nk_size_t, nk_size_t, nk_size_t, nk_size_t, nk_size_t);
extern void nk_error_dots_symmetric_(void const *, nk_size_t, nk_size_t, nk_size_t, void *, nk_size_t, nk_size_t,
                                     nk_size_t);

// Dtype-specific kernel lookup functions
extern void nk_dispatch_f64_find_(nk_capability_t, nk_kernel_kind_t, nk_kernel_punned_t *, nk_capability_t *);
extern void nk_dispatch_f32_find_(nk_capability_t, nk_kernel_kind_t, nk_kernel_punned_t *, nk_capability_t *);
extern void nk_dispatch_f16_find_(nk_capability_t, nk_kernel_kind_t, nk_kernel_punned_t *, nk_capability_t *);
extern void nk_dispatch_bf16_find_(nk_capability_t, nk_kernel_kind_t, nk_kernel_punned_t *, nk_capability_t *);
extern void nk_dispatch_i8_find_(nk_capability_t, nk_kernel_kind_t, nk_kernel_punned_t *, nk_capability_t *);
extern void nk_dispatch_u8_find_(nk_capability_t, nk_kernel_kind_t, nk_kernel_punned_t *, nk_capability_t *);
extern void nk_dispatch_i4_find_(nk_capability_t, nk_kernel_kind_t, nk_kernel_punned_t *, nk_capability_t *);
extern void nk_dispatch_u4_find_(nk_capability_t, nk_kernel_kind_t, nk_kernel_punned_t *, nk_capability_t *);
extern void nk_dispatch_e4m3_find_(nk_capability_t, nk_kernel_kind_t, nk_kernel_punned_t *, nk_capability_t *);
extern void nk_dispatch_e5m2_find_(nk_capability_t, nk_kernel_kind_t, nk_kernel_punned_t *, nk_capability_t *);
extern void nk_dispatch_e2m3_find_(nk_capability_t, nk_kernel_kind_t, nk_kernel_punned_t *, nk_capability_t *);
extern void nk_dispatch_e3m2_find_(nk_capability_t, nk_kernel_kind_t, nk_kernel_punned_t *, nk_capability_t *);
extern void nk_dispatch_u1_find_(nk_capability_t, nk_kernel_kind_t, nk_kernel_punned_t *, nk_capability_t *);
extern void nk_dispatch_f64c_find_(nk_capability_t, nk_kernel_kind_t, nk_kernel_punned_t *, nk_capability_t *);
extern void nk_dispatch_f32c_find_(nk_capability_t, nk_kernel_kind_t, nk_kernel_punned_t *, nk_capability_t *);
extern void nk_dispatch_f16c_find_(nk_capability_t, nk_kernel_kind_t, nk_kernel_punned_t *, nk_capability_t *);
extern void nk_dispatch_bf16c_find_(nk_capability_t, nk_kernel_kind_t, nk_kernel_punned_t *, nk_capability_t *);
extern void nk_dispatch_i16_find_(nk_capability_t, nk_kernel_kind_t, nk_kernel_punned_t *, nk_capability_t *);
extern void nk_dispatch_u16_find_(nk_capability_t, nk_kernel_kind_t, nk_kernel_punned_t *, nk_capability_t *);
extern void nk_dispatch_i32_find_(nk_capability_t, nk_kernel_kind_t, nk_kernel_punned_t *, nk_capability_t *);
extern void nk_dispatch_u32_find_(nk_capability_t, nk_kernel_kind_t, nk_kernel_punned_t *, nk_capability_t *);
extern void nk_dispatch_i64_find_(nk_capability_t, nk_kernel_kind_t, nk_kernel_punned_t *, nk_capability_t *);
extern void nk_dispatch_u64_find_(nk_capability_t, nk_kernel_kind_t, nk_kernel_punned_t *, nk_capability_t *);
extern void nk_dispatch_cast_find_(nk_capability_t, nk_kernel_kind_t, nk_kernel_punned_t *, nk_capability_t *);

#ifdef __cplusplus
}
#endif

#endif // NK_DISPATCH_H

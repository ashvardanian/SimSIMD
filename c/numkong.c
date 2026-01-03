/**
 *  @brief  Dynamic dispatch library for NumKong.
 *  @note   Compile with the most recent compiler available.
 *  @file   lib.c
 */
#define NK_DYNAMIC_DISPATCH 1
#define NK_NATIVE_F16       0
#define NK_NATIVE_BF16      0

/*  Override the primary serial operations to avoid the LibC dependency.
 */
#define NK_F32_SQRT(x)  nk_f32_approximate_square_root(x)
#define NK_F32_RSQRT(x) nk_f32_approximate_inverse_square_root(x)
#define NK_F32_LOG(x)   nk_f32_approximate_log(x)

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
#if !defined(NK_TARGET_ICE) && (defined(_MSC_VER) || defined(__linux__))
#define NK_TARGET_ICE 1
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

// Every time a function is called, it checks if the metric is already loaded. If not, it fetches it.
// If no metric is found, it returns NaN. We can obtain NaN by dividing 0.0 by 0.0, but that annoys
// the MSVC compiler. Instead we can directly write-in the signaling NaN (0x7FF0000000000001)
// or the qNaN (0x7FF8000000000000).
#define nk_declare_dense_(name, extension, output_type)                                                            \
    NK_DYNAMIC void nk_##name##_##extension(nk_##extension##_t const *a, nk_##extension##_t const *b, nk_size_t n, \
                                            nk_##output_type##_t *results) {                                       \
        static nk_metric_dense_punned_t metric = 0;                                                                \
        if (metric == 0) {                                                                                         \
            nk_capability_t used_capability;                                                                       \
            nk_find_kernel_punned(nk_kernel_##name##_k, nk_##extension##_k, nk_capabilities(), nk_cap_any_k,       \
                                  (nk_kernel_punned_t *)&metric, &used_capability);                                \
            if (!metric) {                                                                                         \
                *(nk_u64_t *)results = 0x7FF0000000000001ull;                                                      \
                return;                                                                                            \
            }                                                                                                      \
        }                                                                                                          \
        metric(a, b, n, (void *)results);                                                                          \
    }

#define nk_declare_sparse_(name, extension, type, output_type)                                                      \
    NK_DYNAMIC void nk_##name##_##extension(nk_##type##_t const *a, nk_##type##_t const *b, nk_size_t a_length,     \
                                            nk_size_t b_length, nk_##output_type##_t *result) {                     \
        static nk_sparse_intersect_punned_t metric = 0;                                                             \
        if (metric == 0) {                                                                                          \
            nk_capability_t used_capability;                                                                        \
            nk_find_kernel_punned(nk_kernel_sparse_##name##_k, nk_##extension##_k, nk_capabilities(), nk_cap_any_k, \
                                  (nk_kernel_punned_t *)(&metric), &used_capability);                               \
            if (!metric) {                                                                                          \
                *(nk_u64_t *)result = 0x7FF0000000000001ull;                                                        \
                return;                                                                                             \
            }                                                                                                       \
        }                                                                                                           \
        metric(a, b, a_length, b_length, (void *)result);                                                           \
    }

#define nk_declare_sparse_dot_(name, index_type, weight_type, output_type)                                            \
    NK_DYNAMIC void nk_##name##_##index_type##weight_type(nk_##index_type##_t const *a, nk_##index_type##_t const *b, \
                                                          nk_##weight_type##_t const *a_weights,                      \
                                                          nk_##weight_type##_t const *b_weights, nk_size_t a_length,  \
                                                          nk_size_t b_length, nk_##output_type##_t *product) {        \
        static nk_sparse_dot_punned_t metric = 0;                                                                     \
        if (metric == 0) {                                                                                            \
            nk_capability_t used_capability;                                                                          \
            nk_find_kernel_punned(nk_kernel_sparse_dot_k, nk_##weight_type##_k, nk_capabilities(), nk_cap_any_k,      \
                                  (nk_kernel_punned_t *)&metric, &used_capability);                                   \
            if (!metric) {                                                                                            \
                *(nk_u64_t *)product = 0x7FF0000000000001ull;                                                         \
                return;                                                                                               \
            }                                                                                                         \
        }                                                                                                             \
        metric(a, b, a_weights, b_weights, a_length, b_length, (void *)product);                                      \
    }

#define nk_declare_curved_(name, extension, output_type)                                                              \
    NK_DYNAMIC void nk_##name##_##extension(nk_##extension##_t const *a, nk_##extension##_t const *b,                 \
                                            nk_##extension##_t const *c, nk_size_t n, nk_##output_type##_t *result) { \
        static nk_metric_curved_punned_t metric = 0;                                                                  \
        if (metric == 0) {                                                                                            \
            nk_capability_t used_capability;                                                                          \
            nk_find_kernel_punned(nk_kernel_##name##_k, nk_##extension##_k, nk_capabilities(), nk_cap_any_k,          \
                                  (nk_kernel_punned_t *)(&metric), &used_capability);                                 \
            if (!metric) {                                                                                            \
                *(nk_u64_t *)result = 0x7FF0000000000001ull;                                                          \
                return;                                                                                               \
            }                                                                                                         \
        }                                                                                                             \
        metric(a, b, c, n, (void *)result);                                                                           \
    }

#define nk_declare_geospatial_(name, extension, output_type)                                                    \
    NK_DYNAMIC void nk_##name##_##extension(nk_##extension##_t const *a_lats, nk_##extension##_t const *a_lons, \
                                            nk_##extension##_t const *b_lats, nk_##extension##_t const *b_lons, \
                                            nk_size_t n, nk_##output_type##_t *results) {                       \
        static nk_metric_geospatial_punned_t metric = 0;                                                        \
        if (metric == 0) {                                                                                      \
            nk_capability_t used_capability;                                                                    \
            nk_find_kernel_punned(nk_kernel_##name##_k, nk_##extension##_k, nk_capabilities(), nk_cap_any_k,    \
                                  (nk_kernel_punned_t *)(&metric), &used_capability);                           \
            if (!metric) {                                                                                      \
                *(nk_u64_t *)results = 0x7FF0000000000001ull;                                                   \
                return;                                                                                         \
            }                                                                                                   \
        }                                                                                                       \
        metric(a_lats, a_lons, b_lats, b_lons, n, (void *)results);                                             \
    }

#define nk_declare_fma_(name, extension, scalar_type)                                                        \
    NK_DYNAMIC void nk_##name##_##extension(                                                                 \
        nk_##extension##_t const *a, nk_##extension##_t const *b, nk_##extension##_t const *c, nk_size_t n,  \
        nk_##scalar_type##_t const *alpha, nk_##scalar_type##_t const *beta, nk_##extension##_t *result) {   \
        static nk_kernel_fma_punned_t metric = 0;                                                            \
        if (metric == 0) {                                                                                   \
            nk_capability_t used_capability;                                                                 \
            nk_find_kernel_punned(nk_kernel_##name##_k, nk_##extension##_k, nk_capabilities(), nk_cap_any_k, \
                                  (nk_kernel_punned_t *)(&metric), &used_capability);                        \
        }                                                                                                    \
        metric(a, b, c, n, (void const *)alpha, (void const *)beta, result);                                 \
    }

#define nk_declare_wsum_(name, extension, scalar_type)                                                             \
    NK_DYNAMIC void nk_##name##_##extension(nk_##extension##_t const *a, nk_##extension##_t const *b, nk_size_t n, \
                                            nk_##scalar_type##_t const *alpha, nk_##scalar_type##_t const *beta,   \
                                            nk_##extension##_t *result) {                                          \
        static nk_kernel_wsum_punned_t metric = 0;                                                                 \
        if (metric == 0) {                                                                                         \
            nk_capability_t used_capability;                                                                       \
            nk_find_kernel_punned(nk_kernel_##name##_k, nk_##extension##_k, nk_capabilities(), nk_cap_any_k,       \
                                  (nk_kernel_punned_t *)(&metric), &used_capability);                              \
        }                                                                                                          \
        metric(a, b, n, (void const *)alpha, (void const *)beta, result);                                          \
    }

#define nk_declare_scale_(name, extension, scalar_type)                                                          \
    NK_DYNAMIC void nk_##name##_##extension(nk_##extension##_t const *a, nk_size_t n,                            \
                                            nk_##scalar_type##_t const *alpha, nk_##scalar_type##_t const *beta, \
                                            nk_##extension##_t *result) {                                        \
        static nk_kernel_scale_punned_t metric = 0;                                                              \
        if (metric == 0) {                                                                                       \
            nk_capability_t used_capability;                                                                     \
            nk_find_kernel_punned(nk_kernel_##name##_k, nk_##extension##_k, nk_capabilities(), nk_cap_any_k,     \
                                  (nk_kernel_punned_t *)(&metric), &used_capability);                            \
        }                                                                                                        \
        metric(a, n, (void const *)alpha, (void const *)beta, result);                                           \
    }

#define nk_declare_sum_(name, extension)                                                                           \
    NK_DYNAMIC void nk_##name##_##extension(nk_##extension##_t const *a, nk_##extension##_t const *b, nk_size_t n, \
                                            nk_##extension##_t *result) {                                          \
        static nk_kernel_sum_punned_t metric = 0;                                                                  \
        if (metric == 0) {                                                                                         \
            nk_capability_t used_capability;                                                                       \
            nk_find_kernel_punned(nk_kernel_##name##_k, nk_##extension##_k, nk_capabilities(), nk_cap_any_k,       \
                                  (nk_kernel_punned_t *)(&metric), &used_capability);                              \
        }                                                                                                          \
        metric(a, b, n, result);                                                                                   \
    }

#define nk_declare_trigonometry_(name, extension)                                                            \
    NK_DYNAMIC void nk_##name##_##extension(nk_##extension##_t const *inputs, nk_size_t n,                   \
                                            nk_##extension##_t *outputs) {                                   \
        static nk_kernel_trigonometry_punned_t kernel = 0;                                                   \
        if (kernel == 0) {                                                                                   \
            nk_capability_t used_capability;                                                                 \
            nk_find_kernel_punned(nk_kernel_##name##_k, nk_##extension##_k, nk_capabilities(), nk_cap_any_k, \
                                  (nk_kernel_punned_t *)(&kernel), &used_capability);                        \
        }                                                                                                    \
        kernel(inputs, n, outputs);                                                                          \
    }

#define nk_declare_mesh_(name, extension, mesh_type)                                                               \
    NK_DYNAMIC void nk_##name##_##extension(nk_##extension##_t const *a, nk_##extension##_t const *b, nk_size_t n, \
                                            nk_##mesh_type##_t *a_centroid, nk_##mesh_type##_t *b_centroid,        \
                                            nk_##mesh_type##_t *rotation, nk_##mesh_type##_t *scale,               \
                                            nk_##mesh_type##_t *result) {                                          \
        static nk_metric_mesh_punned_t kernel = 0;                                                                 \
        if (kernel == 0) {                                                                                         \
            nk_capability_t used_capability;                                                                       \
            nk_find_kernel_punned(nk_kernel_##name##_k, nk_##extension##_k, nk_capabilities(), nk_cap_any_k,       \
                                  (nk_kernel_punned_t *)&kernel, &used_capability);                                \
            if (!kernel) {                                                                                         \
                *(nk_u64_t *)result = 0x7FF0000000000001ull;                                                       \
                return;                                                                                            \
            }                                                                                                      \
        }                                                                                                          \
        kernel(a, b, n, (void *)a_centroid, (void *)b_centroid, (void *)rotation, (void *)scale, (void *)result);  \
    }

#define nk_declare_reduce_add_(extension, output_type)                                                                 \
    NK_DYNAMIC void nk_reduce_add_##extension(nk_##extension##_t const *data, nk_size_t count, nk_size_t stride_bytes, \
                                              nk_##output_type##_t *result) {                                          \
        static nk_kernel_reduce_add_punned_t kernel = 0;                                                               \
        if (kernel == 0) {                                                                                             \
            nk_capability_t used_capability;                                                                           \
            nk_find_kernel_punned(nk_kernel_reduce_add_k, nk_##extension##_k, nk_capabilities(), nk_cap_any_k,         \
                                  (nk_kernel_punned_t *)&kernel, &used_capability);                                    \
        }                                                                                                              \
        kernel(data, count, stride_bytes, result);                                                                     \
    }

#define nk_declare_reduce_minmax_(name, extension, output_type)                                                     \
    NK_DYNAMIC void nk_reduce_##name##_##extension(nk_##extension##_t const *data, nk_size_t count,                 \
                                                   nk_size_t stride_bytes, nk_##output_type##_t *value,             \
                                                   nk_size_t *index) {                                              \
        static nk_kernel_reduce_minmax_punned_t kernel = 0;                                                         \
        if (kernel == 0) {                                                                                          \
            nk_capability_t used_capability;                                                                        \
            nk_find_kernel_punned(nk_kernel_reduce_##name##_k, nk_##extension##_k, nk_capabilities(), nk_cap_any_k, \
                                  (nk_kernel_punned_t *)&kernel, &used_capability);                                 \
        }                                                                                                           \
        kernel(data, count, stride_bytes, value, index);                                                            \
    }

#define nk_declare_dots_packed_size_(input_type, accum_type)                                                          \
    NK_DYNAMIC nk_size_t nk_dots_##input_type##input_type##accum_type##_packed_size(nk_size_t n, nk_size_t k) {       \
        static nk_dots_packed_size_punned_t kernel = 0;                                                               \
        if (kernel == 0) {                                                                                            \
            nk_capability_t used_capability;                                                                          \
            nk_find_kernel_punned(nk_kernel_dots_packed_size_k, nk_##input_type##_k, nk_capabilities(), nk_cap_any_k, \
                                  (nk_kernel_punned_t *)&kernel, &used_capability);                                   \
            if (!kernel) return 0;                                                                                    \
        }                                                                                                             \
        return kernel(n, k);                                                                                          \
    }

#define nk_declare_dots_pack_(input_type, accum_type)                                                          \
    NK_DYNAMIC void nk_dots_##input_type##input_type##accum_type##_pack(                                       \
        nk_##input_type##_t const *b, nk_size_t n, nk_size_t k, nk_size_t b_stride, void *b_packed) {          \
        static nk_dots_pack_punned_t kernel = 0;                                                               \
        if (kernel == 0) {                                                                                     \
            nk_capability_t used_capability;                                                                   \
            nk_find_kernel_punned(nk_kernel_dots_pack_k, nk_##input_type##_k, nk_capabilities(), nk_cap_any_k, \
                                  (nk_kernel_punned_t *)&kernel, &used_capability);                            \
            if (!kernel) return;                                                                               \
        }                                                                                                      \
        kernel(b, n, k, b_stride, b_packed);                                                                   \
    }

#define nk_declare_dots_(input_type, accum_type, output_type)                                                  \
    NK_DYNAMIC void nk_dots_##input_type##input_type##accum_type(                                              \
        nk_##input_type##_t const *a, void const *b_packed, nk_##output_type##_t *c, nk_size_t m, nk_size_t n, \
        nk_size_t k, nk_size_t a_stride, nk_size_t c_stride) {                                                 \
        static nk_dots_punned_t kernel = 0;                                                                    \
        if (kernel == 0) {                                                                                     \
            nk_capability_t used_capability;                                                                   \
            nk_find_kernel_punned(nk_kernel_dots_k, nk_##input_type##_k, nk_capabilities(), nk_cap_any_k,      \
                                  (nk_kernel_punned_t *)&kernel, &used_capability);                            \
            if (!kernel) return;                                                                               \
        }                                                                                                      \
        kernel(a, b_packed, c, m, n, k, a_stride, c_stride);                                                   \
    }

// Dot products
nk_declare_dense_(dot, i8, i32)
nk_declare_dense_(dot, u8, u32)
nk_declare_dense_(dot, f16, f32)
nk_declare_dense_(dot, bf16, f32)
nk_declare_dense_(dot, f32, f32)
nk_declare_dense_(dot, f64, f64)
nk_declare_dense_(dot, f16c, f32c)
nk_declare_dense_(dot, bf16c, f32c)
nk_declare_dense_(dot, f32c, f32c)
nk_declare_dense_(dot, f64c, f64c)
nk_declare_dense_(dot, e4m3, f32)
nk_declare_dense_(dot, e5m2, f32)
nk_declare_dense_(vdot, f16c, f32c)
nk_declare_dense_(vdot, bf16c, f32c)
nk_declare_dense_(vdot, f32c, f32c)
nk_declare_dense_(vdot, f64c, f64c)

// Spatial distances
nk_declare_dense_(angular, i8, f32)
nk_declare_dense_(angular, u8, f32)
nk_declare_dense_(angular, f16, f32)
nk_declare_dense_(angular, bf16, f32)
nk_declare_dense_(angular, f32, f32)
nk_declare_dense_(angular, f64, f64)
nk_declare_dense_(angular, e4m3, f32)
nk_declare_dense_(angular, e5m2, f32)
nk_declare_dense_(l2sq, i8, u32)
nk_declare_dense_(l2sq, u8, u32)
nk_declare_dense_(l2sq, f16, f32)
nk_declare_dense_(l2sq, bf16, f32)
nk_declare_dense_(l2sq, f32, f32)
nk_declare_dense_(l2sq, f64, f64)
nk_declare_dense_(l2sq, e4m3, f32)
nk_declare_dense_(l2sq, e5m2, f32)
nk_declare_dense_(l2, i8, f32)
nk_declare_dense_(l2, u8, f32)
nk_declare_dense_(l2, f16, f32)
nk_declare_dense_(l2, bf16, f32)
nk_declare_dense_(l2, f32, f32)
nk_declare_dense_(l2, f64, f64)
nk_declare_dense_(l2, e4m3, f32)
nk_declare_dense_(l2, e5m2, f32)

// Geospatial distances
nk_declare_geospatial_(haversine, f64, f64)
nk_declare_geospatial_(haversine, f32, f32)
nk_declare_geospatial_(vincenty, f64, f64)
nk_declare_geospatial_(vincenty, f32, f32)

// Binary distances
nk_declare_dense_(hamming, b8, u32)
nk_declare_dense_(jaccard, b8, f32)
nk_declare_dense_(jaccard, u32, f32)

// Probability distributions
nk_declare_dense_(kld, f16, f32)
nk_declare_dense_(kld, bf16, f32)
nk_declare_dense_(kld, f32, f32)
nk_declare_dense_(kld, f64, f64)
nk_declare_dense_(jsd, f16, f32)
nk_declare_dense_(jsd, bf16, f32)
nk_declare_dense_(jsd, f32, f32)
nk_declare_dense_(jsd, f64, f64)

// Sparse sets
nk_declare_sparse_(intersect, u16, u16, u32)
nk_declare_sparse_(intersect, u32, u32, u32)
nk_declare_sparse_dot_(sparse_dot, u16, bf16, f32)
nk_declare_sparse_dot_(sparse_dot, u32, f32, f32)

// Curved spaces
nk_declare_curved_(bilinear, f64, f64)
nk_declare_curved_(bilinear, f64c, f64c)
nk_declare_curved_(mahalanobis, f64, f64)
nk_declare_curved_(bilinear, f32, f32)
nk_declare_curved_(bilinear, f32c, f32c)
nk_declare_curved_(mahalanobis, f32, f32)
nk_declare_curved_(bilinear, f16, f32)
nk_declare_curved_(bilinear, f16c, f32c)
nk_declare_curved_(mahalanobis, f16, f32)
nk_declare_curved_(bilinear, bf16, f32)
nk_declare_curved_(bilinear, bf16c, f32c)
nk_declare_curved_(mahalanobis, bf16, f32)

// Element-wise operations
nk_declare_fma_(fma, f64, f64)
nk_declare_fma_(fma, f32, f32)
nk_declare_fma_(fma, f16, f32)
nk_declare_fma_(fma, bf16, f32)
nk_declare_fma_(fma, i8, f32)
nk_declare_fma_(fma, u8, f32)
nk_declare_wsum_(wsum, f64, f64)
nk_declare_wsum_(wsum, f32, f32)
nk_declare_wsum_(wsum, f16, f32)
nk_declare_wsum_(wsum, bf16, f32)
nk_declare_wsum_(wsum, i8, f32)
nk_declare_wsum_(wsum, u8, f32)
nk_declare_scale_(scale, f64, f64)
nk_declare_scale_(scale, f32, f32)
nk_declare_scale_(scale, f16, f32)
nk_declare_scale_(scale, bf16, f32)
nk_declare_scale_(scale, i8, f32)
nk_declare_scale_(scale, u8, f32)
nk_declare_scale_(scale, i16, f32)
nk_declare_scale_(scale, u16, f32)
nk_declare_scale_(scale, i32, f64)
nk_declare_scale_(scale, u32, f64)
nk_declare_scale_(scale, i64, f64)
nk_declare_scale_(scale, u64, f64)
nk_declare_sum_(sum, f64)
nk_declare_sum_(sum, f32)
nk_declare_sum_(sum, f16)
nk_declare_sum_(sum, bf16)
nk_declare_sum_(sum, i8)
nk_declare_sum_(sum, u8)
nk_declare_sum_(sum, i16)
nk_declare_sum_(sum, u16)
nk_declare_sum_(sum, i32)
nk_declare_sum_(sum, u32)
nk_declare_sum_(sum, i64)
nk_declare_sum_(sum, u64)

// Trigonometry functions
nk_declare_trigonometry_(sin, f32)
nk_declare_trigonometry_(sin, f64)
nk_declare_trigonometry_(cos, f32)
nk_declare_trigonometry_(cos, f64)
nk_declare_trigonometry_(atan, f32)
nk_declare_trigonometry_(atan, f64)

// Mesh alignment (RMSD, Kabsch, Umeyama)
nk_declare_mesh_(rmsd, f32, f32)
nk_declare_mesh_(rmsd, f64, f64)
nk_declare_mesh_(kabsch, f32, f32)
nk_declare_mesh_(kabsch, f64, f64)
nk_declare_mesh_(umeyama, f32, f32)
nk_declare_mesh_(umeyama, f64, f64)

// Horizontal reductions - floating point
nk_declare_reduce_add_(f32, f64)
nk_declare_reduce_add_(f64, f64)
nk_declare_reduce_minmax_(min, f32, f32)
nk_declare_reduce_minmax_(max, f32, f32)
nk_declare_reduce_minmax_(min, f64, f64)
nk_declare_reduce_minmax_(max, f64, f64)
// Horizontal reductions - integers (output widened for sum)
nk_declare_reduce_add_(i8, i64)
nk_declare_reduce_add_(u8, u64)
nk_declare_reduce_add_(i16, i64)
nk_declare_reduce_add_(u16, u64)
nk_declare_reduce_add_(i32, i64)
nk_declare_reduce_add_(u32, u64)
nk_declare_reduce_add_(i64, i64)
nk_declare_reduce_add_(u64, u64)
nk_declare_reduce_minmax_(min, i8, i8)
nk_declare_reduce_minmax_(max, i8, i8)
nk_declare_reduce_minmax_(min, u8, u8)
nk_declare_reduce_minmax_(max, u8, u8)
nk_declare_reduce_minmax_(min, i16, i16)
nk_declare_reduce_minmax_(max, i16, i16)
nk_declare_reduce_minmax_(min, u16, u16)
nk_declare_reduce_minmax_(max, u16, u16)
nk_declare_reduce_minmax_(min, i32, i32)
nk_declare_reduce_minmax_(max, i32, i32)
nk_declare_reduce_minmax_(min, u32, u32)
nk_declare_reduce_minmax_(max, u32, u32)
nk_declare_reduce_minmax_(min, i64, i64)
nk_declare_reduce_minmax_(max, i64, i64)
nk_declare_reduce_minmax_(min, u64, u64)
nk_declare_reduce_minmax_(max, u64, u64)
// Horizontal reductions - half-precision types (output widened to f32)
nk_declare_reduce_add_(f16, f32)
nk_declare_reduce_add_(bf16, f32)
nk_declare_reduce_add_(e4m3, f32)
nk_declare_reduce_add_(e5m2, f32)
nk_declare_reduce_minmax_(min, f16, f32)
nk_declare_reduce_minmax_(max, f16, f32)
nk_declare_reduce_minmax_(min, bf16, f32)
nk_declare_reduce_minmax_(max, bf16, f32)
nk_declare_reduce_minmax_(min, e4m3, f32)
nk_declare_reduce_minmax_(max, e4m3, f32)
nk_declare_reduce_minmax_(min, e5m2, f32)
nk_declare_reduce_minmax_(max, e5m2, f32)
// Elementwise operations - FP8 types
nk_declare_sum_(sum, e4m3)
nk_declare_sum_(sum, e5m2)
nk_declare_scale_(scale, e4m3, f32)
nk_declare_scale_(scale, e5m2, f32)
nk_declare_wsum_(wsum, e4m3, f32)
nk_declare_wsum_(wsum, e5m2, f32)
nk_declare_fma_(fma, e4m3, f32)
nk_declare_fma_(fma, e5m2, f32)

// Matrix multiplications (GEMM with packed B)
nk_declare_dots_packed_size_(f32, f32)
nk_declare_dots_packed_size_(f64, f64)
nk_declare_dots_packed_size_(f16, f32)
nk_declare_dots_packed_size_(bf16, f32)
nk_declare_dots_packed_size_(i8, i32)
nk_declare_dots_packed_size_(u8, u32)
nk_declare_dots_pack_(f32, f32)
nk_declare_dots_pack_(f64, f64)
nk_declare_dots_pack_(f16, f32)
nk_declare_dots_pack_(bf16, f32)
nk_declare_dots_pack_(i8, i32)
nk_declare_dots_pack_(u8, u32)
nk_declare_dots_(f32, f32, f32)
nk_declare_dots_(f64, f64, f64)
nk_declare_dots_(f16, f32, f32)
nk_declare_dots_(bf16, f32, f32)
nk_declare_dots_(i8, i32, i32)
nk_declare_dots_(u8, u32, u32)

// ARM NEON capabilities
NK_DYNAMIC int nk_uses_neon(void) { return (nk_capabilities() & nk_cap_neon_k) != 0; }
NK_DYNAMIC int nk_uses_neonhalf(void) { return (nk_capabilities() & nk_cap_neonhalf_k) != 0; }
NK_DYNAMIC int nk_uses_neonfhm(void) { return (nk_capabilities() & nk_cap_neonfhm_k) != 0; }
NK_DYNAMIC int nk_uses_neonbfdot(void) { return (nk_capabilities() & nk_cap_neonbfdot_k) != 0; }
NK_DYNAMIC int nk_uses_neonsdot(void) { return (nk_capabilities() & nk_cap_neonsdot_k) != 0; }
// ARM SVE capabilities
NK_DYNAMIC int nk_uses_sve(void) { return (nk_capabilities() & nk_cap_sve_k) != 0; }
NK_DYNAMIC int nk_uses_svehalf(void) { return (nk_capabilities() & nk_cap_svehalf_k) != 0; }
NK_DYNAMIC int nk_uses_svebfdot(void) { return (nk_capabilities() & nk_cap_svebfdot_k) != 0; }
NK_DYNAMIC int nk_uses_svesdot(void) { return (nk_capabilities() & nk_cap_svesdot_k) != 0; }
NK_DYNAMIC int nk_uses_sve2(void) { return (nk_capabilities() & nk_cap_sve2_k) != 0; }
NK_DYNAMIC int nk_uses_sve2p1(void) { return (nk_capabilities() & nk_cap_sve2p1_k) != 0; }
// ARM SME capabilities
NK_DYNAMIC int nk_uses_sme(void) { return (nk_capabilities() & nk_cap_sme_k) != 0; }
NK_DYNAMIC int nk_uses_sme2(void) { return (nk_capabilities() & nk_cap_sme2_k) != 0; }
NK_DYNAMIC int nk_uses_sme2p1(void) { return (nk_capabilities() & nk_cap_sme2p1_k) != 0; }
NK_DYNAMIC int nk_uses_smef64(void) { return (nk_capabilities() & nk_cap_smef64_k) != 0; }
NK_DYNAMIC int nk_uses_smehalf(void) { return (nk_capabilities() & nk_cap_smehalf_k) != 0; }
NK_DYNAMIC int nk_uses_smebf16(void) { return (nk_capabilities() & nk_cap_smebf16_k) != 0; }
NK_DYNAMIC int nk_uses_smelut2(void) { return (nk_capabilities() & nk_cap_smelut2_k) != 0; }
NK_DYNAMIC int nk_uses_smefa64(void) { return (nk_capabilities() & nk_cap_smefa64_k) != 0; }
// x86 capabilities
NK_DYNAMIC int nk_uses_haswell(void) { return (nk_capabilities() & nk_cap_haswell_k) != 0; }
NK_DYNAMIC int nk_uses_skylake(void) { return (nk_capabilities() & nk_cap_skylake_k) != 0; }
NK_DYNAMIC int nk_uses_ice(void) { return (nk_capabilities() & nk_cap_ice_k) != 0; }
NK_DYNAMIC int nk_uses_genoa(void) { return (nk_capabilities() & nk_cap_genoa_k) != 0; }
NK_DYNAMIC int nk_uses_sapphire(void) { return (nk_capabilities() & nk_cap_sapphire_k) != 0; }
NK_DYNAMIC int nk_uses_sapphire_amx(void) { return (nk_capabilities() & nk_cap_sapphire_amx_k) != 0; }
NK_DYNAMIC int nk_uses_granite_amx(void) { return (nk_capabilities() & nk_cap_granite_amx_k) != 0; }
NK_DYNAMIC int nk_uses_turin(void) { return (nk_capabilities() & nk_cap_turin_k) != 0; }
NK_DYNAMIC int nk_uses_sierra(void) { return (nk_capabilities() & nk_cap_sierra_k) != 0; }
NK_DYNAMIC int nk_uses_dynamic_dispatch(void) { return 1; }
NK_DYNAMIC int nk_configure_thread(nk_capability_t c) { return nk_configure_thread_(c); }

NK_DYNAMIC void nk_f16_to_f32(nk_f16_t const *src, nk_f32_t *dest) { nk_f16_to_f32_serial(src, dest); }
NK_DYNAMIC void nk_f32_to_f16(nk_f32_t const *src, nk_f16_t *dest) { nk_f32_to_f16_serial(src, dest); }
NK_DYNAMIC void nk_bf16_to_f32(nk_bf16_t const *src, nk_f32_t *dest) { nk_bf16_to_f32_serial(src, dest); }
NK_DYNAMIC void nk_f32_to_bf16(nk_f32_t const *src, nk_bf16_t *dest) { nk_f32_to_bf16_serial(src, dest); }
NK_DYNAMIC void nk_e4m3_to_f32(nk_e4m3_t const *src, nk_f32_t *dest) { nk_e4m3_to_f32_serial(src, dest); }
NK_DYNAMIC void nk_f32_to_e4m3(nk_f32_t const *src, nk_e4m3_t *dest) { nk_f32_to_e4m3_serial(src, dest); }
NK_DYNAMIC void nk_e5m2_to_f32(nk_e5m2_t const *src, nk_f32_t *dest) { nk_e5m2_to_f32_serial(src, dest); }
NK_DYNAMIC void nk_f32_to_e5m2(nk_f32_t const *src, nk_e5m2_t *dest) { nk_f32_to_e5m2_serial(src, dest); }

NK_DYNAMIC nk_capability_t nk_capabilities(void) {
    //! The latency of the CPUID instruction can be over 100 cycles, so we cache the result.
    static nk_capability_t static_capabilities = nk_cap_any_k;
    if (static_capabilities != nk_cap_any_k) return static_capabilities;

    static_capabilities = nk_capabilities_();

    // In multithreaded applications we need to ensure that the function pointers are pre-initialized,
    // so the first time we are probing for capabilities, we should also probe all of our metrics
    // with dummy inputs:
    nk_fmax_t dummy_results_buffer[2];
    void *dummy_results = &dummy_results_buffer[0];

    // Passing `NULL` as `x` will trigger all kinds of `nonull` warnings on GCC.
    // Same applies to alpha/beta scalars in FMA/WSUM functions.
    typedef double largest_scalar_t;
    largest_scalar_t dummy_input[1];
    void *x = &dummy_input[0];
    nk_f64_t dummy_alpha = 1, dummy_beta = 1;

    // Dense:
    nk_dot_i8((nk_i8_t *)x, (nk_i8_t *)x, 0, dummy_results);
    nk_dot_u8((nk_u8_t *)x, (nk_u8_t *)x, 0, dummy_results);
    nk_dot_f16((nk_f16_t *)x, (nk_f16_t *)x, 0, dummy_results);
    nk_dot_bf16((nk_bf16_t *)x, (nk_bf16_t *)x, 0, dummy_results);
    nk_dot_f32((nk_f32_t *)x, (nk_f32_t *)x, 0, dummy_results);
    nk_dot_f64((nk_f64_t *)x, (nk_f64_t *)x, 0, dummy_results);

    nk_dot_f16c((nk_f16c_t *)x, (nk_f16c_t *)x, 0, dummy_results);
    nk_dot_bf16c((nk_bf16c_t *)x, (nk_bf16c_t *)x, 0, dummy_results);
    nk_dot_f32c((nk_f32c_t *)x, (nk_f32c_t *)x, 0, dummy_results);
    nk_dot_f64c((nk_f64c_t *)x, (nk_f64c_t *)x, 0, dummy_results);
    nk_vdot_f16c((nk_f16c_t *)x, (nk_f16c_t *)x, 0, dummy_results);
    nk_vdot_bf16c((nk_bf16c_t *)x, (nk_bf16c_t *)x, 0, dummy_results);
    nk_vdot_f32c((nk_f32c_t *)x, (nk_f32c_t *)x, 0, dummy_results);
    nk_vdot_f64c((nk_f64c_t *)x, (nk_f64c_t *)x, 0, dummy_results);

    nk_angular_i8((nk_i8_t *)x, (nk_i8_t *)x, 0, dummy_results);
    nk_angular_u8((nk_u8_t *)x, (nk_u8_t *)x, 0, dummy_results);
    nk_angular_f16((nk_f16_t *)x, (nk_f16_t *)x, 0, dummy_results);
    nk_angular_bf16((nk_bf16_t *)x, (nk_bf16_t *)x, 0, dummy_results);
    nk_angular_f32((nk_f32_t *)x, (nk_f32_t *)x, 0, dummy_results);
    nk_angular_f64((nk_f64_t *)x, (nk_f64_t *)x, 0, dummy_results);

    nk_l2sq_i8((nk_i8_t *)x, (nk_i8_t *)x, 0, dummy_results);
    nk_l2sq_u8((nk_u8_t *)x, (nk_u8_t *)x, 0, dummy_results);
    nk_l2sq_f16((nk_f16_t *)x, (nk_f16_t *)x, 0, dummy_results);
    nk_l2sq_bf16((nk_bf16_t *)x, (nk_bf16_t *)x, 0, dummy_results);
    nk_l2sq_f32((nk_f32_t *)x, (nk_f32_t *)x, 0, dummy_results);
    nk_l2sq_f64((nk_f64_t *)x, (nk_f64_t *)x, 0, dummy_results);

    nk_l2_i8((nk_i8_t *)x, (nk_i8_t *)x, 0, dummy_results);
    nk_l2_i8((nk_i8_t *)x, (nk_i8_t *)x, 0, dummy_results);
    nk_l2_u8((nk_u8_t *)x, (nk_u8_t *)x, 0, dummy_results);
    nk_l2_f16((nk_f16_t *)x, (nk_f16_t *)x, 0, dummy_results);
    nk_l2_bf16((nk_bf16_t *)x, (nk_bf16_t *)x, 0, dummy_results);
    nk_l2_f32((nk_f32_t *)x, (nk_f32_t *)x, 0, dummy_results);
    nk_l2_f64((nk_f64_t *)x, (nk_f64_t *)x, 0, dummy_results);

    nk_haversine_f64((nk_f64_t *)x, (nk_f64_t *)x, (nk_f64_t *)x, (nk_f64_t *)x, 0, dummy_results);
    nk_haversine_f32((nk_f32_t *)x, (nk_f32_t *)x, (nk_f32_t *)x, (nk_f32_t *)x, 0, dummy_results);
    nk_vincenty_f64((nk_f64_t *)x, (nk_f64_t *)x, (nk_f64_t *)x, (nk_f64_t *)x, 0, dummy_results);
    nk_vincenty_f32((nk_f32_t *)x, (nk_f32_t *)x, (nk_f32_t *)x, (nk_f32_t *)x, 0, dummy_results);

    nk_hamming_b8((nk_b8_t *)x, (nk_b8_t *)x, 0, dummy_results);
    nk_jaccard_b8((nk_b8_t *)x, (nk_b8_t *)x, 0, dummy_results);

    nk_kld_f16((nk_f16_t *)x, (nk_f16_t *)x, 0, dummy_results);
    nk_kld_bf16((nk_bf16_t *)x, (nk_bf16_t *)x, 0, dummy_results);
    nk_kld_f32((nk_f32_t *)x, (nk_f32_t *)x, 0, dummy_results);
    nk_kld_f64((nk_f64_t *)x, (nk_f64_t *)x, 0, dummy_results);
    nk_jsd_f16((nk_f16_t *)x, (nk_f16_t *)x, 0, dummy_results);
    nk_jsd_bf16((nk_bf16_t *)x, (nk_bf16_t *)x, 0, dummy_results);
    nk_jsd_f32((nk_f32_t *)x, (nk_f32_t *)x, 0, dummy_results);
    nk_jsd_f64((nk_f64_t *)x, (nk_f64_t *)x, 0, dummy_results);

    // Sparse
    nk_intersect_u16((nk_u16_t *)x, (nk_u16_t *)x, 0, 0, dummy_results);
    nk_intersect_u32((nk_u32_t *)x, (nk_u32_t *)x, 0, 0, dummy_results);

    // Curved:
    nk_bilinear_f64((nk_f64_t *)x, (nk_f64_t *)x, (nk_f64_t *)x, 0, dummy_results);
    nk_mahalanobis_f64((nk_f64_t *)x, (nk_f64_t *)x, (nk_f64_t *)x, 0, dummy_results);
    nk_bilinear_f32((nk_f32_t *)x, (nk_f32_t *)x, (nk_f32_t *)x, 0, dummy_results);
    nk_mahalanobis_f32((nk_f32_t *)x, (nk_f32_t *)x, (nk_f32_t *)x, 0, dummy_results);
    nk_bilinear_f16((nk_f16_t *)x, (nk_f16_t *)x, (nk_f16_t *)x, 0, dummy_results);
    nk_mahalanobis_f16((nk_f16_t *)x, (nk_f16_t *)x, (nk_f16_t *)x, 0, dummy_results);
    nk_bilinear_bf16((nk_bf16_t *)x, (nk_bf16_t *)x, (nk_bf16_t *)x, 0, dummy_results);
    nk_mahalanobis_bf16((nk_bf16_t *)x, (nk_bf16_t *)x, (nk_bf16_t *)x, 0, dummy_results);

    // Elementwise
    nk_wsum_f64((nk_f64_t *)x, (nk_f64_t *)x, 0, &dummy_alpha, &dummy_beta, (nk_f64_t *)x);
    nk_wsum_f32((nk_f32_t *)x, (nk_f32_t *)x, 0, (nk_f32_t *)&dummy_alpha, (nk_f32_t *)&dummy_beta, (nk_f32_t *)x);
    nk_wsum_f16((nk_f16_t *)x, (nk_f16_t *)x, 0, (nk_f32_t *)&dummy_alpha, (nk_f32_t *)&dummy_beta, (nk_f16_t *)x);
    nk_wsum_bf16((nk_bf16_t *)x, (nk_bf16_t *)x, 0, (nk_f32_t *)&dummy_alpha, (nk_f32_t *)&dummy_beta, (nk_bf16_t *)x);
    nk_wsum_i8((nk_i8_t *)x, (nk_i8_t *)x, 0, (nk_f32_t *)&dummy_alpha, (nk_f32_t *)&dummy_beta, (nk_i8_t *)x);
    nk_wsum_u8((nk_u8_t *)x, (nk_u8_t *)x, 0, (nk_f32_t *)&dummy_alpha, (nk_f32_t *)&dummy_beta, (nk_u8_t *)x);
    nk_fma_f64((nk_f64_t *)x, (nk_f64_t *)x, (nk_f64_t *)x, 0, &dummy_alpha, &dummy_beta, (nk_f64_t *)x);
    nk_fma_f32((nk_f32_t *)x, (nk_f32_t *)x, (nk_f32_t *)x, 0, (nk_f32_t *)&dummy_alpha, (nk_f32_t *)&dummy_beta,
               (nk_f32_t *)x);
    nk_fma_f16((nk_f16_t *)x, (nk_f16_t *)x, (nk_f16_t *)x, 0, (nk_f32_t *)&dummy_alpha, (nk_f32_t *)&dummy_beta,
               (nk_f16_t *)x);
    nk_fma_bf16((nk_bf16_t *)x, (nk_bf16_t *)x, (nk_bf16_t *)x, 0, (nk_f32_t *)&dummy_alpha, (nk_f32_t *)&dummy_beta,
                (nk_bf16_t *)x);
    nk_fma_i8((nk_i8_t *)x, (nk_i8_t *)x, (nk_i8_t *)x, 0, (nk_f32_t *)&dummy_alpha, (nk_f32_t *)&dummy_beta,
              (nk_i8_t *)x);
    nk_fma_u8((nk_u8_t *)x, (nk_u8_t *)x, (nk_u8_t *)x, 0, (nk_f32_t *)&dummy_alpha, (nk_f32_t *)&dummy_beta,
              (nk_u8_t *)x);

    return static_capabilities;
}

NK_DYNAMIC void nk_find_kernel_punned( //
    nk_kernel_kind_t kind,             //
    nk_datatype_t datatype,            //
    nk_capability_t supported,         //
    nk_capability_t allowed,           //
    nk_kernel_punned_t *kernel_output, //
    nk_capability_t *capability_output) {
    nk_find_kernel_punned_(kind, datatype, supported, allowed, kernel_output, capability_output);
}

#ifdef __cplusplus
}
#endif

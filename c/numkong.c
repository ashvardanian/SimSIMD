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
#define NK_DECLARATION_DENSE(name, extension, output_type)                                                         \
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

#define NK_DECLARATION_SPARSE(name, extension, type, output_type)                                                   \
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

#define NK_DECLARATION_SPARSE_DOT(name, index_type, weight_type, output_type)                                         \
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

#define NK_DECLARATION_CURVED(name, extension, output_type)                                                           \
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

#define NK_DECLARATION_GEOSPATIAL(name, extension, output_type)                                                 \
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

#define NK_DECLARATION_FMA(name, extension, scalar_type)                                                     \
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

#define NK_DECLARATION_WSUM(name, extension, scalar_type)                                                          \
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

#define NK_DECLARATION_SCALE(name, extension, scalar_type)                                                       \
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

#define NK_DECLARATION_SUM(name, extension)                                                                        \
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

#define NK_DECLARATION_TRIGONOMETRY(name, extension)                                                         \
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

#define NK_DECLARATION_MESH(name, extension, mesh_type)                                                            \
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

#define NK_DECLARATION_REDUCE_ADD(extension, output_type)                                                              \
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

#define NK_DECLARATION_REDUCE_MINMAX(name, extension)                                                               \
    NK_DYNAMIC void nk_reduce_##name##_##extension(nk_##extension##_t const *data, nk_size_t count,                 \
                                                   nk_size_t stride_bytes, nk_##extension##_t *value,               \
                                                   nk_size_t *index) {                                              \
        static nk_kernel_reduce_minmax_punned_t kernel = 0;                                                         \
        if (kernel == 0) {                                                                                          \
            nk_capability_t used_capability;                                                                        \
            nk_find_kernel_punned(nk_kernel_reduce_##name##_k, nk_##extension##_k, nk_capabilities(), nk_cap_any_k, \
                                  (nk_kernel_punned_t *)&kernel, &used_capability);                                 \
        }                                                                                                           \
        kernel(data, count, stride_bytes, value, index);                                                            \
    }

#define NK_DECLARATION_DOTS_PACKED_SIZE(input_type, accum_type)                                                       \
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

#define NK_DECLARATION_DOTS_PACK(input_type, accum_type)                                                       \
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

#define NK_DECLARATION_DOTS(input_type, accum_type, output_type)                                               \
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
NK_DECLARATION_DENSE(dot, i8, i32)
NK_DECLARATION_DENSE(dot, u8, u32)
NK_DECLARATION_DENSE(dot, f16, f32)
NK_DECLARATION_DENSE(dot, bf16, f32)
NK_DECLARATION_DENSE(dot, f32, f32)
NK_DECLARATION_DENSE(dot, f64, f64)
NK_DECLARATION_DENSE(dot, f16c, f32c)
NK_DECLARATION_DENSE(dot, bf16c, f32c)
NK_DECLARATION_DENSE(dot, f32c, f32c)
NK_DECLARATION_DENSE(dot, f64c, f64c)
NK_DECLARATION_DENSE(dot, e4m3, f32)
NK_DECLARATION_DENSE(dot, e5m2, f32)
NK_DECLARATION_DENSE(vdot, f16c, f32c)
NK_DECLARATION_DENSE(vdot, bf16c, f32c)
NK_DECLARATION_DENSE(vdot, f32c, f32c)
NK_DECLARATION_DENSE(vdot, f64c, f64c)

// Spatial distances
NK_DECLARATION_DENSE(angular, i8, f32)
NK_DECLARATION_DENSE(angular, u8, f32)
NK_DECLARATION_DENSE(angular, f16, f32)
NK_DECLARATION_DENSE(angular, bf16, f32)
NK_DECLARATION_DENSE(angular, f32, f32)
NK_DECLARATION_DENSE(angular, f64, f64)
NK_DECLARATION_DENSE(l2sq, i8, u32)
NK_DECLARATION_DENSE(l2sq, u8, u32)
NK_DECLARATION_DENSE(l2sq, f16, f32)
NK_DECLARATION_DENSE(l2sq, bf16, f32)
NK_DECLARATION_DENSE(l2sq, f32, f32)
NK_DECLARATION_DENSE(l2sq, f64, f64)
NK_DECLARATION_DENSE(l2, i8, f32)
NK_DECLARATION_DENSE(l2, u8, f32)
NK_DECLARATION_DENSE(l2, f16, f32)
NK_DECLARATION_DENSE(l2, bf16, f32)
NK_DECLARATION_DENSE(l2, f32, f32)
NK_DECLARATION_DENSE(l2, f64, f64)

// Geospatial distances
NK_DECLARATION_GEOSPATIAL(haversine, f64, f64)
NK_DECLARATION_GEOSPATIAL(haversine, f32, f32)
NK_DECLARATION_GEOSPATIAL(vincenty, f64, f64)
NK_DECLARATION_GEOSPATIAL(vincenty, f32, f32)

// Binary distances
NK_DECLARATION_DENSE(hamming, b8, u32)
NK_DECLARATION_DENSE(jaccard, b8, f32)
NK_DECLARATION_DENSE(jaccard, u32, f32)

// Probability distributions
NK_DECLARATION_DENSE(kld, f16, f32)
NK_DECLARATION_DENSE(kld, bf16, f32)
NK_DECLARATION_DENSE(kld, f32, f32)
NK_DECLARATION_DENSE(kld, f64, f64)
NK_DECLARATION_DENSE(jsd, f16, f32)
NK_DECLARATION_DENSE(jsd, bf16, f32)
NK_DECLARATION_DENSE(jsd, f32, f32)
NK_DECLARATION_DENSE(jsd, f64, f64)

// Sparse sets
NK_DECLARATION_SPARSE(intersect, u16, u16, u32)
NK_DECLARATION_SPARSE(intersect, u32, u32, u32)
NK_DECLARATION_SPARSE_DOT(sparse_dot, u16, bf16, f32)
NK_DECLARATION_SPARSE_DOT(sparse_dot, u32, f32, f32)

// Curved spaces
NK_DECLARATION_CURVED(bilinear, f64, f64)
NK_DECLARATION_CURVED(bilinear, f64c, f64c)
NK_DECLARATION_CURVED(mahalanobis, f64, f64)
NK_DECLARATION_CURVED(bilinear, f32, f32)
NK_DECLARATION_CURVED(bilinear, f32c, f32c)
NK_DECLARATION_CURVED(mahalanobis, f32, f32)
NK_DECLARATION_CURVED(bilinear, f16, f32)
NK_DECLARATION_CURVED(bilinear, f16c, f32c)
NK_DECLARATION_CURVED(mahalanobis, f16, f32)
NK_DECLARATION_CURVED(bilinear, bf16, f32)
NK_DECLARATION_CURVED(bilinear, bf16c, f32c)
NK_DECLARATION_CURVED(mahalanobis, bf16, f32)

// Element-wise operations
NK_DECLARATION_FMA(fma, f64, f64)
NK_DECLARATION_FMA(fma, f32, f32)
NK_DECLARATION_FMA(fma, f16, f32)
NK_DECLARATION_FMA(fma, bf16, f32)
NK_DECLARATION_FMA(fma, i8, f32)
NK_DECLARATION_FMA(fma, u8, f32)
NK_DECLARATION_WSUM(wsum, f64, f64)
NK_DECLARATION_WSUM(wsum, f32, f32)
NK_DECLARATION_WSUM(wsum, f16, f32)
NK_DECLARATION_WSUM(wsum, bf16, f32)
NK_DECLARATION_WSUM(wsum, i8, f32)
NK_DECLARATION_WSUM(wsum, u8, f32)
NK_DECLARATION_SCALE(scale, f64, f64)
NK_DECLARATION_SCALE(scale, f32, f32)
NK_DECLARATION_SCALE(scale, f16, f32)
NK_DECLARATION_SCALE(scale, bf16, f32)
NK_DECLARATION_SCALE(scale, i8, f32)
NK_DECLARATION_SCALE(scale, u8, f32)
NK_DECLARATION_SCALE(scale, i16, f32)
NK_DECLARATION_SCALE(scale, u16, f32)
NK_DECLARATION_SCALE(scale, i32, f64)
NK_DECLARATION_SCALE(scale, u32, f64)
NK_DECLARATION_SCALE(scale, i64, f64)
NK_DECLARATION_SCALE(scale, u64, f64)
NK_DECLARATION_SUM(sum, f64)
NK_DECLARATION_SUM(sum, f32)
NK_DECLARATION_SUM(sum, f16)
NK_DECLARATION_SUM(sum, bf16)
NK_DECLARATION_SUM(sum, i8)
NK_DECLARATION_SUM(sum, u8)
NK_DECLARATION_SUM(sum, i16)
NK_DECLARATION_SUM(sum, u16)
NK_DECLARATION_SUM(sum, i32)
NK_DECLARATION_SUM(sum, u32)
NK_DECLARATION_SUM(sum, i64)
NK_DECLARATION_SUM(sum, u64)

// Trigonometry functions
NK_DECLARATION_TRIGONOMETRY(sin, f32)
NK_DECLARATION_TRIGONOMETRY(sin, f64)
NK_DECLARATION_TRIGONOMETRY(cos, f32)
NK_DECLARATION_TRIGONOMETRY(cos, f64)
NK_DECLARATION_TRIGONOMETRY(atan, f32)
NK_DECLARATION_TRIGONOMETRY(atan, f64)

// Mesh alignment (RMSD, Kabsch, Umeyama)
NK_DECLARATION_MESH(rmsd, f32, f32)
NK_DECLARATION_MESH(rmsd, f64, f64)
NK_DECLARATION_MESH(kabsch, f32, f32)
NK_DECLARATION_MESH(kabsch, f64, f64)
NK_DECLARATION_MESH(umeyama, f32, f32)
NK_DECLARATION_MESH(umeyama, f64, f64)

// Horizontal reductions - floating point
NK_DECLARATION_REDUCE_ADD(f32, f64)
NK_DECLARATION_REDUCE_ADD(f64, f64)
NK_DECLARATION_REDUCE_MINMAX(min, f32)
NK_DECLARATION_REDUCE_MINMAX(max, f32)
NK_DECLARATION_REDUCE_MINMAX(min, f64)
NK_DECLARATION_REDUCE_MINMAX(max, f64)
// Horizontal reductions - integers (output widened for sum)
NK_DECLARATION_REDUCE_ADD(i8, i64)
NK_DECLARATION_REDUCE_ADD(u8, u64)
NK_DECLARATION_REDUCE_ADD(i16, i64)
NK_DECLARATION_REDUCE_ADD(u16, u64)
NK_DECLARATION_REDUCE_ADD(i32, i64)
NK_DECLARATION_REDUCE_ADD(u32, u64)
NK_DECLARATION_REDUCE_ADD(i64, i64)
NK_DECLARATION_REDUCE_ADD(u64, u64)
NK_DECLARATION_REDUCE_MINMAX(min, i8)
NK_DECLARATION_REDUCE_MINMAX(max, i8)
NK_DECLARATION_REDUCE_MINMAX(min, u8)
NK_DECLARATION_REDUCE_MINMAX(max, u8)
NK_DECLARATION_REDUCE_MINMAX(min, i16)
NK_DECLARATION_REDUCE_MINMAX(max, i16)
NK_DECLARATION_REDUCE_MINMAX(min, u16)
NK_DECLARATION_REDUCE_MINMAX(max, u16)
NK_DECLARATION_REDUCE_MINMAX(min, i32)
NK_DECLARATION_REDUCE_MINMAX(max, i32)
NK_DECLARATION_REDUCE_MINMAX(min, u32)
NK_DECLARATION_REDUCE_MINMAX(max, u32)
NK_DECLARATION_REDUCE_MINMAX(min, i64)
NK_DECLARATION_REDUCE_MINMAX(max, i64)
NK_DECLARATION_REDUCE_MINMAX(min, u64)
NK_DECLARATION_REDUCE_MINMAX(max, u64)

// Matrix multiplications (GEMM with packed B)
NK_DECLARATION_DOTS_PACKED_SIZE(f32, f32)
NK_DECLARATION_DOTS_PACKED_SIZE(f64, f64)
NK_DECLARATION_DOTS_PACKED_SIZE(f16, f32)
NK_DECLARATION_DOTS_PACKED_SIZE(bf16, f32)
NK_DECLARATION_DOTS_PACKED_SIZE(i8, i32)
NK_DECLARATION_DOTS_PACKED_SIZE(u8, u32)
NK_DECLARATION_DOTS_PACK(f32, f32)
NK_DECLARATION_DOTS_PACK(f64, f64)
NK_DECLARATION_DOTS_PACK(f16, f32)
NK_DECLARATION_DOTS_PACK(bf16, f32)
NK_DECLARATION_DOTS_PACK(i8, i32)
NK_DECLARATION_DOTS_PACK(u8, u32)
NK_DECLARATION_DOTS(f32, f32, f32)
NK_DECLARATION_DOTS(f64, f64, f64)
NK_DECLARATION_DOTS(f16, f32, f32)
NK_DECLARATION_DOTS(bf16, f32, f32)
NK_DECLARATION_DOTS(i8, i32, i32)
NK_DECLARATION_DOTS(u8, u32, u32)

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

NK_DYNAMIC void nk_f16_to_f32(nk_f16_t const *src, nk_f32_t *dest) { nk_f16_to_f32_(src, dest); }
NK_DYNAMIC void nk_f32_to_f16(nk_f32_t const *src, nk_f16_t *dest) { nk_f32_to_f16_(src, dest); }
NK_DYNAMIC void nk_bf16_to_f32(nk_bf16_t const *src, nk_f32_t *dest) { nk_bf16_to_f32_(src, dest); }
NK_DYNAMIC void nk_f32_to_bf16(nk_f32_t const *src, nk_bf16_t *dest) { nk_f32_to_bf16_(src, dest); }
NK_DYNAMIC void nk_e4m3_to_f32(nk_e4m3_t const *src, nk_f32_t *dest) { nk_e4m3_to_f32_(src, dest); }
NK_DYNAMIC void nk_f32_to_e4m3(nk_f32_t const *src, nk_e4m3_t *dest) { nk_f32_to_e4m3_(src, dest); }
NK_DYNAMIC void nk_e5m2_to_f32(nk_e5m2_t const *src, nk_f32_t *dest) { nk_e5m2_to_f32_(src, dest); }
NK_DYNAMIC void nk_f32_to_e5m2(nk_f32_t const *src, nk_e5m2_t *dest) { nk_f32_to_e5m2_(src, dest); }

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

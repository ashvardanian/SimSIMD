/**
 *  @brief Dynamic dispatch library for NumKong.
 *  @note Compile with the most recent compiler available.
 *  @file c/numkong.c
 */
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

/**
 *  @brief Fill memory with 0xFF - produces NaN for floats, -1/MAX for integers.
 *  Avoids libc dependency on memset.
 */
NK_INTERNAL void nk_fill_error_(void *ptr, nk_size_t bytes) {
    nk_u8_t *p = (nk_u8_t *)ptr;
    while (bytes--) *p++ = 0xFF;
}

// Every time a function is called, it checks if the kernel is already loaded. If not, it fetches it.
// If no kernel is found, we fill the output with 0xFF bytes (NaN for floats, -1/MAX for integers).
#define nk_dispatch_dense_(name, extension, input_type, output_type)                                                 \
    NK_DYNAMIC void nk_##name##_##extension(nk_##input_type##_t const *a, nk_##input_type##_t const *b, nk_size_t n, \
                                            nk_##output_type##_t *results) {                                         \
        static nk_metric_dense_punned_t kernel = 0;                                                                  \
        if (kernel == 0) {                                                                                           \
            nk_capability_t used_capability;                                                                         \
            nk_find_kernel_punned(nk_kernel_##name##_k, nk_##extension##_k, nk_capabilities(), nk_cap_any_k,         \
                                  (nk_kernel_punned_t *)&kernel, &used_capability);                                  \
            if (!kernel) {                                                                                           \
                nk_fill_error_(results, sizeof(nk_##output_type##_t));                                               \
                return;                                                                                              \
            }                                                                                                        \
        }                                                                                                            \
        kernel(a, b, n, (void *)results);                                                                            \
    }

#define nk_dispatch_sparse_(name, extension, type)                                                              \
    NK_DYNAMIC void nk_##name##_##extension(nk_##type##_t const *a, nk_##type##_t const *b, nk_size_t a_length, \
                                            nk_size_t b_length, nk_##type##_t *result, nk_size_t *count) {      \
        static nk_sparse_intersect_punned_t kernel = 0;                                                         \
        if (kernel == 0) {                                                                                      \
            nk_capability_t used_capability;                                                                    \
            nk_find_kernel_punned(nk_kernel_##name##_k, nk_##extension##_k, nk_capabilities(), nk_cap_any_k,    \
                                  (nk_kernel_punned_t *)(&kernel), &used_capability);                           \
            if (!kernel) {                                                                                      \
                if (count) *count = 0;                                                                          \
                return;                                                                                         \
            }                                                                                                   \
        }                                                                                                       \
        kernel(a, b, a_length, b_length, (void *)result, count);                                                \
    }

#define nk_dispatch_sparse_dot_(name, index_type, weight_type, output_type)                                           \
    NK_DYNAMIC void nk_##name##_##index_type##weight_type(nk_##index_type##_t const *a, nk_##index_type##_t const *b, \
                                                          nk_##weight_type##_t const *a_weights,                      \
                                                          nk_##weight_type##_t const *b_weights, nk_size_t a_length,  \
                                                          nk_size_t b_length, nk_##output_type##_t *product) {        \
        static nk_sparse_dot_punned_t kernel = 0;                                                                     \
        if (kernel == 0) {                                                                                            \
            nk_capability_t used_capability;                                                                          \
            nk_find_kernel_punned(nk_kernel_sparse_dot_k, nk_##weight_type##_k, nk_capabilities(), nk_cap_any_k,      \
                                  (nk_kernel_punned_t *)&kernel, &used_capability);                                   \
            if (!kernel) {                                                                                            \
                nk_fill_error_(product, sizeof(nk_##output_type##_t));                                                \
                return;                                                                                               \
            }                                                                                                         \
        }                                                                                                             \
        kernel(a, b, a_weights, b_weights, a_length, b_length, (void *)product);                                      \
    }

#define nk_dispatch_curved_(name, extension, output_type)                                                             \
    NK_DYNAMIC void nk_##name##_##extension(nk_##extension##_t const *a, nk_##extension##_t const *b,                 \
                                            nk_##extension##_t const *c, nk_size_t n, nk_##output_type##_t *result) { \
        static nk_metric_curved_punned_t kernel = 0;                                                                  \
        if (kernel == 0) {                                                                                            \
            nk_capability_t used_capability;                                                                          \
            nk_find_kernel_punned(nk_kernel_##name##_k, nk_##extension##_k, nk_capabilities(), nk_cap_any_k,          \
                                  (nk_kernel_punned_t *)(&kernel), &used_capability);                                 \
            if (!kernel) {                                                                                            \
                nk_fill_error_(result, sizeof(nk_##output_type##_t));                                                 \
                return;                                                                                               \
            }                                                                                                         \
        }                                                                                                             \
        kernel(a, b, c, n, (void *)result);                                                                           \
    }

#define nk_dispatch_geospatial_(name, extension, output_type)                                                   \
    NK_DYNAMIC void nk_##name##_##extension(nk_##extension##_t const *a_lats, nk_##extension##_t const *a_lons, \
                                            nk_##extension##_t const *b_lats, nk_##extension##_t const *b_lons, \
                                            nk_size_t n, nk_##output_type##_t *results) {                       \
        static nk_metric_geospatial_punned_t kernel = 0;                                                        \
        if (kernel == 0) {                                                                                      \
            nk_capability_t used_capability;                                                                    \
            nk_find_kernel_punned(nk_kernel_##name##_k, nk_##extension##_k, nk_capabilities(), nk_cap_any_k,    \
                                  (nk_kernel_punned_t *)(&kernel), &used_capability);                           \
            if (!kernel) {                                                                                      \
                nk_fill_error_(results, sizeof(nk_##output_type##_t));                                          \
                return;                                                                                         \
            }                                                                                                   \
        }                                                                                                       \
        kernel(a_lats, a_lons, b_lats, b_lons, n, (void *)results);                                             \
    }

#define nk_dispatch_each_fma_(extension, scalar_type)                                                        \
    NK_DYNAMIC void nk_each_fma_##extension(                                                                 \
        nk_##extension##_t const *a, nk_##extension##_t const *b, nk_##extension##_t const *c, nk_size_t n,  \
        nk_##scalar_type##_t const *alpha, nk_##scalar_type##_t const *beta, nk_##extension##_t *result) {   \
        static nk_each_fma_punned_t kernel = 0;                                                              \
        if (kernel == 0) {                                                                                   \
            nk_capability_t used_capability;                                                                 \
            nk_find_kernel_punned(nk_kernel_each_fma_k, nk_##extension##_k, nk_capabilities(), nk_cap_any_k, \
                                  (nk_kernel_punned_t *)(&kernel), &used_capability);                        \
            if (!kernel) {                                                                                   \
                nk_fill_error_(result, n * sizeof(nk_##extension##_t));                                      \
                return;                                                                                      \
            }                                                                                                \
        }                                                                                                    \
        kernel(a, b, c, n, (void const *)alpha, (void const *)beta, result);                                 \
    }

#define nk_dispatch_each_blend_(extension, scalar_type)                                                              \
    NK_DYNAMIC void nk_each_blend_##extension(nk_##extension##_t const *a, nk_##extension##_t const *b, nk_size_t n, \
                                              nk_##scalar_type##_t const *alpha, nk_##scalar_type##_t const *beta,   \
                                              nk_##extension##_t *result) {                                          \
        static nk_each_blend_punned_t kernel = 0;                                                                    \
        if (kernel == 0) {                                                                                           \
            nk_capability_t used_capability;                                                                         \
            nk_find_kernel_punned(nk_kernel_each_blend_k, nk_##extension##_k, nk_capabilities(), nk_cap_any_k,       \
                                  (nk_kernel_punned_t *)(&kernel), &used_capability);                                \
            if (!kernel) {                                                                                           \
                nk_fill_error_(result, n * sizeof(nk_##extension##_t));                                              \
                return;                                                                                              \
            }                                                                                                        \
        }                                                                                                            \
        kernel(a, b, n, (void const *)alpha, (void const *)beta, result);                                            \
    }

#define nk_dispatch_each_scale_(extension, scalar_type)                                                            \
    NK_DYNAMIC void nk_each_scale_##extension(nk_##extension##_t const *a, nk_size_t n,                            \
                                              nk_##scalar_type##_t const *alpha, nk_##scalar_type##_t const *beta, \
                                              nk_##extension##_t *result) {                                        \
        static nk_each_scale_punned_t kernel = 0;                                                                  \
        if (kernel == 0) {                                                                                         \
            nk_capability_t used_capability;                                                                       \
            nk_find_kernel_punned(nk_kernel_each_scale_k, nk_##extension##_k, nk_capabilities(), nk_cap_any_k,     \
                                  (nk_kernel_punned_t *)(&kernel), &used_capability);                              \
            if (!kernel) {                                                                                         \
                nk_fill_error_(result, n * sizeof(nk_##extension##_t));                                            \
                return;                                                                                            \
            }                                                                                                      \
        }                                                                                                          \
        kernel(a, n, (void const *)alpha, (void const *)beta, result);                                             \
    }

#define nk_dispatch_each_sum_(extension)                                                                           \
    NK_DYNAMIC void nk_each_sum_##extension(nk_##extension##_t const *a, nk_##extension##_t const *b, nk_size_t n, \
                                            nk_##extension##_t *result) {                                          \
        static nk_each_sum_punned_t kernel = 0;                                                                    \
        if (kernel == 0) {                                                                                         \
            nk_capability_t used_capability;                                                                       \
            nk_find_kernel_punned(nk_kernel_each_sum_k, nk_##extension##_k, nk_capabilities(), nk_cap_any_k,       \
                                  (nk_kernel_punned_t *)(&kernel), &used_capability);                              \
            if (!kernel) {                                                                                         \
                nk_fill_error_(result, n * sizeof(nk_##extension##_t));                                            \
                return;                                                                                            \
            }                                                                                                      \
        }                                                                                                          \
        kernel(a, b, n, result);                                                                                   \
    }

#define nk_dispatch_trigonometry_(name, extension)                                                           \
    NK_DYNAMIC void nk_##name##_##extension(nk_##extension##_t const *inputs, nk_size_t n,                   \
                                            nk_##extension##_t *outputs) {                                   \
        static nk_kernel_trigonometry_punned_t kernel = 0;                                                   \
        if (kernel == 0) {                                                                                   \
            nk_capability_t used_capability;                                                                 \
            nk_find_kernel_punned(nk_kernel_##name##_k, nk_##extension##_k, nk_capabilities(), nk_cap_any_k, \
                                  (nk_kernel_punned_t *)(&kernel), &used_capability);                        \
            if (!kernel) {                                                                                   \
                nk_fill_error_(outputs, n * sizeof(nk_##extension##_t));                                     \
                return;                                                                                      \
            }                                                                                                \
        }                                                                                                    \
        kernel(inputs, n, outputs);                                                                          \
    }

#define nk_dispatch_mesh_(name, extension, mesh_type)                                                              \
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
                if (a_centroid) nk_fill_error_(a_centroid, 3 * sizeof(nk_##mesh_type##_t));                        \
                if (b_centroid) nk_fill_error_(b_centroid, 3 * sizeof(nk_##mesh_type##_t));                        \
                if (rotation) nk_fill_error_(rotation, 9 * sizeof(nk_##mesh_type##_t));                            \
                if (scale) nk_fill_error_(scale, sizeof(nk_##mesh_type##_t));                                      \
                nk_fill_error_(result, sizeof(nk_##mesh_type##_t));                                                \
                return;                                                                                            \
            }                                                                                                      \
        }                                                                                                          \
        kernel(a, b, n, (void *)a_centroid, (void *)b_centroid, (void *)rotation, (void *)scale, (void *)result);  \
    }

#define nk_dispatch_reduce_add_(extension, output_type)                                                                \
    NK_DYNAMIC void nk_reduce_add_##extension(nk_##extension##_t const *data, nk_size_t count, nk_size_t stride_bytes, \
                                              nk_##output_type##_t *result) {                                          \
        static nk_kernel_reduce_add_punned_t kernel = 0;                                                               \
        if (kernel == 0) {                                                                                             \
            nk_capability_t used_capability;                                                                           \
            nk_find_kernel_punned(nk_kernel_reduce_add_k, nk_##extension##_k, nk_capabilities(), nk_cap_any_k,         \
                                  (nk_kernel_punned_t *)&kernel, &used_capability);                                    \
            if (!kernel) {                                                                                             \
                nk_fill_error_(result, sizeof(nk_##output_type##_t));                                                  \
                return;                                                                                                \
            }                                                                                                          \
        }                                                                                                              \
        kernel(data, count, stride_bytes, result);                                                                     \
    }

#define nk_dispatch_reduce_minmax_(name, extension, output_type)                                                    \
    NK_DYNAMIC void nk_reduce_##name##_##extension(nk_##extension##_t const *data, nk_size_t count,                 \
                                                   nk_size_t stride_bytes, nk_##output_type##_t *value,             \
                                                   nk_size_t *index) {                                              \
        static nk_kernel_reduce_minmax_punned_t kernel = 0;                                                         \
        if (kernel == 0) {                                                                                          \
            nk_capability_t used_capability;                                                                        \
            nk_find_kernel_punned(nk_kernel_reduce_##name##_k, nk_##extension##_k, nk_capabilities(), nk_cap_any_k, \
                                  (nk_kernel_punned_t *)&kernel, &used_capability);                                 \
            if (!kernel) {                                                                                          \
                nk_fill_error_(value, sizeof(nk_##output_type##_t));                                                \
                nk_fill_error_(index, sizeof(nk_size_t));                                                           \
                return;                                                                                             \
            }                                                                                                       \
        }                                                                                                           \
        kernel(data, count, stride_bytes, value, index);                                                            \
    }

#define nk_dispatch_dots_packed_size_(name, input_type, accum_type)                                             \
    NK_DYNAMIC nk_size_t nk_dots_packed_size_##name(nk_size_t n, nk_size_t k) {                                 \
        static nk_dots_packed_size_punned_t kernel = 0;                                                         \
        if (kernel == 0) {                                                                                      \
            nk_capability_t used_capability;                                                                    \
            nk_find_kernel_punned(nk_kernel_dots_packed_size_k, nk_##name##_k, nk_capabilities(), nk_cap_any_k, \
                                  (nk_kernel_punned_t *)&kernel, &used_capability);                             \
            if (!kernel) return 0;                                                                              \
        }                                                                                                       \
        return kernel(n, k);                                                                                    \
    }

#define nk_dispatch_dots_pack_(name, input_type, accum_type)                                                        \
    NK_DYNAMIC void nk_dots_pack_##name(nk_##input_type##_t const *b, nk_size_t n, nk_size_t k, nk_size_t b_stride, \
                                        void *b_packed) {                                                           \
        static nk_dots_pack_punned_t kernel = 0;                                                                    \
        if (kernel == 0) {                                                                                          \
            nk_capability_t used_capability;                                                                        \
            nk_find_kernel_punned(nk_kernel_dots_pack_k, nk_##name##_k, nk_capabilities(), nk_cap_any_k,            \
                                  (nk_kernel_punned_t *)&kernel, &used_capability);                                 \
            if (!kernel) {                                                                                          \
                nk_size_t packed_size = nk_dots_packed_size_##name(n, k);                                           \
                if (packed_size) nk_fill_error_(b_packed, packed_size);                                             \
                return;                                                                                             \
            }                                                                                                       \
        }                                                                                                           \
        kernel(b, n, k, b_stride, b_packed);                                                                        \
    }

#define nk_dispatch_dots_packed_(name, input_type, accum_type, output_type)                                            \
    NK_DYNAMIC void nk_dots_packed_##name(nk_##input_type##_t const *a, void const *b_packed, nk_##output_type##_t *c, \
                                          nk_size_t m, nk_size_t n, nk_size_t k, nk_size_t a_stride,                   \
                                          nk_size_t c_stride) {                                                        \
        static nk_dots_punned_t kernel = 0;                                                                            \
        if (kernel == 0) {                                                                                             \
            nk_capability_t used_capability;                                                                           \
            nk_find_kernel_punned(nk_kernel_dots_k, nk_##name##_k, nk_capabilities(), nk_cap_any_k,                    \
                                  (nk_kernel_punned_t *)&kernel, &used_capability);                                    \
            if (!kernel) {                                                                                             \
                for (nk_size_t row = 0; row < m; ++row)                                                                \
                    nk_fill_error_((nk_u8_t *)c + row * c_stride, n * sizeof(nk_##output_type##_t));                   \
                return;                                                                                                \
            }                                                                                                          \
        }                                                                                                              \
        kernel(a, b_packed, c, m, n, k, a_stride, c_stride);                                                           \
    }

// Dot products
nk_dispatch_dense_(dot, i8, i8, i32)
nk_dispatch_dense_(dot, u8, u8, u32)
nk_dispatch_dense_(dot, i4, i4x2, i32)
nk_dispatch_dense_(dot, u4, u4x2, u32)
nk_dispatch_dense_(dot, f16, f16, f32)
nk_dispatch_dense_(dot, bf16, bf16, f32)
nk_dispatch_dense_(dot, f32, f32, f32)
nk_dispatch_dense_(dot, f64, f64, f64)
nk_dispatch_dense_(dot, f16c, f16c, f32c)
nk_dispatch_dense_(dot, bf16c, bf16c, f32c)
nk_dispatch_dense_(dot, f32c, f32c, f32c)
nk_dispatch_dense_(dot, f64c, f64c, f64c)
nk_dispatch_dense_(dot, e4m3, e4m3, f32)
nk_dispatch_dense_(dot, e5m2, e5m2, f32)
nk_dispatch_dense_(vdot, f16c, f16c, f32c)
nk_dispatch_dense_(vdot, bf16c, bf16c, f32c)
nk_dispatch_dense_(vdot, f32c, f32c, f32c)
nk_dispatch_dense_(vdot, f64c, f64c, f64c)

// Spatial distances
nk_dispatch_dense_(angular, i8, i8, f32)
nk_dispatch_dense_(angular, u8, u8, f32)
nk_dispatch_dense_(angular, i4, i4x2, f32)
nk_dispatch_dense_(angular, u4, u4x2, f32)
nk_dispatch_dense_(angular, f16, f16, f32)
nk_dispatch_dense_(angular, bf16, bf16, f32)
nk_dispatch_dense_(angular, f32, f32, f32)
nk_dispatch_dense_(angular, f64, f64, f64)
nk_dispatch_dense_(angular, e4m3, e4m3, f32)
nk_dispatch_dense_(angular, e5m2, e5m2, f32)
nk_dispatch_dense_(l2sq, i8, i8, u32)
nk_dispatch_dense_(l2sq, u8, u8, u32)
nk_dispatch_dense_(l2sq, i4, i4x2, u32)
nk_dispatch_dense_(l2sq, u4, u4x2, u32)
nk_dispatch_dense_(l2sq, f16, f16, f32)
nk_dispatch_dense_(l2sq, bf16, bf16, f32)
nk_dispatch_dense_(l2sq, f32, f32, f32)
nk_dispatch_dense_(l2sq, f64, f64, f64)
nk_dispatch_dense_(l2sq, e4m3, e4m3, f32)
nk_dispatch_dense_(l2sq, e5m2, e5m2, f32)
nk_dispatch_dense_(l2, i8, i8, f32)
nk_dispatch_dense_(l2, u8, u8, f32)
nk_dispatch_dense_(l2, i4, i4x2, f32)
nk_dispatch_dense_(l2, u4, u4x2, f32)
nk_dispatch_dense_(l2, f16, f16, f32)
nk_dispatch_dense_(l2, bf16, bf16, f32)
nk_dispatch_dense_(l2, f32, f32, f32)
nk_dispatch_dense_(l2, f64, f64, f64)
nk_dispatch_dense_(l2, e4m3, e4m3, f32)
nk_dispatch_dense_(l2, e5m2, e5m2, f32)

// Geospatial distances
nk_dispatch_geospatial_(haversine, f64, f64)
nk_dispatch_geospatial_(haversine, f32, f32)
nk_dispatch_geospatial_(vincenty, f64, f64)
nk_dispatch_geospatial_(vincenty, f32, f32)

// Binary distances
nk_dispatch_dense_(hamming, u1, u1x8, u32)
nk_dispatch_dense_(jaccard, u1, u1x8, f32)
nk_dispatch_dense_(jaccard, u32, u32, f32)

// Probability distributions
nk_dispatch_dense_(kld, f16, f16, f32)
nk_dispatch_dense_(kld, bf16, bf16, f32)
nk_dispatch_dense_(kld, f32, f32, f32)
nk_dispatch_dense_(kld, f64, f64, f64)
nk_dispatch_dense_(jsd, f16, f16, f32)
nk_dispatch_dense_(jsd, bf16, bf16, f32)
nk_dispatch_dense_(jsd, f32, f32, f32)
nk_dispatch_dense_(jsd, f64, f64, f64)

// Sparse sets
nk_dispatch_sparse_(sparse_intersect, u16, u16)
nk_dispatch_sparse_(sparse_intersect, u32, u32)
nk_dispatch_sparse_(sparse_intersect, u64, u64)
nk_dispatch_sparse_dot_(sparse_dot, u16, bf16, f32)
nk_dispatch_sparse_dot_(sparse_dot, u32, f32, f32)

// Curved spaces
nk_dispatch_curved_(bilinear, f64, f64)
nk_dispatch_curved_(bilinear, f64c, f64c)
nk_dispatch_curved_(mahalanobis, f64, f64)
nk_dispatch_curved_(bilinear, f32, f32)
nk_dispatch_curved_(bilinear, f32c, f32c)
nk_dispatch_curved_(mahalanobis, f32, f32)
nk_dispatch_curved_(bilinear, f16, f32)
nk_dispatch_curved_(bilinear, f16c, f32c)
nk_dispatch_curved_(mahalanobis, f16, f32)
nk_dispatch_curved_(bilinear, bf16, f32)
nk_dispatch_curved_(bilinear, bf16c, f32c)
nk_dispatch_curved_(mahalanobis, bf16, f32)

// Element-wise operations
nk_dispatch_each_fma_(f32, f32)
nk_dispatch_each_fma_(f16, f32)
nk_dispatch_each_fma_(f64, f64)
nk_dispatch_each_fma_(bf16, f32)
nk_dispatch_each_fma_(i8, f32)
nk_dispatch_each_fma_(u8, f32)
nk_dispatch_each_blend_(f64, f64)
nk_dispatch_each_blend_(f32, f32)
nk_dispatch_each_blend_(f16, f32)
nk_dispatch_each_blend_(bf16, f32)
nk_dispatch_each_blend_(i8, f32)
nk_dispatch_each_blend_(u8, f32)
nk_dispatch_each_scale_(f64, f64)
nk_dispatch_each_scale_(f32, f32)
nk_dispatch_each_scale_(f16, f32)
nk_dispatch_each_scale_(bf16, f32)
nk_dispatch_each_scale_(i8, f32)
nk_dispatch_each_scale_(u8, f32)
nk_dispatch_each_scale_(i16, f32)
nk_dispatch_each_scale_(u16, f32)
nk_dispatch_each_scale_(i32, f64)
nk_dispatch_each_scale_(u32, f64)
nk_dispatch_each_scale_(i64, f64)
nk_dispatch_each_scale_(u64, f64)
nk_dispatch_each_sum_(f64)
nk_dispatch_each_sum_(f32)
nk_dispatch_each_sum_(f16)
nk_dispatch_each_sum_(bf16)
nk_dispatch_each_sum_(i8)
nk_dispatch_each_sum_(u8)
nk_dispatch_each_sum_(i16)
nk_dispatch_each_sum_(u16)
nk_dispatch_each_sum_(i32)
nk_dispatch_each_sum_(u32)
nk_dispatch_each_sum_(i64)
nk_dispatch_each_sum_(u64)

// Trigonometry functions
nk_dispatch_trigonometry_(sin, f32)
nk_dispatch_trigonometry_(sin, f64)
nk_dispatch_trigonometry_(cos, f32)
nk_dispatch_trigonometry_(cos, f64)
nk_dispatch_trigonometry_(atan, f32)
nk_dispatch_trigonometry_(atan, f64)

// Mesh alignment (RMSD, Kabsch, Umeyama)
nk_dispatch_mesh_(rmsd, f32, f32)
nk_dispatch_mesh_(rmsd, f64, f64)
nk_dispatch_mesh_(kabsch, f32, f32)
nk_dispatch_mesh_(kabsch, f64, f64)
nk_dispatch_mesh_(umeyama, f32, f32)
nk_dispatch_mesh_(umeyama, f64, f64)

// Horizontal reductions - floating point
nk_dispatch_reduce_add_(f32, f64)
nk_dispatch_reduce_add_(f64, f64)
nk_dispatch_reduce_minmax_(min, f32, f32)
nk_dispatch_reduce_minmax_(max, f32, f32)
nk_dispatch_reduce_minmax_(min, f64, f64)
nk_dispatch_reduce_minmax_(max, f64, f64)
// Horizontal reductions - integers (output widened for sum)
nk_dispatch_reduce_add_(i8, i64)
nk_dispatch_reduce_add_(u8, u64)
nk_dispatch_reduce_add_(i16, i64)
nk_dispatch_reduce_add_(u16, u64)
nk_dispatch_reduce_add_(i32, i64)
nk_dispatch_reduce_add_(u32, u64)
nk_dispatch_reduce_add_(i64, i64)
nk_dispatch_reduce_add_(u64, u64)
nk_dispatch_reduce_minmax_(min, i8, i8)
nk_dispatch_reduce_minmax_(max, i8, i8)
nk_dispatch_reduce_minmax_(min, u8, u8)
nk_dispatch_reduce_minmax_(max, u8, u8)
nk_dispatch_reduce_minmax_(min, i16, i16)
nk_dispatch_reduce_minmax_(max, i16, i16)
nk_dispatch_reduce_minmax_(min, u16, u16)
nk_dispatch_reduce_minmax_(max, u16, u16)
nk_dispatch_reduce_minmax_(min, i32, i32)
nk_dispatch_reduce_minmax_(max, i32, i32)
nk_dispatch_reduce_minmax_(min, u32, u32)
nk_dispatch_reduce_minmax_(max, u32, u32)
nk_dispatch_reduce_minmax_(min, i64, i64)
nk_dispatch_reduce_minmax_(max, i64, i64)
nk_dispatch_reduce_minmax_(min, u64, u64)
nk_dispatch_reduce_minmax_(max, u64, u64)
// Horizontal reductions - half-precision types (output widened to f32)
nk_dispatch_reduce_add_(f16, f32)
nk_dispatch_reduce_add_(bf16, f32)
nk_dispatch_reduce_add_(e4m3, f32)
nk_dispatch_reduce_add_(e5m2, f32)
nk_dispatch_reduce_minmax_(min, f16, f32)
nk_dispatch_reduce_minmax_(max, f16, f32)
nk_dispatch_reduce_minmax_(min, bf16, f32)
nk_dispatch_reduce_minmax_(max, bf16, f32)
nk_dispatch_reduce_minmax_(min, e4m3, f32)
nk_dispatch_reduce_minmax_(max, e4m3, f32)
nk_dispatch_reduce_minmax_(min, e5m2, f32)
nk_dispatch_reduce_minmax_(max, e5m2, f32)
// Elementwise operations - FP8 types
nk_dispatch_each_sum_(e4m3)
nk_dispatch_each_sum_(e5m2)
nk_dispatch_each_scale_(e4m3, f32)
nk_dispatch_each_scale_(e5m2, f32)
nk_dispatch_each_blend_(e4m3, f32)
nk_dispatch_each_blend_(e5m2, f32)
nk_dispatch_each_fma_(e4m3, f32)
nk_dispatch_each_fma_(e5m2, f32)

// Matrix multiplications (GEMM with packed B)
nk_dispatch_dots_packed_size_(f32, f32, f32)
nk_dispatch_dots_packed_size_(f64, f64, f64)
nk_dispatch_dots_packed_size_(f16, f16, f32)
nk_dispatch_dots_packed_size_(bf16, bf16, f32)
nk_dispatch_dots_packed_size_(i8, i8, i32)
nk_dispatch_dots_packed_size_(u8, u8, u32)
nk_dispatch_dots_packed_size_(e4m3, e4m3, f32)
nk_dispatch_dots_packed_size_(e5m2, e5m2, f32)
nk_dispatch_dots_packed_size_(u1, u1x8, u32)
nk_dispatch_dots_packed_size_(u4, u4x2, u32)
nk_dispatch_dots_packed_size_(i4, i4x2, i32)
nk_dispatch_dots_pack_(f32, f32, f32)
nk_dispatch_dots_pack_(f64, f64, f64)
nk_dispatch_dots_pack_(f16, f16, f32)
nk_dispatch_dots_pack_(bf16, bf16, f32)
nk_dispatch_dots_pack_(i8, i8, i32)
nk_dispatch_dots_pack_(u8, u8, u32)
nk_dispatch_dots_pack_(e4m3, e4m3, f32)
nk_dispatch_dots_pack_(e5m2, e5m2, f32)
nk_dispatch_dots_pack_(u1, u1x8, u32)
nk_dispatch_dots_pack_(u4, u4x2, u32)
nk_dispatch_dots_pack_(i4, i4x2, i32)
nk_dispatch_dots_packed_(f32, f32, f32, f32)
nk_dispatch_dots_packed_(f64, f64, f64, f64)
nk_dispatch_dots_packed_(f16, f16, f32, f32)
nk_dispatch_dots_packed_(bf16, bf16, f32, f32)
nk_dispatch_dots_packed_(i8, i8, i32, i32)
nk_dispatch_dots_packed_(u8, u8, u32, u32)
nk_dispatch_dots_packed_(e4m3, e4m3, f32, f32)
nk_dispatch_dots_packed_(e5m2, e5m2, f32, f32)
nk_dispatch_dots_packed_(u1, u1x8, u32, u32)
nk_dispatch_dots_packed_(u4, u4x2, u32, u32)
nk_dispatch_dots_packed_(i4, i4x2, i32, i32)

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
// RISC-V capabilities
NK_DYNAMIC int nk_uses_spacemit(void) { return (nk_capabilities() & nk_cap_spacemit_k) != 0; }
NK_DYNAMIC int nk_uses_sifive(void) { return (nk_capabilities() & nk_cap_sifive_k) != 0; }
NK_DYNAMIC int nk_uses_xuantie(void) { return (nk_capabilities() & nk_cap_xuantie_k) != 0; }
NK_DYNAMIC int nk_uses_dynamic_dispatch(void) { return 1; }
NK_DYNAMIC int nk_configure_thread(nk_capability_t c) { return nk_configure_thread_(c); }

NK_DYNAMIC void nk_f16_to_f32(nk_f16_t const *src, nk_f32_t *dest) {
#if NK_TARGET_SAPPHIRE
    if (nk_uses_sapphire()) {
        nk_f16_to_f32_sapphire(src, dest);
        return;
    }
#endif
#if NK_TARGET_HASWELL
    if (nk_uses_haswell()) {
        nk_f16_to_f32_haswell(src, dest);
        return;
    }
#endif
#if NK_TARGET_NEON
    if (nk_uses_neon()) {
        nk_f16_to_f32_neon(src, dest);
        return;
    }
#endif
    nk_f16_to_f32_serial(src, dest);
}

NK_DYNAMIC void nk_f32_to_f16(nk_f32_t const *src, nk_f16_t *dest) {
#if NK_TARGET_SAPPHIRE
    if (nk_uses_sapphire()) {
        nk_f32_to_f16_sapphire(src, dest);
        return;
    }
#endif
#if NK_TARGET_HASWELL
    if (nk_uses_haswell()) {
        nk_f32_to_f16_haswell(src, dest);
        return;
    }
#endif
#if NK_TARGET_NEON
    if (nk_uses_neon()) {
        nk_f32_to_f16_neon(src, dest);
        return;
    }
#endif
    nk_f32_to_f16_serial(src, dest);
}

// bf16, e4m3, e5m2 scalar conversions - serial only (no ISA-specific variants)
NK_DYNAMIC void nk_bf16_to_f32(nk_bf16_t const *src, nk_f32_t *dest) { nk_bf16_to_f32_serial(src, dest); }
NK_DYNAMIC void nk_e4m3_to_f32(nk_e4m3_t const *src, nk_f32_t *dest) { nk_e4m3_to_f32_serial(src, dest); }
NK_DYNAMIC void nk_e5m2_to_f32(nk_e5m2_t const *src, nk_f32_t *dest) { nk_e5m2_to_f32_serial(src, dest); }
NK_DYNAMIC void nk_f32_to_bf16(nk_f32_t const *src, nk_bf16_t *dest) { nk_f32_to_bf16_serial(src, dest); }
NK_DYNAMIC void nk_f32_to_e4m3(nk_f32_t const *src, nk_e4m3_t *dest) { nk_f32_to_e4m3_serial(src, dest); }
NK_DYNAMIC void nk_f32_to_e5m2(nk_f32_t const *src, nk_e5m2_t *dest) { nk_f32_to_e5m2_serial(src, dest); }

NK_DYNAMIC void nk_cast(void const *from, nk_dtype_t from_type, nk_size_t n, void *to, nk_dtype_t to_type) {
    static nk_kernel_cast_punned_t kernel = 0;
    if (kernel == 0) {
        nk_capability_t used_capability;
        nk_find_kernel_punned(nk_kernel_cast_k, nk_dtype_unknown_k, nk_capabilities(), nk_cap_any_k,
                              (nk_kernel_punned_t *)&kernel, &used_capability);
        if (!kernel) {
            nk_cast_serial(from, from_type, n, to, to_type);
            return;
        }
    }
    kernel(from, from_type, n, to, to_type);
}

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
    nk_scalar_buffer_t dummy_input;
    nk_dots_packed_buffer_header_t dummy_tensor_header;
    void *x = &dummy_input;
    nk_f64_t dummy_alpha = 1, dummy_beta = 1;
    nk_size_t dummy_index;

    // Dense:
    nk_dot_i8((nk_i8_t *)x, (nk_i8_t *)x, 0, dummy_results);
    nk_dot_u8((nk_u8_t *)x, (nk_u8_t *)x, 0, dummy_results);
    nk_dot_i4((nk_i4x2_t *)x, (nk_i4x2_t *)x, 0, dummy_results);
    nk_dot_u4((nk_u4x2_t *)x, (nk_u4x2_t *)x, 0, dummy_results);
    nk_dot_f16((nk_f16_t *)x, (nk_f16_t *)x, 0, dummy_results);
    nk_dot_bf16((nk_bf16_t *)x, (nk_bf16_t *)x, 0, dummy_results);
    nk_dot_f32((nk_f32_t *)x, (nk_f32_t *)x, 0, dummy_results);
    nk_dot_f64((nk_f64_t *)x, (nk_f64_t *)x, 0, dummy_results);
    nk_dot_e4m3((nk_e4m3_t *)x, (nk_e4m3_t *)x, 0, dummy_results);
    nk_dot_e5m2((nk_e5m2_t *)x, (nk_e5m2_t *)x, 0, dummy_results);

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
    nk_angular_i4((nk_i4x2_t *)x, (nk_i4x2_t *)x, 0, dummy_results);
    nk_angular_u4((nk_u4x2_t *)x, (nk_u4x2_t *)x, 0, dummy_results);
    nk_angular_f16((nk_f16_t *)x, (nk_f16_t *)x, 0, dummy_results);
    nk_angular_bf16((nk_bf16_t *)x, (nk_bf16_t *)x, 0, dummy_results);
    nk_angular_f32((nk_f32_t *)x, (nk_f32_t *)x, 0, dummy_results);
    nk_angular_f64((nk_f64_t *)x, (nk_f64_t *)x, 0, dummy_results);
    nk_angular_e4m3((nk_e4m3_t *)x, (nk_e4m3_t *)x, 0, dummy_results);
    nk_angular_e5m2((nk_e5m2_t *)x, (nk_e5m2_t *)x, 0, dummy_results);

    nk_l2sq_i8((nk_i8_t *)x, (nk_i8_t *)x, 0, dummy_results);
    nk_l2sq_u8((nk_u8_t *)x, (nk_u8_t *)x, 0, dummy_results);
    nk_l2sq_i4((nk_i4x2_t *)x, (nk_i4x2_t *)x, 0, dummy_results);
    nk_l2sq_u4((nk_u4x2_t *)x, (nk_u4x2_t *)x, 0, dummy_results);
    nk_l2sq_f16((nk_f16_t *)x, (nk_f16_t *)x, 0, dummy_results);
    nk_l2sq_bf16((nk_bf16_t *)x, (nk_bf16_t *)x, 0, dummy_results);
    nk_l2sq_f32((nk_f32_t *)x, (nk_f32_t *)x, 0, dummy_results);
    nk_l2sq_f64((nk_f64_t *)x, (nk_f64_t *)x, 0, dummy_results);
    nk_l2sq_e4m3((nk_e4m3_t *)x, (nk_e4m3_t *)x, 0, dummy_results);
    nk_l2sq_e5m2((nk_e5m2_t *)x, (nk_e5m2_t *)x, 0, dummy_results);

    nk_l2_i8((nk_i8_t *)x, (nk_i8_t *)x, 0, dummy_results);
    nk_l2_u8((nk_u8_t *)x, (nk_u8_t *)x, 0, dummy_results);
    nk_l2_i4((nk_i4x2_t *)x, (nk_i4x2_t *)x, 0, dummy_results);
    nk_l2_u4((nk_u4x2_t *)x, (nk_u4x2_t *)x, 0, dummy_results);
    nk_l2_f16((nk_f16_t *)x, (nk_f16_t *)x, 0, dummy_results);
    nk_l2_bf16((nk_bf16_t *)x, (nk_bf16_t *)x, 0, dummy_results);
    nk_l2_f32((nk_f32_t *)x, (nk_f32_t *)x, 0, dummy_results);
    nk_l2_f64((nk_f64_t *)x, (nk_f64_t *)x, 0, dummy_results);
    nk_l2_e4m3((nk_e4m3_t *)x, (nk_e4m3_t *)x, 0, dummy_results);
    nk_l2_e5m2((nk_e5m2_t *)x, (nk_e5m2_t *)x, 0, dummy_results);

    nk_haversine_f64((nk_f64_t *)x, (nk_f64_t *)x, (nk_f64_t *)x, (nk_f64_t *)x, 0, dummy_results);
    nk_haversine_f32((nk_f32_t *)x, (nk_f32_t *)x, (nk_f32_t *)x, (nk_f32_t *)x, 0, dummy_results);
    nk_vincenty_f64((nk_f64_t *)x, (nk_f64_t *)x, (nk_f64_t *)x, (nk_f64_t *)x, 0, dummy_results);
    nk_vincenty_f32((nk_f32_t *)x, (nk_f32_t *)x, (nk_f32_t *)x, (nk_f32_t *)x, 0, dummy_results);

    nk_hamming_u1((nk_u1x8_t *)x, (nk_u1x8_t *)x, 0, dummy_results);
    nk_jaccard_u1((nk_u1x8_t *)x, (nk_u1x8_t *)x, 0, dummy_results);

    nk_kld_f16((nk_f16_t *)x, (nk_f16_t *)x, 0, dummy_results);
    nk_kld_bf16((nk_bf16_t *)x, (nk_bf16_t *)x, 0, dummy_results);
    nk_kld_f32((nk_f32_t *)x, (nk_f32_t *)x, 0, dummy_results);
    nk_kld_f64((nk_f64_t *)x, (nk_f64_t *)x, 0, dummy_results);
    nk_jsd_f16((nk_f16_t *)x, (nk_f16_t *)x, 0, dummy_results);
    nk_jsd_bf16((nk_bf16_t *)x, (nk_bf16_t *)x, 0, dummy_results);
    nk_jsd_f32((nk_f32_t *)x, (nk_f32_t *)x, 0, dummy_results);
    nk_jsd_f64((nk_f64_t *)x, (nk_f64_t *)x, 0, dummy_results);

    // Sparse
    nk_sparse_intersect_u16((nk_u16_t *)x, (nk_u16_t *)x, 0, 0, (nk_u16_t *)x, (nk_size_t *)dummy_results);
    nk_sparse_intersect_u32((nk_u32_t *)x, (nk_u32_t *)x, 0, 0, (nk_u32_t *)x, (nk_size_t *)dummy_results);
    nk_sparse_intersect_u64((nk_u64_t *)x, (nk_u64_t *)x, 0, 0, (nk_u64_t *)x, (nk_size_t *)dummy_results);

    // Curved:
    nk_bilinear_f64((nk_f64_t *)x, (nk_f64_t *)x, (nk_f64_t *)x, 0, dummy_results);
    nk_mahalanobis_f64((nk_f64_t *)x, (nk_f64_t *)x, (nk_f64_t *)x, 0, dummy_results);
    nk_bilinear_f32((nk_f32_t *)x, (nk_f32_t *)x, (nk_f32_t *)x, 0, dummy_results);
    nk_mahalanobis_f32((nk_f32_t *)x, (nk_f32_t *)x, (nk_f32_t *)x, 0, dummy_results);
    nk_bilinear_f16((nk_f16_t *)x, (nk_f16_t *)x, (nk_f16_t *)x, 0, dummy_results);
    nk_mahalanobis_f16((nk_f16_t *)x, (nk_f16_t *)x, (nk_f16_t *)x, 0, dummy_results);
    nk_bilinear_bf16((nk_bf16_t *)x, (nk_bf16_t *)x, (nk_bf16_t *)x, 0, dummy_results);
    nk_mahalanobis_bf16((nk_bf16_t *)x, (nk_bf16_t *)x, (nk_bf16_t *)x, 0, dummy_results);
    nk_bilinear_f64c((nk_f64c_t *)x, (nk_f64c_t *)x, (nk_f64c_t *)x, 0, dummy_results);
    nk_bilinear_f32c((nk_f32c_t *)x, (nk_f32c_t *)x, (nk_f32c_t *)x, 0, dummy_results);
    nk_bilinear_f16c((nk_f16c_t *)x, (nk_f16c_t *)x, (nk_f16c_t *)x, 0, dummy_results);
    nk_bilinear_bf16c((nk_bf16c_t *)x, (nk_bf16c_t *)x, (nk_bf16c_t *)x, 0, dummy_results);

    // Elementwise
    nk_each_blend_f64((nk_f64_t *)x, (nk_f64_t *)x, 0, &dummy_alpha, &dummy_beta, (nk_f64_t *)x);
    nk_each_blend_f32((nk_f32_t *)x, (nk_f32_t *)x, 0, (nk_f32_t *)&dummy_alpha, (nk_f32_t *)&dummy_beta,
                      (nk_f32_t *)x);
    nk_each_blend_f16((nk_f16_t *)x, (nk_f16_t *)x, 0, (nk_f32_t *)&dummy_alpha, (nk_f32_t *)&dummy_beta,
                      (nk_f16_t *)x);
    nk_each_blend_bf16((nk_bf16_t *)x, (nk_bf16_t *)x, 0, (nk_f32_t *)&dummy_alpha, (nk_f32_t *)&dummy_beta,
                       (nk_bf16_t *)x);
    nk_each_blend_i8((nk_i8_t *)x, (nk_i8_t *)x, 0, (nk_f32_t *)&dummy_alpha, (nk_f32_t *)&dummy_beta, (nk_i8_t *)x);
    nk_each_blend_u8((nk_u8_t *)x, (nk_u8_t *)x, 0, (nk_f32_t *)&dummy_alpha, (nk_f32_t *)&dummy_beta, (nk_u8_t *)x);
    nk_each_fma_f64((nk_f64_t *)x, (nk_f64_t *)x, (nk_f64_t *)x, 0, &dummy_alpha, &dummy_beta, (nk_f64_t *)x);
    nk_each_fma_f32((nk_f32_t *)x, (nk_f32_t *)x, (nk_f32_t *)x, 0, (nk_f32_t *)&dummy_alpha, (nk_f32_t *)&dummy_beta,
                    (nk_f32_t *)x);
    nk_each_fma_f16((nk_f16_t *)x, (nk_f16_t *)x, (nk_f16_t *)x, 0, (nk_f32_t *)&dummy_alpha, (nk_f32_t *)&dummy_beta,
                    (nk_f16_t *)x);
    nk_each_fma_bf16((nk_bf16_t *)x, (nk_bf16_t *)x, (nk_bf16_t *)x, 0, (nk_f32_t *)&dummy_alpha,
                     (nk_f32_t *)&dummy_beta, (nk_bf16_t *)x);
    nk_each_fma_i8((nk_i8_t *)x, (nk_i8_t *)x, (nk_i8_t *)x, 0, (nk_f32_t *)&dummy_alpha, (nk_f32_t *)&dummy_beta,
                   (nk_i8_t *)x);
    nk_each_fma_u8((nk_u8_t *)x, (nk_u8_t *)x, (nk_u8_t *)x, 0, (nk_f32_t *)&dummy_alpha, (nk_f32_t *)&dummy_beta,
                   (nk_u8_t *)x);

    // Sparse dot products
    nk_jaccard_u32((nk_u32_t *)x, (nk_u32_t *)x, 0, dummy_results);
    nk_sparse_dot_u16bf16((nk_u16_t *)x, (nk_u16_t *)x, (nk_bf16_t *)x, (nk_bf16_t *)x, 0, 0, dummy_results);
    nk_sparse_dot_u32f32((nk_u32_t *)x, (nk_u32_t *)x, (nk_f32_t *)x, (nk_f32_t *)x, 0, 0, dummy_results);

    // Trigonometry
    nk_sin_f32((nk_f32_t *)x, 0, (nk_f32_t *)x);
    nk_sin_f64((nk_f64_t *)x, 0, (nk_f64_t *)x);
    nk_cos_f32((nk_f32_t *)x, 0, (nk_f32_t *)x);
    nk_cos_f64((nk_f64_t *)x, 0, (nk_f64_t *)x);
    nk_atan_f32((nk_f32_t *)x, 0, (nk_f32_t *)x);
    nk_atan_f64((nk_f64_t *)x, 0, (nk_f64_t *)x);

    // Mesh alignment
    nk_rmsd_f32((nk_f32_t *)x, (nk_f32_t *)x, 0, (nk_f32_t *)x, (nk_f32_t *)x, (nk_f32_t *)x, (nk_f32_t *)x,
                (nk_f32_t *)x);
    nk_rmsd_f64((nk_f64_t *)x, (nk_f64_t *)x, 0, (nk_f64_t *)x, (nk_f64_t *)x, (nk_f64_t *)x, (nk_f64_t *)x,
                (nk_f64_t *)x);
    nk_kabsch_f32((nk_f32_t *)x, (nk_f32_t *)x, 0, (nk_f32_t *)x, (nk_f32_t *)x, (nk_f32_t *)x, (nk_f32_t *)x,
                  (nk_f32_t *)x);
    nk_kabsch_f64((nk_f64_t *)x, (nk_f64_t *)x, 0, (nk_f64_t *)x, (nk_f64_t *)x, (nk_f64_t *)x, (nk_f64_t *)x,
                  (nk_f64_t *)x);
    nk_umeyama_f32((nk_f32_t *)x, (nk_f32_t *)x, 0, (nk_f32_t *)x, (nk_f32_t *)x, (nk_f32_t *)x, (nk_f32_t *)x,
                   (nk_f32_t *)x);
    nk_umeyama_f64((nk_f64_t *)x, (nk_f64_t *)x, 0, (nk_f64_t *)x, (nk_f64_t *)x, (nk_f64_t *)x, (nk_f64_t *)x,
                   (nk_f64_t *)x);

    // Scale
    nk_each_scale_f64((nk_f64_t *)x, 0, &dummy_alpha, &dummy_beta, (nk_f64_t *)x);
    nk_each_scale_f32((nk_f32_t *)x, 0, (nk_f32_t *)&dummy_alpha, (nk_f32_t *)&dummy_beta, (nk_f32_t *)x);
    nk_each_scale_f16((nk_f16_t *)x, 0, (nk_f32_t *)&dummy_alpha, (nk_f32_t *)&dummy_beta, (nk_f16_t *)x);
    nk_each_scale_bf16((nk_bf16_t *)x, 0, (nk_f32_t *)&dummy_alpha, (nk_f32_t *)&dummy_beta, (nk_bf16_t *)x);
    nk_each_scale_i8((nk_i8_t *)x, 0, (nk_f32_t *)&dummy_alpha, (nk_f32_t *)&dummy_beta, (nk_i8_t *)x);
    nk_each_scale_u8((nk_u8_t *)x, 0, (nk_f32_t *)&dummy_alpha, (nk_f32_t *)&dummy_beta, (nk_u8_t *)x);
    nk_each_scale_i16((nk_i16_t *)x, 0, (nk_f32_t *)&dummy_alpha, (nk_f32_t *)&dummy_beta, (nk_i16_t *)x);
    nk_each_scale_u16((nk_u16_t *)x, 0, (nk_f32_t *)&dummy_alpha, (nk_f32_t *)&dummy_beta, (nk_u16_t *)x);
    nk_each_scale_i32((nk_i32_t *)x, 0, &dummy_alpha, &dummy_beta, (nk_i32_t *)x);
    nk_each_scale_u32((nk_u32_t *)x, 0, &dummy_alpha, &dummy_beta, (nk_u32_t *)x);
    nk_each_scale_i64((nk_i64_t *)x, 0, &dummy_alpha, &dummy_beta, (nk_i64_t *)x);
    nk_each_scale_u64((nk_u64_t *)x, 0, &dummy_alpha, &dummy_beta, (nk_u64_t *)x);
    nk_each_scale_e4m3((nk_e4m3_t *)x, 0, (nk_f32_t *)&dummy_alpha, (nk_f32_t *)&dummy_beta, (nk_e4m3_t *)x);
    nk_each_scale_e5m2((nk_e5m2_t *)x, 0, (nk_f32_t *)&dummy_alpha, (nk_f32_t *)&dummy_beta, (nk_e5m2_t *)x);

    // Sum
    nk_each_sum_f64((nk_f64_t *)x, (nk_f64_t *)x, 0, (nk_f64_t *)x);
    nk_each_sum_f32((nk_f32_t *)x, (nk_f32_t *)x, 0, (nk_f32_t *)x);
    nk_each_sum_f16((nk_f16_t *)x, (nk_f16_t *)x, 0, (nk_f16_t *)x);
    nk_each_sum_bf16((nk_bf16_t *)x, (nk_bf16_t *)x, 0, (nk_bf16_t *)x);
    nk_each_sum_i8((nk_i8_t *)x, (nk_i8_t *)x, 0, (nk_i8_t *)x);
    nk_each_sum_u8((nk_u8_t *)x, (nk_u8_t *)x, 0, (nk_u8_t *)x);
    nk_each_sum_i16((nk_i16_t *)x, (nk_i16_t *)x, 0, (nk_i16_t *)x);
    nk_each_sum_u16((nk_u16_t *)x, (nk_u16_t *)x, 0, (nk_u16_t *)x);
    nk_each_sum_i32((nk_i32_t *)x, (nk_i32_t *)x, 0, (nk_i32_t *)x);
    nk_each_sum_u32((nk_u32_t *)x, (nk_u32_t *)x, 0, (nk_u32_t *)x);
    nk_each_sum_i64((nk_i64_t *)x, (nk_i64_t *)x, 0, (nk_i64_t *)x);
    nk_each_sum_u64((nk_u64_t *)x, (nk_u64_t *)x, 0, (nk_u64_t *)x);
    nk_each_sum_e4m3((nk_e4m3_t *)x, (nk_e4m3_t *)x, 0, (nk_e4m3_t *)x);
    nk_each_sum_e5m2((nk_e5m2_t *)x, (nk_e5m2_t *)x, 0, (nk_e5m2_t *)x);

    // FP8 blend/fma
    nk_each_blend_e4m3((nk_e4m3_t *)x, (nk_e4m3_t *)x, 0, (nk_f32_t *)&dummy_alpha, (nk_f32_t *)&dummy_beta,
                       (nk_e4m3_t *)x);
    nk_each_blend_e5m2((nk_e5m2_t *)x, (nk_e5m2_t *)x, 0, (nk_f32_t *)&dummy_alpha, (nk_f32_t *)&dummy_beta,
                       (nk_e5m2_t *)x);
    nk_each_fma_e4m3((nk_e4m3_t *)x, (nk_e4m3_t *)x, (nk_e4m3_t *)x, 0, (nk_f32_t *)&dummy_alpha,
                     (nk_f32_t *)&dummy_beta, (nk_e4m3_t *)x);
    nk_each_fma_e5m2((nk_e5m2_t *)x, (nk_e5m2_t *)x, (nk_e5m2_t *)x, 0, (nk_f32_t *)&dummy_alpha,
                     (nk_f32_t *)&dummy_beta, (nk_e5m2_t *)x);

    // Reduce add
    nk_reduce_add_f32((nk_f32_t *)x, 0, 0, (nk_f64_t *)x);
    nk_reduce_add_f64((nk_f64_t *)x, 0, 0, (nk_f64_t *)x);
    nk_reduce_add_i8((nk_i8_t *)x, 0, 0, (nk_i64_t *)x);
    nk_reduce_add_u8((nk_u8_t *)x, 0, 0, (nk_u64_t *)x);
    nk_reduce_add_i16((nk_i16_t *)x, 0, 0, (nk_i64_t *)x);
    nk_reduce_add_u16((nk_u16_t *)x, 0, 0, (nk_u64_t *)x);
    nk_reduce_add_i32((nk_i32_t *)x, 0, 0, (nk_i64_t *)x);
    nk_reduce_add_u32((nk_u32_t *)x, 0, 0, (nk_u64_t *)x);
    nk_reduce_add_i64((nk_i64_t *)x, 0, 0, (nk_i64_t *)x);
    nk_reduce_add_u64((nk_u64_t *)x, 0, 0, (nk_u64_t *)x);
    nk_reduce_add_f16((nk_f16_t *)x, 0, 0, (nk_f32_t *)x);
    nk_reduce_add_bf16((nk_bf16_t *)x, 0, 0, (nk_f32_t *)x);
    nk_reduce_add_e4m3((nk_e4m3_t *)x, 0, 0, (nk_f32_t *)x);
    nk_reduce_add_e5m2((nk_e5m2_t *)x, 0, 0, (nk_f32_t *)x);

    // Reduce min/max
    nk_reduce_min_f32((nk_f32_t *)x, 0, 0, (nk_f32_t *)x, &dummy_index);
    nk_reduce_max_f32((nk_f32_t *)x, 0, 0, (nk_f32_t *)x, &dummy_index);
    nk_reduce_min_f64((nk_f64_t *)x, 0, 0, (nk_f64_t *)x, &dummy_index);
    nk_reduce_max_f64((nk_f64_t *)x, 0, 0, (nk_f64_t *)x, &dummy_index);
    nk_reduce_min_i8((nk_i8_t *)x, 0, 0, (nk_i8_t *)x, &dummy_index);
    nk_reduce_max_i8((nk_i8_t *)x, 0, 0, (nk_i8_t *)x, &dummy_index);
    nk_reduce_min_u8((nk_u8_t *)x, 0, 0, (nk_u8_t *)x, &dummy_index);
    nk_reduce_max_u8((nk_u8_t *)x, 0, 0, (nk_u8_t *)x, &dummy_index);
    nk_reduce_min_i16((nk_i16_t *)x, 0, 0, (nk_i16_t *)x, &dummy_index);
    nk_reduce_max_i16((nk_i16_t *)x, 0, 0, (nk_i16_t *)x, &dummy_index);
    nk_reduce_min_u16((nk_u16_t *)x, 0, 0, (nk_u16_t *)x, &dummy_index);
    nk_reduce_max_u16((nk_u16_t *)x, 0, 0, (nk_u16_t *)x, &dummy_index);
    nk_reduce_min_i32((nk_i32_t *)x, 0, 0, (nk_i32_t *)x, &dummy_index);
    nk_reduce_max_i32((nk_i32_t *)x, 0, 0, (nk_i32_t *)x, &dummy_index);
    nk_reduce_min_u32((nk_u32_t *)x, 0, 0, (nk_u32_t *)x, &dummy_index);
    nk_reduce_max_u32((nk_u32_t *)x, 0, 0, (nk_u32_t *)x, &dummy_index);
    nk_reduce_min_i64((nk_i64_t *)x, 0, 0, (nk_i64_t *)x, &dummy_index);
    nk_reduce_max_i64((nk_i64_t *)x, 0, 0, (nk_i64_t *)x, &dummy_index);
    nk_reduce_min_u64((nk_u64_t *)x, 0, 0, (nk_u64_t *)x, &dummy_index);
    nk_reduce_max_u64((nk_u64_t *)x, 0, 0, (nk_u64_t *)x, &dummy_index);
    nk_reduce_min_f16((nk_f16_t *)x, 0, 0, (nk_f32_t *)x, &dummy_index);
    nk_reduce_max_f16((nk_f16_t *)x, 0, 0, (nk_f32_t *)x, &dummy_index);
    nk_reduce_min_bf16((nk_bf16_t *)x, 0, 0, (nk_f32_t *)x, &dummy_index);
    nk_reduce_max_bf16((nk_bf16_t *)x, 0, 0, (nk_f32_t *)x, &dummy_index);
    nk_reduce_min_e4m3((nk_e4m3_t *)x, 0, 0, (nk_f32_t *)x, &dummy_index);
    nk_reduce_max_e4m3((nk_e4m3_t *)x, 0, 0, (nk_f32_t *)x, &dummy_index);
    nk_reduce_min_e5m2((nk_e5m2_t *)x, 0, 0, (nk_f32_t *)x, &dummy_index);
    nk_reduce_max_e5m2((nk_e5m2_t *)x, 0, 0, (nk_f32_t *)x, &dummy_index);

    // Matrix multiplications (dots)
    nk_dots_packed_size_f32(0, 0);
    nk_dots_packed_size_f64(0, 0);
    nk_dots_packed_size_f16(0, 0);
    nk_dots_packed_size_bf16(0, 0);
    nk_dots_packed_size_i8(0, 0);
    nk_dots_packed_size_u8(0, 0);
    nk_dots_packed_size_e4m3(0, 0);
    nk_dots_packed_size_e5m2(0, 0);
    nk_dots_pack_f32((nk_f32_t *)x, 0, 0, 0, x);
    nk_dots_pack_f64((nk_f64_t *)x, 0, 0, 0, x);
    nk_dots_pack_f16((nk_f16_t *)x, 0, 0, 0, x);
    nk_dots_pack_bf16((nk_bf16_t *)x, 0, 0, 0, x);
    nk_dots_pack_i8((nk_i8_t *)x, 0, 0, 0, x);
    nk_dots_pack_u8((nk_u8_t *)x, 0, 0, 0, x);
    nk_dots_pack_e4m3((nk_e4m3_t *)x, 0, 0, 0, x);
    nk_dots_pack_e5m2((nk_e5m2_t *)x, 0, 0, 0, x);
    nk_dots_packed_f32((nk_f32_t *)x, (void *)&dummy_tensor_header, (nk_f32_t *)x, 0, 0, 0, 0, 0);
    nk_dots_packed_f64((nk_f64_t *)x, (void *)&dummy_tensor_header, (nk_f64_t *)x, 0, 0, 0, 0, 0);
    nk_dots_packed_f16((nk_f16_t *)x, (void *)&dummy_tensor_header, (nk_f32_t *)x, 0, 0, 0, 0, 0);
    nk_dots_packed_bf16((nk_bf16_t *)x, (void *)&dummy_tensor_header, (nk_f32_t *)x, 0, 0, 0, 0, 0);
    nk_dots_packed_i8((nk_i8_t *)x, (void *)&dummy_tensor_header, (nk_i32_t *)x, 0, 0, 0, 0, 0);
    nk_dots_packed_u8((nk_u8_t *)x, (void *)&dummy_tensor_header, (nk_u32_t *)x, 0, 0, 0, 0, 0);
    nk_dots_packed_e4m3((nk_e4m3_t *)x, (void *)&dummy_tensor_header, (nk_f32_t *)x, 0, 0, 0, 0, 0);
    nk_dots_packed_e5m2((nk_e5m2_t *)x, (void *)&dummy_tensor_header, (nk_f32_t *)x, 0, 0, 0, 0, 0);

    return static_capabilities;
}

NK_DYNAMIC void nk_find_kernel_punned( //
    nk_kernel_kind_t kind,             //
    nk_dtype_t dtype,                  //
    nk_capability_t supported,         //
    nk_capability_t allowed,           //
    nk_kernel_punned_t *kernel_output, //
    nk_capability_t *capability_output) {
    nk_find_kernel_punned_(kind, dtype, supported, allowed, kernel_output, capability_output);
}

#ifdef __cplusplus
}
#endif

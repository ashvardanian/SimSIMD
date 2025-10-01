/**
 *  @brief  Dynamic dispatch library for MathKong.
 *  @note   Compile with the most recent compiler available.
 *  @file   lib.c
 */
#define SIMSIMD_DYNAMIC_DISPATCH 1
#define SIMSIMD_NATIVE_F16 0
#define SIMSIMD_NATIVE_BF16 0

/*  Override the primary serial operations to avoid the LibC dependency.
 */
#define SIMSIMD_SQRT(x) mathkong_f32_sqrt(x)
#define SIMSIMD_RSQRT(x) mathkong_f32_rsqrt(x)
#define SIMSIMD_LOG(x) mathkong_f32_log(x)

/*  Depending on the Operating System, the following intrinsics are available
 *  on recent compiler toolchains:
 *
 *  - Linux: everything is available in GCC 12+ and Clang 16+.
 *  - Windows - MSVC: everything except Sapphire Rapids and ARM SVE.
 *  - macOS - Apple Clang: only Arm NEON and x86 AVX2 Haswell extensions are available.
 */
#if !defined(SIMSIMD_TARGET_NEON) && (defined(__APPLE__) || defined(__linux__))
#define SIMSIMD_TARGET_NEON 1
#endif
#if !defined(SIMSIMD_TARGET_SVE) && (defined(__linux__))
#define SIMSIMD_TARGET_SVE 1
#endif
#if !defined(SIMSIMD_TARGET_SVE2) && (defined(__linux__))
#define SIMSIMD_TARGET_SVE2 1
#endif
#if !defined(SIMSIMD_TARGET_HASWELL) && (defined(_MSC_VER) || defined(__APPLE__) || defined(__linux__))
#define SIMSIMD_TARGET_HASWELL 1
#endif
#if !defined(SIMSIMD_TARGET_SKYLAKE) && (defined(_MSC_VER) || defined(__linux__))
#define SIMSIMD_TARGET_SKYLAKE 1
#endif
#if !defined(SIMSIMD_TARGET_ICE) && (defined(_MSC_VER) || defined(__linux__))
#define SIMSIMD_TARGET_ICE 1
#endif
#if !defined(SIMSIMD_TARGET_GENOA) && (defined(__linux__))
#define SIMSIMD_TARGET_GENOA 1
#endif
#if !defined(SIMSIMD_TARGET_SAPPHIRE) && (defined(__linux__))
#define SIMSIMD_TARGET_SAPPHIRE 1
#endif
#if !defined(SIMSIMD_TARGET_TURIN) && (defined(__linux__))
#define SIMSIMD_TARGET_TURIN 1
#endif
#if !defined(SIMSIMD_TARGET_SIERRA) && (defined(__linux__)) && 0 // TODO: Add target spec to GCC & Clang
#define SIMSIMD_TARGET_SIERRA 1
#endif

#include <mathkong/mathkong.h>

#ifdef __cplusplus
extern "C" {
#endif

// Every time a function is called, it checks if the metric is already loaded. If not, it fetches it.
// If no metric is found, it returns NaN. We can obtain NaN by dividing 0.0 by 0.0, but that annoys
// the MSVC compiler. Instead we can directly write-in the signaling NaN (0x7FF0000000000001)
// or the qNaN (0x7FF8000000000000).
#define SIMSIMD_DECLARATION_DENSE(name, extension)                                                           \
    SIMSIMD_DYNAMIC void mathkong_##name##_##extension(mathkong_##extension##_t const *a,                    \
                                                       mathkong_##extension##_t const *b, mathkong_size_t n, \
                                                       mathkong_distance_t *results) {                       \
        static mathkong_metric_dense_punned_t metric = 0;                                                    \
        if (metric == 0) {                                                                                   \
            mathkong_capability_t used_capability;                                                           \
            mathkong_find_kernel_punned(mathkong_metric_##name##_k, mathkong_datatype_##extension##_k,       \
                                        mathkong_capabilities(), mathkong_cap_any_k,                         \
                                        (mathkong_kernel_punned_t *)&metric, &used_capability);              \
            if (!metric) {                                                                                   \
                *(mathkong_u64_t *)results = 0x7FF0000000000001ull;                                          \
                return;                                                                                      \
            }                                                                                                \
        }                                                                                                    \
        metric(a, b, n, results);                                                                            \
    }

#define SIMSIMD_DECLARATION_SPARSE(name, extension)                                                                 \
    SIMSIMD_DYNAMIC void mathkong_##name##_##extension(mathkong_##extension##_t const *a,                           \
                                                       mathkong_##extension##_t const *b, mathkong_size_t a_length, \
                                                       mathkong_size_t b_length, mathkong_distance_t *result) {     \
        static mathkong_sparse_metric_t metric = 0;                                                                 \
        if (metric == 0) {                                                                                          \
            mathkong_capability_t used_capability;                                                                  \
            mathkong_find_kernel(mathkong_##name##_k, mathkong_##extension##_k, mathkong_capabilities(),            \
                                 mathkong_cap_any_k, (mathkong_kernel_punned_t *)(&metric), &used_capability);      \
            if (!metric) {                                                                                          \
                *(mathkong_u64_t *)result = 0x7FF0000000000001ull;                                                  \
                return;                                                                                             \
            }                                                                                                       \
        }                                                                                                           \
        metric(a, b, a_length, b_length, result);                                                                   \
    }

#define SIMSIMD_DECLARATION_CURVED(name, extension)                                                              \
    SIMSIMD_DYNAMIC void mathkong_##name##_##extension(                                                          \
        mathkong_##extension##_t const *a, mathkong_##extension##_t const *b, mathkong_##extension##_t const *c, \
        mathkong_size_t n, mathkong_distance_t *result) {                                                        \
        static mathkong_metric_curved_punned_t metric = 0;                                                       \
        if (metric == 0) {                                                                                       \
            mathkong_capability_t used_capability;                                                               \
            mathkong_find_kernel_punned(mathkong_metric_##name##_k, mathkong_datatype_##extension##_k,           \
                                        mathkong_capabilities(), mathkong_cap_any_k,                             \
                                        (mathkong_kernel_punned_t *)(&metric), &used_capability);                \
            if (!metric) {                                                                                       \
                *(mathkong_u64_t *)result = 0x7FF0000000000001ull;                                               \
                return;                                                                                          \
            }                                                                                                    \
        }                                                                                                        \
        metric(a, b, c, n, result);                                                                              \
    }

#define SIMSIMD_DECLARATION_SCALE(name, extension)                                                             \
    SIMSIMD_DYNAMIC void mathkong_##name##_##extension(mathkong_##extension##_t const *a, mathkong_size_t n,   \
                                                       mathkong_distance_t alpha, mathkong_distance_t beta,    \
                                                       mathkong_##extension##_t *result) {                     \
        static mathkong_elementwise_scale_t metric = 0;                                                        \
        if (metric == 0) {                                                                                     \
            mathkong_capability_t used_capability;                                                             \
            mathkong_find_kernel(mathkong_##name##_k, mathkong_##extension##_k, mathkong_capabilities(),       \
                                 mathkong_cap_any_k, (mathkong_kernel_punned_t *)(&metric), &used_capability); \
        }                                                                                                      \
        metric(a, n, alpha, beta, result);                                                                     \
    }

#define SIMSIMD_DECLARATION_SUM(name, extension)                                                               \
    SIMSIMD_DYNAMIC void mathkong_##name##_##extension(mathkong_##extension##_t const *a,                      \
                                                       mathkong_##extension##_t const *b, mathkong_size_t n,   \
                                                       mathkong_##extension##_t *result) {                     \
        static mathkong_elementwise_sum_t metric = 0;                                                          \
        if (metric == 0) {                                                                                     \
            mathkong_capability_t used_capability;                                                             \
            mathkong_find_kernel(mathkong_##name##_k, mathkong_##extension##_k, mathkong_capabilities(),       \
                                 mathkong_cap_any_k, (mathkong_kernel_punned_t *)(&metric), &used_capability); \
        }                                                                                                      \
        metric(a, b, n, result);                                                                               \
    }

#define SIMSIMD_DECLARATION_FMA(name, extension)                                                                    \
    SIMSIMD_DYNAMIC void mathkong_##name##_##extension(                                                             \
        mathkong_##extension##_t const *a, mathkong_##extension##_t const *b, mathkong_##extension##_t const *c,    \
        mathkong_size_t n, mathkong_distance_t alpha, mathkong_distance_t beta, mathkong_##extension##_t *result) { \
        static mathkong_kernel_fma_punned_t metric = 0;                                                             \
        if (metric == 0) {                                                                                          \
            mathkong_capability_t used_capability;                                                                  \
            mathkong_find_kernel(mathkong_##name##_k, mathkong_##extension##_k, mathkong_capabilities(),            \
                                 mathkong_cap_any_k, (mathkong_kernel_punned_t *)(&metric), &used_capability);      \
        }                                                                                                           \
        metric(a, b, c, n, alpha, beta, result);                                                                    \
    }

#define SIMSIMD_DECLARATION_WSUM(name, extension)                                                      \
    SIMSIMD_DYNAMIC void mathkong_##name##_##extension(                                                \
        mathkong_##extension##_t const *a, mathkong_##extension##_t const *b, mathkong_size_t n,       \
        mathkong_distance_t alpha, mathkong_distance_t beta, mathkong_##extension##_t *result) {       \
        static mathkong_kernel_wsum_punned_t metric = 0;                                               \
        if (metric == 0) {                                                                             \
            mathkong_capability_t used_capability;                                                     \
            mathkong_find_kernel_punned(mathkong_metric_##name##_k, mathkong_datatype_##extension##_k, \
                                        mathkong_capabilities(), mathkong_cap_any_k,                   \
                                        (mathkong_kernel_punned_t *)(&metric), &used_capability);      \
        }                                                                                              \
        metric(a, b, n, alpha, beta, result);                                                          \
    }

// Dot products
SIMSIMD_DECLARATION_DENSE(dot, i8)
SIMSIMD_DECLARATION_DENSE(dot, u8)
SIMSIMD_DECLARATION_DENSE(dot, f16)
SIMSIMD_DECLARATION_DENSE(dot, bf16)
SIMSIMD_DECLARATION_DENSE(dot, f32)
SIMSIMD_DECLARATION_DENSE(dot, f64)
SIMSIMD_DECLARATION_DENSE(dot, f16c)
SIMSIMD_DECLARATION_DENSE(dot, bf16c)
SIMSIMD_DECLARATION_DENSE(dot, f32c)
SIMSIMD_DECLARATION_DENSE(dot, f64c)
SIMSIMD_DECLARATION_DENSE(vdot, f16c)
SIMSIMD_DECLARATION_DENSE(vdot, bf16c)
SIMSIMD_DECLARATION_DENSE(vdot, f32c)
SIMSIMD_DECLARATION_DENSE(vdot, f64c)

// Spatial distances
SIMSIMD_DECLARATION_DENSE(angular, i8)
SIMSIMD_DECLARATION_DENSE(angular, u8)
SIMSIMD_DECLARATION_DENSE(angular, f16)
SIMSIMD_DECLARATION_DENSE(angular, bf16)
SIMSIMD_DECLARATION_DENSE(angular, f32)
SIMSIMD_DECLARATION_DENSE(angular, f64)
SIMSIMD_DECLARATION_DENSE(l2sq, i8)
SIMSIMD_DECLARATION_DENSE(l2sq, u8)
SIMSIMD_DECLARATION_DENSE(l2sq, f16)
SIMSIMD_DECLARATION_DENSE(l2sq, bf16)
SIMSIMD_DECLARATION_DENSE(l2sq, f32)
SIMSIMD_DECLARATION_DENSE(l2sq, f64)
SIMSIMD_DECLARATION_DENSE(l2, i8)
SIMSIMD_DECLARATION_DENSE(l2, u8)
SIMSIMD_DECLARATION_DENSE(l2, f16)
SIMSIMD_DECLARATION_DENSE(l2, bf16)
SIMSIMD_DECLARATION_DENSE(l2, f32)
SIMSIMD_DECLARATION_DENSE(l2, f64)

// Binary distances
SIMSIMD_DECLARATION_DENSE(hamming, b8)
SIMSIMD_DECLARATION_DENSE(jaccard, b8)

// Probability distributions
SIMSIMD_DECLARATION_DENSE(kl, f16)
SIMSIMD_DECLARATION_DENSE(kl, bf16)
SIMSIMD_DECLARATION_DENSE(kl, f32)
SIMSIMD_DECLARATION_DENSE(kl, f64)
SIMSIMD_DECLARATION_DENSE(js, f16)
SIMSIMD_DECLARATION_DENSE(js, bf16)
SIMSIMD_DECLARATION_DENSE(js, f32)
SIMSIMD_DECLARATION_DENSE(js, f64)

// Sparse sets
SIMSIMD_DECLARATION_SPARSE(intersect, u16, u16)
SIMSIMD_DECLARATION_SPARSE(intersect, u32, u32)

// Curved spaces
SIMSIMD_DECLARATION_CURVED(bilinear, f64)
SIMSIMD_DECLARATION_CURVED(bilinear, f64c)
SIMSIMD_DECLARATION_CURVED(mahalanobis, f64)
SIMSIMD_DECLARATION_CURVED(bilinear, f32)
SIMSIMD_DECLARATION_CURVED(bilinear, f32c)
SIMSIMD_DECLARATION_CURVED(mahalanobis, f32)
SIMSIMD_DECLARATION_CURVED(bilinear, f16)
SIMSIMD_DECLARATION_CURVED(bilinear, f16c)
SIMSIMD_DECLARATION_CURVED(mahalanobis, f16)
SIMSIMD_DECLARATION_CURVED(bilinear, bf16)
SIMSIMD_DECLARATION_CURVED(bilinear, bf16c)
SIMSIMD_DECLARATION_CURVED(mahalanobis, bf16)

// Element-wise operations
SIMSIMD_DECLARATION_SUM(sum, f64)
SIMSIMD_DECLARATION_SUM(sum, f32)
SIMSIMD_DECLARATION_SUM(sum, f16)
SIMSIMD_DECLARATION_SUM(sum, bf16)
SIMSIMD_DECLARATION_SUM(sum, i8)
SIMSIMD_DECLARATION_SUM(sum, u8)
SIMSIMD_DECLARATION_SUM(sum, i16)
SIMSIMD_DECLARATION_SUM(sum, u16)
SIMSIMD_DECLARATION_SUM(sum, i32)
SIMSIMD_DECLARATION_SUM(sum, u32)
SIMSIMD_DECLARATION_SUM(sum, i64)
SIMSIMD_DECLARATION_SUM(sum, u64)
SIMSIMD_DECLARATION_SCALE(scale, f64)
SIMSIMD_DECLARATION_SCALE(scale, f32)
SIMSIMD_DECLARATION_SCALE(scale, f16)
SIMSIMD_DECLARATION_SCALE(scale, bf16)
SIMSIMD_DECLARATION_SCALE(scale, i8)
SIMSIMD_DECLARATION_SCALE(scale, u8)
SIMSIMD_DECLARATION_SCALE(scale, i16)
SIMSIMD_DECLARATION_SCALE(scale, u16)
SIMSIMD_DECLARATION_SCALE(scale, i32)
SIMSIMD_DECLARATION_SCALE(scale, u32)
SIMSIMD_DECLARATION_SCALE(scale, i64)
SIMSIMD_DECLARATION_SCALE(scale, u64)
SIMSIMD_DECLARATION_WSUM(wsum, f64)
SIMSIMD_DECLARATION_WSUM(wsum, f32)
SIMSIMD_DECLARATION_WSUM(wsum, f16)
SIMSIMD_DECLARATION_WSUM(wsum, bf16)
SIMSIMD_DECLARATION_WSUM(wsum, i8)
SIMSIMD_DECLARATION_WSUM(wsum, u8)
SIMSIMD_DECLARATION_WSUM(wsum, i16)
SIMSIMD_DECLARATION_WSUM(wsum, u16)
SIMSIMD_DECLARATION_WSUM(wsum, i32)
SIMSIMD_DECLARATION_WSUM(wsum, u32)
SIMSIMD_DECLARATION_WSUM(wsum, i64)
SIMSIMD_DECLARATION_WSUM(wsum, u64)
SIMSIMD_DECLARATION_FMA(fma, f64)
SIMSIMD_DECLARATION_FMA(fma, f32)
SIMSIMD_DECLARATION_FMA(fma, f16)
SIMSIMD_DECLARATION_FMA(fma, bf16)
SIMSIMD_DECLARATION_FMA(fma, i8)
SIMSIMD_DECLARATION_FMA(fma, u8)
SIMSIMD_DECLARATION_FMA(fma, i16)
SIMSIMD_DECLARATION_FMA(fma, u16)
SIMSIMD_DECLARATION_FMA(fma, i32)
SIMSIMD_DECLARATION_FMA(fma, u32)
SIMSIMD_DECLARATION_FMA(fma, i64)
SIMSIMD_DECLARATION_FMA(fma, u64)

SIMSIMD_DYNAMIC int mathkong_uses_neon(void) { return (mathkong_capabilities() & mathkong_cap_neon_k) != 0; }
SIMSIMD_DYNAMIC int mathkong_uses_neon_f16(void) { return (mathkong_capabilities() & mathkong_cap_neon_f16_k) != 0; }
SIMSIMD_DYNAMIC int mathkong_uses_neon_bf16(void) { return (mathkong_capabilities() & mathkong_cap_neon_bf16_k) != 0; }
SIMSIMD_DYNAMIC int mathkong_uses_neon_i8(void) { return (mathkong_capabilities() & mathkong_cap_neon_i8_k) != 0; }
SIMSIMD_DYNAMIC int mathkong_uses_sve(void) { return (mathkong_capabilities() & mathkong_cap_sve_k) != 0; }
SIMSIMD_DYNAMIC int mathkong_uses_sve_f16(void) { return (mathkong_capabilities() & mathkong_cap_sve_f16_k) != 0; }
SIMSIMD_DYNAMIC int mathkong_uses_sve_bf16(void) { return (mathkong_capabilities() & mathkong_cap_sve_bf16_k) != 0; }
SIMSIMD_DYNAMIC int mathkong_uses_sve_i8(void) { return (mathkong_capabilities() & mathkong_cap_sve_i8_k) != 0; }
SIMSIMD_DYNAMIC int mathkong_uses_haswell(void) { return (mathkong_capabilities() & mathkong_cap_haswell_k) != 0; }
SIMSIMD_DYNAMIC int mathkong_uses_skylake(void) { return (mathkong_capabilities() & mathkong_cap_skylake_k) != 0; }
SIMSIMD_DYNAMIC int mathkong_uses_ice(void) { return (mathkong_capabilities() & mathkong_cap_ice_k) != 0; }
SIMSIMD_DYNAMIC int mathkong_uses_genoa(void) { return (mathkong_capabilities() & mathkong_cap_genoa_k) != 0; }
SIMSIMD_DYNAMIC int mathkong_uses_sapphire(void) { return (mathkong_capabilities() & mathkong_cap_sapphire_k) != 0; }
SIMSIMD_DYNAMIC int mathkong_uses_turin(void) { return (mathkong_capabilities() & mathkong_cap_turin_k) != 0; }
SIMSIMD_DYNAMIC int mathkong_uses_sierra(void) { return (mathkong_capabilities() & mathkong_cap_sierra_k) != 0; }
SIMSIMD_DYNAMIC int mathkong_uses_dynamic_dispatch(void) { return 1; }
SIMSIMD_DYNAMIC int mathkong_flush_denormals(void) { return _mathkong_flush_denormals(); }

SIMSIMD_DYNAMIC mathkong_f32_t mathkong_f16_to_f32(mathkong_f16_t const *x_ptr) {
    return mathkong_f16_to_f32_implementation(x_ptr);
}

SIMSIMD_DYNAMIC void mathkong_f32_to_f16(mathkong_f32_t x, mathkong_f16_t *result_ptr) {
    mathkong_f32_to_f16_implementation(x, result_ptr);
}

SIMSIMD_DYNAMIC mathkong_f32_t mathkong_bf16_to_f32(mathkong_bf16_t const *x_ptr) {
    return mathkong_bf16_to_f32_implementation(x_ptr);
}

SIMSIMD_DYNAMIC void mathkong_f32_to_bf16(mathkong_f32_t x, mathkong_bf16_t *result_ptr) {
    mathkong_f32_to_bf16_implementation(x, result_ptr);
}

SIMSIMD_DYNAMIC mathkong_capability_t mathkong_capabilities(void) {
    //! The latency of the CPUID instruction can be over 100 cycles, so we cache the result.
    static mathkong_capability_t static_capabilities = mathkong_cap_any_k;
    if (static_capabilities != mathkong_cap_any_k) return static_capabilities;

    static_capabilities = _mathkong_capabilities_implementation();

    // In multithreaded applications we need to ensure that the function pointers are pre-initialized,
    // so the first time we are probing for capabilities, we should also probe all of our metrics
    // with dummy inputs:
    mathkong_distance_t dummy_results_buffer[2];
    mathkong_distance_t *dummy_results = &dummy_results_buffer[0];

    // Passing `NULL` as `x` will trigger all kinds of `nonull` warnings on GCC.
    typedef double largest_scalar_t;
    largest_scalar_t dummy_input[1];
    void *x = &dummy_input[0];

    // Dense:
    mathkong_dot_i8((mathkong_i8_t *)x, (mathkong_i8_t *)x, 0, dummy_results);
    mathkong_dot_u8((mathkong_u8_t *)x, (mathkong_u8_t *)x, 0, dummy_results);
    mathkong_dot_f16((mathkong_f16_t *)x, (mathkong_f16_t *)x, 0, dummy_results);
    mathkong_dot_bf16((mathkong_bf16_t *)x, (mathkong_bf16_t *)x, 0, dummy_results);
    mathkong_dot_f32((mathkong_f32_t *)x, (mathkong_f32_t *)x, 0, dummy_results);
    mathkong_dot_f64((mathkong_f64_t *)x, (mathkong_f64_t *)x, 0, dummy_results);

    mathkong_dot_f16c((mathkong_f16c_t *)x, (mathkong_f16c_t *)x, 0, dummy_results);
    mathkong_dot_bf16c((mathkong_bf16c_t *)x, (mathkong_bf16c_t *)x, 0, dummy_results);
    mathkong_dot_f32c((mathkong_f32c_t *)x, (mathkong_f32c_t *)x, 0, dummy_results);
    mathkong_dot_f64c((mathkong_f64c_t *)x, (mathkong_f64c_t *)x, 0, dummy_results);
    mathkong_vdot_f16c((mathkong_f16c_t *)x, (mathkong_f16c_t *)x, 0, dummy_results);
    mathkong_vdot_bf16c((mathkong_bf16c_t *)x, (mathkong_bf16c_t *)x, 0, dummy_results);
    mathkong_vdot_f32c((mathkong_f32c_t *)x, (mathkong_f32c_t *)x, 0, dummy_results);
    mathkong_vdot_f64c((mathkong_f64c_t *)x, (mathkong_f64c_t *)x, 0, dummy_results);

    mathkong_angular_i8((mathkong_i8_t *)x, (mathkong_i8_t *)x, 0, dummy_results);
    mathkong_angular_u8((mathkong_u8_t *)x, (mathkong_u8_t *)x, 0, dummy_results);
    mathkong_angular_f16((mathkong_f16_t *)x, (mathkong_f16_t *)x, 0, dummy_results);
    mathkong_angular_bf16((mathkong_bf16_t *)x, (mathkong_bf16_t *)x, 0, dummy_results);
    mathkong_angular_f32((mathkong_f32_t *)x, (mathkong_f32_t *)x, 0, dummy_results);
    mathkong_angular_f64((mathkong_f64_t *)x, (mathkong_f64_t *)x, 0, dummy_results);

    mathkong_l2sq_i8((mathkong_i8_t *)x, (mathkong_i8_t *)x, 0, dummy_results);
    mathkong_l2sq_u8((mathkong_u8_t *)x, (mathkong_u8_t *)x, 0, dummy_results);
    mathkong_l2sq_f16((mathkong_f16_t *)x, (mathkong_f16_t *)x, 0, dummy_results);
    mathkong_l2sq_bf16((mathkong_bf16_t *)x, (mathkong_bf16_t *)x, 0, dummy_results);
    mathkong_l2sq_f32((mathkong_f32_t *)x, (mathkong_f32_t *)x, 0, dummy_results);
    mathkong_l2sq_f64((mathkong_f64_t *)x, (mathkong_f64_t *)x, 0, dummy_results);

    mathkong_l2_i8((mathkong_i8_t *)x, (mathkong_i8_t *)x, 0, dummy_results);
    mathkong_l2_i8((mathkong_i8_t *)x, (mathkong_i8_t *)x, 0, dummy_results);
    mathkong_l2_u8((mathkong_u8_t *)x, (mathkong_u8_t *)x, 0, dummy_results);
    mathkong_l2_f16((mathkong_f16_t *)x, (mathkong_f16_t *)x, 0, dummy_results);
    mathkong_l2_bf16((mathkong_bf16_t *)x, (mathkong_bf16_t *)x, 0, dummy_results);
    mathkong_l2_f32((mathkong_f32_t *)x, (mathkong_f32_t *)x, 0, dummy_results);
    mathkong_l2_f64((mathkong_f64_t *)x, (mathkong_f64_t *)x, 0, dummy_results);

    mathkong_hamming_b8((mathkong_b8_t *)x, (mathkong_b8_t *)x, 0, dummy_results);
    mathkong_jaccard_b8((mathkong_b8_t *)x, (mathkong_b8_t *)x, 0, dummy_results);

    mathkong_kl_f16((mathkong_f16_t *)x, (mathkong_f16_t *)x, 0, dummy_results);
    mathkong_kl_bf16((mathkong_bf16_t *)x, (mathkong_bf16_t *)x, 0, dummy_results);
    mathkong_kl_f32((mathkong_f32_t *)x, (mathkong_f32_t *)x, 0, dummy_results);
    mathkong_kl_f64((mathkong_f64_t *)x, (mathkong_f64_t *)x, 0, dummy_results);
    mathkong_js_f16((mathkong_f16_t *)x, (mathkong_f16_t *)x, 0, dummy_results);
    mathkong_js_bf16((mathkong_bf16_t *)x, (mathkong_bf16_t *)x, 0, dummy_results);
    mathkong_js_f32((mathkong_f32_t *)x, (mathkong_f32_t *)x, 0, dummy_results);
    mathkong_js_f64((mathkong_f64_t *)x, (mathkong_f64_t *)x, 0, dummy_results);

    // Sparse
    mathkong_intersect_u16((mathkong_u16_t *)x, (mathkong_u16_t *)x, 0, 0, dummy_results);
    mathkong_intersect_u32((mathkong_u32_t *)x, (mathkong_u32_t *)x, 0, 0, dummy_results);

    // Curved:
    mathkong_bilinear_f64((mathkong_f64_t *)x, (mathkong_f64_t *)x, (mathkong_f64_t *)x, 0, dummy_results);
    mathkong_mahalanobis_f64((mathkong_f64_t *)x, (mathkong_f64_t *)x, (mathkong_f64_t *)x, 0, dummy_results);
    mathkong_bilinear_f32((mathkong_f32_t *)x, (mathkong_f32_t *)x, (mathkong_f32_t *)x, 0, dummy_results);
    mathkong_mahalanobis_f32((mathkong_f32_t *)x, (mathkong_f32_t *)x, (mathkong_f32_t *)x, 0, dummy_results);
    mathkong_bilinear_f16((mathkong_f16_t *)x, (mathkong_f16_t *)x, (mathkong_f16_t *)x, 0, dummy_results);
    mathkong_mahalanobis_f16((mathkong_f16_t *)x, (mathkong_f16_t *)x, (mathkong_f16_t *)x, 0, dummy_results);
    mathkong_bilinear_bf16((mathkong_bf16_t *)x, (mathkong_bf16_t *)x, (mathkong_bf16_t *)x, 0, dummy_results);
    mathkong_mahalanobis_bf16((mathkong_bf16_t *)x, (mathkong_bf16_t *)x, (mathkong_bf16_t *)x, 0, dummy_results);

    // Elementwise
    mathkong_sum_f64((mathkong_f64_t *)x, (mathkong_f64_t *)x, 0, (mathkong_f64_t *)x);
    mathkong_sum_f32((mathkong_f32_t *)x, (mathkong_f32_t *)x, 0, (mathkong_f32_t *)x);
    mathkong_sum_f16((mathkong_f16_t *)x, (mathkong_f16_t *)x, 0, (mathkong_f16_t *)x);
    mathkong_sum_bf16((mathkong_bf16_t *)x, (mathkong_bf16_t *)x, 0, (mathkong_bf16_t *)x);
    mathkong_sum_i8((mathkong_i8_t *)x, (mathkong_i8_t *)x, 0, (mathkong_i8_t *)x);
    mathkong_sum_u8((mathkong_u8_t *)x, (mathkong_u8_t *)x, 0, (mathkong_u8_t *)x);
    mathkong_sum_i16((mathkong_i16_t *)x, (mathkong_i16_t *)x, 0, (mathkong_i16_t *)x);
    mathkong_sum_u16((mathkong_u16_t *)x, (mathkong_u16_t *)x, 0, (mathkong_u16_t *)x);
    mathkong_sum_i32((mathkong_i32_t *)x, (mathkong_i32_t *)x, 0, (mathkong_i32_t *)x);
    mathkong_sum_u32((mathkong_u32_t *)x, (mathkong_u32_t *)x, 0, (mathkong_u32_t *)x);
    mathkong_sum_i64((mathkong_i64_t *)x, (mathkong_i64_t *)x, 0, (mathkong_i64_t *)x);
    mathkong_sum_u64((mathkong_u64_t *)x, (mathkong_u64_t *)x, 0, (mathkong_u64_t *)x);
    mathkong_scale_f64((mathkong_f64_t *)x, 0, 0, 0, (mathkong_f64_t *)x);
    mathkong_scale_f32((mathkong_f32_t *)x, 0, 0, 0, (mathkong_f32_t *)x);
    mathkong_scale_f16((mathkong_f16_t *)x, 0, 0, 0, (mathkong_f16_t *)x);
    mathkong_scale_bf16((mathkong_bf16_t *)x, 0, 0, 0, (mathkong_bf16_t *)x);
    mathkong_scale_i8((mathkong_i8_t *)x, 0, 0, 0, (mathkong_i8_t *)x);
    mathkong_scale_u8((mathkong_u8_t *)x, 0, 0, 0, (mathkong_u8_t *)x);
    mathkong_scale_i16((mathkong_i16_t *)x, 0, 0, 0, (mathkong_i16_t *)x);
    mathkong_scale_u16((mathkong_u16_t *)x, 0, 0, 0, (mathkong_u16_t *)x);
    mathkong_scale_i32((mathkong_i32_t *)x, 0, 0, 0, (mathkong_i32_t *)x);
    mathkong_scale_u32((mathkong_u32_t *)x, 0, 0, 0, (mathkong_u32_t *)x);
    mathkong_scale_i64((mathkong_i64_t *)x, 0, 0, 0, (mathkong_i64_t *)x);
    mathkong_scale_u64((mathkong_u64_t *)x, 0, 0, 0, (mathkong_u64_t *)x);
    mathkong_wsum_f64((mathkong_f64_t *)x, (mathkong_f64_t *)x, 0, 0, 0, (mathkong_f64_t *)x);
    mathkong_wsum_f32((mathkong_f32_t *)x, (mathkong_f32_t *)x, 0, 0, 0, (mathkong_f32_t *)x);
    mathkong_wsum_f16((mathkong_f16_t *)x, (mathkong_f16_t *)x, 0, 0, 0, (mathkong_f16_t *)x);
    mathkong_wsum_bf16((mathkong_bf16_t *)x, (mathkong_bf16_t *)x, 0, 0, 0, (mathkong_bf16_t *)x);
    mathkong_wsum_i8((mathkong_i8_t *)x, (mathkong_i8_t *)x, 0, 0, 0, (mathkong_i8_t *)x);
    mathkong_wsum_u8((mathkong_u8_t *)x, (mathkong_u8_t *)x, 0, 0, 0, (mathkong_u8_t *)x);
    mathkong_wsum_i16((mathkong_i16_t *)x, (mathkong_i16_t *)x, 0, 0, 0, (mathkong_i16_t *)x);
    mathkong_wsum_u16((mathkong_u16_t *)x, (mathkong_u16_t *)x, 0, 0, 0, (mathkong_u16_t *)x);
    mathkong_wsum_i32((mathkong_i32_t *)x, (mathkong_i32_t *)x, 0, 0, 0, (mathkong_i32_t *)x);
    mathkong_wsum_u32((mathkong_u32_t *)x, (mathkong_u32_t *)x, 0, 0, 0, (mathkong_u32_t *)x);
    mathkong_wsum_i64((mathkong_i64_t *)x, (mathkong_i64_t *)x, 0, 0, 0, (mathkong_i64_t *)x);
    mathkong_wsum_u64((mathkong_u64_t *)x, (mathkong_u64_t *)x, 0, 0, 0, (mathkong_u64_t *)x);
    mathkong_fma_f64((mathkong_f64_t *)x, (mathkong_f64_t *)x, (mathkong_f64_t *)x, 0, 0, 0, (mathkong_f64_t *)x);
    mathkong_fma_f32((mathkong_f32_t *)x, (mathkong_f32_t *)x, (mathkong_f32_t *)x, 0, 0, 0, (mathkong_f32_t *)x);
    mathkong_fma_f16((mathkong_f16_t *)x, (mathkong_f16_t *)x, (mathkong_f16_t *)x, 0, 0, 0, (mathkong_f16_t *)x);
    mathkong_fma_bf16((mathkong_bf16_t *)x, (mathkong_bf16_t *)x, (mathkong_bf16_t *)x, 0, 0, 0, (mathkong_bf16_t *)x);
    mathkong_fma_i8((mathkong_i8_t *)x, (mathkong_i8_t *)x, (mathkong_i8_t *)x, 0, 0, 0, (mathkong_i8_t *)x);
    mathkong_fma_u8((mathkong_u8_t *)x, (mathkong_u8_t *)x, (mathkong_u8_t *)x, 0, 0, 0, (mathkong_u8_t *)x);
    mathkong_fma_i16((mathkong_i16_t *)x, (mathkong_i16_t *)x, (mathkong_i16_t *)x, 0, 0, 0, (mathkong_i16_t *)x);
    mathkong_fma_u16((mathkong_u16_t *)x, (mathkong_u16_t *)x, (mathkong_u16_t *)x, 0, 0, 0, (mathkong_u16_t *)x);
    mathkong_fma_i32((mathkong_i32_t *)x, (mathkong_i32_t *)x, (mathkong_i32_t *)x, 0, 0, 0, (mathkong_i32_t *)x);
    mathkong_fma_u32((mathkong_u32_t *)x, (mathkong_u32_t *)x, (mathkong_u32_t *)x, 0, 0, 0, (mathkong_u32_t *)x);
    mathkong_fma_i64((mathkong_i64_t *)x, (mathkong_i64_t *)x, (mathkong_i64_t *)x, 0, 0, 0, (mathkong_i64_t *)x);
    mathkong_fma_u64((mathkong_u64_t *)x, (mathkong_u64_t *)x, (mathkong_u64_t *)x, 0, 0, 0, (mathkong_u64_t *)x);

    return static_capabilities;
}

SIMSIMD_DYNAMIC void mathkong_find_kernel(   //
    mathkong_kernel_kind_t kind,             //
    mathkong_datatype_t datatype,            //
    mathkong_capability_t supported,         //
    mathkong_capability_t allowed,           //
    mathkong_kernel_punned_t *kernel_output, //
    mathkong_capability_t *capability_output) {
    _mathkong_find_kernel_implementation(kind, datatype, supported, allowed, kernel_output, capability_output);
}

#ifdef __cplusplus
}
#endif

/**
 *  @brief  Dynamic dispatch library for SimSIMD.
 *  @note   Compile with the most recent compiler available.
 *  @file   lib.c
 */
#define SIMSIMD_DYNAMIC_DISPATCH 1
#define SIMSIMD_NATIVE_F16 0

/*  Depending on the Operating System, the following intrinsics are available
 *  on recent compiler toolchains:
 *
 *  - Linux: everything is available in GCC 12+ and Clang 16+.
 *  - Windows - MSVC: everything except Sapphire Rapids and ARM SVE.
 *  - MacOS - Apple Clang: only Arm NEON and x86 AVX2 Haswell extensions are available.
 */
#if !defined(SIMSIMD_TARGET_NEON) && (defined(__APPLE__) || defined(__linux__))
#define SIMSIMD_TARGET_NEON 1
#endif
#if !defined(SIMSIMD_TARGET_SVE) && (defined(__linux__))
#define SIMSIMD_TARGET_SVE 1
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
#if !defined(SIMSIMD_TARGET_SAPPHIRE) && (defined(__linux__))
#define SIMSIMD_TARGET_SAPPHIRE 1
#endif

#include <simsimd/simsimd.h>

#ifdef __cplusplus
extern "C" {
#endif

simsimd_capability_t simsimd_capabilities(void) {
    static simsimd_capability_t static_capabilities = simsimd_cap_any_k;
    if (static_capabilities == simsimd_cap_any_k)
        static_capabilities = simsimd_capabilities_implementation();
    return static_capabilities;
}

// Every time a function is called, it checks if the metric is already loaded. If not, it fetches it.
// If no metric is found, it returns NaN. We can obtain NaN by dividing 0.0 by 0.0, but that annoys
// the MSVC compiler. Instead we can directly write-in the signaling NaN (0x7FF0000000000001)
// or the qNaN (0x7FF8000000000000).
#define SIMSIMD_METRIC_DECLARATION(name, extension, type)                                                              \
    SIMSIMD_DYNAMIC void simsimd_##name##_##extension(simsimd_##type##_t const* a, simsimd_##type##_t const* b,        \
                                                      simsimd_size_t n, simsimd_distance_t* results) {                 \
        static simsimd_metric_punned_t metric = 0;                                                                     \
        if (metric == 0) {                                                                                             \
            simsimd_capability_t used_capability;                                                                      \
            simsimd_find_metric_punned(simsimd_metric_##name##_k, simsimd_datatype_##extension##_k,                    \
                                       simsimd_capabilities(), simsimd_cap_any_k, &metric, &used_capability);          \
            if (!metric) {                                                                                             \
                *(simsimd_u64_t*)results = 0x7FF0000000000001ull;                                                      \
                return;                                                                                                \
            }                                                                                                          \
        }                                                                                                              \
        metric(a, b, n, results);                                                                                      \
    }

// Dot products
SIMSIMD_METRIC_DECLARATION(dot, f16, f16)
SIMSIMD_METRIC_DECLARATION(dot, f32, f32)
SIMSIMD_METRIC_DECLARATION(dot, f64, f64)
SIMSIMD_METRIC_DECLARATION(dot, f16c, f16)
SIMSIMD_METRIC_DECLARATION(dot, f32c, f32)
SIMSIMD_METRIC_DECLARATION(dot, f64c, f64)
SIMSIMD_METRIC_DECLARATION(vdot, f16c, f16)
SIMSIMD_METRIC_DECLARATION(vdot, f32c, f32)
SIMSIMD_METRIC_DECLARATION(vdot, f64c, f64)

// Spatial distances
SIMSIMD_METRIC_DECLARATION(cos, i8, i8)
SIMSIMD_METRIC_DECLARATION(cos, f16, f16)
SIMSIMD_METRIC_DECLARATION(cos, f32, f32)
SIMSIMD_METRIC_DECLARATION(cos, f64, f64)
SIMSIMD_METRIC_DECLARATION(l2sq, i8, i8)
SIMSIMD_METRIC_DECLARATION(l2sq, f16, f16)
SIMSIMD_METRIC_DECLARATION(l2sq, f32, f32)
SIMSIMD_METRIC_DECLARATION(l2sq, f64, f64)

// Binary distances
SIMSIMD_METRIC_DECLARATION(hamming, b8, b8)
SIMSIMD_METRIC_DECLARATION(jaccard, b8, b8)

// Probability distributions
SIMSIMD_METRIC_DECLARATION(kl, f16, f16)
SIMSIMD_METRIC_DECLARATION(kl, f32, f32)
SIMSIMD_METRIC_DECLARATION(kl, f64, f64)
SIMSIMD_METRIC_DECLARATION(js, f16, f16)
SIMSIMD_METRIC_DECLARATION(js, f32, f32)
SIMSIMD_METRIC_DECLARATION(js, f64, f64)

SIMSIMD_DYNAMIC int simsimd_uses_neon(void) { return (simsimd_capabilities() & simsimd_cap_neon_k) != 0; }
SIMSIMD_DYNAMIC int simsimd_uses_sve(void) { return (simsimd_capabilities() & simsimd_cap_sve_k) != 0; }
SIMSIMD_DYNAMIC int simsimd_uses_haswell(void) { return (simsimd_capabilities() & simsimd_cap_haswell_k) != 0; }
SIMSIMD_DYNAMIC int simsimd_uses_skylake(void) { return (simsimd_capabilities() & simsimd_cap_skylake_k) != 0; }
SIMSIMD_DYNAMIC int simsimd_uses_ice(void) { return (simsimd_capabilities() & simsimd_cap_ice_k) != 0; }
SIMSIMD_DYNAMIC int simsimd_uses_sapphire(void) { return (simsimd_capabilities() & simsimd_cap_sapphire_k) != 0; }

#ifdef __cplusplus
}
#endif

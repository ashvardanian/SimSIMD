#include <simsimd/simsimd.h>

simsimd_capability_t cached_capabilities(void) {
    static simsimd_capability_t static_capabilities = 0;
    if (static_capabilities == 0)
        static_capabilities = simsimd_capabilities();
    return static_capabilities;
}

#define SIMSIMD_METRIC_DECLARATION(name, extension, type)                                                              \
    void simsimd_##name##_##extension(simsimd_##type##_t const* a, simsimd_##type##_t const* b,                        \
                                      simsimd_size_t n_words, simsimd_distance_t* results) {                           \
        static simsimd_metric_punned_t metric = 0;                                                                     \
        if (metric == 0) {                                                                                             \
            simsimd_capability_t used_capability;                                                                      \
            simsimd_find_metric_punned(simsimd_metric_##name##_k, simsimd_datatype_##type##_k, cached_capabilities(),  \
                                       simsimd_cap_any_k, &metric, &used_capability);                                  \
        }                                                                                                              \
        metric(a, b, n_words, results);                                                                                \
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

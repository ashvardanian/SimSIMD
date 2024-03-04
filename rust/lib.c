#include <simsimd/simsimd.h>

simsimd_capability_t cached_capabilities(void) {
    static simsimd_capability_t static_capabilities = 0;
    if (static_capabilities == 0)
        static_capabilities = simsimd_capabilities();
    return static_capabilities;
}

#define SIMSIMD_METRIC_DECLARATION(name, type)                                                                         \
    simsimd_f32_t name##_##type(simsimd_##type##_t const* a, simsimd_##type##_t const* b, simsimd_size_t d) {          \
        static simsimd_metric_punned_t metric = 0;                                                                     \
        if (metric == 0) {                                                                                             \
            simsimd_capability_t used_capability;                                                                      \
            simsimd_find_metric_punned(simsimd_metric_##name##_k, simsimd_datatype_##type##_k, cached_capabilities(),  \
                                       simsimd_cap_any_k, &metric, &used_capability);                                  \
        }                                                                                                              \
        return metric(a, b, d, d);                                                                                     \
    }

// Spatial distances
SIMSIMD_METRIC_DECLARATION(cosine, i8)
SIMSIMD_METRIC_DECLARATION(cosine, f16)
SIMSIMD_METRIC_DECLARATION(cosine, f32)
SIMSIMD_METRIC_DECLARATION(cosine, f64)
SIMSIMD_METRIC_DECLARATION(inner, i8)
SIMSIMD_METRIC_DECLARATION(inner, f16)
SIMSIMD_METRIC_DECLARATION(inner, f32)
SIMSIMD_METRIC_DECLARATION(inner, f64)
SIMSIMD_METRIC_DECLARATION(sqeuclidean, i8)
SIMSIMD_METRIC_DECLARATION(sqeuclidean, f16)
SIMSIMD_METRIC_DECLARATION(sqeuclidean, f32)
SIMSIMD_METRIC_DECLARATION(sqeuclidean, f64)

// Binary distances
SIMSIMD_METRIC_DECLARATION(hamming, b8)
SIMSIMD_METRIC_DECLARATION(jaccard, b8)

// Probability distributions
SIMSIMD_METRIC_DECLARATION(kl, f16)
SIMSIMD_METRIC_DECLARATION(kl, f32)
SIMSIMD_METRIC_DECLARATION(kl, f64)
SIMSIMD_METRIC_DECLARATION(js, f16)
SIMSIMD_METRIC_DECLARATION(js, f32)
SIMSIMD_METRIC_DECLARATION(js, f64)

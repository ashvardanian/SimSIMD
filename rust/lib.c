#include <simsimd/simsimd.h>

simsimd_capability_t cached_capabilities(void) {
    static simsimd_capability_t static_capabilities = 0;
    if (static_capabilities == 0)
        static_capabilities = simsimd_capabilities();
    return static_capabilities;
}

simsimd_f32_t cosine_i8(simsimd_i8_t const* a, simsimd_i8_t const* b, simsimd_size_t d) {
    static simsimd_metric_punned_t metric = 0;
    if (metric == 0) {
        simsimd_capability_t used_capability;
        simsimd_find_metric_punned(simsimd_metric_cosine_k, simsimd_datatype_i8_k, cached_capabilities(),
                                   simsimd_cap_any_k, &metric, &used_capability);
    }
    return metric(a, b, d, d);
}

simsimd_f32_t cosine_f32(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t d) {
    static simsimd_metric_punned_t metric = 0;
    if (metric == 0) {
        simsimd_capability_t used_capability;
        simsimd_find_metric_punned(simsimd_metric_cosine_k, simsimd_datatype_f32_k, cached_capabilities(),
                                   simsimd_cap_any_k, &metric, &used_capability);
    }
    return metric(a, b, d, d);
}

simsimd_f32_t inner_i8(simsimd_i8_t const* a, simsimd_i8_t const* b, simsimd_size_t d) {
    static simsimd_metric_punned_t metric = 0;
    if (metric == 0) {
        simsimd_capability_t used_capability;
        simsimd_find_metric_punned(simsimd_metric_inner_k, simsimd_datatype_i8_k, cached_capabilities(),
                                   simsimd_cap_any_k, &metric, &used_capability);
    }
    return metric(a, b, d, d);
}

simsimd_f32_t inner_f32(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t d) {
    static simsimd_metric_punned_t metric = 0;
    if (metric == 0) {
        simsimd_capability_t used_capability;
        simsimd_find_metric_punned(simsimd_metric_inner_k, simsimd_datatype_f32_k, cached_capabilities(),
                                   simsimd_cap_any_k, &metric, &used_capability);
    }
    return metric(a, b, d, d);
}

simsimd_f32_t sqeuclidean_i8(simsimd_i8_t const* a, simsimd_i8_t const* b, simsimd_size_t d) {
    static simsimd_metric_punned_t metric = 0;
    if (metric == 0) {
        simsimd_capability_t used_capability;
        simsimd_find_metric_punned(simsimd_metric_sqeuclidean_k, simsimd_datatype_i8_k, cached_capabilities(),
                                   simsimd_cap_any_k, &metric, &used_capability);
    }
    return metric(a, b, d, d);
}

simsimd_f32_t sqeuclidean_f32(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t d) {
    static simsimd_metric_punned_t metric = 0;
    if (metric == 0) {
        simsimd_capability_t used_capability;
        simsimd_find_metric_punned(simsimd_metric_sqeuclidean_k, simsimd_datatype_f32_k, cached_capabilities(),
                                   simsimd_cap_any_k, &metric, &used_capability);
    }
    return metric(a, b, d, d);
}

#include <stdlib.h>
#define SIMSIMD_NATIVE_F16 (0)
#include "../include/simsimd/simsimd.h"

inline static simsimd_f32_t 
cosine_i8(simsimd_i8_t const* a, simsimd_i8_t const* b, simsimd_size_t d) {
    static simsimd_metric_punned_t fn;
    if (!fn) {
        fn = simsimd_metric_punned(simsimd_metric_cosine_k, simsimd_datatype_i8_k, simsimd_cap_any_k);
    }
    return fn(a, b, d, d); 
}

inline static simsimd_f32_t 
cosine_f32(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t d) { 
    static simsimd_metric_punned_t fn;
    if (!fn) {
        fn = simsimd_metric_punned(simsimd_metric_cosine_k, simsimd_datatype_f32_k, simsimd_cap_any_k);
    }
    return fn(a, b, d, d); 
}

inline static simsimd_f32_t 
inner_i8(simsimd_i8_t const* a, simsimd_i8_t const* b, simsimd_size_t d) { 
    static simsimd_metric_punned_t fn;
    if (!fn) {
        fn = simsimd_metric_punned(simsimd_metric_inner_k, simsimd_datatype_i8_k, simsimd_cap_any_k);
    }
    return fn(a, b, d, d); 
}

inline static simsimd_f32_t 
inner_f32(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t d) { 
    static simsimd_metric_punned_t fn;
    if (!fn) {
        fn = simsimd_metric_punned(simsimd_metric_inner_k, simsimd_datatype_f32_k, simsimd_cap_any_k);
    }
    return fn(a, b, d, d); 
}

inline static simsimd_f32_t 
sqeuclidean_i8(simsimd_i8_t const* a, simsimd_i8_t const* b, simsimd_size_t d) { 
    static simsimd_metric_punned_t fn;
    if (!fn) {
        fn = simsimd_metric_punned(simsimd_metric_sqeuclidean_k, simsimd_datatype_i8_k, simsimd_cap_any_k);
    }
    return fn(a, b, d, d); 
}

inline static simsimd_f32_t 
sqeuclidean_f32(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t d) { 
    static simsimd_metric_punned_t fn;
    if (!fn) {
        fn = simsimd_metric_punned(simsimd_metric_sqeuclidean_k, simsimd_datatype_f32_k, simsimd_cap_any_k);
    }
    return fn(a, b, d, d); 
}

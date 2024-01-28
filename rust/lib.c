#include <simsimd/simsimd.h>

simsimd_f32_t cosine_i8(simsimd_i8_t const* a, simsimd_i8_t const* b, simsimd_size_t d) {
    return simsimd_metric_punned(simsimd_metric_cosine_k, simsimd_datatype_i8_k, simsimd_cap_any_k)(a, b, d, d);
}
simsimd_f32_t cosine_f32(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t d) {
    return simsimd_metric_punned(simsimd_metric_cosine_k, simsimd_datatype_f32_k, simsimd_cap_any_k)(a, b, d, d);
}
simsimd_f32_t inner_i8(simsimd_i8_t const* a, simsimd_i8_t const* b, simsimd_size_t d) {
    return simsimd_metric_punned(simsimd_metric_inner_k, simsimd_datatype_i8_k, simsimd_cap_any_k)(a, b, d, d);
}
simsimd_f32_t inner_f32(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t d) {
    return simsimd_metric_punned(simsimd_metric_inner_k, simsimd_datatype_f32_k, simsimd_cap_any_k)(a, b, d, d);
}
simsimd_f32_t sqeuclidean_i8(simsimd_i8_t const* a, simsimd_i8_t const* b, simsimd_size_t d) {
    return simsimd_metric_punned(simsimd_metric_sqeuclidean_k, simsimd_datatype_i8_k, simsimd_cap_any_k)(a, b, d, d);
}
simsimd_f32_t sqeuclidean_f32(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t d) {
    return simsimd_metric_punned(simsimd_metric_sqeuclidean_k, simsimd_datatype_f32_k, simsimd_cap_any_k)(a, b, d, d);
}

/**
 *  @brief Dispatch Initialization for E2M3 Data Types.
 *  @file c/dispatch_e2m3.c
 *  @author Ash Vardanian
 *  @date February 3, 2026
 */
#include "dispatch.h"

#ifdef __cplusplus
extern "C" {
#endif

void nk_dispatch_e2m3_find_(nk_capability_t v, nk_kernel_kind_t k, nk_kernel_punned_t *m, nk_capability_t *c) {
    typedef nk_kernel_punned_t m_t;
#if NK_TARGET_SAPPHIRE
    if (v & nk_cap_sapphire_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_e2m3_sapphire, *c = nk_cap_sapphire_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_e2m3_sapphire, *c = nk_cap_sapphire_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_e2m3_sapphire, *c = nk_cap_sapphire_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_e2m3_sapphire, *c = nk_cap_sapphire_k; return;
        default: break;
        }
#endif
#if NK_TARGET_GENOA
    if (v & nk_cap_genoa_k) switch (k) {
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_e2m3_genoa, *c = nk_cap_genoa_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_e2m3_genoa, *c = nk_cap_genoa_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_e2m3_genoa, *c = nk_cap_genoa_k; return;
        case nk_kernel_dots_packed_size_k: *m = (m_t)&nk_dots_packed_size_e2m3_genoa, *c = nk_cap_genoa_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_e2m3_genoa, *c = nk_cap_genoa_k; return;
        case nk_kernel_dots_packed_k: *m = (m_t)&nk_dots_packed_e2m3_genoa, *c = nk_cap_genoa_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_e2m3_genoa, *c = nk_cap_genoa_k; return;
        default: break;
        }
#endif
#if NK_TARGET_RVVHALF
    if (v & nk_cap_rvvhalf_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_e2m3_rvvhalf, *c = nk_cap_rvvhalf_k; return;
        default: break;
        }
#endif
#if NK_TARGET_RVVBF16
    if (v & nk_cap_rvvbf16_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_e2m3_rvvbf16, *c = nk_cap_rvvbf16_k; return;
        default: break;
        }
#endif
#if NK_TARGET_RVV
    if (v & nk_cap_rvv_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_e2m3_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_reduce_add_k: *m = (m_t)&nk_reduce_add_e2m3_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_reduce_min_k: *m = (m_t)&nk_reduce_min_e2m3_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_reduce_max_k: *m = (m_t)&nk_reduce_max_e2m3_rvv, *c = nk_cap_rvv_k; return;
        default: break;
        }
#endif
#if NK_TARGET_NEONFHM
    if (v & nk_cap_neonfhm_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_e2m3_neonfhm, *c = nk_cap_neonfhm_k; return;
        case nk_kernel_dots_packed_size_k: *m = (m_t)&nk_dots_packed_size_e2m3_neonfhm, *c = nk_cap_neonfhm_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_e2m3_neonfhm, *c = nk_cap_neonfhm_k; return;
        case nk_kernel_dots_packed_k: *m = (m_t)&nk_dots_packed_e2m3_neonfhm, *c = nk_cap_neonfhm_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_e2m3_neonfhm, *c = nk_cap_neonfhm_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SKYLAKE
    if (v & nk_cap_skylake_k) switch (k) {
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_e2m3_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_e2m3_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_e2m3_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_dots_packed_size_k: *m = (m_t)&nk_dots_packed_size_e2m3_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_e2m3_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_dots_packed_k: *m = (m_t)&nk_dots_packed_e2m3_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_e2m3_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_reduce_min_k: *m = (m_t)&nk_reduce_min_e2m3_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_reduce_max_k: *m = (m_t)&nk_reduce_max_e2m3_skylake, *c = nk_cap_skylake_k; return;
        default: break;
        }
#endif
#if NK_TARGET_HASWELL
    if (v & nk_cap_haswell_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_e2m3_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_e2m3_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_e2m3_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_e2m3_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_dots_packed_size_k: *m = (m_t)&nk_dots_packed_size_e2m3_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_e2m3_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_dots_packed_k: *m = (m_t)&nk_dots_packed_e2m3_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_e2m3_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_reduce_add_k: *m = (m_t)&nk_reduce_add_e2m3_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_reduce_min_k: *m = (m_t)&nk_reduce_min_e2m3_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_reduce_max_k: *m = (m_t)&nk_reduce_max_e2m3_haswell, *c = nk_cap_haswell_k; return;
        default: break;
        }
#endif
#if NK_TARGET_NEON
    if (v & nk_cap_neon_k) switch (k) {
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_e2m3_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_e2m3_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_e2m3_neon, *c = nk_cap_neon_k; return;
        default: break;
        }
#endif
    if (v & nk_cap_serial_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_e2m3_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_e2m3_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_e2m3_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_e2m3_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_each_sum_k: *m = (m_t)&nk_each_sum_e2m3_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_each_scale_k: *m = (m_t)&nk_each_scale_e2m3_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_each_blend_k: *m = (m_t)&nk_each_blend_e2m3_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_each_fma_k: *m = (m_t)&nk_each_fma_e2m3_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_dots_packed_size_k: *m = (m_t)&nk_dots_packed_size_e2m3_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_e2m3_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_dots_packed_k: *m = (m_t)&nk_dots_packed_e2m3_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_e2m3_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_reduce_add_k: *m = (m_t)&nk_reduce_add_e2m3_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_reduce_min_k: *m = (m_t)&nk_reduce_min_e2m3_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_reduce_max_k: *m = (m_t)&nk_reduce_max_e2m3_serial, *c = nk_cap_serial_k; return;
        default: break;
        }

    // Error fallback - zero capability signals lookup failure
    *m = (m_t)nk_error_dense_, *c = 0;
}

void nk_dispatch_e2m3_init_(nk_capability_t caps) {
    nk_implementations_t *t = &nk_dispatch_table;
    nk_capability_t used;

    nk_dispatch_e2m3_find_(caps, nk_kernel_dot_k, (nk_kernel_punned_t *)&t->dot_e2m3, &used);
    nk_dispatch_e2m3_find_(caps, nk_kernel_angular_k, (nk_kernel_punned_t *)&t->angular_e2m3, &used);
    nk_dispatch_e2m3_find_(caps, nk_kernel_sqeuclidean_k, (nk_kernel_punned_t *)&t->sqeuclidean_e2m3, &used);
    nk_dispatch_e2m3_find_(caps, nk_kernel_euclidean_k, (nk_kernel_punned_t *)&t->euclidean_e2m3, &used);
    nk_dispatch_e2m3_find_(caps, nk_kernel_each_fma_k, (nk_kernel_punned_t *)&t->each_fma_e2m3, &used);
    nk_dispatch_e2m3_find_(caps, nk_kernel_each_blend_k, (nk_kernel_punned_t *)&t->each_blend_e2m3, &used);
    nk_dispatch_e2m3_find_(caps, nk_kernel_each_scale_k, (nk_kernel_punned_t *)&t->each_scale_e2m3, &used);
    nk_dispatch_e2m3_find_(caps, nk_kernel_each_sum_k, (nk_kernel_punned_t *)&t->each_sum_e2m3, &used);
    nk_dispatch_e2m3_find_(caps, nk_kernel_dots_packed_size_k, (nk_kernel_punned_t *)&t->dots_packed_size_e2m3, &used);
    nk_dispatch_e2m3_find_(caps, nk_kernel_dots_pack_k, (nk_kernel_punned_t *)&t->dots_pack_e2m3, &used);
    nk_dispatch_e2m3_find_(caps, nk_kernel_dots_packed_k, (nk_kernel_punned_t *)&t->dots_packed_e2m3, &used);
    nk_dispatch_e2m3_find_(caps, nk_kernel_dots_symmetric_k, (nk_kernel_punned_t *)&t->dots_symmetric_e2m3, &used);
    nk_dispatch_e2m3_find_(caps, nk_kernel_reduce_add_k, (nk_kernel_punned_t *)&t->reduce_add_e2m3, &used);
    nk_dispatch_e2m3_find_(caps, nk_kernel_reduce_min_k, (nk_kernel_punned_t *)&t->reduce_min_e2m3, &used);
    nk_dispatch_e2m3_find_(caps, nk_kernel_reduce_max_k, (nk_kernel_punned_t *)&t->reduce_max_e2m3, &used);
}

#ifdef __cplusplus
}
#endif

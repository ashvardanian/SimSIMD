/**
 *  @brief Dispatch functions for e3m2 (FP6) data type.
 *  @file c/dispatch_e3m2.c
 */
#include "dispatch.h"

#ifdef __cplusplus
extern "C" {
#endif

void nk_dispatch_e3m2_find_(nk_capability_t v, nk_kernel_kind_t k, nk_kernel_punned_t *m, nk_capability_t *c) {
    typedef nk_kernel_punned_t m_t;
#if NK_TARGET_GENOA
    if (v & nk_cap_genoa_k) switch (k) {
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_e3m2_genoa, *c = nk_cap_genoa_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_e3m2_genoa, *c = nk_cap_genoa_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_e3m2_genoa, *c = nk_cap_genoa_k; return;
        case nk_kernel_dots_packed_size_k: *m = (m_t)&nk_dots_packed_size_e3m2_genoa, *c = nk_cap_genoa_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_e3m2_genoa, *c = nk_cap_genoa_k; return;
        case nk_kernel_dots_k: *m = (m_t)&nk_dots_packed_e3m2_genoa, *c = nk_cap_genoa_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_e3m2_genoa, *c = nk_cap_genoa_k; return;
        default: break;
        }
#endif
#if NK_TARGET_NEONFHM
    if (v & nk_cap_neonfhm_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_e3m2_neonfhm, *c = nk_cap_neonfhm_k; return;
        case nk_kernel_dots_packed_size_k: *m = (m_t)&nk_dots_packed_size_e3m2_neonfhm, *c = nk_cap_neonfhm_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_e3m2_neonfhm, *c = nk_cap_neonfhm_k; return;
        case nk_kernel_dots_k: *m = (m_t)&nk_dots_packed_e3m2_neonfhm, *c = nk_cap_neonfhm_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_e3m2_neonfhm, *c = nk_cap_neonfhm_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SKYLAKE
    if (v & nk_cap_skylake_k) switch (k) {
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_e3m2_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_e3m2_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_e3m2_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_dots_packed_size_k: *m = (m_t)&nk_dots_packed_size_e3m2_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_e3m2_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_dots_k: *m = (m_t)&nk_dots_packed_e3m2_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_e3m2_skylake, *c = nk_cap_skylake_k; return;
        default: break;
        }
#endif
#if NK_TARGET_HASWELL
    if (v & nk_cap_haswell_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_e3m2_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_e3m2_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_e3m2_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_e3m2_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_dots_packed_size_k: *m = (m_t)&nk_dots_packed_size_e3m2_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_e3m2_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_dots_k: *m = (m_t)&nk_dots_packed_e3m2_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_e3m2_haswell, *c = nk_cap_haswell_k; return;
        default: break;
        }
#endif
#if NK_TARGET_NEON
    if (v & nk_cap_neon_k) switch (k) {
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_e3m2_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_e3m2_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_e3m2_neon, *c = nk_cap_neon_k; return;
        default: break;
        }
#endif
    if (v & nk_cap_serial_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_e3m2_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_e3m2_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_e3m2_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_e3m2_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_dots_packed_size_k: *m = (m_t)&nk_dots_packed_size_e3m2_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_e3m2_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_dots_k: *m = (m_t)&nk_dots_packed_e3m2_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_e3m2_serial, *c = nk_cap_serial_k; return;
        default: break;
        }

    // Error fallback - zero capability signals lookup failure
    *m = (m_t)nk_error_dense_, *c = 0;
}

void nk_dispatch_e3m2_init_(nk_capability_t caps) {
    nk_implementations_t *t = &nk_dispatch_table;
    nk_capability_t used;

    nk_dispatch_e3m2_find_(caps, nk_kernel_dot_k, (nk_kernel_punned_t *)&t->dot_e3m2, &used);
    nk_dispatch_e3m2_find_(caps, nk_kernel_angular_k, (nk_kernel_punned_t *)&t->angular_e3m2, &used);
    nk_dispatch_e3m2_find_(caps, nk_kernel_sqeuclidean_k, (nk_kernel_punned_t *)&t->sqeuclidean_e3m2, &used);
    nk_dispatch_e3m2_find_(caps, nk_kernel_euclidean_k, (nk_kernel_punned_t *)&t->euclidean_e3m2, &used);
    nk_dispatch_e3m2_find_(caps, nk_kernel_dots_packed_size_k, (nk_kernel_punned_t *)&t->dots_packed_size_e3m2, &used);
    nk_dispatch_e3m2_find_(caps, nk_kernel_dots_pack_k, (nk_kernel_punned_t *)&t->dots_pack_e3m2, &used);
    nk_dispatch_e3m2_find_(caps, nk_kernel_dots_k, (nk_kernel_punned_t *)&t->dots_packed_e3m2, &used);
    nk_dispatch_e3m2_find_(caps, nk_kernel_dots_symmetric_k, (nk_kernel_punned_t *)&t->dots_symmetric_e3m2, &used);
}

#ifdef __cplusplus
}
#endif

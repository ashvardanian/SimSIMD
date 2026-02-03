/**
 *  @brief Dtype dispatch file for u64 (unsigned 64-bit integer) kernels.
 *  @file c/dispatch_u64.c
 */
#include "dispatch.h"

void nk_dispatch_u64_find_(nk_capability_t v, nk_kernel_kind_t k, nk_kernel_punned_t *m, nk_capability_t *c) {
    typedef nk_kernel_punned_t m_t;
#if NK_TARGET_SVE2
    if (v & nk_cap_sve2_k) switch (k) {
        case nk_kernel_sparse_intersect_k: *m = (m_t)&nk_sparse_intersect_u64_sve2, *c = nk_cap_sve2_k; return;
        default: break;
        }
#endif
#if NK_TARGET_NEON
    if (v & nk_cap_neon_k) switch (k) {
        case nk_kernel_sparse_intersect_k: *m = (m_t)&nk_sparse_intersect_u64_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_each_fma_k: *m = (m_t)&nk_each_fma_u64_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_each_scale_k: *m = (m_t)&nk_each_scale_u64_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_each_sum_k: *m = (m_t)&nk_each_sum_u64_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_reduce_add_k: *m = (m_t)&nk_reduce_add_u64_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_reduce_min_k: *m = (m_t)&nk_reduce_min_u64_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_reduce_max_k: *m = (m_t)&nk_reduce_max_u64_neon, *c = nk_cap_neon_k; return;
        default: break;
        }
#endif
#if NK_TARGET_TURIN
    if (v & nk_cap_turin_k) switch (k) {
        case nk_kernel_sparse_intersect_k: *m = (m_t)&nk_sparse_intersect_u64_turin, *c = nk_cap_turin_k; return;
        default: break;
        }
#endif
#if NK_TARGET_ICELAKE
    if (v & nk_cap_icelake_k) switch (k) {
        case nk_kernel_sparse_intersect_k: *m = (m_t)&nk_sparse_intersect_u64_icelake, *c = nk_cap_icelake_k; return;
        case nk_kernel_each_sum_k: *m = (m_t)&nk_each_sum_u64_icelake, *c = nk_cap_icelake_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SKYLAKE
    if (v & nk_cap_skylake_k) switch (k) {
        case nk_kernel_each_fma_k: *m = (m_t)&nk_each_fma_u64_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_each_scale_k: *m = (m_t)&nk_each_scale_u64_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_reduce_add_k: *m = (m_t)&nk_reduce_add_u64_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_reduce_min_k: *m = (m_t)&nk_reduce_min_u64_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_reduce_max_k: *m = (m_t)&nk_reduce_max_u64_skylake, *c = nk_cap_skylake_k; return;
        default: break;
        }
#endif
#if NK_TARGET_HASWELL
    if (v & nk_cap_haswell_k) switch (k) {
        case nk_kernel_reduce_add_k: *m = (m_t)&nk_reduce_add_u64_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_reduce_min_k: *m = (m_t)&nk_reduce_min_u64_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_reduce_max_k: *m = (m_t)&nk_reduce_max_u64_haswell, *c = nk_cap_haswell_k; return;
        default: break;
        }
#endif
    if (v & nk_cap_serial_k) switch (k) {
        case nk_kernel_sparse_intersect_k: *m = (m_t)&nk_sparse_intersect_u64_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_each_fma_k: *m = (m_t)&nk_each_fma_u64_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_each_scale_k: *m = (m_t)&nk_each_scale_u64_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_each_sum_k: *m = (m_t)&nk_each_sum_u64_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_each_blend_k: *m = (m_t)&nk_each_blend_u64_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_reduce_add_k: *m = (m_t)&nk_reduce_add_u64_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_reduce_min_k: *m = (m_t)&nk_reduce_min_u64_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_reduce_max_k: *m = (m_t)&nk_reduce_max_u64_serial, *c = nk_cap_serial_k; return;
        default: break;
        }

    // Error fallback - zero capability signals lookup failure
    *m = (m_t)nk_error_dense_, *c = 0;
}

void nk_dispatch_u64_init_(nk_capability_t caps) {
    nk_implementations_t *t = &nk_dispatch_table;
    nk_capability_t used;

    nk_dispatch_u64_find_(caps, nk_kernel_each_scale_k, (nk_kernel_punned_t *)&t->each_scale_u64, &used);
    nk_dispatch_u64_find_(caps, nk_kernel_each_sum_k, (nk_kernel_punned_t *)&t->each_sum_u64, &used);
    nk_dispatch_u64_find_(caps, nk_kernel_reduce_add_k, (nk_kernel_punned_t *)&t->reduce_add_u64, &used);
    nk_dispatch_u64_find_(caps, nk_kernel_reduce_min_k, (nk_kernel_punned_t *)&t->reduce_min_u64, &used);
    nk_dispatch_u64_find_(caps, nk_kernel_reduce_max_k, (nk_kernel_punned_t *)&t->reduce_max_u64, &used);
    nk_dispatch_u64_find_(caps, nk_kernel_sparse_intersect_k, (nk_kernel_punned_t *)&t->sparse_intersect_u64, &used);
}

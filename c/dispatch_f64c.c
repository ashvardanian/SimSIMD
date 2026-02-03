/**
 *  @brief Dtype dispatch for f64c (64-bit complex floating point).
 *  @file c/dispatch_f64c.c
 */
#include "dispatch.h"

void nk_dispatch_f64c_find_(nk_capability_t v, nk_kernel_kind_t k, nk_kernel_punned_t *m, nk_capability_t *c) {
    typedef nk_kernel_punned_t m_t;
#if NK_TARGET_SVE
    if (v & nk_cap_sve_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_f64c_sve, *c = nk_cap_sve_k; return;
        case nk_kernel_vdot_k: *m = (m_t)&nk_vdot_f64c_sve, *c = nk_cap_sve_k; return;
        default: break;
        }
#endif
#if NK_TARGET_NEON
    if (v & nk_cap_neon_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_f64c_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_vdot_k: *m = (m_t)&nk_vdot_f64c_neon, *c = nk_cap_neon_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SKYLAKE
    if (v & nk_cap_skylake_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_f64c_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_vdot_k: *m = (m_t)&nk_vdot_f64c_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_bilinear_k: *m = (m_t)&nk_bilinear_f64c_skylake, *c = nk_cap_skylake_k; return;
        default: break;
        }
#endif
    if (v & nk_cap_serial_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_f64c_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_vdot_k: *m = (m_t)&nk_vdot_f64c_serial, *c = nk_cap_serial_k; return;
        default: break;
        }

    // Error fallback - zero capability signals lookup failure
    *m = (m_t)nk_error_dense_, *c = 0;
}

void nk_dispatch_f64c_init_(nk_capability_t caps) {
    nk_implementations_t *t = &nk_dispatch_table;
    nk_capability_t used;

    nk_dispatch_f64c_find_(caps, nk_kernel_dot_k, (nk_kernel_punned_t *)&t->dot_f64c, &used);
    nk_dispatch_f64c_find_(caps, nk_kernel_vdot_k, (nk_kernel_punned_t *)&t->vdot_f64c, &used);
    nk_dispatch_f64c_find_(caps, nk_kernel_bilinear_k, (nk_kernel_punned_t *)&t->bilinear_f64c, &used);
}

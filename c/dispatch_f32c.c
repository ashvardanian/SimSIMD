/**
 *  @brief Dispatch Initialization for F32C Data Types.
 *  @file c/dispatch_f32c.c
 *  @author Ash Vardanian
 *  @date February 3, 2026
 */
#include "dispatch.h"

void nk_dispatch_f32c_find_(nk_capability_t v, nk_kernel_kind_t k, nk_kernel_punned_t *m, nk_capability_t *c) {
    typedef nk_kernel_punned_t m_t;
#if NK_TARGET_SVE
    if (v & nk_cap_sve_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_f32c_sve, *c = nk_cap_sve_k; return;
        case nk_kernel_vdot_k: *m = (m_t)&nk_vdot_f32c_sve, *c = nk_cap_sve_k; return;
        default: break;
        }
#endif
#if NK_TARGET_NEON
    if (v & nk_cap_neon_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_f32c_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_vdot_k: *m = (m_t)&nk_vdot_f32c_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_bilinear_k: *m = (m_t)&nk_bilinear_f32c_neon, *c = nk_cap_neon_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SKYLAKE
    if (v & nk_cap_skylake_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_f32c_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_vdot_k: *m = (m_t)&nk_vdot_f32c_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_bilinear_k: *m = (m_t)&nk_bilinear_f32c_skylake, *c = nk_cap_skylake_k; return;
        default: break;
        }
#endif
#if NK_TARGET_HASWELL
    if (v & nk_cap_haswell_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_f32c_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_vdot_k: *m = (m_t)&nk_vdot_f32c_haswell, *c = nk_cap_haswell_k; return;
        default: break;
        }
#endif
    if (v & nk_cap_serial_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_f32c_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_vdot_k: *m = (m_t)&nk_vdot_f32c_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_bilinear_k: *m = (m_t)&nk_bilinear_f32c_serial, *c = nk_cap_serial_k; return;
        default: break;
        }

    // Error fallback - zero capability signals lookup failure
    *m = (m_t)nk_error_dense_, *c = 0;
}

void nk_dispatch_f32c_init_(nk_capability_t caps) {
    nk_implementations_t *t = &nk_dispatch_table;
    nk_capability_t used;

    nk_dispatch_f32c_find_(caps, nk_kernel_dot_k, (nk_kernel_punned_t *)&t->dot_f32c, &used);
    nk_dispatch_f32c_find_(caps, nk_kernel_vdot_k, (nk_kernel_punned_t *)&t->vdot_f32c, &used);
    nk_dispatch_f32c_find_(caps, nk_kernel_bilinear_k, (nk_kernel_punned_t *)&t->bilinear_f32c, &used);
}

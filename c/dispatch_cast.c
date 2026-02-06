/**
 *  @brief Dispatch Initialization for Type Conversions.
 *  @file c/dispatch_cast.c
 *  @author Ash Vardanian
 *  @date February 3, 2026
 */
#include "dispatch.h"

void nk_dispatch_cast_find_(nk_capability_t v, nk_kernel_kind_t k, nk_kernel_punned_t *m, nk_capability_t *c) {
    typedef nk_kernel_punned_t m_t;
#if NK_TARGET_SAPPHIRE
    if (v & nk_cap_sapphire_k) switch (k) {
        case nk_kernel_cast_k: *m = (m_t)&nk_cast_sapphire, *c = nk_cap_sapphire_k; return;
        default: break;
        }
#endif
#if NK_TARGET_ICELAKE
    if (v & nk_cap_icelake_k) switch (k) {
        case nk_kernel_cast_k: *m = (m_t)&nk_cast_icelake, *c = nk_cap_icelake_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SKYLAKE
    if (v & nk_cap_skylake_k) switch (k) {
        case nk_kernel_cast_k: *m = (m_t)&nk_cast_skylake, *c = nk_cap_skylake_k; return;
        default: break;
        }
#endif
#if NK_TARGET_HASWELL
    if (v & nk_cap_haswell_k) switch (k) {
        case nk_kernel_cast_k: *m = (m_t)&nk_cast_haswell, *c = nk_cap_haswell_k; return;
        default: break;
        }
#endif
#if NK_TARGET_NEON
    if (v & nk_cap_neon_k) switch (k) {
        case nk_kernel_cast_k: *m = (m_t)&nk_cast_neon, *c = nk_cap_neon_k; return;
        default: break;
        }
#endif
    if (v & nk_cap_serial_k) switch (k) {
        case nk_kernel_cast_k: *m = (m_t)&nk_cast_serial, *c = nk_cap_serial_k; return;
        default: break;
        }

    // Error fallback - zero capability signals lookup failure
    *m = (m_t)nk_error_dense_, *c = 0;
}

void nk_dispatch_cast_init_(nk_capability_t caps) {
    nk_implementations_t *t = &nk_dispatch_table;
    nk_capability_t used;

    // Type casting (buffer-to-buffer)
    nk_dispatch_cast_find_(caps, nk_kernel_cast_k, (nk_kernel_punned_t *)&t->cast, &used);

    // Scalar conversions: f16 <-> f32
#if NK_TARGET_SAPPHIRE
    if (caps & nk_cap_sapphire_k) {
        t->f16_to_f32 = &nk_f16_to_f32_sapphire;
        t->f32_to_f16 = &nk_f32_to_f16_sapphire;
    }
    else
#endif
#if NK_TARGET_HASWELL
        if (caps & nk_cap_haswell_k) {
        t->f16_to_f32 = &nk_f16_to_f32_haswell;
        t->f32_to_f16 = &nk_f32_to_f16_haswell;
    }
    else
#endif
#if NK_TARGET_NEON
        if (caps & nk_cap_neon_k) {
        t->f16_to_f32 = &nk_f16_to_f32_neon;
        t->f32_to_f16 = &nk_f32_to_f16_neon;
    }
    else
#endif
    {
        t->f16_to_f32 = &nk_f16_to_f32_serial;
        t->f32_to_f16 = &nk_f32_to_f16_serial;
    }

    // Scalar conversions: bf16, e4m3, e5m2, e2m3, e3m2 (serial only)
    t->bf16_to_f32 = &nk_bf16_to_f32_serial;
    t->f32_to_bf16 = &nk_f32_to_bf16_serial;
    t->e4m3_to_f32 = &nk_e4m3_to_f32_serial;
    t->f32_to_e4m3 = &nk_f32_to_e4m3_serial;
    t->e5m2_to_f32 = &nk_e5m2_to_f32_serial;
    t->f32_to_e5m2 = &nk_f32_to_e5m2_serial;
    t->e2m3_to_f32 = &nk_e2m3_to_f32_serial;
    t->f32_to_e2m3 = &nk_f32_to_e2m3_serial;
    t->e3m2_to_f32 = &nk_e3m2_to_f32_serial;
    t->f32_to_e3m2 = &nk_f32_to_e3m2_serial;
}

// Scalar conversion dispatch functions

NK_DYNAMIC void nk_f16_to_f32(nk_f16_t const *src, nk_f32_t *dest) { nk_dispatch_table.f16_to_f32(src, dest); }
NK_DYNAMIC void nk_f32_to_f16(nk_f32_t const *src, nk_f16_t *dest) { nk_dispatch_table.f32_to_f16(src, dest); }
NK_DYNAMIC void nk_bf16_to_f32(nk_bf16_t const *src, nk_f32_t *dest) { nk_dispatch_table.bf16_to_f32(src, dest); }
NK_DYNAMIC void nk_f32_to_bf16(nk_f32_t const *src, nk_bf16_t *dest) { nk_dispatch_table.f32_to_bf16(src, dest); }
NK_DYNAMIC void nk_e4m3_to_f32(nk_e4m3_t const *src, nk_f32_t *dest) { nk_dispatch_table.e4m3_to_f32(src, dest); }
NK_DYNAMIC void nk_f32_to_e4m3(nk_f32_t const *src, nk_e4m3_t *dest) { nk_dispatch_table.f32_to_e4m3(src, dest); }
NK_DYNAMIC void nk_e5m2_to_f32(nk_e5m2_t const *src, nk_f32_t *dest) { nk_dispatch_table.e5m2_to_f32(src, dest); }
NK_DYNAMIC void nk_f32_to_e5m2(nk_f32_t const *src, nk_e5m2_t *dest) { nk_dispatch_table.f32_to_e5m2(src, dest); }
NK_DYNAMIC void nk_e2m3_to_f32(nk_e2m3_t const *src, nk_f32_t *dest) { nk_dispatch_table.e2m3_to_f32(src, dest); }
NK_DYNAMIC void nk_f32_to_e2m3(nk_f32_t const *src, nk_e2m3_t *dest) { nk_dispatch_table.f32_to_e2m3(src, dest); }
NK_DYNAMIC void nk_e3m2_to_f32(nk_e3m2_t const *src, nk_f32_t *dest) { nk_dispatch_table.e3m2_to_f32(src, dest); }
NK_DYNAMIC void nk_f32_to_e3m2(nk_f32_t const *src, nk_e3m2_t *dest) { nk_dispatch_table.f32_to_e3m2(src, dest); }

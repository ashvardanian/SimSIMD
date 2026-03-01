/**
 *  @brief Software-emulated Scalar Math Helpers for SIMD-free CPUs.
 *  @file include/numkong/scalar/serial.h
 *  @author Ash Vardanian
 *  @date March 1, 2026
 *
 *  @sa include/numkong/scalar.h
 *
 *  Uses the Quake 3 fast inverse square root trick with Newton-Raphson refinement.
 *  Three iterations for f32 (~34.9 correct bits), four for f64 (~69.3 correct bits).
 */
#ifndef NK_SCALAR_SERIAL_H
#define NK_SCALAR_SERIAL_H

#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

NK_PUBLIC nk_f32_t nk_f32_rsqrt_serial(nk_f32_t number) {
    nk_fui32_t conv;
    conv.f = number;
    conv.u = 0x5F375A86 - (conv.u >> 1);
    nk_f32_t y = conv.f;
    y = y * (1.5f - 0.5f * number * y * y);
    y = y * (1.5f - 0.5f * number * y * y);
    y = y * (1.5f - 0.5f * number * y * y);
    return y;
}

NK_PUBLIC nk_f32_t nk_f32_sqrt_serial(nk_f32_t number) { return number > 0 ? number * nk_f32_rsqrt_serial(number) : 0; }

NK_PUBLIC nk_f64_t nk_f64_rsqrt_serial(nk_f64_t number) {
    nk_fui64_t conv;
    conv.f = number;
    conv.u = 0x5FE6EB50C7B537A9ULL - (conv.u >> 1);
    nk_f64_t y = conv.f;
    y = y * (1.5 - 0.5 * number * y * y);
    y = y * (1.5 - 0.5 * number * y * y);
    y = y * (1.5 - 0.5 * number * y * y);
    y = y * (1.5 - 0.5 * number * y * y);
    return y;
}

NK_PUBLIC nk_f64_t nk_f64_sqrt_serial(nk_f64_t number) { return number > 0 ? number * nk_f64_rsqrt_serial(number) : 0; }

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_SCALAR_SERIAL_H

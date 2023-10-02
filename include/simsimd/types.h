#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef float simsimd_f32_t;
typedef double simsimd_f64_t;
typedef _Float16 simsimd_f16_t;
typedef signed char simsimd_i8_t;
typedef unsigned char simsimd_b1_t;
typedef unsigned long long simsimd_size_t;

typedef union {
    unsigned i;
    float f;
} simsimd_f32i32_t;

/**
 *  @brief  Computes `1/sqrt(x)` using the trick from Quake 3.
 */
inline static float simsimd_approximate_inverse_square_root(float number) {
    simsimd_f32i32_t conv = {.f = number};
    conv.i = 0x5f3759df - (conv.i >> 1);
    conv.f *= 1.5F - (number * 0.5F * conv.f * conv.f);
    return conv.f;
}

#ifdef __cplusplus
} // extern "C"
#endif
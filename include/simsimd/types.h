#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef int simsimd_i32_t;
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
 *  @brief  Computes `1/sqrt(x)` using the trick from Quake 3, replacing
 *          magic numbers with the ones suggested by Jan Kadlec.
 */
inline static float simsimd_approximate_inverse_square_root(float number) {
    simsimd_f32i32_t conv;
    conv.f = number;
    conv.i = 0x5F1FFFF9 - (conv.i >> 1);
    conv.f *= 0.703952253f * (2.38924456f - number * conv.f * conv.f);
    return conv.f;
}

#ifdef __cplusplus
} // extern "C"
#endif
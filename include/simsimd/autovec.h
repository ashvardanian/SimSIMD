#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SIMSIMD_AUTO_L2SQ(name, input_type, accumulator_type)                                                          \
    inline static simsimd_f32_t simsimd_##name##_##input_type##_l2sq(                                                  \
        simsimd_##input_type##_t const* a, simsimd_##input_type##_t const* b, simsimd_size_t d) {                      \
        simsimd_##accumulator_type##_t d2 = 0;                                                                         \
        for (simsimd_size_t i = 0; i != d; ++i) {                                                                      \
            simsimd_##accumulator_type##_t ai = a[i];                                                                  \
            simsimd_##accumulator_type##_t bi = b[i];                                                                  \
            d2 += (ai - bi) * (ai - bi);                                                                               \
        }                                                                                                              \
        return d2;                                                                                                     \
    }

#define SIMSIMD_AUTO_IP(name, input_type, accumulator_type)                                                            \
    inline static simsimd_f32_t simsimd_##name##_##input_type##_ip(                                                    \
        simsimd_##input_type##_t const* a, simsimd_##input_type##_t const* b, simsimd_size_t d) {                      \
        simsimd_##accumulator_type##_t ab = 0;                                                                         \
        for (simsimd_size_t i = 0; i != d; ++i) {                                                                      \
            simsimd_##accumulator_type##_t ai = a[i];                                                                  \
            simsimd_##accumulator_type##_t bi = b[i];                                                                  \
            ab += ai * bi;                                                                                             \
        }                                                                                                              \
        return 1 - ab;                                                                                                 \
    }

#define SIMSIMD_AUTO_COS(name, input_type, accumulator_type)                                                           \
    inline static simsimd_f32_t simsimd_##name##_##input_type##_cos(                                                   \
        simsimd_##input_type##_t const* a, simsimd_##input_type##_t const* b, simsimd_size_t d) {                      \
        simsimd_##accumulator_type##_t ab = 0, a2 = 0, b2 = 0;                                                         \
        for (simsimd_size_t i = 0; i != d; ++i) {                                                                      \
            simsimd_##accumulator_type##_t ai = a[i];                                                                  \
            simsimd_##accumulator_type##_t bi = b[i];                                                                  \
            ab += ai * bi;                                                                                             \
            a2 += ai * ai;                                                                                             \
            b2 += bi * bi;                                                                                             \
        }                                                                                                              \
        return 1 - ab * simsimd_approximate_inverse_square_root(a2 * b2);                                              \
    }

SIMSIMD_AUTO_L2SQ(auto, f32, f32)
SIMSIMD_AUTO_IP(auto, f32, f32)
SIMSIMD_AUTO_COS(auto, f32, f32)

SIMSIMD_AUTO_L2SQ(auto, f16, f32)
SIMSIMD_AUTO_IP(auto, f16, f32)
SIMSIMD_AUTO_COS(auto, f16, f32)

SIMSIMD_AUTO_L2SQ(auto, i8, f32)
SIMSIMD_AUTO_COS(auto, i8, f32)

inline static simsimd_f32_t simsimd_auto_i8_ip(simsimd_i8_t const* a, simsimd_i8_t const* b, simsimd_size_t d) {
    return simsimd_auto_i8_cos(a, b, d);
}

SIMSIMD_AUTO_L2SQ(accurate, f32, f64)
SIMSIMD_AUTO_IP(accurate, f32, f64)
SIMSIMD_AUTO_COS(accurate, f32, f64)

SIMSIMD_AUTO_L2SQ(accurate, f16, f64)
SIMSIMD_AUTO_IP(accurate, f16, f64)
SIMSIMD_AUTO_COS(accurate, f16, f64)

SIMSIMD_AUTO_L2SQ(accurate, i8, f64)
SIMSIMD_AUTO_COS(accurate, i8, f64)

inline static simsimd_f32_t simsimd_accurate_i8_ip(simsimd_i8_t const* a, simsimd_i8_t const* b, simsimd_size_t d) {
    return simsimd_accurate_i8_cos(a, b, d);
}

#ifdef __cplusplus
} // extern "C"
#endif
#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SIMSIMD_AUTO_L2SQ(type)                                                                                        \
    inline static simsimd_f32_t simsimd_auto_##type##_l2sq(simsimd_##type##_t const* a, simsimd_##type##_t const* b,   \
                                                           simsimd_size_t d) {                                         \
        simsimd_f32_t d2 = 0;                                                                                          \
        for (simsimd_size_t i = 0; i != d; ++i) {                                                                      \
            simsimd_f32_t ai = a[i];                                                                                   \
            simsimd_f32_t bi = b[i];                                                                                   \
            d2 += (ai - bi) * (ai - bi);                                                                               \
        }                                                                                                              \
        return d2;                                                                                                     \
    }

#define SIMSIMD_AUTO_IP(type)                                                                                          \
    inline static simsimd_f32_t simsimd_auto_##type##_ip(simsimd_##type##_t const* a, simsimd_##type##_t const* b,     \
                                                         simsimd_size_t d) {                                           \
        simsimd_f32_t ab = 0;                                                                                          \
        for (simsimd_size_t i = 0; i != d; ++i) {                                                                      \
            simsimd_f32_t ai = a[i];                                                                                   \
            simsimd_f32_t bi = b[i];                                                                                   \
            ab += ai * bi;                                                                                             \
        }                                                                                                              \
        return 1 - ab;                                                                                                 \
    }

#define SIMSIMD_AUTO_COS(type)                                                                                         \
    inline static simsimd_f32_t simsimd_auto_##type##_cos(simsimd_##type##_t const* a, simsimd_##type##_t const* b,    \
                                                          simsimd_size_t d) {                                          \
        simsimd_f32_t ab = 0, a2 = 0, b2 = 0;                                                                          \
        for (simsimd_size_t i = 0; i != d; ++i) {                                                                      \
            simsimd_f32_t ai = a[i];                                                                                   \
            simsimd_f32_t bi = b[i];                                                                                   \
            ab += ai * bi;                                                                                             \
            a2 += ai * ai;                                                                                             \
            b2 += bi * bi;                                                                                             \
        }                                                                                                              \
        return 1 - ab;                                                                                                 \
    }

SIMSIMD_AUTO_L2SQ(f32)
SIMSIMD_AUTO_IP(f32)
SIMSIMD_AUTO_COS(f32)

SIMSIMD_AUTO_L2SQ(f16)
SIMSIMD_AUTO_IP(f16)
SIMSIMD_AUTO_COS(f16)

SIMSIMD_AUTO_L2SQ(i8)
SIMSIMD_AUTO_COS(i8)

inline static simsimd_f32_t simsimd_auto_i8_ip(simsimd_i8_t const* a, simsimd_i8_t const* b, simsimd_size_t d) {
    return simsimd_auto_i8_cos(a, b, d);
}

#ifdef __cplusplus
} // extern "C"
#endif
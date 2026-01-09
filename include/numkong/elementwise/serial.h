/**
 *  @brief SIMD-accelerated Dot Products for Real and Complex Numbers optimized for SIMD-free CPUs.
 *  @file include/numkong/elementwise/serial.h
 *  @sa include/numkong/elementwise.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 */
#ifndef NK_ELEMENTWISE_SERIAL_H
#define NK_ELEMENTWISE_SERIAL_H
#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

#define nk_define_scale_(input_type, accumulator_type, load_and_convert, convert_and_store) \
    NK_PUBLIC void nk_scale_##input_type##_serial(                                          \
        nk_##input_type##_t const *a, nk_size_t n, nk_##accumulator_type##_t const *alpha,  \
        nk_##accumulator_type##_t const *beta, nk_##input_type##_t *result) {               \
        nk_##accumulator_type##_t alpha_val = *alpha;                                       \
        nk_##accumulator_type##_t beta_val = *beta;                                         \
        nk_##accumulator_type##_t ai, sum;                                                  \
        for (nk_size_t i = 0; i != n; ++i) {                                                \
            load_and_convert(a + i, &ai);                                                   \
            sum = (nk_##accumulator_type##_t)(alpha_val * ai + beta_val);                   \
            convert_and_store(&sum, result + i);                                            \
        }                                                                                   \
    }
#define nk_define_sum_(input_type, accumulator_type, load_and_convert, convert_and_store)                   \
    NK_PUBLIC void nk_sum_##input_type##_serial(nk_##input_type##_t const *a, nk_##input_type##_t const *b, \
                                                nk_size_t n, nk_##input_type##_t *result) {                 \
        nk_##accumulator_type##_t ai, bi, sum;                                                              \
        for (nk_size_t i = 0; i != n; ++i) {                                                                \
            load_and_convert(a + i, &ai);                                                                   \
            load_and_convert(b + i, &bi);                                                                   \
            sum = ai + bi;                                                                                  \
            convert_and_store(&sum, result + i);                                                            \
        }                                                                                                   \
    }

#define nk_define_wsum_(input_type, accumulator_type, load_and_convert, convert_and_store)                             \
    NK_PUBLIC void nk_wsum_##input_type##_serial(nk_##input_type##_t const *a, nk_##input_type##_t const *b,           \
                                                 nk_size_t n, nk_##accumulator_type##_t const *alpha,                  \
                                                 nk_##accumulator_type##_t const *beta, nk_##input_type##_t *result) { \
        nk_##accumulator_type##_t alpha_val = *alpha;                                                                  \
        nk_##accumulator_type##_t beta_val = *beta;                                                                    \
        nk_##accumulator_type##_t ai, bi, ai_scaled, bi_scaled, sum;                                                   \
        for (nk_size_t i = 0; i != n; ++i) {                                                                           \
            load_and_convert(a + i, &ai);                                                                              \
            load_and_convert(b + i, &bi);                                                                              \
            ai_scaled = ai * alpha_val;                                                                                \
            bi_scaled = bi * beta_val;                                                                                 \
            sum = ai_scaled + bi_scaled;                                                                               \
            convert_and_store(&sum, result + i);                                                                       \
        }                                                                                                              \
    }

#define nk_define_fma_(input_type, accumulator_type, load_and_convert, convert_and_store)                             \
    NK_PUBLIC void nk_fma_##input_type##_serial(                                                                      \
        nk_##input_type##_t const *a, nk_##input_type##_t const *b, nk_##input_type##_t const *c, nk_size_t n,        \
        nk_##accumulator_type##_t const *alpha, nk_##accumulator_type##_t const *beta, nk_##input_type##_t *result) { \
        nk_##accumulator_type##_t alpha_val = *alpha;                                                                 \
        nk_##accumulator_type##_t beta_val = *beta;                                                                   \
        nk_##accumulator_type##_t ai, bi, ci, abi_scaled, ci_scaled, sum;                                             \
        for (nk_size_t i = 0; i != n; ++i) {                                                                          \
            load_and_convert(a + i, &ai);                                                                             \
            load_and_convert(b + i, &bi);                                                                             \
            load_and_convert(c + i, &ci);                                                                             \
            abi_scaled = ai * bi * alpha_val;                                                                         \
            ci_scaled = ci * beta_val;                                                                                \
            sum = abi_scaled + ci_scaled;                                                                             \
            convert_and_store(&sum, result + i);                                                                      \
        }                                                                                                             \
    }

nk_define_sum_(f64, f64, nk_assign_from_to_, nk_assign_from_to_)        // nk_sum_f64_serial
nk_define_sum_(f32, f32, nk_assign_from_to_, nk_assign_from_to_)        // nk_sum_f32_serial
nk_define_sum_(f16, f32, nk_f16_to_f32_serial, nk_f32_to_f16_serial)    // nk_sum_f16_serial
nk_define_sum_(bf16, f32, nk_bf16_to_f32_serial, nk_f32_to_bf16_serial) // nk_sum_bf16_serial
nk_define_sum_(e4m3, f32, nk_e4m3_to_f32_serial, nk_f32_to_e4m3_serial) // nk_sum_e4m3_serial
nk_define_sum_(e5m2, f32, nk_e5m2_to_f32_serial, nk_f32_to_e5m2_serial) // nk_sum_e5m2_serial
nk_define_sum_(i8, i64, nk_assign_from_to_, nk_i64_to_i8_serial)        // nk_sum_i8_serial
nk_define_sum_(u8, i64, nk_assign_from_to_, nk_i64_to_u8_serial)        // nk_sum_u8_serial
nk_define_sum_(i16, i64, nk_assign_from_to_, nk_i64_to_i16_serial)      // nk_sum_i16_serial
nk_define_sum_(u16, i64, nk_assign_from_to_, nk_i64_to_u16_serial)      // nk_sum_u16_serial
nk_define_sum_(i32, i64, nk_assign_from_to_, nk_i64_to_i32_serial)      // nk_sum_i32_serial
nk_define_sum_(u32, i64, nk_assign_from_to_, nk_i64_to_u32_serial)      // nk_sum_u32_serial
nk_define_sum_(i64, i64, nk_assign_from_to_, nk_assign_from_to_)        // nk_sum_i64_serial
nk_define_sum_(u64, u64, nk_assign_from_to_, nk_assign_from_to_)        // nk_sum_u64_serial

nk_define_scale_(f64, f64, nk_assign_from_to_, nk_assign_from_to_)        // nk_scale_f64_serial
nk_define_scale_(f32, f32, nk_assign_from_to_, nk_assign_from_to_)        // nk_scale_f32_serial
nk_define_scale_(f16, f32, nk_f16_to_f32_serial, nk_f32_to_f16_serial)    // nk_scale_f16_serial
nk_define_scale_(bf16, f32, nk_bf16_to_f32_serial, nk_f32_to_bf16_serial) // nk_scale_bf16_serial
nk_define_scale_(e4m3, f32, nk_e4m3_to_f32_serial, nk_f32_to_e4m3_serial) // nk_scale_e4m3_serial
nk_define_scale_(e5m2, f32, nk_e5m2_to_f32_serial, nk_f32_to_e5m2_serial) // nk_scale_e5m2_serial
nk_define_scale_(i8, f32, nk_assign_from_to_, nk_f32_to_i8_serial)        // nk_scale_i8_serial
nk_define_scale_(u8, f32, nk_assign_from_to_, nk_f32_to_u8_serial)        // nk_scale_u8_serial
nk_define_scale_(i16, f32, nk_assign_from_to_, nk_f32_to_i16_serial)      // nk_scale_i16_serial
nk_define_scale_(u16, f32, nk_assign_from_to_, nk_f32_to_u16_serial)      // nk_scale_u16_serial
nk_define_scale_(i32, f64, nk_assign_from_to_, nk_f64_to_i32_serial)      // nk_scale_i32_serial
nk_define_scale_(u32, f64, nk_assign_from_to_, nk_f64_to_u32_serial)      // nk_scale_u32_serial
nk_define_scale_(i64, f64, nk_assign_from_to_, nk_f64_to_i64_serial)      // nk_scale_i64_serial
nk_define_scale_(u64, f64, nk_assign_from_to_, nk_f64_to_u64_serial)      // nk_scale_u64_serial

nk_define_wsum_(f64, f64, nk_assign_from_to_, nk_assign_from_to_)        // nk_wsum_f64_serial
nk_define_wsum_(f32, f32, nk_assign_from_to_, nk_assign_from_to_)        // nk_wsum_f32_serial
nk_define_wsum_(f16, f32, nk_f16_to_f32_serial, nk_f32_to_f16_serial)    // nk_wsum_f16_serial
nk_define_wsum_(bf16, f32, nk_bf16_to_f32_serial, nk_f32_to_bf16_serial) // nk_wsum_bf16_serial
nk_define_wsum_(e4m3, f32, nk_e4m3_to_f32_serial, nk_f32_to_e4m3_serial) // nk_wsum_e4m3_serial
nk_define_wsum_(e5m2, f32, nk_e5m2_to_f32_serial, nk_f32_to_e5m2_serial) // nk_wsum_e5m2_serial
nk_define_wsum_(i8, f32, nk_assign_from_to_, nk_f32_to_i8_serial)        // nk_wsum_i8_serial
nk_define_wsum_(u8, f32, nk_assign_from_to_, nk_f32_to_u8_serial)        // nk_wsum_u8_serial
nk_define_wsum_(i16, f32, nk_assign_from_to_, nk_f32_to_i16_serial)      // nk_wsum_i16_serial
nk_define_wsum_(u16, f32, nk_assign_from_to_, nk_f32_to_u16_serial)      // nk_wsum_u16_serial
nk_define_wsum_(i32, f64, nk_assign_from_to_, nk_f64_to_i32_serial)      // nk_wsum_i32_serial
nk_define_wsum_(u32, f64, nk_assign_from_to_, nk_f64_to_u32_serial)      // nk_wsum_u32_serial
nk_define_wsum_(i64, f64, nk_assign_from_to_, nk_f64_to_i64_serial)      // nk_wsum_i64_serial
nk_define_wsum_(u64, f64, nk_assign_from_to_, nk_f64_to_u64_serial)      // nk_wsum_u64_serial

nk_define_fma_(f64, f64, nk_assign_from_to_, nk_assign_from_to_)        // nk_fma_f64_serial
nk_define_fma_(f32, f32, nk_assign_from_to_, nk_assign_from_to_)        // nk_fma_f32_serial
nk_define_fma_(f16, f32, nk_f16_to_f32_serial, nk_f32_to_f16_serial)    // nk_fma_f16_serial
nk_define_fma_(bf16, f32, nk_bf16_to_f32_serial, nk_f32_to_bf16_serial) // nk_fma_bf16_serial
nk_define_fma_(e4m3, f32, nk_e4m3_to_f32_serial, nk_f32_to_e4m3_serial) // nk_fma_e4m3_serial
nk_define_fma_(e5m2, f32, nk_e5m2_to_f32_serial, nk_f32_to_e5m2_serial) // nk_fma_e5m2_serial
nk_define_fma_(i8, f32, nk_assign_from_to_, nk_f32_to_i8_serial)        // nk_fma_i8_serial
nk_define_fma_(u8, f32, nk_assign_from_to_, nk_f32_to_u8_serial)        // nk_fma_u8_serial
nk_define_fma_(i16, f32, nk_assign_from_to_, nk_f32_to_i16_serial)      // nk_fma_i16_serial
nk_define_fma_(u16, f32, nk_assign_from_to_, nk_f32_to_u16_serial)      // nk_fma_u16_serial
nk_define_fma_(i32, f64, nk_assign_from_to_, nk_f64_to_i32_serial)      // nk_fma_i32_serial
nk_define_fma_(u32, f64, nk_assign_from_to_, nk_f64_to_u32_serial)      // nk_fma_u32_serial
nk_define_fma_(i64, f64, nk_assign_from_to_, nk_f64_to_i64_serial)      // nk_fma_i64_serial
nk_define_fma_(u64, f64, nk_assign_from_to_, nk_f64_to_u64_serial)      // nk_fma_u64_serial

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_ELEMENTWISE_SERIAL_H

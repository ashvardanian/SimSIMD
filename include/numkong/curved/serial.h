/**
 *  @brief SWAR-accelerated Curved Space Similarity for SIMD-free CPUs.
 *  @file include/numkong/curved/serial.h
 *  @author Ash Vardanian
 *  @date January 14, 2026
 *
 *  @sa include/numkong/curved.h
 *
 *  Implements bilinear forms and Mahalanobis distance with enhanced numerical precision:
 *  - f32 inputs use f64 accumulators to avoid catastrophic cancellation
 *  - All types use Neumaier compensated summation for both inner and outer loops
 *
 *  @section precision Precision Strategy
 *
 *  Bilinear form: aᵀ × C × b = Σᵢ aᵢ × (Σⱼ cᵢⱼ × bⱼ)
 *
 *  The nested loop structure has two accumulation levels:
 *  - Inner accumulation: Σⱼ cᵢⱼ × bⱼ (O(n) terms per row)
 *  - Outer accumulation: Σᵢ aᵢ × inner_result (O(n) terms total)
 *
 *  Both levels apply Neumaier's Kahan-Babuška variant to achieve O(1) error growth.
 *
 *  @see Neumaier, A. (1974). "Rundungsfehleranalyse einiger Verfahren zur Summation endlicher Summen"
 */
#ifndef NK_CURVED_SERIAL_H
#define NK_CURVED_SERIAL_H

#include "numkong/types.h"
#include "numkong/reduce/serial.h"  // `nk_f64_abs_`
#include "numkong/spatial/serial.h" // `nk_f64_sqrt_serial`

#if defined(__cplusplus)
extern "C" {
#endif

/**
 *  @brief Macro for bilinear form aᵀ × C × b with Neumaier compensated summation.
 *
 *  Computes Σᵢ Σⱼ aᵢ × cᵢⱼ × bⱼ using double-Neumaier compensation:
 *  - Inner loop: accumulates cᵢⱼ × bⱼ with compensation
 *  - Outer loop: accumulates aᵢ × inner_result with compensation
 *
 *  @note For f32 inputs, use f64 accumulator to provide ~2× precision headroom.
 */
#define nk_define_bilinear_(input_type, accumulator_type, output_type, load_and_convert)                         \
    NK_PUBLIC void nk_bilinear_##input_type##_serial(nk_##input_type##_t const *a, nk_##input_type##_t const *b, \
                                                     nk_##input_type##_t const *c, nk_size_t n,                  \
                                                     nk_##output_type##_t *result) {                             \
        nk_##accumulator_type##_t outer_sum = 0, outer_compensation = 0;                                         \
        nk_##accumulator_type##_t vector_a_value, vector_b_value, tensor_value;                                  \
        for (nk_size_t row = 0; row != n; ++row) {                                                               \
            nk_##accumulator_type##_t inner_sum = 0, inner_compensation = 0;                                     \
            load_and_convert(a + row, &vector_a_value);                                                          \
            for (nk_size_t column = 0; column != n; ++column) {                                                  \
                load_and_convert(b + column, &vector_b_value);                                                   \
                load_and_convert(c + row * n + column, &tensor_value);                                           \
                nk_##accumulator_type##_t product = tensor_value * vector_b_value;                               \
                nk_##accumulator_type##_t running = inner_sum + product;                                         \
                /* Neumaier: track lost bits depending on which operand is larger */                             \
                inner_compensation += (nk_##accumulator_type##_abs_(inner_sum) >=                                \
                                       nk_##accumulator_type##_abs_(product))                                    \
                                          ? ((inner_sum - running) + product)                                    \
                                          : ((product - running) + inner_sum);                                   \
                inner_sum = running;                                                                             \
            }                                                                                                    \
            inner_sum += inner_compensation;                                                                     \
            nk_##accumulator_type##_t outer_product = vector_a_value * inner_sum;                                \
            nk_##accumulator_type##_t outer_running = outer_sum + outer_product;                                 \
            outer_compensation += (nk_##accumulator_type##_abs_(outer_sum) >=                                    \
                                   nk_##accumulator_type##_abs_(outer_product))                                  \
                                      ? ((outer_sum - outer_running) + outer_product)                            \
                                      : ((outer_product - outer_running) + outer_sum);                           \
            outer_sum = outer_running;                                                                           \
        }                                                                                                        \
        *result = (nk_##output_type##_t)(outer_sum + outer_compensation);                                        \
    }

/**
 *  @brief Macro for complex bilinear form aᵀ × C × b with Neumaier compensated summation.
 *
 *  For complex numbers, computes:
 *    (Σᵢ Σⱼ aᵢ × cᵢⱼ × bⱼ).real and (Σᵢ Σⱼ aᵢ × cᵢⱼ × bⱼ).imag
 *
 *  Complex multiplication: (a + bi)(c + di) = (ac - bd) + (ad + bc)i
 *  Each real and imaginary accumulator has independent Neumaier compensation.
 */
#define nk_define_bilinear_complex_(input_type, accumulator_type, output_type, load_and_convert)                                 \
    NK_PUBLIC void nk_bilinear_##input_type##_serial(                                                                            \
        nk_##input_type##_t const *a_pairs, nk_##input_type##_t const *b_pairs, nk_##input_type##_t const *c_pairs,              \
        nk_size_t n, nk_##output_type##c_t *results) {                                                                           \
        nk_##accumulator_type##_t outer_sum_real = 0, outer_sum_imag = 0;                                                        \
        nk_##accumulator_type##_t outer_compensation_real = 0, outer_compensation_imag = 0;                                      \
        nk_##accumulator_type##_t a_real, a_imag, b_real, b_imag, c_real, c_imag;                                                \
        for (nk_size_t row = 0; row != n; ++row) {                                                                               \
            nk_##accumulator_type##_t inner_sum_real = 0, inner_sum_imag = 0;                                                    \
            nk_##accumulator_type##_t inner_compensation_real = 0, inner_compensation_imag = 0;                                  \
            load_and_convert(&(a_pairs + row)->real, &a_real);                                                                   \
            load_and_convert(&(a_pairs + row)->imag, &a_imag);                                                                   \
            for (nk_size_t column = 0; column != n; ++column) {                                                                  \
                load_and_convert(&(b_pairs + column)->real, &b_real);                                                            \
                load_and_convert(&(b_pairs + column)->imag, &b_imag);                                                            \
                load_and_convert(&(c_pairs + row * n + column)->real, &c_real);                                                  \
                load_and_convert(&(c_pairs + row * n + column)->imag, &c_imag);                                                  \
                /* Complex multiply: cᵢⱼ × bⱼ = (c_real×b_real - c_imag×b_imag) + (c_real×b_imag + c_imag×b_real)i */ \
                nk_##accumulator_type##_t product_real = c_real * b_real - c_imag * b_imag;                                      \
                nk_##accumulator_type##_t product_imag = c_real * b_imag + c_imag * b_real;                                      \
                /* Neumaier for real part */                                                                                     \
                nk_##accumulator_type##_t running_real = inner_sum_real + product_real;                                          \
                inner_compensation_real += (nk_##accumulator_type##_abs_(inner_sum_real) >=                                      \
                                            nk_##accumulator_type##_abs_(product_real))                                          \
                                               ? ((inner_sum_real - running_real) + product_real)                                \
                                               : ((product_real - running_real) + inner_sum_real);                               \
                inner_sum_real = running_real;                                                                                   \
                /* Neumaier for imaginary part */                                                                                \
                nk_##accumulator_type##_t running_imag = inner_sum_imag + product_imag;                                          \
                inner_compensation_imag += (nk_##accumulator_type##_abs_(inner_sum_imag) >=                                      \
                                            nk_##accumulator_type##_abs_(product_imag))                                          \
                                               ? ((inner_sum_imag - running_imag) + product_imag)                                \
                                               : ((product_imag - running_imag) + inner_sum_imag);                               \
                inner_sum_imag = running_imag;                                                                                   \
            }                                                                                                                    \
            inner_sum_real += inner_compensation_real;                                                                           \
            inner_sum_imag += inner_compensation_imag;                                                                           \
            /* Complex multiply: aᵢ × inner_result */                                                                         \
            nk_##accumulator_type##_t outer_product_real = a_real * inner_sum_real - a_imag * inner_sum_imag;                    \
            nk_##accumulator_type##_t outer_product_imag = a_real * inner_sum_imag + a_imag * inner_sum_real;                    \
            /* Neumaier for outer real */                                                                                        \
            nk_##accumulator_type##_t outer_running_real = outer_sum_real + outer_product_real;                                  \
            outer_compensation_real += (nk_##accumulator_type##_abs_(outer_sum_real) >=                                          \
                                        nk_##accumulator_type##_abs_(outer_product_real))                                        \
                                           ? ((outer_sum_real - outer_running_real) + outer_product_real)                        \
                                           : ((outer_product_real - outer_running_real) + outer_sum_real);                       \
            outer_sum_real = outer_running_real;                                                                                 \
            /* Neumaier for outer imaginary */                                                                                   \
            nk_##accumulator_type##_t outer_running_imag = outer_sum_imag + outer_product_imag;                                  \
            outer_compensation_imag += (nk_##accumulator_type##_abs_(outer_sum_imag) >=                                          \
                                        nk_##accumulator_type##_abs_(outer_product_imag))                                        \
                                           ? ((outer_sum_imag - outer_running_imag) + outer_product_imag)                        \
                                           : ((outer_product_imag - outer_running_imag) + outer_sum_imag);                       \
            outer_sum_imag = outer_running_imag;                                                                                 \
        }                                                                                                                        \
        results->real = outer_sum_real + outer_compensation_real;                                                                \
        results->imag = outer_sum_imag + outer_compensation_imag;                                                                \
    }

/**
 *  @brief Macro for Mahalanobis distance √((a-b)ᵀ × C × (a-b)) with Neumaier compensated summation.
 *
 *  Computes √(Σᵢ Σⱼ (aᵢ-bᵢ) × cᵢⱼ × (aⱼ-bⱼ)) using double-Neumaier compensation.
 *  Differences are computed in the accumulator precision to avoid early precision loss.
 */
#define nk_define_mahalanobis_(input_type, accumulator_type, output_type, load_and_convert)                         \
    NK_PUBLIC void nk_mahalanobis_##input_type##_serial(nk_##input_type##_t const *a, nk_##input_type##_t const *b, \
                                                        nk_##input_type##_t const *c, nk_size_t n,                  \
                                                        nk_##output_type##_t *result) {                             \
        nk_##accumulator_type##_t outer_sum = 0, outer_compensation = 0;                                            \
        nk_##accumulator_type##_t a_row_value, b_row_value, a_column_value, b_column_value, tensor_value;           \
        for (nk_size_t row = 0; row != n; ++row) {                                                                  \
            nk_##accumulator_type##_t inner_sum = 0, inner_compensation = 0;                                        \
            load_and_convert(a + row, &a_row_value);                                                                \
            load_and_convert(b + row, &b_row_value);                                                                \
            nk_##accumulator_type##_t difference_row = a_row_value - b_row_value;                                   \
            for (nk_size_t column = 0; column != n; ++column) {                                                     \
                load_and_convert(a + column, &a_column_value);                                                      \
                load_and_convert(b + column, &b_column_value);                                                      \
                load_and_convert(c + row * n + column, &tensor_value);                                              \
                nk_##accumulator_type##_t difference_column = a_column_value - b_column_value;                      \
                nk_##accumulator_type##_t product = tensor_value * difference_column;                               \
                nk_##accumulator_type##_t running = inner_sum + product;                                            \
                /* Neumaier: track lost bits depending on which operand is larger */                                \
                inner_compensation += (nk_##accumulator_type##_abs_(inner_sum) >=                                   \
                                       nk_##accumulator_type##_abs_(product))                                       \
                                          ? ((inner_sum - running) + product)                                       \
                                          : ((product - running) + inner_sum);                                      \
                inner_sum = running;                                                                                \
            }                                                                                                       \
            inner_sum += inner_compensation;                                                                        \
            nk_##accumulator_type##_t outer_product = difference_row * inner_sum;                                   \
            nk_##accumulator_type##_t outer_running = outer_sum + outer_product;                                    \
            outer_compensation += (nk_##accumulator_type##_abs_(outer_sum) >=                                       \
                                   nk_##accumulator_type##_abs_(outer_product))                                     \
                                      ? ((outer_sum - outer_running) + outer_product)                               \
                                      : ((outer_product - outer_running) + outer_sum);                              \
            outer_sum = outer_running;                                                                              \
        }                                                                                                           \
        nk_##accumulator_type##_t quadratic = outer_sum + outer_compensation;                                       \
        *result = nk_##accumulator_type##_sqrt_serial(quadratic > 0 ? quadratic : 0);                               \
    }

// f64 → f64: Native precision with Neumaier compensation
nk_define_bilinear_(f64, f64, f64, nk_assign_from_to_)          // nk_bilinear_f64_serial
nk_define_bilinear_complex_(f64c, f64, f64, nk_assign_from_to_) // nk_bilinear_f64c_serial
nk_define_mahalanobis_(f64, f64, f64, nk_assign_from_to_)       // nk_mahalanobis_f64_serial

// f32 → f64 accumulator → f32 output: Upcast internally for precision + Neumaier compensation
nk_define_bilinear_(f32, f64, f32, nk_assign_from_to_)          // nk_bilinear_f32_serial
nk_define_bilinear_complex_(f32c, f64, f32, nk_assign_from_to_) // nk_bilinear_f32c_serial
nk_define_mahalanobis_(f32, f64, f32, nk_assign_from_to_)       // nk_mahalanobis_f32_serial

// f16 → f32 accumulator → f32 output: f32 provides ample headroom for f16 (~3 vs ~7 decimal digits)
nk_define_bilinear_(f16, f32, f32, nk_f16_to_f32_serial)          // nk_bilinear_f16_serial
nk_define_bilinear_complex_(f16c, f32, f32, nk_f16_to_f32_serial) // nk_bilinear_f16c_serial
nk_define_mahalanobis_(f16, f32, f32, nk_f16_to_f32_serial)       // nk_mahalanobis_f16_serial

// bf16 → f32 accumulator → f32 output: f32 provides ample headroom for bf16 (~2 vs ~7 decimal digits)
nk_define_bilinear_(bf16, f32, f32, nk_bf16_to_f32_serial)          // nk_bilinear_bf16_serial
nk_define_bilinear_complex_(bf16c, f32, f32, nk_bf16_to_f32_serial) // nk_bilinear_bf16c_serial
nk_define_mahalanobis_(bf16, f32, f32, nk_bf16_to_f32_serial)       // nk_mahalanobis_bf16_serial

#undef nk_define_bilinear_
#undef nk_define_bilinear_complex_
#undef nk_define_mahalanobis_

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_CURVED_SERIAL_H

/**
 *  @brief SIMD-accelerated type conversions for FP8/BF16/F16 types optimized for Serial (SIMD-free) CPUs.
 *  @file include/numkong/cast/serial.h
 *  @author Ash Vardanian
 *  @date January 2, 2026
 */
#ifndef NK_CAST_SERIAL_H
#define NK_CAST_SERIAL_H

#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

#pragma region - Type Punned Loads and Stores

/** @brief Type-agnostic 32-bit full load (scalar). */
NK_INTERNAL void nk_load_b32_serial_(void const *src, nk_b32_vec_t *dst) { dst->u32 = *(nk_u32_t const *)src; }

/** @brief Type-agnostic 32-bit full store (scalar). */
NK_INTERNAL void nk_store_b32_serial_(nk_b32_vec_t const *src, void *dst) { *(nk_u32_t *)dst = src->u32; }

#pragma endregion - Type Punned Loads and Stores

/**
 *  @brief Expands an `f16` (IEEE-754 16-bit) to a `float`.
 *
 *  Handles all IEEE-754 edge cases:
 *
 *       Input        F16 Hex   F32 Hex       Description
 *       +0           0x0000    0x00000000    Positive zero
 *       -0           0x8000    0x80000000    Negative zero
 *       +inf         0x7C00    0x7F800000    Positive infinity
 *       -inf         0xFC00    0xFF800000    Negative infinity
 *       NaN          0x7E00    0x7FC00000    Quiet NaN (payload preserved)
 *       Min normal   0x0400    0x38800000    2⁻¹⁴
 *       Max normal   0x7BFF    0x477FE000    65504
 *       Min denorm   0x0001    0x33800000    2⁻²⁴
 *       Max denorm   0x03FF    0x387FC000    2⁻¹⁴ - 2⁻²⁴
 *
 *  https://stackoverflow.com/a/60047308
 *  https://gist.github.com/milhidaka/95863906fe828198f47991c813dbe233
 *  https://github.com/OpenCyphal/libcanard/blob/636795f4bc395f56af8d2c61d3757b5e762bb9e5/canard.c#L811-L834
 */
NK_INTERNAL void nk_f16_to_f32_serial(nk_f16_t const *src, nk_f32_t *dest) {
#if NK_NATIVE_F16
    *dest = (nk_f32_t)(*src);
#else
    unsigned short x;
    nk_copy_bytes_(&x, src, 2);

    unsigned int sign = (x >> 15) & 1;
    unsigned int exponent = (x >> 10) & 0x1F;
    unsigned int mantissa = x & 0x03FF;

    nk_fui32_t conv;

    if (exponent == 0) {
        if (mantissa == 0) {
            // Zero (preserve sign)
            conv.u = sign << 31;
        }
        else {
            // Denormal: value = mantissa × 2⁻²⁴
            // Use FPU normalization, then subtract 24 from exponent
            nk_fui32_t temp;
            temp.f = (float)mantissa;
            conv.u = (sign << 31) | (temp.u - 0x0C000000);
        }
    }
    else if (exponent == 31) {
        // Infinity (mantissa=0) or NaN (mantissa!=0)
        conv.u = (sign << 31) | 0x7F800000 | (mantissa << 13);
    }
    else {
        // Normal: rebias exponent (127-15=112), shift mantissa
        conv.u = (sign << 31) | ((exponent + 112) << 23) | (mantissa << 13);
    }

    *dest = conv.f;
#endif
}

/**
 *  @brief Compresses a `float` to an `f16` (IEEE-754 16-bit).
 *
 *  Handles all IEEE-754 edge cases with round-to-nearest:
 *
 *      Input           F32 Hex       F16 Hex   Description
 *      +0              0x00000000    0x0000    Positive zero
 *      -0              0x80000000    0x8000    Negative zero
 *      +inf            0x7F800000    0x7C00    Positive infinity
 *      -inf            0xFF800000    0xFC00    Negative infinity
 *      NaN             0x7FC00000    0x7E00    Quiet NaN (payload truncated)
 *      1.0             0x3F800000    0x3C00    Normal number
 *      65504           0x477FE000    0x7BFF    Max f16 normal
 *      65520+          >0x477FE000   0x7C00    Overflow → infinity
 *      2⁻¹⁴           0x38800000    0x0400    Min f16 normal
 *      2⁻²⁴           0x33800000    0x0001    Min f16 denormal
 *      <2⁻²⁵          <0x33000000   0x0000    Underflow → zero
 *
 *  https://stackoverflow.com/a/60047308
 *  https://gist.github.com/milhidaka/95863906fe828198f47991c813dbe233
 *  https://github.com/OpenCyphal/libcanard/blob/636795f4bc395f56af8d2c61d3757b5e762bb9e5/canard.c#L811-L834
 */
NK_INTERNAL void nk_f32_to_f16_serial(nk_f32_t const *src, nk_f16_t *dest) {
#if NK_NATIVE_F16
    *dest = (nk_f16_t)(*src);
#else
    nk_fui32_t conv;
    conv.f = *src;

    unsigned int sign = (conv.u >> 31) & 1;
    unsigned int exponent = (conv.u >> 23) & 0xFF;
    unsigned int mantissa = conv.u & 0x007FFFFF;

    unsigned short result;

    if (exponent == 0) {
        // Zero or f32 denormal → f16 zero
        result = (unsigned short)(sign << 15);
    }
    else if (exponent == 255) {
        // Infinity or NaN
        unsigned short payload = (unsigned short)(mantissa >> 13);
        if (mantissa != 0 && payload == 0) payload = 1; // Preserve NaN-ness
        result = (unsigned short)((sign << 15) | 0x7C00 | payload);
    }
    else if (exponent < 103) {
        // Too small for f16 denormal → zero
        result = (unsigned short)(sign << 15);
    }
    else if (exponent < 113) {
        // F16 denormal range
        unsigned int shift = 113 - exponent;
        unsigned int mant = (0x00800000 | mantissa) >> (shift + 13);
        result = (unsigned short)((sign << 15) | mant);
    }
    else if (exponent < 143) {
        // Normal f16 range with rounding
        unsigned int f16_exp = exponent - 112;
        unsigned int f16_mant = mantissa >> 13;
        if (mantissa & 0x1000) { // Round to nearest
            f16_mant++;
            if (f16_mant > 0x3FF) {
                f16_mant = 0;
                f16_exp++;
            }
        }
        if (f16_exp > 30) result = (unsigned short)((sign << 15) | 0x7C00);
        else result = (unsigned short)((sign << 15) | (f16_exp << 10) | f16_mant);
    }
    else {
        // Overflow → infinity
        result = (unsigned short)((sign << 15) | 0x7C00);
    }

    nk_copy_bytes_(dest, &result, 2);
#endif
}

/**
 *  @brief For compilers that don't natively support the `__bf16` type,
 *          upcasts contents into a more conventional `float`.
 *
 *  https://stackoverflow.com/questions/55253233/convert-fp32-to-bfloat16-in-c/55254307#55254307
 *  https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus
 */
NK_INTERNAL void nk_bf16_to_f32_serial(nk_bf16_t const *src, nk_f32_t *dest) {
#if NK_NATIVE_BF16
    *dest = (nk_f32_t)(*src);
#else
    unsigned short x;
    nk_copy_bytes_(&x, src, 2);
    nk_fui32_t conv;
    conv.u = x << 16; // Zero extends the mantissa
    *dest = conv.f;
#endif
}

/**
 *  @brief Compresses a `float` to a `bf16` representation.
 *
 *  https://stackoverflow.com/questions/55253233/convert-fp32-to-bfloat16-in-c/55254307#55254307
 *  https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus
 */
NK_INTERNAL void nk_f32_to_bf16_serial(nk_f32_t const *src, nk_bf16_t *dest) {
#if NK_NATIVE_BF16
    *dest = (nk_bf16_t)(*src);
#else
    nk_fui32_t conv;
    conv.f = *src;
    conv.u += 0x8000; // Rounding is optional
    conv.u >>= 16;
    // Use an intermediate variable to ensure correct behavior on big-endian systems.
    // Copying directly from `&conv.u` would copy the wrong bytes on big-endian,
    // since the lower 16 bits are at offset 2, not offset 0.
    unsigned short result = (unsigned short)conv.u;
    nk_copy_bytes_(dest, &result, 2);
#endif
}

/**
 *  @brief Convert FP8 E4M3 to IEEE 754 single-precision float.
 *
 *  E4M3 (FP8) format: 1 sign bit, 4 exponent bits (bias=7), 3 mantissa bits.
 *  Range: [-448, +448], no ∞, only two NaN encodings (0x7F, 0xFF).
 *  Subnormal values: (-1)ˢ × mantissa × 2⁻⁹ = mantissa / 512.
 *
 *  Special value mappings (E4M3 → F32):
 *      Input        E4M3 Hex  F32 Hex       Description
 *      +0           0x00      0x00000000    Positive zero
 *      -0           0x80      0x80000000    Negative zero
 *      +NaN         0x7F      0x7FC00000    Quiet NaN (exp=15, mant!=0)
 *      -NaN         0xFF      0xFFC00000    Quiet NaN (signed)
 *      +448 (max)   0x7E      0x43E00000    Max normal = 448
 *      -448         0xFE      0xC3E00000    Min normal = -448
 *      1.0          0x38      0x3F800000    Normal (exp=7, mant=0)
 *      Min denorm   0x01      0x3B000000    1/512 = 2⁻⁹
 *      Max denorm   0x07      0x3BE00000    7/512 = 7 × 2⁻⁹
 *
 *  References:
 *      https://arxiv.org/pdf/2209.05433 (NVIDIA/Intel/Arm FP8 paper)
 *      https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-12-01-pdf-1
 *      https://onnx.ai/onnx/technical/float8.html
 */
NK_INTERNAL void nk_e4m3_to_f32_serial(nk_e4m3_t const *src, nk_f32_t *dest) {
    nk_u8_t raw = *src;
    nk_u32_t sign = (nk_u32_t)(raw & 0x80) << 24;
    nk_u32_t exponent = (raw >> 3) & 0x0Fu;
    nk_u32_t mantissa = raw & 0x07u;
    nk_fui32_t conv;

    if (exponent == 0) {
        if (mantissa == 0) {
            conv.u = sign;
            *dest = conv.f;
            return;
        }
        nk_f32_t value = (nk_f32_t)mantissa * (1.0f / 512.0f);
        *dest = sign ? -value : value;
        return;
    }
    // E4M3FN has no ∞. Only exp=15 && mant=7 is NaN.
    // exp=15 && mant=0..6 are normal values (256, 288, 320, 352, 384, 416, 448).
    if (exponent == 0x0Fu && mantissa == 7) {
        conv.u = sign | 0x7FC00000u; // F32 quiet NaN
        *dest = conv.f;
        return;
    }

    nk_u32_t f32_exponent = (exponent + 120u) << 23;
    nk_u32_t f32_mantissa = mantissa << 20;
    conv.u = sign | f32_exponent | f32_mantissa;
    *dest = conv.f;
}

/**
 *  @brief Convert IEEE 754 single-precision float to FP8 E4M3.
 *
 *  E4M3 (FP8) format: 1 sign bit, 4 exponent bits (bias=7), 3 mantissa bits.
 *  Range: [-448, +448], no ∞, only two NaN encodings.
 *  Rounding: RNE (Round to Nearest Even) per IEEE 754 / OCP FP8 spec.
 *  Subnormal threshold: values with |x| < 2⁻⁶ use subnormal encoding.
 *
 *  Special value mappings (F32 → E4M3):
 *      Input        F32 Hex       E4M3 Hex  Description
 *      +0           0x00000000    0x00      Positive zero
 *      -0           0x80000000    0x80      Negative zero
 *      +inf         0x7F800000    0x7E      Saturates to max (+448)
 *      -inf         0xFF800000    0xFE      Saturates to min (-448)
 *      NaN          0x7FC00000    0x7F      Quiet NaN
 *      1.0          0x3F800000    0x38      Normal (exp=7, mant=0)
 *      448+         >0x43E00000   0x7E      Overflow → max
 *      2⁻⁶         0x3E800000    0x08      Min normal
 *      <2⁻¹² × ⁵     <0x39800000   0x00      Underflow → zero (RNE boundary)
 *
 *  References:
 *      https://arxiv.org/pdf/2209.05433 (NVIDIA/Intel/Arm FP8 paper)
 *      https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-12-01-pdf-1
 *      https://onnx.ai/onnx/technical/float8.html
 */
NK_INTERNAL void nk_f32_to_e4m3_serial(nk_f32_t const *src, nk_e4m3_t *dest) {
    nk_f32_t x = *src;
    nk_fui32_t conv;
    conv.f = x;
    nk_u32_t sign_bit = conv.u >> 31;
    nk_u32_t abs_bits = conv.u & 0x7FFFFFFFu;
    nk_u8_t sign = (nk_u8_t)(sign_bit << 7);

    // NaN → E4M3FN NaN (0x7F or 0xFF)
    if (abs_bits > 0x7F800000u) {
        *dest = (nk_e4m3_t)(sign | 0x7Fu);
        return;
    }
    // Infinity → saturate to max (0x7E or 0xFE), E4M3FN has no ∞
    if (abs_bits == 0x7F800000u) {
        *dest = (nk_e4m3_t)(sign | 0x7Eu);
        return;
    }

    if (abs_bits == 0) {
        *dest = (nk_e4m3_t)sign;
        return;
    }

    nk_f32_t abs_x = sign_bit ? -x : x;

    // Subnormal range: [0, 1/64). Use RNE rounding via scaled * 512.
    // The RNE boundary between 0 and 1/512 is at 0.5/512, not 1/512.
    if (abs_x < (1.0f / 64.0f)) {
        nk_f32_t scaled = abs_x * 512.0f;
        nk_i32_t mant = (nk_i32_t)scaled;
        nk_f32_t frac = scaled - (nk_f32_t)mant;
        if (frac > 0.5f || (frac == 0.5f && (mant & 1))) { ++mant; }
        // If rounds to 8, promote to first normal (exp_field=1, mantissa=0)
        if (mant > 7) {
            *dest = (nk_e4m3_t)(sign | 0x08u);
            return;
        }
        if (mant == 0) { *dest = (nk_e4m3_t)sign; }
        else { *dest = (nk_e4m3_t)(sign | (nk_u8_t)mant); }
        return;
    }

    nk_i32_t exp = (nk_i32_t)((abs_bits >> 23) & 0xFFu) - 127;
    nk_u32_t mantissa = abs_bits & 0x7FFFFFu;
    nk_u32_t significand = (1u << 23) | mantissa;
    nk_i32_t shift = 23 - 3;
    nk_u32_t remainder_mask = (1u << shift) - 1;
    nk_u32_t remainder = significand & remainder_mask;
    nk_u32_t halfway = 1u << (shift - 1);
    nk_u32_t significand_rounded = significand >> shift;
    if (remainder > halfway || (remainder == halfway && (significand_rounded & 1))) { ++significand_rounded; }
    if (significand_rounded == (1u << (3 + 1))) {
        significand_rounded >>= 1;
        ++exp;
    }
    if (exp > 8) {
        // Saturate to max value 448 = 0x7E (exp=15, mantissa=6). Note: 0x7F is NaN in e4m3FN.
        *dest = (nk_e4m3_t)(sign | 0x7Eu);
        return;
    }
    if (exp < -6) {
        nk_f32_t scaled = abs_x * 512.0f;
        nk_i32_t mant = (nk_i32_t)scaled;
        nk_f32_t frac = scaled - (nk_f32_t)mant;
        if (frac > 0.5f || (frac == 0.5f && (mant & 1))) { ++mant; }
        // If rounds to 8, promote to first normal (exp_field=1, mantissa=0)
        if (mant > 7) {
            *dest = (nk_e4m3_t)(sign | 0x08u);
            return;
        }
        if (mant == 0) { *dest = (nk_e4m3_t)sign; }
        else { *dest = (nk_e4m3_t)(sign | (nk_u8_t)mant); }
        return;
    }

    nk_u8_t exp_field = (nk_u8_t)(exp + 7);
    nk_u8_t mant_field = (nk_u8_t)(significand_rounded & 0x07u);
    // For exp_field=15, clamp mantissa to 6 to avoid NaN encoding (0x7F in e4m3FN)
    if (exp_field == 15 && mant_field > 6) { mant_field = 6; }
    *dest = (nk_e4m3_t)(sign | (exp_field << 3) | mant_field);
}

/**
 *  @brief Convert FP8 E5M2 to IEEE 754 single-precision float.
 *
 *  E5M2 (FP8) format: 1 sign bit, 5 exponent bits (bias=15), 2 mantissa bits.
 *  Range: [-57344, +57344], supports infinity and NaN (IEEE 754 compatible).
 *  Subnormal values: (-1)ˢ × mantissa × 2⁻¹⁶ = mantissa / 65536.
 *
 *  Special value mappings (E5M2 → F32):
 *      Input        E5M2 Hex  F32 Hex       Description
 *      +0           0x00      0x00000000    Positive zero
 *      -0           0x80      0x80000000    Negative zero
 *      +inf         0x7C      0x7F800000    Positive infinity
 *      -inf         0xFC      0xFF800000    Negative infinity
 *      +NaN         0x7D-7F   0x7FC00000    Quiet NaN (exp=31, mant!=0)
 *      -NaN         0xFD-FF   0xFFC00000    Quiet NaN (signed)
 *      +57344 (max) 0x7B      0x47600000    Max normal
 *      1.0          0x3C      0x3F800000    Normal (exp=15, mant=0)
 *      Min denorm   0x01      0x37800000    1/65536 = 2⁻¹⁶
 *      Max denorm   0x03      0x38000000    3/65536 = 3 × 2⁻¹⁶
 *
 *  References:
 *      https://arxiv.org/pdf/2209.05433 (NVIDIA/Intel/Arm FP8 paper)
 *      https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-12-01-pdf-1
 *      https://onnx.ai/onnx/technical/float8.html
 */
NK_INTERNAL void nk_e5m2_to_f32_serial(nk_e5m2_t const *src, nk_f32_t *dest) {
    nk_u8_t raw = *src;
    nk_u32_t sign = (nk_u32_t)(raw & 0x80) << 24;
    nk_u32_t exponent = (raw >> 2) & 0x1Fu;
    nk_u32_t mantissa = raw & 0x03u;
    nk_fui32_t conv;

    if (exponent == 0) {
        if (mantissa == 0) {
            conv.u = sign;
            *dest = conv.f;
            return;
        }
        nk_f32_t value = (nk_f32_t)mantissa * (1.0f / 65536.0f);
        *dest = sign ? -value : value;
        return;
    }
    if (exponent == 0x1Fu) {
        if (mantissa == 0) { conv.u = sign | 0x7F800000u; }
        else { conv.u = sign | 0x7FC00000u; }
        *dest = conv.f;
        return;
    }

    nk_u32_t f32_exponent = (exponent + 112u) << 23;
    nk_u32_t f32_mantissa = mantissa << 21;
    conv.u = sign | f32_exponent | f32_mantissa;
    *dest = conv.f;
}

/**
 *  @brief Convert IEEE 754 single-precision float to FP8 E5M2.
 *
 *  E5M2 (FP8) format: 1 sign bit, 5 exponent bits (bias=15), 2 mantissa bits.
 *  Range: [-57344, +57344], supports infinity and NaN (IEEE 754 compatible).
 *  Rounding: RNE (Round to Nearest Even) per IEEE 754 / OCP FP8 spec.
 *  Subnormal threshold: values with |x| < 2⁻¹⁴ use subnormal encoding.
 *
 *  Special value mappings (F32 → E5M2):
 *      Input        F32 Hex       E5M2 Hex  Description
 *      +0           0x00000000    0x00      Positive zero
 *      -0           0x80000000    0x80      Negative zero
 *      +inf         0x7F800000    0x7C      Positive infinity
 *      -inf         0xFF800000    0xFC      Negative infinity
 *      NaN          0x7FC00000    0x7D      Quiet NaN
 *      1.0          0x3F800000    0x3C      Normal (exp=15, mant=0)
 *      57344+       >0x47600000   0x7C      Overflow → infinity
 *      2⁻¹⁴        0x38800000    0x04      Min normal
 *      <2⁻¹⁷ × ⁵     <0x36800000   0x00      Underflow → zero (RNE boundary)
 *
 *  References:
 *      https://arxiv.org/pdf/2209.05433 (NVIDIA/Intel/Arm FP8 paper)
 *      https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-12-01-pdf-1
 *      https://onnx.ai/onnx/technical/float8.html
 */
NK_INTERNAL void nk_f32_to_e5m2_serial(nk_f32_t const *src, nk_e5m2_t *dest) {
    nk_f32_t x = *src;
    nk_fui32_t conv;
    conv.f = x;
    nk_u32_t sign_bit = conv.u >> 31;
    nk_u32_t abs_bits = conv.u & 0x7FFFFFFFu;
    nk_u8_t sign = (nk_u8_t)(sign_bit << 7);

    if (abs_bits >= 0x7F800000u) {
        nk_u8_t mant = (abs_bits > 0x7F800000u) ? 0x01u : 0x00u;
        *dest = (nk_e5m2_t)(sign | 0x7Cu | mant);
        return;
    }

    if (abs_bits == 0) {
        *dest = (nk_e5m2_t)sign;
        return;
    }

    nk_f32_t abs_x = sign_bit ? -x : x;

    // Subnormal range: [0, 1/16384). Use RNE rounding via scaled * 65536.
    // The RNE boundary between 0 and 1/65536 is at 0.5/65536, not 1/65536.
    if (abs_x < (1.0f / 16384.0f)) {
        nk_f32_t scaled = abs_x * 65536.0f;
        nk_i32_t mant = (nk_i32_t)scaled;
        nk_f32_t frac = scaled - (nk_f32_t)mant;
        if (frac > 0.5f || (frac == 0.5f && (mant & 1))) { ++mant; }
        // If rounds to 4, promote to first normal (exp_field=1, mantissa=0)
        if (mant > 3) {
            *dest = (nk_e5m2_t)(sign | 0x04u);
            return;
        }
        if (mant == 0) { *dest = (nk_e5m2_t)sign; }
        else { *dest = (nk_e5m2_t)(sign | (nk_u8_t)mant); }
        return;
    }

    nk_i32_t exp = (nk_i32_t)((abs_bits >> 23) & 0xFFu) - 127;
    nk_u32_t mantissa = abs_bits & 0x7FFFFFu;
    nk_u32_t significand = (1u << 23) | mantissa;
    nk_i32_t shift = 23 - 2;
    nk_u32_t remainder_mask = (1u << shift) - 1;
    nk_u32_t remainder = significand & remainder_mask;
    nk_u32_t halfway = 1u << (shift - 1);
    nk_u32_t significand_rounded = significand >> shift;
    if (remainder > halfway || (remainder == halfway && (significand_rounded & 1))) { ++significand_rounded; }
    if (significand_rounded == (1u << (2 + 1))) {
        significand_rounded >>= 1;
        ++exp;
    }
    if (exp > 15) {
        *dest = (nk_e5m2_t)(sign | 0x7Cu);
        return;
    }
    if (exp < -14) {
        nk_f32_t scaled = abs_x * 65536.0f;
        nk_i32_t mant = (nk_i32_t)scaled;
        nk_f32_t frac = scaled - (nk_f32_t)mant;
        if (frac > 0.5f || (frac == 0.5f && (mant & 1))) { ++mant; }
        // If rounds to 4, promote to first normal (exp_field=1, mantissa=0)
        if (mant > 3) {
            *dest = (nk_e5m2_t)(sign | 0x04u);
            return;
        }
        if (mant == 0) { *dest = (nk_e5m2_t)sign; }
        else { *dest = (nk_e5m2_t)(sign | (nk_u8_t)mant); }
        return;
    }

    nk_u8_t exp_field = (nk_u8_t)(exp + 15);
    nk_u8_t mant_field = (nk_u8_t)(significand_rounded & 0x03u);
    *dest = (nk_e5m2_t)(sign | (exp_field << 2) | mant_field);
}

NK_INTERNAL void nk_f16_to_f64_serial(nk_f16_t const *x, nk_f64_t *y) {
    nk_f32_t f32;
    nk_f16_to_f32_serial(x, &f32);
    *y = (nk_f64_t)f32;
}
NK_INTERNAL void nk_f64_to_f16_serial(nk_f64_t const *x, nk_f16_t *y) {
    nk_f32_t f32 = (nk_f32_t)*x;
    nk_f32_to_f16_serial(&f32, y);
}
NK_INTERNAL void nk_bf16_to_f64_serial(nk_bf16_t const *x, nk_f64_t *y) {
    nk_f32_t f32;
    nk_bf16_to_f32_serial(x, &f32);
    *y = (nk_f64_t)f32;
}
NK_INTERNAL void nk_f64_to_bf16_serial(nk_f64_t const *x, nk_bf16_t *y) {
    nk_f32_t f32 = (nk_f32_t)*x;
    nk_f32_to_bf16_serial(&f32, y);
}

/*  Convert floating pointer numbers to integers, clamping them to the range of signed
 *  and unsigned low-resolution integers, and rounding them to the nearest integer.
 *
 *  In C++ the analogous solution with STL could be: `*y = std::clamp(std::round(*x), -128, 127)`.
 *  In C, using the standard library: `*x = fminf(fmaxf(roundf(*x), -128), 127)`.
 */
NK_INTERNAL void nk_f32_to_i8_serial(nk_f32_t const *x, nk_i8_t *y) {
    *y = (nk_i8_t)(*x > 127 ? 127 : (*x < -128 ? -128 : (int)(*x + (*x < 0 ? -0.5f : 0.5f))));
}

NK_INTERNAL void nk_f32_to_u8_serial(nk_f32_t const *x, nk_u8_t *y) {
    *y = (nk_u8_t)(*x > 255 ? 255 : (*x < 0 ? 0 : (int)(*x + (*x < 0 ? -0.5f : 0.5f))));
}

NK_INTERNAL void nk_f32_to_i16_serial(nk_f32_t const *x, nk_i16_t *y) {
    *y = (nk_i16_t)(*x > 32767 ? 32767 : (*x < -32768 ? -32768 : (int)(*x + (*x < 0 ? -0.5 : 0.5))));
}

NK_INTERNAL void nk_f32_to_u16_serial(nk_f32_t const *x, nk_u16_t *y) {
    *y = (nk_u16_t)(*x > 65535 ? 65535 : (*x < 0 ? 0 : (int)(*x + (*x < 0 ? -0.5 : 0.5))));
}

NK_INTERNAL void nk_f64_to_i8_serial(nk_f64_t const *x, nk_i8_t *y) {
    *y = (nk_i8_t)(*x > 127 ? 127 : (*x < -128 ? -128 : (int)(*x + (*x < 0 ? -0.5 : 0.5))));
}

NK_INTERNAL void nk_f64_to_u8_serial(nk_f64_t const *x, nk_u8_t *y) {
    *y = (nk_u8_t)(*x > 255 ? 255 : (*x < 0 ? 0 : (int)(*x + (*x < 0 ? -0.5 : 0.5))));
}

NK_INTERNAL void nk_f64_to_i16_serial(nk_f64_t const *x, nk_i16_t *y) {
    *y = (nk_i16_t)(*x > 32767 ? 32767 : (*x < -32768 ? -32768 : (int)(*x + (*x < 0 ? -0.5 : 0.5))));
}

NK_INTERNAL void nk_f64_to_u16_serial(nk_f64_t const *x, nk_u16_t *y) {
    *y = (nk_u16_t)(*x > 65535 ? 65535 : (*x < 0 ? 0 : (int)(*x + (*x < 0 ? -0.5 : 0.5))));
}

NK_INTERNAL void nk_f64_to_i32_serial(nk_f64_t const *x, nk_i32_t *y) {
    *y = (nk_i32_t)(*x > 2147483647 ? 2147483647
                                    : (*x < -2147483648 ? -2147483648 : (int)(*x + (*x < 0 ? -0.5 : 0.5))));
}

NK_INTERNAL void nk_f64_to_u32_serial(nk_f64_t const *x, nk_u32_t *y) {
    *y = (nk_u32_t)(*x > 4294967295 ? 4294967295 : (*x < 0 ? 0 : (unsigned int)(*x + (*x < 0 ? -0.5 : 0.5))));
}

NK_INTERNAL void nk_f64_to_i64_serial(nk_f64_t const *x, nk_i64_t *y) {
    *y = (nk_i64_t)(*x > 9223372036854775807.0
                        ? 9223372036854775807ll
                        : (*x < -9223372036854775808.0 ? (-9223372036854775807ll - 1ll)
                                                       : (long long)(*x + (*x < 0 ? -0.5 : 0.5))));
}

NK_INTERNAL void nk_f64_to_u64_serial(nk_f64_t const *x, nk_u64_t *y) {
    *y = (nk_u64_t)(*x > 18446744073709551615.0 ? 18446744073709551615ull
                                                : (*x < 0 ? 0 : (unsigned long long)(*x + (*x < 0 ? -0.5 : 0.5))));
}

NK_INTERNAL void nk_i64_to_i8_serial(nk_i64_t const *x, nk_i8_t *y) {
    *y = (nk_i8_t)(*x > 127ll ? 127ll : (*x < -128ll ? -128ll : *x));
}

NK_INTERNAL void nk_i64_to_u8_serial(nk_i64_t const *x, nk_u8_t *y) {
    *y = (nk_u8_t)(*x > 255ll ? 255ll : (*x < 0ll ? 0ll : *x));
}

NK_INTERNAL void nk_i64_to_i16_serial(nk_i64_t const *x, nk_i16_t *y) {
    *y = (nk_i16_t)(*x > 32767ll ? 32767ll : (*x < -32768ll ? -32768ll : *x));
}

NK_INTERNAL void nk_i64_to_u16_serial(nk_i64_t const *x, nk_u16_t *y) {
    *y = (nk_u16_t)(*x > 65535ll ? 65535ll : (*x < 0ll ? 0ll : *x));
}

NK_INTERNAL void nk_i64_to_i32_serial(nk_i64_t const *x, nk_i32_t *y) {
    *y = (nk_i32_t)(*x > 2147483647ll ? 2147483647ll : (*x < -2147483648ll ? -2147483648ll : *x));
}

NK_INTERNAL void nk_i64_to_u32_serial(nk_i64_t const *x, nk_u32_t *y) {
    *y = (nk_u32_t)(*x > 4294967295ll ? 4294967295ll : (*x < 0ll ? 0ll : *x));
}

NK_INTERNAL void nk_u64_to_i8_serial(nk_u64_t const *x, nk_i8_t *y) { *y = (nk_i8_t)(*x > 127ull ? 127ull : *x); }
NK_INTERNAL void nk_u64_to_u8_serial(nk_u64_t const *x, nk_u8_t *y) { *y = (nk_u8_t)(*x > 255ull ? 255ull : *x); }
NK_INTERNAL void nk_u64_to_i16_serial(nk_u64_t const *x, nk_i16_t *y) {
    *y = (nk_i16_t)(*x > 32767ull ? 32767ull : *x);
}
NK_INTERNAL void nk_u64_to_u16_serial(nk_u64_t const *x, nk_u16_t *y) {
    *y = (nk_u16_t)(*x > 65535ull ? 65535ull : *x);
}

NK_INTERNAL void nk_u64_to_i32_serial(nk_u64_t const *x, nk_i32_t *y) {
    *y = (nk_i32_t)(*x > 2147483647ull ? 2147483647ull : *x);
}

NK_INTERNAL void nk_u64_to_u32_serial(nk_u64_t const *x, nk_u32_t *y) {
    *y = (nk_u32_t)(*x > 4294967295ull ? 4294967295ull : *x);
}

NK_PUBLIC void nk_f16_to_f64_(nk_f16_t const *src, nk_f64_t *dest) {
    nk_f32_t f32;
    nk_f16_to_f32_serial(src, &f32);
    *dest = f32;
}
NK_PUBLIC void nk_bf16_to_f64_(nk_bf16_t const *src, nk_f64_t *dest) {
    nk_f32_t f32;
    nk_bf16_to_f32_serial(src, &f32);
    *dest = f32;
}

NK_INTERNAL void nk_u64_to_i64_serial(nk_u64_t const *x, nk_i64_t *y) {
    *y = (nk_i64_t)(*x >= 9223372036854775807ull ? 9223372036854775807ll : *x);
}

NK_INTERNAL void nk_i8_to_u64_serial(nk_i8_t const *x, nk_u64_t *y) { *y = (nk_u64_t)(*x < 0 ? 0 : *x); }
NK_INTERNAL void nk_i16_to_u64_serial(nk_i16_t const *x, nk_u64_t *y) { *y = (nk_u64_t)(*x < 0 ? 0 : *x); }
NK_INTERNAL void nk_i32_to_u64_serial(nk_i32_t const *x, nk_u64_t *y) { *y = (nk_u64_t)(*x < 0 ? 0 : *x); }
NK_INTERNAL void nk_i64_to_u64_serial(nk_i64_t const *x, nk_u64_t *y) { *y = (nk_u64_t)(*x < 0 ? 0 : *x); }

NK_INTERNAL void nk_i64_to_f16_serial(nk_i64_t const *x, nk_f16_t *y) {
    nk_f32_t f32 = (nk_f32_t)*x;
    nk_f32_to_f16_serial(&f32, y);
}
NK_INTERNAL void nk_i64_to_bf16_serial(nk_i64_t const *x, nk_bf16_t *y) {
    nk_f32_t f32 = (nk_f32_t)*x;
    nk_f32_to_bf16_serial(&f32, y);
}
NK_INTERNAL void nk_u64_to_f16_serial(nk_u64_t const *x, nk_f16_t *y) {
    nk_f32_t f32 = (nk_f32_t)*x;
    nk_f32_to_f16_serial(&f32, y);
}
NK_INTERNAL void nk_u64_to_bf16_serial(nk_u64_t const *x, nk_bf16_t *y) {
    nk_f32_t f32 = (nk_f32_t)*x;
    nk_f32_to_bf16_serial(&f32, y);
}

#pragma region - Type Punned Loads and Stores

/** @brief Type-agnostic 256-bit full load. */
NK_INTERNAL void nk_load_b256_serial_(void const *src, nk_b256_vec_t *dst) {
    dst->u64s[0] = 0, dst->u64s[1] = 0, dst->u64s[2] = 0, dst->u64s[3] = 0;
}

/** @brief Type-agnostic 128-bit full load. */
NK_INTERNAL void nk_load_b128_serial_(void const *src, nk_b128_vec_t *dst) { dst->u64s[0] = 0, dst->u64s[1] = 0; }

/** @brief Type-agnostic 64-bit full load. */
NK_INTERNAL void nk_load_b64_serial_(void const *src, nk_b64_vec_t *dst) { dst->u64 = *(nk_u64_t const *)src; }

/** @brief Type-agnostic partial load for 32-bit elements (8 elements max) into 256-bit vector. */
NK_INTERNAL void nk_partial_load_b32x8_serial_(void const *src, nk_b256_vec_t *dst, nk_size_t n) {
    dst->u64s[0] = 0, dst->u64s[1] = 0, dst->u64s[2] = 0, dst->u64s[3] = 0;
    nk_u32_t const *s = (nk_u32_t const *)src;
    switch (n) {
    default:
    case 8: dst->u32s[7] = s[7]; // fallthrough
    case 7: dst->u32s[6] = s[6]; // fallthrough
    case 6: dst->u32s[5] = s[5]; // fallthrough
    case 5: dst->u32s[4] = s[4]; // fallthrough
    case 4: dst->u32s[3] = s[3]; // fallthrough
    case 3: dst->u32s[2] = s[2]; // fallthrough
    case 2: dst->u32s[1] = s[1]; // fallthrough
    case 1: dst->u32s[0] = s[0]; // fallthrough
    case 0: break;
    }
}

/** @brief Type-agnostic partial load for 32-bit elements (4 elements max) into 128-bit vector. */
NK_INTERNAL void nk_partial_load_b32x4_serial_(void const *src, nk_b128_vec_t *dst, nk_size_t n) {
    dst->u64s[0] = 0, dst->u64s[1] = 0;
    nk_u32_t const *s = (nk_u32_t const *)src;
    switch (n) {
    default:
    case 4: dst->u32s[3] = s[3]; // fallthrough
    case 3: dst->u32s[2] = s[2]; // fallthrough
    case 2: dst->u32s[1] = s[1]; // fallthrough
    case 1: dst->u32s[0] = s[0]; // fallthrough
    case 0: break;
    }
}

/** @brief Type-agnostic partial load for 8-bit elements (8 elements max) into 64-bit vector. */
NK_INTERNAL void nk_partial_load_b8x8_serial_(void const *src, nk_b64_vec_t *dst, nk_size_t n) {
    dst->u64 = 0;
    nk_u8_t const *s = (nk_u8_t const *)src;
    switch (n) {
    default:
    case 8: dst->u8s[7] = s[7]; // fallthrough
    case 7: dst->u8s[6] = s[6]; // fallthrough
    case 6: dst->u8s[5] = s[5]; // fallthrough
    case 5: dst->u8s[4] = s[4]; // fallthrough
    case 4: dst->u8s[3] = s[3]; // fallthrough
    case 3: dst->u8s[2] = s[2]; // fallthrough
    case 2: dst->u8s[1] = s[1]; // fallthrough
    case 1: dst->u8s[0] = s[0]; // fallthrough
    case 0: break;
    }
}

/** @brief Type-agnostic partial load for 8-bit elements (4 elements max) into 32-bit vector. */
NK_INTERNAL nk_b32_vec_t nk_partial_load_b8x4_serial_(void const *src, nk_size_t n) {
    nk_b32_vec_t dst = {0};
    nk_u8_t const *s = (nk_u8_t const *)src;
    switch (n) {
    default:
    case 4: dst.u8s[3] = s[3]; // fallthrough
    case 3: dst.u8s[2] = s[2]; // fallthrough
    case 2: dst.u8s[1] = s[1]; // fallthrough
    case 1: dst.u8s[0] = s[0]; // fallthrough
    case 0: break;
    }
    return dst;
}

/** @brief Partial store for 8-bit elements (up to 4) from nk_b32_vec_t. */
NK_INTERNAL void nk_partial_store_b8x4_serial_(nk_b32_vec_t const *src, void *dst, nk_size_t n) {
    nk_u8_t *d = (nk_u8_t *)dst;
    switch (n) {
    default:
    case 4: d[3] = src->u8s[3]; // fallthrough
    case 3: d[2] = src->u8s[2]; // fallthrough
    case 2: d[1] = src->u8s[1]; // fallthrough
    case 1: d[0] = src->u8s[0]; // fallthrough
    case 0: break;
    }
}

/** @brief Type-agnostic partial load for 16-bit elements (8 elements max) into 128-bit vector. */
NK_INTERNAL void nk_partial_load_b16x8_serial_(void const *src, nk_b128_vec_t *dst, nk_size_t n) {
    dst->u64s[0] = 0, dst->u64s[1] = 0;
    nk_u16_t const *s = (nk_u16_t const *)src;
    switch (n) {
    default:
    case 8: dst->u16s[7] = s[7]; // fallthrough
    case 7: dst->u16s[6] = s[6]; // fallthrough
    case 6: dst->u16s[5] = s[5]; // fallthrough
    case 5: dst->u16s[4] = s[4]; // fallthrough
    case 4: dst->u16s[3] = s[3]; // fallthrough
    case 3: dst->u16s[2] = s[2]; // fallthrough
    case 2: dst->u16s[1] = s[1]; // fallthrough
    case 1: dst->u16s[0] = s[0]; // fallthrough
    case 0: break;
    }
}

/** @brief Type-agnostic partial load for 8-bit elements (16 elements max) into 128-bit vector. */
NK_INTERNAL void nk_partial_load_b8x16_serial_(void const *src, nk_b128_vec_t *dst, nk_size_t n) {
    dst->u64s[0] = 0, dst->u64s[1] = 0;
    nk_u8_t const *s = (nk_u8_t const *)src;
    switch (n) {
    default:
    case 16: dst->u8s[15] = s[15]; // fallthrough
    case 15: dst->u8s[14] = s[14]; // fallthrough
    case 14: dst->u8s[13] = s[13]; // fallthrough
    case 13: dst->u8s[12] = s[12]; // fallthrough
    case 12: dst->u8s[11] = s[11]; // fallthrough
    case 11: dst->u8s[10] = s[10]; // fallthrough
    case 10: dst->u8s[9] = s[9];   // fallthrough
    case 9: dst->u8s[8] = s[8];    // fallthrough
    case 8: dst->u8s[7] = s[7];    // fallthrough
    case 7: dst->u8s[6] = s[6];    // fallthrough
    case 6: dst->u8s[5] = s[5];    // fallthrough
    case 5: dst->u8s[4] = s[4];    // fallthrough
    case 4: dst->u8s[3] = s[3];    // fallthrough
    case 3: dst->u8s[2] = s[2];    // fallthrough
    case 2: dst->u8s[1] = s[1];    // fallthrough
    case 1: dst->u8s[0] = s[0];    // fallthrough
    case 0: break;
    }
}

/** @brief Type-agnostic partial load for 16-bit elements (16 elements max) into 256-bit vector. */
NK_INTERNAL void nk_partial_load_b16x16_serial_(void const *src, nk_b256_vec_t *dst, nk_size_t n) {
    dst->u64s[0] = 0, dst->u64s[1] = 0, dst->u64s[2] = 0, dst->u64s[3] = 0;
    nk_u16_t const *s = (nk_u16_t const *)src;
    switch (n) {
    default:
    case 16: dst->u16s[15] = s[15]; // fallthrough
    case 15: dst->u16s[14] = s[14]; // fallthrough
    case 14: dst->u16s[13] = s[13]; // fallthrough
    case 13: dst->u16s[12] = s[12]; // fallthrough
    case 12: dst->u16s[11] = s[11]; // fallthrough
    case 11: dst->u16s[10] = s[10]; // fallthrough
    case 10: dst->u16s[9] = s[9];   // fallthrough
    case 9: dst->u16s[8] = s[8];    // fallthrough
    case 8: dst->u16s[7] = s[7];    // fallthrough
    case 7: dst->u16s[6] = s[6];    // fallthrough
    case 6: dst->u16s[5] = s[5];    // fallthrough
    case 5: dst->u16s[4] = s[4];    // fallthrough
    case 4: dst->u16s[3] = s[3];    // fallthrough
    case 3: dst->u16s[2] = s[2];    // fallthrough
    case 2: dst->u16s[1] = s[1];    // fallthrough
    case 1: dst->u16s[0] = s[0];    // fallthrough
    case 0: break;
    }
}

/** @brief Partial load for 8-bit elements (32 max) into 256-bit vector (zeros in remaining slots). */
NK_INTERNAL void nk_partial_load_b8x32_serial_(void const *src, nk_b256_vec_t *dst, nk_size_t n) {
    dst->u64s[0] = 0, dst->u64s[1] = 0, dst->u64s[2] = 0, dst->u64s[3] = 0;
    nk_u8_t const *s = (nk_u8_t const *)src;
    switch (n) {
    default:
    case 32: dst->u8s[31] = s[31]; // fallthrough
    case 31: dst->u8s[30] = s[30]; // fallthrough
    case 30: dst->u8s[29] = s[29]; // fallthrough
    case 29: dst->u8s[28] = s[28]; // fallthrough
    case 28: dst->u8s[27] = s[27]; // fallthrough
    case 27: dst->u8s[26] = s[26]; // fallthrough
    case 26: dst->u8s[25] = s[25]; // fallthrough
    case 25: dst->u8s[24] = s[24]; // fallthrough
    case 24: dst->u8s[23] = s[23]; // fallthrough
    case 23: dst->u8s[22] = s[22]; // fallthrough
    case 22: dst->u8s[21] = s[21]; // fallthrough
    case 21: dst->u8s[20] = s[20]; // fallthrough
    case 20: dst->u8s[19] = s[19]; // fallthrough
    case 19: dst->u8s[18] = s[18]; // fallthrough
    case 18: dst->u8s[17] = s[17]; // fallthrough
    case 17: dst->u8s[16] = s[16]; // fallthrough
    case 16: dst->u8s[15] = s[15]; // fallthrough
    case 15: dst->u8s[14] = s[14]; // fallthrough
    case 14: dst->u8s[13] = s[13]; // fallthrough
    case 13: dst->u8s[12] = s[12]; // fallthrough
    case 12: dst->u8s[11] = s[11]; // fallthrough
    case 11: dst->u8s[10] = s[10]; // fallthrough
    case 10: dst->u8s[9] = s[9];   // fallthrough
    case 9: dst->u8s[8] = s[8];    // fallthrough
    case 8: dst->u8s[7] = s[7];    // fallthrough
    case 7: dst->u8s[6] = s[6];    // fallthrough
    case 6: dst->u8s[5] = s[5];    // fallthrough
    case 5: dst->u8s[4] = s[4];    // fallthrough
    case 4: dst->u8s[3] = s[3];    // fallthrough
    case 3: dst->u8s[2] = s[2];    // fallthrough
    case 2: dst->u8s[1] = s[1];    // fallthrough
    case 1: dst->u8s[0] = s[0];    // fallthrough
    case 0: break;
    }
}

/** @brief Type-agnostic partial store for 32-bit elements (8 elements max) from 256-bit vector. */
NK_INTERNAL void nk_partial_store_b32x8_serial_(nk_b256_vec_t const *src, void *dst, nk_size_t n) {
    nk_u32_t *d = (nk_u32_t *)dst;
    switch (n) {
    default:
    case 8: d[7] = src->u32s[7]; // fallthrough
    case 7: d[6] = src->u32s[6]; // fallthrough
    case 6: d[5] = src->u32s[5]; // fallthrough
    case 5: d[4] = src->u32s[4]; // fallthrough
    case 4: d[3] = src->u32s[3]; // fallthrough
    case 3: d[2] = src->u32s[2]; // fallthrough
    case 2: d[1] = src->u32s[1]; // fallthrough
    case 1: d[0] = src->u32s[0]; // fallthrough
    case 0: break;
    }
}

/** @brief Type-agnostic partial store for 32-bit elements (4 elements max) from 128-bit vector. */
NK_INTERNAL void nk_partial_store_b32x4_serial_(nk_b128_vec_t const *src, void *dst, nk_size_t n) {
    nk_u32_t *d = (nk_u32_t *)dst;
    switch (n) {
    default:
    case 4: d[3] = src->u32s[3]; // fallthrough
    case 3: d[2] = src->u32s[2]; // fallthrough
    case 2: d[1] = src->u32s[1]; // fallthrough
    case 1: d[0] = src->u32s[0]; // fallthrough
    case 0: break;
    }
}

/** @brief Type-agnostic partial store for 16-bit elements (8 elements max) from 128-bit vector. */
NK_INTERNAL void nk_partial_store_b16x8_serial_(nk_b128_vec_t const *src, void *dst, nk_size_t n) {
    nk_u16_t *d = (nk_u16_t *)dst;
    switch (n) {
    default:
    case 8: d[7] = src->u16s[7]; // fallthrough
    case 7: d[6] = src->u16s[6]; // fallthrough
    case 6: d[5] = src->u16s[5]; // fallthrough
    case 5: d[4] = src->u16s[4]; // fallthrough
    case 4: d[3] = src->u16s[3]; // fallthrough
    case 3: d[2] = src->u16s[2]; // fallthrough
    case 2: d[1] = src->u16s[1]; // fallthrough
    case 1: d[0] = src->u16s[0]; // fallthrough
    case 0: break;
    }
}

/** @brief Type-agnostic partial store for 8-bit elements (8 elements max) from 64-bit vector. */
NK_INTERNAL void nk_partial_store_b8x8_serial_(nk_b64_vec_t const *src, void *dst, nk_size_t n) {
    nk_u8_t *d = (nk_u8_t *)dst;
    switch (n) {
    default:
    case 8: d[7] = src->u8s[7]; // fallthrough
    case 7: d[6] = src->u8s[6]; // fallthrough
    case 6: d[5] = src->u8s[5]; // fallthrough
    case 5: d[4] = src->u8s[4]; // fallthrough
    case 4: d[3] = src->u8s[3]; // fallthrough
    case 3: d[2] = src->u8s[2]; // fallthrough
    case 2: d[1] = src->u8s[1]; // fallthrough
    case 1: d[0] = src->u8s[0]; // fallthrough
    case 0: break;
    }
}

/** @brief Type-agnostic partial load for 64-bit elements (4 elements max) into 256-bit vector. */
NK_INTERNAL void nk_partial_load_b64x4_serial_(void const *src, nk_b256_vec_t *dst, nk_size_t n) {
    nk_u64_t const *s = (nk_u64_t const *)src;
    dst->u64s[0] = 0, dst->u64s[1] = 0, dst->u64s[2] = 0, dst->u64s[3] = 0;
    switch (n) {
    default:
    case 4: dst->u64s[3] = s[3]; // fallthrough
    case 3: dst->u64s[2] = s[2]; // fallthrough
    case 2: dst->u64s[1] = s[1]; // fallthrough
    case 1: dst->u64s[0] = s[0]; // fallthrough
    case 0: break;
    }
}

/** @brief Type-agnostic partial store for 64-bit elements (4 elements max) from 256-bit vector. */
NK_INTERNAL void nk_partial_store_b64x4_serial_(nk_b256_vec_t const *src, void *dst, nk_size_t n) {
    nk_u64_t *d = (nk_u64_t *)dst;
    switch (n) {
    default:
    case 4: d[3] = src->u64s[3]; // fallthrough
    case 3: d[2] = src->u64s[2]; // fallthrough
    case 2: d[1] = src->u64s[1]; // fallthrough
    case 1: d[0] = src->u64s[0]; // fallthrough
    case 0: break;
    }
}

/** @brief Type-agnostic partial load for 32-bit elements (2 elements max) into 64-bit vector. */
NK_INTERNAL void nk_partial_load_b32x2_serial_(void const *src, nk_b64_vec_t *dst, nk_size_t n) {
    dst->u64 = 0;
    nk_u32_t const *s = (nk_u32_t const *)src;
    switch (n) {
    default:
    case 2: dst->u32s[1] = s[1]; // fallthrough
    case 1: dst->u32s[0] = s[0]; // fallthrough
    case 0: break;
    }
}

/** @brief Type-agnostic partial load for 16-bit elements (4 elements max) into 64-bit vector. */
NK_INTERNAL void nk_partial_load_b16x4_serial_(void const *src, nk_b64_vec_t *dst, nk_size_t n) {
    dst->u64 = 0;
    nk_u16_t const *s = (nk_u16_t const *)src;
    switch (n) {
    default:
    case 4: dst->u16s[3] = s[3]; // fallthrough
    case 3: dst->u16s[2] = s[2]; // fallthrough
    case 2: dst->u16s[1] = s[1]; // fallthrough
    case 1: dst->u16s[0] = s[0]; // fallthrough
    case 0: break;
    }
}

/** @brief Type-agnostic partial load for 64-bit elements (2 elements max) into 128-bit vector. */
/** @brief Partial load for 4-bit nibbles (64 max = 32 bytes) into 256-bit vector (zeros in remaining slots). */
NK_INTERNAL void nk_partial_load_b4x64_serial_(void const *src, nk_b256_vec_t *dst, nk_size_t n) {
    dst->u64s[0] = 0, dst->u64s[1] = 0, dst->u64s[2] = 0, dst->u64s[3] = 0;
    nk_u8_t const *s = (nk_u8_t const *)src;
    nk_size_t n_bytes = (n + 1) / 2;
    for (nk_size_t i = 0; i < n_bytes && i < 32; i++) dst->u8s[i] = s[i];
}

NK_INTERNAL void nk_partial_load_b64x2_serial_(void const *src, nk_b128_vec_t *dst, nk_size_t n) {
    dst->u64s[0] = 0, dst->u64s[1] = 0;
    nk_u64_t const *s = (nk_u64_t const *)src;
    switch (n) {
    default:
    case 2: dst->u64s[1] = s[1]; // fallthrough
    case 1: dst->u64s[0] = s[0]; // fallthrough
    case 0: break;
    }
}

/** @brief Type-agnostic partial store for 64-bit elements (2 elements max) from 128-bit vector. */
NK_INTERNAL void nk_partial_store_b64x2_serial_(nk_b128_vec_t const *src, void *dst, nk_size_t n) {
    nk_u64_t *d = (nk_u64_t *)dst;
    switch (n) {
    default:
    case 2: d[1] = src->u64s[1]; // fallthrough
    case 1: d[0] = src->u64s[0]; // fallthrough
    case 0: break;
    }
}

/**
 *  @brief Union for type-punned scalar values at language binding boundaries.
 *
 *  Used to bridge different type systems (Python, JavaScript, etc.) where
 *  scalars arrive as f64 but need to be passed to kernels as typed pointers.
 *  The caller fills the appropriate union member based on the target dtype,
 *  then passes the union address as `void const *` to kernel functions.
 */
typedef union nk_scalar_buffer_t {
    nk_u8_t bytes[16];
    nk_f64_t f64;
    nk_f32_t f32;
    nk_f16_t f16;
    nk_bf16_t bf16;
    nk_f64c_t f64c;
    nk_f32c_t f32c;
    nk_f16c_t f16c;
    nk_bf16c_t bf16c;
    nk_i64_t i64;
    nk_u64_t u64;
    nk_i32_t i32;
    nk_u32_t u32;
    nk_i16_t i16;
    nk_u16_t u16;
    nk_i8_t i8;
    nk_u8_t u8;
} nk_scalar_buffer_t;

/**
 *  @brief Converts up to 8x values from `from_ptr` buffer into 8x puned buffer objects
 *  into a complex 64-bit floating point representation.
 */
NK_INTERNAL void nk_scalar_buffers_fill_f64c_(                         //
    void const *from_ptr, nk_dtype_t from_dtype, nk_size_t from_count, //
    nk_scalar_buffer_t to_buffers[nk_at_least_(8)]) {

    nk_f32_t temporary_f32;
    nk_size_t i;
    switch (from_dtype) {
    case nk_f64_k: {
        nk_f64_t const *p = (nk_f64_t const *)from_ptr;
        for (i = 0; i < from_count; ++i) to_buffers[i].f64c.real = p[i], to_buffers[i].f64c.imag = 0;
    } break;
    case nk_f32_k: {
        nk_f32_t const *p = (nk_f32_t const *)from_ptr;
        for (i = 0; i < from_count; ++i) to_buffers[i].f64c.real = p[i], to_buffers[i].f64c.imag = 0;
    } break;
    case nk_f16_k: {
        nk_f16_t const *p = (nk_f16_t const *)from_ptr;
        for (i = 0; i < from_count; ++i)
            nk_f16_to_f32_serial(&p[i], &temporary_f32), to_buffers[i].f64c.real = temporary_f32,
                                                         to_buffers[i].f64c.imag = 0;
    } break;
    case nk_bf16_k: {
        nk_bf16_t const *p = (nk_bf16_t const *)from_ptr;
        for (i = 0; i < from_count; ++i)
            nk_bf16_to_f32_serial(&p[i], &temporary_f32), to_buffers[i].f64c.real = temporary_f32,
                                                          to_buffers[i].f64c.imag = 0;
    } break;
    case nk_e4m3_k: {
        nk_u8_t const *p = (nk_u8_t const *)from_ptr;
        for (i = 0; i < from_count; ++i)
            nk_e4m3_to_f32_serial(&p[i], &temporary_f32), to_buffers[i].f64c.real = temporary_f32,
                                                          to_buffers[i].f64c.imag = 0;
    } break;
    case nk_e5m2_k: {
        nk_u8_t const *p = (nk_u8_t const *)from_ptr;
        for (i = 0; i < from_count; ++i)
            nk_e5m2_to_f32_serial(&p[i], &temporary_f32), to_buffers[i].f64c.real = temporary_f32,
                                                          to_buffers[i].f64c.imag = 0;
    } break;
    case nk_i64_k: {
        nk_i64_t const *p = (nk_i64_t const *)from_ptr;
        for (i = 0; i < from_count; ++i) to_buffers[i].f64c.real = (nk_f64_t)p[i], to_buffers[i].f64c.imag = 0;
    } break;
    case nk_i32_k: {
        nk_i32_t const *p = (nk_i32_t const *)from_ptr;
        for (i = 0; i < from_count; ++i) to_buffers[i].f64c.real = p[i], to_buffers[i].f64c.imag = 0;
    } break;
    case nk_i16_k: {
        nk_i16_t const *p = (nk_i16_t const *)from_ptr;
        for (i = 0; i < from_count; ++i) to_buffers[i].f64c.real = p[i], to_buffers[i].f64c.imag = 0;
    } break;
    case nk_i8_k: {
        nk_i8_t const *p = (nk_i8_t const *)from_ptr;
        for (i = 0; i < from_count; ++i) to_buffers[i].f64c.real = p[i], to_buffers[i].f64c.imag = 0;
    } break;
    case nk_u64_k: {
        nk_u64_t const *p = (nk_u64_t const *)from_ptr;
        for (i = 0; i < from_count; ++i) to_buffers[i].f64c.real = (nk_f64_t)p[i], to_buffers[i].f64c.imag = 0;
    } break;
    case nk_u32_k: {
        nk_u32_t const *p = (nk_u32_t const *)from_ptr;
        for (i = 0; i < from_count; ++i) to_buffers[i].f64c.real = p[i], to_buffers[i].f64c.imag = 0;
    } break;
    case nk_u16_k: {
        nk_u16_t const *p = (nk_u16_t const *)from_ptr;
        for (i = 0; i < from_count; ++i) to_buffers[i].f64c.real = p[i], to_buffers[i].f64c.imag = 0;
    } break;
    case nk_u8_k: {
        nk_u8_t const *p = (nk_u8_t const *)from_ptr;
        for (i = 0; i < from_count; ++i) to_buffers[i].f64c.real = p[i], to_buffers[i].f64c.imag = 0;
    } break;
    case nk_f64c_k: {
        nk_f64c_t const *p = (nk_f64c_t const *)from_ptr;
        for (i = 0; i < from_count; ++i) to_buffers[i].f64c = p[i];
    } break;
    case nk_f32c_k: {
        nk_f32c_t const *p = (nk_f32c_t const *)from_ptr;
        for (i = 0; i < from_count; ++i) to_buffers[i].f64c.real = p[i].real, to_buffers[i].f64c.imag = p[i].imag;
    } break;
    case nk_f16c_k: {
        nk_f16c_t const *p = (nk_f16c_t const *)from_ptr;
        for (i = 0; i < from_count; ++i) {
            nk_f16_to_f32_serial(&p[i].real, &temporary_f32), to_buffers[i].f64c.real = temporary_f32;
            nk_f16_to_f32_serial(&p[i].imag, &temporary_f32), to_buffers[i].f64c.imag = temporary_f32;
        }
    } break;
    case nk_bf16c_k: {
        nk_bf16c_t const *p = (nk_bf16c_t const *)from_ptr;
        for (i = 0; i < from_count; ++i) {
            nk_bf16_to_f32_serial(&p[i].real, &temporary_f32), to_buffers[i].f64c.real = temporary_f32;
            nk_bf16_to_f32_serial(&p[i].imag, &temporary_f32), to_buffers[i].f64c.imag = temporary_f32;
        }
    } break;
    // Sub-byte: u1 - 8 bits from 1 byte, MSB-first
    case nk_u1_k: {
        nk_u8_t byte = *(nk_u8_t const *)from_ptr;
        for (i = 0; i < 8; ++i) to_buffers[i].f64c.real = (byte >> (7 - i)) & 1, to_buffers[i].f64c.imag = 0;
    } break;
    // Sub-byte: i4 - 8 nibbles from 4 bytes, high nibble = even index, sign-extended
    case nk_i4_k: {
        nk_u8_t const *p = (nk_u8_t const *)from_ptr;
        for (i = 0; i < 4; ++i) {
            nk_i8_t hi = (nk_i8_t)(p[i] >> 4), lo = (nk_i8_t)(p[i] & 0xF);
            to_buffers[i * 2].f64c.real = (hi ^ 8) - 8, to_buffers[i * 2].f64c.imag = 0;
            to_buffers[i * 2 + 1].f64c.real = (lo ^ 8) - 8, to_buffers[i * 2 + 1].f64c.imag = 0;
        }
    } break;
    // Sub-byte: u4 - 8 nibbles from 4 bytes, high nibble = even index
    case nk_u4_k: {
        nk_u8_t const *p = (nk_u8_t const *)from_ptr;
        for (i = 0; i < 4; ++i) {
            to_buffers[i * 2].f64c.real = p[i] >> 4, to_buffers[i * 2].f64c.imag = 0;
            to_buffers[i * 2 + 1].f64c.real = p[i] & 0xF, to_buffers[i * 2 + 1].f64c.imag = 0;
        }
    } break;
    default:
        for (i = 0; i < 8; ++i) to_buffers[i].f64c.real = 0, to_buffers[i].f64c.imag = 0;
        break;
    }
}

/**
 *  @brief Converts up to 8x values from `from_buffers` buffer into 8x typed scalars.
 */
NK_INTERNAL void nk_scalar_buffers_export_f64c_(            //
    nk_scalar_buffer_t const from_buffers[nk_at_least_(8)], //
    void *to_ptr, nk_dtype_t to_dtype, nk_size_t to_count) {

    nk_f32_t temporary_f32;
    nk_size_t i;
    switch (to_dtype) {
    case nk_f64_k: {
        nk_f64_t *p = (nk_f64_t *)to_ptr;
        for (i = 0; i < to_count; ++i) p[i] = from_buffers[i].f64c.real;
    } break;
    case nk_f32_k: {
        nk_f32_t *p = (nk_f32_t *)to_ptr;
        for (i = 0; i < to_count; ++i) p[i] = (nk_f32_t)from_buffers[i].f64c.real;
    } break;
    case nk_f16_k: {
        nk_f16_t *p = (nk_f16_t *)to_ptr;
        for (i = 0; i < to_count; ++i)
            temporary_f32 = (nk_f32_t)from_buffers[i].f64c.real, nk_f32_to_f16_serial(&temporary_f32, &p[i]);
    } break;
    case nk_bf16_k: {
        nk_bf16_t *p = (nk_bf16_t *)to_ptr;
        for (i = 0; i < to_count; ++i)
            temporary_f32 = (nk_f32_t)from_buffers[i].f64c.real, nk_f32_to_bf16_serial(&temporary_f32, &p[i]);
    } break;
    case nk_e4m3_k: {
        nk_u8_t *p = (nk_u8_t *)to_ptr;
        for (i = 0; i < to_count; ++i)
            temporary_f32 = (nk_f32_t)from_buffers[i].f64c.real, nk_f32_to_e4m3_serial(&temporary_f32, &p[i]);
    } break;
    case nk_e5m2_k: {
        nk_u8_t *p = (nk_u8_t *)to_ptr;
        for (i = 0; i < to_count; ++i)
            temporary_f32 = (nk_f32_t)from_buffers[i].f64c.real, nk_f32_to_e5m2_serial(&temporary_f32, &p[i]);
    } break;
    case nk_i64_k: {
        nk_i64_t *p = (nk_i64_t *)to_ptr;
        for (i = 0; i < to_count; ++i) nk_f64_to_i64_serial(&from_buffers[i].f64c.real, &p[i]);
    } break;
    case nk_i32_k: {
        nk_i32_t *p = (nk_i32_t *)to_ptr;
        for (i = 0; i < to_count; ++i) nk_f64_to_i32_serial(&from_buffers[i].f64c.real, &p[i]);
    } break;
    case nk_i16_k: {
        nk_i16_t *p = (nk_i16_t *)to_ptr;
        for (i = 0; i < to_count; ++i) nk_f64_to_i16_serial(&from_buffers[i].f64c.real, &p[i]);
    } break;
    case nk_i8_k: {
        nk_i8_t *p = (nk_i8_t *)to_ptr;
        for (i = 0; i < to_count; ++i) nk_f64_to_i8_serial(&from_buffers[i].f64c.real, &p[i]);
    } break;
    case nk_u64_k: {
        nk_u64_t *p = (nk_u64_t *)to_ptr;
        for (i = 0; i < to_count; ++i) nk_f64_to_u64_serial(&from_buffers[i].f64c.real, &p[i]);
    } break;
    case nk_u32_k: {
        nk_u32_t *p = (nk_u32_t *)to_ptr;
        for (i = 0; i < to_count; ++i) nk_f64_to_u32_serial(&from_buffers[i].f64c.real, &p[i]);
    } break;
    case nk_u16_k: {
        nk_u16_t *p = (nk_u16_t *)to_ptr;
        for (i = 0; i < to_count; ++i) nk_f64_to_u16_serial(&from_buffers[i].f64c.real, &p[i]);
    } break;
    case nk_u8_k: {
        nk_u8_t *p = (nk_u8_t *)to_ptr;
        for (i = 0; i < to_count; ++i) nk_f64_to_u8_serial(&from_buffers[i].f64c.real, &p[i]);
    } break;
    case nk_f64c_k: {
        nk_f64c_t *p = (nk_f64c_t *)to_ptr;
        for (i = 0; i < to_count; ++i) p[i] = from_buffers[i].f64c;
    } break;
    case nk_f32c_k: {
        nk_f32c_t *p = (nk_f32c_t *)to_ptr;
        for (i = 0; i < to_count; ++i)
            p[i].real = (nk_f32_t)from_buffers[i].f64c.real, p[i].imag = (nk_f32_t)from_buffers[i].f64c.imag;
    } break;
    case nk_f16c_k: {
        nk_f16c_t *p = (nk_f16c_t *)to_ptr;
        for (i = 0; i < to_count; ++i) {
            temporary_f32 = (nk_f32_t)from_buffers[i].f64c.real, nk_f32_to_f16_serial(&temporary_f32, &p[i].real);
            temporary_f32 = (nk_f32_t)from_buffers[i].f64c.imag, nk_f32_to_f16_serial(&temporary_f32, &p[i].imag);
        }
    } break;
    case nk_bf16c_k: {
        nk_bf16c_t *p = (nk_bf16c_t *)to_ptr;
        for (i = 0; i < to_count; ++i) {
            temporary_f32 = (nk_f32_t)from_buffers[i].f64c.real, nk_f32_to_bf16_serial(&temporary_f32, &p[i].real);
            temporary_f32 = (nk_f32_t)from_buffers[i].f64c.imag, nk_f32_to_bf16_serial(&temporary_f32, &p[i].imag);
        }
    } break;
    // Sub-byte: u1 - 8 bits to 1 byte, MSB-first, non-zero→1
    case nk_u1_k: {
        nk_u8_t *p = (nk_u8_t *)to_ptr;
        nk_u8_t byte = 0;
        for (i = 0; i < 8; ++i) byte |= (from_buffers[i].f64c.real != 0) << (7 - i);
        *p = byte;
    } break;
    // Sub-byte: i4 - 8 nibbles to 4 bytes, high nibble = even index
    case nk_i4_k: {
        nk_u8_t *p = (nk_u8_t *)to_ptr;
        for (i = 0; i < 4; ++i) {
            nk_i64_t hi = (nk_i64_t)from_buffers[i * 2].f64c.real;
            nk_i64_t lo = (nk_i64_t)from_buffers[i * 2 + 1].f64c.real;
            hi = hi > 7 ? 7 : (hi < -8 ? -8 : hi);
            lo = lo > 7 ? 7 : (lo < -8 ? -8 : lo);
            p[i] = (nk_u8_t)(((hi & 0xF) << 4) | (lo & 0xF));
        }
    } break;
    // Sub-byte: u4 - 8 nibbles to 4 bytes, high nibble = even index
    case nk_u4_k: {
        nk_u8_t *p = (nk_u8_t *)to_ptr;
        for (i = 0; i < 4; ++i) {
            nk_u64_t hi = (nk_u64_t)from_buffers[i * 2].f64c.real;
            nk_u64_t lo = (nk_u64_t)from_buffers[i * 2 + 1].f64c.real;
            hi = hi > 15 ? 15 : hi;
            lo = lo > 15 ? 15 : lo;
            p[i] = (nk_u8_t)((hi << 4) | lo);
        }
    } break;
    default: break;
    }
}

/**
 *  @brief Load 8 values from typed buffer into `buf[i].i64` (lossless widening for signed integers).
 */
NK_INTERNAL void nk_scalar_buffers_fill_i64_(                          //
    void const *from_ptr, nk_dtype_t from_dtype, nk_size_t from_count, //
    nk_scalar_buffer_t to_buffers[nk_at_least_(8)]) {                  //
    nk_size_t i;
    switch (from_dtype) {
    case nk_i64_k: {
        nk_i64_t const *p = (nk_i64_t const *)from_ptr;
        for (i = 0; i < from_count; ++i) to_buffers[i].i64 = p[i];
    } break;
    case nk_i32_k: {
        nk_i32_t const *p = (nk_i32_t const *)from_ptr;
        for (i = 0; i < from_count; ++i) to_buffers[i].i64 = p[i];
    } break;
    case nk_i16_k: {
        nk_i16_t const *p = (nk_i16_t const *)from_ptr;
        for (i = 0; i < from_count; ++i) to_buffers[i].i64 = p[i];
    } break;
    case nk_i8_k: {
        nk_i8_t const *p = (nk_i8_t const *)from_ptr;
        for (i = 0; i < from_count; ++i) to_buffers[i].i64 = p[i];
    } break;
    // Sub-byte: i4 - 4 bytes to 8 nibbles, sign-extend each nibble
    case nk_i4_k: {
        nk_u8_t const *p = (nk_u8_t const *)from_ptr;
        for (i = 0; i < 4; ++i) {
            nk_i8_t hi = (nk_i8_t)(p[i] >> 4), lo = (nk_i8_t)(p[i] & 0xF);
            to_buffers[i * 2].i64 = (hi ^ 8) - 8;
            to_buffers[i * 2 + 1].i64 = (lo ^ 8) - 8;
        }
    } break;
    default: break;
    }
}

/**
 *  @brief Export 8 `buf[i].i64` values to typed buffer with saturation on downcast.
 */
NK_INTERNAL void nk_scalar_buffers_export_i64_(              //
    nk_scalar_buffer_t const from_buffers[nk_at_least_(8)],  //
    void *to_ptr, nk_dtype_t to_dtype, nk_size_t to_count) { //
    nk_size_t i;
    switch (to_dtype) {
    case nk_i64_k: {
        nk_i64_t *p = (nk_i64_t *)to_ptr;
        for (i = 0; i < to_count; ++i) p[i] = from_buffers[i].i64;
    } break;
    case nk_i32_k: {
        nk_i32_t *p = (nk_i32_t *)to_ptr;
        for (i = 0; i < to_count; ++i) nk_i64_to_i32_serial(&from_buffers[i].i64, &p[i]);
    } break;
    case nk_i16_k: {
        nk_i16_t *p = (nk_i16_t *)to_ptr;
        for (i = 0; i < to_count; ++i) nk_i64_to_i16_serial(&from_buffers[i].i64, &p[i]);
    } break;
    case nk_i8_k: {
        nk_i8_t *p = (nk_i8_t *)to_ptr;
        for (i = 0; i < to_count; ++i) nk_i64_to_i8_serial(&from_buffers[i].i64, &p[i]);
    } break;
    // Unsigned targets: clamp negatives to 0
    case nk_u64_k: {
        nk_u64_t *p = (nk_u64_t *)to_ptr;
        for (i = 0; i < to_count; ++i) nk_i64_to_u64_serial(&from_buffers[i].i64, &p[i]);
    } break;
    case nk_u32_k: {
        nk_u32_t *p = (nk_u32_t *)to_ptr;
        for (i = 0; i < to_count; ++i) nk_i64_to_u32_serial(&from_buffers[i].i64, &p[i]);
    } break;
    case nk_u16_k: {
        nk_u16_t *p = (nk_u16_t *)to_ptr;
        for (i = 0; i < to_count; ++i) nk_i64_to_u16_serial(&from_buffers[i].i64, &p[i]);
    } break;
    case nk_u8_k: {
        nk_u8_t *p = (nk_u8_t *)to_ptr;
        for (i = 0; i < to_count; ++i) nk_i64_to_u8_serial(&from_buffers[i].i64, &p[i]);
    } break;
    // Sub-byte: i4 - 8 nibbles to 4 bytes, clamp [-8,7]
    case nk_i4_k: {
        nk_u8_t *p = (nk_u8_t *)to_ptr;
        for (i = 0; i < 4; ++i) {
            nk_i64_t hi = from_buffers[i * 2].i64, lo = from_buffers[i * 2 + 1].i64;
            hi = hi > 7 ? 7 : (hi < -8 ? -8 : hi);
            lo = lo > 7 ? 7 : (lo < -8 ? -8 : lo);
            p[i] = (nk_u8_t)(((hi & 0xF) << 4) | (lo & 0xF));
        }
    } break;
    default: break;
    }
}

/**
 *  @brief Load 8 values from typed buffer into `buf[i].u64` (lossless widening for unsigned integers).
 */
NK_INTERNAL void nk_scalar_buffers_fill_u64_(                          //
    void const *from_ptr, nk_dtype_t from_dtype, nk_size_t from_count, //
    nk_scalar_buffer_t to_buffers[nk_at_least_(8)]) {                  //
    nk_size_t i;
    switch (from_dtype) {
    case nk_u64_k: {
        nk_u64_t const *p = (nk_u64_t const *)from_ptr;
        for (i = 0; i < from_count; ++i) to_buffers[i].u64 = p[i];
    } break;
    case nk_u32_k: {
        nk_u32_t const *p = (nk_u32_t const *)from_ptr;
        for (i = 0; i < from_count; ++i) to_buffers[i].u64 = p[i];
    } break;
    case nk_u16_k: {
        nk_u16_t const *p = (nk_u16_t const *)from_ptr;
        for (i = 0; i < from_count; ++i) to_buffers[i].u64 = p[i];
    } break;
    case nk_u8_k: {
        nk_u8_t const *p = (nk_u8_t const *)from_ptr;
        for (i = 0; i < from_count; ++i) to_buffers[i].u64 = p[i];
    } break;
    // Sub-byte: u4 - 4 bytes to 8 nibbles, zero-extend
    case nk_u4_k: {
        nk_u8_t const *p = (nk_u8_t const *)from_ptr;
        for (i = 0; i < 4; ++i) {
            to_buffers[i * 2].u64 = p[i] >> 4;
            to_buffers[i * 2 + 1].u64 = p[i] & 0xF;
        }
    } break;
    // Sub-byte: u1 - 1 byte to 8 bits, MSB-first
    case nk_u1_k: {
        nk_u8_t byte = *(nk_u8_t const *)from_ptr;
        for (i = 0; i < 8; ++i) to_buffers[i].u64 = (byte >> (7 - i)) & 1;
    } break;
    default: break;
    }
}

/**
 *  @brief Export 8 `buf[i].u64` values to typed buffer with saturation on downcast.
 */
NK_INTERNAL void nk_scalar_buffers_export_u64_(              //
    nk_scalar_buffer_t const from_buffers[nk_at_least_(8)],  //
    void *to_ptr, nk_dtype_t to_dtype, nk_size_t to_count) { //
    nk_size_t i;
    switch (to_dtype) {
    case nk_u64_k: {
        nk_u64_t *p = (nk_u64_t *)to_ptr;
        for (i = 0; i < to_count; ++i) p[i] = from_buffers[i].u64;
    } break;
    case nk_u32_k: {
        nk_u32_t *p = (nk_u32_t *)to_ptr;
        for (i = 0; i < to_count; ++i) nk_u64_to_u32_serial(&from_buffers[i].u64, &p[i]);
    } break;
    case nk_u16_k: {
        nk_u16_t *p = (nk_u16_t *)to_ptr;
        for (i = 0; i < to_count; ++i) nk_u64_to_u16_serial(&from_buffers[i].u64, &p[i]);
    } break;
    case nk_u8_k: {
        nk_u8_t *p = (nk_u8_t *)to_ptr;
        for (i = 0; i < to_count; ++i) nk_u64_to_u8_serial(&from_buffers[i].u64, &p[i]);
    } break;
    // Signed targets: clamp to i64_max
    case nk_i64_k: {
        nk_i64_t *p = (nk_i64_t *)to_ptr;
        for (i = 0; i < to_count; ++i) nk_u64_to_i64_serial(&from_buffers[i].u64, &p[i]);
    } break;
    case nk_i32_k: {
        nk_i32_t *p = (nk_i32_t *)to_ptr;
        for (i = 0; i < to_count; ++i) nk_u64_to_i32_serial(&from_buffers[i].u64, &p[i]);
    } break;
    case nk_i16_k: {
        nk_i16_t *p = (nk_i16_t *)to_ptr;
        for (i = 0; i < to_count; ++i) nk_u64_to_i16_serial(&from_buffers[i].u64, &p[i]);
    } break;
    case nk_i8_k: {
        nk_i8_t *p = (nk_i8_t *)to_ptr;
        for (i = 0; i < to_count; ++i) nk_u64_to_i8_serial(&from_buffers[i].u64, &p[i]);
    } break;
    // Sub-byte: u4 - 8 nibbles to 4 bytes, clamp [0,15]
    case nk_u4_k: {
        nk_u8_t *p = (nk_u8_t *)to_ptr;
        for (i = 0; i < 4; ++i) {
            nk_u64_t hi = from_buffers[i * 2].u64, lo = from_buffers[i * 2 + 1].u64;
            hi = hi > 15 ? 15 : hi;
            lo = lo > 15 ? 15 : lo;
            p[i] = (nk_u8_t)((hi << 4) | lo);
        }
    } break;
    // Sub-byte: u1 - 8 bits to 1 byte, MSB-first, non-zero becomes 1
    case nk_u1_k: {
        nk_u8_t *p = (nk_u8_t *)to_ptr;
        nk_u8_t byte = 0;
        for (i = 0; i < 8; ++i) byte |= (from_buffers[i].u64 != 0) << (7 - i);
        *p = byte;
    } break;
    default: break;
    }
}

#pragma endregion - Type Punned Loads and Stores

#pragma region - Public API

NK_PUBLIC void nk_cast_serial(void const *from, nk_dtype_t from_type, nk_size_t n, void *to, nk_dtype_t to_type) {
    if (from_type == to_type) {
        nk_size_t size_bits = nk_dtype_bits(from_type);
        nk_size_t size_bytes = nk_size_divide_round_up_to_multiple_(n * size_bits, NK_BITS_PER_BYTE);
        if (size_bytes > 0) nk_copy_bytes_(to, from, size_bytes);
        return;
    }

    nk_size_t from_bits = nk_dtype_bits(from_type);
    nk_size_t to_bits = nk_dtype_bits(to_type);
    if (from_bits == 0 || to_bits == 0) return;

    // Byte steps per batch of NK_BITS_PER_BYTE elements
    nk_size_t from_step = from_bits;
    nk_size_t to_step = to_bits;

    nk_u8_t const *src = (nk_u8_t const *)from;
    nk_u8_t *dst = (nk_u8_t *)to;
    nk_dtype_family_k from_family = nk_dtype_family(from_type);
    nk_dtype_family_k to_family = nk_dtype_family(to_type);

    nk_size_t batches = n / NK_BITS_PER_BYTE;
    nk_size_t tail = n % NK_BITS_PER_BYTE;
    nk_scalar_buffer_t bufs[NK_BITS_PER_BYTE];

    // Both unsigned: u64 hub
    if (from_family == nk_dtype_family_uint_k && to_family == nk_dtype_family_uint_k) {
        for (nk_size_t b = 0; b < batches; ++b, src += from_step, dst += to_step) {
            nk_scalar_buffers_fill_u64_(src, from_type, NK_BITS_PER_BYTE, bufs);
            nk_scalar_buffers_export_u64_(bufs, dst, to_type, NK_BITS_PER_BYTE);
        }
        if (tail) {
            nk_scalar_buffers_fill_u64_(src, from_type, tail, bufs);
            nk_scalar_buffers_export_u64_(bufs, dst, to_type, tail);
        }
        return;
    }

    // Both integers, at least one signed: i64 hub
    if ((from_family == nk_dtype_family_int_k || from_family == nk_dtype_family_uint_k) &&
        (to_family == nk_dtype_family_int_k || to_family == nk_dtype_family_uint_k)) {
        for (nk_size_t b = 0; b < batches; ++b, src += from_step, dst += to_step) {
            nk_scalar_buffers_fill_i64_(src, from_type, NK_BITS_PER_BYTE, bufs);
            nk_scalar_buffers_export_i64_(bufs, dst, to_type, NK_BITS_PER_BYTE);
        }
        if (tail) {
            nk_scalar_buffers_fill_i64_(src, from_type, tail, bufs);
            nk_scalar_buffers_export_i64_(bufs, dst, to_type, tail);
        }
        return;
    }

    // Everything else: f64c hub (floats, complex, cross-category)
    for (nk_size_t b = 0; b < batches; ++b, src += from_step, dst += to_step) {
        nk_scalar_buffers_fill_f64c_(src, from_type, NK_BITS_PER_BYTE, bufs);
        nk_scalar_buffers_export_f64c_(bufs, dst, to_type, NK_BITS_PER_BYTE);
    }
    if (tail) {
        nk_scalar_buffers_fill_f64c_(src, from_type, tail, bufs);
        nk_scalar_buffers_export_f64c_(bufs, dst, to_type, tail);
    }
}

/** @brief Convert E4M3 to BF16 via F32 intermediate. */
NK_PUBLIC void nk_e4m3_to_bf16(nk_e4m3_t const *src, nk_bf16_t *dest) {
    nk_f32_t temp;
    nk_e4m3_to_f32_serial(src, &temp);
    nk_f32_to_bf16_serial(&temp, dest);
}

/** @brief Convert E5M2 to BF16 via F32 intermediate. */
NK_PUBLIC void nk_e5m2_to_bf16(nk_e5m2_t const *src, nk_bf16_t *dest) {
    nk_f32_t temp;
    nk_e5m2_to_f32_serial(src, &temp);
    nk_f32_to_bf16_serial(&temp, dest);
}

#pragma endregion - Public API

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_CAST_SERIAL_H

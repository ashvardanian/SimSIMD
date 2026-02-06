/**
 *  @brief SIMD-accelerated Vector Reductions for Ice Lake.
 *  @file include/numkong/reduce/icelake.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 *
 *  @sa include/numkong/reduce.h
 *
 *  @section ice_reduce_instructions Relevant Instructions
 *
 *      Intrinsic                   Instruction                     Ice         Genoa
 *      _mm512_dpwssd_epi32         VPDPWSSD (ZMM, ZMM, ZMM)        5cy @ p0    4cy @ p01
 *      _mm512_cvtepi8_epi16        VPMOVSXBW (ZMM, YMM)            3cy @ p5    3cy @ p12
 *      _mm512_cvtepu8_epi16        VPMOVZXBW (ZMM, YMM)            3cy @ p5    3cy @ p12
 *      _mm512_reduce_add_epi32     (pseudo: shuffle chain)         ~8cy        ~8cy
 *      _mm512_unpacklo_epi8        VPUNPCKLBW (ZMM, ZMM, ZMM)      1cy @ p5    1cy @ p12
 *      _mm512_extracti64x4_epi64   VEXTRACTI64X4 (YMM, ZMM, imm8)  3cy @ p5    3cy @ p12
 *
 *  Ice Lake reductions leverage VNNI's VPDPWSSD for efficient i8/i16 summation by multiplying with a
 *  vector of ones. This approach achieves better throughput than sequential addition for large arrays.
 *  Shuffle operations bottleneck on port 5, making horizontal reductions inherently latency-bound.
 */
#ifndef NK_REDUCE_ICELAKE_H
#define NK_REDUCE_ICELAKE_H

#if NK_TARGET_X86_
#if NK_TARGET_ICELAKE

#include "numkong/types.h"
#include "numkong/reduce/serial.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(                                                                        \
    __attribute__((target("avx2,avx512f,avx512vl,avx512bw,avx512dq,avx512vnni,f16c,fma,bmi,bmi2"))), \
    apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "avx512bw", "avx512dq", "avx512vnni", "f16c", "fma", "bmi", "bmi2")
#endif

NK_INTERNAL __mmask64 nk_stride_mask_u1x64_icelake_(nk_size_t stride) {
    switch (stride) {
    case 2: return (__mmask64)0x5555555555555555ull;
    case 3: return (__mmask64)0x1249249249249249ull;
    case 4: return (__mmask64)0x1111111111111111ull;
    case 5: return (__mmask64)0x1084210842108421ull;
    case 6: return (__mmask64)0x1041041041041041ull;
    case 7: return (__mmask64)0x0408102040810204ull;
    case 8: return (__mmask64)0x0101010101010101ull;
    default: return (__mmask64)0;
    }
}

NK_INTERNAL __mmask32 nk_stride_mask_b16x32_icelake_(nk_size_t stride) {
    switch (stride) {
    case 2: return (__mmask32)0x55555555;
    case 3: return (__mmask32)0x09249249;
    case 4: return (__mmask32)0x11111111;
    case 5: return (__mmask32)0x01084210;
    case 6: return (__mmask32)0x01041041;
    case 7: return (__mmask32)0x00408102;
    case 8: return (__mmask32)0x01010101;
    default: return (__mmask32)0;
    }
}

NK_INTERNAL void nk_reduce_add_i16_icelake_contiguous_( //
    nk_i16_t const *data, nk_size_t count, nk_i64_t *result) {
    __m512i ones_i16x32 = _mm512_set1_epi16(1);
    nk_i64_t sum_i64 = 0;
    nk_size_t idx = 0;
    nk_size_t const chunk_elements = 1024 * 32;

    while (idx + chunk_elements <= count) {
        __m512i sum_i32x16 = _mm512_setzero_si512();
        for (nk_size_t end = idx + chunk_elements; idx < end; idx += 32) {
            __m512i data_i16x32 = _mm512_loadu_si512(data + idx);
            sum_i32x16 = _mm512_dpwssd_epi32(sum_i32x16, data_i16x32, ones_i16x32);
        }
        sum_i64 += _mm512_reduce_add_epi32(sum_i32x16);
    }

    __m512i sum_i32x16 = _mm512_setzero_si512();
    for (; idx + 32 <= count; idx += 32) {
        __m512i data_i16x32 = _mm512_loadu_si512(data + idx);
        sum_i32x16 = _mm512_dpwssd_epi32(sum_i32x16, data_i16x32, ones_i16x32);
    }
    if (idx < count) {
        __mmask32 mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, count - idx);
        __m512i data_i16x32 = _mm512_maskz_loadu_epi16(mask, data + idx);
        sum_i32x16 = _mm512_dpwssd_epi32(sum_i32x16, data_i16x32, ones_i16x32);
    }
    *result = sum_i64 + _mm512_reduce_add_epi32(sum_i32x16);
}

NK_INTERNAL void nk_reduce_add_i16_icelake_strided_(                  //
    nk_i16_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_i64_t *result) {
    __mmask32 stride_mask_m32 = nk_stride_mask_b16x32_icelake_(stride_elements);
    __m512i ones_i16x32 = _mm512_set1_epi16(1);
    nk_i64_t sum_i64 = 0;
    nk_size_t idx_scalars = 0;
    nk_size_t total_scalars = count * stride_elements;
    nk_size_t const chunk_scalars = 1024 * 32;

    while (idx_scalars + chunk_scalars <= total_scalars) {
        __m512i sum_i32x16 = _mm512_setzero_si512();
        for (nk_size_t end = idx_scalars + chunk_scalars; idx_scalars < end; idx_scalars += 32) {
            __m512i data_i16x32 = _mm512_maskz_loadu_epi16(stride_mask_m32, data + idx_scalars);
            sum_i32x16 = _mm512_dpwssd_epi32(sum_i32x16, data_i16x32, ones_i16x32);
        }
        sum_i64 += _mm512_reduce_add_epi32(sum_i32x16);
    }

    __m512i sum_i32x16 = _mm512_setzero_si512();
    for (; idx_scalars + 32 <= total_scalars; idx_scalars += 32) {
        __m512i data_i16x32 = _mm512_maskz_loadu_epi16(stride_mask_m32, data + idx_scalars);
        sum_i32x16 = _mm512_dpwssd_epi32(sum_i32x16, data_i16x32, ones_i16x32);
    }

    // Handle tail with masked SIMD instead of scalar loop
    if (idx_scalars < total_scalars) {
        nk_size_t remaining_scalars = total_scalars - idx_scalars;
        // Compute mask that combines stride pattern with tail boundary
        __mmask32 tail_mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, remaining_scalars);
        __mmask32 combined_mask = _kand_mask32(stride_mask_m32, tail_mask);
        __m512i data_i16x32 = _mm512_maskz_loadu_epi16(combined_mask, data + idx_scalars);
        sum_i32x16 = _mm512_dpwssd_epi32(sum_i32x16, data_i16x32, ones_i16x32);
    }
    sum_i64 += _mm512_reduce_add_epi32(sum_i32x16);

    *result = sum_i64;
}

NK_PUBLIC void nk_reduce_add_i16_icelake(                          //
    nk_i16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *result) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_i16_t);
    int aligned = (stride_bytes % sizeof(nk_i16_t) == 0);
    if (!aligned) nk_reduce_add_i16_serial(data, count, stride_bytes, result);
    else if (stride_elements == 1) nk_reduce_add_i16_icelake_contiguous_(data, count, result);
    else if (stride_elements <= 8) nk_reduce_add_i16_icelake_strided_(data, count, stride_elements, result);
    else nk_reduce_add_i16_serial(data, count, stride_bytes, result);
}

NK_INTERNAL void nk_reduce_add_u16_icelake_contiguous_( //
    nk_u16_t const *data, nk_size_t count, nk_u64_t *result) {
    __m512i ones_i16x32 = _mm512_set1_epi16(1);
    nk_u64_t sum_u64 = 0;
    nk_size_t idx = 0;
    nk_size_t const chunk_elements = 1024 * 32;

    while (idx + chunk_elements <= count) {
        __m512i sum_i32x16 = _mm512_setzero_si512();
        for (nk_size_t end = idx + chunk_elements; idx < end; idx += 32) {
            __m512i data_u16x32 = _mm512_loadu_si512(data + idx);
            sum_i32x16 = _mm512_dpwssd_epi32(sum_i32x16, data_u16x32, ones_i16x32);
        }
        sum_u64 += (nk_u64_t)_mm512_reduce_add_epi32(sum_i32x16);
    }

    __m512i sum_i32x16 = _mm512_setzero_si512();
    for (; idx + 32 <= count; idx += 32) {
        __m512i data_u16x32 = _mm512_loadu_si512(data + idx);
        sum_i32x16 = _mm512_dpwssd_epi32(sum_i32x16, data_u16x32, ones_i16x32);
    }
    if (idx < count) {
        __mmask32 mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, count - idx);
        __m512i data_u16x32 = _mm512_maskz_loadu_epi16(mask, data + idx);
        sum_i32x16 = _mm512_dpwssd_epi32(sum_i32x16, data_u16x32, ones_i16x32);
    }
    *result = sum_u64 + (nk_u64_t)_mm512_reduce_add_epi32(sum_i32x16);
}

NK_INTERNAL void nk_reduce_add_u16_icelake_strided_(                  //
    nk_u16_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_u64_t *result) {
    __mmask32 stride_mask_m32 = nk_stride_mask_b16x32_icelake_(stride_elements);
    __m512i ones_i16x32 = _mm512_set1_epi16(1);
    nk_u64_t sum_u64 = 0;
    nk_size_t idx_scalars = 0;
    nk_size_t total_scalars = count * stride_elements;
    nk_size_t const chunk_scalars = 1024 * 32;

    while (idx_scalars + chunk_scalars <= total_scalars) {
        __m512i sum_i32x16 = _mm512_setzero_si512();
        for (nk_size_t end = idx_scalars + chunk_scalars; idx_scalars < end; idx_scalars += 32) {
            __m512i data_u16x32 = _mm512_maskz_loadu_epi16(stride_mask_m32, data + idx_scalars);
            sum_i32x16 = _mm512_dpwssd_epi32(sum_i32x16, data_u16x32, ones_i16x32);
        }
        sum_u64 += (nk_u64_t)_mm512_reduce_add_epi32(sum_i32x16);
    }

    __m512i sum_i32x16 = _mm512_setzero_si512();
    for (; idx_scalars + 32 <= total_scalars; idx_scalars += 32) {
        __m512i data_u16x32 = _mm512_maskz_loadu_epi16(stride_mask_m32, data + idx_scalars);
        sum_i32x16 = _mm512_dpwssd_epi32(sum_i32x16, data_u16x32, ones_i16x32);
    }

    // Handle tail with masked SIMD instead of scalar loop
    if (idx_scalars < total_scalars) {
        nk_size_t remaining_scalars = total_scalars - idx_scalars;
        __mmask32 tail_mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, remaining_scalars);
        __mmask32 combined_mask = _kand_mask32(stride_mask_m32, tail_mask);
        __m512i data_u16x32 = _mm512_maskz_loadu_epi16(combined_mask, data + idx_scalars);
        sum_i32x16 = _mm512_dpwssd_epi32(sum_i32x16, data_u16x32, ones_i16x32);
    }
    sum_u64 += (nk_u64_t)_mm512_reduce_add_epi32(sum_i32x16);

    *result = sum_u64;
}

NK_PUBLIC void nk_reduce_add_u16_icelake(                          //
    nk_u16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *result) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_u16_t);
    int aligned = (stride_bytes % sizeof(nk_u16_t) == 0);
    if (!aligned) nk_reduce_add_u16_serial(data, count, stride_bytes, result);
    else if (stride_elements == 1) nk_reduce_add_u16_icelake_contiguous_(data, count, result);
    else if (stride_elements <= 8) nk_reduce_add_u16_icelake_strided_(data, count, stride_elements, result);
    else nk_reduce_add_u16_serial(data, count, stride_bytes, result);
}

NK_INTERNAL void nk_reduce_add_i8_icelake_contiguous_( //
    nk_i8_t const *data, nk_size_t count, nk_i64_t *result) {
    __m512i ones_i16x32 = _mm512_set1_epi16(1);
    __m512i sum_i32x16 = _mm512_setzero_si512();
    nk_size_t idx = 0;

    for (; idx + 32 <= count; idx += 32) {
        __m512i data_i16x32 = _mm512_cvtepi8_epi16(_mm256_loadu_si256((__m256i const *)(data + idx)));
        sum_i32x16 = _mm512_dpwssd_epi32(sum_i32x16, data_i16x32, ones_i16x32);
    }

    if (idx < count) {
        __mmask32 mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, count - idx);
        __m512i data_i16x32 = _mm512_cvtepi8_epi16(_mm256_maskz_loadu_epi8(mask, data + idx));
        sum_i32x16 = _mm512_dpwssd_epi32(sum_i32x16, data_i16x32, ones_i16x32);
    }

    *result = _mm512_reduce_add_epi32(sum_i32x16);
}

NK_INTERNAL void nk_reduce_add_i8_icelake_strided_(                  //
    nk_i8_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_i64_t *result) {
    __mmask64 stride_mask_m64 = nk_stride_mask_u1x64_icelake_(stride_elements);
    __m512i ones_i16x32 = _mm512_set1_epi16(1);
    __m512i sum_i32x16 = _mm512_setzero_si512();
    nk_size_t idx_scalars = 0;
    nk_size_t total_scalars = count * stride_elements;

    for (; idx_scalars + 64 <= total_scalars; idx_scalars += 64) {
        __m512i data_i8x64 = _mm512_maskz_loadu_epi8(stride_mask_m64, data + idx_scalars);
        __m256i lo_i8x32 = _mm512_castsi512_si256(data_i8x64);
        __m256i hi_i8x32 = _mm512_extracti64x4_epi64(data_i8x64, 1);
        __m512i lo_i16x32 = _mm512_cvtepi8_epi16(lo_i8x32);
        __m512i hi_i16x32 = _mm512_cvtepi8_epi16(hi_i8x32);
        sum_i32x16 = _mm512_dpwssd_epi32(sum_i32x16, lo_i16x32, ones_i16x32);
        sum_i32x16 = _mm512_dpwssd_epi32(sum_i32x16, hi_i16x32, ones_i16x32);
    }

    // Handle tail with masked SIMD instead of scalar loop
    if (idx_scalars < total_scalars) {
        nk_size_t remaining_scalars = total_scalars - idx_scalars;
        __mmask64 tail_mask = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFFull, remaining_scalars);
        __mmask64 combined_mask = _kand_mask64(stride_mask_m64, tail_mask);
        __m512i data_i8x64 = _mm512_maskz_loadu_epi8(combined_mask, data + idx_scalars);
        __m256i lo_i8x32 = _mm512_castsi512_si256(data_i8x64);
        __m256i hi_i8x32 = _mm512_extracti64x4_epi64(data_i8x64, 1);
        __m512i lo_i16x32 = _mm512_cvtepi8_epi16(lo_i8x32);
        __m512i hi_i16x32 = _mm512_cvtepi8_epi16(hi_i8x32);
        sum_i32x16 = _mm512_dpwssd_epi32(sum_i32x16, lo_i16x32, ones_i16x32);
        sum_i32x16 = _mm512_dpwssd_epi32(sum_i32x16, hi_i16x32, ones_i16x32);
    }

    nk_i64_t sum = _mm512_reduce_add_epi32(sum_i32x16);
    *result = sum;
}

NK_PUBLIC void nk_reduce_add_i8_icelake(                          //
    nk_i8_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *result) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_i8_t);
    int aligned = (stride_bytes % sizeof(nk_i8_t) == 0);
    if (!aligned) nk_reduce_add_i8_serial(data, count, stride_bytes, result);
    else if (stride_elements == 1) nk_reduce_add_i8_icelake_contiguous_(data, count, result);
    else if (stride_elements <= 8) nk_reduce_add_i8_icelake_strided_(data, count, stride_elements, result);
    else nk_reduce_add_i8_serial(data, count, stride_bytes, result);
}

NK_INTERNAL void nk_reduce_add_u8_icelake_contiguous_( //
    nk_u8_t const *data, nk_size_t count, nk_u64_t *result) {
    __m512i ones_i16x32 = _mm512_set1_epi16(1);
    __m512i zeros_i8x64 = _mm512_setzero_si512();
    __m512i sum_i32x16 = _mm512_setzero_si512();
    nk_size_t idx = 0;

    for (; idx + 64 <= count; idx += 64) {
        __m512i data_u8x64 = _mm512_loadu_si512(data + idx);
        __m512i lo_i16x32 = _mm512_unpacklo_epi8(data_u8x64, zeros_i8x64);
        __m512i hi_i16x32 = _mm512_unpackhi_epi8(data_u8x64, zeros_i8x64);
        sum_i32x16 = _mm512_dpwssd_epi32(sum_i32x16, lo_i16x32, ones_i16x32);
        sum_i32x16 = _mm512_dpwssd_epi32(sum_i32x16, hi_i16x32, ones_i16x32);
    }

    if (idx + 32 <= count) {
        __m512i data_i16x32 = _mm512_cvtepu8_epi16(_mm256_loadu_si256((__m256i const *)(data + idx)));
        sum_i32x16 = _mm512_dpwssd_epi32(sum_i32x16, data_i16x32, ones_i16x32);
        idx += 32;
    }

    if (idx < count) {
        __mmask32 mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, count - idx);
        __m512i data_i16x32 = _mm512_cvtepu8_epi16(_mm256_maskz_loadu_epi8(mask, data + idx));
        sum_i32x16 = _mm512_dpwssd_epi32(sum_i32x16, data_i16x32, ones_i16x32);
    }

    *result = (nk_u64_t)_mm512_reduce_add_epi32(sum_i32x16);
}

NK_INTERNAL void nk_reduce_add_u8_icelake_strided_(                  //
    nk_u8_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_u64_t *result) {
    __mmask64 stride_mask_m64 = nk_stride_mask_u1x64_icelake_(stride_elements);
    __m512i ones_i16x32 = _mm512_set1_epi16(1);
    __m512i sum_i32x16 = _mm512_setzero_si512();
    nk_size_t idx_scalars = 0;
    nk_size_t total_scalars = count * stride_elements;

    for (; idx_scalars + 64 <= total_scalars; idx_scalars += 64) {
        __m512i data_u8x64 = _mm512_maskz_loadu_epi8(stride_mask_m64, data + idx_scalars);
        __m256i lo_u8x32 = _mm512_castsi512_si256(data_u8x64);
        __m256i hi_u8x32 = _mm512_extracti64x4_epi64(data_u8x64, 1);
        __m512i lo_i16x32 = _mm512_cvtepu8_epi16(lo_u8x32);
        __m512i hi_i16x32 = _mm512_cvtepu8_epi16(hi_u8x32);
        sum_i32x16 = _mm512_dpwssd_epi32(sum_i32x16, lo_i16x32, ones_i16x32);
        sum_i32x16 = _mm512_dpwssd_epi32(sum_i32x16, hi_i16x32, ones_i16x32);
    }

    // Handle tail with masked SIMD instead of scalar loop
    if (idx_scalars < total_scalars) {
        nk_size_t remaining_scalars = total_scalars - idx_scalars;
        __mmask64 tail_mask = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFFull, remaining_scalars);
        __mmask64 combined_mask = _kand_mask64(stride_mask_m64, tail_mask);
        __m512i data_u8x64 = _mm512_maskz_loadu_epi8(combined_mask, data + idx_scalars);
        __m256i lo_u8x32 = _mm512_castsi512_si256(data_u8x64);
        __m256i hi_u8x32 = _mm512_extracti64x4_epi64(data_u8x64, 1);
        __m512i lo_i16x32 = _mm512_cvtepu8_epi16(lo_u8x32);
        __m512i hi_i16x32 = _mm512_cvtepu8_epi16(hi_u8x32);
        sum_i32x16 = _mm512_dpwssd_epi32(sum_i32x16, lo_i16x32, ones_i16x32);
        sum_i32x16 = _mm512_dpwssd_epi32(sum_i32x16, hi_i16x32, ones_i16x32);
    }

    nk_u64_t sum = (nk_u64_t)_mm512_reduce_add_epi32(sum_i32x16);
    *result = sum;
}

NK_PUBLIC void nk_reduce_add_u8_icelake(                          //
    nk_u8_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *result) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_u8_t);
    int aligned = (stride_bytes % sizeof(nk_u8_t) == 0);
    if (!aligned) nk_reduce_add_u8_serial(data, count, stride_bytes, result);
    else if (stride_elements == 1) nk_reduce_add_u8_icelake_contiguous_(data, count, result);
    else if (stride_elements <= 8) nk_reduce_add_u8_icelake_strided_(data, count, stride_elements, result);
    else nk_reduce_add_u8_serial(data, count, stride_bytes, result);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_ICELAKE
#endif // NK_TARGET_X86_
#endif // NK_REDUCE_ICELAKE_H

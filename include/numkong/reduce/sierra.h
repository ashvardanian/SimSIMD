/**
 *  @brief SIMD-accelerated Vector Reductions for Sierra Forest.
 *  @file include/numkong/reduce/sierra.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 *
 *  @sa include/numkong/reduce.h
 *
 *  Uses AVX-VNNI (256-bit) for efficient widening dot-products:
 *  - `_mm256_dpwssd_epi32`: i16 × i16 → i32 accumulation (sum via dot with ones)
 *  - `_mm256_dpbssd_epi32`: i8 × i8 → i32 signed dot product
 */
#ifndef NK_REDUCE_SIERRA_H
#define NK_REDUCE_SIERRA_H

#if NK_TARGET_X86_
#if NK_TARGET_SIERRA

#include "numkong/types.h"
#include "numkong/reduce/haswell.h"
#include "numkong/reduce/serial.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,f16c,fma,bmi,bmi2,avxvnni,avxvnniint8"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "f16c", "fma", "bmi", "bmi2", "avxvnni", "avxvnniint8")
#endif

NK_INTERNAL void nk_reduce_add_i16_sierra_contiguous_( //
    nk_i16_t const *data, nk_size_t count, nk_i64_t *result) {
    __m256i ones_i16x16 = _mm256_set1_epi16(1);
    nk_i64_t sum_i64 = 0;
    nk_size_t idx = 0;
    nk_size_t const chunk_elements = 1024 * 16;

    while (idx + chunk_elements <= count) {
        __m256i sum_i32x8 = _mm256_setzero_si256();
        for (nk_size_t end = idx + chunk_elements; idx < end; idx += 16) {
            __m256i data_i16x16 = _mm256_loadu_si256((__m256i const *)(data + idx));
            sum_i32x8 = _mm256_dpwssd_epi32(sum_i32x8, data_i16x16, ones_i16x16);
        }
        sum_i64 += nk_reduce_add_i32x8_haswell_(sum_i32x8);
    }

    __m256i sum_i32x8 = _mm256_setzero_si256();
    for (; idx + 16 <= count; idx += 16) {
        __m256i data_i16x16 = _mm256_loadu_si256((__m256i const *)(data + idx));
        sum_i32x8 = _mm256_dpwssd_epi32(sum_i32x8, data_i16x16, ones_i16x16);
    }
    sum_i64 += nk_reduce_add_i32x8_haswell_(sum_i32x8);
    for (; idx < count; ++idx) sum_i64 += data[idx];
    *result = sum_i64;
}

NK_INTERNAL void nk_reduce_add_i16_sierra_strided_( //
    nk_i16_t const *data, nk_size_t count, nk_size_t stride_elements, nk_i64_t *result) {
    __m256i stride_mask_i16x16 = nk_stride_blend_b16x16_(stride_elements);
    __m256i ones_i16x16 = _mm256_set1_epi16(1);
    nk_i64_t sum_i64 = 0;
    nk_size_t idx_scalars = 0;
    nk_size_t total_scalars = count * stride_elements;
    nk_size_t const chunk_scalars = 1024 * 16;

    while (idx_scalars + chunk_scalars <= total_scalars) {
        __m256i sum_i32x8 = _mm256_setzero_si256();
        for (nk_size_t end = idx_scalars + chunk_scalars; idx_scalars < end; idx_scalars += 16) {
            __m256i data_i16x16 = _mm256_loadu_si256((__m256i const *)(data + idx_scalars));
            data_i16x16 = _mm256_and_si256(data_i16x16, stride_mask_i16x16);
            sum_i32x8 = _mm256_dpwssd_epi32(sum_i32x8, data_i16x16, ones_i16x16);
        }
        sum_i64 += nk_reduce_add_i32x8_haswell_(sum_i32x8);
    }

    __m256i sum_i32x8 = _mm256_setzero_si256();
    for (; idx_scalars + 16 <= total_scalars; idx_scalars += 16) {
        __m256i data_i16x16 = _mm256_loadu_si256((__m256i const *)(data + idx_scalars));
        data_i16x16 = _mm256_and_si256(data_i16x16, stride_mask_i16x16);
        sum_i32x8 = _mm256_dpwssd_epi32(sum_i32x8, data_i16x16, ones_i16x16);
    }
    sum_i64 += nk_reduce_add_i32x8_haswell_(sum_i32x8);
    for (nk_size_t idx = idx_scalars / stride_elements; idx < count; ++idx) sum_i64 += data[idx * stride_elements];
    *result = sum_i64;
}

NK_PUBLIC void nk_reduce_add_i16_sierra( //
    nk_i16_t const *data, nk_size_t count, nk_size_t stride_bytes, nk_i64_t *result) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_i16_t);
    int aligned = (stride_bytes % sizeof(nk_i16_t) == 0);
    if (!aligned) nk_reduce_add_i16_serial(data, count, stride_bytes, result);
    else if (stride_elements == 1) nk_reduce_add_i16_sierra_contiguous_(data, count, result);
    else if (stride_elements <= 8) nk_reduce_add_i16_sierra_strided_(data, count, stride_elements, result);
    else nk_reduce_add_i16_serial(data, count, stride_bytes, result);
}

NK_INTERNAL void nk_reduce_add_u16_sierra_contiguous_( //
    nk_u16_t const *data, nk_size_t count, nk_u64_t *result) {
    __m256i ones_i16x16 = _mm256_set1_epi16(1);
    nk_u64_t sum_u64 = 0;
    nk_size_t idx = 0;
    nk_size_t const chunk_elements = 1024 * 16;

    while (idx + chunk_elements <= count) {
        __m256i sum_i32x8 = _mm256_setzero_si256();
        for (nk_size_t end = idx + chunk_elements; idx < end; idx += 16) {
            __m256i data_u16x16 = _mm256_loadu_si256((__m256i const *)(data + idx));
            sum_i32x8 = _mm256_dpwssd_epi32(sum_i32x8, data_u16x16, ones_i16x16);
        }
        sum_u64 += (nk_u64_t)nk_reduce_add_i32x8_haswell_(sum_i32x8);
    }

    __m256i sum_i32x8 = _mm256_setzero_si256();
    for (; idx + 16 <= count; idx += 16) {
        __m256i data_u16x16 = _mm256_loadu_si256((__m256i const *)(data + idx));
        sum_i32x8 = _mm256_dpwssd_epi32(sum_i32x8, data_u16x16, ones_i16x16);
    }
    sum_u64 += (nk_u64_t)nk_reduce_add_i32x8_haswell_(sum_i32x8);
    for (; idx < count; ++idx) sum_u64 += data[idx];
    *result = sum_u64;
}

NK_INTERNAL void nk_reduce_add_u16_sierra_strided_( //
    nk_u16_t const *data, nk_size_t count, nk_size_t stride_elements, nk_u64_t *result) {
    __m256i stride_mask_i16x16 = nk_stride_blend_b16x16_(stride_elements);
    __m256i ones_i16x16 = _mm256_set1_epi16(1);
    nk_u64_t sum_u64 = 0;
    nk_size_t idx_scalars = 0;
    nk_size_t total_scalars = count * stride_elements;
    nk_size_t const chunk_scalars = 1024 * 16;

    while (idx_scalars + chunk_scalars <= total_scalars) {
        __m256i sum_i32x8 = _mm256_setzero_si256();
        for (nk_size_t end = idx_scalars + chunk_scalars; idx_scalars < end; idx_scalars += 16) {
            __m256i data_u16x16 = _mm256_loadu_si256((__m256i const *)(data + idx_scalars));
            data_u16x16 = _mm256_and_si256(data_u16x16, stride_mask_i16x16);
            sum_i32x8 = _mm256_dpwssd_epi32(sum_i32x8, data_u16x16, ones_i16x16);
        }
        sum_u64 += (nk_u64_t)nk_reduce_add_i32x8_haswell_(sum_i32x8);
    }

    __m256i sum_i32x8 = _mm256_setzero_si256();
    for (; idx_scalars + 16 <= total_scalars; idx_scalars += 16) {
        __m256i data_u16x16 = _mm256_loadu_si256((__m256i const *)(data + idx_scalars));
        data_u16x16 = _mm256_and_si256(data_u16x16, stride_mask_i16x16);
        sum_i32x8 = _mm256_dpwssd_epi32(sum_i32x8, data_u16x16, ones_i16x16);
    }
    sum_u64 += (nk_u64_t)nk_reduce_add_i32x8_haswell_(sum_i32x8);
    for (nk_size_t idx = idx_scalars / stride_elements; idx < count; ++idx) sum_u64 += data[idx * stride_elements];
    *result = sum_u64;
}

NK_PUBLIC void nk_reduce_add_u16_sierra( //
    nk_u16_t const *data, nk_size_t count, nk_size_t stride_bytes, nk_u64_t *result) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_u16_t);
    int aligned = (stride_bytes % sizeof(nk_u16_t) == 0);
    if (!aligned) nk_reduce_add_u16_serial(data, count, stride_bytes, result);
    else if (stride_elements == 1) nk_reduce_add_u16_sierra_contiguous_(data, count, result);
    else if (stride_elements <= 8) nk_reduce_add_u16_sierra_strided_(data, count, stride_elements, result);
    else nk_reduce_add_u16_serial(data, count, stride_bytes, result);
}

NK_INTERNAL void nk_reduce_add_i8_sierra_contiguous_( //
    nk_i8_t const *data, nk_size_t count, nk_i64_t *result) {
    __m256i ones_i8x32 = _mm256_set1_epi8(1);
    nk_i64_t sum_i64 = 0;
    nk_size_t idx = 0;
    nk_size_t const chunk_elements = 1024 * 32;

    while (idx + chunk_elements <= count) {
        __m256i sum_i32x8 = _mm256_setzero_si256();
        for (nk_size_t end = idx + chunk_elements; idx < end; idx += 32) {
            __m256i data_i8x32 = _mm256_loadu_si256((__m256i const *)(data + idx));
            sum_i32x8 = _mm256_dpbssd_epi32(sum_i32x8, data_i8x32, ones_i8x32);
        }
        sum_i64 += nk_reduce_add_i32x8_haswell_(sum_i32x8);
    }

    __m256i sum_i32x8 = _mm256_setzero_si256();
    for (; idx + 32 <= count; idx += 32) {
        __m256i data_i8x32 = _mm256_loadu_si256((__m256i const *)(data + idx));
        sum_i32x8 = _mm256_dpbssd_epi32(sum_i32x8, data_i8x32, ones_i8x32);
    }
    sum_i64 += nk_reduce_add_i32x8_haswell_(sum_i32x8);
    for (; idx < count; ++idx) sum_i64 += data[idx];
    *result = sum_i64;
}

NK_INTERNAL void nk_reduce_add_i8_sierra_strided_( //
    nk_i8_t const *data, nk_size_t count, nk_size_t stride_elements, nk_i64_t *result) {
    __m256i stride_mask_i8x32 = nk_stride_blend_u1x32_(stride_elements);
    __m256i ones_i8x32 = _mm256_set1_epi8(1);
    nk_i64_t sum_i64 = 0;
    nk_size_t idx_scalars = 0;
    nk_size_t total_scalars = count * stride_elements;
    nk_size_t const chunk_scalars = 1024 * 32;

    while (idx_scalars + chunk_scalars <= total_scalars) {
        __m256i sum_i32x8 = _mm256_setzero_si256();
        for (nk_size_t end = idx_scalars + chunk_scalars; idx_scalars < end; idx_scalars += 32) {
            __m256i data_i8x32 = _mm256_loadu_si256((__m256i const *)(data + idx_scalars));
            data_i8x32 = _mm256_and_si256(data_i8x32, stride_mask_i8x32);
            sum_i32x8 = _mm256_dpbssd_epi32(sum_i32x8, data_i8x32, ones_i8x32);
        }
        sum_i64 += nk_reduce_add_i32x8_haswell_(sum_i32x8);
    }

    __m256i sum_i32x8 = _mm256_setzero_si256();
    for (; idx_scalars + 32 <= total_scalars; idx_scalars += 32) {
        __m256i data_i8x32 = _mm256_loadu_si256((__m256i const *)(data + idx_scalars));
        data_i8x32 = _mm256_and_si256(data_i8x32, stride_mask_i8x32);
        sum_i32x8 = _mm256_dpbssd_epi32(sum_i32x8, data_i8x32, ones_i8x32);
    }
    sum_i64 += nk_reduce_add_i32x8_haswell_(sum_i32x8);
    for (nk_size_t idx = idx_scalars / stride_elements; idx < count; ++idx) sum_i64 += data[idx * stride_elements];
    *result = sum_i64;
}

NK_PUBLIC void nk_reduce_add_i8_sierra( //
    nk_i8_t const *data, nk_size_t count, nk_size_t stride_bytes, nk_i64_t *result) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_i8_t);
    int aligned = (stride_bytes % sizeof(nk_i8_t) == 0);
    if (!aligned) nk_reduce_add_i8_serial(data, count, stride_bytes, result);
    else if (stride_elements == 1) nk_reduce_add_i8_sierra_contiguous_(data, count, result);
    else if (stride_elements <= 8) nk_reduce_add_i8_sierra_strided_(data, count, stride_elements, result);
    else nk_reduce_add_i8_serial(data, count, stride_bytes, result);
}

NK_INTERNAL void nk_reduce_add_u8_sierra_contiguous_( //
    nk_u8_t const *data, nk_size_t count, nk_u64_t *result) {
    __m256i ones_i8x32 = _mm256_set1_epi8(1);
    nk_u64_t sum_u64 = 0;
    nk_size_t idx = 0;
    nk_size_t const chunk_elements = 1024 * 32;

    while (idx + chunk_elements <= count) {
        __m256i sum_i32x8 = _mm256_setzero_si256();
        for (nk_size_t end = idx + chunk_elements; idx < end; idx += 32) {
            __m256i data_u8x32 = _mm256_loadu_si256((__m256i const *)(data + idx));
            sum_i32x8 = _mm256_dpbusd_epi32(sum_i32x8, data_u8x32, ones_i8x32);
        }
        sum_u64 += (nk_u64_t)(nk_u32_t)nk_reduce_add_i32x8_haswell_(sum_i32x8);
    }

    __m256i sum_i32x8 = _mm256_setzero_si256();
    for (; idx + 32 <= count; idx += 32) {
        __m256i data_u8x32 = _mm256_loadu_si256((__m256i const *)(data + idx));
        sum_i32x8 = _mm256_dpbusd_epi32(sum_i32x8, data_u8x32, ones_i8x32);
    }
    sum_u64 += (nk_u64_t)(nk_u32_t)nk_reduce_add_i32x8_haswell_(sum_i32x8);
    for (; idx < count; ++idx) sum_u64 += data[idx];
    *result = sum_u64;
}

NK_INTERNAL void nk_reduce_add_u8_sierra_strided_( //
    nk_u8_t const *data, nk_size_t count, nk_size_t stride_elements, nk_u64_t *result) {
    __m256i stride_mask_i8x32 = nk_stride_blend_u1x32_(stride_elements);
    __m256i ones_i8x32 = _mm256_set1_epi8(1);
    nk_u64_t sum_u64 = 0;
    nk_size_t idx_scalars = 0;
    nk_size_t total_scalars = count * stride_elements;
    nk_size_t const chunk_scalars = 1024 * 32;

    while (idx_scalars + chunk_scalars <= total_scalars) {
        __m256i sum_i32x8 = _mm256_setzero_si256();
        for (nk_size_t end = idx_scalars + chunk_scalars; idx_scalars < end; idx_scalars += 32) {
            __m256i data_u8x32 = _mm256_loadu_si256((__m256i const *)(data + idx_scalars));
            data_u8x32 = _mm256_and_si256(data_u8x32, stride_mask_i8x32);
            sum_i32x8 = _mm256_dpbusd_epi32(sum_i32x8, data_u8x32, ones_i8x32);
        }
        sum_u64 += (nk_u64_t)(nk_u32_t)nk_reduce_add_i32x8_haswell_(sum_i32x8);
    }

    __m256i sum_i32x8 = _mm256_setzero_si256();
    for (; idx_scalars + 32 <= total_scalars; idx_scalars += 32) {
        __m256i data_u8x32 = _mm256_loadu_si256((__m256i const *)(data + idx_scalars));
        data_u8x32 = _mm256_and_si256(data_u8x32, stride_mask_i8x32);
        sum_i32x8 = _mm256_dpbusd_epi32(sum_i32x8, data_u8x32, ones_i8x32);
    }
    sum_u64 += (nk_u64_t)(nk_u32_t)nk_reduce_add_i32x8_haswell_(sum_i32x8);
    for (nk_size_t idx = idx_scalars / stride_elements; idx < count; ++idx) sum_u64 += data[idx * stride_elements];
    *result = sum_u64;
}

NK_PUBLIC void nk_reduce_add_u8_sierra( //
    nk_u8_t const *data, nk_size_t count, nk_size_t stride_bytes, nk_u64_t *result) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_u8_t);
    int aligned = (stride_bytes % sizeof(nk_u8_t) == 0);
    if (!aligned) nk_reduce_add_u8_serial(data, count, stride_bytes, result);
    else if (stride_elements == 1) nk_reduce_add_u8_sierra_contiguous_(data, count, result);
    else if (stride_elements <= 8) nk_reduce_add_u8_sierra_strided_(data, count, stride_elements, result);
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

#endif // NK_TARGET_SIERRA
#endif // NK_TARGET_X86_
#endif // NK_REDUCE_SIERRA_H

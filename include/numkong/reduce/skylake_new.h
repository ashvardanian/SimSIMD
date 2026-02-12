/**
 *  @brief AVX-512 implementations for the redesigned reduction API (moments + minmax).
 *  @file include/numkong/reduce/skylake_new.h
 *  @author Ash Vardanian
 *  @date February 11, 2026
 *
 *  @sa include/numkong/reduce.h
 *
 *  @section tail_nan_fill  Tail Handling via NaN Fill
 *
 *  In floating-point minmax contiguous kernels (f32, f64), the tail block fills
 *  unloaded lanes with NaN via `_mm512_mask_loadu_ps(nan, mask, ptr)` instead of
 *  `_mm512_maskz_loadu_ps(mask, ptr)`.  This allows the subsequent `_CMP_LT_OQ` /
 *  `_CMP_GT_OQ` comparisons to run without the tail-load mask predicate, because
 *  IEEE-754 ordered-quiet comparisons return false for NaN operands.
 */
#ifndef NK_REDUCE_SKYLAKE_NEW_H
#define NK_REDUCE_SKYLAKE_NEW_H

#if NK_TARGET_X86_
#if NK_TARGET_SKYLAKE

#include "numkong/types.h"
#include "numkong/reduce/serial_new.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,avx512f,avx512vl,avx512bw,avx512dq,f16c,fma,bmi,bmi2"))), \
                             apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "avx512bw", "avx512dq", "f16c", "fma", "bmi", "bmi2")
#endif

NK_INTERNAL void nk_reduce_moments_f32_skylake_contiguous_( //
    nk_f32_t const *data, nk_size_t count,                  //
    nk_f64_t *sum, nk_f64_t *sumsq) {
    __m512d sum_low_f64x8 = _mm512_setzero_pd(), sum_high_f64x8 = _mm512_setzero_pd();
    __m512d sumsq_low_f64x8 = _mm512_setzero_pd(), sumsq_high_f64x8 = _mm512_setzero_pd();
    nk_size_t idx = 0;
    for (; idx + 16 <= count; idx += 16) {
        __m512d low_f64x8 = _mm512_cvtps_pd(_mm256_loadu_ps(data + idx));
        __m512d high_f64x8 = _mm512_cvtps_pd(_mm256_loadu_ps(data + idx + 8));
        sum_low_f64x8 = _mm512_add_pd(sum_low_f64x8, low_f64x8);
        sum_high_f64x8 = _mm512_add_pd(sum_high_f64x8, high_f64x8);
        sumsq_low_f64x8 = _mm512_fmadd_pd(low_f64x8, low_f64x8, sumsq_low_f64x8);
        sumsq_high_f64x8 = _mm512_fmadd_pd(high_f64x8, high_f64x8, sumsq_high_f64x8);
    }
    __m512d sum_f64x8 = _mm512_add_pd(sum_low_f64x8, sum_high_f64x8);
    __m512d sumsq_f64x8 = _mm512_add_pd(sumsq_low_f64x8, sumsq_high_f64x8);
    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        __mmask16 tail_mask = (__mmask16)_bzhi_u32(0xFFFF, (unsigned int)remaining);
        __m512 tail_f32x16 = _mm512_maskz_loadu_ps(tail_mask, data + idx);
        __m512d low_f64x8 = _mm512_cvtps_pd(_mm512_castps512_ps256(tail_f32x16));
        __m512d high_f64x8 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(tail_f32x16, 1));
        sum_f64x8 = _mm512_add_pd(sum_f64x8, low_f64x8);
        sumsq_f64x8 = _mm512_fmadd_pd(low_f64x8, low_f64x8, sumsq_f64x8);
        if (remaining > 8)
            sum_f64x8 = _mm512_add_pd(sum_f64x8, high_f64x8),
            sumsq_f64x8 = _mm512_fmadd_pd(high_f64x8, high_f64x8, sumsq_f64x8);
    }
    *sum = nk_reduce_add_f64x8_skylake_(sum_f64x8);
    *sumsq = nk_reduce_add_f64x8_skylake_(sumsq_f64x8);
}

NK_INTERNAL void nk_reduce_moments_f32_skylake_gather_(            //
    nk_f32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f64_t *sum, nk_f64_t *sumsq) {
    nk_i32_t stride_elements = (nk_i32_t)(stride_bytes / sizeof(nk_f32_t));
    __m512i indices_i32x16 = _mm512_mullo_epi32(_mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
                                                _mm512_set1_epi32(stride_elements));
    __m512d sum_f64x8 = _mm512_setzero_pd();
    __m512d sumsq_f64x8 = _mm512_setzero_pd();
    nk_size_t idx = 0;
    for (; idx + 16 <= count; idx += 16) {
        __m512 gathered_f32x16 = _mm512_i32gather_ps(indices_i32x16, data + idx * stride_elements, sizeof(nk_f32_t));
        __m256 lo_f32x8 = _mm512_castps512_ps256(gathered_f32x16);
        __m256 hi_f32x8 = _mm512_extractf32x8_ps(gathered_f32x16, 1);
        __m512d lo_f64x8 = _mm512_cvtps_pd(lo_f32x8);
        __m512d hi_f64x8 = _mm512_cvtps_pd(hi_f32x8);
        sum_f64x8 = _mm512_add_pd(sum_f64x8, lo_f64x8);
        sum_f64x8 = _mm512_add_pd(sum_f64x8, hi_f64x8);
        sumsq_f64x8 = _mm512_fmadd_pd(lo_f64x8, lo_f64x8, sumsq_f64x8);
        sumsq_f64x8 = _mm512_fmadd_pd(hi_f64x8, hi_f64x8, sumsq_f64x8);
    }
    nk_f64_t s = nk_reduce_add_f64x8_skylake_(sum_f64x8);
    nk_f64_t sq = nk_reduce_add_f64x8_skylake_(sumsq_f64x8);
    unsigned char const *ptr = (unsigned char const *)(data + idx * stride_elements);
    for (; idx < count; ++idx, ptr += stride_bytes) {
        nk_f64_t val = (nk_f64_t)(*(nk_f32_t const *)ptr);
        s += val;
        sq += val * val;
    }
    *sum = s;
    *sumsq = sq;
}

NK_INTERNAL void nk_reduce_moments_f32_skylake_strided_(              //
    nk_f32_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_f64_t *sum, nk_f64_t *sumsq) {
    __mmask16 stride_mask = nk_stride_mask_b32x16_(stride_elements);
    __m512d sum_f64x8 = _mm512_setzero_pd(), sumsq_f64x8 = _mm512_setzero_pd();
    nk_size_t idx = 0, total = count * stride_elements;
    for (; idx + 16 <= total; idx += 16) {
        __m512 data_f32x16 = _mm512_maskz_loadu_ps(stride_mask, data + idx);
        __m512d low_f64x8 = _mm512_cvtps_pd(_mm512_castps512_ps256(data_f32x16));
        __m512d high_f64x8 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(data_f32x16, 1));
        sum_f64x8 = _mm512_add_pd(sum_f64x8, low_f64x8);
        sum_f64x8 = _mm512_add_pd(sum_f64x8, high_f64x8);
        sumsq_f64x8 = _mm512_fmadd_pd(low_f64x8, low_f64x8, sumsq_f64x8);
        sumsq_f64x8 = _mm512_fmadd_pd(high_f64x8, high_f64x8, sumsq_f64x8);
    }
    nk_size_t remaining = total - idx;
    if (remaining > 0) {
        __mmask16 tail_mask = (__mmask16)_bzhi_u32(0xFFFF, (unsigned int)remaining);
        __m512 data_f32x16 = _mm512_maskz_loadu_ps(stride_mask & tail_mask, data + idx);
        __m512d low_f64x8 = _mm512_cvtps_pd(_mm512_castps512_ps256(data_f32x16));
        __m512d high_f64x8 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(data_f32x16, 1));
        sum_f64x8 = _mm512_add_pd(sum_f64x8, low_f64x8);
        sumsq_f64x8 = _mm512_fmadd_pd(low_f64x8, low_f64x8, sumsq_f64x8);
        if (remaining > 8)
            sum_f64x8 = _mm512_add_pd(sum_f64x8, high_f64x8),
            sumsq_f64x8 = _mm512_fmadd_pd(high_f64x8, high_f64x8, sumsq_f64x8);
    }
    *sum = nk_reduce_add_f64x8_skylake_(sum_f64x8);
    *sumsq = nk_reduce_add_f64x8_skylake_(sumsq_f64x8);
}

NK_PUBLIC void nk_reduce_moments_f32_skylake(                      //
    nk_f32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f64_t *sum, nk_f64_t *sumsq) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_f32_t);
    int aligned = (stride_bytes % sizeof(nk_f32_t) == 0);
    if (count == 0) *sum = 0, *sumsq = 0;
    else if (!aligned) nk_reduce_moments_f32_serial(data, count, stride_bytes, sum, sumsq);
    else if (count > (nk_size_t)(NK_U16_MAX + 1) * 16) {
        nk_size_t left_count = count / 2;
        nk_f64_t left_sum, left_sumsq, right_sum, right_sumsq;
        nk_reduce_moments_f32_skylake(data, left_count, stride_bytes, &left_sum, &left_sumsq);
        nk_reduce_moments_f32_skylake(data + left_count * stride_elements, count - left_count, stride_bytes, &right_sum,
                                      &right_sumsq);
        *sum = left_sum + right_sum, *sumsq = left_sumsq + right_sumsq;
    }
    else if (stride_elements == 1) nk_reduce_moments_f32_skylake_contiguous_(data, count, sum, sumsq);
    else if (stride_elements <= 8) nk_reduce_moments_f32_skylake_strided_(data, count, stride_elements, sum, sumsq);
    else nk_reduce_moments_f32_skylake_gather_(data, count, stride_bytes, sum, sumsq);
}

NK_INTERNAL void nk_reduce_minmax_f32_skylake_contiguous_( //
    nk_f32_t const *data, nk_size_t count,                 //
    nk_f32_t *min_value, nk_size_t *min_index,             //
    nk_f32_t *max_value, nk_size_t *max_index) {
    __m512 min_f32x16 = _mm512_loadu_ps(data);
    __m512 max_f32x16 = min_f32x16;
    __m512i min_iter_u32x16 = _mm512_setzero_si512();
    __m512i max_iter_u32x16 = _mm512_setzero_si512();
    __m512i iter_u32x16 = _mm512_set1_epi32(1);
    __m512i one_u32x16 = _mm512_set1_epi32(1);

    nk_size_t idx = 16;
    for (; idx + 16 <= count; idx += 16) {
        __m512 data_f32x16 = _mm512_loadu_ps(data + idx);
        __mmask16 min_changed_mask = _mm512_cmp_ps_mask(data_f32x16, min_f32x16, _CMP_LT_OQ);
        __mmask16 max_changed_mask = _mm512_cmp_ps_mask(data_f32x16, max_f32x16, _CMP_GT_OQ);
        min_f32x16 = _mm512_mask_mov_ps(min_f32x16, min_changed_mask, data_f32x16);
        max_f32x16 = _mm512_mask_mov_ps(max_f32x16, max_changed_mask, data_f32x16);
        min_iter_u32x16 = _mm512_mask_mov_epi32(min_iter_u32x16, min_changed_mask, iter_u32x16);
        max_iter_u32x16 = _mm512_mask_mov_epi32(max_iter_u32x16, max_changed_mask, iter_u32x16);
        iter_u32x16 = _mm512_add_epi32(iter_u32x16, one_u32x16);
    }

    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        __mmask16 tail_load = (__mmask16)_bzhi_u32(0xFFFF, (unsigned int)remaining);
        __m512 tail_f32x16 = _mm512_maskz_loadu_ps(tail_load, data + idx);
        __mmask16 min_changed_mask = _mm512_mask_cmp_ps_mask(tail_load, tail_f32x16, min_f32x16, _CMP_LT_OQ);
        __mmask16 max_changed_mask = _mm512_mask_cmp_ps_mask(tail_load, tail_f32x16, max_f32x16, _CMP_GT_OQ);
        min_f32x16 = _mm512_mask_mov_ps(min_f32x16, min_changed_mask, tail_f32x16);
        max_f32x16 = _mm512_mask_mov_ps(max_f32x16, max_changed_mask, tail_f32x16);
        min_iter_u32x16 = _mm512_mask_mov_epi32(min_iter_u32x16, min_changed_mask, iter_u32x16);
        max_iter_u32x16 = _mm512_mask_mov_epi32(max_iter_u32x16, max_changed_mask, iter_u32x16);
    }

    nk_f32_t min_scalar = nk_reduce_min_f32x16_skylake_(min_f32x16);
    nk_f32_t max_scalar = nk_reduce_max_f32x16_skylake_(max_f32x16);
    __mmask16 min_equality_mask = _mm512_cmp_ps_mask(min_f32x16, _mm512_set1_ps(min_scalar), _CMP_EQ_OQ);
    __mmask16 max_equality_mask = _mm512_cmp_ps_mask(max_f32x16, _mm512_set1_ps(max_scalar), _CMP_EQ_OQ);
    unsigned int min_lane = _tzcnt_u32(min_equality_mask);
    unsigned int max_lane = _tzcnt_u32(max_equality_mask);
    nk_b512_vec_t iter_vec;
    iter_vec.zmm = min_iter_u32x16;
    *min_value = min_scalar, *min_index = (nk_size_t)iter_vec.u32s[min_lane] * 16 + min_lane;
    iter_vec.zmm = max_iter_u32x16;
    *max_value = max_scalar, *max_index = (nk_size_t)iter_vec.u32s[max_lane] * 16 + max_lane;
}

NK_PUBLIC void nk_reduce_minmax_f32_skylake(                       //
    nk_f32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *min_value, nk_size_t *min_index,                     //
    nk_f32_t *max_value, nk_size_t *max_index) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_f32_t);
    int aligned = (stride_bytes % sizeof(nk_f32_t) == 0);
    if (count == 0) *min_value = NK_F32_MAX, *min_index = 0, *max_value = NK_F32_MIN, *max_index = 0;
    else if (!aligned)
        nk_reduce_minmax_f32_serial(data, count, stride_bytes, min_value, min_index, max_value, max_index);
    else if (count > (nk_size_t)NK_U32_MAX * 16) {
        nk_size_t left_count = count / 2;
        nk_f32_t left_min, right_min, left_max, right_max;
        nk_size_t left_min_index, right_min_index, left_max_index, right_max_index;
        nk_reduce_minmax_f32_skylake(data, left_count, stride_bytes, &left_min, &left_min_index, &left_max,
                                     &left_max_index);
        nk_reduce_minmax_f32_skylake(data + left_count * stride_elements, count - left_count, stride_bytes, &right_min,
                                     &right_min_index, &right_max, &right_max_index);
        if (right_min < left_min) *min_value = right_min, *min_index = left_count + right_min_index;
        else *min_value = left_min, *min_index = left_min_index;
        if (right_max > left_max) *max_value = right_max, *max_index = left_count + right_max_index;
        else *max_value = left_max, *max_index = left_max_index;
    }
    else if (stride_elements == 1)
        nk_reduce_minmax_f32_skylake_contiguous_(data, count, min_value, min_index, max_value, max_index);
    else nk_reduce_minmax_f32_serial(data, count, stride_bytes, min_value, min_index, max_value, max_index);
}

NK_INTERNAL void nk_reduce_moments_f64_skylake_contiguous_( //
    nk_f64_t const *data, nk_size_t count,                  //
    nk_f64_t *sum, nk_f64_t *sumsq) {
    __m512d sum_f64x8 = _mm512_setzero_pd();
    __m512d sum_comp_f64x8 = _mm512_setzero_pd();
    __m512d sumsq_f64x8 = _mm512_setzero_pd();
    __m512d sumsq_comp_f64x8 = _mm512_setzero_pd();
    nk_size_t idx = 0;
    for (; idx + 8 <= count; idx += 8) {
        __m512d val_f64x8 = _mm512_loadu_pd(data + idx);
        // Knuth 2-SUM for sum
        __m512d tentative_f64x8 = _mm512_add_pd(sum_f64x8, val_f64x8);
        __m512d round_f64x8 = _mm512_sub_pd(tentative_f64x8, sum_f64x8);
        __m512d corr_f64x8 = _mm512_add_pd(_mm512_sub_pd(sum_f64x8, _mm512_sub_pd(tentative_f64x8, round_f64x8)),
                                           _mm512_sub_pd(val_f64x8, round_f64x8));
        sum_comp_f64x8 = _mm512_add_pd(sum_comp_f64x8, corr_f64x8);
        sum_f64x8 = tentative_f64x8;
        // Knuth 2-SUM for sumsq
        __m512d sq_f64x8 = _mm512_mul_pd(val_f64x8, val_f64x8);
        __m512d tentative_sq_f64x8 = _mm512_add_pd(sumsq_f64x8, sq_f64x8);
        __m512d round_sq_f64x8 = _mm512_sub_pd(tentative_sq_f64x8, sumsq_f64x8);
        __m512d corr_sq_f64x8 = _mm512_add_pd(
            _mm512_sub_pd(sumsq_f64x8, _mm512_sub_pd(tentative_sq_f64x8, round_sq_f64x8)),
            _mm512_sub_pd(sq_f64x8, round_sq_f64x8));
        sumsq_comp_f64x8 = _mm512_add_pd(sumsq_comp_f64x8, corr_sq_f64x8);
        sumsq_f64x8 = tentative_sq_f64x8;
    }
    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        __mmask8 tail_mask = (__mmask8)_bzhi_u32(0xFF, (unsigned int)remaining);
        __m512d val_f64x8 = _mm512_maskz_loadu_pd(tail_mask, data + idx);
        // Knuth 2-SUM for sum
        __m512d tentative_f64x8 = _mm512_add_pd(sum_f64x8, val_f64x8);
        __m512d round_f64x8 = _mm512_sub_pd(tentative_f64x8, sum_f64x8);
        __m512d corr_f64x8 = _mm512_add_pd(_mm512_sub_pd(sum_f64x8, _mm512_sub_pd(tentative_f64x8, round_f64x8)),
                                           _mm512_sub_pd(val_f64x8, round_f64x8));
        sum_comp_f64x8 = _mm512_add_pd(sum_comp_f64x8, corr_f64x8);
        sum_f64x8 = tentative_f64x8;
        // Knuth 2-SUM for sumsq
        __m512d sq_f64x8 = _mm512_mul_pd(val_f64x8, val_f64x8);
        __m512d tentative_sq_f64x8 = _mm512_add_pd(sumsq_f64x8, sq_f64x8);
        __m512d round_sq_f64x8 = _mm512_sub_pd(tentative_sq_f64x8, sumsq_f64x8);
        __m512d corr_sq_f64x8 = _mm512_add_pd(
            _mm512_sub_pd(sumsq_f64x8, _mm512_sub_pd(tentative_sq_f64x8, round_sq_f64x8)),
            _mm512_sub_pd(sq_f64x8, round_sq_f64x8));
        sumsq_comp_f64x8 = _mm512_add_pd(sumsq_comp_f64x8, corr_sq_f64x8);
        sumsq_f64x8 = tentative_sq_f64x8;
    }
    *sum = nk_reduce_add_f64x8_skylake_(_mm512_add_pd(sum_f64x8, sum_comp_f64x8));
    *sumsq = nk_reduce_add_f64x8_skylake_(_mm512_add_pd(sumsq_f64x8, sumsq_comp_f64x8));
}

NK_INTERNAL void nk_reduce_moments_f64_skylake_strided_(              //
    nk_f64_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_f64_t *sum, nk_f64_t *sumsq) {
    __mmask8 stride_mask = nk_stride_mask_b64x8_(stride_elements);
    __m512d sum_f64x8 = _mm512_setzero_pd();
    __m512d sum_comp_f64x8 = _mm512_setzero_pd();
    __m512d sumsq_f64x8 = _mm512_setzero_pd();
    __m512d sumsq_comp_f64x8 = _mm512_setzero_pd();
    nk_size_t idx = 0, total = count * stride_elements;
    for (; idx + 8 <= total; idx += 8) {
        __m512d val_f64x8 = _mm512_maskz_loadu_pd(stride_mask, data + idx);
        // Knuth 2-SUM for sum
        __m512d tentative_f64x8 = _mm512_add_pd(sum_f64x8, val_f64x8);
        __m512d round_f64x8 = _mm512_sub_pd(tentative_f64x8, sum_f64x8);
        __m512d corr_f64x8 = _mm512_add_pd(_mm512_sub_pd(sum_f64x8, _mm512_sub_pd(tentative_f64x8, round_f64x8)),
                                           _mm512_sub_pd(val_f64x8, round_f64x8));
        sum_comp_f64x8 = _mm512_add_pd(sum_comp_f64x8, corr_f64x8);
        sum_f64x8 = tentative_f64x8;
        // Knuth 2-SUM for sumsq
        __m512d sq_f64x8 = _mm512_mul_pd(val_f64x8, val_f64x8);
        __m512d tentative_sq_f64x8 = _mm512_add_pd(sumsq_f64x8, sq_f64x8);
        __m512d round_sq_f64x8 = _mm512_sub_pd(tentative_sq_f64x8, sumsq_f64x8);
        __m512d corr_sq_f64x8 = _mm512_add_pd(
            _mm512_sub_pd(sumsq_f64x8, _mm512_sub_pd(tentative_sq_f64x8, round_sq_f64x8)),
            _mm512_sub_pd(sq_f64x8, round_sq_f64x8));
        sumsq_comp_f64x8 = _mm512_add_pd(sumsq_comp_f64x8, corr_sq_f64x8);
        sumsq_f64x8 = tentative_sq_f64x8;
    }
    nk_size_t remaining = total - idx;
    if (remaining > 0) {
        __mmask8 tail_mask = stride_mask & (__mmask8)_bzhi_u32(0xFF, (unsigned int)remaining);
        __m512d val_f64x8 = _mm512_maskz_loadu_pd(tail_mask, data + idx);
        // Knuth 2-SUM for sum
        __m512d tentative_f64x8 = _mm512_add_pd(sum_f64x8, val_f64x8);
        __m512d round_f64x8 = _mm512_sub_pd(tentative_f64x8, sum_f64x8);
        __m512d corr_f64x8 = _mm512_add_pd(_mm512_sub_pd(sum_f64x8, _mm512_sub_pd(tentative_f64x8, round_f64x8)),
                                           _mm512_sub_pd(val_f64x8, round_f64x8));
        sum_comp_f64x8 = _mm512_add_pd(sum_comp_f64x8, corr_f64x8);
        sum_f64x8 = tentative_f64x8;
        // Knuth 2-SUM for sumsq
        __m512d sq_f64x8 = _mm512_mul_pd(val_f64x8, val_f64x8);
        __m512d tentative_sq_f64x8 = _mm512_add_pd(sumsq_f64x8, sq_f64x8);
        __m512d round_sq_f64x8 = _mm512_sub_pd(tentative_sq_f64x8, sumsq_f64x8);
        __m512d corr_sq_f64x8 = _mm512_add_pd(
            _mm512_sub_pd(sumsq_f64x8, _mm512_sub_pd(tentative_sq_f64x8, round_sq_f64x8)),
            _mm512_sub_pd(sq_f64x8, round_sq_f64x8));
        sumsq_comp_f64x8 = _mm512_add_pd(sumsq_comp_f64x8, corr_sq_f64x8);
        sumsq_f64x8 = tentative_sq_f64x8;
    }
    *sum = nk_reduce_add_f64x8_skylake_(_mm512_add_pd(sum_f64x8, sum_comp_f64x8));
    *sumsq = nk_reduce_add_f64x8_skylake_(_mm512_add_pd(sumsq_f64x8, sumsq_comp_f64x8));
}

NK_INTERNAL void nk_reduce_moments_f64_skylake_gather_(            //
    nk_f64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f64_t *sum, nk_f64_t *sumsq) {
    nk_i32_t stride_elements = (nk_i32_t)(stride_bytes / sizeof(nk_f64_t));
    __m256i indices_i32x8 = _mm256_mullo_epi32(_mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7),
                                               _mm256_set1_epi32(stride_elements));
    __m512d sum_f64x8 = _mm512_setzero_pd();
    __m512d sum_comp_f64x8 = _mm512_setzero_pd();
    __m512d sumsq_f64x8 = _mm512_setzero_pd();
    __m512d sumsq_comp_f64x8 = _mm512_setzero_pd();
    nk_size_t idx = 0;
    for (; idx + 8 <= count; idx += 8) {
        __m512d val_f64x8 = _mm512_i32gather_pd(indices_i32x8, data + idx * stride_elements, sizeof(nk_f64_t));
        // Knuth 2-SUM for sum
        __m512d tentative_f64x8 = _mm512_add_pd(sum_f64x8, val_f64x8);
        __m512d round_f64x8 = _mm512_sub_pd(tentative_f64x8, sum_f64x8);
        __m512d corr_f64x8 = _mm512_add_pd(_mm512_sub_pd(sum_f64x8, _mm512_sub_pd(tentative_f64x8, round_f64x8)),
                                           _mm512_sub_pd(val_f64x8, round_f64x8));
        sum_comp_f64x8 = _mm512_add_pd(sum_comp_f64x8, corr_f64x8);
        sum_f64x8 = tentative_f64x8;
        // Knuth 2-SUM for sumsq
        __m512d sq_f64x8 = _mm512_mul_pd(val_f64x8, val_f64x8);
        __m512d tentative_sq_f64x8 = _mm512_add_pd(sumsq_f64x8, sq_f64x8);
        __m512d round_sq_f64x8 = _mm512_sub_pd(tentative_sq_f64x8, sumsq_f64x8);
        __m512d corr_sq_f64x8 = _mm512_add_pd(
            _mm512_sub_pd(sumsq_f64x8, _mm512_sub_pd(tentative_sq_f64x8, round_sq_f64x8)),
            _mm512_sub_pd(sq_f64x8, round_sq_f64x8));
        sumsq_comp_f64x8 = _mm512_add_pd(sumsq_comp_f64x8, corr_sq_f64x8);
        sumsq_f64x8 = tentative_sq_f64x8;
    }
    nk_f64_t s = nk_reduce_add_f64x8_skylake_(_mm512_add_pd(sum_f64x8, sum_comp_f64x8));
    nk_f64_t sq = nk_reduce_add_f64x8_skylake_(_mm512_add_pd(sumsq_f64x8, sumsq_comp_f64x8));
    unsigned char const *ptr = (unsigned char const *)(data + idx * stride_elements);
    for (; idx < count; ++idx, ptr += stride_bytes) {
        nk_f64_t val = *(nk_f64_t const *)ptr;
        s += val;
        sq += val * val;
    }
    *sum = s;
    *sumsq = sq;
}

NK_PUBLIC void nk_reduce_moments_f64_skylake(                      //
    nk_f64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f64_t *sum, nk_f64_t *sumsq) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_f64_t);
    int aligned = (stride_bytes % sizeof(nk_f64_t) == 0);
    if (count == 0) *sum = 0, *sumsq = 0;
    else if (!aligned) nk_reduce_moments_f64_serial(data, count, stride_bytes, sum, sumsq);
    else if (count > (nk_size_t)(NK_U16_MAX + 1) * 8) {
        nk_size_t left_count = count / 2;
        nk_f64_t left_sum, left_sumsq, right_sum, right_sumsq;
        nk_reduce_moments_f64_skylake(data, left_count, stride_bytes, &left_sum, &left_sumsq);
        nk_reduce_moments_f64_skylake(data + left_count * stride_elements, count - left_count, stride_bytes, &right_sum,
                                      &right_sumsq);
        *sum = left_sum + right_sum, *sumsq = left_sumsq + right_sumsq;
    }
    else if (stride_elements == 1) nk_reduce_moments_f64_skylake_contiguous_(data, count, sum, sumsq);
    else if (stride_elements <= 8) nk_reduce_moments_f64_skylake_strided_(data, count, stride_elements, sum, sumsq);
    else nk_reduce_moments_f64_skylake_gather_(data, count, stride_bytes, sum, sumsq);
}

NK_INTERNAL void nk_reduce_moments_i8_skylake_contiguous_( //
    nk_i8_t const *data, nk_size_t count,                  //
    nk_i64_t *sum, nk_i64_t *sumsq) {
    // Sum: VPSADBW with XOR bias (same as nk_reduce_add_i8_skylake_contiguous_).
    // Sumsq: widen i8→i16, VPMADDWD(x,x) → i32 (pairs of squares), accumulate i32.
    // i32 overflow safe: max per lane = (128² + 128²) * 65536 iters ≈ 2.1B = safe limit.
    // The dispatch recurses at (NK_U16_MAX+1)*64 elements → at most 65536 iterations here.
    __m512i bias_i8x64 = _mm512_set1_epi8((char)0x80);
    __m512i zero_i8x64 = _mm512_setzero_si512();
    __m512i sum_u64x8 = _mm512_setzero_si512();
    __m512i sumsq_lo_i32x16 = _mm512_setzero_si512();
    __m512i sumsq_hi_i32x16 = _mm512_setzero_si512();
    nk_size_t idx = 0;
    for (; idx + 64 <= count; idx += 64) {
        __m512i data_i8x64 = _mm512_loadu_si512(data + idx);
        __m512i unsigned_i8x64 = _mm512_xor_si512(data_i8x64, bias_i8x64);
        sum_u64x8 = _mm512_add_epi64(sum_u64x8, _mm512_sad_epu8(unsigned_i8x64, zero_i8x64));
        __m512i lo_i16x32 = _mm512_cvtepi8_epi16(_mm512_castsi512_si256(data_i8x64));
        __m512i hi_i16x32 = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(data_i8x64, 1));
        sumsq_lo_i32x16 = _mm512_add_epi32(sumsq_lo_i32x16, _mm512_madd_epi16(lo_i16x32, lo_i16x32));
        sumsq_hi_i32x16 = _mm512_add_epi32(sumsq_hi_i32x16, _mm512_madd_epi16(hi_i16x32, hi_i16x32));
    }
    // Flush i32 → i64 once
    sumsq_lo_i32x16 = _mm512_add_epi32(sumsq_lo_i32x16, sumsq_hi_i32x16);
    __m512i sumsq_i64x8 = _mm512_cvtepi32_epi64(_mm512_castsi512_si256(sumsq_lo_i32x16));
    sumsq_i64x8 = _mm512_add_epi64(sumsq_i64x8, _mm512_cvtepi32_epi64(_mm512_extracti64x4_epi64(sumsq_lo_i32x16, 1)));
    nk_i64_t s = (nk_i64_t)nk_reduce_add_u64x8_skylake_(sum_u64x8);
    s -= (nk_i64_t)128 * (nk_i64_t)idx;
    nk_i64_t sq = nk_reduce_add_i64x8_skylake_(sumsq_i64x8);
    for (; idx < count; ++idx) {
        nk_i64_t val = (nk_i64_t)data[idx];
        s += val;
        sq += val * val;
    }
    *sum = s;
    *sumsq = sq;
}

NK_INTERNAL void nk_reduce_moments_i8_skylake_strided_(              //
    nk_i8_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_i64_t *sum, nk_i64_t *sumsq) {
    __mmask64 stride_mask_m64 = nk_stride_mask_u1x64_(stride_elements);
    __m512i masked_bias_i8x64 = _mm512_maskz_mov_epi8(stride_mask_m64, _mm512_set1_epi8((char)0x80));
    __m512i zero_i8x64 = _mm512_setzero_si512();
    __m512i sum_u64x8 = _mm512_setzero_si512();
    __m512i sumsq_lo_i32x16 = _mm512_setzero_si512();
    __m512i sumsq_hi_i32x16 = _mm512_setzero_si512();
    nk_size_t idx_scalars = 0;
    nk_size_t total_scalars = count * stride_elements;
    nk_size_t elements_per_vector = 64 / stride_elements;
    nk_size_t vector_element_count = 0;
    for (; idx_scalars + 64 <= total_scalars; idx_scalars += 64) {
        __m512i data_i8x64 = _mm512_maskz_loadu_epi8(stride_mask_m64, data + idx_scalars);
        __m512i unsigned_i8x64 = _mm512_xor_si512(data_i8x64, masked_bias_i8x64);
        sum_u64x8 = _mm512_add_epi64(sum_u64x8, _mm512_sad_epu8(unsigned_i8x64, zero_i8x64));
        __m512i lo_i16x32 = _mm512_cvtepi8_epi16(_mm512_castsi512_si256(data_i8x64));
        __m512i hi_i16x32 = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(data_i8x64, 1));
        sumsq_lo_i32x16 = _mm512_add_epi32(sumsq_lo_i32x16, _mm512_madd_epi16(lo_i16x32, lo_i16x32));
        sumsq_hi_i32x16 = _mm512_add_epi32(sumsq_hi_i32x16, _mm512_madd_epi16(hi_i16x32, hi_i16x32));
        vector_element_count += elements_per_vector;
    }
    sumsq_lo_i32x16 = _mm512_add_epi32(sumsq_lo_i32x16, sumsq_hi_i32x16);
    __m512i sumsq_i64x8 = _mm512_cvtepi32_epi64(_mm512_castsi512_si256(sumsq_lo_i32x16));
    sumsq_i64x8 = _mm512_add_epi64(sumsq_i64x8, _mm512_cvtepi32_epi64(_mm512_extracti64x4_epi64(sumsq_lo_i32x16, 1)));
    nk_i64_t s = (nk_i64_t)nk_reduce_add_u64x8_skylake_(sum_u64x8);
    s -= (nk_i64_t)128 * (nk_i64_t)vector_element_count;
    nk_i64_t sq = nk_reduce_add_i64x8_skylake_(sumsq_i64x8);
    nk_i8_t const *ptr = data + idx_scalars;
    nk_size_t remaining = count - idx_scalars / stride_elements;
    for (nk_size_t i = 0; i < remaining; ++i, ptr += stride_elements) {
        nk_i64_t val = (nk_i64_t)*ptr;
        s += val;
        sq += val * val;
    }
    *sum = s;
    *sumsq = sq;
}

NK_PUBLIC void nk_reduce_moments_i8_skylake(                      //
    nk_i8_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *sum, nk_i64_t *sumsq) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_i8_t);
    int aligned = (stride_bytes % sizeof(nk_i8_t) == 0);
    if (count == 0) *sum = 0, *sumsq = 0;
    else if (!aligned) nk_reduce_moments_i8_serial(data, count, stride_bytes, sum, sumsq);
    else if (count > (nk_size_t)(NK_U16_MAX + 1) * 64) {
        nk_size_t left_count = count / 2;
        nk_i64_t left_sum, left_sumsq, right_sum, right_sumsq;
        nk_reduce_moments_i8_skylake(data, left_count, stride_bytes, &left_sum, &left_sumsq);
        nk_reduce_moments_i8_skylake(data + left_count * stride_elements, count - left_count, stride_bytes, &right_sum,
                                     &right_sumsq);
        nk_i64_sadd_(&left_sum, &right_sum, sum);
        nk_i64_sadd_(&left_sumsq, &right_sumsq, sumsq);
    }
    else if (stride_elements == 1) nk_reduce_moments_i8_skylake_contiguous_(data, count, sum, sumsq);
    else if (stride_elements <= 32) nk_reduce_moments_i8_skylake_strided_(data, count, stride_elements, sum, sumsq);
    else nk_reduce_moments_i8_serial(data, count, stride_bytes, sum, sumsq);
}

NK_INTERNAL void nk_reduce_minmax_i8_skylake_contiguous_( //
    nk_i8_t const *data, nk_size_t count,                 //
    nk_i8_t *min_value, nk_size_t *min_index,             //
    nk_i8_t *max_value, nk_size_t *max_index) {
    __m512i min_i8x64 = _mm512_loadu_si512(data);
    __m512i max_i8x64 = min_i8x64;
    __m512i min_iter_u8x64 = _mm512_setzero_si512();
    __m512i max_iter_u8x64 = _mm512_setzero_si512();
    __m512i iter_u8x64 = _mm512_set1_epi8(1);
    __m512i one_u8x64 = _mm512_set1_epi8(1);

    nk_size_t idx = 64;
    for (; idx + 64 <= count; idx += 64) {
        __m512i data_i8x64 = _mm512_loadu_si512(data + idx);
        __mmask64 min_changed_mask = _mm512_cmp_epi8_mask(data_i8x64, min_i8x64, _MM_CMPINT_LT);
        __mmask64 max_changed_mask = _mm512_cmp_epi8_mask(data_i8x64, max_i8x64, _MM_CMPINT_NLE);
        min_i8x64 = _mm512_mask_mov_epi8(min_i8x64, min_changed_mask, data_i8x64);
        max_i8x64 = _mm512_mask_mov_epi8(max_i8x64, max_changed_mask, data_i8x64);
        min_iter_u8x64 = _mm512_mask_mov_epi8(min_iter_u8x64, min_changed_mask, iter_u8x64);
        max_iter_u8x64 = _mm512_mask_mov_epi8(max_iter_u8x64, max_changed_mask, iter_u8x64);
        iter_u8x64 = _mm512_add_epi8(iter_u8x64, one_u8x64);
    }

    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        __mmask64 tail_load = _bzhi_u64(0xFFFFFFFFFFFFFFFFull, (unsigned int)remaining);
        __m512i tail_i8x64 = _mm512_maskz_loadu_epi8(tail_load, data + idx);
        __mmask64 min_changed_mask = _mm512_mask_cmp_epi8_mask(tail_load, tail_i8x64, min_i8x64, _MM_CMPINT_LT);
        __mmask64 max_changed_mask = _mm512_mask_cmp_epi8_mask(tail_load, tail_i8x64, max_i8x64, _MM_CMPINT_NLE);
        min_i8x64 = _mm512_mask_mov_epi8(min_i8x64, min_changed_mask, tail_i8x64);
        max_i8x64 = _mm512_mask_mov_epi8(max_i8x64, max_changed_mask, tail_i8x64);
        min_iter_u8x64 = _mm512_mask_mov_epi8(min_iter_u8x64, min_changed_mask, iter_u8x64);
        max_iter_u8x64 = _mm512_mask_mov_epi8(max_iter_u8x64, max_changed_mask, iter_u8x64);
    }

    nk_i8_t min_scalar = nk_reduce_min_i8x64_skylake_(min_i8x64);
    nk_i8_t max_scalar = nk_reduce_max_i8x64_skylake_(max_i8x64);
    __mmask64 min_equality_mask = _mm512_cmpeq_epi8_mask(min_i8x64, _mm512_set1_epi8(min_scalar));
    __mmask64 max_equality_mask = _mm512_cmpeq_epi8_mask(max_i8x64, _mm512_set1_epi8(max_scalar));
    unsigned int min_lane = (unsigned int)_tzcnt_u64(min_equality_mask);
    unsigned int max_lane = (unsigned int)_tzcnt_u64(max_equality_mask);
    nk_b512_vec_t iter_vec;
    iter_vec.zmm = min_iter_u8x64;
    *min_value = min_scalar, *min_index = (nk_size_t)iter_vec.u8s[min_lane] * 64 + min_lane;
    iter_vec.zmm = max_iter_u8x64;
    *max_value = max_scalar, *max_index = (nk_size_t)iter_vec.u8s[max_lane] * 64 + max_lane;
}

NK_PUBLIC void nk_reduce_minmax_i8_skylake(                       //
    nk_i8_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i8_t *min_value, nk_size_t *min_index,                     //
    nk_i8_t *max_value, nk_size_t *max_index) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_i8_t);
    int aligned = (stride_bytes % sizeof(nk_i8_t) == 0);
    if (count == 0) *min_value = NK_I8_MAX, *min_index = 0, *max_value = NK_I8_MIN, *max_index = 0;
    else if (!aligned)
        nk_reduce_minmax_i8_serial(data, count, stride_bytes, min_value, min_index, max_value, max_index);
    else if (count > (nk_size_t)(NK_U8_MAX + 1) * 64) {
        nk_size_t left_count = count / 2;
        nk_i8_t left_min, right_min, left_max, right_max;
        nk_size_t left_min_index, right_min_index, left_max_index, right_max_index;
        nk_reduce_minmax_i8_skylake(data, left_count, stride_bytes, &left_min, &left_min_index, &left_max,
                                    &left_max_index);
        nk_reduce_minmax_i8_skylake(data + left_count * stride_elements, count - left_count, stride_bytes, &right_min,
                                    &right_min_index, &right_max, &right_max_index);
        if (right_min < left_min) *min_value = right_min, *min_index = left_count + right_min_index;
        else *min_value = left_min, *min_index = left_min_index;
        if (right_max > left_max) *max_value = right_max, *max_index = left_count + right_max_index;
        else *max_value = left_max, *max_index = left_max_index;
    }
    else if (stride_elements == 1)
        nk_reduce_minmax_i8_skylake_contiguous_(data, count, min_value, min_index, max_value, max_index);
    else nk_reduce_minmax_i8_serial(data, count, stride_bytes, min_value, min_index, max_value, max_index);
}

NK_INTERNAL void nk_reduce_moments_u8_skylake_contiguous_( //
    nk_u8_t const *data, nk_size_t count,                  //
    nk_u64_t *sum, nk_u64_t *sumsq) {
    // Sum: VPSADBW directly (same as nk_reduce_add_u8_skylake_contiguous_).
    // Sumsq: widen u8→i16, VPMADDWD(x,x) → i32 (pairs of squares), accumulate i32.
    // i32 overflow safe: max per lane = (255² + 255²) * 1024 iters ≈ 133M < 2.1B.
    __m512i zero_u8x64 = _mm512_setzero_si512();
    __m512i sum_u64x8 = _mm512_setzero_si512();
    __m512i sumsq_lo_i32x16 = _mm512_setzero_si512();
    __m512i sumsq_hi_i32x16 = _mm512_setzero_si512();
    nk_size_t idx = 0;
    for (; idx + 64 <= count; idx += 64) {
        __m512i data_u8x64 = _mm512_loadu_si512(data + idx);
        sum_u64x8 = _mm512_add_epi64(sum_u64x8, _mm512_sad_epu8(data_u8x64, zero_u8x64));
        __m512i lo_i16x32 = _mm512_cvtepu8_epi16(_mm512_castsi512_si256(data_u8x64));
        __m512i hi_i16x32 = _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64(data_u8x64, 1));
        sumsq_lo_i32x16 = _mm512_add_epi32(sumsq_lo_i32x16, _mm512_madd_epi16(lo_i16x32, lo_i16x32));
        sumsq_hi_i32x16 = _mm512_add_epi32(sumsq_hi_i32x16, _mm512_madd_epi16(hi_i16x32, hi_i16x32));
    }
    // Flush i32 → u64 once
    sumsq_lo_i32x16 = _mm512_add_epi32(sumsq_lo_i32x16, sumsq_hi_i32x16);
    __m512i sumsq_u64x8 = _mm512_cvtepu32_epi64(_mm512_castsi512_si256(sumsq_lo_i32x16));
    sumsq_u64x8 = _mm512_add_epi64(sumsq_u64x8, _mm512_cvtepu32_epi64(_mm512_extracti64x4_epi64(sumsq_lo_i32x16, 1)));
    nk_u64_t s = nk_reduce_add_u64x8_skylake_(sum_u64x8);
    nk_u64_t sq = nk_reduce_add_u64x8_skylake_(sumsq_u64x8);
    for (; idx < count; ++idx) {
        nk_u64_t val = (nk_u64_t)data[idx];
        s += val;
        sq += val * val;
    }
    *sum = s;
    *sumsq = sq;
}

NK_INTERNAL void nk_reduce_moments_u8_skylake_strided_(              //
    nk_u8_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_u64_t *sum, nk_u64_t *sumsq) {
    __mmask64 stride_mask_m64 = nk_stride_mask_u1x64_(stride_elements);
    __m512i zero_u8x64 = _mm512_setzero_si512();
    __m512i sum_u64x8 = _mm512_setzero_si512();
    __m512i sumsq_lo_i32x16 = _mm512_setzero_si512();
    __m512i sumsq_hi_i32x16 = _mm512_setzero_si512();
    nk_size_t idx_scalars = 0;
    nk_size_t total_scalars = count * stride_elements;
    for (; idx_scalars + 64 <= total_scalars; idx_scalars += 64) {
        __m512i data_u8x64 = _mm512_maskz_loadu_epi8(stride_mask_m64, data + idx_scalars);
        sum_u64x8 = _mm512_add_epi64(sum_u64x8, _mm512_sad_epu8(data_u8x64, zero_u8x64));
        __m512i lo_i16x32 = _mm512_cvtepu8_epi16(_mm512_castsi512_si256(data_u8x64));
        __m512i hi_i16x32 = _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64(data_u8x64, 1));
        sumsq_lo_i32x16 = _mm512_add_epi32(sumsq_lo_i32x16, _mm512_madd_epi16(lo_i16x32, lo_i16x32));
        sumsq_hi_i32x16 = _mm512_add_epi32(sumsq_hi_i32x16, _mm512_madd_epi16(hi_i16x32, hi_i16x32));
    }
    sumsq_lo_i32x16 = _mm512_add_epi32(sumsq_lo_i32x16, sumsq_hi_i32x16);
    __m512i sumsq_u64x8 = _mm512_cvtepu32_epi64(_mm512_castsi512_si256(sumsq_lo_i32x16));
    sumsq_u64x8 = _mm512_add_epi64(sumsq_u64x8, _mm512_cvtepu32_epi64(_mm512_extracti64x4_epi64(sumsq_lo_i32x16, 1)));
    nk_u64_t s = nk_reduce_add_u64x8_skylake_(sum_u64x8);
    nk_u64_t sq = nk_reduce_add_u64x8_skylake_(sumsq_u64x8);
    nk_u8_t const *ptr = data + idx_scalars;
    nk_size_t remaining = count - idx_scalars / stride_elements;
    for (nk_size_t i = 0; i < remaining; ++i, ptr += stride_elements) {
        nk_u64_t val = (nk_u64_t)*ptr;
        s += val;
        sq += val * val;
    }
    *sum = s;
    *sumsq = sq;
}

NK_PUBLIC void nk_reduce_moments_u8_skylake(                      //
    nk_u8_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *sum, nk_u64_t *sumsq) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_u8_t);
    int aligned = (stride_bytes % sizeof(nk_u8_t) == 0);
    if (count == 0) *sum = 0, *sumsq = 0;
    else if (!aligned) nk_reduce_moments_u8_serial(data, count, stride_bytes, sum, sumsq);
    else if (count > (nk_size_t)(NK_U8_MAX + 1) * 64) {
        nk_size_t left_count = count / 2;
        nk_u64_t left_sum, left_sumsq, right_sum, right_sumsq;
        nk_reduce_moments_u8_skylake(data, left_count, stride_bytes, &left_sum, &left_sumsq);
        nk_reduce_moments_u8_skylake(data + left_count * stride_elements, count - left_count, stride_bytes, &right_sum,
                                     &right_sumsq);
        nk_u64_sadd_(&left_sum, &right_sum, sum);
        nk_u64_sadd_(&left_sumsq, &right_sumsq, sumsq);
    }
    else if (stride_elements == 1) nk_reduce_moments_u8_skylake_contiguous_(data, count, sum, sumsq);
    else if (stride_elements <= 32) nk_reduce_moments_u8_skylake_strided_(data, count, stride_elements, sum, sumsq);
    else nk_reduce_moments_u8_serial(data, count, stride_bytes, sum, sumsq);
}

NK_INTERNAL void nk_reduce_minmax_u8_skylake_contiguous_( //
    nk_u8_t const *data, nk_size_t count,                 //
    nk_u8_t *min_value, nk_size_t *min_index,             //
    nk_u8_t *max_value, nk_size_t *max_index) {
    __m512i min_u8x64 = _mm512_loadu_si512(data);
    __m512i max_u8x64 = min_u8x64;
    __m512i min_iter_u8x64 = _mm512_setzero_si512();
    __m512i max_iter_u8x64 = _mm512_setzero_si512();
    __m512i iter_u8x64 = _mm512_set1_epi8(1);
    __m512i one_u8x64 = _mm512_set1_epi8(1);

    nk_size_t idx = 64;
    for (; idx + 64 <= count; idx += 64) {
        __m512i data_u8x64 = _mm512_loadu_si512(data + idx);
        __mmask64 min_changed_mask = _mm512_cmp_epu8_mask(data_u8x64, min_u8x64, _MM_CMPINT_LT);
        __mmask64 max_changed_mask = _mm512_cmp_epu8_mask(data_u8x64, max_u8x64, _MM_CMPINT_NLE);
        min_u8x64 = _mm512_mask_mov_epi8(min_u8x64, min_changed_mask, data_u8x64);
        max_u8x64 = _mm512_mask_mov_epi8(max_u8x64, max_changed_mask, data_u8x64);
        min_iter_u8x64 = _mm512_mask_mov_epi8(min_iter_u8x64, min_changed_mask, iter_u8x64);
        max_iter_u8x64 = _mm512_mask_mov_epi8(max_iter_u8x64, max_changed_mask, iter_u8x64);
        iter_u8x64 = _mm512_add_epi8(iter_u8x64, one_u8x64);
    }

    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        __mmask64 tail_load = _bzhi_u64(0xFFFFFFFFFFFFFFFFull, (unsigned int)remaining);
        __m512i tail_u8x64 = _mm512_maskz_loadu_epi8(tail_load, data + idx);
        __mmask64 min_changed_mask = _mm512_mask_cmp_epu8_mask(tail_load, tail_u8x64, min_u8x64, _MM_CMPINT_LT);
        __mmask64 max_changed_mask = _mm512_mask_cmp_epu8_mask(tail_load, tail_u8x64, max_u8x64, _MM_CMPINT_NLE);
        min_u8x64 = _mm512_mask_mov_epi8(min_u8x64, min_changed_mask, tail_u8x64);
        max_u8x64 = _mm512_mask_mov_epi8(max_u8x64, max_changed_mask, tail_u8x64);
        min_iter_u8x64 = _mm512_mask_mov_epi8(min_iter_u8x64, min_changed_mask, iter_u8x64);
        max_iter_u8x64 = _mm512_mask_mov_epi8(max_iter_u8x64, max_changed_mask, iter_u8x64);
    }

    nk_u8_t min_scalar = nk_reduce_min_u8x64_skylake_(min_u8x64);
    nk_u8_t max_scalar = nk_reduce_max_u8x64_skylake_(max_u8x64);
    __mmask64 min_equality_mask = _mm512_cmpeq_epi8_mask(min_u8x64, _mm512_set1_epi8((char)min_scalar));
    __mmask64 max_equality_mask = _mm512_cmpeq_epi8_mask(max_u8x64, _mm512_set1_epi8((char)max_scalar));
    unsigned int min_lane = (unsigned int)_tzcnt_u64(min_equality_mask);
    unsigned int max_lane = (unsigned int)_tzcnt_u64(max_equality_mask);
    nk_b512_vec_t iter_vec;
    iter_vec.zmm = min_iter_u8x64;
    *min_value = min_scalar, *min_index = (nk_size_t)iter_vec.u8s[min_lane] * 64 + min_lane;
    iter_vec.zmm = max_iter_u8x64;
    *max_value = max_scalar, *max_index = (nk_size_t)iter_vec.u8s[max_lane] * 64 + max_lane;
}

NK_PUBLIC void nk_reduce_minmax_u8_skylake(                       //
    nk_u8_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u8_t *min_value, nk_size_t *min_index,                     //
    nk_u8_t *max_value, nk_size_t *max_index) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_u8_t);
    int aligned = (stride_bytes % sizeof(nk_u8_t) == 0);
    if (count == 0) *min_value = NK_U8_MAX, *min_index = 0, *max_value = 0, *max_index = 0;
    else if (!aligned)
        nk_reduce_minmax_u8_serial(data, count, stride_bytes, min_value, min_index, max_value, max_index);
    else if (count > (nk_size_t)(NK_U8_MAX + 1) * 64) {
        nk_size_t left_count = count / 2;
        nk_u8_t left_min, right_min, left_max, right_max;
        nk_size_t left_min_index, right_min_index, left_max_index, right_max_index;
        nk_reduce_minmax_u8_skylake(data, left_count, stride_bytes, &left_min, &left_min_index, &left_max,
                                    &left_max_index);
        nk_reduce_minmax_u8_skylake(data + left_count * stride_elements, count - left_count, stride_bytes, &right_min,
                                    &right_min_index, &right_max, &right_max_index);
        if (right_min < left_min) *min_value = right_min, *min_index = left_count + right_min_index;
        else *min_value = left_min, *min_index = left_min_index;
        if (right_max > left_max) *max_value = right_max, *max_index = left_count + right_max_index;
        else *max_value = left_max, *max_index = left_max_index;
    }
    else if (stride_elements == 1)
        nk_reduce_minmax_u8_skylake_contiguous_(data, count, min_value, min_index, max_value, max_index);
    else nk_reduce_minmax_u8_serial(data, count, stride_bytes, min_value, min_index, max_value, max_index);
}

NK_INTERNAL void nk_reduce_moments_i16_skylake_contiguous_( //
    nk_i16_t const *data, nk_size_t count,                  //
    nk_i64_t *sum, nk_i64_t *sumsq) {
    // Sum: VPMADDWD(data, ones) → i32 pairs, accumulate i32, single flush at end.
    // Within 65536-element block (2048 iters), max i32 = ±65536 * 2048 ≈ ±134M — safe.
    // Sumsq: VPMADDWD(data, data) → i32, each up to ~2.1B — must flush to i64 every iteration.
    __m512i ones_i16x32 = _mm512_set1_epi16(1);
    __m512i sum_i32x16 = _mm512_setzero_si512();
    __m512i sumsq_i64x8 = _mm512_setzero_si512();
    nk_size_t idx = 0;
    for (; idx + 32 <= count; idx += 32) {
        __m512i data_i16x32 = _mm512_loadu_si512(data + idx);
        sum_i32x16 = _mm512_add_epi32(sum_i32x16, _mm512_madd_epi16(data_i16x32, ones_i16x32));
        __m512i sq_i32x16 = _mm512_madd_epi16(data_i16x32, data_i16x32);
        sumsq_i64x8 = _mm512_add_epi64(sumsq_i64x8, _mm512_cvtepi32_epi64(_mm512_castsi512_si256(sq_i32x16)));
        sumsq_i64x8 = _mm512_add_epi64(sumsq_i64x8, _mm512_cvtepi32_epi64(_mm512_extracti64x4_epi64(sq_i32x16, 1)));
    }
    __m512i sum_i64x8 = _mm512_add_epi64(                                 //
        _mm512_cvtepi32_epi64(_mm512_castsi512_si256(sum_i32x16)),        //
        _mm512_cvtepi32_epi64(_mm512_extracti64x4_epi64(sum_i32x16, 1))); //
    nk_i64_t s = nk_reduce_add_i64x8_skylake_(sum_i64x8);
    nk_i64_t sq = nk_reduce_add_i64x8_skylake_(sumsq_i64x8);
    for (; idx < count; ++idx) {
        nk_i64_t val = (nk_i64_t)data[idx];
        s += val;
        sq += val * val;
    }
    *sum = s;
    *sumsq = sq;
}

NK_INTERNAL void nk_reduce_moments_i16_skylake_strided_(              //
    nk_i16_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_i64_t *sum, nk_i64_t *sumsq) {
    __mmask32 stride_mask_m32 = nk_stride_mask_b16x32_(stride_elements);
    __m512i ones_i16x32 = _mm512_set1_epi16(1);
    __m512i sum_i32x16 = _mm512_setzero_si512();
    __m512i sumsq_i64x8 = _mm512_setzero_si512();
    nk_size_t idx_scalars = 0;
    nk_size_t total_scalars = count * stride_elements;
    for (; idx_scalars + 32 <= total_scalars; idx_scalars += 32) {
        __m512i data_i16x32 = _mm512_maskz_loadu_epi16(stride_mask_m32, data + idx_scalars);
        sum_i32x16 = _mm512_add_epi32(sum_i32x16, _mm512_madd_epi16(data_i16x32, ones_i16x32));
        __m512i sq_i32x16 = _mm512_madd_epi16(data_i16x32, data_i16x32);
        sumsq_i64x8 = _mm512_add_epi64(sumsq_i64x8, _mm512_cvtepi32_epi64(_mm512_castsi512_si256(sq_i32x16)));
        sumsq_i64x8 = _mm512_add_epi64(sumsq_i64x8, _mm512_cvtepi32_epi64(_mm512_extracti64x4_epi64(sq_i32x16, 1)));
    }
    __m512i sum_i64x8 = _mm512_add_epi64(                                 //
        _mm512_cvtepi32_epi64(_mm512_castsi512_si256(sum_i32x16)),        //
        _mm512_cvtepi32_epi64(_mm512_extracti64x4_epi64(sum_i32x16, 1))); //
    nk_i64_t s = nk_reduce_add_i64x8_skylake_(sum_i64x8);
    nk_i64_t sq = nk_reduce_add_i64x8_skylake_(sumsq_i64x8);
    nk_i16_t const *ptr = data + idx_scalars;
    nk_size_t remaining = count - idx_scalars / stride_elements;
    for (nk_size_t i = 0; i < remaining; ++i, ptr += stride_elements) {
        nk_i64_t val = (nk_i64_t)*ptr;
        s += val;
        sq += val * val;
    }
    *sum = s;
    *sumsq = sq;
}

NK_PUBLIC void nk_reduce_moments_i16_skylake(                      //
    nk_i16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *sum, nk_i64_t *sumsq) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_i16_t);
    int aligned = (stride_bytes % sizeof(nk_i16_t) == 0);
    if (count == 0) *sum = 0, *sumsq = 0;
    else if (!aligned) nk_reduce_moments_i16_serial(data, count, stride_bytes, sum, sumsq);
    else if (count > (nk_size_t)(NK_I16_MAX + 1) * 32) {
        nk_size_t left_count = count / 2;
        nk_i64_t left_sum, left_sumsq, right_sum, right_sumsq;
        nk_reduce_moments_i16_skylake(data, left_count, stride_bytes, &left_sum, &left_sumsq);
        nk_reduce_moments_i16_skylake(data + left_count * stride_elements, count - left_count, stride_bytes, &right_sum,
                                      &right_sumsq);
        nk_i64_sadd_(&left_sum, &right_sum, sum);
        nk_i64_sadd_(&left_sumsq, &right_sumsq, sumsq);
    }
    else if (stride_elements == 1) nk_reduce_moments_i16_skylake_contiguous_(data, count, sum, sumsq);
    else if (stride_elements <= 16) nk_reduce_moments_i16_skylake_strided_(data, count, stride_elements, sum, sumsq);
    else nk_reduce_moments_i16_serial(data, count, stride_bytes, sum, sumsq);
}

NK_INTERNAL void nk_reduce_minmax_i16_skylake_contiguous_( //
    nk_i16_t const *data, nk_size_t count,                 //
    nk_i16_t *min_value, nk_size_t *min_index,             //
    nk_i16_t *max_value, nk_size_t *max_index) {
    __m512i min_i16x32 = _mm512_loadu_si512(data);
    __m512i max_i16x32 = min_i16x32;
    __m512i min_iter_u16x32 = _mm512_setzero_si512();
    __m512i max_iter_u16x32 = _mm512_setzero_si512();
    __m512i iter_u16x32 = _mm512_set1_epi16(1);
    __m512i one_u16x32 = _mm512_set1_epi16(1);

    nk_size_t idx = 32;
    for (; idx + 32 <= count; idx += 32) {
        __m512i data_i16x32 = _mm512_loadu_si512(data + idx);
        __mmask32 min_changed_mask = _mm512_cmp_epi16_mask(data_i16x32, min_i16x32, _MM_CMPINT_LT);
        __mmask32 max_changed_mask = _mm512_cmp_epi16_mask(data_i16x32, max_i16x32, _MM_CMPINT_NLE);
        min_i16x32 = _mm512_mask_mov_epi16(min_i16x32, min_changed_mask, data_i16x32);
        max_i16x32 = _mm512_mask_mov_epi16(max_i16x32, max_changed_mask, data_i16x32);
        min_iter_u16x32 = _mm512_mask_mov_epi16(min_iter_u16x32, min_changed_mask, iter_u16x32);
        max_iter_u16x32 = _mm512_mask_mov_epi16(max_iter_u16x32, max_changed_mask, iter_u16x32);
        iter_u16x32 = _mm512_add_epi16(iter_u16x32, one_u16x32);
    }

    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        __mmask32 tail_load = (__mmask32)_bzhi_u32(0xFFFFFFFF, (unsigned int)remaining);
        __m512i tail_i16x32 = _mm512_maskz_loadu_epi16(tail_load, data + idx);
        __mmask32 min_changed_mask = _mm512_mask_cmp_epi16_mask(tail_load, tail_i16x32, min_i16x32, _MM_CMPINT_LT);
        __mmask32 max_changed_mask = _mm512_mask_cmp_epi16_mask(tail_load, tail_i16x32, max_i16x32, _MM_CMPINT_NLE);
        min_i16x32 = _mm512_mask_mov_epi16(min_i16x32, min_changed_mask, tail_i16x32);
        max_i16x32 = _mm512_mask_mov_epi16(max_i16x32, max_changed_mask, tail_i16x32);
        min_iter_u16x32 = _mm512_mask_mov_epi16(min_iter_u16x32, min_changed_mask, iter_u16x32);
        max_iter_u16x32 = _mm512_mask_mov_epi16(max_iter_u16x32, max_changed_mask, iter_u16x32);
    }

    nk_i16_t min_scalar = nk_reduce_min_i16x32_skylake_(min_i16x32);
    nk_i16_t max_scalar = nk_reduce_max_i16x32_skylake_(max_i16x32);
    __mmask32 min_equality_mask = _mm512_cmpeq_epi16_mask(min_i16x32, _mm512_set1_epi16(min_scalar));
    __mmask32 max_equality_mask = _mm512_cmpeq_epi16_mask(max_i16x32, _mm512_set1_epi16(max_scalar));
    unsigned int min_lane = _tzcnt_u32(min_equality_mask);
    unsigned int max_lane = _tzcnt_u32(max_equality_mask);
    nk_b512_vec_t iter_vec;
    iter_vec.zmm = min_iter_u16x32;
    *min_value = min_scalar, *min_index = (nk_size_t)iter_vec.u16s[min_lane] * 32 + min_lane;
    iter_vec.zmm = max_iter_u16x32;
    *max_value = max_scalar, *max_index = (nk_size_t)iter_vec.u16s[max_lane] * 32 + max_lane;
}

NK_PUBLIC void nk_reduce_minmax_i16_skylake(                       //
    nk_i16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i16_t *min_value, nk_size_t *min_index,                     //
    nk_i16_t *max_value, nk_size_t *max_index) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_i16_t);
    int aligned = (stride_bytes % sizeof(nk_i16_t) == 0);
    if (count == 0) *min_value = NK_I16_MAX, *min_index = 0, *max_value = NK_I16_MIN, *max_index = 0;
    else if (!aligned)
        nk_reduce_minmax_i16_serial(data, count, stride_bytes, min_value, min_index, max_value, max_index);
    else if (count > (nk_size_t)(NK_U16_MAX + 1) * 32) {
        nk_size_t left_count = count / 2;
        nk_i16_t left_min, right_min, left_max, right_max;
        nk_size_t left_min_index, right_min_index, left_max_index, right_max_index;
        nk_reduce_minmax_i16_skylake(data, left_count, stride_bytes, &left_min, &left_min_index, &left_max,
                                     &left_max_index);
        nk_reduce_minmax_i16_skylake(data + left_count * stride_elements, count - left_count, stride_bytes, &right_min,
                                     &right_min_index, &right_max, &right_max_index);
        if (right_min < left_min) *min_value = right_min, *min_index = left_count + right_min_index;
        else *min_value = left_min, *min_index = left_min_index;
        if (right_max > left_max) *max_value = right_max, *max_index = left_count + right_max_index;
        else *max_value = left_max, *max_index = left_max_index;
    }
    else if (stride_elements == 1)
        nk_reduce_minmax_i16_skylake_contiguous_(data, count, min_value, min_index, max_value, max_index);
    else nk_reduce_minmax_i16_serial(data, count, stride_bytes, min_value, min_index, max_value, max_index);
}

NK_INTERNAL void nk_reduce_moments_u16_skylake_contiguous_( //
    nk_u16_t const *data, nk_size_t count,                  //
    nk_u64_t *sum, nk_u64_t *sumsq) {
    // Widen u16→u32, square in u32, widen to u64. Avoids bias trick whose
    // VPMADDWD pair-of-squares overflows i32 when both lanes map to -32768.
    __m512i zero = _mm512_setzero_si512();
    __m512i sum_u32x16 = _mm512_setzero_si512();
    __m512i sumsq_u64x8 = _mm512_setzero_si512();
    nk_size_t idx = 0;
    for (; idx + 16 <= count; idx += 16) {
        __m512i data_u32x16 = _mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i const *)(data + idx)));
        sum_u32x16 = _mm512_add_epi32(sum_u32x16, data_u32x16);
        __m512i sq_u32x16 = _mm512_mullo_epi32(data_u32x16, data_u32x16);
        sumsq_u64x8 = _mm512_add_epi64(sumsq_u64x8, _mm512_unpacklo_epi32(sq_u32x16, zero));
        sumsq_u64x8 = _mm512_add_epi64(sumsq_u64x8, _mm512_unpackhi_epi32(sq_u32x16, zero));
    }
    if (idx < count) {
        __mmask16 tail_mask = (__mmask16)((1u << (count - idx)) - 1);
        __m512i data_u32x16 = _mm512_cvtepu16_epi32(_mm256_maskz_loadu_epi16(tail_mask, data + idx));
        sum_u32x16 = _mm512_add_epi32(sum_u32x16, data_u32x16);
        __m512i sq_u32x16 = _mm512_mullo_epi32(data_u32x16, data_u32x16);
        sumsq_u64x8 = _mm512_add_epi64(sumsq_u64x8, _mm512_unpacklo_epi32(sq_u32x16, zero));
        sumsq_u64x8 = _mm512_add_epi64(sumsq_u64x8, _mm512_unpackhi_epi32(sq_u32x16, zero));
    }
    __m512i sum_u64x8 = _mm512_add_epi64(         //
        _mm512_unpacklo_epi32(sum_u32x16, zero),  //
        _mm512_unpackhi_epi32(sum_u32x16, zero)); //
    *sum = (nk_u64_t)nk_reduce_add_i64x8_skylake_(sum_u64x8);
    *sumsq = (nk_u64_t)nk_reduce_add_i64x8_skylake_(sumsq_u64x8);
}

NK_INTERNAL void nk_reduce_moments_u16_skylake_strided_(              //
    nk_u16_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_u64_t *sum, nk_u64_t *sumsq) {
    __mmask32 stride_mask_m32 = nk_stride_mask_b16x32_(stride_elements);
    __m512i zero = _mm512_setzero_si512();
    __m512i sum_u64x8 = _mm512_setzero_si512();
    __m512i sumsq_u64x8 = _mm512_setzero_si512();
    nk_size_t idx_scalars = 0;
    nk_size_t total_scalars = count * stride_elements;
    for (; idx_scalars + 32 <= total_scalars; idx_scalars += 32) {
        __m512i data_u16x32 = _mm512_maskz_loadu_epi16(stride_mask_m32, data + idx_scalars);
        __m512i lo_u32x16 = _mm512_unpacklo_epi16(data_u16x32, zero);
        __m512i hi_u32x16 = _mm512_unpackhi_epi16(data_u16x32, zero);
        sum_u64x8 = _mm512_add_epi64(sum_u64x8, _mm512_unpacklo_epi32(lo_u32x16, zero));
        sum_u64x8 = _mm512_add_epi64(sum_u64x8, _mm512_unpackhi_epi32(lo_u32x16, zero));
        sum_u64x8 = _mm512_add_epi64(sum_u64x8, _mm512_unpacklo_epi32(hi_u32x16, zero));
        sum_u64x8 = _mm512_add_epi64(sum_u64x8, _mm512_unpackhi_epi32(hi_u32x16, zero));
        __m512i lo_sq = _mm512_mullo_epi32(lo_u32x16, lo_u32x16);
        __m512i hi_sq = _mm512_mullo_epi32(hi_u32x16, hi_u32x16);
        sumsq_u64x8 = _mm512_add_epi64(sumsq_u64x8, _mm512_unpacklo_epi32(lo_sq, zero));
        sumsq_u64x8 = _mm512_add_epi64(sumsq_u64x8, _mm512_unpackhi_epi32(lo_sq, zero));
        sumsq_u64x8 = _mm512_add_epi64(sumsq_u64x8, _mm512_unpacklo_epi32(hi_sq, zero));
        sumsq_u64x8 = _mm512_add_epi64(sumsq_u64x8, _mm512_unpackhi_epi32(hi_sq, zero));
    }
    nk_u64_t s = (nk_u64_t)nk_reduce_add_i64x8_skylake_(sum_u64x8);
    nk_u64_t sq = (nk_u64_t)nk_reduce_add_i64x8_skylake_(sumsq_u64x8);
    nk_u16_t const *ptr = data + idx_scalars;
    nk_size_t remaining = count - idx_scalars / stride_elements;
    for (nk_size_t i = 0; i < remaining; ++i, ptr += stride_elements) {
        nk_u64_t val = (nk_u64_t)*ptr;
        s += val;
        sq += val * val;
    }
    *sum = s;
    *sumsq = sq;
}

NK_PUBLIC void nk_reduce_moments_u16_skylake(                      //
    nk_u16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *sum, nk_u64_t *sumsq) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_u16_t);
    int aligned = (stride_bytes % sizeof(nk_u16_t) == 0);
    if (count == 0) *sum = 0, *sumsq = 0;
    else if (!aligned) nk_reduce_moments_u16_serial(data, count, stride_bytes, sum, sumsq);
    else if (count > (nk_size_t)(NK_U16_MAX + 1) * 16) {
        nk_size_t left_count = count / 2;
        nk_u64_t left_sum, left_sumsq, right_sum, right_sumsq;
        nk_reduce_moments_u16_skylake(data, left_count, stride_bytes, &left_sum, &left_sumsq);
        nk_reduce_moments_u16_skylake(data + left_count * stride_elements, count - left_count, stride_bytes, &right_sum,
                                      &right_sumsq);
        nk_u64_sadd_(&left_sum, &right_sum, sum);
        nk_u64_sadd_(&left_sumsq, &right_sumsq, sumsq);
    }
    else if (stride_elements == 1) nk_reduce_moments_u16_skylake_contiguous_(data, count, sum, sumsq);
    else if (stride_elements <= 16) nk_reduce_moments_u16_skylake_strided_(data, count, stride_elements, sum, sumsq);
    else nk_reduce_moments_u16_serial(data, count, stride_bytes, sum, sumsq);
}

NK_INTERNAL void nk_reduce_minmax_u16_skylake_contiguous_( //
    nk_u16_t const *data, nk_size_t count,                 //
    nk_u16_t *min_value, nk_size_t *min_index,             //
    nk_u16_t *max_value, nk_size_t *max_index) {
    __m512i min_u16x32 = _mm512_loadu_si512(data);
    __m512i max_u16x32 = min_u16x32;
    __m512i min_iter_u16x32 = _mm512_setzero_si512();
    __m512i max_iter_u16x32 = _mm512_setzero_si512();
    __m512i iter_u16x32 = _mm512_set1_epi16(1);
    __m512i one_u16x32 = _mm512_set1_epi16(1);

    nk_size_t idx = 32;
    for (; idx + 32 <= count; idx += 32) {
        __m512i data_u16x32 = _mm512_loadu_si512(data + idx);
        __mmask32 min_changed_mask = _mm512_cmp_epu16_mask(data_u16x32, min_u16x32, _MM_CMPINT_LT);
        __mmask32 max_changed_mask = _mm512_cmp_epu16_mask(data_u16x32, max_u16x32, _MM_CMPINT_NLE);
        min_u16x32 = _mm512_mask_mov_epi16(min_u16x32, min_changed_mask, data_u16x32);
        max_u16x32 = _mm512_mask_mov_epi16(max_u16x32, max_changed_mask, data_u16x32);
        min_iter_u16x32 = _mm512_mask_mov_epi16(min_iter_u16x32, min_changed_mask, iter_u16x32);
        max_iter_u16x32 = _mm512_mask_mov_epi16(max_iter_u16x32, max_changed_mask, iter_u16x32);
        iter_u16x32 = _mm512_add_epi16(iter_u16x32, one_u16x32);
    }

    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        __mmask32 tail_load = (__mmask32)_bzhi_u32(0xFFFFFFFF, (unsigned int)remaining);
        __m512i tail_u16x32 = _mm512_maskz_loadu_epi16(tail_load, data + idx);
        __mmask32 min_changed_mask = _mm512_mask_cmp_epu16_mask(tail_load, tail_u16x32, min_u16x32, _MM_CMPINT_LT);
        __mmask32 max_changed_mask = _mm512_mask_cmp_epu16_mask(tail_load, tail_u16x32, max_u16x32, _MM_CMPINT_NLE);
        min_u16x32 = _mm512_mask_mov_epi16(min_u16x32, min_changed_mask, tail_u16x32);
        max_u16x32 = _mm512_mask_mov_epi16(max_u16x32, max_changed_mask, tail_u16x32);
        min_iter_u16x32 = _mm512_mask_mov_epi16(min_iter_u16x32, min_changed_mask, iter_u16x32);
        max_iter_u16x32 = _mm512_mask_mov_epi16(max_iter_u16x32, max_changed_mask, iter_u16x32);
    }

    nk_u16_t min_scalar = nk_reduce_min_u16x32_skylake_(min_u16x32);
    nk_u16_t max_scalar = nk_reduce_max_u16x32_skylake_(max_u16x32);
    __mmask32 min_equality_mask = _mm512_cmpeq_epi16_mask(min_u16x32, _mm512_set1_epi16((short)min_scalar));
    __mmask32 max_equality_mask = _mm512_cmpeq_epi16_mask(max_u16x32, _mm512_set1_epi16((short)max_scalar));
    unsigned int min_lane = _tzcnt_u32(min_equality_mask);
    unsigned int max_lane = _tzcnt_u32(max_equality_mask);
    nk_b512_vec_t iter_vec;
    iter_vec.zmm = min_iter_u16x32;
    *min_value = min_scalar, *min_index = (nk_size_t)iter_vec.u16s[min_lane] * 32 + min_lane;
    iter_vec.zmm = max_iter_u16x32;
    *max_value = max_scalar, *max_index = (nk_size_t)iter_vec.u16s[max_lane] * 32 + max_lane;
}

NK_PUBLIC void nk_reduce_minmax_u16_skylake(                       //
    nk_u16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u16_t *min_value, nk_size_t *min_index,                     //
    nk_u16_t *max_value, nk_size_t *max_index) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_u16_t);
    int aligned = (stride_bytes % sizeof(nk_u16_t) == 0);
    if (count == 0) *min_value = NK_U16_MAX, *min_index = 0, *max_value = 0, *max_index = 0;
    else if (!aligned)
        nk_reduce_minmax_u16_serial(data, count, stride_bytes, min_value, min_index, max_value, max_index);
    else if (count > (nk_size_t)(NK_U16_MAX + 1) * 32) {
        nk_size_t left_count = count / 2;
        nk_u16_t left_min, right_min, left_max, right_max;
        nk_size_t left_min_index, right_min_index, left_max_index, right_max_index;
        nk_reduce_minmax_u16_skylake(data, left_count, stride_bytes, &left_min, &left_min_index, &left_max,
                                     &left_max_index);
        nk_reduce_minmax_u16_skylake(data + left_count * stride_elements, count - left_count, stride_bytes, &right_min,
                                     &right_min_index, &right_max, &right_max_index);
        if (right_min < left_min) *min_value = right_min, *min_index = left_count + right_min_index;
        else *min_value = left_min, *min_index = left_min_index;
        if (right_max > left_max) *max_value = right_max, *max_index = left_count + right_max_index;
        else *max_value = left_max, *max_index = left_max_index;
    }
    else if (stride_elements == 1)
        nk_reduce_minmax_u16_skylake_contiguous_(data, count, min_value, min_index, max_value, max_index);
    else nk_reduce_minmax_u16_serial(data, count, stride_bytes, min_value, min_index, max_value, max_index);
}

NK_INTERNAL void nk_reduce_moments_i32_skylake_contiguous_( //
    nk_i32_t const *data, nk_size_t count,                  //
    nk_i64_t *sum, nk_i64_t *sumsq) {
    // Sum: widen i32→i64, accumulate directly (same as nk_reduce_add_i32_skylake).
    // Sumsq: VPMULDQ (5-cycle, 1 uop) squares even i32 lanes into i64;
    // VPSRLQ+VPMULDQ squares odd lanes. Avoids VPMULLQ (15-cycle, 3 uops).
    __m512i sum_i64x8 = _mm512_setzero_si512();
    __m512i sumsq_i64x8 = _mm512_setzero_si512();
    nk_size_t idx = 0;
    for (; idx + 16 <= count; idx += 16) {
        __m512i data_i32x16 = _mm512_loadu_si512(data + idx);
        __m256i lo_i32x8 = _mm512_castsi512_si256(data_i32x16);
        __m256i hi_i32x8 = _mm512_extracti64x4_epi64(data_i32x16, 1);
        sum_i64x8 = _mm512_add_epi64(sum_i64x8, _mm512_cvtepi32_epi64(lo_i32x8));
        sum_i64x8 = _mm512_add_epi64(sum_i64x8, _mm512_cvtepi32_epi64(hi_i32x8));
        __m512i even_sq_i64x8 = _mm512_mul_epi32(data_i32x16, data_i32x16);
        __m512i odd_i32x16 = _mm512_srli_epi64(data_i32x16, 32);
        __m512i odd_sq_i64x8 = _mm512_mul_epi32(odd_i32x16, odd_i32x16);
        sumsq_i64x8 = _mm512_add_epi64(sumsq_i64x8, even_sq_i64x8);
        sumsq_i64x8 = _mm512_add_epi64(sumsq_i64x8, odd_sq_i64x8);
    }
    nk_i64_t s = nk_reduce_add_i64x8_skylake_(sum_i64x8);
    nk_i64_t sq = nk_reduce_add_i64x8_skylake_(sumsq_i64x8);
    for (; idx < count; ++idx) {
        nk_i64_t val = (nk_i64_t)data[idx];
        s += val;
        sq += val * val;
    }
    *sum = s;
    *sumsq = sq;
}

NK_PUBLIC void nk_reduce_moments_i32_skylake(                      //
    nk_i32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *sum, nk_i64_t *sumsq) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_i32_t);
    int aligned = (stride_bytes % sizeof(nk_i32_t) == 0);
    if (count == 0) *sum = 0, *sumsq = 0;
    else if (!aligned) nk_reduce_moments_i32_serial(data, count, stride_bytes, sum, sumsq);
    else if (count > (nk_size_t)(NK_U16_MAX + 1) * 16) {
        nk_size_t left_count = count / 2;
        nk_i64_t left_sum, left_sumsq, right_sum, right_sumsq;
        nk_reduce_moments_i32_skylake(data, left_count, stride_bytes, &left_sum, &left_sumsq);
        nk_reduce_moments_i32_skylake(data + left_count * stride_elements, count - left_count, stride_bytes, &right_sum,
                                      &right_sumsq);
        nk_i64_sadd_(&left_sum, &right_sum, sum);
        nk_i64_sadd_(&left_sumsq, &right_sumsq, sumsq);
    }
    else if (stride_elements == 1) nk_reduce_moments_i32_skylake_contiguous_(data, count, sum, sumsq);
    else nk_reduce_moments_i32_serial(data, count, stride_bytes, sum, sumsq);
}

NK_INTERNAL void nk_reduce_minmax_i32_skylake_contiguous_( //
    nk_i32_t const *data, nk_size_t count,                 //
    nk_i32_t *min_value, nk_size_t *min_index,             //
    nk_i32_t *max_value, nk_size_t *max_index) {
    __m512i min_i32x16 = _mm512_loadu_si512(data);
    __m512i max_i32x16 = min_i32x16;
    __m512i min_iter_u32x16 = _mm512_setzero_si512();
    __m512i max_iter_u32x16 = _mm512_setzero_si512();
    __m512i iter_u32x16 = _mm512_set1_epi32(1);
    __m512i one_u32x16 = _mm512_set1_epi32(1);

    nk_size_t idx = 16;
    for (; idx + 16 <= count; idx += 16) {
        __m512i data_i32x16 = _mm512_loadu_si512(data + idx);
        __mmask16 min_changed_mask = _mm512_cmp_epi32_mask(data_i32x16, min_i32x16, _MM_CMPINT_LT);
        __mmask16 max_changed_mask = _mm512_cmp_epi32_mask(data_i32x16, max_i32x16, _MM_CMPINT_NLE);
        min_i32x16 = _mm512_mask_mov_epi32(min_i32x16, min_changed_mask, data_i32x16);
        max_i32x16 = _mm512_mask_mov_epi32(max_i32x16, max_changed_mask, data_i32x16);
        min_iter_u32x16 = _mm512_mask_mov_epi32(min_iter_u32x16, min_changed_mask, iter_u32x16);
        max_iter_u32x16 = _mm512_mask_mov_epi32(max_iter_u32x16, max_changed_mask, iter_u32x16);
        iter_u32x16 = _mm512_add_epi32(iter_u32x16, one_u32x16);
    }

    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        __mmask16 tail_load = (__mmask16)_bzhi_u32(0xFFFF, (unsigned int)remaining);
        __m512i tail_i32x16 = _mm512_maskz_loadu_epi32(tail_load, data + idx);
        __mmask16 min_changed_mask = _mm512_mask_cmp_epi32_mask(tail_load, tail_i32x16, min_i32x16, _MM_CMPINT_LT);
        __mmask16 max_changed_mask = _mm512_mask_cmp_epi32_mask(tail_load, tail_i32x16, max_i32x16, _MM_CMPINT_NLE);
        min_i32x16 = _mm512_mask_mov_epi32(min_i32x16, min_changed_mask, tail_i32x16);
        max_i32x16 = _mm512_mask_mov_epi32(max_i32x16, max_changed_mask, tail_i32x16);
        min_iter_u32x16 = _mm512_mask_mov_epi32(min_iter_u32x16, min_changed_mask, iter_u32x16);
        max_iter_u32x16 = _mm512_mask_mov_epi32(max_iter_u32x16, max_changed_mask, iter_u32x16);
    }

    nk_i32_t min_scalar = nk_reduce_min_i32x16_skylake_(min_i32x16);
    nk_i32_t max_scalar = nk_reduce_max_i32x16_skylake_(max_i32x16);
    __mmask16 min_equality_mask = _mm512_cmpeq_epi32_mask(min_i32x16, _mm512_set1_epi32(min_scalar));
    __mmask16 max_equality_mask = _mm512_cmpeq_epi32_mask(max_i32x16, _mm512_set1_epi32(max_scalar));
    unsigned int min_lane = _tzcnt_u32(min_equality_mask);
    unsigned int max_lane = _tzcnt_u32(max_equality_mask);
    nk_b512_vec_t iter_vec;
    iter_vec.zmm = min_iter_u32x16;
    *min_value = min_scalar, *min_index = (nk_size_t)iter_vec.u32s[min_lane] * 16 + min_lane;
    iter_vec.zmm = max_iter_u32x16;
    *max_value = max_scalar, *max_index = (nk_size_t)iter_vec.u32s[max_lane] * 16 + max_lane;
}

NK_PUBLIC void nk_reduce_minmax_i32_skylake(                       //
    nk_i32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i32_t *min_value, nk_size_t *min_index,                     //
    nk_i32_t *max_value, nk_size_t *max_index) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_i32_t);
    int aligned = (stride_bytes % sizeof(nk_i32_t) == 0);
    if (count == 0) *min_value = NK_I32_MAX, *min_index = 0, *max_value = NK_I32_MIN, *max_index = 0;
    else if (!aligned)
        nk_reduce_minmax_i32_serial(data, count, stride_bytes, min_value, min_index, max_value, max_index);
    else if (count > (nk_size_t)NK_U32_MAX * 16) {
        nk_size_t left_count = count / 2;
        nk_i32_t left_min, right_min, left_max, right_max;
        nk_size_t left_min_index, right_min_index, left_max_index, right_max_index;
        nk_reduce_minmax_i32_skylake(data, left_count, stride_bytes, &left_min, &left_min_index, &left_max,
                                     &left_max_index);
        nk_reduce_minmax_i32_skylake(data + left_count * stride_elements, count - left_count, stride_bytes, &right_min,
                                     &right_min_index, &right_max, &right_max_index);
        if (right_min < left_min) *min_value = right_min, *min_index = left_count + right_min_index;
        else *min_value = left_min, *min_index = left_min_index;
        if (right_max > left_max) *max_value = right_max, *max_index = left_count + right_max_index;
        else *max_value = left_max, *max_index = left_max_index;
    }
    else if (stride_elements == 1)
        nk_reduce_minmax_i32_skylake_contiguous_(data, count, min_value, min_index, max_value, max_index);
    else nk_reduce_minmax_i32_serial(data, count, stride_bytes, min_value, min_index, max_value, max_index);
}

NK_INTERNAL void nk_reduce_moments_u32_skylake_contiguous_( //
    nk_u32_t const *data, nk_size_t count,                  //
    nk_u64_t *sum, nk_u64_t *sumsq) {
    // Sum: widen u32→u64, accumulate. Sumsq: VPMULUDQ for even/odd lanes (5-cycle, 1 uop each).
    __m512i sum_u64x8 = _mm512_setzero_si512();
    __m512i sumsq_u64x8 = _mm512_setzero_si512();
    nk_size_t idx = 0;
    for (; idx + 16 <= count; idx += 16) {
        __m512i data_u32x16 = _mm512_loadu_si512(data + idx);
        __m256i lo_u32x8 = _mm512_castsi512_si256(data_u32x16);
        __m256i hi_u32x8 = _mm512_extracti64x4_epi64(data_u32x16, 1);
        sum_u64x8 = _mm512_add_epi64(sum_u64x8, _mm512_cvtepu32_epi64(lo_u32x8));
        sum_u64x8 = _mm512_add_epi64(sum_u64x8, _mm512_cvtepu32_epi64(hi_u32x8));
        __m512i even_sq_u64x8 = _mm512_mul_epu32(data_u32x16, data_u32x16);
        __m512i odd_u32x16 = _mm512_srli_epi64(data_u32x16, 32);
        __m512i odd_sq_u64x8 = _mm512_mul_epu32(odd_u32x16, odd_u32x16);
        sumsq_u64x8 = _mm512_add_epi64(sumsq_u64x8, even_sq_u64x8);
        sumsq_u64x8 = _mm512_add_epi64(sumsq_u64x8, odd_sq_u64x8);
    }
    nk_u64_t s = nk_reduce_add_u64x8_skylake_(sum_u64x8);
    nk_u64_t sq = nk_reduce_add_u64x8_skylake_(sumsq_u64x8);
    for (; idx < count; ++idx) {
        nk_u64_t val = (nk_u64_t)data[idx];
        s += val;
        sq += val * val;
    }
    *sum = s;
    *sumsq = sq;
}

NK_PUBLIC void nk_reduce_moments_u32_skylake(                      //
    nk_u32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *sum, nk_u64_t *sumsq) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_u32_t);
    int aligned = (stride_bytes % sizeof(nk_u32_t) == 0);
    if (count == 0) *sum = 0, *sumsq = 0;
    else if (!aligned) nk_reduce_moments_u32_serial(data, count, stride_bytes, sum, sumsq);
    else if (count > (nk_size_t)(NK_U16_MAX + 1) * 16) {
        nk_size_t left_count = count / 2;
        nk_u64_t left_sum, left_sumsq, right_sum, right_sumsq;
        nk_reduce_moments_u32_skylake(data, left_count, stride_bytes, &left_sum, &left_sumsq);
        nk_reduce_moments_u32_skylake(data + left_count * stride_elements, count - left_count, stride_bytes, &right_sum,
                                      &right_sumsq);
        nk_u64_sadd_(&left_sum, &right_sum, sum);
        nk_u64_sadd_(&left_sumsq, &right_sumsq, sumsq);
    }
    else if (stride_elements == 1) nk_reduce_moments_u32_skylake_contiguous_(data, count, sum, sumsq);
    else nk_reduce_moments_u32_serial(data, count, stride_bytes, sum, sumsq);
}

NK_INTERNAL void nk_reduce_minmax_u32_skylake_contiguous_( //
    nk_u32_t const *data, nk_size_t count,                 //
    nk_u32_t *min_value, nk_size_t *min_index,             //
    nk_u32_t *max_value, nk_size_t *max_index) {
    __m512i min_u32x16 = _mm512_loadu_si512(data);
    __m512i max_u32x16 = min_u32x16;
    __m512i min_iter_u32x16 = _mm512_setzero_si512();
    __m512i max_iter_u32x16 = _mm512_setzero_si512();
    __m512i iter_u32x16 = _mm512_set1_epi32(1);
    __m512i one_u32x16 = _mm512_set1_epi32(1);

    nk_size_t idx = 16;
    for (; idx + 16 <= count; idx += 16) {
        __m512i data_u32x16 = _mm512_loadu_si512(data + idx);
        __mmask16 min_changed_mask = _mm512_cmp_epu32_mask(data_u32x16, min_u32x16, _MM_CMPINT_LT);
        __mmask16 max_changed_mask = _mm512_cmp_epu32_mask(data_u32x16, max_u32x16, _MM_CMPINT_NLE);
        min_u32x16 = _mm512_mask_mov_epi32(min_u32x16, min_changed_mask, data_u32x16);
        max_u32x16 = _mm512_mask_mov_epi32(max_u32x16, max_changed_mask, data_u32x16);
        min_iter_u32x16 = _mm512_mask_mov_epi32(min_iter_u32x16, min_changed_mask, iter_u32x16);
        max_iter_u32x16 = _mm512_mask_mov_epi32(max_iter_u32x16, max_changed_mask, iter_u32x16);
        iter_u32x16 = _mm512_add_epi32(iter_u32x16, one_u32x16);
    }

    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        __mmask16 tail_load = (__mmask16)_bzhi_u32(0xFFFF, (unsigned int)remaining);
        __m512i tail_u32x16 = _mm512_maskz_loadu_epi32(tail_load, data + idx);
        __mmask16 min_changed_mask = _mm512_mask_cmp_epu32_mask(tail_load, tail_u32x16, min_u32x16, _MM_CMPINT_LT);
        __mmask16 max_changed_mask = _mm512_mask_cmp_epu32_mask(tail_load, tail_u32x16, max_u32x16, _MM_CMPINT_NLE);
        min_u32x16 = _mm512_mask_mov_epi32(min_u32x16, min_changed_mask, tail_u32x16);
        max_u32x16 = _mm512_mask_mov_epi32(max_u32x16, max_changed_mask, tail_u32x16);
        min_iter_u32x16 = _mm512_mask_mov_epi32(min_iter_u32x16, min_changed_mask, iter_u32x16);
        max_iter_u32x16 = _mm512_mask_mov_epi32(max_iter_u32x16, max_changed_mask, iter_u32x16);
    }

    nk_u32_t min_scalar = nk_reduce_min_u32x16_skylake_(min_u32x16);
    nk_u32_t max_scalar = nk_reduce_max_u32x16_skylake_(max_u32x16);
    __mmask16 min_equality_mask = _mm512_cmpeq_epi32_mask(min_u32x16, _mm512_set1_epi32((nk_i32_t)min_scalar));
    __mmask16 max_equality_mask = _mm512_cmpeq_epi32_mask(max_u32x16, _mm512_set1_epi32((nk_i32_t)max_scalar));
    unsigned int min_lane = _tzcnt_u32(min_equality_mask);
    unsigned int max_lane = _tzcnt_u32(max_equality_mask);
    nk_b512_vec_t iter_vec;
    iter_vec.zmm = min_iter_u32x16;
    *min_value = min_scalar, *min_index = (nk_size_t)iter_vec.u32s[min_lane] * 16 + min_lane;
    iter_vec.zmm = max_iter_u32x16;
    *max_value = max_scalar, *max_index = (nk_size_t)iter_vec.u32s[max_lane] * 16 + max_lane;
}

NK_PUBLIC void nk_reduce_minmax_u32_skylake(                       //
    nk_u32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u32_t *min_value, nk_size_t *min_index,                     //
    nk_u32_t *max_value, nk_size_t *max_index) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_u32_t);
    int aligned = (stride_bytes % sizeof(nk_u32_t) == 0);
    if (count == 0) *min_value = NK_U32_MAX, *min_index = 0, *max_value = 0, *max_index = 0;
    else if (!aligned)
        nk_reduce_minmax_u32_serial(data, count, stride_bytes, min_value, min_index, max_value, max_index);
    else if (count > (nk_size_t)NK_U32_MAX * 16) {
        nk_size_t left_count = count / 2;
        nk_u32_t left_min, right_min, left_max, right_max;
        nk_size_t left_min_index, right_min_index, left_max_index, right_max_index;
        nk_reduce_minmax_u32_skylake(data, left_count, stride_bytes, &left_min, &left_min_index, &left_max,
                                     &left_max_index);
        nk_reduce_minmax_u32_skylake(data + left_count * stride_elements, count - left_count, stride_bytes, &right_min,
                                     &right_min_index, &right_max, &right_max_index);
        if (right_min < left_min) *min_value = right_min, *min_index = left_count + right_min_index;
        else *min_value = left_min, *min_index = left_min_index;
        if (right_max > left_max) *max_value = right_max, *max_index = left_count + right_max_index;
        else *max_value = left_max, *max_index = left_max_index;
    }
    else if (stride_elements == 1)
        nk_reduce_minmax_u32_skylake_contiguous_(data, count, min_value, min_index, max_value, max_index);
    else nk_reduce_minmax_u32_serial(data, count, stride_bytes, min_value, min_index, max_value, max_index);
}

NK_INTERNAL void nk_reduce_moments_i64_skylake_contiguous_( //
    nk_i64_t const *data, nk_size_t count,                  //
    nk_i64_t *sum, nk_i64_t *sumsq) {
    // Sum: direct i64 accumulation (8 per iteration).
    // Sumsq: VPMULLQ (15-cycle, 3 uops) — unavoidable for i64 squaring. Overflow accepted.
    __m512i sum_i64x8 = _mm512_setzero_si512();
    __m512i sumsq_i64x8 = _mm512_setzero_si512();
    nk_size_t idx = 0;
    for (; idx + 8 <= count; idx += 8) {
        __m512i data_i64x8 = _mm512_loadu_si512(data + idx);
        sum_i64x8 = _mm512_add_epi64(sum_i64x8, data_i64x8);
        sumsq_i64x8 = _mm512_add_epi64(sumsq_i64x8, _mm512_mullo_epi64(data_i64x8, data_i64x8));
    }
    nk_i64_t s = nk_reduce_add_i64x8_skylake_(sum_i64x8);
    nk_i64_t sq = nk_reduce_add_i64x8_skylake_(sumsq_i64x8);
    for (; idx < count; ++idx) {
        s += data[idx];
        sq += data[idx] * data[idx];
    }
    *sum = s;
    *sumsq = sq;
}

NK_PUBLIC void nk_reduce_moments_i64_skylake(                      //
    nk_i64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *sum, nk_i64_t *sumsq) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_i64_t);
    int aligned = (stride_bytes % sizeof(nk_i64_t) == 0);
    if (count == 0) *sum = 0, *sumsq = 0;
    else if (!aligned) nk_reduce_moments_i64_serial(data, count, stride_bytes, sum, sumsq);
    else if (count > (nk_size_t)(NK_U16_MAX + 1) * 8) {
        nk_size_t left_count = count / 2;
        nk_i64_t left_sum, left_sumsq, right_sum, right_sumsq;
        nk_reduce_moments_i64_skylake(data, left_count, stride_bytes, &left_sum, &left_sumsq);
        nk_reduce_moments_i64_skylake(data + left_count * stride_elements, count - left_count, stride_bytes, &right_sum,
                                      &right_sumsq);
        nk_i64_sadd_(&left_sum, &right_sum, sum);
        nk_i64_sadd_(&left_sumsq, &right_sumsq, sumsq);
    }
    else if (stride_elements == 1) nk_reduce_moments_i64_skylake_contiguous_(data, count, sum, sumsq);
    else nk_reduce_moments_i64_serial(data, count, stride_bytes, sum, sumsq);
}

NK_INTERNAL void nk_reduce_minmax_i64_skylake_contiguous_( //
    nk_i64_t const *data, nk_size_t count,                 //
    nk_i64_t *min_value, nk_size_t *min_index,             //
    nk_i64_t *max_value, nk_size_t *max_index) {
    __m512i min_i64x8 = _mm512_loadu_si512(data);
    __m512i max_i64x8 = min_i64x8;
    __m512i min_iter_u64x8 = _mm512_setzero_si512();
    __m512i max_iter_u64x8 = _mm512_setzero_si512();
    __m512i iter_u64x8 = _mm512_set1_epi64(1);
    __m512i one_u64x8 = _mm512_set1_epi64(1);

    nk_size_t idx = 8;
    for (; idx + 8 <= count; idx += 8) {
        __m512i data_i64x8 = _mm512_loadu_si512(data + idx);
        __mmask8 min_changed_mask = _mm512_cmp_epi64_mask(data_i64x8, min_i64x8, _MM_CMPINT_LT);
        __mmask8 max_changed_mask = _mm512_cmp_epi64_mask(data_i64x8, max_i64x8, _MM_CMPINT_NLE);
        min_i64x8 = _mm512_mask_mov_epi64(min_i64x8, min_changed_mask, data_i64x8);
        max_i64x8 = _mm512_mask_mov_epi64(max_i64x8, max_changed_mask, data_i64x8);
        min_iter_u64x8 = _mm512_mask_mov_epi64(min_iter_u64x8, min_changed_mask, iter_u64x8);
        max_iter_u64x8 = _mm512_mask_mov_epi64(max_iter_u64x8, max_changed_mask, iter_u64x8);
        iter_u64x8 = _mm512_add_epi64(iter_u64x8, one_u64x8);
    }

    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        __mmask8 tail_load = (__mmask8)_bzhi_u32(0xFF, (unsigned int)remaining);
        __m512i tail_i64x8 = _mm512_maskz_loadu_epi64(tail_load, data + idx);
        __mmask8 min_changed_mask = _mm512_mask_cmp_epi64_mask(tail_load, tail_i64x8, min_i64x8, _MM_CMPINT_LT);
        __mmask8 max_changed_mask = _mm512_mask_cmp_epi64_mask(tail_load, tail_i64x8, max_i64x8, _MM_CMPINT_NLE);
        min_i64x8 = _mm512_mask_mov_epi64(min_i64x8, min_changed_mask, tail_i64x8);
        max_i64x8 = _mm512_mask_mov_epi64(max_i64x8, max_changed_mask, tail_i64x8);
        min_iter_u64x8 = _mm512_mask_mov_epi64(min_iter_u64x8, min_changed_mask, iter_u64x8);
        max_iter_u64x8 = _mm512_mask_mov_epi64(max_iter_u64x8, max_changed_mask, iter_u64x8);
    }

    nk_i64_t min_scalar = nk_reduce_min_i64x8_skylake_(min_i64x8);
    nk_i64_t max_scalar = nk_reduce_max_i64x8_skylake_(max_i64x8);
    __mmask8 min_equality_mask = _mm512_cmpeq_epi64_mask(min_i64x8, _mm512_set1_epi64(min_scalar));
    __mmask8 max_equality_mask = _mm512_cmpeq_epi64_mask(max_i64x8, _mm512_set1_epi64(max_scalar));
    unsigned int min_lane = _tzcnt_u32(min_equality_mask);
    unsigned int max_lane = _tzcnt_u32(max_equality_mask);
    nk_b512_vec_t iter_vec;
    iter_vec.zmm = min_iter_u64x8;
    *min_value = min_scalar, *min_index = (nk_size_t)iter_vec.u64s[min_lane] * 8 + min_lane;
    iter_vec.zmm = max_iter_u64x8;
    *max_value = max_scalar, *max_index = (nk_size_t)iter_vec.u64s[max_lane] * 8 + max_lane;
}

NK_PUBLIC void nk_reduce_minmax_i64_skylake(                       //
    nk_i64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *min_value, nk_size_t *min_index,                     //
    nk_i64_t *max_value, nk_size_t *max_index) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_i64_t);
    int aligned = (stride_bytes % sizeof(nk_i64_t) == 0);
    if (count == 0) *min_value = NK_I64_MAX, *min_index = 0, *max_value = NK_I64_MIN, *max_index = 0;
    else if (!aligned)
        nk_reduce_minmax_i64_serial(data, count, stride_bytes, min_value, min_index, max_value, max_index);
    else if (stride_elements == 1)
        nk_reduce_minmax_i64_skylake_contiguous_(data, count, min_value, min_index, max_value, max_index);
    else nk_reduce_minmax_i64_serial(data, count, stride_bytes, min_value, min_index, max_value, max_index);
}

NK_INTERNAL void nk_reduce_moments_u64_skylake_contiguous_( //
    nk_u64_t const *data, nk_size_t count,                  //
    nk_u64_t *sum, nk_u64_t *sumsq) {
    __m512i sum_u64x8 = _mm512_setzero_si512();
    __m512i sumsq_u64x8 = _mm512_setzero_si512();
    nk_size_t idx = 0;
    for (; idx + 8 <= count; idx += 8) {
        __m512i data_u64x8 = _mm512_loadu_si512(data + idx);
        sum_u64x8 = _mm512_add_epi64(sum_u64x8, data_u64x8);
        sumsq_u64x8 = _mm512_add_epi64(sumsq_u64x8, _mm512_mullo_epi64(data_u64x8, data_u64x8));
    }
    nk_u64_t s = nk_reduce_add_u64x8_skylake_(sum_u64x8);
    nk_u64_t sq = nk_reduce_add_u64x8_skylake_(sumsq_u64x8);
    for (; idx < count; ++idx) {
        s += data[idx];
        sq += data[idx] * data[idx];
    }
    *sum = s;
    *sumsq = sq;
}

NK_PUBLIC void nk_reduce_moments_u64_skylake(                      //
    nk_u64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *sum, nk_u64_t *sumsq) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_u64_t);
    int aligned = (stride_bytes % sizeof(nk_u64_t) == 0);
    if (count == 0) *sum = 0, *sumsq = 0;
    else if (!aligned) nk_reduce_moments_u64_serial(data, count, stride_bytes, sum, sumsq);
    else if (count > (nk_size_t)(NK_U16_MAX + 1) * 8) {
        nk_size_t left_count = count / 2;
        nk_u64_t left_sum, left_sumsq, right_sum, right_sumsq;
        nk_reduce_moments_u64_skylake(data, left_count, stride_bytes, &left_sum, &left_sumsq);
        nk_reduce_moments_u64_skylake(data + left_count * stride_elements, count - left_count, stride_bytes, &right_sum,
                                      &right_sumsq);
        nk_u64_sadd_(&left_sum, &right_sum, sum);
        nk_u64_sadd_(&left_sumsq, &right_sumsq, sumsq);
    }
    else if (stride_elements == 1) nk_reduce_moments_u64_skylake_contiguous_(data, count, sum, sumsq);
    else nk_reduce_moments_u64_serial(data, count, stride_bytes, sum, sumsq);
}

NK_INTERNAL void nk_reduce_minmax_u64_skylake_contiguous_( //
    nk_u64_t const *data, nk_size_t count,                 //
    nk_u64_t *min_value, nk_size_t *min_index,             //
    nk_u64_t *max_value, nk_size_t *max_index) {
    __m512i min_u64x8 = _mm512_loadu_si512(data);
    __m512i max_u64x8 = min_u64x8;
    __m512i min_iter_u64x8 = _mm512_setzero_si512();
    __m512i max_iter_u64x8 = _mm512_setzero_si512();
    __m512i iter_u64x8 = _mm512_set1_epi64(1);
    __m512i one_u64x8 = _mm512_set1_epi64(1);

    nk_size_t idx = 8;
    for (; idx + 8 <= count; idx += 8) {
        __m512i data_u64x8 = _mm512_loadu_si512(data + idx);
        __mmask8 min_changed_mask = _mm512_cmp_epu64_mask(data_u64x8, min_u64x8, _MM_CMPINT_LT);
        __mmask8 max_changed_mask = _mm512_cmp_epu64_mask(data_u64x8, max_u64x8, _MM_CMPINT_NLE);
        min_u64x8 = _mm512_mask_mov_epi64(min_u64x8, min_changed_mask, data_u64x8);
        max_u64x8 = _mm512_mask_mov_epi64(max_u64x8, max_changed_mask, data_u64x8);
        min_iter_u64x8 = _mm512_mask_mov_epi64(min_iter_u64x8, min_changed_mask, iter_u64x8);
        max_iter_u64x8 = _mm512_mask_mov_epi64(max_iter_u64x8, max_changed_mask, iter_u64x8);
        iter_u64x8 = _mm512_add_epi64(iter_u64x8, one_u64x8);
    }

    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        __mmask8 tail_load = (__mmask8)_bzhi_u32(0xFF, (unsigned int)remaining);
        __m512i tail_u64x8 = _mm512_maskz_loadu_epi64(tail_load, data + idx);
        __mmask8 min_changed_mask = _mm512_mask_cmp_epu64_mask(tail_load, tail_u64x8, min_u64x8, _MM_CMPINT_LT);
        __mmask8 max_changed_mask = _mm512_mask_cmp_epu64_mask(tail_load, tail_u64x8, max_u64x8, _MM_CMPINT_NLE);
        min_u64x8 = _mm512_mask_mov_epi64(min_u64x8, min_changed_mask, tail_u64x8);
        max_u64x8 = _mm512_mask_mov_epi64(max_u64x8, max_changed_mask, tail_u64x8);
        min_iter_u64x8 = _mm512_mask_mov_epi64(min_iter_u64x8, min_changed_mask, iter_u64x8);
        max_iter_u64x8 = _mm512_mask_mov_epi64(max_iter_u64x8, max_changed_mask, iter_u64x8);
    }

    nk_u64_t min_scalar = nk_reduce_min_u64x8_skylake_(min_u64x8);
    nk_u64_t max_scalar = nk_reduce_max_u64x8_skylake_(max_u64x8);
    __mmask8 min_equality_mask = _mm512_cmpeq_epi64_mask(min_u64x8, _mm512_set1_epi64((nk_i64_t)min_scalar));
    __mmask8 max_equality_mask = _mm512_cmpeq_epi64_mask(max_u64x8, _mm512_set1_epi64((nk_i64_t)max_scalar));
    unsigned int min_lane = _tzcnt_u32(min_equality_mask);
    unsigned int max_lane = _tzcnt_u32(max_equality_mask);
    nk_b512_vec_t iter_vec;
    iter_vec.zmm = min_iter_u64x8;
    *min_value = min_scalar, *min_index = (nk_size_t)iter_vec.u64s[min_lane] * 8 + min_lane;
    iter_vec.zmm = max_iter_u64x8;
    *max_value = max_scalar, *max_index = (nk_size_t)iter_vec.u64s[max_lane] * 8 + max_lane;
}

NK_PUBLIC void nk_reduce_minmax_u64_skylake(                       //
    nk_u64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *min_value, nk_size_t *min_index,                     //
    nk_u64_t *max_value, nk_size_t *max_index) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_u64_t);
    int aligned = (stride_bytes % sizeof(nk_u64_t) == 0);
    if (count == 0) *min_value = NK_U64_MAX, *min_index = 0, *max_value = 0, *max_index = 0;
    else if (!aligned)
        nk_reduce_minmax_u64_serial(data, count, stride_bytes, min_value, min_index, max_value, max_index);
    else if (stride_elements == 1)
        nk_reduce_minmax_u64_skylake_contiguous_(data, count, min_value, min_index, max_value, max_index);
    else nk_reduce_minmax_u64_serial(data, count, stride_bytes, min_value, min_index, max_value, max_index);
}

NK_INTERNAL void nk_reduce_minmax_f64_skylake_contiguous_( //
    nk_f64_t const *data, nk_size_t count,                 //
    nk_f64_t *min_value, nk_size_t *min_index,             //
    nk_f64_t *max_value, nk_size_t *max_index) {
    __m512d min_f64x8 = _mm512_loadu_pd(data);
    __m512d max_f64x8 = min_f64x8;
    __m512i min_iter_u64x8 = _mm512_setzero_si512();
    __m512i max_iter_u64x8 = _mm512_setzero_si512();
    __m512i iter_u64x8 = _mm512_set1_epi64(1);
    __m512i one_u64x8 = _mm512_set1_epi64(1);

    nk_size_t idx = 8;
    for (; idx + 8 <= count; idx += 8) {
        __m512d data_f64x8 = _mm512_loadu_pd(data + idx);
        __mmask8 min_changed_mask = _mm512_cmp_pd_mask(data_f64x8, min_f64x8, _CMP_LT_OQ);
        __mmask8 max_changed_mask = _mm512_cmp_pd_mask(data_f64x8, max_f64x8, _CMP_GT_OQ);
        min_f64x8 = _mm512_mask_mov_pd(min_f64x8, min_changed_mask, data_f64x8);
        max_f64x8 = _mm512_mask_mov_pd(max_f64x8, max_changed_mask, data_f64x8);
        min_iter_u64x8 = _mm512_mask_mov_epi64(min_iter_u64x8, min_changed_mask, iter_u64x8);
        max_iter_u64x8 = _mm512_mask_mov_epi64(max_iter_u64x8, max_changed_mask, iter_u64x8);
        iter_u64x8 = _mm512_add_epi64(iter_u64x8, one_u64x8);
    }

    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        __mmask8 tail_load = (__mmask8)_bzhi_u32(0xFF, (unsigned int)remaining);
        __m512d tail_f64x8 = _mm512_maskz_loadu_pd(tail_load, data + idx);
        __mmask8 min_changed_mask = _mm512_mask_cmp_pd_mask(tail_load, tail_f64x8, min_f64x8, _CMP_LT_OQ);
        __mmask8 max_changed_mask = _mm512_mask_cmp_pd_mask(tail_load, tail_f64x8, max_f64x8, _CMP_GT_OQ);
        min_f64x8 = _mm512_mask_mov_pd(min_f64x8, min_changed_mask, tail_f64x8);
        max_f64x8 = _mm512_mask_mov_pd(max_f64x8, max_changed_mask, tail_f64x8);
        min_iter_u64x8 = _mm512_mask_mov_epi64(min_iter_u64x8, min_changed_mask, iter_u64x8);
        max_iter_u64x8 = _mm512_mask_mov_epi64(max_iter_u64x8, max_changed_mask, iter_u64x8);
    }

    nk_f64_t min_scalar = nk_reduce_min_f64x8_skylake_(min_f64x8);
    nk_f64_t max_scalar = nk_reduce_max_f64x8_skylake_(max_f64x8);
    __mmask8 min_equality_mask = _mm512_cmp_pd_mask(min_f64x8, _mm512_set1_pd(min_scalar), _CMP_EQ_OQ);
    __mmask8 max_equality_mask = _mm512_cmp_pd_mask(max_f64x8, _mm512_set1_pd(max_scalar), _CMP_EQ_OQ);
    unsigned int min_lane = _tzcnt_u32(min_equality_mask);
    unsigned int max_lane = _tzcnt_u32(max_equality_mask);
    nk_b512_vec_t iter_vec;
    iter_vec.zmm = min_iter_u64x8;
    *min_value = min_scalar, *min_index = (nk_size_t)iter_vec.u64s[min_lane] * 8 + min_lane;
    iter_vec.zmm = max_iter_u64x8;
    *max_value = max_scalar, *max_index = (nk_size_t)iter_vec.u64s[max_lane] * 8 + max_lane;
}

NK_PUBLIC void nk_reduce_minmax_f64_skylake(                       //
    nk_f64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f64_t *min_value, nk_size_t *min_index,                     //
    nk_f64_t *max_value, nk_size_t *max_index) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_f64_t);
    int aligned = (stride_bytes % sizeof(nk_f64_t) == 0);
    if (count == 0) *min_value = NK_F64_MAX, *min_index = 0, *max_value = NK_F64_MIN, *max_index = 0;
    else if (!aligned)
        nk_reduce_minmax_f64_serial(data, count, stride_bytes, min_value, min_index, max_value, max_index);
    else if (stride_elements == 1)
        nk_reduce_minmax_f64_skylake_contiguous_(data, count, min_value, min_index, max_value, max_index);
    else nk_reduce_minmax_f64_serial(data, count, stride_bytes, min_value, min_index, max_value, max_index);
}

NK_INTERNAL void nk_reduce_moments_e4m3_skylake_contiguous_( //
    nk_e4m3_t const *data, nk_size_t count,                  //
    nk_f32_t *sum, nk_f32_t *sumsq) {
    __m512 sum_f32x16 = _mm512_setzero_ps();
    __m512 sumsq_f32x16 = _mm512_setzero_ps();
    nk_size_t idx = 0;
    for (; idx + 16 <= count; idx += 16) {
        __m512 data_f32x16 = nk_e4m3x16_to_f32x16_skylake_(_mm_loadu_si128((__m128i const *)(data + idx)));
        sum_f32x16 = _mm512_add_ps(sum_f32x16, data_f32x16);
        sumsq_f32x16 = _mm512_fmadd_ps(data_f32x16, data_f32x16, sumsq_f32x16);
    }
    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        nk_b512_vec_t vec;
        nk_partial_load_e4m3x16_to_f32x16_skylake_(data + idx, &vec, remaining);
        sum_f32x16 = _mm512_add_ps(sum_f32x16, vec.zmm_ps);
        sumsq_f32x16 = _mm512_fmadd_ps(vec.zmm_ps, vec.zmm_ps, sumsq_f32x16);
    }
    *sum = nk_reduce_add_f32x16_skylake_(sum_f32x16);
    *sumsq = nk_reduce_add_f32x16_skylake_(sumsq_f32x16);
}

NK_INTERNAL void nk_reduce_moments_e4m3_skylake_strided_(              //
    nk_e4m3_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_f32_t *sum, nk_f32_t *sumsq) {
    __mmask16 stride_mask_m16 = (__mmask16)nk_stride_mask_u1x64_(stride_elements);
    __m512 sum_f32x16 = _mm512_setzero_ps();
    __m512 sumsq_f32x16 = _mm512_setzero_ps();
    nk_size_t idx_scalars = 0;
    nk_size_t total_scalars = count * stride_elements;
    for (; idx_scalars + 16 <= total_scalars; idx_scalars += 16) {
        __m128i data_e4m3x16 = _mm_maskz_loadu_epi8(stride_mask_m16, data + idx_scalars);
        __m512 data_f32x16 = nk_e4m3x16_to_f32x16_skylake_(data_e4m3x16);
        sum_f32x16 = _mm512_add_ps(sum_f32x16, data_f32x16);
        sumsq_f32x16 = _mm512_fmadd_ps(data_f32x16, data_f32x16, sumsq_f32x16);
    }
    nk_size_t remaining_bytes = total_scalars - idx_scalars;
    if (remaining_bytes > 0) {
        __mmask16 tail_mask = stride_mask_m16 & (__mmask16)_bzhi_u32(0xFFFF, (unsigned int)remaining_bytes);
        __m128i data_e4m3x16 = _mm_maskz_loadu_epi8(tail_mask, data + idx_scalars);
        __m512 data_f32x16 = nk_e4m3x16_to_f32x16_skylake_(data_e4m3x16);
        sum_f32x16 = _mm512_add_ps(sum_f32x16, data_f32x16);
        sumsq_f32x16 = _mm512_fmadd_ps(data_f32x16, data_f32x16, sumsq_f32x16);
    }
    *sum = nk_reduce_add_f32x16_skylake_(sum_f32x16);
    *sumsq = nk_reduce_add_f32x16_skylake_(sumsq_f32x16);
}

NK_PUBLIC void nk_reduce_moments_e4m3_skylake(                      //
    nk_e4m3_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *sum, nk_f32_t *sumsq) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_e4m3_t);
    int aligned = (stride_bytes % sizeof(nk_e4m3_t) == 0);
    if (count == 0) *sum = 0, *sumsq = 0;
    else if (!aligned) nk_reduce_moments_e4m3_serial(data, count, stride_bytes, sum, sumsq);
    else if (count > (nk_size_t)(NK_U16_MAX + 1) * 64) {
        nk_size_t left_count = count / 2;
        nk_f32_t left_sum, left_sumsq, right_sum, right_sumsq;
        nk_reduce_moments_e4m3_skylake(data, left_count, stride_bytes, &left_sum, &left_sumsq);
        nk_reduce_moments_e4m3_skylake(data + left_count * stride_elements, count - left_count, stride_bytes,
                                       &right_sum, &right_sumsq);
        *sum = left_sum + right_sum, *sumsq = left_sumsq + right_sumsq;
    }
    else if (stride_elements == 1) nk_reduce_moments_e4m3_skylake_contiguous_(data, count, sum, sumsq);
    else if (stride_elements >= 2 && stride_elements <= 16)
        nk_reduce_moments_e4m3_skylake_strided_(data, count, stride_elements, sum, sumsq);
    else nk_reduce_moments_e4m3_serial(data, count, stride_bytes, sum, sumsq);
}

NK_INTERNAL void nk_reduce_minmax_e4m3_skylake_contiguous_( //
    nk_e4m3_t const *data, nk_size_t count,                 //
    nk_e4m3_t *min_value, nk_size_t *min_index,             //
    nk_e4m3_t *max_value, nk_size_t *max_index) {
    __m512i nan_cmp_u8x64 = _mm512_set1_epi8((char)0xFF);
    __m512i first_i8x64 = _mm512_loadu_si512(data);
    __m512i first_cmp = nk_fp8x64_to_u8x64_comparable_skylake_(first_i8x64);
    nk_b512_vec_t min_vec, max_vec;
    min_vec.zmm = first_cmp;
    max_vec.zmm = first_cmp;
    __m512i min_iter_u8x64 = _mm512_setzero_si512();
    __m512i max_iter_u8x64 = _mm512_setzero_si512();
    __m512i iter_u8x64 = _mm512_set1_epi8(1);
    __m512i one_u8x64 = _mm512_set1_epi8(1);

    nk_size_t idx = 64;
    for (; idx + 64 <= count; idx += 64) {
        __m512i data_i8x64 = _mm512_loadu_si512(data + idx);
        __m512i data_cmp_u8x64 = nk_fp8x64_to_u8x64_comparable_skylake_(data_i8x64);
        __mmask64 min_mask_m64 = nk_min_mask_e4m3x64_skylake_(min_vec.zmm, data_cmp_u8x64, nan_cmp_u8x64);
        min_vec.zmm = _mm512_mask_mov_epi8(data_cmp_u8x64, min_mask_m64, min_vec.zmm);
        __mmask64 min_changed_mask = ~min_mask_m64;
        min_iter_u8x64 = _mm512_mask_mov_epi8(min_iter_u8x64, min_changed_mask, iter_u8x64);
        __mmask64 max_mask_m64 = nk_max_mask_e4m3x64_skylake_(max_vec.zmm, data_cmp_u8x64, nan_cmp_u8x64);
        max_vec.zmm = _mm512_mask_mov_epi8(data_cmp_u8x64, max_mask_m64, max_vec.zmm);
        __mmask64 max_changed_mask = ~max_mask_m64;
        max_iter_u8x64 = _mm512_mask_mov_epi8(max_iter_u8x64, max_changed_mask, iter_u8x64);
        iter_u8x64 = _mm512_add_epi8(iter_u8x64, one_u8x64);
    }

    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        __mmask64 tail_load = _bzhi_u64(0xFFFFFFFFFFFFFFFFull, (unsigned int)remaining);
        __m512i data_i8x64 = _mm512_mask_loadu_epi8(nan_cmp_u8x64, tail_load, data + idx);
        __m512i data_cmp_u8x64 = nk_fp8x64_to_u8x64_comparable_skylake_(data_i8x64);
        __mmask64 min_mask_m64 = nk_min_mask_e4m3x64_skylake_(min_vec.zmm, data_cmp_u8x64, nan_cmp_u8x64);
        __mmask64 min_changed_mask = tail_load & ~min_mask_m64;
        min_vec.zmm = _mm512_mask_mov_epi8(data_cmp_u8x64, min_mask_m64, min_vec.zmm);
        min_iter_u8x64 = _mm512_mask_mov_epi8(min_iter_u8x64, min_changed_mask, iter_u8x64);
        __mmask64 max_mask_m64 = nk_max_mask_e4m3x64_skylake_(max_vec.zmm, data_cmp_u8x64, nan_cmp_u8x64);
        __mmask64 max_changed_mask = tail_load & ~max_mask_m64;
        max_vec.zmm = _mm512_mask_mov_epi8(data_cmp_u8x64, max_mask_m64, max_vec.zmm);
        max_iter_u8x64 = _mm512_mask_mov_epi8(max_iter_u8x64, max_changed_mask, iter_u8x64);
    }

    nk_size_t min_lane = nk_argmin_u8x64_skylake_(min_vec.zmm);
    nk_size_t max_lane = nk_argmax_u8x64_skylake_(max_vec.zmm);
    nk_b512_vec_t iter_vec;
    iter_vec.zmm = min_iter_u8x64;
    *min_index = (nk_size_t)iter_vec.u8s[min_lane] * 64 + min_lane;
    iter_vec.zmm = max_iter_u8x64;
    *max_index = (nk_size_t)iter_vec.u8s[max_lane] * 64 + max_lane;
    min_vec.zmm = nk_u8x64_comparable_to_fp8x64_skylake_(min_vec.zmm);
    max_vec.zmm = nk_u8x64_comparable_to_fp8x64_skylake_(max_vec.zmm);
    *min_value = min_vec.e4m3s[min_lane];
    *max_value = max_vec.e4m3s[max_lane];
}

NK_PUBLIC void nk_reduce_minmax_e4m3_skylake(                       //
    nk_e4m3_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_e4m3_t *min_value, nk_size_t *min_index,                     //
    nk_e4m3_t *max_value, nk_size_t *max_index) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_e4m3_t);
    if (count == 0) { *min_value = 0x7F, *min_index = 0, *max_value = 0x80, *max_index = 0; }
    else if (stride_elements == 1 && count > (nk_size_t)(NK_U8_MAX + 1) * 64) {
        nk_size_t left_count = count / 2;
        nk_e4m3_t left_min, right_min, left_max, right_max;
        nk_size_t left_min_index, right_min_index, left_max_index, right_max_index;
        nk_reduce_minmax_e4m3_skylake(data, left_count, stride_bytes, &left_min, &left_min_index, &left_max,
                                      &left_max_index);
        nk_reduce_minmax_e4m3_skylake(data + left_count, count - left_count, stride_bytes, &right_min, &right_min_index,
                                      &right_max, &right_max_index);
        if (nk_e4m3_compare_(right_min, left_min) < 0)
            *min_value = right_min, *min_index = left_count + right_min_index;
        else *min_value = left_min, *min_index = left_min_index;
        if (nk_e4m3_compare_(right_max, left_max) > 0)
            *max_value = right_max, *max_index = left_count + right_max_index;
        else *max_value = left_max, *max_index = left_max_index;
    }
    else if (stride_elements == 1)
        nk_reduce_minmax_e4m3_skylake_contiguous_(data, count, min_value, min_index, max_value, max_index);
    else nk_reduce_minmax_e4m3_serial(data, count, stride_bytes, min_value, min_index, max_value, max_index);
}

NK_INTERNAL void nk_reduce_moments_e5m2_skylake_contiguous_( //
    nk_e5m2_t const *data, nk_size_t count,                  //
    nk_f32_t *sum, nk_f32_t *sumsq) {
    __m512 sum_f32x16 = _mm512_setzero_ps();
    __m512 sumsq_f32x16 = _mm512_setzero_ps();
    nk_size_t idx = 0;
    for (; idx + 16 <= count; idx += 16) {
        __m512 data_f32x16 = nk_e5m2x16_to_f32x16_skylake_(_mm_loadu_si128((__m128i const *)(data + idx)));
        sum_f32x16 = _mm512_add_ps(sum_f32x16, data_f32x16);
        sumsq_f32x16 = _mm512_fmadd_ps(data_f32x16, data_f32x16, sumsq_f32x16);
    }
    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        nk_b512_vec_t vec;
        nk_partial_load_e5m2x16_to_f32x16_skylake_(data + idx, &vec, remaining);
        sum_f32x16 = _mm512_add_ps(sum_f32x16, vec.zmm_ps);
        sumsq_f32x16 = _mm512_fmadd_ps(vec.zmm_ps, vec.zmm_ps, sumsq_f32x16);
    }
    *sum = nk_reduce_add_f32x16_skylake_(sum_f32x16);
    *sumsq = nk_reduce_add_f32x16_skylake_(sumsq_f32x16);
}

NK_INTERNAL void nk_reduce_moments_e5m2_skylake_strided_(              //
    nk_e5m2_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_f32_t *sum, nk_f32_t *sumsq) {
    __mmask16 stride_mask_m16 = (__mmask16)nk_stride_mask_u1x64_(stride_elements);
    __m512 sum_f32x16 = _mm512_setzero_ps();
    __m512 sumsq_f32x16 = _mm512_setzero_ps();
    nk_size_t idx_scalars = 0;
    nk_size_t total_scalars = count * stride_elements;
    for (; idx_scalars + 16 <= total_scalars; idx_scalars += 16) {
        __m128i data_e5m2x16 = _mm_maskz_loadu_epi8(stride_mask_m16, data + idx_scalars);
        __m512 data_f32x16 = nk_e5m2x16_to_f32x16_skylake_(data_e5m2x16);
        sum_f32x16 = _mm512_add_ps(sum_f32x16, data_f32x16);
        sumsq_f32x16 = _mm512_fmadd_ps(data_f32x16, data_f32x16, sumsq_f32x16);
    }
    nk_size_t remaining_bytes = total_scalars - idx_scalars;
    if (remaining_bytes > 0) {
        __mmask16 tail_mask = stride_mask_m16 & (__mmask16)_bzhi_u32(0xFFFF, (unsigned int)remaining_bytes);
        __m128i data_e5m2x16 = _mm_maskz_loadu_epi8(tail_mask, data + idx_scalars);
        __m512 data_f32x16 = nk_e5m2x16_to_f32x16_skylake_(data_e5m2x16);
        sum_f32x16 = _mm512_add_ps(sum_f32x16, data_f32x16);
        sumsq_f32x16 = _mm512_fmadd_ps(data_f32x16, data_f32x16, sumsq_f32x16);
    }
    *sum = nk_reduce_add_f32x16_skylake_(sum_f32x16);
    *sumsq = nk_reduce_add_f32x16_skylake_(sumsq_f32x16);
}

NK_PUBLIC void nk_reduce_moments_e5m2_skylake(                      //
    nk_e5m2_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *sum, nk_f32_t *sumsq) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_e5m2_t);
    int aligned = (stride_bytes % sizeof(nk_e5m2_t) == 0);
    if (count == 0) *sum = 0, *sumsq = 0;
    else if (!aligned) nk_reduce_moments_e5m2_serial(data, count, stride_bytes, sum, sumsq);
    else if (count > (nk_size_t)(NK_U16_MAX + 1) * 64) {
        nk_size_t left_count = count / 2;
        nk_f32_t left_sum, left_sumsq, right_sum, right_sumsq;
        nk_reduce_moments_e5m2_skylake(data, left_count, stride_bytes, &left_sum, &left_sumsq);
        nk_reduce_moments_e5m2_skylake(data + left_count * stride_elements, count - left_count, stride_bytes,
                                       &right_sum, &right_sumsq);
        *sum = left_sum + right_sum, *sumsq = left_sumsq + right_sumsq;
    }
    else if (stride_elements == 1) nk_reduce_moments_e5m2_skylake_contiguous_(data, count, sum, sumsq);
    else if (stride_elements >= 2 && stride_elements <= 16)
        nk_reduce_moments_e5m2_skylake_strided_(data, count, stride_elements, sum, sumsq);
    else nk_reduce_moments_e5m2_serial(data, count, stride_bytes, sum, sumsq);
}

NK_INTERNAL void nk_reduce_moments_e2m3_skylake_contiguous_( //
    nk_e2m3_t const *data, nk_size_t count,                  //
    nk_f32_t *sum, nk_f32_t *sumsq) {
    __m512 sum_f32x16 = _mm512_setzero_ps();
    __m512 sumsq_f32x16 = _mm512_setzero_ps();
    nk_size_t idx = 0;
    for (; idx + 16 <= count; idx += 16) {
        __m512 data_f32x16 = nk_e2m3x16_to_f32x16_skylake_(_mm_loadu_si128((__m128i const *)(data + idx)));
        sum_f32x16 = _mm512_add_ps(sum_f32x16, data_f32x16);
        sumsq_f32x16 = _mm512_fmadd_ps(data_f32x16, data_f32x16, sumsq_f32x16);
    }
    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        nk_b512_vec_t vec;
        nk_partial_load_e2m3x16_to_f32x16_skylake_(data + idx, &vec, remaining);
        sum_f32x16 = _mm512_add_ps(sum_f32x16, vec.zmm_ps);
        sumsq_f32x16 = _mm512_fmadd_ps(vec.zmm_ps, vec.zmm_ps, sumsq_f32x16);
    }
    *sum = nk_reduce_add_f32x16_skylake_(sum_f32x16);
    *sumsq = nk_reduce_add_f32x16_skylake_(sumsq_f32x16);
}

NK_INTERNAL void nk_reduce_moments_e2m3_skylake_strided_(              //
    nk_e2m3_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_f32_t *sum, nk_f32_t *sumsq) {
    __mmask16 stride_mask_m16 = (__mmask16)nk_stride_mask_u1x64_(stride_elements);
    __m512 sum_f32x16 = _mm512_setzero_ps();
    __m512 sumsq_f32x16 = _mm512_setzero_ps();
    nk_size_t idx_scalars = 0;
    nk_size_t total_scalars = count * stride_elements;
    for (; idx_scalars + 16 <= total_scalars; idx_scalars += 16) {
        __m128i data_e2m3x16 = _mm_maskz_loadu_epi8(stride_mask_m16, data + idx_scalars);
        __m512 data_f32x16 = nk_e2m3x16_to_f32x16_skylake_(data_e2m3x16);
        sum_f32x16 = _mm512_add_ps(sum_f32x16, data_f32x16);
        sumsq_f32x16 = _mm512_fmadd_ps(data_f32x16, data_f32x16, sumsq_f32x16);
    }
    nk_size_t remaining_bytes = total_scalars - idx_scalars;
    if (remaining_bytes > 0) {
        __mmask16 tail_mask = stride_mask_m16 & (__mmask16)_bzhi_u32(0xFFFF, (unsigned int)remaining_bytes);
        __m128i data_e2m3x16 = _mm_maskz_loadu_epi8(tail_mask, data + idx_scalars);
        __m512 data_f32x16 = nk_e2m3x16_to_f32x16_skylake_(data_e2m3x16);
        sum_f32x16 = _mm512_add_ps(sum_f32x16, data_f32x16);
        sumsq_f32x16 = _mm512_fmadd_ps(data_f32x16, data_f32x16, sumsq_f32x16);
    }
    *sum = nk_reduce_add_f32x16_skylake_(sum_f32x16);
    *sumsq = nk_reduce_add_f32x16_skylake_(sumsq_f32x16);
}

NK_PUBLIC void nk_reduce_moments_e2m3_skylake(                      //
    nk_e2m3_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *sum, nk_f32_t *sumsq) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_e2m3_t);
    int aligned = (stride_bytes % sizeof(nk_e2m3_t) == 0);
    if (count == 0) *sum = 0, *sumsq = 0;
    else if (!aligned) nk_reduce_moments_e2m3_serial(data, count, stride_bytes, sum, sumsq);
    else if (count > (nk_size_t)(NK_U16_MAX + 1) * 64) {
        nk_size_t left_count = count / 2;
        nk_f32_t left_sum, left_sumsq, right_sum, right_sumsq;
        nk_reduce_moments_e2m3_skylake(data, left_count, stride_bytes, &left_sum, &left_sumsq);
        nk_reduce_moments_e2m3_skylake(data + left_count * stride_elements, count - left_count, stride_bytes,
                                       &right_sum, &right_sumsq);
        *sum = left_sum + right_sum, *sumsq = left_sumsq + right_sumsq;
    }
    else if (stride_elements == 1) nk_reduce_moments_e2m3_skylake_contiguous_(data, count, sum, sumsq);
    else if (stride_elements >= 2 && stride_elements <= 16)
        nk_reduce_moments_e2m3_skylake_strided_(data, count, stride_elements, sum, sumsq);
    else nk_reduce_moments_e2m3_serial(data, count, stride_bytes, sum, sumsq);
}

NK_INTERNAL void nk_reduce_moments_e3m2_skylake_contiguous_( //
    nk_e3m2_t const *data, nk_size_t count,                  //
    nk_f32_t *sum, nk_f32_t *sumsq) {
    __m512 sum_f32x16 = _mm512_setzero_ps();
    __m512 sumsq_f32x16 = _mm512_setzero_ps();
    nk_size_t idx = 0;
    for (; idx + 16 <= count; idx += 16) {
        __m512 data_f32x16 = nk_e3m2x16_to_f32x16_skylake_(_mm_loadu_si128((__m128i const *)(data + idx)));
        sum_f32x16 = _mm512_add_ps(sum_f32x16, data_f32x16);
        sumsq_f32x16 = _mm512_fmadd_ps(data_f32x16, data_f32x16, sumsq_f32x16);
    }
    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        nk_b512_vec_t vec;
        nk_partial_load_e3m2x16_to_f32x16_skylake_(data + idx, &vec, remaining);
        sum_f32x16 = _mm512_add_ps(sum_f32x16, vec.zmm_ps);
        sumsq_f32x16 = _mm512_fmadd_ps(vec.zmm_ps, vec.zmm_ps, sumsq_f32x16);
    }
    *sum = nk_reduce_add_f32x16_skylake_(sum_f32x16);
    *sumsq = nk_reduce_add_f32x16_skylake_(sumsq_f32x16);
}

NK_INTERNAL void nk_reduce_moments_e3m2_skylake_strided_(              //
    nk_e3m2_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_f32_t *sum, nk_f32_t *sumsq) {
    __mmask16 stride_mask_m16 = (__mmask16)nk_stride_mask_u1x64_(stride_elements);
    __m512 sum_f32x16 = _mm512_setzero_ps();
    __m512 sumsq_f32x16 = _mm512_setzero_ps();
    nk_size_t idx_scalars = 0;
    nk_size_t total_scalars = count * stride_elements;
    for (; idx_scalars + 16 <= total_scalars; idx_scalars += 16) {
        __m128i data_e3m2x16 = _mm_maskz_loadu_epi8(stride_mask_m16, data + idx_scalars);
        __m512 data_f32x16 = nk_e3m2x16_to_f32x16_skylake_(data_e3m2x16);
        sum_f32x16 = _mm512_add_ps(sum_f32x16, data_f32x16);
        sumsq_f32x16 = _mm512_fmadd_ps(data_f32x16, data_f32x16, sumsq_f32x16);
    }
    nk_size_t remaining_bytes = total_scalars - idx_scalars;
    if (remaining_bytes > 0) {
        __mmask16 tail_mask = stride_mask_m16 & (__mmask16)_bzhi_u32(0xFFFF, (unsigned int)remaining_bytes);
        __m128i data_e3m2x16 = _mm_maskz_loadu_epi8(tail_mask, data + idx_scalars);
        __m512 data_f32x16 = nk_e3m2x16_to_f32x16_skylake_(data_e3m2x16);
        sum_f32x16 = _mm512_add_ps(sum_f32x16, data_f32x16);
        sumsq_f32x16 = _mm512_fmadd_ps(data_f32x16, data_f32x16, sumsq_f32x16);
    }
    *sum = nk_reduce_add_f32x16_skylake_(sum_f32x16);
    *sumsq = nk_reduce_add_f32x16_skylake_(sumsq_f32x16);
}

NK_PUBLIC void nk_reduce_moments_e3m2_skylake(                      //
    nk_e3m2_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *sum, nk_f32_t *sumsq) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_e3m2_t);
    int aligned = (stride_bytes % sizeof(nk_e3m2_t) == 0);
    if (count == 0) *sum = 0, *sumsq = 0;
    else if (!aligned) nk_reduce_moments_e3m2_serial(data, count, stride_bytes, sum, sumsq);
    else if (count > (nk_size_t)(NK_U16_MAX + 1) * 64) {
        nk_size_t left_count = count / 2;
        nk_f32_t left_sum, left_sumsq, right_sum, right_sumsq;
        nk_reduce_moments_e3m2_skylake(data, left_count, stride_bytes, &left_sum, &left_sumsq);
        nk_reduce_moments_e3m2_skylake(data + left_count * stride_elements, count - left_count, stride_bytes,
                                       &right_sum, &right_sumsq);
        *sum = left_sum + right_sum, *sumsq = left_sumsq + right_sumsq;
    }
    else if (stride_elements == 1) nk_reduce_moments_e3m2_skylake_contiguous_(data, count, sum, sumsq);
    else if (stride_elements >= 2 && stride_elements <= 16)
        nk_reduce_moments_e3m2_skylake_strided_(data, count, stride_elements, sum, sumsq);
    else nk_reduce_moments_e3m2_serial(data, count, stride_bytes, sum, sumsq);
}

NK_INTERNAL void nk_reduce_minmax_e5m2_skylake_contiguous_( //
    nk_e5m2_t const *data, nk_size_t count,                 //
    nk_e5m2_t *min_value, nk_size_t *min_index,             //
    nk_e5m2_t *max_value, nk_size_t *max_index) {
    __m512i nan_cmp_u8x64 = _mm512_set1_epi8((char)0xFF);
    __m512i first_i8x64 = _mm512_loadu_si512(data);
    __m512i first_cmp = nk_fp8x64_to_u8x64_comparable_skylake_(first_i8x64);
    nk_b512_vec_t min_vec, max_vec;
    min_vec.zmm = first_cmp;
    max_vec.zmm = first_cmp;
    __m512i min_iter_u8x64 = _mm512_setzero_si512();
    __m512i max_iter_u8x64 = _mm512_setzero_si512();
    __m512i iter_u8x64 = _mm512_set1_epi8(1);
    __m512i one_u8x64 = _mm512_set1_epi8(1);

    nk_size_t idx = 64;
    for (; idx + 64 <= count; idx += 64) {
        __m512i data_i8x64 = _mm512_loadu_si512(data + idx);
        __m512i data_cmp_u8x64 = nk_fp8x64_to_u8x64_comparable_skylake_(data_i8x64);
        __mmask64 min_mask_m64 = nk_min_mask_e5m2x64_skylake_(min_vec.zmm, data_cmp_u8x64, nan_cmp_u8x64);
        min_vec.zmm = _mm512_mask_mov_epi8(data_cmp_u8x64, min_mask_m64, min_vec.zmm);
        __mmask64 min_changed_mask = ~min_mask_m64;
        min_iter_u8x64 = _mm512_mask_mov_epi8(min_iter_u8x64, min_changed_mask, iter_u8x64);
        __mmask64 max_mask_m64 = nk_max_mask_e5m2x64_skylake_(max_vec.zmm, data_cmp_u8x64, nan_cmp_u8x64);
        max_vec.zmm = _mm512_mask_mov_epi8(data_cmp_u8x64, max_mask_m64, max_vec.zmm);
        __mmask64 max_changed_mask = ~max_mask_m64;
        max_iter_u8x64 = _mm512_mask_mov_epi8(max_iter_u8x64, max_changed_mask, iter_u8x64);
        iter_u8x64 = _mm512_add_epi8(iter_u8x64, one_u8x64);
    }

    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        __mmask64 tail_load = _bzhi_u64(0xFFFFFFFFFFFFFFFFull, (unsigned int)remaining);
        __m512i data_i8x64 = _mm512_mask_loadu_epi8(nan_cmp_u8x64, tail_load, data + idx);
        __m512i data_cmp_u8x64 = nk_fp8x64_to_u8x64_comparable_skylake_(data_i8x64);
        __mmask64 min_mask_m64 = nk_min_mask_e5m2x64_skylake_(min_vec.zmm, data_cmp_u8x64, nan_cmp_u8x64);
        __mmask64 min_changed_mask = tail_load & ~min_mask_m64;
        min_vec.zmm = _mm512_mask_mov_epi8(data_cmp_u8x64, min_mask_m64, min_vec.zmm);
        min_iter_u8x64 = _mm512_mask_mov_epi8(min_iter_u8x64, min_changed_mask, iter_u8x64);
        __mmask64 max_mask_m64 = nk_max_mask_e5m2x64_skylake_(max_vec.zmm, data_cmp_u8x64, nan_cmp_u8x64);
        __mmask64 max_changed_mask = tail_load & ~max_mask_m64;
        max_vec.zmm = _mm512_mask_mov_epi8(data_cmp_u8x64, max_mask_m64, max_vec.zmm);
        max_iter_u8x64 = _mm512_mask_mov_epi8(max_iter_u8x64, max_changed_mask, iter_u8x64);
    }

    nk_size_t min_lane = nk_argmin_u8x64_skylake_(min_vec.zmm);
    nk_size_t max_lane = nk_argmax_u8x64_skylake_(max_vec.zmm);
    nk_b512_vec_t iter_vec;
    iter_vec.zmm = min_iter_u8x64;
    *min_index = (nk_size_t)iter_vec.u8s[min_lane] * 64 + min_lane;
    iter_vec.zmm = max_iter_u8x64;
    *max_index = (nk_size_t)iter_vec.u8s[max_lane] * 64 + max_lane;
    min_vec.zmm = nk_u8x64_comparable_to_fp8x64_skylake_(min_vec.zmm);
    max_vec.zmm = nk_u8x64_comparable_to_fp8x64_skylake_(max_vec.zmm);
    *min_value = min_vec.e5m2s[min_lane];
    *max_value = max_vec.e5m2s[max_lane];
}

NK_PUBLIC void nk_reduce_minmax_e5m2_skylake(                       //
    nk_e5m2_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_e5m2_t *min_value, nk_size_t *min_index,                     //
    nk_e5m2_t *max_value, nk_size_t *max_index) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_e5m2_t);
    if (count == 0) { *min_value = 0x7B, *min_index = 0, *max_value = 0xFB, *max_index = 0; }
    else if (stride_elements == 1 && count > (nk_size_t)(NK_U8_MAX + 1) * 64) {
        nk_size_t left_count = count / 2;
        nk_e5m2_t left_min, right_min, left_max, right_max;
        nk_size_t left_min_index, right_min_index, left_max_index, right_max_index;
        nk_reduce_minmax_e5m2_skylake(data, left_count, stride_bytes, &left_min, &left_min_index, &left_max,
                                      &left_max_index);
        nk_reduce_minmax_e5m2_skylake(data + left_count, count - left_count, stride_bytes, &right_min, &right_min_index,
                                      &right_max, &right_max_index);
        if (nk_e5m2_compare_(right_min, left_min) < 0)
            *min_value = right_min, *min_index = left_count + right_min_index;
        else *min_value = left_min, *min_index = left_min_index;
        if (nk_e5m2_compare_(right_max, left_max) > 0)
            *max_value = right_max, *max_index = left_count + right_max_index;
        else *max_value = left_max, *max_index = left_max_index;
    }
    else if (stride_elements == 1)
        nk_reduce_minmax_e5m2_skylake_contiguous_(data, count, min_value, min_index, max_value, max_index);
    else nk_reduce_minmax_e5m2_serial(data, count, stride_bytes, min_value, min_index, max_value, max_index);
}

NK_INTERNAL void nk_reduce_minmax_e2m3_skylake_contiguous_( //
    nk_e2m3_t const *data, nk_size_t count,                 //
    nk_e2m3_t *min_value, nk_size_t *min_index,             //
    nk_e2m3_t *max_value, nk_size_t *max_index) {
    __m512i first_i8x64 = _mm512_loadu_si512(data);
    __m512i first_cmp = nk_fp6x64_to_u8x64_comparable_skylake_(first_i8x64);
    nk_b512_vec_t min_vec, max_vec;
    min_vec.zmm = first_cmp;
    max_vec.zmm = first_cmp;
    __m512i min_iter_u8x64 = _mm512_setzero_si512();
    __m512i max_iter_u8x64 = _mm512_setzero_si512();
    __m512i iter_u8x64 = _mm512_set1_epi8(1);
    __m512i one_u8x64 = _mm512_set1_epi8(1);

    nk_size_t idx = 64;
    for (; idx + 64 <= count; idx += 64) {
        __m512i data_i8x64 = _mm512_loadu_si512(data + idx);
        __m512i data_cmp_u8x64 = nk_fp6x64_to_u8x64_comparable_skylake_(data_i8x64);
        __mmask64 min_changed_mask = _mm512_cmp_epu8_mask(data_cmp_u8x64, min_vec.zmm, _MM_CMPINT_LT);
        min_vec.zmm = _mm512_min_epu8(min_vec.zmm, data_cmp_u8x64);
        min_iter_u8x64 = _mm512_mask_mov_epi8(min_iter_u8x64, min_changed_mask, iter_u8x64);
        __mmask64 max_changed_mask = _mm512_cmp_epu8_mask(data_cmp_u8x64, max_vec.zmm, _MM_CMPINT_NLE);
        max_vec.zmm = _mm512_max_epu8(max_vec.zmm, data_cmp_u8x64);
        max_iter_u8x64 = _mm512_mask_mov_epi8(max_iter_u8x64, max_changed_mask, iter_u8x64);
        iter_u8x64 = _mm512_add_epi8(iter_u8x64, one_u8x64);
    }

    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        __mmask64 tail_load = _bzhi_u64(0xFFFFFFFFFFFFFFFFull, (unsigned int)remaining);
        __m512i data_i8x64 = _mm512_mask_loadu_epi8(_mm512_set1_epi8(0x3F), tail_load, data + idx);
        __m512i data_cmp_u8x64 = nk_fp6x64_to_u8x64_comparable_skylake_(data_i8x64);
        __mmask64 min_changed_mask = _mm512_mask_cmp_epu8_mask(tail_load, data_cmp_u8x64, min_vec.zmm, _MM_CMPINT_LT);
        min_vec.zmm = _mm512_min_epu8(min_vec.zmm, data_cmp_u8x64);
        min_iter_u8x64 = _mm512_mask_mov_epi8(min_iter_u8x64, min_changed_mask, iter_u8x64);
        __mmask64 max_changed_mask = _mm512_mask_cmp_epu8_mask(tail_load, data_cmp_u8x64, max_vec.zmm, _MM_CMPINT_NLE);
        max_vec.zmm = _mm512_max_epu8(max_vec.zmm, data_cmp_u8x64);
        max_iter_u8x64 = _mm512_mask_mov_epi8(max_iter_u8x64, max_changed_mask, iter_u8x64);
    }

    nk_size_t min_lane = nk_argmin_u8x64_skylake_(min_vec.zmm);
    nk_size_t max_lane = nk_argmax_u8x64_skylake_(max_vec.zmm);
    nk_b512_vec_t iter_vec;
    iter_vec.zmm = min_iter_u8x64;
    *min_index = (nk_size_t)iter_vec.u8s[min_lane] * 64 + min_lane;
    iter_vec.zmm = max_iter_u8x64;
    *max_index = (nk_size_t)iter_vec.u8s[max_lane] * 64 + max_lane;
    min_vec.zmm = nk_u8x64_comparable_to_fp6x64_skylake_(min_vec.zmm);
    max_vec.zmm = nk_u8x64_comparable_to_fp6x64_skylake_(max_vec.zmm);
    *min_value = min_vec.e2m3s[min_lane];
    *max_value = max_vec.e2m3s[max_lane];
}

NK_PUBLIC void nk_reduce_minmax_e2m3_skylake(                       //
    nk_e2m3_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_e2m3_t *min_value, nk_size_t *min_index,                     //
    nk_e2m3_t *max_value, nk_size_t *max_index) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_e2m3_t);
    if (count == 0) { *min_value = 0x1F, *min_index = 0, *max_value = 0x3F, *max_index = 0; }
    else if (stride_elements == 1 && count > (nk_size_t)(NK_U8_MAX + 1) * 64) {
        nk_size_t left_count = count / 2;
        nk_e2m3_t left_min, right_min, left_max, right_max;
        nk_size_t left_min_index, right_min_index, left_max_index, right_max_index;
        nk_reduce_minmax_e2m3_skylake(data, left_count, stride_bytes, &left_min, &left_min_index, &left_max,
                                      &left_max_index);
        nk_reduce_minmax_e2m3_skylake(data + left_count, count - left_count, stride_bytes, &right_min, &right_min_index,
                                      &right_max, &right_max_index);
        if (nk_e2m3_compare_(right_min, left_min) < 0)
            *min_value = right_min, *min_index = left_count + right_min_index;
        else *min_value = left_min, *min_index = left_min_index;
        if (nk_e2m3_compare_(right_max, left_max) > 0)
            *max_value = right_max, *max_index = left_count + right_max_index;
        else *max_value = left_max, *max_index = left_max_index;
    }
    else if (stride_elements == 1)
        nk_reduce_minmax_e2m3_skylake_contiguous_(data, count, min_value, min_index, max_value, max_index);
    else nk_reduce_minmax_e2m3_serial(data, count, stride_bytes, min_value, min_index, max_value, max_index);
}

NK_INTERNAL void nk_reduce_minmax_e3m2_skylake_contiguous_( //
    nk_e3m2_t const *data, nk_size_t count,                 //
    nk_e3m2_t *min_value, nk_size_t *min_index,             //
    nk_e3m2_t *max_value, nk_size_t *max_index) {
    __m512i first_i8x64 = _mm512_loadu_si512(data);
    __m512i first_cmp = nk_fp6x64_to_u8x64_comparable_skylake_(first_i8x64);
    nk_b512_vec_t min_vec, max_vec;
    min_vec.zmm = first_cmp;
    max_vec.zmm = first_cmp;
    __m512i min_iter_u8x64 = _mm512_setzero_si512();
    __m512i max_iter_u8x64 = _mm512_setzero_si512();
    __m512i iter_u8x64 = _mm512_set1_epi8(1);
    __m512i one_u8x64 = _mm512_set1_epi8(1);

    nk_size_t idx = 64;
    for (; idx + 64 <= count; idx += 64) {
        __m512i data_i8x64 = _mm512_loadu_si512(data + idx);
        __m512i data_cmp_u8x64 = nk_fp6x64_to_u8x64_comparable_skylake_(data_i8x64);
        __mmask64 min_changed_mask = _mm512_cmp_epu8_mask(data_cmp_u8x64, min_vec.zmm, _MM_CMPINT_LT);
        min_vec.zmm = _mm512_min_epu8(min_vec.zmm, data_cmp_u8x64);
        min_iter_u8x64 = _mm512_mask_mov_epi8(min_iter_u8x64, min_changed_mask, iter_u8x64);
        __mmask64 max_changed_mask = _mm512_cmp_epu8_mask(data_cmp_u8x64, max_vec.zmm, _MM_CMPINT_NLE);
        max_vec.zmm = _mm512_max_epu8(max_vec.zmm, data_cmp_u8x64);
        max_iter_u8x64 = _mm512_mask_mov_epi8(max_iter_u8x64, max_changed_mask, iter_u8x64);
        iter_u8x64 = _mm512_add_epi8(iter_u8x64, one_u8x64);
    }

    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        __mmask64 tail_load = _bzhi_u64(0xFFFFFFFFFFFFFFFFull, (unsigned int)remaining);
        __m512i data_i8x64 = _mm512_mask_loadu_epi8(_mm512_set1_epi8(0x3F), tail_load, data + idx);
        __m512i data_cmp_u8x64 = nk_fp6x64_to_u8x64_comparable_skylake_(data_i8x64);
        __mmask64 min_changed_mask = _mm512_mask_cmp_epu8_mask(tail_load, data_cmp_u8x64, min_vec.zmm, _MM_CMPINT_LT);
        min_vec.zmm = _mm512_min_epu8(min_vec.zmm, data_cmp_u8x64);
        min_iter_u8x64 = _mm512_mask_mov_epi8(min_iter_u8x64, min_changed_mask, iter_u8x64);
        __mmask64 max_changed_mask = _mm512_mask_cmp_epu8_mask(tail_load, data_cmp_u8x64, max_vec.zmm, _MM_CMPINT_NLE);
        max_vec.zmm = _mm512_max_epu8(max_vec.zmm, data_cmp_u8x64);
        max_iter_u8x64 = _mm512_mask_mov_epi8(max_iter_u8x64, max_changed_mask, iter_u8x64);
    }

    nk_size_t min_lane = nk_argmin_u8x64_skylake_(min_vec.zmm);
    nk_size_t max_lane = nk_argmax_u8x64_skylake_(max_vec.zmm);
    nk_b512_vec_t iter_vec;
    iter_vec.zmm = min_iter_u8x64;
    *min_index = (nk_size_t)iter_vec.u8s[min_lane] * 64 + min_lane;
    iter_vec.zmm = max_iter_u8x64;
    *max_index = (nk_size_t)iter_vec.u8s[max_lane] * 64 + max_lane;
    min_vec.zmm = nk_u8x64_comparable_to_fp6x64_skylake_(min_vec.zmm);
    max_vec.zmm = nk_u8x64_comparable_to_fp6x64_skylake_(max_vec.zmm);
    *min_value = min_vec.e3m2s[min_lane];
    *max_value = max_vec.e3m2s[max_lane];
}

NK_PUBLIC void nk_reduce_minmax_e3m2_skylake(                       //
    nk_e3m2_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_e3m2_t *min_value, nk_size_t *min_index,                     //
    nk_e3m2_t *max_value, nk_size_t *max_index) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_e3m2_t);
    if (count == 0) { *min_value = 0x1F, *min_index = 0, *max_value = 0x3F, *max_index = 0; }
    else if (stride_elements == 1 && count > (nk_size_t)(NK_U8_MAX + 1) * 64) {
        nk_size_t left_count = count / 2;
        nk_e3m2_t left_min, right_min, left_max, right_max;
        nk_size_t left_min_index, right_min_index, left_max_index, right_max_index;
        nk_reduce_minmax_e3m2_skylake(data, left_count, stride_bytes, &left_min, &left_min_index, &left_max,
                                      &left_max_index);
        nk_reduce_minmax_e3m2_skylake(data + left_count, count - left_count, stride_bytes, &right_min, &right_min_index,
                                      &right_max, &right_max_index);
        if (nk_e3m2_compare_(right_min, left_min) < 0)
            *min_value = right_min, *min_index = left_count + right_min_index;
        else *min_value = left_min, *min_index = left_min_index;
        if (nk_e3m2_compare_(right_max, left_max) > 0)
            *max_value = right_max, *max_index = left_count + right_max_index;
        else *max_value = left_max, *max_index = left_max_index;
    }
    else if (stride_elements == 1)
        nk_reduce_minmax_e3m2_skylake_contiguous_(data, count, min_value, min_index, max_value, max_index);
    else nk_reduce_minmax_e3m2_serial(data, count, stride_bytes, min_value, min_index, max_value, max_index);
}

NK_INTERNAL void nk_reduce_moments_i4_skylake_contiguous_( //
    nk_i4x2_t const *data, nk_size_t count,                //
    nk_i64_t *sum, nk_i64_t *sumsq) {
    // Sum: XOR-bias nibbles to unsigned, vpsadbw, unbias at end.
    // Sumsq: squares are sign-independent; LUT maps nibble→square (max 225 fits u8), vpsadbw to u64.
    __m512i mask_0f_i8x64 = _mm512_set1_epi8(0x0F);
    __m512i eight_i8x64 = _mm512_set1_epi8(8);
    __m512i zero_i8x64 = _mm512_setzero_si512();
    // Squares LUT: sq_lut[n] = n² for n in [0,15], all fit in u8 (max 225)
    __m512i sq_lut_u8x64 = _mm512_set_epi8(                               //
        225, 196, 169, 144, 121, 100, 81, 64, 49, 36, 25, 16, 9, 4, 1, 0, //
        225, 196, 169, 144, 121, 100, 81, 64, 49, 36, 25, 16, 9, 4, 1, 0, //
        225, 196, 169, 144, 121, 100, 81, 64, 49, 36, 25, 16, 9, 4, 1, 0, //
        225, 196, 169, 144, 121, 100, 81, 64, 49, 36, 25, 16, 9, 4, 1, 0);
    __m512i sum_u64x8 = _mm512_setzero_si512();
    __m512i sumsq_u64x8 = _mm512_setzero_si512();
    nk_size_t count_nibbles = count;
    nk_size_t count_bytes = nk_size_divide_round_up_(count, 2);
    unsigned char const *ptr = (unsigned char const *)data;
    while (count_bytes > 0) {
        __m512i raw_i8x64;
        if (count_bytes < 64) {
            __mmask64 tail_mask = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFFULL, (unsigned int)count_bytes);
            raw_i8x64 = _mm512_maskz_loadu_epi8(tail_mask, ptr);
            count_bytes = 0;
        }
        else {
            raw_i8x64 = _mm512_loadu_si512(ptr);
            ptr += 64, count_bytes -= 64;
        }
        // Extract nibbles as unsigned [0,15]
        __m512i low_u4x64 = _mm512_and_si512(raw_i8x64, mask_0f_i8x64);
        __m512i high_u4x64 = _mm512_and_si512(_mm512_srli_epi16(raw_i8x64, 4), mask_0f_i8x64);
        // Sum: XOR-bias nibbles to unsigned [0,15], add lo+hi per byte, vpsadbw
        __m512i lo_biased_u4x64 = _mm512_xor_si512(low_u4x64, eight_i8x64);
        __m512i hi_biased_u4x64 = _mm512_xor_si512(high_u4x64, eight_i8x64);
        __m512i pair_sum = _mm512_add_epi8(lo_biased_u4x64, hi_biased_u4x64);
        sum_u64x8 = _mm512_add_epi64(sum_u64x8, _mm512_sad_epu8(pair_sum, zero_i8x64));
        // Sumsq: squares are sign-independent, use LUT on unsigned nibbles
        __m512i lo_sq_u8x64 = _mm512_shuffle_epi8(sq_lut_u8x64, low_u4x64);
        __m512i hi_sq_u8x64 = _mm512_shuffle_epi8(sq_lut_u8x64, high_u4x64);
        sumsq_u64x8 = _mm512_add_epi64(sumsq_u64x8, _mm512_sad_epu8(lo_sq_u8x64, zero_i8x64));
        sumsq_u64x8 = _mm512_add_epi64(sumsq_u64x8, _mm512_sad_epu8(hi_sq_u8x64, zero_i8x64));
    }
    // Unbias sum: each nibble was biased by +8, total bias = 8 * nibbles_processed
    nk_size_t nibbles_processed = nk_size_divide_round_up_(count_nibbles, 2) * 2;
    nk_i64_t s = (nk_i64_t)nk_reduce_add_u64x8_skylake_(sum_u64x8) - (nk_i64_t)8 * (nk_i64_t)nibbles_processed;
    // Handle odd count: the last byte's high nibble was included but shouldn't be
    if (count_nibbles & 1) {
        nk_u8_t last_byte = ((unsigned char const *)data)[count_nibbles / 2];
        nk_u8_t high_nib = (last_byte >> 4) & 0x0F;
        nk_i64_t signed_high = (nk_i64_t)((nk_i8_t)((high_nib ^ 8) - 8));
        s -= signed_high;
    }
    nk_i64_t sq = (nk_i64_t)nk_reduce_add_u64x8_skylake_(sumsq_u64x8);
    if (count_nibbles & 1) {
        nk_u8_t last_byte = ((unsigned char const *)data)[count_nibbles / 2];
        nk_u8_t high_nib = (last_byte >> 4) & 0x0F;
        nk_i64_t signed_high = (nk_i64_t)((nk_i8_t)((high_nib ^ 8) - 8));
        sq -= signed_high * signed_high;
    }
    *sum = s;
    *sumsq = sq;
}

NK_PUBLIC void nk_reduce_moments_i4_skylake(                        //
    nk_i4x2_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *sum, nk_i64_t *sumsq) {
    if (count == 0) *sum = 0, *sumsq = 0;
    else if (stride_bytes == 1) nk_reduce_moments_i4_skylake_contiguous_(data, count, sum, sumsq);
    else nk_reduce_moments_i4_serial(data, count, stride_bytes, sum, sumsq);
}

NK_INTERNAL void nk_reduce_moments_u4_skylake_contiguous_( //
    nk_u4x2_t const *data, nk_size_t count,                //
    nk_u64_t *sum, nk_u64_t *sumsq) {
    // Sum: VPSADBW on extracted nibbles. Sumsq: LUT maps nibble→square (max 225 fits u8), vpsadbw to u64.
    __m512i mask_0f_i8x64 = _mm512_set1_epi8(0x0F);
    __m512i zero_i8x64 = _mm512_setzero_si512();
    // Squares LUT: sq_lut[n] = n² for n in [0,15], all fit in u8 (max 225)
    __m512i sq_lut_u8x64 = _mm512_set_epi8(                               //
        225, 196, 169, 144, 121, 100, 81, 64, 49, 36, 25, 16, 9, 4, 1, 0, //
        225, 196, 169, 144, 121, 100, 81, 64, 49, 36, 25, 16, 9, 4, 1, 0, //
        225, 196, 169, 144, 121, 100, 81, 64, 49, 36, 25, 16, 9, 4, 1, 0, //
        225, 196, 169, 144, 121, 100, 81, 64, 49, 36, 25, 16, 9, 4, 1, 0);
    __m512i sum_u64x8 = _mm512_setzero_si512();
    __m512i sumsq_u64x8 = _mm512_setzero_si512();
    nk_size_t count_nibbles = count;
    nk_size_t count_bytes = nk_size_divide_round_up_(count, 2);
    unsigned char const *ptr = (unsigned char const *)data;
    while (count_bytes > 0) {
        __m512i raw_i8x64;
        if (count_bytes < 64) {
            __mmask64 tail_mask = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFFULL, (unsigned int)count_bytes);
            raw_i8x64 = _mm512_maskz_loadu_epi8(tail_mask, ptr);
            count_bytes = 0;
        }
        else {
            raw_i8x64 = _mm512_loadu_si512(ptr);
            ptr += 64, count_bytes -= 64;
        }
        __m512i low_u4x64 = _mm512_and_si512(raw_i8x64, mask_0f_i8x64);
        __m512i high_u4x64 = _mm512_and_si512(_mm512_srli_epi16(raw_i8x64, 4), mask_0f_i8x64);
        __m512i pair_sum = _mm512_add_epi8(low_u4x64, high_u4x64);
        sum_u64x8 = _mm512_add_epi64(sum_u64x8, _mm512_sad_epu8(pair_sum, zero_i8x64));
        // Sumsq: LUT maps nibble→square, vpsadbw accumulates into u64
        __m512i lo_sq_u8x64 = _mm512_shuffle_epi8(sq_lut_u8x64, low_u4x64);
        __m512i hi_sq_u8x64 = _mm512_shuffle_epi8(sq_lut_u8x64, high_u4x64);
        sumsq_u64x8 = _mm512_add_epi64(sumsq_u64x8, _mm512_sad_epu8(lo_sq_u8x64, zero_i8x64));
        sumsq_u64x8 = _mm512_add_epi64(sumsq_u64x8, _mm512_sad_epu8(hi_sq_u8x64, zero_i8x64));
    }
    nk_u64_t s = _mm512_reduce_add_epi64(sum_u64x8);
    if (count_nibbles & 1) {
        nk_u8_t last_byte = ((unsigned char const *)data)[count_nibbles / 2];
        s -= (last_byte >> 4) & 0x0F;
    }
    nk_u64_t sq = nk_reduce_add_u64x8_skylake_(sumsq_u64x8);
    if (count_nibbles & 1) {
        nk_u8_t last_byte = ((unsigned char const *)data)[count_nibbles / 2];
        nk_u8_t high_nib = (last_byte >> 4) & 0x0F;
        sq -= (nk_u64_t)high_nib * high_nib;
    }
    *sum = s;
    *sumsq = sq;
}

NK_PUBLIC void nk_reduce_moments_u4_skylake(                        //
    nk_u4x2_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *sum, nk_u64_t *sumsq) {
    if (count == 0) *sum = 0, *sumsq = 0;
    else if (stride_bytes == 1) nk_reduce_moments_u4_skylake_contiguous_(data, count, sum, sumsq);
    else nk_reduce_moments_u4_serial(data, count, stride_bytes, sum, sumsq);
}

NK_INTERNAL void nk_reduce_moments_u1_skylake_contiguous_( //
    nk_u1x8_t const *data, nk_size_t count,                //
    nk_u64_t *sum, nk_u64_t *sumsq) {
    // Sum = popcount via 4-bit LUT (same as nk_reduce_add_u1_skylake). Sumsq = sum for bits.
    __m512i lut_i8x64 = _mm512_set_epi8(                //
        4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0, //
        4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0, //
        4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0, //
        4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0);
    __m512i mask_0f_i8x64 = _mm512_set1_epi8(0x0F);
    __m512i zero_i8x64 = _mm512_setzero_si512();
    __m512i sum_u64x8 = _mm512_setzero_si512();
    nk_size_t count_bits = count;
    nk_size_t count_bytes = nk_size_divide_round_up_(count, 8);
    unsigned char const *ptr = (unsigned char const *)data;
    while (count_bytes > 0) {
        __m512i raw_i8x64;
        if (count_bytes < 64) {
            __mmask64 tail_mask = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFFULL, (unsigned int)count_bytes);
            raw_i8x64 = _mm512_maskz_loadu_epi8(tail_mask, ptr);
            count_bytes = 0;
        }
        else {
            raw_i8x64 = _mm512_loadu_si512(ptr);
            ptr += 64, count_bytes -= 64;
        }
        __m512i low_nibble_u8x64 = _mm512_and_si512(raw_i8x64, mask_0f_i8x64);
        __m512i high_nibble_u8x64 = _mm512_and_si512(_mm512_srli_epi16(raw_i8x64, 4), mask_0f_i8x64);
        __m512i popcnt_u8x64 = _mm512_add_epi8(_mm512_shuffle_epi8(lut_i8x64, low_nibble_u8x64),
                                               _mm512_shuffle_epi8(lut_i8x64, high_nibble_u8x64));
        sum_u64x8 = _mm512_add_epi64(sum_u64x8, _mm512_sad_epu8(popcnt_u8x64, zero_i8x64));
    }
    nk_u64_t s = _mm512_reduce_add_epi64(sum_u64x8);
    if (count_bits % 8) {
        nk_u8_t last_byte = ((unsigned char const *)data)[nk_size_divide_round_up_(count_bits, 8) - 1];
        nk_u8_t mask = (nk_u8_t)((1u << (count_bits % 8)) - 1u);
        s -= nk_u64_popcount_((nk_u64_t)(last_byte & ~mask));
    }
    *sum = s;
    *sumsq = s;
}

NK_PUBLIC void nk_reduce_moments_u1_skylake(                        //
    nk_u1x8_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *sum, nk_u64_t *sumsq) {
    if (count == 0) *sum = 0, *sumsq = 0;
    else if (stride_bytes == 1) nk_reduce_moments_u1_skylake_contiguous_(data, count, sum, sumsq);
    else nk_reduce_moments_u1_serial(data, count, stride_bytes, sum, sumsq);
}

NK_INTERNAL void nk_reduce_moments_bf16_skylake_contiguous_( //
    nk_bf16_t const *data, nk_size_t count,                  //
    nk_f32_t *sum, nk_f32_t *sumsq) {
    __m512 sum_f32x16 = _mm512_setzero_ps();
    __m512 sumsq_f32x16 = _mm512_setzero_ps();
    nk_size_t idx = 0;
    for (; idx + 32 <= count; idx += 32) {
        __m512 low_f32x16 = _mm512_castsi512_ps(
            _mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i const *)(data + idx))), 16));
        __m512 high_f32x16 = _mm512_castsi512_ps(
            _mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i const *)(data + idx + 16))), 16));
        sum_f32x16 = _mm512_add_ps(sum_f32x16, low_f32x16);
        sum_f32x16 = _mm512_add_ps(sum_f32x16, high_f32x16);
        sumsq_f32x16 = _mm512_fmadd_ps(low_f32x16, low_f32x16, sumsq_f32x16);
        sumsq_f32x16 = _mm512_fmadd_ps(high_f32x16, high_f32x16, sumsq_f32x16);
    }
    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        __mmask16 lo_mask = (__mmask16)_bzhi_u32(0xFFFF, (unsigned int)(remaining > 16 ? 16 : remaining));
        __m512 low_f32x16 = _mm512_castsi512_ps(
            _mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_maskz_loadu_epi16(lo_mask, data + idx)), 16));
        sum_f32x16 = _mm512_add_ps(sum_f32x16, low_f32x16);
        sumsq_f32x16 = _mm512_fmadd_ps(low_f32x16, low_f32x16, sumsq_f32x16);
        if (remaining > 16) {
            __mmask16 hi_mask = (__mmask16)_bzhi_u32(0xFFFF, (unsigned int)(remaining - 16));
            __m512 high_f32x16 = _mm512_castsi512_ps(
                _mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_maskz_loadu_epi16(hi_mask, data + idx + 16)), 16));
            sum_f32x16 = _mm512_add_ps(sum_f32x16, high_f32x16);
            sumsq_f32x16 = _mm512_fmadd_ps(high_f32x16, high_f32x16, sumsq_f32x16);
        }
    }
    *sum = nk_reduce_add_f32x16_skylake_(sum_f32x16);
    *sumsq = nk_reduce_add_f32x16_skylake_(sumsq_f32x16);
}

NK_PUBLIC void nk_reduce_moments_bf16_skylake(                      //
    nk_bf16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *sum, nk_f32_t *sumsq) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_bf16_t);
    int aligned = (stride_bytes % sizeof(nk_bf16_t) == 0);
    if (count == 0) *sum = 0, *sumsq = 0;
    else if (!aligned) nk_reduce_moments_bf16_serial(data, count, stride_bytes, sum, sumsq);
    else if (count > (nk_size_t)(NK_U16_MAX + 1) * 32) {
        nk_size_t left_count = count / 2;
        nk_f32_t left_sum, left_sumsq, right_sum, right_sumsq;
        nk_reduce_moments_bf16_skylake(data, left_count, stride_bytes, &left_sum, &left_sumsq);
        nk_reduce_moments_bf16_skylake(data + left_count * stride_elements, count - left_count, stride_bytes,
                                       &right_sum, &right_sumsq);
        *sum = left_sum + right_sum;
        *sumsq = left_sumsq + right_sumsq;
    }
    else if (stride_elements == 1) nk_reduce_moments_bf16_skylake_contiguous_(data, count, sum, sumsq);
    else nk_reduce_moments_bf16_serial(data, count, stride_bytes, sum, sumsq);
}

NK_INTERNAL __m512i nk_bf16x32_to_comparable_i16x32_skylake_(__m512i raw_u16x32) {
    __m512i sign = _mm512_srai_epi16(raw_u16x32, 15);
    __m512i flip = _mm512_srli_epi16(sign, 1);
    return _mm512_xor_si512(raw_u16x32, flip);
}

NK_INTERNAL void nk_reduce_minmax_bf16_skylake_contiguous_( //
    nk_bf16_t const *data, nk_size_t count,                 //
    nk_bf16_t *min_value, nk_size_t *min_index,             //
    nk_bf16_t *max_value, nk_size_t *max_index) {
    __m512i abs_mask_u16x32 = _mm512_set1_epi16(0x7FFF);
    __m512i nan_threshold_u16x32 = _mm512_set1_epi16((short)0x7F80);
    __m512i min_cmp_i16x32 = _mm512_set1_epi16((short)0x7FFF);
    __m512i max_cmp_i16x32 = _mm512_set1_epi16((short)0x8000);
    __m512i min_iter_u16x32 = _mm512_setzero_si512();
    __m512i max_iter_u16x32 = _mm512_setzero_si512();
    __m512i iter_u16x32 = _mm512_setzero_si512();
    __m512i one_u16x32 = _mm512_set1_epi16(1);

    nk_size_t idx = 0;
    for (; idx + 32 <= count; idx += 32) {
        __m512i raw_u16x32 = _mm512_loadu_si512(data + idx);
        __m512i data_cmp_i16x32 = nk_bf16x32_to_comparable_i16x32_skylake_(raw_u16x32);
        __m512i abs_u16x32 = _mm512_and_si512(raw_u16x32, abs_mask_u16x32);
        __mmask32 not_nan_m32 = _mm512_cmp_epu16_mask(abs_u16x32, nan_threshold_u16x32, _MM_CMPINT_LE);
        __mmask32 min_changed_mask = _mm512_mask_cmp_epi16_mask(not_nan_m32, data_cmp_i16x32, min_cmp_i16x32,
                                                                _MM_CMPINT_LT);
        __mmask32 max_changed_mask = _mm512_mask_cmp_epi16_mask(not_nan_m32, data_cmp_i16x32, max_cmp_i16x32,
                                                                _MM_CMPINT_NLE);
        min_cmp_i16x32 = _mm512_mask_mov_epi16(min_cmp_i16x32, min_changed_mask, data_cmp_i16x32);
        max_cmp_i16x32 = _mm512_mask_mov_epi16(max_cmp_i16x32, max_changed_mask, data_cmp_i16x32);
        min_iter_u16x32 = _mm512_mask_mov_epi16(min_iter_u16x32, min_changed_mask, iter_u16x32);
        max_iter_u16x32 = _mm512_mask_mov_epi16(max_iter_u16x32, max_changed_mask, iter_u16x32);
        iter_u16x32 = _mm512_add_epi16(iter_u16x32, one_u16x32);
    }

    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        __mmask32 tail_load_m32 = (__mmask32)_bzhi_u32(0xFFFFFFFF, (unsigned int)remaining);
        __m512i raw_u16x32 = _mm512_maskz_loadu_epi16(tail_load_m32, data + idx);
        __m512i data_cmp_i16x32 = nk_bf16x32_to_comparable_i16x32_skylake_(raw_u16x32);
        __m512i abs_u16x32 = _mm512_and_si512(raw_u16x32, abs_mask_u16x32);
        __mmask32 not_nan_m32 = _mm512_cmp_epu16_mask(abs_u16x32, nan_threshold_u16x32, _MM_CMPINT_LE);
        __mmask32 valid_m32 = tail_load_m32 & not_nan_m32;
        __mmask32 min_changed_mask = _mm512_mask_cmp_epi16_mask(valid_m32, data_cmp_i16x32, min_cmp_i16x32,
                                                                _MM_CMPINT_LT);
        __mmask32 max_changed_mask = _mm512_mask_cmp_epi16_mask(valid_m32, data_cmp_i16x32, max_cmp_i16x32,
                                                                _MM_CMPINT_NLE);
        min_cmp_i16x32 = _mm512_mask_mov_epi16(min_cmp_i16x32, min_changed_mask, data_cmp_i16x32);
        max_cmp_i16x32 = _mm512_mask_mov_epi16(max_cmp_i16x32, max_changed_mask, data_cmp_i16x32);
        min_iter_u16x32 = _mm512_mask_mov_epi16(min_iter_u16x32, min_changed_mask, iter_u16x32);
        max_iter_u16x32 = _mm512_mask_mov_epi16(max_iter_u16x32, max_changed_mask, iter_u16x32);
    }

    nk_i16_t min_scalar_cmp = nk_reduce_min_i16x32_skylake_(min_cmp_i16x32);
    nk_i16_t max_scalar_cmp = nk_reduce_max_i16x32_skylake_(max_cmp_i16x32);
    __mmask32 min_equality_mask = _mm512_cmpeq_epi16_mask(min_cmp_i16x32, _mm512_set1_epi16(min_scalar_cmp));
    __mmask32 max_equality_mask = _mm512_cmpeq_epi16_mask(max_cmp_i16x32, _mm512_set1_epi16(max_scalar_cmp));
    unsigned int min_lane = _tzcnt_u32(min_equality_mask);
    unsigned int max_lane = _tzcnt_u32(max_equality_mask);
    nk_b512_vec_t iter_vec;
    iter_vec.zmm = min_iter_u16x32;
    *min_index = (nk_size_t)iter_vec.u16s[min_lane] * 32 + min_lane;
    iter_vec.zmm = max_iter_u16x32;
    *max_index = (nk_size_t)iter_vec.u16s[max_lane] * 32 + max_lane;
    nk_i16_t min_sign = min_scalar_cmp >> 15;
    *min_value = (nk_bf16_t)((nk_u16_t)min_scalar_cmp ^ ((nk_u16_t)min_sign >> 1));
    nk_i16_t max_sign = max_scalar_cmp >> 15;
    *max_value = (nk_bf16_t)((nk_u16_t)max_scalar_cmp ^ ((nk_u16_t)max_sign >> 1));
}

NK_PUBLIC void nk_reduce_minmax_bf16_skylake(                       //
    nk_bf16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_bf16_t *min_value, nk_size_t *min_index,                     //
    nk_bf16_t *max_value, nk_size_t *max_index) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_bf16_t);
    int aligned = (stride_bytes % sizeof(nk_bf16_t) == 0);
    if (count == 0) *min_value = NK_BF16_MAX, *min_index = 0, *max_value = NK_BF16_MIN, *max_index = 0;
    else if (!aligned)
        nk_reduce_minmax_bf16_serial(data, count, stride_bytes, min_value, min_index, max_value, max_index);
    else if (stride_elements == 1 && count > (nk_size_t)(NK_U16_MAX + 1) * 32) {
        nk_size_t left_count = count / 2;
        nk_bf16_t left_min, right_min, left_max, right_max;
        nk_size_t left_min_index, right_min_index, left_max_index, right_max_index;
        nk_reduce_minmax_bf16_skylake(data, left_count, stride_bytes, &left_min, &left_min_index, &left_max,
                                      &left_max_index);
        nk_reduce_minmax_bf16_skylake(data + left_count, count - left_count, stride_bytes, &right_min, &right_min_index,
                                      &right_max, &right_max_index);
        if (nk_bf16_compare_(right_min, left_min) < 0)
            *min_value = right_min, *min_index = left_count + right_min_index;
        else *min_value = left_min, *min_index = left_min_index;
        if (nk_bf16_compare_(right_max, left_max) > 0)
            *max_value = right_max, *max_index = left_count + right_max_index;
        else *max_value = left_max, *max_index = left_max_index;
    }
    else if (stride_elements == 1)
        nk_reduce_minmax_bf16_skylake_contiguous_(data, count, min_value, min_index, max_value, max_index);
    else nk_reduce_minmax_bf16_serial(data, count, stride_bytes, min_value, min_index, max_value, max_index);
}

NK_INTERNAL void nk_reduce_moments_f16_skylake_contiguous_( //
    nk_f16_t const *data, nk_size_t count,                  //
    nk_f32_t *sum, nk_f32_t *sumsq) {
    __m512 sum_f32x16 = _mm512_setzero_ps();
    __m512 sumsq_f32x16 = _mm512_setzero_ps();
    nk_size_t idx = 0;
    for (; idx + 32 <= count; idx += 32) {
        __m512 low_f32x16 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i const *)(data + idx)));
        __m512 high_f32x16 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i const *)(data + idx + 16)));
        sum_f32x16 = _mm512_add_ps(sum_f32x16, low_f32x16);
        sum_f32x16 = _mm512_add_ps(sum_f32x16, high_f32x16);
        sumsq_f32x16 = _mm512_fmadd_ps(low_f32x16, low_f32x16, sumsq_f32x16);
        sumsq_f32x16 = _mm512_fmadd_ps(high_f32x16, high_f32x16, sumsq_f32x16);
    }
    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        __mmask16 lo_mask_m16 = (__mmask16)_bzhi_u32(0xFFFF, (unsigned int)(remaining > 16 ? 16 : remaining));
        __m512 low_f32x16 = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(lo_mask_m16, data + idx));
        sum_f32x16 = _mm512_add_ps(sum_f32x16, low_f32x16);
        sumsq_f32x16 = _mm512_fmadd_ps(low_f32x16, low_f32x16, sumsq_f32x16);
        if (remaining > 16) {
            __mmask16 hi_mask_m16 = (__mmask16)_bzhi_u32(0xFFFF, (unsigned int)(remaining - 16));
            __m512 high_f32x16 = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(hi_mask_m16, data + idx + 16));
            sum_f32x16 = _mm512_add_ps(sum_f32x16, high_f32x16);
            sumsq_f32x16 = _mm512_fmadd_ps(high_f32x16, high_f32x16, sumsq_f32x16);
        }
    }
    *sum = nk_reduce_add_f32x16_skylake_(sum_f32x16);
    *sumsq = nk_reduce_add_f32x16_skylake_(sumsq_f32x16);
}

NK_PUBLIC void nk_reduce_moments_f16_skylake(                      //
    nk_f16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *sum, nk_f32_t *sumsq) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_f16_t);
    int aligned = (stride_bytes % sizeof(nk_f16_t) == 0);
    if (count == 0) *sum = 0, *sumsq = 0;
    else if (!aligned) nk_reduce_moments_f16_serial(data, count, stride_bytes, sum, sumsq);
    else if (count > (nk_size_t)(NK_U16_MAX + 1) * 32) {
        nk_size_t left_count = count / 2;
        nk_f32_t left_sum, left_sumsq, right_sum, right_sumsq;
        nk_reduce_moments_f16_skylake(data, left_count, stride_bytes, &left_sum, &left_sumsq);
        nk_reduce_moments_f16_skylake(data + left_count * stride_elements, count - left_count, stride_bytes, &right_sum,
                                      &right_sumsq);
        *sum = left_sum + right_sum;
        *sumsq = left_sumsq + right_sumsq;
    }
    else if (stride_elements == 1) nk_reduce_moments_f16_skylake_contiguous_(data, count, sum, sumsq);
    else nk_reduce_moments_f16_serial(data, count, stride_bytes, sum, sumsq);
}

NK_INTERNAL __m512i nk_f16x32_to_comparable_i16x32_skylake_(__m512i raw_u16x32) {
    __m512i sign = _mm512_srai_epi16(raw_u16x32, 15);
    __m512i flip = _mm512_srli_epi16(sign, 1);
    return _mm512_xor_si512(raw_u16x32, flip);
}

NK_INTERNAL void nk_reduce_minmax_f16_skylake_contiguous_( //
    nk_f16_t const *data, nk_size_t count,                 //
    nk_f16_t *min_value, nk_size_t *min_index,             //
    nk_f16_t *max_value, nk_size_t *max_index) {
    __m512i abs_mask_u16x32 = _mm512_set1_epi16(0x7FFF);
    __m512i nan_threshold_u16x32 = _mm512_set1_epi16((short)0x7C00);
    __m512i min_cmp_i16x32 = _mm512_set1_epi16((short)0x7FFF);
    __m512i max_cmp_i16x32 = _mm512_set1_epi16((short)0x8000);
    __m512i min_iter_u16x32 = _mm512_setzero_si512();
    __m512i max_iter_u16x32 = _mm512_setzero_si512();
    __m512i iter_u16x32 = _mm512_setzero_si512();
    __m512i one_u16x32 = _mm512_set1_epi16(1);

    nk_size_t idx = 0;
    for (; idx + 32 <= count; idx += 32) {
        __m512i raw_u16x32 = _mm512_loadu_si512(data + idx);
        __m512i data_cmp_i16x32 = nk_f16x32_to_comparable_i16x32_skylake_(raw_u16x32);
        __m512i abs_u16x32 = _mm512_and_si512(raw_u16x32, abs_mask_u16x32);
        __mmask32 not_nan_m32 = _mm512_cmp_epu16_mask(abs_u16x32, nan_threshold_u16x32, _MM_CMPINT_LE);
        __mmask32 min_changed_mask = _mm512_mask_cmp_epi16_mask(not_nan_m32, data_cmp_i16x32, min_cmp_i16x32,
                                                                _MM_CMPINT_LT);
        __mmask32 max_changed_mask = _mm512_mask_cmp_epi16_mask(not_nan_m32, data_cmp_i16x32, max_cmp_i16x32,
                                                                _MM_CMPINT_NLE);
        min_cmp_i16x32 = _mm512_mask_mov_epi16(min_cmp_i16x32, min_changed_mask, data_cmp_i16x32);
        max_cmp_i16x32 = _mm512_mask_mov_epi16(max_cmp_i16x32, max_changed_mask, data_cmp_i16x32);
        min_iter_u16x32 = _mm512_mask_mov_epi16(min_iter_u16x32, min_changed_mask, iter_u16x32);
        max_iter_u16x32 = _mm512_mask_mov_epi16(max_iter_u16x32, max_changed_mask, iter_u16x32);
        iter_u16x32 = _mm512_add_epi16(iter_u16x32, one_u16x32);
    }

    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        __mmask32 tail_load_m32 = (__mmask32)_bzhi_u32(0xFFFFFFFF, (unsigned int)remaining);
        __m512i raw_u16x32 = _mm512_maskz_loadu_epi16(tail_load_m32, data + idx);
        __m512i data_cmp_i16x32 = nk_f16x32_to_comparable_i16x32_skylake_(raw_u16x32);
        __m512i abs_u16x32 = _mm512_and_si512(raw_u16x32, abs_mask_u16x32);
        __mmask32 not_nan_m32 = _mm512_cmp_epu16_mask(abs_u16x32, nan_threshold_u16x32, _MM_CMPINT_LE);
        __mmask32 valid_m32 = tail_load_m32 & not_nan_m32;
        __mmask32 min_changed_mask = _mm512_mask_cmp_epi16_mask(valid_m32, data_cmp_i16x32, min_cmp_i16x32,
                                                                _MM_CMPINT_LT);
        __mmask32 max_changed_mask = _mm512_mask_cmp_epi16_mask(valid_m32, data_cmp_i16x32, max_cmp_i16x32,
                                                                _MM_CMPINT_NLE);
        min_cmp_i16x32 = _mm512_mask_mov_epi16(min_cmp_i16x32, min_changed_mask, data_cmp_i16x32);
        max_cmp_i16x32 = _mm512_mask_mov_epi16(max_cmp_i16x32, max_changed_mask, data_cmp_i16x32);
        min_iter_u16x32 = _mm512_mask_mov_epi16(min_iter_u16x32, min_changed_mask, iter_u16x32);
        max_iter_u16x32 = _mm512_mask_mov_epi16(max_iter_u16x32, max_changed_mask, iter_u16x32);
    }

    nk_i16_t min_scalar_cmp = nk_reduce_min_i16x32_skylake_(min_cmp_i16x32);
    nk_i16_t max_scalar_cmp = nk_reduce_max_i16x32_skylake_(max_cmp_i16x32);
    __mmask32 min_equality_mask = _mm512_cmpeq_epi16_mask(min_cmp_i16x32, _mm512_set1_epi16(min_scalar_cmp));
    __mmask32 max_equality_mask = _mm512_cmpeq_epi16_mask(max_cmp_i16x32, _mm512_set1_epi16(max_scalar_cmp));
    unsigned int min_lane = _tzcnt_u32(min_equality_mask);
    unsigned int max_lane = _tzcnt_u32(max_equality_mask);
    nk_b512_vec_t iter_vec;
    iter_vec.zmm = min_iter_u16x32;
    *min_index = (nk_size_t)iter_vec.u16s[min_lane] * 32 + min_lane;
    iter_vec.zmm = max_iter_u16x32;
    *max_index = (nk_size_t)iter_vec.u16s[max_lane] * 32 + max_lane;
    nk_i16_t min_sign = min_scalar_cmp >> 15;
    *min_value = (nk_f16_t)((nk_u16_t)min_scalar_cmp ^ ((nk_u16_t)min_sign >> 1));
    nk_i16_t max_sign = max_scalar_cmp >> 15;
    *max_value = (nk_f16_t)((nk_u16_t)max_scalar_cmp ^ ((nk_u16_t)max_sign >> 1));
}

NK_PUBLIC void nk_reduce_minmax_f16_skylake(                       //
    nk_f16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f16_t *min_value, nk_size_t *min_index,                     //
    nk_f16_t *max_value, nk_size_t *max_index) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_f16_t);
    int aligned = (stride_bytes % sizeof(nk_f16_t) == 0);
    if (count == 0) *min_value = NK_F16_MAX, *min_index = 0, *max_value = NK_F16_MIN, *max_index = 0;
    else if (!aligned)
        nk_reduce_minmax_f16_serial(data, count, stride_bytes, min_value, min_index, max_value, max_index);
    else if (stride_elements == 1 && count > (nk_size_t)(NK_U16_MAX + 1) * 32) {
        nk_size_t left_count = count / 2;
        nk_f16_t left_min, right_min, left_max, right_max;
        nk_size_t left_min_index, right_min_index, left_max_index, right_max_index;
        nk_reduce_minmax_f16_skylake(data, left_count, stride_bytes, &left_min, &left_min_index, &left_max,
                                     &left_max_index);
        nk_reduce_minmax_f16_skylake(data + left_count, count - left_count, stride_bytes, &right_min, &right_min_index,
                                     &right_max, &right_max_index);
        if (nk_f16_compare_(right_min, left_min) < 0) *min_value = right_min, *min_index = left_count + right_min_index;
        else *min_value = left_min, *min_index = left_min_index;
        if (nk_f16_compare_(right_max, left_max) > 0) *max_value = right_max, *max_index = left_count + right_max_index;
        else *max_value = left_max, *max_index = left_max_index;
    }
    else if (stride_elements == 1)
        nk_reduce_minmax_f16_skylake_contiguous_(data, count, min_value, min_index, max_value, max_index);
    else nk_reduce_minmax_f16_serial(data, count, stride_bytes, min_value, min_index, max_value, max_index);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_SKYLAKE
#endif // NK_TARGET_X86_
#endif // NK_REDUCE_SKYLAKE_NEW_H

/**
 *  @brief AVX2 implementations for the redesigned reduction API (moments + minmax).
 *  @file include/numkong/reduce/haswell_new.h
 *  @author Ash Vardanian
 *  @date February 12, 2026
 *
 *  @sa include/numkong/reduce.h
 */
#ifndef NK_REDUCE_HASWELL_NEW_H
#define NK_REDUCE_HASWELL_NEW_H

#if NK_TARGET_X86_
#if NK_TARGET_HASWELL

#include "numkong/types.h"
#include "numkong/reduce/serial_new.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,f16c,fma,bmi,bmi2"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "f16c", "fma", "bmi", "bmi2")
#endif

NK_INTERNAL __m256i nk_bf16x16_to_comparable_i16x16_haswell_(__m256i raw_u16x16) {
    __m256i sign_i16x16 = _mm256_srai_epi16(raw_u16x16, 15);
    __m256i flip_i16x16 = _mm256_srli_epi16(sign_i16x16, 1);
    return _mm256_xor_si256(raw_u16x16, flip_i16x16);
}

NK_INTERNAL __m256i nk_f16x16_to_comparable_i16x16_haswell_(__m256i raw_u16x16) {
    __m256i sign_i16x16 = _mm256_srai_epi16(raw_u16x16, 15);
    __m256i flip_i16x16 = _mm256_srli_epi16(sign_i16x16, 1);
    return _mm256_xor_si256(raw_u16x16, flip_i16x16);
}

NK_INTERNAL __m256i nk_u64_sadd_epi64_haswell_(__m256i a, __m256i b) {
    __m256i result = _mm256_add_epi64(a, b);
    nk_b256_vec_t va, vr;
    va.ymm = a;
    vr.ymm = result;
    for (int i = 0; i < 4; i++)
        if (vr.u64s[i] < va.u64s[i]) vr.u64s[i] = (nk_u64_t)-1;
    return vr.ymm;
}

NK_INTERNAL __m256i nk_i64_smul_sq_epi64_haswell_(__m256i val) {
    nk_b256_vec_t v, r;
    v.ymm = val;
    for (int i = 0; i < 4; i++) nk_i64_smul_(&v.i64s[i], &v.i64s[i], &r.i64s[i]);
    return r.ymm;
}

NK_INTERNAL __m256i nk_u64_smul_sq_epi64_haswell_(__m256i val) {
    nk_b256_vec_t v, r;
    v.ymm = val;
    for (int i = 0; i < 4; i++) nk_u64_smul_(&v.u64s[i], &v.u64s[i], &r.u64s[i]);
    return r.ymm;
}

NK_INTERNAL nk_u64_t nk_reduce_sadd_u64x4_haswell_(__m256i v) {
    nk_b256_vec_t vec;
    vec.ymm = v;
    nk_u64_t s = 0;
    for (int i = 0; i < 4; i++) nk_u64_sadd_(&s, &vec.u64s[i], &s);
    return s;
}

NK_INTERNAL void nk_reduce_moments_f32_haswell_contiguous_( //
    nk_f32_t const *data, nk_size_t count,                  //
    nk_f64_t *sum, nk_f64_t *sumsq) {
    __m256d sum_low_f64x4 = _mm256_setzero_pd(), sum_high_f64x4 = _mm256_setzero_pd();
    __m256d sumsq_low_f64x4 = _mm256_setzero_pd(), sumsq_high_f64x4 = _mm256_setzero_pd();
    nk_size_t idx = 0;
    for (; idx + 8 <= count; idx += 8) {
        __m256d low_f64x4 = _mm256_cvtps_pd(_mm_loadu_ps(data + idx));
        __m256d high_f64x4 = _mm256_cvtps_pd(_mm_loadu_ps(data + idx + 4));
        sum_low_f64x4 = _mm256_add_pd(sum_low_f64x4, low_f64x4);
        sum_high_f64x4 = _mm256_add_pd(sum_high_f64x4, high_f64x4);
        sumsq_low_f64x4 = _mm256_fmadd_pd(low_f64x4, low_f64x4, sumsq_low_f64x4);
        sumsq_high_f64x4 = _mm256_fmadd_pd(high_f64x4, high_f64x4, sumsq_high_f64x4);
    }
    __m256d sum_f64x4 = _mm256_add_pd(sum_low_f64x4, sum_high_f64x4);
    __m256d sumsq_f64x4 = _mm256_add_pd(sumsq_low_f64x4, sumsq_high_f64x4);
    nk_f64_t s = nk_reduce_add_f64x4_haswell_(sum_f64x4);
    nk_f64_t sq = nk_reduce_add_f64x4_haswell_(sumsq_f64x4);
    for (; idx < count; ++idx) {
        nk_f64_t val = (nk_f64_t)data[idx];
        s += val;
        sq += val * val;
    }
    *sum = s;
    *sumsq = sq;
}

NK_INTERNAL void nk_reduce_moments_f32_haswell_strided_(              //
    nk_f32_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_f64_t *sum, nk_f64_t *sumsq) {
    __m256i blend_mask_i32x8 = nk_stride_blend_b32x8_(stride_elements);
    __m128 blend_lo_f32x4 = _mm_castsi128_ps(_mm256_castsi256_si128(blend_mask_i32x8));
    __m128 blend_hi_f32x4 = _mm_castsi128_ps(_mm256_extracti128_si256(blend_mask_i32x8, 1));
    __m128 zero_f32x4 = _mm_setzero_ps();
    __m256d sum_low_f64x4 = _mm256_setzero_pd(), sum_high_f64x4 = _mm256_setzero_pd();
    __m256d sumsq_low_f64x4 = _mm256_setzero_pd(), sumsq_high_f64x4 = _mm256_setzero_pd();
    nk_size_t idx = 0, total = count * stride_elements;
    for (; idx + 8 <= total; idx += 8) {
        __m128 lo_f32x4 = _mm_blendv_ps(zero_f32x4, _mm_loadu_ps(data + idx), blend_lo_f32x4);
        __m128 hi_f32x4 = _mm_blendv_ps(zero_f32x4, _mm_loadu_ps(data + idx + 4), blend_hi_f32x4);
        __m256d lo_f64x4 = _mm256_cvtps_pd(lo_f32x4);
        __m256d hi_f64x4 = _mm256_cvtps_pd(hi_f32x4);
        sum_low_f64x4 = _mm256_add_pd(sum_low_f64x4, lo_f64x4);
        sum_high_f64x4 = _mm256_add_pd(sum_high_f64x4, hi_f64x4);
        sumsq_low_f64x4 = _mm256_fmadd_pd(lo_f64x4, lo_f64x4, sumsq_low_f64x4);
        sumsq_high_f64x4 = _mm256_fmadd_pd(hi_f64x4, hi_f64x4, sumsq_high_f64x4);
    }
    __m256d sum_f64x4 = _mm256_add_pd(sum_low_f64x4, sum_high_f64x4);
    __m256d sumsq_f64x4 = _mm256_add_pd(sumsq_low_f64x4, sumsq_high_f64x4);
    nk_f64_t s = nk_reduce_add_f64x4_haswell_(sum_f64x4);
    nk_f64_t sq = nk_reduce_add_f64x4_haswell_(sumsq_f64x4);
    nk_f32_t const *ptr = data + idx;
    nk_size_t remaining = count - idx / stride_elements;
    for (nk_size_t i = 0; i < remaining; ++i, ptr += stride_elements) {
        nk_f64_t val = (nk_f64_t)(*ptr);
        s += val;
        sq += val * val;
    }
    *sum = s;
    *sumsq = sq;
}

NK_INTERNAL void nk_reduce_moments_f32_haswell_gather_(            //
    nk_f32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f64_t *sum, nk_f64_t *sumsq) {
    nk_i32_t stride_elements = (nk_i32_t)(stride_bytes / sizeof(nk_f32_t));
    __m256i indices_i32x8 = _mm256_mullo_epi32(_mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7),
                                               _mm256_set1_epi32(stride_elements));
    __m256d sum_f64x4 = _mm256_setzero_pd();
    __m256d sumsq_f64x4 = _mm256_setzero_pd();
    nk_size_t idx = 0;
    for (; idx + 8 <= count; idx += 8) {
        __m256 gathered_f32x8 = _mm256_i32gather_ps(data + idx * stride_elements, indices_i32x8, sizeof(nk_f32_t));
        __m256d lo_f64x4 = _mm256_cvtps_pd(_mm256_castps256_ps128(gathered_f32x8));
        __m256d hi_f64x4 = _mm256_cvtps_pd(_mm256_extractf128_ps(gathered_f32x8, 1));
        sum_f64x4 = _mm256_add_pd(sum_f64x4, lo_f64x4);
        sum_f64x4 = _mm256_add_pd(sum_f64x4, hi_f64x4);
        sumsq_f64x4 = _mm256_fmadd_pd(lo_f64x4, lo_f64x4, sumsq_f64x4);
        sumsq_f64x4 = _mm256_fmadd_pd(hi_f64x4, hi_f64x4, sumsq_f64x4);
    }
    nk_f64_t s = nk_reduce_add_f64x4_haswell_(sum_f64x4);
    nk_f64_t sq = nk_reduce_add_f64x4_haswell_(sumsq_f64x4);
    unsigned char const *ptr = (unsigned char const *)(data + idx * stride_elements);
    for (; idx < count; ++idx, ptr += stride_bytes) {
        nk_f64_t val = (nk_f64_t)(*(nk_f32_t const *)ptr);
        s += val;
        sq += val * val;
    }
    *sum = s;
    *sumsq = sq;
}

NK_PUBLIC void nk_reduce_moments_f32_haswell(                      //
    nk_f32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f64_t *sum, nk_f64_t *sumsq) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_f32_t);
    int aligned = (stride_bytes % sizeof(nk_f32_t) == 0);
    if (count == 0) *sum = 0, *sumsq = 0;
    else if (!aligned) nk_reduce_moments_f32_serial(data, count, stride_bytes, sum, sumsq);
    else if (count > (nk_size_t)(NK_U16_MAX + 1) * 8) {
        nk_size_t left_count = count / 2;
        nk_f64_t left_sum, left_sumsq, right_sum, right_sumsq;
        nk_reduce_moments_f32_haswell(data, left_count, stride_bytes, &left_sum, &left_sumsq);
        nk_reduce_moments_f32_haswell(data + left_count * stride_elements, count - left_count, stride_bytes, &right_sum,
                                      &right_sumsq);
        *sum = left_sum + right_sum, *sumsq = left_sumsq + right_sumsq;
    }
    else if (stride_elements == 1) nk_reduce_moments_f32_haswell_contiguous_(data, count, sum, sumsq);
    else if (stride_elements <= 8) nk_reduce_moments_f32_haswell_strided_(data, count, stride_elements, sum, sumsq);
    else nk_reduce_moments_f32_haswell_gather_(data, count, stride_bytes, sum, sumsq);
}

NK_INTERNAL void nk_reduce_minmax_f32_haswell_contiguous_( //
    nk_f32_t const *data, nk_size_t count,                 //
    nk_f32_t *min_value, nk_size_t *min_index,             //
    nk_f32_t *max_value, nk_size_t *max_index) {
    __m256 min_f32x8 = _mm256_loadu_ps(data);
    __m256 max_f32x8 = min_f32x8;
    __m256i min_iter_u32x8 = _mm256_setzero_si256();
    __m256i max_iter_u32x8 = _mm256_setzero_si256();
    __m256i iter_u32x8 = _mm256_set1_epi32(1);
    __m256i one_u32x8 = _mm256_set1_epi32(1);

    nk_size_t idx = 8;
    for (; idx + 8 <= count; idx += 8) {
        __m256 data_f32x8 = _mm256_loadu_ps(data + idx);
        __m256 less_b32x8 = _mm256_cmp_ps(data_f32x8, min_f32x8, _CMP_LT_OQ);
        __m256 greater_b32x8 = _mm256_cmp_ps(data_f32x8, max_f32x8, _CMP_GT_OQ);
        min_f32x8 = _mm256_blendv_ps(min_f32x8, data_f32x8, less_b32x8);
        max_f32x8 = _mm256_blendv_ps(max_f32x8, data_f32x8, greater_b32x8);
        min_iter_u32x8 = _mm256_blendv_epi8(min_iter_u32x8, iter_u32x8, _mm256_castps_si256(less_b32x8));
        max_iter_u32x8 = _mm256_blendv_epi8(max_iter_u32x8, iter_u32x8, _mm256_castps_si256(greater_b32x8));
        iter_u32x8 = _mm256_add_epi32(iter_u32x8, one_u32x8);
    }

    // Scalar tail
    nk_f32_t min_scalar = nk_reduce_min_f32x8_haswell_(min_f32x8);
    nk_f32_t max_scalar = nk_reduce_max_f32x8_haswell_(max_f32x8);
    nk_size_t min_idx = 0, max_idx = 0;
    {
        __m256 min_eq_b32x8 = _mm256_cmp_ps(min_f32x8, _mm256_set1_ps(min_scalar), _CMP_EQ_OQ);
        unsigned int min_lane = _tzcnt_u32((unsigned int)_mm256_movemask_ps(min_eq_b32x8));
        nk_b256_vec_t iter_vec;
        iter_vec.ymm = min_iter_u32x8;
        min_idx = (nk_size_t)iter_vec.u32s[min_lane] * 8 + min_lane;
    }
    {
        __m256 max_eq_b32x8 = _mm256_cmp_ps(max_f32x8, _mm256_set1_ps(max_scalar), _CMP_EQ_OQ);
        unsigned int max_lane = _tzcnt_u32((unsigned int)_mm256_movemask_ps(max_eq_b32x8));
        nk_b256_vec_t iter_vec;
        iter_vec.ymm = max_iter_u32x8;
        max_idx = (nk_size_t)iter_vec.u32s[max_lane] * 8 + max_lane;
    }
    for (; idx < count; ++idx) {
        nk_f32_t val = data[idx];
        if (val < min_scalar) min_scalar = val, min_idx = idx;
        if (val > max_scalar) max_scalar = val, max_idx = idx;
    }
    *min_value = min_scalar, *min_index = min_idx;
    *max_value = max_scalar, *max_index = max_idx;
}

NK_PUBLIC void nk_reduce_minmax_f32_haswell(                       //
    nk_f32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *min_value, nk_size_t *min_index,                     //
    nk_f32_t *max_value, nk_size_t *max_index) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_f32_t);
    int aligned = (stride_bytes % sizeof(nk_f32_t) == 0);
    if (count == 0)
        *min_value = NK_F32_MAX, *min_index = NK_SIZE_MAX, *max_value = NK_F32_MIN, *max_index = NK_SIZE_MAX;
    else if (!aligned)
        nk_reduce_minmax_f32_serial(data, count, stride_bytes, min_value, min_index, max_value, max_index);
    else if (count > (nk_size_t)NK_U32_MAX * 8) {
        nk_size_t left_count = count / 2;
        nk_f32_t left_min, right_min, left_max, right_max;
        nk_size_t left_min_index, right_min_index, left_max_index, right_max_index;
        nk_reduce_minmax_f32_haswell(data, left_count, stride_bytes, &left_min, &left_min_index, &left_max,
                                     &left_max_index);
        nk_reduce_minmax_f32_haswell(data + left_count * stride_elements, count - left_count, stride_bytes, &right_min,
                                     &right_min_index, &right_max, &right_max_index);
        if (right_min < left_min) *min_value = right_min, *min_index = left_count + right_min_index;
        else *min_value = left_min, *min_index = left_min_index;
        if (right_max > left_max) *max_value = right_max, *max_index = left_count + right_max_index;
        else *max_value = left_max, *max_index = left_max_index;
    }
    else if (stride_elements == 1)
        nk_reduce_minmax_f32_haswell_contiguous_(data, count, min_value, min_index, max_value, max_index);
    else nk_reduce_minmax_f32_serial(data, count, stride_bytes, min_value, min_index, max_value, max_index);
}

NK_INTERNAL void nk_reduce_moments_f64_haswell_contiguous_( //
    nk_f64_t const *data, nk_size_t count,                  //
    nk_f64_t *sum, nk_f64_t *sumsq) {
    __m256d sum_f64x4 = _mm256_setzero_pd();
    __m256d sum_comp_f64x4 = _mm256_setzero_pd();
    __m256d sumsq_f64x4 = _mm256_setzero_pd();
    __m256d sumsq_comp_f64x4 = _mm256_setzero_pd();
    nk_size_t idx = 0;
    for (; idx + 4 <= count; idx += 4) {
        __m256d val_f64x4 = _mm256_loadu_pd(data + idx);
        __m256d tentative_f64x4 = _mm256_add_pd(sum_f64x4, val_f64x4);
        __m256d round_f64x4 = _mm256_sub_pd(tentative_f64x4, sum_f64x4);
        __m256d corr_f64x4 = _mm256_add_pd(_mm256_sub_pd(sum_f64x4, _mm256_sub_pd(tentative_f64x4, round_f64x4)),
                                           _mm256_sub_pd(val_f64x4, round_f64x4));
        sum_comp_f64x4 = _mm256_add_pd(sum_comp_f64x4, corr_f64x4);
        sum_f64x4 = tentative_f64x4;
        __m256d sq_f64x4 = _mm256_mul_pd(val_f64x4, val_f64x4);
        __m256d tentative_sq_f64x4 = _mm256_add_pd(sumsq_f64x4, sq_f64x4);
        __m256d round_sq_f64x4 = _mm256_sub_pd(tentative_sq_f64x4, sumsq_f64x4);
        __m256d corr_sq_f64x4 = _mm256_add_pd(
            _mm256_sub_pd(sumsq_f64x4, _mm256_sub_pd(tentative_sq_f64x4, round_sq_f64x4)),
            _mm256_sub_pd(sq_f64x4, round_sq_f64x4));
        sumsq_comp_f64x4 = _mm256_add_pd(sumsq_comp_f64x4, corr_sq_f64x4);
        sumsq_f64x4 = tentative_sq_f64x4;
    }
    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        nk_b256_vec_t tail_vec;
        nk_partial_load_b64x4_serial_(data + idx, &tail_vec, remaining);
        __m256d val_f64x4 = tail_vec.ymm_pd;
        __m256d tentative_f64x4 = _mm256_add_pd(sum_f64x4, val_f64x4);
        __m256d round_f64x4 = _mm256_sub_pd(tentative_f64x4, sum_f64x4);
        __m256d corr_f64x4 = _mm256_add_pd(_mm256_sub_pd(sum_f64x4, _mm256_sub_pd(tentative_f64x4, round_f64x4)),
                                           _mm256_sub_pd(val_f64x4, round_f64x4));
        sum_comp_f64x4 = _mm256_add_pd(sum_comp_f64x4, corr_f64x4);
        sum_f64x4 = tentative_f64x4;
        __m256d sq_f64x4 = _mm256_mul_pd(val_f64x4, val_f64x4);
        __m256d tentative_sq_f64x4 = _mm256_add_pd(sumsq_f64x4, sq_f64x4);
        __m256d round_sq_f64x4 = _mm256_sub_pd(tentative_sq_f64x4, sumsq_f64x4);
        __m256d corr_sq_f64x4 = _mm256_add_pd(
            _mm256_sub_pd(sumsq_f64x4, _mm256_sub_pd(tentative_sq_f64x4, round_sq_f64x4)),
            _mm256_sub_pd(sq_f64x4, round_sq_f64x4));
        sumsq_comp_f64x4 = _mm256_add_pd(sumsq_comp_f64x4, corr_sq_f64x4);
        sumsq_f64x4 = tentative_sq_f64x4;
    }
    *sum = nk_reduce_add_f64x4_haswell_(_mm256_add_pd(sum_f64x4, sum_comp_f64x4));
    *sumsq = nk_reduce_add_f64x4_haswell_(_mm256_add_pd(sumsq_f64x4, sumsq_comp_f64x4));
}

NK_INTERNAL void nk_reduce_moments_f64_haswell_strided_(              //
    nk_f64_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_f64_t *sum, nk_f64_t *sumsq) {
    __m256i blend_mask_i64x4 = nk_stride_blend_b64x4_(stride_elements);
    __m256d blend_f64x4 = _mm256_castsi256_pd(blend_mask_i64x4);
    __m256d zero_f64x4 = _mm256_setzero_pd();
    __m256d sum_f64x4 = _mm256_setzero_pd();
    __m256d sum_comp_f64x4 = _mm256_setzero_pd();
    __m256d sumsq_f64x4 = _mm256_setzero_pd();
    __m256d sumsq_comp_f64x4 = _mm256_setzero_pd();
    nk_size_t idx = 0, total = count * stride_elements;
    for (; idx + 4 <= total; idx += 4) {
        __m256d val_f64x4 = _mm256_blendv_pd(zero_f64x4, _mm256_loadu_pd(data + idx), blend_f64x4);
        __m256d tentative_f64x4 = _mm256_add_pd(sum_f64x4, val_f64x4);
        __m256d round_f64x4 = _mm256_sub_pd(tentative_f64x4, sum_f64x4);
        __m256d corr_f64x4 = _mm256_add_pd(_mm256_sub_pd(sum_f64x4, _mm256_sub_pd(tentative_f64x4, round_f64x4)),
                                           _mm256_sub_pd(val_f64x4, round_f64x4));
        sum_comp_f64x4 = _mm256_add_pd(sum_comp_f64x4, corr_f64x4);
        sum_f64x4 = tentative_f64x4;
        __m256d sq_f64x4 = _mm256_mul_pd(val_f64x4, val_f64x4);
        __m256d tentative_sq_f64x4 = _mm256_add_pd(sumsq_f64x4, sq_f64x4);
        __m256d round_sq_f64x4 = _mm256_sub_pd(tentative_sq_f64x4, sumsq_f64x4);
        __m256d corr_sq_f64x4 = _mm256_add_pd(
            _mm256_sub_pd(sumsq_f64x4, _mm256_sub_pd(tentative_sq_f64x4, round_sq_f64x4)),
            _mm256_sub_pd(sq_f64x4, round_sq_f64x4));
        sumsq_comp_f64x4 = _mm256_add_pd(sumsq_comp_f64x4, corr_sq_f64x4);
        sumsq_f64x4 = tentative_sq_f64x4;
    }
    nk_f64_t s = nk_reduce_add_f64x4_haswell_(_mm256_add_pd(sum_f64x4, sum_comp_f64x4));
    nk_f64_t sq = nk_reduce_add_f64x4_haswell_(_mm256_add_pd(sumsq_f64x4, sumsq_comp_f64x4));
    nk_f64_t const *ptr = data + idx;
    nk_size_t remaining_elements = count - idx / stride_elements;
    for (nk_size_t i = 0; i < remaining_elements; ++i, ptr += stride_elements) {
        nk_f64_t val = *ptr;
        s += val;
        sq += val * val;
    }
    *sum = s;
    *sumsq = sq;
}

NK_PUBLIC void nk_reduce_moments_f64_haswell(                      //
    nk_f64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f64_t *sum, nk_f64_t *sumsq) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_f64_t);
    int aligned = (stride_bytes % sizeof(nk_f64_t) == 0);
    if (count == 0) *sum = 0, *sumsq = 0;
    else if (!aligned) nk_reduce_moments_f64_serial(data, count, stride_bytes, sum, sumsq);
    else if (count > (nk_size_t)(NK_U16_MAX + 1) * 4) {
        nk_size_t left_count = count / 2;
        nk_f64_t left_sum, left_sumsq, right_sum, right_sumsq;
        nk_reduce_moments_f64_haswell(data, left_count, stride_bytes, &left_sum, &left_sumsq);
        nk_reduce_moments_f64_haswell(data + left_count * stride_elements, count - left_count, stride_bytes, &right_sum,
                                      &right_sumsq);
        *sum = left_sum + right_sum, *sumsq = left_sumsq + right_sumsq;
    }
    else if (stride_elements == 1) nk_reduce_moments_f64_haswell_contiguous_(data, count, sum, sumsq);
    else if (stride_elements <= 4) nk_reduce_moments_f64_haswell_strided_(data, count, stride_elements, sum, sumsq);
    else nk_reduce_moments_f64_serial(data, count, stride_bytes, sum, sumsq);
}

NK_INTERNAL void nk_reduce_minmax_f64_haswell_contiguous_( //
    nk_f64_t const *data, nk_size_t count,                 //
    nk_f64_t *min_value, nk_size_t *min_index,             //
    nk_f64_t *max_value, nk_size_t *max_index) {
    __m256d min_f64x4 = _mm256_loadu_pd(data);
    __m256d max_f64x4 = min_f64x4;
    __m256i min_iter_u64x4 = _mm256_setzero_si256();
    __m256i max_iter_u64x4 = _mm256_setzero_si256();
    __m256i iter_u64x4 = _mm256_set1_epi64x(1);
    __m256i one_u64x4 = _mm256_set1_epi64x(1);

    nk_size_t idx = 4;
    for (; idx + 4 <= count; idx += 4) {
        __m256d data_f64x4 = _mm256_loadu_pd(data + idx);
        __m256d less_b64x4 = _mm256_cmp_pd(data_f64x4, min_f64x4, _CMP_LT_OQ);
        __m256d greater_b64x4 = _mm256_cmp_pd(data_f64x4, max_f64x4, _CMP_GT_OQ);
        min_f64x4 = _mm256_blendv_pd(min_f64x4, data_f64x4, less_b64x4);
        max_f64x4 = _mm256_blendv_pd(max_f64x4, data_f64x4, greater_b64x4);
        min_iter_u64x4 = _mm256_blendv_epi8(min_iter_u64x4, iter_u64x4, _mm256_castpd_si256(less_b64x4));
        max_iter_u64x4 = _mm256_blendv_epi8(max_iter_u64x4, iter_u64x4, _mm256_castpd_si256(greater_b64x4));
        iter_u64x4 = _mm256_add_epi64(iter_u64x4, one_u64x4);
    }

    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        nk_b256_vec_t tail_vec;
        nk_partial_load_b64x4_serial_(data + idx, &tail_vec, remaining);
        __m256i lane_indices_i64x4 = _mm256_setr_epi64x(0, 1, 2, 3);
        __m256i threshold_i64x4 = _mm256_set1_epi64x((nk_i64_t)remaining);
        __m256i valid_i64x4 = _mm256_cmpgt_epi64(threshold_i64x4, lane_indices_i64x4);
        __m256d valid_b64x4 = _mm256_castsi256_pd(valid_i64x4);
        __m256d nan_f64x4 = _mm256_castsi256_pd(_mm256_set1_epi64x((long long)0x7FF8000000000000LL));
        __m256d data_f64x4 = _mm256_blendv_pd(nan_f64x4, tail_vec.ymm_pd, valid_b64x4);
        __m256d less_b64x4 = _mm256_cmp_pd(data_f64x4, min_f64x4, _CMP_LT_OQ);
        __m256d greater_b64x4 = _mm256_cmp_pd(data_f64x4, max_f64x4, _CMP_GT_OQ);
        min_f64x4 = _mm256_blendv_pd(min_f64x4, data_f64x4, less_b64x4);
        max_f64x4 = _mm256_blendv_pd(max_f64x4, data_f64x4, greater_b64x4);
        min_iter_u64x4 = _mm256_blendv_epi8(min_iter_u64x4, iter_u64x4, _mm256_castpd_si256(less_b64x4));
        max_iter_u64x4 = _mm256_blendv_epi8(max_iter_u64x4, iter_u64x4, _mm256_castpd_si256(greater_b64x4));
    }

    nk_f64_t min_scalar = nk_reduce_min_f64x4_haswell_(min_f64x4);
    nk_f64_t max_scalar = nk_reduce_max_f64x4_haswell_(max_f64x4);
    {
        __m256d min_eq_b64x4 = _mm256_cmp_pd(min_f64x4, _mm256_set1_pd(min_scalar), _CMP_EQ_OQ);
        unsigned int min_lane = _tzcnt_u32((unsigned int)_mm256_movemask_pd(min_eq_b64x4));
        nk_b256_vec_t iter_vec;
        iter_vec.ymm = min_iter_u64x4;
        *min_value = min_scalar;
        *min_index = (nk_size_t)iter_vec.u64s[min_lane] * 4 + min_lane;
    }
    {
        __m256d max_eq_b64x4 = _mm256_cmp_pd(max_f64x4, _mm256_set1_pd(max_scalar), _CMP_EQ_OQ);
        unsigned int max_lane = _tzcnt_u32((unsigned int)_mm256_movemask_pd(max_eq_b64x4));
        nk_b256_vec_t iter_vec;
        iter_vec.ymm = max_iter_u64x4;
        *max_value = max_scalar;
        *max_index = (nk_size_t)iter_vec.u64s[max_lane] * 4 + max_lane;
    }
}

NK_PUBLIC void nk_reduce_minmax_f64_haswell(                       //
    nk_f64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f64_t *min_value, nk_size_t *min_index,                     //
    nk_f64_t *max_value, nk_size_t *max_index) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_f64_t);
    int aligned = (stride_bytes % sizeof(nk_f64_t) == 0);
    if (count == 0)
        *min_value = NK_F64_MAX, *min_index = NK_SIZE_MAX, *max_value = NK_F64_MIN, *max_index = NK_SIZE_MAX;
    else if (!aligned)
        nk_reduce_minmax_f64_serial(data, count, stride_bytes, min_value, min_index, max_value, max_index);
    else if (stride_elements == 1)
        nk_reduce_minmax_f64_haswell_contiguous_(data, count, min_value, min_index, max_value, max_index);
    else nk_reduce_minmax_f64_serial(data, count, stride_bytes, min_value, min_index, max_value, max_index);
}

NK_INTERNAL void nk_reduce_moments_i8_haswell_contiguous_( //
    nk_i8_t const *data, nk_size_t count,                  //
    nk_i64_t *sum, nk_u64_t *sumsq) {
    __m256i bias_i8x32 = _mm256_set1_epi8((char)0x80);
    __m256i zero_i8x32 = _mm256_setzero_si256();
    __m256i sum_u64x4 = _mm256_setzero_si256();
    __m256i sumsq_lo_i32x8 = _mm256_setzero_si256();
    __m256i sumsq_hi_i32x8 = _mm256_setzero_si256();
    nk_size_t idx = 0;
    for (; idx + 32 <= count; idx += 32) {
        __m256i data_i8x32 = _mm256_loadu_si256((__m256i const *)(data + idx));
        __m256i unsigned_u8x32 = _mm256_xor_si256(data_i8x32, bias_i8x32);
        sum_u64x4 = _mm256_add_epi64(sum_u64x4, _mm256_sad_epu8(unsigned_u8x32, zero_i8x32));
        __m256i lo_i16x16 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(data_i8x32));
        __m256i hi_i16x16 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(data_i8x32, 1));
        sumsq_lo_i32x8 = _mm256_add_epi32(sumsq_lo_i32x8, _mm256_madd_epi16(lo_i16x16, lo_i16x16));
        sumsq_hi_i32x8 = _mm256_add_epi32(sumsq_hi_i32x8, _mm256_madd_epi16(hi_i16x16, hi_i16x16));
    }
    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        nk_b256_vec_t tail_vec;
        nk_partial_load_b8x32_serial_(data + idx, &tail_vec, remaining);
        __m256i data_i8x32 = tail_vec.ymm;
        // Build masked bias: only bias valid lanes so zero-padded lanes stay zero
        nk_b256_vec_t mask_vec;
        mask_vec.ymm = _mm256_setzero_si256();
        for (nk_size_t i = 0; i < remaining; ++i) mask_vec.u8s[i] = 0xFF;
        __m256i masked_bias_i8x32 = _mm256_and_si256(bias_i8x32, mask_vec.ymm);
        __m256i unsigned_u8x32 = _mm256_xor_si256(data_i8x32, masked_bias_i8x32);
        sum_u64x4 = _mm256_add_epi64(sum_u64x4, _mm256_sad_epu8(unsigned_u8x32, zero_i8x32));
        __m256i lo_i16x16 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(data_i8x32));
        __m256i hi_i16x16 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(data_i8x32, 1));
        sumsq_lo_i32x8 = _mm256_add_epi32(sumsq_lo_i32x8, _mm256_madd_epi16(lo_i16x16, lo_i16x16));
        sumsq_hi_i32x8 = _mm256_add_epi32(sumsq_hi_i32x8, _mm256_madd_epi16(hi_i16x16, hi_i16x16));
    }
    sumsq_lo_i32x8 = _mm256_add_epi32(sumsq_lo_i32x8, sumsq_hi_i32x8);
    __m256i sumsq_i64x4 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(sumsq_lo_i32x8));
    sumsq_i64x4 = _mm256_add_epi64(sumsq_i64x4, _mm256_cvtepi32_epi64(_mm256_extracti128_si256(sumsq_lo_i32x8, 1)));
    nk_i64_t s = (nk_i64_t)(nk_u64_t)nk_reduce_add_i64x4_haswell_(sum_u64x4);
    s -= (nk_i64_t)128 * (nk_i64_t)count;
    nk_u64_t sq = (nk_u64_t)nk_reduce_add_i64x4_haswell_(sumsq_i64x4);
    *sum = s;
    *sumsq = sq;
}

NK_INTERNAL void nk_reduce_moments_i8_haswell_strided_(              //
    nk_i8_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_i64_t *sum, nk_u64_t *sumsq) {
    __m256i stride_mask_i8x32 = nk_stride_blend_u1x32_(stride_elements);
    __m256i masked_bias_i8x32 = _mm256_and_si256(_mm256_set1_epi8((char)0x80), stride_mask_i8x32);
    __m256i zero_i8x32 = _mm256_setzero_si256();
    __m256i sum_u64x4 = _mm256_setzero_si256();
    __m256i sumsq_lo_i32x8 = _mm256_setzero_si256();
    __m256i sumsq_hi_i32x8 = _mm256_setzero_si256();
    nk_size_t idx_scalars = 0;
    nk_size_t total_scalars = count * stride_elements;
    nk_size_t elements_per_vector = 32 / stride_elements;
    nk_size_t vector_element_count = 0;
    for (; idx_scalars + 32 <= total_scalars; idx_scalars += 32) {
        __m256i data_i8x32 = _mm256_loadu_si256((__m256i const *)(data + idx_scalars));
        data_i8x32 = _mm256_and_si256(data_i8x32, stride_mask_i8x32);
        __m256i unsigned_u8x32 = _mm256_xor_si256(data_i8x32, masked_bias_i8x32);
        sum_u64x4 = _mm256_add_epi64(sum_u64x4, _mm256_sad_epu8(unsigned_u8x32, zero_i8x32));
        __m256i lo_i16x16 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(data_i8x32));
        __m256i hi_i16x16 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(data_i8x32, 1));
        sumsq_lo_i32x8 = _mm256_add_epi32(sumsq_lo_i32x8, _mm256_madd_epi16(lo_i16x16, lo_i16x16));
        sumsq_hi_i32x8 = _mm256_add_epi32(sumsq_hi_i32x8, _mm256_madd_epi16(hi_i16x16, hi_i16x16));
        vector_element_count += elements_per_vector;
    }
    sumsq_lo_i32x8 = _mm256_add_epi32(sumsq_lo_i32x8, sumsq_hi_i32x8);
    __m256i sumsq_i64x4 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(sumsq_lo_i32x8));
    sumsq_i64x4 = _mm256_add_epi64(sumsq_i64x4, _mm256_cvtepi32_epi64(_mm256_extracti128_si256(sumsq_lo_i32x8, 1)));
    nk_i64_t s = (nk_i64_t)(nk_u64_t)nk_reduce_add_i64x4_haswell_(sum_u64x4);
    s -= (nk_i64_t)128 * (nk_i64_t)vector_element_count;
    nk_u64_t sq = (nk_u64_t)nk_reduce_add_i64x4_haswell_(sumsq_i64x4);
    nk_i8_t const *ptr = data + idx_scalars;
    nk_size_t remaining = count - idx_scalars / stride_elements;
    for (nk_size_t i = 0; i < remaining; ++i, ptr += stride_elements) {
        nk_i64_t val = (nk_i64_t)*ptr;
        s += val;
        sq += (nk_u64_t)(val * val);
    }
    *sum = s;
    *sumsq = sq;
}

NK_PUBLIC void nk_reduce_moments_i8_haswell(                      //
    nk_i8_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *sum, nk_u64_t *sumsq) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_i8_t);
    int aligned = (stride_bytes % sizeof(nk_i8_t) == 0);
    if (count == 0) *sum = 0, *sumsq = 0;
    else if (!aligned) nk_reduce_moments_i8_serial(data, count, stride_bytes, sum, sumsq);
    else if (count > (nk_size_t)(NK_U16_MAX + 1) * 32) {
        nk_size_t left_count = count / 2;
        nk_i64_t left_sum, right_sum;
        nk_u64_t left_sumsq, right_sumsq;
        nk_reduce_moments_i8_haswell(data, left_count, stride_bytes, &left_sum, &left_sumsq);
        nk_reduce_moments_i8_haswell(data + left_count * stride_elements, count - left_count, stride_bytes, &right_sum,
                                     &right_sumsq);
        nk_i64_sadd_(&left_sum, &right_sum, sum);
        nk_u64_sadd_(&left_sumsq, &right_sumsq, sumsq);
    }
    else if (stride_elements == 1) nk_reduce_moments_i8_haswell_contiguous_(data, count, sum, sumsq);
    else if (stride_elements <= 8) nk_reduce_moments_i8_haswell_strided_(data, count, stride_elements, sum, sumsq);
    else nk_reduce_moments_i8_serial(data, count, stride_bytes, sum, sumsq);
}

NK_INTERNAL void nk_reduce_minmax_i8_haswell_contiguous_( //
    nk_i8_t const *data, nk_size_t count,                 //
    nk_i8_t *min_value, nk_size_t *min_index,             //
    nk_i8_t *max_value, nk_size_t *max_index) {
    __m256i min_i8x32 = _mm256_loadu_si256((__m256i const *)data);
    __m256i max_i8x32 = min_i8x32;
    __m256i min_iter_u8x32 = _mm256_setzero_si256();
    __m256i max_iter_u8x32 = _mm256_setzero_si256();
    __m256i iter_u8x32 = _mm256_set1_epi8(1);
    __m256i one_u8x32 = _mm256_set1_epi8(1);

    nk_size_t idx = 32;
    for (; idx + 32 <= count; idx += 32) {
        __m256i data_i8x32 = _mm256_loadu_si256((__m256i const *)(data + idx));
        __m256i less_b8x32 = _mm256_cmpgt_epi8(min_i8x32, data_i8x32);
        __m256i greater_b8x32 = _mm256_cmpgt_epi8(data_i8x32, max_i8x32);
        min_i8x32 = _mm256_blendv_epi8(min_i8x32, data_i8x32, less_b8x32);
        max_i8x32 = _mm256_blendv_epi8(max_i8x32, data_i8x32, greater_b8x32);
        min_iter_u8x32 = _mm256_blendv_epi8(min_iter_u8x32, iter_u8x32, less_b8x32);
        max_iter_u8x32 = _mm256_blendv_epi8(max_iter_u8x32, iter_u8x32, greater_b8x32);
        iter_u8x32 = _mm256_add_epi8(iter_u8x32, one_u8x32);
    }

    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        nk_b256_vec_t tail_vec;
        nk_partial_load_b8x32_serial_(data + idx, &tail_vec, remaining);
        __m256i data_i8x32 = tail_vec.ymm;
        // Build lane mask and fill invalid lanes with identity values
        nk_b256_vec_t mask_vec;
        mask_vec.ymm = _mm256_setzero_si256();
        for (nk_size_t i = 0; i < remaining; ++i) mask_vec.u8s[i] = 0xFF;
        __m256i valid_b8x32 = mask_vec.ymm;
        data_i8x32 = _mm256_blendv_epi8(_mm256_set1_epi8(NK_I8_MAX), data_i8x32, valid_b8x32);
        __m256i data_max_i8x32 = _mm256_blendv_epi8(_mm256_set1_epi8(NK_I8_MIN), tail_vec.ymm, valid_b8x32);
        __m256i less_b8x32 = _mm256_cmpgt_epi8(min_i8x32, data_i8x32);
        __m256i greater_b8x32 = _mm256_cmpgt_epi8(data_max_i8x32, max_i8x32);
        min_i8x32 = _mm256_blendv_epi8(min_i8x32, data_i8x32, less_b8x32);
        max_i8x32 = _mm256_blendv_epi8(max_i8x32, data_max_i8x32, greater_b8x32);
        min_iter_u8x32 = _mm256_blendv_epi8(min_iter_u8x32, iter_u8x32, less_b8x32);
        max_iter_u8x32 = _mm256_blendv_epi8(max_iter_u8x32, iter_u8x32, greater_b8x32);
    }

    nk_i8_t min_scalar = nk_reduce_min_i8x32_haswell_(min_i8x32);
    nk_i8_t max_scalar = nk_reduce_max_i8x32_haswell_(max_i8x32);
    __m256i min_eq_b8x32 = _mm256_cmpeq_epi8(min_i8x32, _mm256_set1_epi8(min_scalar));
    __m256i max_eq_b8x32 = _mm256_cmpeq_epi8(max_i8x32, _mm256_set1_epi8(max_scalar));
    unsigned int min_lane = _tzcnt_u32((unsigned int)_mm256_movemask_epi8(min_eq_b8x32));
    unsigned int max_lane = _tzcnt_u32((unsigned int)_mm256_movemask_epi8(max_eq_b8x32));
    nk_b256_vec_t iter_vec;
    iter_vec.ymm = min_iter_u8x32;
    *min_value = min_scalar, *min_index = (nk_size_t)iter_vec.u8s[min_lane] * 32 + min_lane;
    iter_vec.ymm = max_iter_u8x32;
    *max_value = max_scalar, *max_index = (nk_size_t)iter_vec.u8s[max_lane] * 32 + max_lane;
}

NK_PUBLIC void nk_reduce_minmax_i8_haswell(                       //
    nk_i8_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i8_t *min_value, nk_size_t *min_index,                     //
    nk_i8_t *max_value, nk_size_t *max_index) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_i8_t);
    int aligned = (stride_bytes % sizeof(nk_i8_t) == 0);
    if (count == 0) *min_value = NK_I8_MAX, *min_index = NK_SIZE_MAX, *max_value = NK_I8_MIN, *max_index = NK_SIZE_MAX;
    else if (!aligned)
        nk_reduce_minmax_i8_serial(data, count, stride_bytes, min_value, min_index, max_value, max_index);
    else if (count > (nk_size_t)(NK_U8_MAX + 1) * 32) {
        nk_size_t left_count = count / 2;
        nk_i8_t left_min, right_min, left_max, right_max;
        nk_size_t left_min_index, right_min_index, left_max_index, right_max_index;
        nk_reduce_minmax_i8_haswell(data, left_count, stride_bytes, &left_min, &left_min_index, &left_max,
                                    &left_max_index);
        nk_reduce_minmax_i8_haswell(data + left_count * stride_elements, count - left_count, stride_bytes, &right_min,
                                    &right_min_index, &right_max, &right_max_index);
        if (right_min < left_min) *min_value = right_min, *min_index = left_count + right_min_index;
        else *min_value = left_min, *min_index = left_min_index;
        if (right_max > left_max) *max_value = right_max, *max_index = left_count + right_max_index;
        else *max_value = left_max, *max_index = left_max_index;
    }
    else if (stride_elements == 1)
        nk_reduce_minmax_i8_haswell_contiguous_(data, count, min_value, min_index, max_value, max_index);
    else nk_reduce_minmax_i8_serial(data, count, stride_bytes, min_value, min_index, max_value, max_index);
}

NK_INTERNAL void nk_reduce_moments_u8_haswell_contiguous_( //
    nk_u8_t const *data, nk_size_t count,                  //
    nk_u64_t *sum, nk_u64_t *sumsq) {
    __m256i zero_u8x32 = _mm256_setzero_si256();
    __m256i sum_u64x4 = _mm256_setzero_si256();
    __m256i sumsq_lo_i32x8 = _mm256_setzero_si256();
    __m256i sumsq_hi_i32x8 = _mm256_setzero_si256();
    nk_size_t idx = 0;
    for (; idx + 32 <= count; idx += 32) {
        __m256i data_u8x32 = _mm256_loadu_si256((__m256i const *)(data + idx));
        sum_u64x4 = _mm256_add_epi64(sum_u64x4, _mm256_sad_epu8(data_u8x32, zero_u8x32));
        __m256i lo_i16x16 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(data_u8x32));
        __m256i hi_i16x16 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(data_u8x32, 1));
        sumsq_lo_i32x8 = _mm256_add_epi32(sumsq_lo_i32x8, _mm256_madd_epi16(lo_i16x16, lo_i16x16));
        sumsq_hi_i32x8 = _mm256_add_epi32(sumsq_hi_i32x8, _mm256_madd_epi16(hi_i16x16, hi_i16x16));
    }
    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        nk_b256_vec_t tail_vec;
        nk_partial_load_b8x32_serial_(data + idx, &tail_vec, remaining);
        __m256i data_u8x32 = tail_vec.ymm;
        sum_u64x4 = _mm256_add_epi64(sum_u64x4, _mm256_sad_epu8(data_u8x32, zero_u8x32));
        __m256i lo_i16x16 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(data_u8x32));
        __m256i hi_i16x16 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(data_u8x32, 1));
        sumsq_lo_i32x8 = _mm256_add_epi32(sumsq_lo_i32x8, _mm256_madd_epi16(lo_i16x16, lo_i16x16));
        sumsq_hi_i32x8 = _mm256_add_epi32(sumsq_hi_i32x8, _mm256_madd_epi16(hi_i16x16, hi_i16x16));
    }
    sumsq_lo_i32x8 = _mm256_add_epi32(sumsq_lo_i32x8, sumsq_hi_i32x8);
    __m256i sumsq_u64x4 = _mm256_cvtepu32_epi64(_mm256_castsi256_si128(sumsq_lo_i32x8));
    sumsq_u64x4 = _mm256_add_epi64(sumsq_u64x4, _mm256_cvtepu32_epi64(_mm256_extracti128_si256(sumsq_lo_i32x8, 1)));
    *sum = (nk_u64_t)nk_reduce_add_i64x4_haswell_(sum_u64x4);
    *sumsq = (nk_u64_t)nk_reduce_add_i64x4_haswell_(sumsq_u64x4);
}

NK_INTERNAL void nk_reduce_moments_u8_haswell_strided_(              //
    nk_u8_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_u64_t *sum, nk_u64_t *sumsq) {
    __m256i stride_mask_u8x32 = nk_stride_blend_u1x32_(stride_elements);
    __m256i zero_u8x32 = _mm256_setzero_si256();
    __m256i sum_u64x4 = _mm256_setzero_si256();
    __m256i sumsq_lo_i32x8 = _mm256_setzero_si256();
    __m256i sumsq_hi_i32x8 = _mm256_setzero_si256();
    nk_size_t idx_scalars = 0;
    nk_size_t total_scalars = count * stride_elements;
    for (; idx_scalars + 32 <= total_scalars; idx_scalars += 32) {
        __m256i data_u8x32 = _mm256_loadu_si256((__m256i const *)(data + idx_scalars));
        data_u8x32 = _mm256_and_si256(data_u8x32, stride_mask_u8x32);
        sum_u64x4 = _mm256_add_epi64(sum_u64x4, _mm256_sad_epu8(data_u8x32, zero_u8x32));
        __m256i lo_i16x16 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(data_u8x32));
        __m256i hi_i16x16 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(data_u8x32, 1));
        sumsq_lo_i32x8 = _mm256_add_epi32(sumsq_lo_i32x8, _mm256_madd_epi16(lo_i16x16, lo_i16x16));
        sumsq_hi_i32x8 = _mm256_add_epi32(sumsq_hi_i32x8, _mm256_madd_epi16(hi_i16x16, hi_i16x16));
    }
    sumsq_lo_i32x8 = _mm256_add_epi32(sumsq_lo_i32x8, sumsq_hi_i32x8);
    __m256i sumsq_u64x4 = _mm256_cvtepu32_epi64(_mm256_castsi256_si128(sumsq_lo_i32x8));
    sumsq_u64x4 = _mm256_add_epi64(sumsq_u64x4, _mm256_cvtepu32_epi64(_mm256_extracti128_si256(sumsq_lo_i32x8, 1)));
    nk_u64_t s = (nk_u64_t)nk_reduce_add_i64x4_haswell_(sum_u64x4);
    nk_u64_t sq = (nk_u64_t)nk_reduce_add_i64x4_haswell_(sumsq_u64x4);
    nk_u8_t const *ptr = data + idx_scalars;
    nk_size_t remaining_elements = count - idx_scalars / stride_elements;
    for (nk_size_t i = 0; i < remaining_elements; ++i, ptr += stride_elements) {
        nk_u64_t val = (nk_u64_t)*ptr;
        s += val;
        sq += val * val;
    }
    *sum = s;
    *sumsq = sq;
}

NK_PUBLIC void nk_reduce_moments_u8_haswell(                      //
    nk_u8_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *sum, nk_u64_t *sumsq) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_u8_t);
    int aligned = (stride_bytes % sizeof(nk_u8_t) == 0);
    if (count == 0) *sum = 0, *sumsq = 0;
    else if (!aligned) nk_reduce_moments_u8_serial(data, count, stride_bytes, sum, sumsq);
    else if (count > (nk_size_t)(NK_U16_MAX + 1) * 32) {
        nk_size_t left_count = count / 2;
        nk_u64_t left_sum, left_sumsq, right_sum, right_sumsq;
        nk_reduce_moments_u8_haswell(data, left_count, stride_bytes, &left_sum, &left_sumsq);
        nk_reduce_moments_u8_haswell(data + left_count * stride_elements, count - left_count, stride_bytes, &right_sum,
                                     &right_sumsq);
        nk_u64_sadd_(&left_sum, &right_sum, sum);
        nk_u64_sadd_(&left_sumsq, &right_sumsq, sumsq);
    }
    else if (stride_elements == 1) nk_reduce_moments_u8_haswell_contiguous_(data, count, sum, sumsq);
    else if (stride_elements <= 8) nk_reduce_moments_u8_haswell_strided_(data, count, stride_elements, sum, sumsq);
    else nk_reduce_moments_u8_serial(data, count, stride_bytes, sum, sumsq);
}

NK_INTERNAL void nk_reduce_minmax_u8_haswell_contiguous_( //
    nk_u8_t const *data, nk_size_t count,                 //
    nk_u8_t *min_value, nk_size_t *min_index,             //
    nk_u8_t *max_value, nk_size_t *max_index) {
    // XOR-bias to signed domain for _mm256_cmpgt_epi8
    __m256i bias_u8x32 = _mm256_set1_epi8((char)0x80);
    __m256i raw_u8x32 = _mm256_loadu_si256((__m256i const *)data);
    __m256i min_biased_i8x32 = _mm256_xor_si256(raw_u8x32, bias_u8x32);
    __m256i max_biased_i8x32 = min_biased_i8x32;
    __m256i min_iter_u8x32 = _mm256_setzero_si256();
    __m256i max_iter_u8x32 = _mm256_setzero_si256();
    __m256i iter_u8x32 = _mm256_set1_epi8(1);
    __m256i one_u8x32 = _mm256_set1_epi8(1);

    nk_size_t idx = 32;
    for (; idx + 32 <= count; idx += 32) {
        __m256i data_biased_i8x32 = _mm256_xor_si256(_mm256_loadu_si256((__m256i const *)(data + idx)), bias_u8x32);
        __m256i less_b8x32 = _mm256_cmpgt_epi8(min_biased_i8x32, data_biased_i8x32);
        __m256i greater_b8x32 = _mm256_cmpgt_epi8(data_biased_i8x32, max_biased_i8x32);
        min_biased_i8x32 = _mm256_blendv_epi8(min_biased_i8x32, data_biased_i8x32, less_b8x32);
        max_biased_i8x32 = _mm256_blendv_epi8(max_biased_i8x32, data_biased_i8x32, greater_b8x32);
        min_iter_u8x32 = _mm256_blendv_epi8(min_iter_u8x32, iter_u8x32, less_b8x32);
        max_iter_u8x32 = _mm256_blendv_epi8(max_iter_u8x32, iter_u8x32, greater_b8x32);
        iter_u8x32 = _mm256_add_epi8(iter_u8x32, one_u8x32);
    }

    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        nk_b256_vec_t tail_vec;
        nk_partial_load_b8x32_serial_(data + idx, &tail_vec, remaining);
        __m256i data_biased_i8x32 = _mm256_xor_si256(tail_vec.ymm, bias_u8x32);
        nk_b256_vec_t mask_vec;
        mask_vec.ymm = _mm256_setzero_si256();
        for (nk_size_t i = 0; i < remaining; ++i) mask_vec.u8s[i] = 0xFF;
        __m256i valid_b8x32 = mask_vec.ymm;
        // Biased identity: NK_U8_MAX ^ 0x80 = 0x7F for min, NK_U8_MIN ^ 0x80 = 0x80 for max
        __m256i data_min_i8x32 = _mm256_blendv_epi8(_mm256_set1_epi8(0x7F), data_biased_i8x32, valid_b8x32);
        __m256i data_max_i8x32 = _mm256_blendv_epi8(_mm256_set1_epi8((char)0x80), data_biased_i8x32, valid_b8x32);
        __m256i less_b8x32 = _mm256_cmpgt_epi8(min_biased_i8x32, data_min_i8x32);
        __m256i greater_b8x32 = _mm256_cmpgt_epi8(data_max_i8x32, max_biased_i8x32);
        min_biased_i8x32 = _mm256_blendv_epi8(min_biased_i8x32, data_min_i8x32, less_b8x32);
        max_biased_i8x32 = _mm256_blendv_epi8(max_biased_i8x32, data_max_i8x32, greater_b8x32);
        min_iter_u8x32 = _mm256_blendv_epi8(min_iter_u8x32, iter_u8x32, less_b8x32);
        max_iter_u8x32 = _mm256_blendv_epi8(max_iter_u8x32, iter_u8x32, greater_b8x32);
    }

    // Un-bias to get real u8 values
    __m256i min_u8x32 = _mm256_xor_si256(min_biased_i8x32, bias_u8x32);
    __m256i max_u8x32 = _mm256_xor_si256(max_biased_i8x32, bias_u8x32);
    nk_u8_t min_scalar = nk_reduce_min_u8x32_haswell_(min_u8x32);
    nk_u8_t max_scalar = nk_reduce_max_u8x32_haswell_(max_u8x32);
    __m256i min_eq_b8x32 = _mm256_cmpeq_epi8(min_u8x32, _mm256_set1_epi8((char)min_scalar));
    __m256i max_eq_b8x32 = _mm256_cmpeq_epi8(max_u8x32, _mm256_set1_epi8((char)max_scalar));
    unsigned int min_lane = _tzcnt_u32((unsigned int)_mm256_movemask_epi8(min_eq_b8x32));
    unsigned int max_lane = _tzcnt_u32((unsigned int)_mm256_movemask_epi8(max_eq_b8x32));
    nk_b256_vec_t iter_vec;
    iter_vec.ymm = min_iter_u8x32;
    *min_value = min_scalar, *min_index = (nk_size_t)iter_vec.u8s[min_lane] * 32 + min_lane;
    iter_vec.ymm = max_iter_u8x32;
    *max_value = max_scalar, *max_index = (nk_size_t)iter_vec.u8s[max_lane] * 32 + max_lane;
}

NK_PUBLIC void nk_reduce_minmax_u8_haswell(                       //
    nk_u8_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u8_t *min_value, nk_size_t *min_index,                     //
    nk_u8_t *max_value, nk_size_t *max_index) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_u8_t);
    int aligned = (stride_bytes % sizeof(nk_u8_t) == 0);
    if (count == 0) *min_value = NK_U8_MAX, *min_index = NK_SIZE_MAX, *max_value = 0, *max_index = NK_SIZE_MAX;
    else if (!aligned)
        nk_reduce_minmax_u8_serial(data, count, stride_bytes, min_value, min_index, max_value, max_index);
    else if (count > (nk_size_t)(NK_U8_MAX + 1) * 32) {
        nk_size_t left_count = count / 2;
        nk_u8_t left_min, right_min, left_max, right_max;
        nk_size_t left_min_index, right_min_index, left_max_index, right_max_index;
        nk_reduce_minmax_u8_haswell(data, left_count, stride_bytes, &left_min, &left_min_index, &left_max,
                                    &left_max_index);
        nk_reduce_minmax_u8_haswell(data + left_count * stride_elements, count - left_count, stride_bytes, &right_min,
                                    &right_min_index, &right_max, &right_max_index);
        if (right_min < left_min) *min_value = right_min, *min_index = left_count + right_min_index;
        else *min_value = left_min, *min_index = left_min_index;
        if (right_max > left_max) *max_value = right_max, *max_index = left_count + right_max_index;
        else *max_value = left_max, *max_index = left_max_index;
    }
    else if (stride_elements == 1)
        nk_reduce_minmax_u8_haswell_contiguous_(data, count, min_value, min_index, max_value, max_index);
    else nk_reduce_minmax_u8_serial(data, count, stride_bytes, min_value, min_index, max_value, max_index);
}

NK_INTERNAL void nk_reduce_moments_i16_haswell_contiguous_( //
    nk_i16_t const *data, nk_size_t count,                  //
    nk_i64_t *sum, nk_u64_t *sumsq) {
    __m256i ones_i16x16 = _mm256_set1_epi16(1);
    __m256i sum_i32x8 = _mm256_setzero_si256();
    __m256i sumsq_i64x4 = _mm256_setzero_si256();
    nk_size_t idx = 0;
    for (; idx + 16 <= count; idx += 16) {
        __m256i data_i16x16 = _mm256_loadu_si256((__m256i const *)(data + idx));
        sum_i32x8 = _mm256_add_epi32(sum_i32x8, _mm256_madd_epi16(data_i16x16, ones_i16x16));
        __m256i sq_i32x8 = _mm256_madd_epi16(data_i16x16, data_i16x16);
        sumsq_i64x4 = _mm256_add_epi64(sumsq_i64x4, _mm256_cvtepi32_epi64(_mm256_castsi256_si128(sq_i32x8)));
        sumsq_i64x4 = _mm256_add_epi64(sumsq_i64x4, _mm256_cvtepi32_epi64(_mm256_extracti128_si256(sq_i32x8, 1)));
    }
    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        nk_b256_vec_t tail_vec;
        nk_partial_load_b16x16_serial_(data + idx, &tail_vec, remaining);
        __m256i data_i16x16 = tail_vec.ymm;
        sum_i32x8 = _mm256_add_epi32(sum_i32x8, _mm256_madd_epi16(data_i16x16, ones_i16x16));
        __m256i sq_i32x8 = _mm256_madd_epi16(data_i16x16, data_i16x16);
        sumsq_i64x4 = _mm256_add_epi64(sumsq_i64x4, _mm256_cvtepi32_epi64(_mm256_castsi256_si128(sq_i32x8)));
        sumsq_i64x4 = _mm256_add_epi64(sumsq_i64x4, _mm256_cvtepi32_epi64(_mm256_extracti128_si256(sq_i32x8, 1)));
    }
    __m256i sum_i64x4 = _mm256_add_epi64(                               //
        _mm256_cvtepi32_epi64(_mm256_castsi256_si128(sum_i32x8)),       //
        _mm256_cvtepi32_epi64(_mm256_extracti128_si256(sum_i32x8, 1))); //
    *sum = nk_reduce_add_i64x4_haswell_(sum_i64x4);
    *sumsq = (nk_u64_t)nk_reduce_add_i64x4_haswell_(sumsq_i64x4);
}

NK_INTERNAL void nk_reduce_moments_i16_haswell_strided_(              //
    nk_i16_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_i64_t *sum, nk_u64_t *sumsq) {
    __m256i stride_mask_i16x16 = nk_stride_blend_b16x16_(stride_elements);
    __m256i ones_i16x16 = _mm256_set1_epi16(1);
    __m256i sum_i32x8 = _mm256_setzero_si256();
    __m256i sumsq_i64x4 = _mm256_setzero_si256();
    nk_size_t idx_scalars = 0;
    nk_size_t total_scalars = count * stride_elements;
    for (; idx_scalars + 16 <= total_scalars; idx_scalars += 16) {
        __m256i data_i16x16 = _mm256_loadu_si256((__m256i const *)(data + idx_scalars));
        data_i16x16 = _mm256_and_si256(data_i16x16, stride_mask_i16x16);
        sum_i32x8 = _mm256_add_epi32(sum_i32x8, _mm256_madd_epi16(data_i16x16, ones_i16x16));
        __m256i sq_i32x8 = _mm256_madd_epi16(data_i16x16, data_i16x16);
        sumsq_i64x4 = _mm256_add_epi64(sumsq_i64x4, _mm256_cvtepi32_epi64(_mm256_castsi256_si128(sq_i32x8)));
        sumsq_i64x4 = _mm256_add_epi64(sumsq_i64x4, _mm256_cvtepi32_epi64(_mm256_extracti128_si256(sq_i32x8, 1)));
    }
    __m256i sum_i64x4 = _mm256_add_epi64(                               //
        _mm256_cvtepi32_epi64(_mm256_castsi256_si128(sum_i32x8)),       //
        _mm256_cvtepi32_epi64(_mm256_extracti128_si256(sum_i32x8, 1))); //
    nk_i64_t s = nk_reduce_add_i64x4_haswell_(sum_i64x4);
    nk_u64_t sq = (nk_u64_t)nk_reduce_add_i64x4_haswell_(sumsq_i64x4);
    nk_i16_t const *ptr = data + idx_scalars;
    nk_size_t remaining = count - idx_scalars / stride_elements;
    for (nk_size_t i = 0; i < remaining; ++i, ptr += stride_elements) {
        nk_i64_t val = (nk_i64_t)*ptr;
        s += val;
        sq += (nk_u64_t)(val * val);
    }
    *sum = s;
    *sumsq = sq;
}

NK_PUBLIC void nk_reduce_moments_i16_haswell(                      //
    nk_i16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *sum, nk_u64_t *sumsq) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_i16_t);
    int aligned = (stride_bytes % sizeof(nk_i16_t) == 0);
    if (count == 0) *sum = 0, *sumsq = 0;
    else if (!aligned) nk_reduce_moments_i16_serial(data, count, stride_bytes, sum, sumsq);
    else if (count > (nk_size_t)(NK_U16_MAX + 1) * 16) {
        nk_size_t left_count = count / 2;
        nk_i64_t left_sum, right_sum;
        nk_u64_t left_sumsq, right_sumsq;
        nk_reduce_moments_i16_haswell(data, left_count, stride_bytes, &left_sum, &left_sumsq);
        nk_reduce_moments_i16_haswell(data + left_count * stride_elements, count - left_count, stride_bytes, &right_sum,
                                      &right_sumsq);
        nk_i64_sadd_(&left_sum, &right_sum, sum);
        nk_u64_sadd_(&left_sumsq, &right_sumsq, sumsq);
    }
    else if (stride_elements == 1) nk_reduce_moments_i16_haswell_contiguous_(data, count, sum, sumsq);
    else if (stride_elements <= 8) nk_reduce_moments_i16_haswell_strided_(data, count, stride_elements, sum, sumsq);
    else nk_reduce_moments_i16_serial(data, count, stride_bytes, sum, sumsq);
}

NK_INTERNAL void nk_reduce_minmax_i16_haswell_contiguous_( //
    nk_i16_t const *data, nk_size_t count,                 //
    nk_i16_t *min_value, nk_size_t *min_index,             //
    nk_i16_t *max_value, nk_size_t *max_index) {
    __m256i min_i16x16 = _mm256_loadu_si256((__m256i const *)data);
    __m256i max_i16x16 = min_i16x16;
    __m256i min_iter_u16x16 = _mm256_setzero_si256();
    __m256i max_iter_u16x16 = _mm256_setzero_si256();
    __m256i iter_u16x16 = _mm256_set1_epi16(1);
    __m256i one_u16x16 = _mm256_set1_epi16(1);

    nk_size_t idx = 16;
    for (; idx + 16 <= count; idx += 16) {
        __m256i data_i16x16 = _mm256_loadu_si256((__m256i const *)(data + idx));
        __m256i less_b16x16 = _mm256_cmpgt_epi16(min_i16x16, data_i16x16);
        __m256i greater_b16x16 = _mm256_cmpgt_epi16(data_i16x16, max_i16x16);
        min_i16x16 = _mm256_blendv_epi8(min_i16x16, data_i16x16, less_b16x16);
        max_i16x16 = _mm256_blendv_epi8(max_i16x16, data_i16x16, greater_b16x16);
        min_iter_u16x16 = _mm256_blendv_epi8(min_iter_u16x16, iter_u16x16, less_b16x16);
        max_iter_u16x16 = _mm256_blendv_epi8(max_iter_u16x16, iter_u16x16, greater_b16x16);
        iter_u16x16 = _mm256_add_epi16(iter_u16x16, one_u16x16);
    }

    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        nk_b256_vec_t tail_vec;
        nk_partial_load_b16x16_serial_(data + idx, &tail_vec, remaining);
        // Build 16-bit lane mask
        nk_b256_vec_t mask_vec;
        mask_vec.ymm = _mm256_setzero_si256();
        for (nk_size_t i = 0; i < remaining; ++i) mask_vec.u16s[i] = 0xFFFF;
        __m256i valid_b16x16 = mask_vec.ymm;
        __m256i data_min_i16x16 = _mm256_blendv_epi8(_mm256_set1_epi16(NK_I16_MAX), tail_vec.ymm, valid_b16x16);
        __m256i data_max_i16x16 = _mm256_blendv_epi8(_mm256_set1_epi16(NK_I16_MIN), tail_vec.ymm, valid_b16x16);
        __m256i less_b16x16 = _mm256_cmpgt_epi16(min_i16x16, data_min_i16x16);
        __m256i greater_b16x16 = _mm256_cmpgt_epi16(data_max_i16x16, max_i16x16);
        min_i16x16 = _mm256_blendv_epi8(min_i16x16, data_min_i16x16, less_b16x16);
        max_i16x16 = _mm256_blendv_epi8(max_i16x16, data_max_i16x16, greater_b16x16);
        min_iter_u16x16 = _mm256_blendv_epi8(min_iter_u16x16, iter_u16x16, less_b16x16);
        max_iter_u16x16 = _mm256_blendv_epi8(max_iter_u16x16, iter_u16x16, greater_b16x16);
    }

    nk_i16_t min_scalar = nk_reduce_min_i16x16_haswell_(min_i16x16);
    nk_i16_t max_scalar = nk_reduce_max_i16x16_haswell_(max_i16x16);
    // movemask_epi8 gives 32 bits for 16 i16 lanes — take every other bit
    __m256i min_eq_b16x16 = _mm256_cmpeq_epi16(min_i16x16, _mm256_set1_epi16(min_scalar));
    __m256i max_eq_b16x16 = _mm256_cmpeq_epi16(max_i16x16, _mm256_set1_epi16(max_scalar));
    unsigned int min_mask_bits = (unsigned int)_mm256_movemask_epi8(min_eq_b16x16);
    unsigned int max_mask_bits = (unsigned int)_mm256_movemask_epi8(max_eq_b16x16);
    unsigned int min_lane = _tzcnt_u32(min_mask_bits) / 2;
    unsigned int max_lane = _tzcnt_u32(max_mask_bits) / 2;
    nk_b256_vec_t iter_vec;
    iter_vec.ymm = min_iter_u16x16;
    *min_value = min_scalar, *min_index = (nk_size_t)iter_vec.u16s[min_lane] * 16 + min_lane;
    iter_vec.ymm = max_iter_u16x16;
    *max_value = max_scalar, *max_index = (nk_size_t)iter_vec.u16s[max_lane] * 16 + max_lane;
}

NK_PUBLIC void nk_reduce_minmax_i16_haswell(                       //
    nk_i16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i16_t *min_value, nk_size_t *min_index,                     //
    nk_i16_t *max_value, nk_size_t *max_index) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_i16_t);
    int aligned = (stride_bytes % sizeof(nk_i16_t) == 0);
    if (count == 0)
        *min_value = NK_I16_MAX, *min_index = NK_SIZE_MAX, *max_value = NK_I16_MIN, *max_index = NK_SIZE_MAX;
    else if (!aligned)
        nk_reduce_minmax_i16_serial(data, count, stride_bytes, min_value, min_index, max_value, max_index);
    else if (count > (nk_size_t)(NK_U16_MAX + 1) * 16) {
        nk_size_t left_count = count / 2;
        nk_i16_t left_min, right_min, left_max, right_max;
        nk_size_t left_min_index, right_min_index, left_max_index, right_max_index;
        nk_reduce_minmax_i16_haswell(data, left_count, stride_bytes, &left_min, &left_min_index, &left_max,
                                     &left_max_index);
        nk_reduce_minmax_i16_haswell(data + left_count * stride_elements, count - left_count, stride_bytes, &right_min,
                                     &right_min_index, &right_max, &right_max_index);
        if (right_min < left_min) *min_value = right_min, *min_index = left_count + right_min_index;
        else *min_value = left_min, *min_index = left_min_index;
        if (right_max > left_max) *max_value = right_max, *max_index = left_count + right_max_index;
        else *max_value = left_max, *max_index = left_max_index;
    }
    else if (stride_elements == 1)
        nk_reduce_minmax_i16_haswell_contiguous_(data, count, min_value, min_index, max_value, max_index);
    else nk_reduce_minmax_i16_serial(data, count, stride_bytes, min_value, min_index, max_value, max_index);
}

NK_INTERNAL void nk_reduce_moments_u16_haswell_contiguous_( //
    nk_u16_t const *data, nk_size_t count,                  //
    nk_u64_t *sum, nk_u64_t *sumsq) {
    // Widen u16→u32, square in u32, widen to u64.
    __m256i sum_u32x8 = _mm256_setzero_si256();
    __m256i sumsq_u64x4 = _mm256_setzero_si256();
    nk_size_t idx = 0;
    for (; idx + 8 <= count; idx += 8) {
        __m256i data_u32x8 = _mm256_cvtepu16_epi32(_mm_loadu_si128((__m128i const *)(data + idx)));
        sum_u32x8 = _mm256_add_epi32(sum_u32x8, data_u32x8);
        __m256i sq_u32x8 = _mm256_mullo_epi32(data_u32x8, data_u32x8);
        sumsq_u64x4 = _mm256_add_epi64(sumsq_u64x4, _mm256_cvtepu32_epi64(_mm256_castsi256_si128(sq_u32x8)));
        sumsq_u64x4 = _mm256_add_epi64(sumsq_u64x4, _mm256_cvtepu32_epi64(_mm256_extracti128_si256(sq_u32x8, 1)));
    }
    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        nk_b256_vec_t tail_vec;
        nk_partial_load_b16x16_serial_(data + idx, &tail_vec, remaining);
        __m256i data_u32x8 = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(tail_vec.ymm));
        sum_u32x8 = _mm256_add_epi32(sum_u32x8, data_u32x8);
        __m256i sq_u32x8 = _mm256_mullo_epi32(data_u32x8, data_u32x8);
        sumsq_u64x4 = _mm256_add_epi64(sumsq_u64x4, _mm256_cvtepu32_epi64(_mm256_castsi256_si128(sq_u32x8)));
        sumsq_u64x4 = _mm256_add_epi64(sumsq_u64x4, _mm256_cvtepu32_epi64(_mm256_extracti128_si256(sq_u32x8, 1)));
    }
    __m256i sum_u64x4 = _mm256_add_epi64(                               //
        _mm256_cvtepu32_epi64(_mm256_castsi256_si128(sum_u32x8)),       //
        _mm256_cvtepu32_epi64(_mm256_extracti128_si256(sum_u32x8, 1))); //
    *sum = (nk_u64_t)nk_reduce_add_i64x4_haswell_(sum_u64x4);
    *sumsq = (nk_u64_t)nk_reduce_add_i64x4_haswell_(sumsq_u64x4);
}

NK_INTERNAL void nk_reduce_moments_u16_haswell_strided_(              //
    nk_u16_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_u64_t *sum, nk_u64_t *sumsq) {
    __m256i stride_mask_i16x16 = nk_stride_blend_b16x16_(stride_elements);
    __m256i sum_u32x8 = _mm256_setzero_si256();
    __m256i sumsq_u64x4 = _mm256_setzero_si256();
    nk_size_t idx_scalars = 0;
    nk_size_t total_scalars = count * stride_elements;
    for (; idx_scalars + 16 <= total_scalars; idx_scalars += 16) {
        __m256i data_u16x16 = _mm256_loadu_si256((__m256i const *)(data + idx_scalars));
        data_u16x16 = _mm256_and_si256(data_u16x16, stride_mask_i16x16);
        __m256i lo_u32x8 = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(data_u16x16));
        __m256i hi_u32x8 = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(data_u16x16, 1));
        sum_u32x8 = _mm256_add_epi32(sum_u32x8, lo_u32x8);
        sum_u32x8 = _mm256_add_epi32(sum_u32x8, hi_u32x8);
        __m256i lo_sq_u32x8 = _mm256_mullo_epi32(lo_u32x8, lo_u32x8);
        __m256i hi_sq_u32x8 = _mm256_mullo_epi32(hi_u32x8, hi_u32x8);
        sumsq_u64x4 = _mm256_add_epi64(sumsq_u64x4, _mm256_cvtepu32_epi64(_mm256_castsi256_si128(lo_sq_u32x8)));
        sumsq_u64x4 = _mm256_add_epi64(sumsq_u64x4, _mm256_cvtepu32_epi64(_mm256_extracti128_si256(lo_sq_u32x8, 1)));
        sumsq_u64x4 = _mm256_add_epi64(sumsq_u64x4, _mm256_cvtepu32_epi64(_mm256_castsi256_si128(hi_sq_u32x8)));
        sumsq_u64x4 = _mm256_add_epi64(sumsq_u64x4, _mm256_cvtepu32_epi64(_mm256_extracti128_si256(hi_sq_u32x8, 1)));
    }
    __m256i sum_u64x4 = _mm256_add_epi64(                               //
        _mm256_cvtepu32_epi64(_mm256_castsi256_si128(sum_u32x8)),       //
        _mm256_cvtepu32_epi64(_mm256_extracti128_si256(sum_u32x8, 1))); //
    nk_u64_t s = (nk_u64_t)nk_reduce_add_i64x4_haswell_(sum_u64x4);
    nk_u64_t sq = (nk_u64_t)nk_reduce_add_i64x4_haswell_(sumsq_u64x4);
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

NK_PUBLIC void nk_reduce_moments_u16_haswell(                      //
    nk_u16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *sum, nk_u64_t *sumsq) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_u16_t);
    int aligned = (stride_bytes % sizeof(nk_u16_t) == 0);
    if (count == 0) *sum = 0, *sumsq = 0;
    else if (!aligned) nk_reduce_moments_u16_serial(data, count, stride_bytes, sum, sumsq);
    else if (count > (nk_size_t)(NK_U16_MAX + 1) * 8) {
        nk_size_t left_count = count / 2;
        nk_u64_t left_sum, left_sumsq, right_sum, right_sumsq;
        nk_reduce_moments_u16_haswell(data, left_count, stride_bytes, &left_sum, &left_sumsq);
        nk_reduce_moments_u16_haswell(data + left_count * stride_elements, count - left_count, stride_bytes, &right_sum,
                                      &right_sumsq);
        nk_u64_sadd_(&left_sum, &right_sum, sum);
        nk_u64_sadd_(&left_sumsq, &right_sumsq, sumsq);
    }
    else if (stride_elements == 1) nk_reduce_moments_u16_haswell_contiguous_(data, count, sum, sumsq);
    else if (stride_elements <= 8) nk_reduce_moments_u16_haswell_strided_(data, count, stride_elements, sum, sumsq);
    else nk_reduce_moments_u16_serial(data, count, stride_bytes, sum, sumsq);
}

NK_INTERNAL void nk_reduce_minmax_u16_haswell_contiguous_( //
    nk_u16_t const *data, nk_size_t count,                 //
    nk_u16_t *min_value, nk_size_t *min_index,             //
    nk_u16_t *max_value, nk_size_t *max_index) {
    // XOR-bias to signed domain for _mm256_cmpgt_epi16
    __m256i bias_u16x16 = _mm256_set1_epi16((short)0x8000);
    __m256i raw_u16x16 = _mm256_loadu_si256((__m256i const *)data);
    __m256i min_biased_i16x16 = _mm256_xor_si256(raw_u16x16, bias_u16x16);
    __m256i max_biased_i16x16 = min_biased_i16x16;
    __m256i min_iter_u16x16 = _mm256_setzero_si256();
    __m256i max_iter_u16x16 = _mm256_setzero_si256();
    __m256i iter_u16x16 = _mm256_set1_epi16(1);
    __m256i one_u16x16 = _mm256_set1_epi16(1);

    nk_size_t idx = 16;
    for (; idx + 16 <= count; idx += 16) {
        __m256i data_biased_i16x16 = _mm256_xor_si256(_mm256_loadu_si256((__m256i const *)(data + idx)), bias_u16x16);
        __m256i less_b16x16 = _mm256_cmpgt_epi16(min_biased_i16x16, data_biased_i16x16);
        __m256i greater_b16x16 = _mm256_cmpgt_epi16(data_biased_i16x16, max_biased_i16x16);
        min_biased_i16x16 = _mm256_blendv_epi8(min_biased_i16x16, data_biased_i16x16, less_b16x16);
        max_biased_i16x16 = _mm256_blendv_epi8(max_biased_i16x16, data_biased_i16x16, greater_b16x16);
        min_iter_u16x16 = _mm256_blendv_epi8(min_iter_u16x16, iter_u16x16, less_b16x16);
        max_iter_u16x16 = _mm256_blendv_epi8(max_iter_u16x16, iter_u16x16, greater_b16x16);
        iter_u16x16 = _mm256_add_epi16(iter_u16x16, one_u16x16);
    }

    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        nk_b256_vec_t tail_vec;
        nk_partial_load_b16x16_serial_(data + idx, &tail_vec, remaining);
        __m256i data_biased_i16x16 = _mm256_xor_si256(tail_vec.ymm, bias_u16x16);
        nk_b256_vec_t mask_vec;
        mask_vec.ymm = _mm256_setzero_si256();
        for (nk_size_t i = 0; i < remaining; ++i) mask_vec.u16s[i] = 0xFFFF;
        __m256i valid_b16x16 = mask_vec.ymm;
        // Biased identity: NK_U16_MAX ^ 0x8000 = 0x7FFF for min, 0 ^ 0x8000 = 0x8000 for max
        __m256i data_min_i16x16 = _mm256_blendv_epi8(_mm256_set1_epi16(0x7FFF), data_biased_i16x16, valid_b16x16);
        __m256i data_max_i16x16 = _mm256_blendv_epi8(_mm256_set1_epi16((short)0x8000), data_biased_i16x16,
                                                     valid_b16x16);
        __m256i less_b16x16 = _mm256_cmpgt_epi16(min_biased_i16x16, data_min_i16x16);
        __m256i greater_b16x16 = _mm256_cmpgt_epi16(data_max_i16x16, max_biased_i16x16);
        min_biased_i16x16 = _mm256_blendv_epi8(min_biased_i16x16, data_min_i16x16, less_b16x16);
        max_biased_i16x16 = _mm256_blendv_epi8(max_biased_i16x16, data_max_i16x16, greater_b16x16);
        min_iter_u16x16 = _mm256_blendv_epi8(min_iter_u16x16, iter_u16x16, less_b16x16);
        max_iter_u16x16 = _mm256_blendv_epi8(max_iter_u16x16, iter_u16x16, greater_b16x16);
    }

    __m256i min_u16x16 = _mm256_xor_si256(min_biased_i16x16, bias_u16x16);
    __m256i max_u16x16 = _mm256_xor_si256(max_biased_i16x16, bias_u16x16);
    nk_u16_t min_scalar = nk_reduce_min_u16x16_haswell_(min_u16x16);
    nk_u16_t max_scalar = nk_reduce_max_u16x16_haswell_(max_u16x16);
    __m256i min_eq_b16x16 = _mm256_cmpeq_epi16(min_u16x16, _mm256_set1_epi16((short)min_scalar));
    __m256i max_eq_b16x16 = _mm256_cmpeq_epi16(max_u16x16, _mm256_set1_epi16((short)max_scalar));
    unsigned int min_mask_bits = (unsigned int)_mm256_movemask_epi8(min_eq_b16x16);
    unsigned int max_mask_bits = (unsigned int)_mm256_movemask_epi8(max_eq_b16x16);
    unsigned int min_lane = _tzcnt_u32(min_mask_bits) / 2;
    unsigned int max_lane = _tzcnt_u32(max_mask_bits) / 2;
    nk_b256_vec_t iter_vec;
    iter_vec.ymm = min_iter_u16x16;
    *min_value = min_scalar, *min_index = (nk_size_t)iter_vec.u16s[min_lane] * 16 + min_lane;
    iter_vec.ymm = max_iter_u16x16;
    *max_value = max_scalar, *max_index = (nk_size_t)iter_vec.u16s[max_lane] * 16 + max_lane;
}

NK_PUBLIC void nk_reduce_minmax_u16_haswell(                       //
    nk_u16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u16_t *min_value, nk_size_t *min_index,                     //
    nk_u16_t *max_value, nk_size_t *max_index) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_u16_t);
    int aligned = (stride_bytes % sizeof(nk_u16_t) == 0);
    if (count == 0) *min_value = NK_U16_MAX, *min_index = NK_SIZE_MAX, *max_value = 0, *max_index = NK_SIZE_MAX;
    else if (!aligned)
        nk_reduce_minmax_u16_serial(data, count, stride_bytes, min_value, min_index, max_value, max_index);
    else if (count > (nk_size_t)(NK_U16_MAX + 1) * 16) {
        nk_size_t left_count = count / 2;
        nk_u16_t left_min, right_min, left_max, right_max;
        nk_size_t left_min_index, right_min_index, left_max_index, right_max_index;
        nk_reduce_minmax_u16_haswell(data, left_count, stride_bytes, &left_min, &left_min_index, &left_max,
                                     &left_max_index);
        nk_reduce_minmax_u16_haswell(data + left_count * stride_elements, count - left_count, stride_bytes, &right_min,
                                     &right_min_index, &right_max, &right_max_index);
        if (right_min < left_min) *min_value = right_min, *min_index = left_count + right_min_index;
        else *min_value = left_min, *min_index = left_min_index;
        if (right_max > left_max) *max_value = right_max, *max_index = left_count + right_max_index;
        else *max_value = left_max, *max_index = left_max_index;
    }
    else if (stride_elements == 1)
        nk_reduce_minmax_u16_haswell_contiguous_(data, count, min_value, min_index, max_value, max_index);
    else nk_reduce_minmax_u16_serial(data, count, stride_bytes, min_value, min_index, max_value, max_index);
}

NK_INTERNAL void nk_reduce_moments_i32_haswell_contiguous_( //
    nk_i32_t const *data, nk_size_t count,                  //
    nk_i64_t *sum, nk_u64_t *sumsq) {
    __m256i sum_lower_i64x4 = _mm256_setzero_si256();
    __m256i sum_upper_i64x4 = _mm256_setzero_si256();
    __m256i sumsq_u64x4 = _mm256_setzero_si256();
    int sumsq_overflow_mask = 0;
    __m256i sign_bit_i64x4 = _mm256_set1_epi64x((nk_i64_t)0x8000000000000000ULL);
    nk_size_t idx = 0;
    for (; idx + 8 <= count; idx += 8) {
        __m256i data_i32x8 = _mm256_loadu_si256((__m256i const *)(data + idx));
        // 128-bit sum: lo half
        __m256i widened_lo_i64x4 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(data_i32x8));
        __m256i sum_before_i64x4 = sum_lower_i64x4;
        sum_lower_i64x4 = _mm256_add_epi64(sum_lower_i64x4, widened_lo_i64x4);
        __m256i result_biased_i64x4 = _mm256_xor_si256(sum_lower_i64x4, sign_bit_i64x4);
        __m256i before_biased_i64x4 = _mm256_xor_si256(sum_before_i64x4, sign_bit_i64x4);
        __m256i carry_mask_i64x4 = _mm256_cmpgt_epi64(before_biased_i64x4, result_biased_i64x4);
        sum_upper_i64x4 = _mm256_sub_epi64(sum_upper_i64x4, carry_mask_i64x4);
        __m256i sign_ext_i64x4 = _mm256_cmpgt_epi64(_mm256_setzero_si256(), widened_lo_i64x4);
        sum_upper_i64x4 = _mm256_add_epi64(sum_upper_i64x4, sign_ext_i64x4);
        // 128-bit sum: hi half
        __m256i widened_hi_i64x4 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(data_i32x8, 1));
        sum_before_i64x4 = sum_lower_i64x4;
        sum_lower_i64x4 = _mm256_add_epi64(sum_lower_i64x4, widened_hi_i64x4);
        result_biased_i64x4 = _mm256_xor_si256(sum_lower_i64x4, sign_bit_i64x4);
        before_biased_i64x4 = _mm256_xor_si256(sum_before_i64x4, sign_bit_i64x4);
        carry_mask_i64x4 = _mm256_cmpgt_epi64(before_biased_i64x4, result_biased_i64x4);
        sum_upper_i64x4 = _mm256_sub_epi64(sum_upper_i64x4, carry_mask_i64x4);
        sign_ext_i64x4 = _mm256_cmpgt_epi64(_mm256_setzero_si256(), widened_hi_i64x4);
        sum_upper_i64x4 = _mm256_add_epi64(sum_upper_i64x4, sign_ext_i64x4);
        // Sumsq: running mask + wrapping add with unsigned carry detection
        __m256i even_sq_u64x4 = _mm256_mul_epi32(data_i32x8, data_i32x8);
        __m256i odd_i32x8 = _mm256_srli_epi64(data_i32x8, 32);
        __m256i odd_sq_u64x4 = _mm256_mul_epi32(odd_i32x8, odd_i32x8);
        __m256i sumsq_before_u64x4 = sumsq_u64x4;
        sumsq_u64x4 = _mm256_add_epi64(sumsq_u64x4, even_sq_u64x4);
        __m256i sq_result_biased_u64x4 = _mm256_xor_si256(sumsq_u64x4, sign_bit_i64x4);
        __m256i sq_before_biased_u64x4 = _mm256_xor_si256(sumsq_before_u64x4, sign_bit_i64x4);
        sumsq_overflow_mask |= _mm256_movemask_pd(
            _mm256_castsi256_pd(_mm256_cmpgt_epi64(sq_before_biased_u64x4, sq_result_biased_u64x4)));
        sumsq_before_u64x4 = sumsq_u64x4;
        sumsq_u64x4 = _mm256_add_epi64(sumsq_u64x4, odd_sq_u64x4);
        sq_result_biased_u64x4 = _mm256_xor_si256(sumsq_u64x4, sign_bit_i64x4);
        sq_before_biased_u64x4 = _mm256_xor_si256(sumsq_before_u64x4, sign_bit_i64x4);
        sumsq_overflow_mask |= _mm256_movemask_pd(
            _mm256_castsi256_pd(_mm256_cmpgt_epi64(sq_before_biased_u64x4, sq_result_biased_u64x4)));
    }
    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        nk_b256_vec_t tail_vec;
        nk_partial_load_b32x8_serial_(data + idx, &tail_vec, remaining);
        __m256i data_i32x8 = tail_vec.ymm;
        __m256i widened_lo_i64x4 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(data_i32x8));
        __m256i sum_before_i64x4 = sum_lower_i64x4;
        sum_lower_i64x4 = _mm256_add_epi64(sum_lower_i64x4, widened_lo_i64x4);
        __m256i result_biased_i64x4 = _mm256_xor_si256(sum_lower_i64x4, sign_bit_i64x4);
        __m256i before_biased_i64x4 = _mm256_xor_si256(sum_before_i64x4, sign_bit_i64x4);
        __m256i carry_mask_i64x4 = _mm256_cmpgt_epi64(before_biased_i64x4, result_biased_i64x4);
        sum_upper_i64x4 = _mm256_sub_epi64(sum_upper_i64x4, carry_mask_i64x4);
        __m256i sign_ext_i64x4 = _mm256_cmpgt_epi64(_mm256_setzero_si256(), widened_lo_i64x4);
        sum_upper_i64x4 = _mm256_add_epi64(sum_upper_i64x4, sign_ext_i64x4);
        __m256i widened_hi_i64x4 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(data_i32x8, 1));
        sum_before_i64x4 = sum_lower_i64x4;
        sum_lower_i64x4 = _mm256_add_epi64(sum_lower_i64x4, widened_hi_i64x4);
        result_biased_i64x4 = _mm256_xor_si256(sum_lower_i64x4, sign_bit_i64x4);
        before_biased_i64x4 = _mm256_xor_si256(sum_before_i64x4, sign_bit_i64x4);
        carry_mask_i64x4 = _mm256_cmpgt_epi64(before_biased_i64x4, result_biased_i64x4);
        sum_upper_i64x4 = _mm256_sub_epi64(sum_upper_i64x4, carry_mask_i64x4);
        sign_ext_i64x4 = _mm256_cmpgt_epi64(_mm256_setzero_si256(), widened_hi_i64x4);
        sum_upper_i64x4 = _mm256_add_epi64(sum_upper_i64x4, sign_ext_i64x4);
        __m256i even_sq_u64x4 = _mm256_mul_epi32(data_i32x8, data_i32x8);
        __m256i odd_i32x8 = _mm256_srli_epi64(data_i32x8, 32);
        __m256i odd_sq_u64x4 = _mm256_mul_epi32(odd_i32x8, odd_i32x8);
        __m256i sumsq_before_u64x4 = sumsq_u64x4;
        sumsq_u64x4 = _mm256_add_epi64(sumsq_u64x4, even_sq_u64x4);
        __m256i sq_result_biased_u64x4 = _mm256_xor_si256(sumsq_u64x4, sign_bit_i64x4);
        __m256i sq_before_biased_u64x4 = _mm256_xor_si256(sumsq_before_u64x4, sign_bit_i64x4);
        sumsq_overflow_mask |= _mm256_movemask_pd(
            _mm256_castsi256_pd(_mm256_cmpgt_epi64(sq_before_biased_u64x4, sq_result_biased_u64x4)));
        sumsq_before_u64x4 = sumsq_u64x4;
        sumsq_u64x4 = _mm256_add_epi64(sumsq_u64x4, odd_sq_u64x4);
        sq_result_biased_u64x4 = _mm256_xor_si256(sumsq_u64x4, sign_bit_i64x4);
        sq_before_biased_u64x4 = _mm256_xor_si256(sumsq_before_u64x4, sign_bit_i64x4);
        sumsq_overflow_mask |= _mm256_movemask_pd(
            _mm256_castsi256_pd(_mm256_cmpgt_epi64(sq_before_biased_u64x4, sq_result_biased_u64x4)));
    }
    // Sumsq: horizontal unsigned saturating reduction
    nk_u64_t sq;
    if (sumsq_overflow_mask) sq = NK_U64_MAX;
    else sq = nk_reduce_sadd_u64x4_haswell_(sumsq_u64x4);
    // Sum: horizontal 128-bit reduction (4 lanes → scalar)
    nk_b256_vec_t lower_vec, upper_vec;
    lower_vec.ymm = sum_lower_i64x4;
    upper_vec.ymm = sum_upper_i64x4;
    nk_u64_t s_lower = 0;
    nk_i64_t s_upper = 0;
    for (int i = 0; i < 4; i++) {
        nk_u64_t before = s_lower;
        s_lower += lower_vec.u64s[i];
        if (s_lower < before) s_upper++;
        s_upper += upper_vec.i64s[i];
    }
    nk_i64_t s_lower_signed = (nk_i64_t)s_lower;
    if (s_upper == (s_lower_signed >> 63)) *sum = s_lower_signed;
    else if (s_upper >= 0) *sum = NK_I64_MAX;
    else *sum = NK_I64_MIN;
    *sumsq = sq;
}

NK_PUBLIC void nk_reduce_moments_i32_haswell(                      //
    nk_i32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *sum, nk_u64_t *sumsq) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_i32_t);
    int aligned = (stride_bytes % sizeof(nk_i32_t) == 0);
    if (count == 0) *sum = 0, *sumsq = 0;
    else if (!aligned) nk_reduce_moments_i32_serial(data, count, stride_bytes, sum, sumsq);
    else if (stride_elements == 1) nk_reduce_moments_i32_haswell_contiguous_(data, count, sum, sumsq);
    else nk_reduce_moments_i32_serial(data, count, stride_bytes, sum, sumsq);
}

NK_INTERNAL void nk_reduce_minmax_i32_haswell_contiguous_( //
    nk_i32_t const *data, nk_size_t count,                 //
    nk_i32_t *min_value, nk_size_t *min_index,             //
    nk_i32_t *max_value, nk_size_t *max_index) {
    __m256i min_i32x8 = _mm256_loadu_si256((__m256i const *)data);
    __m256i max_i32x8 = min_i32x8;
    __m256i min_iter_u32x8 = _mm256_setzero_si256();
    __m256i max_iter_u32x8 = _mm256_setzero_si256();
    __m256i iter_u32x8 = _mm256_set1_epi32(1);
    __m256i one_u32x8 = _mm256_set1_epi32(1);

    nk_size_t idx = 8;
    for (; idx + 8 <= count; idx += 8) {
        __m256i data_i32x8 = _mm256_loadu_si256((__m256i const *)(data + idx));
        __m256i less_b32x8 = _mm256_cmpgt_epi32(min_i32x8, data_i32x8);
        __m256i greater_b32x8 = _mm256_cmpgt_epi32(data_i32x8, max_i32x8);
        min_i32x8 = _mm256_blendv_epi8(min_i32x8, data_i32x8, less_b32x8);
        max_i32x8 = _mm256_blendv_epi8(max_i32x8, data_i32x8, greater_b32x8);
        min_iter_u32x8 = _mm256_blendv_epi8(min_iter_u32x8, iter_u32x8, less_b32x8);
        max_iter_u32x8 = _mm256_blendv_epi8(max_iter_u32x8, iter_u32x8, greater_b32x8);
        iter_u32x8 = _mm256_add_epi32(iter_u32x8, one_u32x8);
    }

    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        nk_b256_vec_t tail_vec;
        nk_partial_load_b32x8_serial_(data + idx, &tail_vec, remaining);
        nk_b256_vec_t mask_vec;
        mask_vec.ymm = _mm256_setzero_si256();
        for (nk_size_t i = 0; i < remaining; ++i) mask_vec.u32s[i] = 0xFFFFFFFF;
        __m256i valid_b32x8 = mask_vec.ymm;
        __m256i data_min_i32x8 = _mm256_blendv_epi8(_mm256_set1_epi32(NK_I32_MAX), tail_vec.ymm, valid_b32x8);
        __m256i data_max_i32x8 = _mm256_blendv_epi8(_mm256_set1_epi32(NK_I32_MIN), tail_vec.ymm, valid_b32x8);
        __m256i less_b32x8 = _mm256_cmpgt_epi32(min_i32x8, data_min_i32x8);
        __m256i greater_b32x8 = _mm256_cmpgt_epi32(data_max_i32x8, max_i32x8);
        min_i32x8 = _mm256_blendv_epi8(min_i32x8, data_min_i32x8, less_b32x8);
        max_i32x8 = _mm256_blendv_epi8(max_i32x8, data_max_i32x8, greater_b32x8);
        min_iter_u32x8 = _mm256_blendv_epi8(min_iter_u32x8, iter_u32x8, less_b32x8);
        max_iter_u32x8 = _mm256_blendv_epi8(max_iter_u32x8, iter_u32x8, greater_b32x8);
    }

    nk_i32_t min_scalar = nk_reduce_min_i32x8_haswell_(min_i32x8);
    nk_i32_t max_scalar = nk_reduce_max_i32x8_haswell_(max_i32x8);
    __m256i min_eq_b32x8 = _mm256_cmpeq_epi32(min_i32x8, _mm256_set1_epi32(min_scalar));
    __m256i max_eq_b32x8 = _mm256_cmpeq_epi32(max_i32x8, _mm256_set1_epi32(max_scalar));
    unsigned int min_lane = _tzcnt_u32((unsigned int)_mm256_movemask_ps(_mm256_castsi256_ps(min_eq_b32x8)));
    unsigned int max_lane = _tzcnt_u32((unsigned int)_mm256_movemask_ps(_mm256_castsi256_ps(max_eq_b32x8)));
    nk_b256_vec_t iter_vec;
    iter_vec.ymm = min_iter_u32x8;
    *min_value = min_scalar, *min_index = (nk_size_t)iter_vec.u32s[min_lane] * 8 + min_lane;
    iter_vec.ymm = max_iter_u32x8;
    *max_value = max_scalar, *max_index = (nk_size_t)iter_vec.u32s[max_lane] * 8 + max_lane;
}

NK_PUBLIC void nk_reduce_minmax_i32_haswell(                       //
    nk_i32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i32_t *min_value, nk_size_t *min_index,                     //
    nk_i32_t *max_value, nk_size_t *max_index) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_i32_t);
    int aligned = (stride_bytes % sizeof(nk_i32_t) == 0);
    if (count == 0)
        *min_value = NK_I32_MAX, *min_index = NK_SIZE_MAX, *max_value = NK_I32_MIN, *max_index = NK_SIZE_MAX;
    else if (!aligned)
        nk_reduce_minmax_i32_serial(data, count, stride_bytes, min_value, min_index, max_value, max_index);
    else if (count > (nk_size_t)NK_U32_MAX * 8) {
        nk_size_t left_count = count / 2;
        nk_i32_t left_min, right_min, left_max, right_max;
        nk_size_t left_min_index, right_min_index, left_max_index, right_max_index;
        nk_reduce_minmax_i32_haswell(data, left_count, stride_bytes, &left_min, &left_min_index, &left_max,
                                     &left_max_index);
        nk_reduce_minmax_i32_haswell(data + left_count * stride_elements, count - left_count, stride_bytes, &right_min,
                                     &right_min_index, &right_max, &right_max_index);
        if (right_min < left_min) *min_value = right_min, *min_index = left_count + right_min_index;
        else *min_value = left_min, *min_index = left_min_index;
        if (right_max > left_max) *max_value = right_max, *max_index = left_count + right_max_index;
        else *max_value = left_max, *max_index = left_max_index;
    }
    else if (stride_elements == 1)
        nk_reduce_minmax_i32_haswell_contiguous_(data, count, min_value, min_index, max_value, max_index);
    else nk_reduce_minmax_i32_serial(data, count, stride_bytes, min_value, min_index, max_value, max_index);
}

NK_INTERNAL void nk_reduce_moments_u32_haswell_contiguous_( //
    nk_u32_t const *data, nk_size_t count,                  //
    nk_u64_t *sum, nk_u64_t *sumsq) {
    __m256i sum_u64x4 = _mm256_setzero_si256();
    __m256i sumsq_u64x4 = _mm256_setzero_si256();
    nk_size_t idx = 0;
    for (; idx + 8 <= count; idx += 8) {
        __m256i data_u32x8 = _mm256_loadu_si256((__m256i const *)(data + idx));
        sum_u64x4 = _mm256_add_epi64(sum_u64x4, _mm256_cvtepu32_epi64(_mm256_castsi256_si128(data_u32x8)));
        sum_u64x4 = _mm256_add_epi64(sum_u64x4, _mm256_cvtepu32_epi64(_mm256_extracti128_si256(data_u32x8, 1)));
        __m256i even_sq_u64x4 = _mm256_mul_epu32(data_u32x8, data_u32x8);
        __m256i odd_u32x8 = _mm256_srli_epi64(data_u32x8, 32);
        __m256i odd_sq_u64x4 = _mm256_mul_epu32(odd_u32x8, odd_u32x8);
        sumsq_u64x4 = nk_u64_sadd_epi64_haswell_(sumsq_u64x4, even_sq_u64x4);
        sumsq_u64x4 = nk_u64_sadd_epi64_haswell_(sumsq_u64x4, odd_sq_u64x4);
    }
    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        nk_b256_vec_t tail_vec;
        nk_partial_load_b32x8_serial_(data + idx, &tail_vec, remaining);
        __m256i data_u32x8 = tail_vec.ymm;
        sum_u64x4 = _mm256_add_epi64(sum_u64x4, _mm256_cvtepu32_epi64(_mm256_castsi256_si128(data_u32x8)));
        sum_u64x4 = _mm256_add_epi64(sum_u64x4, _mm256_cvtepu32_epi64(_mm256_extracti128_si256(data_u32x8, 1)));
        __m256i even_sq_u64x4 = _mm256_mul_epu32(data_u32x8, data_u32x8);
        __m256i odd_u32x8 = _mm256_srli_epi64(data_u32x8, 32);
        __m256i odd_sq_u64x4 = _mm256_mul_epu32(odd_u32x8, odd_u32x8);
        sumsq_u64x4 = nk_u64_sadd_epi64_haswell_(sumsq_u64x4, even_sq_u64x4);
        sumsq_u64x4 = nk_u64_sadd_epi64_haswell_(sumsq_u64x4, odd_sq_u64x4);
    }
    *sum = (nk_u64_t)nk_reduce_add_i64x4_haswell_(sum_u64x4);
    *sumsq = nk_reduce_sadd_u64x4_haswell_(sumsq_u64x4);
}

NK_PUBLIC void nk_reduce_moments_u32_haswell(                      //
    nk_u32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *sum, nk_u64_t *sumsq) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_u32_t);
    int aligned = (stride_bytes % sizeof(nk_u32_t) == 0);
    if (count == 0) *sum = 0, *sumsq = 0;
    else if (!aligned) nk_reduce_moments_u32_serial(data, count, stride_bytes, sum, sumsq);
    else if (count > (nk_size_t)(NK_U16_MAX + 1) * 8) {
        nk_size_t left_count = count / 2;
        nk_u64_t left_sum, left_sumsq, right_sum, right_sumsq;
        nk_reduce_moments_u32_haswell(data, left_count, stride_bytes, &left_sum, &left_sumsq);
        nk_reduce_moments_u32_haswell(data + left_count * stride_elements, count - left_count, stride_bytes, &right_sum,
                                      &right_sumsq);
        nk_u64_sadd_(&left_sum, &right_sum, sum);
        nk_u64_sadd_(&left_sumsq, &right_sumsq, sumsq);
    }
    else if (stride_elements == 1) nk_reduce_moments_u32_haswell_contiguous_(data, count, sum, sumsq);
    else nk_reduce_moments_u32_serial(data, count, stride_bytes, sum, sumsq);
}

NK_INTERNAL void nk_reduce_minmax_u32_haswell_contiguous_( //
    nk_u32_t const *data, nk_size_t count,                 //
    nk_u32_t *min_value, nk_size_t *min_index,             //
    nk_u32_t *max_value, nk_size_t *max_index) {
    // XOR-bias to signed domain for _mm256_cmpgt_epi32
    __m256i bias_u32x8 = _mm256_set1_epi32((nk_i32_t)0x80000000);
    __m256i raw_u32x8 = _mm256_loadu_si256((__m256i const *)data);
    __m256i min_biased_i32x8 = _mm256_xor_si256(raw_u32x8, bias_u32x8);
    __m256i max_biased_i32x8 = min_biased_i32x8;
    __m256i min_iter_u32x8 = _mm256_setzero_si256();
    __m256i max_iter_u32x8 = _mm256_setzero_si256();
    __m256i iter_u32x8 = _mm256_set1_epi32(1);
    __m256i one_u32x8 = _mm256_set1_epi32(1);

    nk_size_t idx = 8;
    for (; idx + 8 <= count; idx += 8) {
        __m256i data_biased_i32x8 = _mm256_xor_si256(_mm256_loadu_si256((__m256i const *)(data + idx)), bias_u32x8);
        __m256i less_b32x8 = _mm256_cmpgt_epi32(min_biased_i32x8, data_biased_i32x8);
        __m256i greater_b32x8 = _mm256_cmpgt_epi32(data_biased_i32x8, max_biased_i32x8);
        min_biased_i32x8 = _mm256_blendv_epi8(min_biased_i32x8, data_biased_i32x8, less_b32x8);
        max_biased_i32x8 = _mm256_blendv_epi8(max_biased_i32x8, data_biased_i32x8, greater_b32x8);
        min_iter_u32x8 = _mm256_blendv_epi8(min_iter_u32x8, iter_u32x8, less_b32x8);
        max_iter_u32x8 = _mm256_blendv_epi8(max_iter_u32x8, iter_u32x8, greater_b32x8);
        iter_u32x8 = _mm256_add_epi32(iter_u32x8, one_u32x8);
    }

    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        nk_b256_vec_t tail_vec;
        nk_partial_load_b32x8_serial_(data + idx, &tail_vec, remaining);
        __m256i data_biased_i32x8 = _mm256_xor_si256(tail_vec.ymm, bias_u32x8);
        nk_b256_vec_t mask_vec;
        mask_vec.ymm = _mm256_setzero_si256();
        for (nk_size_t i = 0; i < remaining; ++i) mask_vec.u32s[i] = 0xFFFFFFFF;
        __m256i valid_b32x8 = mask_vec.ymm;
        // Biased identity: NK_U32_MAX ^ 0x80000000 = 0x7FFFFFFF for min, 0 ^ 0x80000000 = 0x80000000 for max
        __m256i data_min_i32x8 = _mm256_blendv_epi8(_mm256_set1_epi32(0x7FFFFFFF), data_biased_i32x8, valid_b32x8);
        __m256i data_max_i32x8 = _mm256_blendv_epi8(_mm256_set1_epi32((nk_i32_t)0x80000000), data_biased_i32x8,
                                                    valid_b32x8);
        __m256i less_b32x8 = _mm256_cmpgt_epi32(min_biased_i32x8, data_min_i32x8);
        __m256i greater_b32x8 = _mm256_cmpgt_epi32(data_max_i32x8, max_biased_i32x8);
        min_biased_i32x8 = _mm256_blendv_epi8(min_biased_i32x8, data_min_i32x8, less_b32x8);
        max_biased_i32x8 = _mm256_blendv_epi8(max_biased_i32x8, data_max_i32x8, greater_b32x8);
        min_iter_u32x8 = _mm256_blendv_epi8(min_iter_u32x8, iter_u32x8, less_b32x8);
        max_iter_u32x8 = _mm256_blendv_epi8(max_iter_u32x8, iter_u32x8, greater_b32x8);
    }

    __m256i min_u32x8 = _mm256_xor_si256(min_biased_i32x8, bias_u32x8);
    __m256i max_u32x8 = _mm256_xor_si256(max_biased_i32x8, bias_u32x8);
    nk_u32_t min_scalar = nk_reduce_min_u32x8_haswell_(min_u32x8);
    nk_u32_t max_scalar = nk_reduce_max_u32x8_haswell_(max_u32x8);
    __m256i min_eq_b32x8 = _mm256_cmpeq_epi32(min_u32x8, _mm256_set1_epi32((nk_i32_t)min_scalar));
    __m256i max_eq_b32x8 = _mm256_cmpeq_epi32(max_u32x8, _mm256_set1_epi32((nk_i32_t)max_scalar));
    unsigned int min_lane = _tzcnt_u32((unsigned int)_mm256_movemask_ps(_mm256_castsi256_ps(min_eq_b32x8)));
    unsigned int max_lane = _tzcnt_u32((unsigned int)_mm256_movemask_ps(_mm256_castsi256_ps(max_eq_b32x8)));
    nk_b256_vec_t iter_vec;
    iter_vec.ymm = min_iter_u32x8;
    *min_value = min_scalar, *min_index = (nk_size_t)iter_vec.u32s[min_lane] * 8 + min_lane;
    iter_vec.ymm = max_iter_u32x8;
    *max_value = max_scalar, *max_index = (nk_size_t)iter_vec.u32s[max_lane] * 8 + max_lane;
}

NK_PUBLIC void nk_reduce_minmax_u32_haswell(                       //
    nk_u32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u32_t *min_value, nk_size_t *min_index,                     //
    nk_u32_t *max_value, nk_size_t *max_index) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_u32_t);
    int aligned = (stride_bytes % sizeof(nk_u32_t) == 0);
    if (count == 0) *min_value = NK_U32_MAX, *min_index = NK_SIZE_MAX, *max_value = 0, *max_index = NK_SIZE_MAX;
    else if (!aligned)
        nk_reduce_minmax_u32_serial(data, count, stride_bytes, min_value, min_index, max_value, max_index);
    else if (count > (nk_size_t)NK_U32_MAX * 8) {
        nk_size_t left_count = count / 2;
        nk_u32_t left_min, right_min, left_max, right_max;
        nk_size_t left_min_index, right_min_index, left_max_index, right_max_index;
        nk_reduce_minmax_u32_haswell(data, left_count, stride_bytes, &left_min, &left_min_index, &left_max,
                                     &left_max_index);
        nk_reduce_minmax_u32_haswell(data + left_count * stride_elements, count - left_count, stride_bytes, &right_min,
                                     &right_min_index, &right_max, &right_max_index);
        if (right_min < left_min) *min_value = right_min, *min_index = left_count + right_min_index;
        else *min_value = left_min, *min_index = left_min_index;
        if (right_max > left_max) *max_value = right_max, *max_index = left_count + right_max_index;
        else *max_value = left_max, *max_index = left_max_index;
    }
    else if (stride_elements == 1)
        nk_reduce_minmax_u32_haswell_contiguous_(data, count, min_value, min_index, max_value, max_index);
    else nk_reduce_minmax_u32_serial(data, count, stride_bytes, min_value, min_index, max_value, max_index);
}

NK_INTERNAL void nk_reduce_moments_i64_haswell_contiguous_( //
    nk_i64_t const *data, nk_size_t count,                  //
    nk_i64_t *sum, nk_u64_t *sumsq) {
    nk_u64_t sum_lower = 0;
    nk_i64_t sum_upper = 0;
    __m256i sumsq_u64x4 = _mm256_setzero_si256();
    int sumsq_overflow_mask = 0;
    __m256i sign_bit_i64x4 = _mm256_set1_epi64x((nk_i64_t)0x8000000000000000ULL);
    nk_size_t idx = 0;
    for (; idx + 4 <= count; idx += 4) {
        __m256i data_i64x4 = _mm256_loadu_si256((__m256i const *)(data + idx));
        __m256i squared_u64x4 = nk_i64_smul_sq_epi64_haswell_(data_i64x4);
        __m256i sumsq_before_u64x4 = sumsq_u64x4;
        sumsq_u64x4 = _mm256_add_epi64(sumsq_u64x4, squared_u64x4);
        __m256i sq_result_biased_u64x4 = _mm256_xor_si256(sumsq_u64x4, sign_bit_i64x4);
        __m256i sq_before_biased_u64x4 = _mm256_xor_si256(sumsq_before_u64x4, sign_bit_i64x4);
        sumsq_overflow_mask |= _mm256_movemask_pd(
            _mm256_castsi256_pd(_mm256_cmpgt_epi64(sq_before_biased_u64x4, sq_result_biased_u64x4)));
        nk_b256_vec_t vec;
        vec.ymm = data_i64x4;
        for (int i = 0; i < 4; i++) {
            nk_i64_t val = vec.i64s[i];
            nk_u64_t before = sum_lower;
            sum_lower += (nk_u64_t)val;
            if (sum_lower < before) sum_upper++;
            sum_upper += (val >> 63);
        }
    }
    nk_u64_t sq;
    if (sumsq_overflow_mask) sq = NK_U64_MAX;
    else sq = nk_reduce_sadd_u64x4_haswell_(sumsq_u64x4);
    for (; idx < count; ++idx) {
        nk_i64_t val = data[idx];
        nk_i64_t product;
        nk_i64_smul_(&val, &val, &product);
        nk_u64_t unsigned_product = (nk_u64_t)product;
        nk_u64_sadd_(&sq, &unsigned_product, &sq);
        nk_u64_t before = sum_lower;
        sum_lower += (nk_u64_t)val;
        if (sum_lower < before) sum_upper++;
        sum_upper += (val >> 63);
    }
    nk_i64_t sum_lower_signed = (nk_i64_t)sum_lower;
    if (sum_upper == (sum_lower_signed >> 63)) *sum = sum_lower_signed;
    else if (sum_upper >= 0) *sum = NK_I64_MAX;
    else *sum = NK_I64_MIN;
    *sumsq = sq;
}

NK_PUBLIC void nk_reduce_moments_i64_haswell(                      //
    nk_i64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *sum, nk_u64_t *sumsq) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_i64_t);
    int aligned = (stride_bytes % sizeof(nk_i64_t) == 0);
    if (count == 0) *sum = 0, *sumsq = 0;
    else if (!aligned) nk_reduce_moments_i64_serial(data, count, stride_bytes, sum, sumsq);
    else if (stride_elements == 1) nk_reduce_moments_i64_haswell_contiguous_(data, count, sum, sumsq);
    else nk_reduce_moments_i64_serial(data, count, stride_bytes, sum, sumsq);
}

NK_INTERNAL void nk_reduce_minmax_i64_haswell_contiguous_( //
    nk_i64_t const *data, nk_size_t count,                 //
    nk_i64_t *min_value, nk_size_t *min_index,             //
    nk_i64_t *max_value, nk_size_t *max_index) {
    __m256i min_i64x4 = _mm256_loadu_si256((__m256i const *)data);
    __m256i max_i64x4 = min_i64x4;
    __m256i min_iter_u64x4 = _mm256_setzero_si256();
    __m256i max_iter_u64x4 = _mm256_setzero_si256();
    __m256i iter_u64x4 = _mm256_set1_epi64x(1);
    __m256i one_u64x4 = _mm256_set1_epi64x(1);

    nk_size_t idx = 4;
    for (; idx + 4 <= count; idx += 4) {
        __m256i data_i64x4 = _mm256_loadu_si256((__m256i const *)(data + idx));
        __m256i less_b64x4 = _mm256_cmpgt_epi64(min_i64x4, data_i64x4);
        __m256i greater_b64x4 = _mm256_cmpgt_epi64(data_i64x4, max_i64x4);
        min_i64x4 = _mm256_blendv_epi8(min_i64x4, data_i64x4, less_b64x4);
        max_i64x4 = _mm256_blendv_epi8(max_i64x4, data_i64x4, greater_b64x4);
        min_iter_u64x4 = _mm256_blendv_epi8(min_iter_u64x4, iter_u64x4, less_b64x4);
        max_iter_u64x4 = _mm256_blendv_epi8(max_iter_u64x4, iter_u64x4, greater_b64x4);
        iter_u64x4 = _mm256_add_epi64(iter_u64x4, one_u64x4);
    }

    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        nk_b256_vec_t tail_vec;
        nk_partial_load_b64x4_serial_(data + idx, &tail_vec, remaining);
        nk_b256_vec_t mask_vec;
        mask_vec.ymm = _mm256_setzero_si256();
        for (nk_size_t i = 0; i < remaining; ++i) mask_vec.u64s[i] = (nk_u64_t)-1;
        __m256i valid_b64x4 = mask_vec.ymm;
        __m256i data_min_i64x4 = _mm256_blendv_epi8(_mm256_set1_epi64x(NK_I64_MAX), tail_vec.ymm, valid_b64x4);
        __m256i data_max_i64x4 = _mm256_blendv_epi8(_mm256_set1_epi64x(NK_I64_MIN), tail_vec.ymm, valid_b64x4);
        __m256i less_b64x4 = _mm256_cmpgt_epi64(min_i64x4, data_min_i64x4);
        __m256i greater_b64x4 = _mm256_cmpgt_epi64(data_max_i64x4, max_i64x4);
        min_i64x4 = _mm256_blendv_epi8(min_i64x4, data_min_i64x4, less_b64x4);
        max_i64x4 = _mm256_blendv_epi8(max_i64x4, data_max_i64x4, greater_b64x4);
        min_iter_u64x4 = _mm256_blendv_epi8(min_iter_u64x4, iter_u64x4, less_b64x4);
        max_iter_u64x4 = _mm256_blendv_epi8(max_iter_u64x4, iter_u64x4, greater_b64x4);
    }

    nk_i64_t min_scalar = nk_reduce_min_i64x4_haswell_(min_i64x4);
    nk_i64_t max_scalar = nk_reduce_max_i64x4_haswell_(max_i64x4);
    __m256i min_eq_b64x4 = _mm256_cmpeq_epi64(min_i64x4, _mm256_set1_epi64x(min_scalar));
    __m256i max_eq_b64x4 = _mm256_cmpeq_epi64(max_i64x4, _mm256_set1_epi64x(max_scalar));
    unsigned int min_lane = _tzcnt_u32((unsigned int)_mm256_movemask_pd(_mm256_castsi256_pd(min_eq_b64x4)));
    unsigned int max_lane = _tzcnt_u32((unsigned int)_mm256_movemask_pd(_mm256_castsi256_pd(max_eq_b64x4)));
    nk_b256_vec_t iter_vec;
    iter_vec.ymm = min_iter_u64x4;
    *min_value = min_scalar, *min_index = (nk_size_t)iter_vec.u64s[min_lane] * 4 + min_lane;
    iter_vec.ymm = max_iter_u64x4;
    *max_value = max_scalar, *max_index = (nk_size_t)iter_vec.u64s[max_lane] * 4 + max_lane;
}

NK_PUBLIC void nk_reduce_minmax_i64_haswell(                       //
    nk_i64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *min_value, nk_size_t *min_index,                     //
    nk_i64_t *max_value, nk_size_t *max_index) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_i64_t);
    int aligned = (stride_bytes % sizeof(nk_i64_t) == 0);
    if (count == 0)
        *min_value = NK_I64_MAX, *min_index = NK_SIZE_MAX, *max_value = NK_I64_MIN, *max_index = NK_SIZE_MAX;
    else if (!aligned)
        nk_reduce_minmax_i64_serial(data, count, stride_bytes, min_value, min_index, max_value, max_index);
    else if (stride_elements == 1)
        nk_reduce_minmax_i64_haswell_contiguous_(data, count, min_value, min_index, max_value, max_index);
    else nk_reduce_minmax_i64_serial(data, count, stride_bytes, min_value, min_index, max_value, max_index);
}

NK_INTERNAL void nk_reduce_moments_u64_haswell_contiguous_( //
    nk_u64_t const *data, nk_size_t count,                  //
    nk_u64_t *sum, nk_u64_t *sumsq) {
    nk_u64_t s = 0;
    __m256i sumsq_u64x4 = _mm256_setzero_si256();
    nk_size_t idx = 0;
    for (; idx + 4 <= count; idx += 4) {
        __m256i data_u64x4 = _mm256_loadu_si256((__m256i const *)(data + idx));
        sumsq_u64x4 = nk_u64_sadd_epi64_haswell_(sumsq_u64x4, nk_u64_smul_sq_epi64_haswell_(data_u64x4));
        nk_b256_vec_t vec;
        vec.ymm = data_u64x4;
        for (int i = 0; i < 4; i++) nk_u64_sadd_(&s, &vec.u64s[i], &s);
    }
    nk_u64_t sq = nk_reduce_sadd_u64x4_haswell_(sumsq_u64x4);
    for (; idx < count; ++idx) {
        nk_u64_t val = data[idx];
        nk_u64_t product;
        nk_u64_smul_(&val, &val, &product);
        nk_u64_sadd_(&s, &val, &s);
        nk_u64_sadd_(&sq, &product, &sq);
    }
    *sum = s;
    *sumsq = sq;
}

NK_PUBLIC void nk_reduce_moments_u64_haswell(                      //
    nk_u64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *sum, nk_u64_t *sumsq) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_u64_t);
    int aligned = (stride_bytes % sizeof(nk_u64_t) == 0);
    if (count == 0) *sum = 0, *sumsq = 0;
    else if (!aligned) nk_reduce_moments_u64_serial(data, count, stride_bytes, sum, sumsq);
    else if (stride_elements == 1) nk_reduce_moments_u64_haswell_contiguous_(data, count, sum, sumsq);
    else nk_reduce_moments_u64_serial(data, count, stride_bytes, sum, sumsq);
}

NK_INTERNAL void nk_reduce_minmax_u64_haswell_contiguous_( //
    nk_u64_t const *data, nk_size_t count,                 //
    nk_u64_t *min_value, nk_size_t *min_index,             //
    nk_u64_t *max_value, nk_size_t *max_index) {
    // XOR-bias to signed domain for _mm256_cmpgt_epi64
    __m256i bias_u64x4 = _mm256_set1_epi64x((nk_i64_t)0x8000000000000000ull);
    __m256i raw_u64x4 = _mm256_loadu_si256((__m256i const *)data);
    __m256i min_biased_i64x4 = _mm256_xor_si256(raw_u64x4, bias_u64x4);
    __m256i max_biased_i64x4 = min_biased_i64x4;
    __m256i min_iter_u64x4 = _mm256_setzero_si256();
    __m256i max_iter_u64x4 = _mm256_setzero_si256();
    __m256i iter_u64x4 = _mm256_set1_epi64x(1);
    __m256i one_u64x4 = _mm256_set1_epi64x(1);

    nk_size_t idx = 4;
    for (; idx + 4 <= count; idx += 4) {
        __m256i data_biased_i64x4 = _mm256_xor_si256(_mm256_loadu_si256((__m256i const *)(data + idx)), bias_u64x4);
        __m256i less_b64x4 = _mm256_cmpgt_epi64(min_biased_i64x4, data_biased_i64x4);
        __m256i greater_b64x4 = _mm256_cmpgt_epi64(data_biased_i64x4, max_biased_i64x4);
        min_biased_i64x4 = _mm256_blendv_epi8(min_biased_i64x4, data_biased_i64x4, less_b64x4);
        max_biased_i64x4 = _mm256_blendv_epi8(max_biased_i64x4, data_biased_i64x4, greater_b64x4);
        min_iter_u64x4 = _mm256_blendv_epi8(min_iter_u64x4, iter_u64x4, less_b64x4);
        max_iter_u64x4 = _mm256_blendv_epi8(max_iter_u64x4, iter_u64x4, greater_b64x4);
        iter_u64x4 = _mm256_add_epi64(iter_u64x4, one_u64x4);
    }

    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        nk_b256_vec_t tail_vec;
        nk_partial_load_b64x4_serial_(data + idx, &tail_vec, remaining);
        __m256i data_biased_i64x4 = _mm256_xor_si256(tail_vec.ymm, bias_u64x4);
        nk_b256_vec_t mask_vec;
        mask_vec.ymm = _mm256_setzero_si256();
        for (nk_size_t i = 0; i < remaining; ++i) mask_vec.u64s[i] = (nk_u64_t)-1;
        __m256i valid_b64x4 = mask_vec.ymm;
        // Biased identity: NK_U64_MAX ^ bias = 0x7FFF... for min, 0 ^ bias = 0x8000... for max
        __m256i data_min_i64x4 = _mm256_blendv_epi8(_mm256_set1_epi64x(NK_I64_MAX), data_biased_i64x4, valid_b64x4);
        __m256i data_max_i64x4 = _mm256_blendv_epi8(_mm256_set1_epi64x(NK_I64_MIN), data_biased_i64x4, valid_b64x4);
        __m256i less_b64x4 = _mm256_cmpgt_epi64(min_biased_i64x4, data_min_i64x4);
        __m256i greater_b64x4 = _mm256_cmpgt_epi64(data_max_i64x4, max_biased_i64x4);
        min_biased_i64x4 = _mm256_blendv_epi8(min_biased_i64x4, data_min_i64x4, less_b64x4);
        max_biased_i64x4 = _mm256_blendv_epi8(max_biased_i64x4, data_max_i64x4, greater_b64x4);
        min_iter_u64x4 = _mm256_blendv_epi8(min_iter_u64x4, iter_u64x4, less_b64x4);
        max_iter_u64x4 = _mm256_blendv_epi8(max_iter_u64x4, iter_u64x4, greater_b64x4);
    }

    __m256i min_u64x4 = _mm256_xor_si256(min_biased_i64x4, bias_u64x4);
    __m256i max_u64x4 = _mm256_xor_si256(max_biased_i64x4, bias_u64x4);
    nk_u64_t min_scalar = nk_reduce_min_u64x4_haswell_(min_u64x4);
    nk_u64_t max_scalar = nk_reduce_max_u64x4_haswell_(max_u64x4);
    __m256i min_eq_b64x4 = _mm256_cmpeq_epi64(min_u64x4, _mm256_set1_epi64x((nk_i64_t)min_scalar));
    __m256i max_eq_b64x4 = _mm256_cmpeq_epi64(max_u64x4, _mm256_set1_epi64x((nk_i64_t)max_scalar));
    unsigned int min_lane = _tzcnt_u32((unsigned int)_mm256_movemask_pd(_mm256_castsi256_pd(min_eq_b64x4)));
    unsigned int max_lane = _tzcnt_u32((unsigned int)_mm256_movemask_pd(_mm256_castsi256_pd(max_eq_b64x4)));
    nk_b256_vec_t iter_vec;
    iter_vec.ymm = min_iter_u64x4;
    *min_value = min_scalar, *min_index = (nk_size_t)iter_vec.u64s[min_lane] * 4 + min_lane;
    iter_vec.ymm = max_iter_u64x4;
    *max_value = max_scalar, *max_index = (nk_size_t)iter_vec.u64s[max_lane] * 4 + max_lane;
}

NK_PUBLIC void nk_reduce_minmax_u64_haswell(                       //
    nk_u64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *min_value, nk_size_t *min_index,                     //
    nk_u64_t *max_value, nk_size_t *max_index) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_u64_t);
    int aligned = (stride_bytes % sizeof(nk_u64_t) == 0);
    if (count == 0) *min_value = NK_U64_MAX, *min_index = NK_SIZE_MAX, *max_value = 0, *max_index = NK_SIZE_MAX;
    else if (!aligned)
        nk_reduce_minmax_u64_serial(data, count, stride_bytes, min_value, min_index, max_value, max_index);
    else if (stride_elements == 1)
        nk_reduce_minmax_u64_haswell_contiguous_(data, count, min_value, min_index, max_value, max_index);
    else nk_reduce_minmax_u64_serial(data, count, stride_bytes, min_value, min_index, max_value, max_index);
}

NK_INTERNAL void nk_reduce_moments_e4m3_haswell_contiguous_( //
    nk_e4m3_t const *data, nk_size_t count,                  //
    nk_f32_t *sum, nk_f32_t *sumsq) {
    __m256 sum_f32x8 = _mm256_setzero_ps();
    __m256 sumsq_f32x8 = _mm256_setzero_ps();
    nk_size_t idx = 0;
    for (; idx + 8 <= count; idx += 8) {
        __m256 data_f32x8 = nk_e4m3x8_to_f32x8_haswell_(_mm_loadl_epi64((__m128i const *)(data + idx)));
        sum_f32x8 = _mm256_add_ps(sum_f32x8, data_f32x8);
        sumsq_f32x8 = _mm256_fmadd_ps(data_f32x8, data_f32x8, sumsq_f32x8);
    }
    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        nk_b256_vec_t vec;
        nk_partial_load_e4m3x8_to_f32x8_haswell_(data + idx, &vec, remaining);
        sum_f32x8 = _mm256_add_ps(sum_f32x8, vec.ymm_ps);
        sumsq_f32x8 = _mm256_fmadd_ps(vec.ymm_ps, vec.ymm_ps, sumsq_f32x8);
    }
    *sum = nk_reduce_add_f32x8_haswell_(sum_f32x8);
    *sumsq = nk_reduce_add_f32x8_haswell_(sumsq_f32x8);
}

NK_INTERNAL void nk_reduce_moments_e4m3_haswell_strided_(              //
    nk_e4m3_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_f32_t *sum, nk_f32_t *sumsq) {
    __m256 sum_f32x8 = _mm256_setzero_ps();
    __m256 sumsq_f32x8 = _mm256_setzero_ps();
    nk_size_t idx = 0;
    nk_size_t elems_per_chunk = 8;
    for (; idx + elems_per_chunk <= count; idx += elems_per_chunk) {
        nk_b64_vec_t buf;
        buf.u64 = 0;
        nk_e4m3_t const *ptr = data + idx * stride_elements;
        for (nk_size_t i = 0; i < elems_per_chunk; ++i) buf.u8s[i] = ptr[i * stride_elements];
        __m256 data_f32x8 = nk_e4m3x8_to_f32x8_haswell_(_mm_cvtsi64_si128(buf.u64));
        sum_f32x8 = _mm256_add_ps(sum_f32x8, data_f32x8);
        sumsq_f32x8 = _mm256_fmadd_ps(data_f32x8, data_f32x8, sumsq_f32x8);
    }
    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        nk_b64_vec_t buf;
        buf.u64 = 0;
        nk_e4m3_t const *ptr = data + idx * stride_elements;
        for (nk_size_t i = 0; i < remaining; ++i) buf.u8s[i] = ptr[i * stride_elements];
        nk_b256_vec_t vec;
        nk_partial_load_e4m3x8_to_f32x8_haswell_((nk_e4m3_t *)&buf, &vec, remaining);
        sum_f32x8 = _mm256_add_ps(sum_f32x8, vec.ymm_ps);
        sumsq_f32x8 = _mm256_fmadd_ps(vec.ymm_ps, vec.ymm_ps, sumsq_f32x8);
    }
    *sum = nk_reduce_add_f32x8_haswell_(sum_f32x8);
    *sumsq = nk_reduce_add_f32x8_haswell_(sumsq_f32x8);
}

NK_PUBLIC void nk_reduce_moments_e4m3_haswell(                      //
    nk_e4m3_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *sum, nk_f32_t *sumsq) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_e4m3_t);
    int aligned = (stride_bytes % sizeof(nk_e4m3_t) == 0);
    if (count == 0) *sum = 0, *sumsq = 0;
    else if (!aligned) nk_reduce_moments_e4m3_serial(data, count, stride_bytes, sum, sumsq);
    else if (count > (nk_size_t)(NK_U16_MAX + 1) * 32) {
        nk_size_t left_count = count / 2;
        nk_f32_t left_sum, left_sumsq, right_sum, right_sumsq;
        nk_reduce_moments_e4m3_haswell(data, left_count, stride_bytes, &left_sum, &left_sumsq);
        nk_reduce_moments_e4m3_haswell(data + left_count * stride_elements, count - left_count, stride_bytes,
                                       &right_sum, &right_sumsq);
        *sum = left_sum + right_sum, *sumsq = left_sumsq + right_sumsq;
    }
    else if (stride_elements == 1) nk_reduce_moments_e4m3_haswell_contiguous_(data, count, sum, sumsq);
    else if (stride_elements >= 2 && stride_elements <= 8)
        nk_reduce_moments_e4m3_haswell_strided_(data, count, stride_elements, sum, sumsq);
    else nk_reduce_moments_e4m3_serial(data, count, stride_bytes, sum, sumsq);
}

NK_INTERNAL void nk_reduce_minmax_e4m3_haswell_contiguous_( //
    nk_e4m3_t const *data, nk_size_t count,                 //
    nk_e4m3_t *min_value, nk_size_t *min_index,             //
    nk_e4m3_t *max_value, nk_size_t *max_index) {
    __m256i nan_cmp_u8x32 = _mm256_set1_epi8((char)0xFF);
    __m256i first_i8x32 = _mm256_loadu_si256((__m256i const *)data);
    __m256i first_cmp = nk_fp8x32_to_u8x32_comparable_haswell_(first_i8x32);
    nk_b256_vec_t min_vec, max_vec;
    min_vec.ymm = first_cmp;
    max_vec.ymm = first_cmp;
    __m256i min_iter_u8x32 = _mm256_setzero_si256();
    __m256i max_iter_u8x32 = _mm256_setzero_si256();
    __m256i iter_u8x32 = _mm256_set1_epi8(1);
    __m256i one_u8x32 = _mm256_set1_epi8(1);

    nk_size_t idx = 32;
    for (; idx + 32 <= count; idx += 32) {
        __m256i data_i8x32 = _mm256_loadu_si256((__m256i const *)(data + idx));
        __m256i data_cmp_u8x32 = nk_fp8x32_to_u8x32_comparable_haswell_(data_i8x32);
        __m256i min_mask_i8x32 = nk_min_mask_e4m3x32_haswell_(min_vec.ymm, data_cmp_u8x32, nan_cmp_u8x32);
        __m256i new_min = _mm256_blendv_epi8(data_cmp_u8x32, min_vec.ymm, min_mask_i8x32);
        __m256i min_changed_i8x32 = _mm256_xor_si256(_mm256_cmpeq_epi8(new_min, min_vec.ymm),
                                                     _mm256_set1_epi8((char)0xFF));
        min_vec.ymm = new_min;
        min_iter_u8x32 = _mm256_blendv_epi8(min_iter_u8x32, iter_u8x32, min_changed_i8x32);
        __m256i max_mask_i8x32 = nk_max_mask_e4m3x32_haswell_(max_vec.ymm, data_cmp_u8x32, nan_cmp_u8x32);
        __m256i new_max = _mm256_blendv_epi8(data_cmp_u8x32, max_vec.ymm, max_mask_i8x32);
        __m256i max_changed_i8x32 = _mm256_xor_si256(_mm256_cmpeq_epi8(new_max, max_vec.ymm),
                                                     _mm256_set1_epi8((char)0xFF));
        max_vec.ymm = new_max;
        max_iter_u8x32 = _mm256_blendv_epi8(max_iter_u8x32, iter_u8x32, max_changed_i8x32);
        iter_u8x32 = _mm256_add_epi8(iter_u8x32, one_u8x32);
    }

    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        nk_b256_vec_t tail_vec;
        tail_vec.ymm = nan_cmp_u8x32;
        nk_partial_load_b8x32_serial_(data + idx, &tail_vec, remaining);
        // Fill invalid lanes with NaN comparable form (0xFF)
        for (nk_size_t i = remaining; i < 32; ++i) tail_vec.u8s[i] = 0xFF;
        __m256i data_cmp_u8x32 = nk_fp8x32_to_u8x32_comparable_haswell_(tail_vec.ymm);
        // Rewrite invalid lanes back to 0xFF after conversion
        nk_b256_vec_t cmp_vec;
        cmp_vec.ymm = data_cmp_u8x32;
        for (nk_size_t i = remaining; i < 32; ++i) cmp_vec.u8s[i] = 0xFF;
        data_cmp_u8x32 = cmp_vec.ymm;
        __m256i min_mask_i8x32 = nk_min_mask_e4m3x32_haswell_(min_vec.ymm, data_cmp_u8x32, nan_cmp_u8x32);
        __m256i new_min = _mm256_blendv_epi8(data_cmp_u8x32, min_vec.ymm, min_mask_i8x32);
        __m256i min_changed_i8x32 = _mm256_xor_si256(_mm256_cmpeq_epi8(new_min, min_vec.ymm),
                                                     _mm256_set1_epi8((char)0xFF));
        min_vec.ymm = new_min;
        min_iter_u8x32 = _mm256_blendv_epi8(min_iter_u8x32, iter_u8x32, min_changed_i8x32);
        __m256i max_mask_i8x32 = nk_max_mask_e4m3x32_haswell_(max_vec.ymm, data_cmp_u8x32, nan_cmp_u8x32);
        __m256i new_max = _mm256_blendv_epi8(data_cmp_u8x32, max_vec.ymm, max_mask_i8x32);
        __m256i max_changed_i8x32 = _mm256_xor_si256(_mm256_cmpeq_epi8(new_max, max_vec.ymm),
                                                     _mm256_set1_epi8((char)0xFF));
        max_vec.ymm = new_max;
        max_iter_u8x32 = _mm256_blendv_epi8(max_iter_u8x32, iter_u8x32, max_changed_i8x32);
    }

    nk_size_t min_lane = nk_argmin_u8x32_haswell_(min_vec.ymm);
    nk_size_t max_lane = nk_argmax_u8x32_haswell_(max_vec.ymm);
    nk_b256_vec_t iter_vec;
    iter_vec.ymm = min_iter_u8x32;
    *min_index = (nk_size_t)iter_vec.u8s[min_lane] * 32 + min_lane;
    iter_vec.ymm = max_iter_u8x32;
    *max_index = (nk_size_t)iter_vec.u8s[max_lane] * 32 + max_lane;
    min_vec.ymm = nk_u8x32_comparable_to_fp8x32_haswell_(min_vec.ymm);
    max_vec.ymm = nk_u8x32_comparable_to_fp8x32_haswell_(max_vec.ymm);
    *min_value = min_vec.e4m3s[min_lane];
    *max_value = max_vec.e4m3s[max_lane];
}

NK_PUBLIC void nk_reduce_minmax_e4m3_haswell(                       //
    nk_e4m3_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_e4m3_t *min_value, nk_size_t *min_index,                     //
    nk_e4m3_t *max_value, nk_size_t *max_index) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_e4m3_t);
    if (count == 0)
        *min_value = NK_E4M3_MAX, *min_index = NK_SIZE_MAX, *max_value = NK_E4M3_MIN, *max_index = NK_SIZE_MAX;
    else if (stride_elements == 1 && count > (nk_size_t)(NK_U8_MAX + 1) * 32) {
        nk_size_t left_count = count / 2;
        nk_e4m3_t left_min, right_min, left_max, right_max;
        nk_size_t left_min_index, right_min_index, left_max_index, right_max_index;
        nk_reduce_minmax_e4m3_haswell(data, left_count, stride_bytes, &left_min, &left_min_index, &left_max,
                                      &left_max_index);
        nk_reduce_minmax_e4m3_haswell(data + left_count, count - left_count, stride_bytes, &right_min, &right_min_index,
                                      &right_max, &right_max_index);
        if (nk_e4m3_compare_(right_min, left_min) < 0)
            *min_value = right_min, *min_index = left_count + right_min_index;
        else *min_value = left_min, *min_index = left_min_index;
        if (nk_e4m3_compare_(right_max, left_max) > 0)
            *max_value = right_max, *max_index = left_count + right_max_index;
        else *max_value = left_max, *max_index = left_max_index;
    }
    else if (stride_elements == 1)
        nk_reduce_minmax_e4m3_haswell_contiguous_(data, count, min_value, min_index, max_value, max_index);
    else nk_reduce_minmax_e4m3_serial(data, count, stride_bytes, min_value, min_index, max_value, max_index);
}

NK_INTERNAL void nk_reduce_moments_e5m2_haswell_contiguous_( //
    nk_e5m2_t const *data, nk_size_t count,                  //
    nk_f32_t *sum, nk_f32_t *sumsq) {
    __m256 sum_f32x8 = _mm256_setzero_ps();
    __m256 sumsq_f32x8 = _mm256_setzero_ps();
    nk_size_t idx = 0;
    for (; idx + 8 <= count; idx += 8) {
        __m256 data_f32x8 = nk_e5m2x8_to_f32x8_haswell_(_mm_loadl_epi64((__m128i const *)(data + idx)));
        sum_f32x8 = _mm256_add_ps(sum_f32x8, data_f32x8);
        sumsq_f32x8 = _mm256_fmadd_ps(data_f32x8, data_f32x8, sumsq_f32x8);
    }
    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        nk_b256_vec_t vec;
        nk_partial_load_e5m2x8_to_f32x8_haswell_(data + idx, &vec, remaining);
        sum_f32x8 = _mm256_add_ps(sum_f32x8, vec.ymm_ps);
        sumsq_f32x8 = _mm256_fmadd_ps(vec.ymm_ps, vec.ymm_ps, sumsq_f32x8);
    }
    *sum = nk_reduce_add_f32x8_haswell_(sum_f32x8);
    *sumsq = nk_reduce_add_f32x8_haswell_(sumsq_f32x8);
}

NK_INTERNAL void nk_reduce_moments_e5m2_haswell_strided_(              //
    nk_e5m2_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_f32_t *sum, nk_f32_t *sumsq) {
    __m256 sum_f32x8 = _mm256_setzero_ps();
    __m256 sumsq_f32x8 = _mm256_setzero_ps();
    nk_size_t idx = 0;
    nk_size_t elems_per_chunk = 8;
    for (; idx + elems_per_chunk <= count; idx += elems_per_chunk) {
        nk_b64_vec_t buf;
        buf.u64 = 0;
        nk_e5m2_t const *ptr = data + idx * stride_elements;
        for (nk_size_t i = 0; i < elems_per_chunk; ++i) buf.u8s[i] = ptr[i * stride_elements];
        __m256 data_f32x8 = nk_e5m2x8_to_f32x8_haswell_(_mm_cvtsi64_si128(buf.u64));
        sum_f32x8 = _mm256_add_ps(sum_f32x8, data_f32x8);
        sumsq_f32x8 = _mm256_fmadd_ps(data_f32x8, data_f32x8, sumsq_f32x8);
    }
    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        nk_b64_vec_t buf;
        buf.u64 = 0;
        nk_e5m2_t const *ptr = data + idx * stride_elements;
        for (nk_size_t i = 0; i < remaining; ++i) buf.u8s[i] = ptr[i * stride_elements];
        nk_b256_vec_t vec;
        nk_partial_load_e5m2x8_to_f32x8_haswell_((nk_e5m2_t *)&buf, &vec, remaining);
        sum_f32x8 = _mm256_add_ps(sum_f32x8, vec.ymm_ps);
        sumsq_f32x8 = _mm256_fmadd_ps(vec.ymm_ps, vec.ymm_ps, sumsq_f32x8);
    }
    *sum = nk_reduce_add_f32x8_haswell_(sum_f32x8);
    *sumsq = nk_reduce_add_f32x8_haswell_(sumsq_f32x8);
}

NK_PUBLIC void nk_reduce_moments_e5m2_haswell(                      //
    nk_e5m2_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *sum, nk_f32_t *sumsq) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_e5m2_t);
    int aligned = (stride_bytes % sizeof(nk_e5m2_t) == 0);
    if (count == 0) *sum = 0, *sumsq = 0;
    else if (!aligned) nk_reduce_moments_e5m2_serial(data, count, stride_bytes, sum, sumsq);
    else if (count > (nk_size_t)(NK_U16_MAX + 1) * 32) {
        nk_size_t left_count = count / 2;
        nk_f32_t left_sum, left_sumsq, right_sum, right_sumsq;
        nk_reduce_moments_e5m2_haswell(data, left_count, stride_bytes, &left_sum, &left_sumsq);
        nk_reduce_moments_e5m2_haswell(data + left_count * stride_elements, count - left_count, stride_bytes,
                                       &right_sum, &right_sumsq);
        *sum = left_sum + right_sum, *sumsq = left_sumsq + right_sumsq;
    }
    else if (stride_elements == 1) nk_reduce_moments_e5m2_haswell_contiguous_(data, count, sum, sumsq);
    else if (stride_elements >= 2 && stride_elements <= 8)
        nk_reduce_moments_e5m2_haswell_strided_(data, count, stride_elements, sum, sumsq);
    else nk_reduce_moments_e5m2_serial(data, count, stride_bytes, sum, sumsq);
}

NK_INTERNAL void nk_reduce_minmax_e5m2_haswell_contiguous_( //
    nk_e5m2_t const *data, nk_size_t count,                 //
    nk_e5m2_t *min_value, nk_size_t *min_index,             //
    nk_e5m2_t *max_value, nk_size_t *max_index) {
    // E5M2 NaN threshold in comparable form: values at/above this are NaN
    __m256i nan_threshold_cmp_u8x32 = _mm256_set1_epi8((char)0xFD);
    __m256i first_i8x32 = _mm256_loadu_si256((__m256i const *)data);
    __m256i first_cmp = nk_fp8x32_to_u8x32_comparable_haswell_(first_i8x32);
    nk_b256_vec_t min_vec, max_vec;
    min_vec.ymm = first_cmp;
    max_vec.ymm = first_cmp;
    __m256i min_iter_u8x32 = _mm256_setzero_si256();
    __m256i max_iter_u8x32 = _mm256_setzero_si256();
    __m256i iter_u8x32 = _mm256_set1_epi8(1);
    __m256i one_u8x32 = _mm256_set1_epi8(1);

    nk_size_t idx = 32;
    for (; idx + 32 <= count; idx += 32) {
        __m256i data_i8x32 = _mm256_loadu_si256((__m256i const *)(data + idx));
        __m256i data_cmp_u8x32 = nk_fp8x32_to_u8x32_comparable_haswell_(data_i8x32);
        __m256i min_mask_i8x32 = nk_min_mask_e5m2x32_haswell_(min_vec.ymm, data_cmp_u8x32, nan_threshold_cmp_u8x32);
        __m256i new_min = _mm256_blendv_epi8(data_cmp_u8x32, min_vec.ymm, min_mask_i8x32);
        __m256i min_changed_i8x32 = _mm256_xor_si256(_mm256_cmpeq_epi8(new_min, min_vec.ymm),
                                                     _mm256_set1_epi8((char)0xFF));
        min_vec.ymm = new_min;
        min_iter_u8x32 = _mm256_blendv_epi8(min_iter_u8x32, iter_u8x32, min_changed_i8x32);
        __m256i max_mask_i8x32 = nk_max_mask_e5m2x32_haswell_(max_vec.ymm, data_cmp_u8x32, nan_threshold_cmp_u8x32);
        __m256i new_max = _mm256_blendv_epi8(data_cmp_u8x32, max_vec.ymm, max_mask_i8x32);
        __m256i max_changed_i8x32 = _mm256_xor_si256(_mm256_cmpeq_epi8(new_max, max_vec.ymm),
                                                     _mm256_set1_epi8((char)0xFF));
        max_vec.ymm = new_max;
        max_iter_u8x32 = _mm256_blendv_epi8(max_iter_u8x32, iter_u8x32, max_changed_i8x32);
        iter_u8x32 = _mm256_add_epi8(iter_u8x32, one_u8x32);
    }

    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        nk_b256_vec_t tail_vec;
        tail_vec.ymm = _mm256_set1_epi8((char)0xFF);
        nk_partial_load_b8x32_serial_(data + idx, &tail_vec, remaining);
        for (nk_size_t i = remaining; i < 32; ++i) tail_vec.u8s[i] = 0xFF;
        __m256i data_cmp_u8x32 = nk_fp8x32_to_u8x32_comparable_haswell_(tail_vec.ymm);
        nk_b256_vec_t cmp_vec;
        cmp_vec.ymm = data_cmp_u8x32;
        for (nk_size_t i = remaining; i < 32; ++i) cmp_vec.u8s[i] = 0xFF;
        data_cmp_u8x32 = cmp_vec.ymm;
        __m256i min_mask_i8x32 = nk_min_mask_e5m2x32_haswell_(min_vec.ymm, data_cmp_u8x32, nan_threshold_cmp_u8x32);
        __m256i new_min = _mm256_blendv_epi8(data_cmp_u8x32, min_vec.ymm, min_mask_i8x32);
        __m256i min_changed_i8x32 = _mm256_xor_si256(_mm256_cmpeq_epi8(new_min, min_vec.ymm),
                                                     _mm256_set1_epi8((char)0xFF));
        min_vec.ymm = new_min;
        min_iter_u8x32 = _mm256_blendv_epi8(min_iter_u8x32, iter_u8x32, min_changed_i8x32);
        __m256i max_mask_i8x32 = nk_max_mask_e5m2x32_haswell_(max_vec.ymm, data_cmp_u8x32, nan_threshold_cmp_u8x32);
        __m256i new_max = _mm256_blendv_epi8(data_cmp_u8x32, max_vec.ymm, max_mask_i8x32);
        __m256i max_changed_i8x32 = _mm256_xor_si256(_mm256_cmpeq_epi8(new_max, max_vec.ymm),
                                                     _mm256_set1_epi8((char)0xFF));
        max_vec.ymm = new_max;
        max_iter_u8x32 = _mm256_blendv_epi8(max_iter_u8x32, iter_u8x32, max_changed_i8x32);
    }

    nk_size_t min_lane = nk_argmin_u8x32_haswell_(min_vec.ymm);
    nk_size_t max_lane = nk_argmax_u8x32_haswell_(max_vec.ymm);
    nk_b256_vec_t iter_vec;
    iter_vec.ymm = min_iter_u8x32;
    *min_index = (nk_size_t)iter_vec.u8s[min_lane] * 32 + min_lane;
    iter_vec.ymm = max_iter_u8x32;
    *max_index = (nk_size_t)iter_vec.u8s[max_lane] * 32 + max_lane;
    min_vec.ymm = nk_u8x32_comparable_to_fp8x32_haswell_(min_vec.ymm);
    max_vec.ymm = nk_u8x32_comparable_to_fp8x32_haswell_(max_vec.ymm);
    *min_value = min_vec.e5m2s[min_lane];
    *max_value = max_vec.e5m2s[max_lane];
}

NK_PUBLIC void nk_reduce_minmax_e5m2_haswell(                       //
    nk_e5m2_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_e5m2_t *min_value, nk_size_t *min_index,                     //
    nk_e5m2_t *max_value, nk_size_t *max_index) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_e5m2_t);
    if (count == 0)
        *min_value = NK_E5M2_MAX, *min_index = NK_SIZE_MAX, *max_value = NK_E5M2_MIN, *max_index = NK_SIZE_MAX;
    else if (stride_elements == 1 && count > (nk_size_t)(NK_U8_MAX + 1) * 32) {
        nk_size_t left_count = count / 2;
        nk_e5m2_t left_min, right_min, left_max, right_max;
        nk_size_t left_min_index, right_min_index, left_max_index, right_max_index;
        nk_reduce_minmax_e5m2_haswell(data, left_count, stride_bytes, &left_min, &left_min_index, &left_max,
                                      &left_max_index);
        nk_reduce_minmax_e5m2_haswell(data + left_count, count - left_count, stride_bytes, &right_min, &right_min_index,
                                      &right_max, &right_max_index);
        if (nk_e5m2_compare_(right_min, left_min) < 0)
            *min_value = right_min, *min_index = left_count + right_min_index;
        else *min_value = left_min, *min_index = left_min_index;
        if (nk_e5m2_compare_(right_max, left_max) > 0)
            *max_value = right_max, *max_index = left_count + right_max_index;
        else *max_value = left_max, *max_index = left_max_index;
    }
    else if (stride_elements == 1)
        nk_reduce_minmax_e5m2_haswell_contiguous_(data, count, min_value, min_index, max_value, max_index);
    else nk_reduce_minmax_e5m2_serial(data, count, stride_bytes, min_value, min_index, max_value, max_index);
}

NK_INTERNAL void nk_reduce_moments_e2m3_haswell_contiguous_( //
    nk_e2m3_t const *data, nk_size_t count,                  //
    nk_f32_t *sum, nk_f32_t *sumsq) {
    __m256 sum_f32x8 = _mm256_setzero_ps();
    __m256 sumsq_f32x8 = _mm256_setzero_ps();
    nk_size_t idx = 0;
    for (; idx + 8 <= count; idx += 8) {
        __m256 data_f32x8 = nk_e2m3x8_to_f32x8_haswell_(_mm_loadl_epi64((__m128i const *)(data + idx)));
        sum_f32x8 = _mm256_add_ps(sum_f32x8, data_f32x8);
        sumsq_f32x8 = _mm256_fmadd_ps(data_f32x8, data_f32x8, sumsq_f32x8);
    }
    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        nk_b256_vec_t vec;
        nk_partial_load_e2m3x8_to_f32x8_haswell_(data + idx, &vec, remaining);
        sum_f32x8 = _mm256_add_ps(sum_f32x8, vec.ymm_ps);
        sumsq_f32x8 = _mm256_fmadd_ps(vec.ymm_ps, vec.ymm_ps, sumsq_f32x8);
    }
    *sum = nk_reduce_add_f32x8_haswell_(sum_f32x8);
    *sumsq = nk_reduce_add_f32x8_haswell_(sumsq_f32x8);
}

NK_INTERNAL void nk_reduce_moments_e2m3_haswell_strided_(              //
    nk_e2m3_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_f32_t *sum, nk_f32_t *sumsq) {
    __m256 sum_f32x8 = _mm256_setzero_ps();
    __m256 sumsq_f32x8 = _mm256_setzero_ps();
    nk_size_t idx = 0;
    for (; idx + 8 <= count; idx += 8) {
        nk_b64_vec_t buf;
        buf.u64 = 0;
        nk_e2m3_t const *ptr = data + idx * stride_elements;
        for (nk_size_t i = 0; i < 8; ++i) buf.u8s[i] = ptr[i * stride_elements];
        __m256 data_f32x8 = nk_e2m3x8_to_f32x8_haswell_(_mm_cvtsi64_si128(buf.u64));
        sum_f32x8 = _mm256_add_ps(sum_f32x8, data_f32x8);
        sumsq_f32x8 = _mm256_fmadd_ps(data_f32x8, data_f32x8, sumsq_f32x8);
    }
    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        nk_b64_vec_t buf;
        buf.u64 = 0;
        nk_e2m3_t const *ptr = data + idx * stride_elements;
        for (nk_size_t i = 0; i < remaining; ++i) buf.u8s[i] = ptr[i * stride_elements];
        nk_b256_vec_t vec;
        nk_partial_load_e2m3x8_to_f32x8_haswell_((nk_e2m3_t *)&buf, &vec, remaining);
        sum_f32x8 = _mm256_add_ps(sum_f32x8, vec.ymm_ps);
        sumsq_f32x8 = _mm256_fmadd_ps(vec.ymm_ps, vec.ymm_ps, sumsq_f32x8);
    }
    *sum = nk_reduce_add_f32x8_haswell_(sum_f32x8);
    *sumsq = nk_reduce_add_f32x8_haswell_(sumsq_f32x8);
}

NK_PUBLIC void nk_reduce_moments_e2m3_haswell(                      //
    nk_e2m3_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *sum, nk_f32_t *sumsq) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_e2m3_t);
    int aligned = (stride_bytes % sizeof(nk_e2m3_t) == 0);
    if (count == 0) *sum = 0, *sumsq = 0;
    else if (!aligned) nk_reduce_moments_e2m3_serial(data, count, stride_bytes, sum, sumsq);
    else if (count > (nk_size_t)(NK_U16_MAX + 1) * 32) {
        nk_size_t left_count = count / 2;
        nk_f32_t left_sum, left_sumsq, right_sum, right_sumsq;
        nk_reduce_moments_e2m3_haswell(data, left_count, stride_bytes, &left_sum, &left_sumsq);
        nk_reduce_moments_e2m3_haswell(data + left_count * stride_elements, count - left_count, stride_bytes,
                                       &right_sum, &right_sumsq);
        *sum = left_sum + right_sum, *sumsq = left_sumsq + right_sumsq;
    }
    else if (stride_elements == 1) nk_reduce_moments_e2m3_haswell_contiguous_(data, count, sum, sumsq);
    else if (stride_elements >= 2 && stride_elements <= 8)
        nk_reduce_moments_e2m3_haswell_strided_(data, count, stride_elements, sum, sumsq);
    else nk_reduce_moments_e2m3_serial(data, count, stride_bytes, sum, sumsq);
}

NK_INTERNAL void nk_reduce_minmax_e2m3_haswell_contiguous_( //
    nk_e2m3_t const *data, nk_size_t count,                 //
    nk_e2m3_t *min_value, nk_size_t *min_index,             //
    nk_e2m3_t *max_value, nk_size_t *max_index) {
    // FP6 has no NaN — use simple unsigned min/max on comparable form
    __m256i first_i8x32 = _mm256_loadu_si256((__m256i const *)data);
    __m256i first_cmp = nk_fp6x32_to_u8x32_comparable_haswell_(first_i8x32);
    nk_b256_vec_t min_vec, max_vec;
    min_vec.ymm = first_cmp;
    max_vec.ymm = first_cmp;
    __m256i min_iter_u8x32 = _mm256_setzero_si256();
    __m256i max_iter_u8x32 = _mm256_setzero_si256();
    __m256i iter_u8x32 = _mm256_set1_epi8(1);
    __m256i one_u8x32 = _mm256_set1_epi8(1);

    nk_size_t idx = 32;
    for (; idx + 32 <= count; idx += 32) {
        __m256i data_i8x32 = _mm256_loadu_si256((__m256i const *)(data + idx));
        __m256i data_cmp_u8x32 = nk_fp6x32_to_u8x32_comparable_haswell_(data_i8x32);
        __m256i new_min = _mm256_min_epu8(min_vec.ymm, data_cmp_u8x32);
        __m256i min_changed_i8x32 = _mm256_xor_si256(_mm256_cmpeq_epi8(new_min, min_vec.ymm),
                                                     _mm256_set1_epi8((char)0xFF));
        min_vec.ymm = new_min;
        min_iter_u8x32 = _mm256_blendv_epi8(min_iter_u8x32, iter_u8x32, min_changed_i8x32);
        __m256i new_max = _mm256_max_epu8(max_vec.ymm, data_cmp_u8x32);
        __m256i max_changed_i8x32 = _mm256_xor_si256(_mm256_cmpeq_epi8(new_max, max_vec.ymm),
                                                     _mm256_set1_epi8((char)0xFF));
        max_vec.ymm = new_max;
        max_iter_u8x32 = _mm256_blendv_epi8(max_iter_u8x32, iter_u8x32, max_changed_i8x32);
        iter_u8x32 = _mm256_add_epi8(iter_u8x32, one_u8x32);
    }

    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        nk_b256_vec_t tail_vec;
        nk_partial_load_b8x32_serial_(data + idx, &tail_vec, remaining);
        __m256i data_cmp_u8x32 = nk_fp6x32_to_u8x32_comparable_haswell_(tail_vec.ymm);
        // Fill invalid lanes with identity: 0x3F for min (max comparable), 0x00 for max (min comparable)
        nk_b256_vec_t mask_vec;
        mask_vec.ymm = _mm256_setzero_si256();
        for (nk_size_t i = 0; i < remaining; ++i) mask_vec.u8s[i] = 0xFF;
        __m256i valid_b8x32 = mask_vec.ymm;
        __m256i data_min_u8x32 = _mm256_blendv_epi8(_mm256_set1_epi8(0x3F), data_cmp_u8x32, valid_b8x32);
        __m256i data_max_u8x32 = _mm256_blendv_epi8(_mm256_setzero_si256(), data_cmp_u8x32, valid_b8x32);
        __m256i new_min = _mm256_min_epu8(min_vec.ymm, data_min_u8x32);
        __m256i min_changed_i8x32 = _mm256_xor_si256(_mm256_cmpeq_epi8(new_min, min_vec.ymm),
                                                     _mm256_set1_epi8((char)0xFF));
        min_vec.ymm = new_min;
        min_iter_u8x32 = _mm256_blendv_epi8(min_iter_u8x32, iter_u8x32, min_changed_i8x32);
        __m256i new_max = _mm256_max_epu8(max_vec.ymm, data_max_u8x32);
        __m256i max_changed_i8x32 = _mm256_xor_si256(_mm256_cmpeq_epi8(new_max, max_vec.ymm),
                                                     _mm256_set1_epi8((char)0xFF));
        max_vec.ymm = new_max;
        max_iter_u8x32 = _mm256_blendv_epi8(max_iter_u8x32, iter_u8x32, max_changed_i8x32);
    }

    nk_size_t min_lane = nk_argmin_u8x32_haswell_(min_vec.ymm);
    nk_size_t max_lane = nk_argmax_u8x32_haswell_(max_vec.ymm);
    nk_b256_vec_t iter_vec;
    iter_vec.ymm = min_iter_u8x32;
    *min_index = (nk_size_t)iter_vec.u8s[min_lane] * 32 + min_lane;
    iter_vec.ymm = max_iter_u8x32;
    *max_index = (nk_size_t)iter_vec.u8s[max_lane] * 32 + max_lane;
    min_vec.ymm = nk_u8x32_comparable_to_fp6x32_haswell_(min_vec.ymm);
    max_vec.ymm = nk_u8x32_comparable_to_fp6x32_haswell_(max_vec.ymm);
    *min_value = min_vec.e2m3s[min_lane];
    *max_value = max_vec.e2m3s[max_lane];
}

NK_PUBLIC void nk_reduce_minmax_e2m3_haswell(                       //
    nk_e2m3_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_e2m3_t *min_value, nk_size_t *min_index,                     //
    nk_e2m3_t *max_value, nk_size_t *max_index) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_e2m3_t);
    if (count == 0)
        *min_value = NK_E2M3_MAX, *min_index = NK_SIZE_MAX, *max_value = NK_E2M3_MIN, *max_index = NK_SIZE_MAX;
    else if (stride_elements == 1 && count > (nk_size_t)(NK_U8_MAX + 1) * 32) {
        nk_size_t left_count = count / 2;
        nk_e2m3_t left_min, right_min, left_max, right_max;
        nk_size_t left_min_index, right_min_index, left_max_index, right_max_index;
        nk_reduce_minmax_e2m3_haswell(data, left_count, stride_bytes, &left_min, &left_min_index, &left_max,
                                      &left_max_index);
        nk_reduce_minmax_e2m3_haswell(data + left_count, count - left_count, stride_bytes, &right_min, &right_min_index,
                                      &right_max, &right_max_index);
        if (nk_e2m3_compare_(right_min, left_min) < 0)
            *min_value = right_min, *min_index = left_count + right_min_index;
        else *min_value = left_min, *min_index = left_min_index;
        if (nk_e2m3_compare_(right_max, left_max) > 0)
            *max_value = right_max, *max_index = left_count + right_max_index;
        else *max_value = left_max, *max_index = left_max_index;
    }
    else if (stride_elements == 1)
        nk_reduce_minmax_e2m3_haswell_contiguous_(data, count, min_value, min_index, max_value, max_index);
    else nk_reduce_minmax_e2m3_serial(data, count, stride_bytes, min_value, min_index, max_value, max_index);
}

NK_INTERNAL void nk_reduce_moments_e3m2_haswell_contiguous_( //
    nk_e3m2_t const *data, nk_size_t count,                  //
    nk_f32_t *sum, nk_f32_t *sumsq) {
    __m256 sum_f32x8 = _mm256_setzero_ps();
    __m256 sumsq_f32x8 = _mm256_setzero_ps();
    nk_size_t idx = 0;
    for (; idx + 8 <= count; idx += 8) {
        __m256 data_f32x8 = nk_e3m2x8_to_f32x8_haswell_(_mm_loadl_epi64((__m128i const *)(data + idx)));
        sum_f32x8 = _mm256_add_ps(sum_f32x8, data_f32x8);
        sumsq_f32x8 = _mm256_fmadd_ps(data_f32x8, data_f32x8, sumsq_f32x8);
    }
    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        nk_b256_vec_t vec;
        nk_partial_load_e3m2x8_to_f32x8_haswell_(data + idx, &vec, remaining);
        sum_f32x8 = _mm256_add_ps(sum_f32x8, vec.ymm_ps);
        sumsq_f32x8 = _mm256_fmadd_ps(vec.ymm_ps, vec.ymm_ps, sumsq_f32x8);
    }
    *sum = nk_reduce_add_f32x8_haswell_(sum_f32x8);
    *sumsq = nk_reduce_add_f32x8_haswell_(sumsq_f32x8);
}

NK_INTERNAL void nk_reduce_moments_e3m2_haswell_strided_(              //
    nk_e3m2_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_f32_t *sum, nk_f32_t *sumsq) {
    __m256 sum_f32x8 = _mm256_setzero_ps();
    __m256 sumsq_f32x8 = _mm256_setzero_ps();
    nk_size_t idx = 0;
    for (; idx + 8 <= count; idx += 8) {
        nk_b64_vec_t buf;
        buf.u64 = 0;
        nk_e3m2_t const *ptr = data + idx * stride_elements;
        for (nk_size_t i = 0; i < 8; ++i) buf.u8s[i] = ptr[i * stride_elements];
        __m256 data_f32x8 = nk_e3m2x8_to_f32x8_haswell_(_mm_cvtsi64_si128(buf.u64));
        sum_f32x8 = _mm256_add_ps(sum_f32x8, data_f32x8);
        sumsq_f32x8 = _mm256_fmadd_ps(data_f32x8, data_f32x8, sumsq_f32x8);
    }
    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        nk_b64_vec_t buf;
        buf.u64 = 0;
        nk_e3m2_t const *ptr = data + idx * stride_elements;
        for (nk_size_t i = 0; i < remaining; ++i) buf.u8s[i] = ptr[i * stride_elements];
        nk_b256_vec_t vec;
        nk_partial_load_e3m2x8_to_f32x8_haswell_((nk_e3m2_t *)&buf, &vec, remaining);
        sum_f32x8 = _mm256_add_ps(sum_f32x8, vec.ymm_ps);
        sumsq_f32x8 = _mm256_fmadd_ps(vec.ymm_ps, vec.ymm_ps, sumsq_f32x8);
    }
    *sum = nk_reduce_add_f32x8_haswell_(sum_f32x8);
    *sumsq = nk_reduce_add_f32x8_haswell_(sumsq_f32x8);
}

NK_PUBLIC void nk_reduce_moments_e3m2_haswell(                      //
    nk_e3m2_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *sum, nk_f32_t *sumsq) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_e3m2_t);
    int aligned = (stride_bytes % sizeof(nk_e3m2_t) == 0);
    if (count == 0) *sum = 0, *sumsq = 0;
    else if (!aligned) nk_reduce_moments_e3m2_serial(data, count, stride_bytes, sum, sumsq);
    else if (count > (nk_size_t)(NK_U16_MAX + 1) * 32) {
        nk_size_t left_count = count / 2;
        nk_f32_t left_sum, left_sumsq, right_sum, right_sumsq;
        nk_reduce_moments_e3m2_haswell(data, left_count, stride_bytes, &left_sum, &left_sumsq);
        nk_reduce_moments_e3m2_haswell(data + left_count * stride_elements, count - left_count, stride_bytes,
                                       &right_sum, &right_sumsq);
        *sum = left_sum + right_sum, *sumsq = left_sumsq + right_sumsq;
    }
    else if (stride_elements == 1) nk_reduce_moments_e3m2_haswell_contiguous_(data, count, sum, sumsq);
    else if (stride_elements >= 2 && stride_elements <= 8)
        nk_reduce_moments_e3m2_haswell_strided_(data, count, stride_elements, sum, sumsq);
    else nk_reduce_moments_e3m2_serial(data, count, stride_bytes, sum, sumsq);
}

NK_INTERNAL void nk_reduce_minmax_e3m2_haswell_contiguous_( //
    nk_e3m2_t const *data, nk_size_t count,                 //
    nk_e3m2_t *min_value, nk_size_t *min_index,             //
    nk_e3m2_t *max_value, nk_size_t *max_index) {
    __m256i first_i8x32 = _mm256_loadu_si256((__m256i const *)data);
    __m256i first_cmp = nk_fp6x32_to_u8x32_comparable_haswell_(first_i8x32);
    nk_b256_vec_t min_vec, max_vec;
    min_vec.ymm = first_cmp;
    max_vec.ymm = first_cmp;
    __m256i min_iter_u8x32 = _mm256_setzero_si256();
    __m256i max_iter_u8x32 = _mm256_setzero_si256();
    __m256i iter_u8x32 = _mm256_set1_epi8(1);
    __m256i one_u8x32 = _mm256_set1_epi8(1);

    nk_size_t idx = 32;
    for (; idx + 32 <= count; idx += 32) {
        __m256i data_i8x32 = _mm256_loadu_si256((__m256i const *)(data + idx));
        __m256i data_cmp_u8x32 = nk_fp6x32_to_u8x32_comparable_haswell_(data_i8x32);
        __m256i new_min = _mm256_min_epu8(min_vec.ymm, data_cmp_u8x32);
        __m256i min_changed_i8x32 = _mm256_xor_si256(_mm256_cmpeq_epi8(new_min, min_vec.ymm),
                                                     _mm256_set1_epi8((char)0xFF));
        min_vec.ymm = new_min;
        min_iter_u8x32 = _mm256_blendv_epi8(min_iter_u8x32, iter_u8x32, min_changed_i8x32);
        __m256i new_max = _mm256_max_epu8(max_vec.ymm, data_cmp_u8x32);
        __m256i max_changed_i8x32 = _mm256_xor_si256(_mm256_cmpeq_epi8(new_max, max_vec.ymm),
                                                     _mm256_set1_epi8((char)0xFF));
        max_vec.ymm = new_max;
        max_iter_u8x32 = _mm256_blendv_epi8(max_iter_u8x32, iter_u8x32, max_changed_i8x32);
        iter_u8x32 = _mm256_add_epi8(iter_u8x32, one_u8x32);
    }

    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        nk_b256_vec_t tail_vec;
        nk_partial_load_b8x32_serial_(data + idx, &tail_vec, remaining);
        __m256i data_cmp_u8x32 = nk_fp6x32_to_u8x32_comparable_haswell_(tail_vec.ymm);
        nk_b256_vec_t mask_vec;
        mask_vec.ymm = _mm256_setzero_si256();
        for (nk_size_t i = 0; i < remaining; ++i) mask_vec.u8s[i] = 0xFF;
        __m256i valid_b8x32 = mask_vec.ymm;
        __m256i data_min_u8x32 = _mm256_blendv_epi8(_mm256_set1_epi8(0x3F), data_cmp_u8x32, valid_b8x32);
        __m256i data_max_u8x32 = _mm256_blendv_epi8(_mm256_setzero_si256(), data_cmp_u8x32, valid_b8x32);
        __m256i new_min = _mm256_min_epu8(min_vec.ymm, data_min_u8x32);
        __m256i min_changed_i8x32 = _mm256_xor_si256(_mm256_cmpeq_epi8(new_min, min_vec.ymm),
                                                     _mm256_set1_epi8((char)0xFF));
        min_vec.ymm = new_min;
        min_iter_u8x32 = _mm256_blendv_epi8(min_iter_u8x32, iter_u8x32, min_changed_i8x32);
        __m256i new_max = _mm256_max_epu8(max_vec.ymm, data_max_u8x32);
        __m256i max_changed_i8x32 = _mm256_xor_si256(_mm256_cmpeq_epi8(new_max, max_vec.ymm),
                                                     _mm256_set1_epi8((char)0xFF));
        max_vec.ymm = new_max;
        max_iter_u8x32 = _mm256_blendv_epi8(max_iter_u8x32, iter_u8x32, max_changed_i8x32);
    }

    nk_size_t min_lane = nk_argmin_u8x32_haswell_(min_vec.ymm);
    nk_size_t max_lane = nk_argmax_u8x32_haswell_(max_vec.ymm);
    nk_b256_vec_t iter_vec;
    iter_vec.ymm = min_iter_u8x32;
    *min_index = (nk_size_t)iter_vec.u8s[min_lane] * 32 + min_lane;
    iter_vec.ymm = max_iter_u8x32;
    *max_index = (nk_size_t)iter_vec.u8s[max_lane] * 32 + max_lane;
    min_vec.ymm = nk_u8x32_comparable_to_fp6x32_haswell_(min_vec.ymm);
    max_vec.ymm = nk_u8x32_comparable_to_fp6x32_haswell_(max_vec.ymm);
    *min_value = min_vec.e3m2s[min_lane];
    *max_value = max_vec.e3m2s[max_lane];
}

NK_PUBLIC void nk_reduce_minmax_e3m2_haswell(                       //
    nk_e3m2_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_e3m2_t *min_value, nk_size_t *min_index,                     //
    nk_e3m2_t *max_value, nk_size_t *max_index) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_e3m2_t);
    if (count == 0)
        *min_value = NK_E3M2_MAX, *min_index = NK_SIZE_MAX, *max_value = NK_E3M2_MIN, *max_index = NK_SIZE_MAX;
    else if (stride_elements == 1 && count > (nk_size_t)(NK_U8_MAX + 1) * 32) {
        nk_size_t left_count = count / 2;
        nk_e3m2_t left_min, right_min, left_max, right_max;
        nk_size_t left_min_index, right_min_index, left_max_index, right_max_index;
        nk_reduce_minmax_e3m2_haswell(data, left_count, stride_bytes, &left_min, &left_min_index, &left_max,
                                      &left_max_index);
        nk_reduce_minmax_e3m2_haswell(data + left_count, count - left_count, stride_bytes, &right_min, &right_min_index,
                                      &right_max, &right_max_index);
        if (nk_e3m2_compare_(right_min, left_min) < 0)
            *min_value = right_min, *min_index = left_count + right_min_index;
        else *min_value = left_min, *min_index = left_min_index;
        if (nk_e3m2_compare_(right_max, left_max) > 0)
            *max_value = right_max, *max_index = left_count + right_max_index;
        else *max_value = left_max, *max_index = left_max_index;
    }
    else if (stride_elements == 1)
        nk_reduce_minmax_e3m2_haswell_contiguous_(data, count, min_value, min_index, max_value, max_index);
    else nk_reduce_minmax_e3m2_serial(data, count, stride_bytes, min_value, min_index, max_value, max_index);
}

NK_INTERNAL void nk_reduce_moments_bf16_haswell_contiguous_( //
    nk_bf16_t const *data, nk_size_t count,                  //
    nk_f32_t *sum, nk_f32_t *sumsq) {
    __m256 sum_f32x8 = _mm256_setzero_ps();
    __m256 sumsq_f32x8 = _mm256_setzero_ps();
    nk_size_t idx = 0;
    for (; idx + 16 <= count; idx += 16) {
        __m256 low_f32x8 = _mm256_castsi256_ps(
            _mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((__m128i const *)(data + idx))), 16));
        __m256 high_f32x8 = _mm256_castsi256_ps(
            _mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((__m128i const *)(data + idx + 8))), 16));
        sum_f32x8 = _mm256_add_ps(sum_f32x8, low_f32x8);
        sum_f32x8 = _mm256_add_ps(sum_f32x8, high_f32x8);
        sumsq_f32x8 = _mm256_fmadd_ps(low_f32x8, low_f32x8, sumsq_f32x8);
        sumsq_f32x8 = _mm256_fmadd_ps(high_f32x8, high_f32x8, sumsq_f32x8);
    }
    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        nk_b256_vec_t partial_vec;
        nk_partial_load_b16x16_serial_(data + idx, &partial_vec, remaining);
        nk_size_t first_half = remaining > 8 ? 8 : remaining;
        (void)first_half;
        __m256 low_f32x8 = _mm256_castsi256_ps(
            _mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(partial_vec.ymm)), 16));
        sum_f32x8 = _mm256_add_ps(sum_f32x8, low_f32x8);
        sumsq_f32x8 = _mm256_fmadd_ps(low_f32x8, low_f32x8, sumsq_f32x8);
        if (remaining > 8) {
            __m256 high_f32x8 = _mm256_castsi256_ps(
                _mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm256_extracti128_si256(partial_vec.ymm, 1)), 16));
            sum_f32x8 = _mm256_add_ps(sum_f32x8, high_f32x8);
            sumsq_f32x8 = _mm256_fmadd_ps(high_f32x8, high_f32x8, sumsq_f32x8);
        }
    }
    *sum = nk_reduce_add_f32x8_haswell_(sum_f32x8);
    *sumsq = nk_reduce_add_f32x8_haswell_(sumsq_f32x8);
}

NK_PUBLIC void nk_reduce_moments_bf16_haswell(                      //
    nk_bf16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *sum, nk_f32_t *sumsq) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_bf16_t);
    int aligned = (stride_bytes % sizeof(nk_bf16_t) == 0);
    if (count == 0) *sum = 0, *sumsq = 0;
    else if (!aligned) nk_reduce_moments_bf16_serial(data, count, stride_bytes, sum, sumsq);
    else if (count > (nk_size_t)(NK_U16_MAX + 1) * 16) {
        nk_size_t left_count = count / 2;
        nk_f32_t left_sum, left_sumsq, right_sum, right_sumsq;
        nk_reduce_moments_bf16_haswell(data, left_count, stride_bytes, &left_sum, &left_sumsq);
        nk_reduce_moments_bf16_haswell(data + left_count * stride_elements, count - left_count, stride_bytes,
                                       &right_sum, &right_sumsq);
        *sum = left_sum + right_sum;
        *sumsq = left_sumsq + right_sumsq;
    }
    else if (stride_elements == 1) nk_reduce_moments_bf16_haswell_contiguous_(data, count, sum, sumsq);
    else nk_reduce_moments_bf16_serial(data, count, stride_bytes, sum, sumsq);
}

NK_INTERNAL void nk_reduce_minmax_bf16_haswell_contiguous_( //
    nk_bf16_t const *data, nk_size_t count,                 //
    nk_bf16_t *min_value, nk_size_t *min_index,             //
    nk_bf16_t *max_value, nk_size_t *max_index) {
    __m256i abs_mask_u16x16 = _mm256_set1_epi16(0x7FFF);
    __m256i nan_threshold_u16x16 = _mm256_set1_epi16((short)0x7F80);
    __m256i all_ones_i16x16 = _mm256_set1_epi8((char)0xFF);
    __m256i min_cmp_i16x16 = _mm256_set1_epi16((short)0x7FFF);
    __m256i max_cmp_i16x16 = _mm256_set1_epi16((short)0x8000);
    __m256i min_iter_u16x16 = _mm256_setzero_si256();
    __m256i max_iter_u16x16 = _mm256_setzero_si256();
    __m256i iter_u16x16 = _mm256_setzero_si256();
    __m256i one_u16x16 = _mm256_set1_epi16(1);

    nk_size_t idx = 0;
    for (; idx + 16 <= count; idx += 16) {
        __m256i raw_u16x16 = _mm256_loadu_si256((__m256i const *)(data + idx));
        __m256i data_cmp_i16x16 = nk_bf16x16_to_comparable_i16x16_haswell_(raw_u16x16);
        __m256i abs_u16x16 = _mm256_and_si256(raw_u16x16, abs_mask_u16x16);
        __m256i nan_detect_i16x16 = _mm256_cmpgt_epi16(abs_u16x16, nan_threshold_u16x16);
        __m256i not_nan_i16x16 = _mm256_xor_si256(nan_detect_i16x16, all_ones_i16x16);
        __m256i less_i16x16 = _mm256_cmpgt_epi16(min_cmp_i16x16, data_cmp_i16x16);
        __m256i min_changed_i16x16 = _mm256_and_si256(less_i16x16, not_nan_i16x16);
        min_cmp_i16x16 = _mm256_blendv_epi8(min_cmp_i16x16, data_cmp_i16x16, min_changed_i16x16);
        min_iter_u16x16 = _mm256_blendv_epi8(min_iter_u16x16, iter_u16x16, min_changed_i16x16);
        __m256i greater_i16x16 = _mm256_cmpgt_epi16(data_cmp_i16x16, max_cmp_i16x16);
        __m256i max_changed_i16x16 = _mm256_and_si256(greater_i16x16, not_nan_i16x16);
        max_cmp_i16x16 = _mm256_blendv_epi8(max_cmp_i16x16, data_cmp_i16x16, max_changed_i16x16);
        max_iter_u16x16 = _mm256_blendv_epi8(max_iter_u16x16, iter_u16x16, max_changed_i16x16);
        iter_u16x16 = _mm256_add_epi16(iter_u16x16, one_u16x16);
    }

    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        nk_b256_vec_t partial_vec;
        nk_partial_load_b16x16_serial_(data + idx, &partial_vec, remaining);
        __m256i raw_u16x16 = partial_vec.ymm;
        __m256i data_cmp_i16x16 = nk_bf16x16_to_comparable_i16x16_haswell_(raw_u16x16);
        __m256i abs_u16x16 = _mm256_and_si256(raw_u16x16, abs_mask_u16x16);
        __m256i nan_detect_i16x16 = _mm256_cmpgt_epi16(abs_u16x16, nan_threshold_u16x16);
        nk_b256_vec_t valid_mask_vec;
        valid_mask_vec.ymm = _mm256_setzero_si256();
        for (nk_size_t i = 0; i < remaining; i++) valid_mask_vec.u16s[i] = 0xFFFF;
        __m256i valid_i16x16 = valid_mask_vec.ymm;
        __m256i not_nan_valid_i16x16 = _mm256_andnot_si256(nan_detect_i16x16, valid_i16x16);
        __m256i less_i16x16 = _mm256_cmpgt_epi16(min_cmp_i16x16, data_cmp_i16x16);
        __m256i min_changed_i16x16 = _mm256_and_si256(less_i16x16, not_nan_valid_i16x16);
        min_cmp_i16x16 = _mm256_blendv_epi8(min_cmp_i16x16, data_cmp_i16x16, min_changed_i16x16);
        min_iter_u16x16 = _mm256_blendv_epi8(min_iter_u16x16, iter_u16x16, min_changed_i16x16);
        __m256i greater_i16x16 = _mm256_cmpgt_epi16(data_cmp_i16x16, max_cmp_i16x16);
        __m256i max_changed_i16x16 = _mm256_and_si256(greater_i16x16, not_nan_valid_i16x16);
        max_cmp_i16x16 = _mm256_blendv_epi8(max_cmp_i16x16, data_cmp_i16x16, max_changed_i16x16);
        max_iter_u16x16 = _mm256_blendv_epi8(max_iter_u16x16, iter_u16x16, max_changed_i16x16);
    }

    nk_i16_t min_scalar_cmp = nk_reduce_min_i16x16_haswell_(min_cmp_i16x16);
    nk_i16_t max_scalar_cmp = nk_reduce_max_i16x16_haswell_(max_cmp_i16x16);
    __m256i min_eq_i16x16 = _mm256_cmpeq_epi16(min_cmp_i16x16, _mm256_set1_epi16(min_scalar_cmp));
    __m256i max_eq_i16x16 = _mm256_cmpeq_epi16(max_cmp_i16x16, _mm256_set1_epi16(max_scalar_cmp));
    unsigned int min_lane = _tzcnt_u32((unsigned int)_mm256_movemask_epi8(min_eq_i16x16)) / 2;
    unsigned int max_lane = _tzcnt_u32((unsigned int)_mm256_movemask_epi8(max_eq_i16x16)) / 2;
    nk_b256_vec_t iter_vec;
    iter_vec.ymm = min_iter_u16x16;
    *min_index = (nk_size_t)iter_vec.u16s[min_lane] * 16 + min_lane;
    iter_vec.ymm = max_iter_u16x16;
    *max_index = (nk_size_t)iter_vec.u16s[max_lane] * 16 + max_lane;
    nk_i16_t min_sign = min_scalar_cmp >> 15;
    *min_value = (nk_bf16_t)((nk_u16_t)min_scalar_cmp ^ ((nk_u16_t)min_sign >> 1));
    nk_i16_t max_sign = max_scalar_cmp >> 15;
    *max_value = (nk_bf16_t)((nk_u16_t)max_scalar_cmp ^ ((nk_u16_t)max_sign >> 1));
}

NK_PUBLIC void nk_reduce_minmax_bf16_haswell(                       //
    nk_bf16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_bf16_t *min_value, nk_size_t *min_index,                     //
    nk_bf16_t *max_value, nk_size_t *max_index) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_bf16_t);
    int aligned = (stride_bytes % sizeof(nk_bf16_t) == 0);
    if (count == 0)
        *min_value = NK_BF16_MAX, *min_index = NK_SIZE_MAX, *max_value = NK_BF16_MIN, *max_index = NK_SIZE_MAX;
    else if (!aligned)
        nk_reduce_minmax_bf16_serial(data, count, stride_bytes, min_value, min_index, max_value, max_index);
    else if (stride_elements == 1 && count > (nk_size_t)(NK_U16_MAX + 1) * 16) {
        nk_size_t left_count = count / 2;
        nk_bf16_t left_min, right_min, left_max, right_max;
        nk_size_t left_min_index, right_min_index, left_max_index, right_max_index;
        nk_reduce_minmax_bf16_haswell(data, left_count, stride_bytes, &left_min, &left_min_index, &left_max,
                                      &left_max_index);
        nk_reduce_minmax_bf16_haswell(data + left_count, count - left_count, stride_bytes, &right_min, &right_min_index,
                                      &right_max, &right_max_index);
        if (nk_bf16_compare_(right_min, left_min) < 0)
            *min_value = right_min, *min_index = left_count + right_min_index;
        else *min_value = left_min, *min_index = left_min_index;
        if (nk_bf16_compare_(right_max, left_max) > 0)
            *max_value = right_max, *max_index = left_count + right_max_index;
        else *max_value = left_max, *max_index = left_max_index;
    }
    else if (stride_elements == 1)
        nk_reduce_minmax_bf16_haswell_contiguous_(data, count, min_value, min_index, max_value, max_index);
    else nk_reduce_minmax_bf16_serial(data, count, stride_bytes, min_value, min_index, max_value, max_index);
}

NK_INTERNAL void nk_reduce_moments_f16_haswell_contiguous_( //
    nk_f16_t const *data, nk_size_t count,                  //
    nk_f32_t *sum, nk_f32_t *sumsq) {
    __m256 sum_f32x8 = _mm256_setzero_ps();
    __m256 sumsq_f32x8 = _mm256_setzero_ps();
    nk_size_t idx = 0;
    for (; idx + 16 <= count; idx += 16) {
        __m256 low_f32x8 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i const *)(data + idx)));
        __m256 high_f32x8 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i const *)(data + idx + 8)));
        sum_f32x8 = _mm256_add_ps(sum_f32x8, low_f32x8);
        sum_f32x8 = _mm256_add_ps(sum_f32x8, high_f32x8);
        sumsq_f32x8 = _mm256_fmadd_ps(low_f32x8, low_f32x8, sumsq_f32x8);
        sumsq_f32x8 = _mm256_fmadd_ps(high_f32x8, high_f32x8, sumsq_f32x8);
    }
    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        nk_b256_vec_t partial_vec;
        nk_partial_load_b16x16_serial_(data + idx, &partial_vec, remaining);
        __m256 low_f32x8 = _mm256_cvtph_ps(_mm256_castsi256_si128(partial_vec.ymm));
        sum_f32x8 = _mm256_add_ps(sum_f32x8, low_f32x8);
        sumsq_f32x8 = _mm256_fmadd_ps(low_f32x8, low_f32x8, sumsq_f32x8);
        if (remaining > 8) {
            __m256 high_f32x8 = _mm256_cvtph_ps(_mm256_extracti128_si256(partial_vec.ymm, 1));
            sum_f32x8 = _mm256_add_ps(sum_f32x8, high_f32x8);
            sumsq_f32x8 = _mm256_fmadd_ps(high_f32x8, high_f32x8, sumsq_f32x8);
        }
    }
    *sum = nk_reduce_add_f32x8_haswell_(sum_f32x8);
    *sumsq = nk_reduce_add_f32x8_haswell_(sumsq_f32x8);
}

NK_PUBLIC void nk_reduce_moments_f16_haswell(                      //
    nk_f16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *sum, nk_f32_t *sumsq) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_f16_t);
    int aligned = (stride_bytes % sizeof(nk_f16_t) == 0);
    if (count == 0) *sum = 0, *sumsq = 0;
    else if (!aligned) nk_reduce_moments_f16_serial(data, count, stride_bytes, sum, sumsq);
    else if (count > (nk_size_t)(NK_U16_MAX + 1) * 16) {
        nk_size_t left_count = count / 2;
        nk_f32_t left_sum, left_sumsq, right_sum, right_sumsq;
        nk_reduce_moments_f16_haswell(data, left_count, stride_bytes, &left_sum, &left_sumsq);
        nk_reduce_moments_f16_haswell(data + left_count * stride_elements, count - left_count, stride_bytes, &right_sum,
                                      &right_sumsq);
        *sum = left_sum + right_sum;
        *sumsq = left_sumsq + right_sumsq;
    }
    else if (stride_elements == 1) nk_reduce_moments_f16_haswell_contiguous_(data, count, sum, sumsq);
    else nk_reduce_moments_f16_serial(data, count, stride_bytes, sum, sumsq);
}

NK_INTERNAL void nk_reduce_minmax_f16_haswell_contiguous_( //
    nk_f16_t const *data, nk_size_t count,                 //
    nk_f16_t *min_value, nk_size_t *min_index,             //
    nk_f16_t *max_value, nk_size_t *max_index) {
    __m256i abs_mask_u16x16 = _mm256_set1_epi16(0x7FFF);
    __m256i nan_threshold_u16x16 = _mm256_set1_epi16((short)0x7C00);
    __m256i all_ones_i16x16 = _mm256_set1_epi8((char)0xFF);
    __m256i min_cmp_i16x16 = _mm256_set1_epi16((short)0x7FFF);
    __m256i max_cmp_i16x16 = _mm256_set1_epi16((short)0x8000);
    __m256i min_iter_u16x16 = _mm256_setzero_si256();
    __m256i max_iter_u16x16 = _mm256_setzero_si256();
    __m256i iter_u16x16 = _mm256_setzero_si256();
    __m256i one_u16x16 = _mm256_set1_epi16(1);

    nk_size_t idx = 0;
    for (; idx + 16 <= count; idx += 16) {
        __m256i raw_u16x16 = _mm256_loadu_si256((__m256i const *)(data + idx));
        __m256i data_cmp_i16x16 = nk_f16x16_to_comparable_i16x16_haswell_(raw_u16x16);
        __m256i abs_u16x16 = _mm256_and_si256(raw_u16x16, abs_mask_u16x16);
        __m256i nan_detect_i16x16 = _mm256_cmpgt_epi16(abs_u16x16, nan_threshold_u16x16);
        __m256i not_nan_i16x16 = _mm256_xor_si256(nan_detect_i16x16, all_ones_i16x16);
        __m256i less_i16x16 = _mm256_cmpgt_epi16(min_cmp_i16x16, data_cmp_i16x16);
        __m256i min_changed_i16x16 = _mm256_and_si256(less_i16x16, not_nan_i16x16);
        min_cmp_i16x16 = _mm256_blendv_epi8(min_cmp_i16x16, data_cmp_i16x16, min_changed_i16x16);
        min_iter_u16x16 = _mm256_blendv_epi8(min_iter_u16x16, iter_u16x16, min_changed_i16x16);
        __m256i greater_i16x16 = _mm256_cmpgt_epi16(data_cmp_i16x16, max_cmp_i16x16);
        __m256i max_changed_i16x16 = _mm256_and_si256(greater_i16x16, not_nan_i16x16);
        max_cmp_i16x16 = _mm256_blendv_epi8(max_cmp_i16x16, data_cmp_i16x16, max_changed_i16x16);
        max_iter_u16x16 = _mm256_blendv_epi8(max_iter_u16x16, iter_u16x16, max_changed_i16x16);
        iter_u16x16 = _mm256_add_epi16(iter_u16x16, one_u16x16);
    }

    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        nk_b256_vec_t partial_vec;
        nk_partial_load_b16x16_serial_(data + idx, &partial_vec, remaining);
        __m256i raw_u16x16 = partial_vec.ymm;
        __m256i data_cmp_i16x16 = nk_f16x16_to_comparable_i16x16_haswell_(raw_u16x16);
        __m256i abs_u16x16 = _mm256_and_si256(raw_u16x16, abs_mask_u16x16);
        __m256i nan_detect_i16x16 = _mm256_cmpgt_epi16(abs_u16x16, nan_threshold_u16x16);
        nk_b256_vec_t valid_mask_vec;
        valid_mask_vec.ymm = _mm256_setzero_si256();
        for (nk_size_t i = 0; i < remaining; i++) valid_mask_vec.u16s[i] = 0xFFFF;
        __m256i valid_i16x16 = valid_mask_vec.ymm;
        __m256i not_nan_valid_i16x16 = _mm256_andnot_si256(nan_detect_i16x16, valid_i16x16);
        __m256i less_i16x16 = _mm256_cmpgt_epi16(min_cmp_i16x16, data_cmp_i16x16);
        __m256i min_changed_i16x16 = _mm256_and_si256(less_i16x16, not_nan_valid_i16x16);
        min_cmp_i16x16 = _mm256_blendv_epi8(min_cmp_i16x16, data_cmp_i16x16, min_changed_i16x16);
        min_iter_u16x16 = _mm256_blendv_epi8(min_iter_u16x16, iter_u16x16, min_changed_i16x16);
        __m256i greater_i16x16 = _mm256_cmpgt_epi16(data_cmp_i16x16, max_cmp_i16x16);
        __m256i max_changed_i16x16 = _mm256_and_si256(greater_i16x16, not_nan_valid_i16x16);
        max_cmp_i16x16 = _mm256_blendv_epi8(max_cmp_i16x16, data_cmp_i16x16, max_changed_i16x16);
        max_iter_u16x16 = _mm256_blendv_epi8(max_iter_u16x16, iter_u16x16, max_changed_i16x16);
    }

    nk_i16_t min_scalar_cmp = nk_reduce_min_i16x16_haswell_(min_cmp_i16x16);
    nk_i16_t max_scalar_cmp = nk_reduce_max_i16x16_haswell_(max_cmp_i16x16);
    __m256i min_eq_i16x16 = _mm256_cmpeq_epi16(min_cmp_i16x16, _mm256_set1_epi16(min_scalar_cmp));
    __m256i max_eq_i16x16 = _mm256_cmpeq_epi16(max_cmp_i16x16, _mm256_set1_epi16(max_scalar_cmp));
    unsigned int min_lane = _tzcnt_u32((unsigned int)_mm256_movemask_epi8(min_eq_i16x16)) / 2;
    unsigned int max_lane = _tzcnt_u32((unsigned int)_mm256_movemask_epi8(max_eq_i16x16)) / 2;
    nk_b256_vec_t iter_vec;
    iter_vec.ymm = min_iter_u16x16;
    *min_index = (nk_size_t)iter_vec.u16s[min_lane] * 16 + min_lane;
    iter_vec.ymm = max_iter_u16x16;
    *max_index = (nk_size_t)iter_vec.u16s[max_lane] * 16 + max_lane;
    nk_i16_t min_sign = min_scalar_cmp >> 15;
    *min_value = (nk_f16_t)((nk_u16_t)min_scalar_cmp ^ ((nk_u16_t)min_sign >> 1));
    nk_i16_t max_sign = max_scalar_cmp >> 15;
    *max_value = (nk_f16_t)((nk_u16_t)max_scalar_cmp ^ ((nk_u16_t)max_sign >> 1));
}

NK_PUBLIC void nk_reduce_minmax_f16_haswell(                       //
    nk_f16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f16_t *min_value, nk_size_t *min_index,                     //
    nk_f16_t *max_value, nk_size_t *max_index) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_f16_t);
    int aligned = (stride_bytes % sizeof(nk_f16_t) == 0);
    if (count == 0)
        *min_value = NK_F16_MAX, *min_index = NK_SIZE_MAX, *max_value = NK_F16_MIN, *max_index = NK_SIZE_MAX;
    else if (!aligned)
        nk_reduce_minmax_f16_serial(data, count, stride_bytes, min_value, min_index, max_value, max_index);
    else if (stride_elements == 1 && count > (nk_size_t)(NK_U16_MAX + 1) * 16) {
        nk_size_t left_count = count / 2;
        nk_f16_t left_min, right_min, left_max, right_max;
        nk_size_t left_min_index, right_min_index, left_max_index, right_max_index;
        nk_reduce_minmax_f16_haswell(data, left_count, stride_bytes, &left_min, &left_min_index, &left_max,
                                     &left_max_index);
        nk_reduce_minmax_f16_haswell(data + left_count, count - left_count, stride_bytes, &right_min, &right_min_index,
                                     &right_max, &right_max_index);
        if (nk_f16_compare_(right_min, left_min) < 0) *min_value = right_min, *min_index = left_count + right_min_index;
        else *min_value = left_min, *min_index = left_min_index;
        if (nk_f16_compare_(right_max, left_max) > 0) *max_value = right_max, *max_index = left_count + right_max_index;
        else *max_value = left_max, *max_index = left_max_index;
    }
    else if (stride_elements == 1)
        nk_reduce_minmax_f16_haswell_contiguous_(data, count, min_value, min_index, max_value, max_index);
    else nk_reduce_minmax_f16_serial(data, count, stride_bytes, min_value, min_index, max_value, max_index);
}

NK_INTERNAL void nk_reduce_moments_i4_haswell_contiguous_( //
    nk_i4x2_t const *data, nk_size_t count,                //
    nk_i64_t *sum, nk_u64_t *sumsq) {
    __m256i mask_0f_i8x32 = _mm256_set1_epi8(0x0F);
    __m256i eight_i8x32 = _mm256_set1_epi8(8);
    __m256i zero_i8x32 = _mm256_setzero_si256();
    __m256i sq_lut_u8x32 = _mm256_setr_epi8(                                //
        0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, (char)144, (char)169, //
        (char)196, (char)225,                                               //
        0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, (char)144, (char)169, //
        (char)196, (char)225);
    __m256i sum_u64x4 = _mm256_setzero_si256();
    __m256i sumsq_u64x4 = _mm256_setzero_si256();
    nk_size_t count_nibbles = count;
    nk_size_t count_bytes = nk_size_divide_round_up_(count, 2);
    unsigned char const *ptr = (unsigned char const *)data;
    while (count_bytes > 0) {
        nk_b256_vec_t raw_vec;
        nk_size_t chunk;
        if (count_bytes < 32) {
            nk_partial_load_b8x32_serial_(ptr, &raw_vec, count_bytes);
            chunk = count_bytes;
            count_bytes = 0;
        }
        else {
            raw_vec.ymm = _mm256_loadu_si256((__m256i const *)ptr);
            chunk = 32;
            ptr += 32, count_bytes -= 32;
        }
        (void)chunk;
        __m256i raw_i8x32 = raw_vec.ymm;
        __m256i low_u4x32 = _mm256_and_si256(raw_i8x32, mask_0f_i8x32);
        __m256i high_u4x32 = _mm256_and_si256(_mm256_srli_epi16(raw_i8x32, 4), mask_0f_i8x32);
        __m256i low_biased_u4x32 = _mm256_xor_si256(low_u4x32, eight_i8x32);
        __m256i high_biased_u4x32 = _mm256_xor_si256(high_u4x32, eight_i8x32);
        __m256i pair_sum = _mm256_add_epi8(low_biased_u4x32, high_biased_u4x32);
        sum_u64x4 = _mm256_add_epi64(sum_u64x4, _mm256_sad_epu8(pair_sum, zero_i8x32));
        __m256i low_sq_u8x32 = _mm256_shuffle_epi8(sq_lut_u8x32, low_u4x32);
        __m256i high_sq_u8x32 = _mm256_shuffle_epi8(sq_lut_u8x32, high_u4x32);
        sumsq_u64x4 = _mm256_add_epi64(sumsq_u64x4, _mm256_sad_epu8(low_sq_u8x32, zero_i8x32));
        sumsq_u64x4 = _mm256_add_epi64(sumsq_u64x4, _mm256_sad_epu8(high_sq_u8x32, zero_i8x32));
    }
    nk_size_t nibbles_processed = nk_size_divide_round_up_(count_nibbles, 2) * 2;
    nk_i64_t s = (nk_i64_t)(nk_u64_t)nk_reduce_add_i64x4_haswell_(sum_u64x4) -
                 (nk_i64_t)8 * (nk_i64_t)nibbles_processed;
    if (count_nibbles & 1) {
        nk_u8_t last_byte = ((unsigned char const *)data)[count_nibbles / 2];
        nk_u8_t high_nib = (last_byte >> 4) & 0x0F;
        nk_i64_t signed_high = (nk_i64_t)((nk_i8_t)((high_nib ^ 8) - 8));
        s -= signed_high;
    }
    nk_u64_t sq = (nk_u64_t)nk_reduce_add_i64x4_haswell_(sumsq_u64x4);
    if (count_nibbles & 1) {
        nk_u8_t last_byte = ((unsigned char const *)data)[count_nibbles / 2];
        nk_u8_t high_nib = (last_byte >> 4) & 0x0F;
        nk_i64_t signed_high = (nk_i64_t)((nk_i8_t)((high_nib ^ 8) - 8));
        sq -= (nk_u64_t)(signed_high * signed_high);
    }
    *sum = s;
    *sumsq = sq;
}

NK_PUBLIC void nk_reduce_moments_i4_haswell(                        //
    nk_i4x2_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *sum, nk_u64_t *sumsq) {
    if (count == 0) *sum = 0, *sumsq = 0;
    else if (stride_bytes == 1) nk_reduce_moments_i4_haswell_contiguous_(data, count, sum, sumsq);
    else nk_reduce_moments_i4_serial(data, count, stride_bytes, sum, sumsq);
}

NK_INTERNAL void nk_reduce_moments_u4_haswell_contiguous_( //
    nk_u4x2_t const *data, nk_size_t count,                //
    nk_u64_t *sum, nk_u64_t *sumsq) {
    __m256i mask_0f_i8x32 = _mm256_set1_epi8(0x0F);
    __m256i zero_i8x32 = _mm256_setzero_si256();
    __m256i sq_lut_u8x32 = _mm256_setr_epi8(                                //
        0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, (char)144, (char)169, //
        (char)196, (char)225,                                               //
        0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, (char)144, (char)169, //
        (char)196, (char)225);
    __m256i sum_u64x4 = _mm256_setzero_si256();
    __m256i sumsq_u64x4 = _mm256_setzero_si256();
    nk_size_t count_nibbles = count;
    nk_size_t count_bytes = nk_size_divide_round_up_(count, 2);
    unsigned char const *ptr = (unsigned char const *)data;
    while (count_bytes > 0) {
        nk_b256_vec_t raw_vec;
        nk_size_t chunk;
        if (count_bytes < 32) {
            nk_partial_load_b8x32_serial_(ptr, &raw_vec, count_bytes);
            chunk = count_bytes;
            count_bytes = 0;
        }
        else {
            raw_vec.ymm = _mm256_loadu_si256((__m256i const *)ptr);
            chunk = 32;
            ptr += 32, count_bytes -= 32;
        }
        (void)chunk;
        __m256i raw_i8x32 = raw_vec.ymm;
        __m256i low_u4x32 = _mm256_and_si256(raw_i8x32, mask_0f_i8x32);
        __m256i high_u4x32 = _mm256_and_si256(_mm256_srli_epi16(raw_i8x32, 4), mask_0f_i8x32);
        __m256i pair_sum = _mm256_add_epi8(low_u4x32, high_u4x32);
        sum_u64x4 = _mm256_add_epi64(sum_u64x4, _mm256_sad_epu8(pair_sum, zero_i8x32));
        __m256i low_sq_u8x32 = _mm256_shuffle_epi8(sq_lut_u8x32, low_u4x32);
        __m256i high_sq_u8x32 = _mm256_shuffle_epi8(sq_lut_u8x32, high_u4x32);
        sumsq_u64x4 = _mm256_add_epi64(sumsq_u64x4, _mm256_sad_epu8(low_sq_u8x32, zero_i8x32));
        sumsq_u64x4 = _mm256_add_epi64(sumsq_u64x4, _mm256_sad_epu8(high_sq_u8x32, zero_i8x32));
    }
    nk_u64_t s = (nk_u64_t)nk_reduce_add_i64x4_haswell_(sum_u64x4);
    if (count_nibbles & 1) {
        nk_u8_t last_byte = ((unsigned char const *)data)[count_nibbles / 2];
        s -= (last_byte >> 4) & 0x0F;
    }
    nk_u64_t sq = (nk_u64_t)nk_reduce_add_i64x4_haswell_(sumsq_u64x4);
    if (count_nibbles & 1) {
        nk_u8_t last_byte = ((unsigned char const *)data)[count_nibbles / 2];
        nk_u8_t high_nib = (last_byte >> 4) & 0x0F;
        sq -= (nk_u64_t)high_nib * high_nib;
    }
    *sum = s;
    *sumsq = sq;
}

NK_PUBLIC void nk_reduce_moments_u4_haswell(                        //
    nk_u4x2_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *sum, nk_u64_t *sumsq) {
    if (count == 0) *sum = 0, *sumsq = 0;
    else if (stride_bytes == 1) nk_reduce_moments_u4_haswell_contiguous_(data, count, sum, sumsq);
    else nk_reduce_moments_u4_serial(data, count, stride_bytes, sum, sumsq);
}

NK_INTERNAL void nk_reduce_moments_u1_haswell_contiguous_( //
    nk_u1x8_t const *data, nk_size_t count,                //
    nk_u64_t *sum, nk_u64_t *sumsq) {
    __m256i lut_i8x32 = _mm256_setr_epi8( //
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);
    __m256i mask_0f_i8x32 = _mm256_set1_epi8(0x0F);
    __m256i zero_i8x32 = _mm256_setzero_si256();
    __m256i sum_u64x4 = _mm256_setzero_si256();
    nk_size_t count_bits = count;
    nk_size_t count_bytes = nk_size_divide_round_up_(count, 8);
    unsigned char const *ptr = (unsigned char const *)data;
    while (count_bytes > 0) {
        nk_b256_vec_t raw_vec;
        if (count_bytes < 32) {
            nk_partial_load_b8x32_serial_(ptr, &raw_vec, count_bytes);
            count_bytes = 0;
        }
        else {
            raw_vec.ymm = _mm256_loadu_si256((__m256i const *)ptr);
            ptr += 32, count_bytes -= 32;
        }
        __m256i raw_i8x32 = raw_vec.ymm;
        __m256i low_nibble_u8x32 = _mm256_and_si256(raw_i8x32, mask_0f_i8x32);
        __m256i high_nibble_u8x32 = _mm256_and_si256(_mm256_srli_epi16(raw_i8x32, 4), mask_0f_i8x32);
        __m256i popcnt_u8x32 = _mm256_add_epi8(_mm256_shuffle_epi8(lut_i8x32, low_nibble_u8x32),
                                               _mm256_shuffle_epi8(lut_i8x32, high_nibble_u8x32));
        sum_u64x4 = _mm256_add_epi64(sum_u64x4, _mm256_sad_epu8(popcnt_u8x32, zero_i8x32));
    }
    nk_u64_t s = (nk_u64_t)nk_reduce_add_i64x4_haswell_(sum_u64x4);
    if (count_bits % 8) {
        nk_u8_t last_byte = ((unsigned char const *)data)[nk_size_divide_round_up_(count_bits, 8) - 1];
        nk_u8_t mask = (nk_u8_t)((1u << (count_bits % 8)) - 1u);
        s -= nk_u64_popcount_((nk_u64_t)(last_byte & ~mask));
    }
    *sum = s;
    *sumsq = s;
}

NK_PUBLIC void nk_reduce_moments_u1_haswell(                        //
    nk_u1x8_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *sum, nk_u64_t *sumsq) {
    if (count == 0) *sum = 0, *sumsq = 0;
    else if (stride_bytes == 1) nk_reduce_moments_u1_haswell_contiguous_(data, count, sum, sumsq);
    else nk_reduce_moments_u1_serial(data, count, stride_bytes, sum, sumsq);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_HASWELL
#endif // NK_TARGET_X86_
#endif // NK_REDUCE_HASWELL_NEW_H

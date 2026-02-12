/**
 *  @brief Serial fallbacks for the redesigned reduction API (moments + minmax).
 *  @file include/numkong/reduce/serial_new.h
 *  @author Ash Vardanian
 *  @date February 11, 2026
 *
 *  @sa include/numkong/reduce.h
 *
 *  Provides serial (non-SIMD) implementations of:
 *  - `nk_reduce_moments_*_serial` — sum + sum-of-squares in one pass
 *  - `nk_reduce_minmax_*_serial`  — min + max with indices in one pass
 */
#ifndef NK_REDUCE_SERIAL_NEW_H
#define NK_REDUCE_SERIAL_NEW_H

#include "numkong/types.h"
#include "numkong/cast/serial.h"

#if defined(__cplusplus)
extern "C" {
#endif

NK_PUBLIC void nk_reduce_moments_f32_serial(                       //
    nk_f32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f64_t *sum, nk_f64_t *sumsq) {
    nk_f64_t running_sum = 0, sum_compensation = 0;
    nk_f64_t running_sumsq = 0, sumsq_compensation = 0;
    unsigned char const *ptr = (unsigned char const *)data;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) {
        nk_f64_t val = (nk_f64_t)(*(nk_f32_t const *)ptr);
        nk_f64_t tentative_sum = running_sum + val;
        if (nk_f64_abs_(running_sum) >= nk_f64_abs_(val)) sum_compensation += (running_sum - tentative_sum) + val;
        else sum_compensation += (val - tentative_sum) + running_sum;
        running_sum = tentative_sum;

        nk_f64_t squared_value = val * val;
        nk_f64_t tentative_sumsq = running_sumsq + squared_value;
        if (nk_f64_abs_(running_sumsq) >= nk_f64_abs_(squared_value))
            sumsq_compensation += (running_sumsq - tentative_sumsq) + squared_value;
        else sumsq_compensation += (squared_value - tentative_sumsq) + running_sumsq;
        running_sumsq = tentative_sumsq;
    }
    *sum = running_sum + sum_compensation;
    *sumsq = running_sumsq + sumsq_compensation;
}

NK_PUBLIC void nk_reduce_moments_f64_serial(                       //
    nk_f64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f64_t *sum, nk_f64_t *sumsq) {
    nk_f64_t running_sum = 0, sum_compensation = 0;
    nk_f64_t running_sumsq = 0, sumsq_compensation = 0;
    unsigned char const *ptr = (unsigned char const *)data;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) {
        nk_f64_t val = *(nk_f64_t const *)ptr;
        nk_f64_t tentative_sum = running_sum + val;
        if (nk_f64_abs_(running_sum) >= nk_f64_abs_(val)) sum_compensation += (running_sum - tentative_sum) + val;
        else sum_compensation += (val - tentative_sum) + running_sum;
        running_sum = tentative_sum;

        nk_f64_t squared_value = val * val;
        nk_f64_t tentative_sumsq = running_sumsq + squared_value;
        if (nk_f64_abs_(running_sumsq) >= nk_f64_abs_(squared_value))
            sumsq_compensation += (running_sumsq - tentative_sumsq) + squared_value;
        else sumsq_compensation += (squared_value - tentative_sumsq) + running_sumsq;
        running_sumsq = tentative_sumsq;
    }
    *sum = running_sum + sum_compensation;
    *sumsq = running_sumsq + sumsq_compensation;
}

NK_PUBLIC void nk_reduce_moments_i8_serial(                       //
    nk_i8_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *sum, nk_i64_t *sumsq) {
    nk_i64_t s = 0, sq = 0;
    unsigned char const *ptr = (unsigned char const *)data;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) {
        nk_i64_t val = (nk_i64_t)(*(nk_i8_t const *)ptr);
        s += val;
        sq += val * val;
    }
    *sum = s;
    *sumsq = sq;
}

NK_PUBLIC void nk_reduce_moments_u8_serial(                       //
    nk_u8_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *sum, nk_u64_t *sumsq) {
    nk_u64_t s = 0, sq = 0;
    unsigned char const *ptr = (unsigned char const *)data;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) {
        nk_u64_t val = (nk_u64_t)(*(nk_u8_t const *)ptr);
        s += val;
        sq += val * val;
    }
    *sum = s;
    *sumsq = sq;
}

NK_PUBLIC void nk_reduce_moments_i16_serial(                       //
    nk_i16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *sum, nk_i64_t *sumsq) {
    nk_i64_t s = 0, sq = 0;
    unsigned char const *ptr = (unsigned char const *)data;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) {
        nk_i64_t val = (nk_i64_t)(*(nk_i16_t const *)ptr);
        s += val;
        sq += val * val;
    }
    *sum = s;
    *sumsq = sq;
}

NK_PUBLIC void nk_reduce_moments_u16_serial(                       //
    nk_u16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *sum, nk_u64_t *sumsq) {
    nk_u64_t s = 0, sq = 0;
    unsigned char const *ptr = (unsigned char const *)data;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) {
        nk_u64_t val = (nk_u64_t)(*(nk_u16_t const *)ptr);
        s += val;
        sq += val * val;
    }
    *sum = s;
    *sumsq = sq;
}

NK_PUBLIC void nk_reduce_moments_i32_serial(                       //
    nk_i32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *sum, nk_i64_t *sumsq) {
    nk_i64_t s = 0, sq = 0;
    unsigned char const *ptr = (unsigned char const *)data;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) {
        nk_i64_t val = (nk_i64_t)(*(nk_i32_t const *)ptr);
        s += val;
        sq += val * val;
    }
    *sum = s;
    *sumsq = sq;
}

NK_PUBLIC void nk_reduce_moments_u32_serial(                       //
    nk_u32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *sum, nk_u64_t *sumsq) {
    nk_u64_t s = 0, sq = 0;
    unsigned char const *ptr = (unsigned char const *)data;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) {
        nk_u64_t val = (nk_u64_t)(*(nk_u32_t const *)ptr);
        s += val;
        sq += val * val;
    }
    *sum = s;
    *sumsq = sq;
}

NK_PUBLIC void nk_reduce_moments_i64_serial(                       //
    nk_i64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *sum, nk_i64_t *sumsq) {
    nk_i64_t s = 0, sq = 0;
    unsigned char const *ptr = (unsigned char const *)data;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) {
        nk_i64_t val = *(nk_i64_t const *)ptr;
        s += val;
        sq += val * val;
    }
    *sum = s;
    *sumsq = sq;
}

NK_PUBLIC void nk_reduce_moments_u64_serial(                       //
    nk_u64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *sum, nk_u64_t *sumsq) {
    nk_u64_t s = 0, sq = 0;
    unsigned char const *ptr = (unsigned char const *)data;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) {
        nk_u64_t val = *(nk_u64_t const *)ptr;
        s += val;
        sq += val * val;
    }
    *sum = s;
    *sumsq = sq;
}

NK_PUBLIC void nk_reduce_moments_f16_serial(                       //
    nk_f16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *sum, nk_f32_t *sumsq) {
    nk_f32_t running_sum = 0, sum_compensation = 0;
    nk_f32_t running_sumsq = 0, sumsq_compensation = 0;
    unsigned char const *ptr = (unsigned char const *)data;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) {
        nk_f32_t val;
        nk_f16_to_f32_serial((nk_f16_t const *)ptr, &val);
        nk_f32_t tentative_sum = running_sum + val;
        nk_f32_t abs_running_sum = nk_f32_abs_(running_sum);
        nk_f32_t abs_val = nk_f32_abs_(val);
        if (abs_running_sum >= abs_val) sum_compensation += (running_sum - tentative_sum) + val;
        else sum_compensation += (val - tentative_sum) + running_sum;
        running_sum = tentative_sum;

        nk_f32_t squared_value = val * val;
        nk_f32_t tentative_sumsq = running_sumsq + squared_value;
        nk_f32_t abs_running_sumsq = nk_f32_abs_(running_sumsq);
        nk_f32_t abs_squared_value = nk_f32_abs_(squared_value);
        if (abs_running_sumsq >= abs_squared_value)
            sumsq_compensation += (running_sumsq - tentative_sumsq) + squared_value;
        else sumsq_compensation += (squared_value - tentative_sumsq) + running_sumsq;
        running_sumsq = tentative_sumsq;
    }
    *sum = running_sum + sum_compensation;
    *sumsq = running_sumsq + sumsq_compensation;
}

NK_PUBLIC void nk_reduce_moments_bf16_serial(                       //
    nk_bf16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *sum, nk_f32_t *sumsq) {
    nk_f32_t running_sum = 0, sum_compensation = 0;
    nk_f32_t running_sumsq = 0, sumsq_compensation = 0;
    unsigned char const *ptr = (unsigned char const *)data;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) {
        nk_f32_t val;
        nk_bf16_to_f32_serial((nk_bf16_t const *)ptr, &val);
        nk_f32_t tentative_sum = running_sum + val;
        nk_f32_t abs_running_sum = nk_f32_abs_(running_sum);
        nk_f32_t abs_val = nk_f32_abs_(val);
        if (abs_running_sum >= abs_val) sum_compensation += (running_sum - tentative_sum) + val;
        else sum_compensation += (val - tentative_sum) + running_sum;
        running_sum = tentative_sum;

        nk_f32_t squared_value = val * val;
        nk_f32_t tentative_sumsq = running_sumsq + squared_value;
        nk_f32_t abs_running_sumsq = nk_f32_abs_(running_sumsq);
        nk_f32_t abs_squared_value = nk_f32_abs_(squared_value);
        if (abs_running_sumsq >= abs_squared_value)
            sumsq_compensation += (running_sumsq - tentative_sumsq) + squared_value;
        else sumsq_compensation += (squared_value - tentative_sumsq) + running_sumsq;
        running_sumsq = tentative_sumsq;
    }
    *sum = running_sum + sum_compensation;
    *sumsq = running_sumsq + sumsq_compensation;
}

NK_PUBLIC void nk_reduce_moments_e4m3_serial(                       //
    nk_e4m3_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *sum, nk_f32_t *sumsq) {
    nk_f32_t running_sum = 0, sum_compensation = 0;
    nk_f32_t running_sumsq = 0, sumsq_compensation = 0;
    unsigned char const *ptr = (unsigned char const *)data;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) {
        nk_f32_t val;
        nk_e4m3_to_f32_serial((nk_e4m3_t const *)ptr, &val);
        nk_f32_t tentative_sum = running_sum + val;
        nk_f32_t abs_running_sum = nk_f32_abs_(running_sum);
        nk_f32_t abs_val = nk_f32_abs_(val);
        if (abs_running_sum >= abs_val) sum_compensation += (running_sum - tentative_sum) + val;
        else sum_compensation += (val - tentative_sum) + running_sum;
        running_sum = tentative_sum;

        nk_f32_t squared_value = val * val;
        nk_f32_t tentative_sumsq = running_sumsq + squared_value;
        nk_f32_t abs_running_sumsq = nk_f32_abs_(running_sumsq);
        nk_f32_t abs_squared_value = nk_f32_abs_(squared_value);
        if (abs_running_sumsq >= abs_squared_value)
            sumsq_compensation += (running_sumsq - tentative_sumsq) + squared_value;
        else sumsq_compensation += (squared_value - tentative_sumsq) + running_sumsq;
        running_sumsq = tentative_sumsq;
    }
    *sum = running_sum + sum_compensation;
    *sumsq = running_sumsq + sumsq_compensation;
}

NK_PUBLIC void nk_reduce_moments_e5m2_serial(                       //
    nk_e5m2_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *sum, nk_f32_t *sumsq) {
    nk_f32_t running_sum = 0, sum_compensation = 0;
    nk_f32_t running_sumsq = 0, sumsq_compensation = 0;
    unsigned char const *ptr = (unsigned char const *)data;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) {
        nk_f32_t val;
        nk_e5m2_to_f32_serial((nk_e5m2_t const *)ptr, &val);
        nk_f32_t tentative_sum = running_sum + val;
        nk_f32_t abs_running_sum = nk_f32_abs_(running_sum);
        nk_f32_t abs_val = nk_f32_abs_(val);
        if (abs_running_sum >= abs_val) sum_compensation += (running_sum - tentative_sum) + val;
        else sum_compensation += (val - tentative_sum) + running_sum;
        running_sum = tentative_sum;

        nk_f32_t squared_value = val * val;
        nk_f32_t tentative_sumsq = running_sumsq + squared_value;
        nk_f32_t abs_running_sumsq = nk_f32_abs_(running_sumsq);
        nk_f32_t abs_squared_value = nk_f32_abs_(squared_value);
        if (abs_running_sumsq >= abs_squared_value)
            sumsq_compensation += (running_sumsq - tentative_sumsq) + squared_value;
        else sumsq_compensation += (squared_value - tentative_sumsq) + running_sumsq;
        running_sumsq = tentative_sumsq;
    }
    *sum = running_sum + sum_compensation;
    *sumsq = running_sumsq + sumsq_compensation;
}

NK_PUBLIC void nk_reduce_moments_e2m3_serial(                       //
    nk_e2m3_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *sum, nk_f32_t *sumsq) {
    nk_f32_t running_sum = 0, sum_compensation = 0;
    nk_f32_t running_sumsq = 0, sumsq_compensation = 0;
    unsigned char const *ptr = (unsigned char const *)data;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) {
        nk_f32_t val;
        nk_e2m3_to_f32_serial((nk_e2m3_t const *)ptr, &val);
        nk_f32_t tentative_sum = running_sum + val;
        nk_f32_t abs_running_sum = nk_f32_abs_(running_sum);
        nk_f32_t abs_val = nk_f32_abs_(val);
        if (abs_running_sum >= abs_val) sum_compensation += (running_sum - tentative_sum) + val;
        else sum_compensation += (val - tentative_sum) + running_sum;
        running_sum = tentative_sum;

        nk_f32_t squared_value = val * val;
        nk_f32_t tentative_sumsq = running_sumsq + squared_value;
        nk_f32_t abs_running_sumsq = nk_f32_abs_(running_sumsq);
        nk_f32_t abs_squared_value = nk_f32_abs_(squared_value);
        if (abs_running_sumsq >= abs_squared_value)
            sumsq_compensation += (running_sumsq - tentative_sumsq) + squared_value;
        else sumsq_compensation += (squared_value - tentative_sumsq) + running_sumsq;
        running_sumsq = tentative_sumsq;
    }
    *sum = running_sum + sum_compensation;
    *sumsq = running_sumsq + sumsq_compensation;
}

NK_PUBLIC void nk_reduce_moments_e3m2_serial(                       //
    nk_e3m2_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *sum, nk_f32_t *sumsq) {
    nk_f32_t running_sum = 0, sum_compensation = 0;
    nk_f32_t running_sumsq = 0, sumsq_compensation = 0;
    unsigned char const *ptr = (unsigned char const *)data;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) {
        nk_f32_t val;
        nk_e3m2_to_f32_serial((nk_e3m2_t const *)ptr, &val);
        nk_f32_t tentative_sum = running_sum + val;
        nk_f32_t abs_running_sum = nk_f32_abs_(running_sum);
        nk_f32_t abs_val = nk_f32_abs_(val);
        if (abs_running_sum >= abs_val) sum_compensation += (running_sum - tentative_sum) + val;
        else sum_compensation += (val - tentative_sum) + running_sum;
        running_sum = tentative_sum;

        nk_f32_t squared_value = val * val;
        nk_f32_t tentative_sumsq = running_sumsq + squared_value;
        nk_f32_t abs_running_sumsq = nk_f32_abs_(running_sumsq);
        nk_f32_t abs_squared_value = nk_f32_abs_(squared_value);
        if (abs_running_sumsq >= abs_squared_value)
            sumsq_compensation += (running_sumsq - tentative_sumsq) + squared_value;
        else sumsq_compensation += (squared_value - tentative_sumsq) + running_sumsq;
        running_sumsq = tentative_sumsq;
    }
    *sum = running_sum + sum_compensation;
    *sumsq = running_sumsq + sumsq_compensation;
}

NK_PUBLIC void nk_reduce_moments_i4_serial(                         //
    nk_i4x2_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *sum, nk_i64_t *sumsq) {
    nk_i64_t s = 0, sq = 0;
    unsigned char const *ptr = (unsigned char const *)data;
    nk_size_t i = 0;
    for (; i + 2 <= count; i += 2) {
        unsigned char byte_val = ptr[(i / 2) * stride_bytes];
        nk_i64_t low = (nk_i64_t)nk_i4x2_low_(byte_val);
        nk_i64_t high = (nk_i64_t)nk_i4x2_high_(byte_val);
        s += low + high, sq += low * low + high * high;
    }
    if (i < count) {
        unsigned char byte_val = ptr[(i / 2) * stride_bytes];
        nk_i64_t val = (nk_i64_t)nk_i4x2_low_(byte_val);
        s += val, sq += val * val;
    }
    *sum = s;
    *sumsq = sq;
}

NK_PUBLIC void nk_reduce_moments_u4_serial(                         //
    nk_u4x2_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *sum, nk_u64_t *sumsq) {
    nk_u64_t s = 0, sq = 0;
    unsigned char const *ptr = (unsigned char const *)data;
    nk_size_t i = 0;
    for (; i + 2 <= count; i += 2) {
        unsigned char byte_val = ptr[(i / 2) * stride_bytes];
        nk_u64_t low = nk_u4x2_low_(byte_val);
        nk_u64_t high = nk_u4x2_high_(byte_val);
        s += low + high, sq += low * low + high * high;
    }
    if (i < count) {
        unsigned char byte_val = ptr[(i / 2) * stride_bytes];
        nk_u64_t nibble = nk_u4x2_low_(byte_val);
        s += nibble, sq += nibble * nibble;
    }
    *sum = s;
    *sumsq = sq;
}

NK_PUBLIC void nk_reduce_moments_u1_serial(                         //
    nk_u1x8_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *sum, nk_u64_t *sumsq) {
    nk_u64_t s = 0;
    unsigned char const *ptr = (unsigned char const *)data;
    nk_size_t i = 0;
    for (; i + 8 <= count; i += 8) {
        unsigned char byte_val = ptr[(i / 8) * stride_bytes];
        s += nk_u64_popcount_(byte_val);
    }
    if (i < count) {
        unsigned char byte_val = ptr[(i / 8) * stride_bytes];
        unsigned char mask = (unsigned char)((1u << (count - i)) - 1u);
        s += nk_u64_popcount_(byte_val & mask);
    }
    *sum = s;
    *sumsq = s; // 0^2 = 0, 1^2 = 1, so sumsq == sum
}

NK_PUBLIC void nk_reduce_minmax_f32_serial(                        //
    nk_f32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *min_value, nk_size_t *min_index,                     //
    nk_f32_t *max_value, nk_size_t *max_index) {
    unsigned char const *ptr = (unsigned char const *)data;
    nk_f32_t best_min = NK_F32_MAX, best_max = NK_F32_MIN;
    nk_size_t best_min_idx = 0, best_max_idx = 0;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) {
        nk_f32_t val = *(nk_f32_t const *)ptr;
        if (val < best_min) best_min = val, best_min_idx = i;
        if (val > best_max) best_max = val, best_max_idx = i;
    }
    *min_value = best_min;
    *min_index = best_min_idx;
    *max_value = best_max;
    *max_index = best_max_idx;
}

NK_PUBLIC void nk_reduce_minmax_f64_serial(                        //
    nk_f64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f64_t *min_value, nk_size_t *min_index,                     //
    nk_f64_t *max_value, nk_size_t *max_index) {
    unsigned char const *ptr = (unsigned char const *)data;
    nk_f64_t best_min = NK_F64_MAX, best_max = NK_F64_MIN;
    nk_size_t best_min_idx = 0, best_max_idx = 0;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) {
        nk_f64_t val = *(nk_f64_t const *)ptr;
        if (val < best_min) best_min = val, best_min_idx = i;
        if (val > best_max) best_max = val, best_max_idx = i;
    }
    *min_value = best_min;
    *min_index = best_min_idx;
    *max_value = best_max;
    *max_index = best_max_idx;
}

NK_PUBLIC void nk_reduce_minmax_i8_serial(                        //
    nk_i8_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i8_t *min_value, nk_size_t *min_index,                     //
    nk_i8_t *max_value, nk_size_t *max_index) {
    unsigned char const *ptr = (unsigned char const *)data;
    nk_i8_t best_min = NK_I8_MAX, best_max = NK_I8_MIN;
    nk_size_t best_min_idx = 0, best_max_idx = 0;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) {
        nk_i8_t val = *(nk_i8_t const *)ptr;
        if (val < best_min) best_min = val, best_min_idx = i;
        if (val > best_max) best_max = val, best_max_idx = i;
    }
    *min_value = best_min;
    *min_index = best_min_idx;
    *max_value = best_max;
    *max_index = best_max_idx;
}

NK_PUBLIC void nk_reduce_minmax_u8_serial(                        //
    nk_u8_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u8_t *min_value, nk_size_t *min_index,                     //
    nk_u8_t *max_value, nk_size_t *max_index) {
    unsigned char const *ptr = (unsigned char const *)data;
    nk_u8_t best_min = NK_U8_MAX, best_max = NK_U8_MIN;
    nk_size_t best_min_idx = 0, best_max_idx = 0;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) {
        nk_u8_t val = *(nk_u8_t const *)ptr;
        if (val < best_min) best_min = val, best_min_idx = i;
        if (val > best_max) best_max = val, best_max_idx = i;
    }
    *min_value = best_min;
    *min_index = best_min_idx;
    *max_value = best_max;
    *max_index = best_max_idx;
}

NK_PUBLIC void nk_reduce_minmax_i16_serial(                        //
    nk_i16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i16_t *min_value, nk_size_t *min_index,                     //
    nk_i16_t *max_value, nk_size_t *max_index) {
    unsigned char const *ptr = (unsigned char const *)data;
    nk_i16_t best_min = NK_I16_MAX, best_max = NK_I16_MIN;
    nk_size_t best_min_idx = 0, best_max_idx = 0;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) {
        nk_i16_t val = *(nk_i16_t const *)ptr;
        if (val < best_min) best_min = val, best_min_idx = i;
        if (val > best_max) best_max = val, best_max_idx = i;
    }
    *min_value = best_min;
    *min_index = best_min_idx;
    *max_value = best_max;
    *max_index = best_max_idx;
}

NK_PUBLIC void nk_reduce_minmax_u16_serial(                        //
    nk_u16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u16_t *min_value, nk_size_t *min_index,                     //
    nk_u16_t *max_value, nk_size_t *max_index) {
    unsigned char const *ptr = (unsigned char const *)data;
    nk_u16_t best_min = NK_U16_MAX, best_max = NK_U16_MIN;
    nk_size_t best_min_idx = 0, best_max_idx = 0;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) {
        nk_u16_t val = *(nk_u16_t const *)ptr;
        if (val < best_min) best_min = val, best_min_idx = i;
        if (val > best_max) best_max = val, best_max_idx = i;
    }
    *min_value = best_min;
    *min_index = best_min_idx;
    *max_value = best_max;
    *max_index = best_max_idx;
}

NK_PUBLIC void nk_reduce_minmax_i32_serial(                        //
    nk_i32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i32_t *min_value, nk_size_t *min_index,                     //
    nk_i32_t *max_value, nk_size_t *max_index) {
    unsigned char const *ptr = (unsigned char const *)data;
    nk_i32_t best_min = NK_I32_MAX, best_max = NK_I32_MIN;
    nk_size_t best_min_idx = 0, best_max_idx = 0;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) {
        nk_i32_t val = *(nk_i32_t const *)ptr;
        if (val < best_min) best_min = val, best_min_idx = i;
        if (val > best_max) best_max = val, best_max_idx = i;
    }
    *min_value = best_min;
    *min_index = best_min_idx;
    *max_value = best_max;
    *max_index = best_max_idx;
}

NK_PUBLIC void nk_reduce_minmax_u32_serial(                        //
    nk_u32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u32_t *min_value, nk_size_t *min_index,                     //
    nk_u32_t *max_value, nk_size_t *max_index) {
    unsigned char const *ptr = (unsigned char const *)data;
    nk_u32_t best_min = NK_U32_MAX, best_max = NK_U32_MIN;
    nk_size_t best_min_idx = 0, best_max_idx = 0;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) {
        nk_u32_t val = *(nk_u32_t const *)ptr;
        if (val < best_min) best_min = val, best_min_idx = i;
        if (val > best_max) best_max = val, best_max_idx = i;
    }
    *min_value = best_min;
    *min_index = best_min_idx;
    *max_value = best_max;
    *max_index = best_max_idx;
}

NK_PUBLIC void nk_reduce_minmax_i64_serial(                        //
    nk_i64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *min_value, nk_size_t *min_index,                     //
    nk_i64_t *max_value, nk_size_t *max_index) {
    unsigned char const *ptr = (unsigned char const *)data;
    nk_i64_t best_min = NK_I64_MAX, best_max = NK_I64_MIN;
    nk_size_t best_min_idx = 0, best_max_idx = 0;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) {
        nk_i64_t val = *(nk_i64_t const *)ptr;
        if (val < best_min) best_min = val, best_min_idx = i;
        if (val > best_max) best_max = val, best_max_idx = i;
    }
    *min_value = best_min;
    *min_index = best_min_idx;
    *max_value = best_max;
    *max_index = best_max_idx;
}

NK_PUBLIC void nk_reduce_minmax_u64_serial(                        //
    nk_u64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *min_value, nk_size_t *min_index,                     //
    nk_u64_t *max_value, nk_size_t *max_index) {
    unsigned char const *ptr = (unsigned char const *)data;
    nk_u64_t best_min = NK_U64_MAX, best_max = NK_U64_MIN;
    nk_size_t best_min_idx = 0, best_max_idx = 0;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) {
        nk_u64_t val = *(nk_u64_t const *)ptr;
        if (val < best_min) best_min = val, best_min_idx = i;
        if (val > best_max) best_max = val, best_max_idx = i;
    }
    *min_value = best_min;
    *min_index = best_min_idx;
    *max_value = best_max;
    *max_index = best_max_idx;
}

NK_PUBLIC void nk_reduce_minmax_f16_serial(                        //
    nk_f16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f16_t *min_value, nk_size_t *min_index,                     //
    nk_f16_t *max_value, nk_size_t *max_index) {
    unsigned char const *ptr = (unsigned char const *)data;
    nk_f16_t best_min = nk_f16_from_u16_(NK_F16_MAX), best_max = nk_f16_from_u16_(NK_F16_MIN);
    nk_size_t best_min_index = 0, best_max_index = 0;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) {
        nk_f16_t raw_value = *(nk_f16_t const *)ptr;
        if (nk_f16_compare_(raw_value, best_min) < 0) best_min = raw_value, best_min_index = i;
        if (nk_f16_compare_(raw_value, best_max) > 0) best_max = raw_value, best_max_index = i;
    }
    *min_value = best_min;
    *min_index = best_min_index;
    *max_value = best_max;
    *max_index = best_max_index;
}

NK_PUBLIC void nk_reduce_minmax_bf16_serial(                        //
    nk_bf16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_bf16_t *min_value, nk_size_t *min_index,                     //
    nk_bf16_t *max_value, nk_size_t *max_index) {
    unsigned char const *ptr = (unsigned char const *)data;
    nk_bf16_t best_min = nk_bf16_from_u16_(NK_BF16_MAX), best_max = nk_bf16_from_u16_(NK_BF16_MIN);
    nk_size_t best_min_index = 0, best_max_index = 0;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) {
        nk_bf16_t raw_value = *(nk_bf16_t const *)ptr;
        if (nk_bf16_compare_(raw_value, best_min) < 0) best_min = raw_value, best_min_index = i;
        if (nk_bf16_compare_(raw_value, best_max) > 0) best_max = raw_value, best_max_index = i;
    }
    *min_value = best_min;
    *min_index = best_min_index;
    *max_value = best_max;
    *max_index = best_max_index;
}

NK_PUBLIC void nk_reduce_minmax_e4m3_serial(                        //
    nk_e4m3_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_e4m3_t *min_value, nk_size_t *min_index,                     //
    nk_e4m3_t *max_value, nk_size_t *max_index) {
    unsigned char const *ptr = (unsigned char const *)data;
    nk_e4m3_t best_min = NK_E4M3_MAX, best_max = NK_E4M3_MIN;
    nk_size_t best_min_index = 0, best_max_index = 0;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) {
        nk_e4m3_t raw_value = *(nk_e4m3_t const *)ptr;
        if (nk_e4m3_compare_(raw_value, best_min) < 0) best_min = raw_value, best_min_index = i;
        if (nk_e4m3_compare_(raw_value, best_max) > 0) best_max = raw_value, best_max_index = i;
    }
    *min_value = best_min;
    *min_index = best_min_index;
    *max_value = best_max;
    *max_index = best_max_index;
}

NK_PUBLIC void nk_reduce_minmax_e5m2_serial(                        //
    nk_e5m2_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_e5m2_t *min_value, nk_size_t *min_index,                     //
    nk_e5m2_t *max_value, nk_size_t *max_index) {
    unsigned char const *ptr = (unsigned char const *)data;
    nk_e5m2_t best_min = NK_E5M2_MAX, best_max = NK_E5M2_MIN;
    nk_size_t best_min_index = 0, best_max_index = 0;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) {
        nk_e5m2_t raw_value = *(nk_e5m2_t const *)ptr;
        if (nk_e5m2_compare_(raw_value, best_min) < 0) best_min = raw_value, best_min_index = i;
        if (nk_e5m2_compare_(raw_value, best_max) > 0) best_max = raw_value, best_max_index = i;
    }
    *min_value = best_min;
    *min_index = best_min_index;
    *max_value = best_max;
    *max_index = best_max_index;
}

NK_PUBLIC void nk_reduce_minmax_e2m3_serial(                        //
    nk_e2m3_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_e2m3_t *min_value, nk_size_t *min_index,                     //
    nk_e2m3_t *max_value, nk_size_t *max_index) {
    unsigned char const *ptr = (unsigned char const *)data;
    nk_e2m3_t best_min = NK_E2M3_MAX, best_max = NK_E2M3_MIN;
    nk_size_t best_min_index = 0, best_max_index = 0;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) {
        nk_e2m3_t raw_value = *(nk_e2m3_t const *)ptr;
        if (nk_e2m3_compare_(raw_value, best_min) < 0) best_min = raw_value, best_min_index = i;
        if (nk_e2m3_compare_(raw_value, best_max) > 0) best_max = raw_value, best_max_index = i;
    }
    *min_value = best_min;
    *min_index = best_min_index;
    *max_value = best_max;
    *max_index = best_max_index;
}

NK_PUBLIC void nk_reduce_minmax_e3m2_serial(                        //
    nk_e3m2_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_e3m2_t *min_value, nk_size_t *min_index,                     //
    nk_e3m2_t *max_value, nk_size_t *max_index) {
    unsigned char const *ptr = (unsigned char const *)data;
    nk_e3m2_t best_min = NK_E3M2_MAX, best_max = NK_E3M2_MIN;
    nk_size_t best_min_index = 0, best_max_index = 0;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) {
        nk_e3m2_t raw_value = *(nk_e3m2_t const *)ptr;
        if (nk_e3m2_compare_(raw_value, best_min) < 0) best_min = raw_value, best_min_index = i;
        if (nk_e3m2_compare_(raw_value, best_max) > 0) best_max = raw_value, best_max_index = i;
    }
    *min_value = best_min;
    *min_index = best_min_index;
    *max_value = best_max;
    *max_index = best_max_index;
}

NK_PUBLIC void nk_reduce_minmax_i4_serial(                          //
    nk_i4x2_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i8_t *min_value, nk_size_t *min_index,                       //
    nk_i8_t *max_value, nk_size_t *max_index) {
    unsigned char const *ptr = (unsigned char const *)data;
    nk_i8_t best_min = 7, best_max = -8; // i4 range: -8 to 7
    nk_size_t best_min_idx = 0, best_max_idx = 0;
    for (nk_size_t i = 0; i < count; ++i) {
        nk_size_t byte_idx = i / 2;
        unsigned char byte_val = ptr[byte_idx * stride_bytes];
        nk_i8_t val = nk_i4x2_get_(byte_val, (int)(i & 1));
        if (val < best_min) best_min = val, best_min_idx = i;
        if (val > best_max) best_max = val, best_max_idx = i;
    }
    *min_value = best_min;
    *min_index = best_min_idx;
    *max_value = best_max;
    *max_index = best_max_idx;
}

NK_PUBLIC void nk_reduce_minmax_u4_serial(                          //
    nk_u4x2_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u8_t *min_value, nk_size_t *min_index,                       //
    nk_u8_t *max_value, nk_size_t *max_index) {
    unsigned char const *ptr = (unsigned char const *)data;
    nk_u8_t best_min = 15, best_max = 0; // u4 range: 0 to 15
    nk_size_t best_min_idx = 0, best_max_idx = 0;
    for (nk_size_t i = 0; i < count; ++i) {
        nk_size_t byte_idx = i / 2;
        unsigned char byte_val = ptr[byte_idx * stride_bytes];
        nk_u8_t nibble = nk_u4x2_get_(byte_val, (int)(i & 1));
        if (nibble < best_min) best_min = nibble, best_min_idx = i;
        if (nibble > best_max) best_max = nibble, best_max_idx = i;
    }
    *min_value = best_min;
    *min_index = best_min_idx;
    *max_value = best_max;
    *max_index = best_max_idx;
}

NK_PUBLIC void nk_reduce_minmax_u1_serial(                          //
    nk_u1x8_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u8_t *min_value, nk_size_t *min_index,                       //
    nk_u8_t *max_value, nk_size_t *max_index) {
    unsigned char const *ptr = (unsigned char const *)data;
    nk_u8_t best_min = 1, best_max = 0;
    nk_size_t best_min_idx = 0, best_max_idx = 0;
    for (nk_size_t i = 0; i < count; ++i) {
        nk_size_t byte_idx = i / 8;
        unsigned char byte_val = ptr[byte_idx * stride_bytes];
        nk_u8_t bit = (byte_val >> (i % 8)) & 1;
        if (bit < best_min) {
            best_min = bit;
            best_min_idx = i;
            if (best_min == 0 && best_max == 1) {
                *min_value = best_min;
                *min_index = best_min_idx;
                *max_value = best_max;
                *max_index = best_max_idx;
                return;
            }
        }
        if (bit > best_max) {
            best_max = bit;
            best_max_idx = i;
            if (best_min == 0 && best_max == 1) {
                *min_value = best_min;
                *min_index = best_min_idx;
                *max_value = best_max;
                *max_index = best_max_idx;
                return;
            }
        }
    }
    *min_value = best_min;
    *min_index = best_min_idx;
    *max_value = best_max;
    *max_index = best_max_idx;
}

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_REDUCE_SERIAL_NEW_H

/**
 *  @brief SWAR-accelerated Vector Reductions for SIMD-free CPUs.
 *  @file include/numkong/reduce/serial.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 *
 *  @sa include/numkong/reduce.h
 */
#ifndef NK_REDUCE_SERIAL_H
#define NK_REDUCE_SERIAL_H

#include "numkong/types.h"
#include "numkong/cast/serial.h"

#if defined(__cplusplus)
extern "C" {
#endif

NK_PUBLIC void nk_reduce_add_f32_serial(                           //
    nk_f32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f64_t *result) {
    nk_f64_t sum = 0;
    unsigned char const *ptr = (unsigned char const *)data;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) sum += *(nk_f32_t const *)ptr;
    *result = sum;
}

NK_PUBLIC void nk_reduce_add_f64_serial(                           //
    nk_f64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f64_t *result) {
    nk_f64_t sum = 0, compensation = 0;
    unsigned char const *ptr = (unsigned char const *)data;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) {
        nk_f64_t term = *(nk_f64_t const *)ptr, tentative = sum + term;
        compensation += (nk_f64_abs_(sum) >= nk_f64_abs_(term)) ? ((sum - tentative) + term)
                                                                : ((term - tentative) + sum);
        sum = tentative;
    }
    *result = sum + compensation;
}

NK_PUBLIC void nk_reduce_add_i8_serial(                           //
    nk_i8_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *result) {
    nk_i64_t sum = 0;
    unsigned char const *ptr = (unsigned char const *)data;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) sum += *(nk_i8_t const *)ptr;
    *result = sum;
}

NK_PUBLIC void nk_reduce_add_u8_serial(                           //
    nk_u8_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *result) {
    nk_u64_t sum = 0;
    unsigned char const *ptr = (unsigned char const *)data;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) sum += *(nk_u8_t const *)ptr;
    *result = sum;
}

NK_PUBLIC void nk_reduce_add_i16_serial(                           //
    nk_i16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *result) {
    nk_i64_t sum = 0;
    unsigned char const *ptr = (unsigned char const *)data;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) sum += *(nk_i16_t const *)ptr;
    *result = sum;
}

NK_PUBLIC void nk_reduce_add_u16_serial(                           //
    nk_u16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *result) {
    nk_u64_t sum = 0;
    unsigned char const *ptr = (unsigned char const *)data;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) sum += *(nk_u16_t const *)ptr;
    *result = sum;
}

NK_PUBLIC void nk_reduce_add_i32_serial(                           //
    nk_i32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *result) {
    nk_i64_t sum = 0;
    unsigned char const *ptr = (unsigned char const *)data;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) sum += *(nk_i32_t const *)ptr;
    *result = sum;
}

NK_PUBLIC void nk_reduce_add_u32_serial(                           //
    nk_u32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *result) {
    nk_u64_t sum = 0;
    unsigned char const *ptr = (unsigned char const *)data;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) sum += *(nk_u32_t const *)ptr;
    *result = sum;
}

NK_PUBLIC void nk_reduce_add_i64_serial(                           //
    nk_i64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *result) {
    nk_i64_t sum = 0;
    unsigned char const *ptr = (unsigned char const *)data;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) sum += *(nk_i64_t const *)ptr;
    *result = sum;
}

NK_PUBLIC void nk_reduce_add_u64_serial(                           //
    nk_u64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *result) {
    nk_u64_t sum = 0;
    unsigned char const *ptr = (unsigned char const *)data;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) sum += *(nk_u64_t const *)ptr;
    *result = sum;
}

NK_PUBLIC void nk_reduce_min_i8_serial(                           //
    nk_i8_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i8_t *min_value, nk_size_t *min_index) {
    unsigned char const *ptr = (unsigned char const *)data;
    nk_i8_t best_value = NK_I8_MAX;
    nk_size_t best_index = 0;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) {
        nk_i8_t val = *(nk_i8_t const *)ptr;
        if (val >= best_value) continue;
        best_value = val;
        best_index = i;
    }
    *min_value = best_value;
    *min_index = best_index;
}

NK_PUBLIC void nk_reduce_max_i8_serial(                           //
    nk_i8_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i8_t *max_value, nk_size_t *max_index) {
    unsigned char const *ptr = (unsigned char const *)data;
    nk_i8_t best_value = NK_I8_MIN;
    nk_size_t best_index = 0;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) {
        nk_i8_t val = *(nk_i8_t const *)ptr;
        if (val <= best_value) continue;
        best_value = val;
        best_index = i;
    }
    *max_value = best_value;
    *max_index = best_index;
}

NK_PUBLIC void nk_reduce_min_u8_serial(                           //
    nk_u8_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u8_t *min_value, nk_size_t *min_index) {
    unsigned char const *ptr = (unsigned char const *)data;
    nk_u8_t best_value = NK_U8_MAX;
    nk_size_t best_index = 0;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) {
        nk_u8_t val = *(nk_u8_t const *)ptr;
        if (val >= best_value) continue;
        best_value = val;
        best_index = i;
    }
    *min_value = best_value;
    *min_index = best_index;
}

NK_PUBLIC void nk_reduce_max_u8_serial(                           //
    nk_u8_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u8_t *max_value, nk_size_t *max_index) {
    unsigned char const *ptr = (unsigned char const *)data;
    nk_u8_t best_value = NK_U8_MIN;
    nk_size_t best_index = 0;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) {
        nk_u8_t val = *(nk_u8_t const *)ptr;
        if (val <= best_value) continue;
        best_value = val;
        best_index = i;
    }
    *max_value = best_value;
    *max_index = best_index;
}

NK_PUBLIC void nk_reduce_min_i16_serial(                           //
    nk_i16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i16_t *min_value, nk_size_t *min_index) {
    unsigned char const *ptr = (unsigned char const *)data;
    nk_i16_t best_value = NK_I16_MAX;
    nk_size_t best_index = 0;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) {
        nk_i16_t val = *(nk_i16_t const *)ptr;
        if (val >= best_value) continue;
        best_value = val;
        best_index = i;
    }
    *min_value = best_value;
    *min_index = best_index;
}

NK_PUBLIC void nk_reduce_max_i16_serial(                           //
    nk_i16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i16_t *max_value, nk_size_t *max_index) {
    unsigned char const *ptr = (unsigned char const *)data;
    nk_i16_t best_value = NK_I16_MIN;
    nk_size_t best_index = 0;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) {
        nk_i16_t val = *(nk_i16_t const *)ptr;
        if (val <= best_value) continue;
        best_value = val;
        best_index = i;
    }
    *max_value = best_value;
    *max_index = best_index;
}

NK_PUBLIC void nk_reduce_min_u16_serial(                           //
    nk_u16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u16_t *min_value, nk_size_t *min_index) {
    unsigned char const *ptr = (unsigned char const *)data;
    nk_u16_t best_value = NK_U16_MAX;
    nk_size_t best_index = 0;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) {
        nk_u16_t val = *(nk_u16_t const *)ptr;
        if (val >= best_value) continue;
        best_value = val;
        best_index = i;
    }
    *min_value = best_value;
    *min_index = best_index;
}

NK_PUBLIC void nk_reduce_max_u16_serial(                           //
    nk_u16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u16_t *max_value, nk_size_t *max_index) {
    unsigned char const *ptr = (unsigned char const *)data;
    nk_u16_t best_value = NK_U16_MIN;
    nk_size_t best_index = 0;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) {
        nk_u16_t val = *(nk_u16_t const *)ptr;
        if (val <= best_value) continue;
        best_value = val;
        best_index = i;
    }
    *max_value = best_value;
    *max_index = best_index;
}

NK_PUBLIC void nk_reduce_min_i32_serial(                           //
    nk_i32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i32_t *min_value, nk_size_t *min_index) {
    unsigned char const *ptr = (unsigned char const *)data;
    nk_i32_t best_value = NK_I32_MAX;
    nk_size_t best_index = 0;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) {
        nk_i32_t val = *(nk_i32_t const *)ptr;
        if (val >= best_value) continue;
        best_value = val;
        best_index = i;
    }
    *min_value = best_value;
    *min_index = best_index;
}

NK_PUBLIC void nk_reduce_max_i32_serial(                           //
    nk_i32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i32_t *max_value, nk_size_t *max_index) {
    unsigned char const *ptr = (unsigned char const *)data;
    nk_i32_t best_value = NK_I32_MIN;
    nk_size_t best_index = 0;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) {
        nk_i32_t val = *(nk_i32_t const *)ptr;
        if (val <= best_value) continue;
        best_value = val;
        best_index = i;
    }
    *max_value = best_value;
    *max_index = best_index;
}

NK_PUBLIC void nk_reduce_min_u32_serial(                           //
    nk_u32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u32_t *min_value, nk_size_t *min_index) {
    unsigned char const *ptr = (unsigned char const *)data;
    nk_u32_t best_value = NK_U32_MAX;
    nk_size_t best_index = 0;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) {
        nk_u32_t val = *(nk_u32_t const *)ptr;
        if (val >= best_value) continue;
        best_value = val;
        best_index = i;
    }
    *min_value = best_value;
    *min_index = best_index;
}

NK_PUBLIC void nk_reduce_max_u32_serial(                           //
    nk_u32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u32_t *max_value, nk_size_t *max_index) {
    unsigned char const *ptr = (unsigned char const *)data;
    nk_u32_t best_value = NK_U32_MIN;
    nk_size_t best_index = 0;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) {
        nk_u32_t val = *(nk_u32_t const *)ptr;
        if (val <= best_value) continue;
        best_value = val;
        best_index = i;
    }
    *max_value = best_value;
    *max_index = best_index;
}

NK_PUBLIC void nk_reduce_min_i64_serial(                           //
    nk_i64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *min_value, nk_size_t *min_index) {
    unsigned char const *ptr = (unsigned char const *)data;
    nk_i64_t best_value = NK_I64_MAX;
    nk_size_t best_index = 0;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) {
        nk_i64_t val = *(nk_i64_t const *)ptr;
        if (val >= best_value) continue;
        best_value = val;
        best_index = i;
    }
    *min_value = best_value;
    *min_index = best_index;
}

NK_PUBLIC void nk_reduce_max_i64_serial(                           //
    nk_i64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *max_value, nk_size_t *max_index) {
    unsigned char const *ptr = (unsigned char const *)data;
    nk_i64_t best_value = NK_I64_MIN;
    nk_size_t best_index = 0;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) {
        nk_i64_t val = *(nk_i64_t const *)ptr;
        if (val <= best_value) continue;
        best_value = val;
        best_index = i;
    }
    *max_value = best_value;
    *max_index = best_index;
}

NK_PUBLIC void nk_reduce_min_u64_serial(                           //
    nk_u64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *min_value, nk_size_t *min_index) {
    unsigned char const *ptr = (unsigned char const *)data;
    nk_u64_t best_value = NK_U64_MAX;
    nk_size_t best_index = 0;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) {
        nk_u64_t val = *(nk_u64_t const *)ptr;
        if (val >= best_value) continue;
        best_value = val;
        best_index = i;
    }
    *min_value = best_value;
    *min_index = best_index;
}

NK_PUBLIC void nk_reduce_max_u64_serial(                           //
    nk_u64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *max_value, nk_size_t *max_index) {
    unsigned char const *ptr = (unsigned char const *)data;
    nk_u64_t best_value = NK_U64_MIN;
    nk_size_t best_index = 0;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) {
        nk_u64_t val = *(nk_u64_t const *)ptr;
        if (val <= best_value) continue;
        best_value = val;
        best_index = i;
    }
    *max_value = best_value;
    *max_index = best_index;
}

NK_PUBLIC void nk_reduce_min_f32_serial(                           //
    nk_f32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *min_value, nk_size_t *min_index) {
    unsigned char const *ptr = (unsigned char const *)data;
    nk_f32_t best_value = NK_F32_MAX;
    nk_size_t best_index = 0;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) {
        nk_f32_t val = *(nk_f32_t const *)ptr;
        if (val >= best_value) continue;
        best_value = val;
        best_index = i;
    }
    *min_value = best_value;
    *min_index = best_index;
}

NK_PUBLIC void nk_reduce_max_f32_serial(                           //
    nk_f32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *max_value, nk_size_t *max_index) {
    unsigned char const *ptr = (unsigned char const *)data;
    nk_f32_t best_value = NK_F32_MIN;
    nk_size_t best_index = 0;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) {
        nk_f32_t val = *(nk_f32_t const *)ptr;
        if (val <= best_value) continue;
        best_value = val;
        best_index = i;
    }
    *max_value = best_value;
    *max_index = best_index;
}

NK_PUBLIC void nk_reduce_min_f64_serial(                           //
    nk_f64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f64_t *min_value, nk_size_t *min_index) {
    unsigned char const *ptr = (unsigned char const *)data;
    nk_f64_t best_value = NK_F64_MAX;
    nk_size_t best_index = 0;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) {
        nk_f64_t val = *(nk_f64_t const *)ptr;
        if (val >= best_value) continue;
        best_value = val;
        best_index = i;
    }
    *min_value = best_value;
    *min_index = best_index;
}

NK_PUBLIC void nk_reduce_max_f64_serial(                           //
    nk_f64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f64_t *max_value, nk_size_t *max_index) {
    unsigned char const *ptr = (unsigned char const *)data;
    nk_f64_t best_value = NK_F64_MIN;
    nk_size_t best_index = 0;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) {
        nk_f64_t val = *(nk_f64_t const *)ptr;
        if (val <= best_value) continue;
        best_value = val;
        best_index = i;
    }
    *max_value = best_value;
    *max_index = best_index;
}

NK_PUBLIC void nk_reduce_add_f16_serial(                           //
    nk_f16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *result) {
    nk_f32_t sum = 0;
    unsigned char const *ptr = (unsigned char const *)data;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) {
        nk_f32_t val;
        nk_f16_to_f32_serial((nk_f16_t const *)ptr, &val);
        sum += val;
    }
    *result = sum;
}

NK_PUBLIC void nk_reduce_add_bf16_serial(                           //
    nk_bf16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *result) {
    nk_f32_t sum = 0;
    unsigned char const *ptr = (unsigned char const *)data;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) {
        nk_f32_t val;
        nk_bf16_to_f32_serial((nk_bf16_t const *)ptr, &val);
        sum += val;
    }
    *result = sum;
}

NK_PUBLIC void nk_reduce_add_e4m3_serial(                           //
    nk_e4m3_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *result) {
    nk_f32_t sum = 0;
    unsigned char const *ptr = (unsigned char const *)data;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) {
        nk_f32_t val;
        nk_e4m3_to_f32_serial((nk_e4m3_t const *)ptr, &val);
        sum += val;
    }
    *result = sum;
}

NK_PUBLIC void nk_reduce_add_e5m2_serial(                           //
    nk_e5m2_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *result) {
    nk_f32_t sum = 0;
    unsigned char const *ptr = (unsigned char const *)data;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) {
        nk_f32_t val;
        nk_e5m2_to_f32_serial((nk_e5m2_t const *)ptr, &val);
        sum += val;
    }
    *result = sum;
}

NK_PUBLIC void nk_reduce_min_f16_serial(                           //
    nk_f16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *min_value, nk_size_t *min_index) {
    unsigned char const *ptr = (unsigned char const *)data;
    nk_f32_t best_value = NK_F32_MAX;
    nk_size_t best_index = 0;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) {
        nk_f32_t val;
        nk_f16_to_f32_serial((nk_f16_t const *)ptr, &val);
        if (val >= best_value) continue;
        best_value = val;
        best_index = i;
    }
    *min_value = best_value;
    *min_index = best_index;
}

NK_PUBLIC void nk_reduce_max_f16_serial(                           //
    nk_f16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *max_value, nk_size_t *max_index) {
    unsigned char const *ptr = (unsigned char const *)data;
    nk_f32_t best_value = NK_F32_MIN;
    nk_size_t best_index = 0;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) {
        nk_f32_t val;
        nk_f16_to_f32_serial((nk_f16_t const *)ptr, &val);
        if (val <= best_value) continue;
        best_value = val;
        best_index = i;
    }
    *max_value = best_value;
    *max_index = best_index;
}

NK_PUBLIC void nk_reduce_min_bf16_serial(                           //
    nk_bf16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *min_value, nk_size_t *min_index) {
    unsigned char const *ptr = (unsigned char const *)data;
    nk_f32_t best_value = NK_F32_MAX;
    nk_size_t best_index = 0;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) {
        nk_f32_t val;
        nk_bf16_to_f32_serial((nk_bf16_t const *)ptr, &val);
        if (val >= best_value) continue;
        best_value = val;
        best_index = i;
    }
    *min_value = best_value;
    *min_index = best_index;
}

NK_PUBLIC void nk_reduce_max_bf16_serial(                           //
    nk_bf16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *max_value, nk_size_t *max_index) {
    unsigned char const *ptr = (unsigned char const *)data;
    nk_f32_t best_value = NK_F32_MIN;
    nk_size_t best_index = 0;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) {
        nk_f32_t val;
        nk_bf16_to_f32_serial((nk_bf16_t const *)ptr, &val);
        if (val <= best_value) continue;
        best_value = val;
        best_index = i;
    }
    *max_value = best_value;
    *max_index = best_index;
}

NK_PUBLIC void nk_reduce_min_e4m3_serial(                           //
    nk_e4m3_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *min_value, nk_size_t *min_index) {
    unsigned char const *ptr = (unsigned char const *)data;
    nk_f32_t best_value = NK_F32_MAX;
    nk_size_t best_index = 0;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) {
        nk_f32_t val;
        nk_e4m3_to_f32_serial((nk_e4m3_t const *)ptr, &val);
        if (val >= best_value) continue;
        best_value = val;
        best_index = i;
    }
    *min_value = best_value;
    *min_index = best_index;
}

NK_PUBLIC void nk_reduce_max_e4m3_serial(                           //
    nk_e4m3_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *max_value, nk_size_t *max_index) {
    unsigned char const *ptr = (unsigned char const *)data;
    nk_f32_t best_value = NK_F32_MIN;
    nk_size_t best_index = 0;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) {
        nk_f32_t val;
        nk_e4m3_to_f32_serial((nk_e4m3_t const *)ptr, &val);
        if (val <= best_value) continue;
        best_value = val;
        best_index = i;
    }
    *max_value = best_value;
    *max_index = best_index;
}

NK_PUBLIC void nk_reduce_min_e5m2_serial(                           //
    nk_e5m2_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *min_value, nk_size_t *min_index) {
    unsigned char const *ptr = (unsigned char const *)data;
    nk_f32_t best_value = NK_F32_MAX;
    nk_size_t best_index = 0;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) {
        nk_f32_t val;
        nk_e5m2_to_f32_serial((nk_e5m2_t const *)ptr, &val);
        if (val >= best_value) continue;
        best_value = val;
        best_index = i;
    }
    *min_value = best_value;
    *min_index = best_index;
}

NK_PUBLIC void nk_reduce_max_e5m2_serial(                           //
    nk_e5m2_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *max_value, nk_size_t *max_index) {
    unsigned char const *ptr = (unsigned char const *)data;
    nk_f32_t best_value = NK_F32_MIN;
    nk_size_t best_index = 0;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) {
        nk_f32_t val;
        nk_e5m2_to_f32_serial((nk_e5m2_t const *)ptr, &val);
        if (val <= best_value) continue;
        best_value = val;
        best_index = i;
    }
    *max_value = best_value;
    *max_index = best_index;
}

NK_PUBLIC void nk_reduce_add_e2m3_serial(                           //
    nk_e2m3_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *result) {
    nk_f32_t sum = 0;
    unsigned char const *ptr = (unsigned char const *)data;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) {
        nk_f32_t val;
        nk_e2m3_to_f32_serial((nk_e2m3_t const *)ptr, &val);
        sum += val;
    }
    *result = sum;
}

NK_PUBLIC void nk_reduce_min_e2m3_serial(                           //
    nk_e2m3_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_e2m3_t *min_value, nk_size_t *min_index) {
    unsigned char const *ptr = (unsigned char const *)data;
    nk_f32_t best_value = NK_F32_MAX;
    nk_size_t best_index = 0;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) {
        nk_f32_t val;
        nk_e2m3_to_f32_serial((nk_e2m3_t const *)ptr, &val);
        if (val >= best_value) continue;
        best_value = val;
        best_index = i;
    }
    unsigned char const *best_ptr = (unsigned char const *)data + best_index * stride_bytes;
    *min_value = *(nk_e2m3_t const *)best_ptr;
    *min_index = best_index;
}

NK_PUBLIC void nk_reduce_max_e2m3_serial(                           //
    nk_e2m3_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_e2m3_t *max_value, nk_size_t *max_index) {
    unsigned char const *ptr = (unsigned char const *)data;
    nk_f32_t best_value = NK_F32_MIN;
    nk_size_t best_index = 0;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) {
        nk_f32_t val;
        nk_e2m3_to_f32_serial((nk_e2m3_t const *)ptr, &val);
        if (val <= best_value) continue;
        best_value = val;
        best_index = i;
    }
    unsigned char const *best_ptr = (unsigned char const *)data + best_index * stride_bytes;
    *max_value = *(nk_e2m3_t const *)best_ptr;
    *max_index = best_index;
}

NK_PUBLIC void nk_reduce_add_e3m2_serial(                           //
    nk_e3m2_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *result) {
    nk_f32_t sum = 0;
    unsigned char const *ptr = (unsigned char const *)data;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) {
        nk_f32_t val;
        nk_e3m2_to_f32_serial((nk_e3m2_t const *)ptr, &val);
        sum += val;
    }
    *result = sum;
}

NK_PUBLIC void nk_reduce_min_e3m2_serial(                           //
    nk_e3m2_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_e3m2_t *min_value, nk_size_t *min_index) {
    unsigned char const *ptr = (unsigned char const *)data;
    nk_f32_t best_value = NK_F32_MAX;
    nk_size_t best_index = 0;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) {
        nk_f32_t val;
        nk_e3m2_to_f32_serial((nk_e3m2_t const *)ptr, &val);
        if (val >= best_value) continue;
        best_value = val;
        best_index = i;
    }
    unsigned char const *best_ptr = (unsigned char const *)data + best_index * stride_bytes;
    *min_value = *(nk_e3m2_t const *)best_ptr;
    *min_index = best_index;
}

NK_PUBLIC void nk_reduce_max_e3m2_serial(                           //
    nk_e3m2_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_e3m2_t *max_value, nk_size_t *max_index) {
    unsigned char const *ptr = (unsigned char const *)data;
    nk_f32_t best_value = NK_F32_MIN;
    nk_size_t best_index = 0;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) {
        nk_f32_t val;
        nk_e3m2_to_f32_serial((nk_e3m2_t const *)ptr, &val);
        if (val <= best_value) continue;
        best_value = val;
        best_index = i;
    }
    unsigned char const *best_ptr = (unsigned char const *)data + best_index * stride_bytes;
    *max_value = *(nk_e3m2_t const *)best_ptr;
    *max_index = best_index;
}

NK_PUBLIC void nk_reduce_add_i4_serial(                             //
    nk_i4x2_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *result) {
    nk_i64_t sum = 0;
    unsigned char const *ptr = (unsigned char const *)data;
    for (nk_size_t i = 0; i < count; ++i) {
        nk_size_t byte_idx = i / 2;
        nk_u8_t byte_val = ptr[byte_idx * stride_bytes];
        nk_u8_t nibble = (i % 2 == 0) ? (byte_val & 0x0F) : ((byte_val >> 4) & 0x0F);
        nk_i8_t val = (nk_i8_t)((nibble ^ 8) - 8); // sign-extend
        sum += val;
    }
    *result = sum;
}

NK_PUBLIC void nk_reduce_min_i4_serial(                             //
    nk_i4x2_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i8_t *min_value, nk_size_t *min_index) {
    nk_i8_t best_value = 7;
    nk_size_t best_index = 0;
    unsigned char const *ptr = (unsigned char const *)data;
    for (nk_size_t i = 0; i < count; ++i) {
        nk_size_t byte_idx = i / 2;
        nk_u8_t byte_val = ptr[byte_idx * stride_bytes];
        nk_u8_t nibble = (i % 2 == 0) ? (byte_val & 0x0F) : ((byte_val >> 4) & 0x0F);
        nk_i8_t val = (nk_i8_t)((nibble ^ 8) - 8);
        if (val >= best_value) continue;
        best_value = val;
        best_index = i;
    }
    *min_value = best_value;
    *min_index = best_index;
}

NK_PUBLIC void nk_reduce_max_i4_serial(                             //
    nk_i4x2_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i8_t *max_value, nk_size_t *max_index) {
    nk_i8_t best_value = -8;
    nk_size_t best_index = 0;
    unsigned char const *ptr = (unsigned char const *)data;
    for (nk_size_t i = 0; i < count; ++i) {
        nk_size_t byte_idx = i / 2;
        nk_u8_t byte_val = ptr[byte_idx * stride_bytes];
        nk_u8_t nibble = (i % 2 == 0) ? (byte_val & 0x0F) : ((byte_val >> 4) & 0x0F);
        nk_i8_t val = (nk_i8_t)((nibble ^ 8) - 8);
        if (val <= best_value) continue;
        best_value = val;
        best_index = i;
    }
    *max_value = best_value;
    *max_index = best_index;
}

NK_PUBLIC void nk_reduce_add_u4_serial(                             //
    nk_u4x2_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *result) {
    nk_u64_t sum = 0;
    unsigned char const *ptr = (unsigned char const *)data;
    for (nk_size_t i = 0; i < count; ++i) {
        nk_size_t byte_idx = i / 2;
        nk_u8_t byte_val = ptr[byte_idx * stride_bytes];
        nk_u8_t nibble = (i % 2 == 0) ? (byte_val & 0x0F) : ((byte_val >> 4) & 0x0F);
        sum += nibble;
    }
    *result = sum;
}

NK_PUBLIC void nk_reduce_min_u4_serial(                             //
    nk_u4x2_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u8_t *min_value, nk_size_t *min_index) {
    nk_u8_t best_value = 15;
    nk_size_t best_index = 0;
    unsigned char const *ptr = (unsigned char const *)data;
    for (nk_size_t i = 0; i < count; ++i) {
        nk_size_t byte_idx = i / 2;
        nk_u8_t byte_val = ptr[byte_idx * stride_bytes];
        nk_u8_t nibble = (i % 2 == 0) ? (byte_val & 0x0F) : ((byte_val >> 4) & 0x0F);
        if (nibble >= best_value) continue;
        best_value = nibble;
        best_index = i;
    }
    *min_value = best_value;
    *min_index = best_index;
}

NK_PUBLIC void nk_reduce_max_u4_serial(                             //
    nk_u4x2_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u8_t *max_value, nk_size_t *max_index) {
    nk_u8_t best_value = 0;
    nk_size_t best_index = 0;
    unsigned char const *ptr = (unsigned char const *)data;
    for (nk_size_t i = 0; i < count; ++i) {
        nk_size_t byte_idx = i / 2;
        nk_u8_t byte_val = ptr[byte_idx * stride_bytes];
        nk_u8_t nibble = (i % 2 == 0) ? (byte_val & 0x0F) : ((byte_val >> 4) & 0x0F);
        if (nibble <= best_value) continue;
        best_value = nibble;
        best_index = i;
    }
    *max_value = best_value;
    *max_index = best_index;
}

NK_PUBLIC void nk_reduce_add_u1_serial(                             //
    nk_u1x8_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *result) {
    nk_u64_t sum = 0;
    unsigned char const *ptr = (unsigned char const *)data;
    for (nk_size_t i = 0; i < count; ++i) {
        nk_size_t byte_idx = i / 8;
        nk_u8_t byte_val = ptr[byte_idx * stride_bytes];
        sum += (byte_val >> (i % 8)) & 1;
    }
    *result = sum;
}

NK_PUBLIC void nk_reduce_min_u1_serial(                             //
    nk_u1x8_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u8_t *min_value, nk_size_t *min_index) {
    unsigned char const *ptr = (unsigned char const *)data;
    for (nk_size_t i = 0; i < count; ++i) {
        nk_size_t byte_idx = i / 8;
        nk_u8_t byte_val = ptr[byte_idx * stride_bytes];
        if (((byte_val >> (i % 8)) & 1) == 0) {
            *min_value = 0;
            *min_index = i;
            return;
        }
    }
    *min_value = count ? 1 : 0;
    *min_index = 0;
}

NK_PUBLIC void nk_reduce_max_u1_serial(                             //
    nk_u1x8_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u8_t *max_value, nk_size_t *max_index) {
    unsigned char const *ptr = (unsigned char const *)data;
    for (nk_size_t i = 0; i < count; ++i) {
        nk_size_t byte_idx = i / 8;
        nk_u8_t byte_val = ptr[byte_idx * stride_bytes];
        if (((byte_val >> (i % 8)) & 1) == 1) {
            *max_value = 1;
            *max_index = i;
            return;
        }
    }
    *max_value = 0;
    *max_index = 0;
}

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_REDUCE_SERIAL_H

/**
 *  @brief SIMD-accelerated horizontal reduction operations for SIMD-free CPUs.
 *  @file include/numkong/reduce/serial.h
 *  @sa include/numkong/reduce.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 */
#ifndef NK_REDUCE_SERIAL_H
#define NK_REDUCE_SERIAL_H
#include "numkong/types.h"

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
        compensation += (nk_abs_f64(sum) >= nk_abs_f64(term)) ? ((sum - tentative) + term) : ((term - tentative) + sum);
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
    if (count == 0) {
        *min_value = 0;
        *min_index = 0;
        return;
    }
    unsigned char const *ptr = (unsigned char const *)data;
    nk_i8_t best_value = *(nk_i8_t const *)ptr;
    nk_size_t best_index = 0;
    ptr += stride_bytes;
    for (nk_size_t i = 1; i < count; ++i, ptr += stride_bytes) {
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
    if (count == 0) {
        *max_value = 0;
        *max_index = 0;
        return;
    }
    unsigned char const *ptr = (unsigned char const *)data;
    nk_i8_t best_value = *(nk_i8_t const *)ptr;
    nk_size_t best_index = 0;
    ptr += stride_bytes;
    for (nk_size_t i = 1; i < count; ++i, ptr += stride_bytes) {
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
    if (count == 0) {
        *min_value = 0;
        *min_index = 0;
        return;
    }
    unsigned char const *ptr = (unsigned char const *)data;
    nk_u8_t best_value = *(nk_u8_t const *)ptr;
    nk_size_t best_index = 0;
    ptr += stride_bytes;
    for (nk_size_t i = 1; i < count; ++i, ptr += stride_bytes) {
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
    if (count == 0) {
        *max_value = 0;
        *max_index = 0;
        return;
    }
    unsigned char const *ptr = (unsigned char const *)data;
    nk_u8_t best_value = *(nk_u8_t const *)ptr;
    nk_size_t best_index = 0;
    ptr += stride_bytes;
    for (nk_size_t i = 1; i < count; ++i, ptr += stride_bytes) {
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
    if (count == 0) {
        *min_value = 0;
        *min_index = 0;
        return;
    }
    unsigned char const *ptr = (unsigned char const *)data;
    nk_i16_t best_value = *(nk_i16_t const *)ptr;
    nk_size_t best_index = 0;
    ptr += stride_bytes;
    for (nk_size_t i = 1; i < count; ++i, ptr += stride_bytes) {
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
    if (count == 0) {
        *max_value = 0;
        *max_index = 0;
        return;
    }
    unsigned char const *ptr = (unsigned char const *)data;
    nk_i16_t best_value = *(nk_i16_t const *)ptr;
    nk_size_t best_index = 0;
    ptr += stride_bytes;
    for (nk_size_t i = 1; i < count; ++i, ptr += stride_bytes) {
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
    if (count == 0) {
        *min_value = 0;
        *min_index = 0;
        return;
    }
    unsigned char const *ptr = (unsigned char const *)data;
    nk_u16_t best_value = *(nk_u16_t const *)ptr;
    nk_size_t best_index = 0;
    ptr += stride_bytes;
    for (nk_size_t i = 1; i < count; ++i, ptr += stride_bytes) {
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
    if (count == 0) {
        *max_value = 0;
        *max_index = 0;
        return;
    }
    unsigned char const *ptr = (unsigned char const *)data;
    nk_u16_t best_value = *(nk_u16_t const *)ptr;
    nk_size_t best_index = 0;
    ptr += stride_bytes;
    for (nk_size_t i = 1; i < count; ++i, ptr += stride_bytes) {
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
    if (count == 0) {
        *min_value = 0;
        *min_index = 0;
        return;
    }
    unsigned char const *ptr = (unsigned char const *)data;
    nk_i32_t best_value = *(nk_i32_t const *)ptr;
    nk_size_t best_index = 0;
    ptr += stride_bytes;
    for (nk_size_t i = 1; i < count; ++i, ptr += stride_bytes) {
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
    if (count == 0) {
        *max_value = 0;
        *max_index = 0;
        return;
    }
    unsigned char const *ptr = (unsigned char const *)data;
    nk_i32_t best_value = *(nk_i32_t const *)ptr;
    nk_size_t best_index = 0;
    ptr += stride_bytes;
    for (nk_size_t i = 1; i < count; ++i, ptr += stride_bytes) {
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
    if (count == 0) {
        *min_value = 0;
        *min_index = 0;
        return;
    }
    unsigned char const *ptr = (unsigned char const *)data;
    nk_u32_t best_value = *(nk_u32_t const *)ptr;
    nk_size_t best_index = 0;
    ptr += stride_bytes;
    for (nk_size_t i = 1; i < count; ++i, ptr += stride_bytes) {
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
    if (count == 0) {
        *max_value = 0;
        *max_index = 0;
        return;
    }
    unsigned char const *ptr = (unsigned char const *)data;
    nk_u32_t best_value = *(nk_u32_t const *)ptr;
    nk_size_t best_index = 0;
    ptr += stride_bytes;
    for (nk_size_t i = 1; i < count; ++i, ptr += stride_bytes) {
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
    if (count == 0) {
        *min_value = 0;
        *min_index = 0;
        return;
    }
    unsigned char const *ptr = (unsigned char const *)data;
    nk_i64_t best_value = *(nk_i64_t const *)ptr;
    nk_size_t best_index = 0;
    ptr += stride_bytes;
    for (nk_size_t i = 1; i < count; ++i, ptr += stride_bytes) {
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
    if (count == 0) {
        *max_value = 0;
        *max_index = 0;
        return;
    }
    unsigned char const *ptr = (unsigned char const *)data;
    nk_i64_t best_value = *(nk_i64_t const *)ptr;
    nk_size_t best_index = 0;
    ptr += stride_bytes;
    for (nk_size_t i = 1; i < count; ++i, ptr += stride_bytes) {
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
    if (count == 0) {
        *min_value = 0;
        *min_index = 0;
        return;
    }
    unsigned char const *ptr = (unsigned char const *)data;
    nk_u64_t best_value = *(nk_u64_t const *)ptr;
    nk_size_t best_index = 0;
    ptr += stride_bytes;
    for (nk_size_t i = 1; i < count; ++i, ptr += stride_bytes) {
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
    if (count == 0) {
        *max_value = 0;
        *max_index = 0;
        return;
    }
    unsigned char const *ptr = (unsigned char const *)data;
    nk_u64_t best_value = *(nk_u64_t const *)ptr;
    nk_size_t best_index = 0;
    ptr += stride_bytes;
    for (nk_size_t i = 1; i < count; ++i, ptr += stride_bytes) {
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
    if (count == 0) {
        *min_value = 0;
        *min_index = 0;
        return;
    }
    unsigned char const *ptr = (unsigned char const *)data;
    nk_f32_t best_value = *(nk_f32_t const *)ptr;
    nk_size_t best_index = 0;
    ptr += stride_bytes;
    for (nk_size_t i = 1; i < count; ++i, ptr += stride_bytes) {
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
    if (count == 0) {
        *max_value = 0;
        *max_index = 0;
        return;
    }
    unsigned char const *ptr = (unsigned char const *)data;
    nk_f32_t best_value = *(nk_f32_t const *)ptr;
    nk_size_t best_index = 0;
    ptr += stride_bytes;
    for (nk_size_t i = 1; i < count; ++i, ptr += stride_bytes) {
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
    if (count == 0) {
        *min_value = 0;
        *min_index = 0;
        return;
    }
    unsigned char const *ptr = (unsigned char const *)data;
    nk_f64_t best_value = *(nk_f64_t const *)ptr;
    nk_size_t best_index = 0;
    ptr += stride_bytes;
    for (nk_size_t i = 1; i < count; ++i, ptr += stride_bytes) {
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
    if (count == 0) {
        *max_value = 0;
        *max_index = 0;
        return;
    }
    unsigned char const *ptr = (unsigned char const *)data;
    nk_f64_t best_value = *(nk_f64_t const *)ptr;
    nk_size_t best_index = 0;
    ptr += stride_bytes;
    for (nk_size_t i = 1; i < count; ++i, ptr += stride_bytes) {
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
        nk_f16_to_f32((nk_f16_t const *)ptr, &val);
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
        nk_bf16_to_f32((nk_bf16_t const *)ptr, &val);
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
        nk_e4m3_to_f32((nk_e4m3_t const *)ptr, &val);
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
        nk_e5m2_to_f32((nk_e5m2_t const *)ptr, &val);
        sum += val;
    }
    *result = sum;
}

NK_PUBLIC void nk_reduce_min_f16_serial(                           //
    nk_f16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *min_value, nk_size_t *min_index) {
    if (count == 0) {
        *min_value = 0;
        *min_index = 0;
        return;
    }
    unsigned char const *ptr = (unsigned char const *)data;
    nk_f32_t best_value;
    nk_f16_to_f32((nk_f16_t const *)ptr, &best_value);
    nk_size_t best_index = 0;
    ptr += stride_bytes;
    for (nk_size_t i = 1; i < count; ++i, ptr += stride_bytes) {
        nk_f32_t val;
        nk_f16_to_f32((nk_f16_t const *)ptr, &val);
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
    if (count == 0) {
        *max_value = 0;
        *max_index = 0;
        return;
    }
    unsigned char const *ptr = (unsigned char const *)data;
    nk_f32_t best_value;
    nk_f16_to_f32((nk_f16_t const *)ptr, &best_value);
    nk_size_t best_index = 0;
    ptr += stride_bytes;
    for (nk_size_t i = 1; i < count; ++i, ptr += stride_bytes) {
        nk_f32_t val;
        nk_f16_to_f32((nk_f16_t const *)ptr, &val);
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
    if (count == 0) {
        *min_value = 0;
        *min_index = 0;
        return;
    }
    unsigned char const *ptr = (unsigned char const *)data;
    nk_f32_t best_value;
    nk_bf16_to_f32((nk_bf16_t const *)ptr, &best_value);
    nk_size_t best_index = 0;
    ptr += stride_bytes;
    for (nk_size_t i = 1; i < count; ++i, ptr += stride_bytes) {
        nk_f32_t val;
        nk_bf16_to_f32((nk_bf16_t const *)ptr, &val);
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
    if (count == 0) {
        *max_value = 0;
        *max_index = 0;
        return;
    }
    unsigned char const *ptr = (unsigned char const *)data;
    nk_f32_t best_value;
    nk_bf16_to_f32((nk_bf16_t const *)ptr, &best_value);
    nk_size_t best_index = 0;
    ptr += stride_bytes;
    for (nk_size_t i = 1; i < count; ++i, ptr += stride_bytes) {
        nk_f32_t val;
        nk_bf16_to_f32((nk_bf16_t const *)ptr, &val);
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
    if (count == 0) {
        *min_value = 0;
        *min_index = 0;
        return;
    }
    unsigned char const *ptr = (unsigned char const *)data;
    nk_f32_t best_value;
    nk_e4m3_to_f32((nk_e4m3_t const *)ptr, &best_value);
    nk_size_t best_index = 0;
    ptr += stride_bytes;
    for (nk_size_t i = 1; i < count; ++i, ptr += stride_bytes) {
        nk_f32_t val;
        nk_e4m3_to_f32((nk_e4m3_t const *)ptr, &val);
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
    if (count == 0) {
        *max_value = 0;
        *max_index = 0;
        return;
    }
    unsigned char const *ptr = (unsigned char const *)data;
    nk_f32_t best_value;
    nk_e4m3_to_f32((nk_e4m3_t const *)ptr, &best_value);
    nk_size_t best_index = 0;
    ptr += stride_bytes;
    for (nk_size_t i = 1; i < count; ++i, ptr += stride_bytes) {
        nk_f32_t val;
        nk_e4m3_to_f32((nk_e4m3_t const *)ptr, &val);
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
    if (count == 0) {
        *min_value = 0;
        *min_index = 0;
        return;
    }
    unsigned char const *ptr = (unsigned char const *)data;
    nk_f32_t best_value;
    nk_e5m2_to_f32((nk_e5m2_t const *)ptr, &best_value);
    nk_size_t best_index = 0;
    ptr += stride_bytes;
    for (nk_size_t i = 1; i < count; ++i, ptr += stride_bytes) {
        nk_f32_t val;
        nk_e5m2_to_f32((nk_e5m2_t const *)ptr, &val);
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
    if (count == 0) {
        *max_value = 0;
        *max_index = 0;
        return;
    }
    unsigned char const *ptr = (unsigned char const *)data;
    nk_f32_t best_value;
    nk_e5m2_to_f32((nk_e5m2_t const *)ptr, &best_value);
    nk_size_t best_index = 0;
    ptr += stride_bytes;
    for (nk_size_t i = 1; i < count; ++i, ptr += stride_bytes) {
        nk_f32_t val;
        nk_e5m2_to_f32((nk_e5m2_t const *)ptr, &val);
        if (val <= best_value) continue;
        best_value = val;
        best_index = i;
    }
    *max_value = best_value;
    *max_index = best_index;
}

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_REDUCE_SERIAL_H
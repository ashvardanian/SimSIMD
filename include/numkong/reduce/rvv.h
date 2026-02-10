/**
 *  @brief SIMD-accelerated Vector Reductions for RISC-V.
 *  @file include/numkong/reduce/rvv.h
 *  @author Ash Vardanian
 *  @date February 6, 2026
 *
 *  @sa include/numkong/reduce.h
 *
 *  RVV's key advantage for reductions:
 *  - Native strided loads (`vlse*`) handle ANY stride in hardware — no gather/blend paths needed
 *  - Single-instruction reductions (`vfredusum.vs`, `vfredmin.vs`, etc.) replace multi-step cascades
 *  - Two code paths suffice: contiguous (`vle*`) and strided (`vlse*`)
 *  - Scaled LMUL lets narrow data coexist with wide indices in one vector group —
 *    e.g., e8m1 values pair with u64m8 indices, e16m1 with u64m4, e32m1 with u64m2
 *
 *  Widening reductions:
 *  - i8/u8: widen to i16/u16 via `vwadd_vx`/`vwaddu_vx`, then `vwredsum.vs`/`vwredsumu.vs` → i32/u32
 *  - i16/u16: `vwredsum.vs`/`vwredsumu.vs` → i32/u32, accumulate in scalar i64/u64
 *  - i32/u32: `vwredsum.vs`/`vwredsumu.vs` → i64/u64
 *  - f32: widen to f64 via `vfwcvt`, `vfredusum.vs` → f64
 *  - f16/bf16/e4m3/e5m2: convert to f32 via cast/rvv.h helpers, reduce in f32
 *
 *  Min/max use a single-pass approach with per-lane value+index tracking:
 *  - Hot loop: compare chunk vs best, vmerge to update both value and index vectors
 *  - After loop: single horizontal vredmin/vredmax, then vmseq+vredminu for smallest index
 */
#ifndef NK_REDUCE_RVV_H
#define NK_REDUCE_RVV_H

#if NK_TARGET_RISCV_
#if NK_TARGET_RVV

#include "numkong/types.h"
#include "numkong/cast/rvv.h" // `nk_f16m1_to_f32m2_rvv_`, `nk_bf16m1_to_f32m2_rvv_`, etc.

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=+v"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=+v")
#endif

#if defined(__cplusplus)
extern "C" {
#endif

#pragma region - Add Reductions

NK_INTERNAL void nk_reduce_add_f32_rvv_contiguous_( //
    nk_f32_t const *data, nk_size_t count,          //
    nk_f64_t *result) {
    nk_size_t vlmax = __riscv_vsetvlmax_e64m2();
    vfloat64m2_t sum_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, vlmax);
    for (nk_size_t vector_length; count > 0; count -= vector_length, data += vector_length) {
        vector_length = __riscv_vsetvl_e32m1(count);
        vfloat32m1_t data_f32m1 = __riscv_vle32_v_f32m1(data, vector_length);
        sum_f64m2 = __riscv_vfwadd_wv_f64m2(sum_f64m2, data_f32m1, vector_length);
    }
    vfloat64m1_t zero_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, 1);
    *result = (nk_f64_t)__riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m2_f64m1(sum_f64m2, zero_f64m1, vlmax));
}

NK_INTERNAL void nk_reduce_add_f32_rvv_strided_(                   //
    nk_f32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f64_t *result) {
    nk_size_t vlmax = __riscv_vsetvlmax_e64m2();
    vfloat64m2_t sum_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, vlmax);
    unsigned char const *ptr = (unsigned char const *)data;
    for (nk_size_t vector_length; count > 0; count -= vector_length, ptr += vector_length * stride_bytes) {
        vector_length = __riscv_vsetvl_e32m1(count);
        vfloat32m1_t data_f32m1 = __riscv_vlse32_v_f32m1((nk_f32_t const *)ptr, (nk_ssize_t)stride_bytes,
                                                         vector_length);
        sum_f64m2 = __riscv_vfwadd_wv_f64m2(sum_f64m2, data_f32m1, vector_length);
    }
    vfloat64m1_t zero_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, 1);
    *result = (nk_f64_t)__riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m2_f64m1(sum_f64m2, zero_f64m1, vlmax));
}

NK_PUBLIC void nk_reduce_add_f32_rvv(                              //
    nk_f32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f64_t *result) {
    if (count == 0) *result = 0;
    else if (stride_bytes == sizeof(nk_f32_t)) nk_reduce_add_f32_rvv_contiguous_(data, count, result);
    else nk_reduce_add_f32_rvv_strided_(data, count, stride_bytes, result);
}

NK_INTERNAL void nk_reduce_add_f64_rvv_contiguous_( //
    nk_f64_t const *data, nk_size_t count,          //
    nk_f64_t *result) {
    nk_size_t vlmax = __riscv_vsetvlmax_e64m4();
    vfloat64m4_t sum_f64m4 = __riscv_vfmv_v_f_f64m4(0.0, vlmax);
    for (nk_size_t vector_length; count > 0; count -= vector_length, data += vector_length) {
        vector_length = __riscv_vsetvl_e64m4(count);
        vfloat64m4_t data_f64m4 = __riscv_vle64_v_f64m4(data, vector_length);
        sum_f64m4 = __riscv_vfadd_vv_f64m4(sum_f64m4, data_f64m4, vector_length);
    }
    vfloat64m1_t zero_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, 1);
    *result = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m4_f64m1(sum_f64m4, zero_f64m1, vlmax));
}

NK_INTERNAL void nk_reduce_add_f64_rvv_strided_(                   //
    nk_f64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f64_t *result) {
    nk_size_t vlmax = __riscv_vsetvlmax_e64m4();
    vfloat64m4_t sum_f64m4 = __riscv_vfmv_v_f_f64m4(0.0, vlmax);
    unsigned char const *ptr = (unsigned char const *)data;
    for (nk_size_t vector_length; count > 0; count -= vector_length, ptr += vector_length * stride_bytes) {
        vector_length = __riscv_vsetvl_e64m4(count);
        vfloat64m4_t data_f64m4 = __riscv_vlse64_v_f64m4((nk_f64_t const *)ptr, (nk_ssize_t)stride_bytes,
                                                         vector_length);
        sum_f64m4 = __riscv_vfadd_vv_f64m4(sum_f64m4, data_f64m4, vector_length);
    }
    vfloat64m1_t zero_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, 1);
    *result = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m4_f64m1(sum_f64m4, zero_f64m1, vlmax));
}

NK_PUBLIC void nk_reduce_add_f64_rvv(                              //
    nk_f64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f64_t *result) {
    if (count == 0) *result = 0;
    else if (stride_bytes == sizeof(nk_f64_t)) nk_reduce_add_f64_rvv_contiguous_(data, count, result);
    else nk_reduce_add_f64_rvv_strided_(data, count, stride_bytes, result);
}

NK_INTERNAL void nk_reduce_add_i8_rvv_contiguous_( //
    nk_i8_t const *data, nk_size_t count,          //
    nk_i64_t *result) {
    nk_i64_t sum = 0;
    vint32m1_t chunk_i32m1 = __riscv_vmv_v_x_i32m1(0, 1);
    nk_size_t drain_counter = 0;
    for (nk_size_t vector_length; count > 0; count -= vector_length, data += vector_length) {
        vector_length = __riscv_vsetvl_e8m1(count);
        vint8m1_t data_i8m1 = __riscv_vle8_v_i8m1(data, vector_length);
        // Widen i8 → i16
        vint16m2_t data_i16m2 = __riscv_vwadd_vx_i16m2(data_i8m1, 0, vector_length);
        // Widening reduction i16 → i32
        chunk_i32m1 = __riscv_vwredsum_vs_i16m2_i32m1(data_i16m2, chunk_i32m1, vector_length);
        if (++drain_counter >= 16384) {
            sum += (nk_i64_t)__riscv_vmv_x_s_i32m1_i32(chunk_i32m1);
            chunk_i32m1 = __riscv_vmv_v_x_i32m1(0, 1);
            drain_counter = 0;
        }
    }
    *result = sum + (nk_i64_t)__riscv_vmv_x_s_i32m1_i32(chunk_i32m1);
}

NK_INTERNAL void nk_reduce_add_i8_rvv_strided_(                   //
    nk_i8_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *result) {
    nk_i64_t sum = 0;
    vint32m1_t chunk_i32m1 = __riscv_vmv_v_x_i32m1(0, 1);
    nk_size_t drain_counter = 0;
    unsigned char const *ptr = (unsigned char const *)data;
    for (nk_size_t vector_length; count > 0; count -= vector_length, ptr += vector_length * stride_bytes) {
        vector_length = __riscv_vsetvl_e8m1(count);
        vint8m1_t data_i8m1 = __riscv_vlse8_v_i8m1((nk_i8_t const *)ptr, (nk_ssize_t)stride_bytes, vector_length);
        vint16m2_t data_i16m2 = __riscv_vwadd_vx_i16m2(data_i8m1, 0, vector_length);
        chunk_i32m1 = __riscv_vwredsum_vs_i16m2_i32m1(data_i16m2, chunk_i32m1, vector_length);
        if (++drain_counter >= 16384) {
            sum += (nk_i64_t)__riscv_vmv_x_s_i32m1_i32(chunk_i32m1);
            chunk_i32m1 = __riscv_vmv_v_x_i32m1(0, 1);
            drain_counter = 0;
        }
    }
    *result = sum + (nk_i64_t)__riscv_vmv_x_s_i32m1_i32(chunk_i32m1);
}

NK_PUBLIC void nk_reduce_add_i8_rvv(                              //
    nk_i8_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *result) {
    if (count == 0) *result = 0;
    else if (stride_bytes == sizeof(nk_i8_t)) nk_reduce_add_i8_rvv_contiguous_(data, count, result);
    else nk_reduce_add_i8_rvv_strided_(data, count, stride_bytes, result);
}

NK_INTERNAL void nk_reduce_add_u8_rvv_contiguous_( //
    nk_u8_t const *data, nk_size_t count,          //
    nk_u64_t *result) {
    nk_u64_t sum = 0;
    vuint32m1_t chunk_u32m1 = __riscv_vmv_v_x_u32m1(0, 1);
    nk_size_t drain_counter = 0;
    for (nk_size_t vector_length; count > 0; count -= vector_length, data += vector_length) {
        vector_length = __riscv_vsetvl_e8m1(count);
        vuint8m1_t data_u8m1 = __riscv_vle8_v_u8m1(data, vector_length);
        vuint16m2_t data_u16m2 = __riscv_vwaddu_vx_u16m2(data_u8m1, 0, vector_length);
        chunk_u32m1 = __riscv_vwredsumu_vs_u16m2_u32m1(data_u16m2, chunk_u32m1, vector_length);
        if (++drain_counter >= 16384) {
            sum += (nk_u64_t)__riscv_vmv_x_s_u32m1_u32(chunk_u32m1);
            chunk_u32m1 = __riscv_vmv_v_x_u32m1(0, 1);
            drain_counter = 0;
        }
    }
    *result = sum + (nk_u64_t)__riscv_vmv_x_s_u32m1_u32(chunk_u32m1);
}

NK_INTERNAL void nk_reduce_add_u8_rvv_strided_(                   //
    nk_u8_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *result) {
    nk_u64_t sum = 0;
    vuint32m1_t chunk_u32m1 = __riscv_vmv_v_x_u32m1(0, 1);
    nk_size_t drain_counter = 0;
    unsigned char const *ptr = (unsigned char const *)data;
    for (nk_size_t vector_length; count > 0; count -= vector_length, ptr += vector_length * stride_bytes) {
        vector_length = __riscv_vsetvl_e8m1(count);
        vuint8m1_t data_u8m1 = __riscv_vlse8_v_u8m1((nk_u8_t const *)ptr, (nk_ssize_t)stride_bytes, vector_length);
        vuint16m2_t data_u16m2 = __riscv_vwaddu_vx_u16m2(data_u8m1, 0, vector_length);
        chunk_u32m1 = __riscv_vwredsumu_vs_u16m2_u32m1(data_u16m2, chunk_u32m1, vector_length);
        if (++drain_counter >= 16384) {
            sum += (nk_u64_t)__riscv_vmv_x_s_u32m1_u32(chunk_u32m1);
            chunk_u32m1 = __riscv_vmv_v_x_u32m1(0, 1);
            drain_counter = 0;
        }
    }
    *result = sum + (nk_u64_t)__riscv_vmv_x_s_u32m1_u32(chunk_u32m1);
}

NK_PUBLIC void nk_reduce_add_u8_rvv(                              //
    nk_u8_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *result) {
    if (count == 0) *result = 0;
    else if (stride_bytes == sizeof(nk_u8_t)) nk_reduce_add_u8_rvv_contiguous_(data, count, result);
    else nk_reduce_add_u8_rvv_strided_(data, count, stride_bytes, result);
}

NK_INTERNAL void nk_reduce_add_i16_rvv_contiguous_( //
    nk_i16_t const *data, nk_size_t count,          //
    nk_i64_t *result) {
    nk_i64_t sum = 0;
    vint32m1_t chunk_i32m1 = __riscv_vmv_v_x_i32m1(0, 1);
    nk_size_t drain_counter = 0;
    for (nk_size_t vector_length; count > 0; count -= vector_length, data += vector_length) {
        vector_length = __riscv_vsetvl_e16m1(count);
        vint16m1_t data_i16m1 = __riscv_vle16_v_i16m1(data, vector_length);
        chunk_i32m1 = __riscv_vwredsum_vs_i16m1_i32m1(data_i16m1, chunk_i32m1, vector_length);
        if (++drain_counter >= 32768) {
            sum += (nk_i64_t)__riscv_vmv_x_s_i32m1_i32(chunk_i32m1);
            chunk_i32m1 = __riscv_vmv_v_x_i32m1(0, 1);
            drain_counter = 0;
        }
    }
    *result = sum + (nk_i64_t)__riscv_vmv_x_s_i32m1_i32(chunk_i32m1);
}

NK_INTERNAL void nk_reduce_add_i16_rvv_strided_(                   //
    nk_i16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *result) {
    nk_i64_t sum = 0;
    vint32m1_t chunk_i32m1 = __riscv_vmv_v_x_i32m1(0, 1);
    nk_size_t drain_counter = 0;
    unsigned char const *ptr = (unsigned char const *)data;
    for (nk_size_t vector_length; count > 0; count -= vector_length, ptr += vector_length * stride_bytes) {
        vector_length = __riscv_vsetvl_e16m1(count);
        vint16m1_t data_i16m1 = __riscv_vlse16_v_i16m1((nk_i16_t const *)ptr, (nk_ssize_t)stride_bytes, vector_length);
        chunk_i32m1 = __riscv_vwredsum_vs_i16m1_i32m1(data_i16m1, chunk_i32m1, vector_length);
        if (++drain_counter >= 32768) {
            sum += (nk_i64_t)__riscv_vmv_x_s_i32m1_i32(chunk_i32m1);
            chunk_i32m1 = __riscv_vmv_v_x_i32m1(0, 1);
            drain_counter = 0;
        }
    }
    *result = sum + (nk_i64_t)__riscv_vmv_x_s_i32m1_i32(chunk_i32m1);
}

NK_PUBLIC void nk_reduce_add_i16_rvv(                              //
    nk_i16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *result) {
    if (count == 0) *result = 0;
    else if (stride_bytes == sizeof(nk_i16_t)) nk_reduce_add_i16_rvv_contiguous_(data, count, result);
    else nk_reduce_add_i16_rvv_strided_(data, count, stride_bytes, result);
}

NK_INTERNAL void nk_reduce_add_u16_rvv_contiguous_( //
    nk_u16_t const *data, nk_size_t count,          //
    nk_u64_t *result) {
    nk_u64_t sum = 0;
    vuint32m1_t chunk_u32m1 = __riscv_vmv_v_x_u32m1(0, 1);
    nk_size_t drain_counter = 0;
    for (nk_size_t vector_length; count > 0; count -= vector_length, data += vector_length) {
        vector_length = __riscv_vsetvl_e16m1(count);
        vuint16m1_t data_u16m1 = __riscv_vle16_v_u16m1(data, vector_length);
        chunk_u32m1 = __riscv_vwredsumu_vs_u16m1_u32m1(data_u16m1, chunk_u32m1, vector_length);
        if (++drain_counter >= 32768) {
            sum += (nk_u64_t)__riscv_vmv_x_s_u32m1_u32(chunk_u32m1);
            chunk_u32m1 = __riscv_vmv_v_x_u32m1(0, 1);
            drain_counter = 0;
        }
    }
    *result = sum + (nk_u64_t)__riscv_vmv_x_s_u32m1_u32(chunk_u32m1);
}

NK_INTERNAL void nk_reduce_add_u16_rvv_strided_(                   //
    nk_u16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *result) {
    nk_u64_t sum = 0;
    vuint32m1_t chunk_u32m1 = __riscv_vmv_v_x_u32m1(0, 1);
    nk_size_t drain_counter = 0;
    unsigned char const *ptr = (unsigned char const *)data;
    for (nk_size_t vector_length; count > 0; count -= vector_length, ptr += vector_length * stride_bytes) {
        vector_length = __riscv_vsetvl_e16m1(count);
        vuint16m1_t data_u16m1 = __riscv_vlse16_v_u16m1((nk_u16_t const *)ptr, (nk_ssize_t)stride_bytes, vector_length);
        chunk_u32m1 = __riscv_vwredsumu_vs_u16m1_u32m1(data_u16m1, chunk_u32m1, vector_length);
        if (++drain_counter >= 32768) {
            sum += (nk_u64_t)__riscv_vmv_x_s_u32m1_u32(chunk_u32m1);
            chunk_u32m1 = __riscv_vmv_v_x_u32m1(0, 1);
            drain_counter = 0;
        }
    }
    *result = sum + (nk_u64_t)__riscv_vmv_x_s_u32m1_u32(chunk_u32m1);
}

NK_PUBLIC void nk_reduce_add_u16_rvv(                              //
    nk_u16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *result) {
    if (count == 0) *result = 0;
    else if (stride_bytes == sizeof(nk_u16_t)) nk_reduce_add_u16_rvv_contiguous_(data, count, result);
    else nk_reduce_add_u16_rvv_strided_(data, count, stride_bytes, result);
}

NK_INTERNAL void nk_reduce_add_i32_rvv_contiguous_( //
    nk_i32_t const *data, nk_size_t count,          //
    nk_i64_t *result) {
    vint64m1_t sum_i64m1 = __riscv_vmv_v_x_i64m1(0, 1);
    for (nk_size_t vector_length; count > 0; count -= vector_length, data += vector_length) {
        vector_length = __riscv_vsetvl_e32m1(count);
        vint32m1_t data_i32m1 = __riscv_vle32_v_i32m1(data, vector_length);
        sum_i64m1 = __riscv_vwredsum_vs_i32m1_i64m1(data_i32m1, sum_i64m1, vector_length);
    }
    *result = __riscv_vmv_x_s_i64m1_i64(sum_i64m1);
}

NK_INTERNAL void nk_reduce_add_i32_rvv_strided_(                   //
    nk_i32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *result) {
    vint64m1_t sum_i64m1 = __riscv_vmv_v_x_i64m1(0, 1);
    unsigned char const *ptr = (unsigned char const *)data;
    for (nk_size_t vector_length; count > 0; count -= vector_length, ptr += vector_length * stride_bytes) {
        vector_length = __riscv_vsetvl_e32m1(count);
        vint32m1_t data_i32m1 = __riscv_vlse32_v_i32m1((nk_i32_t const *)ptr, (nk_ssize_t)stride_bytes, vector_length);
        sum_i64m1 = __riscv_vwredsum_vs_i32m1_i64m1(data_i32m1, sum_i64m1, vector_length);
    }
    *result = __riscv_vmv_x_s_i64m1_i64(sum_i64m1);
}

NK_PUBLIC void nk_reduce_add_i32_rvv(                              //
    nk_i32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *result) {
    if (count == 0) *result = 0;
    else if (stride_bytes == sizeof(nk_i32_t)) nk_reduce_add_i32_rvv_contiguous_(data, count, result);
    else nk_reduce_add_i32_rvv_strided_(data, count, stride_bytes, result);
}

NK_INTERNAL void nk_reduce_add_u32_rvv_contiguous_( //
    nk_u32_t const *data, nk_size_t count,          //
    nk_u64_t *result) {
    vuint64m1_t sum_u64m1 = __riscv_vmv_v_x_u64m1(0, 1);
    for (nk_size_t vector_length; count > 0; count -= vector_length, data += vector_length) {
        vector_length = __riscv_vsetvl_e32m1(count);
        vuint32m1_t data_u32m1 = __riscv_vle32_v_u32m1(data, vector_length);
        sum_u64m1 = __riscv_vwredsumu_vs_u32m1_u64m1(data_u32m1, sum_u64m1, vector_length);
    }
    *result = __riscv_vmv_x_s_u64m1_u64(sum_u64m1);
}

NK_INTERNAL void nk_reduce_add_u32_rvv_strided_(                   //
    nk_u32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *result) {
    vuint64m1_t sum_u64m1 = __riscv_vmv_v_x_u64m1(0, 1);
    unsigned char const *ptr = (unsigned char const *)data;
    for (nk_size_t vector_length; count > 0; count -= vector_length, ptr += vector_length * stride_bytes) {
        vector_length = __riscv_vsetvl_e32m1(count);
        vuint32m1_t data_u32m1 = __riscv_vlse32_v_u32m1((nk_u32_t const *)ptr, (nk_ssize_t)stride_bytes, vector_length);
        sum_u64m1 = __riscv_vwredsumu_vs_u32m1_u64m1(data_u32m1, sum_u64m1, vector_length);
    }
    *result = __riscv_vmv_x_s_u64m1_u64(sum_u64m1);
}

NK_PUBLIC void nk_reduce_add_u32_rvv(                              //
    nk_u32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *result) {
    if (count == 0) *result = 0;
    else if (stride_bytes == sizeof(nk_u32_t)) nk_reduce_add_u32_rvv_contiguous_(data, count, result);
    else nk_reduce_add_u32_rvv_strided_(data, count, stride_bytes, result);
}

NK_INTERNAL void nk_reduce_add_i64_rvv_contiguous_( //
    nk_i64_t const *data, nk_size_t count,          //
    nk_i64_t *result) {
    vint64m1_t sum_i64m1 = __riscv_vmv_v_x_i64m1(0, 1);
    for (nk_size_t vector_length; count > 0; count -= vector_length, data += vector_length) {
        vector_length = __riscv_vsetvl_e64m1(count);
        vint64m1_t data_i64m1 = __riscv_vle64_v_i64m1(data, vector_length);
        sum_i64m1 = __riscv_vredsum_vs_i64m1_i64m1(data_i64m1, sum_i64m1, vector_length);
    }
    *result = __riscv_vmv_x_s_i64m1_i64(sum_i64m1);
}

NK_INTERNAL void nk_reduce_add_i64_rvv_strided_(                   //
    nk_i64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *result) {
    vint64m1_t sum_i64m1 = __riscv_vmv_v_x_i64m1(0, 1);
    unsigned char const *ptr = (unsigned char const *)data;
    for (nk_size_t vector_length; count > 0; count -= vector_length, ptr += vector_length * stride_bytes) {
        vector_length = __riscv_vsetvl_e64m1(count);
        vint64m1_t data_i64m1 = __riscv_vlse64_v_i64m1((nk_i64_t const *)ptr, (nk_ssize_t)stride_bytes, vector_length);
        sum_i64m1 = __riscv_vredsum_vs_i64m1_i64m1(data_i64m1, sum_i64m1, vector_length);
    }
    *result = __riscv_vmv_x_s_i64m1_i64(sum_i64m1);
}

NK_PUBLIC void nk_reduce_add_i64_rvv(                              //
    nk_i64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *result) {
    if (count == 0) *result = 0;
    else if (stride_bytes == sizeof(nk_i64_t)) nk_reduce_add_i64_rvv_contiguous_(data, count, result);
    else nk_reduce_add_i64_rvv_strided_(data, count, stride_bytes, result);
}

NK_INTERNAL void nk_reduce_add_u64_rvv_contiguous_( //
    nk_u64_t const *data, nk_size_t count,          //
    nk_u64_t *result) {
    vuint64m1_t sum_u64m1 = __riscv_vmv_v_x_u64m1(0, 1);
    for (nk_size_t vector_length; count > 0; count -= vector_length, data += vector_length) {
        vector_length = __riscv_vsetvl_e64m1(count);
        vuint64m1_t data_u64m1 = __riscv_vle64_v_u64m1(data, vector_length);
        sum_u64m1 = __riscv_vredsum_vs_u64m1_u64m1(data_u64m1, sum_u64m1, vector_length);
    }
    *result = __riscv_vmv_x_s_u64m1_u64(sum_u64m1);
}

NK_INTERNAL void nk_reduce_add_u64_rvv_strided_(                   //
    nk_u64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *result) {
    vuint64m1_t sum_u64m1 = __riscv_vmv_v_x_u64m1(0, 1);
    unsigned char const *ptr = (unsigned char const *)data;
    for (nk_size_t vector_length; count > 0; count -= vector_length, ptr += vector_length * stride_bytes) {
        vector_length = __riscv_vsetvl_e64m1(count);
        vuint64m1_t data_u64m1 = __riscv_vlse64_v_u64m1((nk_u64_t const *)ptr, (nk_ssize_t)stride_bytes, vector_length);
        sum_u64m1 = __riscv_vredsum_vs_u64m1_u64m1(data_u64m1, sum_u64m1, vector_length);
    }
    *result = __riscv_vmv_x_s_u64m1_u64(sum_u64m1);
}

NK_PUBLIC void nk_reduce_add_u64_rvv(                              //
    nk_u64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *result) {
    if (count == 0) *result = 0;
    else if (stride_bytes == sizeof(nk_u64_t)) nk_reduce_add_u64_rvv_contiguous_(data, count, result);
    else nk_reduce_add_u64_rvv_strided_(data, count, stride_bytes, result);
}

NK_INTERNAL void nk_reduce_add_f16_rvv_contiguous_( //
    nk_f16_t const *data, nk_size_t count,          //
    nk_f32_t *result) {
    nk_size_t vlmax = __riscv_vsetvlmax_e32m2();
    vfloat32m2_t sum_f32m2 = __riscv_vfmv_v_f_f32m2(0.0f, vlmax);
    for (nk_size_t vector_length; count > 0; count -= vector_length, data += vector_length) {
        vector_length = __riscv_vsetvl_e16m1(count);
        vuint16m1_t data_u16m1 = __riscv_vle16_v_u16m1((nk_u16_t const *)data, vector_length);
        vfloat32m2_t data_f32m2 = nk_f16m1_to_f32m2_rvv_(data_u16m1, vector_length);
        sum_f32m2 = __riscv_vfadd_vv_f32m2(sum_f32m2, data_f32m2, vector_length);
    }
    vfloat32m1_t zero_f32m1 = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    *result = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m2_f32m1(sum_f32m2, zero_f32m1, vlmax));
}

NK_INTERNAL void nk_reduce_add_f16_rvv_strided_(                   //
    nk_f16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *result) {
    nk_size_t vlmax = __riscv_vsetvlmax_e32m2();
    vfloat32m2_t sum_f32m2 = __riscv_vfmv_v_f_f32m2(0.0f, vlmax);
    unsigned char const *ptr = (unsigned char const *)data;
    for (nk_size_t vector_length; count > 0; count -= vector_length, ptr += vector_length * stride_bytes) {
        vector_length = __riscv_vsetvl_e16m1(count);
        vuint16m1_t data_u16m1 = __riscv_vlse16_v_u16m1((nk_u16_t const *)ptr, (nk_ssize_t)stride_bytes, vector_length);
        vfloat32m2_t data_f32m2 = nk_f16m1_to_f32m2_rvv_(data_u16m1, vector_length);
        sum_f32m2 = __riscv_vfadd_vv_f32m2(sum_f32m2, data_f32m2, vector_length);
    }
    vfloat32m1_t zero_f32m1 = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    *result = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m2_f32m1(sum_f32m2, zero_f32m1, vlmax));
}

NK_PUBLIC void nk_reduce_add_f16_rvv(                              //
    nk_f16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *result) {
    if (count == 0) *result = 0;
    else if (stride_bytes == sizeof(nk_f16_t)) nk_reduce_add_f16_rvv_contiguous_(data, count, result);
    else nk_reduce_add_f16_rvv_strided_(data, count, stride_bytes, result);
}

NK_INTERNAL void nk_reduce_add_bf16_rvv_contiguous_( //
    nk_bf16_t const *data, nk_size_t count,          //
    nk_f32_t *result) {
    nk_size_t vlmax = __riscv_vsetvlmax_e32m2();
    vfloat32m2_t sum_f32m2 = __riscv_vfmv_v_f_f32m2(0.0f, vlmax);
    for (nk_size_t vector_length; count > 0; count -= vector_length, data += vector_length) {
        vector_length = __riscv_vsetvl_e16m1(count);
        vuint16m1_t data_u16m1 = __riscv_vle16_v_u16m1((nk_u16_t const *)data, vector_length);
        vfloat32m2_t data_f32m2 = nk_bf16m1_to_f32m2_rvv_(data_u16m1, vector_length);
        sum_f32m2 = __riscv_vfadd_vv_f32m2(sum_f32m2, data_f32m2, vector_length);
    }
    vfloat32m1_t zero_f32m1 = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    *result = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m2_f32m1(sum_f32m2, zero_f32m1, vlmax));
}

NK_INTERNAL void nk_reduce_add_bf16_rvv_strided_(                   //
    nk_bf16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *result) {
    nk_size_t vlmax = __riscv_vsetvlmax_e32m2();
    vfloat32m2_t sum_f32m2 = __riscv_vfmv_v_f_f32m2(0.0f, vlmax);
    unsigned char const *ptr = (unsigned char const *)data;
    for (nk_size_t vector_length; count > 0; count -= vector_length, ptr += vector_length * stride_bytes) {
        vector_length = __riscv_vsetvl_e16m1(count);
        vuint16m1_t data_u16m1 = __riscv_vlse16_v_u16m1((nk_u16_t const *)ptr, (nk_ssize_t)stride_bytes, vector_length);
        vfloat32m2_t data_f32m2 = nk_bf16m1_to_f32m2_rvv_(data_u16m1, vector_length);
        sum_f32m2 = __riscv_vfadd_vv_f32m2(sum_f32m2, data_f32m2, vector_length);
    }
    vfloat32m1_t zero_f32m1 = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    *result = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m2_f32m1(sum_f32m2, zero_f32m1, vlmax));
}

NK_PUBLIC void nk_reduce_add_bf16_rvv(                              //
    nk_bf16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *result) {
    if (count == 0) *result = 0;
    else if (stride_bytes == sizeof(nk_bf16_t)) nk_reduce_add_bf16_rvv_contiguous_(data, count, result);
    else nk_reduce_add_bf16_rvv_strided_(data, count, stride_bytes, result);
}

NK_INTERNAL void nk_reduce_add_e4m3_rvv_contiguous_( //
    nk_e4m3_t const *data, nk_size_t count,          //
    nk_f32_t *result) {
    nk_size_t vlmax = __riscv_vsetvlmax_e32m4();
    vfloat32m4_t sum_f32m4 = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);
    for (nk_size_t vector_length; count > 0; count -= vector_length, data += vector_length) {
        vector_length = __riscv_vsetvl_e8m1(count);
        vuint8m1_t data_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)data, vector_length);
        vfloat32m4_t data_f32m4 = nk_e4m3m1_to_f32m4_rvv_(data_u8m1, vector_length);
        sum_f32m4 = __riscv_vfadd_vv_f32m4(sum_f32m4, data_f32m4, vector_length);
    }
    vfloat32m1_t zero_f32m1 = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    *result = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(sum_f32m4, zero_f32m1, vlmax));
}

NK_INTERNAL void nk_reduce_add_e4m3_rvv_strided_(                   //
    nk_e4m3_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *result) {
    nk_size_t vlmax = __riscv_vsetvlmax_e32m4();
    vfloat32m4_t sum_f32m4 = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);
    unsigned char const *ptr = (unsigned char const *)data;
    for (nk_size_t vector_length; count > 0; count -= vector_length, ptr += vector_length * stride_bytes) {
        vector_length = __riscv_vsetvl_e8m1(count);
        vuint8m1_t data_u8m1 = __riscv_vlse8_v_u8m1((nk_u8_t const *)ptr, (nk_ssize_t)stride_bytes, vector_length);
        vfloat32m4_t data_f32m4 = nk_e4m3m1_to_f32m4_rvv_(data_u8m1, vector_length);
        sum_f32m4 = __riscv_vfadd_vv_f32m4(sum_f32m4, data_f32m4, vector_length);
    }
    vfloat32m1_t zero_f32m1 = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    *result = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(sum_f32m4, zero_f32m1, vlmax));
}

NK_PUBLIC void nk_reduce_add_e4m3_rvv(                              //
    nk_e4m3_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *result) {
    if (count == 0) *result = 0;
    else if (stride_bytes == sizeof(nk_e4m3_t)) nk_reduce_add_e4m3_rvv_contiguous_(data, count, result);
    else nk_reduce_add_e4m3_rvv_strided_(data, count, stride_bytes, result);
}

NK_INTERNAL void nk_reduce_add_e5m2_rvv_contiguous_( //
    nk_e5m2_t const *data, nk_size_t count,          //
    nk_f32_t *result) {
    nk_size_t vlmax = __riscv_vsetvlmax_e32m4();
    vfloat32m4_t sum_f32m4 = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);
    for (nk_size_t vector_length; count > 0; count -= vector_length, data += vector_length) {
        vector_length = __riscv_vsetvl_e8m1(count);
        vuint8m1_t data_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)data, vector_length);
        vfloat32m4_t data_f32m4 = nk_e5m2m1_to_f32m4_rvv_(data_u8m1, vector_length);
        sum_f32m4 = __riscv_vfadd_vv_f32m4(sum_f32m4, data_f32m4, vector_length);
    }
    vfloat32m1_t zero_f32m1 = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    *result = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(sum_f32m4, zero_f32m1, vlmax));
}

NK_INTERNAL void nk_reduce_add_e5m2_rvv_strided_(                   //
    nk_e5m2_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *result) {
    nk_size_t vlmax = __riscv_vsetvlmax_e32m4();
    vfloat32m4_t sum_f32m4 = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);
    unsigned char const *ptr = (unsigned char const *)data;
    for (nk_size_t vector_length; count > 0; count -= vector_length, ptr += vector_length * stride_bytes) {
        vector_length = __riscv_vsetvl_e8m1(count);
        vuint8m1_t data_u8m1 = __riscv_vlse8_v_u8m1((nk_u8_t const *)ptr, (nk_ssize_t)stride_bytes, vector_length);
        vfloat32m4_t data_f32m4 = nk_e5m2m1_to_f32m4_rvv_(data_u8m1, vector_length);
        sum_f32m4 = __riscv_vfadd_vv_f32m4(sum_f32m4, data_f32m4, vector_length);
    }
    vfloat32m1_t zero_f32m1 = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    *result = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(sum_f32m4, zero_f32m1, vlmax));
}

NK_PUBLIC void nk_reduce_add_e5m2_rvv(                              //
    nk_e5m2_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *result) {
    if (count == 0) *result = 0;
    else if (stride_bytes == sizeof(nk_e5m2_t)) nk_reduce_add_e5m2_rvv_contiguous_(data, count, result);
    else nk_reduce_add_e5m2_rvv_strided_(data, count, stride_bytes, result);
}

NK_INTERNAL void nk_reduce_add_e2m3_rvv_contiguous_( //
    nk_e2m3_t const *data, nk_size_t count,          //
    nk_f32_t *result) {
    nk_size_t vlmax = __riscv_vsetvlmax_e32m4();
    vfloat32m4_t sum_f32m4 = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);
    for (nk_size_t vector_length; count > 0; count -= vector_length, data += vector_length) {
        vector_length = __riscv_vsetvl_e8m1(count);
        vuint8m1_t data_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)data, vector_length);
        vfloat32m4_t data_f32m4 = nk_e2m3m1_to_f32m4_rvv_(data_u8m1, vector_length);
        sum_f32m4 = __riscv_vfadd_vv_f32m4(sum_f32m4, data_f32m4, vector_length);
    }
    vfloat32m1_t zero_f32m1 = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    *result = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(sum_f32m4, zero_f32m1, vlmax));
}

NK_INTERNAL void nk_reduce_add_e2m3_rvv_strided_(                   //
    nk_e2m3_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *result) {
    nk_size_t vlmax = __riscv_vsetvlmax_e32m4();
    vfloat32m4_t sum_f32m4 = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);
    unsigned char const *ptr = (unsigned char const *)data;
    for (nk_size_t vector_length; count > 0; count -= vector_length, ptr += vector_length * stride_bytes) {
        vector_length = __riscv_vsetvl_e8m1(count);
        vuint8m1_t data_u8m1 = __riscv_vlse8_v_u8m1((nk_u8_t const *)ptr, (nk_ssize_t)stride_bytes, vector_length);
        vfloat32m4_t data_f32m4 = nk_e2m3m1_to_f32m4_rvv_(data_u8m1, vector_length);
        sum_f32m4 = __riscv_vfadd_vv_f32m4(sum_f32m4, data_f32m4, vector_length);
    }
    vfloat32m1_t zero_f32m1 = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    *result = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(sum_f32m4, zero_f32m1, vlmax));
}

NK_PUBLIC void nk_reduce_add_e2m3_rvv(                              //
    nk_e2m3_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *result) {
    if (count == 0) *result = 0;
    else if (stride_bytes == sizeof(nk_e2m3_t)) nk_reduce_add_e2m3_rvv_contiguous_(data, count, result);
    else nk_reduce_add_e2m3_rvv_strided_(data, count, stride_bytes, result);
}

NK_INTERNAL void nk_reduce_add_e3m2_rvv_contiguous_( //
    nk_e3m2_t const *data, nk_size_t count,          //
    nk_f32_t *result) {
    nk_size_t vlmax = __riscv_vsetvlmax_e32m4();
    vfloat32m4_t sum_f32m4 = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);
    for (nk_size_t vector_length; count > 0; count -= vector_length, data += vector_length) {
        vector_length = __riscv_vsetvl_e8m1(count);
        vuint8m1_t data_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)data, vector_length);
        vfloat32m4_t data_f32m4 = nk_e3m2m1_to_f32m4_rvv_(data_u8m1, vector_length);
        sum_f32m4 = __riscv_vfadd_vv_f32m4(sum_f32m4, data_f32m4, vector_length);
    }
    vfloat32m1_t zero_f32m1 = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    *result = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(sum_f32m4, zero_f32m1, vlmax));
}

NK_INTERNAL void nk_reduce_add_e3m2_rvv_strided_(                   //
    nk_e3m2_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *result) {
    nk_size_t vlmax = __riscv_vsetvlmax_e32m4();
    vfloat32m4_t sum_f32m4 = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);
    unsigned char const *ptr = (unsigned char const *)data;
    for (nk_size_t vector_length; count > 0; count -= vector_length, ptr += vector_length * stride_bytes) {
        vector_length = __riscv_vsetvl_e8m1(count);
        vuint8m1_t data_u8m1 = __riscv_vlse8_v_u8m1((nk_u8_t const *)ptr, (nk_ssize_t)stride_bytes, vector_length);
        vfloat32m4_t data_f32m4 = nk_e3m2m1_to_f32m4_rvv_(data_u8m1, vector_length);
        sum_f32m4 = __riscv_vfadd_vv_f32m4(sum_f32m4, data_f32m4, vector_length);
    }
    vfloat32m1_t zero_f32m1 = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    *result = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(sum_f32m4, zero_f32m1, vlmax));
}

NK_PUBLIC void nk_reduce_add_e3m2_rvv(                              //
    nk_e3m2_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *result) {
    if (count == 0) *result = 0;
    else if (stride_bytes == sizeof(nk_e3m2_t)) nk_reduce_add_e3m2_rvv_contiguous_(data, count, result);
    else nk_reduce_add_e3m2_rvv_strided_(data, count, stride_bytes, result);
}

#pragma endregion - Add Reductions

#pragma region - Min / Max Reductions

NK_INTERNAL void nk_reduce_min_f32_rvv_contiguous_( //
    nk_f32_t const *data, nk_size_t count,          //
    nk_f32_t *min_value, nk_size_t *min_index) {

    nk_size_t vector_length_max = __riscv_vsetvlmax_e32m1();
    vfloat32m1_t best_f32m1 = __riscv_vfmv_v_f_f32m1(NK_F32_MAX, vector_length_max);
    vuint64m2_t best_index_u64m2 = __riscv_vmv_v_x_u64m2(0, vector_length_max);

    nk_size_t offset = 0;
    for (nk_size_t remaining = count, vector_length; remaining > 0;
         remaining -= vector_length, offset += vector_length) {
        vector_length = __riscv_vsetvl_e32m1(remaining);
        vfloat32m1_t chunk_f32m1 = __riscv_vle32_v_f32m1(data + offset, vector_length);

        vbool32_t less_b32 = __riscv_vmflt_vv_f32m1_b32(chunk_f32m1, best_f32m1, vector_length);
        best_f32m1 = __riscv_vmerge_vvm_f32m1(best_f32m1, chunk_f32m1, less_b32, vector_length);

        vuint64m2_t position_u64m2 = __riscv_vid_v_u64m2(vector_length);
        position_u64m2 = __riscv_vadd_vx_u64m2(position_u64m2, (nk_u64_t)offset, vector_length);
        best_index_u64m2 = __riscv_vmerge_vvm_u64m2(best_index_u64m2, position_u64m2, less_b32, vector_length);
    }

    vfloat32m1_t identity_f32m1 = __riscv_vfmv_v_f_f32m1(NK_F32_MAX, 1);
    vfloat32m1_t reduced_f32m1 = __riscv_vfredmin_vs_f32m1_f32m1(best_f32m1, identity_f32m1, vector_length_max);
    nk_f32_t best = __riscv_vfmv_f_s_f32m1_f32(reduced_f32m1);

    vbool32_t match_b32 = __riscv_vmfeq_vf_f32m1_b32(best_f32m1, best, vector_length_max);
    vuint64m2_t sentinel_u64m2 = __riscv_vmv_v_x_u64m2(NK_U64_MAX, vector_length_max);
    vuint64m2_t candidates_u64m2 = __riscv_vmerge_vvm_u64m2(sentinel_u64m2, best_index_u64m2, match_b32,
                                                            vector_length_max);
    vuint64m1_t identity_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    vuint64m1_t reduced_u64m1 = __riscv_vredminu_vs_u64m2_u64m1(candidates_u64m2, identity_u64m1, vector_length_max);

    *min_value = best;
    *min_index = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(reduced_u64m1);
}

NK_INTERNAL void nk_reduce_min_f32_rvv_strided_(                   //
    nk_f32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *min_value, nk_size_t *min_index) {

    nk_size_t vector_length_max = __riscv_vsetvlmax_e32m1();
    vfloat32m1_t best_f32m1 = __riscv_vfmv_v_f_f32m1(NK_F32_MAX, vector_length_max);
    vuint64m2_t best_index_u64m2 = __riscv_vmv_v_x_u64m2(0, vector_length_max);

    unsigned char const *ptr = (unsigned char const *)data;
    nk_size_t offset = 0;
    for (nk_size_t remaining = count, vector_length; remaining > 0;
         remaining -= vector_length, offset += vector_length, ptr += vector_length * stride_bytes) {
        vector_length = __riscv_vsetvl_e32m1(remaining);
        vfloat32m1_t chunk_f32m1 = __riscv_vlse32_v_f32m1((nk_f32_t const *)ptr, (nk_ssize_t)stride_bytes,
                                                          vector_length);

        vbool32_t less_b32 = __riscv_vmflt_vv_f32m1_b32(chunk_f32m1, best_f32m1, vector_length);
        best_f32m1 = __riscv_vmerge_vvm_f32m1(best_f32m1, chunk_f32m1, less_b32, vector_length);

        vuint64m2_t position_u64m2 = __riscv_vid_v_u64m2(vector_length);
        position_u64m2 = __riscv_vadd_vx_u64m2(position_u64m2, (nk_u64_t)offset, vector_length);
        best_index_u64m2 = __riscv_vmerge_vvm_u64m2(best_index_u64m2, position_u64m2, less_b32, vector_length);
    }

    vfloat32m1_t identity_f32m1 = __riscv_vfmv_v_f_f32m1(NK_F32_MAX, 1);
    vfloat32m1_t reduced_f32m1 = __riscv_vfredmin_vs_f32m1_f32m1(best_f32m1, identity_f32m1, vector_length_max);
    nk_f32_t best = __riscv_vfmv_f_s_f32m1_f32(reduced_f32m1);

    vbool32_t match_b32 = __riscv_vmfeq_vf_f32m1_b32(best_f32m1, best, vector_length_max);
    vuint64m2_t sentinel_u64m2 = __riscv_vmv_v_x_u64m2(NK_U64_MAX, vector_length_max);
    vuint64m2_t candidates_u64m2 = __riscv_vmerge_vvm_u64m2(sentinel_u64m2, best_index_u64m2, match_b32,
                                                            vector_length_max);
    vuint64m1_t identity_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    vuint64m1_t reduced_u64m1 = __riscv_vredminu_vs_u64m2_u64m1(candidates_u64m2, identity_u64m1, vector_length_max);

    *min_value = best;
    *min_index = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(reduced_u64m1);
}

NK_PUBLIC void nk_reduce_min_f32_rvv(                              //
    nk_f32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *min_value, nk_size_t *min_index) {
    if (count == 0) *min_value = NK_F32_MAX, *min_index = 0;
    else if (stride_bytes == sizeof(nk_f32_t)) nk_reduce_min_f32_rvv_contiguous_(data, count, min_value, min_index);
    else nk_reduce_min_f32_rvv_strided_(data, count, stride_bytes, min_value, min_index);
}

NK_INTERNAL void nk_reduce_max_f32_rvv_contiguous_( //
    nk_f32_t const *data, nk_size_t count,          //
    nk_f32_t *max_value, nk_size_t *max_index) {

    nk_size_t vector_length_max = __riscv_vsetvlmax_e32m1();
    vfloat32m1_t best_f32m1 = __riscv_vfmv_v_f_f32m1(NK_F32_MIN, vector_length_max);
    vuint64m2_t best_index_u64m2 = __riscv_vmv_v_x_u64m2(0, vector_length_max);

    nk_size_t offset = 0;
    for (nk_size_t remaining = count, vector_length; remaining > 0;
         remaining -= vector_length, offset += vector_length) {
        vector_length = __riscv_vsetvl_e32m1(remaining);
        vfloat32m1_t chunk_f32m1 = __riscv_vle32_v_f32m1(data + offset, vector_length);

        vbool32_t greater_b32 = __riscv_vmflt_vv_f32m1_b32(best_f32m1, chunk_f32m1, vector_length);
        best_f32m1 = __riscv_vmerge_vvm_f32m1(best_f32m1, chunk_f32m1, greater_b32, vector_length);

        vuint64m2_t position_u64m2 = __riscv_vid_v_u64m2(vector_length);
        position_u64m2 = __riscv_vadd_vx_u64m2(position_u64m2, (nk_u64_t)offset, vector_length);
        best_index_u64m2 = __riscv_vmerge_vvm_u64m2(best_index_u64m2, position_u64m2, greater_b32, vector_length);
    }

    vfloat32m1_t identity_f32m1 = __riscv_vfmv_v_f_f32m1(NK_F32_MIN, 1);
    vfloat32m1_t reduced_f32m1 = __riscv_vfredmax_vs_f32m1_f32m1(best_f32m1, identity_f32m1, vector_length_max);
    nk_f32_t best = __riscv_vfmv_f_s_f32m1_f32(reduced_f32m1);

    vbool32_t match_b32 = __riscv_vmfeq_vf_f32m1_b32(best_f32m1, best, vector_length_max);
    vuint64m2_t sentinel_u64m2 = __riscv_vmv_v_x_u64m2(NK_U64_MAX, vector_length_max);
    vuint64m2_t candidates_u64m2 = __riscv_vmerge_vvm_u64m2(sentinel_u64m2, best_index_u64m2, match_b32,
                                                            vector_length_max);
    vuint64m1_t identity_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    vuint64m1_t reduced_u64m1 = __riscv_vredminu_vs_u64m2_u64m1(candidates_u64m2, identity_u64m1, vector_length_max);

    *max_value = best;
    *max_index = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(reduced_u64m1);
}

NK_INTERNAL void nk_reduce_max_f32_rvv_strided_(                   //
    nk_f32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *max_value, nk_size_t *max_index) {

    nk_size_t vector_length_max = __riscv_vsetvlmax_e32m1();
    vfloat32m1_t best_f32m1 = __riscv_vfmv_v_f_f32m1(NK_F32_MIN, vector_length_max);
    vuint64m2_t best_index_u64m2 = __riscv_vmv_v_x_u64m2(0, vector_length_max);

    unsigned char const *ptr = (unsigned char const *)data;
    nk_size_t offset = 0;
    for (nk_size_t remaining = count, vector_length; remaining > 0;
         remaining -= vector_length, offset += vector_length, ptr += vector_length * stride_bytes) {
        vector_length = __riscv_vsetvl_e32m1(remaining);
        vfloat32m1_t chunk_f32m1 = __riscv_vlse32_v_f32m1((nk_f32_t const *)ptr, (nk_ssize_t)stride_bytes,
                                                          vector_length);

        vbool32_t greater_b32 = __riscv_vmflt_vv_f32m1_b32(best_f32m1, chunk_f32m1, vector_length);
        best_f32m1 = __riscv_vmerge_vvm_f32m1(best_f32m1, chunk_f32m1, greater_b32, vector_length);

        vuint64m2_t position_u64m2 = __riscv_vid_v_u64m2(vector_length);
        position_u64m2 = __riscv_vadd_vx_u64m2(position_u64m2, (nk_u64_t)offset, vector_length);
        best_index_u64m2 = __riscv_vmerge_vvm_u64m2(best_index_u64m2, position_u64m2, greater_b32, vector_length);
    }

    vfloat32m1_t identity_f32m1 = __riscv_vfmv_v_f_f32m1(NK_F32_MIN, 1);
    vfloat32m1_t reduced_f32m1 = __riscv_vfredmax_vs_f32m1_f32m1(best_f32m1, identity_f32m1, vector_length_max);
    nk_f32_t best = __riscv_vfmv_f_s_f32m1_f32(reduced_f32m1);

    vbool32_t match_b32 = __riscv_vmfeq_vf_f32m1_b32(best_f32m1, best, vector_length_max);
    vuint64m2_t sentinel_u64m2 = __riscv_vmv_v_x_u64m2(NK_U64_MAX, vector_length_max);
    vuint64m2_t candidates_u64m2 = __riscv_vmerge_vvm_u64m2(sentinel_u64m2, best_index_u64m2, match_b32,
                                                            vector_length_max);
    vuint64m1_t identity_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    vuint64m1_t reduced_u64m1 = __riscv_vredminu_vs_u64m2_u64m1(candidates_u64m2, identity_u64m1, vector_length_max);

    *max_value = best;
    *max_index = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(reduced_u64m1);
}

NK_PUBLIC void nk_reduce_max_f32_rvv(                              //
    nk_f32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *max_value, nk_size_t *max_index) {
    if (count == 0) *max_value = NK_F32_MIN, *max_index = 0;
    else if (stride_bytes == sizeof(nk_f32_t)) nk_reduce_max_f32_rvv_contiguous_(data, count, max_value, max_index);
    else nk_reduce_max_f32_rvv_strided_(data, count, stride_bytes, max_value, max_index);
}

NK_INTERNAL void nk_reduce_min_f64_rvv_contiguous_( //
    nk_f64_t const *data, nk_size_t count,          //
    nk_f64_t *min_value, nk_size_t *min_index) {

    nk_size_t vector_length_max = __riscv_vsetvlmax_e64m1();
    vfloat64m1_t best_f64m1 = __riscv_vfmv_v_f_f64m1(NK_F64_MAX, vector_length_max);
    vuint64m1_t best_index_u64m1 = __riscv_vmv_v_x_u64m1(0, vector_length_max);

    nk_size_t offset = 0;
    for (nk_size_t remaining = count, vector_length; remaining > 0;
         remaining -= vector_length, offset += vector_length) {
        vector_length = __riscv_vsetvl_e64m1(remaining);
        vfloat64m1_t chunk_f64m1 = __riscv_vle64_v_f64m1(data + offset, vector_length);

        vbool64_t less_b64 = __riscv_vmflt_vv_f64m1_b64(chunk_f64m1, best_f64m1, vector_length);
        best_f64m1 = __riscv_vmerge_vvm_f64m1(best_f64m1, chunk_f64m1, less_b64, vector_length);

        vuint64m1_t position_u64m1 = __riscv_vid_v_u64m1(vector_length);
        position_u64m1 = __riscv_vadd_vx_u64m1(position_u64m1, (nk_u64_t)offset, vector_length);
        best_index_u64m1 = __riscv_vmerge_vvm_u64m1(best_index_u64m1, position_u64m1, less_b64, vector_length);
    }

    vfloat64m1_t identity_f64m1 = __riscv_vfmv_v_f_f64m1(NK_F64_MAX, 1);
    vfloat64m1_t reduced_f64m1 = __riscv_vfredmin_vs_f64m1_f64m1(best_f64m1, identity_f64m1, vector_length_max);
    nk_f64_t best = __riscv_vfmv_f_s_f64m1_f64(reduced_f64m1);

    vbool64_t match_b64 = __riscv_vmfeq_vf_f64m1_b64(best_f64m1, best, vector_length_max);
    vuint64m1_t sentinel_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, vector_length_max);
    vuint64m1_t candidates_u64m1 = __riscv_vmerge_vvm_u64m1(sentinel_u64m1, best_index_u64m1, match_b64,
                                                            vector_length_max);
    vuint64m1_t identity_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    vuint64m1_t reduced_u64m1 = __riscv_vredminu_vs_u64m1_u64m1(candidates_u64m1, identity_u64m1, vector_length_max);

    *min_value = best;
    *min_index = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(reduced_u64m1);
}

NK_INTERNAL void nk_reduce_min_f64_rvv_strided_(                   //
    nk_f64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f64_t *min_value, nk_size_t *min_index) {

    nk_size_t vector_length_max = __riscv_vsetvlmax_e64m1();
    vfloat64m1_t best_f64m1 = __riscv_vfmv_v_f_f64m1(NK_F64_MAX, vector_length_max);
    vuint64m1_t best_index_u64m1 = __riscv_vmv_v_x_u64m1(0, vector_length_max);

    unsigned char const *ptr = (unsigned char const *)data;
    nk_size_t offset = 0;
    for (nk_size_t remaining = count, vector_length; remaining > 0;
         remaining -= vector_length, offset += vector_length, ptr += vector_length * stride_bytes) {
        vector_length = __riscv_vsetvl_e64m1(remaining);
        vfloat64m1_t chunk_f64m1 = __riscv_vlse64_v_f64m1((nk_f64_t const *)ptr, (nk_ssize_t)stride_bytes,
                                                          vector_length);

        vbool64_t less_b64 = __riscv_vmflt_vv_f64m1_b64(chunk_f64m1, best_f64m1, vector_length);
        best_f64m1 = __riscv_vmerge_vvm_f64m1(best_f64m1, chunk_f64m1, less_b64, vector_length);

        vuint64m1_t position_u64m1 = __riscv_vid_v_u64m1(vector_length);
        position_u64m1 = __riscv_vadd_vx_u64m1(position_u64m1, (nk_u64_t)offset, vector_length);
        best_index_u64m1 = __riscv_vmerge_vvm_u64m1(best_index_u64m1, position_u64m1, less_b64, vector_length);
    }

    vfloat64m1_t identity_f64m1 = __riscv_vfmv_v_f_f64m1(NK_F64_MAX, 1);
    vfloat64m1_t reduced_f64m1 = __riscv_vfredmin_vs_f64m1_f64m1(best_f64m1, identity_f64m1, vector_length_max);
    nk_f64_t best = __riscv_vfmv_f_s_f64m1_f64(reduced_f64m1);

    vbool64_t match_b64 = __riscv_vmfeq_vf_f64m1_b64(best_f64m1, best, vector_length_max);
    vuint64m1_t sentinel_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, vector_length_max);
    vuint64m1_t candidates_u64m1 = __riscv_vmerge_vvm_u64m1(sentinel_u64m1, best_index_u64m1, match_b64,
                                                            vector_length_max);
    vuint64m1_t identity_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    vuint64m1_t reduced_u64m1 = __riscv_vredminu_vs_u64m1_u64m1(candidates_u64m1, identity_u64m1, vector_length_max);

    *min_value = best;
    *min_index = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(reduced_u64m1);
}

NK_PUBLIC void nk_reduce_min_f64_rvv(                              //
    nk_f64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f64_t *min_value, nk_size_t *min_index) {
    if (count == 0) *min_value = NK_F64_MAX, *min_index = 0;
    else if (stride_bytes == sizeof(nk_f64_t)) nk_reduce_min_f64_rvv_contiguous_(data, count, min_value, min_index);
    else nk_reduce_min_f64_rvv_strided_(data, count, stride_bytes, min_value, min_index);
}

NK_INTERNAL void nk_reduce_max_f64_rvv_contiguous_( //
    nk_f64_t const *data, nk_size_t count,          //
    nk_f64_t *max_value, nk_size_t *max_index) {

    nk_size_t vector_length_max = __riscv_vsetvlmax_e64m1();
    vfloat64m1_t best_f64m1 = __riscv_vfmv_v_f_f64m1(NK_F64_MIN, vector_length_max);
    vuint64m1_t best_index_u64m1 = __riscv_vmv_v_x_u64m1(0, vector_length_max);

    nk_size_t offset = 0;
    for (nk_size_t remaining = count, vector_length; remaining > 0;
         remaining -= vector_length, offset += vector_length) {
        vector_length = __riscv_vsetvl_e64m1(remaining);
        vfloat64m1_t chunk_f64m1 = __riscv_vle64_v_f64m1(data + offset, vector_length);

        vbool64_t greater_b64 = __riscv_vmflt_vv_f64m1_b64(best_f64m1, chunk_f64m1, vector_length);
        best_f64m1 = __riscv_vmerge_vvm_f64m1(best_f64m1, chunk_f64m1, greater_b64, vector_length);

        vuint64m1_t position_u64m1 = __riscv_vid_v_u64m1(vector_length);
        position_u64m1 = __riscv_vadd_vx_u64m1(position_u64m1, (nk_u64_t)offset, vector_length);
        best_index_u64m1 = __riscv_vmerge_vvm_u64m1(best_index_u64m1, position_u64m1, greater_b64, vector_length);
    }

    vfloat64m1_t identity_f64m1 = __riscv_vfmv_v_f_f64m1(NK_F64_MIN, 1);
    vfloat64m1_t reduced_f64m1 = __riscv_vfredmax_vs_f64m1_f64m1(best_f64m1, identity_f64m1, vector_length_max);
    nk_f64_t best = __riscv_vfmv_f_s_f64m1_f64(reduced_f64m1);

    vbool64_t match_b64 = __riscv_vmfeq_vf_f64m1_b64(best_f64m1, best, vector_length_max);
    vuint64m1_t sentinel_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, vector_length_max);
    vuint64m1_t candidates_u64m1 = __riscv_vmerge_vvm_u64m1(sentinel_u64m1, best_index_u64m1, match_b64,
                                                            vector_length_max);
    vuint64m1_t identity_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    vuint64m1_t reduced_u64m1 = __riscv_vredminu_vs_u64m1_u64m1(candidates_u64m1, identity_u64m1, vector_length_max);

    *max_value = best;
    *max_index = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(reduced_u64m1);
}

NK_INTERNAL void nk_reduce_max_f64_rvv_strided_(                   //
    nk_f64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f64_t *max_value, nk_size_t *max_index) {

    nk_size_t vector_length_max = __riscv_vsetvlmax_e64m1();
    vfloat64m1_t best_f64m1 = __riscv_vfmv_v_f_f64m1(NK_F64_MIN, vector_length_max);
    vuint64m1_t best_index_u64m1 = __riscv_vmv_v_x_u64m1(0, vector_length_max);

    unsigned char const *ptr = (unsigned char const *)data;
    nk_size_t offset = 0;
    for (nk_size_t remaining = count, vector_length; remaining > 0;
         remaining -= vector_length, offset += vector_length, ptr += vector_length * stride_bytes) {
        vector_length = __riscv_vsetvl_e64m1(remaining);
        vfloat64m1_t chunk_f64m1 = __riscv_vlse64_v_f64m1((nk_f64_t const *)ptr, (nk_ssize_t)stride_bytes,
                                                          vector_length);

        vbool64_t greater_b64 = __riscv_vmflt_vv_f64m1_b64(best_f64m1, chunk_f64m1, vector_length);
        best_f64m1 = __riscv_vmerge_vvm_f64m1(best_f64m1, chunk_f64m1, greater_b64, vector_length);

        vuint64m1_t position_u64m1 = __riscv_vid_v_u64m1(vector_length);
        position_u64m1 = __riscv_vadd_vx_u64m1(position_u64m1, (nk_u64_t)offset, vector_length);
        best_index_u64m1 = __riscv_vmerge_vvm_u64m1(best_index_u64m1, position_u64m1, greater_b64, vector_length);
    }

    vfloat64m1_t identity_f64m1 = __riscv_vfmv_v_f_f64m1(NK_F64_MIN, 1);
    vfloat64m1_t reduced_f64m1 = __riscv_vfredmax_vs_f64m1_f64m1(best_f64m1, identity_f64m1, vector_length_max);
    nk_f64_t best = __riscv_vfmv_f_s_f64m1_f64(reduced_f64m1);

    vbool64_t match_b64 = __riscv_vmfeq_vf_f64m1_b64(best_f64m1, best, vector_length_max);
    vuint64m1_t sentinel_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, vector_length_max);
    vuint64m1_t candidates_u64m1 = __riscv_vmerge_vvm_u64m1(sentinel_u64m1, best_index_u64m1, match_b64,
                                                            vector_length_max);
    vuint64m1_t identity_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    vuint64m1_t reduced_u64m1 = __riscv_vredminu_vs_u64m1_u64m1(candidates_u64m1, identity_u64m1, vector_length_max);

    *max_value = best;
    *max_index = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(reduced_u64m1);
}

NK_PUBLIC void nk_reduce_max_f64_rvv(                              //
    nk_f64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f64_t *max_value, nk_size_t *max_index) {
    if (count == 0) *max_value = NK_F64_MIN, *max_index = 0;
    else if (stride_bytes == sizeof(nk_f64_t)) nk_reduce_max_f64_rvv_contiguous_(data, count, max_value, max_index);
    else nk_reduce_max_f64_rvv_strided_(data, count, stride_bytes, max_value, max_index);
}

NK_INTERNAL void nk_reduce_min_i8_rvv_contiguous_( //
    nk_i8_t const *data, nk_size_t count,          //
    nk_i8_t *min_value, nk_size_t *min_index) {

    nk_size_t vector_length_max = __riscv_vsetvlmax_e8m1();
    vint8m1_t best_i8m1 = __riscv_vmv_v_x_i8m1(NK_I8_MAX, vector_length_max);
    vuint64m8_t best_index_u64m8 = __riscv_vmv_v_x_u64m8(0, vector_length_max);

    nk_size_t offset = 0;
    for (nk_size_t remaining = count, vector_length; remaining > 0;
         remaining -= vector_length, offset += vector_length) {
        vector_length = __riscv_vsetvl_e8m1(remaining);
        vint8m1_t chunk_i8m1 = __riscv_vle8_v_i8m1(data + offset, vector_length);

        vbool8_t less_b8 = __riscv_vmslt_vv_i8m1_b8(chunk_i8m1, best_i8m1, vector_length);
        best_i8m1 = __riscv_vmerge_vvm_i8m1(best_i8m1, chunk_i8m1, less_b8, vector_length);

        vuint64m8_t position_u64m8 = __riscv_vid_v_u64m8(vector_length);
        position_u64m8 = __riscv_vadd_vx_u64m8(position_u64m8, (nk_u64_t)offset, vector_length);
        best_index_u64m8 = __riscv_vmerge_vvm_u64m8(best_index_u64m8, position_u64m8, less_b8, vector_length);
    }

    vint8m1_t identity_i8m1 = __riscv_vmv_v_x_i8m1(NK_I8_MAX, 1);
    vint8m1_t reduced_i8m1 = __riscv_vredmin_vs_i8m1_i8m1(best_i8m1, identity_i8m1, vector_length_max);
    nk_i8_t best = (nk_i8_t)__riscv_vmv_x_s_i8m1_i8(reduced_i8m1);

    vbool8_t match_b8 = __riscv_vmseq_vx_i8m1_b8(best_i8m1, best, vector_length_max);
    vuint64m8_t sentinel_u64m8 = __riscv_vmv_v_x_u64m8(NK_U64_MAX, vector_length_max);
    vuint64m8_t candidates_u64m8 = __riscv_vmerge_vvm_u64m8(sentinel_u64m8, best_index_u64m8, match_b8,
                                                            vector_length_max);
    vuint64m1_t identity_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    vuint64m1_t reduced_u64m1 = __riscv_vredminu_vs_u64m8_u64m1(candidates_u64m8, identity_u64m1, vector_length_max);

    *min_value = best;
    *min_index = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(reduced_u64m1);
}

NK_INTERNAL void nk_reduce_min_i8_rvv_strided_(                   //
    nk_i8_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i8_t *min_value, nk_size_t *min_index) {

    nk_size_t vector_length_max = __riscv_vsetvlmax_e8m1();
    vint8m1_t best_i8m1 = __riscv_vmv_v_x_i8m1(NK_I8_MAX, vector_length_max);
    vuint64m8_t best_index_u64m8 = __riscv_vmv_v_x_u64m8(0, vector_length_max);

    unsigned char const *ptr = (unsigned char const *)data;
    nk_size_t offset = 0;
    for (nk_size_t remaining = count, vector_length; remaining > 0;
         remaining -= vector_length, offset += vector_length, ptr += vector_length * stride_bytes) {
        vector_length = __riscv_vsetvl_e8m1(remaining);
        vint8m1_t chunk_i8m1 = __riscv_vlse8_v_i8m1((nk_i8_t const *)ptr, (nk_ssize_t)stride_bytes, vector_length);

        vbool8_t less_b8 = __riscv_vmslt_vv_i8m1_b8(chunk_i8m1, best_i8m1, vector_length);
        best_i8m1 = __riscv_vmerge_vvm_i8m1(best_i8m1, chunk_i8m1, less_b8, vector_length);

        vuint64m8_t position_u64m8 = __riscv_vid_v_u64m8(vector_length);
        position_u64m8 = __riscv_vadd_vx_u64m8(position_u64m8, (nk_u64_t)offset, vector_length);
        best_index_u64m8 = __riscv_vmerge_vvm_u64m8(best_index_u64m8, position_u64m8, less_b8, vector_length);
    }

    vint8m1_t identity_i8m1 = __riscv_vmv_v_x_i8m1(NK_I8_MAX, 1);
    vint8m1_t reduced_i8m1 = __riscv_vredmin_vs_i8m1_i8m1(best_i8m1, identity_i8m1, vector_length_max);
    nk_i8_t best = (nk_i8_t)__riscv_vmv_x_s_i8m1_i8(reduced_i8m1);

    vbool8_t match_b8 = __riscv_vmseq_vx_i8m1_b8(best_i8m1, best, vector_length_max);
    vuint64m8_t sentinel_u64m8 = __riscv_vmv_v_x_u64m8(NK_U64_MAX, vector_length_max);
    vuint64m8_t candidates_u64m8 = __riscv_vmerge_vvm_u64m8(sentinel_u64m8, best_index_u64m8, match_b8,
                                                            vector_length_max);
    vuint64m1_t identity_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    vuint64m1_t reduced_u64m1 = __riscv_vredminu_vs_u64m8_u64m1(candidates_u64m8, identity_u64m1, vector_length_max);

    *min_value = best;
    *min_index = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(reduced_u64m1);
}

NK_PUBLIC void nk_reduce_min_i8_rvv(                              //
    nk_i8_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i8_t *min_value, nk_size_t *min_index) {
    if (count == 0) *min_value = NK_I8_MAX, *min_index = 0;
    else if (stride_bytes == sizeof(nk_i8_t)) nk_reduce_min_i8_rvv_contiguous_(data, count, min_value, min_index);
    else nk_reduce_min_i8_rvv_strided_(data, count, stride_bytes, min_value, min_index);
}

NK_INTERNAL void nk_reduce_max_i8_rvv_contiguous_( //
    nk_i8_t const *data, nk_size_t count,          //
    nk_i8_t *max_value, nk_size_t *max_index) {

    nk_size_t vector_length_max = __riscv_vsetvlmax_e8m1();
    vint8m1_t best_i8m1 = __riscv_vmv_v_x_i8m1(NK_I8_MIN, vector_length_max);
    vuint64m8_t best_index_u64m8 = __riscv_vmv_v_x_u64m8(0, vector_length_max);

    nk_size_t offset = 0;
    for (nk_size_t remaining = count, vector_length; remaining > 0;
         remaining -= vector_length, offset += vector_length) {
        vector_length = __riscv_vsetvl_e8m1(remaining);
        vint8m1_t chunk_i8m1 = __riscv_vle8_v_i8m1(data + offset, vector_length);

        vbool8_t greater_b8 = __riscv_vmslt_vv_i8m1_b8(best_i8m1, chunk_i8m1, vector_length);
        best_i8m1 = __riscv_vmerge_vvm_i8m1(best_i8m1, chunk_i8m1, greater_b8, vector_length);

        vuint64m8_t position_u64m8 = __riscv_vid_v_u64m8(vector_length);
        position_u64m8 = __riscv_vadd_vx_u64m8(position_u64m8, (nk_u64_t)offset, vector_length);
        best_index_u64m8 = __riscv_vmerge_vvm_u64m8(best_index_u64m8, position_u64m8, greater_b8, vector_length);
    }

    vint8m1_t identity_i8m1 = __riscv_vmv_v_x_i8m1(NK_I8_MIN, 1);
    vint8m1_t reduced_i8m1 = __riscv_vredmax_vs_i8m1_i8m1(best_i8m1, identity_i8m1, vector_length_max);
    nk_i8_t best = (nk_i8_t)__riscv_vmv_x_s_i8m1_i8(reduced_i8m1);

    vbool8_t match_b8 = __riscv_vmseq_vx_i8m1_b8(best_i8m1, best, vector_length_max);
    vuint64m8_t sentinel_u64m8 = __riscv_vmv_v_x_u64m8(NK_U64_MAX, vector_length_max);
    vuint64m8_t candidates_u64m8 = __riscv_vmerge_vvm_u64m8(sentinel_u64m8, best_index_u64m8, match_b8,
                                                            vector_length_max);
    vuint64m1_t identity_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    vuint64m1_t reduced_u64m1 = __riscv_vredminu_vs_u64m8_u64m1(candidates_u64m8, identity_u64m1, vector_length_max);

    *max_value = best;
    *max_index = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(reduced_u64m1);
}

NK_INTERNAL void nk_reduce_max_i8_rvv_strided_(                   //
    nk_i8_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i8_t *max_value, nk_size_t *max_index) {

    nk_size_t vector_length_max = __riscv_vsetvlmax_e8m1();
    vint8m1_t best_i8m1 = __riscv_vmv_v_x_i8m1(NK_I8_MIN, vector_length_max);
    vuint64m8_t best_index_u64m8 = __riscv_vmv_v_x_u64m8(0, vector_length_max);

    unsigned char const *ptr = (unsigned char const *)data;
    nk_size_t offset = 0;
    for (nk_size_t remaining = count, vector_length; remaining > 0;
         remaining -= vector_length, offset += vector_length, ptr += vector_length * stride_bytes) {
        vector_length = __riscv_vsetvl_e8m1(remaining);
        vint8m1_t chunk_i8m1 = __riscv_vlse8_v_i8m1((nk_i8_t const *)ptr, (nk_ssize_t)stride_bytes, vector_length);

        vbool8_t greater_b8 = __riscv_vmslt_vv_i8m1_b8(best_i8m1, chunk_i8m1, vector_length);
        best_i8m1 = __riscv_vmerge_vvm_i8m1(best_i8m1, chunk_i8m1, greater_b8, vector_length);

        vuint64m8_t position_u64m8 = __riscv_vid_v_u64m8(vector_length);
        position_u64m8 = __riscv_vadd_vx_u64m8(position_u64m8, (nk_u64_t)offset, vector_length);
        best_index_u64m8 = __riscv_vmerge_vvm_u64m8(best_index_u64m8, position_u64m8, greater_b8, vector_length);
    }

    vint8m1_t identity_i8m1 = __riscv_vmv_v_x_i8m1(NK_I8_MIN, 1);
    vint8m1_t reduced_i8m1 = __riscv_vredmax_vs_i8m1_i8m1(best_i8m1, identity_i8m1, vector_length_max);
    nk_i8_t best = (nk_i8_t)__riscv_vmv_x_s_i8m1_i8(reduced_i8m1);

    vbool8_t match_b8 = __riscv_vmseq_vx_i8m1_b8(best_i8m1, best, vector_length_max);
    vuint64m8_t sentinel_u64m8 = __riscv_vmv_v_x_u64m8(NK_U64_MAX, vector_length_max);
    vuint64m8_t candidates_u64m8 = __riscv_vmerge_vvm_u64m8(sentinel_u64m8, best_index_u64m8, match_b8,
                                                            vector_length_max);
    vuint64m1_t identity_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    vuint64m1_t reduced_u64m1 = __riscv_vredminu_vs_u64m8_u64m1(candidates_u64m8, identity_u64m1, vector_length_max);

    *max_value = best;
    *max_index = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(reduced_u64m1);
}

NK_PUBLIC void nk_reduce_max_i8_rvv(                              //
    nk_i8_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i8_t *max_value, nk_size_t *max_index) {
    if (count == 0) *max_value = NK_I8_MIN, *max_index = 0;
    else if (stride_bytes == sizeof(nk_i8_t)) nk_reduce_max_i8_rvv_contiguous_(data, count, max_value, max_index);
    else nk_reduce_max_i8_rvv_strided_(data, count, stride_bytes, max_value, max_index);
}

NK_INTERNAL void nk_reduce_min_u8_rvv_contiguous_( //
    nk_u8_t const *data, nk_size_t count,          //
    nk_u8_t *min_value, nk_size_t *min_index) {

    nk_size_t vector_length_max = __riscv_vsetvlmax_e8m1();
    vuint8m1_t best_u8m1 = __riscv_vmv_v_x_u8m1(NK_U8_MAX, vector_length_max);
    vuint64m8_t best_index_u64m8 = __riscv_vmv_v_x_u64m8(0, vector_length_max);

    nk_size_t offset = 0;
    for (nk_size_t remaining = count, vector_length; remaining > 0;
         remaining -= vector_length, offset += vector_length) {
        vector_length = __riscv_vsetvl_e8m1(remaining);
        vuint8m1_t chunk_u8m1 = __riscv_vle8_v_u8m1(data + offset, vector_length);

        vbool8_t less_b8 = __riscv_vmsltu_vv_u8m1_b8(chunk_u8m1, best_u8m1, vector_length);
        best_u8m1 = __riscv_vmerge_vvm_u8m1(best_u8m1, chunk_u8m1, less_b8, vector_length);

        vuint64m8_t position_u64m8 = __riscv_vid_v_u64m8(vector_length);
        position_u64m8 = __riscv_vadd_vx_u64m8(position_u64m8, (nk_u64_t)offset, vector_length);
        best_index_u64m8 = __riscv_vmerge_vvm_u64m8(best_index_u64m8, position_u64m8, less_b8, vector_length);
    }

    vuint8m1_t identity_u8m1 = __riscv_vmv_v_x_u8m1(NK_U8_MAX, 1);
    vuint8m1_t reduced_u8m1 = __riscv_vredminu_vs_u8m1_u8m1(best_u8m1, identity_u8m1, vector_length_max);
    nk_u8_t best = (nk_u8_t)__riscv_vmv_x_s_u8m1_u8(reduced_u8m1);

    vbool8_t match_b8 = __riscv_vmseq_vx_u8m1_b8(best_u8m1, best, vector_length_max);
    vuint64m8_t sentinel_u64m8 = __riscv_vmv_v_x_u64m8(NK_U64_MAX, vector_length_max);
    vuint64m8_t candidates_u64m8 = __riscv_vmerge_vvm_u64m8(sentinel_u64m8, best_index_u64m8, match_b8,
                                                            vector_length_max);
    vuint64m1_t identity_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    vuint64m1_t reduced_u64m1 = __riscv_vredminu_vs_u64m8_u64m1(candidates_u64m8, identity_u64m1, vector_length_max);

    *min_value = best;
    *min_index = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(reduced_u64m1);
}

NK_INTERNAL void nk_reduce_min_u8_rvv_strided_(                   //
    nk_u8_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u8_t *min_value, nk_size_t *min_index) {

    nk_size_t vector_length_max = __riscv_vsetvlmax_e8m1();
    vuint8m1_t best_u8m1 = __riscv_vmv_v_x_u8m1(NK_U8_MAX, vector_length_max);
    vuint64m8_t best_index_u64m8 = __riscv_vmv_v_x_u64m8(0, vector_length_max);

    unsigned char const *ptr = (unsigned char const *)data;
    nk_size_t offset = 0;
    for (nk_size_t remaining = count, vector_length; remaining > 0;
         remaining -= vector_length, offset += vector_length, ptr += vector_length * stride_bytes) {
        vector_length = __riscv_vsetvl_e8m1(remaining);
        vuint8m1_t chunk_u8m1 = __riscv_vlse8_v_u8m1((nk_u8_t const *)ptr, (nk_ssize_t)stride_bytes, vector_length);

        vbool8_t less_b8 = __riscv_vmsltu_vv_u8m1_b8(chunk_u8m1, best_u8m1, vector_length);
        best_u8m1 = __riscv_vmerge_vvm_u8m1(best_u8m1, chunk_u8m1, less_b8, vector_length);

        vuint64m8_t position_u64m8 = __riscv_vid_v_u64m8(vector_length);
        position_u64m8 = __riscv_vadd_vx_u64m8(position_u64m8, (nk_u64_t)offset, vector_length);
        best_index_u64m8 = __riscv_vmerge_vvm_u64m8(best_index_u64m8, position_u64m8, less_b8, vector_length);
    }

    vuint8m1_t identity_u8m1 = __riscv_vmv_v_x_u8m1(NK_U8_MAX, 1);
    vuint8m1_t reduced_u8m1 = __riscv_vredminu_vs_u8m1_u8m1(best_u8m1, identity_u8m1, vector_length_max);
    nk_u8_t best = (nk_u8_t)__riscv_vmv_x_s_u8m1_u8(reduced_u8m1);

    vbool8_t match_b8 = __riscv_vmseq_vx_u8m1_b8(best_u8m1, best, vector_length_max);
    vuint64m8_t sentinel_u64m8 = __riscv_vmv_v_x_u64m8(NK_U64_MAX, vector_length_max);
    vuint64m8_t candidates_u64m8 = __riscv_vmerge_vvm_u64m8(sentinel_u64m8, best_index_u64m8, match_b8,
                                                            vector_length_max);
    vuint64m1_t identity_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    vuint64m1_t reduced_u64m1 = __riscv_vredminu_vs_u64m8_u64m1(candidates_u64m8, identity_u64m1, vector_length_max);

    *min_value = best;
    *min_index = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(reduced_u64m1);
}

NK_PUBLIC void nk_reduce_min_u8_rvv(                              //
    nk_u8_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u8_t *min_value, nk_size_t *min_index) {
    if (count == 0) *min_value = NK_U8_MAX, *min_index = 0;
    else if (stride_bytes == sizeof(nk_u8_t)) nk_reduce_min_u8_rvv_contiguous_(data, count, min_value, min_index);
    else nk_reduce_min_u8_rvv_strided_(data, count, stride_bytes, min_value, min_index);
}

NK_INTERNAL void nk_reduce_max_u8_rvv_contiguous_( //
    nk_u8_t const *data, nk_size_t count,          //
    nk_u8_t *max_value, nk_size_t *max_index) {

    nk_size_t vector_length_max = __riscv_vsetvlmax_e8m1();
    vuint8m1_t best_u8m1 = __riscv_vmv_v_x_u8m1(0, vector_length_max);
    vuint64m8_t best_index_u64m8 = __riscv_vmv_v_x_u64m8(0, vector_length_max);

    nk_size_t offset = 0;
    for (nk_size_t remaining = count, vector_length; remaining > 0;
         remaining -= vector_length, offset += vector_length) {
        vector_length = __riscv_vsetvl_e8m1(remaining);
        vuint8m1_t chunk_u8m1 = __riscv_vle8_v_u8m1(data + offset, vector_length);

        vbool8_t greater_b8 = __riscv_vmsltu_vv_u8m1_b8(best_u8m1, chunk_u8m1, vector_length);
        best_u8m1 = __riscv_vmerge_vvm_u8m1(best_u8m1, chunk_u8m1, greater_b8, vector_length);

        vuint64m8_t position_u64m8 = __riscv_vid_v_u64m8(vector_length);
        position_u64m8 = __riscv_vadd_vx_u64m8(position_u64m8, (nk_u64_t)offset, vector_length);
        best_index_u64m8 = __riscv_vmerge_vvm_u64m8(best_index_u64m8, position_u64m8, greater_b8, vector_length);
    }

    vuint8m1_t identity_u8m1 = __riscv_vmv_v_x_u8m1(0, 1);
    vuint8m1_t reduced_u8m1 = __riscv_vredmaxu_vs_u8m1_u8m1(best_u8m1, identity_u8m1, vector_length_max);
    nk_u8_t best = (nk_u8_t)__riscv_vmv_x_s_u8m1_u8(reduced_u8m1);

    vbool8_t match_b8 = __riscv_vmseq_vx_u8m1_b8(best_u8m1, best, vector_length_max);
    vuint64m8_t sentinel_u64m8 = __riscv_vmv_v_x_u64m8(NK_U64_MAX, vector_length_max);
    vuint64m8_t candidates_u64m8 = __riscv_vmerge_vvm_u64m8(sentinel_u64m8, best_index_u64m8, match_b8,
                                                            vector_length_max);
    vuint64m1_t identity_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    vuint64m1_t reduced_u64m1 = __riscv_vredminu_vs_u64m8_u64m1(candidates_u64m8, identity_u64m1, vector_length_max);

    *max_value = best;
    *max_index = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(reduced_u64m1);
}

NK_INTERNAL void nk_reduce_max_u8_rvv_strided_(                   //
    nk_u8_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u8_t *max_value, nk_size_t *max_index) {

    nk_size_t vector_length_max = __riscv_vsetvlmax_e8m1();
    vuint8m1_t best_u8m1 = __riscv_vmv_v_x_u8m1(0, vector_length_max);
    vuint64m8_t best_index_u64m8 = __riscv_vmv_v_x_u64m8(0, vector_length_max);

    unsigned char const *ptr = (unsigned char const *)data;
    nk_size_t offset = 0;
    for (nk_size_t remaining = count, vector_length; remaining > 0;
         remaining -= vector_length, offset += vector_length, ptr += vector_length * stride_bytes) {
        vector_length = __riscv_vsetvl_e8m1(remaining);
        vuint8m1_t chunk_u8m1 = __riscv_vlse8_v_u8m1((nk_u8_t const *)ptr, (nk_ssize_t)stride_bytes, vector_length);

        vbool8_t greater_b8 = __riscv_vmsltu_vv_u8m1_b8(best_u8m1, chunk_u8m1, vector_length);
        best_u8m1 = __riscv_vmerge_vvm_u8m1(best_u8m1, chunk_u8m1, greater_b8, vector_length);

        vuint64m8_t position_u64m8 = __riscv_vid_v_u64m8(vector_length);
        position_u64m8 = __riscv_vadd_vx_u64m8(position_u64m8, (nk_u64_t)offset, vector_length);
        best_index_u64m8 = __riscv_vmerge_vvm_u64m8(best_index_u64m8, position_u64m8, greater_b8, vector_length);
    }

    vuint8m1_t identity_u8m1 = __riscv_vmv_v_x_u8m1(0, 1);
    vuint8m1_t reduced_u8m1 = __riscv_vredmaxu_vs_u8m1_u8m1(best_u8m1, identity_u8m1, vector_length_max);
    nk_u8_t best = (nk_u8_t)__riscv_vmv_x_s_u8m1_u8(reduced_u8m1);

    vbool8_t match_b8 = __riscv_vmseq_vx_u8m1_b8(best_u8m1, best, vector_length_max);
    vuint64m8_t sentinel_u64m8 = __riscv_vmv_v_x_u64m8(NK_U64_MAX, vector_length_max);
    vuint64m8_t candidates_u64m8 = __riscv_vmerge_vvm_u64m8(sentinel_u64m8, best_index_u64m8, match_b8,
                                                            vector_length_max);
    vuint64m1_t identity_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    vuint64m1_t reduced_u64m1 = __riscv_vredminu_vs_u64m8_u64m1(candidates_u64m8, identity_u64m1, vector_length_max);

    *max_value = best;
    *max_index = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(reduced_u64m1);
}

NK_PUBLIC void nk_reduce_max_u8_rvv(                              //
    nk_u8_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u8_t *max_value, nk_size_t *max_index) {
    if (count == 0) *max_value = 0, *max_index = 0;
    else if (stride_bytes == sizeof(nk_u8_t)) nk_reduce_max_u8_rvv_contiguous_(data, count, max_value, max_index);
    else nk_reduce_max_u8_rvv_strided_(data, count, stride_bytes, max_value, max_index);
}

NK_INTERNAL void nk_reduce_min_i16_rvv_contiguous_( //
    nk_i16_t const *data, nk_size_t count,          //
    nk_i16_t *min_value, nk_size_t *min_index) {

    nk_size_t vector_length_max = __riscv_vsetvlmax_e16m1();
    vint16m1_t best_i16m1 = __riscv_vmv_v_x_i16m1(NK_I16_MAX, vector_length_max);
    vuint64m4_t best_index_u64m4 = __riscv_vmv_v_x_u64m4(0, vector_length_max);

    nk_size_t offset = 0;
    for (nk_size_t remaining = count, vector_length; remaining > 0;
         remaining -= vector_length, offset += vector_length) {
        vector_length = __riscv_vsetvl_e16m1(remaining);
        vint16m1_t chunk_i16m1 = __riscv_vle16_v_i16m1(data + offset, vector_length);

        vbool16_t less_b16 = __riscv_vmslt_vv_i16m1_b16(chunk_i16m1, best_i16m1, vector_length);
        best_i16m1 = __riscv_vmerge_vvm_i16m1(best_i16m1, chunk_i16m1, less_b16, vector_length);

        vuint64m4_t position_u64m4 = __riscv_vid_v_u64m4(vector_length);
        position_u64m4 = __riscv_vadd_vx_u64m4(position_u64m4, (nk_u64_t)offset, vector_length);
        best_index_u64m4 = __riscv_vmerge_vvm_u64m4(best_index_u64m4, position_u64m4, less_b16, vector_length);
    }

    vint16m1_t identity_i16m1 = __riscv_vmv_v_x_i16m1(NK_I16_MAX, 1);
    vint16m1_t reduced_i16m1 = __riscv_vredmin_vs_i16m1_i16m1(best_i16m1, identity_i16m1, vector_length_max);
    nk_i16_t best = (nk_i16_t)__riscv_vmv_x_s_i16m1_i16(reduced_i16m1);

    vbool16_t match_b16 = __riscv_vmseq_vx_i16m1_b16(best_i16m1, best, vector_length_max);
    vuint64m4_t sentinel_u64m4 = __riscv_vmv_v_x_u64m4(NK_U64_MAX, vector_length_max);
    vuint64m4_t candidates_u64m4 = __riscv_vmerge_vvm_u64m4(sentinel_u64m4, best_index_u64m4, match_b16,
                                                            vector_length_max);
    vuint64m1_t identity_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    vuint64m1_t reduced_u64m1 = __riscv_vredminu_vs_u64m4_u64m1(candidates_u64m4, identity_u64m1, vector_length_max);

    *min_value = best;
    *min_index = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(reduced_u64m1);
}

NK_INTERNAL void nk_reduce_min_i16_rvv_strided_(                   //
    nk_i16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i16_t *min_value, nk_size_t *min_index) {

    nk_size_t vector_length_max = __riscv_vsetvlmax_e16m1();
    vint16m1_t best_i16m1 = __riscv_vmv_v_x_i16m1(NK_I16_MAX, vector_length_max);
    vuint64m4_t best_index_u64m4 = __riscv_vmv_v_x_u64m4(0, vector_length_max);

    unsigned char const *ptr = (unsigned char const *)data;
    nk_size_t offset = 0;
    for (nk_size_t remaining = count, vector_length; remaining > 0;
         remaining -= vector_length, offset += vector_length, ptr += vector_length * stride_bytes) {
        vector_length = __riscv_vsetvl_e16m1(remaining);
        vint16m1_t chunk_i16m1 = __riscv_vlse16_v_i16m1((nk_i16_t const *)ptr, (nk_ssize_t)stride_bytes, vector_length);

        vbool16_t less_b16 = __riscv_vmslt_vv_i16m1_b16(chunk_i16m1, best_i16m1, vector_length);
        best_i16m1 = __riscv_vmerge_vvm_i16m1(best_i16m1, chunk_i16m1, less_b16, vector_length);

        vuint64m4_t position_u64m4 = __riscv_vid_v_u64m4(vector_length);
        position_u64m4 = __riscv_vadd_vx_u64m4(position_u64m4, (nk_u64_t)offset, vector_length);
        best_index_u64m4 = __riscv_vmerge_vvm_u64m4(best_index_u64m4, position_u64m4, less_b16, vector_length);
    }

    vint16m1_t identity_i16m1 = __riscv_vmv_v_x_i16m1(NK_I16_MAX, 1);
    vint16m1_t reduced_i16m1 = __riscv_vredmin_vs_i16m1_i16m1(best_i16m1, identity_i16m1, vector_length_max);
    nk_i16_t best = (nk_i16_t)__riscv_vmv_x_s_i16m1_i16(reduced_i16m1);

    vbool16_t match_b16 = __riscv_vmseq_vx_i16m1_b16(best_i16m1, best, vector_length_max);
    vuint64m4_t sentinel_u64m4 = __riscv_vmv_v_x_u64m4(NK_U64_MAX, vector_length_max);
    vuint64m4_t candidates_u64m4 = __riscv_vmerge_vvm_u64m4(sentinel_u64m4, best_index_u64m4, match_b16,
                                                            vector_length_max);
    vuint64m1_t identity_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    vuint64m1_t reduced_u64m1 = __riscv_vredminu_vs_u64m4_u64m1(candidates_u64m4, identity_u64m1, vector_length_max);

    *min_value = best;
    *min_index = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(reduced_u64m1);
}

NK_PUBLIC void nk_reduce_min_i16_rvv(                              //
    nk_i16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i16_t *min_value, nk_size_t *min_index) {
    if (count == 0) *min_value = NK_I16_MAX, *min_index = 0;
    else if (stride_bytes == sizeof(nk_i16_t)) nk_reduce_min_i16_rvv_contiguous_(data, count, min_value, min_index);
    else nk_reduce_min_i16_rvv_strided_(data, count, stride_bytes, min_value, min_index);
}

NK_INTERNAL void nk_reduce_max_i16_rvv_contiguous_( //
    nk_i16_t const *data, nk_size_t count,          //
    nk_i16_t *max_value, nk_size_t *max_index) {

    nk_size_t vector_length_max = __riscv_vsetvlmax_e16m1();
    vint16m1_t best_i16m1 = __riscv_vmv_v_x_i16m1(NK_I16_MIN, vector_length_max);
    vuint64m4_t best_index_u64m4 = __riscv_vmv_v_x_u64m4(0, vector_length_max);

    nk_size_t offset = 0;
    for (nk_size_t remaining = count, vector_length; remaining > 0;
         remaining -= vector_length, offset += vector_length) {
        vector_length = __riscv_vsetvl_e16m1(remaining);
        vint16m1_t chunk_i16m1 = __riscv_vle16_v_i16m1(data + offset, vector_length);

        vbool16_t greater_b16 = __riscv_vmslt_vv_i16m1_b16(best_i16m1, chunk_i16m1, vector_length);
        best_i16m1 = __riscv_vmerge_vvm_i16m1(best_i16m1, chunk_i16m1, greater_b16, vector_length);

        vuint64m4_t position_u64m4 = __riscv_vid_v_u64m4(vector_length);
        position_u64m4 = __riscv_vadd_vx_u64m4(position_u64m4, (nk_u64_t)offset, vector_length);
        best_index_u64m4 = __riscv_vmerge_vvm_u64m4(best_index_u64m4, position_u64m4, greater_b16, vector_length);
    }

    vint16m1_t identity_i16m1 = __riscv_vmv_v_x_i16m1(NK_I16_MIN, 1);
    vint16m1_t reduced_i16m1 = __riscv_vredmax_vs_i16m1_i16m1(best_i16m1, identity_i16m1, vector_length_max);
    nk_i16_t best = (nk_i16_t)__riscv_vmv_x_s_i16m1_i16(reduced_i16m1);

    vbool16_t match_b16 = __riscv_vmseq_vx_i16m1_b16(best_i16m1, best, vector_length_max);
    vuint64m4_t sentinel_u64m4 = __riscv_vmv_v_x_u64m4(NK_U64_MAX, vector_length_max);
    vuint64m4_t candidates_u64m4 = __riscv_vmerge_vvm_u64m4(sentinel_u64m4, best_index_u64m4, match_b16,
                                                            vector_length_max);
    vuint64m1_t identity_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    vuint64m1_t reduced_u64m1 = __riscv_vredminu_vs_u64m4_u64m1(candidates_u64m4, identity_u64m1, vector_length_max);

    *max_value = best;
    *max_index = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(reduced_u64m1);
}

NK_INTERNAL void nk_reduce_max_i16_rvv_strided_(                   //
    nk_i16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i16_t *max_value, nk_size_t *max_index) {

    nk_size_t vector_length_max = __riscv_vsetvlmax_e16m1();
    vint16m1_t best_i16m1 = __riscv_vmv_v_x_i16m1(NK_I16_MIN, vector_length_max);
    vuint64m4_t best_index_u64m4 = __riscv_vmv_v_x_u64m4(0, vector_length_max);

    unsigned char const *ptr = (unsigned char const *)data;
    nk_size_t offset = 0;
    for (nk_size_t remaining = count, vector_length; remaining > 0;
         remaining -= vector_length, offset += vector_length, ptr += vector_length * stride_bytes) {
        vector_length = __riscv_vsetvl_e16m1(remaining);
        vint16m1_t chunk_i16m1 = __riscv_vlse16_v_i16m1((nk_i16_t const *)ptr, (nk_ssize_t)stride_bytes, vector_length);

        vbool16_t greater_b16 = __riscv_vmslt_vv_i16m1_b16(best_i16m1, chunk_i16m1, vector_length);
        best_i16m1 = __riscv_vmerge_vvm_i16m1(best_i16m1, chunk_i16m1, greater_b16, vector_length);

        vuint64m4_t position_u64m4 = __riscv_vid_v_u64m4(vector_length);
        position_u64m4 = __riscv_vadd_vx_u64m4(position_u64m4, (nk_u64_t)offset, vector_length);
        best_index_u64m4 = __riscv_vmerge_vvm_u64m4(best_index_u64m4, position_u64m4, greater_b16, vector_length);
    }

    vint16m1_t identity_i16m1 = __riscv_vmv_v_x_i16m1(NK_I16_MIN, 1);
    vint16m1_t reduced_i16m1 = __riscv_vredmax_vs_i16m1_i16m1(best_i16m1, identity_i16m1, vector_length_max);
    nk_i16_t best = (nk_i16_t)__riscv_vmv_x_s_i16m1_i16(reduced_i16m1);

    vbool16_t match_b16 = __riscv_vmseq_vx_i16m1_b16(best_i16m1, best, vector_length_max);
    vuint64m4_t sentinel_u64m4 = __riscv_vmv_v_x_u64m4(NK_U64_MAX, vector_length_max);
    vuint64m4_t candidates_u64m4 = __riscv_vmerge_vvm_u64m4(sentinel_u64m4, best_index_u64m4, match_b16,
                                                            vector_length_max);
    vuint64m1_t identity_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    vuint64m1_t reduced_u64m1 = __riscv_vredminu_vs_u64m4_u64m1(candidates_u64m4, identity_u64m1, vector_length_max);

    *max_value = best;
    *max_index = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(reduced_u64m1);
}

NK_PUBLIC void nk_reduce_max_i16_rvv(                              //
    nk_i16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i16_t *max_value, nk_size_t *max_index) {
    if (count == 0) *max_value = NK_I16_MIN, *max_index = 0;
    else if (stride_bytes == sizeof(nk_i16_t)) nk_reduce_max_i16_rvv_contiguous_(data, count, max_value, max_index);
    else nk_reduce_max_i16_rvv_strided_(data, count, stride_bytes, max_value, max_index);
}

NK_INTERNAL void nk_reduce_min_u16_rvv_contiguous_( //
    nk_u16_t const *data, nk_size_t count,          //
    nk_u16_t *min_value, nk_size_t *min_index) {

    nk_size_t vector_length_max = __riscv_vsetvlmax_e16m1();
    vuint16m1_t best_u16m1 = __riscv_vmv_v_x_u16m1(NK_U16_MAX, vector_length_max);
    vuint64m4_t best_index_u64m4 = __riscv_vmv_v_x_u64m4(0, vector_length_max);

    nk_size_t offset = 0;
    for (nk_size_t remaining = count, vector_length; remaining > 0;
         remaining -= vector_length, offset += vector_length) {
        vector_length = __riscv_vsetvl_e16m1(remaining);
        vuint16m1_t chunk_u16m1 = __riscv_vle16_v_u16m1(data + offset, vector_length);

        vbool16_t less_b16 = __riscv_vmsltu_vv_u16m1_b16(chunk_u16m1, best_u16m1, vector_length);
        best_u16m1 = __riscv_vmerge_vvm_u16m1(best_u16m1, chunk_u16m1, less_b16, vector_length);

        vuint64m4_t position_u64m4 = __riscv_vid_v_u64m4(vector_length);
        position_u64m4 = __riscv_vadd_vx_u64m4(position_u64m4, (nk_u64_t)offset, vector_length);
        best_index_u64m4 = __riscv_vmerge_vvm_u64m4(best_index_u64m4, position_u64m4, less_b16, vector_length);
    }

    vuint16m1_t identity_u16m1 = __riscv_vmv_v_x_u16m1(NK_U16_MAX, 1);
    vuint16m1_t reduced_u16m1 = __riscv_vredminu_vs_u16m1_u16m1(best_u16m1, identity_u16m1, vector_length_max);
    nk_u16_t best = (nk_u16_t)__riscv_vmv_x_s_u16m1_u16(reduced_u16m1);

    vbool16_t match_b16 = __riscv_vmseq_vx_u16m1_b16(best_u16m1, best, vector_length_max);
    vuint64m4_t sentinel_u64m4 = __riscv_vmv_v_x_u64m4(NK_U64_MAX, vector_length_max);
    vuint64m4_t candidates_u64m4 = __riscv_vmerge_vvm_u64m4(sentinel_u64m4, best_index_u64m4, match_b16,
                                                            vector_length_max);
    vuint64m1_t identity_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    vuint64m1_t reduced_u64m1 = __riscv_vredminu_vs_u64m4_u64m1(candidates_u64m4, identity_u64m1, vector_length_max);

    *min_value = best;
    *min_index = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(reduced_u64m1);
}

NK_INTERNAL void nk_reduce_min_u16_rvv_strided_(                   //
    nk_u16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u16_t *min_value, nk_size_t *min_index) {

    nk_size_t vector_length_max = __riscv_vsetvlmax_e16m1();
    vuint16m1_t best_u16m1 = __riscv_vmv_v_x_u16m1(NK_U16_MAX, vector_length_max);
    vuint64m4_t best_index_u64m4 = __riscv_vmv_v_x_u64m4(0, vector_length_max);

    unsigned char const *ptr = (unsigned char const *)data;
    nk_size_t offset = 0;
    for (nk_size_t remaining = count, vector_length; remaining > 0;
         remaining -= vector_length, offset += vector_length, ptr += vector_length * stride_bytes) {
        vector_length = __riscv_vsetvl_e16m1(remaining);
        vuint16m1_t chunk_u16m1 = __riscv_vlse16_v_u16m1((nk_u16_t const *)ptr, (nk_ssize_t)stride_bytes,
                                                         vector_length);

        vbool16_t less_b16 = __riscv_vmsltu_vv_u16m1_b16(chunk_u16m1, best_u16m1, vector_length);
        best_u16m1 = __riscv_vmerge_vvm_u16m1(best_u16m1, chunk_u16m1, less_b16, vector_length);

        vuint64m4_t position_u64m4 = __riscv_vid_v_u64m4(vector_length);
        position_u64m4 = __riscv_vadd_vx_u64m4(position_u64m4, (nk_u64_t)offset, vector_length);
        best_index_u64m4 = __riscv_vmerge_vvm_u64m4(best_index_u64m4, position_u64m4, less_b16, vector_length);
    }

    vuint16m1_t identity_u16m1 = __riscv_vmv_v_x_u16m1(NK_U16_MAX, 1);
    vuint16m1_t reduced_u16m1 = __riscv_vredminu_vs_u16m1_u16m1(best_u16m1, identity_u16m1, vector_length_max);
    nk_u16_t best = (nk_u16_t)__riscv_vmv_x_s_u16m1_u16(reduced_u16m1);

    vbool16_t match_b16 = __riscv_vmseq_vx_u16m1_b16(best_u16m1, best, vector_length_max);
    vuint64m4_t sentinel_u64m4 = __riscv_vmv_v_x_u64m4(NK_U64_MAX, vector_length_max);
    vuint64m4_t candidates_u64m4 = __riscv_vmerge_vvm_u64m4(sentinel_u64m4, best_index_u64m4, match_b16,
                                                            vector_length_max);
    vuint64m1_t identity_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    vuint64m1_t reduced_u64m1 = __riscv_vredminu_vs_u64m4_u64m1(candidates_u64m4, identity_u64m1, vector_length_max);

    *min_value = best;
    *min_index = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(reduced_u64m1);
}

NK_PUBLIC void nk_reduce_min_u16_rvv(                              //
    nk_u16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u16_t *min_value, nk_size_t *min_index) {
    if (count == 0) *min_value = NK_U16_MAX, *min_index = 0;
    else if (stride_bytes == sizeof(nk_u16_t)) nk_reduce_min_u16_rvv_contiguous_(data, count, min_value, min_index);
    else nk_reduce_min_u16_rvv_strided_(data, count, stride_bytes, min_value, min_index);
}

NK_INTERNAL void nk_reduce_max_u16_rvv_contiguous_( //
    nk_u16_t const *data, nk_size_t count,          //
    nk_u16_t *max_value, nk_size_t *max_index) {

    nk_size_t vector_length_max = __riscv_vsetvlmax_e16m1();
    vuint16m1_t best_u16m1 = __riscv_vmv_v_x_u16m1(0, vector_length_max);
    vuint64m4_t best_index_u64m4 = __riscv_vmv_v_x_u64m4(0, vector_length_max);

    nk_size_t offset = 0;
    for (nk_size_t remaining = count, vector_length; remaining > 0;
         remaining -= vector_length, offset += vector_length) {
        vector_length = __riscv_vsetvl_e16m1(remaining);
        vuint16m1_t chunk_u16m1 = __riscv_vle16_v_u16m1(data + offset, vector_length);

        vbool16_t greater_b16 = __riscv_vmsltu_vv_u16m1_b16(best_u16m1, chunk_u16m1, vector_length);
        best_u16m1 = __riscv_vmerge_vvm_u16m1(best_u16m1, chunk_u16m1, greater_b16, vector_length);

        vuint64m4_t position_u64m4 = __riscv_vid_v_u64m4(vector_length);
        position_u64m4 = __riscv_vadd_vx_u64m4(position_u64m4, (nk_u64_t)offset, vector_length);
        best_index_u64m4 = __riscv_vmerge_vvm_u64m4(best_index_u64m4, position_u64m4, greater_b16, vector_length);
    }

    vuint16m1_t identity_u16m1 = __riscv_vmv_v_x_u16m1(0, 1);
    vuint16m1_t reduced_u16m1 = __riscv_vredmaxu_vs_u16m1_u16m1(best_u16m1, identity_u16m1, vector_length_max);
    nk_u16_t best = (nk_u16_t)__riscv_vmv_x_s_u16m1_u16(reduced_u16m1);

    vbool16_t match_b16 = __riscv_vmseq_vx_u16m1_b16(best_u16m1, best, vector_length_max);
    vuint64m4_t sentinel_u64m4 = __riscv_vmv_v_x_u64m4(NK_U64_MAX, vector_length_max);
    vuint64m4_t candidates_u64m4 = __riscv_vmerge_vvm_u64m4(sentinel_u64m4, best_index_u64m4, match_b16,
                                                            vector_length_max);
    vuint64m1_t identity_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    vuint64m1_t reduced_u64m1 = __riscv_vredminu_vs_u64m4_u64m1(candidates_u64m4, identity_u64m1, vector_length_max);

    *max_value = best;
    *max_index = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(reduced_u64m1);
}

NK_INTERNAL void nk_reduce_max_u16_rvv_strided_(                   //
    nk_u16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u16_t *max_value, nk_size_t *max_index) {

    nk_size_t vector_length_max = __riscv_vsetvlmax_e16m1();
    vuint16m1_t best_u16m1 = __riscv_vmv_v_x_u16m1(0, vector_length_max);
    vuint64m4_t best_index_u64m4 = __riscv_vmv_v_x_u64m4(0, vector_length_max);

    unsigned char const *ptr = (unsigned char const *)data;
    nk_size_t offset = 0;
    for (nk_size_t remaining = count, vector_length; remaining > 0;
         remaining -= vector_length, offset += vector_length, ptr += vector_length * stride_bytes) {
        vector_length = __riscv_vsetvl_e16m1(remaining);
        vuint16m1_t chunk_u16m1 = __riscv_vlse16_v_u16m1((nk_u16_t const *)ptr, (nk_ssize_t)stride_bytes,
                                                         vector_length);

        vbool16_t greater_b16 = __riscv_vmsltu_vv_u16m1_b16(best_u16m1, chunk_u16m1, vector_length);
        best_u16m1 = __riscv_vmerge_vvm_u16m1(best_u16m1, chunk_u16m1, greater_b16, vector_length);

        vuint64m4_t position_u64m4 = __riscv_vid_v_u64m4(vector_length);
        position_u64m4 = __riscv_vadd_vx_u64m4(position_u64m4, (nk_u64_t)offset, vector_length);
        best_index_u64m4 = __riscv_vmerge_vvm_u64m4(best_index_u64m4, position_u64m4, greater_b16, vector_length);
    }

    vuint16m1_t identity_u16m1 = __riscv_vmv_v_x_u16m1(0, 1);
    vuint16m1_t reduced_u16m1 = __riscv_vredmaxu_vs_u16m1_u16m1(best_u16m1, identity_u16m1, vector_length_max);
    nk_u16_t best = (nk_u16_t)__riscv_vmv_x_s_u16m1_u16(reduced_u16m1);

    vbool16_t match_b16 = __riscv_vmseq_vx_u16m1_b16(best_u16m1, best, vector_length_max);
    vuint64m4_t sentinel_u64m4 = __riscv_vmv_v_x_u64m4(NK_U64_MAX, vector_length_max);
    vuint64m4_t candidates_u64m4 = __riscv_vmerge_vvm_u64m4(sentinel_u64m4, best_index_u64m4, match_b16,
                                                            vector_length_max);
    vuint64m1_t identity_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    vuint64m1_t reduced_u64m1 = __riscv_vredminu_vs_u64m4_u64m1(candidates_u64m4, identity_u64m1, vector_length_max);

    *max_value = best;
    *max_index = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(reduced_u64m1);
}

NK_PUBLIC void nk_reduce_max_u16_rvv(                              //
    nk_u16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u16_t *max_value, nk_size_t *max_index) {
    if (count == 0) *max_value = 0, *max_index = 0;
    else if (stride_bytes == sizeof(nk_u16_t)) nk_reduce_max_u16_rvv_contiguous_(data, count, max_value, max_index);
    else nk_reduce_max_u16_rvv_strided_(data, count, stride_bytes, max_value, max_index);
}

NK_INTERNAL void nk_reduce_min_i32_rvv_contiguous_( //
    nk_i32_t const *data, nk_size_t count,          //
    nk_i32_t *min_value, nk_size_t *min_index) {

    nk_size_t vector_length_max = __riscv_vsetvlmax_e32m1();
    vint32m1_t best_i32m1 = __riscv_vmv_v_x_i32m1(NK_I32_MAX, vector_length_max);
    vuint64m2_t best_index_u64m2 = __riscv_vmv_v_x_u64m2(0, vector_length_max);

    nk_size_t offset = 0;
    for (nk_size_t remaining = count, vector_length; remaining > 0;
         remaining -= vector_length, offset += vector_length) {
        vector_length = __riscv_vsetvl_e32m1(remaining);
        vint32m1_t chunk_i32m1 = __riscv_vle32_v_i32m1(data + offset, vector_length);

        vbool32_t less_b32 = __riscv_vmslt_vv_i32m1_b32(chunk_i32m1, best_i32m1, vector_length);
        best_i32m1 = __riscv_vmerge_vvm_i32m1(best_i32m1, chunk_i32m1, less_b32, vector_length);

        vuint64m2_t position_u64m2 = __riscv_vid_v_u64m2(vector_length);
        position_u64m2 = __riscv_vadd_vx_u64m2(position_u64m2, (nk_u64_t)offset, vector_length);
        best_index_u64m2 = __riscv_vmerge_vvm_u64m2(best_index_u64m2, position_u64m2, less_b32, vector_length);
    }

    vint32m1_t identity_i32m1 = __riscv_vmv_v_x_i32m1(NK_I32_MAX, 1);
    vint32m1_t reduced_i32m1 = __riscv_vredmin_vs_i32m1_i32m1(best_i32m1, identity_i32m1, vector_length_max);
    nk_i32_t best = (nk_i32_t)__riscv_vmv_x_s_i32m1_i32(reduced_i32m1);

    vbool32_t match_b32 = __riscv_vmseq_vx_i32m1_b32(best_i32m1, best, vector_length_max);
    vuint64m2_t sentinel_u64m2 = __riscv_vmv_v_x_u64m2(NK_U64_MAX, vector_length_max);
    vuint64m2_t candidates_u64m2 = __riscv_vmerge_vvm_u64m2(sentinel_u64m2, best_index_u64m2, match_b32,
                                                            vector_length_max);
    vuint64m1_t identity_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    vuint64m1_t reduced_u64m1 = __riscv_vredminu_vs_u64m2_u64m1(candidates_u64m2, identity_u64m1, vector_length_max);

    *min_value = best;
    *min_index = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(reduced_u64m1);
}

NK_INTERNAL void nk_reduce_min_i32_rvv_strided_(                   //
    nk_i32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i32_t *min_value, nk_size_t *min_index) {

    nk_size_t vector_length_max = __riscv_vsetvlmax_e32m1();
    vint32m1_t best_i32m1 = __riscv_vmv_v_x_i32m1(NK_I32_MAX, vector_length_max);
    vuint64m2_t best_index_u64m2 = __riscv_vmv_v_x_u64m2(0, vector_length_max);

    unsigned char const *ptr = (unsigned char const *)data;
    nk_size_t offset = 0;
    for (nk_size_t remaining = count, vector_length; remaining > 0;
         remaining -= vector_length, offset += vector_length, ptr += vector_length * stride_bytes) {
        vector_length = __riscv_vsetvl_e32m1(remaining);
        vint32m1_t chunk_i32m1 = __riscv_vlse32_v_i32m1((nk_i32_t const *)ptr, (nk_ssize_t)stride_bytes, vector_length);

        vbool32_t less_b32 = __riscv_vmslt_vv_i32m1_b32(chunk_i32m1, best_i32m1, vector_length);
        best_i32m1 = __riscv_vmerge_vvm_i32m1(best_i32m1, chunk_i32m1, less_b32, vector_length);

        vuint64m2_t position_u64m2 = __riscv_vid_v_u64m2(vector_length);
        position_u64m2 = __riscv_vadd_vx_u64m2(position_u64m2, (nk_u64_t)offset, vector_length);
        best_index_u64m2 = __riscv_vmerge_vvm_u64m2(best_index_u64m2, position_u64m2, less_b32, vector_length);
    }

    vint32m1_t identity_i32m1 = __riscv_vmv_v_x_i32m1(NK_I32_MAX, 1);
    vint32m1_t reduced_i32m1 = __riscv_vredmin_vs_i32m1_i32m1(best_i32m1, identity_i32m1, vector_length_max);
    nk_i32_t best = (nk_i32_t)__riscv_vmv_x_s_i32m1_i32(reduced_i32m1);

    vbool32_t match_b32 = __riscv_vmseq_vx_i32m1_b32(best_i32m1, best, vector_length_max);
    vuint64m2_t sentinel_u64m2 = __riscv_vmv_v_x_u64m2(NK_U64_MAX, vector_length_max);
    vuint64m2_t candidates_u64m2 = __riscv_vmerge_vvm_u64m2(sentinel_u64m2, best_index_u64m2, match_b32,
                                                            vector_length_max);
    vuint64m1_t identity_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    vuint64m1_t reduced_u64m1 = __riscv_vredminu_vs_u64m2_u64m1(candidates_u64m2, identity_u64m1, vector_length_max);

    *min_value = best;
    *min_index = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(reduced_u64m1);
}

NK_PUBLIC void nk_reduce_min_i32_rvv(                              //
    nk_i32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i32_t *min_value, nk_size_t *min_index) {
    if (count == 0) *min_value = NK_I32_MAX, *min_index = 0;
    else if (stride_bytes == sizeof(nk_i32_t)) nk_reduce_min_i32_rvv_contiguous_(data, count, min_value, min_index);
    else nk_reduce_min_i32_rvv_strided_(data, count, stride_bytes, min_value, min_index);
}

NK_INTERNAL void nk_reduce_max_i32_rvv_contiguous_( //
    nk_i32_t const *data, nk_size_t count,          //
    nk_i32_t *max_value, nk_size_t *max_index) {

    nk_size_t vector_length_max = __riscv_vsetvlmax_e32m1();
    vint32m1_t best_i32m1 = __riscv_vmv_v_x_i32m1(NK_I32_MIN, vector_length_max);
    vuint64m2_t best_index_u64m2 = __riscv_vmv_v_x_u64m2(0, vector_length_max);

    nk_size_t offset = 0;
    for (nk_size_t remaining = count, vector_length; remaining > 0;
         remaining -= vector_length, offset += vector_length) {
        vector_length = __riscv_vsetvl_e32m1(remaining);
        vint32m1_t chunk_i32m1 = __riscv_vle32_v_i32m1(data + offset, vector_length);

        vbool32_t greater_b32 = __riscv_vmslt_vv_i32m1_b32(best_i32m1, chunk_i32m1, vector_length);
        best_i32m1 = __riscv_vmerge_vvm_i32m1(best_i32m1, chunk_i32m1, greater_b32, vector_length);

        vuint64m2_t position_u64m2 = __riscv_vid_v_u64m2(vector_length);
        position_u64m2 = __riscv_vadd_vx_u64m2(position_u64m2, (nk_u64_t)offset, vector_length);
        best_index_u64m2 = __riscv_vmerge_vvm_u64m2(best_index_u64m2, position_u64m2, greater_b32, vector_length);
    }

    vint32m1_t identity_i32m1 = __riscv_vmv_v_x_i32m1(NK_I32_MIN, 1);
    vint32m1_t reduced_i32m1 = __riscv_vredmax_vs_i32m1_i32m1(best_i32m1, identity_i32m1, vector_length_max);
    nk_i32_t best = (nk_i32_t)__riscv_vmv_x_s_i32m1_i32(reduced_i32m1);

    vbool32_t match_b32 = __riscv_vmseq_vx_i32m1_b32(best_i32m1, best, vector_length_max);
    vuint64m2_t sentinel_u64m2 = __riscv_vmv_v_x_u64m2(NK_U64_MAX, vector_length_max);
    vuint64m2_t candidates_u64m2 = __riscv_vmerge_vvm_u64m2(sentinel_u64m2, best_index_u64m2, match_b32,
                                                            vector_length_max);
    vuint64m1_t identity_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    vuint64m1_t reduced_u64m1 = __riscv_vredminu_vs_u64m2_u64m1(candidates_u64m2, identity_u64m1, vector_length_max);

    *max_value = best;
    *max_index = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(reduced_u64m1);
}

NK_INTERNAL void nk_reduce_max_i32_rvv_strided_(                   //
    nk_i32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i32_t *max_value, nk_size_t *max_index) {

    nk_size_t vector_length_max = __riscv_vsetvlmax_e32m1();
    vint32m1_t best_i32m1 = __riscv_vmv_v_x_i32m1(NK_I32_MIN, vector_length_max);
    vuint64m2_t best_index_u64m2 = __riscv_vmv_v_x_u64m2(0, vector_length_max);

    unsigned char const *ptr = (unsigned char const *)data;
    nk_size_t offset = 0;
    for (nk_size_t remaining = count, vector_length; remaining > 0;
         remaining -= vector_length, offset += vector_length, ptr += vector_length * stride_bytes) {
        vector_length = __riscv_vsetvl_e32m1(remaining);
        vint32m1_t chunk_i32m1 = __riscv_vlse32_v_i32m1((nk_i32_t const *)ptr, (nk_ssize_t)stride_bytes, vector_length);

        vbool32_t greater_b32 = __riscv_vmslt_vv_i32m1_b32(best_i32m1, chunk_i32m1, vector_length);
        best_i32m1 = __riscv_vmerge_vvm_i32m1(best_i32m1, chunk_i32m1, greater_b32, vector_length);

        vuint64m2_t position_u64m2 = __riscv_vid_v_u64m2(vector_length);
        position_u64m2 = __riscv_vadd_vx_u64m2(position_u64m2, (nk_u64_t)offset, vector_length);
        best_index_u64m2 = __riscv_vmerge_vvm_u64m2(best_index_u64m2, position_u64m2, greater_b32, vector_length);
    }

    vint32m1_t identity_i32m1 = __riscv_vmv_v_x_i32m1(NK_I32_MIN, 1);
    vint32m1_t reduced_i32m1 = __riscv_vredmax_vs_i32m1_i32m1(best_i32m1, identity_i32m1, vector_length_max);
    nk_i32_t best = (nk_i32_t)__riscv_vmv_x_s_i32m1_i32(reduced_i32m1);

    vbool32_t match_b32 = __riscv_vmseq_vx_i32m1_b32(best_i32m1, best, vector_length_max);
    vuint64m2_t sentinel_u64m2 = __riscv_vmv_v_x_u64m2(NK_U64_MAX, vector_length_max);
    vuint64m2_t candidates_u64m2 = __riscv_vmerge_vvm_u64m2(sentinel_u64m2, best_index_u64m2, match_b32,
                                                            vector_length_max);
    vuint64m1_t identity_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    vuint64m1_t reduced_u64m1 = __riscv_vredminu_vs_u64m2_u64m1(candidates_u64m2, identity_u64m1, vector_length_max);

    *max_value = best;
    *max_index = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(reduced_u64m1);
}

NK_PUBLIC void nk_reduce_max_i32_rvv(                              //
    nk_i32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i32_t *max_value, nk_size_t *max_index) {
    if (count == 0) *max_value = NK_I32_MIN, *max_index = 0;
    else if (stride_bytes == sizeof(nk_i32_t)) nk_reduce_max_i32_rvv_contiguous_(data, count, max_value, max_index);
    else nk_reduce_max_i32_rvv_strided_(data, count, stride_bytes, max_value, max_index);
}

NK_INTERNAL void nk_reduce_min_u32_rvv_contiguous_( //
    nk_u32_t const *data, nk_size_t count,          //
    nk_u32_t *min_value, nk_size_t *min_index) {

    nk_size_t vector_length_max = __riscv_vsetvlmax_e32m1();
    vuint32m1_t best_u32m1 = __riscv_vmv_v_x_u32m1(NK_U32_MAX, vector_length_max);
    vuint64m2_t best_index_u64m2 = __riscv_vmv_v_x_u64m2(0, vector_length_max);

    nk_size_t offset = 0;
    for (nk_size_t remaining = count, vector_length; remaining > 0;
         remaining -= vector_length, offset += vector_length) {
        vector_length = __riscv_vsetvl_e32m1(remaining);
        vuint32m1_t chunk_u32m1 = __riscv_vle32_v_u32m1(data + offset, vector_length);

        vbool32_t less_b32 = __riscv_vmsltu_vv_u32m1_b32(chunk_u32m1, best_u32m1, vector_length);
        best_u32m1 = __riscv_vmerge_vvm_u32m1(best_u32m1, chunk_u32m1, less_b32, vector_length);

        vuint64m2_t position_u64m2 = __riscv_vid_v_u64m2(vector_length);
        position_u64m2 = __riscv_vadd_vx_u64m2(position_u64m2, (nk_u64_t)offset, vector_length);
        best_index_u64m2 = __riscv_vmerge_vvm_u64m2(best_index_u64m2, position_u64m2, less_b32, vector_length);
    }

    vuint32m1_t identity_u32m1 = __riscv_vmv_v_x_u32m1(NK_U32_MAX, 1);
    vuint32m1_t reduced_u32m1 = __riscv_vredminu_vs_u32m1_u32m1(best_u32m1, identity_u32m1, vector_length_max);
    nk_u32_t best = (nk_u32_t)__riscv_vmv_x_s_u32m1_u32(reduced_u32m1);

    vbool32_t match_b32 = __riscv_vmseq_vx_u32m1_b32(best_u32m1, best, vector_length_max);
    vuint64m2_t sentinel_u64m2 = __riscv_vmv_v_x_u64m2(NK_U64_MAX, vector_length_max);
    vuint64m2_t candidates_u64m2 = __riscv_vmerge_vvm_u64m2(sentinel_u64m2, best_index_u64m2, match_b32,
                                                            vector_length_max);
    vuint64m1_t idx_identity_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    vuint64m1_t idx_reduced_u64m1 = __riscv_vredminu_vs_u64m2_u64m1(candidates_u64m2, idx_identity_u64m1,
                                                                    vector_length_max);

    *min_value = best;
    *min_index = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(idx_reduced_u64m1);
}

NK_INTERNAL void nk_reduce_min_u32_rvv_strided_(                   //
    nk_u32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u32_t *min_value, nk_size_t *min_index) {

    nk_size_t vector_length_max = __riscv_vsetvlmax_e32m1();
    vuint32m1_t best_u32m1 = __riscv_vmv_v_x_u32m1(NK_U32_MAX, vector_length_max);
    vuint64m2_t best_index_u64m2 = __riscv_vmv_v_x_u64m2(0, vector_length_max);

    unsigned char const *ptr = (unsigned char const *)data;
    nk_size_t offset = 0;
    for (nk_size_t remaining = count, vector_length; remaining > 0;
         remaining -= vector_length, offset += vector_length, ptr += vector_length * stride_bytes) {
        vector_length = __riscv_vsetvl_e32m1(remaining);
        vuint32m1_t chunk_u32m1 = __riscv_vlse32_v_u32m1((nk_u32_t const *)ptr, (nk_ssize_t)stride_bytes,
                                                         vector_length);

        vbool32_t less_b32 = __riscv_vmsltu_vv_u32m1_b32(chunk_u32m1, best_u32m1, vector_length);
        best_u32m1 = __riscv_vmerge_vvm_u32m1(best_u32m1, chunk_u32m1, less_b32, vector_length);

        vuint64m2_t position_u64m2 = __riscv_vid_v_u64m2(vector_length);
        position_u64m2 = __riscv_vadd_vx_u64m2(position_u64m2, (nk_u64_t)offset, vector_length);
        best_index_u64m2 = __riscv_vmerge_vvm_u64m2(best_index_u64m2, position_u64m2, less_b32, vector_length);
    }

    vuint32m1_t identity_u32m1 = __riscv_vmv_v_x_u32m1(NK_U32_MAX, 1);
    vuint32m1_t reduced_u32m1 = __riscv_vredminu_vs_u32m1_u32m1(best_u32m1, identity_u32m1, vector_length_max);
    nk_u32_t best = (nk_u32_t)__riscv_vmv_x_s_u32m1_u32(reduced_u32m1);

    vbool32_t match_b32 = __riscv_vmseq_vx_u32m1_b32(best_u32m1, best, vector_length_max);
    vuint64m2_t sentinel_u64m2 = __riscv_vmv_v_x_u64m2(NK_U64_MAX, vector_length_max);
    vuint64m2_t candidates_u64m2 = __riscv_vmerge_vvm_u64m2(sentinel_u64m2, best_index_u64m2, match_b32,
                                                            vector_length_max);
    vuint64m1_t idx_identity_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    vuint64m1_t idx_reduced_u64m1 = __riscv_vredminu_vs_u64m2_u64m1(candidates_u64m2, idx_identity_u64m1,
                                                                    vector_length_max);

    *min_value = best;
    *min_index = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(idx_reduced_u64m1);
}

NK_PUBLIC void nk_reduce_min_u32_rvv(                              //
    nk_u32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u32_t *min_value, nk_size_t *min_index) {
    if (count == 0) *min_value = NK_U32_MAX, *min_index = 0;
    else if (stride_bytes == sizeof(nk_u32_t)) nk_reduce_min_u32_rvv_contiguous_(data, count, min_value, min_index);
    else nk_reduce_min_u32_rvv_strided_(data, count, stride_bytes, min_value, min_index);
}

NK_INTERNAL void nk_reduce_max_u32_rvv_contiguous_( //
    nk_u32_t const *data, nk_size_t count,          //
    nk_u32_t *max_value, nk_size_t *max_index) {

    nk_size_t vector_length_max = __riscv_vsetvlmax_e32m1();
    vuint32m1_t best_u32m1 = __riscv_vmv_v_x_u32m1(0, vector_length_max);
    vuint64m2_t best_index_u64m2 = __riscv_vmv_v_x_u64m2(0, vector_length_max);

    nk_size_t offset = 0;
    for (nk_size_t remaining = count, vector_length; remaining > 0;
         remaining -= vector_length, offset += vector_length) {
        vector_length = __riscv_vsetvl_e32m1(remaining);
        vuint32m1_t chunk_u32m1 = __riscv_vle32_v_u32m1(data + offset, vector_length);

        vbool32_t greater_b32 = __riscv_vmsltu_vv_u32m1_b32(best_u32m1, chunk_u32m1, vector_length);
        best_u32m1 = __riscv_vmerge_vvm_u32m1(best_u32m1, chunk_u32m1, greater_b32, vector_length);

        vuint64m2_t position_u64m2 = __riscv_vid_v_u64m2(vector_length);
        position_u64m2 = __riscv_vadd_vx_u64m2(position_u64m2, (nk_u64_t)offset, vector_length);
        best_index_u64m2 = __riscv_vmerge_vvm_u64m2(best_index_u64m2, position_u64m2, greater_b32, vector_length);
    }

    vuint32m1_t identity_u32m1 = __riscv_vmv_v_x_u32m1(0, 1);
    vuint32m1_t reduced_u32m1 = __riscv_vredmaxu_vs_u32m1_u32m1(best_u32m1, identity_u32m1, vector_length_max);
    nk_u32_t best = (nk_u32_t)__riscv_vmv_x_s_u32m1_u32(reduced_u32m1);

    vbool32_t match_b32 = __riscv_vmseq_vx_u32m1_b32(best_u32m1, best, vector_length_max);
    vuint64m2_t sentinel_u64m2 = __riscv_vmv_v_x_u64m2(NK_U64_MAX, vector_length_max);
    vuint64m2_t candidates_u64m2 = __riscv_vmerge_vvm_u64m2(sentinel_u64m2, best_index_u64m2, match_b32,
                                                            vector_length_max);
    vuint64m1_t idx_identity_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    vuint64m1_t idx_reduced_u64m1 = __riscv_vredminu_vs_u64m2_u64m1(candidates_u64m2, idx_identity_u64m1,
                                                                    vector_length_max);

    *max_value = best;
    *max_index = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(idx_reduced_u64m1);
}

NK_INTERNAL void nk_reduce_max_u32_rvv_strided_(                   //
    nk_u32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u32_t *max_value, nk_size_t *max_index) {

    nk_size_t vector_length_max = __riscv_vsetvlmax_e32m1();
    vuint32m1_t best_u32m1 = __riscv_vmv_v_x_u32m1(0, vector_length_max);
    vuint64m2_t best_index_u64m2 = __riscv_vmv_v_x_u64m2(0, vector_length_max);

    unsigned char const *ptr = (unsigned char const *)data;
    nk_size_t offset = 0;
    for (nk_size_t remaining = count, vector_length; remaining > 0;
         remaining -= vector_length, offset += vector_length, ptr += vector_length * stride_bytes) {
        vector_length = __riscv_vsetvl_e32m1(remaining);
        vuint32m1_t chunk_u32m1 = __riscv_vlse32_v_u32m1((nk_u32_t const *)ptr, (nk_ssize_t)stride_bytes,
                                                         vector_length);

        vbool32_t greater_b32 = __riscv_vmsltu_vv_u32m1_b32(best_u32m1, chunk_u32m1, vector_length);
        best_u32m1 = __riscv_vmerge_vvm_u32m1(best_u32m1, chunk_u32m1, greater_b32, vector_length);

        vuint64m2_t position_u64m2 = __riscv_vid_v_u64m2(vector_length);
        position_u64m2 = __riscv_vadd_vx_u64m2(position_u64m2, (nk_u64_t)offset, vector_length);
        best_index_u64m2 = __riscv_vmerge_vvm_u64m2(best_index_u64m2, position_u64m2, greater_b32, vector_length);
    }

    vuint32m1_t identity_u32m1 = __riscv_vmv_v_x_u32m1(0, 1);
    vuint32m1_t reduced_u32m1 = __riscv_vredmaxu_vs_u32m1_u32m1(best_u32m1, identity_u32m1, vector_length_max);
    nk_u32_t best = (nk_u32_t)__riscv_vmv_x_s_u32m1_u32(reduced_u32m1);

    vbool32_t match_b32 = __riscv_vmseq_vx_u32m1_b32(best_u32m1, best, vector_length_max);
    vuint64m2_t sentinel_u64m2 = __riscv_vmv_v_x_u64m2(NK_U64_MAX, vector_length_max);
    vuint64m2_t candidates_u64m2 = __riscv_vmerge_vvm_u64m2(sentinel_u64m2, best_index_u64m2, match_b32,
                                                            vector_length_max);
    vuint64m1_t idx_identity_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    vuint64m1_t idx_reduced_u64m1 = __riscv_vredminu_vs_u64m2_u64m1(candidates_u64m2, idx_identity_u64m1,
                                                                    vector_length_max);

    *max_value = best;
    *max_index = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(idx_reduced_u64m1);
}

NK_PUBLIC void nk_reduce_max_u32_rvv(                              //
    nk_u32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u32_t *max_value, nk_size_t *max_index) {
    if (count == 0) *max_value = 0, *max_index = 0;
    else if (stride_bytes == sizeof(nk_u32_t)) nk_reduce_max_u32_rvv_contiguous_(data, count, max_value, max_index);
    else nk_reduce_max_u32_rvv_strided_(data, count, stride_bytes, max_value, max_index);
}

NK_INTERNAL void nk_reduce_min_i64_rvv_contiguous_( //
    nk_i64_t const *data, nk_size_t count,          //
    nk_i64_t *min_value, nk_size_t *min_index) {

    nk_size_t vector_length_max = __riscv_vsetvlmax_e64m1();
    vint64m1_t best_i64m1 = __riscv_vmv_v_x_i64m1(NK_I64_MAX, vector_length_max);
    vuint64m1_t best_index_u64m1 = __riscv_vmv_v_x_u64m1(0, vector_length_max);

    nk_size_t offset = 0;
    for (nk_size_t remaining = count, vector_length; remaining > 0;
         remaining -= vector_length, offset += vector_length) {
        vector_length = __riscv_vsetvl_e64m1(remaining);
        vint64m1_t chunk_i64m1 = __riscv_vle64_v_i64m1(data + offset, vector_length);

        vbool64_t less_b64 = __riscv_vmslt_vv_i64m1_b64(chunk_i64m1, best_i64m1, vector_length);
        best_i64m1 = __riscv_vmerge_vvm_i64m1(best_i64m1, chunk_i64m1, less_b64, vector_length);

        vuint64m1_t position_u64m1 = __riscv_vid_v_u64m1(vector_length);
        position_u64m1 = __riscv_vadd_vx_u64m1(position_u64m1, (nk_u64_t)offset, vector_length);
        best_index_u64m1 = __riscv_vmerge_vvm_u64m1(best_index_u64m1, position_u64m1, less_b64, vector_length);
    }

    vint64m1_t identity_i64m1 = __riscv_vmv_v_x_i64m1(NK_I64_MAX, 1);
    vint64m1_t reduced_i64m1 = __riscv_vredmin_vs_i64m1_i64m1(best_i64m1, identity_i64m1, vector_length_max);
    nk_i64_t best = (nk_i64_t)__riscv_vmv_x_s_i64m1_i64(reduced_i64m1);

    vbool64_t match_b64 = __riscv_vmseq_vx_i64m1_b64(best_i64m1, best, vector_length_max);
    vuint64m1_t sentinel_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, vector_length_max);
    vuint64m1_t candidates_u64m1 = __riscv_vmerge_vvm_u64m1(sentinel_u64m1, best_index_u64m1, match_b64,
                                                            vector_length_max);
    vuint64m1_t identity_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    vuint64m1_t reduced_u64m1 = __riscv_vredminu_vs_u64m1_u64m1(candidates_u64m1, identity_u64m1, vector_length_max);

    *min_value = best;
    *min_index = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(reduced_u64m1);
}

NK_INTERNAL void nk_reduce_min_i64_rvv_strided_(                   //
    nk_i64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *min_value, nk_size_t *min_index) {

    nk_size_t vector_length_max = __riscv_vsetvlmax_e64m1();
    vint64m1_t best_i64m1 = __riscv_vmv_v_x_i64m1(NK_I64_MAX, vector_length_max);
    vuint64m1_t best_index_u64m1 = __riscv_vmv_v_x_u64m1(0, vector_length_max);

    unsigned char const *ptr = (unsigned char const *)data;
    nk_size_t offset = 0;
    for (nk_size_t remaining = count, vector_length; remaining > 0;
         remaining -= vector_length, offset += vector_length, ptr += vector_length * stride_bytes) {
        vector_length = __riscv_vsetvl_e64m1(remaining);
        vint64m1_t chunk_i64m1 = __riscv_vlse64_v_i64m1((nk_i64_t const *)ptr, (nk_ssize_t)stride_bytes, vector_length);

        vbool64_t less_b64 = __riscv_vmslt_vv_i64m1_b64(chunk_i64m1, best_i64m1, vector_length);
        best_i64m1 = __riscv_vmerge_vvm_i64m1(best_i64m1, chunk_i64m1, less_b64, vector_length);

        vuint64m1_t position_u64m1 = __riscv_vid_v_u64m1(vector_length);
        position_u64m1 = __riscv_vadd_vx_u64m1(position_u64m1, (nk_u64_t)offset, vector_length);
        best_index_u64m1 = __riscv_vmerge_vvm_u64m1(best_index_u64m1, position_u64m1, less_b64, vector_length);
    }

    vint64m1_t identity_i64m1 = __riscv_vmv_v_x_i64m1(NK_I64_MAX, 1);
    vint64m1_t reduced_i64m1 = __riscv_vredmin_vs_i64m1_i64m1(best_i64m1, identity_i64m1, vector_length_max);
    nk_i64_t best = (nk_i64_t)__riscv_vmv_x_s_i64m1_i64(reduced_i64m1);

    vbool64_t match_b64 = __riscv_vmseq_vx_i64m1_b64(best_i64m1, best, vector_length_max);
    vuint64m1_t sentinel_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, vector_length_max);
    vuint64m1_t candidates_u64m1 = __riscv_vmerge_vvm_u64m1(sentinel_u64m1, best_index_u64m1, match_b64,
                                                            vector_length_max);
    vuint64m1_t identity_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    vuint64m1_t reduced_u64m1 = __riscv_vredminu_vs_u64m1_u64m1(candidates_u64m1, identity_u64m1, vector_length_max);

    *min_value = best;
    *min_index = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(reduced_u64m1);
}

NK_PUBLIC void nk_reduce_min_i64_rvv(                              //
    nk_i64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *min_value, nk_size_t *min_index) {
    if (count == 0) *min_value = NK_I64_MAX, *min_index = 0;
    else if (stride_bytes == sizeof(nk_i64_t)) nk_reduce_min_i64_rvv_contiguous_(data, count, min_value, min_index);
    else nk_reduce_min_i64_rvv_strided_(data, count, stride_bytes, min_value, min_index);
}

NK_INTERNAL void nk_reduce_max_i64_rvv_contiguous_( //
    nk_i64_t const *data, nk_size_t count,          //
    nk_i64_t *max_value, nk_size_t *max_index) {

    nk_size_t vector_length_max = __riscv_vsetvlmax_e64m1();
    vint64m1_t best_i64m1 = __riscv_vmv_v_x_i64m1(NK_I64_MIN, vector_length_max);
    vuint64m1_t best_index_u64m1 = __riscv_vmv_v_x_u64m1(0, vector_length_max);

    nk_size_t offset = 0;
    for (nk_size_t remaining = count, vector_length; remaining > 0;
         remaining -= vector_length, offset += vector_length) {
        vector_length = __riscv_vsetvl_e64m1(remaining);
        vint64m1_t chunk_i64m1 = __riscv_vle64_v_i64m1(data + offset, vector_length);

        vbool64_t greater_b64 = __riscv_vmslt_vv_i64m1_b64(best_i64m1, chunk_i64m1, vector_length);
        best_i64m1 = __riscv_vmerge_vvm_i64m1(best_i64m1, chunk_i64m1, greater_b64, vector_length);

        vuint64m1_t position_u64m1 = __riscv_vid_v_u64m1(vector_length);
        position_u64m1 = __riscv_vadd_vx_u64m1(position_u64m1, (nk_u64_t)offset, vector_length);
        best_index_u64m1 = __riscv_vmerge_vvm_u64m1(best_index_u64m1, position_u64m1, greater_b64, vector_length);
    }

    vint64m1_t identity_i64m1 = __riscv_vmv_v_x_i64m1(NK_I64_MIN, 1);
    vint64m1_t reduced_i64m1 = __riscv_vredmax_vs_i64m1_i64m1(best_i64m1, identity_i64m1, vector_length_max);
    nk_i64_t best = (nk_i64_t)__riscv_vmv_x_s_i64m1_i64(reduced_i64m1);

    vbool64_t match_b64 = __riscv_vmseq_vx_i64m1_b64(best_i64m1, best, vector_length_max);
    vuint64m1_t sentinel_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, vector_length_max);
    vuint64m1_t candidates_u64m1 = __riscv_vmerge_vvm_u64m1(sentinel_u64m1, best_index_u64m1, match_b64,
                                                            vector_length_max);
    vuint64m1_t identity_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    vuint64m1_t reduced_u64m1 = __riscv_vredminu_vs_u64m1_u64m1(candidates_u64m1, identity_u64m1, vector_length_max);

    *max_value = best;
    *max_index = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(reduced_u64m1);
}

NK_INTERNAL void nk_reduce_max_i64_rvv_strided_(                   //
    nk_i64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *max_value, nk_size_t *max_index) {

    nk_size_t vector_length_max = __riscv_vsetvlmax_e64m1();
    vint64m1_t best_i64m1 = __riscv_vmv_v_x_i64m1(NK_I64_MIN, vector_length_max);
    vuint64m1_t best_index_u64m1 = __riscv_vmv_v_x_u64m1(0, vector_length_max);

    unsigned char const *ptr = (unsigned char const *)data;
    nk_size_t offset = 0;
    for (nk_size_t remaining = count, vector_length; remaining > 0;
         remaining -= vector_length, offset += vector_length, ptr += vector_length * stride_bytes) {
        vector_length = __riscv_vsetvl_e64m1(remaining);
        vint64m1_t chunk_i64m1 = __riscv_vlse64_v_i64m1((nk_i64_t const *)ptr, (nk_ssize_t)stride_bytes, vector_length);

        vbool64_t greater_b64 = __riscv_vmslt_vv_i64m1_b64(best_i64m1, chunk_i64m1, vector_length);
        best_i64m1 = __riscv_vmerge_vvm_i64m1(best_i64m1, chunk_i64m1, greater_b64, vector_length);

        vuint64m1_t position_u64m1 = __riscv_vid_v_u64m1(vector_length);
        position_u64m1 = __riscv_vadd_vx_u64m1(position_u64m1, (nk_u64_t)offset, vector_length);
        best_index_u64m1 = __riscv_vmerge_vvm_u64m1(best_index_u64m1, position_u64m1, greater_b64, vector_length);
    }

    vint64m1_t identity_i64m1 = __riscv_vmv_v_x_i64m1(NK_I64_MIN, 1);
    vint64m1_t reduced_i64m1 = __riscv_vredmax_vs_i64m1_i64m1(best_i64m1, identity_i64m1, vector_length_max);
    nk_i64_t best = (nk_i64_t)__riscv_vmv_x_s_i64m1_i64(reduced_i64m1);

    vbool64_t match_b64 = __riscv_vmseq_vx_i64m1_b64(best_i64m1, best, vector_length_max);
    vuint64m1_t sentinel_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, vector_length_max);
    vuint64m1_t candidates_u64m1 = __riscv_vmerge_vvm_u64m1(sentinel_u64m1, best_index_u64m1, match_b64,
                                                            vector_length_max);
    vuint64m1_t identity_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    vuint64m1_t reduced_u64m1 = __riscv_vredminu_vs_u64m1_u64m1(candidates_u64m1, identity_u64m1, vector_length_max);

    *max_value = best;
    *max_index = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(reduced_u64m1);
}

NK_PUBLIC void nk_reduce_max_i64_rvv(                              //
    nk_i64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *max_value, nk_size_t *max_index) {
    if (count == 0) *max_value = NK_I64_MIN, *max_index = 0;
    else if (stride_bytes == sizeof(nk_i64_t)) nk_reduce_max_i64_rvv_contiguous_(data, count, max_value, max_index);
    else nk_reduce_max_i64_rvv_strided_(data, count, stride_bytes, max_value, max_index);
}

NK_INTERNAL void nk_reduce_min_u64_rvv_contiguous_( //
    nk_u64_t const *data, nk_size_t count,          //
    nk_u64_t *min_value, nk_size_t *min_index) {

    nk_size_t vector_length_max = __riscv_vsetvlmax_e64m1();
    vuint64m1_t best_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, vector_length_max);
    vuint64m1_t best_index_u64m1 = __riscv_vmv_v_x_u64m1(0, vector_length_max);

    nk_size_t offset = 0;
    for (nk_size_t remaining = count, vector_length; remaining > 0;
         remaining -= vector_length, offset += vector_length) {
        vector_length = __riscv_vsetvl_e64m1(remaining);
        vuint64m1_t chunk_u64m1 = __riscv_vle64_v_u64m1(data + offset, vector_length);

        vbool64_t less_b64 = __riscv_vmsltu_vv_u64m1_b64(chunk_u64m1, best_u64m1, vector_length);
        best_u64m1 = __riscv_vmerge_vvm_u64m1(best_u64m1, chunk_u64m1, less_b64, vector_length);

        vuint64m1_t position_u64m1 = __riscv_vid_v_u64m1(vector_length);
        position_u64m1 = __riscv_vadd_vx_u64m1(position_u64m1, (nk_u64_t)offset, vector_length);
        best_index_u64m1 = __riscv_vmerge_vvm_u64m1(best_index_u64m1, position_u64m1, less_b64, vector_length);
    }

    vuint64m1_t identity_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    vuint64m1_t reduced_u64m1 = __riscv_vredminu_vs_u64m1_u64m1(best_u64m1, identity_u64m1, vector_length_max);
    nk_u64_t best = (nk_u64_t)__riscv_vmv_x_s_u64m1_u64(reduced_u64m1);

    vbool64_t match_b64 = __riscv_vmseq_vx_u64m1_b64(best_u64m1, best, vector_length_max);
    vuint64m1_t sentinel_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, vector_length_max);
    vuint64m1_t candidates_u64m1 = __riscv_vmerge_vvm_u64m1(sentinel_u64m1, best_index_u64m1, match_b64,
                                                            vector_length_max);
    vuint64m1_t idx_identity_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    vuint64m1_t idx_reduced_u64m1 = __riscv_vredminu_vs_u64m1_u64m1(candidates_u64m1, idx_identity_u64m1,
                                                                    vector_length_max);

    *min_value = best;
    *min_index = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(idx_reduced_u64m1);
}

NK_INTERNAL void nk_reduce_min_u64_rvv_strided_(                   //
    nk_u64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *min_value, nk_size_t *min_index) {

    nk_size_t vector_length_max = __riscv_vsetvlmax_e64m1();
    vuint64m1_t best_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, vector_length_max);
    vuint64m1_t best_index_u64m1 = __riscv_vmv_v_x_u64m1(0, vector_length_max);

    unsigned char const *ptr = (unsigned char const *)data;
    nk_size_t offset = 0;
    for (nk_size_t remaining = count, vector_length; remaining > 0;
         remaining -= vector_length, offset += vector_length, ptr += vector_length * stride_bytes) {
        vector_length = __riscv_vsetvl_e64m1(remaining);
        vuint64m1_t chunk_u64m1 = __riscv_vlse64_v_u64m1((nk_u64_t const *)ptr, (nk_ssize_t)stride_bytes,
                                                         vector_length);

        vbool64_t less_b64 = __riscv_vmsltu_vv_u64m1_b64(chunk_u64m1, best_u64m1, vector_length);
        best_u64m1 = __riscv_vmerge_vvm_u64m1(best_u64m1, chunk_u64m1, less_b64, vector_length);

        vuint64m1_t position_u64m1 = __riscv_vid_v_u64m1(vector_length);
        position_u64m1 = __riscv_vadd_vx_u64m1(position_u64m1, (nk_u64_t)offset, vector_length);
        best_index_u64m1 = __riscv_vmerge_vvm_u64m1(best_index_u64m1, position_u64m1, less_b64, vector_length);
    }

    vuint64m1_t identity_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    vuint64m1_t reduced_u64m1 = __riscv_vredminu_vs_u64m1_u64m1(best_u64m1, identity_u64m1, vector_length_max);
    nk_u64_t best = (nk_u64_t)__riscv_vmv_x_s_u64m1_u64(reduced_u64m1);

    vbool64_t match_b64 = __riscv_vmseq_vx_u64m1_b64(best_u64m1, best, vector_length_max);
    vuint64m1_t sentinel_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, vector_length_max);
    vuint64m1_t candidates_u64m1 = __riscv_vmerge_vvm_u64m1(sentinel_u64m1, best_index_u64m1, match_b64,
                                                            vector_length_max);
    vuint64m1_t idx_identity_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    vuint64m1_t idx_reduced_u64m1 = __riscv_vredminu_vs_u64m1_u64m1(candidates_u64m1, idx_identity_u64m1,
                                                                    vector_length_max);

    *min_value = best;
    *min_index = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(idx_reduced_u64m1);
}

NK_PUBLIC void nk_reduce_min_u64_rvv(                              //
    nk_u64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *min_value, nk_size_t *min_index) {
    if (count == 0) *min_value = NK_U64_MAX, *min_index = 0;
    else if (stride_bytes == sizeof(nk_u64_t)) nk_reduce_min_u64_rvv_contiguous_(data, count, min_value, min_index);
    else nk_reduce_min_u64_rvv_strided_(data, count, stride_bytes, min_value, min_index);
}

NK_INTERNAL void nk_reduce_max_u64_rvv_contiguous_( //
    nk_u64_t const *data, nk_size_t count,          //
    nk_u64_t *max_value, nk_size_t *max_index) {

    nk_size_t vector_length_max = __riscv_vsetvlmax_e64m1();
    vuint64m1_t best_u64m1 = __riscv_vmv_v_x_u64m1(0, vector_length_max);
    vuint64m1_t best_index_u64m1 = __riscv_vmv_v_x_u64m1(0, vector_length_max);

    nk_size_t offset = 0;
    for (nk_size_t remaining = count, vector_length; remaining > 0;
         remaining -= vector_length, offset += vector_length) {
        vector_length = __riscv_vsetvl_e64m1(remaining);
        vuint64m1_t chunk_u64m1 = __riscv_vle64_v_u64m1(data + offset, vector_length);

        vbool64_t greater_b64 = __riscv_vmsltu_vv_u64m1_b64(best_u64m1, chunk_u64m1, vector_length);
        best_u64m1 = __riscv_vmerge_vvm_u64m1(best_u64m1, chunk_u64m1, greater_b64, vector_length);

        vuint64m1_t position_u64m1 = __riscv_vid_v_u64m1(vector_length);
        position_u64m1 = __riscv_vadd_vx_u64m1(position_u64m1, (nk_u64_t)offset, vector_length);
        best_index_u64m1 = __riscv_vmerge_vvm_u64m1(best_index_u64m1, position_u64m1, greater_b64, vector_length);
    }

    vuint64m1_t identity_u64m1 = __riscv_vmv_v_x_u64m1(0, 1);
    vuint64m1_t reduced_u64m1 = __riscv_vredmaxu_vs_u64m1_u64m1(best_u64m1, identity_u64m1, vector_length_max);
    nk_u64_t best = (nk_u64_t)__riscv_vmv_x_s_u64m1_u64(reduced_u64m1);

    vbool64_t match_b64 = __riscv_vmseq_vx_u64m1_b64(best_u64m1, best, vector_length_max);
    vuint64m1_t sentinel_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, vector_length_max);
    vuint64m1_t candidates_u64m1 = __riscv_vmerge_vvm_u64m1(sentinel_u64m1, best_index_u64m1, match_b64,
                                                            vector_length_max);
    vuint64m1_t idx_identity_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    vuint64m1_t idx_reduced_u64m1 = __riscv_vredminu_vs_u64m1_u64m1(candidates_u64m1, idx_identity_u64m1,
                                                                    vector_length_max);

    *max_value = best;
    *max_index = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(idx_reduced_u64m1);
}

NK_INTERNAL void nk_reduce_max_u64_rvv_strided_(                   //
    nk_u64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *max_value, nk_size_t *max_index) {

    nk_size_t vector_length_max = __riscv_vsetvlmax_e64m1();
    vuint64m1_t best_u64m1 = __riscv_vmv_v_x_u64m1(0, vector_length_max);
    vuint64m1_t best_index_u64m1 = __riscv_vmv_v_x_u64m1(0, vector_length_max);

    unsigned char const *ptr = (unsigned char const *)data;
    nk_size_t offset = 0;
    for (nk_size_t remaining = count, vector_length; remaining > 0;
         remaining -= vector_length, offset += vector_length, ptr += vector_length * stride_bytes) {
        vector_length = __riscv_vsetvl_e64m1(remaining);
        vuint64m1_t chunk_u64m1 = __riscv_vlse64_v_u64m1((nk_u64_t const *)ptr, (nk_ssize_t)stride_bytes,
                                                         vector_length);

        vbool64_t greater_b64 = __riscv_vmsltu_vv_u64m1_b64(best_u64m1, chunk_u64m1, vector_length);
        best_u64m1 = __riscv_vmerge_vvm_u64m1(best_u64m1, chunk_u64m1, greater_b64, vector_length);

        vuint64m1_t position_u64m1 = __riscv_vid_v_u64m1(vector_length);
        position_u64m1 = __riscv_vadd_vx_u64m1(position_u64m1, (nk_u64_t)offset, vector_length);
        best_index_u64m1 = __riscv_vmerge_vvm_u64m1(best_index_u64m1, position_u64m1, greater_b64, vector_length);
    }

    vuint64m1_t identity_u64m1 = __riscv_vmv_v_x_u64m1(0, 1);
    vuint64m1_t reduced_u64m1 = __riscv_vredmaxu_vs_u64m1_u64m1(best_u64m1, identity_u64m1, vector_length_max);
    nk_u64_t best = (nk_u64_t)__riscv_vmv_x_s_u64m1_u64(reduced_u64m1);

    vbool64_t match_b64 = __riscv_vmseq_vx_u64m1_b64(best_u64m1, best, vector_length_max);
    vuint64m1_t sentinel_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, vector_length_max);
    vuint64m1_t candidates_u64m1 = __riscv_vmerge_vvm_u64m1(sentinel_u64m1, best_index_u64m1, match_b64,
                                                            vector_length_max);
    vuint64m1_t idx_identity_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    vuint64m1_t idx_reduced_u64m1 = __riscv_vredminu_vs_u64m1_u64m1(candidates_u64m1, idx_identity_u64m1,
                                                                    vector_length_max);

    *max_value = best;
    *max_index = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(idx_reduced_u64m1);
}

NK_PUBLIC void nk_reduce_max_u64_rvv(                              //
    nk_u64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *max_value, nk_size_t *max_index) {
    if (count == 0) *max_value = 0, *max_index = 0;
    else if (stride_bytes == sizeof(nk_u64_t)) nk_reduce_max_u64_rvv_contiguous_(data, count, max_value, max_index);
    else nk_reduce_max_u64_rvv_strided_(data, count, stride_bytes, max_value, max_index);
}

/** @brief Convert FP8 bytes to unsigned-comparable form for ordering (RVV version). */
NK_INTERNAL vuint8m1_t nk_fp8_to_comparable_u8m1_rvv_(vuint8m1_t raw, nk_size_t vl) {
    vbool8_t is_neg = __riscv_vmslt_vx_i8m1_b8(__riscv_vreinterpret_v_u8m1_i8m1(raw), 0, vl);
    vuint8m1_t pos_xor = __riscv_vxor_vx_u8m1(raw, 0x80, vl);
    vuint8m1_t neg_xor = __riscv_vxor_vx_u8m1(raw, 0xFF, vl);
    return __riscv_vmerge_vvm_u8m1(pos_xor, neg_xor, is_neg, vl);
}

/** @brief Convert unsigned-comparable form back to FP8 bytes (RVV version). */
NK_INTERNAL vuint8m1_t nk_comparable_u8m1_to_fp8_rvv_(vuint8m1_t cmp, nk_size_t vl) {
    vbool8_t was_neg = __riscv_vmsltu_vx_u8m1_b8(cmp, 0x80, vl);
    vuint8m1_t pos_xor = __riscv_vxor_vx_u8m1(cmp, 0x80, vl);
    vuint8m1_t neg_xor = __riscv_vxor_vx_u8m1(cmp, 0xFF, vl);
    return __riscv_vmerge_vvm_u8m1(pos_xor, neg_xor, was_neg, vl);
}

/** @brief Convert FP6 bytes to unsigned-comparable form for ordering (RVV version). */
NK_INTERNAL vuint8m1_t nk_fp6_to_comparable_u8m1_rvv_(vuint8m1_t raw, nk_size_t vl) {
    vuint8m1_t masked = __riscv_vand_vx_u8m1(raw, 0x3F, vl);
    vbool8_t is_neg = __riscv_vmsne_vx_u8m1_b8(__riscv_vand_vx_u8m1(masked, 0x20, vl), 0, vl);
    vuint8m1_t pos_xor = __riscv_vxor_vx_u8m1(masked, 0x20, vl);
    vuint8m1_t neg_xor = __riscv_vxor_vx_u8m1(masked, 0x3F, vl);
    return __riscv_vmerge_vvm_u8m1(pos_xor, neg_xor, is_neg, vl);
}

/** @brief Convert unsigned-comparable form back to FP6 bytes (RVV version). */
NK_INTERNAL vuint8m1_t nk_comparable_u8m1_to_fp6_rvv_(vuint8m1_t cmp, nk_size_t vl) {
    vbool8_t was_neg = __riscv_vmsltu_vx_u8m1_b8(cmp, 0x20, vl);
    vuint8m1_t pos_xor = __riscv_vxor_vx_u8m1(cmp, 0x20, vl);
    vuint8m1_t neg_xor = __riscv_vxor_vx_u8m1(cmp, 0x3F, vl);
    return __riscv_vmerge_vvm_u8m1(pos_xor, neg_xor, was_neg, vl);
}

NK_INTERNAL void nk_reduce_min_f16_rvv_contiguous_( //
    nk_f16_t const *data, nk_size_t count,          //
    nk_f32_t *min_value, nk_size_t *min_index) {

    nk_size_t vector_length_max = __riscv_vsetvlmax_e16m1();
    vfloat32m2_t best_f32m2 = __riscv_vfmv_v_f_f32m2(NK_F32_MAX, vector_length_max);
    vuint64m4_t best_index_u64m4 = __riscv_vmv_v_x_u64m4(0, vector_length_max);

    nk_size_t offset = 0;
    for (nk_size_t remaining = count, vector_length; remaining > 0;
         remaining -= vector_length, offset += vector_length) {
        vector_length = __riscv_vsetvl_e16m1(remaining);
        vuint16m1_t chunk_u16m1 = __riscv_vle16_v_u16m1((nk_u16_t const *)data + offset, vector_length);
        vfloat32m2_t chunk_f32m2 = nk_f16m1_to_f32m2_rvv_(chunk_u16m1, vector_length);

        vbool16_t less_b16 = __riscv_vmflt_vv_f32m2_b16(chunk_f32m2, best_f32m2, vector_length);
        best_f32m2 = __riscv_vmerge_vvm_f32m2(best_f32m2, chunk_f32m2, less_b16, vector_length);

        vuint64m4_t position_u64m4 = __riscv_vid_v_u64m4(vector_length);
        position_u64m4 = __riscv_vadd_vx_u64m4(position_u64m4, (nk_u64_t)offset, vector_length);
        best_index_u64m4 = __riscv_vmerge_vvm_u64m4(best_index_u64m4, position_u64m4, less_b16, vector_length);
    }

    vfloat32m1_t identity_f32m1 = __riscv_vfmv_v_f_f32m1(NK_F32_MAX, 1);
    vfloat32m1_t reduced_f32m1 = __riscv_vfredmin_vs_f32m2_f32m1(best_f32m2, identity_f32m1, vector_length_max);
    nk_f32_t best = __riscv_vfmv_f_s_f32m1_f32(reduced_f32m1);

    vbool16_t match_b16 = __riscv_vmfeq_vf_f32m2_b16(best_f32m2, best, vector_length_max);
    vuint64m4_t sentinel_u64m4 = __riscv_vmv_v_x_u64m4(NK_U64_MAX, vector_length_max);
    vuint64m4_t candidates_u64m4 = __riscv_vmerge_vvm_u64m4(sentinel_u64m4, best_index_u64m4, match_b16,
                                                            vector_length_max);
    vuint64m1_t identity_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    vuint64m1_t reduced_u64m1 = __riscv_vredminu_vs_u64m4_u64m1(candidates_u64m4, identity_u64m1, vector_length_max);

    *min_value = best;
    *min_index = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(reduced_u64m1);
}

NK_INTERNAL void nk_reduce_min_f16_rvv_strided_(                   //
    nk_f16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *min_value, nk_size_t *min_index) {

    nk_size_t vector_length_max = __riscv_vsetvlmax_e16m1();
    vfloat32m2_t best_f32m2 = __riscv_vfmv_v_f_f32m2(NK_F32_MAX, vector_length_max);
    vuint64m4_t best_index_u64m4 = __riscv_vmv_v_x_u64m4(0, vector_length_max);

    unsigned char const *ptr = (unsigned char const *)data;
    nk_size_t offset = 0;
    for (nk_size_t remaining = count, vector_length; remaining > 0;
         remaining -= vector_length, offset += vector_length, ptr += vector_length * stride_bytes) {
        vector_length = __riscv_vsetvl_e16m1(remaining);
        vuint16m1_t chunk_u16m1 = __riscv_vlse16_v_u16m1((nk_u16_t const *)ptr, (nk_ssize_t)stride_bytes,
                                                         vector_length);
        vfloat32m2_t chunk_f32m2 = nk_f16m1_to_f32m2_rvv_(chunk_u16m1, vector_length);

        vbool16_t less_b16 = __riscv_vmflt_vv_f32m2_b16(chunk_f32m2, best_f32m2, vector_length);
        best_f32m2 = __riscv_vmerge_vvm_f32m2(best_f32m2, chunk_f32m2, less_b16, vector_length);

        vuint64m4_t position_u64m4 = __riscv_vid_v_u64m4(vector_length);
        position_u64m4 = __riscv_vadd_vx_u64m4(position_u64m4, (nk_u64_t)offset, vector_length);
        best_index_u64m4 = __riscv_vmerge_vvm_u64m4(best_index_u64m4, position_u64m4, less_b16, vector_length);
    }

    vfloat32m1_t identity_f32m1 = __riscv_vfmv_v_f_f32m1(NK_F32_MAX, 1);
    vfloat32m1_t reduced_f32m1 = __riscv_vfredmin_vs_f32m2_f32m1(best_f32m2, identity_f32m1, vector_length_max);
    nk_f32_t best = __riscv_vfmv_f_s_f32m1_f32(reduced_f32m1);

    vbool16_t match_b16 = __riscv_vmfeq_vf_f32m2_b16(best_f32m2, best, vector_length_max);
    vuint64m4_t sentinel_u64m4 = __riscv_vmv_v_x_u64m4(NK_U64_MAX, vector_length_max);
    vuint64m4_t candidates_u64m4 = __riscv_vmerge_vvm_u64m4(sentinel_u64m4, best_index_u64m4, match_b16,
                                                            vector_length_max);
    vuint64m1_t identity_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    vuint64m1_t reduced_u64m1 = __riscv_vredminu_vs_u64m4_u64m1(candidates_u64m4, identity_u64m1, vector_length_max);

    *min_value = best;
    *min_index = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(reduced_u64m1);
}

NK_INTERNAL void nk_reduce_max_f16_rvv_contiguous_( //
    nk_f16_t const *data, nk_size_t count,          //
    nk_f32_t *max_value, nk_size_t *max_index) {

    nk_size_t vector_length_max = __riscv_vsetvlmax_e16m1();
    vfloat32m2_t best_f32m2 = __riscv_vfmv_v_f_f32m2(NK_F32_MIN, vector_length_max);
    vuint64m4_t best_index_u64m4 = __riscv_vmv_v_x_u64m4(0, vector_length_max);

    nk_size_t offset = 0;
    for (nk_size_t remaining = count, vector_length; remaining > 0;
         remaining -= vector_length, offset += vector_length) {
        vector_length = __riscv_vsetvl_e16m1(remaining);
        vuint16m1_t chunk_u16m1 = __riscv_vle16_v_u16m1((nk_u16_t const *)data + offset, vector_length);
        vfloat32m2_t chunk_f32m2 = nk_f16m1_to_f32m2_rvv_(chunk_u16m1, vector_length);

        vbool16_t greater_b16 = __riscv_vmflt_vv_f32m2_b16(best_f32m2, chunk_f32m2, vector_length);
        best_f32m2 = __riscv_vmerge_vvm_f32m2(best_f32m2, chunk_f32m2, greater_b16, vector_length);

        vuint64m4_t position_u64m4 = __riscv_vid_v_u64m4(vector_length);
        position_u64m4 = __riscv_vadd_vx_u64m4(position_u64m4, (nk_u64_t)offset, vector_length);
        best_index_u64m4 = __riscv_vmerge_vvm_u64m4(best_index_u64m4, position_u64m4, greater_b16, vector_length);
    }

    vfloat32m1_t identity_f32m1 = __riscv_vfmv_v_f_f32m1(NK_F32_MIN, 1);
    vfloat32m1_t reduced_f32m1 = __riscv_vfredmax_vs_f32m2_f32m1(best_f32m2, identity_f32m1, vector_length_max);
    nk_f32_t best = __riscv_vfmv_f_s_f32m1_f32(reduced_f32m1);

    vbool16_t match_b16 = __riscv_vmfeq_vf_f32m2_b16(best_f32m2, best, vector_length_max);
    vuint64m4_t sentinel_u64m4 = __riscv_vmv_v_x_u64m4(NK_U64_MAX, vector_length_max);
    vuint64m4_t candidates_u64m4 = __riscv_vmerge_vvm_u64m4(sentinel_u64m4, best_index_u64m4, match_b16,
                                                            vector_length_max);
    vuint64m1_t identity_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    vuint64m1_t reduced_u64m1 = __riscv_vredminu_vs_u64m4_u64m1(candidates_u64m4, identity_u64m1, vector_length_max);

    *max_value = best;
    *max_index = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(reduced_u64m1);
}

NK_INTERNAL void nk_reduce_max_f16_rvv_strided_(                   //
    nk_f16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *max_value, nk_size_t *max_index) {

    nk_size_t vector_length_max = __riscv_vsetvlmax_e16m1();
    vfloat32m2_t best_f32m2 = __riscv_vfmv_v_f_f32m2(NK_F32_MIN, vector_length_max);
    vuint64m4_t best_index_u64m4 = __riscv_vmv_v_x_u64m4(0, vector_length_max);

    unsigned char const *ptr = (unsigned char const *)data;
    nk_size_t offset = 0;
    for (nk_size_t remaining = count, vector_length; remaining > 0;
         remaining -= vector_length, offset += vector_length, ptr += vector_length * stride_bytes) {
        vector_length = __riscv_vsetvl_e16m1(remaining);
        vuint16m1_t chunk_u16m1 = __riscv_vlse16_v_u16m1((nk_u16_t const *)ptr, (nk_ssize_t)stride_bytes,
                                                         vector_length);
        vfloat32m2_t chunk_f32m2 = nk_f16m1_to_f32m2_rvv_(chunk_u16m1, vector_length);

        vbool16_t greater_b16 = __riscv_vmflt_vv_f32m2_b16(best_f32m2, chunk_f32m2, vector_length);
        best_f32m2 = __riscv_vmerge_vvm_f32m2(best_f32m2, chunk_f32m2, greater_b16, vector_length);

        vuint64m4_t position_u64m4 = __riscv_vid_v_u64m4(vector_length);
        position_u64m4 = __riscv_vadd_vx_u64m4(position_u64m4, (nk_u64_t)offset, vector_length);
        best_index_u64m4 = __riscv_vmerge_vvm_u64m4(best_index_u64m4, position_u64m4, greater_b16, vector_length);
    }

    vfloat32m1_t identity_f32m1 = __riscv_vfmv_v_f_f32m1(NK_F32_MIN, 1);
    vfloat32m1_t reduced_f32m1 = __riscv_vfredmax_vs_f32m2_f32m1(best_f32m2, identity_f32m1, vector_length_max);
    nk_f32_t best = __riscv_vfmv_f_s_f32m1_f32(reduced_f32m1);

    vbool16_t match_b16 = __riscv_vmfeq_vf_f32m2_b16(best_f32m2, best, vector_length_max);
    vuint64m4_t sentinel_u64m4 = __riscv_vmv_v_x_u64m4(NK_U64_MAX, vector_length_max);
    vuint64m4_t candidates_u64m4 = __riscv_vmerge_vvm_u64m4(sentinel_u64m4, best_index_u64m4, match_b16,
                                                            vector_length_max);
    vuint64m1_t identity_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    vuint64m1_t reduced_u64m1 = __riscv_vredminu_vs_u64m4_u64m1(candidates_u64m4, identity_u64m1, vector_length_max);

    *max_value = best;
    *max_index = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(reduced_u64m1);
}

NK_PUBLIC void nk_reduce_min_f16_rvv(nk_f16_t const *data, nk_size_t count, nk_size_t stride_bytes, nk_f32_t *min_value,
                                     nk_size_t *min_index) {
    if (count == 0) *min_value = NK_F32_MAX, *min_index = 0;
    else if (stride_bytes == sizeof(nk_f16_t)) nk_reduce_min_f16_rvv_contiguous_(data, count, min_value, min_index);
    else nk_reduce_min_f16_rvv_strided_(data, count, stride_bytes, min_value, min_index);
}

NK_PUBLIC void nk_reduce_max_f16_rvv(nk_f16_t const *data, nk_size_t count, nk_size_t stride_bytes, nk_f32_t *max_value,
                                     nk_size_t *max_index) {
    if (count == 0) *max_value = NK_F32_MIN, *max_index = 0;
    else if (stride_bytes == sizeof(nk_f16_t)) nk_reduce_max_f16_rvv_contiguous_(data, count, max_value, max_index);
    else nk_reduce_max_f16_rvv_strided_(data, count, stride_bytes, max_value, max_index);
}

NK_INTERNAL void nk_reduce_min_bf16_rvv_contiguous_( //
    nk_bf16_t const *data, nk_size_t count,          //
    nk_f32_t *min_value, nk_size_t *min_index) {

    nk_size_t vector_length_max = __riscv_vsetvlmax_e16m1();
    vfloat32m2_t best_f32m2 = __riscv_vfmv_v_f_f32m2(NK_F32_MAX, vector_length_max);
    vuint64m4_t best_index_u64m4 = __riscv_vmv_v_x_u64m4(0, vector_length_max);

    nk_size_t offset = 0;
    for (nk_size_t remaining = count, vector_length; remaining > 0;
         remaining -= vector_length, offset += vector_length) {
        vector_length = __riscv_vsetvl_e16m1(remaining);
        vuint16m1_t chunk_u16m1 = __riscv_vle16_v_u16m1((nk_u16_t const *)data + offset, vector_length);
        vfloat32m2_t chunk_f32m2 = nk_bf16m1_to_f32m2_rvv_(chunk_u16m1, vector_length);

        vbool16_t less_b16 = __riscv_vmflt_vv_f32m2_b16(chunk_f32m2, best_f32m2, vector_length);
        best_f32m2 = __riscv_vmerge_vvm_f32m2(best_f32m2, chunk_f32m2, less_b16, vector_length);

        vuint64m4_t position_u64m4 = __riscv_vid_v_u64m4(vector_length);
        position_u64m4 = __riscv_vadd_vx_u64m4(position_u64m4, (nk_u64_t)offset, vector_length);
        best_index_u64m4 = __riscv_vmerge_vvm_u64m4(best_index_u64m4, position_u64m4, less_b16, vector_length);
    }

    vfloat32m1_t identity_f32m1 = __riscv_vfmv_v_f_f32m1(NK_F32_MAX, 1);
    vfloat32m1_t reduced_f32m1 = __riscv_vfredmin_vs_f32m2_f32m1(best_f32m2, identity_f32m1, vector_length_max);
    nk_f32_t best = __riscv_vfmv_f_s_f32m1_f32(reduced_f32m1);

    vbool16_t match_b16 = __riscv_vmfeq_vf_f32m2_b16(best_f32m2, best, vector_length_max);
    vuint64m4_t sentinel_u64m4 = __riscv_vmv_v_x_u64m4(NK_U64_MAX, vector_length_max);
    vuint64m4_t candidates_u64m4 = __riscv_vmerge_vvm_u64m4(sentinel_u64m4, best_index_u64m4, match_b16,
                                                            vector_length_max);
    vuint64m1_t identity_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    vuint64m1_t reduced_u64m1 = __riscv_vredminu_vs_u64m4_u64m1(candidates_u64m4, identity_u64m1, vector_length_max);

    *min_value = best;
    *min_index = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(reduced_u64m1);
}

NK_INTERNAL void nk_reduce_min_bf16_rvv_strided_(                   //
    nk_bf16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *min_value, nk_size_t *min_index) {

    nk_size_t vector_length_max = __riscv_vsetvlmax_e16m1();
    vfloat32m2_t best_f32m2 = __riscv_vfmv_v_f_f32m2(NK_F32_MAX, vector_length_max);
    vuint64m4_t best_index_u64m4 = __riscv_vmv_v_x_u64m4(0, vector_length_max);

    unsigned char const *ptr = (unsigned char const *)data;
    nk_size_t offset = 0;
    for (nk_size_t remaining = count, vector_length; remaining > 0;
         remaining -= vector_length, offset += vector_length, ptr += vector_length * stride_bytes) {
        vector_length = __riscv_vsetvl_e16m1(remaining);
        vuint16m1_t chunk_u16m1 = __riscv_vlse16_v_u16m1((nk_u16_t const *)ptr, (nk_ssize_t)stride_bytes,
                                                         vector_length);
        vfloat32m2_t chunk_f32m2 = nk_bf16m1_to_f32m2_rvv_(chunk_u16m1, vector_length);

        vbool16_t less_b16 = __riscv_vmflt_vv_f32m2_b16(chunk_f32m2, best_f32m2, vector_length);
        best_f32m2 = __riscv_vmerge_vvm_f32m2(best_f32m2, chunk_f32m2, less_b16, vector_length);

        vuint64m4_t position_u64m4 = __riscv_vid_v_u64m4(vector_length);
        position_u64m4 = __riscv_vadd_vx_u64m4(position_u64m4, (nk_u64_t)offset, vector_length);
        best_index_u64m4 = __riscv_vmerge_vvm_u64m4(best_index_u64m4, position_u64m4, less_b16, vector_length);
    }

    vfloat32m1_t identity_f32m1 = __riscv_vfmv_v_f_f32m1(NK_F32_MAX, 1);
    vfloat32m1_t reduced_f32m1 = __riscv_vfredmin_vs_f32m2_f32m1(best_f32m2, identity_f32m1, vector_length_max);
    nk_f32_t best = __riscv_vfmv_f_s_f32m1_f32(reduced_f32m1);

    vbool16_t match_b16 = __riscv_vmfeq_vf_f32m2_b16(best_f32m2, best, vector_length_max);
    vuint64m4_t sentinel_u64m4 = __riscv_vmv_v_x_u64m4(NK_U64_MAX, vector_length_max);
    vuint64m4_t candidates_u64m4 = __riscv_vmerge_vvm_u64m4(sentinel_u64m4, best_index_u64m4, match_b16,
                                                            vector_length_max);
    vuint64m1_t identity_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    vuint64m1_t reduced_u64m1 = __riscv_vredminu_vs_u64m4_u64m1(candidates_u64m4, identity_u64m1, vector_length_max);

    *min_value = best;
    *min_index = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(reduced_u64m1);
}

NK_INTERNAL void nk_reduce_max_bf16_rvv_contiguous_( //
    nk_bf16_t const *data, nk_size_t count,          //
    nk_f32_t *max_value, nk_size_t *max_index) {

    nk_size_t vector_length_max = __riscv_vsetvlmax_e16m1();
    vfloat32m2_t best_f32m2 = __riscv_vfmv_v_f_f32m2(NK_F32_MIN, vector_length_max);
    vuint64m4_t best_index_u64m4 = __riscv_vmv_v_x_u64m4(0, vector_length_max);

    nk_size_t offset = 0;
    for (nk_size_t remaining = count, vector_length; remaining > 0;
         remaining -= vector_length, offset += vector_length) {
        vector_length = __riscv_vsetvl_e16m1(remaining);
        vuint16m1_t chunk_u16m1 = __riscv_vle16_v_u16m1((nk_u16_t const *)data + offset, vector_length);
        vfloat32m2_t chunk_f32m2 = nk_bf16m1_to_f32m2_rvv_(chunk_u16m1, vector_length);

        vbool16_t greater_b16 = __riscv_vmflt_vv_f32m2_b16(best_f32m2, chunk_f32m2, vector_length);
        best_f32m2 = __riscv_vmerge_vvm_f32m2(best_f32m2, chunk_f32m2, greater_b16, vector_length);

        vuint64m4_t position_u64m4 = __riscv_vid_v_u64m4(vector_length);
        position_u64m4 = __riscv_vadd_vx_u64m4(position_u64m4, (nk_u64_t)offset, vector_length);
        best_index_u64m4 = __riscv_vmerge_vvm_u64m4(best_index_u64m4, position_u64m4, greater_b16, vector_length);
    }

    vfloat32m1_t identity_f32m1 = __riscv_vfmv_v_f_f32m1(NK_F32_MIN, 1);
    vfloat32m1_t reduced_f32m1 = __riscv_vfredmax_vs_f32m2_f32m1(best_f32m2, identity_f32m1, vector_length_max);
    nk_f32_t best = __riscv_vfmv_f_s_f32m1_f32(reduced_f32m1);

    vbool16_t match_b16 = __riscv_vmfeq_vf_f32m2_b16(best_f32m2, best, vector_length_max);
    vuint64m4_t sentinel_u64m4 = __riscv_vmv_v_x_u64m4(NK_U64_MAX, vector_length_max);
    vuint64m4_t candidates_u64m4 = __riscv_vmerge_vvm_u64m4(sentinel_u64m4, best_index_u64m4, match_b16,
                                                            vector_length_max);
    vuint64m1_t identity_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    vuint64m1_t reduced_u64m1 = __riscv_vredminu_vs_u64m4_u64m1(candidates_u64m4, identity_u64m1, vector_length_max);

    *max_value = best;
    *max_index = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(reduced_u64m1);
}

NK_INTERNAL void nk_reduce_max_bf16_rvv_strided_(                   //
    nk_bf16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *max_value, nk_size_t *max_index) {

    nk_size_t vector_length_max = __riscv_vsetvlmax_e16m1();
    vfloat32m2_t best_f32m2 = __riscv_vfmv_v_f_f32m2(NK_F32_MIN, vector_length_max);
    vuint64m4_t best_index_u64m4 = __riscv_vmv_v_x_u64m4(0, vector_length_max);

    unsigned char const *ptr = (unsigned char const *)data;
    nk_size_t offset = 0;
    for (nk_size_t remaining = count, vector_length; remaining > 0;
         remaining -= vector_length, offset += vector_length, ptr += vector_length * stride_bytes) {
        vector_length = __riscv_vsetvl_e16m1(remaining);
        vuint16m1_t chunk_u16m1 = __riscv_vlse16_v_u16m1((nk_u16_t const *)ptr, (nk_ssize_t)stride_bytes,
                                                         vector_length);
        vfloat32m2_t chunk_f32m2 = nk_bf16m1_to_f32m2_rvv_(chunk_u16m1, vector_length);

        vbool16_t greater_b16 = __riscv_vmflt_vv_f32m2_b16(best_f32m2, chunk_f32m2, vector_length);
        best_f32m2 = __riscv_vmerge_vvm_f32m2(best_f32m2, chunk_f32m2, greater_b16, vector_length);

        vuint64m4_t position_u64m4 = __riscv_vid_v_u64m4(vector_length);
        position_u64m4 = __riscv_vadd_vx_u64m4(position_u64m4, (nk_u64_t)offset, vector_length);
        best_index_u64m4 = __riscv_vmerge_vvm_u64m4(best_index_u64m4, position_u64m4, greater_b16, vector_length);
    }

    vfloat32m1_t identity_f32m1 = __riscv_vfmv_v_f_f32m1(NK_F32_MIN, 1);
    vfloat32m1_t reduced_f32m1 = __riscv_vfredmax_vs_f32m2_f32m1(best_f32m2, identity_f32m1, vector_length_max);
    nk_f32_t best = __riscv_vfmv_f_s_f32m1_f32(reduced_f32m1);

    vbool16_t match_b16 = __riscv_vmfeq_vf_f32m2_b16(best_f32m2, best, vector_length_max);
    vuint64m4_t sentinel_u64m4 = __riscv_vmv_v_x_u64m4(NK_U64_MAX, vector_length_max);
    vuint64m4_t candidates_u64m4 = __riscv_vmerge_vvm_u64m4(sentinel_u64m4, best_index_u64m4, match_b16,
                                                            vector_length_max);
    vuint64m1_t identity_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    vuint64m1_t reduced_u64m1 = __riscv_vredminu_vs_u64m4_u64m1(candidates_u64m4, identity_u64m1, vector_length_max);

    *max_value = best;
    *max_index = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(reduced_u64m1);
}

NK_PUBLIC void nk_reduce_min_bf16_rvv(nk_bf16_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                      nk_f32_t *min_value, nk_size_t *min_index) {
    if (count == 0) *min_value = NK_F32_MAX, *min_index = 0;
    else if (stride_bytes == sizeof(nk_bf16_t)) nk_reduce_min_bf16_rvv_contiguous_(data, count, min_value, min_index);
    else nk_reduce_min_bf16_rvv_strided_(data, count, stride_bytes, min_value, min_index);
}

NK_PUBLIC void nk_reduce_max_bf16_rvv(nk_bf16_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                      nk_f32_t *max_value, nk_size_t *max_index) {
    if (count == 0) *max_value = NK_F32_MIN, *max_index = 0;
    else if (stride_bytes == sizeof(nk_bf16_t)) nk_reduce_max_bf16_rvv_contiguous_(data, count, max_value, max_index);
    else nk_reduce_max_bf16_rvv_strided_(data, count, stride_bytes, max_value, max_index);
}

NK_INTERNAL void nk_reduce_min_e4m3_rvv_contiguous_( //
    nk_e4m3_t const *data, nk_size_t count,          //
    nk_f32_t *min_value, nk_size_t *min_index) {

    nk_size_t vector_length_max = __riscv_vsetvlmax_e8m1();
    vuint8m1_t best_cmp_u8m1 = __riscv_vmv_v_x_u8m1(0xFF, vector_length_max);
    vuint64m8_t best_index_u64m8 = __riscv_vmv_v_x_u64m8(0, vector_length_max);

    nk_size_t offset = 0;
    for (nk_size_t remaining = count, vector_length; remaining > 0;
         remaining -= vector_length, offset += vector_length) {
        vector_length = __riscv_vsetvl_e8m1(remaining);
        vuint8m1_t chunk_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)data + offset, vector_length);
        vuint8m1_t chunk_cmp = nk_fp8_to_comparable_u8m1_rvv_(chunk_u8m1, vector_length);

        vbool8_t is_nan_pos = __riscv_vmseq_vx_u8m1_b8(chunk_u8m1, 0x7F, vector_length);
        vbool8_t is_nan_neg = __riscv_vmseq_vx_u8m1_b8(chunk_u8m1, 0xFF, vector_length);
        vbool8_t is_nan = __riscv_vmor_mm_b8(is_nan_pos, is_nan_neg, vector_length);
        chunk_cmp = __riscv_vmerge_vxm_u8m1(chunk_cmp, 0xFF, is_nan, vector_length);

        vbool8_t less_b8 = __riscv_vmsltu_vv_u8m1_b8(chunk_cmp, best_cmp_u8m1, vector_length);
        best_cmp_u8m1 = __riscv_vmerge_vvm_u8m1(best_cmp_u8m1, chunk_cmp, less_b8, vector_length);

        vuint64m8_t position_u64m8 = __riscv_vid_v_u64m8(vector_length);
        position_u64m8 = __riscv_vadd_vx_u64m8(position_u64m8, (nk_u64_t)offset, vector_length);
        best_index_u64m8 = __riscv_vmerge_vvm_u64m8(best_index_u64m8, position_u64m8, less_b8, vector_length);
    }

    vuint8m1_t identity_u8m1 = __riscv_vmv_v_x_u8m1(0xFF, 1);
    vuint8m1_t reduced_u8m1 = __riscv_vredminu_vs_u8m1_u8m1(best_cmp_u8m1, identity_u8m1, vector_length_max);
    nk_u8_t best_cmp = (nk_u8_t)__riscv_vmv_x_s_u8m1_u8(reduced_u8m1);

    vbool8_t match_b8 = __riscv_vmseq_vx_u8m1_b8(best_cmp_u8m1, best_cmp, vector_length_max);
    vuint64m8_t sentinel_u64m8 = __riscv_vmv_v_x_u64m8(NK_U64_MAX, vector_length_max);
    vuint64m8_t candidates_u64m8 = __riscv_vmerge_vvm_u64m8(sentinel_u64m8, best_index_u64m8, match_b8,
                                                            vector_length_max);
    vuint64m1_t identity_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    vuint64m1_t reduced_u64m1 = __riscv_vredminu_vs_u64m8_u64m1(candidates_u64m8, identity_u64m1, vector_length_max);

    *min_index = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(reduced_u64m1);

    nk_u8_t best_raw;
    if (best_cmp < 0x80) best_raw = best_cmp ^ 0xFF;
    else best_raw = best_cmp ^ 0x80;
    nk_e4m3_t best_e4m3;
    *(nk_u8_t *)&best_e4m3 = best_raw;
    nk_e4m3_to_f32(&best_e4m3, min_value);
}

NK_INTERNAL void nk_reduce_min_e4m3_rvv_strided_(                   //
    nk_e4m3_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *min_value, nk_size_t *min_index) {

    nk_size_t vector_length_max = __riscv_vsetvlmax_e8m1();
    vuint8m1_t best_cmp_u8m1 = __riscv_vmv_v_x_u8m1(0xFF, vector_length_max);
    vuint64m8_t best_index_u64m8 = __riscv_vmv_v_x_u64m8(0, vector_length_max);

    unsigned char const *ptr = (unsigned char const *)data;
    nk_size_t offset = 0;
    for (nk_size_t remaining = count, vector_length; remaining > 0;
         remaining -= vector_length, offset += vector_length, ptr += vector_length * stride_bytes) {
        vector_length = __riscv_vsetvl_e8m1(remaining);
        vuint8m1_t chunk_u8m1 = __riscv_vlse8_v_u8m1((nk_u8_t const *)ptr, (nk_ssize_t)stride_bytes, vector_length);
        vuint8m1_t chunk_cmp = nk_fp8_to_comparable_u8m1_rvv_(chunk_u8m1, vector_length);

        vbool8_t is_nan_pos = __riscv_vmseq_vx_u8m1_b8(chunk_u8m1, 0x7F, vector_length);
        vbool8_t is_nan_neg = __riscv_vmseq_vx_u8m1_b8(chunk_u8m1, 0xFF, vector_length);
        vbool8_t is_nan = __riscv_vmor_mm_b8(is_nan_pos, is_nan_neg, vector_length);
        chunk_cmp = __riscv_vmerge_vxm_u8m1(chunk_cmp, 0xFF, is_nan, vector_length);

        vbool8_t less_b8 = __riscv_vmsltu_vv_u8m1_b8(chunk_cmp, best_cmp_u8m1, vector_length);
        best_cmp_u8m1 = __riscv_vmerge_vvm_u8m1(best_cmp_u8m1, chunk_cmp, less_b8, vector_length);

        vuint64m8_t position_u64m8 = __riscv_vid_v_u64m8(vector_length);
        position_u64m8 = __riscv_vadd_vx_u64m8(position_u64m8, (nk_u64_t)offset, vector_length);
        best_index_u64m8 = __riscv_vmerge_vvm_u64m8(best_index_u64m8, position_u64m8, less_b8, vector_length);
    }

    vuint8m1_t identity_u8m1 = __riscv_vmv_v_x_u8m1(0xFF, 1);
    vuint8m1_t reduced_u8m1 = __riscv_vredminu_vs_u8m1_u8m1(best_cmp_u8m1, identity_u8m1, vector_length_max);
    nk_u8_t best_cmp = (nk_u8_t)__riscv_vmv_x_s_u8m1_u8(reduced_u8m1);

    vbool8_t match_b8 = __riscv_vmseq_vx_u8m1_b8(best_cmp_u8m1, best_cmp, vector_length_max);
    vuint64m8_t sentinel_u64m8 = __riscv_vmv_v_x_u64m8(NK_U64_MAX, vector_length_max);
    vuint64m8_t candidates_u64m8 = __riscv_vmerge_vvm_u64m8(sentinel_u64m8, best_index_u64m8, match_b8,
                                                            vector_length_max);
    vuint64m1_t identity_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    vuint64m1_t reduced_u64m1 = __riscv_vredminu_vs_u64m8_u64m1(candidates_u64m8, identity_u64m1, vector_length_max);

    *min_index = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(reduced_u64m1);

    nk_u8_t best_raw;
    if (best_cmp < 0x80) best_raw = best_cmp ^ 0xFF;
    else best_raw = best_cmp ^ 0x80;
    nk_e4m3_t best_e4m3;
    *(nk_u8_t *)&best_e4m3 = best_raw;
    nk_e4m3_to_f32(&best_e4m3, min_value);
}

NK_INTERNAL void nk_reduce_max_e4m3_rvv_contiguous_( //
    nk_e4m3_t const *data, nk_size_t count,          //
    nk_f32_t *max_value, nk_size_t *max_index) {

    nk_size_t vector_length_max = __riscv_vsetvlmax_e8m1();
    vuint8m1_t best_cmp_u8m1 = __riscv_vmv_v_x_u8m1(0x00, vector_length_max);
    vuint64m8_t best_index_u64m8 = __riscv_vmv_v_x_u64m8(0, vector_length_max);

    nk_size_t offset = 0;
    for (nk_size_t remaining = count, vector_length; remaining > 0;
         remaining -= vector_length, offset += vector_length) {
        vector_length = __riscv_vsetvl_e8m1(remaining);
        vuint8m1_t chunk_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)data + offset, vector_length);
        vuint8m1_t chunk_cmp = nk_fp8_to_comparable_u8m1_rvv_(chunk_u8m1, vector_length);

        vbool8_t is_nan_pos = __riscv_vmseq_vx_u8m1_b8(chunk_u8m1, 0x7F, vector_length);
        vbool8_t is_nan_neg = __riscv_vmseq_vx_u8m1_b8(chunk_u8m1, 0xFF, vector_length);
        vbool8_t is_nan = __riscv_vmor_mm_b8(is_nan_pos, is_nan_neg, vector_length);
        chunk_cmp = __riscv_vmerge_vxm_u8m1(chunk_cmp, 0x00, is_nan, vector_length);

        vbool8_t greater_b8 = __riscv_vmsltu_vv_u8m1_b8(best_cmp_u8m1, chunk_cmp, vector_length);
        best_cmp_u8m1 = __riscv_vmerge_vvm_u8m1(best_cmp_u8m1, chunk_cmp, greater_b8, vector_length);

        vuint64m8_t position_u64m8 = __riscv_vid_v_u64m8(vector_length);
        position_u64m8 = __riscv_vadd_vx_u64m8(position_u64m8, (nk_u64_t)offset, vector_length);
        best_index_u64m8 = __riscv_vmerge_vvm_u64m8(best_index_u64m8, position_u64m8, greater_b8, vector_length);
    }

    vuint8m1_t identity_u8m1 = __riscv_vmv_v_x_u8m1(0x00, 1);
    vuint8m1_t reduced_u8m1 = __riscv_vredmaxu_vs_u8m1_u8m1(best_cmp_u8m1, identity_u8m1, vector_length_max);
    nk_u8_t best_cmp = (nk_u8_t)__riscv_vmv_x_s_u8m1_u8(reduced_u8m1);

    vbool8_t match_b8 = __riscv_vmseq_vx_u8m1_b8(best_cmp_u8m1, best_cmp, vector_length_max);
    vuint64m8_t sentinel_u64m8 = __riscv_vmv_v_x_u64m8(NK_U64_MAX, vector_length_max);
    vuint64m8_t candidates_u64m8 = __riscv_vmerge_vvm_u64m8(sentinel_u64m8, best_index_u64m8, match_b8,
                                                            vector_length_max);
    vuint64m1_t identity_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    vuint64m1_t reduced_u64m1 = __riscv_vredminu_vs_u64m8_u64m1(candidates_u64m8, identity_u64m1, vector_length_max);

    *max_index = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(reduced_u64m1);

    nk_u8_t best_raw;
    if (best_cmp < 0x80) best_raw = best_cmp ^ 0xFF;
    else best_raw = best_cmp ^ 0x80;
    nk_e4m3_t best_e4m3;
    *(nk_u8_t *)&best_e4m3 = best_raw;
    nk_e4m3_to_f32(&best_e4m3, max_value);
}

NK_INTERNAL void nk_reduce_max_e4m3_rvv_strided_(                   //
    nk_e4m3_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *max_value, nk_size_t *max_index) {

    nk_size_t vector_length_max = __riscv_vsetvlmax_e8m1();
    vuint8m1_t best_cmp_u8m1 = __riscv_vmv_v_x_u8m1(0x00, vector_length_max);
    vuint64m8_t best_index_u64m8 = __riscv_vmv_v_x_u64m8(0, vector_length_max);

    unsigned char const *ptr = (unsigned char const *)data;
    nk_size_t offset = 0;
    for (nk_size_t remaining = count, vector_length; remaining > 0;
         remaining -= vector_length, offset += vector_length, ptr += vector_length * stride_bytes) {
        vector_length = __riscv_vsetvl_e8m1(remaining);
        vuint8m1_t chunk_u8m1 = __riscv_vlse8_v_u8m1((nk_u8_t const *)ptr, (nk_ssize_t)stride_bytes, vector_length);
        vuint8m1_t chunk_cmp = nk_fp8_to_comparable_u8m1_rvv_(chunk_u8m1, vector_length);

        vbool8_t is_nan_pos = __riscv_vmseq_vx_u8m1_b8(chunk_u8m1, 0x7F, vector_length);
        vbool8_t is_nan_neg = __riscv_vmseq_vx_u8m1_b8(chunk_u8m1, 0xFF, vector_length);
        vbool8_t is_nan = __riscv_vmor_mm_b8(is_nan_pos, is_nan_neg, vector_length);
        chunk_cmp = __riscv_vmerge_vxm_u8m1(chunk_cmp, 0x00, is_nan, vector_length);

        vbool8_t greater_b8 = __riscv_vmsltu_vv_u8m1_b8(best_cmp_u8m1, chunk_cmp, vector_length);
        best_cmp_u8m1 = __riscv_vmerge_vvm_u8m1(best_cmp_u8m1, chunk_cmp, greater_b8, vector_length);

        vuint64m8_t position_u64m8 = __riscv_vid_v_u64m8(vector_length);
        position_u64m8 = __riscv_vadd_vx_u64m8(position_u64m8, (nk_u64_t)offset, vector_length);
        best_index_u64m8 = __riscv_vmerge_vvm_u64m8(best_index_u64m8, position_u64m8, greater_b8, vector_length);
    }

    vuint8m1_t identity_u8m1 = __riscv_vmv_v_x_u8m1(0x00, 1);
    vuint8m1_t reduced_u8m1 = __riscv_vredmaxu_vs_u8m1_u8m1(best_cmp_u8m1, identity_u8m1, vector_length_max);
    nk_u8_t best_cmp = (nk_u8_t)__riscv_vmv_x_s_u8m1_u8(reduced_u8m1);

    vbool8_t match_b8 = __riscv_vmseq_vx_u8m1_b8(best_cmp_u8m1, best_cmp, vector_length_max);
    vuint64m8_t sentinel_u64m8 = __riscv_vmv_v_x_u64m8(NK_U64_MAX, vector_length_max);
    vuint64m8_t candidates_u64m8 = __riscv_vmerge_vvm_u64m8(sentinel_u64m8, best_index_u64m8, match_b8,
                                                            vector_length_max);
    vuint64m1_t identity_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    vuint64m1_t reduced_u64m1 = __riscv_vredminu_vs_u64m8_u64m1(candidates_u64m8, identity_u64m1, vector_length_max);

    *max_index = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(reduced_u64m1);

    nk_u8_t best_raw;
    if (best_cmp < 0x80) best_raw = best_cmp ^ 0xFF;
    else best_raw = best_cmp ^ 0x80;
    nk_e4m3_t best_e4m3;
    *(nk_u8_t *)&best_e4m3 = best_raw;
    nk_e4m3_to_f32(&best_e4m3, max_value);
}

NK_PUBLIC void nk_reduce_min_e4m3_rvv(nk_e4m3_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                      nk_f32_t *min_value, nk_size_t *min_index) {
    if (count == 0) *min_value = NK_F32_MAX, *min_index = 0;
    else if (stride_bytes == sizeof(nk_e4m3_t)) nk_reduce_min_e4m3_rvv_contiguous_(data, count, min_value, min_index);
    else nk_reduce_min_e4m3_rvv_strided_(data, count, stride_bytes, min_value, min_index);
}

NK_PUBLIC void nk_reduce_max_e4m3_rvv(nk_e4m3_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                      nk_f32_t *max_value, nk_size_t *max_index) {
    if (count == 0) *max_value = NK_F32_MAX, *max_index = 0;
    else if (stride_bytes == sizeof(nk_e4m3_t)) nk_reduce_max_e4m3_rvv_contiguous_(data, count, max_value, max_index);
    else nk_reduce_max_e4m3_rvv_strided_(data, count, stride_bytes, max_value, max_index);
}

NK_INTERNAL void nk_reduce_min_e5m2_rvv_contiguous_( //
    nk_e5m2_t const *data, nk_size_t count,          //
    nk_f32_t *min_value, nk_size_t *min_index) {

    nk_size_t vector_length_max = __riscv_vsetvlmax_e8m1();
    vuint8m1_t best_cmp_u8m1 = __riscv_vmv_v_x_u8m1(0xFF, vector_length_max);
    vuint64m8_t best_index_u64m8 = __riscv_vmv_v_x_u64m8(0, vector_length_max);

    nk_size_t offset = 0;
    for (nk_size_t remaining = count, vector_length; remaining > 0;
         remaining -= vector_length, offset += vector_length) {
        vector_length = __riscv_vsetvl_e8m1(remaining);
        vuint8m1_t chunk_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)data + offset, vector_length);
        vuint8m1_t chunk_cmp = nk_fp8_to_comparable_u8m1_rvv_(chunk_u8m1, vector_length);

        vuint8m1_t abs_val = __riscv_vand_vx_u8m1(chunk_u8m1, 0x7F, vector_length);
        vbool8_t is_nan = __riscv_vmsgeu_vx_u8m1_b8(abs_val, 0x7D, vector_length);
        chunk_cmp = __riscv_vmerge_vxm_u8m1(chunk_cmp, 0xFF, is_nan, vector_length);

        vbool8_t less_b8 = __riscv_vmsltu_vv_u8m1_b8(chunk_cmp, best_cmp_u8m1, vector_length);
        best_cmp_u8m1 = __riscv_vmerge_vvm_u8m1(best_cmp_u8m1, chunk_cmp, less_b8, vector_length);

        vuint64m8_t position_u64m8 = __riscv_vid_v_u64m8(vector_length);
        position_u64m8 = __riscv_vadd_vx_u64m8(position_u64m8, (nk_u64_t)offset, vector_length);
        best_index_u64m8 = __riscv_vmerge_vvm_u64m8(best_index_u64m8, position_u64m8, less_b8, vector_length);
    }

    vuint8m1_t identity_u8m1 = __riscv_vmv_v_x_u8m1(0xFF, 1);
    vuint8m1_t reduced_u8m1 = __riscv_vredminu_vs_u8m1_u8m1(best_cmp_u8m1, identity_u8m1, vector_length_max);
    nk_u8_t best_cmp = (nk_u8_t)__riscv_vmv_x_s_u8m1_u8(reduced_u8m1);

    vbool8_t match_b8 = __riscv_vmseq_vx_u8m1_b8(best_cmp_u8m1, best_cmp, vector_length_max);
    vuint64m8_t sentinel_u64m8 = __riscv_vmv_v_x_u64m8(NK_U64_MAX, vector_length_max);
    vuint64m8_t candidates_u64m8 = __riscv_vmerge_vvm_u64m8(sentinel_u64m8, best_index_u64m8, match_b8,
                                                            vector_length_max);
    vuint64m1_t identity_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    vuint64m1_t reduced_u64m1 = __riscv_vredminu_vs_u64m8_u64m1(candidates_u64m8, identity_u64m1, vector_length_max);

    *min_index = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(reduced_u64m1);

    nk_u8_t best_raw;
    if (best_cmp < 0x80) best_raw = best_cmp ^ 0xFF;
    else best_raw = best_cmp ^ 0x80;
    nk_e5m2_t best_e5m2;
    *(nk_u8_t *)&best_e5m2 = best_raw;
    nk_e5m2_to_f32(&best_e5m2, min_value);
}

NK_INTERNAL void nk_reduce_min_e5m2_rvv_strided_(                   //
    nk_e5m2_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *min_value, nk_size_t *min_index) {

    nk_size_t vector_length_max = __riscv_vsetvlmax_e8m1();
    vuint8m1_t best_cmp_u8m1 = __riscv_vmv_v_x_u8m1(0xFF, vector_length_max);
    vuint64m8_t best_index_u64m8 = __riscv_vmv_v_x_u64m8(0, vector_length_max);

    unsigned char const *ptr = (unsigned char const *)data;
    nk_size_t offset = 0;
    for (nk_size_t remaining = count, vector_length; remaining > 0;
         remaining -= vector_length, offset += vector_length, ptr += vector_length * stride_bytes) {
        vector_length = __riscv_vsetvl_e8m1(remaining);
        vuint8m1_t chunk_u8m1 = __riscv_vlse8_v_u8m1((nk_u8_t const *)ptr, (nk_ssize_t)stride_bytes, vector_length);
        vuint8m1_t chunk_cmp = nk_fp8_to_comparable_u8m1_rvv_(chunk_u8m1, vector_length);

        vuint8m1_t abs_val = __riscv_vand_vx_u8m1(chunk_u8m1, 0x7F, vector_length);
        vbool8_t is_nan = __riscv_vmsgeu_vx_u8m1_b8(abs_val, 0x7D, vector_length);
        chunk_cmp = __riscv_vmerge_vxm_u8m1(chunk_cmp, 0xFF, is_nan, vector_length);

        vbool8_t less_b8 = __riscv_vmsltu_vv_u8m1_b8(chunk_cmp, best_cmp_u8m1, vector_length);
        best_cmp_u8m1 = __riscv_vmerge_vvm_u8m1(best_cmp_u8m1, chunk_cmp, less_b8, vector_length);

        vuint64m8_t position_u64m8 = __riscv_vid_v_u64m8(vector_length);
        position_u64m8 = __riscv_vadd_vx_u64m8(position_u64m8, (nk_u64_t)offset, vector_length);
        best_index_u64m8 = __riscv_vmerge_vvm_u64m8(best_index_u64m8, position_u64m8, less_b8, vector_length);
    }

    vuint8m1_t identity_u8m1 = __riscv_vmv_v_x_u8m1(0xFF, 1);
    vuint8m1_t reduced_u8m1 = __riscv_vredminu_vs_u8m1_u8m1(best_cmp_u8m1, identity_u8m1, vector_length_max);
    nk_u8_t best_cmp = (nk_u8_t)__riscv_vmv_x_s_u8m1_u8(reduced_u8m1);

    vbool8_t match_b8 = __riscv_vmseq_vx_u8m1_b8(best_cmp_u8m1, best_cmp, vector_length_max);
    vuint64m8_t sentinel_u64m8 = __riscv_vmv_v_x_u64m8(NK_U64_MAX, vector_length_max);
    vuint64m8_t candidates_u64m8 = __riscv_vmerge_vvm_u64m8(sentinel_u64m8, best_index_u64m8, match_b8,
                                                            vector_length_max);
    vuint64m1_t identity_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    vuint64m1_t reduced_u64m1 = __riscv_vredminu_vs_u64m8_u64m1(candidates_u64m8, identity_u64m1, vector_length_max);

    *min_index = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(reduced_u64m1);

    nk_u8_t best_raw;
    if (best_cmp < 0x80) best_raw = best_cmp ^ 0xFF;
    else best_raw = best_cmp ^ 0x80;
    nk_e5m2_t best_e5m2;
    *(nk_u8_t *)&best_e5m2 = best_raw;
    nk_e5m2_to_f32(&best_e5m2, min_value);
}

NK_INTERNAL void nk_reduce_max_e5m2_rvv_contiguous_( //
    nk_e5m2_t const *data, nk_size_t count,          //
    nk_f32_t *max_value, nk_size_t *max_index) {

    nk_size_t vector_length_max = __riscv_vsetvlmax_e8m1();
    vuint8m1_t best_cmp_u8m1 = __riscv_vmv_v_x_u8m1(0x00, vector_length_max);
    vuint64m8_t best_index_u64m8 = __riscv_vmv_v_x_u64m8(0, vector_length_max);

    nk_size_t offset = 0;
    for (nk_size_t remaining = count, vector_length; remaining > 0;
         remaining -= vector_length, offset += vector_length) {
        vector_length = __riscv_vsetvl_e8m1(remaining);
        vuint8m1_t chunk_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)data + offset, vector_length);
        vuint8m1_t chunk_cmp = nk_fp8_to_comparable_u8m1_rvv_(chunk_u8m1, vector_length);

        vuint8m1_t abs_val = __riscv_vand_vx_u8m1(chunk_u8m1, 0x7F, vector_length);
        vbool8_t is_nan = __riscv_vmsgeu_vx_u8m1_b8(abs_val, 0x7D, vector_length);
        chunk_cmp = __riscv_vmerge_vxm_u8m1(chunk_cmp, 0x00, is_nan, vector_length);

        vbool8_t greater_b8 = __riscv_vmsltu_vv_u8m1_b8(best_cmp_u8m1, chunk_cmp, vector_length);
        best_cmp_u8m1 = __riscv_vmerge_vvm_u8m1(best_cmp_u8m1, chunk_cmp, greater_b8, vector_length);

        vuint64m8_t position_u64m8 = __riscv_vid_v_u64m8(vector_length);
        position_u64m8 = __riscv_vadd_vx_u64m8(position_u64m8, (nk_u64_t)offset, vector_length);
        best_index_u64m8 = __riscv_vmerge_vvm_u64m8(best_index_u64m8, position_u64m8, greater_b8, vector_length);
    }

    vuint8m1_t identity_u8m1 = __riscv_vmv_v_x_u8m1(0x00, 1);
    vuint8m1_t reduced_u8m1 = __riscv_vredmaxu_vs_u8m1_u8m1(best_cmp_u8m1, identity_u8m1, vector_length_max);
    nk_u8_t best_cmp = (nk_u8_t)__riscv_vmv_x_s_u8m1_u8(reduced_u8m1);

    vbool8_t match_b8 = __riscv_vmseq_vx_u8m1_b8(best_cmp_u8m1, best_cmp, vector_length_max);
    vuint64m8_t sentinel_u64m8 = __riscv_vmv_v_x_u64m8(NK_U64_MAX, vector_length_max);
    vuint64m8_t candidates_u64m8 = __riscv_vmerge_vvm_u64m8(sentinel_u64m8, best_index_u64m8, match_b8,
                                                            vector_length_max);
    vuint64m1_t identity_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    vuint64m1_t reduced_u64m1 = __riscv_vredminu_vs_u64m8_u64m1(candidates_u64m8, identity_u64m1, vector_length_max);

    *max_index = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(reduced_u64m1);

    nk_u8_t best_raw;
    if (best_cmp < 0x80) best_raw = best_cmp ^ 0xFF;
    else best_raw = best_cmp ^ 0x80;
    nk_e5m2_t best_e5m2;
    *(nk_u8_t *)&best_e5m2 = best_raw;
    nk_e5m2_to_f32(&best_e5m2, max_value);
}

NK_INTERNAL void nk_reduce_max_e5m2_rvv_strided_(                   //
    nk_e5m2_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *max_value, nk_size_t *max_index) {

    nk_size_t vector_length_max = __riscv_vsetvlmax_e8m1();
    vuint8m1_t best_cmp_u8m1 = __riscv_vmv_v_x_u8m1(0x00, vector_length_max);
    vuint64m8_t best_index_u64m8 = __riscv_vmv_v_x_u64m8(0, vector_length_max);

    unsigned char const *ptr = (unsigned char const *)data;
    nk_size_t offset = 0;
    for (nk_size_t remaining = count, vector_length; remaining > 0;
         remaining -= vector_length, offset += vector_length, ptr += vector_length * stride_bytes) {
        vector_length = __riscv_vsetvl_e8m1(remaining);
        vuint8m1_t chunk_u8m1 = __riscv_vlse8_v_u8m1((nk_u8_t const *)ptr, (nk_ssize_t)stride_bytes, vector_length);
        vuint8m1_t chunk_cmp = nk_fp8_to_comparable_u8m1_rvv_(chunk_u8m1, vector_length);

        vuint8m1_t abs_val = __riscv_vand_vx_u8m1(chunk_u8m1, 0x7F, vector_length);
        vbool8_t is_nan = __riscv_vmsgeu_vx_u8m1_b8(abs_val, 0x7D, vector_length);
        chunk_cmp = __riscv_vmerge_vxm_u8m1(chunk_cmp, 0x00, is_nan, vector_length);

        vbool8_t greater_b8 = __riscv_vmsltu_vv_u8m1_b8(best_cmp_u8m1, chunk_cmp, vector_length);
        best_cmp_u8m1 = __riscv_vmerge_vvm_u8m1(best_cmp_u8m1, chunk_cmp, greater_b8, vector_length);

        vuint64m8_t position_u64m8 = __riscv_vid_v_u64m8(vector_length);
        position_u64m8 = __riscv_vadd_vx_u64m8(position_u64m8, (nk_u64_t)offset, vector_length);
        best_index_u64m8 = __riscv_vmerge_vvm_u64m8(best_index_u64m8, position_u64m8, greater_b8, vector_length);
    }

    vuint8m1_t identity_u8m1 = __riscv_vmv_v_x_u8m1(0x00, 1);
    vuint8m1_t reduced_u8m1 = __riscv_vredmaxu_vs_u8m1_u8m1(best_cmp_u8m1, identity_u8m1, vector_length_max);
    nk_u8_t best_cmp = (nk_u8_t)__riscv_vmv_x_s_u8m1_u8(reduced_u8m1);

    vbool8_t match_b8 = __riscv_vmseq_vx_u8m1_b8(best_cmp_u8m1, best_cmp, vector_length_max);
    vuint64m8_t sentinel_u64m8 = __riscv_vmv_v_x_u64m8(NK_U64_MAX, vector_length_max);
    vuint64m8_t candidates_u64m8 = __riscv_vmerge_vvm_u64m8(sentinel_u64m8, best_index_u64m8, match_b8,
                                                            vector_length_max);
    vuint64m1_t identity_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    vuint64m1_t reduced_u64m1 = __riscv_vredminu_vs_u64m8_u64m1(candidates_u64m8, identity_u64m1, vector_length_max);

    *max_index = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(reduced_u64m1);

    nk_u8_t best_raw;
    if (best_cmp < 0x80) best_raw = best_cmp ^ 0xFF;
    else best_raw = best_cmp ^ 0x80;
    nk_e5m2_t best_e5m2;
    *(nk_u8_t *)&best_e5m2 = best_raw;
    nk_e5m2_to_f32(&best_e5m2, max_value);
}

NK_PUBLIC void nk_reduce_min_e5m2_rvv(nk_e5m2_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                      nk_f32_t *min_value, nk_size_t *min_index) {
    if (count == 0) *min_value = NK_F32_MAX, *min_index = 0;
    else if (stride_bytes == sizeof(nk_e5m2_t)) nk_reduce_min_e5m2_rvv_contiguous_(data, count, min_value, min_index);
    else nk_reduce_min_e5m2_rvv_strided_(data, count, stride_bytes, min_value, min_index);
}

NK_PUBLIC void nk_reduce_max_e5m2_rvv(nk_e5m2_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                      nk_f32_t *max_value, nk_size_t *max_index) {
    if (count == 0) *max_value = NK_F32_MAX, *max_index = 0;
    else if (stride_bytes == sizeof(nk_e5m2_t)) nk_reduce_max_e5m2_rvv_contiguous_(data, count, max_value, max_index);
    else nk_reduce_max_e5m2_rvv_strided_(data, count, stride_bytes, max_value, max_index);
}

NK_INTERNAL void nk_reduce_min_e2m3_rvv_contiguous_( //
    nk_e2m3_t const *data, nk_size_t count,          //
    nk_e2m3_t *min_value, nk_size_t *min_index) {

    nk_size_t vector_length_max = __riscv_vsetvlmax_e8m1();
    vuint8m1_t best_cmp_u8m1 = __riscv_vmv_v_x_u8m1(0x3F, vector_length_max);
    vuint64m8_t best_index_u64m8 = __riscv_vmv_v_x_u64m8(0, vector_length_max);

    nk_size_t offset = 0;
    for (nk_size_t remaining = count, vector_length; remaining > 0;
         remaining -= vector_length, offset += vector_length) {
        vector_length = __riscv_vsetvl_e8m1(remaining);
        vuint8m1_t chunk_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)data + offset, vector_length);
        vuint8m1_t chunk_cmp = nk_fp6_to_comparable_u8m1_rvv_(chunk_u8m1, vector_length);

        vbool8_t less_b8 = __riscv_vmsltu_vv_u8m1_b8(chunk_cmp, best_cmp_u8m1, vector_length);
        best_cmp_u8m1 = __riscv_vmerge_vvm_u8m1(best_cmp_u8m1, chunk_cmp, less_b8, vector_length);

        vuint64m8_t position_u64m8 = __riscv_vid_v_u64m8(vector_length);
        position_u64m8 = __riscv_vadd_vx_u64m8(position_u64m8, (nk_u64_t)offset, vector_length);
        best_index_u64m8 = __riscv_vmerge_vvm_u64m8(best_index_u64m8, position_u64m8, less_b8, vector_length);
    }

    vuint8m1_t identity_u8m1 = __riscv_vmv_v_x_u8m1(0x3F, 1);
    vuint8m1_t reduced_u8m1 = __riscv_vredminu_vs_u8m1_u8m1(best_cmp_u8m1, identity_u8m1, vector_length_max);
    nk_u8_t best_cmp = (nk_u8_t)__riscv_vmv_x_s_u8m1_u8(reduced_u8m1);

    vbool8_t match_b8 = __riscv_vmseq_vx_u8m1_b8(best_cmp_u8m1, best_cmp, vector_length_max);
    vuint64m8_t sentinel_u64m8 = __riscv_vmv_v_x_u64m8(NK_U64_MAX, vector_length_max);
    vuint64m8_t candidates_u64m8 = __riscv_vmerge_vvm_u64m8(sentinel_u64m8, best_index_u64m8, match_b8,
                                                            vector_length_max);
    vuint64m1_t identity_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    vuint64m1_t reduced_u64m1 = __riscv_vredminu_vs_u64m8_u64m1(candidates_u64m8, identity_u64m1, vector_length_max);

    *min_index = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(reduced_u64m1);
    *min_value = data[*min_index];
}

NK_INTERNAL void nk_reduce_min_e2m3_rvv_strided_(                   //
    nk_e2m3_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_e2m3_t *min_value, nk_size_t *min_index) {

    nk_size_t vector_length_max = __riscv_vsetvlmax_e8m1();
    vuint8m1_t best_cmp_u8m1 = __riscv_vmv_v_x_u8m1(0x3F, vector_length_max);
    vuint64m8_t best_index_u64m8 = __riscv_vmv_v_x_u64m8(0, vector_length_max);

    unsigned char const *ptr = (unsigned char const *)data;
    nk_size_t offset = 0;
    for (nk_size_t remaining = count, vector_length; remaining > 0;
         remaining -= vector_length, offset += vector_length, ptr += vector_length * stride_bytes) {
        vector_length = __riscv_vsetvl_e8m1(remaining);
        vuint8m1_t chunk_u8m1 = __riscv_vlse8_v_u8m1((nk_u8_t const *)ptr, (nk_ssize_t)stride_bytes, vector_length);
        vuint8m1_t chunk_cmp = nk_fp6_to_comparable_u8m1_rvv_(chunk_u8m1, vector_length);

        vbool8_t less_b8 = __riscv_vmsltu_vv_u8m1_b8(chunk_cmp, best_cmp_u8m1, vector_length);
        best_cmp_u8m1 = __riscv_vmerge_vvm_u8m1(best_cmp_u8m1, chunk_cmp, less_b8, vector_length);

        vuint64m8_t position_u64m8 = __riscv_vid_v_u64m8(vector_length);
        position_u64m8 = __riscv_vadd_vx_u64m8(position_u64m8, (nk_u64_t)offset, vector_length);
        best_index_u64m8 = __riscv_vmerge_vvm_u64m8(best_index_u64m8, position_u64m8, less_b8, vector_length);
    }

    vuint8m1_t identity_u8m1 = __riscv_vmv_v_x_u8m1(0x3F, 1);
    vuint8m1_t reduced_u8m1 = __riscv_vredminu_vs_u8m1_u8m1(best_cmp_u8m1, identity_u8m1, vector_length_max);
    nk_u8_t best_cmp = (nk_u8_t)__riscv_vmv_x_s_u8m1_u8(reduced_u8m1);

    vbool8_t match_b8 = __riscv_vmseq_vx_u8m1_b8(best_cmp_u8m1, best_cmp, vector_length_max);
    vuint64m8_t sentinel_u64m8 = __riscv_vmv_v_x_u64m8(NK_U64_MAX, vector_length_max);
    vuint64m8_t candidates_u64m8 = __riscv_vmerge_vvm_u64m8(sentinel_u64m8, best_index_u64m8, match_b8,
                                                            vector_length_max);
    vuint64m1_t identity_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    vuint64m1_t reduced_u64m1 = __riscv_vredminu_vs_u64m8_u64m1(candidates_u64m8, identity_u64m1, vector_length_max);

    *min_index = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(reduced_u64m1);
    *min_value = *(nk_e2m3_t const *)((unsigned char const *)data + (*min_index) * stride_bytes);
}

NK_INTERNAL void nk_reduce_max_e2m3_rvv_contiguous_( //
    nk_e2m3_t const *data, nk_size_t count,          //
    nk_e2m3_t *max_value, nk_size_t *max_index) {

    nk_size_t vector_length_max = __riscv_vsetvlmax_e8m1();
    vuint8m1_t best_cmp_u8m1 = __riscv_vmv_v_x_u8m1(0x00, vector_length_max);
    vuint64m8_t best_index_u64m8 = __riscv_vmv_v_x_u64m8(0, vector_length_max);

    nk_size_t offset = 0;
    for (nk_size_t remaining = count, vector_length; remaining > 0;
         remaining -= vector_length, offset += vector_length) {
        vector_length = __riscv_vsetvl_e8m1(remaining);
        vuint8m1_t chunk_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)data + offset, vector_length);
        vuint8m1_t chunk_cmp = nk_fp6_to_comparable_u8m1_rvv_(chunk_u8m1, vector_length);

        vbool8_t greater_b8 = __riscv_vmsltu_vv_u8m1_b8(best_cmp_u8m1, chunk_cmp, vector_length);
        best_cmp_u8m1 = __riscv_vmerge_vvm_u8m1(best_cmp_u8m1, chunk_cmp, greater_b8, vector_length);

        vuint64m8_t position_u64m8 = __riscv_vid_v_u64m8(vector_length);
        position_u64m8 = __riscv_vadd_vx_u64m8(position_u64m8, (nk_u64_t)offset, vector_length);
        best_index_u64m8 = __riscv_vmerge_vvm_u64m8(best_index_u64m8, position_u64m8, greater_b8, vector_length);
    }

    vuint8m1_t identity_u8m1 = __riscv_vmv_v_x_u8m1(0x00, 1);
    vuint8m1_t reduced_u8m1 = __riscv_vredmaxu_vs_u8m1_u8m1(best_cmp_u8m1, identity_u8m1, vector_length_max);
    nk_u8_t best_cmp = (nk_u8_t)__riscv_vmv_x_s_u8m1_u8(reduced_u8m1);

    vbool8_t match_b8 = __riscv_vmseq_vx_u8m1_b8(best_cmp_u8m1, best_cmp, vector_length_max);
    vuint64m8_t sentinel_u64m8 = __riscv_vmv_v_x_u64m8(NK_U64_MAX, vector_length_max);
    vuint64m8_t candidates_u64m8 = __riscv_vmerge_vvm_u64m8(sentinel_u64m8, best_index_u64m8, match_b8,
                                                            vector_length_max);
    vuint64m1_t identity_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    vuint64m1_t reduced_u64m1 = __riscv_vredminu_vs_u64m8_u64m1(candidates_u64m8, identity_u64m1, vector_length_max);

    *max_index = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(reduced_u64m1);
    *max_value = data[*max_index];
}

NK_INTERNAL void nk_reduce_max_e2m3_rvv_strided_(                   //
    nk_e2m3_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_e2m3_t *max_value, nk_size_t *max_index) {

    nk_size_t vector_length_max = __riscv_vsetvlmax_e8m1();
    vuint8m1_t best_cmp_u8m1 = __riscv_vmv_v_x_u8m1(0x00, vector_length_max);
    vuint64m8_t best_index_u64m8 = __riscv_vmv_v_x_u64m8(0, vector_length_max);

    unsigned char const *ptr = (unsigned char const *)data;
    nk_size_t offset = 0;
    for (nk_size_t remaining = count, vector_length; remaining > 0;
         remaining -= vector_length, offset += vector_length, ptr += vector_length * stride_bytes) {
        vector_length = __riscv_vsetvl_e8m1(remaining);
        vuint8m1_t chunk_u8m1 = __riscv_vlse8_v_u8m1((nk_u8_t const *)ptr, (nk_ssize_t)stride_bytes, vector_length);
        vuint8m1_t chunk_cmp = nk_fp6_to_comparable_u8m1_rvv_(chunk_u8m1, vector_length);

        vbool8_t greater_b8 = __riscv_vmsltu_vv_u8m1_b8(best_cmp_u8m1, chunk_cmp, vector_length);
        best_cmp_u8m1 = __riscv_vmerge_vvm_u8m1(best_cmp_u8m1, chunk_cmp, greater_b8, vector_length);

        vuint64m8_t position_u64m8 = __riscv_vid_v_u64m8(vector_length);
        position_u64m8 = __riscv_vadd_vx_u64m8(position_u64m8, (nk_u64_t)offset, vector_length);
        best_index_u64m8 = __riscv_vmerge_vvm_u64m8(best_index_u64m8, position_u64m8, greater_b8, vector_length);
    }

    vuint8m1_t identity_u8m1 = __riscv_vmv_v_x_u8m1(0x00, 1);
    vuint8m1_t reduced_u8m1 = __riscv_vredmaxu_vs_u8m1_u8m1(best_cmp_u8m1, identity_u8m1, vector_length_max);
    nk_u8_t best_cmp = (nk_u8_t)__riscv_vmv_x_s_u8m1_u8(reduced_u8m1);

    vbool8_t match_b8 = __riscv_vmseq_vx_u8m1_b8(best_cmp_u8m1, best_cmp, vector_length_max);
    vuint64m8_t sentinel_u64m8 = __riscv_vmv_v_x_u64m8(NK_U64_MAX, vector_length_max);
    vuint64m8_t candidates_u64m8 = __riscv_vmerge_vvm_u64m8(sentinel_u64m8, best_index_u64m8, match_b8,
                                                            vector_length_max);
    vuint64m1_t identity_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    vuint64m1_t reduced_u64m1 = __riscv_vredminu_vs_u64m8_u64m1(candidates_u64m8, identity_u64m1, vector_length_max);

    *max_index = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(reduced_u64m1);
    *max_value = *(nk_e2m3_t const *)((unsigned char const *)data + (*max_index) * stride_bytes);
}

NK_PUBLIC void nk_reduce_min_e2m3_rvv(nk_e2m3_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                      nk_e2m3_t *min_value, nk_size_t *min_index) {
    if (count == 0) *(nk_u8_t *)min_value = 0, *min_index = 0;
    else if (stride_bytes == sizeof(nk_e2m3_t)) nk_reduce_min_e2m3_rvv_contiguous_(data, count, min_value, min_index);
    else nk_reduce_min_e2m3_rvv_strided_(data, count, stride_bytes, min_value, min_index);
}

NK_PUBLIC void nk_reduce_max_e2m3_rvv(nk_e2m3_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                      nk_e2m3_t *max_value, nk_size_t *max_index) {
    if (count == 0) *(nk_u8_t *)max_value = 0, *max_index = 0;
    else if (stride_bytes == sizeof(nk_e2m3_t)) nk_reduce_max_e2m3_rvv_contiguous_(data, count, max_value, max_index);
    else nk_reduce_max_e2m3_rvv_strided_(data, count, stride_bytes, max_value, max_index);
}

NK_INTERNAL void nk_reduce_min_e3m2_rvv_contiguous_( //
    nk_e3m2_t const *data, nk_size_t count,          //
    nk_e3m2_t *min_value, nk_size_t *min_index) {

    nk_size_t vector_length_max = __riscv_vsetvlmax_e8m1();
    vuint8m1_t best_cmp_u8m1 = __riscv_vmv_v_x_u8m1(0x3F, vector_length_max);
    vuint64m8_t best_index_u64m8 = __riscv_vmv_v_x_u64m8(0, vector_length_max);

    nk_size_t offset = 0;
    for (nk_size_t remaining = count, vector_length; remaining > 0;
         remaining -= vector_length, offset += vector_length) {
        vector_length = __riscv_vsetvl_e8m1(remaining);
        vuint8m1_t chunk_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)data + offset, vector_length);
        vuint8m1_t chunk_cmp = nk_fp6_to_comparable_u8m1_rvv_(chunk_u8m1, vector_length);

        vbool8_t less_b8 = __riscv_vmsltu_vv_u8m1_b8(chunk_cmp, best_cmp_u8m1, vector_length);
        best_cmp_u8m1 = __riscv_vmerge_vvm_u8m1(best_cmp_u8m1, chunk_cmp, less_b8, vector_length);

        vuint64m8_t position_u64m8 = __riscv_vid_v_u64m8(vector_length);
        position_u64m8 = __riscv_vadd_vx_u64m8(position_u64m8, (nk_u64_t)offset, vector_length);
        best_index_u64m8 = __riscv_vmerge_vvm_u64m8(best_index_u64m8, position_u64m8, less_b8, vector_length);
    }

    vuint8m1_t identity_u8m1 = __riscv_vmv_v_x_u8m1(0x3F, 1);
    vuint8m1_t reduced_u8m1 = __riscv_vredminu_vs_u8m1_u8m1(best_cmp_u8m1, identity_u8m1, vector_length_max);
    nk_u8_t best_cmp = (nk_u8_t)__riscv_vmv_x_s_u8m1_u8(reduced_u8m1);

    vbool8_t match_b8 = __riscv_vmseq_vx_u8m1_b8(best_cmp_u8m1, best_cmp, vector_length_max);
    vuint64m8_t sentinel_u64m8 = __riscv_vmv_v_x_u64m8(NK_U64_MAX, vector_length_max);
    vuint64m8_t candidates_u64m8 = __riscv_vmerge_vvm_u64m8(sentinel_u64m8, best_index_u64m8, match_b8,
                                                            vector_length_max);
    vuint64m1_t identity_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    vuint64m1_t reduced_u64m1 = __riscv_vredminu_vs_u64m8_u64m1(candidates_u64m8, identity_u64m1, vector_length_max);

    *min_index = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(reduced_u64m1);
    *min_value = data[*min_index];
}

NK_INTERNAL void nk_reduce_min_e3m2_rvv_strided_(                   //
    nk_e3m2_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_e3m2_t *min_value, nk_size_t *min_index) {

    nk_size_t vector_length_max = __riscv_vsetvlmax_e8m1();
    vuint8m1_t best_cmp_u8m1 = __riscv_vmv_v_x_u8m1(0x3F, vector_length_max);
    vuint64m8_t best_index_u64m8 = __riscv_vmv_v_x_u64m8(0, vector_length_max);

    unsigned char const *ptr = (unsigned char const *)data;
    nk_size_t offset = 0;
    for (nk_size_t remaining = count, vector_length; remaining > 0;
         remaining -= vector_length, offset += vector_length, ptr += vector_length * stride_bytes) {
        vector_length = __riscv_vsetvl_e8m1(remaining);
        vuint8m1_t chunk_u8m1 = __riscv_vlse8_v_u8m1((nk_u8_t const *)ptr, (nk_ssize_t)stride_bytes, vector_length);
        vuint8m1_t chunk_cmp = nk_fp6_to_comparable_u8m1_rvv_(chunk_u8m1, vector_length);

        vbool8_t less_b8 = __riscv_vmsltu_vv_u8m1_b8(chunk_cmp, best_cmp_u8m1, vector_length);
        best_cmp_u8m1 = __riscv_vmerge_vvm_u8m1(best_cmp_u8m1, chunk_cmp, less_b8, vector_length);

        vuint64m8_t position_u64m8 = __riscv_vid_v_u64m8(vector_length);
        position_u64m8 = __riscv_vadd_vx_u64m8(position_u64m8, (nk_u64_t)offset, vector_length);
        best_index_u64m8 = __riscv_vmerge_vvm_u64m8(best_index_u64m8, position_u64m8, less_b8, vector_length);
    }

    vuint8m1_t identity_u8m1 = __riscv_vmv_v_x_u8m1(0x3F, 1);
    vuint8m1_t reduced_u8m1 = __riscv_vredminu_vs_u8m1_u8m1(best_cmp_u8m1, identity_u8m1, vector_length_max);
    nk_u8_t best_cmp = (nk_u8_t)__riscv_vmv_x_s_u8m1_u8(reduced_u8m1);

    vbool8_t match_b8 = __riscv_vmseq_vx_u8m1_b8(best_cmp_u8m1, best_cmp, vector_length_max);
    vuint64m8_t sentinel_u64m8 = __riscv_vmv_v_x_u64m8(NK_U64_MAX, vector_length_max);
    vuint64m8_t candidates_u64m8 = __riscv_vmerge_vvm_u64m8(sentinel_u64m8, best_index_u64m8, match_b8,
                                                            vector_length_max);
    vuint64m1_t identity_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    vuint64m1_t reduced_u64m1 = __riscv_vredminu_vs_u64m8_u64m1(candidates_u64m8, identity_u64m1, vector_length_max);

    *min_index = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(reduced_u64m1);
    *min_value = *(nk_e3m2_t const *)((unsigned char const *)data + (*min_index) * stride_bytes);
}

NK_INTERNAL void nk_reduce_max_e3m2_rvv_contiguous_( //
    nk_e3m2_t const *data, nk_size_t count,          //
    nk_e3m2_t *max_value, nk_size_t *max_index) {

    nk_size_t vector_length_max = __riscv_vsetvlmax_e8m1();
    vuint8m1_t best_cmp_u8m1 = __riscv_vmv_v_x_u8m1(0x00, vector_length_max);
    vuint64m8_t best_index_u64m8 = __riscv_vmv_v_x_u64m8(0, vector_length_max);

    nk_size_t offset = 0;
    for (nk_size_t remaining = count, vector_length; remaining > 0;
         remaining -= vector_length, offset += vector_length) {
        vector_length = __riscv_vsetvl_e8m1(remaining);
        vuint8m1_t chunk_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)data + offset, vector_length);
        vuint8m1_t chunk_cmp = nk_fp6_to_comparable_u8m1_rvv_(chunk_u8m1, vector_length);

        vbool8_t greater_b8 = __riscv_vmsltu_vv_u8m1_b8(best_cmp_u8m1, chunk_cmp, vector_length);
        best_cmp_u8m1 = __riscv_vmerge_vvm_u8m1(best_cmp_u8m1, chunk_cmp, greater_b8, vector_length);

        vuint64m8_t position_u64m8 = __riscv_vid_v_u64m8(vector_length);
        position_u64m8 = __riscv_vadd_vx_u64m8(position_u64m8, (nk_u64_t)offset, vector_length);
        best_index_u64m8 = __riscv_vmerge_vvm_u64m8(best_index_u64m8, position_u64m8, greater_b8, vector_length);
    }

    vuint8m1_t identity_u8m1 = __riscv_vmv_v_x_u8m1(0x00, 1);
    vuint8m1_t reduced_u8m1 = __riscv_vredmaxu_vs_u8m1_u8m1(best_cmp_u8m1, identity_u8m1, vector_length_max);
    nk_u8_t best_cmp = (nk_u8_t)__riscv_vmv_x_s_u8m1_u8(reduced_u8m1);

    vbool8_t match_b8 = __riscv_vmseq_vx_u8m1_b8(best_cmp_u8m1, best_cmp, vector_length_max);
    vuint64m8_t sentinel_u64m8 = __riscv_vmv_v_x_u64m8(NK_U64_MAX, vector_length_max);
    vuint64m8_t candidates_u64m8 = __riscv_vmerge_vvm_u64m8(sentinel_u64m8, best_index_u64m8, match_b8,
                                                            vector_length_max);
    vuint64m1_t identity_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    vuint64m1_t reduced_u64m1 = __riscv_vredminu_vs_u64m8_u64m1(candidates_u64m8, identity_u64m1, vector_length_max);

    *max_index = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(reduced_u64m1);
    *max_value = data[*max_index];
}

NK_INTERNAL void nk_reduce_max_e3m2_rvv_strided_(                   //
    nk_e3m2_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_e3m2_t *max_value, nk_size_t *max_index) {

    nk_size_t vector_length_max = __riscv_vsetvlmax_e8m1();
    vuint8m1_t best_cmp_u8m1 = __riscv_vmv_v_x_u8m1(0x00, vector_length_max);
    vuint64m8_t best_index_u64m8 = __riscv_vmv_v_x_u64m8(0, vector_length_max);

    unsigned char const *ptr = (unsigned char const *)data;
    nk_size_t offset = 0;
    for (nk_size_t remaining = count, vector_length; remaining > 0;
         remaining -= vector_length, offset += vector_length, ptr += vector_length * stride_bytes) {
        vector_length = __riscv_vsetvl_e8m1(remaining);
        vuint8m1_t chunk_u8m1 = __riscv_vlse8_v_u8m1((nk_u8_t const *)ptr, (nk_ssize_t)stride_bytes, vector_length);
        vuint8m1_t chunk_cmp = nk_fp6_to_comparable_u8m1_rvv_(chunk_u8m1, vector_length);

        vbool8_t greater_b8 = __riscv_vmsltu_vv_u8m1_b8(best_cmp_u8m1, chunk_cmp, vector_length);
        best_cmp_u8m1 = __riscv_vmerge_vvm_u8m1(best_cmp_u8m1, chunk_cmp, greater_b8, vector_length);

        vuint64m8_t position_u64m8 = __riscv_vid_v_u64m8(vector_length);
        position_u64m8 = __riscv_vadd_vx_u64m8(position_u64m8, (nk_u64_t)offset, vector_length);
        best_index_u64m8 = __riscv_vmerge_vvm_u64m8(best_index_u64m8, position_u64m8, greater_b8, vector_length);
    }

    vuint8m1_t identity_u8m1 = __riscv_vmv_v_x_u8m1(0x00, 1);
    vuint8m1_t reduced_u8m1 = __riscv_vredmaxu_vs_u8m1_u8m1(best_cmp_u8m1, identity_u8m1, vector_length_max);
    nk_u8_t best_cmp = (nk_u8_t)__riscv_vmv_x_s_u8m1_u8(reduced_u8m1);

    vbool8_t match_b8 = __riscv_vmseq_vx_u8m1_b8(best_cmp_u8m1, best_cmp, vector_length_max);
    vuint64m8_t sentinel_u64m8 = __riscv_vmv_v_x_u64m8(NK_U64_MAX, vector_length_max);
    vuint64m8_t candidates_u64m8 = __riscv_vmerge_vvm_u64m8(sentinel_u64m8, best_index_u64m8, match_b8,
                                                            vector_length_max);
    vuint64m1_t identity_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    vuint64m1_t reduced_u64m1 = __riscv_vredminu_vs_u64m8_u64m1(candidates_u64m8, identity_u64m1, vector_length_max);

    *max_index = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(reduced_u64m1);
    *max_value = *(nk_e3m2_t const *)((unsigned char const *)data + (*max_index) * stride_bytes);
}

NK_PUBLIC void nk_reduce_min_e3m2_rvv(nk_e3m2_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                      nk_e3m2_t *min_value, nk_size_t *min_index) {
    if (count == 0) *(nk_u8_t *)min_value = 0, *min_index = 0;
    else if (stride_bytes == sizeof(nk_e3m2_t)) nk_reduce_min_e3m2_rvv_contiguous_(data, count, min_value, min_index);
    else nk_reduce_min_e3m2_rvv_strided_(data, count, stride_bytes, min_value, min_index);
}

NK_PUBLIC void nk_reduce_max_e3m2_rvv(nk_e3m2_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                      nk_e3m2_t *max_value, nk_size_t *max_index) {
    if (count == 0) *(nk_u8_t *)max_value = 0, *max_index = 0;
    else if (stride_bytes == sizeof(nk_e3m2_t)) nk_reduce_max_e3m2_rvv_contiguous_(data, count, max_value, max_index);
    else nk_reduce_max_e3m2_rvv_strided_(data, count, stride_bytes, max_value, max_index);
}

#pragma endregion - Min / Max Reductions

#if defined(__cplusplus)
} // extern "C"
#endif

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#endif // NK_TARGET_RVV
#endif // NK_TARGET_RISCV_
#endif // NK_REDUCE_RVV_H

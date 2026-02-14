/**
 *  @brief RISC-V Vector implementations for the redesigned reduction API (moments + minmax).
 *  @file include/numkong/reduce/rvv_new.h
 *  @author Ash Vardanian
 *  @date February 13, 2026
 *
 *  @sa include/numkong/reduce.h
 */
#ifndef NK_REDUCE_RVV_H
#define NK_REDUCE_RVV_H

#if NK_TARGET_RISCV_
#if NK_TARGET_RVV

#include "numkong/types.h"
#include "numkong/cast/rvv.h"
#include "numkong/reduce/serial.h"

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=+v"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=+v")
#endif

#if defined(__cplusplus)
extern "C" {
#endif

NK_INTERNAL void nk_reduce_moments_f32_rvv_contiguous_( //
    nk_f32_t const *data, nk_size_t count,              //
    nk_f64_t *sum_ptr, nk_f64_t *sumsq_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e64m2();
    vfloat64m2_t sum_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, vlmax);
    vfloat64m2_t sumsq_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, vlmax);
    for (nk_size_t vl; count > 0; count -= vl, data += vl) {
        vl = __riscv_vsetvl_e32m1(count);
        vfloat32m1_t d = __riscv_vle32_v_f32m1(data, vl);
        sum_f64m2 = __riscv_vfwadd_wv_f64m2(sum_f64m2, d, vl);
        sumsq_f64m2 = __riscv_vfwmacc_vv_f64m2(sumsq_f64m2, d, d, vl);
    }
    vfloat64m1_t zero = __riscv_vfmv_v_f_f64m1(0.0, 1);
    *sum_ptr = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m2_f64m1(sum_f64m2, zero, vlmax)),
    *sumsq_ptr = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m2_f64m1(sumsq_f64m2, zero, vlmax));
}

NK_INTERNAL void nk_reduce_moments_f32_rvv_strided_(               //
    nk_f32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f64_t *sum_ptr, nk_f64_t *sumsq_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e64m2();
    vfloat64m2_t sum_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, vlmax);
    vfloat64m2_t sumsq_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, vlmax);
    unsigned char const *ptr = (unsigned char const *)data;
    for (nk_size_t vl; count > 0; count -= vl, ptr += vl * stride_bytes) {
        vl = __riscv_vsetvl_e32m1(count);
        vfloat32m1_t d = __riscv_vlse32_v_f32m1((nk_f32_t const *)ptr, (nk_ssize_t)stride_bytes, vl);
        sum_f64m2 = __riscv_vfwadd_wv_f64m2(sum_f64m2, d, vl);
        sumsq_f64m2 = __riscv_vfwmacc_vv_f64m2(sumsq_f64m2, d, d, vl);
    }
    vfloat64m1_t zero = __riscv_vfmv_v_f_f64m1(0.0, 1);
    *sum_ptr = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m2_f64m1(sum_f64m2, zero, vlmax)),
    *sumsq_ptr = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m2_f64m1(sumsq_f64m2, zero, vlmax));
}

NK_PUBLIC void nk_reduce_moments_f32_rvv(                          //
    nk_f32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f64_t *sum, nk_f64_t *sumsq) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_f32_t);
    int aligned = (stride_bytes % sizeof(nk_f32_t) == 0);
    if (count == 0) *sum = 0, *sumsq = 0;
    else if (!aligned) nk_reduce_moments_f32_serial(data, count, stride_bytes, sum, sumsq);
    else if (stride_elements == 1) nk_reduce_moments_f32_rvv_contiguous_(data, count, sum, sumsq);
    else nk_reduce_moments_f32_rvv_strided_(data, count, stride_bytes, sum, sumsq);
}

NK_INTERNAL void nk_reduce_minmax_f32_rvv_contiguous_( //
    nk_f32_t const *data, nk_size_t count,             //
    nk_f32_t *min_value, nk_size_t *min_index,         //
    nk_f32_t *max_value, nk_size_t *max_index) {
    nk_size_t vlmax = __riscv_vsetvlmax_e32m1();
    vfloat32m1_t min = __riscv_vfmv_v_f_f32m1(NK_F32_MAX, vlmax);
    vfloat32m1_t max = __riscv_vfmv_v_f_f32m1(NK_F32_MIN, vlmax);
    vuint64m2_t min_indices = __riscv_vmv_v_x_u64m2(0, vlmax);
    vuint64m2_t max_indices = __riscv_vmv_v_x_u64m2(0, vlmax);
    nk_size_t offset = 0;
    for (nk_size_t remaining = count, vl; remaining > 0; remaining -= vl, offset += vl) {
        vl = __riscv_vsetvl_e32m1(remaining);
        vfloat32m1_t d = __riscv_vle32_v_f32m1(data + offset, vl);
        vuint64m2_t pos = __riscv_vadd_vx_u64m2(__riscv_vid_v_u64m2(vl), (nk_u64_t)offset, vl);
        vbool32_t less = __riscv_vmflt_vv_f32m1_b32(d, min, vl);
        min = __riscv_vmerge_vvm_f32m1(min, d, less, vl);
        min_indices = __riscv_vmerge_vvm_u64m2(min_indices, pos, less, vl);
        vbool32_t greater = __riscv_vmflt_vv_f32m1_b32(max, d, vl);
        max = __riscv_vmerge_vvm_f32m1(max, d, greater, vl);
        max_indices = __riscv_vmerge_vvm_u64m2(max_indices, pos, greater, vl);
    }
    vfloat32m1_t id_max = __riscv_vfmv_v_f_f32m1(NK_F32_MAX, 1);
    nk_f32_t mn = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmin_vs_f32m1_f32m1(min, id_max, vlmax));
    vbool32_t min_match = __riscv_vmfeq_vf_f32m1_b32(min, mn, vlmax);
    vuint64m2_t sentinel = __riscv_vmv_v_x_u64m2(NK_U64_MAX, vlmax);
    vuint64m2_t min_cands = __riscv_vmerge_vvm_u64m2(sentinel, min_indices, min_match, vlmax);
    vuint64m1_t id_umax = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    *min_value = mn,
    *min_index = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(__riscv_vredminu_vs_u64m2_u64m1(min_cands, id_umax, vlmax));
    vfloat32m1_t id_min = __riscv_vfmv_v_f_f32m1(NK_F32_MIN, 1);
    nk_f32_t mx = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m1_f32m1(max, id_min, vlmax));
    vbool32_t max_match = __riscv_vmfeq_vf_f32m1_b32(max, mx, vlmax);
    vuint64m2_t max_cands = __riscv_vmerge_vvm_u64m2(sentinel, max_indices, max_match, vlmax);
    *max_value = mx,
    *max_index = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(__riscv_vredminu_vs_u64m2_u64m1(max_cands, id_umax, vlmax));
}

NK_INTERNAL void nk_reduce_minmax_f32_rvv_strided_(                //
    nk_f32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *min_value, nk_size_t *min_index,                     //
    nk_f32_t *max_value, nk_size_t *max_index) {
    nk_size_t vlmax = __riscv_vsetvlmax_e32m1();
    vfloat32m1_t min = __riscv_vfmv_v_f_f32m1(NK_F32_MAX, vlmax);
    vfloat32m1_t max = __riscv_vfmv_v_f_f32m1(NK_F32_MIN, vlmax);
    vuint64m2_t min_indices = __riscv_vmv_v_x_u64m2(0, vlmax);
    vuint64m2_t max_indices = __riscv_vmv_v_x_u64m2(0, vlmax);
    unsigned char const *ptr = (unsigned char const *)data;
    nk_size_t offset = 0;
    for (nk_size_t remaining = count, vl; remaining > 0; remaining -= vl, offset += vl, ptr += vl * stride_bytes) {
        vl = __riscv_vsetvl_e32m1(remaining);
        vfloat32m1_t d = __riscv_vlse32_v_f32m1((nk_f32_t const *)ptr, (nk_ssize_t)stride_bytes, vl);
        vuint64m2_t pos = __riscv_vadd_vx_u64m2(__riscv_vid_v_u64m2(vl), (nk_u64_t)offset, vl);
        vbool32_t less = __riscv_vmflt_vv_f32m1_b32(d, min, vl);
        min = __riscv_vmerge_vvm_f32m1(min, d, less, vl);
        min_indices = __riscv_vmerge_vvm_u64m2(min_indices, pos, less, vl);
        vbool32_t greater = __riscv_vmflt_vv_f32m1_b32(max, d, vl);
        max = __riscv_vmerge_vvm_f32m1(max, d, greater, vl);
        max_indices = __riscv_vmerge_vvm_u64m2(max_indices, pos, greater, vl);
    }
    vfloat32m1_t id_max = __riscv_vfmv_v_f_f32m1(NK_F32_MAX, 1);
    nk_f32_t mn = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmin_vs_f32m1_f32m1(min, id_max, vlmax));
    vbool32_t min_match = __riscv_vmfeq_vf_f32m1_b32(min, mn, vlmax);
    vuint64m2_t sentinel = __riscv_vmv_v_x_u64m2(NK_U64_MAX, vlmax);
    vuint64m2_t min_cands = __riscv_vmerge_vvm_u64m2(sentinel, min_indices, min_match, vlmax);
    vuint64m1_t id_umax = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    *min_value = mn,
    *min_index = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(__riscv_vredminu_vs_u64m2_u64m1(min_cands, id_umax, vlmax));
    vfloat32m1_t id_min = __riscv_vfmv_v_f_f32m1(NK_F32_MIN, 1);
    nk_f32_t mx = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m1_f32m1(max, id_min, vlmax));
    vbool32_t max_match = __riscv_vmfeq_vf_f32m1_b32(max, mx, vlmax);
    vuint64m2_t max_cands = __riscv_vmerge_vvm_u64m2(sentinel, max_indices, max_match, vlmax);
    *max_value = mx,
    *max_index = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(__riscv_vredminu_vs_u64m2_u64m1(max_cands, id_umax, vlmax));
}

NK_PUBLIC void nk_reduce_minmax_f32_rvv(                           //
    nk_f32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *min_value, nk_size_t *min_index,                     //
    nk_f32_t *max_value, nk_size_t *max_index) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_f32_t);
    int aligned = (stride_bytes % sizeof(nk_f32_t) == 0);
    if (count == 0)
        *min_value = NK_F32_MAX, *min_index = NK_SIZE_MAX, *max_value = NK_F32_MIN, *max_index = NK_SIZE_MAX;
    else if (!aligned)
        nk_reduce_minmax_f32_serial(data, count, stride_bytes, min_value, min_index, max_value, max_index);
    else if (stride_elements == 1)
        nk_reduce_minmax_f32_rvv_contiguous_(data, count, min_value, min_index, max_value, max_index);
    else nk_reduce_minmax_f32_rvv_strided_(data, count, stride_bytes, min_value, min_index, max_value, max_index);
}

NK_INTERNAL void nk_reduce_moments_f64_rvv_contiguous_( //
    nk_f64_t const *data, nk_size_t count,              //
    nk_f64_t *sum_ptr, nk_f64_t *sumsq_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e64m4();
    vfloat64m4_t sum_f64m4 = __riscv_vfmv_v_f_f64m4(0.0, vlmax);
    vfloat64m4_t sumsq_f64m4 = __riscv_vfmv_v_f_f64m4(0.0, vlmax);
    for (nk_size_t vl; count > 0; count -= vl, data += vl) {
        vl = __riscv_vsetvl_e64m4(count);
        vfloat64m4_t d = __riscv_vle64_v_f64m4(data, vl);
        sum_f64m4 = __riscv_vfadd_vv_f64m4(sum_f64m4, d, vl);
        sumsq_f64m4 = __riscv_vfmacc_vv_f64m4(sumsq_f64m4, d, d, vl);
    }
    vfloat64m1_t zero = __riscv_vfmv_v_f_f64m1(0.0, 1);
    *sum_ptr = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m4_f64m1(sum_f64m4, zero, vlmax)),
    *sumsq_ptr = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m4_f64m1(sumsq_f64m4, zero, vlmax));
}

NK_INTERNAL void nk_reduce_moments_f64_rvv_strided_(               //
    nk_f64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f64_t *sum_ptr, nk_f64_t *sumsq_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e64m4();
    vfloat64m4_t sum_f64m4 = __riscv_vfmv_v_f_f64m4(0.0, vlmax);
    vfloat64m4_t sumsq_f64m4 = __riscv_vfmv_v_f_f64m4(0.0, vlmax);
    unsigned char const *ptr = (unsigned char const *)data;
    for (nk_size_t vl; count > 0; count -= vl, ptr += vl * stride_bytes) {
        vl = __riscv_vsetvl_e64m4(count);
        vfloat64m4_t d = __riscv_vlse64_v_f64m4((nk_f64_t const *)ptr, (nk_ssize_t)stride_bytes, vl);
        sum_f64m4 = __riscv_vfadd_vv_f64m4(sum_f64m4, d, vl);
        sumsq_f64m4 = __riscv_vfmacc_vv_f64m4(sumsq_f64m4, d, d, vl);
    }
    vfloat64m1_t zero = __riscv_vfmv_v_f_f64m1(0.0, 1);
    *sum_ptr = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m4_f64m1(sum_f64m4, zero, vlmax)),
    *sumsq_ptr = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m4_f64m1(sumsq_f64m4, zero, vlmax));
}

NK_PUBLIC void nk_reduce_moments_f64_rvv(                          //
    nk_f64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f64_t *sum, nk_f64_t *sumsq) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_f64_t);
    int aligned = (stride_bytes % sizeof(nk_f64_t) == 0);
    if (count == 0) *sum = 0, *sumsq = 0;
    else if (!aligned) nk_reduce_moments_f64_serial(data, count, stride_bytes, sum, sumsq);
    else if (stride_elements == 1) nk_reduce_moments_f64_rvv_contiguous_(data, count, sum, sumsq);
    else nk_reduce_moments_f64_rvv_strided_(data, count, stride_bytes, sum, sumsq);
}

NK_INTERNAL void nk_reduce_minmax_f64_rvv_contiguous_( //
    nk_f64_t const *data, nk_size_t count,             //
    nk_f64_t *min_value, nk_size_t *min_index,         //
    nk_f64_t *max_value, nk_size_t *max_index) {
    nk_size_t vlmax = __riscv_vsetvlmax_e64m1();
    vfloat64m1_t min = __riscv_vfmv_v_f_f64m1(NK_F64_MAX, vlmax);
    vfloat64m1_t max = __riscv_vfmv_v_f_f64m1(NK_F64_MIN, vlmax);
    vuint64m1_t min_indices = __riscv_vmv_v_x_u64m1(0, vlmax);
    vuint64m1_t max_indices = __riscv_vmv_v_x_u64m1(0, vlmax);
    nk_size_t offset = 0;
    for (nk_size_t remaining = count, vl; remaining > 0; remaining -= vl, offset += vl) {
        vl = __riscv_vsetvl_e64m1(remaining);
        vfloat64m1_t d = __riscv_vle64_v_f64m1(data + offset, vl);
        vuint64m1_t pos = __riscv_vadd_vx_u64m1(__riscv_vid_v_u64m1(vl), (nk_u64_t)offset, vl);
        vbool64_t less = __riscv_vmflt_vv_f64m1_b64(d, min, vl);
        min = __riscv_vmerge_vvm_f64m1(min, d, less, vl);
        min_indices = __riscv_vmerge_vvm_u64m1(min_indices, pos, less, vl);
        vbool64_t greater = __riscv_vmflt_vv_f64m1_b64(max, d, vl);
        max = __riscv_vmerge_vvm_f64m1(max, d, greater, vl);
        max_indices = __riscv_vmerge_vvm_u64m1(max_indices, pos, greater, vl);
    }
    vfloat64m1_t id_max = __riscv_vfmv_v_f_f64m1(NK_F64_MAX, 1);
    nk_f64_t mn = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredmin_vs_f64m1_f64m1(min, id_max, vlmax));
    vbool64_t min_match = __riscv_vmfeq_vf_f64m1_b64(min, mn, vlmax);
    vuint64m1_t sentinel = __riscv_vmv_v_x_u64m1(NK_U64_MAX, vlmax);
    vuint64m1_t min_cands = __riscv_vmerge_vvm_u64m1(sentinel, min_indices, min_match, vlmax);
    vuint64m1_t id_umax = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    *min_value = mn,
    *min_index = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(__riscv_vredminu_vs_u64m1_u64m1(min_cands, id_umax, vlmax));
    vfloat64m1_t id_min = __riscv_vfmv_v_f_f64m1(NK_F64_MIN, 1);
    nk_f64_t mx = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredmax_vs_f64m1_f64m1(max, id_min, vlmax));
    vbool64_t max_match = __riscv_vmfeq_vf_f64m1_b64(max, mx, vlmax);
    vuint64m1_t max_cands = __riscv_vmerge_vvm_u64m1(sentinel, max_indices, max_match, vlmax);
    *max_value = mx,
    *max_index = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(__riscv_vredminu_vs_u64m1_u64m1(max_cands, id_umax, vlmax));
}

NK_INTERNAL void nk_reduce_minmax_f64_rvv_strided_(                //
    nk_f64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f64_t *min_value, nk_size_t *min_index,                     //
    nk_f64_t *max_value, nk_size_t *max_index) {
    nk_size_t vlmax = __riscv_vsetvlmax_e64m1();
    vfloat64m1_t min = __riscv_vfmv_v_f_f64m1(NK_F64_MAX, vlmax);
    vfloat64m1_t max = __riscv_vfmv_v_f_f64m1(NK_F64_MIN, vlmax);
    vuint64m1_t min_indices = __riscv_vmv_v_x_u64m1(0, vlmax);
    vuint64m1_t max_indices = __riscv_vmv_v_x_u64m1(0, vlmax);
    unsigned char const *ptr = (unsigned char const *)data;
    nk_size_t offset = 0;
    for (nk_size_t remaining = count, vl; remaining > 0; remaining -= vl, offset += vl, ptr += vl * stride_bytes) {
        vl = __riscv_vsetvl_e64m1(remaining);
        vfloat64m1_t d = __riscv_vlse64_v_f64m1((nk_f64_t const *)ptr, (nk_ssize_t)stride_bytes, vl);
        vuint64m1_t pos = __riscv_vadd_vx_u64m1(__riscv_vid_v_u64m1(vl), (nk_u64_t)offset, vl);
        vbool64_t less = __riscv_vmflt_vv_f64m1_b64(d, min, vl);
        min = __riscv_vmerge_vvm_f64m1(min, d, less, vl);
        min_indices = __riscv_vmerge_vvm_u64m1(min_indices, pos, less, vl);
        vbool64_t greater = __riscv_vmflt_vv_f64m1_b64(max, d, vl);
        max = __riscv_vmerge_vvm_f64m1(max, d, greater, vl);
        max_indices = __riscv_vmerge_vvm_u64m1(max_indices, pos, greater, vl);
    }
    vfloat64m1_t id_max = __riscv_vfmv_v_f_f64m1(NK_F64_MAX, 1);
    nk_f64_t mn = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredmin_vs_f64m1_f64m1(min, id_max, vlmax));
    vbool64_t min_match = __riscv_vmfeq_vf_f64m1_b64(min, mn, vlmax);
    vuint64m1_t sentinel = __riscv_vmv_v_x_u64m1(NK_U64_MAX, vlmax);
    vuint64m1_t min_cands = __riscv_vmerge_vvm_u64m1(sentinel, min_indices, min_match, vlmax);
    vuint64m1_t id_umax = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    *min_value = mn,
    *min_index = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(__riscv_vredminu_vs_u64m1_u64m1(min_cands, id_umax, vlmax));
    vfloat64m1_t id_min = __riscv_vfmv_v_f_f64m1(NK_F64_MIN, 1);
    nk_f64_t mx = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredmax_vs_f64m1_f64m1(max, id_min, vlmax));
    vbool64_t max_match = __riscv_vmfeq_vf_f64m1_b64(max, mx, vlmax);
    vuint64m1_t max_cands = __riscv_vmerge_vvm_u64m1(sentinel, max_indices, max_match, vlmax);
    *max_value = mx,
    *max_index = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(__riscv_vredminu_vs_u64m1_u64m1(max_cands, id_umax, vlmax));
}

NK_PUBLIC void nk_reduce_minmax_f64_rvv(                           //
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
        nk_reduce_minmax_f64_rvv_contiguous_(data, count, min_value, min_index, max_value, max_index);
    else nk_reduce_minmax_f64_rvv_strided_(data, count, stride_bytes, min_value, min_index, max_value, max_index);
}

NK_INTERNAL vuint8m1_t nk_fp8m1_to_comparable_u8m1_rvv_(vuint8m1_t raw_u8m1, nk_size_t vl) {
    // Convert FP8 (e4m3/e5m2) to comparable unsigned form (sign bit 7)
    // Positive (sign=0): XOR 0x80 → [0x80, 0xFF]
    // Negative (sign=1): Bitwise NOT → [0x00, 0x7F]
    vbool8_t is_negative_b8 = __riscv_vmsne_vx_u8m1_b8(__riscv_vand_vx_u8m1(raw_u8m1, 0x80, vl), 0, vl);
    vuint8m1_t flip_positive_u8m1 = __riscv_vxor_vx_u8m1(raw_u8m1, 0x80, vl);
    vuint8m1_t flip_negative_u8m1 = __riscv_vnot_v_u8m1(raw_u8m1, vl);
    return __riscv_vmerge_vvm_u8m1(flip_positive_u8m1, flip_negative_u8m1, is_negative_b8, vl);
}

NK_INTERNAL vuint8m1_t nk_comparable_to_fp8m1_rvv_(vuint8m1_t comparable_u8m1, nk_size_t vl) {
    // Reverse: if >= 0x80 (was positive), XOR; else NOT
    vbool8_t was_positive_b8 = __riscv_vmsgeu_vx_u8m1_b8(comparable_u8m1, 0x80, vl);
    vuint8m1_t from_positive_u8m1 = __riscv_vxor_vx_u8m1(comparable_u8m1, 0x80, vl);
    vuint8m1_t from_negative_u8m1 = __riscv_vnot_v_u8m1(comparable_u8m1, vl);
    return __riscv_vmerge_vvm_u8m1(from_negative_u8m1, from_positive_u8m1, was_positive_b8, vl);
}

NK_INTERNAL vuint8m1_t nk_fp6m1_to_comparable_u8m1_rvv_(vuint8m1_t raw_u8m1, nk_size_t vl) {
    // Convert FP6 (e2m3/e3m2) to comparable unsigned form (sign bit 5)
    // Positive (sign=0): XOR 0x20 → [0x20, 0x3F]
    // Negative (sign=1): NOT lower 6 bits → [0x00, 0x1F]
    vbool8_t is_negative_b8 = __riscv_vmsne_vx_u8m1_b8(__riscv_vand_vx_u8m1(raw_u8m1, 0x20, vl), 0, vl);
    vuint8m1_t flip_positive_u8m1 = __riscv_vxor_vx_u8m1(raw_u8m1, 0x20, vl);
    // NOT lower 6 bits: XOR with 0x3F then XOR back upper bits
    vuint8m1_t flip_negative_u8m1 = __riscv_vxor_vx_u8m1(__riscv_vxor_vx_u8m1(raw_u8m1, 0x3F, vl), 0x3F, vl);
    return __riscv_vmerge_vvm_u8m1(flip_positive_u8m1, flip_negative_u8m1, is_negative_b8, vl);
}

NK_INTERNAL vuint8m1_t nk_comparable_to_fp6m1_rvv_(vuint8m1_t comparable_u8m1, nk_size_t vl) {
    // Reverse: if >= 0x20 (was positive), XOR 0x20; else NOT lower 6 bits
    vbool8_t was_positive_b8 = __riscv_vmsgeu_vx_u8m1_b8(comparable_u8m1, 0x20, vl);
    vuint8m1_t from_positive_u8m1 = __riscv_vxor_vx_u8m1(comparable_u8m1, 0x20, vl);
    vuint8m1_t from_negative_u8m1 = __riscv_vxor_vx_u8m1(__riscv_vxor_vx_u8m1(comparable_u8m1, 0x3F, vl), 0x3F, vl);
    return __riscv_vmerge_vvm_u8m1(from_negative_u8m1, from_positive_u8m1, was_positive_b8, vl);
}

NK_INTERNAL void nk_reduce_moments_i8_rvv_contiguous_( //
    nk_i8_t const *data_ptr, nk_size_t count,          //
    nk_i64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e64m4();
    vint64m4_t sum_i64m4 = __riscv_vmv_v_x_i64m4(0, vlmax);
    vuint64m4_t sumsq_u64m4 = __riscv_vmv_v_x_u64m4(0, vlmax);

    for (nk_size_t vl; count > 0; count -= vl, data_ptr += vl) {
        vl = __riscv_vsetvl_e8m1(count);
        vint8m1_t data_i8m1 = __riscv_vle8_v_i8m1(data_ptr, vl);

        // Widen i8 → i16 → i32 → i64 for sum
        vint16m2_t data_i16m2 = __riscv_vsext_vf2_i16m2(data_i8m1, vl);
        vint32m4_t data_i32m4 = __riscv_vsext_vf2_i32m4(data_i16m2, vl);
        vint64m8_t data_i64m8 = __riscv_vsext_vf2_i64m8(data_i32m4, vl);

        // Accumulate sum (split m8 into two m4)
        sum_i64m4 = __riscv_vadd_vv_i64m4(sum_i64m4, __riscv_vget_v_i64m8_i64m4(data_i64m8, 0), vlmax);
        sum_i64m4 = __riscv_vadd_vv_i64m4(sum_i64m4, __riscv_vget_v_i64m8_i64m4(data_i64m8, 1), vlmax);

        // Sumsq: i8 × i8 → i16 (widening multiply)
        vint16m2_t squares_i16m2 = __riscv_vwmul_vv_i16m2(data_i8m1, data_i8m1, vl);
        // Widen i16 → u32 → u64
        vuint32m4_t squares_u32m4 = __riscv_vwcvtu_x_x_v_u32m4(__riscv_vreinterpret_v_i16m2_u16m2(squares_i16m2), vl);
        vuint64m8_t squares_u64m8 = __riscv_vwcvtu_x_x_v_u64m8(squares_u32m4, vl);

        sumsq_u64m4 = __riscv_vadd_vv_u64m4(sumsq_u64m4, __riscv_vget_v_u64m8_u64m4(squares_u64m8, 0), vlmax);
        sumsq_u64m4 = __riscv_vadd_vv_u64m4(sumsq_u64m4, __riscv_vget_v_u64m8_u64m4(squares_u64m8, 1), vlmax);
    }

    // Horizontal reduction
    vint64m1_t zero_i64m1 = __riscv_vmv_v_x_i64m1(0, 1);
    *sum_ptr = __riscv_vmv_x_s_i64m1_i64(__riscv_vredsum_vs_i64m4_i64m1(sum_i64m4, zero_i64m1, vlmax));

    vuint64m1_t zero_u64m1 = __riscv_vmv_v_x_u64m1(0, 1);
    *sumsq_ptr = __riscv_vmv_x_s_u64m1_u64(__riscv_vredsum_vs_u64m4_u64m1(sumsq_u64m4, zero_u64m1, vlmax));
}

NK_INTERNAL void nk_reduce_moments_i8_rvv_strided_(                   //
    nk_i8_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e64m4();
    vint64m4_t sum_i64m4 = __riscv_vmv_v_x_i64m4(0, vlmax);
    vuint64m4_t sumsq_u64m4 = __riscv_vmv_v_x_u64m4(0, vlmax);
    unsigned char const *ptr = (unsigned char const *)data_ptr;

    for (nk_size_t vl; count > 0; count -= vl, ptr += vl * stride_bytes) {
        vl = __riscv_vsetvl_e8m1(count);
        vint8m1_t data_i8m1 = __riscv_vlse8_v_i8m1((nk_i8_t const *)ptr, (nk_ssize_t)stride_bytes, vl);

        // Widen i8 → i16 → i32 → i64 for sum
        vint16m2_t data_i16m2 = __riscv_vsext_vf2_i16m2(data_i8m1, vl);
        vint32m4_t data_i32m4 = __riscv_vsext_vf2_i32m4(data_i16m2, vl);
        vint64m8_t data_i64m8 = __riscv_vsext_vf2_i64m8(data_i32m4, vl);

        // Accumulate sum (split m8 into two m4)
        sum_i64m4 = __riscv_vadd_vv_i64m4(sum_i64m4, __riscv_vget_v_i64m8_i64m4(data_i64m8, 0), vlmax);
        sum_i64m4 = __riscv_vadd_vv_i64m4(sum_i64m4, __riscv_vget_v_i64m8_i64m4(data_i64m8, 1), vlmax);

        // Sumsq: i8 × i8 → i16 (widening multiply)
        vint16m2_t squares_i16m2 = __riscv_vwmul_vv_i16m2(data_i8m1, data_i8m1, vl);
        // Widen i16 → u32 → u64
        vuint32m4_t squares_u32m4 = __riscv_vwcvtu_x_x_v_u32m4(__riscv_vreinterpret_v_i16m2_u16m2(squares_i16m2), vl);
        vuint64m8_t squares_u64m8 = __riscv_vwcvtu_x_x_v_u64m8(squares_u32m4, vl);

        sumsq_u64m4 = __riscv_vadd_vv_u64m4(sumsq_u64m4, __riscv_vget_v_u64m8_u64m4(squares_u64m8, 0), vlmax);
        sumsq_u64m4 = __riscv_vadd_vv_u64m4(sumsq_u64m4, __riscv_vget_v_u64m8_u64m4(squares_u64m8, 1), vlmax);
    }

    // Horizontal reduction
    vint64m1_t zero_i64m1 = __riscv_vmv_v_x_i64m1(0, 1);
    *sum_ptr = __riscv_vmv_x_s_i64m1_i64(__riscv_vredsum_vs_i64m4_i64m1(sum_i64m4, zero_i64m1, vlmax));

    vuint64m1_t zero_u64m1 = __riscv_vmv_v_x_u64m1(0, 1);
    *sumsq_ptr = __riscv_vmv_x_s_u64m1_u64(__riscv_vredsum_vs_u64m4_u64m1(sumsq_u64m4, zero_u64m1, vlmax));
}

NK_PUBLIC void nk_reduce_moments_i8_rvv(                              //
    nk_i8_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_i8_t);
    int aligned = (stride_bytes % sizeof(nk_i8_t) == 0);

    if (count == 0) *sum_ptr = 0, *sumsq_ptr = 0;
    else if (!aligned) { nk_reduce_moments_i8_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr); }
    else if (stride_elements == 1) { nk_reduce_moments_i8_rvv_contiguous_(data_ptr, count, sum_ptr, sumsq_ptr); }
    else { nk_reduce_moments_i8_rvv_strided_(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr); }
}

NK_INTERNAL void nk_reduce_minmax_i8_rvv_contiguous_( //
    nk_i8_t const *data_ptr, nk_size_t count,         //
    nk_i8_t *min_value_ptr, nk_size_t *min_index_ptr, //
    nk_i8_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e8m1();
    vint8m1_t min_i8m1 = __riscv_vmv_v_x_i8m1(NK_I8_MAX, vlmax);
    vint8m1_t max_i8m1 = __riscv_vmv_v_x_i8m1(NK_I8_MIN, vlmax);
    vuint64m8_t min_indices_u64m8 = __riscv_vmv_v_x_u64m8(0, vlmax);
    vuint64m8_t max_indices_u64m8 = __riscv_vmv_v_x_u64m8(0, vlmax);

    nk_size_t offset = 0;
    for (nk_size_t vl; count > 0; count -= vl, offset += vl, data_ptr += vl) {
        vl = __riscv_vsetvl_e8m1(count);
        vint8m1_t data_i8m1 = __riscv_vle8_v_i8m1(data_ptr, vl);

        // VID-based absolute indices
        vuint64m8_t pos_u64m8 = __riscv_vadd_vx_u64m8(__riscv_vid_v_u64m8(vl), (nk_u64_t)offset, vl);

        vbool8_t less_b8 = __riscv_vmslt_vv_i8m1_b8(data_i8m1, min_i8m1, vl);
        min_i8m1 = __riscv_vmerge_vvm_i8m1(min_i8m1, data_i8m1, less_b8, vl);
        min_indices_u64m8 = __riscv_vmerge_vvm_u64m8(min_indices_u64m8, pos_u64m8, less_b8, vl);

        vbool8_t greater_b8 = __riscv_vmslt_vv_i8m1_b8(max_i8m1, data_i8m1, vl);
        max_i8m1 = __riscv_vmerge_vvm_i8m1(max_i8m1, data_i8m1, greater_b8, vl);
        max_indices_u64m8 = __riscv_vmerge_vvm_u64m8(max_indices_u64m8, pos_u64m8, greater_b8, vl);
    }

    // Horizontal reduction for min
    vint8m1_t init_max_i8m1 = __riscv_vmv_v_x_i8m1(NK_I8_MAX, 1);
    nk_i8_t min_val = __riscv_vmv_x_s_i8m1_i8(__riscv_vredmin_vs_i8m1_i8m1(min_i8m1, init_max_i8m1, vlmax));
    vbool8_t min_match_b8 = __riscv_vmseq_vx_i8m1_b8(min_i8m1, min_val, vlmax);
    vuint64m8_t sentinel_u64m8 = __riscv_vmv_v_x_u64m8(NK_U64_MAX, vlmax);
    vuint64m8_t min_cands_u64m8 = __riscv_vmerge_vvm_u64m8(sentinel_u64m8, min_indices_u64m8, min_match_b8, vlmax);
    vuint64m1_t init_umax_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    *min_value_ptr = min_val;
    *min_index_ptr = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(
        __riscv_vredminu_vs_u64m8_u64m1(min_cands_u64m8, init_umax_u64m1, vlmax));

    // Horizontal reduction for max
    vint8m1_t init_min_i8m1 = __riscv_vmv_v_x_i8m1(NK_I8_MIN, 1);
    nk_i8_t max_val = __riscv_vmv_x_s_i8m1_i8(__riscv_vredmax_vs_i8m1_i8m1(max_i8m1, init_min_i8m1, vlmax));
    vbool8_t max_match_b8 = __riscv_vmseq_vx_i8m1_b8(max_i8m1, max_val, vlmax);
    vuint64m8_t max_cands_u64m8 = __riscv_vmerge_vvm_u64m8(sentinel_u64m8, max_indices_u64m8, max_match_b8, vlmax);
    *max_value_ptr = max_val;
    *max_index_ptr = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(
        __riscv_vredminu_vs_u64m8_u64m1(max_cands_u64m8, init_umax_u64m1, vlmax));
}

NK_INTERNAL void nk_reduce_minmax_i8_rvv_strided_(                    //
    nk_i8_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_i8_t *min_value_ptr, nk_size_t *min_index_ptr,                 //
    nk_i8_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e8m1();
    vint8m1_t min_i8m1 = __riscv_vmv_v_x_i8m1(NK_I8_MAX, vlmax);
    vint8m1_t max_i8m1 = __riscv_vmv_v_x_i8m1(NK_I8_MIN, vlmax);
    vuint64m8_t min_indices_u64m8 = __riscv_vmv_v_x_u64m8(0, vlmax);
    vuint64m8_t max_indices_u64m8 = __riscv_vmv_v_x_u64m8(0, vlmax);
    unsigned char const *ptr = (unsigned char const *)data_ptr;

    nk_size_t offset = 0;
    for (nk_size_t vl; count > 0; count -= vl, offset += vl, ptr += vl * stride_bytes) {
        vl = __riscv_vsetvl_e8m1(count);
        vint8m1_t data_i8m1 = __riscv_vlse8_v_i8m1((nk_i8_t const *)ptr, (nk_ssize_t)stride_bytes, vl);

        // VID-based absolute indices
        vuint64m8_t pos_u64m8 = __riscv_vadd_vx_u64m8(__riscv_vid_v_u64m8(vl), (nk_u64_t)offset, vl);

        vbool8_t less_b8 = __riscv_vmslt_vv_i8m1_b8(data_i8m1, min_i8m1, vl);
        min_i8m1 = __riscv_vmerge_vvm_i8m1(min_i8m1, data_i8m1, less_b8, vl);
        min_indices_u64m8 = __riscv_vmerge_vvm_u64m8(min_indices_u64m8, pos_u64m8, less_b8, vl);

        vbool8_t greater_b8 = __riscv_vmslt_vv_i8m1_b8(max_i8m1, data_i8m1, vl);
        max_i8m1 = __riscv_vmerge_vvm_i8m1(max_i8m1, data_i8m1, greater_b8, vl);
        max_indices_u64m8 = __riscv_vmerge_vvm_u64m8(max_indices_u64m8, pos_u64m8, greater_b8, vl);
    }

    // Horizontal reduction for min
    vint8m1_t init_max_i8m1 = __riscv_vmv_v_x_i8m1(NK_I8_MAX, 1);
    nk_i8_t min_val = __riscv_vmv_x_s_i8m1_i8(__riscv_vredmin_vs_i8m1_i8m1(min_i8m1, init_max_i8m1, vlmax));
    vbool8_t min_match_b8 = __riscv_vmseq_vx_i8m1_b8(min_i8m1, min_val, vlmax);
    vuint64m8_t sentinel_u64m8 = __riscv_vmv_v_x_u64m8(NK_U64_MAX, vlmax);
    vuint64m8_t min_cands_u64m8 = __riscv_vmerge_vvm_u64m8(sentinel_u64m8, min_indices_u64m8, min_match_b8, vlmax);
    vuint64m1_t init_umax_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    *min_value_ptr = min_val;
    *min_index_ptr = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(
        __riscv_vredminu_vs_u64m8_u64m1(min_cands_u64m8, init_umax_u64m1, vlmax));

    // Horizontal reduction for max
    vint8m1_t init_min_i8m1 = __riscv_vmv_v_x_i8m1(NK_I8_MIN, 1);
    nk_i8_t max_val = __riscv_vmv_x_s_i8m1_i8(__riscv_vredmax_vs_i8m1_i8m1(max_i8m1, init_min_i8m1, vlmax));
    vbool8_t max_match_b8 = __riscv_vmseq_vx_i8m1_b8(max_i8m1, max_val, vlmax);
    vuint64m8_t max_cands_u64m8 = __riscv_vmerge_vvm_u64m8(sentinel_u64m8, max_indices_u64m8, max_match_b8, vlmax);
    *max_value_ptr = max_val;
    *max_index_ptr = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(
        __riscv_vredminu_vs_u64m8_u64m1(max_cands_u64m8, init_umax_u64m1, vlmax));
}

NK_PUBLIC void nk_reduce_minmax_i8_rvv(                               //
    nk_i8_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_i8_t *min_value_ptr, nk_size_t *min_index_ptr,                 //
    nk_i8_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_i8_t);
    int aligned = (stride_bytes % sizeof(nk_i8_t) == 0);

    if (count == 0)
        *min_value_ptr = NK_I8_MAX, *min_index_ptr = NK_SIZE_MAX, *max_value_ptr = NK_I8_MIN,
        *max_index_ptr = NK_SIZE_MAX;
    else if (!aligned)
        nk_reduce_minmax_i8_serial(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                   max_index_ptr);
    else if (stride_elements == 1)
        nk_reduce_minmax_i8_rvv_contiguous_(data_ptr, count, min_value_ptr, min_index_ptr, max_value_ptr,
                                            max_index_ptr);
    else
        nk_reduce_minmax_i8_rvv_strided_(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                         max_index_ptr);
}

NK_INTERNAL void nk_reduce_moments_u8_rvv_contiguous_( //
    nk_u8_t const *data_ptr, nk_size_t count,          //
    nk_u64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e64m4();
    vuint64m4_t sum_u64m4 = __riscv_vmv_v_x_u64m4(0, vlmax);
    vuint64m4_t sumsq_u64m4 = __riscv_vmv_v_x_u64m4(0, vlmax);

    for (nk_size_t vl; count > 0; count -= vl, data_ptr += vl) {
        vl = __riscv_vsetvl_e8m1(count);
        vuint8m1_t data_u8m1 = __riscv_vle8_v_u8m1(data_ptr, vl);

        // Widen u8 → u16 → u32 → u64 for sum
        vuint16m2_t data_u16m2 = __riscv_vzext_vf2_u16m2(data_u8m1, vl);
        vuint32m4_t data_u32m4 = __riscv_vzext_vf2_u32m4(data_u16m2, vl);
        vuint64m8_t data_u64m8 = __riscv_vzext_vf2_u64m8(data_u32m4, vl);

        // Accumulate sum (split m8 into two m4)
        sum_u64m4 = __riscv_vadd_vv_u64m4(sum_u64m4, __riscv_vget_v_u64m8_u64m4(data_u64m8, 0), vlmax);
        sum_u64m4 = __riscv_vadd_vv_u64m4(sum_u64m4, __riscv_vget_v_u64m8_u64m4(data_u64m8, 1), vlmax);

        // Sumsq: u8 × u8 → u16 (widening multiply)
        vuint16m2_t squares_u16m2 = __riscv_vwmulu_vv_u16m2(data_u8m1, data_u8m1, vl);
        // Widen u16 → u32 → u64
        vuint32m4_t squares_u32m4 = __riscv_vzext_vf2_u32m4(squares_u16m2, vl);
        vuint64m8_t squares_u64m8 = __riscv_vzext_vf2_u64m8(squares_u32m4, vl);

        sumsq_u64m4 = __riscv_vadd_vv_u64m4(sumsq_u64m4, __riscv_vget_v_u64m8_u64m4(squares_u64m8, 0), vlmax);
        sumsq_u64m4 = __riscv_vadd_vv_u64m4(sumsq_u64m4, __riscv_vget_v_u64m8_u64m4(squares_u64m8, 1), vlmax);
    }

    // Horizontal reduction
    vuint64m1_t zero_u64m1 = __riscv_vmv_v_x_u64m1(0, 1);
    *sum_ptr = __riscv_vmv_x_s_u64m1_u64(__riscv_vredsum_vs_u64m4_u64m1(sum_u64m4, zero_u64m1, vlmax)),
    *sumsq_ptr = __riscv_vmv_x_s_u64m1_u64(__riscv_vredsum_vs_u64m4_u64m1(sumsq_u64m4, zero_u64m1, vlmax));
}

NK_INTERNAL void nk_reduce_moments_u8_rvv_strided_(                   //
    nk_u8_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e64m4();
    vuint64m4_t sum_u64m4 = __riscv_vmv_v_x_u64m4(0, vlmax);
    vuint64m4_t sumsq_u64m4 = __riscv_vmv_v_x_u64m4(0, vlmax);
    unsigned char const *ptr = (unsigned char const *)data_ptr;

    for (nk_size_t vl; count > 0; count -= vl, ptr += vl * stride_bytes) {
        vl = __riscv_vsetvl_e8m1(count);
        vuint8m1_t data_u8m1 = __riscv_vlse8_v_u8m1((nk_u8_t const *)ptr, (nk_ssize_t)stride_bytes, vl);

        // Widen u8 → u16 → u32 → u64 for sum
        vuint16m2_t data_u16m2 = __riscv_vzext_vf2_u16m2(data_u8m1, vl);
        vuint32m4_t data_u32m4 = __riscv_vzext_vf2_u32m4(data_u16m2, vl);
        vuint64m8_t data_u64m8 = __riscv_vzext_vf2_u64m8(data_u32m4, vl);

        // Accumulate sum (split m8 into two m4)
        sum_u64m4 = __riscv_vadd_vv_u64m4(sum_u64m4, __riscv_vget_v_u64m8_u64m4(data_u64m8, 0), vlmax);
        sum_u64m4 = __riscv_vadd_vv_u64m4(sum_u64m4, __riscv_vget_v_u64m8_u64m4(data_u64m8, 1), vlmax);

        // Sumsq: u8 × u8 → u16 (widening multiply)
        vuint16m2_t squares_u16m2 = __riscv_vwmulu_vv_u16m2(data_u8m1, data_u8m1, vl);
        // Widen u16 → u32 → u64
        vuint32m4_t squares_u32m4 = __riscv_vzext_vf2_u32m4(squares_u16m2, vl);
        vuint64m8_t squares_u64m8 = __riscv_vzext_vf2_u64m8(squares_u32m4, vl);

        sumsq_u64m4 = __riscv_vadd_vv_u64m4(sumsq_u64m4, __riscv_vget_v_u64m8_u64m4(squares_u64m8, 0), vlmax);
        sumsq_u64m4 = __riscv_vadd_vv_u64m4(sumsq_u64m4, __riscv_vget_v_u64m8_u64m4(squares_u64m8, 1), vlmax);
    }

    // Horizontal reduction
    vuint64m1_t zero_u64m1 = __riscv_vmv_v_x_u64m1(0, 1);
    *sum_ptr = __riscv_vmv_x_s_u64m1_u64(__riscv_vredsum_vs_u64m4_u64m1(sum_u64m4, zero_u64m1, vlmax)),
    *sumsq_ptr = __riscv_vmv_x_s_u64m1_u64(__riscv_vredsum_vs_u64m4_u64m1(sumsq_u64m4, zero_u64m1, vlmax));
}

NK_PUBLIC void nk_reduce_moments_u8_rvv(                              //
    nk_u8_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_u8_t);
    int aligned = (stride_bytes % sizeof(nk_u8_t) == 0);

    if (count == 0) *sum_ptr = 0, *sumsq_ptr = 0;
    else if (!aligned) { nk_reduce_moments_u8_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr); }
    else if (stride_elements == 1) { nk_reduce_moments_u8_rvv_contiguous_(data_ptr, count, sum_ptr, sumsq_ptr); }
    else { nk_reduce_moments_u8_rvv_strided_(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr); }
}

NK_INTERNAL void nk_reduce_minmax_u8_rvv_contiguous_( //
    nk_u8_t const *data_ptr, nk_size_t count,         //
    nk_u8_t *min_value_ptr, nk_size_t *min_index_ptr, //
    nk_u8_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e8m1();
    vuint8m1_t min_u8m1 = __riscv_vmv_v_x_u8m1(NK_U8_MAX, vlmax);
    vuint8m1_t max_u8m1 = __riscv_vmv_v_x_u8m1(NK_U8_MIN, vlmax);
    vuint64m8_t min_indices_u64m8 = __riscv_vmv_v_x_u64m8(0, vlmax);
    vuint64m8_t max_indices_u64m8 = __riscv_vmv_v_x_u64m8(0, vlmax);

    nk_size_t offset = 0;
    for (nk_size_t vl; count > 0; count -= vl, offset += vl, data_ptr += vl) {
        vl = __riscv_vsetvl_e8m1(count);
        vuint8m1_t data_u8m1 = __riscv_vle8_v_u8m1(data_ptr, vl);

        // VID-based absolute indices
        vuint64m8_t pos_u64m8 = __riscv_vadd_vx_u64m8(__riscv_vid_v_u64m8(vl), (nk_u64_t)offset, vl);

        vbool8_t less_b8 = __riscv_vmsltu_vv_u8m1_b8(data_u8m1, min_u8m1, vl);
        min_u8m1 = __riscv_vmerge_vvm_u8m1(min_u8m1, data_u8m1, less_b8, vl);
        min_indices_u64m8 = __riscv_vmerge_vvm_u64m8(min_indices_u64m8, pos_u64m8, less_b8, vl);

        vbool8_t greater_b8 = __riscv_vmsltu_vv_u8m1_b8(max_u8m1, data_u8m1, vl);
        max_u8m1 = __riscv_vmerge_vvm_u8m1(max_u8m1, data_u8m1, greater_b8, vl);
        max_indices_u64m8 = __riscv_vmerge_vvm_u64m8(max_indices_u64m8, pos_u64m8, greater_b8, vl);
    }

    // Horizontal reduction for min
    vuint8m1_t init_max_u8m1 = __riscv_vmv_v_x_u8m1(NK_U8_MAX, 1);
    nk_u8_t min_val = __riscv_vmv_x_s_u8m1_u8(__riscv_vredminu_vs_u8m1_u8m1(min_u8m1, init_max_u8m1, vlmax));
    vbool8_t min_match_b8 = __riscv_vmseq_vx_u8m1_b8(min_u8m1, min_val, vlmax);
    vuint64m8_t sentinel_u64m8 = __riscv_vmv_v_x_u64m8(NK_U64_MAX, vlmax);
    vuint64m8_t min_cands_u64m8 = __riscv_vmerge_vvm_u64m8(sentinel_u64m8, min_indices_u64m8, min_match_b8, vlmax);
    vuint64m1_t init_umax_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    *min_value_ptr = min_val;
    *min_index_ptr = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(
        __riscv_vredminu_vs_u64m8_u64m1(min_cands_u64m8, init_umax_u64m1, vlmax));

    // Horizontal reduction for max
    vuint8m1_t init_min_u8m1 = __riscv_vmv_v_x_u8m1(NK_U8_MIN, 1);
    nk_u8_t max_val = __riscv_vmv_x_s_u8m1_u8(__riscv_vredmaxu_vs_u8m1_u8m1(max_u8m1, init_min_u8m1, vlmax));
    vbool8_t max_match_b8 = __riscv_vmseq_vx_u8m1_b8(max_u8m1, max_val, vlmax);
    vuint64m8_t max_cands_u64m8 = __riscv_vmerge_vvm_u64m8(sentinel_u64m8, max_indices_u64m8, max_match_b8, vlmax);
    *max_value_ptr = max_val;
    *max_index_ptr = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(
        __riscv_vredminu_vs_u64m8_u64m1(max_cands_u64m8, init_umax_u64m1, vlmax));
}

NK_INTERNAL void nk_reduce_minmax_u8_rvv_strided_(                    //
    nk_u8_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_u8_t *min_value_ptr, nk_size_t *min_index_ptr,                 //
    nk_u8_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e8m1();
    vuint8m1_t min_u8m1 = __riscv_vmv_v_x_u8m1(NK_U8_MAX, vlmax);
    vuint8m1_t max_u8m1 = __riscv_vmv_v_x_u8m1(NK_U8_MIN, vlmax);
    vuint64m8_t min_indices_u64m8 = __riscv_vmv_v_x_u64m8(0, vlmax);
    vuint64m8_t max_indices_u64m8 = __riscv_vmv_v_x_u64m8(0, vlmax);
    unsigned char const *ptr = (unsigned char const *)data_ptr;

    nk_size_t offset = 0;
    for (nk_size_t vl; count > 0; count -= vl, offset += vl, ptr += vl * stride_bytes) {
        vl = __riscv_vsetvl_e8m1(count);
        vuint8m1_t data_u8m1 = __riscv_vlse8_v_u8m1((nk_u8_t const *)ptr, (nk_ssize_t)stride_bytes, vl);

        // VID-based absolute indices
        vuint64m8_t pos_u64m8 = __riscv_vadd_vx_u64m8(__riscv_vid_v_u64m8(vl), (nk_u64_t)offset, vl);

        vbool8_t less_b8 = __riscv_vmsltu_vv_u8m1_b8(data_u8m1, min_u8m1, vl);
        min_u8m1 = __riscv_vmerge_vvm_u8m1(min_u8m1, data_u8m1, less_b8, vl);
        min_indices_u64m8 = __riscv_vmerge_vvm_u64m8(min_indices_u64m8, pos_u64m8, less_b8, vl);

        vbool8_t greater_b8 = __riscv_vmsltu_vv_u8m1_b8(max_u8m1, data_u8m1, vl);
        max_u8m1 = __riscv_vmerge_vvm_u8m1(max_u8m1, data_u8m1, greater_b8, vl);
        max_indices_u64m8 = __riscv_vmerge_vvm_u64m8(max_indices_u64m8, pos_u64m8, greater_b8, vl);
    }

    // Horizontal reduction for min
    vuint8m1_t init_max_u8m1 = __riscv_vmv_v_x_u8m1(NK_U8_MAX, 1);
    nk_u8_t min_val = __riscv_vmv_x_s_u8m1_u8(__riscv_vredminu_vs_u8m1_u8m1(min_u8m1, init_max_u8m1, vlmax));
    vbool8_t min_match_b8 = __riscv_vmseq_vx_u8m1_b8(min_u8m1, min_val, vlmax);
    vuint64m8_t sentinel_u64m8 = __riscv_vmv_v_x_u64m8(NK_U64_MAX, vlmax);
    vuint64m8_t min_cands_u64m8 = __riscv_vmerge_vvm_u64m8(sentinel_u64m8, min_indices_u64m8, min_match_b8, vlmax);
    vuint64m1_t init_umax_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    *min_value_ptr = min_val;
    *min_index_ptr = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(
        __riscv_vredminu_vs_u64m8_u64m1(min_cands_u64m8, init_umax_u64m1, vlmax));

    // Horizontal reduction for max
    vuint8m1_t init_min_u8m1 = __riscv_vmv_v_x_u8m1(NK_U8_MIN, 1);
    nk_u8_t max_val = __riscv_vmv_x_s_u8m1_u8(__riscv_vredmaxu_vs_u8m1_u8m1(max_u8m1, init_min_u8m1, vlmax));
    vbool8_t max_match_b8 = __riscv_vmseq_vx_u8m1_b8(max_u8m1, max_val, vlmax);
    vuint64m8_t max_cands_u64m8 = __riscv_vmerge_vvm_u64m8(sentinel_u64m8, max_indices_u64m8, max_match_b8, vlmax);
    *max_value_ptr = max_val;
    *max_index_ptr = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(
        __riscv_vredminu_vs_u64m8_u64m1(max_cands_u64m8, init_umax_u64m1, vlmax));
}

NK_PUBLIC void nk_reduce_minmax_u8_rvv(                               //
    nk_u8_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_u8_t *min_value_ptr, nk_size_t *min_index_ptr,                 //
    nk_u8_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_u8_t);
    int aligned = (stride_bytes % sizeof(nk_u8_t) == 0);

    if (count == 0)
        *min_value_ptr = NK_U8_MAX, *min_index_ptr = NK_SIZE_MAX, *max_value_ptr = NK_U8_MIN,
        *max_index_ptr = NK_SIZE_MAX;
    else if (!aligned)
        nk_reduce_minmax_u8_serial(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                   max_index_ptr);
    else if (stride_elements == 1)
        nk_reduce_minmax_u8_rvv_contiguous_(data_ptr, count, min_value_ptr, min_index_ptr, max_value_ptr,
                                            max_index_ptr);
    else
        nk_reduce_minmax_u8_rvv_strided_(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                         max_index_ptr);
}

NK_INTERNAL void nk_reduce_moments_i16_rvv_contiguous_( //
    nk_i16_t const *data_ptr, nk_size_t count,          //
    nk_i64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e64m4();
    vint64m4_t sum_i64m4 = __riscv_vmv_v_x_i64m4(0, vlmax);
    vuint64m4_t sumsq_u64m4 = __riscv_vmv_v_x_u64m4(0, vlmax);

    for (nk_size_t vl; count > 0; count -= vl, data_ptr += vl) {
        vl = __riscv_vsetvl_e16m1(count);
        vint16m1_t data_i16m1 = __riscv_vle16_v_i16m1(data_ptr, vl);

        // Widen i16 → i32 → i64 for sum
        vint32m2_t data_i32m2 = __riscv_vsext_vf2_i32m2(data_i16m1, vl);
        vint64m4_t data_i64m4 = __riscv_vsext_vf2_i64m4(data_i32m2, vl);
        sum_i64m4 = __riscv_vadd_vv_i64m4(sum_i64m4, data_i64m4, vl);

        // Sumsq: i16 × i16 → i32 (widening multiply)
        vint32m2_t squares_i32m2 = __riscv_vwmul_vv_i32m2(data_i16m1, data_i16m1, vl);
        // Widen i32 → u64
        vuint64m4_t squares_u64m4 = __riscv_vwcvtu_x_x_v_u64m4(__riscv_vreinterpret_v_i32m2_u32m2(squares_i32m2), vl);
        sumsq_u64m4 = __riscv_vadd_vv_u64m4(sumsq_u64m4, squares_u64m4, vl);
    }

    // Horizontal reduction
    vint64m1_t zero_i64m1 = __riscv_vmv_v_x_i64m1(0, 1);
    *sum_ptr = __riscv_vmv_x_s_i64m1_i64(__riscv_vredsum_vs_i64m4_i64m1(sum_i64m4, zero_i64m1, vlmax));

    vuint64m1_t zero_u64m1 = __riscv_vmv_v_x_u64m1(0, 1);
    *sumsq_ptr = __riscv_vmv_x_s_u64m1_u64(__riscv_vredsum_vs_u64m4_u64m1(sumsq_u64m4, zero_u64m1, vlmax));
}

NK_INTERNAL void nk_reduce_moments_i16_rvv_strided_(                   //
    nk_i16_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e64m4();
    vint64m4_t sum_i64m4 = __riscv_vmv_v_x_i64m4(0, vlmax);
    vuint64m4_t sumsq_u64m4 = __riscv_vmv_v_x_u64m4(0, vlmax);
    unsigned char const *ptr = (unsigned char const *)data_ptr;

    for (nk_size_t vl; count > 0; count -= vl, ptr += vl * stride_bytes) {
        vl = __riscv_vsetvl_e16m1(count);
        vint16m1_t data_i16m1 = __riscv_vlse16_v_i16m1((nk_i16_t const *)ptr, (nk_ssize_t)stride_bytes, vl);

        // Widen i16 → i32 → i64 for sum
        vint32m2_t data_i32m2 = __riscv_vsext_vf2_i32m2(data_i16m1, vl);
        vint64m4_t data_i64m4 = __riscv_vsext_vf2_i64m4(data_i32m2, vl);
        sum_i64m4 = __riscv_vadd_vv_i64m4(sum_i64m4, data_i64m4, vl);

        // Sumsq: i16 × i16 → i32 (widening multiply)
        vint32m2_t squares_i32m2 = __riscv_vwmul_vv_i32m2(data_i16m1, data_i16m1, vl);
        // Widen i32 → u64
        vuint64m4_t squares_u64m4 = __riscv_vwcvtu_x_x_v_u64m4(__riscv_vreinterpret_v_i32m2_u32m2(squares_i32m2), vl);
        sumsq_u64m4 = __riscv_vadd_vv_u64m4(sumsq_u64m4, squares_u64m4, vl);
    }

    // Horizontal reduction
    vint64m1_t zero_i64m1 = __riscv_vmv_v_x_i64m1(0, 1);
    *sum_ptr = __riscv_vmv_x_s_i64m1_i64(__riscv_vredsum_vs_i64m4_i64m1(sum_i64m4, zero_i64m1, vlmax));

    vuint64m1_t zero_u64m1 = __riscv_vmv_v_x_u64m1(0, 1);
    *sumsq_ptr = __riscv_vmv_x_s_u64m1_u64(__riscv_vredsum_vs_u64m4_u64m1(sumsq_u64m4, zero_u64m1, vlmax));
}

NK_PUBLIC void nk_reduce_moments_i16_rvv(                              //
    nk_i16_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_i16_t);
    int aligned = (stride_bytes % sizeof(nk_i16_t) == 0);

    if (count == 0) *sum_ptr = 0, *sumsq_ptr = 0;
    else if (!aligned) { nk_reduce_moments_i16_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr); }
    else if (stride_elements == 1) { nk_reduce_moments_i16_rvv_contiguous_(data_ptr, count, sum_ptr, sumsq_ptr); }
    else { nk_reduce_moments_i16_rvv_strided_(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr); }
}

NK_INTERNAL void nk_reduce_minmax_i16_rvv_contiguous_( //
    nk_i16_t const *data_ptr, nk_size_t count,         //
    nk_i16_t *min_value_ptr, nk_size_t *min_index_ptr, //
    nk_i16_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e16m1();
    vint16m1_t min_i16m1 = __riscv_vmv_v_x_i16m1(NK_I16_MAX, vlmax);
    vint16m1_t max_i16m1 = __riscv_vmv_v_x_i16m1(NK_I16_MIN, vlmax);
    vuint64m4_t min_indices_u64m4 = __riscv_vmv_v_x_u64m4(0, vlmax);
    vuint64m4_t max_indices_u64m4 = __riscv_vmv_v_x_u64m4(0, vlmax);

    nk_size_t offset = 0;
    for (nk_size_t vl; count > 0; count -= vl, offset += vl, data_ptr += vl) {
        vl = __riscv_vsetvl_e16m1(count);
        vint16m1_t data_i16m1 = __riscv_vle16_v_i16m1(data_ptr, vl);
        vuint64m4_t pos_u64m4 = __riscv_vadd_vx_u64m4(__riscv_vid_v_u64m4(vl), (nk_u64_t)offset, vl);

        vbool16_t less_b16 = __riscv_vmslt_vv_i16m1_b16(data_i16m1, min_i16m1, vl);
        min_i16m1 = __riscv_vmerge_vvm_i16m1(min_i16m1, data_i16m1, less_b16, vl);
        min_indices_u64m4 = __riscv_vmerge_vvm_u64m4(min_indices_u64m4, pos_u64m4, less_b16, vl);

        vbool16_t greater_b16 = __riscv_vmslt_vv_i16m1_b16(max_i16m1, data_i16m1, vl);
        max_i16m1 = __riscv_vmerge_vvm_i16m1(max_i16m1, data_i16m1, greater_b16, vl);
        max_indices_u64m4 = __riscv_vmerge_vvm_u64m4(max_indices_u64m4, pos_u64m4, greater_b16, vl);
    }

    // Horizontal reduction for min
    vint16m1_t init_max_i16m1 = __riscv_vmv_v_x_i16m1(NK_I16_MAX, 1);
    nk_i16_t min_val = __riscv_vmv_x_s_i16m1_i16(__riscv_vredmin_vs_i16m1_i16m1(min_i16m1, init_max_i16m1, vlmax));
    vbool16_t min_match_b16 = __riscv_vmseq_vx_i16m1_b16(min_i16m1, min_val, vlmax);
    vuint64m4_t sentinel_u64m4 = __riscv_vmv_v_x_u64m4(NK_U64_MAX, vlmax);
    vuint64m4_t min_cands_u64m4 = __riscv_vmerge_vvm_u64m4(sentinel_u64m4, min_indices_u64m4, min_match_b16, vlmax);
    vuint64m1_t init_umax_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    *min_value_ptr = min_val;
    *min_index_ptr = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(
        __riscv_vredminu_vs_u64m4_u64m1(min_cands_u64m4, init_umax_u64m1, vlmax));

    // Horizontal reduction for max
    vint16m1_t init_min_i16m1 = __riscv_vmv_v_x_i16m1(NK_I16_MIN, 1);
    nk_i16_t max_val = __riscv_vmv_x_s_i16m1_i16(__riscv_vredmax_vs_i16m1_i16m1(max_i16m1, init_min_i16m1, vlmax));
    vbool16_t max_match_b16 = __riscv_vmseq_vx_i16m1_b16(max_i16m1, max_val, vlmax);
    vuint64m4_t max_cands_u64m4 = __riscv_vmerge_vvm_u64m4(sentinel_u64m4, max_indices_u64m4, max_match_b16, vlmax);
    *max_value_ptr = max_val;
    *max_index_ptr = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(
        __riscv_vredminu_vs_u64m4_u64m1(max_cands_u64m4, init_umax_u64m1, vlmax));
}

NK_INTERNAL void nk_reduce_minmax_i16_rvv_strided_(                    //
    nk_i16_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_i16_t *min_value_ptr, nk_size_t *min_index_ptr,                 //
    nk_i16_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e16m1();
    vint16m1_t min_i16m1 = __riscv_vmv_v_x_i16m1(NK_I16_MAX, vlmax);
    vint16m1_t max_i16m1 = __riscv_vmv_v_x_i16m1(NK_I16_MIN, vlmax);
    vuint64m4_t min_indices_u64m4 = __riscv_vmv_v_x_u64m4(0, vlmax);
    vuint64m4_t max_indices_u64m4 = __riscv_vmv_v_x_u64m4(0, vlmax);
    unsigned char const *ptr = (unsigned char const *)data_ptr;

    nk_size_t offset = 0;
    for (nk_size_t vl; count > 0; count -= vl, offset += vl, ptr += vl * stride_bytes) {
        vl = __riscv_vsetvl_e16m1(count);
        vint16m1_t data_i16m1 = __riscv_vlse16_v_i16m1((nk_i16_t const *)ptr, (nk_ssize_t)stride_bytes, vl);
        vuint64m4_t pos_u64m4 = __riscv_vadd_vx_u64m4(__riscv_vid_v_u64m4(vl), (nk_u64_t)offset, vl);

        vbool16_t less_b16 = __riscv_vmslt_vv_i16m1_b16(data_i16m1, min_i16m1, vl);
        min_i16m1 = __riscv_vmerge_vvm_i16m1(min_i16m1, data_i16m1, less_b16, vl);
        min_indices_u64m4 = __riscv_vmerge_vvm_u64m4(min_indices_u64m4, pos_u64m4, less_b16, vl);

        vbool16_t greater_b16 = __riscv_vmslt_vv_i16m1_b16(max_i16m1, data_i16m1, vl);
        max_i16m1 = __riscv_vmerge_vvm_i16m1(max_i16m1, data_i16m1, greater_b16, vl);
        max_indices_u64m4 = __riscv_vmerge_vvm_u64m4(max_indices_u64m4, pos_u64m4, greater_b16, vl);
    }

    // Horizontal reduction for min
    vint16m1_t init_max_i16m1 = __riscv_vmv_v_x_i16m1(NK_I16_MAX, 1);
    nk_i16_t min_val = __riscv_vmv_x_s_i16m1_i16(__riscv_vredmin_vs_i16m1_i16m1(min_i16m1, init_max_i16m1, vlmax));
    vbool16_t min_match_b16 = __riscv_vmseq_vx_i16m1_b16(min_i16m1, min_val, vlmax);
    vuint64m4_t sentinel_u64m4 = __riscv_vmv_v_x_u64m4(NK_U64_MAX, vlmax);
    vuint64m4_t min_cands_u64m4 = __riscv_vmerge_vvm_u64m4(sentinel_u64m4, min_indices_u64m4, min_match_b16, vlmax);
    vuint64m1_t init_umax_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    *min_value_ptr = min_val;
    *min_index_ptr = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(
        __riscv_vredminu_vs_u64m4_u64m1(min_cands_u64m4, init_umax_u64m1, vlmax));

    // Horizontal reduction for max
    vint16m1_t init_min_i16m1 = __riscv_vmv_v_x_i16m1(NK_I16_MIN, 1);
    nk_i16_t max_val = __riscv_vmv_x_s_i16m1_i16(__riscv_vredmax_vs_i16m1_i16m1(max_i16m1, init_min_i16m1, vlmax));
    vbool16_t max_match_b16 = __riscv_vmseq_vx_i16m1_b16(max_i16m1, max_val, vlmax);
    vuint64m4_t max_cands_u64m4 = __riscv_vmerge_vvm_u64m4(sentinel_u64m4, max_indices_u64m4, max_match_b16, vlmax);
    *max_value_ptr = max_val;
    *max_index_ptr = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(
        __riscv_vredminu_vs_u64m4_u64m1(max_cands_u64m4, init_umax_u64m1, vlmax));
}

NK_PUBLIC void nk_reduce_minmax_i16_rvv(                               //
    nk_i16_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_i16_t *min_value_ptr, nk_size_t *min_index_ptr,                 //
    nk_i16_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_i16_t);
    int aligned = (stride_bytes % sizeof(nk_i16_t) == 0);

    if (count == 0)
        *min_value_ptr = NK_I16_MAX, *min_index_ptr = NK_SIZE_MAX, *max_value_ptr = NK_I16_MIN,
        *max_index_ptr = NK_SIZE_MAX;
    else if (!aligned)
        nk_reduce_minmax_i16_serial(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                    max_index_ptr);
    else if (stride_elements == 1)
        nk_reduce_minmax_i16_rvv_contiguous_(data_ptr, count, min_value_ptr, min_index_ptr, max_value_ptr,
                                             max_index_ptr);
    else
        nk_reduce_minmax_i16_rvv_strided_(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                          max_index_ptr);
}

NK_INTERNAL void nk_reduce_moments_u16_rvv_contiguous_( //
    nk_u16_t const *data_ptr, nk_size_t count,          //
    nk_u64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e64m4();
    vuint64m4_t sum_u64m4 = __riscv_vmv_v_x_u64m4(0, vlmax);
    vuint64m4_t sumsq_u64m4 = __riscv_vmv_v_x_u64m4(0, vlmax);

    for (nk_size_t vl; count > 0; count -= vl, data_ptr += vl) {
        vl = __riscv_vsetvl_e16m1(count);
        vuint16m1_t data_u16m1 = __riscv_vle16_v_u16m1(data_ptr, vl);

        // Widen u16 → u32 → u64 for sum
        vuint32m2_t data_u32m2 = __riscv_vzext_vf2_u32m2(data_u16m1, vl);
        vuint64m4_t data_u64m4 = __riscv_vzext_vf2_u64m4(data_u32m2, vl);
        sum_u64m4 = __riscv_vadd_vv_u64m4(sum_u64m4, data_u64m4, vl);

        // Sumsq: u16 × u16 → u32 (widening multiply)
        vuint32m2_t squares_u32m2 = __riscv_vwmulu_vv_u32m2(data_u16m1, data_u16m1, vl);
        // Widen u32 → u64
        vuint64m4_t squares_u64m4 = __riscv_vzext_vf2_u64m4(squares_u32m2, vl);
        sumsq_u64m4 = __riscv_vadd_vv_u64m4(sumsq_u64m4, squares_u64m4, vl);
    }

    // Horizontal reduction
    vuint64m1_t zero_u64m1 = __riscv_vmv_v_x_u64m1(0, 1);
    *sum_ptr = __riscv_vmv_x_s_u64m1_u64(__riscv_vredsum_vs_u64m4_u64m1(sum_u64m4, zero_u64m1, vlmax)),
    *sumsq_ptr = __riscv_vmv_x_s_u64m1_u64(__riscv_vredsum_vs_u64m4_u64m1(sumsq_u64m4, zero_u64m1, vlmax));
}

NK_INTERNAL void nk_reduce_moments_u16_rvv_strided_(                   //
    nk_u16_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e64m4();
    vuint64m4_t sum_u64m4 = __riscv_vmv_v_x_u64m4(0, vlmax);
    vuint64m4_t sumsq_u64m4 = __riscv_vmv_v_x_u64m4(0, vlmax);
    unsigned char const *ptr = (unsigned char const *)data_ptr;

    for (nk_size_t vl; count > 0; count -= vl, ptr += vl * stride_bytes) {
        vl = __riscv_vsetvl_e16m1(count);
        vuint16m1_t data_u16m1 = __riscv_vlse16_v_u16m1((nk_u16_t const *)ptr, (nk_ssize_t)stride_bytes, vl);

        // Widen u16 → u32 → u64 for sum
        vuint32m2_t data_u32m2 = __riscv_vzext_vf2_u32m2(data_u16m1, vl);
        vuint64m4_t data_u64m4 = __riscv_vzext_vf2_u64m4(data_u32m2, vl);
        sum_u64m4 = __riscv_vadd_vv_u64m4(sum_u64m4, data_u64m4, vl);

        // Sumsq: u16 × u16 → u32 (widening multiply)
        vuint32m2_t squares_u32m2 = __riscv_vwmulu_vv_u32m2(data_u16m1, data_u16m1, vl);
        // Widen u32 → u64
        vuint64m4_t squares_u64m4 = __riscv_vzext_vf2_u64m4(squares_u32m2, vl);
        sumsq_u64m4 = __riscv_vadd_vv_u64m4(sumsq_u64m4, squares_u64m4, vl);
    }

    // Horizontal reduction
    vuint64m1_t zero_u64m1 = __riscv_vmv_v_x_u64m1(0, 1);
    *sum_ptr = __riscv_vmv_x_s_u64m1_u64(__riscv_vredsum_vs_u64m4_u64m1(sum_u64m4, zero_u64m1, vlmax)),
    *sumsq_ptr = __riscv_vmv_x_s_u64m1_u64(__riscv_vredsum_vs_u64m4_u64m1(sumsq_u64m4, zero_u64m1, vlmax));
}

NK_PUBLIC void nk_reduce_moments_u16_rvv(                              //
    nk_u16_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_u16_t);
    int aligned = (stride_bytes % sizeof(nk_u16_t) == 0);

    if (count == 0) *sum_ptr = 0, *sumsq_ptr = 0;
    else if (!aligned) { nk_reduce_moments_u16_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr); }
    else if (stride_elements == 1) { nk_reduce_moments_u16_rvv_contiguous_(data_ptr, count, sum_ptr, sumsq_ptr); }
    else { nk_reduce_moments_u16_rvv_strided_(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr); }
}

NK_INTERNAL void nk_reduce_minmax_u16_rvv_contiguous_( //
    nk_u16_t const *data_ptr, nk_size_t count,         //
    nk_u16_t *min_value_ptr, nk_size_t *min_index_ptr, //
    nk_u16_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e16m1();
    vuint16m1_t min_u16m1 = __riscv_vmv_v_x_u16m1(NK_U16_MAX, vlmax);
    vuint16m1_t max_u16m1 = __riscv_vmv_v_x_u16m1(NK_U16_MIN, vlmax);
    vuint64m4_t min_indices_u64m4 = __riscv_vmv_v_x_u64m4(0, vlmax);
    vuint64m4_t max_indices_u64m4 = __riscv_vmv_v_x_u64m4(0, vlmax);

    nk_size_t offset = 0;
    for (nk_size_t vl; count > 0; count -= vl, offset += vl, data_ptr += vl) {
        vl = __riscv_vsetvl_e16m1(count);
        vuint16m1_t data_u16m1 = __riscv_vle16_v_u16m1(data_ptr, vl);
        vuint64m4_t pos_u64m4 = __riscv_vadd_vx_u64m4(__riscv_vid_v_u64m4(vl), (nk_u64_t)offset, vl);

        vbool16_t less_b16 = __riscv_vmsltu_vv_u16m1_b16(data_u16m1, min_u16m1, vl);
        min_u16m1 = __riscv_vmerge_vvm_u16m1(min_u16m1, data_u16m1, less_b16, vl);
        min_indices_u64m4 = __riscv_vmerge_vvm_u64m4(min_indices_u64m4, pos_u64m4, less_b16, vl);

        vbool16_t greater_b16 = __riscv_vmsltu_vv_u16m1_b16(max_u16m1, data_u16m1, vl);
        max_u16m1 = __riscv_vmerge_vvm_u16m1(max_u16m1, data_u16m1, greater_b16, vl);
        max_indices_u64m4 = __riscv_vmerge_vvm_u64m4(max_indices_u64m4, pos_u64m4, greater_b16, vl);
    }

    // Horizontal reduction for min
    vuint16m1_t init_max_u16m1 = __riscv_vmv_v_x_u16m1(NK_U16_MAX, 1);
    nk_u16_t min_val = __riscv_vmv_x_s_u16m1_u16(__riscv_vredminu_vs_u16m1_u16m1(min_u16m1, init_max_u16m1, vlmax));
    vbool16_t min_match_b16 = __riscv_vmseq_vx_u16m1_b16(min_u16m1, min_val, vlmax);
    vuint64m4_t sentinel_u64m4 = __riscv_vmv_v_x_u64m4(NK_U64_MAX, vlmax);
    vuint64m4_t min_cands_u64m4 = __riscv_vmerge_vvm_u64m4(sentinel_u64m4, min_indices_u64m4, min_match_b16, vlmax);
    vuint64m1_t init_umax_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    *min_value_ptr = min_val;
    *min_index_ptr = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(
        __riscv_vredminu_vs_u64m4_u64m1(min_cands_u64m4, init_umax_u64m1, vlmax));

    // Horizontal reduction for max
    vuint16m1_t init_min_u16m1 = __riscv_vmv_v_x_u16m1(NK_U16_MIN, 1);
    nk_u16_t max_val = __riscv_vmv_x_s_u16m1_u16(__riscv_vredmaxu_vs_u16m1_u16m1(max_u16m1, init_min_u16m1, vlmax));
    vbool16_t max_match_b16 = __riscv_vmseq_vx_u16m1_b16(max_u16m1, max_val, vlmax);
    vuint64m4_t max_cands_u64m4 = __riscv_vmerge_vvm_u64m4(sentinel_u64m4, max_indices_u64m4, max_match_b16, vlmax);
    *max_value_ptr = max_val;
    *max_index_ptr = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(
        __riscv_vredminu_vs_u64m4_u64m1(max_cands_u64m4, init_umax_u64m1, vlmax));
}

NK_INTERNAL void nk_reduce_minmax_u16_rvv_strided_(                    //
    nk_u16_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_u16_t *min_value_ptr, nk_size_t *min_index_ptr,                 //
    nk_u16_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e16m1();
    vuint16m1_t min_u16m1 = __riscv_vmv_v_x_u16m1(NK_U16_MAX, vlmax);
    vuint16m1_t max_u16m1 = __riscv_vmv_v_x_u16m1(NK_U16_MIN, vlmax);
    vuint64m4_t min_indices_u64m4 = __riscv_vmv_v_x_u64m4(0, vlmax);
    vuint64m4_t max_indices_u64m4 = __riscv_vmv_v_x_u64m4(0, vlmax);
    unsigned char const *ptr = (unsigned char const *)data_ptr;

    nk_size_t offset = 0;
    for (nk_size_t vl; count > 0; count -= vl, offset += vl, ptr += vl * stride_bytes) {
        vl = __riscv_vsetvl_e16m1(count);
        vuint16m1_t data_u16m1 = __riscv_vlse16_v_u16m1((nk_u16_t const *)ptr, (nk_ssize_t)stride_bytes, vl);
        vuint64m4_t pos_u64m4 = __riscv_vadd_vx_u64m4(__riscv_vid_v_u64m4(vl), (nk_u64_t)offset, vl);

        vbool16_t less_b16 = __riscv_vmsltu_vv_u16m1_b16(data_u16m1, min_u16m1, vl);
        min_u16m1 = __riscv_vmerge_vvm_u16m1(min_u16m1, data_u16m1, less_b16, vl);
        min_indices_u64m4 = __riscv_vmerge_vvm_u64m4(min_indices_u64m4, pos_u64m4, less_b16, vl);

        vbool16_t greater_b16 = __riscv_vmsltu_vv_u16m1_b16(max_u16m1, data_u16m1, vl);
        max_u16m1 = __riscv_vmerge_vvm_u16m1(max_u16m1, data_u16m1, greater_b16, vl);
        max_indices_u64m4 = __riscv_vmerge_vvm_u64m4(max_indices_u64m4, pos_u64m4, greater_b16, vl);
    }

    // Horizontal reduction for min
    vuint16m1_t init_max_u16m1 = __riscv_vmv_v_x_u16m1(NK_U16_MAX, 1);
    nk_u16_t min_val = __riscv_vmv_x_s_u16m1_u16(__riscv_vredminu_vs_u16m1_u16m1(min_u16m1, init_max_u16m1, vlmax));
    vbool16_t min_match_b16 = __riscv_vmseq_vx_u16m1_b16(min_u16m1, min_val, vlmax);
    vuint64m4_t sentinel_u64m4 = __riscv_vmv_v_x_u64m4(NK_U64_MAX, vlmax);
    vuint64m4_t min_cands_u64m4 = __riscv_vmerge_vvm_u64m4(sentinel_u64m4, min_indices_u64m4, min_match_b16, vlmax);
    vuint64m1_t init_umax_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    *min_value_ptr = min_val;
    *min_index_ptr = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(
        __riscv_vredminu_vs_u64m4_u64m1(min_cands_u64m4, init_umax_u64m1, vlmax));

    // Horizontal reduction for max
    vuint16m1_t init_min_u16m1 = __riscv_vmv_v_x_u16m1(NK_U16_MIN, 1);
    nk_u16_t max_val = __riscv_vmv_x_s_u16m1_u16(__riscv_vredmaxu_vs_u16m1_u16m1(max_u16m1, init_min_u16m1, vlmax));
    vbool16_t max_match_b16 = __riscv_vmseq_vx_u16m1_b16(max_u16m1, max_val, vlmax);
    vuint64m4_t max_cands_u64m4 = __riscv_vmerge_vvm_u64m4(sentinel_u64m4, max_indices_u64m4, max_match_b16, vlmax);
    *max_value_ptr = max_val;
    *max_index_ptr = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(
        __riscv_vredminu_vs_u64m4_u64m1(max_cands_u64m4, init_umax_u64m1, vlmax));
}

NK_PUBLIC void nk_reduce_minmax_u16_rvv(                               //
    nk_u16_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_u16_t *min_value_ptr, nk_size_t *min_index_ptr,                 //
    nk_u16_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_u16_t);
    int aligned = (stride_bytes % sizeof(nk_u16_t) == 0);

    if (count == 0)
        *min_value_ptr = NK_U16_MAX, *min_index_ptr = NK_SIZE_MAX, *max_value_ptr = NK_U16_MIN,
        *max_index_ptr = NK_SIZE_MAX;
    else if (!aligned)
        nk_reduce_minmax_u16_serial(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                    max_index_ptr);
    else if (stride_elements == 1)
        nk_reduce_minmax_u16_rvv_contiguous_(data_ptr, count, min_value_ptr, min_index_ptr, max_value_ptr,
                                             max_index_ptr);
    else
        nk_reduce_minmax_u16_rvv_strided_(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                          max_index_ptr);
}

NK_INTERNAL void nk_reduce_moments_i32_rvv_contiguous_( //
    nk_i32_t const *data_ptr, nk_size_t count,          //
    nk_i64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e64m2();
    vint64m2_t sum_i64m2 = __riscv_vmv_v_x_i64m2(0, vlmax);
    vuint64m2_t sumsq_u64m2 = __riscv_vmv_v_x_u64m2(0, vlmax);

    for (nk_size_t vl; count > 0; count -= vl, data_ptr += vl) {
        vl = __riscv_vsetvl_e32m1(count);
        vint32m1_t data_i32m1 = __riscv_vle32_v_i32m1(data_ptr, vl);

        // Widen i32 → i64 for sum (single widening step)
        vint64m2_t data_i64m2 = __riscv_vsext_vf2_i64m2(data_i32m1, vl);
        sum_i64m2 = __riscv_vadd_vv_i64m2(sum_i64m2, data_i64m2, vl);

        // Sumsq: i32 × i32 → i64 (widening multiply)
        vint64m2_t squares_i64m2 = __riscv_vwmul_vv_i64m2(data_i32m1, data_i32m1, vl);
        // Reinterpret as unsigned for sumsq
        sumsq_u64m2 = __riscv_vadd_vv_u64m2(sumsq_u64m2, __riscv_vreinterpret_v_i64m2_u64m2(squares_i64m2), vl);
    }

    // Horizontal reduction
    vint64m1_t zero_i64m1 = __riscv_vmv_v_x_i64m1(0, 1);
    *sum_ptr = __riscv_vmv_x_s_i64m1_i64(__riscv_vredsum_vs_i64m2_i64m1(sum_i64m2, zero_i64m1, vlmax));

    vuint64m1_t zero_u64m1 = __riscv_vmv_v_x_u64m1(0, 1);
    *sumsq_ptr = __riscv_vmv_x_s_u64m1_u64(__riscv_vredsum_vs_u64m2_u64m1(sumsq_u64m2, zero_u64m1, vlmax));
}

NK_INTERNAL void nk_reduce_moments_i32_rvv_strided_(                   //
    nk_i32_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e64m2();
    vint64m2_t sum_i64m2 = __riscv_vmv_v_x_i64m2(0, vlmax);
    vuint64m2_t sumsq_u64m2 = __riscv_vmv_v_x_u64m2(0, vlmax);
    unsigned char const *ptr = (unsigned char const *)data_ptr;

    for (nk_size_t vl; count > 0; count -= vl, ptr += vl * stride_bytes) {
        vl = __riscv_vsetvl_e32m1(count);
        vint32m1_t data_i32m1 = __riscv_vlse32_v_i32m1((nk_i32_t const *)ptr, (nk_ssize_t)stride_bytes, vl);

        // Widen i32 → i64 for sum (single widening step)
        vint64m2_t data_i64m2 = __riscv_vsext_vf2_i64m2(data_i32m1, vl);
        sum_i64m2 = __riscv_vadd_vv_i64m2(sum_i64m2, data_i64m2, vl);

        // Sumsq: i32 × i32 → i64 (widening multiply)
        vint64m2_t squares_i64m2 = __riscv_vwmul_vv_i64m2(data_i32m1, data_i32m1, vl);
        // Reinterpret as unsigned for sumsq
        sumsq_u64m2 = __riscv_vadd_vv_u64m2(sumsq_u64m2, __riscv_vreinterpret_v_i64m2_u64m2(squares_i64m2), vl);
    }

    // Horizontal reduction
    vint64m1_t zero_i64m1 = __riscv_vmv_v_x_i64m1(0, 1);
    *sum_ptr = __riscv_vmv_x_s_i64m1_i64(__riscv_vredsum_vs_i64m2_i64m1(sum_i64m2, zero_i64m1, vlmax));

    vuint64m1_t zero_u64m1 = __riscv_vmv_v_x_u64m1(0, 1);
    *sumsq_ptr = __riscv_vmv_x_s_u64m1_u64(__riscv_vredsum_vs_u64m2_u64m1(sumsq_u64m2, zero_u64m1, vlmax));
}

NK_PUBLIC void nk_reduce_moments_i32_rvv(                              //
    nk_i32_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_i32_t);
    int aligned = (stride_bytes % sizeof(nk_i32_t) == 0);

    if (count == 0) *sum_ptr = 0, *sumsq_ptr = 0;
    else if (!aligned) { nk_reduce_moments_i32_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr); }
    else if (stride_elements == 1) { nk_reduce_moments_i32_rvv_contiguous_(data_ptr, count, sum_ptr, sumsq_ptr); }
    else { nk_reduce_moments_i32_rvv_strided_(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr); }
}

NK_INTERNAL void nk_reduce_minmax_i32_rvv_contiguous_( //
    nk_i32_t const *data_ptr, nk_size_t count,         //
    nk_i32_t *min_value_ptr, nk_size_t *min_index_ptr, //
    nk_i32_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e32m1();
    vint32m1_t min_i32m1 = __riscv_vmv_v_x_i32m1(NK_I32_MAX, vlmax);
    vint32m1_t max_i32m1 = __riscv_vmv_v_x_i32m1(NK_I32_MIN, vlmax);
    vuint64m2_t min_indices_u64m2 = __riscv_vmv_v_x_u64m2(0, vlmax);
    vuint64m2_t max_indices_u64m2 = __riscv_vmv_v_x_u64m2(0, vlmax);

    nk_size_t offset = 0;
    for (nk_size_t vl; count > 0; count -= vl, offset += vl, data_ptr += vl) {
        vl = __riscv_vsetvl_e32m1(count);
        vint32m1_t data_i32m1 = __riscv_vle32_v_i32m1(data_ptr, vl);
        vuint64m2_t pos_u64m2 = __riscv_vadd_vx_u64m2(__riscv_vid_v_u64m2(vl), (nk_u64_t)offset, vl);

        vbool32_t less_b32 = __riscv_vmslt_vv_i32m1_b32(data_i32m1, min_i32m1, vl);
        min_i32m1 = __riscv_vmerge_vvm_i32m1(min_i32m1, data_i32m1, less_b32, vl);
        min_indices_u64m2 = __riscv_vmerge_vvm_u64m2(min_indices_u64m2, pos_u64m2, less_b32, vl);

        vbool32_t greater_b32 = __riscv_vmslt_vv_i32m1_b32(max_i32m1, data_i32m1, vl);
        max_i32m1 = __riscv_vmerge_vvm_i32m1(max_i32m1, data_i32m1, greater_b32, vl);
        max_indices_u64m2 = __riscv_vmerge_vvm_u64m2(max_indices_u64m2, pos_u64m2, greater_b32, vl);
    }

    // Horizontal reduction for min
    vint32m1_t init_max_i32m1 = __riscv_vmv_v_x_i32m1(NK_I32_MAX, 1);
    nk_i32_t min_val = __riscv_vmv_x_s_i32m1_i32(__riscv_vredmin_vs_i32m1_i32m1(min_i32m1, init_max_i32m1, vlmax));
    vbool32_t min_match_b32 = __riscv_vmseq_vx_i32m1_b32(min_i32m1, min_val, vlmax);
    vuint64m2_t sentinel_u64m2 = __riscv_vmv_v_x_u64m2(NK_U64_MAX, vlmax);
    vuint64m2_t min_cands_u64m2 = __riscv_vmerge_vvm_u64m2(sentinel_u64m2, min_indices_u64m2, min_match_b32, vlmax);
    vuint64m1_t init_umax_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    *min_value_ptr = min_val;
    *min_index_ptr = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(
        __riscv_vredminu_vs_u64m2_u64m1(min_cands_u64m2, init_umax_u64m1, vlmax));

    // Horizontal reduction for max
    vint32m1_t init_min_i32m1 = __riscv_vmv_v_x_i32m1(NK_I32_MIN, 1);
    nk_i32_t max_val = __riscv_vmv_x_s_i32m1_i32(__riscv_vredmax_vs_i32m1_i32m1(max_i32m1, init_min_i32m1, vlmax));
    vbool32_t max_match_b32 = __riscv_vmseq_vx_i32m1_b32(max_i32m1, max_val, vlmax);
    vuint64m2_t max_cands_u64m2 = __riscv_vmerge_vvm_u64m2(sentinel_u64m2, max_indices_u64m2, max_match_b32, vlmax);
    *max_value_ptr = max_val;
    *max_index_ptr = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(
        __riscv_vredminu_vs_u64m2_u64m1(max_cands_u64m2, init_umax_u64m1, vlmax));
}

NK_INTERNAL void nk_reduce_minmax_i32_rvv_strided_(                    //
    nk_i32_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_i32_t *min_value_ptr, nk_size_t *min_index_ptr,                 //
    nk_i32_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e32m1();
    vint32m1_t min_i32m1 = __riscv_vmv_v_x_i32m1(NK_I32_MAX, vlmax);
    vint32m1_t max_i32m1 = __riscv_vmv_v_x_i32m1(NK_I32_MIN, vlmax);
    vuint64m2_t min_indices_u64m2 = __riscv_vmv_v_x_u64m2(0, vlmax);
    vuint64m2_t max_indices_u64m2 = __riscv_vmv_v_x_u64m2(0, vlmax);
    unsigned char const *ptr = (unsigned char const *)data_ptr;

    nk_size_t offset = 0;
    for (nk_size_t vl; count > 0; count -= vl, offset += vl, ptr += vl * stride_bytes) {
        vl = __riscv_vsetvl_e32m1(count);
        vint32m1_t data_i32m1 = __riscv_vlse32_v_i32m1((nk_i32_t const *)ptr, (nk_ssize_t)stride_bytes, vl);
        vuint64m2_t pos_u64m2 = __riscv_vadd_vx_u64m2(__riscv_vid_v_u64m2(vl), (nk_u64_t)offset, vl);

        vbool32_t less_b32 = __riscv_vmslt_vv_i32m1_b32(data_i32m1, min_i32m1, vl);
        min_i32m1 = __riscv_vmerge_vvm_i32m1(min_i32m1, data_i32m1, less_b32, vl);
        min_indices_u64m2 = __riscv_vmerge_vvm_u64m2(min_indices_u64m2, pos_u64m2, less_b32, vl);

        vbool32_t greater_b32 = __riscv_vmslt_vv_i32m1_b32(max_i32m1, data_i32m1, vl);
        max_i32m1 = __riscv_vmerge_vvm_i32m1(max_i32m1, data_i32m1, greater_b32, vl);
        max_indices_u64m2 = __riscv_vmerge_vvm_u64m2(max_indices_u64m2, pos_u64m2, greater_b32, vl);
    }

    // Horizontal reduction for min
    vint32m1_t init_max_i32m1 = __riscv_vmv_v_x_i32m1(NK_I32_MAX, 1);
    nk_i32_t min_val = __riscv_vmv_x_s_i32m1_i32(__riscv_vredmin_vs_i32m1_i32m1(min_i32m1, init_max_i32m1, vlmax));
    vbool32_t min_match_b32 = __riscv_vmseq_vx_i32m1_b32(min_i32m1, min_val, vlmax);
    vuint64m2_t sentinel_u64m2 = __riscv_vmv_v_x_u64m2(NK_U64_MAX, vlmax);
    vuint64m2_t min_cands_u64m2 = __riscv_vmerge_vvm_u64m2(sentinel_u64m2, min_indices_u64m2, min_match_b32, vlmax);
    vuint64m1_t init_umax_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    *min_value_ptr = min_val;
    *min_index_ptr = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(
        __riscv_vredminu_vs_u64m2_u64m1(min_cands_u64m2, init_umax_u64m1, vlmax));

    // Horizontal reduction for max
    vint32m1_t init_min_i32m1 = __riscv_vmv_v_x_i32m1(NK_I32_MIN, 1);
    nk_i32_t max_val = __riscv_vmv_x_s_i32m1_i32(__riscv_vredmax_vs_i32m1_i32m1(max_i32m1, init_min_i32m1, vlmax));
    vbool32_t max_match_b32 = __riscv_vmseq_vx_i32m1_b32(max_i32m1, max_val, vlmax);
    vuint64m2_t max_cands_u64m2 = __riscv_vmerge_vvm_u64m2(sentinel_u64m2, max_indices_u64m2, max_match_b32, vlmax);
    *max_value_ptr = max_val;
    *max_index_ptr = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(
        __riscv_vredminu_vs_u64m2_u64m1(max_cands_u64m2, init_umax_u64m1, vlmax));
}

NK_PUBLIC void nk_reduce_minmax_i32_rvv(                               //
    nk_i32_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_i32_t *min_value_ptr, nk_size_t *min_index_ptr,                 //
    nk_i32_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_i32_t);
    int aligned = (stride_bytes % sizeof(nk_i32_t) == 0);

    if (count == 0)
        *min_value_ptr = NK_I32_MAX, *min_index_ptr = NK_SIZE_MAX, *max_value_ptr = NK_I32_MIN,
        *max_index_ptr = NK_SIZE_MAX;
    else if (!aligned)
        nk_reduce_minmax_i32_serial(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                    max_index_ptr);
    else if (stride_elements == 1)
        nk_reduce_minmax_i32_rvv_contiguous_(data_ptr, count, min_value_ptr, min_index_ptr, max_value_ptr,
                                             max_index_ptr);
    else
        nk_reduce_minmax_i32_rvv_strided_(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                          max_index_ptr);
}

NK_INTERNAL void nk_reduce_moments_u32_rvv_contiguous_( //
    nk_u32_t const *data_ptr, nk_size_t count,          //
    nk_u64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e64m2();
    vuint64m2_t sum_u64m2 = __riscv_vmv_v_x_u64m2(0, vlmax);
    vuint64m2_t sumsq_u64m2 = __riscv_vmv_v_x_u64m2(0, vlmax);

    for (nk_size_t vl; count > 0; count -= vl, data_ptr += vl) {
        vl = __riscv_vsetvl_e32m1(count);
        vuint32m1_t data_u32m1 = __riscv_vle32_v_u32m1(data_ptr, vl);

        // Widen u32 → u64 for sum (single widening step)
        vuint64m2_t data_u64m2 = __riscv_vzext_vf2_u64m2(data_u32m1, vl);
        sum_u64m2 = __riscv_vadd_vv_u64m2(sum_u64m2, data_u64m2, vl);

        // Sumsq: u32 × u32 → u64 (widening multiply)
        vuint64m2_t squares_u64m2 = __riscv_vwmulu_vv_u64m2(data_u32m1, data_u32m1, vl);
        sumsq_u64m2 = __riscv_vadd_vv_u64m2(sumsq_u64m2, squares_u64m2, vl);
    }

    // Horizontal reduction
    vuint64m1_t zero_u64m1 = __riscv_vmv_v_x_u64m1(0, 1);
    *sum_ptr = __riscv_vmv_x_s_u64m1_u64(__riscv_vredsum_vs_u64m2_u64m1(sum_u64m2, zero_u64m1, vlmax)),
    *sumsq_ptr = __riscv_vmv_x_s_u64m1_u64(__riscv_vredsum_vs_u64m2_u64m1(sumsq_u64m2, zero_u64m1, vlmax));
}

NK_INTERNAL void nk_reduce_moments_u32_rvv_strided_(                   //
    nk_u32_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e64m2();
    vuint64m2_t sum_u64m2 = __riscv_vmv_v_x_u64m2(0, vlmax);
    vuint64m2_t sumsq_u64m2 = __riscv_vmv_v_x_u64m2(0, vlmax);
    unsigned char const *ptr = (unsigned char const *)data_ptr;

    for (nk_size_t vl; count > 0; count -= vl, ptr += vl * stride_bytes) {
        vl = __riscv_vsetvl_e32m1(count);
        vuint32m1_t data_u32m1 = __riscv_vlse32_v_u32m1((nk_u32_t const *)ptr, (nk_ssize_t)stride_bytes, vl);

        // Widen u32 → u64 for sum (single widening step)
        vuint64m2_t data_u64m2 = __riscv_vzext_vf2_u64m2(data_u32m1, vl);
        sum_u64m2 = __riscv_vadd_vv_u64m2(sum_u64m2, data_u64m2, vl);

        // Sumsq: u32 × u32 → u64 (widening multiply)
        vuint64m2_t squares_u64m2 = __riscv_vwmulu_vv_u64m2(data_u32m1, data_u32m1, vl);
        sumsq_u64m2 = __riscv_vadd_vv_u64m2(sumsq_u64m2, squares_u64m2, vl);
    }

    // Horizontal reduction
    vuint64m1_t zero_u64m1 = __riscv_vmv_v_x_u64m1(0, 1);
    *sum_ptr = __riscv_vmv_x_s_u64m1_u64(__riscv_vredsum_vs_u64m2_u64m1(sum_u64m2, zero_u64m1, vlmax)),
    *sumsq_ptr = __riscv_vmv_x_s_u64m1_u64(__riscv_vredsum_vs_u64m2_u64m1(sumsq_u64m2, zero_u64m1, vlmax));
}

NK_PUBLIC void nk_reduce_moments_u32_rvv(                              //
    nk_u32_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_u32_t);
    int aligned = (stride_bytes % sizeof(nk_u32_t) == 0);

    if (count == 0) *sum_ptr = 0, *sumsq_ptr = 0;
    else if (!aligned) { nk_reduce_moments_u32_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr); }
    else if (stride_elements == 1) { nk_reduce_moments_u32_rvv_contiguous_(data_ptr, count, sum_ptr, sumsq_ptr); }
    else { nk_reduce_moments_u32_rvv_strided_(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr); }
}

NK_INTERNAL void nk_reduce_minmax_u32_rvv_contiguous_( //
    nk_u32_t const *data_ptr, nk_size_t count,         //
    nk_u32_t *min_value_ptr, nk_size_t *min_index_ptr, //
    nk_u32_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e32m1();
    vuint32m1_t min_u32m1 = __riscv_vmv_v_x_u32m1(NK_U32_MAX, vlmax);
    vuint32m1_t max_u32m1 = __riscv_vmv_v_x_u32m1(NK_U32_MIN, vlmax);
    vuint64m2_t min_indices_u64m2 = __riscv_vmv_v_x_u64m2(0, vlmax);
    vuint64m2_t max_indices_u64m2 = __riscv_vmv_v_x_u64m2(0, vlmax);

    nk_size_t offset = 0;
    for (nk_size_t vl; count > 0; count -= vl, offset += vl, data_ptr += vl) {
        vl = __riscv_vsetvl_e32m1(count);
        vuint32m1_t data_u32m1 = __riscv_vle32_v_u32m1(data_ptr, vl);
        vuint64m2_t pos_u64m2 = __riscv_vadd_vx_u64m2(__riscv_vid_v_u64m2(vl), (nk_u64_t)offset, vl);

        vbool32_t less_b32 = __riscv_vmsltu_vv_u32m1_b32(data_u32m1, min_u32m1, vl);
        min_u32m1 = __riscv_vmerge_vvm_u32m1(min_u32m1, data_u32m1, less_b32, vl);
        min_indices_u64m2 = __riscv_vmerge_vvm_u64m2(min_indices_u64m2, pos_u64m2, less_b32, vl);

        vbool32_t greater_b32 = __riscv_vmsltu_vv_u32m1_b32(max_u32m1, data_u32m1, vl);
        max_u32m1 = __riscv_vmerge_vvm_u32m1(max_u32m1, data_u32m1, greater_b32, vl);
        max_indices_u64m2 = __riscv_vmerge_vvm_u64m2(max_indices_u64m2, pos_u64m2, greater_b32, vl);
    }

    // Horizontal reduction for min
    vuint32m1_t init_max_u32m1 = __riscv_vmv_v_x_u32m1(NK_U32_MAX, 1);
    nk_u32_t min_val = __riscv_vmv_x_s_u32m1_u32(__riscv_vredminu_vs_u32m1_u32m1(min_u32m1, init_max_u32m1, vlmax));
    vbool32_t min_match_b32 = __riscv_vmseq_vx_u32m1_b32(min_u32m1, min_val, vlmax);
    vuint64m2_t sentinel_u64m2 = __riscv_vmv_v_x_u64m2(NK_U64_MAX, vlmax);
    vuint64m2_t min_cands_u64m2 = __riscv_vmerge_vvm_u64m2(sentinel_u64m2, min_indices_u64m2, min_match_b32, vlmax);
    vuint64m1_t init_umax_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    *min_value_ptr = min_val;
    *min_index_ptr = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(
        __riscv_vredminu_vs_u64m2_u64m1(min_cands_u64m2, init_umax_u64m1, vlmax));

    // Horizontal reduction for max
    vuint32m1_t init_min_u32m1 = __riscv_vmv_v_x_u32m1(NK_U32_MIN, 1);
    nk_u32_t max_val = __riscv_vmv_x_s_u32m1_u32(__riscv_vredmaxu_vs_u32m1_u32m1(max_u32m1, init_min_u32m1, vlmax));
    vbool32_t max_match_b32 = __riscv_vmseq_vx_u32m1_b32(max_u32m1, max_val, vlmax);
    vuint64m2_t max_cands_u64m2 = __riscv_vmerge_vvm_u64m2(sentinel_u64m2, max_indices_u64m2, max_match_b32, vlmax);
    *max_value_ptr = max_val;
    *max_index_ptr = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(
        __riscv_vredminu_vs_u64m2_u64m1(max_cands_u64m2, init_umax_u64m1, vlmax));
}

NK_INTERNAL void nk_reduce_minmax_u32_rvv_strided_(                    //
    nk_u32_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_u32_t *min_value_ptr, nk_size_t *min_index_ptr,                 //
    nk_u32_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e32m1();
    vuint32m1_t min_u32m1 = __riscv_vmv_v_x_u32m1(NK_U32_MAX, vlmax);
    vuint32m1_t max_u32m1 = __riscv_vmv_v_x_u32m1(NK_U32_MIN, vlmax);
    vuint64m2_t min_indices_u64m2 = __riscv_vmv_v_x_u64m2(0, vlmax);
    vuint64m2_t max_indices_u64m2 = __riscv_vmv_v_x_u64m2(0, vlmax);
    unsigned char const *ptr = (unsigned char const *)data_ptr;

    nk_size_t offset = 0;
    for (nk_size_t vl; count > 0; count -= vl, offset += vl, ptr += vl * stride_bytes) {
        vl = __riscv_vsetvl_e32m1(count);
        vuint32m1_t data_u32m1 = __riscv_vlse32_v_u32m1((nk_u32_t const *)ptr, (nk_ssize_t)stride_bytes, vl);
        vuint64m2_t pos_u64m2 = __riscv_vadd_vx_u64m2(__riscv_vid_v_u64m2(vl), (nk_u64_t)offset, vl);

        vbool32_t less_b32 = __riscv_vmsltu_vv_u32m1_b32(data_u32m1, min_u32m1, vl);
        min_u32m1 = __riscv_vmerge_vvm_u32m1(min_u32m1, data_u32m1, less_b32, vl);
        min_indices_u64m2 = __riscv_vmerge_vvm_u64m2(min_indices_u64m2, pos_u64m2, less_b32, vl);

        vbool32_t greater_b32 = __riscv_vmsltu_vv_u32m1_b32(max_u32m1, data_u32m1, vl);
        max_u32m1 = __riscv_vmerge_vvm_u32m1(max_u32m1, data_u32m1, greater_b32, vl);
        max_indices_u64m2 = __riscv_vmerge_vvm_u64m2(max_indices_u64m2, pos_u64m2, greater_b32, vl);
    }

    // Horizontal reduction for min
    vuint32m1_t init_max_u32m1 = __riscv_vmv_v_x_u32m1(NK_U32_MAX, 1);
    nk_u32_t min_val = __riscv_vmv_x_s_u32m1_u32(__riscv_vredminu_vs_u32m1_u32m1(min_u32m1, init_max_u32m1, vlmax));
    vbool32_t min_match_b32 = __riscv_vmseq_vx_u32m1_b32(min_u32m1, min_val, vlmax);
    vuint64m2_t sentinel_u64m2 = __riscv_vmv_v_x_u64m2(NK_U64_MAX, vlmax);
    vuint64m2_t min_cands_u64m2 = __riscv_vmerge_vvm_u64m2(sentinel_u64m2, min_indices_u64m2, min_match_b32, vlmax);
    vuint64m1_t init_umax_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    *min_value_ptr = min_val;
    *min_index_ptr = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(
        __riscv_vredminu_vs_u64m2_u64m1(min_cands_u64m2, init_umax_u64m1, vlmax));

    // Horizontal reduction for max
    vuint32m1_t init_min_u32m1 = __riscv_vmv_v_x_u32m1(NK_U32_MIN, 1);
    nk_u32_t max_val = __riscv_vmv_x_s_u32m1_u32(__riscv_vredmaxu_vs_u32m1_u32m1(max_u32m1, init_min_u32m1, vlmax));
    vbool32_t max_match_b32 = __riscv_vmseq_vx_u32m1_b32(max_u32m1, max_val, vlmax);
    vuint64m2_t max_cands_u64m2 = __riscv_vmerge_vvm_u64m2(sentinel_u64m2, max_indices_u64m2, max_match_b32, vlmax);
    *max_value_ptr = max_val;
    *max_index_ptr = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(
        __riscv_vredminu_vs_u64m2_u64m1(max_cands_u64m2, init_umax_u64m1, vlmax));
}

NK_PUBLIC void nk_reduce_minmax_u32_rvv(                               //
    nk_u32_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_u32_t *min_value_ptr, nk_size_t *min_index_ptr,                 //
    nk_u32_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_u32_t);
    int aligned = (stride_bytes % sizeof(nk_u32_t) == 0);

    if (count == 0)
        *min_value_ptr = NK_U32_MAX, *min_index_ptr = NK_SIZE_MAX, *max_value_ptr = NK_U32_MIN,
        *max_index_ptr = NK_SIZE_MAX;
    else if (!aligned)
        nk_reduce_minmax_u32_serial(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                    max_index_ptr);
    else if (stride_elements == 1)
        nk_reduce_minmax_u32_rvv_contiguous_(data_ptr, count, min_value_ptr, min_index_ptr, max_value_ptr,
                                             max_index_ptr);
    else
        nk_reduce_minmax_u32_rvv_strided_(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                          max_index_ptr);
}

NK_INTERNAL void nk_reduce_moments_i64_rvv_contiguous_( //
    nk_i64_t const *data_ptr, nk_size_t count,          //
    nk_i64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e64m1();
    vint64m1_t sum_i64m1 = __riscv_vmv_v_x_i64m1(0, vlmax);
    vuint64m1_t sumsq_u64m1 = __riscv_vmv_v_x_u64m1(0, vlmax);

    for (nk_size_t vl; count > 0; count -= vl, data_ptr += vl) {
        vl = __riscv_vsetvl_e64m1(count);
        vint64m1_t data_i64m1 = __riscv_vle64_v_i64m1(data_ptr, vl);

        // Direct accumulation (no widening)
        sum_i64m1 = __riscv_vadd_vv_i64m1(sum_i64m1, data_i64m1, vl);

        // Sumsq: i64 × i64 → i64, reinterpret as u64
        vint64m1_t squares_i64m1 = __riscv_vmul_vv_i64m1(data_i64m1, data_i64m1, vl);
        sumsq_u64m1 = __riscv_vadd_vv_u64m1(sumsq_u64m1, __riscv_vreinterpret_v_i64m1_u64m1(squares_i64m1), vl);
    }

    // Horizontal reduction
    vint64m1_t zero_i64m1 = __riscv_vmv_v_x_i64m1(0, 1);
    *sum_ptr = __riscv_vmv_x_s_i64m1_i64(__riscv_vredsum_vs_i64m1_i64m1(sum_i64m1, zero_i64m1, vlmax));

    vuint64m1_t zero_u64m1 = __riscv_vmv_v_x_u64m1(0, 1);
    *sumsq_ptr = __riscv_vmv_x_s_u64m1_u64(__riscv_vredsum_vs_u64m1_u64m1(sumsq_u64m1, zero_u64m1, vlmax));
}

NK_INTERNAL void nk_reduce_moments_i64_rvv_strided_(                   //
    nk_i64_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e64m1();
    vint64m1_t sum_i64m1 = __riscv_vmv_v_x_i64m1(0, vlmax);
    vuint64m1_t sumsq_u64m1 = __riscv_vmv_v_x_u64m1(0, vlmax);
    unsigned char const *ptr = (unsigned char const *)data_ptr;

    for (nk_size_t vl; count > 0; count -= vl, ptr += vl * stride_bytes) {
        vl = __riscv_vsetvl_e64m1(count);
        vint64m1_t data_i64m1 = __riscv_vlse64_v_i64m1((nk_i64_t const *)ptr, (nk_ssize_t)stride_bytes, vl);

        // Direct accumulation (no widening)
        sum_i64m1 = __riscv_vadd_vv_i64m1(sum_i64m1, data_i64m1, vl);

        // Sumsq: i64 × i64 → i64, reinterpret as u64
        vint64m1_t squares_i64m1 = __riscv_vmul_vv_i64m1(data_i64m1, data_i64m1, vl);
        sumsq_u64m1 = __riscv_vadd_vv_u64m1(sumsq_u64m1, __riscv_vreinterpret_v_i64m1_u64m1(squares_i64m1), vl);
    }

    // Horizontal reduction
    vint64m1_t zero_i64m1 = __riscv_vmv_v_x_i64m1(0, 1);
    *sum_ptr = __riscv_vmv_x_s_i64m1_i64(__riscv_vredsum_vs_i64m1_i64m1(sum_i64m1, zero_i64m1, vlmax));

    vuint64m1_t zero_u64m1 = __riscv_vmv_v_x_u64m1(0, 1);
    *sumsq_ptr = __riscv_vmv_x_s_u64m1_u64(__riscv_vredsum_vs_u64m1_u64m1(sumsq_u64m1, zero_u64m1, vlmax));
}

NK_PUBLIC void nk_reduce_moments_i64_rvv(                              //
    nk_i64_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_i64_t);
    int aligned = (stride_bytes % sizeof(nk_i64_t) == 0);

    if (count == 0) *sum_ptr = 0, *sumsq_ptr = 0;
    else if (!aligned) { nk_reduce_moments_i64_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr); }
    else if (stride_elements == 1) { nk_reduce_moments_i64_rvv_contiguous_(data_ptr, count, sum_ptr, sumsq_ptr); }
    else { nk_reduce_moments_i64_rvv_strided_(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr); }
}

NK_INTERNAL void nk_reduce_minmax_i64_rvv_contiguous_( //
    nk_i64_t const *data_ptr, nk_size_t count,         //
    nk_i64_t *min_value_ptr, nk_size_t *min_index_ptr, //
    nk_i64_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e64m1();
    vint64m1_t min_i64m1 = __riscv_vmv_v_x_i64m1(NK_I64_MAX, vlmax);
    vint64m1_t max_i64m1 = __riscv_vmv_v_x_i64m1(NK_I64_MIN, vlmax);
    vuint64m1_t min_indices_u64m1 = __riscv_vmv_v_x_u64m1(0, vlmax);
    vuint64m1_t max_indices_u64m1 = __riscv_vmv_v_x_u64m1(0, vlmax);

    nk_size_t offset = 0;
    for (nk_size_t vl; count > 0; count -= vl, offset += vl, data_ptr += vl) {
        vl = __riscv_vsetvl_e64m1(count);
        vint64m1_t data_i64m1 = __riscv_vle64_v_i64m1(data_ptr, vl);
        vuint64m1_t pos_u64m1 = __riscv_vadd_vx_u64m1(__riscv_vid_v_u64m1(vl), (nk_u64_t)offset, vl);

        vbool64_t less_b64 = __riscv_vmslt_vv_i64m1_b64(data_i64m1, min_i64m1, vl);
        min_i64m1 = __riscv_vmerge_vvm_i64m1(min_i64m1, data_i64m1, less_b64, vl);
        min_indices_u64m1 = __riscv_vmerge_vvm_u64m1(min_indices_u64m1, pos_u64m1, less_b64, vl);

        vbool64_t greater_b64 = __riscv_vmslt_vv_i64m1_b64(max_i64m1, data_i64m1, vl);
        max_i64m1 = __riscv_vmerge_vvm_i64m1(max_i64m1, data_i64m1, greater_b64, vl);
        max_indices_u64m1 = __riscv_vmerge_vvm_u64m1(max_indices_u64m1, pos_u64m1, greater_b64, vl);
    }

    // Horizontal reduction for min
    vint64m1_t init_max_i64m1 = __riscv_vmv_v_x_i64m1(NK_I64_MAX, 1);
    nk_i64_t min_val = __riscv_vmv_x_s_i64m1_i64(__riscv_vredmin_vs_i64m1_i64m1(min_i64m1, init_max_i64m1, vlmax));
    vbool64_t min_match_b64 = __riscv_vmseq_vx_i64m1_b64(min_i64m1, min_val, vlmax);
    vuint64m1_t sentinel_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, vlmax);
    vuint64m1_t min_cands_u64m1 = __riscv_vmerge_vvm_u64m1(sentinel_u64m1, min_indices_u64m1, min_match_b64, vlmax);
    vuint64m1_t init_umax_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    *min_value_ptr = min_val;
    *min_index_ptr = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(
        __riscv_vredminu_vs_u64m1_u64m1(min_cands_u64m1, init_umax_u64m1, vlmax));

    // Horizontal reduction for max
    vint64m1_t init_min_i64m1 = __riscv_vmv_v_x_i64m1(NK_I64_MIN, 1);
    nk_i64_t max_val = __riscv_vmv_x_s_i64m1_i64(__riscv_vredmax_vs_i64m1_i64m1(max_i64m1, init_min_i64m1, vlmax));
    vbool64_t max_match_b64 = __riscv_vmseq_vx_i64m1_b64(max_i64m1, max_val, vlmax);
    vuint64m1_t max_cands_u64m1 = __riscv_vmerge_vvm_u64m1(sentinel_u64m1, max_indices_u64m1, max_match_b64, vlmax);
    *max_value_ptr = max_val;
    *max_index_ptr = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(
        __riscv_vredminu_vs_u64m1_u64m1(max_cands_u64m1, init_umax_u64m1, vlmax));
}

NK_INTERNAL void nk_reduce_minmax_i64_rvv_strided_(                    //
    nk_i64_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *min_value_ptr, nk_size_t *min_index_ptr,                 //
    nk_i64_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e64m1();
    vint64m1_t min_i64m1 = __riscv_vmv_v_x_i64m1(NK_I64_MAX, vlmax);
    vint64m1_t max_i64m1 = __riscv_vmv_v_x_i64m1(NK_I64_MIN, vlmax);
    vuint64m1_t min_indices_u64m1 = __riscv_vmv_v_x_u64m1(0, vlmax);
    vuint64m1_t max_indices_u64m1 = __riscv_vmv_v_x_u64m1(0, vlmax);
    unsigned char const *ptr = (unsigned char const *)data_ptr;

    nk_size_t offset = 0;
    for (nk_size_t vl; count > 0; count -= vl, offset += vl, ptr += vl * stride_bytes) {
        vl = __riscv_vsetvl_e64m1(count);
        vint64m1_t data_i64m1 = __riscv_vlse64_v_i64m1((nk_i64_t const *)ptr, (nk_ssize_t)stride_bytes, vl);
        vuint64m1_t pos_u64m1 = __riscv_vadd_vx_u64m1(__riscv_vid_v_u64m1(vl), (nk_u64_t)offset, vl);

        vbool64_t less_b64 = __riscv_vmslt_vv_i64m1_b64(data_i64m1, min_i64m1, vl);
        min_i64m1 = __riscv_vmerge_vvm_i64m1(min_i64m1, data_i64m1, less_b64, vl);
        min_indices_u64m1 = __riscv_vmerge_vvm_u64m1(min_indices_u64m1, pos_u64m1, less_b64, vl);

        vbool64_t greater_b64 = __riscv_vmslt_vv_i64m1_b64(max_i64m1, data_i64m1, vl);
        max_i64m1 = __riscv_vmerge_vvm_i64m1(max_i64m1, data_i64m1, greater_b64, vl);
        max_indices_u64m1 = __riscv_vmerge_vvm_u64m1(max_indices_u64m1, pos_u64m1, greater_b64, vl);
    }

    // Horizontal reduction for min
    vint64m1_t init_max_i64m1 = __riscv_vmv_v_x_i64m1(NK_I64_MAX, 1);
    nk_i64_t min_val = __riscv_vmv_x_s_i64m1_i64(__riscv_vredmin_vs_i64m1_i64m1(min_i64m1, init_max_i64m1, vlmax));
    vbool64_t min_match_b64 = __riscv_vmseq_vx_i64m1_b64(min_i64m1, min_val, vlmax);
    vuint64m1_t sentinel_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, vlmax);
    vuint64m1_t min_cands_u64m1 = __riscv_vmerge_vvm_u64m1(sentinel_u64m1, min_indices_u64m1, min_match_b64, vlmax);
    vuint64m1_t init_umax_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    *min_value_ptr = min_val;
    *min_index_ptr = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(
        __riscv_vredminu_vs_u64m1_u64m1(min_cands_u64m1, init_umax_u64m1, vlmax));

    // Horizontal reduction for max
    vint64m1_t init_min_i64m1 = __riscv_vmv_v_x_i64m1(NK_I64_MIN, 1);
    nk_i64_t max_val = __riscv_vmv_x_s_i64m1_i64(__riscv_vredmax_vs_i64m1_i64m1(max_i64m1, init_min_i64m1, vlmax));
    vbool64_t max_match_b64 = __riscv_vmseq_vx_i64m1_b64(max_i64m1, max_val, vlmax);
    vuint64m1_t max_cands_u64m1 = __riscv_vmerge_vvm_u64m1(sentinel_u64m1, max_indices_u64m1, max_match_b64, vlmax);
    *max_value_ptr = max_val;
    *max_index_ptr = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(
        __riscv_vredminu_vs_u64m1_u64m1(max_cands_u64m1, init_umax_u64m1, vlmax));
}

NK_PUBLIC void nk_reduce_minmax_i64_rvv(                               //
    nk_i64_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *min_value_ptr, nk_size_t *min_index_ptr,                 //
    nk_i64_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_i64_t);
    int aligned = (stride_bytes % sizeof(nk_i64_t) == 0);

    if (count == 0)
        *min_value_ptr = NK_I64_MAX, *min_index_ptr = NK_SIZE_MAX, *max_value_ptr = NK_I64_MIN,
        *max_index_ptr = NK_SIZE_MAX;
    else if (!aligned)
        nk_reduce_minmax_i64_serial(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                    max_index_ptr);
    else if (stride_elements == 1)
        nk_reduce_minmax_i64_rvv_contiguous_(data_ptr, count, min_value_ptr, min_index_ptr, max_value_ptr,
                                             max_index_ptr);
    else
        nk_reduce_minmax_i64_rvv_strided_(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                          max_index_ptr);
}

NK_INTERNAL void nk_reduce_moments_u64_rvv_contiguous_( //
    nk_u64_t const *data_ptr, nk_size_t count,          //
    nk_u64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e64m1();
    vuint64m1_t sum_u64m1 = __riscv_vmv_v_x_u64m1(0, vlmax);
    vuint64m1_t sumsq_u64m1 = __riscv_vmv_v_x_u64m1(0, vlmax);

    for (nk_size_t vl; count > 0; count -= vl, data_ptr += vl) {
        vl = __riscv_vsetvl_e64m1(count);
        vuint64m1_t data_u64m1 = __riscv_vle64_v_u64m1(data_ptr, vl);

        // Direct accumulation (no widening)
        sum_u64m1 = __riscv_vadd_vv_u64m1(sum_u64m1, data_u64m1, vl);

        // Sumsq: u64 × u64 → u64
        vuint64m1_t squares_u64m1 = __riscv_vmul_vv_u64m1(data_u64m1, data_u64m1, vl);
        sumsq_u64m1 = __riscv_vadd_vv_u64m1(sumsq_u64m1, squares_u64m1, vl);
    }

    // Horizontal reduction
    vuint64m1_t zero_u64m1 = __riscv_vmv_v_x_u64m1(0, 1);
    *sum_ptr = __riscv_vmv_x_s_u64m1_u64(__riscv_vredsum_vs_u64m1_u64m1(sum_u64m1, zero_u64m1, vlmax)),
    *sumsq_ptr = __riscv_vmv_x_s_u64m1_u64(__riscv_vredsum_vs_u64m1_u64m1(sumsq_u64m1, zero_u64m1, vlmax));
}

NK_INTERNAL void nk_reduce_moments_u64_rvv_strided_(                   //
    nk_u64_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e64m1();
    vuint64m1_t sum_u64m1 = __riscv_vmv_v_x_u64m1(0, vlmax);
    vuint64m1_t sumsq_u64m1 = __riscv_vmv_v_x_u64m1(0, vlmax);
    unsigned char const *ptr = (unsigned char const *)data_ptr;

    for (nk_size_t vl; count > 0; count -= vl, ptr += vl * stride_bytes) {
        vl = __riscv_vsetvl_e64m1(count);
        vuint64m1_t data_u64m1 = __riscv_vlse64_v_u64m1((nk_u64_t const *)ptr, (nk_ssize_t)stride_bytes, vl);

        // Direct accumulation (no widening)
        sum_u64m1 = __riscv_vadd_vv_u64m1(sum_u64m1, data_u64m1, vl);

        // Sumsq: u64 × u64 → u64
        vuint64m1_t squares_u64m1 = __riscv_vmul_vv_u64m1(data_u64m1, data_u64m1, vl);
        sumsq_u64m1 = __riscv_vadd_vv_u64m1(sumsq_u64m1, squares_u64m1, vl);
    }

    // Horizontal reduction
    vuint64m1_t zero_u64m1 = __riscv_vmv_v_x_u64m1(0, 1);
    *sum_ptr = __riscv_vmv_x_s_u64m1_u64(__riscv_vredsum_vs_u64m1_u64m1(sum_u64m1, zero_u64m1, vlmax)),
    *sumsq_ptr = __riscv_vmv_x_s_u64m1_u64(__riscv_vredsum_vs_u64m1_u64m1(sumsq_u64m1, zero_u64m1, vlmax));
}

NK_PUBLIC void nk_reduce_moments_u64_rvv(                              //
    nk_u64_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_u64_t);
    int aligned = (stride_bytes % sizeof(nk_u64_t) == 0);

    if (count == 0) *sum_ptr = 0, *sumsq_ptr = 0;
    else if (!aligned) { nk_reduce_moments_u64_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr); }
    else if (stride_elements == 1) { nk_reduce_moments_u64_rvv_contiguous_(data_ptr, count, sum_ptr, sumsq_ptr); }
    else { nk_reduce_moments_u64_rvv_strided_(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr); }
}

NK_INTERNAL void nk_reduce_minmax_u64_rvv_contiguous_( //
    nk_u64_t const *data_ptr, nk_size_t count,         //
    nk_u64_t *min_value_ptr, nk_size_t *min_index_ptr, //
    nk_u64_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e64m1();
    vuint64m1_t min_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, vlmax);
    vuint64m1_t max_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MIN, vlmax);
    vuint64m1_t min_indices_u64m1 = __riscv_vmv_v_x_u64m1(0, vlmax);
    vuint64m1_t max_indices_u64m1 = __riscv_vmv_v_x_u64m1(0, vlmax);

    nk_size_t offset = 0;
    for (nk_size_t vl; count > 0; count -= vl, offset += vl, data_ptr += vl) {
        vl = __riscv_vsetvl_e64m1(count);
        vuint64m1_t data_u64m1 = __riscv_vle64_v_u64m1(data_ptr, vl);
        vuint64m1_t pos_u64m1 = __riscv_vadd_vx_u64m1(__riscv_vid_v_u64m1(vl), (nk_u64_t)offset, vl);

        vbool64_t less_b64 = __riscv_vmsltu_vv_u64m1_b64(data_u64m1, min_u64m1, vl);
        min_u64m1 = __riscv_vmerge_vvm_u64m1(min_u64m1, data_u64m1, less_b64, vl);
        min_indices_u64m1 = __riscv_vmerge_vvm_u64m1(min_indices_u64m1, pos_u64m1, less_b64, vl);

        vbool64_t greater_b64 = __riscv_vmsltu_vv_u64m1_b64(max_u64m1, data_u64m1, vl);
        max_u64m1 = __riscv_vmerge_vvm_u64m1(max_u64m1, data_u64m1, greater_b64, vl);
        max_indices_u64m1 = __riscv_vmerge_vvm_u64m1(max_indices_u64m1, pos_u64m1, greater_b64, vl);
    }

    // Horizontal reduction for min
    vuint64m1_t init_max_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    nk_u64_t min_val = __riscv_vmv_x_s_u64m1_u64(__riscv_vredminu_vs_u64m1_u64m1(min_u64m1, init_max_u64m1, vlmax));
    vbool64_t min_match_b64 = __riscv_vmseq_vx_u64m1_b64(min_u64m1, min_val, vlmax);
    vuint64m1_t sentinel_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, vlmax);
    vuint64m1_t min_cands_u64m1 = __riscv_vmerge_vvm_u64m1(sentinel_u64m1, min_indices_u64m1, min_match_b64, vlmax);
    vuint64m1_t init_umax_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    *min_value_ptr = min_val;
    *min_index_ptr = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(
        __riscv_vredminu_vs_u64m1_u64m1(min_cands_u64m1, init_umax_u64m1, vlmax));

    // Horizontal reduction for max
    vuint64m1_t init_min_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MIN, 1);
    nk_u64_t max_val = __riscv_vmv_x_s_u64m1_u64(__riscv_vredmaxu_vs_u64m1_u64m1(max_u64m1, init_min_u64m1, vlmax));
    vbool64_t max_match_b64 = __riscv_vmseq_vx_u64m1_b64(max_u64m1, max_val, vlmax);
    vuint64m1_t max_cands_u64m1 = __riscv_vmerge_vvm_u64m1(sentinel_u64m1, max_indices_u64m1, max_match_b64, vlmax);
    *max_value_ptr = max_val;
    *max_index_ptr = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(
        __riscv_vredminu_vs_u64m1_u64m1(max_cands_u64m1, init_umax_u64m1, vlmax));
}

NK_INTERNAL void nk_reduce_minmax_u64_rvv_strided_(                    //
    nk_u64_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *min_value_ptr, nk_size_t *min_index_ptr,                 //
    nk_u64_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e64m1();
    vuint64m1_t min_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, vlmax);
    vuint64m1_t max_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MIN, vlmax);
    vuint64m1_t min_indices_u64m1 = __riscv_vmv_v_x_u64m1(0, vlmax);
    vuint64m1_t max_indices_u64m1 = __riscv_vmv_v_x_u64m1(0, vlmax);
    unsigned char const *ptr = (unsigned char const *)data_ptr;

    nk_size_t offset = 0;
    for (nk_size_t vl; count > 0; count -= vl, offset += vl, ptr += vl * stride_bytes) {
        vl = __riscv_vsetvl_e64m1(count);
        vuint64m1_t data_u64m1 = __riscv_vlse64_v_u64m1((nk_u64_t const *)ptr, (nk_ssize_t)stride_bytes, vl);
        vuint64m1_t pos_u64m1 = __riscv_vadd_vx_u64m1(__riscv_vid_v_u64m1(vl), (nk_u64_t)offset, vl);

        vbool64_t less_b64 = __riscv_vmsltu_vv_u64m1_b64(data_u64m1, min_u64m1, vl);
        min_u64m1 = __riscv_vmerge_vvm_u64m1(min_u64m1, data_u64m1, less_b64, vl);
        min_indices_u64m1 = __riscv_vmerge_vvm_u64m1(min_indices_u64m1, pos_u64m1, less_b64, vl);

        vbool64_t greater_b64 = __riscv_vmsltu_vv_u64m1_b64(max_u64m1, data_u64m1, vl);
        max_u64m1 = __riscv_vmerge_vvm_u64m1(max_u64m1, data_u64m1, greater_b64, vl);
        max_indices_u64m1 = __riscv_vmerge_vvm_u64m1(max_indices_u64m1, pos_u64m1, greater_b64, vl);
    }

    // Horizontal reduction for min
    vuint64m1_t init_max_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    nk_u64_t min_val = __riscv_vmv_x_s_u64m1_u64(__riscv_vredminu_vs_u64m1_u64m1(min_u64m1, init_max_u64m1, vlmax));
    vbool64_t min_match_b64 = __riscv_vmseq_vx_u64m1_b64(min_u64m1, min_val, vlmax);
    vuint64m1_t sentinel_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, vlmax);
    vuint64m1_t min_cands_u64m1 = __riscv_vmerge_vvm_u64m1(sentinel_u64m1, min_indices_u64m1, min_match_b64, vlmax);
    vuint64m1_t init_umax_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    *min_value_ptr = min_val;
    *min_index_ptr = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(
        __riscv_vredminu_vs_u64m1_u64m1(min_cands_u64m1, init_umax_u64m1, vlmax));

    // Horizontal reduction for max
    vuint64m1_t init_min_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MIN, 1);
    nk_u64_t max_val = __riscv_vmv_x_s_u64m1_u64(__riscv_vredmaxu_vs_u64m1_u64m1(max_u64m1, init_min_u64m1, vlmax));
    vbool64_t max_match_b64 = __riscv_vmseq_vx_u64m1_b64(max_u64m1, max_val, vlmax);
    vuint64m1_t max_cands_u64m1 = __riscv_vmerge_vvm_u64m1(sentinel_u64m1, max_indices_u64m1, max_match_b64, vlmax);
    *max_value_ptr = max_val;
    *max_index_ptr = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(
        __riscv_vredminu_vs_u64m1_u64m1(max_cands_u64m1, init_umax_u64m1, vlmax));
}

NK_PUBLIC void nk_reduce_minmax_u64_rvv(                               //
    nk_u64_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *min_value_ptr, nk_size_t *min_index_ptr,                 //
    nk_u64_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_u64_t);
    int aligned = (stride_bytes % sizeof(nk_u64_t) == 0);

    if (count == 0)
        *min_value_ptr = NK_U64_MAX, *min_index_ptr = NK_SIZE_MAX, *max_value_ptr = NK_U64_MIN,
        *max_index_ptr = NK_SIZE_MAX;
    else if (!aligned)
        nk_reduce_minmax_u64_serial(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                    max_index_ptr);
    else if (stride_elements == 1)
        nk_reduce_minmax_u64_rvv_contiguous_(data_ptr, count, min_value_ptr, min_index_ptr, max_value_ptr,
                                             max_index_ptr);
    else
        nk_reduce_minmax_u64_rvv_strided_(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                          max_index_ptr);
}

NK_INTERNAL void nk_reduce_moments_bf16_rvv_contiguous_( //
    nk_bf16_t const *data_ptr, nk_size_t count,          //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e64m4();
    vfloat64m4_t sum_f64m4 = __riscv_vfmv_v_f_f64m4(0.0, vlmax);
    vfloat64m4_t sumsq_f64m4 = __riscv_vfmv_v_f_f64m4(0.0, vlmax);

    for (nk_size_t vl; count > 0; count -= vl, data_ptr += vl) {
        vl = __riscv_vsetvl_e16m1(count);
        vuint16m1_t data_u16m1 = __riscv_vle16_v_u16m1((uint16_t const *)data_ptr, vl);

        // Convert bf16 → f32 (m1 → m2)
        vfloat32m2_t data_f32m2 = nk_bf16m1_to_f32m2_rvv_(data_u16m1, vl);

        // Widen f32 → f64 (m2 → m4)
        vfloat64m4_t data_f64m4 = __riscv_vfwcvt_f_f_v_f64m4(data_f32m2, vl);
        sum_f64m4 = __riscv_vfadd_vv_f64m4(sum_f64m4, data_f64m4, vl);

        // Sumsq via widening FMA: f32×f32 → f64
        sumsq_f64m4 = __riscv_vfwmacc_vv_f64m4(sumsq_f64m4, data_f32m2, data_f32m2, vl);
    }

    // Horizontal reduction
    vfloat64m1_t zero_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, 1);
    *sum_ptr = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m4_f64m1(sum_f64m4, zero_f64m1, vlmax)),
    *sumsq_ptr = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m4_f64m1(sumsq_f64m4, zero_f64m1, vlmax));
}

NK_INTERNAL void nk_reduce_moments_bf16_rvv_strided_(                   //
    nk_bf16_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e64m4();
    vfloat64m4_t sum_f64m4 = __riscv_vfmv_v_f_f64m4(0.0, vlmax);
    vfloat64m4_t sumsq_f64m4 = __riscv_vfmv_v_f_f64m4(0.0, vlmax);
    unsigned char const *ptr = (unsigned char const *)data_ptr;

    for (nk_size_t vl; count > 0; count -= vl, ptr += vl * stride_bytes) {
        vl = __riscv_vsetvl_e16m1(count);
        vuint16m1_t data_u16m1 = __riscv_vlse16_v_u16m1((uint16_t const *)ptr, (nk_ssize_t)stride_bytes, vl);

        // Convert bf16 → f32 (m1 → m2)
        vfloat32m2_t data_f32m2 = nk_bf16m1_to_f32m2_rvv_(data_u16m1, vl);

        // Widen f32 → f64 (m2 → m4)
        vfloat64m4_t data_f64m4 = __riscv_vfwcvt_f_f_v_f64m4(data_f32m2, vl);
        sum_f64m4 = __riscv_vfadd_vv_f64m4(sum_f64m4, data_f64m4, vl);

        // Sumsq via widening FMA: f32×f32 → f64
        sumsq_f64m4 = __riscv_vfwmacc_vv_f64m4(sumsq_f64m4, data_f32m2, data_f32m2, vl);
    }

    // Horizontal reduction
    vfloat64m1_t zero_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, 1);
    *sum_ptr = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m4_f64m1(sum_f64m4, zero_f64m1, vlmax)),
    *sumsq_ptr = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m4_f64m1(sumsq_f64m4, zero_f64m1, vlmax));
}

NK_PUBLIC void nk_reduce_moments_bf16_rvv(                              //
    nk_bf16_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_bf16_t);
    int aligned = (stride_bytes % sizeof(nk_bf16_t) == 0);

    if (count == 0) *sum_ptr = 0.0f, *sumsq_ptr = 0.0f;
    else if (!aligned) nk_reduce_moments_bf16_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
    else if (stride_elements == 1) nk_reduce_moments_bf16_rvv_contiguous_(data_ptr, count, sum_ptr, sumsq_ptr);
    else nk_reduce_moments_bf16_rvv_strided_(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
}

NK_INTERNAL void nk_reduce_minmax_bf16_rvv_contiguous_( //
    nk_bf16_t const *data_ptr, nk_size_t count,         //
    nk_bf16_t *min_value_ptr, nk_size_t *min_index_ptr, //
    nk_bf16_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e16m1();
    vuint16m1_t min_u16m1 = __riscv_vmv_v_x_u16m1(0x7F80, vlmax); // +inf in bf16
    vuint16m1_t max_u16m1 = __riscv_vmv_v_x_u16m1(0xFF80, vlmax); // -inf in bf16
    vuint64m4_t min_indices_u64m4 = __riscv_vmv_v_x_u64m4(0, vlmax);
    vuint64m4_t max_indices_u64m4 = __riscv_vmv_v_x_u64m4(0, vlmax);

    nk_size_t offset = 0;
    for (nk_size_t vl; count > 0; count -= vl, offset += vl, data_ptr += vl) {
        vl = __riscv_vsetvl_e16m1(count);
        vuint16m1_t data_u16m1 = __riscv_vle16_v_u16m1((uint16_t const *)data_ptr, vl);
        vuint64m4_t pos_u64m4 = __riscv_vadd_vx_u64m4(__riscv_vid_v_u64m4(vl), (nk_u64_t)offset, vl);

        // Convert to f32 for comparison
        vfloat32m2_t data_f32m2 = nk_bf16m1_to_f32m2_rvv_(data_u16m1, vl);
        vfloat32m2_t min_f32m2 = nk_bf16m1_to_f32m2_rvv_(min_u16m1, vl);
        vfloat32m2_t max_f32m2 = nk_bf16m1_to_f32m2_rvv_(max_u16m1, vl);

        vbool16_t less_b16 = __riscv_vmflt_vv_f32m2_b16(data_f32m2, min_f32m2, vl);
        min_u16m1 = __riscv_vmerge_vvm_u16m1(min_u16m1, data_u16m1, less_b16, vl);
        min_indices_u64m4 = __riscv_vmerge_vvm_u64m4(min_indices_u64m4, pos_u64m4, less_b16, vl);

        vbool16_t greater_b16 = __riscv_vmflt_vv_f32m2_b16(max_f32m2, data_f32m2, vl);
        max_u16m1 = __riscv_vmerge_vvm_u16m1(max_u16m1, data_u16m1, greater_b16, vl);
        max_indices_u64m4 = __riscv_vmerge_vvm_u64m4(max_indices_u64m4, pos_u64m4, greater_b16, vl);
    }

    // Horizontal reduction
    vfloat32m2_t final_min_f32m2 = nk_bf16m1_to_f32m2_rvv_(min_u16m1, vlmax);
    vfloat32m1_t init_max_f32m1 = __riscv_vfmv_v_f_f32m1(NK_F32_MAX, 1);
    nk_f32_t min_val_f32 = __riscv_vfmv_f_s_f32m1_f32(
        __riscv_vfredmin_vs_f32m2_f32m1(final_min_f32m2, init_max_f32m1, vlmax));

    vfloat32m2_t converted_f32m2 = nk_bf16m1_to_f32m2_rvv_(min_u16m1, vlmax);
    vbool16_t min_match_b16 = __riscv_vmfeq_vf_f32m2_b16(converted_f32m2, min_val_f32, vlmax);
    vuint64m4_t sentinel_u64m4 = __riscv_vmv_v_x_u64m4(NK_U64_MAX, vlmax);
    vuint64m4_t min_cands_u64m4 = __riscv_vmerge_vvm_u64m4(sentinel_u64m4, min_indices_u64m4, min_match_b16, vlmax);
    vuint64m1_t init_umax_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);

    vuint16m1_t min_vec_u16m1 = __riscv_vmerge_vxm_u16m1(min_u16m1, 0, min_match_b16, vlmax);
    nk_u16_t min_raw = __riscv_vmv_x_s_u16m1_u16(
        __riscv_vslidedown_vx_u16m1(min_vec_u16m1, __riscv_vfirst_m_b16(min_match_b16, vlmax), vlmax));
    *min_value_ptr = *(nk_bf16_t *)&min_raw;
    *min_index_ptr = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(
        __riscv_vredminu_vs_u64m4_u64m1(min_cands_u64m4, init_umax_u64m1, vlmax));

    // Similar for max
    vfloat32m2_t final_max_f32m2 = nk_bf16m1_to_f32m2_rvv_(max_u16m1, vlmax);
    vfloat32m1_t init_min_f32m1 = __riscv_vfmv_v_f_f32m1(NK_F32_MIN, 1);
    nk_f32_t max_val_f32 = __riscv_vfmv_f_s_f32m1_f32(
        __riscv_vfredmax_vs_f32m2_f32m1(final_max_f32m2, init_min_f32m1, vlmax));

    vbool16_t max_match_b16 = __riscv_vmfeq_vf_f32m2_b16(nk_bf16m1_to_f32m2_rvv_(max_u16m1, vlmax), max_val_f32, vlmax);
    vuint64m4_t max_cands_u64m4 = __riscv_vmerge_vvm_u64m4(sentinel_u64m4, max_indices_u64m4, max_match_b16, vlmax);

    vuint16m1_t max_vec_u16m1 = __riscv_vmerge_vxm_u16m1(max_u16m1, 0, max_match_b16, vlmax);
    nk_u16_t max_raw = __riscv_vmv_x_s_u16m1_u16(
        __riscv_vslidedown_vx_u16m1(max_vec_u16m1, __riscv_vfirst_m_b16(max_match_b16, vlmax), vlmax));
    *max_value_ptr = *(nk_bf16_t *)&max_raw;
    *max_index_ptr = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(
        __riscv_vredminu_vs_u64m4_u64m1(max_cands_u64m4, init_umax_u64m1, vlmax));
}

NK_INTERNAL void nk_reduce_minmax_bf16_rvv_strided_(                    //
    nk_bf16_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_bf16_t *min_value_ptr, nk_size_t *min_index_ptr,                 //
    nk_bf16_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e16m1();
    vuint16m1_t min_u16m1 = __riscv_vmv_v_x_u16m1(0x7F80, vlmax); // +inf in bf16
    vuint16m1_t max_u16m1 = __riscv_vmv_v_x_u16m1(0xFF80, vlmax); // -inf in bf16
    vuint64m4_t min_indices_u64m4 = __riscv_vmv_v_x_u64m4(0, vlmax);
    vuint64m4_t max_indices_u64m4 = __riscv_vmv_v_x_u64m4(0, vlmax);
    unsigned char const *ptr = (unsigned char const *)data_ptr;

    nk_size_t offset = 0;
    for (nk_size_t vl; count > 0; count -= vl, offset += vl, ptr += vl * stride_bytes) {
        vl = __riscv_vsetvl_e16m1(count);
        vuint16m1_t data_u16m1 = __riscv_vlse16_v_u16m1((uint16_t const *)ptr, (nk_ssize_t)stride_bytes, vl);
        vuint64m4_t pos_u64m4 = __riscv_vadd_vx_u64m4(__riscv_vid_v_u64m4(vl), (nk_u64_t)offset, vl);

        // Convert to f32 for comparison
        vfloat32m2_t data_f32m2 = nk_bf16m1_to_f32m2_rvv_(data_u16m1, vl);
        vfloat32m2_t min_f32m2 = nk_bf16m1_to_f32m2_rvv_(min_u16m1, vl);
        vfloat32m2_t max_f32m2 = nk_bf16m1_to_f32m2_rvv_(max_u16m1, vl);

        vbool16_t less_b16 = __riscv_vmflt_vv_f32m2_b16(data_f32m2, min_f32m2, vl);
        min_u16m1 = __riscv_vmerge_vvm_u16m1(min_u16m1, data_u16m1, less_b16, vl);
        min_indices_u64m4 = __riscv_vmerge_vvm_u64m4(min_indices_u64m4, pos_u64m4, less_b16, vl);

        vbool16_t greater_b16 = __riscv_vmflt_vv_f32m2_b16(max_f32m2, data_f32m2, vl);
        max_u16m1 = __riscv_vmerge_vvm_u16m1(max_u16m1, data_u16m1, greater_b16, vl);
        max_indices_u64m4 = __riscv_vmerge_vvm_u64m4(max_indices_u64m4, pos_u64m4, greater_b16, vl);
    }

    // Horizontal reduction (same as contiguous)
    vfloat32m2_t final_min_f32m2 = nk_bf16m1_to_f32m2_rvv_(min_u16m1, vlmax);
    vfloat32m1_t init_max_f32m1 = __riscv_vfmv_v_f_f32m1(NK_F32_MAX, 1);
    nk_f32_t min_val_f32 = __riscv_vfmv_f_s_f32m1_f32(
        __riscv_vfredmin_vs_f32m2_f32m1(final_min_f32m2, init_max_f32m1, vlmax));

    vfloat32m2_t converted_f32m2 = nk_bf16m1_to_f32m2_rvv_(min_u16m1, vlmax);
    vbool16_t min_match_b16 = __riscv_vmfeq_vf_f32m2_b16(converted_f32m2, min_val_f32, vlmax);
    vuint64m4_t sentinel_u64m4 = __riscv_vmv_v_x_u64m4(NK_U64_MAX, vlmax);
    vuint64m4_t min_cands_u64m4 = __riscv_vmerge_vvm_u64m4(sentinel_u64m4, min_indices_u64m4, min_match_b16, vlmax);
    vuint64m1_t init_umax_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);

    vuint16m1_t min_vec_u16m1 = __riscv_vmerge_vxm_u16m1(min_u16m1, 0, min_match_b16, vlmax);
    nk_u16_t min_raw = __riscv_vmv_x_s_u16m1_u16(
        __riscv_vslidedown_vx_u16m1(min_vec_u16m1, __riscv_vfirst_m_b16(min_match_b16, vlmax), vlmax));
    *min_value_ptr = *(nk_bf16_t *)&min_raw;
    *min_index_ptr = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(
        __riscv_vredminu_vs_u64m4_u64m1(min_cands_u64m4, init_umax_u64m1, vlmax));

    vfloat32m2_t final_max_f32m2 = nk_bf16m1_to_f32m2_rvv_(max_u16m1, vlmax);
    vfloat32m1_t init_min_f32m1 = __riscv_vfmv_v_f_f32m1(NK_F32_MIN, 1);
    nk_f32_t max_val_f32 = __riscv_vfmv_f_s_f32m1_f32(
        __riscv_vfredmax_vs_f32m2_f32m1(final_max_f32m2, init_min_f32m1, vlmax));

    vbool16_t max_match_b16 = __riscv_vmfeq_vf_f32m2_b16(nk_bf16m1_to_f32m2_rvv_(max_u16m1, vlmax), max_val_f32, vlmax);
    vuint64m4_t max_cands_u64m4 = __riscv_vmerge_vvm_u64m4(sentinel_u64m4, max_indices_u64m4, max_match_b16, vlmax);

    vuint16m1_t max_vec_u16m1 = __riscv_vmerge_vxm_u16m1(max_u16m1, 0, max_match_b16, vlmax);
    nk_u16_t max_raw = __riscv_vmv_x_s_u16m1_u16(
        __riscv_vslidedown_vx_u16m1(max_vec_u16m1, __riscv_vfirst_m_b16(max_match_b16, vlmax), vlmax));
    *max_value_ptr = *(nk_bf16_t *)&max_raw;
    *max_index_ptr = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(
        __riscv_vredminu_vs_u64m4_u64m1(max_cands_u64m4, init_umax_u64m1, vlmax));
}

NK_PUBLIC void nk_reduce_minmax_bf16_rvv(                               //
    nk_bf16_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_bf16_t *min_value_ptr, nk_size_t *min_index_ptr,                 //
    nk_bf16_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_bf16_t);
    int aligned = (stride_bytes % sizeof(nk_bf16_t) == 0);

    if (count == 0)
        *min_value_ptr = NK_BF16_MAX, *min_index_ptr = NK_SIZE_MAX, *max_value_ptr = NK_BF16_MIN,
        *max_index_ptr = NK_SIZE_MAX;
    else if (!aligned)
        nk_reduce_minmax_bf16_serial(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                     max_index_ptr);
    else if (stride_elements == 1)
        nk_reduce_minmax_bf16_rvv_contiguous_(data_ptr, count, min_value_ptr, min_index_ptr, max_value_ptr,
                                              max_index_ptr);
    else
        nk_reduce_minmax_bf16_rvv_strided_(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                           max_index_ptr);
}

NK_INTERNAL void nk_reduce_moments_f16_rvv_contiguous_( //
    nk_f16_t const *data_ptr, nk_size_t count,          //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e64m4();
    vfloat64m4_t sum_f64m4 = __riscv_vfmv_v_f_f64m4(0.0, vlmax);
    vfloat64m4_t sumsq_f64m4 = __riscv_vfmv_v_f_f64m4(0.0, vlmax);

    for (nk_size_t vl; count > 0; count -= vl, data_ptr += vl) {
        vl = __riscv_vsetvl_e16m1(count);
        vuint16m1_t data_u16m1 = __riscv_vle16_v_u16m1((uint16_t const *)data_ptr, vl);

        // Convert f16 → f32 (m1 → m2)
        vfloat32m2_t data_f32m2 = nk_f16m1_to_f32m2_rvv_(data_u16m1, vl);

        // Widen f32 → f64 (m2 → m4)
        vfloat64m4_t data_f64m4 = __riscv_vfwcvt_f_f_v_f64m4(data_f32m2, vl);
        sum_f64m4 = __riscv_vfadd_vv_f64m4(sum_f64m4, data_f64m4, vl);

        // Sumsq via widening FMA: f32×f32 → f64
        sumsq_f64m4 = __riscv_vfwmacc_vv_f64m4(sumsq_f64m4, data_f32m2, data_f32m2, vl);
    }

    // Horizontal reduction
    vfloat64m1_t zero_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, 1);
    *sum_ptr = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m4_f64m1(sum_f64m4, zero_f64m1, vlmax)),
    *sumsq_ptr = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m4_f64m1(sumsq_f64m4, zero_f64m1, vlmax));
}

NK_INTERNAL void nk_reduce_moments_f16_rvv_strided_(                   //
    nk_f16_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e64m4();
    vfloat64m4_t sum_f64m4 = __riscv_vfmv_v_f_f64m4(0.0, vlmax);
    vfloat64m4_t sumsq_f64m4 = __riscv_vfmv_v_f_f64m4(0.0, vlmax);
    unsigned char const *ptr = (unsigned char const *)data_ptr;

    for (nk_size_t vl; count > 0; count -= vl, ptr += vl * stride_bytes) {
        vl = __riscv_vsetvl_e16m1(count);
        vuint16m1_t data_u16m1 = __riscv_vlse16_v_u16m1((uint16_t const *)ptr, (nk_ssize_t)stride_bytes, vl);

        // Convert f16 → f32 (m1 → m2)
        vfloat32m2_t data_f32m2 = nk_f16m1_to_f32m2_rvv_(data_u16m1, vl);

        // Widen f32 → f64 (m2 → m4)
        vfloat64m4_t data_f64m4 = __riscv_vfwcvt_f_f_v_f64m4(data_f32m2, vl);
        sum_f64m4 = __riscv_vfadd_vv_f64m4(sum_f64m4, data_f64m4, vl);

        // Sumsq via widening FMA: f32×f32 → f64
        sumsq_f64m4 = __riscv_vfwmacc_vv_f64m4(sumsq_f64m4, data_f32m2, data_f32m2, vl);
    }

    // Horizontal reduction
    vfloat64m1_t zero_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, 1);
    *sum_ptr = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m4_f64m1(sum_f64m4, zero_f64m1, vlmax)),
    *sumsq_ptr = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m4_f64m1(sumsq_f64m4, zero_f64m1, vlmax));
}

NK_PUBLIC void nk_reduce_moments_f16_rvv(                              //
    nk_f16_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_f16_t);
    int aligned = (stride_bytes % sizeof(nk_f16_t) == 0);

    if (count == 0) *sum_ptr = 0.0f, *sumsq_ptr = 0.0f;
    else if (!aligned) nk_reduce_moments_f16_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
    else if (stride_elements == 1) nk_reduce_moments_f16_rvv_contiguous_(data_ptr, count, sum_ptr, sumsq_ptr);
    else nk_reduce_moments_f16_rvv_strided_(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
}

NK_INTERNAL void nk_reduce_minmax_f16_rvv_contiguous_( //
    nk_f16_t const *data_ptr, nk_size_t count,         //
    nk_f16_t *min_value_ptr, nk_size_t *min_index_ptr, //
    nk_f16_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e16m1();
    vuint16m1_t min_u16m1 = __riscv_vmv_v_x_u16m1(0x7C00, vlmax); // +inf in f16
    vuint16m1_t max_u16m1 = __riscv_vmv_v_x_u16m1(0xFC00, vlmax); // -inf in f16
    vuint64m4_t min_indices_u64m4 = __riscv_vmv_v_x_u64m4(0, vlmax);
    vuint64m4_t max_indices_u64m4 = __riscv_vmv_v_x_u64m4(0, vlmax);

    nk_size_t offset = 0;
    for (nk_size_t vl; count > 0; count -= vl, offset += vl, data_ptr += vl) {
        vl = __riscv_vsetvl_e16m1(count);
        vuint16m1_t data_u16m1 = __riscv_vle16_v_u16m1((uint16_t const *)data_ptr, vl);
        vuint64m4_t pos_u64m4 = __riscv_vadd_vx_u64m4(__riscv_vid_v_u64m4(vl), (nk_u64_t)offset, vl);

        // Convert to f32 for comparison
        vfloat32m2_t data_f32m2 = nk_f16m1_to_f32m2_rvv_(data_u16m1, vl);
        vfloat32m2_t min_f32m2 = nk_f16m1_to_f32m2_rvv_(min_u16m1, vl);
        vfloat32m2_t max_f32m2 = nk_f16m1_to_f32m2_rvv_(max_u16m1, vl);

        vbool16_t less_b16 = __riscv_vmflt_vv_f32m2_b16(data_f32m2, min_f32m2, vl);
        min_u16m1 = __riscv_vmerge_vvm_u16m1(min_u16m1, data_u16m1, less_b16, vl);
        min_indices_u64m4 = __riscv_vmerge_vvm_u64m4(min_indices_u64m4, pos_u64m4, less_b16, vl);

        vbool16_t greater_b16 = __riscv_vmflt_vv_f32m2_b16(max_f32m2, data_f32m2, vl);
        max_u16m1 = __riscv_vmerge_vvm_u16m1(max_u16m1, data_u16m1, greater_b16, vl);
        max_indices_u64m4 = __riscv_vmerge_vvm_u64m4(max_indices_u64m4, pos_u64m4, greater_b16, vl);
    }

    // Horizontal reduction
    vfloat32m2_t final_min_f32m2 = nk_f16m1_to_f32m2_rvv_(min_u16m1, vlmax);
    vfloat32m1_t init_max_f32m1 = __riscv_vfmv_v_f_f32m1(NK_F32_MAX, 1);
    nk_f32_t min_val_f32 = __riscv_vfmv_f_s_f32m1_f32(
        __riscv_vfredmin_vs_f32m2_f32m1(final_min_f32m2, init_max_f32m1, vlmax));

    vfloat32m2_t converted_f32m2 = nk_f16m1_to_f32m2_rvv_(min_u16m1, vlmax);
    vbool16_t min_match_b16 = __riscv_vmfeq_vf_f32m2_b16(converted_f32m2, min_val_f32, vlmax);
    vuint64m4_t sentinel_u64m4 = __riscv_vmv_v_x_u64m4(NK_U64_MAX, vlmax);
    vuint64m4_t min_cands_u64m4 = __riscv_vmerge_vvm_u64m4(sentinel_u64m4, min_indices_u64m4, min_match_b16, vlmax);
    vuint64m1_t init_umax_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);

    vuint16m1_t min_vec_u16m1 = __riscv_vmerge_vxm_u16m1(min_u16m1, 0, min_match_b16, vlmax);
    nk_u16_t min_raw = __riscv_vmv_x_s_u16m1_u16(
        __riscv_vslidedown_vx_u16m1(min_vec_u16m1, __riscv_vfirst_m_b16(min_match_b16, vlmax), vlmax));
    *min_value_ptr = *(nk_f16_t *)&min_raw;
    *min_index_ptr = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(
        __riscv_vredminu_vs_u64m4_u64m1(min_cands_u64m4, init_umax_u64m1, vlmax));

    // Similar for max
    vfloat32m2_t final_max_f32m2 = nk_f16m1_to_f32m2_rvv_(max_u16m1, vlmax);
    vfloat32m1_t init_min_f32m1 = __riscv_vfmv_v_f_f32m1(NK_F32_MIN, 1);
    nk_f32_t max_val_f32 = __riscv_vfmv_f_s_f32m1_f32(
        __riscv_vfredmax_vs_f32m2_f32m1(final_max_f32m2, init_min_f32m1, vlmax));

    vbool16_t max_match_b16 = __riscv_vmfeq_vf_f32m2_b16(nk_f16m1_to_f32m2_rvv_(max_u16m1, vlmax), max_val_f32, vlmax);
    vuint64m4_t max_cands_u64m4 = __riscv_vmerge_vvm_u64m4(sentinel_u64m4, max_indices_u64m4, max_match_b16, vlmax);

    vuint16m1_t max_vec_u16m1 = __riscv_vmerge_vxm_u16m1(max_u16m1, 0, max_match_b16, vlmax);
    nk_u16_t max_raw = __riscv_vmv_x_s_u16m1_u16(
        __riscv_vslidedown_vx_u16m1(max_vec_u16m1, __riscv_vfirst_m_b16(max_match_b16, vlmax), vlmax));
    *max_value_ptr = *(nk_f16_t *)&max_raw;
    *max_index_ptr = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(
        __riscv_vredminu_vs_u64m4_u64m1(max_cands_u64m4, init_umax_u64m1, vlmax));
}

NK_INTERNAL void nk_reduce_minmax_f16_rvv_strided_(                    //
    nk_f16_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_f16_t *min_value_ptr, nk_size_t *min_index_ptr,                 //
    nk_f16_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e16m1();
    vuint16m1_t min_u16m1 = __riscv_vmv_v_x_u16m1(0x7C00, vlmax); // +inf in f16
    vuint16m1_t max_u16m1 = __riscv_vmv_v_x_u16m1(0xFC00, vlmax); // -inf in f16
    vuint64m4_t min_indices_u64m4 = __riscv_vmv_v_x_u64m4(0, vlmax);
    vuint64m4_t max_indices_u64m4 = __riscv_vmv_v_x_u64m4(0, vlmax);
    unsigned char const *ptr = (unsigned char const *)data_ptr;

    nk_size_t offset = 0;
    for (nk_size_t vl; count > 0; count -= vl, offset += vl, ptr += vl * stride_bytes) {
        vl = __riscv_vsetvl_e16m1(count);
        vuint16m1_t data_u16m1 = __riscv_vlse16_v_u16m1((uint16_t const *)ptr, (nk_ssize_t)stride_bytes, vl);
        vuint64m4_t pos_u64m4 = __riscv_vadd_vx_u64m4(__riscv_vid_v_u64m4(vl), (nk_u64_t)offset, vl);

        // Convert to f32 for comparison
        vfloat32m2_t data_f32m2 = nk_f16m1_to_f32m2_rvv_(data_u16m1, vl);
        vfloat32m2_t min_f32m2 = nk_f16m1_to_f32m2_rvv_(min_u16m1, vl);
        vfloat32m2_t max_f32m2 = nk_f16m1_to_f32m2_rvv_(max_u16m1, vl);

        vbool16_t less_b16 = __riscv_vmflt_vv_f32m2_b16(data_f32m2, min_f32m2, vl);
        min_u16m1 = __riscv_vmerge_vvm_u16m1(min_u16m1, data_u16m1, less_b16, vl);
        min_indices_u64m4 = __riscv_vmerge_vvm_u64m4(min_indices_u64m4, pos_u64m4, less_b16, vl);

        vbool16_t greater_b16 = __riscv_vmflt_vv_f32m2_b16(max_f32m2, data_f32m2, vl);
        max_u16m1 = __riscv_vmerge_vvm_u16m1(max_u16m1, data_u16m1, greater_b16, vl);
        max_indices_u64m4 = __riscv_vmerge_vvm_u64m4(max_indices_u64m4, pos_u64m4, greater_b16, vl);
    }

    // Horizontal reduction (same as contiguous)
    vfloat32m2_t final_min_f32m2 = nk_f16m1_to_f32m2_rvv_(min_u16m1, vlmax);
    vfloat32m1_t init_max_f32m1 = __riscv_vfmv_v_f_f32m1(NK_F32_MAX, 1);
    nk_f32_t min_val_f32 = __riscv_vfmv_f_s_f32m1_f32(
        __riscv_vfredmin_vs_f32m2_f32m1(final_min_f32m2, init_max_f32m1, vlmax));

    vfloat32m2_t converted_f32m2 = nk_f16m1_to_f32m2_rvv_(min_u16m1, vlmax);
    vbool16_t min_match_b16 = __riscv_vmfeq_vf_f32m2_b16(converted_f32m2, min_val_f32, vlmax);
    vuint64m4_t sentinel_u64m4 = __riscv_vmv_v_x_u64m4(NK_U64_MAX, vlmax);
    vuint64m4_t min_cands_u64m4 = __riscv_vmerge_vvm_u64m4(sentinel_u64m4, min_indices_u64m4, min_match_b16, vlmax);
    vuint64m1_t init_umax_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);

    vuint16m1_t min_vec_u16m1 = __riscv_vmerge_vxm_u16m1(min_u16m1, 0, min_match_b16, vlmax);
    nk_u16_t min_raw = __riscv_vmv_x_s_u16m1_u16(
        __riscv_vslidedown_vx_u16m1(min_vec_u16m1, __riscv_vfirst_m_b16(min_match_b16, vlmax), vlmax));
    *min_value_ptr = *(nk_f16_t *)&min_raw;
    *min_index_ptr = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(
        __riscv_vredminu_vs_u64m4_u64m1(min_cands_u64m4, init_umax_u64m1, vlmax));

    vfloat32m2_t final_max_f32m2 = nk_f16m1_to_f32m2_rvv_(max_u16m1, vlmax);
    vfloat32m1_t init_min_f32m1 = __riscv_vfmv_v_f_f32m1(NK_F32_MIN, 1);
    nk_f32_t max_val_f32 = __riscv_vfmv_f_s_f32m1_f32(
        __riscv_vfredmax_vs_f32m2_f32m1(final_max_f32m2, init_min_f32m1, vlmax));

    vbool16_t max_match_b16 = __riscv_vmfeq_vf_f32m2_b16(nk_f16m1_to_f32m2_rvv_(max_u16m1, vlmax), max_val_f32, vlmax);
    vuint64m4_t max_cands_u64m4 = __riscv_vmerge_vvm_u64m4(sentinel_u64m4, max_indices_u64m4, max_match_b16, vlmax);

    vuint16m1_t max_vec_u16m1 = __riscv_vmerge_vxm_u16m1(max_u16m1, 0, max_match_b16, vlmax);
    nk_u16_t max_raw = __riscv_vmv_x_s_u16m1_u16(
        __riscv_vslidedown_vx_u16m1(max_vec_u16m1, __riscv_vfirst_m_b16(max_match_b16, vlmax), vlmax));
    *max_value_ptr = *(nk_f16_t *)&max_raw;
    *max_index_ptr = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(
        __riscv_vredminu_vs_u64m4_u64m1(max_cands_u64m4, init_umax_u64m1, vlmax));
}

NK_PUBLIC void nk_reduce_minmax_f16_rvv(                               //
    nk_f16_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_f16_t *min_value_ptr, nk_size_t *min_index_ptr,                 //
    nk_f16_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_f16_t);
    int aligned = (stride_bytes % sizeof(nk_f16_t) == 0);

    if (count == 0)
        *min_value_ptr = NK_F16_MAX, *min_index_ptr = NK_SIZE_MAX, *max_value_ptr = NK_F16_MIN,
        *max_index_ptr = NK_SIZE_MAX;
    else if (!aligned)
        nk_reduce_minmax_f16_serial(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                    max_index_ptr);
    else if (stride_elements == 1)
        nk_reduce_minmax_f16_rvv_contiguous_(data_ptr, count, min_value_ptr, min_index_ptr, max_value_ptr,
                                             max_index_ptr);
    else
        nk_reduce_minmax_f16_rvv_strided_(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                          max_index_ptr);
}

NK_INTERNAL void nk_reduce_moments_e4m3_rvv_contiguous_( //
    nk_e4m3_t const *data_ptr, nk_size_t count,          //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e32m4();
    vfloat32m4_t sum_f32m4 = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);
    vfloat32m4_t sumsq_f32m4 = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);

    for (nk_size_t vl; count > 0; count -= vl, data_ptr += vl) {
        vl = __riscv_vsetvl_e8m1(count);
        vuint8m1_t data_u8m1 = __riscv_vle8_v_u8m1((uint8_t const *)data_ptr, vl);

        // Convert e4m3 → f32 (m1 → m4)
        vfloat32m4_t data_f32m4 = nk_e4m3m1_to_f32m4_rvv_(data_u8m1, vl);

        // Accumulate at f32 precision
        sum_f32m4 = __riscv_vfadd_vv_f32m4(sum_f32m4, data_f32m4, vl);
        sumsq_f32m4 = __riscv_vfmacc_vv_f32m4(sumsq_f32m4, data_f32m4, data_f32m4, vl);
    }

    // Horizontal reduction
    vfloat32m1_t zero_f32m1 = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    *sum_ptr = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(sum_f32m4, zero_f32m1, vlmax)),
    *sumsq_ptr = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(sumsq_f32m4, zero_f32m1, vlmax));
}

NK_INTERNAL void nk_reduce_moments_e4m3_rvv_strided_(                   //
    nk_e4m3_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e32m4();
    vfloat32m4_t sum_f32m4 = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);
    vfloat32m4_t sumsq_f32m4 = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);
    unsigned char const *ptr = (unsigned char const *)data_ptr;

    for (nk_size_t vl; count > 0; count -= vl, ptr += vl * stride_bytes) {
        vl = __riscv_vsetvl_e8m1(count);
        vuint8m1_t data_u8m1 = __riscv_vlse8_v_u8m1((uint8_t const *)ptr, (nk_ssize_t)stride_bytes, vl);

        // Convert e4m3 → f32 (m1 → m4)
        vfloat32m4_t data_f32m4 = nk_e4m3m1_to_f32m4_rvv_(data_u8m1, vl);

        // Accumulate at f32 precision
        sum_f32m4 = __riscv_vfadd_vv_f32m4(sum_f32m4, data_f32m4, vl);
        sumsq_f32m4 = __riscv_vfmacc_vv_f32m4(sumsq_f32m4, data_f32m4, data_f32m4, vl);
    }

    // Horizontal reduction
    vfloat32m1_t zero_f32m1 = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    *sum_ptr = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(sum_f32m4, zero_f32m1, vlmax)),
    *sumsq_ptr = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(sumsq_f32m4, zero_f32m1, vlmax));
}

NK_PUBLIC void nk_reduce_moments_e4m3_rvv(                              //
    nk_e4m3_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_e4m3_t);
    int aligned = (stride_bytes % sizeof(nk_e4m3_t) == 0);

    if (count == 0) *sum_ptr = 0.0f, *sumsq_ptr = 0.0f;
    else if (!aligned) nk_reduce_moments_e4m3_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
    else if (stride_elements == 1) nk_reduce_moments_e4m3_rvv_contiguous_(data_ptr, count, sum_ptr, sumsq_ptr);
    else nk_reduce_moments_e4m3_rvv_strided_(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
}

NK_INTERNAL void nk_reduce_minmax_e4m3_rvv_contiguous_( //
    nk_e4m3_t const *data_ptr, nk_size_t count,         //
    nk_e4m3_t *min_value_ptr, nk_size_t *min_index_ptr, //
    nk_e4m3_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e8m1();
    vuint8m1_t min_u8m1 = __riscv_vmv_v_x_u8m1(0xFF, vlmax); // Largest comparable
    vuint8m1_t max_u8m1 = __riscv_vmv_v_x_u8m1(0x00, vlmax); // Smallest comparable
    vuint64m8_t min_indices_u64m8 = __riscv_vmv_v_x_u64m8(0, vlmax);
    vuint64m8_t max_indices_u64m8 = __riscv_vmv_v_x_u64m8(0, vlmax);

    nk_size_t offset = 0;
    for (nk_size_t vl; count > 0; count -= vl, offset += vl, data_ptr += vl) {
        vl = __riscv_vsetvl_e8m1(count);
        vuint8m1_t raw_u8m1 = __riscv_vle8_v_u8m1((uint8_t const *)data_ptr, vl);

        // Convert to comparable form
        vuint8m1_t comparable_u8m1 = nk_fp8m1_to_comparable_u8m1_rvv_(raw_u8m1, vl);
        vuint64m8_t pos_u64m8 = __riscv_vadd_vx_u64m8(__riscv_vid_v_u64m8(vl), (nk_u64_t)offset, vl);

        vbool8_t less_b8 = __riscv_vmsltu_vv_u8m1_b8(comparable_u8m1, min_u8m1, vl);
        min_u8m1 = __riscv_vmerge_vvm_u8m1(min_u8m1, comparable_u8m1, less_b8, vl);
        min_indices_u64m8 = __riscv_vmerge_vvm_u64m8(min_indices_u64m8, pos_u64m8, less_b8, vl);

        vbool8_t greater_b8 = __riscv_vmsltu_vv_u8m1_b8(max_u8m1, comparable_u8m1, vl);
        max_u8m1 = __riscv_vmerge_vvm_u8m1(max_u8m1, comparable_u8m1, greater_b8, vl);
        max_indices_u64m8 = __riscv_vmerge_vvm_u64m8(max_indices_u64m8, pos_u64m8, greater_b8, vl);
    }

    // Horizontal reduction + convert back
    vuint8m1_t init_max_u8m1 = __riscv_vmv_v_x_u8m1(0xFF, 1);
    nk_u8_t min_comparable = __riscv_vmv_x_s_u8m1_u8(__riscv_vredminu_vs_u8m1_u8m1(min_u8m1, init_max_u8m1, vlmax));
    vbool8_t min_match_b8 = __riscv_vmseq_vx_u8m1_b8(min_u8m1, min_comparable, vlmax);
    vuint64m8_t sentinel_u64m8 = __riscv_vmv_v_x_u64m8(NK_U64_MAX, vlmax);
    vuint64m8_t min_cands_u64m8 = __riscv_vmerge_vvm_u64m8(sentinel_u64m8, min_indices_u64m8, min_match_b8, vlmax);
    vuint64m1_t init_umax_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    *min_index_ptr = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(
        __riscv_vredminu_vs_u64m8_u64m1(min_cands_u64m8, init_umax_u64m1, vlmax));

    vuint8m1_t min_vec_u8m1 = __riscv_vmv_v_x_u8m1(min_comparable, 1);
    vuint8m1_t min_raw_u8m1 = nk_comparable_to_fp8m1_rvv_(min_vec_u8m1, 1);
    *min_value_ptr = (nk_e4m3_t)__riscv_vmv_x_s_u8m1_u8(min_raw_u8m1);

    // Similar for max
    vuint8m1_t init_min_u8m1 = __riscv_vmv_v_x_u8m1(0x00, 1);
    nk_u8_t max_comparable = __riscv_vmv_x_s_u8m1_u8(__riscv_vredmaxu_vs_u8m1_u8m1(max_u8m1, init_min_u8m1, vlmax));
    vbool8_t max_match_b8 = __riscv_vmseq_vx_u8m1_b8(max_u8m1, max_comparable, vlmax);
    vuint64m8_t max_cands_u64m8 = __riscv_vmerge_vvm_u64m8(sentinel_u64m8, max_indices_u64m8, max_match_b8, vlmax);
    *max_index_ptr = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(
        __riscv_vredminu_vs_u64m8_u64m1(max_cands_u64m8, init_umax_u64m1, vlmax));

    vuint8m1_t max_vec_u8m1 = __riscv_vmv_v_x_u8m1(max_comparable, 1);
    vuint8m1_t max_raw_u8m1 = nk_comparable_to_fp8m1_rvv_(max_vec_u8m1, 1);
    *max_value_ptr = (nk_e4m3_t)__riscv_vmv_x_s_u8m1_u8(max_raw_u8m1);
}

NK_INTERNAL void nk_reduce_minmax_e4m3_rvv_strided_(                    //
    nk_e4m3_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_e4m3_t *min_value_ptr, nk_size_t *min_index_ptr,                 //
    nk_e4m3_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e8m1();
    vuint8m1_t min_u8m1 = __riscv_vmv_v_x_u8m1(0xFF, vlmax);
    vuint8m1_t max_u8m1 = __riscv_vmv_v_x_u8m1(0x00, vlmax);
    vuint64m8_t min_indices_u64m8 = __riscv_vmv_v_x_u64m8(0, vlmax);
    vuint64m8_t max_indices_u64m8 = __riscv_vmv_v_x_u64m8(0, vlmax);
    unsigned char const *ptr = (unsigned char const *)data_ptr;

    nk_size_t offset = 0;
    for (nk_size_t vl; count > 0; count -= vl, offset += vl, ptr += vl * stride_bytes) {
        vl = __riscv_vsetvl_e8m1(count);
        vuint8m1_t raw_u8m1 = __riscv_vlse8_v_u8m1((uint8_t const *)ptr, (nk_ssize_t)stride_bytes, vl);

        vuint8m1_t comparable_u8m1 = nk_fp8m1_to_comparable_u8m1_rvv_(raw_u8m1, vl);
        vuint64m8_t pos_u64m8 = __riscv_vadd_vx_u64m8(__riscv_vid_v_u64m8(vl), (nk_u64_t)offset, vl);

        vbool8_t less_b8 = __riscv_vmsltu_vv_u8m1_b8(comparable_u8m1, min_u8m1, vl);
        min_u8m1 = __riscv_vmerge_vvm_u8m1(min_u8m1, comparable_u8m1, less_b8, vl);
        min_indices_u64m8 = __riscv_vmerge_vvm_u64m8(min_indices_u64m8, pos_u64m8, less_b8, vl);

        vbool8_t greater_b8 = __riscv_vmsltu_vv_u8m1_b8(max_u8m1, comparable_u8m1, vl);
        max_u8m1 = __riscv_vmerge_vvm_u8m1(max_u8m1, comparable_u8m1, greater_b8, vl);
        max_indices_u64m8 = __riscv_vmerge_vvm_u64m8(max_indices_u64m8, pos_u64m8, greater_b8, vl);
    }

    // Horizontal reduction (same as contiguous)
    vuint8m1_t init_max_u8m1 = __riscv_vmv_v_x_u8m1(0xFF, 1);
    nk_u8_t min_comparable = __riscv_vmv_x_s_u8m1_u8(__riscv_vredminu_vs_u8m1_u8m1(min_u8m1, init_max_u8m1, vlmax));
    vbool8_t min_match_b8 = __riscv_vmseq_vx_u8m1_b8(min_u8m1, min_comparable, vlmax);
    vuint64m8_t sentinel_u64m8 = __riscv_vmv_v_x_u64m8(NK_U64_MAX, vlmax);
    vuint64m8_t min_cands_u64m8 = __riscv_vmerge_vvm_u64m8(sentinel_u64m8, min_indices_u64m8, min_match_b8, vlmax);
    vuint64m1_t init_umax_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    *min_index_ptr = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(
        __riscv_vredminu_vs_u64m8_u64m1(min_cands_u64m8, init_umax_u64m1, vlmax));

    vuint8m1_t min_vec_u8m1 = __riscv_vmv_v_x_u8m1(min_comparable, 1);
    vuint8m1_t min_raw_u8m1 = nk_comparable_to_fp8m1_rvv_(min_vec_u8m1, 1);
    *min_value_ptr = (nk_e4m3_t)__riscv_vmv_x_s_u8m1_u8(min_raw_u8m1);

    vuint8m1_t init_min_u8m1 = __riscv_vmv_v_x_u8m1(0x00, 1);
    nk_u8_t max_comparable = __riscv_vmv_x_s_u8m1_u8(__riscv_vredmaxu_vs_u8m1_u8m1(max_u8m1, init_min_u8m1, vlmax));
    vbool8_t max_match_b8 = __riscv_vmseq_vx_u8m1_b8(max_u8m1, max_comparable, vlmax);
    vuint64m8_t max_cands_u64m8 = __riscv_vmerge_vvm_u64m8(sentinel_u64m8, max_indices_u64m8, max_match_b8, vlmax);
    *max_index_ptr = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(
        __riscv_vredminu_vs_u64m8_u64m1(max_cands_u64m8, init_umax_u64m1, vlmax));

    vuint8m1_t max_vec_u8m1 = __riscv_vmv_v_x_u8m1(max_comparable, 1);
    vuint8m1_t max_raw_u8m1 = nk_comparable_to_fp8m1_rvv_(max_vec_u8m1, 1);
    *max_value_ptr = (nk_e4m3_t)__riscv_vmv_x_s_u8m1_u8(max_raw_u8m1);
}

NK_PUBLIC void nk_reduce_minmax_e4m3_rvv(                               //
    nk_e4m3_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_e4m3_t *min_value_ptr, nk_size_t *min_index_ptr,                 //
    nk_e4m3_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_e4m3_t);
    int aligned = (stride_bytes % sizeof(nk_e4m3_t) == 0);

    if (count == 0)
        *min_value_ptr = NK_E4M3_MAX, *min_index_ptr = NK_SIZE_MAX, *max_value_ptr = NK_E4M3_MIN,
        *max_index_ptr = NK_SIZE_MAX;
    else if (!aligned)
        nk_reduce_minmax_e4m3_serial(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                     max_index_ptr);
    else if (stride_elements == 1)
        nk_reduce_minmax_e4m3_rvv_contiguous_(data_ptr, count, min_value_ptr, min_index_ptr, max_value_ptr,
                                              max_index_ptr);
    else
        nk_reduce_minmax_e4m3_rvv_strided_(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                           max_index_ptr);
}

NK_INTERNAL void nk_reduce_moments_e5m2_rvv_contiguous_( //
    nk_e5m2_t const *data_ptr, nk_size_t count,          //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e32m4();
    vfloat32m4_t sum_f32m4 = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);
    vfloat32m4_t sumsq_f32m4 = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);

    for (nk_size_t vl; count > 0; count -= vl, data_ptr += vl) {
        vl = __riscv_vsetvl_e8m1(count);
        vuint8m1_t data_u8m1 = __riscv_vle8_v_u8m1((uint8_t const *)data_ptr, vl);

        // Convert e5m2 → f32 (m1 → m4)
        vfloat32m4_t data_f32m4 = nk_e5m2m1_to_f32m4_rvv_(data_u8m1, vl);

        // Accumulate at f32 precision
        sum_f32m4 = __riscv_vfadd_vv_f32m4(sum_f32m4, data_f32m4, vl);
        sumsq_f32m4 = __riscv_vfmacc_vv_f32m4(sumsq_f32m4, data_f32m4, data_f32m4, vl);
    }

    // Horizontal reduction
    vfloat32m1_t zero_f32m1 = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    *sum_ptr = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(sum_f32m4, zero_f32m1, vlmax)),
    *sumsq_ptr = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(sumsq_f32m4, zero_f32m1, vlmax));
}

NK_INTERNAL void nk_reduce_moments_e5m2_rvv_strided_(                   //
    nk_e5m2_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e32m4();
    vfloat32m4_t sum_f32m4 = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);
    vfloat32m4_t sumsq_f32m4 = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);
    unsigned char const *ptr = (unsigned char const *)data_ptr;

    for (nk_size_t vl; count > 0; count -= vl, ptr += vl * stride_bytes) {
        vl = __riscv_vsetvl_e8m1(count);
        vuint8m1_t data_u8m1 = __riscv_vlse8_v_u8m1((uint8_t const *)ptr, (nk_ssize_t)stride_bytes, vl);

        // Convert e5m2 → f32 (m1 → m4)
        vfloat32m4_t data_f32m4 = nk_e5m2m1_to_f32m4_rvv_(data_u8m1, vl);

        // Accumulate at f32 precision
        sum_f32m4 = __riscv_vfadd_vv_f32m4(sum_f32m4, data_f32m4, vl);
        sumsq_f32m4 = __riscv_vfmacc_vv_f32m4(sumsq_f32m4, data_f32m4, data_f32m4, vl);
    }

    // Horizontal reduction
    vfloat32m1_t zero_f32m1 = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    *sum_ptr = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(sum_f32m4, zero_f32m1, vlmax)),
    *sumsq_ptr = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(sumsq_f32m4, zero_f32m1, vlmax));
}

NK_PUBLIC void nk_reduce_moments_e5m2_rvv(                              //
    nk_e5m2_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_e5m2_t);
    int aligned = (stride_bytes % sizeof(nk_e5m2_t) == 0);

    if (count == 0) *sum_ptr = 0.0f, *sumsq_ptr = 0.0f;
    else if (!aligned) nk_reduce_moments_e5m2_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
    else if (stride_elements == 1) nk_reduce_moments_e5m2_rvv_contiguous_(data_ptr, count, sum_ptr, sumsq_ptr);
    else nk_reduce_moments_e5m2_rvv_strided_(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
}

NK_INTERNAL void nk_reduce_minmax_e5m2_rvv_contiguous_( //
    nk_e5m2_t const *data_ptr, nk_size_t count,         //
    nk_e5m2_t *min_value_ptr, nk_size_t *min_index_ptr, //
    nk_e5m2_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e8m1();
    vuint8m1_t min_u8m1 = __riscv_vmv_v_x_u8m1(0xFF, vlmax);
    vuint8m1_t max_u8m1 = __riscv_vmv_v_x_u8m1(0x00, vlmax);
    vuint64m8_t min_indices_u64m8 = __riscv_vmv_v_x_u64m8(0, vlmax);
    vuint64m8_t max_indices_u64m8 = __riscv_vmv_v_x_u64m8(0, vlmax);

    nk_size_t offset = 0;
    for (nk_size_t vl; count > 0; count -= vl, offset += vl, data_ptr += vl) {
        vl = __riscv_vsetvl_e8m1(count);
        vuint8m1_t raw_u8m1 = __riscv_vle8_v_u8m1((uint8_t const *)data_ptr, vl);

        vuint8m1_t comparable_u8m1 = nk_fp8m1_to_comparable_u8m1_rvv_(raw_u8m1, vl);
        vuint64m8_t pos_u64m8 = __riscv_vadd_vx_u64m8(__riscv_vid_v_u64m8(vl), (nk_u64_t)offset, vl);

        vbool8_t less_b8 = __riscv_vmsltu_vv_u8m1_b8(comparable_u8m1, min_u8m1, vl);
        min_u8m1 = __riscv_vmerge_vvm_u8m1(min_u8m1, comparable_u8m1, less_b8, vl);
        min_indices_u64m8 = __riscv_vmerge_vvm_u64m8(min_indices_u64m8, pos_u64m8, less_b8, vl);

        vbool8_t greater_b8 = __riscv_vmsltu_vv_u8m1_b8(max_u8m1, comparable_u8m1, vl);
        max_u8m1 = __riscv_vmerge_vvm_u8m1(max_u8m1, comparable_u8m1, greater_b8, vl);
        max_indices_u64m8 = __riscv_vmerge_vvm_u64m8(max_indices_u64m8, pos_u64m8, greater_b8, vl);
    }

    // Horizontal reduction + convert back
    vuint8m1_t init_max_u8m1 = __riscv_vmv_v_x_u8m1(0xFF, 1);
    nk_u8_t min_comparable = __riscv_vmv_x_s_u8m1_u8(__riscv_vredminu_vs_u8m1_u8m1(min_u8m1, init_max_u8m1, vlmax));
    vbool8_t min_match_b8 = __riscv_vmseq_vx_u8m1_b8(min_u8m1, min_comparable, vlmax);
    vuint64m8_t sentinel_u64m8 = __riscv_vmv_v_x_u64m8(NK_U64_MAX, vlmax);
    vuint64m8_t min_cands_u64m8 = __riscv_vmerge_vvm_u64m8(sentinel_u64m8, min_indices_u64m8, min_match_b8, vlmax);
    vuint64m1_t init_umax_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    *min_index_ptr = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(
        __riscv_vredminu_vs_u64m8_u64m1(min_cands_u64m8, init_umax_u64m1, vlmax));

    vuint8m1_t min_vec_u8m1 = __riscv_vmv_v_x_u8m1(min_comparable, 1);
    vuint8m1_t min_raw_u8m1 = nk_comparable_to_fp8m1_rvv_(min_vec_u8m1, 1);
    *min_value_ptr = (nk_e5m2_t)__riscv_vmv_x_s_u8m1_u8(min_raw_u8m1);

    vuint8m1_t init_min_u8m1 = __riscv_vmv_v_x_u8m1(0x00, 1);
    nk_u8_t max_comparable = __riscv_vmv_x_s_u8m1_u8(__riscv_vredmaxu_vs_u8m1_u8m1(max_u8m1, init_min_u8m1, vlmax));
    vbool8_t max_match_b8 = __riscv_vmseq_vx_u8m1_b8(max_u8m1, max_comparable, vlmax);
    vuint64m8_t max_cands_u64m8 = __riscv_vmerge_vvm_u64m8(sentinel_u64m8, max_indices_u64m8, max_match_b8, vlmax);
    *max_index_ptr = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(
        __riscv_vredminu_vs_u64m8_u64m1(max_cands_u64m8, init_umax_u64m1, vlmax));

    vuint8m1_t max_vec_u8m1 = __riscv_vmv_v_x_u8m1(max_comparable, 1);
    vuint8m1_t max_raw_u8m1 = nk_comparable_to_fp8m1_rvv_(max_vec_u8m1, 1);
    *max_value_ptr = (nk_e5m2_t)__riscv_vmv_x_s_u8m1_u8(max_raw_u8m1);
}

NK_INTERNAL void nk_reduce_minmax_e5m2_rvv_strided_(                    //
    nk_e5m2_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_e5m2_t *min_value_ptr, nk_size_t *min_index_ptr,                 //
    nk_e5m2_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e8m1();
    vuint8m1_t min_u8m1 = __riscv_vmv_v_x_u8m1(0xFF, vlmax);
    vuint8m1_t max_u8m1 = __riscv_vmv_v_x_u8m1(0x00, vlmax);
    vuint64m8_t min_indices_u64m8 = __riscv_vmv_v_x_u64m8(0, vlmax);
    vuint64m8_t max_indices_u64m8 = __riscv_vmv_v_x_u64m8(0, vlmax);
    unsigned char const *ptr = (unsigned char const *)data_ptr;

    nk_size_t offset = 0;
    for (nk_size_t vl; count > 0; count -= vl, offset += vl, ptr += vl * stride_bytes) {
        vl = __riscv_vsetvl_e8m1(count);
        vuint8m1_t raw_u8m1 = __riscv_vlse8_v_u8m1((uint8_t const *)ptr, (nk_ssize_t)stride_bytes, vl);

        vuint8m1_t comparable_u8m1 = nk_fp8m1_to_comparable_u8m1_rvv_(raw_u8m1, vl);
        vuint64m8_t pos_u64m8 = __riscv_vadd_vx_u64m8(__riscv_vid_v_u64m8(vl), (nk_u64_t)offset, vl);

        vbool8_t less_b8 = __riscv_vmsltu_vv_u8m1_b8(comparable_u8m1, min_u8m1, vl);
        min_u8m1 = __riscv_vmerge_vvm_u8m1(min_u8m1, comparable_u8m1, less_b8, vl);
        min_indices_u64m8 = __riscv_vmerge_vvm_u64m8(min_indices_u64m8, pos_u64m8, less_b8, vl);

        vbool8_t greater_b8 = __riscv_vmsltu_vv_u8m1_b8(max_u8m1, comparable_u8m1, vl);
        max_u8m1 = __riscv_vmerge_vvm_u8m1(max_u8m1, comparable_u8m1, greater_b8, vl);
        max_indices_u64m8 = __riscv_vmerge_vvm_u64m8(max_indices_u64m8, pos_u64m8, greater_b8, vl);
    }

    // Horizontal reduction (same as contiguous)
    vuint8m1_t init_max_u8m1 = __riscv_vmv_v_x_u8m1(0xFF, 1);
    nk_u8_t min_comparable = __riscv_vmv_x_s_u8m1_u8(__riscv_vredminu_vs_u8m1_u8m1(min_u8m1, init_max_u8m1, vlmax));
    vbool8_t min_match_b8 = __riscv_vmseq_vx_u8m1_b8(min_u8m1, min_comparable, vlmax);
    vuint64m8_t sentinel_u64m8 = __riscv_vmv_v_x_u64m8(NK_U64_MAX, vlmax);
    vuint64m8_t min_cands_u64m8 = __riscv_vmerge_vvm_u64m8(sentinel_u64m8, min_indices_u64m8, min_match_b8, vlmax);
    vuint64m1_t init_umax_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    *min_index_ptr = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(
        __riscv_vredminu_vs_u64m8_u64m1(min_cands_u64m8, init_umax_u64m1, vlmax));

    vuint8m1_t min_vec_u8m1 = __riscv_vmv_v_x_u8m1(min_comparable, 1);
    vuint8m1_t min_raw_u8m1 = nk_comparable_to_fp8m1_rvv_(min_vec_u8m1, 1);
    *min_value_ptr = (nk_e5m2_t)__riscv_vmv_x_s_u8m1_u8(min_raw_u8m1);

    vuint8m1_t init_min_u8m1 = __riscv_vmv_v_x_u8m1(0x00, 1);
    nk_u8_t max_comparable = __riscv_vmv_x_s_u8m1_u8(__riscv_vredmaxu_vs_u8m1_u8m1(max_u8m1, init_min_u8m1, vlmax));
    vbool8_t max_match_b8 = __riscv_vmseq_vx_u8m1_b8(max_u8m1, max_comparable, vlmax);
    vuint64m8_t max_cands_u64m8 = __riscv_vmerge_vvm_u64m8(sentinel_u64m8, max_indices_u64m8, max_match_b8, vlmax);
    *max_index_ptr = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(
        __riscv_vredminu_vs_u64m8_u64m1(max_cands_u64m8, init_umax_u64m1, vlmax));

    vuint8m1_t max_vec_u8m1 = __riscv_vmv_v_x_u8m1(max_comparable, 1);
    vuint8m1_t max_raw_u8m1 = nk_comparable_to_fp8m1_rvv_(max_vec_u8m1, 1);
    *max_value_ptr = (nk_e5m2_t)__riscv_vmv_x_s_u8m1_u8(max_raw_u8m1);
}

NK_PUBLIC void nk_reduce_minmax_e5m2_rvv(                               //
    nk_e5m2_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_e5m2_t *min_value_ptr, nk_size_t *min_index_ptr,                 //
    nk_e5m2_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_e5m2_t);
    int aligned = (stride_bytes % sizeof(nk_e5m2_t) == 0);

    if (count == 0)
        *min_value_ptr = NK_E5M2_MAX, *min_index_ptr = NK_SIZE_MAX, *max_value_ptr = NK_E5M2_MIN,
        *max_index_ptr = NK_SIZE_MAX;
    else if (!aligned)
        nk_reduce_minmax_e5m2_serial(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                     max_index_ptr);
    else if (stride_elements == 1)
        nk_reduce_minmax_e5m2_rvv_contiguous_(data_ptr, count, min_value_ptr, min_index_ptr, max_value_ptr,
                                              max_index_ptr);
    else
        nk_reduce_minmax_e5m2_rvv_strided_(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                           max_index_ptr);
}

NK_INTERNAL void nk_reduce_moments_e2m3_rvv_contiguous_( //
    nk_e2m3_t const *data_ptr, nk_size_t count,          //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e32m4();
    vfloat32m4_t sum_f32m4 = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);
    vfloat32m4_t sumsq_f32m4 = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);

    for (nk_size_t vl; count > 0; count -= vl, data_ptr += vl) {
        vl = __riscv_vsetvl_e8m1(count);
        vuint8m1_t data_u8m1 = __riscv_vle8_v_u8m1((uint8_t const *)data_ptr, vl);

        // Convert e2m3 → f32 (m1 → m4)
        vfloat32m4_t data_f32m4 = nk_e2m3m1_to_f32m4_rvv_(data_u8m1, vl);

        // Accumulate at f32 precision
        sum_f32m4 = __riscv_vfadd_vv_f32m4(sum_f32m4, data_f32m4, vl);
        sumsq_f32m4 = __riscv_vfmacc_vv_f32m4(sumsq_f32m4, data_f32m4, data_f32m4, vl);
    }

    // Horizontal reduction
    vfloat32m1_t zero_f32m1 = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    *sum_ptr = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(sum_f32m4, zero_f32m1, vlmax)),
    *sumsq_ptr = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(sumsq_f32m4, zero_f32m1, vlmax));
}

NK_INTERNAL void nk_reduce_moments_e2m3_rvv_strided_(                   //
    nk_e2m3_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e32m4();
    vfloat32m4_t sum_f32m4 = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);
    vfloat32m4_t sumsq_f32m4 = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);
    unsigned char const *ptr = (unsigned char const *)data_ptr;

    for (nk_size_t vl; count > 0; count -= vl, ptr += vl * stride_bytes) {
        vl = __riscv_vsetvl_e8m1(count);
        vuint8m1_t data_u8m1 = __riscv_vlse8_v_u8m1((uint8_t const *)ptr, (nk_ssize_t)stride_bytes, vl);

        // Convert e2m3 → f32 (m1 → m4)
        vfloat32m4_t data_f32m4 = nk_e2m3m1_to_f32m4_rvv_(data_u8m1, vl);

        // Accumulate at f32 precision
        sum_f32m4 = __riscv_vfadd_vv_f32m4(sum_f32m4, data_f32m4, vl);
        sumsq_f32m4 = __riscv_vfmacc_vv_f32m4(sumsq_f32m4, data_f32m4, data_f32m4, vl);
    }

    // Horizontal reduction
    vfloat32m1_t zero_f32m1 = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    *sum_ptr = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(sum_f32m4, zero_f32m1, vlmax)),
    *sumsq_ptr = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(sumsq_f32m4, zero_f32m1, vlmax));
}

NK_PUBLIC void nk_reduce_moments_e2m3_rvv(                              //
    nk_e2m3_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_e2m3_t);
    int aligned = (stride_bytes % sizeof(nk_e2m3_t) == 0);

    if (count == 0) *sum_ptr = 0.0f, *sumsq_ptr = 0.0f;
    else if (!aligned) nk_reduce_moments_e2m3_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
    else if (stride_elements == 1) nk_reduce_moments_e2m3_rvv_contiguous_(data_ptr, count, sum_ptr, sumsq_ptr);
    else nk_reduce_moments_e2m3_rvv_strided_(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
}

NK_INTERNAL void nk_reduce_minmax_e2m3_rvv_contiguous_( //
    nk_e2m3_t const *data_ptr, nk_size_t count,         //
    nk_e2m3_t *min_value_ptr, nk_size_t *min_index_ptr, //
    nk_e2m3_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e8m1();
    vuint8m1_t min_u8m1 = __riscv_vmv_v_x_u8m1(0x3F, vlmax); // Largest FP6 comparable
    vuint8m1_t max_u8m1 = __riscv_vmv_v_x_u8m1(0x00, vlmax); // Smallest FP6 comparable
    vuint64m8_t min_indices_u64m8 = __riscv_vmv_v_x_u64m8(0, vlmax);
    vuint64m8_t max_indices_u64m8 = __riscv_vmv_v_x_u64m8(0, vlmax);

    nk_size_t offset = 0;
    for (nk_size_t vl; count > 0; count -= vl, offset += vl, data_ptr += vl) {
        vl = __riscv_vsetvl_e8m1(count);
        vuint8m1_t raw_u8m1 = __riscv_vle8_v_u8m1((uint8_t const *)data_ptr, vl);

        // Convert to FP6 comparable form
        vuint8m1_t comparable_u8m1 = nk_fp6m1_to_comparable_u8m1_rvv_(raw_u8m1, vl);
        vuint64m8_t pos_u64m8 = __riscv_vadd_vx_u64m8(__riscv_vid_v_u64m8(vl), (nk_u64_t)offset, vl);

        vbool8_t less_b8 = __riscv_vmsltu_vv_u8m1_b8(comparable_u8m1, min_u8m1, vl);
        min_u8m1 = __riscv_vmerge_vvm_u8m1(min_u8m1, comparable_u8m1, less_b8, vl);
        min_indices_u64m8 = __riscv_vmerge_vvm_u64m8(min_indices_u64m8, pos_u64m8, less_b8, vl);

        vbool8_t greater_b8 = __riscv_vmsltu_vv_u8m1_b8(max_u8m1, comparable_u8m1, vl);
        max_u8m1 = __riscv_vmerge_vvm_u8m1(max_u8m1, comparable_u8m1, greater_b8, vl);
        max_indices_u64m8 = __riscv_vmerge_vvm_u64m8(max_indices_u64m8, pos_u64m8, greater_b8, vl);
    }

    // Horizontal reduction + convert back
    vuint8m1_t init_max_u8m1 = __riscv_vmv_v_x_u8m1(0x3F, 1);
    nk_u8_t min_comparable = __riscv_vmv_x_s_u8m1_u8(__riscv_vredminu_vs_u8m1_u8m1(min_u8m1, init_max_u8m1, vlmax));
    vbool8_t min_match_b8 = __riscv_vmseq_vx_u8m1_b8(min_u8m1, min_comparable, vlmax);
    vuint64m8_t sentinel_u64m8 = __riscv_vmv_v_x_u64m8(NK_U64_MAX, vlmax);
    vuint64m8_t min_cands_u64m8 = __riscv_vmerge_vvm_u64m8(sentinel_u64m8, min_indices_u64m8, min_match_b8, vlmax);
    vuint64m1_t init_umax_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    *min_index_ptr = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(
        __riscv_vredminu_vs_u64m8_u64m1(min_cands_u64m8, init_umax_u64m1, vlmax));

    vuint8m1_t min_vec_u8m1 = __riscv_vmv_v_x_u8m1(min_comparable, 1);
    vuint8m1_t min_raw_u8m1 = nk_comparable_to_fp6m1_rvv_(min_vec_u8m1, 1);
    *min_value_ptr = (nk_e2m3_t)__riscv_vmv_x_s_u8m1_u8(min_raw_u8m1);

    vuint8m1_t init_min_u8m1 = __riscv_vmv_v_x_u8m1(0x00, 1);
    nk_u8_t max_comparable = __riscv_vmv_x_s_u8m1_u8(__riscv_vredmaxu_vs_u8m1_u8m1(max_u8m1, init_min_u8m1, vlmax));
    vbool8_t max_match_b8 = __riscv_vmseq_vx_u8m1_b8(max_u8m1, max_comparable, vlmax);
    vuint64m8_t max_cands_u64m8 = __riscv_vmerge_vvm_u64m8(sentinel_u64m8, max_indices_u64m8, max_match_b8, vlmax);
    *max_index_ptr = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(
        __riscv_vredminu_vs_u64m8_u64m1(max_cands_u64m8, init_umax_u64m1, vlmax));

    vuint8m1_t max_vec_u8m1 = __riscv_vmv_v_x_u8m1(max_comparable, 1);
    vuint8m1_t max_raw_u8m1 = nk_comparable_to_fp6m1_rvv_(max_vec_u8m1, 1);
    *max_value_ptr = (nk_e2m3_t)__riscv_vmv_x_s_u8m1_u8(max_raw_u8m1);
}

NK_INTERNAL void nk_reduce_minmax_e2m3_rvv_strided_(                    //
    nk_e2m3_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_e2m3_t *min_value_ptr, nk_size_t *min_index_ptr,                 //
    nk_e2m3_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e8m1();
    vuint8m1_t min_u8m1 = __riscv_vmv_v_x_u8m1(0x3F, vlmax);
    vuint8m1_t max_u8m1 = __riscv_vmv_v_x_u8m1(0x00, vlmax);
    vuint64m8_t min_indices_u64m8 = __riscv_vmv_v_x_u64m8(0, vlmax);
    vuint64m8_t max_indices_u64m8 = __riscv_vmv_v_x_u64m8(0, vlmax);
    unsigned char const *ptr = (unsigned char const *)data_ptr;

    nk_size_t offset = 0;
    for (nk_size_t vl; count > 0; count -= vl, offset += vl, ptr += vl * stride_bytes) {
        vl = __riscv_vsetvl_e8m1(count);
        vuint8m1_t raw_u8m1 = __riscv_vlse8_v_u8m1((uint8_t const *)ptr, (nk_ssize_t)stride_bytes, vl);

        vuint8m1_t comparable_u8m1 = nk_fp6m1_to_comparable_u8m1_rvv_(raw_u8m1, vl);
        vuint64m8_t pos_u64m8 = __riscv_vadd_vx_u64m8(__riscv_vid_v_u64m8(vl), (nk_u64_t)offset, vl);

        vbool8_t less_b8 = __riscv_vmsltu_vv_u8m1_b8(comparable_u8m1, min_u8m1, vl);
        min_u8m1 = __riscv_vmerge_vvm_u8m1(min_u8m1, comparable_u8m1, less_b8, vl);
        min_indices_u64m8 = __riscv_vmerge_vvm_u64m8(min_indices_u64m8, pos_u64m8, less_b8, vl);

        vbool8_t greater_b8 = __riscv_vmsltu_vv_u8m1_b8(max_u8m1, comparable_u8m1, vl);
        max_u8m1 = __riscv_vmerge_vvm_u8m1(max_u8m1, comparable_u8m1, greater_b8, vl);
        max_indices_u64m8 = __riscv_vmerge_vvm_u64m8(max_indices_u64m8, pos_u64m8, greater_b8, vl);
    }

    // Horizontal reduction (same as contiguous)
    vuint8m1_t init_max_u8m1 = __riscv_vmv_v_x_u8m1(0x3F, 1);
    nk_u8_t min_comparable = __riscv_vmv_x_s_u8m1_u8(__riscv_vredminu_vs_u8m1_u8m1(min_u8m1, init_max_u8m1, vlmax));
    vbool8_t min_match_b8 = __riscv_vmseq_vx_u8m1_b8(min_u8m1, min_comparable, vlmax);
    vuint64m8_t sentinel_u64m8 = __riscv_vmv_v_x_u64m8(NK_U64_MAX, vlmax);
    vuint64m8_t min_cands_u64m8 = __riscv_vmerge_vvm_u64m8(sentinel_u64m8, min_indices_u64m8, min_match_b8, vlmax);
    vuint64m1_t init_umax_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    *min_index_ptr = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(
        __riscv_vredminu_vs_u64m8_u64m1(min_cands_u64m8, init_umax_u64m1, vlmax));

    vuint8m1_t min_vec_u8m1 = __riscv_vmv_v_x_u8m1(min_comparable, 1);
    vuint8m1_t min_raw_u8m1 = nk_comparable_to_fp6m1_rvv_(min_vec_u8m1, 1);
    *min_value_ptr = (nk_e2m3_t)__riscv_vmv_x_s_u8m1_u8(min_raw_u8m1);

    vuint8m1_t init_min_u8m1 = __riscv_vmv_v_x_u8m1(0x00, 1);
    nk_u8_t max_comparable = __riscv_vmv_x_s_u8m1_u8(__riscv_vredmaxu_vs_u8m1_u8m1(max_u8m1, init_min_u8m1, vlmax));
    vbool8_t max_match_b8 = __riscv_vmseq_vx_u8m1_b8(max_u8m1, max_comparable, vlmax);
    vuint64m8_t max_cands_u64m8 = __riscv_vmerge_vvm_u64m8(sentinel_u64m8, max_indices_u64m8, max_match_b8, vlmax);
    *max_index_ptr = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(
        __riscv_vredminu_vs_u64m8_u64m1(max_cands_u64m8, init_umax_u64m1, vlmax));

    vuint8m1_t max_vec_u8m1 = __riscv_vmv_v_x_u8m1(max_comparable, 1);
    vuint8m1_t max_raw_u8m1 = nk_comparable_to_fp6m1_rvv_(max_vec_u8m1, 1);
    *max_value_ptr = (nk_e2m3_t)__riscv_vmv_x_s_u8m1_u8(max_raw_u8m1);
}

NK_PUBLIC void nk_reduce_minmax_e2m3_rvv(                               //
    nk_e2m3_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_e2m3_t *min_value_ptr, nk_size_t *min_index_ptr,                 //
    nk_e2m3_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_e2m3_t);
    int aligned = (stride_bytes % sizeof(nk_e2m3_t) == 0);

    if (count == 0)
        *min_value_ptr = NK_E2M3_MAX, *min_index_ptr = NK_SIZE_MAX, *max_value_ptr = NK_E2M3_MIN,
        *max_index_ptr = NK_SIZE_MAX;
    else if (!aligned)
        nk_reduce_minmax_e2m3_serial(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                     max_index_ptr);
    else if (stride_elements == 1)
        nk_reduce_minmax_e2m3_rvv_contiguous_(data_ptr, count, min_value_ptr, min_index_ptr, max_value_ptr,
                                              max_index_ptr);
    else
        nk_reduce_minmax_e2m3_rvv_strided_(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                           max_index_ptr);
}

NK_INTERNAL void nk_reduce_moments_e3m2_rvv_contiguous_( //
    nk_e3m2_t const *data_ptr, nk_size_t count,          //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e32m4();
    vfloat32m4_t sum_f32m4 = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);
    vfloat32m4_t sumsq_f32m4 = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);

    for (nk_size_t vl; count > 0; count -= vl, data_ptr += vl) {
        vl = __riscv_vsetvl_e8m1(count);
        vuint8m1_t data_u8m1 = __riscv_vle8_v_u8m1((uint8_t const *)data_ptr, vl);

        // Convert e3m2 → f32 (m1 → m4)
        vfloat32m4_t data_f32m4 = nk_e3m2m1_to_f32m4_rvv_(data_u8m1, vl);

        // Accumulate at f32 precision
        sum_f32m4 = __riscv_vfadd_vv_f32m4(sum_f32m4, data_f32m4, vl);
        sumsq_f32m4 = __riscv_vfmacc_vv_f32m4(sumsq_f32m4, data_f32m4, data_f32m4, vl);
    }

    // Horizontal reduction
    vfloat32m1_t zero_f32m1 = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    *sum_ptr = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(sum_f32m4, zero_f32m1, vlmax)),
    *sumsq_ptr = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(sumsq_f32m4, zero_f32m1, vlmax));
}

NK_INTERNAL void nk_reduce_moments_e3m2_rvv_strided_(                   //
    nk_e3m2_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e32m4();
    vfloat32m4_t sum_f32m4 = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);
    vfloat32m4_t sumsq_f32m4 = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);
    unsigned char const *ptr = (unsigned char const *)data_ptr;

    for (nk_size_t vl; count > 0; count -= vl, ptr += vl * stride_bytes) {
        vl = __riscv_vsetvl_e8m1(count);
        vuint8m1_t data_u8m1 = __riscv_vlse8_v_u8m1((uint8_t const *)ptr, (nk_ssize_t)stride_bytes, vl);

        // Convert e3m2 → f32 (m1 → m4)
        vfloat32m4_t data_f32m4 = nk_e3m2m1_to_f32m4_rvv_(data_u8m1, vl);

        // Accumulate at f32 precision
        sum_f32m4 = __riscv_vfadd_vv_f32m4(sum_f32m4, data_f32m4, vl);
        sumsq_f32m4 = __riscv_vfmacc_vv_f32m4(sumsq_f32m4, data_f32m4, data_f32m4, vl);
    }

    // Horizontal reduction
    vfloat32m1_t zero_f32m1 = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    *sum_ptr = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(sum_f32m4, zero_f32m1, vlmax)),
    *sumsq_ptr = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(sumsq_f32m4, zero_f32m1, vlmax));
}

NK_PUBLIC void nk_reduce_moments_e3m2_rvv(                              //
    nk_e3m2_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_e3m2_t);
    int aligned = (stride_bytes % sizeof(nk_e3m2_t) == 0);

    if (count == 0) *sum_ptr = 0.0f, *sumsq_ptr = 0.0f;
    else if (!aligned) nk_reduce_moments_e3m2_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
    else if (stride_elements == 1) nk_reduce_moments_e3m2_rvv_contiguous_(data_ptr, count, sum_ptr, sumsq_ptr);
    else nk_reduce_moments_e3m2_rvv_strided_(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
}

NK_INTERNAL void nk_reduce_minmax_e3m2_rvv_contiguous_( //
    nk_e3m2_t const *data_ptr, nk_size_t count,         //
    nk_e3m2_t *min_value_ptr, nk_size_t *min_index_ptr, //
    nk_e3m2_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e8m1();
    vuint8m1_t min_u8m1 = __riscv_vmv_v_x_u8m1(0x3F, vlmax);
    vuint8m1_t max_u8m1 = __riscv_vmv_v_x_u8m1(0x00, vlmax);
    vuint64m8_t min_indices_u64m8 = __riscv_vmv_v_x_u64m8(0, vlmax);
    vuint64m8_t max_indices_u64m8 = __riscv_vmv_v_x_u64m8(0, vlmax);

    nk_size_t offset = 0;
    for (nk_size_t vl; count > 0; count -= vl, offset += vl, data_ptr += vl) {
        vl = __riscv_vsetvl_e8m1(count);
        vuint8m1_t raw_u8m1 = __riscv_vle8_v_u8m1((uint8_t const *)data_ptr, vl);

        vuint8m1_t comparable_u8m1 = nk_fp6m1_to_comparable_u8m1_rvv_(raw_u8m1, vl);
        vuint64m8_t pos_u64m8 = __riscv_vadd_vx_u64m8(__riscv_vid_v_u64m8(vl), (nk_u64_t)offset, vl);

        vbool8_t less_b8 = __riscv_vmsltu_vv_u8m1_b8(comparable_u8m1, min_u8m1, vl);
        min_u8m1 = __riscv_vmerge_vvm_u8m1(min_u8m1, comparable_u8m1, less_b8, vl);
        min_indices_u64m8 = __riscv_vmerge_vvm_u64m8(min_indices_u64m8, pos_u64m8, less_b8, vl);

        vbool8_t greater_b8 = __riscv_vmsltu_vv_u8m1_b8(max_u8m1, comparable_u8m1, vl);
        max_u8m1 = __riscv_vmerge_vvm_u8m1(max_u8m1, comparable_u8m1, greater_b8, vl);
        max_indices_u64m8 = __riscv_vmerge_vvm_u64m8(max_indices_u64m8, pos_u64m8, greater_b8, vl);
    }

    // Horizontal reduction + convert back
    vuint8m1_t init_max_u8m1 = __riscv_vmv_v_x_u8m1(0x3F, 1);
    nk_u8_t min_comparable = __riscv_vmv_x_s_u8m1_u8(__riscv_vredminu_vs_u8m1_u8m1(min_u8m1, init_max_u8m1, vlmax));
    vbool8_t min_match_b8 = __riscv_vmseq_vx_u8m1_b8(min_u8m1, min_comparable, vlmax);
    vuint64m8_t sentinel_u64m8 = __riscv_vmv_v_x_u64m8(NK_U64_MAX, vlmax);
    vuint64m8_t min_cands_u64m8 = __riscv_vmerge_vvm_u64m8(sentinel_u64m8, min_indices_u64m8, min_match_b8, vlmax);
    vuint64m1_t init_umax_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    *min_index_ptr = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(
        __riscv_vredminu_vs_u64m8_u64m1(min_cands_u64m8, init_umax_u64m1, vlmax));

    vuint8m1_t min_vec_u8m1 = __riscv_vmv_v_x_u8m1(min_comparable, 1);
    vuint8m1_t min_raw_u8m1 = nk_comparable_to_fp6m1_rvv_(min_vec_u8m1, 1);
    *min_value_ptr = (nk_e3m2_t)__riscv_vmv_x_s_u8m1_u8(min_raw_u8m1);

    vuint8m1_t init_min_u8m1 = __riscv_vmv_v_x_u8m1(0x00, 1);
    nk_u8_t max_comparable = __riscv_vmv_x_s_u8m1_u8(__riscv_vredmaxu_vs_u8m1_u8m1(max_u8m1, init_min_u8m1, vlmax));
    vbool8_t max_match_b8 = __riscv_vmseq_vx_u8m1_b8(max_u8m1, max_comparable, vlmax);
    vuint64m8_t max_cands_u64m8 = __riscv_vmerge_vvm_u64m8(sentinel_u64m8, max_indices_u64m8, max_match_b8, vlmax);
    *max_index_ptr = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(
        __riscv_vredminu_vs_u64m8_u64m1(max_cands_u64m8, init_umax_u64m1, vlmax));

    vuint8m1_t max_vec_u8m1 = __riscv_vmv_v_x_u8m1(max_comparable, 1);
    vuint8m1_t max_raw_u8m1 = nk_comparable_to_fp6m1_rvv_(max_vec_u8m1, 1);
    *max_value_ptr = (nk_e3m2_t)__riscv_vmv_x_s_u8m1_u8(max_raw_u8m1);
}

NK_INTERNAL void nk_reduce_minmax_e3m2_rvv_strided_(                    //
    nk_e3m2_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_e3m2_t *min_value_ptr, nk_size_t *min_index_ptr,                 //
    nk_e3m2_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e8m1();
    vuint8m1_t min_u8m1 = __riscv_vmv_v_x_u8m1(0x3F, vlmax);
    vuint8m1_t max_u8m1 = __riscv_vmv_v_x_u8m1(0x00, vlmax);
    vuint64m8_t min_indices_u64m8 = __riscv_vmv_v_x_u64m8(0, vlmax);
    vuint64m8_t max_indices_u64m8 = __riscv_vmv_v_x_u64m8(0, vlmax);
    unsigned char const *ptr = (unsigned char const *)data_ptr;

    nk_size_t offset = 0;
    for (nk_size_t vl; count > 0; count -= vl, offset += vl, ptr += vl * stride_bytes) {
        vl = __riscv_vsetvl_e8m1(count);
        vuint8m1_t raw_u8m1 = __riscv_vlse8_v_u8m1((uint8_t const *)ptr, (nk_ssize_t)stride_bytes, vl);

        vuint8m1_t comparable_u8m1 = nk_fp6m1_to_comparable_u8m1_rvv_(raw_u8m1, vl);
        vuint64m8_t pos_u64m8 = __riscv_vadd_vx_u64m8(__riscv_vid_v_u64m8(vl), (nk_u64_t)offset, vl);

        vbool8_t less_b8 = __riscv_vmsltu_vv_u8m1_b8(comparable_u8m1, min_u8m1, vl);
        min_u8m1 = __riscv_vmerge_vvm_u8m1(min_u8m1, comparable_u8m1, less_b8, vl);
        min_indices_u64m8 = __riscv_vmerge_vvm_u64m8(min_indices_u64m8, pos_u64m8, less_b8, vl);

        vbool8_t greater_b8 = __riscv_vmsltu_vv_u8m1_b8(max_u8m1, comparable_u8m1, vl);
        max_u8m1 = __riscv_vmerge_vvm_u8m1(max_u8m1, comparable_u8m1, greater_b8, vl);
        max_indices_u64m8 = __riscv_vmerge_vvm_u64m8(max_indices_u64m8, pos_u64m8, greater_b8, vl);
    }

    // Horizontal reduction (same as contiguous)
    vuint8m1_t init_max_u8m1 = __riscv_vmv_v_x_u8m1(0x3F, 1);
    nk_u8_t min_comparable = __riscv_vmv_x_s_u8m1_u8(__riscv_vredminu_vs_u8m1_u8m1(min_u8m1, init_max_u8m1, vlmax));
    vbool8_t min_match_b8 = __riscv_vmseq_vx_u8m1_b8(min_u8m1, min_comparable, vlmax);
    vuint64m8_t sentinel_u64m8 = __riscv_vmv_v_x_u64m8(NK_U64_MAX, vlmax);
    vuint64m8_t min_cands_u64m8 = __riscv_vmerge_vvm_u64m8(sentinel_u64m8, min_indices_u64m8, min_match_b8, vlmax);
    vuint64m1_t init_umax_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    *min_index_ptr = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(
        __riscv_vredminu_vs_u64m8_u64m1(min_cands_u64m8, init_umax_u64m1, vlmax));

    vuint8m1_t min_vec_u8m1 = __riscv_vmv_v_x_u8m1(min_comparable, 1);
    vuint8m1_t min_raw_u8m1 = nk_comparable_to_fp6m1_rvv_(min_vec_u8m1, 1);
    *min_value_ptr = (nk_e3m2_t)__riscv_vmv_x_s_u8m1_u8(min_raw_u8m1);

    vuint8m1_t init_min_u8m1 = __riscv_vmv_v_x_u8m1(0x00, 1);
    nk_u8_t max_comparable = __riscv_vmv_x_s_u8m1_u8(__riscv_vredmaxu_vs_u8m1_u8m1(max_u8m1, init_min_u8m1, vlmax));
    vbool8_t max_match_b8 = __riscv_vmseq_vx_u8m1_b8(max_u8m1, max_comparable, vlmax);
    vuint64m8_t max_cands_u64m8 = __riscv_vmerge_vvm_u64m8(sentinel_u64m8, max_indices_u64m8, max_match_b8, vlmax);
    *max_index_ptr = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(
        __riscv_vredminu_vs_u64m8_u64m1(max_cands_u64m8, init_umax_u64m1, vlmax));

    vuint8m1_t max_vec_u8m1 = __riscv_vmv_v_x_u8m1(max_comparable, 1);
    vuint8m1_t max_raw_u8m1 = nk_comparable_to_fp6m1_rvv_(max_vec_u8m1, 1);
    *max_value_ptr = (nk_e3m2_t)__riscv_vmv_x_s_u8m1_u8(max_raw_u8m1);
}

NK_PUBLIC void nk_reduce_minmax_e3m2_rvv(                               //
    nk_e3m2_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_e3m2_t *min_value_ptr, nk_size_t *min_index_ptr,                 //
    nk_e3m2_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_e3m2_t);
    int aligned = (stride_bytes % sizeof(nk_e3m2_t) == 0);

    if (count == 0)
        *min_value_ptr = NK_E3M2_MAX, *min_index_ptr = NK_SIZE_MAX, *max_value_ptr = NK_E3M2_MIN,
        *max_index_ptr = NK_SIZE_MAX;
    else if (!aligned)
        nk_reduce_minmax_e3m2_serial(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                     max_index_ptr);
    else if (stride_elements == 1)
        nk_reduce_minmax_e3m2_rvv_contiguous_(data_ptr, count, min_value_ptr, min_index_ptr, max_value_ptr,
                                              max_index_ptr);
    else
        nk_reduce_minmax_e3m2_rvv_strided_(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                           max_index_ptr);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_RVV
#endif // NK_TARGET_RISCV_
#endif // NK_REDUCE_RVV_H

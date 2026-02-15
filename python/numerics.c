/**
 *  @brief Numeric operations for NumKong Python bindings.
 *  @file python/numerics.c
 *
 *  This file implements numeric API functions including elementwise operations,
 *  reductions, trigonometry, and mesh alignment (Kabsch/Umeyama).
 */
#include "numerics.h"
#include "tensor.h"

#pragma region Shared Implementations

int impl_elementwise_add(char *a, char *b, char *out, size_t n, nk_dtype_t dtype, Py_ssize_t stride_a,
                         Py_ssize_t stride_b, Py_ssize_t stride_out) {
    // Fast path: contiguous data
    size_t item_size = bytes_per_dtype(dtype);
    if (stride_a == (Py_ssize_t)item_size && stride_b == (Py_ssize_t)item_size && stride_out == (Py_ssize_t)item_size) {
        switch (dtype) {
        case nk_f64_k: nk_each_sum_f64((nk_f64_t *)a, (nk_f64_t *)b, n, (nk_f64_t *)out); return 0;
        case nk_f32_k: nk_each_sum_f32((nk_f32_t *)a, (nk_f32_t *)b, n, (nk_f32_t *)out); return 0;
        case nk_i8_k: nk_each_sum_i8((nk_i8_t *)a, (nk_i8_t *)b, n, (nk_i8_t *)out); return 0;
        case nk_i32_k: nk_each_sum_i32((nk_i32_t *)a, (nk_i32_t *)b, n, (nk_i32_t *)out); return 0;
        default: break;
        }
    }
    // Strided path
    switch (dtype) {
    case nk_f64_k:
        for (size_t i = 0; i < n; i++)
            *(nk_f64_t *)(out + i * stride_out) = *(nk_f64_t *)(a + i * stride_a) + *(nk_f64_t *)(b + i * stride_b);
        return 0;
    case nk_f32_k:
        for (size_t i = 0; i < n; i++)
            *(nk_f32_t *)(out + i * stride_out) = *(nk_f32_t *)(a + i * stride_a) + *(nk_f32_t *)(b + i * stride_b);
        return 0;
    case nk_i8_k:
        for (size_t i = 0; i < n; i++)
            *(nk_i8_t *)(out + i * stride_out) = *(nk_i8_t *)(a + i * stride_a) + *(nk_i8_t *)(b + i * stride_b);
        return 0;
    case nk_i32_k:
        for (size_t i = 0; i < n; i++)
            *(nk_i32_t *)(out + i * stride_out) = *(nk_i32_t *)(a + i * stride_a) + *(nk_i32_t *)(b + i * stride_b);
        return 0;
    default: return -1;
    }
}

int impl_elementwise_mul(char *a, char *b, char *out, size_t n, nk_dtype_t dtype, Py_ssize_t stride_a,
                         Py_ssize_t stride_b, Py_ssize_t stride_out) {
    size_t item_size = bytes_per_dtype(dtype);
    // Contiguous fast path using FMA with alpha=1, beta=0
    if (stride_a == (Py_ssize_t)item_size && stride_b == (Py_ssize_t)item_size && stride_out == (Py_ssize_t)item_size) {
        switch (dtype) {
        case nk_f64_k: {
            nk_f64_t alpha = 1, beta = 0;
            nk_each_fma_f64((nk_f64_t *)a, (nk_f64_t *)b, (nk_f64_t *)out, n, &alpha, &beta, (nk_f64_t *)out);
            return 0;
        }
        case nk_f32_k: {
            nk_f32_t alpha = 1, beta = 0;
            nk_each_fma_f32((nk_f32_t *)a, (nk_f32_t *)b, (nk_f32_t *)out, n, &alpha, &beta, (nk_f32_t *)out);
            return 0;
        }
        default: break;
        }
    }
    // Strided path
    switch (dtype) {
    case nk_f64_k:
        for (size_t i = 0; i < n; i++)
            *(nk_f64_t *)(out + i * stride_out) = *(nk_f64_t *)(a + i * stride_a) * *(nk_f64_t *)(b + i * stride_b);
        return 0;
    case nk_f32_k:
        for (size_t i = 0; i < n; i++)
            *(nk_f32_t *)(out + i * stride_out) = *(nk_f32_t *)(a + i * stride_a) * *(nk_f32_t *)(b + i * stride_b);
        return 0;
    default: return -1;
    }
}

int impl_elementwise_wsum(char *a, char *b, char *out, size_t n, nk_dtype_t dtype, double alpha, double beta,
                          Py_ssize_t stride_a, Py_ssize_t stride_b, Py_ssize_t stride_out) {
    size_t item_size = bytes_per_dtype(dtype);
    if (stride_a == (Py_ssize_t)item_size && stride_b == (Py_ssize_t)item_size && stride_out == (Py_ssize_t)item_size) {
        switch (dtype) {
        case nk_f64_k: {
            nk_f64_t alpha_f64 = alpha, beta_f64 = beta;
            nk_each_blend_f64((nk_f64_t *)a, (nk_f64_t *)b, n, &alpha_f64, &beta_f64, (nk_f64_t *)out);
            return 0;
        }
        case nk_f32_k: {
            nk_f32_t alpha_f32 = (nk_f32_t)alpha, beta_f32 = (nk_f32_t)beta;
            nk_each_blend_f32((nk_f32_t *)a, (nk_f32_t *)b, n, &alpha_f32, &beta_f32, (nk_f32_t *)out);
            return 0;
        }
        default: break;
        }
    }
    // Strided path
    switch (dtype) {
    case nk_f64_k:
        for (size_t i = 0; i < n; i++)
            *(nk_f64_t *)(out + i * stride_out) = alpha * *(nk_f64_t *)(a + i * stride_a) +
                                                  beta * *(nk_f64_t *)(b + i * stride_b);
        return 0;
    case nk_f32_k:
        for (size_t i = 0; i < n; i++)
            *(nk_f32_t *)(out + i * stride_out) = (nk_f32_t)alpha * *(nk_f32_t *)(a + i * stride_a) +
                                                  (nk_f32_t)beta * *(nk_f32_t *)(b + i * stride_b);
        return 0;
    default: return -1;
    }
}

int impl_elementwise_scale(char *a, char *out, size_t n, nk_dtype_t dtype, double alpha, double beta,
                           Py_ssize_t stride_a, Py_ssize_t stride_out) {
    size_t item_size = bytes_per_dtype(dtype);
    if (stride_a == (Py_ssize_t)item_size && stride_out == (Py_ssize_t)item_size) {
        switch (dtype) {
        case nk_f64_k: {
            nk_f64_t alpha_f64 = alpha, beta_f64 = beta;
            nk_each_scale_f64((nk_f64_t *)a, n, &alpha_f64, &beta_f64, (nk_f64_t *)out);
            return 0;
        }
        case nk_f32_k: {
            nk_f32_t alpha_f32 = (nk_f32_t)alpha, beta_f32 = (nk_f32_t)beta;
            nk_each_scale_f32((nk_f32_t *)a, n, &alpha_f32, &beta_f32, (nk_f32_t *)out);
            return 0;
        }
        default: break;
        }
    }
    // Strided path
    switch (dtype) {
    case nk_f64_k:
        for (size_t i = 0; i < n; i++)
            *(nk_f64_t *)(out + i * stride_out) = alpha * *(nk_f64_t *)(a + i * stride_a) + beta;
        return 0;
    case nk_f32_k:
        for (size_t i = 0; i < n; i++)
            *(nk_f32_t *)(out + i * stride_out) = (nk_f32_t)alpha * *(nk_f32_t *)(a + i * stride_a) + (nk_f32_t)beta;
        return 0;
    default: return -1;
    }
}

#pragma endregion // Shared Implementations

#pragma region Reductions

/**  @brief Recursive helper for impl_reduce_moments.  */
static void reduce_moments_recursive(TensorView const *view, size_t dim, char *ptr, double *sum_f, int64_t *sum_i,
                                     double *sumsq_f, int64_t *sumsq_i) {
    if (dim == view->rank - 1) {
        // Base case: innermost dimension
        size_t const n = (size_t)view->shape[dim];
        Py_ssize_t const stride = view->strides[dim];
        switch (view->dtype) {
        case nk_f64_k:
            for (size_t i = 0; i < n; i++) {
                nk_f64_t v = *(nk_f64_t *)(ptr + i * stride);
                *sum_f += v;
                *sumsq_f += v * v;
            }
            break;
        case nk_f32_k:
            for (size_t i = 0; i < n; i++) {
                nk_f32_t v = *(nk_f32_t *)(ptr + i * stride);
                *sum_f += v;
                *sumsq_f += (double)v * v;
            }
            break;
        case nk_f16_k: {
            nk_f32_t tmp;
            for (size_t i = 0; i < n; i++) {
                nk_f16_to_f32((nk_f16_t *)(ptr + i * stride), &tmp);
                *sum_f += tmp;
                *sumsq_f += (double)tmp * tmp;
            }
        } break;
        case nk_bf16_k: {
            nk_f32_t tmp;
            for (size_t i = 0; i < n; i++) {
                nk_bf16_to_f32((nk_bf16_t *)(ptr + i * stride), &tmp);
                *sum_f += tmp;
                *sumsq_f += (double)tmp * tmp;
            }
        } break;
        case nk_i8_k:
            for (size_t i = 0; i < n; i++) {
                nk_i8_t v = *(nk_i8_t *)(ptr + i * stride);
                *sum_i += v;
                *sumsq_i += (int64_t)v * v;
            }
            break;
        case nk_u8_k:
            for (size_t i = 0; i < n; i++) {
                nk_u8_t v = *(nk_u8_t *)(ptr + i * stride);
                *sum_i += v;
                *sumsq_i += (int64_t)v * v;
            }
            break;
        case nk_i16_k:
            for (size_t i = 0; i < n; i++) {
                nk_i16_t v = *(nk_i16_t *)(ptr + i * stride);
                *sum_i += v;
                *sumsq_i += (int64_t)v * v;
            }
            break;
        case nk_u16_k:
            for (size_t i = 0; i < n; i++) {
                nk_u16_t v = *(nk_u16_t *)(ptr + i * stride);
                *sum_i += v;
                *sumsq_i += (int64_t)v * v;
            }
            break;
        case nk_i32_k:
            for (size_t i = 0; i < n; i++) {
                nk_i32_t v = *(nk_i32_t *)(ptr + i * stride);
                *sum_i += v;
                *sumsq_i += (int64_t)v * v;
            }
            break;
        case nk_u32_k:
            for (size_t i = 0; i < n; i++) {
                nk_u32_t v = *(nk_u32_t *)(ptr + i * stride);
                *sum_i += v;
                *sumsq_i += (int64_t)v * v;
            }
            break;
        case nk_i64_k:
            for (size_t i = 0; i < n; i++) {
                nk_i64_t v = *(nk_i64_t *)(ptr + i * stride);
                *sum_i += v;
                *sumsq_i += v * v;
            }
            break;
        case nk_u64_k:
            for (size_t i = 0; i < n; i++) {
                nk_u64_t v = *(nk_u64_t *)(ptr + i * stride);
                *sum_i += (int64_t)v;
                *sumsq_i += (int64_t)(v * v);
            }
            break;
        default: break;
        }
    }
    else {
        // Recursive case: iterate outer dimension
        size_t const n = (size_t)view->shape[dim];
        Py_ssize_t const stride = view->strides[dim];
        for (size_t i = 0; i < n; i++) {
            reduce_moments_recursive(view, dim + 1, ptr + i * stride, sum_f, sum_i, sumsq_f, sumsq_i);
        }
    }
}

int impl_reduce_moments(TensorView const *view, double *sum_f, int64_t *sum_i, double *sumsq_f, int64_t *sumsq_i) {
    if (sum_f) *sum_f = 0;
    if (sum_i) *sum_i = 0;
    if (sumsq_f) *sumsq_f = 0;
    if (sumsq_i) *sumsq_i = 0;
    if (view->rank == 0) {
        // Scalar case
        switch (view->dtype) {
        case nk_f64_k: {
            double v = *(nk_f64_t *)view->data;
            *sum_f = v;
            *sumsq_f = v * v;
            return 0;
        }
        case nk_f32_k: {
            double v = *(nk_f32_t *)view->data;
            *sum_f = v;
            *sumsq_f = v * v;
            return 0;
        }
        case nk_i64_k: {
            int64_t v = *(nk_i64_t *)view->data;
            *sum_i = v;
            *sumsq_i = v * v;
            return 0;
        }
        case nk_i32_k: {
            int64_t v = *(nk_i32_t *)view->data;
            *sum_i = v;
            *sumsq_i = v * v;
            return 0;
        }
        default: return -1;
        }
    }
    reduce_moments_recursive(view, 0, view->data, sum_f, sum_i, sumsq_f, sumsq_i);
    return 0;
}

/// Macro for simultaneous min/max reduction over all supported types.
/// Updates both min and max trackers in a single pass.
#define reduce_minmax_(min_f_, min_i_, min_idx_, max_f_, max_i_, max_idx_)                \
    do {                                                                                  \
        size_t const n = (size_t)view->shape[dim];                                        \
        Py_ssize_t const stride = view->strides[dim];                                     \
        switch (view->dtype) {                                                            \
        case nk_f64_k:                                                                    \
            for (size_t i = 0; i < n; i++) {                                              \
                nk_f64_t v = *(nk_f64_t *)(ptr + i * stride);                             \
                if (v < *min_f_) *min_f_ = v, *min_idx_ = base_idx + i;                   \
                if (v > *max_f_) *max_f_ = v, *max_idx_ = base_idx + i;                   \
            }                                                                             \
            break;                                                                        \
        case nk_f32_k:                                                                    \
            for (size_t i = 0; i < n; i++) {                                              \
                nk_f32_t v = *(nk_f32_t *)(ptr + i * stride);                             \
                if (v < *min_f_) *min_f_ = v, *min_idx_ = base_idx + i;                   \
                if (v > *max_f_) *max_f_ = v, *max_idx_ = base_idx + i;                   \
            }                                                                             \
            break;                                                                        \
        case nk_f16_k:                                                                    \
            for (size_t i = 0; i < n; i++) {                                              \
                nk_f32_t v;                                                               \
                nk_f16_to_f32((nk_f16_t *)(ptr + i * stride), &v);                        \
                if (v < *min_f_) *min_f_ = v, *min_idx_ = base_idx + i;                   \
                if (v > *max_f_) *max_f_ = v, *max_idx_ = base_idx + i;                   \
            }                                                                             \
            break;                                                                        \
        case nk_bf16_k:                                                                   \
            for (size_t i = 0; i < n; i++) {                                              \
                nk_f32_t v;                                                               \
                nk_bf16_to_f32((nk_bf16_t *)(ptr + i * stride), &v);                      \
                if (v < *min_f_) *min_f_ = v, *min_idx_ = base_idx + i;                   \
                if (v > *max_f_) *max_f_ = v, *max_idx_ = base_idx + i;                   \
            }                                                                             \
            break;                                                                        \
        case nk_i64_k:                                                                    \
            for (size_t i = 0; i < n; i++) {                                              \
                nk_i64_t v = *(nk_i64_t *)(ptr + i * stride);                             \
                if (v < *min_i_) *min_i_ = v, *min_idx_ = base_idx + i;                   \
                if (v > *max_i_) *max_i_ = v, *max_idx_ = base_idx + i;                   \
            }                                                                             \
            break;                                                                        \
        case nk_i32_k:                                                                    \
            for (size_t i = 0; i < n; i++) {                                              \
                nk_i32_t v = *(nk_i32_t *)(ptr + i * stride);                             \
                if (v < *min_i_) *min_i_ = v, *min_idx_ = base_idx + i;                   \
                if (v > *max_i_) *max_i_ = v, *max_idx_ = base_idx + i;                   \
            }                                                                             \
            break;                                                                        \
        case nk_i16_k:                                                                    \
            for (size_t i = 0; i < n; i++) {                                              \
                nk_i16_t v = *(nk_i16_t *)(ptr + i * stride);                             \
                if (v < *min_i_) *min_i_ = v, *min_idx_ = base_idx + i;                   \
                if (v > *max_i_) *max_i_ = v, *max_idx_ = base_idx + i;                   \
            }                                                                             \
            break;                                                                        \
        case nk_i8_k:                                                                     \
            for (size_t i = 0; i < n; i++) {                                              \
                nk_i8_t v = *(nk_i8_t *)(ptr + i * stride);                               \
                if (v < *min_i_) *min_i_ = v, *min_idx_ = base_idx + i;                   \
                if (v > *max_i_) *max_i_ = v, *max_idx_ = base_idx + i;                   \
            }                                                                             \
            break;                                                                        \
        case nk_u64_k:                                                                    \
            for (size_t i = 0; i < n; i++) {                                              \
                nk_u64_t v = *(nk_u64_t *)(ptr + i * stride);                             \
                if ((int64_t)v < *min_i_) *min_i_ = (int64_t)v, *min_idx_ = base_idx + i; \
                if ((int64_t)v > *max_i_) *max_i_ = (int64_t)v, *max_idx_ = base_idx + i; \
            }                                                                             \
            break;                                                                        \
        case nk_u32_k:                                                                    \
            for (size_t i = 0; i < n; i++) {                                              \
                nk_u32_t v = *(nk_u32_t *)(ptr + i * stride);                             \
                if (v < (nk_u32_t) * min_i_) *min_i_ = v, *min_idx_ = base_idx + i;       \
                if (v > (nk_u32_t) * max_i_) *max_i_ = v, *max_idx_ = base_idx + i;       \
            }                                                                             \
            break;                                                                        \
        case nk_u16_k:                                                                    \
            for (size_t i = 0; i < n; i++) {                                              \
                nk_u16_t v = *(nk_u16_t *)(ptr + i * stride);                             \
                if (v < (nk_u16_t) * min_i_) *min_i_ = v, *min_idx_ = base_idx + i;       \
                if (v > (nk_u16_t) * max_i_) *max_i_ = v, *max_idx_ = base_idx + i;       \
            }                                                                             \
            break;                                                                        \
        case nk_u8_k:                                                                     \
            for (size_t i = 0; i < n; i++) {                                              \
                nk_u8_t v = *(nk_u8_t *)(ptr + i * stride);                               \
                if (v < (nk_u8_t) * min_i_) *min_i_ = v, *min_idx_ = base_idx + i;        \
                if (v > (nk_u8_t) * max_i_) *max_i_ = v, *max_idx_ = base_idx + i;        \
            }                                                                             \
            break;                                                                        \
        default: break;                                                                   \
        }                                                                                 \
    } while (0)

/**  @brief Recursive helper for impl_reduce_minmax.  */
static void reduce_minmax_recursive_(TensorView const *view, size_t dim, char *ptr, size_t base_idx, double *min_f,
                                     int64_t *min_i, size_t *min_idx, double *max_f, int64_t *max_i, size_t *max_idx) {
    if (dim == view->rank - 1) { reduce_minmax_(min_f, min_i, min_idx, max_f, max_i, max_idx); }
    else {
        size_t const n = (size_t)view->shape[dim];
        Py_ssize_t const stride = view->strides[dim];
        size_t inner_size = 1;
        for (size_t d = dim + 1; d < view->rank; d++) inner_size *= (size_t)view->shape[d];
        for (size_t i = 0; i < n; i++)
            reduce_minmax_recursive_(view, dim + 1, ptr + i * stride, base_idx + i * inner_size, min_f, min_i, min_idx,
                                     max_f, max_i, max_idx);
    }
}

#undef reduce_minmax_

int impl_reduce_minmax(TensorView const *view, double *min_f, int64_t *min_i, size_t *min_index, double *max_f,
                       int64_t *max_i, size_t *max_index) {
    *min_index = 0;
    *max_index = 0;
    if (min_f) *min_f = NK_F64_MAX;
    if (min_i) *min_i = NK_I64_MAX;
    if (max_f) *max_f = NK_F64_MIN;
    if (max_i) *max_i = NK_I64_MIN;
    if (view->rank == 0) {
        switch (view->dtype) {
        case nk_f64_k: {
            double v = *(nk_f64_t *)view->data;
            *min_f = v;
            *max_f = v;
            return 0;
        }
        case nk_f32_k: {
            double v = *(nk_f32_t *)view->data;
            *min_f = v;
            *max_f = v;
            return 0;
        }
        case nk_i64_k: {
            int64_t v = *(nk_i64_t *)view->data;
            *min_i = v;
            *max_i = v;
            return 0;
        }
        case nk_i32_k: {
            int64_t v = *(nk_i32_t *)view->data;
            *min_i = v;
            *max_i = v;
            return 0;
        }
        case nk_i16_k: {
            int64_t v = *(nk_i16_t *)view->data;
            *min_i = v;
            *max_i = v;
            return 0;
        }
        case nk_i8_k: {
            int64_t v = *(nk_i8_t *)view->data;
            *min_i = v;
            *max_i = v;
            return 0;
        }
        case nk_u64_k: {
            int64_t v = (int64_t)*(nk_u64_t *)view->data;
            *min_i = v;
            *max_i = v;
            return 0;
        }
        case nk_u32_k: {
            int64_t v = *(nk_u32_t *)view->data;
            *min_i = v;
            *max_i = v;
            return 0;
        }
        case nk_u16_k: {
            int64_t v = *(nk_u16_t *)view->data;
            *min_i = v;
            *max_i = v;
            return 0;
        }
        case nk_u8_k: {
            int64_t v = *(nk_u8_t *)view->data;
            *min_i = v;
            *max_i = v;
            return 0;
        }
        default: return -1;
        }
    }
    reduce_minmax_recursive_(view, 0, view->data, 0, min_f, min_i, min_index, max_f, max_i, max_index);
    return 0;
}

#pragma endregion // Reductions

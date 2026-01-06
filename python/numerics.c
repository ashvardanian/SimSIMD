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
        case nk_f64_k: nk_sum_f64((nk_f64_t *)a, (nk_f64_t *)b, n, (nk_f64_t *)out); return 0;
        case nk_f32_k: nk_sum_f32((nk_f32_t *)a, (nk_f32_t *)b, n, (nk_f32_t *)out); return 0;
        case nk_i8_k: nk_sum_i8((nk_i8_t *)a, (nk_i8_t *)b, n, (nk_i8_t *)out); return 0;
        case nk_i32_k: nk_sum_i32((nk_i32_t *)a, (nk_i32_t *)b, n, (nk_i32_t *)out); return 0;
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
            nk_fma_f64((nk_f64_t *)a, (nk_f64_t *)b, (nk_f64_t *)out, n, &alpha, &beta, (nk_f64_t *)out);
            return 0;
        }
        case nk_f32_k: {
            nk_f32_t alpha = 1, beta = 0;
            nk_fma_f32((nk_f32_t *)a, (nk_f32_t *)b, (nk_f32_t *)out, n, &alpha, &beta, (nk_f32_t *)out);
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
            nk_wsum_f64((nk_f64_t *)a, (nk_f64_t *)b, n, &alpha_f64, &beta_f64, (nk_f64_t *)out);
            return 0;
        }
        case nk_f32_k: {
            nk_f32_t alpha_f32 = (nk_f32_t)alpha, beta_f32 = (nk_f32_t)beta;
            nk_wsum_f32((nk_f32_t *)a, (nk_f32_t *)b, n, &alpha_f32, &beta_f32, (nk_f32_t *)out);
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
            nk_scale_f64((nk_f64_t *)a, n, &alpha_f64, &beta_f64, (nk_f64_t *)out);
            return 0;
        }
        case nk_f32_k: {
            nk_f32_t alpha_f32 = (nk_f32_t)alpha, beta_f32 = (nk_f32_t)beta;
            nk_scale_f32((nk_f32_t *)a, n, &alpha_f32, &beta_f32, (nk_f32_t *)out);
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

/**  @brief Recursive helper for impl_reduce_sum.  */
static void reduce_sum_recursive(TensorView const *view, size_t dim, char *ptr, double *result_f, int64_t *result_i) {
    if (dim == view->rank - 1) {
        // Base case: innermost dimension
        size_t const n = (size_t)view->shape[dim];
        Py_ssize_t const stride = view->strides[dim];
        switch (view->dtype) {
        case nk_f64_k:
            for (size_t i = 0; i < n; i++) *result_f += *(nk_f64_t *)(ptr + i * stride);
            break;
        case nk_f32_k:
            for (size_t i = 0; i < n; i++) *result_f += *(nk_f32_t *)(ptr + i * stride);
            break;
        case nk_f16_k: {
            nk_f32_t tmp;
            for (size_t i = 0; i < n; i++) {
                nk_f16_to_f32((nk_f16_t *)(ptr + i * stride), &tmp);
                *result_f += tmp;
            }
        } break;
        case nk_bf16_k: {
            nk_f32_t tmp;
            for (size_t i = 0; i < n; i++) {
                nk_bf16_to_f32((nk_bf16_t *)(ptr + i * stride), &tmp);
                *result_f += tmp;
            }
        } break;
        case nk_i8_k:
            for (size_t i = 0; i < n; i++) *result_i += *(nk_i8_t *)(ptr + i * stride);
            break;
        case nk_u8_k:
            for (size_t i = 0; i < n; i++) *result_i += *(nk_u8_t *)(ptr + i * stride);
            break;
        case nk_i16_k:
            for (size_t i = 0; i < n; i++) *result_i += *(nk_i16_t *)(ptr + i * stride);
            break;
        case nk_u16_k:
            for (size_t i = 0; i < n; i++) *result_i += *(nk_u16_t *)(ptr + i * stride);
            break;
        case nk_i32_k:
            for (size_t i = 0; i < n; i++) *result_i += *(nk_i32_t *)(ptr + i * stride);
            break;
        case nk_u32_k:
            for (size_t i = 0; i < n; i++) *result_i += *(nk_u32_t *)(ptr + i * stride);
            break;
        case nk_i64_k:
            for (size_t i = 0; i < n; i++) *result_i += *(nk_i64_t *)(ptr + i * stride);
            break;
        case nk_u64_k:
            for (size_t i = 0; i < n; i++) *result_i += (int64_t)*(nk_u64_t *)(ptr + i * stride);
            break;
        default: break;
        }
    }
    else {
        // Recursive case: iterate outer dimension
        size_t const n = (size_t)view->shape[dim];
        Py_ssize_t const stride = view->strides[dim];
        for (size_t i = 0; i < n; i++) { reduce_sum_recursive(view, dim + 1, ptr + i * stride, result_f, result_i); }
    }
}

int impl_reduce_sum(TensorView const *view, double *result_f, int64_t *result_i) {
    if (result_f) *result_f = 0;
    if (result_i) *result_i = 0;
    if (view->rank == 0) {
        // Scalar case
        switch (view->dtype) {
        case nk_f64_k: *result_f = *(nk_f64_t *)view->data; return 0;
        case nk_f32_k: *result_f = *(nk_f32_t *)view->data; return 0;
        case nk_i64_k: *result_i = *(nk_i64_t *)view->data; return 0;
        case nk_i32_k: *result_i = *(nk_i32_t *)view->data; return 0;
        default: return -1;
        }
    }
    reduce_sum_recursive(view, 0, view->data, result_f, result_i);
    return 0;
}

/// Macro for min/max reduction over all supported types.
/// @param cmp_op_  Comparison operator (< for min, > for max).
/// @param result_  Pointer to result variable (*global_min_f or *global_max_i).
#define reduce_extremum_(cmp_op_, result_)                                                             \
    do {                                                                                               \
        size_t const n = (size_t)view->shape[dim];                                                     \
        Py_ssize_t const stride = view->strides[dim];                                                  \
        switch (view->dtype) {                                                                         \
        case nk_f64_k:                                                                                 \
            for (size_t i = 0; i < n; i++) {                                                           \
                nk_f64_t v = *(nk_f64_t *)(ptr + i * stride);                                          \
                if (v cmp_op_ * global_f) *global_f = v, *global_idx = base_idx + i;                   \
            }                                                                                          \
            break;                                                                                     \
        case nk_f32_k:                                                                                 \
            for (size_t i = 0; i < n; i++) {                                                           \
                nk_f32_t v = *(nk_f32_t *)(ptr + i * stride);                                          \
                if (v cmp_op_ * global_f) *global_f = v, *global_idx = base_idx + i;                   \
            }                                                                                          \
            break;                                                                                     \
        case nk_f16_k:                                                                                 \
            for (size_t i = 0; i < n; i++) {                                                           \
                nk_f32_t v;                                                                            \
                nk_f16_to_f32((nk_f16_t *)(ptr + i * stride), &v);                                     \
                if (v cmp_op_ * global_f) *global_f = v, *global_idx = base_idx + i;                   \
            }                                                                                          \
            break;                                                                                     \
        case nk_bf16_k:                                                                                \
            for (size_t i = 0; i < n; i++) {                                                           \
                nk_f32_t v;                                                                            \
                nk_bf16_to_f32((nk_bf16_t *)(ptr + i * stride), &v);                                   \
                if (v cmp_op_ * global_f) *global_f = v, *global_idx = base_idx + i;                   \
            }                                                                                          \
            break;                                                                                     \
        case nk_i64_k:                                                                                 \
            for (size_t i = 0; i < n; i++) {                                                           \
                nk_i64_t v = *(nk_i64_t *)(ptr + i * stride);                                          \
                if (v cmp_op_ * global_i) *global_i = v, *global_idx = base_idx + i;                   \
            }                                                                                          \
            break;                                                                                     \
        case nk_i32_k:                                                                                 \
            for (size_t i = 0; i < n; i++) {                                                           \
                nk_i32_t v = *(nk_i32_t *)(ptr + i * stride);                                          \
                if (v cmp_op_ * global_i) *global_i = v, *global_idx = base_idx + i;                   \
            }                                                                                          \
            break;                                                                                     \
        case nk_i16_k:                                                                                 \
            for (size_t i = 0; i < n; i++) {                                                           \
                nk_i16_t v = *(nk_i16_t *)(ptr + i * stride);                                          \
                if (v cmp_op_ * global_i) *global_i = v, *global_idx = base_idx + i;                   \
            }                                                                                          \
            break;                                                                                     \
        case nk_i8_k:                                                                                  \
            for (size_t i = 0; i < n; i++) {                                                           \
                nk_i8_t v = *(nk_i8_t *)(ptr + i * stride);                                            \
                if (v cmp_op_ * global_i) *global_i = v, *global_idx = base_idx + i;                   \
            }                                                                                          \
            break;                                                                                     \
        case nk_u64_k:                                                                                 \
            for (size_t i = 0; i < n; i++) {                                                           \
                nk_u64_t v = *(nk_u64_t *)(ptr + i * stride);                                          \
                if ((int64_t)v cmp_op_ * global_i) *global_i = (int64_t)v, *global_idx = base_idx + i; \
            }                                                                                          \
            break;                                                                                     \
        case nk_u32_k:                                                                                 \
            for (size_t i = 0; i < n; i++) {                                                           \
                nk_u32_t v = *(nk_u32_t *)(ptr + i * stride);                                          \
                if (v cmp_op_ * global_i) *global_i = v, *global_idx = base_idx + i;                   \
            }                                                                                          \
            break;                                                                                     \
        case nk_u16_k:                                                                                 \
            for (size_t i = 0; i < n; i++) {                                                           \
                nk_u16_t v = *(nk_u16_t *)(ptr + i * stride);                                          \
                if (v cmp_op_ * global_i) *global_i = v, *global_idx = base_idx + i;                   \
            }                                                                                          \
            break;                                                                                     \
        case nk_u8_k:                                                                                  \
            for (size_t i = 0; i < n; i++) {                                                           \
                nk_u8_t v = *(nk_u8_t *)(ptr + i * stride);                                            \
                if (v cmp_op_ * global_i) *global_i = v, *global_idx = base_idx + i;                   \
            }                                                                                          \
            break;                                                                                     \
        default: break;                                                                                \
        }                                                                                              \
    } while (0)

/**  @brief Recursive helper for impl_reduce_min.  */
static void reduce_min_recursive_(TensorView const *view, size_t dim, char *ptr, size_t base_idx, double *global_f,
                                  int64_t *global_i, size_t *global_idx) {
    if (dim == view->rank - 1) { reduce_extremum_(<, *global_f); }
    else {
        size_t const n = (size_t)view->shape[dim];
        Py_ssize_t const stride = view->strides[dim];
        size_t inner_size = 1;
        for (size_t d = dim + 1; d < view->rank; d++) inner_size *= (size_t)view->shape[d];
        for (size_t i = 0; i < n; i++)
            reduce_min_recursive_(view, dim + 1, ptr + i * stride, base_idx + i * inner_size, global_f, global_i,
                                  global_idx);
    }
}

/**  @brief Recursive helper for impl_reduce_max.  */
static void reduce_max_recursive_(TensorView const *view, size_t dim, char *ptr, size_t base_idx, double *global_f,
                                  int64_t *global_i, size_t *global_idx) {
    if (dim == view->rank - 1) { reduce_extremum_(>, *global_f); }
    else {
        size_t const n = (size_t)view->shape[dim];
        Py_ssize_t const stride = view->strides[dim];
        size_t inner_size = 1;
        for (size_t d = dim + 1; d < view->rank; d++) inner_size *= (size_t)view->shape[d];
        for (size_t i = 0; i < n; i++)
            reduce_max_recursive_(view, dim + 1, ptr + i * stride, base_idx + i * inner_size, global_f, global_i,
                                  global_idx);
    }
}

#undef reduce_extremum_

int impl_reduce_min(TensorView const *view, double *value_f, int64_t *value_i, size_t *index) {
    *index = 0;
    if (value_f) *value_f = NK_F64_INF;
    if (value_i) *value_i = NK_I64_MAX;
    if (view->rank == 0) {
        switch (view->dtype) {
        case nk_f64_k: *value_f = *(nk_f64_t *)view->data; return 0;
        case nk_f32_k: *value_f = *(nk_f32_t *)view->data; return 0;
        case nk_i64_k: *value_i = *(nk_i64_t *)view->data; return 0;
        case nk_i32_k: *value_i = *(nk_i32_t *)view->data; return 0;
        case nk_i16_k: *value_i = *(nk_i16_t *)view->data; return 0;
        case nk_i8_k: *value_i = *(nk_i8_t *)view->data; return 0;
        case nk_u64_k: *value_i = (int64_t)*(nk_u64_t *)view->data; return 0;
        case nk_u32_k: *value_i = *(nk_u32_t *)view->data; return 0;
        case nk_u16_k: *value_i = *(nk_u16_t *)view->data; return 0;
        case nk_u8_k: *value_i = *(nk_u8_t *)view->data; return 0;
        default: return -1;
        }
    }
    reduce_min_recursive_(view, 0, view->data, 0, value_f, value_i, index);
    return 0;
}

int impl_reduce_max(TensorView const *view, double *value_f, int64_t *value_i, size_t *index) {
    *index = 0;
    if (value_f) *value_f = -NK_F64_INF;
    if (value_i) *value_i = NK_I64_MIN;
    if (view->rank == 0) {
        switch (view->dtype) {
        case nk_f64_k: *value_f = *(nk_f64_t *)view->data; return 0;
        case nk_f32_k: *value_f = *(nk_f32_t *)view->data; return 0;
        case nk_i64_k: *value_i = *(nk_i64_t *)view->data; return 0;
        case nk_i32_k: *value_i = *(nk_i32_t *)view->data; return 0;
        case nk_i16_k: *value_i = *(nk_i16_t *)view->data; return 0;
        case nk_i8_k: *value_i = *(nk_i8_t *)view->data; return 0;
        case nk_u64_k: *value_i = (int64_t)*(nk_u64_t *)view->data; return 0;
        case nk_u32_k: *value_i = *(nk_u32_t *)view->data; return 0;
        case nk_u16_k: *value_i = *(nk_u16_t *)view->data; return 0;
        case nk_u8_k: *value_i = *(nk_u8_t *)view->data; return 0;
        default: return -1;
        }
    }
    reduce_max_recursive_(view, 0, view->data, 0, value_f, value_i, index);
    return 0;
}

#pragma endregion // Reductions

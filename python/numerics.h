/**
 *  @brief NumKong Numeric Operations for Python.
 *  @file python/numerics.h
 *  @author Ash Vardanian
 *  @date December 30, 2025
 *
 *  This header declares numeric API functions including elementwise operations,
 *  reductions, trigonometry, and mesh alignment (Kabsch/Umeyama).
 */
#ifndef NK_PYTHON_NUMERICS_H
#define NK_PYTHON_NUMERICS_H

#include "numkong.h"

#ifdef __cplusplus
extern "C" {
#endif

#pragma region Shared Implementations

/**
 *  @brief Stride-aware elementwise addition: outᵢ = aᵢ + bᵢ.
 *
 *  Handles contiguous and strided data. Both operands must have the same dtype.
 *  Supports f32, f64, i8, i32.
 *
 *  @param[in] a First operand data pointer.
 *  @param[in] b Second operand data pointer.
 *  @param[out] out Output data pointer.
 *  @param[in] n Number of elements.
 *  @param[in] dtype Datatype of all operands.
 *  @param[in] stride_a Stride of a in bytes (use bytes_per_dtype(dtype) for contiguous).
 *  @param[in] stride_b Stride of b in bytes.
 *  @param[in] stride_out Stride of out in bytes.
 *  @return 0 on success, -1 if dtype not supported.
 */
int impl_elementwise_add(char *a, char *b, char *out, size_t n, nk_dtype_t dtype, Py_ssize_t stride_a,
                         Py_ssize_t stride_b, Py_ssize_t stride_out);

/**
 *  @brief Stride-aware elementwise multiplication: outᵢ = aᵢ · bᵢ.
 *  @see impl_elementwise_add for parameter details.
 */
int impl_elementwise_mul(char *a, char *b, char *out, size_t n, nk_dtype_t dtype, Py_ssize_t stride_a,
                         Py_ssize_t stride_b, Py_ssize_t stride_out);

/**
 *  @brief Stride-aware weighted sum: outᵢ = α · aᵢ + β · bᵢ.
 *  @param[in] alpha Coefficient for first operand.
 *  @param[in] beta Coefficient for second operand.
 *  @see impl_elementwise_add for other parameter details.
 */
int impl_elementwise_wsum(char *a, char *b, char *out, size_t n, nk_dtype_t dtype, double alpha, double beta,
                          Py_ssize_t stride_a, Py_ssize_t stride_b, Py_ssize_t stride_out);

/**
 *  @brief Stride-aware scale: outᵢ = α · aᵢ + β.
 *  @param[in] a Input data pointer.
 *  @param[out] out Output data pointer.
 *  @param[in] n Number of elements.
 *  @param[in] dtype Datatype.
 *  @param[in] alpha Multiplicative coefficient.
 *  @param[in] beta Additive offset.
 *  @param[in] stride_a Stride of a in bytes.
 *  @param[in] stride_out Stride of out in bytes.
 *  @return 0 on success, -1 if dtype not supported.
 */
int impl_elementwise_scale(char *a, char *out, size_t n, nk_dtype_t dtype, double alpha, double beta,
                           Py_ssize_t stride_a, Py_ssize_t stride_out);

/**
 *  @brief Stride-aware reduction: ∑ᵢ xᵢ (sum all elements).
 *
 *  Traverses an N-dimensional tensor with arbitrary strides.
 *  Returns floating-point result for float types, integer for int types.
 *
 *  @param[in] view TensorView describing the tensor.
 *  @param[out] result_f Output for floating-point result (may be NULL for int types).
 *  @param[out] result_i Output for integer result (may be NULL for float types).
 *  @return 0 on success, -1 if dtype not supported.
 */
int impl_reduce_sum(TensorView const *view, double *result_f, int64_t *result_i);

/**
 *  @brief Stride-aware reduction: find minimum and its index.
 *  @param[in] view TensorView describing the tensor.
 *  @param[out] value_f Output for minimum value (float types).
 *  @param[out] value_i Output for minimum value (int types).
 *  @param[out] index Output for flat index of minimum element.
 *  @return 0 on success, -1 if dtype not supported.
 */
int impl_reduce_min(TensorView const *view, double *value_f, int64_t *value_i, size_t *index);

/**
 *  @brief Stride-aware reduction: find maximum and its index.
 *  @see impl_reduce_min for parameter details.
 */
int impl_reduce_max(TensorView const *view, double *value_f, int64_t *value_i, size_t *index);

#pragma endregion // Shared Implementations

#ifdef __cplusplus
}
#endif

#endif // NK_PYTHON_NUMERICS_H

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
 *  @brief Stride-aware reduction: compute sum and sum-of-squares (moments).
 *
 *  Traverses an N-dimensional tensor with arbitrary strides.
 *  Returns floating-point results for float types, integer for int types.
 *
 *  @param[in] view TensorView describing the tensor.
 *  @param[out] sum_f Output for floating-point sum (may be NULL for int types).
 *  @param[out] sum_i Output for integer sum (may be NULL for float types).
 *  @param[out] sumsq_f Output for floating-point sum of squares (may be NULL for int types).
 *  @param[out] sumsq_i Output for integer sum of squares (may be NULL for float types).
 *  @return 0 on success, -1 if dtype not supported.
 */
int impl_reduce_moments(TensorView const *view, double *sum_f, int64_t *sum_i, double *sumsq_f, int64_t *sumsq_i);

/**
 *  @brief Stride-aware reduction: find minimum and maximum with their indices.
 *
 *  Traverses an N-dimensional tensor with arbitrary strides.
 *  Returns both min and max values along with their flat indices.
 *
 *  @param[in] view TensorView describing the tensor.
 *  @param[out] min_f Output for minimum value (float types, may be NULL for int types).
 *  @param[out] min_i Output for minimum value (int types, may be NULL for float types).
 *  @param[out] min_index Output for flat index of minimum element.
 *  @param[out] max_f Output for maximum value (float types, may be NULL for int types).
 *  @param[out] max_i Output for maximum value (int types, may be NULL for float types).
 *  @param[out] max_index Output for flat index of maximum element.
 *  @return 0 on success, -1 if dtype not supported.
 */
int impl_reduce_minmax(TensorView const *view, double *min_f, int64_t *min_i, size_t *min_index, double *max_f,
                       int64_t *max_i, size_t *max_index);

#pragma endregion // Shared Implementations

#ifdef __cplusplus
}
#endif

#endif // NK_PYTHON_NUMERICS_H

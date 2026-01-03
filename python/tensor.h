/**
 *  @brief Tensor type declarations for NumKong Python bindings.
 *  @file python/tensor.h
 *
 *  This header declares the Tensor N-dimensional array type, its iterator,
 *  and the MatrixMultiplier (pre-packed matrix for fast GEMM).
 *  These types provide a NumPy-like interface with support for NumKong's
 *  extended type system including bfloat16, float8, and complex types.
 */
#ifndef NK_TENSOR_H
#define NK_TENSOR_H

#include "numkong.h"

#ifdef __cplusplus
extern "C" {
#endif

#pragma region Tensor Type

/**
 *  @brief N-dimensional array type with NumPy-like interface.
 *
 *  Supports arbitrary strides for views and slices, reference counting
 *  for memory management, and NumKong's extended type system.
 *
 *  Memory layout:
 *  - If parent == NULL: owns data in variable-length `start[]` buffer
 *  - If parent != NULL: view into parent's memory, `data` points there
 */
typedef struct Tensor {
    PyObject_HEAD nk_datatype_t datatype;   ///< Logical datatype (f32, f64, bf16, etc.)
    size_t rank;                            ///< Number of dimensions (0 for scalar)
    Py_ssize_t shape[NK_TENSOR_MAX_RANK];   ///< Extent along each dimension
    Py_ssize_t strides[NK_TENSOR_MAX_RANK]; ///< Stride in bytes for each dimension
    PyObject *parent;                       ///< Reference to parent (NULL if owns data)
    char *data;                             ///< Data pointer (start[] if owns, parent's if view)
    char start[];                           ///< Variable-length inline data storage
} Tensor;

/**
 *  @brief Iterator for Tensor.
 *
 *  Iterates over the first dimension of a Tensor, yielding views
 *  (for rank > 1) or scalars (for rank == 1).
 */
typedef struct TensorIter {
    PyObject_HEAD Tensor *array; ///< Tensor being iterated
    Py_ssize_t index;            ///< Current position
} TensorIter;

#pragma endregion // Tensor Type

#pragma region MatrixMultiplier Type

/**
 *  @brief Pre-packed matrix optimized for matrix multiplication.
 *
 *  Stores matrix data in a hardware-optimized layout (e.g., for AMX, AVX-512).
 *  Created via `nk.pack_matrix()` and used with `nk.matmul()` or the `@` operator.
 *
 *  Supported dtypes: bfloat16, int8
 */
typedef struct MatrixMultiplier {
    PyObject_HEAD nk_datatype_t dtype; ///< Packed datatype (bf16 or i8)
    nk_size_t n;                       ///< Number of rows in original matrix
    nk_size_t k;                       ///< Number of columns in original matrix
    char start[];                      ///< Variable-length packed data
} MatrixMultiplier;

#pragma endregion // MatrixMultiplier Type

#pragma region Type Objects

/**  @brief Tensor Python type object.  */
extern PyTypeObject TensorType;

/**  @brief Tensor iterator Python type object.  */
extern PyTypeObject TensorIterType;

/**  @brief MatrixMultiplier Python type object.  */
extern PyTypeObject MatrixMultiplierType;

#pragma endregion // Type Objects

#pragma region Tensor Factory Functions

/**
 *  @brief Allocate a new Tensor with uninitialized data.
 *  @param[in] datatype Logical datatype for elements.
 *  @param[in] rank Number of dimensions.
 *  @param[in] shape Array of dimension sizes.
 *  @return New Tensor, or NULL on allocation failure.
 */
Tensor *Tensor_new(nk_datatype_t datatype, size_t rank, Py_ssize_t const *shape);

/**
 *  @brief Create a view into an existing Tensor.
 *
 *  The view shares memory with the parent and holds a reference to it.
 *
 *  @param[in] parent Parent Tensor (reference count incremented).
 *  @param[in] data Pointer to first element of view.
 *  @param[in] datatype Logical datatype (usually same as parent).
 *  @param[in] rank Number of dimensions.
 *  @param[in] shape Array of dimension sizes.
 *  @param[in] strides Array of byte strides.
 *  @return New Tensor view, or NULL on failure.
 */
Tensor *Tensor_view(Tensor *parent, char *data, nk_datatype_t datatype, size_t rank, Py_ssize_t const *shape,
                    Py_ssize_t const *strides);

#pragma endregion // Tensor Factory Functions

#pragma region Tensor Methods

/**  @brief Copy the tensor.  */
PyObject *Tensor_copy(PyObject *self, PyObject *args);

/**  @brief Reshape the tensor (returns view if possible).  */
PyObject *Tensor_reshape(PyObject *self, PyObject *args);

/**  @brief Sum all elements.  */
PyObject *Tensor_sum(PyObject *self, PyObject *args);

/**  @brief Find minimum element.  */
PyObject *Tensor_min(PyObject *self, PyObject *args);

/**  @brief Find maximum element.  */
PyObject *Tensor_max(PyObject *self, PyObject *args);

/**  @brief Find index of minimum element.  */
PyObject *Tensor_argmin(PyObject *self, PyObject *args);

/**  @brief Find index of maximum element.  */
PyObject *Tensor_argmax(PyObject *self, PyObject *args);

#pragma endregion // Tensor Methods

#pragma region Module API Functions

// Constructors
PyObject *api_empty(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);
PyObject *api_zeros(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);
PyObject *api_ones(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);
PyObject *api_full(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);

// Reductions
PyObject *api_sum(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);
PyObject *api_min(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);
PyObject *api_max(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);
PyObject *api_argmin(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);
PyObject *api_argmax(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);

// Matrix operations
PyObject *api_pack_matrix(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);
PyObject *api_pack_matmul_argument(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);
PyObject *api_matmul(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);

#pragma endregion // Module API Functions

#pragma region Documentation Strings

extern char const doc_empty[];
extern char const doc_zeros[];
extern char const doc_ones[];
extern char const doc_full[];
extern char const doc_reduce_sum[];
extern char const doc_reduce_min[];
extern char const doc_reduce_max[];
extern char const doc_reduce_argmin[];
extern char const doc_reduce_argmax[];
extern char const doc_pack_matrix[];
extern char const doc_pack_matmul_argument[];
extern char const doc_matmul[];

#pragma endregion // Documentation Strings

#ifdef __cplusplus
}
#endif

#endif // NK_TENSOR_H

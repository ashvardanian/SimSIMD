/**
 *  @brief Matrix multiplication and symmetric operations for NumKong Python bindings.
 *  @file python/matrix.h
 *  @author Ash Vardanian
 *  @date February 20, 2026
 *
 *  Declares the PackedMatrix type and API functions for matrix
 *  multiplication (dots_pack, dots_packed, Tensor @), symmetric dot products,
 *  symmetric Hamming distances, and packed Hamming operations.
 */
#ifndef NK_PYTHON_MATRIX_H
#define NK_PYTHON_MATRIX_H

#include "numkong.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 *  @brief Pre-packed matrix optimized for matrix multiplication or Hamming distance.
 *
 *  Stores matrix data in a hardware-optimized layout (e.g., for AMX, AVX-512).
 *  Created via `nk.dots_pack()` or `nk.hammings_pack()` and used with
 *  `nk.dots_packed()`, `nk.hammings_packed()`, or the `@` operator.
 */
typedef struct PackedMatrix {
    PyObject_HEAD
    /** Kernel kind: `nk_kernel_dots_packed_k` or `nk_kernel_hammings_packed_k`. */
    nk_kernel_kind_t kind;
    /** Packed dtype (bf16, i8, f32, etc.). */
    nk_dtype_t dtype;
    /** Number of rows in original matrix. */
    nk_size_t n;
    /** Number of columns in original matrix. */
    nk_size_t k;
    /** Variable-length packed data. */
    char start[];
} PackedMatrix;

/** @brief PackedMatrix Python type object.  */
extern PyTypeObject PackedMatrixType;

/** @brief Pack a matrix into hardware-optimized layout for dot-product matmul. */
PyObject *api_dots_pack(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);
/** @brief Matrix multiplication with a pre-packed B matrix. */
PyObject *api_dots_packed(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);
/** @brief Pack a matrix into hardware-optimized layout for Hamming distance. */
PyObject *api_hammings_pack(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);
/** @brief Hamming distance computation with a pre-packed B matrix. */
PyObject *api_hammings_packed(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);
/** @brief All-pairs dot products within a single matrix. */
PyObject *api_dots_symmetric(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);
/** @brief All-pairs Hamming distances within a single matrix. */
PyObject *api_hammings_symmetric(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);

extern char const doc_dots_pack[];
extern char const doc_dots_packed[];
extern char const doc_hammings_pack[];
extern char const doc_hammings_packed[];
extern char const doc_dots_symmetric[];
extern char const doc_hammings_symmetric[];

/**
 *  @brief Tensor @ PackedMatrix operator implementation.
 *  Used by Tensor's nb_matrix_multiply slot.
 */
PyObject *Tensor_matmul(PyObject *self, PyObject *other);

#ifdef __cplusplus
}
#endif

#endif // NK_PYTHON_MATRIX_H

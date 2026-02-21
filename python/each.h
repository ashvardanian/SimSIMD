/**
 *  @brief Elementwise operation declarations for NumKong Python bindings.
 *  @file python/each.h
 *  @author Ash Vardanian
 *  @date February 19, 2026
 *
 *  Forward declarations for all api_* elementwise and trigonometric functions,
 *  and their documentation strings.
 */
#ifndef NK_PYTHON_EACH_H
#define NK_PYTHON_EACH_H

#include "numkong.h"

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Compute fused multiply-add: alpha*a*b + beta*c. */
PyObject *api_fma(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);
/** @brief Compute weighted sum (blend): alpha*a + beta*b. */
PyObject *api_wsum(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);
/** @brief Scale a tensor: alpha*a + beta. */
PyObject *api_scale(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);
/** @brief Elementwise addition of two tensors or a tensor and a scalar. */
PyObject *api_add(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);
/** @brief Elementwise multiplication of two tensors or a tensor and a scalar. */
PyObject *api_multiply(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);

/** @brief Elementwise sine. */
PyObject *api_sin(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);
/** @brief Elementwise cosine. */
PyObject *api_cos(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);
/** @brief Elementwise arctangent. */
PyObject *api_atan(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);

extern char const doc_fma[];
extern char const doc_wsum[];
extern char const doc_scale[];
extern char const doc_add[];
extern char const doc_multiply[];
extern char const doc_sin[];
extern char const doc_cos[];
extern char const doc_atan[];

#ifdef __cplusplus
}
#endif

#endif // NK_PYTHON_EACH_H

/**
 *  @brief Low-Precision Scalar Wrappers for Python.
 *  @file python/scalars.h
 *  @author Ash Vardanian
 *  @date December 30, 2025
 *
 *  Pure Python C API scalar types for bfloat16, float8_e4m3, and float8_e5m2.
 *  These work without NumPy - they're standalone Python objects:
 *    nk.bfloat16(3.14)           → create a bfloat16 scalar
 *    float(nk.bfloat16(3.14))    → convert back to Python float
 *    nk.bfloat16(3.14) == 3.14   → comparison with other numbers
 *    nk.bfloat16(1.0) + nk.bfloat16(2.0)  → arithmetic
 */
#ifndef NK_PYTHON_SCALARS_H
#define NK_PYTHON_SCALARS_H

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <numkong/types.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    PyObject_HEAD
    /** Stored value. */
    nk_f16_t value;
} NkFloat16ScalarObject;

typedef struct {
    PyObject_HEAD
    /** Stored value. */
    nk_bf16_t value;
} NkBFloat16ScalarObject;

typedef struct {
    PyObject_HEAD
    /** Stored value. */
    nk_e4m3_t value;
} NkFloat8E4M3ScalarObject;

typedef struct {
    PyObject_HEAD
    /** Stored value. */
    nk_e5m2_t value;
} NkFloat8E5M2ScalarObject;

typedef struct {
    PyObject_HEAD
    /** Stored value. */
    nk_e2m3_t value;
} NkFloat6E2M3ScalarObject;

typedef struct {
    PyObject_HEAD
    /** Stored value. */
    nk_e3m2_t value;
} NkFloat6E3M2ScalarObject;

extern PyTypeObject NkFloat16Scalar_Type;
extern PyTypeObject NkBFloat16Scalar_Type;
extern PyTypeObject NkFloat8E4M3Scalar_Type;
extern PyTypeObject NkFloat8E5M2Scalar_Type;
extern PyTypeObject NkFloat6E2M3Scalar_Type;
extern PyTypeObject NkFloat6E3M2Scalar_Type;

/**
 *  @brief Register NumKong scalar types with Python.
 *  @param[in] module The numkong module object.
 *  @return 0 on success, -1 on failure (with Python exception set).
 *
 *  This function should be called from PyInit_numkong().
 *  It adds bfloat16, float16, float8_e4m3, float8_e5m2, float6_e2m3, and
 *  float6_e3m2 scalar types to the module.
 */
int nk_register_scalar_types(PyObject *module);

#ifdef __cplusplus
}
#endif

#endif // NK_PYTHON_SCALARS_H

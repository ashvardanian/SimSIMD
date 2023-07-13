/**
 *  @file python.c
 *  @author Ashot Vardanian
 *  @date 2023-01-30
 *  @copyright Copyright (c) 2023
 *
 *  @brief Pure CPython bindings for simsimd.
 */

#include "simsimd/simsimd.h"
#include "simsimd/simsimd_chem.h"

#define PY_SSIZE_T_CLEAN
#include <Python.h>

static void pseudo_destroy(PyObject* obj) { (void)obj; }

PyObject* distance(void* func) { return PyCapsule_New(func, NULL, pseudo_destroy); }

static PyObject* to_int(PyObject* self, PyObject* args) {
    PyObject* capsule;
    if (!PyArg_ParseTuple(args, "O", &capsule)) {
        return NULL;
    }

    if (!PyCapsule_IsValid(capsule, NULL)) {
        PyErr_SetString(PyExc_ValueError, "Object is not a valid capsule");
        return NULL;
    }

    void* pointer = PyCapsule_GetPointer(capsule, NULL);

    return PyLong_FromVoidPtr(pointer);
}

static PyMethodDef simsimd_methods[] = {
    {"to_int", (PyCFunction)to_int, METH_VARARGS, "Converts CPython capsule to `int`"},
    {NULL, NULL, 0, NULL} /* Sentinel */
};

static PyModuleDef simsimd_module = {
    PyModuleDef_HEAD_INIT,
    .m_name = "SimSIMD",
    .m_doc = "SIMD-accelerated similarity measures for x86 and Arm: AVX2, AVX512, NEON, SVE",
    .m_size = -1,
    .m_methods = simsimd_methods,
};

PyMODINIT_FUNC PyInit_simsimd(void) {
    PyObject* m = PyModule_Create(&simsimd_module);
    if (m == NULL)
        return NULL;

    PyModule_AddObject(m, "dot_f32_sve", distance(&simsimd_dot_f32_sve));
    PyModule_AddObject(m, "dot_f32x4_neon", distance(&simsimd_dot_f32x4_neon));

    PyModule_AddObject(m, "cos_f32_sve", distance(&simsimd_cos_f32_sve));
    PyModule_AddObject(m, "cos_f16_sve", distance(&simsimd_cos_f16_sve));
    PyModule_AddObject(m, "cos_f16x4_neon", distance(&simsimd_cos_f16x4_neon));
    PyModule_AddObject(m, "cos_i8x16_neon", distance(&simsimd_cos_i8x16_neon));
    PyModule_AddObject(m, "cos_f32x4_neon", distance(&simsimd_cos_f32x4_neon));
    PyModule_AddObject(m, "cos_f16x16_avx512", distance(&simsimd_cos_f16x16_avx512));
    PyModule_AddObject(m, "cos_f32x4_avx2", distance(&simsimd_cos_f32x4_avx2));

    PyModule_AddObject(m, "l2sq_f32_sve", distance(&simsimd_l2sq_f32_sve));
    PyModule_AddObject(m, "l2sq_f16_sve", distance(&simsimd_l2sq_f16_sve));

    PyModule_AddObject(m, "hamming_b1x8_sve", distance(&simsimd_hamming_b1x8_sve));
    PyModule_AddObject(m, "hamming_b1x128_sve", distance(&simsimd_hamming_b1x128_sve));
    PyModule_AddObject(m, "hamming_b1x128_avx512", distance(&simsimd_hamming_b1x128_avx512));

    PyModule_AddObject(m, "tanimoto_b1x8_naive", distance(&simsimd_tanimoto_b1x8_naive));

    // Specific vector sizes for chemistry
    PyModule_AddObject(m, "tanimoto_maccs_naive", distance(&simsimd_tanimoto_maccs_naive));
    PyModule_AddObject(m, "tanimoto_maccs_neon", distance(&simsimd_tanimoto_maccs_neon));
    PyModule_AddObject(m, "tanimoto_maccs_sve", distance(&simsimd_tanimoto_maccs_sve));
    PyModule_AddObject(m, "tanimoto_maccs_avx512", distance(&simsimd_tanimoto_maccs_avx512));

    return m;
}

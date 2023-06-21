/**
 *  @file python.c
 *  @author Ashot Vardanian
 *  @date 2023-01-30
 *  @copyright Copyright (c) 2023
 *
 *  @brief Pure CPython bindings for simsimd.
 */

#include "simsimd/simsimd.h"

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define stringify_value_m(a) stringify_m(a)
#define stringify_m(a) #a
#define concat_m(A, B) A##B
#define macro_concat_m(A, B) concat_m(A, B)
#define pyinit_f_m macro_concat_m(PyInit_, SIMDSIMD_PYTHON_MODULE_NAME)

static void destroy(PyObject*) {}

static PyModuleDef simsimd_wrapper_module = {
    PyModuleDef_HEAD_INIT,
    .m_name = "SimSIMD",
    .m_doc =
        "SIMD-accelerated similarity measures, metrics, distance functions for x86 and Arm: AVX2, AVX512, NEON, SVE",
    .m_size = -1,
};

PyMODINIT_FUNC pyinit_f_m(void) {
    PyObject* module = PyModule_Create(&simsimd_wrapper_module);
    if (module == NULL)
        return NULL;

    PyModule_AddObject(module, "dot_f32sve", PyCapsule_New(simsimd_dot_f32sve, NULL, destroy));
    PyModule_AddObject(module, "cos_f32sve", PyCapsule_New(simsimd_cos_f32sve, NULL, destroy));
    PyModule_AddObject(module, "l2sq_f32sve", PyCapsule_New(simsimd_l2sq_f32sve, NULL, destroy));
    PyModule_AddObject(module, "l2sq_f16sve", PyCapsule_New(simsimd_l2sq_f16sve, NULL, destroy));
    PyModule_AddObject(module, "hamming_b1x8sve", PyCapsule_New(simsimd_hamming_b1x8sve, NULL, destroy));
    PyModule_AddObject(module, "dot_f32x4neon", PyCapsule_New(simsimd_dot_f32x4neon, NULL, destroy));
    PyModule_AddObject(module, "cos_f16x4neon", PyCapsule_New(simsimd_cos_f16x4neon, NULL, destroy));
    PyModule_AddObject(module, "cos_i8x16neon", PyCapsule_New(simsimd_cos_i8x16neon, NULL, destroy));
    PyModule_AddObject(module, "cos_f32x4neon", PyCapsule_New(simsimd_cos_f32x4neon, NULL, destroy));
    PyModule_AddObject(module, "hamming_b1x128sve", PyCapsule_New(simsimd_hamming_b1x128sve, NULL, destroy));
    PyModule_AddObject(module, "cos_f16x16avx512", PyCapsule_New(simsimd_cos_f16x16avx512, NULL, destroy));
    PyModule_AddObject(module, "dot_i8x16avx2", PyCapsule_New(simsimd_dot_i8x16avx2, NULL, destroy));
    PyModule_AddObject(module, "cos_f32x4avx2", PyCapsule_New(simsimd_cos_f32x4avx2, NULL, destroy));
    PyModule_AddObject(module, "hamming_b1x128avx512", PyCapsule_New(simsimd_hamming_b1x128avx512, NULL, destroy));

    return module;
}
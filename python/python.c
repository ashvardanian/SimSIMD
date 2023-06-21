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


static void destroy_capsule(PyObject*) {}

static PyModuleDef simsimd_wrapper_module = {
    PyModuleDef_HEAD_INIT,
    .m_name = "simsimd",
    .m_doc = "simsimd module",
    .m_size = -1,
};

PyMODINIT_FUNC pyinit_f_m(void) {
    PyObject* module = PyModule_Create(&simsimd_wrapper_module);
    if (module == NULL) 
        return NULL;

    PyModule_AddObject(module, "dot_f32sve", PyCapsule_New(simsimd_dot_f32sve, NULL, destroy_capsule));
    PyModule_AddObject(module, "cos_f32sve", PyCapsule_New(simsimd_cos_f32sve, NULL, destroy_capsule));
    PyModule_AddObject(module, "l2sq_f32sve", PyCapsule_New(simsimd_l2sq_f32sve, NULL, destroy_capsule));
    PyModule_AddObject(module, "l2sq_f16sve", PyCapsule_New(simsimd_l2sq_f16sve, NULL, destroy_capsule));
    PyModule_AddObject(module, "hamming_b1x8sve", PyCapsule_New(simsimd_hamming_b1x8sve, NULL, destroy_capsule));
    PyModule_AddObject(module, "dot_f32x4neon", PyCapsule_New(simsimd_dot_f32x4neon, NULL, destroy_capsule));
    PyModule_AddObject(module, "cos_f16x4neon", PyCapsule_New(simsimd_cos_f16x4neon, NULL, destroy_capsule));
    PyModule_AddObject(module, "cos_i8x16neon", PyCapsule_New(simsimd_cos_i8x16neon, NULL, destroy_capsule));
    PyModule_AddObject(module, "cos_f32x4neon", PyCapsule_New(simsimd_cos_f32x4neon, NULL, destroy_capsule));
    PyModule_AddObject(module, "hamming_b1x128sve", PyCapsule_New(simsimd_hamming_b1x128sve, NULL, destroy_capsule));
    PyModule_AddObject(module, "cos_f16x16avx512", PyCapsule_New(simsimd_cos_f16x16avx512, NULL, destroy_capsule));
    PyModule_AddObject(module, "dot_i8x16avx2", PyCapsule_New(simsimd_dot_i8x16avx2, NULL, destroy_capsule));
    PyModule_AddObject(module, "cos_f32x4avx2", PyCapsule_New(simsimd_cos_f32x4avx2, NULL, destroy_capsule));
    PyModule_AddObject(module, "hamming_b1x128avx512", PyCapsule_New(simsimd_hamming_b1x128avx512, NULL, destroy_capsule));

    return module;
}

int main(int argc, char* argv[]) {
    wchar_t* program = Py_DecodeLocale(argv[0], NULL);
    if (!program) {
        fprintf(stderr, "Fatal error: cannot decode argv[0]\n");
        exit(1);
    }

    if (PyImport_AppendInittab("simsimd." stringify_value_m(SIMSIMD_PYTHON_MODULE_NAME), pyinit_f_m) == -1) {
        fprintf(stderr, "Error: could not extend in-built modules table\n");
        exit(1);
    }

    Py_SetProgramName(program);
    Py_Initialize();

    PyObject* pmodule = PyImport_ImportModule("simsimd." stringify_value_m(SIMSIMD_PYTHON_MODULE_NAME));
    if (!pmodule) {
        PyErr_Print();
        fprintf(stderr, "Error: could not import module 'simsimd'\n");
    }
    PyMem_RawFree(program);
    return 0;
}
/**
 * @file python.c
 * @author Ashot Vardanian
 * @date 2023-01-30
 * @copyright Copyright (c) 2023
 *
 * @brief Pure CPython bindings for simsimd.
 */

#define PY_SSIZE_T_CLEAN
#include "simsimd.h"
#include <Python.h>

#define stringify_value_m(a) stringify_m(a)
#define stringify_m(a) #a
#define concat_m(A, B) A##B
#define macro_concat_m(A, B) concat_m(A, B)
#define pyinit_f_m macro_concat_m(PyInit_, SIMDSIMD_PYTHON_MODULE_NAME)

static PyObject* dot_f32sve_wrap(PyObject* self, PyObject* args) {
    simsimd_f32_t *a, b;
    PyObject* obj_a;
    PyObject* obj_b;
    if (!PyArg_ParseTuple(args, "OO", &obj_a, &obj_b))
        return nullptr;

    Py_ssize_t d = PyList_Size(obj_a);
    if (d != PyList_Size(obj_b)) {
        PyErr_SetString(PyExc_ValueError, "Arrays must have the same size");
        return nullptr;
    }

    simsimd_f32_t* ptr_a = (simsimd_f32_t*)malloc(d * sizeof(simsimd_f32_t));
    simsimd_f32_t* ptr_b = (simsimd_f32_t*)malloc(d * sizeof(simsimd_f32_t));
    if (ptr_a == nullptr || ptr_b == nullptr) {
        free(ptr_a);
        free(ptr_b);
        PyErr_SetString(PyExc_RuntimeError, "Memory allocation failed");
        return nullptr;
    }

    for (Py_ssize_t i = 0; i < d; i++) {
        PyObject* item_a = PyList_GetItem(obj_a, i);
        PyObject* item_b = PyList_GetItem(obj_b, i);
        ptr_a[i] = PyFloat_AsDouble(item_a);
        ptr_b[i] = PyFloat_AsDouble(item_b);
    }
    simsimd_f32_t result = simsimd_dot_f32sve(ptr_a, ptr_b, d);

    free(ptr_a);
    free(ptr_b);
    return result;
}

static void simsimd_capsule_destructor(PyObject* capsule) {
    void* ptr = PyCapsule_GetPointer(capsule, NULL);
    free(ptr);
}

static PyMethodDef simsimd_wrapper_methods[] = {
    {"dot_f32sve", dot_f32sve_wrap, METH_VARARGS, ""},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef simsimd_wrapper_module = {
    PyModuleDef_HEAD_INIT,
    "simsimd_wrapper",
    NULL,
    -1,
    simsimd_wrapper_methods
};

PyMODINIT_FUNC pyinit_f_m(void) {
    PyObject* module = PyModule_Create(&simsimd_wrapper_module);
    if (module == NULL) 
        return NULL;

    PyObject* dot_f32sve_capsule = PyCapsule_New(simsimd_dot_f32sve, NULL, simsimd_capsule_destructor);
    
    PyModule_AddObject(module, "dot_f32sve", dot_f32sve_capsule);

    return module;
}

int main(int argc, char* argv[]) {
    wchar_t* program = Py_DecodeLocale(argv[0], nullptr);
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
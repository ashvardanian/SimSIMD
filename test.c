#include <stdio.h>

#include "simsimd/simsimd.h"

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define stringify_value_m(a) stringify_m(a)
#define stringify_m(a) #a
#define concat_m(A, B) A##B
#define macro_concat_m(A, B) concat_m(A, B)
#define pyinit_f_m macro_concat_m(PyInit_, SIMDSIMD_PYTHON_MODULE_NAME)

char const* names[14] = {
    "dot_f32sve",       "cos_f32sve",    "l2sq_f32sve",   "l2sq_f16sve",          "hamming_b1x8sve",
    "dot_f32x4neon",    "cos_f16x4neon", "cos_i8x16neon", "cos_f32x4neon",        "hamming_b1x128sve",
    "cos_f16x16avx512", "dot_i8x16avx2", "cos_f32x4avx2", "hamming_b1x128avx512",
};
void* callables[14] = {NULL};

void assign(char const* name, void* function) {
    for (size_t idx = 0; idx < 14; ++idx) {
        if (strcmp(name, names[idx]) == 0)
            callables[idx] = function;
    }
}

typedef simsimd_f32_t (*FunctionPtr)(simsimd_f32_t*, simsimd_f32_t*, size_t);

int main(int argc, char* argv[]) {
    Py_Initialize();
    PyObject* module = PyImport_ImportModule("simsimd");
    if (module == NULL)
        return -1;

    PyObject* dictionary = PyModule_GetDict(module);
    PyObject *key, *value;
    Py_ssize_t pos = 0;

    while (PyDict_Next(dictionary, &pos, &key, &value)) {
        if (PyCapsule_CheckExact(value)) {
            const char* name = PyCapsule_GetName(value);
            void* pointer = PyCapsule_GetPointer(value, NULL);
            assign(name, pointer);
        }
    }

    FunctionPtr function = (FunctionPtr)callables[0];
    simsimd_f32_t a[] = {1, 2, 3};
    simsimd_f32_t b[] = {4, 5, 6};
    size_t d = 3;

    simsimd_f32_t result = function(a, b, d);

    printf("%f", result);
    Py_DECREF(module);
}

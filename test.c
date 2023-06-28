#include <stdio.h>

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "simsimd/simsimd.h"

#define stringify_value_m(a) stringify_m(a)
#define stringify_m(a) #a
#define concat_m(A, B) A##B
#define macro_concat_m(A, B) concat_m(A, B)
#define pyinit_f_m macro_concat_m(PyInit_, SIMDSIMD_PYTHON_MODULE_NAME)

char const* names[14] = {
    "dot_f32sve",    "cos_f32sve",      "l2sq_f32sve",       "cos_f32x4avx2",        "cos_f32x4neon",
    "dot_f32x4neon", "cos_f16x4neon",   "cos_f16x16avx512",  "l2sq_f16sve",          "cos_i8x16neon",
    "dot_i8x16avx2", "hamming_b1x8sve", "hamming_b1x128sve", "hamming_b1x128avx512",
};
void* callables[14] = {NULL};

void assign(size_t idx, void* function) { callables[idx] = function; }

typedef simsimd_f32_t (*function_f32_t)(simsimd_f32_t*, simsimd_f32_t*, size_t);
typedef simsimd_f32_t (*function_f32x16_t)(simsimd_f16_t*, simsimd_f16_t*, size_t);
typedef simsimd_f16_t (*function_f16_t)(simsimd_f16_t*, simsimd_f16_t*, size_t);
typedef simsimd_f32_t (*function_fi8_t)(int8_t*, int8_t*, size_t);
typedef int32_t (*function_i8_t)(int8_t*, int8_t*, size_t);
typedef size_t (*function_ui8_t)(uint8_t*, uint8_t*, size_t);


int main(int argc, char* argv[]) {
    Py_Initialize();
    PyObject* module = PyImport_ImportModule("simsimd");
    PyErr_Print();
    if (module == NULL)
        return -1;

    PyObject* dictionary = PyModule_GetDict(module);
    PyObject *key, *value;
    Py_ssize_t pos = 0;

    size_t idx = 0;
    while (PyDict_Next(dictionary, &pos, &key, &value)) {
        if (PyCapsule_CheckExact(value)) {
            char const* name = PyCapsule_GetName(value);
            void* pointer = PyCapsule_GetPointer(value, NULL);
            assign(idx, pointer);
            ++idx;
        }
    }

    size_t d = 3;
    {
        simsimd_f32_t a[] = {1, 2, 3};
        simsimd_f32_t b[] = {4, 5, 6};
        for (idx = 0; idx < 6; ++idx) {
            function_f32_t function = (function_f32_t)callables[idx];
            simsimd_f32_t result = function(a, b, d);
            printf("%f\n", result);
        }
    }

    {
        simsimd_f16_t a[] = {1, 2, 3};
        simsimd_f16_t b[] = {4, 5, 6};
        function_f32x16_t function = (function_f32x16_t)callables[6];
        simsimd_f32_t result = function(a, b, d);
        printf("%f\n", result);
        function = (function_f32x16_t)callables[7];
        result = function(a, b, d);
        printf("%f\n", result);
    }

    {
        simsimd_f16_t a[] = {1, 2, 3};
        simsimd_f16_t b[] = {4, 5, 6};
        function_f16_t function = (function_f16_t)callables[8];
        simsimd_f16_t result = function(a, b, d);
        printf("%i\n", result);
    }

    {
        int8_t a[] = {1, 2, 3};
        int8_t b[] = {4, 5, 6};
        function_fi8_t function = (function_fi8_t)callables[9];
        simsimd_f32_t result = function(a, b, d);
        printf("%f\n", result);
    }

    {
        int8_t a[] = {1, 2, 3};
        int8_t b[] = {4, 5, 6};
        function_i8_t function = (function_i8_t)callables[10];
        int32_t result = function(a, b, d);
        printf("%i\n", result);
    }

    {
        uint8_t a[] = {1, 2, 3};
        uint8_t b[] = {4, 5, 6};
        for (idx = 11; idx < 14; ++idx) {
            function_ui8_t function = (function_ui8_t)callables[idx];
            size_t result = function(a, b, d);
            printf("%zu\n", result);
        }
    }
    
    Py_DECREF(module);
    return 0;
}

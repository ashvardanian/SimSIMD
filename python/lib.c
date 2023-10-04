/**
 *  @file python.c
 *  @author Ash Vardanian
 *  @date 2023-01-30
 *  @copyright Copyright (c) 2023
 *
 *  @brief Pure CPython bindings for SimSIMD.
 */
#define SIMSIMD_TARGET_ARM_NEON 1
#define SIMSIMD_TARGET_X86_AVX2 1
#if __linux__
#define SIMSIMD_TARGET_ARM_SVE 1
#define SIMSIMD_TARGET_X86_AVX512 1
#endif

#define SIMSIMD_SQRT simsimd_approximate_inverse_square_root
#include "simsimd/simsimd.h"

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>

typedef struct parsed_vector_or_matrix_t {
    char* start;
    size_t dimensions;
    size_t count;
    size_t stride;
    int is_flat;
    simsimd_datatype_t datatype;
} parsed_vector_or_matrix_t;

int same_string(char const* a, char const* b) { return strcmp(a, b) == 0; }

simsimd_datatype_t numpy_string_to_datatype(char const* name) {
    // https://docs.python.org/3/library/struct.html#format-characters
    if (same_string(name, "f") || same_string(name, "<f") || same_string(name, "f4") || same_string(name, "<f4"))
        return simsimd_datatype_f32_k;
    else if (same_string(name, "e") || same_string(name, "<e") || same_string(name, "f2") || same_string(name, "<f2"))
        return simsimd_datatype_f16_k;
    else if (same_string(name, "b") || same_string(name, "<b") || same_string(name, "i1") || same_string(name, "|i1"))
        return simsimd_datatype_i8_k;
    else if (same_string(name, "B") || same_string(name, "<B") || same_string(name, "u1") || same_string(name, "|u1"))
        return simsimd_datatype_b1_k;
    else if (same_string(name, "d") || same_string(name, "<d") || same_string(name, "i8") || same_string(name, "<i8"))
        return simsimd_datatype_f64_k;
    else
        return simsimd_datatype_unknown_k;
}

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

int parse_tensor(PyObject* tensor, Py_buffer* buffer, parsed_vector_or_matrix_t* parsed) {
    if (PyObject_GetBuffer(tensor, buffer, PyBUF_STRIDES | PyBUF_FORMAT) != 0) {
        PyErr_SetString(PyExc_TypeError, "arguments must support buffer protocol");
        return -1;
    }
    parsed->start = buffer->buf;
    parsed->datatype = numpy_string_to_datatype(buffer->format);
    if (buffer->ndim == 1) {
        if (buffer->strides[0] > buffer->itemsize) {
            PyErr_SetString(PyExc_ValueError, "input vectors must be contiguous");
            PyBuffer_Release(buffer);
            return -1;
        }
        parsed->is_flat = 1;
        parsed->dimensions = buffer->shape[0];
        parsed->count = 1;
        parsed->stride = 0;
    } else if (buffer->ndim == 2) {
        if (buffer->strides[1] > buffer->itemsize) {
            PyErr_SetString(PyExc_ValueError, "input vectors must be contiguous");
            PyBuffer_Release(buffer);
            return -1;
        }
        parsed->is_flat = 0;
        parsed->dimensions = buffer->shape[1];
        parsed->count = buffer->shape[0];
        parsed->stride = buffer->strides[0];
    } else {
        PyErr_SetString(PyExc_ValueError, "input tensors must be 1D or 2D");
        PyBuffer_Release(buffer);
        return -1;
    }
    return 0;
}

static PyObject* api_pairs(simsimd_metric_kind_t metric_kind, PyObject* args) {
    if (!PyTuple_Check(args) || PyTuple_Size(args) != 2) {
        PyErr_SetString(PyExc_TypeError, "function expects exactly 2 arguments");
        return NULL;
    }

    PyObject* output = NULL;
    PyObject* input_tensor_a = PyTuple_GetItem(args, 0);
    PyObject* input_tensor_b = PyTuple_GetItem(args, 1);

    Py_buffer buffer_a, buffer_b;
    parsed_vector_or_matrix_t parsed_a, parsed_b;
    if (parse_tensor(input_tensor_a, &buffer_a, &parsed_a) != 0 ||
        parse_tensor(input_tensor_b, &buffer_b, &parsed_b) != 0) {
        return NULL; // Error already set by parse_tensor
    }

    // Check dimensions
    if (parsed_a.dimensions != parsed_b.dimensions) {
        PyErr_SetString(PyExc_ValueError, "vector dimensions don't match");
        goto cleanup;
    }
    if (parsed_a.count == 0 || parsed_b.count == 0) {
        PyErr_SetString(PyExc_ValueError, "collections can't be empty");
        goto cleanup;
    }
    if (parsed_a.count > 1 && parsed_b.count > 1 && parsed_a.count != parsed_b.count) {
        PyErr_SetString(PyExc_ValueError, "collections must have the same number of elements or just one element");
        goto cleanup;
    }

    // Check data types
    if (parsed_a.datatype != parsed_b.datatype && parsed_a.datatype != simsimd_datatype_unknown_k &&
        parsed_b.datatype != simsimd_datatype_unknown_k) {
        PyErr_SetString(PyExc_ValueError, "input tensors must have matching and supported datatypes");
        goto cleanup;
    }

    simsimd_metric_punned_t metric = simsimd_metric_punned(metric_kind, parsed_a.datatype, 0xFFFFFFFF);

    // If the distance is computed between two vectors, rather than matrices, return a scalar
    if (parsed_a.is_flat && parsed_b.is_flat) {
        output = PyFloat_FromDouble(metric(parsed_a.start, parsed_b.start, parsed_a.dimensions, parsed_b.dimensions));
    } else {
        size_t count_max = parsed_a.count > parsed_b.count ? parsed_a.count : parsed_b.count;

        // Compute the distances
        simsimd_f32_t* distances = malloc(count_max * sizeof(simsimd_f32_t));
        for (size_t i = 0; i < count_max; ++i)
            distances[i] = metric(                    //
                parsed_a.start + i * parsed_a.stride, //
                parsed_b.start + i * parsed_b.stride, //
                parsed_a.dimensions,                  //
                parsed_b.dimensions);

        // Create a new PyArray object for the output
        npy_intp dims[1] = {count_max};
        PyArray_Descr* descr = PyArray_DescrFromType(NPY_FLOAT32);
        PyArrayObject* output_array = (PyArrayObject*)PyArray_NewFromDescr( //
            &PyArray_Type, descr, 1, dims, NULL, distances, NPY_ARRAY_OWNDATA, NULL);

        if (!output_array) {
            free(distances);
            goto cleanup;
        }

        output = output_array;
    }

cleanup:
    PyBuffer_Release(&buffer_a);
    PyBuffer_Release(&buffer_b);
    return output;
}

static PyObject* api_combos(simsimd_metric_punned_t metric, PyObject* args) {}

static PyObject* api_pairs_l2sq(PyObject* self, PyObject* args) { return api_pairs(simsimd_metric_l2sq_k, args); }
static PyObject* api_pairs_cos(PyObject* self, PyObject* args) { return api_pairs(simsimd_metric_cos_k, args); }
static PyObject* api_pairs_ip(PyObject* self, PyObject* args) { return api_pairs(simsimd_metric_ip_k, args); }

static PyMethodDef simsimd_methods[] = {
    {"to_int", to_int, METH_VARARGS, "Converts CPython capsule to `int`"},
    {"sqeuclidean", api_pairs_l2sq, METH_VARARGS, "L2sq (Squared Euclidean) distances between a pair of tensors"},
    {"cosine", api_pairs_cos, METH_VARARGS, "Cosine (Angular) distances between a pair of tensors"},
    {"dot", api_pairs_ip, METH_VARARGS, "Inner (Dot) Product distances between a pair of tensors"},
    {NULL, NULL, 0, NULL} /* Sentinel */
};

static PyModuleDef simsimd_module = {
    PyModuleDef_HEAD_INIT,
    .m_name = "SimSIMD",
    .m_doc = "SIMD-accelerated similarity measures for x86 and Arm: AVX2, AVX512, NEON, SVE",
    .m_size = -1,
    .m_methods = simsimd_methods,
};

PyMODINIT_FUNC PyInit_simsimd(void) { return PyModule_Create(&simsimd_module); }

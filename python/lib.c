/**
 *  @brief      Pure CPython bindings for SimSIMD.
 *  @file       lib.c
 *  @author     Ash Vardanian
 *  @date       January 1, 2023
 *  @copyright  Copyright (c) 2023
 */
#include <math.h>

#if __linux__
#define SIMSIMD_TARGET_ARM_NEON 1
#define SIMSIMD_TARGET_ARM_SVE 1
#define SIMSIMD_TARGET_X86_AVX2 1
#define SIMSIMD_TARGET_X86_AVX512 1
#include <omp.h>
#elif defined(_MSC_VER)
#define SIMSIMD_TARGET_ARM_NEON 0
#define SIMSIMD_TARGET_ARM_SVE 0
#define SIMSIMD_TARGET_X86_AVX2 0
#define SIMSIMD_TARGET_X86_AVX512 0
#elif defined(__APPLE__)
#define SIMSIMD_TARGET_ARM_NEON 1
#define SIMSIMD_TARGET_ARM_SVE 0
#define SIMSIMD_TARGET_X86_AVX2 1
#define SIMSIMD_TARGET_X86_AVX512 0
#endif

#define SIMSIMD_RSQRT(x) (1 / sqrtf(x))
#define SIMSIMD_LOG(x) (logf(x))
#include <simsimd/simsimd.h>

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

/// @brief  Global variable that caches the CPU capabilities, and is computed just onc, when the module is loaded.
simsimd_capability_t static_capabilities = simsimd_cap_serial_k;

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
        return simsimd_datatype_b8_k;
    else if (same_string(name, "d") || same_string(name, "<d") || same_string(name, "i8") || same_string(name, "<i8"))
        return simsimd_datatype_f64_k;
    else
        return simsimd_datatype_unknown_k;
}

simsimd_datatype_t python_string_to_datatype(char const* name) {
    if (same_string(name, "f") || same_string(name, "f32"))
        return simsimd_datatype_f32_k;
    else if (same_string(name, "h") || same_string(name, "f16"))
        return simsimd_datatype_f16_k;
    else if (same_string(name, "c") || same_string(name, "i8"))
        return simsimd_datatype_i8_k;
    else if (same_string(name, "b") || same_string(name, "b8"))
        return simsimd_datatype_b8_k;
    else if (same_string(name, "d") || same_string(name, "f64"))
        return simsimd_datatype_f64_k;
    else
        return simsimd_datatype_unknown_k;
}

simsimd_metric_kind_t python_string_to_metric_kind(char const* name) {
    if (same_string(name, "sqeuclidean"))
        return simsimd_metric_sqeuclidean_k;
    else if (same_string(name, "inner"))
        return simsimd_metric_inner_k;
    else if (same_string(name, "cosine"))
        return simsimd_metric_cosine_k;
    else if (same_string(name, "hamming"))
        return simsimd_metric_hamming_k;
    else if (same_string(name, "jaccard"))
        return simsimd_metric_jaccard_k;
    else if (same_string(name, "kullbackleibler") || same_string(name, "kl"))
        return simsimd_metric_kl_k;
    else if (same_string(name, "jensenshannon") || same_string(name, "js"))
        return simsimd_metric_js_k;
    else if (same_string(name, "jaccard"))
        return simsimd_metric_jaccard_k;
    else
        return simsimd_metric_unknown_k;
}

static PyObject* api_get_capabilities(PyObject* self) {
    simsimd_capability_t caps = static_capabilities;
    PyObject* cap_dict = PyDict_New();
    if (!cap_dict)
        return NULL;

#define ADD_CAP(name) PyDict_SetItemString(cap_dict, #name, PyBool_FromLong(caps& simsimd_cap_##name##_k))

    ADD_CAP(serial);
    ADD_CAP(arm_neon);
    ADD_CAP(arm_sve);
    ADD_CAP(arm_sve2);
    ADD_CAP(x86_avx2);
    ADD_CAP(x86_avx512);
    ADD_CAP(x86_avx2fp16);
    ADD_CAP(x86_avx512fp16);
    ADD_CAP(x86_avx512vpopcntdq);
    ADD_CAP(x86_avx512vnni);

#undef ADD_CAP

    return cap_dict;
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

void free_capsule(void *capsule) {
    void* obj = PyCapsule_GetPointer(capsule, PyCapsule_GetName(capsule));
    free(obj);
};

static PyObject* impl_metric(simsimd_metric_kind_t metric_kind, PyObject* const* args, Py_ssize_t nargs) {
    if (nargs != 2) {
        PyErr_SetString(PyExc_TypeError, "function expects exactly 2 arguments");
        return NULL;
    }

    PyObject* output = NULL;
    PyObject* input_tensor_a = args[0];
    PyObject* input_tensor_b = args[1];
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

    simsimd_metric_punned_t metric = NULL;
    simsimd_capability_t capability = simsimd_cap_serial_k;
    simsimd_datatype_t datatype = parsed_a.datatype;
    simsimd_find_metric_punned(metric_kind, datatype, static_capabilities, simsimd_cap_any_k, &metric, &capability);
    if (!metric) {
        PyErr_SetString(PyExc_ValueError, "unsupported metric and datatype combination");
        goto cleanup;
    }

    // If the distance is computed between two vectors, rather than matrices, return a scalar
    if (parsed_a.is_flat && parsed_b.is_flat) {
        output = PyFloat_FromDouble(metric(parsed_a.start, parsed_b.start, parsed_a.dimensions, parsed_b.dimensions));
    } else {

        // In some batch requests we may be computing the distance from multiple vectors to one,
        // so the stride must be set to zero avoid illegal memory access
        if (parsed_a.count == 1)
            parsed_a.stride = 0;
        if (parsed_b.count == 1)
            parsed_b.stride = 0;

        size_t count_max = parsed_a.count > parsed_b.count ? parsed_a.count : parsed_b.count;

        // Compute the distances
        float* distances = malloc(count_max * sizeof(float));
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
            &PyArray_Type, descr, 1, dims, NULL, distances, NPY_ARRAY_WRITEABLE, NULL);

        if (!output_array) {
            free(distances);
            goto cleanup;
        }

        PyObject *wrapper = PyCapsule_New(distances, "wrapper", (PyCapsule_Destructor)&free_capsule);
        if (!wrapper) {
            free(distances);
            Py_DECREF(output_array);
            goto cleanup;
        }

        if (PyArray_SetBaseObject((PyArrayObject *)output_array, wrapper) < 0) {
            free(distances);
            Py_DECREF(output_array);
            Py_DECREF(wrapper);
            goto cleanup;
        }

        output = output_array;
    }

cleanup:
    PyBuffer_Release(&buffer_a);
    PyBuffer_Release(&buffer_b);
    return output;
}

static PyObject* impl_cdist(                            //
    PyObject* input_tensor_a, PyObject* input_tensor_b, //
    simsimd_metric_kind_t metric_kind, size_t threads) {

    PyObject* output = NULL;
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

    // Check data types
    if (parsed_a.datatype != parsed_b.datatype && parsed_a.datatype != simsimd_datatype_unknown_k &&
        parsed_b.datatype != simsimd_datatype_unknown_k) {
        PyErr_SetString(PyExc_ValueError, "input tensors must have matching and supported datatypes");
        goto cleanup;
    }

    simsimd_metric_punned_t metric = NULL;
    simsimd_capability_t capability = simsimd_cap_serial_k;
    simsimd_datatype_t datatype = parsed_a.datatype;
    simsimd_find_metric_punned(metric_kind, datatype, static_capabilities, simsimd_cap_any_k, &metric, &capability);
    if (!metric) {
        PyErr_SetString(PyExc_ValueError, "unsupported metric and datatype combination");
        goto cleanup;
    }

    // If the distance is computed between two vectors, rather than matrices, return a scalar
    if (parsed_a.is_flat && parsed_b.is_flat) {
        output = PyFloat_FromDouble(metric(parsed_a.start, parsed_b.start, parsed_a.dimensions, parsed_b.dimensions));
    } else {

#ifdef __linux__
#ifdef _OPENMP
        if (threads == 0)
            threads = omp_get_num_procs();
        omp_set_num_threads(threads);
#endif
#endif
        // Compute the distances
        float* distances = malloc(parsed_a.count * parsed_b.count * sizeof(float));
#pragma omp parallel for collapse(2)
        for (size_t i = 0; i < parsed_a.count; ++i)
            for (size_t j = 0; j < parsed_b.count; ++j)
                distances[i * parsed_b.count + j] = metric( //
                    parsed_a.start + i * parsed_a.stride,   //
                    parsed_b.start + j * parsed_b.stride,   //
                    parsed_a.dimensions,                    //
                    parsed_b.dimensions);

        // Create a new PyArray object for the output
        npy_intp dims[2] = {parsed_a.count, parsed_b.count};
        PyArray_Descr* descr = PyArray_DescrFromType(NPY_FLOAT32);
        PyArrayObject* output_array = (PyArrayObject*)PyArray_NewFromDescr( //
            &PyArray_Type, descr, 2, dims, NULL, distances, NPY_ARRAY_WRITEABLE, NULL);

        if (!output_array) {
            free(distances);
            goto cleanup;
        }

        PyObject *wrapper = PyCapsule_New(distances, "wrapper", (PyCapsule_Destructor)&free_capsule);
        if (!wrapper) {
            free(distances);
            Py_DECREF(output_array);
            goto cleanup;
        }

        if (PyArray_SetBaseObject((PyArrayObject *)output_array, wrapper) < 0) {
            free(distances);
            Py_DECREF(output_array);
            Py_DECREF(wrapper);
            goto cleanup;
        }

        output = output_array;
    }

cleanup:
    PyBuffer_Release(&buffer_a);
    PyBuffer_Release(&buffer_b);
    return output;
}

static PyObject* impl_pointer(simsimd_metric_kind_t metric_kind, PyObject* args) {
    char const* type_name = PyUnicode_AsUTF8(PyTuple_GetItem(args, 0));
    if (!type_name) {
        PyErr_SetString(PyExc_ValueError, "Invalid type name");
        return NULL;
    }

    simsimd_datatype_t datatype = python_string_to_datatype(type_name);
    if (!type_name) {
        PyErr_SetString(PyExc_ValueError, "Unsupported type");
        return NULL;
    }

    simsimd_metric_punned_t metric = NULL;
    simsimd_capability_t capability = simsimd_cap_serial_k;
    simsimd_find_metric_punned(metric_kind, datatype, static_capabilities, simsimd_cap_any_k, &metric, &capability);
    if (metric == NULL) {
        PyErr_SetString(PyExc_ValueError, "No such metric");
        return NULL;
    }

    return PyLong_FromUnsignedLongLong((unsigned long long)metric);
}

static PyObject* api_cdist(PyObject* self, PyObject* args, PyObject* kwargs) {
    PyObject *input_tensor_a, *input_tensor_b;
    PyObject* metric_obj = NULL;
    PyObject* threads_obj = NULL;

    if (!PyTuple_Check(args) || PyTuple_Size(args) < 2) {
        PyErr_SetString(PyExc_TypeError, "function expects at least 2 positional arguments");
        return NULL;
    }

    input_tensor_a = PyTuple_GetItem(args, 0);
    input_tensor_b = PyTuple_GetItem(args, 1);
    if (PyTuple_Size(args) > 2)
        metric_obj = PyTuple_GetItem(args, 2);
    if (PyTuple_Size(args) > 3)
        threads_obj = PyTuple_GetItem(args, 3);

    // Checking for named arguments in kwargs
    if (kwargs) {
        if (!metric_obj) {
            metric_obj = PyDict_GetItemString(kwargs, "metric");
        } else if (PyDict_GetItemString(kwargs, "metric")) {
            PyErr_SetString(PyExc_TypeError, "Duplicate argument for 'metric'");
            return NULL;
        }

        if (!threads_obj) {
            threads_obj = PyDict_GetItemString(kwargs, "threads");
        } else if (PyDict_GetItemString(kwargs, "threads")) {
            PyErr_SetString(PyExc_TypeError, "Duplicate argument for 'threads'");
            return NULL;
        }
    }

    // Process the PyObject values
    simsimd_metric_kind_t metric_kind = simsimd_metric_l2sq_k;
    if (metric_obj) {
        char const* metric_str = PyUnicode_AsUTF8(metric_obj);
        if (!metric_str && PyErr_Occurred()) {
            PyErr_SetString(PyExc_TypeError, "Expected 'metric' to be a string");
            return NULL;
        }
        metric_kind = python_string_to_metric_kind(metric_str);
        if (metric_kind == simsimd_metric_unknown_k) {
            PyErr_SetString(PyExc_ValueError, "Unsupported metric");
            return NULL;
        }
    }

    size_t threads = 1;
    if (threads_obj)
        threads = PyLong_AsSize_t(threads_obj);
    if (PyErr_Occurred()) {
        PyErr_SetString(PyExc_TypeError, "Expected 'threads' to be an unsigned integer");
        return NULL;
    }

    return impl_cdist(input_tensor_a, input_tensor_b, metric_kind, threads);
}

static PyObject* api_l2sq_pointer(PyObject* self, PyObject* args) { return impl_pointer(simsimd_metric_l2sq_k, args); }
static PyObject* api_cos_pointer(PyObject* self, PyObject* args) { return impl_pointer(simsimd_metric_cos_k, args); }
static PyObject* api_ip_pointer(PyObject* self, PyObject* args) { return impl_pointer(simsimd_metric_ip_k, args); }
static PyObject* api_kl_pointer(PyObject* self, PyObject* args) { return impl_pointer(simsimd_metric_kl_k, args); }
static PyObject* api_js_pointer(PyObject* self, PyObject* args) { return impl_pointer(simsimd_metric_js_k, args); }
static PyObject* api_hamming_pointer(PyObject* self, PyObject* args) {
    return impl_pointer(simsimd_metric_hamming_k, args);
}
static PyObject* api_jaccard_pointer(PyObject* self, PyObject* args) {
    return impl_pointer(simsimd_metric_jaccard_k, args);
}

static PyObject* api_l2sq(PyObject* self, PyObject* const* args, Py_ssize_t nargs) {
    return impl_metric(simsimd_metric_l2sq_k, args, nargs);
}
static PyObject* api_cos(PyObject* self, PyObject* const* args, Py_ssize_t nargs) {
    return impl_metric(simsimd_metric_cos_k, args, nargs);
}
static PyObject* api_ip(PyObject* self, PyObject* const* args, Py_ssize_t nargs) {
    return impl_metric(simsimd_metric_ip_k, args, nargs);
}
static PyObject* api_kl(PyObject* self, PyObject* const* args, Py_ssize_t nargs) {
    return impl_metric(simsimd_metric_kl_k, args, nargs);
}
static PyObject* api_js(PyObject* self, PyObject* const* args, Py_ssize_t nargs) {
    return impl_metric(simsimd_metric_js_k, args, nargs);
}
static PyObject* api_hamming(PyObject* self, PyObject* const* args, Py_ssize_t nargs) {
    return impl_metric(simsimd_metric_hamming_k, args, nargs);
}
static PyObject* api_jaccard(PyObject* self, PyObject* const* args, Py_ssize_t nargs) {
    return impl_metric(simsimd_metric_jaccard_k, args, nargs);
}

static PyMethodDef simsimd_methods[] = {
    // Introspecting library and hardware capabilities
    {"get_capabilities", api_get_capabilities, METH_NOARGS, "Get hardware capabilities"},

    // NumPy and SciPy compatible interfaces (two matrix or vector arguments)
    {"sqeuclidean", api_l2sq, METH_FASTCALL, "L2sq (Sq. Euclidean) distances between a pair of matrices"},
    {"cosine", api_cos, METH_FASTCALL, "Cosine (Angular) distances between a pair of matrices"},
    {"inner", api_ip, METH_FASTCALL, "Inner (Dot) Product distances between a pair of matrices"},
    {"hamming", api_hamming, METH_FASTCALL, "Hamming distances between a pair of matrices"},
    {"jaccard", api_jaccard, METH_FASTCALL, "Jaccard (Bitwise Tanimoto) distances between a pair of matrices"},
    {"kullbackleibler", api_kl, METH_FASTCALL, "Kullback-Leibler divergence between probability distributions"},
    {"jensenshannon", api_js, METH_FASTCALL, "Jensen-Shannon divergence between probability distributions"},

    // Conventional `cdist` and `pdist` insterfaces with third string argument, and optional `threads` arg
    {"cdist", api_cdist, METH_VARARGS | METH_KEYWORDS,
     "Compute distance between each pair of the two collections of inputs"},

    // Exposing underlying API for USearch
    {"pointer_to_sqeuclidean", api_l2sq_pointer, METH_VARARGS, "L2sq (Sq. Euclidean) function pointer as `int`"},
    {"pointer_to_cosine", api_cos_pointer, METH_VARARGS, "Cosine (Angular) function pointer as `int`"},
    {"pointer_to_inner", api_ip_pointer, METH_VARARGS, "Inner (Dot) Product function pointer as `int`"},
    {"pointer_to_kullbackleibler", api_ip_pointer, METH_VARARGS, "Kullback-Leibler function pointer as `int`"},
    {"pointer_to_jensenshannon", api_ip_pointer, METH_VARARGS, "Jensen-Shannon function pointer as `int`"},

    // Sentinel
    {NULL, NULL, 0, NULL}};

static PyModuleDef simsimd_module = {
    PyModuleDef_HEAD_INIT,
    .m_name = "SimSIMD",
    .m_doc = "Fastest SIMD-Accelerated Vector Similarity Functions for x86 and Arm",
    .m_size = -1,
    .m_methods = simsimd_methods,
};

PyMODINIT_FUNC PyInit_simsimd(void) {
    import_array();
    PyObject* module = PyModule_Create(&simsimd_module);

    if (module) {
        char version_str[50];
        sprintf(version_str, "%d.%d.%d", SIMSIMD_VERSION_MAJOR, SIMSIMD_VERSION_MINOR, SIMSIMD_VERSION_PATCH);
        PyModule_AddStringConstant(module, "__version__", version_str);
    }

    static_capabilities = simsimd_capabilities();
    return module;
}

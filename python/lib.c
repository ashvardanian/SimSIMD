/**
 *  @brief      Pure CPython bindings for SimSIMD.
 *  @file       lib.c
 *  @author     Ash Vardanian
 *  @date       January 1, 2023
 *  @copyright  Copyright (c) 2023
 */
#include <math.h>

#if defined(__linux__)
#include <omp.h>
#endif

#define SIMSIMD_RSQRT(x) (1 / sqrtf(x))
#define SIMSIMD_LOG(x) (logf(x))
#include <simsimd/simsimd.h>

#define PY_SSIZE_T_CLEAN
#include <Python.h>

typedef struct TensorArgument {
    char* start;
    size_t dimensions;
    size_t count;
    size_t stride;
    int is_flat;
    simsimd_datatype_t datatype;
} TensorArgument;

typedef struct DistancesTensor {
    PyObject_HEAD                    //
        simsimd_datatype_t datatype; // Double precision real or complex numbers
    size_t dimensions;               // Can be only 1 or 2 dimensions
    Py_ssize_t shape[2];             // Dimensions of the tensor
    Py_ssize_t strides[2];           // Strides for each dimension
    simsimd_distance_t start[];      // Variable length data aligned to 64-bit scalars
} DistancesTensor;

static int DistancesTensor_getbuffer(PyObject* export_from, Py_buffer* view, int flags);
static void DistancesTensor_releasebuffer(PyObject* export_from, Py_buffer* view);

static PyBufferProcs DistancesTensor_as_buffer = {
    .bf_getbuffer = DistancesTensor_getbuffer,
    .bf_releasebuffer = DistancesTensor_releasebuffer,
};

static PyTypeObject DistancesTensorType = {
    PyObject_HEAD_INIT(NULL).tp_name = "simsimd.DistancesTensor",
    .tp_doc = "Zero-copy view of an internal tensor, compatible with NumPy",
    .tp_basicsize = sizeof(DistancesTensor),
    .tp_itemsize = sizeof(simsimd_distance_t),
    .tp_as_buffer = &DistancesTensor_as_buffer,
};

/// @brief  Global variable that caches the CPU capabilities, and is computed just onc, when the module is loaded.
simsimd_capability_t static_capabilities = simsimd_cap_serial_k;

int same_string(char const* a, char const* b) { return strcmp(a, b) == 0; }

int is_complex(simsimd_datatype_t datatype) {
    return datatype == simsimd_datatype_f32c_k || datatype == simsimd_datatype_f64c_k ||
           datatype == simsimd_datatype_f16c_k;
}

simsimd_datatype_t numpy_string_to_datatype(char const* name) {
    // https://docs.python.org/3/library/struct.html#format-characters
    if (same_string(name, "f") || same_string(name, "<f") || same_string(name, "f4") || same_string(name, "<f4") ||
        same_string(name, "float32"))
        return simsimd_datatype_f32_k;
    else if (same_string(name, "e") || same_string(name, "<e") || same_string(name, "f2") || same_string(name, "<f2") ||
             same_string(name, "float16"))
        return simsimd_datatype_f16_k;
    else if (same_string(name, "b") || same_string(name, "<b") || same_string(name, "i1") || same_string(name, "|i1") ||
             same_string(name, "int8"))
        return simsimd_datatype_i8_k;
    else if (same_string(name, "B") || same_string(name, "<B") || same_string(name, "u1") || same_string(name, "|u1"))
        return simsimd_datatype_b8_k;
    else if (same_string(name, "d") || same_string(name, "<d") || same_string(name, "f8") || same_string(name, "<f8") ||
             same_string(name, "float64"))
        return simsimd_datatype_f64_k;
    // Complex numbers:
    else if (same_string(name, "Zf") || same_string(name, "F") || same_string(name, "<F") || same_string(name, "F4") ||
             same_string(name, "<F4") || same_string(name, "complex64"))
        return simsimd_datatype_f32c_k;
    else if (same_string(name, "Zd") || same_string(name, "D") || same_string(name, "<D") || same_string(name, "F8") ||
             same_string(name, "<F8") || same_string(name, "complex128"))
        return simsimd_datatype_f64c_k;
    else if (same_string(name, "Ze") || same_string(name, "E") || same_string(name, "<E") || same_string(name, "F2") ||
             same_string(name, "<F2") || same_string(name, "complex32"))
        return simsimd_datatype_f16c_k;
    else
        return simsimd_datatype_unknown_k;
}

simsimd_datatype_t python_string_to_datatype(char const* name) {
    if (same_string(name, "f") || same_string(name, "f32") || same_string(name, "float32"))
        return simsimd_datatype_f32_k;
    else if (same_string(name, "h") || same_string(name, "f16") || same_string(name, "float16"))
        return simsimd_datatype_f16_k;
    else if (same_string(name, "c") || same_string(name, "i8") || same_string(name, "int8"))
        return simsimd_datatype_i8_k;
    else if (same_string(name, "b") || same_string(name, "b8"))
        return simsimd_datatype_b8_k;
    else if (same_string(name, "d") || same_string(name, "f64") || same_string(name, "float64"))
        return simsimd_datatype_f64_k;
    // Complex numbers:
    else if (same_string(name, "complex64"))
        return simsimd_datatype_f32c_k;
    else if (same_string(name, "complex128"))
        return simsimd_datatype_f64c_k;
    else if (same_string(name, "complex32"))
        return simsimd_datatype_f16c_k;
    else
        return simsimd_datatype_unknown_k;
}

char const* datatype_to_python_string(simsimd_datatype_t dtype) {
    switch (dtype) {
    case simsimd_datatype_f64_k: return "d";
    case simsimd_datatype_f32_k: return "f";
    case simsimd_datatype_f16_k: return "h";
    case simsimd_datatype_f64c_k: return "Zd";
    case simsimd_datatype_f32c_k: return "Zf";
    case simsimd_datatype_f16c_k: return "Zh";
    case simsimd_datatype_i8_k: return "c";
    case simsimd_datatype_b8_k: return "b";
    default: return "unknown";
    }
}

static size_t bytes_per_datatype(simsimd_datatype_t dtype) {
    switch (dtype) {
    case simsimd_datatype_f64_k: return sizeof(simsimd_f64_t);
    case simsimd_datatype_f32_k: return sizeof(simsimd_f32_t);
    case simsimd_datatype_f16_k: return sizeof(simsimd_f16_t);
    case simsimd_datatype_f64c_k: return sizeof(simsimd_f64_t) * 2;
    case simsimd_datatype_f32c_k: return sizeof(simsimd_f32_t) * 2;
    case simsimd_datatype_f16c_k: return sizeof(simsimd_f16_t) * 2;
    case simsimd_datatype_i8_k: return sizeof(simsimd_i8_t);
    case simsimd_datatype_b8_k: return sizeof(simsimd_b8_t);
    default: return 0;
    }
}

simsimd_metric_kind_t python_string_to_metric_kind(char const* name) {
    if (same_string(name, "sqeuclidean"))
        return simsimd_metric_sqeuclidean_k;
    else if (same_string(name, "inner") || same_string(name, "dot"))
        return simsimd_metric_inner_k;
    else if (same_string(name, "cosine") || same_string(name, "cos"))
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

static PyObject* api_enable_capability(PyObject* self, PyObject* args) {
    char const* cap_name;
    if (!PyArg_ParseTuple(args, "s", &cap_name)) {
        return NULL; // Argument parsing failed
    }

    if (same_string(cap_name, "neon")) {
        static_capabilities |= simsimd_cap_neon_k;
    } else if (same_string(cap_name, "sve")) {
        static_capabilities |= simsimd_cap_sve_k;
    } else if (same_string(cap_name, "sve2")) {
        static_capabilities |= simsimd_cap_sve2_k;
    } else if (same_string(cap_name, "haswell")) {
        static_capabilities |= simsimd_cap_haswell_k;
    } else if (same_string(cap_name, "skylake")) {
        static_capabilities |= simsimd_cap_skylake_k;
    } else if (same_string(cap_name, "ice")) {
        static_capabilities |= simsimd_cap_ice_k;
    } else if (same_string(cap_name, "sapphire")) {
        static_capabilities |= simsimd_cap_sapphire_k;
    } else if (same_string(cap_name, "serial")) {
        PyErr_SetString(PyExc_ValueError, "Can't change the serial functionality");
        return NULL;
    } else {
        PyErr_SetString(PyExc_ValueError, "Unknown capability");
        return NULL;
    }

    Py_RETURN_NONE;
}

static PyObject* api_disable_capability(PyObject* self, PyObject* args) {
    char const* cap_name;
    if (!PyArg_ParseTuple(args, "s", &cap_name)) {
        return NULL; // Argument parsing failed
    }

    if (same_string(cap_name, "neon")) {
        static_capabilities &= ~simsimd_cap_neon_k;
    } else if (same_string(cap_name, "sve")) {
        static_capabilities &= ~simsimd_cap_sve_k;
    } else if (same_string(cap_name, "sve2")) {
        static_capabilities &= ~simsimd_cap_sve2_k;
    } else if (same_string(cap_name, "haswell")) {
        static_capabilities &= ~simsimd_cap_haswell_k;
    } else if (same_string(cap_name, "skylake")) {
        static_capabilities &= ~simsimd_cap_skylake_k;
    } else if (same_string(cap_name, "ice")) {
        static_capabilities &= ~simsimd_cap_ice_k;
    } else if (same_string(cap_name, "sapphire")) {
        static_capabilities &= ~simsimd_cap_sapphire_k;
    } else if (same_string(cap_name, "serial")) {
        PyErr_SetString(PyExc_ValueError, "Can't change the serial functionality");
        return NULL;
    } else {
        PyErr_SetString(PyExc_ValueError, "Unknown capability");
        return NULL;
    }

    Py_RETURN_NONE;
}

static PyObject* api_get_capabilities(PyObject* self) {
    simsimd_capability_t caps = static_capabilities;
    PyObject* cap_dict = PyDict_New();
    if (!cap_dict)
        return NULL;

#define ADD_CAP(name) PyDict_SetItemString(cap_dict, #name, PyBool_FromLong((caps) & simsimd_cap_##name##_k))

    ADD_CAP(serial);
    ADD_CAP(neon);
    ADD_CAP(sve);
    ADD_CAP(sve2);
    ADD_CAP(haswell);
    ADD_CAP(skylake);
    ADD_CAP(ice);
    ADD_CAP(sapphire);

#undef ADD_CAP

    return cap_dict;
}

int parse_tensor(PyObject* tensor, Py_buffer* buffer, TensorArgument* parsed) {
    if (PyObject_GetBuffer(tensor, buffer, PyBUF_STRIDES | PyBUF_FORMAT) != 0) {
        PyErr_SetString(PyExc_TypeError, "arguments must support buffer protocol");
        return -1;
    }
    // In case you are debugging some new obscure format string :)
    // printf("buffer format is %s\n", buffer->format);
    // printf("buffer ndim is %d\n", buffer->ndim);
    // printf("buffer shape is %d\n", buffer->shape[0]);
    // printf("buffer shape is %d\n", buffer->shape[1]);
    // printf("buffer itemsize is %d\n", buffer->itemsize);
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

    // We handle complex numbers differently
    if (is_complex(parsed->datatype)) {
        parsed->dimensions *= 2;
    }

    return 0;
}

static int DistancesTensor_getbuffer(PyObject* export_from, Py_buffer* view, int flags) {
    DistancesTensor* tensor = (DistancesTensor*)export_from;
    size_t const total_items = tensor->shape[0] * tensor->shape[1];
    size_t const item_size = bytes_per_datatype(tensor->datatype);

    view->buf = &tensor->start[0];
    view->obj = (PyObject*)tensor;
    view->len = item_size * total_items;
    view->readonly = 0;
    view->itemsize = (Py_ssize_t)item_size;
    view->format = datatype_to_python_string(tensor->datatype);
    view->ndim = (int)tensor->dimensions;
    view->shape = &tensor->shape[0];
    view->strides = &tensor->strides[0];
    view->suboffsets = NULL;
    view->internal = NULL;

    Py_INCREF(tensor);
    return 0;
}

static void DistancesTensor_releasebuffer(PyObject* export_from, Py_buffer* view) {
    // This function MUST NOT decrement view->obj, since that is done automatically in PyBuffer_Release().
    // https://docs.python.org/3/c-api/typeobj.html#c.PyBufferProcs.bf_releasebuffer
}

static PyObject* impl_metric(simsimd_metric_kind_t metric_kind, PyObject* const* args, Py_ssize_t nargs) {
    // Function now accepts up to 3 arguments, the third being optional
    if (nargs < 2 || nargs > 3) {
        PyErr_SetString(PyExc_TypeError, "function expects 2 or 3 arguments");
        return NULL;
    }

    PyObject* output = NULL;
    PyObject* input_tensor_a = args[0];
    PyObject* input_tensor_b = args[1];
    PyObject* value_type_desc = nargs == 3 ? args[2] : NULL;

    Py_buffer buffer_a, buffer_b;
    TensorArgument parsed_a, parsed_b;
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

    // Process the third argument, value_type_desc, if provided
    simsimd_datatype_t datatype = parsed_a.datatype;
    if (value_type_desc != NULL) {
        // Ensure it is a string (or convert it to one if possible)
        if (!PyUnicode_Check(value_type_desc)) {
            PyErr_SetString(PyExc_TypeError, "third argument must be a string describing the value type");
            goto cleanup;
        }
        // Convert Python string to C string
        char const* value_type_str = PyUnicode_AsUTF8(value_type_desc);
        if (!value_type_str) {
            PyErr_SetString(PyExc_ValueError, "could not convert value type description to string");
            goto cleanup;
        }
        datatype = python_string_to_datatype(value_type_str);
    }

    simsimd_metric_punned_t metric = NULL;
    simsimd_capability_t capability = simsimd_cap_serial_k;
    simsimd_find_metric_punned(metric_kind, datatype, static_capabilities, simsimd_cap_any_k, &metric, &capability);
    if (!metric) {
        PyErr_SetString(PyExc_ValueError, "unsupported metric and datatype combination");
        goto cleanup;
    }

    // If the distance is computed between two vectors, rather than matrices, return a scalar
    int datatype_is_complex = is_complex(datatype);
    if (parsed_a.is_flat && parsed_b.is_flat) {
        // For complex numbers we are going to use `PyComplex_FromDoubles`.
        if (datatype_is_complex) {
            simsimd_distance_t distances[2];
            metric(parsed_a.start, parsed_b.start, parsed_a.dimensions, distances);
            output = PyComplex_FromDoubles(distances[0], distances[1]);
        } else {
            simsimd_distance_t distance;
            metric(parsed_a.start, parsed_b.start, parsed_a.dimensions, &distance);
            output = PyFloat_FromDouble(distance);
        }
    } else {

        // In some batch requests we may be computing the distance from multiple vectors to one,
        // so the stride must be set to zero avoid illegal memory access
        if (parsed_a.count == 1)
            parsed_a.stride = 0;
        if (parsed_b.count == 1)
            parsed_b.stride = 0;

        // We take the maximum of the two counts, because if only one entry is present in one of the arrays,
        // all distances will be computed against that single entry.
        size_t const count_pairs = parsed_a.count > parsed_b.count ? parsed_a.count : parsed_b.count;
        size_t const components_per_pair = datatype_is_complex ? 2 : 1;
        size_t const count_components = count_pairs * components_per_pair;
        DistancesTensor* distances_obj = PyObject_NewVar(DistancesTensor, &DistancesTensorType, count_components);
        if (!distances_obj) {
            PyErr_NoMemory();
            goto cleanup;
        }

        // Initialize the object
        distances_obj->datatype = datatype_is_complex ? simsimd_datatype_f64c_k : simsimd_datatype_f64_k;
        distances_obj->dimensions = 1;
        distances_obj->shape[0] = count_pairs;
        distances_obj->shape[1] = 1;
        distances_obj->strides[0] = bytes_per_datatype(distances_obj->datatype);
        distances_obj->strides[1] = 0;
        output = (PyObject*)distances_obj;

        // Compute the distances
        simsimd_distance_t* distances = (simsimd_distance_t*)&distances_obj->start[0];
        for (size_t i = 0; i < count_pairs; ++i)
            metric(                                   //
                parsed_a.start + i * parsed_a.stride, //
                parsed_b.start + i * parsed_b.stride, //
                parsed_a.dimensions,                  //
                distances + i * components_per_pair);
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
    TensorArgument parsed_a, parsed_b;
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
    int datatype_is_complex = is_complex(datatype);
    if (parsed_a.is_flat && parsed_b.is_flat) {
        // For complex numbers we are going to use `PyComplex_FromDoubles`.
        if (datatype_is_complex) {
            simsimd_distance_t distances[2];
            metric(parsed_a.start, parsed_b.start, parsed_a.dimensions, distances);
            output = PyComplex_FromDoubles(distances[0], distances[1]);
        } else {
            simsimd_distance_t distance;
            metric(parsed_a.start, parsed_b.start, parsed_a.dimensions, &distance);
            output = PyFloat_FromDouble(distance);
        }
    } else {

#ifdef __linux__
#ifdef _OPENMP
        if (threads == 0)
            threads = omp_get_num_procs();
        omp_set_num_threads(threads);
#endif
#endif

        size_t const count_pairs = parsed_a.count * parsed_b.count;
        size_t const components_per_pair = datatype_is_complex ? 2 : 1;
        size_t const count_components = count_pairs * components_per_pair;
        DistancesTensor* distances_obj = PyObject_NewVar(DistancesTensor, &DistancesTensorType, count_components);
        if (!distances_obj) {
            PyErr_NoMemory();
            goto cleanup;
        }

        // Initialize the object
        distances_obj->datatype = datatype_is_complex ? simsimd_datatype_f64c_k : simsimd_datatype_f64_k;
        distances_obj->dimensions = 2;
        distances_obj->shape[0] = parsed_a.count;
        distances_obj->shape[1] = parsed_b.count;
        distances_obj->strides[0] = parsed_b.count * bytes_per_datatype(distances_obj->datatype);
        distances_obj->strides[1] = bytes_per_datatype(distances_obj->datatype);
        output = (PyObject*)distances_obj;

        // Compute the distances
        simsimd_distance_t* distances = (simsimd_distance_t*)&distances_obj->start[0];
#pragma omp parallel for collapse(2)
        for (size_t i = 0; i < parsed_a.count; ++i)
            for (size_t j = 0; j < parsed_b.count; ++j)
                metric(                                   //
                    parsed_a.start + i * parsed_a.stride, //
                    parsed_b.start + j * parsed_b.stride, //
                    parsed_a.dimensions,                  //
                    distances + i * components_per_pair * parsed_b.count + j);
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
static PyObject* api_dot_pointer(PyObject* self, PyObject* args) { return impl_pointer(simsimd_metric_dot_k, args); }
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
static PyObject* api_dot(PyObject* self, PyObject* const* args, Py_ssize_t nargs) {
    return impl_metric(simsimd_metric_dot_k, args, nargs);
}
static PyObject* api_vdot(PyObject* self, PyObject* const* args, Py_ssize_t nargs) {
    return impl_metric(simsimd_metric_vdot_k, args, nargs);
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
    {"enable_capability", api_enable_capability, METH_VARARGS, "Enable a specific family of Assembly kernels"},
    {"disable_capability", api_disable_capability, METH_VARARGS, "Disable a specific family of Assembly kernels"},

    // NumPy and SciPy compatible interfaces (two matrix or vector arguments)
    {"sqeuclidean", api_l2sq, METH_FASTCALL, "L2sq (Sq. Euclidean) distances between a pair of matrices"},
    {"cosine", api_cos, METH_FASTCALL, "Cosine (Angular) distances between a pair of matrices"},
    {"inner", api_dot, METH_FASTCALL, "Inner (Dot) Product distances between a pair of matrices"},
    {"dot", api_dot, METH_FASTCALL, "Inner (Dot) Product distances between a pair of matrices"},
    {"vdot", api_vdot, METH_FASTCALL, "Inner (Dot) Product distances between a pair of matrices"},
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
    {"pointer_to_inner", api_dot_pointer, METH_VARARGS, "Inner (Dot) Product function pointer as `int`"},
    {"pointer_to_kullbackleibler", api_dot_pointer, METH_VARARGS, "Kullback-Leibler function pointer as `int`"},
    {"pointer_to_jensenshannon", api_dot_pointer, METH_VARARGS, "Jensen-Shannon function pointer as `int`"},

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
    PyObject* m;

    if (PyType_Ready(&DistancesTensorType) < 0)
        return NULL;

    m = PyModule_Create(&simsimd_module);
    if (m == NULL)
        return NULL;

    // Add version metadata
    {
        char version_str[50];
        sprintf(version_str, "%d.%d.%d", SIMSIMD_VERSION_MAJOR, SIMSIMD_VERSION_MINOR, SIMSIMD_VERSION_PATCH);
        PyModule_AddStringConstant(m, "__version__", version_str);
    }

    Py_INCREF(&DistancesTensorType);
    if (PyModule_AddObject(m, "DistancesTensor", (PyObject*)&DistancesTensorType) < 0) {
        Py_XDECREF(&DistancesTensorType);
        Py_XDECREF(m);
        return NULL;
    }

    static_capabilities = simsimd_capabilities();
    return m;
}

/**
 *  @brief      Pure CPython bindings for SimSIMD.
 *  @file       lib.c
 *  @author     Ash Vardanian
 *  @date       January 1, 2023
 *  @copyright  Copyright (c) 2023
 */
#include <math.h>

#if defined(__linux__)
#ifdef _OPENMP
#include <omp.h>
#endif
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
    int rank;
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
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "simsimd.DistancesTensor",
    .tp_doc = "Zero-copy view of an internal tensor, compatible with NumPy",
    .tp_basicsize = sizeof(DistancesTensor),
    // Instead of using `simsimd_distance_t` for all the elements,
    // we use `char` to allow user to specify the datatype on `cdist`-like functions.
    .tp_itemsize = sizeof(char),
    .tp_as_buffer = &DistancesTensor_as_buffer,
};

/// @brief  Global variable that caches the CPU capabilities, and is computed just onc, when the module is loaded.
simsimd_capability_t static_capabilities = simsimd_cap_serial_k;

/// @brief Helper method to check for string equality.
/// @return 1 if the strings are equal, 0 otherwise.
int same_string(char const* a, char const* b) { return strcmp(a, b) == 0; }

/// @brief Helper method to check if a logical datatype is complex and should be represented as two scalars.
/// @return 1 if the datatype is complex, 0 otherwise.
int is_complex(simsimd_datatype_t datatype) {
    return datatype == simsimd_datatype_f32c_k || datatype == simsimd_datatype_f64c_k ||
           datatype == simsimd_datatype_f16c_k || datatype == simsimd_datatype_bf16c_k;
}

/// @brief Converts a numpy datatype string to a logical datatype, normalizing the format.
/// @return `simsimd_datatype_unknown_k` if the datatype is not supported, otherwise the logical datatype.
/// @see https://docs.python.org/3/library/struct.html#format-characters
simsimd_datatype_t numpy_string_to_datatype(char const* name) {
    // Floating-point numbers:
    if (same_string(name, "f") || same_string(name, "<f") || same_string(name, "f4") || same_string(name, "<f4") ||
        same_string(name, "float32"))
        return simsimd_datatype_f32_k;
    else if (same_string(name, "e") || same_string(name, "<e") || same_string(name, "f2") || same_string(name, "<f2") ||
             same_string(name, "float16"))
        return simsimd_datatype_f16_k;
    else if (same_string(name, "d") || same_string(name, "<d") || same_string(name, "f8") || same_string(name, "<f8") ||
             same_string(name, "float64"))
        return simsimd_datatype_f64_k;
    else if (same_string(name, "bfloat16")) //? Is it what it's gonna look like?
        return simsimd_datatype_bf16_k;

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
    else if (same_string(name, "bcomplex32")) //? Is it what it's gonna look like?
        return simsimd_datatype_bf16c_k;

    // Boolean values:
    else if (same_string(name, "c") || same_string(name, "b8") || same_string(name, "bits"))
        return simsimd_datatype_b8_k;

    // Signed integers:
    else if (same_string(name, "b") || same_string(name, "<b") || same_string(name, "i1") || same_string(name, "|i1") ||
             same_string(name, "<i1") || same_string(name, "int8"))
        return simsimd_datatype_i8_k;
    else if (same_string(name, "h") || same_string(name, "<h") || same_string(name, "i2") || same_string(name, "|i2") ||
             same_string(name, "<i2") || same_string(name, "int16"))
        return simsimd_datatype_i16_k;
    else if (same_string(name, "i") || same_string(name, "<i") || same_string(name, "i4") || same_string(name, "|i4") ||
             same_string(name, "<i4") || same_string(name, "l") || same_string(name, "<l") ||
             same_string(name, "int32"))
        return simsimd_datatype_i32_k;

    // Unsigned integers:
    else if (same_string(name, "B") || same_string(name, "<B") || same_string(name, "u1") || same_string(name, "|u1") ||
             same_string(name, "<u1") || same_string(name, "uint8"))
        return simsimd_datatype_u8_k;
    else if (same_string(name, "H") || same_string(name, "<H") || same_string(name, "u2") || same_string(name, "|u2") ||
             same_string(name, "<u2") || same_string(name, "uint16"))
        return simsimd_datatype_u16_k;
    else if (same_string(name, "I") || same_string(name, "<I") || same_string(name, "L") || same_string(name, "<L") ||
             same_string(name, "u4") || same_string(name, "|u4") || same_string(name, "<u4") ||
             same_string(name, "uint32"))
        return simsimd_datatype_u32_k;

    else
        return simsimd_datatype_unknown_k;
}

/// @brief Converts a Python string to a logical datatype, normalizing the format.
/// @see https://docs.python.org/3/library/struct.html#format-characters
simsimd_datatype_t python_string_to_datatype(char const* name) {
    // Floating-point numbers:
    if (same_string(name, "f") || same_string(name, "f32") || same_string(name, "float32"))
        return simsimd_datatype_f32_k;
    else if (same_string(name, "e") || same_string(name, "f16") || same_string(name, "float16"))
        return simsimd_datatype_f16_k;
    else if (same_string(name, "d") || same_string(name, "f64") || same_string(name, "float64"))
        return simsimd_datatype_f64_k;
    else if (same_string(name, "bh") || same_string(name, "bf16") || same_string(name, "bfloat16"))
        return simsimd_datatype_bf16_k;

    // Complex numbers:
    else if (same_string(name, "complex64"))
        return simsimd_datatype_f32c_k;
    else if (same_string(name, "complex128"))
        return simsimd_datatype_f64c_k;
    else if (same_string(name, "complex32"))
        return simsimd_datatype_f16c_k;
    else if (same_string(name, "bcomplex32"))
        return simsimd_datatype_bf16c_k;

    // Boolean values:
    else if (same_string(name, "c") || same_string(name, "b8") || same_string(name, "bits"))
        return simsimd_datatype_b8_k;

    // Signed integers:
    else if (same_string(name, "b") || same_string(name, "i8") || same_string(name, "int8"))
        return simsimd_datatype_i8_k;
    else if (same_string(name, "h") || same_string(name, "i16") || same_string(name, "int16"))
        return simsimd_datatype_i16_k;
    else if (same_string(name, "i") || same_string(name, "i32") || same_string(name, "int32") || same_string(name, "l"))
        return simsimd_datatype_i32_k;
    else if (same_string(name, "q") || same_string(name, "i64") || same_string(name, "int64"))
        return simsimd_datatype_i64_k;

    // Unsigned integers:
    else if (same_string(name, "B") || same_string(name, "u8") || same_string(name, "uint8"))
        return simsimd_datatype_u8_k;
    else if (same_string(name, "H") || same_string(name, "u16") || same_string(name, "uint16"))
        return simsimd_datatype_u16_k;
    else if (same_string(name, "I") || same_string(name, "u32") || same_string(name, "uint32") ||
             same_string(name, "L"))
        return simsimd_datatype_u32_k;
    else if (same_string(name, "Q") || same_string(name, "u64") || same_string(name, "uint64"))
        return simsimd_datatype_u64_k;

    else
        return simsimd_datatype_unknown_k;
}

/// @brief Returns the Python string representation of a datatype for the buffer protocol.
/// @param dtype Logical datatype, can be complex.
/// @return "unknown" if the datatype is not supported, otherwise a string.
/// @see https://docs.python.org/3/library/struct.html#format-characters
char const* datatype_to_python_string(simsimd_datatype_t dtype) {
    switch (dtype) {
        // Floating-point numbers:
    case simsimd_datatype_f64_k: return "d";
    case simsimd_datatype_f32_k: return "f";
    case simsimd_datatype_f16_k: return "e";
    // Complex numbers:
    case simsimd_datatype_f64c_k: return "Zd";
    case simsimd_datatype_f32c_k: return "Zf";
    case simsimd_datatype_f16c_k: return "Ze";
    // Boolean values:
    case simsimd_datatype_b8_k: return "c";
    // Signed integers:
    case simsimd_datatype_i8_k: return "b";
    case simsimd_datatype_i16_k: return "h";
    case simsimd_datatype_i32_k: return "i";
    case simsimd_datatype_i64_k: return "q";
    // Unsigned integers:
    case simsimd_datatype_u8_k: return "B";
    case simsimd_datatype_u16_k: return "H";
    case simsimd_datatype_u32_k: return "I";
    case simsimd_datatype_u64_k: return "Q";
    // Other:
    default: return "unknown";
    }
}

/// @brief Estimate the number of bytes per element for a given datatype.
/// @param dtype Logical datatype, can be complex.
/// @return Zero if the datatype is not supported, positive integer otherwise.
size_t bytes_per_datatype(simsimd_datatype_t dtype) {
    switch (dtype) {
    case simsimd_datatype_f64_k: return sizeof(simsimd_f64_t);
    case simsimd_datatype_f32_k: return sizeof(simsimd_f32_t);
    case simsimd_datatype_f16_k: return sizeof(simsimd_f16_t);
    case simsimd_datatype_bf16_k: return sizeof(simsimd_bf16_t);
    case simsimd_datatype_f64c_k: return sizeof(simsimd_f64_t) * 2;
    case simsimd_datatype_f32c_k: return sizeof(simsimd_f32_t) * 2;
    case simsimd_datatype_f16c_k: return sizeof(simsimd_f16_t) * 2;
    case simsimd_datatype_bf16c_k: return sizeof(simsimd_bf16_t) * 2;
    case simsimd_datatype_b8_k: return sizeof(simsimd_b8_t);
    case simsimd_datatype_i8_k: return sizeof(simsimd_i8_t);
    case simsimd_datatype_u8_k: return sizeof(simsimd_u8_t);
    case simsimd_datatype_i16_k: return sizeof(simsimd_i16_t);
    case simsimd_datatype_u16_k: return sizeof(simsimd_u16_t);
    case simsimd_datatype_i32_k: return sizeof(simsimd_i32_t);
    case simsimd_datatype_u32_k: return sizeof(simsimd_u32_t);
    case simsimd_datatype_i64_k: return sizeof(simsimd_i64_t);
    case simsimd_datatype_u64_k: return sizeof(simsimd_u64_t);
    default: return 0;
    }
}

/// @brief Copy a distance to a target datatype, downcasting if necessary.
/// @return 1 if the cast was successful, 0 if the target datatype is not supported.
int cast_distance(simsimd_distance_t distance, simsimd_datatype_t target_dtype, void* target_ptr, size_t offset) {
    switch (target_dtype) {
    case simsimd_datatype_f64_k: ((simsimd_f64_t*)target_ptr)[offset] = (simsimd_f64_t)distance; return 1;
    case simsimd_datatype_f32_k: ((simsimd_f32_t*)target_ptr)[offset] = (simsimd_f32_t)distance; return 1;
    case simsimd_datatype_f16_k: simsimd_f32_to_f16(distance, (simsimd_f16_t*)target_ptr + offset); return 1;
    case simsimd_datatype_bf16_k: simsimd_f32_to_bf16(distance, (simsimd_bf16_t*)target_ptr + offset); return 1;
    case simsimd_datatype_i8_k: ((simsimd_i8_t*)target_ptr)[offset] = (simsimd_i8_t)distance; return 1;
    case simsimd_datatype_u8_k: ((simsimd_u8_t*)target_ptr)[offset] = (simsimd_u8_t)distance; return 1;
    case simsimd_datatype_i16_k: ((simsimd_i16_t*)target_ptr)[offset] = (simsimd_i16_t)distance; return 1;
    case simsimd_datatype_u16_k: ((simsimd_u16_t*)target_ptr)[offset] = (simsimd_u16_t)distance; return 1;
    case simsimd_datatype_i32_k: ((simsimd_i32_t*)target_ptr)[offset] = (simsimd_i32_t)distance; return 1;
    case simsimd_datatype_u32_k: ((simsimd_u32_t*)target_ptr)[offset] = (simsimd_u32_t)distance; return 1;
    case simsimd_datatype_i64_k: ((simsimd_i64_t*)target_ptr)[offset] = (simsimd_i64_t)distance; return 1;
    case simsimd_datatype_u64_k: ((simsimd_u64_t*)target_ptr)[offset] = (simsimd_u64_t)distance; return 1;
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
    else if (same_string(name, "jaccard"))
        return simsimd_metric_jaccard_k;
    else if (same_string(name, "kullbackleibler") || same_string(name, "kl"))
        return simsimd_metric_kl_k;
    else if (same_string(name, "jensenshannon") || same_string(name, "js"))
        return simsimd_metric_js_k;
    else if (same_string(name, "hamming"))
        return simsimd_metric_hamming_k;
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
    } else if (same_string(cap_name, "neon_f16")) {
        static_capabilities |= simsimd_cap_neon_f16_k;
    } else if (same_string(cap_name, "neon_bf16")) {
        static_capabilities |= simsimd_cap_neon_bf16_k;
    } else if (same_string(cap_name, "neon_i8")) {
        static_capabilities |= simsimd_cap_neon_i8_k;
    } else if (same_string(cap_name, "sve")) {
        static_capabilities |= simsimd_cap_sve_k;
    } else if (same_string(cap_name, "sve_f16")) {
        static_capabilities |= simsimd_cap_sve_f16_k;
    } else if (same_string(cap_name, "sve_bf16")) {
        static_capabilities |= simsimd_cap_sve_bf16_k;
    } else if (same_string(cap_name, "sve_i8")) {
        static_capabilities |= simsimd_cap_sve_i8_k;
    } else if (same_string(cap_name, "haswell")) {
        static_capabilities |= simsimd_cap_haswell_k;
    } else if (same_string(cap_name, "skylake")) {
        static_capabilities |= simsimd_cap_skylake_k;
    } else if (same_string(cap_name, "ice")) {
        static_capabilities |= simsimd_cap_ice_k;
    } else if (same_string(cap_name, "genoa")) {
        static_capabilities |= simsimd_cap_genoa_k;
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
    } else if (same_string(cap_name, "neon_f16")) {
        static_capabilities &= ~simsimd_cap_neon_f16_k;
    } else if (same_string(cap_name, "neon_bf16")) {
        static_capabilities &= ~simsimd_cap_neon_bf16_k;
    } else if (same_string(cap_name, "neon_i8")) {
        static_capabilities &= ~simsimd_cap_neon_i8_k;
    } else if (same_string(cap_name, "sve")) {
        static_capabilities &= ~simsimd_cap_sve_k;
    } else if (same_string(cap_name, "sve_f16")) {
        static_capabilities &= ~simsimd_cap_sve_f16_k;
    } else if (same_string(cap_name, "sve_bf16")) {
        static_capabilities &= ~simsimd_cap_sve_bf16_k;
    } else if (same_string(cap_name, "sve_i8")) {
        static_capabilities &= ~simsimd_cap_sve_i8_k;
    } else if (same_string(cap_name, "haswell")) {
        static_capabilities &= ~simsimd_cap_haswell_k;
    } else if (same_string(cap_name, "skylake")) {
        static_capabilities &= ~simsimd_cap_skylake_k;
    } else if (same_string(cap_name, "ice")) {
        static_capabilities &= ~simsimd_cap_ice_k;
    } else if (same_string(cap_name, "genoa")) {
        static_capabilities &= ~simsimd_cap_genoa_k;
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
    ADD_CAP(neon_f16);
    ADD_CAP(sve_f16);
    ADD_CAP(neon_bf16);
    ADD_CAP(sve_bf16);
    ADD_CAP(neon_i8);
    ADD_CAP(sve_i8);
    ADD_CAP(haswell);
    ADD_CAP(skylake);
    ADD_CAP(ice);
    ADD_CAP(genoa);
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
    parsed->rank = buffer->ndim;
    if (buffer->ndim == 1) {
        if (buffer->strides[0] > buffer->itemsize) {
            PyErr_SetString(PyExc_ValueError, "Input vectors must be contiguous");
            PyBuffer_Release(buffer);
            return -1;
        }
        parsed->dimensions = buffer->shape[0];
        parsed->count = 1;
        parsed->stride = 0;
    } else if (buffer->ndim == 2) {
        if (buffer->strides[1] > buffer->itemsize) {
            PyErr_SetString(PyExc_ValueError, "Input vectors must be contiguous");
            PyBuffer_Release(buffer);
            return -1;
        }
        parsed->dimensions = buffer->shape[1];
        parsed->count = buffer->shape[0];
        parsed->stride = buffer->strides[0];
    } else {
        PyErr_SetString(PyExc_ValueError, "Input tensors must be 1D or 2D");
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

static PyObject* implement_dense_metric(simsimd_metric_kind_t metric_kind, PyObject* const* args, Py_ssize_t nargs) {
    // Function now accepts up to 3 arguments, the third being optional
    if (nargs < 2 || nargs > 3) {
        PyErr_SetString(PyExc_TypeError, "Function expects 2 or 3 arguments");
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
        PyErr_SetString(PyExc_ValueError, "Vector dimensions don't match");
        goto cleanup;
    }
    if (parsed_a.count == 0 || parsed_b.count == 0) {
        PyErr_SetString(PyExc_ValueError, "Collections can't be empty");
        goto cleanup;
    }
    if (parsed_a.count > 1 && parsed_b.count > 1 && parsed_a.count != parsed_b.count) {
        PyErr_SetString(PyExc_ValueError, "Collections must have the same number of elements or just one element");
        goto cleanup;
    }

    // Check data types
    if (parsed_a.datatype != parsed_b.datatype && parsed_a.datatype != simsimd_datatype_unknown_k &&
        parsed_b.datatype != simsimd_datatype_unknown_k) {
        PyErr_SetString(PyExc_TypeError,
                        "Input tensors must have matching datatypes, check with `X.__array_interface__`");
        goto cleanup;
    }

    // Process the third argument, `value_type_desc`, if provided
    simsimd_datatype_t input_datatype = parsed_a.datatype;
    if (value_type_desc != NULL) {
        // Ensure it is a string (or convert it to one if possible)
        if (!PyUnicode_Check(value_type_desc)) {
            PyErr_SetString(PyExc_TypeError, "third argument must be a string describing the value type");
            goto cleanup;
        }
        // Convert Python string to C string
        char const* value_type_str = PyUnicode_AsUTF8(value_type_desc);
        if (!value_type_str) {
            PyErr_SetString(PyExc_ValueError, "Could not convert value type description to string");
            goto cleanup;
        }
        input_datatype = python_string_to_datatype(value_type_str);
    }

    simsimd_metric_punned_t metric = NULL;
    simsimd_capability_t capability = simsimd_cap_serial_k;
    simsimd_find_metric_punned(metric_kind, input_datatype, static_capabilities, simsimd_cap_any_k, &metric,
                               &capability);
    if (!metric) {
        PyErr_SetString(PyExc_LookupError, "Unsupported metric and datatype combination");
        goto cleanup;
    }

    // If the distance is computed between two vectors, rather than matrices, return a scalar
    int datatype_is_complex = is_complex(input_datatype);
    simsimd_datatype_t return_datatype = datatype_is_complex ? simsimd_datatype_f64c_k : simsimd_datatype_f64_k;
    if (parsed_a.rank == 1 && parsed_b.rank == 1) {
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
        DistancesTensor* distances_obj = PyObject_NewVar(DistancesTensor, &DistancesTensorType,
                                                         count_components * bytes_per_datatype(return_datatype));
        if (!distances_obj) {
            PyErr_NoMemory();
            goto cleanup;
        }

        // Initialize the object
        distances_obj->datatype = return_datatype;
        distances_obj->dimensions = 1;
        distances_obj->shape[0] = count_pairs;
        distances_obj->shape[1] = 1;
        distances_obj->strides[0] = bytes_per_datatype(return_datatype);
        distances_obj->strides[1] = 0;
        output = (PyObject*)distances_obj;

        // Compute the distances
        simsimd_distance_t* distances = (simsimd_distance_t*)&distances_obj->start[0];
        for (size_t i = 0; i < count_pairs; ++i) {
            simsimd_distance_t result[2];
            metric(                                   //
                parsed_a.start + i * parsed_a.stride, //
                parsed_b.start + i * parsed_b.stride, //
                parsed_a.dimensions,                  //
                (simsimd_distance_t*)&result);

            // Export out:
            if (!cast_distance(result[0], return_datatype, distances, i * components_per_pair)) {
                PyErr_SetString(PyExc_ValueError, "Unsupported datatype");
                goto cleanup;
            }
            if (datatype_is_complex)
                cast_distance(result[1], return_datatype, distances, i * components_per_pair + 1);
        }
    }

cleanup:
    PyBuffer_Release(&buffer_a);
    PyBuffer_Release(&buffer_b);
    return output;
}

static PyObject* implement_curved_metric(simsimd_metric_kind_t metric_kind, PyObject* const* args, Py_ssize_t nargs) {
    // Function now accepts up to 4 arguments, the fourth being optional
    if (nargs < 3 || nargs > 4) {
        PyErr_SetString(PyExc_TypeError, "Function expects 4 or 5 arguments");
        return NULL;
    }

    PyObject* output = NULL;
    PyObject* input_tensor_a = args[0];
    PyObject* input_tensor_b = args[1];
    PyObject* input_tensor_c = args[2];
    PyObject* value_type_desc = nargs == 4 ? args[3] : NULL;

    Py_buffer buffer_a, buffer_b, buffer_c;
    TensorArgument parsed_a, parsed_b, parsed_c;
    if (parse_tensor(input_tensor_a, &buffer_a, &parsed_a) != 0 ||
        parse_tensor(input_tensor_b, &buffer_b, &parsed_b) != 0 ||
        parse_tensor(input_tensor_c, &buffer_c, &parsed_c) != 0) {
        return NULL; // Error already set by parse_tensor
    }

    // Check dimensions
    if (parsed_a.rank != 1 || parsed_b.rank != 1) {
        PyErr_SetString(PyExc_ValueError, "First and second argument must be vectors");
        goto cleanup;
    }
    if (parsed_c.rank != 2) {
        PyErr_SetString(PyExc_ValueError, "Third argument must be a matrix (rank-2 tensor)");
        goto cleanup;
    }
    if (parsed_a.count == 0 || parsed_b.count == 0) {
        PyErr_SetString(PyExc_ValueError, "Collections can't be empty");
        goto cleanup;
    }
    if (parsed_a.count > 1 && parsed_b.count > 1 && parsed_a.count != parsed_b.count) {
        PyErr_SetString(PyExc_ValueError, "Collections must have the same number of elements or just one element");
        goto cleanup;
    }

    // Check data types
    if (parsed_a.datatype != parsed_b.datatype && parsed_a.datatype != simsimd_datatype_unknown_k &&
        parsed_b.datatype != simsimd_datatype_unknown_k) {
        PyErr_SetString(PyExc_TypeError,
                        "Input tensors must have matching datatypes, check with `X.__array_interface__`");
        goto cleanup;
    }

    // Process the third argument, `value_type_desc`, if provided
    simsimd_datatype_t input_datatype = parsed_a.datatype;
    if (value_type_desc != NULL) {
        // Ensure it is a string (or convert it to one if possible)
        if (!PyUnicode_Check(value_type_desc)) {
            PyErr_SetString(PyExc_TypeError, "Third argument must be a string describing the value type");
            goto cleanup;
        }
        // Convert Python string to C string
        char const* value_type_str = PyUnicode_AsUTF8(value_type_desc);
        if (!value_type_str) {
            PyErr_SetString(PyExc_ValueError, "Could not convert value type description to string");
            goto cleanup;
        }
        input_datatype = python_string_to_datatype(value_type_str);
    }

    simsimd_metric_curved_punned_t metric = NULL;
    simsimd_capability_t capability = simsimd_cap_serial_k;
    simsimd_find_metric_punned(metric_kind, input_datatype, static_capabilities, simsimd_cap_any_k,
                               (simsimd_metric_punned_t*)&metric, &capability);
    if (!metric) {
        PyErr_Format(PyExc_LookupError,
                     "Unsupported metric '%c' and datatype combination across vectors ('%s'/'%s' and '%s'/'%s') and "
                     "tensor ('%s'/'%s')",
                     metric_kind,                                                   //
                     buffer_a.format, datatype_to_python_string(parsed_a.datatype), //
                     buffer_b.format, datatype_to_python_string(parsed_b.datatype), //
                     buffer_c.format, datatype_to_python_string(parsed_c.datatype));
        goto cleanup;
    }

    simsimd_distance_t distance;
    metric(parsed_a.start, parsed_b.start, parsed_c.start, parsed_a.dimensions, &distance);
    output = PyFloat_FromDouble(distance);

cleanup:
    PyBuffer_Release(&buffer_a);
    PyBuffer_Release(&buffer_b);
    PyBuffer_Release(&buffer_c);
    return output;
}

static PyObject* implement_sparse_metric(simsimd_metric_kind_t metric_kind, PyObject* const* args, Py_ssize_t nargs) {
    if (nargs != 2) {
        PyErr_SetString(PyExc_TypeError, "Function expects only 2 arguments");
        return NULL;
    }

    PyObject* output = NULL;
    PyObject* input_tensor_a = args[0];
    PyObject* input_tensor_b = args[1];

    Py_buffer buffer_a, buffer_b;
    TensorArgument parsed_a, parsed_b;
    if (parse_tensor(input_tensor_a, &buffer_a, &parsed_a) != 0 ||
        parse_tensor(input_tensor_b, &buffer_b, &parsed_b) != 0) {
        return NULL; // Error already set by parse_tensor
    }

    // Check dimensions
    if (parsed_a.rank != 1 || parsed_b.rank != 1) {
        PyErr_SetString(PyExc_ValueError, "First and second argument must be vectors");
        goto cleanup;
    }

    // Check data types
    if (parsed_a.datatype != parsed_b.datatype && parsed_a.datatype != simsimd_datatype_unknown_k &&
        parsed_b.datatype != simsimd_datatype_unknown_k) {
        PyErr_SetString(PyExc_TypeError,
                        "Input tensors must have matching datatypes, check with `X.__array_interface__`");
        goto cleanup;
    }

    simsimd_datatype_t input_datatype = parsed_a.datatype;
    simsimd_metric_sparse_punned_t metric = NULL;
    simsimd_capability_t capability = simsimd_cap_serial_k;
    simsimd_find_metric_punned(metric_kind, input_datatype, static_capabilities, simsimd_cap_any_k,
                               (simsimd_metric_punned_t*)&metric, &capability);
    if (!metric) {
        PyErr_Format(PyExc_LookupError, "Unsupported metric '%c' and datatype combination ('%s'/'%s' and '%s'/'%s')",
                     metric_kind,                                                   //
                     buffer_a.format, datatype_to_python_string(parsed_a.datatype), //
                     buffer_b.format, datatype_to_python_string(parsed_b.datatype));
        goto cleanup;
    }

    simsimd_distance_t distance;
    metric(parsed_a.start, parsed_b.start, parsed_a.dimensions, parsed_b.dimensions, &distance);
    output = PyFloat_FromDouble(distance);

cleanup:
    PyBuffer_Release(&buffer_a);
    PyBuffer_Release(&buffer_b);
    return output;
}

static PyObject* impl_cdist(                            //
    PyObject* input_tensor_a, PyObject* input_tensor_b, //
    simsimd_metric_kind_t metric_kind, size_t threads, simsimd_datatype_t input_datatype,
    simsimd_datatype_t return_datatype) {

    PyObject* output = NULL;
    Py_buffer buffer_a, buffer_b;
    TensorArgument parsed_a, parsed_b;
    if (parse_tensor(input_tensor_a, &buffer_a, &parsed_a) != 0 ||
        parse_tensor(input_tensor_b, &buffer_b, &parsed_b) != 0) {
        return NULL; // Error already set by parse_tensor
    }

    // Check dimensions
    if (parsed_a.dimensions != parsed_b.dimensions) {
        PyErr_Format(PyExc_ValueError, "Vector dimensions don't match (%d != %d)", parsed_a.dimensions,
                     parsed_b.dimensions);
        goto cleanup;
    }
    if (parsed_a.count == 0 || parsed_b.count == 0) {
        PyErr_SetString(PyExc_ValueError, "Collections can't be empty");
        goto cleanup;
    }

    // Check data types
    if (parsed_a.datatype != parsed_b.datatype && parsed_a.datatype != simsimd_datatype_unknown_k &&
        parsed_b.datatype != simsimd_datatype_unknown_k) {
        PyErr_SetString(PyExc_TypeError,
                        "Input tensors must have matching datatypes, check with `X.__array_interface__`");
        goto cleanup;
    }
    if (input_datatype == simsimd_datatype_unknown_k)
        input_datatype = parsed_a.datatype;

    simsimd_metric_punned_t metric = NULL;
    simsimd_capability_t capability = simsimd_cap_serial_k;
    simsimd_find_metric_punned(metric_kind, input_datatype, static_capabilities, simsimd_cap_any_k, &metric,
                               &capability);
    if (!metric) {
        PyErr_Format(PyExc_LookupError, "Unsupported metric '%c' and datatype combination ('%s'/'%s' and '%s'/'%s')",
                     metric_kind,                                                   //
                     buffer_a.format, datatype_to_python_string(parsed_a.datatype), //
                     buffer_b.format, datatype_to_python_string(parsed_b.datatype));
        goto cleanup;
    }

    // Make sure the return datatype is complex if the input datatype is complex,
    // and the same for real numbers
    if (return_datatype != simsimd_datatype_unknown_k) {
        if (is_complex(input_datatype) != is_complex(return_datatype)) {
            PyErr_SetString(
                PyExc_ValueError,
                "If the input datatype is complex, the return datatype must be complex, and same for real.");
            goto cleanup;
        }
    } else {
        return_datatype = is_complex(input_datatype) ? simsimd_datatype_f64c_k : simsimd_datatype_f64_k;
    }

    // If the distance is computed between two vectors, rather than matrices, return a scalar
    int datatype_is_complex = is_complex(input_datatype);
    if (parsed_a.rank == 1 && parsed_b.rank == 1) {
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

        // Check if the downcasting to provided datatype is supported
        {

            char returned_buffer_example[8];
            if (!cast_distance(0, return_datatype, &returned_buffer_example, 0)) {
                PyErr_SetString(PyExc_ValueError, "Unsupported datatype");
                goto cleanup;
            }
        }

        size_t const count_pairs = parsed_a.count * parsed_b.count;
        size_t const components_per_pair = datatype_is_complex ? 2 : 1;
        size_t const count_components = count_pairs * components_per_pair;
        DistancesTensor* distances_obj = PyObject_NewVar(DistancesTensor, &DistancesTensorType,
                                                         count_components * bytes_per_datatype(return_datatype));
        if (!distances_obj) {
            PyErr_NoMemory();
            goto cleanup;
        }

        // Initialize the object
        distances_obj->datatype = return_datatype;
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
            for (size_t j = 0; j < parsed_b.count; ++j) {
                simsimd_distance_t result[2];
                metric(                                   //
                    parsed_a.start + i * parsed_a.stride, //
                    parsed_b.start + j * parsed_b.stride, //
                    parsed_a.dimensions,                  //
                    (simsimd_distance_t*)&result          //
                );
                // Export out:
                cast_distance(result[0], return_datatype, distances,
                              i * components_per_pair * parsed_b.count + j * components_per_pair);
                if (datatype_is_complex)
                    cast_distance(result[1], return_datatype, distances,
                                  i * components_per_pair * parsed_b.count + j * components_per_pair + 1);
            }
    }

cleanup:
    PyBuffer_Release(&buffer_a);
    PyBuffer_Release(&buffer_b);
    return output;
}

static PyObject* implement_pointer_access(simsimd_metric_kind_t metric_kind, PyObject* args) {
    char const* type_name = PyUnicode_AsUTF8(PyTuple_GetItem(args, 0));
    if (!type_name) {
        PyErr_SetString(PyExc_TypeError, "Invalid type name");
        return NULL;
    }

    simsimd_datatype_t datatype = python_string_to_datatype(type_name);
    if (!datatype) { // Check the actual variable here instead of type_name
        PyErr_SetString(PyExc_TypeError, "Unsupported type");
        return NULL;
    }

    simsimd_metric_punned_t metric = NULL;
    simsimd_capability_t capability = simsimd_cap_serial_k;
    simsimd_find_metric_punned(metric_kind, datatype, static_capabilities, simsimd_cap_any_k, &metric, &capability);
    if (metric == NULL) {
        PyErr_SetString(PyExc_LookupError, "No such metric");
        return NULL;
    }

    return PyLong_FromUnsignedLongLong((unsigned long long)metric);
}

static PyObject* api_cdist(PyObject* self, PyObject* const* args, Py_ssize_t args_count, PyObject* kwnames) {
    // This function accepts up to 6 arguments:
    PyObject* input_tensor_a = NULL; // Required object, positional-only
    PyObject* input_tensor_b = NULL; // Required object, positional-only
    PyObject* metric_obj = NULL;     // Optional string, positional or keyword
    PyObject* threads_obj = NULL;    // Optional integer, keyword-only
    PyObject* dtype_obj = NULL;      // Optional string, keyword-only
    PyObject* out_dtype_obj = NULL;  // Optional string, keyword-only

    // Once parsed, the arguments will be stored in these variables:
    char const* metric_str = NULL;
    unsigned long long threads = 1;
    char const* dtype_str = NULL;
    char const* out_dtype_str = NULL;

    // The lazy implementation would be to use `PyArg_ParseTupleAndKeywords` for a `kwnames` dictionary:
    // static char* kwlist[] = {"input_tensor_a", "input_tensor_b", "metric", "threads", "dtype", "out_dtype", NULL};
    // if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|s$Kss", kwlist, &input_tensor_a, &input_tensor_b, &metric_str,
    //                                  &threads, &dtype_str, &out_dtype_str))
    //     return NULL;
    if (args_count < 2 || args_count > 6) {
        PyErr_Format(PyExc_TypeError, "Function expects 2-6 arguments, got %d", args_count);
        return NULL;
    }

    // Positional-only arguments
    input_tensor_a = args[0];
    input_tensor_b = args[1];

    // Positional or keyword arguments
    Py_ssize_t args_progress = 2;
    Py_ssize_t kwnames_progress = 0;
    Py_ssize_t kwnames_count = PyTuple_Size(kwnames);
    if (args_count > 2 || kwnames_count > 0) {
        metric_obj = args[2];
        if (kwnames) {
            PyObject* key = PyTuple_GetItem(kwnames, 0);
            if (key != NULL && PyUnicode_CompareWithASCIIString(key, "metric") != 0) {
                PyErr_SetString(PyExc_ValueError, "Third argument must be 'metric'");
                return NULL;
            }
            args_progress = 3;
            kwnames_progress = 1;
        }
    }

    // The rest of the arguments must be checked in the keyword dictionary
    for (; kwnames_progress < kwnames_count; ++args_progress, ++kwnames_progress) {
        PyObject* key = PyTuple_GetItem(kwnames, kwnames_progress);
        PyObject* value = args[args_progress];
        if (PyUnicode_CompareWithASCIIString(key, "threads") == 0) {
            if (threads_obj != NULL) {
                PyErr_SetString(PyExc_ValueError, "Duplicate argument for 'threads'");
                return NULL;
            }
            threads_obj = value;
        } else if (PyUnicode_CompareWithASCIIString(key, "dtype") == 0) {
            if (dtype_obj != NULL) {
                PyErr_SetString(PyExc_ValueError, "Duplicate argument for 'dtype'");
                return NULL;
            }
            dtype_obj = value;
        } else if (PyUnicode_CompareWithASCIIString(key, "out_dtype") == 0) {
            if (out_dtype_obj != NULL) {
                PyErr_SetString(PyExc_ValueError, "Duplicate argument for 'out_dtype'");
                return NULL;
            }
            out_dtype_obj = value;
        } else {
            PyErr_Format(PyExc_ValueError, "Received unknown keyword argument: %S", key);
            return NULL;
        }
    }

    // Process the PyObject values
    simsimd_metric_kind_t metric_kind = simsimd_metric_l2sq_k;
    if (metric_obj) {
        metric_str = PyUnicode_AsUTF8(metric_obj);
        if (!metric_str && PyErr_Occurred()) {
            PyErr_SetString(PyExc_TypeError, "Expected 'metric' to be a string");
            return NULL;
        }
        metric_kind = python_string_to_metric_kind(metric_str);
        if (metric_kind == simsimd_metric_unknown_k) {
            PyErr_SetString(PyExc_LookupError, "Unsupported metric");
            return NULL;
        }
    }

    threads = 1;
    if (threads_obj)
        threads = PyLong_AsSize_t(threads_obj);
    if (PyErr_Occurred()) {
        PyErr_SetString(PyExc_TypeError, "Expected 'threads' to be an unsigned integer");
        return NULL;
    }

    simsimd_datatype_t dtype = simsimd_datatype_unknown_k;
    if (dtype_obj) {
        dtype_str = PyUnicode_AsUTF8(dtype_obj);
        if (!dtype_str && PyErr_Occurred()) {
            PyErr_SetString(PyExc_TypeError, "Expected 'dtype' to be a string");
            return NULL;
        }
        dtype = python_string_to_datatype(dtype_str);
        if (dtype == simsimd_datatype_unknown_k) {
            PyErr_SetString(PyExc_ValueError, "Unsupported 'dtype'");
            return NULL;
        }
    }

    simsimd_datatype_t out_dtype = simsimd_datatype_f64_k;
    if (out_dtype_obj) {
        out_dtype_str = PyUnicode_AsUTF8(out_dtype_obj);
        if (!out_dtype_str && PyErr_Occurred()) {
            PyErr_SetString(PyExc_TypeError, "Expected 'out_dtype' to be a string");
            return NULL;
        }
        out_dtype = python_string_to_datatype(out_dtype_str);
        if (out_dtype == simsimd_datatype_unknown_k) {
            PyErr_SetString(PyExc_ValueError, "Unsupported 'out_dtype'");
            return NULL;
        }
    }

    return impl_cdist(input_tensor_a, input_tensor_b, metric_kind, threads, dtype, out_dtype);
}

static PyObject* api_l2sq_pointer(PyObject* self, PyObject* args) {
    return implement_pointer_access(simsimd_metric_l2sq_k, args);
}
static PyObject* api_cos_pointer(PyObject* self, PyObject* args) {
    return implement_pointer_access(simsimd_metric_cos_k, args);
}
static PyObject* api_dot_pointer(PyObject* self, PyObject* args) {
    return implement_pointer_access(simsimd_metric_dot_k, args);
}
static PyObject* api_vdot_pointer(PyObject* self, PyObject* args) {
    return implement_pointer_access(simsimd_metric_vdot_k, args);
}
static PyObject* api_kl_pointer(PyObject* self, PyObject* args) {
    return implement_pointer_access(simsimd_metric_kl_k, args);
}
static PyObject* api_js_pointer(PyObject* self, PyObject* args) {
    return implement_pointer_access(simsimd_metric_js_k, args);
}
static PyObject* api_hamming_pointer(PyObject* self, PyObject* args) {
    return implement_pointer_access(simsimd_metric_hamming_k, args);
}
static PyObject* api_jaccard_pointer(PyObject* self, PyObject* args) {
    return implement_pointer_access(simsimd_metric_jaccard_k, args);
}
static PyObject* api_l2sq(PyObject* self, PyObject* const* args, Py_ssize_t nargs) {
    return implement_dense_metric(simsimd_metric_l2sq_k, args, nargs);
}
static PyObject* api_cos(PyObject* self, PyObject* const* args, Py_ssize_t nargs) {
    return implement_dense_metric(simsimd_metric_cos_k, args, nargs);
}
static PyObject* api_dot(PyObject* self, PyObject* const* args, Py_ssize_t nargs) {
    return implement_dense_metric(simsimd_metric_dot_k, args, nargs);
}
static PyObject* api_vdot(PyObject* self, PyObject* const* args, Py_ssize_t nargs) {
    return implement_dense_metric(simsimd_metric_vdot_k, args, nargs);
}
static PyObject* api_kl(PyObject* self, PyObject* const* args, Py_ssize_t nargs) {
    return implement_dense_metric(simsimd_metric_kl_k, args, nargs);
}
static PyObject* api_js(PyObject* self, PyObject* const* args, Py_ssize_t nargs) {
    return implement_dense_metric(simsimd_metric_js_k, args, nargs);
}
static PyObject* api_hamming(PyObject* self, PyObject* const* args, Py_ssize_t nargs) {
    return implement_dense_metric(simsimd_metric_hamming_k, args, nargs);
}
static PyObject* api_jaccard(PyObject* self, PyObject* const* args, Py_ssize_t nargs) {
    return implement_dense_metric(simsimd_metric_jaccard_k, args, nargs);
}
static PyObject* api_bilinear(PyObject* self, PyObject* const* args, Py_ssize_t nargs) {
    return implement_curved_metric(simsimd_metric_bilinear_k, args, nargs);
}
static PyObject* api_mahalanobis(PyObject* self, PyObject* const* args, Py_ssize_t nargs) {
    return implement_curved_metric(simsimd_metric_mahalanobis_k, args, nargs);
}
static PyObject* api_intersect(PyObject* self, PyObject* const* args, Py_ssize_t nargs) {
    return implement_sparse_metric(simsimd_metric_intersect_k, args, nargs);
}

static PyMethodDef simsimd_methods[] = {
    // Introspecting library and hardware capabilities
    {
        "get_capabilities",
        (PyCFunction)api_get_capabilities,
        METH_NOARGS,
        "Get the current hardware SIMD capabilities as a dictionary of feature flags.\n"
        "On x86 includes: 'serial', 'haswell', 'skylake', 'ice', 'genoa', 'sapphire'.\n"
        "On Arm includes: 'serial', 'neon', 'sve', 'sve2', and their extensions.\n",
    },
    {
        "enable_capability",
        (PyCFunction)api_enable_capability,
        METH_VARARGS,
        "Enable a specific SIMD kernel family.\n\n"
        "Args:\n"
        "    capability (str): The name of the SIMD feature to enable (e.g., 'haswell').",
    },
    {
        "disable_capability",
        (PyCFunction)api_disable_capability,
        METH_VARARGS,
        "Disable a specific SIMD kernel family.\n\n"
        "Args:\n"
        "    capability (str): The name of the SIMD feature to disable (e.g., 'haswell').",
    },

    // NumPy and SciPy compatible interfaces for dense vector representations
    // Each function can compute distances between:
    //  - A pair of vectors
    //  - A batch of vector pairs (two matrices of identical shape)
    //  - A matrix of vectors and a single vector
    {
        "sqeuclidean",
        (PyCFunction)api_l2sq,
        METH_FASTCALL,
        "Compute squared Euclidean (L2) distances between two matrices.\n\n"
        "Args:\n"
        "    a (NDArray): First matrix or vector.\n"
        "    b (NDArray): Second matrix or vector.\n"
        "    dtype (Union[IntegralType, FloatType], optional): Override the presumed input type.\n"
        "    out_dtype (Union[FloatType, ComplexType], optional): Result type, default is 'float64'.\n\n"
        "Returns:\n"
        "    DistancesTensor: The squared Euclidean distances.\n\n"
        "Equivalent to: `scipy.spatial.distance.sqeuclidean`.\n"
        "Notes:\n"
        "    * `a` and `b` are positional-only arguments, while `dtype` and `out_dtype` are keyword-only arguments.",
    },
    {
        "cosine",
        (PyCFunction)api_cos,
        METH_FASTCALL,
        "Compute cosine (angular) distances between two matrices.\n\n"
        "Args:\n"
        "    a (NDArray): First matrix or vector.\n"
        "    b (NDArray): Second matrix or vector.\n"
        "    dtype (Union[IntegralType, FloatType], optional): Override the presumed input type.\n"
        "    out_dtype (Union[FloatType, ComplexType], optional): Result type, default is 'float64'.\n\n"
        "Returns:\n"
        "    DistancesTensor: The cosine distances.\n\n"
        "Equivalent to: `scipy.spatial.distance.cosine`.\n"
        "Notes:\n"
        "    * `a` and `b` are positional-only arguments, while `dtype` and `out_dtype` are keyword-only arguments.",
    },
    {
        "inner",
        (PyCFunction)api_dot,
        METH_FASTCALL,
        "Compute the inner (dot) product between two matrices (real or complex).\n\n"
        "Args:\n"
        "    a (NDArray): First matrix or vector.\n"
        "    b (NDArray): Second matrix or vector.\n"
        "    dtype (Union[FloatType, ComplexType], optional): Override the presumed input type.\n\n"
        "Returns:\n"
        "    DistancesTensor: The inner product.\n\n"
        "Equivalent to: `numpy.inner`.\n"
        "Notes:\n"
        "    * `a` and `b` are positional-only arguments, while `dtype` is a keyword-only argument.",
    },
    {
        "dot",
        (PyCFunction)api_dot,
        METH_FASTCALL,
        "Compute the dot product between two matrices (real or complex).\n\n"
        "Args:\n"
        "    a (NDArray): First matrix or vector.\n"
        "    b (NDArray): Second matrix or vector.\n"
        "    dtype (Union[FloatType, ComplexType], optional): Override the presumed input type.\n\n"
        "Returns:\n"
        "    DistancesTensor: The dot product.\n\n"
        "Equivalent to: `numpy.dot`.\n"
        "Notes:\n"
        "    * `a` and `b` are positional-only arguments, while `dtype` is a keyword-only argument.",
    },
    {
        "vdot",
        (PyCFunction)api_vdot,
        METH_FASTCALL,
        "Compute the conjugate dot product between two complex matrices.\n\n"
        "Args:\n"
        "    a (NDArray): First complex matrix or vector.\n"
        "    b (NDArray): Second complex matrix or vector.\n"
        "    dtype (Union[ComplexType], optional): Override the presumed input type.\n\n"
        "Returns:\n"
        "    DistancesTensor: The conjugate dot product.\n\n"
        "Equivalent to: `numpy.vdot`.\n"
        "Notes:\n"
        "    * `a` and `b` are positional-only arguments, while `dtype` is a keyword-only argument.",
    },
    {
        "hamming",
        (PyCFunction)api_hamming,
        METH_FASTCALL,
        "Compute Hamming distances between two matrices.\n\n"
        "Args:\n"
        "    a (NDArray): First binary matrix or vector.\n"
        "    b (NDArray): Second binary matrix or vector.\n"
        "    dtype (IntegralType, optional): Override the presumed input type.\n\n"
        "Returns:\n"
        "    DistancesTensor: The Hamming distances.\n\n"
        "Equivalent to: `scipy.spatial.distance.hamming`.\n"
        "Notes:\n"
        "    * `a` and `b` are positional-only arguments, while `dtype` is a keyword-only argument.",
    },
    {
        "jaccard",
        (PyCFunction)api_jaccard,
        METH_FASTCALL,
        "Compute Jaccard distances (bitwise Tanimoto) between two matrices.\n\n"
        "Args:\n"
        "    a (NDArray): First binary matrix or vector.\n"
        "    b (NDArray): Second binary matrix or vector.\n"
        "    dtype (IntegralType, optional): Override the presumed input type.\n\n"
        "Returns:\n"
        "    DistancesTensor: The Jaccard distances.\n\n"
        "Equivalent to: `scipy.spatial.distance.jaccard`.\n"
        "Notes:\n"
        "    * `a` and `b` are positional-only arguments, while `dtype` is a keyword-only argument.",
    },
    {
        "kullbackleibler",
        (PyCFunction)api_kl,
        METH_FASTCALL,
        "Compute Kullback-Leibler divergences between two matrices.\n\n"
        "Args:\n"
        "    a (NDArray): First floating-point matrix or vector.\n"
        "    b (NDArray): Second floating-point matrix or vector.\n"
        "    dtype (IntegralType, optional): Override the presumed input type.\n\n"
        "Returns:\n"
        "    DistancesTensor: The Kullback-Leibler divergences distances.\n\n"
        "Equivalent to: `scipy.special.kl_div`.\n"
        "Notes:\n"
        "    * `a` and `b` are positional-only arguments, while `dtype` is a keyword-only argument.",
    },
    {
        "jensenshannon",
        (PyCFunction)api_js,
        METH_FASTCALL,
        "Compute Jensen-Shannon divergences between two matrices.\n\n"
        "Args:\n"
        "    a (NDArray): First floating-point matrix or vector.\n"
        "    b (NDArray): Second floating-point matrix or vector.\n"
        "    dtype (IntegralType, optional): Override the presumed input type.\n\n"
        "Returns:\n"
        "    DistancesTensor: The Jensen-Shannon divergences distances.\n\n"
        "Equivalent to: `scipy.spatial.distance.jensenshannon`.\n"
        "Notes:\n"
        "    * `a` and `b` are positional-only arguments, while `dtype` is a keyword-only argument.",
    },

    // Conventional `cdist` interface for pairwise distances
    {
        "cdist",
        (PyCFunction)api_cdist,
        METH_FASTCALL | METH_KEYWORDS,
        "Compute pairwise distances between two sets of input matrices.\n\n"
        "Args:\n"
        "    a (NDArray): First matrix.\n"
        "    b (NDArray): Second matrix.\n"
        "    metric (str, optional): Distance metric to use (e.g., 'sqeuclidean', 'cosine').\n"
        "    threads (int, optional): Number of threads to use (default is 1).\n"
        "    dtype (Union[IntegralType, FloatType, ComplexType], optional): Override the presumed input type.\n"
        "    out_dtype (Union[FloatType, ComplexType], optional): Result type, default is 'float64'.\n\n"
        "Returns:\n"
        "    DistancesTensor: Pairwise distances between all inputs.\n\n"
        "Equivalent to: `scipy.spatial.distance.cdist`.\n"
        "Notes:\n"
        "    * `a` and `b` are positional-only arguments.\n"
        "    * `metric` can be positional or keyword.\n"
        "    * `threads`, `dtype`, and `out_dtype` are keyword-only arguments.",
    },

    // Exposing underlying API for USearch
    {
        "pointer_to_sqeuclidean",
        (PyCFunction)api_l2sq_pointer,
        METH_VARARGS,
        "Retrieve the function pointer for the squared Euclidean distance function as an integer.",
    },
    {
        "pointer_to_cosine",
        (PyCFunction)api_cos_pointer,
        METH_VARARGS,
        "Retrieve the function pointer for the cosine distance function as an integer.",
    },
    {
        "pointer_to_inner",
        (PyCFunction)api_dot_pointer,
        METH_VARARGS,
        "Retrieve the function pointer for the inner (dot) product function as an integer.",
    },
    {
        "pointer_to_dot",
        (PyCFunction)api_dot_pointer,
        METH_VARARGS,
        "Retrieve the function pointer for the dot product function as an integer.",
    },
    {
        "pointer_to_vdot",
        (PyCFunction)api_vdot_pointer,
        METH_VARARGS,
        "Retrieve the function pointer for the conjugate dot product function as an integer.",
    },
    {
        "pointer_to_kullbackleibler",
        (PyCFunction)api_kl_pointer,
        METH_VARARGS,
        "Retrieve the function pointer for the Kullback-Leibler divergence function as an integer.",
    },
    {
        "pointer_to_jensenshannon",
        (PyCFunction)api_js_pointer,
        METH_VARARGS,
        "Retrieve the function pointer for the Jensen-Shannon divergence function as an integer.",
    },

    // Set operations
    {
        "intersect",
        (PyCFunction)api_intersect,
        METH_FASTCALL,
        "Compute the intersection of two sorted integer arrays.\n\n"
        "Args:\n"
        "    a (NDArray): First sorted integer array.\n"
        "    b (NDArray): Second sorted integer array.\n\n"
        "Returns:\n"
        "    float: The number of intersecting elements.\n\n"
        "Similar to: `numpy.intersect1d`.",
    },

    // Curved spaces
    {
        "bilinear",
        (PyCFunction)api_bilinear,
        METH_FASTCALL,
        "Compute the bilinear form between two vectors given a metric tensor.\n\n"
        "Args:\n"
        "    a (NDArray): First vector.\n"
        "    b (NDArray): Second vector.\n"
        "    metric_tensor (NDArray): The metric tensor defining the bilinear form.\n"
        "    dtype (FloatType, optional): Override the presumed input type.\n\n"
        "Returns:\n"
        "    float: The bilinear form.\n\n"
        "Equivalent to: `numpy.dot` with a metric tensor.\n"
        "Notes:\n"
        "    * `a`, `b`, and `metric_tensor` are positional-only arguments, while `dtype` is keyword-only.",
    },

    {
        "mahalanobis",
        (PyCFunction)api_mahalanobis,
        METH_FASTCALL,
        "Compute the Mahalanobis distance between two vectors given an inverse covariance matrix.\n\n"
        "Args:\n"
        "    a (NDArray): First vector.\n"
        "    b (NDArray): Second vector.\n"
        "    inverse_covariance (NDArray): The inverse of the covariance matrix.\n"
        "    dtype (FloatType, optional): Override the presumed input type.\n\n"
        "Returns:\n"
        "    float: The Mahalanobis distance.\n\n"
        "Equivalent to: `scipy.spatial.distance.mahalanobis`.\n"
        "Notes:\n"
        "    * `a`, `b`, and `inverse_covariance` are positional-only arguments, while `dtype` is keyword-only.",
    },

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
        char version_str[64];
        snprintf(version_str, sizeof(version_str), "%d.%d.%d", SIMSIMD_VERSION_MAJOR, SIMSIMD_VERSION_MINOR,
                 SIMSIMD_VERSION_PATCH);
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

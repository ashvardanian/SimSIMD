/**
 *  @brief      Pure CPython bindings for SimSIMD.
 *  @file       lib.c
 *  @author     Ash Vardanian
 *  @date       January 1, 2023
 *  @copyright  Copyright (c) 2023
 *
 *  @section    Latency, Quality, and Arguments Parsing
 *
 *  The complexity of implementing high-quality CPython bindings is often underestimated.
 *  You can't use high-level wrappers like PyBind11 and NanoBind, and you shouldn't use
 *  SWIG-like messy toolchains. Most of them use expensive dynamic data-structures to map
 *  your callbacks to object/module properties, not taking advantage of the CPython API.
 *  They are prohibitively slow for low-latency operations like checking the length of a
 *  container, handling vectors, or strings.
 *
 *  Once you are down to the CPython API, there is a lot of boilerplate code to write and
 *  it's understandable that most people lazily use the `PyArg_ParseTupleAndKeywords` and
 *  `PyArg_ParseTuple` functions. Those, however, need to dynamically parse format specifier
 *  strings at runtime, which @b can't be fast by design! Moreover, they are not suitable
 *  for the Python's "Fast Calling Convention". In a typical scenario, a function is defined
 *  with `METH_VARARGS | METH_KEYWORDS` and has a signature like:
 *
 *  @code {.c}
 *      static PyObject* cdist(
 *          PyObject * self,
 *          PyObject * positional_args_tuple,
 *          PyObject * named_args_dict) {
 *          PyObject * a_obj, b_obj, metric_obj, out_obj, dtype_obj, out_dtype_obj, threads_obj;
 *          static char* names[] = {"a", "b", "metric", "threads", "dtype", "out_dtype", NULL};
 *          if (!PyArg_ParseTupleAndKeywords(
 *              positional_args_tuple, named_args_dict, "OO|s$Kss", names,
 *              &a_obj, &b_obj, &metric_str, &threads, &dtype_str, &out_dtype_str))
 *              return NULL;
 *          ...
 *  @endcode
 *
 *  This `cdist` example takes 2 positional, 1 positional or named, 3 named-only arguments.
 *  The alternative using the `METH_FASTCALL` is to use a function signature like:
 *
 *  @code {.c}
 *     static PyObject* cdist(
 *          PyObject * self,
 *          PyObject * const * args_c_array,    //! C array of `args_count` pointers
 *          Py_ssize_t const positional_args_count,   //! The `args_c_array` may be larger than this
 *          PyObject * args_names_tuple) {      //! May be smaller than `args_count`
 *          Py_ssize_t args_names_count = args_names_tuple ? PyTuple_Size(args_names_tuple) : 0;
 *          Py_ssize_t args_count = positional_args_count + args_names_count;
 *          ...
 *  @endcode
 *
 *  The positional elements are easy to access in that C array, but parsing the named arguments is tricky.
 *  There may be a case, when the call is ill-formed and more positional arguments are provided than needed.
 *
 *  @code {.py}
 *          cdist(a, b, "cos", "dos"):               //! positional_args_count == 4, args_names_count == 0
 *          cdist(a, b, "cos", metric="dos"):        //! positional_args_count == 3, args_names_count == 1
 *          cdist(a, b, metric="cos", metric="dos"): //! positional_args_count == 2, args_names_count == 2
 *  @endcode
 *
 *  If the same argument is provided twice, a @b `TypeError` is raised.
 *  If the argument is not found, a @b `KeyError` is raised.
 *
 *  https://ashvardanian.com/posts/discount-on-keyword-arguments-in-python/
 *
 *  @section    Buffer Protocol and NumPy Compatibility
 *
 *  Most modern Machine Learning frameworks struggle with the buffer protocol compatibility.
 *  At best, they provide zero-copy NumPy views of the underlying data, introducing unnecessary
 *  dependency on NumPy, a memory allocation for the wrapper, and a constraint on the supported
 *  numeric types. The last is a noticeable limitation, as both PyTorch and TensorFlow have
 *  richer type systems than NumPy.
 *
 *  You can't convert a PyTorch `Tensor` to a `memoryview` object.
 *  If you try to convert a `bf16` TensorFlow `Tensor` to a `memoryview` object, you will get an error:
 *
 *      ! ValueError: cannot include dtype 'E' in a buffer
 *
 *  Moreover, the CPython documentation and the NumPy documentation diverge on the format specifiers
 *  for the `typestr` and `format` data-type descriptor strings, making the development error-prone.
 *  At this point, SimSIMD seems to be @b the_only_package that at least attempts to provide interoperability.
 *
 *  https://numpy.org/doc/stable/reference/arrays.interface.html
 *  https://pearu.github.io/array_interface_pytorch.html
 *  https://github.com/pytorch/pytorch/issues/54138
 *  https://github.com/pybind/pybind11/issues/1908
 */
#include <math.h>

#if defined(__linux__)
#ifdef _OPENMP
#include <omp.h>
#endif
#endif

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define SIMSIMD_NDARRAY_MAX_RANK (PyBUF_MAX_NDIM)
#include <simsimd/simsimd.h>

/// @brief  Convenience wrapper for BLAS Level 1 interfaces to infer the layout of rank-1 and rank-2 tensors.
typedef struct VectorOrRowsArgument {
    char *start;
    size_t dimensions;
    size_t count;
    size_t stride;
    int rank;
    simsimd_datatype_t datatype;
} VectorOrRowsArgument;

typedef struct BufferOrScalarArgument {
    enum {
        UndefinedKind = 0,
        /// Just a `float` or an `int`
        ScalarKind,
        /// A single `float` or `int` buffer of any rank, but with a single element
        ScalarBufferKind,
        /// Any rank tensor with more than 1 element along any dimension
        BufferKind,
    } kind;
    /// Populated only for `ScalarKind` and `ScalarBufferKind` kinds.
    simsimd_f64_t as_f64;
    simsimd_u8_t as_scalar[8];
    /// The address of the buffer start, if the kind is `BufferKind` or `ScalarBufferKind`.
    /// Alternatively, points to the `&as_scalar` field for `ScalarKind` kind.
    char *as_buffer_start;
    Py_ssize_t as_buffer_dimensions;
    //? The "shape" and "strides" fields may seem redundant, as they are already part of the `Py_buffer`
    //? object, but we use them to normalize the representation in binary and ternary functions with
    //? non-trivial broadcasting rules.
    Py_ssize_t as_buffer_shape[PyBUF_MAX_NDIM];
    Py_ssize_t as_buffer_strides[PyBUF_MAX_NDIM];
    /// Defines the type of representation stored in the `as_scalar`
    /// and in the `as_buffer_start` contents.
    simsimd_datatype_t datatype;

} BufferOrScalarArgument;

/// @brief  Minimalistic and space-efficient representation of a rank-1 or rank-2 output tensor.
typedef struct DistancesTensor {
    PyObject_HEAD                    //
        simsimd_datatype_t datatype; // Any SimSIMD numeric type
    Py_ssize_t dimensions;           // Can be only 1 or 2 dimensions
    Py_ssize_t shape[2];             // Dimensions of the tensor
    Py_ssize_t strides[2];           // Strides for each dimension
    simsimd_distance_t start[];      // Variable length data aligned to 64-bit scalars
} DistancesTensor;

/// @brief  Generalized high-rank tensor alternative to NumPy, supporting up to 64 dimensions,
///         zero-copy views/slices, the Buffer Protocol, and faster iteration.
typedef struct NDArray {
    PyObject_HEAD                    //
        simsimd_datatype_t datatype; // Any SimSIMD numeric type
    Py_ssize_t ndim; //! Can be up to `PyBUF_MAX_NDIM` (often 64), but NumPy only supports 32 on most platforms!
    Py_ssize_t shape[PyBUF_MAX_NDIM];   // Dimensions of the tensor
    Py_ssize_t strides[PyBUF_MAX_NDIM]; // Strides for each dimension
    simsimd_distance_t start[];         // Variable length data aligned to 64-bit scalars
} NDArray;

/// @brief  Faster alternative to NumPy's `ndindex` object, supporting just as many dimensions,
///         as the `NDArray` object.
typedef struct NDIndex {
    PyObject_HEAD //
        simsimd_mdindices_t mdindices;
} NDIndex;

static int DistancesTensor_getbuffer(PyObject *export_from, Py_buffer *view, int flags);
static void DistancesTensor_releasebuffer(PyObject *export_from, Py_buffer *view);

static PyBufferProcs DistancesTensor_as_buffer = {
    .bf_getbuffer = DistancesTensor_getbuffer,
    .bf_releasebuffer = DistancesTensor_releasebuffer,
};

static PyTypeObject DistancesTensorType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "simsimd.DistancesTensor",
    .tp_doc = "Zero-copy view of a rank-1 or rank-2 tensor, compatible with NumPy",
    .tp_basicsize = sizeof(DistancesTensor),
    // Instead of using `simsimd_distance_t` for all the elements,
    // we use `char` to allow user to specify the datatype on `cdist`-like functions.
    .tp_itemsize = sizeof(char),
    .tp_as_buffer = &DistancesTensor_as_buffer,
};

static int NDArray_getbuffer(PyObject *export_from, Py_buffer *view, int flags);
static void NDArray_releasebuffer(PyObject *export_from, Py_buffer *view);
static PyObject *NDArray_get_shape(NDArray *self, void *closure);
static PyObject *NDArray_get_size(NDArray *self, void *closure);

static PyBufferProcs NDArray_as_buffer = {
    .bf_getbuffer = NDArray_getbuffer,
    .bf_releasebuffer = NDArray_releasebuffer,
};

static PyGetSetDef NDArray_getset[] = {
    {"shape", (getter)NDArray_get_shape, NULL, "Shape of the NDArray", NULL},
    {"size", (getter)NDArray_get_size, NULL, "Total number of elements in the NDArray", NULL},
    {NULL} // Sentinel
};

static PyTypeObject NDArrayType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "simsimd.NDArray",
    .tp_doc = "Zero-copy view of a high-rank tensor, compatible with NumPy", //
    .tp_basicsize = sizeof(NDArray),
    // Instead of using `simsimd_distance_t` for all the elements,
    // we use `char` to allow user to specify the datatype on `cdist`-like functions.
    .tp_itemsize = sizeof(char), //
    .tp_as_buffer = &NDArray_as_buffer,
    .tp_getset = NDArray_getset, // Add the getset array here
};

/// @brief  Global variable that caches the CPU capabilities, and is computed just onc, when the module is loaded.
simsimd_capability_t static_capabilities = simsimd_cap_serial_k;

/// @brief Helper method to check for string equality.
/// @return 1 if the strings are equal, 0 otherwise.
int same_string(char const *a, char const *b) { return strcmp(a, b) == 0; }

/// @brief Helper method to check if a logical datatype is complex and should be represented as two scalars.
/// @return 1 if the datatype is complex, 0 otherwise.
int is_complex(simsimd_datatype_t datatype) {
    return datatype == simsimd_f32c_k || datatype == simsimd_f64c_k || datatype == simsimd_f16c_k ||
           datatype == simsimd_bf16c_k;
}

/// @brief Converts a Python-ic datatype string to a logical datatype, normalizing the format.
/// @return `simsimd_datatype_unknown_k` if the datatype is not supported, otherwise the logical datatype.
/// @see https://docs.python.org/3/library/struct.html#format-characters
/// @see https://numpy.org/doc/stable/reference/arrays.interface.html
/// @see https://github.com/pybind/pybind11/issues/1908
simsimd_datatype_t python_string_to_datatype(char const *name) {
    // Floating-point numbers:
    if (same_string(name, "float32") || same_string(name, "f32") || // SimSIMD-specific
        same_string(name, "f4") || same_string(name, "<f4") ||      // Sized float
        same_string(name, "f") || same_string(name, "<f"))          // Named type
        return simsimd_f32_k;
    else if (same_string(name, "float16") || same_string(name, "f16") || // SimSIMD-specific
             same_string(name, "f2") || same_string(name, "<f2") ||      // Sized float
             same_string(name, "e") || same_string(name, "<e"))          // Named type
        return simsimd_f16_k;
    else if (same_string(name, "float64") || same_string(name, "f64") || // SimSIMD-specific
             same_string(name, "f8") || same_string(name, "<f8") ||      // Sized float
             same_string(name, "d") || same_string(name, "<d"))          // Named type
        return simsimd_f64_k;
    //? The exact format is not defined, but TensorFlow uses 'E' for `bf16`?!
    else if (same_string(name, "bfloat16") || same_string(name, "bf16")) // SimSIMD-specific
        return simsimd_bf16_k;

    // Complex numbers:
    else if (same_string(name, "complex64") ||                                             // SimSIMD-specific
             same_string(name, "F4") || same_string(name, "<F4") ||                        // Sized complex
             same_string(name, "Zf") || same_string(name, "F") || same_string(name, "<F")) // Named type
        return simsimd_f32c_k;
    else if (same_string(name, "complex128") ||                                            // SimSIMD-specific
             same_string(name, "F8") || same_string(name, "<F8") ||                        // Sized complex
             same_string(name, "Zd") || same_string(name, "D") || same_string(name, "<D")) // Named type
        return simsimd_f64c_k;
    else if (same_string(name, "complex32") ||                                             // SimSIMD-specific
             same_string(name, "F2") || same_string(name, "<F2") ||                        // Sized complex
             same_string(name, "Ze") || same_string(name, "E") || same_string(name, "<E")) // Named type
        return simsimd_f16c_k;
    //? The exact format is not defined, but TensorFlow uses 'E' for `bf16`?!
    else if (same_string(name, "bcomplex32")) // SimSIMD-specific
        return simsimd_bf16c_k;

    //! Boolean values:
    else if (same_string(name, "bin8") || same_string(name, "bit8") || // SimSIMD-specific
             same_string(name, "c"))                                   // Named type
        return simsimd_b8_k;

    // Signed integers:
    else if (same_string(name, "int8") ||                                                       // SimSIMD-specific
             same_string(name, "i1") || same_string(name, "|i1") || same_string(name, "<i1") || // Sized integer
             same_string(name, "b") || same_string(name, "<b"))                                 // Named type
        return simsimd_i8_k;
    else if (same_string(name, "int16") ||                                                      // SimSIMD-specific
             same_string(name, "i2") || same_string(name, "|i2") || same_string(name, "<i2") || // Sized integer
             same_string(name, "h") || same_string(name, "<h"))                                 // Named type
        return simsimd_i16_k;

        //! On Windows the 32-bit and 64-bit signed integers will have different specifiers:
        //! https://github.com/pybind/pybind11/issues/1908
#if defined(_MSC_VER) || defined(__i386__)
    else if (same_string(name, "int32") ||                                                      // SimSIMD-specific
             same_string(name, "i4") || same_string(name, "|i4") || same_string(name, "<i4") || // Sized integer
             same_string(name, "l") || same_string(name, "<l"))                                 // Named type
        return simsimd_i32_k;
    else if (same_string(name, "int64") ||                                                      // SimSIMD-specific
             same_string(name, "i8") || same_string(name, "|i8") || same_string(name, "<i8") || // Sized integer
             same_string(name, "q") || same_string(name, "<q"))                                 // Named type
        return simsimd_i64_k;
#else // On Linux and macOS:
    else if (same_string(name, "int32") ||                                                      // SimSIMD-specific
             same_string(name, "i4") || same_string(name, "|i4") || same_string(name, "<i4") || // Sized integer
             same_string(name, "i") || same_string(name, "<i"))                                 // Named type
        return simsimd_i32_k;
    else if (same_string(name, "int64") ||                                                      // SimSIMD-specific
             same_string(name, "i8") || same_string(name, "|i8") || same_string(name, "<i8") || // Sized integer
             same_string(name, "l") || same_string(name, "<l"))                                 // Named type
        return simsimd_i64_k;
#endif

    // Unsigned integers:
    else if (same_string(name, "uint8") ||                                                      // SimSIMD-specific
             same_string(name, "u1") || same_string(name, "|u1") || same_string(name, "<u1") || // Sized integer
             same_string(name, "B") || same_string(name, "<B"))                                 // Named type
        return simsimd_u8_k;
    else if (same_string(name, "uint16") ||                                                     // SimSIMD-specific
             same_string(name, "u2") || same_string(name, "|u2") || same_string(name, "<u2") || // Sized integer
             same_string(name, "H") || same_string(name, "<H"))                                 // Named type
        return simsimd_u16_k;

        //! On Windows the 32-bit and 64-bit unsigned integers will have different specifiers:
        //! https://github.com/pybind/pybind11/issues/1908
#if defined(_MSC_VER) || defined(__i386__)
    else if (same_string(name, "uint32") ||                                                     // SimSIMD-specific
             same_string(name, "i4") || same_string(name, "|i4") || same_string(name, "<i4") || // Sized integer
             same_string(name, "L") || same_string(name, "<L"))                                 // Named type
        return simsimd_u32_k;
    else if (same_string(name, "uint64") ||                                                     // SimSIMD-specific
             same_string(name, "i8") || same_string(name, "|i8") || same_string(name, "<i8") || // Sized integer
             same_string(name, "Q") || same_string(name, "<Q"))                                 // Named type
        return simsimd_u64_k;
#else // On Linux and macOS:
    else if (same_string(name, "uint32") ||                                                     // SimSIMD-specific
             same_string(name, "u4") || same_string(name, "|u4") || same_string(name, "<u4") || // Sized integer
             same_string(name, "I") || same_string(name, "<I"))                                 // Named type
        return simsimd_u32_k;
    else if (same_string(name, "uint64") ||                                                     // SimSIMD-specific
             same_string(name, "u8") || same_string(name, "|u8") || same_string(name, "<u8") || // Sized integer
             same_string(name, "L") || same_string(name, "<L"))                                 // Named type
        return simsimd_u64_k;
#endif

    else
        return simsimd_datatype_unknown_k;
}

/// @brief Returns the Python string representation of a datatype for the buffer protocol.
/// @param dtype Logical datatype, can be complex.
/// @return "unknown" if the datatype is not supported, otherwise a string.
/// @see https://docs.python.org/3/library/struct.html#format-characters
char const *datatype_to_python_string(simsimd_datatype_t dtype) {
    switch (dtype) {
        // Floating-point numbers:
    case simsimd_f64_k: return "d";
    case simsimd_f32_k: return "f";
    case simsimd_f16_k: return "e";
    // Complex numbers:
    case simsimd_f64c_k: return "Zd";
    case simsimd_f32c_k: return "Zf";
    case simsimd_f16c_k: return "Ze";
    // Boolean values:
    case simsimd_b8_k: return "c";
    // Signed integers:
    case simsimd_i8_k: return "b";
    case simsimd_i16_k: return "h";
    case simsimd_i32_k: return "i";
    case simsimd_i64_k: return "q";
    // Unsigned integers:
    case simsimd_u8_k: return "B";
    case simsimd_u16_k: return "H";
    case simsimd_u32_k: return "I";
    case simsimd_u64_k: return "Q";
    // Other:
    default: return "unknown";
    }
}

/// @brief Estimate the number of bytes per element for a given datatype.
/// @param dtype Logical datatype, can be complex.
/// @return Zero if the datatype is not supported, positive integer otherwise.
size_t bytes_per_datatype(simsimd_datatype_t dtype) {
    switch (dtype) {
    case simsimd_f64_k: return sizeof(simsimd_f64_t);
    case simsimd_f32_k: return sizeof(simsimd_f32_t);
    case simsimd_f16_k: return sizeof(simsimd_f16_t);
    case simsimd_bf16_k: return sizeof(simsimd_bf16_t);
    case simsimd_f64c_k: return sizeof(simsimd_f64_t) * 2;
    case simsimd_f32c_k: return sizeof(simsimd_f32_t) * 2;
    case simsimd_f16c_k: return sizeof(simsimd_f16_t) * 2;
    case simsimd_bf16c_k: return sizeof(simsimd_bf16_t) * 2;
    case simsimd_b8_k: return sizeof(simsimd_b8_t);
    case simsimd_i8_k: return sizeof(simsimd_i8_t);
    case simsimd_u8_k: return sizeof(simsimd_u8_t);
    case simsimd_i16_k: return sizeof(simsimd_i16_t);
    case simsimd_u16_k: return sizeof(simsimd_u16_t);
    case simsimd_i32_k: return sizeof(simsimd_i32_t);
    case simsimd_u32_k: return sizeof(simsimd_u32_t);
    case simsimd_i64_k: return sizeof(simsimd_i64_t);
    case simsimd_u64_k: return sizeof(simsimd_u64_t);
    default: return 0;
    }
}

/// @brief Copy a distance to a target datatype, downcasting if necessary.
/// @return 1 if the cast was successful, 0 if the target datatype is not supported.
int cast_distance(simsimd_distance_t distance, simsimd_datatype_t target_dtype, void *target_ptr, size_t offset) {
    _SIMSIMD_STATIC_ASSERT(sizeof(simsimd_distance_t) == sizeof(simsimd_f64_t), distance_size_mismatch);
    switch (target_dtype) {
    case simsimd_f64c_k: ((simsimd_f64_t *)target_ptr)[offset] = (simsimd_f64_t)distance; return 1;
    case simsimd_f64_k: ((simsimd_f64_t *)target_ptr)[offset] = (simsimd_f64_t)distance; return 1;
    case simsimd_f32c_k: ((simsimd_f32_t *)target_ptr)[offset] = (simsimd_f32_t)distance; return 1;
    case simsimd_f32_k: ((simsimd_f32_t *)target_ptr)[offset] = (simsimd_f32_t)distance; return 1;
    case simsimd_f16c_k: _simsimd_f64_to_f16(&distance, (simsimd_f16_t *)target_ptr + offset); return 1;
    case simsimd_f16_k: _simsimd_f64_to_f16(&distance, (simsimd_f16_t *)target_ptr + offset); return 1;
    case simsimd_bf16c_k: _simsimd_f64_to_bf16(&distance, (simsimd_bf16_t *)target_ptr + offset); return 1;
    case simsimd_bf16_k: _simsimd_f64_to_bf16(&distance, (simsimd_bf16_t *)target_ptr + offset); return 1;
    case simsimd_i8_k: ((simsimd_i8_t *)target_ptr)[offset] = (simsimd_i8_t)distance; return 1;
    case simsimd_u8_k: ((simsimd_u8_t *)target_ptr)[offset] = (simsimd_u8_t)distance; return 1;
    case simsimd_i16_k: ((simsimd_i16_t *)target_ptr)[offset] = (simsimd_i16_t)distance; return 1;
    case simsimd_u16_k: ((simsimd_u16_t *)target_ptr)[offset] = (simsimd_u16_t)distance; return 1;
    case simsimd_i32_k: ((simsimd_i32_t *)target_ptr)[offset] = (simsimd_i32_t)distance; return 1;
    case simsimd_u32_k: ((simsimd_u32_t *)target_ptr)[offset] = (simsimd_u32_t)distance; return 1;
    case simsimd_i64_k: ((simsimd_i64_t *)target_ptr)[offset] = (simsimd_i64_t)distance; return 1;
    case simsimd_u64_k: ((simsimd_u64_t *)target_ptr)[offset] = (simsimd_u64_t)distance; return 1;
    default: return 0;
    }
}

simsimd_kernel_kind_t python_string_to_kernel_kind(char const *name) {
    if (same_string(name, "euclidean") || same_string(name, "l2")) return simsimd_euclidean_k;
    else if (same_string(name, "sqeuclidean") || same_string(name, "l2sq"))
        return simsimd_sqeuclidean_k;
    else if (same_string(name, "dot") || same_string(name, "inner"))
        return simsimd_dot_k;
    else if (same_string(name, "vdot"))
        return simsimd_vdot_k;
    else if (same_string(name, "cosine") || same_string(name, "cos"))
        return simsimd_cosine_k;
    else if (same_string(name, "jaccard"))
        return simsimd_jaccard_k;
    else if (same_string(name, "kullbackleibler") || same_string(name, "kl"))
        return simsimd_kl_k;
    else if (same_string(name, "jensenshannon") || same_string(name, "js"))
        return simsimd_js_k;
    else if (same_string(name, "hamming"))
        return simsimd_hamming_k;
    else if (same_string(name, "jaccard"))
        return simsimd_jaccard_k;
    else if (same_string(name, "bilinear"))
        return simsimd_bilinear_k;
    else if (same_string(name, "mahalanobis"))
        return simsimd_mahalanobis_k;
    else
        return simsimd_kernel_unknown_k;
}

/// @brief Check if a metric is commutative, i.e., if `metric(a, b) == metric(b, a)`.
/// @return 1 if the metric is commutative, 0 otherwise.
int kernel_is_commutative(simsimd_kernel_kind_t kind) {
    switch (kind) {
    case simsimd_kl_k: return 0;
    case simsimd_bilinear_k: return 0; //? The kernel is commutative if only the matrix is symmetric
    default: return 1;
    }
}

static char const doc_enable_capability[] = //
    "Enable a specific SIMD kernel family.\n"
    "\n"
    "Args:\n"
    "    capability (str): The name of the SIMD feature to enable (e.g., 'haswell').";

static PyObject *api_enable_capability(PyObject *self, PyObject *cap_name_obj) {
    char const *cap_name = PyUnicode_AsUTF8(cap_name_obj);
    if (!cap_name) {
        PyErr_SetString(PyExc_TypeError, "Capability name must be a string");
        return NULL;
    }

    if (same_string(cap_name, "neon")) { static_capabilities |= simsimd_cap_neon_k; }
    else if (same_string(cap_name, "neon_f16")) { static_capabilities |= simsimd_cap_neon_f16_k; }
    else if (same_string(cap_name, "neon_bf16")) { static_capabilities |= simsimd_cap_neon_bf16_k; }
    else if (same_string(cap_name, "neon_i8")) { static_capabilities |= simsimd_cap_neon_i8_k; }
    else if (same_string(cap_name, "sve")) { static_capabilities |= simsimd_cap_sve_k; }
    else if (same_string(cap_name, "sve_f16")) { static_capabilities |= simsimd_cap_sve_f16_k; }
    else if (same_string(cap_name, "sve_bf16")) { static_capabilities |= simsimd_cap_sve_bf16_k; }
    else if (same_string(cap_name, "sve_i8")) { static_capabilities |= simsimd_cap_sve_i8_k; }
    else if (same_string(cap_name, "haswell")) { static_capabilities |= simsimd_cap_haswell_k; }
    else if (same_string(cap_name, "skylake")) { static_capabilities |= simsimd_cap_skylake_k; }
    else if (same_string(cap_name, "ice")) { static_capabilities |= simsimd_cap_ice_k; }
    else if (same_string(cap_name, "genoa")) { static_capabilities |= simsimd_cap_genoa_k; }
    else if (same_string(cap_name, "sapphire")) { static_capabilities |= simsimd_cap_sapphire_k; }
    else if (same_string(cap_name, "turin")) { static_capabilities |= simsimd_cap_turin_k; }
    else if (same_string(cap_name, "sierra")) { static_capabilities |= simsimd_cap_sierra_k; }
    else if (same_string(cap_name, "serial")) {
        PyErr_SetString(PyExc_ValueError, "Can't change the serial functionality");
        return NULL;
    }
    else {
        PyErr_SetString(PyExc_ValueError, "Unknown capability");
        return NULL;
    }

    Py_RETURN_NONE;
}

static char const doc_disable_capability[] = //
    "Disable a specific SIMD kernel family.\n"
    "\n"
    "Args:\n"
    "    capability (str): The name of the SIMD feature to disable (e.g., 'haswell').";

static PyObject *api_disable_capability(PyObject *self, PyObject *cap_name_obj) {
    char const *cap_name = PyUnicode_AsUTF8(cap_name_obj);
    if (!cap_name) {
        PyErr_SetString(PyExc_TypeError, "Capability name must be a string");
        return NULL;
    }

    if (same_string(cap_name, "neon")) { static_capabilities &= ~simsimd_cap_neon_k; }
    else if (same_string(cap_name, "neon_f16")) { static_capabilities &= ~simsimd_cap_neon_f16_k; }
    else if (same_string(cap_name, "neon_bf16")) { static_capabilities &= ~simsimd_cap_neon_bf16_k; }
    else if (same_string(cap_name, "neon_i8")) { static_capabilities &= ~simsimd_cap_neon_i8_k; }
    else if (same_string(cap_name, "sve")) { static_capabilities &= ~simsimd_cap_sve_k; }
    else if (same_string(cap_name, "sve_f16")) { static_capabilities &= ~simsimd_cap_sve_f16_k; }
    else if (same_string(cap_name, "sve_bf16")) { static_capabilities &= ~simsimd_cap_sve_bf16_k; }
    else if (same_string(cap_name, "sve_i8")) { static_capabilities &= ~simsimd_cap_sve_i8_k; }
    else if (same_string(cap_name, "haswell")) { static_capabilities &= ~simsimd_cap_haswell_k; }
    else if (same_string(cap_name, "skylake")) { static_capabilities &= ~simsimd_cap_skylake_k; }
    else if (same_string(cap_name, "ice")) { static_capabilities &= ~simsimd_cap_ice_k; }
    else if (same_string(cap_name, "genoa")) { static_capabilities &= ~simsimd_cap_genoa_k; }
    else if (same_string(cap_name, "sapphire")) { static_capabilities &= ~simsimd_cap_sapphire_k; }
    else if (same_string(cap_name, "turin")) { static_capabilities &= ~simsimd_cap_turin_k; }
    else if (same_string(cap_name, "sierra")) { static_capabilities &= ~simsimd_cap_sierra_k; }
    else if (same_string(cap_name, "serial")) {
        PyErr_SetString(PyExc_ValueError, "Can't change the serial functionality");
        return NULL;
    }
    else {
        PyErr_SetString(PyExc_ValueError, "Unknown capability");
        return NULL;
    }

    Py_RETURN_NONE;
}

static char const doc_get_capabilities[] = //
    "Get the current hardware SIMD capabilities as a dictionary of feature flags.\n"
    "On x86 includes: 'serial', 'haswell', 'skylake', 'ice', 'genoa', 'sapphire', 'turin', 'sierra'.\n"
    "On Arm includes: 'serial', 'neon', 'sve', 'sve2', and their specialized extensions.\n";

static PyObject *api_get_capabilities(PyObject *self) {
    simsimd_capability_t caps = static_capabilities;
    PyObject *cap_dict = PyDict_New();
    if (!cap_dict) return NULL;

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
    ADD_CAP(turin);
    ADD_CAP(sierra);

#undef ADD_CAP

    return cap_dict;
}

/// @brief Unpacks a Python tensor object into a C structure.
/// @return 1 on success, 0 otherwise.
int parse_rows(PyObject *tensor, Py_buffer *buffer, VectorOrRowsArgument *parsed) {
    if (PyObject_GetBuffer(tensor, buffer, PyBUF_STRIDES | PyBUF_FORMAT) != 0) {
        PyErr_SetString(PyExc_TypeError, "Tensors must support buffer protocol");
        return 0;
    }
    parsed->start = buffer->buf;
    parsed->datatype = python_string_to_datatype(buffer->format);
    if (parsed->datatype == simsimd_datatype_unknown_k) {
        PyErr_Format(PyExc_ValueError, "Unsupported '%s' datatype specifier", buffer->format);
        PyBuffer_Release(buffer);
        return 0;
    }

    parsed->rank = buffer->ndim;
    if (buffer->ndim == 1) {
        if (buffer->strides[0] > buffer->itemsize) {
            PyErr_SetString(PyExc_ValueError, "Input vectors must be contiguous, check with `X.__array_interface__`");
            PyBuffer_Release(buffer);
            return 0;
        }
        parsed->dimensions = buffer->shape[0];
        parsed->count = 1;
        parsed->stride = 0;
    }
    else if (buffer->ndim == 2) {
        if (buffer->strides[1] > buffer->itemsize) {
            PyErr_SetString(PyExc_ValueError, "Input rows must be contiguous, check with `X.__array_interface__`");
            PyBuffer_Release(buffer);
            return 0;
        }
        parsed->dimensions = buffer->shape[1];
        parsed->count = buffer->shape[0];
        parsed->stride = buffer->strides[0];
    }
    else {
        PyErr_SetString(PyExc_ValueError, "Input tensors must be 1D or 2D");
        PyBuffer_Release(buffer);
        return 0;
    }

    // We handle complex numbers differently
    if (is_complex(parsed->datatype)) parsed->dimensions *= 2;
    return 1;
}

/// @brief Unpacks a Python tensor object into a C structure.
/// @return 1 on success, 0 otherwise.
int parse_tensor(PyObject *tensor, Py_buffer *buffer, simsimd_datatype_t *dtype) {
    if (PyObject_GetBuffer(tensor, buffer, PyBUF_STRIDES | PyBUF_FORMAT) != 0) {
        PyErr_SetString(PyExc_TypeError, "Tensors must support buffer protocol");
        return 0;
    }

    *dtype = python_string_to_datatype(buffer->format);
    if (*dtype == simsimd_datatype_unknown_k) {
        PyErr_Format(PyExc_ValueError, "Unsupported '%s' datatype specifier", buffer->format);
        PyBuffer_Release(buffer);
        return 0;
    }

    return 1;
}

int parse_buffer_or_scalar_argument(PyObject *obj, Py_buffer *buffer, BufferOrScalarArgument *parsed) {

    if (PyFloat_Check(obj)) {
        parsed->kind = ScalarKind;
        parsed->as_buffer_start = (char *)&parsed->as_scalar;
        parsed->as_buffer_dimensions = 1;
        parsed->as_buffer_shape[0] = 1;
        parsed->as_buffer_strides[0] = 0;
        // Return a C `double` representation of the contents of `obj`.
        // If `obj` is not a Python floating-point object but has a `__float__()` method,
        // this method will first be called to convert `obj` into a float.
        simsimd_f64_t as_float = PyFloat_AsDouble(obj);
        parsed->as_f64 = as_float;
        // Check if we convert to a smaller floating-point type, without the loss of precision.
        if (SIMSIMD_NATIVE_F16 && as_float == (simsimd_f64_t)(simsimd_f16_t)as_float) {
            simsimd_f16_t as_f16 = (simsimd_f16_t)as_float;
            memcpy(parsed->as_scalar, &as_f16, sizeof(simsimd_f16_t));
            parsed->datatype = simsimd_f16_k;
        }
        else if (as_float == (simsimd_f64_t)(simsimd_f32_t)as_float) {
            simsimd_f32_t as_f32 = (simsimd_f32_t)as_float;
            memcpy(parsed->as_scalar, &as_f32, sizeof(simsimd_f32_t));
            parsed->datatype = simsimd_f32_k;
        }
        else {
            memcpy(parsed->as_scalar, &as_float, sizeof(simsimd_f64_t));
            parsed->datatype = simsimd_f64_k;
        }
        return 1;
    }
    else if (PyLong_Check(obj)) {
        int did_overflow = 0;
        simsimd_i64_t as_integral = PyLong_AsLongLongAndOverflow(obj, &did_overflow);
        if (did_overflow) {
            PyErr_SetString(PyExc_ValueError, "Integer overflow");
            return 0;
        }
        parsed->kind = ScalarKind;
        parsed->as_buffer_start = (char *)&parsed->as_scalar;
        parsed->as_buffer_dimensions = 1;
        parsed->as_buffer_shape[0] = 1;
        parsed->as_buffer_strides[0] = 0;
        parsed->as_f64 = as_integral;
        // Check for smaller unsigned integer types and store in `as_scalar`
        if (as_integral == (simsimd_u64_t)(simsimd_u8_t)as_integral) {
            simsimd_u8_t as_u8 = (simsimd_u8_t)as_integral;
            memcpy(parsed->as_scalar, &as_u8, sizeof(simsimd_u8_t));
            parsed->datatype = simsimd_u8_k;
        }
        else if (as_integral == (simsimd_u64_t)(simsimd_u16_t)as_integral) {
            simsimd_u16_t as_u16 = (simsimd_u16_t)as_integral;
            memcpy(parsed->as_scalar, &as_u16, sizeof(simsimd_u16_t));
            parsed->datatype = simsimd_u16_k;
        }
        else if (as_integral == (simsimd_u64_t)(simsimd_u32_t)as_integral) {
            simsimd_u32_t as_u32 = (simsimd_u32_t)as_integral;
            memcpy(parsed->as_scalar, &as_u32, sizeof(simsimd_u32_t));
            parsed->datatype = simsimd_u32_k;
        }
        else if (as_integral == (simsimd_i64_t)(simsimd_i8_t)as_integral) {
            simsimd_i8_t as_i8 = (simsimd_i8_t)as_integral;
            memcpy(parsed->as_scalar, &as_i8, sizeof(simsimd_i8_t));
            parsed->datatype = simsimd_i8_k;
        }
        else if (as_integral == (simsimd_i64_t)(simsimd_i16_t)as_integral) {
            simsimd_i16_t as_i16 = (simsimd_i16_t)as_integral;
            memcpy(parsed->as_scalar, &as_i16, sizeof(simsimd_i16_t));
            parsed->datatype = simsimd_i16_k;
        }
        else if (as_integral == (simsimd_i64_t)(simsimd_i32_t)as_integral) {
            simsimd_i32_t as_i32 = (simsimd_i32_t)as_integral;
            memcpy(parsed->as_scalar, &as_i32, sizeof(simsimd_i32_t));
            parsed->datatype = simsimd_i32_k;
        }
        else {
            memcpy(parsed->as_scalar, &as_integral, sizeof(simsimd_i64_t));
            parsed->datatype = simsimd_i64_k;
        }
        return 1;
    }
    else if (PyObject_CheckBuffer(obj)) {
        if (!parse_tensor(obj, buffer, &parsed->datatype)) return 0;
        // If the tensor contains just one element, regardless of the shape,
        // we must treat it as a scalar, similar to NumPy.
        if (buffer->len == buffer->itemsize) {
            memcpy(&parsed->as_scalar, buffer->buf, buffer->itemsize);
            parsed->kind = ScalarBufferKind;
            parsed->as_buffer_start = (char *)&parsed->as_scalar;
        }
        else {
            parsed->kind = BufferKind;
            parsed->as_buffer_start = buffer->buf;
        }
        parsed->as_buffer_dimensions = buffer->ndim;
        for (Py_ssize_t i = 0; i < buffer->ndim; i++) {
            parsed->as_buffer_shape[i] = buffer->shape[i];
            parsed->as_buffer_strides[i] = buffer->strides[i];
        }
        return 1;
    }
    else {
        PyErr_SetString(PyExc_TypeError, "Argument must be a scalar, a buffer, or a buffer-like object");
        return 0;
    }
}

static int DistancesTensor_getbuffer(PyObject *export_from, Py_buffer *view, int flags) {
    DistancesTensor *tensor = (DistancesTensor *)export_from;
    size_t const total_items = tensor->shape[0] * tensor->shape[1];
    size_t const item_size = bytes_per_datatype(tensor->datatype);

    view->buf = &tensor->start[0];
    view->obj = (PyObject *)tensor;
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

static void DistancesTensor_releasebuffer(PyObject *export_from, Py_buffer *view) {
    //! This function MUST NOT decrement view->obj, since that is done automatically in PyBuffer_Release().
    //! https://docs.python.org/3/c-api/typeobj.html#c.PyBufferProcs.bf_releasebuffer
}

static int NDArray_getbuffer(PyObject *export_from, Py_buffer *view, int flags) {
    NDArray *tensor = (NDArray *)export_from;
    Py_ssize_t const item_size = (Py_ssize_t)bytes_per_datatype(tensor->datatype);
    Py_ssize_t total_items = 1;
    for (Py_ssize_t i = 0; i < tensor->ndim; ++i) total_items *= tensor->shape[i];

    view->buf = &tensor->start[0];
    view->obj = (PyObject *)tensor;
    view->len = item_size * total_items;
    view->readonly = 0;
    view->itemsize = item_size;
    view->format = datatype_to_python_string(tensor->datatype);
    view->ndim = tensor->ndim;
    view->shape = &tensor->shape[0];
    view->strides = &tensor->strides[0];
    view->suboffsets = NULL;
    view->internal = NULL;

    Py_INCREF(tensor);
    return 0;
}

static void NDArray_releasebuffer(PyObject *export_from, Py_buffer *view) {
    //! This function MUST NOT decrement view->obj, since that is done automatically in PyBuffer_Release().
    //! https://docs.python.org/3/c-api/typeobj.html#c.PyBufferProcs.bf_releasebuffer
}

static PyObject *NDArray_get_shape(NDArray *self, void *closure) {
    PyObject *shape_tuple = PyTuple_New(self->ndim);
    if (!shape_tuple) return NULL;
    for (Py_ssize_t i = 0; i < self->ndim; i++) PyTuple_SET_ITEM(shape_tuple, i, PyLong_FromSsize_t(self->shape[i]));
    return shape_tuple;
}

static PyObject *NDArray_get_size(NDArray *self, void *closure) {
    Py_ssize_t total_items = 1;
    for (Py_ssize_t i = 0; i < self->ndim; i++) total_items *= self->shape[i];
    return PyLong_FromSsize_t(total_items);
}

static PyObject *implement_dense_metric( //
    simsimd_kernel_kind_t kernel_kind,   //
    PyObject *const *args, Py_ssize_t const positional_args_count, PyObject *args_names_tuple) {

    PyObject *return_obj = NULL;

    // This function accepts up to 5 arguments:
    PyObject *a_obj = NULL;         // Required object, positional-only
    PyObject *b_obj = NULL;         // Required object, positional-only
    PyObject *dtype_obj = NULL;     // Optional object, "dtype" keyword or positional
    PyObject *out_obj = NULL;       // Optional object, "out" keyword-only
    PyObject *out_dtype_obj = NULL; // Optional object, "out_dtype" keyword-only

    // Once parsed, the arguments will be stored in these variables:
    char const *dtype_str = NULL, *out_dtype_str = NULL;
    simsimd_datatype_t dtype = simsimd_datatype_unknown_k, out_dtype = simsimd_datatype_unknown_k;
    Py_buffer a_buffer, b_buffer, out_buffer;
    VectorOrRowsArgument a_parsed, b_parsed, out_parsed;
    memset(&a_buffer, 0, sizeof(Py_buffer));
    memset(&b_buffer, 0, sizeof(Py_buffer));
    memset(&out_buffer, 0, sizeof(Py_buffer));

    // Parse the arguments
    Py_ssize_t const args_names_count = args_names_tuple ? PyTuple_Size(args_names_tuple) : 0;
    Py_ssize_t const args_count = positional_args_count + args_names_count;
    if (args_count < 2 || args_count > 5) {
        PyErr_Format(PyExc_TypeError, "Function expects 2-5 arguments, got %zd", args_count);
        return NULL;
    }
    if (positional_args_count > 3) {
        PyErr_Format(PyExc_TypeError, "Only first 3 arguments can be positional, received %zd", positional_args_count);
        return NULL;
    }

    // Positional-only arguments (first and second matrix)
    a_obj = args[0];
    b_obj = args[1];

    // Positional or keyword arguments (dtype)
    if (positional_args_count == 3) dtype_obj = args[2];

    // The rest of the arguments must be checked in the keyword dictionary:
    for (Py_ssize_t args_names_tuple_progress = 0, args_progress = positional_args_count;
         args_names_tuple_progress < args_names_count; ++args_progress, ++args_names_tuple_progress) {
        PyObject *const key = PyTuple_GetItem(args_names_tuple, args_names_tuple_progress);
        PyObject *const value = args[args_progress];
        if (PyUnicode_CompareWithASCIIString(key, "dtype") == 0 && !dtype_obj) { dtype_obj = value; }
        else if (PyUnicode_CompareWithASCIIString(key, "out") == 0 && !out_obj) { out_obj = value; }
        else if (PyUnicode_CompareWithASCIIString(key, "out_dtype") == 0 && !out_dtype_obj) { out_dtype_obj = value; }
        else {
            PyErr_Format(PyExc_TypeError, "Got unexpected keyword argument: %S", key);
            return NULL;
        }
    }

    // Convert `dtype_obj` to `dtype_str` and to `dtype`
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

    // Convert `out_dtype_obj` to `out_dtype_str` and to `out_dtype`
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

    // Convert `a_obj` to `a_buffer` and to `a_parsed`. Same for `b_obj` and `out_obj`.
    if (!parse_rows(a_obj, &a_buffer, &a_parsed) || !parse_rows(b_obj, &b_buffer, &b_parsed)) return NULL;
    if (out_obj && !parse_rows(out_obj, &out_buffer, &out_parsed)) return NULL;

    // Check dimensions
    if (a_parsed.dimensions != b_parsed.dimensions) {
        PyErr_SetString(PyExc_ValueError, "Vector dimensions don't match");
        goto cleanup;
    }
    if (a_parsed.count == 0 || b_parsed.count == 0) {
        PyErr_SetString(PyExc_ValueError, "Collections can't be empty");
        goto cleanup;
    }
    if (a_parsed.count > 1 && b_parsed.count > 1 && a_parsed.count != b_parsed.count) {
        PyErr_SetString(PyExc_ValueError, "Collections must have the same number of elements or just one element");
        goto cleanup;
    }

    // Check data types
    if (a_parsed.datatype != b_parsed.datatype || //
        a_parsed.datatype == simsimd_datatype_unknown_k || b_parsed.datatype == simsimd_datatype_unknown_k) {
        PyErr_SetString(PyExc_TypeError,
                        "Input tensors must have matching datatypes, check with `X.__array_interface__`");
        goto cleanup;
    }
    if (dtype == simsimd_datatype_unknown_k) dtype = a_parsed.datatype;

    // Inference order for the output type:
    // 1. `out_dtype` named argument, if defined
    // 2. `out.dtype` attribute, if `out` is passed
    // 3. double precision float (or its complex variant)
    if (out_dtype == simsimd_datatype_unknown_k) {
        if (out_obj) { out_dtype = out_parsed.datatype; }
        else { out_dtype = is_complex(dtype) ? simsimd_f64c_k : simsimd_f64_k; }
    }

    // Make sure the return datatype is complex if the input datatype is complex, and the same for real numbers
    if (out_dtype != simsimd_datatype_unknown_k) {
        if (is_complex(dtype) != is_complex(out_dtype)) {
            PyErr_SetString(
                PyExc_ValueError,
                "If the input datatype is complex, the return datatype must be complex, and same for real.");
            goto cleanup;
        }
    }

    // Check if the downcasting to provided datatype is supported
    {
        char returned_buffer_example[8];
        if (!cast_distance(0, out_dtype, &returned_buffer_example, 0)) {
            PyErr_SetString(PyExc_ValueError, "Exporting to the provided datatype is not supported");
            goto cleanup;
        }
    }

    // Look up the metric and the capability
    simsimd_dense_metric_t metric = NULL;
    simsimd_capability_t capability = simsimd_cap_serial_k;
    simsimd_find_kernel(kernel_kind, dtype, static_capabilities, simsimd_cap_any_k, (simsimd_kernel_punned_t *)&metric,
                        &capability);
    if (!metric) {
        PyErr_Format( //
            PyExc_LookupError,
            "Unsupported metric '%c' and datatype combination across vectors ('%s'/'%s' and '%s'/'%s') and "
            "`dtype` override ('%s'/'%s')",
            kernel_kind,                                                                             //
            a_buffer.format ? a_buffer.format : "nil", datatype_to_python_string(a_parsed.datatype), //
            b_buffer.format ? b_buffer.format : "nil", datatype_to_python_string(b_parsed.datatype), //
            dtype_str ? dtype_str : "nil", datatype_to_python_string(dtype));
        goto cleanup;
    }

    // If the distance is computed between two vectors, rather than matrices, return a scalar
    int const dtype_is_complex = is_complex(dtype);
    if (a_parsed.rank == 1 && b_parsed.rank == 1) {
        // For complex numbers we are going to use `PyComplex_FromDoubles`.
        if (dtype_is_complex) {
            simsimd_distance_t distances[2];
            metric(a_parsed.start, b_parsed.start, a_parsed.dimensions, distances);
            return_obj = PyComplex_FromDoubles(distances[0], distances[1]);
        }
        else {
            simsimd_distance_t distance;
            metric(a_parsed.start, b_parsed.start, a_parsed.dimensions, &distance);
            return_obj = PyFloat_FromDouble(distance);
        }
        goto cleanup;
    }

    // In some batch requests we may be computing the distance from multiple vectors to one,
    // so the stride must be set to zero avoid illegal memory access
    if (a_parsed.count == 1) a_parsed.stride = 0;
    if (b_parsed.count == 1) b_parsed.stride = 0;

    // We take the maximum of the two counts, because if only one entry is present in one of the arrays,
    // all distances will be computed against that single entry.
    size_t const count_pairs = a_parsed.count > b_parsed.count ? a_parsed.count : b_parsed.count;
    size_t const components_per_pair = dtype_is_complex ? 2 : 1;
    size_t const count_components = count_pairs * components_per_pair;
    char *out_buffer_start = NULL;
    size_t out_buffer_stride_bytes = 0;

    // Allocate the output matrix if it wasn't provided
    if (!out_obj) {
        DistancesTensor *out_buffer_obj =
            PyObject_NewVar(DistancesTensor, &DistancesTensorType, count_components * bytes_per_datatype(out_dtype));
        if (!out_buffer_obj) {
            PyErr_NoMemory();
            goto cleanup;
        }

        // Initialize the object
        out_buffer_obj->datatype = out_dtype;
        out_buffer_obj->dimensions = 1;
        out_buffer_obj->shape[0] = count_pairs;
        out_buffer_obj->shape[1] = 1;
        out_buffer_obj->strides[0] = bytes_per_datatype(out_dtype);
        out_buffer_obj->strides[1] = 0;
        return_obj = (PyObject *)out_buffer_obj;
        out_buffer_start = (char *)&out_buffer_obj->start[0];
        out_buffer_stride_bytes = out_buffer_obj->strides[0];
    }
    else {
        if (bytes_per_datatype(out_parsed.datatype) != bytes_per_datatype(out_dtype)) {
            PyErr_Format( //
                PyExc_LookupError,
                "Output tensor scalar type must be compatible with the output type ('%s' and '%s'/'%s')",
                datatype_to_python_string(out_dtype), out_buffer.format ? out_buffer.format : "nil",
                datatype_to_python_string(out_parsed.datatype));
            goto cleanup;
        }
        out_buffer_start = (char *)&out_parsed.start[0];
        out_buffer_stride_bytes = out_buffer.strides[0];
        //? Logic suggests to return `None` in in-place mode...
        //? SciPy decided differently.
        return_obj = Py_None;
    }

    // Compute the distances
    for (size_t i = 0; i < count_pairs; ++i) {
        simsimd_distance_t result[2];
        metric(                                   //
            a_parsed.start + i * a_parsed.stride, //
            b_parsed.start + i * b_parsed.stride, //
            a_parsed.dimensions,                  //
            (simsimd_distance_t *)&result);

        // Export out:
        cast_distance(result[0], out_dtype, out_buffer_start + i * out_buffer_stride_bytes, 0);
        if (dtype_is_complex) cast_distance(result[1], out_dtype, out_buffer_start + i * out_buffer_stride_bytes, 1);
    }

cleanup:
    PyBuffer_Release(&a_buffer);
    PyBuffer_Release(&b_buffer);
    PyBuffer_Release(&out_buffer);
    if (return_obj == Py_None) Py_INCREF(return_obj);
    return return_obj;
}

static PyObject *implement_curved_metric( //
    simsimd_kernel_kind_t kernel_kind,    //
    PyObject *const *args, Py_ssize_t const positional_args_count, PyObject *args_names_tuple) {

    PyObject *return_obj = NULL;

    // This function accepts up to 6 arguments:
    PyObject *a_obj = NULL;     // Required object, positional-only
    PyObject *b_obj = NULL;     // Required object, positional-only
    PyObject *c_obj = NULL;     // Required object, positional-only
    PyObject *dtype_obj = NULL; // Optional object, "dtype" keyword or positional

    // Once parsed, the arguments will be stored in these variables:
    char const *dtype_str = NULL;
    simsimd_datatype_t dtype = simsimd_datatype_unknown_k;
    Py_buffer a_buffer, b_buffer, c_buffer;
    VectorOrRowsArgument a_parsed, b_parsed, c_parsed;
    memset(&a_buffer, 0, sizeof(Py_buffer));
    memset(&b_buffer, 0, sizeof(Py_buffer));
    memset(&c_buffer, 0, sizeof(Py_buffer));

    // Parse the arguments
    Py_ssize_t const args_names_count = args_names_tuple ? PyTuple_Size(args_names_tuple) : 0;
    Py_ssize_t const args_count = positional_args_count + args_names_count;
    if (args_count < 3 || args_count > 6) {
        PyErr_Format(PyExc_TypeError, "Function expects 2-6 arguments, got %zd", args_count);
        return NULL;
    }
    if (positional_args_count > 4) {
        PyErr_Format(PyExc_TypeError, "Only first 4 arguments can be positional, received %zd", positional_args_count);
        return NULL;
    }

    // Positional-only arguments (first, second, and third matrix)
    a_obj = args[0];
    b_obj = args[1];
    c_obj = args[2];

    // Positional or keyword arguments (dtype)
    if (positional_args_count == 4) dtype_obj = args[3];

    // The rest of the arguments must be checked in the keyword dictionary:
    for (Py_ssize_t args_names_tuple_progress = 0, args_progress = positional_args_count;
         args_names_tuple_progress < args_names_count; ++args_progress, ++args_names_tuple_progress) {
        PyObject *const key = PyTuple_GetItem(args_names_tuple, args_names_tuple_progress);
        PyObject *const value = args[args_progress];
        if (PyUnicode_CompareWithASCIIString(key, "dtype") == 0 && !dtype_obj) { dtype_obj = value; }
        else {
            PyErr_Format(PyExc_TypeError, "Got unexpected keyword argument: %S", key);
            return NULL;
        }
    }

    // Convert `dtype_obj` to `dtype_str` and to `dtype`
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

    // Convert `a_obj` to `a_buffer` and to `a_parsed`. Same for `b_obj` and `out_obj`.
    if (!parse_rows(a_obj, &a_buffer, &a_parsed) || !parse_rows(b_obj, &b_buffer, &b_parsed) ||
        !parse_rows(c_obj, &c_buffer, &c_parsed))
        return NULL;

    // Check dimensions
    if (a_parsed.rank != 1 || b_parsed.rank != 1) {
        PyErr_SetString(PyExc_ValueError, "First and second argument must be vectors");
        goto cleanup;
    }
    if (c_parsed.rank != 2) {
        PyErr_SetString(PyExc_ValueError, "Third argument must be a matrix (rank-2 tensor)");
        goto cleanup;
    }
    if (a_parsed.count == 0 || b_parsed.count == 0) {
        PyErr_SetString(PyExc_ValueError, "Collections can't be empty");
        goto cleanup;
    }
    if (a_parsed.count > 1 && b_parsed.count > 1 && a_parsed.count != b_parsed.count) {
        PyErr_SetString(PyExc_ValueError, "Collections must have the same number of elements or just one element");
        goto cleanup;
    }

    // Check data types
    if (a_parsed.datatype != b_parsed.datatype || a_parsed.datatype != c_parsed.datatype ||
        a_parsed.datatype == simsimd_datatype_unknown_k || b_parsed.datatype == simsimd_datatype_unknown_k ||
        c_parsed.datatype == simsimd_datatype_unknown_k) {
        PyErr_SetString(PyExc_TypeError,
                        "Input tensors must have matching datatypes, check with `X.__array_interface__`");
        goto cleanup;
    }
    if (dtype == simsimd_datatype_unknown_k) dtype = a_parsed.datatype;

    // Look up the metric and the capability
    simsimd_curved_metric_t metric = NULL;
    simsimd_capability_t capability = simsimd_cap_serial_k;
    simsimd_find_kernel(kernel_kind, dtype, static_capabilities, simsimd_cap_any_k, (simsimd_kernel_punned_t *)&metric,
                        &capability);
    if (!metric) {
        PyErr_Format( //
            PyExc_LookupError,
            "Unsupported metric '%c' and datatype combination across vectors ('%s'/'%s' and '%s'/'%s'), "
            "tensor ('%s'/'%s'), and `dtype` override ('%s'/'%s')",
            kernel_kind,                                                                             //
            a_buffer.format ? a_buffer.format : "nil", datatype_to_python_string(a_parsed.datatype), //
            b_buffer.format ? b_buffer.format : "nil", datatype_to_python_string(b_parsed.datatype), //
            c_buffer.format ? c_buffer.format : "nil", datatype_to_python_string(c_parsed.datatype), //
            dtype_str ? dtype_str : "nil", datatype_to_python_string(dtype));
        goto cleanup;
    }

    simsimd_distance_t distance;
    metric(a_parsed.start, b_parsed.start, c_parsed.start, a_parsed.dimensions, &distance);
    return_obj = PyFloat_FromDouble(distance);

cleanup:
    PyBuffer_Release(&a_buffer);
    PyBuffer_Release(&b_buffer);
    PyBuffer_Release(&c_buffer);
    if (return_obj == Py_None) Py_INCREF(return_obj);
    return return_obj;
}

static PyObject *implement_sparse_metric( //
    simsimd_kernel_kind_t kernel_kind,    //
    PyObject *const *args, Py_ssize_t nargs) {
    if (nargs != 2) {
        PyErr_SetString(PyExc_TypeError, "Function expects only 2 arguments");
        return NULL;
    }

    PyObject *return_obj = NULL;
    PyObject *a_obj = args[0];
    PyObject *b_obj = args[1];

    Py_buffer a_buffer, b_buffer;
    VectorOrRowsArgument a_parsed, b_parsed;
    if (!parse_rows(a_obj, &a_buffer, &a_parsed) || !parse_rows(b_obj, &b_buffer, &b_parsed)) return NULL;

    // Check dimensions
    if (a_parsed.rank != 1 || b_parsed.rank != 1) {
        PyErr_SetString(PyExc_ValueError, "First and second argument must be vectors");
        goto cleanup;
    }

    // Check data types
    if (a_parsed.datatype != b_parsed.datatype && a_parsed.datatype != simsimd_datatype_unknown_k &&
        b_parsed.datatype != simsimd_datatype_unknown_k) {
        PyErr_SetString(PyExc_TypeError,
                        "Input tensors must have matching datatypes, check with `X.__array_interface__`");
        goto cleanup;
    }

    simsimd_datatype_t dtype = a_parsed.datatype;
    simsimd_sparse_metric_t metric = NULL;
    simsimd_capability_t capability = simsimd_cap_serial_k;
    simsimd_find_kernel(kernel_kind, dtype, static_capabilities, simsimd_cap_any_k, (simsimd_kernel_punned_t *)&metric,
                        &capability);
    if (!metric) {
        PyErr_Format( //
            PyExc_LookupError, "Unsupported metric '%c' and datatype combination ('%s'/'%s' and '%s'/'%s')",
            kernel_kind,                                                                             //
            a_buffer.format ? a_buffer.format : "nil", datatype_to_python_string(a_parsed.datatype), //
            b_buffer.format ? b_buffer.format : "nil", datatype_to_python_string(b_parsed.datatype));
        goto cleanup;
    }

    simsimd_distance_t distance;
    metric(a_parsed.start, b_parsed.start, a_parsed.dimensions, b_parsed.dimensions, &distance);
    return_obj = PyFloat_FromDouble(distance);

cleanup:
    PyBuffer_Release(&a_buffer);
    PyBuffer_Release(&b_buffer);
    if (return_obj == Py_None) Py_INCREF(return_obj);
    return return_obj;
}

static PyObject *implement_cdist(                        //
    PyObject *a_obj, PyObject *b_obj, PyObject *out_obj, //
    simsimd_kernel_kind_t kernel_kind, size_t threads,   //
    simsimd_datatype_t dtype, simsimd_datatype_t out_dtype) {

    PyObject *return_obj = NULL;

    Py_buffer a_buffer, b_buffer, out_buffer;
    VectorOrRowsArgument a_parsed, b_parsed, out_parsed;
    memset(&a_buffer, 0, sizeof(Py_buffer));
    memset(&b_buffer, 0, sizeof(Py_buffer));
    memset(&out_buffer, 0, sizeof(Py_buffer));

    // Error will be set by `parse_rows` if the input is invalid
    if (!parse_rows(a_obj, &a_buffer, &a_parsed) || !parse_rows(b_obj, &b_buffer, &b_parsed)) return NULL;
    if (out_obj && !parse_rows(out_obj, &out_buffer, &out_parsed)) return NULL;

    // Check dimensions
    if (a_parsed.dimensions != b_parsed.dimensions) {
        PyErr_Format(PyExc_ValueError, "Vector dimensions don't match (%z != %z)", a_parsed.dimensions,
                     b_parsed.dimensions);
        goto cleanup;
    }
    if (a_parsed.count == 0 || b_parsed.count == 0) {
        PyErr_SetString(PyExc_ValueError, "Collections can't be empty");
        goto cleanup;
    }
    if (out_obj &&
        (out_parsed.rank != 2 || out_buffer.shape[0] != a_parsed.count || out_buffer.shape[1] != b_parsed.count)) {
        PyErr_Format(PyExc_ValueError, "Output tensor must have shape (%z, %z)", a_parsed.count, b_parsed.count);
        goto cleanup;
    }

    // Check data types
    if (a_parsed.datatype != b_parsed.datatype || //
        a_parsed.datatype == simsimd_datatype_unknown_k || b_parsed.datatype == simsimd_datatype_unknown_k) {
        PyErr_SetString(PyExc_TypeError,
                        "Input tensors must have matching datatypes, check with `X.__array_interface__`");
        goto cleanup;
    }
    if (dtype == simsimd_datatype_unknown_k) dtype = a_parsed.datatype;

    // Inference order for the output type:
    // 1. `out_dtype` named argument, if defined
    // 2. `out.dtype` attribute, if `out` is passed
    // 3. double precision float (or its complex variant)
    if (out_dtype == simsimd_datatype_unknown_k) {
        if (out_obj) { out_dtype = out_parsed.datatype; }
        else { out_dtype = is_complex(dtype) ? simsimd_f64c_k : simsimd_f64_k; }
    }

    // Make sure the return datatype is complex if the input datatype is complex, and the same for real numbers
    if (out_dtype != simsimd_datatype_unknown_k) {
        if (is_complex(dtype) != is_complex(out_dtype)) {
            PyErr_SetString(
                PyExc_ValueError,
                "If the input datatype is complex, the return datatype must be complex, and same for real.");
            goto cleanup;
        }
    }

    // Check if the downcasting to provided datatype is supported
    {
        char returned_buffer_example[8];
        if (!cast_distance(0, out_dtype, &returned_buffer_example, 0)) {
            PyErr_SetString(PyExc_ValueError, "Exporting to the provided datatype is not supported");
            goto cleanup;
        }
    }

    // Look up the metric and the capability
    simsimd_dense_metric_t metric = NULL;
    simsimd_capability_t capability = simsimd_cap_serial_k;
    simsimd_find_kernel(kernel_kind, dtype, static_capabilities, simsimd_cap_any_k, (simsimd_kernel_punned_t *)&metric,
                        &capability);
    if (!metric) {
        PyErr_Format( //
            PyExc_LookupError, "Unsupported metric '%c' and datatype combination ('%s'/'%s' and '%s'/'%s')",
            kernel_kind,                                                                             //
            a_buffer.format ? a_buffer.format : "nil", datatype_to_python_string(a_parsed.datatype), //
            b_buffer.format ? b_buffer.format : "nil", datatype_to_python_string(b_parsed.datatype));
        goto cleanup;
    }

    // If the distance is computed between two vectors, rather than matrices, return a scalar
    int const dtype_is_complex = is_complex(dtype);
    if (a_parsed.rank == 1 && b_parsed.rank == 1) {
        // For complex numbers we are going to use `PyComplex_FromDoubles`.
        if (dtype_is_complex) {
            simsimd_distance_t distances[2];
            metric(a_parsed.start, b_parsed.start, a_parsed.dimensions, distances);
            return_obj = PyComplex_FromDoubles(distances[0], distances[1]);
        }
        else {
            simsimd_distance_t distance;
            metric(a_parsed.start, b_parsed.start, a_parsed.dimensions, &distance);
            return_obj = PyFloat_FromDouble(distance);
        }
        goto cleanup;
    }

#ifdef __linux__
#ifdef _OPENMP
    if (threads == 0) threads = omp_get_num_procs();
    omp_set_num_threads(threads);
#endif
#endif

    size_t const count_pairs = a_parsed.count * b_parsed.count;
    size_t const components_per_pair = dtype_is_complex ? 2 : 1;
    size_t const count_components = count_pairs * components_per_pair;
    char *out_buffer_start = NULL;
    size_t out_buffer_rows_stride_bytes = 0;
    size_t out_buffer_cols_stride_bytes = 0;

    // Allocate the output matrix if it wasn't provided
    if (!out_obj) {

        DistancesTensor *out_buffer_obj =
            PyObject_NewVar(DistancesTensor, &DistancesTensorType, count_components * bytes_per_datatype(out_dtype));
        if (!out_buffer_obj) {
            PyErr_NoMemory();
            goto cleanup;
        }

        // Initialize the object
        out_buffer_obj->datatype = out_dtype;
        out_buffer_obj->dimensions = 2;
        out_buffer_obj->shape[0] = a_parsed.count;
        out_buffer_obj->shape[1] = b_parsed.count;
        out_buffer_obj->strides[0] = b_parsed.count * bytes_per_datatype(out_buffer_obj->datatype);
        out_buffer_obj->strides[1] = bytes_per_datatype(out_buffer_obj->datatype);
        return_obj = (PyObject *)out_buffer_obj;
        out_buffer_start = (char *)&out_buffer_obj->start[0];
        out_buffer_rows_stride_bytes = out_buffer_obj->strides[0];
        out_buffer_cols_stride_bytes = out_buffer_obj->strides[1];
    }
    else {
        if (bytes_per_datatype(out_parsed.datatype) != bytes_per_datatype(out_dtype)) {
            PyErr_Format( //
                PyExc_LookupError,
                "Output tensor scalar type must be compatible with the output type ('%s' and '%s'/'%s')",
                datatype_to_python_string(out_dtype), out_buffer.format ? out_buffer.format : "nil",
                datatype_to_python_string(out_parsed.datatype));
            goto cleanup;
        }
        out_buffer_start = (char *)&out_parsed.start[0];
        out_buffer_rows_stride_bytes = out_buffer.strides[0];
        out_buffer_cols_stride_bytes = out_buffer.strides[1];
        //? Logic suggests to return `None` in in-place mode...
        //? SciPy decided differently.
        return_obj = Py_None;
    }

    // Assuming most of our kernels are symmetric, we only need to compute the upper triangle
    // if we are computing all pairwise distances within the same set.
    int const is_symmetric = kernel_is_commutative(kernel_kind) && a_parsed.start == b_parsed.start &&
                             a_parsed.stride == b_parsed.stride && a_parsed.count == b_parsed.count;
#pragma omp parallel for collapse(2)
    for (size_t i = 0; i < a_parsed.count; ++i)
        for (size_t j = 0; j < b_parsed.count; ++j) {
            if (is_symmetric && i > j) continue;

            // Export into an on-stack buffer and then copy to the output
            simsimd_distance_t result[2];
            metric(                                   //
                a_parsed.start + i * a_parsed.stride, //
                b_parsed.start + j * b_parsed.stride, //
                a_parsed.dimensions,                  //
                (simsimd_distance_t *)&result         //
            );

            // Export into both the lower and upper triangle
            if (1)
                cast_distance(result[0], out_dtype,
                              out_buffer_start + i * out_buffer_rows_stride_bytes + j * out_buffer_cols_stride_bytes,
                              0);
            if (dtype_is_complex)
                cast_distance(result[1], out_dtype,
                              out_buffer_start + i * out_buffer_rows_stride_bytes + j * out_buffer_cols_stride_bytes,
                              1);
            if (is_symmetric)
                cast_distance(result[0], out_dtype,
                              out_buffer_start + j * out_buffer_rows_stride_bytes + i * out_buffer_cols_stride_bytes,
                              0);
            if (is_symmetric && dtype_is_complex)
                cast_distance(result[1], out_dtype,
                              out_buffer_start + j * out_buffer_rows_stride_bytes + i * out_buffer_cols_stride_bytes,
                              1);
        }

cleanup:
    PyBuffer_Release(&a_buffer);
    PyBuffer_Release(&b_buffer);
    PyBuffer_Release(&out_buffer);
    if (return_obj == Py_None) Py_INCREF(return_obj);
    return return_obj;
}

static PyObject *implement_pointer_access(simsimd_kernel_kind_t kernel_kind, PyObject *dtype_obj) {
    char const *dtype_name = PyUnicode_AsUTF8(dtype_obj);
    if (!dtype_name) {
        PyErr_SetString(PyExc_TypeError, "Data-type name must be a string");
        return NULL;
    }

    simsimd_datatype_t datatype = python_string_to_datatype(dtype_name);
    if (!datatype) { // Check the actual variable here instead of dtype_name
        PyErr_SetString(PyExc_ValueError, "Unsupported type");
        return NULL;
    }

    simsimd_kernel_punned_t metric = NULL;
    simsimd_capability_t capability = simsimd_cap_serial_k;
    simsimd_find_kernel(kernel_kind, datatype, static_capabilities, simsimd_cap_any_k, &metric, &capability);
    if (metric == NULL) {
        PyErr_SetString(PyExc_LookupError, "No such metric");
        return NULL;
    }

    return PyLong_FromUnsignedLongLong((unsigned long long)metric);
}

static char const doc_cdist[] = //
    "Compute pairwise distances between two sets of input matrices.\n"
    "\n"
    "Args:\n"
    "    a (NDArray): First matrix.\n"
    "    b (NDArray): Second matrix.\n"
    "    metric (str, optional): Distance metric to use (e.g., 'sqeuclidean', 'cosine').\n"
    "    out (NDArray, optional): Output matrix to store the result.\n"
    "    dtype (Union[IntegralType, FloatType, ComplexType], optional): Override the presumed input type name.\n"
    "    out_dtype (Union[FloatType, ComplexType], optional): Result type, default is 'float64'.\n"
    "    threads (int, optional): Number of threads to use (default is 1).\n"
    "Returns:\n"
    "    DistancesTensor: Pairwise distances between all inputs.\n"
    "\n"
    "Equivalent to: `scipy.spatial.distance.cdist`.\n"
    "Signature:\n"
    "    >>> def cdist(a, b, /, metric, *, dtype, out, out_dtype, threads) -> Optional[DistancesTensor]: ...";

static PyObject *api_cdist( //
    PyObject *self, PyObject *const *args, Py_ssize_t const positional_args_count, PyObject *args_names_tuple) {

    // This function accepts up to 7 arguments - more than SciPy:
    // https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html
    PyObject *a_obj = NULL;         // Required object, positional-only
    PyObject *b_obj = NULL;         // Required object, positional-only
    PyObject *metric_obj = NULL;    // Optional string, "metric" keyword or positional
    PyObject *out_obj = NULL;       // Optional object, "out" keyword-only
    PyObject *dtype_obj = NULL;     // Optional string, "dtype" keyword-only
    PyObject *out_dtype_obj = NULL; // Optional string, "out_dtype" keyword-only
    PyObject *threads_obj = NULL;   // Optional integer, "threads" keyword-only

    // Once parsed, the arguments will be stored in these variables:
    unsigned long long threads = 1;
    char const *dtype_str = NULL, *out_dtype_str = NULL;
    simsimd_datatype_t dtype = simsimd_datatype_unknown_k, out_dtype = simsimd_datatype_unknown_k;

    /// Same default as in SciPy:
    /// https://docs.scipy.org/doc/scipy-1.11.4/reference/generated/scipy.spatial.distance.cdist.html
    simsimd_kernel_kind_t kernel_kind = simsimd_euclidean_k;
    char const *metric_str = NULL;

    // Parse the arguments
    Py_ssize_t const args_names_count = args_names_tuple ? PyTuple_Size(args_names_tuple) : 0;
    Py_ssize_t const args_count = positional_args_count + args_names_count;
    if (args_count < 2 || args_count > 7) {
        PyErr_Format(PyExc_TypeError, "Function expects 2-7 arguments, got %zd", args_count);
        return NULL;
    }
    if (positional_args_count > 3) {
        PyErr_Format(PyExc_TypeError, "Only first 3 arguments can be positional, received %zd", positional_args_count);
        return NULL;
    }

    // Positional-only arguments (first and second matrix)
    a_obj = args[0];
    b_obj = args[1];

    // Positional or keyword arguments (metric)
    if (positional_args_count == 3) metric_obj = args[2];

    // The rest of the arguments must be checked in the keyword dictionary:
    for (Py_ssize_t args_names_tuple_progress = 0, args_progress = positional_args_count;
         args_names_tuple_progress < args_names_count; ++args_progress, ++args_names_tuple_progress) {
        PyObject *const key = PyTuple_GetItem(args_names_tuple, args_names_tuple_progress);
        PyObject *const value = args[args_progress];
        if (PyUnicode_CompareWithASCIIString(key, "dtype") == 0 && !dtype_obj) { dtype_obj = value; }
        else if (PyUnicode_CompareWithASCIIString(key, "out") == 0 && !out_obj) { out_obj = value; }
        else if (PyUnicode_CompareWithASCIIString(key, "out_dtype") == 0 && !out_dtype_obj) { out_dtype_obj = value; }
        else if (PyUnicode_CompareWithASCIIString(key, "threads") == 0 && !threads_obj) { threads_obj = value; }
        else if (PyUnicode_CompareWithASCIIString(key, "metric") == 0 && !metric_obj) { metric_obj = value; }
        else {
            PyErr_Format(PyExc_TypeError, "Got unexpected keyword argument: %S", key);
            return NULL;
        }
    }

    // Convert `metric_obj` to `metric_str` and to `kernel_kind`
    if (metric_obj) {
        metric_str = PyUnicode_AsUTF8(metric_obj);
        if (!metric_str && PyErr_Occurred()) {
            PyErr_SetString(PyExc_TypeError, "Expected 'metric' to be a string");
            return NULL;
        }
        kernel_kind = python_string_to_kernel_kind(metric_str);
        if (kernel_kind == simsimd_kernel_unknown_k) {
            PyErr_SetString(PyExc_LookupError, "Unsupported metric");
            return NULL;
        }
    }

    // Convert `threads_obj` to `threads` integer
    if (threads_obj) threads = PyLong_AsSize_t(threads_obj);
    if (PyErr_Occurred()) {
        PyErr_SetString(PyExc_TypeError, "Expected 'threads' to be an unsigned integer");
        return NULL;
    }

    // Convert `dtype_obj` to `dtype_str` and to `dtype`
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

    // Convert `out_dtype_obj` to `out_dtype_str` and to `out_dtype`
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

    return implement_cdist(a_obj, b_obj, out_obj, kernel_kind, threads, dtype, out_dtype);
}

static char const doc_l2_pointer[] = "Get (int) pointer to the `simsimd.l2` kernel.";
static PyObject *api_l2_pointer(PyObject *self, PyObject *dtype_obj) {
    return implement_pointer_access(simsimd_l2_k, dtype_obj);
}
static char const doc_l2sq_pointer[] = "Get (int) pointer to the `simsimd.l2sq` kernel.";
static PyObject *api_l2sq_pointer(PyObject *self, PyObject *dtype_obj) {
    return implement_pointer_access(simsimd_l2sq_k, dtype_obj);
}
static char const doc_cos_pointer[] = "Get (int) pointer to the `simsimd.cos` kernel.";
static PyObject *api_cos_pointer(PyObject *self, PyObject *dtype_obj) {
    return implement_pointer_access(simsimd_cos_k, dtype_obj);
}
static char const doc_dot_pointer[] = "Get (int) pointer to the `simsimd.dot` kernel.";
static PyObject *api_dot_pointer(PyObject *self, PyObject *dtype_obj) {
    return implement_pointer_access(simsimd_dot_k, dtype_obj);
}
static char const doc_vdot_pointer[] = "Get (int) pointer to the `simsimd.vdot` kernel.";
static PyObject *api_vdot_pointer(PyObject *self, PyObject *dtype_obj) {
    return implement_pointer_access(simsimd_vdot_k, dtype_obj);
}
static char const doc_kl_pointer[] = "Get (int) pointer to the `simsimd.kl` kernel.";
static PyObject *api_kl_pointer(PyObject *self, PyObject *dtype_obj) {
    return implement_pointer_access(simsimd_kl_k, dtype_obj);
}
static char const doc_js_pointer[] = "Get (int) pointer to the `simsimd.js` kernel.";
static PyObject *api_js_pointer(PyObject *self, PyObject *dtype_obj) {
    return implement_pointer_access(simsimd_js_k, dtype_obj);
}
static char const doc_hamming_pointer[] = "Get (int) pointer to the `simsimd.hamming` kernel.";
static PyObject *api_hamming_pointer(PyObject *self, PyObject *dtype_obj) {
    return implement_pointer_access(simsimd_hamming_k, dtype_obj);
}
static char const doc_jaccard_pointer[] = "Get (int) pointer to the `simsimd.jaccard` kernel.";
static PyObject *api_jaccard_pointer(PyObject *self, PyObject *dtype_obj) {
    return implement_pointer_access(simsimd_jaccard_k, dtype_obj);
}

static char const doc_l2[] = //
    "Compute Euclidean (L2) distances between two matrices.\n"
    "\n"
    "Args:\n"
    "    a (NDArray): First matrix or vector.\n"
    "    b (NDArray): Second matrix or vector.\n"
    "    dtype (Union[IntegralType, FloatType], optional): Override the presumed input type name.\n"
    "    out (NDArray, optional): Vector for resulting distances. Allocates a new tensor by default.\n"
    "    out_dtype (FloatType, optional): Result type, default is 'float64'.\n"
    "Returns:\n"
    "    DistancesTensor: The distances if `out` is not provided.\n"
    "    None: If `out` is provided. Operation will per performed in-place.\n"
    "\n"
    "Equivalent to: `scipy.spatial.distance.euclidean`.\n"
    "Signature:\n"
    "    >>> def euclidean(a, b, /, dtype, *, out, out_dtype) -> Optional[DistancesTensor]: ...";

static PyObject *api_l2(PyObject *self, PyObject *const *args, Py_ssize_t const positional_args_count,
                        PyObject *args_names_tuple) {
    return implement_dense_metric(simsimd_l2_k, args, positional_args_count, args_names_tuple);
}

static char const doc_l2sq[] = //
    "Compute squared Euclidean (L2) distances between two matrices.\n"
    "\n"
    "Args:\n"
    "    a (NDArray): First matrix or vector.\n"
    "    b (NDArray): Second matrix or vector.\n"
    "    dtype (Union[IntegralType, FloatType], optional): Override the presumed input type name.\n"
    "    out (NDArray, optional): Vector for resulting distances. Allocates a new tensor by default.\n"
    "    out_dtype (FloatType, optional): Result type, default is 'float64'.\n"
    "Returns:\n"
    "    DistancesTensor: The distances if `out` is not provided.\n"
    "    None: If `out` is provided. Operation will per performed in-place.\n"
    "\n"
    "Equivalent to: `scipy.spatial.distance.sqeuclidean`.\n"
    "Signature:\n"
    "    >>> def sqeuclidean(a, b, /, dtype, *, out, out_dtype) -> Optional[DistancesTensor]: ...";

static PyObject *api_l2sq(PyObject *self, PyObject *const *args, Py_ssize_t const positional_args_count,
                          PyObject *args_names_tuple) {
    return implement_dense_metric(simsimd_l2sq_k, args, positional_args_count, args_names_tuple);
}

static char const doc_cos[] = //
    "Compute cosine (angular) distances between two matrices.\n"
    "\n"
    "Args:\n"
    "    a (NDArray): First matrix or vector.\n"
    "    b (NDArray): Second matrix or vector.\n"
    "    dtype (Union[IntegralType, FloatType], optional): Override the presumed input type name.\n"
    "    out (NDArray, optional): Vector for resulting distances. Allocates a new tensor by default.\n"
    "    out_dtype (FloatType, optional): Result type, default is 'float64'.\n"
    "Returns:\n"
    "    DistancesTensor: The distances if `out` is not provided.\n"
    "    None: If `out` is provided. Operation will per performed in-place.\n"
    "\n"
    "Equivalent to: `scipy.spatial.distance.cosine`.\n"
    "Signature:\n"
    "    >>> def cosine(a, b, /, dtype, *, out, out_dtype) -> Optional[DistancesTensor]: ...";

static PyObject *api_cos(PyObject *self, PyObject *const *args, Py_ssize_t const positional_args_count,
                         PyObject *args_names_tuple) {
    return implement_dense_metric(simsimd_cos_k, args, positional_args_count, args_names_tuple);
}

static char const doc_dot[] = //
    "Compute the inner (dot) product between two matrices (real or complex).\n"
    "\n"
    "Args:\n"
    "    a (NDArray): First matrix or vector.\n"
    "    b (NDArray): Second matrix or vector.\n"
    "    dtype (Union[IntegralType, FloatType, ComplexType], optional): Override the presumed input type name.\n"
    "    out (NDArray, optional): Vector for resulting distances. Allocates a new tensor by default.\n"
    "    out_dtype (Union[FloatType, ComplexType], optional): Result type, default is 'float64'.\n"
    "Returns:\n"
    "    DistancesTensor: The distances if `out` is not provided.\n"
    "    None: If `out` is provided. Operation will per performed in-place.\n"
    "\n"
    "Equivalent to: `numpy.inner`.\n"
    "Signature:\n"
    "    >>> def dot(a, b, /, dtype, *, out, out_dtype) -> Optional[DistancesTensor]: ...";

static PyObject *api_dot(PyObject *self, PyObject *const *args, Py_ssize_t const positional_args_count,
                         PyObject *args_names_tuple) {
    return implement_dense_metric(simsimd_dot_k, args, positional_args_count, args_names_tuple);
}

static char const doc_vdot[] = //
    "Compute the conjugate dot product between two complex matrices.\n"
    "\n"
    "Args:\n"
    "    a (NDArray): First complex matrix or vector.\n"
    "    b (NDArray): Second complex matrix or vector.\n"
    "    dtype (ComplexType, optional): Override the presumed input type name.\n"
    "    out (NDArray, optional): Vector for resulting distances. Allocates a new tensor by default.\n"
    "    out_dtype (Union[ComplexType], optional): Result type, default is 'float64'.\n"
    "Returns:\n"
    "    DistancesTensor: The distances if `out` is not provided.\n"
    "    None: If `out` is provided. Operation will per performed in-place.\n"
    "\n"
    "Equivalent to: `numpy.vdot`.\n"
    "Signature:\n"
    "    >>> def vdot(a, b, /, dtype, *, out, out_dtype) -> Optional[DistancesTensor]: ...";

static PyObject *api_vdot(PyObject *self, PyObject *const *args, Py_ssize_t const positional_args_count,
                          PyObject *args_names_tuple) {
    return implement_dense_metric(simsimd_vdot_k, args, positional_args_count, args_names_tuple);
}

static char const doc_kl[] = //
    "Compute Kullback-Leibler divergences between two matrices.\n"
    "\n"
    "Args:\n"
    "    a (NDArray): First floating-point matrix or vector.\n"
    "    b (NDArray): Second floating-point matrix or vector.\n"
    "    dtype (FloatType, optional): Override the presumed input type name.\n"
    "    out (NDArray, optional): Vector for resulting distances. Allocates a new tensor by default.\n"
    "    out_dtype (FloatType, optional): Result type, default is 'float64'.\n"
    "Returns:\n"
    "    DistancesTensor: The distances if `out` is not provided.\n"
    "    None: If `out` is provided. Operation will per performed in-place.\n"
    "\n"
    "Equivalent to: `scipy.special.kl_div`.\n"
    "Signature:\n"
    "    >>> def kl(a, b, /, dtype, *, out, out_dtype) -> Optional[DistancesTensor]: ...";

static PyObject *api_kl(PyObject *self, PyObject *const *args, Py_ssize_t const positional_args_count,
                        PyObject *args_names_tuple) {
    return implement_dense_metric(simsimd_kl_k, args, positional_args_count, args_names_tuple);
}

static char const doc_js[] = //
    "Compute Jensen-Shannon divergences between two matrices.\n"
    "\n"
    "Args:\n"
    "    a (NDArray): First floating-point matrix or vector.\n"
    "    b (NDArray): Second floating-point matrix or vector.\n"
    "    dtype (Union[IntegralType, FloatType], optional): Override the presumed input type name.\n"
    "    out (NDArray, optional): Vector for resulting distances. Allocates a new tensor by default.\n"
    "    out_dtype (FloatType, optional): Result type, default is 'float64'.\n"
    "Returns:\n"
    "    DistancesTensor: The distances if `out` is not provided.\n"
    "    None: If `out` is provided. Operation will per performed in-place.\n"
    "\n"
    "Equivalent to: `scipy.spatial.distance.jensenshannon`.\n"
    "Signature:\n"
    "    >>> def kl(a, b, /, dtype, *, out, out_dtype) -> Optional[DistancesTensor]: ...";

static PyObject *api_js(PyObject *self, PyObject *const *args, Py_ssize_t const positional_args_count,
                        PyObject *args_names_tuple) {
    return implement_dense_metric(simsimd_js_k, args, positional_args_count, args_names_tuple);
}

static char const doc_hamming[] = //
    "Compute Hamming distances between two matrices.\n"
    "\n"
    "Args:\n"
    "    a (NDArray): First binary matrix or vector.\n"
    "    b (NDArray): Second binary matrix or vector.\n"
    "    dtype (IntegralType, optional): Override the presumed input type name.\n"
    "    out (NDArray, optional): Vector for resulting distances. Allocates a new tensor by default.\n"
    "    out_dtype (FloatType, optional): Result type, default is 'float64'.\n"
    "Returns:\n"
    "    DistancesTensor: The distances if `out` is not provided.\n"
    "    None: If `out` is provided. Operation will per performed in-place.\n"
    "\n"
    "Similar to: `scipy.spatial.distance.hamming`.\n"
    "Signature:\n"
    "    >>> def hamming(a, b, /, dtype, *, out, out_dtype) -> Optional[DistancesTensor]: ...";

static PyObject *api_hamming(PyObject *self, PyObject *const *args, Py_ssize_t const positional_args_count,
                             PyObject *args_names_tuple) {
    return implement_dense_metric(simsimd_hamming_k, args, positional_args_count, args_names_tuple);
}

static char const doc_jaccard[] = //
    "Compute Jaccard distances (bitwise Tanimoto) between two matrices.\n"
    "\n"
    "Args:\n"
    "    a (NDArray): First binary matrix or vector.\n"
    "    b (NDArray): Second binary matrix or vector.\n"
    "    dtype (IntegralType, optional): Override the presumed input type name.\n"
    "    out (NDArray, optional): Vector for resulting distances. Allocates a new tensor by default.\n"
    "    out_dtype (FloatType, optional): Result type, default is 'float64'.\n"
    "Returns:\n"
    "    DistancesTensor: The distances if `out` is not provided.\n"
    "    None: If `out` is provided. Operation will per performed in-place.\n"
    "\n"
    "Similar to: `scipy.spatial.distance.jaccard`.\n"
    "Signature:\n"
    "    >>> def jaccard(a, b, /, dtype, *, out, out_dtype) -> Optional[DistancesTensor]: ...";

static PyObject *api_jaccard(PyObject *self, PyObject *const *args, Py_ssize_t const positional_args_count,
                             PyObject *args_names_tuple) {
    return implement_dense_metric(simsimd_jaccard_k, args, positional_args_count, args_names_tuple);
}

static char const doc_bilinear[] = //
    "Compute the bilinear form between two vectors given a metric tensor.\n"
    "\n"
    "Args:\n"
    "    a (NDArray): First vector.\n"
    "    b (NDArray): Second vector.\n"
    "    metric_tensor (NDArray): The metric tensor defining the bilinear form.\n"
    "    dtype (FloatType, optional): Override the presumed input type name.\n"
    "Returns:\n"
    "    float: The bilinear form.\n"
    "\n"
    "Equivalent to: `numpy.dot` with a metric tensor.\n"
    "Signature:\n"
    "    >>> def bilinear(a, b, metric_tensor, /, dtype) -> float: ...";

static PyObject *api_bilinear(PyObject *self, PyObject *const *args, Py_ssize_t const positional_args_count,
                              PyObject *args_names_tuple) {
    return implement_curved_metric(simsimd_bilinear_k, args, positional_args_count, args_names_tuple);
}

static char const doc_mahalanobis[] = //
    "Compute the Mahalanobis distance between two vectors given an inverse covariance matrix.\n"
    "\n"
    "Args:\n"
    "    a (NDArray): First vector.\n"
    "    b (NDArray): Second vector.\n"
    "    inverse_covariance (NDArray): The inverse of the covariance matrix.\n"
    "    dtype (FloatType, optional): Override the presumed input type name.\n"
    "Returns:\n"
    "    float: The Mahalanobis distance.\n"
    "\n"
    "Equivalent to: `scipy.spatial.distance.mahalanobis`.\n"
    "Signature:\n"
    "    >>> def mahalanobis(a, b, inverse_covariance, /, dtype) -> float: ...";

static PyObject *api_mahalanobis(PyObject *self, PyObject *const *args, Py_ssize_t const positional_args_count,
                                 PyObject *args_names_tuple) {
    return implement_curved_metric(simsimd_mahalanobis_k, args, positional_args_count, args_names_tuple);
}

static char const doc_intersect[] = //
    "Compute the intersection of two sorted integer arrays.\n"
    "\n"
    "Args:\n"
    "    a (NDArray): First sorted integer array.\n"
    "    b (NDArray): Second sorted integer array.\n"
    "Returns:\n"
    "    float: The number of intersecting elements.\n"
    "\n"
    "Similar to: `numpy.intersect1d`."
    "Signature:\n"
    "    >>> def intersect(a, b, /) -> float: ...";

static PyObject *api_intersect(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    return implement_sparse_metric(simsimd_intersect_k, args, nargs);
}

static char const doc_scale[] = //
    "Scale and Shift an input vectors.\n"
    "\n"
    "Args:\n"
    "    a (NDArray): Vector.\n"
    "    dtype (Union[IntegralType, FloatType], optional): Override the presumed numeric type.\n"
    "    alpha (float, optional): First scale, 1.0 by default.\n"
    "    beta (float, optional): Shift, 0.0 by default.\n"
    "    out (NDArray, optional): Vector for resulting distances.\n"
    "Returns:\n"
    "    DistancesTensor: The distances if `out` is not provided.\n"
    "    None: If `out` is provided. Operation will per performed in-place.\n"
    "\n"
    "Equivalent to: `alpha * a + beta`.\n"
    "Signature:\n"
    "    >>> def scale(a, /, dtype, *, alpha, beta, out) -> Optional[DistancesTensor]: ...";

static PyObject *api_scale(PyObject *self, PyObject *const *args, Py_ssize_t const positional_args_count,
                           PyObject *args_names_tuple) {

    PyObject *return_obj = NULL;

    // This function accepts up to 5 arguments:
    PyObject *a_obj = NULL;     // Required object, positional-only
    PyObject *dtype_obj = NULL; // Optional object, "dtype" keyword or positional
    PyObject *out_obj = NULL;   // Optional object, "out" keyword-only
    PyObject *alpha_obj = NULL; // Optional object, "alpha" keyword-only
    PyObject *beta_obj = NULL;  // Optional object, "beta" keyword-only

    // Once parsed, the arguments will be stored in these variables:
    char const *dtype_str = NULL;
    simsimd_datatype_t dtype = simsimd_datatype_unknown_k;
    simsimd_distance_t alpha = 1, beta = 0;

    Py_buffer a_buffer, out_buffer;
    VectorOrRowsArgument a_parsed, out_parsed;
    memset(&a_buffer, 0, sizeof(Py_buffer));
    memset(&out_buffer, 0, sizeof(Py_buffer));

    Py_ssize_t const args_names_count = args_names_tuple ? PyTuple_Size(args_names_tuple) : 0;
    Py_ssize_t const args_count = positional_args_count + args_names_count;
    if (args_count < 2 || args_count > 5) {
        PyErr_Format(PyExc_TypeError, "Function expects 2-5 arguments, got %zd", args_count);
        return NULL;
    }
    if (positional_args_count > 2) {
        PyErr_Format(PyExc_TypeError, "Only first 2 arguments can be positional, received %zd", positional_args_count);
        return NULL;
    }

    // Positional-only arguments (first matrix)
    a_obj = args[0];

    // Positional or keyword arguments (dtype)
    if (positional_args_count == 2) dtype_obj = args[1];

    // The rest of the arguments must be checked in the keyword dictionary:
    for (Py_ssize_t args_names_tuple_progress = 0, args_progress = positional_args_count;
         args_names_tuple_progress < args_names_count; ++args_progress, ++args_names_tuple_progress) {
        PyObject *const key = PyTuple_GetItem(args_names_tuple, args_names_tuple_progress);
        PyObject *const value = args[args_progress];
        if (PyUnicode_CompareWithASCIIString(key, "dtype") == 0 && !dtype_obj) { dtype_obj = value; }
        else if (PyUnicode_CompareWithASCIIString(key, "out") == 0 && !out_obj) { out_obj = value; }
        else if (PyUnicode_CompareWithASCIIString(key, "alpha") == 0 && !alpha_obj) { alpha_obj = value; }
        else if (PyUnicode_CompareWithASCIIString(key, "beta") == 0 && !beta_obj) { beta_obj = value; }
        else {
            PyErr_Format(PyExc_TypeError, "Got unexpected keyword argument: %S", key);
            return NULL;
        }
    }

    // Convert `dtype_obj` to `dtype_str` and to `dtype`
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

    // Convert `alpha_obj` to `alpha` and `beta_obj` to `beta`
    if (alpha_obj) alpha = PyFloat_AsDouble(alpha_obj);
    if (beta_obj) beta = PyFloat_AsDouble(beta_obj);
    if (PyErr_Occurred()) {
        PyErr_SetString(PyExc_TypeError, "Expected 'alpha' and 'beta' to be a float");
        return NULL;
    }

    // Convert `a_obj` to `a_buffer` and to `a_parsed`.
    if (!parse_rows(a_obj, &a_buffer, &a_parsed)) return NULL;
    if (out_obj && !parse_rows(out_obj, &out_buffer, &out_parsed)) return NULL;

    // Check dimensions
    if (a_parsed.rank != 1 || (out_obj && out_parsed.rank != 1)) {
        PyErr_SetString(PyExc_ValueError, "All tensors must be vectors");
        goto cleanup;
    }
    if (out_obj && a_parsed.dimensions != out_parsed.dimensions) {
        PyErr_SetString(PyExc_ValueError, "Vector dimensions don't match");
        goto cleanup;
    }

    // Check data types
    if (a_parsed.datatype == simsimd_datatype_unknown_k ||
        (out_obj && out_parsed.datatype == simsimd_datatype_unknown_k)) {
        PyErr_SetString(PyExc_TypeError,
                        "Input tensors must have matching datatypes, check with `X.__array_interface__`");
        goto cleanup;
    }
    if (dtype == simsimd_datatype_unknown_k) dtype = a_parsed.datatype;

    // Look up the kernel and the capability
    simsimd_elementwise_scale_t kernel = NULL;
    simsimd_capability_t capability = simsimd_cap_serial_k;
    simsimd_kernel_kind_t const kernel_kind = simsimd_scale_k;
    simsimd_find_kernel(kernel_kind, dtype, static_capabilities, simsimd_cap_any_k, (simsimd_kernel_punned_t *)&kernel,
                        &capability);
    if (!kernel) {
        PyErr_Format( //
            PyExc_LookupError,
            "Unsupported kernel '%c' and datatype combination across vectors ('%s'/'%s') and "
            "`dtype` override ('%s'/'%s')",
            kernel_kind,                                                                             //
            a_buffer.format ? a_buffer.format : "nil", datatype_to_python_string(a_parsed.datatype), //
            dtype_str ? dtype_str : "nil", datatype_to_python_string(dtype));
        goto cleanup;
    }

    char *out_buffer_start = NULL;
    size_t out_buffer_stride_bytes = 0;

    // Allocate the output matrix if it wasn't provided
    if (!out_obj) {
        DistancesTensor *out_buffer_obj =
            PyObject_NewVar(DistancesTensor, &DistancesTensorType, a_parsed.dimensions * bytes_per_datatype(dtype));
        if (!out_buffer_obj) {
            PyErr_NoMemory();
            goto cleanup;
        }

        // Initialize the object
        out_buffer_obj->datatype = dtype;
        out_buffer_obj->dimensions = 1;
        out_buffer_obj->shape[0] = a_parsed.dimensions;
        out_buffer_obj->shape[1] = 1;
        out_buffer_obj->strides[0] = bytes_per_datatype(dtype);
        out_buffer_obj->strides[1] = 0;
        return_obj = (PyObject *)out_buffer_obj;
        out_buffer_start = (char *)&out_buffer_obj->start[0];
        out_buffer_stride_bytes = out_buffer_obj->strides[0];
    }
    else {
        out_buffer_start = (char *)&out_parsed.start[0];
        out_buffer_stride_bytes = out_buffer.strides[0];
        //? Logic suggests to return `None` in in-place mode...
        //? SciPy decided differently.
        return_obj = Py_None;
    }

    kernel(a_parsed.start, a_parsed.dimensions, alpha, beta, out_buffer_start);
cleanup:
    PyBuffer_Release(&a_buffer);
    PyBuffer_Release(&out_buffer);
    if (return_obj == Py_None) Py_INCREF(return_obj);
    return return_obj;
}

static char const doc_sum[] = //
    "Element-wise Sum of 2 input vectors.\n"
    "\n"
    "Args:\n"
    "    a (NDArray): First vector.\n"
    "    b (NDArray): Second vector.\n"
    "    dtype (Union[IntegralType, FloatType], optional): Override the presumed numeric type.\n"
    "    out (NDArray, optional): Vector for resulting distances.\n"
    "Returns:\n"
    "    DistancesTensor: The distances if `out` is not provided.\n"
    "    None: If `out` is provided. Operation will per performed in-place.\n"
    "\n"
    "Equivalent to: `a + b`.\n"
    "Signature:\n"
    "    >>> def sum(a, b, /, dtype, *, out) -> Optional[DistancesTensor]: ...";

static PyObject *api_sum(PyObject *self, PyObject *const *args, Py_ssize_t const positional_args_count,
                         PyObject *args_names_tuple) {

    PyObject *return_obj = NULL;

    // This function accepts up to 4 arguments:
    PyObject *a_obj = NULL;     // Required object, positional-only
    PyObject *b_obj = NULL;     // Required object, positional-only
    PyObject *dtype_obj = NULL; // Optional object, "dtype" keyword or positional
    PyObject *out_obj = NULL;   // Optional object, "out" keyword-only

    // Once parsed, the arguments will be stored in these variables:
    char const *dtype_str = NULL;
    simsimd_datatype_t dtype = simsimd_datatype_unknown_k;

    Py_buffer a_buffer, b_buffer, out_buffer;
    VectorOrRowsArgument a_parsed, b_parsed, out_parsed;
    memset(&a_buffer, 0, sizeof(Py_buffer));
    memset(&b_buffer, 0, sizeof(Py_buffer));
    memset(&out_buffer, 0, sizeof(Py_buffer));

    Py_ssize_t const args_names_count = args_names_tuple ? PyTuple_Size(args_names_tuple) : 0;
    Py_ssize_t const args_count = positional_args_count + args_names_count;
    if (args_count < 2 || args_count > 4) {
        PyErr_Format(PyExc_TypeError, "Function expects 2-4 arguments, got %zd", args_count);
        return NULL;
    }
    if (positional_args_count > 3) {
        PyErr_Format(PyExc_TypeError, "Only first 3 arguments can be positional, received %zd", positional_args_count);
        return NULL;
    }

    // Positional-only arguments (first and second matrix)
    a_obj = args[0];
    b_obj = args[1];

    // Positional or keyword arguments (dtype)
    if (positional_args_count == 3) dtype_obj = args[2];

    // The rest of the arguments must be checked in the keyword dictionary:
    for (Py_ssize_t args_names_tuple_progress = 0, args_progress = positional_args_count;
         args_names_tuple_progress < args_names_count; ++args_progress, ++args_names_tuple_progress) {
        PyObject *const key = PyTuple_GetItem(args_names_tuple, args_names_tuple_progress);
        PyObject *const value = args[args_progress];
        if (PyUnicode_CompareWithASCIIString(key, "dtype") == 0 && !dtype_obj) { dtype_obj = value; }
        else if (PyUnicode_CompareWithASCIIString(key, "out") == 0 && !out_obj) { out_obj = value; }
        else {
            PyErr_Format(PyExc_TypeError, "Got unexpected keyword argument: %S", key);
            return NULL;
        }
    }

    // Convert `dtype_obj` to `dtype_str` and to `dtype`
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

    // Convert `a_obj` to `a_buffer` and to `a_parsed`. Same for `b_obj` and `out_obj`.
    if (!parse_rows(a_obj, &a_buffer, &a_parsed) || !parse_rows(b_obj, &b_buffer, &b_parsed)) return NULL;
    if (out_obj && !parse_rows(out_obj, &out_buffer, &out_parsed)) return NULL;

    // Check dimensions
    if (a_parsed.rank != 1 || b_parsed.rank != 1 || (out_obj && out_parsed.rank != 1)) {
        PyErr_SetString(PyExc_ValueError, "All tensors must be vectors");
        goto cleanup;
    }
    if (a_parsed.dimensions != b_parsed.dimensions || (out_obj && a_parsed.dimensions != out_parsed.dimensions)) {
        PyErr_SetString(PyExc_ValueError, "Vector dimensions don't match");
        goto cleanup;
    }

    // Check data types
    if (a_parsed.datatype != b_parsed.datatype || a_parsed.datatype == simsimd_datatype_unknown_k ||
        b_parsed.datatype == simsimd_datatype_unknown_k ||
        (out_obj && out_parsed.datatype == simsimd_datatype_unknown_k)) {
        PyErr_SetString(PyExc_TypeError,
                        "Input tensors must have matching datatypes, check with `X.__array_interface__`");
        goto cleanup;
    }
    if (dtype == simsimd_datatype_unknown_k) dtype = a_parsed.datatype;

    // Look up the kernel and the capability
    simsimd_elementwise_sum_t kernel = NULL;
    simsimd_capability_t capability = simsimd_cap_serial_k;
    simsimd_kernel_kind_t const kernel_kind = simsimd_sum_k;
    simsimd_find_kernel(kernel_kind, dtype, static_capabilities, simsimd_cap_any_k, (simsimd_kernel_punned_t *)&kernel,
                        &capability);
    if (!kernel) {
        PyErr_Format( //
            PyExc_LookupError,
            "Unsupported kernel '%c' and datatype combination across vectors ('%s'/'%s') and "
            "`dtype` override ('%s'/'%s')",
            kernel_kind,                                                                             //
            a_buffer.format ? a_buffer.format : "nil", datatype_to_python_string(a_parsed.datatype), //
            dtype_str ? dtype_str : "nil", datatype_to_python_string(dtype));
        goto cleanup;
    }

    char *out_buffer_start = NULL;
    size_t out_buffer_stride_bytes = 0;

    // Allocate the output matrix if it wasn't provided
    if (!out_obj) {
        DistancesTensor *out_buffer_obj =
            PyObject_NewVar(DistancesTensor, &DistancesTensorType, a_parsed.dimensions * bytes_per_datatype(dtype));
        if (!out_buffer_obj) {
            PyErr_NoMemory();
            goto cleanup;
        }

        // Initialize the object
        out_buffer_obj->datatype = dtype;
        out_buffer_obj->dimensions = 1;
        out_buffer_obj->shape[0] = a_parsed.dimensions;
        out_buffer_obj->shape[1] = 1;
        out_buffer_obj->strides[0] = bytes_per_datatype(dtype);
        out_buffer_obj->strides[1] = 0;
        return_obj = (PyObject *)out_buffer_obj;
        out_buffer_start = (char *)&out_buffer_obj->start[0];
        out_buffer_stride_bytes = out_buffer_obj->strides[0];
    }
    else {
        out_buffer_start = (char *)&out_parsed.start[0];
        out_buffer_stride_bytes = out_buffer.strides[0];
        //? Logic suggests to return `None` in in-place mode...
        //? SciPy decided differently.
        return_obj = Py_None;
    }

    kernel(a_parsed.start, b_parsed.start, a_parsed.dimensions, out_buffer_start);
cleanup:
    PyBuffer_Release(&a_buffer);
    PyBuffer_Release(&b_buffer);
    PyBuffer_Release(&out_buffer);
    if (return_obj == Py_None) Py_INCREF(return_obj);
    return return_obj;
}

static char const doc_wsum[] = //
    "Weighted Sum of 2 input vectors.\n"
    "\n"
    "Args:\n"
    "    a (NDArray): First vector.\n"
    "    b (NDArray): Second vector.\n"
    "    dtype (Union[IntegralType, FloatType], optional): Override the presumed numeric type.\n"
    "    alpha (float, optional): First scale, 1.0 by default.\n"
    "    beta (float, optional): Second scale, 1.0 by default.\n"
    "    out (NDArray, optional): Vector for resulting distances.\n"
    "Returns:\n"
    "    DistancesTensor: The distances if `out` is not provided.\n"
    "    None: If `out` is provided. Operation will per performed in-place.\n"
    "\n"
    "Equivalent to: `alpha * a + beta * b`.\n"
    "Signature:\n"
    "    >>> def wsum(a, b, /, dtype, *, alpha, beta, out) -> Optional[DistancesTensor]: ...";

static PyObject *api_wsum(PyObject *self, PyObject *const *args, Py_ssize_t const positional_args_count,
                          PyObject *args_names_tuple) {

    PyObject *return_obj = NULL;

    // This function accepts up to 6 arguments:
    PyObject *a_obj = NULL;     // Required object, positional-only
    PyObject *b_obj = NULL;     // Required object, positional-only
    PyObject *dtype_obj = NULL; // Optional object, "dtype" keyword or positional
    PyObject *out_obj = NULL;   // Optional object, "out" keyword-only
    PyObject *alpha_obj = NULL; // Optional object, "alpha" keyword-only
    PyObject *beta_obj = NULL;  // Optional object, "beta" keyword-only

    // Once parsed, the arguments will be stored in these variables:
    char const *dtype_str = NULL;
    simsimd_datatype_t dtype = simsimd_datatype_unknown_k;
    simsimd_distance_t alpha = 1, beta = 1;

    Py_buffer a_buffer, b_buffer, out_buffer;
    VectorOrRowsArgument a_parsed, b_parsed, out_parsed;
    memset(&a_buffer, 0, sizeof(Py_buffer));
    memset(&b_buffer, 0, sizeof(Py_buffer));
    memset(&out_buffer, 0, sizeof(Py_buffer));

    Py_ssize_t const args_names_count = args_names_tuple ? PyTuple_Size(args_names_tuple) : 0;
    Py_ssize_t const args_count = positional_args_count + args_names_count;
    if (args_count < 2 || args_count > 6) {
        PyErr_Format(PyExc_TypeError, "Function expects 2-6 arguments, got %zd", args_count);
        return NULL;
    }
    if (positional_args_count > 3) {
        PyErr_Format(PyExc_TypeError, "Only first 3 arguments can be positional, received %zd", positional_args_count);
        return NULL;
    }

    // Positional-only arguments (first and second matrix)
    a_obj = args[0];
    b_obj = args[1];

    // Positional or keyword arguments (dtype)
    if (positional_args_count == 3) dtype_obj = args[2];

    // The rest of the arguments must be checked in the keyword dictionary:
    for (Py_ssize_t args_names_tuple_progress = 0, args_progress = positional_args_count;
         args_names_tuple_progress < args_names_count; ++args_progress, ++args_names_tuple_progress) {
        PyObject *const key = PyTuple_GetItem(args_names_tuple, args_names_tuple_progress);
        PyObject *const value = args[args_progress];
        if (PyUnicode_CompareWithASCIIString(key, "dtype") == 0 && !dtype_obj) { dtype_obj = value; }
        else if (PyUnicode_CompareWithASCIIString(key, "out") == 0 && !out_obj) { out_obj = value; }
        else if (PyUnicode_CompareWithASCIIString(key, "alpha") == 0 && !alpha_obj) { alpha_obj = value; }
        else if (PyUnicode_CompareWithASCIIString(key, "beta") == 0 && !beta_obj) { beta_obj = value; }
        else {
            PyErr_Format(PyExc_TypeError, "Got unexpected keyword argument: %S", key);
            return NULL;
        }
    }

    // Convert `dtype_obj` to `dtype_str` and to `dtype`
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

    // Convert `alpha_obj` to `alpha` and `beta_obj` to `beta`
    if (alpha_obj) alpha = PyFloat_AsDouble(alpha_obj);
    if (beta_obj) beta = PyFloat_AsDouble(beta_obj);
    if (PyErr_Occurred()) {
        PyErr_SetString(PyExc_TypeError, "Expected 'alpha' and 'beta' to be a float");
        return NULL;
    }

    // Convert `a_obj` to `a_buffer` and to `a_parsed`. Same for `b_obj` and `out_obj`.
    if (!parse_rows(a_obj, &a_buffer, &a_parsed) || !parse_rows(b_obj, &b_buffer, &b_parsed)) return NULL;
    if (out_obj && !parse_rows(out_obj, &out_buffer, &out_parsed)) return NULL;

    // Check dimensions
    if (a_parsed.rank != 1 || b_parsed.rank != 1 || (out_obj && out_parsed.rank != 1)) {
        PyErr_SetString(PyExc_ValueError, "All tensors must be vectors");
        goto cleanup;
    }
    if (a_parsed.dimensions != b_parsed.dimensions || (out_obj && a_parsed.dimensions != out_parsed.dimensions)) {
        PyErr_SetString(PyExc_ValueError, "Vector dimensions don't match");
        goto cleanup;
    }

    // Check data types
    if (a_parsed.datatype != b_parsed.datatype || a_parsed.datatype == simsimd_datatype_unknown_k ||
        b_parsed.datatype == simsimd_datatype_unknown_k ||
        (out_obj && out_parsed.datatype == simsimd_datatype_unknown_k)) {
        PyErr_SetString(PyExc_TypeError,
                        "Input tensors must have matching datatypes, check with `X.__array_interface__`");
        goto cleanup;
    }
    if (dtype == simsimd_datatype_unknown_k) dtype = a_parsed.datatype;

    // Look up the kernel and the capability
    simsimd_elementwise_wsum_t kernel = NULL;
    simsimd_capability_t capability = simsimd_cap_serial_k;
    simsimd_kernel_kind_t const kernel_kind = simsimd_wsum_k;
    simsimd_find_kernel(kernel_kind, dtype, static_capabilities, simsimd_cap_any_k, (simsimd_kernel_punned_t *)&kernel,
                        &capability);
    if (!kernel) {
        PyErr_Format( //
            PyExc_LookupError,
            "Unsupported kernel '%c' and datatype combination across vectors ('%s'/'%s') and "
            "`dtype` override ('%s'/'%s')",
            kernel_kind,                                                                             //
            a_buffer.format ? a_buffer.format : "nil", datatype_to_python_string(a_parsed.datatype), //
            dtype_str ? dtype_str : "nil", datatype_to_python_string(dtype));
        goto cleanup;
    }

    char *out_buffer_start = NULL;
    size_t out_buffer_stride_bytes = 0;

    // Allocate the output matrix if it wasn't provided
    if (!out_obj) {
        DistancesTensor *out_buffer_obj =
            PyObject_NewVar(DistancesTensor, &DistancesTensorType, a_parsed.dimensions * bytes_per_datatype(dtype));
        if (!out_buffer_obj) {
            PyErr_NoMemory();
            goto cleanup;
        }

        // Initialize the object
        out_buffer_obj->datatype = dtype;
        out_buffer_obj->dimensions = 1;
        out_buffer_obj->shape[0] = a_parsed.dimensions;
        out_buffer_obj->shape[1] = 1;
        out_buffer_obj->strides[0] = bytes_per_datatype(dtype);
        out_buffer_obj->strides[1] = 0;
        return_obj = (PyObject *)out_buffer_obj;
        out_buffer_start = (char *)&out_buffer_obj->start[0];
        out_buffer_stride_bytes = out_buffer_obj->strides[0];
    }
    else {
        out_buffer_start = (char *)&out_parsed.start[0];
        out_buffer_stride_bytes = out_buffer.strides[0];
        //? Logic suggests to return `None` in in-place mode...
        //? SciPy decided differently.
        return_obj = Py_None;
    }

    kernel(a_parsed.start, b_parsed.start, a_parsed.dimensions, alpha, beta, out_buffer_start);
cleanup:
    PyBuffer_Release(&a_buffer);
    PyBuffer_Release(&b_buffer);
    PyBuffer_Release(&out_buffer);
    if (return_obj == Py_None) Py_INCREF(return_obj);
    return return_obj;
}

static char const doc_fma[] = //
    "Fused-Multiply-Add between 3 input vectors.\n"
    "\n"
    "Args:\n"
    "    a (NDArray): First vector.\n"
    "    b (NDArray): Second vector.\n"
    "    c (NDArray): Third vector.\n"
    "    dtype (Union[IntegralType, FloatType], optional): Override the presumed numeric type name.\n"
    "    alpha (float, optional): First scale, 1.0 by default.\n"
    "    beta (float, optional): Second scale, 1.0 by default.\n"
    "    out (NDArray, optional): Vector for resulting distances.\n"
    "Returns:\n"
    "    DistancesTensor: The distances if `out` is not provided.\n"
    "    None: If `out` is provided. Operation will per performed in-place.\n"
    "\n"
    "Equivalent to: `alpha * a * b + beta * c`.\n"
    "Signature:\n"
    "    >>> def fma(a, b, c, /, dtype, *, alpha, beta, out) -> Optional[DistancesTensor]: ...";

static PyObject *api_fma(PyObject *self, PyObject *const *args, Py_ssize_t const positional_args_count,
                         PyObject *args_names_tuple) {

    PyObject *return_obj = NULL;

    // This function accepts up to 5 arguments:
    PyObject *a_obj = NULL;     // Required object, positional-only
    PyObject *b_obj = NULL;     // Required object, positional-only
    PyObject *c_obj = NULL;     // Required object, positional-only
    PyObject *dtype_obj = NULL; // Optional object, "dtype" keyword or positional
    PyObject *out_obj = NULL;   // Optional object, "out" keyword-only
    PyObject *alpha_obj = NULL; // Optional object, "alpha" keyword-only
    PyObject *beta_obj = NULL;  // Optional object, "beta" keyword-only

    // Once parsed, the arguments will be stored in these variables:
    char const *dtype_str = NULL;
    simsimd_datatype_t dtype = simsimd_datatype_unknown_k;
    simsimd_distance_t alpha = 1, beta = 1;

    Py_buffer a_buffer, b_buffer, c_buffer, out_buffer;
    VectorOrRowsArgument a_parsed, b_parsed, c_parsed, out_parsed;
    memset(&a_buffer, 0, sizeof(Py_buffer));
    memset(&b_buffer, 0, sizeof(Py_buffer));
    memset(&c_buffer, 0, sizeof(Py_buffer));
    memset(&out_buffer, 0, sizeof(Py_buffer));

    Py_ssize_t const args_names_count = args_names_tuple ? PyTuple_Size(args_names_tuple) : 0;
    Py_ssize_t const args_count = positional_args_count + args_names_count;
    if (args_count < 3 || args_count > 7) {
        PyErr_Format(PyExc_TypeError, "Function expects 3-7 arguments, got %zd", args_count);
        return NULL;
    }
    if (positional_args_count > 4) {
        PyErr_Format(PyExc_TypeError, "Only first 4 arguments can be positional, received %zd", positional_args_count);
        return NULL;
    }

    // Positional-only arguments (first and second matrix)
    a_obj = args[0];
    b_obj = args[1];
    c_obj = args[2];

    // Positional or keyword arguments (dtype)
    if (positional_args_count == 4) dtype_obj = args[3];

    // The rest of the arguments must be checked in the keyword dictionary:
    for (Py_ssize_t args_names_tuple_progress = 0, args_progress = positional_args_count;
         args_names_tuple_progress < args_names_count; ++args_progress, ++args_names_tuple_progress) {
        PyObject *const key = PyTuple_GetItem(args_names_tuple, args_names_tuple_progress);
        PyObject *const value = args[args_progress];
        if (PyUnicode_CompareWithASCIIString(key, "dtype") == 0 && !dtype_obj) { dtype_obj = value; }
        else if (PyUnicode_CompareWithASCIIString(key, "out") == 0 && !out_obj) { out_obj = value; }
        else if (PyUnicode_CompareWithASCIIString(key, "alpha") == 0 && !alpha_obj) { alpha_obj = value; }
        else if (PyUnicode_CompareWithASCIIString(key, "beta") == 0 && !beta_obj) { beta_obj = value; }
        else {
            PyErr_Format(PyExc_TypeError, "Got unexpected keyword argument: %S", key);
            return NULL;
        }
    }

    // Convert `dtype_obj` to `dtype_str` and to `dtype`
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

    // Convert `alpha_obj` to `alpha` and `beta_obj` to `beta`
    if (alpha_obj) alpha = PyFloat_AsDouble(alpha_obj);
    if (beta_obj) beta = PyFloat_AsDouble(beta_obj);
    if (PyErr_Occurred()) {
        PyErr_SetString(PyExc_TypeError, "Expected 'alpha' and 'beta' to be a float");
        return NULL;
    }

    // Convert `a_obj` to `a_buffer` and to `a_parsed`. Same for `b_obj` and `out_obj`.
    if (!parse_rows(a_obj, &a_buffer, &a_parsed) || !parse_rows(b_obj, &b_buffer, &b_parsed) ||
        !parse_rows(c_obj, &c_buffer, &c_parsed))
        return NULL;
    if (out_obj && !parse_rows(out_obj, &out_buffer, &out_parsed)) return NULL;

    // Check dimensions
    if (a_parsed.rank != 1 || b_parsed.rank != 1 || c_parsed.rank != 1 || (out_obj && out_parsed.rank != 1)) {
        PyErr_SetString(PyExc_ValueError, "All tensors must be vectors");
        goto cleanup;
    }
    if (a_parsed.dimensions != b_parsed.dimensions || a_parsed.dimensions != c_parsed.dimensions ||
        (out_obj && a_parsed.dimensions != out_parsed.dimensions)) {
        PyErr_SetString(PyExc_ValueError, "Vector dimensions don't match");
        goto cleanup;
    }

    // Check data types
    if (a_parsed.datatype != b_parsed.datatype || a_parsed.datatype == simsimd_datatype_unknown_k ||
        b_parsed.datatype == simsimd_datatype_unknown_k || c_parsed.datatype == simsimd_datatype_unknown_k ||
        (out_obj && out_parsed.datatype == simsimd_datatype_unknown_k)) {
        PyErr_SetString(PyExc_TypeError,
                        "Input tensors must have matching datatypes, check with `X.__array_interface__`");
        goto cleanup;
    }
    if (dtype == simsimd_datatype_unknown_k) dtype = a_parsed.datatype;

    // Look up the kernel and the capability
    simsimd_elementwise_fma_t kernel = NULL;
    simsimd_capability_t capability = simsimd_cap_serial_k;
    simsimd_kernel_kind_t const kernel_kind = simsimd_fma_k;
    simsimd_find_kernel(kernel_kind, dtype, static_capabilities, simsimd_cap_any_k, (simsimd_kernel_punned_t *)&kernel,
                        &capability);
    if (!kernel) {
        PyErr_Format( //
            PyExc_LookupError,
            "Unsupported kernel '%c' and datatype combination across vectors ('%s'/'%s') and "
            "`dtype` override ('%s'/'%s')",
            kernel_kind,                                                                             //
            a_buffer.format ? a_buffer.format : "nil", datatype_to_python_string(a_parsed.datatype), //
            dtype_str ? dtype_str : "nil", datatype_to_python_string(dtype));
        goto cleanup;
    }

    char *out_buffer_start = NULL;
    size_t out_buffer_stride_bytes = 0;

    // Allocate the output matrix if it wasn't provided
    if (!out_obj) {
        DistancesTensor *out_buffer_obj =
            PyObject_NewVar(DistancesTensor, &DistancesTensorType, a_parsed.dimensions * bytes_per_datatype(dtype));
        if (!out_buffer_obj) {
            PyErr_NoMemory();
            goto cleanup;
        }

        // Initialize the object
        out_buffer_obj->datatype = dtype;
        out_buffer_obj->dimensions = 1;
        out_buffer_obj->shape[0] = a_parsed.dimensions;
        out_buffer_obj->shape[1] = 1;
        out_buffer_obj->strides[0] = bytes_per_datatype(dtype);
        out_buffer_obj->strides[1] = 0;
        return_obj = (PyObject *)out_buffer_obj;
        out_buffer_start = (char *)&out_buffer_obj->start[0];
        out_buffer_stride_bytes = out_buffer_obj->strides[0];
    }
    else {
        out_buffer_start = (char *)&out_parsed.start[0];
        out_buffer_stride_bytes = out_buffer.strides[0];
        //? Logic suggests to return `None` in in-place mode...
        //? SciPy decided differently.
        return_obj = Py_None;
    }

    kernel(a_parsed.start, b_parsed.start, c_parsed.start, a_parsed.dimensions, alpha, beta, out_buffer_start);
cleanup:
    PyBuffer_Release(&a_buffer);
    PyBuffer_Release(&b_buffer);
    PyBuffer_Release(&c_buffer);
    PyBuffer_Release(&out_buffer);
    if (return_obj == Py_None) Py_INCREF(return_obj);
    return return_obj;
}

typedef void (*binary_kernel_t)(void const *, void const *, void *);
typedef void (*unary_kernel_t)(void const *, void *);

void apply_elementwise_binary_operation_to_each_scalar( //
    BufferOrScalarArgument const *a_parsed, BufferOrScalarArgument const *b_parsed,
    BufferOrScalarArgument const *out_parsed, //
    binary_kernel_t elementwise_kernel) {

    // The hardest part of this operations is addressing the elements in a non-continuous tensor of arbitrary rank.
    // While iteratively deepening into the lower layers of the tensor, we need to keep track of the byte offsets
    // for each dimension to avoid recomputing them in the inner loops.
    simsimd_mdindices_t a_mdindices, b_mdindices, out_mdindices;
    memset(&a_mdindices, 0, sizeof(simsimd_mdindices_t));
    memset(&b_mdindices, 0, sizeof(simsimd_mdindices_t));
    memset(&out_mdindices, 0, sizeof(simsimd_mdindices_t));

    // Start from last dimension and move backward, replicating the logic
    // of `simsimd_mdindices_next`, broadcasting the same update logic across the
    // indexes in all three tensors, and avoiding additional branches inside the loops.
    while (1) {
        // Invoke the provided kernel at the current byte offsets
        elementwise_kernel(a_parsed->as_buffer_start + a_mdindices.byte_offset,
                           b_parsed->as_buffer_start + b_mdindices.byte_offset,
                           out_parsed->as_buffer_start + out_mdindices.byte_offset);

        // Advance to the next index
        Py_ssize_t dim;
        for (dim = out_parsed->as_buffer_dimensions - 1; dim >= 0; --dim) {
            out_mdindices.coordinates[dim]++;
            out_mdindices.byte_offset += out_parsed->as_buffer_strides[dim];
            a_mdindices.byte_offset += a_parsed->as_buffer_strides[dim];
            b_mdindices.byte_offset += b_parsed->as_buffer_strides[dim];

            // Successfully moved to the next index in this dimension
            if (out_mdindices.coordinates[dim] < out_parsed->as_buffer_shape[dim]) break;
            else {
                // Reset coordinates and byte offset for this dimension
                out_mdindices.coordinates[dim] = 0;
                out_mdindices.byte_offset -= out_parsed->as_buffer_strides[dim] * out_parsed->as_buffer_shape[dim];
                a_mdindices.byte_offset -= a_parsed->as_buffer_strides[dim] * out_parsed->as_buffer_shape[dim];
                b_mdindices.byte_offset -= b_parsed->as_buffer_strides[dim] * out_parsed->as_buffer_shape[dim];
            }
        }

        // If we've processed all dimensions, we're done
        if (dim < 0) break;
    }
}

void apply_elementwise_casting_binary_operation_to_each_scalar( //
    BufferOrScalarArgument const *a_parsed, BufferOrScalarArgument const *b_parsed,
    BufferOrScalarArgument const *out_parsed,                       //
    unary_kernel_t a_upcast_kernel, unary_kernel_t b_upcast_kernel, //
    unary_kernel_t out_downcast_kernel,                             //
    binary_kernel_t elementwise_kernel) {

    // The hardest part of this operations is addressing the elements in a non-continuous tensor of arbitrary rank.
    // While iteratively deepening into the lower layers of the tensor, we need to keep track of the byte offsets
    // for each dimension to avoid recomputing them in the inner loops.
    simsimd_mdindices_t a_mdindices, b_mdindices, out_mdindices;
    memset(&a_mdindices, 0, sizeof(simsimd_mdindices_t));
    memset(&b_mdindices, 0, sizeof(simsimd_mdindices_t));
    memset(&out_mdindices, 0, sizeof(simsimd_mdindices_t));

    char a_upcast_buffer[8], b_upcast_buffer[8], out_downcast_buffer[8];

    // Start from last dimension and move backward, replicating the logic
    // of `simsimd_mdindices_next`, broadcasting the same update logic across the
    // indexes in all three tensors, and avoiding additional branches inside the loops.
    while (1) {
        // Invoke the provided kernel at the current byte offsets
        a_upcast_kernel(a_parsed->as_buffer_start + a_mdindices.byte_offset, a_upcast_buffer);
        b_upcast_kernel(b_parsed->as_buffer_start + b_mdindices.byte_offset, b_upcast_buffer);
        elementwise_kernel(a_upcast_buffer, b_upcast_buffer, out_downcast_buffer);
        out_downcast_kernel(out_downcast_buffer, out_parsed->as_buffer_start + out_mdindices.byte_offset);

        // Advance to the next index
        Py_ssize_t dim;
        for (dim = out_parsed->as_buffer_dimensions - 1; dim >= 0; --dim) {
            out_mdindices.coordinates[dim]++;
            out_mdindices.byte_offset += out_parsed->as_buffer_strides[dim];
            a_mdindices.byte_offset += a_parsed->as_buffer_strides[dim];
            b_mdindices.byte_offset += b_parsed->as_buffer_strides[dim];

            // Successfully moved to the next index in this dimension
            if (out_mdindices.coordinates[dim] < out_parsed->as_buffer_shape[dim]) break;
            else {
                // Reset coordinates and byte offset for this dimension
                out_mdindices.coordinates[dim] = 0;
                out_mdindices.byte_offset -= out_parsed->as_buffer_strides[dim] * out_parsed->as_buffer_shape[dim];
                a_mdindices.byte_offset -= a_parsed->as_buffer_strides[dim] * out_parsed->as_buffer_shape[dim];
                b_mdindices.byte_offset -= b_parsed->as_buffer_strides[dim] * out_parsed->as_buffer_shape[dim];
            }
        }

        // If we've processed all dimensions, we're done
        if (dim < 0) break;
    }
}

void apply_elementwise_binary_operation_to_each_continuous_slice( //
    BufferOrScalarArgument const *a_parsed, BufferOrScalarArgument const *b_parsed,
    BufferOrScalarArgument const *out_parsed, //
    Py_ssize_t const non_continuous_ranks,    //
    Py_ssize_t const continuous_elements,     //
    void (*binary_kernel)(void const *, void const *, simsimd_size_t, void *)) {

    // The hardest part of this operations is addressing the elements in a non-continuous tensor of arbitrary rank.
    // While iteratively deepening into the lower layers of the tensor, we need to keep track of the byte offsets
    // for each dimension to avoid recomputing them in the inner loops.
    simsimd_mdindices_t a_mdindices, b_mdindices, out_mdindices;
    memset(&a_mdindices, 0, sizeof(simsimd_mdindices_t));
    memset(&b_mdindices, 0, sizeof(simsimd_mdindices_t));
    memset(&out_mdindices, 0, sizeof(simsimd_mdindices_t));

    // Start from last dimension and move backward, replicating the logic
    // of `simsimd_mdindices_next`, broadcasting the same update logic across the
    // indexes in all three tensors, and avoiding additional branches inside the loops.
    while (1) {
        // Invoke the provided kernel at the current byte offsets
        binary_kernel(a_parsed->as_buffer_start + a_mdindices.byte_offset,
                      b_parsed->as_buffer_start + b_mdindices.byte_offset, continuous_elements,
                      out_parsed->as_buffer_start + out_mdindices.byte_offset);

        // Advance to the next index
        Py_ssize_t dim;
        for (dim = non_continuous_ranks - 1; dim >= 0; --dim) {
            out_mdindices.coordinates[dim]++;
            out_mdindices.byte_offset += out_parsed->as_buffer_strides[dim];
            a_mdindices.byte_offset += a_parsed->as_buffer_strides[dim];
            b_mdindices.byte_offset += b_parsed->as_buffer_strides[dim];

            // Successfully moved to the next index in this dimension
            if (out_mdindices.coordinates[dim] < out_parsed->as_buffer_shape[dim]) break;
            else {
                // Reset coordinates and byte offset for this dimension
                out_mdindices.coordinates[dim] = 0;
                out_mdindices.byte_offset -= out_parsed->as_buffer_strides[dim] * out_parsed->as_buffer_shape[dim];
                a_mdindices.byte_offset -= a_parsed->as_buffer_strides[dim] * out_parsed->as_buffer_shape[dim];
                b_mdindices.byte_offset -= b_parsed->as_buffer_strides[dim] * out_parsed->as_buffer_shape[dim];
            }
        }

        // If we've processed all dimensions, we're done
        if (dim < 0) break;
    }
}

void apply_scale_to_each_continuous_slice(                                               //
    BufferOrScalarArgument const *input_parsed, simsimd_f64_t alpha, simsimd_f64_t beta, //
    BufferOrScalarArgument const *out_parsed,                                            //
    Py_ssize_t const non_continuous_ranks,                                               //
    Py_ssize_t const continuous_elements,                                                //
    void (*binary_kernel)(void const *, simsimd_size_t, simsimd_f64_t, simsimd_f64_t, void *)) {

    // The hardest part of this operations is addressing the elements in a non-continuous tensor of arbitrary rank.
    // While iteratively deepening into the lower layers of the tensor, we need to keep track of the byte offsets
    // for each dimension to avoid recomputing them in the inner loops.
    simsimd_mdindices_t input_mdindices, out_mdindices;
    memset(&input_mdindices, 0, sizeof(simsimd_mdindices_t));
    memset(&out_mdindices, 0, sizeof(simsimd_mdindices_t));

    // Start from last dimension and move backward, replicating the logic
    // of `simsimd_mdindices_next`, broadcasting the same update logic across the
    // indexes in all three tensors, and avoiding additional branches inside the loops.
    while (1) {
        // Invoke the provided kernel at the current byte offsets
        binary_kernel(input_parsed->as_buffer_start + input_mdindices.byte_offset, continuous_elements, alpha, beta,
                      out_parsed->as_buffer_start + out_mdindices.byte_offset);

        // Advance to the next index
        Py_ssize_t dim;
        for (dim = non_continuous_ranks - 1; dim >= 0; --dim) {
            out_mdindices.coordinates[dim]++;
            out_mdindices.byte_offset += out_parsed->as_buffer_strides[dim];
            input_mdindices.byte_offset += input_parsed->as_buffer_strides[dim];

            // Successfully moved to the next index in this dimension
            if (out_mdindices.coordinates[dim] < out_parsed->as_buffer_shape[dim]) break;
            else {
                // Reset coordinates and byte offset for this dimension
                out_mdindices.coordinates[dim] = 0;
                out_mdindices.byte_offset -= out_parsed->as_buffer_strides[dim] * out_parsed->as_buffer_shape[dim];
                input_mdindices.byte_offset -= input_parsed->as_buffer_strides[dim] * out_parsed->as_buffer_shape[dim];
            }
        }

        // If we've processed all dimensions, we're done
        if (dim < 0) break;
    }
}

static binary_kernel_t elementwise_sadd(simsimd_datatype_t dtype) {
    switch (dtype) {
    case simsimd_u64_k: return (binary_kernel_t)&_simsimd_u64_sadd;
    case simsimd_u32_k: return (binary_kernel_t)&_simsimd_u32_sadd;
    case simsimd_u16_k: return (binary_kernel_t)&_simsimd_u16_sadd;
    case simsimd_u8_k: return (binary_kernel_t)&_simsimd_u8_sadd;
    case simsimd_i64_k: return (binary_kernel_t)&_simsimd_i64_sadd;
    case simsimd_i32_k: return (binary_kernel_t)&_simsimd_i32_sadd;
    case simsimd_i16_k: return (binary_kernel_t)&_simsimd_i16_sadd;
    case simsimd_i8_k: return (binary_kernel_t)&_simsimd_i8_sadd;
    case simsimd_f64_k: return (binary_kernel_t)&_simsimd_f64_sadd;
    case simsimd_f32_k: return (binary_kernel_t)&_simsimd_f32_sadd;
    case simsimd_f16_k: return (binary_kernel_t)&_simsimd_f16_sadd;
    case simsimd_bf16_k: return (binary_kernel_t)&_simsimd_bf16_sadd;
    default: return NULL;
    }
}

static binary_kernel_t elementwise_smul(simsimd_datatype_t dtype) {
    switch (dtype) {
    case simsimd_u64_k: return (binary_kernel_t)&_simsimd_u64_smul;
    case simsimd_u32_k: return (binary_kernel_t)&_simsimd_u32_smul;
    case simsimd_u16_k: return (binary_kernel_t)&_simsimd_u16_smul;
    case simsimd_u8_k: return (binary_kernel_t)&_simsimd_u8_smul;
    case simsimd_i64_k: return (binary_kernel_t)&_simsimd_i64_smul;
    case simsimd_i32_k: return (binary_kernel_t)&_simsimd_i32_smul;
    case simsimd_i16_k: return (binary_kernel_t)&_simsimd_i16_smul;
    case simsimd_i8_k: return (binary_kernel_t)&_simsimd_i8_smul;
    case simsimd_f64_k: return (binary_kernel_t)&_simsimd_f64_smul;
    case simsimd_f32_k: return (binary_kernel_t)&_simsimd_f32_smul;
    case simsimd_f16_k: return (binary_kernel_t)&_simsimd_f16_smul;
    case simsimd_bf16_k: return (binary_kernel_t)&_simsimd_bf16_smul;
    default: return NULL;
    }
}

static unary_kernel_t elementwise_upcast_to_f64(simsimd_datatype_t dtype) {
    switch (dtype) {
    case simsimd_u64_k: return (unary_kernel_t)&_simsimd_u64_to_f64;
    case simsimd_u32_k: return (unary_kernel_t)&_simsimd_u32_to_f64;
    case simsimd_u16_k: return (unary_kernel_t)&_simsimd_u16_to_f64;
    case simsimd_u8_k: return (unary_kernel_t)&_simsimd_u8_to_f64;
    case simsimd_i64_k: return (unary_kernel_t)&_simsimd_i64_to_f64;
    case simsimd_i32_k: return (unary_kernel_t)&_simsimd_i32_to_f64;
    case simsimd_i16_k: return (unary_kernel_t)&_simsimd_i16_to_f64;
    case simsimd_i8_k: return (unary_kernel_t)&_simsimd_i8_to_f64;
    case simsimd_f64_k: return (unary_kernel_t)&_simsimd_f64_to_f64;
    case simsimd_f32_k: return (unary_kernel_t)&_simsimd_f32_to_f64;
    case simsimd_f16_k: return (unary_kernel_t)&_simsimd_f16_to_f64;
    case simsimd_bf16_k: return (unary_kernel_t)&_simsimd_bf16_to_f64;
    default: return NULL;
    }
}

static unary_kernel_t elementwise_upcast_to_i64(simsimd_datatype_t dtype) {
    switch (dtype) {
    case simsimd_u64_k: return (unary_kernel_t)&_simsimd_u64_to_i64;
    case simsimd_u32_k: return (unary_kernel_t)&_simsimd_u32_to_i64;
    case simsimd_u16_k: return (unary_kernel_t)&_simsimd_u16_to_i64;
    case simsimd_u8_k: return (unary_kernel_t)&_simsimd_u8_to_i64;
    case simsimd_i64_k: return (unary_kernel_t)&_simsimd_i64_to_i64;
    case simsimd_i32_k: return (unary_kernel_t)&_simsimd_i32_to_i64;
    case simsimd_i16_k: return (unary_kernel_t)&_simsimd_i16_to_i64;
    case simsimd_i8_k: return (unary_kernel_t)&_simsimd_i8_to_i64;
    case simsimd_f64_k: return NULL;
    case simsimd_f32_k: return NULL;
    case simsimd_f16_k: return NULL;
    case simsimd_bf16_k: return NULL;
    default: return NULL;
    }
}

static unary_kernel_t elementwise_upcast_to_u64(simsimd_datatype_t dtype) {
    switch (dtype) {
    case simsimd_u64_k: return (unary_kernel_t)&_simsimd_u64_to_u64;
    case simsimd_u32_k: return (unary_kernel_t)&_simsimd_u32_to_u64;
    case simsimd_u16_k: return (unary_kernel_t)&_simsimd_u16_to_u64;
    case simsimd_u8_k: return (unary_kernel_t)&_simsimd_u8_to_u64;
    case simsimd_i64_k: return (unary_kernel_t)&_simsimd_i64_to_u64;
    case simsimd_i32_k: return (unary_kernel_t)&_simsimd_i32_to_u64;
    case simsimd_i16_k: return (unary_kernel_t)&_simsimd_i16_to_u64;
    case simsimd_i8_k: return (unary_kernel_t)&_simsimd_i8_to_u64;
    case simsimd_f64_k: return NULL;
    case simsimd_f32_k: return NULL;
    case simsimd_f16_k: return NULL;
    case simsimd_bf16_k: return NULL;
    default: return NULL;
    }
}

static unary_kernel_t elementwise_downcast_from_f64(simsimd_datatype_t dtype) {
    switch (dtype) {
    case simsimd_u64_k: return (unary_kernel_t)&_simsimd_f64_to_u64;
    case simsimd_u32_k: return (unary_kernel_t)&_simsimd_f64_to_u32;
    case simsimd_u16_k: return (unary_kernel_t)&_simsimd_f64_to_u16;
    case simsimd_u8_k: return (unary_kernel_t)&_simsimd_f64_to_u8;
    case simsimd_i64_k: return (unary_kernel_t)&_simsimd_f64_to_i64;
    case simsimd_i32_k: return (unary_kernel_t)&_simsimd_f64_to_i32;
    case simsimd_i16_k: return (unary_kernel_t)&_simsimd_f64_to_i16;
    case simsimd_i8_k: return (unary_kernel_t)&_simsimd_f64_to_i8;
    case simsimd_f64_k: return (unary_kernel_t)&_simsimd_f64_to_f64;
    case simsimd_f32_k: return (unary_kernel_t)&_simsimd_f64_to_f32;
    case simsimd_f16_k: return (unary_kernel_t)&_simsimd_f64_to_f16;
    case simsimd_bf16_k: return (unary_kernel_t)&_simsimd_f64_to_bf16;
    default: return NULL;
    }
}

static unary_kernel_t elementwise_downcast_from_i64(simsimd_datatype_t dtype) {
    switch (dtype) {
    case simsimd_u64_k: return (unary_kernel_t)&_simsimd_i64_to_u64;
    case simsimd_u32_k: return (unary_kernel_t)&_simsimd_i64_to_u32;
    case simsimd_u16_k: return (unary_kernel_t)&_simsimd_i64_to_u16;
    case simsimd_u8_k: return (unary_kernel_t)&_simsimd_i64_to_u8;
    case simsimd_i64_k: return (unary_kernel_t)&_simsimd_i64_to_i64;
    case simsimd_i32_k: return (unary_kernel_t)&_simsimd_i64_to_i32;
    case simsimd_i16_k: return (unary_kernel_t)&_simsimd_i64_to_i16;
    case simsimd_i8_k: return (unary_kernel_t)&_simsimd_i64_to_i8;
    case simsimd_f64_k: return (unary_kernel_t)&_simsimd_i64_to_f64;
    case simsimd_f32_k: return (unary_kernel_t)&_simsimd_i64_to_f32;
    case simsimd_f16_k: return (unary_kernel_t)&_simsimd_i64_to_f16;
    case simsimd_bf16_k: return (unary_kernel_t)&_simsimd_i64_to_bf16;
    default: return NULL;
    }
}

static unary_kernel_t elementwise_downcast_from_u64(simsimd_datatype_t dtype) {
    switch (dtype) {
    case simsimd_u64_k: return (unary_kernel_t)&_simsimd_u64_to_u64;
    case simsimd_u32_k: return (unary_kernel_t)&_simsimd_u64_to_u32;
    case simsimd_u16_k: return (unary_kernel_t)&_simsimd_u64_to_u16;
    case simsimd_u8_k: return (unary_kernel_t)&_simsimd_u64_to_u8;
    case simsimd_i64_k: return (unary_kernel_t)&_simsimd_u64_to_i64;
    case simsimd_i32_k: return (unary_kernel_t)&_simsimd_u64_to_i32;
    case simsimd_i16_k: return (unary_kernel_t)&_simsimd_u64_to_i16;
    case simsimd_i8_k: return (unary_kernel_t)&_simsimd_u64_to_i8;
    case simsimd_f64_k: return (unary_kernel_t)&_simsimd_u64_to_f64;
    case simsimd_f32_k: return (unary_kernel_t)&_simsimd_u64_to_f32;
    case simsimd_f16_k: return (unary_kernel_t)&_simsimd_u64_to_f16;
    case simsimd_bf16_k: return (unary_kernel_t)&_simsimd_u64_to_bf16;
    default: return NULL;
    }
}

static char const doc_add[] = //
    "Tensor-Tensor or Tensor-Scalar element-wise addition.\n"
    "\n"
    "Args:\n"
    "    a (Union[NDArray, float, int]): First Tensor or scalar.\n"
    "    b (Union[NDArray, float, int]): Second Tensor or scalar.\n"
    "    out (NDArray, optional): Tensor for resulting distances.\n"
    "    a_dtype (Union[IntegralType, FloatType], optional): Override the presumed numeric type of `a`.\n"
    "    b_dtype (Union[IntegralType, FloatType], optional): Override the presumed numeric type of `b`.\n"
    "    out_dtype (Union[IntegralType, FloatType], optional): Override the presumed numeric type of `out`.\n"
    "Returns:\n"
    "    DistancesTensor: The distances if `out` is not provided.\n"
    "    None: If `out` is provided. Operation will per performed in-place.\n"
    "\n"
    "Equivalent to: `a + b`, but unlike the `sum` API supports scalar arguments.\n"
    "Signature:\n"
    "    >>> def add(a, b, /, *, out, a_dtype, b_dtype, out_dtype) -> Optional[DistancesTensor]: ...\n"
    "\n"
    "Performance recommendations:\n"
    "    - Provide an output tensor to avoid memory allocations.\n"
    "    - Use the same datatype for both inputs and outputs, if supplied.\n"
    "    - Ideally keep operands in continuous memory and maximize the number of last continuous dimensions.\n"
    "    - On tiny inputs you may want to avoid passing arguments by name.\n"
    "In most cases, conforming to these recommendations is easy and will result in the best performance.\n"
    "\n"
    "Broadcasting rules:\n"
    "    - If both inputs are scalars, the output will be a scalar.\n"
    "    - If one input is a scalar, the output will be a tensor of the same shape as the other input.\n"
    "    - If both inputs are tensors, in each dimension, the size must match or one of them must be 1.\n"
    "Broadcasting examples for different shapes:\n"
    "    - (3) + (1) -> (3)\n"
    "    - (3, 1) + (1) -> (3, 1)\n"
    "    - (3, 1) + (1, 1) -> (3, 1)\n"
    "    - (4, 7, 5, 3) + (1) -> (4, 7, 5, 3)\n"
    "    - (4, 7, 5, 3) + (1, 1, 1) -> (4, 7, 5, 3)\n"
    "    - (4, 7, 5, 3) + (4, 1, 5, 1) -> (4, 7, 5, 3).\n"
    "    - (4, 7, 5, 3) + (1, 1, 1, 1, 1) -> (1, 4, 7, 5, 3)\n"
    "    - (4, 7, 5, 3) + (2, 1, 1, 1, 1) -> (2, 4, 7, 5, 3)\n"
    "\n"
    "Typecasting rules:\n"
    "    - If one of tensors contains integrals and the other - floats, `float64` addition will be used.\n"
    "    - If input tensors contain different sign integrals, `int64` saturating addition will be used.\n"
    "    - If input tensors contain different size unsigned integrals, `uint64` saturating addition will be used.";

static PyObject *api_add(PyObject *self, PyObject *const *args, Py_ssize_t const positional_args_count,
                         PyObject *args_names_tuple) {

    PyObject *return_obj = NULL;

    // This function accepts up to 6 arguments:
    PyObject *a_obj = NULL;         // Required object, positional-only
    PyObject *b_obj = NULL;         // Required object, positional-only
    PyObject *out_obj = NULL;       // Optional object, "out" keyword-only
    PyObject *a_dtype_obj = NULL;   // Optional object, "a_dtype" keyword-only
    PyObject *b_dtype_obj = NULL;   // Optional object, "b_dtype" keyword-only
    PyObject *out_dtype_obj = NULL; // Optional object, "out_dtype" keyword-only

    Py_ssize_t const args_names_count = args_names_tuple ? PyTuple_Size(args_names_tuple) : 0;
    Py_ssize_t const args_count = positional_args_count + args_names_count;
    if (args_count < 2 || args_count > 6) {
        PyErr_Format(PyExc_TypeError, "Function expects 2-6 arguments, got %zd", args_count);
        return NULL;
    }
    if (positional_args_count > 2) {
        PyErr_Format(PyExc_TypeError, "Only first 2 arguments can be positional, received %zd", positional_args_count);
        return NULL;
    }

    // Positional-only arguments (first and second matrix)
    a_obj = args[0];
    b_obj = args[1];

    // The rest of the arguments must be checked in the keyword dictionary:
    for (Py_ssize_t args_names_tuple_progress = 0, args_progress = positional_args_count;
         args_names_tuple_progress < args_names_count; ++args_progress, ++args_names_tuple_progress) {
        PyObject *const key = PyTuple_GetItem(args_names_tuple, args_names_tuple_progress);
        PyObject *const value = args[args_progress];
        if (PyUnicode_CompareWithASCIIString(key, "a_dtype") == 0 && !a_dtype_obj) { a_dtype_obj = value; }
        else if (PyUnicode_CompareWithASCIIString(key, "b_dtype") == 0 && !b_dtype_obj) { b_dtype_obj = value; }
        else if (PyUnicode_CompareWithASCIIString(key, "out_dtype") == 0 && !out_dtype_obj) { out_dtype_obj = value; }
        else if (PyUnicode_CompareWithASCIIString(key, "out") == 0 && !out_obj) { out_obj = value; }
        else {
            PyErr_Format(PyExc_TypeError, "Got unexpected keyword argument: %S", key);
            return NULL;
        }
    }

    // Convert `a_dtype_obj` to `a_dtype_str` and to `a_dtype`
    char const *a_dtype_str = NULL, *b_dtype_str = NULL, *out_dtype_str = NULL;
    simsimd_datatype_t a_dtype = simsimd_datatype_unknown_k, b_dtype = simsimd_datatype_unknown_k,
                       out_dtype = simsimd_datatype_unknown_k;
    if (a_dtype_obj) a_dtype_str = PyUnicode_AsUTF8(a_dtype_obj), a_dtype = python_string_to_datatype(a_dtype_str);
    if (b_dtype_obj) b_dtype_str = PyUnicode_AsUTF8(b_dtype_obj), b_dtype = python_string_to_datatype(b_dtype_str);
    if (out_dtype_obj)
        out_dtype_str = PyUnicode_AsUTF8(out_dtype_obj), out_dtype = python_string_to_datatype(out_dtype_str);
    if ((a_dtype_obj || b_dtype_obj || out_dtype_obj) && (!a_dtype_str && !b_dtype_str && !out_dtype_str)) {
        if (PyErr_Occurred()) return NULL;
        if (a_dtype == simsimd_datatype_unknown_k && b_dtype == simsimd_datatype_unknown_k &&
            out_dtype == simsimd_datatype_unknown_k) {
            PyErr_SetString(PyExc_ValueError, "Unsupported 'a_dtype'");
            return NULL;
        }
    }

    // Unlike `sum` and `fma` the first and second argument can be either scalars or tensors.
    // Resolving the the shape of the output tensor and the traversal order is going to be nightmare, behold!
    // To understand the logic, look into the the broadcasting rules in the docstring above.
    BufferOrScalarArgument a_parsed, b_parsed, out_parsed;
    memset(&a_parsed, 0, sizeof(BufferOrScalarArgument));
    memset(&b_parsed, 0, sizeof(BufferOrScalarArgument));
    memset(&out_parsed, 0, sizeof(BufferOrScalarArgument));
    Py_buffer a_buffer, b_buffer, out_buffer;
    memset(&a_buffer, 0, sizeof(Py_buffer));
    memset(&b_buffer, 0, sizeof(Py_buffer));
    memset(&out_buffer, 0, sizeof(Py_buffer));

    // Convert `alpha_obj` to `alpha` and `beta_obj` to `beta`
    if (!parse_buffer_or_scalar_argument(a_obj, &a_buffer, &a_parsed)) goto cleanup;
    if (!parse_buffer_or_scalar_argument(b_obj, &b_buffer, &b_parsed)) goto cleanup;
    if (out_obj && !parse_tensor(out_obj, &out_buffer, &out_parsed.datatype)) goto cleanup;

    // Check dimensions, but unlike the `sum`, `scale`, `wsum`, and `fma` APIs
    // we want to provide maximal compatibility with NumPy and OpenCV. In many
    // such cases, the input is not a rank-1 tensor and may not be continuous.
    Py_ssize_t ins_continuous_dimensions = 0;
    Py_ssize_t ins_continuous_elements = 1;
    if (a_parsed.kind != ScalarKind && b_parsed.kind != ScalarKind) {
        //! The ranks of tensors may not match!
        // We need to compare them in reverse order, right to left, assuming all the missing dimensions are 1.
        // To match those, we are going to populate the `a_parsed.as_buffer_shape` and `b_parsed.as_buffer_shape`,
        // simultaneously filling the strides of broadcasted dimensions with zeros.
        Py_ssize_t const max_rank = a_buffer.ndim > b_buffer.ndim ? a_buffer.ndim : b_buffer.ndim;
        Py_ssize_t const min_rank = a_buffer.ndim < b_buffer.ndim ? a_buffer.ndim : b_buffer.ndim;
        a_parsed.as_buffer_dimensions = max_rank;
        b_parsed.as_buffer_dimensions = max_rank;
        out_parsed.as_buffer_dimensions = max_rank;

        // Go through the shared dimensions: back to front
        for (Py_ssize_t i = 0; i < min_rank; ++i) {
            Py_ssize_t const a_dim = a_buffer.shape[a_buffer.ndim - 1 - i];
            Py_ssize_t const b_dim = b_buffer.shape[b_buffer.ndim - 1 - i];
            Py_ssize_t const a_stride = a_buffer.strides[a_buffer.ndim - 1 - i];
            Py_ssize_t const b_stride = b_buffer.strides[b_buffer.ndim - 1 - i];
            // Simplest case! Both dimensions match.
            if (a_dim == b_dim) {
                a_parsed.as_buffer_shape[max_rank - 1 - i] = a_dim;
                b_parsed.as_buffer_shape[max_rank - 1 - i] = b_dim;
                a_parsed.as_buffer_strides[max_rank - 1 - i] = a_stride;
                b_parsed.as_buffer_strides[max_rank - 1 - i] = b_stride;
                out_parsed.as_buffer_shape[max_rank - 1 - i] = a_dim;
                int a_is_continuous = a_stride == (ins_continuous_elements * bytes_per_datatype(a_parsed.datatype));
                int b_is_continuous = b_stride == (ins_continuous_elements * bytes_per_datatype(b_parsed.datatype));
                if (a_is_continuous && b_is_continuous) {
                    ins_continuous_dimensions++;
                    ins_continuous_elements *= a_dim;
                }
            }
            // Broadcast this value from A
            else if (a_dim == 1) {
                a_parsed.as_buffer_shape[max_rank - 1 - i] = 1;
                b_parsed.as_buffer_shape[max_rank - 1 - i] = b_dim;
                a_parsed.as_buffer_strides[max_rank - 1 - i] = 0;
                b_parsed.as_buffer_strides[max_rank - 1 - i] = b_stride;
                out_parsed.as_buffer_shape[max_rank - 1 - i] = b_dim;
            }
            // Broadcast this value from B
            else if (b_dim == 1) {
                a_parsed.as_buffer_shape[max_rank - 1 - i] = a_dim;
                b_parsed.as_buffer_shape[max_rank - 1 - i] = 1;
                a_parsed.as_buffer_strides[max_rank - 1 - i] = a_stride;
                b_parsed.as_buffer_strides[max_rank - 1 - i] = 0;
                out_parsed.as_buffer_shape[max_rank - 1 - i] = a_dim;
            }
            // Report the issue
            else {
                PyErr_Format( //
                    PyExc_ValueError,
                    "The number of entries %zd (along dimension %zd in the first tensor) doesn't "
                    "match the number of entries %zd (along dimension %zd in the second tensor)",
                    a_dim, a_buffer.ndim - 1 - i, b_dim, b_buffer.ndim - 1 - i);
                return_obj = NULL;
                goto cleanup;
            }
        }
        // Populate the remaining dimensions: in any order, front to back for simplicity
        if (a_buffer.ndim > b_buffer.ndim) {
            for (Py_ssize_t i = 0; i < a_buffer.ndim - b_buffer.ndim; ++i) {
                Py_ssize_t const longer_dim = a_buffer.shape[i];
                Py_ssize_t const longer_stride = a_buffer.strides[i];
                a_parsed.as_buffer_shape[i] = longer_dim;
                b_parsed.as_buffer_shape[i] = 1;
                a_parsed.as_buffer_strides[i] = longer_stride;
                b_parsed.as_buffer_strides[i] = 0;
                out_parsed.as_buffer_shape[i] = longer_dim;
            }
        }
        else if (b_buffer.ndim - a_buffer.ndim) {
            for (Py_ssize_t i = 0; i < b_buffer.ndim - a_buffer.ndim; ++i) {
                Py_ssize_t const longer_dim = b_buffer.shape[i];
                Py_ssize_t const longer_stride = b_buffer.strides[i];
                a_parsed.as_buffer_shape[i] = 1;
                b_parsed.as_buffer_shape[i] = longer_dim;
                a_parsed.as_buffer_strides[i] = 0;
                b_parsed.as_buffer_strides[i] = longer_stride;
                out_parsed.as_buffer_shape[i] = longer_dim;
            }
        }
    }
    // If at least one of the entries is actually is a `ScalarKind` our logic becomes much easier:
    else if (a_parsed.kind != ScalarKind) {
        a_parsed.as_buffer_dimensions = a_buffer.ndim;
        memcpy(a_parsed.as_buffer_shape, a_buffer.shape, a_buffer.ndim * sizeof(Py_ssize_t));
        memcpy(a_parsed.as_buffer_strides, a_buffer.strides, a_buffer.ndim * sizeof(Py_ssize_t));
        b_parsed.as_buffer_dimensions = a_buffer.ndim;
        memcpy(b_parsed.as_buffer_shape, a_buffer.shape, a_buffer.ndim * sizeof(Py_ssize_t));
        memset(b_parsed.as_buffer_strides, 0, a_buffer.ndim * sizeof(Py_ssize_t));
        out_parsed.as_buffer_dimensions = a_buffer.ndim;
        memcpy(out_parsed.as_buffer_shape, a_buffer.shape, a_buffer.ndim * sizeof(Py_ssize_t));
    }
    else {
        a_parsed.as_buffer_dimensions = b_buffer.ndim;
        memcpy(a_parsed.as_buffer_shape, b_buffer.shape, b_buffer.ndim * sizeof(Py_ssize_t));
        memset(a_parsed.as_buffer_strides, 0, b_buffer.ndim * sizeof(Py_ssize_t));
        b_parsed.as_buffer_dimensions = b_buffer.ndim;
        memcpy(b_parsed.as_buffer_shape, b_buffer.shape, b_buffer.ndim * sizeof(Py_ssize_t));
        memcpy(b_parsed.as_buffer_strides, b_buffer.strides, b_buffer.ndim * sizeof(Py_ssize_t));
        out_parsed.as_buffer_dimensions = b_buffer.ndim;
        memcpy(out_parsed.as_buffer_shape, b_buffer.shape, b_buffer.ndim * sizeof(Py_ssize_t));
    }

    // At this point the parsed "logical shapes" of both inputs are identical.
    // Now we can use the inferred logical shape to check the output tensor.
    if (out_obj) {
        if (out_buffer.ndim != out_parsed.as_buffer_dimensions) {
            PyErr_Format(PyExc_ValueError, "Output tensor has rank-%zd, but rank-%zd is expected", out_buffer.ndim,
                         out_parsed.as_buffer_dimensions);
            return_obj = NULL;
            goto cleanup;
        }
        for (Py_ssize_t i = 0; i < out_parsed.as_buffer_dimensions; ++i) {
            if (out_buffer.shape[i] != out_parsed.as_buffer_shape[i]) {
                PyErr_Format(PyExc_ValueError,
                             "Output tensor doesn't match the input tensor in shape at dimension %zd: %zd != %zd", i,
                             out_buffer.shape[i], out_parsed.as_buffer_shape[i]);
                return_obj = NULL;
                goto cleanup;
            }
        }
    }

    // First we need to understand the smallest numeric type we may need for this computation.
    simsimd_datatype_t ab_dtype;
    simsimd_datatype_family_k a_family = simsimd_datatype_family(a_parsed.datatype);
    simsimd_datatype_family_k b_family = simsimd_datatype_family(b_parsed.datatype);
    // For addition and multiplication, treat complex numbers as floats
    if (a_family == simsimd_datatype_complex_float_family_k) a_family = simsimd_datatype_float_family_k;
    if (b_family == simsimd_datatype_complex_float_family_k) b_family = simsimd_datatype_float_family_k;
    if (a_family == simsimd_datatype_binary_family_k || b_family == simsimd_datatype_binary_family_k) {
        PyErr_SetString(PyExc_ValueError, "Boolean tensors are not supported in element-wise operations");
        return_obj = NULL;
        goto cleanup;
    }
    // Infer the type-casting rules for the output tensor, if the type was not provided.
    {
        size_t a_itemsize = bytes_per_datatype(a_parsed.datatype);
        size_t b_itemsize = bytes_per_datatype(b_parsed.datatype);
        size_t max_itemsize = a_itemsize > b_itemsize ? a_itemsize : b_itemsize;
        if (a_parsed.datatype == b_parsed.datatype) { ab_dtype = a_parsed.datatype; }
        // Simply take the bigger datatype if they are both floats or both are unsigned integers, etc.
        else if (a_family == b_family) { ab_dtype = a_itemsize > b_itemsize ? a_parsed.datatype : b_parsed.datatype; }
        // If only one of the operands is a float, and the second is integral of same size, the output should be a
        // float, of the next size... If the floating type is bigger, don't upcast.
        // Sum of `float16` and `int32` is a `float64`.
        // Sum of `float16` and `int16` is a `float32`.
        // Sum of `float32` and `int8` is a `float32`.
        else if (a_family == simsimd_datatype_float_family_k || b_family == simsimd_datatype_float_family_k) {
            size_t float_size = a_family == simsimd_datatype_float_family_k ? a_itemsize : b_itemsize;
            size_t integral_size = a_family == simsimd_datatype_float_family_k ? b_itemsize : a_itemsize;
            if (float_size <= integral_size) {
                //? No 128-bit float on most platforms
                if (max_itemsize == 8) { ab_dtype = simsimd_f64_k; }
                else if (max_itemsize == 4) { ab_dtype = simsimd_f64_k; }
                else if (max_itemsize == 2) { ab_dtype = simsimd_f32_k; }
                else if (max_itemsize == 1) { ab_dtype = simsimd_f16_k; }
            }
            else {
                if (max_itemsize == 8) { ab_dtype = simsimd_f64_k; }
                else if (max_itemsize == 4) { ab_dtype = simsimd_f32_k; }
                else if (max_itemsize == 2) { ab_dtype = simsimd_f16_k; }
                else if (max_itemsize == 1) { ab_dtype = simsimd_f16_k; }
            }
        }
        // If only one of the operands is a unsigned, and the second is a signed integral of same size,
        // the output should be a signed integer, of the next size... If the signed type is bigger, don't upcast.
        // Sum of `int16` and `uint32` is a `int64`.
        // Sum of `int16` and `uint16` is a `int32`.
        // Sum of `int32` and `uint8` is a `int32`.
        else if (a_family == simsimd_datatype_int_family_k || b_family == simsimd_datatype_int_family_k) {
            size_t unsigned_size = a_family == simsimd_datatype_int_family_k ? b_itemsize : a_itemsize;
            size_t signed_size = a_family == simsimd_datatype_int_family_k ? a_itemsize : b_itemsize;
            if (signed_size <= unsigned_size) {
                //? No 128-bit integer on most platforms
                if (max_itemsize == 8) { ab_dtype = simsimd_i64_k; }
                else if (max_itemsize == 4) { ab_dtype = simsimd_i64_k; }
                else if (max_itemsize == 2) { ab_dtype = simsimd_i32_k; }
                else if (max_itemsize == 1) { ab_dtype = simsimd_i16_k; }
            }
            else {
                if (max_itemsize == 8) { ab_dtype = simsimd_i64_k; }
                else if (max_itemsize == 4) { ab_dtype = simsimd_i32_k; }
                else if (max_itemsize == 2) { ab_dtype = simsimd_i16_k; }
                else if (max_itemsize == 1) { ab_dtype = simsimd_i16_k; }
            }
        }
        // For boolean and complex types, we don't yet have a clear policy.
        else {
            PyErr_SetString(PyExc_ValueError, "Unsupported combination of datatypes");
            return_obj = NULL;
            goto cleanup;
        }
    }

    // If dealing with scalars, consider up-casting them to the same `ab_dtype` datatype.
    // That way we will be use the `apply_elementwise_binary_operation_to_each_scalar` function
    // instead of the `apply_elementwise_casting_binary_operation_to_each_scalar` function.
    simsimd_datatype_family_k ab_family = simsimd_datatype_family(ab_dtype);
    if (a_parsed.kind != BufferKind && b_parsed.kind != BufferKind && a_parsed.datatype != b_parsed.datatype) {
        if (ab_family == simsimd_datatype_float_family_k) {
            if (a_parsed.datatype != ab_dtype) {
                elementwise_upcast_to_f64(a_parsed.datatype)(a_parsed.as_scalar, a_parsed.as_scalar);
                elementwise_downcast_from_f64(ab_dtype)(a_parsed.as_scalar, a_parsed.as_scalar);
                a_parsed.datatype = ab_dtype;
            }
            if (b_parsed.datatype != ab_dtype) {
                elementwise_upcast_to_f64(b_parsed.datatype)(b_parsed.as_scalar, b_parsed.as_scalar);
                elementwise_downcast_from_f64(ab_dtype)(b_parsed.as_scalar, b_parsed.as_scalar);
                b_parsed.datatype = ab_dtype;
            }
        }
        else if (ab_family == simsimd_datatype_uint_family_k) {
            if (a_parsed.datatype != ab_dtype) {
                elementwise_upcast_to_u64(a_parsed.datatype)(a_parsed.as_scalar, a_parsed.as_scalar);
                elementwise_downcast_from_u64(ab_dtype)(a_parsed.as_scalar, a_parsed.as_scalar);
                a_parsed.datatype = ab_dtype;
            }
            if (b_parsed.datatype != ab_dtype) {
                elementwise_upcast_to_u64(b_parsed.datatype)(b_parsed.as_scalar, b_parsed.as_scalar);
                elementwise_downcast_from_u64(ab_dtype)(b_parsed.as_scalar, b_parsed.as_scalar);
                b_parsed.datatype = ab_dtype;
            }
        }
        else if (ab_family == simsimd_datatype_int_family_k) {
            if (a_parsed.datatype != ab_dtype) {
                elementwise_upcast_to_i64(a_parsed.datatype)(a_parsed.as_scalar, a_parsed.as_scalar);
                elementwise_downcast_from_i64(ab_dtype)(a_parsed.as_scalar, a_parsed.as_scalar);
                a_parsed.datatype = ab_dtype;
            }
            if (b_parsed.datatype != ab_dtype) {
                elementwise_upcast_to_i64(b_parsed.datatype)(b_parsed.as_scalar, b_parsed.as_scalar);
                elementwise_downcast_from_i64(ab_dtype)(b_parsed.as_scalar, b_parsed.as_scalar);
                b_parsed.datatype = ab_dtype;
            }
        }
    }

    // Estimate the total number of output elements:
    Py_ssize_t out_total_elements = 1;
    for (Py_ssize_t i = 0; i < out_parsed.as_buffer_dimensions; ++i)
        out_total_elements *= out_parsed.as_buffer_shape[i];

    // Allocate the output matrix if it wasn't provided. Unlike other kernels,
    // it's shape will exactly match the shape of the input tensors, but the strides
    // may be different, assuming the output tensor is continuous by default.
    Py_ssize_t out_continuous_dimensions = 0;
    Py_ssize_t out_continuous_elements = 1;
    if (!out_obj) {
        Py_ssize_t const expected_size_bytes = out_total_elements * bytes_per_datatype(ab_dtype);
        NDArray *out_buffer_obj = PyObject_NewVar(NDArray, &NDArrayType, expected_size_bytes);
        if (!out_buffer_obj) {
            PyErr_NoMemory();
            return_obj = NULL;
            goto cleanup;
        }

        // Initialize the object
        memset(out_buffer_obj->shape, 0, sizeof(out_buffer_obj->shape));
        memset(out_buffer_obj->strides, 0, sizeof(out_buffer_obj->strides));
        out_buffer_obj->datatype = ab_dtype;
        out_buffer_obj->ndim = out_parsed.as_buffer_dimensions;
        memcpy(out_buffer_obj->shape, out_parsed.as_buffer_shape, out_parsed.as_buffer_dimensions * sizeof(Py_ssize_t));
        out_buffer_obj->strides[out_parsed.as_buffer_dimensions - 1] = bytes_per_datatype(ab_dtype);
        for (Py_ssize_t i = out_parsed.as_buffer_dimensions - 2; i >= 0; --i)
            out_buffer_obj->strides[i] = out_buffer_obj->strides[i + 1] * out_buffer_obj->shape[i + 1];

        return_obj = (PyObject *)out_buffer_obj;
        out_continuous_dimensions = out_parsed.as_buffer_dimensions;
        out_continuous_elements = out_total_elements;

        // Re-export into the `out_parsed` for future use
        out_parsed.datatype = ab_dtype;
        out_parsed.as_buffer_start = (char *)&out_buffer_obj->start[0];
        memcpy(out_parsed.as_buffer_strides, out_buffer_obj->strides,
               out_parsed.as_buffer_dimensions * sizeof(Py_ssize_t));
    }
    else {
        //? Logic suggests to return `None` in in-place mode...
        //? SciPy decided differently.
        return_obj = Py_None;
        // We need to infer the number of (last) continuous dimensions in the output tensor
        // to be able to apply the element-wise operation.
        out_continuous_dimensions = 0;
        Py_ssize_t expected_last_stride_bytes = out_buffer.itemsize;
        for (Py_ssize_t i = out_buffer.ndim - 1; i >= 0; --i) {
            if (out_buffer.strides[i] != expected_last_stride_bytes) break;
            ++out_continuous_dimensions;
            expected_last_stride_bytes *= out_buffer.shape[i];
            out_continuous_elements *= out_buffer.shape[i];
        }

        out_parsed.as_buffer_dimensions = out_buffer.ndim;
        out_parsed.as_buffer_start = (char *)out_buffer.buf;
        memcpy(out_parsed.as_buffer_shape, out_buffer.shape, out_parsed.as_buffer_dimensions * sizeof(Py_ssize_t));
        memcpy(out_parsed.as_buffer_strides, out_buffer.strides, out_parsed.as_buffer_dimensions * sizeof(Py_ssize_t));
    }

    // First of all, check for our optimal case, when:
    // - both input operands are not scalars.
    // - there is at least one continuous dimension.
    // - the types match between both input tensors and output: uses `sum` kernel.
    // ... where we will use the vectorized kernel!
    if (a_parsed.datatype == b_parsed.datatype && a_parsed.datatype == out_parsed.datatype &&
        out_continuous_dimensions && ins_continuous_dimensions) {

        // Look up the kernel and the capability
        simsimd_elementwise_sum_t kernel = NULL;
        simsimd_capability_t capability = simsimd_cap_serial_k;
        simsimd_kernel_kind_t const kernel_kind = simsimd_sum_k;
        simsimd_find_kernel(kernel_kind, ab_dtype, static_capabilities, simsimd_cap_any_k,
                            (simsimd_kernel_punned_t *)&kernel, &capability);
        if (!kernel) {
            PyErr_Format( //
                PyExc_LookupError, "Unsupported kernel '%c' and datatype combination across inputs ('%s' and '%s')",
                kernel_kind,                                  //
                datatype_to_python_string(a_parsed.datatype), //
                datatype_to_python_string(b_parsed.datatype));
            return_obj = NULL;
            goto cleanup;
        }

        Py_ssize_t const continuous_ranks = out_continuous_dimensions < ins_continuous_dimensions
                                                ? out_continuous_dimensions
                                                : ins_continuous_dimensions;
        Py_ssize_t const non_continuous_ranks = out_parsed.as_buffer_dimensions - continuous_ranks;
        Py_ssize_t const continuous_elements =
            out_continuous_elements < ins_continuous_elements ? out_continuous_elements : ins_continuous_elements;
        apply_elementwise_binary_operation_to_each_continuous_slice( //
            &a_parsed, &b_parsed, &out_parsed,                       //
            non_continuous_ranks, continuous_elements, kernel);

        goto cleanup;
    }

    // Our second best case is when:
    // - there is only one input tensor and the second operand is a scalar,
    // - there is at least one continuous dimension in the input and output tensor,
    // - the types match between the input tensor and output tensor.
    int const is_tensor_a_with_scalar_b =
        a_parsed.kind == BufferKind && a_parsed.datatype == out_parsed.datatype && b_parsed.kind != BufferKind;
    int const is_tensor_b_with_scalar_b =
        b_parsed.kind == BufferKind && b_parsed.datatype == out_parsed.datatype && a_parsed.kind != BufferKind;
    if ((is_tensor_a_with_scalar_b || is_tensor_b_with_scalar_b) &&
        (out_continuous_dimensions && ins_continuous_dimensions)) {
        // Look up the kernel and the capability
        simsimd_elementwise_scale_t kernel = NULL;
        simsimd_capability_t capability = simsimd_cap_serial_k;
        simsimd_kernel_kind_t const kernel_kind = simsimd_scale_k;
        simsimd_find_kernel(kernel_kind, ab_dtype, static_capabilities, simsimd_cap_any_k,
                            (simsimd_kernel_punned_t *)&kernel, &capability);
        if (!kernel) {
            PyErr_Format( //
                PyExc_LookupError, "Unsupported kernel '%c' and datatype combination across inputs ('%s' and '%s')",
                kernel_kind,                                  //
                datatype_to_python_string(a_parsed.datatype), //
                datatype_to_python_string(b_parsed.datatype));
            return_obj = NULL;
            goto cleanup;
        }

        Py_ssize_t const continuous_ranks = out_continuous_dimensions < ins_continuous_dimensions
                                                ? out_continuous_dimensions
                                                : ins_continuous_dimensions;
        Py_ssize_t const non_continuous_ranks = out_parsed.as_buffer_dimensions - continuous_ranks;
        Py_ssize_t const continuous_elements =
            out_continuous_elements < ins_continuous_elements ? out_continuous_elements : ins_continuous_elements;

        if (is_tensor_a_with_scalar_b)
            apply_scale_to_each_continuous_slice(           //
                &a_parsed, 1, b_parsed.as_f64, &out_parsed, //
                non_continuous_ranks, continuous_elements, kernel);
        else
            apply_scale_to_each_continuous_slice(           //
                &b_parsed, 1, a_parsed.as_f64, &out_parsed, //
                non_continuous_ranks, continuous_elements, kernel);

        goto cleanup;
    }

    // Finally call the serial kernels!
    // If the output has no continuous dimensions at all, our situation sucks!
    // We can't use SIMD effectively and need to fall back to the scalar operation,
    // but if the input/output types match, at least we don't need to cast the data back and forth.
    if (a_parsed.datatype == b_parsed.datatype && a_parsed.datatype == out_parsed.datatype) {
        binary_kernel_t elementwise_sadd_ptr = elementwise_sadd(a_parsed.datatype);
        apply_elementwise_binary_operation_to_each_scalar(&a_parsed, &b_parsed, &out_parsed, elementwise_sadd_ptr);
        goto cleanup;
    }

    // If the output has no continuous dimensions at all, our situation sucks!
    // If the type of outputs and inputs doesn't match, it also sucks!
    // We can't use SIMD effectively and need to fall back to the scalar operation.
    if (simsimd_datatype_family(a_parsed.datatype) == simsimd_datatype_float_family_k ||
        simsimd_datatype_family(b_parsed.datatype) == simsimd_datatype_float_family_k) {
        unary_kernel_t a_upcast_ptr = elementwise_upcast_to_f64(a_parsed.datatype);
        unary_kernel_t b_upcast_ptr = elementwise_upcast_to_f64(b_parsed.datatype);
        binary_kernel_t elementwise_sadd_ptr = elementwise_sadd(simsimd_f64_k);
        unary_kernel_t out_downcast_ptr = elementwise_downcast_from_f64(out_parsed.datatype);
        apply_elementwise_casting_binary_operation_to_each_scalar( //
            &a_parsed, &b_parsed, &out_parsed,                     //
            a_upcast_ptr, b_upcast_ptr, out_downcast_ptr, elementwise_sadd_ptr);
        goto cleanup;
    }
    else if (simsimd_datatype_family(a_parsed.datatype) == simsimd_datatype_uint_family_k &&
             simsimd_datatype_family(b_parsed.datatype) == simsimd_datatype_uint_family_k) {
        unary_kernel_t a_upcast_ptr = elementwise_upcast_to_u64(a_parsed.datatype);
        unary_kernel_t b_upcast_ptr = elementwise_upcast_to_u64(b_parsed.datatype);
        binary_kernel_t elementwise_sadd_ptr = elementwise_sadd(simsimd_u64_k);
        unary_kernel_t out_downcast_ptr = elementwise_downcast_from_u64(out_parsed.datatype);
        apply_elementwise_casting_binary_operation_to_each_scalar( //
            &a_parsed, &b_parsed, &out_parsed,                     //
            a_upcast_ptr, b_upcast_ptr, out_downcast_ptr, elementwise_sadd_ptr);
        goto cleanup;
    }
    else {
        unary_kernel_t a_upcast_ptr = elementwise_upcast_to_i64(a_parsed.datatype);
        unary_kernel_t b_upcast_ptr = elementwise_upcast_to_i64(b_parsed.datatype);
        binary_kernel_t elementwise_sadd_ptr = elementwise_sadd(simsimd_i64_k);
        unary_kernel_t out_downcast_ptr = elementwise_downcast_from_i64(out_parsed.datatype);
        apply_elementwise_casting_binary_operation_to_each_scalar( //
            &a_parsed, &b_parsed, &out_parsed,                     //
            a_upcast_ptr, b_upcast_ptr, out_downcast_ptr, elementwise_sadd_ptr);
        goto cleanup;
    }

cleanup:
    PyBuffer_Release(&a_buffer);
    PyBuffer_Release(&b_buffer);
    PyBuffer_Release(&out_buffer);
    if (return_obj == Py_None) Py_INCREF(return_obj);
    return return_obj;
}

static char const doc_multiply[] = //
    "Tensor-Tensor or Tensor-Scalar element-wise multiplication.\n"
    "\n"
    "Args:\n"
    "    a (Union[NDArray, float, int]): First Tensor or scalar.\n"
    "    b (Union[NDArray, float, int]): Second Tensor or scalar.\n"
    "    out (NDArray, optional): Tensor for resulting distances.\n"
    "    a_dtype (Union[IntegralType, FloatType], optional): Override the presumed numeric type of `a`.\n"
    "    b_dtype (Union[IntegralType, FloatType], optional): Override the presumed numeric type of `b`.\n"
    "    out_dtype (Union[IntegralType, FloatType], optional): Override the presumed numeric type of `out`.\n"
    "Returns:\n"
    "    DistancesTensor: The distances if `out` is not provided.\n"
    "    None: If `out` is provided. Operation will per performed in-place.\n"
    "\n"
    "Equivalent to: `a + b`, but unlike the `sum` API supports scalar arguments.\n"
    "Signature:\n"
    "    >>> def multiply(a, b, /, *, out, a_dtype, b_dtype, out_dtype) -> Optional[DistancesTensor]: ...\n"
    "\n"
    "Performance recommendations:\n"
    "    - Provide an output tensor to avoid memory allocations.\n"
    "    - Use the same datatype for both inputs and outputs, if supplied.\n"
    "    - Ideally keep operands in continuous memory and maximize the number of last continuous dimensions.\n"
    "    - On tiny inputs you may want to avoid passing arguments by name.\n"
    "In most cases, conforming to these recommendations is easy and will result in the best performance.\n"
    "\n"
    "Broadcasting rules:\n"
    "    - If both inputs are scalars, the output will be a scalar.\n"
    "    - If one input is a scalar, the output will be a tensor of the same shape as the other input.\n"
    "    - If both inputs are tensors, in each dimension, the size must match or one of them must be 1.\n"
    "Broadcasting examples for different shapes:\n"
    "    - (3) + (1) -> (3)\n"
    "    - (3, 1) + (1) -> (3, 1)\n"
    "    - (3, 1) + (1, 1) -> (3, 1)\n"
    "    - (4, 7, 5, 3) + (1) -> (4, 7, 5, 3)\n"
    "    - (4, 7, 5, 3) + (1, 1, 1) -> (4, 7, 5, 3)\n"
    "    - (4, 7, 5, 3) + (4, 1, 5, 1) -> (4, 7, 5, 3).\n"
    "    - (4, 7, 5, 3) + (1, 1, 1, 1, 1) -> (1, 4, 7, 5, 3)\n"
    "    - (4, 7, 5, 3) + (2, 1, 1, 1, 1) -> (2, 4, 7, 5, 3)\n"
    "\n"
    "Typecasting rules:\n"
    "    - If one of tensors contains integrals and the other - floats, `float64` multiplication will be used.\n"
    "    - If input tensors contain different sign integrals, `int64` saturating multiplication will be used.\n"
    "    - If input tensors contain different size unsigned integrals, `uint64` saturating multiplication will be "
    "used.";

static PyObject *api_multiply(PyObject *self, PyObject *const *args, Py_ssize_t const positional_args_count,
                              PyObject *args_names_tuple) {

    PyObject *return_obj = NULL;

    // This function accepts up to 6 arguments:
    PyObject *a_obj = NULL;         // Required object, positional-only
    PyObject *b_obj = NULL;         // Required object, positional-only
    PyObject *out_obj = NULL;       // Optional object, "out" keyword-only
    PyObject *a_dtype_obj = NULL;   // Optional object, "a_dtype" keyword-only
    PyObject *b_dtype_obj = NULL;   // Optional object, "b_dtype" keyword-only
    PyObject *out_dtype_obj = NULL; // Optional object, "out_dtype" keyword-only

    Py_ssize_t const args_names_count = args_names_tuple ? PyTuple_Size(args_names_tuple) : 0;
    Py_ssize_t const args_count = positional_args_count + args_names_count;
    if (args_count < 2 || args_count > 6) {
        PyErr_Format(PyExc_TypeError, "Function expects 2-6 arguments, got %zd", args_count);
        return NULL;
    }
    if (positional_args_count > 2) {
        PyErr_Format(PyExc_TypeError, "Only first 2 arguments can be positional, received %zd", positional_args_count);
        return NULL;
    }

    // Positional-only arguments (first and second matrix)
    a_obj = args[0];
    b_obj = args[1];

    // The rest of the arguments must be checked in the keyword dictionary:
    for (Py_ssize_t args_names_tuple_progress = 0, args_progress = positional_args_count;
         args_names_tuple_progress < args_names_count; ++args_progress, ++args_names_tuple_progress) {
        PyObject *const key = PyTuple_GetItem(args_names_tuple, args_names_tuple_progress);
        PyObject *const value = args[args_progress];
        if (PyUnicode_CompareWithASCIIString(key, "a_dtype") == 0 && !a_dtype_obj) { a_dtype_obj = value; }
        else if (PyUnicode_CompareWithASCIIString(key, "b_dtype") == 0 && !b_dtype_obj) { b_dtype_obj = value; }
        else if (PyUnicode_CompareWithASCIIString(key, "out_dtype") == 0 && !out_dtype_obj) { out_dtype_obj = value; }
        else if (PyUnicode_CompareWithASCIIString(key, "out") == 0 && !out_obj) { out_obj = value; }
        else {
            PyErr_Format(PyExc_TypeError, "Got unexpected keyword argument: %S", key);
            return NULL;
        }
    }

    // Convert `a_dtype_obj` to `a_dtype_str` and to `a_dtype`
    char const *a_dtype_str = NULL, *b_dtype_str = NULL, *out_dtype_str = NULL;
    simsimd_datatype_t a_dtype = simsimd_datatype_unknown_k, b_dtype = simsimd_datatype_unknown_k,
                       out_dtype = simsimd_datatype_unknown_k;
    if (a_dtype_obj) a_dtype_str = PyUnicode_AsUTF8(a_dtype_obj), a_dtype = python_string_to_datatype(a_dtype_str);
    if (b_dtype_obj) b_dtype_str = PyUnicode_AsUTF8(b_dtype_obj), b_dtype = python_string_to_datatype(b_dtype_str);
    if (out_dtype_obj)
        out_dtype_str = PyUnicode_AsUTF8(out_dtype_obj), out_dtype = python_string_to_datatype(out_dtype_str);
    if ((a_dtype_obj || b_dtype_obj || out_dtype_obj) && (!a_dtype_str && !b_dtype_str && !out_dtype_str)) {
        if (PyErr_Occurred()) return NULL;
        if (a_dtype == simsimd_datatype_unknown_k && b_dtype == simsimd_datatype_unknown_k &&
            out_dtype == simsimd_datatype_unknown_k) {
            PyErr_SetString(PyExc_ValueError, "Unsupported 'a_dtype'");
            return NULL;
        }
    }

    // Unlike `sum` and `fma` the first and second argument can be either scalars or tensors.
    // Resolving the the shape of the output tensor and the traversal order is going to be nightmare, behold!
    // To understand the logic, look into the the broadcasting rules in the docstring above.
    BufferOrScalarArgument a_parsed, b_parsed, out_parsed;
    memset(&a_parsed, 0, sizeof(BufferOrScalarArgument));
    memset(&b_parsed, 0, sizeof(BufferOrScalarArgument));
    memset(&out_parsed, 0, sizeof(BufferOrScalarArgument));
    Py_buffer a_buffer, b_buffer, out_buffer;
    memset(&a_buffer, 0, sizeof(Py_buffer));
    memset(&b_buffer, 0, sizeof(Py_buffer));
    memset(&out_buffer, 0, sizeof(Py_buffer));

    // Convert `alpha_obj` to `alpha` and `beta_obj` to `beta`
    if (!parse_buffer_or_scalar_argument(a_obj, &a_buffer, &a_parsed)) goto cleanup;
    if (!parse_buffer_or_scalar_argument(b_obj, &b_buffer, &b_parsed)) goto cleanup;
    if (out_obj && !parse_tensor(out_obj, &out_buffer, &out_parsed.datatype)) goto cleanup;

    // Check dimensions, but unlike the `sum`, `scale`, `wsum`, and `fma` APIs
    // we want to provide maximal compatibility with NumPy and OpenCV. In many
    // such cases, the input is not a rank-1 tensor and may not be continuous.
    Py_ssize_t ins_continuous_dimensions = 0;
    Py_ssize_t ins_continuous_elements = 1;
    if (a_parsed.kind != ScalarKind && b_parsed.kind != ScalarKind) {
        //! The ranks of tensors may not match!
        // We need to compare them in reverse order, right to left, assuming all the missing dimensions are 1.
        // To match those, we are going to populate the `a_parsed.as_buffer_shape` and `b_parsed.as_buffer_shape`,
        // simultaneously filling the strides of broadcasted dimensions with zeros.
        Py_ssize_t const max_rank = a_buffer.ndim > b_buffer.ndim ? a_buffer.ndim : b_buffer.ndim;
        Py_ssize_t const min_rank = a_buffer.ndim < b_buffer.ndim ? a_buffer.ndim : b_buffer.ndim;
        a_parsed.as_buffer_dimensions = max_rank;
        b_parsed.as_buffer_dimensions = max_rank;
        out_parsed.as_buffer_dimensions = max_rank;

        // Go through the shared dimensions: back to front
        for (Py_ssize_t i = 0; i < min_rank; ++i) {
            Py_ssize_t const a_dim = a_buffer.shape[a_buffer.ndim - 1 - i];
            Py_ssize_t const b_dim = b_buffer.shape[b_buffer.ndim - 1 - i];
            Py_ssize_t const a_stride = a_buffer.strides[a_buffer.ndim - 1 - i];
            Py_ssize_t const b_stride = b_buffer.strides[b_buffer.ndim - 1 - i];
            // Simplest case! Both dimensions match.
            if (a_dim == b_dim) {
                a_parsed.as_buffer_shape[max_rank - 1 - i] = a_dim;
                b_parsed.as_buffer_shape[max_rank - 1 - i] = b_dim;
                a_parsed.as_buffer_strides[max_rank - 1 - i] = a_stride;
                b_parsed.as_buffer_strides[max_rank - 1 - i] = b_stride;
                out_parsed.as_buffer_shape[max_rank - 1 - i] = a_dim;
                int a_is_continuous = a_stride == (ins_continuous_elements * bytes_per_datatype(a_parsed.datatype));
                int b_is_continuous = b_stride == (ins_continuous_elements * bytes_per_datatype(b_parsed.datatype));
                if (a_is_continuous && b_is_continuous) {
                    ins_continuous_dimensions++;
                    ins_continuous_elements *= a_dim;
                }
            }
            // Broadcast this value from A
            else if (a_dim == 1) {
                a_parsed.as_buffer_shape[max_rank - 1 - i] = 1;
                b_parsed.as_buffer_shape[max_rank - 1 - i] = b_dim;
                a_parsed.as_buffer_strides[max_rank - 1 - i] = 0;
                b_parsed.as_buffer_strides[max_rank - 1 - i] = b_stride;
                out_parsed.as_buffer_shape[max_rank - 1 - i] = b_dim;
            }
            // Broadcast this value from B
            else if (b_dim == 1) {
                a_parsed.as_buffer_shape[max_rank - 1 - i] = a_dim;
                b_parsed.as_buffer_shape[max_rank - 1 - i] = 1;
                a_parsed.as_buffer_strides[max_rank - 1 - i] = a_stride;
                b_parsed.as_buffer_strides[max_rank - 1 - i] = 0;
                out_parsed.as_buffer_shape[max_rank - 1 - i] = a_dim;
            }
            // Report the issue
            else {
                PyErr_Format( //
                    PyExc_ValueError,
                    "The number of entries %zd (along dimension %zd in the first tensor) doesn't "
                    "match the number of entries %zd (along dimension %zd in the second tensor)",
                    a_dim, a_buffer.ndim - 1 - i, b_dim, b_buffer.ndim - 1 - i);
                return_obj = NULL;
                goto cleanup;
            }
        }
        // Populate the remaining dimensions: in any order, front to back for simplicity
        if (a_buffer.ndim > b_buffer.ndim) {
            for (Py_ssize_t i = 0; i < a_buffer.ndim - b_buffer.ndim; ++i) {
                Py_ssize_t const longer_dim = a_buffer.shape[i];
                Py_ssize_t const longer_stride = a_buffer.strides[i];
                a_parsed.as_buffer_shape[i] = longer_dim;
                b_parsed.as_buffer_shape[i] = 1;
                a_parsed.as_buffer_strides[i] = longer_stride;
                b_parsed.as_buffer_strides[i] = 0;
                out_parsed.as_buffer_shape[i] = longer_dim;
            }
        }
        else if (b_buffer.ndim - a_buffer.ndim) {
            for (Py_ssize_t i = 0; i < b_buffer.ndim - a_buffer.ndim; ++i) {
                Py_ssize_t const longer_dim = b_buffer.shape[i];
                Py_ssize_t const longer_stride = b_buffer.strides[i];
                a_parsed.as_buffer_shape[i] = 1;
                b_parsed.as_buffer_shape[i] = longer_dim;
                a_parsed.as_buffer_strides[i] = 0;
                b_parsed.as_buffer_strides[i] = longer_stride;
                out_parsed.as_buffer_shape[i] = longer_dim;
            }
        }
    }
    // If at least one of the entries is actually is a `ScalarKind` our logic becomes much easier:
    else if (a_parsed.kind != ScalarKind) {
        a_parsed.as_buffer_dimensions = a_buffer.ndim;
        memcpy(a_parsed.as_buffer_shape, a_buffer.shape, a_buffer.ndim * sizeof(Py_ssize_t));
        memcpy(a_parsed.as_buffer_strides, a_buffer.strides, a_buffer.ndim * sizeof(Py_ssize_t));
        b_parsed.as_buffer_dimensions = a_buffer.ndim;
        memcpy(b_parsed.as_buffer_shape, a_buffer.shape, a_buffer.ndim * sizeof(Py_ssize_t));
        memset(b_parsed.as_buffer_strides, 0, a_buffer.ndim * sizeof(Py_ssize_t));
        out_parsed.as_buffer_dimensions = a_buffer.ndim;
        memcpy(out_parsed.as_buffer_shape, a_buffer.shape, a_buffer.ndim * sizeof(Py_ssize_t));
    }
    else {
        a_parsed.as_buffer_dimensions = b_buffer.ndim;
        memcpy(a_parsed.as_buffer_shape, b_buffer.shape, b_buffer.ndim * sizeof(Py_ssize_t));
        memset(a_parsed.as_buffer_strides, 0, b_buffer.ndim * sizeof(Py_ssize_t));
        b_parsed.as_buffer_dimensions = b_buffer.ndim;
        memcpy(b_parsed.as_buffer_shape, b_buffer.shape, b_buffer.ndim * sizeof(Py_ssize_t));
        memcpy(b_parsed.as_buffer_strides, b_buffer.strides, b_buffer.ndim * sizeof(Py_ssize_t));
        out_parsed.as_buffer_dimensions = b_buffer.ndim;
        memcpy(out_parsed.as_buffer_shape, b_buffer.shape, b_buffer.ndim * sizeof(Py_ssize_t));
    }

    // At this point the parsed "logical shapes" of both inputs are identical.
    // Now we can use the inferred logical shape to check the output tensor.
    if (out_obj) {
        if (out_buffer.ndim != out_parsed.as_buffer_dimensions) {
            PyErr_Format(PyExc_ValueError, "Output tensor has rank-%zd, but rank-%zd is expected", out_buffer.ndim,
                         out_parsed.as_buffer_dimensions);
            return_obj = NULL;
            goto cleanup;
        }
        for (Py_ssize_t i = 0; i < out_parsed.as_buffer_dimensions; ++i) {
            if (out_buffer.shape[i] != out_parsed.as_buffer_shape[i]) {
                PyErr_Format(PyExc_ValueError,
                             "Output tensor doesn't match the input tensor in shape at dimension %zd: %zd != %zd", i,
                             out_buffer.shape[i], out_parsed.as_buffer_shape[i]);
                return_obj = NULL;
                goto cleanup;
            }
        }
    }

    // First we need to understand the smallest numeric type we may need for this computation.
    simsimd_datatype_t ab_dtype;
    simsimd_datatype_family_k a_family = simsimd_datatype_family(a_parsed.datatype);
    simsimd_datatype_family_k b_family = simsimd_datatype_family(b_parsed.datatype);
    // For addition and multiplication, treat complex numbers as floats
    if (a_family == simsimd_datatype_complex_float_family_k) a_family = simsimd_datatype_float_family_k;
    if (b_family == simsimd_datatype_complex_float_family_k) b_family = simsimd_datatype_float_family_k;
    if (a_family == simsimd_datatype_binary_family_k || b_family == simsimd_datatype_binary_family_k) {
        PyErr_SetString(PyExc_ValueError, "Boolean tensors are not supported in element-wise operations");
        return_obj = NULL;
        goto cleanup;
    }
    // Infer the type-casting rules for the output tensor, if the type was not provided.
    {
        size_t a_itemsize = bytes_per_datatype(a_parsed.datatype);
        size_t b_itemsize = bytes_per_datatype(b_parsed.datatype);
        size_t max_itemsize = a_itemsize > b_itemsize ? a_itemsize : b_itemsize;
        if (a_parsed.datatype == b_parsed.datatype) { ab_dtype = a_parsed.datatype; }
        // Simply take the bigger datatype if they are both floats or both are unsigned integers, etc.
        else if (a_family == b_family) { ab_dtype = a_itemsize > b_itemsize ? a_parsed.datatype : b_parsed.datatype; }
        // If only one of the operands is a float, and the second is integral of same size, the output should be a
        // float, of the next size... If the floating type is bigger, don't upcast.
        // Sum of `float16` and `int32` is a `float64`.
        // Sum of `float16` and `int16` is a `float32`.
        // Sum of `float32` and `int8` is a `float32`.
        else if (a_family == simsimd_datatype_float_family_k || b_family == simsimd_datatype_float_family_k) {
            size_t float_size = a_family == simsimd_datatype_float_family_k ? a_itemsize : b_itemsize;
            size_t integral_size = a_family == simsimd_datatype_float_family_k ? b_itemsize : a_itemsize;
            if (float_size <= integral_size) {
                //? No 128-bit float on most platforms
                if (max_itemsize == 8) { ab_dtype = simsimd_f64_k; }
                else if (max_itemsize == 4) { ab_dtype = simsimd_f64_k; }
                else if (max_itemsize == 2) { ab_dtype = simsimd_f32_k; }
                else if (max_itemsize == 1) { ab_dtype = simsimd_f16_k; }
            }
            else {
                if (max_itemsize == 8) { ab_dtype = simsimd_f64_k; }
                else if (max_itemsize == 4) { ab_dtype = simsimd_f32_k; }
                else if (max_itemsize == 2) { ab_dtype = simsimd_f16_k; }
                else if (max_itemsize == 1) { ab_dtype = simsimd_f16_k; }
            }
        }
        // If only one of the operands is a unsigned, and the second is a signed integral of same size,
        // the output should be a signed integer, of the next size... If the signed type is bigger, don't upcast.
        // Sum of `int16` and `uint32` is a `int64`.
        // Sum of `int16` and `uint16` is a `int32`.
        // Sum of `int32` and `uint8` is a `int32`.
        else if (a_family == simsimd_datatype_int_family_k || b_family == simsimd_datatype_int_family_k) {
            size_t unsigned_size = a_family == simsimd_datatype_int_family_k ? b_itemsize : a_itemsize;
            size_t signed_size = a_family == simsimd_datatype_int_family_k ? a_itemsize : b_itemsize;
            if (signed_size <= unsigned_size) {
                //? No 128-bit integer on most platforms
                if (max_itemsize == 8) { ab_dtype = simsimd_i64_k; }
                else if (max_itemsize == 4) { ab_dtype = simsimd_i64_k; }
                else if (max_itemsize == 2) { ab_dtype = simsimd_i32_k; }
                else if (max_itemsize == 1) { ab_dtype = simsimd_i16_k; }
            }
            else {
                if (max_itemsize == 8) { ab_dtype = simsimd_i64_k; }
                else if (max_itemsize == 4) { ab_dtype = simsimd_i32_k; }
                else if (max_itemsize == 2) { ab_dtype = simsimd_i16_k; }
                else if (max_itemsize == 1) { ab_dtype = simsimd_i16_k; }
            }
        }
        // For boolean and complex types, we don't yet have a clear policy.
        else {
            PyErr_SetString(PyExc_ValueError, "Unsupported combination of datatypes");
            return_obj = NULL;
            goto cleanup;
        }
    }

    // If dealing with scalars, consider up-casting them to the same `ab_dtype` datatype.
    // That way we will be use the `apply_elementwise_binary_operation_to_each_scalar` function
    // instead of the `apply_elementwise_casting_binary_operation_to_each_scalar` function.
    simsimd_datatype_family_k ab_family = simsimd_datatype_family(ab_dtype);
    if (a_parsed.kind != BufferKind && b_parsed.kind != BufferKind && a_parsed.datatype != b_parsed.datatype) {
        if (ab_family == simsimd_datatype_float_family_k) {
            if (a_parsed.datatype != ab_dtype) {
                elementwise_upcast_to_f64(a_parsed.datatype)(a_parsed.as_scalar, a_parsed.as_scalar);
                elementwise_downcast_from_f64(ab_dtype)(a_parsed.as_scalar, a_parsed.as_scalar);
                a_parsed.datatype = ab_dtype;
            }
            if (b_parsed.datatype != ab_dtype) {
                elementwise_upcast_to_f64(b_parsed.datatype)(b_parsed.as_scalar, b_parsed.as_scalar);
                elementwise_downcast_from_f64(ab_dtype)(b_parsed.as_scalar, b_parsed.as_scalar);
                b_parsed.datatype = ab_dtype;
            }
        }
        else if (ab_family == simsimd_datatype_uint_family_k) {
            if (a_parsed.datatype != ab_dtype) {
                elementwise_upcast_to_u64(a_parsed.datatype)(a_parsed.as_scalar, a_parsed.as_scalar);
                elementwise_downcast_from_u64(ab_dtype)(a_parsed.as_scalar, a_parsed.as_scalar);
                a_parsed.datatype = ab_dtype;
            }
            if (b_parsed.datatype != ab_dtype) {
                elementwise_upcast_to_u64(b_parsed.datatype)(b_parsed.as_scalar, b_parsed.as_scalar);
                elementwise_downcast_from_u64(ab_dtype)(b_parsed.as_scalar, b_parsed.as_scalar);
                b_parsed.datatype = ab_dtype;
            }
        }
        else if (ab_family == simsimd_datatype_int_family_k) {
            if (a_parsed.datatype != ab_dtype) {
                elementwise_upcast_to_i64(a_parsed.datatype)(a_parsed.as_scalar, a_parsed.as_scalar);
                elementwise_downcast_from_i64(ab_dtype)(a_parsed.as_scalar, a_parsed.as_scalar);
                a_parsed.datatype = ab_dtype;
            }
            if (b_parsed.datatype != ab_dtype) {
                elementwise_upcast_to_i64(b_parsed.datatype)(b_parsed.as_scalar, b_parsed.as_scalar);
                elementwise_downcast_from_i64(ab_dtype)(b_parsed.as_scalar, b_parsed.as_scalar);
                b_parsed.datatype = ab_dtype;
            }
        }
    }

    // Estimate the total number of output elements:
    Py_ssize_t out_total_elements = 1;
    for (Py_ssize_t i = 0; i < out_parsed.as_buffer_dimensions; ++i)
        out_total_elements *= out_parsed.as_buffer_shape[i];

    // Allocate the output matrix if it wasn't provided. Unlike other kernels,
    // it's shape will exactly match the shape of the input tensors, but the strides
    // may be different, assuming the output tensor is continuous by default.
    Py_ssize_t out_continuous_dimensions = 0;
    Py_ssize_t out_continuous_elements = 1;
    if (!out_obj) {
        Py_ssize_t const expected_size_bytes = out_total_elements * bytes_per_datatype(ab_dtype);
        NDArray *out_buffer_obj = PyObject_NewVar(NDArray, &NDArrayType, expected_size_bytes);
        if (!out_buffer_obj) {
            PyErr_NoMemory();
            return_obj = NULL;
            goto cleanup;
        }

        // Initialize the object
        memset(out_buffer_obj->shape, 0, sizeof(out_buffer_obj->shape));
        memset(out_buffer_obj->strides, 0, sizeof(out_buffer_obj->strides));
        out_buffer_obj->datatype = ab_dtype;
        out_buffer_obj->ndim = out_parsed.as_buffer_dimensions;
        memcpy(out_buffer_obj->shape, out_parsed.as_buffer_shape, out_parsed.as_buffer_dimensions * sizeof(Py_ssize_t));
        out_buffer_obj->strides[out_parsed.as_buffer_dimensions - 1] = bytes_per_datatype(ab_dtype);
        for (Py_ssize_t i = out_parsed.as_buffer_dimensions - 2; i >= 0; --i)
            out_buffer_obj->strides[i] = out_buffer_obj->strides[i + 1] * out_buffer_obj->shape[i + 1];

        return_obj = (PyObject *)out_buffer_obj;
        out_continuous_dimensions = out_parsed.as_buffer_dimensions;
        out_continuous_elements = out_total_elements;

        // Re-export into the `out_parsed` for future use
        out_parsed.datatype = ab_dtype;
        out_parsed.as_buffer_start = (char *)&out_buffer_obj->start[0];
        memcpy(out_parsed.as_buffer_strides, out_buffer_obj->strides,
               out_parsed.as_buffer_dimensions * sizeof(Py_ssize_t));
    }
    else {
        //? Logic suggests to return `None` in in-place mode...
        //? SciPy decided differently.
        return_obj = Py_None;
        // We need to infer the number of (last) continuous dimensions in the output tensor
        // to be able to apply the element-wise operation.
        out_continuous_dimensions = 0;
        Py_ssize_t expected_last_stride_bytes = out_buffer.itemsize;
        for (Py_ssize_t i = out_buffer.ndim - 1; i >= 0; --i) {
            if (out_buffer.strides[i] != expected_last_stride_bytes) break;
            ++out_continuous_dimensions;
            expected_last_stride_bytes *= out_buffer.shape[i];
            out_continuous_elements *= out_buffer.shape[i];
        }

        out_parsed.as_buffer_dimensions = out_buffer.ndim;
        out_parsed.as_buffer_start = (char *)out_buffer.buf;
        memcpy(out_parsed.as_buffer_shape, out_buffer.shape, out_parsed.as_buffer_dimensions * sizeof(Py_ssize_t));
        memcpy(out_parsed.as_buffer_strides, out_buffer.strides, out_parsed.as_buffer_dimensions * sizeof(Py_ssize_t));
    }

    // First of all, check for our optimal case, when:
    // - both input operands are not scalars.
    // - there is at least one continuous dimension.
    // - the types match between both input tensors and output: uses `fma` kernel.
    // ... where we will use the vectorized kernel!
    if (a_parsed.datatype == b_parsed.datatype && a_parsed.datatype == out_parsed.datatype &&
        out_continuous_dimensions && ins_continuous_dimensions) {

        // Look up the kernel and the capability
        simsimd_elementwise_fma_t kernel = NULL;
        simsimd_capability_t capability = simsimd_cap_serial_k;
        simsimd_kernel_kind_t const kernel_kind = simsimd_fma_k;
        simsimd_find_kernel(kernel_kind, ab_dtype, static_capabilities, simsimd_cap_any_k,
                            (simsimd_kernel_punned_t *)&kernel, &capability);
        if (!kernel) {
            PyErr_Format( //
                PyExc_LookupError, "Unsupported kernel '%c' and datatype combination across inputs ('%s' and '%s')",
                kernel_kind,                                  //
                datatype_to_python_string(a_parsed.datatype), //
                datatype_to_python_string(b_parsed.datatype));
            return_obj = NULL;
            goto cleanup;
        }

        Py_ssize_t const continuous_ranks = out_continuous_dimensions < ins_continuous_dimensions
                                                ? out_continuous_dimensions
                                                : ins_continuous_dimensions;
        Py_ssize_t const non_continuous_ranks = out_parsed.as_buffer_dimensions - continuous_ranks;
        Py_ssize_t const continuous_elements =
            out_continuous_elements < ins_continuous_elements ? out_continuous_elements : ins_continuous_elements;
        apply_elementwise_binary_operation_to_each_continuous_slice( //
            &a_parsed, &b_parsed, &out_parsed,                       //
            non_continuous_ranks, continuous_elements, kernel);

        goto cleanup;
    }

    // Our second best case is when:
    // - there is only one input tensor and the second operand is a scalar,
    // - there is at least one continuous dimension in the input and output tensor,
    // - the types match between the input tensor and output tensor.
    int const is_tensor_a_with_scalar_b =
        a_parsed.kind == BufferKind && a_parsed.datatype == out_parsed.datatype && b_parsed.kind != BufferKind;
    int const is_tensor_b_with_scalar_b =
        b_parsed.kind == BufferKind && b_parsed.datatype == out_parsed.datatype && a_parsed.kind != BufferKind;
    if ((is_tensor_a_with_scalar_b || is_tensor_b_with_scalar_b) &&
        (out_continuous_dimensions && ins_continuous_dimensions)) {
        // Look up the kernel and the capability
        simsimd_elementwise_scale_t kernel = NULL;
        simsimd_capability_t capability = simsimd_cap_serial_k;
        simsimd_kernel_kind_t const kernel_kind = simsimd_scale_k;
        simsimd_find_kernel(kernel_kind, ab_dtype, static_capabilities, simsimd_cap_any_k,
                            (simsimd_kernel_punned_t *)&kernel, &capability);
        if (!kernel) {
            PyErr_Format( //
                PyExc_LookupError, "Unsupported kernel '%c' and datatype combination across inputs ('%s' and '%s')",
                kernel_kind,                                  //
                datatype_to_python_string(a_parsed.datatype), //
                datatype_to_python_string(b_parsed.datatype));
            return_obj = NULL;
            goto cleanup;
        }

        Py_ssize_t const continuous_ranks = out_continuous_dimensions < ins_continuous_dimensions
                                                ? out_continuous_dimensions
                                                : ins_continuous_dimensions;
        Py_ssize_t const non_continuous_ranks = out_parsed.as_buffer_dimensions - continuous_ranks;
        Py_ssize_t const continuous_elements =
            out_continuous_elements < ins_continuous_elements ? out_continuous_elements : ins_continuous_elements;

        if (is_tensor_a_with_scalar_b)
            apply_scale_to_each_continuous_slice(           //
                &a_parsed, b_parsed.as_f64, 0, &out_parsed, //
                non_continuous_ranks, continuous_elements, kernel);
        else
            apply_scale_to_each_continuous_slice(           //
                &b_parsed, a_parsed.as_f64, 0, &out_parsed, //
                non_continuous_ranks, continuous_elements, kernel);

        goto cleanup;
    }

    // Finally call the serial kernels!
    // If the output has no continuous dimensions at all, our situation sucks!
    // We can't use SIMD effectively and need to fall back to the scalar operation,
    // but if the input/output types match, at least we don't need to cast the data back and forth.
    if (a_parsed.datatype == b_parsed.datatype && a_parsed.datatype == out_parsed.datatype) {
        binary_kernel_t elementwise_smul_ptr = elementwise_smul(a_parsed.datatype);
        apply_elementwise_binary_operation_to_each_scalar(&a_parsed, &b_parsed, &out_parsed, elementwise_smul_ptr);
        goto cleanup;
    }

    // If the output has no continuous dimensions at all, our situation sucks!
    // If the type of outputs and inputs doesn't match, it also sucks!
    // We can't use SIMD effectively and need to fall back to the scalar operation.
    if (simsimd_datatype_family(a_parsed.datatype) == simsimd_datatype_float_family_k ||
        simsimd_datatype_family(b_parsed.datatype) == simsimd_datatype_float_family_k) {
        unary_kernel_t a_upcast_ptr = elementwise_upcast_to_f64(a_parsed.datatype);
        unary_kernel_t b_upcast_ptr = elementwise_upcast_to_f64(b_parsed.datatype);
        binary_kernel_t elementwise_smul_ptr = elementwise_smul(simsimd_f64_k);
        unary_kernel_t out_downcast_ptr = elementwise_downcast_from_f64(out_parsed.datatype);
        apply_elementwise_casting_binary_operation_to_each_scalar( //
            &a_parsed, &b_parsed, &out_parsed,                     //
            a_upcast_ptr, b_upcast_ptr, out_downcast_ptr, elementwise_smul_ptr);
        goto cleanup;
    }
    else if (simsimd_datatype_family(a_parsed.datatype) == simsimd_datatype_uint_family_k &&
             simsimd_datatype_family(b_parsed.datatype) == simsimd_datatype_uint_family_k) {
        unary_kernel_t a_upcast_ptr = elementwise_upcast_to_u64(a_parsed.datatype);
        unary_kernel_t b_upcast_ptr = elementwise_upcast_to_u64(b_parsed.datatype);
        binary_kernel_t elementwise_smul_ptr = elementwise_smul(simsimd_u64_k);
        unary_kernel_t out_downcast_ptr = elementwise_downcast_from_u64(out_parsed.datatype);
        apply_elementwise_casting_binary_operation_to_each_scalar( //
            &a_parsed, &b_parsed, &out_parsed,                     //
            a_upcast_ptr, b_upcast_ptr, out_downcast_ptr, elementwise_smul_ptr);
        goto cleanup;
    }
    else {
        unary_kernel_t a_upcast_ptr = elementwise_upcast_to_i64(a_parsed.datatype);
        unary_kernel_t b_upcast_ptr = elementwise_upcast_to_i64(b_parsed.datatype);
        binary_kernel_t elementwise_smul_ptr = elementwise_smul(simsimd_i64_k);
        unary_kernel_t out_downcast_ptr = elementwise_downcast_from_i64(out_parsed.datatype);
        apply_elementwise_casting_binary_operation_to_each_scalar( //
            &a_parsed, &b_parsed, &out_parsed,                     //
            a_upcast_ptr, b_upcast_ptr, out_downcast_ptr, elementwise_smul_ptr);
        goto cleanup;
    }

cleanup:
    PyBuffer_Release(&a_buffer);
    PyBuffer_Release(&b_buffer);
    PyBuffer_Release(&out_buffer);
    if (return_obj == Py_None) Py_INCREF(return_obj);
    return return_obj;
}

// There are several flags we can use to define the functions:
// - `METH_O`: Single object argument
// - `METH_VARARGS`: Variable number of arguments
// - `METH_FASTCALL`: Fast calling convention
// - `METH_KEYWORDS`: Accepts keyword arguments, can be combined with `METH_FASTCALL`
//
// https://llllllllll.github.io/c-extension-tutorial/appendix.html#c.PyMethodDef.ml_flags
static PyMethodDef simsimd_methods[] = {
    // Introspecting library and hardware capabilities
    {"get_capabilities", (PyCFunction)api_get_capabilities, METH_NOARGS, doc_get_capabilities},
    {"enable_capability", (PyCFunction)api_enable_capability, METH_O, doc_enable_capability},
    {"disable_capability", (PyCFunction)api_disable_capability, METH_O, doc_disable_capability},

    // NumPy and SciPy compatible interfaces for dense vector representations
    // Each function can compute distances between:
    //  - A pair of vectors
    //  - A batch of vector pairs (two matrices of identical shape)
    //  - A matrix of vectors and a single vector
    {"l2", (PyCFunction)api_l2, METH_FASTCALL | METH_KEYWORDS, doc_l2},
    {"l2sq", (PyCFunction)api_l2sq, METH_FASTCALL | METH_KEYWORDS, doc_l2sq},
    {"kl", (PyCFunction)api_kl, METH_FASTCALL | METH_KEYWORDS, doc_kl},
    {"js", (PyCFunction)api_js, METH_FASTCALL | METH_KEYWORDS, doc_js},
    {"cos", (PyCFunction)api_cos, METH_FASTCALL | METH_KEYWORDS, doc_cos},
    {"dot", (PyCFunction)api_dot, METH_FASTCALL | METH_KEYWORDS, doc_dot},
    {"vdot", (PyCFunction)api_vdot, METH_FASTCALL | METH_KEYWORDS, doc_vdot},
    {"hamming", (PyCFunction)api_hamming, METH_FASTCALL | METH_KEYWORDS, doc_hamming},
    {"jaccard", (PyCFunction)api_jaccard, METH_FASTCALL | METH_KEYWORDS, doc_jaccard},

    // Aliases
    {"euclidean", (PyCFunction)api_l2, METH_FASTCALL | METH_KEYWORDS, doc_l2},
    {"sqeuclidean", (PyCFunction)api_l2sq, METH_FASTCALL | METH_KEYWORDS, doc_l2sq},
    {"cosine", (PyCFunction)api_cos, METH_FASTCALL | METH_KEYWORDS, doc_cos},
    {"inner", (PyCFunction)api_dot, METH_FASTCALL | METH_KEYWORDS, doc_dot},
    {"kullbackleibler", (PyCFunction)api_kl, METH_FASTCALL | METH_KEYWORDS, doc_kl},
    {"jensenshannon", (PyCFunction)api_js, METH_FASTCALL | METH_KEYWORDS, doc_js},

    // Conventional `cdist` interface for pairwise distances
    {"cdist", (PyCFunction)api_cdist, METH_FASTCALL | METH_KEYWORDS, doc_cdist},

    // Exposing underlying API for USearch `CompiledMetric`
    {"pointer_to_euclidean", (PyCFunction)api_l2_pointer, METH_O, doc_l2_pointer},
    {"pointer_to_sqeuclidean", (PyCFunction)api_l2sq_pointer, METH_O, doc_l2sq_pointer},
    {"pointer_to_cosine", (PyCFunction)api_cos_pointer, METH_O, doc_cos_pointer},
    {"pointer_to_inner", (PyCFunction)api_dot_pointer, METH_O, doc_dot_pointer},
    {"pointer_to_dot", (PyCFunction)api_dot_pointer, METH_O, doc_dot_pointer},
    {"pointer_to_vdot", (PyCFunction)api_vdot_pointer, METH_O, doc_vdot_pointer},
    {"pointer_to_kullbackleibler", (PyCFunction)api_kl_pointer, METH_O, doc_kl_pointer},
    {"pointer_to_jensenshannon", (PyCFunction)api_js_pointer, METH_O, doc_js_pointer},

    // Set operations
    {"intersect", (PyCFunction)api_intersect, METH_FASTCALL, doc_intersect},

    // Curved spaces
    {"bilinear", (PyCFunction)api_bilinear, METH_FASTCALL | METH_KEYWORDS, doc_bilinear},
    {"mahalanobis", (PyCFunction)api_mahalanobis, METH_FASTCALL | METH_KEYWORDS, doc_mahalanobis},

    // Vectorized operations
    {"scale", (PyCFunction)api_scale, METH_FASTCALL | METH_KEYWORDS, doc_scale},
    {"sum", (PyCFunction)api_sum, METH_FASTCALL | METH_KEYWORDS, doc_sum},
    {"wsum", (PyCFunction)api_wsum, METH_FASTCALL | METH_KEYWORDS, doc_wsum},
    {"fma", (PyCFunction)api_fma, METH_FASTCALL | METH_KEYWORDS, doc_fma},

    // NumPy and OpenCV compatible APIs for element-wise binary operations,
    // that support both vector and scalar arguments
    {"add", (PyCFunction)api_add, METH_FASTCALL | METH_KEYWORDS, doc_add},
    {"multiply", (PyCFunction)api_multiply, METH_FASTCALL | METH_KEYWORDS, doc_multiply},

    // Sentinel
    {NULL, NULL, 0, NULL}};

static char const doc_module[] = //
    "Portable mixed-precision BLAS-like vector math library for x86 and ARM.\n"
    "\n"
    "Performance Recommendations:\n"
    " - Avoid converting to NumPy arrays. SimSIMD works with any Tensor implementation\n"
    "   compatible with Python's Buffer Protocol, which can be coming from PyTorch, TensorFlow, etc.\n"
    " - In low-latency environments - provide the output array with the `out=` parameter\n"
    "   to avoid expensive memory allocations on the hot path.\n"
    " - On modern CPUs, if the application allows, prefer low-precision numeric types.\n"
    "   Whenever possible, use 'bf16' and 'f16' over 'f32'. Consider quantizing to 'i8'\n"
    "   and 'u8' for highest hardware compatibility and performance.\n"
    " - If you are only interested in relative proximity instead of the absolute distance\n"
    "   prefer simpler kernels, like the Squared Euclidean distance over the Euclidean distance.\n"
    " - Use row-major continuous matrix representations. Strides between rows won't have significant\n"
    "   impact on performance, but most modern HPC packages explicitly ban non-contiguous rows,\n"
    "   where the nearby matrix cells within a row have multi-byte gaps.\n"
    " - The CPython runtime has a noticeable overhead for function calls, so consider batching\n"
    "   kernel invocations. Many kernels can compute not only 1-to-1 distance between vectors,\n"
    "   but also 1-to-N and N-to-N distances between two batches of vectors packed into matrices.\n"
    "\n"
    "Example:\n"
    "    >>> import simsimd\n"
    "    >>> simsimd.l2(a, b)\n"
    "\n"
    "Mixed-precision 1-to-N example with numeric types missing in NumPy, but present in PyTorch:\n"
    "    >>> import simsimd\n"
    "    >>> import torch\n"
    "    >>> a = torch.randn(1536, dtype=torch.bfloat16)\n"
    "    >>> b = torch.randn((100, 1536), dtype=torch.bfloat16)\n"
    "    >>> c = torch.zeros(100, dtype=torch.float32)\n"
    "    >>> simsimd.l2(a, b, dtype='bfloat16', out=c)\n";

static PyModuleDef simsimd_module = {
    PyModuleDef_HEAD_INIT, .m_name = "SimSIMD", .m_doc = doc_module, .m_size = -1, .m_methods = simsimd_methods,
};

PyMODINIT_FUNC PyInit_simsimd(void) {
    PyObject *m;

    if (PyType_Ready(&DistancesTensorType) < 0) return NULL;
    if (PyType_Ready(&NDArrayType) < 0) return NULL;

    m = PyModule_Create(&simsimd_module);
    if (m == NULL) return NULL;

    // Add version metadata
    {
        char version_str[64];
        snprintf(version_str, sizeof(version_str), "%d.%d.%d", SIMSIMD_VERSION_MAJOR, SIMSIMD_VERSION_MINOR,
                 SIMSIMD_VERSION_PATCH);
        PyModule_AddStringConstant(m, "__version__", version_str);
    }

    Py_INCREF(&DistancesTensorType);
    if (PyModule_AddObject(m, "DistancesTensor", (PyObject *)&DistancesTensorType) < 0) {
        Py_XDECREF(&DistancesTensorType);
        Py_XDECREF(m);
        return NULL;
    }

    Py_INCREF(&NDArrayType);
    if (PyModule_AddObject(m, "NDArray", (PyObject *)&NDArrayType) < 0) {
        Py_XDECREF(&NDArrayType);
        Py_XDECREF(&DistancesTensorType);
        Py_XDECREF(m);
        return NULL;
    }

    static_capabilities = simsimd_capabilities();
    return m;
}

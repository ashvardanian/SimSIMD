/**
 *  @brief      Pure CPython bindings for NumKong.
 *  @file       lib.c
 *  @author     Ash Vardanian
 *  @date       January 1, 2023
 *
 *  @section    Latency, Quality, and Arguments Parsing
 *
 *  The complexity of implementing high-quality CPython bindings is often underestimated.
 *  Do not rely on high-level wrappers like PyBind11 and NanoBind, and avoid SWIG-like
 *  toolchains. Most of them use expensive dynamic data structures to map callbacks to
 *  object or module properties, rather than relying on the CPython API. They are too
 *  slow for low-latency operations such as checking container lengths, handling vectors,
 *  or processing strings.
 *
 *  Once you work directly with the CPython API, there is substantial boilerplate code
 *  to write, and it is common to use `PyArg_ParseTupleAndKeywords` or `PyArg_ParseTuple`.
 *  Those functions parse format specifier strings at runtime, which @b cannot be fast
 *  by design. Moreover, they do not support the Python "Fast Calling Convention". In
 *  a typical scenario, a function is defined with `METH_VARARGS | METH_KEYWORDS` and
 *  has a signature like:
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
 *  This `cdist` example takes 2 positional, 1 positional or named, and 3 named-only arguments.
 *  An alternative with `METH_FASTCALL` uses a function signature like:
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
 *  There are cases where a call is ill-formed and provides more positional arguments than expected.
 *
 *  @code {.py}
 *      cdist(a, b, "cos", "dos"):               //! positional_args_count == 4, args_names_count == 0
 *      cdist(a, b, "cos", metric="dos"):        //! positional_args_count == 3, args_names_count == 1
 *      cdist(a, b, metric="cos", metric="dos"): //! positional_args_count == 2, args_names_count == 2
 *  @endcode
 *
 *  If the same argument is provided twice, a @b `TypeError` is raised.
 *  If the argument is not found, a @b `KeyError` is raised.
 *
 *  https://ashvardanian.com/posts/discount-on-keyword-arguments-in-python/
 *
 *  @section    Buffer Protocol and NumPy Compatibility
 *
 *  Most modern machine learning frameworks struggle with buffer protocol compatibility.
 *  At best, they provide zero-copy NumPy views of the underlying data, which introduces an
 *  unnecessary dependency on NumPy, an allocation for the wrapper, and constraints on the
 *  supported numeric types. This is a limitation because PyTorch and TensorFlow have richer
 *  type systems than NumPy.
 *
 *  A PyTorch `Tensor` cannot be converted to a `memoryview` object.
 *  Converting a `bf16` TensorFlow `Tensor` to a `memoryview` raises:
 *
 *      ! ValueError: cannot include dtype 'E' in a buffer
 *
 *  Moreover, the CPython and NumPy documentation diverge on format specifiers for the `typestr`
 *  and `format` data type descriptor strings, which makes development error-prone. NumKong is
 *  @b one of the few packages that attempts to provide interoperability.
 *
 *  https://numpy.org/doc/stable/reference/arrays.interface.html
 *  https://pearu.github.io/array_interface_pytorch.html
 *  https://github.com/pytorch/pytorch/issues/54138
 *  https://github.com/pybind/pybind11/issues/1908
 */
#include <math.h>

#if defined(__linux__)
#if defined(_OPENMP)
#include <omp.h>
#endif
#endif

#include <numkong/numkong.h>

#define PY_SSIZE_T_CLEAN
#include <Python.h>

/**
 *  @brief  Get the kernel's native output dtype for a given metric and input dtype.
 */
static nk_datatype_t metric_kernel_output_dtype(nk_kernel_kind_t kind, nk_datatype_t input) {
    switch (kind) {
    case nk_kernel_dot_k:
    case nk_kernel_vdot_k: return nk_dot_output_datatype(input);
    case nk_kernel_l2_k: return nk_l2_output_datatype(input);
    case nk_kernel_l2sq_k: return nk_l2sq_output_datatype(input);
    case nk_kernel_angular_k: return nk_angular_output_datatype(input);
    default: return nk_f64_k;
    }
}

typedef struct TensorArgument {
    char *start;
    size_t dimensions;
    size_t count;
    size_t stride;
    int rank;
    nk_datatype_t datatype;
} TensorArgument;

typedef struct NDArray {
    PyObject_HEAD                            // //
    nk_datatype_t datatype;                  // Logical datatype (f32, f64, f16, bf16, i8, etc.)
    size_t rank;                             // Number of dimensions (0 for scalar, up to NK_NDARRAY_MAX_RANK)
    Py_ssize_t shape[NK_NDARRAY_MAX_RANK];   // Extent along each dimension
    Py_ssize_t strides[NK_NDARRAY_MAX_RANK]; // Stride in bytes for each dimension
    PyObject *parent;                        // Reference to parent tensor (NULL if owns data)
    char *data;                              // Pointer to data (start[] if owns, or parent's memory if view)
    char start[];                            // Variable length data (only used when parent == NULL)
} NDArray;

// Forward declaration of the type object.
static PyTypeObject NDArrayType;

static int NDArray_getbuffer(PyObject *export_from, Py_buffer *view, int flags);
static void NDArray_releasebuffer(PyObject *export_from, Py_buffer *view);
static PyObject *tensor_read_scalar(NDArray *tensor, size_t byte_offset);

// Forward declaration of the view factory used by slicing.
static NDArray *NDArray_view(NDArray *parent, char *data_ptr, nk_datatype_t datatype, size_t rank,
                             Py_ssize_t const *shape, Py_ssize_t const *strides);

static PyBufferProcs NDArray_as_buffer = {
    .bf_getbuffer = NDArray_getbuffer,
    .bf_releasebuffer = NDArray_releasebuffer,
};

/// @brief  Convert a 0D NDArray to a Python float.
static PyObject *NDArray_float(PyObject *self) {
    NDArray *tensor = (NDArray *)self;
    if (tensor->rank != 0) {
        PyErr_SetString(PyExc_TypeError, "only 0-dimensional tensors can be converted to float");
        return NULL;
    }
    PyObject *scalar = tensor_read_scalar(tensor, 0);
    if (!scalar) return NULL;
    PyObject *result = PyNumber_Float(scalar);
    Py_DECREF(scalar);
    return result;
}

/// @brief  Convert a 0D NDArray to a Python int.
static PyObject *NDArray_int(PyObject *self) {
    NDArray *tensor = (NDArray *)self;
    if (tensor->rank != 0) {
        PyErr_SetString(PyExc_TypeError, "only 0-dimensional tensors can be converted to int");
        return NULL;
    }
    PyObject *scalar = tensor_read_scalar(tensor, 0);
    if (!scalar) return NULL;
    PyObject *result = PyNumber_Long(scalar);
    Py_DECREF(scalar);
    return result;
}

// Forward declarations for arithmetic operators
static NDArray *NDArray_new(nk_datatype_t datatype, size_t rank, Py_ssize_t const *shape);
static PyObject *NDArray_copy(PyObject *self, PyObject *args);

// region Arithmetic Operators

static PyObject *NDArray_positive(PyObject *self) { return NDArray_copy(self, NULL); }

static PyObject *NDArray_negative(PyObject *self) {
    NDArray *t = (NDArray *)self;
    NDArray *r = NDArray_new(t->datatype, t->rank, t->shape);
    if (!r) return NULL;
    size_t n = 1;
    for (size_t i = 0; i < t->rank; i++) n *= (size_t)t->shape[i];
    switch (t->datatype) {
    case nk_f64_k: {
        nk_f64_t *s = (nk_f64_t *)t->data, *d = (nk_f64_t *)r->data;
        for (size_t i = 0; i < n; i++) d[i] = -s[i];
    } break;
    case nk_f32_k: {
        nk_f32_t *s = (nk_f32_t *)t->data, *d = (nk_f32_t *)r->data;
        for (size_t i = 0; i < n; i++) d[i] = -s[i];
    } break;
    case nk_i8_k: {
        nk_i8_t *s = (nk_i8_t *)t->data, *d = (nk_i8_t *)r->data;
        for (size_t i = 0; i < n; i++) d[i] = -s[i];
    } break;
    case nk_i32_k: {
        nk_i32_t *s = (nk_i32_t *)t->data, *d = (nk_i32_t *)r->data;
        for (size_t i = 0; i < n; i++) d[i] = -s[i];
    } break;
    case nk_i64_k: {
        nk_i64_t *s = (nk_i64_t *)t->data, *d = (nk_i64_t *)r->data;
        for (size_t i = 0; i < n; i++) d[i] = -s[i];
    } break;
    default:
        Py_DECREF(r);
        PyErr_SetString(PyExc_NotImplementedError, "neg not implemented");
        return NULL;
    }
    return (PyObject *)r;
}

static PyObject *NDArray_add(PyObject *self, PyObject *other) {
    if (!PyObject_TypeCheck(self, &NDArrayType)) { Py_RETURN_NOTIMPLEMENTED; }
    NDArray *a = (NDArray *)self;
    if (PyObject_TypeCheck(other, &NDArrayType)) {
        NDArray *b = (NDArray *)other;
        if (a->rank != b->rank || a->datatype != b->datatype) {
            PyErr_SetString(PyExc_ValueError, "shape/dtype mismatch");
            return NULL;
        }
        for (size_t i = 0; i < a->rank; i++)
            if (a->shape[i] != b->shape[i]) {
                PyErr_SetString(PyExc_ValueError, "shape mismatch");
                return NULL;
            }
        NDArray *r = NDArray_new(a->datatype, a->rank, a->shape);
        if (!r) return NULL;
        size_t n = 1;
        for (size_t i = 0; i < a->rank; i++) n *= (size_t)a->shape[i];
        switch (a->datatype) {
        case nk_f64_k: nk_sum_f64((nk_f64_t *)a->data, (nk_f64_t *)b->data, n, (nk_f64_t *)r->data); break;
        case nk_f32_k: nk_sum_f32((nk_f32_t *)a->data, (nk_f32_t *)b->data, n, (nk_f32_t *)r->data); break;
        case nk_i8_k: nk_sum_i8((nk_i8_t *)a->data, (nk_i8_t *)b->data, n, (nk_i8_t *)r->data); break;
        case nk_i32_k: nk_sum_i32((nk_i32_t *)a->data, (nk_i32_t *)b->data, n, (nk_i32_t *)r->data); break;
        default:
            Py_DECREF(r);
            PyErr_SetString(PyExc_NotImplementedError, "add not impl");
            return NULL;
        }
        return (PyObject *)r;
    }
    if (PyFloat_Check(other) || PyLong_Check(other)) {
        double sc = PyFloat_Check(other) ? PyFloat_AsDouble(other) : (double)PyLong_AsLong(other);
        NDArray *r = NDArray_new(a->datatype, a->rank, a->shape);
        if (!r) return NULL;
        size_t n = 1;
        for (size_t i = 0; i < a->rank; i++) n *= (size_t)a->shape[i];
        switch (a->datatype) {
        case nk_f64_k: {
            nk_f64_t al = 1, be = sc;
            nk_scale_f64((nk_f64_t *)a->data, n, &al, &be, (nk_f64_t *)r->data);
        } break;
        case nk_f32_k: {
            nk_f32_t al = 1, be = (nk_f32_t)sc;
            nk_scale_f32((nk_f32_t *)a->data, n, &al, &be, (nk_f32_t *)r->data);
        } break;
        default:
            Py_DECREF(r);
            PyErr_SetString(PyExc_NotImplementedError, "add+sc not impl");
            return NULL;
        }
        return (PyObject *)r;
    }
    Py_RETURN_NOTIMPLEMENTED;
}

static PyObject *NDArray_subtract(PyObject *self, PyObject *other) {
    if (!PyObject_TypeCheck(self, &NDArrayType)) { Py_RETURN_NOTIMPLEMENTED; }
    NDArray *a = (NDArray *)self;
    if (PyObject_TypeCheck(other, &NDArrayType)) {
        NDArray *b = (NDArray *)other;
        if (a->rank != b->rank || a->datatype != b->datatype) {
            PyErr_SetString(PyExc_ValueError, "shape/dtype mismatch");
            return NULL;
        }
        for (size_t i = 0; i < a->rank; i++)
            if (a->shape[i] != b->shape[i]) {
                PyErr_SetString(PyExc_ValueError, "shape mismatch");
                return NULL;
            }
        NDArray *r = NDArray_new(a->datatype, a->rank, a->shape);
        if (!r) return NULL;
        size_t n = 1;
        for (size_t i = 0; i < a->rank; i++) n *= (size_t)a->shape[i];
        switch (a->datatype) {
        case nk_f64_k: {
            nk_f64_t al = 1, be = -1;
            nk_wsum_f64((nk_f64_t *)a->data, (nk_f64_t *)b->data, n, &al, &be, (nk_f64_t *)r->data);
        } break;
        case nk_f32_k: {
            nk_f32_t al = 1, be = -1;
            nk_wsum_f32((nk_f32_t *)a->data, (nk_f32_t *)b->data, n, &al, &be, (nk_f32_t *)r->data);
        } break;
        default:
            Py_DECREF(r);
            PyErr_SetString(PyExc_NotImplementedError, "sub not impl");
            return NULL;
        }
        return (PyObject *)r;
    }
    if (PyFloat_Check(other) || PyLong_Check(other)) {
        double sc = PyFloat_Check(other) ? PyFloat_AsDouble(other) : (double)PyLong_AsLong(other);
        NDArray *r = NDArray_new(a->datatype, a->rank, a->shape);
        if (!r) return NULL;
        size_t n = 1;
        for (size_t i = 0; i < a->rank; i++) n *= (size_t)a->shape[i];
        switch (a->datatype) {
        case nk_f64_k: {
            nk_f64_t al = 1, be = -sc;
            nk_scale_f64((nk_f64_t *)a->data, n, &al, &be, (nk_f64_t *)r->data);
        } break;
        case nk_f32_k: {
            nk_f32_t al = 1, be = (nk_f32_t)(-sc);
            nk_scale_f32((nk_f32_t *)a->data, n, &al, &be, (nk_f32_t *)r->data);
        } break;
        default:
            Py_DECREF(r);
            PyErr_SetString(PyExc_NotImplementedError, "sub-sc not impl");
            return NULL;
        }
        return (PyObject *)r;
    }
    Py_RETURN_NOTIMPLEMENTED;
}

static PyObject *NDArray_multiply(PyObject *self, PyObject *other) {
    if (!PyObject_TypeCheck(self, &NDArrayType)) { Py_RETURN_NOTIMPLEMENTED; }
    NDArray *a = (NDArray *)self;
    if (PyObject_TypeCheck(other, &NDArrayType)) {
        NDArray *b = (NDArray *)other;
        if (a->rank != b->rank || a->datatype != b->datatype) {
            PyErr_SetString(PyExc_ValueError, "shape/dtype mismatch");
            return NULL;
        }
        for (size_t i = 0; i < a->rank; i++)
            if (a->shape[i] != b->shape[i]) {
                PyErr_SetString(PyExc_ValueError, "shape mismatch");
                return NULL;
            }
        NDArray *r = NDArray_new(a->datatype, a->rank, a->shape);
        if (!r) return NULL;
        size_t n = 1;
        for (size_t i = 0; i < a->rank; i++) n *= (size_t)a->shape[i];
        switch (a->datatype) {
        case nk_f64_k: {
            nk_f64_t al = 1, be = 0;
            nk_fma_f64((nk_f64_t *)a->data, (nk_f64_t *)b->data, (nk_f64_t *)r->data, n, &al, &be, (nk_f64_t *)r->data);
        } break;
        case nk_f32_k: {
            nk_f32_t al = 1, be = 0;
            nk_fma_f32((nk_f32_t *)a->data, (nk_f32_t *)b->data, (nk_f32_t *)r->data, n, &al, &be, (nk_f32_t *)r->data);
        } break;
        default:
            Py_DECREF(r);
            PyErr_SetString(PyExc_NotImplementedError, "mul not impl");
            return NULL;
        }
        return (PyObject *)r;
    }
    if (PyFloat_Check(other) || PyLong_Check(other)) {
        double sc = PyFloat_Check(other) ? PyFloat_AsDouble(other) : (double)PyLong_AsLong(other);
        NDArray *r = NDArray_new(a->datatype, a->rank, a->shape);
        if (!r) return NULL;
        size_t n = 1;
        for (size_t i = 0; i < a->rank; i++) n *= (size_t)a->shape[i];
        switch (a->datatype) {
        case nk_f64_k: {
            nk_f64_t al = sc, be = 0;
            nk_scale_f64((nk_f64_t *)a->data, n, &al, &be, (nk_f64_t *)r->data);
        } break;
        case nk_f32_k: {
            nk_f32_t al = (nk_f32_t)sc, be = 0;
            nk_scale_f32((nk_f32_t *)a->data, n, &al, &be, (nk_f32_t *)r->data);
        } break;
        default:
            Py_DECREF(r);
            PyErr_SetString(PyExc_NotImplementedError, "mul*sc not impl");
            return NULL;
        }
        return (PyObject *)r;
    }
    Py_RETURN_NOTIMPLEMENTED;
}

// endregion

static PyNumberMethods NDArray_as_number = {
    .nb_add = NDArray_add,
    .nb_subtract = NDArray_subtract,
    .nb_multiply = NDArray_multiply,
    .nb_negative = NDArray_negative,
    .nb_positive = NDArray_positive,
    .nb_float = NDArray_float,
    .nb_int = NDArray_int,
};

typedef struct {
    nk_datatype_t dtype;
    char const *name;
    char const *buffer_format;
    char const *array_typestr;
    size_t item_size;
    int is_complex;
} nk_dtype_info_t;

static nk_dtype_info_t const nk_dtype_table[] = {
    {nk_f64_k, "float64", "d", "<f8", sizeof(nk_f64_t), 0},
    {nk_f32_k, "float32", "f", "<f4", sizeof(nk_f32_t), 0},
    {nk_f16_k, "float16", "e", "<f2", sizeof(nk_f16_t), 0},
    {nk_bf16_k, "bfloat16", "bf16", "<V2", sizeof(nk_bf16_t), 0},
    {nk_e4m3_k, "e4m3", "e4m3", "|V1", sizeof(nk_e4m3_t), 0},
    {nk_e5m2_k, "e5m2", "e5m2", "|V1", sizeof(nk_e5m2_t), 0},
    {nk_f64c_k, "complex128", "Zd", "<c16", sizeof(nk_f64_t) * 2, 1},
    {nk_f32c_k, "complex64", "Zf", "<c8", sizeof(nk_f32_t) * 2, 1},
    {nk_f16c_k, "complex32", "Ze", "|V4", sizeof(nk_f16_t) * 2, 1},
    {nk_bf16c_k, "bfloat16c", "bcomplex32", "|V4", sizeof(nk_bf16_t) * 2, 1},
    {nk_b8_k, "bin8", "?", "|V1", sizeof(nk_b8_t), 0},
    {nk_i8_k, "int8", "b", "|i1", sizeof(nk_i8_t), 0},
    {nk_u8_k, "uint8", "B", "|u1", sizeof(nk_u8_t), 0},
    {nk_i16_k, "int16", "h", "<i2", sizeof(nk_i16_t), 0},
    {nk_u16_k, "uint16", "H", "<u2", sizeof(nk_u16_t), 0},
    {nk_i32_k, "int32", "i", "<i4", sizeof(nk_i32_t), 0},
    {nk_u32_k, "uint32", "I", "<u4", sizeof(nk_u32_t), 0},
    {nk_i64_k, "int64", "q", "<i8", sizeof(nk_i64_t), 0},
    {nk_u64_k, "uint64", "Q", "<u8", sizeof(nk_u64_t), 0},
};

static nk_dtype_info_t const *datatype_info(nk_datatype_t dtype) {
    for (size_t i = 0; i < sizeof(nk_dtype_table) / sizeof(nk_dtype_table[0]); i++) {
        if (nk_dtype_table[i].dtype == dtype) return &nk_dtype_table[i];
    }
    return NULL;
}

/// @brief Estimate the number of bytes per element for a given datatype.
/// @param dtype Logical datatype, can be complex.
/// @return Zero if the datatype is not supported, positive integer otherwise.
static size_t bytes_per_datatype(nk_datatype_t dtype) {
    nk_dtype_info_t const *info = datatype_info(dtype);
    return info ? info->item_size : 0;
}

/// @brief  Convert a nk_datatype_t to a Python string (for example, "float64", "float32").
static char const *datatype_to_string(nk_datatype_t dtype) {
    nk_dtype_info_t const *info = datatype_info(dtype);
    return info ? info->name : "unknown";
}

static char const *datatype_to_array_typestr(nk_datatype_t dtype) {
    nk_dtype_info_t const *info = datatype_info(dtype);
    return info ? info->array_typestr : "|V1";
}

/// @brief  Get the shape property as a Python tuple.
static PyObject *NDArray_get_shape(PyObject *self, void *closure) {
    (void)closure;
    NDArray *tensor = (NDArray *)self;
    PyObject *shape_tuple = PyTuple_New(tensor->rank);
    if (!shape_tuple) return NULL;
    for (size_t i = 0; i < tensor->rank; i++) {
        PyTuple_SET_ITEM(shape_tuple, i, PyLong_FromSsize_t(tensor->shape[i]));
    }
    return shape_tuple;
}

/// @brief  Get the dtype property as a Python string.
static PyObject *NDArray_get_dtype(PyObject *self, void *closure) {
    (void)closure;
    NDArray *tensor = (NDArray *)self;
    return PyUnicode_FromString(datatype_to_string(tensor->datatype));
}

/// @brief  Get the ndim property (number of dimensions).
static PyObject *NDArray_get_ndim(PyObject *self, void *closure) {
    (void)closure;
    NDArray *tensor = (NDArray *)self;
    return PyLong_FromSize_t(tensor->rank);
}

/// @brief  Get the size property (total number of elements).
static PyObject *NDArray_get_size(PyObject *self, void *closure) {
    (void)closure;
    NDArray *tensor = (NDArray *)self;
    Py_ssize_t total = 1;
    for (size_t i = 0; i < tensor->rank; i++) { total *= tensor->shape[i]; }
    return PyLong_FromSsize_t(total);
}

/// @brief  Get the nbytes property (total bytes of data).
static PyObject *NDArray_get_nbytes(PyObject *self, void *closure) {
    (void)closure;
    NDArray *tensor = (NDArray *)self;
    Py_ssize_t total = 1;
    for (size_t i = 0; i < tensor->rank; i++) { total *= tensor->shape[i]; }
    return PyLong_FromSsize_t(total * (Py_ssize_t)bytes_per_datatype(tensor->datatype));
}

/// @brief  Get the strides property as a Python tuple (in bytes).
static PyObject *NDArray_get_strides(PyObject *self, void *closure) {
    (void)closure;
    NDArray *tensor = (NDArray *)self;
    PyObject *strides_tuple = PyTuple_New(tensor->rank);
    if (!strides_tuple) return NULL;
    for (size_t i = 0; i < tensor->rank; i++) {
        PyTuple_SET_ITEM(strides_tuple, i, PyLong_FromSsize_t(tensor->strides[i]));
    }
    return strides_tuple;
}

/// @brief  Get the itemsize property (bytes per element).
static PyObject *NDArray_get_itemsize(PyObject *self, void *closure) {
    (void)closure;
    NDArray *tensor = (NDArray *)self;
    return PyLong_FromSize_t(bytes_per_datatype(tensor->datatype));
}

/// @brief  Get the __array_interface__ property for NumPy interoperability.
///         Returns a dict with keys: shape, typestr, data, strides, version.
static PyObject *NDArray_get_array_interface(PyObject *self, void *closure) {
    (void)closure;
    NDArray *tensor = (NDArray *)self;

    PyObject *dict = PyDict_New();
    if (!dict) return NULL;

    // shape tuple
    PyObject *shape = NDArray_get_shape(self, NULL);
    if (!shape) {
        Py_DECREF(dict);
        return NULL;
    }
    PyDict_SetItemString(dict, "shape", shape);
    Py_DECREF(shape);

    // typestr - NumPy array interface string (raw for unsupported types)
    char const *typestr = datatype_to_array_typestr(tensor->datatype);
    PyObject *typestr_obj = PyUnicode_FromString(typestr);
    if (!typestr_obj) {
        Py_DECREF(dict);
        return NULL;
    }
    PyDict_SetItemString(dict, "typestr", typestr_obj);
    Py_DECREF(typestr_obj);

    // data tuple: (pointer, readonly_flag)
    PyObject *data_ptr = PyLong_FromVoidPtr(tensor->data);
    if (!data_ptr) {
        Py_DECREF(dict);
        return NULL;
    }
    PyObject *data_tuple = PyTuple_Pack(2, data_ptr, Py_False);
    Py_DECREF(data_ptr);
    if (!data_tuple) {
        Py_DECREF(dict);
        return NULL;
    }
    PyDict_SetItemString(dict, "data", data_tuple);
    Py_DECREF(data_tuple);

    // strides tuple (in bytes)
    PyObject *strides = NDArray_get_strides(self, NULL);
    if (!strides) {
        Py_DECREF(dict);
        return NULL;
    }
    PyDict_SetItemString(dict, "strides", strides);
    Py_DECREF(strides);

    // version
    PyObject *version = PyLong_FromLong(3);
    if (!version) {
        Py_DECREF(dict);
        return NULL;
    }
    PyDict_SetItemString(dict, "version", version);
    Py_DECREF(version);

    return dict;
}

/// @brief  Get the T property as a transpose view that shares data.
///         For 2D tensors, swaps axes. For other ranks, reverses all axes.
static PyObject *NDArray_get_T(PyObject *self, void *closure) {
    (void)closure;
    NDArray *tensor = (NDArray *)self;

    // For 0D or 1D tensors, return self (transpose is no-op)
    if (tensor->rank <= 1) {
        Py_INCREF(self);
        return self;
    }

    // Create transposed shape and strides (reversed)
    Py_ssize_t new_shape[NK_NDARRAY_MAX_RANK];
    Py_ssize_t new_strides[NK_NDARRAY_MAX_RANK];

    for (size_t i = 0; i < tensor->rank; i++) {
        new_shape[i] = tensor->shape[tensor->rank - 1 - i];
        new_strides[i] = tensor->strides[tensor->rank - 1 - i];
    }

    // Get the root parent for the view reference (handles chained views)
    NDArray *root_parent = tensor->parent ? (NDArray *)tensor->parent : tensor;

    // Create view (zero-copy) with same data pointer but swapped shape/strides
    return (PyObject *)NDArray_view(root_parent, tensor->data, tensor->datatype, tensor->rank, new_shape, new_strides);
}

static PyGetSetDef NDArray_getset[] = {
    {"shape", NDArray_get_shape, NULL, "Shape of the tensor as a tuple", NULL},
    {"dtype", NDArray_get_dtype, NULL, "Data type of tensor elements", NULL},
    {"ndim", NDArray_get_ndim, NULL, "Number of dimensions", NULL},
    {"size", NDArray_get_size, NULL, "Total number of elements", NULL},
    {"nbytes", NDArray_get_nbytes, NULL, "Total bytes of data", NULL},
    {"strides", NDArray_get_strides, NULL, "Strides in bytes for each dimension", NULL},
    {"itemsize", NDArray_get_itemsize, NULL, "Bytes per element", NULL},
    {"__array_interface__", NDArray_get_array_interface, NULL, "NumPy array interface dict", NULL},
    {"T", NDArray_get_T, NULL, "Transpose of the tensor", NULL},
    {NULL, NULL, NULL, NULL, NULL} // Sentinel
};

/// @brief  Get the length (first dimension) for the sequence protocol.
static Py_ssize_t NDArray_length(PyObject *self) {
    NDArray *tensor = (NDArray *)self;
    if (tensor->rank == 0) {
        PyErr_SetString(PyExc_TypeError, "0-dimensional tensor has no len()");
        return -1;
    }
    return tensor->shape[0];
}

static PySequenceMethods NDArray_as_sequence = {
    .sq_length = NDArray_length,
};

// Forward declarations for mapping protocol (implemented after type definition to avoid circular dependency)
static PyObject *NDArray_subscript(PyObject *self, PyObject *key);

static PyMappingMethods NDArray_as_mapping = {
    .mp_length = NDArray_length,
    .mp_subscript = NDArray_subscript,
};

// Forward declaration for iterator (implemented after type definition)
static PyTypeObject NDArrayIterType;
static PyObject *NDArray_iter(PyObject *self);

/// @brief  Return the string representation of a NDArray.
static PyObject *NDArray_repr(PyObject *self) {
    NDArray *tensor = (NDArray *)self;
    // Build shape string like "(3,)" or "(3, 3)" or "()"
    char shape_str[256] = "(";
    size_t pos = 1;
    for (size_t i = 0; i < tensor->rank; i++) {
        int written = snprintf(shape_str + pos, sizeof(shape_str) - pos, "%zd", tensor->shape[i]);
        if (written < 0 || (size_t)written >= sizeof(shape_str) - pos) break;
        pos += written;
        if (i < tensor->rank - 1) {
            if (pos < sizeof(shape_str) - 2) {
                shape_str[pos++] = ',';
                shape_str[pos++] = ' ';
            }
        }
        else if (tensor->rank == 1) {
            // Add trailing comma for 1-element tuple
            if (pos < sizeof(shape_str) - 1) shape_str[pos++] = ',';
        }
    }
    if (pos < sizeof(shape_str) - 1) shape_str[pos++] = ')';
    shape_str[pos] = '\0';

    return PyUnicode_FromFormat("NDArray(shape=%s, dtype='%s')", shape_str, datatype_to_string(tensor->datatype));
}

/// @brief  Kinds of scalar values represented by a NDArray element.
typedef enum {
    tensor_scalar_kind_float,
    tensor_scalar_kind_int,
    tensor_scalar_kind_uint,
    tensor_scalar_kind_complex,
} tensor_scalar_kind_t;

/// @brief  Tagged scalar value container used for scalar conversions and formatting.
typedef struct {
    tensor_scalar_kind_t kind;
    double real;
    double imag;
    long long i64;
    unsigned long long u64;
} tensor_scalar_value_t;

/// @brief  Read a scalar value at a byte offset into a tagged container.
static int tensor_read_scalar_value(NDArray *tensor, size_t byte_offset, tensor_scalar_value_t *value) {
    char *ptr = tensor->data + byte_offset;
    nk_f32_t f32_tmp;
    nk_f32_t f32_tmp_imag;

    switch (tensor->datatype) {
    case nk_f64_k:
        value->kind = tensor_scalar_kind_float;
        value->real = (double)*(nk_f64_t *)ptr;
        return 1;
    case nk_f32_k:
        value->kind = tensor_scalar_kind_float;
        value->real = (double)*(nk_f32_t *)ptr;
        return 1;
    case nk_f16_k:
        nk_f16_to_f32((nk_f16_t *)ptr, &f32_tmp);
        value->kind = tensor_scalar_kind_float;
        value->real = (double)f32_tmp;
        return 1;
    case nk_bf16_k:
        nk_bf16_to_f32((nk_bf16_t *)ptr, &f32_tmp);
        value->kind = tensor_scalar_kind_float;
        value->real = (double)f32_tmp;
        return 1;
    case nk_e4m3_k:
        nk_e4m3_to_f32((nk_e4m3_t *)ptr, &f32_tmp);
        value->kind = tensor_scalar_kind_float;
        value->real = (double)f32_tmp;
        return 1;
    case nk_e5m2_k:
        nk_e5m2_to_f32((nk_e5m2_t *)ptr, &f32_tmp);
        value->kind = tensor_scalar_kind_float;
        value->real = (double)f32_tmp;
        return 1;
    case nk_f64c_k: {
        nk_f64_t const *vals = (nk_f64_t const *)ptr;
        value->kind = tensor_scalar_kind_complex;
        value->real = (double)vals[0];
        value->imag = (double)vals[1];
        return 1;
    }
    case nk_f32c_k: {
        nk_f32_t const *vals = (nk_f32_t const *)ptr;
        value->kind = tensor_scalar_kind_complex;
        value->real = (double)vals[0];
        value->imag = (double)vals[1];
        return 1;
    }
    case nk_f16c_k: {
        nk_f16_t const *vals = (nk_f16_t const *)ptr;
        nk_f16_to_f32(&vals[0], &f32_tmp);
        nk_f16_to_f32(&vals[1], &f32_tmp_imag);
        value->kind = tensor_scalar_kind_complex;
        value->real = (double)f32_tmp;
        value->imag = (double)f32_tmp_imag;
        return 1;
    }
    case nk_bf16c_k: {
        nk_bf16_t const *vals = (nk_bf16_t const *)ptr;
        nk_bf16_to_f32(&vals[0], &f32_tmp);
        nk_bf16_to_f32(&vals[1], &f32_tmp_imag);
        value->kind = tensor_scalar_kind_complex;
        value->real = (double)f32_tmp;
        value->imag = (double)f32_tmp_imag;
        return 1;
    }
    case nk_b8_k:
        value->kind = tensor_scalar_kind_uint;
        value->u64 = (unsigned long long)*(nk_b8_t *)ptr;
        return 1;
    case nk_i8_k:
        value->kind = tensor_scalar_kind_int;
        value->i64 = (long long)*(nk_i8_t *)ptr;
        return 1;
    case nk_u8_k:
        value->kind = tensor_scalar_kind_uint;
        value->u64 = (unsigned long long)*(nk_u8_t *)ptr;
        return 1;
    case nk_i16_k:
        value->kind = tensor_scalar_kind_int;
        value->i64 = (long long)*(nk_i16_t *)ptr;
        return 1;
    case nk_u16_k:
        value->kind = tensor_scalar_kind_uint;
        value->u64 = (unsigned long long)*(nk_u16_t *)ptr;
        return 1;
    case nk_i32_k:
        value->kind = tensor_scalar_kind_int;
        value->i64 = (long long)*(nk_i32_t *)ptr;
        return 1;
    case nk_u32_k:
        value->kind = tensor_scalar_kind_uint;
        value->u64 = (unsigned long long)*(nk_u32_t *)ptr;
        return 1;
    case nk_i64_k:
        value->kind = tensor_scalar_kind_int;
        value->i64 = (long long)*(nk_i64_t *)ptr;
        return 1;
    case nk_u64_k:
        value->kind = tensor_scalar_kind_uint;
        value->u64 = (unsigned long long)*(nk_u64_t *)ptr;
        return 1;
    default: PyErr_SetString(PyExc_TypeError, "unsupported datatype for scalar conversion"); return 0;
    }
}

/// @brief  Format a scalar value into a string buffer.
static int tensor_format_scalar(NDArray *tensor, size_t byte_offset, char *buf, size_t bufsize) {
    tensor_scalar_value_t value;
    if (!tensor_read_scalar_value(tensor, byte_offset, &value)) {
        PyErr_Clear();
        return snprintf(buf, bufsize, "?");
    }

    switch (value.kind) {
    case tensor_scalar_kind_float: return snprintf(buf, bufsize, "%.6g", value.real);
    case tensor_scalar_kind_complex: return snprintf(buf, bufsize, "%.6g%+.6gj", value.real, value.imag);
    case tensor_scalar_kind_int: return snprintf(buf, bufsize, "%lld", value.i64);
    case tensor_scalar_kind_uint: return snprintf(buf, bufsize, "%llu", value.u64);
    default: return snprintf(buf, bufsize, "?");
    }
}

/// @brief  Format a pretty-printed representation of a NDArray (similar to NumPy's __str__).
static PyObject *NDArray_str(PyObject *self) {
    NDArray *tensor = (NDArray *)self;
    size_t item_size = bytes_per_datatype(tensor->datatype);

    // For 0D scalar, just print the value
    if (tensor->rank == 0) {
        char buf[64];
        tensor_format_scalar(tensor, 0, buf, sizeof(buf));
        return PyUnicode_FromString(buf);
    }

    // Calculate total elements
    Py_ssize_t total = 1;
    for (size_t i = 0; i < tensor->rank; i++) total *= tensor->shape[i];

    // Limit output for large tensors
    int truncate = total > 100;
    Py_ssize_t max_elements = truncate ? 20 : total;

    // Build output string
    // Estimate buffer size: ~20 chars per element + brackets
    size_t bufsize = (size_t)(max_elements * 24 + tensor->rank * 4 + 64);
    char *buf = (char *)PyMem_Malloc(bufsize);
    if (!buf) return PyErr_NoMemory();

    size_t pos = 0;

    // For 1D tensor
    if (tensor->rank == 1) {
        buf[pos++] = '[';
        Py_ssize_t n = tensor->shape[0];
        Py_ssize_t show = (n > 10 && truncate) ? 6 : n;
        for (Py_ssize_t i = 0; i < show && i < n; i++) {
            if (i > 0) {
                buf[pos++] = ',';
                buf[pos++] = ' ';
            }
            pos += tensor_format_scalar(tensor, i * tensor->strides[0], buf + pos, bufsize - pos);
        }
        if (n > show) {
            pos += snprintf(buf + pos, bufsize - pos, ", ..., ");
            pos += tensor_format_scalar(tensor, (n - 1) * tensor->strides[0], buf + pos, bufsize - pos);
        }
        buf[pos++] = ']';
        buf[pos] = '\0';
    }
    // For 2D tensor
    else if (tensor->rank == 2) {
        buf[pos++] = '[';
        Py_ssize_t rows = tensor->shape[0], cols = tensor->shape[1];
        Py_ssize_t show_rows = (rows > 6 && truncate) ? 3 : rows;
        for (Py_ssize_t i = 0; i < show_rows && i < rows; i++) {
            if (i > 0) {
                buf[pos++] = ',';
                buf[pos++] = '\n';
                buf[pos++] = ' ';
            }
            buf[pos++] = '[';
            Py_ssize_t show_cols = (cols > 10 && truncate) ? 4 : cols;
            for (Py_ssize_t j = 0; j < show_cols && j < cols; j++) {
                if (j > 0) {
                    buf[pos++] = ',';
                    buf[pos++] = ' ';
                }
                pos += tensor_format_scalar(tensor, i * tensor->strides[0] + j * tensor->strides[1], buf + pos,
                                            bufsize - pos);
            }
            if (cols > show_cols) {
                pos += snprintf(buf + pos, bufsize - pos, ", ..., ");
                pos += tensor_format_scalar(tensor, i * tensor->strides[0] + (cols - 1) * tensor->strides[1], buf + pos,
                                            bufsize - pos);
            }
            buf[pos++] = ']';
        }
        if (rows > show_rows) {
            pos += snprintf(buf + pos, bufsize - pos, ",\n ...,\n [");
            Py_ssize_t i = rows - 1;
            Py_ssize_t show_cols = (cols > 10 && truncate) ? 4 : cols;
            for (Py_ssize_t j = 0; j < show_cols && j < cols; j++) {
                if (j > 0) {
                    buf[pos++] = ',';
                    buf[pos++] = ' ';
                }
                pos += tensor_format_scalar(tensor, i * tensor->strides[0] + j * tensor->strides[1], buf + pos,
                                            bufsize - pos);
            }
            if (cols > show_cols) {
                pos += snprintf(buf + pos, bufsize - pos, ", ..., ");
                pos += tensor_format_scalar(tensor, i * tensor->strides[0] + (cols - 1) * tensor->strides[1], buf + pos,
                                            bufsize - pos);
            }
            buf[pos++] = ']';
        }
        buf[pos++] = ']';
        buf[pos] = '\0';
    }
    // For higher rank, fall back to repr-style
    else {
        PyMem_Free(buf);
        return NDArray_repr(self);
    }

    PyObject *result = PyUnicode_FromString(buf);
    PyMem_Free(buf);
    return result;
}

/// @brief  Check if a tensor is C-contiguous.
static int tensor_is_c_contig(NDArray *tensor, size_t item_size) {
    Py_ssize_t expected = (Py_ssize_t)item_size;
    for (size_t i = tensor->rank; i > 0; i--) {
        if (tensor->strides[i - 1] != expected) return 0;
        expected *= tensor->shape[i - 1];
    }
    return 1;
}

/// @brief  Check if a tensor is Fortran-contiguous.
static int tensor_is_f_contig(NDArray *tensor, size_t item_size) {
    Py_ssize_t expected = (Py_ssize_t)item_size;
    for (size_t i = 0; i < tensor->rank; i++) {
        if (tensor->strides[i] != expected) return 0;
        expected *= tensor->shape[i];
    }
    return 1;
}

/// @brief  Rich comparison for NDArray (==, !=).
static PyObject *NDArray_richcompare(PyObject *self, PyObject *other, int op) {
    // Only support == and !=.
    if (op != Py_EQ && op != Py_NE) { Py_RETURN_NOTIMPLEMENTED; }

    // Check whether the other operand is a NDArray.
    if (!PyObject_TypeCheck(other, &NDArrayType)) {
        // Try to compare via the buffer protocol.
        Py_buffer self_buf, other_buf;
        if (PyObject_GetBuffer(self, &self_buf, PyBUF_SIMPLE) != 0) {
            PyErr_Clear();
            Py_RETURN_NOTIMPLEMENTED;
        }
        if (PyObject_GetBuffer(other, &other_buf, PyBUF_SIMPLE) != 0) {
            PyBuffer_Release(&self_buf);
            PyErr_Clear();
            Py_RETURN_NOTIMPLEMENTED;
        }

        int equal = (self_buf.len == other_buf.len) && (memcmp(self_buf.buf, other_buf.buf, self_buf.len) == 0);
        PyBuffer_Release(&self_buf);
        PyBuffer_Release(&other_buf);

        if (op == Py_EQ) return PyBool_FromLong(equal);
        else return PyBool_FromLong(!equal);
    }

    NDArray *a = (NDArray *)self;
    NDArray *b = (NDArray *)other;

    // Check that datatype and rank match.
    if (a->datatype != b->datatype || a->rank != b->rank) {
        if (op == Py_EQ) Py_RETURN_FALSE;
        else Py_RETURN_TRUE;
    }

    // Check that shapes match.
    for (size_t i = 0; i < a->rank; i++) {
        if (a->shape[i] != b->shape[i]) {
            if (op == Py_EQ) Py_RETURN_FALSE;
            else Py_RETURN_TRUE;
        }
    }

    // Compare data.
    size_t item_size = bytes_per_datatype(a->datatype);
    Py_ssize_t total = 1;
    for (size_t i = 0; i < a->rank; i++) total *= a->shape[i];

    // For contiguous tensors, use memcmp.
    int a_contig = tensor_is_c_contig(a, item_size);
    int b_contig = tensor_is_c_contig(b, item_size);

    int equal;
    if (a_contig && b_contig) { equal = (memcmp(a->data, b->data, total * item_size) == 0); }
    else {
        // Element-by-element comparison for non-contiguous data.
        equal = 1;
        for (Py_ssize_t flat = 0; flat < total && equal; flat++) {
            // Compute offsets.
            size_t a_off = 0, b_off = 0;
            Py_ssize_t tmp = flat;
            for (size_t d = a->rank; d > 0; d--) {
                Py_ssize_t idx = tmp % a->shape[d - 1];
                tmp /= a->shape[d - 1];
                a_off += idx * a->strides[d - 1];
                b_off += idx * b->strides[d - 1];
            }
            if (memcmp(a->data + a_off, b->data + b_off, item_size) != 0) equal = 0;
        }
    }

    if (op == Py_EQ) return PyBool_FromLong(equal);
    else return PyBool_FromLong(!equal);
}

/// @brief  copy() method. Return a deep copy of the tensor.
static PyObject *NDArray_copy(PyObject *self, PyObject *args) {
    (void)args;
    NDArray *tensor = (NDArray *)self;
    size_t item_size = bytes_per_datatype(tensor->datatype);

    Py_ssize_t total = 1;
    for (size_t i = 0; i < tensor->rank; i++) total *= tensor->shape[i];

    NDArray *result = PyObject_NewVar(NDArray, &NDArrayType, total * item_size);
    if (!result) return NULL;

    result->datatype = tensor->datatype;
    result->rank = tensor->rank;
    result->parent = NULL;
    result->data = result->start;

    // Copy shape and compute contiguous strides
    Py_ssize_t stride = item_size;
    for (size_t i = tensor->rank; i > 0; i--) {
        result->shape[i - 1] = tensor->shape[i - 1];
        result->strides[i - 1] = stride;
        stride *= tensor->shape[i - 1];
    }
    for (size_t i = tensor->rank; i < NK_NDARRAY_MAX_RANK; i++) {
        result->shape[i] = 0;
        result->strides[i] = 0;
    }

    // Copy data (handle non-contiguous source)
    if (tensor_is_c_contig(tensor, item_size)) { memcpy(result->data, tensor->data, total * item_size); }
    else {
        // Element-by-element copy
        for (Py_ssize_t flat = 0; flat < total; flat++) {
            size_t src_off = 0, dst_off = flat * item_size;
            Py_ssize_t tmp = flat;
            for (size_t d = tensor->rank; d > 0; d--) {
                Py_ssize_t idx = tmp % tensor->shape[d - 1];
                tmp /= tensor->shape[d - 1];
                src_off += idx * tensor->strides[d - 1];
            }
            memcpy(result->data + dst_off, tensor->data + src_off, item_size);
        }
    }

    return (PyObject *)result;
}

/// @brief  reshape() method. Return a new tensor with a different shape and the same number of elements.
static PyObject *NDArray_reshape(PyObject *self, PyObject *args) {
    NDArray *tensor = (NDArray *)self;

    // Parse shape argument (can be tuple or multiple args)
    Py_ssize_t new_shape[NK_NDARRAY_MAX_RANK];
    size_t new_rank = 0;

    if (PyTuple_GET_SIZE(args) == 1 && PyTuple_Check(PyTuple_GET_ITEM(args, 0))) {
        // Single tuple argument
        PyObject *shape_tuple = PyTuple_GET_ITEM(args, 0);
        new_rank = PyTuple_GET_SIZE(shape_tuple);
        if (new_rank > NK_NDARRAY_MAX_RANK) {
            PyErr_Format(PyExc_ValueError, "reshape: too many dimensions (%zu > %d)", new_rank, NK_NDARRAY_MAX_RANK);
            return NULL;
        }
        for (size_t i = 0; i < new_rank; i++) {
            PyObject *item = PyTuple_GET_ITEM(shape_tuple, i);
            if (!PyLong_Check(item)) {
                PyErr_SetString(PyExc_TypeError, "reshape: shape dimensions must be integers");
                return NULL;
            }
            new_shape[i] = PyLong_AsSsize_t(item);
            if (new_shape[i] < 0) {
                PyErr_SetString(PyExc_ValueError, "reshape: negative dimensions not supported");
                return NULL;
            }
        }
    }
    else {
        // Multiple integer arguments
        new_rank = PyTuple_GET_SIZE(args);
        if (new_rank > NK_NDARRAY_MAX_RANK) {
            PyErr_Format(PyExc_ValueError, "reshape: too many dimensions (%zu > %d)", new_rank, NK_NDARRAY_MAX_RANK);
            return NULL;
        }
        for (size_t i = 0; i < new_rank; i++) {
            PyObject *item = PyTuple_GET_ITEM(args, i);
            if (!PyLong_Check(item)) {
                PyErr_SetString(PyExc_TypeError, "reshape: shape dimensions must be integers");
                return NULL;
            }
            new_shape[i] = PyLong_AsSsize_t(item);
            if (new_shape[i] < 0) {
                PyErr_SetString(PyExc_ValueError, "reshape: negative dimensions not supported");
                return NULL;
            }
        }
    }

    // Compute total elements in new shape
    Py_ssize_t new_total = 1;
    for (size_t i = 0; i < new_rank; i++) new_total *= new_shape[i];

    // Compute total elements in current shape
    Py_ssize_t old_total = 1;
    for (size_t i = 0; i < tensor->rank; i++) old_total *= tensor->shape[i];

    if (new_total != old_total) {
        PyErr_Format(PyExc_ValueError, "reshape: cannot reshape tensor of size %zd into shape with size %zd", old_total,
                     new_total);
        return NULL;
    }

    size_t item_size = bytes_per_datatype(tensor->datatype);

    // Check if source is contiguous (can create a view)
    if (tensor_is_c_contig(tensor, item_size)) {
        // Source is contiguous - create a zero-copy view with new shape
        Py_ssize_t new_strides[NK_NDARRAY_MAX_RANK];
        Py_ssize_t stride = item_size;
        for (size_t i = new_rank; i > 0; i--) {
            new_strides[i - 1] = stride;
            stride *= new_shape[i - 1];
        }

        // Get the root parent for the view reference
        NDArray *root_parent = tensor->parent ? (NDArray *)tensor->parent : tensor;

        return (PyObject *)NDArray_view(root_parent, tensor->data, tensor->datatype, new_rank, new_shape, new_strides);
    }

    // Non-contiguous source - must copy data
    NDArray *result = PyObject_NewVar(NDArray, &NDArrayType, new_total * item_size);
    if (!result) return NULL;

    result->datatype = tensor->datatype;
    result->rank = new_rank;
    result->parent = NULL;
    result->data = result->start;

    // Set new shape and compute contiguous strides
    Py_ssize_t stride = item_size;
    for (size_t i = new_rank; i > 0; i--) {
        result->shape[i - 1] = new_shape[i - 1];
        result->strides[i - 1] = stride;
        stride *= new_shape[i - 1];
    }
    for (size_t i = new_rank; i < NK_NDARRAY_MAX_RANK; i++) {
        result->shape[i] = 0;
        result->strides[i] = 0;
    }

    // Element-by-element copy in row-major order
    for (Py_ssize_t flat = 0; flat < old_total; flat++) {
        size_t src_off = 0;
        Py_ssize_t tmp = flat;
        for (size_t d = tensor->rank; d > 0; d--) {
            Py_ssize_t idx = tmp % tensor->shape[d - 1];
            tmp /= tensor->shape[d - 1];
            src_off += idx * tensor->strides[d - 1];
        }
        memcpy(result->data + flat * item_size, tensor->data + src_off, item_size);
    }

    return (PyObject *)result;
}

// region Reduction Methods (Stride-Aware)

/// @brief  Recursively sum all elements, processing innermost dimension with SIMD kernel
static void reduce_sum_recursive(NDArray *t, size_t dim, char *ptr, nk_f64_t *total, nk_i64_t *total_int) {
    if (dim == t->rank - 1) {
        // Base case: innermost dimension - call kernel with actual stride
        switch (t->datatype) {
        case nk_f64_k: {
            nk_f64_t p = 0;
            nk_reduce_add_f64((nk_f64_t *)ptr, t->shape[dim], t->strides[dim], &p);
            *total += p;
        } break;
        case nk_f32_k: {
            nk_f64_t p = 0;
            nk_reduce_add_f32((nk_f32_t *)ptr, t->shape[dim], t->strides[dim], &p);
            *total += p;
        } break;
        case nk_i8_k: {
            nk_i64_t p = 0;
            nk_reduce_add_i8((nk_i8_t *)ptr, t->shape[dim], t->strides[dim], &p);
            *total_int += p;
        } break;
        case nk_u8_k: {
            nk_u64_t p = 0;
            nk_reduce_add_u8((nk_u8_t *)ptr, t->shape[dim], t->strides[dim], &p);
            *total_int += (nk_i64_t)p;
        } break;
        case nk_i16_k: {
            nk_i64_t p = 0;
            nk_reduce_add_i16((nk_i16_t *)ptr, t->shape[dim], t->strides[dim], &p);
            *total_int += p;
        } break;
        case nk_u16_k: {
            nk_u64_t p = 0;
            nk_reduce_add_u16((nk_u16_t *)ptr, t->shape[dim], t->strides[dim], &p);
            *total_int += (nk_i64_t)p;
        } break;
        case nk_i32_k: {
            nk_i64_t p = 0;
            nk_reduce_add_i32((nk_i32_t *)ptr, t->shape[dim], t->strides[dim], &p);
            *total_int += p;
        } break;
        case nk_u32_k: {
            nk_u64_t p = 0;
            nk_reduce_add_u32((nk_u32_t *)ptr, t->shape[dim], t->strides[dim], &p);
            *total_int += (nk_i64_t)p;
        } break;
        case nk_i64_k: {
            nk_i64_t p = 0;
            nk_reduce_add_i64((nk_i64_t *)ptr, t->shape[dim], t->strides[dim], &p);
            *total_int += p;
        } break;
        case nk_u64_k: {
            nk_u64_t p = 0;
            nk_reduce_add_u64((nk_u64_t *)ptr, t->shape[dim], t->strides[dim], &p);
            *total_int += (nk_i64_t)p;
        } break;
        default: break;
        }
        return;
    }
    // Recursive case: iterate over this dimension
    for (Py_ssize_t i = 0; i < t->shape[dim]; i++) {
        reduce_sum_recursive(t, dim + 1, ptr + i * t->strides[dim], total, total_int);
    }
}

/// @brief Sum all elements: tensor.sum() -> scalar
static PyObject *NDArray_sum(PyObject *self, PyObject *args) {
    (void)args;
    NDArray *tensor = (NDArray *)self;
    size_t total_elems = 1;
    for (size_t i = 0; i < tensor->rank; i++) total_elems *= (size_t)tensor->shape[i];
    if (total_elems == 0) return PyFloat_FromDouble(0.0);
    if (tensor->rank == 0) return tensor_read_scalar(tensor, 0);
    nk_f64_t result_f = 0;
    nk_i64_t result_i = 0;
    reduce_sum_recursive(tensor, 0, tensor->data, &result_f, &result_i);
    // Return appropriate Python type based on dtype
    switch (tensor->datatype) {
    case nk_f64_k:
    case nk_f32_k:
    case nk_f16_k:
    case nk_bf16_k: return PyFloat_FromDouble(result_f);
    default: return PyLong_FromLongLong(result_i);
    }
}

/// @brief  Recursively find minimum, tracking both value and flat index
static void reduce_min_recursive(NDArray *t, size_t dim, char *ptr, size_t base_idx, nk_f64_t *global_min_f,
                                 nk_i64_t *global_min_i, size_t *global_idx) {
    if (dim == t->rank - 1) {
        nk_size_t local_idx;
        switch (t->datatype) {
        case nk_f32_k: {
            nk_f32_t v;
            nk_reduce_min_f32((nk_f32_t *)ptr, t->shape[dim], t->strides[dim], &v, &local_idx);
            if (v < *global_min_f) {
                *global_min_f = v;
                *global_idx = base_idx + local_idx;
            }
        } break;
        case nk_f64_k: {
            nk_f64_t v;
            nk_reduce_min_f64((nk_f64_t *)ptr, t->shape[dim], t->strides[dim], &v, &local_idx);
            if (v < *global_min_f) {
                *global_min_f = v;
                *global_idx = base_idx + local_idx;
            }
        } break;
        case nk_i8_k: {
            nk_i8_t v;
            nk_reduce_min_i8((nk_i8_t *)ptr, t->shape[dim], t->strides[dim], &v, &local_idx);
            if (v < *global_min_i) {
                *global_min_i = v;
                *global_idx = base_idx + local_idx;
            }
        } break;
        case nk_u8_k: {
            nk_u8_t v;
            nk_reduce_min_u8((nk_u8_t *)ptr, t->shape[dim], t->strides[dim], &v, &local_idx);
            if ((nk_i64_t)v < *global_min_i) {
                *global_min_i = v;
                *global_idx = base_idx + local_idx;
            }
        } break;
        case nk_i16_k: {
            nk_i16_t v;
            nk_reduce_min_i16((nk_i16_t *)ptr, t->shape[dim], t->strides[dim], &v, &local_idx);
            if (v < *global_min_i) {
                *global_min_i = v;
                *global_idx = base_idx + local_idx;
            }
        } break;
        case nk_u16_k: {
            nk_u16_t v;
            nk_reduce_min_u16((nk_u16_t *)ptr, t->shape[dim], t->strides[dim], &v, &local_idx);
            if ((nk_i64_t)v < *global_min_i) {
                *global_min_i = v;
                *global_idx = base_idx + local_idx;
            }
        } break;
        case nk_i32_k: {
            nk_i32_t v;
            nk_reduce_min_i32((nk_i32_t *)ptr, t->shape[dim], t->strides[dim], &v, &local_idx);
            if (v < *global_min_i) {
                *global_min_i = v;
                *global_idx = base_idx + local_idx;
            }
        } break;
        case nk_u32_k: {
            nk_u32_t v;
            nk_reduce_min_u32((nk_u32_t *)ptr, t->shape[dim], t->strides[dim], &v, &local_idx);
            if ((nk_i64_t)v < *global_min_i) {
                *global_min_i = v;
                *global_idx = base_idx + local_idx;
            }
        } break;
        case nk_i64_k: {
            nk_i64_t v;
            nk_reduce_min_i64((nk_i64_t *)ptr, t->shape[dim], t->strides[dim], &v, &local_idx);
            if (v < *global_min_i) {
                *global_min_i = v;
                *global_idx = base_idx + local_idx;
            }
        } break;
        case nk_u64_k: {
            nk_u64_t v;
            nk_reduce_min_u64((nk_u64_t *)ptr, t->shape[dim], t->strides[dim], &v, &local_idx);
            if ((nk_i64_t)v < *global_min_i) {
                *global_min_i = (nk_i64_t)v;
                *global_idx = base_idx + local_idx;
            }
        } break;
        default: break;
        }
        return;
    }
    size_t inner_size = 1;
    for (size_t d = dim + 1; d < t->rank; d++) inner_size *= t->shape[d];
    for (Py_ssize_t i = 0; i < t->shape[dim]; i++) {
        reduce_min_recursive(t, dim + 1, ptr + i * t->strides[dim], base_idx + i * inner_size, global_min_f,
                             global_min_i, global_idx);
    }
}

/// @brief  Recursively find maximum, tracking both value and flat index
static void reduce_max_recursive(NDArray *t, size_t dim, char *ptr, size_t base_idx, nk_f64_t *global_max_f,
                                 nk_i64_t *global_max_i, size_t *global_idx) {
    if (dim == t->rank - 1) {
        nk_size_t local_idx;
        switch (t->datatype) {
        case nk_f32_k: {
            nk_f32_t v;
            nk_reduce_max_f32((nk_f32_t *)ptr, t->shape[dim], t->strides[dim], &v, &local_idx);
            if (v > *global_max_f) {
                *global_max_f = v;
                *global_idx = base_idx + local_idx;
            }
        } break;
        case nk_f64_k: {
            nk_f64_t v;
            nk_reduce_max_f64((nk_f64_t *)ptr, t->shape[dim], t->strides[dim], &v, &local_idx);
            if (v > *global_max_f) {
                *global_max_f = v;
                *global_idx = base_idx + local_idx;
            }
        } break;
        case nk_i8_k: {
            nk_i8_t v;
            nk_reduce_max_i8((nk_i8_t *)ptr, t->shape[dim], t->strides[dim], &v, &local_idx);
            if (v > *global_max_i) {
                *global_max_i = v;
                *global_idx = base_idx + local_idx;
            }
        } break;
        case nk_u8_k: {
            nk_u8_t v;
            nk_reduce_max_u8((nk_u8_t *)ptr, t->shape[dim], t->strides[dim], &v, &local_idx);
            if ((nk_i64_t)v > *global_max_i) {
                *global_max_i = v;
                *global_idx = base_idx + local_idx;
            }
        } break;
        case nk_i16_k: {
            nk_i16_t v;
            nk_reduce_max_i16((nk_i16_t *)ptr, t->shape[dim], t->strides[dim], &v, &local_idx);
            if (v > *global_max_i) {
                *global_max_i = v;
                *global_idx = base_idx + local_idx;
            }
        } break;
        case nk_u16_k: {
            nk_u16_t v;
            nk_reduce_max_u16((nk_u16_t *)ptr, t->shape[dim], t->strides[dim], &v, &local_idx);
            if ((nk_i64_t)v > *global_max_i) {
                *global_max_i = v;
                *global_idx = base_idx + local_idx;
            }
        } break;
        case nk_i32_k: {
            nk_i32_t v;
            nk_reduce_max_i32((nk_i32_t *)ptr, t->shape[dim], t->strides[dim], &v, &local_idx);
            if (v > *global_max_i) {
                *global_max_i = v;
                *global_idx = base_idx + local_idx;
            }
        } break;
        case nk_u32_k: {
            nk_u32_t v;
            nk_reduce_max_u32((nk_u32_t *)ptr, t->shape[dim], t->strides[dim], &v, &local_idx);
            if ((nk_i64_t)v > *global_max_i) {
                *global_max_i = v;
                *global_idx = base_idx + local_idx;
            }
        } break;
        case nk_i64_k: {
            nk_i64_t v;
            nk_reduce_max_i64((nk_i64_t *)ptr, t->shape[dim], t->strides[dim], &v, &local_idx);
            if (v > *global_max_i) {
                *global_max_i = v;
                *global_idx = base_idx + local_idx;
            }
        } break;
        case nk_u64_k: {
            nk_u64_t v;
            nk_reduce_max_u64((nk_u64_t *)ptr, t->shape[dim], t->strides[dim], &v, &local_idx);
            if ((nk_i64_t)v > *global_max_i) {
                *global_max_i = (nk_i64_t)v;
                *global_idx = base_idx + local_idx;
            }
        } break;
        default: break;
        }
        return;
    }
    size_t inner_size = 1;
    for (size_t d = dim + 1; d < t->rank; d++) inner_size *= t->shape[d];
    for (Py_ssize_t i = 0; i < t->shape[dim]; i++) {
        reduce_max_recursive(t, dim + 1, ptr + i * t->strides[dim], base_idx + i * inner_size, global_max_f,
                             global_max_i, global_idx);
    }
}

/// @brief Minimum element: tensor.min() -> scalar
static PyObject *NDArray_min(PyObject *self, PyObject *args) {
    (void)args;
    NDArray *tensor = (NDArray *)self;
    size_t total_elems = 1;
    for (size_t i = 0; i < tensor->rank; i++) total_elems *= (size_t)tensor->shape[i];
    if (total_elems == 0) {
        PyErr_SetString(PyExc_ValueError, "min of empty array");
        return NULL;
    }
    if (tensor->rank == 0) return tensor_read_scalar(tensor, 0);
    nk_f64_t global_min_f = INFINITY;
    nk_i64_t global_min_i = INT64_MAX;
    size_t global_idx = 0;
    reduce_min_recursive(tensor, 0, tensor->data, 0, &global_min_f, &global_min_i, &global_idx);
    switch (tensor->datatype) {
    case nk_f64_k:
    case nk_f32_k:
    case nk_f16_k:
    case nk_bf16_k: return PyFloat_FromDouble(global_min_f);
    default: return PyLong_FromLongLong(global_min_i);
    }
}

/// @brief Maximum element: tensor.max() -> scalar
static PyObject *NDArray_max(PyObject *self, PyObject *args) {
    (void)args;
    NDArray *tensor = (NDArray *)self;
    size_t total_elems = 1;
    for (size_t i = 0; i < tensor->rank; i++) total_elems *= (size_t)tensor->shape[i];
    if (total_elems == 0) {
        PyErr_SetString(PyExc_ValueError, "max of empty array");
        return NULL;
    }
    if (tensor->rank == 0) return tensor_read_scalar(tensor, 0);
    nk_f64_t global_max_f = -INFINITY;
    nk_i64_t global_max_i = INT64_MIN;
    size_t global_idx = 0;
    reduce_max_recursive(tensor, 0, tensor->data, 0, &global_max_f, &global_max_i, &global_idx);
    switch (tensor->datatype) {
    case nk_f64_k:
    case nk_f32_k:
    case nk_f16_k:
    case nk_bf16_k: return PyFloat_FromDouble(global_max_f);
    default: return PyLong_FromLongLong(global_max_i);
    }
}

/// @brief Index of minimum element: tensor.argmin() -> int
static PyObject *NDArray_argmin(PyObject *self, PyObject *args) {
    (void)args;
    NDArray *tensor = (NDArray *)self;
    size_t total_elems = 1;
    for (size_t i = 0; i < tensor->rank; i++) total_elems *= (size_t)tensor->shape[i];
    if (total_elems == 0) {
        PyErr_SetString(PyExc_ValueError, "argmin of empty array");
        return NULL;
    }
    if (tensor->rank == 0) return PyLong_FromLong(0);
    nk_f64_t global_min_f = INFINITY;
    nk_i64_t global_min_i = INT64_MAX;
    size_t global_idx = 0;
    reduce_min_recursive(tensor, 0, tensor->data, 0, &global_min_f, &global_min_i, &global_idx);
    return PyLong_FromSize_t(global_idx);
}

/// @brief Index of maximum element: tensor.argmax() -> int
static PyObject *NDArray_argmax(PyObject *self, PyObject *args) {
    (void)args;
    NDArray *tensor = (NDArray *)self;
    size_t total_elems = 1;
    for (size_t i = 0; i < tensor->rank; i++) total_elems *= (size_t)tensor->shape[i];
    if (total_elems == 0) {
        PyErr_SetString(PyExc_ValueError, "argmax of empty array");
        return NULL;
    }
    if (tensor->rank == 0) return PyLong_FromLong(0);
    nk_f64_t global_max_f = -INFINITY;
    nk_i64_t global_max_i = INT64_MIN;
    size_t global_idx = 0;
    reduce_max_recursive(tensor, 0, tensor->data, 0, &global_max_f, &global_max_i, &global_idx);
    return PyLong_FromSize_t(global_idx);
}

// endregion

/// @brief  Deallocate a NDArray and release the parent reference for views.
static void NDArray_dealloc(PyObject *self) {
    NDArray *tensor = (NDArray *)self;
    Py_XDECREF(tensor->parent); // Release parent reference if this is a view
    Py_TYPE(self)->tp_free(self);
}

static PyMethodDef NDArray_methods[] = {
    {"copy", NDArray_copy, METH_NOARGS, "Return a deep copy of the tensor"},
    {"reshape", NDArray_reshape, METH_VARARGS, "Return tensor reshaped to given dimensions"},
    // Reduction methods
    {"sum", NDArray_sum, METH_NOARGS, "Sum of all elements"},
    {"min", NDArray_min, METH_NOARGS, "Minimum element"},
    {"max", NDArray_max, METH_NOARGS, "Maximum element"},
    {"argmin", NDArray_argmin, METH_NOARGS, "Index of minimum element"},
    {"argmax", NDArray_argmax, METH_NOARGS, "Index of maximum element"},
    {NULL, NULL, 0, NULL} // Sentinel
};

static PyTypeObject NDArrayType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "numkong.NDArray",
    .tp_doc = "N-dimensional tensor with full NumPy-like API, supporting NumKong's type system",
    .tp_basicsize = sizeof(NDArray),
    // Instead of using `nk_fmax_t` for all the elements,
    // we use `char` to allow user to specify the datatype on `cdist`-like functions.
    .tp_itemsize = sizeof(char),
    .tp_dealloc = NDArray_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_as_buffer = &NDArray_as_buffer,
    .tp_as_number = &NDArray_as_number,
    .tp_as_sequence = &NDArray_as_sequence,
    .tp_as_mapping = &NDArray_as_mapping,
    .tp_getset = NDArray_getset,
    .tp_methods = NDArray_methods,
    .tp_repr = NDArray_repr,
    .tp_str = NDArray_str,
    .tp_richcompare = NDArray_richcompare,
    .tp_iter = NDArray_iter,
};

/// @brief  Return a Python scalar from a tensor byte offset.
static PyObject *tensor_read_scalar(NDArray *tensor, size_t byte_offset) {
    tensor_scalar_value_t value;
    if (!tensor_read_scalar_value(tensor, byte_offset, &value)) return NULL;

    switch (value.kind) {
    case tensor_scalar_kind_float: return PyFloat_FromDouble(value.real);
    case tensor_scalar_kind_complex: return PyComplex_FromDoubles(value.real, value.imag);
    case tensor_scalar_kind_int: return PyLong_FromLongLong(value.i64);
    case tensor_scalar_kind_uint: return PyLong_FromUnsignedLongLong(value.u64);
    default: PyErr_SetString(PyExc_TypeError, "unsupported datatype for indexing"); return NULL;
    }
}

/// @brief  Parse a slice object into start, stop, and step for a given dimension size.
static int parse_slice(PyObject *slice, Py_ssize_t dim_size, Py_ssize_t *start, Py_ssize_t *stop, Py_ssize_t *step,
                       Py_ssize_t *slice_len) {
    Py_ssize_t defstart, defstop;

    if (PySlice_Unpack(slice, start, stop, step) < 0) return -1;

    *slice_len = PySlice_AdjustIndices(dim_size, start, stop, *step);
    return 0;
}

/// @brief  Implement tensor[key] indexing with slice support.
static PyObject *NDArray_subscript(PyObject *self, PyObject *key) {
    NDArray *tensor = (NDArray *)self;

    // Handle 0D tensors; indexing is not allowed.
    if (tensor->rank == 0) {
        PyErr_SetString(PyExc_IndexError, "0-dimensional tensor cannot be indexed");
        return NULL;
    }

    size_t item_size = bytes_per_datatype(tensor->datatype);

    // Check whether this is a single slice (for example, tensor[1:3]).
    if (PySlice_Check(key)) {
        Py_ssize_t start, stop, step, slice_len;
        if (parse_slice(key, tensor->shape[0], &start, &stop, &step, &slice_len) < 0) return NULL;

        // Build the view shape and strides.
        Py_ssize_t new_shape[NK_NDARRAY_MAX_RANK];
        Py_ssize_t new_strides[NK_NDARRAY_MAX_RANK];

        new_shape[0] = slice_len;
        new_strides[0] = tensor->strides[0] * step; // Adjust the stride for step.

        // Copy inner dimensions unchanged.
        for (size_t i = 1; i < tensor->rank; i++) {
            new_shape[i] = tensor->shape[i];
            new_strides[i] = tensor->strides[i];
        }

        // Compute the data pointer for the view.
        char *view_data = tensor->data + start * tensor->strides[0];

        // Use the root parent for the view reference (handles chained views).
        NDArray *root_parent = tensor->parent ? (NDArray *)tensor->parent : tensor;

        // Create a zero-copy view.
        return (PyObject *)NDArray_view(root_parent, view_data, tensor->datatype, tensor->rank, new_shape, new_strides);
    }

    // Collect indices from the key (single int or tuple of ints/slices).
    Py_ssize_t indices[NK_NDARRAY_MAX_RANK];
    int is_slice[NK_NDARRAY_MAX_RANK] = {0};
    Py_ssize_t slice_start[NK_NDARRAY_MAX_RANK], slice_stop[NK_NDARRAY_MAX_RANK];
    Py_ssize_t slice_step[NK_NDARRAY_MAX_RANK], slice_len[NK_NDARRAY_MAX_RANK];
    size_t num_indices = 0;
    int has_slice = 0;

    if (PyLong_Check(key)) {
        // Single integer index.
        indices[0] = PyLong_AsSsize_t(key);
        if (indices[0] == -1 && PyErr_Occurred()) return NULL;
        num_indices = 1;
    }
    else if (PyTuple_Check(key)) {
        // Tuple of indices/slices.
        num_indices = PyTuple_GET_SIZE(key);
        if (num_indices > tensor->rank) {
            PyErr_Format(PyExc_IndexError, "too many indices for tensor of rank %zu", tensor->rank);
            return NULL;
        }
        for (size_t i = 0; i < num_indices; i++) {
            PyObject *idx_obj = PyTuple_GET_ITEM(key, i);
            if (PyLong_Check(idx_obj)) {
                indices[i] = PyLong_AsSsize_t(idx_obj);
                if (indices[i] == -1 && PyErr_Occurred()) return NULL;
            }
            else if (PySlice_Check(idx_obj)) {
                is_slice[i] = 1;
                has_slice = 1;
                if (parse_slice(idx_obj, tensor->shape[i], &slice_start[i], &slice_stop[i], &slice_step[i],
                                &slice_len[i]) < 0)
                    return NULL;
            }
            else {
                PyErr_SetString(PyExc_TypeError, "indices must be integers or slices");
                return NULL;
            }
        }
    }
    else {
        PyErr_SetString(PyExc_TypeError, "indices must be integers, slices, or tuple of integers/slices");
        return NULL;
    }

    // If the tuple contains slices, handle it specially.
    if (has_slice) {
        // For simplicity, only support slicing on the first dimension in a tuple.
        // Full multi-dimensional slicing is not yet implemented.
        if (num_indices == 1 && is_slice[0]) {
            // Already handled above for a single slice; should not reach here.
            PyErr_SetString(PyExc_NotImplementedError, "complex slicing not yet supported");
            return NULL;
        }
        // For now, only support tensor[int, slice] or tensor[slice, int] patterns.
        PyErr_SetString(
            PyExc_NotImplementedError,
            "multi-dimensional slicing (tensor[i, j:k]) not yet fully supported; use tensor[i][j:k] instead");
        return NULL;
    }

    // Validate and normalize indices (support negative indexing).
    size_t byte_offset = 0;
    for (size_t i = 0; i < num_indices; i++) {
        Py_ssize_t idx = indices[i];
        Py_ssize_t dim_size = tensor->shape[i];
        // Handle negative indexing.
        if (idx < 0) idx += dim_size;
        if (idx < 0 || idx >= dim_size) {
            PyErr_Format(PyExc_IndexError, "index %zd out of bounds for dimension %zu with size %zd", indices[i], i,
                         dim_size);
            return NULL;
        }
        byte_offset += idx * tensor->strides[i];
    }

    // If all dimensions are indexed, return a scalar.
    if (num_indices == tensor->rank) { return tensor_read_scalar(tensor, byte_offset); }

    // Otherwise, return a sub-tensor view (zero-copy)
    size_t new_rank = tensor->rank - num_indices;
    Py_ssize_t new_shape[NK_NDARRAY_MAX_RANK];
    Py_ssize_t new_strides[NK_NDARRAY_MAX_RANK];

    // Copy shape and strides from remaining dimensions
    for (size_t i = 0; i < new_rank; i++) {
        new_shape[i] = tensor->shape[num_indices + i];
        new_strides[i] = tensor->strides[num_indices + i];
    }

    // Compute data pointer for the view
    char *view_data = tensor->data + byte_offset;

    // Get the root parent for the view reference (handles chained views)
    NDArray *root_parent = tensor->parent ? (NDArray *)tensor->parent : tensor;

    // Create view (zero-copy)
    return (PyObject *)NDArray_view(root_parent, view_data, tensor->datatype, new_rank, new_shape, new_strides);
}

/// @brief  Iterator type for NDArray.
typedef struct {
    PyObject_HEAD     // //
    NDArray *tensor;  // Reference to the tensor being iterated
    Py_ssize_t index; // Current index in first dimension
} NDArrayIter;

/// @brief  Return the next item from the iterator.
static PyObject *NDArrayIter_next(PyObject *self) {
    NDArrayIter *iter = (NDArrayIter *)self;
    if (iter->index >= iter->tensor->shape[0]) {
        // StopIteration is signaled by returning NULL without setting an error
        return NULL;
    }
    // Use subscript to get the element at current index
    PyObject *idx = PyLong_FromSsize_t(iter->index);
    if (!idx) return NULL;
    iter->index++;
    PyObject *result = NDArray_subscript((PyObject *)iter->tensor, idx);
    Py_DECREF(idx);
    return result;
}

/// @brief  Deallocate the iterator.
static void NDArrayIter_dealloc(PyObject *self) {
    NDArrayIter *iter = (NDArrayIter *)self;
    Py_XDECREF(iter->tensor);
    Py_TYPE(self)->tp_free(self);
}

static PyTypeObject NDArrayIterType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "numkong.NDArrayIter",
    .tp_doc = "Iterator for NDArray",
    .tp_basicsize = sizeof(NDArrayIter),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_dealloc = NDArrayIter_dealloc,
    .tp_iter = PyObject_SelfIter,
    .tp_iternext = NDArrayIter_next,
};

/// @brief  Return an iterator over the first dimension.
static PyObject *NDArray_iter(PyObject *self) {
    NDArray *tensor = (NDArray *)self;
    if (tensor->rank == 0) {
        PyErr_SetString(PyExc_TypeError, "0-dimensional tensor is not iterable");
        return NULL;
    }
    NDArrayIter *iter = PyObject_New(NDArrayIter, &NDArrayIterType);
    if (!iter) return NULL;
    Py_INCREF(tensor);
    iter->tensor = tensor;
    iter->index = 0;
    return (PyObject *)iter;
}

#pragma region PackedMatrix Type

/// @brief  Opaque container for pre-packed matrix data used for matrix multiplication.
///         The packed buffer format is backend-specific and not directly accessible.
typedef struct PackedMatrix {
    PyObject_HEAD nk_datatype_t dtype; // bf16 or i8
    nk_size_t n;                       // Number of rows in original (n × k) matrix
    nk_size_t k;                       // Number of columns in original (n × k) matrix
    char start[];                      // Variable length packed data (owns data, no parent reference needed)
} PackedMatrix;

// Forward declaration
static PyTypeObject PackedMatrixType;

/// @brief  Get the n property (number of rows).
static PyObject *PackedMatrix_get_n(PyObject *self, void *closure) {
    (void)closure;
    PackedMatrix *pm = (PackedMatrix *)self;
    return PyLong_FromSize_t(pm->n);
}

/// @brief  Get the k property (number of columns or dimensions).
static PyObject *PackedMatrix_get_k(PyObject *self, void *closure) {
    (void)closure;
    PackedMatrix *pm = (PackedMatrix *)self;
    return PyLong_FromSize_t(pm->k);
}

/// @brief  Get the dtype property.
static PyObject *PackedMatrix_get_dtype(PyObject *self, void *closure) {
    (void)closure;
    PackedMatrix *pm = (PackedMatrix *)self;
    char const *dtype_str = "unknown";
    if (pm->dtype == nk_bf16_k) dtype_str = "bf16";
    else if (pm->dtype == nk_i8_k) dtype_str = "i8";
    return PyUnicode_FromString(dtype_str);
}

/// @brief  Get the nbytes property (size of the packed buffer).
static PyObject *PackedMatrix_get_nbytes(PyObject *self, void *closure) {
    (void)closure;
    PackedMatrix *pm = (PackedMatrix *)self;
    // Calculate size from structure
    return PyLong_FromSize_t(Py_SIZE(pm));
}

static PyGetSetDef PackedMatrix_getset[] = {
    {"n", PackedMatrix_get_n, NULL, "Number of rows in the original matrix", NULL},
    {"k", PackedMatrix_get_k, NULL, "Number of columns in the original matrix", NULL},
    {"dtype", PackedMatrix_get_dtype, NULL, "Data type of the matrix elements (bf16 or i8)", NULL},
    {"nbytes", PackedMatrix_get_nbytes, NULL, "Size of the packed buffer in bytes", NULL},
    {NULL, NULL, NULL, NULL, NULL} // Sentinel
};

/// @brief  Return the repr for PackedMatrix.
static PyObject *PackedMatrix_repr(PyObject *self) {
    PackedMatrix *pm = (PackedMatrix *)self;
    char const *dtype_str = "unknown";
    if (pm->dtype == nk_bf16_k) dtype_str = "bf16";
    else if (pm->dtype == nk_i8_k) dtype_str = "i8";
    return PyUnicode_FromFormat("<PackedMatrix n=%zu k=%zu dtype='%s' nbytes=%zu>", pm->n, pm->k, dtype_str,
                                (size_t)Py_SIZE(pm));
}

static PyTypeObject PackedMatrixType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "numkong.PackedMatrix",
    .tp_doc = "Opaque pre-packed matrix for fast matrix multiplication",
    .tp_basicsize = sizeof(PackedMatrix),
    .tp_itemsize = sizeof(char),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_getset = PackedMatrix_getset,
    .tp_repr = PackedMatrix_repr,
};

#pragma endregion PackedMatrix Type

/// @brief  Global variable that caches CPU capabilities, computed once when the module is loaded.
nk_capability_t static_capabilities = nk_cap_serial_k;

/// @brief Check string equality.
/// @return 1 if the strings are equal, 0 otherwise.
int same_string(char const *a, char const *b) { return strcmp(a, b) == 0; }

/// @brief Return 1 if a logical datatype is complex and represented as two scalars.
/// @return 1 if the datatype is complex, 0 otherwise.
int is_complex(nk_datatype_t datatype) {
    nk_dtype_info_t const *info = datatype_info(datatype);
    return info ? info->is_complex : 0;
}

/// @brief Convert a Python-style datatype string to a logical datatype, normalizing the format.
/// @return `nk_datatype_unknown_k` if the datatype is not supported, otherwise the logical datatype.
/// @see https://docs.python.org/3/library/struct.html#format-characters
/// @see https://numpy.org/doc/stable/reference/arrays.interface.html
/// @see https://github.com/pybind/pybind11/issues/1908
nk_datatype_t python_string_to_datatype(char const *name) {
    // Floating-point numbers:
    if (same_string(name, "float32") || same_string(name, "f32") || // NumKong-specific
        same_string(name, "f4") || same_string(name, "<f4") ||      // Sized float
        same_string(name, "f") || same_string(name, "<f"))          // Named type
        return nk_f32_k;
    else if (same_string(name, "float16") || same_string(name, "f16") || // NumKong-specific
             same_string(name, "f2") || same_string(name, "<f2") ||      // Sized float
             same_string(name, "e") || same_string(name, "<e"))          // Named type
        return nk_f16_k;
    else if (same_string(name, "float64") || same_string(name, "f64") || // NumKong-specific
             same_string(name, "f8") || same_string(name, "<f8") ||      // Sized float
             same_string(name, "d") || same_string(name, "<d"))          // Named type
        return nk_f64_k;
    //? The exact format is not defined, but TensorFlow uses 'E' for `bf16`?!
    else if (same_string(name, "bfloat16") || same_string(name, "bf16")) // NumKong-specific
        return nk_bf16_k;

    // FP8 formats (ML-focused 8-bit floats):
    else if (same_string(name, "e4m3")) // NumKong-specific
        return nk_e4m3_k;
    else if (same_string(name, "e5m2")) // NumKong-specific
        return nk_e5m2_k;

    // Complex numbers:
    else if (same_string(name, "complex64") ||                                             // NumKong-specific
             same_string(name, "F4") || same_string(name, "<F4") ||                        // Sized complex
             same_string(name, "Zf") || same_string(name, "F") || same_string(name, "<F")) // Named type
        return nk_f32c_k;
    else if (same_string(name, "complex128") ||                                            // NumKong-specific
             same_string(name, "F8") || same_string(name, "<F8") ||                        // Sized complex
             same_string(name, "Zd") || same_string(name, "D") || same_string(name, "<D")) // Named type
        return nk_f64c_k;
    else if (same_string(name, "complex32") ||                                             // NumKong-specific
             same_string(name, "F2") || same_string(name, "<F2") ||                        // Sized complex
             same_string(name, "Ze") || same_string(name, "E") || same_string(name, "<E")) // Named type
        return nk_f16c_k;
    //? The exact format is not defined, but TensorFlow uses 'E' for `bf16`?!
    else if (same_string(name, "bcomplex32") || same_string(name, "bfloat16c") || same_string(name, "bf16c"))
        return nk_bf16c_k;

    //! Boolean values:
    else if (same_string(name, "bin8") || // NumKong-specific
             same_string(name, "?"))      // Named type
        return nk_b8_k;

    // Signed integers:
    else if (same_string(name, "int8") ||                                                       // NumKong-specific
             same_string(name, "i1") || same_string(name, "|i1") || same_string(name, "<i1") || // Sized integer
             same_string(name, "b") || same_string(name, "<b"))                                 // Named type
        return nk_i8_k;
    else if (same_string(name, "int16") ||                                                      // NumKong-specific
             same_string(name, "i2") || same_string(name, "|i2") || same_string(name, "<i2") || // Sized integer
             same_string(name, "h") || same_string(name, "<h"))                                 // Named type
        return nk_i16_k;

    //! On Windows the 32-bit and 64-bit signed integers will have different specifiers:
    //! https://github.com/pybind/pybind11/issues/1908
#if defined(_MSC_VER) || defined(__i386__)
    else if (same_string(name, "int32") ||                                                      // NumKong-specific
             same_string(name, "i4") || same_string(name, "|i4") || same_string(name, "<i4") || // Sized integer
             same_string(name, "l") || same_string(name, "<l"))                                 // Named type
        return nk_i32_k;
    else if (same_string(name, "int64") ||                                                      // NumKong-specific
             same_string(name, "i8") || same_string(name, "|i8") || same_string(name, "<i8") || // Sized integer
             same_string(name, "q") || same_string(name, "<q"))                                 // Named type
        return nk_i64_k;
#else // On Linux and macOS:
    else if (same_string(name, "int32") ||                                                      // NumKong-specific
             same_string(name, "i4") || same_string(name, "|i4") || same_string(name, "<i4") || // Sized integer
             same_string(name, "i") || same_string(name, "<i"))                                 // Named type
        return nk_i32_k;
    else if (same_string(name, "int64") ||                                                      // NumKong-specific
             same_string(name, "i8") || same_string(name, "|i8") || same_string(name, "<i8") || // Sized integer
             same_string(name, "l") || same_string(name, "<l"))                                 // Named type
        return nk_i64_k;
#endif

    // Unsigned integers:
    else if (same_string(name, "uint8") ||                                                      // NumKong-specific
             same_string(name, "u1") || same_string(name, "|u1") || same_string(name, "<u1") || // Sized integer
             same_string(name, "B") || same_string(name, "<B"))                                 // Named type
        return nk_u8_k;
    else if (same_string(name, "uint16") ||                                                     // NumKong-specific
             same_string(name, "u2") || same_string(name, "|u2") || same_string(name, "<u2") || // Sized integer
             same_string(name, "H") || same_string(name, "<H"))                                 // Named type
        return nk_u16_k;

    //! On Windows the 32-bit and 64-bit unsigned integers will have different specifiers:
    //! https://github.com/pybind/pybind11/issues/1908
#if defined(_MSC_VER) || defined(__i386__)
    else if (same_string(name, "uint32") ||                                                     // NumKong-specific
             same_string(name, "i4") || same_string(name, "|i4") || same_string(name, "<i4") || // Sized integer
             same_string(name, "L") || same_string(name, "<L"))                                 // Named type
        return nk_u32_k;
    else if (same_string(name, "uint64") ||                                                     // NumKong-specific
             same_string(name, "i8") || same_string(name, "|i8") || same_string(name, "<i8") || // Sized integer
             same_string(name, "Q") || same_string(name, "<Q"))                                 // Named type
        return nk_u64_k;
#else // On Linux and macOS:
    else if (same_string(name, "uint32") ||                                                     // NumKong-specific
             same_string(name, "u4") || same_string(name, "|u4") || same_string(name, "<u4") || // Sized integer
             same_string(name, "I") || same_string(name, "<I"))                                 // Named type
        return nk_u32_k;
    else if (same_string(name, "uint64") ||                                                     // NumKong-specific
             same_string(name, "u8") || same_string(name, "|u8") || same_string(name, "<u8") || // Sized integer
             same_string(name, "L") || same_string(name, "<L"))                                 // Named type
        return nk_u64_k;
#endif

    else return nk_datatype_unknown_k;
}

/// @brief Return the Python string representation of a datatype for the buffer protocol.
/// @param dtype Logical datatype, can be complex.
/// @return "unknown" if the datatype is not supported, otherwise a string.
/// @see https://docs.python.org/3/library/struct.html#format-characters
char const *datatype_to_python_string(nk_datatype_t dtype) {
    nk_dtype_info_t const *info = datatype_info(dtype);
    return info ? info->buffer_format : "unknown";
}

/// @brief Copy a distance to a target datatype, downcasting when necessary.
/// @return 1 if the cast was successful, 0 if the target datatype is not supported.
/// @note For integer types, we use rounding (not truncation) to minimize precision loss
///       when the source floating-point value is close to an integer boundary.
int cast_distance(nk_fmax_t distance, nk_datatype_t target_dtype, void *target_ptr, size_t offset) {
    nk_f32_t f32_val;
    switch (target_dtype) {
    case nk_f64c_k: ((nk_f64_t *)target_ptr)[offset] = (nk_f64_t)distance; return 1;
    case nk_f64_k: ((nk_f64_t *)target_ptr)[offset] = (nk_f64_t)distance; return 1;
    case nk_f32c_k: ((nk_f32_t *)target_ptr)[offset] = (nk_f32_t)distance; return 1;
    case nk_f32_k: ((nk_f32_t *)target_ptr)[offset] = (nk_f32_t)distance; return 1;
    case nk_f16c_k:
        f32_val = (nk_f32_t)distance;
        nk_f32_to_f16(&f32_val, (nk_f16_t *)target_ptr + offset);
        return 1;
    case nk_f16_k:
        f32_val = (nk_f32_t)distance;
        nk_f32_to_f16(&f32_val, (nk_f16_t *)target_ptr + offset);
        return 1;
    case nk_bf16c_k:
        f32_val = (nk_f32_t)distance;
        nk_f32_to_bf16(&f32_val, (nk_bf16_t *)target_ptr + offset);
        return 1;
    case nk_bf16_k:
        f32_val = (nk_f32_t)distance;
        nk_f32_to_bf16(&f32_val, (nk_bf16_t *)target_ptr + offset);
        return 1;
    // For integer types, use rounding instead of truncation to handle float32 vs float64 precision differences
    case nk_i8_k: ((nk_i8_t *)target_ptr)[offset] = (nk_i8_t)lround(distance); return 1;
    case nk_u8_k: ((nk_u8_t *)target_ptr)[offset] = (nk_u8_t)lround(distance); return 1;
    case nk_i16_k: ((nk_i16_t *)target_ptr)[offset] = (nk_i16_t)lround(distance); return 1;
    case nk_u16_k: ((nk_u16_t *)target_ptr)[offset] = (nk_u16_t)lround(distance); return 1;
    case nk_i32_k: ((nk_i32_t *)target_ptr)[offset] = (nk_i32_t)lround(distance); return 1;
    case nk_u32_k: ((nk_u32_t *)target_ptr)[offset] = (nk_u32_t)lround(distance); return 1;
    case nk_i64_k: ((nk_i64_t *)target_ptr)[offset] = (nk_i64_t)llround(distance); return 1;
    case nk_u64_k: ((nk_u64_t *)target_ptr)[offset] = (nk_u64_t)llround(distance); return 1;
    default: return 0;
    }
}

nk_kernel_kind_t python_string_to_metric_kind(char const *name) {
    if (same_string(name, "euclidean") || same_string(name, "l2")) return nk_kernel_l2_k;
    else if (same_string(name, "sqeuclidean") || same_string(name, "l2sq")) return nk_kernel_l2sq_k;
    else if (same_string(name, "dot") || same_string(name, "inner")) return nk_kernel_dot_k;
    else if (same_string(name, "vdot")) return nk_kernel_vdot_k;
    else if (same_string(name, "angular")) return nk_kernel_angular_k;
    else if (same_string(name, "jaccard")) return nk_kernel_jaccard_k;
    else if (same_string(name, "kullbackleibler") || same_string(name, "kld")) return nk_kernel_kld_k;
    else if (same_string(name, "jensenshannon") || same_string(name, "jsd")) return nk_kernel_jsd_k;
    else if (same_string(name, "hamming")) return nk_kernel_hamming_k;
    else if (same_string(name, "jaccard")) return nk_kernel_jaccard_k;
    else if (same_string(name, "bilinear")) return nk_kernel_bilinear_k;
    else if (same_string(name, "mahalanobis")) return nk_kernel_mahalanobis_k;
    else return nk_kernel_unknown_k;
}

/// @brief Check if a metric is commutative, i.e., if `metric(a, b) == metric(b, a)`.
/// @return 1 if the metric is commutative, 0 otherwise.
int kernel_is_commutative(nk_kernel_kind_t kind) {
    switch (kind) {
    case nk_kernel_kld_k: return 0;
    case nk_kernel_bilinear_k: return 0; // The kernel is commutative only when the matrix is symmetric.
    default: return 1;
    }
}

static char const doc_enable_capability[] = //
    "Enable a specific SIMD kernel family.\n\n"
    "Parameters:\n"
    "    capability : str\n"
    "        Name of the SIMD feature to enable (for example, 'haswell').";

static PyObject *api_enable_capability(PyObject *self, PyObject *cap_name_obj) {
    char const *cap_name = PyUnicode_AsUTF8(cap_name_obj);
    if (!cap_name) {
        PyErr_SetString(PyExc_TypeError, "Capability name must be a string");
        return NULL;
    }

    // ARM NEON capabilities
    if (same_string(cap_name, "neon")) { static_capabilities |= nk_cap_neon_k; }
    else if (same_string(cap_name, "neonhalf")) { static_capabilities |= nk_cap_neonhalf_k; }
    else if (same_string(cap_name, "neonfhm")) { static_capabilities |= nk_cap_neonfhm_k; }
    else if (same_string(cap_name, "neonbfdot")) { static_capabilities |= nk_cap_neonbfdot_k; }
    else if (same_string(cap_name, "neonsdot")) { static_capabilities |= nk_cap_neonsdot_k; }
    // ARM SVE capabilities
    else if (same_string(cap_name, "sve")) { static_capabilities |= nk_cap_sve_k; }
    else if (same_string(cap_name, "svehalf")) { static_capabilities |= nk_cap_svehalf_k; }
    else if (same_string(cap_name, "svebfdot")) { static_capabilities |= nk_cap_svebfdot_k; }
    else if (same_string(cap_name, "svesdot")) { static_capabilities |= nk_cap_svesdot_k; }
    else if (same_string(cap_name, "sve2")) { static_capabilities |= nk_cap_sve2_k; }
    else if (same_string(cap_name, "sve2p1")) { static_capabilities |= nk_cap_sve2p1_k; }
    // ARM SME capabilities
    else if (same_string(cap_name, "sme")) { static_capabilities |= nk_cap_sme_k; }
    else if (same_string(cap_name, "sme2")) { static_capabilities |= nk_cap_sme2_k; }
    else if (same_string(cap_name, "sme2p1")) { static_capabilities |= nk_cap_sme2p1_k; }
    else if (same_string(cap_name, "smef64")) { static_capabilities |= nk_cap_smef64_k; }
    else if (same_string(cap_name, "smehalf")) { static_capabilities |= nk_cap_smehalf_k; }
    else if (same_string(cap_name, "smebf16")) { static_capabilities |= nk_cap_smebf16_k; }
    else if (same_string(cap_name, "smelut2")) { static_capabilities |= nk_cap_smelut2_k; }
    else if (same_string(cap_name, "smefa64")) { static_capabilities |= nk_cap_smefa64_k; }
    // x86 capabilities
    else if (same_string(cap_name, "haswell")) { static_capabilities |= nk_cap_haswell_k; }
    else if (same_string(cap_name, "skylake")) { static_capabilities |= nk_cap_skylake_k; }
    else if (same_string(cap_name, "ice")) { static_capabilities |= nk_cap_ice_k; }
    else if (same_string(cap_name, "genoa")) { static_capabilities |= nk_cap_genoa_k; }
    else if (same_string(cap_name, "sapphire")) { static_capabilities |= nk_cap_sapphire_k; }
    else if (same_string(cap_name, "sapphire_amx")) { static_capabilities |= nk_cap_sapphire_amx_k; }
    else if (same_string(cap_name, "granite_amx")) { static_capabilities |= nk_cap_granite_amx_k; }
    else if (same_string(cap_name, "turin")) { static_capabilities |= nk_cap_turin_k; }
    else if (same_string(cap_name, "sierra")) { static_capabilities |= nk_cap_sierra_k; }
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
    "Disable a specific SIMD kernel family.\n\n"
    "Parameters:\n"
    "    capability : str\n"
    "        Name of the SIMD feature to disable (for example, 'haswell').";

static PyObject *api_disable_capability(PyObject *self, PyObject *cap_name_obj) {
    char const *cap_name = PyUnicode_AsUTF8(cap_name_obj);
    if (!cap_name) {
        PyErr_SetString(PyExc_TypeError, "Capability name must be a string");
        return NULL;
    }

    // ARM NEON capabilities
    if (same_string(cap_name, "neon")) { static_capabilities &= ~nk_cap_neon_k; }
    else if (same_string(cap_name, "neonhalf")) { static_capabilities &= ~nk_cap_neonhalf_k; }
    else if (same_string(cap_name, "neonfhm")) { static_capabilities &= ~nk_cap_neonfhm_k; }
    else if (same_string(cap_name, "neonbfdot")) { static_capabilities &= ~nk_cap_neonbfdot_k; }
    else if (same_string(cap_name, "neonsdot")) { static_capabilities &= ~nk_cap_neonsdot_k; }
    // ARM SVE capabilities
    else if (same_string(cap_name, "sve")) { static_capabilities &= ~nk_cap_sve_k; }
    else if (same_string(cap_name, "svehalf")) { static_capabilities &= ~nk_cap_svehalf_k; }
    else if (same_string(cap_name, "svebfdot")) { static_capabilities &= ~nk_cap_svebfdot_k; }
    else if (same_string(cap_name, "svesdot")) { static_capabilities &= ~nk_cap_svesdot_k; }
    else if (same_string(cap_name, "sve2")) { static_capabilities &= ~nk_cap_sve2_k; }
    else if (same_string(cap_name, "sve2p1")) { static_capabilities &= ~nk_cap_sve2p1_k; }
    // ARM SME capabilities
    else if (same_string(cap_name, "sme")) { static_capabilities &= ~nk_cap_sme_k; }
    else if (same_string(cap_name, "sme2")) { static_capabilities &= ~nk_cap_sme2_k; }
    else if (same_string(cap_name, "sme2p1")) { static_capabilities &= ~nk_cap_sme2p1_k; }
    else if (same_string(cap_name, "smef64")) { static_capabilities &= ~nk_cap_smef64_k; }
    else if (same_string(cap_name, "smehalf")) { static_capabilities &= ~nk_cap_smehalf_k; }
    else if (same_string(cap_name, "smebf16")) { static_capabilities &= ~nk_cap_smebf16_k; }
    else if (same_string(cap_name, "smelut2")) { static_capabilities &= ~nk_cap_smelut2_k; }
    else if (same_string(cap_name, "smefa64")) { static_capabilities &= ~nk_cap_smefa64_k; }
    // x86 capabilities
    else if (same_string(cap_name, "haswell")) { static_capabilities &= ~nk_cap_haswell_k; }
    else if (same_string(cap_name, "skylake")) { static_capabilities &= ~nk_cap_skylake_k; }
    else if (same_string(cap_name, "ice")) { static_capabilities &= ~nk_cap_ice_k; }
    else if (same_string(cap_name, "genoa")) { static_capabilities &= ~nk_cap_genoa_k; }
    else if (same_string(cap_name, "sapphire")) { static_capabilities &= ~nk_cap_sapphire_k; }
    else if (same_string(cap_name, "sapphire_amx")) { static_capabilities &= ~nk_cap_sapphire_amx_k; }
    else if (same_string(cap_name, "granite_amx")) { static_capabilities &= ~nk_cap_granite_amx_k; }
    else if (same_string(cap_name, "turin")) { static_capabilities &= ~nk_cap_turin_k; }
    else if (same_string(cap_name, "sierra")) { static_capabilities &= ~nk_cap_sierra_k; }
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
    "On x86 it includes: 'serial', 'haswell', 'skylake', 'ice', 'genoa', 'sapphire', 'turin'.\n"
    "On Arm it includes: 'serial', 'neon', 'sve', 'sve2', and their extensions.\n";

static PyObject *api_get_capabilities(PyObject *self) {
    nk_capability_t caps = static_capabilities;
    PyObject *cap_dict = PyDict_New();
    if (!cap_dict) return NULL;

#define ADD_CAP(name) PyDict_SetItemString(cap_dict, #name, PyBool_FromLong((caps) & nk_cap_##name##_k))

    ADD_CAP(serial);
    ADD_CAP(neon);
    ADD_CAP(sve);
    ADD_CAP(neonhalf);
    ADD_CAP(svehalf);
    ADD_CAP(neonbfdot);
    ADD_CAP(svebfdot);
    ADD_CAP(neonsdot);
    ADD_CAP(svesdot);
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

/// @brief Check if a Python object is a numeric scalar (int or float).
static int is_scalar(PyObject *obj) { return PyFloat_Check(obj) || PyLong_Check(obj); }

/// @brief Extract the double value from a Python numeric scalar.
/// @return 1 on success, 0 on failure.
static int get_scalar_value(PyObject *obj, double *value) {
    if (PyFloat_Check(obj)) {
        *value = PyFloat_AsDouble(obj);
        return !PyErr_Occurred();
    }
    else if (PyLong_Check(obj)) {
        *value = PyLong_AsDouble(obj);
        return !PyErr_Occurred();
    }
    return 0;
}

/// @brief Unpack a Python tensor object into a C structure.
/// @return 1 on success, 0 otherwise.
int parse_tensor(PyObject *tensor, Py_buffer *buffer, TensorArgument *parsed) {
    if (PyObject_GetBuffer(tensor, buffer, PyBUF_STRIDES | PyBUF_FORMAT) != 0) {
        PyErr_SetString(PyExc_TypeError, "arguments must support buffer protocol");
        return 0;
    }
    // Debug helper for buffer format strings.
    // printf("buffer format is %s\n", buffer->format);
    // printf("buffer ndim is %d\n", buffer->ndim);
    // printf("buffer shape is %d\n", buffer->shape[0]);
    // printf("buffer shape is %d\n", buffer->shape[1]);
    // printf("buffer itemsize is %d\n", buffer->itemsize);
    parsed->start = buffer->buf;
    parsed->datatype = python_string_to_datatype(buffer->format);
    if (parsed->datatype == nk_datatype_unknown_k) {
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
            PyErr_SetString(PyExc_ValueError, "Input vectors must be contiguous, check with `X.__array_interface__`");
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

    return 1;
}

static int NDArray_getbuffer(PyObject *export_from, Py_buffer *view, int flags) {
    NDArray *tensor = (NDArray *)export_from;
    size_t const item_size = bytes_per_datatype(tensor->datatype);

    // Validate buffer flags per PEP 3118.
    // PyBUF_WRITABLE: we support writable buffers.
    // PyBUF_FORMAT: we provide format string
    // PyBUF_ND: we provide shape (ndim > 0)
    // PyBUF_STRIDES: we provide strides
    // PyBUF_C_CONTIGUOUS / PyBUF_F_CONTIGUOUS: check if tensor is contiguous

    // Check contiguity flags - PyBUF_ANY_CONTIGUOUS accepts either C or F contiguous
    // Only check these if explicitly required (not just requested via PyBUF_STRIDES)
    int c_contig = tensor_is_c_contig(tensor, item_size);
    int f_contig = tensor_is_f_contig(tensor, item_size);

    // PyBUF_C_CONTIGUOUS requires C-contiguous layout
    if ((flags & PyBUF_C_CONTIGUOUS) == PyBUF_C_CONTIGUOUS && !c_contig) {
        PyErr_SetString(PyExc_BufferError, "buffer is not C-contiguous");
        view->obj = NULL;
        return -1;
    }

    // PyBUF_F_CONTIGUOUS requires Fortran-contiguous layout
    if ((flags & PyBUF_F_CONTIGUOUS) == PyBUF_F_CONTIGUOUS && !f_contig) {
        PyErr_SetString(PyExc_BufferError, "buffer is not Fortran-contiguous");
        view->obj = NULL;
        return -1;
    }

    // PyBUF_ANY_CONTIGUOUS accepts either C or F contiguous
    if ((flags & PyBUF_ANY_CONTIGUOUS) == PyBUF_ANY_CONTIGUOUS && !c_contig && !f_contig) {
        PyErr_SetString(PyExc_BufferError, "buffer is not contiguous");
        view->obj = NULL;
        return -1;
    }

    // Calculate total elements by multiplying all dimensions
    size_t total_items = 1;
    for (size_t i = 0; i < tensor->rank; i++) total_items *= (size_t)tensor->shape[i];

    view->buf = tensor->data;
    view->obj = (PyObject *)tensor;
    view->len = item_size * total_items;
    view->readonly = 0;
    view->itemsize = (Py_ssize_t)item_size;

    // Format string - only provide if requested
    if ((flags & PyBUF_FORMAT) == PyBUF_FORMAT) { view->format = datatype_to_python_string(tensor->datatype); }
    else { view->format = NULL; }

    // Shape and strides - only provide if requested
    if ((flags & PyBUF_ND) == PyBUF_ND) {
        view->ndim = (int)tensor->rank;
        view->shape = tensor->rank > 0 ? &tensor->shape[0] : NULL;
    }
    else {
        view->ndim = 0;
        view->shape = NULL;
    }

    if ((flags & PyBUF_STRIDES) == PyBUF_STRIDES) { view->strides = tensor->rank > 0 ? &tensor->strides[0] : NULL; }
    else { view->strides = NULL; }

    view->suboffsets = NULL;
    view->internal = NULL;

    Py_INCREF(tensor);
    return 0;
}

static void NDArray_releasebuffer(PyObject *export_from, Py_buffer *view) {
    // This function MUST NOT decrement view->obj, since that is done automatically in PyBuffer_Release().
    // https://docs.python.org/3/c-api/typeobj.html#c.PyBufferProcs.bf_releasebuffer
}

/// @brief  Create a new NDArray with the given shape and datatype.
/// @param  datatype  Logical datatype (f32, f64, f16, bf16, i8, etc.)
/// @param  rank      Number of dimensions (0 for scalar, up to NK_NDARRAY_MAX_RANK)
/// @param  shape     Array of extents for each dimension (can be NULL if rank == 0)
/// @return New NDArray object with uninitialized data, or NULL on failure.
static NDArray *NDArray_new(nk_datatype_t datatype, size_t rank, Py_ssize_t const *shape) {
    if (rank > NK_NDARRAY_MAX_RANK) {
        PyErr_Format(PyExc_ValueError, "Tensor rank %zu exceeds maximum %d", rank, NK_NDARRAY_MAX_RANK);
        return NULL;
    }

    // Calculate total bytes needed
    size_t const item_size = bytes_per_datatype(datatype);
    size_t total_items = 1;
    for (size_t i = 0; i < rank; i++) total_items *= (size_t)shape[i];
    size_t const total_bytes = total_items * item_size;

    // Allocate tensor with space for data
    NDArray *tensor = PyObject_NewVar(NDArray, &NDArrayType, total_bytes);
    if (!tensor) {
        PyErr_NoMemory();
        return NULL;
    }

    // Initialize metadata
    tensor->datatype = datatype;
    tensor->rank = rank;

    // Initialize shape and compute row-major strides
    for (size_t i = 0; i < NK_NDARRAY_MAX_RANK; i++) {
        tensor->shape[i] = (i < rank) ? shape[i] : 0;
        tensor->strides[i] = 0;
    }

    // Compute row-major strides (last dimension is contiguous)
    if (rank > 0) {
        tensor->strides[rank - 1] = (Py_ssize_t)item_size;
        for (size_t i = rank - 1; i > 0; i--) tensor->strides[i - 1] = tensor->strides[i] * tensor->shape[i];
    }

    // Initialize parent reference and data pointer
    tensor->parent = NULL;
    tensor->data = tensor->start;

    return tensor;
}

/// @brief  Create a 0D scalar tensor containing a single value.
/// @param  datatype  Logical datatype (f32, f64, f16, bf16, i8, etc.)
/// @param  value     Pointer to the scalar value to copy (item_size bytes)
/// @return New NDArray object with shape (), or NULL on failure.
static NDArray *NDArray_scalar(nk_datatype_t datatype, void const *value) {
    size_t const item_size = bytes_per_datatype(datatype);

    // Allocate tensor with space for one element
    NDArray *tensor = PyObject_NewVar(NDArray, &NDArrayType, item_size);
    if (!tensor) {
        PyErr_NoMemory();
        return NULL;
    }

    // Initialize as 0D tensor (scalar)
    tensor->datatype = datatype;
    tensor->rank = 0;
    for (size_t i = 0; i < NK_NDARRAY_MAX_RANK; i++) {
        tensor->shape[i] = 0;
        tensor->strides[i] = 0;
    }

    // Initialize parent reference and data pointer
    tensor->parent = NULL;
    tensor->data = tensor->start;

    // Copy the scalar value
    memcpy(tensor->data, value, item_size);

    return tensor;
}

/// @brief  Create a NDArray from an f64 scalar value.
/// @param  value  The f64 scalar value.
/// @return New NDArray object with shape (), or NULL on failure.
static NDArray *NDArray_scalar_f64(nk_f64_t value) { return NDArray_scalar(nk_f64_k, &value); }

/// @brief  Create a NDArray from an f32 scalar value.
/// @param  value  The f32 scalar value.
/// @return New NDArray object with shape (), or NULL on failure.
static NDArray *NDArray_scalar_f32(nk_f32_t value) { return NDArray_scalar(nk_f32_k, &value); }

/// @brief  Create a view tensor that references another tensor's data (zero-copy).
/// @param  parent     The parent tensor whose data will be referenced.
/// @param  data_ptr   Pointer into parent's data buffer.
/// @param  datatype   Logical datatype (inherited from parent).
/// @param  rank       Number of dimensions for the view.
/// @param  shape      Shape array for the view.
/// @param  strides    Strides array for the view (in bytes).
/// @return New NDArray view object, or NULL on failure.
static NDArray *NDArray_view(NDArray *parent, char *data_ptr, nk_datatype_t datatype, size_t rank,
                             Py_ssize_t const *shape, Py_ssize_t const *strides) {
    if (rank > NK_NDARRAY_MAX_RANK) {
        PyErr_Format(PyExc_ValueError, "View rank %zu exceeds maximum %d", rank, NK_NDARRAY_MAX_RANK);
        return NULL;
    }

    // Allocate tensor with NO inline data (0 bytes) - we'll reference parent's data
    NDArray *view = PyObject_NewVar(NDArray, &NDArrayType, 0);
    if (!view) {
        PyErr_NoMemory();
        return NULL;
    }

    // Initialize metadata
    view->datatype = datatype;
    view->rank = rank;

    // Copy shape and strides
    for (size_t i = 0; i < NK_NDARRAY_MAX_RANK; i++) {
        view->shape[i] = (i < rank) ? shape[i] : 0;
        view->strides[i] = (i < rank) ? strides[i] : 0;
    }

    // Set up parent reference and data pointer
    view->parent = (PyObject *)parent;
    Py_INCREF(parent);
    view->data = data_ptr;

    return view;
}

static PyObject *implement_dense_metric( //
    nk_kernel_kind_t metric_kind,        //
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
    nk_datatype_t dtype = nk_datatype_unknown_k, out_dtype = nk_datatype_unknown_k;
    Py_buffer a_buffer, b_buffer, out_buffer;
    TensorArgument a_parsed, b_parsed, out_parsed;
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
        if (dtype == nk_datatype_unknown_k) {
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
        if (out_dtype == nk_datatype_unknown_k) {
            PyErr_SetString(PyExc_ValueError, "Unsupported 'out_dtype'");
            return NULL;
        }
    }

    // Convert `a_obj` to `a_buffer` and to `a_parsed`. Same for `b_obj` and `out_obj`.
    if (!parse_tensor(a_obj, &a_buffer, &a_parsed) || !parse_tensor(b_obj, &b_buffer, &b_parsed)) return NULL;
    if (out_obj && !parse_tensor(out_obj, &out_buffer, &out_parsed)) return NULL;

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
        a_parsed.datatype == nk_datatype_unknown_k || b_parsed.datatype == nk_datatype_unknown_k) {
        PyErr_SetString(PyExc_TypeError,
                        "Input tensors must have matching datatypes, check with `X.__array_interface__`");
        goto cleanup;
    }
    if (dtype == nk_datatype_unknown_k) dtype = a_parsed.datatype;

    // Inference order for the output type:
    // 1. `out_dtype` named argument, if defined
    // 2. `out.dtype` attribute, if `out` is passed
    // 3. double precision float (or its complex variant)
    if (out_dtype == nk_datatype_unknown_k) {
        if (out_obj) { out_dtype = out_parsed.datatype; }
        else { out_dtype = is_complex(dtype) ? nk_f64c_k : nk_f64_k; }
    }

    // Make sure the return datatype is complex if the input datatype is complex, and the same for real numbers
    if (out_dtype != nk_datatype_unknown_k) {
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
    nk_metric_dense_punned_t metric = NULL;
    nk_capability_t capability = nk_cap_serial_k;
    nk_find_kernel_punned(metric_kind, dtype, static_capabilities, nk_cap_any_k, (nk_kernel_punned_t *)&metric,
                          &capability);
    if (!metric) {
        PyErr_Format( //
            PyExc_LookupError,
            "Unsupported metric '%c' and datatype combination across vectors ('%s'/'%s' and '%s'/'%s') and "
            "`dtype` override ('%s'/'%s')",
            metric_kind,                                                                             //
            a_buffer.format ? a_buffer.format : "nil", datatype_to_python_string(a_parsed.datatype), //
            b_buffer.format ? b_buffer.format : "nil", datatype_to_python_string(b_parsed.datatype), //
            dtype_str ? dtype_str : "nil", datatype_to_python_string(dtype));
        goto cleanup;
    }

    // If the distance is computed between two vectors, rather than matrices, return a scalar
    int const dtype_is_complex = is_complex(dtype);
    nk_datatype_t const kernel_out_dtype = metric_kernel_output_dtype(metric_kind, dtype);
    if (a_parsed.rank == 1 && b_parsed.rank == 1) {
        nk_scalar_buffer_t distances[2];
        metric(a_parsed.start, b_parsed.start, a_parsed.dimensions, distances);
        return_obj =         //
            dtype_is_complex //
                ? PyComplex_FromDoubles(nk_scalar_buffer_get_f64(&distances[0], kernel_out_dtype),
                                        nk_scalar_buffer_get_f64(&distances[1], kernel_out_dtype))
                : PyFloat_FromDouble(nk_scalar_buffer_get_f64(&distances[0], kernel_out_dtype));
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
    char *distances_start = NULL;
    size_t distances_stride_bytes = 0;

    // Allocate the output matrix if it wasn't provided
    if (!out_obj) {
        NDArray *distances_obj = PyObject_NewVar(NDArray, &NDArrayType,
                                                 count_components * bytes_per_datatype(out_dtype));
        if (!distances_obj) {
            PyErr_NoMemory();
            goto cleanup;
        }

        // Initialize the object
        distances_obj->datatype = out_dtype;
        distances_obj->rank = 1;
        distances_obj->shape[0] = count_pairs;
        distances_obj->shape[1] = 0;
        distances_obj->strides[0] = bytes_per_datatype(out_dtype);
        distances_obj->strides[1] = 0;
        distances_obj->parent = NULL;
        distances_obj->data = distances_obj->start;
        return_obj = (PyObject *)distances_obj;
        distances_start = distances_obj->data;
        distances_stride_bytes = distances_obj->strides[0];
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
        distances_start = (char *)&out_parsed.start[0];
        distances_stride_bytes = out_buffer.strides[0];
        //? Logic suggests to return `None` in in-place mode...
        //? SciPy decided differently.
        return_obj = Py_None;
    }

    // Now let's release the GIL for the parallel part using the underlying mechanism of `Py_BEGIN_ALLOW_THREADS`.
    PyThreadState *save = PyEval_SaveThread();

    // Compute the distances
    for (size_t i = 0; i < count_pairs; ++i) {
        nk_scalar_buffer_t result[2];
        metric(                                   //
            a_parsed.start + i * a_parsed.stride, //
            b_parsed.start + i * b_parsed.stride, //
            a_parsed.dimensions,                  //
            result);

        // Export out:
        cast_distance(nk_scalar_buffer_get_f64(&result[0], kernel_out_dtype), out_dtype,
                      distances_start + i * distances_stride_bytes, 0);
        if (dtype_is_complex)
            cast_distance(nk_scalar_buffer_get_f64(&result[1], kernel_out_dtype), out_dtype,
                          distances_start + i * distances_stride_bytes, 1);
    }

    PyEval_RestoreThread(save);

cleanup:
    PyBuffer_Release(&a_buffer);
    PyBuffer_Release(&b_buffer);
    PyBuffer_Release(&out_buffer);
    return return_obj;
}

static PyObject *implement_curved_metric( //
    nk_kernel_kind_t metric_kind,         //
    PyObject *const *args, Py_ssize_t const positional_args_count, PyObject *args_names_tuple) {

    PyObject *return_obj = NULL;

    // This function accepts up to 6 arguments:
    PyObject *a_obj = NULL;     // Required object, positional-only
    PyObject *b_obj = NULL;     // Required object, positional-only
    PyObject *c_obj = NULL;     // Required object, positional-only
    PyObject *dtype_obj = NULL; // Optional object, "dtype" keyword or positional

    // Once parsed, the arguments will be stored in these variables:
    char const *dtype_str = NULL;
    nk_datatype_t dtype = nk_datatype_unknown_k;
    Py_buffer a_buffer, b_buffer, c_buffer;
    TensorArgument a_parsed, b_parsed, c_parsed;
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
        if (dtype == nk_datatype_unknown_k) {
            PyErr_SetString(PyExc_ValueError, "Unsupported 'dtype'");
            return NULL;
        }
    }

    // Convert `a_obj` to `a_buffer` and to `a_parsed`. Same for `b_obj` and `out_obj`.
    if (!parse_tensor(a_obj, &a_buffer, &a_parsed) || !parse_tensor(b_obj, &b_buffer, &b_parsed) ||
        !parse_tensor(c_obj, &c_buffer, &c_parsed))
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
        a_parsed.datatype == nk_datatype_unknown_k || b_parsed.datatype == nk_datatype_unknown_k ||
        c_parsed.datatype == nk_datatype_unknown_k) {
        PyErr_SetString(PyExc_TypeError,
                        "Input tensors must have matching datatypes, check with `X.__array_interface__`");
        goto cleanup;
    }
    if (dtype == nk_datatype_unknown_k) dtype = a_parsed.datatype;

    // Look up the metric and the capability
    nk_metric_curved_punned_t metric = NULL;
    nk_capability_t capability = nk_cap_serial_k;
    nk_find_kernel_punned(metric_kind, dtype, static_capabilities, nk_cap_any_k, (nk_kernel_punned_t *)&metric,
                          &capability);
    if (!metric) {
        PyErr_Format( //
            PyExc_LookupError,
            "Unsupported metric '%c' and datatype combination across vectors ('%s'/'%s' and '%s'/'%s'), "
            "tensor ('%s'/'%s'), and `dtype` override ('%s'/'%s')",
            metric_kind,                                                                             //
            a_buffer.format ? a_buffer.format : "nil", datatype_to_python_string(a_parsed.datatype), //
            b_buffer.format ? b_buffer.format : "nil", datatype_to_python_string(b_parsed.datatype), //
            c_buffer.format ? c_buffer.format : "nil", datatype_to_python_string(c_parsed.datatype), //
            dtype_str ? dtype_str : "nil", datatype_to_python_string(dtype));
        goto cleanup;
    }

    // If the distance is computed between two vectors, rather than matrices, return a scalar
    int const dtype_is_complex = is_complex(dtype);
    nk_fmax_t distances[2];
    metric(a_parsed.start, b_parsed.start, c_parsed.start, a_parsed.dimensions, &distances[0]);
    return_obj =         //
        dtype_is_complex //
            ? PyComplex_FromDoubles(distances[0], distances[1])
            : PyFloat_FromDouble(distances[0]);

cleanup:
    PyBuffer_Release(&a_buffer);
    PyBuffer_Release(&b_buffer);
    PyBuffer_Release(&c_buffer);
    return return_obj;
}

static PyObject *implement_geospatial_metric( //
    nk_kernel_kind_t metric_kind,             //
    PyObject *const *args, Py_ssize_t const positional_args_count, PyObject *args_names_tuple) {

    PyObject *return_obj = NULL;

    // This function accepts up to 6 arguments:
    PyObject *a_lats_obj = NULL; // Required object, positional-only
    PyObject *a_lons_obj = NULL; // Required object, positional-only
    PyObject *b_lats_obj = NULL; // Required object, positional-only
    PyObject *b_lons_obj = NULL; // Required object, positional-only
    PyObject *dtype_obj = NULL;  // Optional object, "dtype" keyword or positional
    PyObject *out_obj = NULL;    // Optional object, "out" keyword-only

    // Once parsed, the arguments will be stored in these variables:
    char const *dtype_str = NULL;
    nk_datatype_t dtype = nk_datatype_unknown_k;
    Py_buffer a_lats_buffer, a_lons_buffer, b_lats_buffer, b_lons_buffer, out_buffer;
    TensorArgument a_lats_parsed, a_lons_parsed, b_lats_parsed, b_lons_parsed, out_parsed;
    memset(&a_lats_buffer, 0, sizeof(Py_buffer));
    memset(&a_lons_buffer, 0, sizeof(Py_buffer));
    memset(&b_lats_buffer, 0, sizeof(Py_buffer));
    memset(&b_lons_buffer, 0, sizeof(Py_buffer));
    memset(&out_buffer, 0, sizeof(Py_buffer));

    // Parse the arguments
    Py_ssize_t const args_names_count = args_names_tuple ? PyTuple_Size(args_names_tuple) : 0;
    Py_ssize_t const args_count = positional_args_count + args_names_count;
    if (args_count < 4 || args_count > 6) {
        PyErr_Format(PyExc_TypeError, "Function expects 4-6 arguments, got %zd", args_count);
        return NULL;
    }
    if (positional_args_count > 5) {
        PyErr_Format(PyExc_TypeError, "Only first 5 arguments can be positional, received %zd", positional_args_count);
        return NULL;
    }

    // Positional-only arguments (4 coordinate arrays)
    a_lats_obj = args[0];
    a_lons_obj = args[1];
    b_lats_obj = args[2];
    b_lons_obj = args[3];

    // Positional or keyword argument (dtype)
    if (positional_args_count == 5) dtype_obj = args[4];

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
        if (dtype == nk_datatype_unknown_k) {
            PyErr_SetString(PyExc_ValueError, "Unsupported 'dtype'");
            return NULL;
        }
    }

    // Convert input objects to buffers
    if (!parse_tensor(a_lats_obj, &a_lats_buffer, &a_lats_parsed) ||
        !parse_tensor(a_lons_obj, &a_lons_buffer, &a_lons_parsed) ||
        !parse_tensor(b_lats_obj, &b_lats_buffer, &b_lats_parsed) ||
        !parse_tensor(b_lons_obj, &b_lons_buffer, &b_lons_parsed))
        return NULL;
    if (out_obj && !parse_tensor(out_obj, &out_buffer, &out_parsed)) return NULL;

    // Check dimensions: all inputs must be 1D vectors of equal length
    if (a_lats_parsed.rank != 1 || a_lons_parsed.rank != 1 || b_lats_parsed.rank != 1 || b_lons_parsed.rank != 1) {
        PyErr_SetString(PyExc_ValueError, "All coordinate arrays must be 1D vectors");
        goto cleanup;
    }
    // For geospatial, n is the number of coordinate pairs (shape[0] for 1D arrays)
    size_t const n = a_lats_parsed.dimensions;
    if (a_lons_parsed.dimensions != n || b_lats_parsed.dimensions != n || b_lons_parsed.dimensions != n) {
        PyErr_SetString(PyExc_ValueError, "All coordinate arrays must have the same length");
        goto cleanup;
    }
    if (n == 0) {
        PyErr_SetString(PyExc_ValueError, "Coordinate arrays can't be empty");
        goto cleanup;
    }

    // Check data types: all must match
    if (a_lats_parsed.datatype != a_lons_parsed.datatype || a_lats_parsed.datatype != b_lats_parsed.datatype ||
        a_lats_parsed.datatype != b_lons_parsed.datatype || a_lats_parsed.datatype == nk_datatype_unknown_k) {
        PyErr_SetString(PyExc_TypeError, "All coordinate arrays must have the same datatype");
        goto cleanup;
    }
    if (dtype == nk_datatype_unknown_k) dtype = a_lats_parsed.datatype;

    // Look up the metric kernel
    nk_metric_geospatial_punned_t metric = NULL;
    nk_capability_t capability = nk_cap_serial_k;
    nk_find_kernel_punned(metric_kind, dtype, static_capabilities, nk_cap_any_k, (nk_kernel_punned_t *)&metric,
                          &capability);
    if (!metric) {
        PyErr_Format(PyExc_LookupError, "Unsupported metric '%c' and datatype '%s'", metric_kind,
                     datatype_to_python_string(dtype));
        goto cleanup;
    }

    // Allocate output or use provided
    nk_fmax_t *distances_start = NULL;
    if (!out_obj) {
        NDArray *distances_obj = PyObject_NewVar(NDArray, &NDArrayType, n * sizeof(double));
        if (!distances_obj) {
            PyErr_NoMemory();
            goto cleanup;
        }
        distances_obj->datatype = nk_f64_k;
        distances_obj->rank = 1;
        distances_obj->shape[0] = n;
        distances_obj->shape[1] = 0;
        distances_obj->strides[0] = sizeof(double);
        distances_obj->strides[1] = 0;
        distances_obj->parent = NULL;
        distances_obj->data = distances_obj->start;
        return_obj = (PyObject *)distances_obj;
        distances_start = (nk_fmax_t *)distances_obj->data;
    }
    else {
        if (out_parsed.dimensions < n) {
            PyErr_SetString(PyExc_ValueError, "Output array is too small");
            goto cleanup;
        }
        distances_start = (nk_fmax_t *)out_parsed.start;
        return_obj = Py_None;
    }

    // Call the kernel
    metric(a_lats_parsed.start, a_lons_parsed.start, b_lats_parsed.start, b_lons_parsed.start, n, distances_start);

cleanup:
    if (a_lats_buffer.buf) PyBuffer_Release(&a_lats_buffer);
    if (a_lons_buffer.buf) PyBuffer_Release(&a_lons_buffer);
    if (b_lats_buffer.buf) PyBuffer_Release(&b_lats_buffer);
    if (b_lons_buffer.buf) PyBuffer_Release(&b_lons_buffer);
    if (out_buffer.buf) PyBuffer_Release(&out_buffer);
    return return_obj;
}

static PyObject *implement_sparse_metric( //
    nk_kernel_kind_t metric_kind,         //
    PyObject *const *args, Py_ssize_t nargs) {
    if (nargs != 2) {
        PyErr_SetString(PyExc_TypeError, "Function expects only 2 arguments");
        return NULL;
    }

    PyObject *return_obj = NULL;
    PyObject *a_obj = args[0];
    PyObject *b_obj = args[1];

    Py_buffer a_buffer, b_buffer;
    TensorArgument a_parsed, b_parsed;
    if (!parse_tensor(a_obj, &a_buffer, &a_parsed) || !parse_tensor(b_obj, &b_buffer, &b_parsed)) return NULL;

    // Check dimensions
    if (a_parsed.rank != 1 || b_parsed.rank != 1) {
        PyErr_SetString(PyExc_ValueError, "First and second argument must be vectors");
        goto cleanup;
    }

    // Check data types
    if (a_parsed.datatype != b_parsed.datatype && a_parsed.datatype != nk_datatype_unknown_k &&
        b_parsed.datatype != nk_datatype_unknown_k) {
        PyErr_SetString(PyExc_TypeError,
                        "Input tensors must have matching datatypes, check with `X.__array_interface__`");
        goto cleanup;
    }

    nk_datatype_t dtype = a_parsed.datatype;
    nk_sparse_intersect_punned_t metric = NULL;
    nk_capability_t capability = nk_cap_serial_k;
    nk_find_kernel_punned(metric_kind, dtype, static_capabilities, nk_cap_any_k, (nk_kernel_punned_t *)&metric,
                          &capability);
    if (!metric) {
        PyErr_Format( //
            PyExc_LookupError, "Unsupported metric '%c' and datatype combination ('%s'/'%s' and '%s'/'%s')",
            metric_kind,                                                                             //
            a_buffer.format ? a_buffer.format : "nil", datatype_to_python_string(a_parsed.datatype), //
            b_buffer.format ? b_buffer.format : "nil", datatype_to_python_string(b_parsed.datatype));
        goto cleanup;
    }

    nk_fmax_t distance;
    metric(a_parsed.start, b_parsed.start, a_parsed.dimensions, b_parsed.dimensions, &distance);
    return_obj = PyFloat_FromDouble(distance);

cleanup:
    PyBuffer_Release(&a_buffer);
    PyBuffer_Release(&b_buffer);
    return return_obj;
}

static PyObject *implement_cdist(                        //
    PyObject *a_obj, PyObject *b_obj, PyObject *out_obj, //
    nk_kernel_kind_t metric_kind, size_t threads,        //
    nk_datatype_t dtype, nk_datatype_t out_dtype) {

    PyObject *return_obj = NULL;

    Py_buffer a_buffer, b_buffer, out_buffer;
    TensorArgument a_parsed, b_parsed, out_parsed;
    memset(&a_buffer, 0, sizeof(Py_buffer));
    memset(&b_buffer, 0, sizeof(Py_buffer));
    memset(&out_buffer, 0, sizeof(Py_buffer));

    // Error will be set by `parse_tensor` if the input is invalid
    if (!parse_tensor(a_obj, &a_buffer, &a_parsed) || !parse_tensor(b_obj, &b_buffer, &b_parsed)) return NULL;
    if (out_obj && !parse_tensor(out_obj, &out_buffer, &out_parsed)) return NULL;

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
        a_parsed.datatype == nk_datatype_unknown_k || b_parsed.datatype == nk_datatype_unknown_k) {
        PyErr_SetString(PyExc_TypeError,
                        "Input tensors must have matching datatypes, check with `X.__array_interface__`");
        goto cleanup;
    }
    if (dtype == nk_datatype_unknown_k) dtype = a_parsed.datatype;

    // Inference order for the output type:
    // 1. `out_dtype` named argument, if defined
    // 2. `out.dtype` attribute, if `out` is passed
    // 3. double precision float (or its complex variant)
    if (out_dtype == nk_datatype_unknown_k) {
        if (out_obj) { out_dtype = out_parsed.datatype; }
        else { out_dtype = is_complex(dtype) ? nk_f64c_k : nk_f64_k; }
    }

    // Make sure the return datatype is complex if the input datatype is complex, and the same for real numbers
    if (out_dtype != nk_datatype_unknown_k) {
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
    nk_metric_dense_punned_t metric = NULL;
    nk_capability_t capability = nk_cap_serial_k;
    nk_find_kernel_punned(metric_kind, dtype, static_capabilities, nk_cap_any_k, (nk_kernel_punned_t *)&metric,
                          &capability);
    if (!metric) {
        PyErr_Format( //
            PyExc_LookupError, "Unsupported metric '%c' and datatype combination ('%s'/'%s' and '%s'/'%s')",
            metric_kind,                                                                             //
            a_buffer.format ? a_buffer.format : "nil", datatype_to_python_string(a_parsed.datatype), //
            b_buffer.format ? b_buffer.format : "nil", datatype_to_python_string(b_parsed.datatype));
        goto cleanup;
    }

    // If the distance is computed between two vectors, rather than matrices, return a scalar
    int const dtype_is_complex = is_complex(dtype);
    if (a_parsed.rank == 1 && b_parsed.rank == 1) {
        nk_fmax_t distances[2];
        metric(a_parsed.start, b_parsed.start, a_parsed.dimensions, distances);
        return_obj =         //
            dtype_is_complex //
                ? PyComplex_FromDoubles(distances[0], distances[1])
                : PyFloat_FromDouble(distances[0]);
        goto cleanup;
    }

#if defined(__linux__)
#if defined(_OPENMP)
    if (threads == 0) threads = omp_get_num_procs();
    omp_set_num_threads(threads);
#endif
#endif

    size_t const count_pairs = a_parsed.count * b_parsed.count;
    size_t const components_per_pair = dtype_is_complex ? 2 : 1;
    size_t const count_components = count_pairs * components_per_pair;
    char *distances_start = NULL;
    size_t distances_rows_stride_bytes = 0;
    size_t distances_cols_stride_bytes = 0;

    // Allocate the output matrix if it wasn't provided
    if (!out_obj) {

        NDArray *distances_obj = PyObject_NewVar(NDArray, &NDArrayType,
                                                 count_components * bytes_per_datatype(out_dtype));
        if (!distances_obj) {
            PyErr_NoMemory();
            goto cleanup;
        }

        // Initialize the object
        distances_obj->datatype = out_dtype;
        distances_obj->rank = 2;
        distances_obj->shape[0] = a_parsed.count;
        distances_obj->shape[1] = b_parsed.count;
        distances_obj->strides[0] = b_parsed.count * bytes_per_datatype(distances_obj->datatype);
        distances_obj->strides[1] = bytes_per_datatype(distances_obj->datatype);
        distances_obj->parent = NULL;
        distances_obj->data = distances_obj->start;
        return_obj = (PyObject *)distances_obj;
        distances_start = distances_obj->data;
        distances_rows_stride_bytes = distances_obj->strides[0];
        distances_cols_stride_bytes = distances_obj->strides[1];
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
        distances_start = (char *)&out_parsed.start[0];
        distances_rows_stride_bytes = out_buffer.strides[0];
        distances_cols_stride_bytes = out_buffer.strides[1];
        //? Logic suggests to return `None` in in-place mode...
        //? SciPy decided differently.
        return_obj = Py_None;
    }

    // Now let's release the GIL for the parallel part using the underlying mechanism of `Py_BEGIN_ALLOW_THREADS`.
    PyThreadState *save = PyEval_SaveThread();

    // Assuming most of our kernels are symmetric, we only need to compute the upper triangle
    // if we are computing all pairwise distances within the same set.
    int const is_symmetric = kernel_is_commutative(metric_kind) && a_parsed.start == b_parsed.start &&
                             a_parsed.stride == b_parsed.stride && a_parsed.count == b_parsed.count;
#pragma omp parallel for collapse(2)
    for (size_t i = 0; i < a_parsed.count; ++i)
        for (size_t j = 0; j < b_parsed.count; ++j) {
            if (is_symmetric && i > j) continue;

            // Export into an on-stack buffer and then copy to the output
            nk_fmax_t result[2];
            metric(                                   //
                a_parsed.start + i * a_parsed.stride, //
                b_parsed.start + j * b_parsed.stride, //
                a_parsed.dimensions,                  //
                (nk_fmax_t *)&result                  //
            );

            // Export into both the lower and upper triangle
            if (1)
                cast_distance(result[0], out_dtype,
                              distances_start + i * distances_rows_stride_bytes + j * distances_cols_stride_bytes, 0);
            if (dtype_is_complex)
                cast_distance(result[1], out_dtype,
                              distances_start + i * distances_rows_stride_bytes + j * distances_cols_stride_bytes, 1);
            if (is_symmetric)
                cast_distance(result[0], out_dtype,
                              distances_start + j * distances_rows_stride_bytes + i * distances_cols_stride_bytes, 0);
            if (is_symmetric && dtype_is_complex)
                cast_distance(result[1], out_dtype,
                              distances_start + j * distances_rows_stride_bytes + i * distances_cols_stride_bytes, 1);
        }

    PyEval_RestoreThread(save);

cleanup:
    PyBuffer_Release(&a_buffer);
    PyBuffer_Release(&b_buffer);
    PyBuffer_Release(&out_buffer);
    return return_obj;
}

static PyObject *implement_pointer_access(nk_kernel_kind_t metric_kind, PyObject *dtype_obj) {
    char const *dtype_name = PyUnicode_AsUTF8(dtype_obj);
    if (!dtype_name) {
        PyErr_SetString(PyExc_TypeError, "Data-type name must be a string");
        return NULL;
    }

    nk_datatype_t datatype = python_string_to_datatype(dtype_name);
    if (!datatype) { // Check the actual variable here instead of dtype_name.
        PyErr_SetString(PyExc_ValueError, "Unsupported type");
        return NULL;
    }

    nk_kernel_punned_t metric = NULL;
    nk_capability_t capability = nk_cap_serial_k;
    nk_find_kernel_punned(metric_kind, datatype, static_capabilities, nk_cap_any_k, &metric, &capability);
    if (metric == NULL) {
        PyErr_SetString(PyExc_LookupError, "No such metric");
        return NULL;
    }

    return PyLong_FromUnsignedLongLong((unsigned long long)metric);
}

static char const doc_cdist[] = //
    "Compute pairwise distances between two input sets.\n\n"
    "Parameters:\n"
    "    a (NDArray): First matrix.\n"
    "    b (NDArray): Second matrix.\n"
    "    metric (str, optional): Distance metric to use (e.g., 'sqeuclidean', 'cosine').\n"
    "    out (NDArray, optional): Output matrix to store the result.\n"
    "    dtype (Union[IntegralType, FloatType, ComplexType], optional): Override the presumed input type name.\n"
    "    out_dtype (Union[FloatType, ComplexType], optional): Result type, default is 'float64'.\n"
    "    threads (int, optional): Number of threads to use (default is 1).\n\n"
    "Returns:\n"
    "    NDArray: Pairwise distances between all inputs.\n\n"
    "Equivalent to: `scipy.spatial.distance.cdist`.\n"
    "Signature:\n"
    "    >>> def cdist(a, b, /, metric, *, dtype, out, out_dtype, threads) -> Optional[NDArray]: ...";

static PyObject *api_cdist( //
    PyObject *self, PyObject *const *args, Py_ssize_t const positional_args_count, PyObject *args_names_tuple) {

    // This function accepts up to seven arguments, more than SciPy:
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
    nk_datatype_t dtype = nk_datatype_unknown_k, out_dtype = nk_datatype_unknown_k;

    /// Same default as in SciPy:
    /// https://docs.scipy.org/doc/scipy-1.11.4/reference/generated/scipy.spatial.distance.cdist.html
    nk_kernel_kind_t metric_kind = nk_kernel_l2_k;
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

    // Convert `metric_obj` to `metric_str` and to `metric_kind`
    if (metric_obj) {
        metric_str = PyUnicode_AsUTF8(metric_obj);
        if (!metric_str && PyErr_Occurred()) {
            PyErr_SetString(PyExc_TypeError, "Expected 'metric' to be a string");
            return NULL;
        }
        metric_kind = python_string_to_metric_kind(metric_str);
        if (metric_kind == nk_kernel_unknown_k) {
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
        if (dtype == nk_datatype_unknown_k) {
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
        if (out_dtype == nk_datatype_unknown_k) {
            PyErr_SetString(PyExc_ValueError, "Unsupported 'out_dtype'");
            return NULL;
        }
    }

    return implement_cdist(a_obj, b_obj, out_obj, metric_kind, threads, dtype, out_dtype);
}

static char const doc_l2_pointer[] = "Return an integer pointer to the `numkong.l2` kernel.";
static PyObject *api_l2_pointer(PyObject *self, PyObject *dtype_obj) {
    return implement_pointer_access(nk_kernel_l2_k, dtype_obj);
}
static char const doc_l2sq_pointer[] = "Return an integer pointer to the `numkong.l2sq` kernel.";
static PyObject *api_l2sq_pointer(PyObject *self, PyObject *dtype_obj) {
    return implement_pointer_access(nk_kernel_l2sq_k, dtype_obj);
}
static char const doc_angular_pointer[] = "Return an integer pointer to the `numkong.angular` kernel.";
static PyObject *api_angular_pointer(PyObject *self, PyObject *dtype_obj) {
    return implement_pointer_access(nk_kernel_angular_k, dtype_obj);
}
static char const doc_dot_pointer[] = "Return an integer pointer to the `numkong.dot` kernel.";
static PyObject *api_dot_pointer(PyObject *self, PyObject *dtype_obj) {
    return implement_pointer_access(nk_kernel_dot_k, dtype_obj);
}
static char const doc_vdot_pointer[] = "Return an integer pointer to the `numkong.vdot` kernel.";
static PyObject *api_vdot_pointer(PyObject *self, PyObject *dtype_obj) {
    return implement_pointer_access(nk_kernel_vdot_k, dtype_obj);
}
static char const doc_kld_pointer[] = "Return an integer pointer to the `numkong.kld` kernel.";
static PyObject *api_kld_pointer(PyObject *self, PyObject *dtype_obj) {
    return implement_pointer_access(nk_kernel_kld_k, dtype_obj);
}
static char const doc_jsd_pointer[] = "Return an integer pointer to the `numkong.jsd` kernel.";
static PyObject *api_jsd_pointer(PyObject *self, PyObject *dtype_obj) {
    return implement_pointer_access(nk_kernel_jsd_k, dtype_obj);
}
static char const doc_hamming_pointer[] = "Return an integer pointer to the `numkong.hamming` kernel.";
static PyObject *api_hamming_pointer(PyObject *self, PyObject *dtype_obj) {
    return implement_pointer_access(nk_kernel_hamming_k, dtype_obj);
}
static char const doc_jaccard_pointer[] = "Return an integer pointer to the `numkong.jaccard` kernel.";
static PyObject *api_jaccard_pointer(PyObject *self, PyObject *dtype_obj) {
    return implement_pointer_access(nk_kernel_jaccard_k, dtype_obj);
}

static char const doc_l2[] = //
    "Compute Euclidean (L2) distances between two matrices.\n\n"
    "Parameters:\n"
    "    a (NDArray): First matrix or vector.\n"
    "    b (NDArray): Second matrix or vector.\n"
    "    dtype (Union[IntegralType, FloatType], optional): Override the presumed input type name.\n"
    "    out (NDArray, optional): Vector for resulting distances. Allocates a new tensor by default.\n"
    "    out_dtype (FloatType, optional): Result type, default is 'float64'.\n\n"
    "Returns:\n"
    "    NDArray: The distances if `out` is not provided.\n"
    "    None: If `out` is provided. Operation will be performed in-place.\n\n"
    "Equivalent to: `scipy.spatial.distance.euclidean`.\n"
    "Signature:\n"
    "    >>> def euclidean(a, b, /, dtype, *, out, out_dtype) -> Optional[NDArray]: ...";

static PyObject *api_l2(PyObject *self, PyObject *const *args, Py_ssize_t const positional_args_count,
                        PyObject *args_names_tuple) {
    return implement_dense_metric(nk_kernel_l2_k, args, positional_args_count, args_names_tuple);
}

static char const doc_l2sq[] = //
    "Compute squared Euclidean (L2) distances between two matrices.\n\n"
    "Parameters:\n"
    "    a (NDArray): First matrix or vector.\n"
    "    b (NDArray): Second matrix or vector.\n"
    "    dtype (Union[IntegralType, FloatType], optional): Override the presumed input type name.\n"
    "    out (NDArray, optional): Vector for resulting distances. Allocates a new tensor by default.\n"
    "    out_dtype (FloatType, optional): Result type, default is 'float64'.\n\n"
    "Returns:\n"
    "    NDArray: The distances if `out` is not provided.\n"
    "    None: If `out` is provided. Operation will be performed in-place.\n\n"
    "Equivalent to: `scipy.spatial.distance.sqeuclidean`.\n"
    "Signature:\n"
    "    >>> def sqeuclidean(a, b, /, dtype, *, out, out_dtype) -> Optional[NDArray]: ...";

static PyObject *api_l2sq(PyObject *self, PyObject *const *args, Py_ssize_t const positional_args_count,
                          PyObject *args_names_tuple) {
    return implement_dense_metric(nk_kernel_l2sq_k, args, positional_args_count, args_names_tuple);
}

static char const doc_angular[] = //
    "Compute angular distances between two matrices.\n\n"
    "Parameters:\n"
    "    a (NDArray): First matrix or vector.\n"
    "    b (NDArray): Second matrix or vector.\n"
    "    dtype (Union[IntegralType, FloatType], optional): Override the presumed input type name.\n"
    "    out (NDArray, optional): Vector for resulting distances. Allocates a new tensor by default.\n"
    "    out_dtype (FloatType, optional): Result type, default is 'float64'.\n\n"
    "Returns:\n"
    "    NDArray: The distances if `out` is not provided.\n"
    "    None: If `out` is provided. Operation will be performed in-place.\n\n"
    "Equivalent to: `scipy.spatial.distance.cosine`.\n"
    "Signature:\n"
    "    >>> def angular(a, b, /, dtype, *, out, out_dtype) -> Optional[NDArray]: ...";

static PyObject *api_angular(PyObject *self, PyObject *const *args, Py_ssize_t const positional_args_count,
                             PyObject *args_names_tuple) {
    return implement_dense_metric(nk_kernel_angular_k, args, positional_args_count, args_names_tuple);
}

static char const doc_dot[] = //
    "Compute the inner (dot) product between two matrices (real or complex).\n\n"
    "Parameters:\n"
    "    a (NDArray): First matrix or vector.\n"
    "    b (NDArray): Second matrix or vector.\n"
    "    dtype (Union[IntegralType, FloatType, ComplexType], optional): Override the presumed input type name.\n"
    "    out (NDArray, optional): Vector for resulting distances. Allocates a new tensor by default.\n"
    "    out_dtype (Union[FloatType, ComplexType], optional): Result type, default is 'float64'.\n\n"
    "Returns:\n"
    "    NDArray: The distances if `out` is not provided.\n"
    "    None: If `out` is provided. Operation will be performed in-place.\n\n"
    "Equivalent to: `numpy.inner`.\n"
    "Signature:\n"
    "    >>> def dot(a, b, /, dtype, *, out, out_dtype) -> Optional[NDArray]: ...";

static PyObject *api_dot(PyObject *self, PyObject *const *args, Py_ssize_t const positional_args_count,
                         PyObject *args_names_tuple) {
    return implement_dense_metric(nk_kernel_dot_k, args, positional_args_count, args_names_tuple);
}

static char const doc_vdot[] = //
    "Compute the conjugate dot product between two complex matrices.\n\n"
    "Parameters:\n"
    "    a (NDArray): First complex matrix or vector.\n"
    "    b (NDArray): Second complex matrix or vector.\n"
    "    dtype (ComplexType, optional): Override the presumed input type name.\n"
    "    out (NDArray, optional): Vector for resulting distances. Allocates a new tensor by default.\n"
    "    out_dtype (Union[ComplexType], optional): Result type, default is 'float64'.\n\n"
    "Returns:\n"
    "    NDArray: The distances if `out` is not provided.\n"
    "    None: If `out` is provided. Operation will be performed in-place.\n\n"
    "Equivalent to: `numpy.vdot`.\n"
    "Signature:\n"
    "    >>> def vdot(a, b, /, dtype, *, out, out_dtype) -> Optional[NDArray]: ...";

static PyObject *api_vdot(PyObject *self, PyObject *const *args, Py_ssize_t const positional_args_count,
                          PyObject *args_names_tuple) {
    return implement_dense_metric(nk_kernel_vdot_k, args, positional_args_count, args_names_tuple);
}

static char const doc_kld[] = //
    "Compute Kullback-Leibler divergences between two matrices.\n\n"
    "Parameters:\n"
    "    a (NDArray): First floating-point matrix or vector.\n"
    "    b (NDArray): Second floating-point matrix or vector.\n"
    "    dtype (FloatType, optional): Override the presumed input type name.\n"
    "    out (NDArray, optional): Vector for resulting distances. Allocates a new tensor by default.\n"
    "    out_dtype (FloatType, optional): Result type, default is 'float64'.\n\n"
    "Returns:\n"
    "    NDArray: The distances if `out` is not provided.\n"
    "    None: If `out` is provided. Operation will be performed in-place.\n\n"
    "Equivalent to: `scipy.special.kl_div`.\n"
    "Signature:\n"
    "    >>> def kld(a, b, /, dtype, *, out, out_dtype) -> Optional[NDArray]: ...";

static PyObject *api_kld(PyObject *self, PyObject *const *args, Py_ssize_t const positional_args_count,
                         PyObject *args_names_tuple) {
    return implement_dense_metric(nk_kernel_kld_k, args, positional_args_count, args_names_tuple);
}

static char const doc_jsd[] = //
    "Compute Jensen-Shannon divergences between two matrices.\n\n"
    "Parameters:\n"
    "    a (NDArray): First floating-point matrix or vector.\n"
    "    b (NDArray): Second floating-point matrix or vector.\n"
    "    dtype (Union[IntegralType, FloatType], optional): Override the presumed input type name.\n"
    "    out (NDArray, optional): Vector for resulting distances. Allocates a new tensor by default.\n"
    "    out_dtype (FloatType, optional): Result type, default is 'float64'.\n\n"
    "Returns:\n"
    "    NDArray: The distances if `out` is not provided.\n"
    "    None: If `out` is provided. Operation will be performed in-place.\n\n"
    "Equivalent to: `scipy.spatial.distance.jensenshannon`.\n"
    "Signature:\n"
    "    >>> def jsd(a, b, /, dtype, *, out, out_dtype) -> Optional[NDArray]: ...";

static PyObject *api_jsd(PyObject *self, PyObject *const *args, Py_ssize_t const positional_args_count,
                         PyObject *args_names_tuple) {
    return implement_dense_metric(nk_kernel_jsd_k, args, positional_args_count, args_names_tuple);
}

static char const doc_hamming[] = //
    "Compute Hamming distances between two matrices.\n\n"
    "Parameters:\n"
    "    a (NDArray): First binary matrix or vector.\n"
    "    b (NDArray): Second binary matrix or vector.\n"
    "    dtype (IntegralType, optional): Override the presumed input type name.\n"
    "    out (NDArray, optional): Vector for resulting distances. Allocates a new tensor by default.\n"
    "    out_dtype (FloatType, optional): Result type, default is 'float64'.\n\n"
    "Returns:\n"
    "    NDArray: The distances if `out` is not provided.\n"
    "    None: If `out` is provided. Operation will be performed in-place.\n\n"
    "Similar to: `scipy.spatial.distance.hamming`.\n"
    "Signature:\n"
    "    >>> def hamming(a, b, /, dtype, *, out, out_dtype) -> Optional[NDArray]: ...";

static PyObject *api_hamming(PyObject *self, PyObject *const *args, Py_ssize_t const positional_args_count,
                             PyObject *args_names_tuple) {
    return implement_dense_metric(nk_kernel_hamming_k, args, positional_args_count, args_names_tuple);
}

static char const doc_jaccard[] = //
    "Compute Jaccard distances (bitwise Tanimoto) between two matrices.\n\n"
    "Parameters:\n"
    "    a (NDArray): First binary matrix or vector.\n"
    "    b (NDArray): Second binary matrix or vector.\n"
    "    dtype (IntegralType, optional): Override the presumed input type name.\n"
    "    out (NDArray, optional): Vector for resulting distances. Allocates a new tensor by default.\n"
    "    out_dtype (FloatType, optional): Result type, default is 'float64'.\n\n"
    "Returns:\n"
    "    NDArray: The distances if `out` is not provided.\n"
    "    None: If `out` is provided. Operation will be performed in-place.\n\n"
    "Similar to: `scipy.spatial.distance.jaccard`.\n"
    "Signature:\n"
    "    >>> def jaccard(a, b, /, dtype, *, out, out_dtype) -> Optional[NDArray]: ...";

static PyObject *api_jaccard(PyObject *self, PyObject *const *args, Py_ssize_t const positional_args_count,
                             PyObject *args_names_tuple) {
    return implement_dense_metric(nk_kernel_jaccard_k, args, positional_args_count, args_names_tuple);
}

static char const doc_bilinear[] = //
    "Compute the bilinear form between two vectors given a metric tensor.\n\n"
    "Parameters:\n"
    "    a (NDArray): First vector.\n"
    "    b (NDArray): Second vector.\n"
    "    metric_tensor (NDArray): The metric tensor defining the bilinear form.\n"
    "    dtype (FloatType, optional): Override the presumed input type name.\n\n"
    "Returns:\n"
    "    float: The bilinear form.\n\n"
    "Equivalent to: `numpy.dot` with a metric tensor.\n"
    "Signature:\n"
    "    >>> def bilinear(a, b, metric_tensor, /, dtype) -> float: ...";

static PyObject *api_bilinear(PyObject *self, PyObject *const *args, Py_ssize_t const positional_args_count,
                              PyObject *args_names_tuple) {
    return implement_curved_metric(nk_kernel_bilinear_k, args, positional_args_count, args_names_tuple);
}

static char const doc_mahalanobis[] = //
    "Compute the Mahalanobis distance between two vectors given an inverse covariance matrix.\n\n"
    "Parameters:\n"
    "    a (NDArray): First vector.\n"
    "    b (NDArray): Second vector.\n"
    "    inverse_covariance (NDArray): The inverse of the covariance matrix.\n"
    "    dtype (FloatType, optional): Override the presumed input type name.\n\n"
    "Returns:\n"
    "    float: The Mahalanobis distance.\n\n"
    "Equivalent to: `scipy.spatial.distance.mahalanobis`.\n"
    "Signature:\n"
    "    >>> def mahalanobis(a, b, inverse_covariance, /, dtype) -> float: ...";

static PyObject *api_mahalanobis(PyObject *self, PyObject *const *args, Py_ssize_t const positional_args_count,
                                 PyObject *args_names_tuple) {
    return implement_curved_metric(nk_kernel_mahalanobis_k, args, positional_args_count, args_names_tuple);
}

static char const doc_haversine[] = //
    "Compute the Haversine (great-circle) distance between coordinate pairs.\n\n"
    "Parameters:\n"
    "    a_lats (NDArray): Latitudes of first points in radians.\n"
    "    a_lons (NDArray): Longitudes of first points in radians.\n"
    "    b_lats (NDArray): Latitudes of second points in radians.\n"
    "    b_lons (NDArray): Longitudes of second points in radians.\n"
    "    dtype (FloatType, optional): Override the presumed input type name.\n"
    "    out (NDArray, optional): Pre-allocated output array for distances.\n\n"
    "Returns:\n"
    "    NDArray: Distances in meters (using mean Earth radius).\n"
    "    None: If `out` is provided.\n\n"
    "Note: Input coordinates must be in radians. Uses spherical Earth model.\n"
    "Signature:\n"
    "    >>> def haversine(a_lats, a_lons, b_lats, b_lons, /, dtype, *, out) -> Optional[NDArray]: ...";

static PyObject *api_haversine(PyObject *self, PyObject *const *args, Py_ssize_t const positional_args_count,
                               PyObject *args_names_tuple) {
    return implement_geospatial_metric(nk_kernel_haversine_k, args, positional_args_count, args_names_tuple);
}

static char const doc_vincenty[] = //
    "Compute the Vincenty (ellipsoidal geodesic) distance between coordinate pairs.\n\n"
    "Parameters:\n"
    "    a_lats (NDArray): Latitudes of first points in radians.\n"
    "    a_lons (NDArray): Longitudes of first points in radians.\n"
    "    b_lats (NDArray): Latitudes of second points in radians.\n"
    "    b_lons (NDArray): Longitudes of second points in radians.\n"
    "    dtype (FloatType, optional): Override the presumed input type name.\n"
    "    out (NDArray, optional): Pre-allocated output array for distances.\n\n"
    "Returns:\n"
    "    NDArray: Distances in meters (using WGS84 ellipsoid).\n"
    "    None: If `out` is provided.\n\n"
    "Note: Input coordinates must be in radians. Uses iterative algorithm for accuracy.\n"
    "Signature:\n"
    "    >>> def vincenty(a_lats, a_lons, b_lats, b_lons, /, dtype, *, out) -> Optional[NDArray]: ...";

static PyObject *api_vincenty(PyObject *self, PyObject *const *args, Py_ssize_t const positional_args_count,
                              PyObject *args_names_tuple) {
    return implement_geospatial_metric(nk_kernel_vincenty_k, args, positional_args_count, args_names_tuple);
}

static char const doc_intersect[] = //
    "Compute the intersection of two sorted integer arrays.\n\n"
    "Parameters:\n"
    "    a (NDArray): First sorted integer array.\n"
    "    b (NDArray): Second sorted integer array.\n\n"
    "Returns:\n"
    "    float: The number of intersecting elements.\n\n"
    "Similar to: `numpy.intersect1d`."
    "Signature:\n"
    "    >>> def intersect(a, b, /) -> float: ...";

static PyObject *api_intersect(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    return implement_sparse_metric(nk_kernel_sparse_intersect_k, args, nargs);
}

static char const doc_fma[] = //
    "Fused-Multiply-Add between 3 input vectors.\n\n"
    "Parameters:\n"
    "    a (NDArray): First vector.\n"
    "    b (NDArray): Second vector.\n"
    "    c (NDArray): Third vector.\n"
    "    dtype (Union[IntegralType, FloatType], optional): Override the presumed numeric type name.\n"
    "    alpha (float, optional): First scale, 1.0 by default.\n"
    "    beta (float, optional): Second scale, 1.0 by default.\n"
    "    out (NDArray, optional): Vector for resulting distances.\n\n"
    "Returns:\n"
    "    NDArray: The distances if `out` is not provided.\n"
    "    None: If `out` is provided. Operation will be performed in-place.\n\n"
    "Equivalent to: `alpha * a * b + beta * c`.\n"
    "Signature:\n"
    "    >>> def fma(a, b, c, /, dtype, *, alpha, beta, out) -> Optional[NDArray]: ...";

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
    nk_datatype_t dtype = nk_datatype_unknown_k;
    nk_fmax_t alpha = 1, beta = 1;

    Py_buffer a_buffer, b_buffer, c_buffer, out_buffer;
    TensorArgument a_parsed, b_parsed, c_parsed, out_parsed;
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
        if (dtype == nk_datatype_unknown_k) {
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
    if (!parse_tensor(a_obj, &a_buffer, &a_parsed) || !parse_tensor(b_obj, &b_buffer, &b_parsed) ||
        !parse_tensor(c_obj, &c_buffer, &c_parsed))
        return NULL;
    if (out_obj && !parse_tensor(out_obj, &out_buffer, &out_parsed)) return NULL;

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
    if (a_parsed.datatype != b_parsed.datatype || a_parsed.datatype == nk_datatype_unknown_k ||
        b_parsed.datatype == nk_datatype_unknown_k || c_parsed.datatype == nk_datatype_unknown_k ||
        (out_obj && out_parsed.datatype == nk_datatype_unknown_k)) {
        PyErr_SetString(PyExc_TypeError,
                        "Input tensors must have matching datatypes, check with `X.__array_interface__`");
        goto cleanup;
    }
    if (dtype == nk_datatype_unknown_k) dtype = a_parsed.datatype;

    // Look up the metric and the capability
    nk_kernel_fma_punned_t metric = NULL;
    nk_capability_t capability = nk_cap_serial_k;
    nk_kernel_kind_t const metric_kind = nk_kernel_fma_k;
    nk_find_kernel_punned(metric_kind, dtype, static_capabilities, nk_cap_any_k, (nk_kernel_punned_t *)&metric,
                          &capability);
    if (!metric) {
        PyErr_Format( //
            PyExc_LookupError,
            "Unsupported metric '%c' and datatype combination across vectors ('%s'/'%s') and "
            "`dtype` override ('%s'/'%s')",
            metric_kind,                                                                             //
            a_buffer.format ? a_buffer.format : "nil", datatype_to_python_string(a_parsed.datatype), //
            dtype_str ? dtype_str : "nil", datatype_to_python_string(dtype));
        goto cleanup;
    }

    char *distances_start = NULL;
    size_t distances_stride_bytes = 0;

    // Allocate the output matrix if it wasn't provided
    if (!out_obj) {
        NDArray *distances_obj = PyObject_NewVar(NDArray, &NDArrayType,
                                                 a_parsed.dimensions * bytes_per_datatype(dtype));
        if (!distances_obj) {
            PyErr_NoMemory();
            goto cleanup;
        }

        // Initialize the object
        distances_obj->datatype = dtype;
        distances_obj->rank = 1;
        distances_obj->shape[0] = a_parsed.dimensions;
        distances_obj->shape[1] = 0;
        distances_obj->strides[0] = bytes_per_datatype(dtype);
        distances_obj->strides[1] = 0;
        distances_obj->parent = NULL;
        distances_obj->data = distances_obj->start;
        return_obj = (PyObject *)distances_obj;
        distances_start = distances_obj->data;
        distances_stride_bytes = distances_obj->strides[0];
    }
    else {
        distances_start = out_parsed.start;
        distances_stride_bytes = out_buffer.strides[0];
        //? Logic suggests to return `None` in in-place mode...
        //? SciPy decided differently.
        return_obj = Py_None;
    }

    nk_scalar_buffer_t alpha_buf, beta_buf;
    nk_scalar_buffer_set_f64(&alpha_buf, alpha, dtype);
    nk_scalar_buffer_set_f64(&beta_buf, beta, dtype);
    metric(a_parsed.start, b_parsed.start, c_parsed.start, a_parsed.dimensions, &alpha_buf, &beta_buf, distances_start);
cleanup:
    PyBuffer_Release(&a_buffer);
    PyBuffer_Release(&b_buffer);
    PyBuffer_Release(&c_buffer);
    PyBuffer_Release(&out_buffer);
    return return_obj;
}

static char const doc_wsum[] = //
    "Weighted Sum of 2 input vectors.\n\n"
    "Parameters:\n"
    "    a (NDArray): First vector.\n"
    "    b (NDArray): Second vector.\n"
    "    dtype (Union[IntegralType, FloatType], optional): Override the presumed numeric type name.\n"
    "    alpha (float, optional): First scale, 1.0 by default.\n"
    "    beta (float, optional): Second scale, 1.0 by default.\n"
    "    out (NDArray, optional): Vector for resulting distances.\n\n"
    "Returns:\n"
    "    NDArray: The distances if `out` is not provided.\n"
    "    None: If `out` is provided. Operation will be performed in-place.\n\n"
    "Equivalent to: `alpha * a + beta * b`.\n"
    "Signature:\n"
    "    >>> def wsum(a, b, /, dtype, *, alpha, beta, out) -> Optional[NDArray]: ...";

static PyObject *api_wsum(PyObject *self, PyObject *const *args, Py_ssize_t const positional_args_count,
                          PyObject *args_names_tuple) {

    PyObject *return_obj = NULL;

    // This function accepts up to 5 arguments:
    PyObject *a_obj = NULL;     // Required object, positional-only
    PyObject *b_obj = NULL;     // Required object, positional-only
    PyObject *dtype_obj = NULL; // Optional object, "dtype" keyword or positional
    PyObject *out_obj = NULL;   // Optional object, "out" keyword-only
    PyObject *alpha_obj = NULL; // Optional object, "alpha" keyword-only
    PyObject *beta_obj = NULL;  // Optional object, "beta" keyword-only

    // Once parsed, the arguments will be stored in these variables:
    char const *dtype_str = NULL;
    nk_datatype_t dtype = nk_datatype_unknown_k;
    nk_fmax_t alpha = 1, beta = 1;

    Py_buffer a_buffer, b_buffer, out_buffer;
    TensorArgument a_parsed, b_parsed, out_parsed;
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
        if (dtype == nk_datatype_unknown_k) {
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
    if (!parse_tensor(a_obj, &a_buffer, &a_parsed) || !parse_tensor(b_obj, &b_buffer, &b_parsed)) return NULL;
    if (out_obj && !parse_tensor(out_obj, &out_buffer, &out_parsed)) return NULL;

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
    if (a_parsed.datatype != b_parsed.datatype || a_parsed.datatype == nk_datatype_unknown_k ||
        b_parsed.datatype == nk_datatype_unknown_k || (out_obj && out_parsed.datatype == nk_datatype_unknown_k)) {
        PyErr_SetString(PyExc_TypeError,
                        "Input tensors must have matching datatypes, check with `X.__array_interface__`");
        goto cleanup;
    }
    if (dtype == nk_datatype_unknown_k) dtype = a_parsed.datatype;

    // Look up the metric and the capability
    nk_kernel_wsum_punned_t metric = NULL;
    nk_capability_t capability = nk_cap_serial_k;
    nk_kernel_kind_t const metric_kind = nk_kernel_wsum_k;
    nk_find_kernel_punned(metric_kind, dtype, static_capabilities, nk_cap_any_k, (nk_kernel_punned_t *)&metric,
                          &capability);
    if (!metric) {
        PyErr_Format( //
            PyExc_LookupError,
            "Unsupported metric '%c' and datatype combination across vectors ('%s'/'%s') and "
            "`dtype` override ('%s'/'%s')",
            metric_kind,                                                                             //
            a_buffer.format ? a_buffer.format : "nil", datatype_to_python_string(a_parsed.datatype), //
            dtype_str ? dtype_str : "nil", datatype_to_python_string(dtype));
        goto cleanup;
    }

    char *distances_start = NULL;
    size_t distances_stride_bytes = 0;

    // Allocate the output matrix if it wasn't provided
    if (!out_obj) {
        NDArray *distances_obj = PyObject_NewVar(NDArray, &NDArrayType,
                                                 a_parsed.dimensions * bytes_per_datatype(dtype));
        if (!distances_obj) {
            PyErr_NoMemory();
            goto cleanup;
        }

        // Initialize the object
        distances_obj->datatype = dtype;
        distances_obj->rank = 1;
        distances_obj->shape[0] = a_parsed.dimensions;
        distances_obj->shape[1] = 0;
        distances_obj->strides[0] = bytes_per_datatype(dtype);
        distances_obj->strides[1] = 0;
        distances_obj->parent = NULL;
        distances_obj->data = distances_obj->start;
        return_obj = (PyObject *)distances_obj;
        distances_start = distances_obj->data;
        distances_stride_bytes = distances_obj->strides[0];
    }
    else {
        distances_start = out_parsed.start;
        distances_stride_bytes = out_buffer.strides[0];
        //? Logic suggests to return `None` in in-place mode...
        //? SciPy decided differently.
        return_obj = Py_None;
    }

    nk_scalar_buffer_t alpha_buf, beta_buf;
    nk_scalar_buffer_set_f64(&alpha_buf, alpha, dtype);
    nk_scalar_buffer_set_f64(&beta_buf, beta, dtype);
    metric(a_parsed.start, b_parsed.start, a_parsed.dimensions, &alpha_buf, &beta_buf, distances_start);
cleanup:
    PyBuffer_Release(&a_buffer);
    PyBuffer_Release(&b_buffer);
    PyBuffer_Release(&out_buffer);
    return return_obj;
}

// region NDArray Constructors

/// @brief  Parse a shape argument (int or tuple of ints) into a C array.
static int parse_shape(PyObject *shape_obj, Py_ssize_t *shape, size_t *rank) {
    if (PyLong_Check(shape_obj)) {
        Py_ssize_t size = PyLong_AsSsize_t(shape_obj);
        if (size < 0) {
            PyErr_SetString(PyExc_ValueError, "Shape dimensions must be non-negative");
            return 0;
        }
        shape[0] = size;
        *rank = 1;
        return 1;
    }
    if (!PyTuple_Check(shape_obj)) {
        PyErr_SetString(PyExc_TypeError, "Shape must be an int or tuple of ints");
        return 0;
    }
    Py_ssize_t ndim = PyTuple_Size(shape_obj);
    if (ndim > NK_NDARRAY_MAX_RANK) {
        PyErr_Format(PyExc_ValueError, "Shape has %zd dimensions, max is %d", ndim, NK_NDARRAY_MAX_RANK);
        return 0;
    }
    for (Py_ssize_t i = 0; i < ndim; i++) {
        PyObject *dim = PyTuple_GetItem(shape_obj, i);
        if (!PyLong_Check(dim)) {
            PyErr_SetString(PyExc_TypeError, "Shape dimensions must be integers");
            return 0;
        }
        Py_ssize_t size = PyLong_AsSsize_t(dim);
        if (size < 0) {
            PyErr_SetString(PyExc_ValueError, "Shape dimensions must be non-negative");
            return 0;
        }
        shape[i] = size;
    }
    *rank = (size_t)ndim;
    return 1;
}

static char const doc_empty[] =
    "Create an uninitialized NDArray with the given shape.\n\nParameters:\n    shape: Shape of the array.\n    dtype: "
    "Data type (default 'float32').\n\nReturns:\n    NDArray: Uninitialized array.";

static PyObject *api_empty(PyObject *self, PyObject *const *args, Py_ssize_t const nargs, PyObject *kwnames) {
    (void)self;
    PyObject *shape_obj = NULL, *dtype_obj = NULL;
    Py_ssize_t nkw = kwnames ? PyTuple_Size(kwnames) : 0;
    if (nargs + nkw < 1 || nargs + nkw > 2 || nargs > 1) {
        PyErr_SetString(PyExc_TypeError, "empty(shape, *, dtype='float32')");
        return NULL;
    }
    shape_obj = args[0];
    for (Py_ssize_t i = 0; i < nkw; i++) {
        PyObject *key = PyTuple_GetItem(kwnames, i);
        if (PyUnicode_CompareWithASCIIString(key, "dtype") == 0) dtype_obj = args[nargs + i];
        else {
            PyErr_Format(PyExc_TypeError, "empty() unexpected keyword: %S", key);
            return NULL;
        }
    }
    Py_ssize_t shape[NK_NDARRAY_MAX_RANK];
    size_t rank;
    if (!parse_shape(shape_obj, shape, &rank)) return NULL;
    nk_datatype_t dtype = nk_f32_k;
    if (dtype_obj) {
        char const *s = PyUnicode_AsUTF8(dtype_obj);
        if (!s) return NULL;
        dtype = python_string_to_datatype(s);
        if (dtype == nk_datatype_unknown_k) {
            PyErr_Format(PyExc_ValueError, "Unknown dtype: %s", s);
            return NULL;
        }
    }
    return (PyObject *)NDArray_new(dtype, rank, shape);
}

static char const doc_zeros[] = "Create an NDArray filled with zeros.\n\nParameters:\n    shape: Shape of the array.\n "
                                "   dtype: Data type (default 'float32').\n\nReturns:\n    NDArray: Array of zeros.";

static PyObject *api_zeros(PyObject *self, PyObject *const *args, Py_ssize_t const nargs, PyObject *kwnames) {
    (void)self;
    PyObject *shape_obj = NULL, *dtype_obj = NULL;
    Py_ssize_t nkw = kwnames ? PyTuple_Size(kwnames) : 0;
    if (nargs + nkw < 1 || nargs + nkw > 2 || nargs > 1) {
        PyErr_SetString(PyExc_TypeError, "zeros(shape, *, dtype='float32')");
        return NULL;
    }
    shape_obj = args[0];
    for (Py_ssize_t i = 0; i < nkw; i++) {
        PyObject *key = PyTuple_GetItem(kwnames, i);
        if (PyUnicode_CompareWithASCIIString(key, "dtype") == 0) dtype_obj = args[nargs + i];
        else {
            PyErr_Format(PyExc_TypeError, "zeros() unexpected keyword: %S", key);
            return NULL;
        }
    }
    Py_ssize_t shape[NK_NDARRAY_MAX_RANK];
    size_t rank;
    if (!parse_shape(shape_obj, shape, &rank)) return NULL;
    nk_datatype_t dtype = nk_f32_k;
    if (dtype_obj) {
        char const *s = PyUnicode_AsUTF8(dtype_obj);
        if (!s) return NULL;
        dtype = python_string_to_datatype(s);
        if (dtype == nk_datatype_unknown_k) {
            PyErr_Format(PyExc_ValueError, "Unknown dtype: %s", s);
            return NULL;
        }
    }
    NDArray *result = NDArray_new(dtype, rank, shape);
    if (!result) return NULL;
    size_t nbytes = bytes_per_datatype(dtype);
    for (size_t i = 0; i < rank; i++) nbytes *= (size_t)shape[i];
    memset(result->data, 0, nbytes);
    return (PyObject *)result;
}

static char const doc_ones[] = "Create an NDArray filled with ones.\n\nParameters:\n    shape: Shape of the array.\n   "
                               " dtype: Data type (default 'float32').\n\nReturns:\n    NDArray: Array of ones.";

static PyObject *api_ones(PyObject *self, PyObject *const *args, Py_ssize_t const nargs, PyObject *kwnames) {
    (void)self;
    PyObject *shape_obj = NULL, *dtype_obj = NULL;
    Py_ssize_t nkw = kwnames ? PyTuple_Size(kwnames) : 0;
    if (nargs + nkw < 1 || nargs + nkw > 2 || nargs > 1) {
        PyErr_SetString(PyExc_TypeError, "ones(shape, *, dtype='float32')");
        return NULL;
    }
    shape_obj = args[0];
    for (Py_ssize_t i = 0; i < nkw; i++) {
        PyObject *key = PyTuple_GetItem(kwnames, i);
        if (PyUnicode_CompareWithASCIIString(key, "dtype") == 0) dtype_obj = args[nargs + i];
        else {
            PyErr_Format(PyExc_TypeError, "ones() unexpected keyword: %S", key);
            return NULL;
        }
    }
    Py_ssize_t shape[NK_NDARRAY_MAX_RANK];
    size_t rank;
    if (!parse_shape(shape_obj, shape, &rank)) return NULL;
    nk_datatype_t dtype = nk_f32_k;
    if (dtype_obj) {
        char const *s = PyUnicode_AsUTF8(dtype_obj);
        if (!s) return NULL;
        dtype = python_string_to_datatype(s);
        if (dtype == nk_datatype_unknown_k) {
            PyErr_Format(PyExc_ValueError, "Unknown dtype: %s", s);
            return NULL;
        }
    }
    NDArray *result = NDArray_new(dtype, rank, shape);
    if (!result) return NULL;
    size_t total = 1;
    for (size_t i = 0; i < rank; i++) total *= (size_t)shape[i];
    switch (dtype) {
    case nk_f64_k: {
        nk_f64_t *p = (nk_f64_t *)result->data;
        for (size_t i = 0; i < total; i++) p[i] = 1.0;
    } break;
    case nk_f32_k: {
        nk_f32_t *p = (nk_f32_t *)result->data;
        for (size_t i = 0; i < total; i++) p[i] = 1.0f;
    } break;
    case nk_i8_k: {
        nk_i8_t *p = (nk_i8_t *)result->data;
        for (size_t i = 0; i < total; i++) p[i] = 1;
    } break;
    case nk_i32_k: {
        nk_i32_t *p = (nk_i32_t *)result->data;
        for (size_t i = 0; i < total; i++) p[i] = 1;
    } break;
    case nk_i64_k: {
        nk_i64_t *p = (nk_i64_t *)result->data;
        for (size_t i = 0; i < total; i++) p[i] = 1;
    } break;
    default:
        Py_DECREF(result);
        PyErr_Format(PyExc_NotImplementedError, "ones() not implemented for dtype");
        return NULL;
    }
    return (PyObject *)result;
}

static char const doc_full[] =
    "Create an NDArray filled with a specified value.\n\nParameters:\n    shape: Shape of the array.\n    fill_value: "
    "Value to fill.\n    dtype: Data type (default 'float32').\n\nReturns:\n    NDArray: Filled array.";

static PyObject *api_full(PyObject *self, PyObject *const *args, Py_ssize_t const nargs, PyObject *kwnames) {
    (void)self;
    PyObject *shape_obj = NULL, *fill_obj = NULL, *dtype_obj = NULL;
    Py_ssize_t nkw = kwnames ? PyTuple_Size(kwnames) : 0;
    if (nargs + nkw < 2 || nargs + nkw > 3 || nargs > 2) {
        PyErr_SetString(PyExc_TypeError, "full(shape, fill_value, *, dtype='float32')");
        return NULL;
    }
    shape_obj = args[0];
    fill_obj = args[1];
    for (Py_ssize_t i = 0; i < nkw; i++) {
        PyObject *key = PyTuple_GetItem(kwnames, i);
        if (PyUnicode_CompareWithASCIIString(key, "dtype") == 0) dtype_obj = args[nargs + i];
        else {
            PyErr_Format(PyExc_TypeError, "full() unexpected keyword: %S", key);
            return NULL;
        }
    }
    Py_ssize_t shape[NK_NDARRAY_MAX_RANK];
    size_t rank;
    if (!parse_shape(shape_obj, shape, &rank)) return NULL;
    double fill_value;
    if (!get_scalar_value(fill_obj, &fill_value)) {
        PyErr_SetString(PyExc_TypeError, "fill_value must be a number");
        return NULL;
    }
    nk_datatype_t dtype = nk_f32_k;
    if (dtype_obj) {
        char const *s = PyUnicode_AsUTF8(dtype_obj);
        if (!s) return NULL;
        dtype = python_string_to_datatype(s);
        if (dtype == nk_datatype_unknown_k) {
            PyErr_Format(PyExc_ValueError, "Unknown dtype: %s", s);
            return NULL;
        }
    }
    NDArray *result = NDArray_new(dtype, rank, shape);
    if (!result) return NULL;
    size_t total = 1;
    for (size_t i = 0; i < rank; i++) total *= (size_t)shape[i];
    switch (dtype) {
    case nk_f64_k: {
        nk_f64_t *p = (nk_f64_t *)result->data;
        nk_f64_t v = fill_value;
        for (size_t i = 0; i < total; i++) p[i] = v;
    } break;
    case nk_f32_k: {
        nk_f32_t *p = (nk_f32_t *)result->data;
        nk_f32_t v = (nk_f32_t)fill_value;
        for (size_t i = 0; i < total; i++) p[i] = v;
    } break;
    case nk_i8_k: {
        nk_i8_t *p = (nk_i8_t *)result->data;
        nk_i8_t v = (nk_i8_t)fill_value;
        for (size_t i = 0; i < total; i++) p[i] = v;
    } break;
    case nk_i32_k: {
        nk_i32_t *p = (nk_i32_t *)result->data;
        nk_i32_t v = (nk_i32_t)fill_value;
        for (size_t i = 0; i < total; i++) p[i] = v;
    } break;
    default:
        Py_DECREF(result);
        PyErr_Format(PyExc_NotImplementedError, "full() not implemented for dtype");
        return NULL;
    }
    return (PyObject *)result;
}

// region: Module-level reduction functions

static char const doc_reduce_sum[] =
    "Sum of all elements in an array.\n\nParameters:\n    a: Input array.\n\nReturns:\n    Scalar sum of all elements.";

static PyObject *api_sum(PyObject *self, PyObject *const *args, Py_ssize_t const nargs, PyObject *kwnames) {
    (void)self;
    if (nargs != 1 || (kwnames && PyTuple_Size(kwnames) > 0)) {
        PyErr_SetString(PyExc_TypeError, "sum(a) takes exactly 1 positional argument");
        return NULL;
    }
    PyObject *a_obj = args[0];
    if (PyObject_TypeCheck(a_obj, &NDArrayType)) { return NDArray_sum(a_obj, NULL); }
    PyErr_SetString(PyExc_TypeError, "sum() argument must be an NDArray");
    return NULL;
}

static char const doc_reduce_min[] =
    "Minimum element in an array.\n\nParameters:\n    a: Input array.\n\nReturns:\n    Minimum element.";

static PyObject *api_min(PyObject *self, PyObject *const *args, Py_ssize_t const nargs, PyObject *kwnames) {
    (void)self;
    if (nargs != 1 || (kwnames && PyTuple_Size(kwnames) > 0)) {
        PyErr_SetString(PyExc_TypeError, "min(a) takes exactly 1 positional argument");
        return NULL;
    }
    PyObject *a_obj = args[0];
    if (PyObject_TypeCheck(a_obj, &NDArrayType)) { return NDArray_min(a_obj, NULL); }
    PyErr_SetString(PyExc_TypeError, "min() argument must be an NDArray");
    return NULL;
}

static char const doc_reduce_max[] =
    "Maximum element in an array.\n\nParameters:\n    a: Input array.\n\nReturns:\n    Maximum element.";

static PyObject *api_max(PyObject *self, PyObject *const *args, Py_ssize_t const nargs, PyObject *kwnames) {
    (void)self;
    if (nargs != 1 || (kwnames && PyTuple_Size(kwnames) > 0)) {
        PyErr_SetString(PyExc_TypeError, "max(a) takes exactly 1 positional argument");
        return NULL;
    }
    PyObject *a_obj = args[0];
    if (PyObject_TypeCheck(a_obj, &NDArrayType)) { return NDArray_max(a_obj, NULL); }
    PyErr_SetString(PyExc_TypeError, "max() argument must be an NDArray");
    return NULL;
}

static char const doc_reduce_argmin[] = "Index of minimum element in an array.\n\nParameters:\n    a: Input "
                                        "array.\n\nReturns:\n    int: Flat index of minimum element.";

static PyObject *api_argmin(PyObject *self, PyObject *const *args, Py_ssize_t const nargs, PyObject *kwnames) {
    (void)self;
    if (nargs != 1 || (kwnames && PyTuple_Size(kwnames) > 0)) {
        PyErr_SetString(PyExc_TypeError, "argmin(a) takes exactly 1 positional argument");
        return NULL;
    }
    PyObject *a_obj = args[0];
    if (PyObject_TypeCheck(a_obj, &NDArrayType)) { return NDArray_argmin(a_obj, NULL); }
    PyErr_SetString(PyExc_TypeError, "argmin() argument must be an NDArray");
    return NULL;
}

static char const doc_reduce_argmax[] = "Index of maximum element in an array.\n\nParameters:\n    a: Input "
                                        "array.\n\nReturns:\n    int: Flat index of maximum element.";

static PyObject *api_argmax(PyObject *self, PyObject *const *args, Py_ssize_t const nargs, PyObject *kwnames) {
    (void)self;
    if (nargs != 1 || (kwnames && PyTuple_Size(kwnames) > 0)) {
        PyErr_SetString(PyExc_TypeError, "argmax(a) takes exactly 1 positional argument");
        return NULL;
    }
    PyObject *a_obj = args[0];
    if (PyObject_TypeCheck(a_obj, &NDArrayType)) { return NDArray_argmax(a_obj, NULL); }
    PyErr_SetString(PyExc_TypeError, "argmax() argument must be an NDArray");
    return NULL;
}

// endregion

// endregion

static char const doc_add[] = //
    "Element-wise addition of two vectors or a vector and a scalar.\n\n"
    "Parameters:\n"
    "    a (Union[NDArray, float, int]): First operand (vector or scalar).\n"
    "    b (Union[NDArray, float, int]): Second operand (vector or scalar).\n"
    "    out (NDArray, optional): Output buffer for the result.\n"
    "    a_dtype (Union[IntegralType, FloatType], optional): Override dtype for `a`.\n"
    "    b_dtype (Union[IntegralType, FloatType], optional): Override dtype for `b`.\n"
    "    out_dtype (Union[IntegralType, FloatType], optional): Override dtype for output.\n\n"
    "Returns:\n"
    "    NDArray: The sum if `out` is not provided.\n"
    "    None: If `out` is provided (in-place operation).\n\n"
    "Equivalent to: `a + b`.\n"
    "Signature:\n"
    "    >>> def add(a, b, /, *, out, a_dtype, b_dtype, out_dtype) -> Optional[NDArray]: ...";

static PyObject *api_add(PyObject *self, PyObject *const *args, Py_ssize_t const positional_args_count,
                         PyObject *args_names_tuple) {
    (void)self;
    PyObject *return_obj = NULL;

    // This function accepts up to 6 arguments:
    PyObject *a_obj = NULL;         // Required, positional-only
    PyObject *b_obj = NULL;         // Required, positional-only
    PyObject *out_obj = NULL;       // Optional, "out" keyword-only
    PyObject *a_dtype_obj = NULL;   // Optional, "a_dtype" keyword-only
    PyObject *b_dtype_obj = NULL;   // Optional, "b_dtype" keyword-only
    PyObject *out_dtype_obj = NULL; // Optional, "out_dtype" keyword-only

    nk_datatype_t dtype = nk_datatype_unknown_k;

    Py_buffer a_buffer, b_buffer, out_buffer;
    TensorArgument a_parsed, b_parsed, out_parsed;
    memset(&a_buffer, 0, sizeof(Py_buffer));
    memset(&b_buffer, 0, sizeof(Py_buffer));
    memset(&out_buffer, 0, sizeof(Py_buffer));

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

    // Positional-only arguments
    a_obj = args[0];
    b_obj = args[1];

    // Parse keyword arguments
    for (Py_ssize_t args_names_tuple_progress = 0, args_progress = positional_args_count;
         args_names_tuple_progress < args_names_count; ++args_progress, ++args_names_tuple_progress) {
        PyObject *const key = PyTuple_GetItem(args_names_tuple, args_names_tuple_progress);
        PyObject *const value = args[args_progress];
        if (PyUnicode_CompareWithASCIIString(key, "out") == 0 && !out_obj) { out_obj = value; }
        else if (PyUnicode_CompareWithASCIIString(key, "a_dtype") == 0 && !a_dtype_obj) { a_dtype_obj = value; }
        else if (PyUnicode_CompareWithASCIIString(key, "b_dtype") == 0 && !b_dtype_obj) { b_dtype_obj = value; }
        else if (PyUnicode_CompareWithASCIIString(key, "out_dtype") == 0 && !out_dtype_obj) { out_dtype_obj = value; }
        else {
            PyErr_Format(PyExc_TypeError, "Got unexpected keyword argument: %S", key);
            return NULL;
        }
    }

    // Check for scalar inputs
    int a_is_scalar = is_scalar(a_obj);
    int b_is_scalar = is_scalar(b_obj);

    if (a_is_scalar && b_is_scalar) {
        PyErr_SetString(PyExc_TypeError, "At least one argument must be an array");
        return NULL;
    }

    // Handle scalar + array case using scale kernel: scale(array, alpha=1, beta=scalar)
    if (a_is_scalar || b_is_scalar) {
        double scalar_value;
        PyObject *array_obj = a_is_scalar ? b_obj : a_obj;
        PyObject *scalar_obj = a_is_scalar ? a_obj : b_obj;

        if (!get_scalar_value(scalar_obj, &scalar_value)) {
            PyErr_SetString(PyExc_TypeError, "Failed to extract scalar value");
            return NULL;
        }

        if (!parse_tensor(array_obj, &a_buffer, &a_parsed)) return NULL;
        if (out_obj && !parse_tensor(out_obj, &out_buffer, &out_parsed)) goto cleanup;

        // Validate dimensions
        if (a_parsed.rank != 1 || (out_obj && out_parsed.rank != 1)) {
            PyErr_SetString(PyExc_ValueError, "Tensors must be 1D vectors");
            goto cleanup;
        }
        if (out_obj && a_parsed.dimensions != out_parsed.dimensions) {
            PyErr_SetString(PyExc_ValueError, "Output dimensions don't match input");
            goto cleanup;
        }

        // Determine dtype
        if (out_dtype_obj) {
            char const *dtype_str = PyUnicode_AsUTF8(out_dtype_obj);
            if (!dtype_str) { goto cleanup; }
            dtype = python_string_to_datatype(dtype_str);
        }
        else { dtype = a_parsed.datatype; }

        if (dtype == nk_datatype_unknown_k) {
            PyErr_SetString(PyExc_ValueError, "Unsupported datatype");
            goto cleanup;
        }

        // Find scale kernel
        nk_kernel_scale_punned_t scale_kernel = NULL;
        nk_capability_t capability = nk_cap_serial_k;
        nk_find_kernel_punned(nk_kernel_scale_k, dtype, static_capabilities, nk_cap_any_k,
                              (nk_kernel_punned_t *)&scale_kernel, &capability);
        if (!scale_kernel) {
            PyErr_Format(PyExc_LookupError, "No scale kernel for dtype '%s'", datatype_to_string(dtype));
            goto cleanup;
        }

        char *result_start = NULL;
        if (!out_obj) {
            NDArray *result_obj = PyObject_NewVar(NDArray, &NDArrayType,
                                                  a_parsed.dimensions * bytes_per_datatype(dtype));
            if (!result_obj) {
                PyErr_NoMemory();
                goto cleanup;
            }
            result_obj->datatype = dtype;
            result_obj->rank = 1;
            result_obj->shape[0] = a_parsed.dimensions;
            result_obj->shape[1] = 0;
            result_obj->strides[0] = bytes_per_datatype(dtype);
            result_obj->strides[1] = 0;
            result_obj->parent = NULL;
            result_obj->data = result_obj->start;
            return_obj = (PyObject *)result_obj;
            result_start = result_obj->data;
        }
        else {
            result_start = out_parsed.start;
            return_obj = Py_None;
        }

        // scale(a, n, alpha=1, beta=scalar) -> 1*a + scalar
        nk_scalar_buffer_t alpha_buf, beta_buf;
        nk_scalar_buffer_set_f64(&alpha_buf, 1.0, dtype);
        nk_scalar_buffer_set_f64(&beta_buf, scalar_value, dtype);
        scale_kernel(a_parsed.start, a_parsed.dimensions, &alpha_buf, &beta_buf, result_start);
        goto cleanup;
    }

    // Handle array + array case using sum kernel
    if (!parse_tensor(a_obj, &a_buffer, &a_parsed) || !parse_tensor(b_obj, &b_buffer, &b_parsed)) return NULL;
    if (out_obj && !parse_tensor(out_obj, &out_buffer, &out_parsed)) goto cleanup;

    // Validate dimensions
    if (a_parsed.rank != 1 || b_parsed.rank != 1 || (out_obj && out_parsed.rank != 1)) {
        PyErr_SetString(PyExc_ValueError, "All tensors must be 1D vectors");
        goto cleanup;
    }
    if (a_parsed.dimensions != b_parsed.dimensions) {
        PyErr_SetString(PyExc_ValueError, "Vector dimensions don't match");
        goto cleanup;
    }
    if (out_obj && a_parsed.dimensions != out_parsed.dimensions) {
        PyErr_SetString(PyExc_ValueError, "Output dimensions don't match input");
        goto cleanup;
    }

    // Check datatypes match
    if (a_parsed.datatype != b_parsed.datatype) {
        PyErr_SetString(PyExc_TypeError, "Input arrays must have matching datatypes");
        goto cleanup;
    }

    // Determine output dtype
    if (out_dtype_obj) {
        char const *dtype_str = PyUnicode_AsUTF8(out_dtype_obj);
        if (!dtype_str) { goto cleanup; }
        dtype = python_string_to_datatype(dtype_str);
    }
    else { dtype = a_parsed.datatype; }

    if (dtype == nk_datatype_unknown_k) {
        PyErr_SetString(PyExc_ValueError, "Unsupported datatype");
        goto cleanup;
    }

    // Find sum kernel
    nk_kernel_sum_punned_t sum_kernel = NULL;
    nk_capability_t capability = nk_cap_serial_k;
    nk_find_kernel_punned(nk_kernel_sum_k, dtype, static_capabilities, nk_cap_any_k, (nk_kernel_punned_t *)&sum_kernel,
                          &capability);
    if (!sum_kernel) {
        PyErr_Format(PyExc_LookupError, "No sum kernel for dtype '%s'", datatype_to_string(dtype));
        goto cleanup;
    }

    char *result_start = NULL;
    if (!out_obj) {
        NDArray *result_obj = PyObject_NewVar(NDArray, &NDArrayType, a_parsed.dimensions * bytes_per_datatype(dtype));
        if (!result_obj) {
            PyErr_NoMemory();
            goto cleanup;
        }
        result_obj->datatype = dtype;
        result_obj->rank = 1;
        result_obj->shape[0] = a_parsed.dimensions;
        result_obj->shape[1] = 0;
        result_obj->strides[0] = bytes_per_datatype(dtype);
        result_obj->strides[1] = 0;
        result_obj->parent = NULL;
        result_obj->data = result_obj->start;
        return_obj = (PyObject *)result_obj;
        result_start = result_obj->data;
    }
    else {
        result_start = out_parsed.start;
        return_obj = Py_None;
    }

    sum_kernel(a_parsed.start, b_parsed.start, a_parsed.dimensions, result_start);

cleanup:
    PyBuffer_Release(&a_buffer);
    PyBuffer_Release(&b_buffer);
    PyBuffer_Release(&out_buffer);
    return return_obj;
}

static char const doc_multiply[] = //
    "Element-wise multiplication of two vectors or a vector and a scalar.\n\n"
    "Parameters:\n"
    "    a (Union[NDArray, float, int]): First operand (vector or scalar).\n"
    "    b (Union[NDArray, float, int]): Second operand (vector or scalar).\n"
    "    out (NDArray, optional): Output buffer for the result.\n"
    "    a_dtype (Union[IntegralType, FloatType], optional): Override dtype for `a`.\n"
    "    b_dtype (Union[IntegralType, FloatType], optional): Override dtype for `b`.\n"
    "    out_dtype (Union[IntegralType, FloatType], optional): Override dtype for output.\n\n"
    "Returns:\n"
    "    NDArray: The product if `out` is not provided.\n"
    "    None: If `out` is provided (in-place operation).\n\n"
    "Equivalent to: `a * b`.\n"
    "Signature:\n"
    "    >>> def multiply(a, b, /, *, out, a_dtype, b_dtype, out_dtype) -> Optional[NDArray]: ...";

static PyObject *api_multiply(PyObject *self, PyObject *const *args, Py_ssize_t const positional_args_count,
                              PyObject *args_names_tuple) {
    (void)self;
    PyObject *return_obj = NULL;

    // This function accepts up to 6 arguments:
    PyObject *a_obj = NULL;         // Required, positional-only
    PyObject *b_obj = NULL;         // Required, positional-only
    PyObject *out_obj = NULL;       // Optional, "out" keyword-only
    PyObject *a_dtype_obj = NULL;   // Optional, "a_dtype" keyword-only
    PyObject *b_dtype_obj = NULL;   // Optional, "b_dtype" keyword-only
    PyObject *out_dtype_obj = NULL; // Optional, "out_dtype" keyword-only

    nk_datatype_t dtype = nk_datatype_unknown_k;

    Py_buffer a_buffer, b_buffer, out_buffer;
    TensorArgument a_parsed, b_parsed, out_parsed;
    memset(&a_buffer, 0, sizeof(Py_buffer));
    memset(&b_buffer, 0, sizeof(Py_buffer));
    memset(&out_buffer, 0, sizeof(Py_buffer));

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

    // Positional-only arguments
    a_obj = args[0];
    b_obj = args[1];

    // Parse keyword arguments
    for (Py_ssize_t args_names_tuple_progress = 0, args_progress = positional_args_count;
         args_names_tuple_progress < args_names_count; ++args_progress, ++args_names_tuple_progress) {
        PyObject *const key = PyTuple_GetItem(args_names_tuple, args_names_tuple_progress);
        PyObject *const value = args[args_progress];
        if (PyUnicode_CompareWithASCIIString(key, "out") == 0 && !out_obj) { out_obj = value; }
        else if (PyUnicode_CompareWithASCIIString(key, "a_dtype") == 0 && !a_dtype_obj) { a_dtype_obj = value; }
        else if (PyUnicode_CompareWithASCIIString(key, "b_dtype") == 0 && !b_dtype_obj) { b_dtype_obj = value; }
        else if (PyUnicode_CompareWithASCIIString(key, "out_dtype") == 0 && !out_dtype_obj) { out_dtype_obj = value; }
        else {
            PyErr_Format(PyExc_TypeError, "Got unexpected keyword argument: %S", key);
            return NULL;
        }
    }

    // Check for scalar inputs
    int a_is_scalar = is_scalar(a_obj);
    int b_is_scalar = is_scalar(b_obj);

    if (a_is_scalar && b_is_scalar) {
        PyErr_SetString(PyExc_TypeError, "At least one argument must be an array");
        return NULL;
    }

    // Handle scalar * array case using scale kernel: scale(array, alpha=scalar, beta=0)
    if (a_is_scalar || b_is_scalar) {
        double scalar_value;
        PyObject *array_obj = a_is_scalar ? b_obj : a_obj;
        PyObject *scalar_obj = a_is_scalar ? a_obj : b_obj;

        if (!get_scalar_value(scalar_obj, &scalar_value)) {
            PyErr_SetString(PyExc_TypeError, "Failed to extract scalar value");
            return NULL;
        }

        if (!parse_tensor(array_obj, &a_buffer, &a_parsed)) return NULL;
        if (out_obj && !parse_tensor(out_obj, &out_buffer, &out_parsed)) goto cleanup;

        // Validate dimensions
        if (a_parsed.rank != 1 || (out_obj && out_parsed.rank != 1)) {
            PyErr_SetString(PyExc_ValueError, "Tensors must be 1D vectors");
            goto cleanup;
        }
        if (out_obj && a_parsed.dimensions != out_parsed.dimensions) {
            PyErr_SetString(PyExc_ValueError, "Output dimensions don't match input");
            goto cleanup;
        }

        // Determine dtype
        if (out_dtype_obj) {
            char const *dtype_str = PyUnicode_AsUTF8(out_dtype_obj);
            if (!dtype_str) { goto cleanup; }
            dtype = python_string_to_datatype(dtype_str);
        }
        else { dtype = a_parsed.datatype; }

        if (dtype == nk_datatype_unknown_k) {
            PyErr_SetString(PyExc_ValueError, "Unsupported datatype");
            goto cleanup;
        }

        // Find scale kernel
        nk_kernel_scale_punned_t scale_kernel = NULL;
        nk_capability_t capability = nk_cap_serial_k;
        nk_find_kernel_punned(nk_kernel_scale_k, dtype, static_capabilities, nk_cap_any_k,
                              (nk_kernel_punned_t *)&scale_kernel, &capability);
        if (!scale_kernel) {
            PyErr_Format(PyExc_LookupError, "No scale kernel for dtype '%s'", datatype_to_string(dtype));
            goto cleanup;
        }

        char *result_start = NULL;
        if (!out_obj) {
            NDArray *result_obj = PyObject_NewVar(NDArray, &NDArrayType,
                                                  a_parsed.dimensions * bytes_per_datatype(dtype));
            if (!result_obj) {
                PyErr_NoMemory();
                goto cleanup;
            }
            result_obj->datatype = dtype;
            result_obj->rank = 1;
            result_obj->shape[0] = a_parsed.dimensions;
            result_obj->shape[1] = 0;
            result_obj->strides[0] = bytes_per_datatype(dtype);
            result_obj->strides[1] = 0;
            result_obj->parent = NULL;
            result_obj->data = result_obj->start;
            return_obj = (PyObject *)result_obj;
            result_start = result_obj->data;
        }
        else {
            result_start = out_parsed.start;
            return_obj = Py_None;
        }

        // scale(a, n, alpha=scalar, beta=0) -> scalar*a + 0
        nk_scalar_buffer_t alpha_buf, beta_buf;
        nk_scalar_buffer_set_f64(&alpha_buf, scalar_value, dtype);
        nk_scalar_buffer_set_f64(&beta_buf, 0.0, dtype);
        scale_kernel(a_parsed.start, a_parsed.dimensions, &alpha_buf, &beta_buf, result_start);
        goto cleanup;
    }

    // Handle array * array case using fma kernel: fma(a, b, dummy, n, alpha=1, beta=0) -> 1*a*b + 0
    if (!parse_tensor(a_obj, &a_buffer, &a_parsed) || !parse_tensor(b_obj, &b_buffer, &b_parsed)) return NULL;
    if (out_obj && !parse_tensor(out_obj, &out_buffer, &out_parsed)) goto cleanup;

    // Validate dimensions
    if (a_parsed.rank != 1 || b_parsed.rank != 1 || (out_obj && out_parsed.rank != 1)) {
        PyErr_SetString(PyExc_ValueError, "All tensors must be 1D vectors");
        goto cleanup;
    }
    if (a_parsed.dimensions != b_parsed.dimensions) {
        PyErr_SetString(PyExc_ValueError, "Vector dimensions don't match");
        goto cleanup;
    }
    if (out_obj && a_parsed.dimensions != out_parsed.dimensions) {
        PyErr_SetString(PyExc_ValueError, "Output dimensions don't match input");
        goto cleanup;
    }

    // Check datatypes match
    if (a_parsed.datatype != b_parsed.datatype) {
        PyErr_SetString(PyExc_TypeError, "Input arrays must have matching datatypes");
        goto cleanup;
    }

    // Determine output dtype
    if (out_dtype_obj) {
        char const *dtype_str = PyUnicode_AsUTF8(out_dtype_obj);
        if (!dtype_str) { goto cleanup; }
        dtype = python_string_to_datatype(dtype_str);
    }
    else { dtype = a_parsed.datatype; }

    if (dtype == nk_datatype_unknown_k) {
        PyErr_SetString(PyExc_ValueError, "Unsupported datatype");
        goto cleanup;
    }

    // Find fma kernel
    nk_kernel_fma_punned_t fma_kernel = NULL;
    nk_capability_t capability = nk_cap_serial_k;
    nk_find_kernel_punned(nk_kernel_fma_k, dtype, static_capabilities, nk_cap_any_k, (nk_kernel_punned_t *)&fma_kernel,
                          &capability);
    if (!fma_kernel) {
        PyErr_Format(PyExc_LookupError, "No fma kernel for dtype '%s'", datatype_to_string(dtype));
        goto cleanup;
    }

    char *result_start = NULL;
    if (!out_obj) {
        NDArray *result_obj = PyObject_NewVar(NDArray, &NDArrayType, a_parsed.dimensions * bytes_per_datatype(dtype));
        if (!result_obj) {
            PyErr_NoMemory();
            goto cleanup;
        }
        result_obj->datatype = dtype;
        result_obj->rank = 1;
        result_obj->shape[0] = a_parsed.dimensions;
        result_obj->shape[1] = 0;
        result_obj->strides[0] = bytes_per_datatype(dtype);
        result_obj->strides[1] = 0;
        result_obj->parent = NULL;
        result_obj->data = result_obj->start;
        return_obj = (PyObject *)result_obj;
        result_start = result_obj->data;
    }
    else {
        result_start = out_parsed.start;
        return_obj = Py_None;
    }

    // fma(a, b, c, n, alpha=1, beta=0) -> 1*a*b + 0*c
    // For multiply, we use result_start as c (ignored since beta=0)
    nk_scalar_buffer_t alpha_buf, beta_buf;
    nk_scalar_buffer_set_f64(&alpha_buf, 1.0, dtype);
    nk_scalar_buffer_set_f64(&beta_buf, 0.0, dtype);
    fma_kernel(a_parsed.start, b_parsed.start, result_start, a_parsed.dimensions, &alpha_buf, &beta_buf, result_start);

cleanup:
    PyBuffer_Release(&a_buffer);
    PyBuffer_Release(&b_buffer);
    PyBuffer_Release(&out_buffer);
    return return_obj;
}

static char const doc_sin[] = //
    "Element-wise trigonometric sine.\n\n"
    "Parameters:\n"
    "    a (NDArray): Input vector of angles in radians.\n"
    "    dtype (Union[IntegralType, FloatType], optional): Override the presumed numeric type name.\n"
    "    out (NDArray, optional): Vector for resulting values.\n\n"
    "Returns:\n"
    "    NDArray: The sine values if `out` is not provided.\n"
    "    None: If `out` is provided.\n\n"
    "Signature:\n"
    "    >>> def sin(a, /, dtype, *, out) -> Optional[NDArray]: ...";

static char const doc_cos[] = //
    "Element-wise trigonometric cosine.\n\n"
    "Parameters:\n"
    "    a (NDArray): Input vector of angles in radians.\n"
    "    dtype (Union[IntegralType, FloatType], optional): Override the presumed numeric type name.\n"
    "    out (NDArray, optional): Vector for resulting values.\n\n"
    "Returns:\n"
    "    NDArray: The cosine values if `out` is not provided.\n"
    "    None: If `out` is provided.\n\n"
    "Signature:\n"
    "    >>> def cos(a, /, dtype, *, out) -> Optional[NDArray]: ...";

static char const doc_atan[] = //
    "Element-wise trigonometric arctangent.\n\n"
    "Parameters:\n"
    "    a (NDArray): Input vector of values.\n"
    "    dtype (Union[IntegralType, FloatType], optional): Override the presumed numeric type name.\n"
    "    out (NDArray, optional): Vector for resulting angles in radians.\n\n"
    "Returns:\n"
    "    NDArray: The arctangent values if `out` is not provided.\n"
    "    None: If `out` is provided.\n\n"
    "Signature:\n"
    "    >>> def atan(a, /, dtype, *, out) -> Optional[NDArray]: ...";

static PyObject *implement_trigonometry(nk_kernel_kind_t metric_kind, PyObject *const *args,
                                        Py_ssize_t const positional_args_count, PyObject *args_names_tuple) {

    PyObject *return_obj = NULL;

    // This function accepts up to 3 arguments:
    PyObject *a_obj = NULL;     // Required object, positional-only
    PyObject *dtype_obj = NULL; // Optional object, "dtype" keyword or positional
    PyObject *out_obj = NULL;   // Optional object, "out" keyword-only

    // Once parsed, the arguments will be stored in these variables:
    char const *dtype_str = NULL;
    nk_datatype_t dtype = nk_datatype_unknown_k;

    Py_buffer a_buffer, out_buffer;
    TensorArgument a_parsed, out_parsed;
    memset(&a_buffer, 0, sizeof(Py_buffer));
    memset(&out_buffer, 0, sizeof(Py_buffer));

    Py_ssize_t const args_names_count = args_names_tuple ? PyTuple_Size(args_names_tuple) : 0;
    Py_ssize_t const args_count = positional_args_count + args_names_count;
    if (args_count < 1 || args_count > 3) {
        PyErr_Format(PyExc_TypeError, "Function expects 1-3 arguments, got %zd", args_count);
        return NULL;
    }
    if (positional_args_count > 2) {
        PyErr_Format(PyExc_TypeError, "Only first 2 arguments can be positional, received %zd", positional_args_count);
        return NULL;
    }

    // Positional-only argument (input array)
    a_obj = args[0];

    // Positional or keyword argument (dtype)
    if (positional_args_count == 2) dtype_obj = args[1];

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
        if (dtype == nk_datatype_unknown_k) {
            PyErr_SetString(PyExc_ValueError, "Unsupported 'dtype'");
            return NULL;
        }
    }

    // Convert `a_obj` to `a_buffer` and to `a_parsed`
    if (!parse_tensor(a_obj, &a_buffer, &a_parsed)) return NULL;
    if (out_obj && !parse_tensor(out_obj, &out_buffer, &out_parsed)) return NULL;

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
    if (a_parsed.datatype == nk_datatype_unknown_k || (out_obj && out_parsed.datatype == nk_datatype_unknown_k)) {
        PyErr_SetString(PyExc_TypeError, "Input tensor must have a known datatype, check with `X.__array_interface__`");
        goto cleanup;
    }
    if (dtype == nk_datatype_unknown_k) dtype = a_parsed.datatype;

    // Look up the kernel and the capability
    nk_kernel_trigonometry_punned_t kernel = NULL;
    nk_capability_t capability = nk_cap_serial_k;
    nk_find_kernel_punned(metric_kind, dtype, static_capabilities, nk_cap_any_k, (nk_kernel_punned_t *)&kernel,
                          &capability);
    if (!kernel) {
        PyErr_Format( //
            PyExc_LookupError,
            "Unsupported metric '%c' and datatype combination ('%s'/'%s') and `dtype` override ('%s'/'%s')",
            metric_kind,                                                                             //
            a_buffer.format ? a_buffer.format : "nil", datatype_to_python_string(a_parsed.datatype), //
            dtype_str ? dtype_str : "nil", datatype_to_python_string(dtype));
        goto cleanup;
    }

    char *output_start = NULL;

    // Allocate the output array if it wasn't provided
    if (!out_obj) {
        NDArray *output_obj = PyObject_NewVar(NDArray, &NDArrayType, a_parsed.dimensions * bytes_per_datatype(dtype));
        if (!output_obj) {
            PyErr_NoMemory();
            goto cleanup;
        }

        // Initialize the object
        output_obj->datatype = dtype;
        output_obj->rank = 1;
        output_obj->shape[0] = a_parsed.dimensions;
        output_obj->shape[1] = 0;
        output_obj->strides[0] = bytes_per_datatype(dtype);
        output_obj->strides[1] = 0;
        output_obj->parent = NULL;
        output_obj->data = output_obj->start;
        return_obj = (PyObject *)output_obj;
        output_start = output_obj->data;
    }
    else {
        output_start = out_parsed.start;
        return_obj = Py_None;
    }

    kernel(a_parsed.start, a_parsed.dimensions, output_start);
cleanup:
    PyBuffer_Release(&a_buffer);
    PyBuffer_Release(&out_buffer);
    return return_obj;
}

static PyObject *api_sin(PyObject *self, PyObject *const *args, Py_ssize_t const positional_args_count,
                         PyObject *args_names_tuple) {
    return implement_trigonometry(nk_kernel_sin_k, args, positional_args_count, args_names_tuple);
}

static PyObject *api_cos(PyObject *self, PyObject *const *args, Py_ssize_t const positional_args_count,
                         PyObject *args_names_tuple) {
    return implement_trigonometry(nk_kernel_cos_k, args, positional_args_count, args_names_tuple);
}

static PyObject *api_atan(PyObject *self, PyObject *const *args, Py_ssize_t const positional_args_count,
                          PyObject *args_names_tuple) {
    return implement_trigonometry(nk_kernel_atan_k, args, positional_args_count, args_names_tuple);
}

// Mesh alignment functions (Kabsch, Umeyama, RMSD)
// These compute point cloud alignment and return a dictionary with:
//   - rotation: 3x3 rotation matrix (as 9-element list, row-major)
//   - scale: uniform scaling factor
//   - rmsd: root mean square deviation after alignment
//   - a_centroid: centroid of first point cloud
//   - b_centroid: centroid of second point cloud

static char const doc_kabsch[] = //
    "Compute optimal rigid transformation (Kabsch algorithm) between two point clouds.\n\n"
    "Finds the optimal rotation matrix that minimizes RMSD between point clouds.\n"
    "The transformation aligns point cloud A to point cloud B:\n"
    "    a'_i = scale * R * (a_i - a_centroid) + b_centroid\n\n"
    "Supports both single-pair and batched inputs:\n"
    "    - Single pair: (N, 3) -> rotation (3,3), scale (), rmsd (), centroids (3,)\n"
    "    - Batched: (B, N, 3) -> rotation (B,3,3), scale (B,), rmsd (B,), centroids (B,3)\n\n"
    "Parameters:\n"
    "    a (NDArray): First point cloud(s), shape (N, 3) or (B, N, 3), float32 or float64.\n"
    "    b (NDArray): Second point cloud(s), shape (N, 3) or (B, N, 3), same dtype as a.\n\n"
    "Returns:\n"
    "    tuple: (rotation, scale, rmsd, a_centroid, b_centroid) - all NDArray.\n\n"
    "Example:\n"
    "    >>> rot, scale, rmsd, a_c, b_c = numkong.kabsch(a, b)\n"
    "    >>> np.asarray(rot)  # (3, 3) rotation matrix\n"
    "    >>> float(scale)     # scale factor (always 1.0 for Kabsch)\n";

static char const doc_umeyama[] = //
    "Compute optimal similarity transformation (Umeyama algorithm) between two point clouds.\n\n"
    "Finds the optimal rotation matrix and uniform scaling factor that minimize RMSD.\n"
    "The transformation aligns point cloud A to point cloud B:\n"
    "    a'_i = scale * R * (a_i - a_centroid) + b_centroid\n\n"
    "Supports both single-pair and batched inputs:\n"
    "    - Single pair: (N, 3) -> rotation (3,3), scale (), rmsd (), centroids (3,)\n"
    "    - Batched: (B, N, 3) -> rotation (B,3,3), scale (B,), rmsd (B,), centroids (B,3)\n\n"
    "Parameters:\n"
    "    a (NDArray): First point cloud(s), shape (N, 3) or (B, N, 3), float32 or float64.\n"
    "    b (NDArray): Second point cloud(s), shape (N, 3) or (B, N, 3), same dtype as a.\n\n"
    "Returns:\n"
    "    tuple: (rotation, scale, rmsd, a_centroid, b_centroid) - all NDArray.\n\n"
    "Example:\n"
    "    >>> rot, scale, rmsd, a_c, b_c = numkong.umeyama(a, b)\n"
    "    >>> float(scale)  # Will differ from 1.0 if point clouds have different scales\n";

static char const doc_rmsd[] = //
    "Compute RMSD between two point clouds without alignment optimization.\n\n"
    "Computes root mean square deviation after centering both clouds.\n"
    "Returns identity rotation and scale=1.0.\n\n"
    "Supports both single-pair and batched inputs:\n"
    "    - Single pair: (N, 3) -> rotation (3,3), scale (), rmsd (), centroids (3,)\n"
    "    - Batched: (B, N, 3) -> rotation (B,3,3), scale (B,), rmsd (B,), centroids (B,3)\n\n"
    "Parameters:\n"
    "    a (NDArray): First point cloud(s), shape (N, 3) or (B, N, 3), float32 or float64.\n"
    "    b (NDArray): Second point cloud(s), shape (N, 3) or (B, N, 3), same dtype as a.\n\n"
    "Returns:\n"
    "    tuple: (rotation, scale, rmsd, a_centroid, b_centroid) - all NDArray.\n";

static PyObject *implement_mesh_alignment(nk_kernel_kind_t metric_kind, PyObject *const *args,
                                          Py_ssize_t positional_args_count) {
    // We expect exactly 2 positional arguments: a and b
    if (positional_args_count != 2) {
        PyErr_SetString(PyExc_TypeError, "Expected exactly 2 positional arguments (a, b)");
        return NULL;
    }

    Py_buffer a_buffer, b_buffer;
    memset(&a_buffer, 0, sizeof(Py_buffer));
    memset(&b_buffer, 0, sizeof(Py_buffer));

    // Get buffer for array a
    if (PyObject_GetBuffer(args[0], &a_buffer, PyBUF_STRIDES | PyBUF_FORMAT) != 0) {
        PyErr_SetString(PyExc_TypeError, "First argument must support buffer protocol");
        return NULL;
    }

    // Get buffer for array b
    if (PyObject_GetBuffer(args[1], &b_buffer, PyBUF_STRIDES | PyBUF_FORMAT) != 0) {
        PyBuffer_Release(&a_buffer);
        PyErr_SetString(PyExc_TypeError, "Second argument must support buffer protocol");
        return NULL;
    }

    PyObject *result = NULL;
    NDArray *rot_tensor = NULL;
    NDArray *scale_tensor = NULL;
    NDArray *rmsd_tensor = NULL;
    NDArray *a_cent_tensor = NULL;
    NDArray *b_cent_tensor = NULL;

    // Validate shapes: accept (N, 3) for single pair or (B, N, 3) for batch
    int is_batched = 0;
    Py_ssize_t batch_size = 1;
    Py_ssize_t n_points, last_dim_a, last_dim_b;

    if (a_buffer.ndim == 2 && b_buffer.ndim == 2) {
        // Single pair: (N, 3) shape
        n_points = a_buffer.shape[0];
        last_dim_a = a_buffer.shape[1];
        last_dim_b = b_buffer.shape[1];
    }
    else if (a_buffer.ndim == 3 && b_buffer.ndim == 3) {
        // Batched: (B, N, 3) shape
        is_batched = 1;
        batch_size = a_buffer.shape[0];
        n_points = a_buffer.shape[1];
        last_dim_a = a_buffer.shape[2];
        last_dim_b = b_buffer.shape[2];
        if (a_buffer.shape[0] != b_buffer.shape[0]) {
            PyErr_SetString(PyExc_ValueError, "Batch sizes must match");
            goto cleanup;
        }
        if (a_buffer.shape[1] != b_buffer.shape[1]) {
            PyErr_SetString(PyExc_ValueError, "Point clouds must have the same number of points");
            goto cleanup;
        }
    }
    else {
        PyErr_SetString(PyExc_ValueError, "Point clouds must be 2D (N,3) or 3D (B,N,3) arrays");
        goto cleanup;
    }

    if (last_dim_a != 3 || last_dim_b != 3) {
        PyErr_SetString(PyExc_ValueError, "Point clouds must have 3 columns (x, y, z coordinates)");
        goto cleanup;
    }
    if (!is_batched && a_buffer.shape[0] != b_buffer.shape[0]) {
        PyErr_SetString(PyExc_ValueError, "Point clouds must have the same number of points");
        goto cleanup;
    }
    if (n_points < 3) {
        PyErr_SetString(PyExc_ValueError, "Point clouds must have at least 3 points");
        goto cleanup;
    }

    // Check data types and get kernel
    nk_datatype_t datatype = python_string_to_datatype(a_buffer.format);
    if (datatype != nk_f32_k && datatype != nk_f64_k) {
        PyErr_SetString(PyExc_TypeError, "Point clouds must be float32 or float64");
        goto cleanup;
    }

    // Find the appropriate kernel
    nk_metric_mesh_punned_t kernel = NULL;
    nk_capability_t capability = nk_cap_serial_k;
    nk_find_kernel_punned(metric_kind, datatype, static_capabilities, nk_cap_any_k, (nk_kernel_punned_t *)&kernel,
                          &capability);
    if (!kernel) {
        PyErr_SetString(PyExc_RuntimeError, "No suitable mesh kernel found for this data type");
        goto cleanup;
    }

    // Check contiguity - we need row-major contiguous data for the innermost 2 dimensions
    Py_ssize_t const elem_size = (datatype == nk_f32_k ? 4 : 8);
    Py_ssize_t const inner_stride_a = is_batched ? a_buffer.strides[2] : a_buffer.strides[1];
    Py_ssize_t const inner_stride_b = is_batched ? b_buffer.strides[2] : b_buffer.strides[1];
    if (inner_stride_a != elem_size || inner_stride_b != elem_size) {
        PyErr_SetString(PyExc_ValueError, "Point clouds must be C-contiguous (row-major)");
        goto cleanup;
    }

    // Calculate strides between batches
    Py_ssize_t const batch_stride_a = is_batched ? a_buffer.strides[0] : 0;
    Py_ssize_t const batch_stride_b = is_batched ? b_buffer.strides[0] : 0;
    nk_size_t n = (nk_size_t)n_points;

    // Output dtype for scale/rmsd is always f64 (nk_fmax_t)
    nk_datatype_t out_dtype = nk_f64_k;

    if (!is_batched) {
        // Single pair case - return 0D scalars for scale/rmsd, (3,3) for rotation, (3,) for centroids
        Py_ssize_t rot_shape[2] = {3, 3};
        Py_ssize_t cent_shape[1] = {3};

        rot_tensor = NDArray_new(datatype, 2, rot_shape);
        scale_tensor = NDArray_new(out_dtype, 0, NULL);
        rmsd_tensor = NDArray_new(out_dtype, 0, NULL);
        a_cent_tensor = NDArray_new(datatype, 1, cent_shape);
        b_cent_tensor = NDArray_new(datatype, 1, cent_shape);

        if (!rot_tensor || !scale_tensor || !rmsd_tensor || !a_cent_tensor || !b_cent_tensor) goto cleanup;

        nk_fmax_t scale = 0.0, rmsd_result = 0.0;

        if (datatype == nk_f64_k) {
            nk_f64_t *a_centroid = (nk_f64_t *)a_cent_tensor->data;
            nk_f64_t *b_centroid = (nk_f64_t *)b_cent_tensor->data;
            nk_f64_t *rotation = (nk_f64_t *)rot_tensor->data;
            kernel(a_buffer.buf, b_buffer.buf, n, a_centroid, b_centroid, rotation, &scale, &rmsd_result);
        }
        else { // nk_f32_k
            nk_f32_t *a_centroid = (nk_f32_t *)a_cent_tensor->data;
            nk_f32_t *b_centroid = (nk_f32_t *)b_cent_tensor->data;
            nk_f32_t *rotation = (nk_f32_t *)rot_tensor->data;
            kernel(a_buffer.buf, b_buffer.buf, n, a_centroid, b_centroid, rotation, &scale, &rmsd_result);
        }

        // Copy scalars into 0D tensors
        *(nk_f64_t *)scale_tensor->data = scale;
        *(nk_f64_t *)rmsd_tensor->data = rmsd_result;
    }
    else {
        // Batched case: (B, N, 3) -> rotation (B,3,3), scale (B,), rmsd (B,), centroids (B,3)
        Py_ssize_t rot_shape[3] = {batch_size, 3, 3};
        Py_ssize_t scalar_shape[1] = {batch_size};
        Py_ssize_t cent_shape[2] = {batch_size, 3};

        rot_tensor = NDArray_new(datatype, 3, rot_shape);
        scale_tensor = NDArray_new(out_dtype, 1, scalar_shape);
        rmsd_tensor = NDArray_new(out_dtype, 1, scalar_shape);
        a_cent_tensor = NDArray_new(datatype, 2, cent_shape);
        b_cent_tensor = NDArray_new(datatype, 2, cent_shape);

        if (!rot_tensor || !scale_tensor || !rmsd_tensor || !a_cent_tensor || !b_cent_tensor) goto cleanup;

        char *a_ptr = (char *)a_buffer.buf;
        char *b_ptr = (char *)b_buffer.buf;

        for (Py_ssize_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
            nk_fmax_t scale = 0.0, rmsd_result = 0.0;

            if (datatype == nk_f64_k) {
                nk_f64_t *a_centroid = (nk_f64_t *)(a_cent_tensor->data + batch_idx * 3 * sizeof(nk_f64_t));
                nk_f64_t *b_centroid = (nk_f64_t *)(b_cent_tensor->data + batch_idx * 3 * sizeof(nk_f64_t));
                nk_f64_t *rotation = (nk_f64_t *)(rot_tensor->data + batch_idx * 9 * sizeof(nk_f64_t));
                kernel(a_ptr + batch_idx * batch_stride_a, b_ptr + batch_idx * batch_stride_b, n, a_centroid,
                       b_centroid, rotation, &scale, &rmsd_result);
            }
            else { // nk_f32_k
                nk_f32_t *a_centroid = (nk_f32_t *)(a_cent_tensor->data + batch_idx * 3 * sizeof(nk_f32_t));
                nk_f32_t *b_centroid = (nk_f32_t *)(b_cent_tensor->data + batch_idx * 3 * sizeof(nk_f32_t));
                nk_f32_t *rotation = (nk_f32_t *)(rot_tensor->data + batch_idx * 9 * sizeof(nk_f32_t));
                kernel(a_ptr + batch_idx * batch_stride_a, b_ptr + batch_idx * batch_stride_b, n, a_centroid,
                       b_centroid, rotation, &scale, &rmsd_result);
            }

            // Store scale and rmsd (always f64)
            ((nk_f64_t *)scale_tensor->data)[batch_idx] = scale;
            ((nk_f64_t *)rmsd_tensor->data)[batch_idx] = rmsd_result;
        }
    }

    // Build result tuple: (rotation, scale, rmsd, a_centroid, b_centroid)
    result = PyTuple_Pack(5, (PyObject *)rot_tensor, (PyObject *)scale_tensor, (PyObject *)rmsd_tensor,
                          (PyObject *)a_cent_tensor, (PyObject *)b_cent_tensor);

cleanup:
    // If result tuple was created, it owns the references; otherwise clean up
    if (!result) {
        Py_XDECREF(rot_tensor);
        Py_XDECREF(scale_tensor);
        Py_XDECREF(rmsd_tensor);
        Py_XDECREF(a_cent_tensor);
        Py_XDECREF(b_cent_tensor);
    }
    PyBuffer_Release(&a_buffer);
    PyBuffer_Release(&b_buffer);
    return result;
}

static PyObject *api_kabsch(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                            PyObject *args_names_tuple) {
    (void)self;
    (void)args_names_tuple;
    return implement_mesh_alignment(nk_kernel_kabsch_k, args, positional_args_count);
}

static PyObject *api_umeyama(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                             PyObject *args_names_tuple) {
    (void)self;
    (void)args_names_tuple;
    return implement_mesh_alignment(nk_kernel_umeyama_k, args, positional_args_count);
}

static PyObject *api_rmsd_mesh(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                               PyObject *args_names_tuple) {
    (void)self;
    (void)args_names_tuple;
    return implement_mesh_alignment(nk_kernel_rmsd_k, args, positional_args_count);
}

/// @brief  Parse a Python buffer format string into a NumKong datatype.
static int buffer_datatype(Py_buffer const *buffer, nk_datatype_t *dtype) {
    char const *format = buffer->format ? buffer->format : NULL;
    if (!format) {
        PyErr_SetString(PyExc_TypeError, "Input buffer must expose a format string");
        return 0;
    }
    *dtype = python_string_to_datatype(format);
    if (*dtype == nk_datatype_unknown_k) {
        PyErr_Format(PyExc_TypeError, "Unsupported buffer format '%s'", format);
        return 0;
    }
    return 1;
}

/// @brief  Convert a scalar at a byte address to float32.
static int convert_scalar_to_f32(nk_datatype_t src_dtype, char const *src, nk_f32_t *value) {
    switch (src_dtype) {
    case nk_f64_k: *value = (nk_f32_t) * (nk_f64_t const *)src; return 1;
    case nk_f32_k: *value = *(nk_f32_t const *)src; return 1;
    case nk_f16_k: nk_f16_to_f32((nk_f16_t const *)src, value); return 1;
    case nk_bf16_k: nk_bf16_to_f32((nk_bf16_t const *)src, value); return 1;
    case nk_e4m3_k: nk_e4m3_to_f32((nk_e4m3_t const *)src, value); return 1;
    case nk_e5m2_k: nk_e5m2_to_f32((nk_e5m2_t const *)src, value); return 1;
    case nk_b8_k: *value = (nk_f32_t) * (nk_b8_t const *)src; return 1;
    case nk_i8_k: *value = (nk_f32_t) * (nk_i8_t const *)src; return 1;
    case nk_u8_k: *value = (nk_f32_t) * (nk_u8_t const *)src; return 1;
    case nk_i16_k: *value = (nk_f32_t) * (nk_i16_t const *)src; return 1;
    case nk_u16_k: *value = (nk_f32_t) * (nk_u16_t const *)src; return 1;
    case nk_i32_k: *value = (nk_f32_t) * (nk_i32_t const *)src; return 1;
    case nk_u32_k: *value = (nk_f32_t) * (nk_u32_t const *)src; return 1;
    case nk_i64_k: *value = (nk_f32_t) * (nk_i64_t const *)src; return 1;
    case nk_u64_k: *value = (nk_f32_t) * (nk_u64_t const *)src; return 1;
    default: PyErr_SetString(PyExc_TypeError, "Unsupported datatype for matmul conversion"); return 0;
    }
}

/// @brief  Convert a strided row to bf16.
static int convert_row_to_bf16(nk_datatype_t src_dtype, char const *row_start, nk_size_t count,
                               nk_size_t element_stride, nk_bf16_t *dest_row) {
    for (nk_size_t j = 0; j < count; j++) {
        nk_f32_t value = 0;
        char const *element_ptr = row_start + j * element_stride;
        if (!convert_scalar_to_f32(src_dtype, element_ptr, &value)) return 0;
        nk_f32_to_bf16(&value, &dest_row[j]);
    }
    return 1;
}

/// @brief  Convert a strided row to i8 with clamping.
static int convert_row_to_i8(nk_datatype_t src_dtype, char const *row_start, nk_size_t count, nk_size_t element_stride,
                             nk_i8_t *dest_row) {
    for (nk_size_t j = 0; j < count; j++) {
        nk_f32_t value = 0;
        char const *element_ptr = row_start + j * element_stride;
        if (!convert_scalar_to_f32(src_dtype, element_ptr, &value)) return 0;
        if (value < -128) value = -128;
        if (value > 127) value = 127;
        dest_row[j] = (nk_i8_t)lround((double)value);
    }
    return 1;
}

#pragma region Matmul Packing

static char const doc_pack_matmul_argument[] =
    "pack_matmul_argument(b, dtype='bf16') -> PackedMatrix\n\n"
    "Pack a matrix for repeated matrix multiplication.\n\n"
    "The packed format is opaque and backend-specific (optimized for AMX).\n"
    "Requires AMX support (Sapphire Rapids or newer CPU).\n"
    "Use with matmul() to compute C = A @ B.\n\n"
    "Parameters:\n"
    "    b : array_like\n"
    "        The (n, k) matrix to pack. This is typically the 'database' or 'weights' matrix\n"
    "        that will be multiplied against multiple 'query' matrices.\n"
    "    dtype : str, optional\n"
    "        Data type for packing: 'bf16' (default) or 'i8'.\n\n"
    "Returns:\n"
    "    PackedMatrix : Opaque packed matrix for use with matmul().\n\n"
    "Example:\n"
    "    >>> database = np.random.randn(1000, 768).astype(np.float32)  # 1000 vectors of dim 768\n"
    "    >>> packed = simd.pack_matmul_argument(database, dtype='bf16')\n"
    "    >>> queries = np.random.randn(10, 768).astype(np.float32)  # 10 query vectors\n"
    "    >>> result = simd.matmul(queries, packed)  # (10, 1000) dot products\n";

static char const doc_pack_matrix[] = "pack_matrix(b, dtype='bf16') -> PackedMatrix\n\n"
                                      "Deprecated alias for pack_matmul_argument().\n";

static PyObject *api_pack_matmul_argument(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                          PyObject *args_names_tuple) {
    (void)self;

    // Parse arguments.
    PyObject *b_obj = NULL;
    char const *dtype_str = "bf16";

    Py_ssize_t args_names_count = args_names_tuple ? PyTuple_Size(args_names_tuple) : 0;
    Py_ssize_t args_count = positional_args_count + args_names_count;

    if (positional_args_count < 1 || args_count > 2) {
        PyErr_SetString(PyExc_TypeError, "pack_matmul_argument() requires 1-2 arguments: b, dtype='bf16'");
        return NULL;
    }

    b_obj = args[0];

    // Parse the optional dtype argument.
    for (Py_ssize_t i = 0; i < args_names_count; i++) {
        PyObject *name = PyTuple_GET_ITEM(args_names_tuple, i);
        char const *name_str = PyUnicode_AsUTF8(name);
        if (same_string(name_str, "dtype")) {
            PyObject *val = args[positional_args_count + i];
            if (!PyUnicode_Check(val)) {
                PyErr_SetString(PyExc_TypeError, "dtype must be a string");
                return NULL;
            }
            dtype_str = PyUnicode_AsUTF8(val);
        }
    }
    // Accept an optional second positional argument.
    if (positional_args_count >= 2) {
        if (!PyUnicode_Check(args[1])) {
            PyErr_SetString(PyExc_TypeError, "dtype must be a string");
            return NULL;
        }
        dtype_str = PyUnicode_AsUTF8(args[1]);
    }

    // Resolve the target packing dtype.
    nk_datatype_t target_dtype;
    if (same_string(dtype_str, "bf16") || same_string(dtype_str, "bfloat16")) { target_dtype = nk_bf16_k; }
    else if (same_string(dtype_str, "i8") || same_string(dtype_str, "int8")) { target_dtype = nk_i8_k; }
    else {
        PyErr_Format(PyExc_ValueError, "Unsupported dtype '%s'. Use 'bf16' or 'i8'.", dtype_str);
        return NULL;
    }

    // Get the input buffer.
    Py_buffer b_buffer;
    if (PyObject_GetBuffer(b_obj, &b_buffer, PyBUF_STRIDES | PyBUF_FORMAT) != 0) {
        PyErr_SetString(PyExc_TypeError, "b must support buffer protocol");
        return NULL;
    }

    // Validate the input as a 2D matrix.
    if (b_buffer.ndim != 2) {
        PyBuffer_Release(&b_buffer);
        PyErr_SetString(PyExc_ValueError, "b must be a 2D matrix");
        return NULL;
    }

    nk_datatype_t src_dtype;
    if (!buffer_datatype(&b_buffer, &src_dtype)) {
        PyBuffer_Release(&b_buffer);
        return NULL;
    }
    if (is_complex(src_dtype)) {
        PyBuffer_Release(&b_buffer);
        PyErr_SetString(PyExc_TypeError, "complex matrices are not supported for matmul packing");
        return NULL;
    }
    if (b_buffer.strides[0] < 0 || b_buffer.strides[1] < 0) {
        PyBuffer_Release(&b_buffer);
        PyErr_SetString(PyExc_ValueError, "matmul packing does not support negative strides");
        return NULL;
    }

    nk_size_t n = (nk_size_t)b_buffer.shape[0];
    nk_size_t k = (nk_size_t)b_buffer.shape[1];
    nk_size_t row_stride = (nk_size_t)b_buffer.strides[0];
    nk_size_t col_stride = (nk_size_t)b_buffer.strides[1];

    // Calculate the packed size and allocate the packed buffer.
    // Require the AMX backend (Sapphire Rapids or newer).
#if !NK_TARGET_SAPPHIRE_AMX
    (void)n;
    (void)k;
    (void)row_stride;
    (void)col_stride;
    PyBuffer_Release(&b_buffer);
    PyErr_SetString(PyExc_RuntimeError, "pack_matmul_argument requires AMX support (Sapphire Rapids or newer CPU)");
    return NULL;
#else
    nk_size_t packed_size;
    if (target_dtype == nk_bf16_k) { packed_size = nk_matmul_bf16_packed_size_sapphire_amx(n, k); }
    else { packed_size = nk_matmul_i8_packed_size_sapphire_amx(n, k); }

    PackedMatrix *packed = PyObject_NewVar(PackedMatrix, &PackedMatrixType, packed_size);
    if (!packed) {
        PyBuffer_Release(&b_buffer);
        PyErr_NoMemory();
        return NULL;
    }

    packed->dtype = target_dtype;
    packed->n = n;
    packed->k = k;

    // Convert input data to the packing format and pack it.
    if (target_dtype == nk_bf16_k) {
        nk_bf16_t const *b_bf16 = NULL;
        nk_bf16_t *temp_bf16 = NULL;
        nk_size_t b_bf16_stride = row_stride;

        if (src_dtype == nk_bf16_k && col_stride == sizeof(nk_bf16_t)) {
            b_bf16 = (nk_bf16_t const *)b_buffer.buf;
            b_bf16_stride = row_stride;
        }
        else {
            temp_bf16 = (nk_bf16_t *)PyMem_Malloc(n * k * sizeof(nk_bf16_t));
            if (!temp_bf16) {
                Py_DECREF(packed);
                PyBuffer_Release(&b_buffer);
                PyErr_NoMemory();
                return NULL;
            }

            for (nk_size_t i = 0; i < n; i++) {
                char const *row = (char const *)b_buffer.buf + i * row_stride;
                if (!convert_row_to_bf16(src_dtype, row, k, col_stride, temp_bf16 + i * k)) {
                    PyMem_Free(temp_bf16);
                    Py_DECREF(packed);
                    PyBuffer_Release(&b_buffer);
                    return NULL;
                }
            }
            b_bf16 = temp_bf16;
            b_bf16_stride = k * sizeof(nk_bf16_t);
        }

        nk_matmul_bf16_pack_sapphire_amx(b_bf16, n, k, b_bf16_stride, packed->start);

        if (temp_bf16) PyMem_Free(temp_bf16);
    }
    else {
        nk_i8_t const *b_i8 = NULL;
        nk_i8_t *temp_i8 = NULL;
        nk_size_t b_i8_stride = row_stride;

        if (src_dtype == nk_i8_k && col_stride == sizeof(nk_i8_t)) {
            b_i8 = (nk_i8_t const *)b_buffer.buf;
            b_i8_stride = row_stride;
        }
        else {
            temp_i8 = (nk_i8_t *)PyMem_Malloc(n * k * sizeof(nk_i8_t));
            if (!temp_i8) {
                Py_DECREF(packed);
                PyBuffer_Release(&b_buffer);
                PyErr_NoMemory();
                return NULL;
            }

            for (nk_size_t i = 0; i < n; i++) {
                char const *row = (char const *)b_buffer.buf + i * row_stride;
                if (!convert_row_to_i8(src_dtype, row, k, col_stride, temp_i8 + i * k)) {
                    PyMem_Free(temp_i8);
                    Py_DECREF(packed);
                    PyBuffer_Release(&b_buffer);
                    return NULL;
                }
            }
            b_i8 = temp_i8;
            b_i8_stride = k * sizeof(nk_i8_t);
        }

        nk_matmul_i8_pack_sapphire_amx(b_i8, n, k, b_i8_stride, packed->start);

        if (temp_i8) PyMem_Free(temp_i8);
    }

    PyBuffer_Release(&b_buffer);
    return (PyObject *)packed;
#endif // NK_TARGET_SAPPHIRE_AMX
}

static PyObject *api_pack_matrix(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                 PyObject *args_names_tuple) {
    if (PyErr_WarnEx(PyExc_DeprecationWarning, "pack_matrix() is deprecated; use pack_matmul_argument() instead", 1) <
        0)
        return NULL;
    return api_pack_matmul_argument(self, args, positional_args_count, args_names_tuple);
}

static char const doc_matmul[] = "matmul(a, b) -> NDArray\n\n"
                                 "Compute matrix multiplication C = A @ B with a pre-packed B matrix.\n\n"
                                 "Parameters:\n"
                                 "    a : array_like\n"
                                 "        The (m, k) query/input matrix.\n"
                                 "    b : PackedMatrix\n"
                                 "        Pre-packed (n, k) matrix from pack_matmul_argument().\n\n"
                                 "Returns:\n"
                                 "    NDArray : (m, n) result matrix.\n\n"
                                 "Note:\n"
                                 "    The kernel computes C[i,j] = dot(a[i], b[j]) for all i,j.\n"
                                 "    This is equivalent to A @ B.T where B is the original unpacked matrix.\n\n"
                                 "Example:\n"
                                 "    >>> database = np.random.randn(1000, 768).astype(np.float32)\n"
                                 "    >>> packed = simd.pack_matmul_argument(database, dtype='bf16')\n"
                                 "    >>> queries = np.random.randn(10, 768).astype(np.float32)\n"
                                 "    >>> result = simd.matmul(queries, packed)  # (10, 1000)\n";

static PyObject *api_matmul(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                            PyObject *args_names_tuple) {
    (void)self;
    (void)args_names_tuple;

    if (positional_args_count != 2) {
        PyErr_SetString(PyExc_TypeError, "matmul() requires exactly 2 arguments: a, b");
        return NULL;
    }

    PyObject *a_obj = args[0];
    PyObject *b_obj = args[1];

    // Verify that b is a PackedMatrix.
    if (!PyObject_TypeCheck(b_obj, &PackedMatrixType)) {
        PyErr_SetString(PyExc_TypeError, "b must be a PackedMatrix (use pack_matmul_argument() first)");
        return NULL;
    }
    PackedMatrix *packed = (PackedMatrix *)b_obj;

    // Get the input buffer for a.
    Py_buffer a_buffer;
    if (PyObject_GetBuffer(a_obj, &a_buffer, PyBUF_STRIDES | PyBUF_FORMAT) != 0) {
        PyErr_SetString(PyExc_TypeError, "a must support buffer protocol");
        return NULL;
    }

    // Validate dimensions.
    if (a_buffer.ndim != 2) {
        PyBuffer_Release(&a_buffer);
        PyErr_SetString(PyExc_ValueError, "a must be a 2D matrix");
        return NULL;
    }

    nk_datatype_t src_dtype;
    if (!buffer_datatype(&a_buffer, &src_dtype)) {
        PyBuffer_Release(&a_buffer);
        return NULL;
    }
    if (is_complex(src_dtype)) {
        PyBuffer_Release(&a_buffer);
        PyErr_SetString(PyExc_TypeError, "complex matrices are not supported for matmul");
        return NULL;
    }
    if (a_buffer.strides[0] < 0 || a_buffer.strides[1] < 0) {
        PyBuffer_Release(&a_buffer);
        PyErr_SetString(PyExc_ValueError, "matmul does not support negative strides");
        return NULL;
    }

    nk_size_t m = (nk_size_t)a_buffer.shape[0];
    nk_size_t k_a = (nk_size_t)a_buffer.shape[1];
    nk_size_t row_stride = (nk_size_t)a_buffer.strides[0];
    nk_size_t col_stride = (nk_size_t)a_buffer.strides[1];

    if (k_a != packed->k) {
        PyBuffer_Release(&a_buffer);
        PyErr_Format(PyExc_ValueError, "Dimension mismatch: a has k=%zu but packed matrix has k=%zu", k_a, packed->k);
        return NULL;
    }

    nk_size_t n = packed->n;
    nk_size_t k = packed->k;

    // Allocate output tensor
    Py_ssize_t out_shape[2] = {(Py_ssize_t)m, (Py_ssize_t)n};
    nk_datatype_t out_dtype = nk_f32_k; // BF16 matmul outputs F32
    if (packed->dtype == nk_i8_k) out_dtype = nk_i32_k;

    NDArray *result = NDArray_new(out_dtype, 2, out_shape);
    if (!result) {
        PyBuffer_Release(&a_buffer);
        return NULL;
    }

    nk_size_t c_stride = n * bytes_per_datatype(out_dtype);

    // Convert input A to appropriate format and compute
    if (packed->dtype == nk_bf16_k) {
        nk_bf16_t const *a_bf16 = NULL;
        nk_bf16_t *temp_a = NULL;
        nk_size_t a_bf16_stride = row_stride;

        if (src_dtype == nk_bf16_k && col_stride == sizeof(nk_bf16_t)) {
            a_bf16 = (nk_bf16_t const *)a_buffer.buf;
            a_bf16_stride = row_stride;
        }
        else {
            temp_a = (nk_bf16_t *)PyMem_Malloc(m * k * sizeof(nk_bf16_t));
            if (!temp_a) {
                Py_DECREF(result);
                PyBuffer_Release(&a_buffer);
                PyErr_NoMemory();
                return NULL;
            }

            for (nk_size_t i = 0; i < m; i++) {
                char const *row = (char const *)a_buffer.buf + i * row_stride;
                if (!convert_row_to_bf16(src_dtype, row, k, col_stride, temp_a + i * k)) {
                    PyMem_Free(temp_a);
                    Py_DECREF(result);
                    PyBuffer_Release(&a_buffer);
                    return NULL;
                }
            }
            a_bf16 = temp_a;
            a_bf16_stride = k * sizeof(nk_bf16_t);
        }

#if NK_TARGET_SAPPHIRE_AMX
        nk_matmul_bf16_f32_sapphire_amx(a_bf16, packed->start, (nk_f32_t *)result->data, m, n, k, a_bf16_stride,
                                        c_stride);
#else
        (void)a_bf16;
        Py_DECREF(result);
        if (temp_a) PyMem_Free(temp_a);
        PyBuffer_Release(&a_buffer);
        PyErr_SetString(PyExc_RuntimeError, "matmul requires AMX support (Sapphire Rapids or newer CPU)");
        return NULL;
#endif

        if (temp_a) PyMem_Free(temp_a);
    }
    else {
        // i8 matmul
        nk_i8_t const *a_i8 = NULL;
        nk_i8_t *temp_a = NULL;
        nk_size_t a_i8_stride = row_stride;

        if (src_dtype == nk_i8_k && col_stride == sizeof(nk_i8_t)) {
            a_i8 = (nk_i8_t const *)a_buffer.buf;
            a_i8_stride = row_stride;
        }
        else {
            temp_a = (nk_i8_t *)PyMem_Malloc(m * k * sizeof(nk_i8_t));
            if (!temp_a) {
                Py_DECREF(result);
                PyBuffer_Release(&a_buffer);
                PyErr_NoMemory();
                return NULL;
            }

            for (nk_size_t i = 0; i < m; i++) {
                char const *row = (char const *)a_buffer.buf + i * row_stride;
                if (!convert_row_to_i8(src_dtype, row, k, col_stride, temp_a + i * k)) {
                    PyMem_Free(temp_a);
                    Py_DECREF(result);
                    PyBuffer_Release(&a_buffer);
                    return NULL;
                }
            }
            a_i8 = temp_a;
            a_i8_stride = k * sizeof(nk_i8_t);
        }

#if NK_TARGET_SAPPHIRE_AMX
        nk_matmul_i8_i32_sapphire_amx(a_i8, packed->start, (nk_i32_t *)result->data, m, n, k, a_i8_stride, c_stride);
#else
        (void)a_i8;
        Py_DECREF(result);
        if (temp_a) PyMem_Free(temp_a);
        PyBuffer_Release(&a_buffer);
        PyErr_SetString(PyExc_RuntimeError, "matmul requires AMX support (Sapphire Rapids or newer CPU)");
        return NULL;
#endif

        if (temp_a) PyMem_Free(temp_a);
    }

    PyBuffer_Release(&a_buffer);
    return (PyObject *)result;
}

#pragma endregion Matmul Packing

// There are several flags we can use to define the functions:
// - `METH_O`: Single object argument
// - `METH_VARARGS`: Variable number of arguments
// - `METH_FASTCALL`: Fast calling convention
// - `METH_KEYWORDS`: Accepts keyword arguments, can be combined with `METH_FASTCALL`
//
// https://llllllllll.github.io/c-extension-tutorial/appendix.html#c.PyMethodDef.ml_flags
static PyMethodDef nk_methods[] = {
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
    {"kld", (PyCFunction)api_kld, METH_FASTCALL | METH_KEYWORDS, doc_kld},
    {"jsd", (PyCFunction)api_jsd, METH_FASTCALL | METH_KEYWORDS, doc_jsd},
    {"angular", (PyCFunction)api_angular, METH_FASTCALL | METH_KEYWORDS, doc_angular},
    {"dot", (PyCFunction)api_dot, METH_FASTCALL | METH_KEYWORDS, doc_dot},
    {"vdot", (PyCFunction)api_vdot, METH_FASTCALL | METH_KEYWORDS, doc_vdot},
    {"hamming", (PyCFunction)api_hamming, METH_FASTCALL | METH_KEYWORDS, doc_hamming},
    {"jaccard", (PyCFunction)api_jaccard, METH_FASTCALL | METH_KEYWORDS, doc_jaccard},

    // Aliases
    {"euclidean", (PyCFunction)api_l2, METH_FASTCALL | METH_KEYWORDS, doc_l2},
    {"sqeuclidean", (PyCFunction)api_l2sq, METH_FASTCALL | METH_KEYWORDS, doc_l2sq},
    {"inner", (PyCFunction)api_dot, METH_FASTCALL | METH_KEYWORDS, doc_dot},
    {"kullbackleibler", (PyCFunction)api_kld, METH_FASTCALL | METH_KEYWORDS, doc_kld},
    {"jensenshannon", (PyCFunction)api_jsd, METH_FASTCALL | METH_KEYWORDS, doc_jsd},

    // Conventional `cdist` interface for pairwise distances
    {"cdist", (PyCFunction)api_cdist, METH_FASTCALL | METH_KEYWORDS, doc_cdist},

    // Exposing underlying API for USearch `CompiledMetric`
    {"pointer_to_euclidean", (PyCFunction)api_l2_pointer, METH_O, doc_l2_pointer},
    {"pointer_to_sqeuclidean", (PyCFunction)api_l2sq_pointer, METH_O, doc_l2sq_pointer},
    {"pointer_to_angular", (PyCFunction)api_angular_pointer, METH_O, doc_angular_pointer},
    {"pointer_to_inner", (PyCFunction)api_dot_pointer, METH_O, doc_dot_pointer},
    {"pointer_to_dot", (PyCFunction)api_dot_pointer, METH_O, doc_dot_pointer},
    {"pointer_to_vdot", (PyCFunction)api_vdot_pointer, METH_O, doc_vdot_pointer},
    {"pointer_to_kullbackleibler", (PyCFunction)api_kld_pointer, METH_O, doc_kld_pointer},
    {"pointer_to_jensenshannon", (PyCFunction)api_jsd_pointer, METH_O, doc_jsd_pointer},

    // Set operations
    {"intersect", (PyCFunction)api_intersect, METH_FASTCALL, doc_intersect},

    // Curved spaces
    {"bilinear", (PyCFunction)api_bilinear, METH_FASTCALL | METH_KEYWORDS, doc_bilinear},
    {"mahalanobis", (PyCFunction)api_mahalanobis, METH_FASTCALL | METH_KEYWORDS, doc_mahalanobis},

    // Geospatial distances
    {"haversine", (PyCFunction)api_haversine, METH_FASTCALL | METH_KEYWORDS, doc_haversine},
    {"vincenty", (PyCFunction)api_vincenty, METH_FASTCALL | METH_KEYWORDS, doc_vincenty},

    // NDArray constructors
    {"empty", (PyCFunction)api_empty, METH_FASTCALL | METH_KEYWORDS, doc_empty},
    {"zeros", (PyCFunction)api_zeros, METH_FASTCALL | METH_KEYWORDS, doc_zeros},
    {"ones", (PyCFunction)api_ones, METH_FASTCALL | METH_KEYWORDS, doc_ones},
    {"full", (PyCFunction)api_full, METH_FASTCALL | METH_KEYWORDS, doc_full},

    // NDArray reductions
    {"sum", (PyCFunction)api_sum, METH_FASTCALL | METH_KEYWORDS, doc_reduce_sum},
    {"min", (PyCFunction)api_min, METH_FASTCALL | METH_KEYWORDS, doc_reduce_min},
    {"max", (PyCFunction)api_max, METH_FASTCALL | METH_KEYWORDS, doc_reduce_max},
    {"argmin", (PyCFunction)api_argmin, METH_FASTCALL | METH_KEYWORDS, doc_reduce_argmin},
    {"argmax", (PyCFunction)api_argmax, METH_FASTCALL | METH_KEYWORDS, doc_reduce_argmax},

    // Vectorized operations
    {"fma", (PyCFunction)api_fma, METH_FASTCALL | METH_KEYWORDS, doc_fma},
    {"wsum", (PyCFunction)api_wsum, METH_FASTCALL | METH_KEYWORDS, doc_wsum},
    {"add", (PyCFunction)api_add, METH_FASTCALL | METH_KEYWORDS, doc_add},
    {"multiply", (PyCFunction)api_multiply, METH_FASTCALL | METH_KEYWORDS, doc_multiply},

    // Element-wise trigonometric functions
    {"sin", (PyCFunction)api_sin, METH_FASTCALL | METH_KEYWORDS, doc_sin},
    {"cos", (PyCFunction)api_cos, METH_FASTCALL | METH_KEYWORDS, doc_cos},
    {"atan", (PyCFunction)api_atan, METH_FASTCALL | METH_KEYWORDS, doc_atan},

    // Mesh alignment (point cloud registration)
    {"kabsch", (PyCFunction)api_kabsch, METH_FASTCALL | METH_KEYWORDS, doc_kabsch},
    {"umeyama", (PyCFunction)api_umeyama, METH_FASTCALL | METH_KEYWORDS, doc_umeyama},
    {"rmsd_mesh", (PyCFunction)api_rmsd_mesh, METH_FASTCALL | METH_KEYWORDS, doc_rmsd},

    // Matrix multiplication with pre-packed matrices
    {"pack_matmul_argument", (PyCFunction)api_pack_matmul_argument, METH_FASTCALL | METH_KEYWORDS,
     doc_pack_matmul_argument},
    {"pack_matrix", (PyCFunction)api_pack_matrix, METH_FASTCALL | METH_KEYWORDS, doc_pack_matrix},
    {"matmul", (PyCFunction)api_matmul, METH_FASTCALL | METH_KEYWORDS, doc_matmul},

    // Sentinel
    {NULL, NULL, 0, NULL}};

static char const doc_module[] = //
    "Portable mixed-precision BLAS-like vector math library for x86 and Arm.\n"
    "\n"
    "Performance Recommendations:\n"
    " - Avoid converting to NumPy arrays. NumKong works with any tensor implementation\n"
    "   compatible with the Python buffer protocol, including PyTorch and TensorFlow.\n"
    " - In low-latency environments, provide the output array with the `out=` parameter\n"
    "   to avoid expensive memory allocations on the hot path.\n"
    " - On modern CPUs, when the application allows, prefer low-precision numeric types.\n"
    "   Whenever possible, use 'bf16' and 'f16' over 'f32'. Consider quantizing to 'i8'\n"
    "   and 'u8' for highest hardware compatibility and performance.\n"
    " - If you only need relative proximity rather than absolute distance, prefer simpler\n"
    "   kernels such as squared Euclidean distance over Euclidean distance.\n"
    " - Use row-major contiguous matrix representations. Strides between rows do not have\n"
    "   a significant impact on performance, but most modern HPC packages explicitly ban\n"
    "   non-contiguous rows where nearby cells within a row have multi-byte gaps.\n"
    " - The CPython runtime has noticeable overhead for function calls, so consider batching\n"
    "   kernel invocations. Many kernels compute 1-to-1 distances between vectors, as well as\n"
    "   1-to-N and N-to-N distances between batches of vectors packed into matrices.\n"
    "\n"
    "Example:\n"
    "    >>> import numkong\n"
    "    >>> numkong.l2(a, b)\n"
    "\n"
    "Mixed-precision 1-to-N example with numeric types missing in NumPy, but present in PyTorch:\n"
    "    >>> import numkong\n"
    "    >>> import torch\n"
    "    >>> a = torch.randn(1536, dtype=torch.bfloat16)\n"
    "    >>> b = torch.randn((100, 1536), dtype=torch.bfloat16)\n"
    "    >>> c = torch.zeros(100, dtype=torch.float32)\n"
    "    >>> numkong.l2(a, b, dtype='bfloat16', out=c)\n";

static PyModuleDef nk_module = {
    PyModuleDef_HEAD_INIT, .m_name = "NumKong", .m_doc = doc_module, .m_size = -1, .m_methods = nk_methods,
};

PyMODINIT_FUNC PyInit_numkong(void) {
    PyObject *m;

    if (PyType_Ready(&NDArrayType) < 0) return NULL;
    if (PyType_Ready(&NDArrayIterType) < 0) return NULL;
    if (PyType_Ready(&PackedMatrixType) < 0) return NULL;

    m = PyModule_Create(&nk_module);
    if (m == NULL) return NULL;

#ifdef Py_GIL_DISABLED
    PyUnstable_Module_SetGIL(m, Py_MOD_GIL_NOT_USED);
#endif

    // Add version metadata
    {
        char version_str[64];
        snprintf(version_str, sizeof(version_str), "%d.%d.%d", NK_VERSION_MAJOR, NK_VERSION_MINOR, NK_VERSION_PATCH);
        PyModule_AddStringConstant(m, "__version__", version_str);
    }

    Py_INCREF(&NDArrayType);
    if (PyModule_AddObject(m, "NDArray", (PyObject *)&NDArrayType) < 0) {
        Py_XDECREF(&NDArrayType);
        Py_XDECREF(m);
        return NULL;
    }

    Py_INCREF(&PackedMatrixType);
    if (PyModule_AddObject(m, "PackedMatrix", (PyObject *)&PackedMatrixType) < 0) {
        Py_XDECREF(&PackedMatrixType);
        Py_XDECREF(m);
        return NULL;
    }

    static_capabilities = nk_capabilities();
    return m;
}

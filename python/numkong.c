/**
 *  @brief Pure CPython bindings for NumKong.
 *  @file python/numkong.c
 *  @author Ash Vardanian
 *  @date January 1, 2023
 *
 *  @section Latency, Quality, and Arguments Parsing
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
 *  @section Buffer Protocol and NumPy Compatibility
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

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>

#include <numkong/numkong.h>

#include "numkong.h"
#include "numerics.h"
#include "tensor.h"
#include "scalars.h"

nk_capability_t static_capabilities = 0;

/**
 *  @brief Extract a double value from a scalar buffer given its dtype.
 */
static double nk_scalar_buffer_get_f64(nk_scalar_buffer_t const *buf, nk_dtype_t dtype) {
    switch (dtype) {
    case nk_f64_k: return buf->f64;
    case nk_f32_k: return (double)buf->f32;
    case nk_f16_k: {
        nk_f32_t f32_tmp;
        nk_f16_to_f32(&buf->f16, &f32_tmp);
        return (double)f32_tmp;
    }
    case nk_bf16_k: {
        nk_f32_t f32_tmp;
        nk_bf16_to_f32(&buf->bf16, &f32_tmp);
        return (double)f32_tmp;
    }
    case nk_f64c_k: return buf->f64c.real;
    case nk_f32c_k: return (double)buf->f32c.real;
    case nk_i64_k: return (double)buf->i64;
    case nk_u64_k: return (double)buf->u64;
    case nk_i32_k: return (double)buf->i32;
    case nk_u32_k: return (double)buf->u32;
    case nk_i16_k: return (double)buf->i16;
    case nk_u16_k: return (double)buf->u16;
    case nk_i8_k: return (double)buf->i8;
    case nk_u8_k: return (double)buf->u8;
    default: return 0.0;
    }
}

/**
 *  @brief Store a double value into a scalar buffer given its dtype.
 */
static void nk_scalar_buffer_set_f64(nk_scalar_buffer_t *buf, double value, nk_dtype_t dtype) {
    switch (dtype) {
    case nk_f64_k: buf->f64 = value; break;
    case nk_f32_k: buf->f32 = (float)value; break;
    case nk_f16_k: {
        nk_f32_t f32_tmp = (nk_f32_t)value;
        nk_f32_to_f16(&f32_tmp, &buf->f16);
        break;
    }
    case nk_bf16_k: {
        nk_f32_t f32_tmp = (nk_f32_t)value;
        nk_f32_to_bf16(&f32_tmp, &buf->bf16);
        break;
    }
    case nk_f64c_k:
        buf->f64c.real = value;
        buf->f64c.imag = 0;
        break;
    case nk_f32c_k:
        buf->f32c.real = (nk_f32_t)value;
        buf->f32c.imag = 0;
        break;
    case nk_i64_k: buf->i64 = (nk_i64_t)value; break;
    case nk_u64_k: buf->u64 = (nk_u64_t)value; break;
    case nk_i32_k: buf->i32 = (nk_i32_t)value; break;
    case nk_u32_k: buf->u32 = (nk_u32_t)value; break;
    case nk_i16_k: buf->i16 = (nk_i16_t)value; break;
    case nk_u16_k: buf->u16 = (nk_u16_t)value; break;
    case nk_i8_k: buf->i8 = (nk_i8_t)value; break;
    case nk_u8_k: buf->u8 = (nk_u8_t)value; break;
    default: break;
    }
}

#pragma region Datatype Metadata Table

nk_dtype_info_t const nk_dtype_table[] = {
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
    {nk_u1_k, "bin8", "?", "|V1", sizeof(nk_u1x8_t), 0},
    {nk_i8_k, "int8", "b", "|i1", sizeof(nk_i8_t), 0},
    {nk_u8_k, "uint8", "B", "|u1", sizeof(nk_u8_t), 0},
    {nk_i16_k, "int16", "h", "<i2", sizeof(nk_i16_t), 0},
    {nk_u16_k, "uint16", "H", "<u2", sizeof(nk_u16_t), 0},
    {nk_i32_k, "int32", "i", "<i4", sizeof(nk_i32_t), 0},
    {nk_u32_k, "uint32", "I", "<u4", sizeof(nk_u32_t), 0},
    {nk_i64_k, "int64", "q", "<i8", sizeof(nk_i64_t), 0},
    {nk_u64_k, "uint64", "Q", "<u8", sizeof(nk_u64_t), 0},
};

size_t const nk_dtype_table_size = sizeof(nk_dtype_table) / sizeof(nk_dtype_table[0]);

#pragma endregion // Datatype Metadata Table

#pragma region Datatype Utilities

nk_dtype_info_t const *dtype_info(nk_dtype_t dtype) {
    for (size_t i = 0; i < nk_dtype_table_size; i++) {
        if (nk_dtype_table[i].dtype == dtype) return &nk_dtype_table[i];
    }
    return NULL;
}

size_t bytes_per_dtype(nk_dtype_t dtype) {
    nk_dtype_info_t const *info = dtype_info(dtype);
    return info ? info->item_size : 0;
}

char const *dtype_to_string(nk_dtype_t dtype) {
    nk_dtype_info_t const *info = dtype_info(dtype);
    return info ? info->name : "unknown";
}

char const *dtype_to_array_typestr(nk_dtype_t dtype) {
    nk_dtype_info_t const *info = dtype_info(dtype);
    return info ? info->array_typestr : "|V1";
}

char const *dtype_to_python_string(nk_dtype_t dtype) {
    nk_dtype_info_t const *info = dtype_info(dtype);
    return info ? info->buffer_format : "unknown";
}

int same_string(char const *a, char const *b) { return strcmp(a, b) == 0; }

int is_complex(nk_dtype_t dtype) {
    nk_dtype_info_t const *info = dtype_info(dtype);
    return info ? info->is_complex : 0;
}

nk_dtype_t python_string_to_dtype(char const *name) {
    // Floating-point numbers:
    if (same_string(name, "float32") || same_string(name, "f4") || same_string(name, "<f4") || same_string(name, "f") ||
        same_string(name, "<f"))
        return nk_f32_k;
    else if (same_string(name, "float16") || same_string(name, "f2") || same_string(name, "<f2") ||
             same_string(name, "e") || same_string(name, "<e"))
        return nk_f16_k;
    else if (same_string(name, "float64") || same_string(name, "f8") || same_string(name, "<f8") ||
             same_string(name, "d") || same_string(name, "<d"))
        return nk_f64_k;
    else if (same_string(name, "bfloat16") || same_string(name, "bf16")) return nk_bf16_k;

    // FP8 formats (ML-focused 8-bit floats):
    else if (same_string(name, "e4m3")) return nk_e4m3_k;
    else if (same_string(name, "e5m2")) return nk_e5m2_k;

    // Complex numbers:
    else if (same_string(name, "complex64") || same_string(name, "F4") || same_string(name, "<F4") ||
             same_string(name, "Zf") || same_string(name, "F") || same_string(name, "<F"))
        return nk_f32c_k;
    else if (same_string(name, "complex128") || same_string(name, "F8") || same_string(name, "<F8") ||
             same_string(name, "Zd") || same_string(name, "D") || same_string(name, "<D"))
        return nk_f64c_k;
    else if (same_string(name, "complex32") || same_string(name, "F2") || same_string(name, "<F2") ||
             same_string(name, "Ze") || same_string(name, "E") || same_string(name, "<E"))
        return nk_f16c_k;
    else if (same_string(name, "bcomplex32") || same_string(name, "bfloat16c") || same_string(name, "bf16c"))
        return nk_bf16c_k;

    // Boolean values:
    else if (same_string(name, "bin8") || same_string(name, "?")) return nk_u1_k;

    // Signed integers:
    else if (same_string(name, "int8") || same_string(name, "i1") || same_string(name, "|i1") ||
             same_string(name, "<i1") || same_string(name, "b") || same_string(name, "<b"))
        return nk_i8_k;
    else if (same_string(name, "int16") || same_string(name, "i2") || same_string(name, "|i2") ||
             same_string(name, "<i2") || same_string(name, "h") || same_string(name, "<h"))
        return nk_i16_k;

    // Platform-specific integer formats (Windows vs Unix):
#if defined(_MSC_VER) || defined(__i386__)
    else if (same_string(name, "int32") || same_string(name, "i4") || same_string(name, "|i4") ||
             same_string(name, "<i4") || same_string(name, "l") || same_string(name, "<l"))
        return nk_i32_k;
    else if (same_string(name, "int64") || same_string(name, "i8") || same_string(name, "|i8") ||
             same_string(name, "<i8") || same_string(name, "q") || same_string(name, "<q"))
        return nk_i64_k;
#else
    else if (same_string(name, "int32") || same_string(name, "i4") || same_string(name, "|i4") ||
             same_string(name, "<i4") || same_string(name, "i") || same_string(name, "<i"))
        return nk_i32_k;
    else if (same_string(name, "int64") || same_string(name, "i8") || same_string(name, "|i8") ||
             same_string(name, "<i8") || same_string(name, "l") || same_string(name, "<l"))
        return nk_i64_k;
#endif

    // Unsigned integers:
    else if (same_string(name, "uint8") || same_string(name, "u1") || same_string(name, "|u1") ||
             same_string(name, "<u1") || same_string(name, "B") || same_string(name, "<B"))
        return nk_u8_k;
    else if (same_string(name, "uint16") || same_string(name, "u2") || same_string(name, "|u2") ||
             same_string(name, "<u2") || same_string(name, "H") || same_string(name, "<H"))
        return nk_u16_k;

#if defined(_MSC_VER) || defined(__i386__)
    else if (same_string(name, "uint32") || same_string(name, "u4") || same_string(name, "|u4") ||
             same_string(name, "<u4") || same_string(name, "L") || same_string(name, "<L"))
        return nk_u32_k;
    else if (same_string(name, "uint64") || same_string(name, "u8") || same_string(name, "|u8") ||
             same_string(name, "<u8") || same_string(name, "Q") || same_string(name, "<Q"))
        return nk_u64_k;
#else
    else if (same_string(name, "uint32") || same_string(name, "u4") || same_string(name, "|u4") ||
             same_string(name, "<u4") || same_string(name, "I") || same_string(name, "<I"))
        return nk_u32_k;
    else if (same_string(name, "uint64") || same_string(name, "u8") || same_string(name, "|u8") ||
             same_string(name, "<u8") || same_string(name, "L") || same_string(name, "<L"))
        return nk_u64_k;
#endif

    else return nk_dtype_unknown_k;
}

nk_kernel_kind_t python_string_to_metric_kind(char const *name) {
    if (same_string(name, "euclidean")) return nk_kernel_euclidean_k;
    else if (same_string(name, "sqeuclidean")) return nk_kernel_sqeuclidean_k;
    else if (same_string(name, "dot") || same_string(name, "inner")) return nk_kernel_dot_k;
    else if (same_string(name, "vdot")) return nk_kernel_vdot_k;
    else if (same_string(name, "angular")) return nk_kernel_angular_k;
    else if (same_string(name, "jaccard")) return nk_kernel_jaccard_k;
    else if (same_string(name, "kullbackleibler") || same_string(name, "kld")) return nk_kernel_kld_k;
    else if (same_string(name, "jensenshannon") || same_string(name, "jsd")) return nk_kernel_jsd_k;
    else if (same_string(name, "hamming")) return nk_kernel_hamming_k;
    else if (same_string(name, "bilinear")) return nk_kernel_bilinear_k;
    else if (same_string(name, "mahalanobis")) return nk_kernel_mahalanobis_k;
    else return nk_kernel_unknown_k;
}

int cast_distance(nk_f64_t distance, nk_dtype_t target_dtype, void *target_ptr, size_t offset) {
    nk_f32_t f32_val;
    switch (target_dtype) {
    case nk_f64c_k: // fallthrough
    case nk_f64_k: ((nk_f64_t *)target_ptr)[offset] = distance; return 1;
    case nk_f32c_k: // fallthrough
    case nk_f32_k: ((nk_f32_t *)target_ptr)[offset] = (nk_f32_t)distance; return 1;
    case nk_f16c_k: // fallthrough
    case nk_f16_k:
        f32_val = (nk_f32_t)distance;
        nk_f32_to_f16(&f32_val, (nk_f16_t *)target_ptr + offset);
        return 1;
    case nk_bf16c_k: // fallthrough
    case nk_bf16_k:
        f32_val = (nk_f32_t)distance;
        nk_f32_to_bf16(&f32_val, (nk_bf16_t *)target_ptr + offset);
        return 1;
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

/**
 *  @brief Convert a scalar buffer to the appropriate Python number type.
 */
static PyObject *scalar_to_py_number(nk_scalar_buffer_t const *buf, nk_dtype_t dtype) {
    switch (dtype) {
    case nk_f64_k: return PyFloat_FromDouble(buf->f64);
    case nk_f32_k: return PyFloat_FromDouble((double)buf->f32);
    case nk_f16_k: {
        nk_f32_t f32_tmp;
        nk_f16_to_f32(&buf->f16, &f32_tmp);
        return PyFloat_FromDouble((double)f32_tmp);
    }
    case nk_bf16_k: {
        nk_f32_t f32_tmp;
        nk_bf16_to_f32(&buf->bf16, &f32_tmp);
        return PyFloat_FromDouble((double)f32_tmp);
    }
    case nk_f64c_k: return PyComplex_FromDoubles(buf->f64c.real, buf->f64c.imag);
    case nk_f32c_k: return PyComplex_FromDoubles((double)buf->f32c.real, (double)buf->f32c.imag);
    case nk_i64_k: return PyLong_FromLongLong(buf->i64);
    case nk_u64_k: return PyLong_FromUnsignedLongLong(buf->u64);
    case nk_i32_k: return PyLong_FromLong(buf->i32);
    case nk_u32_k: return PyLong_FromUnsignedLong(buf->u32);
    case nk_i16_k: return PyLong_FromLong(buf->i16);
    case nk_u16_k: return PyLong_FromUnsignedLong(buf->u16);
    case nk_i8_k: return PyLong_FromLong(buf->i8);
    case nk_u8_k: return PyLong_FromUnsignedLong(buf->u8);
    default: return PyFloat_FromDouble(0.0);
    }
}

/**
 *  @brief Store a Python number (float, int, or complex) into a scalar buffer.
 *  @return 1 on success, 0 on error (with Python exception set).
 */
static int py_number_to_scalar_buffer(PyObject *obj, nk_scalar_buffer_t *buf, nk_dtype_t dtype) {
    memset(buf, 0, sizeof(*buf));
    if (is_complex(dtype) && PyComplex_Check(obj)) {
        Py_complex py_complex = PyComplex_AsCComplex(obj);
        switch (dtype) {
        case nk_f64c_k:
            buf->f64c.real = py_complex.real;
            buf->f64c.imag = py_complex.imag;
            return 1;
        case nk_f32c_k:
            buf->f32c.real = (float)py_complex.real;
            buf->f32c.imag = (float)py_complex.imag;
            return 1;
        default: break;
        }
    }
    double value = PyFloat_AsDouble(obj);
    if (PyErr_Occurred()) return 0;
    nk_scalar_buffer_set_f64(buf, value, dtype);
    return 1;
}

/**
 *  @brief Write a scalar buffer result (including complex) to a numpy output array element.
 *  @return 1 on success, 0 on error.
 */
static int cast_scalar_buffer(nk_scalar_buffer_t const *buf, nk_dtype_t src_dtype, nk_dtype_t dst_dtype, void *target) {
    if (is_complex(src_dtype)) {
        double real, imag;
        switch (src_dtype) {
        case nk_f64c_k:
            real = buf->f64c.real;
            imag = buf->f64c.imag;
            break;
        case nk_f32c_k:
            real = (double)buf->f32c.real;
            imag = (double)buf->f32c.imag;
            break;
        default: return 0;
        }
        if (!cast_distance(real, dst_dtype, target, 0)) return 0;
        return cast_distance(imag, dst_dtype, target, 1);
    }
    return cast_distance(nk_scalar_buffer_get_f64(buf, src_dtype), dst_dtype, target, 0);
}

int kernel_is_commutative(nk_kernel_kind_t kind) {
    switch (kind) {
    case nk_kernel_kld_k: return 0;
    case nk_kernel_bilinear_k: return 0;
    default: return 1;
    }
}

#pragma endregion // Datatype Utilities

#pragma region Buffer Protocol Helpers

int is_scalar(PyObject *obj) { return PyFloat_Check(obj) || PyLong_Check(obj); }

int get_scalar_value(PyObject *obj, double *value) {
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

int parse_tensor(PyObject *tensor, Py_buffer *buffer, TensorArgument *parsed) {
    if (PyObject_GetBuffer(tensor, buffer, PyBUF_STRIDES | PyBUF_FORMAT) != 0) {
        PyErr_SetString(PyExc_TypeError, "arguments must support buffer protocol");
        return 0;
    }

    parsed->start = buffer->buf;
    parsed->dtype = python_string_to_dtype(buffer->format);
    if (parsed->dtype == nk_dtype_unknown_k) {
        PyErr_Format(PyExc_ValueError, "Unsupported '%s' dtype specifier", buffer->format);
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

#pragma endregion // Buffer Protocol Helpers

static char const doc_enable_capability[] =     //
    "Enable a specific SIMD kernel family.\n\n" //
    "Parameters:\n"                             //
    "    capability : str\n"                    //
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
    // RISC-V capabilities
    else if (same_string(cap_name, "rvv")) { static_capabilities |= nk_cap_rvv_k; }
    else if (same_string(cap_name, "rvvhalf")) { static_capabilities |= nk_cap_rvvhalf_k; }
    else if (same_string(cap_name, "rvvbf16")) { static_capabilities |= nk_cap_rvvbf16_k; }
    else if (same_string(cap_name, "rvvbb")) { static_capabilities |= nk_cap_rvvbb_k; }
    // WASM capabilities
    else if (same_string(cap_name, "v128relaxed")) { static_capabilities |= nk_cap_v128relaxed_k; }
    // x86 capabilities
    else if (same_string(cap_name, "haswell")) { static_capabilities |= nk_cap_haswell_k; }
    else if (same_string(cap_name, "skylake")) { static_capabilities |= nk_cap_skylake_k; }
    else if (same_string(cap_name, "icelake")) { static_capabilities |= nk_cap_icelake_k; }
    else if (same_string(cap_name, "genoa")) { static_capabilities |= nk_cap_genoa_k; }
    else if (same_string(cap_name, "sapphire")) { static_capabilities |= nk_cap_sapphire_k; }
    else if (same_string(cap_name, "sapphireamx")) { static_capabilities |= nk_cap_sapphireamx_k; }
    else if (same_string(cap_name, "graniteamx")) { static_capabilities |= nk_cap_graniteamx_k; }
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

static char const doc_disable_capability[] =     //
    "Disable a specific SIMD kernel family.\n\n" //
    "Parameters:\n"                              //
    "    capability : str\n"                     //
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
    // RISC-V capabilities
    else if (same_string(cap_name, "rvv")) { static_capabilities &= ~nk_cap_rvv_k; }
    else if (same_string(cap_name, "rvvhalf")) { static_capabilities &= ~nk_cap_rvvhalf_k; }
    else if (same_string(cap_name, "rvvbf16")) { static_capabilities &= ~nk_cap_rvvbf16_k; }
    else if (same_string(cap_name, "rvvbb")) { static_capabilities &= ~nk_cap_rvvbb_k; }
    // WASM capabilities
    else if (same_string(cap_name, "v128relaxed")) { static_capabilities &= ~nk_cap_v128relaxed_k; }
    // x86 capabilities
    else if (same_string(cap_name, "haswell")) { static_capabilities &= ~nk_cap_haswell_k; }
    else if (same_string(cap_name, "skylake")) { static_capabilities &= ~nk_cap_skylake_k; }
    else if (same_string(cap_name, "icelake")) { static_capabilities &= ~nk_cap_icelake_k; }
    else if (same_string(cap_name, "genoa")) { static_capabilities &= ~nk_cap_genoa_k; }
    else if (same_string(cap_name, "sapphire")) { static_capabilities &= ~nk_cap_sapphire_k; }
    else if (same_string(cap_name, "sapphireamx")) { static_capabilities &= ~nk_cap_sapphireamx_k; }
    else if (same_string(cap_name, "graniteamx")) { static_capabilities &= ~nk_cap_graniteamx_k; }
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

static char const doc_get_capabilities[] =                                                           //
    "Get the current hardware SIMD capabilities as a dictionary of feature flags.\n"                 //
    "On x86 it includes: 'serial', 'haswell', 'skylake', 'icelake', 'genoa', 'sapphire', 'turin'.\n" //
    "On Arm it includes: 'serial', 'neon', 'sve', 'sve2', and their extensions.\n";

static PyObject *api_get_capabilities(PyObject *self) {
    nk_capability_t caps = static_capabilities;
    PyObject *cap_dict = PyDict_New();
    if (!cap_dict) return NULL;

#define ADD_CAP(name) PyDict_SetItemString(cap_dict, #name, PyBool_FromLong((caps) & nk_cap_##name##_k))

    // Always available
    ADD_CAP(serial);
    // ARM NEON capabilities
    ADD_CAP(neon);
    ADD_CAP(neonhalf);
    ADD_CAP(neonsdot);
    ADD_CAP(neonfhm);
    ADD_CAP(neonbfdot);
    // ARM SVE capabilities
    ADD_CAP(sve);
    ADD_CAP(svehalf);
    ADD_CAP(svesdot);
    ADD_CAP(svebfdot);
    ADD_CAP(sve2);
    ADD_CAP(sve2p1);
    // ARM SME capabilities
    ADD_CAP(sme);
    ADD_CAP(sme2);
    ADD_CAP(sme2p1);
    ADD_CAP(smef64);
    ADD_CAP(smehalf);
    ADD_CAP(smebf16);
    ADD_CAP(smelut2);
    ADD_CAP(smefa64);
    // x86 capabilities
    ADD_CAP(haswell);
    ADD_CAP(skylake);
    ADD_CAP(icelake);
    ADD_CAP(genoa);
    ADD_CAP(sapphire);
    ADD_CAP(sapphireamx);
    ADD_CAP(graniteamx);
    ADD_CAP(turin);
    ADD_CAP(sierra);
    // RISC-V capabilities
    ADD_CAP(rvv);
    ADD_CAP(rvvhalf);
    ADD_CAP(rvvbf16);
    ADD_CAP(rvvbb);
    // WASM capabilities
    ADD_CAP(v128relaxed);

#undef ADD_CAP

    return cap_dict;
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
    nk_dtype_t dtype = nk_dtype_unknown_k, out_dtype = nk_dtype_unknown_k;
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
        dtype = python_string_to_dtype(dtype_str);
        if (dtype == nk_dtype_unknown_k) {
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
        out_dtype = python_string_to_dtype(out_dtype_str);
        if (out_dtype == nk_dtype_unknown_k) {
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
    if (a_parsed.dtype != b_parsed.dtype || //
        a_parsed.dtype == nk_dtype_unknown_k || b_parsed.dtype == nk_dtype_unknown_k) {
        PyErr_SetString(PyExc_TypeError, "Input tensors must have matching dtypes, check with `X.__array_interface__`");
        goto cleanup;
    }
    if (dtype == nk_dtype_unknown_k) dtype = a_parsed.dtype;

    // Inference order for the output type:
    // 1. `out_dtype` named argument, if defined
    // 2. `out.dtype` attribute, if `out` is passed
    // 3. double precision float (or its complex variant)
    if (out_dtype == nk_dtype_unknown_k) {
        if (out_obj) { out_dtype = out_parsed.dtype; }
        else { out_dtype = is_complex(dtype) ? nk_f64c_k : nk_f64_k; }
    }

    // Make sure the return dtype is complex if the input dtype is complex, and the same for real numbers
    if (out_dtype != nk_dtype_unknown_k) {
        if (is_complex(dtype) != is_complex(out_dtype)) {
            PyErr_SetString(PyExc_ValueError,
                            "If the input dtype is complex, the return dtype must be complex, and same for real.");
            goto cleanup;
        }
    }

    // Check if the downcasting to provided dtype is supported
    {
        char returned_buffer_example[8];
        if (!cast_distance(0, out_dtype, &returned_buffer_example, 0)) {
            PyErr_SetString(PyExc_ValueError, "Exporting to the provided dtype is not supported");
            goto cleanup;
        }
    }

    // Look up the metric and the capability
    nk_metric_dense_punned_t metric = NULL;
    nk_capability_t capability = nk_cap_serial_k;
    nk_find_kernel_punned(metric_kind, dtype, static_capabilities, nk_cap_any_k, (nk_kernel_punned_t *)&metric,
                          &capability);
    if (!metric || !capability) {
        PyErr_Format( //
            PyExc_LookupError,
            "Unsupported metric '%c' and dtype combination across vectors ('%s'/'%s' and '%s'/'%s') and " //
            "`dtype` override ('%s'/'%s')",
            metric_kind,                                                                       //
            a_buffer.format ? a_buffer.format : "nil", dtype_to_python_string(a_parsed.dtype), //
            b_buffer.format ? b_buffer.format : "nil", dtype_to_python_string(b_parsed.dtype), //
            dtype_str ? dtype_str : "nil", dtype_to_python_string(dtype));
        goto cleanup;
    }

    // If the distance is computed between two vectors, rather than matrices, return a scalar
    nk_dtype_t const kernel_out_dtype = nk_kernel_output_dtype(metric_kind, dtype);
    if (a_parsed.rank == 1 && b_parsed.rank == 1) {
        nk_scalar_buffer_t distance;
        metric(a_parsed.start, b_parsed.start, a_parsed.dimensions, &distance);
        return_obj = scalar_to_py_number(&distance, kernel_out_dtype);
        goto cleanup;
    }

    // In some batch requests we may be computing the distance from multiple vectors to one,
    // so the stride must be set to zero avoid illegal memory access
    if (a_parsed.count == 1) a_parsed.stride = 0;
    if (b_parsed.count == 1) b_parsed.stride = 0;

    // We take the maximum of the two counts, because if only one entry is present in one of the arrays,
    // all distances will be computed against that single entry.
    size_t const count_pairs = a_parsed.count > b_parsed.count ? a_parsed.count : b_parsed.count;
    char *distances_start = NULL;
    size_t distances_stride_bytes = 0;

    // Allocate the output matrix if it wasn't provided
    if (!out_obj) {
        Tensor *distances_obj = PyObject_NewVar(Tensor, &TensorType, count_pairs * bytes_per_dtype(out_dtype));
        if (!distances_obj) {
            PyErr_NoMemory();
            goto cleanup;
        }

        // Initialize the object
        distances_obj->dtype = out_dtype;
        distances_obj->rank = 1;
        distances_obj->shape[0] = count_pairs;
        distances_obj->shape[1] = 0;
        distances_obj->strides[0] = bytes_per_dtype(out_dtype);
        distances_obj->strides[1] = 0;
        distances_obj->parent = NULL;
        distances_obj->data = distances_obj->start;
        return_obj = (PyObject *)distances_obj;
        distances_start = distances_obj->data;
        distances_stride_bytes = distances_obj->strides[0];
    }
    else {
        if (bytes_per_dtype(out_parsed.dtype) != bytes_per_dtype(out_dtype)) {
            PyErr_Format( //
                PyExc_LookupError,
                "Output tensor scalar type must be compatible with the output type ('%s' and '%s'/'%s')",
                dtype_to_python_string(out_dtype), out_buffer.format ? out_buffer.format : "nil",
                dtype_to_python_string(out_parsed.dtype));
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
        nk_scalar_buffer_t result;
        metric(                                   //
            a_parsed.start + i * a_parsed.stride, //
            b_parsed.start + i * b_parsed.stride, //
            a_parsed.dimensions,                  //
            &result);

        // Export out:
        cast_scalar_buffer(&result, kernel_out_dtype, out_dtype, distances_start + i * distances_stride_bytes);
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
    nk_dtype_t dtype = nk_dtype_unknown_k;
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
        dtype = python_string_to_dtype(dtype_str);
        if (dtype == nk_dtype_unknown_k) {
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
    if (a_parsed.dtype != b_parsed.dtype || a_parsed.dtype != c_parsed.dtype || a_parsed.dtype == nk_dtype_unknown_k ||
        b_parsed.dtype == nk_dtype_unknown_k || c_parsed.dtype == nk_dtype_unknown_k) {
        PyErr_SetString(PyExc_TypeError, "Input tensors must have matching dtypes, check with `X.__array_interface__`");
        goto cleanup;
    }
    if (dtype == nk_dtype_unknown_k) dtype = a_parsed.dtype;

    // Look up the metric and the capability
    nk_metric_curved_punned_t metric = NULL;
    nk_capability_t capability = nk_cap_serial_k;
    nk_find_kernel_punned(metric_kind, dtype, static_capabilities, nk_cap_any_k, (nk_kernel_punned_t *)&metric,
                          &capability);
    if (!metric || !capability) {
        PyErr_Format( //
            PyExc_LookupError,
            "Unsupported metric '%c' and dtype combination across vectors ('%s'/'%s' and '%s'/'%s'), " //
            "tensor ('%s'/'%s'), and `dtype` override ('%s'/'%s')",
            metric_kind,                                                                       //
            a_buffer.format ? a_buffer.format : "nil", dtype_to_python_string(a_parsed.dtype), //
            b_buffer.format ? b_buffer.format : "nil", dtype_to_python_string(b_parsed.dtype), //
            c_buffer.format ? c_buffer.format : "nil", dtype_to_python_string(c_parsed.dtype), //
            dtype_str ? dtype_str : "nil", dtype_to_python_string(dtype));
        goto cleanup;
    }

    // Return a scalar
    nk_dtype_t const kernel_out_dtype = nk_kernel_output_dtype(metric_kind, dtype);
    nk_scalar_buffer_t distance;
    metric(a_parsed.start, b_parsed.start, c_parsed.start, a_parsed.dimensions, &distance);
    return_obj = scalar_to_py_number(&distance, kernel_out_dtype);

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
    nk_dtype_t dtype = nk_dtype_unknown_k;
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
        dtype = python_string_to_dtype(dtype_str);
        if (dtype == nk_dtype_unknown_k) {
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
    if (a_lats_parsed.dtype != a_lons_parsed.dtype || a_lats_parsed.dtype != b_lats_parsed.dtype ||
        a_lats_parsed.dtype != b_lons_parsed.dtype || a_lats_parsed.dtype == nk_dtype_unknown_k) {
        PyErr_SetString(PyExc_TypeError, "All coordinate arrays must have the same dtype");
        goto cleanup;
    }
    if (dtype == nk_dtype_unknown_k) dtype = a_lats_parsed.dtype;

    // Look up the metric kernel
    nk_metric_geospatial_punned_t metric = NULL;
    nk_capability_t capability = nk_cap_serial_k;
    nk_find_kernel_punned(metric_kind, dtype, static_capabilities, nk_cap_any_k, (nk_kernel_punned_t *)&metric,
                          &capability);
    if (!metric || !capability) {
        PyErr_Format(PyExc_LookupError, "Unsupported metric '%c' and dtype '%s'", metric_kind,
                     dtype_to_python_string(dtype));
        goto cleanup;
    }

    // Allocate output or use provided
    // Output dtype must match input dtype (f32 kernel writes f32, f64 kernel writes f64)
    size_t const item_size = bytes_per_dtype(dtype);
    void *distances_start = NULL;
    if (!out_obj) {
        Tensor *distances_obj = PyObject_NewVar(Tensor, &TensorType, n * item_size);
        if (!distances_obj) {
            PyErr_NoMemory();
            goto cleanup;
        }
        distances_obj->dtype = dtype;
        distances_obj->rank = 1;
        distances_obj->shape[0] = n;
        distances_obj->shape[1] = 0;
        distances_obj->strides[0] = item_size;
        distances_obj->strides[1] = 0;
        distances_obj->parent = NULL;
        distances_obj->data = distances_obj->start;
        return_obj = (PyObject *)distances_obj;
        distances_start = distances_obj->data;
    }
    else {
        if (out_parsed.dimensions < n) {
            PyErr_SetString(PyExc_ValueError, "Output array is too small");
            goto cleanup;
        }
        distances_start = out_parsed.start;
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
    if (a_parsed.dtype != b_parsed.dtype && a_parsed.dtype != nk_dtype_unknown_k &&
        b_parsed.dtype != nk_dtype_unknown_k) {
        PyErr_SetString(PyExc_TypeError, "Input tensors must have matching dtypes, check with `X.__array_interface__`");
        goto cleanup;
    }

    nk_dtype_t dtype = a_parsed.dtype;
    nk_sparse_intersect_punned_t metric = NULL;
    nk_capability_t capability = nk_cap_serial_k;
    nk_find_kernel_punned(metric_kind, dtype, static_capabilities, nk_cap_any_k, (nk_kernel_punned_t *)&metric,
                          &capability);
    if (!metric || !capability) {
        PyErr_Format( //
            PyExc_LookupError, "Unsupported metric '%c' and dtype combination ('%s'/'%s' and '%s'/'%s')",
            metric_kind,                                                                       //
            a_buffer.format ? a_buffer.format : "nil", dtype_to_python_string(a_parsed.dtype), //
            b_buffer.format ? b_buffer.format : "nil", dtype_to_python_string(b_parsed.dtype));
        goto cleanup;
    }

    nk_f64_t distance;
    nk_size_t count = 0;
    metric(a_parsed.start, b_parsed.start, a_parsed.dimensions, b_parsed.dimensions, &distance, &count);
    return_obj = PyFloat_FromDouble(distance);

cleanup:
    PyBuffer_Release(&a_buffer);
    PyBuffer_Release(&b_buffer);
    return return_obj;
}

static PyObject *implement_cdist(                        //
    PyObject *a_obj, PyObject *b_obj, PyObject *out_obj, //
    nk_kernel_kind_t metric_kind, size_t threads,        //
    nk_dtype_t dtype, nk_dtype_t out_dtype) {

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
    if (a_parsed.dtype != b_parsed.dtype || //
        a_parsed.dtype == nk_dtype_unknown_k || b_parsed.dtype == nk_dtype_unknown_k) {
        PyErr_SetString(PyExc_TypeError, "Input tensors must have matching dtypes, check with `X.__array_interface__`");
        goto cleanup;
    }
    if (dtype == nk_dtype_unknown_k) dtype = a_parsed.dtype;

    // Inference order for the output type:
    // 1. `out_dtype` named argument, if defined
    // 2. `out.dtype` attribute, if `out` is passed
    // 3. double precision float (or its complex variant)
    if (out_dtype == nk_dtype_unknown_k) {
        if (out_obj) { out_dtype = out_parsed.dtype; }
        else { out_dtype = is_complex(dtype) ? nk_f64c_k : nk_f64_k; }
    }

    // Make sure the return dtype is complex if the input dtype is complex, and the same for real numbers
    if (out_dtype != nk_dtype_unknown_k) {
        if (is_complex(dtype) != is_complex(out_dtype)) {
            PyErr_SetString(PyExc_ValueError,
                            "If the input dtype is complex, the return dtype must be complex, and same for real.");
            goto cleanup;
        }
    }

    // Check if the downcasting to provided dtype is supported
    {
        char returned_buffer_example[8];
        if (!cast_distance(0, out_dtype, &returned_buffer_example, 0)) {
            PyErr_SetString(PyExc_ValueError, "Exporting to the provided dtype is not supported");
            goto cleanup;
        }
    }

    // Look up the metric and the capability
    nk_metric_dense_punned_t metric = NULL;
    nk_capability_t capability = nk_cap_serial_k;
    nk_find_kernel_punned(metric_kind, dtype, static_capabilities, nk_cap_any_k, (nk_kernel_punned_t *)&metric,
                          &capability);
    if (!metric || !capability) {
        PyErr_Format( //
            PyExc_LookupError, "Unsupported metric '%c' and dtype combination ('%s'/'%s' and '%s'/'%s')",
            metric_kind,                                                                       //
            a_buffer.format ? a_buffer.format : "nil", dtype_to_python_string(a_parsed.dtype), //
            b_buffer.format ? b_buffer.format : "nil", dtype_to_python_string(b_parsed.dtype));
        goto cleanup;
    }

    // If the distance is computed between two vectors, rather than matrices, return a scalar
    nk_dtype_t const kernel_out_dtype = nk_kernel_output_dtype(metric_kind, dtype);
    if (a_parsed.rank == 1 && b_parsed.rank == 1) {
        nk_scalar_buffer_t distance;
        metric(a_parsed.start, b_parsed.start, a_parsed.dimensions, &distance);
        return_obj = scalar_to_py_number(&distance, kernel_out_dtype);
        goto cleanup;
    }

#if defined(__linux__)
#if defined(_OPENMP)
    if (threads == 0) threads = omp_get_num_procs();
    omp_set_num_threads(threads);
#endif
#endif

    size_t const count_pairs = a_parsed.count * b_parsed.count;
    char *distances_start = NULL;
    size_t distances_rows_stride_bytes = 0;
    size_t distances_cols_stride_bytes = 0;

    // Allocate the output matrix if it wasn't provided
    if (!out_obj) {

        Tensor *distances_obj = PyObject_NewVar(Tensor, &TensorType, count_pairs * bytes_per_dtype(out_dtype));
        if (!distances_obj) {
            PyErr_NoMemory();
            goto cleanup;
        }

        // Initialize the object
        distances_obj->dtype = out_dtype;
        distances_obj->rank = 2;
        distances_obj->shape[0] = a_parsed.count;
        distances_obj->shape[1] = b_parsed.count;
        distances_obj->strides[0] = b_parsed.count * bytes_per_dtype(distances_obj->dtype);
        distances_obj->strides[1] = bytes_per_dtype(distances_obj->dtype);
        distances_obj->parent = NULL;
        distances_obj->data = distances_obj->start;
        return_obj = (PyObject *)distances_obj;
        distances_start = distances_obj->data;
        distances_rows_stride_bytes = distances_obj->strides[0];
        distances_cols_stride_bytes = distances_obj->strides[1];
    }
    else {
        if (bytes_per_dtype(out_parsed.dtype) != bytes_per_dtype(out_dtype)) {
            PyErr_Format( //
                PyExc_LookupError,
                "Output tensor scalar type must be compatible with the output type ('%s' and '%s'/'%s')",
                dtype_to_python_string(out_dtype), out_buffer.format ? out_buffer.format : "nil",
                dtype_to_python_string(out_parsed.dtype));
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
            nk_scalar_buffer_t result;
            metric(                                   //
                a_parsed.start + i * a_parsed.stride, //
                b_parsed.start + j * b_parsed.stride, //
                a_parsed.dimensions,                  //
                &result                               //
            );

            // Export into both the lower and upper triangle
            char *ptr_ij = distances_start + i * distances_rows_stride_bytes + j * distances_cols_stride_bytes;
            cast_scalar_buffer(&result, kernel_out_dtype, out_dtype, ptr_ij);
            if (is_symmetric) {
                char *ptr_ji = distances_start + j * distances_rows_stride_bytes + i * distances_cols_stride_bytes;
                cast_scalar_buffer(&result, kernel_out_dtype, out_dtype, ptr_ji);
            }
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

    nk_dtype_t dtype = python_string_to_dtype(dtype_name);
    if (!dtype) { // Check the actual variable here instead of dtype_name.
        PyErr_SetString(PyExc_ValueError, "Unsupported type");
        return NULL;
    }

    nk_kernel_punned_t metric = NULL;
    nk_capability_t capability = nk_cap_serial_k;
    nk_find_kernel_punned(metric_kind, dtype, static_capabilities, nk_cap_any_k, &metric, &capability);
    if (!metric || !capability) {
        PyErr_SetString(PyExc_LookupError, "No such metric");
        return NULL;
    }

    return PyLong_FromUnsignedLongLong((unsigned long long)metric);
}

static char const doc_cdist[] =                                                                                   //
    "Compute pairwise distances between two input sets.\n\n"                                                      //
    "Parameters:\n"                                                                                               //
    "    a (Tensor): First matrix.\n"                                                                             //
    "    b (Tensor): Second matrix.\n"                                                                            //
    "    metric (str, optional): Distance metric to use (e.g., 'sqeuclidean', 'cosine').\n"                       //
    "    out (Tensor, optional): Output matrix to store the result.\n"                                            //
    "    dtype (Union[IntegralType, FloatType, ComplexType], optional): Override the presumed input type name.\n" //
    "    out_dtype (Union[FloatType, ComplexType], optional): Result type, default is 'float64'.\n"               //
    "    threads (int, optional): Number of threads to use (default is 1).\n\n"                                   //
    "Returns:\n"                                                                                                  //
    "    Tensor: Pairwise distances between all inputs.\n\n"                                                      //
    "Equivalent to: `scipy.spatial.distance.cdist`.\n"                                                            //
    "Signature:\n"                                                                                                //
    "    >>> def cdist(a, b, /, metric, *, dtype, out, out_dtype, threads) -> Optional[Tensor]: ...";

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
    nk_dtype_t dtype = nk_dtype_unknown_k, out_dtype = nk_dtype_unknown_k;

    /// Same default as in SciPy:
    /// https://docs.scipy.org/doc/scipy-1.11.4/reference/generated/scipy.spatial.distance.cdist.html
    nk_kernel_kind_t metric_kind = nk_kernel_euclidean_k;
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
        dtype = python_string_to_dtype(dtype_str);
        if (dtype == nk_dtype_unknown_k) {
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
        out_dtype = python_string_to_dtype(out_dtype_str);
        if (out_dtype == nk_dtype_unknown_k) {
            PyErr_SetString(PyExc_ValueError, "Unsupported 'out_dtype'");
            return NULL;
        }
    }

    return implement_cdist(a_obj, b_obj, out_obj, metric_kind, threads, dtype, out_dtype);
}

static char const doc_euclidean_pointer[] = "Return an integer pointer to the `numkong.euclidean` kernel.";
static PyObject *api_euclidean_pointer(PyObject *self, PyObject *dtype_obj) {
    return implement_pointer_access(nk_kernel_euclidean_k, dtype_obj);
}
static char const doc_sqeuclidean_pointer[] = "Return an integer pointer to the `numkong.sqeuclidean` kernel.";
static PyObject *api_sqeuclidean_pointer(PyObject *self, PyObject *dtype_obj) {
    return implement_pointer_access(nk_kernel_sqeuclidean_k, dtype_obj);
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

static char const doc_euclidean[] =                                                                    //
    "Compute Euclidean distances between two matrices.\n\n"                                            //
    "Parameters:\n"                                                                                    //
    "    a (Tensor): First matrix or vector.\n"                                                        //
    "    b (Tensor): Second matrix or vector.\n"                                                       //
    "    dtype (Union[IntegralType, FloatType], optional): Override the presumed input type name.\n"   //
    "    out (Tensor, optional): Vector for resulting distances. Allocates a new tensor by default.\n" //
    "    out_dtype (FloatType, optional): Result type, default is 'float64'.\n\n"                      //
    "Returns:\n"                                                                                       //
    "    Tensor: The distances if `out` is not provided.\n"                                            //
    "    None: If `out` is provided. Operation will be performed in-place.\n\n"                        //
    "Equivalent to: `scipy.spatial.distance.euclidean`.\n"                                             //
    "Signature:\n"                                                                                     //
    "    >>> def euclidean(a, b, /, dtype, *, out, out_dtype) -> Optional[Tensor]: ...";

static PyObject *api_euclidean(PyObject *self, PyObject *const *args, Py_ssize_t const positional_args_count,
                               PyObject *args_names_tuple) {
    return implement_dense_metric(nk_kernel_euclidean_k, args, positional_args_count, args_names_tuple);
}

static char const doc_sqeuclidean[] =                                                                  //
    "Compute squared Euclidean distances between two matrices.\n\n"                                    //
    "Parameters:\n"                                                                                    //
    "    a (Tensor): First matrix or vector.\n"                                                        //
    "    b (Tensor): Second matrix or vector.\n"                                                       //
    "    dtype (Union[IntegralType, FloatType], optional): Override the presumed input type name.\n"   //
    "    out (Tensor, optional): Vector for resulting distances. Allocates a new tensor by default.\n" //
    "    out_dtype (FloatType, optional): Result type, default is 'float64'.\n\n"                      //
    "Returns:\n"                                                                                       //
    "    Tensor: The distances if `out` is not provided.\n"                                            //
    "    None: If `out` is provided. Operation will be performed in-place.\n\n"                        //
    "Equivalent to: `scipy.spatial.distance.sqeuclidean`.\n"                                           //
    "Signature:\n"                                                                                     //
    "    >>> def sqeuclidean(a, b, /, dtype, *, out, out_dtype) -> Optional[Tensor]: ...";

static PyObject *api_sqeuclidean(PyObject *self, PyObject *const *args, Py_ssize_t const positional_args_count,
                                 PyObject *args_names_tuple) {
    return implement_dense_metric(nk_kernel_sqeuclidean_k, args, positional_args_count, args_names_tuple);
}

static char const doc_angular[] =                                                                      //
    "Compute angular distances between two matrices.\n\n"                                              //
    "Parameters:\n"                                                                                    //
    "    a (Tensor): First matrix or vector.\n"                                                        //
    "    b (Tensor): Second matrix or vector.\n"                                                       //
    "    dtype (Union[IntegralType, FloatType], optional): Override the presumed input type name.\n"   //
    "    out (Tensor, optional): Vector for resulting distances. Allocates a new tensor by default.\n" //
    "    out_dtype (FloatType, optional): Result type, default is 'float64'.\n\n"                      //
    "Returns:\n"                                                                                       //
    "    Tensor: The distances if `out` is not provided.\n"                                            //
    "    None: If `out` is provided. Operation will be performed in-place.\n\n"                        //
    "Equivalent to: `scipy.spatial.distance.cosine`.\n"                                                //
    "Signature:\n"                                                                                     //
    "    >>> def angular(a, b, /, dtype, *, out, out_dtype) -> Optional[Tensor]: ...";

static PyObject *api_angular(PyObject *self, PyObject *const *args, Py_ssize_t const positional_args_count,
                             PyObject *args_names_tuple) {
    return implement_dense_metric(nk_kernel_angular_k, args, positional_args_count, args_names_tuple);
}

static char const doc_dot[] =                                                                                     //
    "Compute the inner (dot) product between two matrices (real or complex).\n\n"                                 //
    "Parameters:\n"                                                                                               //
    "    a (Tensor): First matrix or vector.\n"                                                                   //
    "    b (Tensor): Second matrix or vector.\n"                                                                  //
    "    dtype (Union[IntegralType, FloatType, ComplexType], optional): Override the presumed input type name.\n" //
    "    out (Tensor, optional): Vector for resulting distances. Allocates a new tensor by default.\n"            //
    "    out_dtype (Union[FloatType, ComplexType], optional): Result type, default is 'float64'.\n\n"             //
    "Returns:\n"                                                                                                  //
    "    Tensor: The distances if `out` is not provided.\n"                                                       //
    "    None: If `out` is provided. Operation will be performed in-place.\n\n"                                   //
    "Equivalent to: `numpy.inner`.\n"                                                                             //
    "Signature:\n"                                                                                                //
    "    >>> def dot(a, b, /, dtype, *, out, out_dtype) -> Optional[Tensor]: ...";

static PyObject *api_dot(PyObject *self, PyObject *const *args, Py_ssize_t const positional_args_count,
                         PyObject *args_names_tuple) {
    return implement_dense_metric(nk_kernel_dot_k, args, positional_args_count, args_names_tuple);
}

static char const doc_vdot[] =                                                                         //
    "Compute the conjugate dot product between two complex matrices.\n\n"                              //
    "Parameters:\n"                                                                                    //
    "    a (Tensor): First complex matrix or vector.\n"                                                //
    "    b (Tensor): Second complex matrix or vector.\n"                                               //
    "    dtype (ComplexType, optional): Override the presumed input type name.\n"                      //
    "    out (Tensor, optional): Vector for resulting distances. Allocates a new tensor by default.\n" //
    "    out_dtype (Union[ComplexType], optional): Result type, default is 'float64'.\n\n"             //
    "Returns:\n"                                                                                       //
    "    Tensor: The distances if `out` is not provided.\n"                                            //
    "    None: If `out` is provided. Operation will be performed in-place.\n\n"                        //
    "Equivalent to: `numpy.vdot`.\n"                                                                   //
    "Signature:\n"                                                                                     //
    "    >>> def vdot(a, b, /, dtype, *, out, out_dtype) -> Optional[Tensor]: ...";

static PyObject *api_vdot(PyObject *self, PyObject *const *args, Py_ssize_t const positional_args_count,
                          PyObject *args_names_tuple) {
    return implement_dense_metric(nk_kernel_vdot_k, args, positional_args_count, args_names_tuple);
}

static char const doc_kld[] =                                                                          //
    "Compute Kullback-Leibler divergences between two matrices.\n\n"                                   //
    "Parameters:\n"                                                                                    //
    "    a (Tensor): First floating-point matrix or vector.\n"                                         //
    "    b (Tensor): Second floating-point matrix or vector.\n"                                        //
    "    dtype (FloatType, optional): Override the presumed input type name.\n"                        //
    "    out (Tensor, optional): Vector for resulting distances. Allocates a new tensor by default.\n" //
    "    out_dtype (FloatType, optional): Result type, default is 'float64'.\n\n"                      //
    "Returns:\n"                                                                                       //
    "    Tensor: The distances if `out` is not provided.\n"                                            //
    "    None: If `out` is provided. Operation will be performed in-place.\n\n"                        //
    "Equivalent to: `scipy.special.kl_div`.\n"                                                         //
    "Signature:\n"                                                                                     //
    "    >>> def kld(a, b, /, dtype, *, out, out_dtype) -> Optional[Tensor]: ...";

static PyObject *api_kld(PyObject *self, PyObject *const *args, Py_ssize_t const positional_args_count,
                         PyObject *args_names_tuple) {
    return implement_dense_metric(nk_kernel_kld_k, args, positional_args_count, args_names_tuple);
}

static char const doc_jsd[] =                                                                          //
    "Compute Jensen-Shannon divergences between two matrices.\n\n"                                     //
    "Parameters:\n"                                                                                    //
    "    a (Tensor): First floating-point matrix or vector.\n"                                         //
    "    b (Tensor): Second floating-point matrix or vector.\n"                                        //
    "    dtype (Union[IntegralType, FloatType], optional): Override the presumed input type name.\n"   //
    "    out (Tensor, optional): Vector for resulting distances. Allocates a new tensor by default.\n" //
    "    out_dtype (FloatType, optional): Result type, default is 'float64'.\n\n"                      //
    "Returns:\n"                                                                                       //
    "    Tensor: The distances if `out` is not provided.\n"                                            //
    "    None: If `out` is provided. Operation will be performed in-place.\n\n"                        //
    "Equivalent to: `scipy.spatial.distance.jensenshannon`.\n"                                         //
    "Signature:\n"                                                                                     //
    "    >>> def jsd(a, b, /, dtype, *, out, out_dtype) -> Optional[Tensor]: ...";

static PyObject *api_jsd(PyObject *self, PyObject *const *args, Py_ssize_t const positional_args_count,
                         PyObject *args_names_tuple) {
    return implement_dense_metric(nk_kernel_jsd_k, args, positional_args_count, args_names_tuple);
}

static char const doc_hamming[] =                                                                      //
    "Compute Hamming distances between two matrices.\n\n"                                              //
    "Parameters:\n"                                                                                    //
    "    a (Tensor): First binary matrix or vector.\n"                                                 //
    "    b (Tensor): Second binary matrix or vector.\n"                                                //
    "    dtype (IntegralType, optional): Override the presumed input type name.\n"                     //
    "    out (Tensor, optional): Vector for resulting distances. Allocates a new tensor by default.\n" //
    "    out_dtype (FloatType, optional): Result type, default is 'float64'.\n\n"                      //
    "Returns:\n"                                                                                       //
    "    Tensor: The distances if `out` is not provided.\n"                                            //
    "    None: If `out` is provided. Operation will be performed in-place.\n\n"                        //
    "Similar to: `scipy.spatial.distance.hamming`.\n"                                                  //
    "Signature:\n"                                                                                     //
    "    >>> def hamming(a, b, /, dtype, *, out, out_dtype) -> Optional[Tensor]: ...";

static PyObject *api_hamming(PyObject *self, PyObject *const *args, Py_ssize_t const positional_args_count,
                             PyObject *args_names_tuple) {
    return implement_dense_metric(nk_kernel_hamming_k, args, positional_args_count, args_names_tuple);
}

static char const doc_jaccard[] =                                                                      //
    "Compute Jaccard distances (bitwise Tanimoto) between two matrices.\n\n"                           //
    "Parameters:\n"                                                                                    //
    "    a (Tensor): First binary matrix or vector.\n"                                                 //
    "    b (Tensor): Second binary matrix or vector.\n"                                                //
    "    dtype (IntegralType, optional): Override the presumed input type name.\n"                     //
    "    out (Tensor, optional): Vector for resulting distances. Allocates a new tensor by default.\n" //
    "    out_dtype (FloatType, optional): Result type, default is 'float64'.\n\n"                      //
    "Returns:\n"                                                                                       //
    "    Tensor: The distances if `out` is not provided.\n"                                            //
    "    None: If `out` is provided. Operation will be performed in-place.\n\n"                        //
    "Similar to: `scipy.spatial.distance.jaccard`.\n"                                                  //
    "Signature:\n"                                                                                     //
    "    >>> def jaccard(a, b, /, dtype, *, out, out_dtype) -> Optional[Tensor]: ...";

static PyObject *api_jaccard(PyObject *self, PyObject *const *args, Py_ssize_t const positional_args_count,
                             PyObject *args_names_tuple) {
    return implement_dense_metric(nk_kernel_jaccard_k, args, positional_args_count, args_names_tuple);
}

static char const doc_bilinear[] =                                                //
    "Compute the bilinear form between two vectors given a metric tensor.\n\n"    //
    "Parameters:\n"                                                               //
    "    a (Tensor): First vector.\n"                                             //
    "    b (Tensor): Second vector.\n"                                            //
    "    metric_tensor (Tensor): The metric tensor defining the bilinear form.\n" //
    "    dtype (FloatType, optional): Override the presumed input type name.\n\n" //
    "Returns:\n"                                                                  //
    "    float: The bilinear form.\n\n"                                           //
    "Equivalent to: `numpy.dot` with a metric tensor.\n"                          //
    "Signature:\n"                                                                //
    "    >>> def bilinear(a, b, metric_tensor, /, dtype) -> float: ...";

static PyObject *api_bilinear(PyObject *self, PyObject *const *args, Py_ssize_t const positional_args_count,
                              PyObject *args_names_tuple) {
    return implement_curved_metric(nk_kernel_bilinear_k, args, positional_args_count, args_names_tuple);
}

static char const doc_mahalanobis[] =                                                              //
    "Compute the Mahalanobis distance between two vectors given an inverse covariance matrix.\n\n" //
    "Parameters:\n"                                                                                //
    "    a (Tensor): First vector.\n"                                                              //
    "    b (Tensor): Second vector.\n"                                                             //
    "    inverse_covariance (Tensor): The inverse of the covariance matrix.\n"                     //
    "    dtype (FloatType, optional): Override the presumed input type name.\n\n"                  //
    "Returns:\n"                                                                                   //
    "    float: The Mahalanobis distance.\n\n"                                                     //
    "Equivalent to: `scipy.spatial.distance.mahalanobis`.\n"                                       //
    "Signature:\n"                                                                                 //
    "    >>> def mahalanobis(a, b, inverse_covariance, /, dtype) -> float: ...";

static PyObject *api_mahalanobis(PyObject *self, PyObject *const *args, Py_ssize_t const positional_args_count,
                                 PyObject *args_names_tuple) {
    return implement_curved_metric(nk_kernel_mahalanobis_k, args, positional_args_count, args_names_tuple);
}

static char const doc_haversine[] =                                               //
    "Compute the Haversine (great-circle) distance between coordinate pairs.\n\n" //
    "Parameters:\n"                                                               //
    "    a_lats (Tensor): Latitudes of first points in radians.\n"                //
    "    a_lons (Tensor): Longitudes of first points in radians.\n"               //
    "    b_lats (Tensor): Latitudes of second points in radians.\n"               //
    "    b_lons (Tensor): Longitudes of second points in radians.\n"              //
    "    dtype (FloatType, optional): Override the presumed input type name.\n"   //
    "    out (Tensor, optional): Pre-allocated output array for distances.\n\n"   //
    "Returns:\n"                                                                  //
    "    Tensor: Distances in meters (using mean Earth radius).\n"                //
    "    None: If `out` is provided.\n\n"                                         //
    "Note: Input coordinates must be in radians. Uses spherical Earth model.\n"   //
    "Signature:\n"                                                                //
    "    >>> def haversine(a_lats, a_lons, b_lats, b_lons, /, dtype, *, out) -> Optional[Tensor]: ...";

static PyObject *api_haversine(PyObject *self, PyObject *const *args, Py_ssize_t const positional_args_count,
                               PyObject *args_names_tuple) {
    return implement_geospatial_metric(nk_kernel_haversine_k, args, positional_args_count, args_names_tuple);
}

static char const doc_vincenty[] =                                                         //
    "Compute the Vincenty (ellipsoidal geodesic) distance between coordinate pairs.\n\n"   //
    "Parameters:\n"                                                                        //
    "    a_lats (Tensor): Latitudes of first points in radians.\n"                         //
    "    a_lons (Tensor): Longitudes of first points in radians.\n"                        //
    "    b_lats (Tensor): Latitudes of second points in radians.\n"                        //
    "    b_lons (Tensor): Longitudes of second points in radians.\n"                       //
    "    dtype (FloatType, optional): Override the presumed input type name.\n"            //
    "    out (Tensor, optional): Pre-allocated output array for distances.\n\n"            //
    "Returns:\n"                                                                           //
    "    Tensor: Distances in meters (using WGS84 ellipsoid).\n"                           //
    "    None: If `out` is provided.\n\n"                                                  //
    "Note: Input coordinates must be in radians. Uses iterative algorithm for accuracy.\n" //
    "Signature:\n"                                                                         //
    "    >>> def vincenty(a_lats, a_lons, b_lats, b_lons, /, dtype, *, out) -> Optional[Tensor]: ...";

static PyObject *api_vincenty(PyObject *self, PyObject *const *args, Py_ssize_t const positional_args_count,
                              PyObject *args_names_tuple) {
    return implement_geospatial_metric(nk_kernel_vincenty_k, args, positional_args_count, args_names_tuple);
}

static char const doc_intersect[] =                              //
    "Compute the intersection of two sorted integer arrays.\n\n" //
    "Parameters:\n"                                              //
    "    a (Tensor): First sorted integer array.\n"              //
    "    b (Tensor): Second sorted integer array.\n\n"           //
    "Returns:\n"                                                 //
    "    float: The number of intersecting elements.\n\n"        //
    "Similar to: `numpy.intersect1d`.\n"                         //
    "Signature:\n"                                               //
    "    >>> def intersect(a, b, /) -> float: ...";

static PyObject *api_intersect(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    return implement_sparse_metric(nk_kernel_sparse_intersect_k, args, nargs);
}

static char const doc_fma[] =                                                                          //
    "Fused-Multiply-Add between 3 input vectors.\n\n"                                                  //
    "Parameters:\n"                                                                                    //
    "    a (Tensor): First vector.\n"                                                                  //
    "    b (Tensor): Second vector.\n"                                                                 //
    "    c (Tensor): Third vector.\n"                                                                  //
    "    dtype (Union[IntegralType, FloatType], optional): Override the presumed numeric type name.\n" //
    "    alpha (float, optional): First scale, 1.0 by default.\n"                                      //
    "    beta (float, optional): Second scale, 1.0 by default.\n"                                      //
    "    out (Tensor, optional): Vector for resulting distances.\n\n"                                  //
    "Returns:\n"                                                                                       //
    "    Tensor: The distances if `out` is not provided.\n"                                            //
    "    None: If `out` is provided. Operation will be performed in-place.\n\n"                        //
    "Equivalent to: `alpha * a * b + beta * c`.\n"                                                     //
    "Signature:\n"                                                                                     //
    "    >>> def fma(a, b, c, /, dtype, *, alpha, beta, out) -> Optional[Tensor]: ...";

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
    nk_dtype_t dtype = nk_dtype_unknown_k;

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
        dtype = python_string_to_dtype(dtype_str);
        if (dtype == nk_dtype_unknown_k) {
            PyErr_SetString(PyExc_ValueError, "Unsupported 'dtype'");
            return NULL;
        }
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
    if (a_parsed.dtype != b_parsed.dtype || a_parsed.dtype == nk_dtype_unknown_k ||
        b_parsed.dtype == nk_dtype_unknown_k || c_parsed.dtype == nk_dtype_unknown_k ||
        (out_obj && out_parsed.dtype == nk_dtype_unknown_k)) {
        PyErr_SetString(PyExc_TypeError, "Input tensors must have matching dtypes, check with `X.__array_interface__`");
        goto cleanup;
    }
    if (dtype == nk_dtype_unknown_k) dtype = a_parsed.dtype;

    // Convert `alpha_obj` to `alpha_buf` and `beta_obj` to `beta_buf`
    nk_scalar_buffer_t alpha_buf, beta_buf;
    {
        static PyObject *py_one = NULL;
        if (!py_one) py_one = PyFloat_FromDouble(1.0);
        if (!py_number_to_scalar_buffer(alpha_obj ? alpha_obj : py_one, &alpha_buf, dtype)) goto cleanup;
        if (!py_number_to_scalar_buffer(beta_obj ? beta_obj : py_one, &beta_buf, dtype)) goto cleanup;
    }

    // Look up the metric and the capability
    nk_each_fma_punned_t metric = NULL;
    nk_capability_t capability = nk_cap_serial_k;
    nk_kernel_kind_t const metric_kind = nk_kernel_each_fma_k;
    nk_find_kernel_punned(metric_kind, dtype, static_capabilities, nk_cap_any_k, (nk_kernel_punned_t *)&metric,
                          &capability);
    if (!metric || !capability) {
        PyErr_Format( //
            PyExc_LookupError,
            "Unsupported metric '%c' and dtype combination across vectors ('%s'/'%s') and " "`dtype` override " "('%s'/" "'%s')",
            metric_kind,                                                                       //
            a_buffer.format ? a_buffer.format : "nil", dtype_to_python_string(a_parsed.dtype), //
            dtype_str ? dtype_str : "nil", dtype_to_python_string(dtype));
        goto cleanup;
    }

    char *distances_start = NULL;
    size_t distances_stride_bytes = 0;

    // Allocate the output matrix if it wasn't provided
    if (!out_obj) {
        Tensor *distances_obj = PyObject_NewVar(Tensor, &TensorType, a_parsed.dimensions * bytes_per_dtype(dtype));
        if (!distances_obj) {
            PyErr_NoMemory();
            goto cleanup;
        }

        // Initialize the object
        distances_obj->dtype = dtype;
        distances_obj->rank = 1;
        distances_obj->shape[0] = a_parsed.dimensions;
        distances_obj->shape[1] = 0;
        distances_obj->strides[0] = bytes_per_dtype(dtype);
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

    metric(a_parsed.start, b_parsed.start, c_parsed.start, a_parsed.dimensions, &alpha_buf, &beta_buf, distances_start);
cleanup:
    PyBuffer_Release(&a_buffer);
    PyBuffer_Release(&b_buffer);
    PyBuffer_Release(&c_buffer);
    PyBuffer_Release(&out_buffer);
    return return_obj;
}

static char const doc_wsum[] =                                                                         //
    "Weighted Sum of 2 input vectors.\n\n"                                                             //
    "Parameters:\n"                                                                                    //
    "    a (Tensor): First vector.\n"                                                                  //
    "    b (Tensor): Second vector.\n"                                                                 //
    "    dtype (Union[IntegralType, FloatType], optional): Override the presumed numeric type name.\n" //
    "    alpha (float, optional): First scale, 1.0 by default.\n"                                      //
    "    beta (float, optional): Second scale, 1.0 by default.\n"                                      //
    "    out (Tensor, optional): Vector for resulting distances.\n\n"                                  //
    "Returns:\n"                                                                                       //
    "    Tensor: The distances if `out` is not provided.\n"                                            //
    "    None: If `out` is provided. Operation will be performed in-place.\n\n"                        //
    "Equivalent to: `alpha * a + beta * b`.\n"                                                         //
    "Signature:\n"                                                                                     //
    "    >>> def wsum(a, b, /, dtype, *, alpha, beta, out) -> Optional[Tensor]: ...";

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
    nk_dtype_t dtype = nk_dtype_unknown_k;

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
        dtype = python_string_to_dtype(dtype_str);
        if (dtype == nk_dtype_unknown_k) {
            PyErr_SetString(PyExc_ValueError, "Unsupported 'dtype'");
            return NULL;
        }
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
    if (a_parsed.dtype != b_parsed.dtype || a_parsed.dtype == nk_dtype_unknown_k ||
        b_parsed.dtype == nk_dtype_unknown_k || (out_obj && out_parsed.dtype == nk_dtype_unknown_k)) {
        PyErr_SetString(PyExc_TypeError, "Input tensors must have matching dtypes, check with `X.__array_interface__`");
        goto cleanup;
    }
    if (dtype == nk_dtype_unknown_k) dtype = a_parsed.dtype;

    // Convert `alpha_obj` to `alpha_buf` and `beta_obj` to `beta_buf`
    nk_scalar_buffer_t alpha_buf, beta_buf;
    {
        static PyObject *py_one = NULL;
        if (!py_one) py_one = PyFloat_FromDouble(1.0);
        if (!py_number_to_scalar_buffer(alpha_obj ? alpha_obj : py_one, &alpha_buf, dtype)) goto cleanup;
        if (!py_number_to_scalar_buffer(beta_obj ? beta_obj : py_one, &beta_buf, dtype)) goto cleanup;
    }

    // Look up the metric and the capability
    nk_each_blend_punned_t metric = NULL;
    nk_capability_t capability = nk_cap_serial_k;
    nk_kernel_kind_t const metric_kind = nk_kernel_each_blend_k;
    nk_find_kernel_punned(metric_kind, dtype, static_capabilities, nk_cap_any_k, (nk_kernel_punned_t *)&metric,
                          &capability);
    if (!metric || !capability) {
        PyErr_Format( //
            PyExc_LookupError,
            "Unsupported metric '%c' and dtype combination across vectors ('%s'/'%s') and " "`dtype` override " "('%s'/" "'%s')",
            metric_kind,                                                                       //
            a_buffer.format ? a_buffer.format : "nil", dtype_to_python_string(a_parsed.dtype), //
            dtype_str ? dtype_str : "nil", dtype_to_python_string(dtype));
        goto cleanup;
    }

    char *distances_start = NULL;
    size_t distances_stride_bytes = 0;

    // Allocate the output matrix if it wasn't provided
    if (!out_obj) {
        Tensor *distances_obj = PyObject_NewVar(Tensor, &TensorType, a_parsed.dimensions * bytes_per_dtype(dtype));
        if (!distances_obj) {
            PyErr_NoMemory();
            goto cleanup;
        }

        // Initialize the object
        distances_obj->dtype = dtype;
        distances_obj->rank = 1;
        distances_obj->shape[0] = a_parsed.dimensions;
        distances_obj->shape[1] = 0;
        distances_obj->strides[0] = bytes_per_dtype(dtype);
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

    metric(a_parsed.start, b_parsed.start, a_parsed.dimensions, &alpha_buf, &beta_buf, distances_start);
cleanup:
    PyBuffer_Release(&a_buffer);
    PyBuffer_Release(&b_buffer);
    PyBuffer_Release(&out_buffer);
    return return_obj;
}
// endregion

// endregion

static char const doc_scale[] =                                                                        //
    "Element-wise affine transformation of a single vector.\n\n"                                       //
    "Parameters:\n"                                                                                    //
    "    a (Tensor): Input vector.\n"                                                                  //
    "    dtype (Union[IntegralType, FloatType], optional): Override the presumed numeric type name.\n" //
    "    alpha (float, optional): Multiplicative scale, 1.0 by default.\n"                             //
    "    beta (float, optional): Additive offset, 0.0 by default.\n"                                   //
    "    out (Tensor, optional): Vector for resulting output.\n\n"                                     //
    "Returns:\n"                                                                                       //
    "    Tensor: The result if `out` is not provided.\n"                                               //
    "    None: If `out` is provided. Operation will be performed in-place.\n\n"                        //
    "Equivalent to: `alpha * a + beta`.\n"                                                             //
    "Signature:\n"                                                                                     //
    "    >>> def scale(a, /, dtype, *, alpha, beta, out) -> Optional[Tensor]: ...";

static PyObject *api_scale(PyObject *self, PyObject *const *args, Py_ssize_t const positional_args_count,
                           PyObject *args_names_tuple) {
    (void)self;
    PyObject *return_obj = NULL;

    // This function accepts up to 5 arguments:
    PyObject *a_obj = NULL;     // Required object, positional-only
    PyObject *dtype_obj = NULL; // Optional object, "dtype" keyword or positional
    PyObject *out_obj = NULL;   // Optional object, "out" keyword-only
    PyObject *alpha_obj = NULL; // Optional object, "alpha" keyword-only
    PyObject *beta_obj = NULL;  // Optional object, "beta" keyword-only

    // Once parsed, the arguments will be stored in these variables:
    char const *dtype_str = NULL;
    nk_dtype_t dtype = nk_dtype_unknown_k;

    Py_buffer a_buffer, out_buffer;
    TensorArgument a_parsed, out_parsed;
    memset(&a_buffer, 0, sizeof(Py_buffer));
    memset(&out_buffer, 0, sizeof(Py_buffer));

    Py_ssize_t const args_names_count = args_names_tuple ? PyTuple_Size(args_names_tuple) : 0;
    Py_ssize_t const args_count = positional_args_count + args_names_count;
    if (args_count < 1 || args_count > 5) {
        PyErr_Format(PyExc_TypeError, "Function expects 1-5 arguments, got %zd", args_count);
        return NULL;
    }
    if (positional_args_count > 2) {
        PyErr_Format(PyExc_TypeError, "Only first 2 arguments can be positional, received %zd", positional_args_count);
        return NULL;
    }

    // Positional-only arguments (input vector)
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
        dtype = python_string_to_dtype(dtype_str);
        if (dtype == nk_dtype_unknown_k) {
            PyErr_SetString(PyExc_ValueError, "Unsupported 'dtype'");
            return NULL;
        }
    }

    // Convert `a_obj` to `a_buffer` and to `a_parsed`.
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
    if (a_parsed.dtype == nk_dtype_unknown_k || (out_obj && out_parsed.dtype == nk_dtype_unknown_k)) {
        PyErr_SetString(PyExc_TypeError, "Input tensors must have known dtypes, check with `X.__array_interface__`");
        goto cleanup;
    }
    if (dtype == nk_dtype_unknown_k) dtype = a_parsed.dtype;

    // Convert `alpha_obj` to `alpha_buf` and `beta_obj` to `beta_buf`
    nk_scalar_buffer_t alpha_buf, beta_buf;
    {
        static PyObject *py_one = NULL, *py_zero = NULL;
        if (!py_one) py_one = PyFloat_FromDouble(1.0);
        if (!py_zero) py_zero = PyFloat_FromDouble(0.0);
        if (!py_number_to_scalar_buffer(alpha_obj ? alpha_obj : py_one, &alpha_buf, dtype)) goto cleanup;
        if (!py_number_to_scalar_buffer(beta_obj ? beta_obj : py_zero, &beta_buf, dtype)) goto cleanup;
    }

    // Look up the metric and the capability
    nk_each_scale_punned_t metric = NULL;
    nk_capability_t capability = nk_cap_serial_k;
    nk_kernel_kind_t const metric_kind = nk_kernel_each_scale_k;
    nk_find_kernel_punned(metric_kind, dtype, static_capabilities, nk_cap_any_k, (nk_kernel_punned_t *)&metric,
                          &capability);
    if (!metric || !capability) {
        PyErr_Format( //
            PyExc_LookupError,
            "Unsupported metric '%c' and dtype combination across vectors ('%s'/'%s') and " "`dtype` override " "('%s'/" "'%s')",
            metric_kind,                                                                       //
            a_buffer.format ? a_buffer.format : "nil", dtype_to_python_string(a_parsed.dtype), //
            dtype_str ? dtype_str : "nil", dtype_to_python_string(dtype));
        goto cleanup;
    }

    char *distances_start = NULL;
    size_t distances_stride_bytes = 0;

    // Allocate the output matrix if it wasn't provided
    if (!out_obj) {
        Tensor *distances_obj = PyObject_NewVar(Tensor, &TensorType, a_parsed.dimensions * bytes_per_dtype(dtype));
        if (!distances_obj) {
            PyErr_NoMemory();
            goto cleanup;
        }

        // Initialize the object
        distances_obj->dtype = dtype;
        distances_obj->rank = 1;
        distances_obj->shape[0] = a_parsed.dimensions;
        distances_obj->shape[1] = 0;
        distances_obj->strides[0] = bytes_per_dtype(dtype);
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

    metric(a_parsed.start, a_parsed.dimensions, &alpha_buf, &beta_buf, distances_start);
cleanup:
    PyBuffer_Release(&a_buffer);
    PyBuffer_Release(&out_buffer);
    return return_obj;
}

static char const doc_add[] =                                                                  //
    "Element-wise addition of two vectors or a vector and a scalar.\n\n"                       //
    "Parameters:\n"                                                                            //
    "    a (Union[Tensor, float, int]): First operand (vector or scalar).\n"                   //
    "    b (Union[Tensor, float, int]): Second operand (vector or scalar).\n"                  //
    "    out (Tensor, optional): Output buffer for the result.\n"                              //
    "    a_dtype (Union[IntegralType, FloatType], optional): Override dtype for `a`.\n"        //
    "    b_dtype (Union[IntegralType, FloatType], optional): Override dtype for `b`.\n"        //
    "    out_dtype (Union[IntegralType, FloatType], optional): Override dtype for output.\n\n" //
    "Returns:\n"                                                                               //
    "    Tensor: The sum if `out` is not provided.\n"                                          //
    "    None: If `out` is provided (in-place operation).\n\n"                                 //
    "Equivalent to: `a + b`.\n"                                                                //
    "Signature:\n"                                                                             //
    "    >>> def add(a, b, /, *, out, a_dtype, b_dtype, out_dtype) -> Optional[Tensor]: ...";

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

    nk_dtype_t dtype = nk_dtype_unknown_k;

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
        PyObject *array_obj = a_is_scalar ? b_obj : a_obj;
        PyObject *scalar_obj = a_is_scalar ? a_obj : b_obj;

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
            dtype = python_string_to_dtype(dtype_str);
        }
        else { dtype = a_parsed.dtype; }

        if (dtype == nk_dtype_unknown_k) {
            PyErr_SetString(PyExc_ValueError, "Unsupported dtype");
            goto cleanup;
        }

        // Find scale kernel
        nk_each_scale_punned_t scale_kernel = NULL;
        nk_capability_t capability = nk_cap_serial_k;
        nk_find_kernel_punned(nk_kernel_each_scale_k, dtype, static_capabilities, nk_cap_any_k,
                              (nk_kernel_punned_t *)&scale_kernel, &capability);
        if (!scale_kernel || !capability) {
            PyErr_Format(PyExc_LookupError, "No scale kernel for dtype '%s'", dtype_to_string(dtype));
            goto cleanup;
        }

        char *result_start = NULL;
        if (!out_obj) {
            Tensor *result_obj = PyObject_NewVar(Tensor, &TensorType, a_parsed.dimensions * bytes_per_dtype(dtype));
            if (!result_obj) {
                PyErr_NoMemory();
                goto cleanup;
            }
            result_obj->dtype = dtype;
            result_obj->rank = 1;
            result_obj->shape[0] = a_parsed.dimensions;
            result_obj->shape[1] = 0;
            result_obj->strides[0] = bytes_per_dtype(dtype);
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
        {
            static PyObject *py_one = NULL;
            if (!py_one) py_one = PyFloat_FromDouble(1.0);
            if (!py_number_to_scalar_buffer(py_one, &alpha_buf, dtype)) goto cleanup;
            if (!py_number_to_scalar_buffer(scalar_obj, &beta_buf, dtype)) goto cleanup;
        }
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

    // Check dtypes match
    if (a_parsed.dtype != b_parsed.dtype) {
        PyErr_SetString(PyExc_TypeError, "Input arrays must have matching dtypes");
        goto cleanup;
    }

    // Determine output dtype
    if (out_dtype_obj) {
        char const *dtype_str = PyUnicode_AsUTF8(out_dtype_obj);
        if (!dtype_str) { goto cleanup; }
        dtype = python_string_to_dtype(dtype_str);
    }
    else { dtype = a_parsed.dtype; }

    if (dtype == nk_dtype_unknown_k) {
        PyErr_SetString(PyExc_ValueError, "Unsupported dtype");
        goto cleanup;
    }

    // Find sum kernel
    nk_each_sum_punned_t sum_kernel = NULL;
    nk_capability_t capability = nk_cap_serial_k;
    nk_find_kernel_punned(nk_kernel_each_sum_k, dtype, static_capabilities, nk_cap_any_k,
                          (nk_kernel_punned_t *)&sum_kernel, &capability);
    if (!sum_kernel || !capability) {
        PyErr_Format(PyExc_LookupError, "No sum kernel for dtype '%s'", dtype_to_string(dtype));
        goto cleanup;
    }

    char *result_start = NULL;
    if (!out_obj) {
        Tensor *result_obj = PyObject_NewVar(Tensor, &TensorType, a_parsed.dimensions * bytes_per_dtype(dtype));
        if (!result_obj) {
            PyErr_NoMemory();
            goto cleanup;
        }
        result_obj->dtype = dtype;
        result_obj->rank = 1;
        result_obj->shape[0] = a_parsed.dimensions;
        result_obj->shape[1] = 0;
        result_obj->strides[0] = bytes_per_dtype(dtype);
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

static char const doc_multiply[] =                                                             //
    "Element-wise multiplication of two vectors or a vector and a scalar.\n\n"                 //
    "Parameters:\n"                                                                            //
    "    a (Union[Tensor, float, int]): First operand (vector or scalar).\n"                   //
    "    b (Union[Tensor, float, int]): Second operand (vector or scalar).\n"                  //
    "    out (Tensor, optional): Output buffer for the result.\n"                              //
    "    a_dtype (Union[IntegralType, FloatType], optional): Override dtype for `a`.\n"        //
    "    b_dtype (Union[IntegralType, FloatType], optional): Override dtype for `b`.\n"        //
    "    out_dtype (Union[IntegralType, FloatType], optional): Override dtype for output.\n\n" //
    "Returns:\n"                                                                               //
    "    Tensor: The product if `out` is not provided.\n"                                      //
    "    None: If `out` is provided (in-place operation).\n\n"                                 //
    "Equivalent to: `a * b`.\n"                                                                //
    "Signature:\n"                                                                             //
    "    >>> def multiply(a, b, /, *, out, a_dtype, b_dtype, out_dtype) -> Optional[Tensor]: ...";

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

    nk_dtype_t dtype = nk_dtype_unknown_k;

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
        PyObject *array_obj = a_is_scalar ? b_obj : a_obj;
        PyObject *scalar_obj = a_is_scalar ? a_obj : b_obj;

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
            dtype = python_string_to_dtype(dtype_str);
        }
        else { dtype = a_parsed.dtype; }

        if (dtype == nk_dtype_unknown_k) {
            PyErr_SetString(PyExc_ValueError, "Unsupported dtype");
            goto cleanup;
        }

        // Find scale kernel
        nk_each_scale_punned_t scale_kernel = NULL;
        nk_capability_t capability = nk_cap_serial_k;
        nk_find_kernel_punned(nk_kernel_each_scale_k, dtype, static_capabilities, nk_cap_any_k,
                              (nk_kernel_punned_t *)&scale_kernel, &capability);
        if (!scale_kernel || !capability) {
            PyErr_Format(PyExc_LookupError, "No scale kernel for dtype '%s'", dtype_to_string(dtype));
            goto cleanup;
        }

        char *result_start = NULL;
        if (!out_obj) {
            Tensor *result_obj = PyObject_NewVar(Tensor, &TensorType, a_parsed.dimensions * bytes_per_dtype(dtype));
            if (!result_obj) {
                PyErr_NoMemory();
                goto cleanup;
            }
            result_obj->dtype = dtype;
            result_obj->rank = 1;
            result_obj->shape[0] = a_parsed.dimensions;
            result_obj->shape[1] = 0;
            result_obj->strides[0] = bytes_per_dtype(dtype);
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
        {
            static PyObject *py_zero = NULL;
            if (!py_zero) py_zero = PyFloat_FromDouble(0.0);
            if (!py_number_to_scalar_buffer(scalar_obj, &alpha_buf, dtype)) goto cleanup;
            if (!py_number_to_scalar_buffer(py_zero, &beta_buf, dtype)) goto cleanup;
        }
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

    // Check dtypes match
    if (a_parsed.dtype != b_parsed.dtype) {
        PyErr_SetString(PyExc_TypeError, "Input arrays must have matching dtypes");
        goto cleanup;
    }

    // Determine output dtype
    if (out_dtype_obj) {
        char const *dtype_str = PyUnicode_AsUTF8(out_dtype_obj);
        if (!dtype_str) { goto cleanup; }
        dtype = python_string_to_dtype(dtype_str);
    }
    else { dtype = a_parsed.dtype; }

    if (dtype == nk_dtype_unknown_k) {
        PyErr_SetString(PyExc_ValueError, "Unsupported dtype");
        goto cleanup;
    }

    // Find fma kernel
    nk_each_fma_punned_t fma_kernel = NULL;
    nk_capability_t capability = nk_cap_serial_k;
    nk_find_kernel_punned(nk_kernel_each_fma_k, dtype, static_capabilities, nk_cap_any_k,
                          (nk_kernel_punned_t *)&fma_kernel, &capability);
    if (!fma_kernel || !capability) {
        PyErr_Format(PyExc_LookupError, "No fma kernel for dtype '%s'", dtype_to_string(dtype));
        goto cleanup;
    }

    char *result_start = NULL;
    if (!out_obj) {
        Tensor *result_obj = PyObject_NewVar(Tensor, &TensorType, a_parsed.dimensions * bytes_per_dtype(dtype));
        if (!result_obj) {
            PyErr_NoMemory();
            goto cleanup;
        }
        result_obj->dtype = dtype;
        result_obj->rank = 1;
        result_obj->shape[0] = a_parsed.dimensions;
        result_obj->shape[1] = 0;
        result_obj->strides[0] = bytes_per_dtype(dtype);
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
    {
        static PyObject *py_one = NULL, *py_zero = NULL;
        if (!py_one) py_one = PyFloat_FromDouble(1.0);
        if (!py_zero) py_zero = PyFloat_FromDouble(0.0);
        if (!py_number_to_scalar_buffer(py_one, &alpha_buf, dtype)) goto cleanup;
        if (!py_number_to_scalar_buffer(py_zero, &beta_buf, dtype)) goto cleanup;
    }
    fma_kernel(a_parsed.start, b_parsed.start, result_start, a_parsed.dimensions, &alpha_buf, &beta_buf, result_start);

cleanup:
    PyBuffer_Release(&a_buffer);
    PyBuffer_Release(&b_buffer);
    PyBuffer_Release(&out_buffer);
    return return_obj;
}

static char const doc_sin[] =                                                                          //
    "Element-wise trigonometric sine.\n\n"                                                             //
    "Parameters:\n"                                                                                    //
    "    a (Tensor): Input vector of angles in radians.\n"                                             //
    "    dtype (Union[IntegralType, FloatType], optional): Override the presumed numeric type name.\n" //
    "    out (Tensor, optional): Vector for resulting values.\n\n"                                     //
    "Returns:\n"                                                                                       //
    "    Tensor: The sine values if `out` is not provided.\n"                                          //
    "    None: If `out` is provided.\n\n"                                                              //
    "Signature:\n"                                                                                     //
    "    >>> def sin(a, /, dtype, *, out) -> Optional[Tensor]: ...";

static char const doc_cos[] =                                                                          //
    "Element-wise trigonometric cosine.\n\n"                                                           //
    "Parameters:\n"                                                                                    //
    "    a (Tensor): Input vector of angles in radians.\n"                                             //
    "    dtype (Union[IntegralType, FloatType], optional): Override the presumed numeric type name.\n" //
    "    out (Tensor, optional): Vector for resulting values.\n\n"                                     //
    "Returns:\n"                                                                                       //
    "    Tensor: The cosine values if `out` is not provided.\n"                                        //
    "    None: If `out` is provided.\n\n"                                                              //
    "Signature:\n"                                                                                     //
    "    >>> def cos(a, /, dtype, *, out) -> Optional[Tensor]: ...";

static char const doc_atan[] =                                                                         //
    "Element-wise trigonometric arctangent.\n\n"                                                       //
    "Parameters:\n"                                                                                    //
    "    a (Tensor): Input vector of values.\n"                                                        //
    "    dtype (Union[IntegralType, FloatType], optional): Override the presumed numeric type name.\n" //
    "    out (Tensor, optional): Vector for resulting angles in radians.\n\n"                          //
    "Returns:\n"                                                                                       //
    "    Tensor: The arctangent values if `out` is not provided.\n"                                    //
    "    None: If `out` is provided.\n\n"                                                              //
    "Signature:\n"                                                                                     //
    "    >>> def atan(a, /, dtype, *, out) -> Optional[Tensor]: ...";

static PyObject *implement_trigonometry(nk_kernel_kind_t metric_kind, PyObject *const *args,
                                        Py_ssize_t const positional_args_count, PyObject *args_names_tuple) {

    PyObject *return_obj = NULL;

    // This function accepts up to 3 arguments:
    PyObject *a_obj = NULL;     // Required object, positional-only
    PyObject *dtype_obj = NULL; // Optional object, "dtype" keyword or positional
    PyObject *out_obj = NULL;   // Optional object, "out" keyword-only

    // Once parsed, the arguments will be stored in these variables:
    char const *dtype_str = NULL;
    nk_dtype_t dtype = nk_dtype_unknown_k;

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
        dtype = python_string_to_dtype(dtype_str);
        if (dtype == nk_dtype_unknown_k) {
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
    if (a_parsed.dtype == nk_dtype_unknown_k || (out_obj && out_parsed.dtype == nk_dtype_unknown_k)) {
        PyErr_SetString(PyExc_TypeError, "Input tensor must have a known dtype, check with `X.__array_interface__`");
        goto cleanup;
    }
    if (dtype == nk_dtype_unknown_k) dtype = a_parsed.dtype;

    // Look up the kernel and the capability
    nk_kernel_trigonometry_punned_t kernel = NULL;
    nk_capability_t capability = nk_cap_serial_k;
    nk_find_kernel_punned(metric_kind, dtype, static_capabilities, nk_cap_any_k, (nk_kernel_punned_t *)&kernel,
                          &capability);
    if (!kernel || !capability) {
        PyErr_Format( //
            PyExc_LookupError,
            "Unsupported metric '%c' and dtype combination ('%s'/'%s') and `dtype` override ('%s'/'%s')",
            metric_kind,                                                                       //
            a_buffer.format ? a_buffer.format : "nil", dtype_to_python_string(a_parsed.dtype), //
            dtype_str ? dtype_str : "nil", dtype_to_python_string(dtype));
        goto cleanup;
    }

    char *output_start = NULL;

    // Allocate the output array if it wasn't provided
    if (!out_obj) {
        Tensor *output_obj = PyObject_NewVar(Tensor, &TensorType, a_parsed.dimensions * bytes_per_dtype(dtype));
        if (!output_obj) {
            PyErr_NoMemory();
            goto cleanup;
        }

        // Initialize the object
        output_obj->dtype = dtype;
        output_obj->rank = 1;
        output_obj->shape[0] = a_parsed.dimensions;
        output_obj->shape[1] = 0;
        output_obj->strides[0] = bytes_per_dtype(dtype);
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
    return implement_trigonometry(nk_kernel_each_sin_k, args, positional_args_count, args_names_tuple);
}

static PyObject *api_cos(PyObject *self, PyObject *const *args, Py_ssize_t const positional_args_count,
                         PyObject *args_names_tuple) {
    return implement_trigonometry(nk_kernel_each_cos_k, args, positional_args_count, args_names_tuple);
}

static PyObject *api_atan(PyObject *self, PyObject *const *args, Py_ssize_t const positional_args_count,
                          PyObject *args_names_tuple) {
    return implement_trigonometry(nk_kernel_each_atan_k, args, positional_args_count, args_names_tuple);
}

/** @brief Mesh alignment result type — structured return for kabsch/umeyama/rmsd. */
#pragma region MeshAlignmentResult

typedef struct {
    PyObject_HEAD PyObject *rotation; // (3,3) Tensor
    PyObject *scale;                  // 0D Tensor
    PyObject *rmsd;                   // 0D Tensor
    PyObject *a_centroid;             // (3,) Tensor
    PyObject *b_centroid;             // (3,) Tensor
} MeshAlignmentResultObject;

static void MeshAlignmentResult_dealloc(MeshAlignmentResultObject *self) {
    Py_XDECREF(self->rotation);
    Py_XDECREF(self->scale);
    Py_XDECREF(self->rmsd);
    Py_XDECREF(self->a_centroid);
    Py_XDECREF(self->b_centroid);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *MeshAlignmentResult_repr(MeshAlignmentResultObject *self) {
    PyObject *scale_repr = PyObject_Repr(self->scale);
    PyObject *rmsd_repr = PyObject_Repr(self->rmsd);
    if (!scale_repr || !rmsd_repr) {
        Py_XDECREF(scale_repr);
        Py_XDECREF(rmsd_repr);
        return NULL;
    }
    PyObject *result = PyUnicode_FromFormat("MeshAlignmentResult(scale=%U, rmsd=%U)", scale_repr, rmsd_repr);
    Py_DECREF(scale_repr);
    Py_DECREF(rmsd_repr);
    return result;
}

static Py_ssize_t MeshAlignmentResult_length(MeshAlignmentResultObject *self) {
    (void)self;
    return 5;
}

static PyObject *MeshAlignmentResult_item(MeshAlignmentResultObject *self, Py_ssize_t index) {
    PyObject *items[5] = {self->rotation, self->scale, self->rmsd, self->a_centroid, self->b_centroid};
    if (index < 0 || index >= 5) {
        PyErr_SetString(PyExc_IndexError, "MeshAlignmentResult index out of range");
        return NULL;
    }
    Py_INCREF(items[index]);
    return items[index];
}

static PySequenceMethods MeshAlignmentResult_as_sequence = {
    .sq_length = (lenfunc)MeshAlignmentResult_length,
    .sq_item = (ssizeargfunc)MeshAlignmentResult_item,
};

static PyObject *MeshAlignmentResult_transform_point(MeshAlignmentResultObject *self, PyObject *arg) {
    // Extract the point as 3 doubles
    PyObject *point_seq = PySequence_Fast(arg, "transform_point expects a sequence of 3 floats");
    if (!point_seq) return NULL;
    if (PySequence_Fast_GET_SIZE(point_seq) != 3) {
        Py_DECREF(point_seq);
        PyErr_SetString(PyExc_ValueError, "Point must have exactly 3 elements");
        return NULL;
    }
    double point[3];
    for (int idx = 0; idx < 3; idx++) {
        point[idx] = PyFloat_AsDouble(PySequence_Fast_GET_ITEM(point_seq, idx));
        if (PyErr_Occurred()) {
            Py_DECREF(point_seq);
            return NULL;
        }
    }
    Py_DECREF(point_seq);

    // Extract scale factor from stored Tensor
    Tensor *scale_tensor = (Tensor *)self->scale;
    double scale_value = nk_scalar_buffer_get_f64((nk_scalar_buffer_t const *)scale_tensor->data, scale_tensor->dtype);

    // Extract centroids from stored Tensors
    Tensor *a_centroid_tensor = (Tensor *)self->a_centroid;
    Tensor *b_centroid_tensor = (Tensor *)self->b_centroid;
    nk_dtype_t cent_dtype = a_centroid_tensor->dtype;
    size_t cent_bytes = bytes_per_dtype(cent_dtype);
    double a_centroid[3], b_centroid[3];
    for (int idx = 0; idx < 3; idx++) {
        a_centroid[idx] = nk_scalar_buffer_get_f64(
            (nk_scalar_buffer_t const *)(a_centroid_tensor->data + idx * cent_bytes), cent_dtype);
        b_centroid[idx] = nk_scalar_buffer_get_f64(
            (nk_scalar_buffer_t const *)(b_centroid_tensor->data + idx * cent_bytes), cent_dtype);
    }

    // Extract rotation matrix (3x3, row-major) from stored Tensor
    Tensor *rotation_tensor = (Tensor *)self->rotation;
    nk_dtype_t rot_dtype = rotation_tensor->dtype;
    size_t rot_bytes = bytes_per_dtype(rot_dtype);
    double rotation[9];
    for (int idx = 0; idx < 9; idx++)
        rotation[idx] = nk_scalar_buffer_get_f64((nk_scalar_buffer_t const *)(rotation_tensor->data + idx * rot_bytes),
                                                 rot_dtype);

    // Compute: scale * rotation * (point - a_centroid) + b_centroid
    double centered[3] = {point[0] - a_centroid[0], point[1] - a_centroid[1], point[2] - a_centroid[2]};
    double transformed[3];
    for (int row = 0; row < 3; row++) {
        transformed[row] = scale_value * (rotation[row * 3 + 0] * centered[0] + rotation[row * 3 + 1] * centered[1] +
                                          rotation[row * 3 + 2] * centered[2]) +
                           b_centroid[row];
    }

    // Return as a (3,) Tensor matching centroid dtype
    Py_ssize_t output_shape[1] = {3};
    Tensor *output_tensor = Tensor_new(cent_dtype, 1, output_shape);
    if (!output_tensor) return NULL;
    for (int idx = 0; idx < 3; idx++) cast_distance(transformed[idx], cent_dtype, output_tensor->data, idx);
    return (PyObject *)output_tensor;
}

static PyObject *MeshAlignmentResult_transform_points(MeshAlignmentResultObject *self, PyObject *arg) {
    // Parse input as buffer (N,3)
    Py_buffer points_buffer;
    if (PyObject_GetBuffer(arg, &points_buffer, PyBUF_C_CONTIGUOUS | PyBUF_FORMAT) < 0) return NULL;

    if (points_buffer.ndim != 2 || points_buffer.shape[1] != 3) {
        PyBuffer_Release(&points_buffer);
        PyErr_SetString(PyExc_ValueError, "Expected (N, 3) array for transform_points");
        return NULL;
    }

    Py_ssize_t num_points = points_buffer.shape[0];

    // Extract scale factor from stored Tensor
    Tensor *scale_tensor = (Tensor *)self->scale;
    double scale_value = nk_scalar_buffer_get_f64((nk_scalar_buffer_t const *)scale_tensor->data, scale_tensor->dtype);

    // Extract centroids from stored Tensors
    Tensor *a_centroid_tensor = (Tensor *)self->a_centroid;
    Tensor *b_centroid_tensor = (Tensor *)self->b_centroid;
    nk_dtype_t cent_dtype = a_centroid_tensor->dtype;
    size_t cent_bytes = bytes_per_dtype(cent_dtype);
    double a_centroid[3], b_centroid[3];
    for (int idx = 0; idx < 3; idx++) {
        a_centroid[idx] = nk_scalar_buffer_get_f64(
            (nk_scalar_buffer_t const *)(a_centroid_tensor->data + idx * cent_bytes), cent_dtype);
        b_centroid[idx] = nk_scalar_buffer_get_f64(
            (nk_scalar_buffer_t const *)(b_centroid_tensor->data + idx * cent_bytes), cent_dtype);
    }

    // Extract rotation matrix (3x3, row-major) from stored Tensor
    Tensor *rotation_tensor = (Tensor *)self->rotation;
    nk_dtype_t rot_dtype = rotation_tensor->dtype;
    size_t rot_bytes = bytes_per_dtype(rot_dtype);
    double rotation[9];
    for (int idx = 0; idx < 9; idx++)
        rotation[idx] = nk_scalar_buffer_get_f64((nk_scalar_buffer_t const *)(rotation_tensor->data + idx * rot_bytes),
                                                 rot_dtype);

    // Detect input dtype from buffer format
    nk_dtype_t input_dtype = python_string_to_dtype(points_buffer.format);
    if (input_dtype == nk_dtype_unknown_k) input_dtype = nk_f64_k;
    size_t input_bytes = bytes_per_dtype(input_dtype);

    // Allocate output (N,3) Tensor matching input dtype
    Py_ssize_t output_shape[2] = {num_points, 3};
    Tensor *output_tensor = Tensor_new(input_dtype, 2, output_shape);
    if (!output_tensor) {
        PyBuffer_Release(&points_buffer);
        return NULL;
    }

    for (Py_ssize_t point_idx = 0; point_idx < num_points; point_idx++) {
        double point[3];
        for (int idx = 0; idx < 3; idx++)
            point[idx] = nk_scalar_buffer_get_f64(
                (nk_scalar_buffer_t const *)((char *)points_buffer.buf + point_idx * points_buffer.strides[0] +
                                             idx * input_bytes),
                input_dtype);

        double centered[3] = {point[0] - a_centroid[0], point[1] - a_centroid[1], point[2] - a_centroid[2]};
        for (int row = 0; row < 3; row++) {
            double transformed_val = scale_value *
                                         (rotation[row * 3 + 0] * centered[0] + rotation[row * 3 + 1] * centered[1] +
                                          rotation[row * 3 + 2] * centered[2]) +
                                     b_centroid[row];
            cast_distance(transformed_val, input_dtype, output_tensor->data + point_idx * 3 * input_bytes, row);
        }
    }

    PyBuffer_Release(&points_buffer);
    return (PyObject *)output_tensor;
}

static PyMethodDef MeshAlignmentResult_methods[] = {
    {"transform_point", (PyCFunction)MeshAlignmentResult_transform_point, METH_O,
     "Apply the alignment transform to a single (3,) point."},
    {"transform_points", (PyCFunction)MeshAlignmentResult_transform_points, METH_O,
     "Apply the alignment transform to an (N, 3) array of points."},
    {NULL, NULL, 0, NULL},
};

static PyMemberDef MeshAlignmentResult_members[] = {
    {"rotation", T_OBJECT_EX, offsetof(MeshAlignmentResultObject, rotation), READONLY, "(3,3) rotation matrix"},
    {"scale", T_OBJECT_EX, offsetof(MeshAlignmentResultObject, scale), READONLY, "Scale factor"},
    {"rmsd", T_OBJECT_EX, offsetof(MeshAlignmentResultObject, rmsd), READONLY, "Root mean square deviation"},
    {"a_centroid", T_OBJECT_EX, offsetof(MeshAlignmentResultObject, a_centroid), READONLY, "Centroid of first cloud"},
    {"b_centroid", T_OBJECT_EX, offsetof(MeshAlignmentResultObject, b_centroid), READONLY, "Centroid of second cloud"},
    {NULL, 0, 0, 0, NULL},
};

static char const doc_mesh_alignment_result[] =                //
    "Result of mesh alignment (Kabsch, Umeyama, RMSD).\n\n"    //
    "Fields: rotation, scale, rmsd, a_centroid, b_centroid.\n" //
    "Supports iteration and indexing for backward-compatible destructuring.";

static PyTypeObject MeshAlignmentResultType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "numkong.MeshAlignmentResult",
    .tp_basicsize = sizeof(MeshAlignmentResultObject),
    .tp_dealloc = (destructor)MeshAlignmentResult_dealloc,
    .tp_repr = (reprfunc)MeshAlignmentResult_repr,
    .tp_as_sequence = &MeshAlignmentResult_as_sequence,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = doc_mesh_alignment_result,
    .tp_methods = MeshAlignmentResult_methods,
    .tp_members = MeshAlignmentResult_members,
};

#pragma endregion // MeshAlignmentResult

/**
 *  @brief Mesh alignment functions (Kabsch, Umeyama, RMSD).
 *
 *  These compute point cloud alignment and return a MeshAlignmentResult with
 *  rotation, scale, rmsd, a_centroid, and b_centroid fields.
 */

static char const doc_kabsch[] =                                                               //
    "Compute optimal rigid transformation (Kabsch algorithm) between two point clouds.\n\n"    //
    "Finds the optimal rotation matrix that minimizes RMSD between point clouds.\n"            //
    "The transformation aligns point cloud A to point cloud B:\n"                              //
    "    a'_i = scale * R * (a_i - a_centroid) + b_centroid\n\n"                               //
    "Supports both single-pair and batched inputs:\n"                                          //
    "    - Single pair: (N, 3) -> rotation (3,3), scale (), rmsd (), centroids (3,)\n"         //
    "    - Batched: (B, N, 3) -> rotation (B,3,3), scale (B,), rmsd (B,), centroids (B,3)\n\n" //
    "Parameters:\n"                                                                            //
    "    a (Tensor): First point cloud(s), shape (N, 3) or (B, N, 3), float32 or float64.\n"   //
    "    b (Tensor): Second point cloud(s), shape (N, 3) or (B, N, 3), same dtype as a.\n\n"   //
    "Returns:\n"                                                                               //
    "    MeshAlignmentResult: rotation, scale, rmsd, a_centroid, b_centroid fields.\n\n"       //
    "Example:\n"                                                                               //
    "    >>> result = numkong.kabsch(a, b)\n"                                                  //
    "    >>> np.asarray(result.rotation)  # (3, 3) rotation matrix\n"                          //
    "    >>> float(result.scale)          # scale factor (always 1.0 for Kabsch)\n";

static char const doc_umeyama[] =                                                                 //
    "Compute optimal similarity transformation (Umeyama algorithm) between two point clouds.\n\n" //
    "Finds the optimal rotation matrix and uniform scaling factor that minimize RMSD.\n"          //
    "The transformation aligns point cloud A to point cloud B:\n"                                 //
    "    a'_i = scale * R * (a_i - a_centroid) + b_centroid\n\n"                                  //
    "Supports both single-pair and batched inputs:\n"                                             //
    "    - Single pair: (N, 3) -> rotation (3,3), scale (), rmsd (), centroids (3,)\n"            //
    "    - Batched: (B, N, 3) -> rotation (B,3,3), scale (B,), rmsd (B,), centroids (B,3)\n\n"    //
    "Parameters:\n"                                                                               //
    "    a (Tensor): First point cloud(s), shape (N, 3) or (B, N, 3), float32 or float64.\n"      //
    "    b (Tensor): Second point cloud(s), shape (N, 3) or (B, N, 3), same dtype as a.\n\n"      //
    "Returns:\n"                                                                                  //
    "    MeshAlignmentResult: rotation, scale, rmsd, a_centroid, b_centroid fields.\n\n"          //
    "Example:\n"                                                                                  //
    "    >>> result = numkong.umeyama(a, b)\n"                                                    //
    "    >>> float(result.scale)  # Will differ from 1.0 if point clouds have different scales\n";

static char const doc_rmsd[] =                                                                 //
    "Compute RMSD between two point clouds without alignment optimization.\n\n"                //
    "Computes root mean square deviation after centering both clouds.\n"                       //
    "Returns identity rotation and scale=1.0.\n\n"                                             //
    "Supports both single-pair and batched inputs:\n"                                          //
    "    - Single pair: (N, 3) -> rotation (3,3), scale (), rmsd (), centroids (3,)\n"         //
    "    - Batched: (B, N, 3) -> rotation (B,3,3), scale (B,), rmsd (B,), centroids (B,3)\n\n" //
    "Parameters:\n"                                                                            //
    "    a (Tensor): First point cloud(s), shape (N, 3) or (B, N, 3), float32 or float64.\n"   //
    "    b (Tensor): Second point cloud(s), shape (N, 3) or (B, N, 3), same dtype as a.\n\n"   //
    "Returns:\n"                                                                               //
    "    MeshAlignmentResult: rotation, scale, rmsd, a_centroid, b_centroid fields.\n";

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
    Tensor *rot_tensor = NULL;
    Tensor *scale_tensor = NULL;
    Tensor *rmsd_tensor = NULL;
    Tensor *a_cent_tensor = NULL;
    Tensor *b_cent_tensor = NULL;

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
    nk_dtype_t dtype = python_string_to_dtype(a_buffer.format);
    if (dtype != nk_f32_k && dtype != nk_f64_k) {
        PyErr_SetString(PyExc_TypeError, "Point clouds must be float32 or float64");
        goto cleanup;
    }

    // Find the appropriate kernel
    nk_metric_mesh_punned_t kernel = NULL;
    nk_capability_t capability = nk_cap_serial_k;
    nk_find_kernel_punned(metric_kind, dtype, static_capabilities, nk_cap_any_k, (nk_kernel_punned_t *)&kernel,
                          &capability);
    if (!kernel || !capability) {
        PyErr_SetString(PyExc_RuntimeError, "No suitable mesh kernel found for this data type");
        goto cleanup;
    }

    // Check contiguity - we need row-major contiguous data for the innermost 2 dimensions
    Py_ssize_t const elem_size = (Py_ssize_t)bytes_per_dtype(dtype);
    Py_ssize_t const inner_stride_a = is_batched ? a_buffer.strides[2] : a_buffer.strides[1];
    Py_ssize_t const inner_stride_b = is_batched ? b_buffer.strides[2] : b_buffer.strides[1];
    if (inner_stride_a != elem_size || inner_stride_b != elem_size) {
        PyErr_SetString(PyExc_ValueError, "Point clouds must be C-contiguous (row-major)");
        goto cleanup;
    }

    // Calculate strides between batches
    Py_ssize_t const batch_stride_a = is_batched ? a_buffer.strides[0] : 0;
    Py_ssize_t const batch_stride_b = is_batched ? b_buffer.strides[0] : 0;
    nk_size_t num_points = (nk_size_t)n_points;

    nk_dtype_t mesh_out_dtype = nk_kernel_output_dtype(metric_kind, dtype);

    if (!is_batched) {
        // Single pair case - return 0D scalars for scale/rmsd, (3,3) for rotation, (3,) for centroids
        Py_ssize_t rot_shape[2] = {3, 3};
        Py_ssize_t cent_shape[1] = {3};

        rot_tensor = Tensor_new(dtype, 2, rot_shape);
        scale_tensor = Tensor_new(mesh_out_dtype, 0, NULL);
        rmsd_tensor = Tensor_new(mesh_out_dtype, 0, NULL);
        a_cent_tensor = Tensor_new(dtype, 1, cent_shape);
        b_cent_tensor = Tensor_new(dtype, 1, cent_shape);

        if (!rot_tensor || !scale_tensor || !rmsd_tensor || !a_cent_tensor || !b_cent_tensor) goto cleanup;

        nk_scalar_buffer_t scale_buf = {0}, rmsd_buf = {0};
        kernel(a_buffer.buf, b_buffer.buf, num_points, a_cent_tensor->data, b_cent_tensor->data, rot_tensor->data,
               &scale_buf, &rmsd_buf);

        memcpy(scale_tensor->data, &scale_buf, bytes_per_dtype(mesh_out_dtype));
        memcpy(rmsd_tensor->data, &rmsd_buf, bytes_per_dtype(mesh_out_dtype));
    }
    else {
        // Batched case: (B, N, 3) -> rotation (B,3,3), scale (B,), rmsd (B,), centroids (B,3)
        Py_ssize_t rot_shape[3] = {batch_size, 3, 3};
        Py_ssize_t scalar_shape[1] = {batch_size};
        Py_ssize_t cent_shape[2] = {batch_size, 3};

        rot_tensor = Tensor_new(dtype, 3, rot_shape);
        scale_tensor = Tensor_new(mesh_out_dtype, 1, scalar_shape);
        rmsd_tensor = Tensor_new(mesh_out_dtype, 1, scalar_shape);
        a_cent_tensor = Tensor_new(dtype, 2, cent_shape);
        b_cent_tensor = Tensor_new(dtype, 2, cent_shape);

        if (!rot_tensor || !scale_tensor || !rmsd_tensor || !a_cent_tensor || !b_cent_tensor) goto cleanup;

        char *a_ptr = (char *)a_buffer.buf;
        char *b_ptr = (char *)b_buffer.buf;
        size_t const scalar_bytes = bytes_per_dtype(mesh_out_dtype);

        for (Py_ssize_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
            nk_scalar_buffer_t scale_buf = {0}, rmsd_buf = {0};
            size_t const elem_bytes = bytes_per_dtype(dtype);
            kernel(a_ptr + batch_idx * batch_stride_a, b_ptr + batch_idx * batch_stride_b, num_points,
                   a_cent_tensor->data + batch_idx * 3 * elem_bytes, b_cent_tensor->data + batch_idx * 3 * elem_bytes,
                   rot_tensor->data + batch_idx * 9 * elem_bytes, &scale_buf, &rmsd_buf);

            memcpy(scale_tensor->data + batch_idx * scalar_bytes, &scale_buf, scalar_bytes);
            memcpy(rmsd_tensor->data + batch_idx * scalar_bytes, &rmsd_buf, scalar_bytes);
        }
    }

    // Build MeshAlignmentResult
    {
        MeshAlignmentResultObject *mesh_result = PyObject_New(MeshAlignmentResultObject, &MeshAlignmentResultType);
        if (mesh_result) {
            mesh_result->rotation = (PyObject *)rot_tensor;
            Py_INCREF(rot_tensor);
            mesh_result->scale = (PyObject *)scale_tensor;
            Py_INCREF(scale_tensor);
            mesh_result->rmsd = (PyObject *)rmsd_tensor;
            Py_INCREF(rmsd_tensor);
            mesh_result->a_centroid = (PyObject *)a_cent_tensor;
            Py_INCREF(a_cent_tensor);
            mesh_result->b_centroid = (PyObject *)b_cent_tensor;
            Py_INCREF(b_cent_tensor);
            result = (PyObject *)mesh_result;
        }
    }

cleanup:
    // Individual tensors are always decref'd; if result was created, it holds its own references
    Py_XDECREF(rot_tensor);
    Py_XDECREF(scale_tensor);
    Py_XDECREF(rmsd_tensor);
    Py_XDECREF(a_cent_tensor);
    Py_XDECREF(b_cent_tensor);
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
    {"euclidean", (PyCFunction)api_euclidean, METH_FASTCALL | METH_KEYWORDS, doc_euclidean},
    {"sqeuclidean", (PyCFunction)api_sqeuclidean, METH_FASTCALL | METH_KEYWORDS, doc_sqeuclidean},
    {"kld", (PyCFunction)api_kld, METH_FASTCALL | METH_KEYWORDS, doc_kld},
    {"jsd", (PyCFunction)api_jsd, METH_FASTCALL | METH_KEYWORDS, doc_jsd},
    {"angular", (PyCFunction)api_angular, METH_FASTCALL | METH_KEYWORDS, doc_angular},
    {"dot", (PyCFunction)api_dot, METH_FASTCALL | METH_KEYWORDS, doc_dot},
    {"vdot", (PyCFunction)api_vdot, METH_FASTCALL | METH_KEYWORDS, doc_vdot},
    {"hamming", (PyCFunction)api_hamming, METH_FASTCALL | METH_KEYWORDS, doc_hamming},
    {"jaccard", (PyCFunction)api_jaccard, METH_FASTCALL | METH_KEYWORDS, doc_jaccard},

    // Aliases
    {"inner", (PyCFunction)api_dot, METH_FASTCALL | METH_KEYWORDS, doc_dot},
    {"kullbackleibler", (PyCFunction)api_kld, METH_FASTCALL | METH_KEYWORDS, doc_kld},
    {"jensenshannon", (PyCFunction)api_jsd, METH_FASTCALL | METH_KEYWORDS, doc_jsd},

    // Conventional `cdist` interface for pairwise distances
    {"cdist", (PyCFunction)api_cdist, METH_FASTCALL | METH_KEYWORDS, doc_cdist},

    // Exposing underlying API for USearch `CompiledMetric`
    {"pointer_to_euclidean", (PyCFunction)api_euclidean_pointer, METH_O, doc_euclidean_pointer},
    {"pointer_to_sqeuclidean", (PyCFunction)api_sqeuclidean_pointer, METH_O, doc_sqeuclidean_pointer},
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

    // Tensor constructors
    {"empty", (PyCFunction)api_empty, METH_FASTCALL | METH_KEYWORDS, doc_empty},
    {"zeros", (PyCFunction)api_zeros, METH_FASTCALL | METH_KEYWORDS, doc_zeros},
    {"ones", (PyCFunction)api_ones, METH_FASTCALL | METH_KEYWORDS, doc_ones},
    {"full", (PyCFunction)api_full, METH_FASTCALL | METH_KEYWORDS, doc_full},

    // Tensor reductions
    {"moments", (PyCFunction)api_moments, METH_FASTCALL | METH_KEYWORDS, doc_reduce_moments},
    {"minmax", (PyCFunction)api_minmax, METH_FASTCALL | METH_KEYWORDS, doc_reduce_minmax},

    // Vectorized operations
    {"fma", (PyCFunction)api_fma, METH_FASTCALL | METH_KEYWORDS, doc_fma},
    {"wsum", (PyCFunction)api_wsum, METH_FASTCALL | METH_KEYWORDS, doc_wsum},
    {"scale", (PyCFunction)api_scale, METH_FASTCALL | METH_KEYWORDS, doc_scale},
    {"add", (PyCFunction)api_add, METH_FASTCALL | METH_KEYWORDS, doc_add},
    {"multiply", (PyCFunction)api_multiply, METH_FASTCALL | METH_KEYWORDS, doc_multiply},

    // Element-wise trigonometric functions
    {"sin", (PyCFunction)api_sin, METH_FASTCALL | METH_KEYWORDS, doc_sin},
    {"cos", (PyCFunction)api_cos, METH_FASTCALL | METH_KEYWORDS, doc_cos},
    {"atan", (PyCFunction)api_atan, METH_FASTCALL | METH_KEYWORDS, doc_atan},

    // Mesh alignment (point cloud registration)
    {"kabsch", (PyCFunction)api_kabsch, METH_FASTCALL | METH_KEYWORDS, doc_kabsch},
    {"umeyama", (PyCFunction)api_umeyama, METH_FASTCALL | METH_KEYWORDS, doc_umeyama},
    {"rmsd", (PyCFunction)api_rmsd_mesh, METH_FASTCALL | METH_KEYWORDS, doc_rmsd},

    // Matrix multiplication with pre-packed matrices
    {"pack_matmul_argument", (PyCFunction)api_pack_matmul_argument, METH_FASTCALL | METH_KEYWORDS,
     doc_pack_matmul_argument},
    {"pack_matrix", (PyCFunction)api_pack_matrix, METH_FASTCALL | METH_KEYWORDS, doc_pack_matrix},
    {"matmul", (PyCFunction)api_matmul, METH_FASTCALL | METH_KEYWORDS, doc_matmul},

    // Sentinel
    {NULL, NULL, 0, NULL}};

static char const doc_module[] =                                                                    //
    "Portable mixed-precision BLAS-like vector math library for x86 and Arm.\n"                     //
    "\n"                                                                                            //
    "Performance Recommendations:\n"                                                                //
    " - Avoid converting to NumPy arrays. NumKong works with any tensor implementation\n"           //
    "   compatible with the Python buffer protocol, including PyTorch and TensorFlow.\n"            //
    " - In low-latency environments, provide the output array with the `out=` parameter\n"          //
    "   to avoid expensive memory allocations on the hot path.\n"                                   //
    " - On modern CPUs, when the application allows, prefer low-precision numeric types.\n"         //
    "   Whenever possible, use 'bf16' and 'f16' over 'f32'. Consider quantizing to 'i8'\n"          //
    "   and 'u8' for highest hardware compatibility and performance.\n"                             //
    " - If you only need relative proximity rather than absolute distance, prefer simpler\n"        //
    "   kernels such as squared Euclidean distance over Euclidean distance.\n"                      //
    " - Use row-major contiguous matrix representations. Strides between rows do not have\n"        //
    "   a significant impact on performance, but most modern HPC packages explicitly ban\n"         //
    "   non-contiguous rows where nearby cells within a row have multi-byte gaps.\n"                //
    " - The CPython runtime has noticeable overhead for function calls, so consider batching\n"     //
    "   kernel invocations. Many kernels compute 1-to-1 distances between vectors, as well as\n"    //
    "   1-to-N and N-to-N distances between batches of vectors packed into matrices.\n"             //
    "\n"                                                                                            //
    "Example:\n"                                                                                    //
    "    >>> import numkong\n"                                                                      //
    "    >>> numkong.euclidean(a, b)\n"                                                             //
    "\n"                                                                                            //
    "Mixed-precision 1-to-N example with numeric types missing in NumPy, but present in PyTorch:\n" //
    "    >>> import numkong\n"                                                                      //
    "    >>> import torch\n"                                                                        //
    "    >>> a = torch.randn(1536, dtype=torch.bfloat16)\n"                                         //
    "    >>> b = torch.randn((100, 1536), dtype=torch.bfloat16)\n"                                  //
    "    >>> c = torch.zeros(100, dtype=torch.float32)\n"                                           //
    "    >>> numkong.euclidean(a, b, dtype='bfloat16', out=c)\n";

static PyModuleDef nk_module = {
    PyModuleDef_HEAD_INIT, .m_name = "NumKong", .m_doc = doc_module, .m_size = -1, .m_methods = nk_methods,
};

PyMODINIT_FUNC PyInit_numkong(void) {
    PyObject *m;

    if (PyType_Ready(&TensorType) < 0) return NULL;
    if (PyType_Ready(&TensorIterType) < 0) return NULL;
    if (PyType_Ready(&TransposedMatrixMultiplierType) < 0) return NULL;
    if (PyType_Ready(&MeshAlignmentResultType) < 0) return NULL;

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

    // Register Tensor type (primary name)
    Py_INCREF(&TensorType);
    if (PyModule_AddObject(m, "Tensor", (PyObject *)&TensorType) < 0) {
        Py_XDECREF(&TensorType);
        Py_XDECREF(m);
        return NULL;
    }

    // Register TransposedMatrixMultiplier type (primary name)
    Py_INCREF(&TransposedMatrixMultiplierType);
    if (PyModule_AddObject(m, "TransposedMatrixMultiplier", (PyObject *)&TransposedMatrixMultiplierType) < 0) {
        Py_XDECREF(&TransposedMatrixMultiplierType);
        Py_XDECREF(m);
        return NULL;
    }

    // Register MeshAlignmentResult type
    Py_INCREF(&MeshAlignmentResultType);
    if (PyModule_AddObject(m, "MeshAlignmentResult", (PyObject *)&MeshAlignmentResultType) < 0) {
        Py_XDECREF(&MeshAlignmentResultType);
        Py_XDECREF(m);
        return NULL;
    }

    static_capabilities = nk_capabilities();

    // Register scalar types (bfloat16, float8_e4m3, float8_e5m2)
    if (nk_register_scalar_types(m) < 0) {
        Py_XDECREF(m);
        return NULL;
    }
    return m;
}

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

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>

#include <numkong/numkong.h>

#include "numkong.h"
#include "tensor.h"
#include "matrix.h"
#include "scalars.h"
#include "distance.h"
#include "each.h"
#include "mesh.h"

nk_capability_t static_capabilities = 0;

double nk_scalar_buffer_get_f64(nk_scalar_buffer_t const *buf, nk_dtype_t dtype) {
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
    case nk_e4m3_k: {
        nk_f32_t f32_tmp;
        nk_e4m3_to_f32(&buf->u8, &f32_tmp);
        return (double)f32_tmp;
    }
    case nk_e5m2_k: {
        nk_f32_t f32_tmp;
        nk_e5m2_to_f32(&buf->u8, &f32_tmp);
        return (double)f32_tmp;
    }
    case nk_e2m3_k: {
        nk_f32_t f32_tmp;
        nk_e2m3_to_f32(&buf->u8, &f32_tmp);
        return (double)f32_tmp;
    }
    case nk_e3m2_k: {
        nk_f32_t f32_tmp;
        nk_e3m2_to_f32(&buf->u8, &f32_tmp);
        return (double)f32_tmp;
    }
    default: return 0.0;
    }
}

void nk_scalar_buffer_set_f64(nk_scalar_buffer_t *buf, double value, nk_dtype_t dtype) {
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
    case nk_e4m3_k: {
        nk_f32_t f32_tmp = (nk_f32_t)value;
        nk_f32_to_e4m3(&f32_tmp, &buf->u8);
        break;
    }
    case nk_e5m2_k: {
        nk_f32_t f32_tmp = (nk_f32_t)value;
        nk_f32_to_e5m2(&f32_tmp, &buf->u8);
        break;
    }
    case nk_e2m3_k: {
        nk_f32_t f32_tmp = (nk_f32_t)value;
        nk_f32_to_e2m3(&f32_tmp, &buf->u8);
        break;
    }
    case nk_e3m2_k: {
        nk_f32_t f32_tmp = (nk_f32_t)value;
        nk_f32_to_e3m2(&f32_tmp, &buf->u8);
        break;
    }
    default: break;
    }
}

nk_dtype_info_t const nk_dtype_table[] = {
    {nk_f64_k, "float64", "d", "<f8", sizeof(nk_f64_t), 0},
    {nk_f32_k, "float32", "f", "<f4", sizeof(nk_f32_t), 0},
    {nk_f16_k, "float16", "e", "<f2", sizeof(nk_f16_t), 0},
    {nk_bf16_k, "bfloat16", "bf16", "<V2", sizeof(nk_bf16_t), 0},
    {nk_e4m3_k, "e4m3", "e4m3", "|V1", sizeof(nk_e4m3_t), 0},
    {nk_e5m2_k, "e5m2", "e5m2", "|V1", sizeof(nk_e5m2_t), 0},
    {nk_e2m3_k, "e2m3", "e2m3", "|V1", sizeof(nk_e2m3_t), 0},
    {nk_e3m2_k, "e3m2", "e3m2", "|V1", sizeof(nk_e3m2_t), 0},
    {nk_i4_k, "int4", "i4", "|V1", sizeof(nk_i4x2_t), 0},
    {nk_u4_k, "uint4", "u4", "|V1", sizeof(nk_u4x2_t), 0},
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

    // FP6 formats (MX-focused 6-bit floats):
    else if (same_string(name, "e2m3")) return nk_e2m3_k;
    else if (same_string(name, "e3m2")) return nk_e3m2_k;

    // Sub-byte integers:
    else if (same_string(name, "int4")) return nk_i4_k;
    else if (same_string(name, "uint4")) return nk_u4_k;

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

PyObject *scalar_to_py_number(nk_scalar_buffer_t const *buf, nk_dtype_t dtype) {
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
    case nk_e4m3_k: {
        nk_f32_t f32_tmp;
        nk_e4m3_to_f32(&buf->u8, &f32_tmp);
        return PyFloat_FromDouble((double)f32_tmp);
    }
    case nk_e5m2_k: {
        nk_f32_t f32_tmp;
        nk_e5m2_to_f32(&buf->u8, &f32_tmp);
        return PyFloat_FromDouble((double)f32_tmp);
    }
    case nk_e2m3_k: {
        nk_f32_t f32_tmp;
        nk_e2m3_to_f32(&buf->u8, &f32_tmp);
        return PyFloat_FromDouble((double)f32_tmp);
    }
    case nk_e3m2_k: {
        nk_f32_t f32_tmp;
        nk_e3m2_to_f32(&buf->u8, &f32_tmp);
        return PyFloat_FromDouble((double)f32_tmp);
    }
    default: return PyFloat_FromDouble(0.0);
    }
}

int py_number_to_scalar_buffer(PyObject *obj, nk_scalar_buffer_t *buf, nk_dtype_t dtype) {
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

int cast_scalar_buffer(nk_scalar_buffer_t const *buf, nk_dtype_t src_dtype, nk_dtype_t dst_dtype, void *target) {
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

int is_scalar(PyObject *obj) {
    if (PyFloat_Check(obj) || PyLong_Check(obj)) return 1;
    // Check for NumPy scalar types (0D arrays or numpy.generic subclasses)
    if (PyNumber_Check(obj)) {
        // 0D numpy arrays and numpy scalars (e.g. np.int8(-11)) support the buffer protocol
        // but have ndim == 0. Check if the object has ndim attribute == 0.
        PyObject *ndim_obj = PyObject_GetAttrString(obj, "ndim");
        if (ndim_obj) {
            long ndim = PyLong_AsLong(ndim_obj);
            Py_DECREF(ndim_obj);
            if (ndim == 0) return 1;
        }
        else { PyErr_Clear(); }
    }
    return 0;
}

int get_scalar_value(PyObject *obj, double *value) {
    if (PyFloat_Check(obj)) {
        *value = PyFloat_AsDouble(obj);
        return !PyErr_Occurred();
    }
    else if (PyLong_Check(obj)) {
        *value = PyLong_AsDouble(obj);
        return !PyErr_Occurred();
    }
    // Handle NumPy scalars and 0D arrays via PyNumber_Float
    PyObject *as_float = PyNumber_Float(obj);
    if (as_float) {
        *value = PyFloat_AsDouble(as_float);
        Py_DECREF(as_float);
        return !PyErr_Occurred();
    }
    PyErr_Clear();
    return 0;
}

int parse_tensor(PyObject *tensor, Py_buffer *buffer, MatrixOrVectorView *parsed) {
    if (PyObject_GetBuffer(tensor, buffer, PyBUF_STRIDES | PyBUF_FORMAT) != 0) {
        PyErr_SetString(PyExc_TypeError, "arguments must support buffer protocol");
        return 0;
    }

    if (buffer->ndim > NK_TENSOR_MAX_RANK) {
        PyErr_Format(PyExc_ValueError, "Tensor rank %d exceeds maximum supported rank %d", buffer->ndim,
                     NK_TENSOR_MAX_RANK);
        PyBuffer_Release(buffer);
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

static struct {
    char const *name;
    nk_capability_t flag;
} const cap_table[] = {
    {"serial", nk_cap_serial_k},
    // ARM NEON
    {"neon", nk_cap_neon_k},
    {"neonhalf", nk_cap_neonhalf_k},
    {"neonfhm", nk_cap_neonfhm_k},
    {"neonbfdot", nk_cap_neonbfdot_k},
    {"neonsdot", nk_cap_neonsdot_k},
    // ARM SVE
    {"sve", nk_cap_sve_k},
    {"svehalf", nk_cap_svehalf_k},
    {"svebfdot", nk_cap_svebfdot_k},
    {"svesdot", nk_cap_svesdot_k},
    {"sve2", nk_cap_sve2_k},
    {"sve2p1", nk_cap_sve2p1_k},
    // ARM SME
    {"sme", nk_cap_sme_k},
    {"sme2", nk_cap_sme2_k},
    {"sme2p1", nk_cap_sme2p1_k},
    {"smef64", nk_cap_smef64_k},
    {"smehalf", nk_cap_smehalf_k},
    {"smebf16", nk_cap_smebf16_k},
    {"smelut2", nk_cap_smelut2_k},
    {"smefa64", nk_cap_smefa64_k},
    // x86
    {"haswell", nk_cap_haswell_k},
    {"skylake", nk_cap_skylake_k},
    {"icelake", nk_cap_icelake_k},
    {"genoa", nk_cap_genoa_k},
    {"sapphire", nk_cap_sapphire_k},
    {"sapphireamx", nk_cap_sapphireamx_k},
    {"graniteamx", nk_cap_graniteamx_k},
    {"turin", nk_cap_turin_k},
    {"sierra", nk_cap_sierra_k},
    // RISC-V
    {"rvv", nk_cap_rvv_k},
    {"rvvhalf", nk_cap_rvvhalf_k},
    {"rvvbf16", nk_cap_rvvbf16_k},
    {"rvvbb", nk_cap_rvvbb_k},
    // WASM
    {"v128relaxed", nk_cap_v128relaxed_k},
    {NULL, 0},
};

char const doc_enable_capability[] =            //
    "Enable a specific SIMD kernel family.\n\n" //
    "Parameters:\n"                             //
    "    capability : str\n"                    //
    "        Name of the SIMD feature to enable (for example, 'haswell').";

PyObject *api_enable_capability(PyObject *self, PyObject *cap_name_obj) {
    char const *cap_name = PyUnicode_AsUTF8(cap_name_obj);
    if (!cap_name) {
        PyErr_SetString(PyExc_TypeError, "Capability name must be a string");
        return NULL;
    }

    for (size_t i = 0; cap_table[i].name; ++i) {
        if (same_string(cap_name, cap_table[i].name)) {
            if (cap_table[i].flag == nk_cap_serial_k) {
                PyErr_SetString(PyExc_ValueError, "Can't change the serial functionality");
                return NULL;
            }
            static_capabilities |= cap_table[i].flag;
            Py_RETURN_NONE;
        }
    }

    PyErr_SetString(PyExc_ValueError, "Unknown capability");
    return NULL;
}

char const doc_disable_capability[] =            //
    "Disable a specific SIMD kernel family.\n\n" //
    "Parameters:\n"                              //
    "    capability : str\n"                     //
    "        Name of the SIMD feature to disable (for example, 'haswell').";

PyObject *api_disable_capability(PyObject *self, PyObject *cap_name_obj) {
    char const *cap_name = PyUnicode_AsUTF8(cap_name_obj);
    if (!cap_name) {
        PyErr_SetString(PyExc_TypeError, "Capability name must be a string");
        return NULL;
    }

    for (size_t i = 0; cap_table[i].name; ++i) {
        if (same_string(cap_name, cap_table[i].name)) {
            if (cap_table[i].flag == nk_cap_serial_k) {
                PyErr_SetString(PyExc_ValueError, "Can't change the serial functionality");
                return NULL;
            }
            static_capabilities &= ~cap_table[i].flag;
            Py_RETURN_NONE;
        }
    }

    PyErr_SetString(PyExc_ValueError, "Unknown capability");
    return NULL;
}

char const doc_get_capabilities[] =                                                                        //
    "Get the current hardware SIMD capabilities as a dictionary of feature flags.\n\n"                     //
    "The dictionary maps capability names to booleans. Available capabilities:\n"                          //
    "  x86: serial, haswell, skylake, icelake, genoa, sapphire, sapphireamx, graniteamx, turin, sierra.\n" //
    "  ARM NEON: neon, neonhalf, neonfhm, neonbfdot, neonsdot.\n"                                          //
    "  ARM SVE: sve, svehalf, svebfdot, svesdot, sve2, sve2p1.\n"                                          //
    "  ARM SME: sme, sme2, sme2p1, smef64, smehalf, smebf16, smelut2, smefa64.\n"                          //
    "  RISC-V: rvv, rvvhalf, rvvbf16, rvvbb.\n"                                                            //
    "  WASM: v128relaxed.\n";

PyObject *api_get_capabilities(PyObject *self) {
    nk_capability_t caps = static_capabilities;
    PyObject *cap_dict = PyDict_New();
    if (!cap_dict) return NULL;

    for (size_t i = 0; cap_table[i].name; ++i) {
        PyObject *val = PyBool_FromLong(caps & cap_table[i].flag);
        if (PyDict_SetItemString(cap_dict, cap_table[i].name, val) < 0) {
            Py_DECREF(val);
            Py_DECREF(cap_dict);
            return NULL;
        }
        Py_DECREF(val);
    }

    return cap_dict;
}

nk_dtype_t promote_dtypes(nk_dtype_t a, nk_dtype_t b) {
    if (a == b) return a;

    // Classify dtype into class (1=float, 2=signed int, 3=unsigned int, 4=complex) and rank
    // using library helpers nk_dtype_family() and nk_dtype_bits().
    int class_a = 0, class_b = 0;
    int rank_a = 0, rank_b = 0;

    // Helper: classify a single dtype
#define CLASSIFY_DTYPE(dt, cls, rnk)                                                   \
    do {                                                                               \
        nk_dtype_family_t fam = nk_dtype_family(dt);                                   \
        nk_size_t bits = nk_dtype_bits(dt);                                            \
        if (fam == nk_dtype_family_float_k) {                                          \
            cls = 1;                                                                   \
            rnk = bits <= 8 ? 1 : bits <= 16 ? 2 : bits <= 32 ? 3 : 4;                 \
        }                                                                              \
        else if (fam == nk_dtype_family_int_k) {                                       \
            cls = 2;                                                                   \
            rnk = bits <= 4 ? 0 : bits <= 8 ? 1 : bits <= 16 ? 2 : bits <= 32 ? 3 : 4; \
        }                                                                              \
        else if (fam == nk_dtype_family_uint_k) {                                      \
            cls = 3;                                                                   \
            rnk = bits <= 4 ? 0 : bits <= 8 ? 1 : bits <= 16 ? 2 : bits <= 32 ? 3 : 4; \
        }                                                                              \
        else if (fam == nk_dtype_family_complex_float_k) {                             \
            cls = 4;                                                                   \
            rnk = bits <= 32 ? 2 : bits <= 64 ? 3 : 4;                                 \
        }                                                                              \
        else { return nk_dtype_unknown_k; }                                            \
    } while (0)

    CLASSIFY_DTYPE(a, class_a, rank_a);
    CLASSIFY_DTYPE(b, class_b, rank_b);

#undef CLASSIFY_DTYPE

    // Same class: return wider
    if (class_a == class_b) {
        // Float + float -> wider
        if (class_a == 1) {
            // Exotic floats (e4m3, e5m2, bf16) mixed with standard floats -> promote through f32
            if (rank_a == 1 || rank_b == 1 || a == nk_bf16_k || b == nk_bf16_k) {
                // If both are exotic rank-1, promote to f32
                if (rank_a <= 2 && rank_b <= 2) return nk_f32_k;
                // Otherwise take the wider standard float
                return rank_a >= rank_b ? a : b;
            }
            return rank_a >= rank_b ? a : b;
        }
        // Signed int + signed int -> wider
        if (class_a == 2) {
            static nk_dtype_t const signed_ints[] = {nk_i4_k, nk_i8_k, nk_i16_k, nk_i32_k, nk_i64_k};
            return signed_ints[rank_a >= rank_b ? rank_a : rank_b];
        }
        // Unsigned int + unsigned int -> wider
        if (class_a == 3) {
            static nk_dtype_t const unsigned_ints[] = {nk_u4_k, nk_u8_k, nk_u16_k, nk_u32_k, nk_u64_k};
            return unsigned_ints[rank_a >= rank_b ? rank_a : rank_b];
        }
        // Complex + complex -> wider
        if (class_a == 4) { return rank_a >= rank_b ? a : b; }
    }

    // Signed + unsigned -> next wider signed
    if ((class_a == 2 && class_b == 3) || (class_a == 3 && class_b == 2)) {
        int max_rank = rank_a >= rank_b ? rank_a : rank_b;
        // Need one rank wider than the unsigned to accommodate all values
        int unsigned_rank = (class_a == 3) ? rank_a : rank_b;
        int signed_rank = (class_a == 2) ? rank_a : rank_b;
        // If unsigned rank >= signed rank, need next wider signed
        if (unsigned_rank >= signed_rank) {
            int target_rank = unsigned_rank + 1;
            if (target_rank > 4) return nk_dtype_unknown_k; // overflow
            static nk_dtype_t const signed_ints[] = {nk_i4_k, nk_i8_k, nk_i16_k, nk_i32_k, nk_i64_k};
            return signed_ints[target_rank];
        }
        // Signed is already wider
        static nk_dtype_t const signed_ints[] = {nk_i4_k, nk_i8_k, nk_i16_k, nk_i32_k, nk_i64_k};
        return signed_ints[max_rank];
    }

    // Int + float -> float wide enough
    if ((class_a == 1 && (class_b == 2 || class_b == 3)) || ((class_a == 2 || class_a == 3) && class_b == 1)) {
        int float_rank = (class_a == 1) ? rank_a : rank_b;
        int int_rank = (class_a == 1) ? rank_b : rank_a;
        // i8/u8 + f32 -> f32, i32/u32 + f32 -> f64, i64/u64 + f32 -> f64
        int target_float_rank = float_rank;
        if (int_rank >= 3 && float_rank <= 3) target_float_rank = 4;      // promote to f64
        else if (int_rank >= 1 && float_rank <= 2) target_float_rank = 3; // promote to f32
        static nk_dtype_t const floats[] = {0, nk_f32_k, nk_f32_k, nk_f32_k, nk_f64_k};
        return floats[target_float_rank];
    }

    // Complex + real -> complex with promoted component
    if (class_a == 4 || class_b == 4) {
        int complex_rank = (class_a == 4) ? rank_a : rank_b;
        int other_rank = (class_a == 4) ? rank_b : rank_a;
        int target_rank = complex_rank >= other_rank ? complex_rank : other_rank;
        if (target_rank >= 4) return nk_f64c_k;
        return nk_f32c_k;
    }

    return nk_dtype_unknown_k;
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
    {"sparse_dot", (PyCFunction)api_sparse_dot, METH_FASTCALL, doc_sparse_dot},

    // Symmetric pairwise operations
    {"dots_symmetric", (PyCFunction)api_dots_symmetric, METH_FASTCALL | METH_KEYWORDS, doc_dots_symmetric},
    {"hammings_symmetric", (PyCFunction)api_hammings_symmetric, METH_FASTCALL | METH_KEYWORDS, doc_hammings_symmetric},

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
    {"rmsd", (PyCFunction)api_rmsd, METH_FASTCALL | METH_KEYWORDS, doc_rmsd},

    // Matrix multiplication with pre-packed matrices
    {"dots_pack", (PyCFunction)api_dots_pack, METH_FASTCALL | METH_KEYWORDS, doc_dots_pack},
    {"dots_packed", (PyCFunction)api_dots_packed, METH_FASTCALL | METH_KEYWORDS, doc_dots_packed},
    {"hammings_pack", (PyCFunction)api_hammings_pack, METH_FASTCALL | METH_KEYWORDS, doc_hammings_pack},
    {"hammings_packed", (PyCFunction)api_hammings_packed, METH_FASTCALL | METH_KEYWORDS, doc_hammings_packed},

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
    if (PyType_Ready(&PackedMatrixType) < 0) return NULL;
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

    // Register Tensor type
    Py_INCREF(&TensorType);
    if (PyModule_AddObject(m, "Tensor", (PyObject *)&TensorType) < 0) {
        Py_XDECREF(&TensorType);
        Py_XDECREF(m);
        return NULL;
    }

    // Register PackedMatrix type
    Py_INCREF(&PackedMatrixType);
    if (PyModule_AddObject(m, "PackedMatrix", (PyObject *)&PackedMatrixType) < 0) {
        Py_XDECREF(&PackedMatrixType);
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

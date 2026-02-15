/**
 *  @brief Tensor implementation for NumKong Python bindings.
 *  @file python/tensor.c
 *
 *  This file implements the Tensor N-dimensional array type with NumPy-like
 *  interface, the TensorIter iterator, and the TransposedMatrixMultiplier type
 *  for optimized matrix multiplication.
 *
 *  Features:
 *  - Support for all NumKong dtypes (f32, f64, f16, bf16, i8, complex, etc.)
 *  - Arbitrary strides for views and slices
 *  - Zero-copy views with reference counting
 *  - Python buffer protocol for interoperability
 *  - Arithmetic operators (+, -, *, @)
 *  - Reduction operations (sum, min, max, argmin, argmax)
 */
#include "tensor.h"
#include "numerics.h"

#include <float.h>
#include <stdint.h>

#include "numkong/dots.h"

static PyObject *tensor_read_scalar(Tensor *tensor, size_t byte_offset);
static int tensor_is_c_contig(Tensor *tensor, size_t item_size);
static int tensor_is_f_contig(Tensor *tensor, size_t item_size);

typedef enum {
    tensor_scalar_kind_float,
    tensor_scalar_kind_complex,
    tensor_scalar_kind_int,
    tensor_scalar_kind_uint,
} tensor_scalar_kind_t;

typedef struct {
    tensor_scalar_kind_t kind;
    double real;
    double imag;
    int64_t i64;
    uint64_t u64;
} tensor_scalar_value_t;

/// @brief Read a scalar value from a tensor at a given byte offset.
static int tensor_read_scalar_value(Tensor *tensor, size_t byte_offset, tensor_scalar_value_t *value) {
    char *ptr = tensor->data + byte_offset;
    switch (tensor->dtype) {
    case nk_f64_k:
        value->kind = tensor_scalar_kind_float;
        value->real = *(nk_f64_t *)ptr;
        return 1;
    case nk_f32_k:
        value->kind = tensor_scalar_kind_float;
        value->real = *(nk_f32_t *)ptr;
        return 1;
    case nk_f16_k: {
        nk_f32_t tmp;
        nk_f16_to_f32((nk_f16_t *)ptr, &tmp);
        value->kind = tensor_scalar_kind_float;
        value->real = tmp;
        return 1;
    }
    case nk_bf16_k: {
        nk_f32_t tmp;
        nk_bf16_to_f32((nk_bf16_t *)ptr, &tmp);
        value->kind = tensor_scalar_kind_float;
        value->real = tmp;
        return 1;
    }
    case nk_f64c_k:
        value->kind = tensor_scalar_kind_complex;
        value->real = ((nk_f64_t *)ptr)[0];
        value->imag = ((nk_f64_t *)ptr)[1];
        return 1;
    case nk_f32c_k:
        value->kind = tensor_scalar_kind_complex;
        value->real = ((nk_f32_t *)ptr)[0];
        value->imag = ((nk_f32_t *)ptr)[1];
        return 1;
    case nk_i8_k:
        value->kind = tensor_scalar_kind_int;
        value->i64 = *(nk_i8_t *)ptr;
        return 1;
    case nk_u8_k:
        value->kind = tensor_scalar_kind_uint;
        value->u64 = *(nk_u8_t *)ptr;
        return 1;
    case nk_i16_k:
        value->kind = tensor_scalar_kind_int;
        value->i64 = *(nk_i16_t *)ptr;
        return 1;
    case nk_u16_k:
        value->kind = tensor_scalar_kind_uint;
        value->u64 = *(nk_u16_t *)ptr;
        return 1;
    case nk_i32_k:
        value->kind = tensor_scalar_kind_int;
        value->i64 = *(nk_i32_t *)ptr;
        return 1;
    case nk_u32_k:
        value->kind = tensor_scalar_kind_uint;
        value->u64 = *(nk_u32_t *)ptr;
        return 1;
    case nk_i64_k:
        value->kind = tensor_scalar_kind_int;
        value->i64 = *(nk_i64_t *)ptr;
        return 1;
    case nk_u64_k:
        value->kind = tensor_scalar_kind_uint;
        value->u64 = *(nk_u64_t *)ptr;
        return 1;
    default: return 0;
    }
}

/// @brief Return a Python scalar from a tensor byte offset.
static PyObject *tensor_read_scalar(Tensor *tensor, size_t byte_offset) {
    tensor_scalar_value_t value;
    if (!tensor_read_scalar_value(tensor, byte_offset, &value)) {
        PyErr_SetString(PyExc_TypeError, "unsupported dtype for indexing");
        return NULL;
    }
    switch (value.kind) {
    case tensor_scalar_kind_float: return PyFloat_FromDouble(value.real);
    case tensor_scalar_kind_complex: return PyComplex_FromDoubles(value.real, value.imag);
    case tensor_scalar_kind_int: return PyLong_FromLongLong(value.i64);
    case tensor_scalar_kind_uint: return PyLong_FromUnsignedLongLong(value.u64);
    default: return NULL;
    }
}

/// @brief Check if a tensor is C-contiguous (row-major).
static int tensor_is_c_contig(Tensor *tensor, size_t item_size) {
    if (tensor->rank == 0) return 1;
    Py_ssize_t expected = (Py_ssize_t)item_size;
    for (size_t i = tensor->rank; i > 0; i--) {
        if (tensor->strides[i - 1] != expected) return 0;
        expected *= tensor->shape[i - 1];
    }
    return 1;
}

/// @brief Check if a tensor is Fortran-contiguous (column-major).
static int tensor_is_f_contig(Tensor *tensor, size_t item_size) {
    if (tensor->rank == 0) return 1;
    Py_ssize_t expected = (Py_ssize_t)item_size;
    for (size_t i = 0; i < tensor->rank; i++) {
        if (tensor->strides[i] != expected) return 0;
        expected *= tensor->shape[i];
    }
    return 1;
}

static void Tensor_dealloc(PyObject *self) {
    Tensor *tensor = (Tensor *)self;
    Py_XDECREF(tensor->parent);
    Py_TYPE(self)->tp_free(self);
}

Tensor *Tensor_new(nk_dtype_t dtype, size_t rank, Py_ssize_t const *shape) {
    if (rank > NK_TENSOR_MAX_RANK) {
        PyErr_Format(PyExc_ValueError, "Tensor rank %zu exceeds maximum %d", rank, NK_TENSOR_MAX_RANK);
        return NULL;
    }

    size_t const item_size = bytes_per_dtype(dtype);
    size_t total_items = 1;
    for (size_t i = 0; i < rank; i++) total_items *= (size_t)shape[i];
    size_t const total_bytes = total_items * item_size;

    Tensor *tensor = PyObject_NewVar(Tensor, &TensorType, total_bytes);
    if (!tensor) {
        PyErr_NoMemory();
        return NULL;
    }

    tensor->dtype = dtype;
    tensor->rank = rank;

    for (size_t i = 0; i < NK_TENSOR_MAX_RANK; i++) {
        tensor->shape[i] = (i < rank) ? shape[i] : 0;
        tensor->strides[i] = 0;
    }

    if (rank > 0) {
        tensor->strides[rank - 1] = (Py_ssize_t)item_size;
        for (size_t i = rank - 1; i > 0; i--) tensor->strides[i - 1] = tensor->strides[i] * tensor->shape[i];
    }

    tensor->parent = NULL;
    tensor->data = tensor->start;

    return tensor;
}

Tensor *Tensor_view(Tensor *parent, char *data_ptr, nk_dtype_t dtype, size_t rank, Py_ssize_t const *shape,
                    Py_ssize_t const *strides) {
    if (rank > NK_TENSOR_MAX_RANK) {
        PyErr_Format(PyExc_ValueError, "View rank %zu exceeds maximum %d", rank, NK_TENSOR_MAX_RANK);
        return NULL;
    }

    Tensor *view = PyObject_NewVar(Tensor, &TensorType, 0);
    if (!view) {
        PyErr_NoMemory();
        return NULL;
    }

    view->dtype = dtype;
    view->rank = rank;

    for (size_t i = 0; i < NK_TENSOR_MAX_RANK; i++) {
        view->shape[i] = (i < rank) ? shape[i] : 0;
        view->strides[i] = (i < rank) ? strides[i] : 0;
    }

    view->parent = (PyObject *)parent;
    Py_INCREF(parent);
    view->data = data_ptr;

    return view;
}

/// @brief Create a 0D scalar tensor.
static Tensor *Tensor_scalar(nk_dtype_t dtype, void const *value) {
    size_t const item_size = bytes_per_dtype(dtype);
    Tensor *tensor = PyObject_NewVar(Tensor, &TensorType, item_size);
    if (!tensor) {
        PyErr_NoMemory();
        return NULL;
    }

    tensor->dtype = dtype;
    tensor->rank = 0;
    for (size_t i = 0; i < NK_TENSOR_MAX_RANK; i++) {
        tensor->shape[i] = 0;
        tensor->strides[i] = 0;
    }

    tensor->parent = NULL;
    tensor->data = tensor->start;
    memcpy(tensor->data, value, item_size);

    return tensor;
}

/// @brief Convert a 0D Tensor to a Python float.
static PyObject *Tensor_float(PyObject *self) {
    Tensor *tensor = (Tensor *)self;
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

/// @brief Convert a 0D Tensor to a Python int.
static PyObject *Tensor_int(PyObject *self) {
    Tensor *tensor = (Tensor *)self;
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

static PyObject *Tensor_positive(PyObject *self) { return Tensor_copy(self, NULL); }

static PyObject *Tensor_negative(PyObject *self) {
    Tensor *t = (Tensor *)self;
    Tensor *r = Tensor_new(t->dtype, t->rank, t->shape);
    if (!r) return NULL;

    size_t n = 1;
    for (size_t i = 0; i < t->rank; i++) n *= (size_t)t->shape[i];

    switch (t->dtype) {
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
        PyErr_SetString(PyExc_NotImplementedError, "negation not implemented for this dtype");
        return NULL;
    }
    return (PyObject *)r;
}

static PyObject *Tensor_add(PyObject *self, PyObject *other) {
    if (!PyObject_TypeCheck(self, &TensorType)) { Py_RETURN_NOTIMPLEMENTED; }
    Tensor *a = (Tensor *)self;

    if (PyObject_TypeCheck(other, &TensorType)) {
        Tensor *b = (Tensor *)other;
        if (a->rank != b->rank || a->dtype != b->dtype) {
            PyErr_SetString(PyExc_ValueError, "shape/dtype mismatch");
            return NULL;
        }
        for (size_t i = 0; i < a->rank; i++)
            if (a->shape[i] != b->shape[i]) {
                PyErr_SetString(PyExc_ValueError, "shape mismatch");
                return NULL;
            }

        Tensor *r = Tensor_new(a->dtype, a->rank, a->shape);
        if (!r) return NULL;

        size_t n = 1;
        for (size_t i = 0; i < a->rank; i++) n *= (size_t)a->shape[i];

        size_t item_size = bytes_per_dtype(a->dtype);
        if (impl_elementwise_add(a->data, b->data, r->data, n, a->dtype, item_size, item_size, item_size) < 0) {
            Py_DECREF(r);
            PyErr_SetString(PyExc_NotImplementedError, "add not implemented for this dtype");
            return NULL;
        }
        return (PyObject *)r;
    }

    if (PyFloat_Check(other) || PyLong_Check(other)) {
        double sc = PyFloat_Check(other) ? PyFloat_AsDouble(other) : (double)PyLong_AsLong(other);
        Tensor *r = Tensor_new(a->dtype, a->rank, a->shape);
        if (!r) return NULL;

        size_t n = 1;
        for (size_t i = 0; i < a->rank; i++) n *= (size_t)a->shape[i];

        size_t item_size = bytes_per_dtype(a->dtype);
        if (impl_elementwise_scale(a->data, r->data, n, a->dtype, 1.0, sc, item_size, item_size) < 0) {
            Py_DECREF(r);
            PyErr_SetString(PyExc_NotImplementedError, "add not implemented for this dtype");
            return NULL;
        }
        return (PyObject *)r;
    }

    Py_RETURN_NOTIMPLEMENTED;
}

static PyObject *Tensor_subtract(PyObject *self, PyObject *other) {
    if (!PyObject_TypeCheck(self, &TensorType)) { Py_RETURN_NOTIMPLEMENTED; }
    Tensor *a = (Tensor *)self;

    if (PyObject_TypeCheck(other, &TensorType)) {
        Tensor *b = (Tensor *)other;
        if (a->rank != b->rank || a->dtype != b->dtype) {
            PyErr_SetString(PyExc_ValueError, "shape/dtype mismatch");
            return NULL;
        }
        for (size_t i = 0; i < a->rank; i++)
            if (a->shape[i] != b->shape[i]) {
                PyErr_SetString(PyExc_ValueError, "shape mismatch");
                return NULL;
            }

        Tensor *r = Tensor_new(a->dtype, a->rank, a->shape);
        if (!r) return NULL;

        size_t n = 1;
        for (size_t i = 0; i < a->rank; i++) n *= (size_t)a->shape[i];

        size_t item_size = bytes_per_dtype(a->dtype);
        if (impl_elementwise_wsum(a->data, b->data, r->data, n, a->dtype, 1.0, -1.0, item_size, item_size, item_size) <
            0) {
            Py_DECREF(r);
            PyErr_SetString(PyExc_NotImplementedError, "subtract not implemented for this dtype");
            return NULL;
        }
        return (PyObject *)r;
    }

    if (PyFloat_Check(other) || PyLong_Check(other)) {
        double sc = PyFloat_Check(other) ? PyFloat_AsDouble(other) : (double)PyLong_AsLong(other);
        Tensor *r = Tensor_new(a->dtype, a->rank, a->shape);
        if (!r) return NULL;

        size_t n = 1;
        for (size_t i = 0; i < a->rank; i++) n *= (size_t)a->shape[i];

        size_t item_size = bytes_per_dtype(a->dtype);
        if (impl_elementwise_scale(a->data, r->data, n, a->dtype, 1.0, -sc, item_size, item_size) < 0) {
            Py_DECREF(r);
            PyErr_SetString(PyExc_NotImplementedError, "subtract not implemented for this dtype");
            return NULL;
        }
        return (PyObject *)r;
    }

    Py_RETURN_NOTIMPLEMENTED;
}

static PyObject *Tensor_multiply(PyObject *self, PyObject *other) {
    if (!PyObject_TypeCheck(self, &TensorType)) { Py_RETURN_NOTIMPLEMENTED; }
    Tensor *a = (Tensor *)self;

    if (PyObject_TypeCheck(other, &TensorType)) {
        Tensor *b = (Tensor *)other;
        if (a->rank != b->rank || a->dtype != b->dtype) {
            PyErr_SetString(PyExc_ValueError, "shape/dtype mismatch");
            return NULL;
        }
        for (size_t i = 0; i < a->rank; i++)
            if (a->shape[i] != b->shape[i]) {
                PyErr_SetString(PyExc_ValueError, "shape mismatch");
                return NULL;
            }

        Tensor *r = Tensor_new(a->dtype, a->rank, a->shape);
        if (!r) return NULL;

        size_t n = 1;
        for (size_t i = 0; i < a->rank; i++) n *= (size_t)a->shape[i];

        size_t item_size = bytes_per_dtype(a->dtype);
        if (impl_elementwise_mul(a->data, b->data, r->data, n, a->dtype, item_size, item_size, item_size) < 0) {
            Py_DECREF(r);
            PyErr_SetString(PyExc_NotImplementedError, "multiply not implemented for this dtype");
            return NULL;
        }
        return (PyObject *)r;
    }

    if (PyFloat_Check(other) || PyLong_Check(other)) {
        double sc = PyFloat_Check(other) ? PyFloat_AsDouble(other) : (double)PyLong_AsLong(other);
        Tensor *r = Tensor_new(a->dtype, a->rank, a->shape);
        if (!r) return NULL;

        size_t n = 1;
        for (size_t i = 0; i < a->rank; i++) n *= (size_t)a->shape[i];

        size_t item_size = bytes_per_dtype(a->dtype);
        if (impl_elementwise_scale(a->data, r->data, n, a->dtype, sc, 0.0, item_size, item_size) < 0) {
            Py_DECREF(r);
            PyErr_SetString(PyExc_NotImplementedError, "multiply not implemented for this dtype");
            return NULL;
        }
        return (PyObject *)r;
    }

    Py_RETURN_NOTIMPLEMENTED;
}

// Forward declaration for Tensor_matmul (implemented after TransposedMatrixMultiplierType is defined)
static PyObject *Tensor_matmul(PyObject *self, PyObject *other);

static PyNumberMethods Tensor_as_number = {
    .nb_add = Tensor_add,
    .nb_subtract = Tensor_subtract,
    .nb_multiply = Tensor_multiply,
    .nb_matrix_multiply = Tensor_matmul,
    .nb_negative = Tensor_negative,
    .nb_positive = Tensor_positive,
    .nb_float = Tensor_float,
    .nb_int = Tensor_int,
};

static PyObject *Tensor_get_shape(PyObject *self, void *closure) {
    (void)closure;
    Tensor *tensor = (Tensor *)self;
    PyObject *shape_tuple = PyTuple_New(tensor->rank);
    if (!shape_tuple) return NULL;
    for (size_t i = 0; i < tensor->rank; i++) {
        PyTuple_SET_ITEM(shape_tuple, i, PyLong_FromSsize_t(tensor->shape[i]));
    }
    return shape_tuple;
}

static PyObject *Tensor_get_dtype(PyObject *self, void *closure) {
    (void)closure;
    Tensor *tensor = (Tensor *)self;
    return PyUnicode_FromString(dtype_to_string(tensor->dtype));
}

static PyObject *Tensor_get_ndim(PyObject *self, void *closure) {
    (void)closure;
    Tensor *tensor = (Tensor *)self;
    return PyLong_FromSize_t(tensor->rank);
}

static PyObject *Tensor_get_size(PyObject *self, void *closure) {
    (void)closure;
    Tensor *tensor = (Tensor *)self;
    Py_ssize_t total = 1;
    for (size_t i = 0; i < tensor->rank; i++) total *= tensor->shape[i];
    return PyLong_FromSsize_t(total);
}

static PyObject *Tensor_get_nbytes(PyObject *self, void *closure) {
    (void)closure;
    Tensor *tensor = (Tensor *)self;
    Py_ssize_t total = 1;
    for (size_t i = 0; i < tensor->rank; i++) total *= tensor->shape[i];
    return PyLong_FromSsize_t(total * (Py_ssize_t)bytes_per_dtype(tensor->dtype));
}

static PyObject *Tensor_get_strides(PyObject *self, void *closure) {
    (void)closure;
    Tensor *tensor = (Tensor *)self;
    PyObject *strides_tuple = PyTuple_New(tensor->rank);
    if (!strides_tuple) return NULL;
    for (size_t i = 0; i < tensor->rank; i++) {
        PyTuple_SET_ITEM(strides_tuple, i, PyLong_FromSsize_t(tensor->strides[i]));
    }
    return strides_tuple;
}

static PyObject *Tensor_get_itemsize(PyObject *self, void *closure) {
    (void)closure;
    Tensor *tensor = (Tensor *)self;
    return PyLong_FromSize_t(bytes_per_dtype(tensor->dtype));
}

static PyObject *Tensor_get_T(PyObject *self, void *closure) {
    (void)closure;
    Tensor *tensor = (Tensor *)self;

    if (tensor->rank < 2) {
        // 0D or 1D: transpose is a view of itself
        Py_INCREF(self);
        return self;
    }

    // Reverse shape and strides
    Py_ssize_t new_shape[NK_TENSOR_MAX_RANK];
    Py_ssize_t new_strides[NK_TENSOR_MAX_RANK];
    for (size_t i = 0; i < tensor->rank; i++) {
        new_shape[i] = tensor->shape[tensor->rank - 1 - i];
        new_strides[i] = tensor->strides[tensor->rank - 1 - i];
    }

    Tensor *root_parent = tensor->parent ? (Tensor *)tensor->parent : tensor;
    return (PyObject *)Tensor_view(root_parent, tensor->data, tensor->dtype, tensor->rank, new_shape, new_strides);
}

static PyObject *Tensor_get_array_interface(PyObject *self, void *closure) {
    (void)closure;
    Tensor *tensor = (Tensor *)self;

    PyObject *dict = PyDict_New();
    if (!dict) return NULL;

    PyObject *shape = Tensor_get_shape(self, NULL);
    if (!shape) {
        Py_DECREF(dict);
        return NULL;
    }
    PyDict_SetItemString(dict, "shape", shape);
    Py_DECREF(shape);

    char const *typestr = dtype_to_array_typestr(tensor->dtype);
    PyObject *typestr_obj = PyUnicode_FromString(typestr);
    if (!typestr_obj) {
        Py_DECREF(dict);
        return NULL;
    }
    PyDict_SetItemString(dict, "typestr", typestr_obj);
    Py_DECREF(typestr_obj);

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

    PyObject *strides = Tensor_get_strides(self, NULL);
    if (!strides) {
        Py_DECREF(dict);
        return NULL;
    }
    PyDict_SetItemString(dict, "strides", strides);
    Py_DECREF(strides);

    PyDict_SetItemString(dict, "version", PyLong_FromLong(3));

    return dict;
}

static PyGetSetDef Tensor_getset[] = {
    {"shape", Tensor_get_shape, NULL, "Shape of the array", NULL},
    {"dtype", Tensor_get_dtype, NULL, "Data type of the array", NULL},
    {"ndim", Tensor_get_ndim, NULL, "Number of dimensions", NULL},
    {"size", Tensor_get_size, NULL, "Total number of elements", NULL},
    {"nbytes", Tensor_get_nbytes, NULL, "Total bytes of data", NULL},
    {"strides", Tensor_get_strides, NULL, "Strides in bytes", NULL},
    {"itemsize", Tensor_get_itemsize, NULL, "Size of one element in bytes", NULL},
    {"T", Tensor_get_T, NULL, "Transposed view of the array", NULL},
    {"__array_interface__", Tensor_get_array_interface, NULL, "NumPy array interface", NULL},
    {NULL, NULL, NULL, NULL, NULL},
};

PyObject *Tensor_copy(PyObject *self, PyObject *args) {
    (void)args;
    Tensor *tensor = (Tensor *)self;
    size_t item_size = bytes_per_dtype(tensor->dtype);

    Py_ssize_t total = 1;
    for (size_t i = 0; i < tensor->rank; i++) total *= tensor->shape[i];

    Tensor *result = PyObject_NewVar(Tensor, &TensorType, total * item_size);
    if (!result) return NULL;

    result->dtype = tensor->dtype;
    result->rank = tensor->rank;
    result->parent = NULL;
    result->data = result->start;

    Py_ssize_t stride = item_size;
    for (size_t i = tensor->rank; i > 0; i--) {
        result->shape[i - 1] = tensor->shape[i - 1];
        result->strides[i - 1] = stride;
        stride *= tensor->shape[i - 1];
    }
    for (size_t i = tensor->rank; i < NK_TENSOR_MAX_RANK; i++) {
        result->shape[i] = 0;
        result->strides[i] = 0;
    }

    if (tensor_is_c_contig(tensor, item_size)) { memcpy(result->data, tensor->data, total * item_size); }
    else {
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

PyObject *Tensor_reshape(PyObject *self, PyObject *args) {
    Tensor *tensor = (Tensor *)self;

    Py_ssize_t new_shape[NK_TENSOR_MAX_RANK];
    size_t new_rank = 0;

    if (PyTuple_GET_SIZE(args) == 1 && PyTuple_Check(PyTuple_GET_ITEM(args, 0))) {
        PyObject *shape_tuple = PyTuple_GET_ITEM(args, 0);
        new_rank = PyTuple_GET_SIZE(shape_tuple);
        if (new_rank > NK_TENSOR_MAX_RANK) {
            PyErr_Format(PyExc_ValueError, "reshape: too many dimensions (%zu > %d)", new_rank, NK_TENSOR_MAX_RANK);
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
        new_rank = PyTuple_GET_SIZE(args);
        if (new_rank > NK_TENSOR_MAX_RANK) {
            PyErr_Format(PyExc_ValueError, "reshape: too many dimensions (%zu > %d)", new_rank, NK_TENSOR_MAX_RANK);
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

    Py_ssize_t new_total = 1;
    for (size_t i = 0; i < new_rank; i++) new_total *= new_shape[i];

    Py_ssize_t old_total = 1;
    for (size_t i = 0; i < tensor->rank; i++) old_total *= tensor->shape[i];

    if (new_total != old_total) {
        PyErr_Format(PyExc_ValueError, "reshape: cannot reshape tensor of size %zd into shape with size %zd", old_total,
                     new_total);
        return NULL;
    }

    size_t item_size = bytes_per_dtype(tensor->dtype);

    if (tensor_is_c_contig(tensor, item_size)) {
        Py_ssize_t new_strides[NK_TENSOR_MAX_RANK];
        Py_ssize_t stride = item_size;
        for (size_t i = new_rank; i > 0; i--) {
            new_strides[i - 1] = stride;
            stride *= new_shape[i - 1];
        }
        Tensor *root_parent = tensor->parent ? (Tensor *)tensor->parent : tensor;
        return (PyObject *)Tensor_view(root_parent, tensor->data, tensor->dtype, new_rank, new_shape, new_strides);
    }

    // Non-contiguous: must copy
    Tensor *result = PyObject_NewVar(Tensor, &TensorType, new_total * item_size);
    if (!result) return NULL;

    result->dtype = tensor->dtype;
    result->rank = new_rank;
    result->parent = NULL;
    result->data = result->start;

    Py_ssize_t stride = item_size;
    for (size_t i = new_rank; i > 0; i--) {
        result->shape[i - 1] = new_shape[i - 1];
        result->strides[i - 1] = stride;
        stride *= new_shape[i - 1];
    }
    for (size_t i = new_rank; i < NK_TENSOR_MAX_RANK; i++) {
        result->shape[i] = 0;
        result->strides[i] = 0;
    }

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

PyObject *Tensor_moments(PyObject *self, PyObject *args) {
    (void)args;
    Tensor *tensor = (Tensor *)self;

    TensorView view = {
        .dtype = tensor->dtype,
        .rank = tensor->rank,
        .shape = tensor->shape,
        .strides = tensor->strides,
        .data = tensor->data,
    };

    double sum_f = 0, sumsq_f = 0;
    int64_t sum_i = 0, sumsq_i = 0;
    impl_reduce_moments(&view, &sum_f, &sum_i, &sumsq_f, &sumsq_i);

    switch (tensor->dtype) {
    case nk_f64_k:
    case nk_f32_k:
    case nk_f16_k:
    case nk_bf16_k: return Py_BuildValue("(dd)", sum_f, sumsq_f);
    default: return Py_BuildValue("(LL)", (long long)sum_i, (long long)sumsq_i);
    }
}

PyObject *Tensor_minmax(PyObject *self, PyObject *args) {
    (void)args;
    Tensor *tensor = (Tensor *)self;

    TensorView view = {
        .dtype = tensor->dtype,
        .rank = tensor->rank,
        .shape = tensor->shape,
        .strides = tensor->strides,
        .data = tensor->data,
    };

    double min_f = 0, max_f = 0;
    int64_t min_i = 0, max_i = 0;
    size_t min_index = 0, max_index = 0;
    impl_reduce_minmax(&view, &min_f, &min_i, &min_index, &max_f, &max_i, &max_index);

    switch (tensor->dtype) {
    case nk_f64_k:
    case nk_f32_k:
    case nk_f16_k:
    case nk_bf16_k: return Py_BuildValue("(dndn)", min_f, (Py_ssize_t)min_index, max_f, (Py_ssize_t)max_index);
    default:
        return Py_BuildValue("(LnLn)", (long long)min_i, (Py_ssize_t)min_index, (long long)max_i,
                             (Py_ssize_t)max_index);
    }
}

static PyMethodDef Tensor_methods[] = {
    {"copy", Tensor_copy, METH_NOARGS, "Return a deep copy of the tensor"},
    {"reshape", Tensor_reshape, METH_VARARGS, "Return tensor reshaped to given dimensions"},
    {"moments", Tensor_moments, METH_NOARGS, "Returns (sum, sum_of_squares) tuple"},
    {"minmax", Tensor_minmax, METH_NOARGS, "Returns (min_val, min_idx, max_val, max_idx) tuple"},
    {NULL, NULL, 0, NULL},
};

static int Tensor_getbuffer(PyObject *export_from, Py_buffer *view, int flags) {
    Tensor *tensor = (Tensor *)export_from;
    size_t const item_size = bytes_per_dtype(tensor->dtype);

    int c_contig = tensor_is_c_contig(tensor, item_size);
    int f_contig = tensor_is_f_contig(tensor, item_size);

    if ((flags & PyBUF_C_CONTIGUOUS) == PyBUF_C_CONTIGUOUS && !c_contig) {
        PyErr_SetString(PyExc_BufferError, "buffer is not C-contiguous");
        view->obj = NULL;
        return -1;
    }
    if ((flags & PyBUF_F_CONTIGUOUS) == PyBUF_F_CONTIGUOUS && !f_contig) {
        PyErr_SetString(PyExc_BufferError, "buffer is not Fortran-contiguous");
        view->obj = NULL;
        return -1;
    }
    if ((flags & PyBUF_ANY_CONTIGUOUS) == PyBUF_ANY_CONTIGUOUS && !c_contig && !f_contig) {
        PyErr_SetString(PyExc_BufferError, "buffer is not contiguous");
        view->obj = NULL;
        return -1;
    }

    size_t total_items = 1;
    for (size_t i = 0; i < tensor->rank; i++) total_items *= (size_t)tensor->shape[i];

    view->buf = tensor->data;
    view->obj = (PyObject *)tensor;
    view->len = item_size * total_items;
    view->readonly = 0;
    view->itemsize = (Py_ssize_t)item_size;

    if ((flags & PyBUF_FORMAT) == PyBUF_FORMAT) view->format = (char *)dtype_to_python_string(tensor->dtype);
    else view->format = NULL;

    if ((flags & PyBUF_ND) == PyBUF_ND) {
        view->ndim = (int)tensor->rank;
        view->shape = tensor->rank > 0 ? &tensor->shape[0] : NULL;
    }
    else {
        view->ndim = 0;
        view->shape = NULL;
    }

    if ((flags & PyBUF_STRIDES) == PyBUF_STRIDES) view->strides = tensor->rank > 0 ? &tensor->strides[0] : NULL;
    else view->strides = NULL;

    view->suboffsets = NULL;
    view->internal = NULL;

    Py_INCREF(tensor);
    return 0;
}

static void Tensor_releasebuffer(PyObject *export_from, Py_buffer *view) {
    (void)export_from;
    (void)view;
}

static PyBufferProcs Tensor_as_buffer = {
    .bf_getbuffer = Tensor_getbuffer,
    .bf_releasebuffer = Tensor_releasebuffer,
};

static Py_ssize_t Tensor_length(PyObject *self) {
    Tensor *tensor = (Tensor *)self;
    if (tensor->rank == 0) {
        PyErr_SetString(PyExc_TypeError, "len() of 0-dimensional tensor");
        return -1;
    }
    return tensor->shape[0];
}

static PySequenceMethods Tensor_as_sequence = {
    .sq_length = Tensor_length,
};

static int parse_slice(PyObject *slice, Py_ssize_t dim_size, Py_ssize_t *start, Py_ssize_t *stop, Py_ssize_t *step,
                       Py_ssize_t *slice_len) {
    if (PySlice_Unpack(slice, start, stop, step) < 0) return -1;
    *slice_len = PySlice_AdjustIndices(dim_size, start, stop, *step);
    return 0;
}

static PyObject *Tensor_subscript(PyObject *self, PyObject *key) {
    Tensor *tensor = (Tensor *)self;

    if (tensor->rank == 0) {
        PyErr_SetString(PyExc_IndexError, "0-dimensional tensor cannot be indexed");
        return NULL;
    }

    size_t item_size = bytes_per_dtype(tensor->dtype);

    // Single slice
    if (PySlice_Check(key)) {
        Py_ssize_t start, stop, step, slice_len;
        if (parse_slice(key, tensor->shape[0], &start, &stop, &step, &slice_len) < 0) return NULL;

        Py_ssize_t new_shape[NK_TENSOR_MAX_RANK];
        Py_ssize_t new_strides[NK_TENSOR_MAX_RANK];

        new_shape[0] = slice_len;
        new_strides[0] = tensor->strides[0] * step;

        for (size_t i = 1; i < tensor->rank; i++) {
            new_shape[i] = tensor->shape[i];
            new_strides[i] = tensor->strides[i];
        }

        char *view_data = tensor->data + start * tensor->strides[0];
        Tensor *root_parent = tensor->parent ? (Tensor *)tensor->parent : tensor;

        return (PyObject *)Tensor_view(root_parent, view_data, tensor->dtype, tensor->rank, new_shape, new_strides);
    }

    // Single integer
    if (PyLong_Check(key)) {
        Py_ssize_t idx = PyLong_AsSsize_t(key);
        if (idx == -1 && PyErr_Occurred()) return NULL;

        if (idx < 0) idx += tensor->shape[0];
        if (idx < 0 || idx >= tensor->shape[0]) {
            PyErr_SetString(PyExc_IndexError, "index out of bounds");
            return NULL;
        }

        if (tensor->rank == 1) { return tensor_read_scalar(tensor, idx * tensor->strides[0]); }

        // Return a view with reduced rank
        char *view_data = tensor->data + idx * tensor->strides[0];
        Tensor *root_parent = tensor->parent ? (Tensor *)tensor->parent : tensor;

        return (PyObject *)Tensor_view(root_parent, view_data, tensor->dtype, tensor->rank - 1, tensor->shape + 1,
                                       tensor->strides + 1);
    }

    PyErr_SetString(PyExc_TypeError, "indices must be integers or slices");
    return NULL;
}

static PyMappingMethods Tensor_as_mapping = {
    .mp_length = Tensor_length,
    .mp_subscript = Tensor_subscript,
};

static PyObject *Tensor_repr(PyObject *self) {
    Tensor *tensor = (Tensor *)self;

    PyObject *shape_str = Tensor_get_shape(self, NULL);
    if (!shape_str) return NULL;

    PyObject *repr = PyUnicode_FromFormat("Tensor(shape=%R, dtype='%s')", shape_str, dtype_to_string(tensor->dtype));
    Py_DECREF(shape_str);
    return repr;
}

static PyObject *Tensor_str(PyObject *self) { return Tensor_repr(self); }

static PyObject *Tensor_richcompare(PyObject *self, PyObject *other, int op) {
    (void)self;
    (void)other;
    (void)op;
    PyErr_SetString(PyExc_NotImplementedError, "comparison not implemented");
    return NULL;
}

static PyObject *TensorIter_next(PyObject *self) {
    TensorIter *iter = (TensorIter *)self;
    Tensor *array = iter->array;

    if (iter->index >= array->shape[0]) {
        return NULL; // StopIteration
    }

    PyObject *item;
    if (array->rank == 1) { item = tensor_read_scalar(array, iter->index * array->strides[0]); }
    else {
        char *view_data = array->data + iter->index * array->strides[0];
        Tensor *root_parent = array->parent ? (Tensor *)array->parent : array;
        item = (PyObject *)Tensor_view(root_parent, view_data, array->dtype, array->rank - 1, array->shape + 1,
                                       array->strides + 1);
    }

    iter->index++;
    return item;
}

static void TensorIter_dealloc(PyObject *self) {
    TensorIter *iter = (TensorIter *)self;
    Py_XDECREF(iter->array);
    Py_TYPE(self)->tp_free(self);
}

PyTypeObject TensorIterType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "numkong.TensorIter",
    .tp_basicsize = sizeof(TensorIter),
    .tp_dealloc = TensorIter_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_iter = PyObject_SelfIter,
    .tp_iternext = TensorIter_next,
};

static PyObject *Tensor_iter(PyObject *self) {
    Tensor *array = (Tensor *)self;
    if (array->rank == 0) {
        PyErr_SetString(PyExc_TypeError, "cannot iterate over 0-dimensional tensor");
        return NULL;
    }

    TensorIter *iter = PyObject_New(TensorIter, &TensorIterType);
    if (!iter) return NULL;

    iter->array = array;
    Py_INCREF(array);
    iter->index = 0;

    return (PyObject *)iter;
}

PyTypeObject TensorType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "numkong.Tensor",
    .tp_doc = "N-dimensional tensor with full NumPy-like API, supporting NumKong's type system",
    .tp_basicsize = sizeof(Tensor),
    .tp_itemsize = sizeof(char),
    .tp_dealloc = Tensor_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_as_buffer = &Tensor_as_buffer,
    .tp_as_number = &Tensor_as_number,
    .tp_as_sequence = &Tensor_as_sequence,
    .tp_as_mapping = &Tensor_as_mapping,
    .tp_getset = Tensor_getset,
    .tp_methods = Tensor_methods,
    .tp_repr = Tensor_repr,
    .tp_str = Tensor_str,
    .tp_richcompare = Tensor_richcompare,
    .tp_iter = Tensor_iter,
};

static void TransposedMatrixMultiplier_dealloc(PyObject *self) { Py_TYPE(self)->tp_free(self); }

static size_t TransposedMatrixMultiplier_compute_packed_size(TransposedMatrixMultiplier *mm) {
    switch (mm->dtype) {
    case nk_bf16_k: return nk_dots_packed_size_bf16(mm->n, mm->k);
    case nk_i8_k: return nk_dots_packed_size_i8(mm->n, mm->k);
    case nk_f32_k: return nk_dots_packed_size_f32(mm->n, mm->k);
    case nk_f64_k: return nk_dots_packed_size_f64(mm->n, mm->k);
    case nk_f16_k: return nk_dots_packed_size_f16(mm->n, mm->k);
    case nk_u8_k: return nk_dots_packed_size_u8(mm->n, mm->k);
    default: return 0;
    }
}

static PyObject *TransposedMatrixMultiplier_repr(PyObject *self) {
    TransposedMatrixMultiplier *mm = (TransposedMatrixMultiplier *)self;
    size_t packed_size = TransposedMatrixMultiplier_compute_packed_size(mm);
    return PyUnicode_FromFormat("<TransposedMatrixMultiplier n=%zu k=%zu dtype='%s' nbytes=%zu>", (size_t)mm->n,
                                (size_t)mm->k, dtype_to_string(mm->dtype), packed_size);
}

static PyObject *TransposedMatrixMultiplier_get_n(PyObject *self, void *closure) {
    (void)closure;
    return PyLong_FromSize_t(((TransposedMatrixMultiplier *)self)->n);
}

static PyObject *TransposedMatrixMultiplier_get_k(PyObject *self, void *closure) {
    (void)closure;
    return PyLong_FromSize_t(((TransposedMatrixMultiplier *)self)->k);
}

static PyObject *TransposedMatrixMultiplier_get_dtype(PyObject *self, void *closure) {
    (void)closure;
    return PyUnicode_FromString(dtype_to_string(((TransposedMatrixMultiplier *)self)->dtype));
}

static PyObject *TransposedMatrixMultiplier_get_nbytes(PyObject *self, void *closure) {
    (void)closure;
    return PyLong_FromSize_t(TransposedMatrixMultiplier_compute_packed_size((TransposedMatrixMultiplier *)self));
}

static PyGetSetDef TransposedMatrixMultiplier_getset[] = {
    {"n", TransposedMatrixMultiplier_get_n, NULL, "Number of rows in the original matrix", NULL},
    {"k", TransposedMatrixMultiplier_get_k, NULL, "Number of columns in the original matrix", NULL},
    {"dtype", TransposedMatrixMultiplier_get_dtype, NULL, "Data type of the matrix elements (bf16 or i8)", NULL},
    {"nbytes", TransposedMatrixMultiplier_get_nbytes, NULL, "Size of the packed buffer in bytes", NULL},
    {NULL, NULL, NULL, NULL, NULL},
};

PyTypeObject TransposedMatrixMultiplierType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "numkong.TransposedMatrixMultiplier",
    .tp_doc = "Pre-packed matrix optimized for matrix multiplication (AMX backend)",
    .tp_basicsize = sizeof(TransposedMatrixMultiplier),
    .tp_itemsize = sizeof(char),
    .tp_dealloc = TransposedMatrixMultiplier_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_getset = TransposedMatrixMultiplier_getset,
    .tp_repr = TransposedMatrixMultiplier_repr,
};

/// @brief Parse a Python buffer format string into a NumKong dtype.
static int buffer_dtype(Py_buffer const *buffer, nk_dtype_t *dtype) {
    char const *format = buffer->format ? buffer->format : NULL;
    if (!format) {
        PyErr_SetString(PyExc_TypeError, "Input buffer must expose a format string");
        return 0;
    }
    *dtype = python_string_to_dtype(format);
    if (*dtype == nk_dtype_unknown_k) {
        PyErr_Format(PyExc_TypeError, "Unsupported buffer format '%s'", format);
        return 0;
    }
    return 1;
}

/// @brief Convert a scalar at a byte address to float32.
static int convert_scalar_to_f32(nk_dtype_t src_dtype, char const *src, nk_f32_t *value) {
    switch (src_dtype) {
    case nk_f64_k: *value = (nk_f32_t) * (nk_f64_t const *)src; return 1;
    case nk_f32_k: *value = *(nk_f32_t const *)src; return 1;
    case nk_f16_k: nk_f16_to_f32((nk_f16_t const *)src, value); return 1;
    case nk_bf16_k: nk_bf16_to_f32((nk_bf16_t const *)src, value); return 1;
    case nk_e4m3_k: nk_e4m3_to_f32((nk_e4m3_t const *)src, value); return 1;
    case nk_e5m2_k: nk_e5m2_to_f32((nk_e5m2_t const *)src, value); return 1;
    case nk_u1_k: *value = (nk_f32_t) * (nk_u1x8_t const *)src; return 1;
    case nk_i8_k: *value = (nk_f32_t) * (nk_i8_t const *)src; return 1;
    case nk_u8_k: *value = (nk_f32_t) * (nk_u8_t const *)src; return 1;
    case nk_i16_k: *value = (nk_f32_t) * (nk_i16_t const *)src; return 1;
    case nk_u16_k: *value = (nk_f32_t) * (nk_u16_t const *)src; return 1;
    case nk_i32_k: *value = (nk_f32_t) * (nk_i32_t const *)src; return 1;
    case nk_u32_k: *value = (nk_f32_t) * (nk_u32_t const *)src; return 1;
    case nk_i64_k: *value = (nk_f32_t) * (nk_i64_t const *)src; return 1;
    case nk_u64_k: *value = (nk_f32_t) * (nk_u64_t const *)src; return 1;
    default: PyErr_SetString(PyExc_TypeError, "Unsupported dtype for matmul conversion"); return 0;
    }
}

/// @brief Convert a strided row to bf16.
static int convert_row_to_bf16(nk_dtype_t src_dtype, char const *row_start, nk_size_t count, nk_size_t element_stride,
                               nk_bf16_t *dest_row) {
    for (nk_size_t j = 0; j < count; j++) {
        nk_f32_t value = 0;
        char const *element_ptr = row_start + j * element_stride;
        if (!convert_scalar_to_f32(src_dtype, element_ptr, &value)) return 0;
        nk_f32_to_bf16(&value, &dest_row[j]);
    }
    return 1;
}

/// @brief Convert a strided row to i8 with clamping.
static int convert_row_to_i8(nk_dtype_t src_dtype, char const *row_start, nk_size_t count, nk_size_t element_stride,
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

/// @brief Convert a strided row to f32.
static int convert_row_to_f32(nk_dtype_t src_dtype, char const *row_start, nk_size_t count, nk_size_t element_stride,
                              nk_f32_t *dest_row) {
    for (nk_size_t j = 0; j < count; j++) {
        char const *element_ptr = row_start + j * element_stride;
        if (!convert_scalar_to_f32(src_dtype, element_ptr, &dest_row[j])) return 0;
    }
    return 1;
}

/// @brief Convert a strided row to f64.
static int convert_row_to_f64(nk_dtype_t src_dtype, char const *row_start, nk_size_t count, nk_size_t element_stride,
                              nk_f64_t *dest_row) {
    for (nk_size_t j = 0; j < count; j++) {
        nk_f32_t value = 0;
        char const *element_ptr = row_start + j * element_stride;
        if (!convert_scalar_to_f32(src_dtype, element_ptr, &value)) return 0;
        dest_row[j] = (nk_f64_t)value;
    }
    return 1;
}

/// @brief Convert a strided row to f16.
static int convert_row_to_f16(nk_dtype_t src_dtype, char const *row_start, nk_size_t count, nk_size_t element_stride,
                              nk_f16_t *dest_row) {
    for (nk_size_t j = 0; j < count; j++) {
        nk_f32_t value = 0;
        char const *element_ptr = row_start + j * element_stride;
        if (!convert_scalar_to_f32(src_dtype, element_ptr, &value)) return 0;
        nk_f32_to_f16(&value, &dest_row[j]);
    }
    return 1;
}

/// @brief Convert a strided row to u8 with clamping.
static int convert_row_to_u8(nk_dtype_t src_dtype, char const *row_start, nk_size_t count, nk_size_t element_stride,
                             nk_u8_t *dest_row) {
    for (nk_size_t j = 0; j < count; j++) {
        nk_f32_t value = 0;
        char const *element_ptr = row_start + j * element_stride;
        if (!convert_scalar_to_f32(src_dtype, element_ptr, &value)) return 0;
        if (value < 0) value = 0;
        if (value > 255) value = 255;
        dest_row[j] = (nk_u8_t)lround((double)value);
    }
    return 1;
}

/// @brief Get the expected output dtype for a given packed matrix dtype.
static nk_dtype_t matmul_output_dtype(nk_dtype_t packed_dtype) {
    switch (packed_dtype) {
    case nk_bf16_k: return nk_f32_k; // bf16 × bf16 → f32
    case nk_f16_k: return nk_f32_k;  // f16 × f16 → f32
    case nk_i8_k: return nk_i32_k;   // i8 × i8 → i32
    case nk_u8_k: return nk_u32_k;   // u8 × u8 → u32
    case nk_f32_k: return nk_f32_k;  // f32 × f32 → f32
    case nk_f64_k: return nk_f64_k;  // f64 × f64 → f64
    default: return nk_dtype_unknown_k;
    }
}

/// @brief Matrix multiplication operator for Tensor @ TransposedMatrixMultiplier.
static PyObject *Tensor_matmul(PyObject *self, PyObject *other) {
    if (!PyObject_TypeCheck(self, &TensorType)) { Py_RETURN_NOTIMPLEMENTED; }
    Tensor *a = (Tensor *)self;

    // Only support Tensor @ TransposedMatrixMultiplier for now
    if (!PyObject_TypeCheck(other, &TransposedMatrixMultiplierType)) {
        PyErr_SetString(
            PyExc_TypeError,
            "matmul requires TransposedMatrixMultiplier as right operand " "(use nk.pack_matmul_argument() first)");
        return NULL;
    }

    TransposedMatrixMultiplier *packed = (TransposedMatrixMultiplier *)other;

    // Validate dimensions
    if (a->rank != 2) {
        PyErr_SetString(PyExc_ValueError, "matmul requires 2D array as left operand");
        return NULL;
    }

    nk_size_t m = (nk_size_t)a->shape[0];
    nk_size_t k_a = (nk_size_t)a->shape[1];

    if (k_a != packed->k) {
        PyErr_Format(PyExc_ValueError, "Dimension mismatch: array has k=%zu but packed matrix has k=%zu", k_a,
                     packed->k);
        return NULL;
    }

    if (is_complex(a->dtype)) {
        PyErr_SetString(PyExc_TypeError, "complex matrices are not supported for matmul");
        return NULL;
    }

    if (a->strides[0] < 0 || a->strides[1] < 0) {
        PyErr_SetString(PyExc_ValueError, "matmul does not support negative strides");
        return NULL;
    }

    nk_size_t n = packed->n;
    nk_size_t k = packed->k;
    nk_size_t row_stride = (nk_size_t)a->strides[0];
    nk_size_t col_stride = (nk_size_t)a->strides[1];
    nk_dtype_t src_dtype = a->dtype;

    // Determine output dtype based on packed matrix dtype
    nk_dtype_t out_dtype = matmul_output_dtype(packed->dtype);
    if (out_dtype == nk_dtype_unknown_k) {
        PyErr_SetString(PyExc_ValueError, "Unsupported packed matrix dtype");
        return NULL;
    }

    // Allocate output tensor
    Py_ssize_t out_shape[2] = {(Py_ssize_t)m, (Py_ssize_t)n};
    Tensor *result = Tensor_new(out_dtype, 2, out_shape);
    if (!result) return NULL;

    nk_size_t c_stride = n * bytes_per_dtype(out_dtype);

    // Macro to handle conversion and matmul for each dtype
#define DO_TENSOR_MATMUL(NK_TYPE, C_TYPE, OUT_TYPE, CONVERT_FN, MATMUL_FN)                             \
    do {                                                                                               \
        C_TYPE const *a_ptr = NULL;                                                                    \
        C_TYPE *temp_a = NULL;                                                                         \
        nk_size_t a_stride = row_stride;                                                               \
        if (src_dtype == NK_TYPE && col_stride == sizeof(C_TYPE)) { a_ptr = (C_TYPE const *)a->data; } \
        else {                                                                                         \
            temp_a = (C_TYPE *)PyMem_Malloc(m * k * sizeof(C_TYPE));                                   \
            if (!temp_a) {                                                                             \
                Py_DECREF(result);                                                                     \
                PyErr_NoMemory();                                                                      \
                return NULL;                                                                           \
            }                                                                                          \
            for (nk_size_t i = 0; i < m; i++) {                                                        \
                char const *row = a->data + i * row_stride;                                            \
                if (!CONVERT_FN(src_dtype, row, k, col_stride, temp_a + i * k)) {                      \
                    PyMem_Free(temp_a);                                                                \
                    Py_DECREF(result);                                                                 \
                    return NULL;                                                                       \
                }                                                                                      \
            }                                                                                          \
            a_ptr = temp_a;                                                                            \
            a_stride = k * sizeof(C_TYPE);                                                             \
        }                                                                                              \
        MATMUL_FN(a_ptr, packed->start, (OUT_TYPE *)result->data, m, n, k, a_stride, c_stride);        \
        if (temp_a) PyMem_Free(temp_a);                                                                \
    } while (0)

    // Dispatch based on packed dtype
    switch (packed->dtype) {
    case nk_bf16_k: DO_TENSOR_MATMUL(nk_bf16_k, nk_bf16_t, nk_f32_t, convert_row_to_bf16, nk_dots_packed_bf16); break;
    case nk_i8_k: DO_TENSOR_MATMUL(nk_i8_k, nk_i8_t, nk_i32_t, convert_row_to_i8, nk_dots_packed_i8); break;
    case nk_f32_k: DO_TENSOR_MATMUL(nk_f32_k, nk_f32_t, nk_f32_t, convert_row_to_f32, nk_dots_packed_f32); break;
    case nk_f64_k: DO_TENSOR_MATMUL(nk_f64_k, nk_f64_t, nk_f64_t, convert_row_to_f64, nk_dots_packed_f64); break;
    case nk_f16_k: DO_TENSOR_MATMUL(nk_f16_k, nk_f16_t, nk_f32_t, convert_row_to_f16, nk_dots_packed_f16); break;
    case nk_u8_k: DO_TENSOR_MATMUL(nk_u8_k, nk_u8_t, nk_u32_t, convert_row_to_u8, nk_dots_packed_u8); break;
    default:
        Py_DECREF(result);
        PyErr_SetString(PyExc_ValueError, "Unsupported packed matrix dtype");
        return NULL;
    }

#undef DO_TENSOR_MATMUL

    return (PyObject *)result;
}

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
    if (ndim > NK_TENSOR_MAX_RANK) {
        PyErr_Format(PyExc_ValueError, "Shape has %zd dimensions, max is %d", ndim, NK_TENSOR_MAX_RANK);
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

char const
    doc_empty[] = "Create an uninitialized Tensor with the given shape.\n\n" "Parameters:\n" "    shape: Shape of the " "array." "\n" "  " "  " "dt" "yp" "e:" " " "Data type " "(default " "'float32')." "\n\n" "Returns" ":\n" " " " " " " " " "T" "e" "n" "s" "o" "r" ":" " " "U" "n" "i" "n" "i" "t" "i" "a" "l" "i" "z" "e" "d" " " "a" "r" "r" "a" "y" ".";

PyObject *api_empty(PyObject *self, PyObject *const *args, Py_ssize_t const nargs, PyObject *kwnames) {
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

    Py_ssize_t shape[NK_TENSOR_MAX_RANK];
    size_t rank;
    if (!parse_shape(shape_obj, shape, &rank)) return NULL;

    nk_dtype_t dtype = nk_f32_k;
    if (dtype_obj) {
        char const *s = PyUnicode_AsUTF8(dtype_obj);
        if (!s) return NULL;
        dtype = python_string_to_dtype(s);
        if (dtype == nk_dtype_unknown_k) {
            PyErr_Format(PyExc_ValueError, "Unknown dtype: %s", s);
            return NULL;
        }
    }

    return (PyObject *)Tensor_new(dtype, rank, shape);
}

char const doc_zeros[] =
    "Create a Tensor filled with zeros.\n\n" "Parameters:\n" "    shape: Shape of the array.\n" "    dtype: " "Data " "type " "(default " "'float32')." "\n\n" "Returns" ":\n" " " " " " " " " "T" "e" "n" "s" "o" "r" ":" " " "A" "r" "r" "a" "y" " " "o" "f" " " "z" "e" "r" "o" "s" ".";

PyObject *api_zeros(PyObject *self, PyObject *const *args, Py_ssize_t const nargs, PyObject *kwnames) {
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

    Py_ssize_t shape[NK_TENSOR_MAX_RANK];
    size_t rank;
    if (!parse_shape(shape_obj, shape, &rank)) return NULL;

    nk_dtype_t dtype = nk_f32_k;
    if (dtype_obj) {
        char const *s = PyUnicode_AsUTF8(dtype_obj);
        if (!s) return NULL;
        dtype = python_string_to_dtype(s);
        if (dtype == nk_dtype_unknown_k) {
            PyErr_Format(PyExc_ValueError, "Unknown dtype: %s", s);
            return NULL;
        }
    }

    Tensor *result = Tensor_new(dtype, rank, shape);
    if (!result) return NULL;

    size_t nbytes = bytes_per_dtype(dtype);
    for (size_t i = 0; i < rank; i++) nbytes *= (size_t)shape[i];
    memset(result->data, 0, nbytes);

    return (PyObject *)result;
}

char const
    doc_ones[] =
        "Create a Tensor filled with ones.\n\n" "Parameters:\n" "    shape: Shape of the array.\n" "    dtype: " "Data " "t" "y" "p" "e" " " "(" "d" "e" "f" "a" "u" "l" "t" " " "'float32')." "\n\n" "Returns" ":\n" " " " " " " " " "T" "e" "n" "s" "o" "r" ":" " " "A" "r" "r" "a" "y" " " "o" "f" " " "o" "n" "e" "s" ".";

PyObject *api_ones(PyObject *self, PyObject *const *args, Py_ssize_t const nargs, PyObject *kwnames) {
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

    Py_ssize_t shape[NK_TENSOR_MAX_RANK];
    size_t rank;
    if (!parse_shape(shape_obj, shape, &rank)) return NULL;

    nk_dtype_t dtype = nk_f32_k;
    if (dtype_obj) {
        char const *s = PyUnicode_AsUTF8(dtype_obj);
        if (!s) return NULL;
        dtype = python_string_to_dtype(s);
        if (dtype == nk_dtype_unknown_k) {
            PyErr_Format(PyExc_ValueError, "Unknown dtype: %s", s);
            return NULL;
        }
    }

    Tensor *result = Tensor_new(dtype, rank, shape);
    if (!result) return NULL;

    size_t total = 1;
    for (size_t i = 0; i < rank; i++) total *= (size_t)shape[i];

    switch (dtype) {
    case nk_f64_k:
        for (size_t i = 0; i < total; i++) ((nk_f64_t *)result->data)[i] = 1.0;
        break;
    case nk_f32_k:
        for (size_t i = 0; i < total; i++) ((nk_f32_t *)result->data)[i] = 1.0f;
        break;
    case nk_i8_k:
        for (size_t i = 0; i < total; i++) ((nk_i8_t *)result->data)[i] = 1;
        break;
    case nk_i32_k:
        for (size_t i = 0; i < total; i++) ((nk_i32_t *)result->data)[i] = 1;
        break;
    case nk_i64_k:
        for (size_t i = 0; i < total; i++) ((nk_i64_t *)result->data)[i] = 1;
        break;
    default:
        Py_DECREF(result);
        PyErr_SetString(PyExc_NotImplementedError, "ones() not implemented for this dtype");
        return NULL;
    }

    return (PyObject *)result;
}

char const doc_full[] =
    "Create a Tensor filled with a given value.\n\n" "Parameters:\n" "    shape: Shape of the " "array.\n" "    " "fill" "_val" "ue:" " " "Value " "to " "fill " "the array " "with.\n" "    " "dtyp" "e: " "Data" " typ" "e " "(def" "ault" " '" "floa" "t32'" ")." "\n\n" "Returns:\n" "    Tensor: Array filled with fill_value.";

PyObject *api_full(PyObject *self, PyObject *const *args, Py_ssize_t const nargs, PyObject *kwnames) {
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

    double fill_value;
    if (!get_scalar_value(fill_obj, &fill_value)) {
        PyErr_SetString(PyExc_TypeError, "fill_value must be a number");
        return NULL;
    }

    Py_ssize_t shape[NK_TENSOR_MAX_RANK];
    size_t rank;
    if (!parse_shape(shape_obj, shape, &rank)) return NULL;

    nk_dtype_t dtype = nk_f32_k;
    if (dtype_obj) {
        char const *s = PyUnicode_AsUTF8(dtype_obj);
        if (!s) return NULL;
        dtype = python_string_to_dtype(s);
        if (dtype == nk_dtype_unknown_k) {
            PyErr_Format(PyExc_ValueError, "Unknown dtype: %s", s);
            return NULL;
        }
    }

    Tensor *result = Tensor_new(dtype, rank, shape);
    if (!result) return NULL;

    size_t total = 1;
    for (size_t i = 0; i < rank; i++) total *= (size_t)shape[i];

    switch (dtype) {
    case nk_f64_k:
        for (size_t i = 0; i < total; i++) ((nk_f64_t *)result->data)[i] = fill_value;
        break;
    case nk_f32_k:
        for (size_t i = 0; i < total; i++) ((nk_f32_t *)result->data)[i] = (nk_f32_t)fill_value;
        break;
    case nk_i8_k:
        for (size_t i = 0; i < total; i++) ((nk_i8_t *)result->data)[i] = (nk_i8_t)fill_value;
        break;
    case nk_i32_k:
        for (size_t i = 0; i < total; i++) ((nk_i32_t *)result->data)[i] = (nk_i32_t)fill_value;
        break;
    case nk_i64_k:
        for (size_t i = 0; i < total; i++) ((nk_i64_t *)result->data)[i] = (nk_i64_t)fill_value;
        break;
    default:
        Py_DECREF(result);
        PyErr_SetString(PyExc_NotImplementedError, "full() not implemented for this dtype");
        return NULL;
    }

    return (PyObject *)result;
}

char const doc_reduce_moments[] =
    "Compute sum and sum-of-squares (moments) of all elements in an array.\n\n" "Parameters:\n" "    a: Input " "array."
                                                                                                                "\n\n" "Returns:" "\n" "    " "tupl" "e: " "(sum" ", " "sum_" "of_" "squa" "res)" " for" " all" " ele" "ment" "s.";

PyObject *api_moments(PyObject *self, PyObject *const *args, Py_ssize_t const nargs, PyObject *kwnames) {
    (void)self;
    if (nargs != 1 || (kwnames && PyTuple_Size(kwnames) > 0)) {
        PyErr_SetString(PyExc_TypeError, "moments(a) takes exactly 1 positional argument");
        return NULL;
    }
    PyObject *a_obj = args[0];
    if (PyObject_TypeCheck(a_obj, &TensorType)) { return Tensor_moments(a_obj, NULL); }
    PyErr_SetString(PyExc_TypeError, "moments() argument must be a Tensor");
    return NULL;
}

char const
    doc_reduce_minmax[] =
        "Find minimum and maximum elements with their indices in an array.\n\n" "Parameters:\n" "    a: Input " "array."
                                                                                                                "\n\n" "Returns:" "\n" "    " "tupl" "e: " "(min" "_val" ", " "min_" "inde" "x, " "max_" "val," " max" "_ind" "ex)" ".";

PyObject *api_minmax(PyObject *self, PyObject *const *args, Py_ssize_t const nargs, PyObject *kwnames) {
    (void)self;
    if (nargs != 1 || (kwnames && PyTuple_Size(kwnames) > 0)) {
        PyErr_SetString(PyExc_TypeError, "minmax(a) takes exactly 1 positional argument");
        return NULL;
    }
    PyObject *a_obj = args[0];
    if (PyObject_TypeCheck(a_obj, &TensorType)) { return Tensor_minmax(a_obj, NULL); }
    PyErr_SetString(PyExc_TypeError, "minmax() argument must be a Tensor");
    return NULL;
}

char const
    doc_pack_matmul_argument[] =
        "pack_matmul_argument(b, dtype='bf16') -> TransposedMatrixMultiplier\n\n" "Pack a matrix for repeated matrix " "multi" "plica" "tion." "\n\n" "The packed format " "is opaque and " "backend-specific, " "optimized for the " "available\n" "hardwa" "re " "(AMX " "on " "Intel," " NEON/" "SVE " "on " "ARM, " "etc.)." "\n" "U" "s" "e" " " "w" "i" "t" "h" " " "m" "a" "t" "m" "u" "l" "(" ")" " " "o" "r" " " "t" "h" "e" " " "@" " " "o" "p" "e" "r" "a" "t" "o" "r" " " "t" "o" " " "c" "o" "m" "p" "u" "t" "e" " " "C" " " "=" " " "A" " " "@" " " "B" "." "\n\n" "Parameters:\n" "    b : array_like\n" "        The (n, k) matrix to pack. This is typically the 'database' or 'weights' matrix\n" "        that will be multiplied against multiple 'query' matrices.\n" "    dtype : str, optional\n" "        Data type for packing. Supported types:\n" "        - 'bf16'/'bfloat16' (default): BF16 with F32 accumulation\n" "        - 'f16'/'float16': F16 with F32 accumulation\n" "        - 'f32'/'float32': Native F32\n" "        - 'f64'/'float64': Native F64\n" "        - 'i8'/'int8': I8 with I32 accumulation\n" "        - 'u8'/'uint8': U8 with U32 accumulation\n\n" "Returns:\n" "    TransposedMatrixMultiplier : Opaque packed matrix for use with matmul() or @.\n\n" "Example:\n" "    >>> database = np.random.randn(1000, 768).astype(np.float32)\n" "    >>> packed = nk.pack_matmul_argument(database, dtype='bf16')\n" "    >>> queries = nk.zeros((10, 768), dtype='float32')\n" "    >>> result = queries @ packed  # (10, 1000) dot products\n";

char const doc_pack_matrix[] =
    "pack_matrix(b, dtype='bf16') -> TransposedMatrixMultiplier\n\n" "Deprecated alias for pack_matmul_argument().\n";

char const
    doc_matmul[] = "matmul(a, b, *, out=None) -> Tensor\n\n" "Compute matrix multiplication C = A @ B with a "
                                                             "pre-packed B " "m" "a" "t" "r" "i" "x" "." "\n\n" "Pa" "r"
                                                                                                                     "a" "me" "te" "rs" ":" "\n" "    a : array_like\n" "        " "The (m, k) " "query/" "input " "matrix.\n" "    b : TransposedMatrixMultiplier\n" "        Pre-packed (n, k) matrix from pack_matmul_argument().\n" "    out : Tensor, optional\n" "        Pre-allocated output tensor. Must have correct shape (m, n),\n" "        correct dtype for the operation, and be C-contiguous.\n" "        If provided, no memory allocation is performed.\n\n" "Returns:\n" "    Tensor : (m, n) result matrix. If out is provided, returns out.\n\n" "Note:\n" "    The kernel computes C[i,j] = dot(a[i], b[j]) for all i,j.\n" "    This is equivalent to A @ B.T where B is the original unpacked matrix.\n\n" "    Output dtype depends on packed dtype:\n" "    - bf16, f16 -> float32\n" "    - f32 -> float32\n" "    - f64 -> float64\n" "    - i8 -> int32\n" "    - u8 -> uint32\n\n" "Example:\n" "    >>> database = np.random.randn(1000, 768).astype(np.float32)\n" "    >>> packed = nk.pack_matmul_argument(database, dtype='bf16')\n" "    >>> queries = np.random.randn(10, 768).astype(np.float32)\n" "    >>> result = nk.matmul(queries, packed)  # (10, 1000)\n" "    >>>\n" "    >>> # Reuse output buffer for zero-allocation inference:\n" "    >>> out = nk.empty((10, 1000), dtype='float32')\n" "    >>> nk.matmul(queries, packed, out=out)\n";

PyObject *api_pack_matmul_argument(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    (void)self;

    // Parse arguments
    PyObject *b_obj = NULL;
    char const *dtype_str = "bf16";

    Py_ssize_t nkw = kwnames ? PyTuple_Size(kwnames) : 0;
    Py_ssize_t total = nargs + nkw;

    if (nargs < 1 || total > 2) {
        PyErr_SetString(PyExc_TypeError, "pack_matmul_argument() requires 1-2 arguments: b, dtype='bf16'");
        return NULL;
    }

    b_obj = args[0];

    // Parse the optional dtype argument
    for (Py_ssize_t i = 0; i < nkw; i++) {
        PyObject *name = PyTuple_GET_ITEM(kwnames, i);
        char const *name_str = PyUnicode_AsUTF8(name);
        if (same_string(name_str, "dtype")) {
            PyObject *val = args[nargs + i];
            if (!PyUnicode_Check(val)) {
                PyErr_SetString(PyExc_TypeError, "dtype must be a string");
                return NULL;
            }
            dtype_str = PyUnicode_AsUTF8(val);
        }
    }
    // Accept an optional second positional argument
    if (nargs >= 2) {
        if (!PyUnicode_Check(args[1])) {
            PyErr_SetString(PyExc_TypeError, "dtype must be a string");
            return NULL;
        }
        dtype_str = PyUnicode_AsUTF8(args[1]);
    }

    // Resolve the target packing dtype
    nk_dtype_t target_dtype;
    if (same_string(dtype_str, "bf16") || same_string(dtype_str, "bfloat16")) { target_dtype = nk_bf16_k; }
    else if (same_string(dtype_str, "i8") || same_string(dtype_str, "int8")) { target_dtype = nk_i8_k; }
    else if (same_string(dtype_str, "f32") || same_string(dtype_str, "float32")) { target_dtype = nk_f32_k; }
    else if (same_string(dtype_str, "f64") || same_string(dtype_str, "float64")) { target_dtype = nk_f64_k; }
    else if (same_string(dtype_str, "f16") || same_string(dtype_str, "float16")) { target_dtype = nk_f16_k; }
    else if (same_string(dtype_str, "u8") || same_string(dtype_str, "uint8")) { target_dtype = nk_u8_k; }
    else {
        PyErr_Format(PyExc_ValueError, "Unsupported dtype '%s'. Use 'bf16', 'i8', 'f32', 'f64', 'f16', or 'u8'.",
                     dtype_str);
        return NULL;
    }

    // Get the input buffer
    Py_buffer b_buffer;
    if (PyObject_GetBuffer(b_obj, &b_buffer, PyBUF_STRIDES | PyBUF_FORMAT) != 0) {
        PyErr_SetString(PyExc_TypeError, "b must support buffer protocol");
        return NULL;
    }

    // Validate the input as a 2D matrix
    if (b_buffer.ndim != 2) {
        PyBuffer_Release(&b_buffer);
        PyErr_SetString(PyExc_ValueError, "b must be a 2D matrix");
        return NULL;
    }

    nk_dtype_t src_dtype;
    if (!buffer_dtype(&b_buffer, &src_dtype)) {
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

    // Calculate the packed size based on target dtype
    nk_size_t packed_size;
    switch (target_dtype) {
    case nk_bf16_k: packed_size = nk_dots_packed_size_bf16(n, k); break;
    case nk_i8_k: packed_size = nk_dots_packed_size_i8(n, k); break;
    case nk_f32_k: packed_size = nk_dots_packed_size_f32(n, k); break;
    case nk_f64_k: packed_size = nk_dots_packed_size_f64(n, k); break;
    case nk_f16_k: packed_size = nk_dots_packed_size_f16(n, k); break;
    case nk_u8_k: packed_size = nk_dots_packed_size_u8(n, k); break;
    default:
        PyBuffer_Release(&b_buffer);
        PyErr_SetString(PyExc_ValueError, "Internal error: unsupported target dtype");
        return NULL;
    }

    TransposedMatrixMultiplier *packed = PyObject_NewVar(TransposedMatrixMultiplier, &TransposedMatrixMultiplierType,
                                                         packed_size);
    if (!packed) {
        PyBuffer_Release(&b_buffer);
        PyErr_NoMemory();
        return NULL;
    }

    packed->dtype = target_dtype;
    packed->n = n;
    packed->k = k;

    // Macro to handle conversion and packing for each dtype
#define PACK_MATRIX(NK_TYPE, C_TYPE, CONVERT_FN, PACK_FN)                                                   \
    do {                                                                                                    \
        C_TYPE const *b_ptr = NULL;                                                                         \
        C_TYPE *temp = NULL;                                                                                \
        nk_size_t b_stride = row_stride;                                                                    \
        if (src_dtype == NK_TYPE && col_stride == sizeof(C_TYPE)) { b_ptr = (C_TYPE const *)b_buffer.buf; } \
        else {                                                                                              \
            temp = (C_TYPE *)PyMem_Malloc(n * k * sizeof(C_TYPE));                                          \
            if (!temp) {                                                                                    \
                Py_DECREF(packed);                                                                          \
                PyBuffer_Release(&b_buffer);                                                                \
                PyErr_NoMemory();                                                                           \
                return NULL;                                                                                \
            }                                                                                               \
            for (nk_size_t i = 0; i < n; i++) {                                                             \
                char const *row = (char const *)b_buffer.buf + i * row_stride;                              \
                if (!CONVERT_FN(src_dtype, row, k, col_stride, temp + i * k)) {                             \
                    PyMem_Free(temp);                                                                       \
                    Py_DECREF(packed);                                                                      \
                    PyBuffer_Release(&b_buffer);                                                            \
                    return NULL;                                                                            \
                }                                                                                           \
            }                                                                                               \
            b_ptr = temp;                                                                                   \
            b_stride = k * sizeof(C_TYPE);                                                                  \
        }                                                                                                   \
        PACK_FN(b_ptr, n, k, b_stride, packed->start);                                                      \
        if (temp) PyMem_Free(temp);                                                                         \
    } while (0)

    // Pack based on target dtype
    switch (target_dtype) {
    case nk_bf16_k: PACK_MATRIX(nk_bf16_k, nk_bf16_t, convert_row_to_bf16, nk_dots_pack_bf16); break;
    case nk_i8_k: PACK_MATRIX(nk_i8_k, nk_i8_t, convert_row_to_i8, nk_dots_pack_i8); break;
    case nk_f32_k: PACK_MATRIX(nk_f32_k, nk_f32_t, convert_row_to_f32, nk_dots_pack_f32); break;
    case nk_f64_k: PACK_MATRIX(nk_f64_k, nk_f64_t, convert_row_to_f64, nk_dots_pack_f64); break;
    case nk_f16_k: PACK_MATRIX(nk_f16_k, nk_f16_t, convert_row_to_f16, nk_dots_pack_f16); break;
    case nk_u8_k: PACK_MATRIX(nk_u8_k, nk_u8_t, convert_row_to_u8, nk_dots_pack_u8); break;
    default: break; // Already handled above
    }

#undef PACK_MATRIX

    PyBuffer_Release(&b_buffer);
    return (PyObject *)packed;
}

PyObject *api_pack_matrix(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    if (PyErr_WarnEx(PyExc_DeprecationWarning, "pack_matrix() is deprecated; use pack_matmul_argument() instead", 1) <
        0)
        return NULL;
    return api_pack_matmul_argument(self, args, nargs, kwnames);
}

PyObject *api_matmul(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    (void)self;

    // Parse arguments: matmul(a, b, *, out=None)
    PyObject *a_obj = NULL;
    PyObject *b_obj = NULL;
    PyObject *out_obj = NULL;

    Py_ssize_t nkw = kwnames ? PyTuple_Size(kwnames) : 0;
    if (nargs < 2 || nargs > 2) {
        PyErr_SetString(PyExc_TypeError, "matmul() requires exactly 2 positional arguments: a, b");
        return NULL;
    }

    a_obj = args[0];
    b_obj = args[1];

    // Parse keyword arguments
    for (Py_ssize_t i = 0; i < nkw; i++) {
        PyObject *name = PyTuple_GET_ITEM(kwnames, i);
        char const *name_str = PyUnicode_AsUTF8(name);
        if (same_string(name_str, "out")) { out_obj = args[nargs + i]; }
        else {
            PyErr_Format(PyExc_TypeError, "matmul() got unexpected keyword argument '%s'", name_str);
            return NULL;
        }
    }

    // Verify that b is a TransposedMatrixMultiplier
    if (!PyObject_TypeCheck(b_obj, &TransposedMatrixMultiplierType)) {
        PyErr_SetString(PyExc_TypeError, "b must be a TransposedMatrixMultiplier (use pack_matmul_argument() first)");
        return NULL;
    }
    TransposedMatrixMultiplier *packed = (TransposedMatrixMultiplier *)b_obj;

    // Get the input buffer for a
    Py_buffer a_buffer;
    if (PyObject_GetBuffer(a_obj, &a_buffer, PyBUF_STRIDES | PyBUF_FORMAT) != 0) {
        PyErr_SetString(PyExc_TypeError, "a must support buffer protocol");
        return NULL;
    }

    // Validate dimensions
    if (a_buffer.ndim != 2) {
        PyBuffer_Release(&a_buffer);
        PyErr_SetString(PyExc_ValueError, "a must be a 2D matrix");
        return NULL;
    }

    nk_dtype_t src_dtype;
    if (!buffer_dtype(&a_buffer, &src_dtype)) {
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
    nk_dtype_t out_dtype = matmul_output_dtype(packed->dtype);

    // Handle output tensor
    Tensor *result = NULL;
    int owns_result = 0;
    char *out_data = NULL;
    nk_size_t c_stride;

    if (out_obj && out_obj != Py_None) {
        // Validate provided output tensor
        if (!PyObject_TypeCheck(out_obj, &TensorType)) {
            PyBuffer_Release(&a_buffer);
            PyErr_SetString(PyExc_TypeError, "out must be a Tensor");
            return NULL;
        }
        result = (Tensor *)out_obj;

        // Check shape
        if (result->rank != 2 || result->shape[0] != (Py_ssize_t)m || result->shape[1] != (Py_ssize_t)n) {
            PyBuffer_Release(&a_buffer);
            PyErr_Format(PyExc_ValueError, "out has wrong shape: expected (%zu, %zu), got (%zd, %zd)", m, n,
                         result->shape[0], result->shape[1]);
            return NULL;
        }

        // Check dtype
        if (result->dtype != out_dtype) {
            PyBuffer_Release(&a_buffer);
            PyErr_Format(PyExc_ValueError, "out has wrong dtype: expected %s, got %s", dtype_to_string(out_dtype),
                         dtype_to_string(result->dtype));
            return NULL;
        }

        // Check contiguity (must be C-contiguous for matmul output)
        size_t out_item_size = bytes_per_dtype(out_dtype);
        if (result->strides[1] != (Py_ssize_t)out_item_size || result->strides[0] != (Py_ssize_t)(n * out_item_size)) {
            PyBuffer_Release(&a_buffer);
            PyErr_SetString(PyExc_ValueError, "out must be C-contiguous");
            return NULL;
        }

        out_data = result->data;
        c_stride = (nk_size_t)result->strides[0];
        owns_result = 0;
    }
    else {
        // Allocate new output tensor
        Py_ssize_t out_shape[2] = {(Py_ssize_t)m, (Py_ssize_t)n};
        result = Tensor_new(out_dtype, 2, out_shape);
        if (!result) {
            PyBuffer_Release(&a_buffer);
            return NULL;
        }
        out_data = result->data;
        c_stride = n * bytes_per_dtype(out_dtype);
        owns_result = 1;
    }

    // Macro to handle conversion and matmul for each dtype
#define DO_MATMUL(NK_TYPE, C_TYPE, OUT_TYPE, CONVERT_FN, MATMUL_FN)                                         \
    do {                                                                                                    \
        C_TYPE const *a_ptr = NULL;                                                                         \
        C_TYPE *temp_a = NULL;                                                                              \
        nk_size_t a_stride = row_stride;                                                                    \
        if (src_dtype == NK_TYPE && col_stride == sizeof(C_TYPE)) { a_ptr = (C_TYPE const *)a_buffer.buf; } \
        else {                                                                                              \
            temp_a = (C_TYPE *)PyMem_Malloc(m * k * sizeof(C_TYPE));                                        \
            if (!temp_a) {                                                                                  \
                if (owns_result) Py_DECREF(result);                                                         \
                PyBuffer_Release(&a_buffer);                                                                \
                PyErr_NoMemory();                                                                           \
                return NULL;                                                                                \
            }                                                                                               \
            for (nk_size_t i = 0; i < m; i++) {                                                             \
                char const *row = (char const *)a_buffer.buf + i * row_stride;                              \
                if (!CONVERT_FN(src_dtype, row, k, col_stride, temp_a + i * k)) {                           \
                    PyMem_Free(temp_a);                                                                     \
                    if (owns_result) Py_DECREF(result);                                                     \
                    PyBuffer_Release(&a_buffer);                                                            \
                    return NULL;                                                                            \
                }                                                                                           \
            }                                                                                               \
            a_ptr = temp_a;                                                                                 \
            a_stride = k * sizeof(C_TYPE);                                                                  \
        }                                                                                                   \
        MATMUL_FN(a_ptr, packed->start, (OUT_TYPE *)out_data, m, n, k, a_stride, c_stride);                 \
        if (temp_a) PyMem_Free(temp_a);                                                                     \
    } while (0)

    // Dispatch based on packed dtype
    switch (packed->dtype) {
    case nk_bf16_k: DO_MATMUL(nk_bf16_k, nk_bf16_t, nk_f32_t, convert_row_to_bf16, nk_dots_packed_bf16); break;
    case nk_i8_k: DO_MATMUL(nk_i8_k, nk_i8_t, nk_i32_t, convert_row_to_i8, nk_dots_packed_i8); break;
    case nk_f32_k: DO_MATMUL(nk_f32_k, nk_f32_t, nk_f32_t, convert_row_to_f32, nk_dots_packed_f32); break;
    case nk_f64_k: DO_MATMUL(nk_f64_k, nk_f64_t, nk_f64_t, convert_row_to_f64, nk_dots_packed_f64); break;
    case nk_f16_k: DO_MATMUL(nk_f16_k, nk_f16_t, nk_f32_t, convert_row_to_f16, nk_dots_packed_f16); break;
    case nk_u8_k: DO_MATMUL(nk_u8_k, nk_u8_t, nk_u32_t, convert_row_to_u8, nk_dots_packed_u8); break;
    default:
        if (owns_result) Py_DECREF(result);
        PyBuffer_Release(&a_buffer);
        PyErr_SetString(PyExc_ValueError, "Unsupported packed matrix dtype");
        return NULL;
    }

#undef DO_MATMUL

    PyBuffer_Release(&a_buffer);

    if (owns_result) { return (PyObject *)result; }
    else {
        Py_INCREF(result);
        return (PyObject *)result;
    }
}

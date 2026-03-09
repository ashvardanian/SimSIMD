/**
 *  @brief MaxSim late-interaction operations for NumKong Python bindings.
 *  @file python/maxsim.c
 *  @author Ash Vardanian
 *  @date March 9, 2026
 *
 *  This module owns:
 *  - `MaxSimPackedMatrix`: opaque pre-packed matrix for MaxSim scoring.
 *  - Packing API: `maxsim_pack()`.
 *  - Packed scoring API: `maxsim_packed()`.
 *  - Convenience API: `maxsim()` (pack + compute).
 */
#include "maxsim.h"
#include "tensor.h"

#include <numkong/maxsim.h>

static void MaxSimPackedMatrix_dealloc(PyObject *self) { Py_TYPE(self)->tp_free(self); }

static size_t maxsim_packed_matrix_nbytes(MaxSimPackedMatrix *mm) {
    nk_dots_packed_size_punned_t size_fn = NULL;
    nk_capability_t cap = nk_cap_serial_k;
    nk_find_kernel_punned(nk_kernel_maxsim_packed_size_k, mm->dtype, static_capabilities, nk_cap_any_k,
                          (nk_kernel_punned_t *)&size_fn, &cap);
    if (!size_fn) return 0;
    return size_fn(mm->n, mm->k);
}

static PyObject *MaxSimPackedMatrix_repr(PyObject *self) {
    MaxSimPackedMatrix *mm = (MaxSimPackedMatrix *)self;
    size_t packed_size = maxsim_packed_matrix_nbytes(mm);
    return PyUnicode_FromFormat("<MaxSimPackedMatrix n=%zu k=%zu dtype='%s' nbytes=%zu>", (size_t)mm->n, (size_t)mm->k,
                                dtype_to_string(mm->dtype), packed_size);
}

static PyObject *MaxSimPackedMatrix_get_n(PyObject *self, void *closure) {
    (void)closure;
    return PyLong_FromSize_t(((MaxSimPackedMatrix *)self)->n);
}

static PyObject *MaxSimPackedMatrix_get_k(PyObject *self, void *closure) {
    (void)closure;
    return PyLong_FromSize_t(((MaxSimPackedMatrix *)self)->k);
}

static PyObject *MaxSimPackedMatrix_get_dtype(PyObject *self, void *closure) {
    (void)closure;
    return PyUnicode_FromString(dtype_to_string(((MaxSimPackedMatrix *)self)->dtype));
}

static PyObject *MaxSimPackedMatrix_get_nbytes(PyObject *self, void *closure) {
    (void)closure;
    return PyLong_FromSize_t(maxsim_packed_matrix_nbytes((MaxSimPackedMatrix *)self));
}

static PyGetSetDef MaxSimPackedMatrix_getset[] = {
    {"n", MaxSimPackedMatrix_get_n, NULL, "Number of vectors", NULL},
    {"k", MaxSimPackedMatrix_get_k, NULL, "Number of dimensions per vector (depth)", NULL},
    {"dtype", MaxSimPackedMatrix_get_dtype, NULL, "Data type of the packed vectors", NULL},
    {"nbytes", MaxSimPackedMatrix_get_nbytes, NULL, "Size of the packed buffer in bytes", NULL},
    {NULL, NULL, NULL, NULL, NULL},
};

static PyObject *MaxSimPackedMatrix_packed_size(PyObject *cls, PyObject *const *args, Py_ssize_t nargs,
                                                PyObject *kwnames) {
    (void)cls;

    PyObject *n_obj = NULL, *k_obj = NULL, *dtype_obj = NULL;
    Py_ssize_t nkw = kwnames ? PyTuple_Size(kwnames) : 0;
    Py_ssize_t total = nargs + nkw;

    if (nargs < 2 || total > 3 || nargs > 3) {
        PyErr_SetString(PyExc_TypeError, "packed_size(n, k, /, dtype='bf16')");
        return NULL;
    }

    n_obj = args[0];
    k_obj = args[1];
    if (nargs >= 3) dtype_obj = args[2];

    for (Py_ssize_t i = 0; i < nkw; i++) {
        PyObject *name = PyTuple_GET_ITEM(kwnames, i);
        PyObject *value = args[nargs + i];
        if (PyUnicode_CompareWithASCIIString(name, "dtype") == 0) {
            if (dtype_obj) {
                PyErr_SetString(PyExc_TypeError, "packed_size() got multiple values for argument 'dtype'");
                return NULL;
            }
            dtype_obj = value;
        }
        else {
            PyErr_Format(PyExc_TypeError, "packed_size() got unexpected keyword argument '%S'", name);
            return NULL;
        }
    }

    if (!dtype_obj) {
        PyErr_SetString(PyExc_TypeError, "packed_size() requires 'dtype' argument");
        return NULL;
    }

    nk_size_t n = (nk_size_t)PyLong_AsSize_t(n_obj);
    if (n == (nk_size_t)-1 && PyErr_Occurred()) return NULL;
    nk_size_t k = (nk_size_t)PyLong_AsSize_t(k_obj);
    if (k == (nk_size_t)-1 && PyErr_Occurred()) return NULL;

    char const *dtype_str = PyUnicode_AsUTF8(dtype_obj);
    if (!dtype_str) return NULL;
    nk_dtype_t dtype = python_string_to_dtype(dtype_str);
    if (dtype == nk_dtype_unknown_k) {
        PyErr_Format(PyExc_ValueError, "Unknown dtype: '%s'", dtype_str);
        return NULL;
    }

    nk_dots_packed_size_punned_t size_fn = NULL;
    nk_capability_t cap = nk_cap_serial_k;
    nk_find_kernel_punned(nk_kernel_maxsim_packed_size_k, dtype, static_capabilities, nk_cap_any_k,
                          (nk_kernel_punned_t *)&size_fn, &cap);
    if (!size_fn) {
        PyErr_Format(PyExc_LookupError, "No maxsim packed_size kernel for dtype '%s'", dtype_str);
        return NULL;
    }

    return PyLong_FromSize_t(size_fn(n, k));
}

static PyMethodDef MaxSimPackedMatrix_methods[] = {
    {"packed_size", (PyCFunction)MaxSimPackedMatrix_packed_size, METH_CLASS | METH_FASTCALL | METH_KEYWORDS,
     "Return packed buffer size in bytes for given dimensions and dtype."},
    {NULL, NULL, 0, NULL},
};

PyTypeObject MaxSimPackedMatrixType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "numkong.MaxSimPackedMatrix",
    .tp_doc = "Opaque pre-packed matrix for MaxSim late-interaction scoring",
    .tp_basicsize = sizeof(MaxSimPackedMatrix),
    .tp_itemsize = sizeof(char),
    .tp_dealloc = MaxSimPackedMatrix_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_getset = MaxSimPackedMatrix_getset,
    .tp_methods = MaxSimPackedMatrix_methods,
    .tp_repr = MaxSimPackedMatrix_repr,
};

char const doc_maxsim_pack[] =                                              //
    "maxsim_pack(b, /, dtype='bf16') -> MaxSimPackedMatrix\n\n"             //
    "Pack a 2D matrix for MaxSim late-interaction scoring.\n\n"             //
    "Parameters:\n"                                                         //
    "    b (array_like): Source matrix with shape (n, k).\n"                //
    "    dtype (str, optional): Packing dtype. Default: 'bf16'.\n"          //
    "        Supported values: 'bf16', 'f16', 'f32'.\n\n"                   //
    "Returns:\n"                                                            //
    "    MaxSimPackedMatrix: Opaque packed matrix for maxsim_packed().\n\n" //
    "Signature:\n"                                                          //
    "    >>> def maxsim_pack(b, /, dtype='bf16') -> MaxSimPackedMatrix: ...";

PyObject *api_maxsim_pack(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    (void)self;

    PyObject *b_obj = NULL;
    char const *dtype_str = "bf16";

    Py_ssize_t nkw = kwnames ? PyTuple_Size(kwnames) : 0;
    Py_ssize_t total = nargs + nkw;

    if (nargs < 1 || total > 2) {
        PyErr_SetString(PyExc_TypeError, "maxsim_pack() requires 1-2 arguments: b, dtype='bf16'");
        return NULL;
    }

    b_obj = args[0];

    for (Py_ssize_t i = 0; i < nkw; i++) {
        PyObject *name = PyTuple_GET_ITEM(kwnames, i);
        if (PyUnicode_CompareWithASCIIString(name, "dtype") == 0) {
            if (nargs >= 2) {
                PyErr_SetString(PyExc_TypeError, "maxsim_pack() got multiple values for argument 'dtype'");
                return NULL;
            }
            PyObject *val = args[nargs + i];
            if (!PyUnicode_Check(val)) {
                PyErr_SetString(PyExc_TypeError, "dtype must be a string");
                return NULL;
            }
            dtype_str = PyUnicode_AsUTF8(val);
        }
        else {
            char const *name_str = PyUnicode_AsUTF8(name);
            PyErr_Format(PyExc_TypeError, "maxsim_pack() got unexpected keyword argument '%s'", name_str);
            return NULL;
        }
    }
    if (nargs >= 2) {
        if (!PyUnicode_Check(args[1])) {
            PyErr_SetString(PyExc_TypeError, "dtype must be a string");
            return NULL;
        }
        dtype_str = PyUnicode_AsUTF8(args[1]);
    }

    nk_dtype_t target_dtype = python_string_to_dtype(dtype_str);
    if (target_dtype == nk_dtype_unknown_k) {
        PyErr_Format(PyExc_ValueError, "Unsupported dtype '%s'", dtype_str);
        return NULL;
    }

    if (target_dtype != nk_bf16_k && target_dtype != nk_f16_k && target_dtype != nk_f32_k) {
        PyErr_Format(PyExc_ValueError, "maxsim_pack() only supports 'bf16', 'f16', 'f32'; got '%s'", dtype_str);
        return NULL;
    }

    Py_buffer b_buffer;
    if (PyObject_GetBuffer(b_obj, &b_buffer, PyBUF_STRIDES | PyBUF_FORMAT) != 0) {
        PyErr_SetString(PyExc_TypeError, "b must support buffer protocol");
        return NULL;
    }

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

    if (b_buffer.strides[0] < 0 || b_buffer.strides[1] < 0) {
        PyBuffer_Release(&b_buffer);
        PyErr_SetString(PyExc_ValueError, "maxsim packing does not support negative strides");
        return NULL;
    }

    nk_size_t n = (nk_size_t)b_buffer.shape[0];
    nk_size_t k = (nk_size_t)b_buffer.shape[1];
    nk_size_t row_stride = (nk_size_t)b_buffer.strides[0];
    nk_size_t col_stride = (nk_size_t)b_buffer.strides[1];

    if (src_dtype != target_dtype) {
        PyBuffer_Release(&b_buffer);
        PyErr_Format(PyExc_TypeError, "Input dtype '%s' does not match target dtype '%s'. Cast the input first.",
                     dtype_to_python_string(src_dtype), dtype_to_python_string(target_dtype));
        return NULL;
    }
    if (col_stride != (nk_size_t)bytes_per_dtype(target_dtype)) {
        PyBuffer_Release(&b_buffer);
        PyErr_SetString(PyExc_ValueError, "Input matrix must be row-contiguous");
        return NULL;
    }

    nk_dots_packed_size_punned_t size_fn = NULL;
    nk_capability_t cap = nk_cap_serial_k;
    nk_find_kernel_punned(nk_kernel_maxsim_packed_size_k, target_dtype, static_capabilities, nk_cap_any_k,
                          (nk_kernel_punned_t *)&size_fn, &cap);
    if (!size_fn) {
        PyBuffer_Release(&b_buffer);
        PyErr_Format(PyExc_LookupError, "No maxsim packed_size kernel for dtype '%s'",
                     dtype_to_python_string(target_dtype));
        return NULL;
    }
    nk_size_t packed_size = size_fn(n, k);

    MaxSimPackedMatrix *packed = PyObject_NewVar(MaxSimPackedMatrix, &MaxSimPackedMatrixType, packed_size);
    if (!packed) {
        PyBuffer_Release(&b_buffer);
        PyErr_NoMemory();
        return NULL;
    }

    packed->dtype = target_dtype;
    packed->n = n;
    packed->k = k;

    nk_dots_pack_punned_t pack_fn = NULL;
    cap = nk_cap_serial_k;
    nk_find_kernel_punned(nk_kernel_maxsim_pack_k, target_dtype, static_capabilities, nk_cap_any_k,
                          (nk_kernel_punned_t *)&pack_fn, &cap);
    if (!pack_fn) {
        Py_DECREF(packed);
        PyBuffer_Release(&b_buffer);
        PyErr_Format(PyExc_LookupError, "No maxsim pack kernel for dtype '%s'", dtype_to_python_string(target_dtype));
        return NULL;
    }

    {
        PyThreadState *save = PyEval_SaveThread();
        pack_fn(b_buffer.buf, n, k, row_stride, packed->start);
        PyEval_RestoreThread(save);
    }

    PyBuffer_Release(&b_buffer);
    return (PyObject *)packed;
}

char const doc_maxsim_packed[] =                                             //
    "maxsim_packed(q, d, /) -> float\n\n"                                    //
    "Compute MaxSim late-interaction score between two packed matrices.\n\n" //
    "Parameters:\n"                                                          //
    "    q (MaxSimPackedMatrix): Packed query vectors.\n"                    //
    "    d (MaxSimPackedMatrix): Packed document vectors.\n\n"               //
    "Returns:\n"                                                             //
    "    float: Sum of per-query minimum angular distances.\n\n"             //
    "Signature:\n"                                                           //
    "    >>> def maxsim_packed(q, d, /) -> float: ...";

PyObject *api_maxsim_packed(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    (void)self;

    if (nargs != 2 || (kwnames && PyTuple_Size(kwnames) > 0)) {
        PyErr_SetString(PyExc_TypeError, "maxsim_packed() requires exactly 2 positional arguments: q, d");
        return NULL;
    }

    if (!PyObject_TypeCheck(args[0], &MaxSimPackedMatrixType)) {
        PyErr_SetString(PyExc_TypeError, "q must be a MaxSimPackedMatrix (use maxsim_pack() first)");
        return NULL;
    }
    if (!PyObject_TypeCheck(args[1], &MaxSimPackedMatrixType)) {
        PyErr_SetString(PyExc_TypeError, "d must be a MaxSimPackedMatrix (use maxsim_pack() first)");
        return NULL;
    }

    MaxSimPackedMatrix *q = (MaxSimPackedMatrix *)args[0];
    MaxSimPackedMatrix *d = (MaxSimPackedMatrix *)args[1];

    if (q->dtype != d->dtype) {
        PyErr_Format(PyExc_TypeError, "dtype mismatch: q is '%s' but d is '%s'", dtype_to_python_string(q->dtype),
                     dtype_to_python_string(d->dtype));
        return NULL;
    }
    if (q->k != d->k) {
        PyErr_Format(PyExc_ValueError, "depth mismatch: q has k=%zu but d has k=%zu", (size_t)q->k, (size_t)d->k);
        return NULL;
    }

    nk_maxsim_packed_punned_t kernel = NULL;
    nk_capability_t cap = nk_cap_serial_k;
    nk_find_kernel_punned(nk_kernel_maxsim_packed_k, q->dtype, static_capabilities, nk_cap_any_k,
                          (nk_kernel_punned_t *)&kernel, &cap);
    if (!kernel) {
        PyErr_Format(PyExc_LookupError, "No maxsim_packed kernel for dtype '%s'", dtype_to_python_string(q->dtype));
        return NULL;
    }

    nk_f32_t result = 0;
    {
        PyThreadState *save = PyEval_SaveThread();
        kernel(q->start, d->start, q->n, d->n, q->k, &result);
        PyEval_RestoreThread(save);
    }

    return PyFloat_FromDouble((double)result);
}

char const doc_maxsim[] =                                                 //
    "maxsim(q, d, /, dtype='bf16') -> float\n\n"                          //
    "Convenience MaxSim: pack both matrices and compute in one call.\n\n" //
    "Parameters:\n"                                                       //
    "    q (array_like): Query matrix with shape (n_q, depth).\n"         //
    "    d (array_like): Document matrix with shape (n_d, depth).\n"      //
    "    dtype (str, optional): Packing dtype. Default: 'bf16'.\n"        //
    "        Supported values: 'bf16', 'f16', 'f32'.\n\n"                 //
    "Returns:\n"                                                          //
    "    float: Sum of per-query minimum angular distances.\n\n"          //
    "Signature:\n"                                                        //
    "    >>> def maxsim(q, d, /, dtype='bf16') -> float: ...";

PyObject *api_maxsim(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    (void)self;

    PyObject *q_obj = NULL, *d_obj = NULL;
    char const *dtype_str = "bf16";

    Py_ssize_t nkw = kwnames ? PyTuple_Size(kwnames) : 0;
    Py_ssize_t total = nargs + nkw;

    if (nargs < 2 || total > 3) {
        PyErr_SetString(PyExc_TypeError, "maxsim() requires 2-3 arguments: q, d, dtype='bf16'");
        return NULL;
    }

    q_obj = args[0];
    d_obj = args[1];

    for (Py_ssize_t i = 0; i < nkw; i++) {
        PyObject *name = PyTuple_GET_ITEM(kwnames, i);
        if (PyUnicode_CompareWithASCIIString(name, "dtype") == 0) {
            if (nargs >= 3) {
                PyErr_SetString(PyExc_TypeError, "maxsim() got multiple values for argument 'dtype'");
                return NULL;
            }
            PyObject *val = args[nargs + i];
            if (!PyUnicode_Check(val)) {
                PyErr_SetString(PyExc_TypeError, "dtype must be a string");
                return NULL;
            }
            dtype_str = PyUnicode_AsUTF8(val);
        }
        else {
            char const *name_str = PyUnicode_AsUTF8(name);
            PyErr_Format(PyExc_TypeError, "maxsim() got unexpected keyword argument '%s'", name_str);
            return NULL;
        }
    }
    if (nargs >= 3) {
        if (!PyUnicode_Check(args[2])) {
            PyErr_SetString(PyExc_TypeError, "dtype must be a string");
            return NULL;
        }
        dtype_str = PyUnicode_AsUTF8(args[2]);
    }

    nk_dtype_t target_dtype = python_string_to_dtype(dtype_str);
    if (target_dtype == nk_dtype_unknown_k) {
        PyErr_Format(PyExc_ValueError, "Unsupported dtype '%s'", dtype_str);
        return NULL;
    }
    if (target_dtype != nk_bf16_k && target_dtype != nk_f16_k && target_dtype != nk_f32_k) {
        PyErr_Format(PyExc_ValueError, "maxsim() only supports 'bf16', 'f16', 'f32'; got '%s'", dtype_str);
        return NULL;
    }

    Py_buffer q_buffer, d_buffer;
    int have_q = 0, have_d = 0;
    PyObject *return_obj = NULL;
    MaxSimPackedMatrix *q_packed = NULL, *d_packed = NULL;

    if (PyObject_GetBuffer(q_obj, &q_buffer, PyBUF_STRIDES | PyBUF_FORMAT) != 0) {
        PyErr_SetString(PyExc_TypeError, "q must support buffer protocol");
        return NULL;
    }
    have_q = 1;

    if (PyObject_GetBuffer(d_obj, &d_buffer, PyBUF_STRIDES | PyBUF_FORMAT) != 0) {
        PyErr_SetString(PyExc_TypeError, "d must support buffer protocol");
        goto cleanup;
    }
    have_d = 1;

    if (q_buffer.ndim != 2 || d_buffer.ndim != 2) {
        PyErr_SetString(PyExc_ValueError, "q and d must be 2D matrices");
        goto cleanup;
    }

    {
        nk_dtype_t q_dtype, d_dtype;
        if (!buffer_dtype(&q_buffer, &q_dtype)) goto cleanup;
        if (!buffer_dtype(&d_buffer, &d_dtype)) goto cleanup;

        if (q_dtype != target_dtype) {
            PyErr_Format(PyExc_TypeError, "q dtype '%s' does not match target dtype '%s'",
                         dtype_to_python_string(q_dtype), dtype_to_python_string(target_dtype));
            goto cleanup;
        }
        if (d_dtype != target_dtype) {
            PyErr_Format(PyExc_TypeError, "d dtype '%s' does not match target dtype '%s'",
                         dtype_to_python_string(d_dtype), dtype_to_python_string(target_dtype));
            goto cleanup;
        }
    }

    if (q_buffer.strides[1] != (Py_ssize_t)bytes_per_dtype(target_dtype) ||
        d_buffer.strides[1] != (Py_ssize_t)bytes_per_dtype(target_dtype)) {
        PyErr_SetString(PyExc_ValueError, "Input matrices must be row-contiguous");
        goto cleanup;
    }

    {
        nk_size_t q_n = (nk_size_t)q_buffer.shape[0], q_k = (nk_size_t)q_buffer.shape[1];
        nk_size_t d_n = (nk_size_t)d_buffer.shape[0], d_k = (nk_size_t)d_buffer.shape[1];
        nk_size_t q_stride = (nk_size_t)q_buffer.strides[0];
        nk_size_t d_stride = (nk_size_t)d_buffer.strides[0];

        if (q_k != d_k) {
            PyErr_Format(PyExc_ValueError, "Depth mismatch: q has k=%zu but d has k=%zu", (size_t)q_k, (size_t)d_k);
            goto cleanup;
        }

        nk_dots_packed_size_punned_t size_fn = NULL;
        nk_capability_t cap = nk_cap_serial_k;
        nk_find_kernel_punned(nk_kernel_maxsim_packed_size_k, target_dtype, static_capabilities, nk_cap_any_k,
                              (nk_kernel_punned_t *)&size_fn, &cap);
        if (!size_fn) {
            PyErr_Format(PyExc_LookupError, "No maxsim packed_size kernel for dtype '%s'", dtype_str);
            goto cleanup;
        }

        nk_dots_pack_punned_t pack_fn = NULL;
        cap = nk_cap_serial_k;
        nk_find_kernel_punned(nk_kernel_maxsim_pack_k, target_dtype, static_capabilities, nk_cap_any_k,
                              (nk_kernel_punned_t *)&pack_fn, &cap);
        if (!pack_fn) {
            PyErr_Format(PyExc_LookupError, "No maxsim pack kernel for dtype '%s'", dtype_str);
            goto cleanup;
        }

        nk_maxsim_packed_punned_t kernel = NULL;
        cap = nk_cap_serial_k;
        nk_find_kernel_punned(nk_kernel_maxsim_packed_k, target_dtype, static_capabilities, nk_cap_any_k,
                              (nk_kernel_punned_t *)&kernel, &cap);
        if (!kernel) {
            PyErr_Format(PyExc_LookupError, "No maxsim_packed kernel for dtype '%s'", dtype_str);
            goto cleanup;
        }

        nk_size_t q_packed_size = size_fn(q_n, q_k);
        nk_size_t d_packed_size = size_fn(d_n, d_k);

        q_packed = PyObject_NewVar(MaxSimPackedMatrix, &MaxSimPackedMatrixType, q_packed_size);
        if (!q_packed) {
            PyErr_NoMemory();
            goto cleanup;
        }
        q_packed->dtype = target_dtype;
        q_packed->n = q_n;
        q_packed->k = q_k;

        d_packed = PyObject_NewVar(MaxSimPackedMatrix, &MaxSimPackedMatrixType, d_packed_size);
        if (!d_packed) {
            PyErr_NoMemory();
            goto cleanup;
        }
        d_packed->dtype = target_dtype;
        d_packed->n = d_n;
        d_packed->k = d_k;

        {
            PyThreadState *save = PyEval_SaveThread();
            pack_fn(q_buffer.buf, q_n, q_k, q_stride, q_packed->start);
            pack_fn(d_buffer.buf, d_n, d_k, d_stride, d_packed->start);
            PyEval_RestoreThread(save);
        }

        nk_f32_t result = 0;
        {
            PyThreadState *save = PyEval_SaveThread();
            kernel(q_packed->start, d_packed->start, q_n, d_n, q_k, &result);
            PyEval_RestoreThread(save);
        }

        return_obj = PyFloat_FromDouble((double)result);
    }

cleanup:
    if (have_q) PyBuffer_Release(&q_buffer);
    if (have_d) PyBuffer_Release(&d_buffer);
    Py_XDECREF(q_packed);
    Py_XDECREF(d_packed);
    return return_obj;
}

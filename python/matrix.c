/**
 *  @brief Matrix multiplication and symmetric operations for NumKong Python bindings.
 *  @file python/matrix.c
 *  @author Ash Vardanian
 *  @date February 20, 2026
 *
 *  Contains:
 *  - PackedMatrix type (pre-packed matrix for fast GEMM or Hamming distance)
 *  - api_dots_pack: pack a matrix for repeated dot-product multiplication
 *  - api_dots_packed: compute C = A @ packed_B (dot products)
 *  - api_hammings_pack: pack a matrix for repeated Hamming distance
 *  - api_hammings_packed: compute C = hammings(A, packed_B)
 *  - Tensor_matmul: the Tensor @ operator implementation
 *  - api_dots_symmetric: symmetric dot-product (Gram) matrix
 *  - api_hammings_symmetric: symmetric Hamming distance matrix
 */
#include "matrix.h"
#include "tensor.h"

#include <numkong/dots.h>

static void PackedMatrix_dealloc(PyObject *self) { Py_TYPE(self)->tp_free(self); }

/** @brief Compute packed buffer size for a PackedMatrix. */
static size_t packed_matrix_nbytes(PackedMatrix *mm) {
    nk_kernel_kind_t size_kind;
    if (mm->kind == nk_kernel_hammings_packed_k) size_kind = nk_kernel_hammings_packed_size_k;
    else size_kind = nk_kernel_dots_packed_size_k;

    nk_dots_packed_size_punned_t size_fn = NULL;
    nk_capability_t cap = nk_cap_serial_k;
    nk_find_kernel_punned(size_kind, mm->dtype, static_capabilities, nk_cap_any_k, (nk_kernel_punned_t *)&size_fn,
                          &cap);
    if (!size_fn) return 0;
    return size_fn(mm->n, mm->k);
}

static PyObject *PackedMatrix_repr(PyObject *self) {
    PackedMatrix *mm = (PackedMatrix *)self;
    size_t packed_size = packed_matrix_nbytes(mm);
    char const *kind_str = (mm->kind == nk_kernel_hammings_packed_k) ? "hammings" : "dots";
    return PyUnicode_FromFormat("<PackedMatrix kind='%s' n=%zu k=%zu dtype='%s' nbytes=%zu>", kind_str, (size_t)mm->n,
                                (size_t)mm->k, dtype_to_string(mm->dtype), packed_size);
}

static PyObject *PackedMatrix_get_n(PyObject *self, void *closure) {
    (void)closure;
    return PyLong_FromSize_t(((PackedMatrix *)self)->n);
}

static PyObject *PackedMatrix_get_k(PyObject *self, void *closure) {
    (void)closure;
    return PyLong_FromSize_t(((PackedMatrix *)self)->k);
}

static PyObject *PackedMatrix_get_dtype(PyObject *self, void *closure) {
    (void)closure;
    return PyUnicode_FromString(dtype_to_string(((PackedMatrix *)self)->dtype));
}

static PyObject *PackedMatrix_get_nbytes(PyObject *self, void *closure) {
    (void)closure;
    return PyLong_FromSize_t(packed_matrix_nbytes((PackedMatrix *)self));
}

static PyObject *PackedMatrix_get_kind(PyObject *self, void *closure) {
    (void)closure;
    PackedMatrix *mm = (PackedMatrix *)self;
    return PyUnicode_FromString((mm->kind == nk_kernel_hammings_packed_k) ? "hammings" : "dots");
}

static PyGetSetDef PackedMatrix_getset[] = {
    {"n", PackedMatrix_get_n, NULL, "Number of rows in the original matrix", NULL},
    {"k", PackedMatrix_get_k, NULL, "Number of columns in the original matrix", NULL},
    {"dtype", PackedMatrix_get_dtype, NULL, "Data type of the matrix elements", NULL},
    {"nbytes", PackedMatrix_get_nbytes, NULL, "Size of the packed buffer in bytes", NULL},
    {"kind", PackedMatrix_get_kind, NULL, "Kind of packed matrix ('dots' or 'hammings')", NULL},
    {NULL, NULL, NULL, NULL, NULL},
};

static PyObject *PackedMatrix_packed_size(PyObject *cls, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    (void)cls;

    PyObject *n_obj = NULL, *k_obj = NULL, *dtype_obj = NULL, *kind_obj = NULL;
    Py_ssize_t nkw = kwnames ? PyTuple_Size(kwnames) : 0;
    Py_ssize_t total = nargs + nkw;

    if (nargs < 2 || total > 4 || nargs > 3) {
        PyErr_SetString(PyExc_TypeError, "packed_size(n, k, dtype, *, kind='dots')");
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
        else if (PyUnicode_CompareWithASCIIString(name, "kind") == 0) kind_obj = value;
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

    nk_kernel_kind_t size_kind = nk_kernel_dots_packed_size_k;
    if (kind_obj) {
        char const *kind_str = PyUnicode_AsUTF8(kind_obj);
        if (!kind_str) return NULL;
        if (same_string(kind_str, "hammings")) size_kind = nk_kernel_hammings_packed_size_k;
        else if (!same_string(kind_str, "dots")) {
            PyErr_Format(PyExc_ValueError, "Unknown kind: '%s'. Expected 'dots' or 'hammings'.", kind_str);
            return NULL;
        }
    }

    nk_dots_packed_size_punned_t size_fn = NULL;
    nk_capability_t cap = nk_cap_serial_k;
    nk_find_kernel_punned(size_kind, dtype, static_capabilities, nk_cap_any_k, (nk_kernel_punned_t *)&size_fn, &cap);
    if (!size_fn) {
        PyErr_Format(PyExc_LookupError, "No packed_size kernel for dtype '%s'", dtype_str);
        return NULL;
    }

    return PyLong_FromSize_t(size_fn(n, k));
}

static PyMethodDef PackedMatrix_methods[] = {
    {"packed_size", (PyCFunction)PackedMatrix_packed_size, METH_CLASS | METH_FASTCALL | METH_KEYWORDS,
     "Return the packed buffer size in bytes for given dimensions and dtype."},
    {NULL, NULL, 0, NULL},
};

PyTypeObject PackedMatrixType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "numkong.PackedMatrix",
    .tp_doc = "Pre-packed matrix optimized for matrix multiplication or Hamming distance",
    .tp_basicsize = sizeof(PackedMatrix),
    .tp_itemsize = sizeof(char),
    .tp_dealloc = PackedMatrix_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_getset = PackedMatrix_getset,
    .tp_methods = PackedMatrix_methods,
    .tp_repr = PackedMatrix_repr,
};

/** @brief Parse a Python buffer format string into a NumKong dtype. */
static int buffer_dtype(Py_buffer const *buffer, nk_dtype_t *dtype) {
    char const *format = buffer->format;
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

/** @brief Matrix multiplication operator for Tensor @ PackedMatrix. */
PyObject *Tensor_matmul(PyObject *self, PyObject *other) {
    if (!PyObject_TypeCheck(self, &TensorType)) { Py_RETURN_NOTIMPLEMENTED; }
    Tensor *a = (Tensor *)self;

    if (!PyObject_TypeCheck(other, &PackedMatrixType)) {
        PyErr_SetString(PyExc_TypeError, "matmul requires PackedMatrix as right operand " "(use nk.dots_pack() first)");
        return NULL;
    }

    PackedMatrix *packed = (PackedMatrix *)other;

    if (packed->kind != nk_kernel_dots_packed_k) {
        PyErr_SetString(PyExc_TypeError, "@ operator requires a dots-packed matrix (use nk.dots_packed() for dots)");
        return NULL;
    }

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

    // Require matching dtype and row-contiguous input
    if (a->dtype != packed->dtype) {
        PyErr_Format(PyExc_TypeError,
                     "dtype mismatch: tensor is '%s' but packed matrix is '%s'. " "Use .astype('%s') to convert first.",
                     dtype_to_python_string(a->dtype), dtype_to_python_string(packed->dtype),
                     dtype_to_python_string(packed->dtype));
        return NULL;
    }
    if (col_stride != (nk_size_t)bytes_per_dtype(packed->dtype)) {
        PyErr_SetString(PyExc_ValueError, "matmul requires row-contiguous left operand");
        return NULL;
    }

    // Determine output dtype
    nk_dtype_t out_dtype = nk_dot_output_dtype(packed->dtype);
    if (out_dtype == nk_dtype_unknown_k) {
        PyErr_SetString(PyExc_ValueError, "Unsupported packed matrix dtype");
        return NULL;
    }

    // Find matmul kernel via punned dispatch
    nk_dots_punned_t matmul_fn = NULL;
    nk_capability_t cap = nk_cap_serial_k;
    nk_find_kernel_punned(nk_kernel_dots_packed_k, packed->dtype, static_capabilities, nk_cap_any_k,
                          (nk_kernel_punned_t *)&matmul_fn, &cap);
    if (!matmul_fn) {
        PyErr_SetString(PyExc_LookupError, "No matmul kernel for this dtype");
        return NULL;
    }

    // Allocate output tensor
    Py_ssize_t out_shape[2] = {(Py_ssize_t)m, (Py_ssize_t)n};
    Tensor *result = Tensor_new(out_dtype, 2, out_shape);
    if (!result) return NULL;

    nk_size_t c_stride = n * bytes_per_dtype(out_dtype);
    PyThreadState *save = PyEval_SaveThread();
    matmul_fn(a->data, packed->start, result->data, m, n, k, row_stride, c_stride);
    PyEval_RestoreThread(save);

    return (PyObject *)result;
}

char const doc_dots_pack[] =                                                                    //
    "dots_pack(b, dtype='bf16') -> PackedMatrix\n\n"                                            //
    "Pack a matrix for repeated dot-product matrix multiplication.\n\n"                         //
    "The packed format is opaque and backend-specific, optimized for the available\n"           //
    "hardware (AMX on Intel, NEON/SVE on ARM, etc.).\n"                                         //
    "Use with dots_packed() or the @ operator to compute C = A @ B.\n\n"                        //
    "Parameters:\n"                                                                             //
    "    b : array_like\n"                                                                      //
    "        The (n, k) matrix to pack. This is typically the 'database' or 'weights' matrix\n" //
    "        that will be multiplied against multiple 'query' matrices.\n"                      //
    "    dtype : str, optional\n"                                                               //
    "        Data type for packing. Supported types:\n"                                         //
    "        - 'bf16'/'bfloat16' (default): BF16 with F32 accumulation\n"                       //
    "        - 'f16'/'float16': F16 with F32 accumulation\n"                                    //
    "        - 'f32'/'float32': Native F32\n"                                                   //
    "        - 'f64'/'float64': Native F64\n"                                                   //
    "        - 'i8'/'int8': I8 with I32 accumulation\n"                                         //
    "        - 'u8'/'uint8': U8 with U32 accumulation\n\n"                                      //
    "Returns:\n"                                                                                //
    "    PackedMatrix : Opaque packed matrix for use with dots_packed() or @.\n\n"              //
    "Example:\n"                                                                                //
    "    >>> database = np.random.randn(1000, 768).astype(np.float32)\n"                        //
    "    >>> packed = nk.dots_pack(database, dtype='bf16')\n"                                   //
    "    >>> queries = nk.zeros((10, 768), dtype='float32')\n"                                  //
    "    >>> result = queries @ packed  # (10, 1000) dot products\n";                           //

PyObject *api_dots_pack(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    (void)self;

    PyObject *b_obj = NULL;
    char const *dtype_str = "bf16";

    Py_ssize_t nkw = kwnames ? PyTuple_Size(kwnames) : 0;
    Py_ssize_t total = nargs + nkw;

    if (nargs < 1 || total > 2) {
        PyErr_SetString(PyExc_TypeError, "dots_pack() requires 1-2 arguments: b, dtype='bf16'");
        return NULL;
    }

    b_obj = args[0];

    for (Py_ssize_t i = 0; i < nkw; i++) {
        PyObject *name = PyTuple_GET_ITEM(kwnames, i);
        if (PyUnicode_CompareWithASCIIString(name, "dtype") == 0) {
            if (nargs >= 2) {
                PyErr_SetString(PyExc_TypeError, "dots_pack() got multiple values for argument 'dtype'");
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
            PyErr_Format(PyExc_TypeError, "dots_pack() got unexpected keyword argument '%s'", name_str);
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

    // Resolve the target packing dtype (short aliases + full names)
    nk_dtype_t target_dtype;
    if (same_string(dtype_str, "bf16") || same_string(dtype_str, "bfloat16")) target_dtype = nk_bf16_k;
    else if (same_string(dtype_str, "i8") || same_string(dtype_str, "int8")) target_dtype = nk_i8_k;
    else if (same_string(dtype_str, "f32") || same_string(dtype_str, "float32")) target_dtype = nk_f32_k;
    else if (same_string(dtype_str, "f64") || same_string(dtype_str, "float64")) target_dtype = nk_f64_k;
    else if (same_string(dtype_str, "f16") || same_string(dtype_str, "float16")) target_dtype = nk_f16_k;
    else if (same_string(dtype_str, "u8") || same_string(dtype_str, "uint8")) target_dtype = nk_u8_k;
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

    // Require row-contiguous input with matching dtype
    if (src_dtype != target_dtype) {
        PyBuffer_Release(&b_buffer);
        PyErr_Format(
            PyExc_TypeError,
            "Input dtype '%s' does not match target dtype '%s'. " "Cast the input first (e.g., " "array.astype(np." "fl" "oa" "t3" "2)" ")" ".",
            dtype_to_python_string(src_dtype), dtype_to_python_string(target_dtype));
        return NULL;
    }
    if (col_stride != (nk_size_t)bytes_per_dtype(target_dtype)) {
        PyBuffer_Release(&b_buffer);
        PyErr_SetString(PyExc_ValueError, "Input matrix must be row-contiguous");
        return NULL;
    }

    // Get packed size via punned dispatch
    nk_dots_packed_size_punned_t size_fn = NULL;
    nk_capability_t cap = nk_cap_serial_k;
    nk_find_kernel_punned(nk_kernel_dots_packed_size_k, target_dtype, static_capabilities, nk_cap_any_k,
                          (nk_kernel_punned_t *)&size_fn, &cap);
    if (!size_fn) {
        PyBuffer_Release(&b_buffer);
        PyErr_Format(PyExc_LookupError, "No packing kernel for dtype '%s'", dtype_to_python_string(target_dtype));
        return NULL;
    }
    nk_size_t packed_size = size_fn(n, k);

    PackedMatrix *packed = PyObject_NewVar(PackedMatrix, &PackedMatrixType, packed_size);
    if (!packed) {
        PyBuffer_Release(&b_buffer);
        PyErr_NoMemory();
        return NULL;
    }

    packed->kind = nk_kernel_dots_packed_k;
    packed->dtype = target_dtype;
    packed->n = n;
    packed->k = k;

    // Pack via punned dispatch
    nk_dots_pack_punned_t pack_fn = NULL;
    cap = nk_cap_serial_k;
    nk_find_kernel_punned(nk_kernel_dots_pack_k, target_dtype, static_capabilities, nk_cap_any_k,
                          (nk_kernel_punned_t *)&pack_fn, &cap);
    if (!pack_fn) {
        Py_DECREF(packed);
        PyBuffer_Release(&b_buffer);
        PyErr_Format(PyExc_LookupError, "No pack kernel for dtype '%s'", dtype_to_python_string(target_dtype));
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

char const doc_dots_packed[] =                                                       //
    "dots_packed(a, b, *, out=None) -> Tensor\n\n"                                   //
    "Compute matrix multiplication C = A @ B with a pre-packed B matrix.\n\n"        //
    "Parameters:\n"                                                                  //
    "    a : array_like\n"                                                           //
    "        The (m, k) query/input matrix.\n"                                       //
    "    b : PackedMatrix\n"                                                         //
    "        Pre-packed (n, k) matrix from dots_pack().\n"                           //
    "    out : Tensor, optional\n"                                                   //
    "        Pre-allocated output tensor. Must have correct shape (m, n),\n"         //
    "        correct dtype for the operation, and be C-contiguous.\n"                //
    "        If provided, no memory allocation is performed.\n\n"                    //
    "Returns:\n"                                                                     //
    "    Tensor : (m, n) result matrix. If out is provided, returns out.\n\n"        //
    "Note:\n"                                                                        //
    "    The kernel computes C[i,j] = dot(a[i], b[j]) for all i,j.\n"                //
    "    This is equivalent to A @ B.T where B is the original unpacked matrix.\n\n" //
    "    Output dtype depends on packed dtype:\n"                                    //
    "    - bf16, f16 -> float32\n"                                                   //
    "    - f32 -> float32\n"                                                         //
    "    - f64 -> float64\n"                                                         //
    "    - i8 -> int32\n"                                                            //
    "    - u8 -> uint32\n\n"                                                         //
    "Example:\n"                                                                     //
    "    >>> database = np.random.randn(1000, 768).astype(np.float32)\n"             //
    "    >>> packed = nk.dots_pack(database, dtype='bf16')\n"                        //
    "    >>> queries = np.random.randn(10, 768).astype(np.float32)\n"                //
    "    >>> result = nk.dots_packed(queries, packed)  # (10, 1000)\n"               //
    "    >>>\n"                                                                      //
    "    >>> # Reuse output buffer for zero-allocation inference:\n"                 //
    "    >>> out = nk.empty((10, 1000), dtype='float32')\n"                          //
    "    >>> nk.dots_packed(queries, packed, out=out)\n";                            //

PyObject *api_dots_packed(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    (void)self;

    PyObject *a_obj = NULL;
    PyObject *b_obj = NULL;
    PyObject *out_obj = NULL;

    Py_ssize_t nkw = kwnames ? PyTuple_Size(kwnames) : 0;
    if (nargs != 2) {
        PyErr_SetString(PyExc_TypeError, "dots_packed() requires exactly 2 positional arguments: a, b");
        return NULL;
    }

    a_obj = args[0];
    b_obj = args[1];

    for (Py_ssize_t i = 0; i < nkw; i++) {
        PyObject *name = PyTuple_GET_ITEM(kwnames, i);
        if (PyUnicode_CompareWithASCIIString(name, "out") == 0) { out_obj = args[nargs + i]; }
        else {
            char const *name_str = PyUnicode_AsUTF8(name);
            PyErr_Format(PyExc_TypeError, "dots_packed() got unexpected keyword argument '%s'", name_str);
            return NULL;
        }
    }

    if (!PyObject_TypeCheck(b_obj, &PackedMatrixType)) {
        PyErr_SetString(PyExc_TypeError, "b must be a PackedMatrix (use dots_pack() first)");
        return NULL;
    }
    PackedMatrix *packed = (PackedMatrix *)b_obj;

    if (packed->kind != nk_kernel_dots_packed_k) {
        PyErr_SetString(PyExc_TypeError, "b must be a dots-packed matrix (created with dots_pack())");
        return NULL;
    }

    Py_buffer a_buffer;
    if (PyObject_GetBuffer(a_obj, &a_buffer, PyBUF_STRIDES | PyBUF_FORMAT) != 0) {
        PyErr_SetString(PyExc_TypeError, "a must support buffer protocol");
        return NULL;
    }

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

    // Require matching dtype and row-contiguous input
    if (src_dtype != packed->dtype) {
        PyBuffer_Release(&a_buffer);
        PyErr_Format(PyExc_TypeError,
                     "dtype mismatch: input is '%s' but packed matrix is '%s'. " "Cast the input first.",
                     dtype_to_python_string(src_dtype), dtype_to_python_string(packed->dtype));
        return NULL;
    }
    if (col_stride != (nk_size_t)bytes_per_dtype(packed->dtype)) {
        PyBuffer_Release(&a_buffer);
        PyErr_SetString(PyExc_ValueError, "matmul requires row-contiguous left operand");
        return NULL;
    }

    nk_size_t n = packed->n;
    nk_size_t k = packed->k;
    nk_dtype_t out_dtype = nk_dot_output_dtype(packed->dtype);

    // Find matmul kernel
    nk_dots_punned_t matmul_fn = NULL;
    nk_capability_t cap = nk_cap_serial_k;
    nk_find_kernel_punned(nk_kernel_dots_packed_k, packed->dtype, static_capabilities, nk_cap_any_k,
                          (nk_kernel_punned_t *)&matmul_fn, &cap);
    if (!matmul_fn) {
        PyBuffer_Release(&a_buffer);
        PyErr_SetString(PyExc_LookupError, "No matmul kernel for this dtype");
        return NULL;
    }

    // Handle output tensor
    Tensor *result = NULL;
    int owns_result = 0;
    char *out_data = NULL;
    nk_size_t c_stride;

    if (out_obj && out_obj != Py_None) {
        if (!PyObject_TypeCheck(out_obj, &TensorType)) {
            PyBuffer_Release(&a_buffer);
            PyErr_SetString(PyExc_TypeError, "out must be a Tensor");
            return NULL;
        }
        result = (Tensor *)out_obj;

        if (result->rank != 2 || result->shape[0] != (Py_ssize_t)m || result->shape[1] != (Py_ssize_t)n) {
            PyBuffer_Release(&a_buffer);
            PyErr_Format(PyExc_ValueError, "out has wrong shape: expected (%zu, %zu), got (%zd, %zd)", m, n,
                         result->shape[0], result->shape[1]);
            return NULL;
        }

        if (result->dtype != out_dtype) {
            PyBuffer_Release(&a_buffer);
            PyErr_Format(PyExc_TypeError, "out dtype '%s' does not match expected '%s'",
                         dtype_to_python_string(result->dtype), dtype_to_python_string(out_dtype));
            return NULL;
        }

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

    {
        PyThreadState *save = PyEval_SaveThread();
        matmul_fn(a_buffer.buf, packed->start, out_data, m, n, k, row_stride, c_stride);
        PyEval_RestoreThread(save);
    }
    PyBuffer_Release(&a_buffer);

    if (owns_result) { return (PyObject *)result; }
    else {
        Py_INCREF(result);
        return (PyObject *)result;
    }
}

char const doc_hammings_pack[] =                                                 //
    "hammings_pack(b, dtype='bin8') -> PackedMatrix\n\n"                         //
    "Pack a matrix for repeated Hamming distance computation.\n\n"               //
    "Parameters:\n"                                                              //
    "    b : array_like\n"                                                       //
    "        The (n, k) matrix to pack.\n"                                       //
    "    dtype : str, optional\n"                                                //
    "        Data type for packing. Default: 'bin8'.\n\n"                        //
    "Returns:\n"                                                                 //
    "    PackedMatrix : Opaque packed matrix for use with hammings_packed().\n"; //

PyObject *api_hammings_pack(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    (void)self;

    PyObject *b_obj = NULL;
    char const *dtype_str = "bin8";

    Py_ssize_t nkw = kwnames ? PyTuple_Size(kwnames) : 0;
    Py_ssize_t total = nargs + nkw;

    if (nargs < 1 || total > 2) {
        PyErr_SetString(PyExc_TypeError, "hammings_pack() requires 1-2 arguments: b, dtype='bin8'");
        return NULL;
    }

    b_obj = args[0];

    for (Py_ssize_t i = 0; i < nkw; i++) {
        PyObject *name = PyTuple_GET_ITEM(kwnames, i);
        if (PyUnicode_CompareWithASCIIString(name, "dtype") == 0) {
            if (nargs >= 2) {
                PyErr_SetString(PyExc_TypeError, "hammings_pack() got multiple values for argument 'dtype'");
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
            PyErr_Format(PyExc_TypeError, "hammings_pack() got unexpected keyword argument '%s'", name_str);
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
        PyErr_SetString(PyExc_ValueError, "hammings packing does not support negative strides");
        return NULL;
    }

    nk_size_t n = (nk_size_t)b_buffer.shape[0];
    nk_size_t k = (nk_size_t)b_buffer.shape[1];
    nk_size_t row_stride = (nk_size_t)b_buffer.strides[0];
    nk_size_t col_stride = (nk_size_t)b_buffer.strides[1];

    if (src_dtype != target_dtype) {
        PyBuffer_Release(&b_buffer);
        PyErr_Format(PyExc_TypeError, "Input dtype '%s' does not match target dtype '%s'.",
                     dtype_to_python_string(src_dtype), dtype_to_python_string(target_dtype));
        return NULL;
    }
    if (col_stride != (nk_size_t)bytes_per_dtype(target_dtype)) {
        PyBuffer_Release(&b_buffer);
        PyErr_SetString(PyExc_ValueError, "Input matrix must be row-contiguous");
        return NULL;
    }

    nk_hammings_packed_size_punned_t size_fn = NULL;
    nk_capability_t cap = nk_cap_serial_k;
    nk_find_kernel_punned(nk_kernel_hammings_packed_size_k, target_dtype, static_capabilities, nk_cap_any_k,
                          (nk_kernel_punned_t *)&size_fn, &cap);
    if (!size_fn) {
        PyBuffer_Release(&b_buffer);
        PyErr_Format(PyExc_LookupError, "No hammings packing kernel for dtype '%s'",
                     dtype_to_python_string(target_dtype));
        return NULL;
    }
    nk_size_t packed_size = size_fn(n, k);

    PackedMatrix *packed = PyObject_NewVar(PackedMatrix, &PackedMatrixType, packed_size);
    if (!packed) {
        PyBuffer_Release(&b_buffer);
        PyErr_NoMemory();
        return NULL;
    }

    packed->kind = nk_kernel_hammings_packed_k;
    packed->dtype = target_dtype;
    packed->n = n;
    packed->k = k;

    nk_hammings_pack_punned_t pack_fn = NULL;
    cap = nk_cap_serial_k;
    nk_find_kernel_punned(nk_kernel_hammings_pack_k, target_dtype, static_capabilities, nk_cap_any_k,
                          (nk_kernel_punned_t *)&pack_fn, &cap);
    if (!pack_fn) {
        Py_DECREF(packed);
        PyBuffer_Release(&b_buffer);
        PyErr_Format(PyExc_LookupError, "No hammings pack kernel for dtype '%s'", dtype_to_python_string(target_dtype));
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

char const doc_hammings_packed[] =                                        //
    "hammings_packed(a, b, *, out=None) -> Tensor\n\n"                    //
    "Compute Hamming distances C = hammings(A, packed_B).\n\n"            //
    "Parameters:\n"                                                       //
    "    a : array_like\n"                                                //
    "        The (m, k) query matrix.\n"                                  //
    "    b : PackedMatrix\n"                                              //
    "        Pre-packed (n, k) matrix from hammings_pack().\n"            //
    "    out : Tensor, optional\n"                                        //
    "        Pre-allocated output tensor of uint32.\n\n"                  //
    "Returns:\n"                                                          //
    "    Tensor : (m, n) result matrix of Hamming distances (uint32).\n"; //

PyObject *api_hammings_packed(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    (void)self;

    PyObject *a_obj = NULL;
    PyObject *b_obj = NULL;
    PyObject *out_obj = NULL;

    Py_ssize_t nkw = kwnames ? PyTuple_Size(kwnames) : 0;
    if (nargs != 2) {
        PyErr_SetString(PyExc_TypeError, "hammings_packed() requires exactly 2 positional arguments: a, b");
        return NULL;
    }

    a_obj = args[0];
    b_obj = args[1];

    for (Py_ssize_t i = 0; i < nkw; i++) {
        PyObject *name = PyTuple_GET_ITEM(kwnames, i);
        if (PyUnicode_CompareWithASCIIString(name, "out") == 0) { out_obj = args[nargs + i]; }
        else {
            char const *name_str = PyUnicode_AsUTF8(name);
            PyErr_Format(PyExc_TypeError, "hammings_packed() got unexpected keyword argument '%s'", name_str);
            return NULL;
        }
    }

    if (!PyObject_TypeCheck(b_obj, &PackedMatrixType)) {
        PyErr_SetString(PyExc_TypeError, "b must be a PackedMatrix (use hammings_pack() first)");
        return NULL;
    }
    PackedMatrix *packed = (PackedMatrix *)b_obj;

    if (packed->kind != nk_kernel_hammings_packed_k) {
        PyErr_SetString(PyExc_TypeError, "b must be a hammings-packed matrix (created with hammings_pack())");
        return NULL;
    }

    Py_buffer a_buffer;
    if (PyObject_GetBuffer(a_obj, &a_buffer, PyBUF_STRIDES | PyBUF_FORMAT) != 0) {
        PyErr_SetString(PyExc_TypeError, "a must support buffer protocol");
        return NULL;
    }

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

    if (a_buffer.strides[0] < 0 || a_buffer.strides[1] < 0) {
        PyBuffer_Release(&a_buffer);
        PyErr_SetString(PyExc_ValueError, "hammings does not support negative strides");
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

    if (src_dtype != packed->dtype) {
        PyBuffer_Release(&a_buffer);
        PyErr_Format(PyExc_TypeError, "dtype mismatch: input is '%s' but packed matrix is '%s'.",
                     dtype_to_python_string(src_dtype), dtype_to_python_string(packed->dtype));
        return NULL;
    }
    if (col_stride != (nk_size_t)bytes_per_dtype(packed->dtype)) {
        PyBuffer_Release(&a_buffer);
        PyErr_SetString(PyExc_ValueError, "hammings requires row-contiguous left operand");
        return NULL;
    }

    nk_size_t n = packed->n;
    nk_size_t k = packed->k;
    nk_dtype_t out_dtype = nk_u32_k; // Hamming output is always u32

    nk_hammings_punned_t hammings_fn = NULL;
    nk_capability_t cap = nk_cap_serial_k;
    nk_find_kernel_punned(nk_kernel_hammings_packed_k, packed->dtype, static_capabilities, nk_cap_any_k,
                          (nk_kernel_punned_t *)&hammings_fn, &cap);
    if (!hammings_fn) {
        PyBuffer_Release(&a_buffer);
        PyErr_SetString(PyExc_LookupError, "No hammings kernel for this dtype");
        return NULL;
    }

    Tensor *result = NULL;
    int owns_result = 0;
    char *out_data = NULL;
    nk_size_t c_stride;

    if (out_obj && out_obj != Py_None) {
        if (!PyObject_TypeCheck(out_obj, &TensorType)) {
            PyBuffer_Release(&a_buffer);
            PyErr_SetString(PyExc_TypeError, "out must be a Tensor");
            return NULL;
        }
        result = (Tensor *)out_obj;

        if (result->rank != 2 || result->shape[0] != (Py_ssize_t)m || result->shape[1] != (Py_ssize_t)n) {
            PyBuffer_Release(&a_buffer);
            PyErr_Format(PyExc_ValueError, "out has wrong shape: expected (%zu, %zu), got (%zd, %zd)", m, n,
                         result->shape[0], result->shape[1]);
            return NULL;
        }

        if (result->dtype != out_dtype) {
            PyBuffer_Release(&a_buffer);
            PyErr_Format(PyExc_TypeError, "out dtype '%s' does not match expected 'uint32'",
                         dtype_to_python_string(result->dtype));
            return NULL;
        }

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

    {
        PyThreadState *save = PyEval_SaveThread();
        hammings_fn(a_buffer.buf, packed->start, out_data, m, n, k, row_stride, c_stride);
        PyEval_RestoreThread(save);
    }
    PyBuffer_Release(&a_buffer);

    if (owns_result) { return (PyObject *)result; }
    else {
        Py_INCREF(result);
        return (PyObject *)result;
    }
}

char const doc_dots_symmetric[] =                                               //
    "Compute the symmetric dot-product (Gram) matrix for a set of vectors.\n\n" //
    "Parameters:\n"                                                             //
    "    vectors (Tensor): 2D input matrix of shape (n, depth).\n"              //
    "    dtype (str, optional): Override the presumed input type.\n"            //
    "    out (Tensor, optional): Pre-allocated (n, n) output Tensor.\n\n"       //
    "Returns:\n"                                                                //
    "    Tensor: (n, n) symmetric matrix of dot products.\n\n"                  //
    "Signature:\n"                                                              //
    "    >>> def dots_symmetric(vectors, /, *, dtype=None, out=None) -> Tensor: ...";

PyObject *api_dots_symmetric( //
    PyObject *self, PyObject *const *args, Py_ssize_t const positional_args_count, PyObject *args_names_tuple) {
    (void)self;

    PyObject *vectors_obj = NULL;
    PyObject *dtype_obj = NULL;
    PyObject *out_obj = NULL;

    Py_ssize_t const args_names_count = args_names_tuple ? PyTuple_Size(args_names_tuple) : 0;
    Py_ssize_t const args_count = positional_args_count + args_names_count;
    if (args_count < 1 || args_count > 3 || positional_args_count > 1) {
        PyErr_SetString(PyExc_TypeError, "dots_symmetric(vectors, *, dtype=None, out=None)");
        return NULL;
    }

    vectors_obj = args[0];
    for (Py_ssize_t i = 0, j = positional_args_count; i < args_names_count; ++i, ++j) {
        PyObject *key = PyTuple_GetItem(args_names_tuple, i);
        PyObject *value = args[j];
        if (PyUnicode_CompareWithASCIIString(key, "dtype") == 0) dtype_obj = value;
        else if (PyUnicode_CompareWithASCIIString(key, "out") == 0) out_obj = value;
        else {
            PyErr_Format(PyExc_TypeError, "dots_symmetric() unexpected keyword: %S", key);
            return NULL;
        }
    }

    // Use Py_buffer directly instead of parse_tensor
    Py_buffer vec_buf;
    if (PyObject_GetBuffer(vectors_obj, &vec_buf, PyBUF_STRIDES | PyBUF_FORMAT) != 0) {
        PyErr_SetString(PyExc_TypeError, "vectors must support buffer protocol");
        return NULL;
    }

    PyObject *return_obj = NULL;

    if (vec_buf.ndim != 2) {
        PyErr_SetString(PyExc_ValueError, "vectors must be a 2D matrix");
        goto cleanup;
    }

    if (vec_buf.strides[1] != vec_buf.itemsize) {
        PyErr_SetString(PyExc_ValueError, "Input rows must be contiguous");
        goto cleanup;
    }

    nk_dtype_t dtype;
    if (!buffer_dtype(&vec_buf, &dtype)) goto cleanup;

    if (dtype_obj) {
        char const *s = PyUnicode_AsUTF8(dtype_obj);
        if (!s) goto cleanup;
        dtype = python_string_to_dtype(s);
        if (dtype == nk_dtype_unknown_k) {
            PyErr_SetString(PyExc_ValueError, "Unsupported dtype");
            goto cleanup;
        }
    }

    nk_dots_symmetric_punned_t kernel = NULL;
    nk_capability_t cap = nk_cap_serial_k;
    nk_find_kernel_punned(nk_kernel_dots_symmetric_k, dtype, static_capabilities, nk_cap_any_k,
                          (nk_kernel_punned_t *)&kernel, &cap);
    if (!kernel || !cap) {
        PyErr_Format(PyExc_LookupError, "No dots_symmetric kernel for dtype '%s'", dtype_to_python_string(dtype));
        goto cleanup;
    }

    nk_dtype_t out_dtype = nk_dot_output_dtype(dtype);
    if (out_dtype == nk_dtype_unknown_k) {
        PyErr_SetString(PyExc_ValueError, "Cannot determine output dtype for dots_symmetric");
        goto cleanup;
    }

    nk_size_t n_vectors = (nk_size_t)vec_buf.shape[0];
    nk_size_t depth = (nk_size_t)vec_buf.shape[1];
    nk_size_t stride = (nk_size_t)vec_buf.strides[0];

    Tensor *result = NULL;
    if (out_obj && out_obj != Py_None) {
        if (!PyObject_TypeCheck(out_obj, &TensorType)) {
            PyErr_SetString(PyExc_TypeError, "out must be a Tensor");
            goto cleanup;
        }
        result = (Tensor *)out_obj;
        if (result->rank != 2 || result->shape[0] != (Py_ssize_t)n_vectors ||
            result->shape[1] != (Py_ssize_t)n_vectors) {
            PyErr_Format(PyExc_ValueError, "out has wrong shape: expected (%zu, %zu)", n_vectors, n_vectors);
            goto cleanup;
        }
        if (result->dtype != out_dtype) {
            PyErr_Format(PyExc_TypeError, "out dtype '%s' does not match expected '%s'",
                         dtype_to_python_string(result->dtype), dtype_to_python_string(out_dtype));
            goto cleanup;
        }
        size_t out_item_size = bytes_per_dtype(out_dtype);
        if (result->strides[1] != (Py_ssize_t)out_item_size ||
            result->strides[0] != (Py_ssize_t)(n_vectors * out_item_size)) {
            PyErr_SetString(PyExc_ValueError, "out must be row-contiguous (C-contiguous)");
            goto cleanup;
        }
    }
    else {
        Py_ssize_t out_shape[2] = {(Py_ssize_t)n_vectors, (Py_ssize_t)n_vectors};
        result = Tensor_new(out_dtype, 2, out_shape);
        if (!result) goto cleanup;
    }

    nk_size_t result_stride = n_vectors * bytes_per_dtype(out_dtype);
    {
        PyThreadState *save = PyEval_SaveThread();
        kernel(vec_buf.buf, n_vectors, depth, stride, result->data, result_stride, 0, n_vectors);
        PyEval_RestoreThread(save);
    }

    if (out_obj && out_obj != Py_None) {
        Py_INCREF(out_obj);
        return_obj = out_obj;
    }
    else { return_obj = (PyObject *)result; }

cleanup:
    PyBuffer_Release(&vec_buf);
    return return_obj;
}

char const doc_hammings_symmetric[] =                                                          //
    "Compute the symmetric Hamming distance matrix for a set of vectors.\n\n"                  //
    "Parameters:\n"                                                                            //
    "    vectors (Tensor): 2D input matrix of shape (n, depth). Dtype: bin8, uint8, uint32.\n" //
    "    dtype (str, optional): Override the presumed input type.\n"                           //
    "    out (Tensor, optional): Pre-allocated (n, n) output Tensor of uint32.\n\n"            //
    "Returns:\n"                                                                               //
    "    Tensor: (n, n) symmetric matrix of Hamming distances (uint32).\n\n"                   //
    "Signature:\n"                                                                             //
    "    >>> def hammings_symmetric(vectors, /, *, dtype=None, out=None) -> Tensor: ...";

PyObject *api_hammings_symmetric( //
    PyObject *self, PyObject *const *args, Py_ssize_t const positional_args_count, PyObject *args_names_tuple) {
    (void)self;

    PyObject *vectors_obj = NULL;
    PyObject *dtype_obj = NULL;
    PyObject *out_obj = NULL;

    Py_ssize_t const args_names_count = args_names_tuple ? PyTuple_Size(args_names_tuple) : 0;
    Py_ssize_t const args_count = positional_args_count + args_names_count;
    if (args_count < 1 || args_count > 3 || positional_args_count > 1) {
        PyErr_SetString(PyExc_TypeError, "hammings_symmetric(vectors, *, dtype=None, out=None)");
        return NULL;
    }

    vectors_obj = args[0];
    for (Py_ssize_t i = 0, j = positional_args_count; i < args_names_count; ++i, ++j) {
        PyObject *key = PyTuple_GetItem(args_names_tuple, i);
        PyObject *value = args[j];
        if (PyUnicode_CompareWithASCIIString(key, "dtype") == 0) dtype_obj = value;
        else if (PyUnicode_CompareWithASCIIString(key, "out") == 0) out_obj = value;
        else {
            PyErr_Format(PyExc_TypeError, "hammings_symmetric() unexpected keyword: %S", key);
            return NULL;
        }
    }

    Py_buffer vec_buf;
    if (PyObject_GetBuffer(vectors_obj, &vec_buf, PyBUF_STRIDES | PyBUF_FORMAT) != 0) {
        PyErr_SetString(PyExc_TypeError, "vectors must support buffer protocol");
        return NULL;
    }

    PyObject *return_obj = NULL;

    if (vec_buf.ndim != 2) {
        PyErr_SetString(PyExc_ValueError, "vectors must be a 2D matrix");
        goto cleanup;
    }

    if (vec_buf.strides[1] != vec_buf.itemsize) {
        PyErr_SetString(PyExc_ValueError, "Input rows must be contiguous");
        goto cleanup;
    }

    nk_dtype_t dtype;
    if (!buffer_dtype(&vec_buf, &dtype)) goto cleanup;

    if (dtype_obj) {
        char const *s = PyUnicode_AsUTF8(dtype_obj);
        if (!s) goto cleanup;
        dtype = python_string_to_dtype(s);
        if (dtype == nk_dtype_unknown_k) {
            PyErr_SetString(PyExc_ValueError, "Unsupported dtype");
            goto cleanup;
        }
    }

    nk_hammings_symmetric_punned_t kernel = NULL;
    nk_capability_t cap = nk_cap_serial_k;
    nk_find_kernel_punned(nk_kernel_hammings_symmetric_k, dtype, static_capabilities, nk_cap_any_k,
                          (nk_kernel_punned_t *)&kernel, &cap);
    if (!kernel || !cap) {
        PyErr_Format(PyExc_LookupError, "No hammings_symmetric kernel for dtype '%s'", dtype_to_python_string(dtype));
        goto cleanup;
    }

    nk_dtype_t out_dtype = nk_u32_k; // Hamming output is always u32
    nk_size_t n_vectors = (nk_size_t)vec_buf.shape[0];
    nk_size_t depth = (nk_size_t)vec_buf.shape[1];
    nk_size_t stride = (nk_size_t)vec_buf.strides[0];

    Tensor *result = NULL;
    if (out_obj && out_obj != Py_None) {
        if (!PyObject_TypeCheck(out_obj, &TensorType)) {
            PyErr_SetString(PyExc_TypeError, "out must be a Tensor");
            goto cleanup;
        }
        result = (Tensor *)out_obj;
        if (result->rank != 2 || result->shape[0] != (Py_ssize_t)n_vectors ||
            result->shape[1] != (Py_ssize_t)n_vectors) {
            PyErr_Format(PyExc_ValueError, "out has wrong shape: expected (%zu, %zu)", n_vectors, n_vectors);
            goto cleanup;
        }
        if (result->dtype != nk_u32_k) {
            PyErr_Format(PyExc_TypeError, "out dtype '%s' does not match expected '%s'",
                         dtype_to_python_string(result->dtype), dtype_to_python_string(nk_u32_k));
            goto cleanup;
        }
        size_t out_item_size = bytes_per_dtype(out_dtype);
        if (result->strides[1] != (Py_ssize_t)out_item_size ||
            result->strides[0] != (Py_ssize_t)(n_vectors * out_item_size)) {
            PyErr_SetString(PyExc_ValueError, "out must be row-contiguous (C-contiguous)");
            goto cleanup;
        }
    }
    else {
        Py_ssize_t out_shape[2] = {(Py_ssize_t)n_vectors, (Py_ssize_t)n_vectors};
        result = Tensor_new(out_dtype, 2, out_shape);
        if (!result) goto cleanup;
    }

    nk_size_t result_stride = n_vectors * bytes_per_dtype(out_dtype);
    {
        PyThreadState *save = PyEval_SaveThread();
        kernel(vec_buf.buf, n_vectors, depth, stride, result->data, result_stride, 0, n_vectors);
        PyEval_RestoreThread(save);
    }

    if (out_obj && out_obj != Py_None) {
        Py_INCREF(out_obj);
        return_obj = out_obj;
    }
    else { return_obj = (PyObject *)result; }

cleanup:
    PyBuffer_Release(&vec_buf);
    return return_obj;
}

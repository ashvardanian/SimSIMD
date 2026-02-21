/**
 *  @brief Tensor implementation for NumKong Python bindings.
 *  @file python/tensor.c
 *
 *  This file implements the Tensor N-dimensional array type with NumPy-like
 *  interface and the TensorIter iterator.
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
#include "matrix.h"

#include <float.h>
#include <stdint.h>

int buffers_shapes_match(Py_buffer const *first, Py_buffer const *second) {
    if (first->ndim != second->ndim) {
        PyErr_SetString(PyExc_ValueError, "Input tensor ranks don't match");
        return 0;
    }
    for (int dimension = 0; dimension < first->ndim; ++dimension) {
        if (first->shape[dimension] != second->shape[dimension]) {
            PyErr_Format(
                PyExc_ValueError,
                "Input tensor shapes don't match at dimension %d (%zd vs %zd). " "NumKong does not support " "implicit " "shape " "broadcasting.",
                dimension, first->shape[dimension], second->shape[dimension]);
            return 0;
        }
    }
    return 1;
}

size_t shared_contiguous_tail_dimensions(Py_buffer const *buffers[], size_t num_buffers, size_t num_dims) {
    size_t num_contiguous_dims = 0;
    for (size_t dimension = num_dims; dimension-- > 0;) {
        // Compute the expected stride for this dimension if it were packed
        int all_packed = 1;
        for (size_t buffer_idx = 0; buffer_idx < num_buffers; ++buffer_idx) {
            Py_ssize_t expected_stride = buffers[buffer_idx]->itemsize;
            for (size_t inner_dim = num_dims - 1; inner_dim > dimension; --inner_dim)
                expected_stride *= buffers[buffer_idx]->shape[inner_dim];
            if (buffers[buffer_idx]->strides[dimension] != expected_stride) {
                all_packed = 0;
                break;
            }
        }
        if (!all_packed) break;
        ++num_contiguous_dims;
    }
    return num_contiguous_dims;
}

void each_sum_recursive(                                           //
    nk_each_sum_punned_t kernel,                                   //
    char const *a_data, char const *b_data, char *result_data,     //
    Py_ssize_t const *shape, Py_ssize_t const *a_strides,          //
    Py_ssize_t const *b_strides, Py_ssize_t const *result_strides, //
    size_t remaining_dims, size_t contiguous_tail_dims) {

    // Base case: all remaining dimensions are contiguous — one kernel call
    if (remaining_dims <= contiguous_tail_dims) {
        size_t contiguous_elements = 1;
        for (size_t dimension = 0; dimension < remaining_dims; ++dimension)
            contiguous_elements *= (size_t)shape[dimension];
        kernel(a_data, b_data, contiguous_elements, result_data);
        return;
    }

    // Iterate over the outermost non-contiguous dimension, then recurse
    size_t const dim_extent = (size_t)shape[0];
    for (size_t position = 0; position < dim_extent; ++position) {
        each_sum_recursive(kernel,                                     //
                           a_data + position * a_strides[0],           //
                           b_data + position * b_strides[0],           //
                           result_data + position * result_strides[0], //
                           shape + 1, a_strides + 1,                   //
                           b_strides + 1, result_strides + 1,          //
                           remaining_dims - 1, contiguous_tail_dims);
    }
}

void each_scale_recursive(                                           //
    nk_each_scale_punned_t kernel,                                   //
    char const *a_data, char *result_data,                           //
    nk_scalar_buffer_t const *alpha, nk_scalar_buffer_t const *beta, //
    Py_ssize_t const *shape, Py_ssize_t const *a_strides,            //
    Py_ssize_t const *result_strides,                                //
    size_t remaining_dims, size_t contiguous_tail_dims) {

    if (remaining_dims <= contiguous_tail_dims) {
        size_t contiguous_elements = 1;
        for (size_t dimension = 0; dimension < remaining_dims; ++dimension)
            contiguous_elements *= (size_t)shape[dimension];
        kernel(a_data, contiguous_elements, alpha, beta, result_data);
        return;
    }

    size_t const dim_extent = (size_t)shape[0];
    for (size_t position = 0; position < dim_extent; ++position) {
        each_scale_recursive(kernel,                                     //
                             a_data + position * a_strides[0],           //
                             result_data + position * result_strides[0], //
                             alpha, beta,                                //
                             shape + 1, a_strides + 1,                   //
                             result_strides + 1,                         //
                             remaining_dims - 1, contiguous_tail_dims);
    }
}

void each_fma_recursive(                                                           //
    nk_each_fma_punned_t kernel,                                                   //
    char const *a_data, char const *b_data, char const *c_data, char *result_data, //
    nk_scalar_buffer_t const *alpha, nk_scalar_buffer_t const *beta,               //
    Py_ssize_t const *shape, Py_ssize_t const *a_strides,                          //
    Py_ssize_t const *b_strides, Py_ssize_t const *c_strides,                      //
    Py_ssize_t const *result_strides,                                              //
    size_t remaining_dims, size_t contiguous_tail_dims) {

    if (remaining_dims <= contiguous_tail_dims) {
        size_t contiguous_elements = 1;
        for (size_t dimension = 0; dimension < remaining_dims; ++dimension)
            contiguous_elements *= (size_t)shape[dimension];
        kernel(a_data, b_data, c_data, contiguous_elements, alpha, beta, result_data);
        return;
    }

    size_t const dim_extent = (size_t)shape[0];
    for (size_t position = 0; position < dim_extent; ++position) {
        each_fma_recursive(kernel,                                     //
                           a_data + position * a_strides[0],           //
                           b_data + position * b_strides[0],           //
                           c_data + position * c_strides[0],           //
                           result_data + position * result_strides[0], //
                           alpha, beta,                                //
                           shape + 1, a_strides + 1,                   //
                           b_strides + 1, c_strides + 1,               //
                           result_strides + 1,                         //
                           remaining_dims - 1, contiguous_tail_dims);
    }
}

void each_blend_recursive(                                           //
    nk_each_blend_punned_t kernel,                                   //
    char const *a_data, char const *b_data, char *result_data,       //
    nk_scalar_buffer_t const *alpha, nk_scalar_buffer_t const *beta, //
    Py_ssize_t const *shape, Py_ssize_t const *a_strides,            //
    Py_ssize_t const *b_strides, Py_ssize_t const *result_strides,   //
    size_t remaining_dims, size_t contiguous_tail_dims) {

    if (remaining_dims <= contiguous_tail_dims) {
        size_t contiguous_elements = 1;
        for (size_t dimension = 0; dimension < remaining_dims; ++dimension)
            contiguous_elements *= (size_t)shape[dimension];
        kernel(a_data, b_data, contiguous_elements, alpha, beta, result_data);
        return;
    }

    size_t const dim_extent = (size_t)shape[0];
    for (size_t position = 0; position < dim_extent; ++position) {
        each_blend_recursive(kernel,                                     //
                             a_data + position * a_strides[0],           //
                             b_data + position * b_strides[0],           //
                             result_data + position * result_strides[0], //
                             alpha, beta,                                //
                             shape + 1, a_strides + 1,                   //
                             b_strides + 1, result_strides + 1,          //
                             remaining_dims - 1, contiguous_tail_dims);
    }
}

static int tensor_is_c_contig(Tensor *tensor, size_t item_size);
static int tensor_is_f_contig(Tensor *tensor, size_t item_size);

/** @brief Return a Python scalar from a tensor byte offset using scalar_to_py_number. */
static PyObject *tensor_read_scalar(Tensor *tensor, size_t byte_offset) {
    size_t elem_size = bytes_per_dtype(tensor->dtype);
    if (!elem_size) {
        PyErr_SetString(PyExc_TypeError, "unsupported dtype for indexing");
        return NULL;
    }
    nk_scalar_buffer_t buf;
    memset(&buf, 0, sizeof(buf));
    memcpy(&buf, tensor->data + byte_offset, elem_size);
    return scalar_to_py_number(&buf, tensor->dtype);
}

/** @brief Check if a tensor is C-contiguous (row-major). */
static int tensor_is_c_contig(Tensor *tensor, size_t item_size) {
    if (tensor->rank == 0) return 1;
    Py_ssize_t expected = (Py_ssize_t)item_size;
    for (size_t i = tensor->rank; i > 0; i--) {
        if (tensor->strides[i - 1] != expected) return 0;
        expected *= tensor->shape[i - 1];
    }
    return 1;
}

/** @brief Check if a tensor is Fortran-contiguous (column-major). */
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
    for (size_t i = 0; i < rank; i++) {
        if (shape[i] > 0 && total_items > SIZE_MAX / (size_t)shape[i]) {
            PyErr_SetString(PyExc_OverflowError, "Tensor shape too large");
            return NULL;
        }
        total_items *= (size_t)shape[i];
    }
    if (item_size > 0 && total_items > SIZE_MAX / item_size) {
        PyErr_SetString(PyExc_OverflowError, "Tensor allocation too large");
        return NULL;
    }
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

/** @brief Create a 0D scalar tensor. */
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

/** @brief Convert a 0D Tensor to a Python float. */
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

/** @brief Convert a 0D Tensor to a Python int. */
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

/** @brief Compute C-contiguous strides for a tensor shape. */
void compute_contiguous_strides(size_t rank, Py_ssize_t const *shape, size_t item_size, Py_ssize_t *strides_out) {
    if (rank == 0) return;
    strides_out[rank - 1] = (Py_ssize_t)item_size;
    for (size_t d = rank - 1; d > 0; --d) strides_out[d - 1] = strides_out[d] * shape[d];
}

/**
 *  @brief Recursive stride walker for linearize_cast_into.
 *
 *  Walks non-contiguous outer dimensions recursively. Once only contiguous tail
 *  dimensions remain, processes the entire contiguous slice with memcpy or nk_cast.
 */
static void linearize_cast_recursive(                                         //
    char const *src_data, nk_dtype_t src_dtype, char *dest_data,              //
    nk_dtype_t dest_dtype, size_t src_element_size, size_t dest_element_size, //
    Py_ssize_t const *shape, Py_ssize_t const *strides,                       //
    size_t remaining_dims, size_t contiguous_tail_dims) {

    // Base case: all remaining dimensions are contiguous — one operation
    if (remaining_dims <= contiguous_tail_dims) {
        size_t slice_elements = 1;
        for (size_t dim = 0; dim < remaining_dims; ++dim) slice_elements *= (size_t)shape[dim];
        if (src_dtype == dest_dtype) memcpy(dest_data, src_data, slice_elements * src_element_size);
        else nk_cast(src_data, src_dtype, (nk_size_t)slice_elements, dest_data, dest_dtype);
        return;
    }

    // Recursive case: iterate outermost non-contiguous dimension
    size_t const dim_extent = (size_t)shape[0];
    // Compute the contiguous dest stride for this level
    size_t inner_elements = 1;
    for (size_t dim = 1; dim < remaining_dims; ++dim) inner_elements *= (size_t)shape[dim];
    size_t const dest_row_bytes = inner_elements * dest_element_size;

    for (size_t position = 0; position < dim_extent; ++position) {
        linearize_cast_recursive(                              //
            src_data + position * strides[0], src_dtype,       //
            dest_data + position * dest_row_bytes, dest_dtype, //
            src_element_size, dest_element_size,               //
            shape + 1, strides + 1,                            //
            remaining_dims - 1, contiguous_tail_dims);
    }
}

void linearize_cast_into(char const *src_data, nk_dtype_t src_dtype, char *dest_data, nk_dtype_t dest_dtype,
                         size_t rank, Py_ssize_t const *shape, Py_ssize_t const *strides, size_t total_elements) {
    (void)total_elements;
    size_t src_element_size = bytes_per_dtype(src_dtype);
    size_t dest_element_size = bytes_per_dtype(dest_dtype);

    // Count how many trailing dims are contiguous in src
    size_t contiguous_tail_dims = 0;
    Py_ssize_t expected_stride = (Py_ssize_t)src_element_size;
    for (size_t dim = rank; dim-- > 0;) {
        if (strides[dim] != expected_stride) break;
        expected_stride *= shape[dim];
        contiguous_tail_dims++;
    }

    linearize_cast_recursive(src_data, src_dtype, dest_data, dest_dtype, src_element_size, dest_element_size, shape,
                             strides, rank, contiguous_tail_dims);
}

char *ensure_contiguous_buffer(char const *src_data, nk_dtype_t src_dtype, nk_dtype_t target_dtype, size_t rank,
                               Py_ssize_t const *shape, Py_ssize_t const *strides, size_t total_elements,
                               int *needs_free) {
    size_t src_element_size = bytes_per_dtype(src_dtype);
    size_t dest_element_size = bytes_per_dtype(target_dtype);

    // Check full contiguity
    int is_contiguous = 1;
    Py_ssize_t expected_stride = (Py_ssize_t)src_element_size;
    for (size_t dim = rank; dim-- > 0;) {
        if (strides[dim] != expected_stride) {
            is_contiguous = 0;
            break;
        }
        expected_stride *= shape[dim];
    }

    // Zero-copy: contiguous + same dtype
    if (is_contiguous && src_dtype == target_dtype) {
        *needs_free = 0;
        return (char *)src_data;
    }

    // Single allocation, delegate
    char *output = PyMem_Malloc(total_elements * dest_element_size);
    if (!output) {
        PyErr_NoMemory();
        return NULL;
    }
    linearize_cast_into(src_data, src_dtype, output, target_dtype, rank, shape, strides, total_elements);
    *needs_free = 1;
    return output;
}

/** @brief Shared helper for tensor-scalar elementwise operations via scale kernel.
 *  Computes: result = alpha * a + beta */
static PyObject *tensor_elementwise_scalar(Tensor *a, double alpha_value, double beta_value) {
    nk_each_scale_punned_t kernel = NULL;
    nk_capability_t cap = nk_cap_serial_k;
    nk_find_kernel_punned(nk_kernel_each_scale_k, a->dtype, static_capabilities, nk_cap_any_k,
                          (nk_kernel_punned_t *)&kernel, &cap);
    if (!kernel || !cap) {
        PyErr_Format(PyExc_NotImplementedError, "scale not supported for dtype '%s'", dtype_to_python_string(a->dtype));
        return NULL;
    }

    Tensor *r = Tensor_new(a->dtype, a->rank, a->shape);
    if (!r) return NULL;

    size_t item_size = bytes_per_dtype(a->dtype);
    Py_ssize_t r_strides[NK_TENSOR_MAX_RANK];
    compute_contiguous_strides(a->rank, a->shape, item_size, r_strides);

    Py_buffer a_buf = {.ndim = (int)a->rank,
                       .itemsize = (Py_ssize_t)item_size,
                       .shape = a->shape,
                       .strides = a->strides};
    Py_buffer const *bufs[] = {&a_buf};
    size_t contiguous_tail = shared_contiguous_tail_dimensions(bufs, 1, a->rank);

    nk_scalar_buffer_t alpha_buf, beta_buf;
    nk_dtype_t scalar_dtype = nk_each_scale_input_dtype(a->dtype);
    nk_scalar_buffer_set_f64(&alpha_buf, alpha_value, scalar_dtype);
    nk_scalar_buffer_set_f64(&beta_buf, beta_value, scalar_dtype);
    PyThreadState *save = PyEval_SaveThread();
    each_scale_recursive(kernel, a->data, r->data, &alpha_buf, &beta_buf, a->shape, a->strides, r_strides, a->rank,
                         contiguous_tail);
    PyEval_RestoreThread(save);
    return (PyObject *)r;
}

static PyObject *Tensor_negative(PyObject *self) { return tensor_elementwise_scalar((Tensor *)self, -1.0, 0.0); }

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

        nk_each_sum_punned_t kernel = NULL;
        nk_capability_t cap = nk_cap_serial_k;
        nk_find_kernel_punned(nk_kernel_each_sum_k, a->dtype, static_capabilities, nk_cap_any_k,
                              (nk_kernel_punned_t *)&kernel, &cap);
        if (!kernel || !cap) {
            PyErr_Format(PyExc_NotImplementedError, "add not supported for dtype '%s'",
                         dtype_to_python_string(a->dtype));
            return NULL;
        }

        Tensor *r = Tensor_new(a->dtype, a->rank, a->shape);
        if (!r) return NULL;

        size_t item_size = bytes_per_dtype(a->dtype);
        Py_ssize_t r_strides[NK_TENSOR_MAX_RANK];
        compute_contiguous_strides(a->rank, a->shape, item_size, r_strides);

        Py_buffer a_buf = {.ndim = (int)a->rank,
                           .itemsize = (Py_ssize_t)item_size,
                           .shape = a->shape,
                           .strides = a->strides};
        Py_buffer b_buf = {.ndim = (int)b->rank,
                           .itemsize = (Py_ssize_t)item_size,
                           .shape = b->shape,
                           .strides = b->strides};
        Py_buffer const *bufs[] = {&a_buf, &b_buf};
        size_t contiguous_tail = shared_contiguous_tail_dimensions(bufs, 2, a->rank);

        PyThreadState *save = PyEval_SaveThread();
        each_sum_recursive(kernel, a->data, b->data, r->data, a->shape, a->strides, b->strides, r_strides, a->rank,
                           contiguous_tail);
        PyEval_RestoreThread(save);
        return (PyObject *)r;
    }

    if (PyFloat_Check(other) || PyLong_Check(other)) {
        double sc = PyFloat_Check(other) ? PyFloat_AsDouble(other) : (double)PyLong_AsLong(other);
        return tensor_elementwise_scalar(a, 1.0, sc);
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

        // Single-pass subtract via blend: result = 1*a + (-1)*b
        nk_each_blend_punned_t kernel = NULL;
        nk_capability_t cap = nk_cap_serial_k;
        nk_find_kernel_punned(nk_kernel_each_blend_k, a->dtype, static_capabilities, nk_cap_any_k,
                              (nk_kernel_punned_t *)&kernel, &cap);
        if (!kernel || !cap) {
            PyErr_Format(PyExc_NotImplementedError, "subtract not supported for dtype '%s'",
                         dtype_to_python_string(a->dtype));
            return NULL;
        }

        Tensor *r = Tensor_new(a->dtype, a->rank, a->shape);
        if (!r) return NULL;

        size_t item_size = bytes_per_dtype(a->dtype);
        Py_ssize_t r_strides[NK_TENSOR_MAX_RANK];
        compute_contiguous_strides(a->rank, a->shape, item_size, r_strides);

        Py_buffer a_buf = {.ndim = (int)a->rank,
                           .itemsize = (Py_ssize_t)item_size,
                           .shape = a->shape,
                           .strides = a->strides};
        Py_buffer b_buf = {.ndim = (int)b->rank,
                           .itemsize = (Py_ssize_t)item_size,
                           .shape = b->shape,
                           .strides = b->strides};
        Py_buffer const *bufs[] = {&a_buf, &b_buf};
        size_t contiguous_tail = shared_contiguous_tail_dimensions(bufs, 2, a->rank);

        nk_scalar_buffer_t alpha_buf, beta_buf;
        nk_dtype_t scalar_dtype = nk_each_scale_input_dtype(a->dtype);
        nk_scalar_buffer_set_f64(&alpha_buf, 1.0, scalar_dtype);
        nk_scalar_buffer_set_f64(&beta_buf, -1.0, scalar_dtype);
        PyThreadState *save = PyEval_SaveThread();
        each_blend_recursive(kernel, a->data, b->data, r->data, &alpha_buf, &beta_buf, a->shape, a->strides, b->strides,
                             r_strides, a->rank, contiguous_tail);
        PyEval_RestoreThread(save);
        return (PyObject *)r;
    }

    if (PyFloat_Check(other) || PyLong_Check(other)) {
        double sc = PyFloat_Check(other) ? PyFloat_AsDouble(other) : (double)PyLong_AsLong(other);
        return tensor_elementwise_scalar(a, 1.0, -sc);
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

        nk_each_fma_punned_t kernel = NULL;
        nk_capability_t cap = nk_cap_serial_k;
        nk_find_kernel_punned(nk_kernel_each_fma_k, a->dtype, static_capabilities, nk_cap_any_k,
                              (nk_kernel_punned_t *)&kernel, &cap);
        if (!kernel || !cap) {
            PyErr_Format(PyExc_NotImplementedError, "multiply not supported for dtype '%s'",
                         dtype_to_python_string(a->dtype));
            return NULL;
        }

        Tensor *r = Tensor_new(a->dtype, a->rank, a->shape);
        if (!r) return NULL;

        size_t item_size = bytes_per_dtype(a->dtype);
        size_t total_items = 1;
        for (size_t i = 0; i < a->rank; i++) total_items *= (size_t)a->shape[i];
        memset(r->data, 0, total_items * item_size); // prevent 0*NaN=NaN from uninitialized memory

        Py_ssize_t r_strides[NK_TENSOR_MAX_RANK];
        compute_contiguous_strides(a->rank, a->shape, item_size, r_strides);

        Py_buffer a_buf = {.ndim = (int)a->rank,
                           .itemsize = (Py_ssize_t)item_size,
                           .shape = a->shape,
                           .strides = a->strides};
        Py_buffer b_buf = {.ndim = (int)b->rank,
                           .itemsize = (Py_ssize_t)item_size,
                           .shape = b->shape,
                           .strides = b->strides};
        Py_buffer const *bufs[] = {&a_buf, &b_buf};
        size_t contiguous_tail = shared_contiguous_tail_dimensions(bufs, 2, a->rank);

        // fma(a, b, dummy, n, alpha=1, beta=0) -> 1*a*b + 0*dummy
        nk_scalar_buffer_t alpha_buf, beta_buf;
        nk_dtype_t scalar_dtype = nk_each_scale_input_dtype(a->dtype);
        nk_scalar_buffer_set_f64(&alpha_buf, 1.0, scalar_dtype);
        nk_scalar_buffer_set_f64(&beta_buf, 0.0, scalar_dtype);
        PyThreadState *save = PyEval_SaveThread();
        each_fma_recursive(kernel, a->data, b->data, r->data, r->data, &alpha_buf, &beta_buf, a->shape, a->strides,
                           b->strides, r_strides, r_strides, a->rank, contiguous_tail);
        PyEval_RestoreThread(save);
        return (PyObject *)r;
    }

    if (PyFloat_Check(other) || PyLong_Check(other)) {
        double sc = PyFloat_Check(other) ? PyFloat_AsDouble(other) : (double)PyLong_AsLong(other);
        return tensor_elementwise_scalar(a, sc, 0.0);
    }

    Py_RETURN_NOTIMPLEMENTED;
}

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

    PyObject *version = PyLong_FromLong(3);
    if (!version) {
        Py_DECREF(dict);
        return NULL;
    }
    PyDict_SetItemString(dict, "version", version);
    Py_DECREF(version);

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

    size_t total_elements = 1;
    for (size_t i = 0; i < tensor->rank; i++) total_elements *= (size_t)tensor->shape[i];

    Tensor *result = Tensor_new(tensor->dtype, tensor->rank, tensor->shape);
    if (!result) return NULL;

    linearize_cast_into(tensor->data, tensor->dtype, result->data, tensor->dtype, tensor->rank, tensor->shape,
                        tensor->strides, total_elements);
    return (PyObject *)result;
}

PyObject *Tensor_reshape(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    Tensor *tensor = (Tensor *)self;

    Py_ssize_t new_shape[NK_TENSOR_MAX_RANK];
    size_t new_rank = 0;

    if (nargs == 1 && PyTuple_Check(args[0])) {
        PyObject *shape_tuple = args[0];
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
        new_rank = (size_t)nargs;
        if (new_rank > NK_TENSOR_MAX_RANK) {
            PyErr_Format(PyExc_ValueError, "reshape: too many dimensions (%zu > %d)", new_rank, NK_TENSOR_MAX_RANK);
            return NULL;
        }
        for (size_t i = 0; i < new_rank; i++) {
            PyObject *item = args[i];
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
    Tensor *result = Tensor_new(tensor->dtype, new_rank, new_shape);
    if (!result) return NULL;

    linearize_cast_into(tensor->data, tensor->dtype, result->data, tensor->dtype, tensor->rank, tensor->shape,
                        tensor->strides, (size_t)old_total);
    return (PyObject *)result;
}

/** @brief Add a partial sum value into a running accumulator, using the accumulator dtype. */
static void accum_add(void *accum, void const *partial, nk_dtype_t accum_dtype) {
    switch (accum_dtype) {
    case nk_f64_k: *(nk_f64_t *)accum += *(nk_f64_t const *)partial; break;
    case nk_f32_k: *(nk_f32_t *)accum += *(nk_f32_t const *)partial; break;
    case nk_i64_k: nk_i64_sadd_((nk_i64_t const *)accum, (nk_i64_t const *)partial, (nk_i64_t *)accum); break;
    case nk_u64_k: nk_u64_sadd_((nk_u64_t const *)accum, (nk_u64_t const *)partial, (nk_u64_t *)accum); break;
    default: break;
    }
}

/** @brief Recursively reduce moments over an N-D tensor using a SIMD kernel. */
static void reduce_moments_recursive(                   //
    nk_kernel_reduce_moments_punned_t kernel,           //
    nk_dtype_t sum_dtype, nk_dtype_t sumsq_dtype,       //
    char const *data, Py_ssize_t const *shape,          //
    Py_ssize_t const *strides, size_t rank, size_t dim, //
    nk_scalar_buffer_t *sum_accum, nk_scalar_buffer_t *sumsq_accum) {

    if (dim == rank - 1) {
        // Base case: call SIMD kernel on innermost dimension
        nk_scalar_buffer_t partial_sum, partial_sumsq;
        memset(&partial_sum, 0, sizeof(partial_sum));
        memset(&partial_sumsq, 0, sizeof(partial_sumsq));
        kernel(data, (nk_size_t)shape[dim], (nk_size_t)strides[dim], &partial_sum, &partial_sumsq);
        accum_add(sum_accum, &partial_sum, sum_dtype);
        accum_add(sumsq_accum, &partial_sumsq, sumsq_dtype);
    }
    else {
        size_t const n = (size_t)shape[dim];
        Py_ssize_t const stride = strides[dim];
        for (size_t i = 0; i < n; i++)
            reduce_moments_recursive(kernel, sum_dtype, sumsq_dtype, data + i * stride, shape, strides, rank, dim + 1,
                                     sum_accum, sumsq_accum);
    }
}

/** @brief Reduce moments over an N-D tensor, returning typed scalar buffers. */
static int impl_reduce_moments(TensorView const *view, nk_scalar_buffer_t *sum_out, nk_dtype_t *sum_dtype_out,
                               nk_scalar_buffer_t *sumsq_out, nk_dtype_t *sumsq_dtype_out) {

    nk_kernel_reduce_moments_punned_t kernel = NULL;
    nk_capability_t cap = nk_cap_serial_k;
    nk_find_kernel_punned(nk_kernel_reduce_moments_k, view->dtype, static_capabilities, nk_cap_any_k,
                          (nk_kernel_punned_t *)&kernel, &cap);
    if (!kernel) return -1;

    nk_dtype_t sum_dtype = nk_reduce_moments_sum_dtype(view->dtype);
    nk_dtype_t sumsq_dtype = nk_reduce_moments_sumsq_dtype(view->dtype);

    nk_scalar_buffer_t sum_buf, sumsq_buf;
    memset(&sum_buf, 0, sizeof(sum_buf));
    memset(&sumsq_buf, 0, sizeof(sumsq_buf));

    if (view->rank == 0) {
        // Rank-0: single element, count=1, stride doesn't matter
        kernel(view->data, 1, 0, &sum_buf, &sumsq_buf);
    }
    else {
        reduce_moments_recursive(kernel, sum_dtype, sumsq_dtype, view->data, view->shape, view->strides, view->rank, 0,
                                 &sum_buf, &sumsq_buf);
    }

    *sum_out = sum_buf;
    *sumsq_out = sumsq_buf;
    *sum_dtype_out = sum_dtype;
    *sumsq_dtype_out = sumsq_dtype;
    return 0;
}

/** @brief Type-aware less-than comparison for scalar buffers.
 *  Uses native comparisons for standard types and sign-magnitude comparators for mini-floats. */
static int minmax_less_than(nk_scalar_buffer_t const *a, nk_scalar_buffer_t const *b, nk_dtype_t dtype) {
    switch (dtype) {
    case nk_f64_k: return a->f64 < b->f64;
    case nk_f32_k: return a->f32 < b->f32;
    case nk_i64_k: return a->i64 < b->i64;
    case nk_u64_k: return a->u64 < b->u64;
    case nk_i32_k: return a->i32 < b->i32;
    case nk_u32_k: return a->u32 < b->u32;
    case nk_i16_k: return a->i16 < b->i16;
    case nk_u16_k: return a->u16 < b->u16;
    case nk_i8_k: return a->i8 < b->i8;
    case nk_u8_k: return a->u8 < b->u8;
    case nk_f16_k: return nk_f16_compare_(a->f16, b->f16) < 0;
    case nk_bf16_k: return nk_bf16_compare_(a->bf16, b->bf16) < 0;
    case nk_e4m3_k: return nk_e4m3_compare_(a->u8, b->u8) < 0;
    case nk_e5m2_k: return nk_e5m2_compare_(a->u8, b->u8) < 0;
    case nk_e2m3_k: return nk_e2m3_compare_(a->u8, b->u8) < 0;
    case nk_e3m2_k: return nk_e3m2_compare_(a->u8, b->u8) < 0;
    default: return 0;
    }
}

/** @brief Update minmax accumulators with partial results from a SIMD kernel call.
 *  The SIMD kernel returns min/max values in the value dtype and indices relative to the slice.
 *  We compare against the running accumulators and update if better. */
static void minmax_update(nk_scalar_buffer_t *running_min, nk_size_t *running_min_idx, nk_scalar_buffer_t *running_max,
                          nk_size_t *running_max_idx, nk_scalar_buffer_t const *partial_min, nk_size_t partial_min_idx,
                          nk_scalar_buffer_t const *partial_max, nk_size_t partial_max_idx, nk_dtype_t value_dtype,
                          size_t flat_offset) {
    if (minmax_less_than(partial_min, running_min, value_dtype)) {
        memcpy(running_min, partial_min, sizeof(*running_min));
        *running_min_idx = flat_offset + partial_min_idx;
    }
    if (minmax_less_than(running_max, partial_max, value_dtype)) {
        memcpy(running_max, partial_max, sizeof(*running_max));
        *running_max_idx = flat_offset + partial_max_idx;
    }
}

/** @brief Recursively reduce minmax over an N-D tensor using a SIMD kernel. */
static void reduce_minmax_recursive(                         //
    nk_kernel_reduce_minmax_punned_t kernel,                 //
    nk_dtype_t value_dtype,                                  //
    char const *data, Py_ssize_t const *shape,               //
    Py_ssize_t const *strides, size_t rank, size_t dim,      //
    size_t flat_offset,                                      //
    nk_scalar_buffer_t *min_accum, nk_size_t *min_idx_accum, //
    nk_scalar_buffer_t *max_accum, nk_size_t *max_idx_accum) {

    if (dim == rank - 1) {
        nk_scalar_buffer_t partial_min, partial_max;
        memset(&partial_min, 0, sizeof(partial_min));
        memset(&partial_max, 0, sizeof(partial_max));
        nk_size_t partial_min_idx = 0, partial_max_idx = 0;
        kernel(data, (nk_size_t)shape[dim], (nk_size_t)strides[dim], &partial_min, &partial_min_idx, &partial_max,
               &partial_max_idx);
        minmax_update(min_accum, min_idx_accum, max_accum, max_idx_accum, &partial_min, partial_min_idx, &partial_max,
                      partial_max_idx, value_dtype, flat_offset);
    }
    else {
        size_t const n = (size_t)shape[dim];
        Py_ssize_t const stride = strides[dim];
        size_t inner_size = 1;
        for (size_t d = dim + 1; d < rank; d++) inner_size *= (size_t)shape[d];
        for (size_t i = 0; i < n; i++)
            reduce_minmax_recursive(kernel, value_dtype, data + i * stride, shape, strides, rank, dim + 1,
                                    flat_offset + i * inner_size, min_accum, min_idx_accum, max_accum, max_idx_accum);
    }
}

/** @brief Reduce minmax over an N-D tensor, returning typed scalar buffers. */
static int impl_reduce_minmax(TensorView const *view, nk_scalar_buffer_t *min_out, nk_dtype_t *min_dtype_out,
                              size_t *min_index_out, nk_scalar_buffer_t *max_out, nk_dtype_t *max_dtype_out,
                              size_t *max_index_out) {

    nk_kernel_reduce_minmax_punned_t kernel = NULL;
    nk_capability_t cap = nk_cap_serial_k;
    nk_find_kernel_punned(nk_kernel_reduce_minmax_k, view->dtype, static_capabilities, nk_cap_any_k,
                          (nk_kernel_punned_t *)&kernel, &cap);
    if (!kernel) return -1;

    nk_dtype_t value_dtype = nk_reduce_minmax_value_dtype(view->dtype);

    // For minmax, the SIMD kernel already initializes from the data it processes.
    // For multi-dimensional tensors, we need extreme initial values for the running accumulators.
    // Use the first element as initialization by calling the kernel on the first element,
    // then let the full recursive traversal find the real min/max.
    nk_scalar_buffer_t min_buf, max_buf;
    memset(&min_buf, 0, sizeof(min_buf));
    memset(&max_buf, 0, sizeof(max_buf));
    nk_size_t min_idx = 0, max_idx = 0;

    if (view->rank == 0) { kernel(view->data, 1, 0, &min_buf, &min_idx, &max_buf, &max_idx); }
    else {
        // Initialize accumulators from the first element
        size_t elem_size = bytes_per_dtype(value_dtype);
        kernel(view->data, 1, (nk_size_t)elem_size, &min_buf, &min_idx, &max_buf, &max_idx);
        // Now do the full traversal (first element will be compared again, which is fine)
        reduce_minmax_recursive(kernel, value_dtype, view->data, view->shape, view->strides, view->rank, 0, 0, &min_buf,
                                &min_idx, &max_buf, &max_idx);
    }

    *min_out = min_buf;
    *max_out = max_buf;
    *min_dtype_out = value_dtype;
    *max_dtype_out = value_dtype;
    *min_index_out = (size_t)min_idx;
    *max_index_out = (size_t)max_idx;
    return 0;
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

    nk_scalar_buffer_t sum_buf, sumsq_buf;
    nk_dtype_t sum_dtype, sumsq_dtype;
    if (impl_reduce_moments(&view, &sum_buf, &sum_dtype, &sumsq_buf, &sumsq_dtype) < 0) {
        PyErr_Format(PyExc_NotImplementedError, "moments not supported for dtype '%s'",
                     dtype_to_python_string(tensor->dtype));
        return NULL;
    }

    PyObject *sum_obj = scalar_to_py_number(&sum_buf, sum_dtype);
    if (!sum_obj) return NULL;
    PyObject *sumsq_obj = scalar_to_py_number(&sumsq_buf, sumsq_dtype);
    if (!sumsq_obj) {
        Py_DECREF(sum_obj);
        return NULL;
    }
    PyObject *tuple = PyTuple_Pack(2, sum_obj, sumsq_obj);
    Py_DECREF(sum_obj);
    Py_DECREF(sumsq_obj);
    return tuple;
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

    nk_scalar_buffer_t min_buf, max_buf;
    nk_dtype_t min_dtype, max_dtype;
    size_t min_index = 0, max_index = 0;
    if (impl_reduce_minmax(&view, &min_buf, &min_dtype, &min_index, &max_buf, &max_dtype, &max_index) < 0) {
        PyErr_Format(PyExc_NotImplementedError, "minmax not supported for dtype '%s'",
                     dtype_to_python_string(tensor->dtype));
        return NULL;
    }

    PyObject *min_obj = scalar_to_py_number(&min_buf, min_dtype);
    if (!min_obj) return NULL;
    PyObject *min_idx_obj = PyLong_FromSsize_t((Py_ssize_t)min_index);
    if (!min_idx_obj) {
        Py_DECREF(min_obj);
        return NULL;
    }
    PyObject *max_obj = scalar_to_py_number(&max_buf, max_dtype);
    if (!max_obj) {
        Py_DECREF(min_obj);
        Py_DECREF(min_idx_obj);
        return NULL;
    }
    PyObject *max_idx_obj = PyLong_FromSsize_t((Py_ssize_t)max_index);
    if (!max_idx_obj) {
        Py_DECREF(min_obj);
        Py_DECREF(min_idx_obj);
        Py_DECREF(max_obj);
        return NULL;
    }
    PyObject *tuple = PyTuple_Pack(4, min_obj, min_idx_obj, max_obj, max_idx_obj);
    Py_DECREF(min_obj);
    Py_DECREF(min_idx_obj);
    Py_DECREF(max_obj);
    Py_DECREF(max_idx_obj);
    return tuple;
}

PyObject *Tensor_astype(PyObject *self, PyObject *dtype_arg) {
    Tensor *tensor = (Tensor *)self;

    // Parse target dtype from string argument
    char const *dtype_str = PyUnicode_AsUTF8(dtype_arg);
    if (!dtype_str) {
        PyErr_SetString(PyExc_TypeError, "dtype must be a string");
        return NULL;
    }
    nk_dtype_t target_dtype = python_string_to_dtype(dtype_str);
    if (target_dtype == nk_dtype_unknown_k) {
        PyErr_Format(PyExc_ValueError, "Unsupported dtype: '%s'", dtype_str);
        return NULL;
    }

    // Same dtype -> return copy
    if (target_dtype == tensor->dtype) return Tensor_copy(self, NULL);

    // Compute total elements
    Py_ssize_t total = 1;
    for (size_t i = 0; i < tensor->rank; i++) total *= tensor->shape[i];

    // Allocate result tensor
    Tensor *result = Tensor_new(target_dtype, tensor->rank, tensor->shape);
    if (!result) return NULL;

    linearize_cast_into(tensor->data, tensor->dtype, result->data, target_dtype, tensor->rank, tensor->shape,
                        tensor->strides, (size_t)total);
    return (PyObject *)result;
}

static PyObject *Tensor___array__(PyObject *self_obj, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    PyObject *dtype_arg = NULL;

    Py_ssize_t nkw = kwnames ? PyTuple_Size(kwnames) : 0;
    if (nargs > 1 || nargs + nkw > 2) {
        PyErr_SetString(PyExc_TypeError, "__array__(dtype=None, copy=None)");
        return NULL;
    }
    if (nargs >= 1) dtype_arg = args[0];
    for (Py_ssize_t i = 0; i < nkw; i++) {
        PyObject *name = PyTuple_GET_ITEM(kwnames, i);
        PyObject *value = args[nargs + i];
        if (PyUnicode_CompareWithASCIIString(name, "dtype") == 0) dtype_arg = value;
        else if (PyUnicode_CompareWithASCIIString(name, "copy") == 0) { /* ignored */ }
        else {
            PyErr_Format(PyExc_TypeError, "__array__() unexpected keyword: %S", name);
            return NULL;
        }
    }

    Tensor *self = (Tensor *)self_obj;
    nk_dtype_info_t const *info = dtype_info(self->dtype);
    if (!info) {
        PyErr_SetString(PyExc_TypeError, "Unknown dtype");
        return NULL;
    }
    // Reject exotic dtypes that NumPy can't represent natively
    char const *fmt = info->buffer_format;
    if (same_string(fmt, "e2m3") || same_string(fmt, "e3m2") ||       //
        same_string(fmt, "e4m3") || same_string(fmt, "e5m2") ||       //
        same_string(fmt, "bf16") || same_string(fmt, "bcomplex32") || //
        same_string(fmt, "Ze") || same_string(fmt, "i4") ||           //
        same_string(fmt, "u4") || same_string(fmt, "?")) {
        PyErr_Format(PyExc_TypeError,
                     "Cannot convert NumKong tensor of dtype '%s' to NumPy array. " "Use .astype('float32') first.",
                     info->name);
        return NULL;
    }

    // Cache numpy.asarray to avoid repeated import overhead
    static PyObject *cached_asarray = NULL;
    if (!cached_asarray) {
        PyObject *numpy = PyImport_ImportModule("numpy");
        if (!numpy) return NULL;
        cached_asarray = PyObject_GetAttrString(numpy, "asarray");
        Py_DECREF(numpy);
        if (!cached_asarray) return NULL;
    }

    PyObject *result;
    if (dtype_arg && dtype_arg != Py_None)
        result = PyObject_CallFunctionObjArgs(cached_asarray, self_obj, dtype_arg, NULL);
    else result = PyObject_CallOneArg(cached_asarray, self_obj);
    return result;
}

static PyMethodDef Tensor_methods[] = {
    {"copy", Tensor_copy, METH_NOARGS, "Return a deep copy of the tensor"},
    {"reshape", (PyCFunction)Tensor_reshape, METH_FASTCALL, "Return tensor reshaped to given dimensions"},
    {"moments", Tensor_moments, METH_NOARGS, "Returns (sum, sum_of_squares) tuple"},
    {"minmax", Tensor_minmax, METH_NOARGS, "Returns (min_val, min_idx, max_val, max_idx) tuple"},
    {"astype", Tensor_astype, METH_O, "Cast tensor to a different dtype. Returns a new tensor."},
    {"__array__", (PyCFunction)Tensor___array__, METH_FASTCALL | METH_KEYWORDS,
     "Convert to NumPy array. Raises TypeError for exotic dtypes."},
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
        view->ndim = tensor->rank;
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

char const doc_empty[] =                                       //
    "Create an uninitialized Tensor with the given shape.\n\n" //
    "Parameters:\n"                                            //
    "    shape: Shape of the array.\n"                         //
    "    dtype: Data type (default 'float32').\n\n"            //
    "Returns:\n"                                               //
    "    Tensor: Uninitialized array.";

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

char const doc_zeros[] =                            //
    "Create a Tensor filled with zeros.\n\n"        //
    "Parameters:\n"                                 //
    "    shape: Shape of the array.\n"              //
    "    dtype: Data type (default 'float32').\n\n" //
    "Returns:\n"                                    //
    "    Tensor: Array of zeros.";

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

char const doc_ones[] =                             //
    "Create a Tensor filled with ones.\n\n"         //
    "Parameters:\n"                                 //
    "    shape: Shape of the array.\n"              //
    "    dtype: Data type (default 'float32').\n\n" //
    "Returns:\n"                                    //
    "    Tensor: Array of ones.";

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

    {
        size_t elem_size = bytes_per_dtype(dtype);
        nk_scalar_buffer_t one;
        memset(&one, 0, sizeof(one));
        nk_scalar_buffer_set_f64(&one, 1.0, dtype);
        for (size_t i = 0; i < total; i++) memcpy(result->data + i * elem_size, &one, elem_size);
    }

    return (PyObject *)result;
}

char const doc_full[] =                               //
    "Create a Tensor filled with a given value.\n\n"  //
    "Parameters:\n"                                   //
    "    shape: Shape of the array.\n"                //
    "    fill_value: Value to fill the array with.\n" //
    "    dtype: Data type (default 'float32').\n\n"   //
    "Returns:\n"                                      //
    "    Tensor: Array filled with fill_value.";

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

    {
        size_t elem_size = bytes_per_dtype(dtype);
        nk_scalar_buffer_t val;
        memset(&val, 0, sizeof(val));
        nk_scalar_buffer_set_f64(&val, fill_value, dtype);
        for (size_t i = 0; i < total; i++) memcpy(result->data + i * elem_size, &val, elem_size);
    }

    return (PyObject *)result;
}

char const doc_reduce_moments[] =                                               //
    "Compute sum and sum-of-squares (moments) of all elements in an array.\n\n" //
    "Parameters:\n"                                                             //
    "    a: Input array.\n\n"                                                   //
    "Returns:\n"                                                                //
    "    tuple: (sum, sum_of_squares) for all elements.";

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

char const doc_reduce_minmax[] =                                            //
    "Find minimum and maximum elements with their indices in an array.\n\n" //
    "Parameters:\n"                                                         //
    "    a: Input array.\n\n"                                               //
    "Returns:\n"                                                            //
    "    tuple: (min_val, min_index, max_val, max_index).";

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

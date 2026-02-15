/**
 *  @brief NumKong Python Bindings.
 *  @file python/numkong.h
 *  @author Ash Vardanian
 *  @date December 30, 2025
 *
 *  This header provides common data types, buffer protocol helpers, and
 *  dtype conversion utilities used across the NumKong Python extension modules.
 */
#ifndef NK_PYTHON_NUMKONG_H
#define NK_PYTHON_NUMKONG_H

#include <math.h>
#include <string.h>

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <numkong/numkong.h>

#ifdef __cplusplus
extern "C" {
#endif

#pragma region Tensor Argument Descriptor

/**
 *  @brief Parsed tensor argument from Python buffer protocol.
 *
 *  This structure holds the essential information extracted from a Python
 *  object that supports the buffer protocol (NumPy arrays, PyTorch tensors, etc.)
 */
typedef struct TensorArgument {
    char *start;       ///< Pointer to the first element
    size_t dimensions; ///< Vector size (1D) or column count (2D)
    size_t count;      ///< Number of vectors (1 for 1D, num rows for 2D)
    size_t stride;     ///< Stride between rows in bytes (0 for 1D)
    int rank;          ///< Number of dimensions (1 or 2)
    nk_dtype_t dtype;  ///< Logical dtype
} TensorArgument;

/**
 *  @brief Lightweight view for stride-aware operations.
 *
 *  Used by impl_reduce_* functions to traverse N-dimensional tensors
 *  with arbitrary strides.
 */
typedef struct TensorView {
    nk_dtype_t dtype;          ///< Logical dtype
    size_t rank;               ///< Number of dimensions
    Py_ssize_t const *shape;   ///< Shape array (borrowed pointer)
    Py_ssize_t const *strides; ///< Strides array in bytes (borrowed pointer)
    char *data;                ///< Data pointer
} TensorView;

#pragma endregion // Tensor Argument Descriptor

#pragma region Datatype Metadata

/**  @brief Metadata for a single dtype.  */
typedef struct {
    nk_dtype_t dtype;          ///< Logical dtype enum value
    char const *name;          ///< Human-readable name (e.g., "float32")
    char const *buffer_format; ///< Python buffer protocol format string
    char const *array_typestr; ///< NumPy array interface typestr
    size_t item_size;          ///< Size in bytes per element
    int is_complex;            ///< 1 if complex type, 0 otherwise
} nk_dtype_info_t;

/// Global dtype metadata table.
extern nk_dtype_info_t const nk_dtype_table[];
extern size_t const nk_dtype_table_size;

#pragma endregion // Datatype Metadata

#pragma region Datatype Utilities

/**
 *  @brief Look up metadata for a dtype.
 *  @param[in] dtype Logical dtype.
 *  @return Pointer to metadata, or NULL if not found.
 */
nk_dtype_info_t const *dtype_info(nk_dtype_t dtype);

/**
 *  @brief Get byte size of a dtype.
 *  @param[in] dtype Logical dtype.
 *  @return Size in bytes, or 0 if unknown.
 */
size_t bytes_per_dtype(nk_dtype_t dtype);

/**
 *  @brief Get human-readable name of a dtype.
 *  @param[in] dtype Logical dtype.
 *  @return Name string (e.g., "float32"), or "unknown" if not found.
 */
char const *dtype_to_string(nk_dtype_t dtype);

/**
 *  @brief Get NumPy array interface typestr for a dtype.
 *  @param[in] dtype Logical dtype.
 *  @return Typestr (e.g., "<f4"), or "|V1" if not found.
 */
char const *dtype_to_array_typestr(nk_dtype_t dtype);

/**
 *  @brief Get Python buffer protocol format string for a dtype.
 *  @param[in] dtype Logical dtype.
 *  @return Format string (e.g., "f"), or "unknown" if not found.
 */
char const *dtype_to_python_string(nk_dtype_t dtype);

/**
 *  @brief Convert Python-style dtype string to logical dtype.
 *
 *  Handles NumPy format strings, Python struct format characters, and
 *  NumKong-specific names like "bfloat16" and "e4m3".
 *
 *  @param[in] name Format string from buffer protocol or user input.
 *  @return Logical dtype, or nk_dtype_unknown_k if not recognized.
 *  @see https://docs.python.org/3/library/struct.html#format-characters
 *  @see https://numpy.org/doc/stable/reference/arrays.interface.html
 */
nk_dtype_t python_string_to_dtype(char const *name);

/**
 *  @brief Convert metric name string to kernel kind.
 *  @param[in] name Metric name (e.g., "l2", "dot", "angular").
 *  @return Kernel kind, or nk_kernel_unknown_k if not recognized.
 */
nk_kernel_kind_t python_string_to_metric_kind(char const *name);

/**
 *  @brief Check if a dtype is complex.
 *  @param[in] dtype Logical dtype.
 *  @return 1 if complex, 0 otherwise.
 */
int is_complex(nk_dtype_t dtype);

/**
 *  @brief Check string equality.
 *  @param[in] a First string.
 *  @param[in] b Second string.
 *  @return 1 if equal, 0 otherwise.
 */
int same_string(char const *a, char const *b);

/**
 *  @brief Cast a distance value to target dtype and store it.
 *  @param[in] distance The value to store.
 *  @param[in] target_dtype Target dtype.
 *  @param[out] target_ptr Pointer to output buffer.
 *  @param[in] offset Element offset in output buffer.
 *  @return 1 on success, 0 if dtype not supported.
 */
int cast_distance(nk_fmax_t distance, nk_dtype_t target_dtype, void *target_ptr, size_t offset);

/**
 *  @brief Check if a metric is commutative.
 *  @param[in] kind Kernel kind.
 *  @return 1 if commutative, 0 otherwise.
 */
int kernel_is_commutative(nk_kernel_kind_t kind);

#pragma endregion // Datatype Utilities

#pragma region Buffer Protocol Helpers

/**
 *  @brief Parse a Python tensor object into TensorArgument.
 *
 *  Extracts buffer information from any Python object supporting the buffer
 *  protocol. Validates that the tensor is 1D or 2D with contiguous rows.
 *
 *  @param[in] tensor Python object supporting buffer protocol.
 *  @param[out] buffer Output Py_buffer (caller must release with PyBuffer_Release).
 *  @param[out] parsed Output TensorArgument with extracted metadata.
 *  @return 1 on success, 0 on failure (Python exception set).
 */
int parse_tensor(PyObject *tensor, Py_buffer *buffer, TensorArgument *parsed);

/**
 *  @brief Check if a Python object is a numeric scalar.
 *  @param[in] obj Python object.
 *  @return 1 if int or float, 0 otherwise.
 */
int is_scalar(PyObject *obj);

/**
 *  @brief Extract double value from a Python numeric scalar.
 *  @param[in] obj Python int or float.
 *  @param[out] value Output double value.
 *  @return 1 on success, 0 on failure.
 */
int get_scalar_value(PyObject *obj, double *value);

#pragma endregion // Buffer Protocol Helpers

/// CPU capabilities detected at module init time.
extern nk_capability_t static_capabilities;

#ifdef __cplusplus
}
#endif

#endif // NK_PYTHON_NUMKONG_H

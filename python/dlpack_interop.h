/**
 *  @brief DLPack zero-copy interop for NumKong's Python extension.
 *  @file python/dlpack_interop.h
 *  @author Ash Vardanian
 *  @date April 17, 2026
 *
 *  Implements the Python Array API DLPack protocol on `numkong.Tensor` so it exchanges zero-copy with PyTorch,
 *  NumPy, JAX, CuPy, TensorFlow, PyArrow, ONNX Runtime, MXNet, MLX, NNabla, TVM, and any other consumer of
 *  `__dlpack__` / `from_dlpack`. The struct layout we exchange is declared in `python/dlpack_abi.h`.
 *
 *  See `python/dlpack_interop.c` (`@section Interop partners`) for the full list of upstream PRs that ratified
 *  the protocol in each of those projects.
 *
 *  https://github.com/dmlc/dlpack
 *  https://data-apis.org/array-api/latest/design_topics/data_interchange.html
 */
#ifndef NK_PYTHON_DLPACK_INTEROP_H
#define NK_PYTHON_DLPACK_INTEROP_H

#include "numkong.h"
#include "tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 *  @brief `Tensor.__dlpack__(stream=None, max_version=None, dl_device=None, copy=None)`.
 *
 *  Produces a `PyCapsule` named `"dltensor"` (legacy v0) or `"dltensor_versioned"` (v1+). Exporter
 *  reports `kDLCPU` only; consumers asking for any other `dl_device` get a `BufferError`.
 *  Verified consumers (zero-copy): PyTorch, NumPy, JAX, CuPy, TensorFlow, PyArrow, MLX.
 */
PyObject *Tensor_dlpack(PyObject *self, PyObject *args, PyObject *kwargs);

/**
 *  @brief `Tensor.__dlpack_device__()`. Always returns `(kDLCPU=1, 0)`.
 *
 *  Part of the Array API DLPack negotiation: consumers call this before
 *  `__dlpack__` to know whether a copy or stream sync is needed.
 */
PyObject *Tensor_dlpack_device(PyObject *self, PyObject *noargs);

/**
 *  @brief `numkong.from_dlpack(obj)` — zero-copy import.
 *
 *  Accepts a `"dltensor"` / `"dltensor_versioned"` capsule or any object implementing `__dlpack__`. Any
 *  device whose pointer is host-readable (kDLCPU, kDLCUDAHost, kDLROCMHost, kDLCUDAManaged, kDLOneAPI,
 *  kDLMetal) is accepted; pure device memory (kDLCUDA, kDLROCM, kDLOpenCL, kDLVulkan, kDLWebGPU, ...) is
 *  rejected with a clear `ValueError` naming the device code. Verified producers (zero-copy): PyTorch,
 *  NumPy, JAX, CuPy, TensorFlow, PyArrow, MLX, ONNX Runtime (training builds), MXNet.
 */
PyObject *api_from_dlpack(PyObject *self, PyObject *obj);

/**
 *  @brief Initialize the internal `_DLPackOwner` type used to hold imported
 *         capsule ownership. Called once from `PyInit__numkong`.
 */
int nk_dlpack_init(PyObject *module);

extern char const doc_from_dlpack[];
extern char const doc_dlpack[];
extern char const doc_dlpack_device[];

#ifdef __cplusplus
}
#endif

#endif // NK_PYTHON_DLPACK_INTEROP_H

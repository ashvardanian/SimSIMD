from typing import Any, Union, Literal, Optional, TypeAlias

# Many annotation features depend on the Python version:
# - `typing.TypeAlias` type aliases are supported in Python 3.10-3.11.
# - `type` statements are supported from Python 3.12, replacing `typing.TypeAlias`.
# - `typing.Literal` literal types are supported from Python 3.8.
#
# We cannot maintain a separate `.pyi` file for every Python version.
# Assume Python 3.11 with NumPy available.
from numpy.typing import NDArray

_BufferType: TypeAlias = Union[NDArray[Any], memoryview]

_MetricType = Literal[
    "euclidean",
    "sqeuclidean",
    "inner",
    "dot",
    "angular",
    "hamming",
    "jaccard",
    "kullbackleibler",
    "kld",
    "jensenshannon",
    "jsd",
    "intersection",
    "bilinear",
    "mahalanobis",
    "fma",
    "wsum",
]
_IntegralType = Literal[
    "bin8",
    # Signed integers
    "int8",
    "int16",
    "int32",
    "int64",
    # Unsigned integers
    "uint8",
    "uint16",
    "uint32",
    "uint64",
]
_FloatType = Literal[
    "f32",
    "float32",
    "f16",
    "float16",
    "f64",
    "float64",
    "bf16",  #! Not supported by NumPy
    "bfloat16",  #! Not supported by NumPy
    "e4m3",  #! FP8 E4M3 format
    "e5m2",  #! FP8 E5M2 format
]
_ComplexType = Literal[
    "complex32",  #! Not supported by NumPy
    "bcomplex32",  #! Not supported by NumPy
    "complex64",
    "complex128",
]

class DistancesTensor(memoryview):
    """A tensor type returned by SimSIMD distance functions.

    Supports NumPy-like properties and buffer protocol for interoperability.
    """

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the tensor as a tuple of dimensions."""
        ...

    @property
    def dtype(self) -> Union[_IntegralType, _FloatType, _ComplexType]:
        """Data type of the tensor elements (e.g., 'float64')."""
        ...

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        ...

    @property
    def size(self) -> int:
        """Total number of elements."""
        ...

    @property
    def nbytes(self) -> int:
        """Total number of bytes."""
        ...

    @property
    def strides(self) -> tuple[int, ...]:
        """Strides of the tensor in bytes."""
        ...

    @property
    def itemsize(self) -> int:
        """Size of each element in bytes."""
        ...

    def __len__(self) -> int:
        """Return the length of the first dimension."""
        ...

    def __iter__(self) -> Any:
        """Iterate over the first dimension."""
        ...

    def __getitem__(self, key: Union[int, slice, tuple[Union[int, slice], ...]]) -> Union[float, "DistancesTensor"]:
        """Get an element or sub-tensor by index or slice."""
        ...

    def __repr__(self) -> str:
        """Return a string representation."""
        ...

    def __float__(self) -> float:
        """Convert 0D tensor to Python float."""
        ...

    def __int__(self) -> int:
        """Convert 0D tensor to Python int."""
        ...

    def __str__(self) -> str:
        """Return a pretty-printed string representation."""
        ...

    def __eq__(self, other: Any) -> bool:
        """Compare tensors for equality."""
        ...

    def __ne__(self, other: Any) -> bool:
        """Compare tensors for inequality."""
        ...

    @property
    def __array_interface__(self) -> dict[str, Any]:
        """NumPy array interface dict for legacy interoperability."""
        ...

    @property
    def T(self) -> "DistancesTensor":
        """Transpose of the tensor."""
        ...

    def copy(self) -> "DistancesTensor":
        """Return a deep copy of the tensor."""
        ...

    def reshape(self, *shape: int) -> "DistancesTensor":
        """Return tensor reshaped to given dimensions."""
        ...

# region Capabilities

# SIMD capability controls.
def get_capabilities() -> dict[str, bool]: ...
def enable_capability(capability: str, /) -> None: ...
def disable_capability(capability: str, /) -> None: ...

# Kernel pointer accessors.
def pointer_to_euclidean(dtype: Union[_IntegralType, _FloatType], /) -> int: ...
def pointer_to_sqeuclidean(dtype: Union[_IntegralType, _FloatType], /) -> int: ...
def pointer_to_angular(dtype: Union[_IntegralType, _FloatType], /) -> int: ...
def pointer_to_inner(dtype: Union[_FloatType, _ComplexType], /) -> int: ...
def pointer_to_dot(dtype: Union[_FloatType, _ComplexType], /) -> int: ...
def pointer_to_vdot(dtype: Union[_FloatType, _ComplexType], /) -> int: ...
def pointer_to_hamming(dtype: _IntegralType, /) -> int: ...
def pointer_to_jaccard(dtype: _IntegralType, /) -> int: ...
def pointer_to_jensenshannon(dtype: _FloatType, /) -> int: ...
def pointer_to_kullbackleibler(dtype: _FloatType, /) -> int: ...

# endregion Capabilities

# region Pairwise Distances

# Pairwise distances, similar to `scipy.spatial.distance.cdist`.
# https://docs.scipy.org/doc/scipy-1.11.4/reference/generated/scipy.spatial.distance.cdist.html
def cdist(
    a: _BufferType,
    b: _BufferType,
    /,
    metric: _MetricType = "euclidean",
    *,
    threads: int = 1,
    dtype: Optional[Union[_IntegralType, _FloatType, _ComplexType]] = None,
    out: Optional[_BufferType] = None,
    out_dtype: Optional[Union[_FloatType, _ComplexType]] = None,
) -> Optional[Union[float, complex, DistancesTensor]]: ...

# endregion Pairwise Distances

# region Vector Dot Products
# Vector-vector dot products for real and complex numbers.

# Inner product, similar to: `numpy.inner`.
# https://numpy.org/doc/stable/reference/generated/numpy.inner.html
def inner(
    a: _BufferType,
    b: _BufferType,
    /,
    dtype: Optional[Union[_FloatType, _ComplexType]] = None,
    *,
    out: Optional[_BufferType] = None,
    out_dtype: Optional[Union[_FloatType, _ComplexType]] = None,
) -> Optional[Union[float, complex, DistancesTensor]]: ...

# Dot product, similar to: `numpy.dot`.
# https://numpy.org/doc/stable/reference/generated/numpy.dot.html
def dot(
    a: _BufferType,
    b: _BufferType,
    /,
    dtype: Optional[Union[_FloatType, _ComplexType]] = None,
    *,
    out: Optional[_BufferType] = None,
    out_dtype: Union[_FloatType, _ComplexType] = None,
) -> Optional[Union[float, complex, DistancesTensor]]: ...

# Vector-vector dot product for complex conjugates, similar to: `numpy.vdot`.
# https://numpy.org/doc/stable/reference/generated/numpy.vdot.html
def vdot(
    a: _BufferType,
    b: _BufferType,
    /,
    dtype: Optional[_ComplexType] = None,
    *,
    out: Optional[Union[float, complex, DistancesTensor]] = None,
    out_dtype: Optional[_ComplexType] = None,
) -> Optional[Union[complex, DistancesTensor]]: ...

# endregion Vector Dot Products

# region Spatial Distance Metrics
# Vector-vector spatial distance metrics for real and integer numbers.

# Vector-vector squared Euclidean distance, similar to: `scipy.spatial.distance.sqeuclidean`.
# https://docs.scipy.org/doc/scipy-1.11.4/reference/generated/scipy.spatial.distance.sqeuclidean.html
# https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html
def sqeuclidean(
    a: _BufferType,
    b: _BufferType,
    /,
    dtype: Optional[Union[_IntegralType, _FloatType]] = None,
    *,
    out: Optional[_BufferType] = None,
    out_dtype: Union[_FloatType] = None,
) -> Optional[Union[float, DistancesTensor]]: ...

# Vector-vector angular distance (also known as cosine distance), similar to: `scipy.spatial.distance.cosine`.
# https://docs.scipy.org/doc/scipy-1.11.4/reference/generated/scipy.spatial.distance.cosine.html
def angular(
    a: _BufferType,
    b: _BufferType,
    /,
    dtype: Optional[Union[_IntegralType, _FloatType]] = None,
    *,
    out: Optional[_BufferType] = None,
    out_dtype: Union[_FloatType] = None,
) -> Optional[Union[float, DistancesTensor]]: ...

# endregion Spatial Distance Metrics

# region Binary Similarity
# Vector-vector similarity functions for binary vectors.

# Vector-vector Hamming distance, similar to: `scipy.spatial.distance.hamming`.
# https://docs.scipy.org/doc/scipy-1.11.4/reference/generated/scipy.spatial.distance.hamming.html
def hamming(
    a: _BufferType,
    b: _BufferType,
    /,
    dtype: Optional[_IntegralType] = None,
    *,
    out: Optional[_BufferType] = None,
    out_dtype: Union[_FloatType] = None,
) -> Optional[Union[float, DistancesTensor]]: ...

# Vector-vector Jaccard distance, similar to: `scipy.spatial.distance.jaccard`.
# https://docs.scipy.org/doc/scipy-1.11.4/reference/generated/scipy.spatial.distance.jaccard.html
def jaccard(
    a: _BufferType,
    b: _BufferType,
    /,
    dtype: Optional[_IntegralType] = None,
    *,
    out: Optional[_BufferType] = None,
    out_dtype: Union[_FloatType] = None,
) -> Optional[Union[float, DistancesTensor]]: ...

# endregion Binary Similarity

# region Probability Distances
# Vector-vector similarity between probability distributions.

# Vector-vector Jensen-Shannon distance, similar to: `scipy.spatial.distance.jensenshannon`.
# https://docs.scipy.org/doc/scipy-1.11.4/reference/generated/scipy.spatial.distance.jensenshannon.html
def jensenshannon(
    a: _BufferType,
    b: _BufferType,
    /,
    dtype: Optional[_FloatType] = None,
    *,
    out: Optional[_BufferType] = None,
    out_dtype: Union[_FloatType] = None,
) -> Optional[Union[float, DistancesTensor]]: ...
def jsd(
    a: _BufferType,
    b: _BufferType,
    /,
    dtype: Optional[_FloatType] = None,
    *,
    out: Optional[_BufferType] = None,
    out_dtype: Union[_FloatType] = None,
) -> Optional[Union[float, DistancesTensor]]: ...

# Vector-vector Kullback-Leibler divergence, similar to: `scipy.spatial.distance.kullback_leibler`.
# https://docs.scipy.org/doc/scipy-1.11.4/reference/generated/scipy.spatial.distance.kullback_leibler.html
def kullbackleibler(
    a: _BufferType,
    b: _BufferType,
    /,
    dtype: Optional[_FloatType] = None,
    *,
    out: Optional[_BufferType] = None,
    out_dtype: Union[_FloatType] = None,
) -> Optional[Union[float, DistancesTensor]]: ...
def kld(
    a: _BufferType,
    b: _BufferType,
    /,
    dtype: Optional[_FloatType] = None,
    *,
    out: Optional[_BufferType] = None,
    out_dtype: Union[_FloatType] = None,
) -> Optional[Union[float, DistancesTensor]]: ...

# endregion Probability Distances

# region Curved Space Metrics
# Vector-vector similarity between vectors in curved spaces.

# Vector-vector bilinear distance, similar to: `numpy.dot(a, metric_tensor @ vector2)`.
# https://numpy.org/doc/stable/reference/generated/numpy.dot.html
def bilinear(
    a: _BufferType,
    b: _BufferType,
    metric_tensor: _BufferType,
    /,
    dtype: Optional[_FloatType] = None,
) -> float: ...

# Vector-vector Mahalanobis distance, similar to: `scipy.spatial.distance.mahalanobis`.
# https://docs.scipy.org/doc/scipy-1.11.4/reference/generated/scipy.spatial.distance.mahalanobis.html
def mahalanobis(
    a: _BufferType,
    b: _BufferType,
    inverse_covariance: _BufferType,
    /,
    dtype: Optional[_FloatType] = None,
) -> float: ...

# endregion Curved Space Metrics

# region Sparse Similarity
# Vector-vector similarity between sparse vectors.

# Vector-vector intersection similarity, similar to: `numpy.intersect1d`.
# https://numpy.org/doc/stable/reference/generated/numpy.intersect1d.html
def intersection(array1: _BufferType, array2: _BufferType, /) -> float: ...

# endregion Sparse Similarity

# region Vector Math
# Vector-vector math: fused multiply-add and weighted sum.

# Vector-vector element-wise fused multiply-add.
def fma(
    a: _BufferType,
    b: _BufferType,
    c: _BufferType,
    /,
    dtype: Optional[Union[_FloatType, _IntegralType]] = None,
    *,
    alpha: float = 1,
    beta: float = 1,
    out: Optional[_BufferType] = None,
) -> Optional[DistancesTensor]: ...

# Vector-vector element-wise weighted sum.
def wsum(
    a: _BufferType,
    b: _BufferType,
    /,
    dtype: Optional[Union[_FloatType, _IntegralType]] = None,
    *,
    alpha: float = 1,
    beta: float = 1,
    out: Optional[_BufferType] = None,
) -> Optional[DistancesTensor]: ...

# endregion Vector Math

# region Trigonometry
# Element-wise trigonometric functions.

# Element-wise trigonometric sine.
def sin(
    a: _BufferType,
    /,
    dtype: Optional[_FloatType] = None,
    *,
    out: Optional[_BufferType] = None,
) -> Optional[DistancesTensor]: ...

# Element-wise trigonometric cosine.
def cos(
    a: _BufferType,
    /,
    dtype: Optional[_FloatType] = None,
    *,
    out: Optional[_BufferType] = None,
) -> Optional[DistancesTensor]: ...

# Element-wise trigonometric arctangent.
def atan(
    a: _BufferType,
    /,
    dtype: Optional[_FloatType] = None,
    *,
    out: Optional[_BufferType] = None,
) -> Optional[DistancesTensor]: ...

# endregion Trigonometry

# region Elementwise Arithmetic
# Element-wise arithmetic operations.

# Element-wise scale operation.
def scale(
    a: _BufferType,
    /,
    dtype: Optional[Union[_FloatType, _IntegralType]] = None,
    *,
    alpha: float = 1,
    beta: float = 0,
    out: Optional[_BufferType] = None,
) -> Optional[DistancesTensor]: ...

# Element-wise sum (addition).
def sum(
    a: _BufferType,
    b: _BufferType,
    /,
    dtype: Optional[Union[_FloatType, _IntegralType]] = None,
    *,
    out: Optional[_BufferType] = None,
) -> Optional[DistancesTensor]: ...

# Element-wise add (NumPy-compatible with broadcasting).
def add(
    a: Union[_BufferType, float, int],
    b: Union[_BufferType, float, int],
    /,
    *,
    out: Optional[_BufferType] = None,
    a_dtype: Optional[Union[_FloatType, _IntegralType]] = None,
    b_dtype: Optional[Union[_FloatType, _IntegralType]] = None,
    out_dtype: Optional[Union[_FloatType, _IntegralType]] = None,
) -> Optional[DistancesTensor]: ...

# Element-wise multiply (NumPy-compatible with broadcasting).
def multiply(
    a: Union[_BufferType, float, int],
    b: Union[_BufferType, float, int],
    /,
    *,
    out: Optional[_BufferType] = None,
    a_dtype: Optional[Union[_FloatType, _IntegralType]] = None,
    b_dtype: Optional[Union[_FloatType, _IntegralType]] = None,
    out_dtype: Optional[Union[_FloatType, _IntegralType]] = None,
) -> Optional[DistancesTensor]: ...

# endregion Elementwise Arithmetic

# region Packed Matmul
# Packed matrix multiplication (AMX accelerated).

class PackedMatrix:
    """Opaque pre-packed matrix for repeated matrix multiplication.

    Created by pack_matmul_argument() and used with matmul().
    Requires AMX support (Sapphire Rapids or newer CPU).
    """

    @property
    def n(self) -> int:
        """Number of rows in the original matrix."""
        ...

    @property
    def k(self) -> int:
        """Number of columns in the original matrix."""
        ...

    @property
    def dtype(self) -> Union[_IntegralType, _FloatType, _ComplexType]:
        """Data type of the packed matrix (like 'bf16' or 'i8')."""
        ...

    @property
    def nbytes(self) -> int:
        """Size of the packed buffer in bytes."""
        ...

    def __repr__(self) -> str:
        """Return a string representation."""
        ...

# Pack a matrix for repeated matmul.
def pack_matmul_argument(
    b: _BufferType,
    /,
    dtype: Optional[Union[_IntegralType, _FloatType, _ComplexType]] = None,
) -> PackedMatrix: ...

# Matrix multiplication with pre-packed B matrix.
def matmul(
    a: _BufferType,
    b: PackedMatrix,
    /,
) -> DistancesTensor: ...

# endregion Packed Matmul

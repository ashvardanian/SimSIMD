from typing import Any, Literal, Optional, TypeAlias, Union

# Many annotation features depend on the Python version:
# - `typing.TypeAlias` type aliases are supported in Python 3.10-3.11.
# - `type` statements are supported from Python 3.12, replacing `typing.TypeAlias`.
# - `typing.Literal` literal types are supported from Python 3.8.
#
# We cannot maintain a separate `.pyi` file for every Python version.
# Assume Python 3.11 with NumPy available.
from numpy.typing import NDArray

# region Forward Declarations and Shared Types

# Scalar dtype literals used throughout the API.
_IntegralType = Literal[
    "uint1",
    # Sub-byte integers
    "int4",
    "uint4",
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
    "float8_e4m3",  #! FP8 E4M3 format (long-form)
    "e5m2",  #! FP8 E5M2 format
    "float8_e5m2",  #! FP8 E5M2 format (long-form)
    "e2m3",  #! FP6 E2M3 format
    "float6_e2m3",  #! FP6 E2M3 format (long-form)
    "e3m2",  #! FP6 E3M2 format
    "float6_e3m2",  #! FP6 E3M2 format (long-form)
]
_ComplexType = Literal[
    "complex32",  #! Not supported by NumPy
    "bcomplex32",  #! Not supported by NumPy
    "complex64",
    "complex128",
]
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
    "blend",
]

# Buffer-compatible tensor inputs accepted by most functions.
_BufferType: TypeAlias = Union[NDArray[Any], memoryview]

class Tensor(memoryview):
    """N-dimensional tensor type returned by NumKong operations.

    Supports NumPy-like properties and buffer protocol for interoperability.
    """

    def __new__(cls, array_like: _BufferType, /, *, dtype: Optional[str] = None) -> "Tensor":
        """Construct a Tensor by copying data from a buffer-protocol object."""
        ...

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

    def __getitem__(self, key: Union[int, slice, tuple[Union[int, slice], ...]]) -> Union[float, "Tensor"]:
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
    def T(self) -> "Tensor":
        """Transpose of the tensor."""
        ...

    def copy(self) -> "Tensor":
        """Return a deep copy of the tensor."""
        ...

    def reshape(self, *shape: int) -> "Tensor":
        """Return tensor reshaped to given dimensions."""
        ...

    def flatten(self) -> "Tensor":
        """Return a flattened 1D view (copies if non-contiguous)."""
        ...

    def squeeze(self, axis: Optional[int] = None) -> "Tensor":
        """Remove dimensions of size 1."""
        ...

    @property
    def is_contiguous(self) -> bool:
        """Whether the tensor is C-contiguous in memory."""
        ...

    def sum(
        self, axis: Optional[int] = None, *, keepdims: bool = False, out: Optional["Tensor"] = None
    ) -> Union[float, int, "Tensor"]:
        """Return the sum of all elements."""
        ...

    def norm(
        self, axis: Optional[int] = None, *, keepdims: bool = False, out: Optional["Tensor"] = None
    ) -> Union[float, "Tensor"]:
        """Return the L2 norm."""
        ...

    def min(
        self, axis: Optional[int] = None, *, keepdims: bool = False, out: Optional["Tensor"] = None
    ) -> Union[float, int, None, "Tensor"]:
        """Return the minimum element, or None if all elements are NaN."""
        ...

    def max(
        self, axis: Optional[int] = None, *, keepdims: bool = False, out: Optional["Tensor"] = None
    ) -> Union[float, int, None, "Tensor"]:
        """Return the maximum element, or None if all elements are NaN."""
        ...

    def argmin(self, axis: Optional[int] = None, *, out: Optional["Tensor"] = None) -> Union[int, None, "Tensor"]:
        """Return the index of the minimum element, or None if all elements are NaN."""
        ...

    def argmax(self, axis: Optional[int] = None, *, out: Optional["Tensor"] = None) -> Union[int, None, "Tensor"]:
        """Return the index of the maximum element, or None if all elements are NaN."""
        ...

class PackedMatrix:
    """Opaque pre-packed matrix for repeated cross operations.

    Created by dots_pack() or hammings_pack() and used with
    dots_packed(), hammings_packed(), jaccards_packed(), angulars_packed(),
    euclideans_packed(), or the @ operator.
    """

    @property
    def kind(self) -> str:
        """Kernel kind ('dots' or 'hammings')."""
        ...

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

    @classmethod
    def packed_size(
        cls,
        n: int,
        k: int,
        /,
        dtype: Union[_IntegralType, _FloatType, _ComplexType] = "bf16",
        kind: str = "dots",
    ) -> int:
        """Return packed buffer size in bytes for given dimensions and dtype."""
        ...

    def __repr__(self) -> str:
        """Return a string representation."""
        ...

# endregion Forward Declarations and Shared Types

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
) -> Optional[Union[float, complex, Tensor]]: ...

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
) -> Optional[Union[float, complex, Tensor]]: ...

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
) -> Optional[Union[float, complex, Tensor]]: ...

# Vector-vector dot product for complex conjugates, similar to: `numpy.vdot`.
# https://numpy.org/doc/stable/reference/generated/numpy.vdot.html
def vdot(
    a: _BufferType,
    b: _BufferType,
    /,
    dtype: Optional[_ComplexType] = None,
    *,
    out: Optional[Union[float, complex, Tensor]] = None,
    out_dtype: Optional[_ComplexType] = None,
) -> Optional[Union[complex, Tensor]]: ...

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
) -> Optional[Union[float, Tensor]]: ...

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
) -> Optional[Union[float, Tensor]]: ...

# Vector-vector Euclidean distance, similar to: `scipy.spatial.distance.euclidean`.
# https://docs.scipy.org/doc/scipy-1.11.4/reference/generated/scipy.spatial.distance.euclidean.html
def euclidean(
    a: _BufferType,
    b: _BufferType,
    /,
    dtype: Optional[Union[_IntegralType, _FloatType]] = None,
    *,
    out: Optional[_BufferType] = None,
    out_dtype: Optional[_FloatType] = None,
) -> Optional[Union[float, Tensor]]: ...

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
) -> Optional[Union[float, Tensor]]: ...

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
) -> Optional[Union[float, Tensor]]: ...

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
) -> Optional[Union[float, Tensor]]: ...
def jsd(
    a: _BufferType,
    b: _BufferType,
    /,
    dtype: Optional[_FloatType] = None,
    *,
    out: Optional[_BufferType] = None,
    out_dtype: Union[_FloatType] = None,
) -> Optional[Union[float, Tensor]]: ...

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
) -> Optional[Union[float, Tensor]]: ...
def kld(
    a: _BufferType,
    b: _BufferType,
    /,
    dtype: Optional[_FloatType] = None,
    *,
    out: Optional[_BufferType] = None,
    out_dtype: Union[_FloatType] = None,
) -> Optional[Union[float, Tensor]]: ...

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

# region Geospatial Distances

def haversine(
    a_lats: _BufferType,
    a_lons: _BufferType,
    b_lats: _BufferType,
    b_lons: _BufferType,
    /,
    dtype: Optional[_FloatType] = None,
    *,
    out: Optional[_BufferType] = None,
) -> Optional[Union[float, Tensor]]: ...
def vincenty(
    a_lats: _BufferType,
    a_lons: _BufferType,
    b_lats: _BufferType,
    b_lons: _BufferType,
    /,
    dtype: Optional[_FloatType] = None,
    *,
    out: Optional[_BufferType] = None,
) -> Optional[Union[float, Tensor]]: ...

# endregion Geospatial Distances

# region Sparse Similarity
# Vector-vector similarity between sparse vectors.

# Vector-vector intersection similarity, similar to: `numpy.intersect1d`.
# https://numpy.org/doc/stable/reference/generated/numpy.intersect1d.html
def intersect(array1: _BufferType, array2: _BufferType, /) -> float: ...
def sparse_dot(
    a_indices: _BufferType,
    a_values: _BufferType,
    b_indices: _BufferType,
    b_values: _BufferType,
    /,
) -> float: ...

# endregion Sparse Similarity

# region Tensor Constructors
def empty(
    shape: Union[int, tuple[int, ...]],
    /,
    *,
    dtype: Union[_IntegralType, _FloatType, _ComplexType] = "float32",
) -> Tensor: ...
def zeros(
    shape: Union[int, tuple[int, ...]],
    /,
    *,
    dtype: Union[_IntegralType, _FloatType, _ComplexType] = "float32",
) -> Tensor: ...
def ones(
    shape: Union[int, tuple[int, ...]],
    /,
    *,
    dtype: Union[_IntegralType, _FloatType, _ComplexType] = "float32",
) -> Tensor: ...
def full(
    shape: Union[int, tuple[int, ...]],
    fill_value: Union[int, float],
    /,
    *,
    dtype: Union[_IntegralType, _FloatType, _ComplexType] = "float32",
) -> Tensor: ...

# endregion Tensor Constructors

# region Reductions

def moments(a: _BufferType, /) -> tuple[float, float]: ...
def minmax(a: _BufferType, /) -> Optional[tuple[float, int, float, int]]: ...
def sum(
    a: _BufferType, /, axis: Optional[int] = None, *, keepdims: bool = False, out: Optional[Tensor] = None
) -> Union[float, int, Tensor]: ...
def norm(
    a: _BufferType, /, axis: Optional[int] = None, *, keepdims: bool = False, out: Optional[Tensor] = None
) -> Union[float, Tensor]: ...
def min(
    a: _BufferType, /, axis: Optional[int] = None, *, keepdims: bool = False, out: Optional[Tensor] = None
) -> Union[float, int, None, Tensor]: ...
def max(
    a: _BufferType, /, axis: Optional[int] = None, *, keepdims: bool = False, out: Optional[Tensor] = None
) -> Union[float, int, None, Tensor]: ...
def argmin(
    a: _BufferType, /, axis: Optional[int] = None, *, out: Optional[Tensor] = None
) -> Union[int, None, Tensor]: ...
def argmax(
    a: _BufferType, /, axis: Optional[int] = None, *, out: Optional[Tensor] = None
) -> Union[int, None, Tensor]: ...

# endregion Reductions

# region Vector Math

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
) -> Optional[Tensor]: ...

# Vector-vector element-wise blend.
def blend(
    a: _BufferType,
    b: _BufferType,
    /,
    dtype: Optional[Union[_FloatType, _IntegralType]] = None,
    *,
    alpha: float = 1,
    beta: float = 1,
    out: Optional[_BufferType] = None,
) -> Optional[Tensor]: ...

# endregion Vector Math

# region Trigonometry

# Element-wise trigonometric sine.
def sin(
    a: _BufferType,
    /,
    dtype: Optional[_FloatType] = None,
    *,
    out: Optional[_BufferType] = None,
) -> Optional[Tensor]: ...

# Element-wise trigonometric cosine.
def cos(
    a: _BufferType,
    /,
    dtype: Optional[_FloatType] = None,
    *,
    out: Optional[_BufferType] = None,
) -> Optional[Tensor]: ...

# Element-wise trigonometric arctangent.
def atan(
    a: _BufferType,
    /,
    dtype: Optional[_FloatType] = None,
    *,
    out: Optional[_BufferType] = None,
) -> Optional[Tensor]: ...

# endregion Trigonometry

# region Elementwise Arithmetic

# Element-wise scale operation.
def scale(
    a: _BufferType,
    /,
    dtype: Optional[Union[_FloatType, _IntegralType]] = None,
    *,
    alpha: float = 1,
    beta: float = 0,
    out: Optional[_BufferType] = None,
) -> Optional[Tensor]: ...

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
) -> Optional[Tensor]: ...

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
) -> Optional[Tensor]: ...

# endregion Elementwise Arithmetic

# region Symmetric Pairwise Operations
def dots_symmetric(
    vectors: _BufferType,
    /,
    *,
    dtype: Optional[Union[_FloatType, _IntegralType]] = None,
    out: Optional[_BufferType] = None,
    start_row: Optional[int] = None,
    end_row: Optional[int] = None,
) -> Tensor: ...
def hammings_symmetric(
    vectors: _BufferType,
    /,
    *,
    dtype: Optional[_IntegralType] = None,
    out: Optional[_BufferType] = None,
    start_row: Optional[int] = None,
    end_row: Optional[int] = None,
) -> Tensor: ...
def jaccards_symmetric(
    vectors: _BufferType,
    /,
    *,
    dtype: Optional[_IntegralType] = None,
    out: Optional[_BufferType] = None,
    start_row: Optional[int] = None,
    end_row: Optional[int] = None,
) -> Tensor: ...
def angulars_symmetric(
    vectors: _BufferType,
    /,
    *,
    dtype: Optional[Union[_FloatType, _IntegralType]] = None,
    out: Optional[_BufferType] = None,
    start_row: Optional[int] = None,
    end_row: Optional[int] = None,
) -> Tensor: ...
def euclideans_symmetric(
    vectors: _BufferType,
    /,
    *,
    dtype: Optional[Union[_FloatType, _IntegralType]] = None,
    out: Optional[_BufferType] = None,
    start_row: Optional[int] = None,
    end_row: Optional[int] = None,
) -> Tensor: ...

# endregion Symmetric Pairwise Operations

# region Packed Matrix Operations

# Pack a matrix for repeated dot-product matmul.
def dots_pack(
    b: _BufferType,
    /,
    dtype: Optional[Union[_IntegralType, _FloatType, _ComplexType]] = None,
) -> PackedMatrix: ...

# Dot-product matrix multiplication with a pre-packed B matrix.
def dots_packed(
    a: _BufferType,
    b: PackedMatrix,
    /,
    *,
    out: Optional[_BufferType] = None,
    start_row: Optional[int] = None,
    end_row: Optional[int] = None,
) -> Tensor: ...

# Pack a matrix for repeated Hamming distance computation.
def hammings_pack(
    b: _BufferType,
    /,
    dtype: Optional[Union[_IntegralType, _FloatType, _ComplexType]] = None,
) -> PackedMatrix: ...

# Hamming distance computation with a pre-packed B matrix.
def hammings_packed(
    a: _BufferType,
    b: PackedMatrix,
    /,
    *,
    out: Optional[_BufferType] = None,
    start_row: Optional[int] = None,
    end_row: Optional[int] = None,
) -> Tensor: ...

# Jaccard distance computation with a pre-packed B matrix.
def jaccards_packed(
    a: _BufferType,
    b: PackedMatrix,
    /,
    *,
    out: Optional[_BufferType] = None,
    start_row: Optional[int] = None,
    end_row: Optional[int] = None,
) -> Tensor: ...

# Angular distance computation with a pre-packed B matrix.
def angulars_packed(
    a: _BufferType,
    b: PackedMatrix,
    /,
    *,
    out: Optional[_BufferType] = None,
    start_row: Optional[int] = None,
    end_row: Optional[int] = None,
) -> Tensor: ...

# Euclidean distance computation with a pre-packed B matrix.
def euclideans_packed(
    a: _BufferType,
    b: PackedMatrix,
    /,
    *,
    out: Optional[_BufferType] = None,
    start_row: Optional[int] = None,
    end_row: Optional[int] = None,
) -> Tensor: ...

# endregion Packed Matrix Operations

# region Mesh Alignment

class MeshAlignmentResult:
    """Result of a mesh alignment operation (Kabsch, Umeyama, or RMSD)."""

    @property
    def rotation(self) -> Tensor:
        """Rotation matrix."""
        ...

    @property
    def translation(self) -> Tensor:
        """Translation vector."""
        ...

    @property
    def scale(self) -> Tensor:
        """Scale factor."""
        ...

    @property
    def rmsd(self) -> Tensor:
        """Root mean square deviation."""
        ...

def kabsch(
    source: _BufferType,
    target: _BufferType,
    /,
    dtype: Optional[_FloatType] = None,
) -> MeshAlignmentResult: ...
def umeyama(
    source: _BufferType,
    target: _BufferType,
    /,
    dtype: Optional[_FloatType] = None,
) -> MeshAlignmentResult: ...
def rmsd(
    source: _BufferType,
    target: _BufferType,
    /,
    dtype: Optional[_FloatType] = None,
) -> float: ...

# endregion Mesh Alignment

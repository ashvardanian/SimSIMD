from typing import Any, Union, Literal

from numpy.typing import NDArray
from typing_extensions import overload

_MetricType = Literal[
    "sqeuclidean", "inner", "dot", "cosine", "cos", "hamming", "jaccard", "kullbackleibler", "kl", "jensenshannon", "js"
]
_FloatType = Literal[
    "f",
    "f32",
    "float32",
    "h",
    "f16",
    "float16",
    "c",
    "i8",
    "int8",
    "b",
    "b8",
    "d",
    "f64",
    "float64",
    "bh",
    "bf16",
    "bfloat16",
]
_ComplexType = Literal["complex32", "bcomplex32", "complex64", "complex128"]

@overload
def get_capabilities() -> dict[str, bool]: ...
@overload
def enable_capability(capability: str, /) -> None: ...
@overload
def disable_capability(capability: str, /) -> None: ...
@overload
def cdist(
    tensor1: NDArray[Any], tensor2: NDArray[Any], /, metric: _MetricType = "sqeuclidean", threads: int = 1
) -> Union[float, complex, DistancesTensor]: ...
@overload
def sqeuclidean(tensor1: NDArray[Any], tensor2: NDArray[Any], /) -> Union[float, complex, DistancesTensor]: ...
@overload
def sqeuclidean(
    tensor1: NDArray[Any], tensor2: NDArray[Any], datatype: _FloatType, /
) -> Union[float, DistancesTensor]: ...
@overload
def sqeuclidean(
    tensor1: NDArray[Any], tensor2: NDArray[Any], datatype: _ComplexType, /
) -> Union[complex, DistancesTensor]: ...
@overload
def cosine(tensor1: NDArray[Any], tensor2: NDArray[Any], /) -> Union[float, complex, DistancesTensor]: ...
@overload
def cosine(
    tensor1: NDArray[Any],
    tensor2: NDArray[Any],
    datatype: _FloatType,
) -> Union[float, DistancesTensor]: ...
@overload
def cosine(
    tensor1: NDArray[Any], tensor2: NDArray[Any], datatype: _ComplexType, /
) -> Union[complex, DistancesTensor]: ...
@overload
def inner(tensor1: NDArray[Any], tensor2: NDArray[Any], /) -> Union[float, complex, DistancesTensor]: ...
@overload
def inner(tensor1: NDArray[Any], tensor2: NDArray[Any], datatype: _FloatType, /) -> Union[float, DistancesTensor]: ...
@overload
def inner(
    tensor1: NDArray[Any], tensor2: NDArray[Any], datatype: _ComplexType, /
) -> Union[complex, DistancesTensor]: ...
@overload
def dot(tensor1: NDArray[Any], tensor2: NDArray[Any], /) -> Union[float, complex, DistancesTensor]: ...
@overload
def dot(tensor1: NDArray[Any], tensor2: NDArray[Any], datatype: _FloatType, /) -> Union[float, DistancesTensor]: ...
@overload
def dot(tensor1: NDArray[Any], tensor2: NDArray[Any], datatype: _ComplexType, /) -> Union[complex, DistancesTensor]: ...
@overload
def vdot(tensor1: NDArray[Any], tensor2: NDArray[Any], /) -> Union[float, complex, DistancesTensor]: ...
@overload
def vdot(tensor1: NDArray[Any], tensor2: NDArray[Any], datatype: _FloatType, /) -> Union[float, DistancesTensor]: ...
@overload
def vdot(
    tensor1: NDArray[Any], tensor2: NDArray[Any], datatype: _ComplexType, /
) -> Union[complex, DistancesTensor]: ...
@overload
def hamming(tensor1: NDArray[Any], tensor2: NDArray[Any], /) -> Union[float, complex, DistancesTensor]: ...
@overload
def hamming(tensor1: NDArray[Any], tensor2: NDArray[Any], datatype: _FloatType, /) -> Union[float, DistancesTensor]: ...
@overload
def hamming(
    tensor1: NDArray[Any], tensor2: NDArray[Any], datatype: _ComplexType, /
) -> Union[complex, DistancesTensor]: ...
@overload
def jaccard(tensor1: NDArray[Any], tensor2: NDArray[Any], /) -> Union[float, complex, DistancesTensor]: ...
@overload
def jaccard(tensor1: NDArray[Any], tensor2: NDArray[Any], datatype: _FloatType, /) -> Union[float, DistancesTensor]: ...
@overload
def jaccard(
    tensor1: NDArray[Any], tensor2: NDArray[Any], datatype: _ComplexType, /
) -> Union[complex, DistancesTensor]: ...
@overload
def jensenshannon(tensor1: NDArray[Any], tensor2: NDArray[Any], /) -> Union[float, complex, DistancesTensor]: ...
@overload
def jensenshannon(
    tensor1: NDArray[Any], tensor2: NDArray[Any], datatype: _FloatType, /
) -> Union[float, DistancesTensor]: ...
@overload
def jensenshannon(
    tensor1: NDArray[Any], tensor2: NDArray[Any], datatype: _ComplexType, /
) -> Union[complex, DistancesTensor]: ...
@overload
def kullbackleibler(tensor1: NDArray[Any], tensor2: NDArray[Any], /) -> Union[float, complex, DistancesTensor]: ...
@overload
def kullbackleibler(
    tensor1: NDArray[Any], tensor2: NDArray[Any], datatype: _FloatType, /
) -> Union[float, DistancesTensor]: ...
@overload
def kullbackleibler(
    tensor1: NDArray[Any], tensor2: NDArray[Any], datatype: _ComplexType, /
) -> Union[complex, DistancesTensor]: ...
@overload
def pointer_to_sqeuclidean(type_name: Union[_FloatType, _ComplexType], /) -> int: ...
@overload
def pointer_to_cosine(type_name: Union[_FloatType, _ComplexType], /) -> int: ...
@overload
def pointer_to_inner(type_name: Union[_FloatType, _ComplexType], /) -> int: ...
@overload
def pointer_to_jensenshannon(type_name: Union[_FloatType, _ComplexType], /) -> int: ...
@overload
def pointer_to_kullbackleibler(type_name: Union[_FloatType, _ComplexType], /) -> int: ...

class DistancesTensor: ...

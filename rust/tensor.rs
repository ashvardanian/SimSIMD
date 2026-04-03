//! Core N-dimensional tensor types with elementwise, trigonometric, reduction, and cast operations.
//!
//! This module provides:
//!
//! - [`Tensor`]: N-dimensional array with customizable rank and allocator
//! - [`TensorView`]: Immutable view into a tensor
//! - [`TensorSpan`]: Mutable view into a tensor
//! - [`Matrix`]: Type alias for 2D tensors
//!
//! Batch matrix operations (GEMM, packed spatial distances) live in [`crate::matrix`].

extern crate alloc;

use alloc::vec::Vec;
use core::marker::PhantomData;
use core::ops::{Index, IndexMut};
use core::ptr::NonNull;

use crate::cast::{cast, CastDtype};
use crate::each::{EachATan, EachBlend, EachCos, EachFMA, EachScale, EachSin, EachSum};
use crate::reduce::{ReduceMinMax, ReduceMoments};
use crate::spatial::{Dot, Roots};
use crate::types::{DimMut, DimRef, FloatConvertible, NumberLike, StorageElement};
use crate::vector::{Vector, VectorIndex};

// region: Constants and Allocator

/// Default maximum rank for tensors.
pub const DEFAULT_MAX_RANK: usize = 8;

/// Alignment for SIMD-friendly allocations (64 bytes for AVX-512).
pub const SIMD_ALIGNMENT: usize = 64;

/// Memory allocator trait for custom allocation strategies.
///
/// Implement this trait to use custom allocators (arena, pool, etc.) with
/// [`Tensor`] and [`PackedMatrix`].
///
/// # Safety
///
/// Implementations must ensure:
/// - `allocate` returns a valid, properly aligned pointer on success
/// - `deallocate` is called with the same `Layout` used in `allocate`
/// - The returned memory is not aliased
pub unsafe trait Allocator {
    /// Allocates memory with the given layout.
    ///
    /// Returns `None` if allocation fails.
    fn allocate(&self, layout: alloc::alloc::Layout) -> Option<NonNull<u8>>;

    /// Deallocates memory previously allocated with `allocate`.
    ///
    /// # Safety
    ///
    /// - `ptr` must have been returned by a previous call to `allocate`
    /// - `layout` must be the same as the one used for allocation
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: alloc::alloc::Layout);
}

/// Global allocator using the system allocator.
#[derive(Debug, Clone, Copy, Default)]
pub struct Global;

unsafe impl Allocator for Global {
    #[inline]
    fn allocate(&self, layout: alloc::alloc::Layout) -> Option<NonNull<u8>> {
        if layout.size() == 0 {
            return Some(NonNull::dangling());
        }
        unsafe {
            let ptr = alloc::alloc::alloc(layout);
            NonNull::new(ptr)
        }
    }

    #[inline]
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: alloc::alloc::Layout) {
        if layout.size() > 0 {
            alloc::alloc::dealloc(ptr.as_ptr(), layout);
        }
    }
}

// endregion: Constants and Allocator

// region: Error Types

/// Error type for Tensor operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TensorError {
    /// Memory allocation failed.
    AllocationFailed,
    /// Shape mismatch: axis `axis` has size `expected` on one side and `got` on the other.
    ShapeMismatch {
        axis: usize,
        expected: usize,
        got: usize,
    },
    /// Invalid shape specification.
    InvalidShape {
        axis: usize,
        size: usize,
        reason: &'static str,
    },
    /// Operation requires contiguous rows but array has non-contiguous rows.
    NonContiguousRows,
    /// Expected a specific number of dimensions.
    DimensionMismatch { expected: usize, got: usize },
    /// Index out of bounds.
    IndexOutOfBounds { index: usize, size: usize },
    /// Too many dimensions (exceeds MAX_RANK).
    TooManyRanks { got: usize },
    /// Operation not supported for sub-byte types (i4x2, u4x2, u1x8).
    SubByteUnsupported,
}

#[cfg(feature = "std")]
impl std::error::Error for TensorError {}

impl core::fmt::Display for TensorError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            TensorError::AllocationFailed => write!(f, "memory allocation failed"),
            TensorError::ShapeMismatch {
                axis,
                expected,
                got,
            } => {
                write!(f, "shape mismatch on axis {axis}: expected {expected}, got {got}")
            }
            TensorError::InvalidShape {
                axis,
                size,
                reason,
            } => {
                write!(f, "invalid shape: axis {axis} has size {size}: {reason}")
            }
            TensorError::NonContiguousRows => {
                write!(f, "operation requires contiguous rows")
            }
            TensorError::DimensionMismatch { expected, got } => {
                write!(f, "expected {} dimensions, got {}", expected, got)
            }
            TensorError::IndexOutOfBounds { index, size } => {
                write!(f, "index {} out of bounds for size {}", index, size)
            }
            TensorError::TooManyRanks { got } => {
                write!(f, "too many ranks: {}", got)
            }
            TensorError::SubByteUnsupported => {
                write!(f, "operation not supported for sub-byte types")
            }
        }
    }
}

// endregion: Error Types

// region: MinMaxResult

/// Named result from min/max reduction operations.
///
/// For scalar reductions (`try_minmax_all`), `V` is the scalar output type
/// and `I` defaults to `usize`.
/// For axis reductions (`try_minmax_axis`), `V` and `I` are tensors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MinMaxResult<V, I = usize> {
    pub min_value: V,
    pub min_index: I,
    pub max_value: V,
    pub max_index: I,
}

// endregion: MinMaxResult

// region: Tensor

/// N-dimensional array with NumKong-accelerated operations.
///
/// Uses raw memory allocation (no std::Vec) for maximum control.
///
/// Supports:
/// - Slicing and subviews (zero-copy)
/// - Dot-product multiplication with [`PackedMatrix`]
/// - Reductions (sum, min, max)
/// - Elementwise ops (scale, sum, blend, fma)
/// - Trigonometry (sin, cos, atan)
///
/// # Example
///
/// ```rust,ignore
/// // Requires linking against libnumkong C library
/// use numkong::{Tensor, PackedMatrix};
///
/// let a = Tensor::<f32>::try_full(&[1024, 512], 1.0).unwrap();
/// let b = Tensor::<f32>::try_full(&[256, 512], 1.0).unwrap();
///
/// // Pack B once, multiply many times
/// let b_packed = PackedMatrix::try_pack(&b).unwrap();
/// let c = a.dots_packed(&b_packed);  // Returns (1024 × 256)
/// ```
pub struct Tensor<T, A: Allocator = Global, const MAX_RANK: usize = DEFAULT_MAX_RANK> {
    /// Raw pointer to data buffer.
    data: NonNull<T>,
    /// Total allocation size in bytes (used by Drop and as_slice).
    alloc_bytes: usize,
    /// Shape dimensions (always logical).
    shape: [usize; MAX_RANK],
    /// Strides in bytes.
    strides: [isize; MAX_RANK],
    /// Number of dimensions.
    ndim: usize,
    /// Allocator instance.
    pub(crate) alloc: A,
}

// Safety: Tensor owns its data and T: Send implies the array is Send
unsafe impl<T: Send, A: Allocator + Send, const MAX_RANK: usize> Send for Tensor<T, A, MAX_RANK> {}
// Safety: Tensor has no interior mutability, &Tensor<T> is safe to share if T: Sync
unsafe impl<T: Sync, A: Allocator + Sync, const MAX_RANK: usize> Sync for Tensor<T, A, MAX_RANK> {}

impl<T, A: Allocator, const MAX_RANK: usize> Drop for Tensor<T, A, MAX_RANK> {
    fn drop(&mut self) {
        if self.alloc_bytes > 0 {
            unsafe {
                let count = self.alloc_bytes / core::mem::size_of::<T>();
                core::ptr::drop_in_place(core::ptr::slice_from_raw_parts_mut(
                    self.data.as_ptr(),
                    count,
                ));
                let layout =
                    alloc::alloc::Layout::from_size_align(self.alloc_bytes, SIMD_ALIGNMENT)
                        .unwrap();
                self.alloc.deallocate(
                    NonNull::new_unchecked(self.data.as_ptr() as *mut u8),
                    layout,
                );
            }
        }
    }
}

impl<T: StorageElement, A: Allocator + Clone, const MAX_RANK: usize> Tensor<T, A, MAX_RANK> {
    /// Try to clone this tensor, returning an error on allocation failure.
    pub fn try_clone(&self) -> Result<Self, TensorError> {
        Self::try_from_slice_in(self.as_slice(), self.shape(), self.alloc.clone())
    }
}

impl<T: StorageElement, A: Allocator + Clone, const MAX_RANK: usize> Clone
    for Tensor<T, A, MAX_RANK>
{
    fn clone(&self) -> Self { self.try_clone().expect("tensor clone allocation failed") }
}

// Generic allocator-aware methods
impl<T: StorageElement, A: Allocator, const MAX_RANK: usize> Tensor<T, A, MAX_RANK> {
    /// Creates a new Tensor filled with a value using a custom allocator.
    ///
    /// The `shape` specifies logical dimensions. For sub-byte types (e.g.
    /// `i4x2` with `dimensions_per_value() == 2`), the innermost dimension
    /// must be divisible by the packing factor. Storage is allocated for
    /// `total / dimensions_per_value()` packed values.
    ///
    /// Returns `Err` if allocation fails or shape is invalid.
    pub fn try_full_in(shape: &[usize], value: T, alloc: A) -> Result<Self, TensorError> {
        if shape.len() > MAX_RANK {
            return Err(TensorError::TooManyRanks { got: shape.len() });
        }

        let total: usize = shape.iter().product();
        if total == 0 && !shape.is_empty() {
            if let Some(i) = shape.iter().position(|&d| d == 0) {
                return Err(TensorError::InvalidShape {
                    axis: i,
                    size: 0,
                    reason: "zero-sized dimension",
                });
            }
        }

        let dims_per_value = T::dimensions_per_value();
        if dims_per_value > 1 && !shape.is_empty() && shape[shape.len() - 1] % dims_per_value != 0 {
            return Err(TensorError::InvalidShape {
                axis: shape.len() - 1,
                size: shape[shape.len() - 1],
                reason: "innermost dimension must be divisible by dimensions_per_value()",
            });
        }
        let storage_count = if dims_per_value == 1 {
            total
        } else {
            total / dims_per_value
        };

        // Allocate SIMD-aligned raw buffer using our allocator
        let data = if storage_count == 0 {
            NonNull::dangling()
        } else {
            let layout = alloc::alloc::Layout::from_size_align(
                storage_count * core::mem::size_of::<T>(),
                SIMD_ALIGNMENT,
            )
            .map_err(|_| TensorError::AllocationFailed)?;
            let ptr = alloc
                .allocate(layout)
                .ok_or(TensorError::AllocationFailed)?;
            // Initialize all storage elements
            unsafe {
                let ptr = ptr.as_ptr() as *mut T;
                for i in 0..storage_count {
                    core::ptr::write(ptr.add(i), value);
                }
                NonNull::new_unchecked(ptr)
            }
        };

        // Build shape and strides arrays
        let mut shape_arr = [0usize; MAX_RANK];
        shape_arr[..shape.len()].copy_from_slice(shape);

        let mut strides_arr = [0isize; MAX_RANK];
        Self::compute_strides_into(shape, dims_per_value, &mut strides_arr);

        Ok(Self {
            data,
            alloc_bytes: storage_count * core::mem::size_of::<T>(),
            shape: shape_arr,
            strides: strides_arr,
            ndim: shape.len(),
            alloc,
        })
    }

    /// Creates a zero-initialized Tensor using a custom allocator.
    ///
    /// Returns `Err` if allocation fails or shape is invalid.
    pub fn try_zeros_in(shape: &[usize], alloc: A) -> Result<Self, TensorError>
    where
        T: Default,
    {
        Self::try_full_in(shape, T::default(), alloc)
    }

    /// Creates a Tensor filled with ones using a custom allocator.
    ///
    /// Returns `Err` if allocation fails or shape is invalid.
    pub fn try_ones_in(shape: &[usize], alloc: A) -> Result<Self, TensorError>
    where
        T: crate::types::NumberLike,
    {
        Self::try_full_in(shape, T::one(), alloc)
    }

    /// Creates an uninitialized Tensor using a custom allocator.
    ///
    /// # Safety
    /// The returned tensor's contents are uninitialized. Reading before writing
    /// is undefined behavior.
    pub unsafe fn try_empty_in(shape: &[usize], alloc: A) -> Result<Self, TensorError> {
        if shape.len() > MAX_RANK {
            return Err(TensorError::TooManyRanks { got: shape.len() });
        }

        let total: usize = shape.iter().product();
        if total == 0 && !shape.is_empty() {
            if let Some(i) = shape.iter().position(|&d| d == 0) {
                return Err(TensorError::InvalidShape {
                    axis: i,
                    size: 0,
                    reason: "zero-sized dimension",
                });
            }
        }

        let dims_per_value = T::dimensions_per_value();
        if dims_per_value > 1 && !shape.is_empty() && shape[shape.len() - 1] % dims_per_value != 0 {
            return Err(TensorError::InvalidShape {
                axis: shape.len() - 1,
                size: shape[shape.len() - 1],
                reason: "innermost dimension must be divisible by dimensions_per_value()",
            });
        }
        let storage_count = if dims_per_value == 1 {
            total
        } else {
            total / dims_per_value
        };

        let data = if storage_count == 0 {
            NonNull::dangling()
        } else {
            let layout = alloc::alloc::Layout::from_size_align(
                storage_count * core::mem::size_of::<T>(),
                SIMD_ALIGNMENT,
            )
            .map_err(|_| TensorError::AllocationFailed)?;
            let ptr = alloc
                .allocate(layout)
                .ok_or(TensorError::AllocationFailed)?;
            unsafe { NonNull::new_unchecked(ptr.as_ptr() as *mut T) }
        };

        let mut shape_arr = [0usize; MAX_RANK];
        shape_arr[..shape.len()].copy_from_slice(shape);

        let mut strides_arr = [0isize; MAX_RANK];
        Self::compute_strides_into(shape, dims_per_value, &mut strides_arr);

        Ok(Self {
            data,
            alloc_bytes: storage_count * core::mem::size_of::<T>(),
            shape: shape_arr,
            strides: strides_arr,
            ndim: shape.len(),
            alloc,
        })
    }

    /// Creates a Tensor from existing storage data using a custom allocator.
    ///
    /// The `shape` specifies logical dimensions. For sub-byte types, the
    /// `data` slice has `shape.product() / dimensions_per_value()` storage
    /// values. For normal types, `data.len() == shape.product()`.
    pub fn try_from_slice_in(data: &[T], shape: &[usize], alloc: A) -> Result<Self, TensorError> {
        if shape.len() > MAX_RANK {
            return Err(TensorError::TooManyRanks { got: shape.len() });
        }

        let total: usize = shape.iter().product();
        let dims_per_value = T::dimensions_per_value();
        if dims_per_value > 1 && !shape.is_empty() && shape[shape.len() - 1] % dims_per_value != 0 {
            return Err(TensorError::InvalidShape {
                axis: shape.len() - 1,
                size: shape[shape.len() - 1],
                reason: "innermost dimension must be divisible by dimensions_per_value()",
            });
        }
        let expected_storage = if dims_per_value == 1 {
            total
        } else {
            total / dims_per_value
        };
        if data.len() != expected_storage {
            return Err(TensorError::ShapeMismatch {
                axis: 0,
                expected: expected_storage,
                got: data.len(),
            });
        }

        // Allocate SIMD-aligned buffer and copy using our allocator
        let ptr = if expected_storage == 0 {
            NonNull::dangling()
        } else {
            let layout = alloc::alloc::Layout::from_size_align(
                expected_storage * core::mem::size_of::<T>(),
                SIMD_ALIGNMENT,
            )
            .map_err(|_| TensorError::AllocationFailed)?;
            let ptr = alloc
                .allocate(layout)
                .ok_or(TensorError::AllocationFailed)?;
            // Clone all storage elements
            unsafe {
                let ptr = ptr.as_ptr() as *mut T;
                for (i, item) in data[..expected_storage].iter().enumerate() {
                    core::ptr::write(ptr.add(i), *item);
                }
                NonNull::new_unchecked(ptr)
            }
        };

        let mut shape_arr = [0usize; MAX_RANK];
        shape_arr[..shape.len()].copy_from_slice(shape);

        let mut strides_arr = [0isize; MAX_RANK];
        Self::compute_strides_into(shape, dims_per_value, &mut strides_arr);

        Ok(Self {
            data: ptr,
            alloc_bytes: expected_storage * core::mem::size_of::<T>(),
            shape: shape_arr,
            strides: strides_arr,
            ndim: shape.len(),
            alloc,
        })
    }

    /// Compute byte strides from a logical shape.
    ///
    /// For sub-byte types (`dims_per_value > 1`), the innermost dimension is
    /// divided by `dims_per_value` before computing strides, so the innermost
    /// stride is `size_of::<T>()` and covers `dims_per_value` logical elements
    /// per step.
    fn compute_strides_into(
        shape: &[usize],
        dims_per_value: usize,
        strides: &mut [isize; MAX_RANK],
    ) {
        let elem_size = core::mem::size_of::<T>();
        if shape.is_empty() {
            return;
        }

        let mut stride = elem_size as isize;
        for i in (0..shape.len()).rev() {
            strides[i] = stride;
            let dim_storage = if i == shape.len() - 1 && dims_per_value > 1 {
                shape[i] / dims_per_value
            } else {
                shape[i]
            };
            stride *= dim_storage as isize;
        }
    }

    /// Returns a reference to the allocator.
    pub fn allocator(&self) -> &A { &self.alloc }

    /// Number of storage values (for sub-byte types, less than numel).
    pub fn storage_len(&self) -> usize {
        let dims_per_value = T::dimensions_per_value();
        let n = self.numel();
        if dims_per_value == 1 {
            n
        } else {
            n / dims_per_value
        }
    }

    /// Convert a 1D contiguous tensor into a [`Vector`], transferring ownership without copying.
    ///
    /// Returns an error if the tensor is not 1D or not contiguous.
    pub fn try_into_vector(self) -> Result<Vector<T, A>, TensorError> {
        if self.ndim != 1 {
            return Err(TensorError::DimensionMismatch {
                expected: 1,
                got: self.ndim,
            });
        }
        let expected_stride = core::mem::size_of::<T>() as isize;
        if self.strides[0] != expected_stride {
            return Err(TensorError::NonContiguousRows);
        }
        let dims = self.shape[0];
        let dims_per_value = T::dimensions_per_value();
        let values = if dims_per_value == 1 {
            dims
        } else {
            (dims + dims_per_value - 1) / dims_per_value
        };
        let data = self.data;
        let alloc = unsafe { core::ptr::read(&self.alloc) };
        core::mem::forget(self);
        // SAFETY: data and alloc_bytes originate from a valid Tensor allocation
        Ok(unsafe { Vector::from_raw_parts(data, dims, values, alloc) })
    }
}

// Methods that don't require Clone
impl<T, A: Allocator, const MAX_RANK: usize> Tensor<T, A, MAX_RANK> {
    /// Construct a tensor from raw parts, taking ownership of the allocation.
    ///
    /// # Safety
    /// - `data` must point to a valid allocation of `alloc_bytes` bytes
    ///   obtained from `alloc`, aligned to [`SIMD_ALIGNMENT`].
    /// - `shape`, `strides`, and `ndim` must be consistent with the data layout.
    /// - The caller must not free the memory (this tensor takes ownership).
    pub unsafe fn from_raw_parts(
        data: NonNull<T>,
        alloc_bytes: usize,
        shape: [usize; MAX_RANK],
        strides: [isize; MAX_RANK],
        ndim: usize,
        alloc: A,
    ) -> Self {
        Self {
            data,
            alloc_bytes,
            shape,
            strides,
            ndim,
            alloc,
        }
    }

    /// Returns the shape of the array.
    pub fn shape(&self) -> &[usize] { &self.shape[..self.ndim] }

    /// Returns the number of dimensions.
    pub fn ndim(&self) -> usize { self.ndim }

    /// Returns the number of dimensions (alias for `ndim()`).
    pub fn rank(&self) -> usize { self.ndim }

    /// Returns the total number of logical elements.
    ///
    /// For sub-byte types, this is the logical count (e.g. 6 nibbles for
    /// shape `[6]` of `i4x2`), not the storage count (3 packed values).
    pub fn numel(&self) -> usize { self.shape[..self.ndim].iter().product() }

    /// Returns true if the array has no elements.
    pub fn is_empty(&self) -> bool { self.alloc_bytes == 0 }

    /// Returns the stride in bytes for the given dimension.
    pub fn stride_bytes(&self, dim: usize) -> isize { self.strides[dim] }

    /// Returns a pointer to the data.
    pub fn as_ptr(&self) -> *const T { self.data.as_ptr() }

    /// Returns a mutable pointer to the data.
    pub fn as_mut_ptr(&mut self) -> *mut T { self.data.as_ptr() }

    /// Returns the underlying storage data as a slice.
    ///
    /// For sub-byte types, the slice contains packed storage values, not
    /// individual logical elements.
    pub fn as_slice(&self) -> &[T] {
        let count = self.alloc_bytes / core::mem::size_of::<T>();
        unsafe { core::slice::from_raw_parts(self.data.as_ptr(), count) }
    }

    /// Returns the underlying storage data as a mutable slice.
    ///
    /// For sub-byte types, the slice contains packed storage values, not
    /// individual logical elements.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        let count = self.alloc_bytes / core::mem::size_of::<T>();
        unsafe { core::slice::from_raw_parts_mut(self.data.as_ptr(), count) }
    }

    /// Check if rows are contiguous (required for GEMM A matrix).
    pub fn has_contiguous_rows(&self) -> bool {
        if self.ndim != 2 {
            return false;
        }
        // Last dimension stride should be element size
        self.strides[1] == core::mem::size_of::<T>() as isize
    }

    /// Returns a row of a 2D array.
    pub fn row(&self, i: usize) -> Option<&[T]> {
        if self.ndim != 2 {
            return None;
        }
        let (rows, cols) = (self.shape[0], self.shape[1]);
        if i >= rows {
            return None;
        }
        let start = i * cols;
        Some(&self.as_slice()[start..start + cols])
    }

    /// Returns a mutable row of a 2D array.
    pub fn row_mut(&mut self, i: usize) -> Option<&mut [T]> {
        if self.ndim != 2 {
            return None;
        }
        let (rows, cols) = (self.shape[0], self.shape[1]);
        if i >= rows {
            return None;
        }
        let start = i * cols;
        Some(&mut self.as_mut_slice()[start..start + cols])
    }
}

// Convenience methods using Global allocator
impl<T: StorageElement + Clone, const MAX_RANK: usize> Tensor<T, Global, MAX_RANK> {
    /// Creates a new Tensor filled with a value using the global allocator.
    ///
    /// Returns `Err` if allocation fails or shape is invalid.
    pub fn try_full(shape: &[usize], value: T) -> Result<Self, TensorError> {
        Self::try_full_in(shape, value, Global)
    }

    /// Creates a zero-initialized Tensor using the global allocator.
    pub fn try_zeros(shape: &[usize]) -> Result<Self, TensorError>
    where
        T: Default,
    {
        Self::try_zeros_in(shape, Global)
    }

    /// Creates a Tensor filled with ones using the global allocator.
    pub fn try_ones(shape: &[usize]) -> Result<Self, TensorError>
    where
        T: crate::types::NumberLike,
    {
        Self::try_ones_in(shape, Global)
    }

    /// Creates an uninitialized Tensor using the global allocator.
    ///
    /// # Safety
    /// The returned tensor's contents are uninitialized. Reading before writing
    /// is undefined behavior.
    pub unsafe fn try_empty(shape: &[usize]) -> Result<Self, TensorError> {
        unsafe { Self::try_empty_in(shape, Global) }
    }

    /// Creates a Tensor from existing slice data using the global allocator.
    ///
    /// Returns `Err` if shape doesn't match data length or allocation fails.
    pub fn try_from_slice(data: &[T], shape: &[usize]) -> Result<Self, TensorError> {
        Self::try_from_slice_in(data, shape, Global)
    }

    /// Convenience constructor that panics on error.
    pub fn from_slice(data: &[T], shape: &[usize]) -> Self {
        Self::try_from_slice(data, shape).expect("Tensor::from_slice failed")
    }
}

// endregion: Tensor

// region: SliceRange

/// Represents a range specification for slicing along one dimension.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SliceRange {
    /// Full range (equivalent to `..`)
    Full,
    /// Single index (reduces dimension)
    Index(usize),
    /// Range from start to end exclusive (equivalent to `start..end`)
    Range { start: usize, end: usize },
    /// Range from start to end with step (equivalent to `start..end;step`)
    RangeStep {
        start: usize,
        end: usize,
        step: isize,
    },
}

impl SliceRange {
    /// Create a full range.
    pub fn full() -> Self { Self::Full }

    /// Create a single index.
    pub fn index(i: usize) -> Self { Self::Index(i) }

    /// Create a range from start to end.
    pub fn range(start: usize, end: usize) -> Self { Self::Range { start, end } }

    /// Create a range with step.
    pub fn range_step(start: usize, end: usize, step: isize) -> Self {
        Self::RangeStep { start, end, step }
    }
}

/// A stepped range for use in tuple-based slicing (compile-time dispatch).
///
/// Rust has no built-in literal for stepped ranges, so this struct fills that
/// gap.  Use it inside `.slice()` tuples:
/// ```ignore
/// t.slice((.., RangeStep::new(0, 6, 2))).unwrap();  // t[:, 0:6:2]
/// ```
#[derive(Clone, Copy, Debug)]
pub struct RangeStep {
    pub start: usize,
    pub end: usize,
    pub step: isize,
}

impl RangeStep {
    pub fn new(start: usize, end: usize, step: isize) -> Self { Self { start, end, step } }
}

// endregion: SliceRange

// region: SliceArg + SliceSpec

/// Resolve a signed index against a dimension size.  Negative values wrap from
/// the end: `-1` → `dim_size - 1`, `-2` → `dim_size - 2`, etc.
#[inline(always)]
fn resolve_signed_(index: isize, dim_size: usize) -> Result<usize, TensorError> {
    if index >= 0 {
        Ok(index as usize)
    } else {
        dim_size
            .checked_sub(index.unsigned_abs())
            .ok_or(TensorError::IndexOutOfBounds {
                index: index.unsigned_abs(),
                size: dim_size,
            })
    }
}

/// Processes one axis of a slice operation, directly computing the effect on
/// output shape, strides, and byte offset — no intermediate enum dispatch.
///
/// Each impl is monomorphized and fully inlined, so the compiler sees concrete
/// types at every call site with zero runtime branching overhead.
///
/// Unsigned types: `RangeFull`, `usize`, `Range<usize>`, `RangeTo<usize>`,
/// `RangeFrom<usize>`, `RangeInclusive<usize>`.
///
/// Signed types (negative wraps from end): `isize`, `Range<isize>`,
/// `RangeTo<isize>`, `RangeFrom<isize>`, `RangeInclusive<isize>`.
///
/// Stepped: `RangeStep` (Rust has no built-in stepped range literal).
///
/// Note: integer literals default to `i32`, so write `0_usize` not `0`.
pub trait SliceArg {
    fn apply(
        self,
        dim_size: usize,
        dim_stride: isize,
        out_shape: &mut [usize],
        out_strides: &mut [isize],
        new_ndim: &mut usize,
        offset: &mut isize,
    ) -> Result<(), TensorError>;
}

impl SliceArg for core::ops::RangeFull {
    #[inline(always)]
    fn apply(
        self,
        dim_size: usize,
        dim_stride: isize,
        out_shape: &mut [usize],
        out_strides: &mut [isize],
        new_ndim: &mut usize,
        _offset: &mut isize,
    ) -> Result<(), TensorError> {
        out_shape[*new_ndim] = dim_size;
        out_strides[*new_ndim] = dim_stride;
        *new_ndim += 1;
        Ok(())
    }
}

impl SliceArg for usize {
    #[inline(always)]
    fn apply(
        self,
        dim_size: usize,
        dim_stride: isize,
        _out_shape: &mut [usize],
        _out_strides: &mut [isize],
        _new_ndim: &mut usize,
        offset: &mut isize,
    ) -> Result<(), TensorError> {
        if self >= dim_size {
            return Err(TensorError::IndexOutOfBounds {
                index: self,
                size: dim_size,
            });
        }
        *offset += self as isize * dim_stride;
        Ok(())
    }
}

impl SliceArg for isize {
    #[inline(always)]
    fn apply(
        self,
        dim_size: usize,
        dim_stride: isize,
        out_shape: &mut [usize],
        out_strides: &mut [isize],
        new_ndim: &mut usize,
        offset: &mut isize,
    ) -> Result<(), TensorError> {
        resolve_signed_(self, dim_size)?.apply(
            dim_size,
            dim_stride,
            out_shape,
            out_strides,
            new_ndim,
            offset,
        )
    }
}

impl SliceArg for core::ops::Range<usize> {
    #[inline(always)]
    fn apply(
        self,
        dim_size: usize,
        dim_stride: isize,
        out_shape: &mut [usize],
        out_strides: &mut [isize],
        new_ndim: &mut usize,
        offset: &mut isize,
    ) -> Result<(), TensorError> {
        if self.start > self.end || self.end > dim_size {
            return Err(TensorError::IndexOutOfBounds {
                index: self.end,
                size: dim_size,
            });
        }
        out_shape[*new_ndim] = self.end - self.start;
        out_strides[*new_ndim] = dim_stride;
        *new_ndim += 1;
        *offset += self.start as isize * dim_stride;
        Ok(())
    }
}

impl SliceArg for core::ops::Range<isize> {
    #[inline(always)]
    fn apply(
        self,
        dim_size: usize,
        dim_stride: isize,
        out_shape: &mut [usize],
        out_strides: &mut [isize],
        new_ndim: &mut usize,
        offset: &mut isize,
    ) -> Result<(), TensorError> {
        let start = resolve_signed_(self.start, dim_size)?;
        let end = resolve_signed_(self.end, dim_size)?;
        (start..end).apply(
            dim_size,
            dim_stride,
            out_shape,
            out_strides,
            new_ndim,
            offset,
        )
    }
}

impl SliceArg for core::ops::RangeTo<usize> {
    #[inline(always)]
    fn apply(
        self,
        dim_size: usize,
        dim_stride: isize,
        out_shape: &mut [usize],
        out_strides: &mut [isize],
        new_ndim: &mut usize,
        offset: &mut isize,
    ) -> Result<(), TensorError> {
        (0..self.end).apply(
            dim_size,
            dim_stride,
            out_shape,
            out_strides,
            new_ndim,
            offset,
        )
    }
}

impl SliceArg for core::ops::RangeTo<isize> {
    #[inline(always)]
    fn apply(
        self,
        dim_size: usize,
        dim_stride: isize,
        out_shape: &mut [usize],
        out_strides: &mut [isize],
        new_ndim: &mut usize,
        offset: &mut isize,
    ) -> Result<(), TensorError> {
        let end = resolve_signed_(self.end, dim_size)?;
        (0..end).apply(
            dim_size,
            dim_stride,
            out_shape,
            out_strides,
            new_ndim,
            offset,
        )
    }
}

impl SliceArg for core::ops::RangeFrom<usize> {
    #[inline(always)]
    fn apply(
        self,
        dim_size: usize,
        dim_stride: isize,
        out_shape: &mut [usize],
        out_strides: &mut [isize],
        new_ndim: &mut usize,
        offset: &mut isize,
    ) -> Result<(), TensorError> {
        (self.start..dim_size).apply(
            dim_size,
            dim_stride,
            out_shape,
            out_strides,
            new_ndim,
            offset,
        )
    }
}

impl SliceArg for core::ops::RangeFrom<isize> {
    #[inline(always)]
    fn apply(
        self,
        dim_size: usize,
        dim_stride: isize,
        out_shape: &mut [usize],
        out_strides: &mut [isize],
        new_ndim: &mut usize,
        offset: &mut isize,
    ) -> Result<(), TensorError> {
        let start = resolve_signed_(self.start, dim_size)?;
        (start..dim_size).apply(
            dim_size,
            dim_stride,
            out_shape,
            out_strides,
            new_ndim,
            offset,
        )
    }
}

impl SliceArg for core::ops::RangeInclusive<usize> {
    #[inline(always)]
    fn apply(
        self,
        dim_size: usize,
        dim_stride: isize,
        out_shape: &mut [usize],
        out_strides: &mut [isize],
        new_ndim: &mut usize,
        offset: &mut isize,
    ) -> Result<(), TensorError> {
        let start = *self.start();
        let end = self.end().saturating_add(1);
        (start..end).apply(
            dim_size,
            dim_stride,
            out_shape,
            out_strides,
            new_ndim,
            offset,
        )
    }
}

impl SliceArg for core::ops::RangeInclusive<isize> {
    #[inline(always)]
    fn apply(
        self,
        dim_size: usize,
        dim_stride: isize,
        out_shape: &mut [usize],
        out_strides: &mut [isize],
        new_ndim: &mut usize,
        offset: &mut isize,
    ) -> Result<(), TensorError> {
        let start = resolve_signed_(*self.start(), dim_size)?;
        let end = resolve_signed_(*self.end(), dim_size)?.saturating_add(1);
        (start..end).apply(
            dim_size,
            dim_stride,
            out_shape,
            out_strides,
            new_ndim,
            offset,
        )
    }
}

impl SliceArg for RangeStep {
    #[inline(always)]
    fn apply(
        self,
        dim_size: usize,
        dim_stride: isize,
        out_shape: &mut [usize],
        out_strides: &mut [isize],
        new_ndim: &mut usize,
        offset: &mut isize,
    ) -> Result<(), TensorError> {
        if self.start >= dim_size || (self.end > dim_size && self.step > 0) {
            return Err(TensorError::IndexOutOfBounds {
                index: if self.start >= dim_size {
                    self.start
                } else {
                    self.end
                },
                size: dim_size,
            });
        }
        if self.step == 0 {
            return Err(TensorError::InvalidShape {
                axis: 0,
                size: 0,
                reason: "step cannot be zero",
            });
        }
        let count = if self.step > 0 {
            (self.end.saturating_sub(self.start) + self.step as usize - 1) / self.step as usize
        } else {
            let abs_step = (-self.step) as usize;
            (self.start.saturating_sub(self.end) + abs_step - 1) / abs_step
        };
        out_shape[*new_ndim] = count;
        out_strides[*new_ndim] = dim_stride * self.step;
        *new_ndim += 1;
        *offset += self.start as isize * dim_stride;
        Ok(())
    }
}

type LayoutResult<const MAX_RANK: usize> =
    Result<([usize; MAX_RANK], [isize; MAX_RANK], usize, isize, usize), TensorError>;

/// Computes the full slice layout from shape, strides, and ndim.
///
/// Implemented for `&[SliceRange]` / `&[SliceRange; N]` (backward compat, runtime
/// dispatch) and tuples of [`SliceArg`] types from arity 1-8 (compile-time
/// dispatch, fully inlined with zero branching overhead).
pub trait SliceSpec {
    fn apply_layout<const MAX_RANK: usize>(
        self,
        shape: &[usize; MAX_RANK],
        strides: &[isize; MAX_RANK],
        ndim: usize,
    ) -> LayoutResult<MAX_RANK>;
}

impl SliceSpec for &[SliceRange] {
    fn apply_layout<const MAX_RANK: usize>(
        self,
        shape: &[usize; MAX_RANK],
        strides: &[isize; MAX_RANK],
        ndim: usize,
    ) -> LayoutResult<MAX_RANK> {
        slice_layout_(shape, strides, ndim, self)
    }
}

impl<const N: usize> SliceSpec for &[SliceRange; N] {
    fn apply_layout<const MAX_RANK: usize>(
        self,
        shape: &[usize; MAX_RANK],
        strides: &[isize; MAX_RANK],
        ndim: usize,
    ) -> LayoutResult<MAX_RANK> {
        slice_layout_(shape, strides, ndim, self.as_slice())
    }
}

impl<A0: SliceArg> SliceSpec for (A0,) {
    fn apply_layout<const MAX_RANK: usize>(
        self,
        shape: &[usize; MAX_RANK],
        strides: &[isize; MAX_RANK],
        ndim: usize,
    ) -> LayoutResult<MAX_RANK> {
        if ndim != 1 {
            return Err(TensorError::DimensionMismatch {
                expected: 1,
                got: ndim,
            });
        }
        let mut s = [0usize; MAX_RANK];
        let mut st = [0isize; MAX_RANK];
        let (mut nd, mut off) = (0usize, 0isize);
        self.0
            .apply(shape[0], strides[0], &mut s, &mut st, &mut nd, &mut off)?;
        let len = if nd == 0 { 1 } else { s[..nd].iter().product() };
        Ok((s, st, nd, off, len))
    }
}

impl<A0: SliceArg, A1: SliceArg> SliceSpec for (A0, A1) {
    fn apply_layout<const MAX_RANK: usize>(
        self,
        shape: &[usize; MAX_RANK],
        strides: &[isize; MAX_RANK],
        ndim: usize,
    ) -> LayoutResult<MAX_RANK> {
        if ndim != 2 {
            return Err(TensorError::DimensionMismatch {
                expected: 2,
                got: ndim,
            });
        }
        let mut s = [0usize; MAX_RANK];
        let mut st = [0isize; MAX_RANK];
        let (mut nd, mut off) = (0usize, 0isize);
        self.0
            .apply(shape[0], strides[0], &mut s, &mut st, &mut nd, &mut off)?;
        self.1
            .apply(shape[1], strides[1], &mut s, &mut st, &mut nd, &mut off)?;
        let len = if nd == 0 { 1 } else { s[..nd].iter().product() };
        Ok((s, st, nd, off, len))
    }
}

impl<A0: SliceArg, A1: SliceArg, A2: SliceArg> SliceSpec for (A0, A1, A2) {
    fn apply_layout<const MAX_RANK: usize>(
        self,
        shape: &[usize; MAX_RANK],
        strides: &[isize; MAX_RANK],
        ndim: usize,
    ) -> LayoutResult<MAX_RANK> {
        if ndim != 3 {
            return Err(TensorError::DimensionMismatch {
                expected: 3,
                got: ndim,
            });
        }
        let mut s = [0usize; MAX_RANK];
        let mut st = [0isize; MAX_RANK];
        let (mut nd, mut off) = (0usize, 0isize);
        self.0
            .apply(shape[0], strides[0], &mut s, &mut st, &mut nd, &mut off)?;
        self.1
            .apply(shape[1], strides[1], &mut s, &mut st, &mut nd, &mut off)?;
        self.2
            .apply(shape[2], strides[2], &mut s, &mut st, &mut nd, &mut off)?;
        let len = if nd == 0 { 1 } else { s[..nd].iter().product() };
        Ok((s, st, nd, off, len))
    }
}

impl<A0: SliceArg, A1: SliceArg, A2: SliceArg, A3: SliceArg> SliceSpec for (A0, A1, A2, A3) {
    fn apply_layout<const MAX_RANK: usize>(
        self,
        shape: &[usize; MAX_RANK],
        strides: &[isize; MAX_RANK],
        ndim: usize,
    ) -> LayoutResult<MAX_RANK> {
        if ndim != 4 {
            return Err(TensorError::DimensionMismatch {
                expected: 4,
                got: ndim,
            });
        }
        let mut s = [0usize; MAX_RANK];
        let mut st = [0isize; MAX_RANK];
        let (mut nd, mut off) = (0usize, 0isize);
        self.0
            .apply(shape[0], strides[0], &mut s, &mut st, &mut nd, &mut off)?;
        self.1
            .apply(shape[1], strides[1], &mut s, &mut st, &mut nd, &mut off)?;
        self.2
            .apply(shape[2], strides[2], &mut s, &mut st, &mut nd, &mut off)?;
        self.3
            .apply(shape[3], strides[3], &mut s, &mut st, &mut nd, &mut off)?;
        let len = if nd == 0 { 1 } else { s[..nd].iter().product() };
        Ok((s, st, nd, off, len))
    }
}

impl<A0: SliceArg, A1: SliceArg, A2: SliceArg, A3: SliceArg, A4: SliceArg> SliceSpec
    for (A0, A1, A2, A3, A4)
{
    fn apply_layout<const MAX_RANK: usize>(
        self,
        shape: &[usize; MAX_RANK],
        strides: &[isize; MAX_RANK],
        ndim: usize,
    ) -> LayoutResult<MAX_RANK> {
        if ndim != 5 {
            return Err(TensorError::DimensionMismatch {
                expected: 5,
                got: ndim,
            });
        }
        let mut s = [0usize; MAX_RANK];
        let mut st = [0isize; MAX_RANK];
        let (mut nd, mut off) = (0usize, 0isize);
        self.0
            .apply(shape[0], strides[0], &mut s, &mut st, &mut nd, &mut off)?;
        self.1
            .apply(shape[1], strides[1], &mut s, &mut st, &mut nd, &mut off)?;
        self.2
            .apply(shape[2], strides[2], &mut s, &mut st, &mut nd, &mut off)?;
        self.3
            .apply(shape[3], strides[3], &mut s, &mut st, &mut nd, &mut off)?;
        self.4
            .apply(shape[4], strides[4], &mut s, &mut st, &mut nd, &mut off)?;
        let len = if nd == 0 { 1 } else { s[..nd].iter().product() };
        Ok((s, st, nd, off, len))
    }
}

impl<A0: SliceArg, A1: SliceArg, A2: SliceArg, A3: SliceArg, A4: SliceArg, A5: SliceArg> SliceSpec
    for (A0, A1, A2, A3, A4, A5)
{
    fn apply_layout<const MAX_RANK: usize>(
        self,
        shape: &[usize; MAX_RANK],
        strides: &[isize; MAX_RANK],
        ndim: usize,
    ) -> LayoutResult<MAX_RANK> {
        if ndim != 6 {
            return Err(TensorError::DimensionMismatch {
                expected: 6,
                got: ndim,
            });
        }
        let mut s = [0usize; MAX_RANK];
        let mut st = [0isize; MAX_RANK];
        let (mut nd, mut off) = (0usize, 0isize);
        self.0
            .apply(shape[0], strides[0], &mut s, &mut st, &mut nd, &mut off)?;
        self.1
            .apply(shape[1], strides[1], &mut s, &mut st, &mut nd, &mut off)?;
        self.2
            .apply(shape[2], strides[2], &mut s, &mut st, &mut nd, &mut off)?;
        self.3
            .apply(shape[3], strides[3], &mut s, &mut st, &mut nd, &mut off)?;
        self.4
            .apply(shape[4], strides[4], &mut s, &mut st, &mut nd, &mut off)?;
        self.5
            .apply(shape[5], strides[5], &mut s, &mut st, &mut nd, &mut off)?;
        let len = if nd == 0 { 1 } else { s[..nd].iter().product() };
        Ok((s, st, nd, off, len))
    }
}

impl<
        A0: SliceArg,
        A1: SliceArg,
        A2: SliceArg,
        A3: SliceArg,
        A4: SliceArg,
        A5: SliceArg,
        A6: SliceArg,
    > SliceSpec for (A0, A1, A2, A3, A4, A5, A6)
{
    fn apply_layout<const MAX_RANK: usize>(
        self,
        shape: &[usize; MAX_RANK],
        strides: &[isize; MAX_RANK],
        ndim: usize,
    ) -> LayoutResult<MAX_RANK> {
        if ndim != 7 {
            return Err(TensorError::DimensionMismatch {
                expected: 7,
                got: ndim,
            });
        }
        let mut s = [0usize; MAX_RANK];
        let mut st = [0isize; MAX_RANK];
        let (mut nd, mut off) = (0usize, 0isize);
        self.0
            .apply(shape[0], strides[0], &mut s, &mut st, &mut nd, &mut off)?;
        self.1
            .apply(shape[1], strides[1], &mut s, &mut st, &mut nd, &mut off)?;
        self.2
            .apply(shape[2], strides[2], &mut s, &mut st, &mut nd, &mut off)?;
        self.3
            .apply(shape[3], strides[3], &mut s, &mut st, &mut nd, &mut off)?;
        self.4
            .apply(shape[4], strides[4], &mut s, &mut st, &mut nd, &mut off)?;
        self.5
            .apply(shape[5], strides[5], &mut s, &mut st, &mut nd, &mut off)?;
        self.6
            .apply(shape[6], strides[6], &mut s, &mut st, &mut nd, &mut off)?;
        let len = if nd == 0 { 1 } else { s[..nd].iter().product() };
        Ok((s, st, nd, off, len))
    }
}

impl<
        A0: SliceArg,
        A1: SliceArg,
        A2: SliceArg,
        A3: SliceArg,
        A4: SliceArg,
        A5: SliceArg,
        A6: SliceArg,
        A7: SliceArg,
    > SliceSpec for (A0, A1, A2, A3, A4, A5, A6, A7)
{
    fn apply_layout<const MAX_RANK: usize>(
        self,
        shape: &[usize; MAX_RANK],
        strides: &[isize; MAX_RANK],
        ndim: usize,
    ) -> LayoutResult<MAX_RANK> {
        if ndim != 8 {
            return Err(TensorError::DimensionMismatch {
                expected: 8,
                got: ndim,
            });
        }
        let mut s = [0usize; MAX_RANK];
        let mut st = [0isize; MAX_RANK];
        let (mut nd, mut off) = (0usize, 0isize);
        self.0
            .apply(shape[0], strides[0], &mut s, &mut st, &mut nd, &mut off)?;
        self.1
            .apply(shape[1], strides[1], &mut s, &mut st, &mut nd, &mut off)?;
        self.2
            .apply(shape[2], strides[2], &mut s, &mut st, &mut nd, &mut off)?;
        self.3
            .apply(shape[3], strides[3], &mut s, &mut st, &mut nd, &mut off)?;
        self.4
            .apply(shape[4], strides[4], &mut s, &mut st, &mut nd, &mut off)?;
        self.5
            .apply(shape[5], strides[5], &mut s, &mut st, &mut nd, &mut off)?;
        self.6
            .apply(shape[6], strides[6], &mut s, &mut st, &mut nd, &mut off)?;
        self.7
            .apply(shape[7], strides[7], &mut s, &mut st, &mut nd, &mut off)?;
        let len = if nd == 0 { 1 } else { s[..nd].iter().product() };
        Ok((s, st, nd, off, len))
    }
}

// endregion: SliceArg + SliceSpec

#[doc(hidden)]
pub trait TensorCoordinates {
    const ARITY: usize;

    fn resolve<const MAX_RANK: usize>(
        self,
        shape: &[usize; MAX_RANK],
        ndim: usize,
    ) -> Result<[usize; MAX_RANK], TensorError>;
}

impl<I0: VectorIndex, I1: VectorIndex> TensorCoordinates for (I0, I1) {
    const ARITY: usize = 2;

    fn resolve<const MAX_RANK: usize>(
        self,
        shape: &[usize; MAX_RANK],
        ndim: usize,
    ) -> Result<[usize; MAX_RANK], TensorError> {
        if ndim != Self::ARITY {
            return Err(TensorError::DimensionMismatch {
                expected: Self::ARITY,
                got: ndim,
            });
        }
        let mut resolved = [0usize; MAX_RANK];
        resolved[0] = resolve_index_for_size_(self.0, shape[0])?;
        resolved[1] = resolve_index_for_size_(self.1, shape[1])?;
        Ok(resolved)
    }
}

impl<I0: VectorIndex, I1: VectorIndex, I2: VectorIndex> TensorCoordinates for (I0, I1, I2) {
    const ARITY: usize = 3;

    fn resolve<const MAX_RANK: usize>(
        self,
        shape: &[usize; MAX_RANK],
        ndim: usize,
    ) -> Result<[usize; MAX_RANK], TensorError> {
        if ndim != Self::ARITY {
            return Err(TensorError::DimensionMismatch {
                expected: Self::ARITY,
                got: ndim,
            });
        }
        let mut resolved = [0usize; MAX_RANK];
        resolved[0] = resolve_index_for_size_(self.0, shape[0])?;
        resolved[1] = resolve_index_for_size_(self.1, shape[1])?;
        resolved[2] = resolve_index_for_size_(self.2, shape[2])?;
        Ok(resolved)
    }
}

impl<I0: VectorIndex, I1: VectorIndex, I2: VectorIndex, I3: VectorIndex> TensorCoordinates
    for (I0, I1, I2, I3)
{
    const ARITY: usize = 4;

    fn resolve<const MAX_RANK: usize>(
        self,
        shape: &[usize; MAX_RANK],
        ndim: usize,
    ) -> Result<[usize; MAX_RANK], TensorError> {
        if ndim != Self::ARITY {
            return Err(TensorError::DimensionMismatch {
                expected: Self::ARITY,
                got: ndim,
            });
        }
        let mut resolved = [0usize; MAX_RANK];
        resolved[0] = resolve_index_for_size_(self.0, shape[0])?;
        resolved[1] = resolve_index_for_size_(self.1, shape[1])?;
        resolved[2] = resolve_index_for_size_(self.2, shape[2])?;
        resolved[3] = resolve_index_for_size_(self.3, shape[3])?;
        Ok(resolved)
    }
}

impl<I0: VectorIndex, I1: VectorIndex, I2: VectorIndex, I3: VectorIndex, I4: VectorIndex>
    TensorCoordinates for (I0, I1, I2, I3, I4)
{
    const ARITY: usize = 5;

    fn resolve<const MAX_RANK: usize>(
        self,
        shape: &[usize; MAX_RANK],
        ndim: usize,
    ) -> Result<[usize; MAX_RANK], TensorError> {
        if ndim != Self::ARITY {
            return Err(TensorError::DimensionMismatch {
                expected: Self::ARITY,
                got: ndim,
            });
        }
        let mut resolved = [0usize; MAX_RANK];
        resolved[0] = resolve_index_for_size_(self.0, shape[0])?;
        resolved[1] = resolve_index_for_size_(self.1, shape[1])?;
        resolved[2] = resolve_index_for_size_(self.2, shape[2])?;
        resolved[3] = resolve_index_for_size_(self.3, shape[3])?;
        resolved[4] = resolve_index_for_size_(self.4, shape[4])?;
        Ok(resolved)
    }
}

impl<
        I0: VectorIndex,
        I1: VectorIndex,
        I2: VectorIndex,
        I3: VectorIndex,
        I4: VectorIndex,
        I5: VectorIndex,
    > TensorCoordinates for (I0, I1, I2, I3, I4, I5)
{
    const ARITY: usize = 6;

    fn resolve<const MAX_RANK: usize>(
        self,
        shape: &[usize; MAX_RANK],
        ndim: usize,
    ) -> Result<[usize; MAX_RANK], TensorError> {
        if ndim != Self::ARITY {
            return Err(TensorError::DimensionMismatch {
                expected: Self::ARITY,
                got: ndim,
            });
        }
        let mut resolved = [0usize; MAX_RANK];
        resolved[0] = resolve_index_for_size_(self.0, shape[0])?;
        resolved[1] = resolve_index_for_size_(self.1, shape[1])?;
        resolved[2] = resolve_index_for_size_(self.2, shape[2])?;
        resolved[3] = resolve_index_for_size_(self.3, shape[3])?;
        resolved[4] = resolve_index_for_size_(self.4, shape[4])?;
        resolved[5] = resolve_index_for_size_(self.5, shape[5])?;
        Ok(resolved)
    }
}

impl<
        I0: VectorIndex,
        I1: VectorIndex,
        I2: VectorIndex,
        I3: VectorIndex,
        I4: VectorIndex,
        I5: VectorIndex,
        I6: VectorIndex,
    > TensorCoordinates for (I0, I1, I2, I3, I4, I5, I6)
{
    const ARITY: usize = 7;

    fn resolve<const MAX_RANK: usize>(
        self,
        shape: &[usize; MAX_RANK],
        ndim: usize,
    ) -> Result<[usize; MAX_RANK], TensorError> {
        if ndim != Self::ARITY {
            return Err(TensorError::DimensionMismatch {
                expected: Self::ARITY,
                got: ndim,
            });
        }
        let mut resolved = [0usize; MAX_RANK];
        resolved[0] = resolve_index_for_size_(self.0, shape[0])?;
        resolved[1] = resolve_index_for_size_(self.1, shape[1])?;
        resolved[2] = resolve_index_for_size_(self.2, shape[2])?;
        resolved[3] = resolve_index_for_size_(self.3, shape[3])?;
        resolved[4] = resolve_index_for_size_(self.4, shape[4])?;
        resolved[5] = resolve_index_for_size_(self.5, shape[5])?;
        resolved[6] = resolve_index_for_size_(self.6, shape[6])?;
        Ok(resolved)
    }
}

impl<
        I0: VectorIndex,
        I1: VectorIndex,
        I2: VectorIndex,
        I3: VectorIndex,
        I4: VectorIndex,
        I5: VectorIndex,
        I6: VectorIndex,
        I7: VectorIndex,
    > TensorCoordinates for (I0, I1, I2, I3, I4, I5, I6, I7)
{
    const ARITY: usize = 8;

    fn resolve<const MAX_RANK: usize>(
        self,
        shape: &[usize; MAX_RANK],
        ndim: usize,
    ) -> Result<[usize; MAX_RANK], TensorError> {
        if ndim != Self::ARITY {
            return Err(TensorError::DimensionMismatch {
                expected: Self::ARITY,
                got: ndim,
            });
        }
        let mut resolved = [0usize; MAX_RANK];
        resolved[0] = resolve_index_for_size_(self.0, shape[0])?;
        resolved[1] = resolve_index_for_size_(self.1, shape[1])?;
        resolved[2] = resolve_index_for_size_(self.2, shape[2])?;
        resolved[3] = resolve_index_for_size_(self.3, shape[3])?;
        resolved[4] = resolve_index_for_size_(self.4, shape[4])?;
        resolved[5] = resolve_index_for_size_(self.5, shape[5])?;
        resolved[6] = resolve_index_for_size_(self.6, shape[6])?;
        resolved[7] = resolve_index_for_size_(self.7, shape[7])?;
        Ok(resolved)
    }
}

// region: TensorView

/// A read-only view into a Tensor (doesn't own data).
///
/// Views provide zero-copy access to array subregions with potentially
/// different strides than the original array.
pub struct TensorView<'a, T, const MAX_RANK: usize = DEFAULT_MAX_RANK> {
    /// Pointer to first element of view.
    data: *const T,
    /// Shape of the view (always logical).
    shape: [usize; MAX_RANK],
    /// Strides in bytes.
    strides: [isize; MAX_RANK],
    /// Number of dimensions.
    ndim: usize,
    /// Lifetime marker.
    _marker: PhantomData<&'a T>,
}

impl<'a, T, const MAX_RANK: usize> TensorView<'a, T, MAX_RANK> {
    /// Create a view from a raw pointer, shape, and byte strides.
    ///
    /// The `shape` specifies logical dimensions. For sub-byte types, the
    /// storage count is inferred as `shape.product() / dimensions_per_value()`.
    /// For normal types the two are equal.
    ///
    /// # Safety
    /// - `data` must be valid for reads over the region described by `shape` and `strides_bytes`.
    /// - The pointed-to memory must outlive `'a`.
    /// - `shape.len()` must be `<= MAX_RANK`.
    /// - `shape.len()` must equal `strides_bytes.len()`.
    ///
    /// # Panics
    /// Panics if `shape.len() > MAX_RANK` or `shape.len() != strides_bytes.len()`.
    pub unsafe fn from_raw_parts(data: *const T, shape: &[usize], strides_bytes: &[isize]) -> Self {
        assert!(
            shape.len() <= MAX_RANK,
            "ndim {} exceeds MAX_RANK {}",
            shape.len(),
            MAX_RANK
        );
        assert_eq!(
            shape.len(),
            strides_bytes.len(),
            "shape and strides must have the same length"
        );
        let ndim = shape.len();
        let mut s = [0usize; MAX_RANK];
        let mut st = [0isize; MAX_RANK];
        s[..ndim].copy_from_slice(shape);
        st[..ndim].copy_from_slice(strides_bytes);
        Self {
            data,
            shape: s,
            strides: st,
            ndim,
            _marker: PhantomData,
        }
    }

    /// Returns the shape of the view.
    pub fn shape(&self) -> &[usize] { &self.shape[..self.ndim] }

    /// Returns the number of dimensions.
    pub fn ndim(&self) -> usize { self.ndim }

    /// Returns the number of dimensions (alias for `ndim()`).
    pub fn rank(&self) -> usize { self.ndim }

    /// Returns the total number of logical elements (computed from shape).
    pub fn numel(&self) -> usize { self.shape[..self.ndim].iter().product() }

    /// Returns true if the view has no elements.
    pub fn is_empty(&self) -> bool { self.numel() == 0 }

    /// Returns the stride in bytes for the given dimension.
    pub fn stride_bytes(&self, dim: usize) -> isize { self.strides[dim] }

    /// Returns a pointer to the first element.
    pub fn as_ptr(&self) -> *const T { self.data }

    /// Check if the view has contiguous rows.
    pub fn has_contiguous_rows(&self) -> bool {
        if self.ndim != 2 {
            return false;
        }
        self.strides[1] == core::mem::size_of::<T>() as isize
    }

    /// Check if the entire view is contiguous in memory.
    pub fn is_contiguous(&self) -> bool {
        if self.ndim == 0 {
            return true;
        }
        let elem_size = core::mem::size_of::<T>() as isize;
        let mut expected_stride = elem_size;
        for i in (0..self.ndim).rev() {
            if self.strides[i] != expected_stride {
                return false;
            }
            expected_stride *= self.shape[i] as isize;
        }
        true
    }

    /// Get element at flat index (only valid for contiguous views).
    ///
    /// # Safety
    /// Caller must ensure the view is contiguous and index is in bounds.
    pub unsafe fn get_unchecked(&self, index: usize) -> &T { &*self.data.add(index) }

    /// Try to get an element by flat logical row-major index.
    pub fn try_flat<I: VectorIndex>(&self, index: I) -> Result<&T, TensorError> {
        if self.ndim == 0 {
            return Err(TensorError::DimensionMismatch {
                expected: 1,
                got: 0,
            });
        }
        let logical_index = resolve_index_for_size_(index, self.numel())?;
        let offset = offset_from_flat_(&self.shape, &self.strides, self.ndim, logical_index);
        Ok(unsafe { &*((self.data as *const u8).offset(offset) as *const T) })
    }

    /// Try to get an element by exact coordinates.
    pub fn try_coords<C: TensorCoordinates>(&self, coords: C) -> Result<&T, TensorError> {
        let resolved = coords.resolve(&self.shape, self.ndim)?;
        let offset = offset_from_coords_(&self.strides, &resolved, self.ndim);
        Ok(unsafe { &*((self.data as *const u8).offset(offset) as *const T) })
    }

    /// Try to access the scalar value of a rank-0 tensor view.
    pub fn try_scalar(&self) -> Result<&T, TensorError> {
        if self.ndim != 0 {
            return Err(TensorError::DimensionMismatch {
                expected: 0,
                got: self.ndim,
            });
        }
        Ok(unsafe { &*self.data })
    }

    /// Slice the leading axis by one index, reducing rank by one.
    pub fn slice_leading<I: VectorIndex>(
        &self,
        index: I,
    ) -> Result<TensorView<'a, T, MAX_RANK>, TensorError> {
        let (shape, strides, ndim, offset, _) =
            slice_leading_layout_(&self.shape, &self.strides, self.ndim, index)?;
        Ok(TensorView {
            data: unsafe { (self.data as *const u8).offset(offset) as *const T },
            shape,
            strides,
            ndim,
            _marker: PhantomData,
        })
    }

    /// Slice the view along multiple dimensions.
    ///
    /// Accepts tuples of Rust range types or `&[SliceRange]`:
    /// ```ignore
    /// view.slice((.., 0_usize));       // t[:, 0]
    /// view.slice((1..3, ..));           // t[1:3, :]
    /// view.slice(&[SliceRange::full(), SliceRange::index(0)]); // old syntax
    /// ```
    pub fn slice(&self, spec: impl SliceSpec) -> Result<TensorView<'a, T, MAX_RANK>, TensorError> {
        let (shape, strides, ndim, offset, _) =
            spec.apply_layout(&self.shape, &self.strides, self.ndim)?;
        Ok(TensorView {
            data: unsafe { (self.data as *const u8).offset(offset) as *const T },
            shape,
            strides,
            ndim,
            _marker: PhantomData,
        })
    }
}

impl<'a, T: Clone + StorageElement, const MAX_RANK: usize> TensorView<'a, T, MAX_RANK> {
    /// Copy the view contents to a new owned Tensor.
    pub fn to_owned(&self) -> Result<Tensor<T, Global, MAX_RANK>, TensorError> {
        if self.is_contiguous() {
            let slice = unsafe { core::slice::from_raw_parts(self.data, self.storage_len()) };
            Tensor::try_from_slice(slice, self.shape())
        } else {
            // For non-contiguous views, we need to copy element by element
            let mut result = Tensor::try_full(self.shape(), unsafe { *self.data })?;
            self.copy_to_contiguous(result.as_mut_slice());
            Ok(result)
        }
    }

    /// Number of storage values (for sub-byte types, less than numel).
    pub fn storage_len(&self) -> usize {
        let dims_per_value = T::dimensions_per_value();
        let n = self.numel();
        if dims_per_value == 1 {
            n
        } else {
            n / dims_per_value
        }
    }

    /// Convert to slice (only valid for contiguous views).
    pub fn as_contiguous_slice(&self) -> Option<&[T]> {
        if self.is_contiguous() {
            Some(unsafe { core::slice::from_raw_parts(self.data, self.storage_len()) })
        } else {
            None
        }
    }

    fn copy_to_contiguous(&self, dest: &mut [T]) {
        // For 2D case, optimize the copy
        if self.ndim == 2 {
            let rows = self.shape[0];
            let cols = self.shape[1];
            let row_stride = self.strides[0];
            let col_stride = self.strides[1];
            let mut dest_idx = 0;
            for r in 0..rows {
                let row_ptr =
                    unsafe { (self.data as *const u8).offset(r as isize * row_stride) as *const T };
                for c in 0..cols {
                    let elem_ptr = unsafe {
                        (row_ptr as *const u8).offset(c as isize * col_stride) as *const T
                    };
                    dest[dest_idx] = unsafe { *elem_ptr };
                    dest_idx += 1;
                }
            }
        } else {
            // General N-dimensional case: iterate in row-major order
            let mut indices = [0usize; MAX_RANK];
            for dest_slot in dest[..self.numel()].iter_mut() {
                let mut offset = 0isize;
                for (index_val, stride_val) in indices[..self.ndim].iter().zip(self.strides[..self.ndim].iter()) {
                    offset += *index_val as isize * stride_val;
                }
                let elem_ptr = unsafe { (self.data as *const u8).offset(offset) as *const T };
                *dest_slot = unsafe { *elem_ptr };

                // Increment indices (row-major order)
                for d in (0..self.ndim).rev() {
                    indices[d] += 1;
                    if indices[d] < self.shape[d] {
                        break;
                    }
                    indices[d] = 0;
                }
            }
        }
    }
}

// region: TensorSpan

/// A mutable view into a Tensor.
pub struct TensorSpan<'a, T, const MAX_RANK: usize = DEFAULT_MAX_RANK> {
    /// Pointer to first element of view.
    data: *mut T,
    /// Shape of the view (always logical).
    shape: [usize; MAX_RANK],
    /// Strides in bytes.
    strides: [isize; MAX_RANK],
    /// Number of dimensions.
    ndim: usize,
    /// Lifetime marker.
    _marker: PhantomData<&'a mut T>,
}

impl<'a, T, const MAX_RANK: usize> TensorSpan<'a, T, MAX_RANK> {
    /// Create a mutable view from a raw pointer, shape, and byte strides.
    ///
    /// # Safety
    /// - `data` must be valid for reads and writes over the region described by `shape` and `strides_bytes`.
    /// - The pointed-to memory must outlive `'a`.
    /// - `shape.len()` must be `<= MAX_RANK`.
    /// - `shape.len()` must equal `strides_bytes.len()`.
    /// - No other references to the memory may exist for the duration of `'a`.
    ///
    /// # Panics
    /// Panics if `shape.len() > MAX_RANK` or `shape.len() != strides_bytes.len()`.
    pub unsafe fn from_raw_parts(data: *mut T, shape: &[usize], strides_bytes: &[isize]) -> Self {
        assert!(
            shape.len() <= MAX_RANK,
            "ndim {} exceeds MAX_RANK {}",
            shape.len(),
            MAX_RANK
        );
        assert_eq!(
            shape.len(),
            strides_bytes.len(),
            "shape and strides must have the same length"
        );
        let ndim = shape.len();
        let mut s = [0usize; MAX_RANK];
        let mut st = [0isize; MAX_RANK];
        s[..ndim].copy_from_slice(shape);
        st[..ndim].copy_from_slice(strides_bytes);
        Self {
            data,
            shape: s,
            strides: st,
            ndim,
            _marker: PhantomData,
        }
    }

    /// Returns the shape of the view.
    pub fn shape(&self) -> &[usize] { &self.shape[..self.ndim] }

    /// Returns the number of dimensions.
    pub fn ndim(&self) -> usize { self.ndim }

    /// Returns the number of dimensions (alias for `ndim()`).
    pub fn rank(&self) -> usize { self.ndim }

    /// Returns the total number of logical elements (computed from shape).
    pub fn numel(&self) -> usize { self.shape[..self.ndim].iter().product() }

    /// Returns true if the view has no elements.
    pub fn is_empty(&self) -> bool { self.numel() == 0 }

    /// Returns the stride in bytes for the given dimension.
    pub fn stride_bytes(&self, dim: usize) -> isize { self.strides[dim] }

    /// Returns a pointer to the first element.
    pub fn as_ptr(&self) -> *const T { self.data }

    /// Returns a mutable pointer to the first element.
    pub fn as_mut_ptr(&mut self) -> *mut T { self.data }

    /// Check if the view has contiguous rows.
    pub fn has_contiguous_rows(&self) -> bool {
        if self.ndim != 2 {
            return false;
        }
        self.strides[1] == core::mem::size_of::<T>() as isize
    }

    /// Check if the entire view is contiguous in memory.
    pub fn is_contiguous(&self) -> bool {
        if self.ndim == 0 {
            return true;
        }
        let elem_size = core::mem::size_of::<T>() as isize;
        let mut expected_stride = elem_size;
        for i in (0..self.ndim).rev() {
            if self.strides[i] != expected_stride {
                return false;
            }
            expected_stride *= self.shape[i] as isize;
        }
        true
    }

    /// Reborrow as immutable view.
    pub fn as_view(&self) -> TensorView<'_, T, MAX_RANK> {
        TensorView {
            data: self.data,
            shape: self.shape,
            strides: self.strides,
            ndim: self.ndim,
            _marker: PhantomData,
        }
    }

    /// Try to get an element by flat logical row-major index.
    pub fn try_flat<I: VectorIndex>(&self, index: I) -> Result<&T, TensorError> {
        if self.ndim == 0 {
            return Err(TensorError::DimensionMismatch {
                expected: 1,
                got: 0,
            });
        }
        let logical_index = resolve_index_for_size_(index, self.numel())?;
        let offset = offset_from_flat_(&self.shape, &self.strides, self.ndim, logical_index);
        Ok(unsafe { &*((self.data as *const u8).offset(offset) as *const T) })
    }

    /// Try to get a mutable element by flat logical row-major index.
    pub fn try_flat_mut<I: VectorIndex>(&mut self, index: I) -> Result<&mut T, TensorError> {
        if self.ndim == 0 {
            return Err(TensorError::DimensionMismatch {
                expected: 1,
                got: 0,
            });
        }
        let logical_index = resolve_index_for_size_(index, self.numel())?;
        let offset = offset_from_flat_(&self.shape, &self.strides, self.ndim, logical_index);
        Ok(unsafe { &mut *((self.data as *mut u8).offset(offset) as *mut T) })
    }

    /// Try to get an element by exact coordinates.
    pub fn try_coords<C: TensorCoordinates>(&self, coords: C) -> Result<&T, TensorError> {
        let resolved = coords.resolve(&self.shape, self.ndim)?;
        let offset = offset_from_coords_(&self.strides, &resolved, self.ndim);
        Ok(unsafe { &*((self.data as *const u8).offset(offset) as *const T) })
    }

    /// Try to get a mutable element by exact coordinates.
    pub fn try_coords_mut<C: TensorCoordinates>(
        &mut self,
        coords: C,
    ) -> Result<&mut T, TensorError> {
        let resolved = coords.resolve(&self.shape, self.ndim)?;
        let offset = offset_from_coords_(&self.strides, &resolved, self.ndim);
        Ok(unsafe { &mut *((self.data as *mut u8).offset(offset) as *mut T) })
    }

    /// Try to access the scalar value of a rank-0 tensor span.
    pub fn try_scalar(&self) -> Result<&T, TensorError> {
        if self.ndim != 0 {
            return Err(TensorError::DimensionMismatch {
                expected: 0,
                got: self.ndim,
            });
        }
        Ok(unsafe { &*self.data })
    }

    /// Try to access the mutable scalar value of a rank-0 tensor span.
    pub fn try_scalar_mut(&mut self) -> Result<&mut T, TensorError> {
        if self.ndim != 0 {
            return Err(TensorError::DimensionMismatch {
                expected: 0,
                got: self.ndim,
            });
        }
        Ok(unsafe { &mut *self.data })
    }

    /// Slice the leading axis by one index, reducing rank by one.
    pub fn slice_leading<I: VectorIndex>(
        &self,
        index: I,
    ) -> Result<TensorView<'_, T, MAX_RANK>, TensorError> {
        let (shape, strides, ndim, offset, _) =
            slice_leading_layout_(&self.shape, &self.strides, self.ndim, index)?;
        Ok(TensorView {
            data: unsafe { (self.data as *const u8).offset(offset) as *const T },
            shape,
            strides,
            ndim,
            _marker: PhantomData,
        })
    }

    /// Slice the leading axis mutably by one index, reducing rank by one.
    pub fn slice_leading_mut<I: VectorIndex>(
        &mut self,
        index: I,
    ) -> Result<TensorSpan<'_, T, MAX_RANK>, TensorError> {
        let (shape, strides, ndim, offset, _) =
            slice_leading_layout_(&self.shape, &self.strides, self.ndim, index)?;
        Ok(TensorSpan {
            data: unsafe { (self.data as *mut u8).offset(offset) as *mut T },
            shape,
            strides,
            ndim,
            _marker: PhantomData,
        })
    }

    /// Slice the span along multiple dimensions.
    ///
    /// Accepts tuples of Rust range types or `&[SliceRange]`.
    pub fn slice(&self, spec: impl SliceSpec) -> Result<TensorView<'_, T, MAX_RANK>, TensorError> {
        let (shape, strides, ndim, offset, _) =
            spec.apply_layout(&self.shape, &self.strides, self.ndim)?;
        Ok(TensorView {
            data: unsafe { (self.data as *const u8).offset(offset) as *const T },
            shape,
            strides,
            ndim,
            _marker: PhantomData,
        })
    }

    /// Slice the span mutably along multiple dimensions.
    ///
    /// Accepts tuples of Rust range types or `&[SliceRange]`.
    pub fn slice_mut(
        &mut self,
        spec: impl SliceSpec,
    ) -> Result<TensorSpan<'_, T, MAX_RANK>, TensorError> {
        let (shape, strides, ndim, offset, _) =
            spec.apply_layout(&self.shape, &self.strides, self.ndim)?;
        Ok(TensorSpan {
            data: unsafe { (self.data as *mut u8).offset(offset) as *mut T },
            shape,
            strides,
            ndim,
            _marker: PhantomData,
        })
    }
}

impl<'a, T: StorageElement, const MAX_RANK: usize> TensorSpan<'a, T, MAX_RANK> {
    /// Number of storage values (for sub-byte types, less than numel).
    pub fn storage_len(&self) -> usize {
        let dims_per_value = T::dimensions_per_value();
        let n = self.numel();
        if dims_per_value == 1 {
            n
        } else {
            n / dims_per_value
        }
    }

    /// Convert to slice (only valid for contiguous views).
    pub fn as_contiguous_slice(&self) -> Option<&[T]> {
        if self.is_contiguous() {
            Some(unsafe { core::slice::from_raw_parts(self.data, self.storage_len()) })
        } else {
            None
        }
    }

    /// Convert to mutable slice (only valid for contiguous views).
    pub fn as_contiguous_slice_mut(&mut self) -> Option<&mut [T]> {
        if self.is_contiguous() {
            Some(unsafe { core::slice::from_raw_parts_mut(self.data, self.storage_len()) })
        } else {
            None
        }
    }
}

// endregion: TensorSpan

// region: TensorRef / TensorMut Traits

/// Read-only structural access to N-dimensional tensor containers.
///
/// Implemented by [`Tensor`], [`TensorView`], and [`TensorSpan`], enabling
/// generic code over any tensor-like container.
pub trait TensorRef<T: StorageElement, const MAX_RANK: usize> {
    fn shape(&self) -> &[usize];
    fn ndim(&self) -> usize;
    fn stride_bytes(&self, dim: usize) -> isize;
    fn as_ptr(&self) -> *const T;
    /// Borrow as an immutable [`TensorView`].
    fn view(&self) -> TensorView<'_, T, MAX_RANK>;

    /// Total number of logical elements (product of shape dimensions).
    fn numel(&self) -> usize { self.shape().iter().product() }
    fn rank(&self) -> usize { self.ndim() }
    fn is_empty(&self) -> bool { self.numel() == 0 }

    fn has_contiguous_rows(&self) -> bool {
        self.ndim() == 2 && self.stride_bytes(1) == core::mem::size_of::<T>() as isize
    }

    fn is_contiguous(&self) -> bool {
        let dims_per_value = T::dimensions_per_value();
        let elem_size = core::mem::size_of::<T>() as isize;
        let mut expected = elem_size;
        let ndim = self.ndim();
        for i in (0..ndim).rev() {
            if self.stride_bytes(i) != expected {
                return false;
            }
            let dim_storage = if i == ndim - 1 && dims_per_value > 1 {
                self.shape()[i] / dims_per_value
            } else {
                self.shape()[i]
            };
            expected *= dim_storage as isize;
        }
        true
    }
}

/// Mutable structural access to N-dimensional tensor containers.
pub trait TensorMut<T: StorageElement, const MAX_RANK: usize>: TensorRef<T, MAX_RANK> {
    fn as_mut_ptr(&mut self) -> *mut T;
}

impl<T: StorageElement, A: Allocator, const R: usize> TensorRef<T, R> for Tensor<T, A, R> {
    fn shape(&self) -> &[usize] { &self.shape[..self.ndim] }
    fn ndim(&self) -> usize { self.ndim }
    fn stride_bytes(&self, dim: usize) -> isize { self.strides[dim] }
    fn as_ptr(&self) -> *const T { self.data.as_ptr() }
    fn view(&self) -> TensorView<'_, T, R> {
        TensorView {
            data: self.data.as_ptr(),
            shape: self.shape,
            strides: self.strides,
            ndim: self.ndim,
            _marker: PhantomData,
        }
    }
}

impl<T: StorageElement, A: Allocator, const R: usize> TensorMut<T, R> for Tensor<T, A, R> {
    fn as_mut_ptr(&mut self) -> *mut T { self.data.as_ptr() }
}

impl<'a, T: StorageElement, const R: usize> TensorRef<T, R> for TensorView<'a, T, R> {
    fn shape(&self) -> &[usize] { &self.shape[..self.ndim] }
    fn ndim(&self) -> usize { self.ndim }
    fn stride_bytes(&self, dim: usize) -> isize { self.strides[dim] }
    fn as_ptr(&self) -> *const T { self.data }
    fn view(&self) -> TensorView<'_, T, R> {
        TensorView {
            data: self.data,
            shape: self.shape,
            strides: self.strides,
            ndim: self.ndim,
            _marker: PhantomData,
        }
    }
}

impl<'a, T: StorageElement, const R: usize> TensorRef<T, R> for TensorSpan<'a, T, R> {
    fn shape(&self) -> &[usize] { &self.shape[..self.ndim] }
    fn ndim(&self) -> usize { self.ndim }
    fn stride_bytes(&self, dim: usize) -> isize { self.strides[dim] }
    fn as_ptr(&self) -> *const T { self.data }
    fn view(&self) -> TensorView<'_, T, R> { self.as_view() }
}

impl<'a, T: StorageElement, const R: usize> TensorMut<T, R> for TensorSpan<'a, T, R> {
    fn as_mut_ptr(&mut self) -> *mut T { self.data }
}

// endregion: TensorRef / TensorMut Traits

// region: AxisIterator

/// Iterator over sub-tensor views along a given axis.
///
/// Each item is a `TensorView` with the iterated dimension removed (rank - 1).
/// For a rank-2 matrix, `axis_views(0)` yields row views.
pub struct AxisIterator<'a, T, const MAX_RANK: usize = DEFAULT_MAX_RANK> {
    data: *const T,
    shape: [usize; MAX_RANK],
    strides: [isize; MAX_RANK],
    ndim: usize,
    axis: usize,
    axis_size: usize,
    axis_stride: isize,
    current: usize,
    _marker: PhantomData<&'a T>,
}

impl<'a, T, const MAX_RANK: usize> Iterator for AxisIterator<'a, T, MAX_RANK> {
    type Item = TensorView<'a, T, MAX_RANK>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.axis_size {
            return None;
        }
        let offset = self.current as isize * self.axis_stride;
        let sub_ptr = unsafe { (self.data as *const u8).offset(offset) as *const T };

        // Build sub-shape/strides with the axis dimension removed
        let mut sub_shape = [0usize; MAX_RANK];
        let mut sub_strides = [0isize; MAX_RANK];
        let mut j = 0;
        for i in 0..self.ndim {
            if i != self.axis {
                sub_shape[j] = self.shape[i];
                sub_strides[j] = self.strides[i];
                j += 1;
            }
        }
        let sub_ndim = self.ndim - 1;

        self.current += 1;
        Some(TensorView {
            data: sub_ptr,
            shape: sub_shape,
            strides: sub_strides,
            ndim: sub_ndim,
            _marker: PhantomData,
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.axis_size - self.current;
        (remaining, Some(remaining))
    }
}

impl<'a, T, const MAX_RANK: usize> ExactSizeIterator for AxisIterator<'a, T, MAX_RANK> {}
impl<'a, T, const MAX_RANK: usize> core::iter::FusedIterator for AxisIterator<'a, T, MAX_RANK> {}

/// Mutable iterator over sub-tensor spans along a given axis.
///
/// Each item is a `TensorSpan` with the iterated dimension removed (rank - 1).
/// For a rank-2 matrix, `axis_spans(0)` yields mutable row spans.
pub struct AxisIteratorMut<'a, T, const MAX_RANK: usize = DEFAULT_MAX_RANK> {
    data: *mut T,
    shape: [usize; MAX_RANK],
    strides: [isize; MAX_RANK],
    ndim: usize,
    axis: usize,
    axis_size: usize,
    axis_stride: isize,
    current: usize,
    _marker: PhantomData<&'a mut T>,
}

impl<'a, T, const MAX_RANK: usize> Iterator for AxisIteratorMut<'a, T, MAX_RANK> {
    type Item = TensorSpan<'a, T, MAX_RANK>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.axis_size {
            return None;
        }
        let offset = self.current as isize * self.axis_stride;
        let sub_ptr = unsafe { (self.data as *mut u8).offset(offset) as *mut T };

        let mut sub_shape = [0usize; MAX_RANK];
        let mut sub_strides = [0isize; MAX_RANK];
        let mut j = 0;
        for i in 0..self.ndim {
            if i != self.axis {
                sub_shape[j] = self.shape[i];
                sub_strides[j] = self.strides[i];
                j += 1;
            }
        }
        let sub_ndim = self.ndim - 1;

        self.current += 1;
        Some(TensorSpan {
            data: sub_ptr,
            shape: sub_shape,
            strides: sub_strides,
            ndim: sub_ndim,
            _marker: PhantomData,
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.axis_size - self.current;
        (remaining, Some(remaining))
    }
}

impl<'a, T, const MAX_RANK: usize> ExactSizeIterator for AxisIteratorMut<'a, T, MAX_RANK> {}
impl<'a, T, const MAX_RANK: usize> core::iter::FusedIterator for AxisIteratorMut<'a, T, MAX_RANK> {}

impl<'a, T, const MAX_RANK: usize> TensorView<'a, T, MAX_RANK> {
    /// Iterate along the given axis, yielding sub-tensor views with rank-1.
    pub fn axis_views<I: VectorIndex>(
        &self,
        axis: I,
    ) -> Result<AxisIterator<'a, T, MAX_RANK>, TensorError> {
        let axis = normalize_axis(axis, self.ndim)?;
        if self.ndim == 0 {
            return Err(TensorError::IndexOutOfBounds {
                index: 0,
                size: self.ndim,
            });
        }
        Ok(AxisIterator {
            data: self.data,
            shape: self.shape,
            strides: self.strides,
            ndim: self.ndim,
            axis,
            axis_size: self.shape[axis],
            axis_stride: self.strides[axis],
            current: 0,
            _marker: PhantomData,
        })
    }
}

impl<'a, T: StorageElement, const MAX_RANK: usize> TensorView<'a, T, MAX_RANK> {
    /// Transpose (reverse all dimensions, no data copy).
    ///
    /// Returns an error for sub-byte types with ndim >= 2, since transposing
    /// would produce non-contiguous strides that break packed element addressing.
    pub fn transpose(&self) -> Result<TensorView<'a, T, MAX_RANK>, TensorError> {
        if self.ndim < 2 {
            return Ok(TensorView {
                data: self.data,
                shape: self.shape,
                strides: self.strides,
                ndim: self.ndim,
                _marker: PhantomData,
            });
        }
        if T::dimensions_per_value() > 1 {
            return Err(TensorError::SubByteUnsupported);
        }
        let (shape, strides) = transpose_layout(&self.shape, &self.strides, self.ndim);
        Ok(TensorView {
            data: self.data,
            shape,
            strides,
            ndim: self.ndim,
            _marker: PhantomData,
        })
    }

    /// Reshape the view (must have same total elements, contiguous only).
    ///
    /// Returns an error for sub-byte types, since reshape would invalidate
    /// the packed element layout.
    pub fn reshape(&self, new_shape: &[usize]) -> Result<TensorView<'a, T, MAX_RANK>, TensorError> {
        if T::dimensions_per_value() > 1 {
            return Err(TensorError::SubByteUnsupported);
        }
        let (shape, strides, ndim) = reshape_layout::<T, MAX_RANK>(
            &self.shape,
            &self.strides,
            self.ndim,
            self.numel(),
            new_shape,
        )?;
        Ok(TensorView {
            data: self.data,
            shape,
            strides,
            ndim,
            _marker: PhantomData,
        })
    }

    /// Flatten to 1D (requires contiguous layout).
    pub fn flatten(&self) -> Result<TensorView<'a, T, MAX_RANK>, TensorError> {
        self.reshape(&[self.numel()])
    }

    /// Remove dimensions of size 1.
    pub fn squeeze(&self) -> TensorView<'a, T, MAX_RANK> {
        let (shape, strides, ndim) =
            squeeze_layout::<T, MAX_RANK>(&self.shape, &self.strides, self.ndim);
        TensorView {
            data: self.data,
            shape,
            strides,
            ndim,
            _marker: PhantomData,
        }
    }
}

impl<'a, T: StorageElement, const MAX_RANK: usize> TensorSpan<'a, T, MAX_RANK> {
    /// Transpose (reverse all dimensions, no data copy).
    ///
    /// Returns an error for sub-byte types with ndim >= 2.
    pub fn transpose(&self) -> Result<TensorSpan<'a, T, MAX_RANK>, TensorError> {
        if self.ndim < 2 {
            return Ok(TensorSpan {
                data: self.data,
                shape: self.shape,
                strides: self.strides,
                ndim: self.ndim,
                _marker: PhantomData,
            });
        }
        if T::dimensions_per_value() > 1 {
            return Err(TensorError::SubByteUnsupported);
        }
        let (shape, strides) = transpose_layout(&self.shape, &self.strides, self.ndim);
        Ok(TensorSpan {
            data: self.data,
            shape,
            strides,
            ndim: self.ndim,
            _marker: PhantomData,
        })
    }

    /// Reshape the span (must have same total elements, contiguous only).
    ///
    /// Returns an error for sub-byte types.
    pub fn reshape(&self, new_shape: &[usize]) -> Result<TensorSpan<'a, T, MAX_RANK>, TensorError> {
        if T::dimensions_per_value() > 1 {
            return Err(TensorError::SubByteUnsupported);
        }
        let (shape, strides, ndim) = reshape_layout::<T, MAX_RANK>(
            &self.shape,
            &self.strides,
            self.ndim,
            self.numel(),
            new_shape,
        )?;
        Ok(TensorSpan {
            data: self.data,
            shape,
            strides,
            ndim,
            _marker: PhantomData,
        })
    }

    /// Flatten to 1D (requires contiguous layout).
    pub fn flatten(&self) -> Result<TensorSpan<'a, T, MAX_RANK>, TensorError> {
        self.reshape(&[self.numel()])
    }

    /// Remove dimensions of size 1.
    pub fn squeeze(&self) -> TensorSpan<'a, T, MAX_RANK> {
        let (shape, strides, ndim) =
            squeeze_layout::<T, MAX_RANK>(&self.shape, &self.strides, self.ndim);
        TensorSpan {
            data: self.data,
            shape,
            strides,
            ndim,
            _marker: PhantomData,
        }
    }
}

impl<'a, T: Clone + StorageElement, const MAX_RANK: usize> TensorSpan<'a, T, MAX_RANK> {
    /// Copy the span contents to a new owned Tensor.
    pub fn to_owned(&self) -> Result<Tensor<T, Global, MAX_RANK>, TensorError> {
        self.as_view().to_owned()
    }
}

impl<'a, T, const MAX_RANK: usize> TensorSpan<'a, T, MAX_RANK> {
    /// Iterate mutably along the given axis, yielding sub-tensor spans with rank-1.
    pub fn axis_spans<I: VectorIndex>(
        &mut self,
        axis: I,
    ) -> Result<AxisIteratorMut<'a, T, MAX_RANK>, TensorError> {
        let axis = normalize_axis(axis, self.ndim)?;
        if self.ndim == 0 {
            return Err(TensorError::IndexOutOfBounds {
                index: 0,
                size: self.ndim,
            });
        }
        Ok(AxisIteratorMut {
            data: self.data,
            shape: self.shape,
            strides: self.strides,
            ndim: self.ndim,
            axis,
            axis_size: self.shape[axis],
            axis_stride: self.strides[axis],
            current: 0,
            _marker: PhantomData,
        })
    }
}

impl<T, A: Allocator, const MAX_RANK: usize> Tensor<T, A, MAX_RANK> {
    /// Iterate along the given axis, yielding sub-tensor views with rank-1.
    pub fn axis_views<I: VectorIndex>(
        &self,
        axis: I,
    ) -> Result<AxisIterator<'_, T, MAX_RANK>, TensorError> {
        self.view().axis_views(axis)
    }

    /// Iterate mutably along the given axis, yielding sub-tensor spans with rank-1.
    pub fn axis_spans<I: VectorIndex>(
        &mut self,
        axis: I,
    ) -> Result<AxisIteratorMut<'_, T, MAX_RANK>, TensorError> {
        let axis = normalize_axis(axis, self.ndim)?;
        if self.ndim == 0 {
            return Err(TensorError::IndexOutOfBounds {
                index: 0,
                size: self.ndim,
            });
        }
        Ok(AxisIteratorMut {
            data: self.data.as_ptr(),
            shape: self.shape,
            strides: self.strides,
            ndim: self.ndim,
            axis,
            axis_size: self.shape[axis],
            axis_stride: self.strides[axis],
            current: 0,
            _marker: PhantomData,
        })
    }
}

// endregion: AxisIterator

// region: Debug

impl<T: core::fmt::Debug, A: Allocator, const MAX_RANK: usize> core::fmt::Debug
    for Tensor<T, A, MAX_RANK>
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Tensor(shape={:?}, [", &self.shape[..self.ndim])?;
        let slice = self.as_slice();
        for (i, val) in slice.iter().enumerate() {
            if i >= 8 {
                write!(f, ", ...")?;
                break;
            }
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{:?}", val)?;
        }
        write!(f, "])")
    }
}

// endregion: Debug

// region: TensorViewIterator

/// Lazy element iterator over possibly non-contiguous tensor data.
///
/// Yields `(position, DimRef<'a, T>)` pairs in row-major order at the logical-scalar
/// level. For sub-byte types, the innermost dimension is expanded by `dimensions_per_value`.
/// Use [`.dims()`](TensorViewIterator::dims) when only the dimension proxies are needed.
///
/// Cost per `next()` is O(1) amortized (O(ndim) only on carry propagation).
pub struct TensorViewIterator<'a, T: FloatConvertible, const MAX_RANK: usize = DEFAULT_MAX_RANK> {
    data: *const T,
    shape: [usize; MAX_RANK],
    strides: [isize; MAX_RANK],
    ndim: usize,
    dims_per_value: usize,
    indices: [usize; MAX_RANK],
    remaining: usize,
    _marker: PhantomData<&'a T>,
}

/// Backward-compatible alias for [`TensorViewIterator`].
pub type TensorIterator<'a, T, const MAX_RANK: usize = DEFAULT_MAX_RANK> =
    TensorViewIterator<'a, T, MAX_RANK>;

impl<'a, T: FloatConvertible, const MAX_RANK: usize> Iterator
    for TensorViewIterator<'a, T, MAX_RANK>
{
    type Item = ([usize; MAX_RANK], DimRef<'a, T>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        let pos = self.indices;

        // Compute byte offset: outer dims use strides directly,
        // innermost logical index splits into storage_inner / sub_index.
        let mut offset = 0isize;
        if self.ndim > 0 {
            for d in 0..self.ndim - 1 {
                offset += self.indices[d] as isize * self.strides[d];
            }
            let inner_logical = self.indices[self.ndim - 1];
            let storage_inner = inner_logical / self.dims_per_value;
            offset += storage_inner as isize * self.strides[self.ndim - 1];
            let sub_index = inner_logical % self.dims_per_value;
            let ptr = unsafe { (self.data as *const u8).offset(offset) as *const T };
            let scalar = unsafe { *ptr }.unpack().as_ref()[sub_index];

            // Advance indices
            self.remaining -= 1;
            for d in (0..self.ndim).rev() {
                self.indices[d] += 1;
                if self.indices[d] < self.shape[d] {
                    break;
                }
                self.indices[d] = 0;
            }
            Some((pos, DimRef::new(scalar)))
        } else {
            // Scalar tensor (ndim == 0)
            self.remaining -= 1;
            let ptr = self.data;
            let scalar = unsafe { *ptr }.unpack().as_ref()[0];
            Some((pos, DimRef::new(scalar)))
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) { (self.remaining, Some(self.remaining)) }
}

impl<'a, T: FloatConvertible, const MAX_RANK: usize> ExactSizeIterator
    for TensorViewIterator<'a, T, MAX_RANK>
{
}
impl<'a, T: FloatConvertible, const MAX_RANK: usize> core::iter::FusedIterator
    for TensorViewIterator<'a, T, MAX_RANK>
{
}

impl<'a, T: FloatConvertible, const MAX_RANK: usize> TensorViewIterator<'a, T, MAX_RANK> {
    /// Adapt this iterator to yield only dimension proxies, discarding positions.
    pub fn dims(self) -> TensorViewDims<'a, T, MAX_RANK> { TensorViewDims { inner: self } }
}

/// Dimension-only adapter over [`TensorViewIterator`], yielding [`DimRef`] without positions.
///
/// Created by [`TensorViewIterator::dims()`].
pub struct TensorViewDims<'a, T: FloatConvertible, const MAX_RANK: usize = DEFAULT_MAX_RANK> {
    inner: TensorViewIterator<'a, T, MAX_RANK>,
}

/// Backward-compatible alias for [`TensorViewDims`].
pub type TensorDims<'a, T, const MAX_RANK: usize = DEFAULT_MAX_RANK> =
    TensorViewDims<'a, T, MAX_RANK>;
/// Backward-compatible alias for [`TensorViewDims`].
pub type TensorValues<'a, T, const MAX_RANK: usize = DEFAULT_MAX_RANK> =
    TensorViewDims<'a, T, MAX_RANK>;

impl<'a, T: FloatConvertible, const MAX_RANK: usize> Iterator for TensorViewDims<'a, T, MAX_RANK> {
    type Item = DimRef<'a, T>;

    #[inline]
    fn next(&mut self) -> Option<DimRef<'a, T>> { self.inner.next().map(|(_, v)| v) }

    fn size_hint(&self) -> (usize, Option<usize>) { self.inner.size_hint() }
}

impl<'a, T: FloatConvertible, const MAX_RANK: usize> ExactSizeIterator
    for TensorViewDims<'a, T, MAX_RANK>
{
}
impl<'a, T: FloatConvertible, const MAX_RANK: usize> core::iter::FusedIterator
    for TensorViewDims<'a, T, MAX_RANK>
{
}

// endregion: TensorViewIterator

// region: TensorSpanIterator

/// Mutable element iterator over possibly non-contiguous tensor data.
///
/// Yields `(position, DimMut<'a, T>)` pairs in row-major order at the logical-scalar
/// level. For sub-byte types, each proxy performs a read-modify-write on drop.
pub struct TensorSpanIterator<'a, T: FloatConvertible, const MAX_RANK: usize = DEFAULT_MAX_RANK> {
    data: *mut T,
    shape: [usize; MAX_RANK],
    strides: [isize; MAX_RANK],
    ndim: usize,
    dims_per_value: usize,
    indices: [usize; MAX_RANK],
    remaining: usize,
    _marker: PhantomData<&'a mut T>,
}

impl<'a, T: FloatConvertible, const MAX_RANK: usize> Iterator
    for TensorSpanIterator<'a, T, MAX_RANK>
{
    type Item = ([usize; MAX_RANK], DimMut<'a, T>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        let pos = self.indices;

        let mut offset = 0isize;
        if self.ndim > 0 {
            for d in 0..self.ndim - 1 {
                offset += self.indices[d] as isize * self.strides[d];
            }
            let inner_logical = self.indices[self.ndim - 1];
            let storage_inner = inner_logical / self.dims_per_value;
            offset += storage_inner as isize * self.strides[self.ndim - 1];
            let sub_index = inner_logical % self.dims_per_value;
            let ptr = unsafe { (self.data as *mut u8).offset(offset) as *mut T };
            let scalar = unsafe { *ptr }.unpack().as_ref()[sub_index];

            self.remaining -= 1;
            for d in (0..self.ndim).rev() {
                self.indices[d] += 1;
                if self.indices[d] < self.shape[d] {
                    break;
                }
                self.indices[d] = 0;
            }
            Some((pos, unsafe { DimMut::new(ptr, sub_index, scalar) }))
        } else {
            self.remaining -= 1;
            let ptr = self.data;
            let scalar = unsafe { *ptr }.unpack().as_ref()[0];
            Some((pos, unsafe { DimMut::new(ptr, 0, scalar) }))
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) { (self.remaining, Some(self.remaining)) }
}

impl<'a, T: FloatConvertible, const MAX_RANK: usize> ExactSizeIterator
    for TensorSpanIterator<'a, T, MAX_RANK>
{
}
impl<'a, T: FloatConvertible, const MAX_RANK: usize> core::iter::FusedIterator
    for TensorSpanIterator<'a, T, MAX_RANK>
{
}

impl<'a, T: FloatConvertible, const MAX_RANK: usize> TensorSpanIterator<'a, T, MAX_RANK> {
    /// Adapt this iterator to yield only mutable dimension proxies, discarding positions.
    pub fn dims(self) -> TensorSpanDims<'a, T, MAX_RANK> { TensorSpanDims { inner: self } }
}

/// Mutable dimension-only adapter over [`TensorSpanIterator`], yielding [`DimMut`] without positions.
pub struct TensorSpanDims<'a, T: FloatConvertible, const MAX_RANK: usize = DEFAULT_MAX_RANK> {
    inner: TensorSpanIterator<'a, T, MAX_RANK>,
}

impl<'a, T: FloatConvertible, const MAX_RANK: usize> Iterator for TensorSpanDims<'a, T, MAX_RANK> {
    type Item = DimMut<'a, T>;

    #[inline]
    fn next(&mut self) -> Option<DimMut<'a, T>> { self.inner.next().map(|(_, v)| v) }

    fn size_hint(&self) -> (usize, Option<usize>) { self.inner.size_hint() }
}

impl<'a, T: FloatConvertible, const MAX_RANK: usize> ExactSizeIterator
    for TensorSpanDims<'a, T, MAX_RANK>
{
}
impl<'a, T: FloatConvertible, const MAX_RANK: usize> core::iter::FusedIterator
    for TensorSpanDims<'a, T, MAX_RANK>
{
}

// endregion: TensorSpanIterator

// region: Tensor iter() / iter_mut() methods

/// Helper to compute the logical shape and total element count for iteration.
///
/// Shape is already logical (sub-byte types store logical dimensions in shape),
/// so this simply copies the shape and computes the product.
fn logical_shape<const MAX_RANK: usize>(
    shape: &[usize; MAX_RANK],
    ndim: usize,
) -> ([usize; MAX_RANK], usize) {
    let logical = *shape;
    let mut total = 1usize;
    for &dim in &logical[..ndim] {
        total *= dim;
    }
    (logical, total)
}

impl<'a, T: FloatConvertible, const MAX_RANK: usize> TensorView<'a, T, MAX_RANK> {
    /// Returns a lazy iterator over all logical scalars in row-major order.
    ///
    /// Yields `(position, DimRef)` pairs. Use `.iter().dims()` for just dimensions.
    /// For sub-byte types, the innermost dimension is expanded.
    pub fn iter(&self) -> TensorViewIterator<'a, T, MAX_RANK> {
        let dims_per_value = T::dimensions_per_value();
        let (logical, total) = logical_shape::<MAX_RANK>(&self.shape, self.ndim);
        TensorViewIterator {
            data: self.data,
            shape: logical,
            strides: self.strides,
            ndim: self.ndim,
            dims_per_value,
            indices: [0; MAX_RANK],
            remaining: total,
            _marker: PhantomData,
        }
    }
}

impl<'a, T: FloatConvertible, const MAX_RANK: usize> TensorSpan<'a, T, MAX_RANK> {
    /// Returns a lazy iterator over all logical scalars in row-major order.
    ///
    /// Yields `(position, DimRef)` pairs. Use `.iter().dims()` for just dimensions.
    /// For sub-byte types, the innermost dimension is expanded.
    pub fn iter(&self) -> TensorViewIterator<'_, T, MAX_RANK> {
        let dims_per_value = T::dimensions_per_value();
        let (logical, total) = logical_shape::<MAX_RANK>(&self.shape, self.ndim);
        TensorViewIterator {
            data: self.data,
            shape: logical,
            strides: self.strides,
            ndim: self.ndim,
            dims_per_value,
            indices: [0; MAX_RANK],
            remaining: total,
            _marker: PhantomData,
        }
    }

    /// Returns a mutable iterator over all logical scalars in row-major order.
    ///
    /// Yields `(position, DimMut)` pairs. Use `.iter_mut().dims()` for just dimensions.
    pub fn iter_mut(&mut self) -> TensorSpanIterator<'_, T, MAX_RANK> {
        let dims_per_value = T::dimensions_per_value();
        let (logical, total) = logical_shape::<MAX_RANK>(&self.shape, self.ndim);
        TensorSpanIterator {
            data: self.data,
            shape: logical,
            strides: self.strides,
            ndim: self.ndim,
            dims_per_value,
            indices: [0; MAX_RANK],
            remaining: total,
            _marker: PhantomData,
        }
    }
}

impl<T: FloatConvertible, A: Allocator, const MAX_RANK: usize> Tensor<T, A, MAX_RANK> {
    /// Returns a lazy iterator over all logical scalars in row-major order.
    ///
    /// Yields `(position, DimRef)` pairs. Use `.iter().dims()` for just dimensions.
    pub fn iter(&self) -> TensorViewIterator<'_, T, MAX_RANK> { self.view().iter() }

    /// Returns a mutable iterator over all logical scalars in row-major order.
    ///
    /// Yields `(position, DimMut)` pairs. Use `.iter_mut().dims()` for just dimensions.
    pub fn iter_mut(&mut self) -> TensorSpanIterator<'_, T, MAX_RANK> {
        let dims_per_value = T::dimensions_per_value();
        let (logical, total) = logical_shape::<MAX_RANK>(&self.shape, self.ndim);
        TensorSpanIterator {
            data: self.data.as_ptr(),
            shape: logical,
            strides: self.strides,
            ndim: self.ndim,
            dims_per_value,
            indices: [0; MAX_RANK],
            remaining: total,
            _marker: PhantomData,
        }
    }
}

// endregion: Tensor iter() / iter_mut() methods

// region: IntoIterator (immutable)

impl<'a, T: FloatConvertible, A: Allocator, const MAX_RANK: usize> IntoIterator
    for &'a Tensor<T, A, MAX_RANK>
{
    type Item = ([usize; MAX_RANK], DimRef<'a, T>);
    type IntoIter = TensorViewIterator<'a, T, MAX_RANK>;
    fn into_iter(self) -> Self::IntoIter { self.iter() }
}

impl<'a, T: FloatConvertible, const MAX_RANK: usize> IntoIterator
    for &'a TensorView<'a, T, MAX_RANK>
{
    type Item = ([usize; MAX_RANK], DimRef<'a, T>);
    type IntoIter = TensorViewIterator<'a, T, MAX_RANK>;
    fn into_iter(self) -> Self::IntoIter { self.iter() }
}

impl<'a, T: FloatConvertible, const MAX_RANK: usize> IntoIterator
    for &'a TensorSpan<'a, T, MAX_RANK>
{
    type Item = ([usize; MAX_RANK], DimRef<'a, T>);
    type IntoIter = TensorViewIterator<'a, T, MAX_RANK>;
    fn into_iter(self) -> Self::IntoIter { self.iter() }
}

// endregion: IntoIterator (immutable)

// region: IntoIterator (mutable)

impl<'a, T: FloatConvertible, A: Allocator, const MAX_RANK: usize> IntoIterator
    for &'a mut Tensor<T, A, MAX_RANK>
{
    type Item = ([usize; MAX_RANK], DimMut<'a, T>);
    type IntoIter = TensorSpanIterator<'a, T, MAX_RANK>;
    fn into_iter(self) -> Self::IntoIter { self.iter_mut() }
}

impl<'a, T: FloatConvertible, const MAX_RANK: usize> IntoIterator
    for &'a mut TensorSpan<'a, T, MAX_RANK>
{
    type Item = ([usize; MAX_RANK], DimMut<'a, T>);
    type IntoIter = TensorSpanIterator<'a, T, MAX_RANK>;
    fn into_iter(self) -> Self::IntoIter { self.iter_mut() }
}

// endregion: IntoIterator (mutable)

// region: PartialEq

impl<T: PartialEq, A: Allocator, const MAX_RANK: usize> PartialEq for Tensor<T, A, MAX_RANK> {
    fn eq(&self, other: &Self) -> bool {
        self.ndim == other.ndim
            && self.shape[..self.ndim] == other.shape[..other.ndim]
            && self.as_slice() == other.as_slice()
    }
}

impl<'a, T: FloatConvertible, const MAX_RANK: usize> PartialEq for TensorView<'a, T, MAX_RANK>
where
    T::DimScalar: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.ndim == other.ndim
            && self.shape[..self.ndim] == other.shape[..other.ndim]
            && self
                .iter()
                .dims()
                .zip(other.iter().dims())
                .all(|(a, b)| *a == *b)
    }
}

impl<'a, T: FloatConvertible, const MAX_RANK: usize> PartialEq for TensorSpan<'a, T, MAX_RANK>
where
    T::DimScalar: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.ndim == other.ndim
            && self.shape[..self.ndim] == other.shape[..other.ndim]
            && self
                .iter()
                .dims()
                .zip(other.iter().dims())
                .all(|(a, b)| *a == *b)
    }
}

// endregion: PartialEq

// region: Tolerance Equality

/// Extension trait: tolerance-based equality for any [`TensorRef`] implementor.
///
/// Uses the formula `|a - b| <= atol + rtol * |b|` per element.
/// Returns `false` if shapes differ.
pub trait AllCloseOps<T: FloatConvertible, const MAX_RANK: usize>: TensorRef<T, MAX_RANK>
where
    T::DimScalar: NumberLike,
{
    fn allclose(
        &self,
        other: &(impl TensorRef<T, MAX_RANK> + ?Sized),
        atol: f64,
        rtol: f64,
    ) -> bool {
        let a = self.view();
        let b = other.view();
        a.ndim() == b.ndim()
            && a.shape() == b.shape()
            && a.iter()
                .dims()
                .zip(b.iter().dims())
                .all(|(x, y)| crate::types::is_close((*x).to_f64(), (*y).to_f64(), atol, rtol))
    }
}

impl<C, T: FloatConvertible, const R: usize> AllCloseOps<T, R> for C
where
    C: TensorRef<T, R>,
    T::DimScalar: NumberLike,
{
}

// endregion: Tolerance Equality

// region: AsRef

impl<T, A: Allocator, const MAX_RANK: usize> AsRef<[T]> for Tensor<T, A, MAX_RANK> {
    fn as_ref(&self) -> &[T] { self.as_slice() }
}

// endregion: AsRef

// region: Tensor View and Slice Methods

impl<T, A: Allocator, const MAX_RANK: usize> Tensor<T, A, MAX_RANK> {
    /// Create a view of the entire array.
    pub fn view(&self) -> TensorView<'_, T, MAX_RANK> {
        TensorView {
            data: self.data.as_ptr(),
            shape: self.shape,
            strides: self.strides,
            ndim: self.ndim,
            _marker: PhantomData,
        }
    }

    /// Create a mutable span of the entire tensor.
    pub fn span(&mut self) -> TensorSpan<'_, T, MAX_RANK> {
        TensorSpan {
            data: self.data.as_ptr(),
            shape: self.shape,
            strides: self.strides,
            ndim: self.ndim,
            _marker: PhantomData,
        }
    }

    /// Try to get an element by flat logical row-major index.
    pub fn try_flat<I: VectorIndex>(&self, index: I) -> Result<&T, TensorError> {
        if self.ndim == 0 {
            return Err(TensorError::DimensionMismatch {
                expected: 1,
                got: 0,
            });
        }
        let logical_index = resolve_index_for_size_(index, self.numel())?;
        let offset = offset_from_flat_(&self.shape, &self.strides, self.ndim, logical_index);
        Ok(unsafe { &*((self.data.as_ptr() as *const u8).offset(offset) as *const T) })
    }

    /// Try to get a mutable element by flat logical row-major index.
    pub fn try_flat_mut<I: VectorIndex>(&mut self, index: I) -> Result<&mut T, TensorError> {
        if self.ndim == 0 {
            return Err(TensorError::DimensionMismatch {
                expected: 1,
                got: 0,
            });
        }
        let logical_index = resolve_index_for_size_(index, self.numel())?;
        let offset = offset_from_flat_(&self.shape, &self.strides, self.ndim, logical_index);
        Ok(unsafe { &mut *((self.data.as_ptr() as *mut u8).offset(offset) as *mut T) })
    }

    /// Try to get an element by exact coordinates.
    pub fn try_coords<C: TensorCoordinates>(&self, coords: C) -> Result<&T, TensorError> {
        let resolved = coords.resolve(&self.shape, self.ndim)?;
        let offset = offset_from_coords_(&self.strides, &resolved, self.ndim);
        Ok(unsafe { &*((self.data.as_ptr() as *const u8).offset(offset) as *const T) })
    }

    /// Try to get a mutable element by exact coordinates.
    pub fn try_coords_mut<C: TensorCoordinates>(
        &mut self,
        coords: C,
    ) -> Result<&mut T, TensorError> {
        let resolved = coords.resolve(&self.shape, self.ndim)?;
        let offset = offset_from_coords_(&self.strides, &resolved, self.ndim);
        Ok(unsafe { &mut *((self.data.as_ptr() as *mut u8).offset(offset) as *mut T) })
    }

    /// Try to access the scalar value of a rank-0 tensor.
    pub fn try_scalar(&self) -> Result<&T, TensorError> {
        if self.ndim != 0 {
            return Err(TensorError::DimensionMismatch {
                expected: 0,
                got: self.ndim,
            });
        }
        Ok(unsafe { &*self.data.as_ptr() })
    }

    /// Try to access the mutable scalar value of a rank-0 tensor.
    pub fn try_scalar_mut(&mut self) -> Result<&mut T, TensorError> {
        if self.ndim != 0 {
            return Err(TensorError::DimensionMismatch {
                expected: 0,
                got: self.ndim,
            });
        }
        Ok(unsafe { &mut *self.data.as_ptr() })
    }

    /// Slice the leading axis by one index, reducing rank by one.
    pub fn slice_leading<I: VectorIndex>(
        &self,
        index: I,
    ) -> Result<TensorView<'_, T, MAX_RANK>, TensorError> {
        let (shape, strides, ndim, offset, _) =
            slice_leading_layout_(&self.shape, &self.strides, self.ndim, index)?;
        Ok(TensorView {
            data: unsafe { (self.data.as_ptr() as *const u8).offset(offset) as *const T },
            shape,
            strides,
            ndim,
            _marker: PhantomData,
        })
    }

    /// Slice the leading axis mutably by one index, reducing rank by one.
    pub fn slice_leading_mut<I: VectorIndex>(
        &mut self,
        index: I,
    ) -> Result<TensorSpan<'_, T, MAX_RANK>, TensorError> {
        let (shape, strides, ndim, offset, _) =
            slice_leading_layout_(&self.shape, &self.strides, self.ndim, index)?;
        Ok(TensorSpan {
            data: unsafe { (self.data.as_ptr() as *mut u8).offset(offset) as *mut T },
            shape,
            strides,
            ndim,
            _marker: PhantomData,
        })
    }

    /// Slice the array along multiple dimensions.
    ///
    /// Accepts tuples of Rust range types or `&[SliceRange]`:
    /// ```ignore
    /// use numkong::{Tensor, SliceRange};
    ///
    /// let arr = Tensor::<f32>::try_full(&[4, 5], 1.0).unwrap();
    ///
    /// // Tuple syntax
    /// let view = arr.slice((0..2_usize, ..)).unwrap();
    /// let row = arr.slice((1_usize, ..)).unwrap();
    ///
    /// // Old syntax still works
    /// let view = arr.slice(&[SliceRange::range(0, 2), SliceRange::full()]).unwrap();
    /// ```
    pub fn slice(&self, spec: impl SliceSpec) -> Result<TensorView<'_, T, MAX_RANK>, TensorError> {
        self.view().slice(spec)
    }

    /// Slice the array mutably along multiple dimensions.
    ///
    /// Accepts tuples of Rust range types or `&[SliceRange]`.
    pub fn slice_mut(
        &mut self,
        spec: impl SliceSpec,
    ) -> Result<TensorSpan<'_, T, MAX_RANK>, TensorError> {
        let (shape, strides, ndim, offset, _) =
            spec.apply_layout(&self.shape, &self.strides, self.ndim)?;
        Ok(TensorSpan {
            data: unsafe { (self.data.as_ptr() as *mut u8).offset(offset) as *mut T },
            shape,
            strides,
            ndim,
            _marker: PhantomData,
        })
    }
}

impl<T: StorageElement, A: Allocator, const MAX_RANK: usize> Tensor<T, A, MAX_RANK> {
    /// Transpose (reverse all dimensions, no data copy).
    pub fn transpose(&self) -> Result<TensorView<'_, T, MAX_RANK>, TensorError> {
        self.view().transpose()
    }

    /// Reshape the array (must have same total elements, contiguous only).
    pub fn reshape(&self, new_shape: &[usize]) -> Result<TensorView<'_, T, MAX_RANK>, TensorError> {
        self.view().reshape(new_shape)
    }

    /// Flatten to 1D (requires contiguous layout).
    pub fn flatten(&self) -> Result<TensorView<'_, T, MAX_RANK>, TensorError> {
        self.view().flatten()
    }

    /// Remove dimensions of size 1.
    pub fn squeeze(&self) -> TensorView<'_, T, MAX_RANK> { self.view().squeeze() }
}

// endregion: Tensor View and Slice Methods

impl<'a, I: VectorIndex, T, const MAX_RANK: usize> Index<I> for TensorView<'a, T, MAX_RANK> {
    type Output = T;

    fn index(&self, index: I) -> &Self::Output {
        self.try_flat(index).expect("view index out of bounds")
    }
}

impl<'a, I0: VectorIndex, I1: VectorIndex, T, const MAX_RANK: usize> Index<(I0, I1)>
    for TensorView<'a, T, MAX_RANK>
{
    type Output = T;

    fn index(&self, index: (I0, I1)) -> &Self::Output {
        self.try_coords(index)
            .expect("view coordinates out of bounds")
    }
}

impl<'a, I0: VectorIndex, I1: VectorIndex, I2: VectorIndex, T, const MAX_RANK: usize>
    Index<(I0, I1, I2)> for TensorView<'a, T, MAX_RANK>
{
    type Output = T;

    fn index(&self, index: (I0, I1, I2)) -> &Self::Output {
        self.try_coords(index)
            .expect("view coordinates out of bounds")
    }
}

impl<
        'a,
        I0: VectorIndex,
        I1: VectorIndex,
        I2: VectorIndex,
        I3: VectorIndex,
        T,
        const MAX_RANK: usize,
    > Index<(I0, I1, I2, I3)> for TensorView<'a, T, MAX_RANK>
{
    type Output = T;

    fn index(&self, index: (I0, I1, I2, I3)) -> &Self::Output {
        self.try_coords(index)
            .expect("view coordinates out of bounds")
    }
}

impl<
        'a,
        I0: VectorIndex,
        I1: VectorIndex,
        I2: VectorIndex,
        I3: VectorIndex,
        I4: VectorIndex,
        T,
        const MAX_RANK: usize,
    > Index<(I0, I1, I2, I3, I4)> for TensorView<'a, T, MAX_RANK>
{
    type Output = T;

    fn index(&self, index: (I0, I1, I2, I3, I4)) -> &Self::Output {
        self.try_coords(index)
            .expect("view coordinates out of bounds")
    }
}

impl<
        'a,
        I0: VectorIndex,
        I1: VectorIndex,
        I2: VectorIndex,
        I3: VectorIndex,
        I4: VectorIndex,
        I5: VectorIndex,
        T,
        const MAX_RANK: usize,
    > Index<(I0, I1, I2, I3, I4, I5)> for TensorView<'a, T, MAX_RANK>
{
    type Output = T;

    fn index(&self, index: (I0, I1, I2, I3, I4, I5)) -> &Self::Output {
        self.try_coords(index)
            .expect("view coordinates out of bounds")
    }
}

impl<
        'a,
        I0: VectorIndex,
        I1: VectorIndex,
        I2: VectorIndex,
        I3: VectorIndex,
        I4: VectorIndex,
        I5: VectorIndex,
        I6: VectorIndex,
        T,
        const MAX_RANK: usize,
    > Index<(I0, I1, I2, I3, I4, I5, I6)> for TensorView<'a, T, MAX_RANK>
{
    type Output = T;

    fn index(&self, index: (I0, I1, I2, I3, I4, I5, I6)) -> &Self::Output {
        self.try_coords(index)
            .expect("view coordinates out of bounds")
    }
}

impl<
        'a,
        I0: VectorIndex,
        I1: VectorIndex,
        I2: VectorIndex,
        I3: VectorIndex,
        I4: VectorIndex,
        I5: VectorIndex,
        I6: VectorIndex,
        I7: VectorIndex,
        T,
        const MAX_RANK: usize,
    > Index<(I0, I1, I2, I3, I4, I5, I6, I7)> for TensorView<'a, T, MAX_RANK>
{
    type Output = T;

    fn index(&self, index: (I0, I1, I2, I3, I4, I5, I6, I7)) -> &Self::Output {
        self.try_coords(index)
            .expect("view coordinates out of bounds")
    }
}

impl<'a, I: VectorIndex, T, const MAX_RANK: usize> Index<I> for TensorSpan<'a, T, MAX_RANK> {
    type Output = T;

    fn index(&self, index: I) -> &Self::Output {
        self.try_flat(index).expect("span index out of bounds")
    }
}

impl<'a, I: VectorIndex, T, const MAX_RANK: usize> IndexMut<I> for TensorSpan<'a, T, MAX_RANK> {
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        self.try_flat_mut(index).expect("span index out of bounds")
    }
}

impl<'a, I0: VectorIndex, I1: VectorIndex, T, const MAX_RANK: usize> Index<(I0, I1)>
    for TensorSpan<'a, T, MAX_RANK>
{
    type Output = T;

    fn index(&self, index: (I0, I1)) -> &Self::Output {
        self.try_coords(index)
            .expect("span coordinates out of bounds")
    }
}

impl<'a, I0: VectorIndex, I1: VectorIndex, T, const MAX_RANK: usize> IndexMut<(I0, I1)>
    for TensorSpan<'a, T, MAX_RANK>
{
    fn index_mut(&mut self, index: (I0, I1)) -> &mut Self::Output {
        self.try_coords_mut(index)
            .expect("span coordinates out of bounds")
    }
}

impl<'a, I0: VectorIndex, I1: VectorIndex, I2: VectorIndex, T, const MAX_RANK: usize>
    Index<(I0, I1, I2)> for TensorSpan<'a, T, MAX_RANK>
{
    type Output = T;

    fn index(&self, index: (I0, I1, I2)) -> &Self::Output {
        self.try_coords(index)
            .expect("span coordinates out of bounds")
    }
}

impl<'a, I0: VectorIndex, I1: VectorIndex, I2: VectorIndex, T, const MAX_RANK: usize>
    IndexMut<(I0, I1, I2)> for TensorSpan<'a, T, MAX_RANK>
{
    fn index_mut(&mut self, index: (I0, I1, I2)) -> &mut Self::Output {
        self.try_coords_mut(index)
            .expect("span coordinates out of bounds")
    }
}

impl<
        'a,
        I0: VectorIndex,
        I1: VectorIndex,
        I2: VectorIndex,
        I3: VectorIndex,
        T,
        const MAX_RANK: usize,
    > Index<(I0, I1, I2, I3)> for TensorSpan<'a, T, MAX_RANK>
{
    type Output = T;

    fn index(&self, index: (I0, I1, I2, I3)) -> &Self::Output {
        self.try_coords(index)
            .expect("span coordinates out of bounds")
    }
}

impl<
        'a,
        I0: VectorIndex,
        I1: VectorIndex,
        I2: VectorIndex,
        I3: VectorIndex,
        T,
        const MAX_RANK: usize,
    > IndexMut<(I0, I1, I2, I3)> for TensorSpan<'a, T, MAX_RANK>
{
    fn index_mut(&mut self, index: (I0, I1, I2, I3)) -> &mut Self::Output {
        self.try_coords_mut(index)
            .expect("span coordinates out of bounds")
    }
}

impl<
        'a,
        I0: VectorIndex,
        I1: VectorIndex,
        I2: VectorIndex,
        I3: VectorIndex,
        I4: VectorIndex,
        T,
        const MAX_RANK: usize,
    > Index<(I0, I1, I2, I3, I4)> for TensorSpan<'a, T, MAX_RANK>
{
    type Output = T;

    fn index(&self, index: (I0, I1, I2, I3, I4)) -> &Self::Output {
        self.try_coords(index)
            .expect("span coordinates out of bounds")
    }
}

impl<
        'a,
        I0: VectorIndex,
        I1: VectorIndex,
        I2: VectorIndex,
        I3: VectorIndex,
        I4: VectorIndex,
        T,
        const MAX_RANK: usize,
    > IndexMut<(I0, I1, I2, I3, I4)> for TensorSpan<'a, T, MAX_RANK>
{
    fn index_mut(&mut self, index: (I0, I1, I2, I3, I4)) -> &mut Self::Output {
        self.try_coords_mut(index)
            .expect("span coordinates out of bounds")
    }
}

impl<
        'a,
        I0: VectorIndex,
        I1: VectorIndex,
        I2: VectorIndex,
        I3: VectorIndex,
        I4: VectorIndex,
        I5: VectorIndex,
        T,
        const MAX_RANK: usize,
    > Index<(I0, I1, I2, I3, I4, I5)> for TensorSpan<'a, T, MAX_RANK>
{
    type Output = T;

    fn index(&self, index: (I0, I1, I2, I3, I4, I5)) -> &Self::Output {
        self.try_coords(index)
            .expect("span coordinates out of bounds")
    }
}

impl<
        'a,
        I0: VectorIndex,
        I1: VectorIndex,
        I2: VectorIndex,
        I3: VectorIndex,
        I4: VectorIndex,
        I5: VectorIndex,
        T,
        const MAX_RANK: usize,
    > IndexMut<(I0, I1, I2, I3, I4, I5)> for TensorSpan<'a, T, MAX_RANK>
{
    fn index_mut(&mut self, index: (I0, I1, I2, I3, I4, I5)) -> &mut Self::Output {
        self.try_coords_mut(index)
            .expect("span coordinates out of bounds")
    }
}

impl<
        'a,
        I0: VectorIndex,
        I1: VectorIndex,
        I2: VectorIndex,
        I3: VectorIndex,
        I4: VectorIndex,
        I5: VectorIndex,
        I6: VectorIndex,
        T,
        const MAX_RANK: usize,
    > Index<(I0, I1, I2, I3, I4, I5, I6)> for TensorSpan<'a, T, MAX_RANK>
{
    type Output = T;

    fn index(&self, index: (I0, I1, I2, I3, I4, I5, I6)) -> &Self::Output {
        self.try_coords(index)
            .expect("span coordinates out of bounds")
    }
}

impl<
        'a,
        I0: VectorIndex,
        I1: VectorIndex,
        I2: VectorIndex,
        I3: VectorIndex,
        I4: VectorIndex,
        I5: VectorIndex,
        I6: VectorIndex,
        T,
        const MAX_RANK: usize,
    > IndexMut<(I0, I1, I2, I3, I4, I5, I6)> for TensorSpan<'a, T, MAX_RANK>
{
    fn index_mut(&mut self, index: (I0, I1, I2, I3, I4, I5, I6)) -> &mut Self::Output {
        self.try_coords_mut(index)
            .expect("span coordinates out of bounds")
    }
}

impl<
        'a,
        I0: VectorIndex,
        I1: VectorIndex,
        I2: VectorIndex,
        I3: VectorIndex,
        I4: VectorIndex,
        I5: VectorIndex,
        I6: VectorIndex,
        I7: VectorIndex,
        T,
        const MAX_RANK: usize,
    > Index<(I0, I1, I2, I3, I4, I5, I6, I7)> for TensorSpan<'a, T, MAX_RANK>
{
    type Output = T;

    fn index(&self, index: (I0, I1, I2, I3, I4, I5, I6, I7)) -> &Self::Output {
        self.try_coords(index)
            .expect("span coordinates out of bounds")
    }
}

impl<
        'a,
        I0: VectorIndex,
        I1: VectorIndex,
        I2: VectorIndex,
        I3: VectorIndex,
        I4: VectorIndex,
        I5: VectorIndex,
        I6: VectorIndex,
        I7: VectorIndex,
        T,
        const MAX_RANK: usize,
    > IndexMut<(I0, I1, I2, I3, I4, I5, I6, I7)> for TensorSpan<'a, T, MAX_RANK>
{
    fn index_mut(&mut self, index: (I0, I1, I2, I3, I4, I5, I6, I7)) -> &mut Self::Output {
        self.try_coords_mut(index)
            .expect("span coordinates out of bounds")
    }
}

impl<I: VectorIndex, T, A: Allocator, const MAX_RANK: usize> Index<I> for Tensor<T, A, MAX_RANK> {
    type Output = T;

    fn index(&self, index: I) -> &Self::Output {
        self.try_flat(index).expect("tensor index out of bounds")
    }
}

impl<I: VectorIndex, T, A: Allocator, const MAX_RANK: usize> IndexMut<I>
    for Tensor<T, A, MAX_RANK>
{
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        self.try_flat_mut(index)
            .expect("tensor index out of bounds")
    }
}

impl<I0: VectorIndex, I1: VectorIndex, T, A: Allocator, const MAX_RANK: usize> Index<(I0, I1)>
    for Tensor<T, A, MAX_RANK>
{
    type Output = T;

    fn index(&self, index: (I0, I1)) -> &Self::Output {
        self.try_coords(index)
            .expect("tensor coordinates out of bounds")
    }
}

impl<I0: VectorIndex, I1: VectorIndex, T, A: Allocator, const MAX_RANK: usize> IndexMut<(I0, I1)>
    for Tensor<T, A, MAX_RANK>
{
    fn index_mut(&mut self, index: (I0, I1)) -> &mut Self::Output {
        self.try_coords_mut(index)
            .expect("tensor coordinates out of bounds")
    }
}

impl<I0: VectorIndex, I1: VectorIndex, I2: VectorIndex, T, A: Allocator, const MAX_RANK: usize>
    Index<(I0, I1, I2)> for Tensor<T, A, MAX_RANK>
{
    type Output = T;

    fn index(&self, index: (I0, I1, I2)) -> &Self::Output {
        self.try_coords(index)
            .expect("tensor coordinates out of bounds")
    }
}

impl<I0: VectorIndex, I1: VectorIndex, I2: VectorIndex, T, A: Allocator, const MAX_RANK: usize>
    IndexMut<(I0, I1, I2)> for Tensor<T, A, MAX_RANK>
{
    fn index_mut(&mut self, index: (I0, I1, I2)) -> &mut Self::Output {
        self.try_coords_mut(index)
            .expect("tensor coordinates out of bounds")
    }
}

impl<
        I0: VectorIndex,
        I1: VectorIndex,
        I2: VectorIndex,
        I3: VectorIndex,
        T,
        A: Allocator,
        const MAX_RANK: usize,
    > Index<(I0, I1, I2, I3)> for Tensor<T, A, MAX_RANK>
{
    type Output = T;

    fn index(&self, index: (I0, I1, I2, I3)) -> &Self::Output {
        self.try_coords(index)
            .expect("tensor coordinates out of bounds")
    }
}

impl<
        I0: VectorIndex,
        I1: VectorIndex,
        I2: VectorIndex,
        I3: VectorIndex,
        T,
        A: Allocator,
        const MAX_RANK: usize,
    > IndexMut<(I0, I1, I2, I3)> for Tensor<T, A, MAX_RANK>
{
    fn index_mut(&mut self, index: (I0, I1, I2, I3)) -> &mut Self::Output {
        self.try_coords_mut(index)
            .expect("tensor coordinates out of bounds")
    }
}

impl<
        I0: VectorIndex,
        I1: VectorIndex,
        I2: VectorIndex,
        I3: VectorIndex,
        I4: VectorIndex,
        T,
        A: Allocator,
        const MAX_RANK: usize,
    > Index<(I0, I1, I2, I3, I4)> for Tensor<T, A, MAX_RANK>
{
    type Output = T;

    fn index(&self, index: (I0, I1, I2, I3, I4)) -> &Self::Output {
        self.try_coords(index)
            .expect("tensor coordinates out of bounds")
    }
}

impl<
        I0: VectorIndex,
        I1: VectorIndex,
        I2: VectorIndex,
        I3: VectorIndex,
        I4: VectorIndex,
        T,
        A: Allocator,
        const MAX_RANK: usize,
    > IndexMut<(I0, I1, I2, I3, I4)> for Tensor<T, A, MAX_RANK>
{
    fn index_mut(&mut self, index: (I0, I1, I2, I3, I4)) -> &mut Self::Output {
        self.try_coords_mut(index)
            .expect("tensor coordinates out of bounds")
    }
}

impl<
        I0: VectorIndex,
        I1: VectorIndex,
        I2: VectorIndex,
        I3: VectorIndex,
        I4: VectorIndex,
        I5: VectorIndex,
        T,
        A: Allocator,
        const MAX_RANK: usize,
    > Index<(I0, I1, I2, I3, I4, I5)> for Tensor<T, A, MAX_RANK>
{
    type Output = T;

    fn index(&self, index: (I0, I1, I2, I3, I4, I5)) -> &Self::Output {
        self.try_coords(index)
            .expect("tensor coordinates out of bounds")
    }
}

impl<
        I0: VectorIndex,
        I1: VectorIndex,
        I2: VectorIndex,
        I3: VectorIndex,
        I4: VectorIndex,
        I5: VectorIndex,
        T,
        A: Allocator,
        const MAX_RANK: usize,
    > IndexMut<(I0, I1, I2, I3, I4, I5)> for Tensor<T, A, MAX_RANK>
{
    fn index_mut(&mut self, index: (I0, I1, I2, I3, I4, I5)) -> &mut Self::Output {
        self.try_coords_mut(index)
            .expect("tensor coordinates out of bounds")
    }
}

impl<
        I0: VectorIndex,
        I1: VectorIndex,
        I2: VectorIndex,
        I3: VectorIndex,
        I4: VectorIndex,
        I5: VectorIndex,
        I6: VectorIndex,
        T,
        A: Allocator,
        const MAX_RANK: usize,
    > Index<(I0, I1, I2, I3, I4, I5, I6)> for Tensor<T, A, MAX_RANK>
{
    type Output = T;

    fn index(&self, index: (I0, I1, I2, I3, I4, I5, I6)) -> &Self::Output {
        self.try_coords(index)
            .expect("tensor coordinates out of bounds")
    }
}

impl<
        I0: VectorIndex,
        I1: VectorIndex,
        I2: VectorIndex,
        I3: VectorIndex,
        I4: VectorIndex,
        I5: VectorIndex,
        I6: VectorIndex,
        T,
        A: Allocator,
        const MAX_RANK: usize,
    > IndexMut<(I0, I1, I2, I3, I4, I5, I6)> for Tensor<T, A, MAX_RANK>
{
    fn index_mut(&mut self, index: (I0, I1, I2, I3, I4, I5, I6)) -> &mut Self::Output {
        self.try_coords_mut(index)
            .expect("tensor coordinates out of bounds")
    }
}

impl<
        I0: VectorIndex,
        I1: VectorIndex,
        I2: VectorIndex,
        I3: VectorIndex,
        I4: VectorIndex,
        I5: VectorIndex,
        I6: VectorIndex,
        I7: VectorIndex,
        T,
        A: Allocator,
        const MAX_RANK: usize,
    > Index<(I0, I1, I2, I3, I4, I5, I6, I7)> for Tensor<T, A, MAX_RANK>
{
    type Output = T;

    fn index(&self, index: (I0, I1, I2, I3, I4, I5, I6, I7)) -> &Self::Output {
        self.try_coords(index)
            .expect("tensor coordinates out of bounds")
    }
}

impl<
        I0: VectorIndex,
        I1: VectorIndex,
        I2: VectorIndex,
        I3: VectorIndex,
        I4: VectorIndex,
        I5: VectorIndex,
        I6: VectorIndex,
        I7: VectorIndex,
        T,
        A: Allocator,
        const MAX_RANK: usize,
    > IndexMut<(I0, I1, I2, I3, I4, I5, I6, I7)> for Tensor<T, A, MAX_RANK>
{
    fn index_mut(&mut self, index: (I0, I1, I2, I3, I4, I5, I6, I7)) -> &mut Self::Output {
        self.try_coords_mut(index)
            .expect("tensor coordinates out of bounds")
    }
}

// region: Type Aliases

/// Type alias for a 2D matrix (Tensor with MAX_RANK=2).
pub type Matrix<T, A = Global> = Tensor<T, A, 2>;

/// Type alias for an immutable 2D matrix view.
pub type MatrixView<'a, T> = TensorView<'a, T, 2>;

/// Type alias for a mutable 2D matrix view.
pub type MatrixSpan<'a, T> = TensorSpan<'a, T, 2>;

// endregion: Type Aliases

// region: Tensor Internal Helpers

#[inline]
fn validate_same_shape(lhs: &[usize], rhs: &[usize]) -> Result<(), TensorError> {
    if lhs == rhs {
        return Ok(());
    }
    if lhs.len() != rhs.len() {
        return Err(TensorError::DimensionMismatch {
            expected: lhs.len(),
            got: rhs.len(),
        });
    }
    for (i, (&l, &r)) in lhs.iter().zip(rhs).enumerate() {
        if l != r {
            return Err(TensorError::ShapeMismatch {
                axis: i,
                expected: l,
                got: r,
            });
        }
    }
    Ok(())
}

#[inline]
fn normalize_axis<I: VectorIndex>(axis: I, ndim: usize) -> Result<usize, TensorError> {
    if ndim == 0 {
        return Err(TensorError::IndexOutOfBounds {
            index: 0,
            size: ndim,
        });
    }
    axis.resolve(ndim).ok_or(TensorError::IndexOutOfBounds {
        index: 0,
        size: ndim,
    })
}

#[inline]
fn compute_strides_into_<T, const MAX_RANK: usize>(
    shape: &[usize],
    strides: &mut [isize; MAX_RANK],
) {
    let elem_size = core::mem::size_of::<T>();
    if shape.is_empty() {
        return;
    }

    let mut stride = elem_size as isize;
    for dim in (0..shape.len()).rev() {
        strides[dim] = stride;
        stride *= shape[dim] as isize;
    }
}

/// Compute the transposed layout (reverse all dimensions).
fn transpose_layout<const R: usize>(
    shape: &[usize; R],
    strides: &[isize; R],
    ndim: usize,
) -> ([usize; R], [isize; R]) {
    let mut new_shape = [0usize; R];
    let mut new_strides = [0isize; R];
    for i in 0..ndim {
        new_shape[i] = shape[ndim - 1 - i];
        new_strides[i] = strides[ndim - 1 - i];
    }
    (new_shape, new_strides)
}

/// Compute the reshaped layout (same total elements, contiguous only).
fn reshape_layout<T, const R: usize>(
    shape: &[usize; R],
    strides: &[isize; R],
    ndim: usize,
    len: usize,
    new_shape: &[usize],
) -> Result<([usize; R], [isize; R], usize), TensorError> {
    if new_shape.len() > R {
        return Err(TensorError::TooManyRanks {
            got: new_shape.len(),
        });
    }
    let new_len: usize = new_shape.iter().product();
    if new_len != len {
        return Err(TensorError::ShapeMismatch {
            axis: 0,
            expected: new_len,
            got: len,
        });
    }
    // Check contiguous
    let elem_size = core::mem::size_of::<T>() as isize;
    let mut expected = elem_size;
    for i in (0..ndim).rev() {
        if strides[i] != expected {
            return Err(TensorError::NonContiguousRows);
        }
        expected *= shape[i] as isize;
    }
    let mut shape_arr = [0usize; R];
    shape_arr[..new_shape.len()].copy_from_slice(new_shape);
    let mut strides_arr = [0isize; R];
    compute_strides_into_::<T, R>(new_shape, &mut strides_arr);
    Ok((shape_arr, strides_arr, new_shape.len()))
}

/// Compute the squeezed layout (remove dimensions of size 1).
fn squeeze_layout<T, const R: usize>(
    shape: &[usize; R],
    strides: &[isize; R],
    ndim: usize,
) -> ([usize; R], [isize; R], usize) {
    let mut new_shape = [0usize; R];
    let mut new_strides = [0isize; R];
    let mut new_ndim = 0;
    for i in 0..ndim {
        if shape[i] != 1 {
            new_shape[new_ndim] = shape[i];
            new_strides[new_ndim] = strides[i];
            new_ndim += 1;
        }
    }
    if new_ndim == 0 {
        new_ndim = 1;
        new_shape[0] = 1;
        new_strides[0] = core::mem::size_of::<T>() as isize;
    }
    (new_shape, new_strides, new_ndim)
}

#[inline]
fn resolve_index_for_size_<I: VectorIndex>(index: I, size: usize) -> Result<usize, TensorError> {
    index
        .resolve(size)
        .ok_or(TensorError::IndexOutOfBounds { index: 0, size })
}

#[inline]
fn offset_from_coords_<const MAX_RANK: usize>(
    strides: &[isize; MAX_RANK],
    coords: &[usize; MAX_RANK],
    ndim: usize,
) -> isize {
    let mut offset = 0isize;
    for dim in 0..ndim {
        offset += coords[dim] as isize * strides[dim];
    }
    offset
}

#[inline]
fn offset_from_flat_<const MAX_RANK: usize>(
    shape: &[usize; MAX_RANK],
    strides: &[isize; MAX_RANK],
    ndim: usize,
    mut flat_index: usize,
) -> isize {
    let mut offset = 0isize;
    for dim in (0..ndim).rev() {
        let dim_size = shape[dim];
        let coord = flat_index % dim_size;
        flat_index /= dim_size;
        offset += coord as isize * strides[dim];
    }
    offset
}

fn slice_layout_<const MAX_RANK: usize>(
    shape: &[usize; MAX_RANK],
    strides: &[isize; MAX_RANK],
    ndim: usize,
    ranges: &[SliceRange],
) -> LayoutResult<MAX_RANK> {
    if ranges.len() != ndim {
        return Err(TensorError::DimensionMismatch {
            expected: ndim,
            got: ranges.len(),
        });
    }

    let mut new_shape = [0usize; MAX_RANK];
    let mut new_strides = [0isize; MAX_RANK];
    let mut new_ndim = 0usize;
    let mut offset = 0isize;

    for (dim, range) in ranges.iter().enumerate() {
        let dim_size = shape[dim];
        let dim_stride = strides[dim];

        match *range {
            SliceRange::Full => {
                new_shape[new_ndim] = dim_size;
                new_strides[new_ndim] = dim_stride;
                new_ndim += 1;
            }
            SliceRange::Index(index) => {
                if index >= dim_size {
                    return Err(TensorError::IndexOutOfBounds {
                        index,
                        size: dim_size,
                    });
                }
                offset += index as isize * dim_stride;
            }
            SliceRange::Range { start, end } => {
                if start > end || end > dim_size {
                    return Err(TensorError::IndexOutOfBounds {
                        index: end,
                        size: dim_size,
                    });
                }
                new_shape[new_ndim] = end - start;
                new_strides[new_ndim] = dim_stride;
                new_ndim += 1;
                offset += start as isize * dim_stride;
            }
            SliceRange::RangeStep { start, end, step } => {
                if start >= dim_size || (end > dim_size && step > 0) {
                    return Err(TensorError::IndexOutOfBounds {
                        index: if start >= dim_size { start } else { end },
                        size: dim_size,
                    });
                }
                if step == 0 {
                    return Err(TensorError::InvalidShape {
                        axis: dim,
                        size: 0,
                        reason: "step cannot be zero",
                    });
                }
                let count = if step > 0 {
                    (end.saturating_sub(start) + step as usize - 1) / step as usize
                } else {
                    let abs_step = (-step) as usize;
                    (start.saturating_sub(end) + abs_step - 1) / abs_step
                };
                new_shape[new_ndim] = count;
                new_strides[new_ndim] = dim_stride * step;
                new_ndim += 1;
                offset += start as isize * dim_stride;
            }
        }
    }

    let new_len = if new_ndim == 0 {
        1
    } else {
        new_shape[..new_ndim].iter().product()
    };
    Ok((new_shape, new_strides, new_ndim, offset, new_len))
}

fn slice_leading_layout_<I: VectorIndex, const MAX_RANK: usize>(
    shape: &[usize; MAX_RANK],
    strides: &[isize; MAX_RANK],
    ndim: usize,
    index: I,
) -> LayoutResult<MAX_RANK> {
    if ndim == 0 {
        return Err(TensorError::IndexOutOfBounds { index: 0, size: 0 });
    }

    let leading = resolve_index_for_size_(index, shape[0])?;
    let mut new_shape = [0usize; MAX_RANK];
    let mut new_strides = [0isize; MAX_RANK];
    new_shape[..ndim - 1].copy_from_slice(&shape[1..ndim]);
    new_strides[..ndim - 1].copy_from_slice(&strides[1..ndim]);
    let new_ndim = ndim - 1;
    let offset = leading as isize * strides[0];
    let new_len = if new_ndim == 0 {
        1
    } else {
        new_shape[..new_ndim].iter().product()
    };
    Ok((new_shape, new_strides, new_ndim, offset, new_len))
}

fn reduced_shape(shape: &[usize], axis: usize, keep_dims: bool) -> Vec<usize> {
    let mut result = Vec::with_capacity(if keep_dims {
        shape.len()
    } else {
        shape.len() - 1
    });
    for (dim_index, &dim_size) in shape.iter().enumerate() {
        if dim_index == axis {
            if keep_dims {
                result.push(1);
            }
        } else {
            result.push(dim_size);
        }
    }
    result
}

fn shared_contiguous_tail_2(
    shape: &[usize],
    first_strides: &[isize],
    first_item_size: isize,
    second_strides: &[isize],
    second_item_size: isize,
) -> usize {
    let mut tail_dims = 0usize;
    let mut expected_first = first_item_size;
    let mut expected_second = second_item_size;
    for dim_index in (0..shape.len()).rev() {
        if first_strides[dim_index] == expected_first
            && second_strides[dim_index] == expected_second
        {
            tail_dims += 1;
            let dim_extent = shape[dim_index] as isize;
            expected_first = expected_first.saturating_mul(dim_extent);
            expected_second = expected_second.saturating_mul(dim_extent);
        } else {
            break;
        }
    }
    tail_dims
}

fn shared_contiguous_tail_3(
    shape: &[usize],
    first_strides: &[isize],
    first_item_size: isize,
    second_strides: &[isize],
    second_item_size: isize,
    third_strides: &[isize],
    third_item_size: isize,
) -> usize {
    let mut tail_dims = 0usize;
    let mut expected_first = first_item_size;
    let mut expected_second = second_item_size;
    let mut expected_third = third_item_size;
    for dim_index in (0..shape.len()).rev() {
        if first_strides[dim_index] == expected_first
            && second_strides[dim_index] == expected_second
            && third_strides[dim_index] == expected_third
        {
            tail_dims += 1;
            let dim_extent = shape[dim_index] as isize;
            expected_first = expected_first.saturating_mul(dim_extent);
            expected_second = expected_second.saturating_mul(dim_extent);
            expected_third = expected_third.saturating_mul(dim_extent);
        } else {
            break;
        }
    }
    tail_dims
}

fn shared_contiguous_tail_4(
    shape: &[usize],
    first_strides: &[isize],
    first_item_size: isize,
    second_strides: &[isize],
    second_item_size: isize,
    third_strides: &[isize],
    third_item_size: isize,
    fourth_strides: &[isize],
    fourth_item_size: isize,
) -> usize {
    let mut tail_dims = 0usize;
    let mut expected_first = first_item_size;
    let mut expected_second = second_item_size;
    let mut expected_third = third_item_size;
    let mut expected_fourth = fourth_item_size;
    for dim_index in (0..shape.len()).rev() {
        if first_strides[dim_index] == expected_first
            && second_strides[dim_index] == expected_second
            && third_strides[dim_index] == expected_third
            && fourth_strides[dim_index] == expected_fourth
        {
            tail_dims += 1;
            let dim_extent = shape[dim_index] as isize;
            expected_first = expected_first.saturating_mul(dim_extent);
            expected_second = expected_second.saturating_mul(dim_extent);
            expected_third = expected_third.saturating_mul(dim_extent);
            expected_fourth = expected_fourth.saturating_mul(dim_extent);
        } else {
            break;
        }
    }
    tail_dims
}

unsafe fn walk_contiguous_blocks_2<TIn, TOut, F>(
    source_ptr: *const TIn,
    source_strides: &[isize],
    target_ptr: *mut TOut,
    target_strides: &[isize],
    shape: &[usize],
    mut kernel: F,
) where
    F: FnMut(*const TIn, *mut TOut, usize),
{
    let tail_dims = shared_contiguous_tail_2(
        shape,
        source_strides,
        core::mem::size_of::<TIn>() as isize,
        target_strides,
        core::mem::size_of::<TOut>() as isize,
    );
    let tail_len = if tail_dims == 0 {
        1
    } else {
        shape[shape.len() - tail_dims..].iter().product()
    };
    let outer_dims = shape.len().saturating_sub(tail_dims);

    unsafe fn recurse<TIn, TOut, F>(
        dim_index: usize,
        outer_dims: usize,
        source_ptr: *const u8,
        source_strides: &[isize],
        target_ptr: *mut u8,
        target_strides: &[isize],
        shape: &[usize],
        tail_len: usize,
        kernel: &mut F,
    ) where
        F: FnMut(*const TIn, *mut TOut, usize),
    {
        if dim_index == outer_dims {
            kernel(source_ptr as *const TIn, target_ptr as *mut TOut, tail_len);
            return;
        }
        for offset_index in 0..shape[dim_index] {
            let source_child = source_ptr.offset(offset_index as isize * source_strides[dim_index]);
            let target_child = target_ptr.offset(offset_index as isize * target_strides[dim_index]);
            recurse::<TIn, TOut, F>(
                dim_index + 1,
                outer_dims,
                source_child,
                source_strides,
                target_child,
                target_strides,
                shape,
                tail_len,
                kernel,
            );
        }
    }

    recurse::<TIn, TOut, F>(
        0,
        outer_dims,
        source_ptr as *const u8,
        source_strides,
        target_ptr as *mut u8,
        target_strides,
        shape,
        tail_len,
        &mut kernel,
    );
}

unsafe fn walk_contiguous_blocks_3<TFirst, TSecond, TOut, F>(
    first_ptr: *const TFirst,
    first_strides: &[isize],
    second_ptr: *const TSecond,
    second_strides: &[isize],
    target_ptr: *mut TOut,
    target_strides: &[isize],
    shape: &[usize],
    mut kernel: F,
) where
    F: FnMut(*const TFirst, *const TSecond, *mut TOut, usize),
{
    let tail_dims = shared_contiguous_tail_3(
        shape,
        first_strides,
        core::mem::size_of::<TFirst>() as isize,
        second_strides,
        core::mem::size_of::<TSecond>() as isize,
        target_strides,
        core::mem::size_of::<TOut>() as isize,
    );
    let tail_len = if tail_dims == 0 {
        1
    } else {
        shape[shape.len() - tail_dims..].iter().product()
    };
    let outer_dims = shape.len().saturating_sub(tail_dims);

    unsafe fn recurse<TFirst, TSecond, TOut, F>(
        dim_index: usize,
        outer_dims: usize,
        first_ptr: *const u8,
        first_strides: &[isize],
        second_ptr: *const u8,
        second_strides: &[isize],
        target_ptr: *mut u8,
        target_strides: &[isize],
        shape: &[usize],
        tail_len: usize,
        kernel: &mut F,
    ) where
        F: FnMut(*const TFirst, *const TSecond, *mut TOut, usize),
    {
        if dim_index == outer_dims {
            kernel(
                first_ptr as *const TFirst,
                second_ptr as *const TSecond,
                target_ptr as *mut TOut,
                tail_len,
            );
            return;
        }
        for offset_index in 0..shape[dim_index] {
            let first_child = first_ptr.offset(offset_index as isize * first_strides[dim_index]);
            let second_child = second_ptr.offset(offset_index as isize * second_strides[dim_index]);
            let target_child = target_ptr.offset(offset_index as isize * target_strides[dim_index]);
            recurse::<TFirst, TSecond, TOut, F>(
                dim_index + 1,
                outer_dims,
                first_child,
                first_strides,
                second_child,
                second_strides,
                target_child,
                target_strides,
                shape,
                tail_len,
                kernel,
            );
        }
    }

    recurse::<TFirst, TSecond, TOut, F>(
        0,
        outer_dims,
        first_ptr as *const u8,
        first_strides,
        second_ptr as *const u8,
        second_strides,
        target_ptr as *mut u8,
        target_strides,
        shape,
        tail_len,
        &mut kernel,
    );
}

unsafe fn walk_contiguous_blocks_4<TFirst, TSecond, TThird, TOut, F>(
    first_ptr: *const TFirst,
    first_strides: &[isize],
    second_ptr: *const TSecond,
    second_strides: &[isize],
    third_ptr: *const TThird,
    third_strides: &[isize],
    target_ptr: *mut TOut,
    target_strides: &[isize],
    shape: &[usize],
    mut kernel: F,
) where
    F: FnMut(*const TFirst, *const TSecond, *const TThird, *mut TOut, usize),
{
    let tail_dims = shared_contiguous_tail_4(
        shape,
        first_strides,
        core::mem::size_of::<TFirst>() as isize,
        second_strides,
        core::mem::size_of::<TSecond>() as isize,
        third_strides,
        core::mem::size_of::<TThird>() as isize,
        target_strides,
        core::mem::size_of::<TOut>() as isize,
    );
    let tail_len = if tail_dims == 0 {
        1
    } else {
        shape[shape.len() - tail_dims..].iter().product()
    };
    let outer_dims = shape.len().saturating_sub(tail_dims);

    unsafe fn recurse<TFirst, TSecond, TThird, TOut, F>(
        dim_index: usize,
        outer_dims: usize,
        first_ptr: *const u8,
        first_strides: &[isize],
        second_ptr: *const u8,
        second_strides: &[isize],
        third_ptr: *const u8,
        third_strides: &[isize],
        target_ptr: *mut u8,
        target_strides: &[isize],
        shape: &[usize],
        tail_len: usize,
        kernel: &mut F,
    ) where
        F: FnMut(*const TFirst, *const TSecond, *const TThird, *mut TOut, usize),
    {
        if dim_index == outer_dims {
            kernel(
                first_ptr as *const TFirst,
                second_ptr as *const TSecond,
                third_ptr as *const TThird,
                target_ptr as *mut TOut,
                tail_len,
            );
            return;
        }
        for offset_index in 0..shape[dim_index] {
            let first_child = first_ptr.offset(offset_index as isize * first_strides[dim_index]);
            let second_child = second_ptr.offset(offset_index as isize * second_strides[dim_index]);
            let third_child = third_ptr.offset(offset_index as isize * third_strides[dim_index]);
            let target_child = target_ptr.offset(offset_index as isize * target_strides[dim_index]);
            recurse::<TFirst, TSecond, TThird, TOut, F>(
                dim_index + 1,
                outer_dims,
                first_child,
                first_strides,
                second_child,
                second_strides,
                third_child,
                third_strides,
                target_child,
                target_strides,
                shape,
                tail_len,
                kernel,
            );
        }
    }

    recurse::<TFirst, TSecond, TThird, TOut, F>(
        0,
        outer_dims,
        first_ptr as *const u8,
        first_strides,
        second_ptr as *const u8,
        second_strides,
        third_ptr as *const u8,
        third_strides,
        target_ptr as *mut u8,
        target_strides,
        shape,
        tail_len,
        &mut kernel,
    );
}

fn for_each_axis_lane<T, const MAX_RANK: usize, F>(
    view: &TensorView<'_, T, MAX_RANK>,
    axis: usize,
    mut callback: F,
) where
    F: FnMut(*const T, usize, isize, usize),
{
    let lane_len = view.shape[axis];
    let lane_stride = view.strides[axis];
    let mut other_dims = [0usize; MAX_RANK];
    let mut other_ndim = 0usize;
    for dim_index in 0..view.ndim {
        if dim_index != axis {
            other_dims[other_ndim] = dim_index;
            other_ndim += 1;
        }
    }

    if other_ndim == 0 {
        callback(view.data, lane_len, lane_stride, 0);
        return;
    }

    let mut coords = [0usize; MAX_RANK];
    let total_lanes: usize = other_dims[..other_ndim]
        .iter()
        .map(|&dim_index| view.shape[dim_index])
        .product();

    for lane_index in 0..total_lanes {
        let mut lane_offset = 0isize;
        for idx in 0..other_ndim {
            let dim_index = other_dims[idx];
            lane_offset += coords[idx] as isize * view.strides[dim_index];
        }
        let lane_ptr = unsafe { (view.data as *const u8).offset(lane_offset) as *const T };
        callback(lane_ptr, lane_len, lane_stride, lane_index);

        for idx in (0..other_ndim).rev() {
            coords[idx] += 1;
            if coords[idx] < view.shape[other_dims[idx]] {
                break;
            }
            coords[idx] = 0;
        }
    }
}

/// Detect trailing dimensions where `stride[i] == stride[i+1] * shape[i+1]`.
/// Returns `(tail_dims, element_count, abs_stride_bytes)`.
fn uniform_stride_tail(shape: &[usize], strides: &[isize]) -> (usize, usize, usize) {
    let rank = shape.len();
    if rank == 0 {
        return (0, 1, 0);
    }
    let mut tail: usize = 1;
    let innermost = strides[rank - 1];
    let mut abs_expected = if innermost < 0 {
        -innermost
    } else {
        innermost
    };
    for i in (0..rank - 1).rev() {
        let next = abs_expected * shape[i + 1] as isize;
        let actual = strides[i];
        let abs_actual = if actual < 0 { -actual } else { actual };
        if abs_actual != next {
            break;
        }
        abs_expected = next;
        tail += 1;
    }
    let count: usize = shape[rank - tail..].iter().product();
    let abs_stride = if innermost < 0 {
        (-innermost) as usize
    } else {
        innermost as usize
    };
    (tail, count, abs_stride)
}

unsafe fn normalize_reduction_lane<T>(
    lane_ptr: *const T,
    lane_len: usize,
    lane_stride: isize,
) -> (*const T, usize, usize, bool) {
    if lane_len == 0 {
        return (lane_ptr, 0, core::mem::size_of::<T>(), false);
    }
    if lane_stride >= 0 {
        return (lane_ptr, lane_len, lane_stride as usize, false);
    }
    let last_ptr =
        (lane_ptr as *const u8).offset((lane_len as isize - 1) * lane_stride) as *const T;
    (last_ptr, lane_len, (-lane_stride) as usize, true)
}

unsafe fn reduce_moments_recursive<T>(
    data: *const T,
    shape: &[usize],
    strides: &[isize],
) -> (T::SumOutput, T::SumSqOutput)
where
    T: ReduceMoments,
    T::SumOutput: Default + core::ops::AddAssign,
    T::SumSqOutput: Default + core::ops::AddAssign,
{
    if shape.is_empty() {
        return T::reduce_moments_raw(data, 1, core::mem::size_of::<T>());
    }
    if shape[0] == 0 {
        return (T::SumOutput::default(), T::SumSqOutput::default());
    }
    if shape.len() == 1 {
        let (lane_ptr, lane_len, lane_stride, _) =
            normalize_reduction_lane(data, shape[0], strides[0]);
        return T::reduce_moments_raw(lane_ptr, lane_len, lane_stride);
    }

    let mut sum = T::SumOutput::default();
    let mut sumsq = T::SumSqOutput::default();
    for index in 0..shape[0] {
        let child_ptr = (data as *const u8).offset(index as isize * strides[0]) as *const T;
        let (child_sum, child_sumsq) =
            reduce_moments_recursive::<T>(child_ptr, &shape[1..], &strides[1..]);
        sum += child_sum;
        sumsq += child_sumsq;
    }
    (sum, sumsq)
}

unsafe fn reduce_minmax_recursive<T>(
    data: *const T,
    shape: &[usize],
    strides: &[isize],
    logical_offset: usize,
) -> Option<MinMaxResult<T::Output>>
where
    T: ReduceMinMax,
    T::Output: Clone + PartialOrd,
{
    if shape.is_empty() {
        return T::reduce_minmax_raw(data, 1, core::mem::size_of::<T>()).map(
            |(min_value, _, max_value, _)| MinMaxResult {
                min_value,
                min_index: logical_offset,
                max_value,
                max_index: logical_offset,
            },
        );
    }
    if shape[0] == 0 {
        return None;
    }
    if shape.len() == 1 {
        let (lane_ptr, lane_len, lane_stride, reversed) =
            normalize_reduction_lane(data, shape[0], strides[0]);
        return T::reduce_minmax_raw(lane_ptr, lane_len, lane_stride).map(
            |(min_value, min_index, max_value, max_index)| {
                let min_index = if reversed {
                    lane_len - 1 - min_index
                } else {
                    min_index
                };
                let max_index = if reversed {
                    lane_len - 1 - max_index
                } else {
                    max_index
                };
                MinMaxResult {
                    min_value,
                    min_index: logical_offset + min_index,
                    max_value,
                    max_index: logical_offset + max_index,
                }
            },
        );
    }

    let inner_len: usize = shape[1..].iter().product();
    let mut best_min: Option<(T::Output, usize)> = None;
    let mut best_max: Option<(T::Output, usize)> = None;

    for index in 0..shape[0] {
        let child_ptr = (data as *const u8).offset(index as isize * strides[0]) as *const T;
        let child_offset = logical_offset + index * inner_len;
        if let Some(child) =
            reduce_minmax_recursive::<T>(child_ptr, &shape[1..], &strides[1..], child_offset)
        {
            match &best_min {
                Some((best_value, _))
                    if child.min_value.partial_cmp(best_value)
                        != Some(core::cmp::Ordering::Less) => {}
                _ => best_min = Some((child.min_value, child.min_index)),
            }
            match &best_max {
                Some((best_value, _))
                    if child.max_value.partial_cmp(best_value)
                        != Some(core::cmp::Ordering::Greater) => {}
                _ => best_max = Some((child.max_value, child.max_index)),
            }
        }
    }

    match (best_min, best_max) {
        (Some((min_value, min_index)), Some((max_value, max_index))) => Some(MinMaxResult {
            min_value,
            min_index,
            max_value,
            max_index,
        }),
        _ => None,
    }
}

#[doc(hidden)]
pub trait SumSqToF64 {
    fn to_f64(self) -> f64;
}

impl SumSqToF64 for f32 {
    fn to_f64(self) -> f64 { self as f64 }
}
impl SumSqToF64 for f64 {
    fn to_f64(self) -> f64 { self }
}
impl SumSqToF64 for u64 {
    fn to_f64(self) -> f64 { self as f64 }
}
impl SumSqToF64 for i64 {
    fn to_f64(self) -> f64 { self as f64 }
}

fn try_alloc_output_like<D: Clone + StorageElement, F, const MAX_RANK: usize>(
    shape: &[usize],
    fill: F,
) -> Result<Tensor<D, Global, MAX_RANK>, TensorError>
where
    F: FnOnce(&mut TensorSpan<'_, D, MAX_RANK>) -> Result<(), TensorError>,
{
    let mut result = unsafe { Tensor::<D, Global, MAX_RANK>::try_empty(shape) }?;
    {
        let mut span = result.span();
        fill(&mut span)?;
    }
    Ok(result)
}

fn try_reborrow_tensor_into<T: Clone, D: Clone, F, const MAX_RANK: usize>(
    source: &Tensor<T, Global, MAX_RANK>,
    out: &mut Tensor<D, Global, MAX_RANK>,
    apply: F,
) -> Result<(), TensorError>
where
    F: FnOnce(
        &TensorView<'_, T, MAX_RANK>,
        &mut TensorSpan<'_, D, MAX_RANK>,
    ) -> Result<(), TensorError>,
{
    let view = source.view();
    let mut span = out.span();
    apply(&view, &mut span)
}

fn try_reborrow_tensor_inplace<T: Clone, F, const MAX_RANK: usize>(
    tensor: &mut Tensor<T, Global, MAX_RANK>,
    apply: F,
) -> Result<(), TensorError>
where
    F: FnOnce(
        &TensorView<'_, T, MAX_RANK>,
        &mut TensorSpan<'_, T, MAX_RANK>,
    ) -> Result<(), TensorError>,
{
    let view = TensorView {
        data: tensor.data.as_ptr(),
        shape: tensor.shape,
        strides: tensor.strides,
        ndim: tensor.ndim,
        _marker: PhantomData,
    };
    let mut span = tensor.span();
    apply(&view, &mut span)
}

fn rebind_view_rank<'a, T, const TARGET_MAX_RANK: usize, const SOURCE_MAX_RANK: usize>(
    view: &TensorView<'a, T, SOURCE_MAX_RANK>,
) -> Result<TensorView<'a, T, TARGET_MAX_RANK>, TensorError> {
    if view.ndim > TARGET_MAX_RANK {
        return Err(TensorError::DimensionMismatch {
            expected: TARGET_MAX_RANK,
            got: view.ndim,
        });
    }
    let mut shape = [0usize; TARGET_MAX_RANK];
    let mut strides = [0isize; TARGET_MAX_RANK];
    shape[..view.ndim].copy_from_slice(&view.shape[..view.ndim]);
    strides[..view.ndim].copy_from_slice(&view.strides[..view.ndim]);
    Ok(TensorView {
        data: view.data,
        shape,
        strides,
        ndim: view.ndim,
        _marker: PhantomData,
    })
}

fn try_unary_kernel_into<S, D, F, const MAX_RANK: usize>(
    source: &TensorView<'_, S, MAX_RANK>,
    out: &mut TensorSpan<'_, D, MAX_RANK>,
    mut kernel: F,
) -> Result<(), TensorError>
where
    F: FnMut(&[S], &mut [D]),
{
    validate_same_shape(source.shape(), out.shape())?;
    unsafe {
        walk_contiguous_blocks_2(
            source.data,
            &source.strides[..source.ndim],
            out.data,
            &out.strides[..out.ndim],
            source.shape(),
            |source_ptr, target_ptr, tail_len| {
                let source = core::slice::from_raw_parts(source_ptr, tail_len);
                let target = core::slice::from_raw_parts_mut(target_ptr, tail_len);
                kernel(source, target);
            },
        );
    }
    Ok(())
}

fn try_binary_kernel_into<A, B, D, F, const MAX_RANK: usize>(
    first: &TensorView<'_, A, MAX_RANK>,
    second: &TensorView<'_, B, MAX_RANK>,
    out: &mut TensorSpan<'_, D, MAX_RANK>,
    mut kernel: F,
) -> Result<(), TensorError>
where
    F: FnMut(&[A], &[B], &mut [D]),
{
    validate_same_shape(first.shape(), second.shape())?;
    validate_same_shape(first.shape(), out.shape())?;
    unsafe {
        walk_contiguous_blocks_3(
            first.data,
            &first.strides[..first.ndim],
            second.data,
            &second.strides[..second.ndim],
            out.data,
            &out.strides[..out.ndim],
            first.shape(),
            |first_ptr, second_ptr, target_ptr, tail_len| {
                let first = core::slice::from_raw_parts(first_ptr, tail_len);
                let second = core::slice::from_raw_parts(second_ptr, tail_len);
                let target = core::slice::from_raw_parts_mut(target_ptr, tail_len);
                kernel(first, second, target);
            },
        );
    }
    Ok(())
}

fn try_ternary_kernel_into<A, B, C, D, F, const MAX_RANK: usize>(
    first: &TensorView<'_, A, MAX_RANK>,
    second: &TensorView<'_, B, MAX_RANK>,
    third: &TensorView<'_, C, MAX_RANK>,
    out: &mut TensorSpan<'_, D, MAX_RANK>,
    mut kernel: F,
) -> Result<(), TensorError>
where
    F: FnMut(&[A], &[B], &[C], &mut [D]),
{
    validate_same_shape(first.shape(), second.shape())?;
    validate_same_shape(first.shape(), third.shape())?;
    validate_same_shape(first.shape(), out.shape())?;
    unsafe {
        walk_contiguous_blocks_4(
            first.data,
            &first.strides[..first.ndim],
            second.data,
            &second.strides[..second.ndim],
            third.data,
            &third.strides[..third.ndim],
            out.data,
            &out.strides[..out.ndim],
            first.shape(),
            |first_ptr, second_ptr, third_ptr, target_ptr, tail_len| {
                let first = core::slice::from_raw_parts(first_ptr, tail_len);
                let second = core::slice::from_raw_parts(second_ptr, tail_len);
                let third = core::slice::from_raw_parts(third_ptr, tail_len);
                let target = core::slice::from_raw_parts_mut(target_ptr, tail_len);
                kernel(first, second, third, target);
            },
        );
    }
    Ok(())
}

// endregion: Tensor Internal Helpers

// region: Tensor Elementwise Operations

impl<T: Clone + EachScale, const MAX_RANK: usize> Tensor<T, Global, MAX_RANK>
where
    T::Scalar: From<f32> + core::ops::Mul<Output = T::Scalar> + Copy,
{
    /// Apply element-wise scale: result\[i\] = α × self\[i\] + β
    ///
    /// Returns a new array with the scaled values.
    pub fn scale(
        &self,
        alpha: T::Scalar,
        beta: T::Scalar,
    ) -> Result<Tensor<T, Global, MAX_RANK>, TensorError> {
        self.view().try_scale_tensor(alpha, beta)
    }

    /// Apply element-wise scale in-place: self\[i\] = α × self\[i\] + β
    pub fn scale_inplace(&mut self, alpha: T::Scalar, beta: T::Scalar) {
        let _ = try_reborrow_tensor_inplace(self, |view, span| {
            view.try_scale_tensor_into(alpha, beta, span)
        });
    }
}

impl<T: Clone + EachSum, const MAX_RANK: usize> Tensor<T, Global, MAX_RANK> {
    /// Element-wise sum: result\[i\] = self\[i\] + other\[i\]
    ///
    /// Returns a new array with the summed values.
    pub fn add<const OTHER_MAX_RANK: usize>(
        &self,
        other: &Tensor<T, Global, OTHER_MAX_RANK>,
    ) -> Result<Tensor<T, Global, MAX_RANK>, TensorError> {
        validate_same_shape(self.shape(), other.shape())?;
        let other_view = rebind_view_rank::<T, MAX_RANK, OTHER_MAX_RANK>(&other.view())?;
        self.view().try_add_tensor(&other_view)
    }

    /// Element-wise sum in-place: self\[i\] = self\[i\] + other\[i\]
    pub fn add_inplace<const OTHER_MAX_RANK: usize>(
        &mut self,
        other: &Tensor<T, Global, OTHER_MAX_RANK>,
    ) -> Result<(), TensorError> {
        validate_same_shape(self.shape(), other.shape())?;
        let other_view = rebind_view_rank::<T, MAX_RANK, OTHER_MAX_RANK>(&other.view())?;
        try_reborrow_tensor_inplace(self, |view, span| {
            view.try_add_tensor_into(&other_view, span)
        })
    }
}

impl<T: Clone + EachBlend, const MAX_RANK: usize> Tensor<T, Global, MAX_RANK>
where
    T::Scalar: From<f32> + Copy,
{
    /// Blend: result\[i\] = α × self\[i\] + β × other\[i\]
    ///
    /// Returns a new array with the blend.
    pub fn blend<const OTHER_MAX_RANK: usize>(
        &self,
        other: &Tensor<T, Global, OTHER_MAX_RANK>,
        alpha: T::Scalar,
        beta: T::Scalar,
    ) -> Result<Tensor<T, Global, MAX_RANK>, TensorError> {
        validate_same_shape(self.shape(), other.shape())?;
        let other_view = rebind_view_rank::<T, MAX_RANK, OTHER_MAX_RANK>(&other.view())?;
        self.view().try_blend_tensor(&other_view, alpha, beta)
    }
}

impl<T: Clone + EachFMA, const MAX_RANK: usize> Tensor<T, Global, MAX_RANK>
where
    T::Scalar: From<f32> + Copy,
{
    /// Fused multiply-add: result\[i\] = α × self\[i\] × b\[i\] + β × c\[i\]
    ///
    /// Returns a new array with the FMA result.
    pub fn fma<const B_MAX_RANK: usize, const C_MAX_RANK: usize>(
        &self,
        b: &Tensor<T, Global, B_MAX_RANK>,
        c: &Tensor<T, Global, C_MAX_RANK>,
        alpha: T::Scalar,
        beta: T::Scalar,
    ) -> Result<Tensor<T, Global, MAX_RANK>, TensorError> {
        validate_same_shape(self.shape(), b.shape())?;
        validate_same_shape(self.shape(), c.shape())?;
        let b_view = rebind_view_rank::<T, MAX_RANK, B_MAX_RANK>(&b.view())?;
        let c_view = rebind_view_rank::<T, MAX_RANK, C_MAX_RANK>(&c.view())?;
        self.view().try_fma_tensors(&b_view, &c_view, alpha, beta)
    }
}

// endregion: Tensor Elementwise Operations

// region: Tensor Explicit Elementwise + Cast

impl<'a, T: Clone + EachScale, const MAX_RANK: usize> TensorView<'a, T, MAX_RANK>
where
    T::Scalar: From<f32> + core::ops::Mul<Output = T::Scalar> + Copy,
{
    pub fn try_scale_tensor(
        &self,
        alpha: T::Scalar,
        beta: T::Scalar,
    ) -> Result<Tensor<T, Global, MAX_RANK>, TensorError> {
        try_alloc_output_like(self.shape(), |span| {
            self.try_scale_tensor_into(alpha, beta, span)
        })
    }

    pub fn try_scale_tensor_into(
        &self,
        alpha: T::Scalar,
        beta: T::Scalar,
        out: &mut TensorSpan<'_, T, MAX_RANK>,
    ) -> Result<(), TensorError> {
        self.try_affine_into(alpha, beta, out)
    }

    fn try_affine_into(
        &self,
        alpha: T::Scalar,
        beta: T::Scalar,
        out: &mut TensorSpan<'_, T, MAX_RANK>,
    ) -> Result<(), TensorError> {
        try_unary_kernel_into(self, out, |source, target| {
            T::each_scale(source, alpha, beta, target);
        })
    }

    pub fn try_add_scalar(
        &self,
        scalar: T::Scalar,
    ) -> Result<Tensor<T, Global, MAX_RANK>, TensorError> {
        try_alloc_output_like(self.shape(), |span| self.try_add_scalar_into(scalar, span))
    }

    pub fn try_sub_scalar(
        &self,
        scalar: T::Scalar,
    ) -> Result<Tensor<T, Global, MAX_RANK>, TensorError> {
        try_alloc_output_like(self.shape(), |span| self.try_sub_scalar_into(scalar, span))
    }

    pub fn try_mul_scalar(
        &self,
        scalar: T::Scalar,
    ) -> Result<Tensor<T, Global, MAX_RANK>, TensorError> {
        try_alloc_output_like(self.shape(), |span| self.try_mul_scalar_into(scalar, span))
    }

    pub fn try_add_scalar_into(
        &self,
        scalar: T::Scalar,
        out: &mut TensorSpan<'_, T, MAX_RANK>,
    ) -> Result<(), TensorError> {
        self.try_affine_into(T::Scalar::from(1.0f32), scalar, out)
    }

    pub fn try_sub_scalar_into(
        &self,
        scalar: T::Scalar,
        out: &mut TensorSpan<'_, T, MAX_RANK>,
    ) -> Result<(), TensorError> {
        self.try_affine_into(
            T::Scalar::from(1.0f32),
            T::Scalar::from(-1.0f32) * scalar,
            out,
        )
    }

    pub fn try_mul_scalar_into(
        &self,
        scalar: T::Scalar,
        out: &mut TensorSpan<'_, T, MAX_RANK>,
    ) -> Result<(), TensorError> {
        self.try_affine_into(scalar, T::Scalar::from(0.0f32), out)
    }
}

impl<'a, T: Clone + EachSum, const MAX_RANK: usize> TensorView<'a, T, MAX_RANK> {
    pub fn try_add_tensor(
        &self,
        other: &TensorView<'_, T, MAX_RANK>,
    ) -> Result<Tensor<T, Global, MAX_RANK>, TensorError> {
        try_alloc_output_like(self.shape(), |span| self.try_add_tensor_into(other, span))
    }

    pub fn try_add_tensor_into(
        &self,
        other: &TensorView<'_, T, MAX_RANK>,
        out: &mut TensorSpan<'_, T, MAX_RANK>,
    ) -> Result<(), TensorError> {
        try_binary_kernel_into(self, other, out, |first, second, target| {
            T::each_sum(first, second, target);
        })
    }
}

impl<'a, T: Clone + EachBlend, const MAX_RANK: usize> TensorView<'a, T, MAX_RANK>
where
    T::Scalar: From<f32> + Copy,
{
    pub fn try_blend_tensor(
        &self,
        other: &TensorView<'_, T, MAX_RANK>,
        alpha: T::Scalar,
        beta: T::Scalar,
    ) -> Result<Tensor<T, Global, MAX_RANK>, TensorError> {
        try_alloc_output_like(self.shape(), |span| {
            self.try_blend_tensor_into(other, alpha, beta, span)
        })
    }

    pub fn try_blend_tensor_into(
        &self,
        other: &TensorView<'_, T, MAX_RANK>,
        alpha: T::Scalar,
        beta: T::Scalar,
        out: &mut TensorSpan<'_, T, MAX_RANK>,
    ) -> Result<(), TensorError> {
        try_binary_kernel_into(self, other, out, |first, second, target| {
            T::each_blend(first, second, alpha, beta, target);
        })
    }

    pub fn try_sub_tensor(
        &self,
        other: &TensorView<'_, T, MAX_RANK>,
    ) -> Result<Tensor<T, Global, MAX_RANK>, TensorError> {
        try_alloc_output_like(self.shape(), |span| self.try_sub_tensor_into(other, span))
    }

    pub fn try_sub_tensor_into(
        &self,
        other: &TensorView<'_, T, MAX_RANK>,
        out: &mut TensorSpan<'_, T, MAX_RANK>,
    ) -> Result<(), TensorError> {
        self.try_blend_tensor_into(
            other,
            T::Scalar::from(1.0f32),
            T::Scalar::from(-1.0f32),
            out,
        )
    }
}

impl<'a, T: Clone + EachFMA, const MAX_RANK: usize> TensorView<'a, T, MAX_RANK>
where
    T::Scalar: From<f32> + Copy,
{
    pub fn try_fma_tensors(
        &self,
        b: &TensorView<'_, T, MAX_RANK>,
        c: &TensorView<'_, T, MAX_RANK>,
        alpha: T::Scalar,
        beta: T::Scalar,
    ) -> Result<Tensor<T, Global, MAX_RANK>, TensorError> {
        try_alloc_output_like(self.shape(), |span| {
            self.try_fma_tensors_into(b, c, alpha, beta, span)
        })
    }

    pub fn try_fma_tensors_into(
        &self,
        b: &TensorView<'_, T, MAX_RANK>,
        c: &TensorView<'_, T, MAX_RANK>,
        alpha: T::Scalar,
        beta: T::Scalar,
        out: &mut TensorSpan<'_, T, MAX_RANK>,
    ) -> Result<(), TensorError> {
        try_ternary_kernel_into(self, b, c, out, |first, second, third, target| {
            T::each_fma(first, second, third, alpha, beta, target);
        })
    }

    pub fn try_mul_tensor(
        &self,
        other: &TensorView<'_, T, MAX_RANK>,
    ) -> Result<Tensor<T, Global, MAX_RANK>, TensorError> {
        try_alloc_output_like(self.shape(), |span| self.try_mul_tensor_into(other, span))
    }

    pub fn try_mul_tensor_into(
        &self,
        other: &TensorView<'_, T, MAX_RANK>,
        out: &mut TensorSpan<'_, T, MAX_RANK>,
    ) -> Result<(), TensorError> {
        self.try_fma_tensors_into(
            other,
            self,
            T::Scalar::from(1.0f32),
            T::Scalar::from(0.0f32),
            out,
        )
    }
}

impl<'a, S: Clone + CastDtype, const MAX_RANK: usize> TensorView<'a, S, MAX_RANK> {
    pub fn try_cast_dtype<D: Clone + CastDtype>(
        &self,
    ) -> Result<Tensor<D, Global, MAX_RANK>, TensorError> {
        try_alloc_output_like(self.shape(), |span| self.try_cast_dtype_into(span))
    }

    pub fn try_cast_dtype_into<D: Clone + CastDtype>(
        &self,
        out: &mut TensorSpan<'_, D, MAX_RANK>,
    ) -> Result<(), TensorError> {
        try_unary_kernel_into(self, out, |source, target| {
            let _ = cast(source, target);
        })
    }
}

/// Extension trait: scalar arithmetic for any [`TensorRef`] implementor.
pub trait ScaleOps<T: Clone + EachScale, const MAX_RANK: usize>: TensorRef<T, MAX_RANK>
where
    T::Scalar: From<f32> + core::ops::Mul<Output = T::Scalar> + Copy,
{
    fn try_add_scalar(
        &self,
        scalar: T::Scalar,
    ) -> Result<Tensor<T, Global, MAX_RANK>, TensorError> {
        self.view().try_add_scalar(scalar)
    }

    fn try_sub_scalar(
        &self,
        scalar: T::Scalar,
    ) -> Result<Tensor<T, Global, MAX_RANK>, TensorError> {
        self.view().try_sub_scalar(scalar)
    }

    fn try_mul_scalar(
        &self,
        scalar: T::Scalar,
    ) -> Result<Tensor<T, Global, MAX_RANK>, TensorError> {
        self.view().try_mul_scalar(scalar)
    }
}

impl<T: Clone + EachScale, const R: usize, C: TensorRef<T, R>> ScaleOps<T, R> for C where
    T::Scalar: From<f32> + core::ops::Mul<Output = T::Scalar> + Copy
{
}

impl<T: Clone + EachScale, const MAX_RANK: usize> Tensor<T, Global, MAX_RANK>
where
    T::Scalar: From<f32> + core::ops::Mul<Output = T::Scalar> + Copy,
{
    pub fn try_add_scalar_into(
        &self,
        scalar: T::Scalar,
        out: &mut Tensor<T, Global, MAX_RANK>,
    ) -> Result<(), TensorError> {
        try_reborrow_tensor_into(self, out, |view, span| {
            view.try_add_scalar_into(scalar, span)
        })
    }

    pub fn try_sub_scalar_into(
        &self,
        scalar: T::Scalar,
        out: &mut Tensor<T, Global, MAX_RANK>,
    ) -> Result<(), TensorError> {
        try_reborrow_tensor_into(self, out, |view, span| {
            view.try_sub_scalar_into(scalar, span)
        })
    }

    pub fn try_mul_scalar_into(
        &self,
        scalar: T::Scalar,
        out: &mut Tensor<T, Global, MAX_RANK>,
    ) -> Result<(), TensorError> {
        try_reborrow_tensor_into(self, out, |view, span| {
            view.try_mul_scalar_into(scalar, span)
        })
    }

    pub fn try_add_scalar_inplace(&mut self, scalar: T::Scalar) -> Result<(), TensorError> {
        self.add_scalar_inplace(scalar);
        Ok(())
    }

    pub fn try_sub_scalar_inplace(&mut self, scalar: T::Scalar) -> Result<(), TensorError> {
        self.sub_scalar_inplace(scalar);
        Ok(())
    }

    pub fn try_mul_scalar_inplace(&mut self, scalar: T::Scalar) -> Result<(), TensorError> {
        self.mul_scalar_inplace(scalar);
        Ok(())
    }

    /// Element-wise add scalar in-place (infallible — self vs self always matches).
    pub fn add_scalar_inplace(&mut self, scalar: T::Scalar) {
        try_reborrow_tensor_inplace(self, |view, span| view.try_add_scalar_into(scalar, span))
            .expect("inplace scalar op on self cannot fail")
    }

    /// Element-wise subtract scalar in-place (infallible — self vs self always matches).
    pub fn sub_scalar_inplace(&mut self, scalar: T::Scalar) {
        try_reborrow_tensor_inplace(self, |view, span| view.try_sub_scalar_into(scalar, span))
            .expect("inplace scalar op on self cannot fail")
    }

    /// Element-wise multiply scalar in-place (infallible — self vs self always matches).
    pub fn mul_scalar_inplace(&mut self, scalar: T::Scalar) {
        try_reborrow_tensor_inplace(self, |view, span| view.try_mul_scalar_into(scalar, span))
            .expect("inplace scalar op on self cannot fail")
    }
}

impl<T: Clone + EachScale, const MAX_RANK: usize> core::ops::AddAssign<T::Scalar>
    for Tensor<T, Global, MAX_RANK>
where
    T::Scalar: From<f32> + core::ops::Mul<Output = T::Scalar> + Copy,
{
    fn add_assign(&mut self, scalar: T::Scalar) { self.add_scalar_inplace(scalar); }
}

impl<T: Clone + EachScale, const MAX_RANK: usize> core::ops::SubAssign<T::Scalar>
    for Tensor<T, Global, MAX_RANK>
where
    T::Scalar: From<f32> + core::ops::Mul<Output = T::Scalar> + Copy,
{
    fn sub_assign(&mut self, scalar: T::Scalar) { self.sub_scalar_inplace(scalar); }
}

impl<T: Clone + EachScale, const MAX_RANK: usize> core::ops::MulAssign<T::Scalar>
    for Tensor<T, Global, MAX_RANK>
where
    T::Scalar: From<f32> + core::ops::Mul<Output = T::Scalar> + Copy,
{
    fn mul_assign(&mut self, scalar: T::Scalar) { self.mul_scalar_inplace(scalar); }
}

/// Extension trait: element-wise addition for any [`TensorRef`] implementor.
pub trait SumOps<T: Clone + EachSum, const MAX_RANK: usize>: TensorRef<T, MAX_RANK> {
    fn try_add_tensor(
        &self,
        other: &(impl TensorRef<T, MAX_RANK> + ?Sized),
    ) -> Result<Tensor<T, Global, MAX_RANK>, TensorError> {
        self.view().try_add_tensor(&other.view())
    }
}

impl<T: Clone + EachSum, const R: usize, C: TensorRef<T, R>> SumOps<T, R> for C {}

impl<T: Clone + EachSum, const MAX_RANK: usize> Tensor<T, Global, MAX_RANK> {
    pub fn try_add_tensor_into(
        &self,
        other: &Tensor<T, Global, MAX_RANK>,
        out: &mut Tensor<T, Global, MAX_RANK>,
    ) -> Result<(), TensorError> {
        try_reborrow_tensor_into(self, out, |view, span| {
            view.try_add_tensor_into(&other.view(), span)
        })
    }

    pub fn try_add_tensor_inplace(
        &mut self,
        other: &Tensor<T, Global, MAX_RANK>,
    ) -> Result<(), TensorError> {
        try_reborrow_tensor_inplace(self, |view, span| {
            view.try_add_tensor_into(&other.view(), span)
        })
    }
}

/// Extension trait: element-wise subtraction for any [`TensorRef`] implementor.
pub trait BlendOps<T: Clone + EachBlend, const MAX_RANK: usize>: TensorRef<T, MAX_RANK>
where
    T::Scalar: From<f32> + Copy,
{
    fn try_sub_tensor(
        &self,
        other: &(impl TensorRef<T, MAX_RANK> + ?Sized),
    ) -> Result<Tensor<T, Global, MAX_RANK>, TensorError> {
        self.view().try_sub_tensor(&other.view())
    }
}

impl<T: Clone + EachBlend, const R: usize, C: TensorRef<T, R>> BlendOps<T, R> for C where
    T::Scalar: From<f32> + Copy
{
}

impl<T: Clone + EachBlend, const MAX_RANK: usize> Tensor<T, Global, MAX_RANK>
where
    T::Scalar: From<f32> + Copy,
{
    pub fn try_sub_tensor_into(
        &self,
        other: &Tensor<T, Global, MAX_RANK>,
        out: &mut Tensor<T, Global, MAX_RANK>,
    ) -> Result<(), TensorError> {
        try_reborrow_tensor_into(self, out, |view, span| {
            view.try_sub_tensor_into(&other.view(), span)
        })
    }

    pub fn try_sub_tensor_inplace(
        &mut self,
        other: &Tensor<T, Global, MAX_RANK>,
    ) -> Result<(), TensorError> {
        try_reborrow_tensor_inplace(self, |view, span| {
            view.try_sub_tensor_into(&other.view(), span)
        })
    }
}

/// Extension trait: element-wise multiplication for any [`TensorRef`] implementor.
pub trait FmaOps<T: Clone + EachFMA, const MAX_RANK: usize>: TensorRef<T, MAX_RANK>
where
    T::Scalar: From<f32> + Copy,
{
    fn try_mul_tensor(
        &self,
        other: &(impl TensorRef<T, MAX_RANK> + ?Sized),
    ) -> Result<Tensor<T, Global, MAX_RANK>, TensorError> {
        self.view().try_mul_tensor(&other.view())
    }
}

impl<T: Clone + EachFMA, const R: usize, C: TensorRef<T, R>> FmaOps<T, R> for C where
    T::Scalar: From<f32> + Copy
{
}

impl<T: Clone + EachFMA, const MAX_RANK: usize> Tensor<T, Global, MAX_RANK>
where
    T::Scalar: From<f32> + Copy,
{
    pub fn try_mul_tensor_into(
        &self,
        other: &Tensor<T, Global, MAX_RANK>,
        out: &mut Tensor<T, Global, MAX_RANK>,
    ) -> Result<(), TensorError> {
        try_reborrow_tensor_into(self, out, |view, span| {
            view.try_mul_tensor_into(&other.view(), span)
        })
    }

    pub fn try_mul_tensor_inplace(
        &mut self,
        other: &Tensor<T, Global, MAX_RANK>,
    ) -> Result<(), TensorError> {
        try_reborrow_tensor_inplace(self, |view, span| {
            view.try_mul_tensor_into(&other.view(), span)
        })
    }
}

/// Extension trait: type casting for any [`TensorRef`] implementor.
pub trait CastOps<S: Clone + CastDtype, const MAX_RANK: usize>: TensorRef<S, MAX_RANK> {
    fn try_cast_dtype<D: Clone + CastDtype>(
        &self,
    ) -> Result<Tensor<D, Global, MAX_RANK>, TensorError> {
        self.view().try_cast_dtype()
    }
}

impl<S: Clone + CastDtype, const R: usize, C: TensorRef<S, R>> CastOps<S, R> for C {}

impl<S: Clone + CastDtype, const MAX_RANK: usize> Tensor<S, Global, MAX_RANK> {
    pub fn try_cast_dtype_into<D: Clone + CastDtype>(
        &self,
        out: &mut Tensor<D, Global, MAX_RANK>,
    ) -> Result<(), TensorError> {
        try_reborrow_tensor_into(self, out, |view, span| view.try_cast_dtype_into(span))
    }
}

// endregion: Tensor Explicit Elementwise + Cast

// region: Tensor Trigonometry

impl<'a, T: Clone + EachSin, const MAX_RANK: usize> TensorView<'a, T, MAX_RANK> {
    pub fn try_sin(&self) -> Result<Tensor<T, Global, MAX_RANK>, TensorError> {
        try_alloc_output_like(self.shape(), |span| self.try_sin_into(span))
    }

    pub fn try_sin_into(&self, out: &mut TensorSpan<'_, T, MAX_RANK>) -> Result<(), TensorError> {
        try_unary_kernel_into(self, out, |source, target| {
            T::sin(source, target);
        })
    }
}

impl<'a, T: Clone + EachCos, const MAX_RANK: usize> TensorView<'a, T, MAX_RANK> {
    pub fn try_cos(&self) -> Result<Tensor<T, Global, MAX_RANK>, TensorError> {
        try_alloc_output_like(self.shape(), |span| self.try_cos_into(span))
    }

    pub fn try_cos_into(&self, out: &mut TensorSpan<'_, T, MAX_RANK>) -> Result<(), TensorError> {
        try_unary_kernel_into(self, out, |source, target| {
            T::cos(source, target);
        })
    }
}

impl<'a, T: Clone + EachATan, const MAX_RANK: usize> TensorView<'a, T, MAX_RANK> {
    pub fn try_atan(&self) -> Result<Tensor<T, Global, MAX_RANK>, TensorError> {
        try_alloc_output_like(self.shape(), |span| self.try_atan_into(span))
    }

    pub fn try_atan_into(&self, out: &mut TensorSpan<'_, T, MAX_RANK>) -> Result<(), TensorError> {
        try_unary_kernel_into(self, out, |source, target| {
            T::atan(source, target);
        })
    }
}

/// Extension trait: element-wise sine for any [`TensorRef`] implementor.
pub trait TrigSinOps<T: Clone + EachSin, const MAX_RANK: usize>: TensorRef<T, MAX_RANK> {
    fn try_sin(&self) -> Result<Tensor<T, Global, MAX_RANK>, TensorError> { self.view().try_sin() }
}

impl<T: Clone + EachSin, const R: usize, C: TensorRef<T, R>> TrigSinOps<T, R> for C {}

/// Extension trait: element-wise cosine for any [`TensorRef`] implementor.
pub trait TrigCosOps<T: Clone + EachCos, const MAX_RANK: usize>: TensorRef<T, MAX_RANK> {
    fn try_cos(&self) -> Result<Tensor<T, Global, MAX_RANK>, TensorError> { self.view().try_cos() }
}

impl<T: Clone + EachCos, const R: usize, C: TensorRef<T, R>> TrigCosOps<T, R> for C {}

/// Extension trait: element-wise arctangent for any [`TensorRef`] implementor.
pub trait TrigAtanOps<T: Clone + EachATan, const MAX_RANK: usize>: TensorRef<T, MAX_RANK> {
    fn try_atan(&self) -> Result<Tensor<T, Global, MAX_RANK>, TensorError> {
        self.view().try_atan()
    }
}

impl<T: Clone + EachATan, const R: usize, C: TensorRef<T, R>> TrigAtanOps<T, R> for C {}

impl<T: Clone + EachSin, const MAX_RANK: usize> Tensor<T, Global, MAX_RANK> {
    /// Element-wise sine: result\[i\] = sin(self\[i\])
    pub fn sin(&self) -> Result<Tensor<T, Global, MAX_RANK>, TensorError> { self.view().try_sin() }

    /// Element-wise sine in-place (infallible — self vs self always matches).
    pub fn sin_inplace(&mut self) {
        try_reborrow_tensor_inplace(self, |view, span| view.try_sin_into(span))
            .expect("inplace trig op on self cannot fail")
    }

    pub fn try_sin_inplace(&mut self) -> Result<(), TensorError> {
        self.sin_inplace();
        Ok(())
    }
}

impl<T: Clone + EachCos, const MAX_RANK: usize> Tensor<T, Global, MAX_RANK> {
    /// Element-wise cosine: result\[i\] = cos(self\[i\])
    pub fn cos(&self) -> Result<Tensor<T, Global, MAX_RANK>, TensorError> { self.view().try_cos() }

    /// Element-wise cosine in-place (infallible — self vs self always matches).
    pub fn cos_inplace(&mut self) {
        try_reborrow_tensor_inplace(self, |view, span| view.try_cos_into(span))
            .expect("inplace trig op on self cannot fail")
    }

    pub fn try_cos_inplace(&mut self) -> Result<(), TensorError> {
        self.cos_inplace();
        Ok(())
    }
}

impl<T: Clone + EachATan, const MAX_RANK: usize> Tensor<T, Global, MAX_RANK> {
    /// Element-wise arctangent: result\[i\] = atan(self\[i\])
    pub fn atan(&self) -> Result<Tensor<T, Global, MAX_RANK>, TensorError> {
        self.view().try_atan()
    }

    /// Element-wise arctangent in-place (infallible — self vs self always matches).
    pub fn atan_inplace(&mut self) {
        try_reborrow_tensor_inplace(self, |view, span| view.try_atan_into(span))
            .expect("inplace trig op on self cannot fail")
    }

    pub fn try_atan_inplace(&mut self) -> Result<(), TensorError> {
        self.atan_inplace();
        Ok(())
    }
}

// endregion: Tensor Trigonometry

// region: Tensor Reductions

impl<T: Clone + Dot, const MAX_RANK: usize> Tensor<T, Global, MAX_RANK> {
    /// Compute the dot product of this array with another.
    ///
    /// Both arrays must be 1D with the same length.
    pub fn dot_product<const OTHER_MAX_RANK: usize>(
        &self,
        other: &Tensor<T, Global, OTHER_MAX_RANK>,
    ) -> Result<T::Output, TensorError> {
        if self.ndim != 1 || other.ndim != 1 {
            return Err(TensorError::DimensionMismatch {
                expected: 1,
                got: if self.ndim != 1 {
                    self.ndim
                } else {
                    other.ndim
                },
            });
        }
        if self.numel() != other.numel() {
            return Err(TensorError::ShapeMismatch {
                axis: 0,
                expected: self.numel(),
                got: other.numel(),
            });
        }
        // Dot::dot returns Option, unwrap since we verified lengths match
        Ok(T::dot(self.as_slice(), other.as_slice()).expect("dot product failed"))
    }
}

type MomentsAxisResult<T, const MAX_RANK: usize> = Result<
    (
        Tensor<<T as ReduceMoments>::SumOutput, Global, MAX_RANK>,
        Tensor<<T as ReduceMoments>::SumSqOutput, Global, MAX_RANK>,
    ),
    TensorError,
>;

impl<'a, T: Clone + ReduceMoments, const MAX_RANK: usize> TensorView<'a, T, MAX_RANK>
where
    T::SumOutput: Clone + Default + core::ops::AddAssign,
    T::SumSqOutput: Clone + Default + core::ops::AddAssign + SumSqToF64,
{
    pub fn try_moments_all(&self) -> Result<(T::SumOutput, T::SumSqOutput), TensorError> {
        let (tail_dims, count, stride) =
            uniform_stride_tail(self.shape(), &self.strides[..self.ndim]);
        if tail_dims == self.ndim {
            let (ptr, count, stride, _) =
                unsafe { normalize_reduction_lane(self.data, count, stride as isize) };
            return Ok(unsafe { T::reduce_moments_raw(ptr, count, stride) });
        }
        if tail_dims >= 2 {
            let new_rank = self.ndim - tail_dims + 1;
            let mut cshape = [0usize; MAX_RANK];
            let mut cstrides = [0isize; MAX_RANK];
            cshape[..new_rank - 1].copy_from_slice(&self.shape[..new_rank - 1]);
            cstrides[..new_rank - 1].copy_from_slice(&self.strides[..new_rank - 1]);
            cshape[new_rank - 1] = count;
            cstrides[new_rank - 1] = self.strides[self.ndim - 1];
            return Ok(unsafe {
                reduce_moments_recursive::<T>(self.data, &cshape[..new_rank], &cstrides[..new_rank])
            });
        }
        Ok(unsafe {
            reduce_moments_recursive::<T>(self.data, self.shape(), &self.strides[..self.ndim])
        })
    }

    pub fn try_moments_axis<I: VectorIndex>(
        &self,
        axis: I,
        keep_dims: bool,
    ) -> MomentsAxisResult<T, MAX_RANK> {
        let axis = normalize_axis(axis, self.ndim)?;
        let output_shape = reduced_shape(self.shape(), axis, keep_dims);
        let mut sums = Tensor::<T::SumOutput, Global, MAX_RANK>::try_full(
            &output_shape,
            T::SumOutput::default(),
        )?;
        let mut sumsqs = Tensor::<T::SumSqOutput, Global, MAX_RANK>::try_full(
            &output_shape,
            T::SumSqOutput::default(),
        )?;
        self.try_moments_axis_into(axis, keep_dims, &mut sums, &mut sumsqs)?;
        Ok((sums, sumsqs))
    }

    pub fn try_moments_axis_into<I: VectorIndex>(
        &self,
        axis: I,
        keep_dims: bool,
        sum_out: &mut Tensor<T::SumOutput, Global, MAX_RANK>,
        sumsq_out: &mut Tensor<T::SumSqOutput, Global, MAX_RANK>,
    ) -> Result<(), TensorError> {
        let axis = normalize_axis(axis, self.ndim)?;
        let expected_shape = reduced_shape(self.shape(), axis, keep_dims);
        validate_same_shape(&expected_shape, sum_out.shape())?;
        validate_same_shape(&expected_shape, sumsq_out.shape())?;

        for_each_axis_lane(
            self,
            axis,
            |lane_ptr, lane_len, lane_stride, output_index| {
                let (lane_ptr, lane_len, lane_stride, _) =
                    unsafe { normalize_reduction_lane(lane_ptr, lane_len, lane_stride) };
                let (sum, sumsq) =
                    unsafe { T::reduce_moments_raw(lane_ptr, lane_len, lane_stride) };
                sum_out.as_mut_slice()[output_index] = sum;
                sumsq_out.as_mut_slice()[output_index] = sumsq;
            },
        );
        Ok(())
    }

    pub fn try_sum_all(&self) -> Result<T::SumOutput, TensorError> { Ok(self.try_moments_all()?.0) }

    pub fn try_sum_axis<I: VectorIndex>(
        &self,
        axis: I,
        keep_dims: bool,
    ) -> Result<Tensor<T::SumOutput, Global, MAX_RANK>, TensorError> {
        let (sums, _) = self.try_moments_axis(axis, keep_dims)?;
        Ok(sums)
    }

    pub fn try_sum_axis_into<I: VectorIndex>(
        &self,
        axis: I,
        keep_dims: bool,
        out: &mut Tensor<T::SumOutput, Global, MAX_RANK>,
    ) -> Result<(), TensorError> {
        let axis = normalize_axis(axis, self.ndim)?;
        let expected_shape = reduced_shape(self.shape(), axis, keep_dims);
        validate_same_shape(&expected_shape, out.shape())?;
        let mut scratch = Tensor::<T::SumSqOutput, Global, MAX_RANK>::try_full(
            &expected_shape,
            T::SumSqOutput::default(),
        )?;
        self.try_moments_axis_into(axis, keep_dims, out, &mut scratch)
    }

    pub fn try_norm_all(&self) -> Result<f64, TensorError> {
        let (_, sumsq) = self.try_moments_all()?;
        Ok(Roots::sqrt(sumsq.to_f64()))
    }

    pub fn try_norm_axis<I: VectorIndex>(
        &self,
        axis: I,
        keep_dims: bool,
    ) -> Result<Tensor<f64, Global, MAX_RANK>, TensorError> {
        let (_, sumsqs) = self.try_moments_axis(axis, keep_dims)?;
        let mut norms = Tensor::<f64, Global, MAX_RANK>::try_full(sumsqs.shape(), 0.0)?;
        for (target, value) in norms
            .as_mut_slice()
            .iter_mut()
            .zip(sumsqs.as_slice().iter())
        {
            *target = Roots::sqrt(SumSqToF64::to_f64(*value));
        }
        Ok(norms)
    }

    pub fn try_norm_axis_into<I: VectorIndex>(
        &self,
        axis: I,
        keep_dims: bool,
        out: &mut Tensor<f64, Global, MAX_RANK>,
    ) -> Result<(), TensorError> {
        let axis = normalize_axis(axis, self.ndim)?;
        let expected_shape = reduced_shape(self.shape(), axis, keep_dims);
        validate_same_shape(&expected_shape, out.shape())?;
        let mut scratch_sum = Tensor::<T::SumOutput, Global, MAX_RANK>::try_full(
            &expected_shape,
            T::SumOutput::default(),
        )?;
        let mut scratch_sumsq = Tensor::<T::SumSqOutput, Global, MAX_RANK>::try_full(
            &expected_shape,
            T::SumSqOutput::default(),
        )?;
        self.try_moments_axis_into(axis, keep_dims, &mut scratch_sum, &mut scratch_sumsq)?;
        for (target, value) in out
            .as_mut_slice()
            .iter_mut()
            .zip(scratch_sumsq.as_slice().iter())
        {
            *target = Roots::sqrt(SumSqToF64::to_f64(*value));
        }
        Ok(())
    }
}

type MinMaxAxisResult<T, const MAX_RANK: usize> = Result<
    MinMaxResult<
        Tensor<<T as ReduceMinMax>::Output, Global, MAX_RANK>,
        Tensor<usize, Global, MAX_RANK>,
    >,
    TensorError,
>;

impl<'a, T: Clone + ReduceMinMax, const MAX_RANK: usize> TensorView<'a, T, MAX_RANK>
where
    T::Output: Clone + Default + PartialOrd,
{
    pub fn try_minmax_all(&self) -> Result<MinMaxResult<T::Output>, TensorError> {
        let (tail_dims, count, stride) =
            uniform_stride_tail(self.shape(), &self.strides[..self.ndim]);
        if tail_dims == self.ndim && count > 0 {
            let innermost_stride = self.strides[self.ndim - 1];
            let (ptr, count, stride, reversed) =
                unsafe { normalize_reduction_lane(self.data, count, innermost_stride) };
            if let Some((min_value, mut min_index, max_value, mut max_index)) =
                unsafe { T::reduce_minmax_raw(ptr, count, stride) }
            {
                if reversed {
                    min_index = count - 1 - min_index;
                    max_index = count - 1 - max_index;
                }
                return Ok(MinMaxResult {
                    min_value,
                    min_index,
                    max_value,
                    max_index,
                });
            }
        }
        if tail_dims >= 2 {
            let new_rank = self.ndim - tail_dims + 1;
            let mut cshape = [0usize; MAX_RANK];
            let mut cstrides = [0isize; MAX_RANK];
            cshape[..new_rank - 1].copy_from_slice(&self.shape[..new_rank - 1]);
            cstrides[..new_rank - 1].copy_from_slice(&self.strides[..new_rank - 1]);
            cshape[new_rank - 1] = count;
            cstrides[new_rank - 1] = self.strides[self.ndim - 1];
            return unsafe {
                reduce_minmax_recursive::<T>(self.data, &cshape[..new_rank], &cstrides[..new_rank], 0)
            }
            .ok_or(TensorError::InvalidShape {
                axis: 0,
                size: self.numel(),
                reason: "min/max reduction undefined for empty or NaN-only input",
            });
        }
        unsafe {
            reduce_minmax_recursive::<T>(self.data, self.shape(), &self.strides[..self.ndim], 0)
        }
        .ok_or(TensorError::InvalidShape {
            axis: 0,
            size: self.numel(),
            reason: "min/max reduction undefined for empty or NaN-only input",
        })
    }

    pub fn try_minmax_axis<I: VectorIndex>(
        &self,
        axis: I,
        keep_dims: bool,
    ) -> MinMaxAxisResult<T, MAX_RANK> {
        let axis = normalize_axis(axis, self.ndim)?;
        let output_shape = reduced_shape(self.shape(), axis, keep_dims);
        let mut min_values =
            Tensor::<T::Output, Global, MAX_RANK>::try_full(&output_shape, T::Output::default())?;
        let mut min_indices = Tensor::<usize, Global, MAX_RANK>::try_full(&output_shape, 0)?;
        let mut max_values =
            Tensor::<T::Output, Global, MAX_RANK>::try_full(&output_shape, T::Output::default())?;
        let mut max_indices = Tensor::<usize, Global, MAX_RANK>::try_full(&output_shape, 0)?;
        self.try_minmax_axis_into(
            axis,
            keep_dims,
            &mut min_values,
            &mut min_indices,
            &mut max_values,
            &mut max_indices,
        )?;
        Ok(MinMaxResult {
            min_value: min_values,
            min_index: min_indices,
            max_value: max_values,
            max_index: max_indices,
        })
    }

    pub fn try_minmax_axis_into<I: VectorIndex>(
        &self,
        axis: I,
        keep_dims: bool,
        min_out: &mut Tensor<T::Output, Global, MAX_RANK>,
        argmin_out: &mut Tensor<usize, Global, MAX_RANK>,
        max_out: &mut Tensor<T::Output, Global, MAX_RANK>,
        argmax_out: &mut Tensor<usize, Global, MAX_RANK>,
    ) -> Result<(), TensorError> {
        let axis = normalize_axis(axis, self.ndim)?;
        let expected_shape = reduced_shape(self.shape(), axis, keep_dims);
        validate_same_shape(&expected_shape, min_out.shape())?;
        validate_same_shape(&expected_shape, argmin_out.shape())?;
        validate_same_shape(&expected_shape, max_out.shape())?;
        validate_same_shape(&expected_shape, argmax_out.shape())?;

        let mut invalid_lane = false;
        for_each_axis_lane(
            self,
            axis,
            |lane_ptr, lane_len, lane_stride, output_index| {
                if invalid_lane {
                    return;
                }
                let (lane_ptr, lane_len, lane_stride, reversed) =
                    unsafe { normalize_reduction_lane(lane_ptr, lane_len, lane_stride) };
                if let Some((min_value, min_index, max_value, max_index)) =
                    unsafe { T::reduce_minmax_raw(lane_ptr, lane_len, lane_stride) }
                {
                    min_out.as_mut_slice()[output_index] = min_value;
                    argmin_out.as_mut_slice()[output_index] = if reversed {
                        lane_len - 1 - min_index
                    } else {
                        min_index
                    };
                    max_out.as_mut_slice()[output_index] = max_value;
                    argmax_out.as_mut_slice()[output_index] = if reversed {
                        lane_len - 1 - max_index
                    } else {
                        max_index
                    };
                } else {
                    invalid_lane = true;
                }
            },
        );

        if invalid_lane {
            return Err(TensorError::InvalidShape {
                axis,
                size: self.shape()[axis],
                reason: "min/max reduction undefined for empty or NaN-only lanes",
            });
        }
        Ok(())
    }

    pub fn try_min_all(&self) -> Result<T::Output, TensorError> {
        Ok(self.try_minmax_all()?.min_value)
    }

    pub fn try_argmin_all(&self) -> Result<usize, TensorError> {
        Ok(self.try_minmax_all()?.min_index)
    }

    pub fn try_max_all(&self) -> Result<T::Output, TensorError> {
        Ok(self.try_minmax_all()?.max_value)
    }

    pub fn try_argmax_all(&self) -> Result<usize, TensorError> {
        Ok(self.try_minmax_all()?.max_index)
    }

    pub fn try_min_axis<I: VectorIndex>(
        &self,
        axis: I,
        keep_dims: bool,
    ) -> Result<Tensor<T::Output, Global, MAX_RANK>, TensorError> {
        Ok(self.try_minmax_axis(axis, keep_dims)?.min_value)
    }

    pub fn try_argmin_axis<I: VectorIndex>(
        &self,
        axis: I,
        keep_dims: bool,
    ) -> Result<Tensor<usize, Global, MAX_RANK>, TensorError> {
        Ok(self.try_minmax_axis(axis, keep_dims)?.min_index)
    }

    pub fn try_max_axis<I: VectorIndex>(
        &self,
        axis: I,
        keep_dims: bool,
    ) -> Result<Tensor<T::Output, Global, MAX_RANK>, TensorError> {
        Ok(self.try_minmax_axis(axis, keep_dims)?.max_value)
    }

    pub fn try_argmax_axis<I: VectorIndex>(
        &self,
        axis: I,
        keep_dims: bool,
    ) -> Result<Tensor<usize, Global, MAX_RANK>, TensorError> {
        Ok(self.try_minmax_axis(axis, keep_dims)?.max_index)
    }
}

/// Extension trait: statistical reductions for any [`TensorRef`] implementor.
pub trait MomentsOps<T: Clone + ReduceMoments, const MAX_RANK: usize>:
    TensorRef<T, MAX_RANK>
where
    T::SumOutput: Clone + Default + core::ops::AddAssign,
    T::SumSqOutput: Clone + Default + core::ops::AddAssign + SumSqToF64,
{
    fn try_moments_all(&self) -> Result<(T::SumOutput, T::SumSqOutput), TensorError> {
        self.view().try_moments_all()
    }

    fn try_moments_axis<I: VectorIndex>(
        &self,
        axis: I,
        keep_dims: bool,
    ) -> MomentsAxisResult<T, MAX_RANK> {
        self.view().try_moments_axis(axis, keep_dims)
    }

    fn try_moments_axis_into<I: VectorIndex>(
        &self,
        axis: I,
        keep_dims: bool,
        sum_out: &mut Tensor<T::SumOutput, Global, MAX_RANK>,
        sumsq_out: &mut Tensor<T::SumSqOutput, Global, MAX_RANK>,
    ) -> Result<(), TensorError> {
        self.view()
            .try_moments_axis_into(axis, keep_dims, sum_out, sumsq_out)
    }

    fn try_sum_all(&self) -> Result<T::SumOutput, TensorError> { self.view().try_sum_all() }

    fn try_sum_axis<I: VectorIndex>(
        &self,
        axis: I,
        keep_dims: bool,
    ) -> Result<Tensor<T::SumOutput, Global, MAX_RANK>, TensorError> {
        self.view().try_sum_axis(axis, keep_dims)
    }

    fn try_sum_axis_into<I: VectorIndex>(
        &self,
        axis: I,
        keep_dims: bool,
        out: &mut Tensor<T::SumOutput, Global, MAX_RANK>,
    ) -> Result<(), TensorError> {
        self.view().try_sum_axis_into(axis, keep_dims, out)
    }

    fn try_norm_all(&self) -> Result<f64, TensorError> { self.view().try_norm_all() }

    fn try_norm_axis<I: VectorIndex>(
        &self,
        axis: I,
        keep_dims: bool,
    ) -> Result<Tensor<f64, Global, MAX_RANK>, TensorError> {
        self.view().try_norm_axis(axis, keep_dims)
    }

    fn try_norm_axis_into<I: VectorIndex>(
        &self,
        axis: I,
        keep_dims: bool,
        out: &mut Tensor<f64, Global, MAX_RANK>,
    ) -> Result<(), TensorError> {
        self.view().try_norm_axis_into(axis, keep_dims, out)
    }
}

impl<T: Clone + ReduceMoments, const R: usize, C: TensorRef<T, R>> MomentsOps<T, R> for C
where
    T::SumOutput: Clone + Default + core::ops::AddAssign,
    T::SumSqOutput: Clone + Default + core::ops::AddAssign + SumSqToF64,
{
}

/// Extension trait: min/max reductions for any [`TensorRef`] implementor.
pub trait MinMaxOps<T: Clone + ReduceMinMax, const MAX_RANK: usize>:
    TensorRef<T, MAX_RANK>
where
    T::Output: Clone + Default + PartialOrd,
{
    fn try_minmax_all(&self) -> Result<MinMaxResult<T::Output>, TensorError> {
        self.view().try_minmax_all()
    }

    fn try_minmax_axis<I: VectorIndex>(
        &self,
        axis: I,
        keep_dims: bool,
    ) -> MinMaxAxisResult<T, MAX_RANK> {
        self.view().try_minmax_axis(axis, keep_dims)
    }

    fn try_minmax_axis_into<I: VectorIndex>(
        &self,
        axis: I,
        keep_dims: bool,
        min_out: &mut Tensor<T::Output, Global, MAX_RANK>,
        argmin_out: &mut Tensor<usize, Global, MAX_RANK>,
        max_out: &mut Tensor<T::Output, Global, MAX_RANK>,
        argmax_out: &mut Tensor<usize, Global, MAX_RANK>,
    ) -> Result<(), TensorError> {
        self.view()
            .try_minmax_axis_into(axis, keep_dims, min_out, argmin_out, max_out, argmax_out)
    }

    fn try_min_all(&self) -> Result<T::Output, TensorError> { self.view().try_min_all() }

    fn try_argmin_all(&self) -> Result<usize, TensorError> { self.view().try_argmin_all() }

    fn try_max_all(&self) -> Result<T::Output, TensorError> { self.view().try_max_all() }

    fn try_argmax_all(&self) -> Result<usize, TensorError> { self.view().try_argmax_all() }

    fn try_min_axis<I: VectorIndex>(
        &self,
        axis: I,
        keep_dims: bool,
    ) -> Result<Tensor<T::Output, Global, MAX_RANK>, TensorError> {
        self.view().try_min_axis(axis, keep_dims)
    }

    fn try_argmin_axis<I: VectorIndex>(
        &self,
        axis: I,
        keep_dims: bool,
    ) -> Result<Tensor<usize, Global, MAX_RANK>, TensorError> {
        self.view().try_argmin_axis(axis, keep_dims)
    }

    fn try_max_axis<I: VectorIndex>(
        &self,
        axis: I,
        keep_dims: bool,
    ) -> Result<Tensor<T::Output, Global, MAX_RANK>, TensorError> {
        self.view().try_max_axis(axis, keep_dims)
    }

    fn try_argmax_axis<I: VectorIndex>(
        &self,
        axis: I,
        keep_dims: bool,
    ) -> Result<Tensor<usize, Global, MAX_RANK>, TensorError> {
        self.view().try_argmax_axis(axis, keep_dims)
    }
}

impl<T: Clone + ReduceMinMax, const R: usize, C: TensorRef<T, R>> MinMaxOps<T, R> for C where
    T::Output: Clone + Default + PartialOrd
{
}

impl<const MAX_RANK: usize> Tensor<f32, Global, MAX_RANK> {
    /// Sum all elements of the tensor.
    pub fn sum(&self) -> f32 { self.try_sum_all().unwrap_or(0.0) as f32 }
}

impl<const MAX_RANK: usize> Tensor<f64, Global, MAX_RANK> {
    /// Sum all elements of the tensor.
    pub fn sum(&self) -> f64 { self.try_sum_all().unwrap_or(0.0) }
}

// endregion: Tensor Reductions

// region: Tests

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{bf16c, f16, f16c, f32c};

    #[test]
    fn tensor_construction() {
        // Creation
        let arr = Tensor::<f32>::try_full(&[3, 4], 1.0f32).unwrap();
        assert_eq!(arr.shape(), &[3, 4]);
        assert_eq!(arr.ndim(), 2);
        assert_eq!(arr.numel(), 12);
        assert!(!arr.is_empty());

        // From slice
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let arr = Tensor::<f32>::try_from_slice(&data, &[3, 4]).unwrap();
        assert_eq!(arr.shape(), &[3, 4]);
        assert_eq!(arr.as_slice(), &data[..]);

        // Clone
        let arr = Tensor::<f32>::try_full(&[3, 4], 2.5f32).unwrap();
        let cloned = arr.clone();
        assert_eq!(cloned.shape(), arr.shape());
        assert_eq!(cloned.as_slice(), arr.as_slice());

        // Error display
        let err = TensorError::AllocationFailed;
        assert_eq!(format!("{}", err), "memory allocation failed");
        let err = TensorError::TooManyRanks { got: 10 };
        assert_eq!(format!("{}", err), "too many ranks: 10");
    }

    #[test]
    fn tensor_views() {
        // Row access
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let arr = Tensor::<f32>::try_from_slice(&data, &[3, 4]).unwrap();
        assert_eq!(arr.row(0), Some(&[0.0, 1.0, 2.0, 3.0][..]));
        assert_eq!(arr.row(1), Some(&[4.0, 5.0, 6.0, 7.0][..]));
        assert_eq!(arr.row(2), Some(&[8.0, 9.0, 10.0, 11.0][..]));
        assert_eq!(arr.row(3), None);

        // Slicing
        let arr = Tensor::<f32>::try_full(&[4, 5], 1.0f32).unwrap();
        let view = arr
            .slice(&[SliceRange::full(), SliceRange::full()])
            .unwrap();
        assert_eq!(view.shape(), &[4, 5]);
        let view = arr
            .slice(&[SliceRange::range(1, 3), SliceRange::full()])
            .unwrap();
        assert_eq!(view.shape(), &[2, 5]);
        let view = arr
            .slice(&[SliceRange::index(0), SliceRange::full()])
            .unwrap();
        assert_eq!(view.shape(), &[5]);
        assert_eq!(view.ndim(), 1);

        // Transpose
        let arr = Tensor::<f32>::try_full(&[3, 4], 1.0f32).unwrap();
        let transposed = arr.transpose().unwrap();
        assert_eq!(transposed.shape(), &[4, 3]);

        // Contiguous check
        let arr = Tensor::<f32>::try_full(&[3, 4], 1.0f32).unwrap();
        let view = arr.view();
        assert!(view.is_contiguous());
        assert!(arr.has_contiguous_rows());

        // Matrix alias
        let mat: Matrix<f32> = Matrix::try_full(&[3, 4], 1.0f32).unwrap();
        assert_eq!(mat.shape(), &[3, 4]);
    }

    #[test]
    fn tensor_scalar_lookup_and_views() {
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let mut tensor = Tensor::<f32>::try_from_slice(&data, &[3, 4]).unwrap();

        assert_eq!(tensor[0_usize], 0.0);
        assert_eq!(tensor[-1_i32], 11.0);
        assert_eq!(tensor[(1_usize, 2_usize)], 6.0);
        assert_eq!(tensor[(2_i32, -1_i32)], 11.0);
        assert_eq!(*tensor.try_flat(5_usize).unwrap(), 5.0);
        assert_eq!(*tensor.try_coords((1_usize, 3_usize)).unwrap(), 7.0);

        tensor[(1_usize, 2_usize)] = 60.0;
        assert_eq!(tensor[(1_usize, 2_usize)], 60.0);
        *tensor.try_coords_mut((2_usize, 0_usize)).unwrap() = 80.0;
        assert_eq!(tensor[(2_usize, 0_usize)], 80.0);

        let view = tensor.view();
        assert_eq!(view[1_usize], 1.0);
        assert_eq!(view[(1_usize, 2_usize)], 60.0);

        let mut span = tensor.span();
        assert_eq!(span[(2_usize, 0_usize)], 80.0);
        span[(0_usize, 1_usize)] = 10.0;
        assert_eq!(span[(0_usize, 1_usize)], 10.0);
    }

    #[test]
    fn tensor_noncontiguous_lookup_and_rank_zero() {
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let tensor = Tensor::<f32>::try_from_slice(&data, &[3, 4]).unwrap();
        let even_columns = tensor
            .slice(&[SliceRange::full(), SliceRange::range_step(0, 4, 2)])
            .unwrap();

        assert_eq!(even_columns.shape(), &[3, 2]);
        assert_eq!(even_columns[0_usize], 0.0);
        assert_eq!(even_columns[1_usize], 2.0);
        assert_eq!(even_columns[2_usize], 4.0);
        assert_eq!(even_columns[(1_usize, 1_usize)], 6.0);
        assert_eq!(*even_columns.try_coords((2_usize, 1_usize)).unwrap(), 10.0);

        let row = tensor.slice_leading(1_usize).unwrap();
        assert_eq!(row.shape(), &[4]);
        assert_eq!(row[-1_i32], 7.0);

        let scalar = tensor
            .slice(&[SliceRange::index(2), SliceRange::index(3)])
            .unwrap();
        assert_eq!(scalar.ndim(), 0);
        assert_eq!(*scalar.try_scalar().unwrap(), 11.0);
    }

    #[test]
    fn tensor_ops() {
        // Reshape
        let arr = Tensor::<f32>::try_full(&[3, 4], 1.0f32).unwrap();
        let reshaped = arr.reshape(&[2, 6]).unwrap();
        assert_eq!(reshaped.shape(), &[2, 6]);
        assert_eq!(reshaped.numel(), 12);

        // Sum f32
        let arr = Tensor::<f32>::try_full(&[100], 1.0f32).unwrap();
        let sum = arr.sum();
        assert!((sum - 100.0).abs() < 0.001);

        // Sum f64
        let arr = Tensor::<f64>::try_full(&[100], 1.0f64).unwrap();
        let sum = arr.sum();
        assert!((sum - 100.0).abs() < 1e-9);
    }

    #[test]
    fn elementwise_and_cast() {
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let a = Tensor::<f32>::try_from_slice(&data, &[3, 4]).unwrap();
        let b = Tensor::<f32>::try_full(&[3, 4], 2.0).unwrap();

        let a_even = a
            .slice(&[SliceRange::full(), SliceRange::range_step(0, 4, 2)])
            .unwrap();
        let b_even = b
            .slice(&[SliceRange::full(), SliceRange::range_step(0, 4, 2)])
            .unwrap();

        let added = a_even.try_add_tensor(&b_even).unwrap();
        assert_eq!(added.shape(), &[3, 2]);
        assert_eq!(added.as_slice(), &[2.0, 4.0, 6.0, 8.0, 10.0, 12.0]);

        let scaled = a_even.try_mul_scalar(0.5).unwrap();
        assert_eq!(scaled.as_slice(), &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);

        let casted = a_even.try_cast_dtype::<f64>().unwrap();
        assert_eq!(casted.shape(), &[3, 2]);
        assert_eq!(casted.as_slice(), &[0.0, 2.0, 4.0, 6.0, 8.0, 10.0]);

        let complex = a_even.try_cast_dtype::<f32c>().unwrap();
        assert_eq!(complex.shape(), &[3, 2]);
        assert_eq!(complex.as_slice()[0], f32c::from_real_imag(0.0, 0.0));
        assert_eq!(complex.as_slice()[5], f32c::from_real_imag(10.0, 0.0));

        let mut out = Tensor::<f32>::try_full(&[3, 4], 0.0).unwrap();
        a.try_add_tensor_into(&b, &mut out).unwrap();
        assert_eq!(out.as_slice()[0], 2.0);
        assert_eq!(out.as_slice()[11], 13.0);

        let mut inplace = Tensor::<f32>::try_from_slice(&data, &[3, 4]).unwrap();
        inplace.try_add_scalar_inplace(1.0).unwrap();
        assert_eq!(inplace.as_slice()[0], 1.0);
        assert_eq!(inplace.as_slice()[11], 12.0);

        let mut trig_out = Tensor::<f32>::try_full(&[3, 2], 0.0).unwrap();
        {
            let mut span = trig_out.span();
            a_even.try_sin_into(&mut span).unwrap();
        }
        assert_eq!(trig_out.shape(), &[3, 2]);
        assert!((trig_out.as_slice()[0] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn reductions_axis_and_strided_views() {
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let a = Tensor::<f32>::try_from_slice(&data, &[3, 4]).unwrap();
        let a_even = a
            .slice(&[SliceRange::full(), SliceRange::range_step(0, 4, 2)])
            .unwrap();

        let sum_all = a_even.try_sum_all().unwrap();
        assert!((sum_all - 30.0).abs() < 1e-6);

        let norm_all = a_even.try_norm_all().unwrap();
        assert!((norm_all - 14.832396974191326).abs() < 1e-9);

        let (sum_axis0, sumsq_axis0) = a_even.try_moments_axis(0, false).unwrap();
        assert_eq!(sum_axis0.shape(), &[2]);
        assert!((sum_axis0.as_slice()[0] - 12.0).abs() < 1e-6);
        assert!((sum_axis0.as_slice()[1] - 18.0).abs() < 1e-6);
        assert!((sumsq_axis0.as_slice()[0] - 80.0).abs() < 1e-6);
        assert!((sumsq_axis0.as_slice()[1] - 140.0).abs() < 1e-6);

        let sum_axis1_keep = a_even.try_sum_axis(-1_i32, true).unwrap();
        assert_eq!(sum_axis1_keep.shape(), &[3, 1]);
        assert!((sum_axis1_keep.as_slice()[0] - 2.0).abs() < 1e-6);
        assert!((sum_axis1_keep.as_slice()[1] - 10.0).abs() < 1e-6);
        assert!((sum_axis1_keep.as_slice()[2] - 18.0).abs() < 1e-6);

        let MinMaxResult {
            min_value: min_axis0,
            min_index: argmin_axis0,
            max_value: max_axis0,
            max_index: argmax_axis0,
        } = a_even.try_minmax_axis(0, false).unwrap();
        assert_eq!(min_axis0.as_slice(), &[0.0, 2.0]);
        assert_eq!(max_axis0.as_slice(), &[8.0, 10.0]);
        assert_eq!(argmin_axis0.as_slice(), &[0, 0]);
        assert_eq!(argmax_axis0.as_slice(), &[2, 2]);

        let reversed = a
            .slice(&[SliceRange::full(), SliceRange::range_step(3, 0, -1)])
            .unwrap();
        let reversed_sum = reversed.try_sum_axis(-1_i32, false).unwrap();
        assert_eq!(reversed_sum.shape(), &[3]);
        assert_eq!(reversed_sum.as_slice(), &[6.0, 18.0, 30.0]);

        let reversed_argmin = reversed.try_argmin_axis(-1_i32, false).unwrap();
        let reversed_argmax = reversed.try_argmax_axis(-1_i32, false).unwrap();
        assert_eq!(reversed_argmin.as_slice(), &[2, 2, 2]);
        assert_eq!(reversed_argmax.as_slice(), &[0, 0, 0]);
    }

    #[test]
    fn complex_elementwise_view_and_owner_paths() {
        let a_values = [f32c { re: 1.0, im: 2.0 }, f32c { re: 3.0, im: 4.0 }];
        let b_values = [f32c { re: 5.0, im: 6.0 }, f32c { re: 7.0, im: 8.0 }];
        let zeros = Tensor::<f32c>::try_full(&[2], f32c { re: 0.0, im: 0.0 }).unwrap();
        let a = Tensor::<f32c>::try_from_slice(&a_values, &[2]).unwrap();
        let b = Tensor::<f32c>::try_from_slice(&b_values, &[2]).unwrap();

        let added = a.try_add_tensor(&b).unwrap();
        assert_eq!(
            added.as_slice(),
            &[f32c { re: 6.0, im: 8.0 }, f32c { re: 10.0, im: 12.0 }]
        );

        let scaled = a
            .scale(f32c { re: 1.0, im: 0.0 }, f32c { re: 1.0, im: 0.0 })
            .unwrap();
        assert_eq!(
            scaled.as_slice(),
            &[f32c { re: 2.0, im: 2.0 }, f32c { re: 4.0, im: 4.0 }]
        );

        let blended = a
            .view()
            .try_blend_tensor(
                &b.view(),
                f32c { re: 1.0, im: 0.0 },
                f32c { re: -1.0, im: 0.0 },
            )
            .unwrap();
        assert_eq!(
            blended.as_slice(),
            &[f32c { re: -4.0, im: -4.0 }, f32c { re: -4.0, im: -4.0 }]
        );

        let fma = a
            .view()
            .try_fma_tensors(
                &b.view(),
                &zeros.view(),
                f32c { re: 1.0, im: 0.0 },
                f32c { re: 0.0, im: 0.0 },
            )
            .unwrap();
        assert_eq!(
            fma.as_slice(),
            &[
                f32c { re: -7.0, im: 16.0 },
                f32c {
                    re: -11.0,
                    im: 52.0
                }
            ]
        );

        let mut inplace = Tensor::<f32c>::try_from_slice(&a_values, &[2]).unwrap();
        inplace.try_add_tensor_inplace(&b).unwrap();
        assert_eq!(inplace.as_slice(), added.as_slice());

        let widened = a.try_cast_dtype::<bf16c>().unwrap();
        assert_eq!(widened.as_slice()[0].re.to_f32(), 1.0);
        assert_eq!(widened.as_slice()[0].im.to_f32(), 2.0);

        let strided = Tensor::<f16c>::try_from_slice(
            &[
                f16c {
                    re: f16::from_f32(1.0),
                    im: f16::from_f32(2.0),
                },
                f16c {
                    re: f16::from_f32(100.0),
                    im: f16::from_f32(101.0),
                },
                f16c {
                    re: f16::from_f32(3.0),
                    im: f16::from_f32(4.0),
                },
                f16c {
                    re: f16::from_f32(102.0),
                    im: f16::from_f32(103.0),
                },
            ],
            &[2, 2],
        )
        .unwrap();
        let complex_column = strided
            .slice(&[SliceRange::full(), SliceRange::range(0, 1)])
            .unwrap();
        let mut out = Tensor::<f16c>::try_full(
            &[2, 1],
            f16c {
                re: f16::ZERO,
                im: f16::ZERO,
            },
        )
        .unwrap();
        {
            let mut span = out.span();
            complex_column
                .try_scale_tensor_into(
                    f16c {
                        re: f16::ONE,
                        im: f16::ZERO,
                    },
                    f16c {
                        re: f16::ZERO,
                        im: f16::ONE,
                    },
                    &mut span,
                )
                .unwrap();
        }
        assert_eq!(
            out.as_slice(),
            &[
                f16c {
                    re: f16::from_f32(1.0),
                    im: f16::from_f32(3.0)
                },
                f16c {
                    re: f16::from_f32(3.0),
                    im: f16::from_f32(5.0)
                }
            ]
        );
    }
    #[test]
    fn tensor_ref_generic() {
        fn shape_of<T: StorageElement, const R: usize>(t: &impl TensorRef<T, R>) -> Vec<usize> {
            t.shape().to_vec()
        }
        let t = Tensor::<f32>::try_full(&[3, 4], 1.0).unwrap();
        assert_eq!(shape_of(&t), vec![3, 4]);
        assert_eq!(shape_of(&t.view()), vec![3, 4]);

        // TensorRef default methods
        assert_eq!(TensorRef::rank(&t), 2);
        assert!(!TensorRef::is_empty(&t));
        assert!(TensorRef::is_contiguous(&t));
        assert!(TensorRef::has_contiguous_rows(&t));
    }

    #[test]
    fn tensor_ref_extension_traits() {
        crate::capabilities::configure_thread();
        let t = Tensor::<f32>::try_full(&[3, 4], 2.0).unwrap();
        let v = t.view();

        // ScaleOps works on both Tensor and TensorView
        let r1 = ScaleOps::try_add_scalar(&t, 1.0).unwrap();
        let r2 = ScaleOps::try_add_scalar(&v, 1.0).unwrap();
        assert_eq!(r1.as_slice(), r2.as_slice());
        assert!((r1.as_slice()[0] - 3.0).abs() < 0.01);

        // SumOps works on Tensor with TensorView as other
        let other = Tensor::<f32>::try_full(&[3, 4], 1.0).unwrap();
        let r3 = SumOps::try_add_tensor(&t, &other).unwrap();
        assert!((r3.as_slice()[0] - 3.0).abs() < 0.01);

        // TrigSinOps works on both
        let small = Tensor::<f32>::try_full(&[4], 0.0).unwrap();
        let r4 = TrigSinOps::try_sin(&small).unwrap();
        assert!((r4.as_slice()[0] - 0.0).abs() < 0.01);

        // MomentsOps works on both
        let sum_t = MomentsOps::try_sum_all(&t).unwrap();
        let sum_v = MomentsOps::try_sum_all(&v).unwrap();
        assert!((sum_t as f32 - 24.0).abs() < 0.01);
        assert!((sum_v as f32 - 24.0).abs() < 0.01);
    }

    #[test]
    fn tensor_allclose_matching() {
        let a = Tensor::<f32>::try_full(&[2, 3], 1.0).unwrap();
        let b = Tensor::<f32>::try_full(&[2, 3], 1.0 + 1e-7).unwrap();
        assert!(a.allclose(&b, 1e-6, 0.0));
    }

    #[test]
    fn tensor_allclose_mismatching() {
        let a = Tensor::<f32>::try_full(&[2, 3], 1.0).unwrap();
        let b = Tensor::<f32>::try_full(&[2, 3], 2.0).unwrap();
        assert!(!a.allclose(&b, 1e-6, 0.0));
    }

    #[test]
    fn tensor_allclose_different_shapes() {
        let a = Tensor::<f32>::try_full(&[2, 3], 1.0).unwrap();
        let b = Tensor::<f32>::try_full(&[3, 2], 1.0).unwrap();
        assert!(!a.allclose(&b, 1e-6, 1e-6));
    }

    #[test]
    fn tensor_view_allclose() {
        let a = Tensor::<f32>::try_full(&[2, 3], 1.0).unwrap();
        let b = Tensor::<f32>::try_full(&[2, 3], 1.0 + 1e-7).unwrap();
        assert!(a.view().allclose(&b.view(), 1e-6, 0.0));
    }

    #[test]
    fn tensor_view_iter_logical_scalars() {
        use crate::types::i4x2;
        // Shape [6] logical → 3 i4x2 storage values
        let mut t = Tensor::<i4x2>::try_zeros(&[6]).unwrap();
        // Set nibbles: storage[0] has dims 0,1; storage[1] has dims 2,3; etc.
        let slice = t.as_mut_slice();
        slice[0] = i4x2::pack([1, 2]);
        slice[1] = i4x2::pack([3, 4]);
        slice[2] = i4x2::pack([5, 6]);

        let vals: Vec<i8> = t.iter().map(|(_, v)| *v).collect();
        assert_eq!(vals, vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn tensor_span_iter_mut_f32() {
        let mut t = Tensor::<f32>::try_full(&[2, 3], 1.0).unwrap();
        for (_, mut val) in &mut t {
            *val += 10.0;
        }
        for (_, v) in t.iter() {
            assert!((*v - 11.0).abs() < 1e-6);
        }
    }

    #[test]
    fn tensor_span_iter_mut_i4x2() {
        use crate::types::i4x2;
        let mut t = Tensor::<i4x2>::try_zeros(&[6]).unwrap();
        for (pos, mut val) in &mut t {
            *val = pos[0] as i8;
        }
        let vals: Vec<i8> = t.iter().map(|(_, v)| *v).collect();
        assert_eq!(vals, vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn tensor_iterator_alias_compat() {
        let t = Tensor::<f32>::try_full(&[2], 1.0).unwrap();
        let _it: TensorIterator<'_, f32> = t.iter();
    }

    #[test]
    fn tensor_tuple_slice_syntax() {
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let t = Tensor::<f32>::try_from_slice(&data, &[3, 4]).unwrap();

        // Full ranges — tuple of two RangeFull
        let v = t.slice((.., ..)).unwrap();
        assert_eq!(v.shape(), &[3, 4]);

        // Index + full — selects row 0
        let v = t.slice((0_usize, ..)).unwrap();
        assert_eq!(v.shape(), &[4]);
        assert_eq!(v.numel(), 4);

        // Full + index — selects column 1
        let v = t.slice((.., 1_usize)).unwrap();
        assert_eq!(v.shape(), &[3]);

        // Range + full — rows 1..3
        let v = t.slice((1..3_usize, ..)).unwrap();
        assert_eq!(v.shape(), &[2, 4]);

        // Full + RangeTo — columns ..2
        let v = t.slice((.., ..2_usize)).unwrap();
        assert_eq!(v.shape(), &[3, 2]);

        // Full + RangeInclusive — columns 0..=2
        let v = t.slice((.., 0..=2_usize)).unwrap();
        assert_eq!(v.shape(), &[3, 3]);

        // Mixed: tuple with SliceRange pass-through (for RangeStep)
        let v = t.slice((.., RangeStep::new(0, 4, 2))).unwrap();
        assert_eq!(v.shape(), &[3, 2]);

        // Backward compat: &[SliceRange]
        let v = t
            .slice(&[SliceRange::full(), SliceRange::index(0)])
            .unwrap();
        assert_eq!(v.shape(), &[3]);

        // Signed (negative) indices — wrap from end
        let v = t.slice((.., -1_isize)).unwrap(); // last column
        assert_eq!(v.shape(), &[3]);

        let v = t.slice((-2_isize.., ..)).unwrap(); // last 2 rows
        assert_eq!(v.shape(), &[2, 4]);

        let v = t.slice((.., -3..-1_isize)).unwrap(); // columns 1..3
        assert_eq!(v.shape(), &[3, 2]);

        // RangeFrom<usize> — now supported
        let v = t.slice((1_usize.., ..)).unwrap(); // rows 1..end
        assert_eq!(v.shape(), &[2, 4]);

        // Mutable slice with tuple
        let mut t = Tensor::<f32>::try_from_slice(&data, &[3, 4]).unwrap();
        let _s = t.slice_mut((.., 0_usize)).unwrap();
    }
}

// endregion: Tests

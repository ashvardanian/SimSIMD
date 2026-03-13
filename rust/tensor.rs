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
use crate::vector::VecIndex;

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
            // Return a dangling but aligned pointer for zero-size allocations
            return Some(NonNull::new(layout.align() as *mut u8).unwrap_or(NonNull::dangling()));
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

/// Fixed-size shape descriptor for error messages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShapeDescriptor {
    dims: [usize; DEFAULT_MAX_RANK],
    ndim: usize,
}

impl ShapeDescriptor {
    /// Create from a slice (truncates if > DEFAULT_MAX_RANK).
    pub fn from_slice(shape: &[usize]) -> Self {
        let mut dims = [0usize; DEFAULT_MAX_RANK];
        let ndim = shape.len().min(DEFAULT_MAX_RANK);
        dims[..ndim].copy_from_slice(&shape[..ndim]);
        Self { dims, ndim }
    }

    /// Return as a slice.
    pub fn as_slice(&self) -> &[usize] { &self.dims[..self.ndim] }
}

impl core::fmt::Display for ShapeDescriptor {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "[")?;
        for (i, &d) in self.as_slice().iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", d)?;
        }
        write!(f, "]")
    }
}

/// Error type for Tensor operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TensorError {
    /// Memory allocation failed.
    AllocationFailed,
    /// Shape mismatch between arrays.
    ShapeMismatch {
        expected: ShapeDescriptor,
        got: ShapeDescriptor,
    },
    /// Invalid shape specification.
    InvalidShape {
        shape: ShapeDescriptor,
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
}

#[cfg(feature = "std")]
impl std::error::Error for TensorError {}

impl core::fmt::Display for TensorError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            TensorError::AllocationFailed => write!(f, "memory allocation failed"),
            TensorError::ShapeMismatch { expected, got } => {
                write!(f, "shape mismatch: expected {}, got {}", expected, got)
            }
            TensorError::InvalidShape { shape, reason } => {
                write!(f, "invalid shape {}: {}", shape, reason)
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
        }
    }
}

// endregion: Error Types

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
    /// Total number of elements.
    len: usize,
    /// Shape dimensions.
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
        if self.len > 0 {
            unsafe {
                // Drop all elements
                core::ptr::drop_in_place(core::ptr::slice_from_raw_parts_mut(
                    self.data.as_ptr(),
                    self.len,
                ));
                // Deallocate buffer using matching SIMD-aligned layout
                let layout = alloc::alloc::Layout::from_size_align(
                    self.len * core::mem::size_of::<T>(),
                    SIMD_ALIGNMENT,
                )
                .unwrap();
                self.alloc.deallocate(
                    NonNull::new_unchecked(self.data.as_ptr() as *mut u8),
                    layout,
                );
            }
        }
    }
}

impl<T: Clone, A: Allocator + Clone, const MAX_RANK: usize> Clone for Tensor<T, A, MAX_RANK> {
    fn clone(&self) -> Self {
        Self::try_from_slice_in(self.as_slice(), self.shape(), self.alloc.clone())
            .expect("clone allocation failed")
    }
}

// Generic allocator-aware methods
impl<T: Clone, A: Allocator, const MAX_RANK: usize> Tensor<T, A, MAX_RANK> {
    /// Creates a new Tensor filled with a value using a custom allocator.
    ///
    /// Returns `Err` if allocation fails or shape is invalid.
    pub fn try_full_in(shape: &[usize], value: T, alloc: A) -> Result<Self, TensorError> {
        if shape.len() > MAX_RANK {
            return Err(TensorError::TooManyRanks { got: shape.len() });
        }

        let total: usize = shape.iter().product();
        if total == 0 && !shape.is_empty() && shape.iter().any(|&d| d == 0) {
            return Err(TensorError::InvalidShape {
                shape: ShapeDescriptor::from_slice(shape),
                reason: "zero-sized dimension",
            });
        }

        // Allocate SIMD-aligned raw buffer using our allocator
        let data = if total == 0 {
            NonNull::dangling()
        } else {
            let layout = alloc::alloc::Layout::from_size_align(
                total * core::mem::size_of::<T>(),
                SIMD_ALIGNMENT,
            )
            .map_err(|_| TensorError::AllocationFailed)?;
            let ptr = alloc
                .allocate(layout)
                .ok_or(TensorError::AllocationFailed)?;
            // Initialize all elements
            unsafe {
                let ptr = ptr.as_ptr() as *mut T;
                for i in 0..total {
                    core::ptr::write(ptr.add(i), value.clone());
                }
                NonNull::new_unchecked(ptr)
            }
        };

        // Build shape and strides arrays
        let mut shape_arr = [0usize; MAX_RANK];
        shape_arr[..shape.len()].copy_from_slice(shape);

        let mut strides_arr = [0isize; MAX_RANK];
        Self::compute_strides_into(shape, &mut strides_arr);

        Ok(Self {
            data,
            len: total,
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
        if total == 0 && !shape.is_empty() && shape.iter().any(|&d| d == 0) {
            return Err(TensorError::InvalidShape {
                shape: ShapeDescriptor::from_slice(shape),
                reason: "zero-sized dimension",
            });
        }

        let data = if total == 0 {
            NonNull::dangling()
        } else {
            let layout = alloc::alloc::Layout::from_size_align(
                total * core::mem::size_of::<T>(),
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
        Self::compute_strides_into(shape, &mut strides_arr);

        Ok(Self {
            data,
            len: total,
            shape: shape_arr,
            strides: strides_arr,
            ndim: shape.len(),
            alloc,
        })
    }

    /// Creates a Tensor from existing slice data using a custom allocator.
    ///
    /// Returns `Err` if shape doesn't match data length or allocation fails.
    pub fn try_from_slice_in(data: &[T], shape: &[usize], alloc: A) -> Result<Self, TensorError> {
        if shape.len() > MAX_RANK {
            return Err(TensorError::TooManyRanks { got: shape.len() });
        }

        let total: usize = shape.iter().product();
        if data.len() != total {
            return Err(TensorError::ShapeMismatch {
                expected: ShapeDescriptor::from_slice(shape),
                got: ShapeDescriptor::from_slice(&[data.len()]),
            });
        }

        // Allocate SIMD-aligned buffer and copy using our allocator
        let ptr = if total == 0 {
            NonNull::dangling()
        } else {
            let layout = alloc::alloc::Layout::from_size_align(
                total * core::mem::size_of::<T>(),
                SIMD_ALIGNMENT,
            )
            .map_err(|_| TensorError::AllocationFailed)?;
            let ptr = alloc
                .allocate(layout)
                .ok_or(TensorError::AllocationFailed)?;
            // Clone all elements
            unsafe {
                let ptr = ptr.as_ptr() as *mut T;
                for i in 0..total {
                    core::ptr::write(ptr.add(i), data[i].clone());
                }
                NonNull::new_unchecked(ptr)
            }
        };

        let mut shape_arr = [0usize; MAX_RANK];
        shape_arr[..shape.len()].copy_from_slice(shape);

        let mut strides_arr = [0isize; MAX_RANK];
        Self::compute_strides_into(shape, &mut strides_arr);

        Ok(Self {
            data: ptr,
            len: total,
            shape: shape_arr,
            strides: strides_arr,
            ndim: shape.len(),
            alloc,
        })
    }

    fn compute_strides_into(shape: &[usize], strides: &mut [isize; MAX_RANK]) {
        let elem_size = core::mem::size_of::<T>();
        if shape.is_empty() {
            return;
        }

        let mut stride = elem_size as isize;
        for i in (0..shape.len()).rev() {
            strides[i] = stride;
            stride *= shape[i] as isize;
        }
    }

    /// Returns a reference to the allocator.
    pub fn allocator(&self) -> &A { &self.alloc }
}

// Methods that don't require Clone
impl<T, A: Allocator, const MAX_RANK: usize> Tensor<T, A, MAX_RANK> {
    /// Returns the shape of the array.
    pub fn shape(&self) -> &[usize] { &self.shape[..self.ndim] }

    /// Returns the number of dimensions.
    pub fn ndim(&self) -> usize { self.ndim }

    /// Returns the number of dimensions (alias for `ndim()`).
    pub fn rank(&self) -> usize { self.ndim }

    /// Returns the total number of elements.
    pub fn numel(&self) -> usize { self.len }

    /// Returns true if the array has no elements.
    pub fn is_empty(&self) -> bool { self.len == 0 }

    /// Returns the stride in bytes for the given dimension.
    pub fn stride_bytes(&self, dim: usize) -> isize { self.strides[dim] }

    /// Returns a pointer to the data.
    pub fn as_ptr(&self) -> *const T { self.data.as_ptr() }

    /// Returns a mutable pointer to the data.
    pub fn as_mut_ptr(&mut self) -> *mut T { self.data.as_ptr() }

    /// Returns the underlying data as a slice.
    pub fn as_slice(&self) -> &[T] {
        unsafe { core::slice::from_raw_parts(self.data.as_ptr(), self.len) }
    }

    /// Returns the underlying data as a mutable slice.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { core::slice::from_raw_parts_mut(self.data.as_ptr(), self.len) }
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
impl<T: Clone, const MAX_RANK: usize> Tensor<T, Global, MAX_RANK> {
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

// endregion: SliceRange

#[doc(hidden)]
pub trait TensorCoordinates {
    const ARITY: usize;

    fn resolve<const MAX_RANK: usize>(
        self,
        shape: &[usize; MAX_RANK],
        ndim: usize,
    ) -> Result<[usize; MAX_RANK], TensorError>;
}

impl<I0: VecIndex, I1: VecIndex> TensorCoordinates for (I0, I1) {
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

impl<I0: VecIndex, I1: VecIndex, I2: VecIndex> TensorCoordinates for (I0, I1, I2) {
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

impl<I0: VecIndex, I1: VecIndex, I2: VecIndex, I3: VecIndex> TensorCoordinates
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

impl<I0: VecIndex, I1: VecIndex, I2: VecIndex, I3: VecIndex, I4: VecIndex> TensorCoordinates
    for (I0, I1, I2, I3, I4)
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

impl<I0: VecIndex, I1: VecIndex, I2: VecIndex, I3: VecIndex, I4: VecIndex, I5: VecIndex>
    TensorCoordinates for (I0, I1, I2, I3, I4, I5)
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
        I0: VecIndex,
        I1: VecIndex,
        I2: VecIndex,
        I3: VecIndex,
        I4: VecIndex,
        I5: VecIndex,
        I6: VecIndex,
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
        I0: VecIndex,
        I1: VecIndex,
        I2: VecIndex,
        I3: VecIndex,
        I4: VecIndex,
        I5: VecIndex,
        I6: VecIndex,
        I7: VecIndex,
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
    /// Number of elements accessible via this view.
    len: usize,
    /// Shape of the view.
    shape: [usize; MAX_RANK],
    /// Strides in bytes.
    strides: [isize; MAX_RANK],
    /// Number of dimensions.
    ndim: usize,
    /// Lifetime marker.
    _marker: PhantomData<&'a T>,
}

impl<'a, T, const MAX_RANK: usize> TensorView<'a, T, MAX_RANK> {
    /// Returns the shape of the view.
    pub fn shape(&self) -> &[usize] { &self.shape[..self.ndim] }

    /// Returns the number of dimensions.
    pub fn ndim(&self) -> usize { self.ndim }

    /// Returns the number of dimensions (alias for `ndim()`).
    pub fn rank(&self) -> usize { self.ndim }

    /// Returns the total number of elements.
    pub fn numel(&self) -> usize { self.len }

    /// Returns true if the view has no elements.
    pub fn is_empty(&self) -> bool { self.len == 0 }

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

    /// Convert to slice (only valid for contiguous views).
    pub fn as_contiguous_slice(&self) -> Option<&[T]> {
        if self.is_contiguous() {
            Some(unsafe { core::slice::from_raw_parts(self.data, self.len) })
        } else {
            None
        }
    }

    /// Try to get an element by flat logical row-major index.
    pub fn try_flat<I: VecIndex>(&self, index: I) -> Result<&T, TensorError> {
        if self.ndim == 0 {
            return Err(TensorError::DimensionMismatch {
                expected: 1,
                got: 0,
            });
        }
        let logical_index = resolve_index_for_size_(index, self.len)?;
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
    pub fn slice_leading<I: VecIndex>(
        &self,
        index: I,
    ) -> Result<TensorView<'a, T, MAX_RANK>, TensorError> {
        let (shape, strides, ndim, offset, len) =
            slice_leading_layout_(&self.shape, &self.strides, self.ndim, index)?;
        Ok(TensorView {
            data: unsafe { (self.data as *const u8).offset(offset) as *const T },
            len,
            shape,
            strides,
            ndim,
            _marker: PhantomData,
        })
    }

    /// Slice the view along multiple dimensions.
    pub fn slice(&self, ranges: &[SliceRange]) -> Result<TensorView<'a, T, MAX_RANK>, TensorError> {
        let (shape, strides, ndim, offset, len) =
            slice_layout_(&self.shape, &self.strides, self.ndim, ranges)?;
        Ok(TensorView {
            data: unsafe { (self.data as *const u8).offset(offset) as *const T },
            len,
            shape,
            strides,
            ndim,
            _marker: PhantomData,
        })
    }
}

impl<'a, T: Clone, const MAX_RANK: usize> TensorView<'a, T, MAX_RANK> {
    /// Copy the view contents to a new owned Tensor.
    pub fn to_owned(&self) -> Result<Tensor<T, Global, MAX_RANK>, TensorError> {
        if self.is_contiguous() {
            let slice = unsafe { core::slice::from_raw_parts(self.data, self.len) };
            Tensor::try_from_slice(slice, self.shape())
        } else {
            // For non-contiguous views, we need to copy element by element
            let mut result = Tensor::try_full(self.shape(), unsafe { (*self.data).clone() })?;
            self.copy_to_contiguous(result.as_mut_slice());
            Ok(result)
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
                    dest[dest_idx] = unsafe { (*elem_ptr).clone() };
                    dest_idx += 1;
                }
            }
        } else {
            // General N-dimensional case: iterate in row-major order
            let mut indices = [0usize; MAX_RANK];
            for dest_idx in 0..self.len {
                // Compute pointer offset
                let mut offset = 0isize;
                for d in 0..self.ndim {
                    offset += indices[d] as isize * self.strides[d];
                }
                let elem_ptr = unsafe { (self.data as *const u8).offset(offset) as *const T };
                dest[dest_idx] = unsafe { (*elem_ptr).clone() };

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
    /// Number of elements accessible via this view.
    len: usize,
    /// Shape of the view.
    shape: [usize; MAX_RANK],
    /// Strides in bytes.
    strides: [isize; MAX_RANK],
    /// Number of dimensions.
    ndim: usize,
    /// Lifetime marker.
    _marker: PhantomData<&'a mut T>,
}

impl<'a, T, const MAX_RANK: usize> TensorSpan<'a, T, MAX_RANK> {
    /// Returns the shape of the view.
    pub fn shape(&self) -> &[usize] { &self.shape[..self.ndim] }

    /// Returns the number of dimensions.
    pub fn ndim(&self) -> usize { self.ndim }

    /// Returns the number of dimensions (alias for `ndim()`).
    pub fn rank(&self) -> usize { self.ndim }

    /// Returns the total number of elements.
    pub fn numel(&self) -> usize { self.len }

    /// Returns true if the view has no elements.
    pub fn is_empty(&self) -> bool { self.len == 0 }

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

    /// Convert to slice (only valid for contiguous views).
    pub fn as_contiguous_slice(&self) -> Option<&[T]> {
        if self.is_contiguous() {
            Some(unsafe { core::slice::from_raw_parts(self.data, self.len) })
        } else {
            None
        }
    }

    /// Convert to mutable slice (only valid for contiguous views).
    pub fn as_contiguous_slice_mut(&mut self) -> Option<&mut [T]> {
        if self.is_contiguous() {
            Some(unsafe { core::slice::from_raw_parts_mut(self.data, self.len) })
        } else {
            None
        }
    }

    /// Reborrow as immutable view.
    pub fn as_view(&self) -> TensorView<'_, T, MAX_RANK> {
        TensorView {
            data: self.data,
            len: self.len,
            shape: self.shape,
            strides: self.strides,
            ndim: self.ndim,
            _marker: PhantomData,
        }
    }

    /// Try to get an element by flat logical row-major index.
    pub fn try_flat<I: VecIndex>(&self, index: I) -> Result<&T, TensorError> {
        if self.ndim == 0 {
            return Err(TensorError::DimensionMismatch {
                expected: 1,
                got: 0,
            });
        }
        let logical_index = resolve_index_for_size_(index, self.len)?;
        let offset = offset_from_flat_(&self.shape, &self.strides, self.ndim, logical_index);
        Ok(unsafe { &*((self.data as *const u8).offset(offset) as *const T) })
    }

    /// Try to get a mutable element by flat logical row-major index.
    pub fn try_flat_mut<I: VecIndex>(&mut self, index: I) -> Result<&mut T, TensorError> {
        if self.ndim == 0 {
            return Err(TensorError::DimensionMismatch {
                expected: 1,
                got: 0,
            });
        }
        let logical_index = resolve_index_for_size_(index, self.len)?;
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
    pub fn slice_leading<I: VecIndex>(
        &self,
        index: I,
    ) -> Result<TensorView<'_, T, MAX_RANK>, TensorError> {
        let (shape, strides, ndim, offset, len) =
            slice_leading_layout_(&self.shape, &self.strides, self.ndim, index)?;
        Ok(TensorView {
            data: unsafe { (self.data as *const u8).offset(offset) as *const T },
            len,
            shape,
            strides,
            ndim,
            _marker: PhantomData,
        })
    }

    /// Slice the leading axis mutably by one index, reducing rank by one.
    pub fn slice_leading_mut<I: VecIndex>(
        &mut self,
        index: I,
    ) -> Result<TensorSpan<'_, T, MAX_RANK>, TensorError> {
        let (shape, strides, ndim, offset, len) =
            slice_leading_layout_(&self.shape, &self.strides, self.ndim, index)?;
        Ok(TensorSpan {
            data: unsafe { (self.data as *mut u8).offset(offset) as *mut T },
            len,
            shape,
            strides,
            ndim,
            _marker: PhantomData,
        })
    }

    /// Slice the span along multiple dimensions.
    pub fn slice(&self, ranges: &[SliceRange]) -> Result<TensorView<'_, T, MAX_RANK>, TensorError> {
        let (shape, strides, ndim, offset, len) =
            slice_layout_(&self.shape, &self.strides, self.ndim, ranges)?;
        Ok(TensorView {
            data: unsafe { (self.data as *const u8).offset(offset) as *const T },
            len,
            shape,
            strides,
            ndim,
            _marker: PhantomData,
        })
    }

    /// Slice the span mutably along multiple dimensions.
    pub fn slice_mut(
        &mut self,
        ranges: &[SliceRange],
    ) -> Result<TensorSpan<'_, T, MAX_RANK>, TensorError> {
        let (shape, strides, ndim, offset, len) =
            slice_layout_(&self.shape, &self.strides, self.ndim, ranges)?;
        Ok(TensorSpan {
            data: unsafe { (self.data as *mut u8).offset(offset) as *mut T },
            len,
            shape,
            strides,
            ndim,
            _marker: PhantomData,
        })
    }
}

// endregion: TensorSpan

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
        let sub_len: usize = sub_shape[..sub_ndim].iter().product();

        self.current += 1;
        Some(TensorView {
            data: sub_ptr,
            len: sub_len,
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
        let sub_len: usize = sub_shape[..sub_ndim].iter().product();

        self.current += 1;
        Some(TensorSpan {
            data: sub_ptr,
            len: sub_len,
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

impl<'a, T, const MAX_RANK: usize> TensorView<'a, T, MAX_RANK> {
    /// Iterate along the given axis, yielding sub-tensor views with rank-1.
    pub fn axis_views<I: VecIndex>(
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

impl<'a, T, const MAX_RANK: usize> TensorView<'a, T, MAX_RANK> {
    /// Transpose (reverse all dimensions, no data copy).
    pub fn transpose(&self) -> Result<TensorView<'a, T, MAX_RANK>, TensorError> {
        if self.ndim < 2 {
            return Ok(TensorView {
                data: self.data,
                len: self.len,
                shape: self.shape,
                strides: self.strides,
                ndim: self.ndim,
                _marker: PhantomData,
            });
        }

        let mut new_shape = [0usize; MAX_RANK];
        let mut new_strides = [0isize; MAX_RANK];
        for i in 0..self.ndim {
            new_shape[i] = self.shape[self.ndim - 1 - i];
            new_strides[i] = self.strides[self.ndim - 1 - i];
        }

        Ok(TensorView {
            data: self.data,
            len: self.len,
            shape: new_shape,
            strides: new_strides,
            ndim: self.ndim,
            _marker: PhantomData,
        })
    }

    /// Reshape the view (must have same total elements, contiguous only).
    pub fn reshape(&self, new_shape: &[usize]) -> Result<TensorView<'a, T, MAX_RANK>, TensorError> {
        if new_shape.len() > MAX_RANK {
            return Err(TensorError::TooManyRanks {
                got: new_shape.len(),
            });
        }

        let new_len: usize = new_shape.iter().product();
        if new_len != self.len {
            return Err(TensorError::ShapeMismatch {
                expected: ShapeDescriptor::from_slice(new_shape),
                got: ShapeDescriptor::from_slice(self.shape()),
            });
        }

        if !self.is_contiguous() {
            return Err(TensorError::NonContiguousRows);
        }

        let mut shape_arr = [0usize; MAX_RANK];
        shape_arr[..new_shape.len()].copy_from_slice(new_shape);

        let mut strides_arr = [0isize; MAX_RANK];
        compute_strides_into_::<T, MAX_RANK>(new_shape, &mut strides_arr);

        Ok(TensorView {
            data: self.data,
            len: self.len,
            shape: shape_arr,
            strides: strides_arr,
            ndim: new_shape.len(),
            _marker: PhantomData,
        })
    }

    /// Flatten to 1D (requires contiguous layout).
    pub fn flatten(&self) -> Result<TensorView<'a, T, MAX_RANK>, TensorError> {
        self.reshape(&[self.len])
    }

    /// Remove dimensions of size 1.
    pub fn squeeze(&self) -> TensorView<'a, T, MAX_RANK> {
        let mut new_shape = [0usize; MAX_RANK];
        let mut new_strides = [0isize; MAX_RANK];
        let mut new_ndim = 0;
        for i in 0..self.ndim {
            if self.shape[i] != 1 {
                new_shape[new_ndim] = self.shape[i];
                new_strides[new_ndim] = self.strides[i];
                new_ndim += 1;
            }
        }
        if new_ndim == 0 {
            new_ndim = 1;
            new_shape[0] = 1;
            new_strides[0] = core::mem::size_of::<T>() as isize;
        }
        TensorView {
            data: self.data,
            len: self.len,
            shape: new_shape,
            strides: new_strides,
            ndim: new_ndim,
            _marker: PhantomData,
        }
    }
}

impl<'a, T, const MAX_RANK: usize> TensorSpan<'a, T, MAX_RANK> {
    /// Transpose (reverse all dimensions, no data copy).
    pub fn transpose(&self) -> Result<TensorSpan<'a, T, MAX_RANK>, TensorError> {
        if self.ndim < 2 {
            return Ok(TensorSpan {
                data: self.data,
                len: self.len,
                shape: self.shape,
                strides: self.strides,
                ndim: self.ndim,
                _marker: PhantomData,
            });
        }

        let mut new_shape = [0usize; MAX_RANK];
        let mut new_strides = [0isize; MAX_RANK];
        for i in 0..self.ndim {
            new_shape[i] = self.shape[self.ndim - 1 - i];
            new_strides[i] = self.strides[self.ndim - 1 - i];
        }

        Ok(TensorSpan {
            data: self.data,
            len: self.len,
            shape: new_shape,
            strides: new_strides,
            ndim: self.ndim,
            _marker: PhantomData,
        })
    }

    /// Reshape the span (must have same total elements, contiguous only).
    pub fn reshape(&self, new_shape: &[usize]) -> Result<TensorSpan<'a, T, MAX_RANK>, TensorError> {
        if new_shape.len() > MAX_RANK {
            return Err(TensorError::TooManyRanks {
                got: new_shape.len(),
            });
        }

        let new_len: usize = new_shape.iter().product();
        if new_len != self.len {
            return Err(TensorError::ShapeMismatch {
                expected: ShapeDescriptor::from_slice(new_shape),
                got: ShapeDescriptor::from_slice(self.shape()),
            });
        }

        if !self.is_contiguous() {
            return Err(TensorError::NonContiguousRows);
        }

        let mut shape_arr = [0usize; MAX_RANK];
        shape_arr[..new_shape.len()].copy_from_slice(new_shape);

        let mut strides_arr = [0isize; MAX_RANK];
        compute_strides_into_::<T, MAX_RANK>(new_shape, &mut strides_arr);

        Ok(TensorSpan {
            data: self.data,
            len: self.len,
            shape: shape_arr,
            strides: strides_arr,
            ndim: new_shape.len(),
            _marker: PhantomData,
        })
    }

    /// Flatten to 1D (requires contiguous layout).
    pub fn flatten(&self) -> Result<TensorSpan<'a, T, MAX_RANK>, TensorError> {
        self.reshape(&[self.len])
    }

    /// Remove dimensions of size 1.
    pub fn squeeze(&self) -> TensorSpan<'a, T, MAX_RANK> {
        let mut new_shape = [0usize; MAX_RANK];
        let mut new_strides = [0isize; MAX_RANK];
        let mut new_ndim = 0;
        for i in 0..self.ndim {
            if self.shape[i] != 1 {
                new_shape[new_ndim] = self.shape[i];
                new_strides[new_ndim] = self.strides[i];
                new_ndim += 1;
            }
        }
        if new_ndim == 0 {
            new_ndim = 1;
            new_shape[0] = 1;
            new_strides[0] = core::mem::size_of::<T>() as isize;
        }
        TensorSpan {
            data: self.data,
            len: self.len,
            shape: new_shape,
            strides: new_strides,
            ndim: new_ndim,
            _marker: PhantomData,
        }
    }
}

impl<'a, T: Clone, const MAX_RANK: usize> TensorSpan<'a, T, MAX_RANK> {
    /// Copy the span contents to a new owned Tensor.
    pub fn to_owned(&self) -> Result<Tensor<T, Global, MAX_RANK>, TensorError> {
        self.as_view().to_owned()
    }
}

impl<'a, T, const MAX_RANK: usize> TensorSpan<'a, T, MAX_RANK> {
    /// Iterate mutably along the given axis, yielding sub-tensor spans with rank-1.
    pub fn axis_spans<I: VecIndex>(
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
    pub fn axis_views<I: VecIndex>(
        &self,
        axis: I,
    ) -> Result<AxisIterator<'_, T, MAX_RANK>, TensorError> {
        self.view().axis_views(axis)
    }

    /// Iterate mutably along the given axis, yielding sub-tensor spans with rank-1.
    pub fn axis_spans<I: VecIndex>(
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

// region: Tensor View and Slice Methods

impl<T, A: Allocator, const MAX_RANK: usize> Tensor<T, A, MAX_RANK> {
    /// Create a view of the entire array.
    pub fn view(&self) -> TensorView<'_, T, MAX_RANK> {
        TensorView {
            data: self.data.as_ptr(),
            len: self.len,
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
            len: self.len,
            shape: self.shape,
            strides: self.strides,
            ndim: self.ndim,
            _marker: PhantomData,
        }
    }

    /// Try to get an element by flat logical row-major index.
    pub fn try_flat<I: VecIndex>(&self, index: I) -> Result<&T, TensorError> {
        if self.ndim == 0 {
            return Err(TensorError::DimensionMismatch {
                expected: 1,
                got: 0,
            });
        }
        let logical_index = resolve_index_for_size_(index, self.len)?;
        let offset = offset_from_flat_(&self.shape, &self.strides, self.ndim, logical_index);
        Ok(unsafe { &*((self.data.as_ptr() as *const u8).offset(offset) as *const T) })
    }

    /// Try to get a mutable element by flat logical row-major index.
    pub fn try_flat_mut<I: VecIndex>(&mut self, index: I) -> Result<&mut T, TensorError> {
        if self.ndim == 0 {
            return Err(TensorError::DimensionMismatch {
                expected: 1,
                got: 0,
            });
        }
        let logical_index = resolve_index_for_size_(index, self.len)?;
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
    pub fn slice_leading<I: VecIndex>(
        &self,
        index: I,
    ) -> Result<TensorView<'_, T, MAX_RANK>, TensorError> {
        let (shape, strides, ndim, offset, len) =
            slice_leading_layout_(&self.shape, &self.strides, self.ndim, index)?;
        Ok(TensorView {
            data: unsafe { (self.data.as_ptr() as *const u8).offset(offset) as *const T },
            len,
            shape,
            strides,
            ndim,
            _marker: PhantomData,
        })
    }

    /// Slice the leading axis mutably by one index, reducing rank by one.
    pub fn slice_leading_mut<I: VecIndex>(
        &mut self,
        index: I,
    ) -> Result<TensorSpan<'_, T, MAX_RANK>, TensorError> {
        let (shape, strides, ndim, offset, len) =
            slice_leading_layout_(&self.shape, &self.strides, self.ndim, index)?;
        Ok(TensorSpan {
            data: unsafe { (self.data.as_ptr() as *mut u8).offset(offset) as *mut T },
            len,
            shape,
            strides,
            ndim,
            _marker: PhantomData,
        })
    }

    /// Slice the array along multiple dimensions.
    ///
    /// # Arguments
    /// * `ranges` - Slice specification for each dimension. Length must match ndim.
    ///
    /// # Example
    /// ```ignore
    /// use numkong::{Tensor, SliceRange};
    ///
    /// let arr = Tensor::<f32>::try_full(&[4, 5], 1.0).unwrap();
    ///
    /// // Get rows 0..2, all columns
    /// let view = arr.slice(&[SliceRange::range(0, 2), SliceRange::full()]).unwrap();
    ///
    /// // Get row 1 (reduces to 1D)
    /// let row = arr.slice(&[SliceRange::index(1), SliceRange::full()]).unwrap();
    /// ```
    pub fn slice(&self, ranges: &[SliceRange]) -> Result<TensorView<'_, T, MAX_RANK>, TensorError> {
        self.view().slice(ranges)
    }

    /// Slice the array mutably along multiple dimensions.
    ///
    /// Each range selects a sub-range along the corresponding axis. The resulting
    /// span has the same rank, with extents and strides adjusted per the ranges.
    /// Strided ranges produce non-contiguous spans.
    pub fn slice_mut(
        &mut self,
        ranges: &[SliceRange],
    ) -> Result<TensorSpan<'_, T, MAX_RANK>, TensorError> {
        let (shape, strides, ndim, offset, len) =
            slice_layout_(&self.shape, &self.strides, self.ndim, ranges)?;
        Ok(TensorSpan {
            data: unsafe { (self.data.as_ptr() as *mut u8).offset(offset) as *mut T },
            len,
            shape,
            strides,
            ndim,
            _marker: PhantomData,
        })
    }

    /// Transpose (reverse all dimensions, no data copy).
    pub fn transpose(&self) -> Result<TensorView<'_, T, MAX_RANK>, TensorError> {
        self.view().transpose()
    }

    /// Reshape the array (must have same total elements, contiguous only).
    pub fn reshape(&self, new_shape: &[usize]) -> Result<TensorView<'_, T, MAX_RANK>, TensorError> {
        if new_shape.len() > MAX_RANK {
            return Err(TensorError::TooManyRanks {
                got: new_shape.len(),
            });
        }

        let new_len: usize = new_shape.iter().product();
        if new_len != self.len {
            return Err(TensorError::ShapeMismatch {
                expected: ShapeDescriptor::from_slice(new_shape),
                got: ShapeDescriptor::from_slice(self.shape()),
            });
        }

        // Check if currently contiguous
        if !self.view().is_contiguous() {
            return Err(TensorError::NonContiguousRows);
        }

        let mut shape_arr = [0usize; MAX_RANK];
        shape_arr[..new_shape.len()].copy_from_slice(new_shape);

        let mut strides_arr = [0isize; MAX_RANK];
        compute_strides_into_::<T, MAX_RANK>(new_shape, &mut strides_arr);

        Ok(TensorView {
            data: self.data.as_ptr(),
            len: self.len,
            shape: shape_arr,
            strides: strides_arr,
            ndim: new_shape.len(),
            _marker: PhantomData,
        })
    }

    /// Flatten to 1D (requires contiguous layout).
    pub fn flatten(&self) -> Result<TensorView<'_, T, MAX_RANK>, TensorError> {
        self.view().flatten()
    }

    /// Remove dimensions of size 1.
    pub fn squeeze(&self) -> TensorView<'_, T, MAX_RANK> { self.view().squeeze() }
}

// endregion: Tensor View and Slice Methods

impl<'a, I: VecIndex, T, const MAX_RANK: usize> Index<I> for TensorView<'a, T, MAX_RANK> {
    type Output = T;

    fn index(&self, index: I) -> &Self::Output {
        self.try_flat(index).expect("view index out of bounds")
    }
}

impl<'a, I0: VecIndex, I1: VecIndex, T, const MAX_RANK: usize> Index<(I0, I1)>
    for TensorView<'a, T, MAX_RANK>
{
    type Output = T;

    fn index(&self, index: (I0, I1)) -> &Self::Output {
        self.try_coords(index)
            .expect("view coordinates out of bounds")
    }
}

impl<'a, I0: VecIndex, I1: VecIndex, I2: VecIndex, T, const MAX_RANK: usize> Index<(I0, I1, I2)>
    for TensorView<'a, T, MAX_RANK>
{
    type Output = T;

    fn index(&self, index: (I0, I1, I2)) -> &Self::Output {
        self.try_coords(index)
            .expect("view coordinates out of bounds")
    }
}

impl<'a, I0: VecIndex, I1: VecIndex, I2: VecIndex, I3: VecIndex, T, const MAX_RANK: usize>
    Index<(I0, I1, I2, I3)> for TensorView<'a, T, MAX_RANK>
{
    type Output = T;

    fn index(&self, index: (I0, I1, I2, I3)) -> &Self::Output {
        self.try_coords(index)
            .expect("view coordinates out of bounds")
    }
}

impl<
        'a,
        I0: VecIndex,
        I1: VecIndex,
        I2: VecIndex,
        I3: VecIndex,
        I4: VecIndex,
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
        I0: VecIndex,
        I1: VecIndex,
        I2: VecIndex,
        I3: VecIndex,
        I4: VecIndex,
        I5: VecIndex,
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
        I0: VecIndex,
        I1: VecIndex,
        I2: VecIndex,
        I3: VecIndex,
        I4: VecIndex,
        I5: VecIndex,
        I6: VecIndex,
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
        I0: VecIndex,
        I1: VecIndex,
        I2: VecIndex,
        I3: VecIndex,
        I4: VecIndex,
        I5: VecIndex,
        I6: VecIndex,
        I7: VecIndex,
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

impl<'a, I: VecIndex, T, const MAX_RANK: usize> Index<I> for TensorSpan<'a, T, MAX_RANK> {
    type Output = T;

    fn index(&self, index: I) -> &Self::Output {
        self.try_flat(index).expect("span index out of bounds")
    }
}

impl<'a, I: VecIndex, T, const MAX_RANK: usize> IndexMut<I> for TensorSpan<'a, T, MAX_RANK> {
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        self.try_flat_mut(index).expect("span index out of bounds")
    }
}

impl<'a, I0: VecIndex, I1: VecIndex, T, const MAX_RANK: usize> Index<(I0, I1)>
    for TensorSpan<'a, T, MAX_RANK>
{
    type Output = T;

    fn index(&self, index: (I0, I1)) -> &Self::Output {
        self.try_coords(index)
            .expect("span coordinates out of bounds")
    }
}

impl<'a, I0: VecIndex, I1: VecIndex, T, const MAX_RANK: usize> IndexMut<(I0, I1)>
    for TensorSpan<'a, T, MAX_RANK>
{
    fn index_mut(&mut self, index: (I0, I1)) -> &mut Self::Output {
        self.try_coords_mut(index)
            .expect("span coordinates out of bounds")
    }
}

impl<'a, I0: VecIndex, I1: VecIndex, I2: VecIndex, T, const MAX_RANK: usize> Index<(I0, I1, I2)>
    for TensorSpan<'a, T, MAX_RANK>
{
    type Output = T;

    fn index(&self, index: (I0, I1, I2)) -> &Self::Output {
        self.try_coords(index)
            .expect("span coordinates out of bounds")
    }
}

impl<'a, I0: VecIndex, I1: VecIndex, I2: VecIndex, T, const MAX_RANK: usize> IndexMut<(I0, I1, I2)>
    for TensorSpan<'a, T, MAX_RANK>
{
    fn index_mut(&mut self, index: (I0, I1, I2)) -> &mut Self::Output {
        self.try_coords_mut(index)
            .expect("span coordinates out of bounds")
    }
}

impl<'a, I0: VecIndex, I1: VecIndex, I2: VecIndex, I3: VecIndex, T, const MAX_RANK: usize>
    Index<(I0, I1, I2, I3)> for TensorSpan<'a, T, MAX_RANK>
{
    type Output = T;

    fn index(&self, index: (I0, I1, I2, I3)) -> &Self::Output {
        self.try_coords(index)
            .expect("span coordinates out of bounds")
    }
}

impl<'a, I0: VecIndex, I1: VecIndex, I2: VecIndex, I3: VecIndex, T, const MAX_RANK: usize>
    IndexMut<(I0, I1, I2, I3)> for TensorSpan<'a, T, MAX_RANK>
{
    fn index_mut(&mut self, index: (I0, I1, I2, I3)) -> &mut Self::Output {
        self.try_coords_mut(index)
            .expect("span coordinates out of bounds")
    }
}

impl<
        'a,
        I0: VecIndex,
        I1: VecIndex,
        I2: VecIndex,
        I3: VecIndex,
        I4: VecIndex,
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
        I0: VecIndex,
        I1: VecIndex,
        I2: VecIndex,
        I3: VecIndex,
        I4: VecIndex,
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
        I0: VecIndex,
        I1: VecIndex,
        I2: VecIndex,
        I3: VecIndex,
        I4: VecIndex,
        I5: VecIndex,
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
        I0: VecIndex,
        I1: VecIndex,
        I2: VecIndex,
        I3: VecIndex,
        I4: VecIndex,
        I5: VecIndex,
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
        I0: VecIndex,
        I1: VecIndex,
        I2: VecIndex,
        I3: VecIndex,
        I4: VecIndex,
        I5: VecIndex,
        I6: VecIndex,
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
        I0: VecIndex,
        I1: VecIndex,
        I2: VecIndex,
        I3: VecIndex,
        I4: VecIndex,
        I5: VecIndex,
        I6: VecIndex,
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
        I0: VecIndex,
        I1: VecIndex,
        I2: VecIndex,
        I3: VecIndex,
        I4: VecIndex,
        I5: VecIndex,
        I6: VecIndex,
        I7: VecIndex,
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
        I0: VecIndex,
        I1: VecIndex,
        I2: VecIndex,
        I3: VecIndex,
        I4: VecIndex,
        I5: VecIndex,
        I6: VecIndex,
        I7: VecIndex,
        T,
        const MAX_RANK: usize,
    > IndexMut<(I0, I1, I2, I3, I4, I5, I6, I7)> for TensorSpan<'a, T, MAX_RANK>
{
    fn index_mut(&mut self, index: (I0, I1, I2, I3, I4, I5, I6, I7)) -> &mut Self::Output {
        self.try_coords_mut(index)
            .expect("span coordinates out of bounds")
    }
}

impl<I: VecIndex, T, A: Allocator, const MAX_RANK: usize> Index<I> for Tensor<T, A, MAX_RANK> {
    type Output = T;

    fn index(&self, index: I) -> &Self::Output {
        self.try_flat(index).expect("tensor index out of bounds")
    }
}

impl<I: VecIndex, T, A: Allocator, const MAX_RANK: usize> IndexMut<I> for Tensor<T, A, MAX_RANK> {
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        self.try_flat_mut(index)
            .expect("tensor index out of bounds")
    }
}

impl<I0: VecIndex, I1: VecIndex, T, A: Allocator, const MAX_RANK: usize> Index<(I0, I1)>
    for Tensor<T, A, MAX_RANK>
{
    type Output = T;

    fn index(&self, index: (I0, I1)) -> &Self::Output {
        self.try_coords(index)
            .expect("tensor coordinates out of bounds")
    }
}

impl<I0: VecIndex, I1: VecIndex, T, A: Allocator, const MAX_RANK: usize> IndexMut<(I0, I1)>
    for Tensor<T, A, MAX_RANK>
{
    fn index_mut(&mut self, index: (I0, I1)) -> &mut Self::Output {
        self.try_coords_mut(index)
            .expect("tensor coordinates out of bounds")
    }
}

impl<I0: VecIndex, I1: VecIndex, I2: VecIndex, T, A: Allocator, const MAX_RANK: usize>
    Index<(I0, I1, I2)> for Tensor<T, A, MAX_RANK>
{
    type Output = T;

    fn index(&self, index: (I0, I1, I2)) -> &Self::Output {
        self.try_coords(index)
            .expect("tensor coordinates out of bounds")
    }
}

impl<I0: VecIndex, I1: VecIndex, I2: VecIndex, T, A: Allocator, const MAX_RANK: usize>
    IndexMut<(I0, I1, I2)> for Tensor<T, A, MAX_RANK>
{
    fn index_mut(&mut self, index: (I0, I1, I2)) -> &mut Self::Output {
        self.try_coords_mut(index)
            .expect("tensor coordinates out of bounds")
    }
}

impl<
        I0: VecIndex,
        I1: VecIndex,
        I2: VecIndex,
        I3: VecIndex,
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
        I0: VecIndex,
        I1: VecIndex,
        I2: VecIndex,
        I3: VecIndex,
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
        I0: VecIndex,
        I1: VecIndex,
        I2: VecIndex,
        I3: VecIndex,
        I4: VecIndex,
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
        I0: VecIndex,
        I1: VecIndex,
        I2: VecIndex,
        I3: VecIndex,
        I4: VecIndex,
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
        I0: VecIndex,
        I1: VecIndex,
        I2: VecIndex,
        I3: VecIndex,
        I4: VecIndex,
        I5: VecIndex,
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
        I0: VecIndex,
        I1: VecIndex,
        I2: VecIndex,
        I3: VecIndex,
        I4: VecIndex,
        I5: VecIndex,
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
        I0: VecIndex,
        I1: VecIndex,
        I2: VecIndex,
        I3: VecIndex,
        I4: VecIndex,
        I5: VecIndex,
        I6: VecIndex,
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
        I0: VecIndex,
        I1: VecIndex,
        I2: VecIndex,
        I3: VecIndex,
        I4: VecIndex,
        I5: VecIndex,
        I6: VecIndex,
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
        I0: VecIndex,
        I1: VecIndex,
        I2: VecIndex,
        I3: VecIndex,
        I4: VecIndex,
        I5: VecIndex,
        I6: VecIndex,
        I7: VecIndex,
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
        I0: VecIndex,
        I1: VecIndex,
        I2: VecIndex,
        I3: VecIndex,
        I4: VecIndex,
        I5: VecIndex,
        I6: VecIndex,
        I7: VecIndex,
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
    Err(TensorError::ShapeMismatch {
        expected: ShapeDescriptor::from_slice(lhs),
        got: ShapeDescriptor::from_slice(rhs),
    })
}

#[inline]
fn normalize_axis<I: VecIndex>(axis: I, ndim: usize) -> Result<usize, TensorError> {
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

#[inline]
fn resolve_index_for_size_<I: VecIndex>(index: I, size: usize) -> Result<usize, TensorError> {
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
) -> Result<([usize; MAX_RANK], [isize; MAX_RANK], usize, isize, usize), TensorError> {
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
                        shape: ShapeDescriptor::from_slice(&shape[..ndim]),
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

fn slice_leading_layout_<I: VecIndex, const MAX_RANK: usize>(
    shape: &[usize; MAX_RANK],
    strides: &[isize; MAX_RANK],
    ndim: usize,
    index: I,
) -> Result<([usize; MAX_RANK], [isize; MAX_RANK], usize, isize, usize), TensorError> {
    if ndim == 0 {
        return Err(TensorError::IndexOutOfBounds { index: 0, size: 0 });
    }

    let leading = resolve_index_for_size_(index, shape[0])?;
    let mut new_shape = [0usize; MAX_RANK];
    let mut new_strides = [0isize; MAX_RANK];
    for dim in 1..ndim {
        new_shape[dim - 1] = shape[dim];
        new_strides[dim - 1] = strides[dim];
    }
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
) -> Option<(T::Output, usize, T::Output, usize)>
where
    T: ReduceMinMax,
    T::Output: Clone + PartialOrd,
{
    if shape.is_empty() {
        return T::reduce_minmax_raw(data, 1, core::mem::size_of::<T>()).map(
            |(min_value, _, max_value, _)| (min_value, logical_offset, max_value, logical_offset),
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
                (
                    min_value,
                    logical_offset + min_index,
                    max_value,
                    logical_offset + max_index,
                )
            },
        );
    }

    let inner_len: usize = shape[1..].iter().product();
    let mut best_min: Option<(T::Output, usize)> = None;
    let mut best_max: Option<(T::Output, usize)> = None;

    for index in 0..shape[0] {
        let child_ptr = (data as *const u8).offset(index as isize * strides[0]) as *const T;
        let child_offset = logical_offset + index * inner_len;
        if let Some((child_min, child_min_index, child_max, child_max_index)) =
            reduce_minmax_recursive::<T>(child_ptr, &shape[1..], &strides[1..], child_offset)
        {
            match &best_min {
                Some((best_value, _))
                    if child_min.partial_cmp(best_value) != Some(core::cmp::Ordering::Less) => {}
                _ => best_min = Some((child_min, child_min_index)),
            }
            match &best_max {
                Some((best_value, _))
                    if child_max.partial_cmp(best_value) != Some(core::cmp::Ordering::Greater) => {}
                _ => best_max = Some((child_max, child_max_index)),
            }
        }
    }

    match (best_min, best_max) {
        (Some((min_value, min_index)), Some((max_value, max_index))) => {
            Some((min_value, min_index, max_value, max_index))
        }
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

fn try_alloc_output_like<D: Clone, F, const MAX_RANK: usize>(
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
        len: tensor.len,
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
        len: view.len,
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

impl<T: Clone + EachScale, const MAX_RANK: usize> Tensor<T, Global, MAX_RANK>
where
    T::Scalar: From<f32> + core::ops::Mul<Output = T::Scalar> + Copy,
{
    pub fn try_add_scalar(
        &self,
        scalar: T::Scalar,
    ) -> Result<Tensor<T, Global, MAX_RANK>, TensorError> {
        self.view().try_add_scalar(scalar)
    }

    pub fn try_sub_scalar(
        &self,
        scalar: T::Scalar,
    ) -> Result<Tensor<T, Global, MAX_RANK>, TensorError> {
        self.view().try_sub_scalar(scalar)
    }

    pub fn try_mul_scalar(
        &self,
        scalar: T::Scalar,
    ) -> Result<Tensor<T, Global, MAX_RANK>, TensorError> {
        self.view().try_mul_scalar(scalar)
    }

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
        try_reborrow_tensor_inplace(self, |view, span| view.try_add_scalar_into(scalar, span))
    }

    pub fn try_sub_scalar_inplace(&mut self, scalar: T::Scalar) -> Result<(), TensorError> {
        try_reborrow_tensor_inplace(self, |view, span| view.try_sub_scalar_into(scalar, span))
    }

    pub fn try_mul_scalar_inplace(&mut self, scalar: T::Scalar) -> Result<(), TensorError> {
        try_reborrow_tensor_inplace(self, |view, span| view.try_mul_scalar_into(scalar, span))
    }
}

impl<T: Clone + EachSum, const MAX_RANK: usize> Tensor<T, Global, MAX_RANK> {
    pub fn try_add_tensor(
        &self,
        other: &Tensor<T, Global, MAX_RANK>,
    ) -> Result<Tensor<T, Global, MAX_RANK>, TensorError> {
        self.view().try_add_tensor(&other.view())
    }

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

impl<T: Clone + EachBlend, const MAX_RANK: usize> Tensor<T, Global, MAX_RANK>
where
    T::Scalar: From<f32> + Copy,
{
    pub fn try_sub_tensor(
        &self,
        other: &Tensor<T, Global, MAX_RANK>,
    ) -> Result<Tensor<T, Global, MAX_RANK>, TensorError> {
        self.view().try_sub_tensor(&other.view())
    }

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

impl<T: Clone + EachFMA, const MAX_RANK: usize> Tensor<T, Global, MAX_RANK>
where
    T::Scalar: From<f32> + Copy,
{
    pub fn try_mul_tensor(
        &self,
        other: &Tensor<T, Global, MAX_RANK>,
    ) -> Result<Tensor<T, Global, MAX_RANK>, TensorError> {
        self.view().try_mul_tensor(&other.view())
    }

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

impl<S: Clone + CastDtype, const MAX_RANK: usize> Tensor<S, Global, MAX_RANK> {
    pub fn try_cast_dtype<D: Clone + CastDtype>(
        &self,
    ) -> Result<Tensor<D, Global, MAX_RANK>, TensorError> {
        self.view().try_cast_dtype()
    }

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

impl<T: Clone + EachSin, const MAX_RANK: usize> Tensor<T, Global, MAX_RANK> {
    /// Element-wise sine: result\[i\] = sin(self\[i\])
    ///
    /// Input values are in radians.
    pub fn sin(&self) -> Result<Tensor<T, Global, MAX_RANK>, TensorError> { self.view().try_sin() }

    pub fn try_sin(&self) -> Result<Tensor<T, Global, MAX_RANK>, TensorError> {
        self.view().try_sin()
    }

    /// Element-wise sine in-place: self\[i\] = sin(self\[i\])
    pub fn sin_inplace(&mut self) { let _ = self.try_sin_inplace(); }

    pub fn try_sin_inplace(&mut self) -> Result<(), TensorError> {
        try_reborrow_tensor_inplace(self, |view, span| view.try_sin_into(span))
    }
}

impl<T: Clone + EachCos, const MAX_RANK: usize> Tensor<T, Global, MAX_RANK> {
    /// Element-wise cosine: result\[i\] = cos(self\[i\])
    ///
    /// Input values are in radians.
    pub fn cos(&self) -> Result<Tensor<T, Global, MAX_RANK>, TensorError> { self.view().try_cos() }

    pub fn try_cos(&self) -> Result<Tensor<T, Global, MAX_RANK>, TensorError> {
        self.view().try_cos()
    }

    /// Element-wise cosine in-place: self\[i\] = cos(self\[i\])
    pub fn cos_inplace(&mut self) { let _ = self.try_cos_inplace(); }

    pub fn try_cos_inplace(&mut self) -> Result<(), TensorError> {
        try_reborrow_tensor_inplace(self, |view, span| view.try_cos_into(span))
    }
}

impl<T: Clone + EachATan, const MAX_RANK: usize> Tensor<T, Global, MAX_RANK> {
    /// Element-wise arctangent: result\[i\] = atan(self\[i\])
    ///
    /// Output values are in radians in the range (-π/2, π/2).
    pub fn atan(&self) -> Result<Tensor<T, Global, MAX_RANK>, TensorError> {
        self.view().try_atan()
    }

    pub fn try_atan(&self) -> Result<Tensor<T, Global, MAX_RANK>, TensorError> {
        self.view().try_atan()
    }

    /// Element-wise arctangent in-place: self\[i\] = atan(self\[i\])
    pub fn atan_inplace(&mut self) { let _ = self.try_atan_inplace(); }

    pub fn try_atan_inplace(&mut self) -> Result<(), TensorError> {
        try_reborrow_tensor_inplace(self, |view, span| view.try_atan_into(span))
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
        if self.len != other.len {
            return Err(TensorError::ShapeMismatch {
                expected: ShapeDescriptor::from_slice(self.shape()),
                got: ShapeDescriptor::from_slice(other.shape()),
            });
        }
        // Dot::dot returns Option, unwrap since we verified lengths match
        Ok(T::dot(self.as_slice(), other.as_slice()).expect("dot product failed"))
    }
}

impl<'a, T: Clone + ReduceMoments, const MAX_RANK: usize> TensorView<'a, T, MAX_RANK>
where
    T::SumOutput: Clone + Default + core::ops::AddAssign,
    T::SumSqOutput: Clone + Default + core::ops::AddAssign + SumSqToF64,
{
    pub fn try_moments_all(&self) -> Result<(T::SumOutput, T::SumSqOutput), TensorError> {
        Ok(unsafe {
            reduce_moments_recursive::<T>(self.data, self.shape(), &self.strides[..self.ndim])
        })
    }

    pub fn try_moments_axis<I: VecIndex>(
        &self,
        axis: I,
        keep_dims: bool,
    ) -> Result<
        (
            Tensor<T::SumOutput, Global, MAX_RANK>,
            Tensor<T::SumSqOutput, Global, MAX_RANK>,
        ),
        TensorError,
    > {
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

    pub fn try_moments_axis_into<I: VecIndex>(
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

    pub fn try_sum_axis<I: VecIndex>(
        &self,
        axis: I,
        keep_dims: bool,
    ) -> Result<Tensor<T::SumOutput, Global, MAX_RANK>, TensorError> {
        let (sums, _) = self.try_moments_axis(axis, keep_dims)?;
        Ok(sums)
    }

    pub fn try_sum_axis_into<I: VecIndex>(
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

    pub fn try_norm_axis<I: VecIndex>(
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
            *target = Roots::sqrt(value.clone().to_f64());
        }
        Ok(norms)
    }

    pub fn try_norm_axis_into<I: VecIndex>(
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
            *target = Roots::sqrt(value.clone().to_f64());
        }
        Ok(())
    }
}

impl<'a, T: Clone + ReduceMinMax, const MAX_RANK: usize> TensorView<'a, T, MAX_RANK>
where
    T::Output: Clone + Default + PartialOrd,
{
    pub fn try_minmax_all(&self) -> Result<(T::Output, usize, T::Output, usize), TensorError> {
        unsafe {
            reduce_minmax_recursive::<T>(self.data, self.shape(), &self.strides[..self.ndim], 0)
        }
        .ok_or(TensorError::InvalidShape {
            shape: ShapeDescriptor::from_slice(self.shape()),
            reason: "min/max reduction undefined for empty or NaN-only input",
        })
    }

    pub fn try_minmax_axis<I: VecIndex>(
        &self,
        axis: I,
        keep_dims: bool,
    ) -> Result<
        (
            Tensor<T::Output, Global, MAX_RANK>,
            Tensor<usize, Global, MAX_RANK>,
            Tensor<T::Output, Global, MAX_RANK>,
            Tensor<usize, Global, MAX_RANK>,
        ),
        TensorError,
    > {
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
        Ok((min_values, min_indices, max_values, max_indices))
    }

    pub fn try_minmax_axis_into<I: VecIndex>(
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
                shape: ShapeDescriptor::from_slice(self.shape()),
                reason: "min/max reduction undefined for empty or NaN-only lanes",
            });
        }
        Ok(())
    }

    pub fn try_min_all(&self) -> Result<T::Output, TensorError> { Ok(self.try_minmax_all()?.0) }

    pub fn try_argmin_all(&self) -> Result<usize, TensorError> { Ok(self.try_minmax_all()?.1) }

    pub fn try_max_all(&self) -> Result<T::Output, TensorError> { Ok(self.try_minmax_all()?.2) }

    pub fn try_argmax_all(&self) -> Result<usize, TensorError> { Ok(self.try_minmax_all()?.3) }

    pub fn try_min_axis<I: VecIndex>(
        &self,
        axis: I,
        keep_dims: bool,
    ) -> Result<Tensor<T::Output, Global, MAX_RANK>, TensorError> {
        let (min_values, _, _, _) = self.try_minmax_axis(axis, keep_dims)?;
        Ok(min_values)
    }

    pub fn try_argmin_axis<I: VecIndex>(
        &self,
        axis: I,
        keep_dims: bool,
    ) -> Result<Tensor<usize, Global, MAX_RANK>, TensorError> {
        let (_, argmin_values, _, _) = self.try_minmax_axis(axis, keep_dims)?;
        Ok(argmin_values)
    }

    pub fn try_max_axis<I: VecIndex>(
        &self,
        axis: I,
        keep_dims: bool,
    ) -> Result<Tensor<T::Output, Global, MAX_RANK>, TensorError> {
        let (_, _, max_values, _) = self.try_minmax_axis(axis, keep_dims)?;
        Ok(max_values)
    }

    pub fn try_argmax_axis<I: VecIndex>(
        &self,
        axis: I,
        keep_dims: bool,
    ) -> Result<Tensor<usize, Global, MAX_RANK>, TensorError> {
        let (_, _, _, argmax_values) = self.try_minmax_axis(axis, keep_dims)?;
        Ok(argmax_values)
    }
}

impl<T: Clone + ReduceMoments, const MAX_RANK: usize> Tensor<T, Global, MAX_RANK>
where
    T::SumOutput: Clone + Default + core::ops::AddAssign,
    T::SumSqOutput: Clone + Default + core::ops::AddAssign + SumSqToF64,
{
    pub fn try_moments_all(&self) -> Result<(T::SumOutput, T::SumSqOutput), TensorError> {
        self.view().try_moments_all()
    }

    pub fn try_moments_axis<I: VecIndex>(
        &self,
        axis: I,
        keep_dims: bool,
    ) -> Result<
        (
            Tensor<T::SumOutput, Global, MAX_RANK>,
            Tensor<T::SumSqOutput, Global, MAX_RANK>,
        ),
        TensorError,
    > {
        self.view().try_moments_axis(axis, keep_dims)
    }

    pub fn try_moments_axis_into<I: VecIndex>(
        &self,
        axis: I,
        keep_dims: bool,
        sum_out: &mut Tensor<T::SumOutput, Global, MAX_RANK>,
        sumsq_out: &mut Tensor<T::SumSqOutput, Global, MAX_RANK>,
    ) -> Result<(), TensorError> {
        self.view()
            .try_moments_axis_into(axis, keep_dims, sum_out, sumsq_out)
    }

    pub fn try_sum_all(&self) -> Result<T::SumOutput, TensorError> { self.view().try_sum_all() }

    pub fn try_sum_axis<I: VecIndex>(
        &self,
        axis: I,
        keep_dims: bool,
    ) -> Result<Tensor<T::SumOutput, Global, MAX_RANK>, TensorError> {
        self.view().try_sum_axis(axis, keep_dims)
    }

    pub fn try_sum_axis_into<I: VecIndex>(
        &self,
        axis: I,
        keep_dims: bool,
        out: &mut Tensor<T::SumOutput, Global, MAX_RANK>,
    ) -> Result<(), TensorError> {
        self.view().try_sum_axis_into(axis, keep_dims, out)
    }

    pub fn try_norm_all(&self) -> Result<f64, TensorError> { self.view().try_norm_all() }

    pub fn try_norm_axis<I: VecIndex>(
        &self,
        axis: I,
        keep_dims: bool,
    ) -> Result<Tensor<f64, Global, MAX_RANK>, TensorError> {
        self.view().try_norm_axis(axis, keep_dims)
    }

    pub fn try_norm_axis_into<I: VecIndex>(
        &self,
        axis: I,
        keep_dims: bool,
        out: &mut Tensor<f64, Global, MAX_RANK>,
    ) -> Result<(), TensorError> {
        self.view().try_norm_axis_into(axis, keep_dims, out)
    }
}

impl<T: Clone + ReduceMinMax, const MAX_RANK: usize> Tensor<T, Global, MAX_RANK>
where
    T::Output: Clone + Default + PartialOrd,
{
    pub fn try_minmax_all(&self) -> Result<(T::Output, usize, T::Output, usize), TensorError> {
        self.view().try_minmax_all()
    }

    pub fn try_minmax_axis<I: VecIndex>(
        &self,
        axis: I,
        keep_dims: bool,
    ) -> Result<
        (
            Tensor<T::Output, Global, MAX_RANK>,
            Tensor<usize, Global, MAX_RANK>,
            Tensor<T::Output, Global, MAX_RANK>,
            Tensor<usize, Global, MAX_RANK>,
        ),
        TensorError,
    > {
        self.view().try_minmax_axis(axis, keep_dims)
    }

    pub fn try_minmax_axis_into<I: VecIndex>(
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

    pub fn try_min_all(&self) -> Result<T::Output, TensorError> { self.view().try_min_all() }

    pub fn try_argmin_all(&self) -> Result<usize, TensorError> { self.view().try_argmin_all() }

    pub fn try_max_all(&self) -> Result<T::Output, TensorError> { self.view().try_max_all() }

    pub fn try_argmax_all(&self) -> Result<usize, TensorError> { self.view().try_argmax_all() }

    pub fn try_min_axis<I: VecIndex>(
        &self,
        axis: I,
        keep_dims: bool,
    ) -> Result<Tensor<T::Output, Global, MAX_RANK>, TensorError> {
        self.view().try_min_axis(axis, keep_dims)
    }

    pub fn try_argmin_axis<I: VecIndex>(
        &self,
        axis: I,
        keep_dims: bool,
    ) -> Result<Tensor<usize, Global, MAX_RANK>, TensorError> {
        self.view().try_argmin_axis(axis, keep_dims)
    }

    pub fn try_max_axis<I: VecIndex>(
        &self,
        axis: I,
        keep_dims: bool,
    ) -> Result<Tensor<T::Output, Global, MAX_RANK>, TensorError> {
        self.view().try_max_axis(axis, keep_dims)
    }

    pub fn try_argmax_axis<I: VecIndex>(
        &self,
        axis: I,
        keep_dims: bool,
    ) -> Result<Tensor<usize, Global, MAX_RANK>, TensorError> {
        self.view().try_argmax_axis(axis, keep_dims)
    }
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

        let (min_axis0, argmin_axis0, max_axis0, argmax_axis0) =
            a_even.try_minmax_axis(0, false).unwrap();
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
}

// endregion: Tests

//! N-dimensional tensor types and GEMM operations.
//!
//! This module provides:
//!
//! - [`Tensor`]: N-dimensional array with customizable rank and allocator
//! - [`TensorView`]: Immutable view into a tensor
//! - [`TensorViewMut`]: Mutable view into a tensor
//! - [`Matrix`]: Type alias for 2D tensors
//! - [`MatrixMultiplier`]: Pre-packed matrix for efficient GEMM
//!
//! # Example
//!
//! ```rust,ignore
//! use numkong::{Tensor, MatrixMultiplier};
//!
//! let a = Tensor::<f32>::try_new(&[1024, 512], 1.0).unwrap();
//! let b = Tensor::<f32>::try_new(&[256, 512], 1.0).unwrap();
//!
//! // Pack B once, multiply many times
//! let packed_b = MatrixMultiplier::try_pack(&b).unwrap();
//! let c = a.matmul(&packed_b);  // Returns (1024 × 256)
//! ```

extern crate alloc;

use core::marker::PhantomData;
use core::ptr::NonNull;

use crate::numerics::{ATan, Cos, Dot, Scale, Sin, Sum, WSum, FMA};
use crate::scalars::{bf16, e4m3, e5m2, f16};

/// Size type used in C FFI to match `nk_size_t` which is always `uint64_t`.
type u64size = u64;

#[link(name = "numkong")]
extern "C" {

    fn nk_dots_packed_size_f32(n: u64size, k: u64size) -> u64size;
    fn nk_dots_pack_f32(b: *const f32, n: u64size, k: u64size, b_stride: u64size, packed: *mut u8);
    fn nk_dots_packed_f32(
        a: *const f32,
        packed: *const u8,
        c: *mut f32,
        m: u64size,
        n: u64size,
        k: u64size,
        a_stride: u64size,
        c_stride: u64size,
    );

    fn nk_dots_packed_size_f64(n: u64size, k: u64size) -> u64size;
    fn nk_dots_pack_f64(b: *const f64, n: u64size, k: u64size, b_stride: u64size, packed: *mut u8);
    fn nk_dots_packed_f64(
        a: *const f64,
        packed: *const u8,
        c: *mut f64,
        m: u64size,
        n: u64size,
        k: u64size,
        a_stride: u64size,
        c_stride: u64size,
    );

    fn nk_dots_packed_size_f16(n: u64size, k: u64size) -> u64size;
    fn nk_dots_pack_f16(b: *const u16, n: u64size, k: u64size, b_stride: u64size, packed: *mut u8);
    fn nk_dots_packed_f16(
        a: *const u16,
        packed: *const u8,
        c: *mut f32,
        m: u64size,
        n: u64size,
        k: u64size,
        a_stride: u64size,
        c_stride: u64size,
    );

    fn nk_dots_packed_size_bf16(n: u64size, k: u64size) -> u64size;
    fn nk_dots_pack_bf16(b: *const u16, n: u64size, k: u64size, b_stride: u64size, packed: *mut u8);
    fn nk_dots_packed_bf16(
        a: *const u16,
        packed: *const u8,
        c: *mut f32,
        m: u64size,
        n: u64size,
        k: u64size,
        a_stride: u64size,
        c_stride: u64size,
    );

    fn nk_dots_packed_size_i8(n: u64size, k: u64size) -> u64size;
    fn nk_dots_pack_i8(b: *const i8, n: u64size, k: u64size, b_stride: u64size, packed: *mut u8);
    fn nk_dots_packed_i8(
        a: *const i8,
        packed: *const u8,
        c: *mut i32,
        m: u64size,
        n: u64size,
        k: u64size,
        a_stride: u64size,
        c_stride: u64size,
    );

    fn nk_dots_packed_size_u8(n: u64size, k: u64size) -> u64size;
    fn nk_dots_pack_u8(b: *const u8, n: u64size, k: u64size, b_stride: u64size, packed: *mut u8);
    fn nk_dots_packed_u8(
        a: *const u8,
        packed: *const u8,
        c: *mut u32,
        m: u64size,
        n: u64size,
        k: u64size,
        a_stride: u64size,
        c_stride: u64size,
    );

    fn nk_dots_packed_size_e4m3(n: u64size, k: u64size) -> u64size;
    fn nk_dots_pack_e4m3(b: *const u8, n: u64size, k: u64size, b_stride: u64size, packed: *mut u8);
    fn nk_dots_packed_e4m3(
        a: *const u8,
        packed: *const u8,
        c: *mut f32,
        m: u64size,
        n: u64size,
        k: u64size,
        a_stride: u64size,
        c_stride: u64size,
    );

    fn nk_dots_packed_size_e5m2(n: u64size, k: u64size) -> u64size;
    fn nk_dots_pack_e5m2(b: *const u8, n: u64size, k: u64size, b_stride: u64size, packed: *mut u8);
    fn nk_dots_packed_e5m2(
        a: *const u8,
        packed: *const u8,
        c: *mut f32,
        m: u64size,
        n: u64size,
        k: u64size,
        a_stride: u64size,
        c_stride: u64size,
    );
}

// region: Constants and Allocator

/// Default maximum rank for tensors.
pub const DEFAULT_MAX_RANK: usize = 8;

/// Alignment for SIMD-friendly allocations (64 bytes for AVX-512).
pub const SIMD_ALIGNMENT: usize = 64;

/// Memory allocator trait for custom allocation strategies.
///
/// Implement this trait to use custom allocators (arena, pool, etc.) with
/// [`Tensor`] and [`MatrixMultiplier`].
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
    pub fn as_slice(&self) -> &[usize] {
        &self.dims[..self.ndim]
    }
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

// region: Dots Trait

/// Low-level trait for packed GEMM operations.
///
/// Computes C = A × Bᵀ where B is pre-packed for optimal memory access.
/// All strides are in bytes.
pub trait Dots: Sized + Clone {
    /// Accumulator type for the multiplication.
    type Accumulator: Clone + Default;

    /// Returns the size in bytes needed for the packed B matrix buffer.
    fn dots_packed_size(n: usize, k: usize) -> usize;

    /// Packs the B matrix into an optimized backend-specific layout.
    ///
    /// # Safety
    /// - `b` must point to valid memory for `n * k` elements
    /// - `packed` must point to a buffer of at least `dots_packed_size(n, k)` bytes
    unsafe fn dots_pack(b: *const Self, n: usize, k: usize, b_stride: usize, packed: *mut u8);

    /// Computes C = A × Bᵀ using packed B.
    ///
    /// # Safety
    /// - `a` must point to valid memory for `m * k` elements with given stride
    /// - `packed` must be a buffer previously filled by `dots_pack`
    /// - `c` must point to valid memory for `m * n` elements with given stride
    unsafe fn dots(
        a: *const Self,
        packed: *const u8,
        c: *mut Self::Accumulator,
        m: usize,
        n: usize,
        k: usize,
        a_stride: usize,
        c_stride: usize,
    );
}

impl Dots for f32 {
    type Accumulator = f32;

    fn dots_packed_size(n: usize, k: usize) -> usize {
        unsafe { nk_dots_packed_size_f32(n as u64size, k as u64size) as usize }
    }

    unsafe fn dots_pack(b: *const Self, n: usize, k: usize, b_stride: usize, packed: *mut u8) {
        nk_dots_pack_f32(b, n as u64size, k as u64size, b_stride as u64size, packed)
    }

    unsafe fn dots(
        a: *const Self,
        packed: *const u8,
        c: *mut Self::Accumulator,
        m: usize,
        n: usize,
        k: usize,
        a_stride: usize,
        c_stride: usize,
    ) {
        nk_dots_packed_f32(
            a,
            packed,
            c,
            m as u64size,
            n as u64size,
            k as u64size,
            a_stride as u64size,
            c_stride as u64size,
        )
    }
}

impl Dots for f64 {
    type Accumulator = f64;

    fn dots_packed_size(n: usize, k: usize) -> usize {
        unsafe { nk_dots_packed_size_f64(n as u64size, k as u64size) as usize }
    }

    unsafe fn dots_pack(b: *const Self, n: usize, k: usize, b_stride: usize, packed: *mut u8) {
        nk_dots_pack_f64(b, n as u64size, k as u64size, b_stride as u64size, packed)
    }

    unsafe fn dots(
        a: *const Self,
        packed: *const u8,
        c: *mut Self::Accumulator,
        m: usize,
        n: usize,
        k: usize,
        a_stride: usize,
        c_stride: usize,
    ) {
        nk_dots_packed_f64(
            a,
            packed,
            c,
            m as u64size,
            n as u64size,
            k as u64size,
            a_stride as u64size,
            c_stride as u64size,
        )
    }
}

impl Dots for f16 {
    type Accumulator = f32;

    fn dots_packed_size(n: usize, k: usize) -> usize {
        unsafe { nk_dots_packed_size_f16(n as u64size, k as u64size) as usize }
    }

    unsafe fn dots_pack(b: *const Self, n: usize, k: usize, b_stride: usize, packed: *mut u8) {
        nk_dots_pack_f16(
            b as *const u16,
            n as u64size,
            k as u64size,
            b_stride as u64size,
            packed,
        )
    }

    unsafe fn dots(
        a: *const Self,
        packed: *const u8,
        c: *mut Self::Accumulator,
        m: usize,
        n: usize,
        k: usize,
        a_stride: usize,
        c_stride: usize,
    ) {
        nk_dots_packed_f16(
            a as *const u16,
            packed,
            c,
            m as u64size,
            n as u64size,
            k as u64size,
            a_stride as u64size,
            c_stride as u64size,
        )
    }
}

impl Dots for bf16 {
    type Accumulator = f32;

    fn dots_packed_size(n: usize, k: usize) -> usize {
        unsafe { nk_dots_packed_size_bf16(n as u64size, k as u64size) as usize }
    }

    unsafe fn dots_pack(b: *const Self, n: usize, k: usize, b_stride: usize, packed: *mut u8) {
        nk_dots_pack_bf16(
            b as *const u16,
            n as u64size,
            k as u64size,
            b_stride as u64size,
            packed,
        )
    }

    unsafe fn dots(
        a: *const Self,
        packed: *const u8,
        c: *mut Self::Accumulator,
        m: usize,
        n: usize,
        k: usize,
        a_stride: usize,
        c_stride: usize,
    ) {
        nk_dots_packed_bf16(
            a as *const u16,
            packed,
            c,
            m as u64size,
            n as u64size,
            k as u64size,
            a_stride as u64size,
            c_stride as u64size,
        )
    }
}

impl Dots for i8 {
    type Accumulator = i32;

    fn dots_packed_size(n: usize, k: usize) -> usize {
        unsafe { nk_dots_packed_size_i8(n as u64size, k as u64size) as usize }
    }

    unsafe fn dots_pack(b: *const Self, n: usize, k: usize, b_stride: usize, packed: *mut u8) {
        nk_dots_pack_i8(b, n as u64size, k as u64size, b_stride as u64size, packed)
    }

    unsafe fn dots(
        a: *const Self,
        packed: *const u8,
        c: *mut Self::Accumulator,
        m: usize,
        n: usize,
        k: usize,
        a_stride: usize,
        c_stride: usize,
    ) {
        nk_dots_packed_i8(
            a,
            packed,
            c,
            m as u64size,
            n as u64size,
            k as u64size,
            a_stride as u64size,
            c_stride as u64size,
        )
    }
}

impl Dots for u8 {
    type Accumulator = u32;

    fn dots_packed_size(n: usize, k: usize) -> usize {
        unsafe { nk_dots_packed_size_u8(n as u64size, k as u64size) as usize }
    }

    unsafe fn dots_pack(b: *const Self, n: usize, k: usize, b_stride: usize, packed: *mut u8) {
        nk_dots_pack_u8(b, n as u64size, k as u64size, b_stride as u64size, packed)
    }

    unsafe fn dots(
        a: *const Self,
        packed: *const u8,
        c: *mut Self::Accumulator,
        m: usize,
        n: usize,
        k: usize,
        a_stride: usize,
        c_stride: usize,
    ) {
        nk_dots_packed_u8(
            a,
            packed,
            c,
            m as u64size,
            n as u64size,
            k as u64size,
            a_stride as u64size,
            c_stride as u64size,
        )
    }
}

impl Dots for e4m3 {
    type Accumulator = f32;

    fn dots_packed_size(n: usize, k: usize) -> usize {
        unsafe { nk_dots_packed_size_e4m3(n as u64size, k as u64size) as usize }
    }

    unsafe fn dots_pack(b: *const Self, n: usize, k: usize, b_stride: usize, packed: *mut u8) {
        nk_dots_pack_e4m3(
            b as *const u8,
            n as u64size,
            k as u64size,
            b_stride as u64size,
            packed,
        )
    }

    unsafe fn dots(
        a: *const Self,
        packed: *const u8,
        c: *mut Self::Accumulator,
        m: usize,
        n: usize,
        k: usize,
        a_stride: usize,
        c_stride: usize,
    ) {
        nk_dots_packed_e4m3(
            a as *const u8,
            packed,
            c,
            m as u64size,
            n as u64size,
            k as u64size,
            a_stride as u64size,
            c_stride as u64size,
        )
    }
}

impl Dots for e5m2 {
    type Accumulator = f32;

    fn dots_packed_size(n: usize, k: usize) -> usize {
        unsafe { nk_dots_packed_size_e5m2(n as u64size, k as u64size) as usize }
    }

    unsafe fn dots_pack(b: *const Self, n: usize, k: usize, b_stride: usize, packed: *mut u8) {
        nk_dots_pack_e5m2(
            b as *const u8,
            n as u64size,
            k as u64size,
            b_stride as u64size,
            packed,
        )
    }

    unsafe fn dots(
        a: *const Self,
        packed: *const u8,
        c: *mut Self::Accumulator,
        m: usize,
        n: usize,
        k: usize,
        a_stride: usize,
        c_stride: usize,
    ) {
        nk_dots_packed_e5m2(
            a as *const u8,
            packed,
            c,
            m as u64size,
            n as u64size,
            k as u64size,
            a_stride as u64size,
            c_stride as u64size,
        )
    }
}

// endregion: Dots Trait

// region: Tensor

/// N-dimensional array with NumKong-accelerated operations.
///
/// Uses raw memory allocation (no std::Vec) for maximum control.
///
/// Supports:
/// - Slicing and subviews (zero-copy)
/// - Matrix multiplication with [`MatrixMultiplier`]
/// - Reductions (sum, min, max)
/// - Elementwise ops (scale, sum, wsum, fma)
/// - Trigonometry (sin, cos, atan)
///
/// # Example
///
/// ```rust,ignore
/// use numkong::{Tensor, MatrixMultiplier};
///
/// let a = Tensor::<f32>::try_new(&[1024, 512], 1.0).unwrap();
/// let b = Tensor::<f32>::try_new(&[256, 512], 1.0).unwrap();
///
/// // Pack B once, multiply many times
/// let packed_b = MatrixMultiplier::try_pack(&b).unwrap();
/// let c = a.matmul(&packed_b);  // Returns (1024 × 256)
/// ```
pub struct Tensor<T, A: Allocator = Global, const MAX_RANK: usize = DEFAULT_MAX_RANK> {
    /// Raw pointer to data buffer.
    data: NonNull<T>,
    /// Total number of elements.
    len: usize,
    /// Shape dimensions.
    shape: [usize; MAX_RANK],
    /// Strides in bytes.
    strides: [usize; MAX_RANK],
    /// Number of dimensions.
    ndim: usize,
    /// Allocator instance.
    alloc: A,
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
                // Deallocate buffer using our allocator
                let layout = alloc::alloc::Layout::array::<T>(self.len).unwrap();
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
    pub fn try_new_in(shape: &[usize], value: T, alloc: A) -> Result<Self, TensorError> {
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

        // Allocate raw buffer using our allocator
        let data = if total == 0 {
            NonNull::dangling()
        } else {
            let layout = alloc::alloc::Layout::array::<T>(total)
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

        let mut strides_arr = [0usize; MAX_RANK];
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

    /// Creates an Tensor from existing slice data using a custom allocator.
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

        // Allocate and copy using our allocator
        let ptr = if total == 0 {
            NonNull::dangling()
        } else {
            let layout = alloc::alloc::Layout::array::<T>(total)
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

        let mut strides_arr = [0usize; MAX_RANK];
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

    fn compute_strides_into(shape: &[usize], strides: &mut [usize; MAX_RANK]) {
        let elem_size = core::mem::size_of::<T>();
        if shape.is_empty() {
            return;
        }

        let mut stride = elem_size;
        for i in (0..shape.len()).rev() {
            strides[i] = stride;
            stride *= shape[i];
        }
    }

    /// Returns a reference to the allocator.
    pub fn allocator(&self) -> &A {
        &self.alloc
    }
}

// Methods that don't require Clone
impl<T, A: Allocator, const MAX_RANK: usize> Tensor<T, A, MAX_RANK> {
    /// Returns the shape of the array.
    pub fn shape(&self) -> &[usize] {
        &self.shape[..self.ndim]
    }

    /// Returns the number of dimensions.
    pub fn ndim(&self) -> usize {
        self.ndim
    }

    /// Returns the total number of elements.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns true if the array has no elements.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the stride in bytes for the given dimension.
    pub fn stride(&self, dim: usize) -> usize {
        self.strides[dim]
    }

    /// Returns a pointer to the data.
    pub fn as_ptr(&self) -> *const T {
        self.data.as_ptr()
    }

    /// Returns a mutable pointer to the data.
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.data.as_ptr()
    }

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
        self.strides[1] == core::mem::size_of::<T>()
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
    pub fn try_new(shape: &[usize], value: T) -> Result<Self, TensorError> {
        Self::try_new_in(shape, value, Global)
    }

    /// Creates an Tensor from existing slice data using the global allocator.
    ///
    /// Returns `Err` if shape doesn't match data length or allocation fails.
    pub fn try_from_slice(data: &[T], shape: &[usize]) -> Result<Self, TensorError> {
        Self::try_from_slice_in(data, shape, Global)
    }

    /// Convenience constructor that panics on error.
    pub fn new(shape: &[usize], value: T) -> Self {
        Self::try_new(shape, value).expect("Tensor::new failed")
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
    pub fn full() -> Self {
        Self::Full
    }

    /// Create a single index.
    pub fn index(i: usize) -> Self {
        Self::Index(i)
    }

    /// Create a range from start to end.
    pub fn range(start: usize, end: usize) -> Self {
        Self::Range { start, end }
    }

    /// Create a range with step.
    pub fn range_step(start: usize, end: usize, step: isize) -> Self {
        Self::RangeStep { start, end, step }
    }
}

// endregion: SliceRange

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
    strides: [usize; MAX_RANK],
    /// Number of dimensions.
    ndim: usize,
    /// Lifetime marker.
    _marker: PhantomData<&'a T>,
}

impl<'a, T, const MAX_RANK: usize> TensorView<'a, T, MAX_RANK> {
    /// Returns the shape of the view.
    pub fn shape(&self) -> &[usize] {
        &self.shape[..self.ndim]
    }

    /// Returns the number of dimensions.
    pub fn ndim(&self) -> usize {
        self.ndim
    }

    /// Returns the total number of elements.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns true if the view has no elements.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the stride in bytes for the given dimension.
    pub fn stride(&self, dim: usize) -> usize {
        self.strides[dim]
    }

    /// Returns a pointer to the first element.
    pub fn as_ptr(&self) -> *const T {
        self.data
    }

    /// Check if the view has contiguous rows.
    pub fn has_contiguous_rows(&self) -> bool {
        if self.ndim != 2 {
            return false;
        }
        self.strides[1] == core::mem::size_of::<T>()
    }

    /// Check if the entire view is contiguous in memory.
    pub fn is_contiguous(&self) -> bool {
        if self.ndim == 0 {
            return true;
        }
        let elem_size = core::mem::size_of::<T>();
        let mut expected_stride = elem_size;
        for i in (0..self.ndim).rev() {
            if self.strides[i] != expected_stride {
                return false;
            }
            expected_stride *= self.shape[i];
        }
        true
    }

    /// Get element at flat index (only valid for contiguous views).
    ///
    /// # Safety
    /// Caller must ensure the view is contiguous and index is in bounds.
    pub unsafe fn get_unchecked(&self, index: usize) -> &T {
        &*self.data.add(index)
    }

    /// Convert to slice (only valid for contiguous views).
    pub fn as_contiguous_slice(&self) -> Option<&[T]> {
        if self.is_contiguous() {
            Some(unsafe { core::slice::from_raw_parts(self.data, self.len) })
        } else {
            None
        }
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
            let mut result = Tensor::try_new(self.shape(), unsafe { (*self.data).clone() })?;
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
                let row_ptr = unsafe { (self.data as *const u8).add(r * row_stride) as *const T };
                for c in 0..cols {
                    let elem_ptr =
                        unsafe { (row_ptr as *const u8).add(c * col_stride) as *const T };
                    dest[dest_idx] = unsafe { (*elem_ptr).clone() };
                    dest_idx += 1;
                }
            }
        } else {
            // General N-dimensional case: iterate in row-major order
            let mut indices = [0usize; MAX_RANK];
            for dest_idx in 0..self.len {
                // Compute pointer offset
                let mut offset = 0usize;
                for d in 0..self.ndim {
                    offset += indices[d] * self.strides[d];
                }
                let elem_ptr = unsafe { (self.data as *const u8).add(offset) as *const T };
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

// endregion: TensorView

// region: TensorViewMut

/// A mutable view into a Tensor.
pub struct TensorViewMut<'a, T, const MAX_RANK: usize = DEFAULT_MAX_RANK> {
    /// Pointer to first element of view.
    data: *mut T,
    /// Number of elements accessible via this view.
    len: usize,
    /// Shape of the view.
    shape: [usize; MAX_RANK],
    /// Strides in bytes.
    strides: [usize; MAX_RANK],
    /// Number of dimensions.
    ndim: usize,
    /// Lifetime marker.
    _marker: PhantomData<&'a mut T>,
}

impl<'a, T, const MAX_RANK: usize> TensorViewMut<'a, T, MAX_RANK> {
    /// Returns the shape of the view.
    pub fn shape(&self) -> &[usize] {
        &self.shape[..self.ndim]
    }

    /// Returns the number of dimensions.
    pub fn ndim(&self) -> usize {
        self.ndim
    }

    /// Returns the total number of elements.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns true if the view has no elements.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the stride in bytes for the given dimension.
    pub fn stride(&self, dim: usize) -> usize {
        self.strides[dim]
    }

    /// Returns a pointer to the first element.
    pub fn as_ptr(&self) -> *const T {
        self.data
    }

    /// Returns a mutable pointer to the first element.
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.data
    }

    /// Check if the view has contiguous rows.
    pub fn has_contiguous_rows(&self) -> bool {
        if self.ndim != 2 {
            return false;
        }
        self.strides[1] == core::mem::size_of::<T>()
    }

    /// Check if the entire view is contiguous in memory.
    pub fn is_contiguous(&self) -> bool {
        if self.ndim == 0 {
            return true;
        }
        let elem_size = core::mem::size_of::<T>();
        let mut expected_stride = elem_size;
        for i in (0..self.ndim).rev() {
            if self.strides[i] != expected_stride {
                return false;
            }
            expected_stride *= self.shape[i];
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
}

// endregion: TensorViewMut

// region: Tensor View and Slice Methods

impl<T: Clone, const MAX_RANK: usize> Tensor<T, Global, MAX_RANK> {
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

    /// Create a mutable view of the entire array.
    pub fn view_mut(&mut self) -> TensorViewMut<'_, T, MAX_RANK> {
        TensorViewMut {
            data: self.data.as_ptr(),
            len: self.len,
            shape: self.shape,
            strides: self.strides,
            ndim: self.ndim,
            _marker: PhantomData,
        }
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
    /// let arr = Tensor::<f32>::try_new(&[4, 5], 1.0).unwrap();
    ///
    /// // Get rows 0..2, all columns
    /// let view = arr.slice(&[SliceRange::range(0, 2), SliceRange::full()]).unwrap();
    ///
    /// // Get row 1 (reduces to 1D)
    /// let row = arr.slice(&[SliceRange::index(1), SliceRange::full()]).unwrap();
    /// ```
    pub fn slice(&self, ranges: &[SliceRange]) -> Result<TensorView<'_, T, MAX_RANK>, TensorError> {
        if ranges.len() != self.ndim {
            return Err(TensorError::DimensionMismatch {
                expected: self.ndim,
                got: ranges.len(),
            });
        }

        let mut new_shape = [0usize; MAX_RANK];
        let mut new_strides = [0usize; MAX_RANK];
        let mut new_ndim = 0usize;
        let mut offset = 0usize;

        for (dim, range) in ranges.iter().enumerate() {
            let dim_size = self.shape[dim];
            let dim_stride = self.strides[dim];

            match *range {
                SliceRange::Full => {
                    new_shape[new_ndim] = dim_size;
                    new_strides[new_ndim] = dim_stride;
                    new_ndim += 1;
                }
                SliceRange::Index(i) => {
                    if i >= dim_size {
                        return Err(TensorError::IndexOutOfBounds {
                            index: i,
                            size: dim_size,
                        });
                    }
                    // Single index reduces dimension (doesn't add to new shape)
                    offset += i * dim_stride;
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
                    offset += start * dim_stride;
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
                            shape: ShapeDescriptor::from_slice(self.shape()),
                            reason: "step cannot be zero",
                        });
                    }
                    let count = if step > 0 {
                        (end.saturating_sub(start) + (step as usize) - 1) / (step as usize)
                    } else {
                        let abs_step = (-step) as usize;
                        (start.saturating_sub(end) + abs_step - 1) / abs_step
                    };
                    new_shape[new_ndim] = count;
                    // Stride can be negative for reversed views
                    new_strides[new_ndim] = (dim_stride as isize * step) as usize;
                    new_ndim += 1;
                    offset += start * dim_stride;
                }
            }
        }

        let new_len: usize = new_shape[..new_ndim].iter().product();
        let new_ptr = unsafe { (self.data.as_ptr() as *const u8).add(offset) as *const T };

        Ok(TensorView {
            data: new_ptr,
            len: new_len,
            shape: new_shape,
            strides: new_strides,
            ndim: new_ndim,
            _marker: PhantomData,
        })
    }

    /// Slice the array mutably along multiple dimensions.
    pub fn slice_mut(
        &mut self,
        ranges: &[SliceRange],
    ) -> Result<TensorViewMut<'_, T, MAX_RANK>, TensorError> {
        if ranges.len() != self.ndim {
            return Err(TensorError::DimensionMismatch {
                expected: self.ndim,
                got: ranges.len(),
            });
        }

        let mut new_shape = [0usize; MAX_RANK];
        let mut new_strides = [0usize; MAX_RANK];
        let mut new_ndim = 0usize;
        let mut offset = 0usize;

        for (dim, range) in ranges.iter().enumerate() {
            let dim_size = self.shape[dim];
            let dim_stride = self.strides[dim];

            match *range {
                SliceRange::Full => {
                    new_shape[new_ndim] = dim_size;
                    new_strides[new_ndim] = dim_stride;
                    new_ndim += 1;
                }
                SliceRange::Index(i) => {
                    if i >= dim_size {
                        return Err(TensorError::IndexOutOfBounds {
                            index: i,
                            size: dim_size,
                        });
                    }
                    offset += i * dim_stride;
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
                    offset += start * dim_stride;
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
                            shape: ShapeDescriptor::from_slice(self.shape()),
                            reason: "step cannot be zero",
                        });
                    }
                    let count = if step > 0 {
                        (end.saturating_sub(start) + (step as usize) - 1) / (step as usize)
                    } else {
                        let abs_step = (-step) as usize;
                        (start.saturating_sub(end) + abs_step - 1) / abs_step
                    };
                    new_shape[new_ndim] = count;
                    new_strides[new_ndim] = (dim_stride as isize * step) as usize;
                    new_ndim += 1;
                    offset += start * dim_stride;
                }
            }
        }

        let new_len: usize = new_shape[..new_ndim].iter().product();
        let new_ptr = unsafe { (self.data.as_ptr() as *mut u8).add(offset) as *mut T };

        Ok(TensorViewMut {
            data: new_ptr,
            len: new_len,
            shape: new_shape,
            strides: new_strides,
            ndim: new_ndim,
            _marker: PhantomData,
        })
    }

    /// Transpose a 2D array (swaps strides, no data copy).
    pub fn t(&self) -> Result<TensorView<'_, T, MAX_RANK>, TensorError> {
        if self.ndim != 2 {
            return Err(TensorError::DimensionMismatch {
                expected: 2,
                got: self.ndim,
            });
        }

        let mut new_shape = [0usize; MAX_RANK];
        let mut new_strides = [0usize; MAX_RANK];
        new_shape[0] = self.shape[1];
        new_shape[1] = self.shape[0];
        new_strides[0] = self.strides[1];
        new_strides[1] = self.strides[0];

        Ok(TensorView {
            data: self.data.as_ptr(),
            len: self.len,
            shape: new_shape,
            strides: new_strides,
            ndim: 2,
            _marker: PhantomData,
        })
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

        let mut strides_arr = [0usize; MAX_RANK];
        Tensor::<T, Global, MAX_RANK>::compute_strides_into(new_shape, &mut strides_arr);

        Ok(TensorView {
            data: self.data.as_ptr(),
            len: self.len,
            shape: shape_arr,
            strides: strides_arr,
            ndim: new_shape.len(),
            _marker: PhantomData,
        })
    }
}

// endregion: Tensor View and Slice Methods

// region: Type Aliases

/// Type alias for a 2D matrix (Tensor with MAX_RANK=2).
pub type Matrix<T, A = Global> = Tensor<T, A, 2>;

/// Type alias for an immutable 2D matrix view.
pub type MatrixView<'a, T> = TensorView<'a, T, 2>;

/// Type alias for a mutable 2D matrix view.
pub type MatrixViewMut<'a, T> = TensorViewMut<'a, T, 2>;

// endregion: Type Aliases

// region: MatrixMultiplier

/// Pre-packed B matrix for efficient repeated GEMM operations.
///
/// Uses raw memory allocation (no std::Vec) for maximum control.
///
/// When multiplying A × Bᵀ multiple times with the same B matrix,
/// packing B once and reusing it is much faster than packing each time.
///
/// # Usage
///
/// For C = A × Bᵀ where B is (n × k):
/// ```rust,ignore
/// let packed_b = MatrixMultiplier::try_pack(&b_array).unwrap();
/// let c = a_array.matmul(&packed_b);
/// ```
///
/// For C = A × B where B is (k × n) (standard GEMM layout):
/// ```rust,ignore
/// let packed_b = MatrixMultiplier::try_pack_transposed(&b_array).unwrap();
/// let c = a_array.matmul(&packed_b);
/// ```
pub struct MatrixMultiplier<T: Dots, A: Allocator = Global> {
    /// Raw pointer to packed data buffer.
    data: NonNull<u8>,
    /// Size of the packed buffer in bytes.
    size: usize,
    /// Output columns (B rows).
    n: usize,
    /// Inner dimension.
    k: usize,
    /// Allocator instance.
    alloc: A,
    _marker: PhantomData<T>,
}

// Safety: MatrixMultiplier owns its data and is just bytes
unsafe impl<T: Dots + Send, A: Allocator + Send> Send for MatrixMultiplier<T, A> {}
unsafe impl<T: Dots + Sync, A: Allocator + Sync> Sync for MatrixMultiplier<T, A> {}

impl<T: Dots, A: Allocator> Drop for MatrixMultiplier<T, A> {
    fn drop(&mut self) {
        if self.size > 0 {
            unsafe {
                let layout =
                    alloc::alloc::Layout::from_size_align_unchecked(self.size, SIMD_ALIGNMENT);
                self.alloc.deallocate(self.data, layout);
            }
        }
    }
}

impl<T: Dots, A: Allocator + Clone> Clone for MatrixMultiplier<T, A> {
    fn clone(&self) -> Self {
        if self.size == 0 {
            return Self {
                data: NonNull::dangling(),
                size: 0,
                n: self.n,
                k: self.k,
                alloc: self.alloc.clone(),
                _marker: PhantomData,
            };
        }

        let layout = alloc::alloc::Layout::from_size_align(self.size, SIMD_ALIGNMENT)
            .expect("invalid layout");
        let ptr = self
            .alloc
            .allocate(layout)
            .expect("clone allocation failed");
        unsafe {
            core::ptr::copy_nonoverlapping(self.data.as_ptr(), ptr.as_ptr(), self.size);
        }
        Self {
            data: ptr,
            size: self.size,
            n: self.n,
            k: self.k,
            alloc: self.alloc.clone(),
            _marker: PhantomData,
        }
    }
}

// Generic allocator-aware methods
impl<T: Dots, A: Allocator> MatrixMultiplier<T, A> {
    /// Pack B matrix where B is (n × k) row-major using a custom allocator.
    ///
    /// Result computes: C = A × Bᵀ
    ///
    /// Returns `Err` if:
    /// - b is not 2D
    /// - allocation fails
    pub fn try_pack_in<BA: Allocator, const MAX_RANK: usize>(
        b: &Tensor<T, BA, MAX_RANK>,
        alloc: A,
    ) -> Result<Self, TensorError> {
        if b.ndim() != 2 {
            return Err(TensorError::DimensionMismatch {
                expected: 2,
                got: b.ndim(),
            });
        }
        let (n, k) = (b.shape()[0], b.shape()[1]);
        let size = T::dots_packed_size(n, k);

        let data = if size == 0 {
            NonNull::dangling()
        } else {
            // Allocate with SIMD alignment
            let layout = alloc::alloc::Layout::from_size_align(size, SIMD_ALIGNMENT)
                .map_err(|_| TensorError::AllocationFailed)?;
            let ptr = alloc
                .allocate(layout)
                .ok_or(TensorError::AllocationFailed)?;
            // Zero the memory
            unsafe {
                core::ptr::write_bytes(ptr.as_ptr(), 0, size);
            }
            ptr
        };

        if size > 0 {
            unsafe {
                T::dots_pack(b.as_ptr(), n, k, b.stride(0), data.as_ptr());
            }
        }

        Ok(Self {
            data,
            size,
            n,
            k,
            alloc,
            _marker: PhantomData,
        })
    }

    /// Pack Bᵀ where B is (k × n) row-major (standard GEMM layout) using a custom allocator.
    ///
    /// Internally transposes then packs.
    /// Result computes: C = A × B
    ///
    /// Returns `Err` if:
    /// - b is not 2D
    /// - allocation fails
    pub fn try_pack_transposed_in<BA: Allocator, const MAX_RANK: usize>(
        b: &Tensor<T, BA, MAX_RANK>,
        alloc: A,
    ) -> Result<Self, TensorError> {
        if b.ndim() != 2 {
            return Err(TensorError::DimensionMismatch {
                expected: 2,
                got: b.ndim(),
            });
        }
        let (k, n) = (b.shape()[0], b.shape()[1]);
        let size = T::dots_packed_size(n, k);

        let data = if size == 0 {
            NonNull::dangling()
        } else {
            // Allocate with SIMD alignment
            let layout = alloc::alloc::Layout::from_size_align(size, SIMD_ALIGNMENT)
                .map_err(|_| TensorError::AllocationFailed)?;
            let ptr = alloc
                .allocate(layout)
                .ok_or(TensorError::AllocationFailed)?;
            // Zero the memory
            unsafe {
                core::ptr::write_bytes(ptr.as_ptr(), 0, size);
            }
            ptr
        };

        if size > 0 {
            // Pack with transposed view: column stride becomes row stride
            unsafe {
                T::dots_pack(b.as_ptr(), n, k, core::mem::size_of::<T>(), data.as_ptr());
            }
        }

        Ok(Self {
            data,
            size,
            n,
            k,
            alloc,
            _marker: PhantomData,
        })
    }

    /// Returns a reference to the allocator.
    pub fn allocator(&self) -> &A {
        &self.alloc
    }

    /// Returns dimensions (n, k) of the original B matrix.
    pub fn dims(&self) -> (usize, usize) {
        (self.n, self.k)
    }

    /// Returns the packed data buffer.
    pub fn as_bytes(&self) -> &[u8] {
        unsafe { core::slice::from_raw_parts(self.data.as_ptr(), self.size) }
    }

    /// Returns a pointer to the packed data.
    pub fn as_ptr(&self) -> *const u8 {
        self.data.as_ptr()
    }
}

// Convenience methods using Global allocator
impl<T: Dots> MatrixMultiplier<T, Global> {
    /// Pack B matrix where B is (n × k) row-major using the global allocator.
    ///
    /// Result computes: C = A × Bᵀ
    pub fn try_pack<BA: Allocator, const MAX_RANK: usize>(
        b: &Tensor<T, BA, MAX_RANK>,
    ) -> Result<Self, TensorError> {
        Self::try_pack_in(b, Global)
    }

    /// Pack Bᵀ where B is (k × n) row-major (standard GEMM layout) using the global allocator.
    ///
    /// Result computes: C = A × B
    pub fn try_pack_transposed<BA: Allocator, const MAX_RANK: usize>(
        b: &Tensor<T, BA, MAX_RANK>,
    ) -> Result<Self, TensorError> {
        Self::try_pack_transposed_in(b, Global)
    }

    /// Convenience constructor that panics on error.
    pub fn pack<BA: Allocator, const MAX_RANK: usize>(b: &Tensor<T, BA, MAX_RANK>) -> Self {
        Self::try_pack(b).expect("MatrixMultiplier::pack failed")
    }

    /// Convenience constructor that panics on error.
    pub fn pack_transposed<BA: Allocator, const MAX_RANK: usize>(
        b: &Tensor<T, BA, MAX_RANK>,
    ) -> Self {
        Self::try_pack_transposed(b).expect("MatrixMultiplier::pack_transposed failed")
    }
}

// endregion: MatrixMultiplier

// region: Tensor GEMM

impl<T: Dots, A: Allocator + Clone, const MAX_RANK: usize> Tensor<T, A, MAX_RANK>
where
    T::Accumulator: Clone + Default,
{
    /// Matrix multiply: C = self × packed_bᵀ
    ///
    /// self must be 2D (m × k) with contiguous rows.
    /// packed_b contains B (n × k) packed.
    /// Returns C (m × n) using the same allocator as self.
    ///
    /// Returns `Err` if:
    /// - self is not 2D
    /// - self has non-contiguous rows
    /// - inner dimensions don't match
    /// - output allocation fails
    pub fn try_matmul<BA: Allocator>(
        &self,
        packed_b: &MatrixMultiplier<T, BA>,
    ) -> Result<Tensor<T::Accumulator, A, MAX_RANK>, TensorError> {
        if self.ndim() != 2 {
            return Err(TensorError::DimensionMismatch {
                expected: 2,
                got: self.ndim(),
            });
        }
        if !self.has_contiguous_rows() {
            return Err(TensorError::NonContiguousRows);
        }
        let (m, k) = (self.shape()[0], self.shape()[1]);
        let (n, packed_k) = packed_b.dims();
        if k != packed_k {
            return Err(TensorError::ShapeMismatch {
                expected: ShapeDescriptor::from_slice(&[m, packed_k]),
                got: ShapeDescriptor::from_slice(&[m, k]),
            });
        }

        let mut c = Tensor::try_new_in(&[m, n], T::Accumulator::default(), self.alloc.clone())?;
        unsafe {
            T::dots(
                self.as_ptr(),
                packed_b.as_ptr(),
                c.as_mut_ptr(),
                m,
                n,
                k,
                self.stride(0),
                c.stride(0),
            );
        }
        Ok(c)
    }

    /// Convenience method that panics on error.
    pub fn matmul<BA: Allocator>(
        &self,
        packed_b: &MatrixMultiplier<T, BA>,
    ) -> Tensor<T::Accumulator, A, MAX_RANK> {
        self.try_matmul(packed_b).expect("matmul failed")
    }
}

impl<T: Dots, A: Allocator, const MAX_RANK: usize> Tensor<T, A, MAX_RANK> {
    /// Matrix multiply into existing output (avoids allocation).
    pub fn try_matmul_into<BA: Allocator, CA: Allocator, const CA_MAX_RANK: usize>(
        &self,
        packed_b: &MatrixMultiplier<T, BA>,
        c: &mut Tensor<T::Accumulator, CA, CA_MAX_RANK>,
    ) -> Result<(), TensorError> {
        if self.ndim() != 2 {
            return Err(TensorError::DimensionMismatch {
                expected: 2,
                got: self.ndim(),
            });
        }
        if !self.has_contiguous_rows() {
            return Err(TensorError::NonContiguousRows);
        }
        let (m, k) = (self.shape()[0], self.shape()[1]);
        let (n, packed_k) = packed_b.dims();
        if k != packed_k {
            return Err(TensorError::ShapeMismatch {
                expected: ShapeDescriptor::from_slice(&[m, packed_k]),
                got: ShapeDescriptor::from_slice(&[m, k]),
            });
        }
        if c.shape() != &[m, n] {
            return Err(TensorError::ShapeMismatch {
                expected: ShapeDescriptor::from_slice(&[m, n]),
                got: ShapeDescriptor::from_slice(c.shape()),
            });
        }
        if !c.has_contiguous_rows() {
            return Err(TensorError::NonContiguousRows);
        }

        unsafe {
            T::dots(
                self.as_ptr(),
                packed_b.as_ptr(),
                c.as_mut_ptr(),
                m,
                n,
                k,
                self.stride(0),
                c.stride(0),
            );
        }
        Ok(())
    }
}

// Parallel matmul implementations, if Fork Union is available
#[cfg(feature = "parallel")]
impl<T: Dots + Clone + Send + Sync, A: Allocator + Clone, const MAX_RANK: usize>
    Tensor<T, A, MAX_RANK>
where
    T::Accumulator: Clone + Default + Send + Sync,
{
    /// Parallel matrix multiply into pre-allocated output.
    ///
    /// Distributes rows of A across threads; each computes its portion of C.
    /// This is a non-allocating interface - you provide the output tensor.
    ///
    /// # Arguments
    /// * `packed_b` - Pre-packed B matrix from `MatrixMultiplier::try_pack[_transposed]`
    /// * `c` - Pre-allocated output tensor (m × n)
    /// * `pool` - Pre-constructed thread pool
    ///
    /// # Example
    /// ```ignore
    /// use numkong::{Tensor, MatrixMultiplier};
    /// use fork_union::ThreadPool;
    ///
    /// let mut pool = ThreadPool::try_spawn(4).unwrap();
    /// let a = Tensor::<f32>::try_new(&[1024, 512], 1.0).unwrap();
    /// let b = Tensor::<f32>::try_new(&[256, 512], 1.0).unwrap();
    /// let packed_b = MatrixMultiplier::try_pack(&b).unwrap();
    /// let mut c = Tensor::<f32>::try_new(&[1024, 256], 0.0).unwrap();
    /// a.try_matmul_parallel_into(&packed_b, &mut c, &mut pool).unwrap();
    /// ```
    pub fn try_matmul_parallel_into<BA: Allocator, CA: Allocator, const CA_MAX_RANK: usize>(
        &self,
        packed_b: &MatrixMultiplier<T, BA>,
        c: &mut Tensor<T::Accumulator, CA, CA_MAX_RANK>,
        pool: &mut fork_union::ThreadPool,
    ) -> Result<(), TensorError> {
        if self.ndim() != 2 {
            return Err(TensorError::DimensionMismatch {
                expected: 2,
                got: self.ndim(),
            });
        }
        if !self.has_contiguous_rows() {
            return Err(TensorError::NonContiguousRows);
        }
        let (m, k) = (self.shape()[0], self.shape()[1]);
        let (n, packed_k) = packed_b.dims();
        if k != packed_k {
            return Err(TensorError::ShapeMismatch {
                expected: ShapeDescriptor::from_slice(&[m, packed_k]),
                got: ShapeDescriptor::from_slice(&[m, k]),
            });
        }
        if c.shape() != &[m, n] {
            return Err(TensorError::ShapeMismatch {
                expected: ShapeDescriptor::from_slice(&[m, n]),
                got: ShapeDescriptor::from_slice(c.shape()),
            });
        }
        if !c.has_contiguous_rows() {
            return Err(TensorError::NonContiguousRows);
        }

        let a_ptr = fork_union::SyncConstPtr::new(self.as_ptr());
        let c_ptr = fork_union::SyncMutPtr::new(c.as_mut_ptr());
        let packed_ptr = fork_union::SyncConstPtr::new(packed_b.as_ptr());
        let a_stride = self.stride(0);
        let c_stride = c.stride(0);

        // Get actual thread count from pool
        let num_threads = pool.threads().max(1);
        let rows_per_thread = (m + num_threads - 1) / num_threads;

        // Distribute rows across threads using fork_union
        // Safety: Each thread writes to disjoint rows of C, so no data races.
        pool.for_threads(move |thread_idx, _colocation_idx| {
            // Configure each worker thread for optimal SIMD (including AMX)
            // This is idempotent and safe to call multiple times
            crate::capabilities::configure_thread();

            let row_start = thread_idx * rows_per_thread;
            let row_end = (row_start + rows_per_thread).min(m);

            if row_start < m {
                unsafe {
                    T::dots(
                        a_ptr.as_ptr().add(row_start * k),
                        packed_ptr.as_ptr(),
                        c_ptr.as_ptr().add(row_start * n),
                        row_end - row_start,
                        n,
                        k,
                        a_stride,
                        c_stride,
                    );
                }
            }
        })
        .join();

        Ok(())
    }

    /// Parallel matrix multiply with allocation.
    ///
    /// Convenience wrapper that allocates the output tensor.
    /// Prefer `try_matmul_parallel_into` for performance-critical code.
    pub fn try_matmul_parallel<BA: Allocator>(
        &self,
        packed_b: &MatrixMultiplier<T, BA>,
        pool: &mut fork_union::ThreadPool,
    ) -> Result<Tensor<T::Accumulator, Global, MAX_RANK>, TensorError> {
        let m = self.shape()[0];
        let (n, _) = packed_b.dims();
        let mut c = Tensor::<T::Accumulator, Global, MAX_RANK>::try_new(
            &[m, n],
            T::Accumulator::default(),
        )?;
        self.try_matmul_parallel_into(packed_b, &mut c, pool)?;
        Ok(c)
    }

    /// Convenience method that panics on error.
    pub fn matmul_parallel<BA: Allocator>(
        &self,
        packed_b: &MatrixMultiplier<T, BA>,
        pool: &mut fork_union::ThreadPool,
    ) -> Tensor<T::Accumulator, Global, MAX_RANK> {
        self.try_matmul_parallel(packed_b, pool)
            .expect("parallel matmul failed")
    }
}

// endregion: Tensor GEMM

// region: Tensor Elementwise Operations

impl<T: Clone + Scale, const MAX_RANK: usize> Tensor<T, Global, MAX_RANK> {
    /// Apply element-wise scale: result[i] = alpha * self[i] + beta
    ///
    /// Returns a new array with the scaled values.
    pub fn scale(
        &self,
        alpha: T::Scalar,
        beta: T::Scalar,
    ) -> Result<Tensor<T, Global, MAX_RANK>, TensorError> {
        let mut result = Tensor::try_new(self.shape(), unsafe { (*self.as_ptr()).clone() })?;
        T::scale(self.as_slice(), alpha, beta, result.as_mut_slice());
        Ok(result)
    }

    /// Apply element-wise scale in-place: self[i] = alpha * self[i] + beta
    pub fn scale_inplace(&mut self, alpha: T::Scalar, beta: T::Scalar) {
        // Need a temporary for in-place operation since input and output overlap
        let ptr = self.as_ptr();
        let len = self.len;
        unsafe {
            let slice = core::slice::from_raw_parts(ptr, len);
            T::scale(slice, alpha, beta, self.as_mut_slice());
        }
    }
}

impl<T: Clone + Sum, const MAX_RANK: usize> Tensor<T, Global, MAX_RANK> {
    /// Element-wise sum: result[i] = self[i] + other[i]
    ///
    /// Returns a new array with the summed values.
    pub fn add<const OTHER_MAX_RANK: usize>(
        &self,
        other: &Tensor<T, Global, OTHER_MAX_RANK>,
    ) -> Result<Tensor<T, Global, MAX_RANK>, TensorError> {
        if self.shape() != other.shape() {
            return Err(TensorError::ShapeMismatch {
                expected: ShapeDescriptor::from_slice(self.shape()),
                got: ShapeDescriptor::from_slice(other.shape()),
            });
        }
        let mut result = Tensor::try_new(self.shape(), unsafe { (*self.as_ptr()).clone() })?;
        T::sum(self.as_slice(), other.as_slice(), result.as_mut_slice());
        Ok(result)
    }

    /// Element-wise sum in-place: self[i] = self[i] + other[i]
    pub fn add_inplace<const OTHER_MAX_RANK: usize>(
        &mut self,
        other: &Tensor<T, Global, OTHER_MAX_RANK>,
    ) -> Result<(), TensorError> {
        if self.shape() != other.shape() {
            return Err(TensorError::ShapeMismatch {
                expected: ShapeDescriptor::from_slice(self.shape()),
                got: ShapeDescriptor::from_slice(other.shape()),
            });
        }
        let ptr = self.as_ptr();
        let len = self.len;
        unsafe {
            let slice = core::slice::from_raw_parts(ptr, len);
            T::sum(slice, other.as_slice(), self.as_mut_slice());
        }
        Ok(())
    }
}

impl<T: Clone + WSum, const MAX_RANK: usize> Tensor<T, Global, MAX_RANK> {
    /// Weighted sum: result[i] = alpha * self[i] + beta * other[i]
    ///
    /// Returns a new array with the weighted sum.
    pub fn wsum<const OTHER_MAX_RANK: usize>(
        &self,
        other: &Tensor<T, Global, OTHER_MAX_RANK>,
        alpha: T::Scalar,
        beta: T::Scalar,
    ) -> Result<Tensor<T, Global, MAX_RANK>, TensorError> {
        if self.shape() != other.shape() {
            return Err(TensorError::ShapeMismatch {
                expected: ShapeDescriptor::from_slice(self.shape()),
                got: ShapeDescriptor::from_slice(other.shape()),
            });
        }
        let mut result = Tensor::try_new(self.shape(), unsafe { (*self.as_ptr()).clone() })?;
        T::wsum(
            self.as_slice(),
            other.as_slice(),
            alpha,
            beta,
            result.as_mut_slice(),
        );
        Ok(result)
    }
}

impl<T: Clone + FMA, const MAX_RANK: usize> Tensor<T, Global, MAX_RANK> {
    /// Fused multiply-add: result[i] = alpha * self[i] * b[i] + beta * c[i]
    ///
    /// Returns a new array with the FMA result.
    pub fn fma<const B_MAX_RANK: usize, const C_MAX_RANK: usize>(
        &self,
        b: &Tensor<T, Global, B_MAX_RANK>,
        c: &Tensor<T, Global, C_MAX_RANK>,
        alpha: T::Scalar,
        beta: T::Scalar,
    ) -> Result<Tensor<T, Global, MAX_RANK>, TensorError> {
        if self.shape() != b.shape() || self.shape() != c.shape() {
            return Err(TensorError::ShapeMismatch {
                expected: ShapeDescriptor::from_slice(self.shape()),
                got: ShapeDescriptor::from_slice(b.shape()),
            });
        }
        let mut result = Tensor::try_new(self.shape(), unsafe { (*self.as_ptr()).clone() })?;
        T::fma(
            self.as_slice(),
            b.as_slice(),
            c.as_slice(),
            alpha,
            beta,
            result.as_mut_slice(),
        );
        Ok(result)
    }
}

// endregion: Tensor Elementwise Operations

// region: Tensor Trigonometry

impl<T: Clone + Sin, const MAX_RANK: usize> Tensor<T, Global, MAX_RANK> {
    /// Element-wise sine: result[i] = sin(self[i])
    ///
    /// Input values are in radians.
    pub fn sin(&self) -> Result<Tensor<T, Global, MAX_RANK>, TensorError> {
        let mut result = Tensor::try_new(self.shape(), unsafe { (*self.as_ptr()).clone() })?;
        T::sin(self.as_slice(), result.as_mut_slice());
        Ok(result)
    }

    /// Element-wise sine in-place: self[i] = sin(self[i])
    pub fn sin_inplace(&mut self) {
        let ptr = self.as_ptr();
        let len = self.len;
        unsafe {
            let slice = core::slice::from_raw_parts(ptr, len);
            T::sin(slice, self.as_mut_slice());
        }
    }
}

impl<T: Clone + Cos, const MAX_RANK: usize> Tensor<T, Global, MAX_RANK> {
    /// Element-wise cosine: result[i] = cos(self[i])
    ///
    /// Input values are in radians.
    pub fn cos(&self) -> Result<Tensor<T, Global, MAX_RANK>, TensorError> {
        let mut result = Tensor::try_new(self.shape(), unsafe { (*self.as_ptr()).clone() })?;
        T::cos(self.as_slice(), result.as_mut_slice());
        Ok(result)
    }

    /// Element-wise cosine in-place: self[i] = cos(self[i])
    pub fn cos_inplace(&mut self) {
        let ptr = self.as_ptr();
        let len = self.len;
        unsafe {
            let slice = core::slice::from_raw_parts(ptr, len);
            T::cos(slice, self.as_mut_slice());
        }
    }
}

impl<T: Clone + ATan, const MAX_RANK: usize> Tensor<T, Global, MAX_RANK> {
    /// Element-wise arctangent: result[i] = atan(self[i])
    ///
    /// Output values are in radians in the range (-π/2, π/2).
    pub fn atan(&self) -> Result<Tensor<T, Global, MAX_RANK>, TensorError> {
        let mut result = Tensor::try_new(self.shape(), unsafe { (*self.as_ptr()).clone() })?;
        T::atan(self.as_slice(), result.as_mut_slice());
        Ok(result)
    }

    /// Element-wise arctangent in-place: self[i] = atan(self[i])
    pub fn atan_inplace(&mut self) {
        let ptr = self.as_ptr();
        let len = self.len;
        unsafe {
            let slice = core::slice::from_raw_parts(ptr, len);
            T::atan(slice, self.as_mut_slice());
        }
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

impl<const MAX_RANK: usize> Tensor<f32, Global, MAX_RANK> {
    /// Sum all elements of the array.
    pub fn sum(&self) -> f32 {
        let ones: Tensor<f32, Global, MAX_RANK> =
            Tensor::try_new(self.shape(), 1.0f32).expect("allocation failed");
        <f32 as Dot>::dot(self.as_slice(), ones.as_slice()).unwrap_or(0.0)
    }
}

impl<const MAX_RANK: usize> Tensor<f64, Global, MAX_RANK> {
    /// Sum all elements of the array.
    pub fn sum(&self) -> f64 {
        let ones: Tensor<f64, Global, MAX_RANK> =
            Tensor::try_new(self.shape(), 1.0f64).expect("allocation failed");
        <f64 as Dot>::dot(self.as_slice(), ones.as_slice()).unwrap_or(0.0)
    }
}

// endregion: Tensor Reductions

// region: Tests

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_creation() {
        let arr = Tensor::<f32>::try_new(&[3, 4], 1.0f32).unwrap();
        assert_eq!(arr.shape(), &[3, 4]);
        assert_eq!(arr.ndim(), 2);
        assert_eq!(arr.len(), 12);
        assert!(!arr.is_empty());
    }

    #[test]
    fn tensor_from_slice() {
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let arr = Tensor::<f32>::try_from_slice(&data, &[3, 4]).unwrap();
        assert_eq!(arr.shape(), &[3, 4]);
        assert_eq!(arr.as_slice(), &data[..]);
    }

    #[test]
    fn tensor_row_access() {
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let arr = Tensor::<f32>::try_from_slice(&data, &[3, 4]).unwrap();
        assert_eq!(arr.row(0), Some(&[0.0, 1.0, 2.0, 3.0][..]));
        assert_eq!(arr.row(1), Some(&[4.0, 5.0, 6.0, 7.0][..]));
        assert_eq!(arr.row(2), Some(&[8.0, 9.0, 10.0, 11.0][..]));
        assert_eq!(arr.row(3), None);
    }

    #[test]
    fn tensor_slicing() {
        let arr = Tensor::<f32>::try_new(&[4, 5], 1.0f32).unwrap();

        // Full slice
        let view = arr
            .slice(&[SliceRange::full(), SliceRange::full()])
            .unwrap();
        assert_eq!(view.shape(), &[4, 5]);

        // Range slice
        let view = arr
            .slice(&[SliceRange::range(1, 3), SliceRange::full()])
            .unwrap();
        assert_eq!(view.shape(), &[2, 5]);

        // Single index (reduces dimension)
        let view = arr
            .slice(&[SliceRange::index(0), SliceRange::full()])
            .unwrap();
        assert_eq!(view.shape(), &[5]);
        assert_eq!(view.ndim(), 1);
    }

    #[test]
    fn tensor_transpose() {
        let arr = Tensor::<f32>::try_new(&[3, 4], 1.0f32).unwrap();
        let transposed = arr.t().unwrap();
        assert_eq!(transposed.shape(), &[4, 3]);
    }

    #[test]
    fn tensor_reshape() {
        let arr = Tensor::<f32>::try_new(&[3, 4], 1.0f32).unwrap();
        let reshaped = arr.reshape(&[2, 6]).unwrap();
        assert_eq!(reshaped.shape(), &[2, 6]);
        assert_eq!(reshaped.len(), 12);
    }

    #[test]
    fn tensor_clone() {
        let arr = Tensor::<f32>::try_new(&[3, 4], 2.5f32).unwrap();
        let cloned = arr.clone();
        assert_eq!(cloned.shape(), arr.shape());
        assert_eq!(cloned.as_slice(), arr.as_slice());
    }

    #[test]
    fn tensor_contiguous_check() {
        let arr = Tensor::<f32>::try_new(&[3, 4], 1.0f32).unwrap();
        let view = arr.view();
        assert!(view.is_contiguous());
        assert!(arr.has_contiguous_rows());
    }

    #[test]
    fn matrix_alias() {
        let mat: Matrix<f32> = Matrix::try_new(&[3, 4], 1.0f32).unwrap();
        assert_eq!(mat.shape(), &[3, 4]);
    }

    #[test]
    fn tensor_sum_f32() {
        let arr = Tensor::<f32>::try_new(&[100], 1.0f32).unwrap();
        let sum = arr.sum();
        assert!((sum - 100.0).abs() < 0.1);
    }

    #[test]
    fn tensor_sum_f64() {
        let arr = Tensor::<f64>::try_new(&[100], 1.0f64).unwrap();
        let sum = arr.sum();
        assert!((sum - 100.0).abs() < 0.1);
    }

    #[test]
    fn tensor_error_display() {
        let err = TensorError::AllocationFailed;
        assert_eq!(format!("{}", err), "memory allocation failed");

        let err = TensorError::TooManyRanks { got: 10 };
        assert_eq!(format!("{}", err), "too many ranks: 10");
    }

    #[test]
    fn matmul_f32_pack() {
        // A[4×8] × B[16×8]ᵀ = C[4×16]
        let a = Tensor::<f32>::try_new(&[4, 8], 1.0f32).unwrap();
        let b = Tensor::<f32>::try_new(&[16, 8], 1.0f32).unwrap();

        let packed_b = MatrixMultiplier::try_pack(&b).unwrap();
        let c = a.matmul(&packed_b);

        assert_eq!(c.shape(), &[4, 16]);
        // Each element = dot(row_a, row_b) = sum(1.0 * 1.0) * 8 = 8.0
        assert!((c.as_slice()[0] - 8.0).abs() < 1e-5);
    }

    #[test]
    fn matmul_f32_pack_transposed() {
        // A[4×8], B[8×16] (standard k×n layout) → C[4×16]
        let a = Tensor::<f32>::try_new(&[4, 8], 1.0f32).unwrap();
        let b_transposed = Tensor::<f32>::try_new(&[8, 16], 1.0f32).unwrap(); // k × n

        let packed_b = MatrixMultiplier::try_pack_transposed(&b_transposed).unwrap();
        let c = a.matmul(&packed_b);

        assert_eq!(c.shape(), &[4, 16]);
        assert!((c.as_slice()[0] - 8.0).abs() < 1e-5);
    }

    #[test]
    fn matmul_f32_into() {
        let a = Tensor::<f32>::try_new(&[4, 8], 1.0f32).unwrap();
        let b = Tensor::<f32>::try_new(&[16, 8], 1.0f32).unwrap();
        let mut c = Tensor::<f32>::try_new(&[4, 16], 0.0f32).unwrap();

        let packed_b = MatrixMultiplier::try_pack(&b).unwrap();
        a.try_matmul_into(&packed_b, &mut c).unwrap();

        assert!((c.as_slice()[0] - 8.0).abs() < 1e-5);
    }

    #[test]
    fn matmul_f64_pack() {
        let a = Tensor::<f64>::try_new(&[4, 8], 1.0f64).unwrap();
        let b = Tensor::<f64>::try_new(&[16, 8], 1.0f64).unwrap();

        let packed_b = MatrixMultiplier::try_pack(&b).unwrap();
        let c = a.matmul(&packed_b);

        assert_eq!(c.shape(), &[4, 16]);
        assert!((c.as_slice()[0] - 8.0).abs() < 1e-10);
    }

    #[test]
    fn matmul_bf16_pack() {
        let a = Tensor::<bf16>::try_new(&[4, 8], bf16::from_f32(1.0)).unwrap();
        let b = Tensor::<bf16>::try_new(&[16, 8], bf16::from_f32(1.0)).unwrap();

        let packed_b = MatrixMultiplier::try_pack(&b).unwrap();
        let c = a.matmul(&packed_b);

        assert_eq!(c.shape(), &[4, 16]);
        assert!((c.as_slice()[0] - 8.0).abs() < 0.1);
    }

    #[test]
    fn matmul_f16_pack() {
        let a = Tensor::<f16>::try_new(&[4, 8], f16::from_f32(1.0)).unwrap();
        let b = Tensor::<f16>::try_new(&[16, 8], f16::from_f32(1.0)).unwrap();

        let packed_b = MatrixMultiplier::try_pack(&b).unwrap();
        let c = a.matmul(&packed_b);

        assert_eq!(c.shape(), &[4, 16]);
        assert!((c.as_slice()[0] - 8.0).abs() < 0.1);
    }

    #[test]
    fn matmul_i8_pack() {
        let a = Tensor::<i8>::try_new(&[4, 8], 1i8).unwrap();
        let b = Tensor::<i8>::try_new(&[16, 8], 1i8).unwrap();

        let packed_b = MatrixMultiplier::try_pack(&b).unwrap();
        let c = a.matmul(&packed_b);

        assert_eq!(c.shape(), &[4, 16]);
        assert_eq!(c.as_slice()[0], 8);
    }

    #[test]
    fn matmul_u8_pack() {
        let a = Tensor::<u8>::try_new(&[4, 8], 1u8).unwrap();
        let b = Tensor::<u8>::try_new(&[16, 8], 1u8).unwrap();

        let packed_b = MatrixMultiplier::try_pack(&b).unwrap();
        let c = a.matmul(&packed_b);

        assert_eq!(c.shape(), &[4, 16]);
        assert_eq!(c.as_slice()[0], 8);
    }

    #[test]
    fn matmul_e4m3_pack() {
        let a = Tensor::<e4m3>::try_new(&[4, 8], e4m3::ONE).unwrap();
        let b = Tensor::<e4m3>::try_new(&[16, 8], e4m3::ONE).unwrap();

        let packed_b = MatrixMultiplier::try_pack(&b).unwrap();
        let c = a.matmul(&packed_b);

        assert_eq!(c.shape(), &[4, 16]);
        assert!((c.as_slice()[0] - 8.0).abs() < 0.5);
    }

    #[test]
    fn matmul_e4m3_pack_transposed() {
        let a = Tensor::<e4m3>::try_new(&[4, 8], e4m3::ONE).unwrap();
        let b_t = Tensor::<e4m3>::try_new(&[8, 16], e4m3::ONE).unwrap();

        let packed_b = MatrixMultiplier::try_pack_transposed(&b_t).unwrap();
        let c = a.matmul(&packed_b);

        assert_eq!(c.shape(), &[4, 16]);
        assert!((c.as_slice()[0] - 8.0).abs() < 0.5);
    }

    #[test]
    fn matmul_e5m2_pack() {
        let a = Tensor::<e5m2>::try_new(&[4, 8], e5m2::ONE).unwrap();
        let b = Tensor::<e5m2>::try_new(&[16, 8], e5m2::ONE).unwrap();

        let packed_b = MatrixMultiplier::try_pack(&b).unwrap();
        let c = a.matmul(&packed_b);

        assert_eq!(c.shape(), &[4, 16]);
        assert!((c.as_slice()[0] - 8.0).abs() < 0.5);
    }

    #[test]
    fn matmul_e5m2_pack_transposed() {
        let a = Tensor::<e5m2>::try_new(&[4, 8], e5m2::ONE).unwrap();
        let b_t = Tensor::<e5m2>::try_new(&[8, 16], e5m2::ONE).unwrap();

        let packed_b = MatrixMultiplier::try_pack_transposed(&b_t).unwrap();
        let c = a.matmul(&packed_b);

        assert_eq!(c.shape(), &[4, 16]);
        assert!((c.as_slice()[0] - 8.0).abs() < 0.5);
    }

    #[test]
    fn matmul_f32_single_row() {
        let a = Tensor::<f32>::try_new(&[1, 8], 1.0f32).unwrap();
        let b = Tensor::<f32>::try_new(&[4, 8], 1.0f32).unwrap();

        let packed_b = MatrixMultiplier::try_pack(&b).unwrap();
        let c = a.matmul(&packed_b);

        assert_eq!(c.shape(), &[1, 4]);
        assert!((c.as_slice()[0] - 8.0).abs() < 1e-5);
    }

    #[test]
    fn matmul_f32_single_col() {
        let a = Tensor::<f32>::try_new(&[4, 8], 1.0f32).unwrap();
        let b = Tensor::<f32>::try_new(&[1, 8], 1.0f32).unwrap();

        let packed_b = MatrixMultiplier::try_pack(&b).unwrap();
        let c = a.matmul(&packed_b);

        assert_eq!(c.shape(), &[4, 1]);
        assert!((c.as_slice()[0] - 8.0).abs() < 1e-5);
    }

    #[test]
    #[cfg(feature = "parallel")]
    fn matmul_f32_parallel() {
        let mut pool = fork_union::ThreadPool::try_spawn(4).unwrap();
        let a = Tensor::<f32>::try_new(&[64, 128], 1.0f32).unwrap();
        let b = Tensor::<f32>::try_new(&[32, 128], 1.0f32).unwrap();

        let packed_b = MatrixMultiplier::try_pack(&b).unwrap();
        let c = a.matmul_parallel(&packed_b, &mut pool);

        assert_eq!(c.shape(), &[64, 32]);
        // Each element = sum of 128 products of 1.0 * 1.0 = 128.0
        assert!((c.as_slice()[0] - 128.0).abs() < 1e-5);
    }

    #[test]
    #[cfg(feature = "parallel")]
    fn matmul_f32_parallel_into() {
        let mut pool = fork_union::ThreadPool::try_spawn(4).unwrap();
        let a = Tensor::<f32>::try_new(&[64, 128], 1.0f32).unwrap();
        let b = Tensor::<f32>::try_new(&[32, 128], 1.0f32).unwrap();
        let mut c = Tensor::<f32>::try_new(&[64, 32], 0.0f32).unwrap();

        let packed_b = MatrixMultiplier::try_pack(&b).unwrap();
        a.try_matmul_parallel_into(&packed_b, &mut c, &mut pool)
            .unwrap();

        // Verify against serial
        let c_serial = a.matmul(&packed_b);
        for (p, s) in c.as_slice().iter().zip(c_serial.as_slice().iter()) {
            assert!((p - s).abs() < 1e-5);
        }
    }

    #[test]
    #[cfg(feature = "parallel")]
    fn matmul_bf16_parallel() {
        let mut pool = fork_union::ThreadPool::try_spawn(4).unwrap();
        let a = Tensor::<bf16>::try_new(&[32, 64], bf16::from_f32(1.0)).unwrap();
        let b = Tensor::<bf16>::try_new(&[16, 64], bf16::from_f32(1.0)).unwrap();

        let packed_b = MatrixMultiplier::try_pack(&b).unwrap();
        let c = a.matmul_parallel(&packed_b, &mut pool);

        assert_eq!(c.shape(), &[32, 16]);
        // Each element = sum of 64 products = 64.0
        assert!((c.as_slice()[0] - 64.0).abs() < 1.0);
    }
}

// endregion: Tests

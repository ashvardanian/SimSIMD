//! Owning and non-owning vector types with signed indexing and sub-byte support.
//!
//! This module provides:
//!
//! - [`Vector`]: Owning, non-resizable, SIMD-aligned vector
//! - [`VectorView`]: Immutable, strided, non-owning view
//! - [`VectorSpan`]: Mutable, strided, non-owning view
//! - [`VecIndex`]: Signed indexing trait (negative indices wrap from end)
//!
//! All types use [`FloatLike`] as their element bound, with sub-byte types
//! (i4x2, u4x2, u1x8) supported via `try_get`/`try_set` and iterators.

extern crate alloc;

use core::marker::PhantomData;
use core::ptr::NonNull;

use crate::scalar::{FloatConvertible, FloatLike};
use crate::tensor::{Allocator, Global, TensorError, SIMD_ALIGNMENT};

// region: VecIndex — Signed Indexing

mod private {
    pub trait Sealed {}
}

/// Trait for vector index types. Supports signed integers (negative = from end).
pub trait VecIndex: private::Sealed + Copy {
    /// Resolve this index to a `usize` offset, or `None` if out of bounds.
    fn resolve(self, len: usize) -> Option<usize>;
}

macro_rules! impl_vec_index_unsigned {
    ($($t:ty),*) => {$(
        impl private::Sealed for $t {}
        impl VecIndex for $t {
            #[inline]
            fn resolve(self, len: usize) -> Option<usize> {
                let idx = self as usize;
                if idx < len { Some(idx) } else { None }
            }
        }
    )*};
}

macro_rules! impl_vec_index_signed {
    ($($t:ty),*) => {$(
        impl private::Sealed for $t {}
        impl VecIndex for $t {
            #[inline]
            fn resolve(self, len: usize) -> Option<usize> {
                let idx = if self >= 0 {
                    self as usize
                } else {
                    let neg = (-(self as isize)) as usize;
                    if neg > len { return None; }
                    len - neg
                };
                if idx < len { Some(idx) } else { None }
            }
        }
    )*};
}

impl_vec_index_unsigned!(usize, u8, u16, u32, u64);
impl_vec_index_signed!(isize, i8, i16, i32, i64);

// endregion: VecIndex

// region: Sub-byte Proxy Types

/// Immutable reference to a nibble (4-bit value) within a packed byte.
///
/// Provides unsigned (`get_unsigned`) and sign-extended (`get_signed`) access
/// to either the low or high nibble of the referenced byte.
pub struct NibbleRef<'a> {
    byte: *const u8,
    high: bool,
    _marker: PhantomData<&'a u8>,
}

impl<'a> NibbleRef<'a> {
    /// Read the nibble value as u8 (0..16).
    #[inline]
    pub fn get_unsigned(&self) -> u8 {
        // SAFETY: byte pointer is valid for the lifetime 'a
        let b = unsafe { *self.byte };
        if self.high {
            b >> 4
        } else {
            b & 0x0F
        }
    }

    /// Read the nibble value as i8 (-8..7), sign-extending bit 3.
    #[inline]
    pub fn get_signed(&self) -> i8 {
        let nibble = self.get_unsigned();
        if nibble & 0x08 != 0 {
            nibble as i8 | !0x0Fi8
        } else {
            nibble as i8
        }
    }
}

/// Mutable reference to a nibble (4-bit value) within a packed byte.
///
/// Provides read and write access to either the low or high nibble of the
/// referenced byte, preserving the other nibble on writes.
pub struct NibbleRefMut<'a> {
    byte: *mut u8,
    high: bool,
    _marker: PhantomData<&'a mut u8>,
}

impl<'a> NibbleRefMut<'a> {
    /// Read the nibble value as u8.
    #[inline]
    pub fn get_unsigned(&self) -> u8 {
        let b = unsafe { *self.byte };
        if self.high {
            b >> 4
        } else {
            b & 0x0F
        }
    }

    /// Read the nibble value as i8, sign-extending bit 3.
    #[inline]
    pub fn get_signed(&self) -> i8 {
        let nibble = self.get_unsigned();
        if nibble & 0x08 != 0 {
            nibble as i8 | !0x0Fi8
        } else {
            nibble as i8
        }
    }

    /// Set the nibble to an unsigned value (low 4 bits used).
    #[inline]
    pub fn set_unsigned(&self, val: u8) {
        // SAFETY: byte pointer is valid and mutable for the lifetime 'a
        unsafe {
            let b = *self.byte;
            if self.high {
                *self.byte = (b & 0x0F) | ((val & 0x0F) << 4);
            } else {
                *self.byte = (b & 0xF0) | (val & 0x0F);
            }
        }
    }

    /// Set the nibble to a signed value (low 4 bits used).
    #[inline]
    pub fn set_signed(&self, val: i8) {
        self.set_unsigned(val as u8);
    }
}

/// Immutable reference to a single bit within a packed byte.
///
/// The `mask` field selects which bit within the byte to read.
pub struct BitRef<'a> {
    byte: *const u8,
    mask: u8,
    _marker: PhantomData<&'a u8>,
}

impl<'a> BitRef<'a> {
    /// Read the bit as bool.
    #[inline]
    pub fn get(&self) -> bool {
        // SAFETY: byte pointer is valid for the lifetime 'a
        (unsafe { *self.byte } & self.mask) != 0
    }
}

/// Mutable reference to a single bit within a packed byte.
///
/// The `mask` field selects which bit within the byte to read or write.
/// Writes preserve all other bits in the byte.
pub struct BitRefMut<'a> {
    byte: *mut u8,
    mask: u8,
    _marker: PhantomData<&'a mut u8>,
}

impl<'a> BitRefMut<'a> {
    /// Read the bit as bool.
    #[inline]
    pub fn get(&self) -> bool {
        // SAFETY: byte pointer is valid for the lifetime 'a
        (unsafe { *self.byte } & self.mask) != 0
    }

    /// Set the bit.
    #[inline]
    pub fn set(&self, val: bool) {
        // SAFETY: byte pointer is valid and mutable for the lifetime 'a
        unsafe {
            if val {
                *self.byte |= self.mask;
            } else {
                *self.byte &= !self.mask;
            }
        }
    }
}

// endregion: Sub-byte Proxy Types

// region: Vector

/// Owning, non-resizable, SIMD-aligned vector.
///
/// Size is fixed at construction. Uses [`FloatLike`] for element types,
/// including sub-byte packed types via `T::dimensions_per_value()`.
///
/// For normal types (`dimensions_per_value() == 1`), supports `Index`/`IndexMut`.
/// For sub-byte types, use `try_get`/`try_set` or iterators.
pub struct Vector<T: FloatLike, A: Allocator = Global> {
    /// Pointer to the allocated buffer (typed as T for alignment).
    data: NonNull<T>,
    /// Number of logical dimensions.
    dims: usize,
    /// Number of storage values (dims / dimensions_per_value, rounded up).
    values: usize,
    /// Allocator instance.
    alloc: A,
}

unsafe impl<T: FloatLike + Send, A: Allocator + Send> Send for Vector<T, A> {}
unsafe impl<T: FloatLike + Sync, A: Allocator + Sync> Sync for Vector<T, A> {}

impl<T: FloatLike, A: Allocator> Drop for Vector<T, A> {
    fn drop(&mut self) {
        if self.values > 0 {
            let layout = alloc::alloc::Layout::from_size_align(
                self.values * core::mem::size_of::<T>(),
                SIMD_ALIGNMENT,
            )
            .unwrap();
            // SAFETY: data was allocated with this layout in try_zeroed_in,
            // and values > 0 guarantees the pointer is non-dangling.
            unsafe {
                self.alloc.deallocate(
                    NonNull::new_unchecked(self.data.as_ptr() as *mut u8),
                    layout,
                );
            }
        }
    }
}

/// Convert dimension count to value count for type T.
///
/// For sub-byte types where `dimensions_per_value() > 1`, this performs ceiling
/// division: `(dims + dpv - 1) / dpv`.
#[inline]
fn dims_to_values<T: FloatLike>(dims: usize) -> usize {
    let dpv = T::dimensions_per_value();
    (dims + dpv - 1) / dpv
}

impl<T: FloatLike, A: Allocator> Vector<T, A> {
    /// Try to create a zero-initialized vector with the given number of dimensions.
    pub fn try_zeroed_in(dims: usize, alloc: A) -> Result<Self, TensorError> {
        let values = dims_to_values::<T>(dims);
        if values == 0 {
            return Ok(Self {
                data: NonNull::dangling(),
                dims: 0,
                values: 0,
                alloc,
            });
        }
        let size = values * core::mem::size_of::<T>();
        let layout = alloc::alloc::Layout::from_size_align(size, SIMD_ALIGNMENT)
            .map_err(|_| TensorError::AllocationFailed)?;
        let ptr = alloc
            .allocate(layout)
            .ok_or(TensorError::AllocationFailed)?;
        unsafe { core::ptr::write_bytes(ptr.as_ptr(), 0, size) };
        Ok(Self {
            data: unsafe { NonNull::new_unchecked(ptr.as_ptr() as *mut T) },
            dims,
            values,
            alloc,
        })
    }

    /// Try to create a vector filled with `value`.
    pub fn try_filled_in(dims: usize, value: T, alloc: A) -> Result<Self, TensorError> {
        let v = Self::try_zeroed_in(dims, alloc)?;
        if v.values > 0 {
            let ptr = v.data.as_ptr();
            for i in 0..v.values {
                unsafe { ptr.add(i).write(value) };
            }
        }
        Ok(v)
    }

    /// Try to create a vector from a slice of scalars (f32 values).
    ///
    /// Each f32 value is converted through `DimScalar::from_f32()` before storage.
    pub fn try_from_scalars_in(scalars: &[f32], alloc: A) -> Result<Self, TensorError>
    where
        T: FloatConvertible,
    {
        let n = scalars.len();
        let mut v = Self::try_zeroed_in(n, alloc)?;
        for (i, &s) in scalars.iter().enumerate() {
            v.try_set(i, T::DimScalar::from_f32(s))?;
        }
        Ok(v)
    }

    /// Try to create a vector from a slice of per-dimension scalars.
    ///
    /// Each element in `dim_values` corresponds to one logical dimension.
    pub fn try_from_dims_in(dim_values: &[T::DimScalar], alloc: A) -> Result<Self, TensorError>
    where
        T: FloatConvertible,
    {
        let n = dim_values.len();
        let mut v = Self::try_zeroed_in(n, alloc)?;
        for (i, &d) in dim_values.iter().enumerate() {
            v.try_set(i, d)?;
        }
        Ok(v)
    }

    /// Number of logical dimensions.
    #[inline]
    pub fn dims(&self) -> usize {
        self.dims
    }

    /// Number of storage values.
    #[inline]
    pub fn len(&self) -> usize {
        self.dims
    }

    /// Number of underlying storage values (T instances).
    #[inline]
    pub fn values(&self) -> usize {
        self.values
    }

    /// Returns true if the vector has zero dimensions.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.dims == 0
    }

    /// Raw pointer to the underlying data.
    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.data.as_ptr()
    }

    /// Mutable raw pointer to the underlying data.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.data.as_ptr()
    }

    /// Size in bytes.
    #[inline]
    pub fn size_bytes(&self) -> usize {
        self.values * core::mem::size_of::<T>()
    }

    /// Create an immutable view of this vector.
    #[inline]
    pub fn view(&self) -> VectorView<'_, T> {
        VectorView {
            data: self.data.as_ptr() as *const T,
            dims: self.dims,
            stride: core::mem::size_of::<T>() as isize,
            _marker: PhantomData,
        }
    }

    /// Create a mutable span of this vector.
    #[inline]
    pub fn span(&mut self) -> VectorSpan<'_, T> {
        VectorSpan {
            data: self.data.as_ptr(),
            dims: self.dims,
            stride: core::mem::size_of::<T>() as isize,
            _marker: PhantomData,
        }
    }

    /// Try to get the logical dimension at `idx` (supports signed indexing).
    ///
    /// Returns the native `DimScalar` type (e.g., `f64` for `Vector<f64>`, `i8` for `Vector<i4x2>`).
    /// For sub-byte types, unpacks the appropriate sub-dimension from the packed storage value.
    #[inline]
    pub fn try_get<I: VecIndex>(&self, idx: I) -> Result<T::DimScalar, TensorError>
    where
        T: FloatConvertible,
    {
        let i = idx
            .resolve(self.dims)
            .ok_or(TensorError::IndexOutOfBounds {
                index: 0,
                size: self.dims,
            })?;
        let dims_per_value = T::dimensions_per_value();
        let value_index = i / dims_per_value;
        let sub_index = i % dims_per_value;
        // SAFETY: value_index < self.values, guaranteed by dims/values invariant
        let packed = unsafe { *self.data.as_ptr().add(value_index) };
        Ok(packed.unpack().as_ref()[sub_index])
    }

    /// Try to set the logical dimension at `idx`.
    ///
    /// Accepts the native `DimScalar` type. For sub-byte types, reads the current packed value,
    /// updates the targeted sub-dimension, and writes back the modified packed value.
    #[inline]
    pub fn try_set<I: VecIndex>(&mut self, idx: I, val: T::DimScalar) -> Result<(), TensorError>
    where
        T: FloatConvertible,
    {
        let i = idx
            .resolve(self.dims)
            .ok_or(TensorError::IndexOutOfBounds {
                index: 0,
                size: self.dims,
            })?;
        let dims_per_value = T::dimensions_per_value();
        let value_index = i / dims_per_value;
        let sub_index = i % dims_per_value;
        // SAFETY: value_index < self.values, guaranteed by dims/values invariant
        let ptr = unsafe { self.data.as_ptr().add(value_index) };
        let mut unpacked = unsafe { *ptr }.unpack();
        unpacked.as_mut()[sub_index] = val;
        unsafe { ptr.write(T::pack(unpacked)) };
        Ok(())
    }

    /// Get a slice of the underlying storage values (only for normal types).
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        if T::dimensions_per_value() == 1 {
            unsafe { core::slice::from_raw_parts(self.data.as_ptr(), self.values) }
        } else {
            // For sub-byte types, return the packed values
            unsafe { core::slice::from_raw_parts(self.data.as_ptr(), self.values) }
        }
    }

    /// Get a mutable slice of the underlying storage values.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { core::slice::from_raw_parts_mut(self.data.as_ptr(), self.values) }
    }

    /// Returns an iterator over the logical dimension values, yielding `DimScalar`.
    pub fn iter(&self) -> DimIter<'_, T>
    where
        T: FloatConvertible,
    {
        self.view().iter()
    }
}

impl<T: FloatLike> Vector<T, Global> {
    /// Create a zero-initialized vector with the global allocator.
    pub fn try_zeroed(dims: usize) -> Result<Self, TensorError> {
        Self::try_zeroed_in(dims, Global)
    }

    /// Create a zero-initialized vector (panics on allocation failure).
    pub fn zeroed(dims: usize) -> Self {
        Self::try_zeroed(dims).expect("allocation failed")
    }

    /// Create a vector filled with `value`.
    pub fn try_filled(dims: usize, value: T) -> Result<Self, TensorError> {
        Self::try_filled_in(dims, value, Global)
    }

    /// Create a vector from scalar f32 values.
    pub fn try_from_scalars(scalars: &[f32]) -> Result<Self, TensorError>
    where
        T: FloatConvertible,
    {
        Self::try_from_scalars_in(scalars, Global)
    }

    /// Create a vector from per-dimension scalars.
    pub fn try_from_dims(dims: &[T::DimScalar]) -> Result<Self, TensorError>
    where
        T: FloatConvertible,
    {
        Self::try_from_dims_in(dims, Global)
    }
}

// Index for normal types (dimensions_per_value == 1)
impl<I: VecIndex, T: FloatLike, A: Allocator> core::ops::Index<I> for Vector<T, A> {
    type Output = T;

    #[inline]
    fn index(&self, idx: I) -> &T {
        let i = idx.resolve(self.dims).expect("vector index out of bounds");
        debug_assert_eq!(
            T::dimensions_per_value(),
            1,
            "Index trait not supported for sub-byte types"
        );
        unsafe { &*self.data.as_ptr().add(i) }
    }
}

impl<I: VecIndex, T: FloatLike, A: Allocator> core::ops::IndexMut<I> for Vector<T, A> {
    #[inline]
    fn index_mut(&mut self, idx: I) -> &mut T {
        let i = idx.resolve(self.dims).expect("vector index out of bounds");
        debug_assert_eq!(
            T::dimensions_per_value(),
            1,
            "IndexMut trait not supported for sub-byte types"
        );
        unsafe { &mut *self.data.as_ptr().add(i) }
    }
}

// endregion: Vector

// region: VectorView

/// Immutable, possibly strided, non-owning view into a vector.
pub struct VectorView<'a, T: FloatLike> {
    data: *const T,
    dims: usize,
    stride: isize,
    _marker: PhantomData<&'a T>,
}

unsafe impl<'a, T: FloatLike + Sync> Send for VectorView<'a, T> {}
unsafe impl<'a, T: FloatLike + Sync> Sync for VectorView<'a, T> {}

impl<'a, T: FloatLike> Clone for VectorView<'a, T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<'a, T: FloatLike> Copy for VectorView<'a, T> {}

impl<'a, T: FloatLike> VectorView<'a, T> {
    /// Number of logical dimensions.
    #[inline]
    pub fn dims(&self) -> usize {
        self.dims
    }

    /// Number of logical dimensions (alias for dims).
    #[inline]
    pub fn len(&self) -> usize {
        self.dims
    }

    /// Returns true if empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.dims == 0
    }

    /// Stride in bytes between consecutive elements.
    #[inline]
    pub fn stride(&self) -> isize {
        self.stride
    }

    /// Returns true if elements are stored contiguously (stride == sizeof(T)).
    #[inline]
    pub fn is_contiguous(&self) -> bool {
        self.stride == core::mem::size_of::<T>() as isize
    }

    /// Get the underlying pointer.
    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.data
    }

    /// Get a contiguous slice, if this view is contiguous.
    #[inline]
    pub fn as_contiguous_slice(&self) -> Option<&'a [T]> {
        if self.is_contiguous() && T::dimensions_per_value() == 1 {
            Some(unsafe { core::slice::from_raw_parts(self.data, self.dims) })
        } else {
            None
        }
    }

    /// Try to get element at index (supports signed indexing).
    ///
    /// Returns the native `DimScalar` type. For sub-byte types, uses value_index
    /// for stride-based pointer walks to avoid buffer overread.
    #[inline]
    pub fn try_get<I: VecIndex>(&self, idx: I) -> Result<T::DimScalar, TensorError>
    where
        T: FloatConvertible,
    {
        let i = idx
            .resolve(self.dims)
            .ok_or(TensorError::IndexOutOfBounds {
                index: 0,
                size: self.dims,
            })?;
        let dims_per_value = T::dimensions_per_value();
        let value_index = i / dims_per_value;
        let sub_index = i % dims_per_value;
        // SAFETY: stride * value_index stays within allocation
        let ptr = unsafe {
            (self.data as *const u8).offset(self.stride * value_index as isize) as *const T
        };
        Ok(unsafe { *ptr }.unpack().as_ref()[sub_index])
    }

    /// Create a reversed view by negating the stride and pointing to the last element.
    ///
    /// The returned view has the same number of dimensions but iterates in the
    /// opposite direction. For an empty view, returns a copy unchanged.
    pub fn rev(&self) -> Self {
        if self.dims == 0 {
            return *self;
        }
        let last_offset = self.stride * (self.dims as isize - 1);
        Self {
            data: unsafe { (self.data as *const u8).offset(last_offset) as *const T },
            dims: self.dims,
            stride: -self.stride,
            _marker: PhantomData,
        }
    }

    /// Create a sub-view with start, end, and step (Python-style slicing).
    ///
    /// Supports negative steps for reverse iteration. `step` must be non-zero.
    /// Returns an error if `start` or `end` exceed `dims()`, or if `step == 0`.
    pub fn try_strided(&self, start: usize, end: usize, step: isize) -> Result<Self, TensorError> {
        if start > self.dims || end > self.dims || step == 0 {
            return Err(TensorError::IndexOutOfBounds {
                index: start.max(end),
                size: self.dims,
            });
        }
        let count = if step > 0 {
            if end > start {
                (end - start + step as usize - 1) / step as usize
            } else {
                0
            }
        } else {
            if start > end {
                let abs_step = (-step) as usize;
                (start - end + abs_step - 1) / abs_step
            } else {
                0
            }
        };
        let new_data =
            unsafe { (self.data as *const u8).offset(self.stride * start as isize) as *const T };
        Ok(Self {
            data: new_data,
            dims: count,
            stride: self.stride * step,
            _marker: PhantomData,
        })
    }

    /// Returns an iterator over logical dimension values, yielding `DimScalar`.
    pub fn iter(&self) -> DimIter<'a, T>
    where
        T: FloatConvertible,
    {
        DimIter {
            data: self.data,
            stride: self.stride,
            front: 0,
            back: self.dims,
            _marker: PhantomData,
        }
    }
}

impl<'a, I: VecIndex, T: FloatLike> core::ops::Index<I> for VectorView<'a, T> {
    type Output = T;

    #[inline]
    fn index(&self, idx: I) -> &T {
        let i = idx.resolve(self.dims).expect("view index out of bounds");
        debug_assert_eq!(
            T::dimensions_per_value(),
            1,
            "Index trait not supported for sub-byte types"
        );
        unsafe { &*((self.data as *const u8).offset(self.stride * i as isize) as *const T) }
    }
}

// endregion: VectorView

// region: VectorSpan

/// Mutable, possibly strided, non-owning view into a vector.
pub struct VectorSpan<'a, T: FloatLike> {
    data: *mut T,
    dims: usize,
    stride: isize,
    _marker: PhantomData<&'a mut T>,
}

unsafe impl<'a, T: FloatLike + Send> Send for VectorSpan<'a, T> {}
unsafe impl<'a, T: FloatLike + Sync> Sync for VectorSpan<'a, T> {}

impl<'a, T: FloatLike> VectorSpan<'a, T> {
    /// Number of logical dimensions.
    #[inline]
    pub fn dims(&self) -> usize {
        self.dims
    }

    /// Number of logical dimensions (alias for dims).
    #[inline]
    pub fn len(&self) -> usize {
        self.dims
    }

    /// Returns true if empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.dims == 0
    }

    /// Stride in bytes.
    #[inline]
    pub fn stride(&self) -> isize {
        self.stride
    }

    /// Returns true if contiguous.
    #[inline]
    pub fn is_contiguous(&self) -> bool {
        self.stride == core::mem::size_of::<T>() as isize
    }

    /// Get the underlying pointer.
    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.data
    }

    /// Get the mutable underlying pointer.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.data
    }

    /// Reborrow as an immutable view, sharing the same data pointer and stride.
    pub fn as_view(&self) -> VectorView<'_, T> {
        VectorView {
            data: self.data,
            dims: self.dims,
            stride: self.stride,
            _marker: PhantomData,
        }
    }

    /// Try to get element at index.
    #[inline]
    pub fn try_get<I: VecIndex>(&self, idx: I) -> Result<T::DimScalar, TensorError>
    where
        T: FloatConvertible,
    {
        self.as_view().try_get(idx)
    }

    /// Try to set the element at `idx`.
    ///
    /// Accepts the native `DimScalar` type. For sub-byte types, uses value_index
    /// for stride-based pointer walks to avoid buffer overwrite.
    #[inline]
    pub fn try_set<I: VecIndex>(&mut self, idx: I, val: T::DimScalar) -> Result<(), TensorError>
    where
        T: FloatConvertible,
    {
        let i = idx
            .resolve(self.dims)
            .ok_or(TensorError::IndexOutOfBounds {
                index: 0,
                size: self.dims,
            })?;
        let dims_per_value = T::dimensions_per_value();
        let value_index = i / dims_per_value;
        let sub_index = i % dims_per_value;
        // SAFETY: stride * value_index stays within allocation
        let ptr =
            unsafe { (self.data as *mut u8).offset(self.stride * value_index as isize) as *mut T };
        let mut unpacked = unsafe { *ptr }.unpack();
        unpacked.as_mut()[sub_index] = val;
        unsafe { ptr.write(T::pack(unpacked)) };
        Ok(())
    }

    /// Fill all elements with a value, respecting the stride between elements.
    ///
    /// Iterates over storage values (not logical dimensions) to avoid buffer
    /// overwrite for sub-byte types.
    pub fn fill(&mut self, val: T) {
        let dims_per_value = T::dimensions_per_value();
        let values = (self.dims + dims_per_value - 1) / dims_per_value;
        for i in 0..values {
            // SAFETY: i < values; stride * i stays within the allocation
            let ptr = unsafe { (self.data as *mut u8).offset(self.stride * i as isize) as *mut T };
            unsafe { ptr.write(val) };
        }
    }

    /// Get a contiguous slice, if this span is contiguous.
    #[inline]
    pub fn as_contiguous_slice(&self) -> Option<&[T]> {
        if self.is_contiguous() && T::dimensions_per_value() == 1 {
            Some(unsafe { core::slice::from_raw_parts(self.data, self.dims) })
        } else {
            None
        }
    }

    /// Get a mutable contiguous slice, if this span is contiguous.
    #[inline]
    pub fn as_contiguous_slice_mut(&mut self) -> Option<&mut [T]> {
        if self.is_contiguous() && T::dimensions_per_value() == 1 {
            Some(unsafe { core::slice::from_raw_parts_mut(self.data, self.dims) })
        } else {
            None
        }
    }

    /// Returns an iterator over logical dimension values, yielding `DimScalar`.
    pub fn iter(&self) -> DimIter<'_, T>
    where
        T: FloatConvertible,
    {
        DimIter {
            data: self.data,
            stride: self.stride,
            front: 0,
            back: self.dims,
            _marker: PhantomData,
        }
    }
}

impl<'a, I: VecIndex, T: FloatLike> core::ops::Index<I> for VectorSpan<'a, T> {
    type Output = T;

    #[inline]
    fn index(&self, idx: I) -> &T {
        let i = idx.resolve(self.dims).expect("span index out of bounds");
        debug_assert_eq!(
            T::dimensions_per_value(),
            1,
            "Index trait not supported for sub-byte types"
        );
        unsafe { &*((self.data as *const u8).offset(self.stride * i as isize) as *const T) }
    }
}

impl<'a, I: VecIndex, T: FloatLike> core::ops::IndexMut<I> for VectorSpan<'a, T> {
    #[inline]
    fn index_mut(&mut self, idx: I) -> &mut T {
        let i = idx.resolve(self.dims).expect("span index out of bounds");
        debug_assert_eq!(
            T::dimensions_per_value(),
            1,
            "IndexMut trait not supported for sub-byte types"
        );
        unsafe { &mut *((self.data as *mut u8).offset(self.stride * i as isize) as *mut T) }
    }
}

// endregion: VectorSpan

// region: Iterators

/// Stride-aware iterator over logical dimensions, yielding `DimScalar` values.
///
/// For normal types (`dimensions_per_value=1`), yields one `T::DimScalar` per storage value.
/// For sub-byte types, unpacks each storage value and yields sub-dimensions individually.
/// Implements `ExactSizeIterator` and `DoubleEndedIterator`.
pub struct DimIter<'a, T: FloatConvertible> {
    data: *const T,
    stride: isize,
    front: usize,
    back: usize,
    _marker: PhantomData<&'a T>,
}

impl<'a, T: FloatConvertible> Iterator for DimIter<'a, T> {
    type Item = T::DimScalar;

    #[inline]
    fn next(&mut self) -> Option<T::DimScalar> {
        if self.front >= self.back {
            return None;
        }
        let dims_per_value = T::dimensions_per_value();
        let value_index = self.front / dims_per_value;
        let sub_index = self.front % dims_per_value;
        // SAFETY: value_index < values, stride * value_index within allocation
        let ptr = unsafe {
            (self.data as *const u8).offset(self.stride * value_index as isize) as *const T
        };
        let scalar = unsafe { *ptr }.unpack().as_ref()[sub_index];
        self.front += 1;
        Some(scalar)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.back - self.front;
        (n, Some(n))
    }
}

impl<'a, T: FloatConvertible> ExactSizeIterator for DimIter<'a, T> {}

impl<'a, T: FloatConvertible> DoubleEndedIterator for DimIter<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<T::DimScalar> {
        if self.front >= self.back {
            return None;
        }
        self.back -= 1;
        let dims_per_value = T::dimensions_per_value();
        let value_index = self.back / dims_per_value;
        let sub_index = self.back % dims_per_value;
        let ptr = unsafe {
            (self.data as *const u8).offset(self.stride * value_index as isize) as *const T
        };
        Some(unsafe { *ptr }.unpack().as_ref()[sub_index])
    }
}

// endregion: Iterators

// region: Tests

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scalar::{bf16, f16, i4x2, u1x8, u4x2};

    fn check_vector_roundtrip<T: FloatConvertible>() {
        let dims_per_value = T::dimensions_per_value();
        let test_dims = 16 * dims_per_value;
        let v = Vector::<T>::try_zeroed(test_dims).unwrap();
        assert_eq!(v.dims(), test_dims);
        assert_eq!(v.values(), test_dims / dims_per_value);
        let count = v.iter().count();
        assert_eq!(count, test_dims);
    }

    fn check_vector_try_get_set<T: FloatConvertible>()
    where
        T::DimScalar: core::fmt::Debug,
    {
        let dims_per_value = T::dimensions_per_value();
        let test_dims = 4 * dims_per_value;
        let mut v = Vector::<T>::try_zeroed(test_dims).unwrap();
        let one = T::DimScalar::from_f32(1.0);
        v.try_set(0_usize, one).unwrap();
        v.try_set((test_dims - 1) as i32, one).unwrap();
        let first = v.try_get(0_usize).unwrap();
        let last = v.try_get(-1_i32).unwrap();
        assert!(
            first.to_f32() >= 0.5,
            "first dim should be ~1.0, got {:?}",
            first
        );
        assert!(
            last.to_f32() >= 0.5,
            "last dim should be ~1.0, got {:?}",
            last
        );
    }

    #[test]
    fn vector_roundtrip_all_types() {
        check_vector_roundtrip::<f32>();
        check_vector_roundtrip::<f64>();
        check_vector_roundtrip::<f16>();
        check_vector_roundtrip::<bf16>();
        check_vector_roundtrip::<i4x2>();
        check_vector_roundtrip::<u4x2>();
        check_vector_roundtrip::<u1x8>();
    }

    #[test]
    fn vector_try_get_set_all_types() {
        check_vector_try_get_set::<f32>();
        check_vector_try_get_set::<f64>();
        check_vector_try_get_set::<i4x2>();
        check_vector_try_get_set::<u4x2>();
        check_vector_try_get_set::<u1x8>();
    }

    #[test]
    fn test_vec_index_signed() {
        let v = Vector::<f32>::try_from_dims(&[10.0, 20.0, 30.0, 40.0, 50.0]).unwrap();
        // Positive indexing
        assert_eq!(v[0], 10.0);
        assert_eq!(v[4], 50.0);
        // Negative indexing (i32 default for integer literals)
        assert_eq!(v[-1_i32], 50.0);
        assert_eq!(v[-2_i32], 40.0);
        assert_eq!(v[-5_i32], 10.0);
    }

    #[test]
    fn test_vector_view_stride() {
        let v = Vector::<f32>::try_from_scalars(&[1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let view = v.view();
        assert!(view.is_contiguous());
        assert_eq!(view.len(), 5);

        // Reversed view
        let rev = view.rev();
        assert_eq!(rev.try_get(0_usize).unwrap(), 5.0);
        assert_eq!(rev.try_get(4_usize).unwrap(), 1.0);
    }

    #[test]
    fn test_vector_span_fill() {
        let mut v = Vector::<f32>::zeroed(4);
        {
            let mut span = v.span();
            span.fill(42.0);
        }
        assert_eq!(v[0], 42.0);
        assert_eq!(v[3], 42.0);
    }

    #[test]
    fn test_vector_iter() {
        let v = Vector::<f32>::try_from_scalars(&[1.0, 2.0, 3.0]).unwrap();
        let vals: Vec<f32> = v.iter().collect();
        assert_eq!(vals, vec![1.0, 2.0, 3.0]);

        // Double-ended
        let rev_vals: Vec<f32> = v.iter().rev().collect();
        assert_eq!(rev_vals, vec![3.0, 2.0, 1.0]);
    }

    #[test]
    fn test_view_strided_iteration() {
        let v = Vector::<f32>::try_from_scalars(&[1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let view = v.view();

        // Every other element
        let strided = view.try_strided(0, 5, 2).unwrap();
        assert_eq!(strided.len(), 3);
        let vals: Vec<f32> = strided.iter().collect();
        assert_eq!(vals, vec![1.0, 3.0, 5.0]);
    }

    #[test]
    fn test_vector_filled() {
        let v = Vector::<f32>::try_filled(3, 7.5).unwrap();
        assert_eq!(v[0], 7.5);
        assert_eq!(v[1], 7.5);
        assert_eq!(v[2], 7.5);
    }

    #[test]
    fn test_empty_vector() {
        let v = Vector::<f32>::zeroed(0);
        assert!(v.is_empty());
        assert_eq!(v.len(), 0);
    }

    #[test]
    #[should_panic]
    fn test_index_out_of_bounds() {
        let v = Vector::<f32>::zeroed(3);
        let _ = v[3_usize];
    }
}

// endregion: Tests

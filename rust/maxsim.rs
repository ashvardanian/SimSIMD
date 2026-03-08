//! MaxSim (ColBERT late-interaction) scoring with pre-packed matrices.
//!
//! [`MaxSimPackedMatrix`] stores vectors in a quantized format optimized for
//! fast coarse screening followed by full-precision refinement.
//!
//! # Example
//!
//! ```rust,ignore
//! // Requires linking against libnumkong C library
//! use numkong::{MaxSimPackedMatrix, Tensor};
//!
//! let queries = Tensor::<f32>::try_new(&[32, 128], 1.0).unwrap();
//! let documents = Tensor::<f32>::try_new(&[1024, 128], 1.0).unwrap();
//!
//! let queries_view = queries.view();
//! let docs_view = documents.view();
//! let queries_packed = MaxSimPackedMatrix::try_pack(&queries_view).unwrap();
//! let docs_packed = MaxSimPackedMatrix::try_pack(&docs_view).unwrap();
//! let score = queries_packed.score(&docs_packed);
//! ```

extern crate alloc;

use core::marker::PhantomData;
use core::ptr::NonNull;

use crate::types::{bf16, f16};
use crate::tensor::{Allocator, Global, ShapeDescriptor, TensorError, TensorView, SIMD_ALIGNMENT};

// region: FFI

#[link(name = "numkong")]
extern "C" {
    fn nk_maxsim_packed_size_f32(n: usize, k: usize) -> usize;
    fn nk_maxsim_pack_f32(v: *const f32, n: usize, k: usize, stride: usize, packed: *mut u8);
    fn nk_maxsim_packed_f32(
        q: *const u8,
        d: *const u8,
        nq: usize,
        nd: usize,
        k: usize,
        result: *mut f32,
    );

    fn nk_maxsim_packed_size_f16(n: usize, k: usize) -> usize;
    fn nk_maxsim_pack_f16(v: *const f16, n: usize, k: usize, stride: usize, packed: *mut u8);
    fn nk_maxsim_packed_f16(
        q: *const u8,
        d: *const u8,
        nq: usize,
        nd: usize,
        k: usize,
        result: *mut f32,
    );

    fn nk_maxsim_packed_size_bf16(n: usize, k: usize) -> usize;
    fn nk_maxsim_pack_bf16(v: *const bf16, n: usize, k: usize, stride: usize, packed: *mut u8);
    fn nk_maxsim_packed_bf16(
        q: *const u8,
        d: *const u8,
        nq: usize,
        nd: usize,
        k: usize,
        result: *mut f32,
    );
}

// endregion: FFI

// region: MaxSim trait

/// Trait abstracting MaxSim pack/score operations per scalar type.
pub trait MaxSim: Sized + Clone {
    /// Score type returned by MaxSim scoring (always f32).
    type Score: Clone + Default;

    /// Returns the packed buffer size in bytes for `n` vectors of depth `k`.
    fn maxsim_packed_size(n: usize, k: usize) -> usize;

    /// Pack vectors into backend-specific quantized format.
    ///
    /// # Safety
    /// - `vectors` must point to `n` rows of `k` elements, byte stride `stride`
    /// - `packed` must have at least `maxsim_packed_size(n, k)` bytes
    unsafe fn maxsim_pack(vectors: *const Self, n: usize, k: usize, stride: usize, packed: *mut u8);

    /// Compute MaxSim score on pre-packed buffers.
    ///
    /// # Safety
    /// - Both buffers must have been produced by `maxsim_pack` with matching depth
    /// - `result` must point to valid, writable memory for `Self::Score`
    unsafe fn maxsim_packed(
        q: *const u8,
        d: *const u8,
        nq: usize,
        nd: usize,
        k: usize,
        result: *mut Self::Score,
    );
}

// endregion: MaxSim trait

// region: MaxSim impls

impl MaxSim for f32 {
    type Score = f32;

    fn maxsim_packed_size(n: usize, k: usize) -> usize {
        unsafe { nk_maxsim_packed_size_f32(n, k) }
    }

    unsafe fn maxsim_pack(
        vectors: *const Self,
        n: usize,
        k: usize,
        stride: usize,
        packed: *mut u8,
    ) {
        nk_maxsim_pack_f32(vectors, n, k, stride, packed)
    }

    unsafe fn maxsim_packed(
        q: *const u8,
        d: *const u8,
        nq: usize,
        nd: usize,
        k: usize,
        result: *mut Self::Score,
    ) {
        nk_maxsim_packed_f32(q, d, nq, nd, k, result)
    }
}

impl MaxSim for f16 {
    type Score = f32;

    fn maxsim_packed_size(n: usize, k: usize) -> usize {
        unsafe { nk_maxsim_packed_size_f16(n, k) }
    }

    unsafe fn maxsim_pack(
        vectors: *const Self,
        n: usize,
        k: usize,
        stride: usize,
        packed: *mut u8,
    ) {
        nk_maxsim_pack_f16(vectors, n, k, stride, packed)
    }

    unsafe fn maxsim_packed(
        q: *const u8,
        d: *const u8,
        nq: usize,
        nd: usize,
        k: usize,
        result: *mut Self::Score,
    ) {
        nk_maxsim_packed_f16(q, d, nq, nd, k, result)
    }
}

impl MaxSim for bf16 {
    type Score = f32;

    fn maxsim_packed_size(n: usize, k: usize) -> usize {
        unsafe { nk_maxsim_packed_size_bf16(n, k) }
    }

    unsafe fn maxsim_pack(
        vectors: *const Self,
        n: usize,
        k: usize,
        stride: usize,
        packed: *mut u8,
    ) {
        nk_maxsim_pack_bf16(vectors, n, k, stride, packed)
    }

    unsafe fn maxsim_packed(
        q: *const u8,
        d: *const u8,
        nq: usize,
        nd: usize,
        k: usize,
        result: *mut Self::Score,
    ) {
        nk_maxsim_packed_bf16(q, d, nq, nd, k, result)
    }
}

// endregion: MaxSim impls

// region: MaxSimPackedMatrix

/// Pre-packed vector set for MaxSim scoring.
///
/// Both query and document vectors must be packed before scoring.
/// The buffer uses i8 quantization for fast coarse screening,
/// with full-precision originals retained for refinement.
pub struct MaxSimPackedMatrix<T: MaxSim, A: Allocator = Global> {
    data: NonNull<u8>,
    size: usize,
    n: usize,
    k: usize,
    alloc: A,
    _marker: PhantomData<T>,
}

// Safety: MaxSimPackedMatrix owns its data and is just bytes
unsafe impl<T: MaxSim + Send, A: Allocator + Send> Send for MaxSimPackedMatrix<T, A> {}
unsafe impl<T: MaxSim + Sync, A: Allocator + Sync> Sync for MaxSimPackedMatrix<T, A> {}

impl<T: MaxSim, A: Allocator> Drop for MaxSimPackedMatrix<T, A> {
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

impl<T: MaxSim, A: Allocator + Clone> Clone for MaxSimPackedMatrix<T, A> {
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

impl<T: MaxSim, A: Allocator> MaxSimPackedMatrix<T, A> {
    /// Pack vectors from a 2D tensor view using a custom allocator.
    ///
    /// Returns `Err` if the view is not 2D, the depth axis is not contiguous,
    /// the row stride is negative, or allocation fails.
    pub fn try_pack_in<const MAX_RANK: usize>(
        vectors: &TensorView<'_, T, MAX_RANK>,
        alloc: A,
    ) -> Result<Self, TensorError> {
        let (vector_count, depth, row_stride_bytes) = validate_maxsim_view(vectors)?;
        let size = T::maxsim_packed_size(vector_count, depth);

        let data = if size == 0 {
            NonNull::dangling()
        } else {
            let layout = alloc::alloc::Layout::from_size_align(size, SIMD_ALIGNMENT)
                .map_err(|_| TensorError::AllocationFailed)?;
            let ptr = alloc
                .allocate(layout)
                .ok_or(TensorError::AllocationFailed)?;
            unsafe {
                core::ptr::write_bytes(ptr.as_ptr(), 0, size);
            }
            ptr
        };

        if size > 0 {
            unsafe {
                T::maxsim_pack(
                    vectors.as_ptr(),
                    vector_count,
                    depth,
                    row_stride_bytes,
                    data.as_ptr(),
                );
            }
        }

        Ok(Self {
            data,
            size,
            n: vector_count,
            k: depth,
            alloc,
            _marker: PhantomData,
        })
    }

    /// Compute MaxSim score against another packed matrix.
    ///
    /// Returns `Err` if depths don't match.
    pub fn try_score<OA: Allocator>(
        &self,
        other: &MaxSimPackedMatrix<T, OA>,
    ) -> Result<T::Score, TensorError> {
        if self.k != other.k {
            return Err(TensorError::DimensionMismatch {
                expected: self.k,
                got: other.k,
            });
        }
        let mut score = T::Score::default();
        unsafe {
            T::maxsim_packed(
                self.as_ptr(),
                other.as_ptr(),
                self.n,
                other.n,
                self.k,
                &mut score,
            )
        };
        Ok(score)
    }

    /// Convenience that panics on error.
    pub fn score<OA: Allocator>(&self, other: &MaxSimPackedMatrix<T, OA>) -> T::Score {
        self.try_score(other)
            .expect("MaxSimPackedMatrix::score failed")
    }

    /// Returns a reference to the allocator.
    pub fn allocator(&self) -> &A {
        &self.alloc
    }

    /// Returns dimensions (n, k) of the original vector set.
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
impl<T: MaxSim> MaxSimPackedMatrix<T, Global> {
    /// Pack vectors from a 2D tensor view using the global allocator.
    pub fn try_pack<const MAX_RANK: usize>(
        vectors: &TensorView<'_, T, MAX_RANK>,
    ) -> Result<Self, TensorError> {
        Self::try_pack_in(vectors, Global)
    }

    /// Convenience constructor that panics on error.
    pub fn pack<const MAX_RANK: usize>(vectors: &TensorView<'_, T, MAX_RANK>) -> Self {
        Self::try_pack(vectors).expect("MaxSimPackedMatrix::pack failed")
    }
}

// endregion: MaxSimPackedMatrix

fn validate_maxsim_view<T, const MAX_RANK: usize>(
    vectors: &TensorView<'_, T, MAX_RANK>,
) -> Result<(usize, usize, usize), TensorError> {
    if vectors.ndim() != 2 {
        return Err(TensorError::DimensionMismatch {
            expected: 2,
            got: vectors.ndim(),
        });
    }

    if !vectors.has_contiguous_rows() {
        return Err(TensorError::NonContiguousRows);
    }

    let row_stride_bytes = vectors.stride_bytes(0);
    if row_stride_bytes < 0 {
        return Err(TensorError::InvalidShape {
            shape: ShapeDescriptor::from_slice(vectors.shape()),
            reason: "MaxSim requires non-negative row strides",
        });
    }

    Ok((
        vectors.shape()[0],
        vectors.shape()[1],
        row_stride_bytes as usize,
    ))
}

// region: TensorView convenience

impl<'a, T: MaxSim, const MAX_RANK: usize> TensorView<'a, T, MAX_RANK> {
    /// Pack this 2D tensor view for MaxSim scoring using the provided allocator.
    pub fn try_maxsim_pack_in<A: Allocator>(
        &self,
        alloc: A,
    ) -> Result<MaxSimPackedMatrix<T, A>, TensorError> {
        MaxSimPackedMatrix::try_pack_in(self, alloc)
    }

    /// Pack this 2D tensor view for MaxSim scoring using the global allocator.
    pub fn try_maxsim_pack(&self) -> Result<MaxSimPackedMatrix<T, Global>, TensorError> {
        self.try_maxsim_pack_in(Global)
    }
}

// endregion: TensorView convenience

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{SliceRange, Tensor};

    #[test]
    fn maxsim_packs_from_tensor_view() {
        let queries = Tensor::<f32>::try_full(&[4, 16], 1.0).unwrap();
        let docs = Tensor::<f32>::try_full(&[8, 16], 1.0).unwrap();

        let queries_packed = queries.view().try_maxsim_pack().unwrap();
        let docs_packed = docs.view().try_maxsim_pack().unwrap();

        assert_eq!(queries_packed.dims(), (4, 16));
        assert_eq!(docs_packed.dims(), (8, 16));
        assert!(queries_packed.score(&docs_packed).is_finite());
    }

    #[test]
    fn maxsim_rejects_non_contiguous_depth_axis() {
        let queries = Tensor::<f32>::try_full(&[4, 16], 1.0).unwrap();
        let transposed = queries.t().unwrap();
        let result = transposed.try_maxsim_pack();
        assert!(matches!(result, Err(TensorError::NonContiguousRows)));
    }

    #[test]
    fn maxsim_accepts_outer_strided_views() {
        let queries = Tensor::<f32>::try_full(&[8, 16], 1.0).unwrap();
        let odd_rows = queries
            .slice(&[
                SliceRange::range_step(1, 7, 2),
                SliceRange::range_step(0, 16, 1),
            ])
            .unwrap();

        let queries_packed = odd_rows.try_maxsim_pack().unwrap();
        assert_eq!(queries_packed.dims(), (3, 16));
    }

    #[test]
    fn maxsim_rejects_negative_row_stride() {
        let queries = Tensor::<f32>::try_full(&[8, 16], 1.0).unwrap();
        let reversed_rows = queries
            .slice(&[
                SliceRange::range_step(7, 0, -1),
                SliceRange::range_step(0, 16, 1),
            ])
            .unwrap();

        let result = reversed_rows.try_maxsim_pack();
        assert!(matches!(result, Err(TensorError::InvalidShape { .. })));
    }
}

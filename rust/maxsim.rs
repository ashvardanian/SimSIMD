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
//! let packed_q = MaxSimPackedMatrix::try_pack(&queries).unwrap();
//! let packed_d = MaxSimPackedMatrix::try_pack(&documents).unwrap();
//! let score = packed_q.score(&packed_d);
//! ```

extern crate alloc;

use core::marker::PhantomData;
use core::ptr::NonNull;

use crate::scalar::{bf16, f16};
use crate::tensor::{Allocator, Global, Tensor, TensorError, SIMD_ALIGNMENT};

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
    fn nk_maxsim_pack_f16(v: *const u16, n: usize, k: usize, stride: usize, packed: *mut u8);
    fn nk_maxsim_packed_f16(
        q: *const u8,
        d: *const u8,
        nq: usize,
        nd: usize,
        k: usize,
        result: *mut f32,
    );

    fn nk_maxsim_packed_size_bf16(n: usize, k: usize) -> usize;
    fn nk_maxsim_pack_bf16(v: *const u16, n: usize, k: usize, stride: usize, packed: *mut u8);
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
        nk_maxsim_pack_f16(vectors as *const u16, n, k, stride, packed)
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
        nk_maxsim_pack_bf16(vectors as *const u16, n, k, stride, packed)
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
    /// Pack vectors from a 2D tensor using a custom allocator.
    ///
    /// Returns `Err` if the tensor is not 2D or allocation fails.
    pub fn try_pack_in<BA: Allocator, const MAX_RANK: usize>(
        vectors: &Tensor<T, BA, MAX_RANK>,
        alloc: A,
    ) -> Result<Self, TensorError> {
        if vectors.ndim() != 2 {
            return Err(TensorError::DimensionMismatch {
                expected: 2,
                got: vectors.ndim(),
            });
        }
        let (n, k) = (vectors.shape()[0], vectors.shape()[1]);
        let size = T::maxsim_packed_size(n, k);

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
                    n,
                    k,
                    vectors.stride(0) as usize,
                    data.as_ptr(),
                );
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
    /// Pack vectors from a 2D tensor using the global allocator.
    pub fn try_pack<BA: Allocator, const MAX_RANK: usize>(
        vectors: &Tensor<T, BA, MAX_RANK>,
    ) -> Result<Self, TensorError> {
        Self::try_pack_in(vectors, Global)
    }

    /// Convenience constructor that panics on error.
    pub fn pack<BA: Allocator, const MAX_RANK: usize>(vectors: &Tensor<T, BA, MAX_RANK>) -> Self {
        Self::try_pack(vectors).expect("MaxSimPackedMatrix::pack failed")
    }
}

// endregion: MaxSimPackedMatrix

// region: Tensor convenience

impl<T: MaxSim, A: Allocator + Clone, const MAX_RANK: usize> Tensor<T, A, MAX_RANK> {
    /// Pack this tensor's vectors for MaxSim scoring.
    pub fn try_maxsim_pack(&self) -> Result<MaxSimPackedMatrix<T, A>, TensorError> {
        MaxSimPackedMatrix::try_pack_in(self, self.allocator().clone())
    }

    /// Convenience that panics on error.
    pub fn maxsim_pack(&self) -> MaxSimPackedMatrix<T, A> {
        self.try_maxsim_pack().expect("maxsim_pack failed")
    }
}

// endregion: Tensor convenience

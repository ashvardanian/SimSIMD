//! Statistical reductions: moments (sum/sum-of-squares) and min/max.
//!
//! This module provides:
//!
//! - [`ReduceMoments`]: Sum and sum-of-squares over a strided slice
//! - [`ReduceMinMax`]: Minimum and maximum over a strided slice
//! - [`Reductions`]: Blanket trait combining `ReduceMoments + ReduceMinMax`

use crate::types::{bf16, e2m3, e3m2, e4m3, e5m2, f16, i4x2, u1x8, u4x2, StorageElement};

#[link(name = "numkong")]
extern "C" {
    // Reductions: moments (sum + sum-of-squares)
    fn nk_reduce_moments_f64(
        data: *const f64,
        count: usize,
        stride_bytes: usize,
        sum: *mut f64,
        sumsq: *mut f64,
    );
    fn nk_reduce_moments_f32(
        data: *const f32,
        count: usize,
        stride_bytes: usize,
        sum: *mut f64,
        sumsq: *mut f64,
    );
    fn nk_reduce_moments_i8(
        data: *const i8,
        count: usize,
        stride_bytes: usize,
        sum: *mut i64,
        sumsq: *mut u64,
    );
    fn nk_reduce_moments_u8(
        data: *const u8,
        count: usize,
        stride_bytes: usize,
        sum: *mut u64,
        sumsq: *mut u64,
    );
    fn nk_reduce_moments_i16(
        data: *const i16,
        count: usize,
        stride_bytes: usize,
        sum: *mut i64,
        sumsq: *mut u64,
    );
    fn nk_reduce_moments_u16(
        data: *const u16,
        count: usize,
        stride_bytes: usize,
        sum: *mut u64,
        sumsq: *mut u64,
    );
    fn nk_reduce_moments_i32(
        data: *const i32,
        count: usize,
        stride_bytes: usize,
        sum: *mut i64,
        sumsq: *mut u64,
    );
    fn nk_reduce_moments_u32(
        data: *const u32,
        count: usize,
        stride_bytes: usize,
        sum: *mut u64,
        sumsq: *mut u64,
    );
    fn nk_reduce_moments_i64(
        data: *const i64,
        count: usize,
        stride_bytes: usize,
        sum: *mut i64,
        sumsq: *mut u64,
    );
    fn nk_reduce_moments_u64(
        data: *const u64,
        count: usize,
        stride_bytes: usize,
        sum: *mut u64,
        sumsq: *mut u64,
    );
    fn nk_reduce_moments_f16(
        data: *const f16,
        count: usize,
        stride_bytes: usize,
        sum: *mut f32,
        sumsq: *mut f32,
    );
    fn nk_reduce_moments_bf16(
        data: *const bf16,
        count: usize,
        stride_bytes: usize,
        sum: *mut f32,
        sumsq: *mut f32,
    );
    fn nk_reduce_moments_e4m3(
        data: *const e4m3,
        count: usize,
        stride_bytes: usize,
        sum: *mut f32,
        sumsq: *mut f32,
    );
    fn nk_reduce_moments_e5m2(
        data: *const e5m2,
        count: usize,
        stride_bytes: usize,
        sum: *mut f32,
        sumsq: *mut f32,
    );
    fn nk_reduce_moments_e2m3(
        data: *const e2m3,
        count: usize,
        stride_bytes: usize,
        sum: *mut f32,
        sumsq: *mut f32,
    );
    fn nk_reduce_moments_e3m2(
        data: *const e3m2,
        count: usize,
        stride_bytes: usize,
        sum: *mut f32,
        sumsq: *mut f32,
    );
    fn nk_reduce_moments_i4(
        data: *const i4x2,
        count: usize,
        stride_bytes: usize,
        sum: *mut i64,
        sumsq: *mut u64,
    );
    fn nk_reduce_moments_u4(
        data: *const u4x2,
        count: usize,
        stride_bytes: usize,
        sum: *mut u64,
        sumsq: *mut u64,
    );
    fn nk_reduce_moments_u1(
        data: *const u1x8,
        count: usize,
        stride_bytes: usize,
        sum: *mut u64,
        sumsq: *mut u64,
    );

    // Reductions: minmax (min + max + argmin + argmax)
    fn nk_reduce_minmax_f64(
        data: *const f64,
        count: usize,
        stride_bytes: usize,
        min_val: *mut f64,
        min_idx: *mut usize,
        max_val: *mut f64,
        max_idx: *mut usize,
    );
    fn nk_reduce_minmax_f32(
        data: *const f32,
        count: usize,
        stride_bytes: usize,
        min_val: *mut f32,
        min_idx: *mut usize,
        max_val: *mut f32,
        max_idx: *mut usize,
    );
    fn nk_reduce_minmax_i8(
        data: *const i8,
        count: usize,
        stride_bytes: usize,
        min_val: *mut i8,
        min_idx: *mut usize,
        max_val: *mut i8,
        max_idx: *mut usize,
    );
    fn nk_reduce_minmax_u8(
        data: *const u8,
        count: usize,
        stride_bytes: usize,
        min_val: *mut u8,
        min_idx: *mut usize,
        max_val: *mut u8,
        max_idx: *mut usize,
    );
    fn nk_reduce_minmax_i16(
        data: *const i16,
        count: usize,
        stride_bytes: usize,
        min_val: *mut i16,
        min_idx: *mut usize,
        max_val: *mut i16,
        max_idx: *mut usize,
    );
    fn nk_reduce_minmax_u16(
        data: *const u16,
        count: usize,
        stride_bytes: usize,
        min_val: *mut u16,
        min_idx: *mut usize,
        max_val: *mut u16,
        max_idx: *mut usize,
    );
    fn nk_reduce_minmax_i32(
        data: *const i32,
        count: usize,
        stride_bytes: usize,
        min_val: *mut i32,
        min_idx: *mut usize,
        max_val: *mut i32,
        max_idx: *mut usize,
    );
    fn nk_reduce_minmax_u32(
        data: *const u32,
        count: usize,
        stride_bytes: usize,
        min_val: *mut u32,
        min_idx: *mut usize,
        max_val: *mut u32,
        max_idx: *mut usize,
    );
    fn nk_reduce_minmax_i64(
        data: *const i64,
        count: usize,
        stride_bytes: usize,
        min_val: *mut i64,
        min_idx: *mut usize,
        max_val: *mut i64,
        max_idx: *mut usize,
    );
    fn nk_reduce_minmax_u64(
        data: *const u64,
        count: usize,
        stride_bytes: usize,
        min_val: *mut u64,
        min_idx: *mut usize,
        max_val: *mut u64,
        max_idx: *mut usize,
    );
    fn nk_reduce_minmax_f16(
        data: *const f16,
        count: usize,
        stride_bytes: usize,
        min_val: *mut f16,
        min_idx: *mut usize,
        max_val: *mut f16,
        max_idx: *mut usize,
    );
    fn nk_reduce_minmax_bf16(
        data: *const bf16,
        count: usize,
        stride_bytes: usize,
        min_val: *mut bf16,
        min_idx: *mut usize,
        max_val: *mut bf16,
        max_idx: *mut usize,
    );
    fn nk_reduce_minmax_e4m3(
        data: *const e4m3,
        count: usize,
        stride_bytes: usize,
        min_val: *mut e4m3,
        min_idx: *mut usize,
        max_val: *mut e4m3,
        max_idx: *mut usize,
    );
    fn nk_reduce_minmax_e5m2(
        data: *const e5m2,
        count: usize,
        stride_bytes: usize,
        min_val: *mut e5m2,
        min_idx: *mut usize,
        max_val: *mut e5m2,
        max_idx: *mut usize,
    );
    fn nk_reduce_minmax_e2m3(
        data: *const e2m3,
        count: usize,
        stride_bytes: usize,
        min_val: *mut e2m3,
        min_idx: *mut usize,
        max_val: *mut e2m3,
        max_idx: *mut usize,
    );
    fn nk_reduce_minmax_e3m2(
        data: *const e3m2,
        count: usize,
        stride_bytes: usize,
        min_val: *mut e3m2,
        min_idx: *mut usize,
        max_val: *mut e3m2,
        max_idx: *mut usize,
    );
    fn nk_reduce_minmax_i4(
        data: *const i4x2,
        count: usize,
        stride_bytes: usize,
        min_val: *mut i8,
        min_idx: *mut usize,
        max_val: *mut i8,
        max_idx: *mut usize,
    );
    fn nk_reduce_minmax_u4(
        data: *const u4x2,
        count: usize,
        stride_bytes: usize,
        min_val: *mut u8,
        min_idx: *mut usize,
        max_val: *mut u8,
        max_idx: *mut usize,
    );
    fn nk_reduce_minmax_u1(
        data: *const u1x8,
        count: usize,
        stride_bytes: usize,
        min_val: *mut u8,
        min_idx: *mut usize,
        max_val: *mut u8,
        max_idx: *mut usize,
    );
}

/// Compute first and second moments (sum and sum-of-squares) with stride support.
///
/// Returns `(sum, sum_of_squares)` for all elements in a slice, with optional striding.
/// The output types may be wider than the input to avoid overflow.
pub trait ReduceMoments: StorageElement {
    /// Type for the sum output.
    type SumOutput: StorageElement;
    /// Type for the sum-of-squares output.
    type SumSqOutput: StorageElement;
    /// Compute `(sum, sum_of_squares)` for raw pointer input with the given stride in bytes.
    ///
    /// # Safety
    /// `data` must point to at least one reachable element when `count > 0`, and every logical
    /// element addressed by `count` and `stride_bytes` must be valid to read.
    unsafe fn reduce_moments_raw(
        data: *const Self,
        count: usize,
        stride_bytes: usize,
    ) -> (Self::SumOutput, Self::SumSqOutput);
    /// Compute `(sum, sum_of_squares)` for `data` with the given stride (in bytes).
    /// Use `stride_bytes = size_of::<Self>()` for contiguous data.
    fn reduce_moments(data: &[Self], stride_bytes: usize) -> (Self::SumOutput, Self::SumSqOutput) {
        unsafe { Self::reduce_moments_raw(data.as_ptr(), data.len(), stride_bytes) }
    }
}

unsafe fn reduce_moments_via_ffi<Scalar, Sum: Default, SumSq: Default>(
    data: *const Scalar,
    count: usize,
    stride_bytes: usize,
    ffi: unsafe extern "C" fn(*const Scalar, usize, usize, *mut Sum, *mut SumSq),
) -> (Sum, SumSq)
where
    Scalar: StorageElement,
{
    let mut sum: Sum = Default::default();
    let mut sumsq: SumSq = Default::default();
    ffi(
        data,
        count * Scalar::dimensions_per_value(),
        stride_bytes,
        &mut sum,
        &mut sumsq,
    );
    (sum, sumsq)
}

impl ReduceMoments for f64 {
    type SumOutput = f64;
    type SumSqOutput = f64;

    unsafe fn reduce_moments_raw(
        data: *const Self,
        count: usize,
        stride_bytes: usize,
    ) -> (Self::SumOutput, Self::SumSqOutput) {
        reduce_moments_via_ffi(data, count, stride_bytes, nk_reduce_moments_f64)
    }
}

impl ReduceMoments for f32 {
    type SumOutput = f64;
    type SumSqOutput = f64;

    unsafe fn reduce_moments_raw(
        data: *const Self,
        count: usize,
        stride_bytes: usize,
    ) -> (Self::SumOutput, Self::SumSqOutput) {
        reduce_moments_via_ffi(data, count, stride_bytes, nk_reduce_moments_f32)
    }
}

impl ReduceMoments for i8 {
    type SumOutput = i64;
    type SumSqOutput = u64;

    unsafe fn reduce_moments_raw(
        data: *const Self,
        count: usize,
        stride_bytes: usize,
    ) -> (Self::SumOutput, Self::SumSqOutput) {
        reduce_moments_via_ffi(data, count, stride_bytes, nk_reduce_moments_i8)
    }
}

impl ReduceMoments for u8 {
    type SumOutput = u64;
    type SumSqOutput = u64;

    unsafe fn reduce_moments_raw(
        data: *const Self,
        count: usize,
        stride_bytes: usize,
    ) -> (Self::SumOutput, Self::SumSqOutput) {
        reduce_moments_via_ffi(data, count, stride_bytes, nk_reduce_moments_u8)
    }
}

impl ReduceMoments for i16 {
    type SumOutput = i64;
    type SumSqOutput = u64;

    unsafe fn reduce_moments_raw(
        data: *const Self,
        count: usize,
        stride_bytes: usize,
    ) -> (Self::SumOutput, Self::SumSqOutput) {
        reduce_moments_via_ffi(data, count, stride_bytes, nk_reduce_moments_i16)
    }
}

impl ReduceMoments for u16 {
    type SumOutput = u64;
    type SumSqOutput = u64;

    unsafe fn reduce_moments_raw(
        data: *const Self,
        count: usize,
        stride_bytes: usize,
    ) -> (Self::SumOutput, Self::SumSqOutput) {
        reduce_moments_via_ffi(data, count, stride_bytes, nk_reduce_moments_u16)
    }
}

impl ReduceMoments for i32 {
    type SumOutput = i64;
    type SumSqOutput = u64;

    unsafe fn reduce_moments_raw(
        data: *const Self,
        count: usize,
        stride_bytes: usize,
    ) -> (Self::SumOutput, Self::SumSqOutput) {
        reduce_moments_via_ffi(data, count, stride_bytes, nk_reduce_moments_i32)
    }
}

impl ReduceMoments for u32 {
    type SumOutput = u64;
    type SumSqOutput = u64;

    unsafe fn reduce_moments_raw(
        data: *const Self,
        count: usize,
        stride_bytes: usize,
    ) -> (Self::SumOutput, Self::SumSqOutput) {
        reduce_moments_via_ffi(data, count, stride_bytes, nk_reduce_moments_u32)
    }
}

impl ReduceMoments for i64 {
    type SumOutput = i64;
    type SumSqOutput = u64;

    unsafe fn reduce_moments_raw(
        data: *const Self,
        count: usize,
        stride_bytes: usize,
    ) -> (Self::SumOutput, Self::SumSqOutput) {
        reduce_moments_via_ffi(data, count, stride_bytes, nk_reduce_moments_i64)
    }
}

impl ReduceMoments for u64 {
    type SumOutput = u64;
    type SumSqOutput = u64;

    unsafe fn reduce_moments_raw(
        data: *const Self,
        count: usize,
        stride_bytes: usize,
    ) -> (Self::SumOutput, Self::SumSqOutput) {
        reduce_moments_via_ffi(data, count, stride_bytes, nk_reduce_moments_u64)
    }
}

impl ReduceMoments for f16 {
    type SumOutput = f32;
    type SumSqOutput = f32;

    unsafe fn reduce_moments_raw(
        data: *const Self,
        count: usize,
        stride_bytes: usize,
    ) -> (Self::SumOutput, Self::SumSqOutput) {
        reduce_moments_via_ffi(data, count, stride_bytes, nk_reduce_moments_f16)
    }
}

impl ReduceMoments for bf16 {
    type SumOutput = f32;
    type SumSqOutput = f32;

    unsafe fn reduce_moments_raw(
        data: *const Self,
        count: usize,
        stride_bytes: usize,
    ) -> (Self::SumOutput, Self::SumSqOutput) {
        reduce_moments_via_ffi(data, count, stride_bytes, nk_reduce_moments_bf16)
    }
}

impl ReduceMoments for e4m3 {
    type SumOutput = f32;
    type SumSqOutput = f32;

    unsafe fn reduce_moments_raw(
        data: *const Self,
        count: usize,
        stride_bytes: usize,
    ) -> (Self::SumOutput, Self::SumSqOutput) {
        reduce_moments_via_ffi(data, count, stride_bytes, nk_reduce_moments_e4m3)
    }
}

impl ReduceMoments for e5m2 {
    type SumOutput = f32;
    type SumSqOutput = f32;

    unsafe fn reduce_moments_raw(
        data: *const Self,
        count: usize,
        stride_bytes: usize,
    ) -> (Self::SumOutput, Self::SumSqOutput) {
        reduce_moments_via_ffi(data, count, stride_bytes, nk_reduce_moments_e5m2)
    }
}

impl ReduceMoments for e2m3 {
    type SumOutput = f32;
    type SumSqOutput = f32;

    unsafe fn reduce_moments_raw(
        data: *const Self,
        count: usize,
        stride_bytes: usize,
    ) -> (Self::SumOutput, Self::SumSqOutput) {
        reduce_moments_via_ffi(data, count, stride_bytes, nk_reduce_moments_e2m3)
    }
}

impl ReduceMoments for e3m2 {
    type SumOutput = f32;
    type SumSqOutput = f32;

    unsafe fn reduce_moments_raw(
        data: *const Self,
        count: usize,
        stride_bytes: usize,
    ) -> (Self::SumOutput, Self::SumSqOutput) {
        reduce_moments_via_ffi(data, count, stride_bytes, nk_reduce_moments_e3m2)
    }
}

impl ReduceMoments for i4x2 {
    type SumOutput = i64;
    type SumSqOutput = u64;

    unsafe fn reduce_moments_raw(
        data: *const Self,
        count: usize,
        stride_bytes: usize,
    ) -> (Self::SumOutput, Self::SumSqOutput) {
        reduce_moments_via_ffi(data, count, stride_bytes, nk_reduce_moments_i4)
    }
}

impl ReduceMoments for u4x2 {
    type SumOutput = u64;
    type SumSqOutput = u64;

    unsafe fn reduce_moments_raw(
        data: *const Self,
        count: usize,
        stride_bytes: usize,
    ) -> (Self::SumOutput, Self::SumSqOutput) {
        reduce_moments_via_ffi(data, count, stride_bytes, nk_reduce_moments_u4)
    }
}

impl ReduceMoments for u1x8 {
    type SumOutput = u64;
    type SumSqOutput = u64;

    unsafe fn reduce_moments_raw(
        data: *const Self,
        count: usize,
        stride_bytes: usize,
    ) -> (Self::SumOutput, Self::SumSqOutput) {
        reduce_moments_via_ffi(data, count, stride_bytes, nk_reduce_moments_u1)
    }
}

/// Find minimum and maximum values with their indices, with stride support.
///
/// Returns `Some((min_value, min_index, max_value, max_index))` for all elements in a slice,
/// or `None` if all elements are NaN (for NaN-masking formats).
/// The output value type matches the logical reduced scalar type.
pub trait ReduceMinMax: StorageElement {
    /// Output type for the min/max values — matches the C layer's native type.
    type Output: StorageElement;
    /// Whether `NK_SIZE_MAX` indicates that the reduction produced no value.
    const NONE_ON_SENTINEL: bool;
    /// Returns `Some((min_value, min_index, max_value, max_index))` for raw pointer input with the
    /// specified stride, or `None` if all elements are NaN.
    ///
    /// # Safety
    /// `data` must point to at least one reachable element when `count > 0`, and every logical
    /// element addressed by `count` and `stride_bytes` must be valid to read.
    unsafe fn reduce_minmax_raw(
        data: *const Self,
        count: usize,
        stride_bytes: usize,
    ) -> Option<(Self::Output, usize, Self::Output, usize)>;
    /// Returns `Some((min_value, min_index, max_value, max_index))` for the given data with the
    /// specified stride, or `None` if all elements are NaN.
    fn reduce_minmax(
        data: &[Self],
        stride_bytes: usize,
    ) -> Option<(Self::Output, usize, Self::Output, usize)> {
        unsafe { Self::reduce_minmax_raw(data.as_ptr(), data.len(), stride_bytes) }
    }
}

unsafe fn reduce_minmax_via_ffi<Scalar, Out: Default>(
    data: *const Scalar,
    count: usize,
    stride_bytes: usize,
    none_on_sentinel: bool,
    ffi: unsafe extern "C" fn(
        *const Scalar,
        usize,
        usize,
        *mut Out,
        *mut usize,
        *mut Out,
        *mut usize,
    ),
) -> Option<(Out, usize, Out, usize)>
where
    Scalar: StorageElement,
{
    let mut min_value: Out = Default::default();
    let mut min_index: usize = 0;
    let mut max_value: Out = Default::default();
    let mut max_index: usize = 0;
    ffi(
        data,
        count * Scalar::dimensions_per_value(),
        stride_bytes,
        &mut min_value,
        &mut min_index,
        &mut max_value,
        &mut max_index,
    );
    if none_on_sentinel && min_index == usize::MAX {
        return None;
    }
    Some((min_value, min_index, max_value, max_index))
}

impl ReduceMinMax for f64 {
    type Output = f64;
    const NONE_ON_SENTINEL: bool = true;

    unsafe fn reduce_minmax_raw(
        data: *const Self,
        count: usize,
        stride_bytes: usize,
    ) -> Option<(Self::Output, usize, Self::Output, usize)> {
        reduce_minmax_via_ffi(
            data,
            count,
            stride_bytes,
            Self::NONE_ON_SENTINEL,
            nk_reduce_minmax_f64,
        )
    }
}

impl ReduceMinMax for f32 {
    type Output = f32;
    const NONE_ON_SENTINEL: bool = true;

    unsafe fn reduce_minmax_raw(
        data: *const Self,
        count: usize,
        stride_bytes: usize,
    ) -> Option<(Self::Output, usize, Self::Output, usize)> {
        reduce_minmax_via_ffi(
            data,
            count,
            stride_bytes,
            Self::NONE_ON_SENTINEL,
            nk_reduce_minmax_f32,
        )
    }
}

impl ReduceMinMax for i8 {
    type Output = i8;
    const NONE_ON_SENTINEL: bool = false;

    unsafe fn reduce_minmax_raw(
        data: *const Self,
        count: usize,
        stride_bytes: usize,
    ) -> Option<(Self::Output, usize, Self::Output, usize)> {
        reduce_minmax_via_ffi(
            data,
            count,
            stride_bytes,
            Self::NONE_ON_SENTINEL,
            nk_reduce_minmax_i8,
        )
    }
}

impl ReduceMinMax for u8 {
    type Output = u8;
    const NONE_ON_SENTINEL: bool = false;

    unsafe fn reduce_minmax_raw(
        data: *const Self,
        count: usize,
        stride_bytes: usize,
    ) -> Option<(Self::Output, usize, Self::Output, usize)> {
        reduce_minmax_via_ffi(
            data,
            count,
            stride_bytes,
            Self::NONE_ON_SENTINEL,
            nk_reduce_minmax_u8,
        )
    }
}

impl ReduceMinMax for i16 {
    type Output = i16;
    const NONE_ON_SENTINEL: bool = false;

    unsafe fn reduce_minmax_raw(
        data: *const Self,
        count: usize,
        stride_bytes: usize,
    ) -> Option<(Self::Output, usize, Self::Output, usize)> {
        reduce_minmax_via_ffi(
            data,
            count,
            stride_bytes,
            Self::NONE_ON_SENTINEL,
            nk_reduce_minmax_i16,
        )
    }
}

impl ReduceMinMax for u16 {
    type Output = u16;
    const NONE_ON_SENTINEL: bool = false;

    unsafe fn reduce_minmax_raw(
        data: *const Self,
        count: usize,
        stride_bytes: usize,
    ) -> Option<(Self::Output, usize, Self::Output, usize)> {
        reduce_minmax_via_ffi(
            data,
            count,
            stride_bytes,
            Self::NONE_ON_SENTINEL,
            nk_reduce_minmax_u16,
        )
    }
}

impl ReduceMinMax for i32 {
    type Output = i32;
    const NONE_ON_SENTINEL: bool = false;

    unsafe fn reduce_minmax_raw(
        data: *const Self,
        count: usize,
        stride_bytes: usize,
    ) -> Option<(Self::Output, usize, Self::Output, usize)> {
        reduce_minmax_via_ffi(
            data,
            count,
            stride_bytes,
            Self::NONE_ON_SENTINEL,
            nk_reduce_minmax_i32,
        )
    }
}

impl ReduceMinMax for u32 {
    type Output = u32;
    const NONE_ON_SENTINEL: bool = false;

    unsafe fn reduce_minmax_raw(
        data: *const Self,
        count: usize,
        stride_bytes: usize,
    ) -> Option<(Self::Output, usize, Self::Output, usize)> {
        reduce_minmax_via_ffi(
            data,
            count,
            stride_bytes,
            Self::NONE_ON_SENTINEL,
            nk_reduce_minmax_u32,
        )
    }
}

impl ReduceMinMax for i64 {
    type Output = i64;
    const NONE_ON_SENTINEL: bool = false;

    unsafe fn reduce_minmax_raw(
        data: *const Self,
        count: usize,
        stride_bytes: usize,
    ) -> Option<(Self::Output, usize, Self::Output, usize)> {
        reduce_minmax_via_ffi(
            data,
            count,
            stride_bytes,
            Self::NONE_ON_SENTINEL,
            nk_reduce_minmax_i64,
        )
    }
}

impl ReduceMinMax for u64 {
    type Output = u64;
    const NONE_ON_SENTINEL: bool = false;

    unsafe fn reduce_minmax_raw(
        data: *const Self,
        count: usize,
        stride_bytes: usize,
    ) -> Option<(Self::Output, usize, Self::Output, usize)> {
        reduce_minmax_via_ffi(
            data,
            count,
            stride_bytes,
            Self::NONE_ON_SENTINEL,
            nk_reduce_minmax_u64,
        )
    }
}

impl ReduceMinMax for f16 {
    type Output = f16;
    const NONE_ON_SENTINEL: bool = true;

    unsafe fn reduce_minmax_raw(
        data: *const Self,
        count: usize,
        stride_bytes: usize,
    ) -> Option<(Self::Output, usize, Self::Output, usize)> {
        reduce_minmax_via_ffi(
            data,
            count,
            stride_bytes,
            Self::NONE_ON_SENTINEL,
            nk_reduce_minmax_f16,
        )
    }
}

impl ReduceMinMax for bf16 {
    type Output = bf16;
    const NONE_ON_SENTINEL: bool = true;

    unsafe fn reduce_minmax_raw(
        data: *const Self,
        count: usize,
        stride_bytes: usize,
    ) -> Option<(Self::Output, usize, Self::Output, usize)> {
        reduce_minmax_via_ffi(
            data,
            count,
            stride_bytes,
            Self::NONE_ON_SENTINEL,
            nk_reduce_minmax_bf16,
        )
    }
}

impl ReduceMinMax for e4m3 {
    type Output = e4m3;
    const NONE_ON_SENTINEL: bool = true;

    unsafe fn reduce_minmax_raw(
        data: *const Self,
        count: usize,
        stride_bytes: usize,
    ) -> Option<(Self::Output, usize, Self::Output, usize)> {
        reduce_minmax_via_ffi(
            data,
            count,
            stride_bytes,
            Self::NONE_ON_SENTINEL,
            nk_reduce_minmax_e4m3,
        )
    }
}

impl ReduceMinMax for e5m2 {
    type Output = e5m2;
    const NONE_ON_SENTINEL: bool = true;

    unsafe fn reduce_minmax_raw(
        data: *const Self,
        count: usize,
        stride_bytes: usize,
    ) -> Option<(Self::Output, usize, Self::Output, usize)> {
        reduce_minmax_via_ffi(
            data,
            count,
            stride_bytes,
            Self::NONE_ON_SENTINEL,
            nk_reduce_minmax_e5m2,
        )
    }
}

impl ReduceMinMax for e2m3 {
    type Output = e2m3;
    const NONE_ON_SENTINEL: bool = false;

    unsafe fn reduce_minmax_raw(
        data: *const Self,
        count: usize,
        stride_bytes: usize,
    ) -> Option<(Self::Output, usize, Self::Output, usize)> {
        reduce_minmax_via_ffi(
            data,
            count,
            stride_bytes,
            Self::NONE_ON_SENTINEL,
            nk_reduce_minmax_e2m3,
        )
    }
}

impl ReduceMinMax for e3m2 {
    type Output = e3m2;
    const NONE_ON_SENTINEL: bool = false;

    unsafe fn reduce_minmax_raw(
        data: *const Self,
        count: usize,
        stride_bytes: usize,
    ) -> Option<(Self::Output, usize, Self::Output, usize)> {
        reduce_minmax_via_ffi(
            data,
            count,
            stride_bytes,
            Self::NONE_ON_SENTINEL,
            nk_reduce_minmax_e3m2,
        )
    }
}

impl ReduceMinMax for i4x2 {
    type Output = i8;
    const NONE_ON_SENTINEL: bool = false;

    unsafe fn reduce_minmax_raw(
        data: *const Self,
        count: usize,
        stride_bytes: usize,
    ) -> Option<(Self::Output, usize, Self::Output, usize)> {
        reduce_minmax_via_ffi(
            data,
            count,
            stride_bytes,
            Self::NONE_ON_SENTINEL,
            nk_reduce_minmax_i4,
        )
    }
}

impl ReduceMinMax for u4x2 {
    type Output = u8;
    const NONE_ON_SENTINEL: bool = false;

    unsafe fn reduce_minmax_raw(
        data: *const Self,
        count: usize,
        stride_bytes: usize,
    ) -> Option<(Self::Output, usize, Self::Output, usize)> {
        reduce_minmax_via_ffi(
            data,
            count,
            stride_bytes,
            Self::NONE_ON_SENTINEL,
            nk_reduce_minmax_u4,
        )
    }
}

impl ReduceMinMax for u1x8 {
    type Output = u8;
    const NONE_ON_SENTINEL: bool = false;

    unsafe fn reduce_minmax_raw(
        data: *const Self,
        count: usize,
        stride_bytes: usize,
    ) -> Option<(Self::Output, usize, Self::Output, usize)> {
        reduce_minmax_via_ffi(
            data,
            count,
            stride_bytes,
            Self::NONE_ON_SENTINEL,
            nk_reduce_minmax_u1,
        )
    }
}

/// `Reductions` bundles reduction operations: ReduceMoments and ReduceMinMax.
pub trait Reductions: ReduceMoments + ReduceMinMax {}
impl<Scalar: ReduceMoments + ReduceMinMax> Reductions for Scalar {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{
        assert_close, bf16, e2m3, e3m2, e4m3, e5m2, f16, FloatLike, NumberLike, TestableType,
    };

    // region: ReduceMoments

    fn check_reduce_moments<Scalar>(input_values: &[f32])
    where
        Scalar: FloatLike + TestableType + ReduceMoments,
        Scalar::SumOutput: FloatLike,
        Scalar::SumSqOutput: FloatLike,
    {
        let data: Vec<Scalar> = input_values.iter().map(|&v| Scalar::from_f32(v)).collect();
        let stride_bytes = core::mem::size_of::<Scalar>();
        let (actual_sum, actual_sumsq) = Scalar::reduce_moments(&data, stride_bytes);
        let expected_sum: f64 = input_values.iter().map(|&v| v as f64).sum();
        let expected_sumsq: f64 = input_values.iter().map(|&v| (v as f64) * (v as f64)).sum();
        let sample_count = input_values.len() as f64;
        assert_close(
            actual_sum.to_f64(),
            expected_sum,
            Scalar::atol() * sample_count,
            Scalar::rtol(),
            &format!("reduce_moments<{}> sum", core::any::type_name::<Scalar>()),
        );
        assert_close(
            actual_sumsq.to_f64(),
            expected_sumsq,
            Scalar::atol() * sample_count,
            Scalar::rtol(),
            &format!("reduce_moments<{}> sumsq", core::any::type_name::<Scalar>()),
        );
    }

    #[test]
    fn moments_reduction() {
        // Float types — SumOutput/SumSqOutput are FloatLike (f32 or f64)
        let input_values = &[1.0, 2.0, 3.0, 4.0, 5.0];
        check_reduce_moments::<f64>(input_values);
        check_reduce_moments::<f32>(input_values);
        check_reduce_moments::<f16>(input_values);
        check_reduce_moments::<bf16>(input_values);
        check_reduce_moments::<e4m3>(input_values);
        check_reduce_moments::<e5m2>(&[1.0, 2.0, 3.0]);
        check_reduce_moments::<e2m3>(&[1.0, 2.0, 3.0]);
        check_reduce_moments::<e3m2>(&[1.0, 2.0, 3.0]);

        // Integer types — now also go through generics with FloatLike for i64/u64
        let signed = &[1.0_f32, -2.0, 3.0, -4.0, 5.0];
        let unsigned = &[1.0_f32, 2.0, 3.0, 4.0, 5.0];
        check_reduce_moments::<i8>(signed);
        check_reduce_moments::<u8>(unsigned);
        check_reduce_moments::<i16>(signed);
        check_reduce_moments::<u16>(unsigned);
        check_reduce_moments::<i32>(signed);
        check_reduce_moments::<u32>(unsigned);
        check_reduce_moments::<i64>(signed);
        check_reduce_moments::<u64>(unsigned);
    }

    // endregion

    // region: ReduceMinMax

    fn check_reduce_minmax<Scalar>(input_values: &[f32])
    where
        Scalar: FloatLike + TestableType + ReduceMinMax,
        Scalar::Output: FloatLike,
    {
        let data: Vec<Scalar> = input_values.iter().map(|&v| Scalar::from_f32(v)).collect();
        let stride_bytes = core::mem::size_of::<Scalar>();
        let result = Scalar::reduce_minmax(&data, stride_bytes);
        assert!(result.is_some(), "Expected Some for non-NaN input");
        let (actual_min, actual_min_index, actual_max, actual_max_index) = result.unwrap();
        let (expected_min_index, expected_min) = input_values
            .iter()
            .enumerate()
            .min_by(|left, right| left.1.partial_cmp(right.1).unwrap())
            .unwrap();
        let (expected_max_index, expected_max) = input_values
            .iter()
            .enumerate()
            .max_by(|left, right| left.1.partial_cmp(right.1).unwrap())
            .unwrap();
        assert_close(
            actual_min.to_f64(),
            *expected_min as f64,
            Scalar::atol(),
            0.0,
            &format!("reduce_minmax<{}> min", core::any::type_name::<Scalar>()),
        );
        assert_eq!(
            actual_min_index,
            expected_min_index,
            "reduce_minmax<{}> min_index",
            core::any::type_name::<Scalar>()
        );
        assert_close(
            actual_max.to_f64(),
            *expected_max as f64,
            Scalar::atol(),
            0.0,
            &format!("reduce_minmax<{}> max", core::any::type_name::<Scalar>()),
        );
        assert_eq!(
            actual_max_index,
            expected_max_index,
            "reduce_minmax<{}> max_index",
            core::any::type_name::<Scalar>()
        );
    }

    #[test]
    fn minmax_reduction() {
        // All FloatLike types — Output is also FloatLike
        let input_values = &[3.0, 1.0, 4.0, 1.5, 5.0, 2.0];
        check_reduce_minmax::<f64>(input_values);
        check_reduce_minmax::<f32>(input_values);
        check_reduce_minmax::<f16>(input_values);
        check_reduce_minmax::<bf16>(input_values);
        check_reduce_minmax::<e4m3>(input_values);
        check_reduce_minmax::<e5m2>(input_values);
        check_reduce_minmax::<e2m3>(&[3.0, 1.0, 4.0, 1.5, 5.0, 2.0]);
        check_reduce_minmax::<e3m2>(input_values);
        check_reduce_minmax::<i8>(input_values);
        check_reduce_minmax::<u8>(input_values);
        check_reduce_minmax::<i32>(input_values);
        check_reduce_minmax::<u32>(input_values);

        // i16, u16, i64, u64 — now also go through generics with FloatLike
        check_reduce_minmax::<i16>(&[3.0, -1.0, 4.0, -5.0, 2.0]);
        check_reduce_minmax::<u16>(&[3.0, 1.0, 4.0, 5.0, 2.0]);
        check_reduce_minmax::<i64>(&[3.0, -1.0, 4.0, -5.0, 2.0]);
        check_reduce_minmax::<u64>(&[3.0, 1.0, 4.0, 5.0, 2.0]);
    }

    #[test]
    fn minmax_reduction_all_nan() {
        let nan_f64: Vec<f64> = vec![f64::NAN; 16];
        assert_eq!(
            f64::reduce_minmax(&nan_f64, core::mem::size_of::<f64>()),
            None
        );

        let nan_f32: Vec<f32> = vec![f32::NAN; 16];
        assert_eq!(
            f32::reduce_minmax(&nan_f32, core::mem::size_of::<f32>()),
            None
        );

        let nan_f16: Vec<f16> = vec![f16::NAN; 16];
        assert_eq!(
            f16::reduce_minmax(&nan_f16, core::mem::size_of::<f16>()),
            None
        );

        let nan_bf16: Vec<bf16> = vec![bf16::NAN; 16];
        assert_eq!(
            bf16::reduce_minmax(&nan_bf16, core::mem::size_of::<bf16>()),
            None
        );

        let nan_e4m3: Vec<e4m3> = vec![e4m3::NAN; 16];
        assert_eq!(
            e4m3::reduce_minmax(&nan_e4m3, core::mem::size_of::<e4m3>()),
            None
        );

        let nan_e5m2: Vec<e5m2> = vec![e5m2::NAN; 16];
        assert_eq!(
            e5m2::reduce_minmax(&nan_e5m2, core::mem::size_of::<e5m2>()),
            None
        );
    }

    #[test]
    fn minmax_reduction_mixed_nan() {
        let data = vec![f64::NAN, 3.0, f64::NAN, 1.0, f64::NAN, 5.0];
        let result = f64::reduce_minmax(&data, core::mem::size_of::<f64>());
        assert!(result.is_some());
        let (min_value, min_index, max_value, max_index) = result.unwrap();
        assert_close(min_value, 1.0, 1e-10, 0.0, "mixed min");
        assert_eq!(min_index, 3);
        assert_close(max_value, 5.0, 1e-10, 0.0, "mixed max");
        assert_eq!(max_index, 5);

        let data_f32: Vec<f32> = vec![f32::NAN, 3.0, f32::NAN, 1.0, f32::NAN, 5.0];
        let result_f32 = f32::reduce_minmax(&data_f32, core::mem::size_of::<f32>());
        assert!(result_f32.is_some());
        let (min_value, min_index, max_value, max_index) = result_f32.unwrap();
        assert_close(min_value as f64, 1.0, 1e-5, 0.0, "mixed f32 min");
        assert_eq!(min_index, 3);
        assert_close(max_value as f64, 5.0, 1e-5, 0.0, "mixed f32 max");
        assert_eq!(max_index, 5);
    }

    // endregion
}

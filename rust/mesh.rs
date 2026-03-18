//! Mesh superposition and alignment: Kabsch, Umeyama, RMSD.
//!
//! This module provides:
//!
//! - [`MeshAlignmentResult`]: Centroids, rotation matrix, scale, and RMSD
//! - [`MeshAlignment`]: Computes optimal rigid-body alignment of 3D point sets

use crate::types::{bf16, f16};

#[link(name = "numkong")]
extern "C" {
    fn nk_rmsd_f32(
        a: *const f32,
        b: *const f32,
        n: usize,
        a_centroid: *mut f32,
        b_centroid: *mut f32,
        rotation: *mut f32,
        scale: *mut f32,
        result: *mut f64,
    );
    fn nk_rmsd_f64(
        a: *const f64,
        b: *const f64,
        n: usize,
        a_centroid: *mut f64,
        b_centroid: *mut f64,
        rotation: *mut f64,
        scale: *mut f64,
        result: *mut f64,
    );
    fn nk_rmsd_f16(
        a: *const u16,
        b: *const u16,
        n: usize,
        a_centroid: *mut f32,
        b_centroid: *mut f32,
        rotation: *mut f32,
        scale: *mut f32,
        result: *mut f32,
    );
    fn nk_rmsd_bf16(
        a: *const u16,
        b: *const u16,
        n: usize,
        a_centroid: *mut f32,
        b_centroid: *mut f32,
        rotation: *mut f32,
        scale: *mut f32,
        result: *mut f32,
    );
    fn nk_kabsch_f32(
        a: *const f32,
        b: *const f32,
        n: usize,
        a_centroid: *mut f32,
        b_centroid: *mut f32,
        rotation: *mut f32,
        scale: *mut f32,
        result: *mut f64,
    );
    fn nk_kabsch_f64(
        a: *const f64,
        b: *const f64,
        n: usize,
        a_centroid: *mut f64,
        b_centroid: *mut f64,
        rotation: *mut f64,
        scale: *mut f64,
        result: *mut f64,
    );
    fn nk_kabsch_f16(
        a: *const u16,
        b: *const u16,
        n: usize,
        a_centroid: *mut f32,
        b_centroid: *mut f32,
        rotation: *mut f32,
        scale: *mut f32,
        result: *mut f32,
    );
    fn nk_kabsch_bf16(
        a: *const u16,
        b: *const u16,
        n: usize,
        a_centroid: *mut f32,
        b_centroid: *mut f32,
        rotation: *mut f32,
        scale: *mut f32,
        result: *mut f32,
    );
    fn nk_umeyama_f32(
        a: *const f32,
        b: *const f32,
        n: usize,
        a_centroid: *mut f32,
        b_centroid: *mut f32,
        rotation: *mut f32,
        scale: *mut f32,
        result: *mut f64,
    );
    fn nk_umeyama_f64(
        a: *const f64,
        b: *const f64,
        n: usize,
        a_centroid: *mut f64,
        b_centroid: *mut f64,
        rotation: *mut f64,
        scale: *mut f64,
        result: *mut f64,
    );
    fn nk_umeyama_f16(
        a: *const u16,
        b: *const u16,
        n: usize,
        a_centroid: *mut f32,
        b_centroid: *mut f32,
        rotation: *mut f32,
        scale: *mut f32,
        result: *mut f32,
    );
    fn nk_umeyama_bf16(
        a: *const u16,
        b: *const u16,
        n: usize,
        a_centroid: *mut f32,
        b_centroid: *mut f32,
        rotation: *mut f32,
        scale: *mut f32,
        result: *mut f32,
    );
}

/// Result of mesh alignment operations (RMSD, Kabsch, Umeyama).
///
/// Contains the rigid-body transformation (rotation, scale, translation)
/// that best aligns point cloud A onto point cloud B, along with the
/// root-mean-square deviation of the aligned points.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeshAlignmentResult<TTransform, TMetric> {
    /// 3×3 rotation matrix in row-major order.
    pub rotation_matrix: [TTransform; 9],
    /// Uniform scale factor (1.0 for Kabsch, free for Umeyama).
    pub scale: TTransform,
    /// Root-mean-square deviation after alignment.
    pub rmsd: TMetric,
    /// Centroid of point cloud A before alignment.
    pub a_centroid: [TTransform; 3],
    /// Centroid of point cloud B (target).
    pub b_centroid: [TTransform; 3],
}

impl<TMetric> MeshAlignmentResult<f64, TMetric> {
    #[inline]
    pub fn transform_point(&self, point: [f64; 3]) -> [f64; 3] {
        let centered = [
            point[0] - self.a_centroid[0],
            point[1] - self.a_centroid[1],
            point[2] - self.a_centroid[2],
        ];
        let r = &self.rotation_matrix;
        [
            self.scale * (r[0] * centered[0] + r[1] * centered[1] + r[2] * centered[2])
                + self.b_centroid[0],
            self.scale * (r[3] * centered[0] + r[4] * centered[1] + r[5] * centered[2])
                + self.b_centroid[1],
            self.scale * (r[6] * centered[0] + r[7] * centered[1] + r[8] * centered[2])
                + self.b_centroid[2],
        ]
    }

    #[cfg(feature = "std")]
    pub fn transform_points(&self, points: &[[f64; 3]]) -> Vec<[f64; 3]> {
        points.iter().map(|&p| self.transform_point(p)).collect()
    }
}

impl<TMetric> MeshAlignmentResult<f32, TMetric> {
    #[inline]
    pub fn transform_point(&self, point: [f32; 3]) -> [f32; 3] {
        let centered = [
            point[0] - self.a_centroid[0],
            point[1] - self.a_centroid[1],
            point[2] - self.a_centroid[2],
        ];
        let r = &self.rotation_matrix;
        [
            self.scale * (r[0] * centered[0] + r[1] * centered[1] + r[2] * centered[2])
                + self.b_centroid[0],
            self.scale * (r[3] * centered[0] + r[4] * centered[1] + r[5] * centered[2])
                + self.b_centroid[1],
            self.scale * (r[6] * centered[0] + r[7] * centered[1] + r[8] * centered[2])
                + self.b_centroid[2],
        ]
    }

    #[cfg(feature = "std")]
    pub fn transform_points(&self, points: &[[f32; 3]]) -> Vec<[f32; 3]> {
        points.iter().map(|&p| self.transform_point(p)).collect()
    }
}

/// Mesh alignment operations for 3D point clouds.
pub trait MeshAlignment: Sized {
    type Transform: Default + Copy;
    type Metric: Default + Copy;

    fn rmsd(
        a: &[[Self; 3]],
        b: &[[Self; 3]],
    ) -> Option<MeshAlignmentResult<Self::Transform, Self::Metric>>;
    fn kabsch(
        a: &[[Self; 3]],
        b: &[[Self; 3]],
    ) -> Option<MeshAlignmentResult<Self::Transform, Self::Metric>>;
    fn umeyama(
        a: &[[Self; 3]],
        b: &[[Self; 3]],
    ) -> Option<MeshAlignmentResult<Self::Transform, Self::Metric>>;
}

impl MeshAlignment for f64 {
    type Transform = f64;
    type Metric = f64;

    fn rmsd(
        a: &[[Self; 3]],
        b: &[[Self; 3]],
    ) -> Option<MeshAlignmentResult<Self::Transform, Self::Metric>> {
        if a.len() != b.len() || a.len() < 3 {
            return None;
        }
        let mut result = MeshAlignmentResult {
            rotation_matrix: [0.0; 9],
            scale: 0.0,
            rmsd: 0.0,
            a_centroid: [0.0; 3],
            b_centroid: [0.0; 3],
        };
        unsafe {
            nk_rmsd_f64(
                a.as_ptr() as *const f64,
                b.as_ptr() as *const f64,
                a.len(),
                result.a_centroid.as_mut_ptr(),
                result.b_centroid.as_mut_ptr(),
                result.rotation_matrix.as_mut_ptr(),
                &mut result.scale,
                &mut result.rmsd,
            )
        };
        Some(result)
    }

    fn kabsch(
        a: &[[Self; 3]],
        b: &[[Self; 3]],
    ) -> Option<MeshAlignmentResult<Self::Transform, Self::Metric>> {
        if a.len() != b.len() || a.len() < 3 {
            return None;
        }
        let mut result = MeshAlignmentResult {
            rotation_matrix: [0.0; 9],
            scale: 0.0,
            rmsd: 0.0,
            a_centroid: [0.0; 3],
            b_centroid: [0.0; 3],
        };
        unsafe {
            nk_kabsch_f64(
                a.as_ptr() as *const f64,
                b.as_ptr() as *const f64,
                a.len(),
                result.a_centroid.as_mut_ptr(),
                result.b_centroid.as_mut_ptr(),
                result.rotation_matrix.as_mut_ptr(),
                &mut result.scale,
                &mut result.rmsd,
            )
        };
        Some(result)
    }

    fn umeyama(
        a: &[[Self; 3]],
        b: &[[Self; 3]],
    ) -> Option<MeshAlignmentResult<Self::Transform, Self::Metric>> {
        if a.len() != b.len() || a.len() < 3 {
            return None;
        }
        let mut result = MeshAlignmentResult {
            rotation_matrix: [0.0; 9],
            scale: 0.0,
            rmsd: 0.0,
            a_centroid: [0.0; 3],
            b_centroid: [0.0; 3],
        };
        unsafe {
            nk_umeyama_f64(
                a.as_ptr() as *const f64,
                b.as_ptr() as *const f64,
                a.len(),
                result.a_centroid.as_mut_ptr(),
                result.b_centroid.as_mut_ptr(),
                result.rotation_matrix.as_mut_ptr(),
                &mut result.scale,
                &mut result.rmsd,
            )
        };
        Some(result)
    }
}

impl MeshAlignment for f32 {
    type Transform = f32;
    type Metric = f64;

    fn rmsd(
        a: &[[Self; 3]],
        b: &[[Self; 3]],
    ) -> Option<MeshAlignmentResult<Self::Transform, Self::Metric>> {
        if a.len() != b.len() || a.len() < 3 {
            return None;
        }
        let mut result = MeshAlignmentResult {
            rotation_matrix: [0.0; 9],
            scale: 0.0,
            rmsd: 0.0,
            a_centroid: [0.0; 3],
            b_centroid: [0.0; 3],
        };
        unsafe {
            nk_rmsd_f32(
                a.as_ptr() as *const f32,
                b.as_ptr() as *const f32,
                a.len(),
                result.a_centroid.as_mut_ptr(),
                result.b_centroid.as_mut_ptr(),
                result.rotation_matrix.as_mut_ptr(),
                &mut result.scale,
                &mut result.rmsd,
            )
        };
        Some(result)
    }

    fn kabsch(
        a: &[[Self; 3]],
        b: &[[Self; 3]],
    ) -> Option<MeshAlignmentResult<Self::Transform, Self::Metric>> {
        if a.len() != b.len() || a.len() < 3 {
            return None;
        }
        let mut result = MeshAlignmentResult {
            rotation_matrix: [0.0; 9],
            scale: 0.0,
            rmsd: 0.0,
            a_centroid: [0.0; 3],
            b_centroid: [0.0; 3],
        };
        unsafe {
            nk_kabsch_f32(
                a.as_ptr() as *const f32,
                b.as_ptr() as *const f32,
                a.len(),
                result.a_centroid.as_mut_ptr(),
                result.b_centroid.as_mut_ptr(),
                result.rotation_matrix.as_mut_ptr(),
                &mut result.scale,
                &mut result.rmsd,
            )
        };
        Some(result)
    }

    fn umeyama(
        a: &[[Self; 3]],
        b: &[[Self; 3]],
    ) -> Option<MeshAlignmentResult<Self::Transform, Self::Metric>> {
        if a.len() != b.len() || a.len() < 3 {
            return None;
        }
        let mut result = MeshAlignmentResult {
            rotation_matrix: [0.0; 9],
            scale: 0.0,
            rmsd: 0.0,
            a_centroid: [0.0; 3],
            b_centroid: [0.0; 3],
        };
        unsafe {
            nk_umeyama_f32(
                a.as_ptr() as *const f32,
                b.as_ptr() as *const f32,
                a.len(),
                result.a_centroid.as_mut_ptr(),
                result.b_centroid.as_mut_ptr(),
                result.rotation_matrix.as_mut_ptr(),
                &mut result.scale,
                &mut result.rmsd,
            )
        };
        Some(result)
    }
}

impl MeshAlignment for f16 {
    type Transform = f32;
    type Metric = f32;

    fn rmsd(
        a: &[[Self; 3]],
        b: &[[Self; 3]],
    ) -> Option<MeshAlignmentResult<Self::Transform, Self::Metric>> {
        if a.len() != b.len() || a.len() < 3 {
            return None;
        }
        let mut result = MeshAlignmentResult {
            rotation_matrix: [0.0; 9],
            scale: 0.0,
            rmsd: 0.0,
            a_centroid: [0.0; 3],
            b_centroid: [0.0; 3],
        };
        unsafe {
            nk_rmsd_f16(
                a.as_ptr() as *const u16,
                b.as_ptr() as *const u16,
                a.len(),
                result.a_centroid.as_mut_ptr(),
                result.b_centroid.as_mut_ptr(),
                result.rotation_matrix.as_mut_ptr(),
                &mut result.scale,
                &mut result.rmsd,
            )
        };
        Some(result)
    }

    fn kabsch(
        a: &[[Self; 3]],
        b: &[[Self; 3]],
    ) -> Option<MeshAlignmentResult<Self::Transform, Self::Metric>> {
        if a.len() != b.len() || a.len() < 3 {
            return None;
        }
        let mut result = MeshAlignmentResult {
            rotation_matrix: [0.0; 9],
            scale: 0.0,
            rmsd: 0.0,
            a_centroid: [0.0; 3],
            b_centroid: [0.0; 3],
        };
        unsafe {
            nk_kabsch_f16(
                a.as_ptr() as *const u16,
                b.as_ptr() as *const u16,
                a.len(),
                result.a_centroid.as_mut_ptr(),
                result.b_centroid.as_mut_ptr(),
                result.rotation_matrix.as_mut_ptr(),
                &mut result.scale,
                &mut result.rmsd,
            )
        };
        Some(result)
    }

    fn umeyama(
        a: &[[Self; 3]],
        b: &[[Self; 3]],
    ) -> Option<MeshAlignmentResult<Self::Transform, Self::Metric>> {
        if a.len() != b.len() || a.len() < 3 {
            return None;
        }
        let mut result = MeshAlignmentResult {
            rotation_matrix: [0.0; 9],
            scale: 0.0,
            rmsd: 0.0,
            a_centroid: [0.0; 3],
            b_centroid: [0.0; 3],
        };
        unsafe {
            nk_umeyama_f16(
                a.as_ptr() as *const u16,
                b.as_ptr() as *const u16,
                a.len(),
                result.a_centroid.as_mut_ptr(),
                result.b_centroid.as_mut_ptr(),
                result.rotation_matrix.as_mut_ptr(),
                &mut result.scale,
                &mut result.rmsd,
            )
        };
        Some(result)
    }
}

impl MeshAlignment for bf16 {
    type Transform = f32;
    type Metric = f32;

    fn rmsd(
        a: &[[Self; 3]],
        b: &[[Self; 3]],
    ) -> Option<MeshAlignmentResult<Self::Transform, Self::Metric>> {
        if a.len() != b.len() || a.len() < 3 {
            return None;
        }
        let mut result = MeshAlignmentResult {
            rotation_matrix: [0.0; 9],
            scale: 0.0,
            rmsd: 0.0,
            a_centroid: [0.0; 3],
            b_centroid: [0.0; 3],
        };
        unsafe {
            nk_rmsd_bf16(
                a.as_ptr() as *const u16,
                b.as_ptr() as *const u16,
                a.len(),
                result.a_centroid.as_mut_ptr(),
                result.b_centroid.as_mut_ptr(),
                result.rotation_matrix.as_mut_ptr(),
                &mut result.scale,
                &mut result.rmsd,
            )
        };
        Some(result)
    }

    fn kabsch(
        a: &[[Self; 3]],
        b: &[[Self; 3]],
    ) -> Option<MeshAlignmentResult<Self::Transform, Self::Metric>> {
        if a.len() != b.len() || a.len() < 3 {
            return None;
        }
        let mut result = MeshAlignmentResult {
            rotation_matrix: [0.0; 9],
            scale: 0.0,
            rmsd: 0.0,
            a_centroid: [0.0; 3],
            b_centroid: [0.0; 3],
        };
        unsafe {
            nk_kabsch_bf16(
                a.as_ptr() as *const u16,
                b.as_ptr() as *const u16,
                a.len(),
                result.a_centroid.as_mut_ptr(),
                result.b_centroid.as_mut_ptr(),
                result.rotation_matrix.as_mut_ptr(),
                &mut result.scale,
                &mut result.rmsd,
            )
        };
        Some(result)
    }

    fn umeyama(
        a: &[[Self; 3]],
        b: &[[Self; 3]],
    ) -> Option<MeshAlignmentResult<Self::Transform, Self::Metric>> {
        if a.len() != b.len() || a.len() < 3 {
            return None;
        }
        let mut result = MeshAlignmentResult {
            rotation_matrix: [0.0; 9],
            scale: 0.0,
            rmsd: 0.0,
            a_centroid: [0.0; 3],
            b_centroid: [0.0; 3],
        };
        unsafe {
            nk_umeyama_bf16(
                a.as_ptr() as *const u16,
                b.as_ptr() as *const u16,
                a.len(),
                result.a_centroid.as_mut_ptr(),
                result.b_centroid.as_mut_ptr(),
                result.rotation_matrix.as_mut_ptr(),
                &mut result.scale,
                &mut result.rmsd,
            )
        };
        Some(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{assert_close, FloatLike, NumberLike, TestableType};
    /// Convert a point cloud from f32 to T.
    pub(crate) fn convert_cloud<T: FloatLike>(cloud: &[[f32; 3]]) -> Vec<[T; 3]> {
        cloud
            .iter()
            .map(|p| [T::from_f32(p[0]), T::from_f32(p[1]), T::from_f32(p[2])])
            .collect()
    }

    fn check_kabsch_identical<T>(cloud: &[[f32; 3]])
    where
        T: FloatLike + TestableType + MeshAlignment,
        T::Transform: NumberLike,
        T::Metric: FloatLike,
    {
        let cloud_t = convert_cloud::<T>(cloud);
        let result = T::kabsch(&cloud_t, &cloud_t).unwrap();
        let tol = T::atol() + T::rtol();
        assert_close(
            NumberLike::to_f64(result.scale),
            1.0,
            tol,
            0.0,
            &format!("kabsch<{}> scale", core::any::type_name::<T>()),
        );
        assert_close(
            NumberLike::to_f64(result.rmsd),
            0.0,
            tol,
            0.0,
            &format!("kabsch<{}> rmsd", core::any::type_name::<T>()),
        );
    }

    fn check_umeyama_scaled<T>(cloud: &[[f32; 3]], scaled: &[[f32; 3]])
    where
        T: FloatLike + TestableType + MeshAlignment,
        T::Transform: NumberLike,
        T::Metric: FloatLike,
    {
        let cloud_t = convert_cloud::<T>(cloud);
        let scaled_t = convert_cloud::<T>(scaled);
        let result = T::umeyama(&cloud_t, &scaled_t).unwrap();
        let scale = NumberLike::to_f64(result.scale);
        assert!(
            scale > 1.0 && scale < 3.0,
            "umeyama<{}> scale: expected ~2.0, got {}",
            core::any::type_name::<T>(),
            scale
        );
    }

    fn check_rmsd_identical<T>(cloud: &[[f32; 3]])
    where
        T: FloatLike + TestableType + MeshAlignment,
        T::Transform: NumberLike,
        T::Metric: FloatLike,
    {
        let cloud_t = convert_cloud::<T>(cloud);
        let result = T::rmsd(&cloud_t, &cloud_t).unwrap();
        let tol = T::atol() + T::rtol();
        assert_close(
            NumberLike::to_f64(result.scale),
            1.0,
            tol,
            0.0,
            &format!("rmsd<{}> scale", core::any::type_name::<T>()),
        );
        assert_close(
            NumberLike::to_f64(result.rmsd),
            0.0,
            tol,
            0.0,
            &format!("rmsd<{}> rmsd", core::any::type_name::<T>()),
        );
    }

    #[test]
    fn mesh_alignment() {
        let cloud: &[[f32; 3]] = &[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ];
        let scaled: &[[f32; 3]] = &[
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
            [0.0, 0.0, 6.0],
        ];
        let tri: &[[f32; 3]] = &[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

        // Kabsch — identical clouds
        check_kabsch_identical::<f64>(cloud);
        check_kabsch_identical::<f32>(cloud);

        // Umeyama — 2x scaled
        check_umeyama_scaled::<f64>(cloud, scaled);
        check_umeyama_scaled::<f32>(cloud, scaled);

        // RMSD — identical
        check_rmsd_identical::<f64>(tri);
        check_rmsd_identical::<f32>(tri);
    }

    #[test]
    fn mesh_alignment_edge_cases() {
        let tri: &[[f64; 3]] = &[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let pair: &[[f64; 3]] = &[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];

        // Length mismatch
        assert!(f64::kabsch(tri, pair).is_none());
        assert!(f64::rmsd(tri, pair).is_none());
        assert!(f64::umeyama(tri, pair).is_none());

        // Too few points
        assert!(f64::kabsch(pair, pair).is_none());

        // Kabsch on random-ish clouds: check outputs are finite and scale == 1.
        // The underlying SVD solver (Jacobi, 16 fixed iterations) is only approximate,
        // so we verify structural properties rather than exact transform recovery.
        let cloud_a: &[[f64; 3]] = &[
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 2.0],
            [1.0, 1.0, 1.0],
            [-1.0, 0.0, 1.0],
        ];
        let cloud_b: &[[f64; 3]] = &[
            [0.0, 2.0, 0.5],
            [-2.0, 0.1, 0.0],
            [0.3, 0.0, 2.0],
            [-1.0, 1.2, 1.0],
            [0.0, -1.0, 1.5],
        ];
        let result = f64::kabsch(cloud_a, cloud_b).unwrap();

        // Scale must be 1.0 (Kabsch is rigid, no scaling)
        assert!(
            (result.scale - 1.0).abs() < 1e-6,
            "Expected scale ~1.0, got {}",
            result.scale
        );
        // RMSD must be finite and non-negative
        assert!(result.rmsd.is_finite() && result.rmsd >= 0.0);
        // Rotation determinant must be ±1 (orthogonal matrix)
        let r = &result.rotation_matrix;
        let det = r[0] * (r[4] * r[8] - r[5] * r[7]) - r[1] * (r[3] * r[8] - r[5] * r[6])
            + r[2] * (r[3] * r[7] - r[4] * r[6]);
        assert!(
            (det.abs() - 1.0).abs() < 0.01,
            "Expected det(R) ~±1.0, got {}",
            det
        );
        // Centroids must be finite
        assert!(result.a_centroid.iter().all(|c| c.is_finite()));
        assert!(result.b_centroid.iter().all(|c| c.is_finite()));
        // transform_point must produce finite output
        let transformed = result.transform_point([1.0, 2.0, 3.0]);
        assert!(transformed.iter().all(|c| c.is_finite()));
    }
}

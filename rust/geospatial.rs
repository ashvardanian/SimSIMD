//! Geospatial distance functions: Haversine and Vincenty.
//!
//! This module provides:
//!
//! - [`Haversine`]: Great-circle distance on a sphere
//! - [`Vincenty`]: Geodesic distance on an ellipsoid
//! - [`Geospatial`]: Blanket trait combining `Haversine + Vincenty`

#[link(name = "numkong")]
extern "C" {
    fn nk_haversine_f32(
        a_lats: *const f32,
        a_lons: *const f32,
        b_lats: *const f32,
        b_lons: *const f32,
        n: usize,
        results: *mut f32,
    );
    fn nk_haversine_f64(
        a_lats: *const f64,
        a_lons: *const f64,
        b_lats: *const f64,
        b_lons: *const f64,
        n: usize,
        results: *mut f64,
    );
    fn nk_vincenty_f32(
        a_lats: *const f32,
        a_lons: *const f32,
        b_lats: *const f32,
        b_lons: *const f32,
        n: usize,
        results: *mut f32,
    );
    fn nk_vincenty_f64(
        a_lats: *const f64,
        a_lons: *const f64,
        b_lats: *const f64,
        b_lons: *const f64,
        n: usize,
        results: *mut f64,
    );
}

/// Computes **great-circle distances** between geographic coordinates on Earth.
///
/// Uses the Haversine formula for spherical Earth approximation:
///
/// - `a = sin²(Δφ/2) + cos(φ₁) × cos(φ₂) × sin²(Δλ/2)`
/// - `c = 2 × atan2(√a, √(1−a))`
/// - `d = R × c`
///
/// Where φ = latitude, λ = longitude, R = Earth's radius (6335 km).
/// Inputs are in radians, outputs in meters.
pub trait Haversine: Sized {
    fn haversine(
        a_lat: &[Self],
        a_lon: &[Self],
        b_lat: &[Self],
        b_lon: &[Self],
        result: &mut [Self],
    ) -> Option<()>;
}

/// Computes **Vincenty geodesic distances** on the WGS84 ellipsoid.
///
/// Uses Vincenty's iterative formula for oblate spheroid geodesics:
///
/// 1. Reduced latitudes: `tan(U) = (1−f) × tan(φ)`
/// 2. Iterate until convergence: `λ → L + (1−C) × f × sin(α) × [σ + C × sin(σ) × ...]`
/// 3. Compute: `u² = cos²(α) × (a² − b²)/b²`
/// 4. Series coefficients A, B from u²
/// 5. Distance: `s = b × A × (σ − Δσ)`
///
/// Where a = equatorial radius, b = polar radius, f = flattening.
/// ~20× more accurate than Haversine for long distances.
/// Inputs are in radians, outputs in meters.
pub trait Vincenty: Sized {
    fn vincenty(
        a_lat: &[Self],
        a_lon: &[Self],
        b_lat: &[Self],
        b_lon: &[Self],
        result: &mut [Self],
    ) -> Option<()>;
}

/// Combined trait for all geospatial distance computations.
pub trait Geospatial: Haversine + Vincenty {}

impl Haversine for f64 {
    fn haversine(
        a_lat: &[Self],
        a_lon: &[Self],
        b_lat: &[Self],
        b_lon: &[Self],
        result: &mut [Self],
    ) -> Option<()> {
        let n = a_lat.len();
        if a_lon.len() != n || b_lat.len() != n || b_lon.len() != n || result.len() != n {
            return None;
        }
        unsafe {
            nk_haversine_f64(
                a_lat.as_ptr(),
                a_lon.as_ptr(),
                b_lat.as_ptr(),
                b_lon.as_ptr(),
                n,
                result.as_mut_ptr(),
            )
        };
        Some(())
    }
}

impl Vincenty for f64 {
    fn vincenty(
        a_lat: &[Self],
        a_lon: &[Self],
        b_lat: &[Self],
        b_lon: &[Self],
        result: &mut [Self],
    ) -> Option<()> {
        let n = a_lat.len();
        if a_lon.len() != n || b_lat.len() != n || b_lon.len() != n || result.len() != n {
            return None;
        }
        unsafe {
            nk_vincenty_f64(
                a_lat.as_ptr(),
                a_lon.as_ptr(),
                b_lat.as_ptr(),
                b_lon.as_ptr(),
                n,
                result.as_mut_ptr(),
            )
        };
        Some(())
    }
}

impl Geospatial for f64 {}

impl Haversine for f32 {
    fn haversine(
        a_lat: &[Self],
        a_lon: &[Self],
        b_lat: &[Self],
        b_lon: &[Self],
        result: &mut [Self],
    ) -> Option<()> {
        let n = a_lat.len();
        if a_lon.len() != n || b_lat.len() != n || b_lon.len() != n || result.len() != n {
            return None;
        }
        unsafe {
            nk_haversine_f32(
                a_lat.as_ptr(),
                a_lon.as_ptr(),
                b_lat.as_ptr(),
                b_lon.as_ptr(),
                n,
                result.as_mut_ptr(),
            )
        };
        Some(())
    }
}

impl Vincenty for f32 {
    fn vincenty(
        a_lat: &[Self],
        a_lon: &[Self],
        b_lat: &[Self],
        b_lon: &[Self],
        result: &mut [Self],
    ) -> Option<()> {
        let n = a_lat.len();
        if a_lon.len() != n || b_lat.len() != n || b_lon.len() != n || result.len() != n {
            return None;
        }
        unsafe {
            nk_vincenty_f32(
                a_lat.as_ptr(),
                a_lon.as_ptr(),
                b_lat.as_ptr(),
                b_lon.as_ptr(),
                n,
                result.as_mut_ptr(),
            )
        };
        Some(())
    }
}

impl Geospatial for f32 {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{assert_close, FloatLike, TestableType};

    fn check_haversine<T>(
        a_lat_deg: f64,
        a_lon_deg: f64,
        b_lat_deg: f64,
        b_lon_deg: f64,
        expected_meters: f64,
        tolerance: f64,
    ) where
        T: FloatLike + TestableType + Haversine,
    {
        let a_lat = [T::from_f32(a_lat_deg.to_radians() as f32)];
        let a_lon = [T::from_f32(a_lon_deg.to_radians() as f32)];
        let b_lat = [T::from_f32(b_lat_deg.to_radians() as f32)];
        let b_lon = [T::from_f32(b_lon_deg.to_radians() as f32)];
        let mut result = [T::zero()];
        T::haversine(&a_lat, &a_lon, &b_lat, &b_lon, &mut result).unwrap();
        assert_close(
            result[0].to_f64(),
            expected_meters,
            tolerance,
            0.0,
            &format!("haversine<{}>", core::any::type_name::<T>()),
        );
    }

    fn check_vincenty<T>(
        a_lat_deg: f64,
        a_lon_deg: f64,
        b_lat_deg: f64,
        b_lon_deg: f64,
        expected_meters: f64,
        tolerance: f64,
    ) where
        T: FloatLike + TestableType + Vincenty,
    {
        let a_lat = [T::from_f32(a_lat_deg.to_radians() as f32)];
        let a_lon = [T::from_f32(a_lon_deg.to_radians() as f32)];
        let b_lat = [T::from_f32(b_lat_deg.to_radians() as f32)];
        let b_lon = [T::from_f32(b_lon_deg.to_radians() as f32)];
        let mut result = [T::zero()];
        T::vincenty(&a_lat, &a_lon, &b_lat, &b_lon, &mut result).unwrap();
        assert_close(
            result[0].to_f64(),
            expected_meters,
            tolerance,
            0.0,
            &format!("vincenty<{}>", core::any::type_name::<T>()),
        );
    }

    #[test]
    fn geospatial() {
        // New York → Los Angeles
        // Haversine uses NK_EARTH_MEDIATORIAL_RADIUS (6,335,439m) → ~3,913,778m
        // Vincenty uses the WGS-84 ellipsoid → ~3,944,422m
        let hav_expected = 3_914_000.0;
        let vin_expected = 3_944_000.0;
        check_haversine::<f64>(
            40.7128,
            -74.0060,
            34.0522,
            -118.2437,
            hav_expected,
            20_000.0,
        );
        check_haversine::<f32>(
            40.7128,
            -74.0060,
            34.0522,
            -118.2437,
            hav_expected,
            50_000.0,
        );
        check_vincenty::<f64>(
            40.7128,
            -74.0060,
            34.0522,
            -118.2437,
            vin_expected,
            20_000.0,
        );
        check_vincenty::<f32>(
            40.7128,
            -74.0060,
            34.0522,
            -118.2437,
            vin_expected,
            50_000.0,
        );
    }
}

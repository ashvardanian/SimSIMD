#![allow(non_camel_case_types)]

extern "C" {
    fn cosine_i8(a: *const i8, b: *const i8, c: usize) -> f32;
    fn cosine_f32(a: *const f32, b: *const f32, c: usize) -> f32;

    fn inner_i8(a: *const i8, b: *const i8, c: usize) -> f32;
    fn inner_f32(a: *const f32, b: *const f32, c: usize) -> f32;

    fn sqeuclidean_i8(a: *const i8, b: *const i8, c: usize) -> f32;
    fn sqeuclidean_f32(a: *const f32, b: *const f32, c: usize) -> f32;
}

trait SimSIMD
where
    Self: Sized,
{
    fn cosine(a: &[Self], b: &[Self]) -> Option<f32>;
    fn inner(a: &[Self], b: &[Self]) -> Option<f32>;
    fn sqeuclidean(a: &[Self], b: &[Self]) -> Option<f32>;
}

impl SimSIMD for i8 {
    fn cosine(a: &[Self], b: &[Self]) -> Option<f32> {
        if a.len() != b.len() {
            return None;
        }

        let operation = unsafe { cosine_i8(a.as_ptr(), b.as_ptr(), a.len()) };

        Some(operation)
    }

    fn inner(a: &[Self], b: &[Self]) -> Option<f32> {
        if a.len() != b.len() {
            return None;
        }

        let operation = unsafe { inner_i8(a.as_ptr(), b.as_ptr(), a.len()) };

        Some(operation)
    }

    fn sqeuclidean(a: &[Self], b: &[Self]) -> Option<f32> {
        if a.len() != b.len() {
            return None;
        }

        let operation = unsafe { sqeuclidean_i8(a.as_ptr(), b.as_ptr(), a.len()) };

        Some(operation)
    }
}

impl SimSIMD for f32 {
    fn cosine(a: &[Self], b: &[Self]) -> Option<f32> {
        if a.len() != b.len() {
            return None;
        }

        let operation = unsafe { cosine_f32(a.as_ptr(), b.as_ptr(), a.len()) };

        Some(operation)
    }

    fn inner(a: &[Self], b: &[Self]) -> Option<f32> {
        if a.len() != b.len() {
            return None;
        }

        let operation = unsafe { inner_f32(a.as_ptr(), b.as_ptr(), a.len()) };

        Some(operation)
    }

    fn sqeuclidean(a: &[Self], b: &[Self]) -> Option<f32> {
        if a.len() != b.len() {
            return None;
        }

        let operation = unsafe { sqeuclidean_f32(a.as_ptr(), b.as_ptr(), a.len()) };

        Some(operation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    //
    #[test]
    fn test_cosine_i8() {
        let a = &[3, 97, 127];
        let b = &[3, 97, 127];

        if let Some(result) = SimSIMD::cosine(a, b) {
            assert_eq!(0.00012027938, result);
            println!("The result of cosine_i8 is {:.8}", result);
        }
    }

    #[test]
    fn test_cosine_f32() {
        let a = &[1.0, 2.0, 3.0];
        let b = &[1.0, 2.0, 3.0];

        if let Some(result) = SimSIMD::cosine(a, b) {
            assert_eq!(0.004930496, result);
            println!("The result of cosine_f32 is {:.8}", result);
        }
    }

    #[test]
    fn test_inner_i8() {
        let a = &[1, 2, 3];
        let b = &[4, 5, 6];

        if let Some(result) = SimSIMD::inner(a, b) {
            assert_eq!(0.029403687, result);
            println!("The result of inner_i8 is {:.8}", result);
        }
    }

    #[test]
    fn test_inner_f32() {
        let a = &[1.0, 2.0, 3.0];
        let b = &[4.0, 5.0, 6.0];

        if let Some(result) = SimSIMD::inner(a, b) {
            assert_eq!(-31.0, result);
            println!("The result of inner_f32 is {:.8}", result);
        }
    }

    #[test]
    fn test_sqeuclidean_i8() {
        let a = &[1, 2, 3];
        let b = &[4, 5, 6];

        if let Some(result) = SimSIMD::sqeuclidean(a, b) {
            assert_eq!(27.0, result);
            println!("The result of sqeuclidean_i8 is {:.8}", result);
        }
    }

    #[test]
    fn test_sqeuclidean_f32() {
        let a = &[1.0, 2.0, 3.0];
        let b = &[4.0, 5.0, 6.0];

        if let Some(result) = SimSIMD::sqeuclidean(a, b) {
            assert_eq!(27.0, result);
            println!("The result of sqeuclidean_f32 is {:.8}", result);
        }
    }
}

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

pub type Result<T> = core::result::Result<T, Error>;

pub type Error = Box<dyn std::error::Error>; // For early dev.

pub fn consine_i8(a: &[i8], b: &[i8]) -> Result<f32> {
    if a.len() != b.len() {
        return Err("both vectors must have the same length".into());
    }

    let operation: f32 = unsafe {
        cosine_i8_c(
            a.as_ptr() as *const simsimd_i8_t,
            b.as_ptr() as *const simsimd_i8_t,
            a.len() as simsimd_size_t,
        )
    };

    Ok(operation)
}

pub fn cosine_f32(a: &[f32], b: &[f32]) -> Result<f32> {
    if a.len() != b.len() {
        return Err("both vectors must have the same length".into());
    }

    let operation: f32 = unsafe { cosine_f32_c(a.as_ptr(), b.as_ptr(), a.len() as simsimd_size_t) };

    Ok(operation)
}

pub fn inner_i8(a: &[i8], b: &[i8]) -> Result<f32> {
    if a.len() != b.len() {
        return Err("both vectors must have the same length".into());
    }

    let operation: f32 = unsafe {
        inner_i8_c(
            a.as_ptr() as *const simsimd_i8_t,
            b.as_ptr() as *const simsimd_i8_t,
            a.len() as simsimd_size_t,
        )
    };

    Ok(operation)
}

pub fn inner_f32(a: &[f32], b: &[f32]) -> Result<f32> {
    if a.len() != b.len() {
        return Err("both vectors must have the same length".into());
    }

    let operation: f32 = unsafe { inner_f32_c(a.as_ptr(), b.as_ptr(), a.len() as simsimd_size_t) };

    Ok(operation)
}

pub fn sqeuclidean_i8(a: &[i8], b: &[i8]) -> Result<f32> {
    if a.len() != b.len() {
        return Err("both vectors must have the same length".into());
    }

    let operation: f32 = unsafe {
        sqeuclidean_i8_c(
            a.as_ptr() as *const simsimd_i8_t,
            b.as_ptr() as *const simsimd_i8_t,
            a.len() as simsimd_size_t,
        )
    };

    Ok(operation)
}

pub fn sqeuclidean_f32(a: &[f32], b: &[f32]) -> Result<f32> {
    if a.len() != b.len() {
        return Err("both vectors must have the same length".into());
    }

    let operation: f32 =
        unsafe { sqeuclidean_f32_c(a.as_ptr(), b.as_ptr(), a.len() as simsimd_size_t) };

    Ok(operation)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_i8() {
        let a = &[3, 97, 127];
        let b = &[3, 97, 127];

        match consine_i8(a, b) {
            Ok(result) => {
                assert_eq!(0.00012027938, result);
                println!("The result of cosine_i8 is {:.8}", result);
            }
            Err(e) => eprintln!("Could not perform cosine_i8.\n{}", e),
        }
    }

    #[test]
    fn test_cosine_f32() {
        let a = &[1.0, 2.0, 3.0];
        let b = &[1.0, 2.0, 3.0];

        match cosine_f32(a, b) {
            Ok(result) => {
                assert_eq!(0.004930496, result);
                println!("The result of cosine_f32 is {:.8}", result);
            }
            Err(e) => eprintln!("Could not perform cosine_f32.\n{}", e),
        }
    }

    #[test]
    fn test_inner_i8() {
        let a = &[1, 2, 3];
        let b = &[4, 5, 6];

        match inner_i8(a, b) {
            Ok(result) => {
                assert_eq!(0.029403687, result);
                println!("The result of inner_i8 is {:.8}", result);
            }
            Err(e) => eprintln!("Could not perform inner_i8.\n{}", e),
        }
    }

    #[test]
    fn test_inner_f32() {
        let a = &[1.0, 2.0, 3.0];
        let b = &[4.0, 5.0, 6.0];

        match inner_f32(a, b) {
            Ok(result) => {
                assert_eq!(-31.0, result);
                println!("The result of inner_f32 is {:.8}", result);
            }
            Err(e) => eprintln!("Could not perform inner_f32.\n{}", e),
        }
    }

    #[test]
    fn test_sqeuclidean_i8() {
        let a = &[1, 2, 3];
        let b = &[4, 5, 6];

        match sqeuclidean_i8(a, b) {
            Ok(result) => {
                assert_eq!(27.0, result);
                println!("The result of sqeuclidean_i8 is {:.8}", result);
            }
            Err(e) => eprintln!("Could not perform sqeuclidean_i8.\n{}", e),
        }
    }

    #[test]
    fn test_sqeuclidean_f32() {
        let a = &[1.0, 2.0, 3.0];
        let b = &[4.0, 5.0, 6.0];

        match sqeuclidean_f32(a, b) {
            Ok(result) => {
                assert_eq!(27.0, result);
                println!("The result of sqeuclidean_f32 is {:.8}", result);
            }
            Err(e) => eprintln!("Could not perform sqeuclidean_f32.\n{}", e),
        }
    }
}

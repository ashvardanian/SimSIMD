use rand::Rng;
use std::fmt::{Formatter, Display};
use rand::seq::index::sample;
use simsimd::sparse_dot_product;
//use half::bf16 as hbf16;
#[derive(Clone, Debug)]
struct SparseVector {
    indices: Vec<u16>,
    values: Vec<f32>,
}
impl SparseVector {
    fn from_dense(dense_vec: &[f32]) -> Self {
        if dense_vec.len() >= u16::MAX as usize {
            panic!("Dense vector is too large to convert to sparse vector");
        }
        let mut indices: Vec<u16> = Vec::new();
        let mut values = Vec::new();

        for (idx, &value) in dense_vec.iter().enumerate() {
            if value != 0.0 {
                indices.push(idx.try_into().unwrap());
                values.push(value);
            }
        }

        SparseVector { indices, values }
    }

    fn sparse_dot_product(&self, other: &SparseVector) -> (u16, f64) {
        let mut result = 0.0;
        let mut i = 0;
        let mut j = 0;
        let mut matches: u16 = 0;
        while i < self.indices.len() && j < other.indices.len() {
            if self.indices[i] == other.indices[j] {
                matches += 1;
                result += f64::from( self.values[i] * other.values[j]);
                i += 1;
                j += 1;
            } else if self.indices[i] < other.indices[j] {
                i += 1;
            } else {
                j += 1;
            }
        }

        (matches, result)
    }
}


impl Display for SparseVector {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "SparseVector {{ indices: {:?}, values: {:?} }}", self.indices, self.values)
    }
}

fn generate_intersecting_vectors(first_size: usize, second_size: usize, intersection_size: usize) -> (SparseVector, SparseVector) 
{
    let mut rng = rand::thread_rng();    
    let mut first_vector_indices: Vec<u16> = Vec::with_capacity(first_size);
    let mut second_vector_indices: Vec<u16> = Vec::with_capacity(second_size);
    let mut first_vector: Vec<f32> = Vec::with_capacity(first_size);
    let mut second_vector: Vec<f32> = Vec::with_capacity(second_size);
    let unique_first = first_size - intersection_size;
    let unique_second = second_size - intersection_size;
    assert!(intersection_size + unique_first + unique_second <= 65535, "Too many elements in the vectors");
    let total = intersection_size + (first_size - intersection_size) + (second_size - intersection_size);

    let unique_indices: Vec<u16> = sample(&mut rng, 65535, total).into_iter().map(|x| x as u16).collect();
    // assert!( unique_indices.len() == total, "unique_indices length is not correct: {}, expected {}", unique_indices.len(), total);

    first_vector_indices.extend(unique_indices.iter().take(intersection_size));
    second_vector_indices.extend(unique_indices.iter().take(intersection_size));
    first_vector_indices.extend(unique_indices.iter().skip(intersection_size).take(first_size - intersection_size));
    second_vector_indices.extend(unique_indices.iter().skip(intersection_size).skip(first_size - intersection_size).take(second_size - intersection_size));
    first_vector_indices.sort();
    second_vector_indices.sort();

    for _i in 0..first_size {
        let value: f32 = rng.gen();
        first_vector.push(value);
    }
    for _i in 0..second_size {
        let value: f32 = rng.gen();
        second_vector.push(value);
    }

    (SparseVector{indices: first_vector_indices, values: first_vector}, SparseVector{indices: second_vector_indices, values: second_vector})
   
}

pub fn main() {
        for first_len in [64, 128, 512, 1024, 2048] {
            for second_len in [8, 16, 32] {
                for intersection_ratio in [0.1, 0.5, 0.9] {
                        let intersection_size = (intersection_ratio * second_len as f32).ceil() as usize;
                        let (first_vector, second_vector) = generate_intersecting_vectors(first_len, second_len, intersection_size);
                        let mut total_ns: u128 = 0;
                        for _j in 0..10 {
                            let start = std::time::Instant::now();
                            let (_similar_items, _dot_product) = first_vector.sparse_dot_product(&second_vector);                        
                            let elapsed = start.elapsed();
                            assert!(_similar_items == intersection_size as u16, "similar items: {}, intersection_size: {}", _similar_items, intersection_size);
                            total_ns += elapsed.as_nanos();
                        }
                        println!("plain dot product:  {} against {}, avg elapsed_time ns: {}", first_len, second_len, total_ns/10);

                        
    
                }
            }
        }

        for first_len in [64, 128, 512, 1024, 2048] {
            for second_len in [8, 16, 32] {
                for intersection_ratio in [0.1, 0.5, 0.9] {
                        let intersection_size = (intersection_ratio * second_len as f32).ceil() as usize;
                        let (first_vector, second_vector) = generate_intersecting_vectors(first_len, second_len, intersection_size);
                        let mut total_ns: u128 = 0;
                        for _j in 0..10 {
                            let start = std::time::Instant::now();
                            let (neon_similar_items, _dot_product) = sparse_dot_product(
                                first_vector.indices.as_slice(),
                                second_vector.indices.as_slice(),
                                first_vector.values.as_slice(),
                                second_vector.values.as_slice(),
                            );
                            let elapsed = start.elapsed();
                            total_ns += elapsed.as_nanos();
                            assert!(neon_similar_items == intersection_size as f64, "similar items: {}, intersection_size: {}\n {}, \n second {}", neon_similar_items, intersection_size, first_vector, second_vector);
                        }
                        println!("NEON: {} vs {} avg elapsed_time ns: {}",first_len, second_len, total_ns/10);
        
    
                }
            }
        }
        
        
}
     
#![allow(unused)]
use rand::Rng;

pub(crate) fn generate_random_vector(dim: usize) -> Vec<f32> {
    (0..dim).map(|_| rand::thread_rng().gen()).collect()
}

pub(crate) fn cos_similarity_cpu(a: &[f32], b: &[f32]) -> Option<f32> {
    if a.len() != b.len() {
        return None;
    }

    let (dot_product, norm_a, norm_b) = a
        .iter()
        .zip(b)
        .map(|(a, b)| (a * b, a * a, b * b))
        .fold((0.0, 0.0, 0.0), |acc, x| {
            (acc.0 + x.0, acc.1 + x.1, acc.2 + x.2)
        });

    Some(1.0 - (dot_product / (norm_a.sqrt() * norm_b.sqrt())))
}

pub(crate) fn squared_euclidean_cpu(a: &[f32], b: &[f32]) -> Option<f32> {
    if a.len() != b.len() {
        return None;
    }

    Some(a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum())
}

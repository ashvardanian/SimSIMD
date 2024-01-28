#![allow(unused)]
use rand::Rng;

pub(crate) fn generate_random_vector(dim: usize) -> Vec<f32> {
    (0..dim).map(|_| rand::thread_rng().gen()).collect()
}

pub(crate) fn cosine_similarity_cpu(a: &[f32], b: &[f32]) -> Option<f32> {
    if a.len() != b.len() {
        return None;
    }

    let dot_product: f32 = a.iter().zip(b.iter()).map(|(a, b)| a * b).sum();
    let norm_a: f32 = a.iter().map(|a| a.powf(2.0)).sum();
    let norm_b: f32 = b.iter().map(|b| b.powf(2.0)).sum();

    Some(1.0 - (dot_product / (norm_a.sqrt() * norm_b.sqrt())))
}

pub(crate) fn squared_euclidean_cpu(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());

    a.iter()
        .zip(b.iter())
        .map(|(x, y)| ((*x) - (*y)) * ((*x) - (*y)))
        .fold(0.0, ::std::ops::Add::add)
}

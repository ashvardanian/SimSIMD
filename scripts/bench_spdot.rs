use criterion::{black_box, criterion_group, criterion_main, Criterion};
use std::collections::HashMap;
use std::collections::HashSet;
use std::fmt::Display;
use std::fmt::Formatter;
use std::fs::File;
use std::io::BufRead;
use std::io::BufReader;
use std::path::Path;
use regex::Regex;

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

#[derive(Debug)]
struct TfidfCalculator {
    tokens_index: HashMap<String, usize>,
    row_norms: Vec<SparseVector>,
    idfs: HashMap<usize, f32>,
    re: Regex,
}

impl TfidfCalculator {
    pub fn new() -> Self {
        TfidfCalculator {
            tokens_index: HashMap::new(),
            row_norms: Vec::new(),
            idfs: HashMap::new(),
            re: Regex::new(r"(?U)\b\w{2,}\b").unwrap(),
        }
    }

    fn tokenize(&self, text: &str) -> Vec<String> {
        let matches: Vec<String> = self
            .re
            .find_iter(text.to_lowercase().as_str())
            .map(|mat| mat.as_str().to_string())
            .collect();
        matches
    }

    pub fn calculate_document_tfidf_vectors(&mut self, documents: Vec<String>) {
        let mut unique_tokens = HashSet::new();
        let mut document_as_tokens: Vec<Vec<String>> = Vec::new();
        for document in &documents {
            let tokens = self.tokenize(document);
            for token in tokens.iter() {
                unique_tokens.insert(token.clone());
            }
            document_as_tokens.push(tokens);
        }
        let mut tokens: Vec<String> = unique_tokens.into_iter().collect();
        tokens.sort();

        for (idx, term) in tokens.iter().enumerate() {
            self.tokens_index.insert(term.clone(), idx);
        }

        let mut document_term_frequencies: Vec<Vec<f32>> =
            vec![vec![0.0; tokens.len()]; documents.len()];
        // I have count of terms in each document
        for (row_idx, document) in document_as_tokens.iter().enumerate() {
            for token in document.iter() {
                let term_id = self.tokens_index.get(token).unwrap();
                document_term_frequencies[row_idx][*term_id] += 1.0;
            }
        }
        // calculate idf of terms
        for term_id in 0..tokens.len() {
            let mut df = 0;
            for document in document_term_frequencies.iter() {
                if document[term_id] > 0.0 {
                    df += 1;
                }
            }
            let n_documents = document_term_frequencies.len();
            let present = df as f32;
            let idf = (((n_documents + 1) as f32) / (present + 1.0)).ln() + 1.0;
            self.idfs.insert(term_id, idf);
        }

        let mut tfidf_matrix: Vec<Vec<f32>> =
            vec![vec![0.0; tokens.len()]; document_term_frequencies.len()];

        for (row_idx, document) in document_term_frequencies.iter().enumerate() {
            for i in 0..tokens.len() {
                let term_id = i;
                let term_frequency = document[term_id];
                let tf = term_frequency as f32;
                let idf = self.idfs.get(&term_id).unwrap();
                if term_frequency == 0.0 {
                    tfidf_matrix[row_idx][term_id] = 0.0;
                } else {
                    tfidf_matrix[row_idx][term_id] = tf * idf;
                }
            }
        }
        self.row_norms = tfidf_matrix.iter().map(|row| {
          SparseVector::from_dense(row.as_slice())
        }).collect();

    }

    fn sparse_tfidf_representation(&self, row: &str) -> SparseVector {
        let mut row_vector: Vec<f32> = vec![0.0; self.tokens_index.len()];
        let tokens = self.tokenize(row);
        for token in tokens.iter() {
            let term_id = self.tokens_index.get(token).unwrap();
            let term_idf = self.idfs.get(term_id).unwrap();
            row_vector[*term_id] += *term_idf;
        }
        return SparseVector::from_dense(row_vector.as_slice());
    }

}

pub fn spdot_benchmark(c: &mut Criterion) {
    // Prepare your data and calculator here

    let path = Path::new("leipzig10000.txt");
    let file = File::open(path).unwrap();
    let reader = BufReader::new(file);

    let mut calculator = TfidfCalculator::new();

    let mut documents = Vec::new();
    for line in reader.lines() {
        documents.push(line.unwrap());
    }

    calculator.calculate_document_tfidf_vectors(documents);

    let sparse_vectors = &calculator.row_norms;

    let mut group = c.benchmark_group("Sparse Dot Product");

    // let query = "we went over to the whole world";
    // let query_vector = calculator.sparse_tfidf_representation(query);
    let mut largest = &sparse_vectors[0];
    for row_vector in sparse_vectors.iter() {
        if row_vector.indices.len() > largest.indices.len() {
            largest = row_vector;
        }
    }
    let query_vector = largest.clone();
    println!("Query Vector: {}", query_vector.indices.len());
    group.bench_function("SimSIMD spdot", |b| {
        b.iter(|| {
            let total_similarity: f64 = sparse_vectors
                .iter()
                .map(|row_vector| {
                     sparse_dot_product(
                        row_vector.indices.as_slice(),
                        query_vector.indices.as_slice(),
                        query_vector.values.as_slice(),
                        row_vector.values.as_slice(),
                    )
                }
            ).map(|(_similar_items, similarity_dot_product)| similarity_dot_product)
                .sum();
            black_box(total_similarity);
        })
    });

    group.bench_function("Rust plain spdot", |b| {
        b.iter(|| {
            let total_similarity: f64 = sparse_vectors
                .iter()
                .map(|row_vector| black_box(row_vector.sparse_dot_product(&query_vector)))
                .map(|(_similar_items, similarity_dot_product)| similarity_dot_product) 
                .sum();
            black_box(total_similarity)
        })
    });
}

criterion_group!(benches, spdot_benchmark);
criterion_main!(benches);

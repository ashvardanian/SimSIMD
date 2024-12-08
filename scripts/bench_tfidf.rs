
use criterion::{criterion_group, criterion_main, Criterion, black_box};
use std::collections::HashMap;
use std::collections::HashSet;
use std::fs::File;
use std::io::BufRead;
use std::io::BufReader;
// use std::io::{self};
use std::path::Path;
use regex::Regex;

use simsimd::SpatialSimilarity;

// mod native;

fn l2_normalize(v: &mut Vec<f64>) {
  let mut sum = 0.0;
  for i in v.iter() {
      sum += i.powi(2);
  }
  sum = sum.sqrt();
  for element in v.iter_mut() {
      *element = *element / sum;
  }
}


pub fn plain_cosine_similarity(vec1: &[f64], vec2: &[f64]) -> Option<f64> {
  // Check if vectors have the same length
  if vec1.len() != vec2.len() {
      return None;
  }

  // Calculate dot product
  let dot_product: f64 = vec1.iter()
      .zip(vec2.iter())
      .map(|(a, b)| a * b)
      .sum();

  // Calculate magnitudes
  let magnitude1: f64 = vec1.iter()
      .map(|x| x.powi(2))
      .sum::<f64>()
      .sqrt();

  let magnitude2: f64 = vec2.iter()
      .map(|x| x.powi(2))
      .sum::<f64>()
      .sqrt();

  // Prevent division by zero
  if magnitude1 == 0.0 || magnitude2 == 0.0 {
      return None;
  }

  // Calculate cosine similarity
  Some(dot_product / (magnitude1 * magnitude2))
}

#[derive(Debug)]
struct TfidfCalculator {
  tokens: Vec<String>,
  tokens_index: HashMap<String, usize>,
  row_norms: Vec<Vec<f64>>,
  documents: Vec<Vec<f64>>,
  idfs: HashMap<usize, f64>,
  re: Regex
}

impl TfidfCalculator {
  fn new() -> Self {
      TfidfCalculator {
          tokens: Vec::new(),
          tokens_index: HashMap::new(),
          row_norms: Vec::new(),
          documents: Vec::new(),
          idfs: HashMap::new(),
          re: Regex::new(r"(?U)\b\w{2,}\b").unwrap()
      }
  }

  fn tokenize(&self, text: &str) -> Vec<String> {
    let matches: Vec<String> = self.re.find_iter(text.to_lowercase().as_str())
        .map(|mat| mat.as_str().to_string())
        .collect();
    matches
  }

  fn process_documents(&mut self, documents: Vec<String>) {

    let mut unique_tokens = HashSet::new();
    let mut document_tokens: Vec<Vec<String>> = Vec::new();
    for document in &documents {
      let tokens = self.tokenize(document);
      for token in tokens.iter() {
        unique_tokens.insert(token.clone());
      }
      document_tokens.push(tokens);
    }
    self.tokens =  unique_tokens.into_iter().collect();
    self.tokens.sort();

    for (idx, term) in self.tokens.iter().enumerate() {
      self.tokens_index.insert(term.clone(), idx);
    }
    
    let mut matrix = vec![vec![0.0; self.tokens.len()]; documents.len()];
    // I have count of terms in each document
    for (row_idx, document) in document_tokens.iter().enumerate() {
      for token in document.iter() {
        let term_id = self.tokens_index.get(token).unwrap();
        matrix[row_idx][*term_id] += 1.0;
      }
    }
    self.documents = std::mem::take(&mut matrix);

    
  }

  fn calculate_idf(&mut self) {
    for term_id in 0..self.tokens.len() {
      let mut df = 0;
      for document in self.documents.iter() {
        if document[term_id] > 0.0 {
          df += 1;
        }
      }
      let n_documents = self.documents.len();
      let present = df as f64;
      let idf = ((( n_documents + 1 ) as f64) / (present + 1.0)).ln() + 1.0;
      self.idfs.insert(term_id, idf);
    }
  }

  fn calculate_tfidf(&mut self) {
      let mut tfidf_matrix = vec![vec![0.0; self.tokens.len()]; self.documents.len()];

      for (row_idx, document) in self.documents.iter().enumerate() {
        for i in 0..self.tokens.len() {
          let term_id = i;
          let term_frequency = document[term_id];
          let tf = term_frequency as f64;
          let idf = self.idfs.get(&term_id).unwrap();
          if term_frequency == 0.0 {
            tfidf_matrix[row_idx][term_id] = 0.0;
          } else {
            tfidf_matrix[row_idx][term_id] = tf * idf;
          }
        }
      }
      self.row_norms = std::mem::take(&mut tfidf_matrix);
  }

  fn normalize(&mut self) {
    for row in self.row_norms.iter_mut() {
      l2_normalize(row);
    }
  }

  fn tfidf_representation(&self, row: &str) -> Vec<f64> {
    let mut row_vector = vec![0.0; self.tokens.len()];
    let tokens = self.tokenize(row);
    for token in tokens.iter() {
      let term_id = self.tokens_index.get(token).unwrap();
      let term_idf = self.idfs.get(term_id).unwrap();
      row_vector[*term_id] += *term_idf;
    }
    l2_normalize(& mut row_vector);
    return row_vector;
  }
   

}



fn tfidfsimilarity_benchmark(c: &mut Criterion) {
    // Prepare your data and calculator here
    
    let path = Path::new("leipzig10000.txt");
    let file = File::open(path).unwrap();
    let reader = BufReader::new(file);


    let mut calculator = TfidfCalculator::new();
    
    let mut documents = Vec::new();
    for line in reader.lines() {
        documents.push(line.unwrap());
    }
    
    calculator.process_documents(documents);
    calculator.calculate_idf();
    calculator.calculate_tfidf();
    calculator.normalize();

    let mut group = c.benchmark_group("TF-IDF Similarity");
    
    let query = "we went over to the whole world";
    let query_vector = calculator.tfidf_representation(query);

    // Ensure the work cannot be optimized away
    group.bench_function("SimSIMD Similarity", |b| {
        b.iter(|| {
            let total_similarity: f64 = calculator.row_norms.iter()
                .map(|row_vector| {
                    black_box(
                        SpatialSimilarity::cosine(query_vector.as_ref(), row_vector.as_ref())
                            .unwrap_or(0.0)
                    )
                })
                .sum();
            
            // Force use of total_similarity to prevent optimization
            black_box(total_similarity);
        })
    });

    group.bench_function("Rust Procedural Similarity", |b| {
        b.iter(|| {
            let total_similarity: f64 = calculator.row_norms.iter()
                .map(|row_vector| {
                    black_box(
                        plain_cosine_similarity(query_vector.as_ref(), row_vector.as_ref())
                            .unwrap_or(0.0)
                    )
                })
                .sum();
            
            // Force use of total_similarity to prevent optimization
            black_box(total_similarity);
        })
    });

}

criterion_group!(benches, tfidfsimilarity_benchmark);
criterion_main!(benches);
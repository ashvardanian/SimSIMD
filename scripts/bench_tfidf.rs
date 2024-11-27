use std::collections::HashMap;
use std::collections::HashSet;
use std::fs::File;
use std::io::BufRead;
use std::io::BufReader;
use std::io::{self};
use std::path::Path;
use regex::Regex;

const LINES_PER_DOCUMENT: usize = 1000;


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

#[derive(Debug)]
struct TfidfCalculator {
    terms: Vec<String>,
    row_norms: Vec<Vec<f64>>,
    documents: Vec<Vec<usize>>,  // Store original documents
    idfs: HashMap<usize, f64>,
}

impl TfidfCalculator {
    fn new() -> Self {
        TfidfCalculator {
            terms: Vec::new(),
            row_norms: Vec::new(),
            documents: Vec::new(),
            idfs: HashMap::new(),
        }
    }

    fn tokenize(text: &str) -> Vec<String> {
      let re = Regex::new(r"(?U)\b\w{2,}\b").unwrap();
      let matches: Vec<String> = re.find_iter(text.to_lowercase().as_str())
          .map(|mat| mat.as_str().to_string())
          .collect();
      matches
    }

    fn process_documents(&mut self, documents: Vec<String>) {
      let mut unique_terms = HashSet::new();
      for document in &documents {
        let tokens = Self::tokenize(document);
        for token in tokens {       
          if !unique_terms.contains(&token) {
                 unique_terms.insert(token);
          }
        }
      }
      self.terms = unique_terms.into_iter().collect();
      self.terms.sort();
      for document in &documents {
        let tokens = Self::tokenize(document);
        let mut term_document_freq = HashMap::new();
        for token in tokens.iter() {
          let count = term_document_freq.entry(token).or_insert(0);
          *count += 1;
        }
        let mut row_counts = vec![0; self.terms.len()];
        for (idx, term) in self.terms.iter().enumerate() {
          let count = term_document_freq.get(term).unwrap_or(&0);
          row_counts[idx] = *count;
        }
        self.documents.push(row_counts);
      }
    }

    fn calculate_idf(&mut self) {
      for term_id in 0..self.terms.len() {
        let mut df = 0;
        for document in self.documents.iter() {
          if document[term_id] > 0 {
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
      let document = &self.documents[0];
        let mut tfidf_vector = vec![0.0; self.terms.len()];
        for i in 0..self.terms.len() {
          let term_id = i;
          let term_frequency = document[term_id];
          let tf = term_frequency as f64;
          let idf = self.idfs.get(&term_id).unwrap();
          if term_frequency == 0 {
            tfidf_vector[term_id] = 0.0;
          } else {
            tfidf_vector[term_id] = tf * idf;
          }
        }
        self.row_norms.push(tfidf_vector);
    }

    fn normalize(&mut self) {
      for row in self.row_norms.iter_mut() {
        l2_normalize(row);
      }
    }
     

}


fn main() -> io::Result<()> {

    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <filename>", args[0]);
        std::process::exit(1);
    }

    let path = Path::new(&args[1]);
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut calculator = TfidfCalculator::new();

    let mut documents = Vec::new();
    let mut current_document = String::new();
    let mut line_count = 0;

    for line in reader.lines() {
        let line = line?;
        current_document.push_str(&line);
        current_document.push('\n');
        line_count += 1;
        if line_count == LINES_PER_DOCUMENT {
            documents.push(std::mem::take(&mut current_document));
            line_count = 0;
            println!("current document size after take {}", current_document.len());

        }
    }

    if !current_document.is_empty() {
        documents.push(std::mem::take(&mut current_document));
    }
    calculator.process_documents(documents);
    calculator.calculate_idf();
    calculator.calculate_tfidf();
    calculator.normalize();

    
    Ok(())
  

}
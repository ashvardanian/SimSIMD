use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::Path;

const LINES_PER_DOCUMENT: usize = 1000;

#[derive(Debug)]
struct TfidfCalculator {
    document_frequency: HashMap<String, usize>,
    term_frequencies: Vec<HashMap<String, usize>>,
    documents: Vec<String>,  // Store original documents
    n_documents: usize,
}

impl TfidfCalculator {
    fn new() -> Self {
        TfidfCalculator {
            document_frequency: HashMap::new(),
            term_frequencies: Vec::new(),
            documents: Vec::new(),
            n_documents: 0,
        }
    }

    fn tokenize(text: &str) -> Vec<String> {
        text.split_whitespace()
            .map(|word| word.to_lowercase()
                .chars()
                .filter(|c| c.is_alphanumeric())
                .collect::<String>())
            .filter(|word| !word.is_empty())
            .collect()
    }

    fn process_document(&mut self, text: &str) {
        let tokens = Self::tokenize(text);
        let mut term_freq = HashMap::new();
        let mut seen_terms = HashMap::new();

        for token in tokens {
            *term_freq.entry(token.clone()).or_insert(0) += 1;
            seen_terms.insert(token, true);
        }

        for term in seen_terms.keys() {
            *self.document_frequency.entry(term.clone()).or_insert(0) += 1;
        }

        self.term_frequencies.push(term_freq);
        self.documents.push(text.to_string());
        self.n_documents += 1;
    }

    fn calculate_tfidf(&self, term_freq: &HashMap<String, usize>) -> HashMap<String, f64> {
        let mut tfidf = HashMap::new();
        
        for (term, freq) in term_freq {
            if let Some(&df) = self.document_frequency.get(term) {
                let tf = *freq as f64;
                let idf = (self.n_documents as f64 / df as f64).ln();
                tfidf.insert(term.clone(), tf * idf);
            }
        }
        
        tfidf
    }

    fn get_document_tfidf_vectors(&self) -> Vec<HashMap<String, f64>> {
        self.term_frequencies
            .iter()
            .map(|tf| self.calculate_tfidf(tf))
            .collect()
    }

    fn calculate_query_tfidf(&self, query: &str) -> HashMap<String, f64> {
        let tokens = Self::tokenize(query);
        let mut term_freq = HashMap::new();
        
        for token in tokens {
            *term_freq.entry(token).or_insert(0) += 1;
        }
        
        self.calculate_tfidf(&term_freq)
    }

    fn cosine_similarity(v1: &HashMap<String, f64>, v2: &HashMap<String, f64>) -> f64 {
        let mut dot_product = 0.0;
        let mut norm1 = 0.0;
        let mut norm2 = 0.0;

        // Calculate dot product and norms
        for (term, score1) in v1 {
            norm1 += score1 * score1;
            if let Some(score2) = v2.get(term) {
                dot_product += score1 * score2;
            }
        }

        for (_term, score2) in v2 {
            norm2 += score2 * score2;
        }

        // Return cosine similarity
        if norm1 == 0.0 || norm2 == 0.0 {
            0.0
        } else {
            dot_product / (norm1.sqrt() * norm2.sqrt())
        }
    }

    fn find_matching_documents(&self, query: &str, top_n: usize) -> Vec<(usize, f64)> {
        let query_tfidf = self.calculate_query_tfidf(query);
        let doc_tfidfs = self.get_document_tfidf_vectors();
        
        let mut similarities: Vec<(usize, f64)> = doc_tfidfs
            .iter()
            .enumerate()
            .map(|(idx, doc_tfidf)| {
                (idx, Self::cosine_similarity(&query_tfidf, doc_tfidf))
            })
            .collect();
        
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        // Return top N results
        similarities.into_iter().take(top_n).collect()
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

    // Process the file in chunks of LINES_PER_DOCUMENT lines
    let mut current_document = String::new();
    let mut line_count = 0;

    for line in reader.lines() {
        let line = line?;
        current_document.push_str(&line);
        current_document.push('\n');
        line_count += 1;

        if line_count == LINES_PER_DOCUMENT {
            calculator.process_document(&current_document);
            current_document.clear();
            line_count = 0;
        }
    }

    if !current_document.is_empty() {
        calculator.process_document(&current_document);
    }
}